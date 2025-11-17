"""ProcessPlanner builds/manages a hierarchical plan for ProductAgents."""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


class ProcessPlanner:
    """
    Small helper object owned by a ProductAgent to orchestrate phase/task plans.

    - build_high_level()           -> create/reset the phase skeleton
    - build_high_level_from_llm()  -> convenience wrapper that queries the LLM first
    - save()/load()                -> persist or restore the plan tree
    - mark_phase_done(pid)         -> set a phase status to "done"
    - first_pending_phase()        -> id of the next runnable phase
    - expand helpers               -> update_node(), next_pending_task(), etc.
    """

    def __init__(self, product_agent, resource_agents: Iterable[Any]):
        self.product_agent = product_agent
        self.resource_agents = list(resource_agents)
        self.logger = product_agent.logger
        self.nodes: List[Dict[str, Any]] = []  # Ordered list of phase nodes
        self.phase_to_node: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------ #
    # Phase construction
    # ------------------------------------------------------------------ #
    async def build_high_level_from_llm(self, requirement_text: str) -> str:
        """
        Ask the LLM to propose a phase list, then build the internal structure.

        Returns the same text as build_high_level() for convenience.
        """
        raw = await self._extract_phases(requirement_text)
        return self.build_high_level(raw)

    def build_high_level(self, phase_source: Any) -> str:
        """
        Accept raw phase data (list/dict/str) and build the ordered phase skeleton.
        Raises ValueError when the source cannot be normalized into phase names.
        """
        phases = self.parse_phases(phase_source)
        if not phases:
            raise ValueError("No phases could be parsed from the model output.")

        self.nodes.clear()
        self.phase_to_node.clear()

        for idx, phase in enumerate(phases):
            normalized = str(phase).strip()
            node = self._make_phase(normalized, [phases[idx - 1]] if idx else [])
            self.nodes.append(node)
            self.phase_to_node[normalized] = node

        self.logger.info("High-level Plan:")
        self.logger.info(json.dumps(self.nodes, indent=2))
        return "High-level Plan is created."

    def _make_phase(self, pid: str, after: List[str]):
        return {
            "id": pid,
            "function_owner_agent": getattr(self.product_agent, "agent_name", "product"),
            "function": None,
            "params": {},
            "status": "pending",
            "after": after,
            "children": [],
        }

    # ------------------------------------------------------------------ #
    # Phase parsing helpers
    # ------------------------------------------------------------------ #
    def parse_phases(self, raw: Any) -> List[str]:
        """
        Normalize many potential LLM outputs into a clean list of phase ids.
        Accepts:
        - ["design", "assembly"]
        - {"phases": ["design", "assembly"]}
        - "```json [ ... ] ```" or object form
        - "design, assembly" fallback
        """
        if isinstance(raw, list):
            return [str(x).strip() for x in raw if str(x).strip()]

        if isinstance(raw, dict):
            if "phases" in raw and isinstance(raw["phases"], list):
                return [str(x).strip() for x in raw["phases"] if str(x).strip()]
            for value in raw.values():
                if isinstance(value, list) and all(
                    isinstance(entry, (str, int)) for entry in value
                ):
                    return [str(x).strip() for x in value if str(x).strip()]
            raise ValueError("Dict returned but no list of phases found.")

        if isinstance(raw, str):
            s = raw.strip()
            s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
            s = re.sub(r"\s*```$", "", s)

            match = re.search(r"(\[[\s\S]*?\]|\{[\s\S]*?\})", s)
            if match:
                payload = match.group(1)
                try:
                    data = json.loads(payload)
                except json.JSONDecodeError:
                    try:
                        data = json.loads(payload.replace("'", '"'))
                    except json.JSONDecodeError as exc:
                        raise ValueError(
                            f"Could not parse JSON payload: {exc}"
                        ) from exc

                if isinstance(data, list):
                    return [str(x).strip() for x in data if str(x).strip()]
                if isinstance(data, dict):
                    if "phases" in data and isinstance(data["phases"], list):
                        return [
                            str(x).strip() for x in data["phases"] if str(x).strip()
                        ]
                    for value in data.values():
                        if isinstance(value, list) and all(
                            isinstance(entry, (str, int)) for entry in value
                        ):
                            return [str(x).strip() for x in value if str(x).strip()]
                    raise ValueError("JSON object found but no phases list within.")
                raise ValueError(f"Unexpected JSON type: {type(data).__name__}")

            parts = [p.strip() for p in re.split(r"[,\n]+", s) if p.strip()]
            return parts

        raise ValueError(f"Unsupported type for phases: {type(raw).__name__}")

    # ------------------------------------------------------------------ #
    # Status helpers
    # ------------------------------------------------------------------ #
    def mark_phase_done(self, phase_id: str):
        if phase_id in self.phase_to_node:
            self.phase_to_node[phase_id]["status"] = "done"

    def first_pending_phase(self) -> Optional[str]:
        for node in self.nodes:
            if node["status"] == "pending":
                prereq = node["after"][0] if node["after"] else None
                if not prereq or self.phase_to_node.get(prereq, {}).get(
                    "status"
                ) == "done":
                    return node["id"]
        return None

    def first_active_or_pending_phase(self) -> Optional[str]:
        """
        Return the first phase that is runnable:
        - pending with completed prerequisites
        - already in_progress / awaiting_approval
        """
        for node in self.nodes:
            if node["status"] in ("pending", "in_progress", "awaiting_approval"):
                prereq = node["after"][0] if node["after"] else None
                if not prereq or self.phase_to_node.get(prereq, {}).get(
                    "status"
                ) == "done":
                    return node["id"]
        return None

    def mark_task_done(self, phase_id: str, task_id: str, *, plan_path: str):
        phase = self.phase_to_node[phase_id]
        for child in phase["children"]:
            if child["id"] == task_id:
                child["status"] = "completed"
                break

        if not self.next_pending_task(phase_id, plan_path=plan_path):
            self.mark_phase_done(phase_id)

    def _phase_done(self, pid: str) -> bool:
        node = self.phase_to_node.get(pid)
        return bool(node and node["status"] == "done")

    # ------------------------------------------------------------------ #
    # Persistence / updates
    # ------------------------------------------------------------------ #
    def update_node(
        self,
        phase_id: str,
        task_id: Optional[str] = None,
        **changes,
    ) -> bool:
        """
        Update either the phase itself (when task_id is None) or the child task.
        Returns True when an update occurred, False otherwise.
        """
        node = self.phase_to_node.get(phase_id)
        if not node:
            return False

        target = node if task_id is None else next(
            (child for child in node["children"] if child["id"] == task_id),
            None,
        )
        if not target:
            return False

        target.update(changes)
        return True

    def save(self, path: str | os.PathLike) -> None:
        path = Path(path)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(self.nodes, indent=2), encoding="utf-8")

        for _ in range(20):
            try:
                tmp_path.replace(path)
                return
            except PermissionError:
                time.sleep(0.05)

        self.logger.error(
            "Could not replace plan file (in use): %s (left tmp: %s)",
            path,
            tmp_path,
        )

    def load(self, path: str):
        nodes = json.loads(Path(path).read_text())
        self._rebuild_maps(nodes)
        return self.nodes

    # ------------------------------------------------------------------ #
    # Task selection helpers
    # ------------------------------------------------------------------ #
    def next_pending_task(
        self,
        phase_id: str,
        *,
        plan_path: str,
        accept: Tuple[str, ...] = ("pending",),
    ) -> Optional[Dict[str, Any]]:
        """
        Return the first runnable child task within *phase_id* whose status appears in
        *accept* and whose dependencies are fulfilled (either completed tasks or phases).
        """
        nodes: List[Dict[str, Any]] = json.loads(Path(plan_path).read_text())
        phase_map = {node["id"]: node for node in nodes}
        done_tasks = {
            child["id"]
            for node in nodes
            for child in node["children"]
            if child["status"] == "completed"
        }
        done_phases = {node["id"] for node in nodes if node["status"] == "completed"}

        phase = phase_map[phase_id]
        for child in phase["children"]:
            if child["status"] not in accept:
                continue
            deps = child.get("after", [])
            if all(dep in done_tasks or dep in done_phases for dep in deps):
                return child
        return None

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    async def _extract_phases(self, req_txt: str) -> List[str]:
        """
        Ask the ProductAgent's LLM for a JSON array of phase names.
        Returns [] when parsing fails (caller must handle error messaging).
        """
        prompt = (
            "Return ONLY a JSON array of phase names, no Markdown, no prose.\n"
            'Example: ["design","preprocessing","manufacturing","assembly"]\n\n'
            f"Requirements:\n{req_txt}"
        )
        raw = await self.product_agent.ask_llm(
            prompt, with_functions=False, temperature=0.0
        )
        text = raw if isinstance(raw, str) else json.dumps(raw)
        cleaned = re.sub(r"^```.*?\n|\n```$", "", text.strip(), flags=re.S)

        try:
            data = json.loads(cleaned)
        except Exception as exc:
            self.logger.error(
                "Phase parsing failed: %s | raw LLM output: %r", exc, raw
            )
            return []

        if isinstance(data, list):
            return [str(x).strip() for x in data if str(x).strip()]
        if isinstance(data, dict) and "phases" in data:
            return [str(x).strip() for x in data["phases"] if str(x).strip()]
        self.logger.error("Unexpected LLM payload while extracting phases: %r", data)
        return []

    def _rebuild_maps(self, nodes: List[Dict[str, Any]]) -> None:
        self.nodes = nodes
        self.phase_to_node = {node["id"]: node for node in nodes}
