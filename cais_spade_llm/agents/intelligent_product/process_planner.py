"""ProcessPlanner builds/manages a hierarchical plan for ProductAgents."""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from function_analyzer import FunctionAnalyzer


class ProcessPlanner:
    """
    Small helper object owned by a ProductAgent to orchestrate phase/task plans.

    - build_plan()/build_high_level() -> create/reset the full DAG (phases + tasks)
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
    async def build_high_level(self, requirement_text: str) -> str:
        """
        Backwards-compatible wrapper that now builds the full DAG plan.
        Prefer calling build_plan() directly.
        """
        return await self.build_plan(requirement_text)

    async def build_plan(self, requirement_text: str) -> str:
        """
        Generate a directed acyclic graph (DAG) of manufacturing/assembly steps
        using the product requirements, resource tool catalogue, and constraints.
        """
        plan_payload = await self._generate_plan(requirement_text)
        nodes = self.parse_plan(plan_payload)
        if not nodes:
            raise ValueError("LLM did not return any plan nodes.")

        self._rebuild_maps(nodes)
        self.logger.info("Process Plan:")
        self.logger.info(json.dumps(self.nodes, indent=2))
        return "Process plan is created."

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
    # Plan generation helpers
    # ------------------------------------------------------------------ #
    async def _generate_plan(self, requirement_text: str) -> Any:
        context = self._build_plan_context(requirement_text)
        prompt = self._build_plan_prompt(context)
        try:
            raw = await self.product_agent.ask_llm(
                prompt,
                with_functions=False,
                temperature=0.0,
            )
        except Exception as exc:
            self.logger.exception("LLM call failed while generating plan: %s", exc)
            raise
        return self._coerce_plan_payload(raw)

    def _build_plan_context(self, requirement_text: str) -> Dict[str, Any]:
        return {
            "product_name": getattr(self.product_agent, "agent_name", "product"),
            "product_instructions": getattr(self.product_agent, "instructions", None),
            "requirements": requirement_text.strip(),
            "constraints": self._collect_constraints(),
            "resources": self._resource_summaries(),
            "available_tools": self._collect_tools(),
        }

    def _build_plan_prompt(self, context: Dict[str, Any]) -> str:
        schema_hint = {
            "plan": [
                {
                    "id": "string (phase or macro-step id)",
                    "after": ["dependency ids"],
                    "function_owner_agent": "resource responsible for the phase",
                    "function": "optional macro function",
                    "params": {"key": "value"},
                    "children": [
                        {
                            "id": "task id",
                            "function_owner_agent": "resource executing the task",
                            "function": "tool/function name",
                            "params": {"key": "value"},
                            "after": ["phase or task ids this child depends on"],
                        }
                    ],
                }
            ]
        }
        instructions = (
            "You are a manufacturing process planner. Use the JSON context to build a"
            " directed acyclic graph (DAG) that transforms the product requirements"
            " into executable steps using ONLY the available tools."
            " Honor all ordering constraints and capabilities. Every phase or child task"
            " must reference a valid function_owner_agent and use tools from the catalogue."
            " Return strictly valid JSON matching the schema sketch below. No markdown."
        )
        payload = json.dumps(
            {
                "context": context,
                "output_schema": schema_hint,
            },
            indent=2,
        )
        return f"{instructions}\n{payload}"

    def _target_resources(self) -> List[Any]:
        if not self.resource_agents:
            return []
        targets = {
            str(jid).lower()
            for jid in getattr(self.product_agent, "resource_jids", [])
            if jid
        }
        if not targets:
            return list(self.resource_agents)

        selected = []
        for agent in self.resource_agents:
            agent_jid = str(getattr(agent, "jid", "")).lower()
            if agent_jid in targets:
                selected.append(agent)
        return selected

    def _collect_tools(self) -> List[Dict[str, Any]]:
        tools: List[Dict[str, Any]] = []
        for agent in self._target_resources():
            owner = getattr(agent, "agent_name", getattr(agent, "name", agent.__class__.__name__))
            executables = getattr(agent, "executables", {}) or {}
            for fn_name, fn in executables.items():
                entry: Dict[str, Any] = {
                    "function_owner_agent": owner,
                    "function": fn_name,
                }
                meta = FunctionAnalyzer._extract_yaml_frontmatter(fn)
                if meta:
                    entry.update({k: v for k, v in meta.items() if v is not None})
                doc = (fn.__doc__ or "").strip()
                if doc:
                    first_line = doc.splitlines()[0].strip()
                    if first_line:
                        entry.setdefault("description", first_line)
                tools.append(entry)
        return tools

    def _resource_summaries(self) -> List[Dict[str, Any]]:
        summaries: List[Dict[str, Any]] = []
        for agent in self._target_resources():
            summary = {
                "name": getattr(agent, "agent_name", getattr(agent, "name", agent.__class__.__name__)),
                "jid": str(getattr(agent, "jid", "")),
                "instructions": getattr(agent, "instructions", None),
                "static_capabilities": getattr(agent, "static_capabilities", None),
            }
            summaries.append({k: v for k, v in summary.items() if v})
        return summaries

    def _collect_constraints(self) -> List[str]:
        constraints: List[str] = []
        paths: List[Path] = [
            Path("cais_spade_llm/specification/cca/safety_requirements.txt"),
        ]

        spec_file = getattr(self.product_agent, "product_specification_file", None)
        if spec_file:
            spec_path = Path(spec_file)
            candidate = spec_path.with_name(f"{spec_path.stem}_constraints.txt")
            paths.append(candidate)

        for path in paths:
            txt = self._read_optional_text(path)
            if txt:
                constraints.append(txt)
        return constraints

    def _read_optional_text(self, path: Path) -> Optional[str]:
        try:
            if path.exists():
                data = path.read_text(encoding="utf-8").strip()
                return data or None
        except Exception as exc:
            self.logger.warning("Failed reading constraint file %s: %s", path, exc)
        return None

    def _coerce_plan_payload(self, raw: Any) -> Any:
        if isinstance(raw, (list, dict)):
            return raw
        if not isinstance(raw, str):
            raise ValueError(f"Unsupported LLM response type: {type(raw).__name__}")

        cleaned = re.sub(r"^```.*?\n|\n```$", "", raw.strip(), flags=re.S)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as exc:
            self.logger.error("Plan parsing failed: %s | raw LLM output: %r", exc, raw)
            raise

    # ------------------------------------------------------------------ #
    # Plan parsing helpers
    # ------------------------------------------------------------------ #
    def parse_plan(self, raw: Any) -> List[Dict[str, Any]]:
        items = self._extract_plan_items(raw)
        nodes: List[Dict[str, Any]] = []
        for idx, entry in enumerate(items):
            normalized = self._normalize_node(entry, idx)
            if normalized:
                nodes.append(normalized)
        return nodes

    def _extract_plan_items(self, raw: Any) -> List[Dict[str, Any]]:
        if isinstance(raw, list):
            return raw
        if isinstance(raw, dict):
            for key in ("plan", "phases", "nodes", "steps"):
                block = raw.get(key)
                if isinstance(block, list):
                    return block
            return [raw]
        raise ValueError(f"Unsupported plan payload type: {type(raw).__name__}")

    def _normalize_node(self, entry: Any, index: int) -> Optional[Dict[str, Any]]:
        if not isinstance(entry, dict):
            return None

        node_id = str(
            entry.get("id")
            or entry.get("name")
            or entry.get("phase_id")
            or f"phase_{index + 1}"
        )
        after = self._normalize_after(entry.get("after") or entry.get("depends_on"))
        node = self._make_phase(node_id, after)

        owner = entry.get("function_owner_agent") or entry.get("owner")
        if owner:
            node["function_owner_agent"] = owner

        node["function"] = entry.get("function")
        node["params"] = self._normalize_params(entry.get("params") or entry.get("parameters"))

        children = entry.get("children") or entry.get("tasks") or []
        normalized_children: List[Dict[str, Any]] = []
        for idx, child in enumerate(children):
            fallback_id = f"{node_id}_task_{idx + 1}"
            normalized = self._normalize_task(child, node["function_owner_agent"], fallback_id)
            if normalized:
                normalized_children.append(normalized)
        node["children"] = normalized_children
        return node

    def _normalize_task(
        self,
        entry: Any,
        default_owner: str,
        fallback_id: str,
    ) -> Optional[Dict[str, Any]]:
        if not isinstance(entry, dict):
            return None

        task_id = str(entry.get("id") or entry.get("name") or fallback_id)
        task_owner = entry.get("function_owner_agent") or entry.get("owner") or default_owner

        return {
            "id": task_id,
            "function_owner_agent": task_owner,
            "function": entry.get("function"),
            "params": self._normalize_params(entry.get("params") or entry.get("parameters")),
            "status": "pending",
            "after": self._normalize_after(entry.get("after") or entry.get("depends_on")),
        }

    def _normalize_after(self, deps: Any) -> List[str]:
        if not deps:
            return []
        if isinstance(deps, (str, int)):
            deps = [deps]
        normalized = []
        for dep in deps:
            dep_str = str(dep).strip()
            if dep_str:
                normalized.append(dep_str)
        return normalized

    @staticmethod
    def _normalize_params(value: Any) -> Dict[str, Any]:
        return value if isinstance(value, dict) else {}

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
    def _rebuild_maps(self, nodes: List[Dict[str, Any]]) -> None:
        self.nodes = nodes
        self.phase_to_node = {node["id"]: node for node in nodes}

