from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


class ProcessPlanner:
    """
    ProcessPlanner converts NL to structured requirement nodes
    and later expands them into DAG task nodes.
    """

    def __init__(self, product_agent, resource_agents: Iterable[Any]):
        self.product_agent = product_agent
        self.resource_agents = list(resource_agents)
        self.logger = product_agent.logger
        self.nodes: List[Dict[str, Any]] = []
        self.phase_to_node: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------ #
    # NL → Assembly Requirements via LLM (minimal)
    # ------------------------------------------------------------------ #

    async def build_high_level(self, requirement_text: str) -> str:
        """
        Use an LLM to parse natural-language *assembly* requirements into
        minimal requirement nodes.

        Node fields:
          id, type='requirement', raw_text,
          phase='assembly',
          product,
          context={origin, destination}
        """
        self.nodes.clear()
        self.phase_to_node.clear()

        try:
            structured = await self._llm_parse_assembly_requirements(requirement_text)
        except Exception as exc:
            self.logger.exception("[Planner] LLM requirement parsing failed: %s", exc)
            structured = []

        for idx, req in enumerate(structured, start=1):
            node_id = f"REQ_{idx}"

            product_id = self._normalize_id(req.get("product"))
            ctx = req.get("context") or {}
            origin_id = self._normalize_id(ctx.get("origin"))
            dest_id = self._normalize_id(ctx.get("destination"))

            node = {
                "id": node_id,
                "type": "requirement",
                "raw_text": req.get("raw_text", ""),
                "phase": "assembly",
                "product": product_id,
                "context": {
                    "origin": origin_id,
                    "destination": dest_id,
                },
            }

            self.nodes.append(node)

        msg = f"[Planner] Parsed {len(structured)} assembly requirement(s) via LLM."
        self.logger.info(msg)
        return msg

    async def _llm_parse_assembly_requirements(
        self, requirement_text: str
    ) -> List[Dict[str, Any]]:
        """
        LLM JSON schema expected:

        {
          "requirements": [
            {
              "raw_text": "...",
              "product": "SG",
              "context": {
                "origin": "prusa-mk4-2",
                "destination": "assembly station"
              }
            }
          ]
        }
        """

        prompt = (
            "You convert natural-language *assembly* instructions into a minimal structured form.\n"
            "Respond with valid JSON only, no extra commentary.\n\n"
            "Schema:\n"
            "{\n"
            '  "requirements": [\n'
            "    {\n"
            '      "raw_text": string,\n'
            '      "product": string | null,\n'
            '      "context": {\n'
            '         "origin": string | null,\n'
            '         "destination": string | null\n'
            "      }\n"
            "    }\n"
            "  ]\n"
            "}\n\n"
            "Rules:\n"
            "- All requirements refer to assembly.\n"
            '- Use short, domain-relevant tokens for product and locations (e.g. \"SG\", \"prusa-mk4-2\").\n'
            "- If you are unsure about a field, set it to null; do not invent details.\n\n"
            "Now convert the following text:\n\n"
            f"{requirement_text}\n"
        )

        # IMPORTANT: use with_functions=False so ask_llm returns a plain string
        raw = await self.product_agent.ask_llm(
            prompt=prompt,
            with_functions=False,
            temperature=0.0,
        )

        if isinstance(raw, dict):
            self.logger.error("[Planner] ask_llm returned a dict, expected JSON string.")
            raise RuntimeError("ask_llm returned dict; expected JSON string.")

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            self.logger.error("[Planner] LLM did not return valid JSON: %s\nRaw: %s", exc, raw)
            raise

        reqs = parsed.get("requirements", [])
        cleaned: List[Dict[str, Any]] = []

        for r in reqs:
            if not isinstance(r, dict):
                continue
            ctx = r.get("context") or {}
            if not isinstance(ctx, dict):
                ctx = {}
            cleaned.append(
                {
                    "raw_text": r.get("raw_text", ""),
                    "product": r.get("product"),
                    "context": {
                        "origin": ctx.get("origin"),
                        "destination": ctx.get("destination"),
                    },
                }
            )

        return cleaned

    @staticmethod
    def _normalize_id(text: Optional[str]) -> Optional[str]:
        if not text:
            return None
        t = text.upper().strip()
        t = re.sub(r"[^A-Z0-9]+", "_", t)
        return t.strip("_") or None

    # ------------------------------------------------------------------ #
    # Requirement → primitive task chain (DAG expansion)
    # ------------------------------------------------------------------ #
    async def expand_requirements_to_tasks(self) -> None:
        """
        Expand each requirement node into a linear chain of task nodes.
        After this, self.nodes will contain only task nodes.
        """
        new_nodes: List[Dict[str, Any]] = []

        # self.nodes currently holds only requirement nodes (built by build_high_level)
        requirement_nodes = [n for n in self.nodes if n.get("type") == "requirement"]

        for node in requirement_nodes:
            req_id = node["id"]
            product = node.get("product")
            ctx = node.get("context") or {}
            origin = ctx.get("origin")
            dest = ctx.get("destination")

            def make_task(suffix: str, function_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
                tid = f"{req_id}_{suffix}"
                return {
                    "id": tid,
                    "type": "task",
                    "requirement_id": req_id,
                    "function_name": function_name,
                    "params": params,
                    "status": "pending",
                    "predecessors": [],
                    "successors": [],
                }

            t1 = make_task("T1", "move_to_pick_location", {
                "origin_resource_location": origin,
            })
            t2 = make_task("T2", "pick_part", {
                "part_name": product,
                "origin_resource_location": origin,
            })
            t3 = make_task("T3", "move_loaded_to_destination", {
                "destination_location": dest,
            })
            t4 = make_task("T4", "place_part", {
                "destination_location": dest,
            })
            chain = [t1, t2, t3, t4]

            # Wire the chain
            for prev, nxt in zip(chain, chain[1:]):
                prev["successors"].append(nxt["id"])
                nxt["predecessors"].append(prev["id"])

            new_nodes.extend(chain)

        self.nodes = new_nodes
        self.logger.info(
            "[Planner] Expanded requirements into task chains (tasks only, total nodes: %d)",
            len(self.nodes),
        )

    # ------------------------------------------------------------------ #
    # Scheduling helpers
    # ------------------------------------------------------------------ #

    def _find_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        for n in self.nodes:
            if n.get("id") == node_id:
                return n
        return None

    def next_ready_task(self) -> Optional[Dict[str, Any]]:
        """
        Return one task node that is:
          - type == 'task'
          - status == 'pending'
          - all predecessor nodes have status == 'completed'
        or None if no such task exists.
        """
        for node in self.nodes:
            if node.get("type") != "task":
                continue
            if node.get("status") != "pending":
                continue

            preds = node.get("predecessors", [])
            if not preds:
                return node  # no dependencies

            all_done = True
            for pid in preds:
                pred_node = self._find_node(pid)
                if not pred_node or pred_node.get("status") != "completed":
                    all_done = False
                    break

            if all_done:
                return node

        return None

    # ------------------------------------------------------------------ #
    # Persistence helpers
    # ------------------------------------------------------------------ #

    def save(self, path: Path | str) -> None:
        """
        Persist the current planner state (for now, just nodes) as JSON.
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "nodes": self.nodes,
        }

        with p.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        self.logger.info("[Planner] Saved plan to %s", p.resolve())

    def load(self, path: Path | str) -> None:
        """
        Load planner state from a JSON file created by save().
        """
        p = Path(path)
        if not p.exists():
            self.logger.warning("[Planner] Plan file does not exist: %s", p)
            return

        with p.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        self.nodes = payload.get("nodes", [])
        self.phase_to_node = {}

        self.logger.info("[Planner] Loaded plan from %s", p.resolve())

