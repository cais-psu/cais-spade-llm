from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from prompts import build_task_expansion_prompt, build_requirement_parse_prompt, build_replan_prompt

class ProcessPlanner:
    """
    1. NL → structured requirement nodes   (build_high_level)
    2. requirement nodes → executable task DAG   (expand_requirements_to_tasks)
    """

    def __init__(self, product_agent, resource_agents: Iterable[Any]):
        self.product_agent = product_agent
        self.resource_agents = list(resource_agents)
        self.logger = product_agent.logger
        self.nodes: List[Dict[str, Any]] = []
        self.phase_to_node: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------ #
    # 1. NL → High-level requirements
    # ------------------------------------------------------------------ #
    async def build_high_level(self, requirement_text: str) -> str:
        """
        Parse natural-language manufacturing requirements into internal requirement nodes.

        No normalization is applied:
        - phase, process_type, product, context are kept exactly as returned by the LLM.
        """
        # Reset state
        self.nodes.clear()
        self.phase_to_node.clear()

        try:
            structured = await self._llm_parse_requirements(requirement_text)
        except Exception as exc:
            self.logger.exception("[Planner] LLM requirement parsing failed: %s", exc)
            structured = []

        for idx, req in enumerate(structured, start=1):
            node_id = f"REQ_{idx}"

            # Use values exactly as returned by _llm_parse_requirements
            raw_text = req.get("raw_text", "")
            phase = req.get("phase")              # e.g. "ASSEMBLY", "printing", or None
            process_type = req.get("process_type")  # e.g. "PICK_PLACE", "FDM_PRINT", or None
            product = req.get("product")          # e.g. "SG", "MCP", or None
            context = req.get("context") or {}    # e.g. {"origin": "prusa-mk4-2", "destination": "assembly board"}

            node = {
                "id": node_id,
                "type": "requirement",
                "raw_text": raw_text,
                "phase": phase,
                "process_type": process_type,
                "product": product,
                "context": context,
            }

            self.nodes.append(node)

        msg = f"[Planner] Parsed {len(structured)} requirement(s) via LLM."
        self.logger.info(msg)
        return msg


    async def _llm_parse_requirements(self, requirement_text: str) -> list[dict[str, Any]]:
        tools_catalog = getattr(self.product_agent, "tools_catalog", [])

        prompt = build_requirement_parse_prompt(requirement_text, tools_catalog)

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
            self.logger.error(
                "[Planner] LLM did not return valid JSON: %s\nRaw: %s", exc, raw
            )
            raise

        reqs = parsed.get("requirements", [])
        cleaned: list[dict[str, Any]] = []

        for r in reqs:
            if not isinstance(r, dict):
                continue
            ctx = r.get("context") or {}
            if not isinstance(ctx, dict):
                ctx = {}
            cleaned.append(
                {
                    "raw_text": r.get("raw_text", ""),
                    "phase": r.get("phase"),
                    "process_type": r.get("process_type"),
                    "product": r.get("product"),
                    "context": ctx,
                }
            )

        return cleaned

    # ------------------------------------------------------------------ #
    # 2. REQUIREMENT → LLM TASK EXPANSION (replaces hard-coded version)
    # ------------------------------------------------------------------ #
    async def expand_requirements_to_tasks(self, safety_text: str = "") -> None:
        """
        Replace requirement nodes with task nodes produced by LLM.

        The LLM returns a DIRECTED ACYCLIC GRAPH (DAG) of tasks where
        dependencies are expressed via `predecessors` and `successors`.
        """
        req_nodes = [n for n in self.nodes if n.get("type") == "requirement"]
        if not req_nodes:
            self.logger.warning("[Planner] No requirement nodes to expand.")
            self.nodes = []
            return

        # Prepare payload for LLM
        req_payload = [
            {
                "id": n["id"],
                "product": n.get("product"),
                "context": n.get("context"),
                "raw_text": n.get("raw_text"),
            }
            for n in req_nodes
        ]

        tools_catalog = getattr(self.product_agent, "tools_catalog", [])

        # capabilities overview from ProductAgent
        caps_overview = ""
        if hasattr(self.product_agent, "_static_caps_overview"):
            try:
                caps_overview = self.product_agent._static_caps_overview()
            except Exception:
                caps_overview = ""

        resource_infos = [
            {
                "jid": str(getattr(ra, "jid", "")),
                "static_capabilities": getattr(ra, "static_capabilities", {}),
            }
            for ra in self.resource_agents
        ]

        # Build LLM prompt (from prompts.py)
        prompt = build_task_expansion_prompt(
            requirements=req_payload,
            tools_catalog=tools_catalog,
            resource_infos=resource_infos,
            caps_overview=caps_overview,
            safety_text=safety_text
        )

        raw = await self.product_agent.ask_llm(
            prompt=prompt,
            with_functions=False,
            temperature=0.0,
        )

        parsed = json.loads(raw)
        task_specs = parsed.get("tasks", [])

        new_nodes: list[dict[str, Any]] = []

        # Create task nodes from LLM output
        for t in task_specs:
            params = t.get("params") or {}
            # Ensure product_jid is always present
            params["product_jid"] = str(self.product_agent.jid)

            node = {
                "id": t.get("id"),
                "type": "task",
                "requirement_id": t.get("requirement_id"),
                "function_name": t.get("function_name"),
                "params": params,
                "resource_jid": t.get("resource_jid"),
                "sequence_index": t.get("sequence_index", 0),
                "status": "pending",
                # Preserve graph structure from LLM if provided
                "predecessors": t.get("predecessors", []),
                "successors": t.get("successors", []),
            }
            new_nodes.append(node)

        # OPTIONAL: fill in trivial per-requirement chains
        # for tasks that have no predecessors/successors at all.
        # This keeps things backwards-compatible if the LLM omits edges.
        by_req: Dict[str, List[Dict[str, Any]]] = {}
        for n in new_nodes:
            rid = n.get("requirement_id")
            by_req.setdefault(rid, []).append(n)

        for rid, seq in by_req.items():
            # Check if ALL tasks under this requirement have completely empty edges
            all_edges_empty = all(
                not n.get("predecessors") and not n.get("successors") for n in seq
            )
            if not all_edges_empty:
                # LLM already defined some graph structure for this requirement;
                # do NOT overwrite it.
                continue

            # Otherwise, fall back to a simple linear chain by sequence_index
            seq.sort(key=lambda n: n.get("sequence_index", 0))
            for prev, nxt in zip(seq, seq[1:]):
                prev["successors"].append(nxt["id"])
                nxt["predecessors"].append(prev["id"])

        self.nodes = new_nodes
        self.logger.info(
            "[Planner] Expanded to %d LLM-generated task node(s).",
            len(self.nodes),
        )


    # ------------------------------------------------------------------ #
    # 3. REPLAN WHEN SAFETY VIOLATION OCCURS
    # ------------------------------------------------------------------ #
    async def replan_with_feedback(self, violations: list[dict]) -> None:
        self.logger.info("[Planner] Triggering LLM Re-planning with safety feedback...")
        
        # 1. Gather all tasks
        failed_nodes = [n for n in self.nodes if n.get("type") == "task"]

        # 2. Extract the "Conflict Set" - IDs of tasks involved in violations
        conflict_task_ids = set()
        for v in violations:
            # Add tasks from the witness trace
            conflict_task_ids.update(v.get("witness_trace", []))
            
            # Optionally add relevant tasks if provided by the validator
            rel_tasks = v.get("relevant_tasks", [])
            for rt in rel_tasks:
                if "id" in rt:
                    conflict_task_ids.add(rt["id"])

        # 3. Mark the nodes in the payload so the LLM knows what to focus on
        plan_payload = []
        for node in failed_nodes:
            node_copy = node.copy()
            if node_copy["id"] in conflict_task_ids:
                node_copy["_FOCUS_HERE"] = " <<< THIS TASK IS INVOLVED IN A VIOLATION"
            plan_payload.append(node_copy)

        # --- DEFINITIONS RESTORED HERE ---
        tools_catalog = getattr(self.product_agent, "tools_catalog", [])

        # Get capability overview string
        caps_overview = ""
        if hasattr(self.product_agent, "_static_caps_overview"):
            try:
                caps_overview = self.product_agent._static_caps_overview()
            except Exception:
                caps_overview = ""

        # Get resource agent info
        resource_infos = [
            {
                "jid": str(getattr(ra, "jid", "")),
                "static_capabilities": getattr(ra, "static_capabilities", {}),
            }
            for ra in self.resource_agents
        ]
        # ---------------------------------

        # 4. Build Prompt
        prompt = build_replan_prompt(
            failed_plan_nodes=plan_payload, 
            violations=violations,
            tools_catalog=tools_catalog,
            resource_infos=resource_infos,
            caps_overview=caps_overview,
        )

        raw = await self.product_agent.ask_llm(
            prompt=prompt,
            with_functions=False,
            temperature=0.0, # Keep temp low for deterministic fixes
        )

        try:
            parsed = json.loads(raw)
            modified_tasks = parsed.get("tasks", [])

            if not modified_tasks:
                self.logger.warning("[Planner] LLM returned no modified tasks. Plan remains unchanged.")
                return
            
            # --- MERGE STRATEGY (Supports Modification, Insertion, Deletion) ---
            node_map = {n["id"]: n for n in self.nodes}

            for t in modified_tasks:
                tid = t.get("id")
                if not tid: 
                    continue

                # CASE A: DELETION
                if t.get("delete") is True:
                    if tid in node_map:
                        self.logger.info(f"[Planner] DELETING task {tid}: {t.get('change_reason')}")
                        del node_map[tid]
                        # Cleanup: Remove edges pointing to the deleted node
                        for other in node_map.values():
                            if tid in other.get("predecessors", []):
                                other["predecessors"].remove(tid)
                            if tid in other.get("successors", []):
                                other["successors"].remove(tid)
                    continue

                # CASE B: MODIFICATION / INSERTION
                
                # Prepare params: use existing if modifying, empty if new
                params = t.get("params") or {}
                if tid in node_map and not t.get("params"):
                    params = node_map[tid].get("params", {})
                
                # Ensure product_jid is always present
                params["product_jid"] = str(self.product_agent.jid)

                # Initialize new node if it doesn't exist
                if tid not in node_map:
                    node_map[tid] = {
                        "id": tid,
                        "type": "task",
                        "status": "pending",
                        "predecessors": [],
                        "successors": []
                    }

                target = node_map[tid]

                # Update fields ONLY if they are present in the LLM output
                if "function_name" in t: target["function_name"] = t["function_name"]
                if "params" in t: target["params"] = params
                if "resource_jid" in t: target["resource_jid"] = t["resource_jid"]
                if "sequence_index" in t: target["sequence_index"] = t["sequence_index"]
                
                # Overwrite edges if provided (LLM is authoritative on structure changes)
                if "predecessors" in t: target["predecessors"] = t["predecessors"]
                if "successors" in t: target["successors"] = t["successors"]

                if "change_reason" in t:
                    target["change_reason"] = t["change_reason"]
                    self.logger.info(f"[Planner] Applied fix to {tid}: {t['change_reason']}")

                # Reset status so it runs again
                target["status"] = "pending" 

            self.nodes = list(node_map.values())
            
            # 5. Consistency Check (Auto-repair bidirectional links)
            self._ensure_graph_consistency()

            self.logger.info(
                "[Planner] Re-planning successful. Merged %d modifications.",
                len(modified_tasks),
            )

            if hasattr(self.product_agent, "plan_path"):
                self.save(self.product_agent.plan_path)

        except json.JSONDecodeError as exc:
            self.logger.error(f"[Planner] LLM Re-planning returned invalid JSON: {exc}")


    # ------------------------------------------------------------------ #
    # Scheduling helpers
    # ------------------------------------------------------------------ #
    def _ensure_graph_consistency(self) -> None:
        """
        Helper to ensure that if A lists B as a predecessor, 
        B lists A as a successor (and vice versa).
        This fixes 'one-sided' edits from the LLM.
        """
        node_map = {n["id"]: n for n in self.nodes}
        
        for nid, node in node_map.items():
            # 1. Sync Predecessors -> Successors
            # If 'node' thinks 'pid' is a predecessor, make sure 'pid' knows 'node' is a successor.
            preds = node.get("predecessors", [])
            for pid in preds:
                if pid in node_map:
                    p_node = node_map[pid]
                    if nid not in p_node.get("successors", []):
                        p_node.setdefault("successors", []).append(nid)
            
            # 2. Sync Successors -> Predecessors
            # If 'node' thinks 'sid' is a successor, make sure 'sid' knows 'node' is a predecessor.
            succs = node.get("successors", [])
            for sid in succs:
                if sid in node_map:
                    s_node = node_map[sid]
                    if nid not in s_node.get("predecessors", []):
                        s_node.setdefault("predecessors", []).append(nid)

    def _find_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        for n in self.nodes:
            if n.get("id") == node_id:
                return n
        return None

    def next_ready_task(self) -> Optional[Dict[str, Any]]:
        for node in self.nodes:
            if node.get("type") != "task":
                continue
            if node.get("status") != "pending":
                continue

            preds = node.get("predecessors", [])
            if not preds:
                return node

            if all(self._find_node(pid).get("status") == "completed" for pid in preds):
                return node

        return None

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save(self, path: Path | str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {"nodes": self.nodes}
        with p.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        self.logger.info(f"[Planner] Saved plan to {p.resolve()}")

    def load(self, path: Path | str) -> None:
        p = Path(path)
        if not p.exists():
            self.logger.warning(f"[Planner] Plan file missing: {p}")
            return
        with p.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        self.nodes = payload.get("nodes", [])
        self.logger.info(f"[Planner] Loaded plan from {p.resolve()}")
