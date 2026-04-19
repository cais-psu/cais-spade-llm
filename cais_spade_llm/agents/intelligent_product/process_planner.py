"""Process planner that turns requirements into tasks and plan automata."""

from __future__ import annotations

import asyncio
import json
import re
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Set
from uuid import uuid4
from cais_spade_llm.prompts import (
    build_requirement_parse_prompt,
    build_replan_prompt,
    build_task_expansion_prompt,
)
from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge import LlmBridgeReplannerMixin
from collections import defaultdict, deque

class ProcessPlanner(LlmBridgeReplannerMixin):
    """
    1. NL → structured requirement nodes   (build_high_level)
    2. requirement nodes → executable task DAG   (expand_requirements_to_tasks)
    """

    def __init__(self, product_agent, resource_agents: Iterable[Any]):
        """Initialize planner state with agent references and empty node graphs."""
        self.product_agent = product_agent
        self.resource_agents = list(resource_agents)
        self.logger = product_agent.logger
        self.nodes: List[Dict[str, Any]] = []
        self.phase_to_node: Dict[str, Dict[str, Any]] = {}
        self.global_fsa: Optional[Dict[str, Any]] = None
        self.last_bridge_debug: Dict[str, Any] = {}

    @staticmethod
    def _primitive_bridge_macro_tasks(proposal: dict[str, Any]) -> list[dict[str, Any]]:
        raw_tasks = proposal.get("macro_tasks")
        if isinstance(raw_tasks, list) and raw_tasks:
            return [dict(task) for task in raw_tasks if isinstance(task, dict)]
        if proposal.get("primitive_steps"):
            return [dict(proposal)]
        return []

    # ------------------------------------------------------------------ #
    # 1. NL → High-level requirements
    # ------------------------------------------------------------------ #
    async def build_high_level(
        self,
        requirement_text: str,
        *,
        refinement_feedback: str = "",
        previous_preview_requirements: list[dict[str, Any]] | None = None,
    ) -> str:
        """
        Parse natural-language manufacturing requirements into internal requirement nodes.

        No normalization is applied:
        - phase, process_type, product, context are kept exactly as returned by the LLM.
        """
        # Reset state
        self.nodes.clear()
        self.phase_to_node.clear()

        try:
            structured = await self._llm_parse_requirements(
                requirement_text,
                refinement_feedback=refinement_feedback,
                previous_preview_requirements=previous_preview_requirements,
            )
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


    async def _llm_parse_requirements(
        self,
        requirement_text: str,
        *,
        refinement_feedback: str = "",
        previous_preview_requirements: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """Call the LLM to parse requirements into a structured list."""
        tools_catalog = getattr(self.product_agent, "tools_catalog", [])

        prompt = build_requirement_parse_prompt(
            requirement_text,
            tools_catalog,
            refinement_feedback=refinement_feedback,
            previous_preview_requirements=previous_preview_requirements,
        )

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
    async def expand_requirements_to_tasks(
        self,
        safety_text: str = "",
        *,
        refinement_feedback: str = "",
        previous_preview_requirements: list[dict[str, Any]] | None = None,
        previous_preview_tasks: list[dict[str, Any]] | None = None,
    ) -> None:
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
            safety_text=safety_text,
            refinement_feedback=refinement_feedback,
            previous_preview_requirements=previous_preview_requirements,
            previous_preview_tasks=previous_preview_tasks,
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
    # 3. REPLAN WHEN SAFETY VIOLATION OR ONLINE FAILURE OCCURS
    # ------------------------------------------------------------------ #
    @staticmethod
    def _build_des_replan_result(
        *,
        plan_changed: bool = False,
        used_llm_bridge: bool = False,
        human_required: bool = False,
        awaiting_bridge_approval: bool = False,
        awaiting_bridge_generation: bool = False,
        des_recovery_missing: bool = False,
        message: str = "",
        bridge_summary: list[str] | None = None,
        bridge_proposal: dict[str, Any] | None = None,
        bridge_debug: dict[str, Any] | None = None,
        prepared_bridge_request: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            "plan_changed": bool(plan_changed),
            "used_llm_bridge": bool(used_llm_bridge),
            "human_required": bool(human_required),
            "awaiting_bridge_approval": bool(awaiting_bridge_approval),
            "awaiting_bridge_generation": bool(awaiting_bridge_generation),
            "des_recovery_missing": bool(des_recovery_missing),
            "message": str(message).strip(),
            "bridge_summary": list(bridge_summary or []),
            "bridge_proposal": deepcopy(bridge_proposal) if isinstance(bridge_proposal, dict) else None,
            "bridge_debug": deepcopy(bridge_debug) if isinstance(bridge_debug, dict) else None,
            "prepared_bridge_request": (
                deepcopy(prepared_bridge_request)
                if isinstance(prepared_bridge_request, dict)
                else None
            ),
        }

    def _set_last_bridge_debug(self, payload: dict[str, Any] | None) -> None:
        self.last_bridge_debug = deepcopy(payload) if isinstance(payload, dict) else {}

    def get_last_bridge_debug(self) -> dict[str, Any]:
        return deepcopy(self.last_bridge_debug)

    @staticmethod
    def _tool_signature(row: dict[str, Any]) -> str:
        payload = {
            "function_owner_agent": str(row.get("function_owner_agent") or "").strip(),
            "function": str(row.get("function") or "").strip(),
            "in_state": str(row.get("in_state") or "").strip(),
            "out_state": str(row.get("out_state") or "").strip(),
            "part_in_state": str(row.get("part_in_state") or "").strip(),
            "location_type": str(
                (row.get("context_mapping") or {}).get("location_type") or ""
            ).strip(),
            "location_param": str(
                (row.get("context_mapping") or {}).get("location_param") or ""
            ).strip(),
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    def _resource_infos(self) -> list[dict[str, Any]]:
        return [
            {
                "jid": str(getattr(ra, "jid", "")),
                "static_capabilities": getattr(ra, "static_capabilities", {}),
            }
            for ra in self.resource_agents
        ]

    def _resource_by_jid(self, target_jid: str) -> Any | None:
        for ra in self.resource_agents:
            if str(getattr(ra, "jid", "")) == str(target_jid):
                return ra
        return None

    @staticmethod
    def _resource_short_name(value: str) -> str:
        token = str(value or "").strip()
        if "@" in token:
            token = token.split("@", 1)[0]
        return token.lower()

    def _tool_row_for_task(
        self,
        *,
        resource_jid: str,
        function_name: str,
        tools_catalog: list[dict[str, Any]],
    ) -> dict[str, Any]:
        target_resource = self._resource_short_name(resource_jid)
        fallback: dict[str, Any] | None = None
        for row in tools_catalog or []:
            if not isinstance(row, dict):
                continue
            if str(row.get("function", "")).strip() != str(function_name or "").strip():
                continue
            owner = self._resource_short_name(str(row.get("function_owner_agent", "")).strip())
            if owner and owner == target_resource:
                return row
            if fallback is None:
                fallback = row
        return fallback or {}

    def _pending_resource_tasks(
        self,
        resource_jid: str,
        *,
        ignored_task_ids: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        ignored = {
            str(task_id).strip()
            for task_id in (ignored_task_ids or set())
            if str(task_id).strip()
        }
        pending: list[dict[str, Any]] = []
        for node in self.nodes:
            if node.get("type") != "task":
                continue
            node_id = str(node.get("id", "")).strip()
            if node_id and node_id in ignored:
                continue
            if str(node.get("resource_jid", "")).strip() != str(resource_jid).strip():
                continue
            status = str(node.get("status", "")).strip()
            if status in ("pending", "running", "accepted", "dispatched"):
                pending.append(node)

        def sort_key(n: dict[str, Any]) -> tuple[int, str]:
            si = n.get("sequence_index")
            return (10**9 if si is None else int(si), str(n.get("id", "")))

        pending.sort(key=sort_key)
        return pending

    def _reconcile_search_state_for_task(
        self,
        *,
        search_state: dict[str, Any],
        task: dict[str, Any],
        tool_row: dict[str, Any],
    ) -> tuple[dict[str, Any], list[str]]:
        reconciled = {
            "resource_state": search_state.get("resource_state", "idle"),
            "current_part": search_state.get("current_part"),
            "current_location": search_state.get("current_location"),
            "part_states": dict(search_state.get("part_states", {})),
            "part_locations": dict(search_state.get("part_locations", {})),
        }
        notes: list[str] = []
        params = dict(task.get("params") or {})
        task_part = str(params.get("part_name") or "").strip() or None
        current_part = str(reconciled.get("current_part") or "").strip() or None
        ctx_map = tool_row.get("context_mapping") or {}
        loc_param = str(ctx_map.get("location_param") or "").strip()
        loc_type = str(ctx_map.get("location_type") or "").strip()
        part_in_state = str(tool_row.get("part_in_state") or "").strip()

        if task_part and task_part not in reconciled["part_states"]:
            reconciled["part_states"][task_part] = part_in_state or "unknown"
            notes.append(f"initialized part_state[{task_part}]")

        if loc_type == "part_location" and task_part and loc_param:
            task_location = params.get(loc_param)
            if task_location and reconciled["part_locations"].get(task_part) != task_location:
                reconciled["part_locations"][task_part] = task_location
                notes.append(f"aligned part_location[{task_part}]")

        if part_in_state:
            if current_part is None and task_part:
                reconciled["current_part"] = task_part
                current_part = task_part
                notes.append(f"inferred current_part={task_part}")
            if current_part and reconciled["part_states"].get(current_part) != part_in_state:
                reconciled["part_states"][current_part] = part_in_state
                notes.append(f"aligned part_state[{current_part}]={part_in_state}")

        if not reconciled.get("current_location"):
            if loc_type == "current_location" and loc_param and params.get(loc_param):
                reconciled["current_location"] = params.get(loc_param)
                notes.append(f"inferred current_location={params.get(loc_param)}")
            elif loc_type == "part_location" and task_part:
                part_location = reconciled["part_locations"].get(task_part)
                if part_location:
                    reconciled["current_location"] = part_location
                    notes.append(f"inferred current_location={part_location}")

        return reconciled, notes

    @staticmethod
    def _entry_task_ids_from_violations(violations: list[dict[str, Any]]) -> list[str]:
        task_ids: list[str] = []
        seen: set[str] = set()
        for violation in violations or []:
            if not isinstance(violation, dict):
                continue
            for key in ("failed_task_id", "task_id"):
                candidate = str(violation.get(key) or "").strip()
                if candidate and candidate not in seen:
                    seen.add(candidate)
                    task_ids.append(candidate)
            blocked = violation.get("blocked_task_ids")
            if isinstance(blocked, (list, tuple, set)):
                for value in blocked:
                    candidate = str(value or "").strip()
                    if candidate and candidate not in seen:
                        seen.add(candidate)
                        task_ids.append(candidate)
        return task_ids

    @staticmethod
    def _coerce_xyz_pose(payload: Any) -> dict[str, float] | None:
        if not isinstance(payload, dict):
            return None
        if not {"x", "y", "z"} <= set(payload.keys()):
            return None
        try:
            pose = {
                "x": float(payload["x"]),
                "y": float(payload["y"]),
                "z": float(payload["z"]),
            }
            for key in ("qx", "qy", "qz", "qw"):
                if key in payload:
                    pose[key] = float(payload[key])
            return pose
        except (TypeError, ValueError):
            return None

    def _identify_stuck_resource(
        self,
        violations: list[dict[str, Any]],
        resource_states: dict[str, dict[str, Any]],
    ) -> str:
        failed_task_ids: list[str] = []
        for violation in violations:
            if not isinstance(violation, dict):
                continue
            candidate = violation.get("resource_jid") or violation.get("failed_resource_jid")
            if candidate:
                return str(candidate)
            for key in ("failed_task_id", "task_id"):
                candidate_task_id = str(violation.get(key) or "").strip()
                if candidate_task_id:
                    failed_task_ids.append(candidate_task_id)
                    mapped_resource_jid = self._resource_jid_for_task_id(candidate_task_id)
                    if mapped_resource_jid:
                        return mapped_resource_jid
            blocked_task_ids = violation.get("blocked_task_ids")
            if isinstance(blocked_task_ids, (list, tuple, set)):
                for raw_task_id in blocked_task_ids:
                    candidate_task_id = str(raw_task_id or "").strip()
                    if not candidate_task_id:
                        continue
                    mapped_resource_jid = self._resource_jid_for_task_id(candidate_task_id)
                    if mapped_resource_jid:
                        return mapped_resource_jid
            safety_ctx = violation.get("safety_ctx") or {}
            targets = safety_ctx.get("obligation_targets") or []
            if isinstance(targets, list):
                for target in targets:
                    if not isinstance(target, dict):
                        continue
                    target_jid = str(target.get("resource_jid", "")).strip()
                    if target_jid:
                        return target_jid
        failed_like_states = {"failed", "error", "fault", "faulted", "aborted"}
        for ra in self.resource_agents:
            ra_jid_candidate = str(ra.jid)
            rs = resource_states.get(ra_jid_candidate, {})
            current_state = str(rs.get("current_state", "") or "").strip().lower()
            if current_state in failed_like_states:
                return ra_jid_candidate
        for ra in self.resource_agents:
            ra_jid_candidate = str(ra.jid)
            rs = resource_states.get(ra_jid_candidate, {})
            if rs.get("current_state", "idle") != "idle":
                return ra_jid_candidate
        return str(self.resource_agents[0].jid) if self.resource_agents else "unknown"

    def _resource_jid_for_task_id(self, task_id: str) -> str:
        target_task_id = str(task_id or "").strip()
        if not target_task_id:
            return ""
        for node in self.nodes:
            if not isinstance(node, dict):
                continue
            if str(node.get("id") or "").strip() != target_task_id:
                continue
            if str(node.get("type") or "").strip() != "task":
                continue
            resource_jid = str(node.get("resource_jid") or "").strip()
            if resource_jid:
                return resource_jid
        return ""

    def _function_name_for_task_id(self, task_id: str) -> str:
        target_task_id = str(task_id or "").strip()
        if not target_task_id:
            return ""
        for node in self.nodes:
            if not isinstance(node, dict):
                continue
            if str(node.get("id") or "").strip() != target_task_id:
                continue
            if str(node.get("type") or "").strip() != "task":
                continue
            function_name = str(node.get("function_name") or "").strip()
            if function_name:
                return function_name
        return ""

    @staticmethod
    def _extract_resource_states(system_coordination_state: dict[str, Any]) -> dict[str, dict[str, Any]]:
        if not isinstance(system_coordination_state, dict):
            return {}

        for key in ("resource_states", "resources", "robot_states", "robots"):
            payload = system_coordination_state.get(key)
            if isinstance(payload, dict):
                return payload
        return {}

    @staticmethod
    def _default_resource_state(tools_catalog: list[dict[str, Any]]) -> str:
        all_out_states = {t.get("out_state") for t in tools_catalog if t.get("out_state")}
        root_states = [
            t.get("in_state") for t in tools_catalog
            if t.get("in_state") and t.get("in_state") not in all_out_states
        ]
        return str(root_states[0] or "idle") if root_states else "idle"

    def _build_resource_search_state(
        self,
        *,
        resource_jid: str,
        resource_states: dict[str, dict[str, Any]],
        default_resource_state: str,
        part_states: dict[str, Any],
        part_locations: dict[str, Any],
    ) -> dict[str, Any]:
        from cais_spade_llm.resources.resource_profile import (
            get_resource_profile,
        )

        rs = resource_states.get(resource_jid, {})
        resource_type = str(rs.get("resource_type", "") or "").strip().lower()
        profile = get_resource_profile(resource_type or "resource")
        current_part = rs.get(
            str(profile.carried_entity_field or "current_part")
        )
        return {
            "resource_state": rs.get("current_state", default_resource_state),
            "current_part": current_part,
            "current_location": rs.get("current_location"),
            "part_states": part_states,
            "part_locations": part_locations,
        }

    def _project_resource_suffix_state(
        self,
        *,
        resource_jid: str,
        tools_catalog: list[dict[str, Any]],
        resource_states: dict[str, dict[str, Any]],
        default_resource_state: str,
        part_states: dict[str, Any],
        part_locations: dict[str, Any],
        goal_state: str | None,
        ignored_task_ids: set[str] | None = None,
    ) -> tuple[dict[str, Any] | None, str, str]:
        """Project the modeled snapshot after this resource's pending suffix.

        Returns (projected_snapshot, last_task_id, failure_reason). If projection
        cannot be completed consistently from the live modeled state, returns
        (None, last_successfully_projected_task_id, failure_reason) so callers
        can fall back to the live snapshot.
        """
        from cais_spade_llm.agents.intelligent_product.replanner.des_search.resource_bidding import (
            simulate_catalog_transition,
        )

        pending = self._pending_resource_tasks(
            resource_jid,
            ignored_task_ids=ignored_task_ids,
        )

        projected_state = self._build_resource_search_state(
            resource_jid=resource_jid,
            resource_states=resource_states,
            default_resource_state=default_resource_state,
            part_states=part_states,
            part_locations=part_locations,
        )
        if not pending:
            return projected_state, "", ""

        ra = self._resource_by_jid(resource_jid)
        reachability = getattr(ra, "static_capabilities", {}).get("reachability", []) if ra else []
        staging_areas = getattr(ra, "static_capabilities", {}).get("staging_areas", {}) if ra else {}
        last_task_id = ""
        for task in pending:
            fn = str(task.get("function_name", "")).strip()
            params = dict(task.get("params") or {})
            tool_row = self._tool_row_for_task(
                resource_jid=resource_jid,
                function_name=fn,
                tools_catalog=tools_catalog,
            )
            simulated = simulate_catalog_transition(
                x_c=projected_state,
                tools=tools_catalog,
                resource_jid=resource_jid,
                function_name=fn,
                params=params,
                goal_state=goal_state or "",
                reachability=reachability,
                staging_areas=staging_areas,
            )
            reconcile_notes: list[str] = []
            if simulated is None and tool_row:
                reconciled_state, reconcile_notes = self._reconcile_search_state_for_task(
                    search_state=projected_state,
                    task=task,
                    tool_row=tool_row,
                )
                if reconciled_state != projected_state:
                    simulated = simulate_catalog_transition(
                        x_c=reconciled_state,
                        tools=tools_catalog,
                        resource_jid=resource_jid,
                        function_name=fn,
                        params=params,
                        goal_state=goal_state or "",
                        reachability=reachability,
                        staging_areas=staging_areas,
                    )
                    if simulated is not None:
                        projected_state = reconciled_state
                        self.logger.info(
                            "[Planner] Reconciled modeled state for %s before projecting %s: %s",
                            resource_jid,
                            str(task.get("id", "")).strip() or fn,
                            "; ".join(reconcile_notes),
                        )
            if simulated is None:
                reason = (
                    f"task={task.get('id', '')} function={fn} could not be projected "
                    f"from state={projected_state.get('resource_state', 'unknown')}"
                )
                if reconcile_notes:
                    reason += f" after reconciliation ({'; '.join(reconcile_notes)})"
                return None, last_task_id, reason
            projected_state, _ = simulated
            last_task_id = str(task.get("id", "")).strip()

        return projected_state, last_task_id, ""

    def _collect_obligation_targets(
        self,
        violations: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        targets: list[dict[str, Any]] = []
        seen: set[tuple[str, str, tuple[str, ...]]] = set()

        for violation in violations or []:
            if not isinstance(violation, dict):
                continue
            safety_ctx = violation.get("safety_ctx") or {}
            raw_targets = safety_ctx.get("obligation_targets") or []
            if not isinstance(raw_targets, list):
                continue
            for target in raw_targets:
                if not isinstance(target, dict):
                    continue
                rule_id = str(target.get("rule_id", "")).strip()
                resource_jid = str(target.get("resource_jid", "")).strip()
                candidate_tools = target.get("candidate_tools") or []
                signatures = tuple(
                    sorted(
                        str(tool.get("tool_signature", "")).strip()
                        for tool in candidate_tools
                        if isinstance(tool, dict) and str(tool.get("tool_signature", "")).strip()
                    )
                )
                key = (rule_id, resource_jid, signatures)
                if key in seen:
                    continue
                seen.add(key)
                targets.append(deepcopy(target))

        return targets

    def _primary_failure_context(
        self,
        violations: list[dict[str, Any]],
        *,
        fallback_resource_jid: str = "",
    ) -> dict[str, Any]:
        fallback_jid = str(fallback_resource_jid or "").strip()
        for violation in violations or []:
            if not isinstance(violation, dict):
                continue
            failed_task_id = str(
                violation.get("failed_task_id") or violation.get("task_id") or ""
            ).strip()
            failed_resource_jid = str(
                violation.get("resource_jid") or violation.get("failed_resource_jid") or ""
            ).strip()
            if not failed_resource_jid and failed_task_id:
                failed_resource_jid = self._resource_jid_for_task_id(failed_task_id)
            if not failed_resource_jid:
                failed_resource_jid = fallback_jid
            raw_failure_context = violation.get("failure_context")
            if not isinstance(raw_failure_context, dict):
                raw_failure_context = {}
            observations = raw_failure_context.get("observations")
            observations = observations if isinstance(observations, dict) else {}
            failed_function_name = str(
                observations.get("function_name")
                or self._function_name_for_task_id(failed_task_id)
                or ""
            ).strip()
            if raw_failure_context or failed_task_id or failed_resource_jid or failed_function_name:
                return {
                    "failed_task_id": failed_task_id,
                    "failed_resource_jid": failed_resource_jid,
                    "failed_function_name": failed_function_name,
                    "failure_context": deepcopy(raw_failure_context),
                }

        return {
            "failed_task_id": "",
            "failed_resource_jid": fallback_jid,
            "failed_function_name": "",
            "failure_context": {},
        }

    def _node_exists(self, task_id: str) -> bool:
        return any(
            isinstance(node, dict) and str(node.get("id", "")).strip() == str(task_id).strip()
            for node in self.nodes
        )

    def _path_to_recovery_tasks(
        self,
        path: list[dict[str, Any]],
        *,
        anchor_task_id: str = "",
        task_prefix: str = "RECOVERY_DES",
        change_prefix: str = "DES recovery",
        macro_name: str = "",
    ) -> list[dict[str, Any]]:
        from uuid import uuid4

        # Determine a sequence_index base that sorts after all existing tasks.
        max_si = 0
        for n in self.nodes:
            si = n.get("sequence_index")
            if si is not None:
                max_si = max(max_si, int(si))
        base_si = max_si + 1000

        tasks: list[dict[str, Any]] = []
        predecessor = str(anchor_task_id).strip() if self._node_exists(anchor_task_id) else ""
        total_steps = len(path)
        for index, event in enumerate(path, start=1):
            task_id = f"{task_prefix}_{uuid4().hex[:6].upper()}"
            params = {
                **dict(event.get("params") or {}),
                "product_jid": str(self.product_agent.jid),
                "task_id": task_id,
            }
            reason = (
                f"INSERTION: {change_prefix} — {event['function_name']} on {event.get('ra_jid', 'unknown')}"
            )
            if macro_name:
                reason = (
                    f"INSERTION: {change_prefix} '{macro_name}' step {index}/{total_steps} — "
                    f"{event['function_name']} on {event.get('ra_jid', 'unknown')}"
                )
            tasks.append(
                {
                    "id": task_id,
                    "function_name": event["function_name"],
                    "params": params,
                    "resource_jid": event.get("ra_jid"),
                    "predecessors": [predecessor] if predecessor else [],
                    "successors": [],
                    "sequence_index": base_si + index,
                    "change_reason": reason,
                }
            )
            predecessor = task_id
        return tasks

    def _gate_tasks_after_recovery_tail(
        self,
        modified_tasks: list[dict[str, Any]],
        *,
        tail_task_id: str,
        before_task_ids: list[str] | None = None,
        change_prefix: str = "DES recovery",
    ) -> None:
        if not tail_task_id:
            return
        max_sequence_index = 0
        for node in self.nodes:
            try:
                max_sequence_index = max(max_sequence_index, int(node.get("sequence_index") or 0))
            except (TypeError, ValueError):
                continue
        for node in modified_tasks:
            try:
                max_sequence_index = max(max_sequence_index, int(node.get("sequence_index") or 0))
            except (TypeError, ValueError):
                continue
        next_sequence_index = max_sequence_index + 1
        seen: set[str] = set()
        ordered_task_ids: list[str] = []
        existing_by_id: dict[str, dict[str, Any]] = {}
        for task_id in before_task_ids or []:
            blocked_task_id = str(task_id or "").strip()
            if not blocked_task_id or blocked_task_id == tail_task_id or blocked_task_id in seen:
                continue
            seen.add(blocked_task_id)
            existing = self._find_node(blocked_task_id)
            if existing is None:
                continue
            ordered_task_ids.append(blocked_task_id)
            existing_by_id[blocked_task_id] = existing
        resume_task_ids = set(ordered_task_ids)

        def _sort_key(task_id: str) -> tuple[int, str]:
            existing = existing_by_id.get(task_id) or {}
            try:
                sequence_index = int(existing.get("sequence_index") or 0)
            except (TypeError, ValueError):
                sequence_index = 0
            return (sequence_index, task_id)

        # Resolve the tail task's resource so we only bump sequence_index
        # for same-resource gating (cross-resource bumps break local ordering).
        tail_resource_jid = ""
        for t in modified_tasks:
            if str(t.get("id", "")).strip() == tail_task_id:
                tail_resource_jid = str(t.get("resource_jid", "")).strip()
                break
        if not tail_resource_jid:
            existing_tail = self._find_node(tail_task_id)
            if existing_tail:
                tail_resource_jid = str(existing_tail.get("resource_jid", "")).strip()

        for blocked_task_id in sorted(ordered_task_ids, key=_sort_key):
            existing = existing_by_id[blocked_task_id]
            existing_preds = [
                str(pred).strip()
                for pred in (existing.get("predecessors") or [])
                if str(pred or "").strip()
            ]
            gated_by_resume_chain = any(pred in resume_task_ids for pred in existing_preds)
            preds = (
                list(existing_preds)
                if gated_by_resume_chain
                else list(dict.fromkeys(existing_preds + [tail_task_id]))
            )
            blocked_resource_jid = str(existing.get("resource_jid", "")).strip()
            same_resource = (
                tail_resource_jid
                and blocked_resource_jid
                and tail_resource_jid == blocked_resource_jid
            )
            mod: dict[str, Any] = {
                "id": blocked_task_id,
                "predecessors": preds,
                "change_reason": (
                    f"MODIFICATION: {change_prefix} — gate {blocked_task_id} after {tail_task_id}"
                ),
            }
            if same_resource:
                mod["sequence_index"] = next_sequence_index
                next_sequence_index += 1
            modified_tasks.append(mod)

    def apply_bridge_macro_proposal(
        self,
        proposal: dict[str, Any],
        *,
        anchor_task_id: str = "",
    ) -> list[dict[str, Any]]:
        if not isinstance(proposal, dict):
            raise ValueError("bridge proposal is missing")

        # Primitive-based proposal: compile top-level primitive_steps or ordered macro_tasks[]
        # into one serial execute_recovery_macro chain.
        if self._primitive_bridge_macro_tasks(proposal):
            return self._apply_primitive_bridge_proposal(proposal, anchor_task_id=anchor_task_id)

        # Legacy catalog-function-based proposal: compile into multiple task nodes.
        path: list[dict[str, Any]] = []
        proposal_name = str(proposal.get("function_name", "")).strip()
        for step in proposal.get("macro_steps") or []:
            if not isinstance(step, dict):
                continue
            function_name = str(step.get("function_name", "")).strip()
            if not function_name:
                continue
            path.append(
                {
                    "function_name": function_name,
                    "params": dict(step.get("params") or {}),
                    "ra_jid": str(step.get("resource_jid") or proposal.get("resource_jid") or "").strip(),
                }
            )

        if not path:
            raise ValueError("bridge proposal has no executable macro_steps")

        tasks = self._path_to_recovery_tasks(
            path,
            anchor_task_id=anchor_task_id,
            task_prefix="RECOVERY_BRIDGE",
            change_prefix="Approved bridge recovery",
            macro_name=proposal_name,
        )
        self._apply_replan_patch(tasks)
        return tasks

    def _apply_primitive_bridge_proposal(
        self,
        proposal: dict[str, Any],
        *,
        anchor_task_id: str = "",
    ) -> list[dict[str, Any]]:
        """Compile one or more primitive-based bridge macro_tasks into ordered task nodes."""
        from uuid import uuid4

        macro_tasks = self._primitive_bridge_macro_tasks(proposal)
        if not macro_tasks:
            raise ValueError("bridge proposal has no primitive_steps/macro_tasks")

        max_si = 0
        for node in self.nodes:
            si = node.get("sequence_index")
            if si is not None:
                max_si = max(max_si, int(si))
        base_si = max_si + 1000

        compiled_nodes: list[dict[str, Any]] = []
        predecessor = str(anchor_task_id).strip() if self._node_exists(anchor_task_id) else ""
        total_tasks = len(macro_tasks)
        primary_obligation = deepcopy(proposal.get("primary_obligation") or {})
        bridge_sequence_id = f"BRIDGESEQ_{uuid4().hex[:8].upper()}"

        for index, macro_task in enumerate(macro_tasks, start=1):
            macro_name = str(
                macro_task.get("macro_name")
                or proposal.get("macro_name")
                or f"bridge_recovery_macro_{index}"
            ).strip()
            resource_jid = str(macro_task.get("resource_jid") or proposal.get("resource_jid") or "").strip()
            primitive_steps = list(macro_task.get("primitive_steps") or [])
            expected_start_state = str(
                macro_task.get("expected_start_state") or proposal.get("expected_start_state") or ""
            ).strip()
            part_name = str(
                macro_task.get("part_name")
                or macro_task.get("touched_part")
                or proposal.get("part_name")
                or proposal.get("touched_part")
                or ""
            ).strip()
            task_params = macro_task.get("task_params")
            if task_params is None:
                task_params = proposal.get("task_params") if total_tasks == 1 else {}
            task_metadata = macro_task.get("task_metadata")
            if task_metadata is None:
                task_metadata = proposal.get("task_metadata") if total_tasks == 1 else {}
            expected_snapshot = macro_task.get("expected_snapshot")
            if expected_snapshot is None and total_tasks == 1:
                expected_snapshot = proposal.get("expected_snapshot")
            projected_snapshot = macro_task.get("projected_snapshot")
            projected_part_entry = macro_task.get("projected_part_entry")

            if not resource_jid:
                raise ValueError(f"bridge macro_task {index} is missing resource_jid")
            if not primitive_steps:
                raise ValueError(f"bridge macro_task {index} has no primitive_steps")

            task_id = f"RECOVERY_BRIDGE_{uuid4().hex[:6].upper()}"
            params: dict[str, Any] = {
                "macro_name": macro_name,
                "primitive_steps": primitive_steps,
                "expected_start_state": expected_start_state,
                "product_jid": str(self.product_agent.jid),
                "task_id": task_id,
            }
            if isinstance(task_params, dict):
                for key, value in task_params.items():
                    params[str(key)] = deepcopy(value)
            if part_name:
                params["part_name"] = part_name
            if isinstance(expected_snapshot, dict) and expected_snapshot:
                params["expected_snapshot"] = dict(expected_snapshot)
            if isinstance(task_metadata, dict) and task_metadata.get("out_state"):
                params["out_state"] = str(task_metadata["out_state"])

            step_summary = ", ".join(
                str(step.get("primitive", "?")) for step in primitive_steps[:5] if isinstance(step, dict)
            )
            if len(primitive_steps) > 5:
                step_summary += f", ... ({len(primitive_steps)} total)"

            task_node: dict[str, Any] = {
                "id": task_id,
                "function_name": "execute_recovery_macro",
                "params": params,
                "resource_jid": resource_jid,
                "predecessors": [predecessor] if predecessor else [],
                "successors": [],
                "sequence_index": base_si + index,
                "change_reason": (
                    f"INSERTION: Approved bridge recovery macro '{macro_name}' "
                    f"step {index}/{total_tasks} ({len(primitive_steps)} primitives: {step_summary}) "
                    f"on {resource_jid}"
                ),
                "bridge_sequence_id": bridge_sequence_id,
                "bridge_sequence_index": index,
                "bridge_sequence_length": total_tasks,
            }

            if isinstance(task_metadata, dict):
                if task_metadata.get("in_state"):
                    task_node["in_state"] = str(task_metadata["in_state"])
                if task_metadata.get("out_state"):
                    task_node["out_state"] = str(task_metadata["out_state"])
                if task_metadata.get("required_context_keys"):
                    task_node["required_context_keys"] = list(task_metadata["required_context_keys"])
                if task_metadata.get("context_mapping"):
                    task_node["context_mapping"] = dict(task_metadata["context_mapping"])
                if task_metadata.get("part_transition"):
                    task_node["part_transition"] = dict(task_metadata["part_transition"])
            if part_name:
                task_node["part_name"] = part_name
            if isinstance(projected_snapshot, dict) and projected_snapshot:
                task_node["projected_snapshot"] = deepcopy(projected_snapshot)
            if isinstance(projected_part_entry, dict) and projected_part_entry:
                task_node["projected_part_entry"] = deepcopy(projected_part_entry)
            if primary_obligation:
                task_node["primary_obligation"] = deepcopy(primary_obligation)

            compiled_nodes.append(task_node)
            predecessor = task_id

        self._apply_replan_patch(compiled_nodes)
        return compiled_nodes

    async def replan_with_feedback_offline(self, violations: list[dict]) -> None:
        """Offline replan using safety validator feedback."""
        self.logger.info("[Planner] Triggering LLM Re-planning with offline feedback...")

        failed_nodes = [n for n in self.nodes if n.get("type") == "task"]
        conflict_task_ids = self._extract_conflict_task_ids(violations, source="offline")
        plan_payload = []
        for node in failed_nodes:
            node_copy = node.copy()
            if conflict_task_ids and node_copy["id"] in conflict_task_ids:
                node_copy["_FOCUS_HERE"] = " <<< THIS TASK IS INVOLVED IN A VIOLATION"
            plan_payload.append(node_copy)

        if not conflict_task_ids:
            self.logger.warning("[Planner] No conflict task IDs found; sending full plan context.")

        tools_catalog = self._deduplicate_tools_catalog(getattr(self.product_agent, "tools_catalog", []))
        resource_infos = [
            {"jid": str(getattr(ra, "jid", "")), "static_capabilities": getattr(ra, "static_capabilities", {})}
            for ra in self.resource_agents
        ]

        prompt = build_replan_prompt(
            failed_plan_nodes=plan_payload,
            violations=violations,
            tools_catalog=tools_catalog,
            resource_infos=resource_infos,
            source="offline",
            safety_text=self.product_agent.safety_text,
            system_state=None,
        )
        raw = await self.product_agent.ask_llm(prompt=prompt, with_functions=False, temperature=0.0)
        self._dump_replan_debug(source="offline", prompt=prompt, violations=violations,
                                resource_infos=resource_infos, system_state=None, llm_response=raw)
        try:
            modified_tasks = json.loads(raw).get("tasks", [])
            if not modified_tasks:
                self.logger.warning("[Planner] LLM returned no modified tasks.")
                return
            self._apply_replan_patch(modified_tasks)
        except json.JSONDecodeError as exc:
            self.logger.error("[Planner] LLM replanning returned invalid JSON: %s", exc)

    async def replan_with_feedback_online(
        self,
        violations: list[dict],
        system_coordination_state: dict | None = None,
        bridge_feedback: str = "",
        bridge_generation_mode: str = "auto",
    ) -> dict[str, Any] | None:
        """Online replan — always run DES recovery, then DES bridge fallback if needed."""
        return await self.replan_with_feedback_des(
            violations,
            system_coordination_state=system_coordination_state,
            bridge_feedback=bridge_feedback,
            bridge_generation_mode=bridge_generation_mode,
        )

    async def replan_with_feedback_des(
        self,
        violations: list[dict],
        system_coordination_state: dict | None = None,
        bridge_feedback: str = "",
        *,
        allow_bridge_fallback: bool = True,
        bridge_generation_mode: str = "auto",
        ignored_task_ids: set[str] | None = None,
    ) -> dict[str, Any]:
        """
        DES replanning: PA computes bids per resource, compiles M_e, runs BFS.
        LLM bridge if stuck. Falls back to human intervention if no path found.
        """
        from cais_spade_llm.agents.intelligent_product.replanner.des_search.environment_model import (
            compile_environment_model,
            plan_on_environment_model,
        )
        from cais_spade_llm.agents.intelligent_product.replanner.des_search.resource_bidding import (
            Bid,
            compute_bid,
        )

        self.logger.info("[Planner] DES replanning triggered (%d violations).", len(violations))
        self._set_last_bridge_debug({})
        ignored_task_ids = {
            str(task_id).strip()
            for task_id in (ignored_task_ids or set())
            if str(task_id).strip()
        }

        # 1. Build P_id: parts not yet at goal state
        product_state = self.product_agent._build_product_state()
        derived_part_tracker = self._derive_part_tracker_from_violations(violations)
        part_tracker = product_state.get("parts")
        if not isinstance(part_tracker, dict) or not part_tracker:
            legacy_part_tracker = product_state.get("part_tracker")
            if isinstance(legacy_part_tracker, dict) and legacy_part_tracker:
                part_tracker = legacy_part_tracker
            else:
                part_tracker = derived_part_tracker
        if isinstance(part_tracker, dict) and derived_part_tracker:
            merged_part_tracker = deepcopy(part_tracker)
            for part_name, derived_entry in derived_part_tracker.items():
                if not isinstance(derived_entry, dict):
                    continue
                current_entry = dict(merged_part_tracker.get(part_name) or {})
                force_keys = {
                    str(key).strip()
                    for key in (derived_entry.get("_force_keys") or [])
                    if str(key).strip()
                }
                for key, value in derived_entry.items():
                    if key == "_force_keys":
                        continue
                    if key not in force_keys and value in (None, "", [], {}):
                        continue
                    if (
                        key in {"state", "location"}
                        and key not in force_keys
                        and current_entry.get(key) not in (None, "", "unknown")
                    ):
                        continue
                    current_entry[key] = deepcopy(value)
                merged_part_tracker[str(part_name)] = current_entry
            part_tracker = merged_part_tracker
        tools_catalog = getattr(self.product_agent, "tools_catalog", [])
        goal_state = self._resolve_goal_part_state(tools_catalog)
        P_id = [
            name for name, info in part_tracker.items()
            if not goal_state or info.get("state") != goal_state
        ]

        # 2. Build x_c per resource from coordination state and part tracker
        scs = system_coordination_state or {}
        resource_states = self._extract_resource_states(scs)
        part_states = {name: info.get("state") for name, info in part_tracker.items()}
        part_locations = {name: info.get("location") for name, info in part_tracker.items()}
        default_resource_state = self._default_resource_state(tools_catalog)
        obligation_targets = self._collect_obligation_targets(violations)
        bridge_safety_context = self._collect_bridge_safety_context(violations)
        used_llm_bridge = False
        bridge_summary: list[str] = []
        path: list[dict[str, Any]] | None = None
        x_c: dict[str, Any] | None = None
        stuck_ra_jid = self._identify_stuck_resource(violations, resource_states)
        failure_context_payload = self._primary_failure_context(
            violations,
            fallback_resource_jid=stuck_ra_jid,
        )

        # Track the best anchor for obligation recovery (the target resource's
        # last pending task, NOT the blocked task on a different resource).
        obligation_anchor_task_id: str = ""

        if obligation_targets:
            best_target: dict[str, Any] | None = None
            best_path: list[dict[str, Any]] | None = None
            for target in obligation_targets:
                if not isinstance(target, dict):
                    continue
                target_ra_jid = str(target.get("resource_jid", "")).strip()
                if not target_ra_jid:
                    continue
                candidate_signatures = {
                    str(tool.get("tool_signature", "")).strip()
                    for tool in (target.get("candidate_tools") or [])
                    if isinstance(tool, dict) and str(tool.get("tool_signature", "")).strip()
                }
                if not candidate_signatures:
                    continue
                ra = self._resource_by_jid(target_ra_jid)
                if ra is None:
                    continue

                live_candidate_state = self._build_resource_search_state(
                    resource_jid=target_ra_jid,
                    resource_states=resource_states,
                    default_resource_state=default_resource_state,
                    part_states=part_states,
                    part_locations=part_locations,
                )
                projected_candidate_state, last_pending_tid, projection_reason = (
                    self._project_resource_suffix_state(
                        resource_jid=target_ra_jid,
                        tools_catalog=tools_catalog,
                        resource_states=resource_states,
                        default_resource_state=default_resource_state,
                        part_states=part_states,
                        part_locations=part_locations,
                        goal_state=goal_state,
                        ignored_task_ids=ignored_task_ids,
                    )
                )
                pending_suffix = self._pending_resource_tasks(
                    target_ra_jid,
                    ignored_task_ids=ignored_task_ids,
                )
                has_pending_suffix = bool(pending_suffix)
                if projected_candidate_state is not None:
                    candidate_state = projected_candidate_state
                else:
                    if projection_reason:
                        if has_pending_suffix:
                            # Check for in_state="any" tools that bypass projection
                            any_state_tools = [
                                tool
                                for tool in (target.get("candidate_tools") or [])
                                if isinstance(tool, dict)
                                and str(tool.get("in_state", "")).strip().lower() == "any"
                            ]
                            if any_state_tools:
                                last_pending_id = str(
                                    pending_suffix[-1].get("id", "")
                                ).strip()
                                direct_tool = any_state_tools[0]
                                direct_path = [
                                    {
                                        "function_name": str(
                                            direct_tool["function_name"]
                                        ),
                                        "ra_jid": target_ra_jid,
                                        "params": {},
                                    }
                                ]
                                if best_path is None or len(direct_path) < len(
                                    best_path
                                ):
                                    best_target = target
                                    best_path = direct_path
                                    stuck_ra_jid = target_ra_jid
                                    x_c = live_candidate_state
                                    obligation_anchor_task_id = last_pending_id
                                self.logger.info(
                                    "[Planner] Obligation recovery: projection failed "
                                    "for %s but candidate tool %s has in_state=any; "
                                    "anchoring after last pending task %s.",
                                    target_ra_jid,
                                    direct_tool["function_name"],
                                    last_pending_id,
                                )
                                continue
                            self.logger.info(
                                "[Planner] Obligation recovery projection skipped for %s: %s. "
                                "Live-state fallback disabled because %d pending/running task(s) "
                                "would otherwise be replayed as duplicate recovery.",
                                target_ra_jid,
                                projection_reason,
                                len(pending_suffix),
                            )
                        else:
                            self.logger.info(
                                "[Planner] Obligation recovery projection skipped for %s: %s. "
                                "Falling back to live modeled state.",
                                target_ra_jid,
                                projection_reason,
                            )
                    if has_pending_suffix:
                        continue
                    candidate_state = live_candidate_state

                bid = compute_bid(
                    x_c=candidate_state,
                    P_id=[],
                    goal_state=goal_state or "",
                    tools=tools_catalog,
                    reachability=getattr(ra, "static_capabilities", {}).get("reachability", []),
                    staging_areas=getattr(ra, "static_capabilities", {}).get("staging_areas", {}),
                    resource_jid=target_ra_jid,
                    goal_event_signatures=candidate_signatures,
                )
                if not bid or not bid.str_e:
                    continue
                candidate_path = [{**event, "ra_jid": target_ra_jid} for event in bid.str_e]
                if best_path is None or len(candidate_path) < len(best_path):
                    best_target = target
                    best_path = candidate_path
                    stuck_ra_jid = target_ra_jid
                    x_c = candidate_state
                    obligation_anchor_task_id = (
                        last_pending_tid if projected_candidate_state is not None else ""
                    )

            if best_path:
                path = best_path
                self.logger.info(
                    "[Planner] Modeled obligation recovery matched rule %s with %d step(s).",
                    best_target.get("rule_id") if isinstance(best_target, dict) else "unknown",
                    len(best_path),
                )

        # 3. Compute a bid for each resource agent
        if path is None and P_id and not obligation_targets:
            bids: list[Bid] = []
            for ra in self.resource_agents:
                ra_jid = str(ra.jid)
                x_c = self._build_resource_search_state(
                    resource_jid=ra_jid,
                    resource_states=resource_states,
                    default_resource_state=default_resource_state,
                    part_states=part_states,
                    part_locations=part_locations,
                )
                reachability = getattr(ra, "static_capabilities", {}).get("reachability", [])
                staging_areas = getattr(ra, "static_capabilities", {}).get("staging_areas", {})
                bid = compute_bid(
                    x_c=x_c,
                    P_id=P_id,
                    goal_state=goal_state,
                    tools=tools_catalog,
                    reachability=reachability,
                    staging_areas=staging_areas,
                    resource_jid=ra_jid,
                )
                if bid:
                    self.logger.debug("[Planner] Bid from %s: complete=%s", ra_jid, bid.complete)
                    bids.append(bid)
                else:
                    self.logger.debug("[Planner] No bid from %s.", ra_jid)

            M_e = compile_environment_model(bids)

            # If we know the product agent will auto-load a preprogrammed scenario,
            # skip the 8s BFS search entirely.
            if bridge_generation_mode == "manual":
                x_c = self._build_resource_search_state(
                    resource_jid=stuck_ra_jid,
                    resource_states=resource_states,
                    default_resource_state=default_resource_state,
                    part_states=part_states,
                    part_locations=part_locations,
                )
                self.logger.info("[Planner] Fast-tracking to bridge request (skipping DES search for preprogrammed scenarios).")
            else:
                if bids:
                    x_c = bids[0].str_x[0]
                else:
                    x_c = {
                        "part_states": part_states,
                        "part_locations": part_locations,
                        "resource_state": default_resource_state,
                    }
                path = plan_on_environment_model(M_e, x_c, P_id, goal_state)
        elif path is None and not P_id and not obligation_targets:
            x_c = self._build_resource_search_state(
                resource_jid=stuck_ra_jid,
                resource_states=resource_states,
                default_resource_state=default_resource_state,
                part_states=part_states,
                part_locations=part_locations,
            )
            ra = self._resource_by_jid(stuck_ra_jid)
            modeled_bid = None
            if ra is not None:
                modeled_bid = compute_bid(
                    x_c=x_c,
                    P_id=[],
                    goal_state=goal_state or "",
                    goal_resource_state=default_resource_state,
                    tools=tools_catalog,
                    reachability=getattr(ra, "static_capabilities", {}).get("reachability", []),
                    staging_areas=getattr(ra, "static_capabilities", {}).get("staging_areas", {}),
                    resource_jid=stuck_ra_jid,
                )
            if modeled_bid and modeled_bid.str_e:
                self.logger.info(
                    "[Planner] Modeled DES suffix recovery found %d step(s) for %s.",
                    len(modeled_bid.str_e),
                    stuck_ra_jid,
                )
                path = [{**event, "ra_jid": stuck_ra_jid} for event in modeled_bid.str_e]

        if path is None:
            if x_c is None:
                x_c = self._build_resource_search_state(
                    resource_jid=stuck_ra_jid,
                    resource_states=resource_states,
                    default_resource_state=default_resource_state,
                    part_states=part_states,
                    part_locations=part_locations,
                )
            if not allow_bridge_fallback:
                message = "DES reevaluation found no modeled continuation from the refreshed runtime state."
                self.logger.info("[Planner] %s", message)
                return self._build_des_replan_result(
                    des_recovery_missing=True,
                    used_llm_bridge=False,
                    human_required=False,
                    message=message,
                    bridge_summary=bridge_summary,
                    bridge_debug=None,
                )
            self.logger.info("[Planner] DES found no modeled continuation; requesting bridge proposal.")
            prepared_bridge_request = await self.prepare_bridge_request(
                stuck_state=x_c,
                P_id=P_id,
                ra_jid=stuck_ra_jid,
                goal_state=goal_state or "unknown",
                tools_catalog=tools_catalog,
                part_tracker=part_tracker,
                obligation_targets=obligation_targets,
                bridge_feedback=bridge_feedback,
                resource_states=resource_states,
                default_resource_state=default_resource_state,
                part_states=part_states,
                part_locations=part_locations,
                bridge_safety_context=bridge_safety_context,
                failure_context=failure_context_payload,
            )
            if str(bridge_generation_mode or "auto").strip().lower() == "manual":
                message = (
                    "DES found no modeled continuation. Review the prepared bridge request and "
                    "run LLM exploration from the dashboard when ready."
                )
                self.logger.info("[Planner] %s", message)
                return self._build_des_replan_result(
                    plan_changed=False,
                    used_llm_bridge=False,
                    human_required=False,
                    awaiting_bridge_generation=True,
                    message=message,
                    bridge_summary=bridge_summary,
                    bridge_debug=self.get_last_bridge_debug(),
                    prepared_bridge_request=prepared_bridge_request,
                )
            bridge_proposal = await self.execute_prepared_bridge_request(prepared_bridge_request)
            if bridge_proposal:
                used_llm_bridge = True
                bridge_summary = self._bridge_summary(bridge_proposal)
                message = (
                    "DES found no catalog-valid continuation. A bridge macro proposal is ready for approval."
                )
                self.logger.info("[Planner] %s", message)
                return self._build_des_replan_result(
                    plan_changed=False,
                    used_llm_bridge=True,
                    human_required=False,
                    awaiting_bridge_approval=True,
                    message=message,
                    bridge_summary=bridge_summary,
                    bridge_proposal=bridge_proposal,
                    bridge_debug=self.get_last_bridge_debug(),
                )
            bridge_debug = self.get_last_bridge_debug()
            bridge_status = str((bridge_debug or {}).get("status") or "").strip().lower()
            if bridge_status == "ready_for_llm":
                message = (
                    "DES found no modeled continuation. The bridge request is ready for "
                    "LLM review, but no validated bridge proposal is available yet."
                )
                self.logger.info("[Planner] %s", message)
                return self._build_des_replan_result(
                    plan_changed=False,
                    used_llm_bridge=False,
                    human_required=False,
                    awaiting_bridge_generation=True,
                    message=message,
                    bridge_summary=bridge_summary,
                    bridge_debug=bridge_debug,
                    prepared_bridge_request=prepared_bridge_request,
                )
            if bridge_status == "paused_after_grounding":
                message = (
                    "DES found no modeled continuation. Bridge grounding completed and "
                    "paused before recovery synthesis for review."
                )
                self.logger.info("[Planner] %s", message)
                return self._build_des_replan_result(
                    plan_changed=False,
                    used_llm_bridge=True,
                    human_required=False,
                    awaiting_bridge_generation=True,
                    message=message,
                    bridge_summary=bridge_summary,
                    bridge_debug=bridge_debug,
                    prepared_bridge_request=prepared_bridge_request,
                )
            if bridge_status == "paused_after_outline_turn":
                message = (
                    "DES found no modeled continuation. Multi-turn bridge accepted an outline "
                    "and is ready to continue primitive generation."
                )
                self.logger.info("[Planner] %s", message)
                return self._build_des_replan_result(
                    plan_changed=False,
                    used_llm_bridge=True,
                    human_required=False,
                    awaiting_bridge_generation=True,
                    message=message,
                    bridge_summary=bridge_summary,
                    bridge_debug=bridge_debug,
                    prepared_bridge_request=prepared_bridge_request,
                )
            if bridge_status == "paused_after_primitive_turn":
                message = (
                    "DES found no modeled continuation. Multi-turn bridge requested another "
                    "primitive-generation retry from the prepared session."
                )
                self.logger.info("[Planner] %s", message)
                return self._build_des_replan_result(
                    plan_changed=False,
                    used_llm_bridge=True,
                    human_required=False,
                    awaiting_bridge_generation=True,
                    message=message,
                    bridge_summary=bridge_summary,
                    bridge_debug=bridge_debug,
                    prepared_bridge_request=prepared_bridge_request,
                )
            if bridge_status in {"paused_after_primitive_blocked", "paused_after_primitive_stuck"}:
                message = (
                    "DES found no modeled continuation. Multi-turn bridge stalled during "
                    "primitive generation and requires operator review."
                )
                self.logger.warning("[Planner] %s", message)
                return self._build_des_replan_result(
                    human_required=True,
                    used_llm_bridge=True,
                    message=message,
                    bridge_summary=bridge_summary,
                    bridge_debug=bridge_debug,
                    prepared_bridge_request=prepared_bridge_request,
                )
            if bridge_status == "unsupported_reasoning_mode":
                message = (
                    "DES found no modeled continuation, but the selected bridge reasoning mode "
                    "is not implemented."
                )
                self.logger.error("[Planner] %s", message)
                return self._build_des_replan_result(
                    human_required=True,
                    used_llm_bridge=False,
                    message=message,
                    bridge_summary=bridge_summary,
                    bridge_debug=bridge_debug,
                    prepared_bridge_request=prepared_bridge_request,
                )
            message = "DES recovery could not find a modeled path and the LLM bridge produced no compilable proposal."
            self.logger.error("[Planner] %s", message)
            return self._build_des_replan_result(
                human_required=True,
                used_llm_bridge=used_llm_bridge,
                message=message,
                bridge_summary=bridge_summary,
                bridge_debug=bridge_debug,
                prepared_bridge_request=prepared_bridge_request,
            )

        failed_task_id = ""
        for violation in violations or []:
            if not isinstance(violation, dict):
                continue
            candidate_task_id = str(violation.get("failed_task_id", "")).strip()
            if candidate_task_id:
                failed_task_id = candidate_task_id
                break

        # For obligation recovery, anchor to the target resource's last
        # pending task (not the blocked task on a different resource).
        anchor = obligation_anchor_task_id if obligation_anchor_task_id else failed_task_id
        tasks = self._path_to_recovery_tasks(
            path,
            anchor_task_id=anchor,
        )
        entry_task_ids = [
            task_id for task_id in self._entry_task_ids_from_violations(violations)
            if task_id and task_id != anchor
        ]
        if tasks and entry_task_ids:
            self._gate_tasks_after_recovery_tail(
                tasks,
                tail_task_id=str(tasks[-1].get("id", "")).strip(),
                before_task_ids=entry_task_ids,
            )
        self.logger.info("[Planner] DES recovery path: %d tasks.", len(tasks))
        self._apply_replan_patch(tasks)
        message = f"DES recovery produced {len(tasks)} task(s)."
        return self._build_des_replan_result(
            plan_changed=bool(tasks),
            used_llm_bridge=False,
            human_required=False,
            message=message,
            bridge_summary=bridge_summary,
            bridge_debug=None,
        )

    @staticmethod
    def _resolve_goal_part_state(tools_catalog: list[dict[str, Any]]) -> str | None:
        completed_states: list[str] = []
        intermediate_states: set[str] = set()
        for tool in tools_catalog or []:
            if not isinstance(tool, dict):
                continue
            part_in_state = str(tool.get("part_in_state") or "").strip()
            if part_in_state:
                intermediate_states.add(part_in_state)
            completed_state = str(
                tool.get("part_transition", {}).get("completed", {}).get("state") or ""
            ).strip()
            if completed_state:
                completed_states.append(completed_state)

        for state in completed_states:
            if state not in intermediate_states:
                return state
        return completed_states[-1] if completed_states else None

    def _derive_part_tracker_from_violations(
        self,
        violations: list[dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        derived: dict[str, dict[str, Any]] = {}
        for violation in violations or []:
            if not isinstance(violation, dict):
                continue
            failure_context = violation.get("failure_context")
            if not isinstance(failure_context, dict):
                continue
            observations = failure_context.get("observations")
            observations = observations if isinstance(observations, dict) else {}
            failed_task_id = str(
                violation.get("failed_task_id") or violation.get("task_id") or ""
            ).strip()
            failed_resource_jid = str(
                violation.get("resource_jid") or violation.get("failed_resource_jid") or ""
            ).strip()
            if not failed_resource_jid and failed_task_id:
                failed_resource_jid = self._resource_jid_for_task_id(failed_task_id)
            pose_candidate = None
            for key in ("observed_pose", "pose", "dropped_location", "last_known_position"):
                pose_candidate = self._coerce_xyz_pose(observations.get(key))
                if pose_candidate is not None:
                    break
            state_before = observations.get("state_before")
            state_before = state_before if isinstance(state_before, dict) else {}
            state_after = observations.get("state_after")
            state_after = state_after if isinstance(state_after, dict) else {}
            before_held = str(state_before.get("held_part") or "").strip()
            after_held = str(state_after.get("held_part") or "").strip()
            released_part_name = before_held if before_held and not after_held else ""
            observed_part_state = str(observations.get("part_state") or "").strip()
            affected_entities = failure_context.get("affected_entities")
            if not isinstance(affected_entities, list):
                affected_entities = []
            if not affected_entities and released_part_name:
                affected_entities = [
                    {
                        "entity_type": "part",
                        "entity_id": released_part_name,
                        "state": observed_part_state or "unknown",
                    }
                ]
            for entity in affected_entities:
                if not isinstance(entity, dict):
                    continue
                if str(entity.get("entity_type") or "").strip().lower() != "part":
                    continue
                part_name = str(entity.get("entity_id") or "").strip()
                if not part_name:
                    continue
                entry = derived.setdefault(part_name, {"state": "unknown", "location": None})
                force_keys = {
                    str(key).strip()
                    for key in (entry.get("_force_keys") or [])
                    if str(key).strip()
                }
                state = observed_part_state or str(entity.get("state") or "").strip()
                if state:
                    entry["state"] = state
                    if state.lower() in {"unknown", "untracked", "misplaced"}:
                        force_keys.add("state")
                location = entity.get("location")
                if isinstance(location, str) and location.strip():
                    entry["location"] = location.strip()
                if pose_candidate is not None:
                    entry["observed_pose"] = deepcopy(pose_candidate)
                    entry["location"] = None
                    force_keys.add("location")
                if observations.get("observation_required") or (
                    released_part_name == part_name and pose_candidate is None
                ):
                    entry["observation_required"] = True
                if failed_resource_jid and released_part_name == part_name:
                    entry["location"] = None
                    entry["last_known_location"] = (
                        entry.get("last_known_location") or f"{failed_resource_jid}_gripper"
                    )
                    force_keys.add("location")
                if force_keys:
                    entry["_force_keys"] = sorted(force_keys)
        return derived

    def _apply_replan_patch(self, modified_tasks: list[dict]) -> None:
        """
        Merge task modifications into self.nodes.

        Supports MODIFICATION, INSERTION, and DELETION.
        Called by both PDDL replanner and pure-LLM replanner paths.
        """
        original_nodes = deepcopy(self.nodes)
        node_map = {n["id"]: n for n in original_nodes}

        for t in modified_tasks:
            tid = t.get("id")
            if not tid:
                continue

            # CASE A: DELETION
            if t.get("delete") is True:
                if tid in node_map:
                    self.logger.info(f"[Planner] DELETING task {tid}: {t.get('change_reason')}")
                    del node_map[tid]
                    for other in node_map.values():
                        if tid in other.get("predecessors", []):
                            other["predecessors"].remove(tid)
                        if tid in other.get("successors", []):
                            other["successors"].remove(tid)
                continue

            # CASE B: MODIFICATION / INSERTION
            params = t.get("params") or {}
            if tid in node_map and not t.get("params"):
                params = node_map[tid].get("params", {})

            params["product_jid"] = str(self.product_agent.jid)

            if tid not in node_map:
                node_map[tid] = {
                    "id": tid,
                    "type": "task",
                    "status": "pending",
                    "predecessors": [],
                    "successors": [],
                }

            target = node_map[tid]

            if "function_name" in t: target["function_name"] = t["function_name"]
            if "params" in t: target["params"] = params
            if "resource_jid" in t: target["resource_jid"] = t["resource_jid"]
            if "sequence_index" in t: target["sequence_index"] = t["sequence_index"]

            if "predecessors" in t:
                target["predecessors"] = t["predecessors"]

            if "successors" in t and tid not in node_map:
                target["successors"] = t["successors"]

            if "change_reason" in t:
                target["change_reason"] = t["change_reason"]
                self.logger.info(f"[Planner] Applied fix to {tid}: {t['change_reason']}")

            for extra_key in (
                "part_name",
                "in_state",
                "out_state",
                "required_context_keys",
                "context_mapping",
                "part_transition",
                "primary_obligation",
                "bridge_sequence_id",
                "bridge_sequence_index",
                "bridge_sequence_length",
                "projected_snapshot",
                "projected_part_entry",
            ):
                if extra_key in t:
                    target[extra_key] = deepcopy(t[extra_key])

            target["status"] = "pending"

        tentative_nodes = list(node_map.values())
        self._ensure_graph_consistency(tentative_nodes)
        self._validate_task_graph(tentative_nodes)

        self.nodes = tentative_nodes

        self.logger.info(
            "[Planner] Re-planning successful. Merged %d modifications.",
            len(modified_tasks),
        )

        if hasattr(self.product_agent, "plan_path"):
            self.save(self.product_agent.plan_path)

        self.global_fsa = None
        self.save_global_fsa(self.product_agent.global_fsa_path)

    # ------------------------------------------------------------------ #
    # Prompt helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _deduplicate_tools_catalog(catalog: list) -> list:
        """
        Merge per-resource tool entries into one entry per function.
        function_owner_agent is replaced by capable_agents: [resource1, resource2, ...]
        so the LLM sees each capability once and knows which resources can execute it.
        """
        seen: Dict[str, Dict[str, Any]] = {}
        for row in catalog:
            fn = row.get("function")
            if not fn:
                continue
            if fn not in seen:
                merged = {k: v for k, v in row.items() if k != "function_owner_agent"}
                merged["capable_agents"] = [row["function_owner_agent"]] if row.get("function_owner_agent") else []
                seen[fn] = merged
            else:
                agent = row.get("function_owner_agent")
                if agent and agent not in seen[fn]["capable_agents"]:
                    seen[fn]["capable_agents"].append(agent)
        return list(seen.values())

    # ------------------------------------------------------------------ #
    # Debug helpers
    # ------------------------------------------------------------------ #
    def _dump_replan_debug(
        self,
        *,
        source: str,
        prompt: str,
        violations: list,
        resource_infos: list,
        system_state: dict | None,
        llm_response: str | None = None,
    ) -> None:
        """Write a timestamped Markdown report to cais_spade_llm/monitor/debug/ for each replan."""
        try:
            debug_dir = Path("cais_spade_llm/monitor/debug")
            debug_dir.mkdir(parents=True, exist_ok=True)

            ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
            suffix = "response" if llm_response is not None else "input"
            fname = debug_dir / f"replan_{source}_{ts}_{suffix}.md"

            def _json_block(obj: Any) -> str:
                return "```json\n" + json.dumps(obj, indent=2, default=str) + "\n```"

            lines = [
                f"# Replan Debug — {source.upper()} | {datetime.now(timezone.utc).isoformat()}",
                "",
                "---",
                "",
                "## Violations (what triggered the replan)",
                "",
                _json_block(violations),
                "",
                "## Resource Agents (capabilities available to LLM)",
                "",
                _json_block(resource_infos),
                "",
                "## System State (runtime context: robot states, part tracker, timeline)",
                "",
                _json_block(system_state) if system_state else "_No system state (offline replan)._",
                "",
                "## Prompt (full text sent to LLM)",
                "",
                "```",
                prompt,
                "```",
                "",
            ]

            if llm_response is not None:
                lines += [
                    "## LLM Response (raw)",
                    "",
                    "```",
                    llm_response,
                    "```",
                    "",
                    "## LLM Response (parsed)",
                    "",
                ]
                try:
                    lines.append(_json_block(json.loads(llm_response)))
                except json.JSONDecodeError:
                    lines.append("_Response was not valid JSON._")
                lines.append("")

            fname.write_text("\n".join(lines), encoding="utf-8")
            self.logger.info("[Planner] Debug report written to %s", fname)
        except Exception:
            self.logger.exception("[Planner] Failed to write replan debug report.")

    # ------------------------------------------------------------------ #
    # Scheduling helpers
    # ------------------------------------------------------------------ #
    def _extract_conflict_task_ids(self, violations: list[dict], *, source: str) -> Set[str]:
        """Normalize offline/online feedback into a conflict task ID set."""
        conflict_task_ids: Set[str] = set()

        def _add_ids(values: Any) -> None:
            if isinstance(values, (list, tuple, set)):
                for v in values:
                    if v:
                        conflict_task_ids.add(str(v))
            elif isinstance(values, str) and values:
                conflict_task_ids.add(values)

        for v in violations:
            if not isinstance(v, dict):
                continue

            if source == "offline":
                _add_ids(v.get("witness_trace"))
                _add_ids(v.get("witness_task_ids"))
                _add_ids(v.get("conflict_task_ids"))

                witness_transitions = v.get("witness_transitions") or []
                for t in witness_transitions:
                    tid = t.get("task_id") if isinstance(t, dict) else None
                    if tid:
                        conflict_task_ids.add(str(tid))

                rel_tasks = v.get("affected_tasks", [])
                if isinstance(rel_tasks, list):
                    for rt in rel_tasks:
                        if isinstance(rt, dict) and "id" in rt:
                            conflict_task_ids.add(str(rt["id"]))
            else:
                _add_ids(v.get("failed_task_id"))
                _add_ids(v.get("task_id"))
                _add_ids(v.get("blocked_task_ids"))
                _add_ids(v.get("unreachable_task_ids"))
                _add_ids(v.get("impact_task_ids"))
                _add_ids(v.get("affected_task_ids"))

                rel_tasks = v.get("affected_tasks", [])
                if isinstance(rel_tasks, list):
                    for rt in rel_tasks:
                        if isinstance(rt, dict) and "id" in rt:
                            conflict_task_ids.add(str(rt["id"]))
                dep_tasks = v.get("dependent_tasks", [])
                if isinstance(dep_tasks, list):
                    for rt in dep_tasks:
                        if isinstance(rt, dict) and "id" in rt:
                            conflict_task_ids.add(str(rt["id"]))

        return conflict_task_ids
    def _ensure_graph_consistency(self, nodes: Optional[List[Dict[str, Any]]] = None) -> None:
        """
        Helper to ensure that if A lists B as a predecessor, 
        B lists A as a successor (and vice versa).
        This fixes 'one-sided' edits from the LLM.
        """
        working_nodes = nodes if nodes is not None else self.nodes
        node_map = {n["id"]: n for n in working_nodes}

        for nid, node in node_map.items():
            # 1. Sync Predecessors -> Successors
            # If 'node' thinks 'pid' is a predecessor, make sure 'pid' knows 'node' is a successor.
            preds = list(dict.fromkeys(node.get("predecessors", []) or []))
            node["predecessors"] = preds
            for pid in preds:
                if pid in node_map:
                    p_node = node_map[pid]
                    if nid not in p_node.get("successors", []):
                        p_node.setdefault("successors", []).append(nid)
            
            # 2. Sync Successors -> Predecessors
            # If 'node' thinks 'sid' is a successor, make sure 'sid' knows 'node' is a predecessor.
            succs = list(dict.fromkeys(node.get("successors", []) or []))
            node["successors"] = succs
            for sid in succs:
                if sid in node_map:
                    s_node = node_map[sid]
                    if nid not in s_node.get("predecessors", []):
                        s_node.setdefault("predecessors", []).append(nid)

    def _validate_task_graph(self, nodes: Optional[List[Dict[str, Any]]] = None) -> None:
        """Reject invalid task graphs before FSA compilation or execution."""
        working_nodes = nodes if nodes is not None else self.nodes
        tasks = [n for n in working_nodes if n.get("type") == "task"]
        node_map = {n["id"]: n for n in tasks}
        successors: Dict[str, List[str]] = {nid: [] for nid in node_map}
        indegree: Dict[str, int] = {nid: 0 for nid in node_map}

        for nid, node in node_map.items():
            for pid in node.get("predecessors", []) or []:
                if pid not in node_map:
                    raise ValueError(
                        f"Task graph references unknown predecessor '{pid}' from task '{nid}'."
                    )
                if pid == nid:
                    raise ValueError(f"Task graph contains a self-dependency on '{nid}'.")
                if nid not in successors[pid]:
                    successors[pid].append(nid)
                    indegree[nid] += 1

        q = deque(sorted(nid for nid, degree in indegree.items() if degree == 0))
        visited: List[str] = []

        while q:
            nid = q.popleft()
            visited.append(nid)
            for sid in successors[nid]:
                indegree[sid] -= 1
                if indegree[sid] == 0:
                    q.append(sid)

        if len(visited) == len(node_map):
            return

        cycle = self._extract_task_cycle(node_map, successors)
        if cycle:
            cycle_text = " -> ".join(cycle)
            raise ValueError(f"Task graph must remain a DAG; cycle detected: {cycle_text}")

        remaining = sorted(nid for nid, degree in indegree.items() if degree > 0)
        raise ValueError(
            "Task graph must remain a DAG; unresolved cyclic dependency among tasks: "
            + ", ".join(remaining)
        )

    def _extract_task_cycle(
        self,
        node_map: Dict[str, Dict[str, Any]],
        successors: Dict[str, List[str]],
    ) -> List[str]:
        """Return one cycle path for diagnostics, e.g. A -> B -> A."""
        color: Dict[str, int] = {nid: 0 for nid in node_map}
        stack: List[str] = []
        stack_index: Dict[str, int] = {}

        def _dfs(nid: str) -> List[str]:
            color[nid] = 1
            stack_index[nid] = len(stack)
            stack.append(nid)

            for sid in successors.get(nid, []):
                if color[sid] == 0:
                    cycle = _dfs(sid)
                    if cycle:
                        return cycle
                elif color[sid] == 1:
                    start = stack_index[sid]
                    return stack[start:] + [sid]

            stack.pop()
            stack_index.pop(nid, None)
            color[nid] = 2
            return []

        for nid in node_map:
            if color[nid] == 0:
                cycle = _dfs(nid)
                if cycle:
                    return cycle
        return []

    def _find_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Return the first node with the matching id (or None)."""
        for n in self.nodes:
            if n.get("id") == node_id:
                return n
        return None

    def next_ready_task(self) -> Optional[Dict[str, Any]]:
        """Select the next task whose predecessors are all satisfied."""
        def _pred_satisfied(status: Any) -> bool:
            s = str(status) if status else ""
            return s == "completed"

        def _pred_ready(pred_id: str) -> bool:
            pred = self._find_node(pred_id)
            if pred is None:
                return False
            return _pred_satisfied(pred.get("status"))

        for node in self.nodes:
            if node.get("type") != "task":
                continue
            if node.get("status") != "pending":
                continue

            preds = node.get("predecessors", [])
            if not preds:
                return node

            if all(_pred_ready(pid) for pid in preds):
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
        self.logger.debug(f"[Planner] Saved plan to {p.resolve()}")

    def load(self, path: Path | str) -> None:
        p = Path(path)
        if not p.exists():
            self.logger.warning(f"[Planner] Plan file missing: {p}")
            return
        with p.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        self.nodes = payload.get("nodes", [])
        self.logger.debug(f"[Planner] Loaded plan from {p.resolve()}")

    # ------------------------------------------------------------------ #
    # Build Global FSA
    # ------------------------------------------------------------------ #
    def compile_global_fsa(self) -> Dict[str, Any]:
        """Compile the task DAG into a global plan FSA for offline/online monitoring."""

        # ---- extract task nodes ----
        tasks = [n for n in self.nodes if n.get("type") == "task"]
        if not tasks:
            raise ValueError("No task nodes exist in self.nodes (cannot compile FSA).")

        self._validate_task_graph(self.nodes)

        by_id = {t["id"]: t for t in tasks}
        tools_catalog = list(getattr(self.product_agent, "tools_catalog", []) or [])

        def _resource_short_name(value: str) -> str:
            token = str(value or "").strip()
            if "@" in token:
                token = token.split("@", 1)[0]
            return token.lower()

        tool_meta_by_resource_fn: Dict[tuple[str, str], dict[str, Any]] = {}
        fallback_tool_meta_by_fn: Dict[str, dict[str, Any]] = {}
        for row in tools_catalog:
            if not isinstance(row, dict):
                continue
            fn = str(row.get("function", "")).strip()
            if not fn:
                continue
            owner = _resource_short_name(str(row.get("function_owner_agent", "")).strip())
            if owner:
                tool_meta_by_resource_fn.setdefault((owner, fn), row)
            fallback_tool_meta_by_fn.setdefault(fn, row)

        def _tool_meta_for_task(task: dict[str, Any]) -> dict[str, Any]:
            """Look up tool metadata, preferring node-level in_state/out_state for bridge macros."""
            resource = _resource_short_name(str(task.get("resource_jid", "")).strip())
            fn = str(task.get("function_name", "")).strip()
            catalog_meta = (
                tool_meta_by_resource_fn.get((resource, fn))
                or fallback_tool_meta_by_fn.get(fn)
                or {}
            )
            # Prefer node-level metadata (set by bridge macro proposals)
            # over shared-catalog metadata.
            if task.get("in_state") or task.get("out_state"):
                merged = dict(catalog_meta)
                if task.get("in_state"):
                    merged["in_state"] = task["in_state"]
                if task.get("out_state"):
                    merged["out_state"] = task["out_state"]
                return merged
            return catalog_meta

        # ---- deterministic per-resource local order ----
        res_to_tasks: Dict[str, List[str]] = defaultdict(list)
        for t in tasks:
            res = t.get("resource_jid")
            if not res:
                raise ValueError(f"Task {t.get('id')} missing resource_jid.")
            res_to_tasks[res].append(t["id"])

        def sort_key(tid: str):
            si = by_id[tid].get("sequence_index", None)
            return (10**9 if si is None else int(si), tid)

        for res in res_to_tasks:
            res_to_tasks[res].sort(key=sort_key)

        resources = sorted(res_to_tasks.keys())

        pos: Dict[str, Dict[str, int]] = {
            res: {tid: i for i, tid in enumerate(res_to_tasks[res])}
            for res in resources
        }

        # ---- labeling helpers (NEW) ----
        def task_sig(tid: str) -> str:
            """
            Human-readable task signature.
            Example: 'pick_grasp(SG)' or 'place_approach(MCP)'.
            """
            t = by_id[tid]
            fn = t.get("function_name") or "unknown_fn"
            params = t.get("params") or {}

            # Prefer showing the part if present (often the most informative)
            part = params.get("part_name")
            if part:
                return f"{fn}({part})"
            return fn

        def readable_event(tid: str, phase: str) -> str:
            """
            Example: 'pick_grasp(SG).start' / 'pick_grasp(SG).done'
            """
            return f"{task_sig(tid)}.{phase}"

        # ---- helpers ----
        def _idx(res: str) -> int:
            return resources.index(res)

        def _get_local(x, res):
            return x[_idx(res)]

        def _set_local(x, res, new_local):
            xl = list(x)
            xl[_idx(res)] = new_local
            return tuple(xl)

        def is_completed(x, task_id):
            t = by_id[task_id]
            res = t["resource_jid"]
            k, run = _get_local(x, res)

            # task position in that resource's local list
            task_pos = pos[res][task_id]

            # done iff k has advanced past task_pos AND we aren't currently running it
            return (k > task_pos) and not (run == task_pos)

        def next_local_task_if_any(x, res):
            k, run = _get_local(x, res)
            local = res_to_tasks[res]
            return None if k >= len(local) else local[k]

        def is_next_task_enabled(x, tid):
            preds = by_id[tid].get("predecessors", []) or []
            for p in preds:
                if p not in by_id:
                    return False

                # If predecessor is still running, block start
                t = by_id[p]
                res = t["resource_jid"]
                k, run = _get_local(x, res)
                if run is not None and res_to_tasks[res][run] == p:
                    return False

                # If predecessor not completed yet, block start
                if not is_completed(x, p):
                    return False

            return True


        # ---- initial / marked ----
        x0 = tuple((0, None) for _ in resources)
        x_marked = tuple((len(res_to_tasks[r]), None) for r in resources)

        def state_name(x):
            """
            Make states readable by showing which task is running (by function_name + part)
            instead of run index.
            """
            parts = []
            for i, res in enumerate(resources):
                k, run = x[i]
                if run is None:
                    parts.append(f"{res}=(k={k},idle)")
                else:
                    tid = res_to_tasks[res][run]
                    fn = by_id[tid].get("function_name") or "unknown_fn"
                    parts.append(f"{res}=(k={k},run={tid}:{fn})")
            return "(" + ",".join(parts) + ")"

        # ---- BFS ----
        visited = {x0}
        q = deque([x0])
        name_map = {x0: state_name(x0)}
        transitions = []

        while q:
            x = q.popleft()
            sx = name_map[x]

            for res in resources:
                k, run = _get_local(x, res)

                # IDLE → start
                if run is None:
                    tid = next_local_task_if_any(x, res)
                    if tid and is_next_task_enabled(x, tid):
                        e = f"{tid}.start"
                        x_next = _set_local(x, res, (k, pos[res][tid]))

                        if x_next not in visited:
                            visited.add(x_next)
                            name_map[x_next] = state_name(x_next)
                            q.append(x_next)

                        transitions.append({
                            "from": sx,
                            "event": e,  # canonical
                            "readable_event": readable_event(tid, "start"),
                            "to": name_map[x_next],
                            "resource_jid": res,
                            "task_id": tid,
                            "function_name": by_id[tid].get("function_name"),
                            "params": dict(by_id[tid].get("params") or {}),
                            "in_state": _tool_meta_for_task(by_id[tid]).get("in_state"),
                            "out_state": _tool_meta_for_task(by_id[tid]).get("out_state"),
                        })

                # RUNNING → done
                else:
                    tid = res_to_tasks[res][run]
                    e = f"{tid}.done"
                    x_done = _set_local(x, res, (k + 1, None))

                    if x_done not in visited:
                        visited.add(x_done)
                        name_map[x_done] = state_name(x_done)
                        q.append(x_done)

                    transitions.append({
                        "from": sx,
                        "event": e,  # canonical
                        "readable_event": readable_event(tid, "done"),
                        "to": name_map[x_done],
                        "resource_jid": res,
                        "task_id": tid,
                        "function_name": by_id[tid].get("function_name"),
                        "params": dict(by_id[tid].get("params") or {}),
                        "in_state": _tool_meta_for_task(by_id[tid]).get("in_state"),
                        "out_state": _tool_meta_for_task(by_id[tid]).get("out_state"),
                    })

        if x_marked not in name_map:
            raise ValueError(
                "Global FSA has no reachable marked state. "
                "The current task graph likely contains a cycle or an unreachable dependency."
            )

        fsa = {
            "A": {
                "X": sorted(name_map.values()),
                "E": sorted({t["event"] for t in transitions}),
                "Tr": transitions,
                "x0": name_map[x0],
                "Xm": [name_map[x_marked]],
            },
            "meta": {
                "resources": resources,
                "num_reachable_states": len(visited),
                "num_transitions": len(transitions),
                "has_failure_states": False,
            },
        }

        self.global_fsa = fsa        # ← STORE IT
        return fsa

    def save_global_fsa(self, path: Path | str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)

        if self.global_fsa is None:
            self.compile_global_fsa()

        with p.open("w", encoding="utf-8") as f:
            json.dump(self.global_fsa, f, indent=2)

        self.logger.debug(f"[Planner] Saved global FSA to {p.resolve()}")

    def load_global_fsa(self, path: Path | str) -> None:
        p = Path(path)
        if not p.exists():
            self.logger.warning(f"[Planner] Global FSA file missing: {p}")
            return
        with p.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, dict):
            self.logger.warning(f"[Planner] Invalid global FSA payload in {p}")
            return
        self.global_fsa = payload
        self.logger.debug(f"[Planner] Loaded global FSA from {p.resolve()}")

    # ------------------------------------------------------------------ #
    # v2 Universal Repair — apply validated program to live graph
    # ------------------------------------------------------------------ #

    def apply_repair_program(
        self,
        validated_program: dict[str, Any],
        *,
        prepared_bridge_request: dict[str, Any] | None = None,
    ) -> tuple[bool, list[str]]:
        """Apply a validated RepairProgram to the live task graph.

        This is the execution bridge between the v2 universal repair session
        and the existing ``_apply_replan_patch()`` + ``compile_global_fsa()``
        infrastructure.

        Parameters
        ----------
        validated_program:
            Serialized :class:`ValidatedRepairProgram` dict (from
            ``validated_program_to_dict``).
        prepared_bridge_request:
            Optional bridge request for context (resource JIDs, etc.).

        Returns
        -------
        tuple:
            ``(success, errors)`` — if *errors* is non-empty the program
            was not applied.
        """
        from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.mutation_compiler import (
            compile_mutations,
        )
        from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.mutation_types import (
            RepairStepKind,
            TaskMutationStep,
            TaskMutationType,
            repair_program_from_dict,
        )
        from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.function_synthesis import (
            compile_synthesized_function_to_macro,
        )

        errors: list[str] = []
        program_dict = validated_program.get("program") or {}
        if not program_dict:
            errors.append("validated_program has no 'program' field")
            return False, errors

        if not validated_program.get("is_valid", False):
            errors.append("program is not valid (is_valid=False)")
            return False, errors

        try:
            program = repair_program_from_dict(program_dict)
        except Exception as exc:
            errors.append(f"failed to deserialize repair program: {exc}")
            return False, errors

        # Build function lookup.
        fn_defs = {fn.name: fn for fn in program.function_defs}

        # Collect all patches from mutations and function calls.
        all_patches: list[dict[str, Any]] = []
        macro_tasks: list[dict[str, Any]] = []

        for step_idx, step in enumerate(program.steps):
            kind = step.kind

            if kind == RepairStepKind.TASK_MUTATION:
                mutation_payload = step.payload
                mutation_type_str = str(
                    mutation_payload.get("mutation_type", "")
                ).strip()
                try:
                    mutation_type = TaskMutationType(mutation_type_str)
                except ValueError:
                    errors.append(
                        f"step {step_idx}: invalid mutation_type '{mutation_type_str}'"
                    )
                    continue

                mutation_step = TaskMutationStep(
                    mutation_type=mutation_type,
                    target_task_ids=list(
                        mutation_payload.get("target_task_ids") or []
                    ),
                    payload=dict(mutation_payload.get("payload") or {}),
                )
                patches, mut_errors = compile_mutations(
                    [mutation_step],
                    self.nodes,
                )
                if mut_errors:
                    errors.extend(
                        f"step {step_idx} mutation: {e}" for e in mut_errors
                    )
                else:
                    all_patches.extend(patches)

            elif kind == RepairStepKind.CALL_FUNCTION:
                fn_name = str(step.payload.get("function_name", "")).strip()
                resource_jid = str(step.payload.get("resource_jid", "")).strip()
                fn_def = fn_defs.get(fn_name)
                if fn_def is None:
                    errors.append(
                        f"step {step_idx}: function '{fn_name}' not found in function_defs"
                    )
                    continue

                macro = compile_synthesized_function_to_macro(
                    fn_def,
                    resource_jid=resource_jid,
                    macro_index=step_idx,
                )
                macro_tasks.append(macro)

            elif kind == RepairStepKind.RESUME_SUFFIX:
                # No action needed — suffix tasks are already in the graph.
                pass

            elif kind == RepairStepKind.WAIT:
                # Wait steps are runtime directives, not graph mutations.
                # They will be handled by the execution engine.
                pass

        if errors:
            return False, errors

        # Apply patches to the live graph.
        if all_patches:
            try:
                self._apply_replan_patch(all_patches)
            except Exception as exc:
                errors.append(f"failed to apply patches: {exc}")
                return False, errors

        # Convert macro_tasks to patch format and insert into the graph.
        if macro_tasks:
            macro_patches: list[dict[str, Any]] = []
            for macro in macro_tasks:
                task_id = f"repair_fn_{uuid4().hex[:8]}"
                patch = {
                    "id": task_id,
                    "type": "task",
                    "function_name": macro.get("macro_name", ""),
                    "resource_jid": macro.get("resource_jid", ""),
                    "params": deepcopy(macro.get("primitive_steps") or []),
                    "status": "pending",
                    "change_reason": f"v2 repair: {macro.get('description', '')}",
                }
                task_metadata = macro.get("task_metadata") or {}
                if task_metadata.get("in_state"):
                    patch["in_state"] = task_metadata["in_state"]
                if task_metadata.get("out_state"):
                    patch["out_state"] = task_metadata["out_state"]
                if task_metadata.get("part_transition"):
                    patch["part_transition"] = deepcopy(task_metadata["part_transition"])
                if macro.get("bridge_sequence_id"):
                    patch["bridge_sequence_id"] = macro["bridge_sequence_id"]
                    patch["bridge_sequence_index"] = macro.get("bridge_sequence_index", 0)
                macro_patches.append(patch)

            if macro_patches:
                try:
                    self._apply_replan_patch(macro_patches)
                except Exception as exc:
                    errors.append(f"failed to apply macro patches: {exc}")
                    return False, errors

        # Recompile FSA.
        try:
            self.compile_global_fsa()
            if hasattr(self.product_agent, "global_fsa_path"):
                self.save_global_fsa(self.product_agent.global_fsa_path)
        except Exception as exc:
            errors.append(f"FSA recompilation failed: {exc}")
            return False, errors

        self.logger.info(
            "[Planner] v2 repair program applied: %d patches, %d macros",
            len(all_patches), len(macro_tasks),
        )
        return True, []
