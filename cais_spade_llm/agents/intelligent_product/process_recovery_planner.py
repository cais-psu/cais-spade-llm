"""Runtime DES recovery and bridge-plan mutation helpers for ProcessPlanner."""

from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from cais_spade_llm.prompts import build_replan_prompt


class ProcessRecoveryPlanner:
    EXPORTED_METHODS = (
        '_build_des_replan_result',
        '_set_last_bridge_debug',
        'get_last_bridge_debug',
        '_tool_signature',
        '_resource_by_jid',
        '_resource_short_name',
        '_tool_row_for_task',
        '_pending_resource_tasks',
        '_reconcile_search_state_for_task',
        '_entry_task_ids_from_violations',
        '_coerce_xyz_pose',
        '_identify_stuck_resource',
        '_resource_jid_for_task_id',
        '_function_name_for_task_id',
        '_extract_resource_states',
        '_default_resource_state',
        '_build_resource_search_state',
        '_project_resource_suffix_state',
        '_collect_obligation_targets',
        '_primary_failure_context',
        '_node_exists',
        '_bridge_sequence_nodes',
        'remove_bridge_sequence_tail',
        '_path_to_recovery_tasks',
        '_gate_tasks_after_recovery_tail',
        '_splice_runtime_des_repair_before_task',
        'apply_bridge_macro_proposal',
        '_apply_primitive_bridge_proposal',
        'replan_with_feedback_offline',
        'replan_with_feedback_online',
        'replan_with_feedback_des',
        '_resolve_goal_part_state',
        '_derive_part_tracker_from_violations',
        '_apply_replan_patch',
        '_deduplicate_tools_catalog',
        '_dump_replan_debug',
    )

    def __init__(self, planner: Any) -> None:
        object.__setattr__(self, "_planner", planner)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._planner, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "_planner":
            object.__setattr__(self, name, value)
            return
        setattr(self._planner, name, value)

    def bind_methods(self) -> None:
        for name in self.EXPORTED_METHODS:
            setattr(self._planner, name, getattr(self, name))

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

    def _bridge_sequence_nodes(self, bridge_sequence_id: str) -> list[dict[str, Any]]:
        """Return bridge macro tasks for a compiled bridge sequence in execution order."""
        sequence_id = str(bridge_sequence_id or "").strip()
        if not sequence_id:
            return []
        nodes = [
            node
            for node in self.nodes
            if isinstance(node, dict)
            and str(node.get("bridge_sequence_id") or "").strip() == sequence_id
        ]
        return sorted(
            nodes,
            key=lambda node: (
                int(node.get("bridge_sequence_index") or 0),
                int(node.get("sequence_index") or 0),
                str(node.get("id") or ""),
            ),
        )

    def remove_bridge_sequence_tail(
        self,
        *,
        bridge_sequence_id: str,
        completed_task_id: str,
    ) -> list[dict[str, Any]]:
        """Delete remaining bridge macro tasks that have not started executing."""
        completed_task_id = str(completed_task_id or "").strip()
        deletions: list[dict[str, Any]] = []
        for node in self._bridge_sequence_nodes(bridge_sequence_id):
            node_id = str(node.get("id") or "").strip()
            if not node_id or node_id == completed_task_id:
                continue
            status = str(node.get("status") or "").strip().lower()
            if status in {"accepted", "running", "dispatched", "completed", "finished"}:
                continue
            deletions.append(
                {
                    "id": node_id,
                    "delete": True,
                    "change_reason": (
                        "Removed unexecuted bridge sequence tail after "
                        f"{completed_task_id or 'bridge failure'}"
                    ),
                }
            )
        if deletions:
            self._apply_replan_patch(deletions)
        return deletions

    def _bridge_resume_task_ids_by_resource(
        self,
        task_ids_by_resource: dict[str, list[str]] | None,
    ) -> dict[str, list[str]]:
        normalized: dict[str, list[str]] = {}
        for raw_resource_jid, raw_task_ids in (task_ids_by_resource or {}).items():
            resource_jid = str(raw_resource_jid or "").strip()
            if not resource_jid:
                continue
            ordered_task_ids: list[str] = []
            seen: set[str] = set()
            for raw_task_id in raw_task_ids or []:
                task_id = str(raw_task_id or "").strip()
                if not task_id or task_id in seen:
                    continue
                node = self._find_node(task_id)
                if not isinstance(node, dict):
                    continue
                if str(node.get("resource_jid") or "").strip() != resource_jid:
                    continue
                status = str(node.get("status") or "").strip().lower()
                if status not in {"pending", "blocked"}:
                    continue
                ordered_task_ids.append(task_id)
                seen.add(task_id)
            if ordered_task_ids:
                normalized[resource_jid] = ordered_task_ids
        return normalized

    def _bridge_insertion_sequence_base_by_resource(
        self,
        *,
        bridge_resource_jids: list[str],
        resume_task_ids_by_resource: dict[str, list[str]] | None = None,
    ) -> dict[str, int]:
        base_by_resource: dict[str, int] = {}
        normalized_resume = self._bridge_resume_task_ids_by_resource(
            resume_task_ids_by_resource
        )
        global_max_sequence_index = max(
            (
                int(node.get("sequence_index") or 0)
                for node in self.nodes
                if isinstance(node, dict)
            ),
            default=0,
        )
        for resource_jid in [
            str(item or "").strip()
            for item in bridge_resource_jids
            if str(item or "").strip()
        ]:
            resume_task_ids = list(normalized_resume.get(resource_jid) or [])
            target_task: dict[str, Any] | None = None
            for task_id in resume_task_ids:
                node = self._find_node(task_id)
                if isinstance(node, dict):
                    target_task = node
                    break
            if target_task is None:
                candidate_nodes = sorted(
                    [
                        node
                        for node in self.nodes
                        if isinstance(node, dict)
                        and str(node.get("type") or "").strip() == "task"
                        and str(node.get("resource_jid") or "").strip() == resource_jid
                        and str(node.get("status") or "").strip().lower() in {"pending", "blocked"}
                    ],
                    key=lambda node: (
                        int(node.get("sequence_index") or 0),
                        str(node.get("id") or ""),
                    ),
                )
                if candidate_nodes:
                    target_task = candidate_nodes[0]
            if isinstance(target_task, dict):
                try:
                    base_by_resource[resource_jid] = int(
                        target_task.get("sequence_index") or 0
                    )
                except (TypeError, ValueError):
                    base_by_resource[resource_jid] = 0
                continue

            resource_max_sequence_index = max(
                (
                    int(node.get("sequence_index") or 0)
                    for node in self.nodes
                    if isinstance(node, dict)
                    and str(node.get("resource_jid") or "").strip() == resource_jid
                ),
                default=global_max_sequence_index,
            )
            base_by_resource[resource_jid] = resource_max_sequence_index + 1
        return base_by_resource

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
                    "status": "pending",
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
        tail_task_id: str = "",
        tail_task_ids: list[str] | None = None,
        tail_task_ids_by_resource: dict[str, list[str]] | None = None,
        before_task_ids: list[str] | None = None,
        deleted_task_ids: list[str] | None = None,
        change_prefix: str = "DES recovery",
        recovery_tasks: list[dict[str, Any]] | None = None,
    ) -> None:
        gating_task_ids = [
            str(task_id or "").strip()
            for task_id in (tail_task_ids or [])
            if str(task_id or "").strip()
        ]
        if tail_task_id and tail_task_id not in gating_task_ids:
            gating_task_ids.append(str(tail_task_id).strip())
        gating_task_ids_by_resource = self._bridge_resume_task_ids_by_resource(
            tail_task_ids_by_resource
        )
        if not gating_task_ids and not gating_task_ids_by_resource:
            return
        deleted_task_id_set = {
            str(task_id or "").strip()
            for task_id in (deleted_task_ids or [])
            if str(task_id or "").strip()
        }
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

        recovery_task_rows = list(recovery_tasks or modified_tasks or [])
        inserted_highwater_by_resource: dict[str, int] = {}
        for task in recovery_task_rows:
            if not isinstance(task, dict) or task.get("delete") is True:
                continue
            resource_jid = str(task.get("resource_jid", "")).strip()
            if not resource_jid:
                continue
            try:
                sequence_index = int(task.get("sequence_index") or 0)
            except (TypeError, ValueError):
                continue
            inserted_highwater_by_resource[resource_jid] = max(
                inserted_highwater_by_resource.get(resource_jid, sequence_index),
                sequence_index,
            )
        next_sequence_index_by_resource = {
            resource_jid: highwater + 1
            for resource_jid, highwater in inserted_highwater_by_resource.items()
        }
        reorder_logs: list[tuple[str, str, int, int, int]] = []

        for blocked_task_id in sorted(ordered_task_ids, key=_sort_key):
            existing = existing_by_id[blocked_task_id]
            blocked_resource_jid = str(existing.get("resource_jid", "")).strip()
            resource_gating_task_ids = list(
                gating_task_ids_by_resource.get(blocked_resource_jid) or gating_task_ids
            )
            if not resource_gating_task_ids:
                continue
            existing_preds = [
                str(pred).strip()
                for pred in (existing.get("predecessors") or [])
                if str(pred or "").strip()
                and str(pred).strip() not in deleted_task_id_set
            ]
            gated_by_resume_chain = any(pred in resume_task_ids for pred in existing_preds)
            preds = (
                list(existing_preds)
                if gated_by_resume_chain
                else list(dict.fromkeys(existing_preds + resource_gating_task_ids))
            )
            mod: dict[str, Any] = {
                "id": blocked_task_id,
                "predecessors": preds,
                "change_reason": (
                    f"MODIFICATION: {change_prefix} — gate {blocked_task_id} after {resource_gating_task_ids}"
                ),
            }
            if blocked_resource_jid in next_sequence_index_by_resource:
                try:
                    old_sequence_index = int(existing.get("sequence_index") or 0)
                except (TypeError, ValueError):
                    old_sequence_index = 0
                resource_highwater = inserted_highwater_by_resource[blocked_resource_jid]
                next_sequence_index = next_sequence_index_by_resource[blocked_resource_jid]
                if old_sequence_index < next_sequence_index:
                    new_sequence_index = next_sequence_index
                    mod["sequence_index"] = new_sequence_index
                    next_sequence_index_by_resource[blocked_resource_jid] = new_sequence_index + 1
                    reorder_logs.append(
                        (
                            blocked_task_id,
                            blocked_resource_jid,
                            old_sequence_index,
                            new_sequence_index,
                            resource_highwater,
                        )
                    )
                else:
                    next_sequence_index_by_resource[blocked_resource_jid] = old_sequence_index + 1
            modified_tasks.append(mod)
        for (
            blocked_task_id,
            blocked_resource_jid,
            old_sequence_index,
            new_sequence_index,
            resource_highwater,
        ) in reorder_logs:
            self.logger.info(
                "[Planner] %s reordered survivor %s on %s after inserted recovery tasks: sequence_index %d -> %d (recovery_highwater=%d)",
                change_prefix,
                blocked_task_id,
                blocked_resource_jid,
                old_sequence_index,
                new_sequence_index,
                resource_highwater,
            )

    def _splice_runtime_des_repair_before_task(
        self,
        modified_tasks: list[dict[str, Any]],
        *,
        repair_task: dict[str, Any],
        target_task_id: str,
        change_prefix: str = "Runtime DES guard-restoration repair",
    ) -> None:
        repair_task_id = str((repair_task or {}).get("id") or "").strip()
        target_task_id = str(target_task_id or "").strip()
        if not repair_task_id:
            raise ValueError("repair task id is required")
        if not target_task_id:
            raise ValueError("target task id is required")

        target_task = self._find_node(target_task_id)
        if not isinstance(target_task, dict):
            raise ValueError(
                f"target task '{target_task_id}' was not found for runtime DES repair splice"
            )

        repair_resource_jid = str((repair_task or {}).get("resource_jid") or "").strip()
        target_resource_jid = str(target_task.get("resource_jid") or "").strip()
        if not repair_resource_jid:
            raise ValueError(
                f"repair task '{repair_task_id}' is missing resource_jid"
            )
        if not target_resource_jid:
            raise ValueError(
                f"target task '{target_task_id}' is missing resource_jid"
            )
        if repair_resource_jid != target_resource_jid:
            raise ValueError(
                f"repair task '{repair_task_id}' resource '{repair_resource_jid}' "
                f"does not match target task '{target_task_id}' resource '{target_resource_jid}'"
            )

        try:
            target_sequence_index = int(target_task.get("sequence_index") or 0)
        except (TypeError, ValueError):
            target_sequence_index = 0

        target_predecessors = [
            str(pred).strip()
            for pred in (target_task.get("predecessors") or [])
            if str(pred or "").strip()
        ]
        live_target_predecessors: list[str] = []
        skipped_completed_predecessors: list[str] = []
        for predecessor_task_id in target_predecessors:
            predecessor_task = self._find_node(predecessor_task_id)
            predecessor_status = (
                str((predecessor_task or {}).get("status") or "").strip().lower()
            )
            if predecessor_task is not None and predecessor_status in {"completed", "finished"}:
                skipped_completed_predecessors.append(predecessor_task_id)
                continue
            live_target_predecessors.append(predecessor_task_id)
        target_successors = [
            str(succ).strip()
            for succ in (target_task.get("successors") or [])
            if str(succ or "").strip()
        ]

        repair_patch = deepcopy(repair_task)
        repair_patch["predecessors"] = list(live_target_predecessors)
        repair_patch["successors"] = [target_task_id]
        repair_patch["sequence_index"] = target_sequence_index
        repair_patch["change_reason"] = (
            f"INSERTION: {change_prefix} — splice {repair_task_id} before {target_task_id}"
        )
        modified_tasks.append(repair_patch)

        predecessor_rewire_logs: list[tuple[str, list[str], list[str]]] = []
        for predecessor_task_id in live_target_predecessors:
            predecessor_task = self._find_node(predecessor_task_id)
            if not isinstance(predecessor_task, dict):
                continue
            old_successors = [
                str(succ).strip()
                for succ in (predecessor_task.get("successors") or [])
                if str(succ or "").strip()
            ]
            new_successors = [
                succ for succ in old_successors if succ != target_task_id
            ]
            if repair_task_id not in new_successors:
                new_successors.append(repair_task_id)
            modified_tasks.append(
                {
                    "id": predecessor_task_id,
                    "successors": new_successors,
                    "change_reason": (
                        f"MODIFICATION: {change_prefix} — reroute {predecessor_task_id} "
                        f"through {repair_task_id} before {target_task_id}"
                    ),
                }
            )
            predecessor_rewire_logs.append(
                (predecessor_task_id, old_successors, list(new_successors))
            )

        historical_edge_removal_logs: list[tuple[str, list[str], list[str]]] = []
        for predecessor_task_id in skipped_completed_predecessors:
            predecessor_task = self._find_node(predecessor_task_id)
            if not isinstance(predecessor_task, dict):
                continue
            old_successors = [
                str(succ).strip()
                for succ in (predecessor_task.get("successors") or [])
                if str(succ or "").strip()
            ]
            if target_task_id not in old_successors:
                continue
            new_successors = [
                succ for succ in old_successors if succ != target_task_id
            ]
            modified_tasks.append(
                {
                    "id": predecessor_task_id,
                    "successors": new_successors,
                    "change_reason": (
                        f"MODIFICATION: {change_prefix} — remove historical edge "
                        f"{predecessor_task_id} -> {target_task_id} after inserting {repair_task_id}"
                    ),
                }
            )
            historical_edge_removal_logs.append(
                (predecessor_task_id, old_successors, list(new_successors))
            )

        target_sort_key = (target_sequence_index, target_task_id)
        tasks_to_shift: list[tuple[tuple[int, str], dict[str, Any]]] = []
        for node in self.nodes:
            if not isinstance(node, dict):
                continue
            node_id = str(node.get("id") or "").strip()
            if not node_id or node_id == repair_task_id:
                continue
            if str(node.get("resource_jid") or "").strip() != target_resource_jid:
                continue
            try:
                node_sequence_index = int(node.get("sequence_index") or 0)
            except (TypeError, ValueError):
                node_sequence_index = 0
            status = str(node.get("status") or "").strip().lower()
            node_sort_key = (node_sequence_index, node_id)
            if node_id == target_task_id or (
                status in {"pending", "blocked"} and node_sort_key >= target_sort_key
            ):
                tasks_to_shift.append((node_sort_key, node))

        tasks_to_shift.sort(key=lambda row: row[0])
        next_sequence_index = target_sequence_index + 1
        sequence_shift_logs: list[tuple[str, int, int]] = []
        for _sort_key, node in tasks_to_shift:
            node_id = str(node.get("id") or "").strip()
            try:
                old_sequence_index = int(node.get("sequence_index") or 0)
            except (TypeError, ValueError):
                old_sequence_index = 0
            patch: dict[str, Any] = {
                "id": node_id,
                "sequence_index": next_sequence_index,
            }
            if node_id == target_task_id:
                patch["predecessors"] = [repair_task_id]
                patch["change_reason"] = (
                    f"MODIFICATION: {change_prefix} — gate {target_task_id} after {repair_task_id}"
                )
            else:
                patch["change_reason"] = (
                    f"MODIFICATION: {change_prefix} — shift {node_id} after inserted repair {repair_task_id}"
                )
            modified_tasks.append(patch)
            sequence_shift_logs.append(
                (node_id, old_sequence_index, next_sequence_index)
            )
            next_sequence_index += 1

        self.logger.info(
            "[Planner] %s spliced %s on %s before %s: predecessors=%s successors=%s sequence_index=%d",
            change_prefix,
            repair_task_id,
            target_resource_jid,
            target_task_id,
            live_target_predecessors,
            target_successors,
            target_sequence_index,
        )
        if skipped_completed_predecessors:
            self.logger.info(
                "[Planner] %s skipped completed predecessor(s) for %s before %s: %s",
                change_prefix,
                repair_task_id,
                target_task_id,
                skipped_completed_predecessors,
            )
        for predecessor_task_id, old_successors, new_successors in historical_edge_removal_logs:
            self.logger.info(
                "[Planner] %s removed historical successor edge from completed predecessor %s: %s -> %s",
                change_prefix,
                predecessor_task_id,
                old_successors,
                new_successors,
            )
        for predecessor_task_id, old_successors, new_successors in predecessor_rewire_logs:
            self.logger.info(
                "[Planner] %s rewired predecessor %s successors: %s -> %s",
                change_prefix,
                predecessor_task_id,
                old_successors,
                new_successors,
            )
        for node_id, old_sequence_index, new_sequence_index in sequence_shift_logs:
            if old_sequence_index == new_sequence_index:
                continue
            self.logger.info(
                "[Planner] %s shifted %s on %s: sequence_index %d -> %d",
                change_prefix,
                node_id,
                target_resource_jid,
                old_sequence_index,
                new_sequence_index,
            )

    def apply_bridge_macro_proposal(
        self,
        proposal: dict[str, Any],
        *,
        anchor_task_id: str = "",
        resume_task_ids_by_resource: dict[str, list[str]] | None = None,
    ) -> list[dict[str, Any]]:
        if not isinstance(proposal, dict):
            raise ValueError("bridge proposal is missing")

        # Primitive-based proposal: compile top-level primitive_steps or ordered macro_tasks[]
        # into one serial execute_recovery_macro chain.
        if self._primitive_bridge_macro_tasks(proposal):
            return self._apply_primitive_bridge_proposal(
                proposal,
                anchor_task_id=anchor_task_id,
                resume_task_ids_by_resource=resume_task_ids_by_resource,
            )

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
        resume_task_ids_by_resource: dict[str, list[str]] | None = None,
    ) -> list[dict[str, Any]]:
        """Compile one or more primitive-based bridge macro_tasks into ordered task nodes."""
        from uuid import uuid4

        macro_tasks = self._primitive_bridge_macro_tasks(proposal)
        if not macro_tasks:
            raise ValueError("bridge proposal has no primitive_steps/macro_tasks")

        compiled_nodes: list[dict[str, Any]] = []
        total_tasks = len(macro_tasks)
        primary_obligation = deepcopy(proposal.get("primary_obligation") or {})
        bridge_sequence_id = f"BRIDGESEQ_{uuid4().hex[:8].upper()}"
        anchor_predecessor = (
            str(anchor_task_id).strip() if self._node_exists(anchor_task_id) else ""
        )
        sequence_base_by_resource = self._bridge_insertion_sequence_base_by_resource(
            bridge_resource_jids=[
                str((macro_task or {}).get("resource_jid") or proposal.get("resource_jid") or "").strip()
                for macro_task in macro_tasks
                if isinstance(macro_task, dict)
            ],
            resume_task_ids_by_resource=resume_task_ids_by_resource,
        )
        next_sequence_index_by_resource = dict(sequence_base_by_resource)
        compiled_task_ids_by_outline_id: dict[str, str] = {}
        previous_outline_id = ""
        start_safety_mode = str(
            proposal.get("start_safety_mode") or "fast_path"
        ).strip().lower() or "fast_path"

        for index, macro_task in enumerate(macro_tasks, start=1):
            macro_name = str(
                macro_task.get("macro_name")
                or proposal.get("macro_name")
                or f"bridge_recovery_macro_{index}"
            ).strip()
            outline_id = str(
                macro_task.get("outline_id")
                or f"bridge_outline_{index}"
            ).strip()
            outline_predecessors = [
                str(item).strip()
                for item in (macro_task.get("predecessors") or [])
                if str(item).strip()
            ]
            if "predecessors" not in macro_task and previous_outline_id:
                outline_predecessors = [previous_outline_id]
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
                "outline_id": outline_id,
                "predecessors": deepcopy(outline_predecessors),
                "primitive_steps": primitive_steps,
                "expected_start_state": expected_start_state,
                "start_safety_mode": start_safety_mode,
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

            predecessors: list[str] = []
            if outline_predecessors:
                for dependency_outline_id in outline_predecessors:
                    dependency_task_id = compiled_task_ids_by_outline_id.get(
                        dependency_outline_id
                    )
                    if not dependency_task_id:
                        raise ValueError(
                            f"bridge macro_task {index} predecessors include unknown or later outline_id {dependency_outline_id!r}"
                        )
                    predecessors.append(dependency_task_id)
            elif anchor_predecessor:
                predecessors.append(anchor_predecessor)

            task_node: dict[str, Any] = {
                "id": task_id,
                "function_name": "execute_recovery_macro",
                "params": params,
                "resource_jid": resource_jid,
                "status": "pending",
                "predecessors": predecessors,
                "successors": [],
                "sequence_index": next_sequence_index_by_resource.get(resource_jid, 0),
                "change_reason": (
                    f"INSERTION: Approved bridge recovery macro '{macro_name}' "
                    f"step {index}/{total_tasks} ({len(primitive_steps)} primitives: {step_summary}) "
                    f"on {resource_jid}"
                ),
                "bridge_sequence_id": bridge_sequence_id,
                "bridge_sequence_index": index,
                "bridge_sequence_length": total_tasks,
                "bridge_outline_id": outline_id,
                "predecessor_outline_ids": deepcopy(outline_predecessors),
                "recovery_group_id": bridge_sequence_id,
                "recovery_kind": "bridge_macro",
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
            next_sequence_index_by_resource[resource_jid] = (
                int(task_node.get("sequence_index") or 0) + 1
            )
            compiled_task_ids_by_outline_id[outline_id] = task_id
            previous_outline_id = outline_id

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
            bridge_debug = deepcopy(self.get_last_bridge_debug() or {})
            runtime_handoff = dict(bridge_debug.get("runtime_handoff") or {})
            runtime_handoff.update(
                {
                    "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
                    "handoff_owner": "product_agent",
                    "bridge_generation_mode": str(bridge_generation_mode or "auto").strip().lower() or "auto",
                    "auto_start_requested": str(bridge_generation_mode or "auto").strip().lower() == "auto",
                    "auto_start_started": False,
                }
            )
            bridge_debug["runtime_handoff"] = runtime_handoff
            prepared_bridge_request["bridge_debug"] = deepcopy(bridge_debug)
            if hasattr(self, "_set_last_bridge_debug"):
                self._set_last_bridge_debug(bridge_debug)
            message = (
                "DES found no modeled continuation. Prepared bridge request returned for runtime handoff."
                if str(bridge_generation_mode or "auto").strip().lower() != "manual"
                else "DES found no modeled continuation. Review the prepared bridge request and "
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

            if "successors" in t:
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
                "bridge_outline_id",
                "predecessor_outline_ids",
                "recovery_group_id",
                "recovery_parent_failure_id",
                "recovery_kind",
                "projected_snapshot",
                "projected_part_entry",
            ):
                if extra_key in t:
                    target[extra_key] = deepcopy(t[extra_key])
            if "status" in t:
                target["status"] = str(t.get("status") or "pending").strip() or "pending"

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
        """Write a timestamped Markdown report to the llm_bridge runtime-data directory for each replan."""
        try:
            debug_dir = Path(
                "cais_spade_llm/agents/intelligent_product/replanner/llm_bridge/runtime_data"
            )
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
