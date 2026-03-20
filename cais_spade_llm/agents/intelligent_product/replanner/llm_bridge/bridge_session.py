"""Bridge session preparation and multi-turn DES-guided ReAct loop helpers."""

from __future__ import annotations

import asyncio
import json
import time
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.bridge_adapters import (
    canonical_bridge_event,
    canonical_bridge_resource,
)
from cais_spade_llm.resources.resource_profile import (
    all_registered_operation_kinds,
    get_resource_profile,
    resource_snapshot_carried_entity,
    resource_snapshot_carried_entity_location,
    resource_snapshot_fields_map,
)
from cais_spade_llm.prompts import build_bridge_turn_prompt


def _bridge_compact_react_trace(turn_response: dict[str, Any]) -> str:
    payload = turn_response if isinstance(turn_response, dict) else {}
    parts: list[str] = []
    reason_summary = str(payload.get("reason_summary", "") or "").strip()
    if reason_summary:
        parts.append(f"reason={reason_summary}")
    react_trace = payload.get("react_trace") or {}
    if isinstance(react_trace, dict):
        for key, label in (
            ("observed_facts", "facts"),
            ("gap_to_close", "gap"),
            ("decision_basis", "basis"),
            ("expected_progress", "progress"),
        ):
            values = [
                str(item).strip()
                for item in (react_trace.get(key) or [])
                if str(item).strip()
            ]
            if values:
                parts.append(f"{label}=" + " | ".join(values[:2]))
    return "; ".join(parts[:5])


class BridgeSessionMixin:
    def _refresh_bridge_grounding_context(
        self,
        prepared_bridge_request: dict[str, Any],
    ) -> dict[str, Any]:
        bridge_resources = deepcopy(prepared_bridge_request.get("bridge_resources") or {})
        focused_resource_jid = str(prepared_bridge_request.get("ra_jid", "") or "").strip()
        focused_entry = dict(bridge_resources.get(focused_resource_jid) or {})
        bridge_snapshot = deepcopy(
            focused_entry.get("bridge_snapshot")
            or prepared_bridge_request.get("bridge_snapshot")
            or {}
        )
        bridge_session = dict(prepared_bridge_request.get("bridge_session") or {})
        projected_resource_snapshots = deepcopy(
            bridge_session.get("projected_resource_snapshots") or {}
        )
        projected_parts = deepcopy(bridge_session.get("projected_parts") or {})
        incremental_mode = self._bridge_incremental_event_mode(prepared_bridge_request)
        prompt_bridge_resources = deepcopy(bridge_resources)
        prompt_part_tracker = deepcopy(prepared_bridge_request.get("part_tracker") or {})
        prompt_bridge_snapshot = deepcopy(bridge_snapshot)

        if incremental_mode and list(bridge_session.get("approved_bridge_events") or []):
            for resource_jid, raw_snapshot in projected_resource_snapshots.items():
                jid = str(resource_jid or "").strip()
                if not jid or not isinstance(raw_snapshot, dict):
                    continue
                resource_entry = dict(prompt_bridge_resources.get(jid) or {})
                if not resource_entry:
                    continue
                resource_entry["bridge_snapshot"] = deepcopy(raw_snapshot)
                prompt_bridge_resources[jid] = resource_entry
                if jid == focused_resource_jid:
                    prompt_bridge_snapshot = deepcopy(raw_snapshot)

            for part_name, raw_entry in projected_parts.items():
                name = str(part_name or "").strip()
                if not name or not isinstance(raw_entry, dict):
                    continue
                part_entry = dict(prompt_part_tracker.get(name) or {})
                for field in (
                    "state",
                    "location",
                    "last_known_location",
                    "observed_pose",
                    "pose",
                    "pose_status",
                ):
                    if raw_entry.get(field) is not None:
                        part_entry[field] = deepcopy(raw_entry.get(field))
                prompt_part_tracker[name] = part_entry

        grounding_context = self._bridge_grounding_context(
            focused_resource_jid=focused_resource_jid,
            bridge_snapshot=prompt_bridge_snapshot,
            bridge_resources=prompt_bridge_resources,
            part_tracker=prompt_part_tracker,
            goal_state=str(prepared_bridge_request.get("goal_state", "") or "").strip(),
            P_id=deepcopy(list(prepared_bridge_request.get("P_id") or [])),
            obligation_targets=deepcopy(list(prepared_bridge_request.get("obligation_targets") or [])),
        )
        observation_store = deepcopy(bridge_session.get("observation_store") or {})
        if observation_store:
            grounding_context["step_outputs"] = observation_store

        prepared_bridge_request["bridge_snapshot"] = bridge_snapshot
        prepared_bridge_request["grounding_context"] = grounding_context
        prepared_bridge_request["bridge_resources"] = bridge_resources
        marked_reentry_context = self._bridge_marked_reentry_context(
            prepared_bridge_request,
            projected_resource_snapshots=(
                projected_resource_snapshots
                if incremental_mode and list(bridge_session.get("approved_bridge_events") or [])
                else None
            ),
            projected_parts=(
                projected_parts
                if incremental_mode and list(bridge_session.get("approved_bridge_events") or [])
                else None
            ),
        )
        prepared_bridge_request["marked_reentry_context"] = marked_reentry_context
        prepared_bridge_request["continuation_context"] = deepcopy(marked_reentry_context)
        prepared_bridge_request["bridge_safety_context"] = self._derive_bridge_safety_constraints(
            prepared_bridge_request,
            marked_reentry_context=marked_reentry_context,
        )
        bridge_session = dict(prepared_bridge_request.get("bridge_session") or {})
        if bridge_session:
            bridge_session["marked_reentry_conditions"] = deepcopy(
                marked_reentry_context.get("marked_reentry_conditions") or []
            )
            bridge_session["unmet_reentry_conditions"] = deepcopy(
                marked_reentry_context.get("unmet_reentry_conditions") or []
            )
            bridge_session["pending_suffix_summary"] = deepcopy(
                marked_reentry_context.get("pending_suffix_summary") or []
            )
            bridge_session["disrupted_state"] = deepcopy(
                prepared_bridge_request.get("disrupted_state")
                or prepared_bridge_request.get("stuck_state")
                or {}
            )
            if str(bridge_session.get("phase", "") or "").strip().lower() != "review":
                bridge_session["phase"] = self._bridge_current_phase(prepared_bridge_request)
            prepared_bridge_request["bridge_session"] = bridge_session

        bridge_debug = deepcopy(prepared_bridge_request.get("bridge_debug") or {})
        if isinstance(bridge_debug, dict):
            bridge_debug["bridge_context"] = {
                "primitive_mode": bool(
                    prepared_bridge_request.get("primitive_catalog") or bridge_resources
                ),
                "primitive_catalog": deepcopy(prepared_bridge_request.get("primitive_catalog") or []),
                "bridge_snapshot": deepcopy(bridge_snapshot),
                "bridge_resources": deepcopy(bridge_resources),
                "grounding_context": deepcopy(grounding_context),
                "marked_reentry_context": deepcopy(marked_reentry_context),
                "continuation_context": deepcopy(marked_reentry_context),
                "projected_resource_snapshots": deepcopy(projected_resource_snapshots),
                "projected_parts": deepcopy(projected_parts),
                "bridge_safety_context": deepcopy(
                    prepared_bridge_request.get("bridge_safety_context") or {}
                ),
                "resource_infos": deepcopy(prepared_bridge_request.get("resource_infos") or []),
            }
            prepared_bridge_request["bridge_debug"] = bridge_debug
        return grounding_context

    def _build_bridge_turn_prompt_preview(
        self,
        prepared_bridge_request: dict[str, Any],
    ) -> str:
        def _strip_named_poses(value: Any) -> Any:
            if isinstance(value, dict):
                return {
                    key: _strip_named_poses(raw_child)
                    for key, raw_child in value.items()
                    if str(key) != "named_poses"
                }
            if isinstance(value, list):
                return [_strip_named_poses(item) for item in value]
            return deepcopy(value)

        bridge_session = dict(prepared_bridge_request.get("bridge_session") or {})
        marked_reentry_context = deepcopy(
            prepared_bridge_request.get("marked_reentry_context")
            or prepared_bridge_request.get("continuation_context")
            or self._bridge_marked_reentry_context(prepared_bridge_request)
        )
        return build_bridge_turn_prompt(
            session_id=str(bridge_session.get("session_id", "") or "").strip(),
            turn_index=max(1, int(bridge_session.get("turn_index", 0) or 0)),
            max_turns=int(bridge_session.get("max_turns", 6) or 6),
            phase=self._bridge_current_phase(prepared_bridge_request),
            focused_resource_jid=str(prepared_bridge_request.get("ra_jid", "") or "").strip(),
            stuck_state=deepcopy(prepared_bridge_request.get("stuck_state") or {}),
            goal_state=str(prepared_bridge_request.get("goal_state", "") or "").strip(),
            pending_parts=deepcopy(list(prepared_bridge_request.get("P_id") or [])),
            obligation_targets=deepcopy(list(prepared_bridge_request.get("obligation_targets") or [])),
            bridge_resources=_strip_named_poses(
                prepared_bridge_request.get("bridge_resources") or {}
            ),
            grounding_context=_strip_named_poses(
                prepared_bridge_request.get("grounding_context") or {}
            ),
            observation_history=deepcopy(list(bridge_session.get("observation_history") or [])),
            operator_feedback_history=deepcopy(
                list(bridge_session.get("operator_feedback_history") or [])
            ),
            validation_feedback=deepcopy(list(bridge_session.get("validation_feedback") or [])[-2:]),
            marked_reentry_conditions=deepcopy(
                list(marked_reentry_context.get("marked_reentry_conditions") or [])
            ),
            unmet_reentry_conditions=deepcopy(
                list(marked_reentry_context.get("unmet_reentry_conditions") or [])
            ),
            pending_suffix_summary=deepcopy(
                list(marked_reentry_context.get("pending_suffix_summary") or [])
            ),
            last_plan_failure=deepcopy(bridge_session.get("last_plan_failure") or {}),
            allowed_observation_primitives=deepcopy(
                list(
                    bridge_session.get("allowed_observation_primitives")
                    or self._bridge_turn_observation_primitives()
                )
            ),
            bridge_outline=deepcopy(list(bridge_session.get("bridge_outline") or [])),
            approved_bridge_events=deepcopy(
                list(bridge_session.get("approved_bridge_events") or [])
            ),
            infeasible_assignments=deepcopy(
                list(bridge_session.get("infeasible_assignments") or [])
            ),
            executor_bindings=deepcopy(
                list(bridge_session.get("executor_bindings") or [])
            ),
            handoff_requirements=deepcopy(
                list(bridge_session.get("handoff_requirements") or [])
            ),
            bridge_safety_context=deepcopy(
                prepared_bridge_request.get("bridge_safety_context") or {}
            ),
            draft_final_plan=deepcopy(bridge_session.get("draft_final_plan") or {}),
            draft_final_plan_status=deepcopy(bridge_session.get("draft_final_plan_status") or {}),
        )

    def _bridge_effective_part_facts(
        self,
        *,
        fallback_part_tracker: dict[str, Any],
        projected_parts: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, Any]]]:
        part_states: dict[str, Any] = {}
        part_locations: dict[str, Any] = {}
        part_entries: dict[str, dict[str, Any]] = {}

        for part_name, raw_info in (fallback_part_tracker or {}).items():
            name = str(part_name or "").strip()
            if not name:
                continue
            info = raw_info if isinstance(raw_info, dict) else {}
            part_states[name] = info.get("state")
            part_locations[name] = info.get("location")
            part_entries[name] = deepcopy(info)

        for part_name, raw_info in (projected_parts or {}).items():
            name = str(part_name or "").strip()
            if not name:
                continue
            info = raw_info if isinstance(raw_info, dict) else {}
            entry = dict(part_entries.get(name) or {})
            entry.update(deepcopy(info))
            part_entries[name] = entry
            if "state" in info:
                part_states[name] = info.get("state")
            if "location" in info:
                part_locations[name] = info.get("location")

        return part_states, part_locations, part_entries

    def _bridge_effective_resource_facts(
        self,
        *,
        bridge_resources: dict[str, dict[str, Any]],
        projected_resource_snapshots: dict[str, dict[str, Any]] | None,
    ) -> dict[str, dict[str, Any]]:
        effective: dict[str, dict[str, Any]] = {}
        for resource_jid, raw_entry in (bridge_resources or {}).items():
            jid = str(resource_jid or "").strip()
            if not jid:
                continue
            entry = raw_entry if isinstance(raw_entry, dict) else {}
            modeled_state = dict(entry.get("modeled_state") or {})
            snapshot = dict(
                (projected_resource_snapshots or {}).get(jid)
                or entry.get("bridge_snapshot")
                or {}
            )
            canonical_snapshot = canonical_bridge_resource(
                resource_jid=jid,
                resource_type=str(
                    snapshot.get("resource_type")
                    or dict(snapshot.get("resource_core") or {}).get("resource_type")
                    or entry.get("resource_type")
                    or dict(entry.get("resource_core") or {}).get("resource_type")
                    or dict((entry.get("bridge_snapshot") or {})).get("resource_type")
                    or "resource"
                ),
                snapshot=snapshot,
                modeled_state=modeled_state,
            )
            resource_core = dict(
                canonical_snapshot.get("resource_core")
                or entry.get("resource_core")
                or {}
            )
            resource_facets = dict(
                canonical_snapshot.get("resource_facets")
                or entry.get("resource_facets")
                or {}
            )
            profile = get_resource_profile(
                str(
                    resource_core.get("resource_type")
                    or canonical_snapshot.get("resource_type")
                    or snapshot.get("resource_type")
                    or modeled_state.get("resource_type")
                    or "resource"
                ).strip().lower()
                or "resource"
            )
            effective[jid] = {
                "current_state": (
                    resource_core.get("current_state")
                    if resource_core.get("current_state") is not None
                    else canonical_snapshot.get("current_state")
                    if canonical_snapshot.get("current_state") is not None
                    else snapshot.get("current_state")
                    if snapshot.get("current_state") is not None
                    else modeled_state.get("resource_state")
                ),
                "current_location": (
                    resource_core.get("current_location")
                    if resource_core.get("current_location") is not None
                    else canonical_snapshot.get("current_location")
                    if canonical_snapshot.get("current_location") is not None
                    else snapshot.get("current_location")
                    if snapshot.get("current_location") is not None
                    else modeled_state.get("current_location")
                ),
                "resource_type": (
                    resource_core.get("resource_type")
                    or canonical_snapshot.get("resource_type")
                    or snapshot.get("resource_type")
                    or modeled_state.get("resource_type")
                ),
                "resource_core": deepcopy(resource_core),
                "resource_facets": deepcopy(resource_facets),
                "occupancy": deepcopy(
                    resource_core.get("occupancy")
                    or canonical_snapshot.get("occupancy")
                    or snapshot.get("occupancy")
                    or {}
                ),
                **resource_snapshot_fields_map(
                    canonical_snapshot,
                    profile.snapshot_fields,
                    profile=profile,
                ),
            }
        return effective

    def _bridge_select_resumable_entry_task(
        self,
        *,
        resource_jid: str,
        pending_tasks: list[dict[str, Any]],
        tools_catalog: list[dict[str, Any]],
        effective_resources: dict[str, dict[str, Any]],
        part_states: dict[str, Any],
        part_locations: dict[str, Any],
    ) -> dict[str, Any] | None:
        effective_resource = dict(effective_resources.get(resource_jid) or {})
        profile = get_resource_profile(str(effective_resource.get("resource_type") or "resource"))
        current_state = str(effective_resource.get("current_state") or "").strip()
        carried_entity = str(
            resource_snapshot_carried_entity(
                effective_resource,
                profile=profile,
            )
            or ""
        ).strip()
        current_location = effective_resource.get("current_location")
        carried_location = resource_snapshot_carried_entity_location(
            resource_jid=resource_jid,
            snapshot=effective_resource,
            profile=profile,
        )

        for task in pending_tasks:
            if not isinstance(task, dict):
                continue
            function_name = str(task.get("function_name", "")).strip()
            if not function_name:
                continue
            tool_row = self._tool_row_for_task(
                resource_jid=resource_jid,
                function_name=function_name,
                tools_catalog=tools_catalog,
            )
            params = dict(task.get("params") or {})
            part_name = str(params.get("part_name") or "").strip()
            in_state = str(tool_row.get("in_state") or "").strip()
            if in_state and in_state.lower() != "any" and in_state != current_state:
                continue

            part_in_state = str(tool_row.get("part_in_state") or "").strip()
            if part_name and part_in_state and part_states.get(part_name) != part_in_state:
                continue

            if part_name:
                if carried_entity and carried_entity != part_name:
                    continue
                if part_locations.get(part_name) == carried_location and carried_entity != part_name:
                    continue

            ctx_map = dict(tool_row.get("context_mapping") or {})
            location_param = str(ctx_map.get("location_param") or "").strip()
            location_type = str(ctx_map.get("location_type") or "").strip()
            location_value = params.get(location_param) if location_param else None
            if location_param and location_value not in (None, ""):
                if location_type == "current_location" and current_location != location_value:
                    continue
                if location_type == "part_location" and part_name:
                    if part_locations.get(part_name) != location_value:
                        continue

            return task

        return pending_tasks[0] if pending_tasks else None

    def _bridge_marked_reentry_context(
        self,
        prepared_bridge_request: dict[str, Any],
        *,
        projected_resource_snapshots: dict[str, dict[str, Any]] | None = None,
        projected_parts: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        focused_resource_jid = str(prepared_bridge_request.get("ra_jid", "") or "").strip()
        goal_state = str(prepared_bridge_request.get("goal_state", "") or "").strip()
        bridge_resources = deepcopy(prepared_bridge_request.get("bridge_resources") or {})
        tools_catalog = deepcopy(list(prepared_bridge_request.get("tools_catalog") or []))
        part_states, part_locations, _ = self._bridge_effective_part_facts(
            fallback_part_tracker=deepcopy(prepared_bridge_request.get("part_tracker") or {}),
            projected_parts=deepcopy(projected_parts or {}),
        )
        effective_resources = self._bridge_effective_resource_facts(
            bridge_resources=bridge_resources,
            projected_resource_snapshots=projected_resource_snapshots,
        )

        pending_suffix_summary: list[dict[str, Any]] = []
        requirement_entries: list[dict[str, Any]] = []

        def _add_requirement(entry: dict[str, Any]) -> None:
            entity_kind = str(entry.get("entity_kind", "")).strip()
            entity = str(entry.get("entity", "")).strip()
            field = str(entry.get("field", "")).strip()
            expected = entry.get("expected")
            if not entity_kind or not entity or not field or expected in (None, ""):
                return
            entry["_key"] = (entity_kind, entity, field, json.dumps(expected, sort_keys=True, default=str))
            requirement_entries.append(entry)

        for resource_jid, raw_entry in (bridge_resources or {}).items():
            jid = str(resource_jid or "").strip()
            if not jid:
                continue
            entry = raw_entry if isinstance(raw_entry, dict) else {}
            pending_tasks = [
                task for task in (entry.get("pending_tasks") or [])
                if isinstance(task, dict)
            ]
            if not pending_tasks:
                continue

            role = "bridge_replaced" if jid == focused_resource_jid else "resume_suffix"
            task_ids = [
                str(task.get("id", "")).strip()
                for task in pending_tasks
                if str(task.get("id", "")).strip()
            ]
            task_parts = sorted(
                {
                    str((task.get("params") or {}).get("part_name") or "").strip()
                    for task in pending_tasks
                    if str((task.get("params") or {}).get("part_name") or "").strip()
                }
            )
            suffix_summary = {
                "resource_jid": jid,
                "role": role,
                "pending_task_ids": task_ids,
                "parts": task_parts,
            }

            if jid == focused_resource_jid:
                last_task = pending_tasks[-1]
                last_fn = str(last_task.get("function_name", "")).strip()
                last_tool = self._tool_row_for_task(
                    resource_jid=jid,
                    function_name=last_fn,
                    tools_catalog=tools_catalog,
                )
                terminal_state = str(last_tool.get("out_state") or "").strip()
                suffix_summary["terminal_task_id"] = str(last_task.get("id", "")).strip()
                suffix_summary["terminal_function_name"] = last_fn
                if terminal_state:
                    suffix_summary["terminal_resource_state"] = terminal_state
                    _add_requirement(
                        {
                            "kind": "focused_resource_terminal_state",
                            "entity_kind": "resource",
                            "entity": jid,
                            "field": "current_state",
                            "expected": terminal_state,
                            "source_task_id": str(last_task.get("id", "")).strip(),
                            "source_function_name": last_fn,
                            "role": role,
                        }
                    )
                if goal_state:
                    for part_name in task_parts:
                        _add_requirement(
                            {
                                "kind": "bridge_part_goal",
                                "entity_kind": "part",
                                "entity": part_name,
                                "field": "state",
                                "expected": goal_state,
                                "source_task_ids": task_ids,
                                "role": role,
                            }
                        )
                        target_location = str(
                            (
                                (
                                    (prepared_bridge_request.get("grounding_context") or {})
                                    .get("parts", {})
                                    .get(part_name, {})
                                    .get("target", {})
                                ).get("location")
                            )
                            or ""
                        ).strip()
                        if target_location:
                            _add_requirement(
                                {
                                    "kind": "bridge_part_goal_location",
                                    "entity_kind": "part",
                                    "entity": part_name,
                                    "field": "location",
                                    "expected": target_location,
                                    "source_task_ids": task_ids,
                                    "role": role,
                                }
                            )
            else:
                first_task = self._bridge_select_resumable_entry_task(
                    resource_jid=jid,
                    pending_tasks=pending_tasks,
                    tools_catalog=tools_catalog,
                    effective_resources=effective_resources,
                    part_states=part_states,
                    part_locations=part_locations,
                )
                if first_task is None:
                    pending_suffix_summary.append(suffix_summary)
                    continue
                first_fn = str(first_task.get("function_name", "")).strip()
                first_tool = self._tool_row_for_task(
                    resource_jid=jid,
                    function_name=first_fn,
                    tools_catalog=tools_catalog,
                )
                entry_task_id = str(first_task.get("id", "")).strip()
                entry_part_name = str((first_task.get("params") or {}).get("part_name") or "").strip()
                entry_in_state = str(first_tool.get("in_state") or "").strip()
                entry_part_state = str(first_tool.get("part_in_state") or "").strip()
                suffix_summary.update(
                    {
                        "entry_task_id": entry_task_id,
                        "entry_function_name": first_fn,
                        "entry_part_name": entry_part_name,
                    }
                )
                if entry_in_state:
                    suffix_summary["required_resource_state"] = entry_in_state
                if entry_part_state and entry_part_name:
                    suffix_summary["required_part_state"] = entry_part_state
                ctx_map = dict(first_tool.get("context_mapping") or {})
                location_param = str(ctx_map.get("location_param") or "").strip()
                location_type = str(ctx_map.get("location_type") or "").strip()
                location_value = first_task.get("params", {}).get(location_param) if location_param else None
                if location_param and location_value not in (None, ""):
                    suffix_summary["required_location"] = location_value
                    suffix_summary["required_location_type"] = location_type

                if entry_in_state and entry_in_state.lower() != "any":
                    _add_requirement(
                        {
                            "kind": "resume_entry_resource_state",
                            "entity_kind": "resource",
                            "entity": jid,
                            "field": "current_state",
                            "expected": entry_in_state,
                            "source_task_id": entry_task_id,
                            "source_function_name": first_fn,
                            "role": role,
                        }
                    )
                if location_param and location_value not in (None, ""):
                    if location_type == "current_location":
                        _add_requirement(
                            {
                                "kind": "resume_entry_resource_location",
                                "entity_kind": "resource",
                                "entity": jid,
                                "field": "current_location",
                                "expected": location_value,
                                "source_task_id": entry_task_id,
                                "source_function_name": first_fn,
                                "role": role,
                            }
                        )
                    elif location_type == "part_location" and entry_part_name:
                        _add_requirement(
                            {
                                "kind": "resume_entry_part_location",
                                "entity_kind": "part",
                                "entity": entry_part_name,
                                "field": "location",
                                "expected": location_value,
                                "source_task_id": entry_task_id,
                                "source_function_name": first_fn,
                                "role": role,
                            }
                        )
                if entry_part_name and entry_part_state:
                    profile = get_resource_profile(
                        str((effective_resources.get(jid) or {}).get("resource_type") or "resource")
                    )
                    _add_requirement(
                        {
                            "kind": "resume_entry_part_state",
                            "entity_kind": "part",
                            "entity": entry_part_name,
                            "field": "state",
                            "expected": entry_part_state,
                            "source_task_id": entry_task_id,
                            "source_function_name": first_fn,
                            "role": role,
                        }
                    )
                    carried_entity_field = str(profile.carried_entity_field or "").strip()
                    if carried_entity_field:
                        _add_requirement(
                            {
                                "kind": "resume_entry_carried_entity",
                                "entity_kind": "resource",
                                "entity": jid,
                                "field": carried_entity_field,
                                "expected": entry_part_name,
                                "source_task_id": entry_task_id,
                                "source_function_name": first_fn,
                                "role": role,
                            }
                        )

            pending_suffix_summary.append(suffix_summary)

        deduped_requirements: list[dict[str, Any]] = []
        seen_requirement_keys: set[tuple[str, str, str, str]] = set()
        for entry in requirement_entries:
            key = entry.pop("_key", None)
            if key in seen_requirement_keys:
                continue
            seen_requirement_keys.add(key)
            deduped_requirements.append(entry)

        unmet_reentry_conditions: list[dict[str, Any]] = []
        for requirement in deduped_requirements:
            entity_kind = str(requirement.get("entity_kind", "")).strip()
            entity = str(requirement.get("entity", "")).strip()
            field = str(requirement.get("field", "")).strip()
            expected = requirement.get("expected")
            actual = None
            if entity_kind == "resource":
                actual = (effective_resources.get(entity) or {}).get(field)
            elif entity_kind == "part":
                if field == "state":
                    actual = part_states.get(entity)
                elif field == "location":
                    actual = part_locations.get(entity)
            if actual != expected:
                unmet = deepcopy(requirement)
                unmet["actual"] = actual
                unmet_reentry_conditions.append(unmet)

        return {
            "goal_state": goal_state,
            "focused_resource_jid": focused_resource_jid,
            "pending_suffix_summary": pending_suffix_summary,
            "marked_reentry_conditions": deduped_requirements,
            "unmet_reentry_conditions": unmet_reentry_conditions,
            # Backward-compatible aliases during migration.
            "pending_suffixes": pending_suffix_summary,
            "continuation_requirements": deduped_requirements,
            "unmet_requirements": unmet_reentry_conditions,
        }

    def _bridge_continuation_context(
        self,
        prepared_bridge_request: dict[str, Any],
        *,
        projected_resource_snapshots: dict[str, dict[str, Any]] | None = None,
        projected_parts: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self._bridge_marked_reentry_context(
            prepared_bridge_request,
            projected_resource_snapshots=projected_resource_snapshots,
            projected_parts=projected_parts,
        )

    @staticmethod
    def _format_bridge_marked_reentry_feedback(marked_reentry_context: dict[str, Any]) -> str:
        unmet = list(marked_reentry_context.get("unmet_reentry_conditions") or [])
        if not unmet:
            return "final projected bridge state did not satisfy the marked re-entry conditions"

        lines = ["final projected bridge state did not satisfy the marked re-entry conditions."]
        for entry in unmet[:6]:
            entity = str(entry.get("entity", "")).strip() or "unknown"
            field = str(entry.get("field", "")).strip() or "unknown"
            expected = entry.get("expected")
            actual = entry.get("actual")
            source_task_id = str(entry.get("source_task_id", "")).strip()
            source_fn = str(entry.get("source_function_name", "")).strip()
            suffix = ""
            if source_task_id or source_fn:
                suffix = f" (needed by {source_task_id or source_fn}"
                if source_task_id and source_fn:
                    suffix += f" / {source_fn}"
                suffix += ")"
            lines.append(
                f"- unmet {entity}.{field}: expected {expected!r}, actual {actual!r}{suffix}"
            )
        return "\n".join(lines)

    @staticmethod
    def _format_bridge_continuation_feedback(continuation_context: dict[str, Any]) -> str:
        return BridgeSessionMixin._format_bridge_marked_reentry_feedback(continuation_context)

    @staticmethod
    def _bridge_unmet_condition_lines(
        unmet_conditions: list[dict[str, Any]] | None,
        *,
        limit: int = 4,
    ) -> list[str]:
        lines: list[str] = []
        for entry in (unmet_conditions or [])[:limit]:
            if not isinstance(entry, dict):
                continue
            entity = str(entry.get("entity", "") or "unknown").strip()
            field = str(entry.get("field", "") or "unknown").strip()
            expected = entry.get("expected")
            actual = entry.get("actual")
            lines.append(f"{entity}.{field} -> expected {expected!r}, actual {actual!r}")
        return lines

    @staticmethod
    def _marked_reentry_condition_key(condition: dict[str, Any]) -> tuple[str, str, str, str]:
        entity_kind = str(condition.get("entity_kind", "")).strip()
        entity = str(condition.get("entity", "")).strip()
        field = str(condition.get("field", "")).strip()
        expected = json.dumps(condition.get("expected"), sort_keys=True, default=str)
        return (entity_kind, entity, field, expected)

    def _bridge_event_identity(
        self,
        event: dict[str, Any],
    ) -> str:
        canonical = canonical_bridge_event(
            event,
            bridge_resources={},
        )
        return json.dumps(canonical, sort_keys=True, default=str)

    def _bridge_new_event_slice(
        self,
        *,
        approved_events: list[dict[str, Any]],
        proposed_events: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        approved = [
            self._bridge_event_identity(event)
            for event in (approved_events or [])
            if isinstance(event, dict)
        ]
        proposed = [
            event for event in (proposed_events or [])
            if isinstance(event, dict)
        ]
        if not approved or not proposed:
            return proposed

        proposed_keys = [
            self._bridge_event_identity(event)
            for event in proposed
        ]
        approved_len = len(approved)
        if proposed_keys[:approved_len] == approved:
            return proposed[approved_len:]
        return proposed

    @staticmethod
    def _bridge_projection_changed(
        *,
        previous_resource_snapshots: dict[str, Any] | None,
        previous_parts: dict[str, Any] | None,
        next_resource_snapshots: dict[str, Any] | None,
        next_parts: dict[str, Any] | None,
    ) -> bool:
        return (
            deepcopy(previous_resource_snapshots or {}) != deepcopy(next_resource_snapshots or {})
            or deepcopy(previous_parts or {}) != deepcopy(next_parts or {})
        )

    @staticmethod
    def _bridge_executor_scope(part_name: str) -> str:
        name = str(part_name or "").strip()
        return f"recover_part_to_goal:{name}" if name else "recover_part_to_goal"

    def _bridge_part_entry_pose(
        self,
        *,
        part_entry: dict[str, Any] | None,
        grounding_part: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        entry = dict(part_entry or {})
        pose = self._coerce_xyz_pose(entry.get("observed_pose") or entry.get("pose"))
        if pose is not None:
            return pose
        pose_status = str(entry.get("pose_status") or "").strip().lower()
        if pose_status == "projected_target":
            target = dict((grounding_part or {}).get("target") or {})
            return self._coerce_xyz_pose(target.get("slot_pose") or target.get("pose"))
        return None

    def _bridge_part_pose_grounded(
        self,
        *,
        part_entry: dict[str, Any] | None,
        grounding_part: dict[str, Any] | None,
    ) -> bool:
        entry = dict(part_entry or {})
        pose_status = str(entry.get("pose_status") or "").strip().lower()
        if pose_status == "carried":
            return False
        return self._bridge_part_entry_pose(
            part_entry=entry,
            grounding_part=grounding_part,
        ) is not None

    def _bridge_update_executor_memory_after_event(
        self,
        *,
        normalized_event: dict[str, Any],
        projected_part_entries: dict[str, dict[str, Any]],
        grounding_parts: dict[str, Any],
        executor_bindings: dict[str, dict[str, Any]],
        handoff_requirements: dict[str, dict[str, Any]],
    ) -> None:
        resource_jid = str(normalized_event.get("resource_jid", "") or "").strip()
        part_name = str(normalized_event.get("part_name", "") or "").strip()
        if not part_name:
            return
        semantic_kind = self._bridge_event_semantic_kind(normalized_event)
        part_delta = self._bridge_event_expected_part_delta(normalized_event)
        part_to = str((part_delta or {}).get("to", "") or "").strip().lower()
        part_entry = dict(projected_part_entries.get(part_name) or {})
        grounding_part = dict(grounding_parts.get(part_name) or {})
        grounded_destination = self._bridge_part_pose_grounded(
            part_entry=part_entry,
            grounding_part=grounding_part,
        )
        scope = self._bridge_executor_scope(part_name)
        pose_status = str(part_entry.get("pose_status") or "").strip() or (
            "grounded" if grounded_destination else "unknown"
        )

        if semantic_kind == "pick":
            executor_bindings[part_name] = {
                "part_name": part_name,
                "scope": scope,
                "resource_jid": resource_jid,
                "status": "active",
            }
            handoff_requirements[part_name] = {
                "part_name": part_name,
                "bound_resource_jid": resource_jid,
                "required_for_switch": True,
                "grounded_destination_available": False,
                "current_location": part_entry.get("location"),
                "pose_status": pose_status or "carried",
                "reason": "part is currently carried by the active executor",
            }
            return

        if semantic_kind in {"stage", "place", "assemble", "pick_place"}:
            if part_to == "assembled":
                executor_bindings.pop(part_name, None)
                handoff_requirements.pop(part_name, None)
                return
            bound_resource_jid = str(
                (executor_bindings.get(part_name) or {}).get("resource_jid") or resource_jid
            ).strip()
            if not bound_resource_jid:
                return
            executor_bindings[part_name] = {
                "part_name": part_name,
                "scope": scope,
                "resource_jid": bound_resource_jid,
                "status": "released_pending_goal",
            }
            handoff_requirements[part_name] = {
                "part_name": part_name,
                "bound_resource_jid": bound_resource_jid,
                "required_for_switch": True,
                "grounded_destination_available": bool(grounded_destination),
                "current_location": part_entry.get("location"),
                "pose_status": pose_status,
                "reason": (
                    "executor switch allowed because the released part has a grounded destination pose"
                    if grounded_destination
                    else "executor switch requires a grounded handoff or new observation"
                ),
            }

    def _bridge_executor_memory_from_events(
        self,
        prepared_bridge_request: dict[str, Any],
        *,
        events: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        projected_resources = self._bridge_effective_resource_facts(
            bridge_resources=deepcopy(prepared_bridge_request.get("bridge_resources") or {}),
            projected_resource_snapshots=None,
        )
        part_states, part_locations, part_entries = self._bridge_effective_part_facts(
            fallback_part_tracker=deepcopy(prepared_bridge_request.get("part_tracker") or {}),
            projected_parts=None,
        )
        grounding_parts = dict(
            (prepared_bridge_request.get("grounding_context") or {}).get("parts") or {}
        )
        executor_bindings: dict[str, dict[str, Any]] = {}
        handoff_requirements: dict[str, dict[str, Any]] = {}

        for raw_event in (events or []):
            if not isinstance(raw_event, dict):
                continue
            normalized_event = canonical_bridge_event(
                raw_event,
                bridge_resources=deepcopy(prepared_bridge_request.get("bridge_resources") or {}),
            )
            self._bridge_apply_event_projection(
                normalized_event,
                projected_resources=projected_resources,
                projected_part_states=part_states,
                projected_part_locations=part_locations,
                projected_part_entries=part_entries,
                grounding_parts=grounding_parts,
            )
            self._bridge_update_executor_memory_after_event(
                normalized_event=normalized_event,
                projected_part_entries=part_entries,
                grounding_parts=grounding_parts,
                executor_bindings=executor_bindings,
                handoff_requirements=handoff_requirements,
            )

        return (
            [deepcopy(executor_bindings[key]) for key in sorted(executor_bindings)],
            [deepcopy(handoff_requirements[key]) for key in sorted(handoff_requirements)],
        )

    def _bridge_event_summary_from_macro_tasks(
        self,
        *,
        prepared_bridge_request: dict[str, Any],
        proposal: dict[str, Any],
    ) -> list[dict[str, Any]]:
        macro_tasks = [
            task for task in (proposal.get("macro_tasks") or [])
            if isinstance(task, dict)
        ]
        if not macro_tasks:
            return []

        provided_summary = [
            event for event in (proposal.get("bridge_event_summary") or [])
            if isinstance(event, dict)
        ]
        cumulative_snapshots: dict[str, dict[str, Any]] = {}
        cumulative_parts: dict[str, dict[str, Any]] = {}
        current_context = self._bridge_marked_reentry_context(
            prepared_bridge_request,
            projected_resource_snapshots=cumulative_snapshots,
            projected_parts=cumulative_parts,
        )
        current_unmet = {
            self._marked_reentry_condition_key(entry)
            for entry in (current_context.get("unmet_reentry_conditions") or [])
            if isinstance(entry, dict)
        }
        current_unmet_lookup = {
            self._marked_reentry_condition_key(entry): deepcopy(entry)
            for entry in (current_context.get("unmet_reentry_conditions") or [])
            if isinstance(entry, dict)
        }

        derived_summary: list[dict[str, Any]] = []
        for index, macro_task in enumerate(macro_tasks):
            resource_jid = str(macro_task.get("resource_jid", "") or "").strip()
            part_name = str(macro_task.get("part_name", "") or "").strip()
            if resource_jid and isinstance(macro_task.get("projected_snapshot"), dict):
                cumulative_snapshots[resource_jid] = deepcopy(
                    macro_task.get("projected_snapshot") or {}
                )
            if part_name and isinstance(macro_task.get("projected_part_entry"), dict):
                cumulative_parts[part_name] = deepcopy(macro_task.get("projected_part_entry") or {})

            next_context = self._bridge_marked_reentry_context(
                prepared_bridge_request,
                projected_resource_snapshots=cumulative_snapshots,
                projected_parts=cumulative_parts,
            )
            next_unmet = {
                self._marked_reentry_condition_key(entry)
                for entry in (next_context.get("unmet_reentry_conditions") or [])
                if isinstance(entry, dict)
            }
            closed_keys = current_unmet - next_unmet
            closes_conditions = [
                deepcopy(current_unmet_lookup[key])
                for key in current_unmet_lookup
                if key in closed_keys
            ]

            provided_event = provided_summary[index] if index < len(provided_summary) else {}
            if not isinstance(provided_event, dict):
                provided_event = {}
            task_metadata = dict(macro_task.get("task_metadata") or {})
            projected_part_entry = dict(macro_task.get("projected_part_entry") or {})
            expected_resource_delta = None
            in_state = str(task_metadata.get("in_state", "") or "").strip()
            out_state = str(task_metadata.get("out_state", "") or "").strip()
            if in_state and out_state:
                expected_resource_delta = {"from": in_state, "to": out_state}

            expected_part_delta = None
            part_transition = dict(task_metadata.get("part_transition") or {})
            completed = dict(part_transition.get("completed") or {})
            if part_name and completed:
                expected_part_delta = {
                    "part_name": part_name,
                    "from": "",
                    "to": str(completed.get("state") or "").strip(),
                }
                location_to = (
                    str(projected_part_entry.get("location") or "").strip()
                    or str((provided_event.get("expected_part_delta") or {}).get("location_to", "")).strip()
                )
                if location_to:
                    expected_part_delta["location_to"] = location_to
                if not expected_part_delta["to"]:
                    expected_part_delta = None

            entry = {
                "event_name": str(
                    provided_event.get("event_name")
                    or macro_task.get("macro_name")
                    or f"bridge_event_{index + 1}"
                ).strip(),
                "resource_jid": str(
                    provided_event.get("resource_jid") or resource_jid
                ).strip(),
                "part_name": str(
                    provided_event.get("part_name") or part_name
                ).strip(),
                "closes_conditions": deepcopy(
                    provided_event.get("closes_conditions") or closes_conditions
                ),
                "rationale": str(
                    provided_event.get("rationale")
                    or macro_task.get("rationale")
                    or ""
                ).strip(),
            }
            if isinstance(provided_event.get("expected_resource_delta"), dict):
                entry["expected_resource_delta"] = deepcopy(
                    provided_event.get("expected_resource_delta")
                )
            elif expected_resource_delta is not None:
                entry["expected_resource_delta"] = expected_resource_delta
            if isinstance(provided_event.get("expected_part_delta"), dict):
                entry["expected_part_delta"] = deepcopy(
                    provided_event.get("expected_part_delta")
                )
            elif expected_part_delta is not None:
                entry["expected_part_delta"] = expected_part_delta
            derived_summary.append(
                canonical_bridge_event(
                    entry,
                    bridge_resources=deepcopy(prepared_bridge_request.get("bridge_resources") or {}),
                )
            )
            current_context = next_context
            current_unmet = next_unmet
            current_unmet_lookup = {
                self._marked_reentry_condition_key(entry): deepcopy(entry)
                for entry in (current_context.get("unmet_reentry_conditions") or [])
                if isinstance(entry, dict)
            }

        return derived_summary

    def _bridge_missing_observation_feedback(
        self,
        *,
        prepared_bridge_request: dict[str, Any],
        marked_reentry_context: dict[str, Any],
    ) -> str:
        grounding_parts = dict(
            (prepared_bridge_request.get("grounding_context") or {}).get("parts") or {}
        )
        for unmet in (marked_reentry_context.get("unmet_reentry_conditions") or []):
            if not isinstance(unmet, dict):
                continue
            if str(unmet.get("entity_kind", "")).strip() != "part":
                continue
            part_name = str(unmet.get("entity", "")).strip()
            if not part_name:
                continue
            part_info = dict(grounding_parts.get(part_name) or {})
            if not isinstance(part_info.get("observed_pose"), dict):
                return (
                    f"insufficient live observation for part '{part_name}'; "
                    "request an observation before returning final_plan"
                )
        return ""

    async def prepare_bridge_session(
        self,
        *,
        stuck_state: dict[str, Any],
        P_id: list[str],
        ra_jid: str,
        goal_state: str,
        tools_catalog: list[dict[str, Any]],
        part_tracker: dict[str, Any],
        obligation_targets: list[dict[str, Any]],
        bridge_feedback: str,
        resource_states: dict[str, dict[str, Any]],
        default_resource_state: str,
        part_states: dict[str, Any],
        part_locations: dict[str, Any],
        bridge_safety_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        operator_feedback_history = []
        feedback_text = str(bridge_feedback or "").strip()
        if feedback_text:
            operator_feedback_history.append(feedback_text)

        bridge_debug: dict[str, Any] = {
            "requested_at_utc": datetime.now(timezone.utc).isoformat(),
            "request": {
                "stuck_state": deepcopy(stuck_state),
                "P_id": deepcopy(P_id),
                "ra_jid": str(ra_jid or "").strip(),
                "goal_state": str(goal_state or "").strip(),
                "part_tracker": deepcopy(part_tracker),
                "obligation_targets": deepcopy(obligation_targets),
                "bridge_feedback": feedback_text,
                "resource_states": deepcopy(resource_states),
                "default_resource_state": str(default_resource_state or "").strip(),
                "part_states": deepcopy(part_states),
                "part_locations": deepcopy(part_locations),
                "bridge_safety_context": deepcopy(bridge_safety_context or {}),
            },
            "turns": [],
            "warning_messages": [],
            "normalized_proposal": None,
            "status": "ready",
        }
        self._set_last_bridge_debug(bridge_debug)

        primitive_catalog, bridge_snapshot, bridge_resources = await self._bridge_primitive_context(
            target_jid=ra_jid,
            resource_states=resource_states,
            default_resource_state=default_resource_state,
            part_states=part_states,
            part_locations=part_locations,
        )
        resource_infos = self._resource_infos()
        primitive_mode = bool(primitive_catalog or bridge_resources)
        bridge_session = {
            "session_id": f"bridge_{uuid4().hex}",
            "focused_resource_jid": str(ra_jid or "").strip(),
            "disrupted_state": deepcopy(stuck_state),
            "phase": "observe_required",
            "turn_index": 0,
            "max_turns": 6,
            "max_observations": 3,
            "max_final_retries": 2,
            "observation_count": 0,
            "final_retry_count": 0,
            "allowed_observation_primitives": self._bridge_turn_observation_primitives(),
            "operator_feedback_history": operator_feedback_history,
            "observation_store": {},
            "observation_history": [],
            "bridge_outline": [],
            "approved_bridge_events": [],
            "bridge_events_complete": False,
            "infeasible_assignments": [],
            "executor_bindings": [],
            "handoff_requirements": [],
            "modeled_continuation_gap": {},
            "projected_resource_snapshots": {},
            "projected_parts": {},
            "validation_feedback": [],
            "last_plan_failure": {},
            "bridge_safety_context": deepcopy(bridge_safety_context or {}),
            "draft_final_plan": {},
            "draft_final_plan_status": {},
            "validated_deterministic_plan": {},
        }
        prepared_bridge_request = {
            "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
            "disrupted_state": deepcopy(stuck_state),
            "stuck_state": deepcopy(stuck_state),
            "P_id": deepcopy(P_id),
            "ra_jid": str(ra_jid or "").strip(),
            "goal_state": str(goal_state or "").strip(),
            "tools_catalog": deepcopy(tools_catalog),
            "part_tracker": deepcopy(part_tracker),
            "obligation_targets": deepcopy(obligation_targets),
            "bridge_feedback": feedback_text,
            "resource_states": deepcopy(resource_states),
            "default_resource_state": str(default_resource_state or "").strip(),
            "part_states": deepcopy(part_states),
            "part_locations": deepcopy(part_locations),
            "bridge_safety_context": deepcopy(bridge_safety_context or {}),
            "primitive_catalog": deepcopy(primitive_catalog),
            "bridge_snapshot": deepcopy(bridge_snapshot),
            "grounding_context": {},
            "bridge_resources": deepcopy(bridge_resources),
            "resource_infos": deepcopy(resource_infos),
            "plan_nodes": deepcopy(self.nodes),
            "bridge_session": deepcopy(bridge_session),
            "bridge_debug": deepcopy(bridge_debug),
            "marked_reentry_context": {},
            "continuation_context": {},
        }
        grounding_context = self._refresh_bridge_grounding_context(prepared_bridge_request)
        bridge_session = dict(prepared_bridge_request.get("bridge_session") or {})
        bridge_session["phase"] = self._bridge_current_phase(prepared_bridge_request)
        prepared_bridge_request["bridge_session"] = bridge_session
        marked_reentry_context = deepcopy(
            prepared_bridge_request.get("marked_reentry_context")
            or prepared_bridge_request.get("continuation_context")
            or {}
        )
        bridge_debug["primitive_mode"] = primitive_mode
        bridge_debug["session"] = deepcopy(bridge_session)
        bridge_debug["bridge_context"] = {
            "primitive_mode": primitive_mode,
            "primitive_catalog": deepcopy(primitive_catalog),
            "bridge_snapshot": deepcopy(prepared_bridge_request.get("bridge_snapshot") or {}),
            "bridge_resources": deepcopy(bridge_resources),
            "grounding_context": deepcopy(grounding_context),
            "marked_reentry_context": deepcopy(marked_reentry_context),
            "continuation_context": deepcopy(marked_reentry_context),
            "resource_infos": deepcopy(resource_infos),
        }
        bridge_debug["llm_inputs"] = {
            "disrupted_state": deepcopy(stuck_state),
            "stuck_state": deepcopy(stuck_state),
            "P_id": deepcopy(P_id),
            "ra_jid": str(ra_jid or "").strip(),
            "goal_state": str(goal_state or "").strip(),
            "part_tracker": deepcopy(part_tracker),
            "obligation_targets": deepcopy(obligation_targets),
            "operator_feedback_history": deepcopy(operator_feedback_history),
            "resource_infos": deepcopy(resource_infos),
            "tools_catalog": deepcopy(tools_catalog),
            "primitive_catalog": deepcopy(primitive_catalog),
            "bridge_snapshot": deepcopy(prepared_bridge_request.get("bridge_snapshot") or {}),
            "grounding_context": deepcopy(grounding_context),
            "marked_reentry_context": deepcopy(marked_reentry_context),
            "continuation_context": deepcopy(marked_reentry_context),
            "bridge_safety_context": deepcopy(prepared_bridge_request.get("bridge_safety_context") or {}),
            "bridge_resources": deepcopy(bridge_resources),
            "bridge_session": deepcopy(bridge_session),
        }
        bridge_debug["prompt"] = self._build_bridge_turn_prompt_preview(prepared_bridge_request)
        prepared_bridge_request["bridge_debug"] = deepcopy(bridge_debug)
        self._set_last_bridge_debug(bridge_debug)
        return prepared_bridge_request

    async def prepare_bridge_request(
        self,
        *,
        stuck_state: dict[str, Any],
        P_id: list[str],
        ra_jid: str,
        goal_state: str,
        tools_catalog: list[dict[str, Any]],
        part_tracker: dict[str, Any],
        obligation_targets: list[dict[str, Any]],
        bridge_feedback: str,
        resource_states: dict[str, dict[str, Any]],
        default_resource_state: str,
        part_states: dict[str, Any],
        part_locations: dict[str, Any],
        bridge_safety_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return await self.prepare_bridge_session(
            stuck_state=stuck_state,
            P_id=P_id,
            ra_jid=ra_jid,
            goal_state=goal_state,
            tools_catalog=tools_catalog,
            part_tracker=part_tracker,
            obligation_targets=obligation_targets,
            bridge_feedback=bridge_feedback,
            resource_states=resource_states,
            default_resource_state=default_resource_state,
            part_states=part_states,
            part_locations=part_locations,
            bridge_safety_context=bridge_safety_context,
        )

    def validate_preprogrammed_bridge_proposal(
        self,
        *,
        proposal: dict[str, Any],
        prepared_bridge_request: dict[str, Any],
        source: str = "preprogrammed_scenario",
        scenario_id: str = "",
    ) -> dict[str, Any]:
        from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.bridge_generation import (
            _normalize_primitive_bridge_proposal,
        )

        if not isinstance(proposal, dict) or not proposal:
            raise ValueError("preprogrammed bridge proposal is missing")
        if not isinstance(prepared_bridge_request, dict) or not prepared_bridge_request:
            raise ValueError("prepared bridge request is missing")

        bridge_debug = deepcopy(prepared_bridge_request.get("bridge_debug") or {})
        if not isinstance(bridge_debug, dict):
            bridge_debug = {}
        bridge_debug["source"] = str(source or "preprogrammed_scenario")
        if scenario_id:
            bridge_debug["scenario_id"] = str(scenario_id)
        bridge_debug["generation_started_at_utc"] = datetime.now(timezone.utc).isoformat()
        bridge_debug["status"] = "running"
        bridge_debug["raw_response"] = json.dumps(proposal, indent=2, default=str)
        self._set_last_bridge_debug(bridge_debug)

        normalized = _normalize_primitive_bridge_proposal(
            raw=json.dumps(proposal, default=str),
            ra_jid=str(prepared_bridge_request.get("ra_jid", "") or "").strip(),
            primitive_catalog=deepcopy(list(prepared_bridge_request.get("primitive_catalog") or [])),
            bridge_snapshot=deepcopy(prepared_bridge_request.get("bridge_snapshot") or {}),
            grounding_context=deepcopy(prepared_bridge_request.get("grounding_context") or {}),
            bridge_resources=deepcopy(prepared_bridge_request.get("bridge_resources") or {}),
            obligation_targets=deepcopy(list(prepared_bridge_request.get("obligation_targets") or [])),
        )
        bridge_debug["normalized_proposal"] = deepcopy(normalized) if isinstance(normalized, dict) else None
        if not isinstance(normalized, dict):
            bridge_debug["modeled_continuation_check"] = {
                "accepted": False,
                "reason": "preprogrammed bridge proposal is not valid",
            }
            bridge_debug["status"] = "rejected"
            self._set_last_bridge_debug(bridge_debug)
            raise ValueError("preprogrammed bridge proposal is not valid")

        normalized["bridge_event_summary"] = self._bridge_event_summary_from_macro_tasks(
            prepared_bridge_request=prepared_bridge_request,
            proposal=normalized,
        )
        safety_ok, safety_error = self._bridge_validate_safety_constraints(
            prepared_bridge_request,
            events=deepcopy(normalized.get("bridge_event_summary") or []),
        )
        bridge_debug["bridge_safety_check"] = {
            "accepted": bool(safety_ok),
            "reason": str(safety_error or "").strip(),
            "context": deepcopy(prepared_bridge_request.get("bridge_safety_context") or {}),
        }
        if not safety_ok:
            bridge_debug["status"] = "rejected_bridge_safety"
            self._set_last_bridge_debug(bridge_debug)
            raise ValueError(
                "preprogrammed bridge proposal violated bridge safety constraints"
            )
        projected_marked_reentry_context = self._bridge_marked_reentry_context(
            prepared_bridge_request,
            projected_resource_snapshots=deepcopy(
                normalized.get("projected_resource_snapshots") or {}
            ),
            projected_parts=deepcopy(normalized.get("projected_parts") or {}),
        )
        bridge_debug["marked_reentry_check"] = {
            "accepted": not bool(
                projected_marked_reentry_context.get("unmet_reentry_conditions") or []
            ),
            "reason": "",
            "context": deepcopy(projected_marked_reentry_context),
        }
        bridge_debug["bridge_event_summary"] = deepcopy(normalized.get("bridge_event_summary") or [])
        if projected_marked_reentry_context.get("unmet_reentry_conditions"):
            marked_error = self._format_bridge_marked_reentry_feedback(
                projected_marked_reentry_context
            )
            bridge_debug["marked_reentry_check"]["accepted"] = False
            bridge_debug["marked_reentry_check"]["reason"] = marked_error
            bridge_debug["status"] = "rejected_marked_reentry"
            self._set_last_bridge_debug(bridge_debug)
            raise ValueError(
                "preprogrammed bridge proposal did not satisfy the marked re-entry conditions"
            )

        bridge_resources = deepcopy(prepared_bridge_request.get("bridge_resources") or {})
        if bridge_resources and not self._bridge_restores_modeled_continuation(
            proposal=normalized,
            goal_state=str(prepared_bridge_request.get("goal_state", "") or "unknown"),
            tools_catalog=deepcopy(list(prepared_bridge_request.get("tools_catalog") or [])),
            bridge_resources=bridge_resources,
            fallback_part_tracker=deepcopy(prepared_bridge_request.get("part_tracker") or {}),
        ):
            modeled_error = (
                "projected bridge state satisfied the marked re-entry conditions but DES still "
                "found no resumable modeled continuation"
            )
            bridge_debug["modeled_continuation_check"] = {
                "accepted": False,
                "reason": modeled_error,
            }
            bridge_debug["status"] = "rejected_modeled_continuation"
            self._set_last_bridge_debug(bridge_debug)
            raise ValueError(
                "preprogrammed bridge proposal did not restore a modeled continuation"
            )

        bridge_debug["modeled_continuation_check"] = {
            "accepted": True,
            "reason": "",
        }
        normalized["plan_rewrite"] = self._bridge_synthesize_plan_rewrite(
            prepared_bridge_request
        )
        bridge_debug["plan_rewrite"] = deepcopy(normalized.get("plan_rewrite") or {})
        bridge_debug["status"] = "accepted"
        self._set_last_bridge_debug(bridge_debug)
        return normalized

    async def execute_prepared_bridge_request(
        self,
        prepared_bridge_request: dict[str, Any],
    ) -> dict[str, Any] | None:
        if not isinstance(prepared_bridge_request, dict):
            raise ValueError("prepared bridge request is missing")
        return await self.run_bridge_react_session(prepared_bridge_request)

    def _append_bridge_validation_feedback(
        self,
        prepared_bridge_request: dict[str, Any],
        *,
        kind: str,
        message: str,
    ) -> None:
        bridge_session = dict(prepared_bridge_request.get("bridge_session") or {})
        validation_feedback = list(bridge_session.get("validation_feedback") or [])
        feedback_entry = {
            "kind": str(kind or "validation_error").strip() or "validation_error",
            "message": str(message or "").strip(),
        }
        validation_feedback.append(feedback_entry)
        bridge_session["validation_feedback"] = validation_feedback[-6:]
        prepared_bridge_request["bridge_session"] = bridge_session

        bridge_debug = deepcopy(prepared_bridge_request.get("bridge_debug") or {})
        if isinstance(bridge_debug, dict):
            warnings = list(bridge_debug.get("warning_messages") or [])
            warnings.append(feedback_entry["message"])
            bridge_debug["warning_messages"] = warnings[-12:]
            bridge_debug["session"] = deepcopy(bridge_session)
            prepared_bridge_request["bridge_debug"] = bridge_debug
            self._set_last_bridge_debug(bridge_debug)

    def _bridge_remember_infeasible_assignments(
        self,
        prepared_bridge_request: dict[str, Any],
        *,
        feasibility_decisions: list[dict[str, Any]] | None,
    ) -> None:
        bridge_session = dict(prepared_bridge_request.get("bridge_session") or {})
        remembered = [
            item
            for item in (bridge_session.get("infeasible_assignments") or [])
            if isinstance(item, dict)
        ]
        seen = {
            json.dumps(
                {
                    "resource_jid": str(item.get("resource_jid") or "").strip(),
                    "part_name": str(item.get("part_name") or "").strip(),
                    "operation_kind": str(item.get("operation_kind") or "").strip(),
                    "scope": str(item.get("scope") or "").strip(),
                },
                sort_keys=True,
            )
            for item in remembered
        }
        for decision in (feasibility_decisions or []):
            if not isinstance(decision, dict) or decision.get("allowed", True):
                continue
            operation_kind = str(decision.get("operation_kind") or "").strip()
            part_name = str(decision.get("part_name") or "").strip()
            scope = operation_kind
            if operation_kind in {"pick", "pick_place"} and part_name:
                scope = "recover_part_from_current_pose"
            entry = {
                "resource_jid": str(decision.get("resource_jid") or "").strip(),
                "part_name": part_name,
                "operation_kind": operation_kind,
                "scope": scope,
                "reason": str(decision.get("reason") or "").strip(),
            }
            signature = json.dumps(
                {
                    "resource_jid": entry["resource_jid"],
                    "part_name": entry["part_name"],
                    "operation_kind": entry["operation_kind"],
                    "scope": entry["scope"],
                },
                sort_keys=True,
            )
            if signature in seen:
                continue
            seen.add(signature)
            remembered.append(entry)
        bridge_session["infeasible_assignments"] = remembered[-8:]
        prepared_bridge_request["bridge_session"] = bridge_session

    def _normalize_primitive_bridge_plan_with_warnings(
        self,
        *,
        prepared_bridge_request: dict[str, Any],
        plan: dict[str, Any],
    ) -> tuple[dict[str, Any] | None, list[str]]:
        from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge import bridge_generation

        warnings: list[str] = []
        capture = bridge_generation._BridgeWarningCapture(warnings)
        bridge_generation.logger.addHandler(capture)
        try:
            normalized = bridge_generation._normalize_primitive_bridge_proposal(
                raw=json.dumps(plan or {}, default=str),
                ra_jid=str(prepared_bridge_request.get("ra_jid", "") or "").strip(),
                primitive_catalog=deepcopy(list(prepared_bridge_request.get("primitive_catalog") or [])),
                bridge_snapshot=deepcopy(prepared_bridge_request.get("bridge_snapshot") or {}),
                grounding_context=deepcopy(prepared_bridge_request.get("grounding_context") or {}),
                bridge_resources=deepcopy(prepared_bridge_request.get("bridge_resources") or {}),
                obligation_targets=deepcopy(list(prepared_bridge_request.get("obligation_targets") or [])),
            )
        finally:
            bridge_generation.logger.removeHandler(capture)
            capture.close()
        return normalized, warnings

    def _bridge_preview_approved_events(
        self,
        prepared_bridge_request: dict[str, Any],
        *,
        approved_events: list[dict[str, Any]],
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None, str | None]:
        compiled_plan, compile_error = self._compile_bridge_events_to_macro_tasks(
            prepared_bridge_request,
            approved_events=deepcopy(approved_events or []),
        )
        if not isinstance(compiled_plan, dict):
            return None, None, str(
                compile_error or "deterministic bridge-event compilation failed"
            ).strip()

        normalized_plan, warning_messages = self._normalize_primitive_bridge_plan_with_warnings(
            prepared_bridge_request=prepared_bridge_request,
            plan=compiled_plan,
        )
        if not isinstance(normalized_plan, dict):
            preview_error = str(
                warning_messages[-1]
                if warning_messages
                else "bridge_events failed deterministic primitive validation"
            ).strip()
            return compiled_plan, None, preview_error

        return compiled_plan, normalized_plan, None

    def _bridge_symbolic_preview_approved_events(
        self,
        prepared_bridge_request: dict[str, Any],
        *,
        approved_events: list[dict[str, Any]],
    ) -> dict[str, Any]:
        projected_resources = self._bridge_effective_resource_facts(
            bridge_resources=deepcopy(prepared_bridge_request.get("bridge_resources") or {}),
            projected_resource_snapshots=None,
        )
        part_states, part_locations, part_entries = self._bridge_effective_part_facts(
            fallback_part_tracker=deepcopy(prepared_bridge_request.get("part_tracker") or {}),
            projected_parts=None,
        )

        for raw_event in (approved_events or []):
            if not isinstance(raw_event, dict):
                continue
            normalized_event = canonical_bridge_event(
                raw_event,
                bridge_resources=deepcopy(prepared_bridge_request.get("bridge_resources") or {}),
            )
            self._bridge_apply_event_projection(
                normalized_event,
                projected_resources=projected_resources,
                projected_part_states=part_states,
                projected_part_locations=part_locations,
                projected_part_entries=part_entries,
                grounding_parts=dict(
                    (prepared_bridge_request.get("grounding_context") or {}).get("parts") or {}
                ),
            )
            part_name = str(normalized_event.get("part_name", "") or "").strip()
            if part_name:
                updated_entry = dict(part_entries.get(part_name) or {})
                if part_name in part_states:
                    updated_entry["state"] = part_states.get(part_name)
                if part_name in part_locations:
                    updated_entry["location"] = part_locations.get(part_name)
                if updated_entry:
                    part_entries[part_name] = updated_entry

        projected_resource_snapshots = {
            str(resource_jid): deepcopy(resource_entry)
            for resource_jid, resource_entry in projected_resources.items()
            if str(resource_jid).strip()
        }
        projected_parts = {
            str(part_name): deepcopy(part_entry)
            for part_name, part_entry in part_entries.items()
            if str(part_name).strip()
        }
        return {
            "bridge_event_summary": deepcopy(approved_events or []),
            "projected_resource_snapshots": projected_resource_snapshots,
            "projected_parts": projected_parts,
        }

    async def _execute_bridge_observation_turn(
        self,
        prepared_bridge_request: dict[str, Any],
        *,
        action: dict[str, Any],
    ) -> tuple[dict[str, Any] | None, str | None]:
        bridge_session = dict(prepared_bridge_request.get("bridge_session") or {})
        resource_jid = str(action.get("resource_jid", "") or "").strip()
        primitive = str(action.get("primitive", "") or "").strip()
        params = deepcopy(action.get("params") or {})
        resource = self._resource_by_jid(resource_jid)
        if resource is None:
            return None, f"bridge observation targeted unknown resource '{resource_jid}'"
        execute_observation = getattr(resource, "execute_bridge_observation", None)
        if not callable(execute_observation):
            return None, f"resource '{resource_jid}' does not expose execute_bridge_observation"

        observation_response = execute_observation(primitive, params)
        if asyncio.iscoroutine(observation_response):
            observation_response = await observation_response
        if not isinstance(observation_response, dict):
            return None, f"bridge observation '{primitive}' returned invalid payload"
        if not observation_response.get("success", False):
            return None, str(observation_response.get("message", "") or f"{primitive} failed")

        observation = deepcopy(observation_response.get("observation") or {})
        if not isinstance(observation, dict):
            return None, f"bridge observation '{primitive}' produced no normalized observation"

        observation_history = list(bridge_session.get("observation_history") or [])
        alias = str(action.get("store_as", "") or "").strip()
        if not alias:
            alias = f"observation_{len(observation_history) + 1}"
        observation_store = dict(bridge_session.get("observation_store") or {})
        observation_store[alias] = deepcopy(observation)
        bridge_session["observation_store"] = observation_store
        bridge_session["observation_count"] = int(bridge_session.get("observation_count", 0) or 0) + 1

        bridge_resources = deepcopy(prepared_bridge_request.get("bridge_resources") or {})
        resource_entry = dict(bridge_resources.get(resource_jid) or {})
        snapshot = deepcopy(
            observation_response.get("snapshot")
            or (resource.get_bridge_snapshot() if hasattr(resource, "get_bridge_snapshot") else {})
            or {}
        )
        resource_entry["bridge_snapshot"] = snapshot
        bridge_resources[resource_jid] = resource_entry
        prepared_bridge_request["bridge_resources"] = bridge_resources
        if resource_jid == str(prepared_bridge_request.get("ra_jid", "") or "").strip():
            prepared_bridge_request["bridge_snapshot"] = deepcopy(snapshot)

        part_tracker = deepcopy(prepared_bridge_request.get("part_tracker") or {})
        observed_part_name = str(observation.get("part_name") or params.get("part_name") or "").strip()
        if observed_part_name:
            entry = dict(part_tracker.get(observed_part_name) or {})
            if isinstance(observation.get("pose"), dict):
                entry["observed_pose"] = deepcopy(observation.get("pose"))
            if primitive == "detect_parts":
                entry["last_known_location"] = deepcopy(entry.get("last_known_location"))
            part_tracker[observed_part_name] = entry
            prepared_bridge_request["part_tracker"] = part_tracker

        prepared_bridge_request["bridge_session"] = bridge_session
        grounding_context = self._refresh_bridge_grounding_context(prepared_bridge_request)
        observation_row = {
            "turn_index": int(bridge_session.get("turn_index", 0) or 0),
            "resource_jid": resource_jid,
            "primitive": primitive,
            "params": deepcopy(params),
            "store_as": alias,
            "observation": deepcopy(observation),
            "reason_summary": str(action.get("reason_summary", "") or "").strip(),
        }
        observation_history.append(observation_row)
        bridge_session["observation_history"] = observation_history[-12:]
        prepared_bridge_request["bridge_session"] = bridge_session

        bridge_debug = deepcopy(prepared_bridge_request.get("bridge_debug") or {})
        if isinstance(bridge_debug, dict):
            bridge_debug["session"] = deepcopy(bridge_session)
            bridge_debug["grounding_context"] = deepcopy(grounding_context)
            prepared_bridge_request["bridge_debug"] = bridge_debug
            self._set_last_bridge_debug(bridge_debug)
        return observation_row, None

    async def run_bridge_react_session(
        self,
        prepared_bridge_request: dict[str, Any],
    ) -> dict[str, Any] | None:
        from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.bridge_generation import (
            normalize_bridge_turn_response,
        )

        if not isinstance(prepared_bridge_request, dict):
            raise ValueError("prepared bridge request is missing")

        bridge_session = dict(prepared_bridge_request.get("bridge_session") or {})
        if not bridge_session:
            prepared_bridge_request.update(
                await self.prepare_bridge_session(
                    stuck_state=deepcopy(prepared_bridge_request.get("stuck_state") or {}),
                    P_id=deepcopy(list(prepared_bridge_request.get("P_id") or [])),
                    ra_jid=str(prepared_bridge_request.get("ra_jid", "") or "").strip(),
                    goal_state=str(prepared_bridge_request.get("goal_state", "") or "").strip(),
                    tools_catalog=deepcopy(list(prepared_bridge_request.get("tools_catalog") or [])),
                    part_tracker=deepcopy(prepared_bridge_request.get("part_tracker") or {}),
                    obligation_targets=deepcopy(list(prepared_bridge_request.get("obligation_targets") or [])),
                    bridge_feedback=str(prepared_bridge_request.get("bridge_feedback", "") or "").strip(),
                    resource_states=deepcopy(prepared_bridge_request.get("resource_states") or {}),
                    default_resource_state=str(prepared_bridge_request.get("default_resource_state", "") or "").strip(),
                    part_states=deepcopy(prepared_bridge_request.get("part_states") or {}),
                    part_locations=deepcopy(prepared_bridge_request.get("part_locations") or {}),
                )
            )
            bridge_session = dict(prepared_bridge_request.get("bridge_session") or {})

        bridge_debug = deepcopy(prepared_bridge_request.get("bridge_debug") or {})
        if not isinstance(bridge_debug, dict):
            bridge_debug = {}
        bridge_debug["generation_started_at_utc"] = datetime.now(timezone.utc).isoformat()
        bridge_debug["status"] = "running"
        bridge_debug.setdefault("turns", [])
        prepared_bridge_request["bridge_debug"] = bridge_debug
        self._set_last_bridge_debug(bridge_debug)

        available_resource_jids = sorted(
            str(resource_jid).strip()
            for resource_jid in (prepared_bridge_request.get("bridge_resources") or {})
            if str(resource_jid).strip()
        )
        allowed_observation_primitives = list(
            bridge_session.get("allowed_observation_primitives")
            or self._bridge_turn_observation_primitives()
        )
        max_turns = int(bridge_session.get("max_turns", 6) or 6)
        max_observations = int(bridge_session.get("max_observations", 3) or 3)
        max_final_retries = int(bridge_session.get("max_final_retries", 2) or 2)
        if self._bridge_incremental_event_mode(prepared_bridge_request):
            max_turns = max(max_turns, 16)
            max_observations = max(max_observations, 5)
            max_final_retries = max(max_final_retries, 4)
            bridge_session["max_turns"] = max_turns
            bridge_session["max_observations"] = max_observations
            bridge_session["max_final_retries"] = max_final_retries
            prepared_bridge_request["bridge_session"] = bridge_session
        _bridge_t0 = time.monotonic()
        self.logger.info(
            "[Bridge] Session started — max_turns=%d, max_observations=%d, max_final_retries=%d",
            max_turns, max_observations, max_final_retries,
        )

        while int(bridge_session.get("turn_index", 0) or 0) < max_turns:
            bridge_session = dict(prepared_bridge_request.get("bridge_session") or {})
            bridge_session["turn_index"] = int(bridge_session.get("turn_index", 0) or 0) + 1
            prepared_bridge_request["bridge_session"] = bridge_session
            self._refresh_bridge_grounding_context(prepared_bridge_request)
            bridge_session = dict(prepared_bridge_request.get("bridge_session") or {})
            phase = self._bridge_current_phase(prepared_bridge_request)
            bridge_session["phase"] = phase
            prepared_bridge_request["bridge_session"] = bridge_session
            _turn_idx = int(bridge_session.get("turn_index", 0) or 0)
            _turn_t0 = time.monotonic()
            self.logger.info(
                "[Bridge] Turn %d — phase=%s, sending prompt to LLM...",
                _turn_idx, phase,
            )
            current_marked_reentry_context = deepcopy(
                prepared_bridge_request.get("marked_reentry_context")
                or prepared_bridge_request.get("continuation_context")
                or {}
            )
            turn_debug: dict[str, Any] = {
                "turn_index": int(bridge_session.get("turn_index", 0) or 0),
                "phase": phase,
                "unmet_reentry_conditions_before": deepcopy(
                    current_marked_reentry_context.get("unmet_reentry_conditions") or []
                ),
                "approved_bridge_events_before": deepcopy(
                    bridge_session.get("approved_bridge_events") or []
                ),
            }
            phase_history = list(bridge_debug.get("phase_history") or [])
            phase_history.append(
                {
                    "turn_index": turn_debug["turn_index"],
                    "phase": phase,
                }
            )
            bridge_debug["phase_history"] = phase_history[-24:]

            raw_response: Any = None
            prompt = ""
            response_source = "llm"
            compile_path = "llm_repair"
            llm_latency_s = 0.0
            draft_status = dict(bridge_session.get("draft_final_plan_status") or {})
            cached_draft_plan = deepcopy(bridge_session.get("draft_final_plan") or {})

            if (
                phase == "final_plan"
                and bridge_session.get("approved_bridge_events")
                and str(draft_status.get("compile_path", "")).strip() != "llm_repair"
            ):
                compiled_plan = None
                compile_error = ""
                if isinstance(cached_draft_plan, dict) and cached_draft_plan:
                    compiled_plan = cached_draft_plan
                else:
                    compiled_plan, compile_error = self._compile_bridge_events_to_macro_tasks(
                        prepared_bridge_request,
                        approved_events=deepcopy(
                            list(bridge_session.get("approved_bridge_events") or [])
                        ),
                    )
                if isinstance(compiled_plan, dict):
                    response_source = "planner_compiler"
                    compile_path = "deterministic"
                    bridge_session["draft_final_plan"] = deepcopy(compiled_plan)
                    bridge_session["draft_final_plan_status"] = {
                        "compile_path": "deterministic",
                        "status": "compiled",
                    }
                    prepared_bridge_request["bridge_session"] = bridge_session
                    raw_response = {
                        "type": "final_plan",
                        "plan": deepcopy(compiled_plan),
                        "reason_summary": "Planner-compiled deterministic final plan from approved bridge events.",
                    }
                else:
                    bridge_session["draft_final_plan"] = {}
                    bridge_session["draft_final_plan_status"] = {
                        "compile_path": "llm_repair",
                        "status": "compile_failed",
                        "error": str(
                            compile_error or "deterministic final-plan compilation failed"
                        ).strip(),
                    }
                    bridge_session["validated_deterministic_plan"] = {}
                    prepared_bridge_request["bridge_session"] = bridge_session
                    bridge_session = dict(prepared_bridge_request.get("bridge_session") or {})

            if raw_response is None:
                prompt = self._build_bridge_turn_prompt_preview(prepared_bridge_request)
                turn_debug["prompt"] = prompt
                bridge_debug["prompt"] = prompt
                bridge_debug["session"] = deepcopy(bridge_session)
                bridge_debug["phase"] = phase
                prepared_bridge_request["bridge_debug"] = bridge_debug
                self._set_last_bridge_debug(bridge_debug)

                try:
                    _llm_t0 = time.monotonic()
                    raw_response = await self.product_agent.ask_llm(
                        prompt=prompt,
                        with_functions=False,
                        temperature=0.0,
                    )
                    llm_latency_s = time.monotonic() - _llm_t0
                except Exception as exc:
                    bridge_debug["status"] = "exception"
                    bridge_debug["exception"] = repr(exc)
                    bridge_debug["turns"].append({**turn_debug, "exception": repr(exc)})
                    prepared_bridge_request["bridge_debug"] = bridge_debug
                    self._set_last_bridge_debug(bridge_debug)
                    raise
            else:
                turn_debug["prompt"] = "<planner_compiler>"
                bridge_debug["prompt"] = "<planner_compiler>"
                bridge_debug["session"] = deepcopy(bridge_session)
                bridge_debug["phase"] = phase
                prepared_bridge_request["bridge_debug"] = bridge_debug
                self._set_last_bridge_debug(bridge_debug)

            raw_text = (
                raw_response
                if isinstance(raw_response, str)
                else json.dumps(raw_response, indent=2, default=str)
            )
            turn_debug["raw_response"] = raw_text
            bridge_debug["raw_response"] = raw_text
            turn_debug["response_source"] = response_source
            turn_debug["compile_path"] = compile_path
            turn_debug["turn_llm_latency_s"] = round(llm_latency_s, 4)
            turn_debug["session_elapsed_s"] = round(time.monotonic() - _bridge_t0, 4)

            turn_response, turn_error = normalize_bridge_turn_response(
                raw=raw_response,
                available_resource_jids=available_resource_jids,
                allowed_observation_primitives=allowed_observation_primitives,
                bridge_resources=deepcopy(prepared_bridge_request.get("bridge_resources") or {}),
            )
            if turn_error:
                self._append_bridge_validation_feedback(
                    prepared_bridge_request,
                    kind="invalid_turn_response",
                    message=turn_error,
                )
                bridge_session = dict(prepared_bridge_request.get("bridge_session") or {})
                bridge_debug = deepcopy(prepared_bridge_request.get("bridge_debug") or {})
                turn_debug["error"] = turn_error
                bridge_debug["turns"].append(turn_debug)
                prepared_bridge_request["bridge_debug"] = bridge_debug
                self._set_last_bridge_debug(bridge_debug)
                continue

            turn_debug["normalized_response"] = deepcopy(turn_response)
            reason_summary = str(turn_response.get("reason_summary", "") or "").strip()
            if reason_summary:
                turn_debug["reason_summary"] = reason_summary
            react_trace = deepcopy(turn_response.get("react_trace") or {})
            if isinstance(react_trace, dict) and react_trace:
                turn_debug["react_trace"] = react_trace
            response_type = str(turn_response.get("type", "")).strip().lower()
            if response_source == "planner_compiler":
                self.logger.info(
                    "[Bridge] Turn %d — planner compiled type=%s (turn=%.2fs, session=%.1fs)",
                    _turn_idx,
                    response_type,
                    time.monotonic() - _turn_t0,
                    time.monotonic() - _bridge_t0,
                )
            else:
                self.logger.info(
                    "[Bridge] Turn %d — LLM responded type=%s (turn=%.2fs, session=%.1fs)",
                    _turn_idx,
                    response_type,
                    max(llm_latency_s, time.monotonic() - _turn_t0),
                    time.monotonic() - _bridge_t0,
                )
            react_trace_line = _bridge_compact_react_trace(turn_response)
            if react_trace_line:
                self.logger.info(
                    "[Bridge] Turn %d — ReAct trace: %s",
                    _turn_idx,
                    react_trace_line,
                )
            allowed_types = self._bridge_phase_allowed_types(phase)
            if response_type not in allowed_types:
                phase_error = (
                    f"phase '{phase}' only accepts {sorted(allowed_types)} responses; "
                    f"received '{response_type}'"
                )
                self._append_bridge_validation_feedback(
                    prepared_bridge_request,
                    kind="phase_mismatch",
                    message=phase_error,
                )
                bridge_session = dict(prepared_bridge_request.get("bridge_session") or {})
                bridge_debug = deepcopy(prepared_bridge_request.get("bridge_debug") or {})
                turn_debug["error"] = phase_error
                bridge_debug["turns"].append(turn_debug)
                prepared_bridge_request["bridge_debug"] = bridge_debug
                self._set_last_bridge_debug(bridge_debug)
                continue

            if response_type == "observe":
                if int(bridge_session.get("observation_count", 0) or 0) >= max_observations:
                    observation_error = (
                        f"observation budget exhausted ({max_observations}); return final_plan"
                    )
                    self._append_bridge_validation_feedback(
                        prepared_bridge_request,
                        kind="observation_budget_exhausted",
                        message=observation_error,
                    )
                    turn_debug["error"] = observation_error
                    bridge_session = dict(prepared_bridge_request.get("bridge_session") or {})
                    bridge_debug = deepcopy(prepared_bridge_request.get("bridge_debug") or {})
                    bridge_debug["turns"].append(turn_debug)
                    prepared_bridge_request["bridge_debug"] = bridge_debug
                    self._set_last_bridge_debug(bridge_debug)
                    continue

                observation_row, observation_error = await self._execute_bridge_observation_turn(
                    prepared_bridge_request,
                    action=turn_response,
                )
                _obs_primitive = str(turn_response.get("primitive", "")).strip()
                _obs_part = str((turn_response.get("params") or {}).get("part_name", "")).strip()
                if observation_error:
                    self._append_bridge_validation_feedback(
                        prepared_bridge_request,
                        kind="observation_failed",
                        message=observation_error,
                    )
                    bridge_debug = deepcopy(prepared_bridge_request.get("bridge_debug") or {})
                    turn_debug["error"] = observation_error
                    self.logger.warning(
                        "[Bridge] Turn %d — observation %s(%s) FAILED: %s",
                        _turn_idx, _obs_primitive, _obs_part, observation_error,
                    )
                else:
                    turn_debug["observation"] = deepcopy(observation_row)
                    self.logger.info(
                        "[Bridge] Turn %d — observation %s(%s) succeeded",
                        _turn_idx, _obs_primitive, _obs_part,
                    )
                bridge_session = dict(prepared_bridge_request.get("bridge_session") or {})
                bridge_session["phase"] = self._bridge_current_phase(prepared_bridge_request)
                prepared_bridge_request["bridge_session"] = bridge_session
                updated_marked_reentry_context = deepcopy(
                    prepared_bridge_request.get("marked_reentry_context")
                    or prepared_bridge_request.get("continuation_context")
                    or {}
                )
                turn_debug["phase_after"] = bridge_session.get("phase")
                turn_debug["unmet_reentry_conditions_after"] = deepcopy(
                    updated_marked_reentry_context.get("unmet_reentry_conditions") or []
                )
                bridge_debug["turns"].append(turn_debug)
                bridge_debug["session"] = deepcopy(bridge_session)
                prepared_bridge_request["bridge_debug"] = bridge_debug
                self._set_last_bridge_debug(bridge_debug)
                continue

            if response_type == "bridge_outline":
                bridge_session = dict(prepared_bridge_request.get("bridge_session") or {})
                bridge_session["bridge_outline"] = deepcopy(turn_response.get("steps") or [])
                bridge_session["last_plan_failure"] = {}
                bridge_session["phase"] = self._bridge_current_phase(prepared_bridge_request)
                prepared_bridge_request["bridge_session"] = bridge_session
                updated_marked_reentry_context = deepcopy(
                    prepared_bridge_request.get("marked_reentry_context")
                    or prepared_bridge_request.get("continuation_context")
                    or {}
                )
                self.logger.info(
                    "[Bridge] Turn %d — bridge_outline ACCEPTED: %d steps",
                    _turn_idx,
                    len(bridge_session.get("bridge_outline") or []),
                )
                for _idx, _step in enumerate(bridge_session.get("bridge_outline") or [], start=1):
                    if not isinstance(_step, dict):
                        continue
                    _step_name = str(_step.get("step_name", "") or "").strip() or f"step_{_idx}"
                    _objective = str(_step.get("objective", "") or "").strip()
                    _success = str(_step.get("success_signal", "") or "").strip()
                    _detail = _objective or _success or "no details provided"
                    if _objective and _success:
                        _detail = f"{_objective} => {_success}"
                    self.logger.info(
                        "[Bridge]   outline step %d: %s — %s",
                        _idx,
                        _step_name,
                        _detail,
                    )
                turn_debug["bridge_outline"] = deepcopy(
                    bridge_session.get("bridge_outline") or []
                )
                turn_debug["accepted"] = True
                turn_debug["phase_after"] = bridge_session.get("phase")
                turn_debug["unmet_reentry_conditions_after"] = deepcopy(
                    updated_marked_reentry_context.get("unmet_reentry_conditions") or []
                )
                bridge_debug = deepcopy(prepared_bridge_request.get("bridge_debug") or {})
                bridge_debug["turns"].append(turn_debug)
                bridge_debug["session"] = deepcopy(bridge_session)
                prepared_bridge_request["bridge_debug"] = bridge_debug
                self._set_last_bridge_debug(bridge_debug)
                continue

            if response_type == "bridge_events":
                bridge_session = dict(prepared_bridge_request.get("bridge_session") or {})
                current_approved_events = list(
                    bridge_session.get("approved_bridge_events") or []
                )
                proposed_slice = self._bridge_new_event_slice(
                    approved_events=current_approved_events,
                    proposed_events=deepcopy(turn_response.get("events") or []),
                )
                if not proposed_slice:
                    no_progress_error = (
                        "bridge_events response did not add any new event beyond the approved prefix"
                    )
                    self._append_bridge_validation_feedback(
                        prepared_bridge_request,
                        kind="bridge_events_invalid",
                        message=no_progress_error,
                    )
                    bridge_session = dict(prepared_bridge_request.get("bridge_session") or {})
                    bridge_debug = deepcopy(prepared_bridge_request.get("bridge_debug") or {})
                    turn_debug["error"] = no_progress_error
                    bridge_debug["turns"].append(turn_debug)
                    prepared_bridge_request["bridge_debug"] = bridge_debug
                    self._set_last_bridge_debug(bridge_debug)
                    continue

                candidate_events = current_approved_events + proposed_slice
                approved_events, feasibility_decisions, bridge_events_error = (
                    self._bridge_validate_bridge_events(
                        prepared_bridge_request,
                        events=deepcopy(candidate_events),
                        require_full_gamma_closure=(
                            not self._bridge_incremental_event_mode(prepared_bridge_request)
                        ),
                        require_primitive_preview=(
                            not self._bridge_incremental_event_mode(prepared_bridge_request)
                        ),
                    )
                )
                turn_debug["feasibility_decisions"] = deepcopy(feasibility_decisions)
                if bridge_events_error:
                    self._bridge_remember_infeasible_assignments(
                        prepared_bridge_request,
                        feasibility_decisions=feasibility_decisions,
                    )
                    bridge_session = dict(prepared_bridge_request.get("bridge_session") or {})
                    self.logger.warning(
                        "[Bridge] Turn %d — bridge_events REJECTED: %s",
                        _turn_idx, bridge_events_error,
                    )
                    bridge_session["last_plan_failure"] = {
                        "kind": "bridge_events_invalid",
                        "message": bridge_events_error,
                        "unmet_reentry_conditions": deepcopy(
                            current_marked_reentry_context.get("unmet_reentry_conditions")
                            or []
                        ),
                    }
                    bridge_session["draft_final_plan"] = {}
                    bridge_session["draft_final_plan_status"] = {}
                    bridge_session["validated_deterministic_plan"] = {}
                    prepared_bridge_request["bridge_session"] = bridge_session
                    self._append_bridge_validation_feedback(
                        prepared_bridge_request,
                        kind="bridge_events_invalid",
                        message=bridge_events_error,
                    )
                    bridge_session = dict(prepared_bridge_request.get("bridge_session") or {})
                    bridge_debug = deepcopy(prepared_bridge_request.get("bridge_debug") or {})
                    turn_debug["error"] = bridge_events_error
                    bridge_debug["turns"].append(turn_debug)
                    prepared_bridge_request["bridge_debug"] = bridge_debug
                    self._set_last_bridge_debug(bridge_debug)
                    continue

                incremental_mode = self._bridge_incremental_event_mode(prepared_bridge_request)
                if incremental_mode:
                    preview_plan = None
                    preview_normalized = self._bridge_symbolic_preview_approved_events(
                        prepared_bridge_request,
                        approved_events=deepcopy(approved_events or []),
                    )
                    preview_error = None
                else:
                    preview_plan, preview_normalized, preview_error = self._bridge_preview_approved_events(
                        prepared_bridge_request,
                        approved_events=deepcopy(approved_events or []),
                    )
                    if preview_error:
                        self.logger.warning(
                            "[Bridge] Turn %d — bridge_events REJECTED: %s",
                            _turn_idx, preview_error,
                        )
                        bridge_session["last_plan_failure"] = {
                            "kind": "bridge_events_invalid",
                            "message": preview_error,
                            "unmet_reentry_conditions": deepcopy(
                                current_marked_reentry_context.get("unmet_reentry_conditions")
                                or []
                            ),
                        }
                        bridge_session["draft_final_plan"] = {}
                        bridge_session["draft_final_plan_status"] = {}
                        bridge_session["validated_deterministic_plan"] = {}
                        prepared_bridge_request["bridge_session"] = bridge_session
                        self._append_bridge_validation_feedback(
                            prepared_bridge_request,
                            kind="bridge_events_invalid",
                            message=preview_error,
                        )
                        bridge_session = dict(prepared_bridge_request.get("bridge_session") or {})
                        bridge_debug = deepcopy(prepared_bridge_request.get("bridge_debug") or {})
                        turn_debug["error"] = preview_error
                        bridge_debug["turns"].append(turn_debug)
                        prepared_bridge_request["bridge_debug"] = bridge_debug
                        self._set_last_bridge_debug(bridge_debug)
                        continue

                projected_marked_reentry_context = self._bridge_marked_reentry_context(
                    prepared_bridge_request,
                    projected_resource_snapshots=deepcopy(
                        (preview_normalized or {}).get("projected_resource_snapshots") or {}
                    ),
                    projected_parts=deepcopy(
                        (preview_normalized or {}).get("projected_parts") or {}
                    ),
                )
                current_unmet_keys = {
                    self._marked_reentry_condition_key(entry)
                    for entry in (
                        current_marked_reentry_context.get("unmet_reentry_conditions") or []
                    )
                    if isinstance(entry, dict)
                }
                next_unmet_keys = {
                    self._marked_reentry_condition_key(entry)
                    for entry in (
                        projected_marked_reentry_context.get("unmet_reentry_conditions") or []
                    )
                    if isinstance(entry, dict)
                }
                regressed_keys = next_unmet_keys - current_unmet_keys
                closed_keys = current_unmet_keys - next_unmet_keys
                projection_changed = self._bridge_projection_changed(
                    previous_resource_snapshots=deepcopy(
                        bridge_session.get("projected_resource_snapshots") or {}
                    ),
                    previous_parts=deepcopy(bridge_session.get("projected_parts") or {}),
                    next_resource_snapshots=deepcopy(
                        (preview_normalized or {}).get("projected_resource_snapshots") or {}
                    ),
                    next_parts=deepcopy((preview_normalized or {}).get("projected_parts") or {}),
                )
                if regressed_keys:
                    regression_error = (
                        "bridge_events proposal regressed previously satisfied continuation conditions"
                    )
                    self.logger.warning(
                        "[Bridge] Turn %d — bridge_events REJECTED: %s",
                        _turn_idx,
                        regression_error,
                    )
                    bridge_session["last_plan_failure"] = {
                        "kind": "bridge_events_invalid",
                        "message": regression_error,
                        "unmet_reentry_conditions": deepcopy(
                            projected_marked_reentry_context.get("unmet_reentry_conditions") or []
                        ),
                    }
                    bridge_session["draft_final_plan"] = {}
                    bridge_session["draft_final_plan_status"] = {}
                    bridge_session["validated_deterministic_plan"] = {}
                    prepared_bridge_request["bridge_session"] = bridge_session
                    self._append_bridge_validation_feedback(
                        prepared_bridge_request,
                        kind="bridge_events_invalid",
                        message=regression_error,
                    )
                    bridge_session = dict(prepared_bridge_request.get("bridge_session") or {})
                    bridge_debug = deepcopy(prepared_bridge_request.get("bridge_debug") or {})
                    turn_debug["error"] = regression_error
                    bridge_debug["turns"].append(turn_debug)
                    prepared_bridge_request["bridge_debug"] = bridge_debug
                    self._set_last_bridge_debug(bridge_debug)
                    continue
                if not closed_keys and not projection_changed:
                    no_progress_error = (
                        "bridge_events proposal did not change the projected bridge state or close any remaining condition"
                    )
                    self.logger.warning(
                        "[Bridge] Turn %d — bridge_events REJECTED: %s",
                        _turn_idx,
                        no_progress_error,
                    )
                    bridge_session["last_plan_failure"] = {
                        "kind": "bridge_events_invalid",
                        "message": no_progress_error,
                        "unmet_reentry_conditions": deepcopy(
                            current_marked_reentry_context.get("unmet_reentry_conditions")
                            or []
                        ),
                    }
                    bridge_session["draft_final_plan"] = {}
                    bridge_session["draft_final_plan_status"] = {}
                    bridge_session["validated_deterministic_plan"] = {}
                    prepared_bridge_request["bridge_session"] = bridge_session
                    self._append_bridge_validation_feedback(
                        prepared_bridge_request,
                        kind="bridge_events_invalid",
                        message=no_progress_error,
                    )
                    bridge_session = dict(prepared_bridge_request.get("bridge_session") or {})
                    bridge_debug = deepcopy(prepared_bridge_request.get("bridge_debug") or {})
                    turn_debug["error"] = no_progress_error
                    bridge_debug["turns"].append(turn_debug)
                    prepared_bridge_request["bridge_debug"] = bridge_debug
                    self._set_last_bridge_debug(bridge_debug)
                    continue

                modeled_continuation_restored = False
                if not list(projected_marked_reentry_context.get("unmet_reentry_conditions") or []):
                    bridge_resources = deepcopy(prepared_bridge_request.get("bridge_resources") or {})
                    modeled_continuation_restored = (
                        not bridge_resources
                        or self._bridge_restores_modeled_continuation(
                            proposal=deepcopy(preview_normalized or {}),
                            goal_state=str(prepared_bridge_request.get("goal_state", "") or "unknown"),
                            tools_catalog=deepcopy(list(prepared_bridge_request.get("tools_catalog") or [])),
                            bridge_resources=bridge_resources,
                            fallback_part_tracker=deepcopy(prepared_bridge_request.get("part_tracker") or {}),
                        )
                    )

                bridge_events_complete = (
                    not list(projected_marked_reentry_context.get("unmet_reentry_conditions") or [])
                    and modeled_continuation_restored
                )
                executor_bindings, handoff_requirements = self._bridge_executor_memory_from_events(
                    prepared_bridge_request,
                    events=deepcopy(approved_events or []),
                )

                bridge_session["approved_bridge_events"] = deepcopy(approved_events or [])
                bridge_session["bridge_events_complete"] = bool(bridge_events_complete)
                bridge_session["executor_bindings"] = deepcopy(executor_bindings)
                bridge_session["handoff_requirements"] = deepcopy(handoff_requirements)
                bridge_session["projected_resource_snapshots"] = deepcopy(
                    (preview_normalized or {}).get("projected_resource_snapshots") or {}
                )
                bridge_session["projected_parts"] = deepcopy(
                    (preview_normalized or {}).get("projected_parts") or {}
                )
                if incremental_mode:
                    bridge_session["draft_final_plan"] = {}
                    bridge_session["draft_final_plan_status"] = {
                        "compile_path": "deterministic",
                        "status": (
                            "awaiting_final_compile"
                            if bridge_events_complete
                            else "event_prefix_projected"
                        ),
                    }
                    bridge_session["validated_deterministic_plan"] = {}
                else:
                    bridge_session["draft_final_plan"] = deepcopy(preview_plan or {})
                    bridge_session["draft_final_plan_status"] = {
                        "compile_path": "deterministic",
                        "status": "preview_validated" if bridge_events_complete else "preview_partial",
                    }
                    bridge_session["validated_deterministic_plan"] = deepcopy(
                        preview_normalized or {}
                    )
                if bridge_events_complete:
                    bridge_session["phase"] = "final_plan"
                    bridge_session["last_plan_failure"] = {}
                    bridge_session["modeled_continuation_gap"] = {}
                elif (
                    not list(projected_marked_reentry_context.get("unmet_reentry_conditions") or [])
                    and not modeled_continuation_restored
                ):
                    modeled_error = (
                        "approved bridge prefix closes marked re-entry conditions but does not yet restore a resumable modeled continuation"
                    )
                    modeled_gap = self._bridge_modeled_continuation_gap(
                        proposal=deepcopy(preview_normalized or {}),
                        goal_state=str(prepared_bridge_request.get("goal_state", "") or "unknown"),
                        tools_catalog=deepcopy(list(prepared_bridge_request.get("tools_catalog") or [])),
                        bridge_resources=deepcopy(prepared_bridge_request.get("bridge_resources") or {}),
                        fallback_part_tracker=deepcopy(prepared_bridge_request.get("part_tracker") or {}),
                    )
                    bridge_session["phase"] = "bridge_events"
                    bridge_session["modeled_continuation_gap"] = deepcopy(modeled_gap)
                    bridge_session["last_plan_failure"] = {
                        "kind": "modeled_continuation_rejected",
                        "message": modeled_error,
                        "unmet_reentry_conditions": [],
                        "pending_suffix_summary": deepcopy(
                            projected_marked_reentry_context.get("pending_suffix_summary") or []
                        ),
                        "modeled_continuation_gap": deepcopy(modeled_gap),
                    }
                    prepared_bridge_request["bridge_session"] = bridge_session
                    self._append_bridge_validation_feedback(
                        prepared_bridge_request,
                        kind="modeled_continuation_rejected",
                        message=modeled_error,
                    )
                    bridge_session = dict(prepared_bridge_request.get("bridge_session") or {})
                else:
                    bridge_session["phase"] = "bridge_events"
                    bridge_session["last_plan_failure"] = {}
                    bridge_session["modeled_continuation_gap"] = {}
                prepared_bridge_request["bridge_session"] = bridge_session
                _event_names = [
                    str(e.get("event_name", "")).strip()
                    for e in (proposed_slice or []) if isinstance(e, dict)
                ]
                if bridge_events_complete:
                    self.logger.info(
                        "[Bridge] Turn %d — bridge_events ACCEPTED: %d total events [%s]",
                        _turn_idx,
                        len(approved_events or []),
                        ", ".join(
                            str(e.get("event_name", "")).strip()
                            for e in (approved_events or [])
                            if isinstance(e, dict)
                        ),
                    )
                else:
                    self.logger.info(
                        "[Bridge] Turn %d — bridge_events ACCEPTED PARTIAL: +%d events [%s]; %d conditions remain",
                        _turn_idx,
                        len(proposed_slice or []),
                        ", ".join(_event_names),
                        len(projected_marked_reentry_context.get("unmet_reentry_conditions") or []),
                    )
                    for _line in self._bridge_unmet_condition_lines(
                        projected_marked_reentry_context.get("unmet_reentry_conditions") or []
                    ):
                        self.logger.info("[Bridge]   remaining target: %s", _line)
                bridge_debug = deepcopy(prepared_bridge_request.get("bridge_debug") or {})
                bridge_debug["approved_bridge_events"] = deepcopy(
                    bridge_session.get("approved_bridge_events") or []
                )
                bridge_debug["feasibility_decisions"] = deepcopy(feasibility_decisions)
                bridge_debug["marked_reentry_context"] = deepcopy(projected_marked_reentry_context)
                turn_debug["approved_bridge_events"] = deepcopy(
                    bridge_session.get("approved_bridge_events") or []
                )
                turn_debug["accepted_bridge_event_slice"] = deepcopy(proposed_slice or [])
                if incremental_mode:
                    turn_debug["symbolic_projection"] = deepcopy(preview_normalized or {})
                else:
                    turn_debug["preview_compiled_plan"] = deepcopy(preview_plan or {})
                turn_debug["accepted"] = True
                turn_debug["phase_after"] = bridge_session.get("phase")
                turn_debug["unmet_reentry_conditions_after"] = deepcopy(
                    projected_marked_reentry_context.get("unmet_reentry_conditions") or []
                )
                bridge_debug["turns"].append(turn_debug)
                bridge_debug["session"] = deepcopy(bridge_session)
                prepared_bridge_request["bridge_debug"] = bridge_debug
                self._set_last_bridge_debug(bridge_debug)
                continue

            cached_validated_plan = deepcopy(
                bridge_session.get("validated_deterministic_plan") or {}
            )
            normalization_warnings: list[str] = []
            if (
                response_source == "planner_compiler"
                and str(draft_status.get("status", "")).strip() == "preview_validated"
                and isinstance(cached_validated_plan, dict)
                and cached_validated_plan
            ):
                normalized_proposal = cached_validated_plan
            else:
                normalized_proposal, normalization_warnings = (
                    self._normalize_primitive_bridge_plan_with_warnings(
                        prepared_bridge_request=prepared_bridge_request,
                        plan=turn_response.get("plan") or {},
                    )
                )
            bridge_debug["normalized_proposal"] = (
                deepcopy(normalized_proposal) if isinstance(normalized_proposal, dict) else None
            )
            if not isinstance(normalized_proposal, dict):
                normalization_error = str(
                    normalization_warnings[-1]
                    if normalization_warnings
                    else "final_plan did not normalize into a compilable bridge proposal"
                ).strip()
                _retry_n = int(bridge_session.get("final_retry_count", 0) or 0) + 1
                self.logger.warning(
                    "[Bridge] Turn %d — final_plan normalization FAILED (retry %d/%d): %s",
                    _turn_idx, _retry_n, max_final_retries, normalization_error,
                )
                bridge_session["final_retry_count"] = _retry_n
                missing_observation_feedback = self._bridge_missing_observation_feedback(
                    prepared_bridge_request=prepared_bridge_request,
                    marked_reentry_context=current_marked_reentry_context,
                )
                bridge_session["last_plan_failure"] = {
                    "kind": "final_plan_invalid",
                    "message": normalization_error,
                    "unmet_reentry_conditions": deepcopy(
                        current_marked_reentry_context.get(
                            "unmet_reentry_conditions"
                        )
                        or []
                    ),
                }
                bridge_session["draft_final_plan"] = deepcopy(turn_response.get("plan") or {})
                bridge_session["draft_final_plan_status"] = {
                    "compile_path": "llm_repair",
                    "status": "normalization_failed",
                    "error": normalization_error,
                }
                bridge_session["validated_deterministic_plan"] = {}
                prepared_bridge_request["bridge_session"] = bridge_session
                if missing_observation_feedback:
                    self._append_bridge_validation_feedback(
                        prepared_bridge_request,
                        kind="insufficient_observation",
                        message=missing_observation_feedback,
                    )
                self._append_bridge_validation_feedback(
                    prepared_bridge_request,
                    kind="final_plan_invalid",
                    message=normalization_error,
                    )
                bridge_debug = deepcopy(prepared_bridge_request.get("bridge_debug") or {})
                turn_debug["error"] = normalization_error
                bridge_debug["turns"].append(turn_debug)
                prepared_bridge_request["bridge_debug"] = bridge_debug
                self._set_last_bridge_debug(bridge_debug)
                if int(bridge_session.get("final_retry_count", 0) or 0) >= max_final_retries:
                    break
                continue

            normalized_proposal["bridge_event_summary"] = self._bridge_event_summary_from_macro_tasks(
                prepared_bridge_request=prepared_bridge_request,
                proposal=normalized_proposal,
            )
            turn_debug["realized_bridge_events"] = deepcopy(
                normalized_proposal.get("bridge_event_summary") or []
            )
            bridge_debug["bridge_event_summary"] = deepcopy(
                normalized_proposal.get("bridge_event_summary") or []
            )
            approved_events = list(bridge_session.get("approved_bridge_events") or [])
            realizes_approved, realization_error = self._bridge_final_plan_realizes_approved_events(
                approved_events=deepcopy(approved_events),
                realized_events=deepcopy(normalized_proposal.get("bridge_event_summary") or []),
            )
            if not realizes_approved:
                self.logger.warning(
                    "[Bridge] Turn %d — final_plan EVENT MISMATCH: %s",
                    _turn_idx,
                    realization_error or "final_plan did not realize approved bridge events",
                )
                bridge_session["final_retry_count"] = int(
                    bridge_session.get("final_retry_count", 0) or 0
                ) + 1
                bridge_session["last_plan_failure"] = {
                    "kind": "final_plan_event_mismatch",
                    "message": realization_error or "final_plan did not realize approved bridge events",
                    "approved_bridge_events": deepcopy(approved_events),
                }
                bridge_session["draft_final_plan"] = deepcopy(normalized_proposal)
                bridge_session["draft_final_plan_status"] = {
                    "compile_path": "llm_repair",
                    "status": "event_mismatch",
                    "error": realization_error or "final_plan did not realize approved bridge events",
                }
                prepared_bridge_request["bridge_session"] = bridge_session
                self._append_bridge_validation_feedback(
                    prepared_bridge_request,
                    kind="final_plan_event_mismatch",
                    message=realization_error or "final_plan did not realize approved bridge events",
                )
                bridge_debug = deepcopy(prepared_bridge_request.get("bridge_debug") or {})
                turn_debug["error"] = realization_error or "final_plan did not realize approved bridge events"
                bridge_debug["turns"].append(turn_debug)
                prepared_bridge_request["bridge_debug"] = bridge_debug
                self._set_last_bridge_debug(bridge_debug)
                if int(bridge_session.get("final_retry_count", 0) or 0) >= max_final_retries:
                    break
                continue

            feasibility_decisions: list[dict[str, Any]] = []
            final_plan_infeasible_error = ""
            allowed_operation_kinds = all_registered_operation_kinds()
            for event in normalized_proposal.get("bridge_event_summary") or []:
                if not isinstance(event, dict):
                    continue
                resource_jid = str(event.get("resource_jid", "") or "").strip()
                part_name = str(event.get("part_name", "") or "").strip()
                operation_kind = self._bridge_event_operation_kind(event)
                if not resource_jid or operation_kind not in allowed_operation_kinds:
                    continue
                decision = self._bridge_feasibility_decision(
                    prepared_bridge_request,
                    resource_jid=resource_jid,
                    operation_kind=operation_kind,
                    part_name=part_name or None,
                )
                feasibility_decisions.append(decision)
                if not decision.get("allowed", False):
                    final_plan_infeasible_error = (
                        f"final_plan bridge event "
                        f"'{str(event.get('event_name', '')).strip() or resource_jid}' "
                        f"is infeasible on {resource_jid}: "
                        f"{decision.get('reason') or 'unknown reason'}"
                    )
                    break
            turn_debug["feasibility_decisions"] = deepcopy(feasibility_decisions)
            if final_plan_infeasible_error:
                self.logger.warning(
                    "[Bridge] Turn %d — final_plan INFEASIBLE: %s",
                    _turn_idx, final_plan_infeasible_error,
                )
                bridge_session["final_retry_count"] = int(
                    bridge_session.get("final_retry_count", 0) or 0
                ) + 1
                bridge_session["last_plan_failure"] = {
                    "kind": "final_plan_infeasible",
                    "message": final_plan_infeasible_error,
                    "approved_bridge_events": deepcopy(approved_events),
                }
                bridge_session["draft_final_plan"] = deepcopy(normalized_proposal)
                bridge_session["draft_final_plan_status"] = {
                    "compile_path": "llm_repair",
                    "status": "infeasible",
                    "error": final_plan_infeasible_error,
                }
                prepared_bridge_request["bridge_session"] = bridge_session
                self._append_bridge_validation_feedback(
                    prepared_bridge_request,
                    kind="final_plan_infeasible",
                    message=final_plan_infeasible_error,
                )
                bridge_debug = deepcopy(prepared_bridge_request.get("bridge_debug") or {})
                turn_debug["error"] = final_plan_infeasible_error
                bridge_debug["turns"].append(turn_debug)
                prepared_bridge_request["bridge_debug"] = bridge_debug
                self._set_last_bridge_debug(bridge_debug)
                if int(bridge_session.get("final_retry_count", 0) or 0) >= max_final_retries:
                    break
                continue

            safety_ok, safety_error = self._bridge_validate_safety_constraints(
                prepared_bridge_request,
                events=deepcopy(normalized_proposal.get("bridge_event_summary") or []),
            )
            bridge_debug["bridge_safety_check"] = {
                "accepted": bool(safety_ok),
                "reason": str(safety_error or "").strip(),
                "context": deepcopy(prepared_bridge_request.get("bridge_safety_context") or {}),
            }
            if not safety_ok:
                self.logger.warning(
                    "[Bridge] Turn %d — final_plan SAFETY REJECTED: %s",
                    _turn_idx,
                    safety_error or "final_plan violated bridge safety constraints",
                )
                bridge_session["final_retry_count"] = int(
                    bridge_session.get("final_retry_count", 0) or 0
                ) + 1
                bridge_session["last_plan_failure"] = {
                    "kind": "bridge_safety_rejected",
                    "message": safety_error or "final_plan violated bridge safety constraints",
                    "approved_bridge_events": deepcopy(approved_events),
                }
                bridge_session["draft_final_plan"] = deepcopy(normalized_proposal)
                bridge_session["draft_final_plan_status"] = {
                    "compile_path": "llm_repair",
                    "status": "bridge_safety_rejected",
                    "error": safety_error or "final_plan violated bridge safety constraints",
                }
                prepared_bridge_request["bridge_session"] = bridge_session
                self._append_bridge_validation_feedback(
                    prepared_bridge_request,
                    kind="bridge_safety_rejected",
                    message=safety_error or "final_plan violated bridge safety constraints",
                )
                bridge_debug = deepcopy(prepared_bridge_request.get("bridge_debug") or {})
                turn_debug["error"] = safety_error or "final_plan violated bridge safety constraints"
                bridge_debug["turns"].append(turn_debug)
                prepared_bridge_request["bridge_debug"] = bridge_debug
                self._set_last_bridge_debug(bridge_debug)
                if int(bridge_session.get("final_retry_count", 0) or 0) >= max_final_retries:
                    break
                continue

            projected_marked_reentry_context = self._bridge_marked_reentry_context(
                prepared_bridge_request,
                projected_resource_snapshots=deepcopy(
                    normalized_proposal.get("projected_resource_snapshots") or {}
                ),
                projected_parts=deepcopy(normalized_proposal.get("projected_parts") or {}),
            )
            bridge_debug["marked_reentry_check"] = {
                "accepted": not bool(
                    projected_marked_reentry_context.get("unmet_reentry_conditions") or []
                ),
                "reason": "",
                "context": deepcopy(projected_marked_reentry_context),
            }
            if projected_marked_reentry_context.get("unmet_reentry_conditions"):
                _unmet_count = len(
                    projected_marked_reentry_context.get("unmet_reentry_conditions") or []
                )
                self.logger.warning(
                    "[Bridge] Turn %d — M_bridge check FAILED: %d conditions still unmet",
                    _turn_idx, _unmet_count,
                )
                bridge_session["final_retry_count"] = int(
                    bridge_session.get("final_retry_count", 0) or 0
                ) + 1
                marked_error = self._format_bridge_marked_reentry_feedback(
                    projected_marked_reentry_context
                )
                bridge_session["last_plan_failure"] = {
                    "kind": "marked_reentry_conditions_unmet",
                    "message": marked_error,
                    "unmet_reentry_conditions": deepcopy(
                        projected_marked_reentry_context.get("unmet_reentry_conditions") or []
                    ),
                    "pending_suffix_summary": deepcopy(
                        projected_marked_reentry_context.get("pending_suffix_summary") or []
                    ),
                }
                bridge_session["draft_final_plan"] = deepcopy(normalized_proposal)
                bridge_session["draft_final_plan_status"] = {
                    "compile_path": "llm_repair",
                    "status": "marked_reentry_unmet",
                    "error": marked_error,
                }
                prepared_bridge_request["bridge_session"] = bridge_session
                self._append_bridge_validation_feedback(
                    prepared_bridge_request,
                    kind="marked_reentry_conditions_unmet",
                    message=marked_error,
                )
                missing_observation_feedback = self._bridge_missing_observation_feedback(
                    prepared_bridge_request=prepared_bridge_request,
                    marked_reentry_context=projected_marked_reentry_context,
                )
                if missing_observation_feedback:
                    self._append_bridge_validation_feedback(
                        prepared_bridge_request,
                        kind="insufficient_observation",
                        message=missing_observation_feedback,
                    )
                bridge_debug = deepcopy(prepared_bridge_request.get("bridge_debug") or {})
                turn_debug["error"] = marked_error
                bridge_debug["marked_reentry_check"] = {
                    "accepted": False,
                    "reason": marked_error,
                    "context": deepcopy(projected_marked_reentry_context),
                }
                bridge_debug["turns"].append(turn_debug)
                prepared_bridge_request["bridge_debug"] = bridge_debug
                self._set_last_bridge_debug(bridge_debug)
                if int(bridge_session.get("final_retry_count", 0) or 0) >= max_final_retries:
                    break
                continue

            bridge_resources = deepcopy(prepared_bridge_request.get("bridge_resources") or {})
            if bridge_resources and not self._bridge_restores_modeled_continuation(
                proposal=normalized_proposal,
                goal_state=str(prepared_bridge_request.get("goal_state", "") or "unknown"),
                tools_catalog=deepcopy(list(prepared_bridge_request.get("tools_catalog") or [])),
                bridge_resources=bridge_resources,
                fallback_part_tracker=deepcopy(prepared_bridge_request.get("part_tracker") or {}),
            ):
                self.logger.warning(
                    "[Bridge] Turn %d — final_plan MODELED CONTINUATION REJECTED: %s",
                    _turn_idx,
                    "projected bridge state satisfies marked re-entry conditions but DES still found no resumable modeled continuation",
                )
                bridge_session["final_retry_count"] = int(
                    bridge_session.get("final_retry_count", 0) or 0
                ) + 1
                modeled_error = (
                    "projected bridge state satisfies marked re-entry conditions but DES still found "
                    "no resumable modeled continuation"
                )
                modeled_gap = self._bridge_modeled_continuation_gap(
                    proposal=deepcopy(normalized_proposal),
                    goal_state=str(prepared_bridge_request.get("goal_state", "") or "unknown"),
                    tools_catalog=deepcopy(list(prepared_bridge_request.get("tools_catalog") or [])),
                    bridge_resources=bridge_resources,
                    fallback_part_tracker=deepcopy(prepared_bridge_request.get("part_tracker") or {}),
                )
                bridge_session["modeled_continuation_gap"] = deepcopy(modeled_gap)
                bridge_session["last_plan_failure"] = {
                    "kind": "modeled_continuation_rejected",
                    "message": modeled_error,
                    "unmet_reentry_conditions": deepcopy(
                        projected_marked_reentry_context.get("unmet_reentry_conditions") or []
                    ),
                    "pending_suffix_summary": deepcopy(
                        projected_marked_reentry_context.get("pending_suffix_summary") or []
                    ),
                    "modeled_continuation_gap": deepcopy(modeled_gap),
                }
                bridge_session["draft_final_plan"] = deepcopy(normalized_proposal)
                bridge_session["draft_final_plan_status"] = {
                    "compile_path": "llm_repair",
                    "status": "modeled_continuation_rejected",
                    "error": modeled_error,
                }
                prepared_bridge_request["bridge_session"] = bridge_session
                self._append_bridge_validation_feedback(
                    prepared_bridge_request,
                    kind="modeled_continuation_rejected",
                    message=modeled_error,
                )
                bridge_debug = deepcopy(prepared_bridge_request.get("bridge_debug") or {})
                turn_debug["error"] = modeled_error
                bridge_debug["modeled_continuation_check"] = {
                    "accepted": False,
                    "reason": modeled_error,
                }
                bridge_debug["turns"].append(turn_debug)
                prepared_bridge_request["bridge_debug"] = bridge_debug
                self._set_last_bridge_debug(bridge_debug)
                if int(bridge_session.get("final_retry_count", 0) or 0) >= max_final_retries:
                    break
                continue

            bridge_debug["modeled_continuation_check"] = {
                "accepted": True,
                "reason": "",
            }
            bridge_debug["marked_reentry_check"] = {
                "accepted": True,
                "reason": "",
                "context": deepcopy(projected_marked_reentry_context),
            }
            normalized_proposal["plan_rewrite"] = self._bridge_synthesize_plan_rewrite(
                prepared_bridge_request
            )
            bridge_debug["status"] = "accepted"
            _macro_count = len(normalized_proposal.get("macro_tasks") or [])
            _step_count = sum(
                len(m.get("primitive_steps") or m.get("steps") or [])
                for m in (normalized_proposal.get("macro_tasks") or [])
                if isinstance(m, dict)
            )
            self.logger.info(
                "[Bridge] Turn %d — final_plan ACCEPTED: %d macros, %d steps (%.1fs total)",
                _turn_idx, _macro_count, _step_count, time.monotonic() - _bridge_t0,
            )
            bridge_debug["normalized_proposal"] = deepcopy(normalized_proposal)
            bridge_debug["bridge_event_summary"] = deepcopy(
                normalized_proposal.get("bridge_event_summary") or []
            )
            bridge_debug["approved_bridge_events"] = deepcopy(approved_events)
            bridge_debug["feasibility_decisions"] = deepcopy(feasibility_decisions)
            bridge_debug["plan_rewrite"] = deepcopy(normalized_proposal.get("plan_rewrite") or {})
            bridge_debug["compile_path"] = compile_path
            bridge_debug["turns"].append({**turn_debug, "accepted": True})
            bridge_session["phase"] = "review"
            bridge_session["last_plan_failure"] = {}
            bridge_session["draft_final_plan"] = {}
            bridge_session["draft_final_plan_status"] = {
                "compile_path": compile_path,
                "status": "accepted",
            }
            prepared_bridge_request["bridge_session"] = bridge_session
            bridge_debug["session"] = deepcopy(bridge_session)
            bridge_debug["phase"] = "review"
            prepared_bridge_request["bridge_debug"] = bridge_debug
            self._set_last_bridge_debug(bridge_debug)
            return normalized_proposal

        self.logger.warning(
            "[Bridge] Session EXHAUSTED — no accepted plan after %d turns (%.1fs total)",
            int(
                (prepared_bridge_request.get("bridge_session") or {}).get("turn_index", 0) or 0
            ),
            time.monotonic() - _bridge_t0,
        )
        bridge_session = dict(prepared_bridge_request.get("bridge_session") or {})
        bridge_debug = deepcopy(prepared_bridge_request.get("bridge_debug") or {})
        bridge_debug["status"] = "rejected_budget_exhausted"
        bridge_debug["modeled_continuation_check"] = {
            "accepted": False,
            "reason": "bridge session exhausted its turn/final-plan budget",
        }
        bridge_debug["session"] = deepcopy(bridge_session)
        prepared_bridge_request["bridge_debug"] = bridge_debug
        self._set_last_bridge_debug(bridge_debug)
        return None

    async def _request_bridge_proposal(
        self,
        *,
        stuck_state: dict[str, Any],
        P_id: list[str],
        ra_jid: str,
        goal_state: str,
        tools_catalog: list[dict[str, Any]],
        part_tracker: dict[str, Any],
        obligation_targets: list[dict[str, Any]],
        bridge_feedback: str,
        resource_states: dict[str, dict[str, Any]],
        default_resource_state: str,
        part_states: dict[str, Any],
        part_locations: dict[str, Any],
    ) -> dict[str, Any] | None:
        prepared_bridge_request = await self.prepare_bridge_request(
            stuck_state=stuck_state,
            P_id=P_id,
            ra_jid=ra_jid,
            goal_state=goal_state,
            tools_catalog=tools_catalog,
            part_tracker=part_tracker,
            obligation_targets=obligation_targets,
            bridge_feedback=bridge_feedback,
            resource_states=resource_states,
            default_resource_state=default_resource_state,
            part_states=part_states,
            part_locations=part_locations,
        )
        return await self.execute_prepared_bridge_request(prepared_bridge_request)

    def _bridge_summary(self, proposal: dict[str, Any] | None) -> list[str]:
        if not isinstance(proposal, dict):
            return []
        summary: list[str] = []
        primary_obligation = proposal.get("primary_obligation") or {}
        rule_id = str(primary_obligation.get("rule_id") or "").strip() if isinstance(primary_obligation, dict) else ""
        if rule_id:
            summary.append(f"rule:{rule_id}")
        for task in proposal.get("macro_tasks") or []:
            if not isinstance(task, dict):
                continue
            macro_name = str(task.get("macro_name", "")).strip()
            if macro_name:
                summary.append(macro_name)
            for step in task.get("primitive_steps") or []:
                if not isinstance(step, dict):
                    continue
                primitive = str(step.get("primitive", "")).strip()
                if primitive:
                    summary.append(primitive)
        macro_name = str(proposal.get("macro_name", "")).strip()
        if macro_name:
            summary.append(macro_name)
        function_name = str(proposal.get("function_name", "")).strip()
        if function_name:
            summary.append(function_name)
        for step in proposal.get("primitive_steps") or []:
            if not isinstance(step, dict):
                continue
            primitive = str(step.get("primitive", "")).strip()
            if primitive:
                summary.append(primitive)
        for step in proposal.get("macro_steps") or []:
            if not isinstance(step, dict):
                continue
            fn = str(step.get("function_name", "")).strip()
            if fn:
                summary.append(fn)
        return summary

    @staticmethod
    def _primitive_bridge_macro_tasks(proposal: dict[str, Any]) -> list[dict[str, Any]]:
        raw_tasks = proposal.get("macro_tasks")
        if isinstance(raw_tasks, list) and raw_tasks:
            return [dict(task) for task in raw_tasks if isinstance(task, dict)]
        if proposal.get("primitive_steps"):
            return [dict(proposal)]
        return []

    def _bridge_sequence_nodes(self, bridge_sequence_id: str) -> list[dict[str, Any]]:
        sequence_id = str(bridge_sequence_id or "").strip()
        if not sequence_id:
            return []
        nodes = [
            node
            for node in self.nodes
            if node.get("type") == "task"
            and str(node.get("bridge_sequence_id", "")).strip() == sequence_id
        ]
        nodes.sort(
            key=lambda node: (
                int(node.get("bridge_sequence_index") or 0),
                str(node.get("id", "")),
            )
        )
        return nodes

    def remove_bridge_sequence_tail(
        self,
        *,
        bridge_sequence_id: str,
        completed_task_id: str = "",
    ) -> list[dict[str, Any]]:
        sequence_nodes = self._bridge_sequence_nodes(bridge_sequence_id)
        if not sequence_nodes:
            return []

        completed_task_id = str(completed_task_id or "").strip()
        cutoff_index = -1
        if completed_task_id:
            current_node = self._find_node(completed_task_id)
            if current_node is not None:
                cutoff_index = int(current_node.get("bridge_sequence_index") or 0)

        deletions: list[dict[str, Any]] = []
        for node in sequence_nodes:
            node_id = str(node.get("id", "")).strip()
            sequence_index = int(node.get("bridge_sequence_index") or 0)
            if completed_task_id and node_id == completed_task_id:
                continue
            if cutoff_index > 0 and sequence_index <= cutoff_index:
                continue
            deletions.append(
                {
                    "id": node_id,
                    "delete": True,
                    "change_reason": (
                        "DELETION: remove remaining approved bridge tail "
                        f"for sequence {bridge_sequence_id}"
                    ),
                }
            )

        if deletions:
            self._apply_replan_patch(deletions)
        return deletions

    def can_execute_task_from_system_state(
        self,
        *,
        task_id: str,
        system_coordination_state: dict[str, Any] | None,
        part_tracker: dict[str, Any] | None,
    ) -> bool:
        from cais_spade_llm.agents.intelligent_product.replanner.des_search.resource_bidding import (
            simulate_catalog_transition,
        )

        node = self._find_node(task_id)
        if not isinstance(node, dict) or node.get("type") != "task":
            return False

        function_name = str(node.get("function_name", "")).strip()
        resource_jid = str(node.get("resource_jid", "")).strip()
        if not function_name or not resource_jid or function_name == "execute_recovery_macro":
            return False

        tools_catalog = getattr(self.product_agent, "tools_catalog", [])
        default_resource_state = self._default_resource_state(tools_catalog)
        resource_states = self._extract_resource_states(system_coordination_state or {})
        part_states: dict[str, Any] = {}
        part_locations: dict[str, Any] = {}
        for part_name, raw_info in (part_tracker or {}).items():
            name = str(part_name or "").strip()
            if not name:
                continue
            info = raw_info if isinstance(raw_info, dict) else {}
            part_states[name] = info.get("state")
            part_locations[name] = info.get("location")

        x_c = self._build_resource_search_state(
            resource_jid=resource_jid,
            resource_states=resource_states,
            default_resource_state=default_resource_state,
            part_states=part_states,
            part_locations=part_locations,
        )
        resource = self._resource_by_jid(resource_jid)
        reachability = getattr(resource, "static_capabilities", {}).get("reachability", []) if resource else []
        staging_areas = getattr(resource, "static_capabilities", {}).get("staging_areas", {}) if resource else {}
        goal_state = self._resolve_goal_part_state(tools_catalog) or ""
        simulated = simulate_catalog_transition(
            x_c=x_c,
            tools=tools_catalog,
            resource_jid=resource_jid,
            function_name=function_name,
            params=dict(node.get("params") or {}),
            goal_state=goal_state,
            reachability=reachability,
            staging_areas=staging_areas,
        )
        return simulated is not None

    def _bridge_projected_search_state(
        self,
        *,
        resource_jid: str,
        bridge_resources: dict[str, dict[str, Any]],
        projected_resource_snapshots: dict[str, dict[str, Any]],
        part_states: dict[str, Any],
        part_locations: dict[str, Any],
    ) -> dict[str, Any]:
        resource_entry = dict(bridge_resources.get(resource_jid) or {})
        modeled_state = dict(resource_entry.get("modeled_state") or {})
        projected_snapshot = dict(projected_resource_snapshots.get(resource_jid) or {})
        search_state = {
            "resource_state": modeled_state.get("resource_state", "idle"),
            "current_part": modeled_state.get("current_part"),
            "current_location": modeled_state.get("current_location"),
            "part_states": dict(part_states),
            "part_locations": dict(part_locations),
        }
        if "current_state" in projected_snapshot:
            search_state["resource_state"] = projected_snapshot.get("current_state") or search_state["resource_state"]
        profile = get_resource_profile(str(projected_snapshot.get("resource_type") or "resource"))
        carried_entity_location = resource_snapshot_carried_entity_location(
            resource_jid=resource_jid,
            snapshot=projected_snapshot,
            profile=profile,
        )
        if carried_entity_location:
            carried_parts = [
                str(part_name).strip()
                for part_name, location in (part_locations or {}).items()
                if str(part_name).strip()
                and str(location or "").strip() == carried_entity_location
            ]
            if len(carried_parts) == 1:
                search_state["current_part"] = carried_parts[0]
        carried_entity = resource_snapshot_carried_entity(projected_snapshot, profile=profile)
        if search_state.get("current_part") in (None, "") and carried_entity not in (None, ""):
            search_state["current_part"] = carried_entity
        current_location = projected_snapshot.get("current_location")
        if current_location is not None:
            search_state["current_location"] = current_location
        return search_state

    def _bridge_restores_modeled_continuation(
        self,
        *,
        proposal: dict[str, Any],
        goal_state: str,
        tools_catalog: list[dict[str, Any]],
        bridge_resources: dict[str, dict[str, Any]],
        fallback_part_tracker: dict[str, Any],
    ) -> bool:
        diagnostics = self._bridge_modeled_continuation_gap(
            proposal=proposal,
            goal_state=goal_state,
            tools_catalog=tools_catalog,
            bridge_resources=bridge_resources,
            fallback_part_tracker=fallback_part_tracker,
        )
        return bool(diagnostics.get("restored", False))

    def _bridge_modeled_continuation_gap(
        self,
        *,
        proposal: dict[str, Any],
        goal_state: str,
        tools_catalog: list[dict[str, Any]],
        bridge_resources: dict[str, dict[str, Any]],
        fallback_part_tracker: dict[str, Any],
    ) -> dict[str, Any]:
        from cais_spade_llm.agents.intelligent_product.replanner.des_search.resource_bidding import (
            compute_bid,
        )

        if not goal_state:
            return {"restored": True, "goal_state": goal_state, "remaining_parts": [], "candidate_resources": []}

        projected_resource_snapshots = proposal.get("projected_resource_snapshots") or {}
        if not isinstance(projected_resource_snapshots, dict):
            projected_resource_snapshots = {}

        projected_parts_raw = proposal.get("projected_parts") or {}
        projected_parts = projected_parts_raw if isinstance(projected_parts_raw, dict) else {}
        part_states: dict[str, Any] = {}
        part_locations: dict[str, Any] = {}
        for part_name, raw_info in (fallback_part_tracker or {}).items():
            name = str(part_name or "").strip()
            if not name:
                continue
            info = raw_info if isinstance(raw_info, dict) else {}
            part_states[name] = info.get("state")
            part_locations[name] = info.get("location")
        for part_name, raw_info in projected_parts.items():
            name = str(part_name or "").strip()
            if not name:
                continue
            info = raw_info if isinstance(raw_info, dict) else {}
            if "state" in info:
                part_states[name] = info.get("state")
            if "location" in info:
                part_locations[name] = info.get("location")

        remaining_parts = [
            name for name, state in part_states.items()
            if name and state != goal_state
        ]
        if not remaining_parts:
            return {
                "restored": True,
                "goal_state": goal_state,
                "remaining_parts": [],
                "candidate_resources": [],
                "pending_suffix_summary": deepcopy(
                    (proposal.get("pending_suffix_summary") or [])
                ),
            }

        diagnostics = {
            "restored": False,
            "goal_state": goal_state,
            "remaining_parts": [
                {
                    "part_name": name,
                    "current_state": part_states.get(name),
                    "current_location": part_locations.get(name),
                }
                for name in remaining_parts
            ],
            "candidate_resources": [],
        }

        for resource in self.resource_agents:
            resource_jid = str(getattr(resource, "jid", "")).strip()
            if not resource_jid:
                continue
            if resource_jid not in bridge_resources and resource_jid not in projected_resource_snapshots:
                continue

            x_c = self._bridge_projected_search_state(
                resource_jid=resource_jid,
                bridge_resources=bridge_resources,
                projected_resource_snapshots=projected_resource_snapshots,
                part_states=part_states,
                part_locations=part_locations,
            )
            bid = compute_bid(
                x_c=x_c,
                P_id=remaining_parts,
                goal_state=goal_state,
                tools=tools_catalog,
                reachability=getattr(resource, "static_capabilities", {}).get("reachability", []),
                staging_areas=getattr(resource, "static_capabilities", {}).get("staging_areas", {}),
                resource_jid=resource_jid,
            )
            candidate = {
                "resource_jid": resource_jid,
                "resource_state": x_c.get("resource_state"),
                "current_part": x_c.get("current_part"),
                "current_location": x_c.get("current_location"),
                "has_bid": bool(bid and bid.str_e),
            }
            if bid and bid.str_e:
                candidate["complete"] = bool(getattr(bid, "complete", False))
                first_event = dict((bid.str_e or [])[0] or {})
                if first_event:
                    candidate["next_function_name"] = str(
                        first_event.get("function_name") or ""
                    ).strip()
                    params = dict(first_event.get("params") or {})
                    if params:
                        candidate["next_params"] = deepcopy(params)
                diagnostics["candidate_resources"].append(candidate)
                diagnostics["restored"] = True
            else:
                diagnostics["candidate_resources"].append(candidate)
        return diagnostics

        return False
