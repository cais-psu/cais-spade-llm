"""Bridge session preparation and preprogrammed proposal validation."""

from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.v3.bridge_adapters import (
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
                        parts_ctx = (
                            (prepared_bridge_request.get("grounding_context") or {})
                            .get("parts", {})
                        )
                        part_ctx = parts_ctx.get(part_name) if isinstance(parts_ctx, dict) else {}
                        if not isinstance(part_ctx, dict):
                            part_ctx = {}
                        target_ctx = part_ctx.get("target") or {}
                        if not isinstance(target_ctx, dict):
                            target_ctx = {}
                        target_location = str(target_ctx.get("location") or "").strip()
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
            "repair_mode": "recover",
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
        from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.v3.bridge_generation import (
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
        """Execute a prepared bridge request using the v2 universal repair session."""
        if not isinstance(prepared_bridge_request, dict):
            raise ValueError("prepared bridge request is missing")
        result = await self.run_universal_repair_session(prepared_bridge_request)
        # Translate v2 result to the proposal dict expected by callers.
        if not isinstance(result, dict):
            return None
        validated = result.get("validated_program")
        if validated is None:
            return None
        return validated

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
