"""Deterministic grounding compiler for bridge outline tasks.

This module converts a free-form outline task into one uniquely grounded bridge
action before downstream feasibility checks run. It does not perform resource
feasibility or temporal validation; it only determines whether the task is
specific enough to evaluate.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any


def _first_non_empty(mapping: dict[str, Any], *field_names: str) -> Any:
    for field_name in field_names:
        if field_name not in mapping:
            continue
        value = mapping.get(field_name)
        if value in (None, "", [], {}):
            continue
        return deepcopy(value)
    return None


_EXACT_STATE_UNAVAILABLE = object()


def _exact_mapping_value(mapping: dict[str, Any], field_name: str) -> Any:
    if field_name not in mapping:
        return _EXACT_STATE_UNAVAILABLE
    return deepcopy(mapping.get(field_name))


def _dedupe_tokens(values: list[str]) -> list[str]:
    deduped: list[str] = []
    for value in values:
        token = str(value or "").strip()
        if token and token not in deduped:
            deduped.append(token)
    return deduped


def _task_resource_jid(task: dict[str, Any]) -> str:
    return str(
        task.get("resource_jid")
        or ""
    ).strip()


def _task_part_name(task: dict[str, Any]) -> str:
    return str(
        task.get("part_name")
        or ""
    ).strip()


def _task_action_target(task: dict[str, Any]) -> dict[str, Any]:
    action_target = task.get("action_target")
    normalized = dict(action_target) if isinstance(action_target, dict) else {}
    start_state = dict(task.get("expected_start_state") or {})
    end_state = dict(task.get("expected_end_state") or {})
    source_ref = str(
        task.get("source_ref")
        or _state_location_token(start_state)
        or ""
    ).strip()
    target_ref = str(
        task.get("target_ref")
        or _state_location_token(end_state)
        or ""
    ).strip()
    if source_ref and "source_location" not in normalized and "source_ref" not in normalized:
        normalized["source_ref"] = source_ref
        normalized["source_location"] = source_ref
    if target_ref and "target_location" not in normalized and "target_ref" not in normalized:
        normalized["target_ref"] = target_ref
        normalized["target_location"] = target_ref
    return normalized


def _state_location_token(state: dict[str, Any]) -> str:
    return str(
        _first_non_empty(
            state,
            "part_location",
            "resource_location",
            "location",
            "current_location",
            "named_pose",
        )
        or ""
    ).strip()


def _state_part_location_token(state: dict[str, Any]) -> str:
    return str(
        _first_non_empty(
            state,
            "part_location",
            "location",
            "current_location",
        )
        or ""
    ).strip()


def _state_resource_location_token(state: dict[str, Any]) -> str:
    return str(
        _first_non_empty(
            state,
            "resource_location",
            "named_pose",
            "location",
            "current_location",
        )
        or ""
    ).strip()


def _state_pose_value(state: dict[str, Any]) -> dict[str, Any] | None:
    pose = state.get("position") or state.get("pose") or state.get("current_pose")
    if not isinstance(pose, dict) or "x" not in pose:
        return None
    return deepcopy(pose)


def _state_pose_ref_token(state: dict[str, Any]) -> str:
    token = str(
        _first_non_empty(state, "current_pose_ref", "pose_ref", "named_pose") or ""
    ).strip()
    if token:
        return token
    current_pose = state.get("current_pose")
    if isinstance(current_pose, str):
        return str(current_pose or "").strip()
    if isinstance(current_pose, dict):
        return str(current_pose.get("named_pose") or "").strip()
    pose = state.get("position") or state.get("pose")
    if isinstance(pose, dict):
        return str(pose.get("named_pose") or "").strip()
    return ""


def _state_resource_state_token(state: dict[str, Any]) -> str:
    return str(
        _first_non_empty(state, "current_state", "state", "resource_state") or ""
    ).strip()


def _state_part_state_token(state: dict[str, Any]) -> str:
    return str(
        _first_non_empty(state, "part_state", "part_status", "current_state", "state") or ""
    ).strip()


def _state_effect_part_state_token(state: dict[str, Any]) -> str:
    return str(
        _first_non_empty(
            state,
            "part_state",
            "part_status",
            "current_state",
            "state",
        )
        or ""
    ).strip()


def _part_row_location_token(part_row: dict[str, Any]) -> str:
    token = str(
        _first_non_empty(part_row, "current_location", "location", "current_pose_ref")
        or ""
    ).strip()
    if token:
        return token
    if _part_row_pose_value(part_row) is not None:
        return "observed_pose"
    return ""


def _part_row_pose_value(part_row: dict[str, Any]) -> dict[str, Any] | None:
    pose = part_row.get("observed_pose") or part_row.get("pose") or part_row.get("position")
    if not isinstance(pose, dict) or "x" not in pose:
        return None
    return deepcopy(pose)


def _part_row_holder_token(part_row: dict[str, Any]) -> str:
    return str(
        _first_non_empty(part_row, "current_holder_resource_jid", "holder_resource_jid")
        or ""
    ).strip()


def _state_part_holder_token(state: dict[str, Any]) -> str:
    return str(
        _first_non_empty(state, "part_holder_resource_jid", "current_holder_resource_jid")
        or ""
    ).strip()


def _part_row_observed_pose_aliases(part_row: dict[str, Any]) -> set[str]:
    aliases = {
        "observed_pose",
    }
    observed_store_as = str(part_row.get("observed_store_as") or "").strip()
    if observed_store_as:
        aliases.add(observed_store_as)
    for raw_alias in (part_row.get("observed_aliases") or []):
        alias = str(raw_alias or "").strip()
        if alias:
            aliases.add(alias)
    return aliases


def _canonicalize_observed_pose_location_token(
    location_token: str,
    *,
    part_row: dict[str, Any],
) -> str:
    normalized_location = str(location_token or "").strip()
    if not normalized_location:
        return ""
    if normalized_location in _part_row_observed_pose_aliases(part_row):
        return "observed_pose"
    return normalized_location


def _resource_named_pose_tokens(resource_row: dict[str, Any]) -> list[str]:
    named_pose_tokens: list[str] = []
    raw_named_poses = resource_row.get("named_poses")
    if isinstance(raw_named_poses, dict):
        named_pose_tokens.extend(
            str(pose_name).strip()
            for pose_name in raw_named_poses.keys()
            if str(pose_name).strip()
        )
    else:
        named_pose_tokens.extend(
            str(pose_name).strip()
            for pose_name in (raw_named_poses or [])
            if str(pose_name).strip()
        )
    named_pose_tokens.extend(
        str(pose_name).strip()
        for pose_name in (resource_row.get("available_named_poses") or [])
        if str(pose_name).strip()
    )
    return _dedupe_tokens(named_pose_tokens)


def _resource_state_tokens(
    resource_row: dict[str, Any],
    *,
    resources_by_jid: dict[str, dict[str, Any]] | None = None,
) -> list[str]:
    state_tokens: list[str] = [
        str(resource_row.get("current_state") or "").strip(),
    ]
    for raw_resource_row in (resources_by_jid or {}).values():
        state_token = str(dict(raw_resource_row or {}).get("current_state") or "").strip()
        if state_token:
            state_tokens.append(state_token)
    state_tokens.extend(
        str(state_token).strip()
        for state_token in (
            resource_row.get("supported_recovery_states")
            or resource_row.get("available_recovery_states")
            or []
        )
        if str(state_token).strip()
    )
    return _dedupe_tokens(state_tokens)


def _explicit_state_location_tokens(state: dict[str, Any]) -> list[str]:
    return _dedupe_tokens(
        [
            str(state.get("part_location") or "").strip(),
            str(state.get("resource_location") or "").strip(),
            str(state.get("location") or "").strip(),
            str(state.get("current_location") or "").strip(),
        ]
    )


def _known_location_tokens(
    *,
    resources_by_jid: dict[str, dict[str, Any]],
    parts_by_name: dict[str, dict[str, Any]],
    part_name: str = "",
) -> list[str]:
    location_tokens: list[str] = []
    part_names = [part_name] if part_name else list(parts_by_name)
    for candidate_part_name in part_names:
        row = dict(parts_by_name.get(candidate_part_name) or {})
        if not row:
            continue
        location_tokens.extend(
            str(row.get(field_name) or "").strip()
            for field_name in ("current_location", "location", "origin_location", "goal_location")
            if str(row.get(field_name) or "").strip()
        )
        holder = _part_row_holder_token(row)
        if holder:
            location_tokens.extend([holder, f"{holder}_gripper"])
        location_tokens.extend(_part_row_observed_pose_aliases(row))
    for resource_jid, raw_resource_row in (resources_by_jid or {}).items():
        resource_row = dict(raw_resource_row or {})
        if resource_jid:
            location_tokens.extend([resource_jid, f"{resource_jid}_gripper"])
        current_location = str(resource_row.get("current_location") or "").strip()
        if current_location:
            location_tokens.append(current_location)
        for field_name in ("reachability", "reachable_locations", "known_locations"):
            location_tokens.extend(
                str(token).strip()
                for token in (resource_row.get(field_name) or [])
                if str(token).strip()
            )
        staging_areas = dict(resource_row.get("staging_areas") or {})
        for name, staging_row in staging_areas.items():
            clean_name = str(name).strip()
            if not clean_name or not isinstance(staging_row, dict):
                continue
            location_tokens.extend([clean_name, f"{clean_name}@anchor"])
            anchor_pose = dict(
                staging_row.get("anchor_pose")
                or staging_row.get("board_center")
                or {}
            )
            coords: list[str] = []
            for axis in ("x", "y", "z"):
                try:
                    coords.append(f"{axis}={float(anchor_pose.get(axis)):.2f}")
                except (TypeError, ValueError):
                    coords = []
                    break
            if coords:
                location_tokens.append(f"{clean_name}@anchor({','.join(coords)})")
    return _dedupe_tokens(location_tokens)


def _task_part_references(task: dict[str, Any]) -> list[str]:
    tokens: list[str] = []
    explicit_part_name = _task_part_name(task)
    if explicit_part_name:
        tokens.append(explicit_part_name)
    for state_key in ("expected_start_state", "expected_end_state"):
        state = task.get(state_key)
        if not isinstance(state, dict):
            continue
        for field_name in ("part_name", "held_part"):
            token = str(state.get(field_name) or "").strip()
            if token:
                tokens.append(token)
    return _dedupe_tokens(tokens)


def _part_names_matching_location(
    *,
    location_token: str,
    parts_by_name: dict[str, dict[str, Any]],
    include_current: bool = True,
    include_goal: bool = True,
) -> list[str]:
    normalized_location = str(location_token or "").strip()
    if not normalized_location or normalized_location == "observed_pose":
        return []
    matches: list[str] = []
    for part_name, raw_row in (parts_by_name or {}).items():
        row = dict(raw_row or {})
        canonical_location = _canonicalize_observed_pose_location_token(
            normalized_location,
            part_row=row,
        )
        if canonical_location == "observed_pose" and _part_row_pose_value(row) is not None:
            matches.append(str(part_name))
            continue
        current_location = str(
            row.get("current_location") or row.get("location") or ""
        ).strip()
        goal_location = str(row.get("goal_location") or "").strip()
        if include_current and current_location and current_location == normalized_location:
            matches.append(str(part_name))
            continue
        if include_goal and goal_location and goal_location == normalized_location:
            matches.append(str(part_name))
    return _dedupe_tokens(matches)


def _task_part_binding(
    task: dict[str, Any],
    *,
    parts_by_name: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    explicit_candidates = _dedupe_tokens(
        [
            _task_part_name(task),
            str(dict(task.get("expected_start_state") or {}).get("part_name") or "").strip(),
            str(dict(task.get("expected_end_state") or {}).get("part_name") or "").strip(),
        ]
    )
    candidates = list(explicit_candidates)
    if not candidates:
        end_held_part = str(
            dict(task.get("expected_end_state") or {}).get("held_part") or ""
        ).strip()
        if end_held_part:
            candidates.append(end_held_part)
    if not candidates:
        start_held_part = str(
            dict(task.get("expected_start_state") or {}).get("held_part") or ""
        ).strip()
        if start_held_part:
            candidates.append(start_held_part)
    if not candidates:
        candidates.extend(_task_part_references(task))

    if not candidates:
        raw_action_target = task.get("action_target")
        action_target = dict(raw_action_target) if isinstance(raw_action_target, dict) else {}
        candidates.extend(
            _part_names_matching_location(
                location_token=str(
                    action_target.get("source_location")
                    or task.get("source_ref")
                    or ""
                ).strip(),
                parts_by_name=parts_by_name,
                include_current=True,
                include_goal=False,
            )
        )
        candidates.extend(
            _part_names_matching_location(
                location_token=str(
                    action_target.get("target_location")
                    or task.get("target_ref")
                    or ""
                ).strip(),
                parts_by_name=parts_by_name,
                include_current=False,
                include_goal=True,
            )
        )
        for state_key in ("expected_start_state", "expected_end_state"):
            candidates.extend(
                _part_names_matching_location(
                    location_token=_state_part_location_token(dict(task.get(state_key) or {})),
                    parts_by_name=parts_by_name,
                    include_current=True,
                    include_goal=True,
                )
            )

    deduped_candidates = _dedupe_tokens(candidates)
    explicit_part_name = str(task.get("part_name") or "").strip()
    effective_part_name = explicit_part_name
    if not effective_part_name and len(deduped_candidates) == 1:
        effective_part_name = deduped_candidates[0]

    return {
        "candidate_part_names": deduped_candidates,
        "effective_part_name": effective_part_name,
        "is_ambiguous": len(deduped_candidates) > 1,
        "is_unbound": not deduped_candidates,
    }


def _task_has_part_semantics(
    task: dict[str, Any],
    *,
    part_binding: dict[str, Any],
) -> bool:
    if part_binding.get("candidate_part_names"):
        action_target = _task_action_target(task)
        if str(action_target.get("source_location") or "").strip():
            return True
        if str(action_target.get("target_location") or "").strip():
            return True
        for state_key in ("expected_start_state", "expected_end_state"):
            state = dict(task.get(state_key) or {})
            state_part_name = str(state.get("part_name") or "").strip()
            held_part = str(state.get("held_part") or "").strip()
            if (
                state_part_name
                or held_part in set(part_binding.get("candidate_part_names") or [])
                or _state_part_holder_token(state)
                or _state_part_state_token(state)
                or _state_part_location_token(state)
                or (_state_pose_value(state) and state_part_name)
            ):
                return True
    return False


def _task_has_resource_semantics(
    task: dict[str, Any],
) -> bool:
    action_target = _task_action_target(task)
    if str(action_target.get("named_pose") or "").strip():
        return True
    for state_key in ("expected_start_state", "expected_end_state"):
        state = dict(task.get(state_key) or {})
        if (
            _state_resource_state_token(state)
            or _state_pose_value(state)
            or _state_pose_ref_token(state)
            or str(state.get("gripper_state") or "").strip()
            or _state_resource_location_token(state)
        ):
            return True
    return False


def _infer_task_kind(
    task: dict[str, Any],
    *,
    part_binding: dict[str, Any],
) -> str:
    if _task_has_part_semantics(task, part_binding=part_binding):
        return "part_handling"
    if _task_has_resource_semantics(task):
        return "resource_only"
    return "resource_only"


def _is_structured_continuation_resume(
    task: dict[str, Any],
    *,
    resource_row: dict[str, Any],
    part_row: dict[str, Any],
    part_name: str,
) -> bool:
    if not part_name:
        return False
    action_target = _task_action_target(task)
    end_state = dict(task.get("expected_end_state") or {})
    if str(action_target.get("source_location") or "").strip():
        return False
    target_location = str(
        action_target.get("target_location") or _state_location_token(end_state) or ""
    ).strip()
    goal_location = str(part_row.get("goal_location") or "").strip()
    if not target_location or not goal_location or target_location != goal_location:
        return False
    if str(end_state.get("held_part") or "").strip() == part_name:
        return False
    if _state_part_holder_token(end_state):
        return False

    part_pending = {
        str(item).strip()
        for item in (part_row.get("pending_nominal_task_ids") or [])
        if str(item).strip()
    }
    if not part_pending:
        return False
    resource_pending = {
        str(item).strip()
        for item in (resource_row.get("pending_nominal_task_ids") or [])
        if str(item).strip()
    }
    if resource_pending and not part_pending.intersection(resource_pending):
        return False
    return True


def _infer_operation_kind(
    task: dict[str, Any],
    *,
    task_kind: str,
    part_name: str,
) -> str:
    action_target = _task_action_target(task)
    start_state = dict(task.get("expected_start_state") or {})
    end_state = dict(task.get("expected_end_state") or {})
    start_held_part = str(start_state.get("held_part") or "").strip()
    end_held_part = str(end_state.get("held_part") or "").strip()
    start_location = _state_location_token(start_state)
    end_location = _state_location_token(end_state)
    start_pose = _state_pose_value(start_state)
    end_pose = _state_pose_value(end_state)
    has_source_anchor = bool(str(action_target.get("source_location") or "").strip() or start_location or start_pose)
    has_target_anchor = bool(
        str(action_target.get("target_location") or "").strip()
        or str(action_target.get("named_pose") or "").strip()
        or end_location
        or end_pose
    )
    acquires_part = bool(
        part_name
        and (
            end_held_part == part_name
            or _state_part_holder_token(end_state)
        )
        and start_held_part != part_name
    )
    releases_part = bool(part_name and start_held_part == part_name and end_held_part != part_name)

    if task_kind == "resource_only":
        if has_target_anchor or _state_resource_state_token(end_state):
            return "resource_transition"
        return ""

    if acquires_part and releases_part:
        return "part_transfer"
    if acquires_part:
        return "part_acquire"
    if releases_part:
        return "part_release"
    if has_source_anchor and has_target_anchor:
        return "part_transfer"
    if has_source_anchor:
        return "part_acquire"
    if has_target_anchor:
        return "part_release"
    if _state_part_state_token(end_state):
        return "part_interaction"
    return ""


def _task_is_projectable(
    task: dict[str, Any],
    *,
    task_kind: str,
) -> bool:
    action_target = _task_action_target(task)
    start_state = dict(task.get("expected_start_state") or {})
    end_state = dict(task.get("expected_end_state") or {})
    if task_kind == "resource_only":
        return bool(
            str(action_target.get("named_pose") or "").strip()
            or _state_pose_value(end_state)
            or _state_pose_ref_token(end_state)
            or _state_location_token(end_state)
            or _state_resource_state_token(end_state)
        )
    return bool(
        str(action_target.get("source_location") or "").strip()
        or str(action_target.get("target_location") or "").strip()
        or str(action_target.get("named_pose") or "").strip()
        or _state_part_state_token(start_state)
            or _state_part_state_token(end_state)
        or _state_part_holder_token(start_state)
        or _state_part_holder_token(end_state)
        or _state_pose_value(start_state)
        or _state_pose_value(end_state)
        or _state_pose_ref_token(start_state)
        or _state_pose_ref_token(end_state)
        or _state_location_token(start_state)
        or _state_location_token(end_state)
        or str(start_state.get("held_part") or "").strip()
        or str(end_state.get("held_part") or "").strip()
    )


def _infer_source_ref(
    *,
    requested_source_location: str,
    part_row: dict[str, Any],
    fallback_start_state: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    requested_source_location = _canonicalize_observed_pose_location_token(
        requested_source_location,
        part_row=part_row,
    )
    fallback_start_state = dict(fallback_start_state or {})
    holder = _part_row_holder_token(part_row)
    current_location = _part_row_location_token(part_row)
    observed_pose = _part_row_pose_value(part_row)
    fallback_location = _state_location_token(fallback_start_state)
    fallback_pose = _state_pose_value(fallback_start_state)

    if requested_source_location == "observed_pose":
        if observed_pose is None and fallback_pose is not None:
            observed_pose = deepcopy(fallback_pose)
        if observed_pose is None:
            return {
                "kind": "observed_pose",
                "location": "observed_pose",
            }
        return {
            "kind": "observed_pose",
            "location": "observed_pose",
            "pose": deepcopy(observed_pose),
        }
    if requested_source_location:
        if holder and requested_source_location in {holder, f"{holder}_gripper"}:
            return {
                "kind": "holder",
                "holder_resource_jid": holder,
                "location": f"{holder}_gripper",
            }
        if current_location and requested_source_location == current_location:
            payload: dict[str, Any] = {
                "kind": "location",
                "location": current_location,
            }
            if observed_pose is not None:
                payload["pose"] = deepcopy(observed_pose)
            return payload
        if fallback_location and requested_source_location == fallback_location:
            payload = {
                "kind": "location",
                "location": fallback_location,
            }
            if fallback_pose is not None:
                payload["pose"] = deepcopy(fallback_pose)
            return payload
        return None

    if holder:
        return {
            "kind": "holder",
            "holder_resource_jid": holder,
            "location": f"{holder}_gripper",
        }
    if current_location:
        payload = {
            "kind": "location",
            "location": current_location,
        }
        if observed_pose is not None:
            payload["pose"] = deepcopy(observed_pose)
        return payload
    if fallback_location:
        payload = {
            "kind": "location",
            "location": fallback_location,
        }
        if fallback_pose is not None:
            payload["pose"] = deepcopy(fallback_pose)
        return payload
    if observed_pose is not None:
        return {
            "kind": "observed_pose",
            "location": "observed_pose",
            "pose": deepcopy(observed_pose),
        }
    if fallback_pose is not None:
        return {
            "kind": "observed_pose",
            "location": "observed_pose",
            "pose": deepcopy(fallback_pose),
        }
    return None


def _build_preconditions_and_effects(
    task: dict[str, Any],
    *,
    resource_jid: str,
    resource_row: dict[str, Any],
    part_name: str,
    part_row: dict[str, Any],
    task_kind: str,
) -> tuple[dict[str, Any], dict[str, Any], str, bool]:
    action_target = _task_action_target(task)
    start_state = dict(task.get("expected_start_state") or {})
    end_state = dict(task.get("expected_end_state") or {})

    expected_resource_effect = {
        key: deepcopy(value)
        for key, value in (
            ("current_state", _state_resource_state_token(end_state) or None),
            (
                "gripper_state",
                end_state.get("gripper_state")
                if "gripper_state" in end_state
                else "closed"
                if "held_part" in end_state and str(end_state.get("held_part") or "").strip()
                else "open"
                if "held_part" in end_state
                else None,
            ),
            ("held_part", end_state.get("held_part")),
            (
                "location",
                _state_location_token(end_state)
                or _state_pose_ref_token(end_state)
                or None,
            ),
        )
        if value not in (None, "", [], {})
    }

    if task_kind == "resource_only":
        return (
            {
                "resource": {
                    key: deepcopy(value)
                    for key, value in (
                        ("current_state", _state_resource_state_token(resource_row) or None),
                        ("held_part", resource_row.get("held_part")),
                        ("gripper_state", resource_row.get("gripper_state")),
                    )
                    if value not in (None, "", [], {})
                }
            },
            {"resource": expected_resource_effect, "part": {}},
            "resource_only",
            False,
        )

    requested_target_location = str(action_target.get("target_location") or "").strip()
    requested_source_location = str(action_target.get("source_location") or "").strip()
    resource_holds_part = str(resource_row.get("held_part") or "").strip() == part_name
    current_part_holder = _part_row_holder_token(part_row)

    expected_part_effect: dict[str, Any] = {"part_name": part_name}
    part_state = _state_effect_part_state_token(end_state)
    if part_state:
        expected_part_effect["state"] = part_state
    part_location = _state_location_token(end_state) or requested_target_location
    if part_location:
        expected_part_effect["location"] = part_location
    part_pose = _state_pose_value(end_state)
    if part_pose is not None:
        expected_part_effect["pose"] = deepcopy(part_pose)

    if "held_part" in end_state:
        end_held_part = str(end_state.get("held_part") or "").strip()
        expected_part_effect["holder"] = resource_jid if end_held_part == part_name else None
    elif requested_target_location or part_location or part_pose is not None:
        expected_part_effect["holder"] = None

    part_affecting = any(
        key in expected_part_effect
        for key in ("state", "location", "pose", "holder")
    )
    requires_acquisition = bool(
        part_affecting
        and part_name
        and task_kind != "continuation_resume"
        and not resource_holds_part
        and current_part_holder != resource_jid
    )
    source_ref = (
        _infer_source_ref(
            requested_source_location=requested_source_location,
            part_row=part_row,
            fallback_start_state=start_state,
        )
        if requires_acquisition
        else None
    )
    if not part_affecting and requested_source_location and source_ref is None:
        source_ref = _infer_source_ref(
            requested_source_location=requested_source_location,
            part_row=part_row,
            fallback_start_state=start_state,
        )

    if requires_acquisition and expected_part_effect.get("holder") in (None, "") and not (
        requested_target_location or part_location or part_pose is not None
    ):
        expected_part_effect["holder"] = resource_jid
        expected_part_effect.setdefault("location", f"{resource_jid}_gripper")

    preconditions = {
        "resource": {
            key: deepcopy(value)
            for key, value in (
                ("current_state", _state_resource_state_token(resource_row) or None),
                ("held_part", resource_row.get("held_part")),
                ("gripper_state", resource_row.get("gripper_state")),
            )
            if value not in (None, "", [], {})
        },
        "part": {
            key: deepcopy(value)
            for key, value in (
                (
                    "current_state",
                    _first_non_empty(part_row, "current_state", "state", "part_state", "part_status")
                    or None,
                ),
                ("location", _part_row_location_token(part_row) or None),
                ("holder", current_part_holder or None),
                ("requires_acquisition", requires_acquisition),
            )
            if value not in (None, "", [], {})
        },
    }
    if source_ref is not None:
        preconditions["source_ref"] = deepcopy(source_ref)

    effect_scope = "resource_and_part" if expected_resource_effect and part_affecting else "part_only"
    return (
        preconditions,
        {"resource": expected_resource_effect, "part": expected_part_effect},
        effect_scope,
        requires_acquisition,
    )


def _grounded_action(
    task: dict[str, Any],
    *,
    resource_jid: str,
    resource_row: dict[str, Any],
    part_name: str,
    part_row: dict[str, Any],
    task_kind: str,
    operation_kind: str,
) -> dict[str, Any]:
    action_target = _task_action_target(task)
    end_state = dict(task.get("expected_end_state") or {})
    start_state = dict(task.get("expected_start_state") or {})

    target: dict[str, Any] = {}
    for field_name, key in (
        ("source_location", "source_location"),
        ("target_location", "target_location"),
        ("named_pose", "named_pose"),
    ):
        token = str(action_target.get(field_name) or "").strip()
        if field_name == "source_location":
            token = _canonicalize_observed_pose_location_token(
                token,
                part_row=part_row,
            )
        if token:
            target[key] = token
    pose_ref = _state_pose_ref_token(end_state) or _state_pose_ref_token(start_state)
    if pose_ref and "named_pose" not in target:
        target["named_pose"] = pose_ref
    explicit_pose = _state_pose_value(end_state) or _state_pose_value(start_state)
    if explicit_pose is not None:
        target["pose"] = deepcopy(explicit_pose)
    (
        preconditions,
        expected_effect,
        effect_scope,
        requires_acquisition,
    ) = _build_preconditions_and_effects(
        task,
        resource_jid=resource_jid,
        resource_row=resource_row,
        part_name=part_name,
        part_row=part_row,
        task_kind=task_kind,
    )

    if task_kind == "continuation_resume":
        operation_kind = "continuation_resume"
    elif task_kind != "resource_only":
        has_target = bool(
            str(target.get("target_location") or "").strip()
            or str(target.get("named_pose") or "").strip()
            or target.get("pose")
        )
        if requires_acquisition and has_target:
            operation_kind = "part_transfer"
        elif requires_acquisition:
            operation_kind = "part_acquire"
        elif has_target and dict(expected_effect.get("part") or {}).get("holder", "__missing__") is None:
            operation_kind = "part_release"
        elif not operation_kind:
            operation_kind = "part_interaction"
    elif not operation_kind:
        operation_kind = "resource_transition"

    return {
        "task_id": str(task.get("outline_id") or "").strip(),
        "resource_jid": resource_jid,
        "task_kind": task_kind,
        "operation_kind": operation_kind,
        "part_name": part_name or None,
        "target": target,
        "preconditions": preconditions,
        "expected_effect": expected_effect,
        "effect_scope": effect_scope,
        "expected_start_state": deepcopy(start_state),
        "expected_end_state": deepcopy(end_state),
        "raw_task": deepcopy(task),
    }


def _binding_finding(
    *,
    task: dict[str, Any],
    constraint_code: str,
    resource_jid: str | None = None,
    part_name: str | None = None,
    reason: str,
    evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "task_id": str(task.get("outline_id") or "").strip(),
        "resource_jid": str(resource_jid or "").strip() or None,
        "part_name": str(part_name or "").strip() or None,
        "pose_source": "task_contract",
        "pose": None,
        "workspace_bounds": None,
        "failed_axes": [constraint_code],
        "constraint_owner": "binding",
        "constraint_family": "binding",
        "constraint_code": constraint_code,
        "reason": reason,
        "evidence": deepcopy(evidence or {}),
    }


def _outline_contract_finding(
    *,
    task: dict[str, Any],
    outline_contract: dict[str, Any] | None,
    resource_jid: str,
    part_name: str,
    resource_row: dict[str, Any],
    part_row: dict[str, Any],
) -> list[dict[str, Any]]:
    contract = dict(outline_contract or {})
    start_state = dict(task.get("expected_start_state") or {})
    end_state = dict(task.get("expected_end_state") or {})
    action_target = _task_action_target(task)

    if bool(contract.get("disallow_unknown_state_fields")):
        allowed_state_fields = {
            str(field_name or "").strip()
            for field_name in list(contract.get("allowed_state_fields") or [])
            if str(field_name or "").strip()
        }
        if allowed_state_fields:
            invalid_fields = [
                f"{state_key}.{field_name}"
                for state_key in ("expected_start_state", "expected_end_state")
                for field_name in dict(task.get(state_key) or {})
                if str(field_name or "").strip()
                and str(field_name or "").strip() not in allowed_state_fields
            ]
            if invalid_fields:
                invalid_fields = sorted(dict.fromkeys(invalid_fields))
                return _binding_finding(
                    task=task,
                    constraint_code="disallowed_outline_state_field",
                    resource_jid=resource_jid or None,
                    part_name=part_name or None,
                    reason=(
                        "outline state objects may use only "
                        f"{', '.join(sorted(allowed_state_fields))} "
                        f"({', '.join(invalid_fields)})"
                    ),
                    evidence={"state_fields": deepcopy(invalid_fields)},
                )

    if bool(contract.get("require_expected_start_match")):
        mismatches: list[dict[str, Any]] = []
        for field_name, actual in (
            ("resource_state", _exact_mapping_value(resource_row, "resource_state")),
            ("held_part", _exact_mapping_value(resource_row, "held_part")),
        ):
            if field_name not in start_state:
                continue
            expected = deepcopy(start_state.get(field_name))
            if actual is _EXACT_STATE_UNAVAILABLE or actual != expected:
                mismatches.append({
                    "field": field_name,
                    "expected": expected,
                    "actual": None if actual is _EXACT_STATE_UNAVAILABLE else deepcopy(actual),
                    "available": actual is not _EXACT_STATE_UNAVAILABLE,
                })

        if part_name:
            for field_name, actual in (
                ("part_state", _exact_mapping_value(part_row, "part_state")),
                ("part_location", _exact_mapping_value(part_row, "part_location")),
            ):
                if field_name not in start_state:
                    continue
                expected = deepcopy(start_state.get(field_name))
                if actual is _EXACT_STATE_UNAVAILABLE or actual != expected:
                    mismatches.append({
                        "field": field_name,
                        "expected": expected,
                        "actual": None if actual is _EXACT_STATE_UNAVAILABLE else deepcopy(actual),
                        "available": actual is not _EXACT_STATE_UNAVAILABLE,
                    })

        if mismatches:
            field_names = ", ".join(
                sorted({str(item.get("field") or "").strip() for item in mismatches if str(item.get("field") or "").strip()})
            )
            return _binding_finding(
                task=task,
                constraint_code="expected_start_state_mismatch",
                resource_jid=resource_jid or None,
                part_name=part_name or None,
                reason=(
                    "expected_start_state does not match the projected current state"
                    + (f" for {field_names}" if field_names else "")
                ),
                evidence={"mismatches": deepcopy(mismatches)},
            )

    if bool(contract.get("require_meaningful_delta")):
        has_delta = False
        for field_name, before in (
            ("resource_state", _exact_mapping_value(resource_row, "resource_state")),
            ("resource_location", _exact_mapping_value(resource_row, "resource_location")),
            ("held_part", _exact_mapping_value(resource_row, "held_part")),
            ("part_state", _exact_mapping_value(part_row, "part_state")),
            ("part_location", _exact_mapping_value(part_row, "part_location")),
        ):
            if field_name not in end_state:
                continue
            if before is _EXACT_STATE_UNAVAILABLE or before != end_state.get(field_name):
                has_delta = True
                break
        if not has_delta:
            return _binding_finding(
                task=task,
                constraint_code="no_state_change",
                resource_jid=resource_jid or None,
                part_name=part_name or None,
                reason="Task does not change the projected symbolic state.",
                evidence={"field": "expected_end_state", "deltas": []},
            )

    if bool(contract.get("require_release_destination_for_release")) and resource_jid and part_name:
        current_holder = str(
            (None if _exact_mapping_value(part_row, "part_holder_resource_jid") is _EXACT_STATE_UNAVAILABLE
             else _exact_mapping_value(part_row, "part_holder_resource_jid"))
            or ""
        ).strip()
        current_held_part = str(resource_row.get("held_part") or "").strip()
        start_held_part = str(start_state.get("held_part") or "").strip()
        release_requested = ("held_part" in end_state and end_state.get("held_part") in (None, "")) and (
            current_held_part == part_name
            or current_holder == resource_jid
            or start_held_part == part_name
        )
        if release_requested:
            has_release_destination = bool(
                _state_location_token(end_state)
                or str(action_target.get("target_location") or "").strip()
                or str(action_target.get("named_pose") or "").strip()
                or _state_pose_ref_token(end_state)
                or _state_pose_value(end_state) is not None
                or action_target.get("pose") is not None
                or action_target.get("slot_pose") is not None
            )
            if not has_release_destination:
                return _binding_finding(
                    task=task,
                    constraint_code="missing_release_destination",
                    resource_jid=resource_jid,
                    part_name=part_name,
                    reason=(
                        f"Task releases '{part_name}' without specifying a concrete grounded "
                        "destination."
                    ),
                    evidence={"field": "expected_end_state", "release_destination": None},
                )

    if bool(contract.get("require_carrier_for_part_relocation")) and resource_jid and part_name:
        changed_part_fields = [
            field_name
            for field_name, before in (
                ("part_location", _exact_mapping_value(part_row, "part_location")),
            )
            if field_name in end_state
            and (before is _EXACT_STATE_UNAVAILABLE or before != end_state.get(field_name))
        ]
        if changed_part_fields:
            resource_controls_part = any(
                str(value or "").strip() == part_name
                for value in (
                    resource_row.get("held_part"),
                    start_state.get("held_part"),
                    end_state.get("held_part"),
                )
            ) or any(
                str(value or "").strip() == resource_jid
                for value in (
                    None
                    if _exact_mapping_value(part_row, "part_holder_resource_jid") is _EXACT_STATE_UNAVAILABLE
                    else _exact_mapping_value(part_row, "part_holder_resource_jid"),
                )
            )
            if not resource_controls_part:
                return _binding_finding(
                    task=task,
                    constraint_code="part_relocation_without_carrier",
                    resource_jid=resource_jid,
                    part_name=part_name,
                    reason=(
                        f"Task changes part '{part_name}' location/holder without the named "
                        f"resource '{resource_jid}' carrying or holding it."
                    ),
                    evidence={"field": "part_motion", "changed_fields": changed_part_fields},
                )

    return None


def _binding_token_findings(
    *,
    task: dict[str, Any],
    task_kind: str,
    resource_jid: str,
    resource_row: dict[str, Any],
    part_name: str,
    part_row: dict[str, Any],
    resources_by_jid: dict[str, dict[str, Any]],
    parts_by_name: dict[str, dict[str, Any]],
    location_validation_mode: str = "strict",
) -> list[dict[str, Any]]:
    raw_action_target = dict(task.get("action_target") or {})
    start_state = dict(task.get("expected_start_state") or {})
    end_state = dict(task.get("expected_end_state") or {})

    requested_named_pose = str(raw_action_target.get("named_pose") or "").strip()
    if requested_named_pose:
        available_named_poses = set(_resource_named_pose_tokens(resource_row))
        if requested_named_pose not in available_named_poses:
            return [
                _binding_finding(
                    task=task,
                    constraint_code="unknown_named_pose",
                    resource_jid=resource_jid,
                    part_name=part_name or None,
                    reason=(
                        f"task '{str(task.get('outline_id') or '').strip()}' references unknown "
                        f"named pose '{requested_named_pose}' for resource '{resource_jid}'"
                    ),
                    evidence={"named_pose": requested_named_pose},
                )
            ]

    location_candidates: list[tuple[str, str]] = []
    for field_name in ("source_location", "target_location"):
        token = str(raw_action_target.get(field_name) or "").strip()
        if token:
            location_candidates.append((field_name, token))
    explicit_source_ref = str(task.get("source_ref") or "").strip()
    if explicit_source_ref:
        location_candidates.append(("source_ref", explicit_source_ref))
    explicit_target_ref = str(task.get("target_ref") or "").strip()
    if explicit_target_ref:
        location_candidates.append(("target_ref", explicit_target_ref))

    known_locations = set(
        _known_location_tokens(
            resources_by_jid=resources_by_jid,
            parts_by_name=parts_by_name,
            part_name=part_name,
        )
    )

    # In relaxed mode, collect goal/origin locations that must still be exact.
    _critical_locations: set[str] = set()
    if location_validation_mode == "relaxed":
        for _pn, _pr in (parts_by_name or {}).items():
            if not isinstance(_pr, dict):
                continue
            for _fl in ("goal_location", "origin_location"):
                _loc = str(_pr.get(_fl) or "").strip()
                if _loc:
                    _critical_locations.add(_loc)

    for field_name, raw_token in location_candidates:
        token = str(raw_token or "").strip()
        if not token:
            continue
        if part_row:
            token = _canonicalize_observed_pose_location_token(token, part_row=part_row)
        if token and token not in known_locations:
            if location_validation_mode == "relaxed" and token not in _critical_locations:
                # Accept as abstract location intent — deferred to primitive
                # generation for concrete grounding.
                continue
            return [
                _binding_finding(
                    task=task,
                    constraint_code="unknown_location_token",
                    resource_jid=resource_jid,
                    part_name=part_name or None,
                    reason=(
                        f"task '{str(task.get('outline_id') or '').strip()}' references unknown "
                        f"location token '{token}' in {field_name}"
                    ),
                    evidence={"location_token": token, "location_field": field_name},
                )
            ]

    return []


def compile_grounded_outline_task(
    task: dict[str, Any],
    *,
    resources_by_jid: dict[str, dict[str, Any]],
    parts_by_name: dict[str, dict[str, Any]],
    outline_contract: dict[str, Any] | None = None,
    location_validation_mode: str = "strict",
) -> dict[str, Any]:
    task_id = str(task.get("outline_id") or "").strip()
    resource_jid = _task_resource_jid(task)
    resource_row = dict(resources_by_jid.get(resource_jid) or {})
    part_binding = _task_part_binding(task, parts_by_name=parts_by_name)
    candidate_part_names = list(part_binding.get("candidate_part_names") or [])
    effective_part_name = str(part_binding.get("effective_part_name") or "").strip()
    task_kind = _infer_task_kind(task, part_binding=part_binding)
    candidate_bindings = [
        {
            "resource_jid": resource_jid or None,
            "part_name": part_name,
        }
        for part_name in candidate_part_names
    ]

    if not resource_jid or resource_jid not in resources_by_jid:
        reason = (
            f"task '{task_id}' does not bind a resource that exists in current bridge state"
        )
        return {
            "status": "resource_unbound",
            "grounded_action": None,
            "candidate_bindings": candidate_bindings,
            "finding": _binding_finding(
                task=task,
                constraint_code="resource_unbound",
                reason=reason,
                evidence={"candidate_resource_jids": sorted(resources_by_jid)},
            ),
        }

    if task_kind != "resource_only" and bool(part_binding.get("is_ambiguous")):
        reason = (
            f"task '{task_id}' could refer to multiple parts: {', '.join(candidate_part_names)}"
        )
        return {
            "status": "part_ambiguous",
            "grounded_action": None,
            "candidate_bindings": candidate_bindings,
            "finding": _binding_finding(
                task=task,
                constraint_code="part_ambiguous",
                resource_jid=resource_jid,
                reason=reason,
                evidence={"candidate_part_names": candidate_part_names},
            ),
        }

    if task_kind != "resource_only" and not effective_part_name:
        reason = f"task '{task_id}' does not bind a manipulable part"
        return {
            "status": "part_unbound",
            "grounded_action": None,
            "candidate_bindings": candidate_bindings,
            "finding": _binding_finding(
                task=task,
                constraint_code="part_unbound",
                resource_jid=resource_jid,
                reason=reason,
            ),
        }

    if (
        task_kind != "resource_only"
        and effective_part_name
        and _is_structured_continuation_resume(
            task,
            resource_row=resource_row,
            part_row=dict(parts_by_name.get(effective_part_name) or {}),
            part_name=effective_part_name,
        )
    ):
        task_kind = "continuation_resume"

    operation_kind = _infer_operation_kind(
        task,
        task_kind=task_kind,
        part_name=effective_part_name,
    )
    outline_contract_finding = _outline_contract_finding(
        task=task,
        outline_contract=outline_contract,
        resource_jid=resource_jid,
        part_name=effective_part_name,
        resource_row=resource_row,
        part_row=dict(parts_by_name.get(effective_part_name) or {}),
    )
    if outline_contract_finding:
        return {
            "status": "outline_contract_violation",
            "grounded_action": None,
            "candidate_bindings": candidate_bindings,
            "finding": deepcopy(outline_contract_finding),
        }
    binding_token_findings = _binding_token_findings(
        task=task,
        task_kind=task_kind,
        resource_jid=resource_jid,
        resource_row=resource_row,
        part_name=effective_part_name,
        part_row=dict(parts_by_name.get(effective_part_name) or {}),
        resources_by_jid=resources_by_jid,
        parts_by_name=parts_by_name,
        location_validation_mode=location_validation_mode,
    )
    if binding_token_findings:
        return {
            "status": "binding_token_unresolved",
            "grounded_action": None,
            "candidate_bindings": candidate_bindings,
            "finding": deepcopy(binding_token_findings[0]),
        }
    if not operation_kind or not _task_is_projectable(task, task_kind=task_kind):
        reason = (
            f"task '{task_id}' does not specify enough target or effect information "
            "to project one concrete recovery action"
        )
        return {
            "status": "task_not_projectable",
            "grounded_action": None,
            "candidate_bindings": candidate_bindings,
            "finding": _binding_finding(
                task=task,
                constraint_code="task_not_projectable",
                resource_jid=resource_jid,
                part_name=effective_part_name or None,
                reason=reason,
                evidence={
                    "task_kind": task_kind,
                    "operation_kind": operation_kind or None,
                    "action_target": deepcopy(_task_action_target(task)),
                },
            ),
        }

    grounded_action = _grounded_action(
        task,
        resource_jid=resource_jid,
        resource_row=resource_row,
        part_name=effective_part_name,
        part_row=dict(parts_by_name.get(effective_part_name) or {}),
        task_kind=task_kind,
        operation_kind=operation_kind,
    )
    preconditions = dict(grounded_action.get("preconditions") or {})
    requires_acquisition = bool(
        dict(preconditions.get("part") or {}).get("requires_acquisition")
    )
    if task_kind != "resource_only" and requires_acquisition and not dict(
        preconditions.get("source_ref") or {}
    ):
        reason = (
            f"task '{task_id}' changes part '{effective_part_name}' but does not ground a "
            "concrete current source reference for acquiring it first"
        )
        return {
            "status": "task_not_projectable",
            "grounded_action": None,
            "candidate_bindings": candidate_bindings,
            "finding": _binding_finding(
                task=task,
                constraint_code="task_not_projectable",
                resource_jid=resource_jid,
                part_name=effective_part_name or None,
                reason=reason,
                evidence={
                    "task_kind": task_kind,
                    "operation_kind": grounded_action.get("operation_kind"),
                    "action_target": deepcopy(_task_action_target(task)),
                    "preconditions": deepcopy(preconditions),
                },
            ),
        }
    return {
        "status": "grounded",
        "grounded_action": grounded_action,
        "candidate_bindings": [
            {
                "resource_jid": resource_jid,
                "part_name": effective_part_name or None,
            }
        ],
        "finding": None,
    }


__all__ = ["compile_grounded_outline_task"]
