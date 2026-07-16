"""Shared outline-state helpers used by multi-turn recovery."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


def _task_findings_block_projected_state(findings: list[dict[str, Any]]) -> bool:
    return bool(findings)


def _first_non_empty_state_value(state: dict[str, Any], *field_names: str) -> Any:
    for field_name in field_names:
        if field_name not in state:
            continue
        value = state.get(field_name)
        if value in (None, "", [], {}):
            continue
        return deepcopy(value)
    return None


def _is_part_lifecycle_state(token: str) -> bool:
    return bool(str(token or "").strip())


def _is_resource_state_token(token: str) -> bool:
    return bool(str(token or "").strip())


def _outline_state_resource_state_token(state: dict[str, Any]) -> str:
    return str(
        _first_non_empty_state_value(state, "current_state", "state", "resource_state") or ""
    ).strip()


def _outline_state_part_state_token(
    state: dict[str, Any],
    *,
    candidate_part_name: str,
) -> str:
    state_part_name = str(state.get("part_name") or "").strip()
    if state_part_name and candidate_part_name and state_part_name != candidate_part_name:
        return ""
    token = str(
        _first_non_empty_state_value(
            state,
            "part_state",
            "part_status",
            "current_state",
            "state",
        )
        or ""
    ).strip()
    if _is_part_lifecycle_state(token):
        return token
    return ""


def _outline_state_location_token(state: dict[str, Any]) -> str:
    return str(
        _first_non_empty_state_value(
            state,
            "part_location",
            "resource_location",
            "location",
            "current_location",
            "named_pose",
        )
        or ""
    ).strip()


def _outline_state_pose_value(state: dict[str, Any]) -> dict[str, Any] | None:
    pose = state.get("position") or state.get("pose") or state.get("current_pose")
    if not isinstance(pose, dict) or "x" not in pose:
        return None
    return dict(pose)


def _outline_resource_named_pose_token(
    *,
    state: dict[str, Any],
    action_target: dict[str, Any],
) -> str:
    return str(
        _first_non_empty_state_value(
            state,
            "named_pose",
            "resource_location",
            "location",
            "current_location",
        )
        or action_target.get("named_pose")
        or ""
    ).strip()


def _outline_state_part_holder_token(state: dict[str, Any]) -> str:
    return str(
        _first_non_empty_state_value(
            state,
            "part_holder_resource_jid",
            "current_holder_resource_jid",
        )
        or ""
    ).strip()


def _outline_task_part_references(task: dict[str, Any]) -> list[str]:
    part_names: list[str] = []
    for candidate in (task.get("part_name"),):
        token = str(candidate or "").strip()
        if token and token not in part_names:
            part_names.append(token)
    for state_key in ("expected_start_state", "expected_end_state"):
        state = task.get(state_key)
        if not isinstance(state, dict):
            continue
        for field_name in ("part_name", "held_part"):
            token = str(state.get(field_name) or "").strip()
            if token and token not in part_names:
                part_names.append(token)
    return part_names


def _dedupe_outline_part_candidates(part_names: list[str]) -> list[str]:
    deduped: list[str] = []
    for part_name in part_names:
        token = str(part_name or "").strip()
        if token and token not in deduped:
            deduped.append(token)
    return deduped


def _outline_part_names_matching_location(
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
    for part_name, raw_row in parts_by_name.items():
        row = dict(raw_row or {})
        current_location = str(row.get("current_location") or row.get("location") or "").strip()
        goal_location = str(row.get("goal_location") or "").strip()
        if include_current and current_location and current_location == normalized_location:
            matches.append(part_name)
            continue
        if include_goal and goal_location and goal_location == normalized_location:
            matches.append(part_name)
    return _dedupe_outline_part_candidates(matches)


def _outline_task_action_target(task: dict[str, Any]) -> dict[str, Any]:
    action_target = task.get("action_target")
    if not isinstance(action_target, dict):
        return {}
    return dict(action_target)


def _outline_task_part_binding(
    task: dict[str, Any],
    *,
    parts_by_name: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    explicit_candidates = _dedupe_outline_part_candidates(
        [
            str(task.get("part_name") or "").strip(),
            str(dict(task.get("expected_start_state") or {}).get("part_name") or "").strip(),
            str(dict(task.get("expected_end_state") or {}).get("part_name") or "").strip(),
        ]
    )
    candidates = list(explicit_candidates)
    if not candidates:
        candidates.extend(_outline_task_part_references(task))
    if not explicit_candidates:
        action_target = _outline_task_action_target(task)
        source_matches = _outline_part_names_matching_location(
            location_token=str(action_target.get("source_location") or "").strip(),
            parts_by_name=parts_by_name,
            include_current=True,
            include_goal=False,
        )
        if len(source_matches) == 1:
            candidates.extend(source_matches)
        target_matches = _outline_part_names_matching_location(
            location_token=str(action_target.get("target_location") or "").strip(),
            parts_by_name=parts_by_name,
            include_current=False,
            include_goal=True,
        )
        if len(target_matches) == 1:
            candidates.extend(target_matches)
        for state_key in ("expected_start_state", "expected_end_state"):
            state = dict(task.get(state_key) or {})
            location_matches = _outline_part_names_matching_location(
                location_token=_outline_state_location_token(state),
                parts_by_name=parts_by_name,
                include_current=True,
                include_goal=True,
            )
            if len(location_matches) == 1:
                candidates.extend(location_matches)
    deduped_candidates = _dedupe_outline_part_candidates(candidates)
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


def _outline_task_predecessors(task: dict[str, Any]) -> list[str]:
    predecessor_ids: list[str] = []
    for item in task.get("predecessors") or []:
        token = str(item).strip()
        if token and token not in predecessor_ids:
            predecessor_ids.append(token)
    return predecessor_ids


def _resource_row_state_token(row: dict[str, Any]) -> str:
    return str(
        _first_non_empty_state_value(row, "current_state", "state", "resource_state") or ""
    ).strip()


def _resource_row_held_part_token(row: dict[str, Any]) -> str:
    return str(_first_non_empty_state_value(row, "held_part") or "").strip()


def _part_row_state_token(row: dict[str, Any]) -> str:
    return str(
        _first_non_empty_state_value(row, "part_state", "current_state", "state") or ""
    ).strip()


def _part_row_location_token(row: dict[str, Any]) -> str:
    return str(
        _first_non_empty_state_value(
            row,
            "part_location",
            "current_location",
            "location",
            "current_pose_ref",
        )
        or ""
    ).strip()


def _outline_task_effective_part_name(task: dict[str, Any]) -> str:
    explicit_part_name = str(task.get("part_name") or "").strip()
    if explicit_part_name:
        return explicit_part_name
    for token in _outline_task_part_references(task):
        normalized = str(token or "").strip()
        if normalized:
            return normalized
    return ""


_FACT_VALUE_UNAVAILABLE = object()


def _exact_fact_value(mapping: dict[str, Any], field_name: str) -> Any:
    if field_name not in mapping:
        return _FACT_VALUE_UNAVAILABLE
    return deepcopy(mapping.get(field_name))


def _outline_task_requirement_facts(task: dict[str, Any]) -> list[tuple[tuple[str, str, str], Any]]:
    requirements: list[tuple[tuple[str, str, str], Any]] = []
    resource_jid = str(task.get("resource_jid") or "").strip()
    part_name = _outline_task_effective_part_name(task)
    start_state = dict(task.get("expected_start_state") or {})

    if resource_jid:
        if "resource_state" in start_state:
            requirements.append(
                (
                    ("resource", resource_jid, "resource_state"),
                    deepcopy(start_state.get("resource_state")),
                )
            )
        if "held_part" in start_state:
            requirements.append(
                (("resource", resource_jid, "held_part"), deepcopy(start_state.get("held_part")))
            )
        if "resource_location" in start_state:
            requirements.append(
                (
                    ("resource", resource_jid, "resource_location"),
                    deepcopy(start_state.get("resource_location")),
                )
            )

    if part_name:
        if "part_state" in start_state:
            requirements.append(
                (("part", part_name, "part_state"), deepcopy(start_state.get("part_state")))
            )
        if "part_location" in start_state:
            requirements.append(
                (("part", part_name, "part_location"), deepcopy(start_state.get("part_location")))
            )
        if "part_holder_resource_jid" in start_state:
            requirements.append(
                (
                    ("part", part_name, "part_holder_resource_jid"),
                    deepcopy(start_state.get("part_holder_resource_jid")),
                )
            )
    return requirements


def _outline_task_produced_facts(task: dict[str, Any]) -> dict[tuple[str, str, str], Any]:
    produced: dict[tuple[str, str, str], Any] = {}
    resource_jid = str(task.get("resource_jid") or "").strip()
    part_name = _outline_task_effective_part_name(task)
    end_state = dict(task.get("expected_end_state") or {})

    if resource_jid:
        if "resource_state" in end_state:
            produced[("resource", resource_jid, "resource_state")] = deepcopy(
                end_state.get("resource_state")
            )
        if "held_part" in end_state:
            produced[("resource", resource_jid, "held_part")] = deepcopy(end_state.get("held_part"))
        if "resource_location" in end_state:
            produced[("resource", resource_jid, "resource_location")] = deepcopy(
                end_state.get("resource_location")
            )

    if part_name:
        if "part_state" in end_state:
            produced[("part", part_name, "part_state")] = deepcopy(end_state.get("part_state"))
        if "part_location" in end_state:
            produced[("part", part_name, "part_location")] = deepcopy(
                end_state.get("part_location")
            )
        if "part_holder_resource_jid" in end_state:
            produced[("part", part_name, "part_holder_resource_jid")] = deepcopy(
                end_state.get("part_holder_resource_jid")
            )
    return produced


def _state_fact_value(
    fact_key: tuple[str, str, str],
    *,
    resources_by_jid: dict[str, dict[str, Any]],
    parts_by_name: dict[str, dict[str, Any]],
) -> Any:
    scope, entity_id, field_name = fact_key
    if scope == "resource":
        row = dict(resources_by_jid.get(entity_id) or {})
        return _exact_fact_value(row, field_name)
    row = dict(parts_by_name.get(entity_id) or {})
    return _exact_fact_value(row, field_name)


def _task_transitive_predecessors(
    predecessor_ids: list[str],
    predecessors_by_outline_id: dict[str, list[str]],
) -> set[str]:
    visited: set[str] = set()
    stack = list(predecessor_ids)
    while stack:
        outline_id = str(stack.pop() or "").strip()
        if not outline_id or outline_id in visited:
            continue
        visited.add(outline_id)
        stack.extend(predecessors_by_outline_id.get(outline_id) or [])
    return visited


def _is_failed_resource_recovery_task(
    task: dict[str, Any],
    *,
    initial_resources_by_jid: dict[str, dict[str, Any]],
) -> bool:
    resource_jid = str(task.get("resource_jid") or "").strip()
    if not resource_jid:
        return False
    initial_resource_state = _resource_row_state_token(
        dict(initial_resources_by_jid.get(resource_jid) or {})
    )
    start_state = dict(task.get("expected_start_state") or {})
    end_state = dict(task.get("expected_end_state") or {})
    start_token = _outline_state_resource_state_token(start_state) or initial_resource_state
    end_token = _outline_state_resource_state_token(end_state) or start_token
    return initial_resource_state == "failed" and start_token == "failed" and end_token == "idle"


def infer_outline_predecessors(
    outline_tasks: list[dict[str, Any]],
    *,
    resources_by_jid: dict[str, dict[str, Any]],
    parts_by_name: dict[str, dict[str, Any]],
    llm_input: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    inferred_tasks: list[dict[str, Any]] = []
    task_types_by_id = _build_outline_task_type_lookup(
        outline_tasks,
        resources_by_jid=resources_by_jid,
        parts_by_name=parts_by_name,
        llm_input=llm_input,
    )
    projected_resources = deepcopy(resources_by_jid or {})
    projected_parts = deepcopy(parts_by_name or {})
    predecessors_by_outline_id: dict[str, list[str]] = {}
    previous_outline_id_by_resource: dict[str, str] = {}
    prior_failed_resource_recoveries: list[str] = []

    for index, raw_task in enumerate(outline_tasks):
        if not isinstance(raw_task, dict):
            continue
        task = deepcopy(raw_task)
        outline_id = str(task.get("outline_id") or "").strip() or f"task_{index}"
        resource_jid = str(task.get("resource_jid") or "").strip()
        task_predecessors: list[str] = []
        initial_unsatisfied = False

        requirements = _outline_task_requirement_facts(task)
        for fact_key, expected_value in requirements:
            if (
                _state_fact_value(
                    fact_key,
                    resources_by_jid=resources_by_jid,
                    parts_by_name=parts_by_name,
                )
                == expected_value
            ):
                continue
            initial_unsatisfied = True
            for prior_task in reversed(inferred_tasks):
                prior_outline_id = str(prior_task.get("outline_id") or "").strip()
                if not prior_outline_id:
                    continue
                prior_produced_facts = _outline_task_produced_facts(prior_task)
                if (
                    fact_key in prior_produced_facts
                    and prior_produced_facts.get(fact_key) == expected_value
                ):
                    if prior_outline_id not in task_predecessors:
                        task_predecessors.append(prior_outline_id)
                    break

        predecessor_closure = _task_transitive_predecessors(
            task_predecessors,
            predecessors_by_outline_id,
        )
        if initial_unsatisfied and not _is_failed_resource_recovery_task(
            task,
            initial_resources_by_jid=resources_by_jid,
        ):
            for recovery_outline_id in prior_failed_resource_recoveries:
                if (
                    recovery_outline_id not in task_predecessors
                    and recovery_outline_id not in predecessor_closure
                ):
                    task_predecessors.append(recovery_outline_id)
                    predecessor_closure = _task_transitive_predecessors(
                        task_predecessors,
                        predecessors_by_outline_id,
                    )

        previous_outline_id = previous_outline_id_by_resource.get(resource_jid) or ""
        if (
            previous_outline_id
            and previous_outline_id not in task_predecessors
            and previous_outline_id not in predecessor_closure
        ):
            task_predecessors.append(previous_outline_id)

        task["predecessors"] = list(task_predecessors)
        inferred_tasks.append(task)
        predecessors_by_outline_id[outline_id] = list(task_predecessors)

        if _is_failed_resource_recovery_task(
            task,
            initial_resources_by_jid=resources_by_jid,
        ):
            prior_failed_resource_recoveries.append(outline_id)

        task_type = str(task_types_by_id.get(outline_id) or "resource_only")
        _apply_outline_task_effects(
            task,
            resources_by_jid=projected_resources,
            parts_by_name=projected_parts,
            task_type=task_type,
        )
        if resource_jid:
            previous_outline_id_by_resource[resource_jid] = outline_id

    return inferred_tasks


def _outline_task_requirement_id(
    task: dict[str, Any],
    *,
    parts_by_name: dict[str, dict[str, Any]],
) -> str:
    action_target = _outline_task_action_target(task)
    requirement_id = str(action_target.get("requirement_id") or "").strip()
    if requirement_id:
        return requirement_id
    for part_name in _outline_task_part_references(task):
        requirement_id = str(
            dict(parts_by_name.get(part_name) or {}).get("goal_requirement_id") or ""
        ).strip()
        if requirement_id:
            return requirement_id
    return ""


def _modeled_gap_pending_nominal_tasks(llm_input: dict[str, Any] | None) -> list[dict[str, Any]]:
    modeled_gap = dict((llm_input or {}).get("modeled_continuation_gap") or {})
    return [
        dict(row)
        for row in (modeled_gap.get("pending_nominal_tasks") or [])
        if isinstance(row, dict)
    ]


def _outline_task_target_locations(task: dict[str, Any]) -> list[str]:
    action_target = _outline_task_action_target(task)
    end_state = dict(task.get("expected_end_state") or {})
    tokens = [
        str(action_target.get("target_location") or "").strip(),
        str(end_state.get("part_location") or "").strip(),
        str(end_state.get("resource_location") or "").strip(),
        str(end_state.get("location") or "").strip(),
        str(end_state.get("current_location") or "").strip(),
    ]
    deduped: list[str] = []
    for token in tokens:
        if token and token not in deduped:
            deduped.append(token)
    return deduped


def _outline_task_matches_pending_nominal_suffix(
    task: dict[str, Any],
    *,
    parts_by_name: dict[str, dict[str, Any]],
    llm_input: dict[str, Any] | None,
) -> bool:
    resource_jid = str(task.get("resource_jid") or "").strip()
    if not resource_jid:
        return False
    part_binding = _outline_task_part_binding(task, parts_by_name=parts_by_name)
    candidate_parts = list(part_binding.get("candidate_part_names") or [])
    effective_part_name = str(part_binding.get("effective_part_name") or "").strip()
    if effective_part_name and effective_part_name not in candidate_parts:
        candidate_parts.append(effective_part_name)
    if not candidate_parts:
        return False
    requirement_id = _outline_task_requirement_id(task, parts_by_name=parts_by_name)
    target_locations = set(_outline_task_target_locations(task))
    end_state = dict(task.get("expected_end_state") or {})
    end_state_token = (
        str(
            end_state.get("part_state")
            or end_state.get("current_state")
            or end_state.get("state")
            or ""
        )
        .strip()
        .lower()
    )
    for pending_task in _modeled_gap_pending_nominal_tasks(llm_input):
        pending_part = str(pending_task.get("part") or "").strip()
        pending_resource = str(pending_task.get("resource") or "").strip()
        if not pending_part:
            continue
        if pending_resource and pending_resource != resource_jid:
            continue
        if pending_part not in candidate_parts:
            continue
        part_row = dict(parts_by_name.get(pending_part) or {})
        goal_requirement_id = str(part_row.get("goal_requirement_id") or "").strip()
        if requirement_id and goal_requirement_id and requirement_id != goal_requirement_id:
            continue
        goal_location = str(part_row.get("goal_location") or "").strip()
        if goal_location:
            if goal_location in target_locations:
                return True
            if _outline_state_location_token(end_state) == goal_location:
                return True
            continue
        if end_state_token in {"placed", "assembled"}:
            return True
    return False


def _state_delta(before: Any, after: Any) -> dict[str, Any] | None:
    if before == after:
        return None
    return {"before": deepcopy(before), "after": deepcopy(after)}


def _infer_outline_macro_signature(
    task: dict[str, Any],
    *,
    resources_by_jid: dict[str, dict[str, Any]] | None,
    parts_by_name: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    resource_map = resources_by_jid or {}
    resource_jid = str(task.get("resource_jid") or "").strip()
    resource_row = dict(resource_map.get(resource_jid) or {})
    start_state = dict(task.get("expected_start_state") or {})
    end_state = dict(task.get("expected_end_state") or {})
    action_target = _outline_task_action_target(task)
    referenced_parts = _outline_task_part_references(task)
    task_part_name = str(task.get("part_name") or "").strip()
    explicit_start_part_name = str(start_state.get("part_name") or "").strip()
    explicit_end_part_name = str(end_state.get("part_name") or "").strip()

    before_resource_state = (
        _outline_state_resource_state_token(start_state)
        or str(resource_row.get("current_state") or "").strip()
    )
    after_resource_state = _outline_state_resource_state_token(end_state) or before_resource_state
    if "held_part" in start_state:
        before_held_part = str(start_state.get("held_part") or "").strip()
    else:
        before_held_part = str(resource_row.get("held_part") or "").strip()
    if "held_part" in end_state:
        after_held_part = str(end_state.get("held_part") or "").strip()
    else:
        after_held_part = before_held_part
    before_named_pose = (
        _outline_resource_named_pose_token(
            state=start_state,
            action_target={},
        )
        or str(resource_row.get("current_location") or "").strip()
    )
    after_named_pose = (
        _outline_resource_named_pose_token(
            state=end_state,
            action_target=action_target,
        )
        or before_named_pose
    )
    before_resource_pose = (
        _outline_state_pose_value(start_state)
        or dict(resource_row.get("current_pose") or {})
        or None
    )
    after_resource_pose = _outline_state_pose_value(end_state) or before_resource_pose

    resource_delta: dict[str, Any] = {}
    for field_name, before_value, after_value in (
        ("current_state", before_resource_state or None, after_resource_state or None),
        ("held_part", before_held_part or None, after_held_part or None),
        ("current_location", before_named_pose or None, after_named_pose or None),
        ("current_pose", before_resource_pose, after_resource_pose),
    ):
        delta = _state_delta(before_value, after_value)
        if delta is not None:
            resource_delta[field_name] = delta

    candidate_parts: list[str] = []
    for token in referenced_parts + [before_held_part, after_held_part]:
        normalized = str(token or "").strip()
        if normalized and normalized not in candidate_parts:
            candidate_parts.append(normalized)

    inferable_primary_part = task_part_name
    if not inferable_primary_part:
        explicit_contract_candidates = [
            token
            for token in (
                explicit_end_part_name,
                explicit_start_part_name,
                str(end_state.get("held_part") or "").strip() if "held_part" in end_state else "",
                str(start_state.get("held_part") or "").strip()
                if "held_part" in start_state
                else "",
            )
            if token
        ]
        deduped_contract_candidates: list[str] = []
        for token in explicit_contract_candidates:
            if token not in deduped_contract_candidates:
                deduped_contract_candidates.append(token)
        if "held_part" in end_state and after_held_part and after_held_part != before_held_part:
            inferable_primary_part = after_held_part
        elif explicit_end_part_name:
            inferable_primary_part = explicit_end_part_name
        elif explicit_start_part_name:
            inferable_primary_part = explicit_start_part_name
        elif len(deduped_contract_candidates) == 1:
            inferable_primary_part = deduped_contract_candidates[0]

    part_deltas: dict[str, dict[str, Any]] = {}
    direct_observed_pickup_parts: list[str] = []
    for part_name in candidate_parts:
        part_row = dict(parts_by_name.get(part_name) or {})
        before_holder = str(part_row.get("current_holder_resource_jid") or "").strip()
        if not before_holder and before_held_part == part_name:
            before_holder = resource_jid
        after_holder = before_holder
        if after_held_part == part_name:
            after_holder = resource_jid
        elif before_held_part == part_name and after_held_part != part_name:
            after_holder = ""

        before_part_state = str(part_row.get("current_state") or "").strip()
        explicit_start_part_state = _outline_state_part_state_token(
            start_state,
            candidate_part_name=part_name,
        )
        explicit_end_part_state = _outline_state_part_state_token(
            end_state,
            candidate_part_name=part_name,
        )
        if explicit_start_part_state:
            before_part_state = explicit_start_part_state
        after_part_state = explicit_end_part_state or before_part_state

        before_part_location = str(
            part_row.get("current_location") or part_row.get("location") or ""
        ).strip()
        after_part_location = before_part_location
        explicit_end_location = _outline_state_location_token(end_state)
        if explicit_end_location and (
            task_part_name == part_name
            or inferable_primary_part == part_name
            or str(end_state.get("part_name") or "").strip() == part_name
        ):
            after_part_location = explicit_end_location
        elif (
            str(action_target.get("target_location") or "").strip()
            and (task_part_name == part_name or inferable_primary_part == part_name)
            and after_holder != resource_jid
        ):
            after_part_location = str(action_target.get("target_location") or "").strip()
        elif before_holder == resource_jid and after_holder != resource_jid:
            after_part_location = explicit_end_location

        before_part_pose = dict(part_row.get("observed_pose") or {}) or None
        after_part_pose = _outline_state_pose_value(end_state) or before_part_pose

        delta_row: dict[str, Any] = {}
        for field_name, before_value, after_value in (
            ("state", before_part_state or None, after_part_state or None),
            ("holder", before_holder or None, after_holder or None),
            ("location", before_part_location or None, after_part_location or None),
            ("pose", before_part_pose, after_part_pose),
        ):
            delta = _state_delta(before_value, after_value)
            if delta is not None:
                delta_row[field_name] = delta
        if delta_row:
            part_deltas[part_name] = delta_row
            if (
                isinstance(part_row.get("observed_pose"), dict)
                and "x" in dict(part_row.get("observed_pose") or {})
                and before_holder != resource_jid
            ):
                direct_observed_pickup_parts.append(part_name)

    changes_part_world = bool(part_deltas)
    part_intent_without_delta = False
    primary_part_for_intent = task_part_name or inferable_primary_part
    if not changes_part_world and primary_part_for_intent:
        state_scoped_part_anchor = False
        for state in (start_state, end_state):
            if str(state.get("part_name") or "").strip() != primary_part_for_intent:
                continue
            if (
                _outline_state_pose_value(state)
                or _outline_state_location_token(state)
                or str(state.get("held_part") or "").strip() == primary_part_for_intent
            ):
                state_scoped_part_anchor = True
                break
        part_intent_without_delta = bool(
            _outline_state_part_state_token(
                start_state,
                candidate_part_name=primary_part_for_intent,
            )
            or _outline_state_part_state_token(
                end_state,
                candidate_part_name=primary_part_for_intent,
            )
            or state_scoped_part_anchor
            or str(action_target.get("source_location") or "").strip()
            or str(action_target.get("target_location") or "").strip()
        )
        if part_intent_without_delta:
            primary_part_row = dict(parts_by_name.get(primary_part_for_intent) or {})
            if (
                isinstance(primary_part_row.get("observed_pose"), dict)
                and "x" in dict(primary_part_row.get("observed_pose") or {})
                and str(primary_part_row.get("current_holder_resource_jid") or "").strip()
                != resource_jid
                and primary_part_for_intent not in direct_observed_pickup_parts
            ):
                direct_observed_pickup_parts.append(primary_part_for_intent)
    illegal_holder_swap = bool(
        before_held_part and after_held_part and before_held_part != after_held_part
    )
    if changes_part_world or part_intent_without_delta:
        task_kind = "part_handling"
    else:
        task_kind = "resource_only"

    return {
        "resource_before": {
            "current_state": before_resource_state or None,
            "held_part": before_held_part or None,
            "current_location": before_named_pose or None,
            "current_pose": deepcopy(before_resource_pose),
        },
        "resource_after": {
            "current_state": after_resource_state or None,
            "held_part": after_held_part or None,
            "current_location": after_named_pose or None,
            "current_pose": deepcopy(after_resource_pose),
        },
        "resource_delta": resource_delta,
        "part_deltas": deepcopy(part_deltas),
        "changes_part_world": changes_part_world,
        "changes_part_holder": any("holder" in row for row in part_deltas.values()),
        "changes_part_location": any("location" in row for row in part_deltas.values()),
        "changes_part_pose": any("pose" in row for row in part_deltas.values()),
        "changes_held_part": "held_part" in resource_delta,
        "illegal_holder_swap": illegal_holder_swap,
        "direct_observed_pickup_parts": direct_observed_pickup_parts,
        "inferable_primary_part": inferable_primary_part,
        "task_kind": task_kind,
    }


def _classify_outline_task(
    task: dict[str, Any],
    *,
    resources_by_jid: dict[str, dict[str, Any]] | None = None,
    parts_by_name: dict[str, dict[str, Any]],
    llm_input: dict[str, Any] | None = None,
) -> str:
    signature = _infer_outline_macro_signature(
        task,
        resources_by_jid=resources_by_jid,
        parts_by_name=parts_by_name,
    )
    task_kind = str(signature.get("task_kind") or "resource_only")
    if task_kind == "part_handling" and _outline_task_matches_pending_nominal_suffix(
        task,
        parts_by_name=parts_by_name,
        llm_input=llm_input,
    ):
        return "continuation_resume"
    return task_kind


def _build_outline_task_type_lookup(
    outline_tasks: list[dict[str, Any]],
    *,
    resources_by_jid: dict[str, dict[str, Any]] | None = None,
    parts_by_name: dict[str, dict[str, Any]],
    llm_input: dict[str, Any] | None = None,
) -> dict[str, str]:
    task_types: dict[str, str] = {}
    for index, raw_task in enumerate(outline_tasks):
        if not isinstance(raw_task, dict):
            continue
        outline_id = str(raw_task.get("outline_id") or "").strip() or f"task_{index}"
        task_types[outline_id] = _classify_outline_task(
            raw_task,
            resources_by_jid=resources_by_jid,
            parts_by_name=parts_by_name,
            llm_input=llm_input,
        )
    return task_types


def _apply_outline_task_effects(
    task: dict[str, Any],
    *,
    resources_by_jid: dict[str, dict[str, Any]],
    parts_by_name: dict[str, dict[str, Any]],
    task_type: str,
    state_field_scopes: dict[str, str] | None = None,
) -> None:
    resource_jid = str(task.get("resource_jid") or "").strip()
    if not resource_jid or resource_jid not in resources_by_jid:
        return

    resource_row = dict(resources_by_jid.get(resource_jid) or {})
    end_state = dict(task.get("expected_end_state") or {})
    declared_scopes = {
        str(field_name): str(scope or "resource")
        for field_name, scope in dict(state_field_scopes or {}).items()
        if str(field_name)
    }
    task_part_name = str(
        _outline_task_part_binding(task, parts_by_name=parts_by_name).get("effective_part_name")
        or task.get("part_name")
        or ""
    ).strip()

    for field_name, value in end_state.items():
        if declared_scopes.get(str(field_name)) == "resource":
            resource_row[str(field_name)] = deepcopy(value)
    if "resource_state" in end_state:
        candidate_state = deepcopy(end_state.get("resource_state"))
        resource_row["resource_state"] = candidate_state
        resource_row["current_state"] = candidate_state
    if "held_part" in end_state:
        resource_row["held_part"] = deepcopy(end_state.get("held_part"))
    if "resource_location" in end_state:
        resource_row["resource_location"] = deepcopy(end_state.get("resource_location"))
        resource_row["current_location"] = deepcopy(end_state.get("resource_location"))
    end_pose = end_state.get("position") or end_state.get("pose")
    if isinstance(end_pose, dict):
        if "x" in end_pose:
            resource_row["current_pose"] = deepcopy(end_pose)
    resources_by_jid[resource_jid] = deepcopy(resource_row)

    if task_type != "part_handling" or not task_part_name:
        return

    part_row = dict(parts_by_name.get(task_part_name) or {})
    if not part_row:
        return
    for field_name, value in end_state.items():
        if declared_scopes.get(str(field_name)) == "part":
            part_row[str(field_name)] = deepcopy(value)
    if "part_state" in end_state:
        candidate_state = deepcopy(end_state.get("part_state"))
        part_row["part_state"] = candidate_state
        part_row["current_state"] = candidate_state
    if "part_location" in end_state:
        candidate_location = deepcopy(end_state.get("part_location"))
        part_row["part_location"] = candidate_location
        part_row["current_location"] = candidate_location
    if "held_part" in end_state:
        held_part = str(end_state.get("held_part") or "").strip()
        if held_part == task_part_name:
            part_row["part_holder_resource_jid"] = resource_jid
            part_row["current_holder_resource_jid"] = resource_jid
        elif not held_part:
            part_row["part_holder_resource_jid"] = None
            part_row["current_holder_resource_jid"] = None
    elif "part_holder_resource_jid" in end_state:
        candidate_holder = deepcopy(end_state.get("part_holder_resource_jid"))
        part_row["part_holder_resource_jid"] = candidate_holder
        part_row["current_holder_resource_jid"] = candidate_holder
    part_end_pose = end_state.get("position") or end_state.get("pose")
    if isinstance(part_end_pose, dict) and "x" in part_end_pose:
        part_row["observed_pose"] = deepcopy(part_end_pose)
    parts_by_name[task_part_name] = deepcopy(part_row)
