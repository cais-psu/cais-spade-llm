"""CCA-owned generic bridge AP projection and outline-time safety validation."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from cais_spade_llm.agents.central_controller.online_safety_monitor import OnlineSafetyMonitor


def _normalize_token(value: Any) -> str:
    return str(value or "").strip().lower()


def _dedupe_tokens(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw_value in values:
        token = str(raw_value or "").strip()
        if not token or token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out


def _modeled_gap_pending_tasks_by_id(llm_input: dict[str, Any]) -> dict[str, dict[str, Any]]:
    modeled_gap = dict(llm_input.get("modeled_continuation_gap") or {})
    tasks_by_id: dict[str, dict[str, Any]] = {}
    for raw_task in modeled_gap.get("pending_nominal_tasks") or []:
        if not isinstance(raw_task, dict):
            continue
        task_id = str(raw_task.get("id") or "").strip()
        if task_id:
            tasks_by_id[task_id] = deepcopy(raw_task)
    return tasks_by_id


def _modeled_gap_unmet_conditions_by_id(llm_input: dict[str, Any]) -> dict[str, dict[str, Any]]:
    modeled_gap = dict(llm_input.get("modeled_continuation_gap") or {})
    conditions_by_id: dict[str, dict[str, Any]] = {}
    for raw_condition in modeled_gap.get("unmet_continuation_conditions") or []:
        if not isinstance(raw_condition, dict):
            continue
        condition_id = str(raw_condition.get("condition_id") or "").strip()
        if condition_id:
            conditions_by_id[condition_id] = deepcopy(raw_condition)
    return conditions_by_id


def _task_dependency_ids(task: dict[str, Any]) -> list[str]:
    return [str(item).strip() for item in (task.get("predecessors") or []) if str(item).strip()]


def _task_closes_condition_ids(task: dict[str, Any]) -> list[str]:
    return _dedupe_tokens(
        [
            str(item).strip()
            for item in (task.get("closes_condition_ids") or [])
            if str(item).strip()
        ]
    )


def _task_enables_task_ids(task: dict[str, Any]) -> list[str]:
    return _dedupe_tokens(
        [str(item).strip() for item in (task.get("enables_task_ids") or []) if str(item).strip()]
    )


def _task_part_names(task: dict[str, Any]) -> list[str]:
    tokens: list[str] = []
    for candidate in (
        task.get("part_name"),
        dict(task.get("expected_start_state") or {}).get("part_name"),
        dict(task.get("expected_end_state") or {}).get("part_name"),
        dict(task.get("expected_start_state") or {}).get("held_part"),
        dict(task.get("expected_end_state") or {}).get("held_part"),
    ):
        token = str(candidate or "").strip()
        if token and token not in tokens:
            tokens.append(token)
    return tokens


def _task_type_for_cca(
    task: dict[str, Any],
    *,
    task_types_by_id: dict[str, str],
) -> str:
    task_id = str(task.get("outline_id") or "").strip()
    if task_id and str(task_types_by_id.get(task_id) or "").strip():
        return str(task_types_by_id.get(task_id) or "").strip()
    return str(task.get("task_kind") or "").strip()


def _fault_event_fallback_parts(llm_input: dict[str, Any]) -> list[str]:
    fault_event = dict(llm_input.get("fault_event") or {})
    return [
        str(item).strip()
        for item in (fault_event.get("affected_part_names") or [])
        if str(item).strip()
    ]


def _extract_blocker_part_names(
    *,
    blocking_reason: str,
    parts_by_name: dict[str, dict[str, Any]],
    fallback_parts: list[str],
) -> list[str]:
    preferred_fallback = [
        str(part_name).strip() for part_name in fallback_parts if str(part_name).strip()
    ]
    if preferred_fallback:
        return preferred_fallback
    blocker_text = str(blocking_reason or "").strip().lower()
    blocker_parts = [
        part_name for part_name in parts_by_name if part_name and part_name.lower() in blocker_text
    ]
    return blocker_parts or preferred_fallback


def _continuation_condition_cleared(
    condition: dict[str, Any],
    *,
    resources_by_jid: dict[str, dict[str, Any]],
    parts_by_name: dict[str, dict[str, Any]],
    fallback_parts: list[str],
) -> bool:
    kind = str(condition.get("kind") or "").strip()
    if kind == "focused_resource_terminal_state":
        resource_jid = str(condition.get("entity") or "").strip()
        expected_state = str(condition.get("expected") or "").strip()
        current_state = str(
            dict(resources_by_jid.get(resource_jid) or {}).get("current_state") or ""
        ).strip()
        return bool(expected_state and current_state == expected_state)
    if kind != "safety_blocked_suffix_task":
        return False
    blocker_parts = _extract_blocker_part_names(
        blocking_reason=str(condition.get("blocking_reason") or "").strip(),
        parts_by_name=parts_by_name,
        fallback_parts=fallback_parts,
    )
    if not blocker_parts:
        return False
    for blocker_part in blocker_parts:
        part_row = dict(parts_by_name.get(blocker_part) or {})
        goal_location = str(part_row.get("goal_location") or "").strip()
        current_state = str(part_row.get("current_state") or "").strip().lower()
        current_location = _part_location(part_row)
        if current_state in {"placed", "assembled"}:
            continue
        if goal_location and current_location == goal_location:
            continue
        return False
    return True


def _continuation_prerequisite_task_ids(
    task: dict[str, Any],
    *,
    outline_tasks: list[dict[str, Any]],
    task_types_by_id: dict[str, str],
    llm_input: dict[str, Any],
    parts_by_name: dict[str, dict[str, Any]],
) -> list[str]:
    if (
        _task_type_for_cca(
            task,
            task_types_by_id=task_types_by_id,
        )
        != "continuation_resume"
    ):
        return []
    referenced_parts = _task_part_names(task)
    pending_nominal_task_ids: list[str] = []
    for part_name in referenced_parts:
        pending_nominal_task_ids.extend(
            str(item).strip()
            for item in (
                dict(parts_by_name.get(part_name) or {}).get("pending_nominal_task_ids") or []
            )
            if str(item).strip()
        )
    unmet_conditions = [
        dict(row)
        for row in (_modeled_gap_unmet_conditions_by_id(llm_input).values())
        if isinstance(row, dict)
    ]
    fallback_parts = _fault_event_fallback_parts(llm_input)
    prerequisite_ids: list[str] = []
    for condition in unmet_conditions:
        kind = str(condition.get("kind") or "").strip()
        if kind == "focused_resource_terminal_state":
            resource_jid = str(condition.get("entity") or "").strip()
            expected_state = str(condition.get("expected") or "").strip()
            for index, raw_task in enumerate(outline_tasks):
                if not isinstance(raw_task, dict):
                    continue
                outline_id = str(raw_task.get("outline_id") or "").strip() or f"task_{index}"
                if task_types_by_id.get(outline_id) != "resource_only":
                    continue
                if str(raw_task.get("resource_jid") or "").strip() != resource_jid:
                    continue
                end_state = dict(raw_task.get("expected_end_state") or {})
                end_token = str(
                    end_state.get("current_state") or end_state.get("state") or ""
                ).strip()
                if end_token == expected_state:
                    prerequisite_ids.append(outline_id)
        elif kind == "safety_blocked_suffix_task":
            source_task_id = str(condition.get("source_task_id") or "").strip()
            if (
                pending_nominal_task_ids
                and source_task_id
                and source_task_id not in pending_nominal_task_ids
            ):
                continue
            blocker_parts = _extract_blocker_part_names(
                blocking_reason=str(condition.get("blocking_reason") or "").strip(),
                parts_by_name=parts_by_name,
                fallback_parts=fallback_parts,
            )
            for blocker_part in blocker_parts:
                goal_location = str(
                    dict(parts_by_name.get(blocker_part) or {}).get("goal_location") or ""
                ).strip()
                for index, raw_task in enumerate(outline_tasks):
                    if not isinstance(raw_task, dict):
                        continue
                    outline_id = str(raw_task.get("outline_id") or "").strip() or f"task_{index}"
                    if task_types_by_id.get(outline_id) == "continuation_resume":
                        continue
                    if str(raw_task.get("part_name") or "").strip() != blocker_part:
                        continue
                    end_state = dict(raw_task.get("expected_end_state") or {})
                    end_current_state = (
                        str(end_state.get("current_state") or end_state.get("state") or "")
                        .strip()
                        .lower()
                    )
                    end_location = _part_location(end_state)
                    if end_current_state in {"placed", "assembled"} or (
                        goal_location and end_location == goal_location
                    ):
                        prerequisite_ids.append(outline_id)
    return _dedupe_tokens(prerequisite_ids)


def _dependency_reaches(
    *,
    task_id: str,
    target_id: str,
    dependency_map: dict[str, list[str]],
) -> bool:
    if task_id == target_id:
        return True
    seen: set[str] = set()
    frontier = list(dependency_map.get(task_id) or [])
    while frontier:
        current = frontier.pop(0)
        if current in seen:
            continue
        seen.add(current)
        if current == target_id:
            return True
        frontier.extend(dependency_map.get(current) or [])
    return False


def _sequence_finding(
    *,
    task: dict[str, Any],
    constraint_code: str,
    reason: str,
    part_name: str | None = None,
    resource_jid: str | None = None,
    evidence: dict[str, Any] | None = None,
    claimed_condition_ids: list[str] | None = None,
    claimed_task_ids: list[str] | None = None,
) -> dict[str, Any]:
    finding = {
        "task_id": str(task.get("outline_id") or "").strip(),
        "resource_jid": str(resource_jid or task.get("resource_jid") or "").strip() or None,
        "part_name": str(part_name or task.get("part_name") or "").strip() or None,
        "pose_source": "task_contract",
        "pose": None,
        "workspace_bounds": None,
        "failed_axes": [constraint_code],
        "constraint_owner": "cca",
        "constraint_family": "sequence",
        "constraint_code": constraint_code,
        "reason": reason,
        "evidence": deepcopy(evidence or {}),
    }
    if claimed_condition_ids:
        finding["claimed_condition_ids"] = _dedupe_tokens(claimed_condition_ids)
    if claimed_task_ids:
        finding["claimed_task_ids"] = _dedupe_tokens(claimed_task_ids)
    return finding


def _projected_enabled_task_ids(
    *,
    pending_tasks_by_id: dict[str, dict[str, Any]],
    projected_cleared_condition_ids: list[str],
) -> list[str]:
    enabled_task_ids: list[str] = []
    cleared_set = {
        str(condition_id).strip()
        for condition_id in projected_cleared_condition_ids
        if str(condition_id).strip()
    }
    for task_id, pending_task in pending_tasks_by_id.items():
        blocked_by_condition_ids = [
            str(item).strip()
            for item in (dict(pending_task).get("blocked_by_condition_ids") or [])
            if str(item).strip()
        ]
        if not blocked_by_condition_ids:
            continue
        if all(condition_id in cleared_set for condition_id in blocked_by_condition_ids):
            enabled_task_ids.append(task_id)
    return _dedupe_tokens(enabled_task_ids)


def _effective_task_part_name(task: dict[str, Any], signature: dict[str, Any]) -> str:
    return str(task.get("part_name") or signature.get("inferable_primary_part") or "").strip()


def _bridge_loaded_rules(llm_input: dict[str, Any]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for raw_rule in llm_input.get("loaded_safety_rules") or []:
        if not isinstance(raw_rule, dict):
            continue
        ap_scope = str(raw_rule.get("ap_scope") or "").strip().lower()
        if ap_scope not in {"bridge", "both", "nominal"}:
            continue
        dfa_dot = str(raw_rule.get("dfa_dot") or "").strip()
        bridge_aps = _bridge_rule_aps(raw_rule)
        if not dfa_dot or not bridge_aps:
            continue
        rule = deepcopy(raw_rule)
        rule.setdefault("rule_id", str(rule.get("id") or rule.get("rule_id") or "").strip())
        rule["bridge_aps"] = bridge_aps
        rule["dfa_dot"] = dfa_dot
        selected.append(rule)
    return selected


def _parse_bridge_selector_from_ap_full(ap_full: str) -> dict[str, Any] | None:
    full = str(ap_full or "").strip()
    if not full:
        return None
    segments = [segment.strip() for segment in full.split("/") if segment.strip()]
    if len(segments) < 5:
        return None
    ap_kind = segments[0]
    part = segments[2] if len(segments) > 2 else "any"
    resource = segments[3] if len(segments) > 3 else "any"
    verb = segments[4] if len(segments) > 4 else ""
    params: dict[str, str] = {}
    for segment in segments[5:]:
        if "=" not in segment:
            continue
        key, value = segment.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key and value:
            params[key] = value
    destination = str(params.get("destination") or "").strip()
    if ap_kind == "ap_event":
        if destination:
            return {
                "mode": (
                    "move_part_to_destination"
                    if part not in {"", "any"}
                    else "resource_move_to_destination"
                ),
                "part": part or "any",
                "resource": resource or "any",
                "destination": destination,
            }
        return None
    if ap_kind != "ap_state":
        return None
    if verb in {"assembled", "placed"} and part not in {"", "any"}:
        return {
            "mode": "part_goal_satisfied",
            "part": part,
            "resource": resource or "any",
            "destination": destination,
            "states": [verb],
        }
    if verb in {"positioned", "placed"}:
        return {
            "mode": "resource_in_destination",
            "part": part or "any",
            "resource": resource or "any",
            "destination": destination,
        }
    return None


def _bridge_rule_aps(raw_rule: dict[str, Any]) -> list[dict[str, Any]]:
    explicit_bridge_aps = [
        deepcopy(ap)
        for ap in (raw_rule.get("bridge_aps") or [])
        if isinstance(ap, dict)
        and str(ap.get("label") or "").strip()
        and str(ap.get("full") or "").strip()
    ]
    if explicit_bridge_aps:
        return explicit_bridge_aps
    derived_aps: list[dict[str, Any]] = []
    for raw_ap in raw_rule.get("aps") or []:
        if not isinstance(raw_ap, dict):
            continue
        label = str(raw_ap.get("label") or "").strip()
        full = str(raw_ap.get("full") or "").strip()
        selector = dict(raw_ap.get("selector") or {})
        if not selector:
            selector = dict(_parse_bridge_selector_from_ap_full(full) or {})
        if not label or not full or not selector:
            continue
        derived_aps.append(
            {
                "label": label,
                "full": full,
                "selector": selector,
            }
        )
    return derived_aps


def _bridge_monitor_rules(rules: list[dict[str, Any]]) -> list[dict[str, Any]]:
    monitor_rules: list[dict[str, Any]] = []
    for raw_rule in rules:
        rule_id = str(raw_rule.get("id") or raw_rule.get("rule_id") or "").strip()
        bridge_aps = [
            {
                "label": str(ap.get("label") or "").strip(),
                "full": str(ap.get("full") or "").strip(),
            }
            for ap in (raw_rule.get("bridge_aps") or [])
            if isinstance(ap, dict)
            and str(ap.get("label") or "").strip()
            and str(ap.get("full") or "").strip()
        ]
        if not rule_id or not bridge_aps:
            continue
        monitor_rules.append(
            {
                "id": rule_id,
                "rule_id": rule_id,
                "constraint_type": deepcopy(raw_rule.get("constraint_type")),
                "aps": bridge_aps,
            }
        )
    return monitor_rules


def _bridge_dfa_dots(rules: list[dict[str, Any]]) -> dict[str, str]:
    dfa_dots: dict[str, str] = {}
    for raw_rule in rules:
        rule_id = str(raw_rule.get("id") or raw_rule.get("rule_id") or "").strip()
        dfa_dot = str(raw_rule.get("dfa_dot") or "").strip()
        if rule_id and dfa_dot:
            dfa_dots[rule_id] = dfa_dot
    return dfa_dots


def _bridge_rule_lookup(rules: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for raw_rule in rules:
        rule_id = str(raw_rule.get("id") or raw_rule.get("rule_id") or "").strip()
        if rule_id:
            lookup[rule_id] = deepcopy(raw_rule)
    return lookup


def _bridge_rule_labels(rules: list[dict[str, Any]]) -> set[str]:
    labels: set[str] = set()
    for raw_rule in rules:
        for raw_ap in raw_rule.get("bridge_aps") or []:
            if not isinstance(raw_ap, dict):
                continue
            label = str(raw_ap.get("label") or "").strip()
            if label:
                labels.add(label)
    return labels


def _part_location(row: dict[str, Any]) -> str:
    return str(row.get("current_location") or row.get("location") or "").strip()


def _resource_location(row: dict[str, Any]) -> str:
    return str(row.get("current_location") or row.get("location") or "").strip()


def _selector_destination_tokens(
    task: dict[str, Any],
    *,
    effective_part_name: str,
    projected_resources: dict[str, dict[str, Any]],
    projected_parts: dict[str, dict[str, Any]],
) -> set[str]:
    action_target = dict(task.get("action_target") or {})
    end_state = dict(task.get("expected_end_state") or {})
    tokens = {
        str(action_target.get("target_location") or "").strip(),
        str(end_state.get("location") or "").strip(),
        str(end_state.get("current_location") or "").strip(),
        str(end_state.get("part_location") or "").strip(),
    }
    if effective_part_name:
        projected_part = dict(projected_parts.get(effective_part_name) or {})
        tokens.add(_part_location(projected_part))
    resource_jid = str(task.get("resource_jid") or "").strip()
    if resource_jid:
        projected_resource = dict(projected_resources.get(resource_jid) or {})
        tokens.add(_resource_location(projected_resource))
    return {token for token in tokens if token}


def _selector_matches_event(
    selector: dict[str, Any],
    *,
    task: dict[str, Any],
    signature: dict[str, Any],
    projected_resources: dict[str, dict[str, Any]],
    projected_parts: dict[str, dict[str, Any]],
) -> bool:
    mode = str(selector.get("mode") or "").strip()
    resource_jid = _normalize_token(task.get("resource_jid"))
    effective_part_name = _normalize_token(_effective_task_part_name(task, signature))
    destination = str(selector.get("destination") or "").strip()
    resource_selector = _normalize_token(selector.get("resource") or "any")
    part_selector = _normalize_token(selector.get("part") or "any")
    destination_tokens = _selector_destination_tokens(
        task,
        effective_part_name=_effective_task_part_name(task, signature),
        projected_resources=projected_resources,
        projected_parts=projected_parts,
    )
    if destination and destination not in destination_tokens:
        return False
    if resource_selector not in {"", "any"} and resource_selector != resource_jid.split("@", 1)[0]:
        return False
    if part_selector not in {"", "any"} and part_selector != effective_part_name.lower():
        return False
    if mode == "move_part_to_destination":
        return bool(
            signature.get("changes_part_world") and effective_part_name and destination_tokens
        )
    if mode == "resource_move_to_destination":
        return bool(destination_tokens)
    return False


def _selector_matches_state(
    selector: dict[str, Any],
    *,
    resources_by_jid: dict[str, dict[str, Any]],
    parts_by_name: dict[str, dict[str, Any]],
) -> bool:
    mode = str(selector.get("mode") or "").strip()
    destination = str(selector.get("destination") or "").strip()
    resource_selector = _normalize_token(selector.get("resource") or "any")
    part_selector = _normalize_token(selector.get("part") or "any")
    allowed_states = {
        _normalize_token(item) for item in (selector.get("states") or []) if _normalize_token(item)
    }
    if mode == "part_goal_satisfied":
        for part_name, raw_row in (parts_by_name or {}).items():
            if part_selector not in {"", "any"} and part_selector != _normalize_token(part_name):
                continue
            row = dict(raw_row or {})
            current_state = _normalize_token(row.get("current_state") or row.get("state"))
            current_location = _part_location(row)
            if destination and current_location != destination:
                continue
            if allowed_states and current_state not in allowed_states:
                continue
            return True
        return False
    if mode == "part_at_destination":
        for part_name, raw_row in (parts_by_name or {}).items():
            if part_selector not in {"", "any"} and part_selector != _normalize_token(part_name):
                continue
            row = dict(raw_row or {})
            if destination and _part_location(row) != destination:
                continue
            return True
        return False
    if mode == "resource_in_destination":
        for resource_jid, raw_row in (resources_by_jid or {}).items():
            resource_token = _normalize_token(resource_jid).split("@", 1)[0]
            if resource_selector not in {"", "any"} and resource_selector != resource_token:
                continue
            row = dict(raw_row or {})
            if destination and _resource_location(row) != destination:
                continue
            return True
        return False
    if mode == "resource_state":
        expected_state = _normalize_token(selector.get("state"))
        for resource_jid, raw_row in (resources_by_jid or {}).items():
            resource_token = _normalize_token(resource_jid).split("@", 1)[0]
            if resource_selector not in {"", "any"} and resource_selector != resource_token:
                continue
            row = dict(raw_row or {})
            current_state = _normalize_token(row.get("current_state") or row.get("state"))
            if expected_state and current_state != expected_state:
                continue
            return True
        return False
    return False


def project_outline_macro_bridge_aps(
    *,
    task: dict[str, Any],
    signature: dict[str, Any],
    pre_resources: dict[str, dict[str, Any]],
    pre_parts: dict[str, dict[str, Any]],
    projected_resources: dict[str, dict[str, Any]],
    projected_parts: dict[str, dict[str, Any]],
    llm_input: dict[str, Any],
) -> dict[str, Any]:
    rules = _bridge_loaded_rules(llm_input)
    candidate_aps: list[str] = []
    predicted_state_aps: list[str] = []
    current_state_aps: list[str] = []
    for raw_rule in rules:
        for raw_ap in raw_rule.get("bridge_aps") or []:
            if not isinstance(raw_ap, dict):
                continue
            label = str(raw_ap.get("label") or "").strip()
            selector = dict(raw_ap.get("selector") or {})
            full = str(raw_ap.get("full") or "").strip()
            if not label or not selector or not full:
                continue
            if full.startswith("ap_event/") and _selector_matches_event(
                selector,
                task=task,
                signature=signature,
                projected_resources=projected_resources,
                projected_parts=projected_parts,
            ):
                candidate_aps.append(label)
            elif full.startswith("ap_state/"):
                if _selector_matches_state(
                    selector,
                    resources_by_jid=pre_resources,
                    parts_by_name=pre_parts,
                ):
                    current_state_aps.append(label)
                if _selector_matches_state(
                    selector,
                    resources_by_jid=projected_resources,
                    parts_by_name=projected_parts,
                ):
                    predicted_state_aps.append(label)
    return {
        "candidate_aps": _dedupe_tokens(candidate_aps),
        "predicted_state_aps": _dedupe_tokens(predicted_state_aps),
        "current_state_aps": _dedupe_tokens(current_state_aps),
        "ap_context": {
            "effective_part_name": _effective_task_part_name(task, signature),
            "task_kind": str(signature.get("task_kind") or "").strip(),
        },
    }


def _bridge_running_aps(llm_input: dict[str, Any], rules: list[dict[str, Any]]) -> list[str]:
    bridge_ctx = dict(llm_input.get("bridge_safety_context") or {})
    allowed_labels = _bridge_rule_labels(rules)
    running_aps = [
        str(label).strip()
        for label in (bridge_ctx.get("running_aps") or [])
        if str(label).strip() in allowed_labels
    ]
    return _dedupe_tokens(running_aps)


def _build_bridge_safety_monitor(
    *,
    rules: list[dict[str, Any]],
    state_aps: list[str],
    llm_input: dict[str, Any],
) -> OnlineSafetyMonitor:
    monitor = OnlineSafetyMonitor(
        _bridge_dfa_dots(rules),
        _bridge_monitor_rules(rules),
    )
    if state_aps:
        monitor.resource_state_aps["bridge_scope"] = set(state_aps)
    running_aps = _bridge_running_aps(llm_input, rules)
    if running_aps:
        monitor.running_aps.update(running_aps)
    return monitor


def _safety_violation_reason(rule: dict[str, Any]) -> str:
    return str(
        rule.get("summary")
        or rule.get("generated_interpretation")
        or rule.get("text")
        or rule.get("raw_text")
        or ""
    ).strip()


def _claimed_safety_condition_ids_for_rule(
    *,
    claimed_condition_ids: list[str],
    conditions_by_id: dict[str, dict[str, Any]],
    rule_id: str,
) -> list[str]:
    matched: list[str] = []
    for condition_id in claimed_condition_ids:
        condition = dict(conditions_by_id.get(condition_id) or {})
        if str(condition.get("kind") or "").strip() != "safety_blocked_suffix_task":
            continue
        blocking_rule_id = str(condition.get("blocking_rule_id") or "").strip()
        if blocking_rule_id and blocking_rule_id == rule_id:
            matched.append(condition_id)
    return matched


def _blocked_suffix_proxy_task(
    *,
    condition: dict[str, Any],
    blocked_task: dict[str, Any],
    rule: dict[str, Any],
) -> dict[str, Any]:
    destination = str((rule.get("context") or {}).get("destination") or "").strip()
    return {
        "outline_id": str(blocked_task.get("id") or condition.get("entity") or "").strip(),
        "resource_jid": str(
            blocked_task.get("resource") or condition.get("blocked_resource_jid") or ""
        ).strip(),
        "part_name": str(
            blocked_task.get("part") or condition.get("blocked_part_name") or ""
        ).strip(),
        "action_target": {
            "target_location": destination,
        },
        "expected_end_state": {
            "location": destination,
        },
    }


def _safe_next_task_ids_after_projection(
    *,
    rules: list[dict[str, Any]],
    conditions_by_id: dict[str, dict[str, Any]],
    pending_tasks_by_id: dict[str, dict[str, Any]],
    claimed_condition_ids: list[str],
    projected_resources: dict[str, dict[str, Any]],
    projected_parts: dict[str, dict[str, Any]],
    llm_input: dict[str, Any],
) -> tuple[list[str], list[str]]:
    rule_lookup = _bridge_rule_lookup(rules)
    safe_next_task_ids: list[str] = []
    cleared_condition_ids: list[str] = []
    for condition_id in claimed_condition_ids:
        condition = dict(conditions_by_id.get(condition_id) or {})
        if str(condition.get("kind") or "").strip() != "safety_blocked_suffix_task":
            continue
        blocked_task_id = str(
            condition.get("source_task_id") or condition.get("entity") or ""
        ).strip()
        blocking_rule_id = str(condition.get("blocking_rule_id") or "").strip()
        blocked_task = dict(pending_tasks_by_id.get(blocked_task_id) or {})
        rule = dict(rule_lookup.get(blocking_rule_id) or {})
        if not blocked_task_id or not blocked_task or not rule:
            continue
        proxy_task = _blocked_suffix_proxy_task(
            condition=condition,
            blocked_task=blocked_task,
            rule=rule,
        )
        proxy_signature = {
            "inferable_primary_part": str(proxy_task.get("part_name") or "").strip(),
            "task_kind": "part_handling",
            "changes_part_world": True,
        }
        projection = project_outline_macro_bridge_aps(
            task=proxy_task,
            signature=proxy_signature,
            pre_resources=projected_resources,
            pre_parts=projected_parts,
            projected_resources=projected_resources,
            projected_parts=projected_parts,
            llm_input={
                "loaded_safety_rules": [rule],
                "bridge_safety_context": llm_input.get("bridge_safety_context") or {},
            },
        )
        monitor = _build_bridge_safety_monitor(
            rules=[rule],
            state_aps=list(projection.get("current_state_aps") or []),
            llm_input=llm_input,
        )
        allowed, _ = monitor.online_safety_validation(
            list(projection.get("candidate_aps") or []),
            predicted_state_aps=list(projection.get("predicted_state_aps") or []),
        )
        if allowed:
            safe_next_task_ids.append(blocked_task_id)
            cleared_condition_ids.append(condition_id)
    return _dedupe_tokens(safe_next_task_ids), _dedupe_tokens(cleared_condition_ids)


def validate_outline_macro_bridge_safety(
    *,
    task: dict[str, Any],
    signature: dict[str, Any],
    pre_resources: dict[str, dict[str, Any]],
    pre_parts: dict[str, dict[str, Any]],
    projected_resources: dict[str, dict[str, Any]],
    projected_parts: dict[str, dict[str, Any]],
    llm_input: dict[str, Any],
) -> dict[str, Any]:
    rules = _bridge_loaded_rules(llm_input)
    projection = project_outline_macro_bridge_aps(
        task=task,
        signature=signature,
        pre_resources=pre_resources,
        pre_parts=pre_parts,
        projected_resources=projected_resources,
        projected_parts=projected_parts,
        llm_input=llm_input,
    )
    claimed_condition_ids = [
        str(item).strip() for item in (task.get("closes_condition_ids") or []) if str(item).strip()
    ]
    conditions_by_id = _modeled_gap_unmet_conditions_by_id(llm_input)
    pending_tasks_by_id = _modeled_gap_pending_tasks_by_id(llm_input)

    if not rules:
        return {
            "is_safe": True,
            "safety_ctx": {
                "rule_ids": [],
                "running_aps": [],
                "candidate_aps": list(projection.get("candidate_aps") or []),
                "predicted_state_aps": list(projection.get("predicted_state_aps") or []),
                "safe_next_task_ids": [],
                "status": "",
                "reason": "",
            },
            "findings": [],
            "handled_condition_ids": [],
            "cleared_condition_ids": [],
        }

    monitor = _build_bridge_safety_monitor(
        rules=rules,
        state_aps=list(projection.get("current_state_aps") or []),
        llm_input=llm_input,
    )
    allowed, info = monitor.online_safety_validation(
        list(projection.get("candidate_aps") or []),
        predicted_state_aps=list(projection.get("predicted_state_aps") or []),
    )
    rule_lookup = _bridge_rule_lookup(rules)
    violated_rule_id = str(info.get("violated_rule") or "").strip()
    violated_rule = dict(rule_lookup.get(violated_rule_id) or {})
    related_condition_ids = _claimed_safety_condition_ids_for_rule(
        claimed_condition_ids=claimed_condition_ids,
        conditions_by_id=conditions_by_id,
        rule_id=violated_rule_id,
    )
    safe_next_task_ids: list[str] = []
    cleared_condition_ids: list[str] = []
    if allowed:
        safe_next_task_ids, cleared_condition_ids = _safe_next_task_ids_after_projection(
            rules=rules,
            conditions_by_id=conditions_by_id,
            pending_tasks_by_id=pending_tasks_by_id,
            claimed_condition_ids=claimed_condition_ids,
            projected_resources=projected_resources,
            projected_parts=projected_parts,
            llm_input=llm_input,
        )

    safety_ctx = {
        "rule_ids": [violated_rule_id] if violated_rule_id else [],
        "running_aps": _dedupe_tokens(
            list(info.get("running_snapshot") or [])
            + list(projection.get("current_state_aps") or [])
        ),
        "candidate_aps": list(info.get("candidate_aps") or projection.get("candidate_aps") or []),
        "predicted_state_aps": list(
            info.get("predicted_state_aps") or projection.get("predicted_state_aps") or []
        ),
        "safe_next_task_ids": safe_next_task_ids,
        "status": "violated" if not allowed else "safe",
        "reason": _safety_violation_reason(violated_rule) if violated_rule else "",
        "safety_rules": [deepcopy(violated_rule)] if violated_rule else [],
    }

    findings: list[dict[str, Any]] = []
    if not allowed and violated_rule_id:
        findings.append(
            {
                "task_id": str(task.get("outline_id") or "").strip(),
                "resource_jid": str(task.get("resource_jid") or "").strip() or None,
                "part_name": _effective_task_part_name(task, signature) or None,
                "pose_source": "bridge_safety_rule",
                "pose": None,
                "workspace_bounds": None,
                "failed_axes": ["safety_rule_violation"],
                "constraint_owner": "cca",
                "constraint_family": "safety",
                "constraint_code": "safety_rule_violation",
                "claimed_condition_ids": deepcopy(related_condition_ids),
                "rule_id": violated_rule_id,
                "candidate_aps": deepcopy(safety_ctx.get("candidate_aps") or []),
                "predicted_state_aps": deepcopy(safety_ctx.get("predicted_state_aps") or []),
                "running_aps": deepcopy(safety_ctx.get("running_aps") or []),
                "violated_from": deepcopy(info.get("violated_from")),
                "violated_to": deepcopy(info.get("violated_to")),
                "status": deepcopy(safety_ctx.get("status")),
                "reason": deepcopy(safety_ctx.get("reason")),
                "evidence": {
                    "rule_id": violated_rule_id,
                    "candidate_aps": deepcopy(safety_ctx.get("candidate_aps") or []),
                    "predicted_state_aps": deepcopy(safety_ctx.get("predicted_state_aps") or []),
                    "running_aps": deepcopy(safety_ctx.get("running_aps") or []),
                    "violated_from": deepcopy(info.get("violated_from")),
                    "violated_to": deepcopy(info.get("violated_to")),
                },
                "safe_next_task_ids": deepcopy(safe_next_task_ids),
            }
        )

    handled_condition_ids = _dedupe_tokens(related_condition_ids + cleared_condition_ids)
    return {
        "is_safe": bool(allowed),
        "safety_ctx": safety_ctx,
        "findings": findings,
        "handled_condition_ids": handled_condition_ids,
        "cleared_condition_ids": cleared_condition_ids,
    }


def validate_outline_macro_cca_constraints(
    *,
    task: dict[str, Any],
    grounded_action: dict[str, Any] | None = None,
    event_instance: dict[str, Any] | Any | None = None,
    projection: dict[str, Any] | Any | None = None,
    signature: dict[str, Any],
    pre_resources: dict[str, dict[str, Any]],
    pre_parts: dict[str, dict[str, Any]],
    projected_resources: dict[str, dict[str, Any]],
    projected_parts: dict[str, dict[str, Any]],
    llm_input: dict[str, Any],
    outline_tasks: list[dict[str, Any]],
    task_types_by_id: dict[str, str],
    task_index_by_id: dict[str, int],
    dependency_map: dict[str, list[str]],
    previously_cleared_condition_ids: list[str] | None = None,
) -> dict[str, Any]:
    del grounded_action, event_instance, projection
    findings: list[dict[str, Any]] = []
    condition_lookup = _modeled_gap_unmet_conditions_by_id(llm_input)
    pending_tasks_by_id = _modeled_gap_pending_tasks_by_id(llm_input)
    fallback_parts = _fault_event_fallback_parts(llm_input)
    task_type = _task_type_for_cca(
        task,
        task_types_by_id=task_types_by_id,
    )
    invalid_dependency_ids = [
        dependency_id
        for dependency_id in _task_dependency_ids(task)
        if dependency_id not in task_index_by_id
    ]
    if invalid_dependency_ids:
        finding = _sequence_finding(
            task=task,
            constraint_code="invalid_dependency_reference",
            reason=(
                "predecessors may reference only outline_id values from outline rows in "
                f"this same response ({', '.join(invalid_dependency_ids)})"
            ),
            part_name=task.get("part_name"),
            evidence={"dependency_ids": deepcopy(invalid_dependency_ids)},
        )
        finding["dependency_ids"] = deepcopy(invalid_dependency_ids)
        findings.append(finding)

    if task_type == "continuation_resume":
        prerequisite_ids = _continuation_prerequisite_task_ids(
            task,
            outline_tasks=outline_tasks,
            task_types_by_id=task_types_by_id,
            llm_input=llm_input,
            parts_by_name=pre_parts,
        )
        task_id = str(task.get("outline_id") or "").strip()
        missing_dependency_ids = [
            prerequisite_id
            for prerequisite_id in prerequisite_ids
            if not _dependency_reaches(
                task_id=task_id,
                target_id=prerequisite_id,
                dependency_map=dependency_map,
            )
        ]
        if missing_dependency_ids:
            findings.append(
                _sequence_finding(
                    task=task,
                    constraint_code="dependency_unsatisfied",
                    reason=(
                        f"continuation task is missing prerequisite dependencies on "
                        f"{', '.join(missing_dependency_ids)}"
                    ),
                    part_name=task.get("part_name"),
                    evidence={"required_dependency_ids": deepcopy(missing_dependency_ids)},
                )
            )
        continuation_index = int(task_index_by_id.get(task_id) or 0)
        late_prerequisite_ids = [
            prerequisite_id
            for prerequisite_id in prerequisite_ids
            if int(task_index_by_id.get(prerequisite_id) or -1) >= continuation_index
        ]
        if late_prerequisite_ids:
            findings.append(
                _sequence_finding(
                    task=task,
                    constraint_code="order_violation",
                    reason=(
                        f"continuation task appears before prerequisite tasks "
                        f"{', '.join(late_prerequisite_ids)}"
                    ),
                    part_name=task.get("part_name"),
                    evidence={"required_dependency_ids": deepcopy(late_prerequisite_ids)},
                )
            )
        blocking_condition_ids = [
            condition_id
            for condition_id, condition in condition_lookup.items()
            if not _continuation_condition_cleared(
                condition,
                resources_by_jid=pre_resources,
                parts_by_name=pre_parts,
                fallback_parts=fallback_parts,
            )
        ]
        if blocking_condition_ids:
            findings.append(
                _sequence_finding(
                    task=task,
                    constraint_code="blocker_open",
                    reason=(
                        "continuation blockers are still uncleared in symbolic state "
                        f"({', '.join(blocking_condition_ids)})"
                    ),
                    part_name=task.get("part_name"),
                    evidence={"condition_ids": deepcopy(blocking_condition_ids)},
                )
            )

    projected_cleared_condition_ids = [
        condition_id
        for condition_id, condition in condition_lookup.items()
        if _continuation_condition_cleared(
            condition,
            resources_by_jid=projected_resources,
            parts_by_name=projected_parts,
            fallback_parts=fallback_parts,
        )
    ]
    claimed_condition_ids = _task_closes_condition_ids(task)
    if claimed_condition_ids:
        not_currently_unmet = [
            condition_id
            for condition_id in claimed_condition_ids
            if condition_id not in condition_lookup
        ]
        if not_currently_unmet:
            findings.append(
                _sequence_finding(
                    task=task,
                    constraint_code="claimed_condition_not_currently_unmet",
                    reason=(
                        "closes_condition_ids references continuation condition ids "
                        "that are not currently unmet "
                        f"({', '.join(not_currently_unmet)})"
                    ),
                    part_name=task.get("part_name"),
                    evidence={"claimed_condition_ids": deepcopy(not_currently_unmet)},
                    claimed_condition_ids=not_currently_unmet,
                )
            )
        not_cleared = [
            condition_id
            for condition_id in claimed_condition_ids
            if condition_id in condition_lookup
            and condition_id not in projected_cleared_condition_ids
        ]
        if not_cleared:
            findings.append(
                _sequence_finding(
                    task=task,
                    constraint_code="claimed_condition_not_cleared",
                    reason=(
                        "closes_condition_ids claims continuation conditions that remain "
                        f"unmet after projection ({', '.join(not_cleared)})"
                    ),
                    part_name=task.get("part_name"),
                    evidence={"claimed_condition_ids": deepcopy(not_cleared)},
                    claimed_condition_ids=not_cleared,
                )
            )
    claimed_task_ids = _task_enables_task_ids(task)
    if claimed_task_ids:
        not_pending = [
            task_id for task_id in claimed_task_ids if task_id not in pending_tasks_by_id
        ]
        if not_pending:
            findings.append(
                _sequence_finding(
                    task=task,
                    constraint_code="claimed_task_not_pending",
                    reason=(
                        "enables_task_ids references task ids that are not pending "
                        f"nominal tasks ({', '.join(not_pending)})"
                    ),
                    part_name=task.get("part_name"),
                    evidence={"claimed_task_ids": deepcopy(not_pending)},
                    claimed_task_ids=not_pending,
                )
            )
        not_currently_blocked = [
            task_id
            for task_id in claimed_task_ids
            if task_id in pending_tasks_by_id
            and not [
                str(item).strip()
                for item in (
                    dict(pending_tasks_by_id.get(task_id) or {}).get("blocked_by_condition_ids")
                    or []
                )
                if str(item).strip()
            ]
        ]
        if not_currently_blocked:
            findings.append(
                _sequence_finding(
                    task=task,
                    constraint_code="claimed_task_not_currently_blocked",
                    reason=(
                        "enables_task_ids references nominal tasks that are not currently "
                        f"blocked in recovery gap state ({', '.join(not_currently_blocked)})"
                    ),
                    part_name=task.get("part_name"),
                    evidence={"claimed_task_ids": deepcopy(not_currently_blocked)},
                    claimed_task_ids=not_currently_blocked,
                )
            )
        projected_enabled_task_ids = _projected_enabled_task_ids(
            pending_tasks_by_id=pending_tasks_by_id,
            projected_cleared_condition_ids=projected_cleared_condition_ids,
        )
        not_enabled = [
            task_id
            for task_id in claimed_task_ids
            if task_id in pending_tasks_by_id and task_id not in projected_enabled_task_ids
        ]
        if not_enabled:
            findings.append(
                _sequence_finding(
                    task=task,
                    constraint_code="claimed_task_not_enabled",
                    reason=(
                        "enables_task_ids claims blocked nominal tasks that remain blocked "
                        f"after projection ({', '.join(not_enabled)})"
                    ),
                    part_name=task.get("part_name"),
                    evidence={"claimed_task_ids": deepcopy(not_enabled)},
                    claimed_task_ids=not_enabled,
                )
            )
    if previously_cleared_condition_ids:
        reopened_condition_ids = [
            condition_id
            for condition_id in previously_cleared_condition_ids
            if condition_id not in projected_cleared_condition_ids
        ]
        task_id = str(task.get("outline_id") or "").strip()
        current_index = int(task_index_by_id.get(task_id) or 0)
        has_future_continuation = any(
            isinstance(candidate_task, dict)
            and int(task_index_by_id.get(str(candidate_task.get("outline_id") or "").strip()) or -1)
            > current_index
            and str(
                task_types_by_id.get(str(candidate_task.get("outline_id") or "").strip()) or ""
            ).strip()
            == "continuation_resume"
            for candidate_task in (outline_tasks or [])
        )
        if reopened_condition_ids and has_future_continuation:
            findings.append(
                _sequence_finding(
                    task=task,
                    constraint_code="condition_reopened",
                    reason=(
                        "projected state transition reopens previously cleared continuation "
                        f"conditions ({', '.join(reopened_condition_ids)})"
                    ),
                    part_name=task.get("part_name"),
                    evidence={"condition_ids": deepcopy(reopened_condition_ids)},
                )
            )

    safety_result = validate_outline_macro_bridge_safety(
        task=task,
        signature=signature,
        pre_resources=pre_resources,
        pre_parts=pre_parts,
        projected_resources=projected_resources,
        projected_parts=projected_parts,
        llm_input=llm_input,
    )
    findings.extend(list(safety_result.get("findings") or []))
    return {
        "is_valid": not findings,
        "findings": findings,
        "cleared_condition_ids": _dedupe_tokens(
            list(projected_cleared_condition_ids)
            + list(safety_result.get("cleared_condition_ids") or [])
        ),
        "monitor_state": {
            "projected_cleared_condition_ids": deepcopy(projected_cleared_condition_ids),
            "safety_ctx": deepcopy(safety_result.get("safety_ctx") or {}),
        },
    }


__all__ = [
    "project_outline_macro_bridge_aps",
    "validate_outline_macro_bridge_safety",
    "validate_outline_macro_cca_constraints",
]
