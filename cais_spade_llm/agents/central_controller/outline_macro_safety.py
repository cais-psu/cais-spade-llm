"""CCA-owned generic recovery AP projection and outline-time safety validation."""

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
    entity_kind = str(condition.get("entity_kind") or "").strip().lower()
    entity = str(condition.get("entity") or "").strip()
    field = str(condition.get("field") or "").strip()
    expected = condition.get("expected")
    if kind in {"focused_resource_terminal_state", "resource_terminal_state"}:
        resource_jid = str(condition.get("entity") or "").strip()
        expected_state = str(condition.get("expected") or "").strip()
        if entity_kind == "part":
            row = dict(parts_by_name.get(entity) or {})
        else:
            row = dict(resources_by_jid.get(resource_jid) or {})
        if field:
            return row.get(field) == expected
        current_state = str(row.get("current_state") or "").strip()
        return bool(expected_state and current_state == expected_state)
    if kind == "safety_destination_occupancy" and entity_kind == "resource":
        row = dict(resources_by_jid.get(entity) or {})
        expected_not = (
            str(dict(expected).get("not") or "").strip()
            if isinstance(expected, dict)
            else ""
        )
        current_location = str(
            row.get("current_location")
            or row.get("resource_location")
            or dict(row.get("occupancy") or {}).get("location")
            or ""
        ).strip()
        return bool(expected_not and current_location != expected_not)
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


def _effective_task_part_name(task: dict[str, Any], signature: dict[str, Any]) -> str:
    return str(task.get("part_name") or signature.get("inferable_primary_part") or "").strip()


def _recovery_loaded_rules(llm_input: dict[str, Any]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for raw_rule in llm_input.get("loaded_safety_rules") or []:
        if not isinstance(raw_rule, dict):
            continue
        ap_scope = str(raw_rule.get("ap_scope") or "").strip().lower()
        if ap_scope not in {"recovery", "bridge", "both", "nominal"}:
            continue
        dfa_dot = str(raw_rule.get("dfa_dot") or "").strip()
        recovery_aps = _recovery_rule_aps(raw_rule)
        if not dfa_dot or not recovery_aps:
            continue
        rule = deepcopy(raw_rule)
        rule.setdefault("rule_id", str(rule.get("id") or rule.get("rule_id") or "").strip())
        rule["recovery_aps"] = recovery_aps
        rule["dfa_dot"] = dfa_dot
        selected.append(rule)
    return selected


def _parse_recovery_selector_from_ap_full(ap_full: str) -> dict[str, Any] | None:
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


def _recovery_rule_aps(raw_rule: dict[str, Any]) -> list[dict[str, Any]]:
    explicit_recovery_aps = [
        deepcopy(ap)
        for ap in (raw_rule.get("recovery_aps") or [])
        if isinstance(ap, dict)
        and str(ap.get("label") or "").strip()
        and str(ap.get("full") or "").strip()
    ]
    if explicit_recovery_aps:
        return explicit_recovery_aps
    derived_aps: list[dict[str, Any]] = []
    for raw_ap in raw_rule.get("aps") or []:
        if not isinstance(raw_ap, dict):
            continue
        label = str(raw_ap.get("label") or "").strip()
        full = str(raw_ap.get("full") or "").strip()
        selector = dict(raw_ap.get("selector") or {})
        if not selector:
            selector = dict(_parse_recovery_selector_from_ap_full(full) or {})
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


def _recovery_monitor_rules(rules: list[dict[str, Any]]) -> list[dict[str, Any]]:
    monitor_rules: list[dict[str, Any]] = []
    for raw_rule in rules:
        rule_id = str(raw_rule.get("id") or raw_rule.get("rule_id") or "").strip()
        recovery_aps = [
            {
                "label": str(ap.get("label") or "").strip(),
                "full": str(ap.get("full") or "").strip(),
            }
            for ap in (raw_rule.get("recovery_aps") or [])
            if isinstance(ap, dict)
            and str(ap.get("label") or "").strip()
            and str(ap.get("full") or "").strip()
        ]
        if not rule_id or not recovery_aps:
            continue
        monitor_rules.append(
            {
                "id": rule_id,
                "rule_id": rule_id,
                "constraint_type": deepcopy(raw_rule.get("constraint_type")),
                "aps": recovery_aps,
            }
        )
    return monitor_rules


def _recovery_dfa_dots(rules: list[dict[str, Any]]) -> dict[str, str]:
    dfa_dots: dict[str, str] = {}
    for raw_rule in rules:
        rule_id = str(raw_rule.get("id") or raw_rule.get("rule_id") or "").strip()
        dfa_dot = str(raw_rule.get("dfa_dot") or "").strip()
        if rule_id and dfa_dot:
            dfa_dots[rule_id] = dfa_dot
    return dfa_dots


def _recovery_rule_lookup(rules: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for raw_rule in rules:
        rule_id = str(raw_rule.get("id") or raw_rule.get("rule_id") or "").strip()
        if rule_id:
            lookup[rule_id] = deepcopy(raw_rule)
    return lookup


def _recovery_rule_labels(rules: list[dict[str, Any]]) -> set[str]:
    labels: set[str] = set()
    for raw_rule in rules:
        for raw_ap in raw_rule.get("recovery_aps") or []:
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


def project_outline_macro_recovery_aps(
    *,
    task: dict[str, Any],
    signature: dict[str, Any],
    pre_resources: dict[str, dict[str, Any]],
    pre_parts: dict[str, dict[str, Any]],
    projected_resources: dict[str, dict[str, Any]],
    projected_parts: dict[str, dict[str, Any]],
    llm_input: dict[str, Any],
) -> dict[str, Any]:
    rules = _recovery_loaded_rules(llm_input)
    candidate_aps: list[str] = []
    predicted_state_aps: list[str] = []
    current_state_aps: list[str] = []
    for raw_rule in rules:
        for raw_ap in raw_rule.get("recovery_aps") or []:
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


def _recovery_running_aps(llm_input: dict[str, Any], rules: list[dict[str, Any]]) -> list[str]:
    recovery_ctx = dict(llm_input.get("recovery_safety_context") or {})
    allowed_labels = _recovery_rule_labels(rules)
    running_aps = [
        str(label).strip()
        for label in (recovery_ctx.get("running_aps") or [])
        if str(label).strip() in allowed_labels
    ]
    return _dedupe_tokens(running_aps)


def _build_recovery_safety_monitor(
    *,
    rules: list[dict[str, Any]],
    state_aps: list[str],
    llm_input: dict[str, Any],
) -> OnlineSafetyMonitor:
    monitor = OnlineSafetyMonitor(
        _recovery_dfa_dots(rules),
        _recovery_monitor_rules(rules),
    )
    if state_aps:
        monitor.resource_state_aps["recovery_scope"] = set(state_aps)
    running_aps = _recovery_running_aps(llm_input, rules)
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
    safety_dfa_states: dict[str, str],
) -> tuple[list[str], list[str]]:
    rule_lookup = _recovery_rule_lookup(rules)
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
        projection = project_outline_macro_recovery_aps(
            task=proxy_task,
            signature=proxy_signature,
            pre_resources=projected_resources,
            pre_parts=projected_parts,
            projected_resources=projected_resources,
            projected_parts=projected_parts,
            llm_input={
                "loaded_safety_rules": [rule],
                "recovery_safety_context": llm_input.get("recovery_safety_context") or {},
            },
        )
        monitor = _build_recovery_safety_monitor(
            rules=[rule],
            state_aps=list(projection.get("current_state_aps") or []),
            llm_input=llm_input,
        )
        projected_rule_state = str(
            safety_dfa_states.get(blocking_rule_id) or ""
        ).strip()
        if projected_rule_state and blocking_rule_id in monitor.current_states:
            monitor.current_states[blocking_rule_id] = projected_rule_state
        allowed, _ = monitor.online_safety_validation(
            list(projection.get("candidate_aps") or []),
            predicted_state_aps=list(projection.get("predicted_state_aps") or []),
        )
        if allowed:
            safe_next_task_ids.append(blocked_task_id)
            cleared_condition_ids.append(condition_id)
    return _dedupe_tokens(safe_next_task_ids), _dedupe_tokens(cleared_condition_ids)


def validate_outline_macro_recovery_safety(
    *,
    task: dict[str, Any],
    signature: dict[str, Any],
    pre_resources: dict[str, dict[str, Any]],
    pre_parts: dict[str, dict[str, Any]],
    projected_resources: dict[str, dict[str, Any]],
    projected_parts: dict[str, dict[str, Any]],
    llm_input: dict[str, Any],
    safety_dfa_states_before: dict[str, str] | None = None,
) -> dict[str, Any]:
    rules = _recovery_loaded_rules(llm_input)
    projection = project_outline_macro_recovery_aps(
        task=task,
        signature=signature,
        pre_resources=pre_resources,
        pre_parts=pre_parts,
        projected_resources=projected_resources,
        projected_parts=projected_parts,
        llm_input=llm_input,
    )
    conditions_by_id = _modeled_gap_unmet_conditions_by_id(llm_input)
    active_safety_condition_ids = [
        condition_id
        for condition_id, condition in conditions_by_id.items()
        if str(condition.get("blocking_rule_id") or "").strip()
        or str(condition.get("kind") or "").strip().startswith("safety_")
    ]
    pending_tasks_by_id = _modeled_gap_pending_tasks_by_id(llm_input)

    if not rules:
        if safety_dfa_states_before:
            raise ValueError(
                "projected safety DFA states reference rules that are not active"
            )
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
            "safety_dfa_states_before": {},
            "safety_dfa_states_after": {},
        }

    monitor = _build_recovery_safety_monitor(
        rules=rules,
        state_aps=list(projection.get("current_state_aps") or []),
        llm_input=llm_input,
    )
    if safety_dfa_states_before is not None:
        supplied_rule_ids = set(safety_dfa_states_before)
        active_rule_ids = set(monitor.current_states)
        if supplied_rule_ids != active_rule_ids:
            raise ValueError(
                "projected safety DFA rule identifiers do not match the active rules"
            )
        for rule_id, state in safety_dfa_states_before.items():
            state_token = str(state or "").strip()
            dfa = dict(monitor.dfas.get(rule_id) or {})
            transitions = dict(dfa.get("transitions") or {})
            known_states = set(transitions)
            known_states.update(
                str(destination)
                for rows in transitions.values()
                for _, destination in rows
            )
            if not state_token or state_token not in known_states:
                raise ValueError(
                    f"projected safety DFA state is invalid for rule '{rule_id}'"
                )
            monitor.current_states[rule_id] = state_token
    dfa_states_before = {
        rule_id: str(monitor.current_states[rule_id])
        for rule_id in sorted(monitor.current_states)
    }
    allowed, info = monitor.online_safety_validation(
        list(projection.get("candidate_aps") or []),
        predicted_state_aps=list(projection.get("predicted_state_aps") or []),
    )
    candidate_dfa_states_after = deepcopy(dfa_states_before)
    if allowed:
        candidate_dfa_states_after.update(
            {
                rule_id: str(state)
                for rule_id, state in sorted(
                    dict(info.get("next_states") or {}).items()
                )
            }
        )
    rule_lookup = _recovery_rule_lookup(rules)
    violated_rule_id = str(info.get("violated_rule") or "").strip()
    violated_rule = dict(rule_lookup.get(violated_rule_id) or {})
    related_condition_ids = _claimed_safety_condition_ids_for_rule(
        claimed_condition_ids=active_safety_condition_ids,
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
            claimed_condition_ids=active_safety_condition_ids,
            projected_resources=projected_resources,
            projected_parts=projected_parts,
            llm_input=llm_input,
            safety_dfa_states=candidate_dfa_states_after,
        )
        cleared_condition_ids = _dedupe_tokens(
            cleared_condition_ids
            + [
                condition_id
                for condition_id, condition in conditions_by_id.items()
                if (
                    str(condition.get("blocking_rule_id") or "").strip()
                    or str(condition.get("kind") or "").strip().startswith("safety_")
                )
                and _continuation_condition_cleared(
                    condition,
                    resources_by_jid=projected_resources,
                    parts_by_name=projected_parts,
                    fallback_parts=_fault_event_fallback_parts(llm_input),
                )
            ]
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
                "pose_source": "recovery_safety_rule",
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
    dfa_states_after = candidate_dfa_states_after
    return {
        "is_safe": bool(allowed),
        "safety_ctx": safety_ctx,
        "findings": findings,
        "handled_condition_ids": handled_condition_ids,
        "cleared_condition_ids": cleared_condition_ids,
        "safety_dfa_states_before": dfa_states_before,
        "safety_dfa_states_after": dfa_states_after,
    }


__all__ = [
    "project_outline_macro_recovery_aps",
    "validate_outline_macro_recovery_safety",
]
