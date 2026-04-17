"""Internal helper functions for DES recovery outline runtime."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from typing import Any

from cais_spade_llm.agents.central_controller.outline_macro_safety import (
    validate_outline_macro_cca_constraints,
)
from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.modes.des_recovery_automaton import (
    _delta,
    _map_event_to_aps,
)
from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.bridge_grounding_compiler import (
    compile_grounded_outline_task,
)
from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.modes.des_recovery_outline_state import (
    apply_outline_task_effects,
    build_outline_task_type_lookup,
    infer_outline_macro_signature,
    outline_task_depends_on,
    task_findings_block_projected_state,
)
from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.modes.des_recovery_validation_context import (
    projected_outline_validation_context,
)

_DEFAULT_CANDIDATE_BOUND = 5
_DEFAULT_CANDIDATE_BOUND_CAP = 8
_CANDIDATE_PRUNE_REPEAT_THRESHOLD = 2
_DES_RECOVERY_OUTLINE_CONTRACT = {
    "allowed_state_fields": [
        "resource_state",
        "held_part",
        "part_state",
        "part_location",
        "part_holder_resource_jid",
    ],
    "disallow_unknown_state_fields": True,
    "require_expected_start_match": True,
    "require_meaningful_delta": True,
    "require_release_destination_for_release": True,
    "require_carrier_for_part_relocation": True,
}
_DES_RECOVERY_DURABLE_PRUNED_CONSTRAINT_CODES = {
    "workspace_unreachable",
    "holder_conflict",
    "required_part_not_held",
    "source_reference_unavailable",
    "part_relocation_without_carrier",
    "safety_rule_violation",
    "blocker_open",
    "dependency_unsatisfied",
    "order_violation",
}


def sync_des_recovery_aliases(
    session_state: dict[str, Any],
    *,
    turn_entry: dict[str, Any] | None = None,
    transition_validation: dict[str, Any] | None = None,
    unresolved_target_predicates: list[dict[str, Any]] | None = None,
) -> None:
    """Maintain DES-style debug aliases without changing parser-facing fields."""
    accepted_prefix = [
        deepcopy(row)
        for row in (session_state.get("accepted_outline_prefix") or [])
        if isinstance(row, dict)
    ]
    session_state["accepted_transition_prefix"] = deepcopy(accepted_prefix)
    session_state["des_event_sequence"] = deepcopy(accepted_prefix)
    session_state["transition_trace"] = deepcopy(accepted_prefix)

    if transition_validation is not None:
        session_state["transition_validation"] = deepcopy(transition_validation)
    if unresolved_target_predicates is not None:
        session_state["unresolved_target_predicates"] = deepcopy(
            unresolved_target_predicates
        )

    if turn_entry is None:
        return
    turn_entry["accepted_transition_prefix"] = deepcopy(accepted_prefix)
    turn_entry["des_event_sequence"] = deepcopy(accepted_prefix)
    turn_entry["transition_trace"] = deepcopy(accepted_prefix)
    if transition_validation is not None:
        turn_entry["transition_validation"] = deepcopy(transition_validation)
    if unresolved_target_predicates is not None:
        turn_entry["unresolved_target_predicates"] = deepcopy(
            unresolved_target_predicates
        )


def symbolic_state_fingerprint(
    symbolic_resources: dict[str, dict[str, Any]],
    symbolic_parts: dict[str, dict[str, Any]],
) -> str:
    """Content-addressed hash of projected symbolic state -> deterministic state name."""

    def _canonical(d: dict[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in sorted(d.items()) if v is not None}

    payload = json.dumps(
        {
            "r": {k: _canonical(v) for k, v in sorted(symbolic_resources.items())},
            "p": {k: _canonical(v) for k, v in sorted(symbolic_parts.items())},
        },
        sort_keys=True,
        default=str,
    )
    return "S_" + hashlib.sha256(payload.encode()).hexdigest()[:12]


def extend_plant_with_event(
    plant: dict[str, Any],
    event_name: str,
    event_dict: dict[str, Any],
    from_state: str,
    to_state: str,
) -> None:
    """Append a transition to the running DES plant."""
    states = plant["states"]
    if from_state not in states:
        states.append(from_state)
    if to_state not in states:
        states.append(to_state)
    plant["events"][event_name] = {
        **event_dict,
        "from": from_state,
        "to": to_state,
    }
    plant["transitions"].setdefault(from_state, []).append((event_name, to_state))


def advance_des_safety_state(
    candidate_event: dict[str, Any],
    current_safety_q: tuple[str, ...],
    safety_dfas: dict[str, dict[str, Any]],
    ap_descriptors: list[dict[str, Any]],
) -> tuple[bool, tuple[str, ...], list[str]]:
    """Advance cached DES safety DFA state for an already-accepted event."""
    if not safety_dfas:
        return False, current_safety_q, []

    sigma = _map_event_to_aps(candidate_event, ap_descriptors)
    new_q_list: list[str] = []
    violated: list[str] = []

    for (rule_id, dfa), q_current in zip(safety_dfas.items(), current_safety_q):
        q_next = _delta(dfa, q_current, sigma)
        new_q_list.append(q_next)
        violation_state = str(dfa.get("violation_state") or "").strip()
        if violation_state and q_next == violation_state:
            violated.append(rule_id)

    return bool(violated), tuple(new_q_list), violated


def _resource_agent_map(planner: Any) -> dict[str, Any]:
    return {
        str(getattr(agent, "jid", "")).strip(): agent
        for agent in (getattr(planner, "resource_agents", None) or [])
        if str(getattr(agent, "jid", "")).strip()
    }


def _resource_constraint_finding(
    *,
    task: dict[str, Any],
    constraint_code: str,
    reason: str,
    resource_jid: str = "",
    part_name: str = "",
    evidence: dict[str, Any] | None = None,
    guard: dict[str, Any] | None = None,
) -> dict[str, Any]:
    evidence = dict(evidence or {})
    return {
        "task_id": str(task.get("outline_id") or "").strip(),
        "resource_jid": resource_jid or None,
        "part_name": part_name or None,
        "pose_source": "resource_feasibility",
        "pose": deepcopy(evidence.get("checked_pose")),
        "workspace_bounds": deepcopy(evidence.get("workspace_bounds")),
        "failed_axes": [constraint_code],
        "constraint_owner": "resource",
        "constraint_family": "resource_feasibility",
        "constraint_code": constraint_code,
        "reason": reason,
        "guard": deepcopy(guard),
        "evidence": deepcopy(evidence),
    }


def _resource_part_context(
    *,
    grounded_action: dict[str, Any],
    resource_row: dict[str, Any],
    parts_by_name: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    part_name = str(grounded_action.get("part_name") or "").strip()
    action_target = dict(grounded_action.get("target") or {})
    part_context = deepcopy(dict(parts_by_name.get(part_name) or {}))
    part_context["target"] = deepcopy(action_target)
    part_context["resource_held_part"] = str(resource_row.get("held_part") or "").strip() or None
    part_context["resource_gripper_state"] = str(
        resource_row.get("gripper_state") or ""
    ).strip() or None
    part_context["named_pose"] = str(action_target.get("named_pose") or "").strip() or None
    if "pose" in action_target:
        part_context["pose"] = deepcopy(action_target.get("pose"))
    return part_context


def validate_single_outline_task(
    *,
    planner: Any,
    task: dict[str, Any],
    session_state: dict[str, Any],
    prepared_bridge_request: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    resources_by_jid, parts_by_name = projected_outline_validation_context(
        session_state=session_state,
        prepared_bridge_request=prepared_bridge_request,
    )
    grounding_result = compile_grounded_outline_task(
        task,
        resources_by_jid=resources_by_jid,
        parts_by_name=parts_by_name,
        outline_contract=deepcopy(_DES_RECOVERY_OUTLINE_CONTRACT),
        location_validation_mode="strict",
    )
    finding = grounding_result.get("finding")
    if isinstance(finding, dict):
        return [deepcopy(finding)], None

    grounded_action = dict(grounding_result.get("grounded_action") or {})
    if not grounded_action:
        return [], None

    findings: list[dict[str, Any]] = []
    findings.extend(
        _validate_outline_task_ra(
            planner=planner,
            task=task,
            grounded_action=grounded_action,
            resources_by_jid=resources_by_jid,
            parts_by_name=parts_by_name,
        )
    )
    findings.extend(
        _validate_outline_task_cca(
            task=task,
            grounded_action=grounded_action,
            resources_by_jid=resources_by_jid,
            parts_by_name=parts_by_name,
            llm_input=dict(prepared_bridge_request.get("llm_input") or {}),
            prior_findings=findings,
        )
    )
    return findings, grounded_action


def _validate_outline_task_ra(
    *,
    planner: Any,
    task: dict[str, Any],
    grounded_action: dict[str, Any],
    resources_by_jid: dict[str, dict[str, Any]],
    parts_by_name: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    resource_jid = str(grounded_action.get("resource_jid") or "").strip()
    part_name = str(grounded_action.get("part_name") or "").strip()
    resource_row = dict(resources_by_jid.get(resource_jid) or {})
    if not resource_jid or not resource_row:
        return [
            _resource_constraint_finding(
                task=task,
                constraint_code="resource_unavailable",
                reason=(
                    f"resource '{resource_jid or 'unknown'}' is not available in current bridge state"
                ),
                resource_jid=resource_jid,
                part_name=part_name,
                guard={"kind": "resource_not_available", "resource_jid": resource_jid}
                if resource_jid
                else None,
            )
        ]

    resource_agent = _resource_agent_map(planner).get(resource_jid)
    oracle = getattr(resource_agent, "bridge_feasibility_oracle", None)
    if not callable(oracle):
        return []

    resource_snapshot = deepcopy(resource_row)
    get_bridge_snapshot = getattr(resource_agent, "get_bridge_snapshot", None)
    if callable(get_bridge_snapshot):
        try:
            maybe_snapshot = get_bridge_snapshot()
        except Exception:
            maybe_snapshot = {}
        if isinstance(maybe_snapshot, dict):
            for field_name in (
                "workspace_bounds",
                "available_named_poses",
                "bridge_adapter",
                "resource_type",
                "role",
            ):
                if field_name not in resource_snapshot and field_name in maybe_snapshot:
                    resource_snapshot[field_name] = deepcopy(maybe_snapshot.get(field_name))

    try:
        oracle_result = oracle(
            operation_kind=str(grounded_action.get("operation_kind") or "").strip(),
            part_name=part_name or None,
            part_context=_resource_part_context(
                grounded_action=grounded_action,
                resource_row=resource_row,
                parts_by_name=parts_by_name,
            ),
            bridge_snapshot=resource_snapshot,
            grounded_action=deepcopy(grounded_action),
        )
    except Exception as exc:
        return [
            _resource_constraint_finding(
                task=task,
                constraint_code="resource_unavailable",
                reason=f"resource feasibility oracle failed: {exc}",
                resource_jid=resource_jid,
                part_name=part_name,
            )
        ]

    result = dict(oracle_result or {})
    if bool(result.get("allowed", True)):
        return []

    constraint_code = (
        str(result.get("constraint_code") or "").strip() or "resource_unavailable"
    )
    reason = (
        str(result.get("reason") or "").strip()
        or "resource feasibility rejected the grounded action"
    )
    return [
        _resource_constraint_finding(
            task=task,
            constraint_code=constraint_code,
            reason=reason,
            resource_jid=resource_jid,
            part_name=part_name,
            evidence=dict(result.get("evidence") or {}),
            guard=dict(result.get("guard") or {}),
        )
    ]


def _validate_outline_task_cca(
    *,
    task: dict[str, Any],
    grounded_action: dict[str, Any],
    resources_by_jid: dict[str, dict[str, Any]],
    parts_by_name: dict[str, dict[str, Any]],
    llm_input: dict[str, Any],
    prior_findings: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    outline_tasks = [deepcopy(task)]
    task_id = str(task.get("outline_id") or "").strip() or "task_0"
    task_types_by_id = build_outline_task_type_lookup(
        outline_tasks,
        resources_by_jid=resources_by_jid,
        parts_by_name=parts_by_name,
        llm_input=llm_input,
    )
    signature = infer_outline_macro_signature(
        task,
        resources_by_jid=resources_by_jid,
        parts_by_name=parts_by_name,
    )

    projected_resources = deepcopy(resources_by_jid)
    projected_parts = deepcopy(parts_by_name)
    if not task_findings_block_projected_state(list(prior_findings or [])):
        task_type = str(task_types_by_id.get(task_id) or "").strip()
        apply_outline_task_effects(
            task,
            resources_by_jid=projected_resources,
            parts_by_name=projected_parts,
            task_type=task_type,
            grounded_action=grounded_action,
        )

    cca_result = validate_outline_macro_cca_constraints(
        task=deepcopy(task),
        grounded_action=deepcopy(grounded_action),
        signature=deepcopy(signature),
        pre_resources=deepcopy(resources_by_jid),
        pre_parts=deepcopy(parts_by_name),
        projected_resources=projected_resources,
        projected_parts=projected_parts,
        llm_input=deepcopy(llm_input),
        outline_tasks=outline_tasks,
        task_types_by_id=deepcopy(task_types_by_id),
        task_index_by_id={task_id: 0},
        dependency_map={task_id: outline_task_depends_on(task)},
        previously_cleared_condition_ids=None,
    )
    return [
        deepcopy(row)
        for row in (cca_result.get("findings") or [])
        if isinstance(row, dict)
    ]


def _outline_validation_finding_key(
    finding: dict[str, Any],
) -> tuple[str, str, str, str, str]:
    """Small stable key for de-duping DES recovery validation findings."""
    evidence = dict(finding.get("evidence") or {})
    return (
        str(finding.get("constraint_owner") or "").strip().lower(),
        str(finding.get("constraint_code") or "").strip().lower(),
        str(finding.get("resource_jid") or "").strip(),
        str(finding.get("part_name") or "").strip(),
        str(evidence.get("field") or evidence.get("token") or "").strip().lower(),
    )


def merge_outline_validation_findings(
    existing: list[dict[str, Any]],
    new: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Append new findings while de-duping equivalent active findings."""
    merged: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    for row in list(existing or []) + list(new or []):
        if not isinstance(row, dict):
            continue
        merged[_outline_validation_finding_key(row)] = deepcopy(row)
    return list(merged.values())


def _finding_still_unresolved(
    finding: dict[str, Any],
    *,
    session_state: dict[str, Any],
    prepared_bridge_request: dict[str, Any],
) -> bool:
    del prepared_bridge_request
    constraint_code = str(finding.get("constraint_code") or "").strip().lower()
    resource_jid = str(finding.get("resource_jid") or "").strip()
    part_name = str(finding.get("part_name") or "").strip()
    symbolic_resources = dict(session_state.get("symbolic_resources") or {})
    symbolic_parts = dict(session_state.get("symbolic_parts") or {})
    resource_row = dict(symbolic_resources.get(resource_jid) or {})
    part_row = dict(symbolic_parts.get(part_name) or {})

    if constraint_code == "workspace_unreachable":
        if not resource_jid or not part_name:
            return False
        current_holder = str(part_row.get("current_holder_resource_jid") or "").strip()
        current_state = str(part_row.get("current_state") or "").strip().lower()
        current_location = str(part_row.get("current_location") or "").strip()
        if current_holder or current_state in {"held", "in_gripper", "assembled", "placed"}:
            return False
        if current_location:
            return False
        return True
    if constraint_code == "holder_conflict":
        return bool(
            part_name
            and resource_jid
            and str(part_row.get("current_holder_resource_jid") or "").strip() != resource_jid
            and str(part_row.get("current_holder_resource_jid") or "").strip()
        )
    if constraint_code == "required_part_not_held":
        return bool(resource_jid and part_name and str(resource_row.get("held_part") or "").strip() != part_name)
    if constraint_code == "part_relocation_without_carrier":
        current_holder = str(part_row.get("current_holder_resource_jid") or "").strip()
        actual_held = str(resource_row.get("held_part") or "").strip()
        return bool(part_name and resource_jid and current_holder != resource_jid and actual_held != part_name)
    return False


def prune_resolved_outline_validation_findings(
    active_findings: list[dict[str, Any]],
    *,
    session_state: dict[str, Any],
    prepared_bridge_request: dict[str, Any],
) -> list[dict[str, Any]]:
    unresolved: list[dict[str, Any]] = []
    for row in active_findings or []:
        if not isinstance(row, dict):
            continue
        if _finding_still_unresolved(
            row,
            session_state=session_state,
            prepared_bridge_request=prepared_bridge_request,
        ):
            unresolved.append(deepcopy(row))
    return unresolved


def _active_continuation_conditions(
    prepared_bridge_request: dict[str, Any],
) -> list[dict[str, Any]]:
    llm_input = dict(prepared_bridge_request.get("llm_input") or {})
    modeled_gap = dict(llm_input.get("modeled_continuation_gap") or {})
    raw_conditions = (
        modeled_gap.get("unmet_continuation_conditions")
        or modeled_gap.get("unsatisfied_conditions")
        or []
    )
    return [deepcopy(row) for row in raw_conditions if isinstance(row, dict)]


def _normalized_blocker_kind(condition: dict[str, Any]) -> str:
    kind = str(condition.get("kind") or "").strip().lower()
    if kind == "focused_resource_terminal_state":
        return "resource_terminal_state"
    return kind


def _fault_event_fallback_parts(
    prepared_bridge_request: dict[str, Any],
) -> list[str]:
    fault_event = dict(dict(prepared_bridge_request.get("llm_input") or {}).get("fault_event") or {})
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
    blocker_text = str(blocking_reason or "").strip().lower()
    blocker_parts = [
        part_name
        for part_name in parts_by_name
        if part_name and part_name.lower() in blocker_text
    ]
    return blocker_parts or [str(part_name).strip() for part_name in fallback_parts if str(part_name).strip()]


def _extract_safety_blocker_part_names(
    *,
    blocking_reason: str,
    parts_by_name: dict[str, dict[str, Any]],
    fallback_parts: list[str],
) -> list[str]:
    blocker_text = str(blocking_reason or "").strip()
    lowered = blocker_text.lower()
    if " before " in lowered:
        prefix = blocker_text[:lowered.index(" before ")].strip()
        blocker_parts = _extract_blocker_part_names(
            blocking_reason=prefix,
            parts_by_name=parts_by_name,
            fallback_parts=[],
        )
        if blocker_parts:
            return blocker_parts
    return _extract_blocker_part_names(
        blocking_reason=blocking_reason,
        parts_by_name=parts_by_name,
        fallback_parts=fallback_parts,
    )


def _condition_expected_matches(actual: Any, expected: Any) -> bool:
    if isinstance(expected, (dict, list)):
        return actual == expected
    if expected in (None, ""):
        return actual in (None, "")
    return str(actual or "").strip() == str(expected or "").strip()


def _continuation_condition_satisfied(
    condition: dict[str, Any],
    *,
    session_state: dict[str, Any],
    prepared_bridge_request: dict[str, Any],
) -> bool:
    kind = _normalized_blocker_kind(condition)
    entity_kind = str(condition.get("entity_kind") or "").strip().lower()
    entity = str(condition.get("entity") or "").strip()
    field = str(condition.get("field") or "").strip()
    expected = condition.get("expected")
    resources_by_jid, parts_by_name = projected_outline_validation_context(
        session_state=session_state,
        prepared_bridge_request=prepared_bridge_request,
    )
    if kind == "resource_terminal_state" and entity_kind == "resource" and entity:
        row = dict(resources_by_jid.get(entity) or {})
        if not row:
            return False
        if field in {"current_state", "current_location", "held_part", "gripper_state"}:
            return _condition_expected_matches(row.get(field), expected)
        return False
    if kind == "resource_terminal_state" and entity_kind == "part" and entity:
        row = dict(parts_by_name.get(entity) or {})
        if not row:
            return False
        if field in {"current_state", "current_location", "current_holder_resource_jid"}:
            return _condition_expected_matches(row.get(field), expected)
        return False
    if kind != "safety_blocked_suffix_task":
        return False
    blocker_parts = _extract_safety_blocker_part_names(
        blocking_reason=str(condition.get("blocking_reason") or "").strip(),
        parts_by_name=parts_by_name,
        fallback_parts=_fault_event_fallback_parts(prepared_bridge_request),
    )
    if not blocker_parts:
        return False
    for blocker_part in blocker_parts:
        part_row = dict(parts_by_name.get(blocker_part) or {})
        goal_location = str(part_row.get("goal_location") or "").strip()
        current_state = str(part_row.get("current_state") or "").strip().lower()
        current_location = str(part_row.get("current_location") or part_row.get("location") or "").strip()
        if current_state in {"placed", "assembled"}:
            continue
        if goal_location and current_location == goal_location:
            continue
        return False
    return True


def _candidate_recovery_blocker_key(
    blocker: dict[str, Any],
) -> tuple[str, str, str, str, str, str]:
    return (
        str(blocker.get("kind") or "").strip().lower(),
        str(blocker.get("entity_kind") or "").strip().lower(),
        str(blocker.get("entity") or "").strip(),
        str(blocker.get("field") or "").strip(),
        str(blocker.get("expected") or "").strip(),
        str(blocker.get("blocking_rule_id") or blocker.get("source_task_id") or "").strip(),
    )


def _candidate_recovery_blocker_summary(
    blocker: dict[str, Any],
    *,
    prepared_bridge_request: dict[str, Any],
    parts_by_name: dict[str, dict[str, Any]],
) -> str:
    kind = _normalized_blocker_kind(blocker)
    if kind == "resource_terminal_state":
        entity = str(blocker.get("entity") or "").strip()
        expected = str(blocker.get("expected") or "").strip()
        field = str(blocker.get("field") or "").strip()
        if entity and expected and field == "current_state":
            return f"{entity} must reach {expected}"
        if entity and expected and field:
            return f"{entity} {field} must reach {expected}"
        return f"{entity or 'resource'} blocker remains"
    if kind == "safety_blocked_suffix_task":
        blocking_reason = str(blocker.get("blocking_reason") or "").strip()
        if blocking_reason:
            return blocking_reason
        blocker_parts = _extract_safety_blocker_part_names(
            blocking_reason=blocking_reason,
            parts_by_name=parts_by_name,
            fallback_parts=_fault_event_fallback_parts(prepared_bridge_request),
        )
        if blocker_parts:
            blocker_part = blocker_parts[0]
            goal_location = str(dict(parts_by_name.get(blocker_part) or {}).get("goal_location") or "").strip()
            if goal_location:
                return f"{blocker_part} must be at {goal_location} before blocked suffix can resume"
            return f"{blocker_part} must be restored before blocked suffix can resume"
        return "Blocked suffix must be cleared before continuation can resume"
    return "Recovery blocker remains"


def _active_candidate_recovery_blockers(
    *,
    session_state: dict[str, Any],
    prepared_bridge_request: dict[str, Any],
) -> list[dict[str, Any]]:
    resources_by_jid, parts_by_name = projected_outline_validation_context(
        session_state=session_state,
        prepared_bridge_request=prepared_bridge_request,
    )
    blockers: dict[tuple[str, str, str, str, str, str], dict[str, Any]] = {}
    for condition in _active_continuation_conditions(prepared_bridge_request):
        if not isinstance(condition, dict):
            continue
        kind = _normalized_blocker_kind(condition)
        if kind not in {"resource_terminal_state", "safety_blocked_suffix_task"}:
            continue
        if _continuation_condition_satisfied(
            condition,
            session_state=session_state,
            prepared_bridge_request=prepared_bridge_request,
        ):
            continue
        blocker = deepcopy(condition)
        blocker["kind"] = kind
        blocker["summary"] = _candidate_recovery_blocker_summary(
            blocker,
            prepared_bridge_request=prepared_bridge_request,
            parts_by_name=parts_by_name,
        )
        blockers[_candidate_recovery_blocker_key(blocker)] = blocker
    return list(blockers.values())


def no_blocker_reduction_finding(
    *,
    task: dict[str, Any],
) -> dict[str, Any]:
    return {
        "task_id": str(task.get("outline_id") or "").strip(),
        "resource_jid": str(task.get("resource_jid") or "").strip() or None,
        "part_name": str(task.get("part_name") or "").strip() or None,
        "constraint_owner": "selector",
        "constraint_family": "candidate_selection",
        "constraint_code": "no_blocker_reduction",
        "reason": "Task does not directly reduce the current recovery blockers.",
        "evidence": {"field": "recovery_blockers"},
    }


def _candidate_session_after_task(
    *,
    session_state: dict[str, Any],
    task: dict[str, Any],
) -> dict[str, Any]:
    candidate_session_state = deepcopy(session_state)
    apply_task_effects_to_symbolic_state(task, candidate_session_state)
    return candidate_session_state


def _current_safety_blocker_parts(
    *,
    current_blockers: list[dict[str, Any]],
    parts_by_name: dict[str, dict[str, Any]],
    prepared_bridge_request: dict[str, Any],
) -> set[str]:
    blocker_parts: set[str] = set()
    for blocker in current_blockers:
        if not isinstance(blocker, dict):
            continue
        if _normalized_blocker_kind(blocker) != "safety_blocked_suffix_task":
            continue
        blocker_parts.update(
            _extract_safety_blocker_part_names(
                blocking_reason=str(blocker.get("blocking_reason") or "").strip(),
                parts_by_name=parts_by_name,
                fallback_parts=_fault_event_fallback_parts(prepared_bridge_request),
            )
        )
    return blocker_parts


def _task_ends_with_part_held_by_resource(task: dict[str, Any]) -> bool:
    resource_jid = str(task.get("resource_jid") or "").strip()
    part_name = str(task.get("part_name") or "").strip()
    end_state = dict(task.get("expected_end_state") or {})
    held_part = str(end_state.get("held_part") or "").strip()
    part_holder = str(end_state.get("part_holder_resource_jid") or "").strip()
    return bool(part_name and resource_jid and (held_part == part_name or part_holder == resource_jid))


def _task_ends_with_part_clear_of_resource(task: dict[str, Any]) -> bool:
    part_name = str(task.get("part_name") or "").strip()
    if not part_name:
        return False
    end_state = dict(task.get("expected_end_state") or {})
    held_part = end_state.get("held_part")
    part_holder = end_state.get("part_holder_resource_jid")
    return bool(
        ("part_location" in end_state or str(task.get("target_ref") or "").strip())
        and (held_part in (None, ""))
        and (part_holder in (None, ""))
    )


def _normalize_place_verb_in_action_name(task: dict[str, Any]) -> None:
    """Rewrite release-style labels into place-style labels for committed events."""
    if not _task_ends_with_part_clear_of_resource(task):
        return
    action_name = str(task.get("action_name") or "").strip()
    if not action_name:
        return
    first, _, rest = action_name.partition(" ")
    if first.lower() != "release":
        return
    task["action_name"] = (
        f"place {rest}".replace(" to ", " at ", 1) if rest else "place"
    )


def _part_release_frees_resource_for_blocker(
    *,
    task: dict[str, Any],
    session_state: dict[str, Any],
    candidate_session_state: dict[str, Any],
    prepared_bridge_request: dict[str, Any],
    current_blockers: list[dict[str, Any]],
) -> bool:
    if not _task_ends_with_part_clear_of_resource(task):
        return False
    resource_jid = str(task.get("resource_jid") or "").strip()
    if any(
        _normalized_blocker_kind(blocker) == "resource_terminal_state"
        and str(blocker.get("entity") or "").strip() == resource_jid
        for blocker in current_blockers
        if isinstance(blocker, dict)
    ):
        return False
    released_part = str(task.get("part_name") or "").strip()
    if not resource_jid or not released_part:
        return False
    resources_by_jid, parts_by_name = projected_outline_validation_context(
        session_state=session_state,
        prepared_bridge_request=prepared_bridge_request,
    )
    candidate_resources_by_jid, _ = projected_outline_validation_context(
        session_state=candidate_session_state,
        prepared_bridge_request=prepared_bridge_request,
    )
    current_resource = dict(resources_by_jid.get(resource_jid) or {})
    candidate_resource = dict(candidate_resources_by_jid.get(resource_jid) or {})
    if str(current_resource.get("held_part") or "").strip() != released_part:
        return False
    if str(candidate_resource.get("held_part") or "").strip():
        return False
    blocker_parts = _current_safety_blocker_parts(
        current_blockers=current_blockers,
        parts_by_name=parts_by_name,
        prepared_bridge_request=prepared_bridge_request,
    )
    return bool(blocker_parts and released_part not in blocker_parts)


def _part_acquisition_counts_as_blocker_progress(
    *,
    task: dict[str, Any],
    session_state: dict[str, Any],
    candidate_session_state: dict[str, Any],
    prepared_bridge_request: dict[str, Any],
    current_blockers: list[dict[str, Any]],
) -> bool:
    if not _task_ends_with_part_held_by_resource(task):
        return False
    resource_jid = str(task.get("resource_jid") or "").strip()
    if any(
        _normalized_blocker_kind(blocker) == "resource_terminal_state"
        and str(blocker.get("entity") or "").strip() == resource_jid
        for blocker in current_blockers
        if isinstance(blocker, dict)
    ):
        return False
    acquired_part = str(task.get("part_name") or "").strip()
    if not resource_jid or not acquired_part:
        return False
    _, parts_by_name = projected_outline_validation_context(
        session_state=session_state,
        prepared_bridge_request=prepared_bridge_request,
    )
    blocker_parts = _current_safety_blocker_parts(
        current_blockers=current_blockers,
        parts_by_name=parts_by_name,
        prepared_bridge_request=prepared_bridge_request,
    )
    if acquired_part not in blocker_parts:
        return False
    candidate_resources_by_jid, candidate_parts_by_name = projected_outline_validation_context(
        session_state=candidate_session_state,
        prepared_bridge_request=prepared_bridge_request,
    )
    candidate_resource = dict(candidate_resources_by_jid.get(resource_jid) or {})
    candidate_part = dict(candidate_parts_by_name.get(acquired_part) or {})
    return bool(
        str(candidate_resource.get("held_part") or "").strip() == acquired_part
        and str(candidate_part.get("current_holder_resource_jid") or "").strip() == resource_jid
    )


def _preparatory_transit_toward_blocker(
    *,
    task: dict[str, Any],
    session_state: dict[str, Any],
    candidate_session_state: dict[str, Any],
    prepared_bridge_request: dict[str, Any],
    current_blockers: list[dict[str, Any]],
) -> bool:
    resource_jid = str(task.get("resource_jid") or "").strip()
    part_name = str(task.get("part_name") or "").strip()
    if not resource_jid or not part_name:
        return False
    resources_by_jid, parts_by_name = projected_outline_validation_context(
        session_state=session_state,
        prepared_bridge_request=prepared_bridge_request,
    )
    resource_row = dict(resources_by_jid.get(resource_jid) or {})
    if str(resource_row.get("held_part") or "").strip() != part_name:
        return False
    blocker_parts = _current_safety_blocker_parts(
        current_blockers=current_blockers,
        parts_by_name=parts_by_name,
        prepared_bridge_request=prepared_bridge_request,
    )
    if part_name not in blocker_parts:
        return False
    target_ref = str(task.get("target_ref") or "").strip()
    action_target = dict(task.get("action_target") or {})
    target_location = str(action_target.get("target_location") or "").strip()
    end_state = dict(task.get("expected_end_state") or {})
    end_part_location = str(end_state.get("part_location") or "").strip()
    if target_ref or target_location or end_part_location:
        return True
    candidate_resources_by_jid, _ = projected_outline_validation_context(
        session_state=candidate_session_state,
        prepared_bridge_request=prepared_bridge_request,
    )
    candidate_resource = dict(candidate_resources_by_jid.get(resource_jid) or {})
    return str(candidate_resource.get("current_location") or "").strip() != str(resource_row.get("current_location") or "").strip()


def _candidate_effect_match_key(task: dict[str, Any]) -> dict[str, Any]:
    return {
        "expected_end_state": deepcopy(task.get("expected_end_state") or {}),
        "action_target": deepcopy(task.get("action_target") or {}),
    }


def _candidate_pruned_task_match_key(task: dict[str, Any]) -> str:
    return json.dumps(
        {
            "resource_jid": str(task.get("resource_jid") or "").strip(),
            "part_name": str(task.get("part_name") or "").strip(),
            "target_ref": str(task.get("target_ref") or "").strip(),
            "effect": _candidate_effect_match_key(task),
        },
        sort_keys=True,
        ensure_ascii=True,
    )


def _is_durable_candidate_finding(finding: dict[str, Any]) -> bool:
    return str(finding.get("constraint_code") or "").strip().lower() in _DES_RECOVERY_DURABLE_PRUNED_CONSTRAINT_CODES


def _current_candidate_state_signature(
    *,
    task: dict[str, Any],
    finding: dict[str, Any],
    session_state: dict[str, Any],
    prepared_bridge_request: dict[str, Any],
) -> str:
    resources_by_jid, parts_by_name = projected_outline_validation_context(
        session_state=session_state,
        prepared_bridge_request=prepared_bridge_request,
    )
    resource_jid = str(task.get("resource_jid") or finding.get("resource_jid") or "").strip()
    part_name = str(task.get("part_name") or finding.get("part_name") or "").strip()
    target_ref = str(task.get("target_ref") or "").strip()
    constraint_code = str(finding.get("constraint_code") or "").strip().lower()
    resource_row = dict(resources_by_jid.get(resource_jid) or {})
    part_row = dict(parts_by_name.get(part_name) or {})
    active_blockers = _active_candidate_recovery_blockers(
        session_state=session_state,
        prepared_bridge_request=prepared_bridge_request,
    )
    payload: dict[str, Any] = {
        "constraint_code": constraint_code,
        "resource_jid": resource_jid,
        "part_name": part_name,
        "target_ref": target_ref,
    }
    if constraint_code == "workspace_unreachable":
        payload.update({
            "part_current_location": str(part_row.get("current_location") or "").strip(),
            "part_current_holder_resource_jid": str(part_row.get("current_holder_resource_jid") or "").strip(),
            "has_observed_pose": bool(dict(part_row.get("observed_pose") or {})),
            "checked_pose": deepcopy(
                finding.get("checked_pose")
                or finding.get("pose")
                or part_row.get("observed_pose")
                or {}
            ),
            "workspace_bounds": deepcopy(
                finding.get("workspace_bounds")
                or resource_row.get("workspace_bounds")
                or {}
            ),
        })
    elif constraint_code in {
        "holder_conflict",
        "required_part_not_held",
        "part_relocation_without_carrier",
    }:
        payload.update({
            "resource_held_part": str(resource_row.get("held_part") or "").strip(),
            "part_current_holder_resource_jid": str(part_row.get("current_holder_resource_jid") or "").strip(),
            "part_current_location": str(part_row.get("current_location") or "").strip(),
            "part_current_state": str(part_row.get("current_state") or "").strip(),
        })
    elif constraint_code == "source_reference_unavailable":
        payload.update({
            "part_current_holder_resource_jid": str(part_row.get("current_holder_resource_jid") or "").strip(),
            "part_current_location": str(part_row.get("current_location") or "").strip(),
            "has_observed_pose": bool(dict(part_row.get("observed_pose") or {})),
        })
    elif constraint_code in {
        "blocker_open",
        "dependency_unsatisfied",
        "order_violation",
        "safety_rule_violation",
    }:
        payload.update({
            "condition_ids": list(finding.get("condition_ids") or []),
            "rule_id": str(finding.get("rule_id") or "").strip(),
            "active_blockers": [
                str(row.get("summary") or "").strip()
                for row in active_blockers
                if isinstance(row, dict) and str(row.get("summary") or "").strip()
            ],
            "des_safety_dfa_vector": list(session_state.get("des_safety_dfa_vector") or ()),
        })
    else:
        payload.update({
            "resource_held_part": str(resource_row.get("held_part") or "").strip(),
            "part_current_holder_resource_jid": str(part_row.get("current_holder_resource_jid") or "").strip(),
            "part_current_location": str(part_row.get("current_location") or "").strip(),
        })
    return json.dumps(payload, sort_keys=True, default=str, ensure_ascii=True)


def _build_durable_pruned_action_row(
    *,
    task: dict[str, Any],
    finding: dict[str, Any],
    activation_signature: str,
    repeat_count: int,
) -> dict[str, Any]:
    return {
        "resource_jid": str(task.get("resource_jid") or "").strip(),
        "part_name": str(task.get("part_name") or "").strip(),
        "target_ref": str(task.get("target_ref") or "").strip(),
        "task": deepcopy(task),
        "action": deepcopy(task),
        "guard": deepcopy(finding),
        "reason": str(finding.get("reason") or "").strip(),
        "activation_signature": activation_signature,
        "repeat_count": max(1, int(repeat_count or 1)),
    }


def active_des_recovery_pruned_actions(
    session_state: dict[str, Any],
    prepared_bridge_request: dict[str, Any],
) -> list[dict[str, Any]]:
    active_rows: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str, str]] = set()
    for raw_row in (session_state.get("pruned_actions") or []):
        if not isinstance(raw_row, dict):
            continue
        task = dict(raw_row.get("task") or raw_row.get("action") or {})
        guard = dict(raw_row.get("guard") or {})
        activation_signature = str(raw_row.get("activation_signature") or "").strip()
        if not task or not guard or not activation_signature:
            continue
        if not _is_durable_candidate_finding(guard):
            continue
        current_signature = _current_candidate_state_signature(
            task=task,
            finding=guard,
            session_state=session_state,
            prepared_bridge_request=prepared_bridge_request,
        )
        if current_signature != activation_signature:
            continue
        dedupe_key = (
            _candidate_pruned_task_match_key(task),
            str(guard.get("constraint_code") or "").strip().lower(),
            activation_signature,
        )
        if dedupe_key in seen_keys:
            continue
        seen_keys.add(dedupe_key)
        active_rows.append(deepcopy(raw_row))
    return active_rows


def matching_active_des_recovery_pruned_action(
    *,
    task: dict[str, Any],
    session_state: dict[str, Any],
    prepared_bridge_request: dict[str, Any],
) -> dict[str, Any] | None:
    task_key = _candidate_pruned_task_match_key(task)
    for row in active_des_recovery_pruned_actions(session_state, prepared_bridge_request):
        row_task = dict(row.get("task") or row.get("action") or {})
        if _candidate_pruned_task_match_key(row_task) == task_key:
            return deepcopy(row)
    return None


def retarget_candidate_finding_to_task(
    finding: dict[str, Any],
    task: dict[str, Any],
) -> dict[str, Any]:
    retargeted = deepcopy(finding or {})
    outline_id = str(task.get("outline_id") or "").strip()
    resource_jid = str(task.get("resource_jid") or "").strip()
    part_name = str(task.get("part_name") or "").strip()
    if outline_id:
        retargeted["task_id"] = outline_id
    if resource_jid:
        retargeted["resource_jid"] = resource_jid
    if part_name:
        retargeted["part_name"] = part_name
    return retargeted


def promote_durable_candidate_rejections(
    *,
    session_state: dict[str, Any],
    prepared_bridge_request: dict[str, Any],
    candidate_evaluations: list[dict[str, Any]],
) -> None:
    prune_history = dict(session_state.get("candidate_prune_history") or {})
    active_rows = active_des_recovery_pruned_actions(session_state, prepared_bridge_request)
    seen_active = {
        (
            _candidate_pruned_task_match_key(dict(row.get("task") or row.get("action") or {})),
            str(dict(row.get("guard") or {}).get("constraint_code") or "").strip().lower(),
            str(row.get("activation_signature") or "").strip(),
        )
        for row in active_rows
        if isinstance(row, dict)
    }
    for evaluation in candidate_evaluations:
        if not isinstance(evaluation, dict) or bool(evaluation.get("valid")):
            continue
        task = dict(evaluation.get("task") or {})
        findings = [
            dict(item)
            for item in (evaluation.get("validation_findings") or [])
            if isinstance(item, dict)
        ]
        durable_finding = next((item for item in findings if _is_durable_candidate_finding(item)), None)
        if durable_finding is None:
            continue
        activation_signature = _current_candidate_state_signature(
            task=task,
            finding=durable_finding,
            session_state=session_state,
            prepared_bridge_request=prepared_bridge_request,
        )
        history_key = json.dumps(
            {
                "task": _candidate_pruned_task_match_key(task),
                "constraint_code": str(durable_finding.get("constraint_code") or "").strip().lower(),
                "activation_signature": activation_signature,
            },
            sort_keys=True,
            ensure_ascii=True,
        )
        repeat_count = int(prune_history.get(history_key) or 0) + 1
        prune_history[history_key] = repeat_count
        if repeat_count < _CANDIDATE_PRUNE_REPEAT_THRESHOLD:
            continue
        dedupe_key = (
            _candidate_pruned_task_match_key(task),
            str(durable_finding.get("constraint_code") or "").strip().lower(),
            activation_signature,
        )
        if dedupe_key in seen_active:
            continue
        seen_active.add(dedupe_key)
        active_rows.append(
            _build_durable_pruned_action_row(
                task=task,
                finding=durable_finding,
                activation_signature=activation_signature,
                repeat_count=repeat_count,
            )
        )
    session_state["candidate_prune_history"] = prune_history
    session_state["pruned_actions"] = active_rows


def candidate_progress_score(
    *,
    task: dict[str, Any],
    session_state: dict[str, Any],
    prepared_bridge_request: dict[str, Any],
) -> tuple[int, dict[str, int]]:
    current_blockers = _active_candidate_recovery_blockers(
        session_state=session_state,
        prepared_bridge_request=prepared_bridge_request,
    )
    candidate_session_state = _candidate_session_after_task(
        session_state=session_state,
        task=task,
    )
    remaining_blockers = _active_candidate_recovery_blockers(
        session_state=candidate_session_state,
        prepared_bridge_request=prepared_bridge_request,
    )
    current_keys = {_candidate_recovery_blocker_key(row) for row in current_blockers if isinstance(row, dict)}
    remaining_keys = {_candidate_recovery_blocker_key(row) for row in remaining_blockers if isinstance(row, dict)}
    resolved_blockers = len(current_keys - remaining_keys)
    blocker_part_acquired = 0
    if resolved_blockers == 0 and _part_acquisition_counts_as_blocker_progress(
        task=task,
        session_state=session_state,
        candidate_session_state=candidate_session_state,
        prepared_bridge_request=prepared_bridge_request,
        current_blockers=current_blockers,
    ):
        blocker_part_acquired = 1
    resource_freed_for_blocker = 0
    if resolved_blockers == 0 and blocker_part_acquired == 0 and _part_release_frees_resource_for_blocker(
        task=task,
        session_state=session_state,
        candidate_session_state=candidate_session_state,
        prepared_bridge_request=prepared_bridge_request,
        current_blockers=current_blockers,
    ):
        resource_freed_for_blocker = 1
    preparatory_transit = 0
    if (
        resolved_blockers == 0
        and blocker_part_acquired == 0
        and resource_freed_for_blocker == 0
        and _preparatory_transit_toward_blocker(
            task=task,
            session_state=session_state,
            candidate_session_state=candidate_session_state,
            prepared_bridge_request=prepared_bridge_request,
            current_blockers=current_blockers,
        )
    ):
        preparatory_transit = 1
    remaining_blocked_issues = len(remaining_keys)
    secondary_progress = blocker_part_acquired + resource_freed_for_blocker + preparatory_transit
    return (
        resolved_blockers + secondary_progress,
        {
            "resolved_direct_blockers": resolved_blockers,
            "blocker_part_acquired": blocker_part_acquired,
            "freed_resource_for_blocker": resource_freed_for_blocker,
            "preparatory_transit": preparatory_transit,
            "resolved_continuation_conditions": resolved_blockers,
            "remaining_continuation_conditions": remaining_blocked_issues,
            "remaining_blocked_issues": remaining_blocked_issues,
        },
    )


def active_candidate_recovery_blockers(
    *,
    session_state: dict[str, Any],
    prepared_bridge_request: dict[str, Any],
) -> list[dict[str, Any]]:
    return _active_candidate_recovery_blockers(
        session_state=session_state,
        prepared_bridge_request=prepared_bridge_request,
    )


def compute_enabled_candidate_bound(
    session_state: dict[str, Any],
) -> int:
    """Return the configured candidate budget for candidate selection."""
    candidate_bound = max(
        1,
        int(session_state.get("candidate_bound") or _DEFAULT_CANDIDATE_BOUND),
    )
    candidate_bound_cap = max(
        1,
        int(session_state.get("candidate_bound_cap") or _DEFAULT_CANDIDATE_BOUND_CAP),
    )
    return min(candidate_bound, candidate_bound_cap)


def candidate_feedback_rows(
    candidate_evaluations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in candidate_evaluations:
        if not isinstance(row, dict) or bool(row.get("valid")):
            continue
        findings = [
            deepcopy(item)
            for item in (row.get("validation_findings") or [])
            if isinstance(item, dict)
        ]
        if not findings:
            continue
        feedback_row = {
            "candidate_index": int(row.get("candidate_index") or 0),
            "task": deepcopy(dict(row.get("task") or {})),
            "validation_findings": findings,
        }
        if isinstance(row.get("surface_task"), dict):
            feedback_row["surface_task"] = deepcopy(dict(row.get("surface_task") or {}))
        if isinstance(row.get("repaired_task"), dict):
            feedback_row["repaired_task"] = deepcopy(dict(row.get("repaired_task") or {}))
        if str(row.get("repair_applied") or "").strip():
            feedback_row["repair_applied"] = str(row.get("repair_applied") or "").strip()
        rows.append(feedback_row)
    return rows


def finding_event_status_for_logging(finding: dict[str, Any]) -> str:
    constraint_code = str(finding.get("constraint_code") or "").strip().lower()
    constraint_family = str(finding.get("constraint_family") or "").strip().lower()
    if constraint_code in {
        "blocker_open",
        "dependency_unsatisfied",
        "order_violation",
        "invalid_dependency_reference",
        "claimed_condition_not_currently_unmet",
        "safety_rule_violation",
    } or constraint_family in {"safety", "continuation"}:
        return "blocked_by_supervisor"
    if constraint_code in {"no_state_change", "no_blocker_reduction"} or constraint_family == "candidate_selection":
        return "nonprogressing"
    return "disabled"


def remaining_blocked_issue_counts(
    *,
    session_state: dict[str, Any],
    prepared_bridge_request: dict[str, Any],
) -> tuple[int, int]:
    remaining_blockers = len(
        _active_candidate_recovery_blockers(
            session_state=session_state,
            prepared_bridge_request=prepared_bridge_request,
        )
    )
    return 0, remaining_blockers


def next_recovery_sequence_index(session_state: dict[str, Any]) -> int:
    return len(list(session_state.get("accepted_outline_prefix") or [])) + 1


def _candidate_outline_id(*, sequence_index: int, candidate_index: int) -> str:
    return f"RECOVERY_SEQ{sequence_index}_{candidate_index + 1}"


def _committed_outline_id(*, sequence_index: int) -> str:
    return f"RECOVERY_SEQ{sequence_index}"


def _candidate_schema_finding(
    *,
    task: dict[str, Any],
    reason: str,
    evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "task_id": str(task.get("outline_id") or "").strip(),
        "resource_jid": str(task.get("resource_jid") or "").strip() or None,
        "part_name": str(task.get("part_name") or "").strip() or None,
        "constraint_owner": "binding",
        "constraint_family": "binding",
        "constraint_code": "candidate_schema_violation",
        "reason": reason,
        "evidence": deepcopy(evidence or {}),
    }


def _candidate_named_pose_tokens(resource_row: dict[str, Any]) -> set[str]:
    tokens: set[str] = set()
    raw_named_poses = resource_row.get("named_poses")
    if isinstance(raw_named_poses, dict):
        tokens.update(str(token).strip() for token in raw_named_poses.keys() if str(token).strip())
    else:
        tokens.update(str(token).strip() for token in (raw_named_poses or []) if str(token).strip())
    tokens.update(
        str(token).strip()
        for token in (resource_row.get("available_named_poses") or [])
        if str(token).strip()
    )
    return tokens


def _part_current_location_token(part_row: dict[str, Any]) -> str:
    return str(part_row.get("current_location") or part_row.get("location") or "").strip()


def _part_current_holder_token(part_row: dict[str, Any]) -> str:
    return str(part_row.get("current_holder_resource_jid") or part_row.get("holder_resource_jid") or "").strip()


def _part_current_state_token(part_row: dict[str, Any]) -> str:
    return str(part_row.get("current_state") or part_row.get("state") or "").strip()


def _resource_current_state_token(resource_row: dict[str, Any]) -> str:
    return str(resource_row.get("current_state") or resource_row.get("state") or "").strip()


def _candidate_action_name_from_task(task: dict[str, Any]) -> str:
    event_name = str(task.get("event_name") or "").strip()
    if event_name:
        return event_name
    legacy_action_name = str(task.get("action_name") or "").strip()
    return legacy_action_name if legacy_action_name else ""


def _candidate_target_ref_from_surface_task(task: dict[str, Any]) -> str:
    if str(task.get("target_ref") or "").strip():
        return str(task.get("target_ref") or "").strip()
    action_target = dict(task.get("action_target") or {})
    return str(action_target.get("target_location") or action_target.get("named_pose") or "").strip()


def normalize_candidate_task(
    *,
    task: dict[str, Any],
    sequence_index: int,
    candidate_index: int,
) -> dict[str, Any]:
    raw_task = deepcopy(task or {})
    original_outline_id = str(raw_task.get("outline_id") or "").strip()
    normalized: dict[str, Any] = {
        "resource_jid": str(raw_task.get("resource_jid") or "").strip(),
    }
    action_name = _candidate_action_name_from_task(raw_task)
    description = str(raw_task.get("description") or "").strip()
    part_name = str(raw_task.get("part_name") or "").strip()
    target_ref = _candidate_target_ref_from_surface_task(raw_task)
    if action_name:
        normalized["event_name"] = action_name
        normalized["action_name"] = action_name
    if description:
        normalized["description"] = description
    if part_name:
        normalized["part_name"] = part_name
    if target_ref:
        normalized["target_ref"] = target_ref
    if original_outline_id:
        normalized["llm_outline_id"] = original_outline_id
    normalized["outline_id"] = _candidate_outline_id(
        sequence_index=sequence_index,
        candidate_index=candidate_index,
    )
    return normalized


def derive_candidate_outline_task(
    *,
    candidate_task: dict[str, Any],
    session_state: dict[str, Any],
    prepared_bridge_request: dict[str, Any],
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    resources_by_jid, parts_by_name = projected_outline_validation_context(
        session_state=session_state,
        prepared_bridge_request=prepared_bridge_request,
    )
    resource_jid = str(candidate_task.get("resource_jid") or "").strip()
    action_name = str(candidate_task.get("event_name") or candidate_task.get("action_name") or "").strip()
    part_name = str(candidate_task.get("part_name") or "").strip()
    target_ref = str(candidate_task.get("target_ref") or "").strip()
    description = str(candidate_task.get("description") or "").strip()
    resource_row = dict(resources_by_jid.get(resource_jid) or {})
    part_row = dict(parts_by_name.get(part_name) or {}) if part_name else {}
    if not resource_jid:
        return None, [_candidate_schema_finding(task=candidate_task, reason="candidate must include resource_jid", evidence={"field": "resource_jid"})]
    if not action_name:
        return None, [_candidate_schema_finding(task=candidate_task, reason="candidate must include event_name naming the DES event (legacy action_name accepted)", evidence={"field": "event_name"})]
    if not description:
        return None, [_candidate_schema_finding(task=candidate_task, reason="candidate must include description", evidence={"field": "description"})]
    if part_name and part_name not in parts_by_name:
        return None, [_candidate_schema_finding(task=candidate_task, reason=f"candidate references unknown part '{part_name}'", evidence={"field": "part_name", "token": part_name})]
    start_state: dict[str, Any] = {}
    current_resource_state = _resource_current_state_token(resource_row)
    if current_resource_state or "current_state" in resource_row or "state" in resource_row:
        start_state["resource_state"] = current_resource_state or None
    if "held_part" in resource_row:
        start_state["held_part"] = resource_row.get("held_part") or None
    if part_name:
        current_part_state = _part_current_state_token(part_row)
        current_part_location = _part_current_location_token(part_row)
        current_part_holder = _part_current_holder_token(part_row)
        if current_part_state or "current_state" in part_row or "state" in part_row:
            start_state["part_state"] = current_part_state or None
        if current_part_location:
            start_state["part_location"] = current_part_location
        elif isinstance(part_row.get("observed_pose"), dict) and "x" in dict(part_row.get("observed_pose") or {}):
            start_state["part_location"] = "observed_pose"
        else:
            start_state["part_location"] = None
        if current_part_holder or "current_holder_resource_jid" in part_row or "holder_resource_jid" in part_row:
            start_state["part_holder_resource_jid"] = current_part_holder or None
    action_target: dict[str, Any] = {}
    end_state: dict[str, Any] = {}
    if part_name and not target_ref:
        current_part_location = _part_current_location_token(part_row)
        if current_part_location:
            action_target["source_location"] = current_part_location
        elif isinstance(part_row.get("observed_pose"), dict) and "x" in dict(part_row.get("observed_pose") or {}):
            action_target["source_location"] = "observed_pose"
        end_state.update({
            "resource_state": "picked",
            "held_part": part_name,
            "part_location": f"{resource_jid}_gripper",
            "part_holder_resource_jid": resource_jid,
        })
    elif part_name and target_ref:
        action_target["target_location"] = target_ref
        end_state.update({
            "resource_state": "idle",
            "held_part": None,
            "part_location": target_ref,
            "part_holder_resource_jid": None,
        })
    else:
        end_state["resource_state"] = "idle"
        if target_ref:
            if target_ref in _candidate_named_pose_tokens(resource_row):
                action_target["named_pose"] = target_ref
            else:
                action_target["target_location"] = target_ref
    normalized_task: dict[str, Any] = {
        "outline_id": str(candidate_task.get("outline_id") or "").strip(),
        "resource_jid": resource_jid,
        "event_name": action_name,
        "action_name": action_name,
        "description": description,
        "expected_start_state": start_state,
        "expected_end_state": end_state,
    }
    normalized_task["action_type"] = (
        "acquire_part" if part_name and not target_ref else
        "release_part" if part_name and target_ref else
        "recover_resource"
    )
    if part_name:
        normalized_task["part_name"] = part_name
    if target_ref:
        normalized_task["target_ref"] = target_ref
    if action_target:
        normalized_task["action_target"] = action_target
    if str(candidate_task.get("llm_outline_id") or "").strip():
        normalized_task["llm_outline_id"] = str(candidate_task.get("llm_outline_id") or "").strip()
    return normalized_task, []


def commit_selected_candidate_task(
    *,
    task: dict[str, Any],
    sequence_index: int,
) -> dict[str, Any]:
    committed = deepcopy(task or {})
    committed["candidate_outline_id"] = str(committed.get("outline_id") or "").strip()
    committed["outline_id"] = _committed_outline_id(sequence_index=sequence_index)
    _normalize_place_verb_in_action_name(committed)
    return committed


def apply_task_effects_to_symbolic_state(
    task: dict[str, Any],
    session_state: dict[str, Any],
) -> None:
    resource_jid = str(task.get("resource_jid") or "").strip()
    part_name = str(task.get("part_name") or "").strip()
    target_ref = str(task.get("target_ref") or "").strip()
    end_state = dict(task.get("expected_end_state") or {})
    ends_with_part_held = _task_ends_with_part_held_by_resource(task)
    ends_with_part_clear = _task_ends_with_part_clear_of_resource(task)
    symbolic_resources = dict(session_state.get("symbolic_resources") or {})
    symbolic_parts = dict(session_state.get("symbolic_parts") or {})
    if resource_jid:
        res = symbolic_resources.setdefault(resource_jid, {"resource_jid": resource_jid})
        if "resource_state" in end_state:
            res["current_state"] = end_state["resource_state"]
        if "held_part" in end_state:
            res["held_part"] = end_state["held_part"] or None
            res["gripper_state"] = "closed" if end_state["held_part"] else "open"
    if part_name:
        part = symbolic_parts.setdefault(part_name, {"part_name": part_name})
        if "part_state" in end_state:
            part["current_state"] = end_state["part_state"]
        elif ends_with_part_held:
            part["current_state"] = "in_gripper"
        elif ends_with_part_clear:
            goal_location = str(part.get("goal_location") or "").strip()
            resolved_target = str(end_state.get("part_location") or target_ref or "").strip()
            part["current_state"] = "placed" if goal_location and resolved_target == goal_location else "misplaced"
        if "part_location" in end_state:
            part["current_location"] = end_state["part_location"] or None
        elif ends_with_part_held and resource_jid:
            part["current_location"] = f"{resource_jid}_gripper"
        elif ends_with_part_clear:
            part["current_location"] = target_ref or None
        if "part_holder_resource_jid" in end_state:
            part["current_holder_resource_jid"] = end_state["part_holder_resource_jid"] or None
        elif ends_with_part_held:
            part["current_holder_resource_jid"] = resource_jid or None
        elif ends_with_part_clear:
            part["current_holder_resource_jid"] = None
        elif "held_part" in end_state:
            part["current_holder_resource_jid"] = resource_jid if end_state["held_part"] == part_name else None
        if "held_part" in end_state and end_state["held_part"] == part_name and "part_location" not in end_state:
            part["current_location"] = f"{resource_jid}_gripper"
        elif "held_part" in end_state and end_state["held_part"] in (None, "") and "part_holder_resource_jid" not in end_state:
            part["current_holder_resource_jid"] = None
    session_state["symbolic_resources"] = symbolic_resources
    session_state["symbolic_parts"] = symbolic_parts


def parsed_response_object(
    parsed_response: dict[str, Any],
    *,
    primary_key: str,
    legacy_key: str,
) -> dict[str, Any]:
    raw = parsed_response.get(primary_key)
    if not isinstance(raw, dict):
        raw = parsed_response.get(legacy_key)
    return dict(raw or {}) if isinstance(raw, dict) else {}


def parsed_response_rows(
    parsed_response: dict[str, Any],
    *,
    primary_key: str,
    legacy_key: str,
) -> list[dict[str, Any]]:
    raw = parsed_response.get(primary_key)
    if not isinstance(raw, list):
        raw = parsed_response.get(legacy_key)
    return [dict(row) for row in (raw or []) if isinstance(row, dict)]


def parsed_response_rows_any(
    parsed_response: dict[str, Any],
    *,
    keys: tuple[str, ...],
) -> list[dict[str, Any]]:
    for key in keys:
        raw = parsed_response.get(key)
        if isinstance(raw, list):
            return [dict(row) for row in raw if isinstance(row, dict)]
    return []


__all__ = [
    "_DEFAULT_CANDIDATE_BOUND",
    "active_candidate_recovery_blockers",
    "active_des_recovery_pruned_actions",
    "advance_des_safety_state",
    "apply_task_effects_to_symbolic_state",
    "candidate_feedback_rows",
    "candidate_progress_score",
    "compute_enabled_candidate_bound",
    "commit_selected_candidate_task",
    "derive_candidate_outline_task",
    "extend_plant_with_event",
    "finding_event_status_for_logging",
    "matching_active_des_recovery_pruned_action",
    "merge_outline_validation_findings",
    "next_recovery_sequence_index",
    "no_blocker_reduction_finding",
    "normalize_candidate_task",
    "parsed_response_object",
    "parsed_response_rows",
    "parsed_response_rows_any",
    "promote_durable_candidate_rejections",
    "prune_resolved_outline_validation_findings",
    "remaining_blocked_issue_counts",
    "retarget_candidate_finding_to_task",
    "symbolic_state_fingerprint",
    "sync_des_recovery_aliases",
    "validate_single_outline_task",
]
