"""Multi-turn prompt builders and response schemas for the active v4 bridge."""

from __future__ import annotations

from copy import deepcopy
import json
from typing import Any

from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.prompts.shared import (
    build_shared_fact_sections,
    json_block,
    prompt_part_facts,
)


_PHASE_TITLES = {
    "grounding": "Grounding Assessment",
    "outline": "Recovery Outline",
    "primitive_generation": "Primitive Generation",
    "finalize": "Finalize Proposal",
}


def _compact_session_state(session_state: dict[str, Any]) -> dict[str, Any]:
    state = dict(session_state or {})
    return {
        "session_id": str(state.get("session_id") or "").strip(),
        "current_phase": str(state.get("current_phase") or "").strip(),
        "turn_index": int(state.get("turn_index") or 0),
        "max_turns": int(state.get("max_turns") or 0),
        "observation_count": int(state.get("observation_count") or 0),
        "observation_fact_count": len(dict(state.get("observation_fact_ledger") or {})),
        "max_observations": int(state.get("max_observations") or 0),
        "max_observe_batch": int(state.get("max_observe_batch") or 0),
        "pruned_action_count": len(list(state.get("pruned_actions") or [])),
        "stop_after_phase": str(state.get("stop_after_phase") or "").strip() or None,
    }


def _compact_outline_tasks(outline_tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compact_tasks: list[dict[str, Any]] = []
    for raw_task in (outline_tasks or []):
        if not isinstance(raw_task, dict):
            continue
        outline_id = str(raw_task.get("outline_id") or "").strip()
        resource_jid = str(raw_task.get("resource_jid") or "").strip()
        if not outline_id and not resource_jid:
            continue
        row: dict[str, Any] = {
            "outline_id": outline_id or None,
            "resource_jid": resource_jid or None,
        }
        part_name = str(raw_task.get("part_name") or "").strip()
        if part_name:
            row["part_name"] = part_name
        depends_on = [
            str(item).strip()
            for item in (raw_task.get("depends_on") or [])
            if str(item).strip()
        ]
        if depends_on:
            row["depends_on"] = depends_on
        compact_tasks.append(row)
    return compact_tasks


def _compact_turn_history(session_state: dict[str, Any]) -> list[dict[str, Any]]:
    history: list[dict[str, Any]] = []
    for raw_turn in (session_state.get("turns") or []):
        if not isinstance(raw_turn, dict):
            continue
        row: dict[str, Any] = {
            "turn_index": int(raw_turn.get("turn_index") or 0),
            "phase": str(raw_turn.get("phase") or "").strip(),
        }
        raw_response = deepcopy(raw_turn.get("raw_response") or {})
        if str(raw_turn.get("phase") or "").strip().lower() == "grounding":
            if not isinstance(raw_response, dict):
                raw_response = {}
            else:
                if not isinstance(raw_response.get("blocking_reasons"), list):
                    legacy_blocking = raw_response.get("blocking_summary")
                    if isinstance(legacy_blocking, list):
                        raw_response["blocking_reasons"] = deepcopy(legacy_blocking)
                raw_response.pop("blocking_summary", None)
                sanitized_requests: list[dict[str, Any]] = []
                for request in (raw_response.get("observe_requests") or []):
                    if not isinstance(request, dict):
                        continue
                    request_row = deepcopy(request)
                    request_row.pop("resource_jid", None)
                    sanitized_requests.append(request_row)
                if sanitized_requests:
                    raw_response["observe_requests"] = sanitized_requests
        elif str(raw_turn.get("phase") or "").strip().lower() == "outline":
            if not isinstance(raw_response, dict):
                raw_response = {}
            else:
                raw_response = {}
        if raw_response not in ({}, [], "", None):
            row["model_response"] = raw_response
        observation_results = [
            {
                key: deepcopy(value)
                for key, value in dict(observation_result or {}).items()
                if key != "resource_jid"
            }
            for observation_result in (raw_turn.get("observation_results") or [])
            if isinstance(observation_result, dict)
        ]
        if observation_results:
            row["observation_results"] = observation_results
        error = str(raw_turn.get("error") or "").strip()
        if error:
            row["runtime_error"] = error
        runtime_decision = str(raw_turn.get("decision") or "").strip()
        model_decision = ""
        if isinstance(raw_response, dict):
            model_decision = str(raw_response.get("decision") or "").strip()
        if runtime_decision and runtime_decision != model_decision:
            row["runtime_decision"] = runtime_decision
        outline_validation = dict(raw_turn.get("outline_validation") or {})
        validation_status = str(outline_validation.get("status") or "").strip()
        if validation_status:
            row["outline_validation"] = {"status": validation_status}
            validation_reason = str(outline_validation.get("reason") or "").strip()
            if validation_reason:
                row["outline_validation"]["reason"] = validation_reason
        outline_revision_coverage = dict(raw_turn.get("outline_revision_coverage") or {})
        coverage_status = str(outline_revision_coverage.get("status") or "").strip()
        if coverage_status:
            row["outline_revision_coverage"] = {"status": coverage_status}
            coverage_violations = [
                str(item).strip()
                for item in (outline_revision_coverage.get("violations") or [])
                if str(item).strip()
            ]
            if coverage_violations:
                row["outline_revision_coverage"]["violations"] = coverage_violations
        history.append(row)
    return history


def _previous_outline_attempt(session_state: dict[str, Any]) -> dict[str, Any]:
    latest_outline_turn: dict[str, Any] = {}
    for raw_turn in (session_state.get("turns") or []):
        if not isinstance(raw_turn, dict):
            continue
        if str(raw_turn.get("phase") or "").strip().lower() != "outline":
            continue
        latest_outline_turn = raw_turn
    if not latest_outline_turn:
        return {}

    previous_attempt: dict[str, Any] = {}
    runtime_decision = str(latest_outline_turn.get("decision") or "").strip()
    if runtime_decision:
        previous_attempt["runtime_decision"] = runtime_decision
    outline_validation = dict(latest_outline_turn.get("outline_validation") or {})
    if outline_validation:
        previous_attempt["outline_validation"] = deepcopy(outline_validation)
    validation_violations = [
        str(item).strip()
        for item in (latest_outline_turn.get("validation_violations") or [])
        if str(item).strip()
    ]
    if validation_violations:
        previous_attempt["validation_violations"] = validation_violations
    return previous_attempt


def _outline_prompt_resource_facts(resources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rendered_rows: list[dict[str, Any]] = []
    for raw_row in (resources or []):
        if not isinstance(raw_row, dict):
            continue
        row = deepcopy(raw_row)
        row.pop("availability", None)
        rendered_rows.append(row)
    return rendered_rows


def _outline_prompt_part_facts(part_facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rendered_rows = prompt_part_facts(part_facts)
    for row in rendered_rows:
        if not isinstance(row, dict):
            continue
        row.pop("nominal_requirement_resource_jid", None)
    return rendered_rows


def _outline_prompt_loaded_safety_rules(
    safety_rules: list[dict[str, Any]],
    obligation_targets: list[dict[str, Any]],
) -> dict[str, Any]:
    compact_rules: list[dict[str, Any]] = []
    for raw_rule in (safety_rules or []):
        if not isinstance(raw_rule, dict):
            continue
        row: dict[str, Any] = {}
        rule_id = str(raw_rule.get("rule_id") or "").strip()
        if rule_id:
            row["rule_id"] = rule_id
        constraint_type = str(raw_rule.get("constraint_type") or "").strip()
        if constraint_type:
            row["constraint_type"] = constraint_type
        if row:
            compact_rules.append(row)
    return {
        "obligation_targets": deepcopy(obligation_targets or []),
        "loaded_safety_rules": compact_rules,
    }


def _outline_prompt_relevant_requirements(
    requirements: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    compact_requirements: list[dict[str, Any]] = []
    for raw_requirement in (requirements or []):
        if not isinstance(raw_requirement, dict):
            continue
        row: dict[str, Any] = {}
        requirement_id = str(raw_requirement.get("requirement_id") or "").strip()
        if requirement_id:
            row["requirement_id"] = requirement_id
        status = str(raw_requirement.get("status") or "").strip()
        if status:
            row["status"] = status
        product = str(raw_requirement.get("product") or "").strip()
        if product:
            row["product"] = product
        pending_task_ids = [
            str(item).strip()
            for item in (raw_requirement.get("pending_task_ids") or [])
            if str(item).strip()
        ]
        if pending_task_ids:
            row["pending_task_ids"] = pending_task_ids
        if row:
            compact_requirements.append(row)
    return compact_requirements


def _outline_prompt_llm_input(llm_input: dict[str, Any]) -> dict[str, Any]:
    payload = deepcopy(llm_input or {})
    payload.pop("bridge_safety_context", None)
    observed_runtime_state = dict(payload.get("observed_runtime_state") or {})
    payload["observed_runtime_state"] = {
        **observed_runtime_state,
        "resources": _outline_prompt_resource_facts(observed_runtime_state.get("resources") or []),
    }
    payload["part_facts"] = _outline_prompt_part_facts(payload.get("part_facts") or [])
    payload["loaded_safety_rules"] = _outline_prompt_loaded_safety_rules(
        payload.get("loaded_safety_rules") or [],
        payload.get("obligation_targets") or [],
    )["loaded_safety_rules"]
    payload["relevant_assembly_requirements"] = _outline_prompt_relevant_requirements(
        payload.get("relevant_assembly_requirements") or []
    )
    return payload


def _outline_prompt_grounded_feasibility_facts(
    grounded_feasibility_facts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rendered_rows: list[dict[str, Any]] = []
    for raw_row in (grounded_feasibility_facts or []):
        if not isinstance(raw_row, dict):
            continue
        row = {
            "part_name": deepcopy(raw_row.get("part_name")),
            "pose_source": deepcopy(raw_row.get("pose_source")),
            "pose": deepcopy(raw_row.get("pose")),
            "resource_evidence": [],
        }
        for raw_evidence in (raw_row.get("resource_evidence") or []):
            if not isinstance(raw_evidence, dict):
                continue
            evidence = {
                "resource_jid": deepcopy(raw_evidence.get("resource_jid")),
                "workspace_bounds": deepcopy(raw_evidence.get("workspace_bounds")),
                "workspace_contains_observed_pose": bool(
                    raw_evidence.get("workspace_contains_observed_pose")
                ),
                "workspace_violations": deepcopy(raw_evidence.get("workspace_violations") or []),
            }
            temporary_state_blockers = [
                str(item).strip()
                for item in (raw_evidence.get("readiness_blockers") or [])
                if str(item).strip()
            ]
            if temporary_state_blockers:
                evidence["temporary_state_blockers"] = temporary_state_blockers
            row["resource_evidence"].append(evidence)
        rendered_rows.append(row)
    return rendered_rows


def _outline_prompt_validation_findings(
    outline_validation_findings: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rendered_rows: list[dict[str, Any]] = []
    for raw_row in (outline_validation_findings or []):
        if not isinstance(raw_row, dict):
            continue
        row = deepcopy(raw_row)
        failed_axes = [
            str(item).strip()
            for item in (row.get("failed_axes") or [])
            if str(item).strip()
        ]
        if failed_axes == ["named_pose_not_available"]:
            rendered_rows.append(
                {
                    "task_id": str(row.get("task_id") or "").strip(),
                    "resource_jid": str(row.get("resource_jid") or "").strip(),
                    "part_name": str(row.get("part_name") or "").strip() or None,
                    "pose_source": str(row.get("pose_source") or "").strip(),
                    "finding_kind": "contract_violation",
                    "contract_violation": "named_pose_not_available",
                    "named_pose": deepcopy(row.get("named_pose")),
                }
            )
            continue
        rendered_rows.append(row)
    return rendered_rows


def _outline_prompt_pruned_actions(
    pruned_actions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rendered_rows: list[dict[str, Any]] = []
    for raw_row in (pruned_actions or []):
        if not isinstance(raw_row, dict):
            continue
        action = dict(raw_row.get("action") or {})
        row: dict[str, Any] = {}
        for field_name in (
            "resource_jid",
            "task_kind",
            "part_name",
            "named_pose",
            "source_location",
            "target_location",
            "end_resource_state",
            "end_gripper_state",
            "end_held_part",
        ):
            value = action.get(field_name)
            if value not in (None, "", [], {}):
                row[field_name] = deepcopy(value)
        reason = str(raw_row.get("reason") or "").strip()
        if reason:
            row["reason"] = reason
        if row:
            rendered_rows.append(row)
    return rendered_rows


def _outline_prompt_blocked_nominal_tasks(
    recovery_gap_state: dict[str, Any],
) -> list[dict[str, Any]]:
    rendered_rows: list[dict[str, Any]] = []
    for raw_row in (recovery_gap_state.get("blocked_nominal_tasks") or []):
        if not isinstance(raw_row, dict):
            continue
        row = deepcopy(raw_row)
        row.pop("blocked_by_condition_ids", None)
        row.pop("blocking_condition_ids", None)
        rendered_rows.append(row)
    return rendered_rows


def _outline_prompt_condition_summary(condition_row: dict[str, Any]) -> str:
    kind = str(condition_row.get("kind") or "").strip()
    source_task_id = str(condition_row.get("source_task_id") or "").strip()
    entity = str(condition_row.get("entity") or "").strip()
    expected = str(condition_row.get("expected") or "").strip()
    blocking_reason = str(condition_row.get("blocking_reason") or "").strip()

    if kind == "focused_resource_terminal_state":
        task_text = f" before nominal task '{source_task_id}' can resume" if source_task_id else ""
        if entity and expected:
            return f"{entity} must reach state '{expected}'{task_text}."
        if entity:
            return f"{entity} must recover to its required terminal state{task_text}."
    if kind == "safety_blocked_suffix_task":
        if source_task_id and blocking_reason:
            return f"Nominal task '{source_task_id}' remains blocked: {blocking_reason}"
        if blocking_reason:
            return blocking_reason
        if source_task_id:
            return f"Nominal task '{source_task_id}' remains blocked until its factual blocker is cleared."
    return blocking_reason or ""


def _outline_prompt_unmet_continuation_conditions(
    recovery_gap_state: dict[str, Any],
) -> list[dict[str, Any]]:
    rendered_rows: list[dict[str, Any]] = []
    for raw_row in (recovery_gap_state.get("unmet_continuation_conditions") or []):
        if not isinstance(raw_row, dict):
            continue
        summary = _outline_prompt_condition_summary(raw_row)
        row: dict[str, Any] = {}
        if summary:
            row["summary"] = summary
        kind = str(raw_row.get("kind") or "").strip()
        if kind:
            row["kind"] = kind
        entity = str(raw_row.get("entity") or "").strip()
        if entity:
            row["entity"] = entity
        expected = str(raw_row.get("expected") or "").strip()
        if expected:
            row["expected"] = expected
        source_task_id = str(raw_row.get("source_task_id") or "").strip()
        if source_task_id:
            row["blocked_nominal_task_id"] = source_task_id
        blocking_reason = str(raw_row.get("blocking_reason") or "").strip()
        if blocking_reason:
            row["blocking_reason"] = blocking_reason
        if row:
            rendered_rows.append(row)
    return rendered_rows


def _outline_prompt_repair_contract(
    *,
    outline_validation_findings: list[dict[str, Any]],
    required_addressed_validation_findings: list[dict[str, Any]],
    pruned_actions: list[dict[str, Any]],
    session_state: dict[str, Any],
) -> dict[str, Any] | None:
    repair_contract: dict[str, Any] = {}
    if outline_validation_findings:
        repair_contract["unresolved_findings"] = deepcopy(outline_validation_findings)
    if required_addressed_validation_findings:
        repair_contract["required_addressed_validation_findings"] = deepcopy(
            required_addressed_validation_findings
        )
    if pruned_actions:
        repair_contract["active_pruned_actions"] = deepcopy(pruned_actions)
    accepted_prefix = _compact_outline_tasks(
        list(
            session_state.get("accepted_outline_prefix")
            or session_state.get("accepted_prefix")
            or []
        )
    )
    if accepted_prefix:
        repair_contract["accepted_prefix"] = accepted_prefix
    outline_revision_coverage = deepcopy(session_state.get("outline_revision_coverage") or {})
    if outline_revision_coverage not in ({}, [], "", None):
        repair_contract["revision_feedback"] = outline_revision_coverage
    return repair_contract or None


def _build_outline_fact_sections(
    llm_input: dict[str, Any],
    *,
    recovery_gap_state: dict[str, Any],
    repair_contract: dict[str, Any] | None,
) -> list[str]:
    payload = dict(llm_input or {})
    observed_runtime_state = dict(payload.get("observed_runtime_state") or {})
    safety_section = {
        "obligation_targets": deepcopy(payload.get("obligation_targets") or []),
        "loaded_safety_rules": deepcopy(payload.get("loaded_safety_rules") or []),
    }
    sections: list[str] = [
        "Fault Event",
        json_block(payload.get("fault_event") or {}),
        "",
        "Current Resource Facts",
        json_block(observed_runtime_state.get("resources") or []),
        "",
        "Current Part Facts",
        json_block(prompt_part_facts(payload.get("part_facts") or [])),
        "",
        "Loaded Safety Rules",
        json_block(safety_section),
        "",
        "Relevant Assembly Requirements",
        json_block(payload.get("relevant_assembly_requirements") or []),
        "",
        "Blocked Nominal Tasks",
        json_block(_outline_prompt_blocked_nominal_tasks(recovery_gap_state)),
        "",
        "Unmet Continuation Conditions",
        json_block(_outline_prompt_unmet_continuation_conditions(recovery_gap_state)),
    ]
    if repair_contract not in ({}, [], "", None):
        sections.extend(["", "Repair Contract", json_block(repair_contract)])
    return sections


def _latest_session_delta(session_state: dict[str, Any]) -> dict[str, Any]:
    turns = [
        dict(raw_turn)
        for raw_turn in (session_state.get("turns") or [])
        if isinstance(raw_turn, dict)
    ]
    if not turns:
        return {}
    latest_turn = turns[-1]
    delta: dict[str, Any] = {
        "turn_index": int(latest_turn.get("turn_index") or 0),
        "phase": str(latest_turn.get("phase") or "").strip(),
    }
    observation_results = [
        {
            key: deepcopy(value)
            for key, value in dict(observation_result or {}).items()
            if key not in {"resource_jid", "store_as"}
        }
        for observation_result in (latest_turn.get("observation_results") or [])
        if isinstance(observation_result, dict)
    ]
    if observation_results:
        delta["observation_results"] = observation_results
    runtime_error = str(latest_turn.get("error") or "").strip()
    if runtime_error:
        delta["runtime_error"] = runtime_error
    if len(delta) <= 2:
        return {}
    return delta


def _grounding_observation_fact_key(
    fact_type: Any,
    entity: Any,
    scope: Any = None,
) -> str:
    payload: dict[str, Any] = {
        "fact_type": str(fact_type or "").strip(),
        "entity": str(entity or "").strip(),
    }
    if scope not in (None, "", [], {}):
        payload["scope"] = deepcopy(scope)
    return json.dumps(payload, sort_keys=True, default=str, ensure_ascii=True)


def _build_observation_fulfillment_status(
    session_state: dict[str, Any],
    turn_history: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Cross-reference prior observe_requests with fulfilled observation facts."""
    observation_fact_ledger = {
        str(fact_key).strip(): dict(row)
        for fact_key, row in dict(session_state.get("observation_fact_ledger") or {}).items()
        if str(fact_key).strip() and isinstance(row, dict)
    }
    prior_requests: list[dict[str, Any]] = []
    for turn in turn_history:
        response = turn.get("model_response") or {}
        for req in response.get("observe_requests") or []:
            fact_type = str(req.get("fact_type") or "").strip()
            entity = str(req.get("entity") or "").strip()
            scope = deepcopy(req.get("scope"))
            fact_key = (
                _grounding_observation_fact_key(fact_type, entity, scope)
                if fact_type and entity
                else ""
            )
            fact_row = dict(observation_fact_ledger.get(fact_key) or {}) if fact_key else {}
            is_fulfilled = bool(
                fact_row
                and str(fact_row.get("validity") or "current").strip().lower() != "stale"
            )
            row = {
                "fact_type": fact_type,
                "entity": entity,
                "requested_at_turn": turn.get("turn_index"),
                "status": "fulfilled" if is_fulfilled else "pending",
            }
            if scope not in (None, "", [], {}):
                row["scope"] = scope
            prior_requests.append({
                key: value
                for key, value in row.items()
            })
    if not prior_requests:
        return None
    unfulfilled = [r for r in prior_requests if r["status"] != "fulfilled"]
    return {
        "prior_requests": prior_requests,
        "fulfilled_count": len(prior_requests) - len(unfulfilled),
        "unfulfilled_count": len(unfulfilled),
    }


def _grounding_prompt_part_facts(part_facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rendered_rows = prompt_part_facts(part_facts)
    for row in rendered_rows:
        if not isinstance(row, dict):
            continue
        row.pop("observed_store_as", None)
        row.pop("observed_aliases", None)
    return rendered_rows


def _build_grounding_fact_sections(
    llm_input: dict[str, Any],
    *,
    session_state: dict[str, Any],
    world_observation_surface: dict[str, Any],
    turn_history: list[dict[str, Any]],
) -> list[str]:
    payload = dict(llm_input or {})
    observed_runtime_state = dict(payload.get("observed_runtime_state") or {})
    safety_section = {
        "obligation_targets": deepcopy(payload.get("obligation_targets") or []),
        "loaded_safety_rules": deepcopy(payload.get("loaded_safety_rules") or []),
    }
    latest_delta = _latest_session_delta(session_state)
    observation_store = deepcopy(session_state.get("observation_store") or {})
    observation_fact_ledger = deepcopy(session_state.get("observation_fact_ledger") or {})
    grounding_contract = deepcopy(session_state.get("grounding_contract") or {})
    sections: list[str] = [
        "Fault Event",
        json_block(payload.get("fault_event") or {}),
    ]
    fulfillment = _build_observation_fulfillment_status(session_state, turn_history)
    if fulfillment is not None:
        sections.extend(
            [
                "",
                "Observation Fulfillment Status",
                json_block(fulfillment),
            ]
        )
    if grounding_contract not in ({}, [], "", None):
        sections.extend(
            [
                "",
                "Grounding Contract",
                json_block(grounding_contract),
            ]
        )
    if latest_delta:
        sections.extend(
            [
                "",
                "Latest Session Delta",
                json_block(latest_delta),
            ]
        )
    sections.extend(
        [
            "",
            "Current Resource Facts",
            json_block(observed_runtime_state.get("resources") or []),
            "",
            "Current Part Facts",
            json_block(_grounding_prompt_part_facts(payload.get("part_facts") or [])),
            "",
            "Loaded Safety Rules",
            json_block(safety_section),
            "",
            "Relevant Assembly Requirements",
            json_block(payload.get("relevant_assembly_requirements") or []),
            "",
            "Modeled Continuation Gap",
            json_block(payload.get("modeled_continuation_gap") or {}),
            "",
            "World Observation Surface",
            json_block(deepcopy(world_observation_surface or {})),
        ]
    )
    return sections


def _grounding_contract() -> dict[str, Any]:
    return {
        "required_fields": [
            "thought",
            "decision",
            "blocking_reasons",
            "grounded_facts",
            "recovery_implications",
        ],
        "decision": ["observe", "grounded"],
        "observe_required_fields": ["observe_reason", "observe_requests"],
        "observe_request": {
            "required_fields": ["fact_type", "entity"],
            "optional_fields": ["scope", "reason"],
            "storage": "Observation results are stored internally by the runtime.",
        },
    }


def _outline_validation_ref(finding: dict[str, Any]) -> dict[str, Any]:
    ref = {
        "task_id": str(finding.get("task_id") or "").strip(),
        "pose_source": str(finding.get("pose_source") or "").strip(),
        "failed_axes": [
            str(item).strip()
            for item in (finding.get("failed_axes") or [])
            if str(item).strip()
        ],
    }
    resource_jid = str(finding.get("resource_jid") or "").strip()
    if resource_jid:
        ref["resource_jid"] = resource_jid
    failed_reason = str(finding.get("failed_reason") or "").strip()
    if failed_reason:
        ref["failed_reason"] = failed_reason
    return ref


def _outline_contract() -> dict[str, Any]:
    return {
        "required_fields": ["thought", "addressed_validation_findings", "outline_tasks"],
        "addressed_validation_findings": {
            "required_fields": ["task_id", "pose_source", "failed_axes"],
            "optional_fields": ["resource_jid", "failed_reason"],
            "usage": (
                "Use the exact refs listed in Repair Contract.required_addressed_validation_findings. "
                "Use [] when no outstanding validation findings are present."
            ),
        },
        "outline_task": {
            "required_fields": [
                "outline_id",
                "resource_jid",
                "description",
                "rationale",
                "expected_start_state",
                "expected_end_state",
                "depends_on",
            ],
            "optional_fields": ["part_name", "action_target"],
        },
    }


def _primitive_generation_contract() -> dict[str, Any]:
    return {
        "required_fields": ["thought", "decision", "macro_tasks"],
        "decision": ["need_outline_revision", "draft_ready"],
        "macro_task": {
            "required_fields": [
                "resource_jid",
                "macro_name",
                "description",
                "rationale",
                "expected_start_state",
                "task_params",
                "task_metadata",
                "primitive_steps",
            ],
            "optional_fields": ["part_name"],
        },
    }


def _finalize_contract() -> dict[str, Any]:
    return {
        "required_fields": ["thought", "decision", "final_proposal"],
        "decision": [
            "final_ready",
            "need_outline_revision",
            "need_primitive_revision",
        ],
        "final_proposal": {
            "required_fields": ["thought", "primary_obligation", "macro_tasks"],
        },
    }


def _contract_for_phase(phase: str) -> dict[str, Any]:
    normalized = str(phase or "").strip().lower()
    if normalized == "grounding":
        return _grounding_contract()
    if normalized == "outline":
        return _outline_contract()
    if normalized == "primitive_generation":
        return _primitive_generation_contract()
    if normalized == "finalize":
        return _finalize_contract()
    raise ValueError(f"unsupported multi-turn phase: {phase!r}")


def build_multi_turn_phase_prompt_input(
    *,
    phase: str,
    llm_input: dict[str, Any],
    session_state: dict[str, Any],
    world_observation_surface: dict[str, Any] | None = None,
    recovery_gap_state: dict[str, Any] | None = None,
    grounded_feasibility_facts: list[dict[str, Any]] | None = None,
    pruned_actions: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    normalized_phase = str(phase or "").strip().lower()
    return {
        "reasoning_mode": "multi_turn",
        "phase": normalized_phase,
        "llm_input": deepcopy(llm_input or {}),
        "session_state": deepcopy(session_state or {}),
        "world_observation_surface": deepcopy(world_observation_surface or {}),
        "recovery_gap_state": deepcopy(recovery_gap_state or {}),
        "grounded_feasibility_facts": deepcopy(grounded_feasibility_facts or []),
        "pruned_actions": deepcopy(pruned_actions or []),
        "response_contract": _contract_for_phase(normalized_phase),
    }


def render_multi_turn_phase_prompt(prompt_input: dict[str, Any]) -> str:
    payload = deepcopy(prompt_input or {})
    llm_input = dict(payload.get("llm_input") or {})
    session_state = dict(payload.get("session_state") or {})
    phase = str(payload.get("phase") or "").strip().lower()
    response_contract = deepcopy(payload.get("response_contract") or {})
    observation_store = deepcopy(session_state.get("observation_store") or {})
    accepted_outline = deepcopy(session_state.get("accepted_outline"))
    proposal_draft = deepcopy(session_state.get("proposal_draft"))
    turn_history = _compact_turn_history(session_state)
    world_observation_surface = deepcopy(payload.get("world_observation_surface") or {})
    recovery_gap_state = deepcopy(payload.get("recovery_gap_state") or {})
    pruned_actions = deepcopy(payload.get("pruned_actions") or [])
    if phase == "outline":
        llm_input = _outline_prompt_llm_input(llm_input)
        pruned_actions = _outline_prompt_pruned_actions(pruned_actions)
    outline_validation_findings = [
        deepcopy(row)
        for row in (session_state.get("outline_validation_findings") or [])
        if isinstance(row, dict)
    ]
    prompt_outline_validation_findings = (
        _outline_prompt_validation_findings(outline_validation_findings)
        if phase == "outline"
        else deepcopy(outline_validation_findings)
    )
    required_addressed_validation_findings = [
        _outline_validation_ref(row) for row in outline_validation_findings
    ]
    repair_contract = (
        _outline_prompt_repair_contract(
            outline_validation_findings=prompt_outline_validation_findings,
            required_addressed_validation_findings=required_addressed_validation_findings,
            pruned_actions=pruned_actions,
            session_state=session_state,
        )
        if phase == "outline"
        else None
    )

    extra_sections: list[tuple[str, Any]] = [
        ("Session State", _compact_session_state(session_state)),
        ("Session Observation Store", observation_store),
    ]
    if phase == "grounding":
        extra_sections.append(
            (
                "World Observation Surface",
                deepcopy(payload.get("world_observation_surface") or {}),
            )
        )
    elif phase == "primitive_generation":
        extra_sections.append(("Accepted Outline", accepted_outline))
    elif phase == "finalize":
        extra_sections.extend(
            [
                ("Accepted Outline", accepted_outline),
                ("Proposal Draft", proposal_draft),
            ]
        )
    if turn_history and phase != "outline":
        extra_sections.append(("Session Turn History", turn_history))

    include_surface = phase == "primitive_generation"
    sections: list[str] = [
        "Task and Role",
        (
            "You are the active replanner for a DES fallback recovery session.\n"
            f"Current phase: {_PHASE_TITLES.get(phase, phase)}.\n"
            "Use the structured context below to produce only the output required for this phase.\n"
            "Do not skip ahead to a later phase unless the current phase contract explicitly allows it."
        ),
        "",
        *(
            _build_grounding_fact_sections(
                llm_input,
                session_state=session_state,
                world_observation_surface=world_observation_surface,
                turn_history=turn_history,
            )
            if phase == "grounding"
            else (
                _build_outline_fact_sections(
                    llm_input,
                    recovery_gap_state=recovery_gap_state,
                    repair_contract=repair_contract,
                )
                if phase == "outline"
                else build_shared_fact_sections(
                    llm_input,
                    extra_sections=extra_sections,
                    include_allowed_execution_surface=include_surface,
                )
            )
        ),
        "",
        "Required JSON Response Contract",
        json_block(response_contract),
        "",
        "Hard Constraints",
        (
            "- Use only the listed resources."
            if phase == "outline"
            else "- Use only the listed resources and controller primitives."
        ),
        "- Keep the response inside the current phase purpose and contract.",
        "- Do not invent observations, safety obligations, or grounded locations.",
        "- Do not contradict grounded runtime facts already present in the prompt.",
    ]
    if phase == "outline":
        sections.extend(
            [
                "- addressed_validation_findings must exactly match Repair Contract.required_addressed_validation_findings when that field is present; otherwise use [].",
                "- If Repair Contract is present, thought must summarize the factual outline changes made to resolve its unresolved findings.",
                "- Produce a sequence of concrete physical recovery macro-steps; each outline row should represent one task for one resource.",
                "- Each outline task must describe a real grounded state transition using facts already present in the prompt.",
                "- Each outline macro must be state-consistent with the current grounded resource/part state and with the projected state produced by predecessor macros.",
                "- Actions listed in Repair Contract.active_pruned_actions are blocked in the current state; do not reuse them until earlier recovery steps change the blocking state.",
                "- Resource-limit or safety failures are not resolved by only changing a resource's internal state, labels, or expected values; the outline must change the assignment or grounded interaction that causes the failure.",
                "- Continuation or resume tasks must depend on the concrete recovery tasks that clear the factual blockers shown in Unmet Continuation Conditions.",
                "- Do not emit raw continuation condition ids; describe recovery steps in natural physical terms.",
                "- Outline is always validated after this phase response; do not emit a phase decision field.",
            ]
        )
    if phase == "grounding":
        sections.extend(
            [
                "- Use only the observation facts listed in World Observation Surface.",
                "- Observation requests must use the exact fact_type/entity schema shown in World Observation Surface and the response contract.",
                "- Do not emit resource_jid in grounding observe requests; runtime resolves observation facts internally.",
                "- Choose \"observe\" only when at least one unresolved observation fact is still needed for recovery grounding.",
                "- Choose \"grounded\" when the prompt already contains the needed observation facts and Observation Fulfillment Status shows no unresolved observation request remains.",
                "- If decision is \"observe\", provide a non-empty observe_reason and list only unresolved observation facts.",
                "- If decision is \"grounded\", leave observe_requests empty.",
                "- grounded_facts must summarize the concrete grounded facts established by the current prompt state and prior observations.",
                "- recovery_implications must summarize the immediate recovery-relevant consequences implied by those facts.",
                "- Do not request observations for facts already marked \"fulfilled\" in Observation Fulfillment Status.",
            ]
        )
    return "\n".join(sections).strip() + "\n"


def multi_turn_phase_response_schema(phase: str) -> dict[str, Any]:
    normalized = str(phase or "").strip().lower()
    if normalized == "grounding":
        return {
            "name": "multi_turn_grounding_response",
            "strict": False,
            "schema": {
                "type": "object",
                "properties": {
                    "thought": {"type": "string"},
                    "decision": {"type": "string", "enum": ["observe", "grounded"]},
                    "blocking_reasons": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "grounded_facts": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "recovery_implications": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "observe_reason": {"type": "string"},
                    "observe_requests": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "fact_type": {"type": "string", "minLength": 1},
                                "entity": {"type": "string", "minLength": 1},
                                "scope": {},
                                "reason": {"type": "string"},
                            },
                            "required": ["fact_type", "entity"],
                        },
                    },
                },
                "required": [
                    "thought",
                    "decision",
                    "blocking_reasons",
                    "grounded_facts",
                    "recovery_implications",
                ],
            },
        }
    if normalized == "outline":
        return {
            "name": "multi_turn_outline_response",
            "strict": False,
            "schema": {
                "type": "object",
                "properties": {
                    "thought": {"type": "string"},
                    "addressed_validation_findings": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "task_id": {"type": "string"},
                                "resource_jid": {"type": "string"},
                                "pose_source": {"type": "string"},
                                "failed_axes": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                                "failed_reason": {"type": "string"},
                            },
                            "required": ["task_id", "pose_source", "failed_axes"],
                        },
                    },
                    "outline_tasks": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "outline_id": {"type": "string"},
                                "resource_jid": {"type": "string"},
                                "description": {"type": "string"},
                                "rationale": {"type": "string"},
                                "part_name": {"type": "string"},
                                "action_target": {
                                    "type": "object",
                                    "properties": {
                                        "target_location": {"type": "string"},
                                        "source_location": {"type": "string"},
                                        "named_pose": {"type": "string"},
                                        "requirement_id": {"type": "string"},
                                    },
                                },
                                "expected_start_state": {"type": "object"},
                                "expected_end_state": {"type": "object"},
                                "depends_on": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                            },
                            "required": [
                                "outline_id",
                                "resource_jid",
                                "description",
                                "rationale",
                                "expected_start_state",
                                "expected_end_state",
                                "depends_on",
                            ],
                        },
                    },
                },
                "required": ["thought", "addressed_validation_findings", "outline_tasks"],
            },
        }
    if normalized == "primitive_generation":
        return {
            "name": "multi_turn_primitive_generation_response",
            "strict": False,
            "schema": {
                "type": "object",
                "properties": {
                    "thought": {"type": "string"},
                    "decision": {
                        "type": "string",
                        "enum": [
                            "need_outline_revision",
                            "draft_ready",
                        ],
                    },
                    "macro_tasks": {"type": "array", "items": {"type": "object"}},
                },
                "required": ["thought", "decision", "macro_tasks"],
            },
        }
    if normalized == "finalize":
        return {
            "name": "multi_turn_finalize_response",
            "strict": False,
            "schema": {
                "type": "object",
                "properties": {
                    "thought": {"type": "string"},
                    "decision": {
                        "type": "string",
                        "enum": [
                            "final_ready",
                            "need_outline_revision",
                            "need_primitive_revision",
                        ],
                    },
                    "final_proposal": {"type": "object"},
                },
                "required": ["thought", "decision", "final_proposal"],
            },
        }
    raise ValueError(f"unsupported multi-turn phase: {phase!r}")


__all__ = [
    "build_multi_turn_phase_prompt_input",
    "multi_turn_phase_response_schema",
    "render_multi_turn_phase_prompt",
]
