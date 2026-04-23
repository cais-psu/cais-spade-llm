"""Product-owned bridge validation orchestration for typed DES/PPR events."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable

from cais_spade_llm.agents.central_controller.outline_macro_safety import (
    validate_outline_macro_cca_constraints,
)
from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.bridge_des_semantics import (
    BridgeEventProjection,
    BridgeEventInstance,
    BridgeValidationContext,
    BridgeValidationFinding,
    BridgeValidationResult,
    ProcessSchema,
    build_outline_task_row,
    build_bridge_validation_context,
    parse_bridge_event_instance,
    validate_bridge_event_instance,
)
from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.modes.multi_turn_outline_state import (
    _apply_outline_task_effects,
    _build_outline_task_type_lookup,
    _infer_outline_macro_signature,
    _outline_task_predecessors,
    _task_findings_block_projected_state,
)


ProgressEvaluator = Callable[..., tuple[int, dict[str, Any]]]

_LIVE_EVENT_INSTANCE_KEYS = frozenset(
    {
        "outline_id",
        "event_schema_id",
        "resource_binding",
        "object_bindings",
        "parameters",
        "rationale",
    }
)


def _candidate_contract_findings(
    candidate_task: dict[str, Any],
) -> list[BridgeValidationFinding]:
    raw = dict(candidate_task or {})
    outline_id = str(raw.get("outline_id") or "").strip()
    unexpected_fields = sorted(
        key
        for key in raw
        if str(key or "").strip() and str(key or "").strip() not in _LIVE_EVENT_INSTANCE_KEYS
    )
    if unexpected_fields:
        return [
            BridgeValidationFinding(
                stage="schema_grounding",
                code="unexpected_event_fields",
                reason=(
                    "candidate event rows must use only canonical fields: "
                    f"{', '.join(sorted(_LIVE_EVENT_INSTANCE_KEYS))}; got "
                    f"{', '.join(unexpected_fields)}"
                ),
                task_id=outline_id,
                evidence={
                    "unexpected_fields": unexpected_fields,
                    "allowed_fields": sorted(_LIVE_EVENT_INSTANCE_KEYS),
                },
            )
        ]

    findings: list[BridgeValidationFinding] = []
    if "object_bindings" not in raw:
        findings.append(
            BridgeValidationFinding(
                stage="schema_grounding",
                code="missing_object_bindings",
                reason="candidate must include object_bindings",
                task_id=outline_id,
                evidence={"field": "object_bindings"},
            )
        )
    elif not isinstance(raw.get("object_bindings"), dict):
        findings.append(
            BridgeValidationFinding(
                stage="schema_grounding",
                code="invalid_object_bindings_type",
                reason="object_bindings must be an object",
                task_id=outline_id,
                evidence={"field": "object_bindings"},
            )
        )

    if "parameters" not in raw:
        findings.append(
            BridgeValidationFinding(
                stage="schema_grounding",
                code="missing_parameters",
                reason="candidate must include parameters",
                task_id=outline_id,
                evidence={"field": "parameters"},
            )
        )
    elif not isinstance(raw.get("parameters"), dict):
        findings.append(
            BridgeValidationFinding(
                stage="schema_grounding",
                code="invalid_parameters_type",
                reason="parameters must be an object",
                task_id=outline_id,
                evidence={"field": "parameters"},
            )
        )

    if "rationale" not in raw:
        findings.append(
            BridgeValidationFinding(
                stage="schema_grounding",
                code="missing_rationale",
                reason="candidate must include rationale",
                task_id=outline_id,
                evidence={"field": "rationale"},
            )
        )
    elif not isinstance(raw.get("rationale"), str):
        findings.append(
            BridgeValidationFinding(
                stage="schema_grounding",
                code="invalid_rationale_type",
                reason="rationale must be a string",
                task_id=outline_id,
                evidence={"field": "rationale"},
            )
        )
    return findings


def projected_outline_validation_context(
    *,
    session_state: dict[str, Any],
    prepared_bridge_request: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    llm_input = dict(prepared_bridge_request.get("llm_input") or {})
    observed_runtime_state = dict(llm_input.get("observed_runtime_state") or {})

    resources_by_jid: dict[str, dict[str, Any]] = {}
    for row in (observed_runtime_state.get("resources") or []):
        if not isinstance(row, dict):
            continue
        resource_jid = str(row.get("resource_jid") or "").strip()
        if resource_jid:
            resources_by_jid[resource_jid] = deepcopy(row)
    for resource_jid, row in dict(session_state.get("symbolic_resources") or {}).items():
        token = str(resource_jid or "").strip()
        if token and isinstance(row, dict):
            resources_by_jid[token] = deepcopy(row)
    bridge_resources = dict(prepared_bridge_request.get("bridge_resources") or {})
    for resource_jid, raw_entry in bridge_resources.items():
        token = str(resource_jid or "").strip()
        if not token or not isinstance(raw_entry, dict):
            continue
        entry = dict(raw_entry)
        bridge_snapshot = dict(entry.get("bridge_snapshot") or {})
        static_capabilities = dict(entry.get("static_capabilities") or {})
        resource_row = resources_by_jid.setdefault(token, {"resource_jid": token})
        for key in (
            "named_poses",
            "available_named_poses",
            "supported_recovery_states",
            "available_recovery_states",
            "reachability",
            "reachable_locations",
            "known_locations",
            "staging_areas",
            "workspace_bounds",
        ):
            if resource_row.get(key) not in (None, "", [], {}):
                continue
            if static_capabilities.get(key) not in (None, "", [], {}):
                resource_row[key] = deepcopy(static_capabilities.get(key))
            elif bridge_snapshot.get(key) not in (None, "", [], {}):
                resource_row[key] = deepcopy(bridge_snapshot.get(key))

    parts_by_name: dict[str, dict[str, Any]] = {}
    for row in (llm_input.get("part_facts") or []):
        if not isinstance(row, dict):
            continue
        part_name = str(row.get("part_name") or "").strip()
        if part_name:
            parts_by_name[part_name] = deepcopy(row)
    for part_name, row in dict(session_state.get("symbolic_parts") or {}).items():
        token = str(part_name or "").strip()
        if token and isinstance(row, dict):
            parts_by_name[token] = deepcopy(row)

    for entry in dict(session_state.get("observation_store") or {}).values():
        if not isinstance(entry, dict):
            continue
        part_name = str(entry.get("part_name") or "").strip()
        if not part_name:
            continue
        is_new_part = part_name not in parts_by_name
        part_row = parts_by_name.setdefault(part_name, {"part_name": part_name})
        pose = dict(entry.get("pose") or {})
        if not pose and entry.get("x") is not None:
            pose = {"x": entry.get("x"), "y": entry.get("y"), "z": entry.get("z")}
        if pose:
            part_row["observed_pose"] = deepcopy(pose)
        if (
            is_new_part
            and part_row.get("current_location") in (None, "")
            and entry.get("current_location") not in (None, "")
        ):
            part_row["current_location"] = deepcopy(entry.get("current_location"))
        holder = str(entry.get("current_holder_resource_jid") or "").strip()
        if (
            is_new_part
            and not str(part_row.get("current_holder_resource_jid") or "").strip()
            and holder
        ):
            part_row["current_holder_resource_jid"] = holder
    return resources_by_jid, parts_by_name


def validate_bridge_candidate_task(
    *,
    planner: Any,
    candidate_task: dict[str, Any],
    session_state: dict[str, Any],
    prepared_bridge_request: dict[str, Any],
    progress_evaluator: ProgressEvaluator | None = None,
) -> BridgeValidationResult:
    raw_candidate_task = deepcopy(dict(candidate_task or {}))
    resources_by_jid, parts_by_name = projected_outline_validation_context(
        session_state=session_state,
        prepared_bridge_request=prepared_bridge_request,
    )
    context = build_bridge_validation_context(
        resources_by_jid=resources_by_jid,
        parts_by_name=parts_by_name,
        prepared_bridge_request=prepared_bridge_request,
    )
    contract_findings = _candidate_contract_findings(raw_candidate_task)
    if contract_findings:
        return BridgeValidationResult(
            ok=False,
            event_instance=parse_bridge_event_instance(
                raw_candidate_task,
                outline_id=str(raw_candidate_task.get("outline_id") or "").strip(),
            ),
            findings=deepcopy(contract_findings),
        )
    event_instance = parse_bridge_event_instance(
        raw_candidate_task,
        outline_id=str(raw_candidate_task.get("outline_id") or "").strip(),
    )
    return validate_bridge_candidate_event(
        planner=planner,
        event_instance=event_instance,
        session_state=session_state,
        prepared_bridge_request=prepared_bridge_request,
        progress_evaluator=progress_evaluator,
        context=context,
    )


def validate_bridge_candidate_event(
    *,
    planner: Any,
    event_instance: BridgeEventInstance,
    session_state: dict[str, Any],
    prepared_bridge_request: dict[str, Any],
    progress_evaluator: ProgressEvaluator | None = None,
    context: BridgeValidationContext | None = None,
) -> BridgeValidationResult:
    if context is None:
        resources_by_jid, parts_by_name = projected_outline_validation_context(
            session_state=session_state,
            prepared_bridge_request=prepared_bridge_request,
        )
        context = build_bridge_validation_context(
            resources_by_jid=resources_by_jid,
            parts_by_name=parts_by_name,
            prepared_bridge_request=prepared_bridge_request,
        )
    resources_by_jid = deepcopy(context.resources_by_jid)
    parts_by_name = deepcopy(context.parts_by_name)
    semantic_result = validate_bridge_event_instance(
        event_instance,
        context=context,
    )
    if not semantic_result.ok:
        return semantic_result
    schema = semantic_result.schema
    projection = semantic_result.projection
    if schema is None or projection is None:
        return _result_with_findings(
            semantic_result,
            [
                BridgeValidationFinding(
                    stage="schema_grounding",
                    code="missing_semantic_projection",
                    reason="validated bridge event did not expose schema/projection semantics",
                    task_id=str(event_instance.outline_id or "").strip(),
                    resource_jid=str(event_instance.resource_binding or "").strip(),
                )
            ],
        )
    outline_task = build_outline_task_row(
        instance=event_instance,
        schema=schema,
        projection=projection,
    )

    cca_findings = _validate_outline_task_cca(
        task=outline_task,
        event_instance=event_instance,
        projection=projection,
        resources_by_jid=resources_by_jid,
        parts_by_name=parts_by_name,
        llm_input=dict(prepared_bridge_request.get("llm_input") or {}),
        prior_findings=[],
    )
    if cca_findings:
        return _result_with_findings(
            semantic_result,
            _translate_findings(
                task=outline_task,
                findings=cca_findings,
                stage="supervisor_admissibility",
                default_code="supervisor_blocked",
                default_reason="supervisor rejected the candidate event",
            ),
        )

    progress_score, progress_detail = _evaluate_marked_progress(
        outline_task=outline_task,
        event_instance=event_instance,
        projection=projection,
        session_state=session_state,
        prepared_bridge_request=prepared_bridge_request,
        progress_evaluator=progress_evaluator,
        context=context,
        semantic_result=semantic_result,
    )
    if int(progress_score or 0) <= 0:
        return _result_with_findings(
            semantic_result,
            [
                BridgeValidationFinding(
                    stage="marked_progress",
                    code="no_marked_progress",
                    reason=(
                        "candidate event does not reduce the active recovery gap "
                        "toward a marked or continuation-ready state"
                    ),
                    task_id=str(outline_task.get("outline_id") or "").strip(),
                    resource_jid=str(outline_task.get("resource_jid") or "").strip(),
                    part_name=str(outline_task.get("part_name") or "").strip(),
                    evidence=deepcopy(progress_detail),
                    retry_hint=(
                        "choose an enabled event that strictly reduces the active "
                        "continuation blockers"
                    ),
                )
            ],
        )

    return BridgeValidationResult(
        ok=True,
        event_instance=semantic_result.event_instance,
        findings=[],
        schema=semantic_result.schema,
        projection=projection,
        projected_resources=deepcopy(semantic_result.projected_resources or {}),
        projected_parts=deepcopy(semantic_result.projected_parts or {}),
    )


def _result_with_findings(
    semantic_result: BridgeValidationResult,
    findings: list[BridgeValidationFinding],
) -> BridgeValidationResult:
    return BridgeValidationResult(
        ok=False,
        event_instance=semantic_result.event_instance,
        findings=deepcopy(findings),
        schema=semantic_result.schema,
        projection=deepcopy(semantic_result.projection),
        projected_resources=deepcopy(semantic_result.projected_resources or {}),
        projected_parts=deepcopy(semantic_result.projected_parts or {}),
    )


def _translate_findings(
    *,
    task: dict[str, Any],
    findings: list[dict[str, Any]],
    stage: str,
    default_code: str,
    default_reason: str,
) -> list[BridgeValidationFinding]:
    translated: list[BridgeValidationFinding] = []
    for finding in findings:
        row = deepcopy(dict(finding or {}))
        translated.append(
            BridgeValidationFinding(
                stage=stage,
                code=str(row.get("constraint_code") or default_code).strip(),
                reason=str(row.get("reason") or default_reason).strip(),
                task_id=str(
                    row.get("task_id")
                    or task.get("outline_id")
                    or ""
                ).strip(),
                resource_jid=str(
                    row.get("resource_jid")
                    or task.get("resource_jid")
                    or ""
                ).strip(),
                part_name=str(
                    row.get("part_name")
                    or task.get("part_name")
                    or ""
                ).strip(),
                unsatisfied_predicates=[
                    str(item).strip()
                    for item in (row.get("unsatisfied_predicates") or [])
                    if str(item).strip()
                ],
                evidence=deepcopy(row.get("evidence") or {}),
                retry_hint=str(row.get("retry_hint") or "").strip(),
            )
        )
    return translated


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
        "constraint_owner": "resource",
        "constraint_family": "resource_feasibility",
        "constraint_code": constraint_code,
        "reason": reason,
        "guard": deepcopy(guard),
        "evidence": deepcopy(evidence),
    }


def _validate_outline_task_cca(
    *,
    task: dict[str, Any],
    event_instance: BridgeEventInstance,
    projection: BridgeEventProjection,
    resources_by_jid: dict[str, dict[str, Any]],
    parts_by_name: dict[str, dict[str, Any]],
    llm_input: dict[str, Any],
    prior_findings: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    validation_trace = [deepcopy(task)]
    task_id = str(task.get("outline_id") or "").strip() or "task_0"
    task_types_by_id = _build_outline_task_type_lookup(
        validation_trace,
        resources_by_jid=resources_by_jid,
        parts_by_name=parts_by_name,
        llm_input=llm_input,
    )
    signature = _infer_outline_macro_signature(
        task,
        resources_by_jid=resources_by_jid,
        parts_by_name=parts_by_name,
    )

    projected_resources = deepcopy(resources_by_jid)
    projected_parts = deepcopy(parts_by_name)
    if not _task_findings_block_projected_state(list(prior_findings or [])):
        task_type = str(task_types_by_id.get(task_id) or "").strip()
        _apply_outline_task_effects(
            task,
            resources_by_jid=projected_resources,
            parts_by_name=projected_parts,
            task_type=task_type,
            event_instance=event_instance,
            projection=projection,
        )

    cca_result = validate_outline_macro_cca_constraints(
        task=deepcopy(task),
        event_instance=deepcopy(event_instance),
        projection=deepcopy(projection),
        signature=deepcopy(signature),
        pre_resources=deepcopy(resources_by_jid),
        pre_parts=deepcopy(parts_by_name),
        projected_resources=projected_resources,
        projected_parts=projected_parts,
        llm_input=deepcopy(llm_input),
        outline_tasks=validation_trace,
        task_types_by_id=deepcopy(task_types_by_id),
        task_index_by_id={task_id: 0},
        dependency_map={task_id: _outline_task_predecessors(task)},
        previously_cleared_condition_ids=None,
    )
    return [
        deepcopy(row)
        for row in (cca_result.get("findings") or [])
        if isinstance(row, dict)
    ]


def _evaluate_marked_progress(
    *,
    outline_task: dict[str, Any],
    event_instance: BridgeEventInstance,
    projection: BridgeEventProjection,
    session_state: dict[str, Any],
    prepared_bridge_request: dict[str, Any],
    progress_evaluator: ProgressEvaluator | None,
    context: BridgeValidationContext,
    semantic_result: BridgeValidationResult,
) -> tuple[int, dict[str, Any]]:
    if callable(progress_evaluator):
        return progress_evaluator(
            task=deepcopy(outline_task),
            session_state=session_state,
            prepared_bridge_request=prepared_bridge_request,
        )

    current_satisfied = _marked_condition_satisfied_count(
        resources_by_jid=context.resources_by_jid,
        parts_by_name=context.parts_by_name,
        marked_conditions=context.marked_conditions,
    )
    projected_satisfied = _marked_condition_satisfied_count(
        resources_by_jid=semantic_result.projected_resources,
        parts_by_name=semantic_result.projected_parts,
        marked_conditions=context.marked_conditions,
    )
    if projected_satisfied > current_satisfied:
        return (
            projected_satisfied - current_satisfied,
            {
                "resolved_marked_conditions": projected_satisfied - current_satisfied,
                "remaining_marked_conditions": max(
                    0, len(context.marked_conditions) - projected_satisfied
                ),
            },
        )

    if callable(getattr(semantic_result.schema, "progress_policy", None)):
        return semantic_result.schema.progress_policy(
            event_instance=deepcopy(event_instance),
            projection=deepcopy(projection),
            context=context,
            semantic_result=semantic_result,
            projected_satisfied=projected_satisfied,
        )

    return (
        0,
        {
            "resolved_marked_conditions": 0,
            "remaining_marked_conditions": max(
                0, len(context.marked_conditions) - projected_satisfied
            ),
        },
    )


def _marked_condition_satisfied_count(
    *,
    resources_by_jid: dict[str, dict[str, Any]],
    parts_by_name: dict[str, dict[str, Any]],
    marked_conditions: list[dict[str, Any]],
) -> int:
    satisfied = 0
    for condition in marked_conditions:
        entity_kind = str(condition.get("entity_kind") or "").strip().lower()
        entity = str(condition.get("entity") or "").strip()
        field = str(condition.get("field") or "").strip()
        expected = condition.get("expected")
        row = (
            dict(parts_by_name.get(entity) or {})
            if entity_kind == "part"
            else dict(resources_by_jid.get(entity) or {})
        )
        if not row:
            continue
        actual = row.get(field)
        if isinstance(expected, (dict, list)):
            if actual == expected:
                satisfied += 1
        elif expected in (None, ""):
            if actual in (None, ""):
                satisfied += 1
        elif str(actual or "").strip() == str(expected or "").strip():
            satisfied += 1
    return satisfied


__all__ = [
    "projected_outline_validation_context",
    "validate_bridge_candidate_event",
    "validate_bridge_candidate_task",
]
