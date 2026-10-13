from __future__ import annotations

"""Prove a saved nominal continuation without appending generated events."""

from collections import deque
from copy import deepcopy
import math
from types import SimpleNamespace
from typing import Any

from cais_spade_llm.agents.shared_information.recovery_validation_protocol import (
    recovery_validation_fingerprint,
)


def _nominal_event(node: dict, prepared: dict) -> dict:
    entry = prepared.get("recovery_resources", {}).get(node.get("resource_jid"), {})
    events = [row for row in entry.get("recovery_des_model", {}).get("events", [])
              if row.get("event_name") == node.get("function_name")]
    matches = []
    for event in events:
        contract = event.get("composition_contract", {})
        if "parameter_bindings" in contract:
            bindings = contract["parameter_bindings"]
            if not isinstance(bindings, dict):
                raise ValueError("NEEDS_CONTEXT: nominal task parameter bindings are malformed")
            if recovery_validation_fingerprint(bindings) != recovery_validation_fingerprint(node.get("params") or {}):
                continue
        matches.append(event)
    if len(matches) != 1:
        raise ValueError("NEEDS_CONTEXT: saved nominal task requires one exact declared transition and parameter binding")
    return matches[0]


def _nominal_arguments(node: dict, prepared: dict) -> tuple[dict, dict]:
    """Bind saved environment parameters through the registered owner adapter.

    Environment completion requirements remain in the evidence. They are not
    controller arguments and their presence does not prove their completion.
    """
    jid = node.get("resource_jid")
    entry = prepared.get("recovery_resources", {}).get(jid, {})
    event = _nominal_event(node, prepared)
    contract = event.get("composition_contract", {})
    declarations = contract.get("params", {})
    authored = deepcopy(node.get("params") or {})
    if "parameter_bindings" in contract:
        return authored, {"source": "recorded environment task parameter binding",
            "task_id": node["id"], "environment_parameters": deepcopy(authored),
            "environment_event": deepcopy(event),
            "environment_event_fingerprint": recovery_validation_fingerprint(event),
            "completion_evidence_status": "requires_validation"}
    if not set(authored) - set(declarations):
        return authored, {}
    if prepared.get("nominal_continuation_semantics") != "recorded_unfinished_tasks":
        raise ValueError("NEEDS_CONTEXT: nominal parameters are absent from the registered task contract")
    physical = prepared.get("physical_evidence", {})
    owners = [rid for rid, recorded_jid in physical.get("jids", {}).items() if recorded_jid == jid]
    if len(owners) != 1:
        raise ValueError("NEEDS_CONTEXT: nominal task has no unique recorded owner")
    rid = owners[0]
    model = physical.get("environment_models", {}).get(rid, {})
    from cais_spade_llm.resources.environment_models import _task_event

    matches = []
    unsupported_bindings = []
    for event in model.get("events", []):
        if event.get("function_name", event.get("event_name")) != node["function_name"]:
            continue
        bindings = event.get("parameter_bindings", {})
        unresolved = [name for name, binding in bindings.items()
                      if (not isinstance(binding, dict)
                          or set(binding) - {"equals", "required", "type", "reference"}
                          or ("type" in binding and binding["type"] != "string")
                          or ("reference" in binding and binding["reference"] != "part_name"))]
        if unresolved:
            unsupported_bindings.extend(unresolved)
            continue
        candidate = {"resource_id": rid, "event_id": event["event_id"],
                     "event_name": event["event_name"], "parameters": authored, "task_id": node["id"]}
        try:
            _task_event(physical["environment_models"], physical.get("products", {}), candidate)
        except ValueError:
            continue
        matches.append((event, candidate))
    if unsupported_bindings:
        raise ValueError("NEEDS_CONTEXT: unsupported nominal parameter binding semantics: "
                         + ", ".join(sorted(set(unsupported_bindings))))
    if len(matches) != 1:
        raise ValueError("NEEDS_CONTEXT: nominal environment parameters require one exact recorded event binding")
    event, candidate = matches[0]
    evidence = entry.get("resource_evidence", {})
    if entry.get("resource_type", evidence.get("resource_type")) != "robot":
        raise ValueError("NEEDS_CONTEXT: this resource has no recorded nominal argument adapter")
    from cais_spade_llm.recovery_framework.workflow_execution import _robot_arguments
    from cais_spade_llm.resources.robot.robot_task_registry import robot_task_registry

    registered = robot_task_registry().get(node["function_name"])
    if registered is None or {argument.name for argument in registered.arguments} != set(declarations):
        raise ValueError("NEEDS_CONTEXT: current nominal adapter differs from the recorded task arguments")
    inputs = evidence.get("inputs", {})
    geometry = inputs.get("geometry", {}).get("parts", {})
    part = authored.get("part_name")
    if part:
        dimensions = geometry.get("dimensions_m", {}).get(part)
        model_name = geometry.get("model_map", {}).get(part)
        if (not isinstance(dimensions, (list, tuple)) or len(dimensions) != 3
                or any(isinstance(value, bool) or not isinstance(value, (int, float))
                       or not math.isfinite(value) or value <= 0 for value in dimensions)
                or not isinstance(model_name, str) or not model_name):
            raise ValueError("NEEDS_CONTEXT: nominal geometry lacks recorded dimensions or model identity")
    context = SimpleNamespace(inputs=inputs, product_name=physical.get("product_name"), geometry={
        name: {"dimensions_m": deepcopy(size), "model_name": geometry.get("model_map", {}).get(name)}
        for name, size in geometry.get("dimensions_m", {}).items()})
    try:
        function_name, arguments = _robot_arguments(context, SimpleNamespace(agent_name=rid), candidate)
    except (KeyError, ValueError, StopIteration) as exc:
        raise ValueError(f"NEEDS_CONTEXT: recorded nominal argument preparation: {exc}") from exc
    if function_name != node["function_name"] or set(arguments) - set(declarations):
        raise ValueError("NEEDS_CONTEXT: nominal adapter does not match the exact registered task")
    provenance = {"source": "recorded environment task through workflow_execution._robot_arguments",
        "task_id": node["id"], "environment_parameters": authored,
        "environment_event": deepcopy(event), "environment_event_fingerprint": recovery_validation_fingerprint(event),
        "owner_arguments": deepcopy(arguments), "inputs_fingerprint": recovery_validation_fingerprint(inputs),
        "source_capture": deepcopy(evidence.get("source", {})), "completion_evidence_status": "requires_validation"}
    return arguments, provenance


def _nominal_task(node: dict, session: dict, prepared: dict,
                  public_findings: list | None = None) -> tuple[dict | None, str]:  # noqa: C901, PLR0912 - explicit supported guard/effect semantics
    from . import multi_turn as shared

    jid = node.get("resource_jid")
    entry = prepared.get("recovery_resources", {}).get(jid, {})
    model = entry.get("recovery_des_model") or {}
    try:
        event = _nominal_event(node, prepared)
        params, argument_evidence = _nominal_arguments(node, prepared)
    except ValueError as exc:
        if not str(exc).startswith("NEEDS_CONTEXT:"):
            raise
        return None, str(exc)
    part = params.get("part_name")
    declarations = model.get("state_variables") or {}
    resource_row = session.get("symbolic_resources", {}).get(jid, {})
    part_row = session.get("symbolic_parts", {}).get(part, {})
    for field, condition in event.get("guards", {}).items():
        if field not in declarations:
            return None, f"NEEDS_CONTEXT: nominal guard {field} has no declaration"
        scope = declarations[field].get("scope", "resource")
        known, value = shared._projected_state_value(row=part_row if scope == "part" else resource_row,
                                                    field_name=field, scope=scope)
        if not known:
            if public_findings is not None:
                finding = {"constraint_code": "nominal_task_condition_unavailable"}
                if not declarations[field].get("private") and not field.startswith(("task_ctx.", "_")):
                    finding["evidence"] = {"field": field, "available": False}
                public_findings.append(finding)
            return None, f"NEEDS_CONTEXT: nominal guard {field} is unavailable"
        for operator, expected in condition.items():
            if operator.endswith("_from_param"):
                if expected not in params:
                    return None, f"NEEDS_CONTEXT: nominal parameter {expected} is unavailable"
                expected, operator = params[expected], operator.removesuffix("_from_param")
            equal = type(value) is type(expected) and value == expected
            if operator == "equals":
                passed = equal
            elif operator == "not_equals":
                passed = not equal
            elif operator == "exists":
                passed = (value is not None) == expected
            elif operator in {"in", "not_in"} and isinstance(expected, list):
                passed = any(type(value) is type(item) and value == item for item in expected)
                if operator == "not_in":
                    passed = not passed
            else:
                return None, f"NEEDS_CONTEXT: unsupported nominal guard {operator}"
            if not passed:
                if public_findings is not None:
                    finding = {"constraint_code": "nominal_task_condition_not_met"}
                    if not declarations[field].get("private") and not field.startswith(("task_ctx.", "_")):
                        finding["evidence"] = {"field": field, "actual": deepcopy(value),
                            "expected": deepcopy(expected), "condition_operator": operator, "available": True}
                    public_findings.append(finding)
                return None, f"Nominal guard {field} is not satisfied"
    end = {}
    for field, effect in event.get("updates", {}).items():
        if declarations.get(field, {}).get("private") or field.startswith("task_ctx."):
            continue
        if field not in declarations:
            return None, f"NEEDS_CONTEXT: nominal effect {field} is undeclared"
        if set(effect) == {"set"}:
            end[field] = deepcopy(effect["set"])
        elif set(effect) == {"set_from_param"} and effect["set_from_param"] in params:
            end[field] = deepcopy(params[effect["set_from_param"]])
        else:
            return None, f"NEEDS_CONTEXT: nominal effect {field} cannot be resolved"
    task = {"outline_id": node["id"], "event_name": node["function_name"], "resource_jid": jid,
            "expected_end_state": end, "rationale": "Recorded unfinished nominal task"}
    if part:
        task["part_name"] = part
    task, findings = shared._derive_candidate_outline_task(candidate_task=task, session_state=session,
                                                          prepared_recovery_request=prepared)
    if findings:
        return None, "NEEDS_CONTEXT: " + "; ".join(row.get("reason", "") for row in findings)
    task["params"] = deepcopy(params)
    task["required_task_contract"] = node["function_name"]
    if argument_evidence:
        task["nominal_argument_evidence"] = argument_evidence
    return task, ""


def nominal_continuation_findings(result: dict, prepared: dict) -> list[dict]:
    """Expose recorded nominal blockers without private validation payloads."""
    if result.get("complete") is True:
        return []
    from .multi_turn_prompts import _outline_feedback_for_prompt, _outline_safety_checks_for_prompt

    fields_by_jid = {jid: {name for name, declaration in entry.get("recovery_des_model", {}).get("state_variables", {}).items()
                          if not declaration.get("private") and not name.startswith(("task_ctx.", "_"))}
                     for jid, entry in prepared.get("recovery_resources", {}).items()}
    fields = set().union(*fields_by_jid.values()) if fields_by_jid else set()
    rows = []
    for item in [result, *result.get("failures", [])]:
        for key, code in (("missing_conditions", "nominal_goal_fact_unavailable"),
                          ("unmet_conditions", "nominal_goal_not_met")):
            for condition in item.get(key, []):
                if not isinstance(condition, dict) or condition.get("field") not in fields:
                    continue
                finding = {"phase": "nominal_continuation", "constraint_code": code,
                    "evidence": {"field": condition["field"], "expected": deepcopy(condition.get("expected")),
                                 "available": key != "missing_conditions"}}
                if condition.get("entity_kind") == "resource":
                    finding["resource_jid"] = condition.get("entity")
                elif condition.get("entity_kind") == "part":
                    finding["part_name"] = condition.get("entity")
                rows.append(_outline_feedback_for_prompt(finding, fields, resource_fields=fields_by_jid))
        for finding in item.get("public_findings", []) if not item.get("task_id") else []:
            rows.append(_outline_feedback_for_prompt({"phase": "nominal_continuation", **finding},
                                                     fields, resource_fields=fields_by_jid))
    for failure in result.get("failures", []):
        task = failure.get("task") or {}
        common = {"phase": "nominal_continuation", "task_id": failure.get("task_id"),
                  "resource_jid": task.get("resource_jid", failure.get("resource_jid"))}
        if task.get("part_name"):
            common["part_name"] = task["part_name"]
        findings = deepcopy(failure.get("public_findings") or [])
        final_stages = {stage.get("validator_role"): stage
                        for stage in failure.get("validation_stages", [])}
        for stage in final_stages.values():
            evidence = stage.get("evidence") or {}
            if stage.get("validator_role") == "ResourceAgent" and evidence.get("allowed") is not True:
                findings.append({"constraint_code": "resource_validation_unavailable"
                    if evidence.get("feasibility_status") == "NEEDS_CONTEXT" else "transition_feasibility_rejected",
                    "feasibility_status": evidence.get("feasibility_status")})
            elif stage.get("validator_role") == "CentralControllerAgent" and evidence.get("is_safe") is not True:
                findings.append({"constraint_code": "continuous_mating_rejected" if evidence.get("continuous_mating_rejected") is True
                    else "physical_endpoint_rejected" if evidence.get("physical_endpoint_rejected") is True
                    else "physical_motion_rejected" if evidence.get("physical_motion_rejected") is True
                    else "physical_motion_unresolved" if evidence.get("physical_motion_unresolved") is True
                    else "physical_capability_rejected" if evidence.get("physical_capability_rejected") is True
                    else "owner_effect_rejected" if evidence.get("owner_effect_rejected") is True
                    else "safety_validation_unavailable"
                    if evidence.get("validation_status") in {"NEEDS_CONTEXT", "unresolved", "unavailable"} else "safety_rule_violation",
                    "validation_status": evidence.get("validation_status"),
                    **({"reason": evidence.get("reason")} if evidence.get("physical_motion_unresolved") is True
                       or evidence.get("physical_capability_rejected") is True or evidence.get("physical_endpoint_rejected") is True else {}),
                    "evidence": {"safety_checks": _outline_safety_checks_for_prompt(evidence.get("checks") or []),
                        **({"solver_summary": deepcopy(evidence["solver_summary"])}
                           if evidence.get("physical_motion_unresolved") is True else {}),
                        **({"mating_counterexample": {key: deepcopy(evidence["mating_rejection"]["counterexample"][key])
                            for key in ("segment", "time_from_start", "fraction", "violated_conditions")
                            if key in evidence["mating_rejection"]["counterexample"]}}
                           if evidence.get("continuous_mating_rejected") is True else {}),
                        **({"endpoint_counterexample": deepcopy(evidence["endpoint_rejection"])}
                           if evidence.get("physical_endpoint_rejected") is True else {}),
                        **({key: deepcopy(evidence[key]) for key in ("capability_result", "capability_source")}
                           if evidence.get("physical_capability_rejected") is True else {}),
                        **({"owner_effect_result": deepcopy(evidence["owner_effect_result"])}
                           if evidence.get("owner_effect_rejected") is True else {})}})
        for finding in findings:
            finding = {**common, **finding}
            finding.setdefault("evidence", {})["projected_after_task_ids"] = deepcopy(failure.get("projected_after_task_ids", []))
            rows.append(_outline_feedback_for_prompt(finding, fields, resource_fields=fields_by_jid))
    summary = {"phase": "nominal_continuation", "constraint_code":
        "nominal_continuation_unavailable" if result.get("status") == "NEEDS_CONTEXT" else "nominal_continuation_not_established"}
    details = {}
    for key in ("missing_task_ids", "pending_rule_ids"):
        if isinstance(result.get(key), list) and all(isinstance(value, str) for value in result[key]):
            details[key] = deepcopy(result[key])
    if details:
        summary["evidence"] = details
    rows.insert(0, _outline_feedback_for_prompt(summary, fields, resource_fields=fields_by_jid))
    return list({recovery_validation_fingerprint(row): row for row in rows}.values())


def _nominal_step_coverage(event: dict, witness: dict) -> str:
    """Require complete owner-replayed coverage of the registered program."""
    program = event.get("composition_contract")
    if (not isinstance(program, dict) or not isinstance(program.get("steps"), list)
            or type(witness.get("coverage_version")) is not int or witness["coverage_version"] != 1
            or witness.get("program_fingerprint") != recovery_validation_fingerprint(program)):
        return "NEEDS_CONTEXT: nominal owner witness lacks its exact registered program coverage"
    declared = program["steps"]
    covered = witness.get("steps")
    count = witness.get("interval_count")
    if (not isinstance(covered, list) or len(covered) != len(declared)
            or type(count) is not int or count < 0):
        return "NEEDS_CONTEXT: nominal owner witness does not cover every registered step"
    intervals = set()
    for index, (step, row) in enumerate(zip(declared, covered, strict=True)):
        if (not isinstance(step, dict) or not isinstance(row, dict)
                or type(row.get("index")) is not int or row["index"] != index
                or not isinstance(step.get("id"), str) or not step["id"]
                or row.get("step_id") != step["id"]
                or row.get("step_fingerprint") != recovery_validation_fingerprint(step)):
            return "NEEDS_CONTEXT: nominal owner witness changes a registered step identity"
        condition = row.get("condition")
        references = row.get("interval_indices")
        if (not isinstance(condition, dict) or not isinstance(condition.get("facts"), list)
                or not isinstance(references, list)
                or any(type(value) is not int or value < 0 or value >= count for value in references)
                or references != sorted(set(references))):
            return "NEEDS_CONTEXT: nominal owner step has unresolved condition or interval evidence"
        if step.get("when") and not condition["facts"]:
            return "NEEDS_CONTEXT: nominal owner step condition has no recorded facts"
        if condition.get("status") == "not_applicable":
            if (not step.get("when") or not condition["facts"] or row.get("status") != "skipped"
                    or references):
                return "NEEDS_CONTEXT: nominal owner witness skips a step without supported guard evidence"
        elif condition.get("status") == "applicable":
            effect = row.get("effect_evidence_fingerprint")
            if (row.get("status") != "supported" or not isinstance(row.get("params"), dict)
                    or not isinstance(effect, str) or len(effect) != 64
                    or any(character not in "0123456789abcdef" for character in effect)):
                return "NEEDS_CONTEXT: nominal owner step effect is not supported"
            intervals.update(references)
        else:
            return "NEEDS_CONTEXT: nominal owner step condition is unresolved"
    if intervals != set(range(count)):
        return "NEEDS_CONTEXT: nominal owner witness contains motion outside its registered steps"
    return ""


def _exact_nominal_witness(node: dict, ra: dict, prepared: dict) -> str:
    """Require the actual registered nominal task, without preceding recovery work."""
    model = prepared.get("recovery_resources", {}).get(node.get("resource_jid"), {}).get("recovery_des_model", {})
    try:
        events = [_nominal_event(node, prepared)]
        authored, _ = _nominal_arguments(node, prepared)
    except ValueError as exc:
        if not str(exc).startswith("NEEDS_CONTEXT:"):
            raise
        return str(exc)
    if ra.get("primitive_support", {}).get("semantics") == "successor_primitive_union":
        witness = ra.get("geometry_evidence", {}).get("task_contract_witness")
        if (len(events) != 1 or not isinstance(witness, dict)
                or witness.get("task_contract") != node.get("function_name")
                or witness.get("declaration_fingerprint") != recovery_validation_fingerprint(events[0])):
            return "NEEDS_CONTEXT: nominal continuation requires its exact owner event contract"
        declarations = events[0].get("composition_contract", {}).get("params", {})
        if set(authored) - set(declarations):
            return "NEEDS_CONTEXT: nominal parameters are absent from the registered task contract"
        expected = {name: deepcopy(authored[name] if name in authored else schema["default"])
                    for name, schema in declarations.items() if name in authored or "default" in schema}
        if witness.get("params") != expected:
            return "NEEDS_CONTEXT: nominal owner event witness changes the registered task parameter binding"
        if prepared.get("physical_evidence", {}).get("physical_preview_contract_version") == 7:
            missing = _nominal_step_coverage(events[0], witness)
            if missing:
                return missing
        supported = {**witness.get("resource_updates", {}), **witness.get("part_updates", {})}
        for field, effect in events[0].get("updates", {}).items():
            declaration = model.get("state_variables", {}).get(field, {})
            if declaration.get("private") or field.startswith("task_ctx."):
                continue
            if set(effect) == {"set"}:
                value = effect["set"]
            elif set(effect) == {"set_from_param"} and effect["set_from_param"] in expected:
                value = expected[effect["set_from_param"]]
            else:
                return "NEEDS_CONTEXT: nominal owner event effect is unsupported"
            if field not in supported or type(supported[field]) is not type(value) or supported[field] != value:
                return "NEEDS_CONTEXT: nominal owner event witness leaves a registered effect unresolved"
        return ""
    trace = ra.get("primitive_support", {}).get("trace", [])
    completions = [row for row in trace if row.get("task_contract") is not None]
    if len(events) != 1 or len(completions) != 1 or not trace or completions[0] != trace[-1]:
        return "NEEDS_CONTEXT: nominal continuation requires its exact completed task contract"
    event, completed = events[0], completions[0]
    if (completed.get("task_contract") != node.get("function_name")
            or completed.get("declaration_fingerprint") != recovery_validation_fingerprint(event)
            or completed.get("primitive_start_index") != 0
            or completed.get("primitive_end_index") != len(trace) - 1):
        return "NEEDS_CONTEXT: nominal witness differs from the registered task or includes additional recovery work"
    declarations = event.get("composition_contract", {}).get("params", {})
    if set(authored) - set(declarations):
        return "NEEDS_CONTEXT: nominal parameters are absent from the registered task contract"
    expected = {name: deepcopy(authored[name] if name in authored else schema["default"])
                for name, schema in declarations.items() if name in authored or "default" in schema}
    if recovery_validation_fingerprint(completed.get("params")) != recovery_validation_fingerprint(expected):
        return "NEEDS_CONTEXT: nominal witness changes the registered task parameter binding"
    return ""


def _missing_completion_facts(conditions: list, session: dict) -> list:
    aliases = {
        "resource": {"state": ("current_state", "resource_state"),
                     "current_state": ("current_state", "resource_state"),
                     "resource_state": ("resource_state", "current_state"),
                     "location": ("current_location", "resource_location"),
                     "current_location": ("current_location", "resource_location"),
                     "resource_location": ("resource_location", "current_location")},
        "part": {"state": ("current_state", "part_state"),
                 "current_state": ("current_state", "part_state"),
                 "part_state": ("part_state", "current_state"),
                 "location": ("current_location", "part_location"),
                 "current_location": ("current_location", "part_location"),
                 "part_location": ("part_location", "current_location"),
                 "holder": ("current_holder_resource_jid", "part_holder_resource_jid"),
                 "current_holder_resource_jid": ("current_holder_resource_jid", "part_holder_resource_jid"),
                 "part_holder_resource_jid": ("part_holder_resource_jid", "current_holder_resource_jid")},
    }
    if session.get("validation_contract_version") == 2:
        aliases = {
            "resource": {
                "state": ("resource_state", "current_state", "state"),
                "location": ("resource_location", "current_location", "location"),
            },
            "part": {
                "state": ("part_state", "current_state", "state"),
                "location": ("part_location", "current_location", "location"),
                "holder": ("part_holder_resource_jid", "current_holder_resource_jid"),
            },
        }
    missing = []
    for condition in conditions:
        scope, entity, field = (condition.get(key) for key in ("entity_kind", "entity", "field"))
        rows = session.get("symbolic_resources" if scope == "resource" else "symbolic_parts", {})
        row = rows.get(entity, {})
        if not any(key in row for key in aliases.get(scope, {}).get(field, (field,))):
            missing.append(deepcopy(condition))
    return missing


def _monitor_completion(session: dict, prepared: dict) -> dict | None:
    scopes = session.get("outline_monitor_context", {})
    pending = []
    for specification in prepared.get("outline_safety_specifications") or []:
        context = scopes.get(specification["source_sha256"], {})
        if "pending_rule_ids" not in context:
            return {"complete": False, "status": "NEEDS_CONTEXT",
                    "reason": "Nominal completion requires retained accepting-state evidence for every selected rule"}
        pending.extend(context["pending_rule_ids"])
    if pending:
        return {"complete": False, "status": "not_ready",
                "reason": "Selected safety requirements remain pending at nominal completion",
                "pending_rule_ids": pending}
    return None


def _completion_goals(session: dict, prepared: dict, conditions: list) -> dict | None:
    """Check final goals against supported projections after a nominal suffix."""
    from . import multi_turn as shared

    if not conditions and not (prepared.get("nominal_continuation_semantics") == "recorded_unfinished_tasks"
                               and prepared.get("physical_evidence", {}).get("requirements")):
        return {"complete": False, "status": "NEEDS_CONTEXT", "reason": "No recorded recovery completion conditions"}
    missing = _missing_completion_facts(conditions, session)
    if missing:
        return {"complete": False, "status": "NEEDS_CONTEXT",
                "reason": "Recovery completion conditions require missing recorded facts", "missing_conditions": missing}
    unmet = [deepcopy(row) for row in conditions if not shared._continuation_condition_satisfied(
        row, session_state=session, prepared_recovery_request=prepared)]
    if unmet:
        return {"complete": False, "status": "not_ready", "unmet_conditions": unmet}
    if prepared.get("nominal_continuation_semantics") == "recorded_unfinished_tasks":
        from cais_spade_llm.resources.environment_models import matches_requirement

        evidence = session.get("projected_physical_evidence") or prepared.get("physical_evidence", {})
        for part, requirements in prepared.get("physical_evidence", {}).get("requirements", {}).items():
            product = evidence.get("products", {}).get(part)
            if product is None:
                return {"complete": False, "status": "NEEDS_CONTEXT",
                        "reason": f"Recorded product completion evidence is unavailable for {part}",
                        "public_findings": [{"constraint_code": "nominal_product_completion_unavailable", "part_name": part}]}
            if any(not matches_requirement(product, requirement) for requirement in requirements):
                return {"complete": False, "status": "not_ready",
                        "reason": f"Recorded product requirements remain unfinished for {part}",
                        "public_findings": [{"constraint_code": "nominal_product_requirements_not_met", "part_name": part}]}
    return None


def _completion_from_other_owner(node: dict, binding: dict, session: dict, physical: dict) -> dict | None:
    """Match a supported completion without treating its owner assignment as a goal."""
    from cais_spade_llm.agents.central_controller._product_effect_evidence import _declared_product_effects

    params = node.get("params", {})
    part, original_owner = params.get("part_name"), params.get("resource_id")
    event = binding.get("environment_event", {})
    models = physical.get("environment_models", {})
    parts = physical.get("snapshot", {}).get("parts", {})
    geometry = physical.get("geometry", {})
    declaration = {"event_id": event.get("event_id"), "event_name": event.get("event_name"),
                   "parameters": deepcopy(params)}
    try:
        required_effects = _declared_product_effects(models, parts, geometry, original_owner, declaration)
    except (KeyError, TypeError, ValueError):
        return None
    for record in physical.get("product_completion_records", []):
        owner = record.get("resource_id")
        recorded_declaration = record.get("declaration", {})
        recorded_params = recorded_declaration.get("parameters", {})
        if (owner == original_owner or owner not in models
                or record.get("source_kind") != "resource_event_projection" or record.get("kind") != "predicted"
                or recovery_validation_fingerprint({key: value for key, value in params.items() if key != "resource_id"})
                != recovery_validation_fingerprint({key: value for key, value in recorded_params.items() if key != "resource_id"})):
            continue
        declared = [row for row in models[owner].get("events", [])
                    if row.get("event_id") == recorded_declaration.get("event_id")
                    and row.get("event_name") == recorded_declaration.get("event_name")]
        if len(declared) != 1 or record.get("declaration_fingerprint") != recovery_validation_fingerprint(declared[0]):
            continue
        try:
            effects = _declared_product_effects(models, parts, geometry, owner, recorded_declaration)
        except (KeyError, TypeError, ValueError):
            continue
        if (recovery_validation_fingerprint(effects) != recovery_validation_fingerprint(required_effects)
                or recovery_validation_fingerprint(record.get("product_effects")) != recovery_validation_fingerprint(effects)):
            continue
        destination = params.get("destination_location")
        if destination is not None and (record.get("part_location") != destination
                or parts.get(part, {}).get("location") != destination):
            continue
        accepted = [task for task in session.get("accepted_outline_prefix", [])
            if task.get("resource_jid") == physical.get("jids", {}).get(owner)
            and task.get("part_name") == part
            and record.get("task_id") == recovery_validation_fingerprint({
                field: task[field] for field in ("resource_jid", "part_name", "expected_start_state",
                    "expected_end_state", "params", "required_task_contract") if field in task})]
        if accepted:
            return record
    return None


def _fulfilled_nominal_tasks(nodes: dict, session: dict, prepared: dict) -> dict:
    """Record retained work superseded by an exact supported product completion.

    A different owner can fulfill the same declared product effects; that does
    not establish the original owner's resource effects. Resource cleanup remains.
    Cross-owner predecessor proof covers this part's preparation and recorded
    completed task IDs; separate pending product groups are not combined here.
    """
    from cais_spade_llm.resources.environment_models import matches_requirement

    physical = session.get("projected_physical_evidence") or prepared.get("physical_evidence", {})
    result = {}
    for identifier, node in nodes.items():
        try:
            _, binding = _nominal_arguments(node, prepared)
        except ValueError as exc:
            if not str(exc).startswith("NEEDS_CONTEXT:"):
                raise
            continue
        event = binding.get("environment_event", {})
        params = node.get("params", {})
        part = params.get("part_name")
        requirements = physical.get("requirements", {}).get(part)
        product = physical.get("products", {}).get(part)
        if (not requirements or product is None
                or any(not matches_requirement(product, requirement) for requirement in requirements)):
            continue
        declaration = {"event_id": event.get("event_id"), "event_name": event.get("event_name"),
                       "parameters": deepcopy(params)}
        records = [row for row in physical.get("product_completion_records", [])
            if row.get("declaration") == declaration and row.get("resource_id") == params.get("resource_id")
            and row.get("declaration_fingerprint") == binding.get("environment_event_fingerprint")
            and row.get("source_kind") == "resource_event_projection"
            and row.get("product_effects", {}).get(part, {}).get("processCompleted")]
        other_owner = False
        if not records:
            completion = _completion_from_other_owner(node, binding, session, physical)
            if completion is not None:
                records = [completion]
                other_owner = True
        if not records:
            continue
        proof = {"source_task_id": identifier, "product_completion": deepcopy(records[-1]),
                 "recorded_task": deepcopy(node), "execution_authorized": False}
        if other_owner:
            proof.update(reason="Declared product effects fulfilled by an accepted event from another resource",
                         original_environment_event_fingerprint=binding["environment_event_fingerprint"],
                         resource_effects_fulfilled=False)
        fulfilled = {identifier: proof}
        if type(node.get("sequence_index")) is not int:
            if not other_owner or set(node.get("predecessors", [])).issubset(set(prepared.get("completed_task_ids", []))):
                result.update(fulfilled)
            continue
        for prior_id, prior in nodes.items():
            previous = prior.get("params", {})
            if (prior.get("resource_jid") != node.get("resource_jid")
                    or previous.get("part_name") != part
                    or previous.get("destination_location") != params.get("destination_location")
                    or type(prior.get("sequence_index")) is not int
                    or prior["sequence_index"] >= node["sequence_index"]):
                continue
            try:
                _, earlier = _nominal_arguments(prior, prepared)
            except ValueError as exc:
                if not str(exc).startswith("NEEDS_CONTEXT:"):
                    raise
                continue
            effects = earlier.get("environment_event", {}).get("product_effects")
            if not isinstance(effects, dict) or effects.get("processCompleted") != []:
                continue
            fulfilled[prior_id] = {**deepcopy(proof), "recorded_task": deepcopy(prior),
                                   "reason": "Earlier part preparation is superseded by supported product completion"}
        if other_owner:
            supported = set(prepared.get("completed_task_ids", []))
            remaining = set(fulfilled)
            while remaining:
                ready = {key for key in remaining if set(nodes[key].get("predecessors", [])).issubset(supported)}
                if not ready:
                    break
                supported.update(ready)
                remaining -= ready
            if remaining:
                continue
        result.update(fulfilled)
    return result


async def validate_nominal_continuation(*, session: dict, prepared: dict, planner: Any) -> dict:  # noqa: C901, PLR0912 - bounded dependency search
    """Replay a finite saved suffix with the same RA and CCA authorities."""
    from . import multi_turn as shared
    from .multi_turn_validated_event import _project, validate_event

    conditions = shared._active_continuation_conditions(prepared)
    recorded_unfinished = prepared.get("nominal_continuation_semantics") == "recorded_unfinished_tasks"
    if not recorded_unfinished:
        incomplete = _completion_goals(session, prepared, conditions)
        if incomplete:
            return incomplete
    nodes = {row["id"]: deepcopy(row) for row in prepared.get("saved_pending_nodes", [])
             if isinstance(row, dict) and row.get("id") and row.get("status") not in {"completed", "done"}}
    for entry in prepared.get("recovery_resources", {}).values():
        for row in entry.get("pending_tasks", []):
            if row.get("id") and row.get("status") not in {"completed", "done"}:
                nodes.setdefault(row["id"], deepcopy(row))
    replacement_evidence = _fulfilled_nominal_tasks(nodes, session, prepared) if recorded_unfinished else {}
    replaced = set(replacement_evidence) if recorded_unfinished else {
        task for suffix in prepared.get("marked_reentry_context", {}).get("pending_suffix_summary", [])
        if suffix.get("role") == "recovery_replaced" for task in suffix.get("pending_task_ids", [])}
    pending = {key: row for key, row in nodes.items() if key not in replaced}
    if not pending:
        if "saved_pending_nodes" not in prepared:
            return {"complete": False, "status": "NEEDS_CONTEXT", "reason": "Unfinished task capture is unavailable"}
        incomplete = ((_completion_goals(session, prepared, conditions) if recorded_unfinished else None)
                      or _monitor_completion(session, prepared))
        if incomplete:
            return incomplete
        return {"complete": True, "status": "passed", "tasks": [], "replaced_task_ids": sorted(replaced),
                "replacement_evidence": replacement_evidence, "execution_authorized": False}
    external = {item for node in pending.values() for item in node.get("predecessors", [])
                if item not in nodes and item not in prepared.get("completed_task_ids", [])}
    if external:
        return {"complete": False, "status": "NEEDS_CONTEXT", "reason": "Nominal predecessor completion is unavailable",
                "missing_task_ids": sorted(external)}
    # Search branches need the current validation state, not copies of its
    # archived attempts. Those records remain on the original session.
    queue = deque([(deepcopy({key: value for key, value in session.items()
        if key not in {"turns", "nominal_continuation_evidence", "pending_outline_validation"}}),
        frozenset(replaced), [])])
    seen, failures, expansions = set(), [], 0
    while queue and expansions < 4096:
        current, completed, witness = queue.popleft()
        if set(pending).issubset(completed):
            incomplete = ((_completion_goals(current, prepared, conditions) if recorded_unfinished else None)
                          or _monitor_completion(current, prepared))
            if incomplete:
                failures.append(incomplete)
                continue
            return {"complete": True, "status": "passed", "tasks": witness,
                    "replaced_task_ids": sorted(replaced), "replacement_evidence": replacement_evidence,
                    "execution_authorized": False}
        identity = recovery_validation_fingerprint({"completed": sorted(completed),
                    "resources": current.get("symbolic_resources"), "parts": current.get("symbolic_parts"),
                    "monitor": current.get("outline_monitor_context"),
                    "physical_evidence": current.get("projected_physical_evidence")})
        if identity in seen:
            continue
        seen.add(identity)
        for identifier, node in pending.items():
            if identifier in completed or not set(node.get("predecessors", [])).issubset(completed | set(prepared.get("completed_task_ids", []))):
                continue
            if any(other not in completed and row.get("resource_jid") == node.get("resource_jid")
                   and row.get("sequence_index", 0) < node.get("sequence_index", 0) for other, row in pending.items()):
                continue
            expansions += 1
            if expansions > 4096:
                break
            public_findings = []
            task, reason = _nominal_task(node, current, prepared, public_findings)
            if task is None:
                failures.append({"task_id": identifier, "resource_jid": node.get("resource_jid"), "reason": reason,
                    "public_findings": public_findings, "projected_after_task_ids": [row["task_id"] for row in witness]})
                continue
            candidate, audit = deepcopy(current), {}
            candidate["observation_count"] = session.get("observation_count", 0)
            try:
                ra, cca = await validate_event(task=task, session=candidate, prepared=prepared, planner=planner, turn=audit)
            finally:
                session["observation_count"] = max(session.get("observation_count", 0), candidate.get("observation_count", 0))
            if not ra.get("allowed") or not cca.get("is_safe"):
                failures.append({"task_id": identifier, "reason": (ra if not ra.get("allowed") else cca).get("reason"),
                    "task": deepcopy(task), "projected_physical_evidence": deepcopy(candidate.get("projected_physical_evidence", {})),
                    "recovery_snapshot": deepcopy(candidate.get("derived_resource_snapshots", {}).get(task.get("resource_jid"),
                        prepared.get("recovery_resources", {}).get(task.get("resource_jid"), {}).get("recovery_snapshot", {}))),
                    "validation_stages": audit.get("validation_stages", []),
                    "projected_after_task_ids": [row["task_id"] for row in witness]})
                continue
            reason = _exact_nominal_witness(node, ra, prepared)
            if reason:
                failures.append({"task_id": identifier, "reason": reason})
                continue
            _project(task, ra, cca, candidate)
            queue.append((candidate, completed | {identifier}, [*witness, {
                "task_id": identifier, "task": deepcopy(task), **audit}]))
    missing, other_missing, obstructed = False, False, False
    revision_identities = []
    for failure in failures:
        final_stages = {stage.get("validator_role"): stage.get("evidence") or {}
                        for stage in failure.get("validation_stages", [])}
        obstructed |= any(row.get("feasibility_status") == "INFEASIBLE"
                          or row.get("validation_status") in {"INFEASIBLE", "rejected"}
                          for row in final_stages.values()) or any(
            row.get("constraint_code") == "nominal_task_condition_not_met"
            for row in failure.get("public_findings", []))
        missing_fact = (failure.get("status") == "NEEDS_CONTEXT"
            or str(failure.get("reason", "")).startswith("NEEDS_CONTEXT:")
            or any(row.get("feasibility_status") == "NEEDS_CONTEXT" or
                   row.get("validation_status") in {"NEEDS_CONTEXT", "unresolved", "unavailable"}
                   for row in final_stages.values()))
        motion = final_stages.get("CentralControllerAgent", {})
        if (motion.get("physical_motion_unresolved") is True and motion.get("revision_identity")
                and final_stages.get("ResourceAgent", {}).get("allowed") is True):
            revision_identities.append(motion["revision_identity"])
        elif missing_fact:
            other_missing = True
        missing |= missing_fact
    # A demonstrated obstruction is enough to request another recovery event.
    # Retain other unresolved checks; they still must pass before continuation.
    return {"complete": False, "status": "NEEDS_CONTEXT" if queue or (missing and not obstructed) else "not_ready",
            "reason": "Nominal continuation is not established", "failures": failures,
            "proposal_revision_identities": list(dict.fromkeys(revision_identities)) if not queue and not other_missing else [],
            "expanded_states": expansions, "limit_reached": bool(queue)}
