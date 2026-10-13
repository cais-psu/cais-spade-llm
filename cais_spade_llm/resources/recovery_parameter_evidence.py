from __future__ import annotations

"""Resolve resource-owned parameter facts without composing primitive effects."""

import math
from copy import deepcopy
from typing import Any

from cais_spade_llm.agents.shared_information.recovery_validation_protocol import (
    recovery_validation_fingerprint,
)
from cais_spade_llm.resources.recovery_feasibility import derive_primitive_union


def projected_resource_evidence(*, resource_jid: str, resource_evidence: dict,
                                physical_evidence: dict, recovery_snapshot: dict) -> dict:
    """Overlay an accepted projected scene without relabelling it as observed."""
    result = deepcopy(resource_evidence)
    source = physical_evidence.get("provenance") or {}
    if source.get("kind") != "projected" or not source.get("contract_fingerprint"):
        return result
    ids = [rid for rid, jid in physical_evidence.get("jids", {}).items() if jid == resource_jid]
    if len(ids) != 1:
        return result
    snapshot = physical_evidence.get("snapshot") or {}
    owner = snapshot.get("resources", {}).get(ids[0])
    if not isinstance(owner, dict):
        return result
    if "held_part" in recovery_snapshot and owner.get("held_part") != recovery_snapshot["held_part"]:
        result.setdefault("conflicts", []).append({"ref": f"/resources/{resource_jid}/resource_evidence/attachment",
            "reason": "Projected physical custody differs from the validating resource snapshot"})
        return result
    recorded_owner = result.get("snapshot") or {}
    unchanged_owner = all(field in recorded_owner and field in owner and recorded_owner[field] == owner[field]
                          for field in ("current_pose", "held_part", "grasp_transform"))
    result["recorded_snapshot"] = deepcopy(result.get("snapshot"))
    result["snapshot"] = {**deepcopy(recovery_snapshot), **deepcopy(owner)}
    result["projection_source"] = deepcopy(source)
    state = result.setdefault("state", {})
    state["_position"] = deepcopy(owner.get("current_pose"))
    if "held_part" in owner:
        state["_held_part"] = deepcopy(owner["held_part"])
        result["recorded_attachment"] = deepcopy(result.get("attachment"))
        part = owner["held_part"]
        part_row = snapshot.get("parts", {}).get(part, {})
        result["attachment"] = {"part_name": part, "model_name": part_row.get("model_name"),
            "grasp_transform": deepcopy(owner.get("grasp_transform")),
            "projected_attachment_complete": bool(part is not None and owner.get("grasp_transform") is not None),
            "source": deepcopy(source)}
        # Controller task context describes an executed task; it cannot stand in
        # for a grasp or destination reached by a merely projected event.
        if not unchanged_owner:
            result["recorded_task_context"] = deepcopy(state.get("_task_ctx"))
            state.pop("_task_ctx", None)
        for field, value in owner.items():
            if field.startswith("task_ctx."):
                state.setdefault("_task_ctx", {})[field.removeprefix("task_ctx.")] = deepcopy(value)
    result.setdefault("recorded_observed_parts", deepcopy(result.get("observed_parts")))
    result["observed_parts"] = {part: {**deepcopy(row), "source": deepcopy(source)}
                                for part, row in snapshot.get("parts", {}).items()}
    for row in result["observed_parts"].values():
        row.pop("observed_pose", None)
    return result


def _reference(root: str, path: list) -> str:
    return root + "".join("/" + str(key).replace("~", "~0").replace("/", "~1") for key in path)


def _resolve(value: Any, bindings: dict, state: dict, root: str, step_outputs: dict | None = None) -> tuple[bool, Any, list[str]]:
    """Evaluate the supported declared expression subset against one snapshot."""
    if isinstance(value, dict):
        if set(value) == {"$arg"} and isinstance(value["$arg"], str):
            name = value["$arg"]
            return name in bindings, deepcopy(bindings.get(name)), []
        reference_kind = "$state" if "$state" in value else "$step" if "$step" in value else None
        if reference_kind and set(value) <= {reference_kind, "path"} and isinstance(value.get(reference_kind), str):
            path = value.get("path", [])
            if not isinstance(path, list) or any(not isinstance(key, (str, int)) or isinstance(key, bool) for key in path):
                return False, None, [root + "/state"]
            keys = [value[reference_kind], *path]
            current = state if reference_kind == "$state" else step_outputs or {}
            source_root = root + ("/state" if reference_kind == "$state" else "/step_outputs")
            for key in keys:
                if isinstance(current, dict) and key in current:
                    current = current[key]
                elif isinstance(current, list) and type(key) is int and 0 <= key < len(current):
                    current = current[key]
                else:
                    return False, None, [_reference(source_root, keys)]
            return True, deepcopy(current), []
        if set(value) == {"$sub"} and isinstance(value["$sub"], list) and len(value["$sub"]) == 2:
            results = [_resolve(item, bindings, state, root, step_outputs) for item in value["$sub"]]
            missing = [ref for _, _, refs in results for ref in refs]
            if all(known and type(item) in (int, float) and math.isfinite(item) for known, item, _ in results):
                return True, results[0][1] - results[1][1], []
            return False, None, missing or [root + "/state"]
        if any(str(key).startswith("$") for key in value):
            return False, None, [root + "/state"]
        result, missing, complete = {}, [], True
        for key, child in value.items():
            known, resolved, refs = _resolve(child, bindings, state, root, step_outputs)
            complete &= known
            result[key] = resolved
            missing.extend(refs)
        return complete, result, missing
    if isinstance(value, list):
        values = [_resolve(child, bindings, state, root, step_outputs) for child in value]
        return (all(known for known, _, _ in values), [item for _, item, _ in values],
                [ref for _, _, refs in values for ref in refs])
    return True, deepcopy(value), []


def evaluate_resource_primitive_condition(*, step: dict, bindings: dict, snapshot: dict,
                                         resource_evidence: dict, resource_jid: str) -> dict:
    """Evaluate supported optional guards using one snapshot and owner facts."""
    root = f"/resources/{resource_jid}/resource_evidence"
    state = deepcopy(resource_evidence.get("state") or {})
    state.update(deepcopy(snapshot))
    # This is the public condition spelling declared by the robot task model.
    if "_task_ctx" in state:
        state["task_ctx"] = deepcopy(state["_task_ctx"])
    checks, missing = [], []
    when = step.get("when", [])
    if not isinstance(when, list):
        return {"status": "NEEDS_CONTEXT", "reason": "Declared when must be a list"}
    for guard in when:
        if not isinstance(guard, dict):
            checks.append(None)
            continue
        predicate = guard.get("predicate", "always")
        if predicate not in ("", "always"):
            if resource_evidence.get("resource_type") == "robot":
                from cais_spade_llm.resources.robot.recovery_capability import evaluate_robot_primitive_predicate
                checked = evaluate_robot_primitive_predicate(predicate=predicate, resource_evidence=resource_evidence,
                                                            recovery_snapshot=snapshot, resource_jid=resource_jid)
                checks.append(checked.get("value"))
                missing.extend(checked.get("context_requests") or [])
            else:
                checks.append(None)
        condition = guard.get("condition", {})
        if not condition:
            if predicate in ("", "always"):
                checks.append(True)
            continue
        if not isinstance(condition, dict) or set(condition) - {"field", "operator", "value"}:
            checks.append(None)
            continue
        path = condition.get("field")
        if not isinstance(path, str) or not path:
            checks.append(None)
            continue
        keys = path.split(".")
        value, known = state, True
        for key in keys:
            if not isinstance(value, dict) or key not in value:
                known = False
                break
            value = value[key]
        if not known:
            checks.append(None)
            source_keys = ["_task_ctx" if keys[0] == "task_ctx" else keys[0], *keys[1:]]
            missing.append(_reference(root + "/state", source_keys))
            continue
        operator = condition.get("operator")
        if operator == "exists":
            checks.append(value is not None)
        elif operator in ("equals", "not_equals") and "value" in condition:
            resolved, expected, refs = _resolve(condition["value"], bindings, state, root, resource_evidence.get("step_outputs"))
            missing.extend(refs)
            equal = type(value) is type(expected) and value == expected
            checks.append((equal if operator == "equals" else not equal) if resolved else None)
        else:
            checks.append(None)
    # Unsupported conditions are kept unresolved even alongside a false term.
    status = "NEEDS_CONTEXT" if None in checks else "not_applicable" if False in checks else "applicable"
    return {"status": status, "reason": {"NEEDS_CONTEXT": "Declared optional condition lacks supported facts or semantics",
        "not_applicable": "Declared optional condition is false against the starting facts",
        "applicable": "Declared optional condition is true against the starting facts"}[status],
        "when": deepcopy(when), "context_requests": list(dict.fromkeys(missing)),
        "source_fingerprint": recovery_validation_fingerprint(resource_evidence),
        "snapshot_fingerprint": recovery_validation_fingerprint(snapshot), "command_sent": False}


def prepare_resource_primitive_parameters(
    *, task: dict, recovery_des_model: dict, recovery_snapshot: dict, resource_evidence: dict,
    resource_jid: str, calculated_step_outputs: list[dict] | None = None,
) -> dict:
    """Prepare scoped facts without advancing state or invoking another primitive.

    Only recorded state and declared bindings are consumed. Resource owners may
    provide calculated state facts with separate provenance before this call.
    Missing facts remain explicit; a null fact is never replaced with a default.
    """
    root = f"/resources/{resource_jid}/resource_evidence"
    support = derive_primitive_union(task=task, recovery_des_model=recovery_des_model, recovery_snapshot=recovery_snapshot)
    catalog = {row.get("name"): row for row in recovery_des_model.get("primitive_catalog", []) if isinstance(row, dict)}
    raw_state = resource_evidence.get("state")
    state = deepcopy(raw_state) if isinstance(raw_state, dict) else {}
    # These existing owner fields name the same starting snapshot. No primitive
    # effect or an earlier check's hypothetical result is applied here.
    for name, field in (("_held_part", "held_part"), ("_position", "current_pose")):
        if field in recovery_snapshot:
            state[name] = deepcopy(recovery_snapshot[field])
            if (name == "_position" and resource_evidence.get("resource_type") == "robot"
                    and isinstance(state[name], list) and len(state[name]) == 7):
                state[name] = dict(zip(("x", "y", "z", "qx", "qy", "qz", "qw"), state[name], strict=True))
    witnesses, missing, conflicts = [], [], []
    for primitive in support["primitives"]:
        for contributor in primitive["contributors"]:
            condition = evaluate_resource_primitive_condition(step=contributor["step"],
                bindings=contributor["parameter_bindings"], snapshot=recovery_snapshot,
                resource_evidence=resource_evidence, resource_jid=resource_jid)
            if condition["status"] == "not_applicable":
                continue
            params, omitted, references = {}, [], []
            bindings = contributor["parameter_bindings"]
            default_evidence = {}
            resolved_bindings = deepcopy(bindings)
            if resource_evidence.get("resource_type") == "robot":
                from cais_spade_llm.resources.robot.recovery_task_projection import nominal_task_bindings

                declared = nominal_task_bindings(task={
                    "required_task_contract": contributor["event_name"], "params": bindings},
                    recovery_des_model=recovery_des_model)
                if declared["status"] == "supported":
                    resolved_bindings = declared["bindings"]
                    default_evidence = declared["default_evidence"]
            outputs = deepcopy(resource_evidence.get("step_outputs") or {})
            calculations = [row for row in calculated_step_outputs or []
                            if row["event_name"] == contributor["event_name"]
                            and row["parameter_bindings"] == bindings]
            conflicting_outputs = set()
            for calculation in calculations:
                name, value = calculation["store_as"], calculation["output"]
                if name in outputs and outputs[name] != value:
                    conflicting_outputs.add(name)
                    conflicts.append({"ref": _reference(root + "/step_outputs", [name]),
                        "event_name": contributor["event_name"], "step_index": contributor["step_index"],
                        "reason": "Recorded or calculated outputs conflict for the same contributor"})
                else:
                    outputs[name] = deepcopy(value)
            for name in conflicting_outputs:
                outputs.pop(name, None)
            authored = contributor["step"].get("params") or {}
            required = catalog.get(primitive["primitive"], {}).get("required_params", [])
            for name, expression in authored.items():
                # Applicability is established by the robot owner's primitive
                # contract, never by the proposed event's arbitrary identifier.
                applicability = resource_evidence.get("parameter_applicability", {}).get(primitive["primitive"], {}).get(name)
                if isinstance(applicability, dict) and applicability.get("applicable") is False:
                    omitted.append(name)
                    continue
                known, value, refs = _resolve(expression, resolved_bindings, state, root, outputs)
                references.extend(refs)
                if known:
                    params[name] = value
                elif name in bindings:
                    params[name] = deepcopy(bindings[name])
                elif name in required:
                    missing.extend({"ref": ref, "primitive": primitive["primitive"],
                                    "event_name": contributor["event_name"], "step_index": contributor["step_index"],
                                    "parameter": name, "reason": "Required declared parameter source is unavailable"}
                                   for ref in refs or [root + "/state"])
            # A partial row can establish controller identity/custody while
            # another required coordinate remains explicitly unavailable.
            if params or omitted:
                row = {"primitive": primitive["primitive"], "event_name": contributor["event_name"],
                       "step_index": contributor["step_index"], "params": params, "source": root,
                       "parameter_bindings": deepcopy(bindings),
                       "source_kind": "resource_owned_calculation", "source_capture": deepcopy(resource_evidence.get("source") or {}),
                       "snapshot_fingerprint": recovery_validation_fingerprint(recovery_snapshot),
                       "model_fingerprint": recovery_validation_fingerprint(recovery_des_model),
                       "source_fingerprint": recovery_validation_fingerprint(resource_evidence),
                       "omitted_params": omitted, "missing_source_references": list(dict.fromkeys(references)),
                       "sources": deepcopy(resource_evidence.get("sources") or {})}
                if calculations:
                    row["step_output_calculations"] = deepcopy(calculations)
                if default_evidence:
                    row["parameter_default_evidence"] = deepcopy(default_evidence)
                if omitted:
                    row["omission_reason"] = "; ".join(resource_evidence["parameter_applicability"][primitive["primitive"]][name]["reason"] for name in omitted)
                witnesses.append(row)
    return {"parameter_witnesses": witnesses, "missing_facts": missing,
            "context_requests": list(dict.fromkeys(row["ref"] for row in missing)),
            "resource_evidence_fingerprint": recovery_validation_fingerprint(resource_evidence),
            "calculated_resource_evidence": deepcopy(resource_evidence), "conflicts": conflicts}
