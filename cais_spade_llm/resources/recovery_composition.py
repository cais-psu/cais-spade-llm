from __future__ import annotations

"""Validate an authored primitive program against its owning resource contracts."""

from collections.abc import Callable
from copy import deepcopy
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError

from cais_spade_llm.agents.shared_information.recovery_validation_protocol import recovery_validation_fingerprint
from cais_spade_llm.resources import recovery_transition_derivation as contracts
from cais_spade_llm.resources.resource_primitives import filter_recovery_primitive_catalog

COMPOSITION_VALIDATION_VERSION = 1


def resolve_composition_value(value: Any, context: dict) -> Any:
    """Resolve explicit references using exact keys, preserving literal strings.

    Args:
        value: An authored parameter value or an explicit context_ref object.
        context: Recorded facts and outputs from earlier steps only.

    Returns:
        A detached value with explicit references resolved.

    Raises:
        KeyError: A referenced fact is missing.
        ValueError: A reference is malformed.
    """
    if isinstance(value, dict):
        if "context_ref" in value:
            if set(value) != {"context_ref"} or not isinstance(value["context_ref"], str):
                raise ValueError("A parameter reference requires only context_ref with an exact path")
            ref = value["context_ref"]
            tokens = ref[1:].split("/") if ref.startswith("/") else ref.split(".")
            current = context
            for token in tokens:
                if isinstance(current, dict) and token in current:
                    current = current[token]
                elif isinstance(current, list) and token.isdecimal() and int(token) < len(current):
                    current = current[int(token)]
                else:
                    raise KeyError(ref)
            return deepcopy(current)
        return {key: resolve_composition_value(item, context) for key, item in value.items()}
    if isinstance(value, list):
        return [resolve_composition_value(item, context) for item in value]
    return deepcopy(value)


def composition_start_views(*, session: dict, prepared: dict, task: dict) -> tuple[dict, dict]:
    """Restore the checkpoint plus accepted composition effects in outline order."""
    jid, part = task["resource_jid"], task.get("part_name")
    entry = prepared.get("recovery_resources", {}).get(jid, {})
    resource = deepcopy(entry.get("recovery_snapshot") or {})
    resource.update(deepcopy(session.get("base_symbolic_resources", {}).get(jid, {})))
    part_context = deepcopy(prepared.get("grounding_context", {}).get("parts", {}).get(part, {}))
    part_context.update(deepcopy(session.get("base_symbolic_parts", {}).get(part, {})))
    for row in [*session.get("composition_prefix", []), *session.get("accepted_primitive_program", [])]:
        evidence = row.get("composition_validation", {})
        if evidence.get("feasibility_status") != "FEASIBLE":
            continue
        if row.get("resource_jid") == jid:
            resource = deepcopy(evidence["projected_snapshot"])
        if jid in evidence.get("projected_resources", {}):
            resource.update(deepcopy(evidence["projected_resources"][jid]))
        if part is not None and row.get("part_name") == part:
            part_context = deepcopy(evidence["projected_part_context"])
        if part in evidence.get("projected_parts", {}):
            part_context.update(deepcopy(evidence["projected_parts"][part]))
    return resource, part_context


def composition_grounding_context(*, session: dict, prepared: dict, task: dict) -> dict:
    """Expose recorded contracts and the current accepted state to retrieval."""
    context = deepcopy(prepared.get("grounding_context", {}))
    resources, parts = {}, deepcopy(context.get("parts", {}))
    for jid, entry in prepared.get("recovery_resources", {}).items():
        resource, _ = composition_start_views(session=session, prepared=prepared,
                                             task={"resource_jid": jid})
        resources[jid] = {**deepcopy(entry), **resource, "snapshot": deepcopy(resource)}
        # Step templates are served from code under /capability_decompositions. The template fields a
        # capture recorded inside its DES model can be stale, so this served copy drops them; the owner
        # check reads the prepared model, which keeps them.
        for event in (resources[jid].get("recovery_des_model") or {}).get("events") or []:
            event.pop("recovery_visible_steps", None)
            for step in (event.get("composition_contract") or {}).get("steps") or []:
                step.pop("public_params", None)
                step.pop("note", None)
    part_names = set(parts) | set(session.get("base_symbolic_parts", {}))
    for row in [*session.get("composition_prefix", []), *session.get("accepted_primitive_program", [])]:
        part_names.update(row.get("composition_validation", {}).get("projected_parts", {}))
        if row.get("part_name") is not None:
            part_names.add(row["part_name"])
    for part in part_names:
        _, parts[part] = composition_start_views(session=session, prepared=prepared,
            task={"resource_jid": task["resource_jid"], "part_name": part})
    context.update(resources=resources, parts=parts, outline_event=deepcopy(task),
        active_outline_event=deepcopy(task), resource=resources.get(task["resource_jid"], {}),
        part=parts.get(task.get("part_name"), {}), resources_by_jid=resources, parts_by_name=parts)
    for row in [*session.get("composition_prefix", []), *session.get("accepted_primitive_program", [])]:
        evidence = row.get("composition_validation", {})
        if evidence.get("feasibility_status") == "FEASIBLE" and "resource_evidence" in evidence:
            resources[row["resource_jid"]]["resource_evidence"] = deepcopy(evidence["resource_evidence"])
    return context


def composition_program_error(events: list, program: list, *, prepared: dict, session: dict,
                              complete: bool = True) -> str | None:
    """Check ordered coverage and evidence bindings without provider or equipment calls."""
    if not isinstance(events, list) or not isinstance(program, list) or any(not isinstance(row, dict) for row in [*events, *program]):
        return "The outline and composition must be lists of event objects"
    ids = [event.get("outline_id") for event in events]
    if not ids or any(not isinstance(value, str) or not value for value in ids) or len(set(ids)) != len(ids):
        return "The accepted outline requires unique, nonempty event identities"
    if len(program) > len(events) or (complete and len(program) != len(events)):
        return "Every accepted outline event requires exactly one validated program"
    prefix = {**session, "composition_prefix": [], "accepted_primitive_program": []}
    for task, row in zip(events, program):
        evidence = row.get("composition_validation", {})
        if any(row.get(key) != task.get(key) for key in ("outline_id", "resource_jid", "part_name")):
            return "Composition identities or order differ from the accepted outline"
        if evidence.get("composition_validation_version") != COMPOSITION_VALIDATION_VERSION or evidence.get("feasibility_status") != "FEASIBLE":
            return f"{task['outline_id']}: current RA feasibility evidence is missing"
        if evidence.get("evidence_fingerprint") != recovery_validation_fingerprint(
                {key: value for key, value in evidence.items() if key != "evidence_fingerprint"}):
            return f"{task['outline_id']}: RA evidence content differs from its recorded fingerprint"
        entry = prepared.get("recovery_resources", {}).get(task["resource_jid"], {})
        resource, part = composition_start_views(session=prefix, prepared=prepared, task=task)
        # The owner validated against its recovery-visible catalog, without internal gripper controls.
        catalog = filter_recovery_primitive_catalog(entry.get("execution_primitive_catalog", []),
                                                    resource_type=entry.get("resource_type", ""))
        bindings = {"task": task, "program": row.get("primitive_steps"),
                    "model": entry.get("recovery_des_model", {}), "catalog": catalog, "start": [resource, part]}
        if any(evidence.get(key + "_fingerprint") != recovery_validation_fingerprint(value)
               for key, value in bindings.items()):
            return f"{task['outline_id']}: composition evidence does not bind the current contracts and prefix"
        if evidence.get("derived_end_state") != task.get("expected_end_state"):
            return f"{task['outline_id']}: not all intended postconditions were established"
        prefix["accepted_primitive_program"].append(row)
    return None


def _declared_value(field: str, declaration: dict, resource: dict, part: dict) -> Any:
    """Read a declared condition; a derived one, such as pose, is the effect field it follows, on its fields."""
    if "effect" not in declaration:
        return contracts._read_initial(field, declaration.get("scope", "resource"), resource, part)
    source = contracts._read_initial(declaration["effect"], "resource", resource, part)
    fields = declaration.get("fields") or []
    if not isinstance(source, dict) or not fields or any(name not in source for name in fields):
        return contracts._MISSING
    return {name: deepcopy(source[name]) for name in fields}


def _on_declared_fields(declaration: dict, value: Any) -> Any:
    """Restrict a stated value of a derived condition to the fields the condition declares."""
    if "effect" in declaration and isinstance(value, dict):
        return {name: value[name] for name in declaration.get("fields") or [] if name in value}
    return value


def _declares(catalog: list, field: str, value: Any) -> bool:
    """Return whether a primitive of the resource declares an effect that may set ``field``, or its alias, to value."""
    names = (field, *contracts._RESOURCE_NAMES.get(field, ()))
    for row in catalog:
        for name in names:
            effect = (row.get("effects") or {}).get(name) if isinstance(row, dict) else None
            if isinstance(effect, dict) and (("set" in effect and contracts._equal(effect["set"], value)) or (
                    value is not None and bool(set(effect) & {"set_from_param", "set_from_param_any_of"}))):
                return True
    return False


def _contract_refusal(name: str, contract: dict, state: dict, params: dict, declarations: dict) -> tuple[str, str] | None:
    """Return why one declared contract of a primitive does not apply at a step, or None when it applies."""
    for parameter, schema in (contract.get("parameters") or {}).items():
        if parameter not in params:
            continue
        value = params[parameter]
        if "enum" in schema and not any(contracts._equal(value, item) for item in schema["enum"]):
            return "INFEASIBLE", f"{name}: {parameter} {value!r} is outside its declared values {schema['enum']}"
        numeric = isinstance(value, (int, float)) and not isinstance(value, bool)
        if "minimum" in schema and not (numeric and value >= schema["minimum"]):
            return "INFEASIBLE", f"{name}: {parameter} {value!r} is below its declared minimum {schema['minimum']}"
        if "maximum" in schema and not (numeric and value <= schema["maximum"]):
            return "INFEASIBLE", f"{name}: {parameter} {value!r} is above its declared maximum {schema['maximum']}"
    for group in ("preconditions", "capability_constraints", "product_guards"):
        passed, error = contracts._conditions(contract.get(group, {}), state, params, declarations)
        if not passed:
            return ("NEEDS_CONTEXT", error) if error else ("INFEASIBLE", f"{name}: {group} are not satisfied")
    return None


def validate_recovery_primitive_composition(
    *, task: dict, primitive_steps: list, recovery_des_model: dict, recovery_snapshot: dict,
    part_context: dict, grounding_context: dict, primitive_catalog: list,
    validate_primitive: Callable[..., dict] | None,
    preview_primitive: Callable[..., dict] | None = None,
    project_completion: Callable[..., dict] | None = None,
    captured_snapshot: dict | None = None,
) -> dict:
    """Check the exact authored order without execution or goal-state substitution.

    Args:
        task: Accepted recovery event with declared start and end conditions.
        primitive_steps: LLM-authored ordered primitives and parameter bindings.
        recovery_des_model: The bound owner's state and transition declarations.
        recovery_snapshot: Recorded or previously validated resource state.
        part_context: Recorded or previously validated product state.
        grounding_context: Facts available for explicit parameter retrieval.
        primitive_catalog: Exact execution contracts owned by the resource.
        validate_primitive: Nonexecuting capability evaluator at each step's state.
        preview_primitive: Owner calculation of required data-producing outputs.
        project_completion: Owner proof of additional complete-program effects.
        captured_snapshot: The owner's state in the capture, before any recovery program.

    Returns:
        FEASIBLE, INFEASIBLE, or NEEDS_CONTEXT with a bound step trace. Passing
        establishes declared composition feasibility, never CCA or execution approval.
    """
    declarations = recovery_des_model.get("state_variables")
    result = {
        "composition_validation_version": COMPOSITION_VALIDATION_VERSION,
        "outline_id": task.get("outline_id"), "resource_jid": task.get("resource_jid"),
        "task_fingerprint": recovery_validation_fingerprint(task),
        "program_fingerprint": recovery_validation_fingerprint(primitive_steps),
        "model_fingerprint": recovery_validation_fingerprint(recovery_des_model),
        "catalog_fingerprint": recovery_validation_fingerprint(primitive_catalog),
        "start_fingerprint": recovery_validation_fingerprint([recovery_snapshot, part_context]),
        "trace": [], "findings": [], "derived_end_state": {},
        "execution_authorized": False, "cca_status": "not_checked",
    }

    def finish(status: str, reason: str, *, index: int | None = None) -> dict:
        result.update(feasibility_status=status, allowed=status == "FEASIBLE", reason=reason)
        if status != "FEASIBLE":
            result["findings"].append({"constraint_owner": "resource", "step_index": index,
                "constraint_code": "composition_missing_context" if status == "NEEDS_CONTEXT" else "composition_infeasible",
                "reason": reason})
        result["evidence_fingerprint"] = recovery_validation_fingerprint(result)
        return deepcopy(result)

    if not isinstance(declarations, dict) or not declarations or not primitive_catalog:
        return finish("NEEDS_CONTEXT", "Owner state declarations and execution primitive contracts are required")
    if any(not isinstance(value, dict) or value.get("scope", "resource") not in {"resource", "part"}
           for value in declarations.values()):
        return finish("NEEDS_CONTEXT", "Owner state declarations have an unsupported scope")
    if not isinstance(task.get("expected_start_state"), dict):
        return finish("NEEDS_CONTEXT", "The accepted event has no declared starting conditions")
    if not isinstance(primitive_steps, list) or not primitive_steps:
        return finish("INFEASIBLE", "A nonempty ordered primitive program is required")
    if not callable(validate_primitive):
        return finish("NEEDS_CONTEXT", "The responsible RA has no composition capability evaluator")
    # A primitive may declare alternative contracts, such as a machine's job start for each program.
    catalog: dict[str, list[dict]] = {}
    for row in primitive_catalog:
        if not isinstance(row, dict) or not isinstance(row.get("name"), str):
            return finish("NEEDS_CONTEXT", "Owner primitive identities are missing")
        catalog.setdefault(row["name"], []).append(row)
    fields = set(declarations)
    for row in (variant for variants in catalog.values() for variant in variants):
        for key in ("preconditions", "capability_constraints", "effects", "product_guards", "product_effects"):
            if not isinstance(row.get(key, {}), dict):
                return finish("NEEDS_CONTEXT", f"Malformed owner {key} for {row['name']!r}")
            fields.update(row.get(key, {}))
    state = {"resource": {}, "part": {}, "unknown": {"resource": [], "part": []}}
    for field in fields:
        scope, name = contracts._field(field, declarations)
        if scope not in {"resource", "part"}:
            return finish("NEEDS_CONTEXT", f"Unsupported declared scope for {field!r}")
        value = contracts._read_initial(name, scope, recovery_snapshot, part_context)
        if value is not contracts._MISSING:
            state[scope][name] = value
    for field, expected in task.get("expected_start_state", {}).items():
        declaration = declarations.get(field)
        if not isinstance(declaration, dict):
            return finish("NEEDS_CONTEXT", f"Starting field {field!r} is not declared")
        scope = declaration.get("scope", "resource")
        actual = (_declared_value(field, declaration, recovery_snapshot, part_context) if "effect" in declaration
                  else state[scope].get(field, contracts._MISSING))
        if actual is contracts._MISSING:
            return finish("NEEDS_CONTEXT", f"Starting fact {field!r} is missing")
        stated = _on_declared_fields(declaration, expected)
        if not contracts._equal(actual, stated):
            captured = (_declared_value(field, declaration, captured_snapshot, part_context)
                        if "effect" in declaration and isinstance(captured_snapshot, dict) else contracts._MISSING)
            # The outline carries a resource's captured pose until one of its events states another, while a
            # program accepted earlier has moved it; the program then starts where that program left it.
            if captured is not contracts._MISSING and contracts._equal(captured, stated) and not contracts._equal(captured, actual):
                continue
            return finish("INFEASIBLE", f"Starting fact {field!r} conflicts with the validated composition prefix")

    context = deepcopy(grounding_context)
    context["event_facts"] = {}
    facts: set[tuple] = set()
    written: set[str] = set()
    for index, step in enumerate(primitive_steps):
        if (not isinstance(step, dict) or set(step) - {"primitive", "params"}
                or not isinstance(step.get("primitive"), str) or not isinstance(step.get("params"), dict)):
            return finish("INFEASIBLE", "Each step requires an exact primitive name and params object", index=index)
        name = step["primitive"]
        variants = catalog.get(name)
        if variants is None:
            return finish("INFEASIBLE", f"Primitive {name!r} is absent from this owner's catalog", index=index)
        entry = variants[0]
        resource, part = contracts._runtime_views(state, recovery_snapshot, part_context)
        owner_context = context.setdefault("resources", {}).setdefault(task["resource_jid"], {})
        owner_context, _ = contracts._runtime_views(state, owner_context, part_context)
        owner_context.update(deepcopy(resource))
        owner_context["snapshot"] = deepcopy(resource)
        context["resources"][task["resource_jid"]] = owner_context
        context["resource"] = deepcopy(owner_context)
        context["part"] = deepcopy(part)
        if task.get("part_name") is not None:
            context.setdefault("parts", {})[task["part_name"]] = deepcopy(part)
        try:
            params = resolve_composition_value(step["params"], context)
        except KeyError as exc:
            return finish("NEEDS_CONTEXT", f"Unresolved parameter reference {exc}", index=index)
        except ValueError as exc:
            return finish("INFEASIBLE", str(exc), index=index)
        schema = {"type": "object", "properties": entry.get("params", {}),
                  "required": entry.get("required_params", []), "additionalProperties": False}
        try:
            Draft202012Validator.check_schema(schema)
        except SchemaError as exc:
            return finish("NEEDS_CONTEXT", f"Malformed primitive parameter schema: {exc.message}", index=index)
        errors = list(Draft202012Validator(schema).iter_errors(params))
        if errors:
            return finish("INFEASIBLE", errors[0].message, index=index)
        # A contract's part_name is the event's part unless the program binds it, as Algorithm 1 binds it.
        bound = {**({"part_name": task["part_name"]} if task.get("part_name") is not None else {}), **params}
        refusals = [_contract_refusal(name, variant, state, bound, declarations) for variant in variants]
        if all(refusals):
            status = "INFEASIBLE" if all(kind == "INFEASIBLE" for kind, _ in refusals) else "NEEDS_CONTEXT"
            reasons = [reason for _, reason in refusals]
            return finish(status, reasons[0] if len(reasons) == 1 else
                          f"{name}: no declared alternative applies: " + "; ".join(reasons), index=index)
        entry = variants[refusals.index(None)]
        row = {"step_index": index, "primitive": name, "params": deepcopy(step["params"]),
               "resolved_params": deepcopy(params), "start_snapshot": resource}
        if entry.get("controllable") is False:
            # The resource's own outcome is waited for, never commanded, so no command capability is checked; its
            # declared preconditions and guards above are the conditions under which it happens.
            row.update(controllable=False, after_sec=deepcopy(entry.get("after_sec")))
        else:
            capability = validate_primitive(primitive=name, params=deepcopy(params),
                                           recovery_snapshot=resource, part_context=part)
            if not isinstance(capability, dict) or capability.get("feasibility_status") != "FEASIBLE" or capability.get("allowed") is not True:
                capability = capability if isinstance(capability, dict) else {}
                return finish("INFEASIBLE" if capability.get("feasibility_status") == "INFEASIBLE" else "NEEDS_CONTEXT",
                              capability.get("reason") or f"{name}: capability evidence is unavailable", index=index)
            row["capability_evidence"] = deepcopy(capability)
        def references(value):
            if isinstance(value, dict):
                if "context_ref" in value:
                    yield value["context_ref"]
                else:
                    for item in value.values():
                        yield from references(item)
            elif isinstance(value, list):
                for item in value:
                    yield from references(item)
        row["context_refs"] = list(references(step["params"]))
        result["trace"].append(row)
        output = preview_primitive(primitive=name, params=deepcopy(params), recovery_snapshot=resource,
            part_context=part, grounding_context=deepcopy(context)) if callable(preview_primitive) else {}
        if output.get("status") == "NEEDS_CONTEXT":
            return finish("NEEDS_CONTEXT", output.get("reason") or f"{name}: output evidence is missing", index=index)
        if output.get("event_fact_key"):
            target = context["event_facts"]
            tokens = output["event_fact_key"].split(".")
            for token in tokens[:-1]:
                target = target.setdefault(token, {})
            target[tokens[-1]] = deepcopy(output["output"])
            row.update(event_fact_path="event_facts." + output["event_fact_key"], preview_output=deepcopy(output["output"]))
        requirements = entry.get("required_evidence") or {}
        if not isinstance(requirements, dict) or any(not isinstance(requirements.get(key, []), list)
                                                    for key in ("requires", "establishes")):
            return finish("NEEDS_CONTEXT", "Malformed owner evidence requirements", index=index)
        conditional = any(isinstance(item, dict) and "when" in item
                          for key in ("requires", "establishes") for item in requirements.get(key, []))
        if conditional:
            if not callable(project_completion):
                return finish("NEEDS_CONTEXT", "Conditional evidence requires an owner sequence validator", index=index)
        else:
            for key in ("requires", "establishes"):
                values, error = contracts._evidence_rows(entry, key, params)
                if error:
                    return finish("NEEDS_CONTEXT", error, index=index)
                if key == "requires" and not set(values) <= facts:
                    return finish("INFEASIBLE", f"{name}: required earlier evidence is absent", index=index)
                if key == "establishes":
                    facts.update(values)
        if "effects" not in entry:
            return finish("NEEDS_CONTEXT", f"{name}: declared effects are missing", index=index)
        previous = deepcopy(state)
        effects = {**entry["effects"], **(entry.get("product_effects") or {})}
        state, error = contracts._apply(effects, state, bound, declarations)
        if error:
            return finish("NEEDS_CONTEXT", error, index=index)
        written.update(contracts._field(key, declarations)[1] for key in effects)
        for field, effect in effects.items():
            if isinstance(effect, dict) and set(effect) & {"pose_absolute_from_params", "pose_relative_from_params"}:
                scope, key = contracts._field(field, declarations)
                old_pose = previous[scope].get(key, {})
                pose = state[scope][key]
                orientation = {name: params.get(name, old_pose.get(name)) for name in ("qx", "qy", "qz", "qw")}
                if all(value is not None for value in orientation.values()):
                    pose.update(orientation)
        row["projected_snapshot"], _ = contracts._runtime_views(state, recovery_snapshot, part_context)

    resource, part = contracts._runtime_views(state, recovery_snapshot, part_context)
    if callable(project_completion):
        completion = project_completion(task=deepcopy(task), trace=deepcopy(result["trace"]),
            recovery_des_model=deepcopy(recovery_des_model), recovery_snapshot=deepcopy(recovery_snapshot),
            projected_snapshot=resource, part_context=part, grounding_context=context)
        result["completion_evidence"] = deepcopy(completion)
        if completion.get("feasibility_status") != "FEASIBLE":
            return finish(completion.get("feasibility_status", "NEEDS_CONTEXT"), completion.get("reason", "Owner completion evidence is missing"))
        resource, part = completion["projected_snapshot"], completion["projected_part_context"]
        for key in ("resource_evidence", "projected_resources", "projected_parts"):
            if key in completion:
                result[key] = deepcopy(completion[key])
    result.update(projected_snapshot=deepcopy(resource), projected_part_context=deepcopy(part))
    goal = task.get("expected_end_state")
    if not isinstance(goal, dict) or not goal:
        return finish("INFEASIBLE", "A nonempty intended postcondition is required")
    for field, expected in goal.items():
        declaration = declarations.get(field)
        if not isinstance(declaration, dict):
            return finish("NEEDS_CONTEXT", f"Postcondition {field!r} is not declared")
        stated = _on_declared_fields(declaration, expected)
        value = _declared_value(field, declaration, resource, part)
        if field == "resource_state" and field not in written and not contracts._equal(value, stated) and not _declares(
                primitive_catalog, field, stated):
            # As Algorithm 1 has it, a state that no primitive of the resource declares is named by the event
            # itself, unless a step's declared effect set another; a state one of them declares comes only from
            # that primitive.
            named = {"resource": {field: deepcopy(stated)}, "part": {}, "unknown": {"resource": [], "part": []}}
            resource, _ = contracts._runtime_views(named, resource, part)
            result["projected_snapshot"] = deepcopy(resource)
            value = stated
        if value is contracts._MISSING:
            return finish("NEEDS_CONTEXT", f"The composition has no supported effect for {field!r}")
        if not contracts._equal(value, stated):
            result["derived_end_state"][field] = deepcopy(value)
            return finish("INFEASIBLE", f"Postcondition {field!r}: achieved {value!r}, required {stated!r}")
        result["derived_end_state"][field] = deepcopy(expected)
    # The event's stated values for other resources must be what the program leaves them, as for its own.
    for jid, values in (task.get("neighbor_end_state") or {}).items():
        neighbor = grounding_context.get("resources", {}).get(jid) or {}
        current = {**neighbor, **(result.get("projected_resources", {}).get(jid) or {})}
        for field, stated in (values or {}).items():
            achieved = contracts._get(current, field)
            if field == "resource_state" and not contracts._equal(achieved, stated) and not _declares(
                    neighbor.get("execution_primitive_catalog") or [], field, stated):
                # As for the acting resource, and as the RA has it for every resource: a state none of the
                # neighbour's own primitives declares is named by the event, and later events start from it.
                result.setdefault("projected_resources", {}).setdefault(jid, {})[field] = deepcopy(stated)
                continue
            if achieved is contracts._MISSING:
                return finish("NEEDS_CONTEXT", f"The composition has no value for {jid}'s {field}")
            if not contracts._equal(achieved, stated):
                return finish("INFEASIBLE", f"{jid}'s {field}: achieved {achieved!r}, required {stated!r}")
    return finish("FEASIBLE", "The authored sequence satisfies the declared resource and product postconditions")
