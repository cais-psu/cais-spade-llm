from __future__ import annotations

"""Bind checked PA measurements without changing the RA-authored primitive graph."""

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any

from .primitive_composition import (
    _CompositionInputs, _evidence_value, _recorded_request_inputs, _result_schema, _validate_steps,
)
from .program_dependencies import assess_program_dependencies, selected_references
from .refinement_records import read_pin, verify_evidence_tree, verify_record


def _pointer_parts(path: str) -> list[str]:
    if not isinstance(path, str) or not path.startswith("/"):
        raise ValueError("A parameter binding requires an exact JSON pointer.")
    return [field.replace("~1", "/").replace("~0", "~") for field in path.split("/")[1:]]


def _child_pointer(pointer: str, field: str) -> str:
    return pointer + "/" + field.replace("~", "~0").replace("/", "~1")


def _expand_object(node: dict[str, Any], inputs: _CompositionInputs) -> None:
    if set(node) != {"value_ref"}:
        return
    ref = node["value_ref"]
    measured = _evidence_value(inputs, ref["record_ref"], ref["field_path"])
    if not isinstance(measured, dict):
        raise ValueError("A partial object binding requires measured object fields.")
    children = {key: {"value_ref": {"record_ref": ref["record_ref"],
                                   "field_path": _child_pointer(ref["field_path"], key)}}
                for key in measured}
    node.clear()
    node.update(children)


def _set_binding(params: dict[str, Any], path: str, value: Any, inputs: _CompositionInputs) -> None:
    node: Any = params
    parts = _pointer_parts(path)
    for field in parts[:-1]:
        if isinstance(node, dict):
            if set(node) == {"result_ref"}:
                raise ValueError("Deterministic binding cannot replace an RA-selected result dependency.")
            _expand_object(node, inputs)
            node = node.setdefault(field, {})
        elif isinstance(node, list) and field.isdecimal() and int(field) < len(node):
            node = node[int(field)]
        else:
            raise ValueError("The requested binding path crosses a supplied scalar or absent array item.")
    if isinstance(node, dict):
        _expand_object(node, inputs)
    if isinstance(node, dict) and set(node) != {"result_ref"}:
        node[parts[-1]] = deepcopy(value)
    elif isinstance(node, list) and parts[-1].isdecimal() and int(parts[-1]) < len(node):
        node[int(parts[-1])] = deepcopy(value)
    else:
        raise ValueError("The requested binding path would replace an existing result dependency.")


def apply_primitive_bindings(
    inputs: _CompositionInputs, candidate_ref: Mapping[str, str],
    pa_answer_refs: list[dict[str, str]],
) -> tuple[list[dict[str, Any]], _CompositionInputs]:
    """Apply only missing or invalid measurement paths requested from this proposal.

    Args:
        inputs: Current base composition authority.
        candidate_ref: Pin of the unchanged RA proposal.
        pa_answer_refs: Pins of checked answer checkpoints for this proposal.

    Returns:
        Bound steps and inputs extended by their verified evidence sources.
    """
    from ..pa.primitive_context import _read_answer_checkpoint, _read_pa_value

    candidate = read_pin(inputs.root, candidate_ref)
    if candidate.get("record_type") != "PrimitiveProgramCandidate" or candidate.get("status") != "proposed":
        raise ValueError("Parameter binding requires a proposed RA program.")
    request = read_pin(inputs.root, {"ref": candidate["request_ref"], "sha256": candidate["request_sha256"]})
    if request.get("context_refs") != inputs.context_refs:
        raise ValueError("Parameter binding has different composition authority.")
    inputs = _recorded_request_inputs(inputs, request)
    original = candidate["primitive_steps"]
    dependencies = assess_program_dependencies(
        original, inputs.catalog, inputs.composition_input["robot_state"],
        read_evidence=lambda ref, pointer: _evidence_value(inputs, ref, pointer),
        result_schema=lambda ref: _result_schema(original, ref, inputs),
    )
    allowed = {(need["step_index"], need["parameter_path"]) for need in dependencies["context_requests"]
               if need["authority"] == "PA"}
    steps, hashes, answers = deepcopy(original), dict(inputs.record_hashes), []
    for answer_ref in pa_answer_refs:
        checkpoint = _read_answer_checkpoint(inputs.root, answer_ref)
        pa_request = verify_record(inputs.root, checkpoint["request_ref"])
        if (pa_request.get("record_type") != "PrimitiveContextRequest"
                or pa_request.get("candidate_ref") != candidate_ref or pa_request.get("primitive_steps") != original
                or pa_request.get("base_context_refs") != inputs.context_refs
                or pa_request.get("assignment_fingerprint") != inputs.assignment.fingerprint):
            raise ValueError("PA answers belong to another proposal or composition authority.")
        for answer in checkpoint["answers"]:
            if "source_ref" not in answer:
                continue
            source = answer["source_ref"]
            record = verify_evidence_tree(inputs.root, source)
            if source["ref"] in hashes and hashes[source["ref"]] != source["sha256"]:
                raise ValueError("PA binding sources contain conflicting hashes.")
            hashes[source["ref"]] = source["sha256"]
            if "value_ref" in answer:
                ref = answer["value_ref"]
                if ref["record_ref"] != source["ref"]:
                    raise ValueError("PA selected value and source reference disagree.")
                selected = _read_pa_value(inputs.root, record, ref["field_path"], {source["ref"]: record})
                if selected.get("value") != answer.get("value") or "value" not in selected:
                    raise ValueError("PA selected value differs from its pinned measurement.")
                answers.append(answer)
    inputs = replace(inputs, record_hashes=hashes, binding_ref=None)
    updates = {}
    for answer in answers:
        need = answer["need"]
        index, parent = need["step_index"], need.get("parameter_path", need["quantity"])
        if index is None or not isinstance(parent, str) or not parent.startswith("/"):
            continue
        for step_index, path in sorted(allowed):
            if step_index != index or not (path == parent or path.startswith(parent + "/")):
                continue
            if any(kind == "result_ref" and (path == selected or path.startswith(selected + "/"))
                   for selected, kind, _ in selected_references(original[index - 1]["params"])):
                continue
            selection = {**answer["value_ref"], "field_path": answer["value_ref"]["field_path"] + path[len(parent):]}
            _evidence_value(inputs, selection["record_ref"], selection["field_path"])
            if (index, path) in updates and updates[index, path] != selection:
                raise ValueError("PA answers select conflicting bindings for the same parameter.")
            updates[index, path] = selection
    for (index, path), selection in sorted(updates.items()):
        _set_binding(steps[index - 1]["params"], path, {"value_ref": selection}, inputs)
    _validate_steps(steps, inputs)
    return steps, inputs


def read_program_binding(
    inputs: _CompositionInputs, reference: Mapping[str, str],
) -> tuple[dict[str, Any], _CompositionInputs]:
    """Verify the saved binding by replaying its exact proposal and checked answers.

    Args:
        inputs: Current base composition authority.
        reference: Pinned PrimitiveProgramBinding.

    Returns:
        The verified record and its evidence-extended composition inputs.
    """
    record = verify_record(inputs.root, reference)
    if record.get("record_type") != "PrimitiveProgramBinding":
        raise ValueError("Expected a PrimitiveProgramBinding.")
    directory = Path(reference["ref"]).parent
    run = verify_record(inputs.root, record["run_request_ref"])
    if (run.get("record_type") != "PrimitiveRefinementRequest"
            or Path(record["run_request_ref"]["ref"]).parent != directory
            or run["base_context_refs"] != inputs.context_refs):
        raise ValueError("Primitive binding belongs to another refinement run.")
    for answer_ref in record["pa_answer_refs"]:
        if Path(answer_ref["ref"]).parent.parent != directory:
            raise ValueError("Primitive binding uses answers from another run.")
        checkpoint = verify_record(inputs.root, answer_ref)
        request = verify_record(inputs.root, checkpoint["request_ref"])
        if request.get("run_request_ref") != record["run_request_ref"]:
            raise ValueError("Primitive binding uses a PA request with different run authority.")
    steps, extended = apply_primitive_bindings(inputs, record["candidate_ref"], record["pa_answer_refs"])
    from .validation_scope import read_validation_scope

    if read_validation_scope(run["profile"]) != extended.validation_scope:
        raise ValueError("Primitive binding and proposal have different validation scopes.")
    if steps != record["primitive_steps"]:
        raise ValueError("Saved primitive binding differs from its proposal and checked PA answers.")
    return record, replace(extended, binding_ref=dict(reference))


def display_primitive_steps(
    inputs: _CompositionInputs, steps: list[dict[str, Any]], report: Mapping[str, Any] | None = None,
    *, calculation_refs: Sequence[Mapping[str, str]] = (),
) -> list[dict[str, Any]]:
    """Render measured values and completed calculations without rewriting records.

    Args:
        inputs: Hash-checked binding inputs.
        steps: Reference-preserving bound program.
        report: Matching validation report, when available.
        calculation_refs: Completed helper calculations from this binding's progress.

    Returns:
        Display-only steps with available numerical values and pending result refs.
    """
    from .composition_context import _resolve_json_pointer

    results = {}
    for reference in calculation_refs:
        if (inputs.binding_ref is None or Path(reference["ref"]).parent.parent
                != Path(inputs.binding_ref["ref"]).parent):
            raise ValueError("Displayed calculation belongs to another primitive binding run.")
        calculation = verify_record(inputs.root, reference)
        index = calculation.get("step_index")
        if (calculation.get("record_type") != "PrimitiveCalculationRecord"
                or calculation.get("binding_ref") != inputs.binding_ref
                or type(index) is not int or not 1 <= index <= len(steps)
                or calculation.get("primitive_symbol") != steps[index - 1]["primitive_symbol"]):
            raise ValueError("Displayed calculation differs from the pinned primitive binding.")
        # Binding authority has its own reader; its context IDs are not artifact paths.
        for source in calculation["source_refs"]:
            verify_evidence_tree(inputs.root, source)
        results[index] = calculation["result"]

    def resolve(value: Any) -> Any:
        if isinstance(value, dict):
            if set(value) == {"value_ref"}:
                ref = value["value_ref"]
                return deepcopy(_evidence_value(inputs, ref["record_ref"], ref["field_path"]))
            if set(value) == {"result_ref"} and value["result_ref"]["step_index"] in results:
                ref = value["result_ref"]
                try:
                    return deepcopy(_resolve_json_pointer(results[ref["step_index"]], ref["field_path"]))
                except ValueError:
                    # A conditional output can remain unavailable after its helper ran.
                    return deepcopy(value)
            return {key: resolve(item) for key, item in value.items()}
        if isinstance(value, list):
            return [resolve(item) for item in value]
        return value

    displayed = [{**deepcopy(step), "params": resolve(step["params"])} for step in steps]
    for checked in (report or {}).get("checked_steps", []):
        if "resolved_params" in checked:
            displayed[checked["step_index"] - 1]["params"] = deepcopy(checked["resolved_params"])
    return displayed
