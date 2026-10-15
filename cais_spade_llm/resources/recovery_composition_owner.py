from __future__ import annotations

"""Bind primitive composition validation to resource-owned evidence and evaluators."""

from copy import deepcopy
from typing import Any

from cais_spade_llm.resources.recovery_composition import validate_recovery_primitive_composition


def validate_owner_composition(owner: Any, *, task: dict, primitive_steps: list,
                               prepared: dict, recovery_snapshot: dict,
                               part_context: dict, grounding_context: dict) -> dict:
    """Validate an exact event program without reading or controlling equipment.

    Args:
        owner: Responsible live RA or detached saved resource.
        task: Immutable accepted outline event.
        primitive_steps: Authored program to validate.
        prepared: Bound models and recorded evidence for this attempt.
        recovery_snapshot: State at the accepted composition prefix.
        part_context: Product facts at the same prefix.
        grounding_context: Recorded values available to parameter references.

    Returns:
        Bound RA composition evidence with execution authorization disabled.
    """
    jid = task["resource_jid"]
    if str(owner.jid) != jid:
        raise ValueError("Composition was assigned to a different resource owner")
    entry = prepared.get("recovery_resources", {}).get(jid, {})
    evidence = deepcopy(grounding_context.get("resources", {}).get(jid, {}).get("resource_evidence")
                        or entry.get("resource_evidence") or {})
    model = entry.get("recovery_des_model") or {}
    physical = deepcopy(prepared.get("physical_evidence") or {})
    for resource_jid, current in grounding_context.get("resources", {}).items():
        resource_id = current.get("resource_id") or next(
            (name for name, value in physical.get("jids", {}).items() if value == resource_jid), None)
        declarations = physical.get("environment_models", {}).get(resource_id, {}).get("state_variables", {})
        if resource_id in physical.get("environment_valuation", {}):
            physical["environment_valuation"][resource_id].update(
                {key: deepcopy(current[key]) for key in declarations if key in current})
    for name, current in grounding_context.get("parts", {}).items():
        if name in physical.get("products", {}):
            physical["products"][name].update(deepcopy(current))
    physical_input = {"resource_evidence": evidence, "physical_evidence": physical,
                      "part_context": deepcopy(part_context), "primitive_support_semantics": "successor_primitive_union"}
    prepare = getattr(owner, "prepare_recovery_primitive_parameters", None)
    if callable(prepare):
        preparation = prepare(task=task, recovery_des_model=model,
            recovery_snapshot=recovery_snapshot, physical_input=physical_input)
        evidence = deepcopy(preparation.get("calculated_resource_evidence", evidence))

    def capability(**values) -> dict:
        evaluator = getattr(owner, "check_recovery_primitive_feasibility", None)
        if not callable(evaluator):
            return {"feasibility_status": "NEEDS_CONTEXT", "reason": "Owner capability evaluator is missing"}
        return evaluator(**values, task=deepcopy(task), grounded_action={},
                         resource_evidence=deepcopy(evidence), physical_evidence=deepcopy(physical))

    preview, completion = None, None
    if entry.get("resource_type") == "robot":
        from cais_spade_llm.resources.robot.recovery_composition import preview_composition_output, project_composition_completion

        def preview(**values) -> dict:
            return preview_composition_output(**values, task=task, resource_evidence=evidence,
                                              physical_evidence=physical)

        def completion(**values) -> dict:
            return project_composition_completion(**values, primitive_catalog=entry.get("execution_primitive_catalog", []),
                resource_evidence=evidence, physical_evidence=physical)

    return validate_recovery_primitive_composition(task=task, primitive_steps=primitive_steps,
        recovery_des_model=model, recovery_snapshot=recovery_snapshot, part_context=part_context,
        grounding_context=grounding_context, primitive_catalog=entry.get("execution_primitive_catalog") or [],
        validate_primitive=capability, preview_primitive=preview, project_completion=completion)
