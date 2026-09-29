"""Validated function transitions shared by the resource graph and dispatch."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


def validated_function_contract(models: dict[str, dict], event_id: int) -> dict[str, Any]:
    """Return one function's formal transition and saved execution program.

    Args:
        models: Resource descriptors from the same configured scene revision.
        event_id: Existing formal event identity to resolve.

    Returns:
        The owning function, ordered steps, and every participant's conditions.

    Raises:
        ValueError: The formal event and saved function cannot be paired safely.
    """
    participants = {
        rid: event
        for rid, model in models.items()
        for event in model["events"]
        if event["event_id"] == event_id
    }
    if not participants:
        raise ValueError(f"Unknown function event_id {event_id}")
    first = next(iter(participants.values()))
    actor = first["parameter_bindings"]["resource_id"]["equals"]
    owner = participants.get(actor)
    if owner is None:
        raise ValueError(f"Function event_id {event_id} has no owning resource")
    program = owner.get("program")
    status = owner.get("program_status")
    if not isinstance(program, dict) or status not in {"implemented", "planned"}:
        raise ValueError(f"Function event_id {event_id} has no validated program")
    variants = owner.get("program_variants") or {}
    if not isinstance(variants, dict):
        raise ValueError(f"Function event_id {event_id} has invalid program variants")
    candidates = [program, *variants.values()]
    for candidate in candidates:
        if not isinstance(candidate, dict) or not isinstance(candidate.get("steps"), list):
            raise ValueError(f"Function event_id {event_id} has invalid saved steps")
    if status == "planned" and any(candidate["steps"] for candidate in candidates):
        raise ValueError(f"Planned function event_id {event_id} has executable steps")
    guard = owner["guards"].get("resource_state", {}).get("equals")
    update = owner["updates"].get("resource_state", {}).get("set")
    if guard is not None and not any(
        candidate.get("entry_state") in {guard, "any"}
        and (update is None or candidate.get("success_state") == update)
        for candidate in candidates
    ):
        raise ValueError(f"Function event_id {event_id} disagrees with its in state")
    if update is not None and not any(
        candidate.get("success_state") == update
        and (guard is None or candidate.get("entry_state") in {guard, "any"})
        for candidate in candidates
    ):
        raise ValueError(f"Function event_id {event_id} disagrees with its out state")
    for rid, event in participants.items():
        if (event["event_name"] != owner["event_name"]
                or event["parameter_bindings"] != owner["parameter_bindings"]
                or event.get("function_name") != owner.get("function_name")
                or event.get("program_status") != status
                or set(event.get("participants", [])) != set(participants)):
            raise ValueError(f"Function event_id {event_id} disagrees at {rid}")
        if models[rid].get("program_revision") != models[actor].get("program_revision"):
            raise ValueError(f"Function event_id {event_id} has mixed program revisions")
    return {
        "event_id": event_id,
        "event_name": owner["event_name"],
        "resource_id": actor,
        "function_name": owner.get("function_name", owner["event_name"]),
        "program_key": owner.get("program_key", owner["event_name"]),
        "program_status": status,
        "program_revision": models[actor].get("program_revision", ""),
        "parameter_bindings": deepcopy(owner["parameter_bindings"]),
        "participants": list(participants),
        "in_state": {
            rid: {
                "guards": deepcopy(event["guards"]),
                "collection_guards": deepcopy(event.get("collection_guards", {})),
            }
            for rid, event in participants.items()
        },
        "out_state": {
            rid: {
                "updates": deepcopy(event["updates"]),
                "collection_effects": deepcopy(event.get("collection_effects", {})),
            }
            for rid, event in participants.items()
        },
        "product_effects": deepcopy(owner.get("product_effects", {})),
        "steps": deepcopy(program["steps"]),
        "program_variants": {
            name: deepcopy(candidate["steps"]) for name, candidate in variants.items()
        },
    }


def validate_function_contracts(models: dict[str, dict]) -> None:
    """Reject any resource model whose event lacks a matching saved function."""
    for event_id in {
        event["event_id"] for model in models.values() for event in model["events"]
    }:
        validated_function_contract(models, event_id)
