"""Migrate copies of archived test inputs to the current AP schema.

Archived run evidence remains unchanged. Production code never accepts these
retired descriptors; this adapter exists only to preserve regression scenarios.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any
from urllib.parse import unquote

from cais_spade_llm.agents.central_controller.ppr_ap import (
    ap_record, canonical_ap_key, make_ap_definition, parse_ap_definition,
)
from cais_spade_llm.agents.central_controller.reviewed_primitive_program_safety import (
    _FIRST, _SECOND, _ENTRY, _COMPLETED, _TARGET_COMPLETED, _RECEIVING_ENTRY, _CONTAINS_OTHER,
)

_PHYSICAL = {
    "ap_state/physical_observation/shared_area_first_resource": _FIRST,
    "ap_state/physical_observation/shared_area_second_resource": _SECOND,
    "ap_event/physical_observation/part_region_entry": _ENTRY,
    "ap_state/processCompleted/process_result_completed": _COMPLETED,
    "ap_state/processCompleted/process_target_completed": _TARGET_COMPLETED,
    "ap_event/physical_observation/receiving_region_entry": _RECEIVING_ENTRY,
    "ap_state/physical_observation/receiving_resource_contains_other_part": _CONTAINS_OTHER,
}


def migrate_ap_key(value: str) -> str:
    """Translate an archived descriptor for a regression scenario."""
    if value in _PHYSICAL:
        return _PHYSICAL[value]
    if not value.startswith(("ap_state/", "ap_event/")):
        return value
    parts = value.split("/", 5)
    if len(parts) != 6:
        return value
    kind, process, product, resource, symbol, context = parts
    arguments = {}
    if context != "any":
        for pair in context.split("&"):
            if "=" not in pair:
                arguments["location"] = context
                break
            key, item = pair.split("=", 1)
            arguments[unquote(key)] = unquote(item)
    if symbol == "part_state=assembled" and arguments.get("destination_location"):
        # A part stores no handling state; an assembled part is located on its assembly destination.
        symbol = f"part_location={arguments['destination_location']}"
    return canonical_ap_key(make_ap_definition(kind, product, process, resource, symbol, arguments))


def migrate_resource_programs(programs: dict) -> None:
    """Bring archived saved programs to the current transport and machine job contracts."""
    from cais_spade_llm.resources.machine.machine_jobs import machine_job_functions

    resources = programs.get("resources", {})
    for rid, event in (("Conveyor", "advance_conveyor"), ("Buffer For Machined parts", "advance_part")):
        bundle = resources.get(rid)
        if not isinstance(bundle, dict) or set(bundle.get("primitives", {})) != {"move_relative"}:
            continue
        program = bundle["functions"][event]["program"]
        program["steps"] = [{"id": "observe", "op": "detect_parts"},
                            {"id": "targets", "op": "compute_transport_targets"}, *program["steps"]]
        bundle["primitives"] = {name: {"status": "executable", "recovery_selectable": True}
                                for name in ("detect_parts", "compute_transport_targets", "move_relative")}
    kmr = resources.get("KMR")
    if isinstance(kmr, dict):
        # Coordinate travel is composed from move_base; it is no longer a saved KMR function.
        kmr.get("functions", {}).pop("move_to_location", None)
    printer = resources.get("3D Printing Station")
    if isinstance(printer, dict) and "print_part" in printer.get("functions", {}):
        # Printing starts the print; its output appears when the completion is observed.
        printer["functions"]["print_part"]["program"].update(entry_state="idle", success_state="printing")
    baseline = machine_job_functions()
    for bundle in resources.values():
        if isinstance(bundle, dict) and "start_job" in bundle.get("functions", {}):
            # Job-controlled machining is start_job then the observed complete_job.
            bundle["functions"].pop("machine_part", None)
        for name, function in (bundle.get("functions", {}) if isinstance(bundle, dict) else {}).items():
            if "start_job" in bundle["functions"] and name in baseline:
                for key in ("entry_state", "success_state"):
                    if key in baseline[name]["program"]:
                        function["program"][key] = baseline[name]["program"][key]


def migrate_scene(scene: dict) -> None:
    """Bring an archived scene's saved programs and printer states to the current models."""
    migrate_resource_programs(scene["resource_programs"])
    printer = scene.get("3D Printing Station")
    state = ((printer.get("state_variables") or {}).get("resource_state")
             if isinstance(printer, dict) and isinstance(printer.get("state_variables"), dict) else None)
    if isinstance(state, dict) and state.get("domain") == ["idle", "completed"]:
        state["domain"] = ["idle", "printing", "completed"]


def migrate_ppr_fixture(value: Any) -> Any:
    """Copy an old fixture with typed AP records and current catalog versions."""
    if isinstance(value, list):
        return [migrate_ppr_fixture(item) for item in value]
    if isinstance(value, str):
        return migrate_ap_key(value)
    if not isinstance(value, dict):
        return deepcopy(value)
    result = {migrate_ap_key(key): migrate_ppr_fixture(item) for key, item in value.items()}
    if isinstance(result.get("resource_programs"), dict):
        migrate_scene(result)
    if "specifications" in result and "version" in result:
        result["version"] = 2
    if result.get("mode") == "predefined":
        result["version"] = 2
    if {"label", "full", "meaning"} <= set(result) and result["full"].startswith("{"):
        result["definition"] = parse_ap_definition(result["full"])
    if "ap_groundings" in result:
        for binding in result["ap_groundings"].values():
            context = binding.pop("context", "any")
            if context != "any":
                binding["arguments"] = {
                    unquote(key): unquote(item)
                    for key, item in (pair.split("=", 1) for pair in context.split("&"))
                }
    return result
