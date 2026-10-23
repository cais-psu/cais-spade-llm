from __future__ import annotations

"""Resource-owned transition models for runtime environmental exploration.

The existing task builders supply the configured topology and task programs.
Catalogue-dependent DES expansions are replaced by parameterized declarations;
only a runtime valuation contains the registered workpiece identities. The v1
projector remains available for delivery and its tests.

Within one configured model, a concrete composition event is identified by its
event_id and complete parameter binding. event_name is the task-function name;
it is not a synchronization symbol by itself.
"""

import json
import math
from copy import deepcopy
from dataclasses import asdict
from functools import lru_cache
from itertools import combinations_with_replacement
from typing import Any

from cais_spade_llm.recovery_framework import SCENE_PATH
from cais_spade_llm.recovery_framework.kmr_tasks import KMR_TASKS
from cais_spade_llm.resources import nominal_des as tasks
from cais_spade_llm.resources.function_contracts import validate_function_contracts
from cais_spade_llm.resources.gazebo_programs import resource_program_revision
from cais_spade_llm.resources.nominal_conveyor import (
    CONVEYOR_LOCATIONS,
    conveyor_advance_parameters,
    conveyor_load_parameters,
    conveyor_parts,
)
from cais_spade_llm.resources.resource_safety_preparation import (
    resource_model_declarations,
    resource_provider,
)
from cais_spade_llm.resources.robot.robot_task_registry import robot_task_registry
from cais_spade_llm.resources.workflow_task_programs import workflow_task_program


@lru_cache(maxsize=1)
def declared_fault_states() -> frozenset[str]:
    """Return resource states entered by an uncontrollable fault event.

    Returns:
        The configured ``execution_monitoring.fault_report_values``.
    """
    scene = json.loads(SCENE_PATH.read_text(encoding="utf-8"))
    return frozenset(scene.get("execution_monitoring", {}).get("fault_report_values", []))


PART_REFERENCE = {"scope": "resource", "type": ["string", "null"], "reference": "part_name"}
PART_PARAMETER = {"required": True, "type": "string", "reference": "part_name"}
# Job commands that change the status of the part's unfinished process.
_JOB_STATUS = {"pause_job": "paused", "resume_job": "running", "stop_job": "stopped"}


class ResourceGuardError(ValueError):
    """Retain the exact recorded condition that blocks a declared transition."""

    def __init__(self, resource_id: str, field: str, operator: str, expected: Any, actual: Any) -> None:
        super().__init__(f"Guard blocked: {resource_id}.{field}")
        self.resource_id = resource_id
        self.field = field
        self.operator = operator
        self.expected = deepcopy(expected)
        self.actual = deepcopy(actual)


def build_environment_models(
    scene: dict, parts: list[str] | None = None, *, schema_version: int = 3
) -> dict:
    """Build all resource graphs, with identities confined to runtime occupancy.

    Args:
        scene: Exact configured resource identifiers and connections.
        parts: Registered order/inventory identities, never an eligibility list.
        schema_version: Process semantics; version 2 preserves historical effects.

    Returns:
        Resource descriptors whose state declarations and events are reusable.
    """
    if schema_version not in {2, 3}:
        raise ValueError("Environmental models require schema_version 2 or 3")
    program_revision = resource_program_revision(scene) if "resource_programs" in scene else ""
    # Validate the configured topology independently of scenario part assignments.
    # Inventory is initialized below; it never restricts capability eligibility.
    topology = deepcopy(scene)
    for machine in topology["machines"]:
        machine["nominal_parts"] = []
    topology["Storage"]["slots"] = []
    topology["3D Printing Station"]["supported_products"] = []
    for zone in topology["Buffer For Machined parts"]["zones"]:
        zone["initial_part"] = None
    models = tasks.build_nominal_resource_des_models(topology)
    for row in scene["robots"]:
        robot = row["resource_id"]
        for connection in row.get("handling_connections", []):
            source = connection["origin_resource_location"]
            destination = connection["destination_location"]
            if source != "Storage" or destination not in {
                m["resource_id"] for m in scene["machines"]
            }:
                raise ValueError(
                    "Additional handling_connections require Storage and a configured machine"
                )
            tasks._robot_pick_events(
                models,
                robot,
                source,
                source,
                {"inventory.{part_name}": {"equals": True}},
                {"inventory.{part_name}": {"set": False}},
            )
            tasks._robot_place_events(
                models,
                robot,
                destination,
                destination,
                tasks._equals(resource_state="idle", part_name=None),
                {"resource_state": {"set": "loaded"}, "part_name": {"set_from_param": "part_name"}},
            )
            for field in (
                "resource_location",
                "task_ctx.origin_resource_location",
                "task_ctx.destination_location",
            ):
                domain = models[robot]["state_variables"][field]["domain"]
                domain.extend(value for value in (source, destination) if value not in domain)
    _declare_machine_jobs(models, scene)
    registered = list(
        dict.fromkeys(
            parts
            if parts is not None
            else [
                *scene["Storage"]["slots"],
                *scene["3D Printing Station"]["initial_products"],
                *(
                    zone["initial_part"]
                    for zone in scene["Buffer For Machined parts"]["zones"]
                    if zone["initial_part"] is not None
                ),
                scene["Exit"]["completed_product"],
            ]
        )
    )
    registry = robot_task_registry()
    for rid, model in models.items():
        _declare_runtime_state(model, registered, scene)
        for key in (
            "nominal_parts",
            "supported_products",
            "slots",
            "initial_products",
            "task_execution",
        ):
            model["assignments"].pop(key, None)
        model["schema_version"] = schema_version
        model["current_configuration"] = deepcopy(
            model["assignments"].get("current_configuration", {})
        )
        model["process_capabilities"] = deepcopy(
            model["assignments"].get("process_capabilities", {})
        )
        model["configuration_revision"] = 0
        model["program_revision"] = program_revision
        model["primitive_catalog"] = deepcopy(
            scene.get("resource_programs", {}).get("resources", {}).get(rid, {}).get("primitives", {})
        )
        if "held_part" not in model["state_variables"]:
            model["marked_state_conditions"] = []
        model["events"] = [
            _parameterized_event(event, rid, registry, schema_version, scene)
            for event in model["events"]
        ]
        model["events"] = [event for event in model["events"] if event is not None]
        # These names bind executors. Shared DES events use event_id and the
        # complete parameter binding, since names repeat across unrelated tasks.
        model["local_event_alphabet"] = list(
            dict.fromkeys(e["event_name"] for e in model["events"])
        )
        model["controllable_event_alphabet"] = model["local_event_alphabet"][:]
        model["observable_event_alphabet"] = model["local_event_alphabet"][:]
    if schema_version == 3:
        _declare_printer_state(models, scene["3D Printing Station"])
    _append_resource_models(models, scene, registered, schema_version, program_revision)
    # Removing product-owned assembly guards also removes passive participants.
    for model in models.values():
        for event in model["events"]:
            event["participants"] = [
                rid
                for rid, peer in models.items()
                if any(e["event_id"] == event["event_id"] for e in peer["events"])
            ]
    for model in models.values():
        model["neighbors"] = sorted(
            {
                rid
                for event in model["events"]
                for rid in event["participants"]
                if rid != model["resource_id"]
            }
        )
    validate_environment_composition(models)
    validate_function_contracts(models)
    return models


def _declare_machine_jobs(models: dict, scene: dict) -> None:
    """Declare job control on versioned machine models as states of the one machine automaton.

    A job runs, pauses, stops or completes the machine's loaded part, so those are the machine's own
    states. The part carries its unfinished job as its ``current_process`` (product state, written by
    ``product_effects``), so a job paused or stopped on one part stays with that part while the machine
    processes another, and a reloaded part resumes only on the machine that holds its job. Completion is
    observed, not commanded, so machining is a commanded ``start_job`` followed by the observed
    ``complete_job``; the combined ``machine_part`` event is removed. The machine assigns a started job its
    identity.
    """
    for machine in scene["machines"]:
        job_control = machine.get("simulation_process", {}).get("job_control", {})
        if job_control.get("version") != 1:
            continue
        rid, selected = machine["resource_id"], machine["selected_program"]
        model = models[rid]
        model["events"] = [event for event in model["events"] if event["event_name"] != "machine_part"]
        model["state_variables"]["resource_state"]["domain"] = [
            "idle", "loaded", "running", "paused", "stopped", "completed"]
        tasks._variable(model, "selected_program", list(machine["program_options"]), selected)
        # An interrupted job is observed, never commanded: it is the machine's fault state.
        model["fault_states"] = ["interrupted"]
        # A part whose job paused or stopped may be unloaded like a completed one; it keeps its job state.
        access_events = [event['event_id'] for event in model['events']
            if event['event_name'] in {'pick_approach', 'pick_grasp', 'pick_part'}
            and event['guards'].get('resource_state') == {'equals': 'completed'}]
        for identifier in access_events:
            for state in ('paused', 'stopped'):
                event_id = max(event['event_id'] for peer in models.values() for event in peer['events']) + 1
                for owner, peer in models.items():
                    original = next((event for event in peer['events'] if event['event_id'] == identifier), None)
                    if original is None:
                        continue
                    event = deepcopy(original)
                    event['event_id'] = event_id
                    if owner == rid:
                        event['guards']['resource_state'] = {'equals': state}
                    source = event['capability_transition']['source']
                    if f"{rid}.resource_state" in source:
                        source[f"{rid}.resource_state"] = state
                    peer['events'].append(event)
        bindings = {"resource_id": rid}
        tasks._event(models, "select_program", bindings, {rid: {"resource_state": {"not_equals": "running"}}},
                     {rid: {"selected_program": {"set_from_param": "program"}}},
                     parameters={"program": {"required": True, "type": "string",
                                             "domain": list(machine["program_options"])}},
                     capability=({}, {}))
        # (event, machine state before, machine state after, the part's current process it requires)
        # A part may start here unless this machine holds its unfinished job: a paused job resumes, a stopped
        # one cannot restart, and a job paused on another machine stays in that machine's record.
        transitions = [("start_job", "loaded", "running", {"not_matches": {"machine": rid}}),
                       ("pause_job", "running", "paused", None), ("resume_job", "paused", "running", None),
                       ("stop_job", "running", "stopped", None), ("stop_job", "paused", "stopped", None),
                       ("complete_job", "running", "completed", None)]
        if job_control.get("resume_after_reload") is True:
            # A reloaded part resumes only the job this machine paused; the job record is the machine's own.
            transitions.append(("resume_job", "loaded", "running", {"matches": {"status": "paused", "machine": rid}}))
        for name, before, after, process_guard in transitions:
            parameters = {"part_name": deepcopy(PART_PARAMETER)}
            if name not in {"start_job", "complete_job"}:
                parameters["job_id"] = {"required": True, "type": "string"}
            if name in {"start_job", "complete_job"}:
                parameters.update(process={"equals": "trim"}, result={"required": True, "type": "string"})
            guards = {"resource_state": {"equals": before}, "part_name": {"equals_from_param": "part_name"}}
            # The part stays at the machine while the machine's state changes around it.
            tasks._event(models, name, bindings, {rid: guards}, {rid: {"resource_state": {"set": after}}},
                         parameters=parameters,
                         capability=({"part_location": rid, f"{rid}.resource_state": before},
                                     {"part_location": rid, f"{rid}.resource_state": after}))
            if process_guard is not None:
                model["events"][-1]["product_guards"] = {"current_process": process_guard}
            if name == "complete_job":
                model["events"][-1]["controllable"] = False


def _append_resource_models(models: dict, scene: dict, registered: list, schema_version: int, program_revision: str) -> None:
    """Append trusted declarations without changing established owner identities."""
    declarations = resource_model_declarations(scene)
    for rid, entry in declarations.items():
        model = deepcopy(entry["model"])
        provider = resource_provider(entry["owner_provider"])
        bundle = scene.get("resource_programs", {}).get("resources", {}).get(rid)
        if not isinstance(bundle, dict):
            raise ValueError("Additional resource has no declared program catalog: " + rid)
        model.update(schema_version=schema_version, program_revision=program_revision,
                     configuration_revision=model.get("configuration_revision", 0),
                     current_configuration=model.get("current_configuration", {}),
                     process_capabilities=model.get("process_capabilities", {}),
                     primitive_catalog=deepcopy(bundle["primitives"]),
                     owner_provider=entry["owner_provider"], owner_provider_version=provider.version)
        expected_fields = {bound for field in model["state_variables"]
                           for bound in ([field.replace("{part_name}", part) for part in registered]
                                         if "{part_name}" in field else [field])}
        if set(model["current_valuation"]) != expected_fields:
            raise ValueError("Additional resource valuation must cover its exact registered part references")
        for field, value in model["current_valuation"].items():
            if not check_value(declaration_for(model, field), value, dict.fromkeys(registered)):
                raise ValueError("Additional resource initial state violates its declaration: " + rid + "." + field)
        for event in model["events"]:
            actor = event["parameter_bindings"]["resource_id"]["equals"]
            if actor not in {*models, *declarations}:
                raise ValueError("Additional resource action has an unknown owner")
            if (not isinstance(event["event_name"], str) or not event["event_name"]
                    or type(event.get("controllable")) is not bool or type(event.get("observable")) is not bool):
                raise ValueError("Additional resource event has invalid identity or flags")
            key = event.get("program_key", event["event_name"])
            saved = scene["resource_programs"]["resources"][actor]["functions"][key]
            event.update(program_key=key, function_name=saved["function_name"],
                         program_status=saved["status"], program=deepcopy(saved["program"]))
            if saved.get("variants"):
                event["program_variants"] = deepcopy(saved["variants"])
        model["local_event_alphabet"] = list(dict.fromkeys(e["event_name"] for e in model["events"]))
        model["controllable_event_alphabet"] = [e["event_name"] for e in model["events"] if e.get("controllable") is True]
        model["observable_event_alphabet"] = [e["event_name"] for e in model["events"] if e.get("observable") is True]
        models[rid] = model
    # Validate additional participants before legacy bookkeeping derives lists.
    for rid in declarations:
        for event in models[rid]["events"]:
            actual = [key for key, model in models.items() if any(peer["event_id"] == event["event_id"] for peer in model["events"])]
            if set(event["participants"]) != set(actual) or len(event["participants"]) != len(actual):
                raise ValueError("Additional resource event participants disagree")


def _declare_printer_state(models: dict[str, dict], configuration: dict) -> None:
    declaration = configuration.get("state_variables", {}).get("resource_state")
    if declaration is None:
        return
    if declaration != {"scope": "resource", "domain": ["idle", "printing", "completed"]}:
        raise ValueError("Invalid 3D Printing Station resource_state declaration")
    printer = models["3D Printing Station"]
    printer["state_variables"]["resource_state"] = deepcopy(declaration)
    printer["current_valuation"]["resource_state"] = configuration["initial_state"]
    _check_printer_state(printer["current_valuation"])
    printer["marked_state_conditions"] = [tasks._equals(resource_state="idle")]
    next_id = max(event["event_id"] for model in models.values() for event in model["events"]) + 1
    for event in list(printer["events"]):
        if event["event_name"] == "print_part":
            # Printing starts on a clear bed; the output and its process record appear only when the
            # print is observed to complete, never by command.
            completion = {key: deepcopy(value) for key, value in event.items() if key not in {
                "program", "program_key", "program_status", "program_variants", "function_name"}}
            event["guards"]["resource_state"] = {"equals": "idle"}
            event["collection_guards"] = {
                "output.{part_name}": {"all_other_equals": False}
            }
            event["updates"] = {"resource_state": {"set": "printing"}}
            event["product_effects"] = {"processCompleted": []}
            # Starting a print changes no part yet; the declared output transition is the completion's.
            source = deepcopy(event["capability_transition"]["source"])
            event["capability_transition"] = {"source": source, "target": deepcopy(source)}
            completion.update(
                event_id=next_id, event_name="complete_job", controllable=False, participants=[printer["resource_id"]],
                guards={"resource_state": {"equals": "printing"}, "output.{part_name}": {"equals": False}},
                updates={"resource_state": {"set": "completed"}, "output.{part_name}": {"set": True}})
            printer["events"].append(completion)
            next_id += 1
        elif "output.{part_name}" in event["guards"]:
            event["guards"]["resource_state"] = {"equals": "completed"}
            if event["event_name"] != "pick_grasp":
                continue
            # The shared pickup has two exclusive outcomes, selected by the
            # remaining outputs rather than by a caller-supplied state.
            event["updates"]["resource_state"] = {"set": "completed"}
            event["collection_guards"] = {
                "output.{part_name}": {"any_other_equals": True}
            }
            for model in models.values():
                for peer in list(model["events"]):
                    if peer["event_id"] != event["event_id"]:
                        continue
                    last_pickup = deepcopy(peer)
                    last_pickup["event_id"] = next_id
                    if model is printer:
                        last_pickup["updates"]["resource_state"] = {"set": "idle"}
                        last_pickup["collection_guards"] = {
                            "output.{part_name}": {"all_other_equals": False}
                        }
                    model["events"].append(last_pickup)
            next_id += 1


def _check_printer_state(values: dict) -> None:
    if "resource_state" not in values:
        return
    # A present output means a completed print; an empty bed is idle or still printing.
    expected = {"completed"} if any(
        value is True for field, value in values.items() if field.startswith("output.")
    ) else {"idle", "printing"}
    if values["resource_state"] not in expected:
        raise ValueError("3D Printing Station resource_state disagrees with output occupancy")


def validate_environment_composition(models: dict[str, dict]) -> None:
    """Check that each shared event has one local definition per participant.

    Args:
        models: Resource descriptors in one configured model snapshot.

    Raises:
        ValueError: An event identity, participant, binding, or local field is invalid.
    """
    variants: dict[int, list[tuple[str, dict]]] = {}
    for resource_id, model in models.items():
        for event in model["events"]:
            event_id = event["event_id"]
            if type(event_id) is not int:
                raise ValueError("Composition event_id must be an integer")
            variants.setdefault(event_id, []).append((resource_id, event))
            _validate_composition_fields(model, event, resource_id)

    shared_fields = (
        "event_name", "parameter_bindings", "participants", "product_effects",
        "product_guards", "capability_transition", "controllable", "observable",
    )
    for event_id, entries in variants.items():
        event = entries[0][1]
        actual = [resource_id for resource_id, _ in entries]
        declared = event["participants"]
        actor_binding = event["parameter_bindings"].get("resource_id", {})
        actor = actor_binding.get("equals")
        if (
            len(actual) != len(set(actual))
            or len(declared) != len(set(declared))
            or set(actual) != set(declared)
            or set(actor_binding) != {"equals"}
            or actor not in declared
        ):
            raise ValueError(f"Composition participants disagree for event_id {event_id}")
        for _, peer in entries[1:]:
            if any(peer.get(field) != event.get(field) for field in shared_fields):
                raise ValueError(f"Composition variants disagree for event_id {event_id}")


def _validate_composition_fields(model: dict, event: dict, resource_id: str) -> None:
    output_guard = event.get("collection_guards", {}).get("output.{part_name}")
    if output_guard is not None and (
        resource_id != "3D Printing Station"
        or "output.{part_name}" not in model["state_variables"]
        or "part_name" not in event["parameter_bindings"]
        or output_guard not in ({"any_other_equals": True}, {"all_other_equals": False})
    ):
        raise ValueError("Invalid printer output collection guard")
    for section in ("guards", "updates"):
        for field, rule in event[section].items():
            declaration = field
            if field.endswith(".{delivered_part}"):
                declaration = field.removesuffix(".{delivered_part}") + ".{part_name}"
            if declaration not in model["state_variables"]:
                raise ValueError(f"Undeclared composition field: {resource_id}.{field}")
            for parameter in ("part_name", "delivered_part"):
                if field.endswith(".{" + parameter + "}") and parameter not in event["parameter_bindings"]:
                    raise ValueError(f"Unbound composition field: {resource_id}.{field}")
            for operator, parameter in rule.items():
                if operator.endswith("_from_param") and parameter not in event["parameter_bindings"]:
                    raise ValueError(f"Unbound composition parameter: {resource_id}.{field}")


def _declare_runtime_state(model: dict, registered: list[str], scene: dict) -> None:
    for field in model["state_variables"]:
        if field in {"held_part", "approached_part", "part_name", "staging_part", "task_ctx.part_name"} or (
            field.startswith("zone_") and field.endswith("_part")
        ):
            model["state_variables"][field] = deepcopy(PART_REFERENCE)
    if model["resource_id"] == "Buffer For Machined parts":
        for zone in scene["Buffer For Machined parts"]["zones"]:
            model["current_valuation"][f"zone_{zone['zone']}_part"] = zone["initial_part"]
    indexed = {
        "Storage": {"inventory": [False, True]},
        "Conveyor": {"part_location": [None, *CONVEYOR_LOCATIONS], "part_order": None},
        "3D Printing Station": {"output": [False, True]},
    }.get(model["resource_id"], {})
    for prefix, domain in indexed.items():
        model["state_variables"][prefix + ".{part_name}"] = (
            {"scope": "resource", "type": ["integer", "null"], "minimum": 0}
            if domain is None
            else {"scope": "resource", "domain": domain}
        )
        for part in registered:
            value = None
            if prefix == "inventory":
                value = part in scene["Storage"]["slots"]
            elif prefix == "output":
                value = part in scene["3D Printing Station"]["initial_products"]
            model["current_valuation"][f"{prefix}.{part}"] = value


def _parameterized_event(
    event: dict, rid: str, registry: dict, schema_version: int, scene: dict
) -> dict | None:
    event = deepcopy(event)
    name = event["event_name"]
    for parameter, binding in event["parameter_bindings"].items():
        if "from_assignment" in binding or parameter == "part_name":
            event["parameter_bindings"][parameter] = deepcopy(PART_PARAMETER)
    for section in ("guards", "updates"):
        event[section] = {
            field: rule
            for field, rule in event[section].items()
            if not field.startswith("assembled.")
            and not (
                rid == "Conveyor"
                and field.startswith(("part_location.", "part_order."))
                and "{" not in field
            )
        }
    # Exit handling uses the bound product identity, while its configured location
    # remains an exact endpoint. Completion is checked against PA requirements.
    if event["parameter_bindings"].get("product_completed"):
        for field in ("task_ctx.part_name", "held_part"):
            if field in event["updates"]:
                event["updates"][field] = {"set_from_param": "part_name"}
            if field in event["guards"]:
                event["guards"][field] = (
                    {"equals_from_param": "part_name"} if field != "held_part" else {"equals": None}
                )
    if rid == "Exit" and "part_name" in event["updates"]:
        event["updates"]["part_name"] = {"set_from_param": "part_name"}
    _declare_product_effects(event, schema_version)
    if rid == "Conveyor" and name in {"place_approach", "place_release"}:
        event["collection_guards"] = {
            "loading_position": "The configured loading area must be empty before approach or release",
            "validation_source": "cais_spade_llm.resources.nominal_conveyor.conveyor_load_parameters",
        }
    if rid == "Conveyor" and name in {"place_release", "advance_conveyor"}:
        event["collection_effects"] = {
            "part_location.{part_name}": "acknowledged shared belt occupancy",
            "part_order.{part_name}": "contiguous downstream order for every resident part",
            "validation_source": "cais_spade_llm.resources.nominal_conveyor",
        }
    event["completion_evidence"] = {
        key: deepcopy(value)
        for key, value in event["parameter_bindings"].items()
        if value == {"equals": True}
    }
    actor = event["parameter_bindings"]["resource_id"]["equals"]
    # place_release is the capability event; the robot executes place_insert.
    # KMR owns a separate place_release function and keeps its own program.
    function_name = "place_insert" if actor != "KMR" and name == "place_release" else name
    program_key = function_name if function_name in registry and actor != "KMR" else name
    saved = scene.get("resource_programs", {}).get("resources", {}).get(actor, {}).get("functions", {}).get(program_key)
    if saved is not None:
        event["program_key"] = program_key
        event["function_name"] = saved["function_name"]
        event["program"] = deepcopy(saved["program"])
        event["program_status"] = saved["status"]
        if saved.get("variants"):
            event["program_variants"] = deepcopy(saved["variants"])
    elif actor == "KMR" and name in KMR_TASKS:
        event["function_name"] = name
        event["program"] = json.loads(json.dumps(asdict(KMR_TASKS[name].program)))
        event["program_status"] = "implemented"
    elif function_name in registry:
        event["function_name"] = function_name
        event["program"] = json.loads(json.dumps(asdict(registry[function_name].program)))
        event["program_status"] = "implemented"
    else:
        program = workflow_task_program(name)
        if program:
            event["function_name"] = program.pop("function_name", name)
            event["program_status"] = program.pop("status")
            event["program"] = program
    if (
        not event["guards"]
        and not event["updates"]
        and event["parameter_bindings"]["resource_id"]["equals"] != rid
    ):
        return None
    return event


def _declare_product_effects(event: dict, schema_version: int) -> None:
    event["product_effects"] = {"processCompleted": []}
    status = _JOB_STATUS.get(event["event_name"])
    if status is not None and "job_id" in event["parameter_bindings"]:
        event["product_effects"]["current_process"] = {"set_status": status}
    elif event["event_name"] == "start_job":
        event["product_effects"]["current_process"] = {"set": {
            "process": {"set_from_param": "process"}, "result": {"set_from_param": "result"},
            "status": "running", "machine": {"set_from_param": "resource_id"}}}
    elif event["event_name"] == "complete_job" and "job_id" not in event["parameter_bindings"]:
        event["product_effects"]["current_process"] = {"set": None}
    if event["event_name"] in {"machine_part", "complete_job"}:
        event["parameter_bindings"]["process"] = {"equals": "trim"}
        event["parameter_bindings"]["result"] = {"required": True, "type": "string"}
        event["product_effects"]["processCompleted"] = [
            {"process": "trim", "result": {"set_from_param": "result"}}
        ]
        event["capability_transition"]["target"]["processCompleted"] = deepcopy(
            event["product_effects"]["processCompleted"]
        )
    elif event["event_name"] == "print_part":
        event["product_effects"]["processCompleted"] = [{"process": "print_part"}]
    elif event["event_name"] == "place_insert":
        event["parameter_bindings"]["target"] = {"required": True, "type": "string"}
        event["product_effects"] = {
            "state": "assembled",
            "target": {"set_from_param": "target"},
            "processCompleted": [],
        }
        event["product_guards"] = {
            "ordered_requirements": "preceding results must be acknowledged",
            "target": "must equal the product geometry target",
        }
        if schema_version == 3:
            event["product_effects"]["processCompleted"] = [
                {"process": "assembly", "target": {"set_from_param": "target"}}
            ]
            event["capability_transition"]["target"]["processCompleted"] = deepcopy(
                event["product_effects"]["processCompleted"]
            )
    if event["parameter_bindings"].get("product_completed"):
        event["product_guards"] = {
            "requirements": "all selected component results must be acknowledged"
        }
    if schema_version == 3 and event["product_effects"]["processCompleted"]:
        event.setdefault("product_guards", {})["ordered_requirements"] = (
            "process effects must satisfy the current processPlan step"
        )


def declaration_for(model: dict, field: str) -> dict:
    """Resolve an exact control field or a parameterized occupancy declaration."""
    if field in model["state_variables"]:
        return model["state_variables"][field]
    template = field.split(".", 1)[0] + ".{part_name}"
    if "." not in field or template not in model["state_variables"]:
        raise ValueError(f"Undeclared state field: {model['resource_id']}.{field}")
    return model["state_variables"][template]


def check_value(declaration: dict, value: Any, parts: dict) -> bool:
    """Validate control domains and references without using eligibility lists."""
    if "domain" in declaration:
        return any(type(value) is type(item) and value == item for item in declaration["domain"])
    if declaration.get("reference") == "part_name":
        return value is None or isinstance(value, str) and value in parts
    if "string" in declaration.get("type", []):
        return isinstance(value, str) or value is None and "null" in declaration["type"]
    return value is None or type(value) is int and value >= declaration.get("minimum", 0)


def event_bindings(event: dict, part: str, desired: dict, requirements: list[dict]) -> dict | None:
    """Instantiate a task for one requirement, retaining exact task identifiers."""
    effect_parameters = {}
    effects = event["product_effects"]["processCompleted"]
    if effects:
        for effect in effects:
            if set(effect) != set(desired):
                continue
            if any(
                desired[key] != value for key, value in effect.items()
                if not isinstance(value, dict)
            ):
                continue
            effect_parameters = {
                value["set_from_param"]: desired[key]
                for key, value in effect.items() if isinstance(value, dict)
            }
            break
        else:
            return None
    params = {}
    for name, binding in event["parameter_bindings"].items():
        if "equals" in binding:
            params[name] = binding["equals"]
        elif name == "part_name":
            params[name] = part
        elif name in effect_parameters:
            params[name] = effect_parameters[name]
        elif name == "target" and desired.get("state") == "assembled":
            params[name] = desired["target"]
        elif name in {"next_locations", "delivered_part"}:
            continue
        elif not isinstance(desired.get(name, {}), dict):
            # A command that causes a later observed effect takes the requirement's value directly.
            params[name] = desired[name]
        else:
            return None
    return params


def candidates(model: dict, valuation: dict, part: str, desired: dict, requirements: list[dict]):
    """Yield this resource's task alternatives, including coupled belt outcomes."""
    if "processesToComplete" in desired:
        seen = []
        for requirement in desired["processesToComplete"]:
            for task in candidates(model, valuation, part, requirement, requirements):
                if task not in seen:
                    seen.append(task)
                    yield task
        return
    for event in model["events"]:
        if event["parameter_bindings"]["resource_id"]["equals"] != model["resource_id"]:
            continue
        params = event_bindings(event, part, desired, requirements)
        if params is None:
            continue
        alternatives = [params]
        if event["event_name"] == "advance_conveyor":
            resident = conveyor_parts(model, valuation[model["resource_id"]])
            if not resident:
                continue
            delivered = None if "delivered_part" in params else resident[0]
            remaining = [value for value in resident if value != delivered]
            alternatives = (
                {
                    **params,
                    "delivered_part": delivered,
                    "next_locations": dict(
                        zip(
                            remaining,
                            (CONVEYOR_LOCATIONS[i] for i in reversed(indices)),
                            strict=True,
                        )
                    ),
                }
                for indices in combinations_with_replacement(
                    range(len(CONVEYOR_LOCATIONS)), len(remaining)
                )
            )
        for parameters in alternatives:
            yield {
                "resource_id": model["resource_id"],
                "event_id": event["event_id"],
                "event_name": event["event_name"],
                "parameters": parameters,
            }


def matches_requirement(state: dict, desired: dict) -> bool:
    """Match explicit completed properties; task names never imply a trim result."""
    if "processesToComplete" in desired:
        return all(matches_requirement(state, item) for item in desired["processesToComplete"])
    if "process" in desired:
        return desired in state["processCompleted"]
    if desired.get("state") == "assembled":
        # Orders state assembly as {"state": "assembled", "target": ...}; only place_insert sets a target.
        return state.get("target") == desired.get("target") and all(
            state.get(key) == value for key, value in desired.items() if key not in {"state", "target"})
    return all(state.get(key) == value for key, value in desired.items())


def _machining_feasibility(model: dict, params: dict) -> tuple[str, list[str]]:
    capability = model["process_capabilities"].get(params["process"], {})
    if params["result"] not in capability.get("supported_results", []):
        return "INFEASIBLE", ["Requested process result is outside declared capabilities"]
    config = model["current_configuration"]
    program = config.get("program")
    if not all(config.get(key) for key in ("tool", "workholding", "program")):
        return "NEEDS_CONTEXT", ["Current tooling, workholding, and program evidence is required"]
    if not isinstance(program, dict) or program.get("validated") is not True:
        return "NEEDS_CONTEXT", ["Current program has no validated process effects"]
    if {"process": params["process"], "result": params["result"]} not in program.get("effects", []):
        return "INFEASIBLE", ["Current program cannot establish the requested result"]
    if any(
        config.get(key) != value for key, value in program.get("required_configuration", {}).items()
    ):
        return "INFEASIBLE", ["Current tooling or operating parameters do not satisfy the program"]
    return "FEASIBLE", []


def _geometry_feasibility(model: dict, geometry: dict) -> tuple[str, list[str]]:
    reasons = []
    dimensions = geometry.get("dimensions_m")
    bounds = model["assignments"].get("supported_part_cross_section_m")
    if (
        not isinstance(dimensions, (list, tuple))
        or len(dimensions) != 3
        or any(type(v) not in {int, float} or not math.isfinite(v) or v <= 0 for v in dimensions)
    ):
        return "NEEDS_CONTEXT", ["Part dimensions are unavailable"]
    if bounds and not bounds[0] - 1e-9 <= max(dimensions[:2]) <= bounds[1] + 1e-9:
        return "INFEASIBLE", ["Part cross section exceeds the resource constraints"]
    elif (
        model["assignments"].get("guide_clear_width_m")
        and max(dimensions[:2]) > model["assignments"]["guide_clear_width_m"]
    ):
        return "INFEASIBLE", ["Part does not fit the guide clearance"]
    constraints = model["assignments"].get("part_constraints", {})
    if (
        dimensions
        and "max_dimensions_m" in constraints
        and any(
            actual > limit
            for actual, limit in zip(dimensions, constraints["max_dimensions_m"], strict=True)
        )
    ):
        return "INFEASIBLE", ["Part dimensions exceed resource capacity"]
    if "max_payload_kg" in constraints:
        mass = geometry.get("mass_kg")
        if type(mass) not in {int, float} or not math.isfinite(mass) or mass <= 0:
            reasons.append("Part mass is unavailable")
        elif mass > constraints["max_payload_kg"]:
            return "INFEASIBLE", ["Part exceeds resource payload"]
    if (
        "gripper_opening_m" in constraints
        and dimensions
        and min(dimensions[:2]) > constraints["gripper_opening_m"]
    ):
        return "INFEASIBLE", ["Part exceeds gripper opening"]
    return ("NEEDS_CONTEXT", reasons) if reasons else ("FEASIBLE", [])


def feasibility(model: dict, task: dict, geometry: dict) -> tuple[str, list[str]]:
    """Evaluate process and geometry using resource-owned configuration facts."""
    name, params = task["event_name"], task["parameters"]
    if name == "move_home":
        return "FEASIBLE", []
    status, reasons = _geometry_feasibility(model, geometry)
    if status == "INFEASIBLE":
        return status, reasons
    if name in {"machine_part", "start_job"}:
        process_status, process_reasons = _machining_feasibility(model, params)
        if process_status == "INFEASIBLE":
            return process_status, process_reasons
        reasons.extend(process_reasons)
    if name == "place_insert" and params["target"] != geometry.get("target"):
        return "INFEASIBLE", ["Assembly target does not match product geometry"]
    if name == "print_part" and not model["current_configuration"].get("program"):
        reasons.append("Printing program and material evidence is required")
    return ("NEEDS_CONTEXT", reasons) if reasons else ("FEASIBLE", [])


def _bound(field: str, params: dict) -> str:
    for name in ("part_name", "delivered_part"):
        suffix = ".{" + name + "}"
        if field.endswith(suffix):
            value = params.get(name)
            if not isinstance(value, str):
                raise ValueError("Missing workpiece binding")
            return field[: -len(suffix)] + "." + value
    return field


def owners(models: dict, valuation: dict, products: dict, product_name: str) -> dict:
    """Check coherent control state and exclusive custody for registered parts."""
    if set(valuation) != set(models):
        raise ValueError("Valuation must contain every configured resource")
    result = {}
    for rid, values in valuation.items():
        model = models[rid]
        expected = set(model["current_valuation"])
        if set(values) != expected:
            raise ValueError(f"Incomplete resource valuation: {rid}")
        for field, value in values.items():
            declaration = declaration_for(model, field)
            # A fault is an uncontrollable event into a declared fault state;
            # models recorded before that declaration omit it from their domain.
            if not check_value(declaration, value, products) and not (
                field == "resource_state"
                and (value in declared_fault_states() or value in model.get("fault_states", ()))
            ):
                raise ValueError(f"Invalid resource value: {rid}.{field}")
            part = _custody_part(rid, field, value, product_name)
            if part is not None:
                if part not in products or part in result:
                    raise ValueError("Unknown or duplicate part custody")
                result[part] = rid
        if "held_part" in values and (
            (values["resource_state"] in {"carrying", "picked", "positioned"})
            != (values["held_part"] is not None)
        ):
            raise ValueError("Resource state disagrees with held_part")
        if "approached_part" in values and (
            (values["resource_state"] == "at_pick") != (values["approached_part"] is not None)
        ):
            raise ValueError("KMR approach state disagrees with approached_part")
        if "staging_part" in values and (
            (values["resource_state"] == "idle") != (values["part_name"] is None)
        ):
            raise ValueError("Machine state disagrees with workholding")
        if rid == "3D Printing Station":
            _check_printer_state(values)
    for part, state in products.items():
        # A part assembled on the product is located on it; the product holds it.
        if part != product_name and state.get("location") == product_name:
            if part in result:
                raise ValueError("Assembled part also has resource custody")
            result[part] = product_name
    conveyor_parts(models["Conveyor"], valuation["Conveyor"])
    if (valuation["Exit"]["resource_state"] == "occupied") != (
        valuation["Exit"]["part_name"] is not None
    ):
        raise ValueError("Exit state disagrees with occupancy")
    return result


def _custody_part(rid: str, field: str, value: Any, product_name: str) -> str | None:
    if field in {"held_part", "part_name", "staging_part"} or field.startswith("zone_"):
        return value
    if (field.startswith(("inventory.", "output.")) and value is True) or (
        rid == "Conveyor" and field.startswith("part_location.") and value is not None
    ):
        return field.split(".", 1)[1]
    if field == "product_location" and value == product_name:
        return product_name
    return None


def _task_event(models: dict, products: dict, task: dict) -> dict:
    rid, params = task["resource_id"], task["parameters"]
    event = next((e for e in models[rid]["events"] if e["event_id"] == task["event_id"]), None)
    if event is None or event["event_name"] != task["event_name"]:
        raise ValueError("Unknown resource event")
    if event["parameter_bindings"]["resource_id"]["equals"] != rid:
        raise ValueError("Task is owned by another ResourceAgent")
    if set(params) != set(event["parameter_bindings"]):
        raise ValueError("Incomplete task parameter bindings")
    for name, binding in event["parameter_bindings"].items():
        value = params[name]
        if "equals" in binding and (
            type(value) is not type(binding["equals"]) or value != binding["equals"]
        ):
            raise ValueError("Task parameter differs from declared transition")
        if binding.get("reference") == "part_name" and (
            not isinstance(value, str) or value not in products
        ):
            raise ValueError("Unknown workpiece reference")
        if binding.get("type") == "string" and (not isinstance(value, str) or not value):
            raise ValueError("Task parameter must be a nonempty string")
    return event


def _check_product_requirements(
    task: dict, products: dict, product_name: str, requirements: dict, event: dict
) -> None:
    params = task["parameters"]
    if params.get("product_completed") and (
        params.get("part_name") != product_name
        or not all(
            all(matches_requirement(products[part], desired) for desired in desired_properties)
            for part, desired_properties in requirements.items()
        )
    ):
        raise ValueError("Product assembly requirements are outstanding")
    required = requirements.get(params.get("part_name"), [])
    if any(sequence and "processesToComplete" in sequence[0] for sequence in requirements.values()):
        completed = _process_effects(event, params)
        if completed:
            state = products[params["part_name"]]
            step = next((step for step in required if not matches_requirement(state, step)), None)
            if step is None or any(
                effect not in step["processesToComplete"] or matches_requirement(state, effect)
                for effect in completed
            ):
                raise ValueError("Process effect does not satisfy the current processPlan step")
    elif task["event_name"] == "place_insert":
        goal = {"state": "assembled", "target": params["target"]}
        if goal not in required or not all(
            matches_requirement(products[params["part_name"]], item)
            for item in required[: required.index(goal)]
        ):
            raise ValueError("Preceding product requirements are outstanding")


def _check_product_guards(event: dict, state: dict | None, params: dict) -> None:
    """Check an event's guards on the part's own state, such as which machine holds its paused job."""
    guard = event.get("product_guards", {}).get("current_process")
    if not isinstance(guard, dict):
        return
    current = (state or {}).get("current_process")
    if "equals" in guard:
        allowed = current == guard["equals"]
    elif "matches" in guard:
        allowed = isinstance(current, dict) and all(current.get(key) == value for key, value in guard["matches"].items())
    else:
        allowed = not isinstance(current, dict) or any(current.get(key) != value
                                                       for key, value in guard["not_matches"].items())
    if not allowed:
        raise ValueError(f"Guard blocked: {params.get('part_name')}.current_process")


def current_process_after(event: dict, current: dict | None, params: dict) -> dict | None:
    """Return the part's unfinished process after an event's declared product effect.

    Args:
        event: The declared event.
        current: The part's ``current_process`` before the event, or None.
        params: The event's bound parameters.

    Returns:
        The process record ``{process, result, status, machine}``, or None when no process is unfinished.
    """
    effect = event.get("product_effects", {}).get("current_process")
    if not isinstance(effect, dict):
        return deepcopy(current)
    if "set_status" in effect:
        if not isinstance(current, dict):
            raise ValueError(f"No unfinished process of {params.get('part_name')} to {event['event_name']}")
        return {**current, "status": effect["set_status"]}
    record = effect["set"]
    if record is None:
        return None
    return {key: params[value["set_from_param"]] if isinstance(value, dict) else value
            for key, value in record.items()}


def _process_effects(event: dict, params: dict) -> list[dict]:
    return [
        {
            key: params[value["set_from_param"]] if isinstance(value, dict) else value
            for key, value in effect.items()
        }
        for effect in event["product_effects"]["processCompleted"]
    ]


def _apply_resource_effects(
    models: dict, valuation: dict, after: dict, event: dict, params: dict,
    *, defer_resource_guards: bool = False,
) -> None:
    for participant in event["participants"]:
        peers = [e for e in models[participant]["events"] if e["event_id"] == event["event_id"]]
        if len(peers) != 1 or any(
            peers[0][key] != event[key]
            for key in ("event_name", "parameter_bindings", "participants", "product_effects")
        ):
            raise ValueError("Shared handoff participants disagree")
        local = peers[0]
        output_guard = local.get("collection_guards", {}).get("output.{part_name}")
        if output_guard is not None and not defer_resource_guards:
            remaining = [
                value for field, value in valuation[participant].items()
                if field.startswith("output.") and field != f"output.{params['part_name']}"
            ]
            if output_guard == {"any_other_equals": True}:
                allowed = any(value is True for value in remaining)
            elif output_guard == {"all_other_equals": False}:
                allowed = all(value is False for value in remaining)
            else:
                raise ValueError("Invalid printer output collection guard")
            if not allowed:
                raise ValueError(f"Guard blocked: {participant}.output collection")
        for field, guard in ({} if defer_resource_guards else local["guards"]).items():
            field = _bound(field, params)
            if len(guard) != 1:
                raise ValueError("Invalid resource guard")
            operator, expected = next(iter(guard.items()))
            if operator not in {
                "equals",
                "not_equals",
                "equals_from_param",
                "not_equals_from_param",
            }:
                raise ValueError("Unknown resource guard")
            if operator.endswith("_from_param"):
                expected = params[expected]
            value = valuation[participant][field]
            equal = type(value) is type(expected) and value == expected
            if (operator.startswith("not_equals") and equal) or (
                operator.startswith("equals") and not equal
            ):
                raise ResourceGuardError(participant, field, operator, expected, value)
        for field, update in local["updates"].items():
            field = _bound(field, params)
            if set(update) not in ({"set"}, {"set_from_param"}):
                raise ValueError("Unknown resource effect")
            after[participant][field] = deepcopy(
                update["set"] if "set" in update else params[update["set_from_param"]]
            )


def project_transition(
    models: dict, valuation: dict, products: dict, task: dict, product_name: str, requirements: dict,
    *, defer_resource_guards: bool = False,
) -> tuple[dict, dict]:
    """Predict declared effects without manufacturing execution evidence.

    Recovery successor-union validation may defer nominal resource guards to
    primitive composition. Custody conservation, declared effects, and product
    requirements still apply. Nominal callers retain the default guard checks.
    """
    before_owners = owners(models, valuation, products, product_name)
    event = _task_event(models, products, task)
    _check_product_requirements(task, products, product_name, requirements, event)
    params = task["parameters"]
    if params.get("part_name"):
        _check_product_guards(event, products.get(params["part_name"]), params)
    after, predicted = deepcopy(valuation), deepcopy(products)
    if task["event_name"] == "place_approach" and params.get("destination_location") == "Conveyor":
        loading = models["Conveyor"]["assignments"]["loading_positions"][task["resource_id"]]
        conveyor_load_parameters(
            models["Conveyor"],
            valuation["Conveyor"],
            params["part_name"],
            loading["loading_position"],
        )
    _apply_resource_effects(models, valuation, after, event, params,
                            defer_resource_guards=defer_resource_guards)
    if task["event_name"] == "place_release" and params.get("destination_location") == "Conveyor":
        derived = conveyor_load_parameters(
            models["Conveyor"],
            valuation["Conveyor"],
            params["part_name"],
            params["loading_position"],
        )
        after["Conveyor"].update(
            {field.removeprefix("next_"): value for field, value in derived.items()}
        )
    elif task["event_name"] == "advance_conveyor":
        derived = conveyor_advance_parameters(
            models["Conveyor"],
            valuation["Conveyor"],
            params["next_locations"],
            params["delivered_part"],
        )
        after["Conveyor"].update(
            {field.removeprefix("next_"): value for field, value in derived.items()}
        )
    part = params.get("part_name")
    if task["event_name"] == "place_insert":
        predicted[part].update(location=product_name, target=params["target"])
    if part:
        for result in _process_effects(event, params):
            if result not in predicted[part]["processCompleted"]:
                predicted[part]["processCompleted"].append(result)
        if "current_process" in event["product_effects"]:
            predicted[part]["current_process"] = current_process_after(
                event, predicted[part].get("current_process"), params)
    after_owners = owners(models, after, predicted, product_name)
    # Only an event that makes an output present introduces its part.
    introduces = event["updates"].get("output.{part_name}") == {"set": True}
    expected = set(before_owners) | ({part} if introduces else set())
    if set(after_owners) != expected:
        raise ValueError("Transition does not conserve part custody")
    for name, owner in after_owners.items():
        if owner == product_name:
            continue
        predicted[name].update(product_state_from_custody(name, owner, after[owner], product_name))
    return after, predicted


def product_state_from_custody(part_name: str, owner: str, values: dict, product_name: str) -> dict:
    """Locate a part from the custody its owner's valuation establishes.

    A part's location is the resource that holds it, the staging tray it rests on, or the product
    itself for a product at its assembly place. How it is held follows from that location, so no
    handling state is returned. This function does not establish custody, completion, or physical
    support. Callers must supply the owning resource's validated resulting valuation.
    """
    location = owner
    if values.get("staging_part") == part_name:
        location = f"{owner} staging tray"
    if part_name == product_name and owner == "Exit" and values["part_name"] is None:
        location = product_name
    return {"location": location}


def process_json(models: dict, event_name: str) -> dict:
    """Export complete task models from the descriptors used during discovery."""
    events = {}
    for rid, model in models.items():
        for event in model["events"]:
            if event["event_name"] != event_name:
                continue
            row = events.setdefault(
                event["event_id"],
                {
                    **deepcopy(event),
                    "guards": {},
                    "updates": {},
                    "collection_guards": {},
                    "collection_effects": {},
                },
            )
            row["guards"][rid], row["updates"][rid] = (
                deepcopy(event["guards"]),
                deepcopy(event["updates"]),
            )
            for field in ("collection_guards", "collection_effects"):
                if field in event:
                    row[field][rid] = deepcopy(event[field])
    participants = {rid for event in events.values() for rid in event["participants"]}
    return {
        "schema_version": next(iter(models.values()))["schema_version"],
        "event_name": event_name,
        "events": list(events.values()),
        "validation_source": "cais_spade_llm.resources.environment_models",
        "resources": {
            rid: {
                key: deepcopy(models[rid][key])
                for key in (
                    "state_variables",
                    "current_configuration",
                    "process_capabilities",
                    "execution_support",
                )
            }
            for rid in models
            if rid in participants
        },
    }
