"""Resource-owned transition models for runtime environmental exploration.

The existing task builders supply the configured topology and task programs.
Catalogue-dependent DES expansions are replaced by parameterized declarations;
only a runtime valuation contains the registered workpiece identities. The v1
projector remains available for delivery and its tests.
"""

from __future__ import annotations

import json
import math
from copy import deepcopy
from dataclasses import asdict
from itertools import combinations_with_replacement
from typing import Any

from cais_spade_llm.resources import nominal_des as tasks
from cais_spade_llm.resources.nominal_conveyor import (
    CONVEYOR_LOCATIONS,
    conveyor_advance_parameters,
    conveyor_load_parameters,
    conveyor_parts,
)
from cais_spade_llm.resources.robot.robot_task_registry import robot_task_registry

PART_REFERENCE = {"scope": "resource", "type": ["string", "null"], "reference": "part_name"}
PART_PARAMETER = {"required": True, "type": "string", "reference": "part_name"}


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
                "part_location",
            ):
                domain = models[robot]["state_variables"][field]["domain"]
                domain.extend(value for value in (source, destination) if value not in domain)
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
        model["marked_state_conditions"] = []
        model["events"] = [
            _parameterized_event(event, rid, registry, schema_version)
            for event in model["events"]
        ]
        model["events"] = [event for event in model["events"] if event is not None]
        model["local_event_alphabet"] = list(
            dict.fromkeys(e["event_name"] for e in model["events"])
        )
        model["controllable_event_alphabet"] = model["local_event_alphabet"][:]
        model["observable_event_alphabet"] = model["local_event_alphabet"][:]
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
    return models


def _declare_runtime_state(model: dict, registered: list[str], scene: dict) -> None:
    for field in model["state_variables"]:
        if field in {"held_part", "part_name", "staging_part", "task_ctx.part_name"} or (
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
    event: dict, rid: str, registry: dict, schema_version: int
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
    if name in registry:
        event["program"] = json.loads(json.dumps(asdict(registry[name].program)))
    if (
        not event["guards"]
        and not event["updates"]
        and event["parameter_bindings"]["resource_id"]["equals"] != rid
    ):
        return None
    return event


def _declare_product_effects(event: dict, schema_version: int) -> None:
    event["product_effects"] = {"processCompleted": []}
    if event["event_name"] == "machine_part":
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
    status, reasons = _geometry_feasibility(model, geometry)
    if status == "INFEASIBLE":
        return status, reasons
    if name == "machine_part":
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
            if not check_value(declaration, value, products):
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
        if "staging_part" in values and (
            (values["resource_state"] == "idle") != (values["part_name"] is None)
        ):
            raise ValueError("Machine state disagrees with workholding")
    for part, state in products.items():
        if state.get("state") == "assembled":
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


def _process_effects(event: dict, params: dict) -> list[dict]:
    return [
        {
            key: params[value["set_from_param"]] if isinstance(value, dict) else value
            for key, value in effect.items()
        }
        for effect in event["product_effects"]["processCompleted"]
    ]


def _apply_resource_effects(
    models: dict, valuation: dict, after: dict, event: dict, params: dict
) -> None:
    for participant in event["participants"]:
        peers = [e for e in models[participant]["events"] if e["event_id"] == event["event_id"]]
        if len(peers) != 1 or any(
            peers[0][key] != event[key]
            for key in ("event_name", "parameter_bindings", "participants", "product_effects")
        ):
            raise ValueError("Shared handoff participants disagree")
        local = peers[0]
        for field, guard in local["guards"].items():
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
                raise ValueError(f"Guard blocked: {participant}.{field}")
        for field, update in local["updates"].items():
            field = _bound(field, params)
            if set(update) not in ({"set"}, {"set_from_param"}):
                raise ValueError("Unknown resource effect")
            after[participant][field] = deepcopy(
                update["set"] if "set" in update else params[update["set_from_param"]]
            )


def project_transition(
    models: dict, valuation: dict, products: dict, task: dict, product_name: str, requirements: dict
) -> tuple[dict, dict]:
    """Predict an atomic transition without manufacturing execution evidence."""
    before_owners = owners(models, valuation, products, product_name)
    event = _task_event(models, products, task)
    _check_product_requirements(task, products, product_name, requirements, event)
    params = task["parameters"]
    after, predicted = deepcopy(valuation), deepcopy(products)
    if task["event_name"] == "place_approach" and params.get("destination_location") == "Conveyor":
        loading = models["Conveyor"]["assignments"]["loading_positions"][task["resource_id"]]
        conveyor_load_parameters(
            models["Conveyor"],
            valuation["Conveyor"],
            params["part_name"],
            loading["loading_position"],
        )
    _apply_resource_effects(models, valuation, after, event, params)
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
        predicted[part].update(state="assembled", location=product_name, target=params["target"])
    if part:
        for result in _process_effects(event, params):
            if result not in predicted[part]["processCompleted"]:
                predicted[part]["processCompleted"].append(result)
    after_owners = owners(models, after, predicted, product_name)
    expected = set(before_owners) | ({part} if task["event_name"] == "print_part" else set())
    if set(after_owners) != expected:
        raise ValueError("Transition does not conserve part custody")
    for name, owner in after_owners.items():
        if predicted[name].get("state") == "assembled":
            continue
        location = owner
        state = "ready"
        if after[owner].get("held_part") == name:
            state = after[owner].get("part_state") or "in_gripper"
        elif after[owner].get("part_name") == name:
            state = "completed" if owner == "Exit" else after[owner]["resource_state"]
        elif after[owner].get("staging_part") == name:
            location = f"{owner} staging tray"
        if name == product_name and owner == "Exit" and after["Exit"]["part_name"] is None:
            location = product_name
        predicted[name].update(location=location, state=state)
    return after, predicted


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
