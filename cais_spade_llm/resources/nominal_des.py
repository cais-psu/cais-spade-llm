"""V1 nominal DES compatibility for delivery execution and historical traces.

Runtime environmental matching uses resources.environment_models. The task
builders here also supply its configured topology and existing robot programs.

Task events bind part parameters from resource assignments. Exact identities
remain in the configured inventory and concrete state valuations. Indexed fields
such as inventory.{part_name} are resolved only after checking task eligibility.

Descriptors and their pure symbolic projection have no agent registration, ROS
imports, commands, or persistence. An event denotes an acknowledged completed
task. Input acknowledgments are assumptions of a symbolic trace, not fabricated
live observations. Physical pose, grasp, interlock, and timing checks remain the
responsibility of the existing or future resource controller.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from cais_spade_llm.resources.nominal_conveyor import (
    CONVEYOR_LOCATIONS,
    conveyor_advance_parameters,
    conveyor_load_parameters,
    conveyor_parts,
)
from cais_spade_llm.resources.robot.robot_task_registry import robot_task_registry


def _model(resource_id: str, assignments: dict[str, Any], support: str) -> dict[str, Any]:
    return {
        "resource_id": resource_id,
        "model_type": "extended_finite_automaton",
        "state_variables": {},
        "current_valuation": {},
        "events": [],
        "marked_state_conditions": [],
        "assignments": deepcopy(assignments),
        "execution_support": support,
        "notes": [],
    }


def _variable(
    model: dict, name: str, domain: list, initial: Any, *, scope: str = "resource"
) -> None:
    if initial not in domain:
        raise ValueError(f"Initial value outside {model['resource_id']}.{name} domain")
    model["state_variables"][name] = {"scope": scope, "domain": deepcopy(domain)}
    model["current_valuation"][name] = deepcopy(initial)


def _event(
    models: dict,
    name: str,
    bindings: dict,
    guards: dict,
    updates: dict,
    *,
    capability: tuple[dict, dict],
    parameters: dict | None = None,
    notes: str = "",
) -> None:
    participants = [
        rid for rid in dict.fromkeys([*guards, *updates]) if guards.get(rid) or updates.get(rid)
    ]
    event_id = sum(len(model["events"]) for model in models.values())
    for resource_id in participants:
        models[resource_id]["events"].append(
            {
                "event_id": event_id,
                "event_name": name,
                "controllable": True,
                "observable": True,
                "parameter_bindings": {
                    **{key: {"equals": value} for key, value in bindings.items()},
                    **deepcopy(parameters or {}),
                },
                "capability_transition": {
                    "source": deepcopy(capability[0]),
                    "target": deepcopy(capability[1]),
                },
                "product_effects": {
                    "processCompleted": [name]
                    if name in {"machine_part", "print_part", "place_insert"}
                    else [],
                },
                "participants": participants,
                "guards": deepcopy(guards.get(resource_id, {})),
                "updates": deepcopy(updates.get(resource_id, {})),
                "notes": notes,
            }
        )


def _part_parameter(resource_id: str, field: str = "nominal_parts") -> dict:
    return {
        "required": True,
        "from_assignment": {"resource_id": resource_id, "field": field},
    }


def _equals(**values: Any) -> dict:
    return {key: {"equals": value} for key, value in values.items()}


def _sets(**values: Any) -> dict:
    return {key: {"set": value} for key, value in values.items()}


def _initial_models(scene: dict) -> dict[str, dict]:
    models = {}
    for row in scene["robots"]:
        rid = row["resource_id"]
        models[rid] = _model(
            rid,
            row,
            "Robot task definitions exist; this resource has no nominal execution binding.",
        )
    for row in scene["machines"]:
        rid = row["resource_id"]
        models[rid] = _model(rid, row, "Nominal model only; machining execution is not connected.")
    for rid in (
        "Storage",
        "KMR",
        "Conveyor",
        "Buffer For Machined parts",
        "3D Printing Station",
        "Exit",
    ):
        models[rid] = _model(
            rid, scene[rid], "Nominal model only; task execution is not connected."
        )
    kmr = scene["KMR"]
    models["KMR"]["execution_support"] = (
        f"DockKMR controller binding: {kmr['docking_action']}; "
        f"simulation_control_integrated={kmr['simulation_control_integrated']}. "
        "Nominal task execution and part handling are not connected."
    )
    if kmr.get("integrated") and kmr.get("task_execution"):
        models["KMR"]["execution_support"] = (
            "Gazebo execution: pick_part, move_to_resource, place_release for "
            "assembly_board-v1-kmr-storage-m1.json only. "
            f"DockKMR controller binding: {kmr['docking_action']}. "
            "Other orders and physical execution are not integrated."
        )
        models["Storage"]["execution_support"] = "Participating inventory context for acknowledged KMR Storage-to-M1 pickup."
        models["M1"]["execution_support"] = "Participating loading context for acknowledged KMR Storage-to-M1 placement; machining is not integrated."
    for rid in ("Conveyor", "Buffer For Machined parts"):
        models[rid]["execution_support"] = (
            f"transport_enabled={scene[rid]['transport_enabled']}; nominal model only."
        )
    return models


def _declare_states(models: dict, scene: dict) -> None:
    pegs = [part for machine in scene["machines"] for part in machine["nominal_parts"]]
    gears = scene["3D Printing Station"]["supported_products"]
    product = scene["Exit"]["completed_product"]
    machines = {row["handling_robot"]: row for row in scene["machines"]}
    for row in scene["robots"]:
        robot = row["resource_id"]
        model = models[robot]
        if robot in machines:
            machine = machines[robot]
            tray = f"{machine['resource_id']} staging tray"
            parts = machine["nominal_parts"]
            sources = [machine["resource_id"], tray]
            destinations = ["Conveyor", tray]
        elif robot == scene["Buffer For Machined parts"]["handling_robot"]:
            parts = [*pegs, product]
            sources = ["Buffer For Machined parts", product]
            destinations = [product, "Exit"]
        else:
            parts = gears
            sources = ["3D Printing Station"]
            destinations = [product]
        locations = list(dict.fromkeys([*sources, *destinations, "home"]))
        _variable(
            model, "resource_state", ["idle", "at_pick", "picked", "positioned", "placed"], "idle"
        )
        _variable(model, "held_part", [None, *parts], None)
        _variable(model, "resource_location", [None, *locations], None)
        _variable(model, "task_ctx.origin_resource_location", [None, *sources], None)
        _variable(model, "task_ctx.destination_location", [None, *destinations], None)
        _variable(model, "task_ctx.part_name", [None, *parts], None)
        _variable(
            model,
            "part_state",
            [
                None,
                "ready",
                "in_gripper",
                "in_transit",
                *(["assembled"] if robot not in machines else []),
            ],
            None,
            scope="part",
        )
        _variable(
            model,
            "part_location",
            [None, *dict.fromkeys([*sources, robot, *destinations])],
            None,
            scope="part",
        )
        model["marked_state_conditions"] = [_equals(resource_state="idle", held_part=None)]
        model["notes"] = [
            "Initial idle/empty-gripper values are modeling assumptions, not observations.",
            "Existing pose, gripper, and trajectory checks remain in task execution.",
        ]
    for machine in scene["machines"]:
        model = models[machine["resource_id"]]
        _variable(model, "resource_state", ["idle", "loaded", "completed"], "idle")
        for field in ("part_name", "staging_part"):
            _variable(model, field, [None, *machine["nominal_parts"]], None)
        model["marked_state_conditions"] = [
            _equals(resource_state="idle", part_name=None, staging_part=None)
        ]
        model["notes"] = [
            "Initially idle with empty workholding and staging is a modeling assumption."
        ]
    storage = models["Storage"]
    for part in scene["Storage"]["slots"]:
        _variable(storage, f"inventory.{part}", [False, True], True)
    storage["marked_state_conditions"] = [
        {field: {"equals": False} for field in storage["state_variables"]}
    ]
    kmr = models["KMR"]
    route_resources = list(
        dict.fromkeys(rid for route in scene["KMR"]["predefined_routes"] for rid in route)
    )
    _variable(kmr, "resource_state", ["idle", "carrying"], "idle")
    _variable(kmr, "held_part", [None, *pegs], None)
    _variable(kmr, "resource_location", route_resources, "Storage")
    kmr["marked_state_conditions"] = [
        _equals(resource_state="idle", held_part=None, resource_location="Storage")
    ]
    kmr["notes"] = [
        "Initially empty at Storage is a modeling assumption. Routes are instance configuration."
    ]
    conveyor = models["Conveyor"]
    conveyor["assignments"]["nominal_parts"] = pegs
    conveyor["assignments"]["buffer_zone_1_pose"] = scene["Buffer For Machined parts"]["zones"][0][
        "pose"
    ]
    conveyor["assignments"]["loading_positions"] = {
        row["handling_robot"]: {
            "resource_id": row["resource_id"],
            "loading_position": f"loading_position_{index}",
            "pose": row["conveyor_loading_pose"],
        }
        for index, row in enumerate(
            sorted(scene["machines"], key=lambda row: row["conveyor_loading_pose"][0]), 1
        )
    }
    _variable(conveyor, "belt_stopped", [False, True], True)
    _variable(
        conveyor, "loading_reserved_by", [None, *conveyor["assignments"]["loading_positions"]], None
    )
    for part in pegs:
        _variable(conveyor, f"part_location.{part}", [None, *CONVEYOR_LOCATIONS], None)
        _variable(conveyor, f"part_order.{part}", [None, *range(len(pegs))], None)
    conveyor["marked_state_conditions"] = [
        {
            **_equals(belt_stopped=True, loading_reserved_by=None),
            **{f"part_location.{part}": {"equals": None} for part in pegs},
        }
    ]
    conveyor["notes"] = [
        "part_order 0 is the leading part. Intermediate regions describe order, not distance.",
        "Movement advances the shared belt; regions are not independently driven zones.",
        "A completed place_release reserves and clears its loading area within the task. A pending external reservation blocks advance_conveyor.",
        "Start transport after one acknowledged load and robot clearance; no second load is required. Stop at a clear requested loading opportunity or output handoff.",
        "Exact spacing, stopping distances, and motion feasibility require controller validation.",
    ]
    buffer = models["Buffer For Machined parts"]
    buffer["assignments"]["nominal_parts"] = pegs
    for zone in scene["Buffer For Machined parts"]["zones"]:
        _variable(buffer, f"zone_{zone['zone']}_part", [None, *pegs], zone["initial_part"])
    buffer["marked_state_conditions"] = [
        {field: {"equals": None} for field in buffer["state_variables"]}
    ]
    printer = models["3D Printing Station"]
    for part in gears:
        _variable(
            printer,
            f"output.{part}",
            [False, True],
            part in scene["3D Printing Station"]["initial_products"],
        )
    printer["marked_state_conditions"] = [
        {field: {"equals": False} for field in printer["state_variables"]}
    ]
    printer["notes"] = [
        f"Configured initial_state: {scene['3D Printing Station']['initial_state']}. Each true output is completed and available for collection."
    ]
    exit_model = models["Exit"]
    _variable(exit_model, "resource_state", ["empty", "occupied"], scene["Exit"]["initial_state"])
    _variable(exit_model, "part_name", [None, product], None)
    _variable(
        exit_model,
        "product_location",
        [product, scene["Exit"]["handling_robot"], "Exit"],
        product,
        scope="part",
    )
    exit_model["marked_state_conditions"] = [_equals(resource_state="occupied", part_name=product)]


def _machine_and_mobile_events(models: dict, scene: dict) -> None:
    for source, target in scene["KMR"]["predefined_routes"]:
        for start, end in ((source, target), (target, source)):
            _event(
                models,
                "move_to_resource",
                {
                    "resource_id": "KMR",
                    "target_resource": end,
                    "source_resource": start,
                    "arrival_acknowledged": True,
                    "arm_parked": True,
                },
                {"KMR": _equals(resource_location=start)},
                {"KMR": _sets(resource_location=end)},
                capability=({"resource_location": start}, {"resource_location": end}),
                notes="Retain held_part while moving along the configured route.",
            )
    _event(
        models,
        "pick_part",
        {
            "resource_id": "KMR",
            "origin_resource_location": "Storage",
            "handoff_acknowledged": True,
        },
        {
            "KMR": _equals(resource_state="idle", resource_location="Storage", held_part=None),
            "Storage": {"inventory.{part_name}": {"equals": True}},
        },
        {
            "KMR": {
                **_sets(resource_state="carrying"),
                "held_part": {"set_from_param": "part_name"},
            },
            "Storage": {"inventory.{part_name}": {"set": False}},
        },
        parameters={"part_name": _part_parameter("Storage", "slots")},
        capability=({"part_location": "Storage"}, {"part_location": "KMR"}),
    )
    for machine in scene["machines"]:
        rid = machine["resource_id"]
        parameters = {"part_name": _part_parameter(rid)}
        _event(
            models,
            "place_release",
            {
                "resource_id": "KMR",
                "destination_location": rid,
                "handoff_acknowledged": True,
                "robot_clear": True,
            },
            {
                "KMR": {
                    **_equals(resource_state="carrying", resource_location=rid),
                    "held_part": {"equals_from_param": "part_name"},
                },
                rid: _equals(resource_state="idle", part_name=None),
            },
            {
                "KMR": _sets(resource_state="idle", held_part=None),
                rid: {
                    **_sets(resource_state="loaded"),
                    "part_name": {"set_from_param": "part_name"},
                },
            },
            parameters=parameters,
            capability=(
                {"part_location": "KMR"},
                {"part_location": rid, f"{rid}.resource_state": "loaded"},
            ),
        )
        _event(
            models,
            "machine_part",
            {
                "resource_id": rid,
                "machining_acknowledged": True,
                "robot_clear": True,
            },
            {
                rid: {
                    **_equals(resource_state="loaded"),
                    "part_name": {"equals_from_param": "part_name"},
                }
            },
            {rid: _sets(resource_state="completed")},
            parameters=parameters,
            capability=(
                {"part_location": rid, f"{rid}.resource_state": "loaded"},
                {"part_location": rid, f"{rid}.resource_state": "completed"},
            ),
        )


def _robot_task_event(
    models: dict,
    robot: str,
    name: str,
    bindings: dict,
    guards: dict,
    updates: dict,
    *,
    capability: tuple[dict, dict],
    parameters: dict | None = None,
) -> None:
    task = robot_task_registry()[name]
    local_guards = deepcopy(guards)
    local_updates = deepcopy(updates)
    if task.program.entry_state != "any":
        local_guards.setdefault(robot, {})["resource_state"] = {"equals": task.program.entry_state}
    local_updates.setdefault(robot, {})["resource_state"] = {"set": task.program.success_state}
    _event(
        models,
        name,
        {"resource_id": robot, **bindings},
        local_guards,
        local_updates,
        capability=capability,
        parameters=parameters,
        notes=f"Nominal transition from {task.source}: {task.program.modeled_transition()}.",
    )


def _robot_pick_events(
    models: dict,
    robot: str,
    source: str,
    owner: str,
    source_guards: dict,
    source_updates: dict,
    *,
    inputs: dict | None = None,
) -> None:
    bindings = {"origin_resource_location": source, **(inputs or {})}
    parameters = {"part_name": _part_parameter(robot)}
    source_state = {"part_location": source}
    if "resource_state" in source_guards:
        source_state[f"{owner}.resource_state"] = source_guards["resource_state"]["equals"]
    if "zone_4_part" in source_guards:
        source_state["zone"] = 4
    if "output.{part_name}" in source_guards:
        source_state["output.{part_name}"] = source_guards["output.{part_name}"]["equals"]
    _robot_task_event(
        models,
        robot,
        "pick_approach",
        bindings,
        {
            robot: _equals(held_part=None),
            owner: source_guards,
        },
        {
            robot: {
                **_sets(resource_location=source),
                "task_ctx.origin_resource_location": {"set": source},
                "task_ctx.part_name": {"set_from_param": "part_name"},
            }
        },
        parameters=parameters,
        capability=(source_state, source_state),
    )
    _robot_task_event(
        models,
        robot,
        "pick_grasp",
        {**bindings, "handoff_acknowledged": True, "source_clear": True, "robot_clear": True},
        {
            robot: {
                **_equals(held_part=None),
                "task_ctx.origin_resource_location": {"equals": source},
                "task_ctx.part_name": {"equals_from_param": "part_name"},
            },
            owner: source_guards,
        },
        {
            robot: {
                **_sets(part_state="in_gripper", part_location=robot),
                "held_part": {"set_from_param": "part_name"},
            },
            owner: source_updates,
        },
        parameters=parameters,
        capability=(source_state, {"part_location": robot, "part_state": "in_gripper"}),
    )


def _robot_place_events(
    models: dict,
    robot: str,
    destination: str,
    owner: str,
    destination_guards: dict,
    destination_updates: dict,
    *,
    inputs: dict | None = None,
    fixed_part: str | None = None,
) -> None:
    assembly = destination == models["Exit"]["assignments"]["completed_product"]
    bindings = {"destination_location": destination}
    parameters = {
        "part_name": {"equals": fixed_part} if fixed_part is not None else _part_parameter(robot)
    }
    in_gripper = {"part_location": robot, "part_state": "in_gripper"}
    positioned = {
        "part_location": robot,
        "part_state": "in_transit",
        "task_ctx.destination_location": destination,
    }
    downstream_guards = (
        {"Buffer For Machined parts": _equals(zone_1_part=None)}
        if destination == "Conveyor"
        else {}
    )
    _robot_task_event(
        models,
        robot,
        "place_approach",
        bindings,
        {
            robot: {"held_part": {"equals_from_param": "part_name"}},
            owner: destination_guards,
            **downstream_guards,
        },
        {
            robot: {
                **_sets(
                    resource_location=destination, part_state="in_transit", part_location=robot
                ),
                "task_ctx.destination_location": {"set": destination},
            }
        },
        parameters=parameters,
        capability=(in_gripper, positioned),
    )
    guards = {
        robot: {
            **_equals(resource_state="positioned"),
            "held_part": {"equals_from_param": "part_name"},
            "task_ctx.destination_location": {"equals": destination},
        },
        owner: destination_guards,
        **downstream_guards,
    }
    updates = {
        robot: {
            **_sets(resource_state="placed", held_part=None, part_location=destination),
            **{
                field: {"set": None}
                for field in models[robot]["state_variables"]
                if field.startswith("task_ctx.")
            },
        },
        owner: destination_updates,
    }
    if assembly:
        guards.pop(owner)
        updates.pop(owner)
        updates[robot]["part_state"] = {"set": "assembled"}
        guards[robot]["assembled.{part_name}"] = {"equals": False}
        updates[robot]["assembled.{part_name}"] = {"set": True}
        _robot_task_event(
            models,
            robot,
            "place_insert",
            {
                **bindings,
                "handoff_acknowledged": True,
                "assembly_acknowledged": True,
            },
            guards,
            updates,
            parameters=parameters,
            capability=(positioned, {"part_location": destination, "part_state": "assembled"}),
        )
    else:
        _event(
            models,
            "place_release",
            {
                "resource_id": robot,
                **bindings,
                "handoff_acknowledged": True,
                "robot_clear": True,
                **(inputs or {}),
            },
            guards,
            updates,
            parameters=parameters,
            capability=(
                positioned,
                {
                    "part_location": destination,
                    **(
                        {f"{owner}.resource_state": destination_updates["resource_state"]["set"]}
                        if "resource_state" in destination_updates
                        else {}
                    ),
                },
            ),
            notes="Ordinary placement transfers custody without declaring assembled.",
        )


def _robot_events(models: dict, scene: dict) -> None:
    conveyor = models["Conveyor"]
    pegs = conveyor["assignments"]["nominal_parts"]
    for machine in scene["machines"]:
        rid, robot = machine["resource_id"], machine["handling_robot"]
        tray = f"{rid} staging tray"
        loading = conveyor["assignments"]["loading_positions"][robot]["loading_position"]
        other_robot = next(
            rid for rid in conveyor["assignments"]["loading_positions"] if rid != robot
        )
        models[robot]["assignments"].update(
            machine=rid, nominal_parts=machine["nominal_parts"], destinations=["Conveyor", tray]
        )
        _robot_pick_events(
            models,
            robot,
            rid,
            rid,
            {
                **_equals(resource_state="completed"),
                "part_name": {"equals_from_param": "part_name"},
            },
            _sets(resource_state="idle", part_name=None),
        )
        _robot_pick_events(
            models,
            robot,
            tray,
            rid,
            {"staging_part": {"equals_from_param": "part_name"}},
            _sets(staging_part=None),
        )
        _robot_place_events(
            models,
            robot,
            tray,
            rid,
            _equals(staging_part=None),
            {"staging_part": {"set_from_param": "part_name"}},
        )
        _robot_place_events(
            models,
            robot,
            "Conveyor",
            "Conveyor",
            {
                **_equals(belt_stopped=True),
                "loading_reserved_by": {"not_equals": other_robot},
                **{f"part_location.{peg}": {"not_equals": loading} for peg in pegs},
                "part_location.{part_name}": {"equals": None},
            },
            {
                "part_location.{part_name}": {"set": loading},
                "loading_reserved_by": {"set": None},
                **{
                    f"part_order.{peg}": {"set_from_param": f"next_part_order.{peg}"}
                    for peg in pegs
                },
            },
            inputs={"reserved_by": robot, "loading_position": loading},
        )
    product = scene["Exit"]["completed_product"]
    buffer = "Buffer For Machined parts"
    assembly_robot = scene[buffer]["handling_robot"]
    printer = "3D Printing Station"
    printer_robot = scene[printer]["handling_robot"]
    for source, robot, parts in (
        (buffer, assembly_robot, pegs),
        (printer, printer_robot, scene[printer]["supported_products"]),
    ):
        models[robot]["assignments"].update(source=source, nominal_parts=parts, destination=product)
        for part in parts:
            _variable(models[robot], f"assembled.{part}", [False, True], False, scope="part")
        field = "zone_4_part" if source == buffer else "output.{part_name}"
        present = {"equals_from_param": "part_name"} if source == buffer else {"equals": True}
        absent = None if source == buffer else False
        _robot_pick_events(
            models,
            robot,
            source,
            source,
            {field: present},
            {field: {"set": absent}},
            inputs={
                "part_identity_observed": True,
                **(
                    {
                        "zone_stopped": True,
                        "stop_raised": True,
                        "assembly_destination_available": True,
                    }
                    if source == buffer
                    else {}
                ),
            },
        )
        _robot_place_events(models, robot, product, "Exit", {}, {})
    exit_robot = scene["Exit"]["handling_robot"]
    completion_guards = {
        robot: {
            f"assembled.{part}": {"equals": True}
            for part in models[robot]["assignments"]["nominal_parts"]
        }
        for robot in (assembly_robot, printer_robot)
    }
    completion_guards["Exit"] = _equals(part_name=None, product_location=product)
    completion_guards[exit_robot].update(_equals(held_part=None))
    product_bindings = {
        "part_name": product,
        "origin_resource_location": product,
        "product_completed": True,
    }
    _robot_task_event(
        models,
        exit_robot,
        "pick_approach",
        product_bindings,
        completion_guards,
        {
            exit_robot: {
                **_sets(resource_location=product),
                "task_ctx.origin_resource_location": {"set": product},
                "task_ctx.part_name": {"set": product},
            },
        },
        capability=({"part_location": product}, {"part_location": product}),
    )
    completion_guards[exit_robot].update(
        {
            "task_ctx.origin_resource_location": {"equals": product},
            "task_ctx.part_name": {"equals": product},
        }
    )
    _robot_task_event(
        models,
        exit_robot,
        "pick_grasp",
        {
            **product_bindings,
            "handoff_acknowledged": True,
            "source_clear": True,
            "robot_clear": True,
        },
        completion_guards,
        {
            exit_robot: _sets(held_part=product, part_state="in_gripper", part_location=exit_robot),
            "Exit": _sets(product_location=exit_robot),
        },
        capability=(
            {"part_location": product},
            {"part_location": exit_robot, "part_state": "in_gripper"},
        ),
    )
    _robot_place_events(
        models,
        exit_robot,
        "Exit",
        "Exit",
        _equals(resource_state="empty", part_name=None, product_location=exit_robot),
        _sets(resource_state="occupied", part_name=product, product_location="Exit"),
        fixed_part=product,
    )
    for row in scene["robots"]:
        robot = row["resource_id"]
        _robot_task_event(
            models,
            robot,
            "move_home",
            {"home_available": True},
            {robot: _equals(held_part=None)},
            {
                robot: {
                    **_sets(resource_location="home"),
                    **{
                        field: {"set": None}
                        for field in models[robot]["state_variables"]
                        if field.startswith("task_ctx.")
                    },
                }
            },
            capability=(
                {
                    "resource_id": robot,
                    "resource_state": robot_task_registry()["move_home"].program.entry_state,
                },
                {
                    "resource_id": robot,
                    "resource_location": "home",
                    "resource_state": robot_task_registry()["move_home"].program.success_state,
                },
            ),
        )


def _transport_and_print_events(models: dict, scene: dict) -> None:
    conveyor = models["Conveyor"]
    buffer = "Buffer For Machined parts"
    all_updates = {
        field: {"set_from_param": f"next_{field}"}
        for field in conveyor["state_variables"]
        if field.startswith(("part_location.", "part_order."))
    }
    common = {"resource_id": "Conveyor", "robot_clear": True, "drives_synchronized": True}
    for handoff in (False, True):
        guards = {
            "Conveyor": _equals(belt_stopped=True, loading_reserved_by=None),
            buffer: _equals(zone_1_part=None),
        }
        updates = {"Conveyor": all_updates, buffer: {}}
        bindings = {**common, "downstream_reserved": True}
        parameters = {}
        if handoff:
            parameters["delivered_part"] = _part_parameter("Conveyor")
            guards["Conveyor"]["part_location.{delivered_part}"] = {"equals": "output_nest"}
            updates[buffer] = {"zone_1_part": {"set_from_param": "delivered_part"}}
            bindings.update(
                handoff_acknowledged=True, source_clear=True, destination_acknowledged=True
            )
        else:
            bindings["delivered_part"] = None
        _event(
            models,
            "advance_conveyor",
            bindings,
            guards,
            updates,
            parameters={**parameters, "next_locations": {"required": True}},
            capability=(
                {"part_location": "Conveyor"},
                {"part_location": buffer, "zone": 1} if handoff else {"part_location": "Conveyor"},
            ),
            notes="Bind every remaining part's downstream region. Preserve spatial order; move the shared belt together. Deliver only its leading part with acknowledged arrival and departure. Stop for a clear loading opportunity or output handoff.",
        )
    for zone in scene[buffer]["zones"]:
        downstream = zone["downstream_zone"]
        if downstream is None:
            continue
        src, dst = f"zone_{zone['zone']}_part", f"zone_{downstream}_part"
        _event(
            models,
            "advance_part",
            {
                "resource_id": buffer,
                "zone": zone["zone"],
                "downstream_zone": downstream,
                "downstream_reserved": True,
                "robot_clear": True,
                "drives_synchronized": True,
                "source_clear": True,
                "destination_acknowledged": True,
            },
            {buffer: {src: {"equals_from_param": "part_name"}, dst: {"equals": None}}},
            {buffer: {src: {"set": None}, dst: {"set_from_param": "part_name"}}},
            parameters={"part_name": _part_parameter(buffer)},
            capability=(
                {"part_location": buffer, "zone": zone["zone"]},
                {"part_location": buffer, "zone": downstream},
            ),
        )
    printer = "3D Printing Station"
    _event(
        models,
        "print_part",
        {"resource_id": printer, "printing_acknowledged": True},
        {printer: {"output.{part_name}": {"equals": False}}},
        {printer: {"output.{part_name}": {"set": True}}},
        parameters={"part_name": _part_parameter(printer, "supported_products")},
        capability=(
            {"part_location": printer, "output.{part_name}": False},
            {"part_location": printer, "output.{part_name}": True},
        ),
    )


def build_nominal_resource_des_models(scene: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Build the twelve resource descriptors from the configured nominal scene.

    Args:
        scene: Parsed recovery_framework_gazebo.json, with exact identifiers.

    Returns:
        Descriptors keyed by resource identifier, without changing the input.

    Raises:
        ValueError: Resource identities, peg inventory, or buffer topology are inconsistent.
        KeyError: A required scene field is missing.
    """
    robot_ids = [row["resource_id"] for row in scene["robots"]]
    machine_ids = [row["resource_id"] for row in scene["machines"]]
    fixed_resources = {
        "Storage",
        "KMR",
        "Conveyor",
        "Buffer For Machined parts",
        "3D Printing Station",
        "Exit",
    }
    pegs = [part for row in scene["machines"] for part in row["nominal_parts"]]
    if len(robot_ids) != 4 or len(machine_ids) != 2 or len(set([*robot_ids, *machine_ids])) != 6:
        raise ValueError(
            "The nominal scene requires four distinct robots and two distinct machines"
        )
    if fixed_resources.intersection([*robot_ids, *machine_ids]):
        raise ValueError("Resource identifiers must not collide with the named scene resources")
    handlers = [row["handling_robot"] for row in scene["machines"]]
    handlers.extend(
        scene[rid]["handling_robot"] for rid in ("Buffer For Machined parts", "3D Printing Station")
    )
    if (
        set(handlers) != set(robot_ids)
        or scene["Exit"]["handling_robot"] != scene["Buffer For Machined parts"]["handling_robot"]
    ):
        raise ValueError("Nominal handling_robot assignments must match the four configured robots")
    if len(pegs) != len(set(pegs)) or set(pegs) != set(scene["Storage"]["slots"]):
        raise ValueError("nominal_parts must partition the exact Storage inventory")
    zones = scene["Buffer For Machined parts"]["zones"]
    if [zone["zone"] for zone in zones] != [1, 2, 3, 4] or [
        zone["downstream_zone"] for zone in zones
    ] != [2, 3, 4, None]:
        raise ValueError("The nominal buffer requires the configured four-zone chain")
    if any(zone["capacity"] != 1 for zone in zones):
        raise ValueError("The nominal buffer requires capacity one per zone")
    if scene["Buffer For Machined parts"]["capacity"] != len(zones):
        raise ValueError("Buffer capacity must equal its configured zone count")
    if scene["Conveyor"]["normal_output_transfer"] != "Buffer For Machined parts":
        raise ValueError("Conveyor normal_output_transfer must name Buffer For Machined parts")
    loading_x = sorted(row["conveyor_loading_pose"][0] for row in scene["machines"])
    if (
        not loading_x[0]
        < loading_x[1]
        < scene["Conveyor"]["output_nest_pose"][0]
        < zones[0]["pose"][0]
    ):
        raise ValueError(
            "Conveyor loading positions, output_nest, and buffer zone 1 must be downstream in order"
        )
    if scene["Exit"]["capacity"] != 1 or any(
        row["staging_capacity"] != 1 for row in scene["machines"]
    ):
        raise ValueError("Exit and machine staging require their configured capacity of one")
    models = _initial_models(scene)
    _declare_states(models, scene)
    _machine_and_mobile_events(models, scene)
    _robot_events(models, scene)
    _transport_and_print_events(models, scene)
    for model in models.values():
        model["local_event_alphabet"] = list(
            dict.fromkeys(event["event_name"] for event in model["events"])
        )
        model["controllable_event_alphabet"] = model["local_event_alphabet"][:]
        model["observable_event_alphabet"] = model["local_event_alphabet"][:]
    _check_valuation(models, initial_nominal_valuation(models))
    return models


def initial_nominal_valuation(models: dict[str, dict]) -> dict[str, dict]:
    """Return an independent copy of the configured/assumed initial valuations.

    Args:
        models: Nominal resource descriptors.

    Returns:
        Resource valuations suitable for an offline symbolic trace.
    """
    return {rid: deepcopy(model["current_valuation"]) for rid, model in models.items()}


def _matches_bindings(models: dict, event: dict, parameters: dict) -> bool:
    bindings = event["parameter_bindings"]
    if set(parameters) != set(bindings):
        return False
    for name, binding in bindings.items():
        value = parameters[name]
        if "equals" in binding and (
            type(value) is not type(binding["equals"]) or value != binding["equals"]
        ):
            return False
        if "from_assignment" in binding:
            reference = binding["from_assignment"]
            domain = models[reference["resource_id"]]["assignments"][reference["field"]]
            if not any(type(value) is type(item) and value == item for item in domain):
                return False
    return True


def _bound_field(model: dict, field: str, parameters: dict) -> str:
    # Only declared part parameters address indexed facts. Never evaluate an
    # expression or change an identifier supplied by scene configuration.
    for parameter in ("part_name", "delivered_part"):
        suffix = f".{{{parameter}}}"
        if field.endswith(suffix):
            value = parameters.get(parameter)
            if not isinstance(value, str):
                raise ValueError(f"Unbound nominal state field: {field}")
            field = field[: -len(suffix)] + "." + value
            break
    if field not in model["state_variables"]:
        raise ValueError(f"Undeclared nominal state field: {model['resource_id']}.{field}")
    return field


def _check_valuation(models: dict, valuation: dict) -> None:
    if set(valuation) != set(models):
        raise ValueError("Nominal valuation must contain every exact resource identifier")
    for rid, model in models.items():
        if set(valuation[rid]) != set(model["state_variables"]):
            raise ValueError(f"Incomplete nominal valuation for {rid}")
        for field, declaration in model["state_variables"].items():
            value = valuation[rid][field]
            if not any(
                type(value) is type(item) and value == item for item in declaration["domain"]
            ):
                raise ValueError(f"Out-of-domain nominal value: {rid}.{field}")
    conveyor_parts(models["Conveyor"], valuation["Conveyor"])
    _part_owners(models, valuation)
    for rid in models:
        values = valuation[rid]
        if "held_part" in values:
            carrying = values["resource_state"] in {"carrying", "picked", "positioned"}
            if carrying != (values["held_part"] is not None):
                raise ValueError(f"Resource state disagrees with held_part: {rid}")
        if "staging_part" in values and (values["resource_state"] == "idle") != (
            values["part_name"] is None
        ):
            raise ValueError(f"Machine state disagrees with workholding: {rid}")
    exit_state = valuation["Exit"]
    if (exit_state["resource_state"] == "occupied") != (exit_state["part_name"] is not None):
        raise ValueError("Exit state disagrees with occupancy")
    product_location = exit_state["product_location"]
    if product_location == "Exit" and exit_state["part_name"] is None:
        raise ValueError("Completed product location disagrees with Exit custody")
    if (
        product_location in models
        and product_location != "Exit"
        and valuation[product_location].get("held_part")
        != models["Exit"]["assignments"]["completed_product"]
    ):
        raise ValueError("Completed product location disagrees with robot custody")


def _part_owners(models: dict, valuation: dict) -> dict[str, str]:
    owners: dict[str, str] = {}
    product = models["Exit"]["assignments"]["completed_product"]
    for rid, values in valuation.items():
        parts = []
        for field, value in values.items():
            if field in {"held_part", "part_name", "staging_part"} or (
                field.startswith("zone_") and field.endswith("_part")
            ):
                if value is not None:
                    parts.append(value)
            elif (field.startswith(("inventory.", "output.", "assembled.")) and value is True) or (
                rid == "Conveyor" and field.startswith("part_location.") and value is not None
            ):
                parts.append(field.split(".", 1)[1])
            elif field == "product_location" and value == product:
                parts.append(product)
        for part in parts:
            if part in owners:
                raise ValueError(f"Duplicate nominal part custody: {part}")
            owners[part] = rid
    required = {*models["Conveyor"]["assignments"]["nominal_parts"], product}
    if not required <= owners.keys():
        raise ValueError("Nominal valuation loses a configured peg or assembly_board-v1")
    return owners


def _event_participants(
    models: dict, valuation: dict, event: dict, parameters: dict
) -> list[tuple[str, dict]]:
    participants = []
    for rid in event["participants"]:
        local = [item for item in models[rid]["events"] if item["event_id"] == event["event_id"]]
        if len(local) != 1 or any(
            local[0][key] != event[key]
            for key in ("event_name", "parameter_bindings", "participants", "product_effects")
        ):
            raise ValueError("Shared handoff does not have matching participant events")
        for template, guard in local[0]["guards"].items():
            field = _bound_field(models[rid], template, parameters)
            value = valuation[rid][field]
            if len(guard) != 1:
                raise ValueError(f"Unknown nominal guard: {rid}.{field}")
            operator, expected = next(iter(guard.items()))
            if operator in ("equals_from_param", "not_equals_from_param"):
                if expected not in parameters:
                    raise ValueError(f"Unbound nominal guard: {rid}.{field}")
                expected = parameters[expected]
            elif operator not in ("equals", "not_equals"):
                raise ValueError(f"Unknown nominal guard: {rid}.{field}")
            allowed = type(value) is type(expected) and value == expected
            if operator in ("not_equals", "not_equals_from_param"):
                allowed = not allowed
            if not allowed:
                raise ValueError(f"Nominal guard blocked: {rid}.{field}")
        participants.append((rid, local[0]))
    return participants


def project_nominal_event(
    models: dict[str, dict],
    valuation: dict[str, dict],
    resource_id: str,
    event_name: str,
    parameters: dict[str, Any],
) -> dict[str, dict]:
    """Project a completed task for offline DES verification, atomically.

    Args:
        models: Code-defined nominal descriptors; no runtime authority is used.
        valuation: Complete pre-event resource valuations.
        resource_id: Resource performing the task, not an observing participant.
        event_name: An exact declared nominal event name.
        parameters: Exact event bindings and acknowledged completion assumptions.

    Returns:
        A new valuation with every shared handoff applied together.

    Raises:
        ValueError: An event, binding, guard, occupancy, or update is invalid.
    """
    _check_valuation(models, valuation)
    if resource_id not in models:
        raise ValueError(f"Unknown nominal resource: {resource_id}")
    if "resource_id" in parameters and parameters["resource_id"] != resource_id:
        raise ValueError("Nominal event resource_id does not match its binding")
    supplied = {**parameters, "resource_id": resource_id}
    matches = [
        event
        for event in models[resource_id]["events"]
        if event["event_name"] == event_name and _matches_bindings(models, event, supplied)
    ]
    if len(matches) != 1:
        raise ValueError(f"Unknown or incomplete nominal event binding: {resource_id}.{event_name}")
    event = matches[0]
    participants = _event_participants(models, valuation, event, supplied)
    if event_name == "place_release" and parameters.get("destination_location") == "Conveyor":
        supplied.update(
            conveyor_load_parameters(
                models["Conveyor"],
                valuation["Conveyor"],
                parameters["part_name"],
                parameters["loading_position"],
            )
        )
    if event_name == "advance_conveyor":
        supplied.update(
            conveyor_advance_parameters(
                models["Conveyor"],
                valuation["Conveyor"],
                parameters["next_locations"],
                parameters["delivered_part"],
            )
        )
    projected = deepcopy(valuation)
    for rid, local in participants:
        for template, update in local["updates"].items():
            field = _bound_field(models[rid], template, supplied)
            if set(update) == {"set"}:
                projected[rid][field] = deepcopy(update["set"])
            elif set(update) == {"set_from_param"} and update["set_from_param"] in supplied:
                projected[rid][field] = deepcopy(supplied[update["set_from_param"]])
            else:
                raise ValueError(f"Unbound nominal update: {rid}.{field}")
    _check_valuation(models, projected)
    before_parts, after_parts = (
        set(_part_owners(models, valuation)),
        set(_part_owners(models, projected)),
    )
    if event_name == "print_part":
        if after_parts != before_parts | {parameters["part_name"]}:
            raise ValueError("print_part must introduce exactly its declared output")
    elif before_parts != after_parts:
        raise ValueError("Nominal event does not conserve part custody")
    return projected
