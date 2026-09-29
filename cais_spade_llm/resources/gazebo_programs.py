"""Validated Gazebo program and primitive catalog stored in the scene manifest."""

from __future__ import annotations

import hashlib
import inspect
import json
import math
from copy import deepcopy
from dataclasses import asdict, replace
from typing import Any

from cais_spade_llm.recovery_framework.kmr_tasks import (
    KMR_LOCATION_TASK, KMR_LOCATION_VARIANTS, KMR_MOVE_VARIANTS, KMR_TASKS, PRIMITIVE_CONTRACTS,
)
from cais_spade_llm.resources.robot.robot_task_model import (
    RobotTaskEffect, RobotTaskGuard, RobotTaskProgram, RobotTaskStep,
)
from cais_spade_llm.resources.robot.robot_task_registry import robot_task_registry
from cais_spade_llm.resources.workflow_task_programs import workflow_task_program

_ROBOT_EXECUTORS = {
    "detect_parts", "compute_pick_targets", "compute_place_targets",
    "move_to_named_pose", "move_cartesian", "move_relative",
    "grasp_part", "release_part", "open_gripper", "close_gripper",
    "attach_part", "detach_part", "move_joints", "rotate_joint",
}
_WORKFLOW_EXECUTORS = {
    "M1": {"dwell"}, "M2": {"dwell"},
    "Conveyor": {"move_relative"},
    "Buffer For Machined parts": {"move_relative"},
}


def _gazebo_robot_program(task, *, assembly_release: bool) -> dict[str, Any]:
    """Keep physical-only operations outside the saved Gazebo composition."""
    program = json.loads(json.dumps(asdict(task.program)))
    steps = []
    for step in program["steps"]:
        if step["op"] in {"localize_assembly_board_v1", "snap_part_to_slot"}:
            continue
        if step["op"] == "move_insert":
            # Gazebo still needs the final insertion displacement before release.
            step.update(
                id="seat_part",
                op="move_cartesian",
                exposed=True,
                params={axis: {"$state": "_task_ctx", "path": ["insert_pose", axis]}
                        for axis in ("x", "y", "z", "qx", "qy", "qz", "qw")},
                public_params={"target_pose": "insert_pose"},
                note="Cartesian movement from the configured pre-insert pose to insert_pose.",
            )
        steps.append(step)
    program["steps"] = steps
    if task.name == "place_approach":
        for step in program["steps"]:
            if step["op"] == "compute_place_targets":
                step["params"]["assembly_board_v1_aruco"] = None
    if task.name == "place_insert" and assembly_release:
        for step in program["steps"]:
            if step["op"] == "release_part":
                step["params"]["assembly_slot"] = {
                    "slot_x": {"$state": "_task_ctx", "path": ["slot_x"]},
                    "slot_y": {"$state": "_task_ctx", "path": ["slot_y"]},
                    "part_height": {"$state": "_task_ctx", "path": ["part_height"]},
                    "board_top_z": {"$state": "_task_ctx", "path": ["board_top_z"]},
                    "part_origin_z": {"$state": "_task_ctx", "path": ["place_part_origin_z"]},
                    "destination_location": {"$arg": "destination_location"},
                }
    return program

_WORKFLOW_EVENTS = {
    "M1": "machine_part", "M2": "machine_part", "Conveyor": "advance_conveyor",
    "Buffer For Machined parts": "advance_part", "3D Printing Station": "print_part",
}


def _expected_functions(resource_id: str, scene: dict) -> dict[str, dict[str, Any]]:
    """Return immutable safety structure supplied by the bound executors."""
    if resource_id.startswith("ur5e-"):
        assembly_robots = {
            scene["Buffer For Machined parts"]["handling_robot"],
            scene["3D Printing Station"]["handling_robot"],
        }
        return {
            name: {"function_name": name, "status": "implemented",
                   "program": _gazebo_robot_program(
                       task, assembly_release=resource_id in assembly_robots,
                   )}
            for name, task in robot_task_registry().items()
        }
    if resource_id == "KMR":
        functions = {
            name: {"function_name": name, "status": "implemented", "program": json.loads(json.dumps(asdict(task.program)))}
            for name, task in KMR_TASKS.items()
        }
        functions["move_to_resource"]["variants"] = {
            name: json.loads(json.dumps(asdict(task.program))) for name, task in KMR_MOVE_VARIANTS.items()
        }
        functions["move_to_location"] = {
            "function_name": "move_to_location", "status": "implemented",
            "program": json.loads(json.dumps(asdict(KMR_LOCATION_TASK.program))),
            "variants": {name: json.loads(json.dumps(asdict(task.program)))
                         for name, task in KMR_LOCATION_VARIANTS.items()},
        }
        return functions
    if resource_id in _WORKFLOW_EVENTS:
        name = _WORKFLOW_EVENTS[resource_id]
        program = workflow_task_program(name)
        return {name: {"function_name": program.pop("function_name"),
                       "status": program.pop("status"), "program": program}}
    return {}


def _executor_names(resource_id: str) -> set[str]:
    if resource_id.startswith("ur5e-"):
        return _ROBOT_EXECUTORS
    if resource_id == "KMR":
        return set(PRIMITIVE_CONTRACTS)
    return _WORKFLOW_EXECUTORS.get(resource_id, set())


def _bindings(value):
    """Yield nested parameter bindings for step-output validation."""
    if isinstance(value, dict):
        if "$step" in value or "$state" in value or "$arg" in value:
            yield value
        for item in value.values():
            yield from _bindings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _bindings(item)


def _check_program(actual: dict, expected: dict, *, path: str, resource_id: str) -> None:
    if not isinstance(actual, dict) or set(actual) != set(expected):
        raise ValueError(f"{path} has an invalid program structure")
    originals = {step["id"]: step for step in expected["steps"]}
    steps = actual.get("steps")
    if (not isinstance(steps, list) or len(steps) > 128
            or any(not isinstance(step, dict) for step in steps)):
        raise ValueError(f"{path} has an invalid step list")
    identifiers = [step.get("id") for step in steps]
    if any(not isinstance(key, str) or not key for key in identifiers) or len(set(identifiers)) != len(steps):
        raise ValueError(f"{path} requires unique nonempty step IDs")
    required = [identifier for identifier in identifiers if identifier in originals]
    if required != list(originals) or (originals and identifiers[-1:] != required[-1:]):
        raise ValueError(f"{path} must preserve its guarded step sequence and final step")
    available_outputs = set()
    auxiliary = {"detect_parts", "compute_pick_targets", "compute_place_targets",
                 "move_to_named_pose", "move_cartesian", "move_relative"}
    for index, step in enumerate(steps):
        original = originals.get(step["id"])
        if original is not None:
            allowed_fields = set(original) | {"params", "note"}
            if not set(original) <= set(step) <= allowed_fields:
                raise ValueError(f"{path}.steps[{index}] has an invalid structure")
            if any(step[key] != original[key] for key in original if key not in {"params", "note"}):
                raise ValueError(f"{path}.steps[{index}] changes an executor or safety condition")
            if "params" in original and set(step.get("params", {})) != set(original["params"]):
                raise ValueError(f"{path}.steps[{index}] changes supported executor parameters")
        else:
            if (resource_id != "KMR" and not resource_id.startswith("ur5e-")) or step.get("op") not in auxiliary:
                raise ValueError(f"{path}.steps[{index}] cannot add this primitive to the function contract")
            defaults = json.loads(json.dumps(asdict(RobotTaskStep(id=step["id"], op=step["op"]))))
            editable = {"id", "op", "params", "store_as", "note", "public_params"}
            if set(step) - set(defaults) or any(
                step[key] != defaults[key] for key in step if key not in editable
            ):
                raise ValueError(f"{path}.steps[{index}] cannot add conditional or optional execution")
            if resource_id == "KMR":
                from cais_spade_llm.recovery_framework.kmr_primitives import KMRPrimitives

                method = getattr(KMRPrimitives, step["op"])
            else:
                from cais_spade_llm.resources.robot.gazebo_pick_place_controller import GazeboPickPlaceController

                method = getattr(GazeboPickPlaceController, step["op"])
            try:
                inspect.signature(method).bind(None, **step.get("params", {}))
            except TypeError as exc:
                raise ValueError(f"{path}.steps[{index}] has unsupported executor parameters: {exc}") from exc
        if not isinstance(step.get("params", {}), dict) or not isinstance(step.get("note", ""), str):
            raise ValueError(f"{path}.steps[{index}] needs parameter sources and a note")
        for binding in _bindings(step.get("params", {})):
            if "$step" in binding and binding["$step"] not in available_outputs:
                raise ValueError(f"{path}.steps[{index}] references an unavailable step output")
        output = step.get("store_as") or step["id"]
        if not isinstance(output, str) or output in available_outputs:
            raise ValueError(f"{path}.steps[{index}] would overwrite a step output")
        available_outputs.add(output)
    for field, value in expected.items():
        if field != "steps" and actual[field] != value:
            raise ValueError(f"{path}.{field} must preserve the formal function contract")


def validate_resource_programs(scene: dict) -> None:
    """Reject catalog claims without executors and changes to safety structure.

    Args:
        scene: Gazebo scene with its saved resource programs.
    """
    root = scene.get("resource_programs")
    if not isinstance(root, dict) or type(root.get("revision")) is not int or root["revision"] < 1:
        raise ValueError("Gazebo scene lacks a valid resource program revision")
    resources = root.get("resources")
    ids = {*(row["resource_id"] for row in scene["robots"]),
           *(row["resource_id"] for row in scene["machines"]),
           "KMR", "Conveyor", "Buffer For Machined parts", "3D Printing Station", "Storage", "Exit"}
    if not isinstance(resources, dict) or set(resources) != ids:
        raise ValueError("Gazebo resource program catalog does not match the scene")
    for resource_id, bundle in resources.items():
        expected = _expected_functions(resource_id, scene)
        if not isinstance(bundle, dict) or set(bundle) != {"functions", "primitives"}:
            raise ValueError(f"Invalid program bundle for {resource_id}")
        functions, primitives = bundle["functions"], bundle["primitives"]
        if not isinstance(functions, dict) or set(functions) != set(expected):
            raise ValueError(f"Invalid function list for {resource_id}")
        if not isinstance(primitives, dict) or set(primitives) != _executor_names(resource_id):
            raise ValueError(f"Incomplete primitive catalog for {resource_id}")
        for name, primitive in primitives.items():
            if not isinstance(primitive, dict) or primitive.get("status") not in {"executable", "planned"}:
                raise ValueError(f"Invalid primitive catalog row {resource_id}.{name}")
            if primitive["status"] == "executable" and name not in _executor_names(resource_id):
                raise ValueError(f"No Gazebo executor for {resource_id}.{name}")
            if type(primitive.get("recovery_selectable")) is not bool:
                raise ValueError(f"Invalid recovery availability for {resource_id}.{name}")
            if primitive["status"] == "planned" and primitive["recovery_selectable"]:
                raise ValueError(f"Planned primitive cannot be recovery selectable: {resource_id}.{name}")
            if name in {"move_joints", "rotate_joint"} and primitive["recovery_selectable"]:
                policy = (scene["KMR"]["task_execution"].get("cartesian_motion_only")
                          if resource_id == "KMR" else next(
                              (row.get("cartesian_motion", {}).get("only") for row in scene["robots"]
                               if row["resource_id"] == resource_id), False))
                if policy:
                    raise ValueError(f"Cartesian-only scene cannot select {resource_id}.{name} for recovery")
        for name, function in functions.items():
            baseline = expected[name]
            if (not isinstance(function, dict) or set(function) != set(baseline)
                    or function["function_name"] != baseline["function_name"]
                    or function["status"] != baseline["status"]):
                raise ValueError(f"Invalid function contract for {resource_id}.{name}")
            _check_program(function["program"], baseline["program"], path=f"{resource_id}.{name}", resource_id=resource_id)
            for variant, program in baseline.get("variants", {}).items():
                _check_program(function["variants"][variant], program,
                               path=f"{resource_id}.{name}.{variant}", resource_id=resource_id)
            for program in [function["program"], *function.get("variants", {}).values()]:
                for step in program["steps"]:
                    params = step.get("params", {})
                    if resource_id in _WORKFLOW_EVENTS and params:
                        if step["op"] == "dwell":
                            minimum = next(row["simulation_process"]["processing_time_sec"]
                                           for row in scene["machines"] if row["resource_id"] == resource_id)
                            valid = (set(params) == {"duration_sec"}
                                     and type(params["duration_sec"]) in {int, float}
                                     and math.isfinite(params["duration_sec"])
                                     and params["duration_sec"] >= minimum)
                        elif step["op"] == "move_relative":
                            maximum = scene[resource_id]["simulation_transport"]["speed_mps"]
                            valid = (set(params) == {"speed_mps"}
                                     and type(params["speed_mps"]) in {int, float}
                                     and math.isfinite(params["speed_mps"])
                                     and 0 < params["speed_mps"] <= maximum)
                        else:
                            valid = False
                        if not valid:
                            raise ValueError(f"Unsafe workflow primitive parameters: {resource_id}.{name}.{step['op']}")
                    status = primitives.get(step["op"], {}).get("status")
                    if function["status"] == "implemented" and status != "executable":
                        raise ValueError(f"No executable primitive for {resource_id}.{name}.{step['op']}")
                    if status not in {"executable", "planned"}:
                        raise ValueError(f"Undeclared primitive {resource_id}.{name}.{step['op']}")


def resource_program_revision(scene: dict) -> str:
    """Return a digest of the complete saved program and primitive catalog."""
    validate_resource_programs(scene)
    payload = json.dumps(scene["resource_programs"], sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def saved_function(scene: dict, resource_id: str, event_name: str, *, variant: str = "") -> dict:
    """Return the validated function that the configured Gazebo executor will run."""
    validate_resource_programs(scene)
    function = deepcopy(scene["resource_programs"]["resources"][resource_id]["functions"][event_name])
    if variant:
        function["program"] = function["variants"][variant]
    return function


def _guard(row: dict) -> RobotTaskGuard:
    return RobotTaskGuard(**row)


def _robot_program(row: dict) -> RobotTaskProgram:
    values = deepcopy(row)
    values["entry_guards"] = tuple(_guard(item) for item in values["entry_guards"])
    values["steps"] = tuple(RobotTaskStep(**{**item, "when": tuple(_guard(g) for g in item.get("when", []))})
                            for item in values["steps"])
    values["effects"] = tuple(RobotTaskEffect(**{**item, "when": tuple(_guard(g) for g in item["when"])})
                              for item in values["effects"])
    values["required_context_keys"] = tuple(values["required_context_keys"])
    values["notes"] = tuple(values["notes"])
    return RobotTaskProgram(**values)


def saved_robot_definition(scene: dict, resource_id: str, function_name: str, *, variant: str = ""):
    """Bind saved Gazebo steps to an existing robot or KMR function contract."""
    function = saved_function(scene, resource_id, function_name, variant=variant)
    if function["status"] != "implemented":
        raise ValueError(f"No Gazebo executor for {resource_id}.{function_name}")
    if resource_id == "KMR" and function_name == "move_to_location":
        baseline = KMR_LOCATION_VARIANTS[variant] if variant else KMR_LOCATION_TASK
    elif resource_id == "KMR":
        baseline = KMR_MOVE_VARIANTS[variant] if variant else KMR_TASKS[function_name]
    else:
        baseline = robot_task_registry()[function_name]
    return replace(baseline, program=_robot_program(function["program"]))


def saved_workflow_program(scene: dict, resource_id: str, event_name: str) -> dict:
    """Return a saved machine or transport program for its exact resource."""
    function = saved_function(scene, resource_id, event_name)
    return {**function["program"], "function_name": function["function_name"],
            "status": function["status"]}


def current_program_revision(scene_path) -> str:
    """Read the currently saved Gazebo program revision from a scene path."""
    from pathlib import Path

    return resource_program_revision(json.loads(Path(scene_path).read_text()))


def primitive_available(scene: dict, resource_id: str, primitive: str, *, recovery: bool) -> bool:
    """Check the saved executor and recovery availability of one exact primitive."""
    validate_resource_programs(scene)
    row = scene["resource_programs"]["resources"][resource_id]["primitives"].get(primitive, {})
    return row.get("status") == "executable" and (not recovery or row.get("recovery_selectable") is True)
