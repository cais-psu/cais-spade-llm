"""Prepare configured buffer starting conditions before nominal dispatch."""

from __future__ import annotations

import asyncio
import hashlib
import math
import os
import time
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

BUFFER = "Buffer For Machined parts"


def apply_initial_conditions(scene: dict, failure: dict, order: dict, permitted: list[str]) -> dict:
    """Copy and validate starting custody and completed process facts.

    Args:
        scene: Saved scene, which remains unchanged.
        failure: Selected slippage configuration and optional initial conditions.
        order: Selected parts and their required processes.
        permitted: Resources available to this run.

    Returns:
        A separate scene containing the validated initial custody.
    """
    scene = deepcopy(scene)
    rows = failure.get("initial_conditions")
    if rows is None:
        return scene
    if (failure.get("scenario") != "Part slippage"
            or failure.get("checkpoint") != "during_place_lowering"
            or not isinstance(rows, list) or not rows):
        raise ValueError("initial_conditions require placement slippage buffer entries")
    seen_parts, seen_zones = set(), set()
    requirements = order.get("processPlan", order.get("requirements", {}))
    selected = list(requirements) if order["parts"] == "all" else order["parts"]
    for row in rows:
        if not isinstance(row, dict) or row.get("resource_id") != BUFFER or BUFFER not in permitted:
            raise ValueError("Slippage initialization requires the permitted buffer")
        zone_number, part = row.get("zone"), row.get("part_name")
        zones = scene[BUFFER]["zones"]
        zone = next((zone for zone in zones if zone["zone"] == zone_number), None)
        if type(zone_number) is not int or zone is None or zone_number != 4:
            raise ValueError("Slippage initialization requires pickup-ready buffer zone 4")
        if (part not in selected or part in seen_parts or zone_number in seen_zones
                or failure.get("additional_condition") != {"resource_id": scene[BUFFER]["handling_robot"],
                                                          "part_name": part}):
            raise ValueError("Slippage buffer part must match the selected retaining robot")
        if part not in scene["Storage"]["slots"] or zone.get("initial_part") is not None:
            raise ValueError("Slippage buffer initialization conflicts with existing custody")
        if any(other.get("initial_part") == part for other in zones):
            raise ValueError("Slippage initialization duplicates buffer custody")
        facts = row.get("processCompleted")
        if not isinstance(facts, list) or not facts:
            raise ValueError("Slippage initialization requires completed machining facts")
        expected = [fact for step in requirements[part] for fact in step["processesToComplete"]
                    if fact["process"] == "trim"]
        if facts != expected:
            raise ValueError("Slippage initial machining facts do not match the selected order")
        _initial_orientation(row, scene[BUFFER])
        scene["Storage"]["slots"].pop(part)
        zone["initial_part"] = part
        seen_parts.add(part)
        seen_zones.add(zone_number)
    scene["failure_initial_conditions"] = deepcopy(rows)
    return scene


def _pose(observed: dict):
    return SimpleNamespace(position=SimpleNamespace(**{key: observed[key] for key in ("x", "y", "z")}),
                           orientation=SimpleNamespace(**{key: observed["q" + key]
                                                          for key in ("x", "y", "z", "w")}))


def _bounds(boxes: list[dict]) -> dict:
    from cais_spade_llm.recovery_framework.geometry import rotate

    lows, highs = [], []
    for box in boxes:
        basis = [rotate(box["pose"][3:], [float(i == axis) for i in range(3)]) for axis in range(3)]
        extent = [sum(abs(basis[j][i]) * box["size"][j] / 2 for j in range(3)) for i in range(3)]
        lows.append([box["pose"][i] - extent[i] for i in range(3)])
        highs.append([box["pose"][i] + extent[i] for i in range(3)])
    return {"min": [min(row[i] for row in lows) for i in range(3)],
            "max": [max(row[i] for row in highs) for i in range(3)]}


def _quaternion(rpy: list[float]) -> dict:
    roll, pitch, yaw = [float(value) / 2 for value in rpy]
    cr, sr, cp, sp, cy, sy = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch), math.cos(yaw), math.sin(yaw)
    return {"qx": sr * cp * cy - cr * sp * sy, "qy": cr * sp * cy + sr * cp * sy,
            "qz": cr * cp * sy - sr * sp * cy, "qw": cr * cp * cy + sr * sp * sy}


def _initial_orientation(entry: dict, buffer: dict) -> dict:
    if "orientation_quat" not in entry:
        return _quaternion(buffer["part_orientation_rpy"])
    orientation = entry["orientation_quat"]
    keys = ("qx", "qy", "qz", "qw")
    if (not isinstance(orientation, dict) or set(orientation) != set(keys)
            or any(type(orientation[key]) not in (int, float)
                   or not math.isfinite(orientation[key]) for key in keys)
            or abs(sum(orientation[key] ** 2 for key in keys) - 1.) > 1e-6):
        raise ValueError("Slippage initial orientation_quat must be a finite unit quaternion")
    return dict(orientation)


def _prepare_part(runtime, entry: dict) -> dict:
    from cais_spade_llm.recovery_framework.failure_effects import _observe_part, _robot
    from cais_spade_llm.recovery_framework.part_collision import (
        collision_geometry_evidence,
        observed_part_boxes,
    )

    if runtime.stopped:
        raise asyncio.CancelledError()
    context = runtime.context
    buffer = context.inputs["scene"][BUFFER]
    zone = next(zone for zone in buffer["zones"] if zone["zone"] == entry["zone"])
    part = entry["part_name"]
    model = context.geometry[part]["model_name"]
    robot = _robot(runtime, buffer["handling_robot"])
    controller = robot._controller
    if controller._attached_model is not None:
        raise ValueError("Buffer initialization requires an empty retaining gripper")
    before = _observe_part(controller, model)
    target = dict(zip(("x", "y", "z"), zone["pose"][:3], strict=True))
    target.update(_initial_orientation(entry, buffer))
    local = _bounds(observed_part_boxes(model, _pose({**target, "z": 0.0})))
    target["z"] = float(zone["pose"][2]) - local["min"][2] + .0005
    if runtime.stopped:
        raise asyncio.CancelledError()
    placed = controller.set_entity_pose(model, **target)
    if not placed.get("success"):
        raise ValueError("Gazebo rejected buffer initial part placement")
    observations = []
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        if runtime.stopped:
            raise asyncio.CancelledError()
        observed = _observe_part(controller, model)
        if any(not math.isfinite(value) for value in observed.values()):
            raise ValueError("Buffer initial pose observation is non-finite")
        observations.append({"observed_at_unix": time.time(), "pose": observed})
        if len(observations) >= 8:
            recent = observations[-8:]
            displacement = max(math.dist([a["pose"][axis] for axis in ("x", "y", "z")],
                                         [recent[0]["pose"][axis] for axis in ("x", "y", "z")]) for a in recent)
            angle = max(2 * math.acos(min(1., abs(sum(a["pose"][key] * recent[0]["pose"][key]
                                                     for key in ("qx", "qy", "qz", "qw"))))) for a in recent)
            if displacement <= .001 and angle <= .02:
                break
        time.sleep(.1)
    else:
        raise ValueError("Buffer initial part did not reach stable support")
    orientation_error = 2 * math.acos(min(1., abs(sum(
        observed[key] * target[key] for key in ("qx", "qy", "qz", "qw")))))
    if "orientation_quat" in entry and orientation_error > .05:
        raise ValueError("Buffer initial part did not retain its configured orientation")
    boxes = observed_part_boxes(model, _pose(observed))
    bounds = _bounds(boxes)
    floor_error = bounds["min"][2] - zone["pose"][2]
    if (abs(floor_error) > .003
            or max(abs(bounds[edge][0] - zone["pose"][0]) for edge in ("min", "max")) > buffer["zone_pitch"] / 2
            or max(abs(bounds[edge][1] - zone["pose"][1]) for edge in ("min", "max")) > buffer["guide_clear_width_m"] / 2):
        raise ValueError("Buffer initial part is outside its supported pickup zone")
    if not controller._sync_part_collision(model):
        raise ValueError("Buffer initial part collision geometry was not acknowledged")
    return {"resource_id": BUFFER, "zone": entry["zone"], "part_name": part, "model_name": model,
            "pose_before": before, "requested_pose": target, "observed_pose": observed,
            "support_observations": observations[-8:], "support_floor_error_m": floor_error,
             "orientation_error_rad": orientation_error,
            "collision_geometry": collision_geometry_evidence(boxes),
            "collision_scene": deepcopy(controller._last_command_evidence),
            "pickup_ready": True, "attachment": None}


def _sync_initial_world(runtime) -> dict:
    from moveit_msgs.msg import PlanningScene, PlanningSceneComponents
    from moveit_msgs.srv import ApplyPlanningScene, GetPlanningScene

    from cais_spade_llm.recovery_framework import ROOT
    from cais_spade_llm.recovery_framework.failure_effects import _observe_part, _robot
    from cais_spade_llm.recovery_framework.geometry import collision_boxes
    from cais_spade_llm.recovery_framework.part_collision import collision_object

    if runtime.stopped:
        raise asyncio.CancelledError()
    context = runtime.context
    controller = _robot(runtime, context.inputs["scene"][BUFFER]["handling_robot"])._controller
    if controller._attached_model is not None:
        raise ValueError("Initial collision registration requires an empty gripper")
    geometry = context.inputs["geometry"]
    models = set(geometry["parts"]["model_map"].values()) | {
        geometry["assembly_board"]["model_name"], "Gear_Plate"}
    table_roots = [Path(path) for path in os.environ.get("GAZEBO_MODEL_PATH", "").split(os.pathsep) if path]
    table_roots.append(Path.home() / ".gazebo/models")
    table_model = next((root / "table/model.sdf" for root in table_roots
                        if (root / "table/model.sdf").is_file()), None)
    if table_model is None:
        raise ValueError("The Gazebo table collision model could not be resolved")
    models.update({"table_xarm6", "table_ur5e"})
    poses = {}
    for model in sorted(models):
        if runtime.stopped:
            raise asyncio.CancelledError()
        observed = _observe_part(controller, model)
        poses[model] = [observed[key] for key in ("x", "y", "z", "qx", "qy", "qz", "qw")]
    rows = collision_boxes(ROOT / "ros2/cais_lab_robotics/worlds/table_recovery_framework.world",
                           ROOT / "ros2/cais_lab_robotics/models", poses, exact_models=models,
                           include_model_files={"model://table": table_model})
    update = PlanningScene(is_diff=True)
    update.robot_state.is_diff = True
    update.world.collision_objects = [collision_object(row) for row in rows]
    clients = [
        controller._node.create_client(kind, endpoint, callback_group=controller._cb_group)
        for kind, endpoint in ((ApplyPlanningScene, "/apply_planning_scene"),
                               (GetPlanningScene, "/get_planning_scene"))
    ]
    try:
        if not all(client.wait_for_service(timeout_sec=2.) for client in clients):
            raise ValueError("Initial world collision services are unavailable")
        if runtime.stopped:
            raise asyncio.CancelledError()
        answer = controller._wait_future(clients[0].call_async(
            ApplyPlanningScene.Request(scene=update)), timeout_sec=10.,
            label="slippage initial world collision acknowledgement")
        if answer is None or not answer.success:
            raise ValueError("Initial world collision geometry was not acknowledged")
        query = GetPlanningScene.Request(components=PlanningSceneComponents(
            components=PlanningSceneComponents.WORLD_OBJECT_NAMES))
        scene = controller._wait_future(clients[1].call_async(query), timeout_sec=5.,
                                        label="slippage initial world collision readback")
        expected = {row["id"] for row in rows}
        installed = set() if scene is None else {obj.id for obj in scene.scene.world.collision_objects}
        if not expected or not expected <= installed:
            raise ValueError("Initial world collision geometry readback is incomplete")
        return {"acknowledged": True, "collision_object_ids": sorted(expected),
                "observed_model_poses": poses,
                "external_model_geometry": {"model://table": {
                    "path": str(table_model), "sha256": hashlib.sha256(table_model.read_bytes()).hexdigest()}},
                "observed_at_unix": time.time()}
    finally:
        for client in clients:
            controller._node.destroy_client(client)


async def prepare_slippage_initial_conditions(runtime) -> None:
    """Physically prepare this run's buffer state before agent negotiation.

    Args:
        runtime: Owning run with validated custody and simulation controllers.
    """
    configuration = getattr(getattr(runtime, "conveyor_fault", None), "configuration", {}) or {}
    entries = configuration.get("initial_conditions")
    if not entries:
        return
    context = runtime.context
    previous = getattr(runtime, "slippage_initialization_evidence", None)
    if previous and previous.get("run_id") == context.run_id and previous.get("status") == "completed":
        return
    evidence = {"run_id": context.run_id, "status": "preparing", "parts": []}
    runtime.slippage_initialization_evidence = evidence
    for entry in entries:
        if (context.resources[BUFFER].valuation[f"zone_{entry['zone']}_part"] != entry["part_name"]
                or context.part_tracker[entry["part_name"]]["location"] != BUFFER):
            raise ValueError("Physical buffer initialization does not match run custody")
        evidence["parts"].append(await asyncio.to_thread(_prepare_part, runtime, entry))
    evidence["world_collision_scene"] = await asyncio.to_thread(_sync_initial_world, runtime)
    if runtime.stopped:
        raise asyncio.CancelledError()
    evidence.update(status="completed", completed_at_unix=time.time())
    context.negotiations.append({"kind": "slippage_initialization", **deepcopy(evidence)})
    runtime.queue_save()
