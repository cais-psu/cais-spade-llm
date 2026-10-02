"""Resource-owned simulation effects and observations for failure injection."""

from __future__ import annotations

import asyncio
import math
from copy import deepcopy


def _robot(runtime, resource_id: str):
    agent = next((agent for agent in runtime.resource_agents if agent.agent_name == resource_id), None)
    if agent is None or agent.execution_mode != "simulation" or agent._controller is None:
        raise ValueError("Failure effects require the owning simulation controller")
    return agent


async def observe_robot(runtime, resource_id: str) -> dict:
    """Read a failed robot's pose without commanding it home."""
    result = await asyncio.to_thread(_robot(runtime, resource_id)._controller.get_current_pose)
    pose = result.get("pose") or {}
    if not result.get("success") or any(
        not isinstance(pose.get(axis), (int, float)) or not math.isfinite(pose[axis])
        for axis in ("x", "y", "z")
    ):
        raise ValueError("Failed robot pose could not be observed")
    return deepcopy(pose)


def _observe_part(controller, model: str) -> dict:
    query = controller._GetEntityState.Request(name=model, reference_frame="world")
    response = controller._wait_future(
        controller._get_state_client.call_async(query), timeout_sec=5.0,
        label="failure_part_observation",
    )
    if response is None or not response.success:
        raise ValueError("Dropped part pose could not be observed")
    pose = response.state.pose
    return {**{axis: float(getattr(pose.position, axis)) for axis in ("x", "y", "z")},
            **{"q" + axis: float(getattr(pose.orientation, axis)) for axis in ("x", "y", "z", "w")}}


async def slip_part(runtime, configuration: dict, evidence: dict) -> None:
    """Detach only the selected held part and retain partial-effect evidence.

    Args:
        runtime: Stopped simulation runtime owning the controllers and custody.
        configuration: Validated two-robot Part slippage configuration.
        evidence: Failure record updated after each confirmed physical effect.
    """
    rid, part = configuration["resource_id"], configuration["part_name"]
    agent = _robot(runtime, rid)
    controller = agent._controller
    context = runtime.context
    model = context.geometry[part]["model_name"]
    evidence["requested_drop_pose"] = deepcopy(configuration["drop_pose"])
    evidence["requested_orientation_quat"] = deepcopy(configuration["orientation_quat"])
    evidence["robot_pose"] = await observe_robot(runtime, rid)
    evidence["other_robot_pose"] = await observe_robot(runtime, configuration["additional_condition"]["resource_id"])
    evidence["gripper_opened"] = await asyncio.to_thread(controller.open_gripper)
    if evidence["gripper_opened"] is not True:
        raise ValueError("Slipping robot gripper did not open")
    agent._gripper_state = "open"
    # An open gripper alone cannot prove that the simulated attachment was removed.
    result = await asyncio.to_thread(controller.detach_part, model, assume_released_if_open=False)
    evidence["detach"] = deepcopy(result)
    if not result.get("success") or str(result.get("release_mode", "")).startswith("assumed"):
        evidence["custody_uncertain"] = True
        raise ValueError("Slipped part detachment was not confirmed")
    with context.admission_lock:
        actor = context.resources[rid]
        actor.valuation.update(held_part=None, resource_state="failed")
        actor.revision += 1
        agent._held_part = None
        agent._current_state = "failed"
        agent._task_ctx = {}
        context.part_tracker[part].update(state="misplaced", location=None)
        context.revision += 1
    placed = await asyncio.to_thread(
        controller.set_entity_pose, model,
        **configuration["drop_pose"], **configuration["orientation_quat"],
    )
    evidence["set_entity_pose"] = deepcopy(placed)
    if not placed.get("success"):
        raise ValueError("Gazebo rejected the configured slippage target")
    previous = None
    for _ in range(20):
        observed = await asyncio.to_thread(_observe_part, controller, model)
        if any(not math.isfinite(observed[axis]) for axis in ("x", "y", "z")):
            raise ValueError("Dropped part observation contains non-finite coordinates")
        evidence["observed_drop_pose"] = observed
        if previous is not None and math.dist(
            [observed[axis] for axis in ("x", "y", "z")],
            [previous[axis] for axis in ("x", "y", "z")],
        ) <= 0.002:
            break
        previous = observed
        await asyncio.sleep(0.1)
    else:
        raise ValueError("Dropped part did not produce a stable observed pose")
    synced = await asyncio.to_thread(controller._sync_part_collision, model)
    evidence["collision_scene"] = deepcopy(getattr(controller, "_last_command_evidence", {}))
    if not synced:
        raise ValueError("Dropped part collision scene could not be acknowledged")
    with context.admission_lock:
        context.part_tracker[part]["observed_pose"] = deepcopy(observed)
        context.revision += 1
