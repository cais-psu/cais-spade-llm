"""Observe and interrupt a resource-owned placement trajectory in simulation."""

from __future__ import annotations

import math
import time
from copy import deepcopy


def _pose_values(pose) -> dict:
    return {
        **{axis: float(getattr(pose.position, axis)) for axis in ("x", "y", "z")},
        **{"q" + axis: float(getattr(pose.orientation, axis)) for axis in ("x", "y", "z", "w")},
    }


def placement_motion(controller, target) -> dict | None:
    """Bind one observed downward primitive to the configured fault request."""
    request = getattr(controller, "_simulation_fault_request", None)
    if not request or getattr(controller, "_robot_task_step", None) != ("place_approach", "descend"):
        return None
    if (controller.execution_mode != "simulation"
            or request.get("checkpoint") != "during_place_lowering"
            or request.get("resource_id") != controller.robot_name
            or request.get("task_id") != getattr(controller, "_simulation_task_id", None)
            or request.get("model_name") != controller._attached_model):
        raise ValueError("Placement fault does not match the owning controller task and attachment")
    observed = controller._fresh_simulation_tf_tool_pose()
    if observed is None:
        raise ValueError("Placement lowering start has no fresh observed tool pose")
    start, end = _pose_values(observed), _pose_values(target)
    if any(not math.isfinite(value) for value in [*start.values(), *end.values()]):
        raise ValueError("Placement lowering poses must be finite")
    # A Cartesian helper may first translate at travel height; only its actual
    # downward segment belongs to the placement checkpoint.
    if end["z"] >= start["z"] - .001:
        return None
    if math.dist([start["x"], start["y"]], [end["x"], end["y"]]) > .02:
        raise ValueError("Placement checkpoint requires the vertical lowering segment")
    progress = request.get("placement_progress")
    if type(progress) not in {int, float} or not math.isfinite(progress) or not 0 < progress < 1:
        raise ValueError("Placement progress must lie strictly inside the downward motion")
    return {**deepcopy(request), "started_pose": start, "target_pose": end,
            "function_name": "place_approach", "step_id": "descend",
            "source": "gazebo_placement_motion"}


def _time_observed_lowering(controller, trajectory, motion: dict) -> None:
    """Allow fresh simulation observations to resolve the halfway checkpoint."""
    from moveit_msgs.msg import RobotTrajectory

    last = trajectory.points[-1].time_from_start
    duration = last.sec + last.nanosec / 1e9
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("Observed lowering requires a positive trajectory duration")
    scale = max(1., 5. / duration)
    if scale > 1.:
        controller._scale_trajectory_timing(RobotTrajectory(joint_trajectory=trajectory), scale)
    motion["trajectory_time_scale"] = scale
    motion["trajectory_duration_sec"] = duration * scale


def wait_for_placement_motion(controller, goal, future, duration: float, motion: dict):
    """Observe real downward travel, then require terminal cancellation and rest."""
    motion = {**motion, "started_at_unix": time.time()}
    pending = controller._motion_pending(duration)
    while (controller._rclpy.ok() and not future.done() and pending()
           and not getattr(controller, "_shutdown_requested", False)):
        armed = getattr(controller, "_simulation_fault_armed", None)
        if callable(armed) and armed():
            observed = controller._fresh_simulation_tf_tool_pose()
            if observed is not None:
                values = _pose_values(observed)
                distance = motion["started_pose"]["z"] - motion["target_pose"]["z"]
                progress = (motion["started_pose"]["z"] - values["z"]) / distance
                if (all(math.isfinite(value) for value in values.values())
                        and motion["placement_progress"] <= progress < 1.0
                        and goal.status in (1, 2)):
                    evidence = {
                        **deepcopy(motion), "observed_pose": values, "progress": progress,
                        "goal_active": True, "goal_cancelled": False, "motion_stopped": False,
                        "controller_goal_id": [int(value) for value in goal.goal_id.uuid],
                        "observed_at_unix": time.time(),
                        "motion_path_validation": deepcopy(getattr(controller, "_last_path_validation", {})),
                    }
                    controller._simulation_fault_evidence = evidence
                    cancel = controller._wait_future(
                        goal.cancel_goal_async(), timeout_sec=5.0, label="placement fault cancellation")
                    acknowledged = bool(cancel and any(
                        [int(value) for value in row.goal_id.uuid] == evidence["controller_goal_id"]
                        for row in cancel.goals_canceling))
                    terminal = controller._wait_future(
                        future, timeout_sec=5.0, label="placement fault terminal result")
                    evidence["goal_cancelled"] = acknowledged and bool(terminal and terminal.status == 5)
                    evidence["terminal_status"] = terminal.status if terminal else None
                    if not evidence["goal_cancelled"]:
                        raise RuntimeError("Placement trajectory cancellation was not confirmed")
                    controller._simulation_goal = None
                    deadline = time.monotonic() + 5.0
                    while time.monotonic() < deadline:
                        positions, missing = controller._get_arm_joint_positions(timeout_sec=.1)
                        if positions is not None and not missing and controller._fresh_stable_joint_target(
                            dict(zip(controller.arm_joint_names, positions, strict=True)),
                            tolerance=.005, stable_for_sec=.15,
                        ):
                            evidence["motion_stopped"] = True
                            evidence["stopped_joint_observation"] = deepcopy(
                                controller._last_joint_target_observation)
                            stopped = controller._fresh_simulation_tf_tool_pose()
                            if stopped is None:
                                raise RuntimeError("Stopped placement pose could not be observed")
                            evidence["stopped_pose"] = _pose_values(stopped)
                            evidence["stopped_at_unix"] = time.time()
                            controller._last_failure_message = "Part slippage during placement lowering"
                            return None
                        time.sleep(.02)
                    raise RuntimeError("Interrupted placement motion did not reach observed rest")
        time.sleep(.02)
    return controller._wait_future(future, timeout_sec=.1, label="placement motion result")
