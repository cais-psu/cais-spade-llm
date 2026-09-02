"""Run collision-aware MoveIt plans without executing robot motion."""

from __future__ import annotations

import asyncio
import math
import re
import time
from collections.abc import Mapping, Sequence
from typing import Any


class MoveItPlanOnlyRuntime:
    """Validate two endpoint positions through the shared MoveGroup action."""

    def __init__(
        self,
        *,
        action_name: str = "/move_action",
        server_timeout_sec: float = 10.0,
        result_timeout_sec: float = 20.0,
        position_tolerance_m: float = 0.005,
    ) -> None:
        """Configure bounded no-motion MoveIt validation."""
        self._action_name = action_name
        self._server_timeout_sec = server_timeout_sec
        self._result_timeout_sec = result_timeout_sec
        self._position_tolerance_m = position_tolerance_m

    async def validate(self, request: Mapping[str, object]) -> Mapping[str, object]:
        """Plan to both states without blocking the caller's agent event loop."""
        return await asyncio.to_thread(self._validate_sync, dict(request))

    def _validate_sync(self, request: Mapping[str, object]) -> Mapping[str, object]:
        """Plan to current then desired state while keeping execution disabled."""
        validated = _validated_request(request)
        try:
            import rclpy
            from moveit_msgs.action import MoveGroup
            from rclpy.action import ActionClient
            from rclpy.node import Node
        except ImportError as exc:
            return _unavailable_response(f"ROS2 MoveIt Python interfaces are unavailable: {exc}")

        initialized_here = False
        if not rclpy.ok():
            rclpy.init()
            initialized_here = True
        node_name = re.sub(
            r"[^A-Za-z0-9_]",
            "_",
            f"spec2primitives_{validated['resource_symbol']}_plan_only",
        )
        node = Node(node_name)
        client = ActionClient(node, MoveGroup, self._action_name)
        try:
            if not client.wait_for_server(timeout_sec=self._server_timeout_sec):
                return _unavailable_response(
                    f"{self._action_name} is unavailable for plan-only validation."
                )
            current_result, terminal_state = self._plan_endpoint(
                rclpy,
                node,
                client,
                validated,
                state_name="current_state",
                start_state=None,
            )
            if current_result["status"] != "accepted":
                desired_result = {
                    "status": "needs_context",
                    "message": (
                        "Desired-state path was not planned because the current-state "
                        "endpoint was not accepted."
                    ),
                    "error_code": None,
                }
                return _combined_response(current_result, desired_result)
            desired_result, _ = self._plan_endpoint(
                rclpy,
                node,
                client,
                validated,
                state_name="desired_state",
                start_state=terminal_state,
            )
            return _combined_response(current_result, desired_result)
        finally:
            node.destroy_node()
            if initialized_here and rclpy.ok():
                rclpy.shutdown()

    def _plan_endpoint(
        self,
        rclpy: Any,
        node: Any,
        client: Any,
        request: Mapping[str, object],
        *,
        state_name: str,
        start_state: object | None,
    ) -> tuple[dict[str, object], object | None]:
        goal = _move_group_goal(
            request,
            state_name=state_name,
            start_state=start_state,
            position_tolerance_m=self._position_tolerance_m,
        )
        goal_future = client.send_goal_async(goal)
        if not _wait_future(
            rclpy,
            node,
            goal_future,
            timeout_sec=self._result_timeout_sec,
        ):
            return (
                {
                    "status": "needs_context",
                    "message": f"{state_name} MoveIt goal acceptance timed out.",
                    "error_code": None,
                },
                None,
            )
        goal_handle = goal_future.result()
        if goal_handle is None or goal_handle.accepted is not True:
            return (
                {
                    "status": "rejected",
                    "message": f"MoveIt rejected the {state_name} plan-only goal.",
                    "error_code": None,
                },
                None,
            )
        result_future = goal_handle.get_result_async()
        if not _wait_future(
            rclpy,
            node,
            result_future,
            timeout_sec=self._result_timeout_sec,
        ):
            goal_handle.cancel_goal_async()
            return (
                {
                    "status": "needs_context",
                    "message": f"{state_name} MoveIt planning timed out.",
                    "error_code": None,
                },
                None,
            )
        response = result_future.result()
        moveit_result = getattr(response, "result", None)
        error_code = getattr(getattr(moveit_result, "error_code", None), "val", None)
        if not isinstance(error_code, int):
            return (
                {
                    "status": "needs_context",
                    "message": f"{state_name} MoveIt returned no error code.",
                    "error_code": None,
                },
                None,
            )
        trajectory = getattr(moveit_result, "planned_trajectory", None)
        terminal_state = _trajectory_terminal_state(trajectory)
        if error_code != 1 or terminal_state is None:
            return (
                {
                    "status": "rejected",
                    "message": (
                        f"{state_name} endpoint IK/collision/path planning failed "
                        f"with MoveIt error code {error_code}."
                    ),
                    "error_code": error_code,
                },
                None,
            )
        return (
            {
                "status": "accepted",
                "message": (f"{state_name} endpoint has a collision-aware plan-only path."),
                "error_code": error_code,
            },
            terminal_state,
        )


def _validated_request(value: Mapping[str, object]) -> Mapping[str, object]:
    expected = {
        "process_symbol",
        "process_iri",
        "feature_iri",
        "resource_symbol",
        "resource_iri",
        "resource_jid",
        "execution_mode",
        "moveit_group",
        "end_effector_link",
        "target_frame",
        "validation_scope",
        "checked_constraints",
        "unvalidated_constraints",
        "current_state",
        "desired_state",
        "mode",
        "motion_executed",
        "request_fingerprint",
    }
    if set(value) != expected:
        raise ValueError("Plan-only allocation request fields are invalid.")
    for field in (
        "process_symbol",
        "process_iri",
        "feature_iri",
        "resource_symbol",
        "resource_iri",
        "resource_jid",
        "execution_mode",
        "moveit_group",
        "end_effector_link",
        "target_frame",
        "request_fingerprint",
    ):
        if not isinstance(value[field], str) or not str(value[field]).strip():
            raise ValueError(f"Plan-only allocation {field} is invalid.")
    if value["mode"] != "plan_only" or value["motion_executed"] is not False:
        raise ValueError("Plan-only allocation may not execute motion.")
    if (
        value["validation_scope"] != "endpoint_motion"
        or value["checked_constraints"]
        != ["positional_ik", "collision_aware_endpoints", "path_between_endpoints"]
        or value["unvalidated_constraints"]
        != [
            "grasping",
            "end_effector_orientation",
            "attached_object_geometry",
            f"{value['process_symbol']}_tolerance",
            "force_contact",
            "insertion_constraints",
        ]
    ):
        raise ValueError("Plan-only endpoint validation scope is invalid.")
    for state_name in ("current_state", "desired_state"):
        state = value[state_name]
        if not isinstance(state, Mapping) or set(state) != {
            "state_iri",
            "evidence_handle",
            "translation_m",
            "location_record_ref",
            "location_record_sha256",
        }:
            raise ValueError(f"Plan-only allocation {state_name} is invalid.")
        if not isinstance(state["state_iri"], str) or not state["state_iri"]:
            raise ValueError(f"Plan-only allocation {state_name} state_iri is invalid.")
        if not isinstance(state["evidence_handle"], str) or not state["evidence_handle"]:
            raise ValueError(f"Plan-only allocation {state_name} evidence_handle is invalid.")
        _finite_vector3(state["translation_m"], state_name)
    return value


def _move_group_goal(
    request: Mapping[str, object],
    *,
    state_name: str,
    start_state: object | None,
    position_tolerance_m: float,
) -> Any:
    from geometry_msgs.msg import Pose
    from moveit_msgs.action import MoveGroup
    from moveit_msgs.msg import BoundingVolume, Constraints, PositionConstraint
    from shape_msgs.msg import SolidPrimitive

    state = request[state_name]
    assert isinstance(state, Mapping)
    x, y, z = _finite_vector3(state["translation_m"], state_name)
    goal = MoveGroup.Goal()
    goal.request.group_name = str(request["moveit_group"])
    goal.request.num_planning_attempts = 5
    goal.request.allowed_planning_time = 10.0
    goal.request.max_velocity_scaling_factor = 0.10
    goal.request.max_acceleration_scaling_factor = 0.10
    if start_state is None:
        goal.request.start_state.is_diff = True
    else:
        goal.request.start_state = start_state
    position = PositionConstraint()
    position.header.frame_id = str(request["target_frame"])
    position.link_name = str(request["end_effector_link"])
    primitive = SolidPrimitive()
    primitive.type = SolidPrimitive.SPHERE
    primitive.dimensions = [position_tolerance_m]
    primitive_pose = Pose()
    primitive_pose.position.x = x
    primitive_pose.position.y = y
    primitive_pose.position.z = z
    primitive_pose.orientation.w = 1.0
    region = BoundingVolume()
    region.primitives = [primitive]
    region.primitive_poses = [primitive_pose]
    position.constraint_region = region
    position.weight = 1.0
    constraints = Constraints()
    constraints.name = f"{state_name}_endpoint"
    constraints.position_constraints = [position]
    goal.request.goal_constraints = [constraints]
    goal.planning_options.plan_only = True
    goal.planning_options.look_around = False
    goal.planning_options.replan = False
    return goal


def _trajectory_terminal_state(trajectory: object) -> object | None:
    joint_trajectory = getattr(trajectory, "joint_trajectory", None)
    joint_names = getattr(joint_trajectory, "joint_names", None)
    points = getattr(joint_trajectory, "points", None)
    if (
        not isinstance(joint_names, Sequence)
        or isinstance(joint_names, (str, bytes))
        or not joint_names
        or not isinstance(points, Sequence)
        or not points
    ):
        return None
    positions = getattr(points[-1], "positions", None)
    if not isinstance(positions, Sequence) or len(positions) != len(joint_names):
        return None
    from moveit_msgs.msg import RobotState

    state = RobotState()
    state.joint_state.name = [str(name) for name in joint_names]
    state.joint_state.position = [float(position) for position in positions]
    state.is_diff = True
    return state


def _wait_future(
    rclpy: Any,
    node: Any,
    future: Any,
    *,
    timeout_sec: float,
) -> bool:
    deadline = time.monotonic() + timeout_sec
    while rclpy.ok() and time.monotonic() < deadline and not future.done():
        rclpy.spin_once(node, timeout_sec=0.1)
    return future.done()


def _combined_response(
    current_state: Mapping[str, object],
    desired_state: Mapping[str, object],
) -> Mapping[str, object]:
    endpoint_statuses = {current_state["status"], desired_state["status"]}
    status = (
        "rejected"
        if "rejected" in endpoint_statuses
        else "needs_context"
        if "needs_context" in endpoint_statuses
        else "accepted"
    )
    feedback = None
    if status != "accepted":
        feedback = " ".join(
            str(item["message"])
            for item in (current_state, desired_state)
            if item["status"] != "accepted"
        )
    return {
        "status": status,
        "current_state": dict(current_state),
        "desired_state": dict(desired_state),
        "feedback": feedback,
    }


def _unavailable_response(message: str) -> Mapping[str, object]:
    endpoint = {
        "status": "needs_context",
        "message": message,
        "error_code": None,
    }
    return {
        "status": "needs_context",
        "current_state": dict(endpoint),
        "desired_state": dict(endpoint),
        "feedback": message,
    }


def _finite_vector3(value: object, label: str) -> tuple[float, float, float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 3:
        raise ValueError(f"{label} translation_m is invalid.")
    result = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{label} translation_m is invalid.")
    return result  # type: ignore[return-value]


__all__ = ["MoveItPlanOnlyRuntime"]
