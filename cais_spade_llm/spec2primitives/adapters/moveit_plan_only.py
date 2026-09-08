from __future__ import annotations

"""Run collision-aware MoveIt position plans without executing robot motion."""


import asyncio
import time
from collections.abc import Mapping
from typing import Any


class MoveItPlanOnlyRuntime:
    """Validate bound current and desired positions through MoveIt."""

    def __init__(
        self,
        *,
        service_timeout_sec: float = 10.0,
        result_timeout_sec: float = 20.0,
    ) -> None:
        """Configure bounded no-motion position planning."""
        self._service_timeout_sec = service_timeout_sec
        self._result_timeout_sec = result_timeout_sec

    async def validate_state_locations(self, request: Mapping[str, object]) -> Mapping[str, object]:
        """Plan to bound positions with unconstrained tool orientation and no execution."""
        return await asyncio.to_thread(self._validate_state_locations_sync, dict(request))

    async def read_resource_base_pose(
        self, *, base_frame: str, target_frame: str
    ) -> Mapping[str, object]:
        """Read live TF for advisory allocation proximity without activating a robot."""
        return await asyncio.to_thread(self._read_resource_base_pose_sync, base_frame, target_frame)

    def _read_resource_base_pose_sync(
        self, base_frame: str, target_frame: str
    ) -> Mapping[str, object]:
        try:
            import rclpy
            from rclpy.node import Node
            from tf2_ros import Buffer, TransformException, TransformListener
        except ImportError as exc:
            raise RuntimeError("ROS2 TF interfaces are unavailable.") from exc
        initialized_here = not rclpy.ok()
        if initialized_here:
            rclpy.init()
        node = Node("spec2primitives_resource_base_pose")
        buffer = Buffer()
        listener = TransformListener(buffer, node)
        try:
            deadline = time.monotonic() + self._service_timeout_sec
            while time.monotonic() < deadline:
                rclpy.spin_once(node, timeout_sec=0.1)
                if not buffer.can_transform(target_frame, base_frame, rclpy.time.Time()):
                    continue
                transform = buffer.lookup_transform(target_frame, base_frame, rclpy.time.Time())
                point = transform.transform.translation
                return {
                    "base_frame": transform.child_frame_id,
                    "target_frame": transform.header.frame_id,
                    "translation_m": [point.x, point.y, point.z],
                    # Static TF stamps can be zero; this records when the live buffer was read.
                    "observed_at_ns": time.time_ns(),
                }
            raise RuntimeError("Robot base transform is unavailable.")
        except TransformException as exc:
            raise RuntimeError("Robot base transform is unavailable.") from exc
        finally:
            listener.unregister()
            node.destroy_node()
            if initialized_here and rclpy.ok():
                rclpy.shutdown()

    def _validate_state_locations_sync(self, request: Mapping[str, object]) -> Mapping[str, object]:
        from ..agents.pa.resource_grounding import validate_location_planning_request

        validate_location_planning_request(request)
        try:
            import rclpy
            from moveit_msgs.srv import GetMotionPlan
            from rclpy.node import Node
        except ImportError:
            return _locations_unavailable(
                request, "ROS2 MoveIt planning interfaces are unavailable."
            )

        initialized_here = not rclpy.ok()
        if initialized_here:
            rclpy.init()
        node = Node("spec2primitives_location_plan_only")
        client = node.create_client(GetMotionPlan, str(request["motion_plan_service"]))
        try:
            if not client.wait_for_service(timeout_sec=self._service_timeout_sec):
                return _locations_unavailable(
                    request, "MoveIt motion planning service is unavailable."
                )
            results: dict[str, list[dict[str, object]]] = {}
            planned: dict[tuple[object, ...], dict[str, object]] = {}
            for state_name, locations in request["state_locations"].items():
                results[state_name] = []
                for location in locations:
                    key = (location["location_record_sha256"], *location["translation_m"])
                    if key not in planned:
                        goal = _location_motion_plan_request(GetMotionPlan, request, location)
                        future = client.call_async(goal)
                        rclpy.spin_until_future_complete(
                            node, future, timeout_sec=self._result_timeout_sec
                        )
                        if not future.done():
                            future.cancel()
                            planned[key] = _location_unavailable("MoveIt planning timed out.")
                        elif future.result() is None:
                            planned[key] = _location_unavailable(
                                "MoveIt returned no planning result."
                            )
                        else:
                            planned[key] = _location_plan_result(
                                future.result().motion_plan_response, request
                            )
                    results[state_name].append(
                        {"evidence_handle": location["evidence_handle"], **planned[key]}
                    )
            statuses = [item["status"] for group in results.values() for item in group]
            status = (
                "needs_context"
                if "needs_context" in statuses
                else "rejected"
                if "rejected" in statuses
                else "accepted"
            )
            return {
                "status": status,
                "state_locations": results,
                "feedback": None
                if status == "accepted"
                else "MoveIt did not validate every bound location.",
            }
        except (AttributeError, RuntimeError, TypeError, ValueError):
            return _locations_unavailable(request, "MoveIt location planning is unavailable.")
        finally:
            node.destroy_node()
            if initialized_here and rclpy.ok():
                rclpy.shutdown()


def _location_motion_plan_request(
    service_type: Any, request: Mapping[str, object], location: Mapping[str, object]
) -> Any:
    """Build a position goal; MoveIt supplies the live scene and current robot state."""
    from geometry_msgs.msg import Pose
    from moveit_msgs.msg import Constraints, PositionConstraint
    from shape_msgs.msg import SolidPrimitive

    message = service_type.Request()
    goal = message.motion_plan_request
    goal.group_name = str(request["moveit_group"])
    goal.start_state.is_diff = True
    goal.num_planning_attempts = 3
    goal.allowed_planning_time = 5.0
    constraint = PositionConstraint()
    constraint.header.frame_id = str(request["target_frame"])
    constraint.link_name = str(request["end_effector_link"])
    constraint.weight = 1.0
    region = SolidPrimitive()
    region.type = SolidPrimitive.SPHERE
    region.dimensions = [float(request["position_tolerance_m"])]
    pose = Pose()
    pose.position.x, pose.position.y, pose.position.z = map(float, location["translation_m"])
    pose.orientation.w = 1.0
    constraint.constraint_region.primitives = [region]
    constraint.constraint_region.primitive_poses = [pose]
    constraints = Constraints()
    constraints.position_constraints = [constraint]
    # A referenced object position does not specify a grasp or insertion orientation.
    goal.goal_constraints = [constraints]
    return message


def _location_plan_result(response: Any, request: Mapping[str, object]) -> dict[str, object]:
    code = int(response.error_code.val)
    if code != 1:
        return {
            "status": "rejected",
            "message": "MoveIt found no valid plan to this location.",
            "error_code": code,
            "plan": None,
        }
    trajectory = response.trajectory.joint_trajectory
    start = response.trajectory_start.joint_state
    if response.group_name != request["moveit_group"] or not trajectory.points or not start.name:
        return _location_unavailable("MoveIt returned incomplete planning evidence.")
    return {
        "status": "accepted",
        "message": "MoveIt planned to this position with joint limits and scene collision checking.",
        "error_code": code,
        "plan": {
            "joint_names": list(trajectory.joint_names),
            "points": [list(point.positions) for point in trajectory.points],
            "start_joint_names": list(start.name),
            "start_joint_positions": list(start.position),
        },
    }


def _location_unavailable(message: str) -> dict[str, object]:
    return {"status": "needs_context", "message": message, "error_code": None, "plan": None}


def _locations_unavailable(request: Mapping[str, object], message: str) -> dict[str, object]:
    return {
        "status": "needs_context",
        "state_locations": {
            state: [
                {"evidence_handle": location["evidence_handle"], **_location_unavailable(message)}
                for location in locations
            ]
            for state, locations in request["state_locations"].items()
        },
        "feedback": message,
    }
