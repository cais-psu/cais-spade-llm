#!/usr/bin/env python3
"""RViz targets initialized from the live recovery-framework robot state."""

from __future__ import annotations

import copy
import math
import threading
import time
from typing import Any

from kmr_base_controller import (
    occupancy_grid_footprint_is_clear,
    planar_pose,
    prepare_base_path,
)

UR_JOINT_SUFFIXES = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)
KMR_ARM_JOINTS = tuple(f"joint_a{index}" for index in range(1, 8))
ARM_SPECS = {
    **{
        f"ur5e-{index}": {
            "group": f"ur5e_{index}_ur_manipulator",
            "tcp": f"ur5e_{index}_rg2_gripper_tcp",
            "joints": tuple(f"ur5e_{index}_{suffix}" for suffix in UR_JOINT_SUFFIXES),
        }
        for index in range(1, 5)
    },
    "KMR": {
        "group": "KMR_iiwa_arm",
        "tcp": "rg2_gripper_tcp",
        "joints": KMR_ARM_JOINTS,
    },
}
EXPECTED_CONTROLLED_JOINTS = {
    *(joint for spec in ARM_SPECS.values() for joint in spec["joints"]),
    *(f"ur5e_{index}_rg2_finger_width" for index in range(1, 5)),
    "KMR_rg2_finger_width",
}


def main() -> None:
    """Run the recovery-specific RViz marker server."""

    import rclpy
    from action_msgs.msg import GoalStatus
    from cais_lab_robotics.action import DockKMR
    from geometry_msgs.msg import Pose, Pose2D, PoseStamped
    from interactive_markers.interactive_marker_server import InteractiveMarkerServer
    from interactive_markers.menu_handler import MenuHandler
    from moveit_msgs.action import ExecuteTrajectory, MoveGroup
    from moveit_msgs.msg import (
        Constraints,
        JointConstraint,
        OrientationConstraint,
        PlanningOptions,
        PositionConstraint,
        RobotState,
    )
    from nav2_msgs.action import ComputePathToPose, FollowPath
    from nav_msgs.msg import OccupancyGrid, Path
    from rclpy.action import ActionClient
    from rclpy.callback_groups import ReentrantCallbackGroup
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import JointState
    from shape_msgs.msg import SolidPrimitive
    from std_srvs.srv import Trigger
    from tf2_ros import Buffer, TransformException, TransformListener
    from visualization_msgs.msg import InteractiveMarker, InteractiveMarkerControl, Marker

    class RecoveryDragMarkers(Node):
        """Expose current-state arm goals and configured KMR docking commands."""

        def __init__(self) -> None:
            super().__init__("recovery_drag_markers")
            callback_group = ReentrantCallbackGroup()
            self.server = InteractiveMarkerServer(self, "recovery_drag_markers")
            self.tf_buffer = Buffer()
            self.tf_listener = TransformListener(self.tf_buffer, self)
            self.move_group_client = ActionClient(
                self, MoveGroup, "/move_action", callback_group=callback_group
            )
            self.execute_client = ActionClient(
                self, ExecuteTrajectory, "/execute_trajectory", callback_group=callback_group
            )
            self.dock_client = ActionClient(
                self, DockKMR, "/KMR/dock", callback_group=callback_group
            )
            self.compute_path_client = ActionClient(
                self,
                ComputePathToPose,
                "/KMR/compute_path_to_pose",
                callback_group=callback_group,
            )
            self.follow_path_client = ActionClient(
                self,
                FollowPath,
                "/KMR/validated_follow_path",
                callback_group=callback_group,
            )
            self.cancel_base_client = self.create_client(
                Trigger,
                "/KMR/cancel_base_motion",
                callback_group=callback_group,
            )
            self.joint_positions: dict[str, float] = {}
            self.targets: dict[str, Pose] = {}
            self.last_trajectories: dict[str, Any] = {}
            self.busy = False
            self.base_pose: Pose2D | None = None
            self.base_target: Pose2D | None = None
            self.occupancy_map: OccupancyGrid | None = None
            self.last_base_path: Path | None = None
            self.last_base_plan_start: Pose2D | None = None
            self.dock_goal_handle: Any | None = None
            self.compute_path_goal_handle: Any | None = None
            self.follow_path_goal_handle: Any | None = None
            self.markers_ready = False
            self.arm_menu = MenuHandler()
            self.base_menu = MenuHandler()
            self.arm_menu_actions: dict[int, str] = {}
            self.base_menu_actions: dict[int, str] = {}
            for label, action in (
                ("Plan selected arm", "plan_arm"),
                ("Execute selected plan", "execute_arm"),
                ("Plan+Execute selected arm", "plan_execute_arm"),
                ("Plan all_robots", "plan_all"),
                ("Plan+Execute all_robots", "plan_execute_all"),
                ("Reset target to current", "reset_one"),
                ("Reset all targets to current", "reset_all"),
            ):
                entry = self.arm_menu.insert(label, callback=self._arm_menu_cb)
                self.arm_menu_actions[int(entry)] = action
            for label, action in (
                ("Plan KMR base (path only)", "plan_base"),
                ("Execute stored KMR base plan (moves)", "execute_base"),
                ("Plan+Execute KMR base (moves)", "plan_execute_base"),
                ("Reset KMR base target to current", "reset_base"),
            ):
                entry = self.base_menu.insert(label, callback=self._base_menu_cb)
                self.base_menu_actions[int(entry)] = action
            for resource in ("Storage", "M1", "M2"):
                entry = self.base_menu.insert(
                    f"Dock at {resource}", callback=self._base_menu_cb
                )
                self.base_menu_actions[int(entry)] = f"dock_{resource}"
            cancel_entry = self.base_menu.insert(
                "Cancel KMR base motion", callback=self._base_menu_cb
            )
            self.base_menu_actions[int(cancel_entry)] = "cancel_base"
            self.base_path_pub = self.create_publisher(Path, "/KMR/planned_path", 1)
            self.create_subscription(
                JointState,
                "/joint_states",
                self._joint_state_cb,
                50,
                callback_group=callback_group,
            )
            self.create_subscription(
                Pose2D,
                "/KMR/current_pose",
                self._base_pose_cb,
                20,
                callback_group=callback_group,
            )
            map_qos = QoSProfile(depth=1)
            map_qos.reliability = ReliabilityPolicy.RELIABLE
            map_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
            self.create_subscription(
                OccupancyGrid,
                "/KMR/map",
                self._map_cb,
                map_qos,
                callback_group=callback_group,
            )
            self.create_service(
                Trigger,
                "/recovery_drag_markers/resync",
                self._resync_cb,
                callback_group=callback_group,
            )
            self.create_timer(0.25, self._refresh_markers, callback_group=callback_group)

        def _joint_state_cb(self, message: JointState) -> None:
            for name, position in zip(message.name, message.position):
                self.joint_positions[str(name)] = float(position)

        def _base_pose_cb(self, message: Pose2D) -> None:
            self.base_pose = copy.deepcopy(message)

        def _map_cb(self, message: OccupancyGrid) -> None:
            self.occupancy_map = message

        def _current_arm_pose(self, resource: str) -> Pose | None:
            try:
                transform = self.tf_buffer.lookup_transform(
                    "world", str(ARM_SPECS[resource]["tcp"]), rclpy.time.Time()
                )
            except TransformException:
                return None
            pose = Pose()
            pose.position.x = transform.transform.translation.x
            pose.position.y = transform.transform.translation.y
            pose.position.z = transform.transform.translation.z
            pose.orientation = transform.transform.rotation
            return pose

        @staticmethod
        def _arm_marker(resource: str, pose: Pose) -> InteractiveMarker:
            marker = InteractiveMarker()
            marker.header.frame_id = "world"
            marker.name = f"{resource}_goal"
            marker.description = f"{resource} goal"
            marker.scale = 0.22
            marker.pose = copy.deepcopy(pose)
            sphere = Marker()
            sphere.type = Marker.SPHERE
            sphere.scale.x = sphere.scale.y = sphere.scale.z = 0.075
            sphere.color.r, sphere.color.g, sphere.color.b, sphere.color.a = 1.0, 0.45, 0.1, 0.85
            visible = InteractiveMarkerControl()
            visible.name = f"{resource}_goal_sphere"
            visible.always_visible = True
            visible.interaction_mode = InteractiveMarkerControl.MOVE_3D
            visible.markers.append(sphere)
            marker.controls.append(visible)
            axes = (
                ("x", 1.0, 0.0, 0.0),
                ("y", 0.0, 0.0, 1.0),
                ("z", 0.0, 1.0, 0.0),
            )
            for axis, x, y, z in axes:
                for mode, suffix in (
                    (InteractiveMarkerControl.ROTATE_AXIS, "rotate"),
                    (InteractiveMarkerControl.MOVE_AXIS, "move"),
                ):
                    control = InteractiveMarkerControl()
                    control.name = f"{resource}_{suffix}_{axis}"
                    control.orientation.w = 1.0
                    control.orientation.x = x
                    control.orientation.y = y
                    control.orientation.z = z
                    control.orientation_mode = InteractiveMarkerControl.FIXED
                    control.interaction_mode = mode
                    marker.controls.append(control)
            return marker

        @staticmethod
        def _base_marker(pose: Pose2D) -> InteractiveMarker:
            marker = InteractiveMarker()
            marker.header.frame_id = "world"
            marker.name = "KMR_base"
            marker.description = "KMR_base: drag blue pad to move; drag ring to rotate; right-click for actions"
            marker.scale = 0.7
            marker.pose.position.x = pose.x
            marker.pose.position.y = pose.y
            marker.pose.position.z = 0.76
            marker.pose.orientation.z = math.sin(pose.theta / 2.0)
            marker.pose.orientation.w = math.cos(pose.theta / 2.0)
            body = Marker()
            body.type = Marker.CUBE
            body.pose.position.x = 0.25
            body.pose.orientation.w = 1.0
            body.scale.x, body.scale.y, body.scale.z = 0.35, 0.25, 0.06
            body.color.r, body.color.g, body.color.b, body.color.a = 0.1, 0.45, 1.0, 0.55
            control = InteractiveMarkerControl()
            control.name = "KMR_move_xy"
            control.description = "Drag blue pad to move KMR_base"
            control.always_visible = True
            control.orientation.w = math.sqrt(0.5)
            control.orientation.y = math.sqrt(0.5)
            # The XY plane is unchanged by yaw, and the pad must show the staged heading.
            control.orientation_mode = InteractiveMarkerControl.INHERIT
            control.interaction_mode = InteractiveMarkerControl.MOVE_PLANE
            control.markers.append(body)
            marker.controls.append(control)
            rotate = InteractiveMarkerControl()
            rotate.name = "KMR_rotate_yaw"
            rotate.description = "Drag ring to rotate KMR_base"
            rotate.always_visible = True
            rotate.orientation.w = math.sqrt(0.5)
            rotate.orientation.y = math.sqrt(0.5)
            rotate.orientation_mode = InteractiveMarkerControl.INHERIT
            rotate.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
            marker.controls.append(rotate)
            return marker

        def _state_ready(self) -> bool:
            return EXPECTED_CONTROLLED_JOINTS <= set(self.joint_positions) and self.base_pose is not None

        def _refresh_markers(self, force: bool = False) -> None:
            if not self._state_ready() or (self.markers_ready and not force):
                return
            changed = False
            for resource in ARM_SPECS:
                pose = self.targets.get(resource)
                if pose is None:
                    pose = self._current_arm_pose(resource)
                if pose is None:
                    return
                marker = self._arm_marker(resource, pose)
                self.server.insert(marker, feedback_callback=self._arm_feedback_cb)
                self.arm_menu.apply(self.server, marker.name)
                changed = True
            if self.base_pose is not None:
                base = self._base_marker(self.base_target or self.base_pose)
                self.server.insert(base, feedback_callback=self._base_feedback_cb)
                self.base_menu.apply(self.server, base.name)
                changed = True
            if changed:
                self.server.applyChanges()
                if not self.markers_ready:
                    self.get_logger().info(
                        "Recovery markers initialized from live joint and KMR odometry state"
                    )
                self.markers_ready = True

        @staticmethod
        def _resource_from_marker(name: str) -> str | None:
            suffix = "_goal"
            resource = name[:-len(suffix)] if name.endswith(suffix) else ""
            return resource if resource in ARM_SPECS else None

        def _arm_feedback_cb(self, feedback: Any) -> None:
            from visualization_msgs.msg import InteractiveMarkerFeedback

            resource = self._resource_from_marker(feedback.marker_name)
            if resource is None:
                return
            if feedback.event_type in (
                InteractiveMarkerFeedback.POSE_UPDATE,
                InteractiveMarkerFeedback.MOUSE_UP,
            ):
                self.targets[resource] = copy.deepcopy(feedback.pose)

        def _base_feedback_cb(self, feedback: Any) -> None:
            from visualization_msgs.msg import InteractiveMarkerFeedback

            if feedback.event_type not in (
                InteractiveMarkerFeedback.POSE_UPDATE,
                InteractiveMarkerFeedback.MOUSE_UP,
            ):
                return
            self._stage_base_target(feedback.pose)

        def _stage_base_target(self, pose: Pose) -> None:
            """Store the staged planar target carried by RViz marker feedback."""

            orientation = pose.orientation
            yaw = math.atan2(
                2.0 * (orientation.w * orientation.z + orientation.x * orientation.y),
                1.0 - 2.0 * (orientation.y * orientation.y + orientation.z * orientation.z),
            )
            self._clear_base_path()
            self.base_target = Pose2D(
                x=pose.position.x,
                y=pose.position.y,
                theta=yaw,
            )

        def _resync_cb(self, _request: Any, response: Any) -> Any:
            self.targets.clear()
            self.last_trajectories.clear()
            self.base_target = None
            self._clear_base_path()
            self._refresh_markers(force=True)
            response.success = self._state_ready()
            response.message = (
                "Recovery targets reset to current robot poses"
                if response.success
                else "Current recovery robot state is incomplete"
            )
            return response

        def _arm_menu_cb(self, feedback: Any) -> None:
            action = self.arm_menu_actions.get(int(feedback.menu_entry_id))
            resource = self._resource_from_marker(feedback.marker_name)
            if action is None or resource is None or self.busy:
                return
            if action == "reset_one":
                self.targets.pop(resource, None)
                self.last_trajectories.pop(resource, None)
                self._refresh_markers(force=True)
                return
            if action == "reset_all":
                self.targets.clear()
                self.last_trajectories.clear()
                self._refresh_markers(force=True)
                return
            self.busy = True
            threading.Thread(
                target=self._run_arm_action,
                args=(resource, action),
                daemon=True,
            ).start()

        def _base_menu_cb(self, feedback: Any) -> None:
            command = self.base_menu_actions.get(int(feedback.menu_entry_id))
            if command is None or (self.busy and command != "cancel_base"):
                return
            if command == "cancel_base":
                for handle in (
                    self.compute_path_goal_handle,
                    self.follow_path_goal_handle,
                    self.dock_goal_handle,
                ):
                    if handle is not None:
                        handle.cancel_goal_async()
                if self.cancel_base_client.service_is_ready():
                    self.cancel_base_client.call_async(Trigger.Request())
                else:
                    self.get_logger().warning("/KMR/cancel_base_motion is unavailable")
                return
            if command == "reset_base":
                self.base_target = None
                self._clear_base_path()
                self._refresh_markers(force=True)
                return
            if command in {"plan_base", "plan_execute_base"}:
                self._stage_base_target(feedback.pose)
            self.busy = True
            threading.Thread(target=self._run_base_action, args=(command,), daemon=True).start()

        def _clear_base_path(self) -> None:
            self.last_base_path = None
            self.last_base_plan_start = None
            empty = Path()
            empty.header.frame_id = "world"
            empty.header.stamp = self.get_clock().now().to_msg()
            self.base_path_pub.publish(empty)

        def _plan_base(self) -> bool:
            if self.base_target is None:
                self.get_logger().warning("Drag KMR_base before planning")
                return False
            if self.base_pose is None:
                self.get_logger().error("Current KMR base pose is unavailable")
                return False
            self._clear_base_path()
            plan_start = copy.deepcopy(self.base_pose)
            plan_target = copy.deepcopy(self.base_target)
            if self.occupancy_map is None:
                self.get_logger().error("/KMR/map is unavailable")
                return False
            map_info = self.occupancy_map.info
            if not occupancy_grid_footprint_is_clear(
                self.occupancy_map.data,
                map_info.width,
                map_info.height,
                map_info.resolution,
                map_info.origin.position.x,
                map_info.origin.position.y,
                plan_target.x,
                plan_target.y,
                plan_target.theta,
            ):
                self.get_logger().error(
                    "KMR base target is outside the map or overlaps a fixed obstacle"
                )
                return False
            if not self.compute_path_client.wait_for_server(timeout_sec=2.0):
                self.get_logger().error("/KMR/compute_path_to_pose is unavailable")
                return False
            goal = ComputePathToPose.Goal()
            goal.goal.header.frame_id = "world"
            goal.goal.header.stamp = self.get_clock().now().to_msg()
            goal.goal.pose.position.x = plan_target.x
            goal.goal.pose.position.y = plan_target.y
            goal.goal.pose.orientation.z = math.sin(plan_target.theta / 2.0)
            goal.goal.pose.orientation.w = math.cos(plan_target.theta / 2.0)
            goal.planner_id = "GridBased"
            goal.use_start = True
            goal.start = copy.deepcopy(goal.goal)
            goal.start.pose.position.x = plan_start.x
            goal.start.pose.position.y = plan_start.y
            goal.start.pose.orientation.z = math.sin(plan_start.theta / 2.0)
            goal.start.pose.orientation.w = math.cos(plan_start.theta / 2.0)
            if math.hypot(plan_start.x - plan_target.x, plan_start.y - plan_target.y) < 1e-5:
                path = Path()
                path.header = copy.deepcopy(goal.goal.header)
                path.poses = [goal.start, goal.goal]
            else:
                handle = self._wait_future(self.compute_path_client.send_goal_async(goal), 5.0)
                self.compute_path_goal_handle = handle
                if handle is None or not handle.accepted:
                    self.get_logger().error("KMR base target was rejected")
                    return False
                wrapped = self._wait_future(handle.get_result_async(), 20.0)
                self.compute_path_goal_handle = None
                if wrapped is None or wrapped.status != GoalStatus.STATUS_SUCCEEDED:
                    self.get_logger().error("No collision-free KMR base path was found")
                    return False
                path = wrapped.result.path
            if not path.poses:
                self.get_logger().error("Nav2 returned an empty KMR base path")
                return False
            try:
                path = prepare_base_path(
                    path, (plan_start.x, plan_start.y, plan_start.theta),
                    (plan_target.x, plan_target.y, plan_target.theta),
                    self.occupancy_map,
                )
            except ValueError as exc:
                self.get_logger().error(str(exc))
                return False
            if self.base_target != plan_target:
                self.get_logger().warning("KMR target changed during planning; plan again")
                return False
            self.last_base_path = path
            self.last_base_plan_start = plan_start
            self.base_path_pub.publish(path)
            self.get_logger().info(f"KMR base path contains {len(path.poses)} poses")
            return True

        def _execute_base(self) -> bool:
            if self.last_base_path is None or not self.last_base_path.poses:
                self.get_logger().error("No stored KMR base path")
                return False
            if self.base_pose is None or self.last_base_plan_start is None:
                self.get_logger().error("KMR base plan start is unavailable")
                return False
            start = self.last_base_plan_start
            if (
                math.hypot(start.x - self.base_pose.x, start.y - self.base_pose.y)
                > 0.10
                or abs((start.theta - self.base_pose.theta + math.pi) % (2.0 * math.pi) - math.pi)
                > 0.15
            ):
                self.get_logger().error("KMR moved after planning; plan again from the current pose")
                return False
            if not self.follow_path_client.wait_for_server(timeout_sec=2.0):
                self.get_logger().error("/KMR/validated_follow_path is unavailable")
                return False
            goal = FollowPath.Goal()
            try:
                goal.path = prepare_base_path(
                    self.last_base_path,
                    (self.base_pose.x, self.base_pose.y, self.base_pose.theta),
                    planar_pose(self.last_base_path.poses[-1].pose), self.occupancy_map,
                )
            except ValueError as exc:
                self.get_logger().error(str(exc))
                self._clear_base_path()
                return False
            goal.controller_id = "FollowPath"
            goal.goal_checker_id = "precise_goal_checker"
            handle = self._wait_future(self.follow_path_client.send_goal_async(goal), 5.0)
            self.follow_path_goal_handle = handle
            if handle is None or not handle.accepted:
                self.get_logger().error("KMR base path execution was rejected")
                return False
            wrapped = self._wait_future(handle.get_result_async(), 180.0)
            self.follow_path_goal_handle = None
            if wrapped is None or wrapped.status != GoalStatus.STATUS_SUCCEEDED:
                self.get_logger().error("KMR base path execution failed or was canceled")
                return False
            self.base_target = None
            self._clear_base_path()
            return True

        def _dock_base(self, resource: str) -> bool:
            if not self.dock_client.wait_for_server(timeout_sec=2.0):
                self.get_logger().error("/KMR/dock is unavailable")
                return False
            goal = DockKMR.Goal()
            goal.target_resource = resource
            handle = self._wait_future(self.dock_client.send_goal_async(goal), 5.0)
            self.dock_goal_handle = handle
            if handle is None or not handle.accepted:
                self.get_logger().warning("KMR docking request was rejected")
                return False
            wrapped = self._wait_future(handle.get_result_async(), 180.0)
            self.dock_goal_handle = None
            if wrapped is None:
                self.get_logger().error("KMR docking timed out")
                return False
            result = wrapped.result
            logger = self.get_logger().info if result.success else self.get_logger().error
            logger(result.message)
            if result.success:
                self.base_target = None
                self._clear_base_path()
            return bool(result.success)

        def _run_base_action(self, command: str) -> None:
            try:
                if command == "plan_base":
                    self._plan_base()
                elif command == "execute_base":
                    self._execute_base()
                elif command == "plan_execute_base" and self._plan_base():
                    self._execute_base()
                elif command.startswith("dock_"):
                    self._dock_base(command.removeprefix("dock_"))
            except Exception as exc:
                self.get_logger().error(f"KMR base command failed: {exc}")
            finally:
                self.compute_path_goal_handle = None
                self.follow_path_goal_handle = None
                self.dock_goal_handle = None
                self.busy = False
                self._refresh_markers(force=True)

        @staticmethod
        def _pose_constraints(
            resource: str, pose: Pose, position_constraint_type: Any, orientation_constraint_type: Any,
            solid_primitive_type: Any, pose_stamped_type: Any,
        ) -> tuple[Any, Any]:
            tcp = str(ARM_SPECS[resource]["tcp"])
            position = position_constraint_type()
            position.header.frame_id = "world"
            position.link_name = tcp
            box = solid_primitive_type()
            box.type = solid_primitive_type.BOX
            box.dimensions = [0.004, 0.004, 0.004]
            region_pose = pose_stamped_type().pose
            region_pose.position = copy.deepcopy(pose.position)
            region_pose.orientation.w = 1.0
            position.constraint_region.primitives.append(box)
            position.constraint_region.primitive_poses.append(region_pose)
            position.weight = 1.0
            orientation = orientation_constraint_type()
            orientation.header.frame_id = "world"
            orientation.link_name = tcp
            orientation.orientation = copy.deepcopy(pose.orientation)
            orientation.absolute_x_axis_tolerance = 0.015
            orientation.absolute_y_axis_tolerance = 0.015
            orientation.absolute_z_axis_tolerance = 0.015
            orientation.weight = 1.0
            return position, orientation

        def _move_group_goal(self, resources: list[str], plan_only: bool) -> Any:
            goal = MoveGroup.Goal()
            goal.request.group_name = (
                "all_robots" if len(resources) > 1 else str(ARM_SPECS[resources[0]]["group"])
            )
            goal.request.pipeline_id = "ompl"
            goal.request.planner_id = "RRTConnectkConfigDefault"
            goal.request.num_planning_attempts = 5
            goal.request.allowed_planning_time = 5.0
            goal.request.max_velocity_scaling_factor = 0.4
            goal.request.max_acceleration_scaling_factor = 0.3
            goal.request.start_state = RobotState(is_diff=True)
            constraints = Constraints()
            for resource in resources:
                pose = self.targets.get(resource)
                if pose is None:
                    continue
                position, orientation = self._pose_constraints(
                    resource,
                    pose,
                    PositionConstraint,
                    OrientationConstraint,
                    SolidPrimitive,
                    PoseStamped,
                )
                constraints.position_constraints.append(position)
                constraints.orientation_constraints.append(orientation)
            if len(resources) > 1:
                for resource, spec in ARM_SPECS.items():
                    if resource in self.targets:
                        continue
                    for joint in spec["joints"]:
                        constraint = JointConstraint()
                        constraint.joint_name = str(joint)
                        constraint.position = self.joint_positions[str(joint)]
                        constraint.tolerance_above = 0.002
                        constraint.tolerance_below = 0.002
                        constraint.weight = 1.0
                        constraints.joint_constraints.append(constraint)
            goal.request.goal_constraints = [constraints]
            goal.planning_options = PlanningOptions()
            goal.planning_options.plan_only = plan_only
            goal.planning_options.look_around = False
            goal.planning_options.replan = False
            return goal

        @staticmethod
        def _wait_future(future: Any, timeout: float) -> Any:
            deadline = time.monotonic() + timeout
            while not future.done() and time.monotonic() < deadline:
                time.sleep(0.05)
            return future.result() if future.done() else None

        def _run_arm_action(self, resource: str, action: str) -> None:
            key = "all_robots" if action.endswith("all") else resource
            try:
                if action.startswith("execute"):
                    trajectory = self.last_trajectories.get(key)
                    if trajectory is None:
                        self.get_logger().error(f"No stored trajectory for {key}")
                        return
                    if not self.execute_client.wait_for_server(timeout_sec=2.0):
                        self.get_logger().error("/execute_trajectory is unavailable")
                        return
                    goal = ExecuteTrajectory.Goal(trajectory=trajectory)
                    handle = self._wait_future(self.execute_client.send_goal_async(goal), 5.0)
                    if handle is None or not handle.accepted:
                        self.get_logger().error(f"{key} trajectory execution was rejected")
                        return
                    result = self._wait_future(handle.get_result_async(), 120.0)
                    if result is None or result.result.error_code.val != 1:
                        self.get_logger().error(f"{key} trajectory execution failed")
                        return
                    self.targets.clear() if key == "all_robots" else self.targets.pop(resource, None)
                    self.last_trajectories.pop(key, None)
                    return
                resources = list(ARM_SPECS) if action.endswith("all") else [resource]
                if not any(item in self.targets for item in resources):
                    self.get_logger().warning(f"No staged target for {key}")
                    return
                if not self.move_group_client.wait_for_server(timeout_sec=2.0):
                    self.get_logger().error("/move_action is unavailable")
                    return
                plan_only = action.startswith("plan_") and not action.startswith("plan_execute")
                goal = self._move_group_goal(resources, plan_only=plan_only)
                handle = self._wait_future(self.move_group_client.send_goal_async(goal), 5.0)
                if handle is None or not handle.accepted:
                    self.get_logger().error(f"{key} planning request was rejected")
                    return
                result = self._wait_future(handle.get_result_async(), 120.0)
                if result is None or result.result.error_code.val != 1:
                    self.get_logger().error(f"{key} planning or execution failed")
                    return
                if plan_only:
                    self.last_trajectories[key] = result.result.planned_trajectory
                else:
                    self.targets.clear() if key == "all_robots" else self.targets.pop(resource, None)
                    self.last_trajectories.pop(key, None)
            finally:
                self.busy = False
                self._refresh_markers(force=True)

    rclpy.init()
    node = RecoveryDragMarkers()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
