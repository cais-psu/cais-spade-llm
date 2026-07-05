#!/usr/bin/env python3.10
"""Dual robots paired RViz drag markers."""

from __future__ import annotations

import argparse
import copy
import json
import threading
import time
from pathlib import Path

import rclpy
from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import Pose
from interactive_markers.interactive_marker_server import InteractiveMarkerServer
from interactive_markers.menu_handler import MenuHandler
from moveit_msgs.action import ExecuteTrajectory
from moveit_msgs.msg import DisplayTrajectory, RobotState, RobotTrajectory
from moveit_msgs.srv import GetCartesianPath, GetStateValidity
from rclpy.action import ActionClient
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectoryPoint
from visualization_msgs.msg import (
    InteractiveMarker,
    InteractiveMarkerControl,
    InteractiveMarkerFeedback,
    Marker,
)

try:
    from keyboard_teleop import ROBOTS, KeyboardTeleop, wait_for_joint_positions
except Exception:
    import sys

    sys.path.append(str(Path(__file__).resolve().parent))
    from keyboard_teleop import ROBOTS, KeyboardTeleop, wait_for_joint_positions


UR5E_TRAJECTORY_ACTION = "/cais_ur5e_rtde_trajectory_controller/follow_joint_trajectory"
UR5E_JOINT_STATES_STALE_SEC = 3.0


class DualDragMarkers(KeyboardTeleop):
    def __init__(
        self,
        *,
        mode: str,
        marker_scale: float,
        velocity_scale: float,
        cartesian_max_step_mm: float,
        service_timeout_sec: float,
        execution_policy: str,
        status_file: str = "",
    ) -> None:
        self._ur5e_last_joint_state_monotonic: float | None = None
        super().__init__(
            cartesian_max_step_mm=cartesian_max_step_mm,
            joint_duration_sec=0.35,
            gripper_duration_sec=0.20,
            node_name="dual_drag_markers",
        )
        self.mode = str(mode or "monitor").strip().lower()
        self.marker_scale = max(0.05, float(marker_scale))
        self.velocity_scale = self._normalize_velocity_scale(velocity_scale)
        self.service_timeout_sec = max(1.0, float(service_timeout_sec))
        self.execution_policy = str(execution_policy or "immediate").strip().lower()
        self.status_file = Path(status_file) if str(status_file or "").strip() else None
        if self.execution_policy not in {"immediate", "paired"}:
            self.execution_policy = "immediate"
        self.server = InteractiveMarkerServer(self, "dual_drag_markers")
        self.menu_handler = MenuHandler()
        self._menu_actions: dict[int, str] = {}
        self._busy: dict[str, bool] = {"xarm6": False, "ur5e": False}
        self._last_feedback_pose: dict[str, Pose] = {}
        self._paired_targets: dict[str, Pose] = {}
        self._paired_plan: RobotTrajectory | None = None
        self._paired_busy = False
        self.ur5e_trajectory_action_client = ActionClient(
            self,
            FollowJointTrajectory,
            UR5E_TRAJECTORY_ACTION,
            callback_group=self.cb_group,
        )
        self.state_validity_client = self.create_client(
            GetStateValidity, "/check_state_validity", callback_group=self.cb_group
        )
        self.display_pub = self.create_publisher(DisplayTrajectory, "/move_group/display_planned_path", 10)
        if self.execution_policy == "paired":
            self._install_paired_menu()
        self._write_status(
            state="ready",
            action="",
            stage="ready",
            message=(
                f"dual_drag_markers ready mode={self.mode}; "
                f"execution-policy={self.execution_policy}"
            ),
        )
        self.create_timer(0.5, self._refresh_markers)

    def _joint_state_cb(self, msg: JointState) -> None:
        super()._joint_state_cb(msg)
        names = {str(name) for name in msg.name}
        for joint_names in self._joint_name_candidates("ur5e"):
            if all(str(joint) in names for joint in joint_names):
                self._ur5e_last_joint_state_monotonic = time.monotonic()
                break

    def _ur5e_joint_states_fresh(self) -> bool:
        if self._ur5e_last_joint_state_monotonic is None:
            return False
        return (time.monotonic() - self._ur5e_last_joint_state_monotonic) <= UR5E_JOINT_STATES_STALE_SEC

    def _ur5e_execution_health(self) -> tuple[bool, str]:
        trajectory_action_available = self.ur5e_trajectory_action_client.wait_for_server(timeout_sec=1.0)
        joint_states_fresh = self._ur5e_joint_states_fresh()
        ready = (
            bool(trajectory_action_available)
            and joint_states_fresh
        )
        detail = (
            f"trajectory_action_available={bool(trajectory_action_available)}; "
            f"rtde_action={UR5E_TRAJECTORY_ACTION}; "
            f"joint_states_fresh={joint_states_fresh}"
        )
        if ready:
            return True, f"UR5e execution health ready; {detail}"
        return False, f"UR5e execution health not ready; {detail}"

    def _write_status(
        self,
        *,
        state: str,
        action: str,
        stage: str,
        message: str,
        last_error: str = "",
    ) -> None:
        if self.status_file is None:
            return
        payload = {
            "updated_at": time.time(),
            "node": "dual_drag_markers",
            "mode": self.mode,
            "execution_policy": self.execution_policy,
            "state": str(state or "unknown"),
            "action": str(action or ""),
            "stage": str(stage or ""),
            "message": str(message or ""),
            "last_error": str(last_error or ""),
        }
        try:
            self.status_file.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.status_file.with_suffix(self.status_file.suffix + ".tmp")
            tmp.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
            tmp.replace(self.status_file)
        except Exception as exc:
            self.get_logger().warn(f"dual_drag_markers status write failed: {exc}")

    def _install_paired_menu(self) -> None:
        for label, action in (
            ("Plan dual_robots", "plan"),
            ("Execute dual_robots", "execute"),
            ("Plan+Execute dual_robots", "plan_execute"),
            ("Clear dual_robots plan", "clear"),
        ):
            entry_id = self.menu_handler.insert(label, callback=self._menu_feedback_cb)
            self._menu_actions[int(entry_id)] = action

    def _marker_color(self, robot: str) -> tuple[float, float, float]:
        if robot == "xarm6":
            return 0.10, 0.55, 1.00
        return 1.00, 0.45, 0.10

    def _marker_for_robot(self, robot: str, pose: Pose) -> InteractiveMarker:
        marker = InteractiveMarker()
        marker.header.frame_id = self._current_frame_id(robot)
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.name = f"{robot}_drag_ball"
        marker.description = f"{robot} drag ball"
        marker.scale = self.marker_scale
        marker.pose = copy.deepcopy(pose)

        sphere = Marker()
        sphere.type = Marker.SPHERE
        sphere.scale.x = self.marker_scale * 0.42
        sphere.scale.y = self.marker_scale * 0.42
        sphere.scale.z = self.marker_scale * 0.42
        r, g, b = self._marker_color(robot)
        sphere.color.r = r
        sphere.color.g = g
        sphere.color.b = b
        sphere.color.a = 0.82

        sphere_control = InteractiveMarkerControl()
        sphere_control.name = f"{robot}_move_3d"
        sphere_control.always_visible = True
        sphere_control.interaction_mode = InteractiveMarkerControl.MOVE_3D
        sphere_control.markers.append(sphere)
        marker.controls.append(sphere_control)

        for name, axis in (
            ("move_x", "x"),
            ("move_y", "y"),
            ("move_z", "z"),
        ):
            control = InteractiveMarkerControl()
            control.name = f"{robot}_{name}"
            control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
            control.orientation_mode = InteractiveMarkerControl.FIXED
            control.orientation.w = 1.0
            if axis == "x":
                control.orientation.x = 1.0
            elif axis == "y":
                control.orientation.y = 1.0
            else:
                control.orientation.z = 1.0
            marker.controls.append(control)

        return marker

    @staticmethod
    def _robot_from_marker_name(name: str) -> str:
        value = str(name or "")
        if value.startswith("xarm6"):
            return "xarm6"
        if value.startswith("ur5e"):
            return "ur5e"
        return ""

    def _refresh_markers(self) -> None:
        changed = False
        for robot in ("xarm6", "ur5e"):
            if self._busy.get(robot):
                continue
            pose = copy.deepcopy(self._paired_targets.get(robot)) if self.execution_policy == "paired" else None
            if pose is None:
                pose = self.get_ee_pose(robot)
            if pose is None:
                continue
            marker = self._marker_for_robot(robot, pose)
            self.server.insert(marker, feedback_callback=self._feedback_cb)
            if self.execution_policy == "paired":
                self.menu_handler.apply(self.server, marker.name)
            changed = True
        if changed:
            self.server.applyChanges()

    def _feedback_cb(self, feedback: InteractiveMarkerFeedback) -> None:
        robot = self._robot_from_marker_name(feedback.marker_name)
        if robot not in {"xarm6", "ur5e"}:
            return
        if feedback.event_type == InteractiveMarkerFeedback.POSE_UPDATE:
            self._last_feedback_pose[robot] = copy.deepcopy(feedback.pose)
            return
        if feedback.event_type != InteractiveMarkerFeedback.MOUSE_UP:
            return
        target = copy.deepcopy(self._last_feedback_pose.get(robot) or feedback.pose)
        if self.execution_policy == "paired":
            self._stage_paired_target(robot, target)
            return
        if self._busy.get(robot):
            self.get_logger().warn(f"{robot} drag ignored; previous marker command still running")
            return
        self._busy[robot] = True
        threading.Thread(target=self._execute_marker_pose, args=(robot, target), daemon=True).start()

    def _menu_feedback_cb(self, feedback: InteractiveMarkerFeedback) -> None:
        action = self._menu_actions.get(int(feedback.menu_entry_id))
        if not action:
            return
        if self._paired_busy:
            self.get_logger().warn("dual_robots menu ignored; previous paired command still running")
            return
        self._paired_busy = True
        threading.Thread(target=self._run_paired_menu_action, args=(action,), daemon=True).start()

    def _stage_paired_target(self, robot: str, target: Pose) -> None:
        self._paired_targets[robot] = copy.deepcopy(target)
        self._paired_plan = None
        message = f"{robot} target staged for dual_robots"
        self.get_logger().info(message)
        self._write_status(
            state="staged",
            action="stage",
            stage=robot,
            message=message,
        )
        self._refresh_markers()

    def _run_paired_menu_action(self, action: str) -> None:
        action_label = {
            "plan": "Plan dual_robots",
            "execute": "Execute dual_robots",
            "plan_execute": "Plan+Execute dual_robots",
            "clear": "Clear dual_robots plan",
        }.get(action, str(action or "unknown dual_robots action"))
        stage = "starting"
        final_message = ""
        ok = False
        try:
            self._write_status(
                state="running",
                action=action_label,
                stage=stage,
                message=f"{action_label}: started.",
            )
            if action == "clear":
                self._paired_targets.clear()
                self._paired_plan = None
                ok, msg = True, "dual_robots plan cleared"
            elif action == "plan":
                stage = "plan"
                ok, msg = self.plan_dual_robots()
            elif action == "execute":
                stage = "execute"
                ok, msg = self.execute_dual_robots()
            elif action == "plan_execute":
                stage = "plan"
                ok, msg = self.plan_dual_robots()
                if ok:
                    self._write_status(
                        state="running",
                        action=action_label,
                        stage="execute",
                        message=f"{action_label}: plan succeeded; executing dual_robots.",
                    )
                    stage = "execute"
                    ok, msg = self.execute_dual_robots()
            else:
                ok, msg = False, f"unknown dual_robots menu action: {action}"

            if ok:
                final_message = f"{action_label}: {msg}"
            else:
                final_message = f"{action_label} failed during {stage}: {msg}"
            self._write_status(
                state="succeeded" if ok else "failed",
                action=action_label,
                stage=stage,
                message=final_message,
                last_error="" if ok else final_message,
            )
            if ok:
                self.get_logger().info(final_message)
            else:
                self.get_logger().warn(final_message)
        finally:
            self._paired_busy = False
            time.sleep(0.2)
            self._refresh_markers()

    def _execute_marker_pose(self, robot: str, target: Pose) -> None:
        try:
            ok, msg = self.move_to_pose(robot, target, velocity_scale=self.velocity_scale)
            if ok:
                self.get_logger().info(f"{robot} drag executed")
            else:
                self.get_logger().warn(f"{robot} drag failed: {msg}")
        finally:
            self._busy[robot] = False
            time.sleep(0.2)
            self._refresh_markers()

    @staticmethod
    def _point_seconds(point: JointTrajectoryPoint) -> float:
        return float(point.time_from_start.sec) + float(point.time_from_start.nanosec) * 1e-9

    def _trajectory_duration(self, trajectory: RobotTrajectory) -> float:
        points = list(getattr(trajectory.joint_trajectory, "points", []) or [])
        if not points:
            return 0.0
        return max(0.0, self._point_seconds(points[-1]))

    def _compute_robot_cartesian_path(self, robot: str, target: Pose) -> tuple[object | None, str]:
        group_candidates = [self._current_group_name(robot)] + [
            group for group in self._group_name_candidates(robot)
            if group != self._current_group_name(robot)
        ]
        group_errors: list[str] = []

        for group_name in group_candidates:
            request = GetCartesianPath.Request()
            request.header.frame_id = self._current_frame_id(robot)
            request.header.stamp = self.get_clock().now().to_msg()
            request.group_name = group_name
            request.link_name = self._current_ee_link(robot)
            request.waypoints = [copy.deepcopy(target)]
            request.max_step = self.cartesian_max_step_m
            request.jump_threshold = 0.0
            request.avoid_collisions = False
            request.start_state.is_diff = True

            future = self.cartesian_client.call_async(request)
            if not self._wait_future(future, timeout=10.0):
                group_errors.append(f"{group_name}: timeout")
                continue
            candidate = future.result()
            if candidate is None:
                group_errors.append(f"{group_name}: service failed")
                continue
            err_code = getattr(getattr(candidate, "error_code", None), "val", None)
            if err_code not in (None, 1):
                group_errors.append(f"{group_name}: error_code={err_code}")
                continue
            if float(candidate.fraction) <= 0.0:
                group_errors.append(f"{group_name}: no valid cartesian path")
                continue
            if float(candidate.fraction) < 0.9:
                return None, f"{robot} Path incomplete ({float(candidate.fraction):.0%})"

            self.active_group_name[robot] = group_name
            return candidate, "OK"

        if group_errors:
            return None, f"{robot} CartesianPath failed: {'; '.join(group_errors[-2:])}"
        return None, f"{robot} CartesianPath failed"

    def _stretch_trajectory_to_duration(self, trajectory: RobotTrajectory, duration_sec: float) -> None:
        current_duration = self._trajectory_duration(trajectory)
        if current_duration <= 1e-6 or duration_sec <= 1e-6:
            return
        self._scale_trajectory_timing(trajectory, duration_sec / current_duration)

    def _interpolate_joint_positions(self, trajectory: RobotTrajectory, time_sec: float) -> dict[str, float]:
        joint_names = list(trajectory.joint_trajectory.joint_names)
        points = list(trajectory.joint_trajectory.points)
        if not joint_names or not points:
            return {}

        first_time = self._point_seconds(points[0])
        if time_sec <= first_time:
            return dict(zip(joint_names, points[0].positions))

        last_time = self._point_seconds(points[-1])
        if time_sec >= last_time:
            return dict(zip(joint_names, points[-1].positions))

        previous = points[0]
        previous_time = first_time
        for point in points[1:]:
            point_time = self._point_seconds(point)
            if time_sec <= point_time:
                span = max(1e-9, point_time - previous_time)
                ratio = (time_sec - previous_time) / span
                positions = [
                    float(a) + (float(b) - float(a)) * ratio
                    for a, b in zip(previous.positions, point.positions)
                ]
                return dict(zip(joint_names, positions))
            previous = point
            previous_time = point_time
        return dict(zip(joint_names, points[-1].positions))

    def _merge_dual_robots_trajectories(
        self,
        xarm6_trajectory: RobotTrajectory,
        ur5e_trajectory: RobotTrajectory,
    ) -> tuple[RobotTrajectory | None, str]:
        xarm6_duration = self._trajectory_duration(xarm6_trajectory)
        ur5e_duration = self._trajectory_duration(ur5e_trajectory)
        shared_duration = max(xarm6_duration, ur5e_duration)
        if shared_duration <= 1e-6:
            return None, "dual_robots trajectories have no duration"

        xarm6_trajectory = copy.deepcopy(xarm6_trajectory)
        ur5e_trajectory = copy.deepcopy(ur5e_trajectory)
        self._stretch_trajectory_to_duration(xarm6_trajectory, shared_duration)
        self._stretch_trajectory_to_duration(ur5e_trajectory, shared_duration)

        times = {0.0, shared_duration}
        for trajectory in (xarm6_trajectory, ur5e_trajectory):
            for point in trajectory.joint_trajectory.points:
                point_time = min(shared_duration, max(0.0, self._point_seconds(point)))
                times.add(point_time)

        merged = RobotTrajectory()
        merged.joint_trajectory.header.frame_id = self._current_frame_id("xarm6")
        merged.joint_trajectory.header.stamp = self.get_clock().now().to_msg()
        merged.joint_trajectory.joint_names = (
            list(xarm6_trajectory.joint_trajectory.joint_names)
            + list(ur5e_trajectory.joint_trajectory.joint_names)
        )

        for point_time in sorted(times):
            xarm6_positions = self._interpolate_joint_positions(xarm6_trajectory, point_time)
            ur5e_positions = self._interpolate_joint_positions(ur5e_trajectory, point_time)
            positions = [
                xarm6_positions[joint_name]
                for joint_name in xarm6_trajectory.joint_trajectory.joint_names
            ] + [
                ur5e_positions[joint_name]
                for joint_name in ur5e_trajectory.joint_trajectory.joint_names
            ]
            point = JointTrajectoryPoint()
            point.positions = [float(value) for value in positions]
            point.time_from_start = self._duration_msg(point_time)
            merged.joint_trajectory.points.append(point)

        return merged, "OK"

    def _robot_state_from_joint_map(
        self,
        joint_names: list[str] | None = None,
        positions: list[float] | None = None,
    ) -> RobotState:
        joint_state = JointState()
        joint_state.name = list(self.joint_state_map.keys())
        joint_state.position = [float(self.joint_state_map[name]) for name in joint_state.name]

        if joint_names and positions:
            index_by_name = {name: idx for idx, name in enumerate(joint_state.name)}
            for joint_name, position in zip(joint_names, positions):
                if joint_name in index_by_name:
                    joint_state.position[index_by_name[joint_name]] = float(position)
                else:
                    joint_state.name.append(joint_name)
                    joint_state.position.append(float(position))

        robot_state = RobotState()
        robot_state.joint_state = joint_state
        robot_state.is_diff = False
        return robot_state

    def _validate_dual_robots_trajectory(self, trajectory: RobotTrajectory) -> tuple[bool, str]:
        if not self.state_validity_client.wait_for_service(timeout_sec=self.service_timeout_sec):
            return False, "/check_state_validity not available"

        points = list(trajectory.joint_trajectory.points)
        if not points:
            return False, "dual_robots trajectory has no points"

        max_samples = min(25, len(points))
        if max_samples <= 1:
            sample_indexes = [0]
        elif len(points) <= max_samples:
            sample_indexes = list(range(len(points)))
        else:
            sample_indexes = sorted({
                round(i * (len(points) - 1) / (max_samples - 1))
                for i in range(max_samples)
            })

        joint_names = list(trajectory.joint_trajectory.joint_names)
        for index in sample_indexes:
            point = points[index]
            request = GetStateValidity.Request()
            request.group_name = "dual_robots"
            request.robot_state = self._robot_state_from_joint_map(joint_names, list(point.positions))

            future = self.state_validity_client.call_async(request)
            if not self._wait_future(future, timeout=3.0):
                return False, f"dual_robots state validity timeout at sample {index}"
            response = future.result()
            if response is None:
                return False, f"dual_robots state validity failed at sample {index}"
            if not bool(response.valid):
                contacts = getattr(response, "contacts", []) or []
                if contacts:
                    contact = contacts[0]
                    body_1 = str(getattr(contact, "contact_body_1", "") or "")
                    body_2 = str(getattr(contact, "contact_body_2", "") or "")
                    return False, f"dual_robots state invalid at sample {index}: {body_1} vs {body_2}"
                return False, f"dual_robots state invalid at sample {index}"

        return True, "OK"

    def _publish_dual_robots_preview(self, trajectory: RobotTrajectory) -> None:
        display = DisplayTrajectory()
        display.model_id = "dual_robots"
        display.trajectory_start = self._robot_state_from_joint_map()
        display.trajectory.append(trajectory)
        self.display_pub.publish(display)

    def plan_dual_robots(self) -> tuple[bool, str]:
        missing = [robot for robot in ("xarm6", "ur5e") if robot not in self._paired_targets]
        if missing:
            return False, f"dual_robots missing target: {', '.join(missing)}"
        if not wait_for_joint_positions(self, ["xarm6", "ur5e"], timeout_sec=self.service_timeout_sec):
            return False, "no joint state for dual_robots"
        if not self.cartesian_client.wait_for_service(timeout_sec=self.service_timeout_sec):
            return False, "/compute_cartesian_path not available"

        responses: dict[str, object] = {}
        for robot in ("xarm6", "ur5e"):
            response, message = self._compute_robot_cartesian_path(robot, self._paired_targets[robot])
            if response is None:
                return False, message
            responses[robot] = response

        velocity_scale = self._normalize_velocity_scale(self.velocity_scale)
        for response in responses.values():
            self._scale_trajectory_timing(response.solution, 1.0 / velocity_scale)

        merged, message = self._merge_dual_robots_trajectories(
            responses["xarm6"].solution,
            responses["ur5e"].solution,
        )
        if merged is None:
            return False, message

        ok, message = self._validate_dual_robots_trajectory(merged)
        if not ok:
            self._paired_plan = None
            return False, message

        self._paired_plan = merged
        self._publish_dual_robots_preview(merged)
        duration = self._trajectory_duration(merged)
        return True, f"dual_robots plan ready ({duration:.2f}s)"

    def execute_dual_robots(self) -> tuple[bool, str]:
        if self._paired_plan is None:
            return False, "no dual_robots plan; use Plan dual_robots first"
        ok, message = self._ur5e_execution_health()
        if not ok:
            return False, message
        if not self.execute_client.wait_for_server(timeout_sec=self.service_timeout_sec):
            return False, "/execute_trajectory not available"

        goal = ExecuteTrajectory.Goal()
        goal.trajectory = self._paired_plan
        duration = self._trajectory_duration(self._paired_plan)
        point_count = len(list(self._paired_plan.joint_trajectory.points))
        joint_names = list(self._paired_plan.joint_trajectory.joint_names)
        future = self.execute_client.send_goal_async(goal)
        if not self._wait_future(future, timeout=10.0):
            return False, (
                f"/execute_trajectory send timeout; duration={duration:.2f}s; "
                f"points={point_count}; joints={','.join(joint_names)}"
            )
        handle = future.result()
        if handle is None or not handle.accepted:
            return False, (
                f"/execute_trajectory goal rejected; duration={duration:.2f}s; "
                f"points={point_count}; joints={','.join(joint_names)}"
            )
        result_future = handle.get_result_async()
        if not self._wait_future(result_future, timeout=max(30.0, duration + 15.0)):
            return False, (
                f"/execute_trajectory result timeout; duration={duration:.2f}s; "
                f"points={point_count}; joints={','.join(joint_names)}"
            )
        result = result_future.result()
        code = result.result.error_code.val if result else None
        if code == 1:
            return True, "dual_robots executed"
        return False, (
            f"/execute_trajectory error_code={code}; duration={duration:.2f}s; "
            f"points={point_count}; joints={','.join(joint_names)}"
        )

    def move_to_pose(self, robot: str, target: Pose, velocity_scale: float = 1.0) -> tuple[bool, str]:
        if robot not in ROBOTS:
            return False, f"unknown robot: {robot}"
        if not wait_for_joint_positions(self, [robot], timeout_sec=self.service_timeout_sec):
            return False, f"no joint state for {robot}"
        if not self.cartesian_client.wait_for_service(timeout_sec=self.service_timeout_sec):
            return False, "/compute_cartesian_path not available"
        if not self.execute_client.wait_for_server(timeout_sec=self.service_timeout_sec):
            return False, "/execute_trajectory not available"

        group_candidates = [self._current_group_name(robot)] + [
            group for group in self._group_name_candidates(robot)
            if group != self._current_group_name(robot)
        ]
        group_errors: list[str] = []
        response = None

        for group_name in group_candidates:
            request = GetCartesianPath.Request()
            request.header.frame_id = self._current_frame_id(robot)
            request.header.stamp = self.get_clock().now().to_msg()
            request.group_name = group_name
            request.link_name = self._current_ee_link(robot)
            request.waypoints = [copy.deepcopy(target)]
            request.max_step = self.cartesian_max_step_m
            request.jump_threshold = 0.0
            request.avoid_collisions = False
            request.start_state.is_diff = True

            future = self.cartesian_client.call_async(request)
            if not self._wait_future(future, timeout=10.0):
                group_errors.append(f"{group_name}: timeout")
                continue
            candidate = future.result()
            if candidate is None:
                group_errors.append(f"{group_name}: service failed")
                continue
            err_code = getattr(getattr(candidate, "error_code", None), "val", None)
            if err_code not in (None, 1):
                group_errors.append(f"{group_name}: error_code={err_code}")
                continue
            if float(candidate.fraction) <= 0.0:
                group_errors.append(f"{group_name}: no valid cartesian path")
                continue
            response = candidate
            self.active_group_name[robot] = group_name
            break

        if response is None:
            return False, "; ".join(group_errors[-2:]) if group_errors else "CartesianPath failed"
        if float(response.fraction) < 0.9:
            return False, f"Path incomplete ({float(response.fraction):.0%})"

        velocity_scale = self._normalize_velocity_scale(velocity_scale)
        self._scale_trajectory_timing(response.solution, 1.0 / velocity_scale)

        goal = ExecuteTrajectory.Goal()
        goal.trajectory = response.solution
        future = self.execute_client.send_goal_async(goal)
        if not self._wait_future(future, timeout=10.0):
            return False, "Execute send timeout"
        handle = future.result()
        if handle is None or not handle.accepted:
            return False, "Execute rejected"
        result_future = handle.get_result_async()
        if not self._wait_future(result_future, timeout=30.0):
            return False, "Execute timeout"
        result = result_future.result()
        code = result.result.error_code.val if result else None
        if code == 1:
            return True, "OK"
        return False, f"Error code {code}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Dual robots paired RViz drag markers")
    parser.add_argument("--mode", choices=("monitor", "teach"), default="monitor")
    parser.add_argument("--marker-scale", type=float, default=0.25)
    parser.add_argument("--velocity-scale", type=float, default=0.35)
    parser.add_argument("--cart-max-step-mm", type=float, default=30.0)
    parser.add_argument("--service-timeout-sec", type=float, default=8.0)
    parser.add_argument("--execution-policy", choices=("immediate", "paired"), default="immediate")
    parser.add_argument("--status-file", default="")
    args = parser.parse_args()

    rclpy.init()
    node = DualDragMarkers(
        mode=args.mode,
        marker_scale=args.marker_scale,
        velocity_scale=args.velocity_scale,
        cartesian_max_step_mm=args.cart_max_step_mm,
        service_timeout_sec=args.service_timeout_sec,
        execution_policy=args.execution_policy,
        status_file=args.status_file,
    )
    try:
        node.get_logger().info(
            f"dual drag markers ready mode={args.mode}; "
            f"execution-policy={args.execution_policy}; "
            "add/use RViz InteractiveMarkers display topic /dual_drag_markers/update"
        )
        rclpy.spin(node)
    finally:
        try:
            node.server.shutdown()
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
