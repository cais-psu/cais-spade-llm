#!/usr/bin/env python3
"""Bridge collision-aware Nav2 motion to the recovery-framework KMR base."""

from __future__ import annotations

import json
import math
import threading
import time
from pathlib import Path
from typing import Any, NamedTuple


BASE_STATE_JOINTS = (
    "KMR_base_x_joint",
    "KMR_base_y_joint",
    "KMR_base_yaw_joint",
)
ALLOWED_RESOURCES = ("Storage", "M1", "M2")


class DockRoute(NamedTuple):
    """One reversible route between two exact recovery resources."""

    resources: tuple[str, str]
    poses: tuple[tuple[float, float, float], ...]


def normalize_angle(value: float) -> float:
    """Return an angle in the closed-open interval [-pi, pi)."""

    return (float(value) + math.pi) % (2.0 * math.pi) - math.pi


def load_kmr_config(path: str | Path) -> tuple[dict[str, Any], tuple[DockRoute, ...]]:
    """Load and validate the KMR controller and route configuration."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    kmr = payload.get("KMR")
    if not isinstance(kmr, dict):
        raise ValueError("recovery configuration is missing KMR")
    configured_pairs = {
        tuple(str(value) for value in route)
        for route in kmr.get("predefined_routes", [])
    }
    routes: list[DockRoute] = []
    for route in kmr.get("predefined_route_waypoints", []):
        resources = tuple(str(value) for value in route.get("resources", []))
        if len(resources) != 2 or resources not in configured_pairs:
            raise ValueError(f"KMR route does not match predefined_routes: {resources}")
        if any(resource not in ALLOWED_RESOURCES for resource in resources):
            raise ValueError(f"KMR route contains an unsupported resource: {resources}")
        poses = tuple(tuple(float(value) for value in pose) for pose in route.get("poses", []))
        if len(poses) < 2 or any(len(pose) != 3 for pose in poses):
            raise ValueError(f"KMR route requires x/y/yaw poses: {resources}")
        if not all(math.isfinite(value) for pose in poses for value in pose):
            raise ValueError(f"KMR route contains a non-finite pose: {resources}")
        routes.append(DockRoute(resources=resources, poses=poses))
    if {route.resources for route in routes} != configured_pairs:
        raise ValueError("every predefined KMR route requires waypoint poses")
    return kmr, tuple(routes)


def route_for(
    routes: tuple[DockRoute, ...], source: str, target: str
) -> tuple[tuple[float, float, float], ...] | None:
    """Return the configured route in the requested direction."""

    for route in routes:
        if route.resources == (source, target):
            return route.poses
        if route.resources == (target, source):
            return tuple(reversed(route.poses))
    return None


def docking_poses(
    routes: tuple[DockRoute, ...],
    endpoints: dict[str, tuple[float, float, float]],
    source: str | None,
    target: str,
) -> tuple[tuple[float, float, float], ...] | None:
    """Return Nav2 goals for a permitted docking request.

    A KMR at a configured endpoint follows the corresponding route. From an
    arbitrary collision-free position, it may only return to Storage.
    """

    if target not in ALLOWED_RESOURCES or source == target:
        return None
    if source is None:
        storage_m1 = route_for(routes, "Storage", "M1")
        if target != "Storage" or storage_m1 is None:
            return None
        storage = storage_m1[0]
        m1 = storage_m1[-1]
        storage_approach = (
            (storage[0] + m1[0]) / 2.0,
            (storage[1] + m1[1]) / 2.0,
            storage[2],
        )
        return storage_approach, storage
    route = route_for(routes, source, target)
    return route[1:] if route is not None else None


def clamp_planar_velocity(
    x: float,
    y: float,
    angular: float,
    max_linear_speed: float,
    max_angular_speed: float,
    minimum_in_place_angular_speed: float = 0.0,
) -> tuple[float, float, float]:
    """Bound a holonomic Nav2 command without changing its direction."""

    magnitude = math.hypot(x, y)
    scale = min(1.0, float(max_linear_speed) / magnitude) if magnitude > 0.0 else 1.0
    bounded_x = float(x) * scale
    bounded_y = float(y) * scale
    bounded_angular = max(
        -float(max_angular_speed), min(float(max_angular_speed), float(angular))
    )
    if (
        math.hypot(bounded_x, bounded_y) <= 0.02
        and 0.0 < abs(bounded_angular) < float(minimum_in_place_angular_speed)
    ):
        bounded_angular = math.copysign(
            float(minimum_in_place_angular_speed), bounded_angular
        )
    return bounded_x, bounded_y, bounded_angular


def slew_planar_velocity(
    previous: tuple[float, float, float],
    requested: tuple[float, float, float],
    max_linear_delta: float,
    max_angular_delta: float,
) -> tuple[float, float, float]:
    """Limit planar command changes while retaining holonomic direction."""

    delta_x = requested[0] - previous[0]
    delta_y = requested[1] - previous[1]
    magnitude = math.hypot(delta_x, delta_y)
    scale = min(1.0, float(max_linear_delta) / magnitude) if magnitude > 0.0 else 1.0
    angular_delta = max(
        -float(max_angular_delta),
        min(float(max_angular_delta), requested[2] - previous[2]),
    )
    return (
        previous[0] + delta_x * scale,
        previous[1] + delta_y * scale,
        previous[2] + angular_delta,
    )


def base_state_stop_reason(
    now: float,
    odom_updated: float | None,
    odom_timeout: float,
    arm_updated: float | None,
    arm_timeout: float,
    arm_parked: bool,
) -> str | None:
    """Explain a failed base feedback gate, using monotonic wall-clock time.

    Gazebo may pause or run slowly. Feedback expiry must still stop the base
    without waiting for simulation time to advance.
    """

    if odom_updated is None or now - odom_updated > odom_timeout:
        return "KMR odometry is stale"
    if arm_updated is None or now - arm_updated > arm_timeout:
        return "KMR arm state is stale"
    if not arm_parked:
        return "KMR arm left its parked configuration"
    return None


def occupancy_grid_footprint_is_clear(
    data: tuple[int, ...] | list[int],
    width: int,
    height: int,
    resolution: float,
    origin_x: float,
    origin_y: float,
    x: float,
    y: float,
    yaw: float,
    half_length: float = 0.625,
    half_width: float = 0.39,
) -> bool:
    """Return whether the padded KMR footprint occupies known free cells."""

    if width <= 0 or height <= 0 or resolution <= 0.0:
        return False
    if len(data) != width * height:
        return False

    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    corners = tuple(
        (
            x + cosine * local_x - sine * local_y,
            y + sine * local_x + cosine * local_y,
        )
        for local_x in (-half_length, half_length)
        for local_y in (-half_width, half_width)
    )
    map_max_x = origin_x + width * resolution
    map_max_y = origin_y + height * resolution
    if any(
        corner_x < origin_x
        or corner_x > map_max_x
        or corner_y < origin_y
        or corner_y > map_max_y
        for corner_x, corner_y in corners
    ):
        return False

    cell_padding = 0.5 * resolution * (abs(cosine) + abs(sine))
    minimum_x = max(origin_x, min(corner[0] for corner in corners) - resolution / 2.0)
    maximum_x = min(map_max_x, max(corner[0] for corner in corners) + resolution / 2.0)
    minimum_y = max(origin_y, min(corner[1] for corner in corners) - resolution / 2.0)
    maximum_y = min(map_max_y, max(corner[1] for corner in corners) + resolution / 2.0)
    minimum_column = max(0, int(math.floor((minimum_x - origin_x) / resolution)))
    maximum_column = min(width - 1, int(math.floor((maximum_x - origin_x) / resolution)))
    minimum_row = max(0, int(math.floor((minimum_y - origin_y) / resolution)))
    maximum_row = min(height - 1, int(math.floor((maximum_y - origin_y) / resolution)))

    for row in range(minimum_row, maximum_row + 1):
        cell_y = origin_y + (row + 0.5) * resolution
        for column in range(minimum_column, maximum_column + 1):
            if data[row * width + column] < 0 or data[row * width + column] >= 65:
                cell_x = origin_x + (column + 0.5) * resolution
                delta_x = cell_x - x
                delta_y = cell_y - y
                local_x = cosine * delta_x + sine * delta_y
                local_y = -sine * delta_x + cosine * delta_y
                if (
                    abs(local_x) <= half_length + cell_padding
                    and abs(local_y) <= half_width + cell_padding
                ):
                    return False
    return True


def docking_velocity(
    pose: tuple[float, float, float],
    target: tuple[float, float, float],
    max_linear_speed: float,
    max_angular_speed: float,
    position_tolerance: float,
    yaw_tolerance: float,
) -> tuple[float, float, float, bool]:
    """Return a body-frame command for a straight holonomic final dock."""

    delta_x = target[0] - pose[0]
    delta_y = target[1] - pose[1]
    distance = math.hypot(delta_x, delta_y)
    yaw_error = normalize_angle(target[2] - pose[2])
    if distance <= position_tolerance and abs(yaw_error) <= yaw_tolerance:
        return 0.0, 0.0, 0.0, True

    angular = max(
        -max_angular_speed,
        min(max_angular_speed, 1.5 * yaw_error),
    )
    if abs(yaw_error) > 0.10 or distance <= position_tolerance:
        return 0.0, 0.0, angular, False

    speed = min(max_linear_speed, max(0.05, 1.5 * distance))
    world_x = speed * delta_x / distance
    world_y = speed * delta_y / distance
    cosine = math.cos(pose[2])
    sine = math.sin(pose[2])
    body_x = cosine * world_x + sine * world_y
    body_y = -sine * world_x + cosine * world_y
    return body_x, body_y, angular, False


def main() -> None:
    """Run the simulation-only KMR docking server and Nav2 command gate."""

    import rclpy
    from action_msgs.msg import GoalStatus, GoalStatusArray
    from builtin_interfaces.msg import Duration, Time
    from cais_lab_robotics.action import DockKMR
    from control_msgs.action import FollowJointTrajectory
    from geometry_msgs.msg import Pose2D, PoseStamped, Twist
    from nav2_msgs.action import FollowPath, NavigateToPose
    from nav_msgs.msg import OccupancyGrid, Odometry
    from rclpy.action import ActionClient, ActionServer, CancelResponse, GoalResponse
    from rclpy.callback_groups import ReentrantCallbackGroup
    from rclpy.clock import Clock, ClockType
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.node import Node
    from rclpy.qos import (
        DurabilityPolicy,
        QoSProfile,
        ReliabilityPolicy,
        qos_profile_sensor_data,
    )
    from sensor_msgs.msg import JointState
    from std_srvs.srv import Trigger
    from trajectory_msgs.msg import JointTrajectoryPoint

    class KMRBaseController(Node):
        """Gate Nav2 velocity and expose configured docking through Nav2."""

        def __init__(self) -> None:
            super().__init__("KMR_base_controller")
            self.declare_parameter("config_file", "")
            config_file = self.get_parameter("config_file").get_parameter_value().string_value
            if not config_file:
                raise RuntimeError("config_file is required")
            payload = json.loads(Path(config_file).read_text(encoding="utf-8"))
            self.kmr, self.routes = load_kmr_config(config_file)
            control = self.kmr["base_control"]
            self.max_linear_speed = float(control["linear_speed_mps"])
            self.max_angular_speed = float(control["angular_speed_radps"])
            self.max_linear_acceleration = float(control["linear_acceleration_mps2"])
            self.max_angular_acceleration = float(control["angular_acceleration_radps2"])
            self.docking_linear_speed = min(
                self.max_linear_speed,
                float(control["docking_linear_speed_mps"]),
            )
            self.docking_slow_distance = float(control["docking_slow_distance_m"])
            self.arm_parking_duration = float(control["arm_parking_duration_sec"])
            self.arm_hold_duration = float(control["arm_hold_duration_sec"])
            self.arm_parked_tolerance = float(control["arm_parked_tolerance_rad"])
            self.minimum_in_place_angular_speed = float(
                control["minimum_in_place_angular_speed_radps"]
            )
            self.position_tolerance = float(control["position_tolerance_m"])
            self.yaw_tolerance = float(control["yaw_tolerance_rad"])
            self.control_period = 1.0 / float(control["control_rate_hz"])
            self.odom_timeout = float(control["odometry_timeout_sec"])
            self.arm_state_timeout = float(control["arm_state_timeout_sec"])
            self.command_timeout = float(control["command_timeout_sec"])
            self.waypoint_timeout = float(control["waypoint_timeout_sec"])
            self._lock = threading.Lock()
            self._pose: tuple[float, float, float] | None = (
                float(self.kmr["initial_pose"][0]),
                float(self.kmr["initial_pose"][1]),
                float(self.kmr["initial_pose"][5]),
            )
            self._odom_monotonic: float | None = None
            self._odom_stamp: Time | None = None
            self._nav_command: Twist | None = None
            self._nav_command_monotonic: float | None = None
            self._navigation_active = False
            self._follow_path_active = False
            self._last_output = (0.0, 0.0, 0.0)
            self._last_hold_reason: str | None = None
            self._arm_positions: dict[str, float] = {}
            self._arm_state_monotonic: float | None = None
            self._initial_arm_parked = False
            self._arm_parking_in_progress = False
            self._arm_parking_command_succeeded = False
            self._active_goal = False
            self._cancel_base_motion_requested = False
            self._nav_goal_handle: Any | None = None
            self._occupancy_map: OccupancyGrid | None = None
            self.endpoints = {
                "Storage": (
                    float(self.kmr["initial_pose"][0]),
                    float(self.kmr["initial_pose"][1]),
                    float(self.kmr["initial_pose"][5]),
                ),
                **{
                    str(machine["resource_id"]): (
                        float(machine["KMR_docking_pose"][0]),
                        float(machine["KMR_docking_pose"][1]),
                        float(machine["KMR_docking_pose"][5]),
                    )
                    for machine in payload["machines"]
                },
            }
            callback_group = ReentrantCallbackGroup()
            self.cmd_pub = self.create_publisher(Twist, "/KMR/cmd_vel", 10)
            self.joint_pub = self.create_publisher(JointState, "/joint_states", 50)
            self.pose_pub = self.create_publisher(Pose2D, "/KMR/current_pose", 20)
            self.create_subscription(
                Odometry, "/KMR/odom", self._odom_cb, qos_profile_sensor_data,
                callback_group=callback_group,
            )
            self.create_subscription(
                JointState, "/KMR/joint_states", self._kmr_joint_state_cb,
                qos_profile_sensor_data, callback_group=callback_group,
            )
            self.create_subscription(
                Twist, "/KMR/nav_cmd_vel", self._nav_command_cb, 10,
                callback_group=callback_group,
            )
            self.create_subscription(
                GoalStatusArray,
                "/KMR/follow_path/_action/status",
                self._follow_path_status_cb,
                10,
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
            self.create_timer(
                self.control_period,
                self._control_tick,
                callback_group=callback_group,
                clock=Clock(clock_type=ClockType.STEADY_TIME),
            )
            self.nav_client = ActionClient(
                self, NavigateToPose, "/KMR/navigate_to_pose",
                callback_group=callback_group,
            )
            self.follow_path_client = ActionClient(
                self, FollowPath, "/KMR/follow_path",
                callback_group=callback_group,
            )
            self.arm_client = ActionClient(
                self,
                FollowJointTrajectory,
                f'{self.kmr["arm_controller"]}/follow_joint_trajectory',
                callback_group=callback_group,
            )
            self.arm_parking_timer = self.create_timer(
                0.5,
                self._ensure_initial_arm_parked,
                callback_group=callback_group,
                clock=Clock(clock_type=ClockType.STEADY_TIME),
            )
            self.dock_action_server = ActionServer(
                self, DockKMR, "/KMR/dock", execute_callback=self._execute_dock,
                goal_callback=self._dock_goal, cancel_callback=self._cancel,
                callback_group=callback_group,
            )
            self.navigation_action_server = ActionServer(
                self,
                NavigateToPose,
                "/KMR/validated_navigate_to_pose",
                execute_callback=self._execute_navigation,
                goal_callback=self._navigation_goal,
                cancel_callback=self._cancel,
                callback_group=callback_group,
            )
            self.rviz_navigation_action_server = ActionServer(
                self,
                NavigateToPose,
                "/navigate_to_pose",
                execute_callback=self._execute_navigation,
                goal_callback=self._navigation_goal,
                cancel_callback=self._cancel,
                callback_group=callback_group,
            )
            self.follow_path_action_server = ActionServer(
                self,
                FollowPath,
                "/KMR/validated_follow_path",
                execute_callback=self._execute_follow_path,
                goal_callback=self._follow_path_goal,
                cancel_callback=self._cancel,
                callback_group=callback_group,
            )
            self.create_service(
                Trigger,
                "/KMR/cancel_base_motion",
                self._cancel_base_motion,
                callback_group=callback_group,
            )

        @staticmethod
        def _yaw_from_quaternion(quaternion: Any) -> float:
            siny = 2.0 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y)
            cosy = 1.0 - 2.0 * (quaternion.y * quaternion.y + quaternion.z * quaternion.z)
            return math.atan2(siny, cosy)

        @staticmethod
        def _pose_stamped(pose: tuple[float, float, float]) -> PoseStamped:
            message = PoseStamped()
            message.header.frame_id = "world"
            message.pose.position.x = pose[0]
            message.pose.position.y = pose[1]
            message.pose.orientation.z = math.sin(pose[2] / 2.0)
            message.pose.orientation.w = math.cos(pose[2] / 2.0)
            return message

        def _odom_cb(self, message: Odometry) -> None:
            pose = message.pose.pose
            with self._lock:
                self._pose = (
                    float(pose.position.x), float(pose.position.y),
                    self._yaw_from_quaternion(pose.orientation),
                )
                self._odom_monotonic = time.monotonic()
                self._odom_stamp = message.header.stamp

        def _nav_command_cb(self, message: Twist) -> None:
            with self._lock:
                self._nav_command = message
                self._nav_command_monotonic = time.monotonic()

        def _map_cb(self, message: OccupancyGrid) -> None:
            self._occupancy_map = message

        def _follow_path_status_cb(self, message: GoalStatusArray) -> None:
            active_states = {
                GoalStatus.STATUS_ACCEPTED,
                GoalStatus.STATUS_EXECUTING,
            }
            with self._lock:
                self._follow_path_active = any(
                    status.status in active_states for status in message.status_list
                )

        def _control_tick(self) -> None:
            pose, odom_updated, command, command_updated, odom_stamp = self._snapshot()
            if pose is not None:
                state = JointState()
                # /clock can advance slower than odometry; preserve measurement time for TF.
                state.header.stamp = odom_stamp if odom_stamp is not None else self.get_clock().now().to_msg()
                state.name = list(BASE_STATE_JOINTS)
                state.position = list(pose)
                self.joint_pub.publish(state)
                if odom_updated is not None:
                    self.pose_pub.publish(Pose2D(x=pose[0], y=pose[1], theta=pose[2]))
            now = time.monotonic()
            if (
                not (self._navigation_active or self._follow_path_active)
                or self._cancel_base_motion_requested
            ):
                self._last_hold_reason = None
                self._stop()
                return
            reason = self._state_stop_reason()
            if not self._initial_arm_parked:
                reason = "KMR arm parking has not completed"
            if reason is None and (
                command is None or command_updated is None
                or now - command_updated > self.command_timeout
            ):
                reason = "KMR velocity command is stale"
            if reason is not None:
                if reason != self._last_hold_reason:
                    self.get_logger().warning(f"KMR base held: {reason}")
                self._last_hold_reason = reason
                self._stop()
                return
            if self._last_hold_reason is not None:
                self.get_logger().info("KMR base feedback and velocity command are fresh; resuming")
                self._last_hold_reason = None
            x, y, angular = clamp_planar_velocity(
                command.linear.x, command.linear.y, command.angular.z,
                self.max_linear_speed, self.max_angular_speed,
                self.minimum_in_place_angular_speed,
            )
            x, y, angular = slew_planar_velocity(
                self._last_output,
                (x, y, angular),
                self.max_linear_acceleration * self.control_period,
                self.max_angular_acceleration * self.control_period,
            )
            self._last_output = (x, y, angular)
            gated = Twist()
            gated.linear.x = x
            gated.linear.y = y
            gated.angular.z = angular
            self.cmd_pub.publish(gated)

        def _kmr_joint_state_cb(self, message: JointState) -> None:
            for name, position in zip(message.name, message.position):
                self._arm_positions[str(name)] = float(position)
            self._arm_state_monotonic = time.monotonic()
            bridged = JointState()
            bridged.header = message.header
            bridged.name = list(message.name)
            bridged.position = list(message.position)
            bridged.velocity = list(message.velocity)
            bridged.effort = list(message.effort)
            self.joint_pub.publish(bridged)

        def _snapshot(
            self,
        ) -> tuple[tuple[float, float, float] | None, float | None, Twist | None, float | None, Time | None]:
            with self._lock:
                return (
                    self._pose, self._odom_monotonic, self._nav_command,
                    self._nav_command_monotonic, self._odom_stamp,
                )

        def _fresh_pose(self) -> tuple[float, float, float] | None:
            pose, updated, _, _, _ = self._snapshot()
            if pose is None or updated is None or time.monotonic() - updated > self.odom_timeout:
                return None
            return pose

        def _resource_at(self, pose: tuple[float, float, float]) -> str | None:
            for resource, endpoint in self.endpoints.items():
                if (
                    math.dist(pose[:2], endpoint[:2]) <= self.position_tolerance
                    and abs(normalize_angle(pose[2] - endpoint[2])) <= self.yaw_tolerance
                ):
                    return resource
            return None

        def _arm_state_is_fresh(self) -> bool:
            """Return whether KMR arm feedback is recent enough for base motion."""

            return not (
                self._arm_state_monotonic is None
                or time.monotonic() - self._arm_state_monotonic > self.arm_state_timeout
            )

        def _arm_is_parked(self) -> bool:
            if not self._arm_state_is_fresh():
                return False
            expected = dict(zip(self.kmr["arm_joint_names"], self.kmr["parked_arm_configuration"]))
            expected[self.kmr["gripper_joint"]] = float(self.kmr["gripper_stroke_m"])
            return all(
                name in self._arm_positions
                and abs(self._arm_positions[name] - float(position))
                <= self.arm_parked_tolerance
                for name, position in expected.items()
            )

        def _state_stop_reason(self) -> str | None:
            return base_state_stop_reason(
                time.monotonic(),
                self._odom_monotonic,
                self.odom_timeout,
                self._arm_state_monotonic,
                self.arm_state_timeout,
                self._arm_is_parked(),
            )

        def _ensure_initial_arm_parked(self) -> None:
            """Command and confirm the configured upright iiwa pose once."""

            if self._initial_arm_parked:
                self.arm_parking_timer.cancel()
                return
            if self._arm_parking_in_progress:
                if self._arm_is_parked():
                    self._initial_arm_parked = True
                    self.get_logger().info(
                        "KMR arm is holding the configured parked configuration"
                    )
                    self.arm_parking_timer.cancel()
                return
            if self._arm_parking_command_succeeded:
                if self._arm_is_parked():
                    self._initial_arm_parked = True
                    self.get_logger().info(
                        "KMR arm is holding the configured parked configuration"
                    )
                    self.arm_parking_timer.cancel()
                return
            if (
                not self._arm_state_is_fresh()
                or not self.arm_client.server_is_ready()
            ):
                return

            goal = FollowJointTrajectory.Goal()
            goal.trajectory.joint_names = list(self.kmr["arm_joint_names"])
            point = JointTrajectoryPoint()
            point.positions = [
                float(value) for value in self.kmr["parked_arm_configuration"]
            ]
            point.velocities = [0.0] * len(point.positions)
            duration_sec = int(self.arm_parking_duration)
            point.time_from_start = Duration(
                sec=duration_sec,
                nanosec=int((self.arm_parking_duration - duration_sec) * 1_000_000_000),
            )
            hold_point = JointTrajectoryPoint()
            hold_point.positions = list(point.positions)
            hold_point.velocities = list(point.velocities)
            hold_duration_sec = int(self.arm_hold_duration)
            hold_point.time_from_start = Duration(
                sec=hold_duration_sec,
                nanosec=int((self.arm_hold_duration - hold_duration_sec) * 1_000_000_000),
            )
            goal.trajectory.points = [point, hold_point]
            self._arm_parking_in_progress = True
            future = self.arm_client.send_goal_async(goal)
            future.add_done_callback(self._arm_parking_goal_response)

        def _arm_parking_goal_response(self, future: Any) -> None:
            try:
                goal_handle = future.result()
            except Exception as exc:
                self._arm_parking_in_progress = False
                self.get_logger().warning(f"KMR arm parking request failed: {exc}")
                return
            if not goal_handle.accepted:
                self._arm_parking_in_progress = False
                self.get_logger().warning("KMR arm controller rejected the parking request")
                return
            result_future = goal_handle.get_result_async()
            result_future.add_done_callback(self._arm_parking_result)

        def _arm_parking_result(self, future: Any) -> None:
            self._arm_parking_in_progress = False
            try:
                wrapped = future.result()
            except Exception as exc:
                self.get_logger().warning(f"KMR arm parking result failed: {exc}")
                return
            if wrapped.status != GoalStatus.STATUS_SUCCEEDED:
                self.get_logger().warning("KMR arm failed to reach its parked configuration")
                return
            self._arm_parking_command_succeeded = True

        def _claim_base_action(self, client_ready: bool) -> bool:
            """Reserve the single KMR base action slot after common checks."""

            pose = self._fresh_pose()
            if (
                pose is None
                or not self._initial_arm_parked
                or not self._arm_is_parked()
                or not client_ready
            ):
                return False
            with self._lock:
                if self._active_goal or self._follow_path_active:
                    return False
                self._active_goal = True
                self._cancel_base_motion_requested = False
                self._nav_command = None
                self._nav_command_monotonic = None
            return True

        def _release_base_action(self) -> None:
            with self._lock:
                self._active_goal = False
                # Cancellation stays latched until a new validated goal claims the base.
                self._nav_command = None
                self._nav_command_monotonic = None

        def _dock_goal(self, request: DockKMR.Goal) -> GoalResponse:
            target = str(request.target_resource)
            if target not in ALLOWED_RESOURCES:
                return GoalResponse.REJECT
            pose = self._fresh_pose()
            if pose is None:
                return GoalResponse.REJECT
            source = self._resource_at(pose)
            if docking_poses(self.routes, self.endpoints, source, target) is None:
                return GoalResponse.REJECT
            client_ready = source is not None or self.nav_client.server_is_ready()
            if not self._claim_base_action(client_ready):
                return GoalResponse.REJECT
            return GoalResponse.ACCEPT

        def _navigation_target_is_clear(self, request: NavigateToPose.Goal) -> bool:
            occupancy_map = self._occupancy_map
            if occupancy_map is None or request.pose.header.frame_id != "world":
                return False
            orientation = request.pose.pose.orientation
            yaw = self._yaw_from_quaternion(orientation)
            map_info = occupancy_map.info
            return occupancy_grid_footprint_is_clear(
                occupancy_map.data,
                map_info.width,
                map_info.height,
                map_info.resolution,
                map_info.origin.position.x,
                map_info.origin.position.y,
                request.pose.pose.position.x,
                request.pose.pose.position.y,
                yaw,
            )

        def _navigation_goal(self, request: NavigateToPose.Goal) -> GoalResponse:
            if not self._navigation_target_is_clear(request):
                self._stop()
                return GoalResponse.REJECT
            if not self._claim_base_action(self.nav_client.server_is_ready()):
                self._stop()
                return GoalResponse.REJECT
            return GoalResponse.ACCEPT

        def _follow_path_goal(self, _request: FollowPath.Goal) -> GoalResponse:
            if not self._claim_base_action(self.follow_path_client.server_is_ready()):
                self._stop()
                return GoalResponse.REJECT
            return GoalResponse.ACCEPT

        def _cancel(self, _goal_handle: Any) -> CancelResponse:
            self._cancel_base_motion_requested = True
            if self._nav_goal_handle is not None:
                self._nav_goal_handle.cancel_goal_async()
            self._navigation_active = False
            self._stop()
            return CancelResponse.ACCEPT

        def _cancel_base_motion(self, _request: Any, response: Any) -> Any:
            self._cancel_base_motion_requested = True
            if self._nav_goal_handle is not None:
                self._nav_goal_handle.cancel_goal_async()
            self._navigation_active = False
            self._stop()
            response.success = True
            response.message = "KMR base cancellation requested; zero velocity published"
            return response

        def _stop(self) -> None:
            self._last_output = (0.0, 0.0, 0.0)
            if self.context.ok():
                # A safety output must not replace or refresh the received command.
                self.cmd_pub.publish(Twist())

        @staticmethod
        def _result(success: bool, final_resource: str, message: str) -> DockKMR.Result:
            result = DockKMR.Result()
            result.success = success
            result.final_resource = final_resource
            result.message = message
            return result

        async def _execute_navigation(self, goal_handle: Any) -> NavigateToPose.Result:
            """Forward an RViz Nav2 Goal only while recovery gates remain valid."""

            try:
                self._navigation_active = True

                def feedback_callback(message: Any) -> None:
                    goal_handle.publish_feedback(message.feedback)

                nav_handle = await self.nav_client.send_goal_async(
                    goal_handle.request,
                    feedback_callback=feedback_callback,
                )
                self._nav_goal_handle = nav_handle
                if not nav_handle.accepted:
                    goal_handle.abort()
                    return NavigateToPose.Result()
                result_future = nav_handle.get_result_async()
                while not result_future.done():
                    canceled = (
                        goal_handle.is_cancel_requested
                        or self._cancel_base_motion_requested
                    )
                    if canceled:
                        await nav_handle.cancel_goal_async()
                        if goal_handle.is_cancel_requested:
                            goal_handle.canceled()
                        else:
                            goal_handle.abort()
                        return NavigateToPose.Result()
                    reason = self._state_stop_reason()
                    if reason is not None:
                        self.get_logger().error(f"KMR navigation aborted: {reason}")
                        self._cancel_base_motion_requested = True
                        self._stop()
                        await nav_handle.cancel_goal_async()
                        goal_handle.abort()
                        return NavigateToPose.Result()
                    time.sleep(0.05)
                wrapped = result_future.result()
                if wrapped.status == GoalStatus.STATUS_SUCCEEDED:
                    goal_handle.succeed()
                elif wrapped.status == GoalStatus.STATUS_CANCELED:
                    goal_handle.canceled()
                else:
                    goal_handle.abort()
                return wrapped.result
            finally:
                self._navigation_active = False
                self._nav_goal_handle = None
                self._release_base_action()
                self._stop()

        async def _execute_follow_path(self, goal_handle: Any) -> FollowPath.Result:
            """Forward a reviewed blue-marker path through the same base gates."""

            try:
                self._navigation_active = True

                def feedback_callback(message: Any) -> None:
                    goal_handle.publish_feedback(message.feedback)

                nav_handle = await self.follow_path_client.send_goal_async(
                    goal_handle.request,
                    feedback_callback=feedback_callback,
                )
                self._nav_goal_handle = nav_handle
                if not nav_handle.accepted:
                    goal_handle.abort()
                    return FollowPath.Result()
                result_future = nav_handle.get_result_async()
                while not result_future.done():
                    canceled = (
                        goal_handle.is_cancel_requested
                        or self._cancel_base_motion_requested
                    )
                    if canceled:
                        await nav_handle.cancel_goal_async()
                        if goal_handle.is_cancel_requested:
                            goal_handle.canceled()
                        else:
                            goal_handle.abort()
                        return FollowPath.Result()
                    reason = self._state_stop_reason()
                    if reason is not None:
                        self.get_logger().error(f"KMR path execution aborted: {reason}")
                        self._cancel_base_motion_requested = True
                        self._stop()
                        await nav_handle.cancel_goal_async()
                        goal_handle.abort()
                        return FollowPath.Result()
                    time.sleep(0.05)
                wrapped = result_future.result()
                if wrapped.status == GoalStatus.STATUS_SUCCEEDED:
                    goal_handle.succeed()
                elif wrapped.status == GoalStatus.STATUS_CANCELED:
                    goal_handle.canceled()
                else:
                    goal_handle.abort()
                return wrapped.result
            finally:
                self._navigation_active = False
                self._nav_goal_handle = None
                self._release_base_action()
                self._stop()

        async def _execute_dock(self, goal_handle: Any) -> DockKMR.Result:
            pose = self._fresh_pose()
            target = str(goal_handle.request.target_resource)
            source = self._resource_at(pose) if pose is not None else None
            goals = docking_poses(self.routes, self.endpoints, source, target)
            if pose is None or goals is None:
                self._active_goal = False
                goal_handle.abort()
                return self._result(False, source or "", "KMR is not at a permitted route state")

            source_name = source or "arbitrary"

            try:
                nav_goals = goals[:-1] if source is None else ()
                direct_goals = goals[-1:] if source is None else goals
                if nav_goals and not self.nav_client.wait_for_server(timeout_sec=2.0):
                    goal_handle.abort()
                    return self._result(False, source or "", "KMR Nav2 is unavailable")
                self._navigation_active = True
                for waypoint_index, waypoint in enumerate(nav_goals):
                    nav_goal = NavigateToPose.Goal()
                    nav_goal.pose = self._pose_stamped(waypoint)

                    def feedback_callback(
                        message: Any, current_waypoint: int = waypoint_index,
                    ) -> None:
                        feedback = message.feedback
                        current = feedback.current_pose.pose
                        current_yaw = self._yaw_from_quaternion(current.orientation)
                        dock_feedback = DockKMR.Feedback()
                        dock_feedback.active_route_from = source_name
                        dock_feedback.active_route_to = target
                        dock_feedback.waypoint_index = current_waypoint
                        dock_feedback.current_pose = Pose2D(
                            x=current.position.x,
                            y=current.position.y,
                            theta=current_yaw,
                        )
                        dock_feedback.remaining_distance_m = float(
                            feedback.distance_remaining
                        )
                        dock_feedback.remaining_yaw_rad = abs(
                            normalize_angle(waypoint[2] - current_yaw)
                        )
                        goal_handle.publish_feedback(dock_feedback)

                    nav_handle = await self.nav_client.send_goal_async(
                        nav_goal, feedback_callback=feedback_callback,
                    )
                    self._nav_goal_handle = nav_handle
                    if not nav_handle.accepted:
                        goal_handle.abort()
                        return self._result(
                            False, source or "", "KMR Nav2 rejected the route"
                        )
                    nav_result_future = nav_handle.get_result_async()
                    while not nav_result_future.done():
                        if goal_handle.is_cancel_requested or self._cancel_base_motion_requested:
                            await nav_handle.cancel_goal_async()
                            if goal_handle.is_cancel_requested:
                                goal_handle.canceled()
                            else:
                                goal_handle.abort()
                            return self._result(
                                False,
                                self._resource_at(self._fresh_pose() or pose) or "",
                                "KMR docking canceled",
                            )
                        if self._fresh_pose() is None:
                            await nav_handle.cancel_goal_async()
                            goal_handle.abort()
                            return self._result(False, "", "KMR odometry became stale")
                        if not self._arm_state_is_fresh():
                            await nav_handle.cancel_goal_async()
                            goal_handle.abort()
                            return self._result(False, "", "KMR arm state became stale")
                        if not self._arm_is_parked():
                            await nav_handle.cancel_goal_async()
                            goal_handle.abort()
                            return self._result(
                                False, "", "KMR arm left its parked configuration"
                            )
                        time.sleep(0.05)
                    wrapped = nav_result_future.result()
                    if wrapped.status != GoalStatus.STATUS_SUCCEEDED:
                        goal_handle.abort()
                        return self._result(
                            False, "", "KMR Nav2 route failed or was canceled"
                        )
                self._nav_goal_handle = None
                for direct_offset, direct_target in enumerate(direct_goals):
                    direct_index = len(nav_goals) + direct_offset
                    final_direct_goal = direct_offset == len(direct_goals) - 1
                    docking_started = time.monotonic()
                    while True:
                        if goal_handle.is_cancel_requested or self._cancel_base_motion_requested:
                            self._navigation_active = False
                            self._stop()
                            if goal_handle.is_cancel_requested:
                                goal_handle.canceled()
                            else:
                                goal_handle.abort()
                            return self._result(
                                False,
                                self._resource_at(self._fresh_pose() or pose) or "",
                                "KMR docking canceled",
                            )
                        final_pose = self._fresh_pose()
                        if final_pose is None:
                            goal_handle.abort()
                            return self._result(False, "", "KMR odometry became stale")
                        if not self._arm_state_is_fresh():
                            goal_handle.abort()
                            return self._result(False, "", "KMR arm state became stale")
                        if not self._arm_is_parked():
                            goal_handle.abort()
                            return self._result(
                                False, "", "KMR arm left its parked configuration"
                            )
                        if time.monotonic() - docking_started > self.waypoint_timeout:
                            goal_handle.abort()
                            return self._result(
                                False, "", "KMR deterministic route motion timed out"
                            )
                        remaining_distance = math.dist(
                            final_pose[:2], direct_target[:2]
                        )
                        linear_limit = (
                            self.docking_linear_speed
                            if (
                                final_direct_goal
                                and remaining_distance <= self.docking_slow_distance
                            )
                            else self.max_linear_speed
                        )
                        position_tolerance = (
                            self.position_tolerance if final_direct_goal else 0.06
                        )
                        yaw_tolerance = (
                            self.yaw_tolerance if final_direct_goal else 0.05
                        )
                        x, y, angular, arrived = docking_velocity(
                            final_pose,
                            direct_target,
                            linear_limit,
                            self.max_angular_speed,
                            position_tolerance,
                            yaw_tolerance,
                        )
                        dock_feedback = DockKMR.Feedback()
                        dock_feedback.active_route_from = source_name
                        dock_feedback.active_route_to = target
                        dock_feedback.waypoint_index = direct_index
                        dock_feedback.current_pose = Pose2D(
                            x=final_pose[0], y=final_pose[1], theta=final_pose[2]
                        )
                        dock_feedback.remaining_distance_m = remaining_distance
                        dock_feedback.remaining_yaw_rad = abs(
                            normalize_angle(direct_target[2] - final_pose[2])
                        )
                        goal_handle.publish_feedback(dock_feedback)
                        if arrived:
                            break
                        command = Twist()
                        command.linear.x = x
                        command.linear.y = y
                        command.angular.z = angular
                        with self._lock:
                            self._nav_command = command
                            self._nav_command_monotonic = time.monotonic()
                        time.sleep(self.control_period)

                self._navigation_active = False
                self._stop()
                final_resource = self._resource_at(final_pose)
                if final_resource != target:
                    goal_handle.abort()
                    return self._result(
                        False, final_resource or "",
                        "KMR stopped outside the requested docking tolerance",
                    )
                goal_handle.succeed()
                return self._result(True, target, f"KMR docked at {target}")
            finally:
                self._navigation_active = False
                self._nav_goal_handle = None
                self._release_base_action()
                self._stop()

        def destroy_node(self) -> bool:
            self._navigation_active = False
            self._stop()
            self.dock_action_server.destroy()
            self.navigation_action_server.destroy()
            self.rviz_navigation_action_server.destroy()
            self.follow_path_action_server.destroy()
            return super().destroy_node()

    rclpy.init()
    node = KMRBaseController()
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
