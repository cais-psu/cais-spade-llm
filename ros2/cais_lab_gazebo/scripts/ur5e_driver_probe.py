#!/usr/bin/env python3
"""Probe UR5e hardware control through ur_robot_driver."""

from __future__ import annotations

import argparse
import math
import sys
import time
from typing import Any

import rclpy
from control_msgs.action import FollowJointTrajectory
from controller_manager_msgs.srv import ListControllers
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64
from trajectory_msgs.msg import JointTrajectoryPoint
from ur_dashboard_msgs.srv import GetProgramState, GetRobotMode, GetSafetyMode, IsProgramRunning


ARM_JOINTS = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]


class UR5eDriverProbe(Node):
    def __init__(self) -> None:
        super().__init__("ur5e_driver_probe")
        self.joint_state: JointState | None = None
        self.speed_scaling: float | None = None
        self.create_subscription(JointState, "/joint_states", self._on_joint_state, 10)
        self.create_subscription(
            Float64,
            "/speed_scaling_state_broadcaster/speed_scaling",
            self._on_speed_scaling,
            10,
        )
        self.trajectory_action = ActionClient(
            self,
            FollowJointTrajectory,
            "/scaled_joint_trajectory_controller/follow_joint_trajectory",
        )

    def _on_joint_state(self, msg: JointState) -> None:
        self.joint_state = msg

    def _on_speed_scaling(self, msg: Float64) -> None:
        self.speed_scaling = float(msg.data)

    def wait_for_samples(self, timeout_sec: float) -> None:
        deadline = time.monotonic() + timeout_sec
        while rclpy.ok() and time.monotonic() < deadline:
            if self.joint_state is not None and self.speed_scaling is not None:
                return
            rclpy.spin_once(self, timeout_sec=0.1)

    def call_service(self, service_name: str, service_type: type, timeout_sec: float = 3.0) -> Any | None:
        client = self.create_client(service_type, service_name)
        if not client.wait_for_service(timeout_sec=timeout_sec):
            print(f"{service_name}: unavailable")
            return None
        future = client.call_async(service_type.Request())
        deadline = time.monotonic() + timeout_sec
        while rclpy.ok() and time.monotonic() < deadline and not future.done():
            rclpy.spin_once(self, timeout_sec=0.1)
        if not future.done():
            print(f"{service_name}: timeout")
            return None
        try:
            return future.result()
        except Exception as exc:
            print(f"{service_name}: error: {exc}")
            return None

    def arm_positions(self) -> list[float] | None:
        msg = self.joint_state
        if msg is None:
            return None
        positions_by_name = dict(zip(msg.name, msg.position))
        if not all(joint in positions_by_name for joint in ARM_JOINTS):
            return None
        return [float(positions_by_name[joint]) for joint in ARM_JOINTS]

    def send_small_motion(self, joint: str, delta_rad: float, duration_sec: float) -> bool:
        current = self.arm_positions()
        if current is None:
            print("small motion: no complete arm joint state")
            return False
        if joint not in ARM_JOINTS:
            print(f"small motion: {joint} is not in {ARM_JOINTS}")
            return False
        if not self.trajectory_action.wait_for_server(timeout_sec=3.0):
            print("/scaled_joint_trajectory_controller/follow_joint_trajectory: unavailable")
            return False

        target = list(current)
        joint_index = ARM_JOINTS.index(joint)
        target[joint_index] += float(delta_rad)

        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = list(ARM_JOINTS)
        point = JointTrajectoryPoint()
        point.positions = target
        point.velocities = [0.0] * len(ARM_JOINTS)
        point.time_from_start.sec = int(duration_sec)
        point.time_from_start.nanosec = int((duration_sec - int(duration_sec)) * 1_000_000_000)
        goal.trajectory.points = [point]

        print(f"Sending small trajectory: {joint} += {delta_rad:.6f} rad ({math.degrees(delta_rad):.3f} deg)")
        print(f"Start joints:  {current}")
        print(f"Target joints: {target}")
        send_future = self.trajectory_action.send_goal_async(goal)
        while rclpy.ok() and not send_future.done():
            rclpy.spin_once(self, timeout_sec=0.1)
        goal_handle = send_future.result()
        if not goal_handle or not goal_handle.accepted:
            print("small motion: action goal rejected")
            return False

        result_future = goal_handle.get_result_async()
        deadline = time.monotonic() + max(5.0, duration_sec + 5.0)
        while rclpy.ok() and time.monotonic() < deadline and not result_future.done():
            rclpy.spin_once(self, timeout_sec=0.1)
        if not result_future.done():
            print("small motion: action result timeout")
            return False

        result = result_future.result().result
        time.sleep(0.2)
        rclpy.spin_once(self, timeout_sec=0.1)
        end = self.arm_positions()
        actual_delta = None if end is None else end[joint_index] - current[joint_index]
        print(f"Action error_code: {result.error_code}")
        print(f"Action error_string: {result.error_string!r}")
        print(f"End joints:    {end}")
        if actual_delta is not None:
            print(f"Actual {joint} delta: {actual_delta:.6f} rad ({math.degrees(actual_delta):.3f} deg)")
        return result.error_code == 0 and actual_delta is not None and abs(actual_delta) > abs(delta_rad) * 0.5


def print_controller_summary(response: Any | None) -> None:
    if response is None:
        return
    print("controllers:")
    for controller in response.controller:
        if controller.name in {
            "joint_state_broadcaster",
            "scaled_joint_trajectory_controller",
            "joint_trajectory_controller",
            "io_and_status_controller",
            "speed_scaling_state_broadcaster",
        }:
            print(f"  {controller.name}: {controller.state} ({controller.type})")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute-small-motion", action="store_true")
    parser.add_argument("--joint", default="wrist_3_joint", choices=ARM_JOINTS)
    parser.add_argument("--delta-rad", type=float, default=0.02)
    parser.add_argument("--duration-sec", type=float, default=3.0)
    args = parser.parse_args(argv)

    rclpy.init()
    node = UR5eDriverProbe()
    try:
        node.wait_for_samples(timeout_sec=3.0)

        print_controller_summary(
            node.call_service("/controller_manager/list_controllers", ListControllers)
        )
        program_state = node.call_service("/dashboard_client/program_state", GetProgramState)
        program_running = node.call_service("/dashboard_client/program_running", IsProgramRunning)
        robot_mode = node.call_service("/dashboard_client/get_robot_mode", GetRobotMode)
        safety_mode = node.call_service("/dashboard_client/get_safety_mode", GetSafetyMode)

        print(f"program_state: {getattr(program_state, 'answer', None)!r}")
        print(f"program_running: {getattr(program_running, 'program_running', None)!r}")
        print(f"robot_mode: {getattr(robot_mode, 'robot_mode', None)!r}")
        print(f"safety_mode: {getattr(safety_mode, 'safety_mode', None)!r}")
        print(f"speed_scaling: {node.speed_scaling!r}")
        print(f"arm_positions: {node.arm_positions()!r}")
        print(
            "trajectory_action_available: "
            f"{node.trajectory_action.wait_for_server(timeout_sec=1.0)!r}"
        )

        ready = (
            getattr(program_running, "program_running", False)
            and node.speed_scaling is not None
            and node.speed_scaling > 0.0
            and node.arm_positions() is not None
            and node.trajectory_action.wait_for_server(timeout_sec=0.1)
        )
        print(f"driver_execution_ready: {ready!r}")

        if not args.execute_small_motion:
            print("small motion: skipped; pass --execute-small-motion to command the arm")
            return 0 if ready else 2

        if not ready:
            print("small motion: blocked because driver_execution_ready is false")
            return 2
        ok = node.send_small_motion(args.joint, args.delta_rad, args.duration_sec)
        print(f"small_motion_observed: {ok!r}")
        return 0 if ok else 3
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
