#!/usr/bin/env python3
"""Small MoveIt hardware test for the real UR5e."""

from __future__ import annotations

import argparse
import math
import sys
import time
from typing import Any

import rclpy
from control_msgs.action import FollowJointTrajectory
from moveit_msgs.action import ExecuteTrajectory
from moveit_msgs.msg import RobotTrajectory
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

ARM_JOINTS = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]


class UR5eMoveitCommander(Node):
    def __init__(self) -> None:
        super().__init__("ur5e_moveit_commander")
        self.joint_state: JointState | None = None
        self.positions_by_name: dict[str, float] = {}

        self.create_subscription(JointState, "/joint_states", self._on_joint_state, 10)
        self.execute_client = ActionClient(self, ExecuteTrajectory, "/execute_trajectory")
        self.controller_client = ActionClient(
            self,
            FollowJointTrajectory,
            "/cais_ur5e_rtde_trajectory_controller/follow_joint_trajectory",
        )

    def _on_joint_state(self, msg: JointState) -> None:
        self.joint_state = msg
        for name, position in zip(msg.name, msg.position):
            self.positions_by_name[str(name)] = float(position)

    def wait_for_samples(self, timeout_sec: float = 3.0) -> None:
        deadline = time.monotonic() + float(timeout_sec)
        while rclpy.ok() and time.monotonic() < deadline:
            if self.joint_state is not None:
                return
            rclpy.spin_once(self, timeout_sec=0.1)

    def call_service(self, service_name: str, service_type: type, timeout_sec: float = 3.0) -> Any | None:
        client = self.create_client(service_type, service_name)
        if not client.wait_for_service(timeout_sec=timeout_sec):
            print(f"{service_name}: unavailable")
            return None
        future = client.call_async(service_type.Request())
        deadline = time.monotonic() + float(timeout_sec)
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
        if not all(joint in self.positions_by_name for joint in ARM_JOINTS):
            return None
        return [float(self.positions_by_name[joint]) for joint in ARM_JOINTS]

    def diagnose(self) -> bool:
        self.wait_for_samples(timeout_sec=3.0)

        execute_available = self.execute_client.wait_for_server(timeout_sec=1.0)
        controller_available = self.controller_client.wait_for_server(timeout_sec=1.0)
        arm_positions = self.arm_positions()

        print(f"arm_positions: {arm_positions!r}")
        print(f"execute_trajectory_available: {execute_available!r}")
        print(f"rtde_trajectory_action_available: {controller_available!r}")

        ready = (
            arm_positions is not None
            and execute_available
            and controller_available
        )
        print(f"moveit_execution_ready: {ready!r}")
        return bool(ready)

    def move_joint(self, joint: str, delta_rad: float, duration_sec: float) -> bool:
        current = self.arm_positions()
        if current is None:
            print("move_joint: no complete arm joint state")
            return False
        if joint not in ARM_JOINTS:
            print(f"move_joint: {joint} is not in {ARM_JOINTS}")
            return False
        if not self.execute_client.wait_for_server(timeout_sec=3.0):
            print("/execute_trajectory: unavailable")
            return False

        target = list(current)
        joint_index = ARM_JOINTS.index(joint)
        target[joint_index] += float(delta_rad)

        start_point = JointTrajectoryPoint()
        start_point.positions = [float(value) for value in current]
        start_point.velocities = [0.0] * len(ARM_JOINTS)
        start_point.time_from_start.sec = 0
        start_point.time_from_start.nanosec = 0

        target_point = JointTrajectoryPoint()
        target_point.positions = [float(value) for value in target]
        target_point.velocities = [0.0] * len(ARM_JOINTS)
        duration = max(0.1, float(duration_sec))
        sec = int(duration)
        target_point.time_from_start.sec = sec
        target_point.time_from_start.nanosec = int((duration - sec) * 1_000_000_000)

        trajectory = JointTrajectory()
        trajectory.joint_names = list(ARM_JOINTS)
        trajectory.points = [start_point, target_point]

        robot_trajectory = RobotTrajectory()
        robot_trajectory.joint_trajectory = trajectory

        goal = ExecuteTrajectory.Goal()
        goal.trajectory = robot_trajectory

        print(f"Sending MoveIt trajectory: {joint} += {delta_rad:.6f} rad ({math.degrees(delta_rad):.3f} deg)")
        print(f"Start joints:  {current}")
        print(f"Target joints: {target}")

        send_future = self.execute_client.send_goal_async(goal)
        deadline = time.monotonic() + 10.0
        while rclpy.ok() and time.monotonic() < deadline and not send_future.done():
            rclpy.spin_once(self, timeout_sec=0.1)
        if not send_future.done():
            print("MoveIt execute send timeout")
            return False
        goal_handle = send_future.result()
        if not goal_handle or not goal_handle.accepted:
            print("MoveIt execute goal rejected")
            return False

        result_future = goal_handle.get_result_async()
        deadline = time.monotonic() + max(8.0, duration + 8.0)
        while rclpy.ok() and time.monotonic() < deadline and not result_future.done():
            rclpy.spin_once(self, timeout_sec=0.1)
        if not result_future.done():
            print("MoveIt execute result timeout")
            return False

        wrapped_result = result_future.result()
        result = wrapped_result.result if wrapped_result else None
        code = result.error_code.val if result else None
        print(f"MoveIt action status: {getattr(wrapped_result, 'status', None)!r}")
        print(f"MoveIt error_code: {code!r}")

        time.sleep(0.2)
        rclpy.spin_once(self, timeout_sec=0.1)
        end = self.arm_positions()
        actual_delta = None if end is None else end[joint_index] - current[joint_index]
        print(f"End joints:    {end}")
        if actual_delta is not None:
            print(f"Actual {joint} delta: {actual_delta:.6f} rad ({math.degrees(actual_delta):.3f} deg)")
        observed = actual_delta is not None and abs(actual_delta) > abs(float(delta_rad)) * 0.5
        print(f"moveit_motion_observed: {observed!r}")
        return code == 1 and observed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnose-only", action="store_true")
    parser.add_argument("--joint", default="wrist_3_joint", choices=ARM_JOINTS)
    parser.add_argument("--delta-rad", type=float, default=0.02)
    parser.add_argument("--duration-sec", type=float, default=3.0)
    args = parser.parse_args(argv)

    rclpy.init()
    commander = UR5eMoveitCommander()
    try:
        ready = commander.diagnose()
        if args.diagnose_only:
            print("MoveIt motion: skipped because --diagnose-only was set")
            return 0 if ready else 2
        if not ready:
            print("MoveIt motion: blocked because moveit_execution_ready is false")
            return 2
        ok = commander.move_joint(args.joint, args.delta_rad, args.duration_sec)
        return 0 if ok else 3
    finally:
        commander.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    # Change these values when running this file directly from VS Code.
    # This is a joint-space MoveIt hardware test, so TEST_DELTA_RAD is radians, not mm.
    DIAGNOSE_ONLY = False
    TEST_JOINT = "wrist_3_joint"
    TEST_DELTA_RAD = 0.10
    TEST_DURATION_SEC = 5.0

    if len(sys.argv) > 1:
        raise SystemExit(main(sys.argv[1:]))

    default_args = [
        "--joint",
        TEST_JOINT,
        "--delta-rad",
        str(TEST_DELTA_RAD),
        "--duration-sec",
        str(TEST_DURATION_SEC),
    ]
    if DIAGNOSE_ONLY:
        default_args.append("--diagnose-only")
    exit_code = main(default_args)
    print(f"Moveit_controller.py finished with exit code {exit_code}")
