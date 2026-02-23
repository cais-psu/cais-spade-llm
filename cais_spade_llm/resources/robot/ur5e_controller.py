"""
Low-level UR5e controller over ROS2.

Connects to the UR5e joint trajectory controller in Gazebo or real hardware.
"""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Standard UR5e joint names used by the ROS2 driver and Gazebo simulation.
JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

# ROS2 topic names (from ur_simulation_gz)
TRAJECTORY_TOPIC = "/scaled_joint_trajectory_controller/joint_trajectory"
JOINT_STATES_TOPIC = "/joint_states"


class UR5eController:
    """UR5e low-level controller via ROS2."""

    def __init__(self):
        self._node = None
        self._traj_pub = None
        self._initialized = False

    def init(self) -> bool:
        """Initialize ROS2 node and trajectory publisher."""
        if self._initialized:
            return True

        import rclpy
        from trajectory_msgs.msg import JointTrajectory

        if not rclpy.ok():
            rclpy.init()

        self._node = rclpy.create_node("ur5e_controller")
        self._traj_pub = self._node.create_publisher(
            JointTrajectory, TRAJECTORY_TOPIC, 10
        )
        self._initialized = True
        logger.info("[UR5e] Initialized on %s", TRAJECTORY_TOPIC)
        return True

    def move_joints(self, positions: list[float], duration_sec: int = 2) -> bool:
        """
        Send joint trajectory command (non-blocking).

        Args:
            positions: 6 joint angles in radians.
            duration_sec: Time to reach target.
        """
        if not self._initialized:
            self.init()

        from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
        from builtin_interfaces.msg import Duration

        msg = JointTrajectory()
        msg.joint_names = list(JOINT_NAMES)

        point = JointTrajectoryPoint()
        point.positions = list(positions)
        point.time_from_start = Duration(sec=duration_sec)
        msg.points = [point]

        self._traj_pub.publish(msg)
        logger.info("[UR5e] Moving to %s", positions)
        return True

    def get_joint_positions(self) -> Optional[dict]:
        """Read current joint positions. TODO: implement."""
        return None

    def open_gripper(self) -> bool:
        """TODO: implement."""
        return False

    def close_gripper(self) -> bool:
        """TODO: implement."""
        return False

    def shutdown(self):
        """Clean up ROS2 resources."""
        if self._node:
            self._node.destroy_node()
            self._node = None
            self._initialized = False
