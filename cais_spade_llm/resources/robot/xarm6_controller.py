"""
xArm6 low-level controller over ROS2.

Provides:
- Legacy APIs: move_joints, open_gripper, close_gripper
- Phased pick/place APIs used by RobotAgent-style tool execution:
  pick_approach, pick_grasp, place_approach, place_insert, move_home
"""

from __future__ import annotations

import os
from typing import Any

from .ros2_pick_place_controller import Ros2PickPlaceController

JOINT_NAMES = [
    "xarm6_joint1",
    "xarm6_joint2",
    "xarm6_joint3",
    "xarm6_joint4",
    "xarm6_joint5",
    "xarm6_joint6",
]
JOINT_STATES_TOPIC = "/joint_states"


class XArm6Controller(Ros2PickPlaceController):
    """Config-driven xArm6 ROS2 controller."""

    def __init__(
        self,
        *,
        trajectory_topic: str | None = None,
        joint_states_topic: str = JOINT_STATES_TOPIC,
        controller_config: dict[str, Any] | None = None,
        named_positions: dict[str, Any] | None = None,
        execution_mode: str = "simulation",
    ) -> None:
        super().__init__(
            robot_name="xarm6",
            node_name=f"xarm6_controller_{os.getpid()}",
            controller_config=controller_config or {},
            named_positions=named_positions,
            execution_mode=execution_mode,
            arm_joint_names=JOINT_NAMES,
            arm_trajectory_topic=trajectory_topic,
            joint_states_topic=joint_states_topic,
        )
