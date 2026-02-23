"""
Low-level xArm6 controller over ROS2.

Placeholder — will be implemented when xArm6 ROS2 packages are set up.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class XArm6Controller:
    """Placeholder for xArm6 ROS2 controller. To be implemented."""

    def __init__(self):
        logger.info("[xArm6] Controller not yet implemented")

    def move_joints(self, positions: list[float], duration_sec: int = 2) -> bool:
        logger.info("[xArm6] Mock move_joints: %s", positions)
        return False

    def open_gripper(self) -> bool:
        return False

    def close_gripper(self) -> bool:
        return False

    def shutdown(self):
        pass
