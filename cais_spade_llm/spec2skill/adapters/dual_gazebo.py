"""Narrow runtime adapter for the Spec2Skill no-hardware dual Gazebo UI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

DUAL_GAZEBO_NAME = "gazebo_dual_spec2skill"
_HARDWARE_STACK_NAMES = ("xarm6", "ur5e", "dual robots")
_HARDWARE_BLOCKED_REASON = "Blocked: hardware stack is running. Stop hardware first."


class DualGazeboRuntime(Protocol):
    """Runtime operations used by the isolated Spec2Skill Gazebo adapter."""

    def ros2_all_statuses(self) -> dict[str, str]:
        """Return tracked ROS2 process statuses."""
        ...

    def hardware_stack_status(self, robot: str) -> dict[str, object]:
        """Return one hardware stack's current status."""
        ...

    def ros2_start(self, name: str) -> str | None:
        """Start one tracked ROS2 process and return an error when blocked."""
        ...

    def ros2_stop(self, name: str) -> None:
        """Stop one tracked ROS2 process."""
        ...


@dataclass(frozen=True)
class DualGazeboStatus:
    """Current dual Gazebo state and any hardware interlock message."""

    state: str
    blocked_reason: str | None = None


def read_dual_gazebo_status(runtime: DualGazeboRuntime) -> DualGazeboStatus:
    """Read fresh dual Gazebo and hardware-stack state from the runtime."""
    statuses = runtime.ros2_all_statuses()
    state = str(statuses.get(DUAL_GAZEBO_NAME) or "stopped")
    if state == "running":
        return DualGazeboStatus(state=state)

    for robot in _HARDWARE_STACK_NAMES:
        hardware_status = runtime.hardware_stack_status(robot)
        if hardware_status.get("overall") == "running":
            return DualGazeboStatus(
                state=state,
                blocked_reason=_HARDWARE_BLOCKED_REASON,
            )
    return DualGazeboStatus(state=state)


def start_dual_gazebo(runtime: DualGazeboRuntime) -> str | None:
    """Start `gazebo_dual_spec2skill` after a fresh state and hardware check."""
    status = read_dual_gazebo_status(runtime)
    if status.blocked_reason:
        return status.blocked_reason
    if status.state == "running":
        return "Dual Robots (xArm6 + UR5e) is already running."
    if status.state != "stopped":
        return f"Dual Robots (xArm6 + UR5e) cannot start while status is {status.state}."
    return runtime.ros2_start(DUAL_GAZEBO_NAME)


def stop_dual_gazebo(runtime: DualGazeboRuntime) -> None:
    """Stop only the tracked `gazebo_dual_spec2skill` process."""
    runtime.ros2_stop(DUAL_GAZEBO_NAME)
