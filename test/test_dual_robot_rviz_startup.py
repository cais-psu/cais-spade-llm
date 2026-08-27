"""Focused no-motion regression tests for dual robots Start Twin RViz startup."""

from __future__ import annotations

from typing import Any

from cais_spade_llm.ui.bridge import SystemBridge


class _LaunchRecorder:
    def __init__(self) -> None:
        self.command = ""
        self.ensure_calls: list[dict[str, Any]] = []

    def _ros2_launch_prereq_error(self, _launch_name: str) -> None:
        return None

    def _render_ros2_launch_cmd(self, _launch_name: str) -> str:
        return (
            "ros2 launch cais_lab_robotics dual_robots_hardware_moveit.launch.py "
            "launch_rviz:=false"
        )

    def _start_tracked_ros2_command(
        self,
        _process_name: str,
        command: str,
        *,
        ros_domain_id: int,
    ) -> None:
        assert ros_domain_id == 40
        self.command = command
        return None

    def _digital_twin_dual_robots_processes(
        self,
        _cfg: dict[str, Any],
    ) -> tuple[str, str, str, str]:
        return "xarm6_driver", "ur5e_driver", "", "dual_moveit"

    def _ensure_digital_twin_launch(
        self,
        process_name: str,
        launch_name: str,
        *,
        ros_domain_id: int,
        extra_args: str = "",
    ) -> None:
        self.ensure_calls.append(
            {
                "process_name": process_name,
                "launch_name": launch_name,
                "ros_domain_id": ros_domain_id,
                "extra_args": extra_args,
            }
        )
        return None

    def _start_ur5e_rtde_trajectory_server(
        self,
        _process_name: str,
        *,
        ros_domain_id: int,
    ) -> None:
        assert ros_domain_id == 40
        return None

    def _wait_with_ros2_daemon_retry(self, *_args: Any, **_kwargs: Any) -> None:
        return None


def test_digital_twin_launch_extra_argument_replaces_hard_coded_value() -> None:
    bridge = _LaunchRecorder()

    result = SystemBridge._start_digital_twin_launch(
        bridge,
        "dual_moveit",
        "hardware_dual_robots_moveit",
        ros_domain_id=40,
        extra_args="launch_rviz:=true",
    )

    assert result is None
    assert bridge.command.endswith("launch_rviz:=true")
    assert "launch_rviz:=false" not in bridge.command


def test_dual_robots_monitor_start_explicitly_enables_rviz() -> None:
    bridge = _LaunchRecorder()

    result = SystemBridge._start_digital_twin_dual_robots_hardware_launches(
        bridge,
        {},
        ros_domain_id=40,
        launch_rviz=True,
    )

    assert result is None
    moveit_call = next(
        call
        for call in bridge.ensure_calls
        if call["launch_name"] == "hardware_dual_robots_moveit"
    )
    assert moveit_call["extra_args"] == "launch_rviz:=true"
