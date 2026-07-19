"""Focused contracts for real UR5e Interactive Teleop trajectory routing."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _teleop_module() -> Any:
    path = ROOT / "ros2/cais_lab_robotics/scripts/keyboard_teleop.py"
    spec = importlib.util.spec_from_file_location("keyboard_teleop_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _ImmediateFuture:
    def __init__(self, result: Any) -> None:
        self._result = result

    @staticmethod
    def done() -> bool:
        return True

    def result(self) -> Any:
        return self._result


class _PendingFuture:
    @staticmethod
    def done() -> bool:
        return False


def test_real_ur5e_joint_target_uses_rtde_action_not_topic() -> None:
    module = _teleop_module()
    teleop = object.__new__(module.KeyboardTeleop)
    joint_names = list(module.ROBOTS["ur5e"]["joint_name_candidates"][1])
    targets = [0.1, -1.0, -2.0, -1.2, 1.5, 0.0]
    teleop.active_joint_names = {"ur5e": joint_names}
    teleop.joint_duration_sec = 0.08
    calls: list[tuple[list[str], list[float], float]] = []
    teleop._move_ur5e_arm_action = lambda names, positions, duration_sec: (
        calls.append((list(names), list(positions), float(duration_sec))) or (True, "RTDE")
    )
    teleop._pick_publisher = lambda _publishers: pytest.fail(
        "real UR5e must not publish to a Gazebo trajectory topic"
    )

    assert teleop.move_arm_to_joints("ur5e", targets, duration_sec=1.2) == (True, "RTDE")
    assert calls == [(joint_names, targets, 1.2)]


def test_gazebo_ur5e_joint_target_preserves_topic_path() -> None:
    module = _teleop_module()
    teleop = object.__new__(module.KeyboardTeleop)
    joint_names = list(module.ROBOTS["ur5e"]["joint_name_candidates"][0])
    targets = [0.1, -1.0, -2.0, -1.2, 1.5, 0.0]
    publisher = object()
    teleop.active_joint_names = {"ur5e": joint_names}
    teleop.joint_duration_sec = 0.08
    teleop.arm_publishers = {"ur5e": {"/simulation": publisher}}
    current_positions = [0.0, -1.1, -1.9, -1.3, 1.4, 0.1]
    teleop.joint_positions = {"ur5e": current_positions}
    teleop.joint_state_map = {}
    teleop._move_ur5e_arm_action = lambda *_args, **_kwargs: pytest.fail(
        "Gazebo UR5e must retain its trajectory topic"
    )
    teleop._pick_publisher = lambda _publishers: (publisher, "/simulation")
    published: list[tuple[list[str], list[float], float]] = []
    teleop._publish_joint_trajectory = lambda _publisher, names, positions, duration_sec: (
        published.append((list(names), list(positions), float(duration_sec))) or (True, "OK")
    )

    assert teleop.move_arm_to_joints("ur5e", targets, duration_sec=1.2) == (True, "OK")
    assert published == [(joint_names, targets, 1.2)]


def test_teleop_state_reports_joint_state_freshness() -> None:
    module = _teleop_module()
    teleop = object.__new__(module.KeyboardTeleop)
    teleop.joint_positions = {"ur5e": [0.0] * 6}
    teleop.joint_state_received_monotonic = {"ur5e": module.time.monotonic() - 0.25}
    teleop.get_ee_pose = lambda _robot: None

    state = teleop.get_robot_state("ur5e")

    assert state["joint_state_age_sec"] == pytest.approx(0.25, abs=0.05)


@pytest.mark.parametrize(
    ("server_ready", "accepted", "error_code", "error_string", "expected"),
    [
        (False, True, 0, "", "is not available"),
        (True, False, 0, "", "goal rejected"),
        (True, True, -1, "start state mismatch", "start state mismatch"),
        (True, True, 0, "", "succeeded"),
    ],
)
def test_real_ur5e_action_reports_controller_outcome(
    server_ready: bool,
    accepted: bool,
    error_code: int,
    error_string: str,
    expected: str,
) -> None:
    module = _teleop_module()
    teleop = object.__new__(module.KeyboardTeleop)
    action_name = "/cais_ur5e_rtde_trajectory_controller/follow_joint_trajectory"
    wrapped_result = SimpleNamespace(
        result=SimpleNamespace(error_code=error_code, error_string=error_string)
    )
    goal_handle = SimpleNamespace(
        accepted=accepted,
        get_result_async=lambda: _ImmediateFuture(wrapped_result),
    )
    sent_goals: list[Any] = []

    def _send_goal(goal: Any) -> _ImmediateFuture:
        sent_goals.append(goal)
        return _ImmediateFuture(goal_handle)

    client = SimpleNamespace(
        wait_for_server=lambda timeout_sec: server_ready,
        send_goal_async=_send_goal,
    )
    teleop.ur5e_hardware_trajectory_client = client
    teleop.ur5e_hardware_trajectory_action = action_name
    current_positions = [0.0, -1.1, -1.9, -1.3, 1.4, 0.1]
    teleop.joint_positions = {"ur5e": current_positions}
    teleop.joint_state_map = {}
    teleop._wait_future = lambda future, timeout: future.done()
    joint_names = list(module.ROBOTS["ur5e"]["joint_name_candidates"][1])
    targets = [0.1, -1.0, -2.0, -1.2, 1.5, 0.0]

    ok, message = teleop._move_ur5e_arm_action(joint_names, targets, duration_sec=1.2)

    assert ok is (server_ready and accepted and error_code == 0)
    assert expected in message
    if server_ready:
        assert len(sent_goals) == 1
        assert list(sent_goals[0].trajectory.points[0].positions) == current_positions
        assert list(sent_goals[0].trajectory.points[1].positions) == targets
    if ok:
        assert teleop.joint_positions["ur5e"] == targets


def test_real_ur5e_action_reports_send_timeout() -> None:
    module = _teleop_module()
    teleop = object.__new__(module.KeyboardTeleop)
    teleop.ur5e_hardware_trajectory_action = (
        "/cais_ur5e_rtde_trajectory_controller/follow_joint_trajectory"
    )
    teleop.ur5e_hardware_trajectory_client = SimpleNamespace(
        wait_for_server=lambda timeout_sec: True,
        send_goal_async=lambda _goal: _PendingFuture(),
    )
    teleop.joint_positions = {"ur5e": [0.0] * 6}
    teleop._wait_future = lambda future, timeout: future.done()

    ok, message = teleop._move_ur5e_arm_action(
        list(module.ROBOTS["ur5e"]["joint_name_candidates"][1]),
        [0.1, -1.0, -2.0, -1.2, 1.5, 0.0],
        duration_sec=1.2,
    )

    assert ok is False
    assert "send timeout" in message


def test_real_ur5e_action_requires_current_joint_state() -> None:
    module = _teleop_module()
    teleop = object.__new__(module.KeyboardTeleop)
    teleop.ur5e_hardware_trajectory_action = (
        "/cais_ur5e_rtde_trajectory_controller/follow_joint_trajectory"
    )
    teleop.ur5e_hardware_trajectory_client = SimpleNamespace(
        wait_for_server=lambda timeout_sec: True,
        send_goal_async=lambda _goal: pytest.fail("goal must not be sent without joint state"),
    )
    teleop.joint_positions = {}

    ok, message = teleop._move_ur5e_arm_action(
        list(module.ROBOTS["ur5e"]["joint_name_candidates"][1]),
        [0.1, -1.0, -2.0, -1.2, 1.5, 0.0],
        duration_sec=1.2,
    )

    assert ok is False
    assert "current joint state is unavailable" in message


def test_real_ur5e_action_reports_result_timeout() -> None:
    module = _teleop_module()
    teleop = object.__new__(module.KeyboardTeleop)
    goal_handle = SimpleNamespace(
        accepted=True,
        get_result_async=lambda: _PendingFuture(),
    )
    teleop.ur5e_hardware_trajectory_action = (
        "/cais_ur5e_rtde_trajectory_controller/follow_joint_trajectory"
    )
    teleop.ur5e_hardware_trajectory_client = SimpleNamespace(
        wait_for_server=lambda timeout_sec: True,
        send_goal_async=lambda _goal: _ImmediateFuture(goal_handle),
    )
    teleop.joint_positions = {"ur5e": [0.0] * 6}
    teleop._wait_future = lambda future, timeout: future.done()

    ok, message = teleop._move_ur5e_arm_action(
        list(module.ROBOTS["ur5e"]["joint_name_candidates"][1]),
        [0.1, -1.0, -2.0, -1.2, 1.5, 0.0],
        duration_sec=1.2,
    )

    assert ok is False
    assert "result timeout" in message


def test_bridge_passes_configured_rtde_action_and_uses_daemon_free_preflight() -> None:
    bridge = (ROOT / "cais_spade_llm/ui/bridge.py").read_text(encoding="utf-8")
    teleop = (ROOT / "ros2/cais_lab_robotics/scripts/keyboard_teleop.py").read_text(
        encoding="utf-8"
    )
    assert "--ur5e-hardware-trajectory-action" in bridge
    assert "teleop_named_position_readiness" in bridge
    assert "self._wait_for_ros_action(" in bridge
    assert "status_target = self._active_digital_twin_target_from_status()" in bridge
    assert "external_ur5e_runtime = (" in bridge
    assert "self.infer_robot_environment(robot) == 'real'" in teleop
    assert "self._move_ur5e_arm_action(" in teleop


def test_rtde_server_keeps_read_only_joint_monitoring_in_local_control() -> None:
    server = (
        ROOT / "ros2/cais_lab_robotics/scripts/ur5e_rtde_trajectory_server.py"
    ).read_text(encoding="utf-8")
    connect_method = server.split("    def _connect_rtde(self)", maxsplit=1)[1].split(
        "    def _reconnect_receive_locked", maxsplit=1
    )[0]

    assert connect_method.index("self.receive = self.receive_factory") < connect_method.index(
        "self.control = self.control_factory"
    )
    assert "UR5e joint-state monitoring ready in Local Control" in server
    assert '"rtde_receive_connected": False' in server
    assert '"rtde_control_connected": False' in server
    assert "Set the teach pendant to Remote Control" in server
    assert "before commanding motion" in server
    assert 'parser.add_argument(\n        "--monitor-only"' in server
    assert "if not self.monitor_only:" in server
    assert "disabled for read-only calibration monitoring" in server
    assert 'body["action"] = ""' in server
    assert 'body["action_name"] = ""' in server


def test_ur5e_state_publisher_can_launch_without_moveit_or_rviz() -> None:
    launch = (
        ROOT / "ros2/cais_lab_robotics/launch/ur5e_rg2_hardware_moveit.launch.py"
    ).read_text(encoding="utf-8")

    assert '"launch_move_group"' in launch
    assert "if not launch_move_group and not launch_rviz:" in launch
    assert "return launch_actions" in launch

    bridge = (ROOT / "cais_spade_llm/ui/bridge.py").read_text(encoding="utf-8")
    assert '"ur5e_calibration_rtde_monitor"' in bridge
    assert '"ur5e_calibration_state_publisher"' in bridge
    assert 'reason="ur5e_control_stack_start"' in bridge


def test_control_page_bounds_ros_readiness_refresh_work() -> None:
    control = (ROOT / "cais_spade_llm/ui/pages/control.py").read_text(encoding="utf-8")

    assert "named_pos_readiness_state = {" in control
    assert '"ready": False,' in control
    assert "def _update_named_position_go_enabled()" in control
    assert 'named_pos_readiness_state["ready"] = bool(ready)' in control
    assert "and not named_pos_busy[\"moving\"]" in control
    assert 'if named_pos_readiness_state["busy"]:' in control
    assert 'active_target_refresh = {"busy": False}' in control
    assert 'if active_target_refresh["busy"]:' in control
    assert "await asyncio.to_thread(bridge.digital_twin_statuses)" in control
    assert "signature = _signature(rows)" in control
    assert "signature = _signature()" not in control
    assert 'initial_robot = (' in control
    assert 'bridge.teleop_target("ur5e", "state")' in control
    assert 'f"Trajectory interface ({robot}): {message}"' in control
    assert "ui.timer(3.0, _refresh_named_position_readiness)" in control


def test_external_ur5e_digital_twin_status_resolves_named_position_target() -> None:
    from cais_spade_llm.ui.bridge import SystemBridge

    bridge = object.__new__(SystemBridge)
    bridge.robot_env = "real"
    bridge._ros2_procs = {}
    bridge._active_digital_twin_target_from_status = lambda: "ur5e only"

    target = bridge.teleop_target("ur5e", "move_joints")

    assert target["ready"] is True
    assert target["environment"] == "real"
    assert target["ros_domain_id"] == 42
    assert target["source"] == "digital_twin:ur5e only"
    assert target["externally_detected"] is True
