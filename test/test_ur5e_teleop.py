"""Focused contracts for real UR5e Interactive Teleop trajectory routing."""

from __future__ import annotations

import importlib.util
import json
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
    assert "--include-hidden-services --no-daemon --spin-time 0.5" in bridge


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
    assert "self._next_status_heartbeat_monotonic = now + 1.0" in server
    assert 'state="stopped"' in server
    assert "UR5e RTDE trajectory server exited:" in server


def test_rtde_result_timeout_stops_and_clears_goal_without_destroying_server() -> None:
    server = (
        ROOT / "ros2/cais_lab_robotics/scripts/ur5e_rtde_trajectory_server.py"
    ).read_text(encoding="utf-8")
    timeout_block = server.split(
        'reason = "UR5e RTDE trajectory result timeout"', maxsplit=1
    )[1].split("        except Exception as exc:", maxsplit=1)[0]

    assert timeout_block.index("self._stop_motion()") < timeout_block.index(
        "goal_handle.abort()"
    )
    assert "self._clear_active_goal(goal_handle)" in timeout_block
    assert "self._action_server.destroy" not in timeout_block


def test_rtde_result_timeout_remains_blocked_until_server_status_is_repaired() -> None:
    from cais_spade_llm.ui.bridge import SystemBridge

    status = {
        "state": "failed",
        "message": "UR5e RTDE trajectory result timeout",
        "updated_at": 100.0,
    }

    assert SystemBridge._ur5e_rtde_result_timeout_requires_repair(status, now=110.0) is True
    assert SystemBridge._ur5e_rtde_result_timeout_requires_repair(status, now=131.0) is True
    assert SystemBridge._ur5e_rtde_result_timeout_requires_repair(
        {**status, "updated_at": None}, now=131.0
    ) is False


def test_named_position_readiness_reports_action_domain_and_repair_instruction() -> None:
    from cais_spade_llm.ui.bridge import SystemBridge

    bridge = object.__new__(SystemBridge)
    bridge.teleop_target = lambda _robot, _operation: {
        "warning": "",
        "environment": "real",
        "ros_domain_id": 42,
        "source": "digital_twin:dual robots",
    }
    bridge._ur5e_rtde_trajectory_status = lambda: {
        "state": "failed",
        "message": "UR5e RTDE trajectory result timeout",
        "updated_at": 100.0,
    }
    bridge._wait_for_ros_action = lambda *_args, **_kwargs: pytest.fail(
        "result timeout must block before another action readiness attempt"
    )

    ready, message = bridge.teleop_named_position_readiness("ur5e")

    assert ready is False
    assert "/cais_ur5e_rtde_trajectory_controller/follow_joint_trajectory" in message
    assert "ROS_DOMAIN_ID=42" in message
    assert "Click Repair Twin for dual robots" in message


def test_named_position_readiness_rejects_stale_rtde_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cais_spade_llm.ui.bridge import SystemBridge

    bridge = object.__new__(SystemBridge)
    bridge.teleop_target = lambda _robot, _operation: {
        "warning": "",
        "environment": "real",
        "ros_domain_id": 42,
        "source": "digital_twin:dual robots",
    }
    bridge._ur5e_rtde_trajectory_status = lambda: {
        "state": "ready",
        "message": "UR5e RTDE trajectory server ready",
        "rtde_control_connected": True,
        "updated_at": 100.0,
    }
    bridge._wait_for_ros_action = lambda *_args, **_kwargs: pytest.fail(
        "stale status must block before ROS discovery"
    )
    monkeypatch.setattr("cais_spade_llm.ui.bridge.time.time", lambda: 105.0)

    ready, message = bridge.teleop_named_position_readiness("ur5e")

    assert ready is False
    assert "status is stale (5.0 s old)" in message
    assert "Click Repair Twin for dual robots" in message


def test_dual_twin_waiting_robot_mirror_reports_repair_needed() -> None:
    from cais_spade_llm.ui.bridge import SystemBridge

    bridge = object.__new__(SystemBridge)
    cfg = {
        "hardware_supported": True,
        "hardware": ("xarm6", "ur5e"),
        "gazebo_process": "dual_gazebo",
        "hardware_processes": {},
        "sync_processes": {"xarm6": "sync_xarm6", "ur5e": "sync_ur5e"},
    }
    bridge._DIGITAL_TWIN_TARGETS = {"dual robots": cfg}
    bridge._digital_twin_domain_ids = lambda: {"gazebo": 41, "hardware": 42}
    bridge._digital_twin_gazebo_launch = lambda _target, _cfg: "gazebo_dual_passive"
    bridge._digital_twin_hardware_status = lambda _cfg: {
        "overall": "running",
        "moveit": "running",
        "ur5e": {"overall": "running"},
    }
    bridge._ur5e_rg2_gripper_status = lambda: {}
    bridge.ros2_proc_status = lambda _name: "running"
    sync_snapshot = {
        "process": "sync_xarm6, sync_ur5e",
        "process_status": "running",
        "status_data": {
            "state": "waiting",
            "message": "ur5e is waiting for /joint_states",
        },
        "status_age_ms": 100.0,
        "robots": {
            "xarm6": {"state": "mirroring", "status_age_ms": 100.0},
            "ur5e": {"state": "waiting", "status_age_ms": 100.0},
        },
    }
    bridge._digital_twin_sync_status_snapshot = lambda _target, _cfg, _now: sync_snapshot
    bridge._digital_twin_dual_drag_markers_status_path = lambda _target: Path(
        "/tmp/not-used.json"
    )
    bridge._read_json_file = lambda _path: {}
    bridge._digital_twin_direction = lambda _target: "hardware -> gazebo"
    bridge._digital_twin_blocked_reason = lambda _target, _cfg: ""
    bridge._digital_twin_sim_mode = lambda _target: "monitor"
    bridge._digital_twin_allowed_sim_modes = lambda _cfg: ("monitor",)
    bridge._digital_twin_hardware_domain_id = (
        lambda _cfg, _robot, domains: domains["hardware"]
    )
    bridge._digital_twin_status_path = lambda _target: Path("/tmp/not-used.json")

    row = bridge.digital_twin_statuses()["dual robots"]

    assert row["repair_needed"] is True
    assert "ur5e mirror is waiting" in row["repair_reason"]
    assert row["hardware"]["domain"] == 42

    sync_snapshot["status_data"] = {
        "state": "mirroring",
        "message": "both robots are mirroring",
    }
    sync_snapshot["robots"]["ur5e"] = {
        "state": "mirroring",
        "status_age_ms": 100.0,
    }

    healthy_row = bridge.digital_twin_statuses()["dual robots"]

    assert healthy_row["repair_needed"] is False
    assert healthy_row["repair_reason"] == ""


def test_repair_stop_is_scoped_and_contains_no_motion_operation() -> None:
    from cais_spade_llm.ui.bridge import SystemBridge

    bridge = object.__new__(SystemBridge)
    events: list[str] = []
    bridge._digital_twin_process_names = lambda _cfg: ["driver", "moveit", "gazebo"]
    bridge.ros2_stop = lambda name, reason="": events.append(f"stop:{name}:{reason}")
    bridge._stop_teleop_server = lambda: events.append("stop:teleop")
    bridge._force_kill_digital_twin_helpers = lambda: events.append("stop:helpers")
    bridge._kill_stale_gazebo_helpers = lambda: events.append("stop:gazebo_helpers")
    bridge._force_kill_gazebo_core = lambda reason="": events.append(
        f"stop:gazebo_core:{reason}"
    )

    bridge._stop_digital_twin_stack({}, reason="digital_twin_repair")

    assert events[:3] == [
        "stop:gazebo:digital_twin_repair",
        "stop:moveit:digital_twin_repair",
        "stop:driver:digital_twin_repair",
    ]
    assert all("replay" not in event and "trajectory" not in event for event in events)


def test_home2_is_removed_while_home_and_prusa_mk4_2_remain_unchanged() -> None:
    payload = json.loads(
        (ROOT / "cais_spade_llm/initialization/resources/robot_ur5e.json").read_text(
            encoding="utf-8"
        )
    )

    named_positions = payload["ur5e"]["real"]["named_positions"]
    assert "home2" not in named_positions
    assert named_positions["home"] == [
        -1.453988,
        -0.856088,
        -2.346209,
        -1.510223,
        1.568386,
        -3.021614,
    ]
    assert named_positions["prusa-mk4-2"] == [
        0.087405,
        -0.888637,
        -2.15416,
        -1.669312,
        1.553256,
        -3.148605,
    ]


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
    assert '"Repair Twin" if repair_needed else "Start Twin"' in control
    assert "repair=repair_needed" in control


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
