"""Focused contracts for guarded UR5e RTDE insertion."""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
import math
import re
import tempfile
import threading
import time as stdlib_time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SERVER_PATH = ROOT / "ros2" / "cais_lab_robotics" / "scripts" / "ur5e_rtde_trajectory_server.py"


def _server_module() -> Any:
    spec = importlib.util.spec_from_file_location("ur5e_rtde_insert_test", SERVER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _set_insert_hard_caps(module: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    values = {
        "UR5E_RTDE_INSERT_MAX_CONTACT_SPEED_M_S": 0.01,
        "UR5E_RTDE_INSERT_MAX_CONTACT_FORCE_DELTA_N": 10.0,
        "UR5E_RTDE_INSERT_MAX_ENGAGEMENT_PROGRESS_M": 0.005,
        "UR5E_RTDE_INSERT_MAX_INSERTION_FORCE_N": 10.0,
        "UR5E_RTDE_INSERT_MAX_SPIRAL_RADIUS_M": 0.01,
        "UR5E_RTDE_INSERT_MAX_SPIRAL_PITCH_M": 0.005,
        "UR5E_RTDE_INSERT_MAX_SPIRAL_SPEED_M_S": 0.01,
        "UR5E_RTDE_INSERT_MAX_SPIRAL_ACCELERATION_M_S2": 0.1,
        "UR5E_RTDE_INSERT_MAX_AXIAL_FORCE_N": 20.0,
        "UR5E_RTDE_INSERT_MAX_LATERAL_FORCE_N": 10.0,
        "UR5E_RTDE_INSERT_MAX_TORQUE_NM": 5.0,
        "UR5E_RTDE_INSERT_MAX_TOOL_FLANGE_TORQUE_NM": 8.0,
        "UR5E_RTDE_INSERT_SOFT_FILTER_WINDOW_SEC": 0.06,
        "UR5E_RTDE_INSERT_SOFT_OVERLOAD_HOLD_SEC": 0.10,
        "UR5E_RTDE_INSERT_RELIEF_UNLOAD_DWELL_SEC": 0.01,
        "UR5E_RTDE_INSERT_RELIEF_CLEAR_DWELL_SEC": 0.01,
        "UR5E_RTDE_INSERT_RELIEF_CLEAR_HYSTERESIS_RATIO": 0.8,
        "UR5E_RTDE_INSERT_RELIEF_TIMEOUT_SEC": 0.3,
        "UR5E_RTDE_INSERT_RELIEF_AXIAL_FORCE_RATIO": 0.5,
        "UR5E_RTDE_INSERT_RELIEF_REVERSE_FORCE_RATIO": 0.25,
        "UR5E_RTDE_INSERT_RELIEF_RESUME_RAMP_SEC": 0.02,
        "UR5E_RTDE_INSERT_RELIEF_SEARCH_FORCE_RATIO": 0.5,
        "UR5E_RTDE_INSERT_RELIEF_SEARCH_SPEED_RATIO": 0.5,
        "UR5E_RTDE_INSERT_RELIEF_BACKOFF_STEP_M": 0.0005,
        "UR5E_RTDE_INSERT_MAX_RELIEF_RETREAT_M": 0.0015,
        "UR5E_RTDE_INSERT_RELIEF_STATIONARY_SPEED_M_S": 0.001,
        "UR5E_RTDE_INSERT_RELIEF_STATIONARY_ANGULAR_SPEED_RAD_S": 0.02,
        "UR5E_RTDE_INSERT_MAX_RELIEF_CYCLES": 3,
        "UR5E_RTDE_INSERT_MAX_TILT_TOLERANCE_RAD": 0.05,
        "UR5E_RTDE_INSERT_MAX_SEATED_DEPTH_TOLERANCE_M": 0.002,
        "UR5E_RTDE_INSERT_MAX_SETTLE_TIME_SEC": 0.05,
        "UR5E_RTDE_INSERT_MAX_TIMEOUT_SEC": 1.0,
        "UR5E_RTDE_INSERT_MAX_TRAVEL_M": 0.05,
        "UR5E_RTDE_INSERT_START_POSITION_TOLERANCE_M": 0.002,
        "UR5E_RTDE_INSERT_START_ORIENTATION_TOLERANCE_RAD": 0.05,
    }
    for name, value in values.items():
        monkeypatch.setattr(module, name, value)
    monkeypatch.setattr(
        module,
        "UR5E_RTDE_INSERT_MG_HARD_CAPS",
        {
            "insert_max_insertion_force_n": 10.0,
            "insert_max_axial_force_n": 20.0,
            "insert_max_lateral_force_n": 10.0,
            "insert_max_torque_nm": 5.0,
            "insert_max_tool_flange_torque_nm": 8.0,
            "insert_max_relief_retreat_m": 0.0015,
            "insert_max_contact_search_radius_m": 0.01,
            "insert_max_disengagement_cycles": 6.0,
            "insert_search_peck_retreat_m": 0.003,
            "insert_search_peck_interval_sec": 0.75,
        },
    )


def _insert_pose(module: Any, z_m: float) -> Any:
    return _pose_stamped_from_transform(
        module,
        ((0.0, 0.5, z_m), (0.0, 0.0, 0.0, 1.0)),
    )


def _pose_stamped_from_transform(module: Any, value: Any) -> Any:
    translation, rotation = value
    pose = module.PoseStamped()
    pose.header.frame_id = "world"
    pose.pose.position.x = float(translation[0])
    pose.pose.position.y = float(translation[1])
    pose.pose.position.z = float(translation[2])
    pose.pose.orientation.x = float(rotation[0])
    pose.pose.orientation.y = float(rotation[1])
    pose.pose.orientation.z = float(rotation[2])
    pose.pose.orientation.w = float(rotation[3])
    return pose


def _insert_request(module: Any, *, spiral_radius_m: float = 0.002) -> Any:
    hard_caps = module._insert_hard_caps("MG")
    return SimpleNamespace(
        part_name="MG",
        calibration_id="assembly_board-v1",
        profile_sha256="a" * 64,
        hard_caps_sha256=module._insert_hard_caps_sha256(hard_caps),
        trial_id="trial-test",
        force_depth_fraction=[index / 15.0 for index in range(16)],
        force_depth_axial_upper_n=[10.0] * 16,
        force_depth_lateral_upper_n=[5.0] * 16,
        force_depth_torque_upper_nm=[2.0] * 16,
        expected_start_tool0_pose=_insert_pose(module, 1.1),
        target_tool0_pose=_insert_pose(module, 1.09),
        insertion_axis_world=SimpleNamespace(x=0.0, y=0.0, z=-1.0),
        contact_speed_m_s=0.002,
        contact_force_delta_n=3.0,
        engagement_progress_m=0.001,
        insertion_force_n=4.0,
        spiral_radius_m=spiral_radius_m,
        spiral_pitch_m=0.0005,
        spiral_speed_m_s=0.002,
        spiral_acceleration_m_s2=0.02,
        max_axial_force_n=10.0,
        max_lateral_force_n=5.0,
        max_torque_nm=2.0,
        baseline_force_uncertainty_n=1.0,
        baseline_torque_uncertainty_nm=0.1,
        tilt_tolerance_rad=0.02,
        seated_depth_tolerance_m=0.0005,
        settle_time_sec=0.001,
        timeout_sec=0.8,
    )


def _refresh_request_hard_caps_sha256(module: Any, goal: Any) -> None:
    goal.request.hard_caps_sha256 = module._insert_hard_caps_sha256(
        module._insert_hard_caps(str(goal.request.part_name))
    )


class _InsertGoal:
    def __init__(self, request: Any, *, cancel_after_checks: int | None = None) -> None:
        self.request = request
        self.cancel_after_checks = cancel_after_checks
        self.cancel_checks = 0
        self.outcomes: list[str] = []
        self.feedback: list[Any] = []

    @property
    def is_cancel_requested(self) -> bool:
        self.cancel_checks += 1
        return (
            self.cancel_after_checks is not None
            and self.cancel_checks > self.cancel_after_checks
        )

    def abort(self) -> None:
        self.outcomes.append("aborted")

    def canceled(self) -> None:
        self.outcomes.append("canceled")

    def succeed(self) -> None:
        self.outcomes.append("succeeded")

    def publish_feedback(self, feedback: Any) -> None:
        self.feedback.append(feedback)


class _AdvancingClock:
    def __init__(self, step_sec: float = 0.005) -> None:
        self.value = 0.0
        self.step_sec = float(step_sec)

    def monotonic(self) -> float:
        self.value += self.step_sec
        return self.value


def _insert_execution_harness(  # noqa: C901, PLR0913, PLR0915 - explicit safety controls.
    module: Any,
    monkeypatch: pytest.MonkeyPatch,
    *,
    force_samples: list[list[float] | None],
    cancel_after_checks: int | None = None,
    stationary_results: list[bool] | None = None,
    force_mode_failure: Exception | None = None,
    spiral_radius_m: float = 0.002,
    actual_transforms: list[Any] | None = None,
    tcp_speed_samples: list[list[float] | None] | None = None,
    feedback_timestamps: list[float | None] | None = None,
) -> tuple[Any, _InsertGoal, list[str]]:
    class _Result:
        pass

    class _Feedback:
        pass

    module.MoveUR5eInsert = SimpleNamespace(Result=_Result, Feedback=_Feedback)
    _set_insert_hard_caps(module, monkeypatch)
    monkeypatch.setattr(
        module,
        "INSERT_TRIAL_TRACE_ROOT",
        Path(tempfile.mkdtemp(prefix="cais-ur5e-insert-test-")),
    )
    monkeypatch.setattr(module.rclpy, "ok", lambda: True)
    monkeypatch.setattr(
        module,
        "time",
        SimpleNamespace(
            time=stdlib_time.time,
            monotonic=stdlib_time.monotonic,
            sleep=lambda _duration: None,
        ),
    )
    events: list[str] = []
    force_mode_commands: list[tuple[Any, ...]] = []
    request = _insert_request(module, spiral_radius_m=spiral_radius_m)
    goal = _InsertGoal(request, cancel_after_checks=cancel_after_checks)
    start_transform = ((0.0, 0.5, 1.1), (0.0, 0.0, 0.0, 1.0))
    pose_values = list(actual_transforms or [start_transform])
    last_pose = start_transform
    force_values = list(force_samples)
    last_force = [0.0] * 6
    speed_values = list(tcp_speed_samples or [[0.0] * 6])
    last_speed = [0.0] * 6
    timestamp_values = list(feedback_timestamps or [])
    last_timestamp = 0.0
    stationary_values = list(stationary_results or [True])

    server = object.__new__(module.UR5eRTDETrajectoryServer)
    server._active_lock = threading.Lock()
    server._shutdown_requested = False
    server._rtde_reset_required = False
    server._latched_terminal_status = None
    server._active_goal = None
    server._active_goal_status = None
    server._active_motion_kind = ""
    server._insert_force_mode_active = False
    server._insert_force_mode_command = None
    server._insert_servo_active = False
    def force_mode(*args: Any) -> bool:
        events.append("forceMode")
        force_mode_commands.append(args)
        if force_mode_failure is not None:
            raise force_mode_failure
        return True

    server.control = SimpleNamespace(
        forceMode=force_mode,
        forceModeStop=lambda: events.append("forceModeStop") or True,
        getTCPOffset=lambda: [0.0] * 6,
        isPoseWithinSafetyLimits=lambda _pose: True,
        moveL=lambda *_args, **_kwargs: True,
        servoL=lambda *_args, **_kwargs: events.append("servoL") or True,
        servoStop=lambda: events.append("servoStop") or True,
        stopL=lambda *_args: events.append("stopL") or True,
    )
    server._connect_control_for_goal = lambda: None
    server._joint_states_fresh = lambda: True
    server._read_actual_q = lambda: [0.0] * 6
    server._ensure_control_program_for_goal = lambda: None
    server._lookup_rigid_transform = lambda _target, _source: (
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )
    server._active_tcp_offset = lambda: (
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )
    server._validated_cartesian_world_base = lambda: (
        ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
        "ready",
        0.0,
        0.0,
    )
    server._cartesian_frame_validation = lambda: (True, "ready", 0.0, 0.0)
    server._pose_stamped_from_transform = lambda value: _pose_stamped_from_transform(
        module, value
    )
    server._test_insert_statuses = []
    server._test_force_mode_commands = force_mode_commands

    def read_pose() -> Any:
        nonlocal last_pose
        if pose_values:
            last_pose = pose_values.pop(0)
        return last_pose

    def read_force() -> list[float] | None:
        nonlocal last_force
        events.append("force_sample")
        if force_values:
            value = force_values.pop(0)
            if value is None:
                return None
            last_force = list(value)
        return list(last_force)

    def read_speed() -> list[float] | None:
        nonlocal last_speed
        if speed_values:
            value = speed_values.pop(0)
            if value is None:
                return None
            last_speed = list(value)
        return list(last_speed)

    def read_timestamp() -> float | None:
        nonlocal last_timestamp
        if timestamp_values:
            value = timestamp_values.pop(0)
            if value is None:
                return None
            last_timestamp = float(value)
        elif feedback_timestamps is None:
            last_timestamp += 0.008
        return last_timestamp

    def confirm_stationary(**_kwargs: Any) -> bool:
        events.append("stationary")
        assert server._active_goal is goal
        if len(stationary_values) > 1:
            return stationary_values.pop(0)
        return stationary_values[0]

    def stop_motion() -> bool:
        events.append("stop")
        assert server._active_goal is goal
        server._insert_force_mode_active = False
        server._insert_force_mode_command = None
        server._insert_servo_active = False
        server._insert_force_mode_stop_acknowledged = True
        server._insert_servo_stop_acknowledged = True
        server._insert_stop_l_command_completed = True
        return True

    def finish(
        finished_goal: Any,
        _status: dict[str, Any],
        *,
        latch_status: bool = False,
    ) -> None:
        assert finished_goal is goal
        server._test_insert_statuses.append(dict(_status))
        events.append("finish_latched" if latch_status else "finish")
        server._active_goal = None
        server._active_goal_status = None
        server._active_motion_kind = ""

    def clear(cleared_goal: Any) -> None:
        assert cleared_goal is goal
        events.append("clear")
        if server._active_goal is goal:
            server._active_goal = None
            server._active_goal_status = None
            server._active_motion_kind = ""

    server._read_actual_tcp_force = read_force
    server._read_actual_tcp_speed = read_speed
    server._read_feedback_timestamp = read_timestamp
    server._read_actual_tcp_transform = read_pose
    server._confirm_stationary_after_stop = confirm_stationary
    server._stop_motion = stop_motion
    server._write_active_goal_status = (
        lambda _goal, status: server._test_insert_statuses.append(dict(status))
    )
    server._finish_active_goal_status = finish
    server._clear_active_goal = clear
    server._mark_rtde_reset_required = (
        lambda _reason, **kwargs: events.append(
            f"reset:{kwargs.get('failure_kind', '')}"
        )
    )
    return server, goal, events


def test_commissioned_relief_values_enable_insert_readiness() -> None:
    module = _server_module()

    error = module._insert_hard_cap_error()
    status = module._status_base()

    assert error is None
    assert status["insert_action_name"] == ("/cais_ur5e_rtde_cartesian_controller/move_insert")
    assert status["insert_supported_part_names"] == [
        "SG",
        "MG",
        "LG",
        "SCP",
        "MCP",
        "LCP",
    ]
    assert status["insert_max_insertion_force_n"] == pytest.approx(15.0)
    assert status["insert_max_spiral_radius_m"] == pytest.approx(0.002)
    assert status["insert_max_timeout_sec"] == pytest.approx(60.0)
    assert status["insert_max_tool_flange_torque_nm"] == pytest.approx(3.0)
    assert status["insert_soft_filter_window_sec"] == pytest.approx(0.05)
    assert status["insert_soft_overload_hold_sec"] == pytest.approx(0.10)
    assert status["insert_relief_backoff_step_m"] == pytest.approx(0.0001)
    assert status["insert_max_relief_retreat_m"] == pytest.approx(0.0003)
    assert module._insert_hard_caps("MG")[
        "insert_max_relief_retreat_m"
    ] == pytest.approx(0.0006)
    assert module._insert_hard_caps("SG")[
        "insert_max_relief_retreat_m"
    ] == pytest.approx(0.0003)
    assert status["insert_exact_part_hard_caps"]["MG"] == status[
        "insert_MG_hard_caps"
    ]
    assert status["insert_exact_part_hard_caps_error"]["MG"] == ""
    assert status["insert_exact_part_hard_caps_sha256"]["MG"] == status[
        "insert_MG_hard_caps_sha256"
    ]
    assert status["insert_exact_part_hard_caps_sha256"]["SG"] == (
        module._insert_hard_caps_sha256(module._insert_hard_caps("SG"))
    )
    assert status["insert_max_relief_cycles"] == 3
    assert status["insert_function_ready"] is False
    assert status["insert_readiness_message"] == (
        "Insertion interface and TCP force/speed feedback validation have not completed"
    )
    assert status["peak_axial_force_n"] is None
    assert status["peak_lateral_force_n"] is None
    assert status["peak_torque_nm"] is None
    assert status["insert_engagement_detected"] is False
    assert status["insert_seated_detected"] is False
    assert status["rtde_feedback_timestamp_sec"] is None
    write_status_source = inspect.getsource(
        module.UR5eRTDETrajectoryServer._write_status
    )
    assert 'body["rtde_feedback_timestamp_sec"] = getattr(' in write_status_source
    assert '"_last_receive_timestamp"' in write_status_source
    assert "actual_TCP_force" in status["rtde_receive_variables"]
    assert "actual_TCP_speed" in status["rtde_receive_variables"]


def test_exact_mg_caps_do_not_change_other_supported_parts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _server_module()
    _set_insert_hard_caps(module, monkeypatch)
    changed_mg_caps = dict(module.UR5E_RTDE_INSERT_MG_HARD_CAPS)
    changed_mg_caps["insert_max_axial_force_n"] = 80.0
    monkeypatch.setattr(
        module,
        "UR5E_RTDE_INSERT_MG_HARD_CAPS",
        changed_mg_caps,
    )

    assert module._insert_hard_caps("MG")[
        "insert_max_axial_force_n"
    ] == pytest.approx(80.0)
    for part_name in ("SG", "LG", "SCP", "MCP", "LCP"):
        part_caps = module._insert_hard_caps(part_name)
        assert part_caps[
            "insert_max_axial_force_n"
        ] == pytest.approx(20.0)
        assert "insert_max_contact_search_radius_m" not in part_caps
        assert "insert_max_disengagement_cycles" not in part_caps
        assert "insert_search_peck_retreat_m" not in part_caps
        assert "insert_search_peck_interval_sec" not in part_caps
    execute_source = inspect.getsource(
        module.UR5eRTDETrajectoryServer._execute_insert
    )
    assert 'part_name == "MG"' not in execute_source
    assert 'part_name != "MG"' not in execute_source


def test_exact_lg_policy_values_enable_advanced_recovery_without_changing_mg(
    tmp_path: Path,
) -> None:
    module = _server_module()
    config = module._load_hardware_arms_config(module.DEFAULT_CONFIG_FILE)
    original_mg_caps = dict(module.UR5E_RTDE_INSERT_MG_HARD_CAPS)
    config["ur5e"]["rtde"]["LG"] = {
        "insert_max_contact_search_radius_m": 0.006,
        "insert_max_disengagement_cycles": 4,
        "insert_search_peck_retreat_m": 0.001,
        "insert_search_peck_interval_sec": 0.5,
    }
    configured_path = tmp_path / "hardware_runtime.yaml"
    configured_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    module._apply_hardware_arms_config(configured_path)

    lg_caps = module._insert_hard_caps("LG")
    assert module.UR5E_RTDE_INSERT_MG_HARD_CAPS == original_mg_caps
    assert lg_caps["insert_max_insertion_force_n"] == pytest.approx(15.0)
    assert lg_caps["insert_max_contact_search_radius_m"] == pytest.approx(0.006)
    assert lg_caps["insert_max_disengagement_cycles"] == pytest.approx(4.0)
    assert module._insert_hard_cap_error("LG") is None
    status = module._status_base()
    assert status["insert_exact_part_hard_caps"]["LG"] == lg_caps
    assert status["insert_exact_part_hard_caps_error"]["LG"] == ""
    assert status["insert_exact_part_hard_caps_sha256"]["LG"] == (
        module._insert_hard_caps_sha256(lg_caps)
    )
    assert status["insert_exact_part_hard_caps_sha256"]["LG"] != status[
        "insert_exact_part_hard_caps_sha256"
    ]["SG"]


def test_partial_exact_lg_advanced_recovery_policy_is_rejected(
    tmp_path: Path,
) -> None:
    module = _server_module()
    config = module._load_hardware_arms_config(module.DEFAULT_CONFIG_FILE)
    config["ur5e"]["rtde"]["LG"] = {
        "insert_max_contact_search_radius_m": 0.006,
    }
    configured_path = tmp_path / "hardware_runtime.yaml"
    configured_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    module._apply_hardware_arms_config(configured_path)

    error = module._insert_hard_cap_error("LG")
    assert error is not None
    assert "LG protected advanced recovery policy is incomplete" in error
    status = module._status_base()
    assert status["insert_exact_part_hard_caps_error"]["LG"] == error
    assert status["insert_exact_part_hard_caps_sha256"]["LG"] == ""


def test_missing_exact_mg_caps_block_before_force_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _server_module()
    server, goal, events = _insert_execution_harness(
        module,
        monkeypatch,
        force_samples=[[0.0] * 6],
        stationary_results=[True],
    )
    monkeypatch.setattr(
        module,
        "UR5E_RTDE_INSERT_MG_HARD_CAPS",
        {
            "insert_max_insertion_force_n": None,
            "insert_max_axial_force_n": None,
        "insert_max_lateral_force_n": None,
        "insert_max_torque_nm": None,
        "insert_max_tool_flange_torque_nm": None,
        "insert_max_relief_retreat_m": None,
        },
    )
    _refresh_request_hard_caps_sha256(module, goal)

    result = server._execute_insert(goal)

    assert result.error_code == -2
    assert "Insertion hard caps are missing or invalid" in result.error_string
    assert "forceMode" not in events
    assert "servoL" not in events
    assert "moveL" not in events


def test_relief_hysteresis_config_is_strict_and_has_no_fallback(tmp_path: Path) -> None:
    module = _server_module()
    config = module._load_hardware_arms_config(module.DEFAULT_CONFIG_FILE)

    assert module.UR5E_RTDE_INSERT_RELIEF_CLEAR_HYSTERESIS_RATIO == pytest.approx(0.8)
    config["ur5e"]["rtde"]["insert_relief_clear_hysteresis_ratio"] = 0.75
    configured_path = tmp_path / "hardware_runtime.yaml"
    configured_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    module._apply_hardware_arms_config(configured_path)
    assert pytest.approx(0.75) == (
        module.UR5E_RTDE_INSERT_RELIEF_CLEAR_HYSTERESIS_RATIO
    )

    assert module._optional_float(
        {"ur5e": {"rtde": {"value": True}}},
        ("ur5e", "rtde", "value"),
    ) is None
    assert module._optional_float(
        {"ur5e": {"rtde": {"value": float("nan")}}},
        ("ur5e", "rtde", "value"),
    ) is None


def test_relief_config_rejects_invalid_ratio_and_retreat_relationships(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _server_module()
    _set_insert_hard_caps(module, monkeypatch)
    assert module._insert_hard_cap_error() is None

    monkeypatch.setattr(
        module,
        "UR5E_RTDE_INSERT_RELIEF_CLEAR_HYSTERESIS_RATIO",
        1.0,
    )
    assert "ratios must be less than 1" in module._insert_hard_cap_error()
    monkeypatch.setattr(
        module,
        "UR5E_RTDE_INSERT_RELIEF_CLEAR_HYSTERESIS_RATIO",
        0.8,
    )
    monkeypatch.setattr(module, "UR5E_RTDE_INSERT_RELIEF_BACKOFF_STEP_M", 0.002)
    assert "exceeds insert_max_relief_retreat_m" in module._insert_hard_cap_error()


def test_three_relief_cycles_cannot_exceed_protected_cumulative_retreat() -> None:
    module = _server_module()

    first_two_cycles_m = 0.001
    third_cycle_current_m = 0.00051
    protected_total_m = module._protected_relief_retreat_m(
        first_two_cycles_m,
        third_cycle_current_m,
        relief_backoff_committed=False,
    )

    assert protected_total_m > 0.0015
    assert module._protected_relief_retreat_m(
        protected_total_m,
        third_cycle_current_m,
        relief_backoff_committed=True,
    ) == pytest.approx(protected_total_m)


def test_insert_rejects_wrong_world_base_before_force_or_servo_motion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _server_module()
    server, goal, events = _insert_execution_harness(
        module,
        monkeypatch,
        force_samples=[[0.0] * 6],
    )
    server._validated_cartesian_world_base = (
        module.UR5eRTDETrajectoryServer._validated_cartesian_world_base.__get__(
            server,
            module.UR5eRTDETrajectoryServer,
        )
    )

    result = server._execute_insert(goal)

    assert result.error_code == -2
    assert "protected ur5e.rtde.cartesian_world_base" in result.error_string
    assert "forceMode" not in events
    assert "servoL" not in events
    assert not any(event.startswith("reset:") for event in events)


def test_insert_trace_rejects_trial_id_path_escape_before_hardware(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _server_module()
    server, goal, events = _insert_execution_harness(
        module,
        monkeypatch,
        force_samples=[[0.0] * 6],
    )
    goal.request.trial_id = "../escape"
    server._connect_control_for_goal = lambda: pytest.fail(
        "unsafe trial_id must be rejected before hardware access"
    )

    result = server._execute_insert(goal)

    assert result.error_code == -2
    assert "trial_id must match" in result.error_string
    assert "forceMode" not in events
    assert not (module.INSERT_TRIAL_TRACE_ROOT.parent / "escape").exists()


def test_insert_bounds_allow_direct_insertion_but_reject_cap_overrun() -> None:
    module = _server_module()

    assert module._bounded_insert_value(
        "spiral_radius_m", 0.0, 0.004, allow_zero=True
    ) == pytest.approx(0.0)
    with pytest.raises(ValueError, match="exceeds configured hard cap"):
        module._bounded_insert_value("insertion_force_n", 11.0, 10.0)
    with pytest.raises(RuntimeError, match="hard cap is not configured"):
        module._bounded_insert_value("timeout_sec", 1.0, None)


@pytest.mark.parametrize(
    "axis",
    [
        (0.0, 0.0, -1.0),
        (1.0, 0.0, 0.0),
        (0.2, -0.3, 0.9),
    ],
)
def test_insert_basis_is_right_handed_with_requested_axis(axis: tuple[float, ...]) -> None:
    module = _server_module()

    x_axis, y_axis, z_axis = module._insertion_basis(axis)

    expected_z = tuple(value / math.sqrt(sum(item * item for item in axis)) for value in axis)
    assert z_axis == pytest.approx(expected_z)
    assert module._vector_dot(x_axis, y_axis) == pytest.approx(0.0, abs=1e-12)
    assert module._vector_dot(x_axis, z_axis) == pytest.approx(0.0, abs=1e-12)
    assert module._vector_dot(y_axis, z_axis) == pytest.approx(0.0, abs=1e-12)
    assert module._vector_cross(x_axis, y_axis) == pytest.approx(z_axis)
    force_frame_rotation = module._quaternion_from_basis(x_axis, y_axis, z_axis)
    assert module._rotate_vector(force_frame_rotation, (0.0, 0.0, 1.0)) == pytest.approx(
        z_axis
    )


def test_insert_pose_and_force_metrics_use_insertion_axis() -> None:
    module = _server_module()
    identity = (0.0, 0.0, 0.0, 1.0)
    start = ((0.1, 0.2, 1.2), identity)
    target = ((0.1, 0.2, 1.18), identity)
    actual = ((0.101, 0.2, 1.185), identity)

    depth, depth_error, lateral, tilt = module._insertion_pose_metrics(
        actual,
        start,
        target,
        (0.0, 0.0, -1.0),
    )
    raw_axial, axial, lateral_force, torque, flange_torque, tared_force = (
        module._insertion_force_metrics(
        [1.0, 2.0, -8.0, 0.1, 0.2, 0.3],
        [1.0, 1.0, -2.0, 0.0, 0.0, 0.0],
        (0.0, 0.0, -1.0),
        (0.0, 0.0, 0.0),
        )
    )

    assert depth == pytest.approx(0.015)
    assert depth_error == pytest.approx(0.005)
    assert lateral == pytest.approx(0.001)
    assert tilt == pytest.approx(0.0)
    assert axial == pytest.approx(0.0)
    assert raw_axial == pytest.approx(6.0)
    assert lateral_force == pytest.approx(1.0)
    assert torque == pytest.approx(math.sqrt(0.14))
    assert flange_torque == pytest.approx(math.sqrt(0.14))
    assert tared_force == pytest.approx([0.0, 1.0, -6.0, 0.1, 0.2, 0.3])


def test_active_tcp_torque_removes_tool_offset_force_moment() -> None:
    module = _server_module()

    _raw_axial, _axial, _lateral, active_tcp_torque, flange_torque, _tared = (
        module._insertion_force_metrics(
            [0.0, 10.0, 0.0, 0.0, 0.0, 1.0],
            [0.0] * 6,
            (0.0, 0.0, 1.0),
            (0.1, 0.0, 0.0),
        )
    )

    assert flange_torque == pytest.approx(1.0)
    assert active_tcp_torque == pytest.approx(0.0)


def test_actual_tcp_force_reader_accepts_only_six_finite_values() -> None:
    module = _server_module()
    server = object.__new__(module.UR5eRTDETrajectoryServer)
    server._rtde_reset_required = False
    server._receive_lock = threading.Lock()
    server._last_actual_tcp_force = None
    server.receive = SimpleNamespace(getActualTCPForce=lambda: [1, 2, 3, 4, 5, 6])

    assert server._read_actual_tcp_force() == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    assert server._last_actual_tcp_force == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

    server.receive = SimpleNamespace(getActualTCPForce=lambda: [1, 2, 3, 4, 5, math.inf])
    assert server._read_actual_tcp_force() is None


def test_actual_tcp_speed_reader_accepts_only_six_finite_values() -> None:
    module = _server_module()
    server = object.__new__(module.UR5eRTDETrajectoryServer)
    server._rtde_reset_required = False
    server._receive_lock = threading.Lock()
    server._last_actual_tcp_speed = None
    server.receive = SimpleNamespace(getActualTCPSpeed=lambda: [1, 2, 3, 4, 5, 6])

    assert server._read_actual_tcp_speed() == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    assert server._last_actual_tcp_speed == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

    server.receive = SimpleNamespace(getActualTCPSpeed=lambda: [1, 2, 3, 4, 5, math.nan])
    assert server._read_actual_tcp_speed() is None


@pytest.mark.parametrize(
    ("reader_name", "getter_name"),
    [
        ("_read_actual_tcp_transform", "getActualTCPPose"),
        ("_read_actual_tcp_force", "getActualTCPForce"),
        ("_read_actual_tcp_speed", "getActualTCPSpeed"),
    ],
)
def test_insert_receive_failure_defers_disconnect_until_stop_attempt(
    reader_name: str,
    getter_name: str,
) -> None:
    module = _server_module()
    reset_calls: list[str] = []

    def fail_receive() -> Any:
        raise RuntimeError("transport lost")

    server = object.__new__(module.UR5eRTDETrajectoryServer)
    server._rtde_reset_required = False
    server._receive_lock = threading.Lock()
    server._active_lock = threading.Lock()
    server._active_goal = object()
    server._active_motion_kind = "insert"
    server._insert_motion_started = True
    server._receive_error = ""
    server._receive_transport_failed = False
    server.receive = SimpleNamespace(**{getter_name: fail_receive})
    server._mark_rtde_reset_required = lambda _reason, **_kwargs: reset_calls.append(
        "reset"
    )

    assert getattr(server, reader_name)() is None
    assert reset_calls == []
    assert server.receive is not None

    server._insert_motion_started = False
    assert getattr(server, reader_name)() is None
    assert reset_calls == ["reset"]


def test_insert_stop_exits_force_and_servo_before_stop_l() -> None:
    module = _server_module()
    calls: list[str] = []
    server = object.__new__(module.UR5eRTDETrajectoryServer)
    server._active_lock = threading.Lock()
    server._active_motion_kind = "insert"
    server._insert_force_mode_active = True
    server._insert_servo_active = True
    server.control = SimpleNamespace(
        forceModeStop=lambda: calls.append("forceModeStop") or True,
        servoStop=lambda: calls.append("servoStop") or True,
        stopL=lambda _acceleration: calls.append("stopL") or True,
    )

    assert server._stop_motion() is True
    assert calls == ["forceModeStop", "servoStop", "stopL"]
    assert server._insert_force_mode_active is False
    assert server._insert_servo_active is False


def test_insert_stop_requires_affirmative_stop_l_acknowledgement() -> None:
    module = _server_module()
    server = object.__new__(module.UR5eRTDETrajectoryServer)
    server._active_lock = threading.Lock()
    server._active_motion_kind = "insert"
    server._insert_force_mode_active = False
    server._insert_servo_active = False
    server.control = SimpleNamespace(stopL=lambda _acceleration: False)

    assert server._stop_motion() is False


def test_insert_stop_accepts_ur_rtde_stop_l_none_completion() -> None:
    module = _server_module()
    server = object.__new__(module.UR5eRTDETrajectoryServer)
    server._active_lock = threading.Lock()
    server._active_motion_kind = "insert"
    server._insert_force_mode_active = False
    server._insert_servo_active = False
    server.control = SimpleNamespace(stopL=lambda _acceleration: None)

    assert server._stop_motion() is True
    assert server._insert_force_mode_stop_acknowledged is True
    assert server._insert_servo_stop_acknowledged is True
    assert server._insert_stop_l_command_completed is True


@pytest.mark.parametrize(
    ("timestamp_reader", "tcp_speed", "expected"),
    [
        (lambda: 1.0, [0.0] * 6, False),
        (
            iter(float(index) for index in range(1, 100)).__next__,
            [0.002, 0.0, 0.0, 0.0, 0.0, 0.0],
            False,
        ),
        (
            iter(float(index) for index in range(1, 100)).__next__,
            [0.0] * 6,
            True,
        ),
    ],
)
def test_insert_stop_stationary_requires_advancing_joint_and_tcp_evidence(
    monkeypatch: pytest.MonkeyPatch,
    timestamp_reader: Any,
    tcp_speed: list[float],
    expected: bool,
) -> None:
    module = _server_module()
    monkeypatch.setattr(module, "UR5E_RTDE_STATIONARY_HOLD_SEC", 0.02)
    clock = _AdvancingClock(step_sec=0.005)
    monkeypatch.setattr(
        module,
        "time",
        SimpleNamespace(monotonic=clock.monotonic, sleep=lambda _duration: None),
    )
    server = object.__new__(module.UR5eRTDETrajectoryServer)
    server._read_feedback_timestamp = timestamp_reader
    server._read_actual_qd = lambda: [0.0] * 6
    server._read_actual_tcp_speed = lambda: list(tcp_speed)

    assert server._confirm_stationary_after_stop(
        timeout_sec=0.1,
        linear_speed_limit_m_s=0.001,
        angular_speed_limit_rad_s=0.02,
    ) is expected


def test_insert_force_mode_enables_lateral_compliance_only_for_search() -> None:
    module = _server_module()
    commands: list[tuple[Any, ...]] = []
    server = object.__new__(module.UR5eRTDETrajectoryServer)
    server._insert_force_mode_active = False
    server._insert_force_mode_command = None
    server.control = SimpleNamespace(forceMode=lambda *command: commands.append(command) or True)

    server._start_insert_force_mode(
        actual_base_tcp=((0.1, 0.2, 1.0), (0.0, 0.0, 0.0, 1.0)),
        insertion_axis_base=(0.0, 0.0, -1.0),
        insertion_force_n=8.0,
        contact_speed_m_s=0.002,
        spiral_speed_m_s=0.003,
        tilt_tolerance_rad=0.02,
    )
    server._refresh_insert_force_mode(
        lateral_force_x_n=1.0,
        lateral_force_y_n=-2.0,
        lateral_compliant=True,
    )
    server._refresh_insert_force_mode(
        axial_force_n=-2.0,
        lateral_compliant=False,
    )

    assert len(commands) == 3
    assert commands[0][0] == commands[1][0]
    assert commands[0][3:] == commands[1][3:]
    assert commands[0][1] == [0, 0, 1, 0, 0, 0]
    assert commands[1][1] == [1, 1, 1, 0, 0, 0]
    assert commands[2][1] == [0, 0, 1, 0, 0, 0]
    assert commands[0][2] == [0.0, 0.0, 8.0, 0.0, 0.0, 0.0]
    assert commands[1][2] == [1.0, -2.0, 8.0, 0.0, 0.0, 0.0]
    assert commands[2][2] == [0.0, 0.0, -2.0, 0.0, 0.0, 0.0]
    assert commands[0][3] == 2
    assert commands[0][4] == pytest.approx([0.003, 0.003, 0.002, 0.02, 0.02, 0.02])


def test_mg_search_guard_uses_search_entry_not_pre_search_lateral_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _server_module()
    start = ((0.0, 0.5, 1.1), (0.0, 0.0, 0.0, 1.0))
    drifted_contact = ((0.004, 0.5, 1.1), (0.0, 0.0, 0.0, 1.0))
    server, goal, events = _insert_execution_harness(
        module,
        monkeypatch,
        force_samples=[[0.0] * 6],
        stationary_results=[True, True],
        spiral_radius_m=0.01,
        actual_transforms=[start],
    )
    goal.request.timeout_sec = 0.4
    _refresh_request_hard_caps_sha256(module, goal)
    clock = _AdvancingClock(step_sec=0.005)
    module.time.monotonic = clock.monotonic

    server._read_actual_tcp_force = lambda: (
        [0.0] * 6
        if server._insert_force_mode_command is None
        else [0.0, 0.0, 4.0, 0.0, 0.0, 0.0]
    )
    server._read_actual_tcp_transform = lambda: (
        start
        if server._insert_force_mode_command is None
        else drifted_contact
    )

    result = server._execute_insert(goal)

    assert result.error_code == -5
    assert result.disengagement_cycle_count == 0
    searching_feedback = [
        feedback
        for feedback in goal.feedback
        if feedback.phase in {"searching", "expanded_searching"}
    ]
    assert len(searching_feedback) > 1
    assert any(
        command[1] == [1, 1, 1, 0, 0, 0]
        and (abs(command[2][0]) > 0.0 or abs(command[2][1]) > 0.0)
        for command in server._test_force_mode_commands
    )
    assert all(
        command[1] == [0, 0, 1, 0, 0, 0]
        for command in server._test_force_mode_commands
        if command[2][0] == 0.0 and command[2][1] == 0.0
    )


def test_insert_force_mode_unknown_acceptance_remains_owned_until_stop() -> None:
    module = _server_module()
    calls: list[str] = []
    server = object.__new__(module.UR5eRTDETrajectoryServer)
    server._insert_force_mode_active = False
    server._insert_force_mode_command = None
    server._insert_servo_active = False
    server.control = SimpleNamespace(
        forceMode=lambda *_args: (_ for _ in ()).throw(RuntimeError("send timeout")),
    )

    with pytest.raises(RuntimeError, match="send timeout"):
        server._start_insert_force_mode(
            actual_base_tcp=((0.1, 0.2, 1.0), (0.0, 0.0, 0.0, 1.0)),
            insertion_axis_base=(0.0, 0.0, -1.0),
            insertion_force_n=8.0,
            contact_speed_m_s=0.002,
            spiral_speed_m_s=0.003,
            tilt_tolerance_rad=0.02,
        )

    assert server._insert_force_mode_active is True
    assert server._insert_force_mode_command is not None
    server.control = SimpleNamespace(
        forceModeStop=lambda: calls.append("forceModeStop") or True,
        stopL=lambda _acceleration: calls.append("stopL") or True,
    )
    assert server._stop_insert_motion() is True
    assert calls == ["forceModeStop", "stopL"]


def test_insert_servo_unknown_acceptance_remains_owned_until_stop() -> None:
    module = _server_module()
    calls: list[str] = []
    server = object.__new__(module.UR5eRTDETrajectoryServer)
    server._insert_force_mode_active = False
    server._insert_servo_active = False
    server.control = SimpleNamespace(
        servoL=lambda *_args: (_ for _ in ()).throw(RuntimeError("send timeout")),
    )

    with pytest.raises(RuntimeError, match="send timeout"):
        server._execute_insert_servo_pose(
            ((0.1, 0.2, 1.0), (0.0, 0.0, 0.0, 1.0)),
            speed_m_s=0.002,
            acceleration_m_s2=0.02,
            cycle_sec=0.008,
        )

    assert server._insert_servo_active is True
    server.control = SimpleNamespace(
        servoStop=lambda: calls.append("servoStop") or True,
        stopL=lambda _acceleration: calls.append("stopL") or True,
    )
    assert server._stop_insert_motion() is True
    assert calls == ["servoStop", "stopL"]


def test_insert_direct_success_returns_complete_final_pose_and_fixed_phases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _server_module()
    start = ((0.0, 0.5, 1.1), (0.0, 0.0, 0.0, 1.0))
    target = ((0.0, 0.5, 1.09), (0.0, 0.0, 0.0, 1.0))
    server, goal, events = _insert_execution_harness(
        module,
        monkeypatch,
        force_samples=[*([[0.0] * 6] * 5), [0.0, 0.0, 4.0, 0.0, 0.0, 0.0]],
        stationary_results=[True, True],
        spiral_radius_m=0.0,
        actual_transforms=[start, start, target, target, target, target],
    )
    clock = _AdvancingClock()
    module.time.monotonic = clock.monotonic

    result = server._execute_insert(goal)

    assert result.error_code == 0
    assert result.error_string == ""
    assert result.state_uncertain is False
    assert result.motion_settled is True
    assert result.contact_detected is True
    assert result.engagement_detected is True
    assert result.seated_detected is True
    assert result.force_bias_valid is True
    assert result.force_bias == pytest.approx([0.0] * 6)
    assert result.final_tool0_pose_valid is True
    assert result.final_insertion_depth_m == pytest.approx(0.01)
    assert result.final_depth_error_m == pytest.approx(0.0)
    assert result.final_lateral_offset_m == pytest.approx(0.0)
    final_pose = result.final_tool0_pose
    assert final_pose.header.frame_id == "world"
    assert [
        final_pose.pose.position.x,
        final_pose.pose.position.y,
        final_pose.pose.position.z,
        final_pose.pose.orientation.x,
        final_pose.pose.orientation.y,
        final_pose.pose.orientation.z,
        final_pose.pose.orientation.w,
    ] == pytest.approx([0.0, 0.5, 1.09, 0.0, 0.0, 0.0, 1.0])
    assert goal.outcomes == ["succeeded"]
    assert events.count("forceMode") >= 2
    assert events.count("stop") == 1
    assert "moveL" not in events
    assert "servoL" not in events
    assert events.index("stationary") < events.index("forceMode")
    status_phases = {
        status["insert_phase"]
        for status in server._test_insert_statuses
        if status.get("insert_phase")
    }
    assert status_phases == {"checking", "zeroing_force", "seating", "settling"}
    assert {feedback.phase for feedback in goal.feedback} == {
        "checking",
        "zeroing_force",
        "seating",
        "settling",
    }
    assert goal.feedback[-1].engagement_detected is True
    assert goal.feedback[-1].seated_detected is True
    assert goal.feedback[-1].actual_tcp_force == pytest.approx(
        [0.0, 0.0, 4.0, 0.0, 0.0, 0.0]
    )
    assert goal.feedback[-1].actual_tcp_speed == pytest.approx([0.0] * 6)
    zeroing_feedback = next(
        feedback for feedback in goal.feedback if feedback.phase == "zeroing_force"
    )
    assert zeroing_feedback.force_bias_valid is True
    assert zeroing_feedback.force_bias == pytest.approx([0.0] * 6)


@pytest.mark.parametrize("part_name", ("SG", "MG", "LG", "SCP", "MCP", "LCP"))
def test_supported_exact_part_executes_same_direct_closed_loop(
    monkeypatch: pytest.MonkeyPatch,
    part_name: str,
) -> None:
    module = _server_module()
    start = ((0.0, 0.5, 1.1), (0.0, 0.0, 0.0, 1.0))
    target = ((0.0, 0.5, 1.09), (0.0, 0.0, 0.0, 1.0))
    server, goal, _events = _insert_execution_harness(
        module,
        monkeypatch,
        force_samples=[
            *([[0.0] * 6] * 5),
            [0.0, 0.0, 4.0, 0.0, 0.0, 0.0],
        ],
        stationary_results=[True, True],
        spiral_radius_m=0.0,
        actual_transforms=[start, start, target, target, target, target],
    )
    selected_caps = module._insert_hard_caps(part_name)
    goal.request.part_name = part_name
    goal.request.hard_caps_sha256 = module._insert_hard_caps_sha256(selected_caps)
    clock = _AdvancingClock()
    module.time.monotonic = clock.monotonic

    result = server._execute_insert(goal)

    assert result.error_code == 0, result.error_string
    assert result.engagement_detected is True
    assert result.seated_detected is True
    assert any(
        status.get("insert_selected_part_name") == part_name
        and status.get("insert_selected_hard_caps_sha256")
        == goal.request.hard_caps_sha256
        for status in server._test_insert_statuses
    )
    assert {feedback.phase for feedback in goal.feedback} == {
        "checking",
        "zeroing_force",
        "seating",
        "settling",
    }
    if part_name != "MG":
        assert selected_caps != module._insert_hard_caps("MG")
        assert "insert_max_disengagement_cycles" not in selected_caps


def test_insert_single_repeated_rtde_timestamp_waits_for_next_sample(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _server_module()
    start = ((0.0, 0.5, 1.1), (0.0, 0.0, 0.0, 1.0))
    target = ((0.0, 0.5, 1.09), (0.0, 0.0, 0.0, 1.0))
    server, goal, events = _insert_execution_harness(
        module,
        monkeypatch,
        force_samples=[*([[0.0] * 6] * 5), [0.0, 0.0, 4.0, 0.0, 0.0, 0.0]],
        stationary_results=[True, True],
        spiral_radius_m=0.0,
        actual_transforms=[start, start, target, target, target, target],
        feedback_timestamps=[
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
            5.0,
            *(float(value) for value in range(6, 100)),
        ],
    )
    clock = _AdvancingClock(step_sec=0.001)
    module.time.monotonic = clock.monotonic

    result = server._execute_insert(goal)

    assert result.error_code == 0
    assert result.state_uncertain is False
    assert result.motion_settled is True
    assert goal.outcomes == ["succeeded"]
    assert not any(event.startswith("reset:") for event in events)


def test_insert_ignores_reverse_depth_jitter_within_start_tolerance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _server_module()
    start = ((0.0, 0.5, 1.1), (0.0, 0.0, 0.0, 1.0))
    reverse_jitter = ((0.0, 0.5, 1.100018), (0.0, 0.0, 0.0, 1.0))
    target = ((0.0, 0.5, 1.09), (0.0, 0.0, 0.0, 1.0))
    server, goal, events = _insert_execution_harness(
        module,
        monkeypatch,
        force_samples=[
            *([[0.0] * 6] * 5),
            [0.0, 0.0, 4.0, 0.0, 0.0, 0.0],
        ],
        stationary_results=[True, True],
        spiral_radius_m=0.0,
        actual_transforms=[
            start,
            reverse_jitter,
            target,
            target,
            target,
            target,
        ],
    )
    clock = _AdvancingClock()
    module.time.monotonic = clock.monotonic

    result = server._execute_insert(goal)

    assert result.error_code == 0, result.error_string
    assert result.limit_trigger == ""
    assert result.seated_detected is True
    assert events.count("forceMode") >= 2
    assert events.count("stop") == 1


def test_derived_timeout_covers_progressive_direct_detection_holds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cais_spade_llm.resources.robot.gazebo_pick_place_controller import (
        derive_move_insert_timeout_sec,
    )

    module = _server_module()
    start = ((0.0, 0.5, 1.1), (0.0, 0.0, 0.0, 1.0))
    target = ((0.0, 0.5, 1.099), (0.0, 0.0, 0.0, 1.0))
    progressive = [
        ((0.0, 0.5, 1.1 - (step * 0.0001)), (0.0, 0.0, 0.0, 1.0))
        for step in range(1, 11)
    ]
    server, goal, _events = _insert_execution_harness(
        module,
        monkeypatch,
        force_samples=[*([[0.0] * 6] * 5), [0.0, 0.0, 4.0, 0.0, 0.0, 0.0]],
        stationary_results=[True, True],
        spiral_radius_m=0.0,
        actual_transforms=[start, start, *progressive, *([target] * 40)],
    )
    goal.request.target_tool0_pose = _insert_pose(module, 1.099)
    goal.request.contact_speed_m_s = 0.01
    goal.request.engagement_progress_m = 0.0002
    profile = {
        field: getattr(goal.request, field)
        for field in (
            "contact_speed_m_s",
            "spiral_radius_m",
            "spiral_pitch_m",
            "spiral_speed_m_s",
            "spiral_acceleration_m_s2",
            "settle_time_sec",
        )
    }
    timeout_sec, timeout_error = derive_move_insert_timeout_sec(
        {"x": 0.0, "y": 0.5, "z": 1.1},
        {"x": 0.0, "y": 0.5, "z": 1.099},
        {"x": 0.0, "y": 0.0, "z": -1.0},
        profile,
    )
    assert timeout_error == ""
    assert timeout_sec > 0.60
    goal.request.timeout_sec = timeout_sec
    clock = _AdvancingClock()
    module.time.monotonic = clock.monotonic

    result = server._execute_insert(goal)

    assert result.error_code == 0
    assert result.engagement_detected is True
    assert result.seated_detected is True


def test_derived_mg_timeout_uses_complete_protected_trial_window() -> None:
    from cais_spade_llm.resources.robot.gazebo_pick_place_controller import (
        derive_move_insert_timeout_sec,
    )

    profile = {
        "contact_speed_m_s": 0.002,
        "spiral_radius_m": 0.002,
        "spiral_pitch_m": 0.0005,
        "spiral_speed_m_s": 0.002,
        "spiral_acceleration_m_s2": 0.02,
        "settle_time_sec": 0.001,
    }

    timeout_sec, timeout_error = derive_move_insert_timeout_sec(
        {"x": 0.0, "y": 0.5, "z": 1.1},
        {"x": 0.0, "y": 0.5, "z": 1.09},
        {"x": 0.0, "y": 0.0, "z": -1.0},
        profile,
        part_name="MG",
        insert_max_timeout_sec=60.0,
    )

    assert timeout_error == ""
    assert timeout_sec == pytest.approx(60.0)


def test_high_axial_profile_exceedance_with_progress_does_not_trigger_relief(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _server_module()
    start = ((0.0, 0.5, 1.1), (0.0, 0.0, 0.0, 1.0))
    target = ((0.0, 0.5, 1.09), (0.0, 0.0, 0.0, 1.0))
    progressive = [
        ((0.0, 0.5, 1.1 - step * 0.0005), (0.0, 0.0, 0.0, 1.0))
        for step in range(1, 21)
    ]
    high_advancing_load = [0.0, 0.0, 9.0, 0.0, 0.0, 0.0]
    final_profile_load = [0.0, 0.0, 6.0, 0.0, 0.0, 0.0]
    server, goal, _events = _insert_execution_harness(
        module,
        monkeypatch,
        force_samples=[
            *([[0.0] * 6] * 5),
            *([high_advancing_load] * 24),
            *([final_profile_load] * 80),
        ],
        stationary_results=[True, True],
        spiral_radius_m=0.0,
        actual_transforms=[start, start, *progressive, *([target] * 100)],
    )
    goal.request.force_depth_axial_upper_n = [7.0] * 16
    clock = _AdvancingClock(step_sec=0.005)
    module.time.monotonic = clock.monotonic

    result = server._execute_insert(goal)

    assert result.error_code == 0, result.error_string
    assert result.peak_filtered_axial_force_n == pytest.approx(9.0)
    assert result.relief_cycle_count == 0
    assert result.soft_overload_detected is False
    assert result.engagement_detected is True
    assert result.seated_detected is True
    assert "relieving" not in [feedback.phase for feedback in goal.feedback]


def test_target_depth_torque_profile_variation_settles_without_disengaging(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _server_module()
    start = ((0.0, 0.5, 1.1), (0.0, 0.0, 0.0, 1.0))
    target = ((0.0, 0.5, 1.09), (0.0, 0.0, 0.0, 1.0))
    progressive = [
        ((0.0, 0.5, 1.1 - step * 0.0005), (0.0, 0.0, 0.0, 1.0))
        for step in range(1, 21)
    ]
    server, goal, _events = _insert_execution_harness(
        module,
        monkeypatch,
        force_samples=[[0.0] * 6],
        stationary_results=[True, True],
        spiral_radius_m=0.0,
        actual_transforms=[
            start,
            start,
            *progressive,
            *([target] * 160),
        ],
    )
    goal.request.force_depth_torque_upper_nm = [0.5] * 16
    goal.request.max_torque_nm = 2.0
    goal.request.settle_time_sec = 0.05
    force_reads = 0

    def read_force() -> list[float]:
        nonlocal force_reads
        force_reads += 1
        if force_reads <= 5:
            return [0.0] * 6
        if 10 <= force_reads <= 20:
            return [0.0, 0.0, 4.0, 0.6, 0.0, 0.0]
        return [0.0, 0.0, 4.0, 0.0, 0.0, 0.0]

    server._read_actual_tcp_force = read_force
    clock = _AdvancingClock(step_sec=0.005)
    module.time.monotonic = clock.monotonic

    result = server._execute_insert(goal)

    assert result.error_code == 0, result.error_string
    assert result.engagement_detected is True
    assert result.seated_detected is True
    assert result.disengagement_cycle_count == 0
    assert "disengaging" not in [feedback.phase for feedback in goal.feedback]
    assert any(
        feedback.phase == "seating"
        and feedback.filtered_torque_nm > feedback.force_depth_torque_upper_nm
        for feedback in goal.feedback
    )
    assert not any(
        float(command[2][2]) < 0.0
        for command in server._test_force_mode_commands
    )


def test_advancing_engagement_latches_above_learned_torque_below_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _server_module()
    server, goal, _events = _insert_execution_harness(
        module,
        monkeypatch,
        force_samples=[[0.0] * 6],
        stationary_results=[True],
        spiral_radius_m=0.0,
    )
    monkeypatch.setattr(module, "UR5E_RTDE_INSERT_SOFT_FILTER_WINDOW_SEC", 0.01)
    monkeypatch.setattr(module, "UR5E_RTDE_INSERT_MAX_SETTLE_TIME_SEC", 0.25)
    goal.request.force_depth_torque_upper_nm = [1.0] * 16
    goal.request.max_torque_nm = 1.0
    goal.request.settle_time_sec = 0.25
    _refresh_request_hard_caps_sha256(module, goal)
    clock = _AdvancingClock(step_sec=0.001)
    module.time.monotonic = clock.monotonic
    force_reads = 0
    insertion_depth_m = 0.0
    candidate_observed_at: float | None = None
    bump_started_at: float | None = None
    bump_ended_at: float | None = None
    engagement_observed_at: float | None = None

    def read_pose() -> Any:
        nonlocal insertion_depth_m
        if server._insert_force_mode_active:
            insertion_depth_m = min(0.01, insertion_depth_m + 0.0001)
        return (
            (0.0, 0.5, 1.1 - insertion_depth_m),
            (0.0, 0.0, 0.0, 1.0),
        )

    def read_force() -> list[float]:
        nonlocal force_reads
        nonlocal candidate_observed_at
        nonlocal bump_started_at
        nonlocal bump_ended_at
        nonlocal engagement_observed_at
        force_reads += 1
        if force_reads <= 5:
            return [0.0] * 6
        if goal.feedback:
            latest_feedback = goal.feedback[-1]
            if latest_feedback.engagement_detected and engagement_observed_at is None:
                engagement_observed_at = clock.value
            if (
                candidate_observed_at is None
                and latest_feedback.contact_detected
                and not latest_feedback.engagement_detected
            ):
                candidate_observed_at = clock.value
        if candidate_observed_at is not None:
            candidate_elapsed_sec = clock.value - candidate_observed_at
            if 0.227 <= candidate_elapsed_sec < 0.272:
                if bump_started_at is None:
                    bump_started_at = clock.value
                return [0.0, 0.0, 4.0, 1.1, 0.0, 0.0]
            if bump_started_at is not None and bump_ended_at is None:
                bump_ended_at = clock.value
        return [0.0, 0.0, 4.0, 0.0, 0.0, 0.0]

    server._read_actual_tcp_transform = read_pose
    server._read_actual_tcp_force = read_force

    result = server._execute_insert(goal)

    assert result.error_code == 0, result.error_string
    assert candidate_observed_at is not None
    assert bump_started_at is not None
    assert bump_ended_at is not None
    assert engagement_observed_at is not None
    assert bump_started_at - candidate_observed_at == pytest.approx(
        0.227,
        abs=0.003,
    )
    assert bump_ended_at - bump_started_at == pytest.approx(0.045, abs=0.003)
    assert bump_started_at < engagement_observed_at < bump_ended_at
    assert result.peak_filtered_torque_nm > goal.request.max_torque_nm
    overload_feedback = [
        feedback
        for feedback in goal.feedback
        if feedback.filtered_torque_nm > goal.request.max_torque_nm
    ]
    assert overload_feedback[-1].insertion_depth_m > (
        overload_feedback[0].insertion_depth_m
    )
    assert result.soft_overload_detected is True
    assert result.relief_cycle_count == 0
    assert result.hard_limit_detected is False
    assert result.engagement_detected is True
    assert result.seated_detected is True
    assert "relieving" not in [feedback.phase for feedback in goal.feedback]


def test_engagement_candidate_survives_quantized_expected_band_plateaus(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _server_module()
    server, goal, _events = _insert_execution_harness(
        module,
        monkeypatch,
        force_samples=[[0.0] * 6],
        stationary_results=[True],
        spiral_radius_m=0.0,
    )
    monkeypatch.setattr(module, "UR5E_RTDE_INSERT_SOFT_FILTER_WINDOW_SEC", 0.01)
    monkeypatch.setattr(module, "UR5E_RTDE_INSERT_MAX_SETTLE_TIME_SEC", 0.25)
    goal.request.force_depth_torque_upper_nm = [1.0] * 16
    goal.request.max_torque_nm = 1.0
    goal.request.settle_time_sec = 0.25
    _refresh_request_hard_caps_sha256(module, goal)
    clock = _AdvancingClock(step_sec=0.001)
    module.time.monotonic = clock.monotonic
    force_reads = 0
    active_pose_reads = 0
    insertion_depth_m = 0.0

    def engagement_observed() -> bool:
        return any(feedback.engagement_detected for feedback in goal.feedback)

    def read_pose() -> Any:
        nonlocal active_pose_reads
        nonlocal insertion_depth_m
        if server._insert_force_mode_active:
            active_pose_reads += 1
            if active_pose_reads % 2:
                insertion_depth_m = min(0.01, insertion_depth_m + 0.0002)
        return (
            (0.0, 0.5, 1.1 - insertion_depth_m),
            (0.0, 0.0, 0.0, 1.0),
        )

    def read_force() -> list[float]:
        nonlocal force_reads
        force_reads += 1
        if force_reads <= 5:
            return [0.0] * 6
        torque_nm = 0.0 if engagement_observed() else 1.1
        return [0.0, 0.0, 4.0, torque_nm, 0.0, 0.0]

    server._read_actual_tcp_transform = read_pose
    server._read_actual_tcp_force = read_force

    result = server._execute_insert(goal)

    assert result.error_code == 0, result.error_string
    assert result.engagement_detected is True
    assert result.seated_detected is True
    assert result.soft_overload_detected is True
    assert result.relief_cycle_count == 0
    expected_band_feedback = [
        feedback
        for feedback in goal.feedback
        if feedback.filtered_torque_nm > goal.request.max_torque_nm
    ]
    assert any(
        current.insertion_depth_m == pytest.approx(previous.insertion_depth_m)
        and not current.axial_progress_stalled
        for previous, current in zip(
            expected_band_feedback,
            expected_band_feedback[1:],
            strict=False,
        )
    )
    assert "relieving" not in [feedback.phase for feedback in goal.feedback]


@pytest.mark.parametrize(
    "abnormal_load",
    (
        [4.0, 0.0, 4.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 4.0, 1.5, 0.0, 0.0],
    ),
)
def test_lateral_or_torque_profile_exceedance_reliefs_during_progress(
    monkeypatch: pytest.MonkeyPatch,
    abnormal_load: list[float],
) -> None:
    module = _server_module()
    start = ((0.0, 0.5, 1.1), (0.0, 0.0, 0.0, 1.0))
    target = ((0.0, 0.5, 1.09), (0.0, 0.0, 0.0, 1.0))
    progressive = [
        ((0.0, 0.5, 1.1 - step * 0.0005), (0.0, 0.0, 0.0, 1.0))
        for step in range(1, 21)
    ]
    normal_load = [0.0, 0.0, 4.0, 0.0, 0.0, 0.0]
    server, goal, _events = _insert_execution_harness(
        module,
        monkeypatch,
        force_samples=[
            *([[0.0] * 6] * 5),
            *([abnormal_load] * 32),
            *([normal_load] * 120),
        ],
        stationary_results=[True, True],
        spiral_radius_m=0.0,
        actual_transforms=[start, start, *progressive, *([target] * 160)],
    )
    goal.request.force_depth_lateral_upper_n = [3.0] * 16
    goal.request.force_depth_torque_upper_nm = [1.0] * 16
    clock = _AdvancingClock(step_sec=0.005)
    module.time.monotonic = clock.monotonic

    result = server._execute_insert(goal)

    assert result.error_code == 0, result.error_string
    assert result.relief_cycle_count == 0
    assert result.soft_overload_detected is False
    assert result.engagement_detected is True
    assert result.seated_detected is True
    assert "relieving" not in [feedback.phase for feedback in goal.feedback]
    profile_field = (
        "lateral_profile_exceeded"
        if abnormal_load[0] != 0.0
        else "torque_profile_exceeded"
    )
    trace_records = [
        json.loads(line)
        for line in Path(result.server_trace_path).read_text().splitlines()
    ]
    assert any(
        record.get("kind") == "sample"
        and record.get(profile_field) is True
        and record.get("axial_progress_stalled") is False
        for record in trace_records
    )


def test_overall_learned_torque_limit_is_expected_band_while_progress_advances(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _server_module()
    start = ((0.0, 0.5, 1.1), (0.0, 0.0, 0.0, 1.0))
    progressive = [
        ((0.0, 0.5, 1.1 - step * 0.0001), (0.0, 0.0, 0.0, 1.0))
        for step in range(1, 61)
    ]
    overall_learned_overload = [0.0, 0.0, 4.0, 3.0, 0.0, 0.0]
    server, goal, _events = _insert_execution_harness(
        module,
        monkeypatch,
        force_samples=[
            *([[0.0] * 6] * 5),
            *([overall_learned_overload] * 160),
        ],
        stationary_results=[True, True],
        spiral_radius_m=0.002,
        actual_transforms=[start, start, *progressive, *([progressive[-1]] * 160)],
    )
    clock = _AdvancingClock(step_sec=0.005)
    module.time.monotonic = clock.monotonic

    result = server._execute_insert(goal)

    assert result.error_code == -5, result.error_string
    assert result.hard_limit_detected is False
    assert result.soft_overload_detected is True
    assert result.relief_cycle_count == 0
    assert "overall learned limit" in result.last_soft_overload_reason
    assert any(
        feedback.filtered_torque_nm > goal.request.max_torque_nm
        and not feedback.axial_progress_stalled
        for feedback in goal.feedback
    )
    assert "relieving" not in [feedback.phase for feedback in goal.feedback]


def test_persistent_guarded_torque_exceedance_reliefs_while_progress_advances(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _server_module()
    start = ((0.0, 0.5, 1.1), (0.0, 0.0, 0.0, 1.0))
    progressive = [
        ((0.0, 0.5, 1.1 - step * 0.0001), (0.0, 0.0, 0.0, 1.0))
        for step in range(1, 101)
    ]
    guarded_overload = [0.0, 0.0, 4.0, 4.95, 0.0, 0.0]
    server, goal, _events = _insert_execution_harness(
        module,
        monkeypatch,
        force_samples=[
            *([[0.0] * 6] * 5),
            *([guarded_overload] * 240),
        ],
        stationary_results=[True, True],
        spiral_radius_m=0.002,
        actual_transforms=[start, start, *progressive, *([progressive[-1]] * 240)],
    )
    goal.request.engagement_progress_m = 0.005
    clock = _AdvancingClock(step_sec=0.005)
    module.time.monotonic = clock.monotonic

    result = server._execute_insert(goal)

    assert result.error_code == -5, result.error_string
    assert result.hard_limit_detected is False
    assert result.soft_overload_detected is True
    assert result.relief_cycle_count == 1
    assert "guarded ceiling" in result.last_soft_overload_reason
    assert result.limit_trigger == "soft_active_tcp_torque_nm"
    assert result.limit_trigger_threshold == pytest.approx(4.9)
    assert any(
        feedback.phase == "relieving" and not feedback.axial_progress_stalled
        for feedback in goal.feedback
    )


def test_stalled_persistent_torque_profile_overload_triggers_relief(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _server_module()
    start = ((0.0, 0.5, 1.1), (0.0, 0.0, 0.0, 1.0))
    contact_pose = ((0.0, 0.5, 1.098), (0.0, 0.0, 0.0, 1.0))
    profile_overload = [0.0, 0.0, 4.0, 1.5, 0.0, 0.0]
    server, goal, _events = _insert_execution_harness(
        module,
        monkeypatch,
        force_samples=[*([[0.0] * 6] * 5), *([profile_overload] * 160)],
        stationary_results=[True, True],
        spiral_radius_m=0.002,
        actual_transforms=[start, *([contact_pose] * 200)],
    )
    goal.request.force_depth_torque_upper_nm = [1.0] * 16
    clock = _AdvancingClock(step_sec=0.005)
    module.time.monotonic = clock.monotonic

    result = server._execute_insert(goal)

    assert result.error_code == -5, result.error_string
    assert result.hard_limit_detected is False
    assert result.soft_overload_detected is True
    assert result.relief_cycle_count == 1
    assert "depth-profile limit" in result.last_soft_overload_reason
    assert any(
        feedback.phase == "relieving" and feedback.axial_progress_stalled
        for feedback in goal.feedback
    )


def test_insert_spiral_engagement_success_returns_complete_final_pose(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _server_module()
    start = ((0.0, 0.5, 1.1), (0.0, 0.0, 0.0, 1.0))
    engaged = ((0.0, 0.5, 1.098), (0.0, 0.0, 0.0, 1.0))
    target = ((0.0, 0.5, 1.09), (0.0, 0.0, 0.0, 1.0))
    contact_force = [0.0, 0.0, 4.0, 0.0, 0.0, 0.0]
    server, goal, events = _insert_execution_harness(
        module,
        monkeypatch,
        force_samples=[*([[0.0] * 6] * 5), contact_force],
        stationary_results=[True, True],
        spiral_radius_m=0.01,
        actual_transforms=[
            start,
            start,
            *([start] * 12),
            *([engaged] * 10),
            *([target] * 12),
        ],
    )
    clock = _AdvancingClock()
    module.time.monotonic = clock.monotonic

    result = server._execute_insert(goal)

    assert result.error_code == 0
    assert result.state_uncertain is False
    assert result.contact_detected is True
    assert result.engagement_detected is True
    assert result.seated_detected is True
    assert result.final_tool0_pose_valid is True
    assert result.final_insertion_depth_m == pytest.approx(0.01)
    assert result.final_depth_error_m == pytest.approx(0.0)
    assert result.final_search_radius_m > 0.0
    final_pose = result.final_tool0_pose
    assert final_pose.header.frame_id == "world"
    assert [
        final_pose.pose.position.x,
        final_pose.pose.position.y,
        final_pose.pose.position.z,
        final_pose.pose.orientation.x,
        final_pose.pose.orientation.y,
        final_pose.pose.orientation.z,
        final_pose.pose.orientation.w,
    ] == pytest.approx([0.0, 0.5, 1.09, 0.0, 0.0, 0.0, 1.0])
    assert goal.outcomes == ["succeeded"]
    assert "moveL" not in events
    assert events.count("stop") == 1
    assert "servoL" not in events
    assert "servoStop" not in events
    assert events.count("forceMode") >= 2
    status_phases = {
        status["insert_phase"]
        for status in server._test_insert_statuses
        if status.get("insert_phase")
    }
    assert status_phases == {
        "checking",
        "zeroing_force",
        "searching",
        "seating",
        "settling",
    }
    assert {feedback.phase for feedback in goal.feedback} == {
        "checking",
        "zeroing_force",
        "searching",
        "seating",
        "settling",
    }


def test_mg_expanded_search_captures_pin_with_four_mm_nominal_xy_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _server_module()
    start = ((0.0, 0.5, 1.1), (0.0, 0.0, 0.0, 1.0))
    server, goal, _events = _insert_execution_harness(
        module,
        monkeypatch,
        force_samples=[[0.0] * 6],
        stationary_results=[True, True],
        spiral_radius_m=0.000001,
        actual_transforms=[start],
    )
    clock = _AdvancingClock(step_sec=0.005)
    module.time.monotonic = clock.monotonic
    insertion_reads = 0
    pin_found = False

    def latest_phase() -> str:
        if not server._test_insert_statuses:
            return ""
        return str(server._test_insert_statuses[-1].get("insert_phase") or "")

    def read_force() -> list[float]:
        if server._insert_force_mode_command is None:
            return [0.0] * 6
        return [0.0, 0.0, 4.0, 0.0, 0.0, 0.0]

    def read_pose() -> Any:
        nonlocal insertion_reads
        nonlocal pin_found
        pin_found = pin_found or latest_phase() == "expanded_searching"
        if not pin_found:
            return start
        insertion_reads += 1
        depth_m = min(0.01, insertion_reads * 0.00075)
        return ((0.004, 0.5, 1.1 - depth_m), (0.0, 0.0, 0.0, 1.0))

    server._read_actual_tcp_force = read_force
    server._read_actual_tcp_transform = read_pose

    result = server._execute_insert(goal)

    assert result.error_code == 0, result.error_string
    assert result.seated_detected is True
    assert result.tactile_center_valid is True
    assert result.tactile_center_tool0_pose.pose.position.x == pytest.approx(0.004)
    assert result.final_lateral_offset_m == pytest.approx(0.004)
    assert "expanded_searching" in {
        feedback.phase for feedback in goal.feedback
    }


def test_mg_expanded_search_stages_three_five_ten_mm_before_no_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _server_module()
    contact_force = [0.0, 0.0, 4.0, 0.0, 0.0, 0.0]
    server, goal, _events = _insert_execution_harness(
        module,
        monkeypatch,
        force_samples=[*([[0.0] * 6] * 5), contact_force],
        stationary_results=[True, True],
        spiral_radius_m=0.000001,
    )
    monkeypatch.setattr(module, "UR5E_RTDE_INSERT_MAX_TIMEOUT_SEC", 60.0)
    goal.request.spiral_pitch_m = 0.005
    goal.request.timeout_sec = 50.0
    _refresh_request_hard_caps_sha256(module, goal)
    clock = _AdvancingClock(step_sec=0.005)
    module.time.monotonic = clock.monotonic

    result = server._execute_insert(goal)

    assert result.error_code == -5, result.error_string
    assert result.error_string.endswith("No pin entry found within 10 mm")
    assert result.explored_search_radius_m < 0.01
    assert result.explored_search_radius_m >= 0.0095
    expanded_radii = [
        feedback.search_radius_m
        for feedback in goal.feedback
        if feedback.phase == "expanded_searching"
    ]
    assert any(0.0025 <= radius_m < 0.003 for radius_m in expanded_radii)
    assert any(0.0045 <= radius_m < 0.005 for radius_m in expanded_radii)
    assert any(radius_m >= 0.0095 for radius_m in expanded_radii)
    assert result.disengagement_cycle_count == 0


def test_insert_target_depth_without_bottom_contact_never_reports_seated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _server_module()
    start = ((0.0, 0.5, 1.1), (0.0, 0.0, 0.0, 1.0))
    target = ((0.0, 0.5, 1.09), (0.0, 0.0, 0.0, 1.0))
    server, goal, events = _insert_execution_harness(
        module,
        monkeypatch,
        force_samples=[[0.0] * 6] * 6,
        stationary_results=[True, True],
        spiral_radius_m=0.0,
        actual_transforms=[start, start, target, target, target, target],
    )
    clock = _AdvancingClock()
    module.time.monotonic = clock.monotonic

    result = server._execute_insert(goal)

    assert result.error_code == -5, result.error_string
    assert result.state_uncertain is False
    assert result.contact_detected is False
    assert result.engagement_detected is False
    assert result.seated_detected is False
    assert "timed out without engagement" in result.error_string
    assert goal.outcomes == ["aborted"]
    assert events.count("stop") == 1


def test_insert_target_depth_with_nonstationary_tcp_speed_never_reports_seated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _server_module()
    start = ((0.0, 0.5, 1.1), (0.0, 0.0, 0.0, 1.0))
    target = ((0.0, 0.5, 1.09), (0.0, 0.0, 0.0, 1.0))
    server, goal, events = _insert_execution_harness(
        module,
        monkeypatch,
        force_samples=[*([[0.0] * 6] * 5), [0.0, 0.0, 4.0, 0.0, 0.0, 0.0]],
        stationary_results=[True, True],
        spiral_radius_m=0.0,
        actual_transforms=[start, start, *([target] * 40)],
        tcp_speed_samples=[[0.0, 0.0, -0.001, 0.0, 0.0, 0.0]],
    )
    clock = _AdvancingClock()
    module.time.monotonic = clock.monotonic

    result = server._execute_insert(goal)

    assert result.error_code == -5
    assert result.state_uncertain is False
    assert result.engagement_detected is True
    assert result.seated_detected is False
    assert "stable seating evidence" in result.error_string
    assert goal.outcomes == ["aborted"]
    assert events.count("stop") == 1


def test_insert_single_force_spike_does_not_establish_contact_or_engagement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _server_module()
    start = ((0.0, 0.5, 1.1), (0.0, 0.0, 0.0, 1.0))
    target = ((0.0, 0.5, 1.09), (0.0, 0.0, 0.0, 1.0))
    force_spike = [0.0, 0.0, 4.0, 0.0, 0.0, 0.0]
    server, goal, events = _insert_execution_harness(
        module,
        monkeypatch,
        force_samples=[
            *([[0.0] * 6] * 5),
            force_spike,
            *([[0.0] * 6] * 5),
        ],
        stationary_results=[True, True],
        spiral_radius_m=0.0,
        actual_transforms=[start, start, target, target, target],
    )
    clock = _AdvancingClock()
    module.time.monotonic = clock.monotonic

    result = server._execute_insert(goal)

    assert result.error_code == -5
    assert result.state_uncertain is False
    assert result.peak_axial_force_n == pytest.approx(4.0)
    assert result.contact_detected is False
    assert result.engagement_detected is False
    assert result.seated_detected is False
    assert goal.outcomes == ["aborted"]
    assert events.count("stop") == 1


def test_insert_tensile_axial_load_never_establishes_contact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _server_module()
    tensile_load = [0.0, 0.0, -12.0, 0.0, 0.0, 0.0]
    server, goal, _events = _insert_execution_harness(
        module,
        monkeypatch,
        force_samples=[*([[0.0] * 6] * 5), tensile_load],
        stationary_results=[True, True],
        spiral_radius_m=0.0,
    )
    clock = _AdvancingClock(step_sec=0.005)
    module.time.monotonic = clock.monotonic

    result = server._execute_insert(goal)

    assert result.error_code == -5
    assert result.peak_axial_force_n == pytest.approx(12.0), [
        (feedback.phase, getattr(feedback, "raw_axial_force_n", None))
        for feedback in goal.feedback
    ]
    assert result.contact_detected is False
    insertion_feedback = [
        feedback for feedback in goal.feedback if feedback.phase == "seating"
    ]
    assert insertion_feedback
    assert all(feedback.axial_force_n == pytest.approx(0.0) for feedback in insertion_feedback)
    assert any(
        feedback.raw_axial_force_n == pytest.approx(12.0)
        for feedback in insertion_feedback
    )


def test_sustained_soft_overload_unloads_and_resumes_axially_without_spiral(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _server_module()
    start = ((0.0, 0.5, 1.1), (0.0, 0.0, 0.0, 1.0))
    contact_pose = ((0.0, 0.5, 1.098), (0.0, 0.0, 0.0, 1.0))
    overload = [6.0, 0.0, 4.0, 0.0, 0.0, 0.0]
    cleared = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    server, goal, _events = _insert_execution_harness(
        module,
        monkeypatch,
        force_samples=[
            *([[0.0] * 6] * 5),
            *([overload] * 15),
            *([cleared] * 120),
        ],
        stationary_results=[True, True],
        actual_transforms=[start, *([contact_pose] * 240)],
    )
    monkeypatch.setattr(module, "UR5E_RTDE_INSERT_RELIEF_UNLOAD_DWELL_SEC", 0.25)
    monkeypatch.setattr(module, "UR5E_RTDE_INSERT_RELIEF_TIMEOUT_SEC", 0.6)
    _refresh_request_hard_caps_sha256(module, goal)
    clock = _AdvancingClock(step_sec=0.005)
    module.time.monotonic = clock.monotonic

    result = server._execute_insert(goal)

    assert result.error_code == -5, result.error_string
    assert result.soft_overload_detected is True
    assert result.soft_overload_recovered is True
    assert result.relief_cycle_count == 1
    phases = [feedback.phase for feedback in goal.feedback]
    assert "relieving" in phases
    assert "resuming" in phases
    assert "backing_off" not in phases
    resuming_feedback = [
        feedback for feedback in goal.feedback if feedback.phase == "resuming"
    ]
    assert resuming_feedback
    assert all(
        feedback.commanded_lateral_force_x_n == pytest.approx(0.0)
        and feedback.commanded_lateral_force_y_n == pytest.approx(0.0)
        for feedback in resuming_feedback
    )


@pytest.mark.parametrize(
    ("retreat_m", "stationary_retreat_m", "clear_during_backoff"),
    [(0.0005, 0.00065, False), (0.0002, 0.00025, True)],
)
def test_backoff_stops_on_step_or_early_clear_then_restarts_reduced_force_mode(
    monkeypatch: pytest.MonkeyPatch,
    retreat_m: float,
    stationary_retreat_m: float,
    clear_during_backoff: bool,
) -> None:
    module = _server_module()
    start = ((0.0, 0.5, 1.1), (0.0, 0.0, 0.0, 1.0))
    contact_depth_m = 0.004
    contact_pose = (
        (0.0, 0.5, 1.1 - contact_depth_m),
        (0.0, 0.0, 0.0, 1.0),
    )
    retreat_pose = (
        (0.0, 0.5, 1.1 - (contact_depth_m - retreat_m)),
        (0.0, 0.0, 0.0, 1.0),
    )
    stationary_retreat_pose = (
        (0.0, 0.5, 1.1 - (contact_depth_m - stationary_retreat_m)),
        (0.0, 0.0, 0.0, 1.0),
    )
    server, goal, events = _insert_execution_harness(
        module,
        monkeypatch,
        force_samples=[[0.0] * 6],
        stationary_results=[True, True, True],
    )
    force_reads = 0
    saw_reverse = False

    def read_force() -> list[float]:
        nonlocal force_reads
        force_reads += 1
        if force_reads <= 5:
            return [0.0] * 6
        command = server._insert_force_mode_command
        reverse_active = bool(command is not None and float(command[2][2]) < 0.0)
        if events.count("stop") >= 1:
            return [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
        if reverse_active and clear_during_backoff:
            return [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
        return [6.0, 0.0, 4.0, 0.0, 0.0, 0.0]

    def read_pose() -> Any:
        nonlocal saw_reverse
        command = server._insert_force_mode_command
        if command is not None and float(command[2][2]) < 0.0:
            saw_reverse = True
        if saw_reverse and command is None:
            return stationary_retreat_pose
        if saw_reverse:
            return retreat_pose
        if command is not None:
            return contact_pose
        return start

    server._read_actual_tcp_force = read_force
    server._read_actual_tcp_transform = read_pose
    clock = _AdvancingClock(step_sec=0.005)
    module.time.monotonic = clock.monotonic

    result = server._execute_insert(goal)

    assert result.error_code == -5, result.error_string
    assert result.relief_cycle_count == 1
    assert result.relief_force_mode_stop_acknowledged is True
    assert result.relief_stop_l_command_completed is True
    assert result.relief_stationary_confirmed is True
    assert result.relief_force_mode_restart_acknowledged is True
    assert result.relief_backoff_m == pytest.approx(stationary_retreat_m)
    assert result.total_relief_backoff_m == pytest.approx(stationary_retreat_m)
    assert events.count("stop") == 2
    records = [
        json.loads(line)
        for line in Path(result.server_trace_path).read_text().splitlines()
    ]
    restart_record = next(
        record for record in records if record["kind"] == "relief_restart"
    )
    assert restart_record["relief_force_mode_restart_acknowledged"] is True
    backoff_record = next(
        record for record in records if record["kind"] == "relief_backoff_complete"
    )
    assert backoff_record["relief_backoff_m"] == pytest.approx(
        stationary_retreat_m
    )
    assert backoff_record["total_relief_backoff_m"] == pytest.approx(
        stationary_retreat_m
    )
    assert backoff_record["relief_stationary_confirmed"] is True


@pytest.mark.parametrize("shallow_depth_m", [0.0025, 0.00249])
def test_shallow_contact_never_receives_reverse_force_past_original_start(
    monkeypatch: pytest.MonkeyPatch,
    shallow_depth_m: float,
) -> None:
    module = _server_module()
    start = ((0.0, 0.5, 1.1), (0.0, 0.0, 0.0, 1.0))
    shallow_contact = (
        (0.0, 0.5, 1.1 - shallow_depth_m),
        (0.0, 0.0, 0.0, 1.0),
    )
    overload = [6.0, 0.0, 4.0, 0.0, 0.0, 0.0]
    server, goal, _events = _insert_execution_harness(
        module,
        monkeypatch,
        force_samples=[*([[0.0] * 6] * 5), overload],
        stationary_results=[True, True],
        actual_transforms=[start, *([shallow_contact] * 80)],
    )
    clock = _AdvancingClock(step_sec=0.005)
    module.time.monotonic = clock.monotonic

    result = server._execute_insert(goal)

    assert result.error_code == -5
    assert "made no protected withdrawal progress" in result.error_string
    assert "before contact cleared" in result.error_string
    assert result.disengagement_cycle_count == 1
    assert any(
        float(command[2][2]) < 0.0
        for command in server._test_force_mode_commands
    )


def test_search_overload_resumes_direct_then_returns_to_frozen_spiral(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _server_module()
    start = ((0.0, 0.5, 1.1), (0.0, 0.0, 0.0, 1.0))
    contact_pose = ((0.0, 0.5, 1.098), (0.0, 0.0, 0.0, 1.0))
    server, goal, _events = _insert_execution_harness(
        module,
        monkeypatch,
        force_samples=[[0.0] * 6],
        stationary_results=[True, True],
        spiral_radius_m=0.01,
        actual_transforms=[start, *([contact_pose] * 240)],
    )
    force_reads = 0

    def read_force() -> list[float]:
        nonlocal force_reads
        force_reads += 1
        if force_reads <= 5:
            return [0.0] * 6
        phases = [
            status.get("insert_phase") for status in server._test_insert_statuses
        ]
        if "relieving" in phases:
            return [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
        if "searching" in phases:
            return [6.0, 0.0, 4.0, 0.0, 0.0, 0.0]
        return [0.0, 0.0, 4.0, 0.0, 0.0, 0.0]

    server._read_actual_tcp_force = read_force
    monkeypatch.setattr(module, "UR5E_RTDE_INSERT_RELIEF_UNLOAD_DWELL_SEC", 0.25)
    monkeypatch.setattr(module, "UR5E_RTDE_INSERT_RELIEF_TIMEOUT_SEC", 0.6)
    _refresh_request_hard_caps_sha256(module, goal)
    clock = _AdvancingClock(step_sec=0.005)
    module.time.monotonic = clock.monotonic

    result = server._execute_insert(goal)

    assert result.error_code == -5, result.error_string
    assert result.relief_resume_phase == "searching"
    phases = [feedback.phase for feedback in goal.feedback]
    relieving_index = phases.index("relieving")
    resuming_index = phases.index("resuming")
    post_resume_search_index = phases.index("searching", resuming_index + 1)
    pre_relief_search_radii = [
        feedback.search_radius_m
        for feedback in goal.feedback[:relieving_index]
        if feedback.phase == "searching"
    ]
    assert pre_relief_search_radii
    assert (
        goal.feedback[post_resume_search_index].search_radius_m
        >= pre_relief_search_radii[-1] - 1e-12
    )
    assert all(
        feedback.commanded_lateral_force_x_n == pytest.approx(0.0)
        and feedback.commanded_lateral_force_y_n == pytest.approx(0.0)
        for feedback in goal.feedback[resuming_index:post_resume_search_index]
        if feedback.phase == "resuming"
    )


def test_mg_relaxed_axial_profile_limit_does_not_trigger_relief(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _server_module()
    start = ((0.0, 0.5, 1.1), (0.0, 0.0, 0.0, 1.0))
    contact_pose = ((0.0, 0.5, 1.098), (0.0, 0.0, 0.0, 1.0))
    overload = [0.0, 0.0, 8.0, 0.0, 0.0, 0.0]
    server, goal, _events = _insert_execution_harness(
        module,
        monkeypatch,
        force_samples=[*([[0.0] * 6] * 5), overload],
        stationary_results=[True, True],
        spiral_radius_m=0.0002,
        actual_transforms=[start, *([contact_pose] * 160)],
    )
    goal.request.force_depth_axial_upper_n = [5.0] * 16
    clock = _AdvancingClock(step_sec=0.005)
    module.time.monotonic = clock.monotonic

    result = server._execute_insert(goal)

    assert result.error_code == -5, result.error_string
    assert "timed out without engagement" in result.error_string
    assert result.relief_cycle_count == 0
    assert result.soft_overload_detected is False
    assert "relieving" not in [feedback.phase for feedback in goal.feedback]
    assert 0.0 < result.final_search_radius_m <= 0.02
    searching_feedback = [
        feedback
        for feedback in goal.feedback
        if feedback.phase in {"searching", "expanded_searching"}
    ]
    assert searching_feedback
    assert any(
        abs(feedback.commanded_lateral_force_x_n) > 0.0
        or abs(feedback.commanded_lateral_force_y_n) > 0.0
        for feedback in searching_feedback
    )


def test_fourth_soft_overload_is_blocked_after_three_relief_cycles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _server_module()
    start = ((0.0, 0.5, 1.1), (0.0, 0.0, 0.0, 1.0))
    initial_depth_m = 0.004
    server, goal, events = _insert_execution_harness(
        module,
        monkeypatch,
        force_samples=[[0.0] * 6],
        stationary_results=[True] * 6,
    )
    monkeypatch.setattr(module, "UR5E_RTDE_INSERT_MAX_TIMEOUT_SEC", 5.0)
    goal.request.timeout_sec = 3.0
    _refresh_request_hard_caps_sha256(module, goal)
    force_reads = 0
    total_retreat_m = 0.0
    reverse_active_last = False

    def read_force() -> list[float]:
        nonlocal force_reads
        force_reads += 1
        if force_reads <= 5:
            return [0.0] * 6
        return [6.0, 0.0, 4.0, 0.0, 0.0, 0.0]

    def read_pose() -> Any:
        nonlocal total_retreat_m
        nonlocal reverse_active_last
        command = server._insert_force_mode_command
        reverse_active = bool(command is not None and float(command[2][2]) < 0.0)
        if reverse_active and not reverse_active_last:
            total_retreat_m += 0.0005
        reverse_active_last = reverse_active
        if command is None and total_retreat_m == 0.0:
            return start
        depth_m = initial_depth_m - total_retreat_m
        return ((0.0, 0.5, 1.1 - depth_m), (0.0, 0.0, 0.0, 1.0))

    server._read_actual_tcp_force = read_force
    server._read_actual_tcp_transform = read_pose
    clock = _AdvancingClock(step_sec=0.002)
    module.time.monotonic = clock.monotonic

    result = server._execute_insert(goal)

    assert result.error_code == -5, result.error_string
    assert "made no protected withdrawal progress" in result.error_string
    assert "before contact cleared" in result.error_string
    assert result.relief_cycle_count == 3
    assert result.disengagement_cycle_count == 1
    assert result.total_relief_backoff_m == pytest.approx(0.0015)
    assert total_retreat_m == pytest.approx(0.002)
    assert events.count("stop") == 4


def test_stuck_contact_disengagement_times_out_before_global_deadline_and_stops(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _server_module()
    start = ((0.0, 0.5, 1.1), (0.0, 0.0, 0.0, 1.0))
    stuck_contact = ((0.0, 0.5, 1.096), (0.0, 0.0, 0.0, 1.0))
    server, goal, events = _insert_execution_harness(
        module,
        monkeypatch,
        force_samples=[[0.0] * 6],
        stationary_results=[True],
        spiral_radius_m=0.002,
    )
    monkeypatch.setattr(module, "UR5E_RTDE_INSERT_MAX_TIMEOUT_SEC", 5.0)
    goal.request.timeout_sec = 5.0
    _refresh_request_hard_caps_sha256(module, goal)
    clock = _AdvancingClock(step_sec=0.002)
    module.time.monotonic = clock.monotonic
    force_reads = 0
    insertion_started = False

    def read_pose() -> Any:
        nonlocal insertion_started
        insertion_started = insertion_started or server._insert_force_mode_active
        return stuck_contact if insertion_started else start

    def read_force() -> list[float]:
        nonlocal force_reads
        force_reads += 1
        if force_reads <= 5:
            return [0.0] * 6
        return [6.0, 0.0, 4.0, 0.0, 0.0, 0.0]

    server._read_actual_tcp_transform = read_pose
    server._read_actual_tcp_force = read_force

    result = server._execute_insert(goal)

    assert result.error_code == -5, result.error_string
    assert "MG disengagement made no protected withdrawal progress" in (
        result.error_string
    )
    assert "before contact cleared" in result.error_string
    timeout_match = re.search(
        r"withdrawal progress for ([0-9.]+) s",
        result.error_string,
    )
    assert timeout_match is not None
    assert float(timeout_match.group(1)) == pytest.approx(
        module.UR5E_RTDE_INSERT_RELIEF_TIMEOUT_SEC,
        abs=0.01,
    )
    assert clock.value < goal.request.timeout_sec / 2.0
    assert result.soft_overload_detected is True
    assert result.relief_cycle_count == 1
    assert result.disengagement_cycle_count == 1
    assert result.disengagement_contact_cleared is False
    assert result.disengagement_withdrawal_m == pytest.approx(0.0)
    assert result.hard_limit_detected is False
    assert result.motion_settled is True
    assert result.stationary_confirmed is True
    assert result.state_uncertain is False
    assert events.count("stop") == 1
    terminal_stationary_index = max(
        index for index, event in enumerate(events) if event == "stationary"
    )
    assert events.index("stop") < terminal_stationary_index < events.index("finish")


def test_slow_disengagement_progress_over_one_second_completes_recovery(  # noqa: C901
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _server_module()
    start = ((0.0, 0.5, 1.1), (0.0, 0.0, 0.0, 1.0))
    server, goal, events = _insert_execution_harness(
        module,
        monkeypatch,
        force_samples=[[0.0] * 6],
        stationary_results=[True] * 20,
        actual_transforms=[start],
    )
    monkeypatch.setattr(module, "UR5E_RTDE_INSERT_MAX_TIMEOUT_SEC", 10.0)
    goal.request.timeout_sec = 8.0
    _refresh_request_hard_caps_sha256(module, goal)
    clock = _AdvancingClock(step_sec=0.002)
    module.time.monotonic = clock.monotonic
    original_confirm_stationary = server._confirm_stationary_after_stop
    stationary_confirmation_count = 0

    def confirm_stationary_with_disengagement_delay(**kwargs: Any) -> bool:
        nonlocal stationary_confirmation_count
        stationary_confirmation_count += 1
        if stationary_confirmation_count == 2:
            clock.value += module.UR5E_RTDE_INSERT_RELIEF_TIMEOUT_SEC + 0.05
        return bool(original_confirm_stationary(**kwargs))

    server._confirm_stationary_after_stop = (
        confirm_stationary_with_disengagement_delay
    )
    force_reads = 0
    insertion_depth_m = 0.0
    withdrawal_depth_m = 0.004
    retry_depth_m = 0.0
    disengagement_observed_at: float | None = None
    retry_observed_at: float | None = None

    def latest_status() -> dict[str, Any]:
        return server._test_insert_statuses[-1] if server._test_insert_statuses else {}

    def engagement_observed() -> bool:
        return any(feedback.engagement_detected for feedback in goal.feedback)

    def read_pose() -> Any:
        nonlocal insertion_depth_m
        nonlocal withdrawal_depth_m
        nonlocal retry_depth_m
        nonlocal disengagement_observed_at
        nonlocal retry_observed_at
        status = latest_status()
        phase = str(status.get("insert_phase") or "")
        cycle = int(status.get("insert_disengagement_cycle_count") or 0)
        if cycle == 0:
            if engagement_observed():
                insertion_depth_m = 0.004
            elif server._insert_force_mode_active:
                insertion_depth_m = min(0.006, insertion_depth_m + 0.0003)
            depth_m = insertion_depth_m
        elif phase == "cocked":
            if disengagement_observed_at is None:
                disengagement_observed_at = clock.value
            depth_m = withdrawal_depth_m
        elif phase == "disengaging":
            withdrawal_depth_m = max(0.0, withdrawal_depth_m - 0.00003)
            depth_m = withdrawal_depth_m
        elif phase in {"recentering", "retaring"}:
            depth_m = 0.0
        else:
            if phase == "retrying" and retry_observed_at is None:
                retry_observed_at = clock.value
            retry_depth_m = min(0.01, retry_depth_m + 0.0005)
            depth_m = retry_depth_m
        return ((0.0, 0.5, 1.1 - depth_m), (0.0, 0.0, 0.0, 1.0))

    def read_force() -> list[float]:
        nonlocal force_reads
        force_reads += 1
        if force_reads <= 5:
            return [0.0] * 6
        status = latest_status()
        phase = str(status.get("insert_phase") or "")
        cycle = int(status.get("insert_disengagement_cycle_count") or 0)
        if (
            cycle > 0
            and phase in {"cocked", "disengaging"}
            and withdrawal_depth_m <= 0.001
        ):
            return [0.0] * 6
        if phase in {"recentering", "retaring"}:
            return [0.0] * 6
        return [0.0, 0.0, 4.0, 0.0, 0.0, 0.0]

    server._read_actual_tcp_transform = read_pose
    server._read_actual_tcp_force = read_force

    result = server._execute_insert(goal)

    assert result.error_code == 0, result.error_string
    assert disengagement_observed_at is not None
    assert retry_observed_at is not None
    assert retry_observed_at - disengagement_observed_at > 1.0
    assert result.disengagement_cycle_count == 1
    assert result.disengagement_contact_cleared is True
    assert result.retare_baseline_consistent is True
    assert "servoL" in events


def test_final_partial_relief_cycle_stops_at_remaining_total_retreat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _server_module()
    start = ((0.0, 0.5, 1.1), (0.0, 0.0, 0.0, 1.0))
    initial_depth_m = 0.006
    server, goal, events = _insert_execution_harness(
        module,
        monkeypatch,
        force_samples=[[0.0] * 6],
        stationary_results=[True] * 6,
    )
    exact_mg_caps = dict(module.UR5E_RTDE_INSERT_MG_HARD_CAPS)
    exact_mg_caps["insert_max_relief_retreat_m"] = 0.0012
    monkeypatch.setattr(module, "UR5E_RTDE_INSERT_MG_HARD_CAPS", exact_mg_caps)
    monkeypatch.setattr(module, "UR5E_RTDE_INSERT_MAX_TIMEOUT_SEC", 5.0)
    _refresh_request_hard_caps_sha256(module, goal)
    goal.request.timeout_sec = 3.0
    force_reads = 0
    reverse_cycle_count = 0
    total_retreat_m = 0.0
    reverse_active_last = False
    cycle_retreats_m = [0.0005, 0.0005, 0.0002, 0.0048]

    def read_force() -> list[float]:
        nonlocal force_reads
        force_reads += 1
        if force_reads <= 5:
            return [0.0] * 6
        return [6.0, 0.0, 4.0, 0.0, 0.0, 0.0]

    def read_pose() -> Any:
        nonlocal reverse_active_last
        nonlocal reverse_cycle_count
        nonlocal total_retreat_m
        command = server._insert_force_mode_command
        reverse_active = bool(command is not None and float(command[2][2]) < 0.0)
        if reverse_active and not reverse_active_last:
            total_retreat_m += cycle_retreats_m[reverse_cycle_count]
            reverse_cycle_count += 1
        reverse_active_last = reverse_active
        if command is None and total_retreat_m == 0.0:
            return start
        depth_m = initial_depth_m - total_retreat_m
        return ((0.0, 0.5, 1.1 - depth_m), (0.0, 0.0, 0.0, 1.0))

    server._read_actual_tcp_force = read_force
    server._read_actual_tcp_transform = read_pose
    clock = _AdvancingClock(step_sec=0.002)
    module.time.monotonic = clock.monotonic

    result = server._execute_insert(goal)

    assert result.error_code == -5, result.error_string
    assert "made no protected withdrawal progress" in result.error_string
    assert result.relief_cycle_count == 3
    assert result.disengagement_cycle_count == 1
    assert result.relief_planned_backoff_m == pytest.approx(0.0002)
    assert result.total_relief_backoff_m == pytest.approx(0.0012)
    assert total_retreat_m == pytest.approx(0.006)
    assert reverse_cycle_count == 4
    assert result.disengagement_contact_cleared is False
    assert events.count("stop") == 4
    assert all(
        feedback.commanded_axial_force_n == pytest.approx(0.0)
        for feedback in goal.feedback
        if feedback.phase == "cocked"
    )
    records = [
        json.loads(line)
        for line in Path(result.server_trace_path).read_text().splitlines()
    ]
    final_cycle_samples = [
        record
        for record in records
        if record["kind"] == "sample"
        and record.get("phase") == "backing_off"
        and record.get("relief_cycle_count") == 3
    ]
    assert any(
        record["relief_planned_backoff_m"] == pytest.approx(0.0002)
        and record["relief_backoff_m"] == pytest.approx(0.0002)
        for record in final_cycle_samples
    )
    final_backoff = next(
        record
        for record in records
        if record["kind"] == "relief_backoff_complete"
        and record["relief_cycle_count"] == 3
    )
    assert final_backoff["relief_planned_backoff_m"] == pytest.approx(0.0002)
    assert final_backoff["total_relief_backoff_m"] == pytest.approx(0.0012)


def test_insert_transient_pose_progress_does_not_establish_engagement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _server_module()
    start = ((0.0, 0.5, 1.1), (0.0, 0.0, 0.0, 1.0))
    transient_progress = ((0.0, 0.5, 1.098), (0.0, 0.0, 0.0, 1.0))
    contact_force = [0.0, 0.0, 4.0, 0.0, 0.0, 0.0]
    server, goal, events = _insert_execution_harness(
        module,
        monkeypatch,
        force_samples=[*([[0.0] * 6] * 5), contact_force],
        stationary_results=[True, True],
        spiral_radius_m=0.01,
        actual_transforms=[
            start,
            start,
            *([start] * 12),
            transient_progress,
            start,
        ],
    )
    clock = _AdvancingClock()
    module.time.monotonic = clock.monotonic

    result = server._execute_insert(goal)

    assert result.error_code == -5
    assert result.state_uncertain is False
    assert result.contact_detected is True
    assert result.engagement_detected is False
    assert result.seated_detected is False
    assert goal.outcomes == ["aborted"]
    assert events.count("stop") == 1


def test_insert_engagement_must_persist_without_rebound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _server_module()
    start = ((0.0, 0.5, 1.1), (0.0, 0.0, 0.0, 1.0))
    engaged = ((0.0, 0.5, 1.098), (0.0, 0.0, 0.0, 1.0))
    contact_force = [0.0, 0.0, 4.0, 0.0, 0.0, 0.0]
    server, goal, events = _insert_execution_harness(
        module,
        monkeypatch,
        force_samples=[*([[0.0] * 6] * 5), contact_force],
        stationary_results=[True, True],
        spiral_radius_m=0.01,
        actual_transforms=[
            start,
            start,
            *([start] * 12),
            *([engaged] * 10),
            start,
        ],
    )
    clock = _AdvancingClock()
    module.time.monotonic = clock.monotonic

    result = server._execute_insert(goal)

    assert result.error_code == -5
    assert result.state_uncertain is False
    assert result.engagement_detected is True
    assert result.seated_detected is False
    assert "engagement rebounded" in result.last_disengagement_reason
    assert "made no protected withdrawal progress" in result.error_string
    assert result.disengagement_contact_cleared is False
    assert goal.outcomes == ["aborted"]
    assert events.count("stop") == 1


def test_mg_cocking_recovery_withdraws_recenters_retares_and_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _server_module()
    start = ((0.0, 0.5, 1.1), (0.0, 0.0, 0.0, 1.0))
    server, goal, events = _insert_execution_harness(
        module,
        monkeypatch,
        force_samples=[[0.0] * 6],
        stationary_results=[True] * 20,
        actual_transforms=[start],
    )
    clock = _AdvancingClock(step_sec=0.005)
    module.time.monotonic = clock.monotonic
    after_engagement_reads = 0
    retry_reads = 0

    def latest_status() -> dict[str, Any]:
        return server._test_insert_statuses[-1] if server._test_insert_statuses else {}

    def feedback_detected(field_name: str) -> bool:
        return any(bool(getattr(item, field_name, False)) for item in goal.feedback)

    def read_force() -> list[float]:
        phase = str(latest_status().get("insert_phase") or "")
        if server._insert_force_mode_command is None or phase in {
            "cocked",
            "disengaging",
            "recentering",
            "retaring",
        }:
            return [0.0] * 6
        return [0.0, 0.0, 4.0, 0.0, 0.0, 0.0]

    def read_pose() -> Any:
        nonlocal after_engagement_reads
        nonlocal retry_reads
        current = latest_status()
        phase = str(current.get("insert_phase") or "")
        cycle = int(current.get("insert_disengagement_cycle_count") or 0)
        if cycle == 0:
            if feedback_detected("engagement_detected"):
                after_engagement_reads += 1
                depth_m = 0.004 if after_engagement_reads <= 2 else 0.0
            elif feedback_detected("contact_detected"):
                depth_m = 0.004
            else:
                depth_m = 0.001
        elif phase in {"cocked", "disengaging"}:
            return (
                (0.010007, 0.5, 1.1),
                (0.0, 0.0, 0.0, 1.0),
            )
        elif phase in {"recentering", "retaring"}:
            depth_m = 0.0
        else:
            retry_reads += 1
            depth_m = min(0.01, retry_reads * 0.00075)
        return ((0.0, 0.5, 1.1 - depth_m), (0.0, 0.0, 0.0, 1.0))

    server._read_actual_tcp_force = read_force
    server._read_actual_tcp_transform = read_pose
    goal.request.timeout_sec = 1.0

    result = server._execute_insert(goal)

    assert result.error_code == 0, result.error_string
    assert result.disengagement_cycle_count == 1
    assert result.disengagement_contact_cleared is True
    assert result.disengagement_force_mode_stop_acknowledged is True
    assert result.disengagement_stationary_confirmed is True
    assert result.recenter_command_acknowledged is True
    assert result.recenter_position_error_m == pytest.approx(0.0)
    assert result.retare_baseline_consistent is True
    assert result.tactile_center_valid is True
    assert 0.0 < result.tactile_center_confidence <= 1.0
    assert len(result.tactile_center_evidence_sha256) == 64
    phases = {
        str(status.get("insert_phase") or "")
        for status in server._test_insert_statuses
    }
    assert {
        "cocked",
        "disengaging",
        "recentering",
        "retaring",
        "retrying",
        "settling",
    } <= phases
    assert "servoL" in events
    assert "moveL" not in events
    source = inspect.getsource(module.UR5eRTDETrajectoryServer._execute_insert)
    assert "move_home" not in source
    trace_records = [
        json.loads(line)
        for line in Path(result.server_trace_path).read_text().splitlines()
    ]
    assert max(
        float(record.get("lateral_offset_m", 0.0))
        for record in trace_records
        if record.get("kind") == "sample"
    ) == pytest.approx(0.010007)
    assert all(
        float(record.get("insertion_depth_m", 0.0)) >= -1e-12
        for record in trace_records
        if record.get("kind") == "sample"
    )


def test_disengagement_clear_remains_false_when_retare_rejects_contact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _server_module()
    start = ((0.0, 0.5, 1.1), (0.0, 0.0, 0.0, 1.0))
    server, goal, _events = _insert_execution_harness(
        module,
        monkeypatch,
        force_samples=[[0.0] * 6],
        stationary_results=[True] * 20,
        actual_transforms=[start],
    )
    clock = _AdvancingClock(step_sec=0.005)
    module.time.monotonic = clock.monotonic
    after_engagement_reads = 0

    def latest_status() -> dict[str, Any]:
        return server._test_insert_statuses[-1] if server._test_insert_statuses else {}

    def feedback_detected(field_name: str) -> bool:
        return any(bool(getattr(item, field_name, False)) for item in goal.feedback)

    def read_pose() -> Any:
        nonlocal after_engagement_reads
        status = latest_status()
        phase = str(status.get("insert_phase") or "")
        cycle = int(status.get("insert_disengagement_cycle_count") or 0)
        if cycle > 0 and phase in {
            "cocked",
            "disengaging",
            "recentering",
            "retaring",
        }:
            return start
        if feedback_detected("engagement_detected"):
            after_engagement_reads += 1
            depth_m = 0.004 if after_engagement_reads <= 2 else 0.0
        elif feedback_detected("contact_detected"):
            depth_m = 0.004
        else:
            depth_m = 0.001
        return ((0.0, 0.5, 1.1 - depth_m), (0.0, 0.0, 0.0, 1.0))

    def read_force() -> list[float]:
        phase = str(latest_status().get("insert_phase") or "")
        if phase == "retaring":
            return [0.0, 0.0, 4.0, 0.0, 0.0, 0.0]
        if server._insert_force_mode_command is None or phase in {
            "cocked",
            "disengaging",
            "recentering",
        }:
            return [0.0] * 6
        return [0.0, 0.0, 4.0, 0.0, 0.0, 0.0]

    server._read_actual_tcp_transform = read_pose
    server._read_actual_tcp_force = read_force

    result = server._execute_insert(goal)

    assert result.error_code == -5, result.error_string
    assert "may have shifted in RG2" in result.error_string
    assert result.retare_baseline_consistent is False
    assert result.disengagement_contact_cleared is False
    assert not any(
        bool(status.get("insert_disengagement_contact_cleared"))
        for status in server._test_insert_statuses
    )


def test_mg_cocking_recovery_returns_from_partial_contact_to_exact_start_depth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _server_module()
    start = ((0.0, 0.5, 1.1), (0.0, 0.0, 0.0, 1.0))
    cocked = ((0.008, 0.5, 1.0926), (0.0, 0.0, 0.0, 1.0))
    axial_clear = ((0.008, 0.5, 1.1), (0.0, 0.0, 0.0, 1.0))
    server, goal, events = _insert_execution_harness(
        module,
        monkeypatch,
        force_samples=[[0.0] * 6],
        stationary_results=[True] * 20,
        actual_transforms=[start],
    )
    monkeypatch.setattr(module, "UR5E_RTDE_INSERT_MAX_TIMEOUT_SEC", 10.0)
    goal.request.target_tool0_pose = _insert_pose(module, 1.08)
    goal.request.timeout_sec = 8.0
    goal.request.baseline_torque_uncertainty_nm = 0.094
    _refresh_request_hard_caps_sha256(module, goal)
    clock = _AdvancingClock(step_sec=0.005)
    module.time.monotonic = clock.monotonic
    after_engagement_reads = 0
    retry_reads = 0
    axial_withdrawal_started = False
    servo_targets: list[list[float]] = []

    def latest_status() -> dict[str, Any]:
        return server._test_insert_statuses[-1] if server._test_insert_statuses else {}

    def feedback_detected(field_name: str) -> bool:
        return any(bool(getattr(item, field_name, False)) for item in goal.feedback)

    def servo_l(pose: list[float], *_args: Any, **_kwargs: Any) -> bool:
        nonlocal axial_withdrawal_started
        servo_targets.append(list(pose))
        events.append("servoL")
        if float(pose[0]) > 0.004:
            axial_withdrawal_started = True
        return True

    def read_force() -> list[float]:
        phase = str(latest_status().get("insert_phase") or "")
        if phase == "retaring":
            return [0.0, 0.0, 0.0, 0.114, 0.0, 0.0]
        if server._insert_force_mode_command is None or phase in {
            "cocked",
            "disengaging",
            "recentering",
        }:
            return [0.0] * 6
        return [0.0, 0.0, 4.0, 0.0, 0.0, 0.0]

    def read_pose() -> Any:
        nonlocal after_engagement_reads
        nonlocal retry_reads
        current = latest_status()
        phase = str(current.get("insert_phase") or "")
        cycle = int(current.get("insert_disengagement_cycle_count") or 0)
        if cycle == 0:
            if feedback_detected("engagement_detected"):
                after_engagement_reads += 1
                if after_engagement_reads > 2:
                    return cocked
            depth_m = 0.010 if feedback_detected("contact_detected") else 0.001
        elif phase in {"cocked", "disengaging"}:
            return axial_clear if axial_withdrawal_started else cocked
        elif phase in {"recentering", "retaring"}:
            return start
        else:
            retry_reads += 1
            depth_m = min(0.020, retry_reads * 0.0015)
        return ((0.0, 0.5, 1.1 - depth_m), (0.0, 0.0, 0.0, 1.0))

    server.control.servoL = servo_l
    server._read_actual_tcp_force = read_force
    server._read_actual_tcp_transform = read_pose

    result = server._execute_insert(goal)

    assert result.error_code == 0, result.error_string
    assert result.disengagement_cycle_count == 1
    assert result.disengagement_withdrawal_m == pytest.approx(0.0074)
    assert len(servo_targets) >= 2
    assert servo_targets[0][0] == pytest.approx(0.008)
    assert servo_targets[0][2] == pytest.approx(1.1)
    assert servo_targets[-1][2] == pytest.approx(1.1)
    assert result.recenter_position_error_m == pytest.approx(0.0)
    retare_record = next(
        record
        for record in (
            json.loads(line)
            for line in Path(result.server_trace_path).read_text().splitlines()
        )
        if record.get("kind") == "retare"
    )
    assert retare_record["torque_bias_delta_nm"] == pytest.approx(0.114)
    assert retare_record["torque_bias_delta_nm"] > (
        goal.request.baseline_torque_uncertainty_nm
    )
    assert retare_record["retare_samples_stable"] is True
    assert retare_record["retare_contact_free"] is True
    assert retare_record["retare_baseline_consistent"] is True
    assert result.seated_detected is True


def test_mg_cocking_recovery_stops_after_six_complete_cycles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _server_module()
    start = ((0.0, 0.5, 1.1), (0.0, 0.0, 0.0, 1.0))
    server, goal, _events = _insert_execution_harness(
        module,
        monkeypatch,
        force_samples=[[0.0] * 6],
        stationary_results=[True] * 40,
        actual_transforms=[start],
    )
    monkeypatch.setattr(module, "UR5E_RTDE_INSERT_MAX_TIMEOUT_SEC", 10.0)
    goal.request.timeout_sec = 8.0
    _refresh_request_hard_caps_sha256(module, goal)
    clock = _AdvancingClock(step_sec=0.005)
    module.time.monotonic = clock.monotonic
    rebound_reads_by_cycle: dict[int, int] = {}

    def latest_status() -> dict[str, Any]:
        return server._test_insert_statuses[-1] if server._test_insert_statuses else {}

    def read_force() -> list[float]:
        phase = str(latest_status().get("insert_phase") or "")
        if server._insert_force_mode_command is None or phase in {
            "cocked",
            "disengaging",
            "recentering",
            "retaring",
        }:
            return [0.0] * 6
        return [0.0, 0.0, 4.0, 0.0, 0.0, 0.0]

    def read_pose() -> Any:
        status = latest_status()
        phase = str(status.get("insert_phase") or "")
        cycle = int(status.get("insert_disengagement_cycle_count") or 0)
        if phase in {"cocked", "disengaging", "recentering", "retaring"}:
            return start
        last_feedback = goal.feedback[-1] if goal.feedback else None
        if bool(getattr(last_feedback, "engagement_detected", False)):
            rebound_reads_by_cycle[cycle] = rebound_reads_by_cycle.get(cycle, 0) + 1
            depth_m = 0.004 if rebound_reads_by_cycle[cycle] <= 2 else 0.0
        elif bool(getattr(last_feedback, "contact_detected", False)):
            depth_m = 0.004
        else:
            depth_m = 0.001
        return ((0.0, 0.5, 1.1 - depth_m), (0.0, 0.0, 0.0, 1.0))

    server._read_actual_tcp_force = read_force
    server._read_actual_tcp_transform = read_pose

    result = server._execute_insert(goal)

    assert result.error_code == -5
    assert result.disengagement_cycle_count == 6
    assert "protected disengagement cycle limit 6 exhausted" in result.error_string
    assert result.seated_detected is False


def test_insert_seating_accepts_stable_high_load_within_force_depth_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _server_module()
    start = ((0.0, 0.5, 1.1), (0.0, 0.0, 0.0, 1.0))
    target = ((0.0, 0.5, 1.09), (0.0, 0.0, 0.0, 1.0))
    normal_contact = [0.0, 0.0, 4.0, 0.0, 0.0, 0.0]
    loaded_contact = [3.0, 0.0, 4.0, 0.0, 0.0, 0.0]
    server, goal, events = _insert_execution_harness(
        module,
        monkeypatch,
        force_samples=[
            *([[0.0] * 6] * 5),
            *([normal_contact] * 5),
            *([loaded_contact] * 30),
        ],
        stationary_results=[True, True],
        spiral_radius_m=0.0,
        actual_transforms=[start, start, target, target, target, target],
    )
    clock = _AdvancingClock()
    module.time.monotonic = clock.monotonic

    result = server._execute_insert(goal)

    assert result.error_code == 0
    assert result.state_uncertain is False
    assert result.contact_detected is True
    assert result.engagement_detected is True
    assert result.seated_detected is True
    assert result.peak_lateral_force_n == pytest.approx(3.0)
    assert result.error_string == ""
    assert goal.outcomes == ["succeeded"]
    assert events.count("stop") == 1


def test_insert_seating_accepts_stable_axial_soft_overload_at_target_depth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _server_module()
    start = ((0.0, 0.5, 1.1), (0.0, 0.0, 0.0, 1.0))
    target = ((0.0, 0.5, 1.09), (0.0, 0.0, 0.0, 1.0))
    normal_contact = [0.0, 0.0, 4.0, 0.0, 0.0, 0.0]
    seated_axial_load = [0.0, 0.0, 12.0, 0.0, 0.0, 0.0]
    server, goal, events = _insert_execution_harness(
        module,
        monkeypatch,
        force_samples=[
            *([[0.0] * 6] * 5),
            *([normal_contact] * 8),
            *([seated_axial_load] * 80),
        ],
        stationary_results=[True, True],
        spiral_radius_m=0.0,
        actual_transforms=[start, start, *([target] * 100)],
    )
    clock = _AdvancingClock()
    module.time.monotonic = clock.monotonic

    result = server._execute_insert(goal)

    assert result.error_code == 0, result.error_string
    assert result.hard_limit_detected is False
    assert result.soft_overload_detected is True
    assert result.relief_cycle_count == 0
    assert result.disengagement_cycle_count == 0
    assert result.engagement_detected is True
    assert result.seated_detected is True
    assert "cocked" not in [feedback.phase for feedback in goal.feedback]
    assert "disengaging" not in [feedback.phase for feedback in goal.feedback]
    assert goal.outcomes == ["succeeded"]
    assert events.count("stop") == 1


def test_insert_rejects_force_zeroing_without_stationary_hold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _server_module()
    server, goal, events = _insert_execution_harness(
        module,
        monkeypatch,
        force_samples=[[0.0] * 6] * 5,
        stationary_results=[False],
    )

    result = server._execute_insert(goal)

    assert result.error_code == -1
    assert result.state_uncertain is False
    assert "stationary joint-velocity hold" in result.error_string
    assert goal.outcomes == ["aborted"]
    assert events.count("stationary") == 1
    assert "force_sample" not in events
    assert "moveL" not in events
    assert "stop" not in events


def test_insert_rejects_unstable_tared_force_baseline_before_motion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _server_module()
    baseline_samples = [
        [0.0, 0.0, force_z, 0.0, 0.0, 0.0]
        for force_z in (0.0, 1.0, 4.0, 0.5, 0.0)
    ]
    server, goal, events = _insert_execution_harness(
        module,
        monkeypatch,
        force_samples=baseline_samples,
        stationary_results=[True],
    )

    result = server._execute_insert(goal)

    assert result.error_code == -1
    assert result.state_uncertain is False
    assert result.force_bias_valid is False
    assert "force baseline varied" in result.error_string
    assert goal.outcomes == ["aborted"]
    assert "forceMode" not in events
    assert "stop" not in events
    assert [feedback.phase for feedback in goal.feedback] == ["checking"]
    assert goal.feedback[0].force_bias_valid is False


@pytest.mark.parametrize(
    ("stationary_results", "expected_uncertain", "expected_finish"),
    [
        ([True, True], False, "finish"),
        ([True, False], True, "finish_latched"),
    ],
)
def test_insert_cancellation_stops_and_settles_before_releasing_motion_slot(
    monkeypatch: pytest.MonkeyPatch,
    stationary_results: list[bool],
    expected_uncertain: bool,
    expected_finish: str,
) -> None:
    module = _server_module()
    server, goal, events = _insert_execution_harness(
        module,
        monkeypatch,
        force_samples=[[0.0] * 6] * 5,
        cancel_after_checks=6,
        stationary_results=stationary_results,
    )

    result = server._execute_insert(goal)

    assert result.error_code == -3
    assert result.state_uncertain is expected_uncertain
    assert result.motion_settled is (not expected_uncertain)
    assert goal.outcomes == ["canceled"]
    assert "moveL" not in events
    assert events.count("forceMode") == 1
    assert "servoL" not in events
    stop_index = events.index("stop")
    settlement_index = events.index("stationary", events.index("stationary") + 1)
    finish_index = events.index(expected_finish)
    assert stop_index < settlement_index < finish_index < events.index("clear")
    assert server._active_goal is None
    if expected_uncertain:
        assert "reset:insert_cancel_stop_unconfirmed" in events
    else:
        assert not any(event.startswith("reset:") for event in events)


def test_mg_target_may_differ_from_place_approach_descend_within_search_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _server_module()
    server, goal, events = _insert_execution_harness(
        module,
        monkeypatch,
        force_samples=[[0.0] * 6] * 5,
        cancel_after_checks=6,
        stationary_results=[True, True],
    )
    goal.request.target_tool0_pose = _pose_stamped_from_transform(
        module,
        ((0.005, 0.5, 1.09), (0.0, 0.0, 0.0, 1.0)),
    )

    result = server._execute_insert(goal)

    assert result.error_code == -3
    assert goal.outcomes == ["canceled"]
    assert "outside the protected lateral start boundary" not in result.error_string
    assert events.count("forceMode") == 1


@pytest.mark.parametrize(
    ("force", "peak_field", "threshold"),
    [
        ([0.0, 0.0, 21.0, 0.0, 0.0, 0.0], "peak_axial_force_n", 21.0),
        ([11.0, 0.0, 0.0, 0.0, 0.0, 0.0], "peak_lateral_force_n", 11.0),
        ([0.0, 0.0, 0.0, 0.0, 0.0, 6.0], "peak_torque_nm", 6.0),
    ],
)
def test_insert_force_and_torque_crossings_stop_without_continuation(
    monkeypatch: pytest.MonkeyPatch,
    force: list[float],
    peak_field: str,
    threshold: float,
) -> None:
    module = _server_module()
    server, goal, events = _insert_execution_harness(
        module,
        monkeypatch,
        force_samples=[*([[0.0] * 6] * 5), force],
        stationary_results=[True, True],
    )

    result = server._execute_insert(goal)

    assert result.error_code == -6
    assert result.state_uncertain is False
    assert getattr(result, peak_field) == pytest.approx(threshold)
    assert goal.outcomes == ["aborted"]
    assert "moveL" not in events
    assert events.count("stop") == 1
    assert events.count("forceMode") >= 2
    assert "servoL" not in events
    assert events.index("stop") < events.index("finish") < events.index("clear")


def test_hard_limit_exact_trigger_matches_feedback_status_result_and_server_trace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _server_module()
    trigger_force = [0.0, 0.0, 21.0, 0.1, 0.2, 0.3]
    server, goal, _events = _insert_execution_harness(
        module,
        monkeypatch,
        force_samples=[*([[0.0] * 6] * 5), trigger_force],
        stationary_results=[True, True],
    )

    result = server._execute_insert(goal)

    assert result.error_code == -6
    assert result.trial_id == "trial-test"
    assert result.hard_limit_detected is True
    assert result.limit_trigger == "axial_force_n"
    assert result.limit_trigger_value == pytest.approx(21.0)
    assert result.limit_trigger_threshold == pytest.approx(20.0)
    assert result.limit_trigger_actual_tcp_force == pytest.approx(trigger_force)
    assert result.limit_trigger_tared_tcp_force == pytest.approx(trigger_force)
    trigger_feedback = next(
        feedback for feedback in goal.feedback if feedback.hard_limit_detected
    )
    assert trigger_feedback.limit_trigger == result.limit_trigger
    assert trigger_feedback.limit_trigger_value == pytest.approx(
        result.limit_trigger_value
    )
    assert trigger_feedback.limit_trigger_actual_tcp_force == pytest.approx(
        result.limit_trigger_actual_tcp_force
    )
    terminal_status = server._test_insert_statuses[-1]
    assert terminal_status["insert_limit_trigger"] == result.limit_trigger
    assert terminal_status["insert_limit_trigger_value"] == pytest.approx(
        result.limit_trigger_value
    )
    assert terminal_status["insert_limit_trigger_actual_tcp_force"] == pytest.approx(
        trigger_force
    )
    assert result.server_trace_complete is True
    trace_path = Path(result.server_trace_path)
    assert trace_path == module.INSERT_TRIAL_TRACE_ROOT / "trial-test" / "trace.jsonl"
    assert trace_path.is_file()
    assert hashlib.sha256(trace_path.read_bytes()).hexdigest() == (
        result.server_trace_sha256
    )
    records = [json.loads(line) for line in trace_path.read_text().splitlines()]
    trigger_record = next(record for record in records if record["kind"] == "hard_trigger")
    assert trigger_record["limit_trigger"] == result.limit_trigger
    assert trigger_record["limit_trigger_value"] == pytest.approx(21.0)
    assert trigger_record["actual_tcp_force"] == pytest.approx(trigger_force)
    assert result.server_trace_sample_count == sum(
        record["kind"] == "sample" for record in records
    )


def test_tool_flange_hard_ceiling_is_independent_of_active_tcp_torque(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _server_module()
    server, goal, _events = _insert_execution_harness(
        module,
        monkeypatch,
        force_samples=[
            *([[0.0] * 6] * 5),
            [0.0, 10.0, 0.0, 0.0, 0.0, 1.0],
        ],
        stationary_results=[True, True],
    )
    exact_mg_caps = dict(module.UR5E_RTDE_INSERT_MG_HARD_CAPS)
    exact_mg_caps["insert_max_tool_flange_torque_nm"] = 0.5
    monkeypatch.setattr(module, "UR5E_RTDE_INSERT_MG_HARD_CAPS", exact_mg_caps)
    _refresh_request_hard_caps_sha256(module, goal)
    server._active_tcp_offset = lambda: (
        (0.1, 0.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )
    server._read_actual_tcp_transform = lambda: (
        (0.1, 0.5, 1.1),
        (0.0, 0.0, 0.0, 1.0),
    )

    result = server._execute_insert(goal)

    assert result.error_code == -6
    assert result.limit_trigger == "tool_flange_torque_nm"
    assert result.peak_torque_nm == pytest.approx(0.0, abs=1e-9)
    assert result.peak_tool_flange_torque_nm == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("unsafe_transform", "message"),
    [
        (
            ((0.0, 0.5, 1.103), (0.0, 0.0, 0.0, 1.0)),
            "opposite insertion_axis_world",
        ),
        (
            ((0.0, 0.5, 1.088), (0.0, 0.0, 0.0, 1.0)),
            "exceeded target depth",
        ),
        (
            ((0.021, 0.5, 1.1), (0.0, 0.0, 0.0, 1.0)),
            "lateral insertion offset",
        ),
        (
            (
                (0.0, 0.5, 1.1),
                (math.sin(0.015), 0.0, 0.0, math.cos(0.015)),
            ),
            "insertion tilt",
        ),
    ],
)
def test_insert_pose_limit_crossings_stop_without_continuation(
    monkeypatch: pytest.MonkeyPatch,
    unsafe_transform: Any,
    message: str,
) -> None:
    module = _server_module()
    start_transform = ((0.0, 0.5, 1.1), (0.0, 0.0, 0.0, 1.0))
    server, goal, events = _insert_execution_harness(
        module,
        monkeypatch,
        force_samples=[[0.0] * 6] * 6,
        stationary_results=[True, True],
        actual_transforms=[start_transform, unsafe_transform],
    )

    result = server._execute_insert(goal)

    assert result.error_code == -6
    assert result.state_uncertain is False
    assert message in result.error_string
    assert goal.outcomes == ["aborted"]
    assert "moveL" not in events
    assert events.count("stop") == 1
    assert events.count("forceMode") >= 2
    assert "servoL" not in events


def test_search_uses_protected_maximum_lateral_envelope_not_learned_radius(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _server_module()
    start = ((0.0, 0.5, 1.1), (0.0, 0.0, 0.0, 1.0))
    within_protected_envelope = (
        (0.004564, 0.5, 1.091),
        (0.0, 0.0, 0.0, 1.0),
    )
    server, goal, _events = _insert_execution_harness(
        module,
        monkeypatch,
        force_samples=[
            *([[0.0] * 6] * 5),
            *([[0.0, 0.0, 4.0, 0.0, 0.0, 0.0]] * 120),
        ],
        stationary_results=[True, True],
        spiral_radius_m=0.0015,
        actual_transforms=[
            start,
            *([within_protected_envelope] * 140),
        ],
    )
    monkeypatch.setattr(module, "UR5E_RTDE_INSERT_MAX_SPIRAL_RADIUS_M", 0.002)
    monkeypatch.setattr(
        module,
        "UR5E_RTDE_INSERT_START_POSITION_TOLERANCE_M",
        0.003,
    )
    _refresh_request_hard_caps_sha256(module, goal)
    clock = _AdvancingClock(step_sec=0.01)
    module.time.monotonic = clock.monotonic

    result = server._execute_insert(goal)

    assert result.error_code != -6, result.error_string
    assert result.limit_trigger != "lateral_offset_m"
    assert result.final_lateral_offset_m == pytest.approx(0.004564)


def test_insert_workspace_violation_is_immediate_hard_stop_before_relief(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _server_module()
    start = ((0.0, 0.5, 1.1), (0.0, 0.0, 0.0, 1.0))
    outside_workspace = ((0.71, 0.5, 1.1), (0.0, 0.0, 0.0, 1.0))
    server, goal, events = _insert_execution_harness(
        module,
        monkeypatch,
        force_samples=[[0.0] * 6] * 6,
        stationary_results=[True, True],
        actual_transforms=[start, outside_workspace],
    )

    result = server._execute_insert(goal)

    assert result.error_code == -6
    assert result.limit_trigger == "workspace_pose"
    assert "left protected workspace" in result.error_string
    assert result.soft_overload_detected is False
    assert events.count("stop") == 1
    assert not any(feedback.phase in {"relieving", "backing_off"} for feedback in goal.feedback)


def test_insert_force_transport_loss_after_dispatch_is_uncertain_and_not_retried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _server_module()
    server, goal, events = _insert_execution_harness(
        module,
        monkeypatch,
        force_samples=[*([[0.0] * 6] * 5), None],
        stationary_results=[True, True],
    )

    result = server._execute_insert(goal)

    assert result.error_code == -4
    assert result.state_uncertain is True
    assert result.motion_settled is True
    assert "actual_TCP_force is unavailable" in result.error_string
    assert goal.outcomes == ["aborted"]
    assert "moveL" not in events
    assert events.count("stop") == 1
    assert events.count("forceMode") >= 2
    assert "servoL" not in events
    assert "reset:insert_execution_unknown" in events
    assert events.index("stop") < events.index("finish_latched") < events.index("clear")


def test_insert_frozen_rtde_timestamp_aborts_before_cached_evidence_accumulates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _server_module()
    start = ((0.0, 0.5, 1.1), (0.0, 0.0, 0.0, 1.0))
    target = ((0.0, 0.5, 1.09), (0.0, 0.0, 0.0, 1.0))
    server, goal, events = _insert_execution_harness(
        module,
        monkeypatch,
        force_samples=[*([[0.0] * 6] * 5), [0.0, 0.0, 4.0, 0.0, 0.0, 0.0]],
        stationary_results=[True, True],
        actual_transforms=[start, start, target, target, target],
        feedback_timestamps=[1.0, 2.0, 3.0, 4.0, 5.0, 5.0],
    )
    clock = _AdvancingClock(step_sec=0.01)
    module.time.monotonic = clock.monotonic

    result = server._execute_insert(goal)

    assert result.error_code == -4
    assert result.state_uncertain is True
    assert result.motion_settled is True
    assert result.contact_detected is False
    assert result.engagement_detected is False
    assert result.seated_detected is False
    assert "timestamp stopped advancing" in result.error_string
    assert goal.outcomes == ["aborted"]
    assert events.count("forceMode") >= 2
    assert events.count("stop") == 1


def test_insert_unknown_force_mode_acceptance_with_unsettled_stop_latches_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _server_module()
    server, goal, events = _insert_execution_harness(
        module,
        monkeypatch,
        force_samples=[[0.0] * 6] * 5,
        stationary_results=[True, False],
        force_mode_failure=RuntimeError("send timeout; acceptance unknown"),
    )

    result = server._execute_insert(goal)

    assert result.error_code == -4
    assert result.state_uncertain is True
    assert result.motion_settled is False
    assert "acceptance unknown" in result.error_string
    assert goal.outcomes == ["aborted"]
    assert "moveL" not in events
    assert events.count("stop") == 1
    assert events.count("forceMode") == 1
    assert "servoL" not in events
    assert "reset:insert_stop_unconfirmed" in events
    assert events.index("stop") < events.index("finish_latched") < events.index("clear")


def test_insert_contact_timeout_maps_to_no_entry_and_does_not_retract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _server_module()
    server, goal, events = _insert_execution_harness(
        module,
        monkeypatch,
        force_samples=[[0.0] * 6] * 5,
        stationary_results=[True, True],
    )
    clock = _AdvancingClock(step_sec=0.05)
    module.time.monotonic = clock.monotonic

    result = server._execute_insert(goal)

    assert result.error_code == -5
    assert result.state_uncertain is False
    assert "timed out without engagement" in result.error_string
    assert goal.outcomes == ["aborted"]
    assert "moveL" not in events
    assert events.count("stop") == 1
    assert events.count("forceMode") >= 1
    assert "servoL" not in events


def test_insert_spiral_radius_exhaustion_stops_without_retraction_or_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _server_module()
    contact_force = [0.0, 0.0, 4.0, 0.0, 0.0, 0.0]
    server, goal, events = _insert_execution_harness(
        module,
        monkeypatch,
        force_samples=[*([[0.0] * 6] * 5), contact_force],
        stationary_results=[True, True, True],
        spiral_radius_m=0.000001,
    )
    clock = _AdvancingClock()
    module.time.monotonic = clock.monotonic

    result = server._execute_insert(goal)

    assert result.error_code == -5
    assert result.state_uncertain is False
    assert result.contact_detected is True
    assert 0.000001 < result.final_search_radius_m <= 0.02
    assert "timed out without engagement" in result.error_string
    assert any(
        feedback.phase == "expanded_searching" for feedback in goal.feedback
    )
    assert goal.outcomes == ["aborted"]
    assert "moveL" not in events
    assert events.count("forceMode") >= 2
    assert "servoL" not in events
    assert events.count("stop") == 1
    assert events.index("forceMode") < events.index("stop") < events.index("finish")


def test_insert_spiral_uses_bounded_lateral_force_without_cartesian_servo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _server_module()
    contact_force = [0.0, 0.0, 4.0, 0.0, 0.0, 0.0]
    server, goal, events = _insert_execution_harness(
        module,
        monkeypatch,
        force_samples=[*([[0.0] * 6] * 5), contact_force],
        stationary_results=[True, True],
        spiral_radius_m=0.00001,
    )
    clock = _AdvancingClock()
    module.time.monotonic = clock.monotonic

    result = server._execute_insert(goal)

    assert result.error_code == -5
    assert result.state_uncertain is False
    assert goal.outcomes == ["aborted"]
    assert "moveL" not in events
    assert "servoL" not in events
    search_wrenches = [
        command[2]
        for command in server._test_force_mode_commands
        if abs(command[2][0]) > 0.0 or abs(command[2][1]) > 0.0
    ]
    assert search_wrenches
    assert all(
        math.hypot(wrench[0], wrench[1])
        <= goal.request.max_lateral_force_n * 0.5 + 1e-12
        for wrench in search_wrenches
    )


def test_insert_action_shares_the_existing_motion_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _server_module()

    class _Result:
        pass

    module.MoveUR5eInsert = SimpleNamespace(Result=_Result)
    monkeypatch.setattr(module, "_insert_hard_cap_error", lambda _part_name="": None)
    server = object.__new__(module.UR5eRTDETrajectoryServer)
    server._active_lock = threading.Lock()
    server._shutdown_requested = False
    server._rtde_reset_required = False
    server._latched_terminal_status = None
    server._active_goal = object()
    aborted: list[bool] = []
    goal = SimpleNamespace(abort=lambda: aborted.append(True))

    result = server._execute_insert(goal)

    assert aborted == [True]
    assert result.error_code == -1
    assert result.error_string == "UR5e RTDE motion already executing"
    assert result.state_uncertain is False


@pytest.mark.parametrize("part_name", ["mg", " MG", "gear", ""])
def test_insert_rejects_nonexact_part_tokens_before_hardware(
    monkeypatch: pytest.MonkeyPatch,
    part_name: str,
) -> None:
    module = _server_module()

    class _Result:
        pass

    module.MoveUR5eInsert = SimpleNamespace(Result=_Result)
    monkeypatch.setattr(
        module,
        "_insert_hard_cap_error",
        lambda _part_name="": None,
    )
    server = object.__new__(module.UR5eRTDETrajectoryServer)
    server._active_lock = threading.Lock()
    server._shutdown_requested = False
    server._rtde_reset_required = False
    server._latched_terminal_status = None
    server._active_goal = None
    server._active_goal_status = None
    server._active_motion_kind = ""
    server._write_active_goal_status = lambda *_args, **_kwargs: None
    server._finish_active_goal_status = lambda *_args, **_kwargs: None
    server._clear_active_goal = lambda _goal: setattr(server, "_active_goal", None)
    server._connect_control_for_goal = lambda: pytest.fail(
        "an invalid part_name must be rejected before hardware access"
    )
    outcome: list[str] = []
    goal = SimpleNamespace(
        request=SimpleNamespace(
            part_name=part_name,
            calibration_id="calibration-v1",
            profile_sha256="a" * 64,
        ),
        abort=lambda: outcome.append("aborted"),
    )

    result = server._execute_insert(goal)

    assert outcome == ["aborted"]
    assert result.error_code == -2
    assert "exact supported tokens" in result.error_string


def test_insert_action_keeps_distinct_terminal_error_codes() -> None:
    module = _server_module()
    source = inspect.getsource(module.UR5eRTDETrajectoryServer._execute_insert)

    assert re.search(r"return result\(\s*-3,", source)
    assert re.search(r"return result\(\s*-5,", source)
    assert re.search(r"return result\(\s*-6,", source)
    assert "_confirm_stationary_after_stop" in source
    assert "move_home" not in source


def test_insert_phases_use_only_the_fixed_action_symbols() -> None:
    module = _server_module()
    source = inspect.getsource(module.UR5eRTDETrajectoryServer._execute_insert)
    phase_literals = set(
        re.findall(r'(?:insert_phase=|sample\(|phase = )"([a-z_]+)"', source)
    )

    assert phase_literals == {
        "checking",
        "zeroing_force",
        "searching",
        "expanded_searching",
        "seating",
        "relieving",
        "backing_off",
        "resuming",
        "settling",
        "cocked",
        "disengaging",
        "recentering",
        "retaring",
        "retrying",
    }
    assert "contacting" not in source


@pytest.mark.parametrize("active_motion_kind", ["relative_cartesian", "cartesian_jog"])
def test_insertion_demonstration_records_passively_with_advancing_feedback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    active_motion_kind: str,
) -> None:
    module = _server_module()

    class _Result:
        pass

    class _Feedback:
        pass

    module.RecordUR5eInsertionDemonstration = SimpleNamespace(
        Result=_Result,
        Feedback=_Feedback,
    )
    monkeypatch.setattr(module.rclpy, "ok", lambda: True)
    monkeypatch.setattr(module, "INSERT_DEMONSTRATION_TRACE_ROOT", tmp_path)
    monkeypatch.setattr(
        module,
        "UR5E_RTDE_INSERT_DEMONSTRATION_BASELINE_SEC",
        0.02,
    )
    monkeypatch.setattr(
        module,
        "UR5E_RTDE_INSERT_DEMONSTRATION_MAX_DURATION_SEC",
        1.0,
    )
    clock = _AdvancingClock(step_sec=0.0005)
    monkeypatch.setattr(
        module,
        "time",
        SimpleNamespace(
            time=stdlib_time.time,
            monotonic=clock.monotonic,
            sleep=lambda _duration: None,
        ),
    )
    request = SimpleNamespace(
        recording_id="insertion-demonstration-test",
        part_name="MG",
        destination_location="assembly_board-v1",
        context_sha256="a" * 64,
        expected_start_tool0_pose=_insert_pose(module, 1.1),
        max_duration_sec=0.5,
    )
    goal = _InsertGoal(request, cancel_after_checks=80)
    server = object.__new__(module.UR5eRTDETrajectoryServer)
    server.monitor_only = False
    server.control = object()
    server.receive = object()
    server._rtde_reset_required = False
    server._rtde_reset_reason = ""
    server._insertion_demonstration_lock = threading.Lock()
    server._active_insertion_demonstration_goal = None
    server._active_insertion_demonstration_status = {}
    server._active_lock = threading.Lock()
    server._active_motion_kind = active_motion_kind
    server._active_goal_status = {
        "state": "executing",
        "world_linear_velocity_m_s": [0.0, 0.0, -0.002],
        "message": "matching low-speed Cartesian jog",
    }
    server._lookup_rigid_transform = lambda _target, _source: (
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )
    server._validated_cartesian_world_base = lambda: (
        ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
        "ready",
        0.0,
        0.0,
    )
    server._active_tcp_offset = lambda: (
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )
    actual_transform = (
        (0.0, 0.5, 1.1),
        (0.0, 0.0, 0.0, 1.0),
    )
    server._read_actual_tcp_transform = lambda: actual_transform
    timestamp = iter(float(index) for index in range(1, 100))
    server._read_feedback_timestamp = lambda: next(timestamp)
    server._read_actual_q = lambda: [0.0] * 6
    server._read_actual_tcp_force = lambda: [0.0, 0.0, 5.0, 0.0, 0.0, 0.1]
    server._read_actual_tcp_speed = lambda: [0.0] * 6
    server._world_tool0_from_actual_tcp = (
        lambda actual_base_tcp, *, world_base, tool0_tcp: actual_base_tcp
    )
    server._pose_stamped_from_transform = lambda value: _pose_stamped_from_transform(
        module, value
    )
    server._confirm_stationary_after_stop = lambda *, timeout_sec: True

    result = server._execute_insertion_demonstration(goal)

    assert goal.outcomes == ["canceled"]
    assert result.error_code == 0
    assert result.motion_settled is True
    assert result.baseline_valid is True
    assert result.sample_count > 5
    assert Path(result.trace_path).is_file()
    assert len(result.trace_sha256) == 64
    assert {feedback.phase for feedback in goal.feedback} == {"recording_insertion"}
    trace_rows = [
        json.loads(line)
        for line in Path(result.trace_path).read_text(encoding="utf-8").splitlines()
    ]
    trace_phases = {row["phase"] for row in trace_rows}
    assert trace_phases == {"recording_baseline", "recording_insertion"}
    assert {row["active_motion_kind"] for row in trace_rows} == {
        active_motion_kind
    }
    assert all(
        row["active_motion_status"]["world_linear_velocity_m_s"]
        == [0.0, 0.0, -0.002]
        for row in trace_rows
    )
    assert server._active_insertion_demonstration_goal is None
    assert server._insertion_demonstration_lock.acquire(blocking=False)
    server._insertion_demonstration_lock.release()
