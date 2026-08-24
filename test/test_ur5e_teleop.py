"""Focused contracts for real UR5e Interactive Teleop trajectory routing."""

from __future__ import annotations

import importlib.util
import inspect
import io
import json
import math
import os
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _isolate_operator_insertion_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cais_spade_llm.ui import bridge as bridge_module

    monkeypatch.setattr(
        bridge_module,
        "_MOVE_INSERT_TRIALS_DIR",
        tmp_path / "operator_move_insert_trials",
    )
    monkeypatch.setattr(
        bridge_module,
        "_INSERTION_DEMONSTRATIONS_DIR",
        tmp_path / "operator_move_insert_demonstrations",
    )
    monkeypatch.setattr(
        bridge_module,
        "_HARDWARE_STATE_DIR",
        tmp_path / "operator_hardware_state",
    )


def _teleop_module() -> Any:
    path = ROOT / "ros2/cais_lab_robotics/scripts/keyboard_teleop.py"
    spec = importlib.util.spec_from_file_location("keyboard_teleop_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rtde_server_module() -> Any:
    path = ROOT / "ros2/cais_lab_robotics/scripts/ur5e_rtde_trajectory_server.py"
    spec = importlib.util.spec_from_file_location("ur5e_rtde_server_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cartesian_readiness_waits_for_joint_state_before_environment_inference() -> None:
    module = _teleop_module()
    source = inspect.getsource(module.run_server)
    readiness_branch = source.split("if op == 'cartesian_readiness':", 1)[1].split(
        "if op == 'cartesian_smooth':",
        1,
    )[0]

    assert readiness_branch.index('wait_for_joint_positions(') < readiness_branch.index(
        'infer_robot_environment(robot)'
    )
    assert "readiness_deadline" not in readiness_branch
    assert "_xarm6_restore_trajectory_control" not in readiness_branch
    assert "_xarm6_prepare_firmware_cartesian_mode" not in readiness_branch
    smooth_branch = source.split("if op == 'cartesian_smooth':", 1)[1].split(
        "if op == 'cartesian':",
        1,
    )[0]
    assert "node._xarm6_last_stop_motion_confirmed" in smooth_branch
    assert "node._ur5e_last_stop_motion_confirmed" in smooth_branch
    assert "response['state_uncertain']" in smooth_branch


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


def test_real_xarm6_joint_target_uses_action_not_topic() -> None:
    module = _teleop_module()
    teleop = object.__new__(module.KeyboardTeleop)
    joint_names = list(module.ROBOTS["xarm6"]["joint_name_candidates"][1])
    targets = [0.1, -1.0, -2.0, -1.2, 1.5, 0.0]
    teleop.active_joint_names = {"xarm6": joint_names}
    teleop.joint_duration_sec = 0.08
    teleop.xarm6_hardware_joint_duration_scale = 4.0
    calls: list[tuple[list[str], list[float], float]] = []
    teleop._move_xarm6_arm_action = lambda names, positions, duration_sec: (
        calls.append((list(names), list(positions), float(duration_sec))) or (True, "xArm6")
    )
    teleop._pick_publisher = lambda _publishers: pytest.fail(
        "real xArm6 must use FollowJointTrajectory instead of topic publication"
    )

    assert teleop.move_arm_to_joints("xarm6", targets, duration_sec=1.2) == (
        True,
        "xArm6",
    )
    assert calls == [(joint_names, targets, 4.8)]


@pytest.mark.parametrize("robot", ["xarm6", "ur5e"])
def test_real_cartesian_jog_uses_direct_hardware_not_moveit(robot: str) -> None:
    module = _teleop_module()
    teleop = object.__new__(module.KeyboardTeleop)
    teleop.active_joint_names = {
        robot: list(module.ROBOTS[robot]["joint_name_candidates"][1])
    }
    teleop.active_ee_link = {}
    teleop.active_frame_id = {}
    current = module.Pose()
    current.position.x = 0.1
    current.position.y = -0.2 if robot == "xarm6" else 0.3
    current.position.z = 1.2
    current.orientation.w = 1.0
    teleop._get_world_ee_pose = lambda _robot: current
    teleop.get_ee_pose = lambda _robot: pytest.fail(
        "real Cartesian jog must require exact world TF"
    )
    direct_calls: list[tuple[str, Any, float]] = []
    teleop._move_xarm6_relative_cartesian = lambda delta, scale: (
        direct_calls.append(("xarm6", tuple(delta), float(scale)))
        or (True, "xArm6 direct relative")
    )
    teleop._move_ur5e_relative_cartesian = lambda delta, scale: (
        direct_calls.append(("ur5e", tuple(delta), float(scale)))
        or (True, "UR5e direct relative")
    )
    teleop.cartesian_client = SimpleNamespace(
        call_async=lambda _request: pytest.fail(
            "real Cartesian jog must not call /compute_cartesian_path"
        )
    )
    teleop.execute_client = SimpleNamespace(
        send_goal_async=lambda _goal: pytest.fail(
            "real Cartesian jog must not call /execute_trajectory"
        )
    )

    ok, message = teleop.move_cartesian(
        robot,
        dx_mm=10.0,
        dy_mm=-2.0,
        dz_mm=1.0,
        velocity_scale=0.35,
    )

    assert ok is True
    assert "direct" in message
    assert len(direct_calls) == 1
    called_robot, delta, scale = direct_calls[0]
    assert called_robot == robot
    assert delta == pytest.approx((0.010, -0.002, 0.001))
    assert scale == pytest.approx(0.35)


def test_real_ur5e_cartesian_jog_sends_guarded_rtde_action() -> None:
    module = _teleop_module()
    class _CartesianGoal:
        def __init__(self) -> None:
            self.target_tool0_pose = None
            self.speed_m_s = 0.0
            self.acceleration_m_s2 = 0.0

    module.MoveUR5eCartesian = SimpleNamespace(Goal=_CartesianGoal)
    teleop = object.__new__(module.KeyboardTeleop)
    teleop.ur5e_hardware_cartesian_action = module.UR5E_HARDWARE_CARTESIAN_ACTION
    teleop.ur5e_hardware_cartesian_speed_m_s = 0.05
    teleop.ur5e_hardware_cartesian_acceleration_m_s2 = 0.10
    teleop.ur5e_hardware_result_timeout_sec = 45.0
    teleop.get_clock = lambda: SimpleNamespace(
        now=lambda: module.rclpy.time.Time()
    )
    teleop._wait_future = lambda future, timeout: future.done()
    result = SimpleNamespace(
        error_code=0,
        error_string="",
        final_position_error_m=0.0005,
        final_orientation_error_rad=0.001,
    )
    goal_handle = SimpleNamespace(
        accepted=True,
        get_result_async=lambda: _ImmediateFuture(
            SimpleNamespace(status=4, result=result)
        ),
    )
    sent_goals: list[Any] = []
    teleop.ur5e_hardware_cartesian_client = SimpleNamespace(
        wait_for_server=lambda timeout_sec: timeout_sec == pytest.approx(2.0),
        send_goal_async=lambda goal: (
            sent_goals.append(goal) or _ImmediateFuture(goal_handle)
        ),
    )
    target = module.Pose()
    target.position.x = 0.2
    target.position.y = 0.4
    target.position.z = 1.3
    target.orientation.w = 1.0

    ok, message = teleop._move_ur5e_hardware_cartesian(
        target,
        velocity_scale=0.35,
    )

    assert ok is True
    assert "succeeded" in message
    assert len(sent_goals) == 1
    goal = sent_goals[0]
    assert goal.target_tool0_pose.header.frame_id == "world"
    assert goal.target_tool0_pose.pose.position.x == pytest.approx(0.2)
    assert goal.speed_m_s == pytest.approx(0.0175)
    assert goal.acceleration_m_s2 == pytest.approx(0.035)


def test_real_ur5e_joint_jog_sends_exact_speed_to_guarded_action() -> None:
    module = _teleop_module()

    class _JointJogGoal:
        def __init__(self) -> None:
            self.joint = 0
            self.delta_rad = 0.0
            self.speed_rad_s = 0.0
            self.acceleration_rad_s2 = 0.0

    module.MoveUR5eJointJog = SimpleNamespace(Goal=_JointJogGoal)
    teleop = object.__new__(module.KeyboardTeleop)
    teleop.ur5e_hardware_joint_jog_action = module.UR5E_HARDWARE_JOINT_JOG_ACTION
    teleop.ur5e_hardware_max_joint_speed_rad_s = 1.125
    teleop.ur5e_hardware_max_joint_acceleration_rad_s2 = 1.263
    teleop.ur5e_hardware_result_timeout_sec = 45.0
    teleop._wait_future = lambda future, timeout: future.done()
    result = SimpleNamespace(
        error_code=0,
        error_string="",
        final_joint_error_rad=0.0005,
        state_uncertain=False,
    )
    goal_handle = SimpleNamespace(
        accepted=True,
        get_result_async=lambda: _ImmediateFuture(
            SimpleNamespace(status=4, result=result)
        ),
    )
    sent_goals: list[Any] = []
    teleop.ur5e_hardware_joint_jog_client = SimpleNamespace(
        wait_for_server=lambda timeout_sec: timeout_sec == pytest.approx(2.0),
        send_goal_async=lambda goal: (
            sent_goals.append(goal) or _ImmediateFuture(goal_handle)
        ),
    )

    ok, message = teleop._move_ur5e_joint_jog(3, -2.0, 24.5)

    assert ok is True, message
    assert len(sent_goals) == 1
    goal = sent_goals[0]
    assert goal.joint == 3
    assert goal.delta_rad == pytest.approx(math.radians(-2.0))
    assert goal.speed_rad_s == pytest.approx(math.radians(24.5))
    assert goal.acceleration_rad_s2 == pytest.approx(1.263)
    assert teleop._last_ur5e_joint_jog_state_uncertain is False


def test_real_ur5e_cartesian_step_accepts_low_positive_speed_and_extends_timeout() -> None:
    module = _teleop_module()

    class _RelativeGoal:
        def __init__(self) -> None:
            self.world_translation_m = SimpleNamespace(x=0.0, y=0.0, z=0.0)
            self.speed_m_s = 0.0
            self.acceleration_m_s2 = 0.0

    module.MoveUR5eRelativeCartesian = SimpleNamespace(Goal=_RelativeGoal)
    teleop = object.__new__(module.KeyboardTeleop)
    teleop.ur5e_hardware_relative_cartesian_action = (
        module.UR5E_HARDWARE_RELATIVE_CARTESIAN_ACTION
    )
    teleop.ur5e_hardware_cartesian_speed_m_s = 0.05
    teleop.ur5e_hardware_cartesian_max_speed_m_s = 0.10
    teleop.ur5e_hardware_cartesian_acceleration_m_s2 = 0.10
    teleop.ur5e_hardware_result_timeout_sec = 45.0
    timeouts: list[float] = []
    teleop._wait_future = lambda future, timeout: (
        timeouts.append(float(timeout)) or future.done()
    )
    result = SimpleNamespace(
        error_code=0,
        error_string="",
        final_translation_error_m=0.0001,
        final_orientation_drift_rad=0.0001,
    )
    goal_handle = SimpleNamespace(
        accepted=True,
        get_result_async=lambda: _ImmediateFuture(
            SimpleNamespace(status=4, result=result)
        ),
    )
    sent_goals: list[Any] = []
    teleop.ur5e_hardware_relative_cartesian_client = SimpleNamespace(
        wait_for_server=lambda timeout_sec: timeout_sec == pytest.approx(2.0),
        send_goal_async=lambda goal: (
            sent_goals.append(goal) or _ImmediateFuture(goal_handle)
        ),
    )

    ok, message = teleop._move_ur5e_relative_cartesian(
        (0.001, 0.0, 0.0),
        velocity_scale=1.0,
        speed_mm_s=0.025,
    )

    assert ok is True, message
    assert len(sent_goals) == 1
    assert sent_goals[0].speed_m_s == pytest.approx(0.000025)
    assert timeouts == pytest.approx([3.0, 55.0])


def test_ur5e_joint_jog_server_convergence_cancellation_and_uncertainty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _rtde_server_module()

    class _Result:
        def __init__(self) -> None:
            self.error_code = 0
            self.error_string = ""
            self.final_joint_error_rad = math.inf
            self.state_uncertain = False

    class _Feedback:
        def __init__(self) -> None:
            self.joint_error_rad = math.inf

    class _Goal:
        def __init__(self, *, cancel: bool = False) -> None:
            self.request = SimpleNamespace(
                joint=1,
                delta_rad=0.1,
                speed_rad_s=0.5,
                acceleration_rad_s2=1.0,
            )
            self.is_cancel_requested = cancel
            self.outcome = ""
            self.feedback: list[float] = []

        def abort(self) -> None:
            self.outcome = "aborted"

        def succeed(self) -> None:
            self.outcome = "succeeded"

        def canceled(self) -> None:
            self.outcome = "canceled"

        def publish_feedback(self, feedback: Any) -> None:
            self.feedback.append(float(feedback.joint_error_rad))

    module.MoveUR5eJointJog = SimpleNamespace(Result=_Result, Feedback=_Feedback)
    monkeypatch.setattr(module, "UR5E_RTDE_STATIONARY_HOLD_SEC", 0.0)

    def _server(actual_samples: list[list[float]], *, safety: bool = True) -> Any:
        server = object.__new__(module.UR5eRTDETrajectoryServer)
        server._active_lock = threading.Lock()
        server._shutdown_requested = False
        server._rtde_reset_required = False
        server._rtde_reset_reason = ""
        server._latched_terminal_status = None
        server._active_goal = None
        server._active_goal_status = None
        server._active_motion_kind = ""
        samples = iter(actual_samples)
        server.control = SimpleNamespace(
            isJointsWithinSafetyLimits=lambda _target: safety
        )
        server._connect_control_for_goal = lambda: None
        server._joint_states_fresh = lambda: True
        server._ensure_control_program_for_goal = lambda: None
        server._read_actual_q = lambda: list(next(samples))
        server._read_actual_qd = lambda: [0.0] * 6
        server._execute_movej_target = lambda *_args, **_kwargs: True
        server._write_active_goal_status = lambda *_args, **_kwargs: None
        server._finish_active_goal_status = lambda *_args, **_kwargs: None
        server._clear_active_goal = lambda _goal: setattr(server, "_active_goal", None)
        server._stop_motion = lambda: None
        server._mark_rtde_reset_required = lambda *_args, **_kwargs: None
        return server

    monkeypatch.setattr(module.rclpy, "ok", lambda: True)
    success_goal = _Goal()
    success = _server([[0.0] * 6, [0.1, 0.0, 0.0, 0.0, 0.0, 0.0]])
    success_result = success._execute_joint_jog(success_goal)
    assert success_goal.outcome == "succeeded"
    assert success_result.error_code == 0
    assert success_result.state_uncertain is False
    assert success_goal.feedback[-1] == pytest.approx(0.0)

    cancel_goal = _Goal(cancel=True)
    canceled = _server([[0.0] * 6])
    canceled_result = canceled._execute_joint_jog(cancel_goal)
    assert cancel_goal.outcome == "canceled"
    assert canceled_result.state_uncertain is False

    monkeypatch.setattr(module.rclpy, "ok", lambda: False)
    timeout_goal = _Goal()
    timed_out = _server([[0.0] * 6])
    timeout_result = timed_out._execute_joint_jog(timeout_goal)
    assert timeout_goal.outcome == "aborted"
    assert timeout_result.state_uncertain is True

    unsafe_goal = _Goal()
    unsafe = _server([[0.0] * 6], safety=False)
    unsafe_result = unsafe._execute_joint_jog(unsafe_goal)
    assert unsafe_goal.outcome == "aborted"
    assert "outside safety limits" in unsafe_result.error_string
    assert unsafe_result.state_uncertain is False


def test_ur5e_world_z_step_preserves_exact_rtde_rotation_vector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _rtde_server_module()

    class _Result:
        pass

    class _Feedback:
        pass

    module.MoveUR5eRelativeCartesian = SimpleNamespace(
        Result=_Result,
        Feedback=_Feedback,
    )
    monkeypatch.setattr(module.rclpy, "ok", lambda: True)
    monkeypatch.setattr(module, "UR5E_RTDE_STATIONARY_HOLD_SEC", 0.0)
    server = object.__new__(module.UR5eRTDETrajectoryServer)
    server._active_lock = threading.Lock()
    server._shutdown_requested = False
    server._rtde_reset_required = False
    server._rtde_reset_reason = ""
    server._active_goal = None
    server._active_goal_status = None
    server._active_motion_kind = ""
    server._latched_terminal_status = None
    server._connect_control_for_goal = lambda: None
    server._joint_states_fresh = lambda: True
    server._read_actual_q = lambda: [0.0] * 6
    server._ensure_control_program_for_goal = lambda: None
    frame_validations: list[bool] = []
    server._cartesian_frame_validation = lambda: (
        frame_validations.append(True) or (True, "ready", 0.0, 0.0)
    )
    world_base = ((0.0, 0.0, 0.9), (0.0, 0.0, 0.0, 1.0))
    server._validated_cartesian_world_base = lambda: (
        world_base,
        "ready",
        0.0,
        0.0,
    )
    server._active_tcp_offset = lambda: (
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )
    start_pose = [0.2, 0.3, 0.3, 2.123456, -0.456789, 0.912345]
    target_pose = [0.2, 0.3, 0.31, *start_pose[3:6]]
    poses = iter((start_pose, target_pose))
    server._read_actual_tcp_pose = lambda: list(next(poses))
    server._read_actual_qd = lambda: [0.0] * 6
    server.control = SimpleNamespace(isPoseWithinSafetyLimits=lambda _pose: True)
    move_l_targets: list[list[float]] = []
    server._execute_movel_pose = lambda pose, **_kwargs: (
        move_l_targets.append(list(pose)) or True
    )
    server._write_status = lambda _status: None
    server._write_terminal_status = lambda _status: None
    server._stop_motion = lambda: None
    request = SimpleNamespace(
        world_translation_m=SimpleNamespace(x=0.0, y=0.0, z=0.010),
        speed_m_s=0.05,
        acceleration_m_s2=0.10,
    )
    goal = SimpleNamespace(
        request=request,
        is_cancel_requested=False,
        publish_feedback=lambda _feedback: None,
        succeed=lambda: None,
        abort=lambda: pytest.fail("valid translation-only Step was aborted"),
        canceled=lambda: pytest.fail("valid translation-only Step was canceled"),
    )

    result = server._execute_relative_cartesian(goal)

    assert result.error_code == 0
    assert frame_validations == [True]
    assert move_l_targets == [target_pose]
    assert move_l_targets[0][3:6] == start_pose[3:6]


def test_ur5e_active_tcp_offset_round_trip_uses_base_and_tool0_only() -> None:
    module = _rtde_server_module()
    server = object.__new__(module.UR5eRTDETrajectoryServer)
    world_base = ((0.5, -0.2, 0.8), (0.0, 0.0, 1.0, 0.0))
    base_tool0 = ((0.2, -0.3, 0.4), (0.0, 0.0, 0.0, 1.0))
    tool0_tcp = ((0.0, 0.0, 0.218), (0.0, 0.0, 0.0, 1.0))
    actual_base_tcp = module._compose_transform(base_tool0, tool0_tcp)
    expected_world_tool0 = module._compose_transform(world_base, base_tool0)

    reconstructed = server._world_tool0_from_actual_tcp(
        actual_base_tcp,
        world_base=world_base,
        tool0_tcp=tool0_tcp,
    )

    position_error, orientation_error = module._pose_errors(
        reconstructed,
        expected_world_tool0,
    )
    assert position_error < 1e-12
    assert orientation_error < 1e-12
    conversion_source = inspect.getsource(server._resolve_cartesian_target)
    validation_source = inspect.getsource(server._cartesian_frame_validation)
    mount_validation_source = inspect.getsource(
        server._cartesian_frame_validation_with_world_base
    )
    assert "_validated_cartesian_world_base()" in conversion_source
    assert '_lookup_rigid_transform("world", "base")' not in conversion_source
    assert '_lookup_rigid_transform("world", "base")' in mount_validation_source
    assert '"base_link"' not in conversion_source + validation_source
    assert '"flange"' not in conversion_source + validation_source


def test_ur5e_cartesian_world_base_matches_protected_mount(tmp_path: Path) -> None:
    module = _rtde_server_module()
    server = object.__new__(module.UR5eRTDETrajectoryServer)
    expected_world_base = module._configured_cartesian_world_base()
    actual_base_tcp = (
        (0.1, -0.2, 0.3),
        (0.0, 0.0, 0.0, 1.0),
    )
    tool0_tcp = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
    expected_world_tool0 = module._compose_transform(
        expected_world_base,
        actual_base_tcp,
    )
    transforms = {
        ("world", "base"): expected_world_base,
        ("world", "tool0"): expected_world_tool0,
    }
    server._lookup_rigid_transform = lambda target, source: transforms[(target, source)]
    server._active_tcp_offset = lambda: tool0_tcp
    server._read_actual_tcp_transform = lambda: actual_base_tcp

    ready, message, position_error, orientation_error = (
        server._cartesian_frame_validation()
    )

    assert expected_world_base[0] == pytest.approx((0.0, 0.5, 1.021))
    assert expected_world_base[1] == pytest.approx((0.0, 0.0, 0.0, 1.0))
    assert ready is True
    assert message == "UR5e Cartesian frame validation ready"
    assert position_error == pytest.approx(0.0)
    assert orientation_error == pytest.approx(0.0)
    assert server._cartesian_world_base_ready is True
    assert server._cartesian_world_base_expected == expected_world_base
    assert server._cartesian_world_base_observed == expected_world_base
    assert server._cartesian_world_base_position_error_m == pytest.approx(0.0)
    assert server._cartesian_world_base_orientation_error_rad == pytest.approx(0.0)
    server.monitor_only = False
    server.ros_domain_id = 42
    server.status_file = tmp_path / "status.json"
    server.terminal_status_file = tmp_path / "terminal.json"
    server._status_lock = threading.Lock()
    server._insertion_demonstration_lock = threading.Lock()
    server._active_insertion_demonstration_status = {}
    server._rtde_reset_required = False
    server._write_status({"state": "ready"})
    status = json.loads(server.status_file.read_text(encoding="utf-8"))
    assert status["cartesian_world_base_ready"] is True
    assert status["cartesian_world_base_expected"]["y"] == pytest.approx(0.5)
    assert status["cartesian_world_base_observed"]["y"] == pytest.approx(0.5)
    assert status["cartesian_world_base_position_error_m"] == pytest.approx(0.0)
    assert status["cartesian_world_base_orientation_error_rad"] == pytest.approx(0.0)


def test_ur5e_cartesian_world_base_rejects_internally_consistent_wrong_tf_once() -> None:
    module = _rtde_server_module()
    server = object.__new__(module.UR5eRTDETrajectoryServer)
    wrong_world_base = (
        (0.0, 0.0, 1.021),
        (0.0, 0.0, 0.0, 1.0),
    )
    correct_world_base = module._configured_cartesian_world_base()
    actual_base_tcp = (
        (0.1, -0.2, 0.3),
        (0.0, 0.0, 0.0, 1.0),
    )
    wrong_world_tool0 = module._compose_transform(
        wrong_world_base,
        actual_base_tcp,
    )
    world_base_values = iter((wrong_world_base, correct_world_base))
    world_base_lookups: list[bool] = []

    def lookup(target: str, source: str) -> Any:
        if (target, source) == ("world", "base"):
            world_base_lookups.append(True)
            return next(world_base_values)
        if (target, source) == ("world", "tool0"):
            return wrong_world_tool0
        return pytest.fail(f"unexpected TF lookup: {target} <- {source}")

    server._lookup_rigid_transform = lookup
    server._active_tcp_offset = lambda: (
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )
    server._read_actual_tcp_transform = lambda: actual_base_tcp

    ready, message, _position_error, _orientation_error = (
        server._cartesian_frame_validation()
    )

    assert ready is False
    assert "protected ur5e.rtde.cartesian_world_base" in message
    assert world_base_lookups == [True]
    assert server._cartesian_world_base_ready is False
    assert server._cartesian_world_base_expected == correct_world_base
    assert server._cartesian_world_base_observed == wrong_world_base
    assert server._cartesian_world_base_position_error_m == pytest.approx(0.5)
    assert server._cartesian_world_base_orientation_error_rad == pytest.approx(0.0)


def test_ur5e_cartesian_world_base_missing_config_is_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _rtde_server_module()
    monkeypatch.setattr(module, "UR5E_RTDE_CARTESIAN_WORLD_BASE", None)
    monkeypatch.setattr(
        module,
        "UR5E_RTDE_CARTESIAN_WORLD_BASE_CONFIG_ERROR",
        "ur5e.rtde.cartesian_world_base.y_m is missing or is not a finite number",
    )
    server = object.__new__(module.UR5eRTDETrajectoryServer)
    server._lookup_rigid_transform = lambda *_args: pytest.fail(
        "missing protected mount config must reject before TF lookup"
    )

    ready, message, position_error, orientation_error = (
        server._cartesian_frame_validation()
    )

    assert ready is False
    assert "cartesian_world_base.y_m is missing" in message
    assert math.isinf(position_error)
    assert math.isinf(orientation_error)
    assert server._cartesian_world_base_expected is None
    assert server._cartesian_world_base_observed is None


def test_ur5e_cartesian_world_base_parser_has_no_invalid_value_fallback(
    tmp_path: Path,
) -> None:
    module = _rtde_server_module()
    config_path = tmp_path / "invalid_mount.yaml"
    config_path.write_text(
        """ur5e:
  rtde:
    cartesian_world_base:
      x_m: 0.0
      y_m: 0.5
      z_m: 1.021
      roll_rad: 0.0
      pitch_rad: 0.0
      yaw_rad: .nan
""",
        encoding="utf-8",
    )

    module._apply_hardware_arms_config(config_path)

    assert module.UR5E_RTDE_CARTESIAN_WORLD_BASE is None
    assert (
        module.UR5E_RTDE_CARTESIAN_WORLD_BASE_CONFIG_ERROR
        == "ur5e.rtde.cartesian_world_base.yaw_rad is missing or is not a finite number"
    )
    with pytest.raises(RuntimeError, match="protected ur5e.rtde.cartesian_world_base"):
        module._configured_cartesian_world_base()


def test_ur5e_world_frame_conversions_have_no_unvalidated_mount_lookup() -> None:
    module = _rtde_server_module()
    server = object.__new__(module.UR5eRTDETrajectoryServer)

    for method in (
        server._resolve_cartesian_target,
        server._execute_insert,
        server._execute_insertion_demonstration,
        server._execute_relative_cartesian,
        server._set_cartesian_jog,
    ):
        source = inspect.getsource(method)
        assert '_lookup_rigid_transform("world", "base")' not in source


def _install_wrong_ur5e_world_base(server: Any) -> None:
    wrong_world_base = (
        (0.0, 0.0, 1.021),
        (0.0, 0.0, 0.0, 1.0),
    )
    server._lookup_rigid_transform = lambda target, source: (
        wrong_world_base
        if (target, source) == ("world", "base")
        else pytest.fail(f"wrong mount must reject before {target} <- {source} lookup")
    )
    server._active_tcp_offset = lambda: pytest.fail(
        "wrong mount must reject before reading the active TCP offset"
    )
    server._read_actual_tcp_transform = lambda: pytest.fail(
        "wrong mount must reject before reading the active TCP pose"
    )


def test_ur5e_main_cartesian_rejects_wrong_mount_before_move_l() -> None:
    module = _rtde_server_module()

    class _Result:
        pass

    module.MoveUR5eCartesian = SimpleNamespace(Result=_Result)
    server = object.__new__(module.UR5eRTDETrajectoryServer)
    server._active_lock = threading.Lock()
    server._shutdown_requested = False
    server._rtde_reset_required = False
    server._rtde_reset_reason = ""
    server._latched_terminal_status = None
    server._active_goal = None
    server._active_goal_status = None
    server._active_motion_kind = ""
    server._connect_control_for_goal = lambda: None
    server._joint_states_fresh = lambda: True
    server._read_actual_q = lambda: [0.0] * 6
    server._ensure_control_program_for_goal = lambda: None
    server.control = SimpleNamespace(
        moveL=lambda *_args, **_kwargs: pytest.fail(
            "wrong mount must reject before moveL"
        )
    )
    _install_wrong_ur5e_world_base(server)
    server._finish_active_goal_status = lambda *_args, **_kwargs: None
    server._clear_active_goal = lambda _goal: None
    server._stop_motion = lambda: pytest.fail(
        "a pre-motion mount rejection must not stop unknown motion"
    )
    request = SimpleNamespace(
        target_tool0_pose=SimpleNamespace(
            header=SimpleNamespace(frame_id="world"),
            pose=SimpleNamespace(
                position=SimpleNamespace(x=0.0, y=0.1, z=1.2),
                orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
            ),
        ),
        speed_m_s=0.05,
        acceleration_m_s2=0.10,
    )
    outcomes: list[str] = []
    goal = SimpleNamespace(
        request=request,
        abort=lambda: outcomes.append("aborted"),
        is_cancel_requested=False,
    )

    result = server._execute_cartesian(goal)

    assert result.error_code == -2
    assert "protected ur5e.rtde.cartesian_world_base" in result.error_string
    assert outcomes == ["aborted"]


def test_ur5e_relative_cartesian_rejects_wrong_mount_before_move_l() -> None:
    module = _rtde_server_module()

    class _Result:
        pass

    module.MoveUR5eRelativeCartesian = SimpleNamespace(Result=_Result)
    server = object.__new__(module.UR5eRTDETrajectoryServer)
    server._active_lock = threading.Lock()
    server._shutdown_requested = False
    server._rtde_reset_required = False
    server._rtde_reset_reason = ""
    server._active_goal = None
    server._active_goal_status = None
    server._active_motion_kind = ""
    server._latched_terminal_status = None
    server._connect_control_for_goal = lambda: None
    server._joint_states_fresh = lambda: True
    server._read_actual_q = lambda: [0.0] * 6
    server._ensure_control_program_for_goal = lambda: None
    server.control = SimpleNamespace(
        moveL=lambda *_args, **_kwargs: pytest.fail(
            "wrong mount must reject before moveL"
        )
    )
    _install_wrong_ur5e_world_base(server)
    server._finish_active_goal_status = lambda *_args, **_kwargs: None
    server._clear_active_goal = lambda _goal: None
    server._stop_motion = lambda: pytest.fail(
        "a pre-motion mount rejection must not stop unknown motion"
    )
    outcomes: list[str] = []
    goal = SimpleNamespace(
        request=SimpleNamespace(
            world_translation_m=SimpleNamespace(x=0.01, y=0.0, z=0.0),
            speed_m_s=0.05,
            acceleration_m_s2=0.10,
        ),
        abort=lambda: outcomes.append("aborted"),
    )

    result = server._execute_relative_cartesian(goal)

    assert result.error_code == -2
    assert "protected ur5e.rtde.cartesian_world_base" in result.error_string
    assert outcomes == ["aborted"]


def test_ur5e_cartesian_smooth_hold_rejects_wrong_mount_before_jog_start() -> None:
    module = _rtde_server_module()
    server = object.__new__(module.UR5eRTDETrajectoryServer)
    server._shutdown_requested = False
    server._active_lock = threading.Lock()
    server._active_goal = None
    server._jog_session_token = object()
    server._connect_control_for_goal = lambda: None
    server._joint_states_fresh = lambda: True
    server.control = SimpleNamespace(
        jogStart=lambda *_args, **_kwargs: pytest.fail(
            "wrong mount must reject before jogStart"
        )
    )
    statuses: list[dict[str, Any]] = []
    server._write_status = lambda status: statuses.append(dict(status))
    _install_wrong_ur5e_world_base(server)
    request = SimpleNamespace(
        stop=False,
        world_linear_velocity_m_s=SimpleNamespace(x=0.01, y=0.0, z=0.0),
        acceleration_m_s2=0.10,
        watchdog_sec=0.25,
    )
    response = SimpleNamespace(accepted=None, message="")

    result = server._set_cartesian_jog(request, response)

    assert result is response
    assert response.accepted is False
    assert "protected ur5e.rtde.cartesian_world_base" in response.message
    assert statuses[-1]["state"] == "blocked"


@pytest.mark.parametrize(
    ("speed_mm_s", "speed_m_s"),
    [
        (5.0, 0.005),
        (100.0, 0.100),
    ],
)
def test_ur5e_cartesian_smooth_hold_sends_metres_per_second_to_jog_start(
    speed_mm_s: float,
    speed_m_s: float,
) -> None:
    """Verify service m/s is converted to ur_rtde jogStart translation mm/s."""
    module = _rtde_server_module()
    server = object.__new__(module.UR5eRTDETrajectoryServer)
    server._shutdown_requested = False
    server._active_lock = threading.Lock()
    server._active_goal = None
    server._active_motion_kind = ""
    server._active_goal_status = None
    server._jog_session_token = object()
    server._connect_control_for_goal = lambda: None
    server._joint_states_fresh = lambda: True
    frame_validations: list[bool] = []
    server._validated_cartesian_world_base = lambda: (
        frame_validations.append(True)
        or (
            ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
            "UR5e Cartesian frame validation ready",
            0.0,
            0.0,
        )
    )
    server._cartesian_frame_validation = lambda: pytest.fail(
        "Smooth Hold must not repeat full frame validation after the accepted sample"
    )
    server._read_actual_tcp_pose = lambda: [0.0] * 6
    jog_calls: list[tuple[list[float], int, float]] = []
    server.control = SimpleNamespace(
        FEATURE_BASE=0,
        isPoseWithinSafetyLimits=lambda _pose: True,
        jogStart=lambda speeds, feature, acceleration: (
            jog_calls.append((list(speeds), int(feature), float(acceleration))) or True
        ),
    )
    statuses: list[dict[str, Any]] = []
    server._write_status = lambda status: statuses.append(dict(status))
    request = SimpleNamespace(
        stop=False,
        world_linear_velocity_m_s=SimpleNamespace(x=speed_m_s, y=0.0, z=0.0),
        acceleration_m_s2=0.10,
        watchdog_sec=0.25,
    )
    response = SimpleNamespace(accepted=None, message="")

    first_result = server._set_cartesian_jog(request, response)
    second_response = SimpleNamespace(accepted=None, message="")
    second_result = server._set_cartesian_jog(request, second_response)

    assert first_result is response
    assert second_result is second_response
    assert response.accepted is True
    assert second_response.accepted is True
    assert frame_validations == [True]
    assert jog_calls == [
        ([speed_mm_s, 0.0, 0.0, 0.0, 0.0, 0.0], 0, 0.10),
        ([speed_mm_s, 0.0, 0.0, 0.0, 0.0, 0.0], 0, 0.10),
    ]
    assert statuses[-1]["world_linear_velocity_m_s"] == pytest.approx(
        [speed_m_s, 0.0, 0.0]
    )
    assert statuses[-1]["base_linear_velocity_mm_s"] == pytest.approx(
        [speed_mm_s, 0.0, 0.0]
    )


def test_ur5e_insertion_demonstration_rejects_wrong_mount_before_trace(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _rtde_server_module()

    class _Result:
        pass

    module.RecordUR5eInsertionDemonstration = SimpleNamespace(Result=_Result)
    monkeypatch.setattr(module, "INSERT_DEMONSTRATION_TRACE_ROOT", tmp_path / "traces")
    server = object.__new__(module.UR5eRTDETrajectoryServer)
    server.monitor_only = False
    server.receive = object()
    server.control = object()
    server._rtde_reset_required = False
    server._rtde_reset_reason = ""
    server._confirm_stationary_after_stop = lambda *, timeout_sec: True
    server._read_feedback_timestamp = lambda: pytest.fail(
        "wrong mount must reject before trace sampling"
    )
    statuses: list[dict[str, Any]] = []
    server._write_status = lambda status: statuses.append(dict(status))
    _install_wrong_ur5e_world_base(server)
    outcomes: list[str] = []
    request = SimpleNamespace(
        recording_id="wrong-mount-demonstration",
        part_name="MG",
        destination_location="assembly_board-v1",
        context_sha256="a" * 64,
        expected_start_tool0_pose=SimpleNamespace(
            header=SimpleNamespace(frame_id="world"),
            pose=SimpleNamespace(
                position=SimpleNamespace(x=0.0, y=0.1, z=1.2),
                orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
            ),
        ),
        max_duration_sec=1.0,
    )
    goal = SimpleNamespace(
        request=request,
        abort=lambda: outcomes.append("aborted"),
    )

    result = server._execute_insertion_demonstration(goal)

    assert result.error_code == -1
    assert "protected ur5e.rtde.cartesian_world_base" in result.error_string
    assert outcomes == ["aborted"]
    assert statuses[-1]["state"] == "blocked"
    assert not (tmp_path / "traces").exists()


def test_ur5e_prusa_mk4_2_mg_target_is_inside_configured_coarse_reach() -> None:
    module = _rtde_server_module()
    target = (
        (0.3627526806567198, -0.1363532373779684, 1.4525105390809694),
        (0.0, 0.0, 0.0, 1.0),
    )

    assert module.UR5E_RTDE_CARTESIAN_REACH_ORIGIN == pytest.approx(
        (0.0, 0.5, 1.021)
    )
    assert module.UR5E_RTDE_CARTESIAN_REACH_RADIUS_M == pytest.approx(0.8)
    assert module._workspace_error(target) is None


@pytest.mark.parametrize(
    ("world_delta", "expected_base_delta"),
    [
        ((0.01, 0.0, 0.0), (-0.01, 0.0, 0.0)),
        ((-0.01, 0.0, 0.0), (0.01, 0.0, 0.0)),
        ((0.0, 0.01, 0.0), (0.0, -0.01, 0.0)),
        ((0.0, -0.01, 0.0), (0.0, 0.01, 0.0)),
        ((0.0, 0.0, 0.01), (0.0, 0.0, 0.01)),
        ((0.0, 0.0, -0.01), (0.0, 0.0, -0.01)),
    ],
)
def test_xarm6_world_axes_rotate_into_pi_yaw_base(
    world_delta: tuple[float, float, float],
    expected_base_delta: tuple[float, float, float],
) -> None:
    module = _teleop_module()
    teleop = object.__new__(module.KeyboardTeleop)
    yaw_pi = SimpleNamespace(
        transform=SimpleNamespace(
            rotation=SimpleNamespace(x=0.0, y=0.0, z=1.0, w=0.0)
        )
    )
    teleop.tf_buffer = SimpleNamespace(
        lookup_transform=lambda target, source, _time: (
            yaw_pi
            if (target, source) == ("world", "link_base")
            else pytest.fail(f"unexpected TF lookup: {target} <- {source}")
        )
    )

    base_delta = teleop._world_vector_in_robot_base(world_delta)

    assert base_delta == pytest.approx(expected_base_delta, abs=1e-12)


def test_xarm6_step_sends_relative_translation_with_zero_rotation() -> None:
    module = _teleop_module()
    teleop = object.__new__(module.KeyboardTeleop)
    teleop.xarm6_hardware_cartesian_service = module.XARM6_HARDWARE_CARTESIAN_SERVICE
    teleop.xarm6_hardware_cartesian_speed_mm_s = 50.0
    teleop.xarm6_hardware_cartesian_acceleration_mm_s2 = 100.0
    teleop.xarm6_hardware_cartesian_position_tolerance_m = 0.003
    teleop.xarm6_hardware_workspace_bounds = {
        "x_min_m": -0.6,
        "x_max_m": 0.6,
        "y_min_m": -1.0,
        "y_max_m": 0.1,
        "z_min_m": 0.9,
        "z_max_m": 1.5,
    }
    teleop.xarm6_hardware_cartesian_client = SimpleNamespace(
        wait_for_service=lambda timeout_sec: timeout_sec == pytest.approx(2.0)
    )
    teleop._xarm6_cartesian_readiness = lambda: (True, "ready", {})
    current = module.Pose()
    current.position.x = 0.1
    current.position.y = -0.4
    current.position.z = 1.2
    current.orientation.w = 1.0
    teleop._get_world_ee_pose = lambda _robot: current
    teleop._world_vector_in_robot_base = lambda _delta: (0.01, 0.0, 0.0)
    snapshots = iter(
        (
            {"pose": [100.0, -400.0, 1200.0, 0.0, 0.0, 1.2]},
            {"pose": [110.0, -400.0, 1200.0, 0.0, 0.0, 1.2]},
        )
    )
    teleop._xarm6_robot_state_snapshot = lambda: (next(snapshots), "")
    handoffs: list[str] = []
    teleop._xarm6_prepare_firmware_cartesian_mode = lambda: (
        handoffs.append("prepare_mode_0") or (True, "ready")
    )
    teleop._xarm6_restore_trajectory_control = lambda: (
        handoffs.append("restore_mode_1") or (True, "restored")
    )
    requests: list[Any] = []
    teleop._call_service = lambda _client, request, timeout_sec: (
        requests.append(request) or (SimpleNamespace(ret=0, message="OK"), None)
    )

    ok, message = teleop._move_xarm6_relative_cartesian(
        (0.01, 0.0, 0.0),
        velocity_scale=0.35,
    )

    assert ok is True
    assert "relative Step succeeded" in message
    assert len(requests) == 1
    assert list(requests[0].pose) == pytest.approx(
        [10.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    )
    assert requests[0].relative is True
    assert teleop._xarm6_cartesian_motion_attempted is True
    assert handoffs == ["prepare_mode_0", "restore_mode_1"]


def test_xarm6_step_session_reuses_mode_zero_and_exact_speed() -> None:
    module = _teleop_module()
    assert module.MoveCartesian is not None
    teleop = object.__new__(module.KeyboardTeleop)
    teleop._xarm6_cartesian_session_mode = "step"
    teleop._xarm6_cartesian_motion_attempted = False
    teleop.xarm6_hardware_cartesian_service = module.XARM6_HARDWARE_CARTESIAN_SERVICE
    teleop.xarm6_hardware_cartesian_speed_mm_s = 50.0
    teleop.xarm6_hardware_cartesian_max_speed_mm_s = 100.0
    teleop.xarm6_hardware_cartesian_acceleration_mm_s2 = 42.25
    teleop.xarm6_hardware_cartesian_position_tolerance_m = 0.003
    teleop.xarm6_hardware_workspace_bounds = {
        "x_min_m": -0.6,
        "x_max_m": 0.6,
        "y_min_m": -1.0,
        "y_max_m": 0.1,
        "z_min_m": 0.9,
        "z_max_m": 1.5,
    }
    teleop.xarm6_hardware_cartesian_client = SimpleNamespace(
        wait_for_service=lambda timeout_sec: timeout_sec == pytest.approx(2.0)
    )
    current_world = module.Pose()
    current_world.position.x = 0.1
    current_world.position.y = -0.4
    current_world.position.z = 1.2
    current_world.orientation.w = 1.0
    teleop._get_world_ee_pose = lambda _robot: current_world
    teleop._world_vector_in_robot_base = lambda delta: tuple(delta)
    state = {"mode": 0, "state": 0, "pose": [100.0, -400.0, 1200.0, 0.0, 0.0, 0.0]}
    teleop._xarm6_robot_state_snapshot = lambda: (dict(state), "")
    teleop._xarm6_cartesian_readiness = lambda **_kwargs: pytest.fail(
        "prepared Step commands must not rerun full Cartesian readiness"
    )
    teleop._xarm6_confirm_cartesian_mode = lambda _mode: pytest.fail(
        "prepared Step commands must not poll controller lifecycle"
    )
    requests: list[Any] = []

    def _call_service(_client: Any, request: Any, timeout_sec: float) -> tuple[Any, None]:
        assert timeout_sec == pytest.approx(float(request.timeout) + 2.0)
        requests.append(request)
        state["pose"][0] += float(request.pose[0])
        state["pose"][1] += float(request.pose[1])
        state["pose"][2] += float(request.pose[2])
        return SimpleNamespace(ret=0, message="OK"), None

    teleop._call_service = _call_service

    for delta in ((0.01, 0.0, 0.0), (0.0, 0.01, 0.0)):
        ok, message = teleop._move_xarm6_relative_cartesian(
            delta,
            velocity_scale=0.1,
            speed_mm_s=12.3,
            restore_trajectory_control=False,
        )
        assert ok is True, message

    ok, message = teleop._move_xarm6_relative_cartesian(
        (0.0, 0.0, 0.01),
        velocity_scale=0.1,
        speed_mm_s=100.0,
        restore_trajectory_control=False,
    )
    assert ok is True, message
    ok, message = teleop._move_xarm6_relative_cartesian(
        (0.0, 0.0, 0.01),
        velocity_scale=1.0,
        restore_trajectory_control=False,
    )
    assert ok is True, message
    ok, message = teleop._move_xarm6_relative_cartesian(
        (0.0, 0.0, 0.01),
        velocity_scale=1.0,
        speed_mm_s=0.025,
        restore_trajectory_control=False,
    )
    assert ok is True, message
    ok, message = teleop._move_xarm6_relative_cartesian(
        (0.0, 0.0, 0.01),
        velocity_scale=1.0,
        speed_mm_s=100.025,
        restore_trajectory_control=False,
    )
    assert ok is False
    assert "(0, 100.000] mm/s" in message

    assert len(requests) == 5
    assert [request.speed for request in requests] == pytest.approx(
        [12.3, 12.3, 100.0, 50.0, 0.025]
    )
    assert [request.relative for request in requests] == [True] * 5


def test_xarm6_workspace_rejection_is_identified_before_motion_attempt() -> None:
    module = _teleop_module()
    teleop = object.__new__(module.KeyboardTeleop)
    teleop.xarm6_hardware_cartesian_service = module.XARM6_HARDWARE_CARTESIAN_SERVICE
    teleop.xarm6_hardware_workspace_bounds = {
        "x_min_m": -0.6,
        "x_max_m": 0.6,
        "y_min_m": -1.0,
        "y_max_m": 0.1,
        "z_min_m": 0.9,
        "z_max_m": 1.6,
    }
    teleop.xarm6_hardware_cartesian_client = SimpleNamespace(
        wait_for_service=lambda timeout_sec: timeout_sec == pytest.approx(2.0)
    )
    teleop._xarm6_cartesian_readiness = lambda: (True, "ready", {})
    current = module.Pose()
    current.position.x = 0.1
    current.position.y = -0.4
    current.position.z = 1.59
    current.orientation.w = 1.0
    teleop._get_world_ee_pose = lambda _robot: current
    teleop._call_service = lambda *_args, **_kwargs: pytest.fail(
        "workspace rejection must precede the xArm6 service call"
    )

    ok, message = teleop._move_xarm6_relative_cartesian(
        (0.0, 0.0, 0.02),
        velocity_scale=0.35,
    )

    assert ok is False
    assert "outside [0.900000, 1.600000]" in message
    assert teleop._xarm6_cartesian_motion_attempted is False


def test_xarm6_smooth_hold_uses_the_same_cartesian_speed_limit() -> None:
    module = _teleop_module()
    teleop = object.__new__(module.KeyboardTeleop)
    teleop.xarm6_hardware_cartesian_speed_mm_s = 50.0
    teleop.xarm6_hardware_cartesian_max_speed_mm_s = 100.0
    commands: list[tuple[list[float], float]] = []
    teleop._set_xarm6_cartesian_jog = lambda world_velocity, watchdog_sec: (
        commands.append((list(world_velocity), float(watchdog_sec))) or (True, "OK")
    )

    assert teleop.set_cartesian_jog(
        "xarm6",
        "x",
        30.0,
        watchdog_sec=0.3,
    ) == (True, "OK")
    assert teleop.set_cartesian_jog(
        "xarm6",
        "y",
        -100.0,
        watchdog_sec=0.3,
    ) == (True, "OK")
    assert teleop.set_cartesian_jog(
        "xarm6",
        "z",
        0.025,
        watchdog_sec=0.3,
    ) == (True, "OK")
    ok, message = teleop.set_cartesian_jog(
        "xarm6",
        "x",
        100.025,
        watchdog_sec=0.3,
    )
    assert ok is False
    assert "(0, 100.000] mm/s" in message
    assert commands == [
        ([0.03, 0.0, 0.0], 0.3),
        ([0.0, -0.1, 0.0], 0.3),
        ([0.0, 0.0, 0.000025], 0.3),
    ]


def test_xarm6_cartesian_handoff_releases_and_restores_trajectory_controller() -> None:
    module = _teleop_module()
    teleop = object.__new__(module.KeyboardTeleop)
    teleop._xarm6_cartesian_motion_attempted = False
    calls: list[tuple[str, int | str]] = []
    teleop._xarm6_set_int16 = lambda suffix, value: (
        calls.append((suffix, value)) or (True, "OK")
    )
    teleop._xarm6_wait_for_mode = lambda mode: (
        calls.append(("wait_mode", mode)) or (True, "OK")
    )
    teleop._xarm6_wait_for_trajectory_controller_state = lambda state: (
        calls.append(("wait_controller", state)) or (True, "OK")
    )

    assert teleop._xarm6_prepare_firmware_cartesian_mode() == (
        True,
        "xArm6 firmware Cartesian Mode 0 ready",
    )
    assert teleop._xarm6_restore_trajectory_control() == (
        True,
        "xArm6 trajectory controller Mode 1 restored",
    )
    assert calls == [
        ("set_mode", 0),
        ("set_state", 0),
        ("wait_mode", 0),
        ("wait_controller", "inactive"),
        ("set_mode", 1),
        ("set_state", 0),
        ("wait_mode", 1),
        ("wait_controller", "active"),
    ]


def _initialize_xarm6_smooth_state(teleop: Any) -> None:
    teleop._xarm6_smooth_state_lock = threading.Lock()
    teleop._xarm6_smooth_service_lock = threading.Lock()
    teleop._xarm6_smooth_speeds = [0.0] * 6
    teleop._xarm6_smooth_watchdog_sec = 0.50
    teleop._xarm6_smooth_heartbeat_monotonic = time.monotonic()
    teleop._xarm6_smooth_pending_error = ""
    teleop._xarm6_smooth_refresh_stop = threading.Event()
    teleop._xarm6_smooth_refresh_thread = None


def _initialize_ur5e_smooth_state(teleop: Any) -> None:
    teleop._ur5e_smooth_state_lock = threading.Lock()
    teleop._ur5e_smooth_service_lock = threading.Lock()
    teleop._ur5e_smooth_world_velocity_m_s = [0.0] * 3
    teleop._ur5e_smooth_watchdog_sec = 0.50
    teleop._ur5e_smooth_heartbeat_monotonic = time.monotonic()
    teleop._ur5e_smooth_pending_error = ""
    teleop._ur5e_smooth_refresh_stop = threading.Event()
    teleop._ur5e_smooth_refresh_thread = None
    teleop._ur5e_last_stop_motion_confirmed = True


def test_ur5e_smooth_hold_starts_refresh_beside_ros_service() -> None:
    module = _teleop_module()
    assert module.SetUR5eCartesianJog is not None
    teleop = object.__new__(module.KeyboardTeleop)
    teleop._ur5e_smooth_active = False
    _initialize_ur5e_smooth_state(teleop)
    teleop.ur5e_hardware_cartesian_jog_service = (
        "/cais_ur5e_rtde_cartesian_controller/set_cartesian_jog"
    )
    teleop.ur5e_hardware_cartesian_jog_client = SimpleNamespace(
        wait_for_service=lambda timeout_sec: timeout_sec == pytest.approx(2.0)
    )
    sends: list[tuple[list[float], float, bool, float]] = []
    teleop._send_ur5e_cartesian_jog = (
        lambda velocity, watchdog, *, stop, timeout_sec: (
            sends.append(
                (
                    list(velocity),
                    float(watchdog),
                    bool(stop),
                    float(timeout_sec),
                )
            )
            or (True, "UR5e Cartesian Smooth Hold active")
        )
    )
    refresh_starts: list[bool] = []
    teleop._start_ur5e_cartesian_jog_refresh = lambda: (
        refresh_starts.append(True) or (True, "OK")
    )

    assert teleop._set_ur5e_cartesian_jog((0.08, 0.0, 0.0), 0.50) == (
        True,
        "UR5e Cartesian Smooth Hold active",
    )
    assert teleop._ur5e_smooth_active is True
    assert sends == [([0.08, 0.0, 0.0], 0.50, False, 3.0)]
    assert refresh_starts == [True]


def test_ur5e_smooth_hold_update_refreshes_ui_heartbeat_without_ros_round_trip() -> None:
    module = _teleop_module()
    assert module.SetUR5eCartesianJog is not None
    teleop = object.__new__(module.KeyboardTeleop)
    teleop._ur5e_smooth_active = True
    _initialize_ur5e_smooth_state(teleop)
    teleop.ur5e_hardware_cartesian_jog_service = (
        "/cais_ur5e_rtde_cartesian_controller/set_cartesian_jog"
    )
    teleop.ur5e_hardware_cartesian_jog_client = SimpleNamespace(
        wait_for_service=lambda **_kwargs: pytest.fail(
            "the UI heartbeat must not wait for service discovery"
        )
    )
    teleop._send_ur5e_cartesian_jog = lambda *_args, **_kwargs: pytest.fail(
        "the UI heartbeat must not wait for the ROS service round trip"
    )
    previous_heartbeat = teleop._ur5e_smooth_heartbeat_monotonic

    assert teleop._set_ur5e_cartesian_jog((0.0, -0.04, 0.0), 0.50) == (
        True,
        "UR5e Cartesian Smooth Hold active",
    )
    assert teleop._ur5e_smooth_world_velocity_m_s == pytest.approx(
        [0.0, -0.04, 0.0]
    )
    assert teleop._ur5e_smooth_heartbeat_monotonic >= previous_heartbeat


def test_ur5e_smooth_hold_refresh_runs_next_to_ros_service() -> None:
    module = _teleop_module()
    teleop = object.__new__(module.KeyboardTeleop)
    teleop._ur5e_smooth_active = True
    _initialize_ur5e_smooth_state(teleop)
    teleop._ur5e_smooth_world_velocity_m_s = [0.0, 0.0, -0.025]
    refreshes: list[tuple[list[float], float, bool, float]] = []
    teleop._send_ur5e_cartesian_jog = (
        lambda velocity, watchdog, *, stop, timeout_sec: (
            refreshes.append(
                (
                    list(velocity),
                    float(watchdog),
                    bool(stop),
                    float(timeout_sec),
                )
            )
            or (True, "UR5e Cartesian Smooth Hold active")
        )
    )

    assert teleop._ur5e_refresh_cartesian_jog_once() is True
    assert refreshes == [([0.0, 0.0, -0.025], 0.50, False, 0.40)]


def test_ur5e_smooth_hold_expired_ui_heartbeat_stops_with_exact_error() -> None:
    module = _teleop_module()
    teleop = object.__new__(module.KeyboardTeleop)
    teleop._ur5e_smooth_active = True
    _initialize_ur5e_smooth_state(teleop)
    teleop._ur5e_smooth_heartbeat_monotonic = time.monotonic() - 1.0
    stops: list[bool] = []

    def _stop(*, clear_pending_error: bool = True) -> tuple[bool, str]:
        stops.append(clear_pending_error)
        teleop._ur5e_smooth_active = False
        return True, "stopped"

    teleop._stop_ur5e_cartesian_jog = _stop

    assert teleop._ur5e_refresh_cartesian_jog_once() is False
    assert stops == [False]
    assert teleop._ur5e_smooth_pending_error.startswith(
        "UR5e Cartesian Smooth Hold UI heartbeat expired after "
    )


def test_xarm6_controller_state_wait_observes_ufactory_transition() -> None:
    module = _teleop_module()
    assert module.ListControllers is not None
    teleop = object.__new__(module.KeyboardTeleop)
    teleop.xarm6_controller_list_client = SimpleNamespace(
        wait_for_service=lambda timeout_sec: 0.0 < timeout_sec <= 0.5
    )
    states = iter(("inactive", "active"))
    observed_states: list[str] = []

    def _list_controllers(_client: Any, _request: Any, timeout_sec: float) -> Any:
        state = next(states)
        observed_states.append(state)
        return (
            SimpleNamespace(
                controller=[
                    SimpleNamespace(name="xarm6_traj_controller", state=state)
                ]
            ),
            None,
        )

    teleop._call_service = _list_controllers

    assert teleop._xarm6_wait_for_trajectory_controller_state("active") == (
        True,
        "xArm6 trajectory controller is active",
    )
    assert observed_states == ["inactive", "active"]
    source = (
        ROOT / "ros2/cais_lab_robotics/scripts/keyboard_teleop.py"
    ).read_text(encoding="utf-8")
    assert "SwitchController" not in source
    assert "_xarm6_switch_trajectory_controller" not in source


def test_xarm6_mode_services_use_exact_persistent_clients() -> None:
    module = _teleop_module()
    assert module.SetInt16 is not None
    teleop = object.__new__(module.KeyboardTeleop)
    waits: list[tuple[str, float]] = []
    mode_client = SimpleNamespace(
        wait_for_service=lambda timeout_sec: (
            waits.append(("set_mode", float(timeout_sec))) or True
        )
    )
    state_client = SimpleNamespace(
        wait_for_service=lambda timeout_sec: (
            waits.append(("set_state", float(timeout_sec))) or True
        )
    )
    teleop.xarm6_set_mode_client = mode_client
    teleop.xarm6_set_state_client = state_client
    calls: list[tuple[Any, int, float]] = []
    teleop._call_service = lambda client, request, timeout_sec: (
        calls.append((client, int(request.data), float(timeout_sec)))
        or (SimpleNamespace(ret=0, message=""), None)
    )

    assert teleop._xarm6_set_int16("set_mode", 1) == (True, "OK")
    assert teleop._xarm6_set_int16("set_state", 0) == (True, "OK")
    assert waits == [
        ("set_mode", module.XARM6_HARDWARE_CONTROL_SERVICE_WAIT_SEC),
        ("set_state", module.XARM6_HARDWARE_CONTROL_SERVICE_WAIT_SEC),
    ]
    assert calls == [
        (mode_client, 1, 3.0),
        (state_client, 0, 3.0),
    ]


def test_xarm6_mode_service_reports_exact_discovery_timeout() -> None:
    module = _teleop_module()
    assert module.SetInt16 is not None
    teleop = object.__new__(module.KeyboardTeleop)
    teleop.xarm6_set_mode_client = SimpleNamespace(
        wait_for_service=lambda timeout_sec: False
    )
    teleop.xarm6_set_state_client = object()

    assert teleop._xarm6_set_int16("set_mode", 1) == (
        False,
        "/xarm6/xarm/set_mode is unavailable after 5.0s",
    )


def test_xarm6_smooth_hold_enters_verified_mode_five() -> None:
    module = _teleop_module()
    assert module.MoveVelocity is not None
    teleop = object.__new__(module.KeyboardTeleop)
    teleop._xarm6_smooth_active = False
    _initialize_xarm6_smooth_state(teleop)
    teleop.xarm6_hardware_cartesian_velocity_service = "/xarm6/xarm/vc_set_cartesian_velocity"
    teleop.xarm6_hardware_cartesian_velocity_client = SimpleNamespace(
        wait_for_service=lambda timeout_sec: timeout_sec == pytest.approx(2.0)
    )
    teleop._xarm6_cartesian_readiness = lambda **_kwargs: (True, "ready", {})
    teleop._world_vector_in_robot_base = lambda velocity: tuple(velocity)
    calls: list[tuple[str, int | str]] = []
    teleop._xarm6_set_int16 = lambda suffix, value: (
        calls.append((suffix, value)) or (True, "OK")
    )
    teleop._xarm6_wait_for_mode = lambda mode: (
        calls.append(("wait_mode", mode)) or (True, "OK")
    )
    teleop._xarm6_wait_for_trajectory_controller_state = lambda state: (
        calls.append(("wait_controller", state)) or (True, "OK")
    )
    teleop._xarm6_set_tcp_maxacc = lambda: (
        calls.append(("set_tcp_maxacc", 0)) or (True, "OK")
    )
    teleop._call_service = lambda _client, _request, timeout_sec: (
        calls.append(("velocity", int(timeout_sec)))
        or (SimpleNamespace(ret=0), None)
    )
    teleop._start_xarm6_cartesian_jog_refresh = lambda: (True, "OK")

    ok, message = teleop._set_xarm6_cartesian_jog((0.01, 0.0, 0.0), 0.30)

    assert ok is True
    assert message == "xArm6 Cartesian Smooth Hold active"
    assert teleop._xarm6_smooth_active is True
    assert calls == [
        ("set_mode", 5),
        ("set_state", 0),
        ("wait_mode", 5),
        ("wait_controller", "inactive"),
        ("set_tcp_maxacc", 0),
        ("velocity", 2),
    ]


def test_xarm6_smooth_hold_update_uses_fresh_mode_five_feedback() -> None:
    module = _teleop_module()
    assert module.MoveVelocity is not None
    teleop = object.__new__(module.KeyboardTeleop)
    teleop._xarm6_smooth_active = True
    _initialize_xarm6_smooth_state(teleop)
    teleop.xarm6_hardware_cartesian_velocity_service = (
        "/xarm6/xarm/vc_set_cartesian_velocity"
    )
    teleop.xarm6_hardware_cartesian_velocity_client = SimpleNamespace(
        wait_for_service=lambda timeout_sec: timeout_sec == pytest.approx(2.0)
    )
    teleop._xarm6_robot_state_snapshot = lambda: (
        {"mode": 5, "state": 0},
        "",
    )
    teleop._xarm6_cartesian_readiness = lambda **_kwargs: pytest.fail(
        "active Smooth Hold updates must not repeat the full TF validation"
    )
    teleop._world_vector_in_robot_base = lambda velocity: tuple(velocity)
    requests: list[Any] = []
    teleop._call_service = lambda _client, request, timeout_sec: (
        requests.append(request) or (SimpleNamespace(ret=0), None)
    )

    assert teleop._set_xarm6_cartesian_jog((0.01, 0.0, 0.0), 0.50) == (
        True,
        "xArm6 Cartesian Smooth Hold active",
    )
    assert requests == []
    assert teleop._xarm6_smooth_speeds == pytest.approx(
        [10.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    )
    assert teleop._xarm6_smooth_watchdog_sec == pytest.approx(0.50)


def test_xarm6_smooth_hold_refresh_runs_next_to_ros2_driver() -> None:
    module = _teleop_module()
    teleop = object.__new__(module.KeyboardTeleop)
    teleop._xarm6_smooth_active = True
    _initialize_xarm6_smooth_state(teleop)
    teleop._xarm6_smooth_speeds = [21.125, 0.0, 0.0, 0.0, 0.0, 0.0]
    refreshes: list[tuple[list[float], float, float]] = []
    teleop._send_xarm6_cartesian_velocity = (
        lambda speeds, *, duration, timeout_sec: (
            refreshes.append((list(speeds), float(duration), float(timeout_sec)))
            or (True, "OK")
        )
    )

    assert teleop._xarm6_refresh_cartesian_jog_once() is True
    assert refreshes == [
        ([21.125, 0.0, 0.0, 0.0, 0.0, 0.0], 0.30, 0.40)
    ]


def test_xarm6_smooth_hold_expired_ui_heartbeat_stops_with_exact_error() -> None:
    module = _teleop_module()
    teleop = object.__new__(module.KeyboardTeleop)
    teleop._xarm6_smooth_active = True
    _initialize_xarm6_smooth_state(teleop)
    teleop._xarm6_smooth_heartbeat_monotonic = time.monotonic() - 1.0
    stops: list[bool] = []

    def _stop(*, clear_pending_error: bool = True) -> tuple[bool, str]:
        stops.append(clear_pending_error)
        teleop._xarm6_smooth_active = False
        return True, "stopped"

    teleop._stop_xarm6_cartesian_jog = _stop

    assert teleop._xarm6_refresh_cartesian_jog_once() is False
    assert stops == [False]
    assert teleop._xarm6_smooth_pending_error.startswith(
        "xArm6 Cartesian Smooth Hold UI heartbeat expired after "
    )


def test_xarm6_smooth_hold_stop_uses_verified_trajectory_restore() -> None:
    module = _teleop_module()
    assert module.MoveVelocity is not None
    teleop = object.__new__(module.KeyboardTeleop)
    teleop._xarm6_smooth_active = True
    _initialize_xarm6_smooth_state(teleop)
    teleop._xarm6_last_stop_motion_confirmed = False
    teleop.xarm6_hardware_cartesian_velocity_service = "/xarm6/xarm/vc_set_cartesian_velocity"
    teleop.xarm6_hardware_cartesian_velocity_client = object()
    calls: list[str] = []
    teleop._call_service = lambda _client, request, timeout_sec: (
        calls.append(f"zero_velocity:{request.duration}:{timeout_sec}")
        or (SimpleNamespace(ret=0), None)
    )
    teleop._xarm6_restore_trajectory_control = lambda: (
        calls.append("restore_mode_1") or (True, "restored")
    )

    ok, message = teleop._stop_xarm6_cartesian_jog()

    assert ok is True
    assert message == "xArm6 Cartesian Smooth Hold stopped"
    assert teleop._xarm6_smooth_active is False
    assert teleop._xarm6_last_stop_motion_confirmed is True
    assert calls == ["zero_velocity:0.3:2.0", "restore_mode_1"]


def test_xarm6_smooth_hold_confirmed_stop_separates_restore_readiness() -> None:
    module = _teleop_module()
    assert module.MoveVelocity is not None
    teleop = object.__new__(module.KeyboardTeleop)
    teleop._xarm6_smooth_active = True
    _initialize_xarm6_smooth_state(teleop)
    teleop._xarm6_last_stop_motion_confirmed = False
    teleop.xarm6_hardware_cartesian_velocity_service = (
        "/xarm6/xarm/vc_set_cartesian_velocity"
    )
    teleop.xarm6_hardware_cartesian_velocity_client = object()
    teleop._call_service = lambda *_args, **_kwargs: (
        SimpleNamespace(ret=0),
        None,
    )
    teleop._xarm6_restore_trajectory_control = lambda: (
        False,
        "xArm6 trajectory controller did not become active; state=inactive",
    )

    ok, message = teleop._stop_xarm6_cartesian_jog()

    assert ok is False
    assert "trajectory control restore failed" in message
    assert teleop._xarm6_last_stop_motion_confirmed is True
    assert teleop._xarm6_smooth_active is False


def test_xarm6_smooth_hold_unconfirmed_stop_remains_uncertain() -> None:
    module = _teleop_module()
    assert module.MoveVelocity is not None
    teleop = object.__new__(module.KeyboardTeleop)
    teleop._xarm6_smooth_active = True
    _initialize_xarm6_smooth_state(teleop)
    teleop._xarm6_last_stop_motion_confirmed = True
    teleop.xarm6_hardware_cartesian_velocity_service = (
        "/xarm6/xarm/vc_set_cartesian_velocity"
    )
    teleop.xarm6_hardware_cartesian_velocity_client = object()
    teleop._call_service = lambda *_args, **_kwargs: (
        SimpleNamespace(ret=-1),
        None,
    )
    teleop._xarm6_restore_trajectory_control = lambda: (True, "restored")

    ok, message = teleop._stop_xarm6_cartesian_jog()

    assert ok is False
    assert "ret=-1" in message
    assert teleop._xarm6_last_stop_motion_confirmed is False
    assert teleop._xarm6_smooth_active is False
    assert teleop._stop_xarm6_cartesian_jog()[0] is False
    assert teleop._xarm6_last_stop_motion_confirmed is False


def test_real_xarm6_cartesian_jog_calls_exact_driver_service() -> None:
    module = _teleop_module()
    assert module.MoveCartesian is not None
    teleop = object.__new__(module.KeyboardTeleop)
    teleop.xarm6_hardware_cartesian_service = module.XARM6_HARDWARE_CARTESIAN_SERVICE
    teleop.xarm6_hardware_cartesian_speed_mm_s = 50.0
    teleop.xarm6_hardware_cartesian_acceleration_mm_s2 = 100.0
    teleop.xarm6_hardware_cartesian_position_tolerance_m = 0.003
    teleop.xarm6_hardware_cartesian_orientation_tolerance_rad = 0.0523598776
    teleop.xarm6_hardware_workspace_bounds = {
        "x_min_m": -0.6,
        "x_max_m": 0.6,
        "y_min_m": -1.0,
        "y_max_m": 0.1,
        "z_min_m": 0.9,
        "z_max_m": 1.5,
    }
    identity_transform = SimpleNamespace(
        transform=SimpleNamespace(
            translation=SimpleNamespace(x=0.0, y=0.0, z=0.0),
            rotation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
        )
    )
    teleop.tf_buffer = SimpleNamespace(
        lookup_transform=lambda target, source, _time: (
            identity_transform
            if (target, source) == ("world", "link_base")
            else pytest.fail(f"unexpected TF lookup: {target} <- {source}")
        )
    )
    teleop.xarm6_hardware_cartesian_client = SimpleNamespace(
        wait_for_service=lambda timeout_sec: timeout_sec == pytest.approx(2.0)
    )
    requests: list[Any] = []
    teleop._call_service = lambda _client, request, timeout_sec: (
        requests.append(request) or (SimpleNamespace(ret=0, message="OK"), None)
    )
    handoffs: list[str] = []
    teleop._xarm6_prepare_firmware_cartesian_mode = lambda: (
        handoffs.append("prepare_mode_0") or (True, "ready")
    )
    teleop._xarm6_restore_trajectory_control = lambda: (
        handoffs.append("restore_mode_1") or (True, "restored")
    )
    target = module.Pose()
    target.position.x = 0.2
    target.position.y = -0.4
    target.position.z = 1.2
    target.orientation.w = 1.0
    teleop._get_world_ee_pose = lambda _robot: target

    ok, message = teleop._move_xarm6_hardware_cartesian(
        target,
        velocity_scale=0.35,
    )

    assert ok is True
    assert "succeeded" in message
    assert len(requests) == 1
    request = requests[0]
    assert list(request.pose) == pytest.approx([200.0, -400.0, 1200.0, 0.0, 0.0, 0.0])
    assert request.speed == pytest.approx(17.5)
    assert request.acc == pytest.approx(35.0)
    assert request.wait is True
    assert request.relative is False
    assert handoffs == ["prepare_mode_0", "restore_mode_1"]


@pytest.mark.parametrize(
    ("controller_state", "expected_ready"),
    [(0, True), (1, True), (2, True), (3, False), (4, False)],
)
def test_xarm6_cartesian_readiness_matches_driver_ready_states(
    controller_state: int,
    expected_ready: bool,
) -> None:
    module = _teleop_module()
    teleop = object.__new__(module.KeyboardTeleop)
    teleop.xarm6_hardware_cartesian_position_tolerance_m = 0.003
    teleop.xarm6_hardware_cartesian_orientation_tolerance_rad = 0.0523598776
    identity_transform = SimpleNamespace(
        transform=SimpleNamespace(
            translation=SimpleNamespace(x=0.0, y=0.0, z=0.0),
            rotation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
        )
    )
    teleop.tf_buffer = SimpleNamespace(
        lookup_transform=lambda _target, _source, _time: identity_transform
    )
    teleop._xarm6_robot_state_snapshot = lambda: (
        {
            "pose": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "offset": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "mode": 1,
            "state": controller_state,
        },
        "",
    )

    ready, message, diagnostics = teleop._xarm6_cartesian_readiness()

    assert ready is expected_ready
    assert diagnostics["controller_state"] == controller_state
    if expected_ready:
        assert message.startswith("xArm6 Cartesian frame validation ready")
    else:
        assert "expected a driver-ready state from 0 to 2" in message


def test_xarm6_trajectory_mode_preparation_restores_mode_one() -> None:
    module = _teleop_module()
    teleop = object.__new__(module.KeyboardTeleop)
    snapshots = iter(
        [
            ({"mode": 0, "state": 0}, ""),
            ({"mode": 0, "state": 0}, ""),
            ({"mode": 1, "state": 0}, ""),
        ]
    )
    service_calls: list[tuple[str, int]] = []
    teleop._xarm6_robot_state_snapshot = lambda: next(snapshots)
    teleop._xarm6_set_int16 = lambda suffix, value: (
        service_calls.append((suffix, value)) or (True, "OK")
    )
    controller_states: list[str] = []
    observed_controller_states = iter(("active", "inactive", "active"))

    def _controller_state(**_kwargs: Any) -> tuple[str, str]:
        state = next(observed_controller_states)
        controller_states.append(state)
        return state, ""

    teleop._xarm6_trajectory_controller_state = _controller_state

    ready, message, diagnostics = teleop._prepare_xarm6_trajectory_mode()

    assert ready is True
    assert message == "xArm6 trajectory controller Mode 1 is ready"
    assert diagnostics == {"controller_mode": 1, "controller_state": 0}
    assert service_calls == [("set_mode", 1), ("set_state", 0)]
    assert controller_states == ["active", "inactive", "active"]


def test_xarm6_trajectory_mode_preparation_waits_for_feedback_without_recommanding() -> None:
    module = _teleop_module()
    teleop = object.__new__(module.KeyboardTeleop)
    snapshots = iter(
        [
            (None, "xArm6 robot_states feedback has not been received"),
            ({"mode": 1, "state": 0}, "OK"),
        ]
    )
    service_calls: list[tuple[str, int]] = []
    teleop._xarm6_robot_state_snapshot = lambda: next(snapshots)
    teleop._xarm6_set_int16 = lambda suffix, value: (
        service_calls.append((suffix, value)) or (True, "OK")
    )
    teleop._xarm6_trajectory_controller_state = lambda **_kwargs: ("active", "")

    ready, message, diagnostics = teleop._prepare_xarm6_trajectory_mode()

    assert ready is True
    assert message == "xArm6 trajectory controller Mode 1 is ready"
    assert diagnostics == {"controller_mode": 1, "controller_state": 0}
    assert service_calls == []


def test_xarm6_trajectory_mode_preparation_reports_discovered_publishers() -> None:
    module = _teleop_module()
    teleop = object.__new__(module.KeyboardTeleop)
    teleop._xarm6_robot_state_subscription = SimpleNamespace(
        get_publisher_count=lambda: 1
    )
    teleop._xarm6_robot_state_snapshot = lambda: (
        None,
        "xArm6 robot_states feedback has not been received",
    )
    teleop._xarm6_trajectory_controller_state = lambda **_kwargs: ("missing", "")
    monotonic_values = iter((0.0, 0.0, 9.0))
    original_monotonic = module.time.monotonic
    original_sleep = module.time.sleep
    module.time.monotonic = lambda: next(monotonic_values)
    module.time.sleep = lambda _seconds: None
    try:
        ready, message, diagnostics = teleop._prepare_xarm6_trajectory_mode()
    finally:
        module.time.monotonic = original_monotonic
        module.time.sleep = original_sleep

    assert ready is False
    assert message.endswith("discovered_publishers=1")
    assert diagnostics == {}


def test_xarm6_trajectory_mode_preparation_only_clears_non_ready_state() -> None:
    module = _teleop_module()
    teleop = object.__new__(module.KeyboardTeleop)
    snapshots = iter(
        [
            ({"mode": 1, "state": 5}, "OK"),
            ({"mode": 1, "state": 0}, "OK"),
        ]
    )
    service_calls: list[tuple[str, int]] = []
    teleop._xarm6_robot_state_snapshot = lambda: next(snapshots)
    teleop._xarm6_set_int16 = lambda suffix, value: (
        service_calls.append((suffix, value)) or (True, "OK")
    )
    controller_states = iter(("inactive", "active"))
    teleop._xarm6_trajectory_controller_state = lambda **_kwargs: (
        next(controller_states),
        "",
    )

    ready, message, diagnostics = teleop._prepare_xarm6_trajectory_mode()

    assert ready is True
    assert message == "xArm6 trajectory controller Mode 1 is ready"
    assert diagnostics == {"controller_mode": 1, "controller_state": 0}
    assert service_calls == [("set_state", 0)]


def test_xarm6_trajectory_mode_preparation_tolerates_state_five_controller_race() -> None:
    module = _teleop_module()
    teleop = object.__new__(module.KeyboardTeleop)
    snapshots = iter(
        [
            ({"mode": 0, "state": 0}, "OK"),
            ({"mode": 1, "state": 5}, "OK"),
            ({"mode": 1, "state": 5}, "OK"),
            ({"mode": 1, "state": 0}, "OK"),
        ]
    )
    controller_states = iter(("active", "inactive", "inactive", "active"))
    service_calls: list[tuple[str, int]] = []
    teleop._xarm6_robot_state_snapshot = lambda: next(snapshots)
    teleop._xarm6_trajectory_controller_state = lambda **_kwargs: (
        next(controller_states),
        "",
    )
    teleop._xarm6_set_int16 = lambda suffix, value: (
        service_calls.append((suffix, value)) or (True, "OK")
    )

    ready, message, diagnostics = teleop._prepare_xarm6_trajectory_mode()

    assert ready is True
    assert message == "xArm6 trajectory controller Mode 1 is ready"
    assert diagnostics == {"controller_mode": 1, "controller_state": 0}
    assert service_calls == [("set_mode", 1), ("set_state", 0)]


def test_xarm6_trajectory_mode_preparation_fails_when_activation_stalls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _teleop_module()
    monkeypatch.setattr(
        module,
        "XARM6_TRAJECTORY_MODE_NO_PROGRESS_TIMEOUT_SEC",
        0.02,
    )
    monkeypatch.setattr(
        module,
        "XARM6_TRAJECTORY_MODE_POLL_INTERVAL_SEC",
        0.005,
    )
    teleop = object.__new__(module.KeyboardTeleop)
    teleop._xarm6_robot_state_snapshot = lambda: (
        {"mode": 1, "state": 0},
        "OK",
    )
    teleop._xarm6_trajectory_controller_state = lambda **_kwargs: (
        "inactive",
        "",
    )
    teleop._xarm6_set_int16 = lambda *_args: pytest.fail(
        "ready Mode 1 feedback must not resend mode services"
    )

    ready, message, diagnostics = teleop._prepare_xarm6_trajectory_mode()

    assert ready is False
    assert "activation stopped progressing; state=inactive" in message
    assert "trajectory_controller=inactive" in message
    assert diagnostics == {"controller_mode": 1, "controller_state": 0}


def test_bridge_mode_one_transport_encloses_dynamic_preparation_watchdog() -> None:
    from cais_spade_llm.ui.bridge import SystemBridge

    module = _teleop_module()
    bridge = object.__new__(SystemBridge)
    requests: list[tuple[dict[str, Any], float, int | None]] = []
    bridge._teleop_request_payload = lambda payload, timeout_sec, ros_domain_id: (
        requests.append((dict(payload), float(timeout_sec), ros_domain_id))
        or (True, "ready", {})
    )

    result = bridge._prepare_xarm6_trajectory_mode(ros_domain_id=42)

    assert result is None
    assert requests[0][0] == {
        "op": "prepare_xarm6_trajectory_mode",
        "robot": "xarm6",
    }
    assert requests[0][1] > module.XARM6_TRAJECTORY_MODE_HARD_SAFETY_TIMEOUT_SEC
    assert requests[0][2] == 42


@pytest.mark.parametrize(
    ("robot", "driver_process"),
    [
        ("xarm6", "hardware_xarm6_driver"),
        ("ur5e", "hardware_ur5e_rtde_trajectory_server"),
    ],
)
def test_direct_hardware_stack_enables_cartesian_jog(
    robot: str,
    driver_process: str,
) -> None:
    from cais_spade_llm.ui.bridge import SystemBridge

    bridge = object.__new__(SystemBridge)
    bridge.robot_env = "real"
    bridge._hardware_stack_selected = "dual robots"
    bridge._hardware_stack_lifecycle_state = "running"
    bridge._hardware_stack_lifecycle_generation = 4
    bridge._hardware_cartesian_readiness = {
        robot: {
            "cartesian_jog_ready": True,
            "cartesian_function_ready": True,
            "generation": 4,
            "message": "ready",
        }
    }
    bridge._default_ros_domain_id = lambda: 0
    bridge._digital_twin_domain_ids = lambda: {"hardware": 0, "gazebo": 1}
    bridge._digital_twin_teleop_target = lambda _robot: None
    bridge._normal_hardware_teleop_processes = lambda _robot: {
        "driver": driver_process,
        "gripper": driver_process,
    }
    bridge._teleop_process_running = lambda process_name: process_name == driver_process
    bridge._any_teleop_environment_running = lambda: True

    target = bridge.teleop_target(robot, "cartesian")

    assert target["environment"] == "real"
    assert target["source"] == "hardware"
    assert target["required_processes"] == [driver_process]
    assert target["ready"] is True
    assert target["warning"] == ""


def test_failed_hardware_stack_blocks_cartesian_jog_with_surviving_driver() -> None:
    from cais_spade_llm.ui.bridge import SystemBridge

    bridge = object.__new__(SystemBridge)
    bridge.robot_env = "real"
    bridge._hardware_stack_selected = "dual robots"
    bridge._hardware_stack_lifecycle_state = "failed"
    bridge._hardware_stack_lifecycle_generation = 4
    bridge._hardware_cartesian_readiness = {
        "xarm6": {
            "cartesian_jog_ready": False,
            "cartesian_function_ready": False,
            "generation": 4,
            "message": "failed",
        }
    }
    bridge._default_ros_domain_id = lambda: 0
    bridge._digital_twin_domain_ids = lambda: {"hardware": 0, "gazebo": 1}
    bridge._digital_twin_teleop_target = lambda _robot: None
    bridge._normal_hardware_teleop_processes = lambda _robot: {
        "driver": "hardware_xarm6_driver",
        "gripper": "hardware_xarm6_driver",
    }
    bridge._teleop_process_running = lambda _process_name: True
    bridge._any_teleop_environment_running = lambda: True

    target = bridge.teleop_target("xarm6", "cartesian")

    assert target["ready"] is False
    assert target["warning"] == (
        "dual robots Hardware Stack lifecycle is failed. Motion is available only "
        "after Hardware Stack reaches running."
    )


def test_xarm6_hardware_action_candidates_include_namespaced_driver_action() -> None:
    module = _teleop_module()
    teleop = object.__new__(module.KeyboardTeleop)
    teleop.get_action_names_and_types = lambda: []

    candidates = teleop._candidate_action_names(
        "xarm6_traj_controller/follow_joint_trajectory"
    )

    assert candidates[0] == "/xarm6/xarm6_traj_controller/follow_joint_trajectory"


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


def test_teleop_response_discards_late_request_before_current_response() -> None:
    from cais_spade_llm.ui.bridge import SystemBridge

    bridge = object.__new__(SystemBridge)
    read_fd, write_fd = os.pipe()
    stdout = os.fdopen(read_fd, "r", encoding="utf-8")
    bridge._teleop_server_proc = SimpleNamespace(
        stdout=stdout,
        stderr=None,
        poll=lambda: None,
    )
    os.write(
        write_fd,
        b'{"ok":true,"msg":"old","request_id":"teleop-1"}\n',
    )

    def _write_current_response() -> None:
        time.sleep(0.02)
        os.write(
            write_fd,
            b'{"ok":true,"msg":"current","request_id":"teleop-2"}\n',
        )

    writer = threading.Thread(target=_write_current_response)
    writer.start()
    try:
        ok, message, payload = bridge._read_teleop_response_locked(
            1.0,
            expected_request_id="teleop-2",
        )
    finally:
        writer.join(timeout=1.0)
        os.close(write_fd)
        stdout.close()

    assert ok is True
    assert message == "current"
    assert payload["request_id"] == "teleop-2"


def test_teleop_request_writes_and_waits_for_same_request_id() -> None:
    from cais_spade_llm.ui.bridge import SystemBridge

    bridge = object.__new__(SystemBridge)
    stdin = io.StringIO()
    bridge._teleop_server_lock = threading.Lock()
    bridge._teleop_server_proc = SimpleNamespace(
        stdin=stdin,
        poll=lambda: None,
    )
    bridge._teleop_request_sequence = 0
    bridge._ensure_teleop_server_locked = lambda _domain_id: None
    expected_ids: list[str] = []

    def _read_response(
        timeout_sec: float,
        *,
        expected_request_id: str | None = None,
    ) -> tuple[bool, str, dict[str, Any]]:
        assert timeout_sec == pytest.approx(5.0)
        expected_ids.append(str(expected_request_id or ""))
        return True, "OK", {"request_id": expected_request_id}

    bridge._read_teleop_response_locked = _read_response

    result = bridge._teleop_request_payload(
        {"op": "cartesian", "robot": "ur5e", "axis": "z", "step_mm": 1.0},
        timeout_sec=5.0,
        ros_domain_id=42,
    )

    request = json.loads(stdin.getvalue())
    assert result == (True, "OK", {"request_id": "teleop-1"})
    assert request["request_id"] == "teleop-1"
    assert expected_ids == ["teleop-1"]


def test_teleop_server_echoes_request_id_in_every_command_response() -> None:
    source = (
        ROOT / "ros2/cais_lab_robotics/scripts/keyboard_teleop.py"
    ).read_text(encoding="utf-8")

    assert "active_request_id = str(cmd.get('request_id') or '').strip() or None" in source
    assert "body['request_id'] = active_request_id" in source


def test_ur5e_teleop_waits_for_pick_approach_motion_handoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cais_spade_llm.ui import bridge as bridge_module
    from cais_spade_llm.ui.bridge import SystemBridge

    monkeypatch.setattr(bridge_module, "_UR5E_TELEOP_HANDOFF_TIMEOUT_SEC", 0.5)
    bridge = object.__new__(SystemBridge)
    bridge._ur5e_robot_function_execution_lock = threading.Lock()
    bridge._ur5e_robot_function_execution_lock.acquire()
    bridge._ur5e_robot_function_execution_active = "pick_approach"
    agent = SimpleNamespace(_robot_motion_lock=threading.Lock())
    bridge._physical_ur5e_robot_agent = lambda: agent
    bridge._teleop_preflight = lambda _robot, _op: {
        "warning": "",
        "ros_domain_id": 42,
    }
    requests: list[dict[str, Any]] = []
    bridge._teleop_request_payload = lambda payload, timeout_sec, ros_domain_id: (
        requests.append(dict(payload)) or (True, "OK", {})
    )

    def _release_pick_approach_handoff() -> None:
        time.sleep(0.02)
        bridge._ur5e_robot_function_execution_lock.release()

    releaser = threading.Thread(target=_release_pick_approach_handoff)
    releaser.start()
    try:
        result = bridge._teleop_request(
            {"op": "cartesian", "robot": "ur5e", "axis": "z", "step_mm": 1.0},
            timeout_sec=5.0,
        )
    finally:
        releaser.join(timeout=1.0)

    assert result == (True, "OK")
    assert requests == [
        {"op": "cartesian", "robot": "ur5e", "axis": "z", "step_mm": 1.0}
    ]
    assert bridge._ur5e_robot_function_execution_lock.locked() is False


def test_ur5e_cartesian_jog_uses_rtde_result_response_window() -> None:
    from cais_spade_llm.ui import bridge as bridge_module
    from cais_spade_llm.ui.bridge import SystemBridge

    bridge = object.__new__(SystemBridge)
    requests: list[tuple[dict[str, Any], float]] = []
    bridge._teleop_request = lambda payload, timeout_sec: (
        requests.append((dict(payload), float(timeout_sec))) or (True, "OK")
    )

    assert bridge.teleop_jog("ur5e", "z", 1.0, 0.35) == (True, "OK")
    assert requests == [
        (
            {
                "op": "cartesian",
                "robot": "ur5e",
                "axis": "z",
                "step_mm": 1.0,
                "velocity_scale": 0.35,
            },
            bridge_module._UR5E_RTDE_CLIENT_RESULT_TIMEOUT_SEC + 5.0,
        )
    ]


def test_xarm6_pre_motion_cartesian_rejection_does_not_latch_uncertainty() -> None:
    from cais_spade_llm.ui.bridge import SystemBridge

    bridge = object.__new__(SystemBridge)
    bridge._ur5e_robot_function_execution_lock = threading.Lock()
    bridge._ur5e_robot_function_execution_active = None
    bridge._xarm6_robot_function_state_uncertain = False
    bridge._xarm6_robot_function_state_uncertain_reason = ""
    bridge._physical_xarm6_robot_agent = lambda: None
    bridge._teleop_preflight = lambda _robot, _op: {
        "warning": "",
        "environment": "real",
        "ros_domain_id": 42,
    }
    bridge._teleop_request_payload = lambda **_kwargs: (
        False,
        "xArm6 Cartesian target z=1.610000 m is outside [0.900000, 1.600000] m",
        {"state_uncertain": False},
    )

    ok, message = bridge._teleop_request(
        {"op": "cartesian", "robot": "xarm6", "axis": "z", "step_mm": 10.0},
        timeout_sec=5.0,
    )

    assert ok is False
    assert "outside" in message
    assert bridge._xarm6_robot_function_state_uncertain is False
    assert bridge._xarm6_robot_function_state_uncertain_reason == ""


def test_xarm6_gripper_timeout_does_not_latch_arm_uncertainty() -> None:
    from cais_spade_llm.ui.bridge import SystemBridge

    bridge = object.__new__(SystemBridge)
    bridge._ur5e_robot_function_execution_lock = threading.Lock()
    bridge._ur5e_robot_function_execution_active = None
    bridge._xarm6_robot_function_state_uncertain = False
    bridge._xarm6_robot_function_state_uncertain_reason = ""
    bridge._physical_xarm6_robot_agent = lambda: None
    bridge._teleop_preflight = lambda _robot, _op: {
        "warning": "",
        "environment": "real",
        "ros_domain_id": 42,
    }
    bridge._teleop_request_payload = lambda **_kwargs: (
        False,
        "teleop response timeout; command outcome is still pending",
        {},
    )

    ok, message = bridge._teleop_request(
        {"op": "gripper", "robot": "xarm6", "action": "open"},
        timeout_sec=10.0,
    )

    assert ok is False
    assert "pending" in message
    assert bridge._xarm6_robot_function_state_uncertain is False
    assert bridge._xarm6_robot_function_state_uncertain_reason == ""


def test_xarm6_gripper_uses_hardware_service_budget() -> None:
    from cais_spade_llm.ui.bridge import SystemBridge

    bridge = object.__new__(SystemBridge)
    requests: list[tuple[dict[str, Any], float]] = []
    bridge._teleop_preflight = lambda _robot, _op: {
        "warning": "",
        "environment": "real",
    }
    bridge._teleop_request = lambda payload, timeout_sec: (
        requests.append((dict(payload), float(timeout_sec))) or (True, "OK")
    )

    assert bridge.teleop_gripper("xarm6", "open", 0.1, 0.5) == (True, "OK")
    assert requests == [
        (
            {
                "op": "gripper",
                "robot": "xarm6",
                "action": "open",
                "velocity_scale": 0.5,
                "step": 0.1,
            },
            10.0,
        )
    ]


def test_real_ur5e_jog_preflight_surfaces_rtde_readiness_failure(
    tmp_path: Path,
) -> None:
    from cais_spade_llm.ui.bridge import SystemBridge

    bridge = object.__new__(SystemBridge)
    bridge._move_insert_trials_dir = tmp_path / "move_insert_trials"
    bridge.teleop_target = lambda _robot, _op: {
        "ready": True,
        "warning": "",
        "environment": "real",
    }
    bridge.teleop_named_position_readiness = lambda _robot: (
        False,
        "UR5e RTDE trajectory status is stale",
    )

    target = bridge._teleop_preflight("ur5e", "cartesian")

    assert target["ready"] is False
    assert target["warning"] == "UR5e RTDE trajectory status is stale"


def test_smooth_hold_keeps_motion_locks_until_explicit_stop() -> None:
    from cais_spade_llm.ui import bridge as bridge_module
    from cais_spade_llm.ui.bridge import SystemBridge

    bridge = object.__new__(SystemBridge)
    bridge._teleop_smooth_session_lock = threading.Lock()
    bridge._teleop_smooth_session = None
    bridge._teleop_xarm6_cartesian_session_lock = threading.RLock()
    bridge._teleop_xarm6_cartesian_session = None
    bridge._teleop_xarm6_cartesian_idle_timer = None
    bridge._teleop_cartesian_modes = {"xarm6": "off", "ur5e": "off"}
    bridge._ur5e_robot_function_execution_lock = threading.Lock()
    bridge._ur5e_robot_function_execution_active = None
    bridge._ur5e_robot_function_state_uncertain = False
    bridge._xarm6_robot_function_state_uncertain = False
    agent_motion_lock = threading.Lock()
    bridge._physical_xarm6_robot_agent = lambda: SimpleNamespace(
        _robot_motion_lock=agent_motion_lock
    )
    bridge._physical_ur5e_robot_agent = lambda: None
    bridge.teleop_target = lambda _robot, _op: {
        "ready": True,
        "warning": "",
        "environment": "real",
        "ros_domain_id": 0,
    }
    bridge._teleop_preflight = lambda _robot, _op: {
        "ready": True,
        "warning": "",
        "environment": "real",
        "ros_domain_id": 0,
    }
    requests: list[tuple[dict[str, Any], float]] = []
    bridge._teleop_request_payload = lambda payload, timeout_sec, **_kwargs: (
        requests.append((dict(payload), float(timeout_sec))) or (True, "OK", {})
    )

    assert bridge.teleop_cartesian_mode("xarm6", "smooth")[0] is True
    assert bridge.teleop_cartesian_smooth("xarm6", "x", 10.0, "start") == (
        True,
        "OK",
    )
    assert bridge._ur5e_robot_function_execution_lock.locked()
    assert agent_motion_lock.locked()
    assert bridge.teleop_cartesian_smooth("xarm6", "x", 10.0, "update") == (
        True,
        "OK",
    )
    assert bridge.teleop_cartesian_smooth("xarm6", "x", 0.0, "stop") == (
        True,
        "OK",
    )
    assert bridge._ur5e_robot_function_execution_lock.locked()
    assert agent_motion_lock.locked()
    assert bridge.teleop_cartesian_mode("xarm6", "off")[0] is True
    assert not bridge._ur5e_robot_function_execution_lock.locked()
    assert not agent_motion_lock.locked()
    assert [
        request.get("command") or request.get("mode")
        for request, _timeout in requests
    ] == [
        "smooth",
        "start",
        "update",
        "stop",
        "off",
    ]
    assert [
        request["watchdog_sec"]
        for request, _timeout in requests
        if request.get("op") == "cartesian_smooth"
    ] == [
        0.50,
        0.50,
        0.50,
    ]
    assert [timeout for _request, timeout in requests] == [
        bridge_module._XARM6_CARTESIAN_SMOOTH_HANDOFF_TIMEOUT_SEC,
        bridge_module._XARM6_CARTESIAN_SMOOTH_HANDOFF_TIMEOUT_SEC,
        4.0,
        bridge_module._XARM6_CARTESIAN_SMOOTH_HANDOFF_TIMEOUT_SEC,
        bridge_module._XARM6_CARTESIAN_SMOOTH_HANDOFF_TIMEOUT_SEC,
    ]


def test_xarm6_cartesian_idle_restores_mode_one_once() -> None:
    from cais_spade_llm.ui import bridge as bridge_module
    from cais_spade_llm.ui.bridge import SystemBridge

    bridge = object.__new__(SystemBridge)
    bridge._teleop_xarm6_cartesian_session_lock = threading.RLock()
    bridge._teleop_xarm6_cartesian_session = None
    bridge._teleop_xarm6_cartesian_idle_timer = None
    bridge._teleop_cartesian_modes = {"xarm6": "off", "ur5e": "off"}
    bridge._ur5e_robot_function_execution_lock = threading.Lock()
    bridge._ur5e_robot_function_execution_active = None
    bridge._xarm6_cartesian_jog_state_uncertain = False
    bridge._xarm6_cartesian_jog_state_uncertain_reason = ""
    bridge._physical_xarm6_robot_agent = lambda: None
    target = {
        "ready": True,
        "warning": "",
        "environment": "real",
        "ros_domain_id": 42,
    }
    bridge.teleop_target = lambda _robot, _op: dict(target)
    bridge._teleop_preflight = lambda _robot, _op: dict(target)
    requests: list[dict[str, Any]] = []
    bridge._teleop_request_payload = lambda payload, **_kwargs: (
        requests.append(dict(payload)) or (True, "OK", {"state_uncertain": False})
    )

    assert bridge.teleop_cartesian_mode("xarm6", "step")[0] is True
    timer = bridge._teleop_xarm6_cartesian_idle_timer
    assert timer is not None
    timer.cancel()
    bridge._teleop_xarm6_cartesian_idle_timer = None
    assert bridge._teleop_xarm6_cartesian_session is not None
    bridge._teleop_xarm6_cartesian_session["last_activity_monotonic"] = (
        time.monotonic() - bridge_module._XARM6_CARTESIAN_SESSION_IDLE_SEC - 1.0
    )

    bridge._teleop_xarm6_cartesian_session["busy"] = True
    bridge._expire_xarm6_cartesian_session()
    assert [request["mode"] for request in requests] == ["step"]
    retry_timer = bridge._teleop_xarm6_cartesian_idle_timer
    assert retry_timer is not None
    retry_timer.cancel()
    bridge._teleop_xarm6_cartesian_idle_timer = None
    bridge._teleop_xarm6_cartesian_session["busy"] = False
    bridge._expire_xarm6_cartesian_session()
    bridge._expire_xarm6_cartesian_session()

    assert [request["mode"] for request in requests] == ["step", "off"]
    assert bridge._teleop_xarm6_cartesian_session is None
    assert bridge._teleop_cartesian_modes["xarm6"] == "off"
    assert not bridge._ur5e_robot_function_execution_lock.locked()


def test_bridge_sends_exact_physical_speeds_and_rejects_invalid_values() -> None:
    from cais_spade_llm.ui.bridge import SystemBridge

    bridge = object.__new__(SystemBridge)
    requests: list[dict[str, Any]] = []
    bridge._teleop_request = lambda payload, timeout_sec: (
        requests.append({**payload, "timeout_sec": timeout_sec}) or (True, "OK")
    )
    bridge.teleop_motion_settings = lambda _robot: {
        "cartesian_speed_min_mm_s": 0.0,
        "cartesian_speed_max_mm_s": 50.0,
        "joint_speed_min_deg_s": 0.0,
        "joint_speed_max_deg_s": 64.0,
    }

    assert bridge.teleop_jog("ur5e", "x", 2.0, speed_mm_s=37.5) == (
        True,
        "OK",
    )
    assert bridge.teleop_joint("ur5e", 3, -2.0, speed_deg_s=24.5) == (
        True,
        "OK",
    )
    assert requests[0]["speed_mm_s"] == pytest.approx(37.5)
    assert requests[1]["speed_deg_s"] == pytest.approx(24.5)
    assert bridge.teleop_jog("ur5e", "x", 2.0, speed_mm_s=math.nan)[0] is False
    assert bridge.teleop_joint("ur5e", 3, -2.0, speed_deg_s=100.0)[0] is False
    assert bridge.teleop_jog("ur5e", "x", 2.0, speed_mm_s=0.0)[0] is False
    assert bridge.teleop_joint("ur5e", 3, -2.0, speed_deg_s=0.0)[0] is False
    assert len(requests) == 2


def test_bridge_reports_distinct_xarm6_cartesian_default_and_maximum() -> None:
    from cais_spade_llm.ui.bridge import SystemBridge

    bridge = object.__new__(SystemBridge)
    bridge.teleop_target = lambda _robot, _op: {"environment": "real"}

    settings = bridge.teleop_motion_settings("xarm6")

    assert settings["cartesian_speed_min_mm_s"] == pytest.approx(0.0)
    assert settings["joint_speed_min_deg_s"] == pytest.approx(0.0)
    assert settings["cartesian_speed_default_mm_s"] == pytest.approx(50.0)
    assert settings["cartesian_speed_max_mm_s"] == pytest.approx(100.0)
    assert settings["cartesian_acceleration_mm_s2"] == pytest.approx(42.25)


def test_bridge_reports_distinct_ur5e_cartesian_default_and_maximum() -> None:
    from cais_spade_llm.ui.bridge import SystemBridge

    bridge = object.__new__(SystemBridge)
    bridge.teleop_target = lambda _robot, _op: {"environment": "real"}

    settings = bridge.teleop_motion_settings("ur5e")

    assert settings["cartesian_speed_min_mm_s"] == pytest.approx(0.0)
    assert settings["joint_speed_min_deg_s"] == pytest.approx(0.0)
    assert settings["cartesian_speed_default_mm_s"] == pytest.approx(80.0)
    assert settings["cartesian_speed_max_mm_s"] == pytest.approx(100.0)


def test_ur5e_rtde_cartesian_default_and_maximum_are_distinct() -> None:
    module = _rtde_server_module()

    assert module.UR5E_RTDE_CARTESIAN_SPEED_M_S == pytest.approx(0.08)
    assert module.UR5E_RTDE_CARTESIAN_MAX_SPEED_M_S == pytest.approx(0.10)
    status = module._status_base()
    assert module.UR5E_RTDE_GOAL_TOLERANCE_RAD == pytest.approx(0.025)
    assert status["joint_goal_tolerance_rad"] == pytest.approx(0.025)
    assert status["cartesian_speed_default_m_s"] == pytest.approx(0.08)
    assert status["cartesian_speed_limit_m_s"] == pytest.approx(0.10)


def test_xarm6_smooth_hold_confirmed_stop_readiness_failure_allows_next_axis() -> None:
    from cais_spade_llm.ui.bridge import SystemBridge

    bridge = object.__new__(SystemBridge)
    bridge._teleop_smooth_session_lock = threading.Lock()
    bridge._teleop_smooth_session = None
    bridge._teleop_xarm6_cartesian_session_lock = threading.RLock()
    bridge._teleop_xarm6_cartesian_idle_timer = None
    bridge._teleop_cartesian_modes = {"xarm6": "smooth", "ur5e": "off"}
    bridge._ur5e_robot_function_execution_lock = threading.Lock()
    bridge._ur5e_robot_function_execution_active = None
    bridge._xarm6_robot_function_state_uncertain = False
    bridge._xarm6_cartesian_jog_state_uncertain = False
    bridge._xarm6_cartesian_jog_state_uncertain_reason = ""
    agent_motion_lock = threading.Lock()
    bridge._ur5e_robot_function_execution_lock.acquire()
    agent_motion_lock.acquire()
    bridge._teleop_xarm6_cartesian_session = {
        "mode": "smooth",
        "ros_domain_id": 42,
        "execution_lock_acquired": True,
        "agent_motion_lock": agent_motion_lock,
        "agent_lock_acquired": True,
        "last_activity_monotonic": time.monotonic(),
        "busy": False,
    }
    bridge._physical_xarm6_robot_agent = lambda: SimpleNamespace(
        _robot_motion_lock=agent_motion_lock
    )
    bridge._teleop_preflight = lambda _robot, _op: {
        "ready": True,
        "warning": "",
        "environment": "real",
        "ros_domain_id": 42,
    }
    responses = iter(
        (
            (True, "xArm6 Cartesian Smooth Hold active", {}),
            (
                False,
                "xArm6 trajectory controller did not become active; state=inactive",
                {"state_uncertain": False},
            ),
            (True, "xArm6 Cartesian Smooth Hold active", {}),
            (True, "xArm6 Cartesian Smooth Hold stopped", {"state_uncertain": False}),
        )
    )
    requests: list[dict[str, Any]] = []
    bridge._teleop_request_payload = lambda payload, **_kwargs: (
        requests.append(dict(payload)) or next(responses)
    )

    assert bridge.teleop_cartesian_smooth("xarm6", "x", 10.0, "start")[0] is True
    stop_ok, stop_message = bridge.teleop_cartesian_smooth(
        "xarm6", "x", 0.0, "stop"
    )

    assert stop_ok is False
    assert "did not become active" in stop_message
    assert bridge._xarm6_cartesian_jog_state_uncertain is False
    assert bridge._xarm6_cartesian_jog_state_uncertain_reason == ""
    assert bridge.teleop_cartesian_smooth("xarm6", "y", 10.0, "start")[0] is True
    assert bridge.teleop_cartesian_smooth("xarm6", "y", 0.0, "stop")[0] is True
    assert [request["command"] for request in requests] == [
        "start",
        "stop",
        "start",
        "stop",
    ]
    assert bridge._ur5e_robot_function_execution_lock.locked()
    assert agent_motion_lock.locked()
    with bridge._teleop_xarm6_cartesian_session_lock:
        bridge._release_xarm6_cartesian_session_locked()
    assert not bridge._ur5e_robot_function_execution_lock.locked()
    assert not agent_motion_lock.locked()


def test_xarm6_smooth_hold_unconfirmed_stop_latches_uncertainty() -> None:
    from cais_spade_llm.ui.bridge import SystemBridge

    bridge = object.__new__(SystemBridge)
    bridge._teleop_smooth_session_lock = threading.Lock()
    bridge._teleop_smooth_session = None
    bridge._teleop_xarm6_cartesian_session_lock = threading.RLock()
    bridge._teleop_xarm6_cartesian_idle_timer = None
    bridge._teleop_cartesian_modes = {"xarm6": "smooth", "ur5e": "off"}
    bridge._ur5e_robot_function_execution_lock = threading.Lock()
    bridge._ur5e_robot_function_execution_active = None
    bridge._xarm6_robot_function_state_uncertain = False
    bridge._xarm6_cartesian_jog_state_uncertain = False
    bridge._xarm6_cartesian_jog_state_uncertain_reason = ""
    bridge._ur5e_robot_function_execution_lock.acquire()
    bridge._teleop_xarm6_cartesian_session = {
        "mode": "smooth",
        "ros_domain_id": 42,
        "execution_lock_acquired": True,
        "agent_motion_lock": None,
        "agent_lock_acquired": False,
        "last_activity_monotonic": time.monotonic(),
        "busy": False,
    }
    bridge._physical_xarm6_robot_agent = lambda: None
    bridge._teleop_preflight = lambda _robot, _op: {
        "ready": True,
        "warning": "",
        "environment": "real",
        "ros_domain_id": 42,
    }
    responses = iter(
        (
            (True, "xArm6 Cartesian Smooth Hold active", {}),
            (False, "zero velocity returned ret=-1", {"state_uncertain": True}),
        )
    )
    bridge._teleop_request_payload = lambda *_args, **_kwargs: next(responses)

    assert bridge.teleop_cartesian_smooth("xarm6", "z", 10.0, "start")[0] is True
    ok, message = bridge.teleop_cartesian_smooth("xarm6", "z", 0.0, "stop")

    assert ok is False
    assert message == "zero velocity returned ret=-1"
    assert bridge._xarm6_cartesian_jog_state_uncertain is True
    assert bridge._xarm6_cartesian_jog_state_uncertain_reason == (
        "Cartesian Smooth Hold stop failed: zero velocity returned ret=-1"
    )
    with bridge._teleop_xarm6_cartesian_session_lock:
        bridge._release_xarm6_cartesian_session_locked()


def test_smooth_hold_command_failure_with_confirmed_stop_does_not_latch_uncertain() -> None:
    from cais_spade_llm.ui.bridge import SystemBridge

    bridge = object.__new__(SystemBridge)
    bridge._teleop_smooth_session_lock = threading.Lock()
    bridge._teleop_smooth_session = None
    bridge._ur5e_robot_function_execution_lock = threading.Lock()
    bridge._ur5e_robot_function_execution_active = None
    bridge._ur5e_robot_function_state_uncertain = False
    bridge._ur5e_robot_function_state_uncertain_reason = ""
    bridge._ur5e_cartesian_jog_state_uncertain = False
    bridge._ur5e_cartesian_jog_state_uncertain_reason = ""
    agent_motion_lock = threading.Lock()
    bridge._physical_ur5e_robot_agent = lambda: SimpleNamespace(
        _robot_motion_lock=agent_motion_lock
    )
    bridge._teleop_preflight = lambda _robot, _op: {
        "ready": True,
        "warning": "",
        "environment": "real",
        "ros_domain_id": 42,
    }
    responses = iter(
        (
            (True, "UR5e Cartesian Smooth Hold active", {}),
            (False, "teleop response timeout", {}),
            (True, "UR5e Cartesian jog watchdog stopped motion", {}),
        )
    )
    bridge._teleop_request_payload = lambda *_args, **_kwargs: next(responses)

    assert bridge.teleop_cartesian_smooth("ur5e", "z", 10.0, "start")[0] is True
    ok, message = bridge.teleop_cartesian_smooth("ur5e", "z", 10.0, "update")

    assert ok is False
    assert "motion stop confirmed" in message
    assert bridge._ur5e_robot_function_state_uncertain is False
    assert bridge._ur5e_robot_function_state_uncertain_reason == ""
    assert bridge._teleop_smooth_session is None
    assert not bridge._ur5e_robot_function_execution_lock.locked()
    assert not agent_motion_lock.locked()


def test_smooth_hold_stop_failure_latches_exact_uncertain_reason() -> None:
    from cais_spade_llm.ui.bridge import SystemBridge

    bridge = object.__new__(SystemBridge)
    bridge._teleop_smooth_session_lock = threading.Lock()
    bridge._teleop_smooth_session = None
    bridge._ur5e_robot_function_execution_lock = threading.Lock()
    bridge._ur5e_robot_function_execution_active = None
    bridge._ur5e_robot_function_state_uncertain = False
    bridge._ur5e_robot_function_state_uncertain_reason = ""
    bridge._physical_ur5e_robot_agent = lambda: None
    bridge._teleop_preflight = lambda _robot, _op: {
        "ready": True,
        "warning": "",
        "environment": "real",
        "ros_domain_id": 42,
    }
    responses = iter(
        (
            (True, "UR5e Cartesian Smooth Hold active", {}),
            (False, "RTDE jogStop returned False", {}),
        )
    )
    bridge._teleop_request_payload = lambda *_args, **_kwargs: next(responses)

    assert bridge.teleop_cartesian_smooth("ur5e", "x", 10.0, "start")[0] is True
    ok, message = bridge.teleop_cartesian_smooth("ur5e", "x", 0.0, "stop")

    assert ok is False
    assert message == "RTDE jogStop returned False"
    assert bridge._ur5e_robot_function_state_uncertain is False
    assert bridge._ur5e_cartesian_jog_state_uncertain is True
    assert bridge._ur5e_cartesian_jog_state_uncertain_reason == (
        "Cartesian Smooth Hold stop failed: RTDE jogStop returned False"
    )


def test_repair_stop_releases_active_smooth_hold_without_preflight() -> None:
    from cais_spade_llm.ui.bridge import SystemBridge

    bridge = object.__new__(SystemBridge)
    bridge._teleop_smooth_session_lock = threading.Lock()
    bridge._ur5e_robot_function_execution_lock = threading.Lock()
    bridge._ur5e_robot_function_execution_lock.acquire()
    bridge._ur5e_robot_function_execution_active = (
        "Interactive Teleop ur5e Cartesian Smooth Hold"
    )
    agent_motion_lock = threading.Lock()
    agent_motion_lock.acquire()
    bridge._teleop_smooth_session = {
        "robot": "ur5e",
        "axis": "z",
        "ros_domain_id": 42,
        "execution_lock_acquired": True,
        "agent_motion_lock": agent_motion_lock,
        "agent_lock_acquired": True,
    }
    requests: list[dict[str, object]] = []
    bridge._teleop_request_payload = lambda payload, **_kwargs: (
        requests.append(dict(payload)) or (True, "jog stopped", {})
    )

    message = bridge._stop_cartesian_smooth_for_repair("dual robots")

    assert message == "Cartesian Smooth Hold stop confirmed: jog stopped"
    assert requests == [
        {
            "op": "cartesian_smooth",
            "robot": "ur5e",
            "command": "stop",
            "axis": "z",
            "speed_mm_s": 0.0,
            "watchdog_sec": 0.30,
        }
    ]
    assert bridge._teleop_smooth_session is None
    assert not bridge._ur5e_robot_function_execution_lock.locked()
    assert not agent_motion_lock.locked()


def test_repair_waits_for_xarm6_smooth_hold_mode_restore() -> None:
    from cais_spade_llm.ui import bridge as bridge_module
    from cais_spade_llm.ui.bridge import SystemBridge

    bridge = object.__new__(SystemBridge)
    bridge._teleop_smooth_session_lock = threading.Lock()
    bridge._ur5e_robot_function_execution_lock = threading.Lock()
    bridge._ur5e_robot_function_execution_lock.acquire()
    bridge._ur5e_robot_function_execution_active = (
        "Interactive Teleop xarm6 Cartesian Smooth Hold"
    )
    agent_motion_lock = threading.Lock()
    agent_motion_lock.acquire()
    bridge._teleop_smooth_session = {
        "robot": "xarm6",
        "axis": "x",
        "ros_domain_id": 42,
        "execution_lock_acquired": True,
        "agent_motion_lock": agent_motion_lock,
        "agent_lock_acquired": True,
    }
    timeouts: list[float] = []
    bridge._teleop_request_payload = lambda _payload, timeout_sec, **_kwargs: (
        timeouts.append(float(timeout_sec)) or (True, "jog stopped", {})
    )

    message = bridge._stop_cartesian_smooth_for_repair("dual robots")

    assert message == "Cartesian Smooth Hold stop confirmed: jog stopped"
    assert timeouts == [bridge_module._XARM6_CARTESIAN_SMOOTH_HANDOFF_TIMEOUT_SEC]
    assert bridge._teleop_smooth_session is None
    assert not bridge._ur5e_robot_function_execution_lock.locked()
    assert not agent_motion_lock.locked()


def test_repair_does_not_latch_confirmed_xarm6_stop_as_uncertain() -> None:
    from cais_spade_llm.ui.bridge import SystemBridge

    bridge = object.__new__(SystemBridge)
    bridge._teleop_smooth_session_lock = threading.Lock()
    bridge._ur5e_robot_function_execution_lock = threading.Lock()
    bridge._ur5e_robot_function_execution_lock.acquire()
    bridge._ur5e_robot_function_execution_active = (
        "Interactive Teleop xarm6 Cartesian Smooth Hold"
    )
    bridge._xarm6_cartesian_jog_state_uncertain = False
    bridge._xarm6_cartesian_jog_state_uncertain_reason = ""
    agent_motion_lock = threading.Lock()
    agent_motion_lock.acquire()
    bridge._teleop_smooth_session = {
        "robot": "xarm6",
        "axis": "x",
        "ros_domain_id": 42,
        "execution_lock_acquired": True,
        "agent_motion_lock": agent_motion_lock,
        "agent_lock_acquired": True,
    }
    bridge._teleop_request_payload = lambda *_args, **_kwargs: (
        False,
        "xArm6 trajectory controller did not become active; state=inactive",
        {"state_uncertain": False},
    )

    message = bridge._stop_cartesian_smooth_for_repair("dual robots")

    assert message.startswith(
        "Cartesian Smooth Hold stop confirmed; trajectory readiness failed:"
    )
    assert bridge._xarm6_cartesian_jog_state_uncertain is False
    assert bridge._xarm6_cartesian_jog_state_uncertain_reason == ""
    assert bridge._teleop_smooth_session is None
    assert not bridge._ur5e_robot_function_execution_lock.locked()
    assert not agent_motion_lock.locked()


def test_ur5e_jog_stop_false_requires_rtde_reset() -> None:
    module = _rtde_server_module()
    server = object.__new__(module.UR5eRTDETrajectoryServer)
    server._active_lock = threading.Lock()
    server._jog_session_token = object()
    server._active_goal = server._jog_session_token
    server._active_goal_status = {"state": "executing"}
    server._active_motion_kind = "cartesian_jog"
    server._jog_stop_in_progress = False
    server._jog_watchdog_deadline = module.time.monotonic() + 0.3
    server.control = SimpleNamespace(jogStop=lambda: False)
    reset_reasons: list[str] = []
    server._mark_rtde_reset_required = lambda reason, **_kwargs: reset_reasons.append(
        reason
    )

    ok, message = server._stop_cartesian_jog("release")

    assert ok is False
    assert "jogStop returned False" in message
    assert reset_reasons == [message]
    assert server._active_goal is None
    assert server._active_motion_kind == ""


def test_ur5e_jog_stop_does_not_hide_latched_rtde_reset() -> None:
    module = _rtde_server_module()
    server = object.__new__(module.UR5eRTDETrajectoryServer)
    server._active_lock = threading.Lock()
    server._jog_session_token = object()
    server._active_goal = None
    server._jog_stop_in_progress = False
    server._jog_watchdog_deadline = 0.0
    server._rtde_reset_required = True
    server._rtde_reset_reason = "UR5e RTDE jogStop returned False"

    ok, message = server._stop_cartesian_jog("release")

    assert ok is False
    assert message == "UR5e RTDE jogStop returned False"


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


def test_real_ur5e_action_uses_configured_result_timeout() -> None:
    module = _teleop_module()
    teleop = object.__new__(module.KeyboardTeleop)
    result_future = _PendingFuture()
    goal_handle = SimpleNamespace(
        accepted=True,
        get_result_async=lambda: result_future,
    )
    teleop.ur5e_hardware_trajectory_action = (
        "/cais_ur5e_rtde_trajectory_controller/follow_joint_trajectory"
    )
    teleop.ur5e_hardware_result_timeout_sec = 52.0
    teleop.ur5e_hardware_trajectory_client = SimpleNamespace(
        wait_for_server=lambda timeout_sec: True,
        send_goal_async=lambda _goal: _ImmediateFuture(goal_handle),
    )
    teleop.joint_positions = {"ur5e": [0.0] * 6}
    waits: list[float] = []

    def _wait(future: Any, timeout: float) -> bool:
        waits.append(float(timeout))
        return future.done()

    teleop._wait_future = _wait

    ok, message = teleop._move_ur5e_arm_action(
        list(module.ROBOTS["ur5e"]["joint_name_candidates"][1]),
        [0.1, -1.0, -2.0, -1.2, 1.5, 0.0],
        duration_sec=1.2,
    )

    assert ok is False
    assert "result timeout" in message
    assert waits == [3.0, 52.0]


def test_named_position_request_outlasts_rtde_client_timeout() -> None:
    from cais_spade_llm.ui import bridge as bridge_module
    from cais_spade_llm.ui.bridge import SystemBridge

    bridge = object.__new__(SystemBridge)
    bridge.list_named_positions = lambda _robot: {"home": [0.0] * 6}
    bridge.teleop_named_position_readiness = lambda _robot: (True, "ready")
    requests: list[tuple[dict[str, Any], float]] = []
    bridge._teleop_request = lambda payload, timeout_sec: (
        requests.append((dict(payload), float(timeout_sec))) or (True, "OK")
    )

    assert bridge.teleop_go_to_position("ur5e", "home") == (True, "OK")
    assert requests == [
        (
            {
                "op": "move_joints",
                "robot": "ur5e",
                "positions": [0.0] * 6,
                "named_position": "home",
            },
            bridge_module._UR5E_RTDE_CLIENT_RESULT_TIMEOUT_SEC + 5.0,
        )
    ]


def test_completed_physical_ur5e_named_home_clears_robot_function_recovery() -> None:
    from cais_spade_llm.ui.bridge import SystemBridge

    bridge = object.__new__(SystemBridge)
    bridge._ur5e_robot_function_execution_lock = threading.Lock()
    bridge._ur5e_robot_function_execution_active = None
    bridge._ur5e_robot_function_state_uncertain = True
    bridge._ur5e_robot_function_state_uncertain_reason = "RTDE feedback stopped"
    agent_motion_lock = threading.Lock()
    agent = SimpleNamespace(
        _robot_motion_lock=agent_motion_lock,
        _held_part=None,
        _current_state="at_pick",
        _task_ctx={"part_name": "MG"},
        _recovery_pose_ref="",
    )
    bridge._physical_ur5e_robot_agent = lambda: agent
    bridge._teleop_preflight = lambda _robot, _op: {
        "warning": "",
        "environment": "real",
        "ros_domain_id": 42,
    }
    bridge._teleop_request_payload = lambda *_args, **_kwargs: (
        True,
        "UR5e trajectory completed",
        {"state_uncertain": False},
    )

    ok, message = bridge._teleop_request(
        {
            "op": "move_joints",
            "robot": "ur5e",
            "positions": [0.0] * 6,
            "named_position": "home",
        },
        timeout_sec=60.0,
    )

    assert ok is True
    assert "UR5e home verified" in message
    assert bridge._ur5e_robot_function_state_uncertain is False
    assert bridge._ur5e_robot_function_state_uncertain_reason == ""
    assert agent._current_state == "idle"
    assert agent._task_ctx == {}
    assert agent._recovery_pose_ref == "home"
    assert not bridge._ur5e_robot_function_execution_lock.locked()
    assert not agent_motion_lock.locked()


def test_non_home_named_position_does_not_clear_ur5e_recovery() -> None:
    from cais_spade_llm.ui.bridge import SystemBridge

    bridge = object.__new__(SystemBridge)
    bridge._ur5e_robot_function_execution_lock = threading.Lock()
    bridge._ur5e_robot_function_execution_active = None
    bridge._ur5e_robot_function_state_uncertain = True
    bridge._ur5e_robot_function_state_uncertain_reason = "RTDE feedback stopped"
    agent = SimpleNamespace(
        _robot_motion_lock=threading.Lock(),
        _held_part=None,
        _current_state="idle",
        _task_ctx={},
    )
    bridge._physical_ur5e_robot_agent = lambda: agent
    bridge._teleop_preflight = lambda _robot, _op: {
        "warning": "",
        "environment": "real",
        "ros_domain_id": 42,
    }
    bridge._teleop_request_payload = lambda *_args, **_kwargs: (
        True,
        "UR5e trajectory completed",
        {"state_uncertain": False},
    )

    ok, _message = bridge._teleop_request(
        {
            "op": "move_joints",
            "robot": "ur5e",
            "positions": [0.0] * 6,
            "named_position": "prusa-mk4-2",
        },
        timeout_sec=60.0,
    )

    assert ok is True
    assert bridge._ur5e_robot_function_state_uncertain is True
    assert bridge._ur5e_robot_function_state_uncertain_reason == "RTDE feedback stopped"


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
    assert "--include-hidden-services --no-daemon --spin-time 2.0" in bridge


def test_ros_action_discovery_refreshes_negative_snapshot_without_losing_rate_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cais_spade_llm.ui import bridge as bridge_module
    from cais_spade_llm.ui.bridge import SystemBridge

    bridge = object.__new__(SystemBridge)
    bridge._ros2_procs = {}
    bridge._ros_action_service_snapshot = None
    calls = 0
    now = [100.0]
    monkeypatch.setattr(bridge_module.time, "monotonic", lambda: now[0])

    required_services = "\n".join(
        (
            "/cais_ur5e_rtde_trajectory_controller/follow_joint_trajectory/_action/send_goal",
            "/cais_ur5e_rtde_trajectory_controller/follow_joint_trajectory/_action/get_result",
            "/cais_ur5e_rtde_trajectory_controller/follow_joint_trajectory/_action/cancel_goal",
        )
    )

    def _command(*_args, **_kwargs) -> tuple[bool, str]:
        nonlocal calls
        calls += 1
        return (False, "") if calls == 1 else (True, required_services)

    bridge._ros2_command_output = _command

    ok, _services = bridge._ros_action_service_snapshot_for_domain(
        ros_domain_id=42,
        timeout_sec=8.0,
    )
    assert ok is False
    now[0] = 101.0
    bridge._ros_action_service_snapshot_for_domain(
        ros_domain_id=42,
        timeout_sec=8.0,
    )
    assert calls == 1

    now[0] = 103.0
    ok, services = bridge._ros_action_service_snapshot_for_domain(
        ros_domain_id=42,
        timeout_sec=8.0,
    )
    assert calls == 2
    assert ok is True
    assert set(required_services.splitlines()) <= services

    now[0] = 110.0
    bridge._ros_action_service_snapshot_for_domain(
        ros_domain_id=42,
        timeout_sec=8.0,
    )
    assert calls == 2


def test_rtde_server_keeps_read_only_joint_monitoring_in_local_control() -> None:
    server = (ROOT / "ros2/cais_lab_robotics/scripts/ur5e_rtde_trajectory_server.py").read_text(
        encoding="utf-8"
    )
    connect_method = server.split("    def _connect_rtde(self)", maxsplit=1)[1].split(
        "    def _mark_rtde_reset_required", maxsplit=1
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
    assert 'body["ros_domain_id"] = self.ros_domain_id' in server
    assert 'body["process_id"] = os.getpid()' in server
    assert 'body["rtde_reset_required"]' in server
    assert "def _reconnect_receive" not in server
    assert "self._next_status_heartbeat_monotonic = now + 1.0" in server
    assert 'state="stopped"' in server
    assert "UR5e RTDE trajectory server exited:" in server


def test_rtde_result_timeout_stops_and_clears_goal_without_destroying_server() -> None:
    server = (ROOT / "ros2/cais_lab_robotics/scripts/ur5e_rtde_trajectory_server.py").read_text(
        encoding="utf-8"
    )
    timeout_block = server.split('reason = "UR5e RTDE trajectory result timeout"', maxsplit=1)[
        1
    ].split("        except Exception as exc:", maxsplit=1)[0]

    assert timeout_block.index("self._stop_motion()") < timeout_block.index("goal_handle.abort()")
    assert "self._finish_active_goal_status(goal_handle, status, latch_status=True)" in timeout_block
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
    assert (
        SystemBridge._ur5e_rtde_result_timeout_requires_repair(
            {**status, "updated_at": None}, now=131.0
        )
        is False
    )


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
        "ros_domain_id": 42,
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
        "ros_domain_id": 42,
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


def test_named_position_readiness_rejects_another_domain_status() -> None:
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
        "ros_domain_id": 43,
        "rtde_control_connected": True,
        "updated_at": time.time(),
    }

    ready, message = bridge.teleop_named_position_readiness("ur5e")

    assert ready is False
    assert "ROS_DOMAIN_ID=43" in message
    assert "requested ROS_DOMAIN_ID=42" in message


def test_ur5e_cartesian_preflight_reuses_fresh_motion_readiness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cais_spade_llm.ui.bridge import SystemBridge

    bridge = object.__new__(SystemBridge)
    bridge._ur5e_named_position_readiness_cache = None
    bridge.teleop_target = lambda _robot, _operation: {
        "warning": "",
        "environment": "real",
        "ros_domain_id": 42,
        "source": "hardware",
        "hardware_stack_generation": 7,
        "required_processes": ["hardware_ur5e_rtde_trajectory_server"],
    }
    bridge._ur5e_rtde_trajectory_status = lambda: {
        "state": "ready",
        "message": "UR5e RTDE trajectory server ready",
        "ros_domain_id": 42,
        "rtde_control_connected": True,
        "updated_at": 100.0,
    }
    action_checks: list[bool] = []
    bridge._ros_action_readiness_error = lambda *_args, **_kwargs: (
        action_checks.append(True) or None
    )
    state_checks: list[bool] = []
    bridge.teleop_state = lambda _robot: (
        state_checks.append(True)
        or (
            True,
            "fresh UR5e state",
            {"joint_state_age_sec": 0.05},
        )
    )
    wall_time = [100.5]
    monotonic_time = [10.0]
    monkeypatch.setattr("cais_spade_llm.ui.bridge.time.time", lambda: wall_time[0])
    monkeypatch.setattr(
        "cais_spade_llm.ui.bridge.time.monotonic",
        lambda: monotonic_time[0],
    )

    first = bridge.teleop_named_position_readiness("ur5e")
    monotonic_time[0] = 10.2
    second = bridge.teleop_named_position_readiness("ur5e")

    assert first == second
    assert first[0] is True
    assert action_checks == [True]
    assert state_checks == [True]

    monotonic_time[0] = 12.0
    wall_time[0] = 100.7
    assert bridge.teleop_named_position_readiness("ur5e")[0] is True
    assert action_checks == [True, True]
    assert state_checks == [True, True]


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
    bridge._digital_twin_dual_drag_markers_status_path = lambda _target: Path("/tmp/not-used.json")
    bridge._read_json_file = lambda _path: {}
    bridge._digital_twin_direction = lambda _target: "hardware -> gazebo"
    bridge._digital_twin_blocked_reason = lambda _target, _cfg: ""
    bridge._digital_twin_sim_mode = lambda _target: "monitor"
    bridge._digital_twin_allowed_sim_modes = lambda _cfg: ("monitor",)
    bridge._digital_twin_hardware_domain_id = lambda _cfg, _robot, domains: domains["hardware"]
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


def test_dual_twin_fresh_mirroring_heartbeats_keep_process_status_running(
    tmp_path: Path,
) -> None:
    from cais_spade_llm.ui.bridge import SystemBridge

    bridge = object.__new__(SystemBridge)
    cfg = {
        "sync_processes": {"xarm6": "sync_xarm6", "ur5e": "sync_ur5e"},
    }
    now = time.time()
    for robot in ("xarm6", "ur5e"):
        (tmp_path / f"{robot}.json").write_text(
            json.dumps({"state": "mirroring", "updated_at": now - 0.5}),
            encoding="utf-8",
        )
    bridge.ros2_proc_status = lambda _name: "stopped"
    bridge._digital_twin_sync_status_path = (
        lambda _target, robot="": tmp_path / f"{robot}.json"
    )

    snapshot = bridge._digital_twin_sync_status_snapshot("dual robots", cfg, now)

    assert snapshot["process_status"] == "running"
    assert all(
        robot_status["state"] == "mirroring"
        for robot_status in snapshot["robots"].values()
    )


def test_dual_twin_stale_stopped_mirror_schedules_monitor_restart() -> None:
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
    bridge.ros2_proc_status = lambda name: "running" if name == "dual_gazebo" else "stopped"
    bridge._digital_twin_sync_status_snapshot = lambda _target, _cfg, _now: {
        "process": "sync_xarm6, sync_ur5e",
        "process_status": "stopped",
        "status_data": {"state": "starting", "message": "mirror processes stopped"},
        "status_age_ms": 7000.0,
        "robots": {},
    }
    bridge._digital_twin_dual_drag_markers_status_path = lambda _target: Path(
        "/tmp/not-used.json"
    )
    bridge._read_json_file = lambda _path: {}
    bridge._digital_twin_direction = lambda _target: "hardware -> gazebo"
    bridge._digital_twin_blocked_reason = lambda _target, _cfg: ""
    bridge._digital_twin_sim_mode = lambda _target: "monitor"
    bridge._digital_twin_allowed_sim_modes = lambda _cfg: ("monitor",)
    bridge._digital_twin_hardware_domain_id = lambda _cfg, _robot, domains: domains[
        "hardware"
    ]
    bridge._digital_twin_status_path = lambda _target: Path("/tmp/not-used.json")
    restart_calls: list[tuple[str, str]] = []
    bridge._schedule_digital_twin_monitor_sync_restart = (
        lambda target, _cfg, *, gazebo_process, domains, reason: (
            restart_calls.append((target, reason)) or False
        )
    )

    row = bridge.digital_twin_statuses()["dual robots"]

    assert restart_calls == [("dual robots", "sync process is stopped")]
    assert row["repair_needed"] is True


def test_dual_hardware_rviz_allows_external_goal_state_refresh() -> None:
    config = (
        ROOT / "ros2/cais_lab_robotics/rviz/dual_robots_hardware_moveit.rviz"
    ).read_text(encoding="utf-8")

    assert "MoveIt_Allow_External_Program: true" in config


def test_repair_stop_is_scoped_and_contains_no_motion_operation() -> None:
    from cais_spade_llm.ui.bridge import SystemBridge

    bridge = object.__new__(SystemBridge)
    events: list[str] = []
    bridge._digital_twin_process_names = lambda _cfg: ["driver", "moveit", "gazebo"]
    bridge.ros2_stop = lambda name, reason="": events.append(f"stop:{name}:{reason}")
    bridge._stop_teleop_server = lambda: events.append("stop:teleop")
    bridge._force_kill_digital_twin_helpers = lambda: events.append("stop:helpers")
    bridge._kill_stale_gazebo_helpers = lambda: events.append("stop:gazebo_helpers")
    bridge._force_kill_gazebo_core = lambda reason="": events.append(f"stop:gazebo_core:{reason}")

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
    launch = (ROOT / "ros2/cais_lab_robotics/launch/ur5e_rg2_hardware_moveit.launch.py").read_text(
        encoding="utf-8"
    )

    assert '"launch_move_group"' in launch
    assert "if not launch_move_group and not launch_rviz:" in launch
    assert "return launch_actions" in launch

    bridge = (ROOT / "cais_spade_llm/ui/bridge.py").read_text(encoding="utf-8")
    assert '"ur5e_calibration_rtde_monitor"' in bridge
    assert '"ur5e_calibration_state_publisher"' in bridge
    assert 'reason="ur5e_control_stack_start"' in bridge


def test_control_page_bounds_ros_readiness_refresh_work() -> None:
    control = (ROOT / "cais_spade_llm/ui/pages/control.py").read_text(encoding="utf-8")

    assert "await asyncio.to_thread(\n                    _launch_snapshot" in control
    assert "named_pos_readiness_state = {" in control
    assert '"ready": False,' in control
    assert "def _update_named_position_go_enabled()" in control
    assert 'named_pos_readiness_state["ready"] = bool(ready)' in control
    assert 'and not named_pos_busy["moving"]' in control
    assert 'if named_pos_readiness_state["busy"]:' in control
    assert "active_target_refresh: dict[str, object] = {" in control
    assert 'if active_target_refresh["busy"]:' in control
    assert "await asyncio.to_thread(bridge.digital_twin_statuses)" in control
    assert "signature = _signature(rows)" in control
    assert "signature = _signature()" not in control
    assert 'initial_robot = "ur5e"' in control
    assert 'f"Trajectory interface ({robot}): {message}"' in control
    assert "ui.timer(3.0, _refresh_named_position_readiness)" in control
    assert '"Repair Twin" if repair_needed else "Start Twin"' in control
    assert "repair=repair_needed" in control


def test_cartesian_smooth_active_status_does_not_run_full_teleop_preflight() -> None:
    from cais_spade_llm.ui.bridge import SystemBridge

    bridge = object.__new__(SystemBridge)
    bridge._teleop_smooth_session_lock = threading.Lock()
    bridge._teleop_smooth_session = None

    assert bridge.teleop_cartesian_smooth_active("ur5e") is False
    bridge._teleop_smooth_session = {"robot": "ur5e"}
    assert bridge.teleop_cartesian_smooth_active("ur5e") is True
    assert bridge.teleop_cartesian_smooth_active("xarm6") is False


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
