"""Focused regression tests for physical pick_approach synchronization."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from cais_spade_llm.resources.sensor.physical import realsense_roboflow_node
from cais_spade_llm.resources.sensor.physical.realsense_pose_estimator import (
    RigidTransform,
)
from cais_spade_llm.resources.sensor.physical.realsense_roboflow_node import (
    ColorIntrinsics,
    RealSenseRoboflowNode,
)
from cais_spade_llm.ui import bridge as bridge_module
from cais_spade_llm.ui.bridge import SystemBridge

ROOT = Path(__file__).resolve().parents[1]


def _transform(x: float = 0.0) -> RigidTransform:
    return RigidTransform((x, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))


def test_perception_waits_for_two_stable_world_tool0_samples(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    perception = object.__new__(RealSenseRoboflowNode)
    perception.camera_role = "ur5e"
    perception.world_frame = "world"
    perception.tool_frame = "tool0"
    samples = iter((_transform(0.0), _transform(0.002), _transform(0.0025), _transform(0.0028)))
    perception._lookup_transform = lambda *_args, **_kwargs: next(samples)
    monkeypatch.setattr(realsense_roboflow_node.time, "sleep", lambda _seconds: None)

    result = perception._wait_for_stationary_tool_pose()

    assert result.translation == pytest.approx((0.0028, 0.0, 0.0))


def test_perception_stationary_timeout_does_not_start_inference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    perception = object.__new__(RealSenseRoboflowNode)
    perception._inference_lock = threading.Lock()
    perception._last_rows = []
    perception.camera_role = "ur5e"
    perception.world_frame = "world"
    perception.tool_frame = "tool0"
    perception._reload_table_plane_calibration = lambda: None
    detector_calls: list[np.ndarray] = []
    perception._detector = SimpleNamespace(
        detect=lambda image: detector_calls.append(image) or [],
    )
    now = [0.0]
    position = [0.0]

    def _sleep(seconds: float) -> None:
        now[0] += seconds

    def _lookup(*_args: Any, **_kwargs: Any) -> RigidTransform:
        position[0] += 0.002
        return _transform(position[0])

    perception._lookup_transform = _lookup
    monkeypatch.setattr(realsense_roboflow_node.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(realsense_roboflow_node.time, "sleep", _sleep)

    with pytest.raises(RuntimeError, match="did not become stationary.*inference was not started"):
        perception._run_detection()

    assert detector_calls == []


def test_detection_requests_fresh_frame_after_stationary_wait() -> None:
    perception = object.__new__(RealSenseRoboflowNode)
    perception._inference_lock = threading.Lock()
    perception._last_rows = []
    perception._last_inference_latency_ms = None
    perception._roboflow_model_validated = False
    perception._table_surface_z_m = None
    perception.camera_role = "ur5e"
    perception.world_frame = "world"
    perception.tool_frame = "tool0"
    perception.camera_optical_frame = "camera_color_optical_frame"
    perception._reload_table_plane_calibration = lambda: None
    order: list[str] = []
    perception._wait_for_stationary_tool_pose = lambda: order.append("stationary") or _transform()
    stamp = SimpleNamespace(sec=10, nanosec=0)
    color = np.zeros((2, 2, 3), dtype=np.uint8)
    depth = np.ones((2, 2), dtype=np.float32)

    def _frame_copy(**kwargs: Any) -> tuple[Any, np.ndarray, np.ndarray, ColorIntrinsics]:
        assert float(kwargs["captured_after_sec"]) > 0.0
        order.append("frame")
        return stamp, color, depth, ColorIntrinsics(fx=1.0, fy=1.0, cx=0.5, cy=0.5)

    perception._frame_copy = _frame_copy
    perception._detector = SimpleNamespace(
        detect=lambda _image: order.append("inference") or [],
        settings=SimpleNamespace(model_id="model"),
    )
    perception._lookup_transform = lambda *_args, **_kwargs: _transform()
    perception._write_detection_preview = lambda *_args, **_kwargs: None
    perception._write_detection_status = lambda *_args, **_kwargs: None
    perception._write_snapshot = lambda _rows: None

    assert perception._run_detection() == []
    assert order == ["stationary", "frame", "inference"]


def test_detect_all_waits_for_background_inference_then_captures_fresh_frame() -> None:
    perception = object.__new__(RealSenseRoboflowNode)
    perception._inference_lock = threading.Lock()
    perception._inference_lock.acquire()
    perception.on_demand_inference_wait_sec = 1.0
    perception._last_rows = []
    perception._last_error = ""
    perception._last_inference_latency_ms = None
    perception._roboflow_model_validated = False
    perception._table_surface_z_m = None
    perception.camera_role = "ur5e"
    perception.world_frame = "world"
    perception.tool_frame = "tool0"
    perception.camera_optical_frame = "camera_color_optical_frame"
    perception._reload_table_plane_calibration = lambda: None
    perception._wait_for_stationary_tool_pose = lambda: _transform()
    order: list[str] = []
    waiting = threading.Event()

    def _write_progress(stage: str, *, message: str = "") -> None:
        order.append(message or stage)
        if message.startswith("Waiting for the active Roboflow inference"):
            waiting.set()

    perception._write_detection_progress = _write_progress
    stamp = SimpleNamespace(sec=10, nanosec=0)
    color = np.zeros((2, 2, 3), dtype=np.uint8)
    depth = np.ones((2, 2), dtype=np.float32)

    def _frame_copy(**kwargs: Any) -> tuple[Any, np.ndarray, np.ndarray, ColorIntrinsics]:
        assert float(kwargs["captured_after_sec"]) > 0.0
        order.append("fresh_frame")
        return stamp, color, depth, ColorIntrinsics(fx=1.0, fy=1.0, cx=0.5, cy=0.5)

    perception._frame_copy = _frame_copy
    detector_calls: list[np.ndarray] = []
    perception._detector = SimpleNamespace(
        detect=lambda image: detector_calls.append(image) or [],
        settings=SimpleNamespace(model_id="model"),
    )
    perception._lookup_transform = lambda *_args, **_kwargs: _transform()
    perception._write_detection_preview = lambda *_args, **_kwargs: None
    perception._write_detection_status = lambda *_args, **_kwargs: None
    snapshots: list[list[dict[str, Any]]] = []
    perception._write_snapshot = lambda rows: snapshots.append(rows)
    responses: list[Any] = []

    def _request_detection() -> None:
        responses.append(
            perception._service_result(SimpleNamespace(success=False, message=""))
        )

    request_thread = threading.Thread(target=_request_detection)
    request_thread.start()
    try:
        assert waiting.wait(timeout=1.0)
    finally:
        if perception._inference_lock.locked():
            perception._inference_lock.release()
    request_thread.join(timeout=2.0)

    assert not request_thread.is_alive()
    assert responses[0].success is True
    assert responses[0].message == "[]"
    assert len(detector_calls) == 1
    assert detector_calls[0] is color
    assert snapshots == [[]]
    assert order[:4] == [
        "Waiting for the active Roboflow inference before fresh /detect_all.",
        "settling",
        "fresh_frame",
        "detection",
    ]


def test_background_inference_lock_race_does_not_clear_detection_snapshot() -> None:
    class _LostRaceLock:
        @staticmethod
        def locked() -> bool:
            return False

        @staticmethod
        def acquire(*_args: Any, **_kwargs: Any) -> bool:
            return False

    perception = object.__new__(RealSenseRoboflowNode)
    perception._latest_frame = object()
    perception._inference_lock = _LostRaceLock()
    perception._last_rows = [{"part_name": "MG"}]
    perception._last_error = ""
    perception._write_snapshot = lambda _rows: pytest.fail(
        "a harmless background lock race must not clear the executable snapshot"
    )

    perception._background_detection()

    assert perception._last_rows == [{"part_name": "MG"}]
    assert perception._last_error == ""


def _load_rtde_server_module() -> Any:
    path = ROOT / "ros2/cais_lab_robotics/scripts/ur5e_rtde_trajectory_server.py"
    spec = importlib.util.spec_from_file_location("pick_approach_rtde_server_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _unsupported_async_operation_status() -> dict[str, Any]:
    return {
        "rtde_async_operation_supported": False,
        "rtde_async_operation_running": None,
        "rtde_async_operation_progress": None,
        "rtde_async_operation_id": None,
        "rtde_async_operation_change_count": None,
        "rtde_async_operation_value": None,
    }


def test_rtde_movej_path_removes_duplicate_current_hold_point() -> None:
    module = _load_rtde_server_module()

    def _point(positions: list[float]) -> Any:
        return SimpleNamespace(positions=positions)

    trajectory = SimpleNamespace(
        joint_names=list(module.ARM_JOINTS),
        points=[_point([0.0] * 6), _point([0.0] * 6), _point([0.5] * 6)],
    )

    path = module.rtde_movej_path(trajectory)

    assert len(path) == 2
    assert path[0][:6] == [0.0] * 6
    assert path[0][-1] == module.UR5E_RTDE_INTERMEDIATE_BLEND_RAD
    assert path[1][:6] == [0.5] * 6
    assert path[1][-1] == 0.0


def test_rtde_cached_timestamp_does_not_refresh_or_publish_joint_states(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_rtde_server_module()
    server = object.__new__(module.UR5eRTDETrajectoryServer)
    server.receive = SimpleNamespace(
        getActualQ=lambda: [0.25] * 6,
        getTimestamp=lambda: 100.0,
    )
    server.receive_factory = None
    server.control = object()
    server._receive_lock = threading.Lock()
    server._next_receive_connect_monotonic = 0.0
    server._receive_error = ""
    server._last_receive_timestamp = None
    server._receive_watch_started_monotonic = 0.0
    server._last_receive_reconnect_monotonic = 0.0
    server._idle_receive_reconnect_count = 0
    server.current_positions = None
    server.current_positions_monotonic = None
    server._joint_status_announced = True
    server._active_lock = threading.Lock()
    server._active_goal = None
    statuses: list[dict[str, Any]] = []
    server._write_status = lambda status: statuses.append(dict(status))
    published: list[Any] = []
    server._joint_state_pub = SimpleNamespace(
        publish=lambda message: published.append(message)
    )
    server.get_clock = lambda: SimpleNamespace(
        now=lambda: SimpleNamespace(to_msg=lambda: SimpleNamespace())
    )
    monkeypatch.setattr(
        module,
        "JointState",
        lambda: SimpleNamespace(header=SimpleNamespace(), name=[], position=[]),
    )
    now = [1.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: now[0])

    assert server._read_actual_q() == [0.25] * 6
    assert server.current_positions_monotonic == 1.0

    now[0] = 4.0
    reconnects: list[float] = []

    def _reconnect_receive() -> str:
        reconnects.append(now[0])
        server.receive = SimpleNamespace(
            getActualQ=lambda: [0.3] * 6,
            getTimestamp=lambda: 101.0,
        )
        return ""

    server._reconnect_receive = _reconnect_receive
    server._publish_joint_state()

    assert server.current_positions_monotonic == 1.0
    assert server._joint_states_fresh() is False
    assert published == []
    assert reconnects == [4.0]
    assert statuses[-1]["state"] == "recovering"

    server._next_status_heartbeat_monotonic = 10.0
    now[0] = 4.02
    server._publish_joint_state()

    assert server.current_positions_monotonic == 4.02
    assert len(published) == 1


def test_rtde_reuploads_stopped_control_program_before_goal() -> None:
    module = _load_rtde_server_module()
    server = object.__new__(module.UR5eRTDETrajectoryServer)
    program_running = [False]
    reuploads: list[bool] = []

    def _reupload_script() -> bool:
        reuploads.append(True)
        program_running[0] = True
        return True

    server.control = SimpleNamespace(
        isProgramRunning=lambda: program_running[0],
        reuploadScript=_reupload_script,
    )

    assert server._ensure_control_program_for_goal() is None
    assert reuploads == [True]


def test_rtde_success_requires_continuous_stationary_hold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_rtde_server_module()
    server = object.__new__(module.UR5eRTDETrajectoryServer)
    server.control = object()
    server.receive = object()
    server._active_lock = threading.Lock()
    server._active_goal_status = None
    server._latched_terminal_status = None
    published: list[object] = []
    server._rviz_goal_state_pub = SimpleNamespace(publish=lambda message: published.append(message))
    server.get_logger = lambda: SimpleNamespace(warning=lambda _message: None)
    server._connect_control_for_goal = lambda: None
    server._current_position_map = lambda: dict.fromkeys(module.ARM_JOINTS, 0.0)
    server._joint_states_fresh = lambda: True
    server._execute_movej_path = lambda _path: ("True", "asynchronous_keyword")
    server._stop_motion = lambda: None
    statuses: list[dict[str, Any]] = []
    server._write_status = lambda status: statuses.append(dict(status))
    server._write_active_goal_status = lambda _goal, status: setattr(
        server, "_active_goal_status", dict(status)
    )
    server._finish_active_goal_status = (
        lambda _goal, status, **_kwargs: statuses.append(dict(status))
    )
    server._clear_active_goal = lambda _goal: None
    actual_calls = [0]
    server._read_actual_q = lambda: actual_calls.__setitem__(0, actual_calls[0] + 1) or [0.0] * 6
    velocities = [0.0] * 8 + [0.02] + [0.0] * 40
    server._read_actual_qd = lambda: [velocities.pop(0) if velocities else 0.0] * 6

    point = SimpleNamespace(time_from_start=SimpleNamespace(sec=0, nanosec=0))
    trajectory = SimpleNamespace(points=[point])
    monkeypatch.setattr(
        module,
        "prepare_rtde_trajectory",
        lambda _trajectory, _positions: (True, trajectory, module._status_base()),
    )
    monkeypatch.setattr(
        module,
        "rtde_movej_path",
        lambda _trajectory: [[0.0] * 6 + [0.1, 0.1, 0.0]],
    )
    monkeypatch.setattr(module.rclpy, "ok", lambda: True)
    now = [0.0]
    server._read_feedback_timestamp = lambda: now[0]
    server._async_operation_status = _unsupported_async_operation_status
    monkeypatch.setattr(module.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(module.time, "time", lambda: now[0])
    monkeypatch.setattr(module.time, "sleep", lambda seconds: now.__setitem__(0, now[0] + seconds))

    class _Goal:
        def __init__(self) -> None:
            self.request = SimpleNamespace(trajectory=trajectory)
            self.is_cancel_requested = False
            self.succeeded = False

        def succeed(self) -> None:
            self.succeeded = True

        def abort(self) -> None:
            pytest.fail("stationary trajectory should not abort")

        def canceled(self) -> None:
            pytest.fail("stationary trajectory should not cancel")

    goal = _Goal()
    server._active_goal = None

    result = server._execute(goal)

    assert goal.succeeded is True
    assert result.error_code == 0
    assert actual_calls[0] >= 22
    assert len(published) == 1
    assert statuses[-1]["max_actual_joint_velocity_rad_s"] == 0.0
    assert statuses[-1]["max_observed_joint_velocity_rad_s"] == 0.02
    assert statuses[-1]["stationary_hold_sec"] >= 0.25


def test_rtde_accepted_command_that_does_not_move_fails_before_result_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_rtde_server_module()
    server = object.__new__(module.UR5eRTDETrajectoryServer)
    server.control = object()
    server.receive = object()
    server._active_lock = threading.Lock()
    server._active_goal_status = None
    server._latched_terminal_status = None
    published: list[object] = []
    server._rviz_goal_state_pub = SimpleNamespace(publish=lambda message: published.append(message))
    server.get_logger = lambda: SimpleNamespace(warning=lambda _message: None)
    server._connect_control_for_goal = lambda: None
    server._current_position_map = lambda: dict.fromkeys(module.ARM_JOINTS, 0.0)
    server._joint_states_fresh = lambda: True
    server._execute_movej_path = lambda _path: ("True", "asynchronous_positional")
    stopped: list[bool] = []
    server._stop_motion = lambda: stopped.append(True)
    statuses: list[dict[str, Any]] = []
    latched: list[bool] = []
    server._write_status = lambda status: statuses.append(dict(status))
    server._write_active_goal_status = lambda _goal, status: setattr(
        server, "_active_goal_status", dict(status)
    )

    def _finish(_goal: Any, status: dict[str, Any], *, latch_status: bool = False) -> None:
        statuses.append(dict(status))
        latched.append(latch_status)

    server._finish_active_goal_status = _finish
    server._clear_active_goal = lambda _goal: None
    server._read_actual_q = lambda: [0.0] * 6
    server._read_actual_qd = lambda: [0.0] * 6
    server._async_operation_status = _unsupported_async_operation_status

    point = SimpleNamespace(time_from_start=SimpleNamespace(sec=1, nanosec=0))
    trajectory = SimpleNamespace(points=[point])
    monkeypatch.setattr(
        module,
        "prepare_rtde_trajectory",
        lambda _trajectory, _positions: (True, trajectory, module._status_base()),
    )
    monkeypatch.setattr(
        module,
        "rtde_movej_path",
        lambda _trajectory: [[1.0] * 6 + [0.1, 0.1, 0.0]],
    )
    monkeypatch.setattr(module.rclpy, "ok", lambda: True)
    now = [0.0]
    server._read_feedback_timestamp = lambda: now[0]
    monkeypatch.setattr(module.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(module.time, "time", lambda: now[0])
    monkeypatch.setattr(module.time, "sleep", lambda seconds: now.__setitem__(0, now[0] + seconds))

    class _Goal:
        def __init__(self) -> None:
            self.request = SimpleNamespace(trajectory=trajectory)
            self.is_cancel_requested = False
            self.aborted = False

        def succeed(self) -> None:
            pytest.fail("a command that never moved must not succeed")

        def abort(self) -> None:
            self.aborted = True

        def canceled(self) -> None:
            pytest.fail("trajectory should not be canceled")

    goal = _Goal()
    server._active_goal = None

    result = server._execute(goal)

    assert goal.aborted is True
    assert result.error_code == -1
    assert result.error_string == "UR5e RTDE trajectory did not start after moveJ was accepted"
    assert now[0] == pytest.approx(module.UR5E_RTDE_MOTION_START_TIMEOUT_SEC)
    assert stopped == [True]
    assert latched[-1] is True
    assert statuses[-1]["motion_started"] is False
    assert published == []


def test_rtde_recovers_receive_feedback_without_redispatching_motion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_rtde_server_module()
    server = object.__new__(module.UR5eRTDETrajectoryServer)
    server.control = object()
    server.receive = object()
    server._active_lock = threading.Lock()
    server._active_goal_status = None
    server._latched_terminal_status = None
    published: list[object] = []
    server._rviz_goal_state_pub = SimpleNamespace(
        publish=lambda message: published.append(message)
    )
    server.get_logger = lambda: SimpleNamespace(warning=lambda _message: None)
    server._connect_control_for_goal = lambda: None
    server._current_position_map = lambda: dict.fromkeys(module.ARM_JOINTS, 0.0)
    server._joint_states_fresh = lambda: True
    dispatches: list[list[list[float]]] = []
    server._execute_movej_path = (
        lambda path: dispatches.append(path) or ("True", "asynchronous_keyword")
    )
    server._stop_motion = lambda: pytest.fail(
        "recoverable read-only feedback loss must not stop the accepted move"
    )
    statuses: list[dict[str, Any]] = []
    server._write_status = lambda status: statuses.append(dict(status))
    server._write_active_goal_status = lambda _goal, status: setattr(
        server, "_active_goal_status", dict(status)
    )
    server._finish_active_goal_status = (
        lambda _goal, status, **_kwargs: statuses.append(dict(status))
    )
    server._clear_active_goal = lambda _goal: None

    point = SimpleNamespace(time_from_start=SimpleNamespace(sec=2, nanosec=0))
    trajectory = SimpleNamespace(points=[point])
    monkeypatch.setattr(
        module,
        "prepare_rtde_trajectory",
        lambda _trajectory, _positions: (True, trajectory, module._status_base()),
    )
    path = [[1.0] * 6 + [0.1, 0.1, 0.0]]
    monkeypatch.setattr(module, "rtde_movej_path", lambda _trajectory: path)
    monkeypatch.setattr(module.rclpy, "ok", lambda: True)
    now = [0.0]
    recovered = [False]
    reconnects: list[float] = []

    def _reconnect_receive() -> str:
        reconnects.append(now[0])
        recovered[0] = True
        return ""

    server._reconnect_receive = _reconnect_receive
    server._read_feedback_timestamp = lambda: 100.0 + now[0] if recovered[0] else 100.0
    server._read_actual_q = lambda: [min(now[0], 1.0) if recovered[0] else 0.0] * 6
    server._read_actual_qd = lambda: [0.2 if recovered[0] and now[0] < 1.0 else 0.0] * 6
    monkeypatch.setattr(module.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(module.time, "time", lambda: now[0])
    monkeypatch.setattr(
        module.time,
        "sleep",
        lambda seconds: now.__setitem__(0, now[0] + seconds),
    )

    class _Goal:
        def __init__(self) -> None:
            self.request = SimpleNamespace(trajectory=trajectory)
            self.is_cancel_requested = False
            self.succeeded = False

        def succeed(self) -> None:
            self.succeeded = True

        def abort(self) -> None:
            pytest.fail("the move should complete after receive-only feedback recovery")

        def canceled(self) -> None:
            pytest.fail("trajectory should not be canceled")

    goal = _Goal()
    server._active_goal = None

    result = server._execute(goal)

    assert goal.succeeded is True
    assert result.error_code == 0
    assert dispatches == [path]
    assert len(reconnects) == 1
    assert reconnects[0] >= module.UR5E_RTDE_FEEDBACK_RECONNECT_AFTER_SEC
    assert statuses[-1]["rtde_feedback_reconnect_count"] == 1
    assert len(published) == 1


def test_rtde_persistent_feedback_loss_stops_and_fails_without_redispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_rtde_server_module()
    server = object.__new__(module.UR5eRTDETrajectoryServer)
    server.control = object()
    server.receive = object()
    server._active_lock = threading.Lock()
    server._active_goal_status = None
    server._latched_terminal_status = None
    published: list[object] = []
    server._rviz_goal_state_pub = SimpleNamespace(
        publish=lambda message: published.append(message)
    )
    server.get_logger = lambda: SimpleNamespace(warning=lambda _message: None)
    server._connect_control_for_goal = lambda: None
    server._current_position_map = lambda: dict.fromkeys(module.ARM_JOINTS, 0.0)
    server._joint_states_fresh = lambda: True
    dispatches: list[list[list[float]]] = []
    server._execute_movej_path = (
        lambda path: dispatches.append(path) or ("True", "asynchronous_keyword")
    )
    stopped: list[bool] = []
    server._stop_motion = lambda: stopped.append(True)
    statuses: list[dict[str, Any]] = []
    latched: list[bool] = []
    server._write_status = lambda status: statuses.append(dict(status))
    server._write_active_goal_status = lambda _goal, status: setattr(
        server, "_active_goal_status", dict(status)
    )

    def _finish(_goal: Any, status: dict[str, Any], *, latch_status: bool = False) -> None:
        statuses.append(dict(status))
        latched.append(latch_status)

    server._finish_active_goal_status = _finish
    server._clear_active_goal = lambda _goal: None
    server._read_actual_q = lambda: [0.0] * 6
    server._read_actual_qd = lambda: [0.0] * 6
    server._read_feedback_timestamp = lambda: 100.0
    reconnects: list[bool] = []
    server._reconnect_receive = lambda: reconnects.append(True) or "End of file"

    point = SimpleNamespace(time_from_start=SimpleNamespace(sec=2, nanosec=0))
    trajectory = SimpleNamespace(points=[point])
    monkeypatch.setattr(
        module,
        "prepare_rtde_trajectory",
        lambda _trajectory, _positions: (True, trajectory, module._status_base()),
    )
    path = [[1.0] * 6 + [0.1, 0.1, 0.0]]
    monkeypatch.setattr(module, "rtde_movej_path", lambda _trajectory: path)
    monkeypatch.setattr(module.rclpy, "ok", lambda: True)
    now = [0.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(module.time, "time", lambda: now[0])
    monkeypatch.setattr(
        module.time,
        "sleep",
        lambda seconds: now.__setitem__(0, now[0] + seconds),
    )

    class _Goal:
        def __init__(self) -> None:
            self.request = SimpleNamespace(trajectory=trajectory)
            self.is_cancel_requested = False
            self.aborted = False

        def succeed(self) -> None:
            pytest.fail("stale physical feedback must not report success")

        def abort(self) -> None:
            self.aborted = True

        def canceled(self) -> None:
            pytest.fail("trajectory should not be canceled")

    goal = _Goal()
    server._active_goal = None

    result = server._execute(goal)

    expected = (
        "UR5e RTDE trajectory feedback stopped advancing and did not recover within "
        f"{module.UR5E_RTDE_FEEDBACK_RECOVERY_TIMEOUT_SEC:.2f} s"
    )
    assert goal.aborted is True
    assert result.error_code == -1
    assert result.error_string == expected
    assert dispatches == [path]
    assert reconnects
    assert stopped == [True]
    assert latched[-1] is True
    assert statuses[-1]["joint_states_fresh"] is False
    assert published == []


def test_rtde_async_status_cannot_end_a_physically_moving_trajectory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_rtde_server_module()
    server = object.__new__(module.UR5eRTDETrajectoryServer)
    server.control = object()
    server.receive = object()
    server._active_lock = threading.Lock()
    server._active_goal_status = None
    server._latched_terminal_status = None
    server._rviz_goal_state_pub = SimpleNamespace(publish=lambda _message: None)
    server.get_logger = lambda: SimpleNamespace(warning=lambda _message: None)
    server._connect_control_for_goal = lambda: None
    server._current_position_map = lambda: dict.fromkeys(module.ARM_JOINTS, 0.0)
    server._joint_states_fresh = lambda: True
    server._execute_movej_path = lambda _path: ("True", "asynchronous_keyword")
    server._stop_motion = lambda: pytest.fail(
        "asynchronous-operation flags must not stop physical motion"
    )
    server._async_operation_status = lambda: pytest.fail(
        "physical completion must not query asynchronous-operation flags"
    )
    statuses: list[dict[str, Any]] = []
    server._write_status = lambda status: statuses.append(dict(status))
    server._write_active_goal_status = lambda _goal, status: setattr(
        server, "_active_goal_status", dict(status)
    )
    server._finish_active_goal_status = (
        lambda _goal, status, **_kwargs: statuses.append(dict(status))
    )
    server._clear_active_goal = lambda _goal: None

    point = SimpleNamespace(time_from_start=SimpleNamespace(sec=2, nanosec=0))
    trajectory = SimpleNamespace(points=[point])
    monkeypatch.setattr(
        module,
        "prepare_rtde_trajectory",
        lambda _trajectory, _positions: (True, trajectory, module._status_base()),
    )
    monkeypatch.setattr(
        module,
        "rtde_movej_path",
        lambda _trajectory: [[1.0] * 6 + [0.1, 0.1, 0.0]],
    )
    monkeypatch.setattr(module.rclpy, "ok", lambda: True)
    now = [0.0]
    server._read_actual_q = lambda: [min(now[0], 1.0)] * 6
    server._read_actual_qd = lambda: [0.2 if now[0] < 1.0 else 0.0] * 6
    server._read_feedback_timestamp = lambda: now[0]
    monkeypatch.setattr(module.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(module.time, "time", lambda: now[0])
    monkeypatch.setattr(
        module.time,
        "sleep",
        lambda seconds: now.__setitem__(0, now[0] + seconds),
    )

    class _Goal:
        def __init__(self) -> None:
            self.request = SimpleNamespace(trajectory=trajectory)
            self.is_cancel_requested = False
            self.succeeded = False

        def succeed(self) -> None:
            self.succeeded = True

        def abort(self) -> None:
            pytest.fail("physical motion should continue to the final target")

        def canceled(self) -> None:
            pytest.fail("trajectory should not be canceled")

    goal = _Goal()
    server._active_goal = None

    result = server._execute(goal)

    assert goal.succeeded is True
    assert result.error_code == 0
    assert statuses[-1]["final_joint_error_rad"] == pytest.approx(0.0)


def test_rtde_fresh_stationary_feedback_detects_a_real_early_stop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_rtde_server_module()
    server = object.__new__(module.UR5eRTDETrajectoryServer)
    server.control = object()
    server.receive = object()
    server._active_lock = threading.Lock()
    server._active_goal_status = None
    server._latched_terminal_status = None
    server._rviz_goal_state_pub = SimpleNamespace(publish=lambda _message: None)
    server.get_logger = lambda: SimpleNamespace(warning=lambda _message: None)
    server._connect_control_for_goal = lambda: None
    server._current_position_map = lambda: dict.fromkeys(module.ARM_JOINTS, 0.0)
    server._joint_states_fresh = lambda: True
    server._execute_movej_path = lambda _path: ("True", "asynchronous_keyword")
    server._stop_motion = lambda: pytest.fail(
        "a fresh stationary sample proves the robot is already stopped"
    )
    statuses: list[dict[str, Any]] = []
    latched: list[bool] = []
    server._write_status = lambda status: statuses.append(dict(status))
    server._write_active_goal_status = lambda _goal, status: setattr(
        server, "_active_goal_status", dict(status)
    )

    def _finish(_goal: Any, status: dict[str, Any], *, latch_status: bool = False) -> None:
        statuses.append(dict(status))
        latched.append(latch_status)

    server._finish_active_goal_status = _finish
    server._clear_active_goal = lambda _goal: None

    point = SimpleNamespace(time_from_start=SimpleNamespace(sec=2, nanosec=0))
    trajectory = SimpleNamespace(points=[point])
    monkeypatch.setattr(
        module,
        "prepare_rtde_trajectory",
        lambda _trajectory, _positions: (True, trajectory, module._status_base()),
    )
    monkeypatch.setattr(
        module,
        "rtde_movej_path",
        lambda _trajectory: [[1.0] * 6 + [0.1, 0.1, 0.0]],
    )
    monkeypatch.setattr(module.rclpy, "ok", lambda: True)
    now = [0.0]
    server._read_actual_q = lambda: [min(now[0] * 2.5, 0.5)] * 6
    server._read_actual_qd = lambda: [0.2 if now[0] < 0.2 else 0.0] * 6
    server._read_feedback_timestamp = lambda: now[0]
    monkeypatch.setattr(module.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(module.time, "time", lambda: now[0])
    monkeypatch.setattr(
        module.time,
        "sleep",
        lambda seconds: now.__setitem__(0, now[0] + seconds),
    )

    class _Goal:
        def __init__(self) -> None:
            self.request = SimpleNamespace(trajectory=trajectory)
            self.is_cancel_requested = False
            self.aborted = False

        def succeed(self) -> None:
            pytest.fail("a physical stop away from target must not succeed")

        def abort(self) -> None:
            self.aborted = True

        def canceled(self) -> None:
            pytest.fail("trajectory should not be canceled")

    goal = _Goal()
    server._active_goal = None

    result = server._execute(goal)

    assert goal.aborted is True
    assert result.error_string == (
        "UR5e RTDE trajectory ended before reaching the final joint target"
    )
    assert latched[-1] is True
    assert statuses[-1]["stopped_away_hold_sec"] >= (
        module.UR5E_RTDE_STOPPED_AWAY_HOLD_SEC
    )


def test_rtde_result_timeout_uses_moveit_configured_allowance(tmp_path: Path) -> None:
    module = _load_rtde_server_module()
    config_path = tmp_path / "hardware_runtime.yaml"
    config_path.write_text(
        """\
ur5e:
  moveit:
    rtde_allowed_execution_duration_scaling: 6.0
    rtde_allowed_goal_duration_margin: 17.0
""",
        encoding="utf-8",
    )

    module._apply_hardware_arms_config(config_path)

    assert module.UR5E_RTDE_ALLOWED_EXECUTION_DURATION_SCALING == 6.0
    assert module.UR5E_RTDE_RESULT_MARGIN_SEC == 17.0
    assert module._trajectory_result_timeout_sec(2.0, 10.0) == 29.0
    assert module._trajectory_result_timeout_sec(3.0, 10.0) == 35.0


def test_rtde_does_not_timeout_at_old_fixed_margin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_rtde_server_module()
    server = object.__new__(module.UR5eRTDETrajectoryServer)
    server.control = object()
    server.receive = object()
    server._active_lock = threading.Lock()
    server._active_goal_status = None
    server._latched_terminal_status = None
    server._rviz_goal_state_pub = SimpleNamespace(publish=lambda _message: None)
    server.get_logger = lambda: SimpleNamespace(warning=lambda _message: None)
    server._connect_control_for_goal = lambda: None
    server._current_position_map = lambda: dict.fromkeys(module.ARM_JOINTS, 0.0)
    server._joint_states_fresh = lambda: True
    server._execute_movej_path = lambda _path: ("True", "asynchronous_keyword")
    server._stop_motion = lambda: pytest.fail("the extended valid trajectory must not be stopped")
    statuses: list[dict[str, Any]] = []
    server._write_status = lambda status: statuses.append(dict(status))
    server._write_active_goal_status = lambda _goal, status: setattr(
        server, "_active_goal_status", dict(status)
    )
    server._finish_active_goal_status = (
        lambda _goal, status, **_kwargs: statuses.append(dict(status))
    )
    server._clear_active_goal = lambda _goal: None
    server._read_actual_q = lambda: [0.0] * 6

    point = SimpleNamespace(time_from_start=SimpleNamespace(sec=2, nanosec=0))
    trajectory = SimpleNamespace(points=[point])
    monkeypatch.setattr(
        module,
        "prepare_rtde_trajectory",
        lambda _trajectory, _positions: (True, trajectory, module._status_base()),
    )
    monkeypatch.setattr(
        module,
        "rtde_movej_path",
        lambda _trajectory: [[0.0] * 6 + [0.1, 0.1, 0.0]],
    )
    monkeypatch.setattr(module.rclpy, "ok", lambda: True)
    now = [0.0]
    server._read_actual_qd = lambda: [0.02 if now[0] < 15.0 else 0.0] * 6
    server._read_feedback_timestamp = lambda: now[0]
    server._async_operation_status = _unsupported_async_operation_status
    monkeypatch.setattr(module.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(module.time, "time", lambda: now[0])
    monkeypatch.setattr(module.time, "sleep", lambda seconds: now.__setitem__(0, now[0] + seconds))

    class _Goal:
        def __init__(self) -> None:
            self.request = SimpleNamespace(trajectory=trajectory)
            self.is_cancel_requested = False
            self.succeeded = False

        def succeed(self) -> None:
            self.succeeded = True

        def abort(self) -> None:
            pytest.fail("dynamic deadline should allow this trajectory to settle")

        def canceled(self) -> None:
            pytest.fail("trajectory should not be canceled")

    goal = _Goal()
    server._active_goal = None

    result = server._execute(goal)

    assert goal.succeeded is True
    assert result.error_code == 0
    assert now[0] > 15.0
    assert statuses[-1]["trajectory_result_timeout_sec"] == 36.0
    assert statuses[-1]["trajectory_elapsed_sec"] > 15.0


def test_rtde_terminal_diagnostics_survive_primary_status_replacement(
    tmp_path: Path,
) -> None:
    module = _load_rtde_server_module()
    server = object.__new__(module.UR5eRTDETrajectoryServer)
    server.status_file = tmp_path / "cais_ur5e_rtde_trajectory_status.json"
    server.terminal_status_file = tmp_path / "cais_ur5e_rtde_trajectory_status_last_terminal.json"
    server.monitor_only = False
    server.ros_domain_id = 42
    server._status_lock = threading.Lock()
    server._active_lock = threading.Lock()
    goal = object()
    server._active_goal = goal
    server._active_goal_status = {}
    server._latched_terminal_status = None
    terminal = {
        **module._status_base(),
        "state": "failed",
        "message": "UR5e RTDE trajectory result timeout",
        "blocked_reason": "UR5e RTDE trajectory result timeout",
        "trajectory_elapsed_sec": 28.2,
        "final_joint_error_rad": 0.031,
        "max_observed_joint_velocity_rad_s": 0.42,
        "stationary_hold_sec": 0.08,
    }

    server._finish_active_goal_status(goal, terminal, latch_status=True)
    server._write_status({**module._status_base(), "state": "ready"})

    primary = json.loads(server.status_file.read_text(encoding="utf-8"))
    preserved = json.loads(server.terminal_status_file.read_text(encoding="utf-8"))
    assert primary["state"] == "ready"
    assert preserved["state"] == "failed"
    assert preserved["trajectory_elapsed_sec"] == 28.2
    assert preserved["final_joint_error_rad"] == 0.031
    assert preserved["max_observed_joint_velocity_rad_s"] == 0.42
    assert preserved["stationary_hold_sec"] == 0.08


def test_prepared_controller_reuses_actions_and_world_tool0_without_subprocess() -> None:
    action_calls: list[str] = []

    def _action_ready(label: str) -> Any:
        return SimpleNamespace(
            wait_for_server=lambda **_kwargs: action_calls.append(label) or True
        )

    action = _action_ready("arm")
    ready, error = SystemBridge._prepared_action_client_ready(
        action,
        action_name="/cais_ur5e_rtde_trajectory_controller/follow_joint_trajectory",
    )
    assert ready is True
    assert error == ""

    controller = SimpleNamespace(
        frame_id="world",
        ee_link="tool0",
        _ur5e_hardware_trajectory_action=(
            "/cais_ur5e_rtde_trajectory_controller/follow_joint_trajectory"
        ),
        _ur5e_hardware_trajectory_client=action,
        _rg2_action_name="/ur5e_rg2_gripper_traj_controller/follow_joint_trajectory",
        _rg2_action_client=_action_ready("gripper"),
        get_current_pose=lambda: {
            "success": True,
            "pose": {"x": 0.1, "y": 0.2, "z": 1.1, "qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0},
        },
    )
    agent = SimpleNamespace(_controller=controller)
    bridge = object.__new__(SystemBridge)
    bridge._robot_function_capture_snapshot = lambda *_args: pytest.fail(
        "prepared world -> tool0 must not launch the snapshot subprocess"
    )
    bridge._digital_twin_domain_ids = lambda: {"hardware": 42}
    bridge._digital_twin_hardware_domain_id = lambda *_args: 42
    bridge._ur5e_rtde_trajectory_status = lambda: {
        "updated_at": realsense_roboflow_node.time.time(),
        "ros_domain_id": 42,
        "action_name": (
            "/cais_ur5e_rtde_trajectory_controller/follow_joint_trajectory"
        ),
        "rtde_receive_connected": True,
        "joint_states_fresh": True,
        "rtde_control_connected": True,
    }
    bridge._wait_for_ros_action = lambda *_args, **_kwargs: pytest.fail(
        "warm Run and Confirm Run must not invoke ROS CLI action discovery"
    )

    for _phase in ("Run", "Confirm Run"):
        motion_readiness, motion_error = bridge._digital_twin_ur5e_motion_readiness(
            "ur5e only",
            {},
            agent,
        )
        gripper_readiness, gripper_error = bridge._digital_twin_ur5e_gripper_readiness(
            "ur5e only",
            42,
            agent,
        )
        assert motion_error == ""
        assert motion_readiness["trajectory_action_ready"] is True
        assert gripper_error == ""
        assert gripper_readiness["gripper_action_ready"] is True

    result = bridge._robot_function_execution_pose_readiness("ur5e only", "ur5e", agent)

    assert result["success"] is True
    assert result["world_tool0_ready"] is True
    assert result["waypoint"]["pose"]["child_frame_id"] == "tool0"
    assert action_calls == ["arm", "arm", "gripper", "arm", "gripper"]


def test_cached_controller_prewarm_runs_once(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[float] = []
    controller = SimpleNamespace(
        wait_for_services=lambda timeout_sec: calls.append(timeout_sec) or True,
    )
    agent = SimpleNamespace(
        _controller=controller,
        _controller_prewarm_done=False,
        controller_prewarm_timeout_s=20.0,
    )
    bridge = object.__new__(SystemBridge)

    async def _to_thread(function: Any, *args: Any) -> Any:
        return function(*args)

    monkeypatch.setattr(asyncio, "to_thread", _to_thread)

    async def _exercise() -> tuple[str, str]:
        first = await bridge._prewarm_ur5e_robot_function_agent(agent)
        second = await bridge._prewarm_ur5e_robot_function_agent(agent)
        return first, second

    assert asyncio.run(_exercise()) == ("", "")
    assert calls == [20.0]


def test_execution_progress_distinguishes_settling_from_detection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    detection_status = tmp_path / "detection_status.json"
    detection_status.write_text(
        json.dumps({"updated_at": 11.0, "stage": "detection"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(bridge_module, "_UR5E_DETECTION_STATUS", detection_status)
    bridge = object.__new__(SystemBridge)
    bridge._ur5e_robot_function_execution_active = "pick_approach"
    bridge._ur5e_robot_function_execution_stage = "settling"
    bridge._ur5e_robot_function_execution_started_at = 10.0

    result = bridge.digital_twin_robot_function_execution_progress()

    assert result["stage"] == "detection"
    assert result["message"] == "Running fresh /detect_all inference."


def test_execution_progress_surfaces_wait_for_active_inference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    detection_status = tmp_path / "detection_status.json"
    message = "Waiting for the active Roboflow inference before fresh /detect_all."
    detection_status.write_text(
        json.dumps({"updated_at": 11.0, "stage": "settling", "message": message}),
        encoding="utf-8",
    )
    monkeypatch.setattr(bridge_module, "_UR5E_DETECTION_STATUS", detection_status)
    bridge = object.__new__(SystemBridge)
    bridge._ur5e_robot_function_execution_active = "pick_approach"
    bridge._ur5e_robot_function_execution_stage = "settling"
    bridge._ur5e_robot_function_execution_started_at = 10.0

    result = bridge.digital_twin_robot_function_execution_progress()

    assert result["stage"] == "settling"
    assert result["message"] == message
