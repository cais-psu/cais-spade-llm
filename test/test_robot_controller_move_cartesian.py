"""Focused tests for Cartesian primitive orientation handling."""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import threading
import time
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from cais_spade_llm.product.profile import ProductProfile
from cais_spade_llm.resources.robot import hardware_pick_place_controller
from cais_spade_llm.resources.robot.gazebo_pick_place_controller import (
    _MOVE_INSERT_MG_TACTILE_POLICY_FIELDS,
    _PHYSICAL_XARM6_ASSEMBLY_SLOT_INSERT_ERROR,
    GazeboPickPlaceController,
    compute_move_insert_geometry,
    derive_move_insert_timeout_sec,
    move_insert_learning_evidence_sha256,
    move_insert_learning_policy_sha256,
    move_insert_profile_sha256,
    resolve_move_insert_profile,
)
from cais_spade_llm.resources.robot.hardware_pick_place_controller import (
    HardwarePickPlaceController,
    UR5eHardwareController,
    XArm6HardwareController,
)

ROOT = Path(__file__).resolve().parents[1]


class _FakePose:
    def __init__(self) -> None:
        self.position = SimpleNamespace(x=0.0, y=0.0, z=0.0)
        self.orientation = SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0)


class _TransformLookupError(Exception):
    pass


def _tf_controller_double() -> GazeboPickPlaceController:
    controller = object.__new__(GazeboPickPlaceController)
    controller.frame_id = "world"
    controller.ee_link = "tool0"
    controller.tf_lookup_timeout_sec = 0.2
    controller._last_failure_message = ""
    controller._Pose = _FakePose
    controller._rclpy = SimpleNamespace(time=SimpleNamespace(Time=lambda: object()))
    controller._tf2_ros = SimpleNamespace(
        LookupException=_TransformLookupError,
        ConnectivityException=_TransformLookupError,
        ExtrapolationException=_TransformLookupError,
    )
    controller._log = lambda: SimpleNamespace(error=lambda _message: None)
    return controller


def test_get_ee_pose_waits_for_new_tf_listener_to_receive_static_transform() -> None:
    controller = _tf_controller_double()
    attempts = 0
    rotation = SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0)

    def lookup_transform(*_args: Any) -> Any:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise _TransformLookupError('"world" does not exist yet')
        return SimpleNamespace(
            transform=SimpleNamespace(
                translation=SimpleNamespace(x=0.4, y=0.5, z=1.2),
                rotation=rotation,
            )
        )

    controller._tf_buffer = SimpleNamespace(lookup_transform=lookup_transform)

    pose = controller._get_ee_pose()

    assert attempts == 3
    assert pose is not None
    assert (pose.position.x, pose.position.y, pose.position.z) == (0.4, 0.5, 1.2)
    assert pose.orientation is rotation
    assert controller._last_failure_message == ""


def test_get_current_pose_reports_tf_frames_after_retry_timeout() -> None:
    controller = _tf_controller_double()
    controller.tf_lookup_timeout_sec = 0.0
    controller.wait_for_services = lambda: True
    controller._tf_buffer = SimpleNamespace(
        lookup_transform=lambda *_args: (_ for _ in ()).throw(
            _TransformLookupError('"world" does not exist')
        )
    )

    result = controller.get_current_pose()

    assert result["success"] is False
    assert result["message"].startswith("cannot read current ee pose:")
    assert "world -> tool0" in result["message"]


def _controller_double(
    controller_type: type[GazeboPickPlaceController] = GazeboPickPlaceController,
    *,
    current_x: float = 0.1,
    current_y: float = 0.2,
) -> tuple[GazeboPickPlaceController, dict[str, Any]]:
    controller = object.__new__(controller_type)
    current_orientation = SimpleNamespace(x=0.1, y=0.2, z=0.3, w=0.9)
    calls: dict[str, Any] = {"wait_count": 0, "moves": []}

    def wait_for_services() -> bool:
        calls["wait_count"] += 1
        return True

    def record_move(kind: str):
        def move(x: float, y: float, z: float, **kwargs: Any) -> dict[str, Any]:
            calls["moves"].append((kind, x, y, z, kwargs))
            return {"success": True, "message": kind}

        return move

    controller._Pose = _FakePose
    controller.wait_for_services = wait_for_services
    controller._get_ee_pose = lambda: SimpleNamespace(
        position=SimpleNamespace(x=current_x, y=current_y, z=0.3),
        orientation=current_orientation,
    )
    controller._move_pose_direct = record_move("direct")
    controller._move_xy_at_z = record_move("xy")
    calls["current_orientation"] = current_orientation
    return controller, calls


@pytest.mark.parametrize(
    "controller_type",
    [GazeboPickPlaceController, HardwarePickPlaceController],
)
def test_move_cartesian_without_quaternion_keeps_current_orientation_and_call_shape(
    controller_type: type[GazeboPickPlaceController],
) -> None:
    controller, calls = _controller_double(controller_type)

    result = controller.move_cartesian(0.1, 0.2, 0.4, 1.7)

    assert result["success"] is True, result
    assert calls["wait_count"] == 1
    assert len(calls["moves"]) == 1
    kind, x, y, z, kwargs = calls["moves"][0]
    assert (kind, x, y, z) == ("direct", 0.1, 0.2, 0.4)
    assert kwargs["speed"] == 1.7
    current = calls["current_orientation"]
    current_norm = math.hypot(current.x, current.y, current.z, current.w)
    expected_orientation = tuple(
        value / current_norm
        for value in (current.x, current.y, current.z, current.w)
    )
    orientation = kwargs["orientation"]
    assert (orientation.x, orientation.y, orientation.z, orientation.w) == pytest.approx(
        expected_orientation
    )
    assert result["absolute_position"] == pytest.approx(
        {
            "x": 0.1,
            "y": 0.2,
            "z": 0.4,
            "qx": current.x / current_norm,
            "qy": current.y / current_norm,
            "qz": current.z / current_norm,
            "qw": current.w / current_norm,
        }
    )


@pytest.mark.parametrize(
    "controller_type",
    [GazeboPickPlaceController, HardwarePickPlaceController],
)
def test_move_cartesian_uses_normalized_complete_quaternion(
    controller_type: type[GazeboPickPlaceController],
) -> None:
    controller, calls = _controller_double(controller_type, current_x=0.0, current_y=0.0)

    result = controller.move_cartesian(
        0.1,
        0.2,
        0.3,
        qx=0.0,
        qy=0.0,
        qz=0.0,
        qw=2.0,
    )

    assert result["success"] is True
    assert calls["wait_count"] == 1
    assert len(calls["moves"]) == 1
    kind, _x, _y, _z, kwargs = calls["moves"][0]
    assert kind == "xy"
    orientation = kwargs["orientation"]
    assert (orientation.x, orientation.y, orientation.z, orientation.w) == (
        0.0,
        0.0,
        0.0,
        1.0,
    )
    assert math.isclose(
        math.hypot(orientation.x, orientation.y, orientation.z, orientation.w),
        1.0,
    )
    assert result["absolute_position"] == pytest.approx(
        {
            "x": 0.1,
            "y": 0.2,
            "z": 0.3,
            "qx": 0.0,
            "qy": 0.0,
            "qz": 0.0,
            "qw": 1.0,
        }
    )


@pytest.mark.parametrize(
    ("quaternion", "message"),
    [
        ({"qx": 0.0}, "requires qx, qy, qz, and qw together"),
        (
            {"qx": float("nan"), "qy": 0.0, "qz": 0.0, "qw": 1.0},
            "must contain finite numeric values",
        ),
        (
            {"qx": "invalid", "qy": 0.0, "qz": 0.0, "qw": 1.0},
            "must contain finite numeric values",
        ),
        (
            {"qx": 10**10000, "qy": 0.0, "qz": 0.0, "qw": 1.0},
            "must contain finite numeric values",
        ),
        (
            {"qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 0.0},
            "must be non-zero",
        ),
    ],
)
def test_move_cartesian_rejects_invalid_quaternion_before_controller_readiness(
    quaternion: dict[str, Any],
    message: str,
) -> None:
    controller, calls = _controller_double()

    result = controller.move_cartesian(0.1, 0.2, 0.3, **quaternion)

    assert result["success"] is False
    assert message in result["message"]
    assert calls["wait_count"] == 0
    assert calls["moves"] == []


class _CartesianGoal:
    def __init__(self) -> None:
        self.target_tool0_pose: Any | None = None
        self.speed_m_s = 0.0
        self.acceleration_m_s2 = 0.0


class _PoseStamped:
    def __init__(self) -> None:
        self.header = SimpleNamespace(frame_id="", stamp=None)
        self.pose = _FakePose()


class _ImmediateFuture:
    def __init__(self, value: Any) -> None:
        self.value = value

    def done(self) -> bool:
        return True

    def result(self) -> Any:
        return self.value


class _PendingFuture:
    def done(self) -> bool:
        return False


class _SettableFuture:
    def __init__(self) -> None:
        self._event = threading.Event()
        self._value: Any = None

    def done(self) -> bool:
        return self._event.is_set()

    def result(self) -> Any:
        assert self._event.is_set()
        return self._value

    def set_result(self, value: Any) -> None:
        self._value = value
        self._event.set()


def test_ur5e_physical_cartesian_uses_only_direct_rtde_action() -> None:
    controller = object.__new__(UR5eHardwareController)
    goals: list[Any] = []
    wrapped = SimpleNamespace(
        status=4,
        result=SimpleNamespace(error_code=0, error_string=""),
    )
    goal_handle = SimpleNamespace(
        accepted=True,
        get_result_async=lambda: _ImmediateFuture(wrapped),
    )
    client = SimpleNamespace(
        wait_for_server=lambda timeout_sec: timeout_sec == 2.0,
        send_goal_async=lambda goal: goals.append(goal) or _ImmediateFuture(goal_handle),
    )
    controller.wait_for_services = lambda: True
    controller._ur5e_hardware_cartesian_client = client
    controller._ur5e_hardware_cartesian_action = (
        "/cais_ur5e_rtde_cartesian_controller/move_cartesian"
    )
    controller._MoveUR5eCartesian = SimpleNamespace(Goal=_CartesianGoal)
    controller._PoseStamped = _PoseStamped
    controller._node = SimpleNamespace(
        get_clock=lambda: SimpleNamespace(
            now=lambda: SimpleNamespace(to_msg=lambda: "stamp")
        )
    )
    controller._wait_future = lambda future, **_kwargs: future.value
    controller._cancel_ur5e_hardware_trajectory_goal = lambda *_args, **_kwargs: ""
    controller._ur5e_cartesian_speed_m_s = 0.05
    controller._ur5e_cartesian_acceleration_m_s2 = 0.10
    controller.trajectory_time_scale = 0.45
    controller._last_failure_message = ""
    controller._cart_client = SimpleNamespace(
        call_async=lambda *_args: pytest.fail("physical UR5e called MoveIt")
    )
    controller._exec_client = SimpleNamespace(
        send_goal_async=lambda *_args: pytest.fail("physical UR5e called MoveIt")
    )
    target = _FakePose()
    target.position.x, target.position.y, target.position.z = (0.35, 0.36, 1.25)

    assert controller._cartesian_move(target, label="descend") is True

    assert len(goals) == 1
    assert goals[0].target_tool0_pose.header.frame_id == "world"
    assert goals[0].target_tool0_pose.header.stamp == "stamp"
    assert goals[0].speed_m_s == pytest.approx(0.05)
    assert goals[0].acceleration_m_s2 == pytest.approx(0.10)


def test_ur5e_physical_lift_retains_call_until_delayed_acceptance_settles() -> None:
    controller = object.__new__(UR5eHardwareController)
    send_future = _SettableFuture()
    wrapped = SimpleNamespace(
        status=4,
        result=SimpleNamespace(error_code=0, error_string=""),
    )
    goal_handle = SimpleNamespace(
        accepted=True,
        get_result_async=lambda: _ImmediateFuture(wrapped),
    )
    controller.wait_for_services = lambda: True
    controller._ur5e_hardware_cartesian_client = SimpleNamespace(
        wait_for_server=lambda timeout_sec: timeout_sec == 2.0,
        send_goal_async=lambda _goal: send_future,
    )
    controller._ur5e_hardware_cartesian_action = (
        "/cais_ur5e_rtde_cartesian_controller/move_cartesian"
    )
    controller._MoveUR5eCartesian = SimpleNamespace(Goal=_CartesianGoal)
    controller._PoseStamped = _PoseStamped
    controller._node = SimpleNamespace(
        get_clock=lambda: SimpleNamespace(
            now=lambda: SimpleNamespace(to_msg=lambda: "stamp")
        )
    )
    controller._ur5e_action_send_timeout_sec = 0.01
    controller._ur5e_cartesian_speed_m_s = 0.05
    controller._ur5e_cartesian_acceleration_m_s2 = 0.10
    controller.trajectory_time_scale = 0.45
    controller._last_failure_message = ""
    target = _FakePose()
    target.position.x, target.position.y, target.position.z = (0.35, 0.36, 1.25)
    results: list[bool] = []

    worker = threading.Thread(
        target=lambda: results.append(controller._cartesian_move(target, label="lift"))
    )
    worker.start()
    time.sleep(0.05)

    assert worker.is_alive()

    send_future.set_result(goal_handle)
    worker.join(timeout=1.0)

    assert worker.is_alive() is False
    assert results == [True]


def test_ur5e_physical_lift_retains_call_when_terminal_observer_is_unavailable() -> None:
    controller = object.__new__(UR5eHardwareController)
    goal_handle = SimpleNamespace(
        accepted=True,
        get_result_async=lambda: (_ for _ in ()).throw(RuntimeError("observer lost")),
    )
    controller.wait_for_services = lambda: True
    controller._ur5e_hardware_cartesian_client = SimpleNamespace(
        wait_for_server=lambda timeout_sec: timeout_sec == 2.0,
        send_goal_async=lambda _goal: _ImmediateFuture(goal_handle),
    )
    controller._ur5e_hardware_cartesian_action = (
        "/cais_ur5e_rtde_cartesian_controller/move_cartesian"
    )
    controller._MoveUR5eCartesian = SimpleNamespace(Goal=_CartesianGoal)
    controller._PoseStamped = _PoseStamped
    controller._node = SimpleNamespace(
        get_clock=lambda: SimpleNamespace(
            now=lambda: SimpleNamespace(to_msg=lambda: "stamp")
        )
    )
    controller._ur5e_cartesian_speed_m_s = 0.05
    controller._ur5e_cartesian_acceleration_m_s2 = 0.10
    controller.trajectory_time_scale = 0.45
    controller._last_failure_message = ""
    entered = threading.Event()
    release = threading.Event()
    controller._retain_accepted_action_without_terminal_observer = (
        lambda: entered.set() or release.wait()
    )
    target = _FakePose()
    results: list[bool] = []
    worker = threading.Thread(
        target=lambda: results.append(controller._cartesian_move(target, label="lift"))
    )
    worker.start()

    assert entered.wait(timeout=1.0)
    assert worker.is_alive()

    release.set()
    worker.join(timeout=1.0)

    assert results == [False]
    assert "observer lost" in controller._last_failure_message


class _MoveCartesianRequest:
    pass


def test_xarm6_physical_cartesian_converts_world_link_eef_to_native_service() -> None:
    controller = object.__new__(XArm6HardwareController)
    requests: list[Any] = []
    handoffs: list[str] = []
    client = SimpleNamespace(
        call_async=lambda request: requests.append(request)
        or _ImmediateFuture(SimpleNamespace(ret=0, message="OK"))
    )
    controller.wait_for_services = lambda: True
    controller._xarm6_hardware_cartesian_client = client
    controller._xarm6_hardware_cartesian_service = "/xarm6/xarm/set_position"
    controller._MoveCartesian = SimpleNamespace(Request=_MoveCartesianRequest)
    controller._xarm6_cartesian_speed_mm_s = 50.0
    controller._xarm6_cartesian_acceleration_mm_s2 = 100.0
    controller._xarm6_cartesian_position_tolerance_m = 0.003
    controller._xarm6_cartesian_orientation_tolerance_rad = math.radians(3.0)
    controller._xarm6_robot_state_lock = threading.Lock()
    controller._xarm6_robot_state = SimpleNamespace(
        pose=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        offset=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        mode=1,
        state=0,
    )
    controller._xarm6_robot_state_received_monotonic = time.monotonic()
    controller.trajectory_time_scale = 0.45
    controller.frame_id = "world"
    controller._rclpy = SimpleNamespace(time=SimpleNamespace(Time=lambda: object()))
    controller._tf2_ros = SimpleNamespace(
        LookupException=_TransformLookupError,
        ConnectivityException=_TransformLookupError,
        ExtrapolationException=_TransformLookupError,
    )
    controller._tf_buffer = SimpleNamespace(
        lookup_transform=lambda *_args: SimpleNamespace(
            transform=SimpleNamespace(
                translation=SimpleNamespace(x=0.2, y=-0.1, z=0.3),
                rotation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
            )
        )
    )
    controller._wait_future = lambda future, **_kwargs: future.value
    controller._wait_for_xarm6_trajectory_controller_state = (
        lambda state, **_kwargs: (state == "active", "OK")
    )
    controller._prepare_xarm6_firmware_cartesian_mode = lambda: (
        handoffs.append("prepare_mode_0") or (True, "ready")
    )
    controller._restore_xarm6_trajectory_control = lambda: (
        handoffs.append("restore_mode_1") or (True, "restored")
    )
    target = _FakePose()
    target.position.x, target.position.y, target.position.z = (0.5, 0.2, 0.8)
    controller._get_ee_pose = lambda: target
    controller._last_failure_message = ""
    controller._cart_client = SimpleNamespace(
        call_async=lambda *_args: pytest.fail("physical xArm6 called MoveIt")
    )
    controller._exec_client = SimpleNamespace(
        send_goal_async=lambda *_args: pytest.fail("physical xArm6 called MoveIt")
    )

    assert controller._cartesian_move(target, label="move_above_part") is True

    assert len(requests) == 1
    request = requests[0]
    assert request.pose == pytest.approx([300.0, 300.0, 500.0, 0.0, 0.0, 0.0])
    assert request.speed == pytest.approx(50.0)
    assert request.acc == pytest.approx(100.0)
    assert request.wait is True
    assert request.is_tool_coord is False
    assert request.relative is False
    assert handoffs == ["prepare_mode_0", "restore_mode_1"]


def test_base_physical_cartesian_fails_closed_without_direct_implementation() -> None:
    controller = object.__new__(HardwarePickPlaceController)
    controller.robot_name = "unsupported"
    controller._last_failure_message = ""
    controller._log = lambda: SimpleNamespace(error=lambda _message: None)

    assert controller._cartesian_move(_FakePose(), label="descend") is False
    assert "direct physical Cartesian control is not implemented" in (
        controller._last_failure_message
    )


def test_physical_move_above_uses_one_direct_request_and_surfaces_failure() -> None:
    controller = object.__new__(HardwarePickPlaceController)
    controller._Pose = _FakePose
    controller.trajectory_time_scale = 1.0
    controller._last_failure_message = ""
    controller.wait_for_services = lambda: True
    controller._get_ee_pose = lambda: SimpleNamespace(
        orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0)
    )
    requests: list[Any] = []

    def reject_target(target: Any, label: str, **_kwargs: Any) -> bool:
        requests.append((target, label))
        controller._last_failure_message = (
            "/cais_ur5e_rtde_cartesian_controller/move_cartesian: "
            "world -> tool0 XY radius rejected"
        )
        return False

    controller._cartesian_move = reject_target

    result = controller._move_xy_at_z(0.363, -0.136, 1.453)

    assert result["success"] is False
    assert len(requests) == 1
    assert requests[0][1] == "move_xy_at_z"
    assert "world -> tool0 XY radius rejected" in result["message"]


def test_physical_controller_initialization_does_not_construct_moveit_or_gazebo_clients() -> None:
    source = inspect.getsource(HardwarePickPlaceController.init)

    assert "moveit_msgs" not in source
    assert "gazebo_msgs" not in source


def _move_insert_profile(*, validated_parts: list[str] | None = None) -> dict[str, Any]:
    return {
        "calibration_id": "insert-calibration-1",
        "validated_parts": list(validated_parts or ["MG"]),
        "pre_insert_offset_m": 0.01,
        "contact_speed_m_s": 0.002,
        "contact_force_delta_n": 2.0,
        "engagement_progress_m": 0.003,
        "insertion_force_n": 5.0,
        "spiral_radius_m": 0.001,
        "spiral_pitch_m": 0.0005,
        "spiral_speed_m_s": 0.002,
        "spiral_acceleration_m_s2": 0.02,
        "max_axial_force_n": 20.0,
        "max_lateral_force_n": 10.0,
        "max_torque_nm": 2.0,
        "tilt_tolerance_rad": 0.1,
        "seated_depth_tolerance_m": 0.001,
        "settle_time_sec": 0.2,
        "part_overrides": {},
        "qualifications": {},
    }


def _confirm_move_insert_profile(raw: dict[str, Any], part_name: str = "MG") -> None:
    profile_sha256, error = move_insert_profile_sha256(raw, part_name)
    assert error == ""
    recipes = dict(raw.get("demonstration_recipes") or {})
    recipe = dict(recipes.get(part_name) or {})
    recording_id = str(recipe.get("recording_id") or "recording-test")
    demonstration_sha256 = str(
        recipe.get("demonstration_sha256") or "e" * 64
    )
    trial_ids = ["trial-mg-1"]
    result_sha256s = ["1" * 64]
    trace_sha256s = ["4" * 64]
    qualification_policy = {
        "qualification_policy_version": 3,
        "required_consecutive_confirmed_trials": 1,
        "failure_resets_confirmed_trials": True,
        "recovered_soft_overload_may_count": True,
        "hard_limit_may_count": False,
        "identity_fields": [
            "robot",
            "tool_frame",
            "destination_location",
            "part_name",
            "profile_sha256",
            "hard_caps_sha256",
            "place_approach_recording_sha256",
            "board_calibration_id",
            "board_geometry_sha256",
            "recording_id",
            "demonstration_sha256",
        ],
    }
    qualification_identity = {
        "robot": "ur5e",
        "tool_frame": "tool0",
        "destination_location": "assembly_board-v1",
        "part_name": part_name,
        "profile_sha256": profile_sha256,
        "hard_caps_sha256": str(
            recipe.get("hard_caps_sha256") or "d" * 64
        ),
        "place_approach_recording_sha256": "b" * 64,
        "board_calibration_id": "board-calibration-1",
        "board_geometry_sha256": "c" * 64,
        "recording_id": recording_id,
        "demonstration_sha256": demonstration_sha256,
    }
    qualification_policy_sha256 = _canonical_sha256(qualification_policy)
    qualification_evidence_sha256 = _canonical_sha256(
        {
            "qualification_policy_sha256": qualification_policy_sha256,
            "qualification_identity": qualification_identity,
            "confirmed_trial_ids": trial_ids,
            "confirmed_trial_result_sha256s": result_sha256s,
            "confirmed_trial_trace_sha256s": trace_sha256s,
        }
    )
    raw.setdefault("qualifications", {})[part_name] = {
        "trial_id": trial_ids[-1],
        "confirmed_at": "2026-08-18T12:30:00Z",
        **qualification_identity,
        "board_generation": 1,
        "generation": 1,
        "qualification_policy_version": 3,
        "qualification_policy_sha256": qualification_policy_sha256,
        "required_confirmed_trials": 1,
        "confirmed_trial_count": 1,
        "confirmed_trial_ids": trial_ids,
        "confirmed_trial_result_sha256s": result_sha256s,
        "confirmed_trial_trace_sha256s": trace_sha256s,
        "qualification_evidence_sha256": qualification_evidence_sha256,
    }


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _refresh_move_insert_qualification_evidence(
    raw: dict[str, Any],
    part_name: str = "MG",
) -> None:
    qualification = raw["qualifications"][part_name]
    qualification_identity = {
        field_name: qualification[field_name]
        for field_name in (
            "robot",
            "tool_frame",
            "destination_location",
            "part_name",
            "profile_sha256",
            "hard_caps_sha256",
            "place_approach_recording_sha256",
            "board_calibration_id",
            "board_geometry_sha256",
            "recording_id",
            "demonstration_sha256",
        )
    }
    qualification["qualification_evidence_sha256"] = _canonical_sha256(
        {
            "qualification_policy_sha256": qualification[
                "qualification_policy_sha256"
            ],
            "qualification_identity": qualification_identity,
            "confirmed_trial_ids": qualification["confirmed_trial_ids"],
            "confirmed_trial_result_sha256s": qualification[
                "confirmed_trial_result_sha256s"
            ],
            "confirmed_trial_trace_sha256s": qualification[
                "confirmed_trial_trace_sha256s"
            ],
        }
    )


def _current_move_insert_demonstration_recipe(
    raw_recipe: dict[str, Any],
    *,
    recovery_values: dict[str, float] | None = None,
) -> dict[str, Any]:
    exact_recovery_values = {
        "insert_max_contact_search_radius_m": 0.01,
        "insert_max_disengagement_cycles": 6.0,
        "insert_search_peck_retreat_m": 0.003,
        "insert_search_peck_interval_sec": 0.75,
        **dict(recovery_values or {}),
    }
    learning_policy = {
        "learning_policy_version": 12,
        "axial_force_sign_convention": "compression_negative_dot",
        "force_filter": "time_window_median",
        "contact_hold_sec": 0.10,
        "torque_reference": "active_tcp",
        "limit_policy": "reject_not_clip",
        "relief_sequence": "unload_then_bounded_micro_backoff",
        "hard_limit_policy": "immediate_stop",
        "demonstration_speed_policy": "diagnostic_only",
        "rebound_policy": (
            "final_saved_depth_within_tolerance_of_post_contact_maximum"
        ),
        "force_depth_profile_points": 16,
        "axial_soft_overload_policy": (
            "learned_and_profile_exceedance_with_stalled_progress_"
            "guarded_below_hard_cap"
        ),
        "engagement_policy": "sustained_axial_progress_within_force_depth_profile",
        "seating_policy": "target_depth_stationary_stable_force_within_force_depth_profile",
        "force_uncertainty_floor_n": 4.0,
        "torque_uncertainty_floor_nm": 0.05,
        "insert_max_tool_flange_torque_nm": 3.0,
        "insert_soft_filter_window_sec": 0.05,
        "insert_soft_overload_hold_sec": 0.10,
        "insert_relief_unload_dwell_sec": 0.10,
        "insert_relief_clear_dwell_sec": 0.10,
        "insert_relief_timeout_sec": 1.0,
        "insert_relief_axial_force_ratio": 0.5,
        "insert_relief_reverse_force_ratio": 0.25,
        "insert_relief_clear_hysteresis_ratio": 0.8,
        "insert_relief_resume_ramp_sec": 0.10,
        "insert_relief_search_force_ratio": 0.5,
        "insert_relief_search_speed_ratio": 0.5,
        "insert_relief_backoff_step_m": 0.0001,
        "insert_max_relief_retreat_m": 0.0003,
        "insert_relief_stationary_speed_m_s": 0.0005,
        "insert_relief_stationary_angular_speed_rad_s": 0.01,
        "insert_max_relief_cycles": 3.0,
        "tactile_center_policy": (
            "deepest_stable_progress_then_lowest_normalized_lateral_load"
        ),
        "expanded_search_policy": "local_then_staged_3mm_5mm_10mm_low_preload",
        "cocked_recovery_sequence": (
            "unload_then_exact_pre_insert_withdrawal_recenter_retare_retry"
        ),
        "disengagement_lateral_clearance_policy": (
            "search_boundary_plus_start_position_tolerance"
        ),
        "search_peck_policy": (
            "stalled_spiral_bounded_axial_unload_then_low_preload_recontact"
        ),
        **exact_recovery_values,
    }
    recipe = {
        **raw_recipe,
        "recipe_version": 9,
        "learning_policy_version": 12,
        "learning_policy": learning_policy,
        "hard_caps": {
            "insert_max_axial_force_n": 30.0,
            "insert_max_lateral_force_n": 15.0,
            "insert_max_torque_nm": 3.0,
            "insert_max_tool_flange_torque_nm": 5.0,
            **exact_recovery_values,
            "insert_max_timeout_sec": 60.0,
        },
        "baseline_force_uncertainty_n": 0.2,
        "baseline_torque_uncertainty_nm": 0.01,
        "observed_filtered_axial_force_n": 10.0,
        "observed_filtered_lateral_force_n": 5.0,
        "observed_filtered_torque_nm": 1.0,
        "observed_tool_flange_torque_nm": 1.5,
        "observed_raw_axial_force_n": 11.0,
        "observed_raw_lateral_force_n": 5.5,
        "observed_raw_torque_nm": 1.1,
        "observed_raw_tool_flange_torque_nm": 1.6,
        "observed_advancing_speed_m_s": 0.007,
        "observed_peak_filtered_advancing_speed_m_s": 0.009,
        "seated_filtered_axial_force_n": 4.0,
        "seated_filtered_lateral_force_n": 1.0,
        "seated_filtered_torque_nm": 0.2,
        "seated_filtered_tool_flange_torque_nm": 0.3,
        "force_depth_profile": {
            "depth_fraction": [index / 15.0 for index in range(16)],
            "axial_upper_n": [12.0] * 16,
            "lateral_upper_n": [6.0] * 16,
            "torque_upper_nm": [1.2] * 16,
        },
    }
    recipe["force_depth_profile_sha256"] = _canonical_sha256(
        recipe["force_depth_profile"]
    )
    recipe["hard_caps_sha256"] = _canonical_sha256(recipe["hard_caps"])
    learning_policy_sha256, learning_policy_error = (
        move_insert_learning_policy_sha256(recipe)
    )
    assert learning_policy_error == ""
    recipe["learning_policy_sha256"] = learning_policy_sha256
    evidence_sha256, evidence_error = move_insert_learning_evidence_sha256(
        recipe
    )
    assert evidence_error == ""
    recipe["learning_evidence_sha256"] = evidence_sha256
    return recipe


def _move_insert_profile_with_current_recipe(
    part_name: str = "MG",
    *,
    recovery_values: dict[str, float] | None = None,
) -> dict[str, Any]:
    raw_profile = _move_insert_profile(validated_parts=[])
    learned_values = {
        key: value
        for key, value in raw_profile.items()
        if key not in {"validated_parts", "part_overrides", "qualifications"}
    }
    raw_profile["demonstration_recipes"] = {
        part_name: _current_move_insert_demonstration_recipe(
            {
                **learned_values,
                "calibration_id": f"learned-{part_name.lower()}-insertion-1",
                "recording_id": f"insertion-demonstration-{part_name.lower()}-1",
                "demonstration_sha256": "a" * 64,
                "updated_at": "2026-08-20T12:00:00Z",
                "robot": "ur5e",
                "tool_frame": "tool0",
                "destination_location": "assembly_board-v1",
                "part_name": part_name,
                "context_sha256": "b" * 64,
                "place_approach_recording_sha256": "c" * 64,
                "board_calibration_id": "board-calibration-1",
                "board_generation": 12,
                "aruco_to_seated_held_part": {
                    "x": 0.1,
                    "y": 0.2,
                    "z": 0.3,
                    "qx": 0.0,
                    "qy": 0.0,
                    "qz": 0.0,
                    "qw": 1.0,
                },
                "aruco_insertion_axis": {"x": 1.0, "y": 0.0, "z": 0.0},
            },
            recovery_values=recovery_values,
        )
    }
    return raw_profile


def test_move_insert_profile_rejects_legacy_demonstration_recipe() -> None:
    raw_profile = _move_insert_profile_with_current_recipe()
    raw_profile["demonstration_recipes"]["MG"]["recipe_version"] = 1

    resolved = resolve_move_insert_profile(
        {"move_insert": raw_profile},
        "MG",
        require_qualification=False,
    )

    assert resolved["success"] is False
    assert "legacy or unsafe" in resolved["message"]


def test_installed_mg_recipe_migration_preserves_learned_evidence() -> None:
    resource = json.loads(
        (
            ROOT
            / "cais_spade_llm"
            / "initialization"
            / "resources"
            / "robot_ur5e.json"
        ).read_text(encoding="utf-8")
    )["ur5e"]["real"]
    profile = resource["controller"]["parts_tuning"]["move_insert"]
    recipe = profile["demonstration_recipes"]["MG"]

    assert recipe["recipe_version"] == 9
    assert recipe["learning_policy_version"] == 12
    assert recipe["recording_id"] == (
        "insertion-demonstration-1787357919294-bf270210"
    )
    assert recipe["demonstration_sha256"] == (
        "8e6a7c2b4690f0fe91b45a05fab1e91fe538e53484c415e50d1e4b8aa71421a8"
    )
    assert recipe["place_approach_recording_sha256"] == (
        "234f3cc4c6c324bd3e5f43299315f168605d7db9b92508eb561483aa03d12f9c"
    )
    assert recipe["force_depth_profile_sha256"] == (
        "6d7fb806b513f684e51347aaade10df5915752bf8bc7a6bbb624ab072d02305d"
    )
    assert len(recipe["force_depth_profile"]["depth_fraction"]) == 16
    assert recipe["hard_caps"]["insert_max_contact_search_radius_m"] == 0.01
    assert recipe["hard_caps"]["insert_max_disengagement_cycles"] == 6.0
    assert recipe["learning_policy"]["tactile_center_policy"] == (
        "deepest_stable_progress_then_lowest_normalized_lateral_load"
    )
    assert recipe["learning_policy"][
        "disengagement_lateral_clearance_policy"
    ] == "search_boundary_plus_start_position_tolerance"
    assert profile["validated_parts"] == ["MG"]
    qualification = profile["qualifications"]["MG"]
    assert qualification["trial_id"] == (
        "move-insert-1787528562320-508194c6"
    )
    assert qualification["qualification_policy_version"] == 3
    assert qualification["required_confirmed_trials"] == 1
    assert qualification["confirmed_trial_count"] == 1
    assert qualification["confirmed_trial_ids"] == [qualification["trial_id"]]
    assert resolve_move_insert_profile({"move_insert": profile}, "MG")[
        "success"
    ] is True

    expected_policy_sha256, policy_error = move_insert_learning_policy_sha256(
        recipe
    )
    expected_evidence_sha256, evidence_error = (
        move_insert_learning_evidence_sha256(recipe)
    )
    assert policy_error == ""
    assert evidence_error == ""
    assert recipe["learning_policy_sha256"] == expected_policy_sha256
    assert recipe["hard_caps_sha256"] == _canonical_sha256(recipe["hard_caps"])
    assert recipe["learning_evidence_sha256"] == expected_evidence_sha256


def test_move_insert_recipes_keep_six_exact_part_recovery_policies_isolated() -> None:
    recovery_by_part = {
        "SG": {
            "insert_max_contact_search_radius_m": 0.004,
            "insert_max_disengagement_cycles": 2.0,
            "insert_search_peck_retreat_m": 0.0010,
            "insert_search_peck_interval_sec": 0.40,
        },
        "MG": {
            "insert_max_contact_search_radius_m": 0.005,
            "insert_max_disengagement_cycles": 3.0,
            "insert_search_peck_retreat_m": 0.0012,
            "insert_search_peck_interval_sec": 0.50,
        },
        "LG": {
            "insert_max_contact_search_radius_m": 0.006,
            "insert_max_disengagement_cycles": 4.0,
            "insert_search_peck_retreat_m": 0.0014,
            "insert_search_peck_interval_sec": 0.60,
        },
        "SCP": {
            "insert_max_contact_search_radius_m": 0.007,
            "insert_max_disengagement_cycles": 5.0,
            "insert_search_peck_retreat_m": 0.0016,
            "insert_search_peck_interval_sec": 0.70,
        },
        "MCP": {
            "insert_max_contact_search_radius_m": 0.008,
            "insert_max_disengagement_cycles": 6.0,
            "insert_search_peck_retreat_m": 0.0018,
            "insert_search_peck_interval_sec": 0.80,
        },
        "LCP": {
            "insert_max_contact_search_radius_m": 0.009,
            "insert_max_disengagement_cycles": 7.0,
            "insert_search_peck_retreat_m": 0.0020,
            "insert_search_peck_interval_sec": 0.90,
        },
    }
    raw_profile = _move_insert_profile(validated_parts=[])
    raw_profile["demonstration_recipes"] = {
        part_name: _move_insert_profile_with_current_recipe(
            part_name,
            recovery_values=recovery_values,
        )["demonstration_recipes"][part_name]
        for part_name, recovery_values in recovery_by_part.items()
    }

    selected_hashes: dict[str, str] = {}
    for part_name, recovery_values in recovery_by_part.items():
        selected_hash, hash_error = move_insert_profile_sha256(
            raw_profile,
            part_name,
        )
        assert hash_error == ""
        selected_hashes[part_name] = selected_hash
        resolved = resolve_move_insert_profile(
            {"move_insert": raw_profile},
            part_name,
            require_qualification=False,
        )
        assert resolved["success"] is True, resolved
        assert resolved["demonstration_recipe"]["part_name"] == part_name
        for field_name, expected_value in recovery_values.items():
            assert resolved["demonstration_recipe"]["learning_policy"][
                field_name
            ] == pytest.approx(expected_value)
            assert resolved["demonstration_recipe"]["hard_caps"][
                field_name
            ] == pytest.approx(expected_value)

    assert len(set(selected_hashes.values())) == len(recovery_by_part)
    changed_sg = deepcopy(raw_profile)
    changed_sg["demonstration_recipes"]["SG"] = (
        _move_insert_profile_with_current_recipe(
            "SG",
            recovery_values={
                **recovery_by_part["SG"],
                "insert_max_contact_search_radius_m": 0.0035,
            },
        )["demonstration_recipes"]["SG"]
    )
    assert move_insert_profile_sha256(changed_sg, "MG") == (
        selected_hashes["MG"],
        "",
    )


def test_move_insert_recipe_without_advanced_recovery_uses_base_policy() -> None:
    raw_profile = _move_insert_profile_with_current_recipe("SG")
    recipe = raw_profile["demonstration_recipes"]["SG"]
    for field_name in _MOVE_INSERT_MG_TACTILE_POLICY_FIELDS:
        recipe["learning_policy"].pop(field_name, None)
        recipe["hard_caps"].pop(field_name, None)
    recipe["hard_caps_sha256"] = _canonical_sha256(recipe["hard_caps"])
    policy_sha256, policy_error = move_insert_learning_policy_sha256(recipe)
    assert policy_error == ""
    recipe["learning_policy_sha256"] = policy_sha256
    evidence_sha256, evidence_error = move_insert_learning_evidence_sha256(
        recipe
    )
    assert evidence_error == ""
    recipe["learning_evidence_sha256"] = evidence_sha256

    resolved = resolve_move_insert_profile(
        {"move_insert": raw_profile},
        "SG",
        require_qualification=False,
    )

    assert resolved["success"] is True, resolved
    resolved_policy = resolved["demonstration_recipe"]["learning_policy"]
    assert all(
        field_name not in resolved_policy
        for field_name in _MOVE_INSERT_MG_TACTILE_POLICY_FIELDS
    )


def test_move_insert_recipe_rejects_hard_caps_without_advanced_policy() -> None:
    raw_profile = _move_insert_profile_with_current_recipe("SG")
    recipe = raw_profile["demonstration_recipes"]["SG"]
    for field_name in _MOVE_INSERT_MG_TACTILE_POLICY_FIELDS:
        recipe["learning_policy"].pop(field_name, None)
    _policy_sha256, policy_error = move_insert_learning_policy_sha256(recipe)

    resolved = resolve_move_insert_profile(
        {"move_insert": raw_profile},
        "SG",
        require_qualification=False,
    )

    assert "learning_policy fields do not match" in policy_error
    assert resolved["success"] is False
    assert "learning_policy_sha256 does not match" in resolved["message"]


@pytest.mark.parametrize(
    ("field_name", "replacement"),
    [("robot", "xarm6"), ("tool_frame", "xarm6_link6")],
)
def test_move_insert_recipe_rejects_wrong_robot_or_tool_frame(
    field_name: str,
    replacement: str,
) -> None:
    raw_profile = _move_insert_profile_with_current_recipe()
    raw_profile["demonstration_recipes"]["MG"][field_name] = replacement

    resolved = resolve_move_insert_profile(
        {"move_insert": raw_profile},
        "MG",
        require_qualification=False,
    )

    assert resolved["success"] is False
    assert f"MG.{field_name} must be exact" in resolved["message"]


def test_move_insert_profile_rejects_tampered_learning_evidence() -> None:
    raw_profile = _move_insert_profile_with_current_recipe()
    raw_profile["demonstration_recipes"]["MG"][
        "observed_filtered_axial_force_n"
    ] += 0.5

    resolved = resolve_move_insert_profile(
        {"move_insert": raw_profile},
        "MG",
        require_qualification=False,
    )

    assert resolved["success"] is False
    assert "learning_evidence_sha256 does not match" in resolved["message"]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("insert_relief_clear_hysteresis_ratio", 0.7),
        ("insert_max_relief_cycles", 2.0),
    ],
)
def test_move_insert_profile_rejects_changed_relief_policy(
    field: str,
    value: float,
) -> None:
    raw_profile = _move_insert_profile_with_current_recipe()
    raw_profile["demonstration_recipes"]["MG"]["learning_policy"][field] = (
        value
    )

    resolved = resolve_move_insert_profile(
        {"move_insert": raw_profile},
        "MG",
        require_qualification=False,
    )

    assert resolved["success"] is False
    assert "learning_policy_sha256 does not match" in resolved["message"]


def test_move_insert_profile_resolves_exact_sparse_override_and_selected_hash() -> None:
    raw = _move_insert_profile()
    raw["part_overrides"] = {
        "MG": {
            "calibration_id": "mg-insert-1",
            "generation": 1,
            "updated_at": "2026-08-18T12:00:00+00:00",
            "insertion_force_n": 4.0,
        }
    }
    selected_hash, hash_error = move_insert_profile_sha256(raw, "MG")
    assert hash_error == ""
    raw["part_overrides"]["MG"]["profile_sha256"] = selected_hash
    _confirm_move_insert_profile(raw)

    resolved = resolve_move_insert_profile({"move_insert": raw}, "MG")

    assert resolved["success"] is True
    assert resolved["insertion_force_n"] == pytest.approx(4.0)
    assert resolved["spiral_radius_m"] == pytest.approx(0.001)
    assert resolved["profile_sha256"] == selected_hash
    unrelated = json.loads(json.dumps(raw))
    unrelated["part_overrides"]["SG"] = {
        "calibration_id": "sg-insert-1",
        "generation": 1,
        "updated_at": "2026-08-18T12:00:00+00:00",
        "spiral_radius_m": 0.002,
    }
    sg_hash, sg_hash_error = move_insert_profile_sha256(unrelated, "SG")
    assert sg_hash_error == ""
    unrelated["part_overrides"]["SG"]["profile_sha256"] = sg_hash
    assert move_insert_profile_sha256(unrelated, "MG") == (selected_hash, "")
    assert resolve_move_insert_profile({"move_insert": unrelated}, "MG")[
        "success"
    ] is True

    authorization_only = json.loads(json.dumps(raw))
    authorization_only["validated_parts"].append("SG")
    authorization_only["qualifications"]["SG"] = {"future": "authorization"}
    assert move_insert_profile_sha256(authorization_only, "MG") == (
        selected_hash,
        "",
    )


def test_move_insert_profile_requires_supervised_qualification_for_normal_use() -> None:
    raw = _move_insert_profile()

    normal = resolve_move_insert_profile({"move_insert": raw}, "MG")
    trial = resolve_move_insert_profile(
        {"move_insert": raw},
        "MG",
        require_qualification=False,
    )

    assert normal["success"] is False
    assert "Supervised" not in normal["message"]
    assert "supervised move_insert trial" in normal["message"]
    assert trial["success"] is True
    assert trial["qualification"] == {}


def test_move_insert_qualification_policy_or_evidence_change_fails_closed() -> None:
    raw = _move_insert_profile()
    _confirm_move_insert_profile(raw)
    assert resolve_move_insert_profile({"move_insert": raw}, "MG")[
        "success"
    ] is True

    changed_policy = deepcopy(raw)
    changed_policy["qualifications"]["MG"][
        "qualification_policy_version"
    ] = 1
    policy_result = resolve_move_insert_profile(
        {"move_insert": changed_policy},
        "MG",
    )
    assert policy_result["success"] is False
    assert "protected qualification policy" in policy_result["message"]

    changed_evidence = deepcopy(raw)
    changed_evidence["qualifications"]["MG"][
        "confirmed_trial_trace_sha256s"
    ][0] = "9" * 64
    evidence_result = resolve_move_insert_profile(
        {"move_insert": changed_evidence},
        "MG",
    )
    assert evidence_result["success"] is False
    assert "qualification_evidence_sha256" in evidence_result["message"]


@pytest.mark.parametrize(
    ("field_name", "replacement"),
    [("robot", "xarm6"), ("tool_frame", "xarm6_link6")],
)
def test_move_insert_qualification_rejects_wrong_robot_or_tool_frame(
    field_name: str,
    replacement: str,
) -> None:
    raw = _move_insert_profile()
    _confirm_move_insert_profile(raw)
    raw["qualifications"]["MG"][field_name] = replacement

    resolved = resolve_move_insert_profile({"move_insert": raw}, "MG")

    assert resolved["success"] is False
    assert f"MG.{field_name} does not match" in resolved["message"]


def test_move_insert_profile_rejects_malformed_unrelated_override() -> None:
    raw = _move_insert_profile()
    raw["part_overrides"] = {"SG": {"spiral_radius_m": 0.002}}

    resolved = resolve_move_insert_profile({"move_insert": raw}, "MG")

    assert resolved["success"] is False
    assert "SG is missing server-owned metadata" in resolved["message"]


@pytest.mark.parametrize("part_name", [" MG ", "mg", "Mg"])
def test_move_insert_profile_rejects_non_exact_part_identifier(part_name: str) -> None:
    resolved = resolve_move_insert_profile(
        {"move_insert": _move_insert_profile()},
        part_name,
    )

    assert resolved["success"] is False


@pytest.mark.parametrize("part_name", ["SRP", "MRP", "LRP"])
def test_move_insert_profile_blocks_rectangular_parts(part_name: str) -> None:
    resolved = resolve_move_insert_profile(
        {"move_insert": _move_insert_profile()},
        part_name,
    )

    assert resolved["success"] is False
    assert "orientation is measured and validated" in resolved["message"]


def test_incomplete_move_insert_profile_reports_every_missing_field() -> None:
    resolved = resolve_move_insert_profile(
        {"move_insert": {"validated_parts": [], "part_overrides": {}}},
        "MG",
    )

    assert resolved["success"] is False
    assert resolved["missing_fields"] == [
        "calibration_id",
        "pre_insert_offset_m",
        "contact_speed_m_s",
        "contact_force_delta_n",
        "engagement_progress_m",
        "insertion_force_n",
        "spiral_radius_m",
        "spiral_pitch_m",
        "spiral_speed_m_s",
        "spiral_acceleration_m_s2",
        "max_axial_force_n",
        "max_lateral_force_n",
        "max_torque_nm",
        "tilt_tolerance_rad",
        "seated_depth_tolerance_m",
        "settle_time_sec",
    ]


def test_move_insert_profile_requires_positive_seating_dwell() -> None:
    raw = _move_insert_profile(validated_parts=[])
    raw["settle_time_sec"] = 0.0

    resolved = resolve_move_insert_profile(
        {"move_insert": raw},
        "MG",
        require_qualification=False,
    )

    assert resolved["success"] is False
    assert "settle_time_sec" in resolved["message"]


def test_move_insert_timeout_uses_axial_and_archimedean_spiral_path() -> None:
    profile = _move_insert_profile()
    start = {"x": 0.0, "y": 0.0, "z": 0.01}
    target = {"x": 0.0, "y": 0.0, "z": 0.0}
    axis = {"x": 0.0, "y": 0.0, "z": -1.0}

    timeout, error = derive_move_insert_timeout_sec(start, target, axis, profile)

    theta_max = 2.0 * math.pi * 0.001 / 0.0005
    b = 0.0005 / (2.0 * math.pi)
    arc_length = 0.5 * b * (
        theta_max * math.sqrt(1.0 + theta_max**2) + math.asinh(theta_max)
    )
    expected = (
        0.01 / 0.002
        + arc_length / 0.002
        + 2.0 * 0.002 / 0.02
        + 0.10  # force filter establishment
        + 0.10  # contact persistence
        + 0.20  # stall detection
        + 0.20  # engagement persistence
        + 0.20  # seated dwell
        + 0.10  # scheduling margin
    )
    assert error == ""
    assert timeout == pytest.approx(expected)


def test_production_board_registration_placeholder_blocks_move_insert_geometry() -> None:
    geometry_path = (
        Path(__file__).resolve().parents[1]
        / "cais_spade_llm/specification/products/geometry/assembly_board-v1.json"
    )
    document = json.loads(geometry_path.read_text(encoding="utf-8"))
    assert "assembly_board-v1_aruco_to_assembly_board-v1" not in document["gazebo"][
        "assembly_board"
    ]
    placeholder = document["real"]["assembly_board"][
        "assembly_board-v1_aruco_to_assembly_board-v1"
    ]
    assert placeholder == {
        "calibration_id": None,
        "x": None,
        "y": None,
        "z": None,
        "qx": None,
        "qy": None,
        "qz": None,
        "qw": None,
    }
    geometry = ProductProfile.geometry_for_part_from_geometry("MG", document["real"])
    raw_profile = _move_insert_profile()
    _confirm_move_insert_profile(raw_profile)
    profile = resolve_move_insert_profile(
        {"move_insert": raw_profile},
        "MG",
    )
    assert profile["success"] is True

    result = compute_move_insert_geometry(
        part_name="MG",
        product_geometry=geometry,
        assembly_board_v1_aruco={
            "pose": {
                "x": 0.0,
                "y": 0.0,
                "z": 0.0,
                "qx": 0.0,
                "qy": 0.0,
                "qz": 0.0,
                "qw": 1.0,
            }
        },
        held_part_handoff={},
        move_insert_profile=profile,
        move_insert_profile_sha256=profile["profile_sha256"],
    )

    assert result["success"] is False
    assert result.get("missing_held_part_handoff") is not True
    assert "assembly_board-v1_aruco_to_assembly_board-v1" in result["message"]


def test_move_insert_geometry_composes_full_held_part_se3_on_tilted_board() -> None:
    raw_profile = _move_insert_profile()
    _confirm_move_insert_profile(raw_profile)
    profile = resolve_move_insert_profile(
        {"move_insert": raw_profile},
        "MG",
    )
    assert profile["success"] is True
    half_sqrt = math.sqrt(0.5)
    product_geometry = {
        "slot_xy": [0.0, 0.0],
        "part_height_m": 0.02,
        "slot_floor_z_m": 0.0,
        "board_center": {"x": 0.0, "y": 0.0, "z": 0.0},
        "target_reference": {
            "target_point": "inserted_part_origin",
            "surface_role": "assembly_slot",
        },
        "assembly_board-v1_aruco_to_assembly_board-v1": {
            "calibration_id": "marker-board-tilted-1",
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
            "qx": 0.0,
            "qy": half_sqrt,
            "qz": 0.0,
            "qw": half_sqrt,
        },
    }
    handoff = {
        "part_name": "MG",
        "frame_id": "world",
        "tool_frame": "tool0",
        "part_frame": "held_part_origin",
        "world_tool0_pose_at_grasp": {
            "x": 1.0,
            "y": 0.0,
            "z": 0.0,
            "qx": 0.0,
            "qy": 0.0,
            "qz": half_sqrt,
            "qw": half_sqrt,
        },
        "world_held_part_pose_at_grasp": {
            "x": 1.0,
            "y": 1.0,
            "z": 0.0,
            "qx": 0.0,
            "qy": 0.0,
            "qz": 0.0,
            "qw": 1.0,
        },
        "tool0_to_held_part": {
            "x": 1.0,
            "y": 0.0,
            "z": 0.0,
            "qx": 0.0,
            "qy": 0.0,
            "qz": -half_sqrt,
            "qw": half_sqrt,
        },
    }

    result = compute_move_insert_geometry(
        part_name="MG",
        product_geometry=product_geometry,
        assembly_board_v1_aruco={
            "pose": {
                "x": 0.0,
                "y": 0.0,
                "z": 0.0,
                "qx": 0.0,
                "qy": 0.0,
                "qz": 0.0,
                "qw": 1.0,
            }
        },
        held_part_handoff=handoff,
        move_insert_profile=profile,
        move_insert_profile_sha256=profile["profile_sha256"],
    )

    assert result["success"] is True, result
    assert result["insertion_axis_world"] == pytest.approx(
        {"x": -1.0, "y": 0.0, "z": 0.0}, abs=1e-12
    )
    assert {
        axis: result["pre_insert_pose"][axis] - result["insert_pose"][axis]
        for axis in ("x", "y", "z")
    } == pytest.approx({"x": 0.01, "y": 0.0, "z": 0.0}, abs=1e-12)
    assert result["insert_pose"] == pytest.approx(
        {
            "x": 0.01,
            "y": -1.0,
            "z": 0.0,
            "qx": 0.5,
            "qy": 0.5,
            "qz": 0.5,
            "qw": 0.5,
        },
        abs=1e-12,
    )


def test_learned_move_insert_geometry_uses_fresh_aruco_pose_and_handoff() -> None:
    raw_profile = _move_insert_profile(validated_parts=[])
    learned_values = {
        key: value
        for key, value in raw_profile.items()
        if key not in {"validated_parts", "part_overrides", "qualifications"}
    }
    raw_profile["demonstration_recipes"] = {
        "MG": _current_move_insert_demonstration_recipe({
            **learned_values,
            "calibration_id": "learned-mg-insertion-1",
            "recording_id": "insertion-demonstration-mg-1",
            "demonstration_sha256": "a" * 64,
            "updated_at": "2026-08-20T12:00:00Z",
            "robot": "ur5e",
            "tool_frame": "tool0",
            "destination_location": "assembly_board-v1",
            "part_name": "MG",
            "context_sha256": "b" * 64,
            "place_approach_recording_sha256": "c" * 64,
            "board_calibration_id": "board-calibration-1",
            "board_generation": 12,
            "aruco_to_seated_held_part": {
                "x": 0.1,
                "y": 0.2,
                "z": 0.3,
                "qx": 0.0,
                "qy": 0.0,
                "qz": 0.0,
                "qw": 1.0,
            },
            "aruco_insertion_axis": {"x": 1.0, "y": 0.0, "z": 0.0},
        })
    }
    profile = resolve_move_insert_profile(
        {"move_insert": raw_profile},
        "MG",
        require_qualification=False,
    )
    assert profile["success"] is True, profile
    demonstration_recipe = dict(profile["demonstration_recipe"])
    profile["force_depth_profile"] = deepcopy(
        demonstration_recipe["force_depth_profile"]
    )
    profile["hard_caps_sha256"] = demonstration_recipe["hard_caps_sha256"]
    product_geometry = {
        "part_height_m": 0.02,
        "target_reference": {
            "target_point": "inserted_part_origin",
            "surface_role": "assembly_slot",
        },
    }
    missing_handoff = compute_move_insert_geometry(
        part_name="MG",
        product_geometry=product_geometry,
        assembly_board_v1_aruco={
            "pose": {
                "x": 1.0,
                "y": 0.0,
                "z": 0.0,
                "qx": 0.0,
                "qy": 0.0,
                "qz": 0.0,
                "qw": 1.0,
            }
        },
        held_part_handoff={},
        move_insert_profile=profile,
        move_insert_profile_sha256=profile["profile_sha256"],
    )
    assert missing_handoff["missing_held_part_handoff"] is True

    half_sqrt = math.sqrt(0.5)
    fresh_handoff = {
        "part_name": "MG",
        "frame_id": "world",
        "tool_frame": "tool0",
        "part_frame": "held_part_origin",
        "world_tool0_pose_at_grasp": {
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
            "qx": 0.0,
            "qy": 0.0,
            "qz": 0.0,
            "qw": 1.0,
        },
        "world_held_part_pose_at_grasp": {
            "x": 0.05,
            "y": 0.0,
            "z": 0.0,
            "qx": 0.0,
            "qy": 0.0,
            "qz": 0.0,
            "qw": 1.0,
        },
        "tool0_to_held_part": {
            "x": 0.05,
            "y": 0.0,
            "z": 0.0,
            "qx": 0.0,
            "qy": 0.0,
            "qz": 0.0,
            "qw": 1.0,
        },
    }
    result = compute_move_insert_geometry(
        part_name="MG",
        product_geometry=product_geometry,
        assembly_board_v1_aruco={
            "pose": {
                "x": 2.0,
                "y": 3.0,
                "z": 4.0,
                "qx": 0.0,
                "qy": 0.0,
                "qz": half_sqrt,
                "qw": half_sqrt,
            }
        },
        held_part_handoff=fresh_handoff,
        move_insert_profile=profile,
        move_insert_profile_sha256=profile["profile_sha256"],
    )

    assert result["success"] is True, result
    assert result["geometry_source"] == "insertion_demonstration"
    assert result["move_insert_profile"]["force_depth_profile"] == (
        profile["force_depth_profile"]
    )
    assert result["move_insert_profile"]["hard_caps_sha256"] == (
        profile["hard_caps_sha256"]
    )
    assert result["target_origin_pose"] == pytest.approx(
        {
            "x": 1.8,
            "y": 3.1,
            "z": 4.3,
            "qx": 0.0,
            "qy": 0.0,
            "qz": half_sqrt,
            "qw": half_sqrt,
        }
    )
    assert result["insert_pose"] == pytest.approx(
        {
            "x": 1.8,
            "y": 3.05,
            "z": 4.3,
            "qx": 0.0,
            "qy": 0.0,
            "qz": half_sqrt,
            "qw": half_sqrt,
        }
    )
    assert result["insertion_axis_world"] == pytest.approx(
        {"x": 0.0, "y": 1.0, "z": 0.0},
        abs=1e-12,
    )


def test_learned_recipe_does_not_replace_place_approach_nominal_poses() -> None:
    controller = object.__new__(GazeboPickPlaceController)
    controller.execution_mode = "physical"
    controller.robot_name = "ur5e"
    controller.wait_for_services = lambda: True
    controller.pick_tcp_z_bias_min_m = 0.003
    controller.pick_tcp_z_bias_max_m = 0.02
    controller.place_surface_gap_m = -0.01
    controller.insertion_depth_m = 0.0025
    controller._get_ee_tcp_world_z_offset = lambda: 0.0

    nominal_target_z = 1.2529777277557472
    learned_pre_insert_z = 1.3236119619414102
    learned_pre_insert_offset_m = 0.022948322000541744
    learned_insert_z = learned_pre_insert_z - learned_pre_insert_offset_m
    product_geometry = {
        "slot_xy": [0.0, 0.08],
        "part_height_m": 0.02,
        "model_name": "gear_medium",
        "slot_floor_z_m": 1.025,
        "board_center": {"x": 0.0, "y": 0.0, "z": 1.02},
        "target_reference": {
            "target_point": "inserted_part_origin",
            "surface_role": "assembly_slot",
        },
        "assembly_board-v1_aruco_to_assembly_board-v1": {
            "calibration_id": "marker-board-nominal-1",
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
            "qx": 0.0,
            "qy": 0.0,
            "qz": 0.0,
            "qw": 1.0,
        },
    }
    immutable_grasp_pose = {
        "x": 0.0,
        "y": 0.0,
        "z": 0.0,
        "qx": 0.0,
        "qy": 0.0,
        "qz": 0.0,
        "qw": 1.0,
    }
    half_sqrt = math.sqrt(0.5)
    pick_ctx = {
        "part_name": "MG",
        "model_name": "gear_medium",
        "origin_resource_location": "prusa-mk4-2",
        "origin_pose": deepcopy(immutable_grasp_pose),
        "tx": 0.0,
        "ty": 0.0,
        "tz": 0.0,
        "pick_tcp_z": nominal_target_z - 1.035,
        "tcp_offset_z": 0.0,
        "resolved_cartesian_positions": {
            "descend": {
                **immutable_grasp_pose,
                "qz": half_sqrt,
                "qw": half_sqrt,
            }
        },
        "held_part_handoff": {
            "part_name": "MG",
            "model_name": "gear_medium",
            "origin_resource_location": "prusa-mk4-2",
            "frame_id": "world",
            "tool_frame": "tool0",
            "part_frame": "held_part_origin",
            "world_tool0_pose_at_grasp": deepcopy(immutable_grasp_pose),
            "world_held_part_pose_at_grasp": deepcopy(immutable_grasp_pose),
            "tool0_to_held_part": {
                "x": 0.0,
                "y": 0.0,
                "z": 0.0,
                "qx": 0.0,
                "qy": 0.0,
                "qz": 0.0,
                "qw": 1.0,
            },
        },
    }
    frozen_board = {
        "destination_location": "assembly_board-v1",
        "camera_role": "ur5e",
        "frame_id": "world",
        "generation": 14,
        "captured_at": time.time(),
        "calibration_id": "hand-eye-nominal-1",
        "pose": {
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
            "qx": 0.0,
            "qy": 0.0,
            "qz": 0.0,
            "qw": 1.0,
        },
    }

    shared_profile = _move_insert_profile(validated_parts=[])
    shared_profile["pre_insert_offset_m"] = learned_pre_insert_offset_m
    controller.controller_config = {
        "parts_tuning": {"move_insert": deepcopy(shared_profile)}
    }
    shared_only = controller.compute_place_targets(
        pick_ctx=deepcopy(pick_ctx),
        product_geometry=deepcopy(product_geometry),
        part_name="MG",
        destination_location="assembly_board-v1",
        assembly_board_v1_aruco=deepcopy(frozen_board),
    )

    learned_profile = deepcopy(shared_profile)
    learned_values = {
        key: value
        for key, value in learned_profile.items()
        if key not in {"validated_parts", "part_overrides", "qualifications"}
    }
    learned_profile["demonstration_recipes"] = {
        "MG": _current_move_insert_demonstration_recipe({
            **learned_values,
            "calibration_id": "learned-mg-nominal-1",
            "recording_id": "insertion-demonstration-nominal-1",
            "demonstration_sha256": "a" * 64,
            "updated_at": "2026-08-20T12:00:00Z",
            "robot": "ur5e",
            "tool_frame": "tool0",
            "destination_location": "assembly_board-v1",
            "part_name": "MG",
            "context_sha256": "b" * 64,
            "place_approach_recording_sha256": "c" * 64,
            "board_calibration_id": "hand-eye-nominal-1",
            "board_generation": 14,
            "aruco_to_seated_held_part": {
                "x": 0.0,
                "y": 0.08,
                "z": learned_insert_z,
                "qx": 0.0,
                "qy": 0.0,
                "qz": 0.0,
                "qw": 1.0,
            },
            "aruco_insertion_axis": {"x": 0.0, "y": 0.0, "z": -1.0},
        })
    }
    controller.controller_config = {
        "parts_tuning": {"move_insert": learned_profile}
    }
    learned = controller.compute_place_targets(
        pick_ctx=deepcopy(pick_ctx),
        product_geometry=deepcopy(product_geometry),
        part_name="MG",
        destination_location="assembly_board-v1",
        assembly_board_v1_aruco=deepcopy(frozen_board),
    )

    assert shared_only["success"] is True, shared_only
    assert learned["success"] is True, learned
    assert learned["approach_pose"] == pytest.approx(shared_only["approach_pose"])
    assert learned["target_pose"] == pytest.approx(shared_only["target_pose"])
    assert learned["approach_pose"] == pytest.approx(
        {
            "x": 0.0,
            "y": 0.08,
            "z": 1.3029777277557473,
            "qx": 0.0,
            "qy": 0.0,
            "qz": 0.0,
            "qw": 1.0,
        }
    )
    assert learned["target_pose"] == pytest.approx(
        {
            "x": 0.0,
            "y": 0.08,
            "z": nominal_target_z,
            "qx": 0.0,
            "qy": 0.0,
            "qz": 0.0,
            "qw": 1.0,
        }
    )
    assert learned["pre_insert_pose"]["z"] == pytest.approx(learned_pre_insert_z)
    assert learned["insert_pose"]["z"] == pytest.approx(learned_insert_z)
    assert learned["target_pose"] != pytest.approx(learned["pre_insert_pose"])


def test_physical_ur5e_place_targets_compose_board_pose_and_stop_pre_insert() -> None:
    controller = object.__new__(GazeboPickPlaceController)
    controller.execution_mode = "physical"
    controller.robot_name = "ur5e"
    controller.controller_config = {"parts_tuning": {}}
    controller.wait_for_services = lambda: True
    controller.pick_tcp_z_bias_min_m = 0.003
    controller.pick_tcp_z_bias_max_m = 0.02
    controller.place_surface_gap_m = -0.01
    controller.insertion_depth_m = 0.0025
    controller._get_ee_tcp_world_z_offset = lambda: 0.0
    product_geometry = {
        "slot_xy": [0.0, 0.08],
        "part_height_m": 0.02,
        "model_name": "gear_medium",
        "slot_floor_z_m": 1.025,
        "board_center": {"x": 0.0, "y": 0.0, "z": 1.02},
        "target_reference": {
            "target_point": "inserted_part_origin",
            "surface_role": "assembly_slot",
        },
        "assembly_board-v1_aruco_to_assembly_board-v1": {
            "calibration_id": "marker-board-1",
            "x": 0.1,
            "y": 0.2,
            "z": 0.3,
            "qx": 0.0,
            "qy": 0.0,
            "qz": 0.0,
            "qw": 1.0,
        },
    }
    raw_profile = _move_insert_profile()
    _confirm_move_insert_profile(raw_profile)
    qualification = raw_profile["qualifications"]["MG"]
    qualification["board_calibration_id"] = "hand-eye-1"
    qualification["board_geometry_sha256"] = _canonical_sha256(product_geometry)
    _refresh_move_insert_qualification_evidence(raw_profile)
    profile = resolve_move_insert_profile(
        {"move_insert": raw_profile},
        "MG",
    )
    assert profile["success"] is True
    controller.controller_config = {
        "parts_tuning": {"move_insert": raw_profile}
    }
    product_geometry["move_insert_profile"] = profile
    product_geometry["move_insert_profile_sha256"] = profile["profile_sha256"]
    product_geometry["move_insert_hard_caps"] = {
        "insert_max_travel_m": 0.05,
        "insert_start_position_tolerance_m": 0.003,
        "insert_start_orientation_tolerance_rad": 0.034906585,
        "insert_max_timeout_sec": 30.0,
    }
    product_geometry["move_insert_hard_caps_sha256"] = _canonical_sha256(
        product_geometry["move_insert_hard_caps"]
    )
    pick_ctx = {
        "part_name": "MG",
        "origin_resource_location": "prusa-mk4-2",
        "origin_pose": {"x": 0.0, "y": 0.0, "z": 0.0},
        "tx": 0.0,
        "ty": 0.0,
        "tz": 0.0,
        "pick_tcp_z": 0.3,
        "tcp_offset_z": 0.0,
        "resolved_cartesian_positions": {
            "descend": {
                "x": 0.01,
                "y": 0.02,
                "z": 0.3,
                "qx": 0.0,
                "qy": 0.0,
                "qz": 0.0,
                "qw": 1.0,
            }
        },
        "held_part_handoff": {
            "part_name": "MG",
            "model_name": "gear_medium",
            "origin_resource_location": "prusa-mk4-2",
            "frame_id": "world",
            "tool_frame": "tool0",
            "part_frame": "held_part_origin",
            "world_tool0_pose_at_grasp": {
                "x": 0.01,
                "y": 0.02,
                "z": 0.3,
                "qx": 0.0,
                "qy": 0.0,
                "qz": 0.0,
                "qw": 1.0,
            },
            "world_held_part_pose_at_grasp": {
                "x": 0.0,
                "y": 0.0,
                "z": 0.0,
                "qx": 0.0,
                "qy": 0.0,
                "qz": 0.0,
                "qw": 1.0,
            },
            "tool0_to_held_part": {
                "x": -0.01,
                "y": -0.02,
                "z": -0.3,
                "qx": 0.0,
                "qy": 0.0,
                "qz": 0.0,
                "qw": 1.0,
            },
        },
    }
    frozen_board = {
        "destination_location": "assembly_board-v1",
        "camera_role": "ur5e",
        "frame_id": "world",
        "generation": 7,
        "captured_at": time.time(),
        "calibration_id": "hand-eye-1",
        "pose": {
            "x": 1.0,
            "y": 2.0,
            "z": 3.0,
            "qx": 0.0,
            "qy": 0.0,
            "qz": 0.0,
            "qw": 1.0,
        },
    }

    result = controller.compute_place_targets(
        pick_ctx=pick_ctx,
        product_geometry=product_geometry,
        part_name="MG",
        destination_location="assembly_board-v1",
        assembly_board_v1_aruco=frozen_board,
    )

    assert result["success"] is True, result
    assert result["assembly_board_v1_pose"] == pytest.approx(
        {"x": 1.1, "y": 2.2, "z": 3.3, "qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0}
    )
    assert result["insert_pose"] == pytest.approx(
        {"x": 1.11, "y": 2.30, "z": 3.615, "qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0}
    )
    assert result["pre_insert_pose"] == pytest.approx(
        {"x": 1.11, "y": 2.30, "z": 3.625, "qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0}
    )
    assert result["approach_pose"] == pytest.approx(
        {"x": 0.0, "y": 0.08, "z": 1.385, "qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0}
    )
    assert result["target_pose"] == pytest.approx(
        {"x": 0.0, "y": 0.08, "z": 1.335, "qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0}
    )
    assert result["target_pose"] != pytest.approx(result["pre_insert_pose"])
    assert result["insertion_axis_world"] == pytest.approx(
        {"x": 0.0, "y": 0.0, "z": -1.0}
    )
    assert result["move_insert_mode"] == "force_limited"
    assert result["move_insert_profile"]["qualification"] == qualification
    assert result["move_insert_hard_caps"] == product_geometry[
        "move_insert_hard_caps"
    ]
    assert result["move_insert_hard_caps_sha256"] == product_geometry[
        "move_insert_hard_caps_sha256"
    ]

    changed_camera_calibration = deepcopy(frozen_board)
    changed_camera_calibration["calibration_id"] = "hand-eye-2"
    changed_camera_result = controller.compute_place_targets(
        pick_ctx=pick_ctx,
        product_geometry=product_geometry,
        part_name="MG",
        destination_location="assembly_board-v1",
        assembly_board_v1_aruco=changed_camera_calibration,
    )
    assert changed_camera_result["success"] is True
    assert changed_camera_result["move_insert_mode"] == "force_limited_trial"
    assert changed_camera_result["move_insert_profile"]["qualification"] == {}

    changed_calibration = deepcopy(product_geometry)
    changed_calibration["assembly_board-v1_aruco_to_assembly_board-v1"][
        "calibration_id"
    ] = "marker-board-2"
    changed_calibration_result = controller.compute_place_targets(
        pick_ctx=pick_ctx,
        product_geometry=changed_calibration,
        part_name="MG",
        destination_location="assembly_board-v1",
        assembly_board_v1_aruco=frozen_board,
    )
    assert changed_calibration_result["success"] is True
    assert changed_calibration_result["move_insert_mode"] == "force_limited_trial"
    assert changed_calibration_result["move_insert_profile"]["qualification"] == {}

    changed_slot = deepcopy(product_geometry)
    changed_slot["slot_xy"] = [0.001, 0.08]
    changed_slot_result = controller.compute_place_targets(
        pick_ctx=pick_ctx,
        product_geometry=changed_slot,
        part_name="MG",
        destination_location="assembly_board-v1",
        assembly_board_v1_aruco=frozen_board,
    )
    assert changed_slot_result["success"] is True
    assert changed_slot_result["move_insert_mode"] == "force_limited_trial"
    assert changed_slot_result["move_insert_profile"]["qualification"] == {}

    forged_profile = deepcopy(product_geometry)
    forged_profile["move_insert_profile"]["insertion_force_n"] = 1.0
    forged_profile_result = controller.compute_place_targets(
        pick_ctx=pick_ctx,
        product_geometry=forged_profile,
        part_name="MG",
        destination_location="assembly_board-v1",
        assembly_board_v1_aruco=frozen_board,
    )
    assert forged_profile_result["success"] is False
    assert "protected controller profile" in forged_profile_result["message"]

    unqualified_profile = deepcopy(raw_profile)
    unqualified_profile["qualifications"] = {}
    controller.controller_config = {
        "parts_tuning": {"move_insert": unqualified_profile}
    }
    forged_qualification_result = controller.compute_place_targets(
        pick_ctx=pick_ctx,
        product_geometry=product_geometry,
        part_name="MG",
        destination_location="assembly_board-v1",
        assembly_board_v1_aruco=frozen_board,
    )
    assert forged_qualification_result["success"] is True
    assert forged_qualification_result["move_insert_mode"] == "force_limited_trial"
    assert forged_qualification_result["move_insert_profile"]["qualification"] == {}


def _physical_ur5e_repeat_place_target_case() -> tuple[
    GazeboPickPlaceController,
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    controller = object.__new__(GazeboPickPlaceController)
    controller.execution_mode = "physical"
    controller.robot_name = "ur5e"
    controller.wait_for_services = lambda: True
    controller.pick_tcp_z_bias_min_m = 0.003
    controller.pick_tcp_z_bias_max_m = 0.02
    controller.place_surface_gap_m = -0.01
    controller.insertion_depth_m = 0.0025
    controller._get_ee_tcp_world_z_offset = lambda: 0.0
    product_geometry = {
        "slot_xy": [0.1, 0.2],
        "part_height_m": 0.02,
        "model_name": "gear_medium",
        "slot_floor_z_m": 0.0,
        "board_center": {"x": 0.0, "y": 0.0, "z": 0.0},
        "target_reference": {
            "target_point": "inserted_part_origin",
            "surface_role": "assembly_slot",
        },
        "assembly_board-v1_aruco_to_assembly_board-v1": {
            "calibration_id": "marker-board-repeat-1",
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
            "qx": 0.0,
            "qy": 0.0,
            "qz": 0.0,
            "qw": 1.0,
        },
    }
    raw_profile = _move_insert_profile()
    _confirm_move_insert_profile(raw_profile)
    qualification = raw_profile["qualifications"]["MG"]
    qualification["board_calibration_id"] = "hand-eye-repeat-1"
    qualification["board_geometry_sha256"] = _canonical_sha256(product_geometry)
    _refresh_move_insert_qualification_evidence(raw_profile)
    profile = resolve_move_insert_profile(
        {"move_insert": raw_profile},
        "MG",
    )
    assert profile["success"] is True
    controller.controller_config = {
        "parts_tuning": {"move_insert": raw_profile}
    }
    product_geometry["move_insert_profile"] = profile
    product_geometry["move_insert_profile_sha256"] = profile["profile_sha256"]
    grasp_tool_pose = {
        "x": 0.4,
        "y": 0.2,
        "z": 0.5,
        "qx": 0.0,
        "qy": 0.0,
        "qz": 0.0,
        "qw": 1.0,
    }
    pick_ctx = {
        "part_name": "MG",
        "model_name": "gear_medium",
        "origin_resource_location": "prusa-mk4-2",
        "origin_pose": {
            "x": 0.4,
            "y": 0.2,
            "z": 0.4,
            "qx": 0.0,
            "qy": 0.0,
            "qz": 0.0,
            "qw": 1.0,
        },
        "tx": 0.4,
        "ty": 0.2,
        "tz": 0.4,
        "pick_tcp_z": 0.5,
        "tcp_offset_z": 0.0,
        "resolved_cartesian_positions": {"descend": deepcopy(grasp_tool_pose)},
        "held_part_handoff": {
            "part_name": "MG",
            "model_name": "gear_medium",
            "origin_resource_location": "prusa-mk4-2",
            "frame_id": "world",
            "tool_frame": "tool0",
            "part_frame": "held_part_origin",
            "world_tool0_pose_at_grasp": deepcopy(grasp_tool_pose),
            "world_held_part_pose_at_grasp": {
                "x": 0.4,
                "y": 0.2,
                "z": 0.4,
                "qx": 0.0,
                "qy": 0.0,
                "qz": 0.0,
                "qw": 1.0,
            },
            "tool0_to_held_part": {
                "x": 0.0,
                "y": 0.0,
                "z": -0.1,
                "qx": 0.0,
                "qy": 0.0,
                "qz": 0.0,
                "qw": 1.0,
            },
        },
    }
    frozen_board = {
        "destination_location": "assembly_board-v1",
        "camera_role": "ur5e",
        "frame_id": "world",
        "generation": 7,
        "captured_at": time.time(),
        "calibration_id": "hand-eye-repeat-1",
        "pose": {
            "x": 1.0,
            "y": 2.0,
            "z": 3.0,
            "qx": 0.0,
            "qy": 0.0,
            "qz": 0.0,
            "qw": 1.0,
        },
    }
    return controller, pick_ctx, product_geometry, frozen_board


def _compute_physical_ur5e_repeat_place_target(
    controller: GazeboPickPlaceController,
    pick_ctx: dict[str, Any],
    product_geometry: dict[str, Any],
    frozen_board: dict[str, Any],
    *,
    z_adjustment_m: float = 0.0,
) -> dict[str, Any]:
    return controller.compute_place_targets(
        pick_ctx=pick_ctx,
        product_geometry=product_geometry,
        part_name="MG",
        z_adjustment_m=z_adjustment_m,
        destination_location="assembly_board-v1",
        assembly_board_v1_aruco=frozen_board,
    )


def test_physical_ur5e_repeat_place_targets_use_immutable_handoff() -> None:
    controller, pick_ctx, product_geometry, frozen_board = (
        _physical_ur5e_repeat_place_target_case()
    )
    baseline = _compute_physical_ur5e_repeat_place_target(
        controller,
        pick_ctx,
        product_geometry,
        frozen_board,
    )
    first = _compute_physical_ur5e_repeat_place_target(
        controller,
        pick_ctx,
        product_geometry,
        frozen_board,
        z_adjustment_m=0.0511,
    )
    repeated_pick_ctx = deepcopy(pick_ctx)
    repeated_pick_ctx["resolved_cartesian_positions"] = {
        "descend": deepcopy(first["target_pose"])
    }

    second = _compute_physical_ur5e_repeat_place_target(
        controller,
        repeated_pick_ctx,
        product_geometry,
        frozen_board,
        z_adjustment_m=0.0511,
    )
    third_pick_ctx = deepcopy(repeated_pick_ctx)
    third_pick_ctx["resolved_cartesian_positions"] = {
        "descend": deepcopy(second["target_pose"])
    }
    third = _compute_physical_ur5e_repeat_place_target(
        controller,
        third_pick_ctx,
        product_geometry,
        frozen_board,
        z_adjustment_m=0.0511,
    )

    assert baseline["success"] is True, baseline
    assert first["success"] is True, first
    assert second["success"] is True, second
    assert third["success"] is True, third
    for pose_name in ("approach_pose", "target_pose", "pre_insert_pose", "insert_pose"):
        assert second[pose_name] == pytest.approx(first[pose_name])
        assert third[pose_name] == pytest.approx(first[pose_name])
    assert first["insert_pose"]["z"] - baseline["insert_pose"]["z"] == pytest.approx(
        0.0511
    )
    assert repeated_pick_ctx["held_part_handoff"] == pick_ctx["held_part_handoff"]


def test_physical_ur5e_place_targets_pass_through_move_insert_hard_caps() -> None:
    controller, pick_ctx, product_geometry, frozen_board = (
        _physical_ur5e_repeat_place_target_case()
    )
    without_caps = _compute_physical_ur5e_repeat_place_target(
        controller,
        pick_ctx,
        product_geometry,
        frozen_board,
    )
    hard_caps = {
        "insert_max_travel_m": 0.05,
        "insert_start_position_tolerance_m": 0.003,
        "insert_start_orientation_tolerance_rad": 0.04,
        "insert_max_timeout_sec": "runtime-validates-this-value",
    }
    geometry_with_caps = deepcopy(product_geometry)
    geometry_with_caps["move_insert_hard_caps"] = hard_caps

    with_caps = _compute_physical_ur5e_repeat_place_target(
        controller,
        deepcopy(pick_ctx),
        geometry_with_caps,
        deepcopy(frozen_board),
    )

    assert without_caps["success"] is True, without_caps
    assert "move_insert_hard_caps" not in without_caps
    assert with_caps["success"] is True, with_caps
    assert with_caps["move_insert_hard_caps"] == hard_caps
    assert with_caps["move_insert_hard_caps"] is not hard_caps
    hard_caps["insert_max_travel_m"] = 99.0
    assert with_caps["move_insert_hard_caps"]["insert_max_travel_m"] == pytest.approx(
        0.05
    )


def test_physical_ur5e_repeat_place_targets_follow_changed_board_pose() -> None:
    controller, pick_ctx, product_geometry, frozen_board = (
        _physical_ur5e_repeat_place_target_case()
    )
    initial = _compute_physical_ur5e_repeat_place_target(
        controller,
        pick_ctx,
        product_geometry,
        frozen_board,
    )
    repeated_pick_ctx = deepcopy(pick_ctx)
    repeated_pick_ctx["resolved_cartesian_positions"] = {
        "descend": deepcopy(initial["target_pose"])
    }
    half_sqrt = math.sqrt(0.5)
    changed_board = deepcopy(frozen_board)
    changed_board["generation"] = 8
    changed_board["pose"] = {
        "x": 1.2,
        "y": 1.9,
        "z": 3.05,
        "qx": 0.0,
        "qy": 0.0,
        "qz": half_sqrt,
        "qw": half_sqrt,
    }

    changed = _compute_physical_ur5e_repeat_place_target(
        controller,
        repeated_pick_ctx,
        product_geometry,
        changed_board,
    )
    repeated_pick_ctx["resolved_cartesian_positions"] = {
        "descend": deepcopy(changed["target_pose"])
    }
    changed_again = _compute_physical_ur5e_repeat_place_target(
        controller,
        repeated_pick_ctx,
        product_geometry,
        changed_board,
    )

    assert initial["success"] is True, initial
    assert changed["success"] is True, changed
    assert changed_again["success"] is True, changed_again
    changed_target_origin_pose = {
        field: changed["target_origin_pose"][field]
        for field in ("x", "y", "z", "qx", "qy", "qz", "qw")
    }
    assert changed_target_origin_pose == pytest.approx(
        {
            "x": 1.0,
            "y": 2.0,
            "z": 3.06,
            "qx": 0.0,
            "qy": 0.0,
            "qz": half_sqrt,
            "qw": half_sqrt,
        }
    )
    assert changed["approach_pose"] == pytest.approx(initial["approach_pose"])
    assert changed["target_pose"] == pytest.approx(initial["target_pose"])
    assert changed["pre_insert_pose"] != pytest.approx(initial["pre_insert_pose"])
    assert changed_again["target_pose"] == pytest.approx(changed["target_pose"])
    assert changed_again["insert_pose"] == pytest.approx(changed["insert_pose"])


@pytest.mark.parametrize(
    ("malformed_field", "message_fragment"),
    [
        ("part_name", "held-part SE(3) provenance is invalid"),
        ("tool0_to_held_part", "does not reconstruct its frozen pick poses"),
    ],
)
def test_physical_ur5e_repeat_place_targets_reject_malformed_immutable_handoff(
    malformed_field: str,
    message_fragment: str,
) -> None:
    controller, pick_ctx, product_geometry, frozen_board = (
        _physical_ur5e_repeat_place_target_case()
    )
    initial = _compute_physical_ur5e_repeat_place_target(
        controller,
        pick_ctx,
        product_geometry,
        frozen_board,
    )
    pick_ctx["resolved_cartesian_positions"] = {
        "descend": deepcopy(initial["target_pose"])
    }
    handoff = pick_ctx["held_part_handoff"]
    if malformed_field == "part_name":
        handoff["part_name"] = "SG"
    else:
        handoff["tool0_to_held_part"]["x"] = 0.002

    result = _compute_physical_ur5e_repeat_place_target(
        controller,
        pick_ctx,
        product_geometry,
        frozen_board,
    )

    assert result["success"] is False
    assert message_fragment in result["message"]


def test_physical_ur5e_held_place_approach_succeeds_with_zero_move_insert_recipe() -> None:
    controller = object.__new__(GazeboPickPlaceController)
    controller.execution_mode = "physical"
    controller.robot_name = "ur5e"
    controller.controller_config = {"parts_tuning": {"move_insert": {}}}
    controller.wait_for_services = lambda: True
    controller.pick_tcp_z_bias_min_m = 0.003
    controller.pick_tcp_z_bias_max_m = 0.02
    controller.place_surface_gap_m = -0.01
    controller.insertion_depth_m = 0.0025
    controller._get_ee_tcp_world_z_offset = lambda: 0.0
    held_pose = {
        "x": 0.3,
        "y": 0.2,
        "z": 1.2,
        "qx": 0.0,
        "qy": 0.0,
        "qz": 0.0,
        "qw": 1.0,
    }

    result = controller.compute_place_targets(
        pick_ctx={
            "part_name": "MG",
            "model_name": "gear_medium",
            "origin_resource_location": "prusa-mk4-2",
            "pick_tcp_z": 1.2,
            "tz": 1.0,
            "tcp_offset_z": 0.0,
            "resolved_cartesian_positions": {"descend": deepcopy(held_pose)},
            "held_part_handoff": {
                "world_tool0_pose_at_grasp": deepcopy(held_pose),
            },
        },
        product_geometry={
            "slot_xy": [0.0, 0.08],
            "part_height_m": 0.02,
            "model_name": "gear_medium",
            "slot_floor_z_m": 1.025,
            "board_center": {"x": 0.0, "y": 0.0, "z": 1.02},
            "target_reference": {
                "target_point": "inserted_part_origin",
                "surface_role": "assembly_slot",
            },
            "assembly_board-v1_aruco_to_assembly_board-v1": {
                "calibration_id": "marker-board-zero-recipe-1",
                "x": 0.0,
                "y": 0.0,
                "z": 0.0,
                "qx": 0.0,
                "qy": 0.0,
                "qz": 0.0,
                "qw": 1.0,
            },
        },
        part_name="MG",
        destination_location="assembly_board-v1",
        assembly_board_v1_aruco={
            "destination_location": "assembly_board-v1",
            "camera_role": "ur5e",
            "frame_id": "world",
            "generation": 7,
            "captured_at": time.time(),
            "calibration_id": "hand-eye-zero-recipe-1",
            "pose": {
                "x": 0.1,
                "y": 0.2,
                "z": 1.0,
                "qx": 0.0,
                "qy": 0.0,
                "qz": 0.0,
                "qw": 1.0,
            },
        },
    )

    assert result["success"] is True, result
    assert result["move_insert_mode"] == ""
    assert "move_insert_profile" not in result
    assert "move_insert_profile_sha256" not in result
    assert "move_insert_timeout_sec" not in result
    assert result["target_pose"] == pytest.approx(result["pre_insert_pose"])
    assert result["pre_insert_pose"] == pytest.approx(result["insert_pose"])
    assert result["approach_pose"]["z"] - result["target_pose"]["z"] == pytest.approx(
        0.05
    )


def test_physical_ur5e_independent_place_approach_does_not_authorize_move_insert() -> None:
    controller = object.__new__(GazeboPickPlaceController)
    controller.execution_mode = "physical"
    controller.robot_name = "ur5e"
    controller.controller_config = {"parts_tuning": {}}
    controller.wait_for_services = lambda: True
    controller.pick_tcp_z_bias_min_m = 0.003
    controller.pick_tcp_z_bias_max_m = 0.02
    controller.place_surface_gap_m = -0.01
    controller.insertion_depth_m = 0.0025
    controller._get_ee_tcp_world_z_offset = lambda: 0.0

    result = controller.compute_place_targets(
        pick_ctx={},
        product_geometry={
            "slot_xy": [0.0, 0.08],
            "part_height_m": 0.02,
            "model_name": "gear_medium",
            "slot_floor_z_m": 1.025,
            "board_center": {"x": 0.0, "y": 0.0, "z": 1.02},
            "target_reference": {
                "target_point": "inserted_part_origin",
                "surface_role": "assembly_slot",
            },
        },
        part_name="MG",
        destination_location="assembly_board-v1",
        assembly_board_v1_aruco={
            "destination_location": "assembly_board-v1",
            "camera_role": "ur5e",
            "frame_id": "world",
            "generation": 7,
            "captured_at": time.time(),
            "calibration_id": "hand-eye-1",
            "pose": {
                "x": 1.0,
                "y": 2.0,
                "z": 3.0,
                "qx": 0.0,
                "qy": 0.0,
                "qz": 0.0,
                "qw": 1.0,
            },
        },
    )

    assert result["success"] is True, result
    assert result["move_insert_mode"] == ""
    assert "move_insert_profile" not in result
    assert result["pre_insert_pose"] == result["insert_pose"]


@pytest.mark.parametrize(
    "manual_marker",
    ("operator_confirmed_held_part", "manual_function_execution"),
)
def test_manual_place_approach_does_not_require_move_insert_profile(
    manual_marker: str,
) -> None:
    controller = object.__new__(GazeboPickPlaceController)
    controller.execution_mode = "physical"
    controller.robot_name = "ur5e"
    controller.controller_config = {"parts_tuning": {"move_insert": {}}}
    controller.wait_for_services = lambda: True
    controller.pick_tcp_z_bias_min_m = 0.003
    controller.pick_tcp_z_bias_max_m = 0.02
    controller.place_surface_gap_m = -0.01
    controller.insertion_depth_m = 0.0025
    controller._get_ee_tcp_world_z_offset = lambda: 0.0
    held_pose = {
        "x": 0.4,
        "y": 0.3,
        "z": 1.2,
        "qx": 0.0,
        "qy": 0.0,
        "qz": 0.0,
        "qw": 1.0,
    }

    result = controller.compute_place_targets(
        pick_ctx={
            "part_name": "MG",
            "model_name": "gear_medium",
            manual_marker: True,
            "pick_tcp_z": 1.2,
            "tz": 1.0,
            "tcp_offset_z": 0.0,
            "resolved_cartesian_positions": {"descend": held_pose},
            "held_part_handoff": {
                "world_tool0_pose_at_grasp": held_pose,
            },
        },
        product_geometry={
            "slot_xy": [0.0, 0.08],
            "part_height_m": 0.02,
            "model_name": "gear_medium",
            "slot_floor_z_m": 1.025,
            "board_center": {"x": 0.0, "y": 0.0, "z": 1.02},
            "target_reference": {
                "target_point": "inserted_part_origin",
                "surface_role": "assembly_slot",
            },
            "assembly_board-v1_aruco_to_assembly_board-v1": {
                "calibration_id": None,
                "x": None,
                "y": None,
                "z": None,
                "qx": None,
                "qy": None,
                "qz": None,
                "qw": None,
            },
        },
        part_name="MG",
        destination_location="assembly_board-v1",
        assembly_board_v1_aruco={
            "destination_location": "assembly_board-v1",
            "camera_role": "ur5e",
            "frame_id": "world",
            "generation": 7,
            "captured_at": time.time(),
            "calibration_id": "hand-eye-1",
            "pose": {
                "x": 1.0,
                "y": 2.0,
                "z": 3.0,
                "qx": 0.0,
                "qy": 0.0,
                "qz": 0.0,
                "qw": 1.0,
            },
        },
    )

    assert result["success"] is True, result
    assert result["move_insert_mode"] == ""
    assert "move_insert_profile" not in result
    assert result["pre_insert_pose"] == result["insert_pose"]


def test_physical_xarm6_held_part_assembly_slot_requires_ur5e_move_insert() -> None:
    controller = object.__new__(GazeboPickPlaceController)
    controller.execution_mode = "physical"
    controller.robot_name = "xarm6"
    controller.wait_for_services = lambda: True

    result = controller.compute_place_targets(
        pick_ctx={"part_name": "MG"},
        product_geometry={
            "slot_xy": [0.0, 0.08],
            "part_height_m": 0.02,
            "model_name": "gear_medium",
            "slot_floor_z_m": 1.025,
            "board_center": {"x": 0.0, "y": 0.0, "z": 1.02},
            "target_reference": {
                "target_point": "inserted_part_origin",
                "surface_role": "assembly_slot",
            },
        },
        part_name="MG",
        destination_location="assembly_board-v1",
        assembly_board_v1_aruco={
            "destination_location": "assembly_board-v1",
            "camera_role": "xarm6",
            "frame_id": "world",
            "generation": 7,
            "captured_at": time.time(),
            "pose": {
                "x": 1.0,
                "y": 2.0,
                "z": 3.0,
                "qx": 0.0,
                "qy": 0.0,
                "qz": 0.0,
                "qw": 1.0,
            },
        },
    )

    assert result == {
        "success": False,
        "message": _PHYSICAL_XARM6_ASSEMBLY_SLOT_INSERT_ERROR,
    }


class _InsertGoal:
    pass


class _Vector3:
    def __init__(self) -> None:
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0


def test_ur5e_physical_move_insert_threads_action_and_complete_final_pose(  # noqa: PLR0915 - complete action contract.
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    status_path = tmp_path / "rtde-status.json"
    status_path.write_text(
        json.dumps(
            {
                "updated_at": time.time(),
                "insert_action_ready": True,
                "insert_max_timeout_sec": 60.0,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        hardware_pick_place_controller,
        "UR5E_RTDE_STATUS_PATH",
        status_path,
    )
    controller = object.__new__(UR5eHardwareController)
    goals: list[Any] = []
    final_pose = _PoseStamped()
    final_pose.header.frame_id = "world"
    final_pose.pose.position.z = 0.0
    wrapped = SimpleNamespace(
        status=4,
        result=SimpleNamespace(
            error_code=0,
            error_string="",
            trial_id="move-insert-test-1",
            hard_caps_sha256="d" * 64,
            state_uncertain=False,
            motion_settled=True,
            final_tool0_pose_valid=True,
            final_tool0_pose=final_pose,
            final_insertion_depth_m=0.01,
            final_depth_error_m=0.0,
            final_lateral_offset_m=0.0,
            final_tilt_error_rad=0.0,
            final_search_radius_m=0.0005,
            peak_axial_force_n=5.0,
            peak_lateral_force_n=1.0,
            peak_torque_nm=0.1,
            peak_filtered_axial_force_n=4.8,
            peak_filtered_lateral_force_n=0.8,
            peak_filtered_torque_nm=0.08,
            peak_tool_flange_torque_nm=0.2,
            contact_detected=True,
            engagement_detected=True,
            seated_detected=True,
            force_bias_valid=True,
            force_bias=[0.0, 0.0, 1.5, 0.0, 0.0, 0.05],
            final_phase="settling",
            soft_overload_detected=True,
            soft_overload_recovered=True,
            relief_exhausted=False,
            relief_cycle_count=1,
            last_soft_overload_reason="filtered torque crossed the recipe limit",
            relief_load_cleared=True,
            relief_backoff_m=0.0001,
            relief_planned_backoff_m=0.0002,
            total_relief_backoff_m=0.0001,
            relief_resume_phase="searching",
            relief_force_mode_stop_acknowledged=True,
            relief_stop_l_command_completed=True,
            relief_stationary_confirmed=True,
            relief_force_mode_restart_acknowledged=True,
            hard_limit_detected=False,
            hard_limit_reason="",
            limit_trigger="soft_torque",
            limit_trigger_value=0.12,
            limit_trigger_threshold=0.1,
            limit_trigger_actual_tcp_force=[0.1, 0.2, 5.0, 0.01, 0.02, 0.1],
            limit_trigger_tared_tcp_force=[0.1, 0.2, 3.5, 0.01, 0.02, 0.05],
            force_mode_stop_acknowledged=True,
            servo_stop_acknowledged=True,
            stop_l_command_completed=True,
            stationary_confirmed=True,
            server_trace_id="move-insert-test-1",
            server_trace_path=(
                "/tmp/move_insert_trials/move-insert-test-1/server_trace.jsonl"
            ),
            server_trace_sha256="e" * 64,
            server_trace_status="complete",
            server_trace_complete=True,
            server_trace_sample_count=147,
        ),
    )
    goal_handle = SimpleNamespace(
        accepted=True,
        get_result_async=lambda: _ImmediateFuture(wrapped),
    )
    def send_goal_async(goal: Any, *, feedback_callback: Any) -> Any:
        goals.append(goal)
        feedback_callback(
            SimpleNamespace(
                feedback=SimpleNamespace(
                    trial_id="move-insert-test-1",
                    phase="settling",
                    actual_tool0_pose=final_pose,
                    insertion_depth_m=0.01,
                    depth_error_m=0.0,
                    lateral_offset_m=0.0,
                    search_radius_m=0.0005,
                    axial_force_n=5.0,
                    lateral_force_n=1.0,
                    torque_nm=0.1,
                    filtered_axial_force_n=4.8,
                    filtered_lateral_force_n=0.8,
                    filtered_torque_nm=0.08,
                    tool_flange_torque_nm=0.2,
                    filtered_tool_flange_torque_nm=0.18,
                    contact_detected=True,
                    engagement_detected=True,
                    seated_detected=True,
                    soft_overload_detected=True,
                    soft_overload_reason="filtered torque crossed the recipe limit",
                    soft_overload_duration_sec=0.12,
                    relief_cycle_count=1,
                    relief_elapsed_sec=0.2,
                    relief_retreat_m=0.0002,
                    relief_load_cleared=True,
                    relief_backoff_m=0.0001,
                    relief_planned_backoff_m=0.0002,
                    total_relief_backoff_m=0.0001,
                    relief_resume_phase="searching",
                    commanded_axial_force_n=3.0,
                    commanded_lateral_force_x_n=0.1,
                    commanded_lateral_force_y_n=-0.1,
                    hard_limit_detected=False,
                    hard_limit_reason="",
                    limit_trigger="soft_torque",
                    limit_trigger_value=0.12,
                    limit_trigger_threshold=0.1,
                    limit_trigger_actual_tcp_force=[
                        0.1,
                        0.2,
                        5.0,
                        0.01,
                        0.02,
                        0.1,
                    ],
                    limit_trigger_tared_tcp_force=[
                        0.1,
                        0.2,
                        3.5,
                        0.01,
                        0.02,
                        0.05,
                    ],
                    force_bias_valid=True,
                    force_bias=[0.0, 0.0, 1.5, 0.0, 0.0, 0.05],
                    actual_tcp_force=[0.1, 0.2, 5.0, 0.01, 0.02, 0.1],
                    tared_tcp_force=[0.1, 0.2, 3.5, 0.01, 0.02, 0.05],
                    actual_tcp_speed=[0.0, 0.0, -0.0001, 0.0, 0.0, 0.0],
                )
            )
        )
        return _ImmediateFuture(goal_handle)

    controller._ur5e_hardware_insert_client = SimpleNamespace(
        wait_for_server=lambda timeout_sec: timeout_sec == 2.0,
        send_goal_async=send_goal_async,
    )
    controller._ur5e_hardware_insert_action = (
        "/cais_ur5e_rtde_cartesian_controller/move_insert"
    )
    controller._MoveUR5eInsert = SimpleNamespace(Goal=_InsertGoal)
    controller._PoseStamped = _PoseStamped
    controller._Vector3 = _Vector3
    controller._node = SimpleNamespace(
        get_clock=lambda: SimpleNamespace(
            now=lambda: SimpleNamespace(to_msg=lambda: "stamp")
        )
    )
    controller._wait_future = lambda future, **_kwargs: future.value
    controller._cancel_ur5e_hardware_trajectory_goal = lambda *_args, **_kwargs: ""
    controller._ur5e_action_send_timeout_sec = 10.0
    controller._move_insert_goal_condition = threading.Condition()
    controller._move_insert_dispatch_active = False
    controller._active_move_insert_goal_handle = None
    controller._active_move_insert_result_future = None
    controller._last_failure_message = ""
    controller.wait_for_services = lambda: True
    profile = _move_insert_profile()
    start = {"x": 0.0, "y": 0.0, "z": 0.01, "qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0}
    target = {"x": 0.0, "y": 0.0, "z": 0.0, "qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0}
    axis = {"x": 0.0, "y": 0.0, "z": -1.0}
    timeout, timeout_error = derive_move_insert_timeout_sec(
        start,
        target,
        axis,
        profile,
        part_name="MG",
        insert_max_timeout_sec=60.0,
    )
    assert timeout_error == ""

    result = controller.move_insert(
        part_name="MG",
        calibration_id="insert-calibration-1",
        profile_sha256="a" * 64,
        hard_caps_sha256="d" * 64,
            force_depth_profile={
            "depth_fraction": [index / 15.0 for index in range(16)],
            "axial_upper_n": [12.0] * 16,
            "lateral_upper_n": [6.0] * 16,
                "torque_upper_nm": [1.2] * 16,
            },
            baseline_force_uncertainty_n=0.2,
            baseline_torque_uncertainty_nm=0.01,
        trial_id="move-insert-test-1",
        expected_start_pose=start,
        target_pose=target,
        insertion_axis_world=axis,
        timeout_sec=timeout,
        **{key: profile[key] for key in (
            "contact_speed_m_s", "contact_force_delta_n", "engagement_progress_m",
            "insertion_force_n", "spiral_radius_m", "spiral_pitch_m",
            "spiral_speed_m_s", "spiral_acceleration_m_s2", "max_axial_force_n",
            "max_lateral_force_n", "max_torque_nm", "tilt_tolerance_rad",
            "seated_depth_tolerance_m", "settle_time_sec",
        )},
    )

    assert result["success"] is True
    assert result["motion_settled"] is True
    assert result["trial_id"] == "move-insert-test-1"
    assert result["final_phase"] == "settling"
    assert result["engagement_detected"] is True
    assert result["seated_detected"] is True
    assert result["force_bias_valid"] is True
    assert result["force_bias"] == pytest.approx(
        [0.0, 0.0, 1.5, 0.0, 0.0, 0.05]
    )
    assert result["feedback_trace"][0]["phase"] == "settling"
    assert result["feedback_trace"][0]["trial_id"] == "move-insert-test-1"
    assert result["feedback_trace"][0]["actual_tool0_pose"] == pytest.approx(target)
    assert result["feedback_trace"][0]["actual_tcp_force"] == pytest.approx(
        [0.1, 0.2, 5.0, 0.01, 0.02, 0.1]
    )
    assert result["feedback_trace"][0]["actual_tcp_speed"] == pytest.approx(
        [0.0, 0.0, -0.0001, 0.0, 0.0, 0.0]
    )
    assert result["feedback_trace"][0]["force_bias_valid"] is True
    assert result["feedback_trace"][0]["force_bias"] == pytest.approx(
        [0.0, 0.0, 1.5, 0.0, 0.0, 0.05]
    )
    assert result["feedback_trace"][0]["tared_tcp_force"] == pytest.approx(
        [0.1, 0.2, 3.5, 0.01, 0.02, 0.05]
    )
    assert result["feedback_trace"][0]["filtered_torque_nm"] == pytest.approx(
        0.08
    )
    assert result["feedback_trace"][0]["tool_flange_torque_nm"] == pytest.approx(
        0.2
    )
    assert result["feedback_trace"][0]["soft_overload_detected"] is True
    assert result["feedback_trace"][0]["relief_cycle_count"] == 1
    assert result["feedback_trace"][0]["relief_retreat_m"] == pytest.approx(
        0.0002
    )
    assert result["feedback_trace"][0]["relief_load_cleared"] is True
    assert result["feedback_trace"][0]["relief_backoff_m"] == pytest.approx(
        0.0001
    )
    assert result["feedback_trace"][0][
        "relief_planned_backoff_m"
    ] == pytest.approx(0.0002)
    assert result["feedback_trace"][0][
        "total_relief_backoff_m"
    ] == pytest.approx(0.0001)
    assert result["feedback_trace"][0]["relief_resume_phase"] == "searching"
    assert result["feedback_trace"][0]["hard_limit_detected"] is False
    assert result["feedback_trace"][0]["hard_limit_reason"] == ""
    assert result["feedback_trace"][0]["limit_trigger"] == "soft_torque"
    assert result["feedback_trace"][0]["limit_trigger_value"] == pytest.approx(
        0.12
    )
    assert result["feedback_trace"][0][
        "limit_trigger_threshold"
    ] == pytest.approx(0.1)
    assert result["feedback_trace"][0][
        "limit_trigger_actual_tcp_force"
    ] == pytest.approx([0.1, 0.2, 5.0, 0.01, 0.02, 0.1])
    assert result["feedback_trace"][0][
        "limit_trigger_tared_tcp_force"
    ] == pytest.approx([0.1, 0.2, 3.5, 0.01, 0.02, 0.05])
    assert result["feedback_trace"][0]["commanded_axial_force_n"] == pytest.approx(
        3.0
    )
    assert result["soft_overload_detected"] is True
    assert result["soft_overload_recovered"] is True
    assert result["relief_exhausted"] is False
    assert result["relief_cycle_count"] == 1
    assert result["peak_filtered_axial_force_n"] == pytest.approx(4.8)
    assert result["peak_filtered_lateral_force_n"] == pytest.approx(0.8)
    assert result["peak_filtered_torque_nm"] == pytest.approx(0.08)
    assert result["peak_tool_flange_torque_nm"] == pytest.approx(0.2)
    assert result["last_soft_overload_reason"] == (
        "filtered torque crossed the recipe limit"
    )
    assert result["relief_load_cleared"] is True
    assert result["relief_backoff_m"] == pytest.approx(0.0001)
    assert result["relief_planned_backoff_m"] == pytest.approx(0.0002)
    assert result["total_relief_backoff_m"] == pytest.approx(0.0001)
    assert result["relief_resume_phase"] == "searching"
    assert result["relief_force_mode_stop_acknowledged"] is True
    assert result["relief_stop_l_command_completed"] is True
    assert result["relief_stationary_confirmed"] is True
    assert result["relief_force_mode_restart_acknowledged"] is True
    assert result["hard_limit_detected"] is False
    assert result["hard_limit_reason"] == ""
    assert result["limit_trigger"] == "soft_torque"
    assert result["limit_trigger_value"] == pytest.approx(0.12)
    assert result["limit_trigger_threshold"] == pytest.approx(0.1)
    assert result["limit_trigger_actual_tcp_force"] == pytest.approx(
        [0.1, 0.2, 5.0, 0.01, 0.02, 0.1]
    )
    assert result["limit_trigger_tared_tcp_force"] == pytest.approx(
        [0.1, 0.2, 3.5, 0.01, 0.02, 0.05]
    )
    assert result["force_mode_stop_acknowledged"] is True
    assert result["servo_stop_acknowledged"] is True
    assert result["stop_l_command_completed"] is True
    assert result["stationary_confirmed"] is True
    assert result["server_trace_id"] == "move-insert-test-1"
    assert result["server_trace_path"] == (
        "/tmp/move_insert_trials/move-insert-test-1/server_trace.jsonl"
    )
    assert result["server_trace_sha256"] == "e" * 64
    assert result["server_trace_status"] == "complete"
    assert result["server_trace_complete"] is True
    assert result["server_trace_sample_count"] == 147
    json.dumps(result["feedback_trace"], allow_nan=False)
    assert result["absolute_position"] == pytest.approx(target)
    assert len(goals) == 1
    assert goals[0].part_name == "MG"
    assert goals[0].trial_id == "move-insert-test-1"
    assert goals[0].profile_sha256 == "a" * 64
    assert goals[0].expected_start_tool0_pose.header.frame_id == "world"
    assert goals[0].insertion_axis_world.z == pytest.approx(-1.0)


def test_insertion_demonstration_client_records_feedback_without_motion_command() -> None:
    class _DemonstrationGoal:
        pass

    result_future = _SettableFuture()
    goal_handle = SimpleNamespace(
        accepted=True,
        get_result_async=lambda: result_future,
        cancel_goal_async=lambda: _ImmediateFuture(
            SimpleNamespace(goals_canceling=[object()])
        ),
    )
    goals: list[Any] = []

    def send_goal_async(goal: Any, *, feedback_callback: Any) -> Any:
        goals.append(goal)
        feedback_callback_holder.append(feedback_callback)
        return _ImmediateFuture(goal_handle)

    feedback_callback_holder: list[Any] = []
    controller = object.__new__(UR5eHardwareController)
    controller._ur5e_hardware_insertion_demonstration_action = (
        "/cais_ur5e_rtde_cartesian_controller/record_insertion_demonstration"
    )
    controller._ur5e_hardware_insertion_demonstration_client = SimpleNamespace(
        wait_for_server=lambda timeout_sec: timeout_sec == 2.0,
        send_goal_async=send_goal_async,
    )
    controller._RecordUR5eInsertionDemonstration = SimpleNamespace(
        Goal=_DemonstrationGoal
    )
    controller._PoseStamped = _PoseStamped
    controller._node = SimpleNamespace(
        get_clock=lambda: SimpleNamespace(
            now=lambda: SimpleNamespace(to_msg=lambda: "stamp")
        )
    )
    controller._ur5e_action_send_timeout_sec = 10.0
    controller._insertion_demonstration_condition = threading.Condition()
    controller._active_insertion_demonstration_send_future = None
    controller._active_insertion_demonstration_goal_handle = None
    controller._active_insertion_demonstration_result_future = None
    controller._active_insertion_demonstration_status = {}
    controller._wait_ur5e_action_future_without_cancel = (
        lambda future, _timeout: future.result()
    )
    start_pose = {
        "x": 0.1,
        "y": 0.2,
        "z": 0.3,
        "qx": 0.0,
        "qy": 0.0,
        "qz": 0.0,
        "qw": 1.0,
    }

    started = controller.start_insertion_demonstration(
        recording_id="insertion-demonstration-test",
        part_name="MG",
        destination_location="assembly_board-v1",
        context_sha256="a" * 64,
        expected_start_tool0_pose=start_pose,
    )

    assert started["active"] is True
    assert len(goals) == 1
    assert goals[0].part_name == "MG"
    assert goals[0].expected_start_tool0_pose.header.frame_id == "world"
    feedback_pose = _PoseStamped()
    feedback_pose.pose.position.z = 0.28
    feedback_callback_holder[0](
        SimpleNamespace(
            feedback=SimpleNamespace(
                phase="recording_insertion",
                sample_count=25,
                elapsed_sec=0.2,
                actual_tool0_pose=feedback_pose,
                actual_tcp_force=[0.0, 0.0, 5.0, 0.0, 0.0, 0.1],
                actual_tcp_speed=[0.0] * 6,
                baseline_valid=True,
                force_bias=[0.0] * 6,
            )
        )
    )
    assert controller.insertion_demonstration_status()["sample_count"] == 25
    wrapped = SimpleNamespace(
        status=5,
        result=SimpleNamespace(
            error_code=0,
            error_string="recording stopped",
            state_uncertain=False,
            motion_settled=True,
            recording_id="insertion-demonstration-test",
            trace_path="/tmp/cais_ur5e_insertion_demonstrations/insertion-demonstration-test.jsonl",
            trace_sha256="b" * 64,
            sample_count=25,
            started_at=1.0,
            finished_at=2.0,
            baseline_valid=True,
            force_bias=[0.0] * 6,
            baseline_force_span_n=0.2,
            baseline_torque_span_nm=0.01,
        ),
    )
    result_future.set_result(wrapped)

    stopped = controller.stop_insertion_demonstration(timeout_sec=10.0)

    assert stopped["success"] is True, stopped
    assert stopped["motion_settled"] is True
    assert stopped["trace_sha256"] == "b" * 64


def test_move_insert_retains_call_until_delayed_acceptance_is_canceled_and_settled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    status_path = tmp_path / "rtde-status.json"
    status_path.write_text(
        json.dumps(
            {
                "updated_at": time.time(),
                "insert_action_ready": True,
                "insert_max_timeout_sec": 60.0,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        hardware_pick_place_controller,
        "UR5E_RTDE_STATUS_PATH",
        status_path,
    )
    wrapped = SimpleNamespace(
        status=5,
        result=SimpleNamespace(
            error_code=-3,
            error_string="canceled",
            hard_caps_sha256="d" * 64,
            state_uncertain=False,
            force_mode_stop_acknowledged=True,
            servo_stop_acknowledged=True,
            stop_l_command_completed=True,
            stationary_confirmed=True,
            relief_load_cleared=False,
            relief_backoff_m=0.0,
            relief_planned_backoff_m=0.0005,
            total_relief_backoff_m=0.0,
            relief_resume_phase="",
            relief_force_mode_stop_acknowledged=True,
            relief_stop_l_command_completed=True,
            relief_stationary_confirmed=True,
            relief_force_mode_restart_acknowledged=False,
            server_trace_id="move-insert-delayed-acceptance",
            server_trace_path=(
                "/tmp/move_insert_trials/move-insert-delayed-acceptance/"
                "server_trace.jsonl"
            ),
            server_trace_sha256="2" * 64,
            server_trace_status="complete",
            server_trace_complete=True,
            server_trace_sample_count=37,
        ),
    )
    goal_handle = SimpleNamespace(
        accepted=True,
        get_result_async=lambda: _ImmediateFuture(wrapped),
        cancel_goal_async=lambda: _ImmediateFuture(
            SimpleNamespace(goals_canceling=[object()])
        ),
    )
    send_future = _SettableFuture()
    controller = object.__new__(UR5eHardwareController)
    controller._ur5e_hardware_insert_client = SimpleNamespace(
        wait_for_server=lambda timeout_sec: timeout_sec == 2.0,
        send_goal_async=lambda _goal, *, feedback_callback: send_future,
    )
    controller._ur5e_hardware_insert_action = (
        "/cais_ur5e_rtde_cartesian_controller/move_insert"
    )
    controller._MoveUR5eInsert = SimpleNamespace(Goal=_InsertGoal)
    controller._PoseStamped = _PoseStamped
    controller._Vector3 = _Vector3
    controller._node = SimpleNamespace(
        get_clock=lambda: SimpleNamespace(
            now=lambda: SimpleNamespace(to_msg=lambda: "stamp")
        )
    )
    controller._wait_future = lambda future, **_kwargs: future.value
    controller._ur5e_action_send_timeout_sec = 0.01
    controller._move_insert_goal_condition = threading.Condition()
    controller._move_insert_dispatch_active = False
    controller._active_move_insert_send_future = None
    controller._active_move_insert_goal_handle = None
    controller._active_move_insert_result_future = None
    controller._last_failure_message = ""
    controller.wait_for_services = lambda: True
    profile = _move_insert_profile()
    start = {
        "x": 0.0,
        "y": 0.0,
        "z": 0.01,
        "qx": 0.0,
        "qy": 0.0,
        "qz": 0.0,
        "qw": 1.0,
    }
    target = {**start, "z": 0.0}
    axis = {"x": 0.0, "y": 0.0, "z": -1.0}
    timeout, timeout_error = derive_move_insert_timeout_sec(
        start,
        target,
        axis,
        profile,
        part_name="MG",
        insert_max_timeout_sec=60.0,
    )
    assert timeout_error == ""
    responses: list[dict[str, Any]] = []

    def execute() -> None:
        responses.append(
            controller.move_insert(
                part_name="MG",
                calibration_id="insert-calibration-1",
                profile_sha256="a" * 64,
                hard_caps_sha256="d" * 64,
                force_depth_profile={
                    "depth_fraction": [index / 15.0 for index in range(16)],
                    "axial_upper_n": [12.0] * 16,
                    "lateral_upper_n": [6.0] * 16,
                    "torque_upper_nm": [1.2] * 16,
                },
                baseline_force_uncertainty_n=0.2,
                baseline_torque_uncertainty_nm=0.01,
                trial_id="move-insert-delayed-acceptance",
                expected_start_pose=start,
                target_pose=target,
                insertion_axis_world=axis,
                timeout_sec=timeout,
                **{
                    key: profile[key]
                    for key in (
                        "contact_speed_m_s",
                        "contact_force_delta_n",
                        "engagement_progress_m",
                        "insertion_force_n",
                        "spiral_radius_m",
                        "spiral_pitch_m",
                        "spiral_speed_m_s",
                        "spiral_acceleration_m_s2",
                        "max_axial_force_n",
                        "max_lateral_force_n",
                        "max_torque_nm",
                        "tilt_tolerance_rad",
                        "seated_depth_tolerance_m",
                        "settle_time_sec",
                    )
                },
            )
        )

    worker = threading.Thread(target=execute)
    worker.start()
    time.sleep(0.05)

    assert worker.is_alive()
    assert controller._move_insert_dispatch_active is True
    assert controller._active_move_insert_send_future is send_future

    send_future.set_result(goal_handle)
    worker.join(timeout=1.0)

    assert worker.is_alive() is False
    assert len(responses) == 1
    assert responses[0]["success"] is False
    assert responses[0]["state_uncertain"] is False
    assert responses[0]["force_mode_stop_acknowledged"] is True
    assert responses[0]["servo_stop_acknowledged"] is True
    assert responses[0]["stop_l_command_completed"] is True
    assert responses[0]["stationary_confirmed"] is True
    assert responses[0]["relief_load_cleared"] is False
    assert responses[0]["relief_backoff_m"] == pytest.approx(0.0)
    assert responses[0]["relief_planned_backoff_m"] == pytest.approx(0.0005)
    assert responses[0]["total_relief_backoff_m"] == pytest.approx(0.0)
    assert responses[0]["relief_resume_phase"] == ""
    assert responses[0]["relief_force_mode_stop_acknowledged"] is True
    assert responses[0]["relief_stop_l_command_completed"] is True
    assert responses[0]["relief_stationary_confirmed"] is True
    assert responses[0]["relief_force_mode_restart_acknowledged"] is False
    assert responses[0]["server_trace_id"] == "move-insert-delayed-acceptance"
    assert responses[0]["server_trace_path"] == (
        "/tmp/move_insert_trials/move-insert-delayed-acceptance/"
        "server_trace.jsonl"
    )
    assert responses[0]["server_trace_sha256"] == "2" * 64
    assert responses[0]["server_trace_status"] == "complete"
    assert responses[0]["server_trace_complete"] is True
    assert responses[0]["server_trace_sample_count"] == 37
    assert "send acknowledgement timeout" in responses[0]["message"]
    assert "settlement confirmed" in responses[0]["message"]
    assert controller._move_insert_dispatch_active is False


def test_cancel_move_insert_waits_for_terminal_settlement_and_clears_goal() -> None:
    controller = object.__new__(UR5eHardwareController)
    wrapped = SimpleNamespace(
        status=5,
        result=SimpleNamespace(
            error_code=-3,
            error_string="canceled",
            state_uncertain=False,
            force_mode_stop_acknowledged=True,
            servo_stop_acknowledged=True,
            stop_l_command_completed=True,
            stationary_confirmed=True,
            relief_load_cleared=False,
            relief_backoff_m=0.0,
            relief_planned_backoff_m=0.0,
            total_relief_backoff_m=0.0,
            relief_resume_phase="",
            relief_force_mode_stop_acknowledged=True,
            relief_stop_l_command_completed=True,
            relief_stationary_confirmed=True,
            relief_force_mode_restart_acknowledged=False,
            server_trace_id="move-insert-canceled",
            server_trace_path=(
                "/tmp/move_insert_trials/move-insert-canceled/server_trace.jsonl"
            ),
            server_trace_sha256="f" * 64,
            server_trace_status="complete",
            server_trace_complete=True,
            server_trace_sample_count=51,
        ),
    )
    result_future = _ImmediateFuture(wrapped)
    goal_handle = SimpleNamespace(
        cancel_goal_async=lambda: _ImmediateFuture(
            SimpleNamespace(goals_canceling=[object()])
        )
    )
    controller._move_insert_goal_condition = threading.Condition()
    controller._move_insert_dispatch_active = True
    controller._active_move_insert_goal_handle = goal_handle
    controller._active_move_insert_result_future = result_future
    controller._wait_future = lambda future, **_kwargs: future.value

    result = controller.cancel_move_insert(timeout_sec=1.0)

    assert result == {
        "success": True,
        "canceled": True,
        "settled": True,
        "terminal": True,
        "state_uncertain": False,
        "motion_settled": True,
        "message": "move_insert canceled and stationary settlement confirmed",
        "cancel_accepted": True,
        "goal_status": 5,
        "error_code": -3,
        "trial_id": "",
        "force_mode_stop_acknowledged": True,
        "servo_stop_acknowledged": True,
        "stop_l_command_completed": True,
        "stationary_confirmed": True,
        "relief_load_cleared": False,
        "relief_backoff_m": 0.0,
        "relief_planned_backoff_m": 0.0,
        "total_relief_backoff_m": 0.0,
        "relief_resume_phase": "",
        "relief_force_mode_stop_acknowledged": True,
        "relief_stop_l_command_completed": True,
        "relief_stationary_confirmed": True,
        "relief_force_mode_restart_acknowledged": False,
        "server_trace_id": "move-insert-canceled",
        "server_trace_path": (
            "/tmp/move_insert_trials/move-insert-canceled/server_trace.jsonl"
        ),
        "server_trace_sha256": "f" * 64,
        "server_trace_status": "complete",
        "server_trace_complete": True,
        "server_trace_sample_count": 51,
    }
    assert controller._active_move_insert_goal_handle is None
    assert controller._active_move_insert_result_future is None
    assert controller._move_insert_dispatch_active is False


def test_cancel_move_insert_retains_goal_when_terminal_settlement_is_unknown() -> None:
    controller = object.__new__(UR5eHardwareController)
    result_future = _PendingFuture()
    goal_handle = SimpleNamespace(
        cancel_goal_async=lambda: _ImmediateFuture(
            SimpleNamespace(goals_canceling=[object()])
        )
    )
    controller._move_insert_goal_condition = threading.Condition()
    controller._move_insert_dispatch_active = True
    controller._active_move_insert_goal_handle = goal_handle
    controller._active_move_insert_result_future = result_future
    controller._wait_future = lambda future, **_kwargs: future.value

    result = controller.cancel_move_insert(timeout_sec=0.1)

    assert result["success"] is False
    assert result["settled"] is False
    assert result["state_uncertain"] is True
    assert "settlement was not confirmed" in result["message"]
    assert controller._active_move_insert_goal_handle is goal_handle
    assert controller._active_move_insert_result_future is result_future
    assert controller._move_insert_dispatch_active is True


@pytest.mark.parametrize(
    ("motion_settled", "expected_settled"),
    [(True, True), (False, False)],
)
def test_cancel_move_insert_separates_task_uncertainty_from_motion_settlement(
    motion_settled: bool,
    expected_settled: bool,
) -> None:
    controller = object.__new__(UR5eHardwareController)
    wrapped = SimpleNamespace(
        status=6,
        result=SimpleNamespace(
            error_code=-4,
            error_string="task outcome uncertain",
            state_uncertain=True,
            motion_settled=motion_settled,
        ),
    )
    goal_handle = SimpleNamespace(
        cancel_goal_async=lambda: _ImmediateFuture(
            SimpleNamespace(goals_canceling=[object()])
        )
    )
    controller._move_insert_goal_condition = threading.Condition()
    controller._move_insert_dispatch_active = True
    controller._active_move_insert_send_future = None
    controller._active_move_insert_goal_handle = goal_handle
    controller._active_move_insert_result_future = _ImmediateFuture(wrapped)
    controller._wait_future = lambda future, **_kwargs: future.value

    result = controller.cancel_move_insert(timeout_sec=1.0)

    assert result["settled"] is expected_settled
    assert result["terminal"] is True
    assert result["state_uncertain"] is True
    assert result["motion_settled"] is expected_settled
    assert controller._move_insert_dispatch_active is False
    assert controller._active_move_insert_goal_handle is None


def test_move_insert_terminal_unsettled_result_releases_action_ownership() -> None:
    controller = object.__new__(UR5eHardwareController)
    wrapped = SimpleNamespace(
        status=6,
        result=SimpleNamespace(
            error_code=-6,
            error_string="hard torque limit stop settlement is unconfirmed",
            trial_id="move-insert-unsettled",
            state_uncertain=True,
            motion_settled=False,
            force_mode_stop_acknowledged=True,
            servo_stop_acknowledged=True,
            stop_l_command_completed=False,
            stationary_confirmed=False,
            relief_load_cleared=False,
            relief_backoff_m=0.0002,
            relief_planned_backoff_m=0.0005,
            total_relief_backoff_m=0.0006,
            relief_resume_phase="",
            relief_force_mode_stop_acknowledged=True,
            relief_stop_l_command_completed=True,
            relief_stationary_confirmed=True,
            relief_force_mode_restart_acknowledged=False,
            server_trace_id="move-insert-unsettled",
            server_trace_path=(
                "/tmp/move_insert_trials/move-insert-unsettled/server_trace.jsonl"
            ),
            server_trace_sha256="1" * 64,
            server_trace_status="complete",
            server_trace_complete=True,
            server_trace_sample_count=83,
        ),
    )
    result_future = _ImmediateFuture(wrapped)
    goal_handle = object()
    controller._move_insert_goal_condition = threading.Condition()
    controller._move_insert_dispatch_active = True
    controller._active_move_insert_send_future = None
    controller._active_move_insert_goal_handle = goal_handle
    controller._active_move_insert_result_future = result_future

    result = controller._retain_move_insert_until_terminal_settlement(
        goal_handle,
        result_future,
        {
            "settled": False,
            "terminal": False,
            "state_uncertain": True,
            "message": "waiting for terminal result",
        },
    )

    assert result == {
        "settled": False,
        "terminal": True,
        "state_uncertain": True,
        "message": (
            "hard torque limit stop settlement is unconfirmed; action ownership "
            "was released for Hardware Stack recovery"
        ),
        "goal_status": 6,
        "error_code": -6,
        "trial_id": "move-insert-unsettled",
        "force_mode_stop_acknowledged": True,
        "servo_stop_acknowledged": True,
        "stop_l_command_completed": False,
        "stationary_confirmed": False,
        "relief_load_cleared": False,
        "relief_backoff_m": 0.0002,
        "relief_planned_backoff_m": 0.0005,
        "total_relief_backoff_m": 0.0006,
        "relief_resume_phase": "",
        "relief_force_mode_stop_acknowledged": True,
        "relief_stop_l_command_completed": True,
        "relief_stationary_confirmed": True,
        "relief_force_mode_restart_acknowledged": False,
        "server_trace_id": "move-insert-unsettled",
        "server_trace_path": (
            "/tmp/move_insert_trials/move-insert-unsettled/server_trace.jsonl"
        ),
        "server_trace_sha256": "1" * 64,
        "server_trace_status": "complete",
        "server_trace_complete": True,
        "server_trace_sample_count": 83,
    }
    assert controller._move_insert_dispatch_active is False
    assert controller._active_move_insert_goal_handle is None
    assert controller._active_move_insert_result_future is None


def test_cancel_move_insert_retains_dispatch_when_goal_acceptance_is_unknown() -> None:
    controller = object.__new__(UR5eHardwareController)
    controller._move_insert_goal_condition = threading.Condition()
    controller._move_insert_dispatch_active = False
    controller._active_move_insert_goal_handle = None
    controller._active_move_insert_result_future = None
    assert controller._claim_move_insert_dispatch() is True

    result = controller.cancel_move_insert(timeout_sec=0.1)

    assert result == {
        "success": False,
        "canceled": False,
        "settled": False,
        "state_uncertain": True,
        "message": "move_insert goal acceptance is still pending",
    }
    assert controller._active_move_insert_goal_handle is None
    assert controller._active_move_insert_result_future is None
    assert controller._move_insert_dispatch_active is True


def test_cancel_move_insert_recovers_late_goal_acceptance_before_canceling() -> None:
    controller = object.__new__(UR5eHardwareController)
    wrapped = SimpleNamespace(
        status=5,
        result=SimpleNamespace(
            error_code=-3,
            error_string="canceled",
            state_uncertain=False,
        ),
    )
    goal_handle = SimpleNamespace(
        accepted=True,
        get_result_async=lambda: _ImmediateFuture(wrapped),
        cancel_goal_async=lambda: _ImmediateFuture(
            SimpleNamespace(goals_canceling=[object()])
        ),
    )
    controller._move_insert_goal_condition = threading.Condition()
    controller._move_insert_dispatch_active = True
    controller._active_move_insert_send_future = _ImmediateFuture(goal_handle)
    controller._active_move_insert_goal_handle = None
    controller._active_move_insert_result_future = None
    controller._wait_future = lambda future, **_kwargs: future.value

    result = controller.cancel_move_insert(timeout_sec=1.0)

    assert result["success"] is True
    assert result["canceled"] is True
    assert result["settled"] is True
    assert result["state_uncertain"] is False
    assert controller._active_move_insert_send_future is None
    assert controller._active_move_insert_goal_handle is None
    assert controller._active_move_insert_result_future is None
    assert controller._move_insert_dispatch_active is False
