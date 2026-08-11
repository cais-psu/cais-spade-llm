"""Focused tests for Cartesian primitive orientation handling."""

from __future__ import annotations

import inspect
import math
import threading
import time
from types import SimpleNamespace
from typing import Any

import pytest

from cais_spade_llm.resources.robot.gazebo_pick_place_controller import (
    GazeboPickPlaceController,
)
from cais_spade_llm.resources.robot.hardware_pick_place_controller import (
    HardwarePickPlaceController,
    UR5eHardwareController,
    XArm6HardwareController,
)


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


def test_move_cartesian_without_quaternion_keeps_current_orientation_and_call_shape() -> None:
    controller, calls = _controller_double()

    result = controller.move_cartesian(0.1, 0.2, 0.4, 1.7)

    assert result["success"] is True
    assert calls["wait_count"] == 1
    assert len(calls["moves"]) == 1
    kind, x, y, z, kwargs = calls["moves"][0]
    assert (kind, x, y, z) == ("direct", 0.1, 0.2, 0.4)
    assert kwargs["speed"] == 1.7
    assert kwargs["orientation"] is calls["current_orientation"]


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
