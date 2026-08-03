"""Focused tests for Cartesian primitive orientation handling."""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any

import pytest

from cais_spade_llm.resources.robot.gazebo_pick_place_controller import (
    GazeboPickPlaceController,
)
from cais_spade_llm.resources.robot.hardware_pick_place_controller import (
    HardwarePickPlaceController,
)


class _FakePose:
    def __init__(self) -> None:
        self.position = SimpleNamespace(x=0.0, y=0.0, z=0.0)
        self.orientation = SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0)


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
