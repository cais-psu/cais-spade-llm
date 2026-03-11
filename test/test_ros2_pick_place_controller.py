from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
PKG_ROOT = REPO_ROOT / "cais_spade_llm"
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from cais_spade_llm.resources.robot.ros2_pick_place_controller import (
    Ros2PickPlaceController,
)


class _DummyLogger:
    def __init__(self) -> None:
        self.warn_messages: list[str] = []
        self.error_messages: list[str] = []
        self.info_messages: list[str] = []

    def warn(self, message: str) -> None:
        self.warn_messages.append(str(message))

    def error(self, message: str) -> None:
        self.error_messages.append(str(message))

    def info(self, message: str) -> None:
        self.info_messages.append(str(message))


class _Pose:
    def __init__(self) -> None:
        self.position = SimpleNamespace(x=0.0, y=0.0, z=0.0)
        self.orientation = None


def _build_controller() -> tuple[Ros2PickPlaceController, _DummyLogger]:
    controller = Ros2PickPlaceController(
        robot_name="testbot",
        node_name="testbot_controller",
        controller_config={
            "move_group": {
                "group_name": "arm",
                "ee_link": "tool0",
                "tcp_link": "tcp",
                "frame_id": "world",
            },
            "gripper": {
                "joint": "gripper_joint",
                "topic": "/gripper",
                "open": 0.0,
                "close": 1.0,
                "move_time_sec": 0.4,
                "settle_sec": 0.1,
                "feedback_timeout_pad_sec": 0.5,
                "position_tolerance": 0.01,
            },
            "services": {
                "detect_all": "/detect_all",
                "cartesian_path": "/compute_cartesian_path",
                "execute_trajectory": "/execute_trajectory",
                "attach": "/attach",
                "detach": "/detach",
                "set_entity_state": "/set_entity_state",
            },
            "attach": {
                "robot_model_name": "robot",
                "attach_link_candidates": ["tool0"],
                "primary_attach_link": "tool0",
                "detach_timeout_sec": 1.0,
                "detach_max_link_attempts": 1,
                "release_detach_timeout_sec": 1.0,
            },
            "motion": {
                "approach_height_m": 0.2,
                "pick_tcp_z_bias_max_m": 0.02,
                "pick_tcp_z_bias_min_m": 0.003,
                "min_pick_tcp_z_m": 1.07,
                "place_surface_gap_m": -0.01,
                "release_preopen_settle_sec": 0.15,
                "release_postopen_settle_sec": 0.6,
                "release_postdetach_settle_sec": 0.35,
                "release_descend_time_scale": 1.35,
                "release_detach_retry_count": 2,
                "release_detach_retry_delay_sec": 0.35,
                "release_retry_lift_m": 0.005,
                "trajectory_time_scale": 0.9,
            },
            "parts_tuning": {
                "insertion_depth_m": 0.0025,
            },
        },
        execution_mode="simulation",
    )
    controller._Pose = _Pose
    logger = _DummyLogger()
    controller._log = lambda: logger
    return controller, logger


def _pose(x: float, y: float, z: float, orientation) -> _Pose:
    pose = _Pose()
    pose.position.x = float(x)
    pose.position.y = float(y)
    pose.position.z = float(z)
    pose.orientation = orientation
    return pose


def test_move_xy_direct_returns_after_direct_success():
    controller, logger = _build_controller()
    orientation = object()
    calls: list[str] = []

    def fake_cartesian_move(target, label: str, **_kwargs) -> bool:
        calls.append(label)
        assert target.position.x == 0.4
        assert target.position.y == -0.2
        assert target.position.z == 1.3
        assert target.orientation is orientation
        return True

    controller._cartesian_move = fake_cartesian_move
    controller._get_ee_pose = lambda: _pose(0.0, 0.0, 1.3, orientation)

    assert controller._move_xy_direct(0.4, -0.2, 1.3, orientation, "Move above part")
    assert calls == ["Move above part"]
    assert logger.warn_messages == []


def test_move_xy_direct_retries_with_alternate_axis_order():
    controller, logger = _build_controller()
    orientation = object()
    calls: list[tuple[str, float, float]] = []
    ee_reads = [
        _pose(0.1, 0.2, 1.25, orientation),
        _pose(0.1, 0.2, 1.25, orientation),
    ]

    def fake_get_ee_pose():
        return ee_reads.pop(0)

    outcomes = {
        "Move above part": False,
        "Move above part (leg X)": True,
        "Move above part (leg Y)": False,
        "Move above part (leg Y, no-collision)": False,
        "Move above part (leg Y, alt-order)": True,
        "Move above part (leg X, alt-order)": True,
    }

    def fake_cartesian_move(target, label: str, **_kwargs) -> bool:
        calls.append((label, target.position.x, target.position.y))
        return outcomes[label]

    controller._cartesian_move = fake_cartesian_move
    controller._get_ee_pose = fake_get_ee_pose

    assert controller._move_xy_direct(0.4, -0.2, 1.3, orientation, "Move above part")
    assert calls == [
        ("Move above part", 0.4, -0.2),
        ("Move above part (leg X)", 0.4, 0.2),
        ("Move above part (leg Y)", 0.4, -0.2),
        ("Move above part (leg Y, no-collision)", 0.4, -0.2),
        ("Move above part (leg Y, alt-order)", 0.1, -0.2),
        ("Move above part (leg X, alt-order)", 0.4, -0.2),
    ]
    assert logger.warn_messages == [
        "[Move above part] direct Cartesian move failed; retrying staged XY fallback"
    ]
    assert logger.info_messages == [
        "[Move above part] staged XY fallback succeeded with axis order Y->X"
    ]


def test_move_xy_direct_fails_when_current_pose_cannot_be_read():
    controller, logger = _build_controller()
    controller._cartesian_move = lambda _target, _label, **_kwargs: False
    controller._get_ee_pose = lambda: None

    assert controller._move_xy_direct(0.4, -0.2, 1.3, object(), "Move above part") is False
    assert logger.error_messages == [
        "[Move above part] cannot read current EE pose for staged fallback"
    ]


def test_move_xy_direct_segments_long_axis_legs_for_small_step_size():
    controller, _logger = _build_controller()
    controller.xy_axis_step_m = 0.1
    orientation = object()
    calls: list[str] = []

    controller._get_ee_pose = lambda: _pose(0.1, 0.2, 1.25, orientation)

    def fake_cartesian_move(target, label: str, **_kwargs) -> bool:
        calls.append(label)
        if label == "Move above part":
            return False
        return True

    controller._cartesian_move = fake_cartesian_move

    assert controller._move_xy_direct(0.4, -0.2, 1.3, orientation, "Move above part")
    assert calls == [
        "Move above part",
        "Move above part (leg X, step 1/4)",
        "Move above part (leg X, step 2/4)",
        "Move above part (leg X, step 3/4)",
        "Move above part (leg X, step 4/4)",
        "Move above part (leg Y, step 1/4)",
        "Move above part (leg Y, step 2/4)",
        "Move above part (leg Y, step 3/4)",
        "Move above part (leg Y, step 4/4)",
    ]


def test_move_xy_direct_retries_direct_no_collision_when_xy_unchanged():
    controller, logger = _build_controller()
    orientation = object()
    calls: list[str] = []

    controller._get_ee_pose = lambda: _pose(0.4, -0.2, 1.25, orientation)

    def fake_cartesian_move(target, label: str, **_kwargs) -> bool:
        calls.append(label)
        assert target.position.x == 0.4
        assert target.position.y == -0.2
        assert target.position.z == 1.3
        return label.endswith("(no-collision)")

    controller._cartesian_move = fake_cartesian_move

    assert controller._move_xy_direct(0.4, -0.2, 1.3, orientation, "Descend")
    assert calls == ["Descend", "Descend (no-collision)"]
    assert logger.warn_messages == [
        "[Descend] direct Cartesian move failed with no XY delta; retrying direct no-collision fallback"
    ]


def test_move_cartesian_uses_direct_pose_helper_when_xy_is_unchanged():
    controller, _logger = _build_controller()
    orientation = object()
    helper_calls: list[tuple[str, tuple[float, float, float], dict[str, float | str | object | None]]] = []

    controller.wait_for_services = lambda: True
    controller._get_ee_pose = lambda: _pose(0.4, -0.2, 1.25, orientation)

    def fake_move_pose_direct(x: float, y: float, z: float, **kwargs):
        helper_calls.append(("_move_pose_direct", (x, y, z), kwargs))
        return {"success": True, "message": "direct path"}

    def fake_move_xy_at_z(x: float, y: float, z: float, **kwargs):
        helper_calls.append(("_move_xy_at_z", (x, y, z), kwargs))
        return {"success": True, "message": "xy path"}

    controller._move_pose_direct = fake_move_pose_direct
    controller._move_xy_at_z = fake_move_xy_at_z

    result = controller.move_cartesian(0.4, -0.2, 1.3, speed=0.75)

    assert result == {"success": True, "message": "direct path"}
    assert helper_calls == [
        (
            "_move_pose_direct",
            (0.4, -0.2, 1.3),
            {
                "orientation": orientation,
                "label": "move_cartesian",
                "speed": 0.75,
            },
        )
    ]
