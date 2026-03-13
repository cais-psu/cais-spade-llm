from __future__ import annotations

import math
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


def test_move_to_named_pose_waits_for_joint_targets_when_using_arm_publisher():
    controller, _logger = _build_controller()
    controller.wait_for_services = lambda: True
    controller.arm_joint_names = [
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    ]
    controller.named_positions = {
        "side_pick_approach": [0.1, -0.2, 0.3, -0.4, 0.5, -0.6],
    }
    controller._arm_pub = object()
    controller._exec_client = None

    published: list[tuple[list[float], float]] = []
    waited: list[tuple[list[float], float, float, bool]] = []

    controller.move_joints = lambda positions, duration_sec=2.0: (  # type: ignore[assignment]
        published.append((list(positions), float(duration_sec))) or True
    )
    controller._wait_for_arm_joint_targets = (  # type: ignore[assignment]
        lambda positions, timeout_sec, tolerance_rad=0.08, log_miss=True: (
            waited.append((list(positions), float(timeout_sec), float(tolerance_rad), bool(log_miss)))
            or True
        )
    )

    result = controller.move_to_named_pose("side_pick_approach", speed=0.5)

    assert result["success"] is True
    assert published == [([0.1, -0.2, 0.3, -0.4, 0.5, -0.6], 2.0)]
    assert waited == [([0.1, -0.2, 0.3, -0.4, 0.5, -0.6], 4.0, 0.08, False)]


def test_rotate_wrist_waits_for_joint_targets_before_reporting_success():
    controller, _logger = _build_controller()
    controller.wait_for_services = lambda: True
    controller.arm_joint_names = [
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    ]
    controller._arm_pub = object()
    controller._exec_client = None

    published: list[tuple[list[float], float]] = []
    waited: list[tuple[list[float], float, float, bool]] = []

    controller._get_arm_joint_positions = lambda timeout_sec=0.0: (  # type: ignore[assignment]
        [0.0, 0.1, 0.2, 0.3, 0.4, 1.0],
        [],
    )
    controller.move_joints = lambda positions, duration_sec=2.0: (  # type: ignore[assignment]
        published.append((list(positions), float(duration_sec))) or True
    )
    controller._wait_for_arm_joint_targets = (  # type: ignore[assignment]
        lambda positions, timeout_sec, tolerance_rad=0.08, log_miss=True: (
            waited.append((list(positions), float(timeout_sec), float(tolerance_rad), bool(log_miss)))
            or True
        )
    )

    result = controller.rotate_wrist(90.0, speed=0.8)

    expected_targets = [0.0, 0.1, 0.2, 0.3, 0.4, 1.0 + (math.pi / 2.0)]
    assert result["success"] is True
    assert published[0][0] == expected_targets
    assert math.isclose(published[0][1], 1.2, rel_tol=0.0, abs_tol=1e-9)
    assert waited[0][0] == expected_targets
    assert math.isclose(waited[0][1], 3.2, rel_tol=0.0, abs_tol=1e-9)
    assert math.isclose(waited[0][2], math.radians(5.0), rel_tol=0.0, abs_tol=1e-9)
    assert waited[0][3] is False


def test_get_arm_joint_command_names_prefers_prefixed_joint_names_when_available():
    controller, _logger = _build_controller()
    controller.robot_name = "ur5e"
    controller.arm_joint_names = [
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    ]
    controller._joint_positions = {
        "ur5e_shoulder_pan_joint": 0.1,
        "ur5e_shoulder_lift_joint": 0.2,
        "ur5e_elbow_joint": 0.3,
        "ur5e_wrist_1_joint": 0.4,
        "ur5e_wrist_2_joint": 0.5,
        "ur5e_wrist_3_joint": 0.6,
    }

    assert controller._get_arm_joint_command_names() == [
        "ur5e_shoulder_pan_joint",
        "ur5e_shoulder_lift_joint",
        "ur5e_elbow_joint",
        "ur5e_wrist_1_joint",
        "ur5e_wrist_2_joint",
        "ur5e_wrist_3_joint",
    ]


def test_detach_part_retries_before_failing():
    controller, logger = _build_controller()
    controller.wait_for_services = lambda: True
    controller.release_detach_retry_count = 2
    controller.release_detach_retry_delay_sec = 0.0

    calls: list[str] = []
    outcomes = iter([False, True])

    def fake_detach_part(model_name: str = "", timeout_sec=None, attached_link_only: bool = False, log_failure: bool = True):
        calls.append(f"{model_name}|{timeout_sec}|{attached_link_only}|{log_failure}")
        return next(outcomes)

    controller._detach_part = fake_detach_part  # type: ignore[assignment]

    result = controller.detach_part("circ_pin_medium")

    assert result == {"success": True, "message": "detached circ_pin_medium"}
    assert len(calls) == 2
    assert calls[0].startswith("circ_pin_medium|1.0|False|False")
    assert calls[1].startswith("circ_pin_medium|1.0|False|False")
    assert logger.warn_messages == ["Detach retry 1/2 for circ_pin_medium"]


def test_detach_part_can_assume_release_when_gripper_is_already_open():
    controller, logger = _build_controller()
    controller.wait_for_services = lambda: True
    controller.release_detach_retry_count = 0
    controller._attached_model = "circ_pin_medium"
    controller._attached_link = "ur5e_rg2_gripper_tcp"
    controller._joint_positions = {"gripper_joint": 0.11}
    controller._detach_part = lambda *args, **kwargs: False  # type: ignore[assignment]

    result = controller.detach_part("circ_pin_medium", assume_released_if_open=True)

    assert result == {
        "success": True,
        "message": "assumed detached circ_pin_medium after gripper opened",
    }
    assert controller._attached_model is None
    assert controller._attached_link is None
    assert logger.warn_messages == [
        "Assuming circ_pin_medium was released because the gripper is already open"
    ]


def test_compute_place_targets_supports_recovery_insert_without_pick_ctx():
    controller, _logger = _build_controller()
    controller.wait_for_services = lambda: True
    controller._get_ee_tcp_world_z_offset = lambda: -0.172  # type: ignore[assignment]

    result = controller.compute_place_targets(
        product_geometry={
            "slot_xy": [0.1, -0.08],
            "slot_floor_z_m": 1.025,
            "board_center": {"z": 1.025},
            "part_height_m": 0.1,
            "model_name": "circ_pin_large",
        },
        part_name="LCP",
    )

    assert result["success"] is True
    assert result["part_name"] == "LCP"
    assert math.isclose(result["slot_x"], 0.1, rel_tol=0.0, abs_tol=1e-9)
    assert math.isclose(result["slot_y"], -0.08, rel_tol=0.0, abs_tol=1e-9)
    assert math.isclose(result["board_top_z"], 1.025, rel_tol=0.0, abs_tol=1e-9)
    assert math.isclose(result["grasp_tcp_to_part_origin_z"], 0.02, rel_tol=0.0, abs_tol=1e-9)
    assert math.isclose(result["tcp_offset_z"], -0.172, rel_tol=0.0, abs_tol=1e-9)
    assert math.isclose(result["place_z"], 1.2545, rel_tol=0.0, abs_tol=1e-9)


def test_compute_place_targets_applies_z_adjustment():
    controller, _logger = _build_controller()
    controller.wait_for_services = lambda: True
    controller._get_ee_tcp_world_z_offset = lambda: -0.172  # type: ignore[assignment]

    result = controller.compute_place_targets(
        product_geometry={
            "slot_xy": [0.1, -0.08],
            "slot_floor_z_m": 1.025,
            "board_center": {"z": 1.025},
            "part_height_m": 0.1,
            "model_name": "gear_large",
        },
        part_name="LG",
        z_adjustment_m=0.006,
    )

    assert result["success"] is True
    assert math.isclose(result["place_z"], 1.2605, rel_tol=0.0, abs_tol=1e-9)


def test_compute_pick_targets_applies_part_specific_pick_z_adjustment():
    controller, _logger = _build_controller()
    controller.wait_for_services = lambda: True
    controller.detect_parts = lambda: [  # type: ignore[assignment]
        {"part_name": "LG", "x": 0.1, "y": 0.08, "z": 1.03, "model_name": "gear_large"}
    ]
    controller._get_ee_pose = lambda: _pose(0.0, 0.0, 1.3, object())
    controller._get_ee_tcp_world_z_offset = lambda: -0.172  # type: ignore[assignment]
    controller.pick_z_adjustments_m = {"LG": -0.003}

    result = controller.compute_pick_targets(
        part_name="LG",
        product_geometry={
            "board_center": {"z": 1.025},
            "part_height_m": 0.02,
            "model_name": "gear_large",
        },
    )

    assert result["success"] is True
    assert math.isclose(result["pick_z"], 1.239, rel_tol=0.0, abs_tol=1e-9)


def test_compute_pick_targets_can_ignore_current_height_for_travel_z():
    controller, _logger = _build_controller()
    controller.wait_for_services = lambda: True
    controller.detect_parts = lambda: [  # type: ignore[assignment]
        {"part_name": "LG", "x": 0.1, "y": 0.08, "z": 1.03, "model_name": "gear_large"}
    ]
    controller._get_ee_pose = lambda: _pose(0.0, 0.0, 1.45, object())
    controller._get_ee_tcp_world_z_offset = lambda: -0.172  # type: ignore[assignment]

    result = controller.compute_pick_targets(
        part_name="LG",
        product_geometry={
            "board_center": {"z": 1.025},
            "part_height_m": 0.02,
            "model_name": "gear_large",
        },
        approach_height_override_m=0.06,
        ignore_current_height_for_travel_z=True,
    )

    assert result["success"] is True
    assert math.isclose(result["travel_z"], 1.292, rel_tol=0.0, abs_tol=1e-9)


def test_move_relative_retries_vertical_motion_without_collision_check():
    controller, logger = _build_controller()
    orientation = object()
    calls: list[tuple[str, bool, float, bool]] = []
    controller.wait_for_services = lambda: True
    controller._get_ee_pose = lambda: _pose(0.4, -0.2, 1.1, orientation)

    def fake_cartesian_move(target, label: str, **kwargs) -> bool:
        calls.append(
            (
                label,
                bool(kwargs.get("avoid_collisions", True)),
                float(kwargs.get("min_fraction", 0.9)),
                bool(kwargs.get("allow_partial", False)),
            )
        )
        assert target.position.x == 0.4
        assert target.position.y == -0.2
        assert target.position.z == 1.2
        assert target.orientation is orientation
        return label.endswith("(no-collision)")

    controller._cartesian_move = fake_cartesian_move

    result = controller.move_relative(0.0, 0.0, 0.1, speed=0.55)

    assert result == {"success": True, "message": "moved relative (0.0, 0.0, 0.1)"}
    assert calls == [
        ("move_relative(dx=0.0, dy=0.0, dz=0.1)", True, 0.9, False),
        ("move_relative(dx=0.0, dy=0.0, dz=0.1) (no-collision)", False, 0.7, True),
    ]
    assert logger.warn_messages == [
        "move_relative vertical fallback: retrying no-collision move for dz=0.1000"
    ]


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
