"""Focused tests for physical assembly_board-v1 ArUco localization."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path

import pytest
import yaml

from cais_spade_llm.resources.robot import (
    gazebo_pick_place_controller as controller_module,
)
from cais_spade_llm.resources.robot.gazebo_pick_place_controller import (
    GazeboPickPlaceController,
)


class _FakeClock:
    """Advance controller polling deterministically without waiting in real time."""

    def __init__(self, on_sleep: Callable[[float], None] | None = None) -> None:
        self.wall_start = time.time()
        self.elapsed_sec = 0.0
        self.on_sleep = on_sleep

    def time(self) -> float:
        return self.wall_start + self.elapsed_sec

    def monotonic(self) -> float:
        return self.elapsed_sec

    def sleep(self, duration_sec: float) -> None:
        self.elapsed_sec += duration_sec
        if self.on_sleep is not None:
            self.on_sleep(self.elapsed_sec)


def _pose(*, x: float = 0.4, yaw_quarter_turn: bool = False) -> dict[str, float]:
    if yaw_quarter_turn:
        return {
            "x": x,
            "y": 0.2,
            "z": 1.02,
            "qx": 0.0,
            "qy": 0.0,
            "qz": 2**-0.5,
            "qw": 2**-0.5,
        }
    return {
        "x": x,
        "y": 0.2,
        "z": 1.02,
        "qx": 0.0,
        "qy": 0.0,
        "qz": 0.0,
        "qw": 1.0,
    }


def _controller(tmp_path: Path, *, robot: str = "ur5e") -> GazeboPickPlaceController:
    controller = object.__new__(GazeboPickPlaceController)
    controller.robot_name = robot
    controller.execution_mode = "physical"
    controller._assembly_board_v1_aruco_snapshot_path = tmp_path / "snapshot.json"
    controller._assembly_board_v1_aruco_config_path = tmp_path / "perception_cameras.yaml"
    controller._assembly_board_v1_aruco_wait_sec = 0.0
    controller._assembly_board_v1_aruco_config_path.write_text(
        yaml.safe_dump(
            {
                "assembly_board-v1_aruco": {
                    "marker_length_m": 0.076,
                    "roles": {
                        robot: {
                            "accepted_generation": 3,
                            "accepted_pose": _pose(),
                            "accepted_at": time.time() - 5.0,
                            "frame_captured_at": time.time() - 5.0,
                            "calibration_id": f"{robot}-calibration",
                        }
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    return controller


def _snapshot(
    *,
    robot: str = "ur5e",
    pose: dict[str, float] | None = None,
    timestamp: float | None = None,
) -> dict[str, object]:
    captured_at = time.time() + 0.1 if timestamp is None else timestamp
    return {
        "camera_role": robot,
        "resource_location": "assembly_board-v1",
        "marker_dictionary": "DICT_ARUCO_ORIGINAL",
        "marker_id": 70,
        "marker_length_m": 0.076,
        "visible": True,
        "valid": True,
        "stable": True,
        "world_pose_ready": True,
        "pose_ambiguous": False,
        "sample_count": 10,
        "sample_started_at": captured_at,
        "frame_captured_at": captured_at,
        "reprojection_error_px": 0.2,
        "translation_spread_m": 0.001,
        "rotation_spread_deg": 0.2,
        "frame_id": "world",
        "pose": pose or _pose(x=0.405),
        "calibration_id": f"{robot}-calibration",
        "last_error": "",
    }


def test_localize_assembly_board_v1_returns_frozen_role_pose(tmp_path: Path) -> None:
    controller = _controller(tmp_path)
    Path(controller._assembly_board_v1_aruco_snapshot_path).write_text(
        json.dumps(_snapshot()), encoding="utf-8"
    )

    result = controller.localize_assembly_board_v1("assembly_board-v1", "MG")

    assert result["success"] is True
    assert result["camera_role"] == "ur5e"
    assert result["generation"] == 3
    assert result["marker_dictionary"] == "DICT_ARUCO_ORIGINAL"
    assert result["marker_id"] == 70
    assert result["marker_length_m"] == pytest.approx(0.076)
    assert result["translation_delta_m"] == pytest.approx(0.005)
    assert result["pose"]["x"] == pytest.approx(0.405)
    assert result["part_name"] == "MG"


def test_localize_assembly_board_v1_rejects_large_board_movement(tmp_path: Path) -> None:
    controller = _controller(tmp_path, robot="xarm6")
    Path(controller._assembly_board_v1_aruco_snapshot_path).write_text(
        json.dumps(_snapshot(robot="xarm6", pose=_pose(x=0.425))), encoding="utf-8"
    )

    result = controller.localize_assembly_board_v1("assembly_board-v1", "SG")

    assert result["success"] is False
    assert "moved 25.0 mm" in result["message"]
    assert "Locate & Accept Board" in result["message"]


def test_localize_assembly_board_v1_rejects_latched_movement_block(
    tmp_path: Path,
) -> None:
    controller = _controller(tmp_path, robot="xarm6")
    config_path = Path(controller._assembly_board_v1_aruco_config_path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload["assembly_board-v1_aruco"]["roles"]["xarm6"][
        "movement_blocked"
    ] = True
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    result = controller.localize_assembly_board_v1("assembly_board-v1", "SG")

    assert result["success"] is False
    assert "moved more than 10 mm or 2 deg" in result["message"]
    assert "Locate & Accept Board again" in result["message"]


def test_localize_assembly_board_v1_without_callback_rejects_missing_baseline(
    tmp_path: Path,
) -> None:
    controller = _controller(tmp_path)
    config_path = Path(controller._assembly_board_v1_aruco_config_path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload["assembly_board-v1_aruco"]["roles"]["ur5e"] = {}
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    result = controller.localize_assembly_board_v1("assembly_board-v1", "MG")

    assert result["success"] is False
    assert "has no accepted assembly_board-v1 ArUco pose" in result["message"]


def test_post_staging_callback_accepts_missing_board_baseline(tmp_path: Path) -> None:
    controller = _controller(tmp_path)
    config_path = Path(controller._assembly_board_v1_aruco_config_path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload["assembly_board-v1_aruco"]["roles"]["ur5e"] = {}
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    snapshot = _snapshot()
    Path(controller._assembly_board_v1_aruco_snapshot_path).write_text(
        json.dumps(snapshot), encoding="utf-8"
    )
    callback_calls: list[float] = []

    def accept_after_staging(requested_at: float) -> None:
        callback_calls.append(requested_at)
        assert float(snapshot["sample_started_at"]) >= requested_at
        current = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        current["assembly_board-v1_aruco"]["roles"]["ur5e"] = {
            "accepted_generation": 1,
            "accepted_pose": snapshot["pose"],
            "accepted_at": time.time(),
            "frame_captured_at": snapshot["frame_captured_at"],
            "calibration_id": "ur5e-calibration",
            "movement_blocked": False,
        }
        config_path.write_text(yaml.safe_dump(current), encoding="utf-8")

    controller._assembly_board_v1_post_staging_accept_callback = accept_after_staging

    result = controller.localize_assembly_board_v1("assembly_board-v1", "MG")

    assert result["success"] is True
    assert result["generation"] == 1
    assert result["translation_delta_m"] == pytest.approx(0.0)
    assert len(callback_calls) == 1


@pytest.mark.parametrize("movement_blocked", [False, True])
def test_post_staging_callback_reaccepts_large_or_latched_board_movement(
    tmp_path: Path,
    movement_blocked: bool,
) -> None:
    controller = _controller(tmp_path, robot="xarm6")
    config_path = Path(controller._assembly_board_v1_aruco_config_path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload["assembly_board-v1_aruco"]["roles"]["xarm6"][
        "movement_blocked"
    ] = movement_blocked
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    snapshot = _snapshot(robot="xarm6", pose=_pose(x=0.425))
    Path(controller._assembly_board_v1_aruco_snapshot_path).write_text(
        json.dumps(snapshot), encoding="utf-8"
    )
    callback_calls: list[float] = []

    def reaccept_after_staging(requested_at: float) -> None:
        callback_calls.append(requested_at)
        assert float(snapshot["sample_started_at"]) >= requested_at
        current = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        role_config = current["assembly_board-v1_aruco"]["roles"]["xarm6"]
        role_config.update(
            {
                "accepted_generation": 4,
                "accepted_pose": snapshot["pose"],
                "accepted_at": time.time(),
                "frame_captured_at": snapshot["frame_captured_at"],
                "calibration_id": "xarm6-calibration",
                "movement_blocked": False,
            }
        )
        config_path.write_text(yaml.safe_dump(current), encoding="utf-8")

    controller._assembly_board_v1_post_staging_accept_callback = reaccept_after_staging

    result = controller.localize_assembly_board_v1("assembly_board-v1", "SG")

    assert result["success"] is True
    assert result["generation"] == 4
    assert result["translation_delta_m"] == pytest.approx(0.0)
    assert len(callback_calls) == 1


def test_post_staging_callback_is_not_used_for_usable_small_movement(
    tmp_path: Path,
) -> None:
    controller = _controller(tmp_path)
    Path(controller._assembly_board_v1_aruco_snapshot_path).write_text(
        json.dumps(_snapshot()), encoding="utf-8"
    )
    callback_calls: list[float] = []
    controller._assembly_board_v1_post_staging_accept_callback = callback_calls.append

    result = controller.localize_assembly_board_v1("assembly_board-v1", "MG")

    assert result["success"] is True
    assert result["generation"] == 3
    assert callback_calls == []


def test_default_wait_accepts_stable_window_arriving_after_eight_seconds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _controller(tmp_path)
    del controller._assembly_board_v1_aruco_wait_sec
    snapshot_path = Path(controller._assembly_board_v1_aruco_snapshot_path)
    stable_published = False

    def _publish_stable_window(elapsed_sec: float) -> None:
        nonlocal stable_published
        if stable_published or elapsed_sec < 8.5:
            return
        stable_published = True
        snapshot_path.write_text(
            json.dumps(_snapshot(timestamp=clock.time())),
            encoding="utf-8",
        )

    clock = _FakeClock(_publish_stable_window)
    unstable = _snapshot(timestamp=clock.time())
    unstable.update(
        {
            "valid": False,
            "stable": False,
            "world_pose_ready": False,
            "pose": None,
            "rotation_spread_deg": 0.829,
            "last_error": (
                "assembly_board-v1 ArUco pose is unstable: 0.300 mm / 0.829 deg"
            ),
        }
    )
    snapshot_path.write_text(json.dumps(unstable), encoding="utf-8")
    monkeypatch.setattr(controller_module, "time", clock)

    result = controller.localize_assembly_board_v1("assembly_board-v1", "MG")

    assert result["success"] is True
    assert stable_published is True
    assert 8.5 <= clock.elapsed_sec < 15.0


def test_default_wait_timeout_reports_final_orientation_instability(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _controller(tmp_path)
    del controller._assembly_board_v1_aruco_wait_sec
    clock = _FakeClock()
    unstable = _snapshot(timestamp=clock.time())
    instability = "assembly_board-v1 ArUco pose is unstable: 0.300 mm / 0.829 deg"
    unstable.update(
        {
            "valid": False,
            "stable": False,
            "world_pose_ready": False,
            "pose": None,
            "rotation_spread_deg": 0.829,
            "last_error": instability,
        }
    )
    Path(controller._assembly_board_v1_aruco_snapshot_path).write_text(
        json.dumps(unstable),
        encoding="utf-8",
    )
    monkeypatch.setattr(controller_module, "time", clock)

    result = controller.localize_assembly_board_v1("assembly_board-v1", "MG")

    assert result["success"] is False
    assert instability in result["message"]
    assert 15.0 <= clock.elapsed_sec < 15.1


def test_post_staging_callback_missing_tag_times_out_without_acceptance(
    tmp_path: Path,
) -> None:
    controller = _controller(tmp_path)
    config_path = Path(controller._assembly_board_v1_aruco_config_path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload["assembly_board-v1_aruco"]["roles"]["ur5e"][
        "movement_blocked"
    ] = True
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    snapshot = _snapshot()
    snapshot.update(
        {
            "valid": False,
            "visible": False,
            "stable": False,
            "world_pose_ready": False,
            "sample_count": 0,
            "last_error": "assembly_board-v1 ArUco ID 70 is not visible",
        }
    )
    Path(controller._assembly_board_v1_aruco_snapshot_path).write_text(
        json.dumps(snapshot), encoding="utf-8"
    )
    callback_calls: list[float] = []

    def reject_missing_tag(requested_at: float) -> None:
        callback_calls.append(requested_at)
        raise RuntimeError("assembly_board-v1 ArUco ID 70 is not visible")

    controller._assembly_board_v1_post_staging_accept_callback = reject_missing_tag

    result = controller.localize_assembly_board_v1("assembly_board-v1", "MG")

    assert result["success"] is False
    assert "after its observation pose" in result["message"]
    assert "ArUco ID 70 is not visible" in result["message"]
    assert len(callback_calls) == 1


def test_localize_assembly_board_v1_rejects_other_camera_role(tmp_path: Path) -> None:
    controller = _controller(tmp_path, robot="ur5e")
    Path(controller._assembly_board_v1_aruco_snapshot_path).write_text(
        json.dumps(_snapshot(robot="xarm6")), encoding="utf-8"
    )

    result = controller.localize_assembly_board_v1("assembly_board-v1", "MG")

    assert result["success"] is False
    assert "camera_role does not match ur5e" in result["message"]


def test_snapshot_window_must_be_entirely_after_observation_motion(tmp_path: Path) -> None:
    controller = _controller(tmp_path)
    acceptance, error = controller._assembly_board_v1_aruco_acceptance()
    assert error == ""
    requested_at = time.time()
    snapshot = _snapshot(timestamp=requested_at)
    snapshot["sample_started_at"] = requested_at - 0.01

    _validated, validation_error = (
        controller._validated_assembly_board_v1_aruco_snapshot(
            snapshot=snapshot,
            acceptance=acceptance,
            requested_at=requested_at,
            now=requested_at,
        )
    )

    assert validation_error == (
        "waiting for ten ArUco observations captured after staging motion"
    )


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("valid", False, "snapshot is not valid"),
        ("translation_spread_m", float("nan"), "non-finite numeric provenance"),
    ],
)
def test_snapshot_rejects_invalid_or_non_finite_authority(
    tmp_path: Path,
    field: str,
    value: object,
    expected: str,
) -> None:
    controller = _controller(tmp_path)
    acceptance, error = controller._assembly_board_v1_aruco_acceptance()
    assert error == ""
    requested_at = time.time()
    snapshot = _snapshot(timestamp=requested_at)
    snapshot[field] = value

    _validated, validation_error = (
        controller._validated_assembly_board_v1_aruco_snapshot(
            snapshot=snapshot,
            acceptance=acceptance,
            requested_at=requested_at,
            now=requested_at,
        )
    )

    assert expected in validation_error


def test_compute_place_targets_blocks_physical_fixed_board_fallback() -> None:
    controller = object.__new__(GazeboPickPlaceController)
    controller.execution_mode = "physical"
    controller.robot_name = "ur5e"
    controller.wait_for_services = lambda: True

    result = controller.compute_place_targets(
        part_name="MG",
        destination_location="assembly_board-v1",
    )

    assert result["success"] is False
    assert "requires a fresh frozen localize_assembly_board_v1 observation" in result["message"]
