"""Focused tests for live detection input to compute_pick_targets."""

from __future__ import annotations

import inspect
import time
from types import SimpleNamespace
from typing import Any

import pytest

from cais_spade_llm.resources.robot.gazebo_pick_place_controller import (
    GazeboPickPlaceController,
)


class _Logger:
    def info(self, *_args: Any, **_kwargs: Any) -> None:
        return None


def _controller_double(*, execution_mode: str) -> tuple[GazeboPickPlaceController, dict[str, int]]:
    controller = object.__new__(GazeboPickPlaceController)
    calls = {"detect_parts": 0, "get_ee_pose": 0}

    controller.execution_mode = execution_mode
    controller.wait_for_services = lambda: True
    controller.pick_tcp_z_bias_min_m = 0.003
    controller.pick_tcp_z_bias_max_m = 0.020
    controller.min_pick_tcp_z_m = 1.070
    controller.pick_z_adjustments_m = {}
    controller.approach_height_m = 0.200
    controller._derive_gripper_close_position = lambda **_kwargs: 0.02
    controller._get_ee_tcp_world_z_offset = lambda: -0.17
    controller._make_pose = lambda x, y, z, orientation: (x, y, z, orientation)
    controller._log = lambda: _Logger()

    def get_ee_pose() -> Any:
        calls["get_ee_pose"] += 1
        return SimpleNamespace(
            position=SimpleNamespace(x=0.0, y=0.0, z=1.25),
            orientation=SimpleNamespace(x=0.0, y=1.0, z=0.0, w=0.0),
        )

    def detect_parts(_part_name: str | None = None) -> list[dict[str, Any]]:
        calls["detect_parts"] += 1
        return []

    controller._get_ee_pose = get_ee_pose
    controller.detect_parts = detect_parts
    return controller, calls


def _physical_detection(**overrides: Any) -> dict[str, Any]:
    row = {
        "part_name": "MG",
        "model_name": "gear_medium",
        "x": 0.41,
        "y": 0.31,
        "z": 1.021,
        "qx": 0.5,
        "qy": 0.5,
        "qz": 0.5,
        "qw": 0.5,
        "frame_id": "world",
        "confidence": 0.96,
        "captured_at": time.time(),
        "table_surface_z_m": 1.011,
    }
    row.update(overrides)
    return row


def test_compute_pick_targets_consumes_detected_parts_without_second_detection() -> None:
    controller, calls = _controller_double(execution_mode="physical")
    physical_detection = _physical_detection()
    readiness_calls = {"init": 0, "full": 0, "motion": 0}

    def init() -> bool:
        readiness_calls["init"] += 1
        return True

    def reject_full_service_readiness() -> bool:
        readiness_calls["full"] += 1
        raise AssertionError("supplied detection preview must not wait for motion services")

    def reject_motion(*_args: Any, **_kwargs: Any) -> bool:
        readiness_calls["motion"] += 1
        raise AssertionError("compute_pick_targets must not command motion")

    controller.init = init
    controller.wait_for_services = reject_full_service_readiness
    controller._cartesian_move = reject_motion

    result = controller.compute_pick_targets(
        part_name="MG",
        product_geometry={"part_height_m": 0.02, "model_name": "gear_medium"},
        detected_parts=[physical_detection],
    )

    assert result["success"] is True
    assert readiness_calls == {"init": 1, "full": 0, "motion": 0}
    assert calls["detect_parts"] == 0
    assert calls["get_ee_pose"] == 1
    assert controller._last_start_pose[3] == SimpleNamespace(x=0.0, y=1.0, z=0.0, w=0.0)
    assert result["tx"] == pytest.approx(0.41)
    assert result["ty"] == pytest.approx(0.31)
    assert result["tz"] == pytest.approx(1.021)
    assert not {"qx", "qy", "qz", "qw"} & set(result)
    assert result["confidence"] == pytest.approx(0.96)
    assert result["captured_at"] == pytest.approx(physical_detection["captured_at"])
    assert result["frame_id"] == "world"
    assert result["table_surface_z_m"] == pytest.approx(1.011)


def test_compute_pick_targets_without_detected_parts_keeps_service_lookup_compatible() -> None:
    controller, calls = _controller_double(execution_mode="simulation")
    controller.detect_parts = lambda _part_name=None: calls.__setitem__(
        "detect_parts", calls["detect_parts"] + 1
    ) or [
        {
            "part_name": "MG",
            "model_name": "gear_medium",
            "x": 0.4,
            "y": 0.3,
            "z": 1.02,
        }
    ]

    result = controller.compute_pick_targets(part_name="MG")

    assert result["success"] is True
    assert calls["detect_parts"] == 1


def test_detected_parts_is_appended_without_shifting_existing_parameters() -> None:
    parameter_names = list(
        inspect.signature(GazeboPickPlaceController.compute_pick_targets).parameters
    )

    assert parameter_names[:5] == [
        "self",
        "part_name",
        "product_geometry",
        "target_pose",
        "target_pose_source",
    ]
    assert parameter_names[-1] == "detected_parts"


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ([_physical_detection(part_name="SG")], "did not match exactly one detection"),
        ([_physical_detection(captured_at=time.time() - 11.0)], "is stale"),
        ([_physical_detection(x=float("nan"))], "x coordinate is not finite"),
        ([_physical_detection(frame_id="base")], "frame_id must be world"),
        ([_physical_detection(table_surface_z_m=None)], "no accepted table-plane evidence"),
        ([_physical_detection(table_plane_ready=False)], "table-plane evidence is not ready"),
    ],
)
def test_physical_detected_parts_fail_before_pose_use(
    rows: list[dict[str, Any]],
    message: str,
) -> None:
    controller, calls = _controller_double(execution_mode="physical")
    if message != "is stale":
        for row in rows:
            row["captured_at"] = time.time()

    result = controller.compute_pick_targets(part_name="MG", detected_parts=rows)

    assert result["success"] is False
    assert message in result["message"]
    assert calls["detect_parts"] == 0
    assert calls["get_ee_pose"] == 0


def test_physical_service_detection_is_validated_when_no_list_is_supplied() -> None:
    controller, calls = _controller_double(execution_mode="physical")
    controller.detect_parts = lambda _part_name=None: calls.__setitem__(
        "detect_parts", calls["detect_parts"] + 1
    ) or [_physical_detection(frame_id="camera_color_optical_frame")]

    result = controller.compute_pick_targets(part_name="MG")

    assert result["success"] is False
    assert "frame_id must be world" in result["message"]
    assert calls["detect_parts"] == 1
    assert calls["get_ee_pose"] == 0


def test_physical_live_detection_mismatch_does_not_fall_back_to_target_pose() -> None:
    controller, calls = _controller_double(execution_mode="physical")
    controller.detect_parts = lambda _part_name=None: calls.__setitem__(
        "detect_parts", calls["detect_parts"] + 1
    ) or [_physical_detection(part_name="SG")]

    result = controller.compute_pick_targets(
        part_name="MG",
        target_pose={"x": 0.4, "y": 0.3, "z": 1.02},
        prefer_live_detection=True,
    )

    assert result["success"] is False
    assert "did not match exactly one detection" in result["message"]
    assert calls["detect_parts"] == 1
    assert calls["get_ee_pose"] == 0


def test_physical_xyz_target_pose_remains_compatible_without_detection_metadata() -> None:
    controller, calls = _controller_double(execution_mode="physical")

    result = controller.compute_pick_targets(
        part_name="MG",
        target_pose={"x": 0.4, "y": 0.3, "z": 1.02},
        target_pose_source="observed_pose",
    )

    assert result["success"] is True
    assert calls["detect_parts"] == 0
    assert calls["get_ee_pose"] == 1
