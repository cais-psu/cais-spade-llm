"""Focused contracts for the actual-STL physical MG hub grasp."""

from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from cais_spade_llm.product.order import part_place_geometry
from cais_spade_llm.product.profile import ProductProfile
from cais_spade_llm.product.stl_geometry import actual_mg_stl_geometry_for_part
from cais_spade_llm.resources.robot import gazebo_pick_place_controller
from cais_spade_llm.resources.robot.hardware_pick_place_controller import (
    UR5eHardwareController,
)
from cais_spade_llm.resources.robot.robot_task_runtime import _normalize_pick_targets

ROOT = Path(__file__).resolve().parents[1]
PRODUCT_GEOMETRY = ROOT / "cais_spade_llm/specification/products/geometry/assembly_board-v1.json"
UR5E_MANIFEST = ROOT / "cais_spade_llm/initialization/resources/robot_ur5e.json"


class _Logger:
    def info(self, *_args: Any, **_kwargs: Any) -> None:
        return None


def _actual_mg_geometry() -> dict[str, Any]:
    geometry = ProductProfile.load_product_geometry(str(PRODUCT_GEOMETRY), robot_env="real")
    return ProductProfile.geometry_for_part_from_geometry("MG", geometry)


def _controller_double() -> UR5eHardwareController:
    manifest = json.loads(UR5E_MANIFEST.read_text(encoding="utf-8"))
    controller_config = manifest["ur5e"]["real"]["controller"]
    controller = object.__new__(UR5eHardwareController)
    controller.execution_mode = "physical"
    controller.controller_config = controller_config
    controller.gripper_open = float(controller_config["gripper"]["open"])
    controller.gripper_close = float(controller_config["gripper"]["close"])
    controller.pick_tcp_z_bias_min_m = 0.003
    controller.pick_tcp_z_bias_max_m = 0.020
    controller.min_pick_tcp_z_m = 1.070
    controller.pick_z_adjustments_m = {
        key: float(value)
        for key, value in controller_config["parts_tuning"][
            "pick_z_adjustments_m"
        ].items()
    }
    controller.pick_tool0_z_adjustment_m = 0.005
    controller.approach_height_m = 0.200
    controller._last_failure_message = ""
    controller.init = lambda: True
    controller._get_ee_tcp_world_z_offset = lambda: -0.218
    controller._make_pose = lambda x, y, z, orientation: (x, y, z, orientation)
    controller._get_ee_pose = lambda: SimpleNamespace(
        position=SimpleNamespace(x=0.4, y=0.3, z=1.294),
        orientation=SimpleNamespace(x=0.0, y=1.0, z=0.0, w=0.0),
    )
    controller._log = lambda: _Logger()
    return controller


def _physical_detection(table_surface_z_m: float = 1.0116499062) -> dict[str, Any]:
    return {
        "part_name": "MG",
        "model_name": "gear_medium",
        "x": 0.41,
        "y": 0.31,
        "z": table_surface_z_m + 0.01,
        "frame_id": "world",
        "confidence": 0.97,
        "captured_at": time.time(),
        "table_surface_z_m": table_surface_z_m,
        "table_plane_ready": True,
        "table_plane_accepted": True,
    }


def test_real_mg_geometry_is_derived_from_actual_source_stl_only() -> None:
    gazebo_geometry = ProductProfile.load_product_geometry(
        str(PRODUCT_GEOMETRY), robot_env="gazebo"
    )
    real_mg = _actual_mg_geometry()
    order_mg = part_place_geometry(
        "MG",
        ProductProfile.load_product_geometry(str(PRODUCT_GEOMETRY), robot_env="real"),
    )
    gazebo_mg = ProductProfile.geometry_for_part_from_geometry("MG", gazebo_geometry)

    assert real_mg["part_name"] == "MG"
    assert real_mg["model_name"] == "gear_medium"
    assert real_mg["source_stl"] == str(
        (ROOT / "ros2/cais_lab_robotics/cad_models/Gear_Medium.STL").resolve()
    )
    assert real_mg["source_stl_sha256"] == (
        "73d2c5d06497db2042ec6624a45558398ee47c9e55ccd87d49184c179a501507"
    )
    assert real_mg["hub_up"] is True
    assert real_mg["part_height_m"] == pytest.approx(0.020, abs=2e-6)
    assert real_mg["hub_diameter_m"] == pytest.approx(0.030, abs=2e-6)
    assert real_mg["hub_height_m"] == pytest.approx(0.010, abs=2e-6)
    assert real_mg["tooth_diameter_m"] == pytest.approx(0.042, abs=2e-6)
    assert real_mg["grasp_width_m"] == pytest.approx(0.028, abs=2e-6)
    assert order_mg["source_stl_sha256"] == real_mg["source_stl_sha256"]
    assert order_mg["grasp_width_m"] == pytest.approx(real_mg["grasp_width_m"])
    assert "source_stl" not in gazebo_mg
    assert "grasp_width_m" not in gazebo_mg


def test_missing_actual_mg_stl_fails_closed(tmp_path: Path) -> None:
    parts = {
        "source_stl_map": {"MG": str(tmp_path / "missing.STL")},
        "source_stl_scale_m_per_unit": {"MG": 0.001},
        "hub_up": {"MG": True},
        "grasp_width_preload_m": {"MG": 0.002},
        "tooth_clearance_m": {"MG": 0.002},
        "minimum_hub_overlap_m": {"MG": 0.006},
    }

    with pytest.raises(ValueError, match="actual MG STL is unavailable"):
        actual_mg_stl_geometry_for_part("MG", parts)


def test_incorrect_actual_mg_stl_scale_fails_closed() -> None:
    parts = {
        "source_stl_map": {"MG": "ros2/cais_lab_robotics/cad_models/Gear_Medium.STL"},
        "source_stl_scale_m_per_unit": {"MG": 1.0},
        "hub_up": {"MG": True},
        "grasp_width_preload_m": {"MG": 0.002},
        "tooth_clearance_m": {"MG": 0.002},
        "minimum_hub_overlap_m": {"MG": 0.006},
    }

    with pytest.raises(ValueError, match="outside the physical MG range"):
        actual_mg_stl_geometry_for_part("MG", parts)


def test_physical_mg_target_uses_stl_hub_and_never_gazebo_footprint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        gazebo_pick_place_controller,
        "_model_footprint_width_from_gazebo_world",
        lambda _model: pytest.fail("physical MG must not read Gazebo model geometry"),
    )
    controller = _controller_double()
    table_surface_z_m = 1.0116499062

    result = controller.compute_pick_targets(
        part_name="MG",
        product_geometry=_actual_mg_geometry(),
        detected_parts=[_physical_detection(table_surface_z_m)],
    )

    assert result["success"] is True
    assert result["use_global_min_pick_tcp_z"] is False
    assert result["effective_min_pick_tcp_z"] is None
    assert result["pick_tcp_z"] == pytest.approx(table_surface_z_m + 0.02165, abs=2e-6)
    assert result["pick_z"] == pytest.approx(table_surface_z_m + 0.24465, abs=2e-6)
    assert result["pick_z_adjustment_m"] == pytest.approx(0.001)
    assert result["pick_tool0_z_adjustment_m"] == pytest.approx(0.005)
    assert result["grasp_width_m"] == pytest.approx(0.028, abs=2e-6)
    assert result["gripper_close_position"] == pytest.approx(0.047, abs=2e-6)
    assert result["finger_tooth_clearance_m"] == pytest.approx(0.003, abs=2e-6)
    assert result["finger_hub_overlap_m"] == pytest.approx(0.007, abs=2e-6)
    assert result["open_inner_pad_lower_z_from_tcp_m"] == pytest.approx(0.01751)
    assert result["closed_inner_pad_lower_z_from_tcp_m"] == pytest.approx(-0.00865)
    assert result["predicted_closing_z_displacement_m"] == pytest.approx(
        -0.02616,
        abs=5e-6,
    )
    assert _normalize_pick_targets(result)["target_pose"]["z"] == pytest.approx(
        result["pick_z"]
    )


def test_pick_tool0_z_adjustment_is_validated_and_enabled_only_for_real_ur5e() -> None:
    manifest = json.loads(UR5E_MANIFEST.read_text(encoding="utf-8"))
    gazebo_motion = manifest["ur5e"]["gazebo"]["controller"]["motion"]
    real_controller_config = manifest["ur5e"]["real"]["controller"]

    assert "pick_tool0_z_adjustment_m" not in gazebo_motion
    assert real_controller_config["motion"]["pick_tool0_z_adjustment_m"] == pytest.approx(
        0.005
    )

    gazebo_controller = gazebo_pick_place_controller.UR5eGazeboController(
        controller_config=manifest["ur5e"]["gazebo"]["controller"]
    )
    assert gazebo_controller._config_valid is True
    assert gazebo_controller.pick_tool0_z_adjustment_m == pytest.approx(0.0)

    controller = UR5eHardwareController(controller_config=real_controller_config)
    assert controller._config_valid is True
    assert controller.pick_tool0_z_adjustment_m == pytest.approx(0.005)

    for invalid_value in ("not-a-number", float("inf")):
        invalid_controller_config = json.loads(json.dumps(real_controller_config))
        invalid_controller_config["motion"]["pick_tool0_z_adjustment_m"] = invalid_value
        invalid_controller = UR5eHardwareController(
            controller_config=invalid_controller_config
        )
        assert invalid_controller._config_valid is False
        assert (
            "controller.motion.pick_tool0_z_adjustment_m"
            in invalid_controller._config_errors
        )


def test_physical_mg_requires_actual_stl_and_rejects_unsafe_height_override() -> None:
    controller = _controller_double()
    geometry = _actual_mg_geometry()
    no_stl = dict(geometry)
    no_stl.pop("source_stl")

    missing = controller.compute_pick_targets(
        part_name="MG",
        product_geometry=no_stl,
        detected_parts=[_physical_detection()],
    )
    unsafe = controller.compute_pick_targets(
        part_name="MG",
        product_geometry=geometry,
        detected_parts=[_physical_detection()],
        min_pick_tcp_z_override_m=1.0116499062 + 0.023,
    )

    assert missing["success"] is False
    assert "actual source_stl" in missing["message"]
    assert unsafe["success"] is False
    assert "insufficient smooth hub overlap" in unsafe["message"]


def test_rg2_width_mapping_produces_expected_action_position() -> None:
    controller = _controller_double()

    readiness = controller._physical_stl_pick_readiness(_actual_mg_geometry())

    assert readiness["success"] is True
    assert readiness["grasp_width_m"] == pytest.approx(0.028, abs=2e-6)
    assert readiness["gripper_close_position"] == pytest.approx(0.047, abs=2e-6)
    assert readiness["open_gripper_position"] == pytest.approx(0.11)
    assert readiness["predicted_closing_z_displacement_m"] == pytest.approx(
        -0.02616,
        abs=5e-6,
    )
    assert readiness["finger_tooth_clearance_m"] == pytest.approx(0.003, abs=2e-6)
    assert readiness["finger_hub_overlap_m"] == pytest.approx(0.007, abs=2e-6)


def test_combined_mount_and_mg_adjustments_raise_only_the_arm_target_by_6_mm() -> None:
    controller = _controller_double()
    corrected = controller.compute_pick_targets(
        part_name="MG",
        product_geometry=_actual_mg_geometry(),
        detected_parts=[_physical_detection()],
    )
    controller.pick_z_adjustments_m = {}
    controller.pick_tool0_z_adjustment_m = 0.0
    restored_baseline = controller.compute_pick_targets(
        part_name="MG",
        product_geometry=_actual_mg_geometry(),
        detected_parts=[_physical_detection()],
    )

    assert corrected["success"] is True
    assert restored_baseline["success"] is True
    assert corrected["pick_z"] - restored_baseline["pick_z"] == pytest.approx(0.006)
    assert corrected["gripper_close_position"] == pytest.approx(
        restored_baseline["gripper_close_position"]
    )


@pytest.mark.parametrize(
    ("field_name", "invalid_value", "message"),
    [
        ("open_gripper_position", 0.10, "position 0.11"),
        ("mg_gripper_close_position", 0.05, "approximately 0.047"),
        (
            "open_inner_pad_lower_z_from_tcp_m",
            0.010,
            "approximately -0.02616 m",
        ),
    ],
)
def test_mg_readiness_rejects_mismatched_fixed_rg2_calibration(
    field_name: str,
    invalid_value: float,
    message: str,
) -> None:
    controller = _controller_double()
    controller.controller_config = json.loads(json.dumps(controller.controller_config))
    controller.controller_config["gripper"]["stock_fingertip"][field_name] = invalid_value

    result = controller._physical_stl_pick_readiness(_actual_mg_geometry())

    assert result["success"] is False
    assert message in result["message"]


def test_mg_adjustment_above_2_mm_fails_readiness() -> None:
    controller = _controller_double()
    controller.pick_z_adjustments_m["MG"] = 0.003

    result = controller._physical_stl_pick_readiness(_actual_mg_geometry())

    assert result["success"] is False
    assert "between 0.000 and 0.002 m" in result["message"]
