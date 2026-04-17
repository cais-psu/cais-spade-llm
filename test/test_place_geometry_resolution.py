from __future__ import annotations

import json
from pathlib import Path

from cais_spade_llm.resources.robot.place_geometry_resolution import (
    has_place_geometry_fields,
    resolve_place_geometry,
)


_ROOT = Path(__file__).resolve().parents[1]
_DRY_RUN_RESOURCE_CONFIGS = [
    _ROOT / "cais_spade_llm" / "initialization" / "resources" / "robot_ur5e_2.json",
    _ROOT / "cais_spade_llm" / "initialization" / "resources" / "robot_xarm6_2.json",
    _ROOT / "writing" / "experiments" / "resources" / "robot_ur5e_2.json",
    _ROOT / "writing" / "experiments" / "resources" / "robot_xarm6_2.json",
]
_PRIMARY_RESOURCE_CONFIGS = [
    _ROOT / "cais_spade_llm" / "initialization" / "resources" / "robot_ur5e.json",
    _ROOT / "cais_spade_llm" / "initialization" / "resources" / "robot_xarm6.json",
]


def test_known_printer_tokens_resolve_geometry() -> None:
    expected_centers = {
        "prusa-mk4-1": {"x": 0.4, "y": 0.3, "z": 1.04},
        "prusa-mk4-2": {"x": 0.4, "y": -0.3, "z": 1.04},
        "prusa-mk3": {"x": -0.4, "y": 0.0, "z": 1.04},
        "prusa-mk3-2": {"x": -0.45, "y": -0.25, "z": 1.04},
    }

    for token, expected_center in expected_centers.items():
        geometry = resolve_place_geometry(
            part_name="MCP",
            destination_location=token,
            execution_mode="simulation",
        )
        assert has_place_geometry_fields(geometry), token
        assert geometry["board_center"] == expected_center
        assert geometry["slot_xy"] == [0.0, 0.0]
        assert geometry["slot_floor_z_m"] == 1.04
        assert geometry["model_name"] == "circ_pin_medium"
        assert geometry["part_height_m"] == 0.08


def test_destination_suffix_normalizes_to_same_geometry() -> None:
    base = resolve_place_geometry(
        part_name="MCP",
        destination_location="prusa-mk4-2",
        execution_mode="simulation",
    )
    with_suffix = resolve_place_geometry(
        part_name="MCP",
        destination_location="prusa-mk4-2@intermediate_anchor",
        execution_mode="simulation",
    )

    assert with_suffix == base


def test_assembly_board_geometry_regression() -> None:
    geometry = resolve_place_geometry(
        part_name="MCP",
        destination_location="assembly_board-v1",
        execution_mode="simulation",
    )

    assert has_place_geometry_fields(geometry)
    assert geometry["board_center"] == {"x": 0.0, "y": 0.0, "z": 1.02}
    assert geometry["slot_xy"] == [0.0, -0.08]
    assert geometry["slot_floor_z_m"] == 1.025
    assert geometry["model_name"] == "circ_pin_medium"
    assert geometry["part_height_m"] == 0.08


def test_prusa_mk3_2_dry_run_configs_share_canonical_anchor() -> None:
    expected_center = {"x": -0.45, "y": -0.25, "z": 1.04}

    for path in _DRY_RUN_RESOURCE_CONFIGS:
        payload = json.loads(path.read_text(encoding="utf-8"))
        (_, resource_meta), = payload.items()
        staging = (
            resource_meta.get("gazebo", {})
            .get("static_capabilities", {})
            .get("staging_areas", {})
        )
        printer = staging.get("prusa-mk3-2")
        if not isinstance(printer, dict):
            continue
        assert printer.get("anchor_pose") == expected_center, path
        assert printer.get("board_center") == expected_center, path


def test_printer_staging_anchors_fit_workspace_bounds() -> None:
    for path in _PRIMARY_RESOURCE_CONFIGS + _DRY_RUN_RESOURCE_CONFIGS:
        payload = json.loads(path.read_text(encoding="utf-8"))
        (_, resource_meta), = payload.items()
        static_caps = (
            resource_meta.get("gazebo", {})
            .get("static_capabilities", {})
        )
        bounds = static_caps.get("workspace_bounds", {})
        staging = static_caps.get("staging_areas", {})
        for token, area in staging.items():
            if not str(token).startswith("prusa-") or not isinstance(area, dict):
                continue
            board_center = area.get("board_center", {})
            assert bounds.get("x_min_m") <= board_center.get("x") <= bounds.get("x_max_m"), (
                path,
                token,
                board_center,
                bounds,
            )
            assert bounds.get("y_min_m") <= board_center.get("y") <= bounds.get("y_max_m"), (
                path,
                token,
                board_center,
                bounds,
            )
            assert bounds.get("z_min_m") <= board_center.get("z") <= bounds.get("z_max_m"), (
                path,
                token,
                board_center,
                bounds,
            )
