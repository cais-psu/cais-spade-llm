from __future__ import annotations

"""Calculate selected vertical helpers from explicit inputs and frozen robot context."""

import math
from collections.abc import Mapping
from copy import deepcopy
from typing import Any

import numpy as np

from cais_spade_llm.resources.robot.target_calculations import (
    controlled_link_height,
    pick_travel_height,
    placement_poses,
    vertical_pick_bias,
)

from ..agents.ra.composition_context import _without_model_name
from .robot_validation_context import pose_matrix


class CalculationUnavailable(ValueError):
    """Report missing inputs or geometry outside the supported helper semantics."""


def _number(value: Any, label: str, *, positive: bool = False) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or (positive and value <= 0)
    ):
        raise CalculationUnavailable(
            f"{label} requires a finite{' positive' if positive else ''} measurement."
        )
    return float(value)


def calculate_target(
    primitive_symbol: str,
    params: Mapping[str, Any],
    robot_context: Mapping[str, Any],
    preceding_pose: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate one authored helper; do not infer inputs, invoke perception or move."""
    from cais_spade_llm.resources.robot.robot_primitives import ROBOT_EXTRACT_OUTPUT_MAP

    if primitive_symbol not in {"compute_pick_targets", "compute_place_targets"}:
        raise CalculationUnavailable("No audited calculation adapter exists for this primitive.")
    if robot_context.get("frame_id") != "world":
        raise CalculationUnavailable(
            "The existing vertical helpers require world geometry and world controlled-link feedback."
        )
    if any(key in params for key in ("model_name", "assembly_board_v1_aruco")):
        raise CalculationUnavailable(
            "Execution-only or unsupported physical helper inputs were supplied."
        )
    try:
        ee = pose_matrix(preceding_pose)
        tool = np.asarray(robot_context["ee_from_tcp"], dtype=float)
        if tool.shape != (4, 4) or not np.isfinite(tool).all():
            raise CalculationUnavailable("A measured full EE–TCP transform is required.")
        offset = (ee @ tool)[:3, 3] - ee[:3, 3]
        if not np.allclose(offset[:2], 0, atol=1e-6):
            raise CalculationUnavailable(
                "The vertical helper cannot represent this lateral EE–TCP offset at the proposed orientation."
            )
        geometry = params["product_geometry"]
        if not isinstance(geometry, Mapping) or "record_type" in geometry:
            raise CalculationUnavailable(
                "product_geometry must select actual geometry fields, not a raw record."
            )
        policy = robot_context["policy"]
        if primitive_symbol == "compute_pick_targets":
            result = _pick(params, geometry, policy, preceding_pose, float(offset[2]))
        else:
            result = _place(params, geometry, policy, float(offset[2]))
        output, error = ROBOT_EXTRACT_OUTPUT_MAP[primitive_symbol](
            dict(params), {"success": True, **result}
        )
        if error or output is None:
            raise CalculationUnavailable(error or "The helper output extractor returned no values.")
        return _without_model_name(output)
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        if isinstance(exc, CalculationUnavailable):
            raise
        raise CalculationUnavailable(
            f"Target calculation lacks a valid explicit input: {exc}"
        ) from exc


def _pick(
    params: Mapping[str, Any],
    geometry: Mapping[str, Any],
    policy: Mapping[str, Any],
    pose: Mapping[str, Any],
    tcp_offset_z: float,
) -> dict[str, Any]:
    if params.get("prefer_live_detection"):
        raise CalculationUnavailable(
            "Live detection is unavailable in the calculation stage; PA must supply approved observation evidence."
        )
    target = params.get("target_pose")
    if target is None:
        parts = params.get("detected_parts")
        if (
            not isinstance(parts, list)
            or len(parts) != 1
            or parts[0].get("part_name") != params.get("part_name")
        ):
            raise CalculationUnavailable(
                "An explicit target_pose or one bound observed part is required."
            )
        target = parts[0]
    x, y, z = (_number(target[key], "target_pose." + key) for key in ("x", "y", "z"))
    height = _number(geometry["part_height_m"], "part_height_m", positive=True)
    support_z = _number(geometry["board_center"]["z"], "support surface")
    bias = vertical_pick_bias(
        height, policy["pick_tcp_z_bias_min_m"], policy["pick_tcp_z_bias_max_m"]
    )
    clearance = max(
        0.0, _number(params.get("surface_clearance_override_m", 0.0), "surface clearance policy")
    )
    tcp_z = z + bias + clearance
    if params.get("min_pick_tcp_z_override_m") is not None:
        tcp_z = max(tcp_z, _number(params["min_pick_tcp_z_override_m"], "minimum TCP policy"))
    elif params.get("use_global_min_pick_tcp_z", True):
        tcp_z = max(tcp_z, _number(policy["min_pick_tcp_z_m"], "configured minimum TCP policy"))
    adjustment = (
        policy["pick_z_adjustments_m"].get(params.get("part_name", "").upper(), 0.0)
        if params.get("apply_pick_z_adjustments", True)
        else 0.0
    )
    pick_z = controlled_link_height(tcp_z, tcp_offset_z, adjustment)
    approach = _number(
        params.get("approach_height_override_m", policy["approach_height_m"]), "approach policy"
    )
    travel_z = pick_travel_height(
        z,
        support_z,
        pick_z,
        approach,
        None if params.get("ignore_current_height_for_travel_z", False) else pose["z"],
    )
    return {
        "part_name": params.get("part_name", ""),
        "tx": x,
        "ty": y,
        "tz": z,
        "pick_z": pick_z,
        "travel_z": travel_z,
        "part_height": height,
        "tcp_offset_z": tcp_offset_z,
        "pick_tcp_z": tcp_z,
        "start_x": pose["x"],
        "start_y": pose["y"],
        "start_z": pose["z"],
    }


def _place(
    params: Mapping[str, Any],
    geometry: Mapping[str, Any],
    policy: Mapping[str, Any],
    measured_offset: float,
) -> dict[str, Any]:
    if params.get("destination_location"):
        raise CalculationUnavailable(
            "A destination token has no approved resolver in this calculation stage."
        )
    pick = params["pick_ctx"]
    height = _number(
        geometry.get("part_height_m", pick["part_height"]), "part_height", positive=True
    )
    tcp_offset = _number(pick["tcp_offset_z"], "pick_ctx.tcp_offset_z")
    if not math.isclose(tcp_offset, measured_offset, abs_tol=1e-6):
        raise CalculationUnavailable(
            "The selected pick tool offset is incompatible with the proposed placement orientation."
        )
    grasp_offset = _number(pick["pick_tcp_z"], "pick_ctx.pick_tcp_z") - _number(
        pick["tz"], "pick_ctx.tz"
    )
    board = geometry["board_center"]
    slot = geometry["slot_xy"]
    x = _number(board["x"], "board_center.x") + _number(slot[0], "slot_xy[0]")
    y = _number(board["y"], "board_center.y") + _number(slot[1], "slot_xy[1]")
    support = _number(geometry["slot_floor_z_m"], "slot_floor_z_m")
    reference = geometry["target_reference"]
    if reference["target_point"] not in {"part_origin", "inserted_part_origin"}:
        raise CalculationUnavailable(
            "Placement requires a resolved final part-origin relationship; the support-height fallback is not admitted."
        )
    origin = geometry["target_origin_pose"]
    if pick.get("origin_pose"):
        # The runtime gives this field precedence when no destination is supplied.
        # Reject the ambiguous mode instead of silently changing RA's parameters.
        raise CalculationUnavailable(
            "pick_ctx.origin_pose would override the selected placement target in the runtime helper."
        )
    if not math.isclose(
        _number(origin["x"], "target_origin_pose.x"), x, abs_tol=1e-6
    ) or not math.isclose(_number(origin["y"], "target_origin_pose.y"), y, abs_tol=1e-6):
        raise CalculationUnavailable(
            "The final part origin and selected slot coordinates disagree."
        )
    origin_z = _number(origin["z"], "target_origin_pose.z")
    tcp_z = origin_z + grasp_offset
    place_z = controlled_link_height(
        tcp_z, tcp_offset, _number(params.get("z_adjustment_m", 0.0), "z adjustment policy")
    )
    raw_handoff = pick.get("held_part_handoff", {})
    raw_pose = raw_handoff.get("world_tool0_pose_at_grasp") or pick.get(
        "resolved_cartesian_positions", {}
    ).get("descend", {})
    orientation = {}
    if raw_pose:
        pose_matrix(raw_pose)
        orientation = {key: raw_pose[key] for key in ("qx", "qy", "qz", "qw")}
    poses = placement_poses(
        x,
        y,
        place_z,
        orientation,
        simulation_assembly_slot=reference.get("surface_role") == "assembly_slot",
        insertion_depth=policy["insertion_depth_m"],
    )
    return {
        "part_name": params.get("part_name", pick.get("part_name", "")),
        "slot_x": x,
        "slot_y": y,
        "board_top_z": support,
        "place_z": place_z,
        "place_tcp_z": tcp_z,
        "place_part_origin_z": origin_z,
        "part_height": height,
        "tcp_offset_z": tcp_offset,
        "grasp_tcp_to_part_origin_z": grasp_offset,
        "target_reference": deepcopy(reference),
        "target_origin_pose": deepcopy(origin),
        "insertion_axis_world": {"x": 0.0, "y": 0.0, "z": -1.0},
        **poses,
    }
