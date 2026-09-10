from __future__ import annotations

"""Calculate vertical assembly targets using explicit geometry and measured tool transforms."""

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
from ..agents.ra.validation_scope import GAZEBO_OBSERVED_SCOPE, VALIDATION_SCOPE, read_validation_scope
from .robot_validation_context import matrix_pose, pose_matrix


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
    *,
    validation_scope: str = VALIDATION_SCOPE,
    held_part_transform: np.ndarray | None = None,
) -> dict[str, Any]:
    """Evaluate an authored helper without perception or motion.

    Args:
        primitive_symbol: The exact RA-selected helper.
        params: Its explicitly resolved parameters.
        robot_context: Measured tool transform and recorded controller policy.
        preceding_pose: The checked preceding EE pose.
        validation_scope: The program's declared validation scope.
        held_part_transform: Transform from the held part reference into the EE frame.

    Returns:
        The existing helper outputs, including distinct fitting insertion poses.
    """
    from cais_spade_llm.resources.robot.robot_primitives import ROBOT_EXTRACT_OUTPUT_MAP

    scope = read_validation_scope({"validation_scope": validation_scope})
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
        orientation = {key: preceding_pose[key] for key in ("qx", "qy", "qz", "qw")}
        if primitive_symbol == "compute_place_targets":
            pick = params["pick_ctx"]
            if not isinstance(pick, Mapping):
                raise CalculationUnavailable("pick_ctx requires an explicitly bound object.")
            handoff = pick.get("held_part_handoff", {})
            positions = pick.get("resolved_cartesian_positions", {})
            if not isinstance(handoff, Mapping) or not isinstance(positions, Mapping):
                raise CalculationUnavailable("Selected grasp orientation containers must be objects.")
            raw_pose = handoff.get("world_tool0_pose_at_grasp") or positions.get("descend")
            if raw_pose:
                ee = pose_matrix(raw_pose)
                orientation = {key: raw_pose[key] for key in ("qx", "qy", "qz", "qw")}
        tool = np.asarray(robot_context["ee_from_tcp"], dtype=float)
        if tool.shape != (4, 4) or not np.isfinite(tool).all():
            raise CalculationUnavailable("A measured full EE–TCP transform is required.")
        matrix_pose(tool)
        offset = ee[:3, :3] @ tool[:3, 3]
        geometry = params["product_geometry"]
        if not isinstance(geometry, Mapping) or "record_type" in geometry:
            raise CalculationUnavailable(
                "product_geometry must select actual geometry fields, not a raw record."
            )
        policy = robot_context["policy"]
        if (primitive_symbol == "compute_place_targets" and scope == GAZEBO_OBSERVED_SCOPE
                and "target_origin_pose" in geometry):
            if held_part_transform is None:
                raise CalculationUnavailable("Fitting requires the measured rigid part-to-EE transform at grasp.")
            held = np.asarray(held_part_transform, dtype=float)
            matrix_pose(held)
            final_part = pose_matrix(geometry["target_origin_pose"])
            insertion = np.asarray(geometry["insertion_axis"], dtype=float)
            if (insertion.shape != (3,) or not np.isfinite(insertion).all()
                    or not math.isclose(float(np.linalg.norm(insertion)), 1.0, abs_tol=1e-6)):
                raise CalculationUnavailable("Fitting requires a measured unit insertion_axis.")
            distance = _number(geometry["insertion_distance_m"], "insertion_distance_m", positive=True)
            clearance = _number(policy["insertion_depth_m"], "insertion approach clearance", positive=True)
            final_ee = final_part @ np.linalg.inv(held)
            pre_insert = final_ee.copy()
            pre_insert[:3, 3] -= insertion * (distance + clearance)
            approach = pre_insert.copy()
            approach[:3, 3] -= insertion * 0.05
            final_tcp = final_ee @ tool
            surface = {key: _number(geometry["placement_surface_point"][key], "placement_surface_point." + key)
                       for key in ("x", "y", "z")}
            result = {"part_name": params["part_name"], "part_height": _number(geometry["part_height_m"], "part_height_m", positive=True),
                      "slot_x": surface["x"], "slot_y": surface["y"], "board_top_z": surface["z"],
                      "place_z": float(final_ee[2, 3]), "place_tcp_z": float(final_tcp[2, 3]),
                      "tcp_offset_z": float(final_tcp[2, 3] - final_ee[2, 3]),
                      "grasp_tcp_to_part_origin_z": float(final_tcp[2, 3] - final_part[2, 3]),
                      "place_part_origin_z": float(final_part[2, 3]),
                      "target_pose": matrix_pose(pre_insert), "pre_insert_pose": matrix_pose(pre_insert),
                      "insert_pose": matrix_pose(final_ee), "approach_pose": matrix_pose(approach),
                      "target_origin_pose": dict(geometry["target_origin_pose"]),
                      "insertion_axis_world": dict(zip(("x", "y", "z"), insertion.tolist()))}
            output, error = ROBOT_EXTRACT_OUTPUT_MAP[primitive_symbol](dict(params), {"success": True, **result})
            if error or output is None:
                raise CalculationUnavailable(error or "The fitting helper returned no outputs.")
            return _without_model_name(output)
        if primitive_symbol == "compute_pick_targets":
            result = _pick(params, geometry, policy, preceding_pose, float(offset[2]))
        elif scope == GAZEBO_OBSERVED_SCOPE:
            result = _place_observed(params, geometry, policy, float(offset[2]), orientation)
        else:
            result = _place(params, geometry, policy, float(offset[2]), orientation)
        # Scalar object/slot coordinates retain their meaning. Only controlled-link
        # targets receive the measured XY correction; vertical arithmetic already
        # subtracts the Z offset. Validation and owned execution use this same path.
        for name in ("approach_pose", "target_pose", "pre_insert_pose", "insert_pose"):
            if name in result:
                result[name]["x"] -= float(offset[0])
                result[name]["y"] -= float(offset[1])
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
        "approach_pose": {"x": x, "y": y, "z": travel_z},
        "target_pose": {"x": x, "y": y, "z": pick_z},
        "part_height": height,
        "tcp_offset_z": tcp_offset_z,
        "pick_tcp_z": tcp_z,
        "start_x": pose["x"],
        "start_y": pose["y"],
        "start_z": pose["z"],
    }


def _place_observed(
    params: Mapping[str, Any], geometry: Mapping[str, Any], policy: Mapping[str, Any],
    measured_offset: float, orientation: Mapping[str, float],
) -> dict[str, Any]:
    if params.get("destination_location"):
        raise CalculationUnavailable("Select a measured placement_surface_point instead of a destination token.")
    pick = params["pick_ctx"]
    if pick.get("origin_pose"):
        raise CalculationUnavailable("pick_ctx.origin_pose cannot override the selected observed placement surface.")
    height = _number(geometry.get("part_height_m", pick["part_height"]), "part_height", positive=True)
    tcp_offset = _number(pick["tcp_offset_z"], "pick_ctx.tcp_offset_z")
    if not math.isclose(tcp_offset, measured_offset, abs_tol=1e-6):
        raise CalculationUnavailable("The pick tool offset is incompatible with the placement orientation.")
    surface = geometry["placement_surface_point"]
    x, y, z = (_number(surface[axis], "placement_surface_point." + axis) for axis in ("x", "y", "z"))
    reference_z = z + height / 2 + _number(policy["place_surface_gap_m"], "place surface gap")
    grasp_offset = _number(pick["pick_tcp_z"], "pick_tcp_z") - _number(pick["tz"], "pick reference Z")
    tcp_z = reference_z + grasp_offset
    place_z = controlled_link_height(tcp_z, tcp_offset, _number(params.get("z_adjustment_m", 0.0), "z adjustment"))
    return {
        "part_name": params.get("part_name", pick.get("part_name", "")),
        "slot_x": x, "slot_y": y, "board_top_z": z,
        "place_z": place_z, "place_tcp_z": tcp_z, "place_part_origin_z": reference_z,
        "part_height": height, "tcp_offset_z": tcp_offset,
        "grasp_tcp_to_part_origin_z": grasp_offset,
        "target_reference": {"target_point": "observed_bounds_center", "surface_role": "observed_surface"},
        **placement_poses(x, y, place_z, dict(orientation), simulation_assembly_slot=False,
                          insertion_depth=policy["insertion_depth_m"]),
    }


def _place(
    params: Mapping[str, Any],
    geometry: Mapping[str, Any],
    policy: Mapping[str, Any],
    measured_offset: float,
    orientation: Mapping[str, float],
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
    reference_point = reference.get("grasp_point", "CAD_origin")
    if reference_point == "selected_CAD_feature":
        raw_offset = geometry.get("grasp_point_offset_world_m")
        if not isinstance(raw_offset, list) or len(raw_offset) != 3:
            raise CalculationUnavailable(
                "A selected CAD grasp feature requires the measured grasp_point_offset_world_m XYZ vector."
            )
        point_offset = [_number(value, "grasp_point_offset_world_m") for value in raw_offset]
    elif reference_point == "CAD_origin":
        if "grasp_point_offset_world_m" in geometry:
            raise CalculationUnavailable(
                "A grasp-point offset requires its explicitly selected target_reference.grasp_point."
            )
        point_offset = [0.0, 0.0, 0.0]
    else:
        raise CalculationUnavailable("The selected grasp reference is not supported by this calculation.")
    grasp_offset += point_offset[2]
    tcp_z = origin_z + grasp_offset
    place_z = controlled_link_height(
        tcp_z, tcp_offset, _number(params.get("z_adjustment_m", 0.0), "z adjustment policy")
    )
    poses = placement_poses(
        x + point_offset[0],
        y + point_offset[1],
        place_z,
        dict(orientation),
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
