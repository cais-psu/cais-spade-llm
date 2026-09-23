from __future__ import annotations

"""
Config-driven Gazebo pick/place controller.

This module intentionally avoids importing ROS2 packages at module import time.
All ROS2 imports happen lazily inside `init()` so non-ROS workflows can still
import the package.
"""

import hashlib
import importlib
import json
import logging
import math
import os
import sys
import threading
import time
import xml.etree.ElementTree as ET
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from cais_spade_llm.product.profile import ProductProfile
from cais_spade_llm.resources.robot.target_calculations import (
    controlled_link_height,
    pick_travel_height,
    placement_poses,
    supported_part_origin_height,
    vertical_pick_bias,
)

destination_token_from_place_inputs = ProductProfile.destination_token_from_place_inputs
has_place_geometry_fields = ProductProfile.has_place_geometry_fields
resolve_place_geometry = ProductProfile.resolve_place_geometry

logger = logging.getLogger(__name__)
_RCLPY_INIT_LOCK = threading.Lock()

_PHYSICAL_DETECTION_MAX_AGE_SEC = 10.0
_PHYSICAL_DETECTION_FUTURE_TOLERANCE_SEC = 1.0
_ASSEMBLY_BOARD_V1 = "assembly_board-v1"
_ASSEMBLY_BOARD_ARUCO_DICTIONARY = "DICT_ARUCO_ORIGINAL"
_ASSEMBLY_BOARD_ARUCO_ID = 70
_ASSEMBLY_BOARD_ARUCO_MARKER_LENGTH_M = 0.076
_ASSEMBLY_BOARD_ARUCO_MAX_AGE_SEC = 2.0
_ASSEMBLY_BOARD_ARUCO_MINIMUM_SAMPLES = 10
_ASSEMBLY_BOARD_ARUCO_MAX_TRANSLATION_SPREAD_M = 0.002
_ASSEMBLY_BOARD_ARUCO_MAX_ROTATION_SPREAD_DEG = 0.5
_ASSEMBLY_BOARD_ARUCO_MAX_BASELINE_TRANSLATION_M = 0.010
_ASSEMBLY_BOARD_ARUCO_MAX_BASELINE_ROTATION_DEG = 2.0
_ASSEMBLY_BOARD_ARUCO_WAIT_SEC = 15.0
_PERCEPTION_CAMERA_CONFIG = Path(
    "~/.config/cais-spade-llm/perception_cameras.yaml"
).expanduser()
_PERCEPTION_PREVIEW_ROOT = Path("/tmp/cais_perception_previews")
_MOVE_INSERT_SUPPORTED_PARTS = frozenset({"SG", "MG", "LG", "SCP", "MCP", "LCP"})
_MOVE_INSERT_RECTANGULAR_PARTS = frozenset({"SRP", "MRP", "LRP"})
_MOVE_INSERT_PROFILE_FIELDS = (
    "pre_insert_offset_m",
    "contact_speed_m_s",
    "contact_force_delta_n",
    "engagement_progress_m",
    "insertion_force_n",
    "spiral_radius_m",
    "spiral_pitch_m",
    "spiral_speed_m_s",
    "spiral_acceleration_m_s2",
    "max_axial_force_n",
    "max_lateral_force_n",
    "max_torque_nm",
    "tilt_tolerance_rad",
    "seated_depth_tolerance_m",
    "settle_time_sec",
)
_MOVE_INSERT_OVERRIDE_VALUE_FIELDS = frozenset(
    {"insertion_force_n", "spiral_radius_m"}
)
_MOVE_INSERT_OVERRIDE_METADATA_FIELDS = frozenset(
    {"calibration_id", "generation", "updated_at", "profile_sha256"}
)
_MOVE_INSERT_DEMONSTRATION_RECIPE_VERSION = 9
_MOVE_INSERT_LEARNING_POLICY_VERSION = 12
_MOVE_INSERT_FORCE_DEPTH_PROFILE_POINTS = 16
_MOVE_INSERT_QUALIFICATION_POLICY_VERSION = 3
_MOVE_INSERT_REQUIRED_CONFIRMED_TRIALS = 1
_MOVE_INSERT_QUALIFICATION_IDENTITY_FIELDS = (
    "robot",
    "tool_frame",
    "destination_location",
    "part_name",
    "profile_sha256",
    "hard_caps_sha256",
    "place_approach_recording_sha256",
    "board_calibration_id",
    "board_geometry_sha256",
    "recording_id",
    "demonstration_sha256",
)
_MOVE_INSERT_RELIEF_POLICY_FIELDS = (
    "insert_max_tool_flange_torque_nm",
    "insert_soft_filter_window_sec",
    "insert_soft_overload_hold_sec",
    "insert_relief_unload_dwell_sec",
    "insert_relief_clear_dwell_sec",
    "insert_relief_timeout_sec",
    "insert_relief_axial_force_ratio",
    "insert_relief_reverse_force_ratio",
    "insert_relief_clear_hysteresis_ratio",
    "insert_relief_resume_ramp_sec",
    "insert_relief_search_force_ratio",
    "insert_relief_search_speed_ratio",
    "insert_relief_backoff_step_m",
    "insert_max_relief_retreat_m",
    "insert_relief_stationary_speed_m_s",
    "insert_relief_stationary_angular_speed_rad_s",
    "insert_max_relief_cycles",
)
_MOVE_INSERT_MG_TACTILE_POLICY_FIELDS = (
    "tactile_center_policy",
    "expanded_search_policy",
    "cocked_recovery_sequence",
    "disengagement_lateral_clearance_policy",
    "search_peck_policy",
    "insert_max_contact_search_radius_m",
    "insert_max_disengagement_cycles",
    "insert_search_peck_retreat_m",
    "insert_search_peck_interval_sec",
)
_MOVE_INSERT_LEARNING_POLICY_FIELDS = (
    "learning_policy_version",
    "axial_force_sign_convention",
    "force_filter",
    "contact_hold_sec",
    "torque_reference",
    "limit_policy",
    "relief_sequence",
    "hard_limit_policy",
    "demonstration_speed_policy",
    "rebound_policy",
    "force_depth_profile_points",
    "axial_soft_overload_policy",
    "engagement_policy",
    "seating_policy",
    "force_uncertainty_floor_n",
    "torque_uncertainty_floor_nm",
    *_MOVE_INSERT_RELIEF_POLICY_FIELDS,
)
_PHYSICAL_XARM6_ASSEMBLY_SLOT_INSERT_ERROR = (
    "physical xarm6 assembly_board-v1 assembly_slot insertion with a held part "
    "is blocked: move_insert is available only for ur5e. Select ur5e for Assembly."
)


def _import_linkattacher_srvs():
    """Import IFRA service types, even if workspace setup wasn't sourced."""

    def _load_srvs():
        srv_module = importlib.import_module("linkattacher_msgs.srv")
        return srv_module.AttachLink, srv_module.DetachLink

    try:
        return _load_srvs()
    except ModuleNotFoundError:
        py_ver = f"python{sys.version_info.major}.{sys.version_info.minor}"
        candidates = [
            os.path.expanduser(
                f"~/ros2_ws/install/linkattacher_msgs/local/lib/{py_ver}/dist-packages"
            ),
            os.path.expanduser(
                f"~/ros2_ws/install/ros2_linkattacher/local/lib/{py_ver}/dist-packages"
            ),
            f"/opt/ros/humble/lib/{py_ver}/dist-packages",
        ]
        for path in candidates:
            if os.path.isdir(path) and path not in sys.path:
                sys.path.append(path)
        try:
            return _load_srvs()
        except Exception:
            return None, None


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _normalized_optional_quaternion(
    qx: float | None,
    qy: float | None,
    qz: float | None,
    qw: float | None,
) -> tuple[tuple[float, float, float, float] | None, str | None]:
    values = (qx, qy, qz, qw)
    supplied = tuple(value is not None for value in values)
    if not any(supplied):
        return None, None
    if not all(supplied):
        return None, "orientation requires qx, qy, qz, and qw together"

    try:
        numeric = tuple(float(value) for value in values)
    except (TypeError, ValueError, OverflowError):
        return None, "orientation quaternion must contain finite numeric values"
    if not all(math.isfinite(value) for value in numeric):
        return None, "orientation quaternion must contain finite numeric values"

    norm = math.hypot(*numeric)
    if not math.isfinite(norm):
        return None, "orientation quaternion must have a finite norm"
    if norm <= 1e-12:
        return None, "orientation quaternion must be non-zero"
    return tuple(value / norm for value in numeric), None


def _pose_from_mapping(value: Any) -> dict[str, float]:
    if not isinstance(value, dict):
        raise ValueError("pose must be an object")
    pose = {
        field: float(value[field])
        for field in ("x", "y", "z", "qx", "qy", "qz", "qw")
    }
    if not all(math.isfinite(item) for item in pose.values()):
        raise ValueError("pose contains a non-finite value")
    quaternion, error = _normalized_optional_quaternion(
        pose["qx"], pose["qy"], pose["qz"], pose["qw"]
    )
    if error or quaternion is None:
        raise ValueError(error or "pose quaternion is invalid")
    pose.update(dict(zip(("qx", "qy", "qz", "qw"), quaternion, strict=True)))
    return pose


def _canonical_json_sha256(value: Any) -> tuple[str, str]:
    try:
        canonical = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        return "", f"move_insert profile is not canonical JSON: {exc}"
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest(), ""


def _move_insert_qualification_policy_sha256() -> str:
    """Hash the protected exact-part supervised qualification policy."""
    digest, error = _canonical_json_sha256(
        {
            "qualification_policy_version": (
                _MOVE_INSERT_QUALIFICATION_POLICY_VERSION
            ),
            "required_consecutive_confirmed_trials": (
                _MOVE_INSERT_REQUIRED_CONFIRMED_TRIALS
            ),
            "failure_resets_confirmed_trials": True,
            "recovered_soft_overload_may_count": True,
            "hard_limit_may_count": False,
            "identity_fields": list(_MOVE_INSERT_QUALIFICATION_IDENTITY_FIELDS),
        }
    )
    if error:
        raise RuntimeError(error)
    return digest


def _move_insert_qualification_evidence_sha256(
    *,
    qualification_identity: Mapping[str, Any],
    trial_ids: list[str],
    result_sha256s: list[str],
    trace_sha256s: list[str],
) -> str:
    """Hash the complete confirmed-trial qualification evidence."""
    digest, error = _canonical_json_sha256(
        {
            "qualification_policy_sha256": (
                _move_insert_qualification_policy_sha256()
            ),
            "qualification_identity": dict(qualification_identity),
            "confirmed_trial_ids": list(trial_ids),
            "confirmed_trial_result_sha256s": list(result_sha256s),
            "confirmed_trial_trace_sha256s": list(trace_sha256s),
        }
    )
    if error:
        raise RuntimeError(error)
    return digest


def move_insert_learning_policy_sha256(  # noqa: C901, PLR0912
    raw_recipe: Mapping[str, Any] | None,
) -> tuple[str, str]:
    """Validate and hash the exact protected learning and relief policy."""
    if not isinstance(raw_recipe, Mapping):
        return "", "move_insert learning policy recipe is missing"
    raw_policy = raw_recipe.get("learning_policy")
    if not isinstance(raw_policy, Mapping):
        return "", "move_insert learning_policy is missing or is not an object"
    policy = dict(raw_policy)
    recipe_part = raw_recipe.get("part_name")
    expected_fields = set(_MOVE_INSERT_LEARNING_POLICY_FIELDS)
    advanced_policy_fields = set(_MOVE_INSERT_MG_TACTILE_POLICY_FIELDS)
    advanced_hard_cap_fields = {
        field_name
        for field_name in _MOVE_INSERT_MG_TACTILE_POLICY_FIELDS
        if field_name.startswith("insert_")
    }
    raw_hard_caps = raw_recipe.get("hard_caps")
    advanced_recovery_enabled = bool(
        advanced_policy_fields.intersection(policy)
        or (
            isinstance(raw_hard_caps, Mapping)
            and advanced_hard_cap_fields.intersection(raw_hard_caps)
        )
    )
    if recipe_part in _MOVE_INSERT_SUPPORTED_PARTS and advanced_recovery_enabled:
        expected_fields.update(_MOVE_INSERT_MG_TACTILE_POLICY_FIELDS)
    unexpected = sorted(set(policy) - expected_fields)
    missing = sorted(expected_fields - set(policy))
    if unexpected or missing:
        return "", (
            "move_insert learning_policy fields do not match the protected schema: "
            f"missing={missing}, unexpected={unexpected}"
        )
    exact_values = {
        "learning_policy_version": _MOVE_INSERT_LEARNING_POLICY_VERSION,
        "axial_force_sign_convention": "compression_negative_dot",
        "force_filter": "time_window_median",
        "contact_hold_sec": 0.10,
        "torque_reference": "active_tcp",
        "limit_policy": "reject_not_clip",
        "relief_sequence": "unload_then_bounded_micro_backoff",
        "hard_limit_policy": "immediate_stop",
        "demonstration_speed_policy": "diagnostic_only",
        "rebound_policy": (
            "final_saved_depth_within_tolerance_of_post_contact_maximum"
        ),
        "force_depth_profile_points": _MOVE_INSERT_FORCE_DEPTH_PROFILE_POINTS,
        "axial_soft_overload_policy": (
            "learned_and_profile_exceedance_with_stalled_progress_"
            "guarded_below_hard_cap"
        ),
        "engagement_policy": (
            "sustained_axial_progress_within_force_depth_profile"
        ),
        "seating_policy": (
            "target_depth_stationary_stable_force_within_force_depth_profile"
        ),
        "force_uncertainty_floor_n": 4.0,
        "torque_uncertainty_floor_nm": 0.05,
        "insert_max_relief_cycles": 3.0,
    }
    if recipe_part in _MOVE_INSERT_SUPPORTED_PARTS and advanced_recovery_enabled:
        exact_values.update(
            {
                "tactile_center_policy": (
                    "deepest_stable_progress_then_lowest_normalized_lateral_load"
                ),
                "expanded_search_policy": (
                    "local_then_staged_3mm_5mm_10mm_low_preload"
                ),
                "cocked_recovery_sequence": (
                    "unload_then_exact_pre_insert_withdrawal_recenter_retare_retry"
                ),
                "disengagement_lateral_clearance_policy": (
                    "search_boundary_plus_start_position_tolerance"
                ),
                "search_peck_policy": (
                    "stalled_spiral_bounded_axial_unload_then_low_preload_recontact"
                ),
            }
        )
    for field_name, expected in exact_values.items():
        if policy.get(field_name) != expected:
            return "", (
                f"move_insert learning_policy.{field_name} must be exact {expected!r}"
            )
    for field_name in _MOVE_INSERT_RELIEF_POLICY_FIELDS:
        value = policy.get(field_name)
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError):
            number = math.nan
        if isinstance(value, bool) or not math.isfinite(number) or number <= 0.0:
            return "", (
                f"move_insert learning_policy.{field_name} must be finite and positive"
            )
    if recipe_part in _MOVE_INSERT_SUPPORTED_PARTS and advanced_recovery_enabled:
        if not isinstance(raw_hard_caps, Mapping):
            return "", "move_insert exact-part recovery hard_caps are missing"
        for field_name in (
            "insert_max_contact_search_radius_m",
            "insert_max_disengagement_cycles",
            "insert_search_peck_retreat_m",
            "insert_search_peck_interval_sec",
        ):
            value = policy.get(field_name)
            try:
                number = float(value)
            except (TypeError, ValueError, OverflowError):
                number = math.nan
            if isinstance(value, bool) or not math.isfinite(number) or number <= 0.0:
                return "", (
                    f"move_insert learning_policy.{field_name} must be finite and positive"
                )
            hard_cap_value = raw_hard_caps.get(field_name)
            try:
                hard_cap_number = float(hard_cap_value)
            except (TypeError, ValueError, OverflowError):
                hard_cap_number = math.nan
            if (
                isinstance(hard_cap_value, bool)
                or not math.isfinite(hard_cap_number)
                or hard_cap_number <= 0.0
                or hard_cap_number != number
            ):
                return "", (
                    f"move_insert learning_policy.{field_name} must match the exact "
                    f"{recipe_part} hard cap"
                )
    for field_name in (
        "insert_relief_axial_force_ratio",
        "insert_relief_reverse_force_ratio",
        "insert_relief_clear_hysteresis_ratio",
        "insert_relief_search_force_ratio",
        "insert_relief_search_speed_ratio",
    ):
        if float(policy[field_name]) >= 1.0:
            return "", f"move_insert learning_policy.{field_name} must be less than 1"
    if float(policy["insert_relief_backoff_step_m"]) > float(
        policy["insert_max_relief_retreat_m"]
    ):
        return "", "move_insert learning_policy backoff step exceeds retreat ceiling"
    for dwell_name in (
        "insert_relief_unload_dwell_sec",
        "insert_relief_clear_dwell_sec",
    ):
        if float(policy[dwell_name]) >= float(
            policy["insert_relief_timeout_sec"]
        ):
            return "", (
                f"move_insert learning_policy.{dwell_name} must be below "
                "insert_relief_timeout_sec"
            )
    if float(policy["insert_soft_filter_window_sec"]) > float(
        policy["insert_soft_overload_hold_sec"]
    ):
        return "", "move_insert learning_policy filter window exceeds overload hold"
    return _canonical_json_sha256(policy)


def move_insert_profile_sha256(
    raw_profile: Mapping[str, Any] | None,
    part_name: str,
) -> tuple[str, str]:
    """Hash the selected shared profile plus only the exact selected override."""
    if not isinstance(raw_profile, Mapping):
        return "", "move_insert profile is missing or is not an object"
    profile = dict(raw_profile)
    raw_overrides = profile.pop("part_overrides", {})
    raw_demonstration_recipes = profile.pop("demonstration_recipes", {})
    # Qualification is operator/workcell state, not part of the calibrated motion
    # recipe.  Excluding both authorization fields keeps an existing exact-part
    # qualification stable when another part is confirmed later.
    profile.pop("validated_parts", None)
    profile.pop("qualifications", None)
    if not isinstance(raw_overrides, Mapping):
        return "", "move_insert part_overrides must be an object"
    if not isinstance(raw_demonstration_recipes, Mapping):
        return "", "move_insert demonstration_recipes must be an object"
    if not isinstance(part_name, str) or not part_name or part_name != part_name.strip():
        return "", "move_insert requires an exact non-empty part identifier"
    requested_part = part_name
    override_present = requested_part in raw_overrides
    raw_override = raw_overrides.get(requested_part, {})
    if not isinstance(raw_override, Mapping):
        return "", f"move_insert override for {requested_part!r} must be an object"
    selected_override = {
        key: value
        for key, value in dict(raw_override).items()
        if key != "profile_sha256"
    }
    raw_demonstration_recipe = raw_demonstration_recipes.get(requested_part, {})
    if not isinstance(raw_demonstration_recipe, Mapping):
        return (
            "",
            f"move_insert demonstration_recipes.{requested_part} must be an object",
        )
    return _canonical_json_sha256(
        {
            "shared": profile,
            "selected_demonstration_recipe": {
                "part_name": requested_part,
                "present": requested_part in raw_demonstration_recipes,
                "values": dict(raw_demonstration_recipe),
            },
            "selected_override": {
                "part_name": requested_part,
                "present": override_present,
                "values": selected_override,
            },
        }
    )


def resolve_move_insert_profile(  # noqa: C901, PLR0912, PLR0915 - fail-closed profile validation.
    parts_tuning: Mapping[str, Any] | None,
    part_name: str,
    *,
    require_qualification: bool = True,
) -> dict[str, Any]:
    """Validate and resolve the exact per-part ``move_insert`` controller profile."""
    tuning = dict(parts_tuning) if isinstance(parts_tuning, Mapping) else {}
    raw_profile = tuning.get("move_insert")
    if not isinstance(raw_profile, Mapping):
        return {
            "success": False,
            "message": "controller.parts_tuning.move_insert is missing or is not an object",
            "missing_fields": [
                "calibration_id",
                "validated_parts",
                *_MOVE_INSERT_PROFILE_FIELDS,
                "part_overrides",
            ],
        }
    raw_profile = dict(raw_profile)
    if not isinstance(part_name, str) or not part_name or part_name != part_name.strip():
        return {
            "success": False,
            "message": "move_insert requires an exact non-empty part identifier",
        }
    requested_part = part_name
    profile_sha256, hash_error = move_insert_profile_sha256(
        raw_profile,
        requested_part,
    )
    if hash_error:
        return {"success": False, "message": hash_error}

    allowed_fields = {
        "calibration_id",
        "validated_parts",
        "qualifications",
        "demonstration_recipes",
        "part_overrides",
        *_MOVE_INSERT_PROFILE_FIELDS,
    }
    unexpected_fields = sorted(set(raw_profile) - allowed_fields)
    if unexpected_fields:
        return {
            "success": False,
            "message": (
                "controller.parts_tuning.move_insert contains unsupported fields: "
                f"{unexpected_fields}"
            ),
            "profile_sha256": profile_sha256,
        }

    raw_validated_parts = raw_profile.get("validated_parts")
    if not isinstance(raw_validated_parts, list) or any(
        not isinstance(value, str) or not value for value in raw_validated_parts
    ):
        return {
            "success": False,
            "message": "controller.parts_tuning.move_insert.validated_parts must be a list of exact part identifiers",
            "profile_sha256": profile_sha256,
        }
    validated_parts = list(raw_validated_parts)
    if len(set(validated_parts)) != len(validated_parts):
        return {
            "success": False,
            "message": "controller.parts_tuning.move_insert.validated_parts contains duplicates",
            "profile_sha256": profile_sha256,
        }
    invalid_validated_parts = sorted(
        set(validated_parts) - _MOVE_INSERT_SUPPORTED_PARTS
    )
    if invalid_validated_parts:
        return {
            "success": False,
            "message": (
                "controller.parts_tuning.move_insert.validated_parts contains parts "
                f"without validated insertion orientation: {invalid_validated_parts}"
            ),
            "profile_sha256": profile_sha256,
        }

    raw_overrides = raw_profile.get("part_overrides")
    if not isinstance(raw_overrides, Mapping):
        return {
            "success": False,
            "message": "controller.parts_tuning.move_insert.part_overrides must be an object",
            "profile_sha256": profile_sha256,
            "validated_parts": validated_parts,
        }

    raw_demonstration_recipes = raw_profile.get("demonstration_recipes", {})
    if not isinstance(raw_demonstration_recipes, Mapping):
        return {
            "success": False,
            "message": (
                "controller.parts_tuning.move_insert.demonstration_recipes "
                "must be an object"
            ),
            "profile_sha256": profile_sha256,
            "validated_parts": validated_parts,
        }
    demonstration_recipes = dict(raw_demonstration_recipes)
    allowed_demonstration_recipe_fields = {
        "recipe_version",
        "learning_policy_version",
        "learning_policy_sha256",
        "learning_policy",
        "hard_caps",
        "force_depth_profile",
        "force_depth_profile_sha256",
        "baseline_force_uncertainty_n",
        "baseline_torque_uncertainty_nm",
        "observed_filtered_axial_force_n",
        "observed_filtered_lateral_force_n",
        "observed_filtered_torque_nm",
        "observed_tool_flange_torque_nm",
        "observed_raw_axial_force_n",
        "observed_raw_lateral_force_n",
        "observed_raw_torque_nm",
        "observed_raw_tool_flange_torque_nm",
        "observed_advancing_speed_m_s",
        "observed_peak_filtered_advancing_speed_m_s",
        "seated_filtered_axial_force_n",
        "seated_filtered_lateral_force_n",
        "seated_filtered_torque_nm",
        "seated_filtered_tool_flange_torque_nm",
        "calibration_id",
        "recording_id",
        "demonstration_sha256",
        "updated_at",
        "robot",
        "tool_frame",
        "destination_location",
        "part_name",
        "context_sha256",
        "place_approach_recording_sha256",
        "board_calibration_id",
        "board_generation",
        "aruco_to_seated_held_part",
        "aruco_insertion_axis",
        *_MOVE_INSERT_PROFILE_FIELDS,
    }
    for recipe_part, raw_recipe in demonstration_recipes.items():
        if (
            not isinstance(recipe_part, str)
            or recipe_part not in _MOVE_INSERT_SUPPORTED_PARTS
        ):
            return {
                "success": False,
                "message": (
                    "controller.parts_tuning.move_insert.demonstration_recipes "
                    "contains an unsupported exact part identifier: "
                    f"{recipe_part!r}"
                ),
                "profile_sha256": profile_sha256,
                "validated_parts": validated_parts,
            }
        if not isinstance(raw_recipe, Mapping):
            return {
                "success": False,
                "message": (
                    "controller.parts_tuning.move_insert.demonstration_recipes."
                    f"{recipe_part} must be an object"
                ),
                "profile_sha256": profile_sha256,
                "validated_parts": validated_parts,
            }
        unexpected_recipe_fields = sorted(
            set(raw_recipe) - allowed_demonstration_recipe_fields
        )
        if unexpected_recipe_fields:
            return {
                "success": False,
                "message": (
                    "controller.parts_tuning.move_insert.demonstration_recipes."
                    f"{recipe_part} contains unsupported fields: "
                    f"{unexpected_recipe_fields}"
                ),
                "profile_sha256": profile_sha256,
                "validated_parts": validated_parts,
            }
        recipe = dict(raw_recipe)
        if recipe.get("recipe_version") != _MOVE_INSERT_DEMONSTRATION_RECIPE_VERSION:
            return {
                "success": False,
                "message": (
                    "controller.parts_tuning.move_insert.demonstration_recipes."
                    f"{recipe_part} is legacy or unsafe; install the migrated "
                    "recipe or use Reanalyze Saved Recording. Another manual "
                    "demonstration is unnecessary when its preserved trace remains "
                    "compatible"
                ),
                "profile_sha256": profile_sha256,
                "validated_parts": validated_parts,
            }
        if (
            recipe.get("learning_policy_version")
            != _MOVE_INSERT_LEARNING_POLICY_VERSION
        ):
            return {
                "success": False,
                "message": (
                    "controller.parts_tuning.move_insert.demonstration_recipes."
                    f"{recipe_part}.learning_policy_version is unsupported"
                ),
                "profile_sha256": profile_sha256,
                "validated_parts": validated_parts,
            }
        expected_policy_sha256, policy_hash_error = (
            move_insert_learning_policy_sha256(recipe)
        )
        if (
            policy_hash_error
            or recipe.get("learning_policy_sha256")
            != expected_policy_sha256
        ):
            return {
                "success": False,
                "message": (
                    "controller.parts_tuning.move_insert.demonstration_recipes."
                    f"{recipe_part}.learning_policy_sha256 does not match the "
                    "protected learning policy"
                ),
                "profile_sha256": profile_sha256,
                "validated_parts": validated_parts,
            }
        for digest_field in (
            "learning_policy_sha256",
            "force_depth_profile_sha256",
        ):
            digest = recipe.get(digest_field)
            try:
                valid_digest = (
                    isinstance(digest, str)
                    and len(digest) == 64
                    and int(digest, 16) >= 0
                )
            except ValueError:
                valid_digest = False
            if not valid_digest:
                return {
                    "success": False,
                    "message": (
                        "controller.parts_tuning.move_insert.demonstration_recipes."
                        f"{recipe_part}.{digest_field} is not a SHA-256 digest"
                    ),
                    "profile_sha256": profile_sha256,
                    "validated_parts": validated_parts,
                }
        raw_hard_caps = recipe.get("hard_caps")
        if not isinstance(raw_hard_caps, Mapping) or not raw_hard_caps:
            return {
                "success": False,
                "message": (
                    "controller.parts_tuning.move_insert.demonstration_recipes."
                    f"{recipe_part}.hard_caps must be a nonempty object"
                ),
                "profile_sha256": profile_sha256,
                "validated_parts": validated_parts,
            }
        try:
            hard_caps = {
                str(field_name): float(value)
                for field_name, value in raw_hard_caps.items()
            }
        except (TypeError, ValueError, OverflowError):
            hard_caps = {}
        if (
            not hard_caps
            or set(hard_caps) != set(raw_hard_caps)
            or any(
                not field_name
                or not math.isfinite(value)
                or value < 0.0
                for field_name, value in hard_caps.items()
            )
        ):
            return {
                "success": False,
                "message": (
                    "controller.parts_tuning.move_insert.demonstration_recipes."
                    f"{recipe_part}.hard_caps contains invalid evidence"
                ),
                "profile_sha256": profile_sha256,
                "validated_parts": validated_parts,
            }
        for evidence_field in (
            "baseline_force_uncertainty_n",
            "baseline_torque_uncertainty_nm",
            "observed_filtered_axial_force_n",
            "observed_filtered_lateral_force_n",
            "observed_filtered_torque_nm",
            "observed_tool_flange_torque_nm",
            "observed_raw_axial_force_n",
            "observed_raw_lateral_force_n",
            "observed_raw_torque_nm",
            "observed_raw_tool_flange_torque_nm",
            "observed_advancing_speed_m_s",
            "observed_peak_filtered_advancing_speed_m_s",
            "seated_filtered_axial_force_n",
            "seated_filtered_lateral_force_n",
            "seated_filtered_torque_nm",
            "seated_filtered_tool_flange_torque_nm",
        ):
            raw_evidence = recipe.get(evidence_field)
            try:
                evidence_value = float(raw_evidence)
            except (TypeError, ValueError, OverflowError):
                evidence_value = math.nan
            if (
                isinstance(raw_evidence, bool)
                or not math.isfinite(evidence_value)
                or evidence_value < 0.0
            ):
                return {
                    "success": False,
                    "message": (
                        "controller.parts_tuning.move_insert.demonstration_recipes."
                        f"{recipe_part}.{evidence_field} must be finite and non-negative"
                    ),
                    "profile_sha256": profile_sha256,
                    "validated_parts": validated_parts,
                }
        raw_force_depth_profile = recipe.get("force_depth_profile")
        if not isinstance(raw_force_depth_profile, Mapping):
            return {
                "success": False,
                "message": (
                    "controller.parts_tuning.move_insert.demonstration_recipes."
                    f"{recipe_part}.force_depth_profile must be an object"
                ),
                "profile_sha256": profile_sha256,
                "validated_parts": validated_parts,
            }
        force_depth_profile = dict(raw_force_depth_profile)
        expected_force_depth_fields = {
            "depth_fraction",
            "axial_upper_n",
            "lateral_upper_n",
            "torque_upper_nm",
        }
        if set(force_depth_profile) != expected_force_depth_fields:
            return {
                "success": False,
                "message": (
                    "controller.parts_tuning.move_insert.demonstration_recipes."
                    f"{recipe_part}.force_depth_profile fields are invalid"
                ),
                "profile_sha256": profile_sha256,
                "validated_parts": validated_parts,
            }
        try:
            profile_series = {
                field_name: [float(value) for value in force_depth_profile[field_name]]
                for field_name in expected_force_depth_fields
            }
        except (TypeError, ValueError, OverflowError):
            profile_series = {}
        if (
            not profile_series
            or any(
                len(values) != _MOVE_INSERT_FORCE_DEPTH_PROFILE_POINTS
                for values in profile_series.values()
            )
            or not all(
                math.isfinite(value)
                for values in profile_series.values()
                for value in values
            )
        ):
            return {
                "success": False,
                "message": (
                    "controller.parts_tuning.move_insert.demonstration_recipes."
                    f"{recipe_part}.force_depth_profile requires 16 finite points"
                ),
                "profile_sha256": profile_sha256,
                "validated_parts": validated_parts,
            }
        fractions = profile_series["depth_fraction"]
        if (
            abs(fractions[0]) > 1e-12
            or abs(fractions[-1] - 1.0) > 1e-12
            or any(
                right <= left
                for left, right in zip(fractions, fractions[1:])
            )
            or any(
                value <= 0.0
                for field_name, values in profile_series.items()
                if field_name != "depth_fraction"
                for value in values
            )
        ):
            return {
                "success": False,
                "message": (
                    "controller.parts_tuning.move_insert.demonstration_recipes."
                    f"{recipe_part}.force_depth_profile is not a protected envelope"
                ),
                "profile_sha256": profile_sha256,
                "validated_parts": validated_parts,
            }
        expected_force_depth_sha256, force_depth_hash_error = (
            _canonical_json_sha256(force_depth_profile)
        )
        if (
            force_depth_hash_error
            or recipe.get("force_depth_profile_sha256")
            != expected_force_depth_sha256
        ):
            return {
                "success": False,
                "message": (
                    "controller.parts_tuning.move_insert.demonstration_recipes."
                    f"{recipe_part}.force_depth_profile_sha256 does not match"
                ),
                "profile_sha256": profile_sha256,
                "validated_parts": validated_parts,
            }
        exact_values = {
            "robot": "ur5e",
            "tool_frame": "tool0",
            "destination_location": "assembly_board-v1",
            "part_name": recipe_part,
        }
        for field_name, expected_value in exact_values.items():
            if recipe.get(field_name) != expected_value:
                return {
                    "success": False,
                    "message": (
                        "controller.parts_tuning.move_insert.demonstration_recipes."
                        f"{recipe_part}.{field_name} must be exact {expected_value}"
                    ),
                    "profile_sha256": profile_sha256,
                    "validated_parts": validated_parts,
                }
        for field_name in (
            "calibration_id",
            "recording_id",
            "demonstration_sha256",
            "context_sha256",
            "place_approach_recording_sha256",
            "board_calibration_id",
        ):
            value = recipe.get(field_name)
            if (
                not isinstance(value, str)
                or not value
                or value != value.strip()
            ):
                return {
                    "success": False,
                    "message": (
                        "controller.parts_tuning.move_insert.demonstration_recipes."
                        f"{recipe_part}.{field_name} is missing or is not exact"
                    ),
                    "profile_sha256": profile_sha256,
                    "validated_parts": validated_parts,
                }
        for field_name in (
            "demonstration_sha256",
            "context_sha256",
            "place_approach_recording_sha256",
        ):
            digest = str(recipe.get(field_name) or "")
            try:
                valid_digest = len(digest) == 64 and int(digest, 16) >= 0
            except ValueError:
                valid_digest = False
            if not valid_digest:
                return {
                    "success": False,
                    "message": (
                        "controller.parts_tuning.move_insert.demonstration_recipes."
                        f"{recipe_part}.{field_name} is not a SHA-256 digest"
                    ),
                    "profile_sha256": profile_sha256,
                    "validated_parts": validated_parts,
                }
        try:
            updated_at = datetime.fromisoformat(
                str(recipe.get("updated_at") or "").replace("Z", "+00:00")
            )
        except ValueError:
            updated_at = None
        if updated_at is None or updated_at.tzinfo is None:
            return {
                "success": False,
                "message": (
                    "controller.parts_tuning.move_insert.demonstration_recipes."
                    f"{recipe_part}.updated_at must be a UTC ISO-8601 timestamp"
                ),
                "profile_sha256": profile_sha256,
                "validated_parts": validated_parts,
            }
        board_generation = recipe.get("board_generation")
        if (
            isinstance(board_generation, bool)
            or not isinstance(board_generation, int)
            or board_generation < 1
        ):
            return {
                "success": False,
                "message": (
                    "controller.parts_tuning.move_insert.demonstration_recipes."
                    f"{recipe_part}.board_generation must be a positive integer"
                ),
                "profile_sha256": profile_sha256,
                "validated_parts": validated_parts,
            }
        try:
            _pose_from_mapping(recipe.get("aruco_to_seated_held_part"))
            axis_values = tuple(
                float(dict(recipe.get("aruco_insertion_axis") or {})[field])
                for field in ("x", "y", "z")
            )
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            return {
                "success": False,
                "message": (
                    "controller.parts_tuning.move_insert.demonstration_recipes."
                    f"{recipe_part} geometry is invalid: {exc}"
                ),
                "profile_sha256": profile_sha256,
                "validated_parts": validated_parts,
            }
        axis_norm = math.sqrt(sum(value * value for value in axis_values))
        if not math.isfinite(axis_norm) or axis_norm <= 1e-12:
            return {
                "success": False,
                "message": (
                    "controller.parts_tuning.move_insert.demonstration_recipes."
                    f"{recipe_part}.aruco_insertion_axis is invalid"
                ),
                "profile_sha256": profile_sha256,
                "validated_parts": validated_parts,
            }

    raw_qualifications = raw_profile.get("qualifications", {})
    if not isinstance(raw_qualifications, Mapping):
        return {
            "success": False,
            "message": "controller.parts_tuning.move_insert.qualifications must be an object",
            "profile_sha256": profile_sha256,
            "validated_parts": validated_parts,
        }
    qualifications = dict(raw_qualifications)
    invalid_qualification_parts = sorted(
        part
        for part in qualifications
        if not isinstance(part, str) or part not in _MOVE_INSERT_SUPPORTED_PARTS
    )
    if invalid_qualification_parts:
        return {
            "success": False,
            "message": (
                "controller.parts_tuning.move_insert.qualifications contains "
                f"unsupported exact part identifiers: {invalid_qualification_parts}"
            ),
            "profile_sha256": profile_sha256,
            "validated_parts": validated_parts,
        }
    overrides = dict(raw_overrides)
    allowed_override_fields = (
        _MOVE_INSERT_OVERRIDE_VALUE_FIELDS | _MOVE_INSERT_OVERRIDE_METADATA_FIELDS
    )
    for override_part, raw_override in overrides.items():
        if (
            not isinstance(override_part, str)
            or override_part not in _MOVE_INSERT_SUPPORTED_PARTS
        ):
            return {
                "success": False,
                "message": (
                    "controller.parts_tuning.move_insert.part_overrides contains an "
                    f"unsupported exact part identifier: {override_part!r}"
                ),
                "profile_sha256": profile_sha256,
                "validated_parts": validated_parts,
            }
        if not isinstance(raw_override, Mapping):
            return {
                "success": False,
                "message": (
                    "controller.parts_tuning.move_insert.part_overrides."
                    f"{override_part} must be an object"
                ),
                "profile_sha256": profile_sha256,
                "validated_parts": validated_parts,
            }
        unexpected_override_fields = sorted(
            set(raw_override) - allowed_override_fields
        )
        if unexpected_override_fields:
            return {
                "success": False,
                "message": (
                    "controller.parts_tuning.move_insert.part_overrides."
                    f"{override_part} contains unsupported fields: "
                    f"{unexpected_override_fields}"
                ),
                "profile_sha256": profile_sha256,
                "validated_parts": validated_parts,
            }
        override_dict = dict(raw_override)
        metadata_fields = set(override_dict) & _MOVE_INSERT_OVERRIDE_METADATA_FIELDS
        if metadata_fields != _MOVE_INSERT_OVERRIDE_METADATA_FIELDS:
            missing_metadata = sorted(
                _MOVE_INSERT_OVERRIDE_METADATA_FIELDS - metadata_fields
            )
            return {
                "success": False,
                "message": (
                    "controller.parts_tuning.move_insert.part_overrides."
                    f"{override_part} is missing server-owned metadata: "
                    f"{missing_metadata}"
                ),
                "profile_sha256": profile_sha256,
                "validated_parts": validated_parts,
            }
        override_calibration_id = override_dict.get("calibration_id")
        if (
            not isinstance(override_calibration_id, str)
            or not override_calibration_id
            or override_calibration_id != override_calibration_id.strip()
        ):
            return {
                "success": False,
                "message": (
                    "controller.parts_tuning.move_insert.part_overrides."
                    f"{override_part}.calibration_id is missing or is not exact"
                ),
                "profile_sha256": profile_sha256,
                "validated_parts": validated_parts,
            }
        generation = override_dict.get("generation")
        if isinstance(generation, bool) or not isinstance(generation, int) or generation < 1:
            return {
                "success": False,
                "message": (
                    "controller.parts_tuning.move_insert.part_overrides."
                    f"{override_part}.generation must be a positive integer"
                ),
                "profile_sha256": profile_sha256,
                "validated_parts": validated_parts,
            }
        updated_at_text = override_dict.get("updated_at")
        try:
            updated_at = datetime.fromisoformat(
                updated_at_text.replace("Z", "+00:00")
            )
        except (AttributeError, TypeError, ValueError):
            updated_at = None
        if (
            updated_at is None
            or updated_at.tzinfo is None
            or updated_at.utcoffset() != timezone.utc.utcoffset(updated_at)
        ):
            return {
                "success": False,
                "message": (
                    "controller.parts_tuning.move_insert.part_overrides."
                    f"{override_part}.updated_at must be a UTC ISO-8601 timestamp"
                ),
                "profile_sha256": profile_sha256,
                "validated_parts": validated_parts,
            }
        stored_hash = override_dict.get("profile_sha256")
        expected_hash, selected_hash_error = move_insert_profile_sha256(
            raw_profile,
            override_part,
        )
        if (
            selected_hash_error
            or not isinstance(stored_hash, str)
            or stored_hash != expected_hash
        ):
            return {
                "success": False,
                "message": (
                    "controller.parts_tuning.move_insert.part_overrides."
                    f"{override_part}.profile_sha256 does not match its selected profile"
                ),
                "profile_sha256": profile_sha256,
                "validated_parts": validated_parts,
            }

    if requested_part in _MOVE_INSERT_RECTANGULAR_PARTS:
        return {
            "success": False,
            "message": (
                f"move_insert for {requested_part} is blocked until rectangular-part "
                "orientation is measured and validated"
            ),
            "profile_sha256": profile_sha256,
            "validated_parts": validated_parts,
        }
    if requested_part not in _MOVE_INSERT_SUPPORTED_PARTS:
        return {
            "success": False,
            "message": f"move_insert does not support exact part identifier {requested_part!r}",
            "profile_sha256": profile_sha256,
            "validated_parts": validated_parts,
        }
    if requested_part in demonstration_recipes:
        require_qualification = False
    if require_qualification and requested_part not in validated_parts:
        missing_fields = [
            field_name
            for field_name in ("calibration_id", *_MOVE_INSERT_PROFILE_FIELDS)
            if raw_profile.get(field_name) in (None, "")
        ]
        missing_detail = (
            f" Missing required fields: {missing_fields}." if missing_fields else ""
        )
        return {
            "success": False,
            "message": (
                f"move_insert for {requested_part} is not commissioned; add the exact "
                "part identifier to controller.parts_tuning.move_insert.validated_parts "
                f"only after physical validation.{missing_detail}"
            ),
            "profile_sha256": profile_sha256,
            "validated_parts": validated_parts,
            "missing_fields": missing_fields,
        }

    selected_demonstration_recipe = dict(
        demonstration_recipes.get(requested_part) or {}
    )
    shared_calibration_id = selected_demonstration_recipe.get(
        "calibration_id", raw_profile.get("calibration_id")
    )
    if (
        not isinstance(shared_calibration_id, str)
        or not shared_calibration_id
        or shared_calibration_id != shared_calibration_id.strip()
    ):
        return {
            "success": False,
            "message": (
                "controller.parts_tuning.move_insert.calibration_id is missing or is not exact"
            ),
            "profile_sha256": profile_sha256,
            "validated_parts": validated_parts,
        }

    effective_values: dict[str, float] = {}
    selected_override = {
        key: value
        for key, value in dict(overrides.get(requested_part) or {}).items()
        if key in _MOVE_INSERT_OVERRIDE_VALUE_FIELDS
    }
    selected_override_metadata = dict(overrides.get(requested_part) or {})
    override_calibration_id = selected_override_metadata.get("calibration_id")
    effective_calibration_id = (
        override_calibration_id
        if isinstance(override_calibration_id, str) and override_calibration_id
        else shared_calibration_id
    )
    for field_name in _MOVE_INSERT_PROFILE_FIELDS:
        raw_value = selected_override.get(
            field_name,
            selected_demonstration_recipe.get(
                field_name,
                raw_profile.get(field_name),
            ),
        )
        try:
            value = float(raw_value)
        except (TypeError, ValueError, OverflowError):
            return {
                "success": False,
                "message": (
                    f"controller.parts_tuning.move_insert.{field_name} is missing or invalid"
                ),
                "profile_sha256": profile_sha256,
                "validated_parts": validated_parts,
            }
        if not math.isfinite(value):
            return {
                "success": False,
                "message": f"controller.parts_tuning.move_insert.{field_name} is not finite",
                "profile_sha256": profile_sha256,
                "validated_parts": validated_parts,
            }
        effective_values[field_name] = value

    positive_fields = set(_MOVE_INSERT_PROFILE_FIELDS) - {"spiral_radius_m"}
    invalid_positive = sorted(
        field_name
        for field_name in positive_fields
        if effective_values[field_name] <= 0.0
    )
    if invalid_positive:
        return {
            "success": False,
            "message": (
                "controller.parts_tuning.move_insert fields must be positive: "
                f"{invalid_positive}"
            ),
            "profile_sha256": profile_sha256,
            "validated_parts": validated_parts,
        }
    if effective_values["spiral_radius_m"] < 0.0:
        return {
            "success": False,
            "message": (
                "controller.parts_tuning.move_insert.spiral_radius_m must be non-negative"
            ),
            "profile_sha256": profile_sha256,
            "validated_parts": validated_parts,
        }
    if effective_values["engagement_progress_m"] > effective_values["pre_insert_offset_m"]:
        return {
            "success": False,
            "message": (
                "controller.parts_tuning.move_insert.engagement_progress_m must not "
                "exceed pre_insert_offset_m"
            ),
            "profile_sha256": profile_sha256,
            "validated_parts": validated_parts,
        }
    if effective_values["seated_depth_tolerance_m"] > effective_values["pre_insert_offset_m"]:
        return {
            "success": False,
            "message": (
                "controller.parts_tuning.move_insert.seated_depth_tolerance_m must not "
                "exceed pre_insert_offset_m"
            ),
            "profile_sha256": profile_sha256,
            "validated_parts": validated_parts,
        }
    if effective_values["insertion_force_n"] >= effective_values["max_axial_force_n"]:
        return {
            "success": False,
            "message": (
                "controller.parts_tuning.move_insert.insertion_force_n must be less "
                "than max_axial_force_n"
            ),
            "profile_sha256": profile_sha256,
            "validated_parts": validated_parts,
        }
    if effective_values["contact_force_delta_n"] >= effective_values["max_axial_force_n"]:
        return {
            "success": False,
            "message": (
                "controller.parts_tuning.move_insert.contact_force_delta_n must be less "
                "than max_axial_force_n"
            ),
            "profile_sha256": profile_sha256,
            "validated_parts": validated_parts,
        }
    if effective_values["tilt_tolerance_rad"] > math.pi:
        return {
            "success": False,
            "message": "controller.parts_tuning.move_insert.tilt_tolerance_rad must not exceed pi",
            "profile_sha256": profile_sha256,
            "validated_parts": validated_parts,
        }

    qualification: dict[str, Any] = {}
    if require_qualification:
        raw_qualification = qualifications.get(requested_part)
        if not isinstance(raw_qualification, Mapping):
            return {
                "success": False,
                "message": (
                    f"move_insert for {requested_part} has not been confirmed by a "
                    "successful supervised move_insert trial"
                ),
                "profile_sha256": profile_sha256,
                "validated_parts": validated_parts,
            }
        qualification = dict(raw_qualification)
        required_qualification_fields = {
            "trial_id",
            "confirmed_at",
            "robot",
            "tool_frame",
            "destination_location",
            "part_name",
            "profile_sha256",
            "hard_caps_sha256",
            "place_approach_recording_sha256",
            "board_calibration_id",
            "board_geometry_sha256",
            "board_generation",
            "generation",
            "recording_id",
            "demonstration_sha256",
            "qualification_policy_version",
            "qualification_policy_sha256",
            "required_confirmed_trials",
            "confirmed_trial_count",
            "confirmed_trial_ids",
            "confirmed_trial_result_sha256s",
            "confirmed_trial_trace_sha256s",
            "qualification_evidence_sha256",
        }
        unexpected_qualification_fields = sorted(
            set(qualification) - required_qualification_fields
        )
        missing_qualification_fields = sorted(
            required_qualification_fields - set(qualification)
        )
        if unexpected_qualification_fields or missing_qualification_fields:
            return {
                "success": False,
                "message": (
                    f"move_insert qualification for {requested_part} has invalid fields; "
                    f"missing={missing_qualification_fields}, "
                    f"unsupported={unexpected_qualification_fields}"
                ),
                "profile_sha256": profile_sha256,
                "validated_parts": validated_parts,
            }
        exact_values = {
            "robot": "ur5e",
            "tool_frame": "tool0",
            "destination_location": "assembly_board-v1",
            "part_name": requested_part,
            "profile_sha256": profile_sha256,
        }
        if selected_demonstration_recipe:
            exact_values["hard_caps_sha256"] = (
                selected_demonstration_recipe.get("hard_caps_sha256")
            )
            exact_values["recording_id"] = selected_demonstration_recipe.get(
                "recording_id"
            )
            exact_values["demonstration_sha256"] = (
                selected_demonstration_recipe.get("demonstration_sha256")
            )
        for field_name, expected_value in exact_values.items():
            if qualification.get(field_name) != expected_value:
                return {
                    "success": False,
                    "message": (
                        f"move_insert qualification for {requested_part}.{field_name} "
                        "does not match the selected physical profile"
                    ),
                    "profile_sha256": profile_sha256,
                    "validated_parts": validated_parts,
                }
        qualification_hard_caps_sha256 = qualification.get(
            "hard_caps_sha256"
        )
        try:
            valid_qualification_hard_caps_sha256 = bool(
                isinstance(qualification_hard_caps_sha256, str)
                and len(qualification_hard_caps_sha256) == 64
                and int(qualification_hard_caps_sha256, 16) >= 0
            )
        except ValueError:
            valid_qualification_hard_caps_sha256 = False
        if not valid_qualification_hard_caps_sha256:
            return {
                "success": False,
                "message": (
                    f"move_insert qualification for {requested_part}."
                    "hard_caps_sha256 is not a SHA-256 digest"
                ),
                "profile_sha256": profile_sha256,
                "validated_parts": validated_parts,
            }
        for field_name in (
            "trial_id",
            "place_approach_recording_sha256",
            "board_calibration_id",
            "board_geometry_sha256",
            "recording_id",
            "demonstration_sha256",
        ):
            value = qualification.get(field_name)
            if not isinstance(value, str) or not value or value != value.strip():
                return {
                    "success": False,
                    "message": (
                        f"move_insert qualification for {requested_part}.{field_name} "
                        "is missing or is not exact"
                    ),
                    "profile_sha256": profile_sha256,
                    "validated_parts": validated_parts,
                }
        for field_name in (
            "profile_sha256",
            "hard_caps_sha256",
            "place_approach_recording_sha256",
            "board_geometry_sha256",
            "demonstration_sha256",
            "qualification_policy_sha256",
            "qualification_evidence_sha256",
        ):
            value = str(qualification.get(field_name) or "")
            try:
                valid_digest = len(value) == 64 and int(value, 16) >= 0
            except ValueError:
                valid_digest = False
            if not valid_digest:
                return {
                    "success": False,
                    "message": (
                        f"move_insert qualification for {requested_part}.{field_name} "
                        "is not a SHA-256 digest"
                    ),
                    "profile_sha256": profile_sha256,
                    "validated_parts": validated_parts,
                }
        confirmed_at_text = qualification.get("confirmed_at")
        try:
            confirmed_at = datetime.fromisoformat(
                confirmed_at_text.replace("Z", "+00:00")
            )
        except (AttributeError, TypeError, ValueError):
            confirmed_at = None
        if (
            confirmed_at is None
            or confirmed_at.tzinfo is None
            or confirmed_at.utcoffset() != timezone.utc.utcoffset(confirmed_at)
        ):
            return {
                "success": False,
                "message": (
                    f"move_insert qualification for {requested_part}.confirmed_at "
                    "must be a UTC ISO-8601 timestamp"
                ),
                "profile_sha256": profile_sha256,
                "validated_parts": validated_parts,
            }
        for field_name in ("board_generation", "generation"):
            value = qualification.get(field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                return {
                    "success": False,
                    "message": (
                        f"move_insert qualification for {requested_part}.{field_name} "
                        "must be a positive integer"
                    ),
                    "profile_sha256": profile_sha256,
                    "validated_parts": validated_parts,
                }

        expected_policy_sha256 = _move_insert_qualification_policy_sha256()
        policy_values = {
            "qualification_policy_version": (
                _MOVE_INSERT_QUALIFICATION_POLICY_VERSION
            ),
            "qualification_policy_sha256": expected_policy_sha256,
            "required_confirmed_trials": (
                _MOVE_INSERT_REQUIRED_CONFIRMED_TRIALS
            ),
            "confirmed_trial_count": _MOVE_INSERT_REQUIRED_CONFIRMED_TRIALS,
        }
        for field_name, expected_value in policy_values.items():
            if qualification.get(field_name) != expected_value:
                return {
                    "success": False,
                    "message": (
                        f"move_insert qualification for {requested_part}."
                        f"{field_name} does not match the protected qualification "
                        "policy"
                    ),
                    "profile_sha256": profile_sha256,
                    "validated_parts": validated_parts,
                }

        confirmed_trial_ids = qualification.get("confirmed_trial_ids")
        confirmed_trial_result_sha256s = qualification.get(
            "confirmed_trial_result_sha256s"
        )
        confirmed_trial_trace_sha256s = qualification.get(
            "confirmed_trial_trace_sha256s"
        )
        evidence_lists = {
            "confirmed_trial_ids": confirmed_trial_ids,
            "confirmed_trial_result_sha256s": confirmed_trial_result_sha256s,
            "confirmed_trial_trace_sha256s": confirmed_trial_trace_sha256s,
        }
        for field_name, values in evidence_lists.items():
            if (
                not isinstance(values, list)
                or len(values) != _MOVE_INSERT_REQUIRED_CONFIRMED_TRIALS
                or any(
                    not isinstance(value, str)
                    or not value
                    or value != value.strip()
                    for value in values
                )
            ):
                return {
                    "success": False,
                    "message": (
                        f"move_insert qualification for {requested_part}."
                        f"{field_name} must contain exactly one exact value"
                    ),
                    "profile_sha256": profile_sha256,
                    "validated_parts": validated_parts,
                }
        if len(set(confirmed_trial_ids)) != _MOVE_INSERT_REQUIRED_CONFIRMED_TRIALS:
            return {
                "success": False,
                "message": (
                    f"move_insert qualification for {requested_part}."
                    "confirmed_trial_ids must identify one trial"
                ),
                "profile_sha256": profile_sha256,
                "validated_parts": validated_parts,
            }
        if qualification.get("trial_id") != confirmed_trial_ids[-1]:
            return {
                "success": False,
                "message": (
                    f"move_insert qualification for {requested_part}.trial_id "
                    "must identify the confirmed trial"
                ),
                "profile_sha256": profile_sha256,
                "validated_parts": validated_parts,
            }
        for field_name in (
            "confirmed_trial_result_sha256s",
            "confirmed_trial_trace_sha256s",
        ):
            for value in qualification[field_name]:
                try:
                    valid_digest = len(value) == 64 and int(value, 16) >= 0
                except ValueError:
                    valid_digest = False
                if not valid_digest:
                    return {
                        "success": False,
                        "message": (
                            f"move_insert qualification for {requested_part}."
                            f"{field_name} contains a non-SHA-256 value"
                        ),
                        "profile_sha256": profile_sha256,
                        "validated_parts": validated_parts,
                    }
        qualification_identity = {
            field_name: qualification.get(field_name)
            for field_name in _MOVE_INSERT_QUALIFICATION_IDENTITY_FIELDS
        }
        expected_evidence_sha256 = _move_insert_qualification_evidence_sha256(
            qualification_identity=qualification_identity,
            trial_ids=confirmed_trial_ids,
            result_sha256s=confirmed_trial_result_sha256s,
            trace_sha256s=confirmed_trial_trace_sha256s,
        )
        if (
            qualification.get("qualification_evidence_sha256")
            != expected_evidence_sha256
        ):
            return {
                "success": False,
                "message": (
                    f"move_insert qualification for {requested_part}."
                    "qualification_evidence_sha256 does not match the "
                    "confirmed trial"
                ),
                "profile_sha256": profile_sha256,
                "validated_parts": validated_parts,
            }

    return {
        "success": True,
        "message": f"move_insert profile resolved for {requested_part}",
        "part_name": requested_part,
        "calibration_id": effective_calibration_id,
        "shared_calibration_id": shared_calibration_id,
        "override_calibration_id": override_calibration_id,
        "profile_sha256": profile_sha256,
        "validated_parts": validated_parts,
        "qualification": deepcopy(qualification),
        "demonstration_recipe": deepcopy(selected_demonstration_recipe),
        **effective_values,
    }


def _validated_frozen_move_insert_profile(
    profile: Mapping[str, Any] | None,
    *,
    part_name: str,
    profile_sha256: Any,
) -> dict[str, Any]:
    if not isinstance(profile, Mapping):
        return {"success": False, "message": "frozen move_insert_profile is missing"}
    frozen = dict(profile)
    frozen_part = frozen.get("part_name", part_name)
    if frozen_part != part_name:
        return {
            "success": False,
            "message": (
                "frozen move_insert_profile part_name does not match the held part: "
                f"expected {part_name!r}, found {frozen_part or '<empty>'!r}"
            ),
        }
    shared_calibration_id = frozen.get("shared_calibration_id")
    override_calibration_id = frozen.get("override_calibration_id")
    effective_calibration_id = frozen.get("calibration_id")
    if (
        not isinstance(shared_calibration_id, str)
        or not shared_calibration_id
        or shared_calibration_id != shared_calibration_id.strip()
    ):
        return {
            "success": False,
            "message": "frozen move_insert shared_calibration_id is missing or is not exact",
        }
    if override_calibration_id is not None and (
        not isinstance(override_calibration_id, str)
        or not override_calibration_id
        or override_calibration_id != override_calibration_id.strip()
    ):
        return {
            "success": False,
            "message": "frozen move_insert override_calibration_id is not exact",
        }
    expected_calibration_id = override_calibration_id or shared_calibration_id
    if effective_calibration_id != expected_calibration_id:
        return {
            "success": False,
            "message": (
                "frozen move_insert calibration_id does not match its selected "
                "shared/override calibration identity"
            ),
        }
    raw_demonstration_recipe = frozen.get("demonstration_recipe", {})
    if not isinstance(raw_demonstration_recipe, Mapping):
        return {
            "success": False,
            "message": "frozen move_insert_profile demonstration_recipe is not an object",
        }
    synthetic_raw = {
        "calibration_id": effective_calibration_id,
        "validated_parts": [part_name],
        **{field: frozen.get(field) for field in _MOVE_INSERT_PROFILE_FIELDS},
        "part_overrides": {},
        "demonstration_recipes": (
            {part_name: deepcopy(dict(raw_demonstration_recipe))}
            if raw_demonstration_recipe
            else {}
        ),
        "qualifications": {},
    }
    validated = resolve_move_insert_profile(
        {"move_insert": synthetic_raw},
        part_name,
        require_qualification=False,
    )
    if not validated.get("success"):
        return validated
    frozen_hash = str(profile_sha256 or frozen.get("profile_sha256") or "").strip()
    try:
        valid_hash = len(frozen_hash) == 64 and int(frozen_hash, 16) >= 0
    except ValueError:
        valid_hash = False
    if not valid_hash:
        return {
            "success": False,
            "message": "frozen move_insert_profile_sha256 is missing or invalid",
        }
    supplied_profile_hash = str(frozen.get("profile_sha256") or "").strip()
    if supplied_profile_hash and supplied_profile_hash != frozen_hash:
        return {
            "success": False,
            "message": "frozen move_insert_profile hash does not match move_insert_profile_sha256",
        }
    raw_qualification = frozen.get("qualification", {})
    if not isinstance(raw_qualification, Mapping):
        return {
            "success": False,
            "message": "frozen move_insert_profile qualification is not an object",
        }
    validated["profile_sha256"] = frozen_hash
    validated["validated_parts"] = list(frozen.get("validated_parts") or [part_name])
    validated["calibration_id"] = effective_calibration_id
    validated["shared_calibration_id"] = shared_calibration_id
    validated["override_calibration_id"] = override_calibration_id
    validated["qualification"] = deepcopy(dict(raw_qualification))
    validated["demonstration_recipe"] = deepcopy(
        dict(raw_demonstration_recipe)
    )
    if raw_demonstration_recipe:
        validated["force_depth_profile"] = deepcopy(
            dict(raw_demonstration_recipe.get("force_depth_profile") or {})
        )
    return validated


def derive_move_insert_timeout_sec(
    expected_start_pose: Mapping[str, Any],
    target_pose: Mapping[str, Any],
    insertion_axis_world: Mapping[str, Any],
    profile: Mapping[str, Any],
    *,
    part_name: str = "",
    insert_max_timeout_sec: Any = None,
) -> tuple[float, str]:
    """Derive the insertion action timeout from the frozen motion inputs."""
    try:
        delta = tuple(
            float(target_pose[field]) - float(expected_start_pose[field])
            for field in ("x", "y", "z")
        )
        axis = tuple(float(insertion_axis_world[field]) for field in ("x", "y", "z"))
        axis_norm = math.sqrt(sum(value * value for value in axis))
        if not math.isfinite(axis_norm) or axis_norm <= 1e-12:
            raise ValueError("insertion_axis_world must be a finite nonzero vector")
        normalized_axis = tuple(value / axis_norm for value in axis)
        axial_distance = sum(
            value * direction for value, direction in zip(delta, normalized_axis, strict=True)
        )
        contact_speed = float(profile["contact_speed_m_s"])
        radius = float(profile["spiral_radius_m"])
        pitch = float(profile["spiral_pitch_m"])
        spiral_speed = float(profile["spiral_speed_m_s"])
        spiral_acceleration = float(profile["spiral_acceleration_m_s2"])
        settle_time = float(profile["settle_time_sec"])
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        return 0.0, f"cannot derive move_insert timeout: {exc}"
    numeric = (
        *delta,
        *normalized_axis,
        axial_distance,
        contact_speed,
        radius,
        pitch,
        spiral_speed,
        spiral_acceleration,
        settle_time,
    )
    if not all(math.isfinite(value) for value in numeric):
        return 0.0, "cannot derive move_insert timeout from non-finite values"
    if axial_distance <= 0.0:
        return 0.0, "move_insert target must have positive progress along insertion_axis_world"
    if min(contact_speed, pitch, spiral_speed, spiral_acceleration) <= 0.0:
        return 0.0, "move_insert timeout speeds, pitch, and acceleration must be positive"
    if radius < 0.0 or settle_time < 0.0:
        return 0.0, "move_insert timeout radius and settle_time_sec must be non-negative"

    contact_time = axial_distance / contact_speed
    if radius == 0.0:
        spiral_time = 0.0
        ramp_time = 0.0
    else:
        b = pitch / (2.0 * math.pi)
        theta_max = 2.0 * math.pi * radius / pitch
        arc_length = 0.5 * b * (
            theta_max * math.sqrt(1.0 + theta_max * theta_max)
            + math.asinh(theta_max)
        )
        spiral_time = arc_length / spiral_speed
        ramp_time = 2.0 * spiral_speed / spiral_acceleration
    engagement_hold_sec = max(0.10, min(settle_time, 0.25))
    stall_hold_sec = max(0.10, min(settle_time, 0.50))
    seated_hold_sec = max(0.10, settle_time)
    force_filter_window_sec = max(0.06, min(settle_time, 0.10))
    contact_hold_sec = max(0.06, min(engagement_hold_sec, 0.10))
    scheduling_margin_sec = 0.10
    timeout_sec = (
        contact_time
        + spiral_time
        + ramp_time
        + force_filter_window_sec
        + contact_hold_sec
        + stall_hold_sec
        + engagement_hold_sec
        + seated_hold_sec
        + scheduling_margin_sec
    )
    if not math.isfinite(timeout_sec) or timeout_sec <= 0.0:
        return 0.0, "derived move_insert timeout is not finite and positive"
    if part_name in _MOVE_INSERT_SUPPORTED_PARTS:
        try:
            protected_timeout_sec = float(insert_max_timeout_sec)
        except (TypeError, ValueError, OverflowError):
            protected_timeout_sec = math.nan
        if not math.isfinite(protected_timeout_sec) or protected_timeout_sec <= 0.0:
            return 0.0, (
                f"cannot derive {part_name} move_insert recovery timeout without the exact "
                "insert_max_timeout_sec hard cap"
            )
        timeout_sec = protected_timeout_sec
    return timeout_sec, ""


def _quaternion_multiply(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    lx, ly, lz, lw = left
    rx, ry, rz, rw = right
    return (
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
        lw * rw - lx * rx - ly * ry - lz * rz,
    )


def _rotate_vector(
    quaternion: tuple[float, float, float, float],
    vector: tuple[float, float, float],
) -> tuple[float, float, float]:
    qx, qy, qz, qw = quaternion
    vx, vy, vz = vector
    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)
    return (
        vx + qw * tx + qy * tz - qz * ty,
        vy + qw * ty + qz * tx - qx * tz,
        vz + qw * tz + qx * ty - qy * tx,
    )


def _compose_pose(
    parent: Mapping[str, Any],
    relative: Mapping[str, Any],
) -> dict[str, float]:
    parent_pose = _pose_from_mapping(dict(parent))
    relative_pose = _pose_from_mapping(dict(relative))
    parent_quaternion = tuple(
        parent_pose[field] for field in ("qx", "qy", "qz", "qw")
    )
    relative_quaternion = tuple(
        relative_pose[field] for field in ("qx", "qy", "qz", "qw")
    )
    rotated = _rotate_vector(
        parent_quaternion,
        tuple(relative_pose[field] for field in ("x", "y", "z")),
    )
    quaternion, quaternion_error = _normalized_optional_quaternion(
        *_quaternion_multiply(parent_quaternion, relative_quaternion)
    )
    if quaternion_error or quaternion is None:
        raise ValueError(quaternion_error or "composed pose quaternion is invalid")
    return {
        "x": parent_pose["x"] + rotated[0],
        "y": parent_pose["y"] + rotated[1],
        "z": parent_pose["z"] + rotated[2],
        **dict(zip(("qx", "qy", "qz", "qw"), quaternion, strict=True)),
    }


def _inverse_pose(pose: Mapping[str, Any]) -> dict[str, float]:
    normalized = _pose_from_mapping(dict(pose))
    inverse_quaternion = (
        -normalized["qx"],
        -normalized["qy"],
        -normalized["qz"],
        normalized["qw"],
    )
    inverse_translation = _rotate_vector(
        inverse_quaternion,
        (-normalized["x"], -normalized["y"], -normalized["z"]),
    )
    return {
        "x": inverse_translation[0],
        "y": inverse_translation[1],
        "z": inverse_translation[2],
        **dict(
            zip(
                ("qx", "qy", "qz", "qw"),
                inverse_quaternion,
                strict=True,
            )
        ),
    }


def _pose_delta(left: dict[str, float], right: dict[str, float]) -> tuple[float, float]:
    translation_m = math.sqrt(
        sum((float(left[field]) - float(right[field])) ** 2 for field in ("x", "y", "z"))
    )
    left_q = tuple(float(left[field]) for field in ("qx", "qy", "qz", "qw"))
    right_q = tuple(float(right[field]) for field in ("qx", "qy", "qz", "qw"))
    dot = abs(sum(a * b for a, b in zip(left_q, right_q, strict=True)))
    rotation_deg = math.degrees(2.0 * math.acos(max(-1.0, min(1.0, dot))))
    return translation_m, rotation_deg


def compute_move_insert_geometry(  # noqa: C901, PLR0912 - fail-closed geometry gates.
    *,
    part_name: str,
    product_geometry: Mapping[str, Any],
    assembly_board_v1_aruco: Mapping[str, Any],
    held_part_handoff: Mapping[str, Any],
    move_insert_profile: Mapping[str, Any],
    move_insert_profile_sha256: str,
    z_adjustment_m: float = 0.0,
) -> dict[str, Any]:
    """Compute complete physical UR5e insertion poses without commanding motion."""
    if part_name not in _MOVE_INSERT_SUPPORTED_PARTS:
        return {
            "success": False,
            "message": f"move_insert does not support exact part identifier {part_name!r}",
        }
    geometry = dict(product_geometry) if isinstance(product_geometry, Mapping) else {}
    target_reference = dict(geometry.get("target_reference") or {})
    if (
        target_reference.get("target_point") != "inserted_part_origin"
        or target_reference.get("surface_role") != "assembly_slot"
    ):
        return {
            "success": False,
            "message": (
                "move_insert requires target_reference.target_point="
                "'inserted_part_origin' and surface_role='assembly_slot'"
            ),
        }
    try:
        marker_pose = _pose_from_mapping(dict(assembly_board_v1_aruco).get("pose"))
    except (KeyError, TypeError, ValueError) as exc:
        return {
            "success": False,
            "message": f"assembly_board-v1 ArUco pose is missing or invalid: {exc}",
        }
    profile = _validated_frozen_move_insert_profile(
        move_insert_profile,
        part_name=part_name,
        profile_sha256=move_insert_profile_sha256,
    )
    if not profile.get("success"):
        return profile
    demonstration_recipe = dict(profile.get("demonstration_recipe") or {})
    if demonstration_recipe:
        handoff = (
            dict(held_part_handoff)
            if isinstance(held_part_handoff, Mapping)
            else {}
        )
        if not handoff:
            return {
                "success": False,
                "missing_held_part_handoff": True,
                "message": (
                    "move_insert learned geometry is valid; held_part_handoff is "
                    "required after pick_grasp to compute complete insertion poses"
                ),
            }
        if (
            handoff.get("part_name") != part_name
            or handoff.get("frame_id") != "world"
            or handoff.get("tool_frame") != "tool0"
            or handoff.get("part_frame") != "held_part_origin"
        ):
            return {"success": False, "message": "held-part SE(3) provenance is invalid"}
        try:
            world_tool0_at_grasp = _pose_from_mapping(
                handoff["world_tool0_pose_at_grasp"]
            )
            world_held_part_at_grasp = _pose_from_mapping(
                handoff["world_held_part_pose_at_grasp"]
            )
            tool0_to_held_part = _pose_from_mapping(
                handoff["tool0_to_held_part"]
            )
            recomposed_part = _compose_pose(
                world_tool0_at_grasp,
                tool0_to_held_part,
            )
            aruco_to_seated_held_part = _pose_from_mapping(
                demonstration_recipe["aruco_to_seated_held_part"]
            )
            aruco_axis = tuple(
                float(demonstration_recipe["aruco_insertion_axis"][field])
                for field in ("x", "y", "z")
            )
            pre_insert_offset_m = float(profile["pre_insert_offset_m"])
            z_adjustment = float(z_adjustment_m)
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            return {
                "success": False,
                "message": f"learned insertion geometry is incomplete: {exc}",
            }
        handoff_delta = _pose_delta(recomposed_part, world_held_part_at_grasp)
        if handoff_delta[0] > 1e-6 or handoff_delta[1] > 1e-4:
            return {
                "success": False,
                "message": (
                    "held-part SE(3) handoff does not reconstruct its frozen pick poses"
                ),
            }
        marker_quaternion = tuple(
            marker_pose[field] for field in ("qx", "qy", "qz", "qw")
        )
        insertion_axis_tuple = _rotate_vector(marker_quaternion, aruco_axis)
        axis_norm = math.sqrt(sum(value * value for value in insertion_axis_tuple))
        if not math.isfinite(axis_norm) or axis_norm <= 1e-12:
            return {"success": False, "message": "learned insertion axis is invalid"}
        insertion_axis_world = dict(
            zip(
                ("x", "y", "z"),
                (value / axis_norm for value in insertion_axis_tuple),
                strict=True,
            )
        )
        target_part_origin = _compose_pose(
            marker_pose,
            aruco_to_seated_held_part,
        )
        insert_pose = _compose_pose(
            target_part_origin,
            _inverse_pose(tool0_to_held_part),
        )
        insert_pose["z"] += z_adjustment
        pre_insert_pose = {
            axis: insert_pose[axis]
            - insertion_axis_world[axis] * pre_insert_offset_m
            for axis in ("x", "y", "z")
        }
        pre_insert_pose.update(
            {field: insert_pose[field] for field in ("qx", "qy", "qz", "qw")}
        )
        approach_pose = {
            axis: pre_insert_pose[axis] - insertion_axis_world[axis] * 0.05
            for axis in ("x", "y", "z")
        }
        approach_pose.update(
            {field: insert_pose[field] for field in ("qx", "qy", "qz", "qw")}
        )
        timeout_sec, timeout_error = derive_move_insert_timeout_sec(
            pre_insert_pose,
            insert_pose,
            insertion_axis_world,
            profile,
            part_name=part_name,
            insert_max_timeout_sec=dict(
                demonstration_recipe.get("hard_caps") or {}
            ).get("insert_max_timeout_sec"),
        )
        if timeout_error:
            return {"success": False, "message": timeout_error}
        return {
            "success": True,
            "part_name": part_name,
            "assembly_board_v1_pose": marker_pose,
            "assembly_board_v1_registration": {
                "calibration_id": str(
                    demonstration_recipe.get("board_calibration_id") or ""
                ),
                "x": 0.0,
                "y": 0.0,
                "z": 0.0,
                "qx": 0.0,
                "qy": 0.0,
                "qz": 0.0,
                "qw": 1.0,
            },
            "target_origin_pose": target_part_origin,
            "approach_pose": approach_pose,
            "pre_insert_pose": pre_insert_pose,
            "insert_pose": insert_pose,
            "insertion_axis_world": insertion_axis_world,
            "move_insert_profile": {
                key: deepcopy(value)
                for key, value in profile.items()
                if key not in {"success", "message"}
            },
            "move_insert_profile_sha256": profile["profile_sha256"],
            "move_insert_timeout_sec": timeout_sec,
            "slot_x": target_part_origin["x"],
            "slot_y": target_part_origin["y"],
            "board_top_z": target_part_origin["z"],
            "part_height": float(geometry.get("part_height_m") or 0.0),
            "place_part_origin_z": target_part_origin["z"],
            "place_z": insert_pose["z"],
            "geometry_source": "insertion_demonstration",
        }
    try:
        registration = dict(
            geometry["assembly_board-v1_aruco_to_assembly_board-v1"]
        )
        marker_to_board = _pose_from_mapping(registration)
        world_board_pose = _compose_pose(marker_pose, marker_to_board)
    except (KeyError, TypeError, ValueError) as exc:
        return {
            "success": False,
            "message": (
                "assembly_board-v1_aruco_to_assembly_board-v1 is missing or invalid: "
                f"{exc}"
            ),
        }
    registration_calibration_id = registration.get("calibration_id")
    if (
        not isinstance(registration_calibration_id, str)
        or not registration_calibration_id
        or registration_calibration_id != registration_calibration_id.strip()
    ):
        return {
            "success": False,
            "message": (
                "assembly_board-v1_aruco_to_assembly_board-v1.calibration_id "
                "is missing or is not exact"
            ),
        }
    board_center = dict(geometry.get("board_center") or {})
    slot_xy = geometry.get("slot_xy")
    try:
        local_slot_x = float(slot_xy[0])
        local_slot_y = float(slot_xy[1])
        local_slot_floor_z = float(geometry["slot_floor_z_m"]) - float(
            board_center["z"]
        )
        part_height_m = float(geometry["part_height_m"])
        z_adjustment = float(z_adjustment_m)
    except (KeyError, IndexError, TypeError, ValueError, OverflowError) as exc:
        return {"success": False, "message": f"assembly_slot geometry is invalid: {exc}"}
    if not all(
        math.isfinite(value)
        for value in (
            local_slot_x,
            local_slot_y,
            local_slot_floor_z,
            part_height_m,
            z_adjustment,
        )
    ) or part_height_m <= 0.0:
        return {"success": False, "message": "assembly_slot geometry is not finite"}
    handoff = dict(held_part_handoff) if isinstance(held_part_handoff, Mapping) else {}
    if not handoff:
        return {
            "success": False,
            "missing_held_part_handoff": True,
            "message": (
                "move_insert static geometry is valid; held_part_handoff is required "
                "after pick_grasp to compute complete insertion poses"
            ),
        }
    if (
        handoff.get("part_name") != part_name
        or handoff.get("frame_id") != "world"
        or handoff.get("tool_frame") != "tool0"
        or handoff.get("part_frame") != "held_part_origin"
    ):
        return {"success": False, "message": "held-part SE(3) provenance is invalid"}
    try:
        world_tool0_at_grasp = _pose_from_mapping(
            handoff["world_tool0_pose_at_grasp"]
        )
        world_held_part_at_grasp = _pose_from_mapping(
            handoff["world_held_part_pose_at_grasp"]
        )
        tool0_to_held_part = _pose_from_mapping(handoff["tool0_to_held_part"])
        recomposed_part = _compose_pose(world_tool0_at_grasp, tool0_to_held_part)
    except (KeyError, TypeError, ValueError) as exc:
        return {
            "success": False,
            "message": f"held-part SE(3) handoff is incomplete: {exc}",
        }
    handoff_delta = _pose_delta(recomposed_part, world_held_part_at_grasp)
    if handoff_delta[0] > 1e-6 or handoff_delta[1] > 1e-4:
        return {
            "success": False,
            "message": "held-part SE(3) handoff does not reconstruct its frozen pick poses",
        }
    board_quaternion = tuple(
        world_board_pose[field] for field in ("qx", "qy", "qz", "qw")
    )
    board_normal = _rotate_vector(board_quaternion, (0.0, 0.0, 1.0))
    insertion_axis_world = {
        "x": -board_normal[0],
        "y": -board_normal[1],
        "z": -board_normal[2],
    }
    local_part_origin = (
        local_slot_x,
        local_slot_y,
        local_slot_floor_z + part_height_m * 0.5,
    )
    rotated_part_origin = _rotate_vector(board_quaternion, local_part_origin)
    target_part_origin = {
        "x": world_board_pose["x"] + rotated_part_origin[0],
        "y": world_board_pose["y"] + rotated_part_origin[1],
        "z": world_board_pose["z"] + rotated_part_origin[2],
        **{
            field: world_board_pose[field]
            for field in ("qx", "qy", "qz", "qw")
        },
    }
    insert_pose = _compose_pose(target_part_origin, _inverse_pose(tool0_to_held_part))
    insert_pose["z"] += z_adjustment
    pre_insert_offset_m = float(profile["pre_insert_offset_m"])
    pre_insert_pose = {
        axis: insert_pose[axis] - insertion_axis_world[axis] * pre_insert_offset_m
        for axis in ("x", "y", "z")
    }
    pre_insert_pose.update(
        {field: insert_pose[field] for field in ("qx", "qy", "qz", "qw")}
    )
    approach_pose = {
        axis: pre_insert_pose[axis] - insertion_axis_world[axis] * 0.05
        for axis in ("x", "y", "z")
    }
    approach_pose.update(
        {field: insert_pose[field] for field in ("qx", "qy", "qz", "qw")}
    )
    timeout_sec, timeout_error = derive_move_insert_timeout_sec(
        pre_insert_pose,
        insert_pose,
        insertion_axis_world,
        profile,
    )
    if timeout_error:
        return {"success": False, "message": timeout_error}
    return {
        "success": True,
        "part_name": part_name,
        "assembly_board_v1_pose": world_board_pose,
        "assembly_board_v1_registration": {
            "calibration_id": registration_calibration_id,
            **{
                field: float(registration[field])
                for field in ("x", "y", "z", "qx", "qy", "qz", "qw")
            },
        },
        "target_origin_pose": target_part_origin,
        "approach_pose": approach_pose,
        "pre_insert_pose": pre_insert_pose,
        "insert_pose": insert_pose,
        "insertion_axis_world": insertion_axis_world,
        "move_insert_profile": {
            key: deepcopy(value)
            for key, value in profile.items()
            if key not in {"success", "message"}
        },
        "move_insert_profile_sha256": profile["profile_sha256"],
        "move_insert_timeout_sec": timeout_sec,
        "slot_x": target_part_origin["x"],
        "slot_y": target_part_origin["y"],
        "board_top_z": world_board_pose["z"]
        + _rotate_vector(
            board_quaternion,
            (local_slot_x, local_slot_y, local_slot_floor_z),
        )[2],
        "part_height": part_height_m,
        "place_part_origin_z": target_part_origin["z"],
        "place_z": insert_pose["z"],
    }


def _physical_detection_error(
    detection: dict[str, Any],
    *,
    requested_part: str,
    now: float | None = None,
) -> str:
    detected_part = str(detection.get("part_name") or "").strip()
    if requested_part and detected_part != requested_part:
        return (
            f"physical detection part mismatch: requested {requested_part!r}, "
            f"received {detected_part or '<empty>'!r}"
        )

    coordinates: dict[str, float] = {}
    for axis in ("x", "y", "z"):
        try:
            value = float(detection[axis])
        except (KeyError, TypeError, ValueError, OverflowError):
            return f"physical detection {axis} coordinate is missing or invalid"
        if not math.isfinite(value):
            return f"physical detection {axis} coordinate is not finite"
        coordinates[axis] = value

    if str(detection.get("frame_id") or "").strip() != "world":
        return "physical detection frame_id must be world"

    try:
        captured_at = float(detection["captured_at"])
    except (KeyError, TypeError, ValueError, OverflowError):
        return "physical detection captured_at is missing or invalid"
    if not math.isfinite(captured_at):
        return "physical detection captured_at is not finite"
    detection_age = float(time.time() if now is None else now) - captured_at
    if (
        detection_age < -_PHYSICAL_DETECTION_FUTURE_TOLERANCE_SEC
        or detection_age > _PHYSICAL_DETECTION_MAX_AGE_SEC
    ):
        return f"physical detection is stale (age={detection_age:.2f}s)"

    try:
        table_surface_z_m = float(detection["table_surface_z_m"])
    except (KeyError, TypeError, ValueError, OverflowError):
        return "physical detection has no accepted table-plane evidence"
    if not math.isfinite(table_surface_z_m):
        return "physical detection table_surface_z_m is not finite"
    if detection.get("table_plane_ready") is False:
        return "physical detection table-plane evidence is not ready"
    if detection.get("table_plane_accepted") is False:
        return "physical detection table-plane evidence is not accepted"

    detection.update(coordinates)
    detection["captured_at"] = captured_at
    detection["table_surface_z_m"] = table_surface_z_m
    return ""


def _gazebo_world_file_candidates() -> list[Path]:
    candidates: list[Path] = []
    env_path = str(os.environ.get("CAIS_GAZEBO_WORLD_FILE") or "").strip()
    if env_path:
        candidates.append(Path(env_path).expanduser())
    module_path = Path(__file__).resolve()
    for parent in module_path.parents:
        candidates.append(parent / "ros2/cais_lab_robotics/worlds/table_recovery_framework.world")
    candidates.append(Path.home() / "ros2_ws/src/cais_lab_robotics/worlds/table_recovery_framework.world")
    return candidates


def _gazebo_model_file_candidates(model_name: str) -> list[Path]:
    candidates: list[Path] = []
    module_path = Path(__file__).resolve()
    for parent in module_path.parents:
        candidates.append(
            parent / f"ros2/cais_lab_robotics/models/{model_name}/model.sdf"
        )
    candidates.append(
        Path.home()
        / f"ros2_ws/install/cais_lab_robotics/share/cais_lab_robotics/models/{model_name}/model.sdf"
    )
    return candidates


def _footprint_width_from_xml(root: ET.Element) -> float | None:
    for geometry in root.iter("geometry"):
        cylinder = geometry.find("cylinder")
        if cylinder is not None:
            radius = _as_float(cylinder.findtext("radius"), 0.0)
            if radius > 0.0:
                return radius * 2.0
        box = geometry.find("box")
        if box is not None:
            tokens = str(box.findtext("size") or "").split()
            if len(tokens) >= 2:
                try:
                    return max(float(tokens[0]), float(tokens[1]))
                except (TypeError, ValueError):
                    continue
    return None


def _model_footprint_width_from_gazebo_world(model_name: str) -> float | None:
    target_model = str(model_name or "").strip()
    if not target_model:
        return None
    for world_path in _gazebo_world_file_candidates():
        if not world_path.is_file():
            continue
        try:
            root = ET.parse(world_path).getroot()
        except Exception:
            continue
        for model in root.iter("model"):
            if str(model.attrib.get("name") or "").strip() != target_model:
                continue
            width = _footprint_width_from_xml(model)
            if width is not None:
                return width
        included_names = {
            str(include.findtext("name") or "").strip()
            for include in root.iter("include")
        }
        if target_model not in included_names:
            continue
        for model_path in _gazebo_model_file_candidates(target_model):
            if not model_path.is_file():
                continue
            try:
                width = _footprint_width_from_xml(ET.parse(model_path).getroot())
            except (ET.ParseError, OSError):
                continue
            if width is not None:
                return width
    return None


def _gazebo_timing_scale_from_env(execution_mode: str) -> float:
    """Clock acceleration replaces the legacy simulation wait multiplier."""
    return 1.0


class GazeboPickPlaceController:
    """
    Generic Gazebo pick/place controller for a single robot.

    Public phase methods map to framework tool names:
      - pick_approach
      - pick_grasp
      - place_approach
      - place_insert
      - move_home
    """

    def __init__(
        self,
        *,
        robot_name: str,
        node_name: str,
        controller_config: dict[str, Any],
        named_positions: dict[str, Any] | None = None,
        execution_mode: str = "simulation",
        arm_joint_names: list[str] | None = None,
        arm_trajectory_topic: str | None = None,
        joint_states_topic: str = "/joint_states",
    ) -> None:
        self.robot_name = robot_name
        self.node_name = node_name
        self.execution_mode = str(execution_mode or "simulation").strip().lower()
        self.controller_config = controller_config or {}
        self.named_positions = named_positions or {}
        self.arm_joint_names = list(arm_joint_names or [])
        self.arm_trajectory_topic = arm_trajectory_topic
        self.joint_states_topic = joint_states_topic
        self._last_failure_message = ""
        self._planning_wall_time_sec = 0.
        self._trajectory_duration_sec = 0.
        self._simulation_goal = None
        self._last_simulation_controller_succeeded = False
        self._first_motion_at_unix = None
        self._simulation_joint_clients = {}
        self._prepared_motion_lock = threading.Lock()
        self._prepared_cartesian: dict[str, dict[str, Any]] = {}
        self._queued_motion_preparation: dict[str, Any] | None = None
        self._planning_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix=f"{node_name}_planning",
        )
        self._planning_future = None
        self._background_preparation_result: dict[str, Any] | None = None
        self._preparation_generation = 0
        self._state_validity_client = None
        self._scene_identity_client = None
        self._config_errors: list[str] = []

        move_group = self.controller_config.get("move_group", {})
        gripper = self.controller_config.get("gripper", {})
        services = self.controller_config.get("services", {})
        attach_cfg = self.controller_config.get("attach", {})
        motion = self.controller_config.get("motion", {})
        parts_tuning = self.controller_config.get("parts_tuning", {})

        def need_str(section: dict[str, Any], key: str, path: str) -> str:
            raw = section.get(key)
            if raw is None or str(raw).strip() == "":
                self._config_errors.append(path)
                return ""
            return str(raw)

        def need_float(section: dict[str, Any], key: str, path: str) -> float:
            raw = section.get(key)
            if raw is None:
                self._config_errors.append(path)
                return 0.0
            try:
                return float(raw)
            except Exception:
                self._config_errors.append(path)
                return 0.0

        def need_int(section: dict[str, Any], key: str, path: str) -> int:
            raw = section.get(key)
            if raw is None:
                self._config_errors.append(path)
                return 0
            try:
                return int(raw)
            except Exception:
                self._config_errors.append(path)
                return 0

        def opt_float(section: dict[str, Any], key: str, default: float) -> float:
            raw = section.get(key)
            if raw is None:
                return float(default)
            try:
                return float(raw)
            except Exception:
                return float(default)

        def opt_int(section: dict[str, Any], key: str, default: int) -> int:
            raw = section.get(key)
            if raw is None:
                return int(default)
            try:
                return int(raw)
            except Exception:
                return int(default)

        self.group_name = need_str(move_group, "group_name", "controller.move_group.group_name")
        self.ee_link = need_str(move_group, "ee_link", "controller.move_group.ee_link")
        self.tcp_link = need_str(move_group, "tcp_link", "controller.move_group.tcp_link")
        self.frame_id = need_str(move_group, "frame_id", "controller.move_group.frame_id")
        self.cartesian_position_tolerance_m = max(
            0.001,
            opt_float(move_group, "position_tolerance_m", 0.005),
        )
        self.cartesian_orientation_tolerance_rad = max(
            0.001,
            opt_float(move_group, "orientation_tolerance_rad", 0.02),
        )

        self.gripper_joint = need_str(gripper, "joint", "controller.gripper.joint")
        self.gripper_topic = need_str(gripper, "topic", "controller.gripper.topic")
        self.gripper_open = need_float(gripper, "open", "controller.gripper.open")
        self.gripper_close = need_float(gripper, "close", "controller.gripper.close")
        self.gripper_move_time_sec = need_float(
            gripper, "move_time_sec", "controller.gripper.move_time_sec"
        )
        self.gripper_settle_sec = need_float(gripper, "settle_sec", "controller.gripper.settle_sec")
        self.gripper_feedback_timeout_pad_sec = need_float(
            gripper,
            "feedback_timeout_pad_sec",
            "controller.gripper.feedback_timeout_pad_sec",
        )
        self.gripper_position_tol = need_float(
            gripper, "position_tolerance", "controller.gripper.position_tolerance"
        )

        self.service_detect_all = need_str(services, "detect_all", "controller.services.detect_all")
        self.service_motion_plan = need_str(
            services, "motion_plan", "controller.services.motion_plan"
        )
        self.service_cartesian_path = need_str(
            services, "cartesian_path", "controller.services.cartesian_path"
        )
        self.service_execute_traj = need_str(
            services, "execute_trajectory", "controller.services.execute_trajectory"
        )
        self.service_attach = need_str(services, "attach", "controller.services.attach")
        self.service_detach = need_str(services, "detach", "controller.services.detach")
        self.service_set_entity_state = need_str(
            services, "set_entity_state", "controller.services.set_entity_state"
        )
        self.service_get_entity_state = str(
            services.get("get_entity_state") or "/get_entity_state"
        ).strip()

        self.robot_model_name = need_str(
            attach_cfg, "robot_model_name", "controller.attach.robot_model_name"
        )
        raw_candidates = attach_cfg.get("attach_link_candidates")
        if isinstance(raw_candidates, list) and raw_candidates:
            self.attach_link_candidates = [str(v) for v in raw_candidates if str(v).strip()]
        else:
            self.attach_link_candidates = []
            self._config_errors.append("controller.attach.attach_link_candidates")
        raw_release_candidates = attach_cfg.get("release_detach_link_candidates")
        if isinstance(raw_release_candidates, list):
            self.release_detach_link_candidates = [
                str(v) for v in raw_release_candidates if str(v).strip()
            ]
        else:
            self.release_detach_link_candidates = []
        self.primary_attach_link = need_str(
            attach_cfg, "primary_attach_link", "controller.attach.primary_attach_link"
        )
        self.detach_timeout_sec = need_float(
            attach_cfg, "detach_timeout_sec", "controller.attach.detach_timeout_sec"
        )
        self.detach_max_link_attempts = need_int(
            attach_cfg,
            "detach_max_link_attempts",
            "controller.attach.detach_max_link_attempts",
        )
        self.release_detach_timeout_sec = max(
            self.detach_timeout_sec,
            opt_float(attach_cfg, "release_detach_timeout_sec", 5.0),
        )

        self.approach_height_m = need_float(
            motion, "approach_height_m", "controller.motion.approach_height_m"
        )
        self.pick_tcp_z_bias_max_m = need_float(
            motion, "pick_tcp_z_bias_max_m", "controller.motion.pick_tcp_z_bias_max_m"
        )
        self.pick_tcp_z_bias_min_m = need_float(
            motion, "pick_tcp_z_bias_min_m", "controller.motion.pick_tcp_z_bias_min_m"
        )
        raw_pick_tool0_z_adjustment_m = motion.get("pick_tool0_z_adjustment_m")
        self.pick_tool0_z_adjustment_m = 0.0
        if raw_pick_tool0_z_adjustment_m is not None:
            try:
                self.pick_tool0_z_adjustment_m = float(raw_pick_tool0_z_adjustment_m)
            except (TypeError, ValueError):
                self._config_errors.append("controller.motion.pick_tool0_z_adjustment_m")
            if not math.isfinite(self.pick_tool0_z_adjustment_m):
                self.pick_tool0_z_adjustment_m = 0.0
                self._config_errors.append("controller.motion.pick_tool0_z_adjustment_m")
        self.min_pick_tcp_z_m = need_float(
            motion, "min_pick_tcp_z_m", "controller.motion.min_pick_tcp_z_m"
        )
        self.place_surface_gap_m = need_float(
            motion, "place_surface_gap_m", "controller.motion.place_surface_gap_m"
        )
        self.release_preopen_settle_sec = need_float(
            motion,
            "release_preopen_settle_sec",
            "controller.motion.release_preopen_settle_sec",
        )
        self.release_postopen_settle_sec = need_float(
            motion,
            "release_postopen_settle_sec",
            "controller.motion.release_postopen_settle_sec",
        )
        self.release_postdetach_settle_sec = need_float(
            motion,
            "release_postdetach_settle_sec",
            "controller.motion.release_postdetach_settle_sec",
        )
        self.release_descend_time_scale = max(
            1.0,
            opt_float(
                motion,
                "release_descend_time_scale",
                1.35,
            ),
        )
        self.release_detach_retry_count = max(
            0,
            opt_int(motion, "release_detach_retry_count", 2),
        )
        self.release_detach_retry_delay_sec = max(
            0.0,
            opt_float(motion, "release_detach_retry_delay_sec", 0.35),
        )
        self.release_detach_verify_distance_m = max(
            0.005,
            opt_float(motion, "release_detach_verify_distance_m", 0.04),
        )
        self.release_detach_verify_timeout_sec = max(
            0.0,
            opt_float(motion, "release_detach_verify_timeout_sec", 0.75),
        )
        self.release_detach_verify_poll_sec = max(
            0.05,
            opt_float(motion, "release_detach_verify_poll_sec", 0.1),
        )
        self.release_best_effort_detach_timeout_sec = max(
            0.05,
            opt_float(
                motion,
                "release_best_effort_detach_timeout_sec",
                min(
                    self.release_detach_timeout_sec,
                    max(2.0, self.detach_timeout_sec),
                ),
            ),
        )
        self.snap_to_slot_timeout_sec = max(
            0.1,
            opt_float(motion, "snap_to_slot_timeout_sec", 5.0),
        )
        self.snap_to_slot_retry_count = max(
            0,
            opt_int(motion, "snap_to_slot_retry_count", 1),
        )
        self.snap_to_slot_retry_delay_sec = max(
            0.0,
            opt_float(motion, "snap_to_slot_retry_delay_sec", 0.25),
        )
        self.snap_to_slot_position_tolerance_m = max(
            0.0005,
            opt_float(motion, "snap_to_slot_position_tolerance_m", 0.001),
        )
        self.snap_to_slot_observation_samples = max(
            1,
            opt_int(motion, "snap_to_slot_observation_samples", 3),
        )
        self.snap_to_slot_observation_interval_sec = max(
            0.05,
            opt_float(motion, "snap_to_slot_observation_interval_sec", 0.1),
        )
        self.simulation_insert_start_position_tolerance_m = max(
            0.003,
            opt_float(
                motion,
                "simulation_insert_start_position_tolerance_m",
                self.cartesian_position_tolerance_m + 0.001,
            ),
        )
        self.release_retry_lift_m = max(
            0.0,
            opt_float(motion, "release_retry_lift_m", 0.005),
        )
        self.trajectory_time_scale = need_float(
            motion, "trajectory_time_scale", "controller.motion.trajectory_time_scale"
        )
        self.tf_lookup_timeout_sec = max(
            0.0,
            opt_float(motion, "tf_lookup_timeout_sec", 2.0),
        )
        self.named_pose_duration_sec = max(
            0.1,
            opt_float(motion, "named_pose_duration_sec", 4.0),
        )
        self.move_home_duration_sec = max(
            0.1,
            opt_float(motion, "move_home_duration_sec", self.named_pose_duration_sec),
        )
        self.xy_axis_step_m = max(
            0.01,
            opt_float(motion, "xy_axis_step_m", 1.0),
        )

        self.insertion_depth_m = need_float(
            parts_tuning, "insertion_depth_m", "controller.parts_tuning.insertion_depth_m"
        )
        raw_pick_z_adjustments = parts_tuning.get("pick_z_adjustments_m", {})
        self.pick_z_adjustments_m: dict[str, float] = {}
        if isinstance(raw_pick_z_adjustments, dict):
            for raw_key, raw_value in raw_pick_z_adjustments.items():
                key = str(raw_key or "").strip().upper()
                if not key:
                    continue
                try:
                    self.pick_z_adjustments_m[key] = float(raw_value)
                except (TypeError, ValueError):
                    self._config_errors.append(
                        f"controller.parts_tuning.pick_z_adjustments_m.{key}"
                    )
        if (
            self.pick_tcp_z_bias_min_m > 0.0
            and self.pick_tcp_z_bias_max_m > 0.0
            and self.pick_tcp_z_bias_min_m > self.pick_tcp_z_bias_max_m
        ):
            self._config_errors.append(
                "controller.motion.pick_tcp_z_bias_min_m<=pick_tcp_z_bias_max_m"
            )

        self._apply_gazebo_fast_timing_profile()

        self._config_valid = not self._config_errors
        if not self._config_valid:
            self._last_failure_message = "invalid controller_config; missing/invalid: " + ", ".join(
                sorted(set(self._config_errors))
            )

        self._initialized = False
        self._services_ready = False
        self._spin_thread: threading.Thread | None = None
        self._shutdown_requested = False
        self._executor = None

        self._rclpy = None
        self._node = None
        self._cb_group = None
        self._tf_buffer = None
        self._tf_listener = None
        self._cart_client = None
        self._exec_client = None
        self._detect_all_client_legacy = None
        self._attach_client = None
        self._detach_client = None
        self._set_state_client = None
        self._get_state_client = None
        self._gripper_pub = None
        self._arm_pub = None

        self._attach_srv = None
        self._detach_srv = None
        self._link_attacher_enabled = False
        self._attached_model: str | None = None
        self._attached_link: str | None = None

        self._joint_lock = threading.Lock()
        self._joint_positions: dict[str, float] = {}
        self._joint_received_times: dict[str, float] = {}
        self._joint_sim_stamps: dict[str, float] = {}
        self._joint_stable_since: dict[str, float] = {}
        self._last_joint_sim_time = None
        self._joint_state_received_monotonic = 0.0
        self._last_command_evidence: dict[str, Any] | None = None

        # Remembered start pose for move_home (set externally or by UI recovery).
        self._last_start_pose = None

    @staticmethod
    def _prepared_motion_key(params: dict[str, Any]) -> str:
        payload = json.dumps(params, sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def queue_next_motion_preparation(self, primitive: str, params: dict[str, Any]) -> bool:
        """Queue resolved plan-only work to begin after the current goal is accepted."""
        if (
            self.execution_mode != "simulation"
            or primitive != "move_cartesian"
            or self.controller_config.get("background_preparation_enabled") is not True
        ):
            self._queued_motion_preparation = None
            return False
        clean = {key: deepcopy(value) for key, value in params.items() if value is not None}
        required = ("x", "y", "z")
        if any(key not in clean for key in required):
            self._queued_motion_preparation = None
            return False
        self._queued_motion_preparation = {
            "primitive": primitive,
            "params": clean,
            "key": self._prepared_motion_key(clean),
            "attachment": self._attached_model,
            "generation": self._preparation_generation,
        }
        return True

    def clear_motion_preparation(self) -> None:
        """Invalidate queued and cached plan-only work without touching live motion."""
        with self._prepared_motion_lock:
            self._preparation_generation += 1
            self._queued_motion_preparation = None
            self._prepared_cartesian.clear()
            self._background_preparation_result = None

    def _scene_launch_identity(self) -> str:
        from rcl_interfaces.srv import GetParameters

        if self._scene_identity_client is None:
            self._scene_identity_client = self._node.create_client(
                GetParameters,
                "/KMR_base_controller/get_parameters",
                callback_group=self._cb_group,
            )
        if not self._scene_identity_client.wait_for_service(timeout_sec=0.2):
            return "recovery_framework"
        response = self._wait_future(
            self._scene_identity_client.call_async(
                GetParameters.Request(names=["launch_id"]),
            ),
            timeout_sec=1.,
            label="background planning scene identity",
            timeout_log_level="debug",
        )
        if response is None or not response.values:
            return "recovery_framework"
        return str(response.values[0].string_value or "recovery_framework")

    def _prepare_queued_cartesian(self, queued: dict[str, Any], start_trajectory, orientation) -> None:
        from cais_spade_llm.recovery_framework.planning_permit import scene_planning_permit

        started = time.monotonic()
        launch_id = self._scene_launch_identity()
        with scene_planning_permit(launch_id) as acquired:
            if not acquired or self._shutdown_requested:
                with self._prepared_motion_lock:
                    self._background_preparation_result = {
                        "key": queued["key"], "used": False,
                        "reason": "scene planning permit is busy" if not acquired else "controller stopping",
                    }
                return
            params = queued["params"]
            quaternion = [params.get(name) for name in ("qx", "qy", "qz", "qw")]
            if any(value is None for value in quaternion):
                quaternion = [orientation.x, orientation.y, orientation.z, orientation.w]
            target = self._make_pose(
                float(params["x"]), float(params["y"]), float(params["z"]),
                self._make_orientation(*(float(value) for value in quaternion)),
            )
            request = self._GetCartesianPath.Request()
            request.header.frame_id = self.frame_id
            request.header.stamp = self._node.get_clock().now().to_msg()
            request.group_name = self.group_name
            request.link_name = self.ee_link
            request.waypoints = [target]
            request.max_step = 0.01
            request.jump_threshold = 2.0
            request.avoid_collisions = True
            request.start_state.is_diff = True
            request.start_state.joint_state.name = list(start_trajectory.joint_names)
            request.start_state.joint_state.position = list(start_trajectory.points[-1].positions)
            response = self._wait_future(
                self._cart_client.call_async(request),
                timeout_sec=10.,
                label="background plan:move_cartesian",
                timeout_log_level="debug",
            )
            self._planning_wall_time_sec += time.monotonic() - started
            if response is None or response.fraction < .999 or self._shutdown_requested:
                with self._prepared_motion_lock:
                    self._background_preparation_result = {
                        "key": queued["key"], "used": False,
                        "reason": "planning did not produce a complete collision-free path",
                    }
                return
            with self._prepared_motion_lock:
                if queued["generation"] != self._preparation_generation:
                    self._background_preparation_result = {
                        "key": queued["key"], "used": False,
                        "reason": "preparation invalidated before completion",
                    }
                    return
                self._prepared_cartesian[queued["key"]] = {
                    "solution": response.solution,
                    "target": {
                        "x": float(params["x"]), "y": float(params["y"]),
                        "z": float(params["z"]),
                        "qx": float(quaternion[0]), "qy": float(quaternion[1]),
                        "qz": float(quaternion[2]), "qw": float(quaternion[3]),
                    },
                    "attachment": queued["attachment"],
                    "planned_at_unix": time.time(),
                    "planning_wall_time_sec": time.monotonic() - started,
                }
                self._background_preparation_result = {
                    "key": queued["key"], "used": False, "reason": "prepared",
                    "planning_wall_time_sec": time.monotonic() - started,
                }

    def _start_queued_motion_preparation(self, current_solution, orientation) -> None:
        queued = self._queued_motion_preparation
        self._queued_motion_preparation = None
        if queued is None or self._shutdown_requested:
            return
        if self._planning_future is not None and not self._planning_future.done():
            return
        self._planning_future = self._planning_executor.submit(
            self._prepare_queued_cartesian,
            deepcopy(queued),
            deepcopy(current_solution.joint_trajectory),
            deepcopy(orientation),
        )

    def _consume_prepared_cartesian(self, target):
        """Return a freshly revalidated matching trajectory, otherwise fall back."""
        match_key = None
        expected = (
            target.position.x, target.position.y, target.position.z,
            target.orientation.x, target.orientation.y,
            target.orientation.z, target.orientation.w,
        )
        with self._prepared_motion_lock:
            for key, entry in self._prepared_cartesian.items():
                values = entry["target"]
                candidate = tuple(values[name] for name in ("x", "y", "z", "qx", "qy", "qz", "qw"))
                if all(math.isclose(a, b, abs_tol=1e-6) for a, b in zip(expected, candidate)):
                    match_key = key
                    break
            entry = self._prepared_cartesian.pop(match_key, None) if match_key else None
            preparation_result = deepcopy(self._background_preparation_result)
        if entry is None:
            reason = "cache miss"
            if preparation_result is not None:
                reason = str(preparation_result.get("reason") or reason)
            return None, reason
        if entry["attachment"] != self._attached_model:
            return None, "attachment changed"
        trajectory = entry["solution"].joint_trajectory
        if not trajectory.points:
            return None, "prepared trajectory is empty"
        targets = dict(zip(trajectory.joint_names, trajectory.points[0].positions, strict=True))
        if not self._fresh_stable_joint_target(targets, tolerance=.02, stable_for_sec=0.):
            return None, "start joints changed"
        if self._state_validity_client is None:
            return None, "state validity service is unavailable"
        from moveit_msgs.srv import GetStateValidity
        for point in trajectory.points:
            request = GetStateValidity.Request()
            request.group_name = self.group_name
            request.robot_state.is_diff = True
            request.robot_state.joint_state.name = list(trajectory.joint_names)
            request.robot_state.joint_state.position = list(point.positions)
            response = self._wait_future(
                self._state_validity_client.call_async(request),
                timeout_sec=3.,
                label="prepared trajectory revalidation",
                timeout_log_level="debug",
            )
            if response is None or not response.valid:
                return None, "collision scene changed"
        with self._prepared_motion_lock:
            self._background_preparation_result = {
                "key": match_key, "used": True,
                "reason": "reused after complete collision validation",
                "planning_wall_time_sec": entry["planning_wall_time_sec"],
            }
        return entry["solution"], "reused after complete collision validation"

    def _apply_gazebo_fast_timing_profile(self, scale: float | None = None) -> None:
        """Use native simulation durations; speed is owned by the scene clock."""
        self._gazebo_wait_scale = 1.0
        if self.execution_mode == 'simulation':
            self.trajectory_time_scale = 1.0

    def _scaled_wall_wait_sec(self, seconds: float, *, minimum: float = 0.0) -> float:
        return max(float(minimum), float(seconds))

    def _wait_process_time(self, seconds: float) -> None:
        """Wait for settling in simulation time, keeping Stop responsive."""
        if self.execution_mode != 'simulation':
            time.sleep(max(0.0, seconds))
            return
        from cais_spade_llm.resources.robot.simulation_timing import wait_for_simulation

        if self._node is None and not self.init():
            raise RuntimeError('Simulation clock is unavailable')
        wait_for_simulation(
            max(0.0, seconds), now=lambda: self._node.get_clock().now().nanoseconds / 1e9,
            cancelled=lambda: self._shutdown_requested,
        )

    def _motion_pending(self, seconds: float):
        if self.execution_mode == 'simulation':
            from cais_spade_llm.resources.robot.simulation_timing import MotionDeadline
            return MotionDeadline(
                seconds, now=lambda: self._node.get_clock().now().nanoseconds / 1e9,
                cancelled=lambda: self._shutdown_requested,
            ).pending
        deadline = time.monotonic() + seconds
        return lambda: time.monotonic() < deadline

    def _simulation_joint_limits(self, names: list[str]) -> dict:
        """Read and cache the running robot model's limits, never guessed limits."""
        from rcl_interfaces.srv import GetParameters

        if getattr(self, '_limits_client', None) is None:
            self._limits_client = self._node.create_client(
                GetParameters, '/move_group/get_parameters', callback_group=self._cb_group)
        cache = getattr(self, '_joint_limits_cache', {})
        if all(name in cache for name in names):
            return {name: cache[name] for name in names}
        parameters = ['robot_description']
        parameters += [f'robot_description_planning.joint_limits.{name}.{field}'
                       for name in names for field in ('max_velocity', 'max_acceleration')]
        if not self._limits_client.wait_for_service(timeout_sec=5.0):
            raise RuntimeError('Running MoveIt joint limits are unavailable')
        response = self._wait_future(
            self._limits_client.call_async(GetParameters.Request(names=parameters)),
            timeout_sec=10.0, label='joint limits')
        if response is None or len(response.values) != len(parameters):
            raise RuntimeError('Running MoveIt joint limits are incomplete')
        root = ET.fromstring(response.values[0].string_value)
        for index, name in enumerate(names):
            element = root.find(f"joint[@name='{name}']/limit")
            if element is None:
                raise ValueError(f'Missing running joint limits: {name}')
            v, a = response.values[1+2*index:3+2*index]
            velocity = float(element.get('velocity'))
            if v.type != 0:
                velocity = min(velocity, v.double_value)
            if a.type == 0 or a.double_value <= 0:
                raise ValueError(f'Missing configured acceleration limit: {name}')
            cache[name] = {'lower': float(element.get('lower', '-inf')),
                           'upper': float(element.get('upper', 'inf')),
                           'velocity': velocity, 'acceleration': a.double_value}
        self._joint_limits_cache = cache
        return {name: cache[name] for name in names}

    def _validate_simulation_trajectory(self, trajectory) -> None:
        if self.execution_mode != 'simulation':
            return
        from cais_spade_llm.recovery_framework.kmr_motion import retime_trajectory

        retime_trajectory(trajectory, self._simulation_joint_limits(trajectory.joint_names), 1., 1.)

    def _simulation_trajectory_is_collision_free(self, trajectory, *, mating_contact: dict | None = None) -> bool:
        """Validate timed samples and their connecting motion before dispatch.

        Cartesian IK validity precedes time parameterization in MoveIt. Joint
        branch changes can create colliding interpolated motion even when the
        service reports a complete path.
        """
        if (self.execution_mode != "simulation" or not getattr(
            self, "controller_config", {}).get("payload_collision", {}).get("enabled")):
            return True
        from moveit_msgs.srv import GetStateValidity

        started = time.monotonic()
        evidence = {"checked_states": 0, "collision_free": False}
        self._last_path_validation = evidence
        if mating_contact:
            evidence['mating_contact'] = deepcopy(mating_contact)
            evidence['permitted_contact_states'] = 0
        previous = None
        try:
            for point in trajectory.points:
                current = list(point.positions)
                origin = current if previous is None else previous
                steps = max(1, math.ceil(max(abs(b - a) for a, b in zip(
                    origin, current, strict=True)) / .05))
                for index in range(1, steps + 1):
                    if self._shutdown_requested:
                        self._last_failure_message = "Motion validation cancelled"
                        return False
                    request = GetStateValidity.Request()
                    request.group_name = self.group_name
                    request.robot_state.is_diff = True
                    request.robot_state.joint_state.name = list(trajectory.joint_names)
                    request.robot_state.joint_state.position = [
                        a + (b - a) * index / steps for a, b in zip(origin, current, strict=True)
                    ]
                    response = None
                    deadline = time.monotonic() + float(getattr(self, "tf_lookup_timeout_sec", 5.))
                    while response is None and time.monotonic() < deadline:
                        if self._shutdown_requested:
                            self._last_failure_message = "Motion validation cancelled"
                            return False
                        evidence["observation_attempts"] = evidence.get("observation_attempts", 0) + 1
                        response = self._wait_future(
                            self._state_validity_client.call_async(request),
                            timeout_sec=min(1., max(.001, deadline - time.monotonic())),
                            label="timed trajectory collision validation", timeout_log_level="debug",
                        )
                    evidence["checked_states"] += 1
                    permitted_contact = False
                    if response is not None and not response.valid and mating_contact:
                        permitted_contact = self._validate_simulation_mating_contact(
                            mating_contact, response.contacts, request.robot_state)
                        if permitted_contact:
                            evidence['permitted_contact_states'] += 1
                    if response is None or (not response.valid and not permitted_contact):
                        evidence["observation_received"] = response is not None
                        evidence["contacts"] = [] if response is None else [
                            [contact.contact_body_1, contact.contact_body_2]
                            for contact in response.contacts
                        ]
                        self._last_failure_message = (
                            "Timed trajectory collision observation unavailable"
                            if response is None else "Timed trajectory is in collision"
                        )
                        self._last_command_evidence = {
                            "command_sent": False, "motion_path_validation": deepcopy(evidence),
                        }
                        return False
                previous = current
            evidence["collision_free"] = bool(trajectory.points)
            return evidence["collision_free"]
        finally:
            evidence["wall_time_sec"] = time.monotonic() - started
            self._planning_wall_time_sec += evidence["wall_time_sec"]

    def _time_joint_target(self, trajectory) -> None:
        """Bound the complete quintic transition from observed joints to a target."""
        if self.execution_mode != 'simulation':
            return
        limits = self._simulation_joint_limits(trajectory.joint_names)
        target = trajectory.points[-1]
        duration = target.time_from_start.sec + target.time_from_start.nanosec / 1e9
        for name, position in zip(trajectory.joint_names, target.positions, strict=True):
            deadline = time.monotonic() + max(0.0, float(getattr(self, "tf_lookup_timeout_sec", 2.0)))
            current = self._get_joint_position(name)
            while current is None and time.monotonic() < deadline:
                if getattr(self, "_shutdown_requested", False):
                    raise InterruptedError("Joint target timing was cancelled")
                time.sleep(0.01)
                current = self._get_joint_position(name)
            if current is None:
                raise ValueError(f'Missing observed joint position: {name}')
            bound = limits[name]
            if not math.isfinite(position) or not bound['lower'] <= position <= bound['upper']:
                raise ValueError(f'Target exceeds joint limits: {name}')
            distance = abs(position-current)
            duration = max(duration, 1.875*distance/bound['velocity'],
                           math.sqrt(5.78*distance/bound['acceleration']))
        target.velocities = [0.] * len(trajectory.joint_names)
        target.accelerations = [0.] * len(trajectory.joint_names)
        ns = math.ceil(duration * 1e9)
        target.time_from_start.sec, target.time_from_start.nanosec = divmod(ns, 1_000_000_000)
        self._trajectory_duration_sec += duration
        if len(trajectory.joint_names) > 1:
            self._last_arm_duration_sec = duration

    def _cancel_simulation_goal(self) -> None:
        goal = self._simulation_goal
        if self.execution_mode != 'simulation' or goal is None:
            return
        future = goal.cancel_goal_async()
        deadline = time.monotonic() + 3.
        while not future.done() and time.monotonic() < deadline:
            time.sleep(.01)
        self._simulation_goal = None

    def _note_motion_dispatch(self) -> None:
        """Record the first controller-accepted motion in this resource session."""
        if self._first_motion_at_unix is None:
            self._first_motion_at_unix = time.time()

    def _send_simulation_joint_trajectory(self, topic: str, trajectory) -> bool:
        from control_msgs.action import FollowJointTrajectory

        self._last_simulation_controller_succeeded = False
        if not topic.endswith('/joint_trajectory'):
            raise ValueError(f'No simulation controller action for {topic}')
        endpoint = topic.removesuffix('/joint_trajectory') + '/follow_joint_trajectory'
        if endpoint not in self._simulation_joint_clients:
            self._simulation_joint_clients[endpoint] = self._ActionClient(
                self._node, FollowJointTrajectory, endpoint, callback_group=self._cb_group)
        client = self._simulation_joint_clients[endpoint]
        if not client.wait_for_server(timeout_sec=5.):
            return False
        goal = self._wait_future(client.send_goal_async(
            FollowJointTrajectory.Goal(trajectory=trajectory)), 10., f'send:{endpoint}')
        if goal is None or not goal.accepted:
            return False
        self._note_motion_dispatch()
        self._simulation_goal = goal
        try:
            last = trajectory.points[-1].time_from_start
            result = self._wait_future(goal.get_result_async(),
                                       last.sec + last.nanosec/1e9 + 5., f'result:{endpoint}')
            if not (result and result.status == 4 and result.result.error_code == 0):
                return False
            self._last_simulation_controller_succeeded = True
            pending = self._motion_pending(5.)
            targets = dict(zip(trajectory.joint_names, trajectory.points[-1].positions, strict=True))
            last_measured: dict[str, float | None] = {}
            while pending():
                measured = {name: self._get_joint_position(name) for name in targets}
                last_measured = measured
                if all(
                    value is not None
                    and (
                        self._angular_joint_error(value, targets[name])
                        if name in self.arm_joint_names
                        else abs(value - targets[name])
                    )
                    <= 0.005
                    for name, value in measured.items()
                ):
                    return True
                time.sleep(.01)
            errors = {
                name: (
                    None
                    if last_measured.get(name) is None
                    else (
                        self._angular_joint_error(last_measured[name], target)
                        if name in self.arm_joint_names
                        else abs(last_measured[name] - target)
                    )
                )
                for name, target in targets.items()
            }
            self._last_failure_message = (
                "Simulation controller completed without observed joint targets: "
                f"errors={errors}; feedback_stamp={getattr(self, '_last_joint_sim_time', None)}; "
                f"simulation_time={self._node.get_clock().now().nanoseconds / 1e9}"
            )
            return False
        finally:
            if not goal.status in (4, 5, 6):
                self._cancel_simulation_goal()
            self._simulation_goal = None

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def _spin_executor(self) -> None:
        """Run executor spin loop and suppress teardown-time RCLError noise."""
        try:
            if self._executor is not None:
                self._executor.spin()
        except Exception as exc:
            msg = str(exc)
            is_context_invalid = (
                "context is not valid" in msg.lower() or exc.__class__.__name__ == "RCLError"
            )
            if self._shutdown_requested and is_context_invalid:
                self._log().debug(f"Executor stopped during shutdown: {msg}")
                return
            self._log().warning(
                f"Executor spin terminated unexpectedly for {self.node_name}: {msg}"
            )

    def init(self) -> bool:
        if self._initialized:
            return True

        if self._planning_executor is None:
            self._planning_executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix=f"{self.node_name}_planning",
            )

        if not self._config_valid:
            logger.error("[%s] %s", self.robot_name, self._last_failure_message)
            return False

        if self.execution_mode not in {"simulation", "physical"}:
            logger.warning(
                "[%s] Unknown execution_mode '%s'; treating as 'simulation'",
                self.robot_name,
                self.execution_mode,
            )

        try:
            import rclpy
            import tf2_ros
            from builtin_interfaces.msg import Duration
            from gazebo_msgs.srv import GetEntityState, SetEntityState
            from geometry_msgs.msg import Pose
            from moveit_msgs.action import ExecuteTrajectory
            from moveit_msgs.srv import GetCartesianPath, GetMotionPlan, GetStateValidity
            from rclpy.action import ActionClient
            from rclpy.callback_groups import ReentrantCallbackGroup
            from rclpy.executors import SingleThreadedExecutor
            from rclpy.qos import qos_profile_sensor_data
            from sensor_msgs.msg import JointState
            from std_srvs.srv import Trigger
            from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
        except Exception:
            logger.exception(
                "[%s] ROS2 imports failed. Is the ROS2 environment sourced?",
                self.robot_name,
            )
            self._last_failure_message = "ros2 imports failed (environment not sourced?)"
            return False

        with _RCLPY_INIT_LOCK:
            if not rclpy.ok():
                rclpy.init()

        self._rclpy = rclpy
        self._ActionClient = ActionClient
        self._ReentrantCallbackGroup = ReentrantCallbackGroup
        self._SingleThreadedExecutor = SingleThreadedExecutor
        self._Trigger = Trigger
        self._ExecuteTrajectory = ExecuteTrajectory
        self._GetCartesianPath = GetCartesianPath
        self._GetMotionPlan = GetMotionPlan
        self._SetEntityState = SetEntityState
        self._GetEntityState = GetEntityState
        self._Pose = Pose
        self._JointTrajectory = JointTrajectory
        self._JointTrajectoryPoint = JointTrajectoryPoint
        self._Duration = Duration
        self._JointState = JointState
        self._tf2_ros = tf2_ros

        from rclpy.parameter import Parameter
        self._node = rclpy.create_node(self.node_name, parameter_overrides=[
            Parameter('use_sim_time', value=self.execution_mode == 'simulation')])
        self._joint_limits_cache = {}
        self._limits_client = None
        from rcl_interfaces.msg import ParameterEvent
        def parameters_changed(event):
            if event.node in ('/move_group', '/KMR_base_controller'):
                self._joint_limits_cache.clear()
                self.clear_motion_preparation()
        self._node.create_subscription(ParameterEvent, '/parameter_events', parameters_changed, 10)
        self._cb_group = ReentrantCallbackGroup()

        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(
            self._tf_buffer, self._node,
            qos=(qos_profile_sensor_data
                 if self.execution_mode == "simulation"
                 and self.controller_config.get("payload_collision", {}).get("enabled") else None),
        )

        self._cart_client = self._node.create_client(
            GetCartesianPath, self.service_cartesian_path, callback_group=self._cb_group
        )
        self._motion_plan_client = self._node.create_client(
            GetMotionPlan, self.service_motion_plan, callback_group=self._cb_group
        )
        self._state_validity_client = self._node.create_client(
            GetStateValidity, '/check_state_validity', callback_group=self._cb_group
        )
        self._exec_client = ActionClient(
            self._node,
            ExecuteTrajectory,
            self.service_execute_traj,
            callback_group=self._cb_group,
        )
        self._detect_all_client_legacy = self._node.create_client(
            Trigger, self.service_detect_all, callback_group=self._cb_group
        )
        self._set_state_client = self._node.create_client(
            SetEntityState, self.service_set_entity_state, callback_group=self._cb_group
        )
        self._get_state_client = self._node.create_client(
            GetEntityState, self.service_get_entity_state, callback_group=self._cb_group
        )

        if self.gripper_topic and self.gripper_joint:
            self._gripper_pub = self._node.create_publisher(JointTrajectory, self.gripper_topic, 10)

        if self.arm_trajectory_topic:
            self._arm_pub = self._node.create_publisher(
                JointTrajectory, self.arm_trajectory_topic, 10
            )

        self._node.create_subscription(
            JointState, self.joint_states_topic, self._on_joint_state,
            qos_profile_sensor_data if self.execution_mode == "simulation" else 50,
        )

        self._attach_srv, self._detach_srv = _import_linkattacher_srvs()
        if self.execution_mode != "physical" and self._attach_srv and self._detach_srv:
            self._attach_client = self._node.create_client(
                self._attach_srv, self.service_attach, callback_group=self._cb_group
            )
            self._detach_client = self._node.create_client(
                self._detach_srv, self.service_detach, callback_group=self._cb_group
            )
            self._link_attacher_enabled = True
        else:
            self._link_attacher_enabled = False
            self._log().warn("linkattacher_msgs.srv not importable; attach/detach disabled")

        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)
        self._shutdown_requested = False
        self._spin_thread = threading.Thread(target=self._spin_executor, daemon=True)
        self._spin_thread.start()

        self._initialized = True
        self._log().info(
            f"[{self.robot_name}] Controller initialized (execution_mode={self.execution_mode})"
        )
        self._last_failure_message = ""
        return True

    def shutdown(self) -> None:
        if not self._initialized:
            return

        self._services_ready = False
        self._shutdown_requested = True
        self.clear_motion_preparation()
        if self.execution_mode == 'simulation':
            self._cancel_simulation_goal()

        try:
            if self._executor and self._node:
                try:
                    self._executor.remove_node(self._node)
                except Exception:
                    pass
                try:
                    self._executor.shutdown(timeout_sec=1.0)
                except TypeError:
                    self._executor.shutdown()
        except Exception:
            pass

        if self._spin_thread:
            self._spin_thread.join(timeout=2.0)

        try:
            if self._node:
                self._node.destroy_node()
        except Exception:
            pass

        self._spin_thread = None
        self._executor = None
        self._node = None
        self._initialized = False
        self._planning_executor.shutdown(wait=False, cancel_futures=True)
        self._planning_executor = None

    def is_usable(self) -> bool:
        """Whether this controller instance is healthy enough for reuse."""
        spin_alive = bool(self._spin_thread and self._spin_thread.is_alive())
        return bool(self._initialized and self._services_ready and spin_alive)

    def wait_for_services(self, timeout_sec: float = 60.0) -> bool:
        if not self.init():
            return False
        if self._services_ready:
            return True

        self._log().info("Waiting for services/actions...")
        deadline = time.monotonic() + timeout_sec

        if self.execution_mode != "physical" and not self._wait_service(
            self._detect_all_client_legacy, self.service_detect_all, deadline
        ):
            return False
        if not self._wait_service(self._cart_client, self.service_cartesian_path, deadline):
            return False
        if not self._wait_service(self._motion_plan_client, self.service_motion_plan, deadline):
            return False
        if not self._wait_action_server(self._exec_client, self.service_execute_traj, deadline):
            return False

        if self._link_attacher_enabled:
            if not self._wait_service(self._attach_client, self.service_attach, deadline):
                return False
            if not self._wait_service(self._detach_client, self.service_detach, deadline):
                return False

        while time.monotonic() < deadline:
            try:
                if self._tf_buffer.can_transform(
                    self.frame_id, self.ee_link, self._rclpy.time.Time()
                ):
                    break
            except Exception:
                pass
            time.sleep(0.1)
        else:
            self._log().error(f"TF not ready for {self.frame_id} -> {self.ee_link}")
            self._last_failure_message = f"tf not ready for {self.frame_id} -> {self.ee_link}"
            return False

        # Best effort gripper feedback readiness. Keep the warmup short so
        # Gazebo startup is gated by core services, not late joint-state echo.
        feedback_deadline = min(deadline, time.monotonic() + 1.0)
        while time.monotonic() < feedback_deadline:
            if self._get_joint_position(self.gripper_joint) is not None:
                break
            time.sleep(0.05)

        self._services_ready = True
        self._last_failure_message = ""
        self._log().info("All services ready.")
        return True

    # ------------------------------------------------------------------ #
    # Public primitives (recovery-visible low-level API)
    # ------------------------------------------------------------------ #
    def move_cartesian(
        self,
        x: float,
        y: float,
        z: float,
        speed: float | None = None,
        qx: float | None = None,
        qy: float | None = None,
        qz: float | None = None,
        qw: float | None = None,
    ) -> dict[str, Any]:
        """
        ---
        description: Move end-effector to an absolute Cartesian position and optional orientation.
        params:
          x: {type: number, description: "Controlled end-effector X in metres in the configured planning frame", x-frame-source: controller_config.move_group.frame_id, x-binding-role: end_effector_coordinate}
          y: {type: number, description: "Controlled end-effector Y in metres in the configured planning frame", x-frame-source: controller_config.move_group.frame_id, x-binding-role: end_effector_coordinate}
          z: {type: number, description: "Controlled end-effector Z in metres in the configured planning frame", x-frame-source: controller_config.move_group.frame_id, x-binding-role: end_effector_coordinate}
          speed: {type: number, description: "Trajectory time scale (>1 slower, <1 faster). Optional."}
          qx: {type: number, description: "Target quaternion X component. Optional with qy, qz, and qw."}
          qy: {type: number, description: "Target quaternion Y component. Optional with qx, qz, and qw."}
          qz: {type: number, description: "Target quaternion Z component. Optional with qx, qy, and qw."}
          qw: {type: number, description: "Target quaternion W component. Optional with qx, qy, and qz."}
        preconditions: {}
        effects:
          current_pose:
            pose_absolute_from_params: [x, y, z]
          current_pose_ref:
            set_unknown: true
        ---
        """
        normalized_quaternion, orientation_error = _normalized_optional_quaternion(
            qx,
            qy,
            qz,
            qw,
        )
        if orientation_error:
            return {"success": False, "message": f"move_cartesian {orientation_error}"}
        try:
            target_x = float(x)
            target_y = float(y)
            target_z = float(z)
        except (TypeError, ValueError, OverflowError):
            return {
                "success": False,
                "message": "move_cartesian position must contain finite numeric values",
            }
        if not all(math.isfinite(value) for value in (target_x, target_y, target_z)):
            return {
                "success": False,
                "message": "move_cartesian position must contain finite numeric values",
            }
        if not self.wait_for_services():
            return {"success": False, "message": self._unavailable_message("services not ready")}
        ee = self._get_ee_pose()
        if ee is None:
            return {
                "success": False,
                "message": self._unavailable_message("cannot read current ee pose"),
            }
        if normalized_quaternion is None:
            normalized_quaternion, orientation_error = _normalized_optional_quaternion(
                getattr(ee.orientation, "x", None),
                getattr(ee.orientation, "y", None),
                getattr(ee.orientation, "z", None),
                getattr(ee.orientation, "w", None),
            )
            if orientation_error or normalized_quaternion is None:
                return {
                    "success": False,
                    "message": (
                        "move_cartesian current end-effector orientation is invalid: "
                        f"{orientation_error or 'orientation is unavailable'}"
                    ),
                }
        orientation = self._make_orientation(*normalized_quaternion)
        same_xy = math.isclose(float(ee.position.x), target_x, abs_tol=1e-6) and math.isclose(
            float(ee.position.y), target_y, abs_tol=1e-6
        )
        if same_xy:
            result = self._move_pose_direct(
                target_x,
                target_y,
                target_z,
                orientation=orientation,
                label="move_cartesian",
                speed=speed,
            )
        else:
            result = self._move_xy_at_z(
                target_x,
                target_y,
                target_z,
                orientation=orientation,
                label="move_cartesian",
                speed=speed,
            )
        if result.get("success"):
            result["absolute_position"] = {
                "x": target_x,
                "y": target_y,
                "z": target_z,
                **dict(
                    zip(
                        ("qx", "qy", "qz", "qw"),
                        normalized_quaternion,
                        strict=True,
                    )
                ),
            }
        return result

    def move_insert(  # noqa: C901, PLR0913 - mirrors the fixed ROS action contract.
        self,
        part_name: str,
        calibration_id: str,
        profile_sha256: str,
        hard_caps_sha256: str,
        expected_start_pose: dict[str, Any],
        target_pose: dict[str, Any],
        insertion_axis_world: dict[str, Any],
        contact_speed_m_s: float,
        contact_force_delta_n: float,
        engagement_progress_m: float,
        insertion_force_n: float,
        spiral_radius_m: float,
        spiral_pitch_m: float,
        spiral_speed_m_s: float,
        spiral_acceleration_m_s2: float,
        max_axial_force_n: float,
        max_lateral_force_n: float,
        max_torque_nm: float,
        force_depth_profile: dict[str, Any],
        baseline_force_uncertainty_n: float,
        baseline_torque_uncertainty_nm: float,
        tilt_tolerance_rad: float,
        seated_depth_tolerance_m: float,
        settle_time_sec: float,
        timeout_sec: float,
        trial_id: str = "",
    ) -> dict[str, Any]:
        """Execute the internal insertion move in simulation without contact search."""
        requested_part = part_name if isinstance(part_name, str) else ""
        requested_trial_id = trial_id if isinstance(trial_id, str) else ""
        if not isinstance(trial_id, str) or requested_trial_id != requested_trial_id.strip():
            return {
                "success": False,
                "message": "move_insert trial_id is invalid",
                "trial_id": requested_trial_id,
            }
        if not requested_part or requested_part != requested_part.strip():
            return {
                "success": False,
                "message": "move_insert requires an exact non-empty part identifier",
            }
        try:
            start = _pose_from_mapping(expected_start_pose)
            target = _pose_from_mapping(target_pose)
        except (KeyError, TypeError, ValueError) as exc:
            return {"success": False, "message": f"move_insert pose is invalid: {exc}"}

        if self.execution_mode == "simulation":
            if not self.wait_for_services():
                return {
                    "success": False,
                    "message": self._unavailable_message("services not ready"),
                }
            ee = self._get_ee_pose()
            if ee is None:
                return {
                    "success": False,
                    "message": self._unavailable_message("cannot read current ee pose"),
                }
            try:
                current = _pose_from_mapping(
                    {
                        "x": ee.position.x,
                        "y": ee.position.y,
                        "z": ee.position.z,
                        "qx": ee.orientation.x,
                        "qy": ee.orientation.y,
                        "qz": ee.orientation.z,
                        "qw": ee.orientation.w,
                    }
                )
            except (AttributeError, KeyError, TypeError, ValueError) as exc:
                return {
                    "success": False,
                    "message": f"move_insert current tool pose is invalid: {exc}",
                }
            translation_error_m, rotation_error_deg = _pose_delta(current, start)
            start_position_tolerance_m = max(
                0.003,
                _as_float(
                    getattr(
                        self,
                        "simulation_insert_start_position_tolerance_m",
                        None,
                    ),
                    _as_float(
                        getattr(self, "cartesian_position_tolerance_m", None),
                        0.005,
                    )
                    + 0.001,
                ),
            )
            if (
                translation_error_m > start_position_tolerance_m
                or rotation_error_deg > 3.0
            ):
                return {
                    "success": False,
                    "message": (
                        "move_insert expected start pose mismatch: "
                        f"translation={translation_error_m:.4f} m, "
                        f"rotation={rotation_error_deg:.2f} deg, "
                        f"simulation translation tolerance="
                        f"{start_position_tolerance_m:.4f} m"
                    ),
                }
            orientation = self._make_orientation(
                target["qx"], target["qy"], target["qz"], target["qw"]
            )
            result = self._move_pose_direct(
                target["x"],
                target["y"],
                target["z"],
                orientation=orientation,
                label="move_insert:simulation_direct",
            )
            if result.get("success"):
                result.update(
                    {
                        "absolute_position": target,
                        "trial_id": requested_trial_id,
                        "move_insert_mode": "simulation_direct",
                        "hard_caps_sha256": str(hard_caps_sha256 or ""),
                        "message": (
                            "simulation_direct moved to insert_pose without force or "
                            "contact search"
                        ),
                    }
                )
            return result

        if requested_part not in _MOVE_INSERT_SUPPORTED_PARTS:
            return {
                "success": False,
                "message": f"move_insert does not support exact part identifier {requested_part!r}",
            }

        if not isinstance(calibration_id, str) or not calibration_id:
            return {"success": False, "message": "move_insert calibration_id is missing"}
        supplied_hash = profile_sha256 if isinstance(profile_sha256, str) else ""
        supplied_hard_caps_hash = (
            hard_caps_sha256 if isinstance(hard_caps_sha256, str) else ""
        )
        try:
            valid_hash = len(supplied_hash) == 64 and int(supplied_hash, 16) >= 0
            valid_hard_caps_hash = (
                len(supplied_hard_caps_hash) == 64
                and int(supplied_hard_caps_hash, 16) >= 0
            )
        except ValueError:
            valid_hash = False
            valid_hard_caps_hash = False
        if not valid_hash or not valid_hard_caps_hash:
            return {
                "success": False,
                "message": (
                    "move_insert profile_sha256 or hard_caps_sha256 is invalid"
                ),
            }
        if not isinstance(force_depth_profile, dict):
            return {
                "success": False,
                "message": "move_insert force_depth_profile must be an object",
            }

        profile = {
            "contact_speed_m_s": contact_speed_m_s,
            "contact_force_delta_n": contact_force_delta_n,
            "engagement_progress_m": engagement_progress_m,
            "insertion_force_n": insertion_force_n,
            "spiral_radius_m": spiral_radius_m,
            "spiral_pitch_m": spiral_pitch_m,
            "spiral_speed_m_s": spiral_speed_m_s,
            "spiral_acceleration_m_s2": spiral_acceleration_m_s2,
            "max_axial_force_n": max_axial_force_n,
            "max_lateral_force_n": max_lateral_force_n,
            "max_torque_nm": max_torque_nm,
            "tilt_tolerance_rad": tilt_tolerance_rad,
            "seated_depth_tolerance_m": seated_depth_tolerance_m,
            "settle_time_sec": settle_time_sec,
        }
        derived_timeout, timeout_error = derive_move_insert_timeout_sec(
            start,
            target,
            insertion_axis_world,
            profile,
            part_name=requested_part,
            insert_max_timeout_sec=timeout_sec,
        )
        if timeout_error:
            return {"success": False, "message": timeout_error}
        try:
            supplied_timeout = float(timeout_sec)
        except (TypeError, ValueError, OverflowError):
            supplied_timeout = math.nan
        if not math.isfinite(supplied_timeout) or not math.isclose(
            supplied_timeout,
            derived_timeout,
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            return {
                "success": False,
                "message": (
                    "move_insert timeout_sec does not match the timeout derived from "
                    "the frozen poses and profile"
                ),
            }
        if not self.wait_for_services():
            return {"success": False, "message": self._unavailable_message("services not ready")}
        ee = self._get_ee_pose()
        if ee is None:
            return {
                "success": False,
                "message": self._unavailable_message("cannot read current ee pose"),
            }
        try:
            current = _pose_from_mapping(
                {
                    "x": ee.position.x,
                    "y": ee.position.y,
                    "z": ee.position.z,
                    "qx": ee.orientation.x,
                    "qy": ee.orientation.y,
                    "qz": ee.orientation.z,
                    "qw": ee.orientation.w,
                }
            )
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            return {
                "success": False,
                "message": f"move_insert current tool0 pose is invalid: {exc}",
            }
        translation_error_m, rotation_error_deg = _pose_delta(current, start)
        if translation_error_m > 0.003 or rotation_error_deg > 3.0:
            return {
                "success": False,
                "message": (
                    "move_insert expected start pose mismatch: "
                    f"translation={translation_error_m:.4f} m, "
                    f"rotation={rotation_error_deg:.2f} deg"
                ),
            }
        orientation = self._make_orientation(
            target["qx"], target["qy"], target["qz"], target["qw"]
        )
        result = self._move_pose_direct(
            target["x"],
            target["y"],
            target["z"],
            orientation=orientation,
            label="move_insert",
        )
        if result.get("success"):
            result.update(
                {
                    "absolute_position": target,
                    "profile_sha256": supplied_hash,
                    "hard_caps_sha256": supplied_hard_caps_hash,
                    "contact_detected": False,
                    "final_insertion_depth_m": math.sqrt(
                        sum(
                            (target[field] - start[field]) ** 2
                            for field in ("x", "y", "z")
                        )
                    ),
                }
            )
        result["trial_id"] = requested_trial_id
        return result

    def move_relative(
        self,
        dx: float,
        dy: float,
        dz: float,
        speed: float | None = None,
    ) -> dict[str, Any]:
        """
        ---
        description: Move end-effector relative to its current position.
        params:
          dx: {type: number, description: "Delta X in meters"}
          dy: {type: number, description: "Delta Y in meters"}
          dz: {type: number, description: "Delta Z in meters"}
          speed: {type: number, description: "Trajectory time scale (>1 slower, <1 faster). Optional."}
        preconditions:
          current_pose:
            exists: true
        effects:
          current_pose:
            pose_relative_from_params: [dx, dy, dz]
          current_pose_ref:
            set_unknown: true
        ---
        """
        if not self.wait_for_services():
            return {"success": False, "message": self._unavailable_message("services not ready")}
        ee = self._get_ee_pose()
        if ee is None:
            return {
                "success": False,
                "message": self._unavailable_message("cannot read current ee pose"),
            }
        target_x = ee.position.x + float(dx)
        target_y = ee.position.y + float(dy)
        target_z = ee.position.z + float(dz)
        time_scale = _as_float(speed, self.trajectory_time_scale)
        ok = self._cartesian_move(
            self._make_pose(target_x, target_y, target_z, ee.orientation),
            f"move_relative(dx={dx}, dy={dy}, dz={dz})",
            time_scale=time_scale,
        )
        direct_failure = str(getattr(self, "_last_failure_message", "") or "").strip().lower()
        if (
            not ok
            and "timed out" not in direct_failure
            and math.isclose(float(dx), 0.0, abs_tol=1e-9)
            and math.isclose(float(dy), 0.0, abs_tol=1e-9)
            and not math.isclose(float(dz), 0.0, abs_tol=1e-9)
        ):
            target = self._make_pose(target_x, target_y, target_z, ee.orientation)
            if self.execution_mode == "simulation":
                ok = self._execute_simulation_motion_plan(
                    label=f"move_relative(dx={dx}, dy={dy}, dz={dz})",
                    target_pose=target,
                    time_scale=time_scale,
                )
            else:
                self._log().warn(
                    "move_relative vertical fallback: retrying no-collision move "
                    f"for dz={float(dz):.4f}"
                )
                ok = self._cartesian_move(
                    target,
                    f"move_relative(dx={dx}, dy={dy}, dz={dz}) (no-collision)",
                    avoid_collisions=False,
                    min_fraction=0.70,
                    allow_partial=True,
                    time_scale=time_scale,
                )
        if not ok:
            return {"success": False, "message": f"failed relative move ({dx}, {dy}, {dz})"}
        return {"success": True, "message": f"moved relative ({dx}, {dy}, {dz})"}

    def move_pose(
        self,
        x: float,
        y: float,
        z: float,
        qx: float,
        qy: float,
        qz: float,
        qw: float,
        speed: float | None = None,
    ) -> dict[str, Any]:
        """
        ---
        description: Move end-effector to an absolute pose with explicit quaternion orientation.
        params:
          x: {type: number, description: "Controlled end-effector X in metres in the configured planning frame"}
          y: {type: number, description: "Controlled end-effector Y in metres in the configured planning frame"}
          z: {type: number, description: "Controlled end-effector Z in metres in the configured planning frame"}
          qx: {type: number, description: "Target quaternion X component"}
          qy: {type: number, description: "Target quaternion Y component"}
          qz: {type: number, description: "Target quaternion Z component"}
          qw: {type: number, description: "Target quaternion W component"}
          speed: {type: number, description: "Trajectory time scale (>1 slower, <1 faster). Optional."}
        preconditions: {}
        effects:
          current_pose:
            pose_absolute_from_params: [x, y, z]
          current_pose_ref:
            set_unknown: true
        ---
        """
        if not self.wait_for_services():
            return {"success": False, "message": self._unavailable_message("services not ready")}
        target_orientation = self._make_orientation(qx, qy, qz, qw)
        ee = self._get_ee_pose()
        if ee is None:
            return {
                "success": False,
                "message": self._unavailable_message("cannot read current ee pose"),
            }
        same_xy = math.isclose(float(ee.position.x), float(x), abs_tol=1e-6) and math.isclose(
            float(ee.position.y), float(y), abs_tol=1e-6
        )
        if same_xy:
            return self._move_pose_direct(
                float(x),
                float(y),
                float(z),
                orientation=target_orientation,
                label="move_pose",
                speed=speed,
            )
        return self._move_xy_at_z(
            float(x),
            float(y),
            float(z),
            orientation=target_orientation,
            label="move_pose",
            speed=speed,
        )

    def move_to_named_pose(self, pose_name: str, speed: float | None = None) -> dict[str, Any]:
        """
        ---
        description: Move to a named joint configuration (e.g. 'home').
        params:
          pose_name: {type: string, description: "Name of the joint configuration from robot manifest"}
          speed: {type: number, description: "Trajectory time scale (>1 slower, <1 faster). Optional."}
        preconditions:
          held_part:
            equals: null
        effects:
          current_state:
            set: idle
          current_pose_ref:
            set_from_param: pose_name
          current_pose:
            set_unknown: true
          occupancy.location:
            set_from_param: pose_name
        ---
        """
        if not self.wait_for_services():
            return {"success": False, "message": self._unavailable_message("services not ready")}
        positions = self.named_positions.get(str(pose_name))
        if not isinstance(positions, (list, tuple)) or not positions:
            available = sorted(self.named_positions.keys()) if self.named_positions else []
            return {
                "success": False,
                "message": f"unknown pose '{pose_name}'; available={available}",
            }
        joint_values = [float(v) for v in positions]
        if self._fresh_stable_joint_target(
            dict(zip(self.arm_joint_names, joint_values, strict=True)), tolerance=0.02,
        ):
            evidence = {
                "command_sent": False,
                "reason": "fresh stable endpoint already observed",
                "target": dict(zip(self.arm_joint_names, joint_values, strict=True)),
            }
            self._last_command_evidence = evidence
            return {
                "success": True,
                "message": f"already at named pose '{pose_name}'",
                **evidence,
            }
        if (
            self.execution_mode == "simulation"
            and self.controller_config.get("retain_observed_clear_pose_as_home") is True
            and self._current_simulation_state_is_collision_free()
        ):
            evidence = {
                "command_sent": False,
                "reason": "retained fresh collision-free post-task clearance pose",
                "observed_clear_pose_as_home": True,
            }
            self._last_command_evidence = evidence
            return {
                "success": True,
                "message": "retained observed collision-free clearance pose as home",
                **evidence,
            }
        duration_sec = self._scaled_joint_duration(self.named_pose_duration_sec, speed)
        if self.execution_mode == "simulation" and self.arm_trajectory_topic:
            if self._move_joints_via_moveit(joint_values, duration_sec=duration_sec):
                self._last_command_evidence = {
                    **dict(self._last_command_evidence or {}),
                    "command_sent": True,
                }
                return {
                    "success": True,
                    "message": f"moved to named pose '{pose_name}'",
                    "command_sent": True,
                }
            return {
                "success": False,
                "message": self._with_last_failure(
                    f"failed to move to named pose '{pose_name}'"
                ),
            }
        # Try trajectory publisher first, then MoveIt fallback.
        if self._arm_pub and self._publish_arm_joint_trajectory_and_wait(
            joint_values,
            duration_sec=duration_sec,
        ):
            self._last_command_evidence = {"command_sent": True}
            return {"success": True, "message": f"moved to named pose '{pose_name}'",
                    "command_sent": True}
        if self._exec_client and self._move_joints_via_moveit(
            joint_values, duration_sec=duration_sec
        ):
            self._last_command_evidence = {"command_sent": True}
            return {"success": True, "message": f"moved to named pose '{pose_name}' via MoveIt",
                    "command_sent": True}
        return {"success": False, "message": f"failed to move to named pose '{pose_name}'"}

    def get_current_pose(self) -> dict[str, Any]:
        """
        ---
        description: Return the current controlled end-effector pose in the configured planning frame.
        params: {}
        preconditions: {}
        effects:
          current_pose_ref:
            set_unknown: true
        ---
        """
        if not self.wait_for_services():
            return {"success": False, "message": self._unavailable_message("services not ready")}
        ee = self._get_ee_pose()
        if ee is None:
            return {
                "success": False,
                "message": self._unavailable_message("cannot read current ee pose"),
            }
        return {
            "success": True,
            "message": "current pose",
            "pose": {
                "x": ee.position.x,
                "y": ee.position.y,
                "z": ee.position.z,
                "qx": ee.orientation.x,
                "qy": ee.orientation.y,
                "qz": ee.orientation.z,
                "qw": ee.orientation.w,
            },
        }

    def attach_part(
        self,
        model_name: str,
        link: str | None = None,
        part_name: str = "",
    ) -> dict[str, Any]:
        """
        ---
        description: Attach a part model to the robot gripper (Gazebo link attacher).
        params:
          model_name: {type: string, description: "Gazebo model name of the part to attach"}
          link: {type: string, description: "Optional specific attach link. Uses default candidates if omitted."}
          part_name: {type: string, description: "Optional canonical part identifier for semantic state projection."}
        preconditions:
          held_part:
            equals: null
          gripper_state:
            equals: closed
        effects:
          held_part:
            set_from_param_any_of: ["part_name", "model_name"]
        ---
        """
        if not self.wait_for_services():
            return {"success": False, "message": self._unavailable_message("services not ready")}
        ok = self._attach_part(str(model_name))
        return {
            "success": ok,
            "message": f"attached {model_name}" if ok else self._with_last_failure(f"failed to attach {model_name}"),
            "payload_collision": (self._last_command_evidence or {}).get("payload_collision"),
        }

    def detach_part(
        self,
        model_name: str = "",
        link: str | None = None,
        assume_released_if_open: bool = False,
    ) -> dict[str, Any]:
        """
        ---
        description: Detach a part model from the robot gripper (Gazebo link detacher).
        params:
          model_name: {type: string, description: "Gazebo model name to detach. Uses currently attached model if empty."}
          link: {type: string, description: "Optional specific detach link. Tries all candidates if omitted."}
        preconditions:
          held_part:
            not_equals: null
          gripper_state:
            equals: open
        effects:
          held_part:
            set: null
        ---
        """
        if not self.wait_for_services():
            return {"success": False, "message": self._unavailable_message("services not ready")}
        target_model = str(model_name or "")
        attempts = max(1, 1 + self.release_detach_retry_count)
        ok = False
        for attempt_idx in range(attempts):
            ok = self._detach_part(
                target_model,
                timeout_sec=self.release_detach_timeout_sec,
                log_failure=(attempt_idx == attempts - 1),
            )
            if ok:
                break

            if attempt_idx + 1 < attempts:
                self._log().warn(
                    f"Detach retry {attempt_idx + 1}/{attempts - 1} for "
                    f"{target_model or 'held part'}"
                )
                time.sleep(self.release_detach_retry_delay_sec)
        if not ok:
            if (
                self._release_open_fallback_allowed(assume_released_if_open)
                and self._gripper_is_open_enough()
            ):
                if not target_model and self._attached_model:
                    target_model = str(self._attached_model)
                verified_release = self._verify_detach_timeout_release(target_model)
                if verified_release is False:
                    return {
                        "success": False,
                        "message": (
                            f"detach timed out and release verification kept "
                            f"{target_model or 'held part'} near the gripper"
                        ),
                        "release_mode": "verification_failed_after_detach_timeout",
                    }
                self._attached_model = None
                self._attached_link = None
                if verified_release is True:
                    self._log().warn(
                        f"Confirmed {target_model or 'held part'} was released after detach timeout because the model is separated from the gripper"
                    )
                    return {
                        "success": True,
                        "message": (
                            f"verified detached {target_model or 'held part'} after gripper opened"
                        ),
                        "release_mode": "verified_open_after_detach_timeout",
                    }
                self._log().warn(
                    f"Assuming {target_model or 'held part'} was released because the gripper is already open and detach verification is unavailable"
                )
                return {
                    "success": True,
                    "message": (
                        f"assumed detached {target_model or 'held part'} after gripper "
                        "opened (verification unavailable)"
                    ),
                    "release_mode": "assumed_open_after_detach_timeout",
                }
            return {"success": False, "message": f"failed to detach {model_name or 'held part'}"}
        return {"success": True, "message": f"detached {model_name or 'held part'}"}

    def grasp_part(
        self,
        model_name: str,
        part_name: str = "",
        position: float | None = None,
    ) -> dict[str, Any]:
        """
        ---
        description: Close the gripper and attach the target part as one high-level grasp primitive.
        params:
          model_name: {type: string, description: "Controller identifier of the part to attach; a semantic part label does not establish this binding", x-binding-role: controller_identifier}
          part_name: {type: string, description: "Optional canonical part identifier for held-part tracking."}
          position: {type: number, description: "Optional gripper closing position override."}
        preconditions:
          held_part:
            equals: null
        effects:
          current_state:
            set: picked
          gripper_state:
            set: closed
          held_part:
            set_from_param_any_of: ["part_name", "model_name"]
        ---
        """
        if not self.wait_for_services():
            return {
                "success": False,
                "message": self._unavailable_message("services not ready"),
            }

        target_model = str(model_name or "").strip()
        target_part = str(part_name or "").strip()

        if not self.close_gripper(position=position):
            return {
                "success": False,
                "message": (
                    self._last_failure_message
                    or f"failed to close gripper to grasp {target_part or target_model or 'part'}"
                ),
            }

        attached = self.attach_part(target_model, part_name=target_part)
        if attached.get("success"):
            return {
                "success": True,
                "message": (
                    f"grasped {target_part or target_model or 'part'}"
                    if (target_part or target_model)
                    else "grasped part"
                ),
            }

        attachment_evidence = deepcopy(self._last_command_evidence or {})
        rollback_ok = self.open_gripper()
        attachment_evidence["rollback"] = {
            "primitive": "open_gripper", "success": bool(rollback_ok),
            "command_evidence": deepcopy(self._last_command_evidence or {}),
        }
        self._last_command_evidence = attachment_evidence
        rollback_message = (
            "reopened gripper after failed attach"
            if rollback_ok
            else (self._last_failure_message or "failed to reopen gripper after failed attach")
        )
        return {
            "success": False,
            "message": (
                f"{str(attached.get('message') or 'failed to attach part')}; "
                f"rollback: {rollback_message}"
            ),
            "payload_collision": attached.get("payload_collision"),
            "rollback": attachment_evidence["rollback"],
        }

    def release_part(
        self,
        model_name: str = "",
        part_name: str = "",
        assume_released_if_open: bool = False,
    ) -> dict[str, Any]:
        """
        ---
        description: Open the gripper and detach the currently held part as one high-level release primitive.
        params:
          model_name: {type: string, description: "Optional controller identifier of the part to detach.", x-binding-role: controller_identifier}
          part_name: {type: string, description: "Optional canonical recovery part name for release trace validation."}
          assume_released_if_open: {type: boolean, description: "Treat an already-open gripper as an idempotent release when true."}
        preconditions:
          held_part:
            not_equals: null
        effects:
          current_state:
            set: idle
          gripper_state:
            set: open
          held_part:
            set: null
        ---
        """
        if not self.wait_for_services():
            return {
                "success": False,
                "message": self._unavailable_message("services not ready"),
            }

        target_model = str(model_name or "").strip()
        target_part = str(part_name or "").strip()
        if (
            not assume_released_if_open
            and str(getattr(self, "execution_mode", "") or "").strip().lower() == "simulation"
        ):
            assume_released_if_open = True

        self._wait_process_time(self.release_preopen_settle_sec)
        if not self.open_gripper():
            return {
                "success": False,
                "message": (
                    self._last_failure_message
                    or f"failed to open gripper to release {target_part or target_model or 'part'}"
                ),
            }
        self._wait_process_time(self.release_postopen_settle_sec)

        used_simulation_release_fallback = False
        if self._simulation_release_fallback_enabled(assume_released_if_open):
            used_simulation_release_fallback = True
            detached = self._release_part_simulation_best_effort_detach(
                target_model,
                target_part,
            )
        else:
            detached = self.detach_part(
                target_model,
                assume_released_if_open=assume_released_if_open,
            )
        if detached.get("success"):
            self._wait_process_time(self.release_postdetach_settle_sec)
            release_mode = str(detached.get("release_mode") or "").strip()
            if release_mode == "verification_unavailable_after_detach_timeout":
                release_message = str(detached.get("message") or "")
            elif target_part or target_model:
                release_message = f"released {target_part or target_model or 'part'}"
            else:
                release_message = "released part"
            result = {
                "success": True,
                "message": release_message,
            }
            if release_mode:
                result["release_mode"] = release_mode
            return result

        if (
            self._release_open_fallback_allowed(assume_released_if_open)
            and not used_simulation_release_fallback
        ):
            fallback_model = target_model
            if not fallback_model and self._attached_model:
                fallback_model = str(self._attached_model)
            verified_release = self._verify_detach_timeout_release(fallback_model)
            if verified_release is not False:
                self._attached_model = None
                self._attached_link = None
                self._wait_process_time(self.release_postdetach_settle_sec)
                result = {
                    "success": True,
                    "message": (
                        f"released {target_part or fallback_model or 'part'}"
                        if (target_part or fallback_model)
                        else "released part"
                    ),
                    "release_mode": (
                        "verified_open_after_detach_timeout"
                        if verified_release is True
                        else "assumed_open_after_detach_timeout"
                    ),
                }
                if verified_release is True:
                    self._log().warn(
                        f"Confirmed {fallback_model or 'held part'} was released after detach failure because the model is separated from the gripper"
                    )
                else:
                    self._log().warn(
                        f"Assuming {fallback_model or 'held part'} was released because the gripper open command succeeded and detach verification is unavailable"
                    )
                return result
            detached = {
                "success": False,
                "message": (
                    f"detach failed and release verification kept "
                    f"{fallback_model or 'held part'} near the gripper"
                ),
            }

        if used_simulation_release_fallback:
            return {
                "success": False,
                "message": str(detached.get("message") or "failed to detach part"),
                "release_mode": detached.get(
                    "release_mode",
                    "verification_unavailable_after_detach_timeout",
                ),
            }

        rollback_ok = self.close_gripper()
        rollback_message = (
            "reclosed gripper after failed detach"
            if rollback_ok
            else (self._last_failure_message or "failed to reclose gripper after failed detach")
        )
        return {
            "success": False,
            "message": (
                f"{str(detached.get('message') or 'failed to detach part')}; "
                f"rollback: {rollback_message}"
            ),
        }

    def _release_open_fallback_allowed(self, assume_released_if_open: bool) -> bool:
        if not assume_released_if_open:
            return False
        mode = str(getattr(self, "execution_mode", "") or "").strip().lower()
        return mode != "physical"

    def _simulation_release_fallback_enabled(self, assume_released_if_open: bool) -> bool:
        if not self._release_open_fallback_allowed(assume_released_if_open):
            return False
        mode = str(getattr(self, "execution_mode", "") or "").strip().lower()
        return mode == "simulation"

    def _simulation_release_detach_timeout_sec(self) -> float:
        configured = _as_float(
            getattr(self, "release_best_effort_detach_timeout_sec", None),
            0.75,
        )
        release_timeout = _as_float(
            getattr(self, "release_detach_timeout_sec", None),
            configured,
        )
        if configured <= 0.0:
            configured = 0.75
        if release_timeout > 0.0:
            configured = min(configured, release_timeout)
        return max(0.05, configured)

    def _release_part_simulation_best_effort_detach(
        self,
        target_model: str,
        target_part: str,
    ) -> dict[str, Any]:
        fallback_model = str(target_model or "").strip()
        if not fallback_model and self._attached_model:
            fallback_model = str(self._attached_model)
        display_name = target_part or fallback_model or "held part"

        ok = self._detach_part(
            fallback_model,
            timeout_sec=self._simulation_release_detach_timeout_sec(),
            attached_link_only=False,
            log_failure=False,
            timeout_log_level="warn",
            break_on_timeout=False,
            prefer_attached_link=False,
            extra_link_candidates=getattr(self, "release_detach_link_candidates", []),
        )
        if ok:
            return {
                "success": True,
                "message": f"detached {display_name}",
            }

        verified_release = self._verify_detach_timeout_release(
            fallback_model,
            timeout_log_level="debug",
        )
        if verified_release is False:
            return {
                "success": False,
                "message": (
                    f"detach failed and release verification kept "
                    f"{fallback_model or 'held part'} near the gripper"
                ),
                "release_mode": "verification_failed_after_detach_timeout",
            }

        if verified_release is True:
            self._attached_model = None
            self._attached_link = None
            self._log().warn(
                f"Confirmed {fallback_model or 'held part'} was released after detach timeout because the model is separated from the gripper"
            )
            return {
                "success": True,
                "message": f"verified detached {display_name} after gripper opened",
                "release_mode": "verified_open_after_detach_timeout",
            }

        return {
            "success": True,
            "message": (
                f"released {display_name} after gripper opened (detach verification unavailable)"
            ),
            "release_mode": "verification_unavailable_after_detach_timeout",
        }

    # ------------------------------------------------------------------ #
    # Legacy low-level API (kept for backward compat)
    # ------------------------------------------------------------------ #
    def move_joints(self, positions: list[float], duration_sec: float = 2.0) -> bool:
        if not self.wait_for_services():
            return False
        if not self._arm_pub:
            self._log().error("Arm trajectory publisher not configured")
            return False
        if len(positions) != len(self.arm_joint_names):
            self._log().error(
                f"move_joints expected {len(self.arm_joint_names)} joints, got {len(positions)}"
            )
            return False

        traj = self._JointTrajectory()
        traj.joint_names = self._get_arm_joint_command_names()
        point = self._JointTrajectoryPoint()
        point.positions = [float(v) for v in positions]
        duration = max(0.1, float(duration_sec))
        sec = int(duration)
        nsec = int((duration - sec) * 1_000_000_000)
        point.time_from_start = self._Duration(sec=sec, nanosec=nsec)
        traj.points = [point]
        self._time_joint_target(traj)
        if self.execution_mode == 'simulation':
            return self._send_simulation_joint_trajectory(self.arm_trajectory_topic, traj)
        self._arm_pub.publish(traj)
        return True

    def _publish_arm_joint_trajectory_and_wait(
        self,
        positions: list[float],
        *,
        duration_sec: float,
        tolerance_rad: float = 0.08,
    ) -> bool:
        if not self.move_joints(positions, duration_sec=duration_sec):
            return False
        timeout_sec = max(2.0, float(duration_sec) + 2.0)
        if self._wait_for_arm_joint_targets(
            positions,
            timeout_sec=timeout_sec,
            tolerance_rad=tolerance_rad,
            log_miss=False,
        ):
            return True
        self._log().warn(
            f"Arm joint trajectory command did not converge within {timeout_sec:.2f}s; falling back"
        )
        return False

    def open_gripper(self) -> bool:
        """
        ---
        description: Open the robot gripper.
        params: {}
        preconditions: {}
        effects:
          gripper_state:
            set: open
        ---
        """
        if not self.wait_for_services():
            return False
        return self._gripper_command(self.gripper_open, "OPEN")

    def close_gripper(self, position: float | None = None) -> bool:
        """
        ---
        description: Close the robot gripper.
        params:
          position: {type: number, description: "Optional custom gripper position override. If omitted, uses the default close position."}
        preconditions: {}
        effects:
          gripper_state:
            set: closed
        ---
        """
        if not self.wait_for_services():
            return False
        target = float(position) if position is not None else self.gripper_close
        return self._gripper_command(target, "CLOSE")

    def delay(self, duration_sec: float) -> dict[str, Any]:
        """
        ---
        description: Wait intentionally between robot task steps.
        params:
          duration_sec: {type: number, description: "Intentional wait duration in seconds."}
        preconditions: {}
        effects: {}
        synthesis_hidden: true
        ---
        """
        try:
            duration = float(duration_sec)
        except (TypeError, ValueError):
            return {
                "success": False,
                "message": f"delay duration_sec must be numeric: {duration_sec!r}",
            }
        if not math.isfinite(duration) or duration < 0.0:
            return {
                "success": False,
                "message": "delay duration_sec must be finite and non-negative",
            }
        wait_sec = self._scaled_wall_wait_sec(duration)
        self._wait_process_time(wait_sec)
        return {
            "success": True,
            "message": f"delay {duration:.3f}s",
            "duration_sec": duration,
            "wait_sec": wait_sec,
        }

    def _derive_gripper_close_position(
        self,
        *,
        model_name: str = "",
        product_geometry: dict[str, Any] | None = None,
    ) -> float | None:
        if not (
            float(self.gripper_open) > float(self.gripper_close)
            and 0.0 <= float(self.gripper_close) <= float(self.gripper_open) <= 0.25
        ):
            return None

        geometry = product_geometry if isinstance(product_geometry, dict) else {}
        width = None
        for key in (
            "grasp_width_m",
            "part_width_m",
            "part_diameter_m",
            "diameter_m",
            "width_m",
        ):
            if key in geometry:
                candidate = _as_float(geometry.get(key), 0.0)
                if candidate > 0.0:
                    width = candidate
                    break
        if width is None:
            width = _model_footprint_width_from_gazebo_world(str(model_name or ""))
        if width is None or width <= 0.0:
            return None

        target = width
        target = max(float(self.gripper_close), min(float(self.gripper_open), target))
        return target

    # ------------------------------------------------------------------ #
    # Geometry helpers (agent calls these to compute targets, then moves)
    # ------------------------------------------------------------------ #
    def compute_pick_targets(
        self,
        part_name: str = "",
        product_geometry: dict[str, Any] | None = None,
        target_pose: dict[str, Any] | None = None,
        target_pose_source: str = "",
        prefer_live_detection: bool = False,
        approach_height_override_m: float | None = None,
        ignore_current_height_for_travel_z: bool = False,
        min_pick_tcp_z_override_m: float | None = None,
        use_global_min_pick_tcp_z: bool = True,
        surface_clearance_override_m: float | None = None,
        apply_pick_z_adjustments: bool = True,
        detected_parts: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """
        ---
        description: Compute vertical pick target positions from world object geometry and configured end-effector/TCP feedback without moving. The frames must agree.
        params:
          part_name: {type: string, description: "Name of the detected part to pick"}
          product_geometry:
            type: object
            description: "Optional pick geometry override, not a CADMeshRecord. Missing dimensions or surface height use controller defaults."
            x-grounding-required: true
            x-grounding-fields: [board_center, part_height_m]
            properties:
              board_center:
                type: object
                description: "Support reference in world metres; z supplies the support height."
                x-grounding-fields: [z]
                properties:
                  x: {type: number, x-frame-source: world}
                  y: {type: number, x-frame-source: world}
                  z: {type: number, x-frame-source: world}
              part_height_m: {type: number, exclusiveMinimum: 0, x-binding-role: vertical_part_height, description: "Part height along the supported vertical pick direction in metres."}
              model_name: {type: string, x-binding-role: controller_identifier}
          target_pose:
            type: object
            description: "Optional observed object location in world metres; this is not the computed end-effector pick position. Skips perception when provided."
            x-grounding-fields: [x, y, z]
            properties:
              x: {type: number, x-frame-source: world}
              y: {type: number, x-frame-source: world}
              z: {type: number, x-frame-source: world}
              model_name: {type: string, x-binding-role: controller_identifier}
          detected_parts:
            type: array
            description: "Optional detected object records to consume without another detection."
            minItems: 1
            items:
              type: object
              x-grounding-fields: [part_name, x, y, z]
              properties:
                part_name: {type: string}
                model_name: {type: string, x-binding-role: controller_identifier}
                x: {type: number, x-frame-source: world}
                y: {type: number, x-frame-source: world}
                z: {type: number, x-frame-source: world}
          target_pose_source: {type: string, description: "Optional source label for target_pose, e.g. observed_pose"}
          prefer_live_detection: {type: boolean, description: "When true, try perception first and use target_pose only as fallback"}
          approach_height_override_m: {type: number, description: "Optional vertical approach distance"}
          ignore_current_height_for_travel_z: {type: boolean}
          min_pick_tcp_z_override_m: {type: number}
          use_global_min_pick_tcp_z: {type: boolean, description: "When false, do not clamp pick TCP Z to controller.motion.min_pick_tcp_z_m"}
          surface_clearance_override_m: {type: number, description: "Optional target-surface clearance added to the raw pick TCP Z"}
          apply_pick_z_adjustments: {type: boolean, description: "When false, skip per-part pick Z adjustments"}
        preconditions: {}
        effects: {}
        ---
        Compute pick target positions from perception + geometry without moving.

        Returns a dict with keys: part_name, model_name, tx, ty, tz, pick_z,
        travel_z, part_height, tcp_offset_z, pick_tcp_z, start_x, start_y,
        start_z, or {"success": False, "message": ...} on failure.
        """
        supplied_detected_parts = detected_parts is not None
        if not supplied_detected_parts and not self.wait_for_services():
            return {"success": False, "message": self._unavailable_message("services not ready")}

        target: dict[str, Any] | None = None
        parts: list[dict[str, Any]] | None = None
        detection_attempted = supplied_detected_parts
        normalized_target_pose = None
        if isinstance(target_pose, dict) and {"x", "y", "z"} <= set(target_pose.keys()):
            try:
                normalized_target_pose = {
                    "x": float(target_pose["x"]),
                    "y": float(target_pose["y"]),
                    "z": float(target_pose["z"]),
                }
            except (TypeError, ValueError):
                normalized_target_pose = None

        if detected_parts is not None:
            parts = [dict(row) for row in detected_parts if isinstance(row, dict)]
            if part_name:
                matching_parts = [row for row in parts if row.get("part_name") == part_name]
                if self.execution_mode == "physical" and len(matching_parts) != 1:
                    detected_names = sorted(
                        {
                            str(row.get("part_name"))
                            for row in parts
                            if str(row.get("part_name") or "").strip()
                        }
                    )
                    return {
                        "success": False,
                        "message": (
                            f"requested physical part {part_name!r} did not match exactly one "
                            f"detection; detected={detected_names}"
                        ),
                    }
                target = matching_parts[0] if matching_parts else None
            elif parts:
                target = parts[0]
        elif bool(prefer_live_detection) and str(part_name or "").strip():
            detection_attempted = True
            parts = self.detect_parts()
            if parts:
                target = next((p for p in parts if p.get("part_name") == part_name), None)
            if self.execution_mode == "physical":
                matching_parts = [
                    row for row in (parts or []) if row.get("part_name") == part_name
                ]
                if len(matching_parts) != 1:
                    detected_names = sorted(
                        {
                            str(row.get("part_name"))
                            for row in (parts or [])
                            if str(row.get("part_name") or "").strip()
                        }
                    )
                    return {
                        "success": False,
                        "message": (
                            f"requested physical part {part_name!r} did not match exactly one "
                            f"detection; detected={detected_names}"
                        ),
                    }

        if target is not None:
            if self.execution_mode == "physical":
                detection_error = _physical_detection_error(
                    target,
                    requested_part=str(part_name or "").strip(),
                )
                if detection_error:
                    return {"success": False, "message": detection_error}
            tx = _as_float(target.get("x"), 0.0)
            ty = _as_float(target.get("y"), 0.0)
            tz = _as_float(target.get("z"), 0.0)
            target_part_name = str(target.get("part_name") or part_name or "")
            target_pose_source_used = "live_detection"
        elif normalized_target_pose is None:
            if parts is None and not detection_attempted:
                parts = self.detect_parts()
            if not parts:
                return {"success": False, "message": "no parts detected"}

            if part_name:
                target = next((p for p in parts if p.get("part_name") == part_name), None)
                if target is None:
                    detected_names = sorted(
                        {
                            str(p.get("part_name"))
                            for p in parts
                            if str(p.get("part_name") or "").strip()
                        }
                    )
                    return {
                        "success": False,
                        "message": f"requested part '{part_name}' not detected; detected={detected_names}",
                    }
            if target is None:
                target = parts[0]
            if self.execution_mode == "physical":
                detection_error = _physical_detection_error(
                    target,
                    requested_part=str(part_name or "").strip(),
                )
                if detection_error:
                    return {"success": False, "message": detection_error}
            tx = _as_float(target.get("x"), 0.0)
            ty = _as_float(target.get("y"), 0.0)
            tz = _as_float(target.get("z"), 0.0)
            target_part_name = str(target.get("part_name") or part_name or "")
            target_pose_source_used = "live_detection"
        else:
            tx = float(normalized_target_pose["x"])
            ty = float(normalized_target_pose["y"])
            tz = float(normalized_target_pose["z"])
            target_part_name = str(part_name or target_pose.get("part_name") or "")
            target_pose_source_used = str(target_pose_source or "")

        geo = product_geometry or {}
        board_center = geo.get("board_center", {}) if isinstance(geo, dict) else {}
        board_center_z = _as_float(board_center.get("z"), 1.02)

        target_height = _as_float(geo.get("part_height_m"), 0.08)
        pose_model_name = target_pose.get("model_name") if isinstance(target_pose, dict) else ""
        target_model = str(
            geo.get("model_name") or (target or {}).get("model_name") or pose_model_name or ""
        )

        physical_stl_pick: dict[str, Any] = {}
        source_stl = str(geo.get("source_stl") or "").strip()
        stl_readiness = getattr(self, "_physical_stl_pick_readiness", None)
        if (
            self.execution_mode == "physical"
            and target_part_name == "MG"
            and callable(stl_readiness)
            and not source_stl
        ):
            return {
                "success": False,
                "message": "physical MG grasp requires the actual source_stl geometry",
            }
        if self.execution_mode == "physical" and source_stl:
            if target is None:
                return {
                    "success": False,
                    "message": (
                        "physical MG actual STL grasp requires a fresh validated /detect_all "
                        "result with table_surface_z_m"
                    ),
                }
            if not callable(stl_readiness):
                return {
                    "success": False,
                    "message": "physical controller does not support the actual STL MG grasp",
                }
            physical_stl_pick = dict(stl_readiness(geo) or {})
            if not bool(physical_stl_pick.get("success")):
                return {
                    "success": False,
                    "message": str(
                        physical_stl_pick.get("message")
                        or "physical MG actual STL grasp is not ready"
                    ),
                }

        if supplied_detected_parts and not self.init():
            return {
                "success": False,
                "message": self._unavailable_message("read-only controller initialization failed"),
            }
        ee = self._get_ee_pose()
        if ee is None:
            return {
                "success": False,
                "message": self._unavailable_message("cannot read current ee pose"),
            }
        self._last_start_pose = self._make_pose(
            ee.position.x,
            ee.position.y,
            ee.position.z,
            ee.orientation,
        )

        handling_access = dict(geo.get("handling_robot_access") or {})
        configured_orientation = handling_access.get("grasp_orientation_xyzw")
        grasp_quaternion = None
        if isinstance(configured_orientation, (list, tuple)) and len(configured_orientation) == 4:
            grasp_quaternion, grasp_orientation_error = _normalized_optional_quaternion(
                *configured_orientation
            )
            if grasp_orientation_error or grasp_quaternion is None:
                return {
                    "success": False,
                    "message": (
                        "configured handling_robot_access grasp orientation is invalid: "
                        f"{grasp_orientation_error or 'orientation is unavailable'}"
                    ),
                }
        configured_tcp_offset_z = handling_access.get("tcp_offset_z_m")
        if configured_tcp_offset_z is None:
            ee_tcp_offset_z = self._get_ee_tcp_world_z_offset()
        else:
            try:
                ee_tcp_offset_z = float(configured_tcp_offset_z)
            except (TypeError, ValueError, OverflowError):
                return {
                    "success": False,
                    "message": "configured handling_robot_access tcp_offset_z_m is invalid",
                }
            if not math.isfinite(ee_tcp_offset_z):
                return {
                    "success": False,
                    "message": "configured handling_robot_access tcp_offset_z_m is invalid",
                }
        pick_bias = vertical_pick_bias(
            target_height, self.pick_tcp_z_bias_min_m, self.pick_tcp_z_bias_max_m
        )
        surface_clearance_m = max(0.0, _as_float(surface_clearance_override_m, 0.0))
        if physical_stl_pick:
            table_surface_z_m = float((target or {})["table_surface_z_m"])
            pick_tcp_z_raw = table_surface_z_m + float(
                physical_stl_pick["pick_tcp_z_offset_from_table_m"]
            )
        else:
            table_surface_z_m = None
            pick_tcp_z_raw = tz + pick_bias + surface_clearance_m
        if min_pick_tcp_z_override_m is not None:
            effective_min_tcp_z = float(min_pick_tcp_z_override_m)
            pick_tcp_z = max(pick_tcp_z_raw, effective_min_tcp_z)
        elif bool(use_global_min_pick_tcp_z) and not physical_stl_pick:
            effective_min_tcp_z = self.min_pick_tcp_z_m
            pick_tcp_z = max(pick_tcp_z_raw, effective_min_tcp_z)
        else:
            effective_min_tcp_z = None
            pick_tcp_z = pick_tcp_z_raw
        pick_z_adjustment_m = (
            self.pick_z_adjustments_m.get(target_part_name.upper(), 0.0)
            if bool(apply_pick_z_adjustments)
            else 0.0
        )
        if physical_stl_pick:
            pick_tcp_z += pick_z_adjustment_m
            tcp_offset_from_table_m = pick_tcp_z - float(table_surface_z_m)
            pad_lower_m = tcp_offset_from_table_m + float(
                physical_stl_pick["lowest_closing_endpoint_z_from_tcp_m"]
            )
            closed_pad_lower_m = tcp_offset_from_table_m + float(
                physical_stl_pick["closed_inner_pad_lower_z_from_tcp_m"]
            )
            closed_pad_upper_m = tcp_offset_from_table_m + float(
                physical_stl_pick["closed_inner_pad_upper_z_from_tcp_m"]
            )
            tooth_height_m = float(physical_stl_pick["tooth_height_m"])
            part_height_m = float(physical_stl_pick["part_height_m"])
            finger_tooth_clearance_m = pad_lower_m - tooth_height_m
            finger_hub_overlap_m = max(
                0.0,
                min(closed_pad_upper_m, part_height_m)
                - max(closed_pad_lower_m, tooth_height_m),
            )
            required_tooth_clearance_m = float(physical_stl_pick["tooth_clearance_m"])
            required_hub_overlap_m = float(physical_stl_pick["minimum_hub_overlap_m"])
            if finger_tooth_clearance_m + 1e-9 < required_tooth_clearance_m:
                return {
                    "success": False,
                    "message": (
                        "physical MG actual STL target would contact the teeth: "
                        f"clearance={finger_tooth_clearance_m * 1000.0:.2f} mm, "
                        f"required={required_tooth_clearance_m * 1000.0:.2f} mm"
                    ),
                }
            if finger_hub_overlap_m + 1e-9 < required_hub_overlap_m:
                return {
                    "success": False,
                    "message": (
                        "physical MG actual STL target has insufficient smooth hub overlap: "
                        f"overlap={finger_hub_overlap_m * 1000.0:.2f} mm, "
                        f"required={required_hub_overlap_m * 1000.0:.2f} mm"
                    ),
                }
            gripper_close_position = float(physical_stl_pick["gripper_close_position"])
            pick_z = pick_tcp_z - ee_tcp_offset_z
        else:
            finger_tooth_clearance_m = None
            finger_hub_overlap_m = None
            gripper_close_position = self._derive_gripper_close_position(
                model_name=target_model,
                product_geometry=geo,
            )
            pick_z = pick_tcp_z - ee_tcp_offset_z
            pick_z += pick_z_adjustment_m
        pick_tool0_z_adjustment_m = (
            self.pick_tool0_z_adjustment_m if self.execution_mode == "physical" else 0.0
        )
        pick_z += pick_tool0_z_adjustment_m
        if self.execution_mode == "physical" and gripper_close_position is None:
            return {
                "success": False,
                "message": str(
                    self._last_failure_message
                    or "physical grasp geometry did not produce an RG2 close position"
                ),
            }

        access_approach_height = handling_access.get("approach_height_m")
        approach_height = _as_float(
            approach_height_override_m,
            _as_float(access_approach_height, self.approach_height_m),
        )
        travel_z = pick_travel_height(
            tz, board_center_z, pick_z, approach_height,
            None if bool(ignore_current_height_for_travel_z) else ee.position.z,
        )
        explicit_access_poses = None
        raw_access_approach = handling_access.get("approach_pose")
        raw_access_target = handling_access.get("target_pose")
        if raw_access_approach is not None or raw_access_target is not None:
            if handling_access.get("pose_format") != "xyz_xyzw":
                return {
                    "success": False,
                    "message": "configured handling_robot_access pose_format must be xyz_xyzw",
                }
            if not all(
                isinstance(value, (list, tuple)) and len(value) == 7
                for value in (raw_access_approach, raw_access_target)
            ):
                return {
                    "success": False,
                    "message": "configured handling_robot_access requires approach_pose and target_pose",
                }

            access_poses = []
            for label, raw_pose in (
                ("approach_pose", raw_access_approach),
                ("target_pose", raw_access_target),
            ):
                try:
                    xyz = [float(value) for value in raw_pose[:3]]
                except (TypeError, ValueError, OverflowError):
                    return {
                        "success": False,
                        "message": f"configured handling_robot_access {label} is invalid",
                    }
                orientation, orientation_error = _normalized_optional_quaternion(
                    *raw_pose[3:]
                )
                if (
                    not all(math.isfinite(value) for value in xyz)
                    or orientation_error
                    or orientation is None
                ):
                    return {
                        "success": False,
                        "message": f"configured handling_robot_access {label} is invalid",
                    }
                if grasp_quaternion is not None and abs(
                    sum(a * b for a, b in zip(orientation, grasp_quaternion, strict=True))
                ) < 0.999:
                    return {
                        "success": False,
                        "message": (
                            f"configured handling_robot_access {label} orientation "
                            "does not match grasp_orientation_xyzw"
                        ),
                    }
                access_poses.append(
                    {
                        "x": xyz[0],
                        "y": xyz[1],
                        "z": xyz[2] - ee_tcp_offset_z,
                        **dict(
                            zip(
                                ("qx", "qy", "qz", "qw"),
                                orientation,
                                strict=True,
                            )
                        ),
                    }
                )
            if math.dist(
                (access_poses[1]["x"], access_poses[1]["y"]), (tx, ty)
            ) > 0.08:
                return {
                    "success": False,
                    "message": "configured handling_robot_access target disagrees with the observed part",
                }
            explicit_access_poses = tuple(access_poses)
            pick_tcp_z = access_poses[1]["z"] + ee_tcp_offset_z
            pick_z = access_poses[1]["z"]
            travel_z = access_poses[0]["z"]

        self._log().info(
            "[ComputePickTargets] "
            f"part={target_part_name} "
            f"current=({ee.position.x:.3f}, {ee.position.y:.3f}, {ee.position.z:.3f}) "
            f"target=({tx:.3f}, {ty:.3f}, {tz:.3f}) "
            f"source={target_pose_source_used or 'perception'} "
            f"source_stl={source_stl or '<none>'} "
            f"surface_clearance={surface_clearance_m:.3f} "
            f"pick_tcp_z={pick_tcp_z:.3f} "
            f"pick_tool0_z_adjustment_m={pick_tool0_z_adjustment_m:.3f} "
            f"travel_z={travel_z:.3f} pick_z={pick_z:.3f} tcp_offset_z={ee_tcp_offset_z:.3f}"
        )

        raw_part_pose = target if isinstance(target, dict) else target_pose
        raw_part_pose = raw_part_pose if isinstance(raw_part_pose, dict) else {}
        if all(field in raw_part_pose for field in ("qx", "qy", "qz", "qw")):
            part_quaternion, part_quaternion_error = _normalized_optional_quaternion(
                raw_part_pose.get("qx"),
                raw_part_pose.get("qy"),
                raw_part_pose.get("qz"),
                raw_part_pose.get("qw"),
            )
            if part_quaternion_error or part_quaternion is None:
                return {
                    "success": False,
                    "message": (
                        "detected part orientation is invalid: "
                        f"{part_quaternion_error or 'orientation is unavailable'}"
                    ),
                }
            part_orientation_source = target_pose_source_used or "live_detection"
        else:
            part_quaternion = (0.0, 0.0, 0.0, 1.0)
            part_orientation_source = "upright_axial_part_assumption"
        origin_pose = {
            "x": tx,
            "y": ty,
            "z": tz,
            **dict(
                zip(
                    ("qx", "qy", "qz", "qw"),
                    part_quaternion,
                    strict=True,
                )
            ),
        }

        result = {
            "success": True,
            "part_name": target_part_name,
            "model_name": target_model,
            "tx": tx,
            "ty": ty,
            "tz": tz,
            "pick_z": pick_z,
            "travel_z": travel_z,
            "part_height": target_height,
            "tcp_offset_z": ee_tcp_offset_z,
            "pick_tcp_z": pick_tcp_z,
            "pick_tcp_z_raw": pick_tcp_z_raw,
            "surface_clearance_m": surface_clearance_m,
            "pick_z_adjustment_m": pick_z_adjustment_m,
            "pick_tool0_z_adjustment_m": pick_tool0_z_adjustment_m,
            "gripper_close_position": gripper_close_position,
            "apply_pick_z_adjustments": bool(apply_pick_z_adjustments),
            "effective_min_pick_tcp_z": effective_min_tcp_z,
            "use_global_min_pick_tcp_z": bool(use_global_min_pick_tcp_z)
            and not bool(physical_stl_pick),
            "target_pose_source": target_pose_source_used,
            "origin_pose": origin_pose,
            "origin_pose_provenance": {
                "frame_id": "world",
                "part_name": target_part_name,
                "model_name": target_model,
                "source": target_pose_source_used or "perception",
                "orientation_source": part_orientation_source,
            },
            "prefer_live_detection": bool(prefer_live_detection),
            "start_x": ee.position.x,
            "start_y": ee.position.y,
            "start_z": ee.position.z,
        }
        if explicit_access_poses is not None:
            result["approach_pose"] = explicit_access_poses[0]
            result["target_pose"] = explicit_access_poses[1]
            result["access_retreat_pose"] = explicit_access_poses[0]
            result["handling_robot_access_executed"] = True
        elif grasp_quaternion is not None:
            orientation = dict(
                zip(("qx", "qy", "qz", "qw"), grasp_quaternion, strict=True)
            )
            result["approach_pose"] = {
                "x": tx,
                "y": ty,
                "z": travel_z,
                **orientation,
            }
            result["target_pose"] = {
                "x": tx,
                "y": ty,
                "z": pick_z,
                **orientation,
            }
        if physical_stl_pick:
            result.update(
                {
                    "source_stl": source_stl,
                    "source_stl_sha256": str(
                        physical_stl_pick.get("source_stl_sha256") or ""
                    ),
                    "hub_up": True,
                    "hub_diameter_m": float(physical_stl_pick["hub_diameter_m"]),
                    "hub_height_m": float(physical_stl_pick["hub_height_m"]),
                    "tooth_diameter_m": float(physical_stl_pick["tooth_diameter_m"]),
                    "tooth_height_m": float(physical_stl_pick["tooth_height_m"]),
                    "grasp_width_m": float(physical_stl_pick["grasp_width_m"]),
                    "tooth_clearance_m": float(physical_stl_pick["tooth_clearance_m"]),
                    "minimum_hub_overlap_m": float(
                        physical_stl_pick["minimum_hub_overlap_m"]
                    ),
                    "finger_tooth_clearance_m": finger_tooth_clearance_m,
                    "finger_hub_overlap_m": finger_hub_overlap_m,
                    "open_gripper_position": float(
                        physical_stl_pick["open_gripper_position"]
                    ),
                    "mg_gripper_close_position": float(
                        physical_stl_pick["mg_gripper_close_position"]
                    ),
                    "open_inner_pad_lower_z_from_tcp_m": float(
                        physical_stl_pick["open_inner_pad_lower_z_from_tcp_m"]
                    ),
                    "open_inner_pad_upper_z_from_tcp_m": float(
                        physical_stl_pick["open_inner_pad_upper_z_from_tcp_m"]
                    ),
                    "closed_inner_pad_lower_z_from_tcp_m": float(
                        physical_stl_pick["closed_inner_pad_lower_z_from_tcp_m"]
                    ),
                    "closed_inner_pad_upper_z_from_tcp_m": float(
                        physical_stl_pick["closed_inner_pad_upper_z_from_tcp_m"]
                    ),
                    "predicted_closing_z_displacement_m": float(
                        physical_stl_pick["predicted_closing_z_displacement_m"]
                    ),
                    "pick_tcp_z_offset_from_table_m": pick_tcp_z
                    - float(table_surface_z_m),
                }
            )
        if target is not None:
            for provenance_field in (
                "confidence",
                "captured_at",
                "frame_id",
                "table_surface_z_m",
            ):
                if provenance_field in target:
                    result[provenance_field] = target[provenance_field]
        return result

    def _assembly_board_v1_aruco_paths(self) -> tuple[Path, Path]:
        snapshot_path = Path(
            getattr(
                self,
                "_assembly_board_v1_aruco_snapshot_path",
                _PERCEPTION_PREVIEW_ROOT
                / str(self.robot_name or "").strip().lower()
                / "assembly_board-v1_aruco.json",
            )
        ).expanduser()
        config_path = Path(
            getattr(
                self,
                "_assembly_board_v1_aruco_config_path",
                _PERCEPTION_CAMERA_CONFIG,
            )
        ).expanduser()
        return snapshot_path, config_path

    def _assembly_board_v1_aruco_acceptance(self) -> tuple[dict[str, Any], str]:
        _snapshot_path, config_path = self._assembly_board_v1_aruco_paths()
        try:
            payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        except FileNotFoundError:
            return {}, f"ArUco configuration is unavailable: {config_path}"
        except (OSError, yaml.YAMLError) as exc:
            return {}, f"could not read ArUco configuration {config_path}: {exc}"
        config = dict(payload.get(_ASSEMBLY_BOARD_V1 + "_aruco") or {})
        role = str(self.robot_name or "").strip().lower()
        role_config = dict(dict(config.get("roles") or {}).get(role) or {})
        marker_length_m = _as_float(
            config.get("marker_length_m"),
            _ASSEMBLY_BOARD_ARUCO_MARKER_LENGTH_M,
        )
        try:
            accepted_pose = _pose_from_mapping(role_config.get("accepted_pose"))
            accepted_generation = int(role_config.get("accepted_generation", 0) or 0)
            accepted_at = float(role_config.get("accepted_at", 0.0) or 0.0)
        except (TypeError, ValueError, KeyError):
            return {}, (
                f"{role} has no accepted {_ASSEMBLY_BOARD_V1} ArUco pose; "
                "use Perception -> Locate & Accept Board"
            )
        calibration_id = str(role_config.get("calibration_id") or "").strip()
        if accepted_generation < 1 or accepted_at <= 0.0 or not calibration_id:
            return {}, (
                f"{role} has no accepted {_ASSEMBLY_BOARD_V1} ArUco pose; "
                "use Perception -> Locate & Accept Board"
            )
        if bool(role_config.get("movement_blocked", False)):
            return {}, (
                f"{_ASSEMBLY_BOARD_V1} moved more than 10 mm or 2 deg from the "
                f"accepted {role} pose; use Perception -> Locate & Accept Board again"
            )
        if not math.isclose(
            marker_length_m,
            _ASSEMBLY_BOARD_ARUCO_MARKER_LENGTH_M,
            abs_tol=1e-9,
        ):
            return {}, (
                f"{_ASSEMBLY_BOARD_V1} marker length is {marker_length_m * 1000.0:.3f} mm; "
                "76.000 mm is required"
            )
        return {
            "marker_length_m": marker_length_m,
            "accepted_pose": accepted_pose,
            "accepted_generation": accepted_generation,
            "accepted_at": accepted_at,
            "calibration_id": calibration_id,
        }, ""

    def _validated_assembly_board_v1_aruco_snapshot(  # noqa: C901, PLR0912
        self,
        *,
        snapshot: dict[str, Any],
        acceptance: dict[str, Any],
        requested_at: float,
        now: float,
    ) -> tuple[dict[str, Any], str]:
        role = str(self.robot_name or "").strip().lower()
        if str(snapshot.get("camera_role") or "").strip().lower() != role:
            return {}, f"ArUco snapshot camera_role does not match {role}"
        if str(snapshot.get("resource_location") or "").strip() != _ASSEMBLY_BOARD_V1:
            return {}, f"ArUco snapshot does not describe {_ASSEMBLY_BOARD_V1}"
        dictionary = str(
            snapshot.get("marker_dictionary") or snapshot.get("dictionary") or ""
        ).strip()
        if dictionary != _ASSEMBLY_BOARD_ARUCO_DICTIONARY:
            return {}, f"ArUco snapshot dictionary must be {_ASSEMBLY_BOARD_ARUCO_DICTIONARY}"
        try:
            marker_id = int(snapshot.get("marker_id"))
            marker_length_m = float(snapshot.get("marker_length_m"))
            sample_count = int(snapshot.get("sample_count", 0) or 0)
            frame_captured_at = float(snapshot.get("frame_captured_at", 0.0) or 0.0)
            sample_started_at = float(
                snapshot.get("sample_started_at")
                or snapshot.get("window_started_at")
                or 0.0
            )
            reprojection_error_px = float(
                snapshot.get("reprojection_error_px", math.inf)
            )
            translation_spread_m = float(
                snapshot.get("translation_spread_m", math.inf)
            )
            rotation_spread_deg = float(
                snapshot.get("rotation_spread_deg", math.inf)
            )
        except (TypeError, ValueError) as exc:
            return {}, f"ArUco snapshot has invalid numeric provenance: {exc}"
        if marker_id != _ASSEMBLY_BOARD_ARUCO_ID:
            return {}, f"ArUco marker ID must be {_ASSEMBLY_BOARD_ARUCO_ID}"
        if not math.isclose(
            marker_length_m,
            float(acceptance["marker_length_m"]),
            abs_tol=1e-9,
        ):
            return {}, "ArUco snapshot marker length differs from the accepted configuration"
        if not all(
            math.isfinite(value)
            for value in (
                marker_length_m,
                frame_captured_at,
                sample_started_at,
                reprojection_error_px,
                translation_spread_m,
                rotation_spread_deg,
            )
        ):
            return {}, "ArUco snapshot has non-finite numeric provenance"
        if not bool(snapshot.get("valid")):
            return {}, str(
                snapshot.get("last_error")
                or "ArUco ID 70 snapshot is not valid"
            )
        if not bool(snapshot.get("visible")):
            return {}, str(snapshot.get("last_error") or "ArUco ID 70 is not visible")
        if bool(snapshot.get("pose_ambiguous")):
            return {}, "ArUco ID 70 pose is geometrically ambiguous"
        if not bool(snapshot.get("stable")) or not bool(snapshot.get("world_pose_ready")):
            return {}, str(
                snapshot.get("last_error")
                or "ArUco ID 70 does not yet have a stable world pose"
            )
        if sample_count < _ASSEMBLY_BOARD_ARUCO_MINIMUM_SAMPLES:
            return {}, (
                f"ArUco stability window has {sample_count}/"
                f"{_ASSEMBLY_BOARD_ARUCO_MINIMUM_SAMPLES} samples"
            )
        if sample_started_at + 1e-6 < requested_at:
            return {}, "waiting for ten ArUco observations captured after staging motion"
        age_sec = now - frame_captured_at
        if age_sec < -_PHYSICAL_DETECTION_FUTURE_TOLERANCE_SEC:
            return {}, "ArUco snapshot timestamp is in the future"
        if age_sec > _ASSEMBLY_BOARD_ARUCO_MAX_AGE_SEC:
            return {}, f"ArUco snapshot is stale (age={age_sec:.2f}s)"
        if reprojection_error_px > 1.0:
            return {}, (
                f"ArUco reprojection error {reprojection_error_px:.3f} px exceeds 1.000 px"
            )
        if translation_spread_m > _ASSEMBLY_BOARD_ARUCO_MAX_TRANSLATION_SPREAD_M:
            return {}, (
                f"ArUco translation spread {translation_spread_m * 1000.0:.3f} mm "
                "exceeds 2.000 mm"
            )
        if rotation_spread_deg > _ASSEMBLY_BOARD_ARUCO_MAX_ROTATION_SPREAD_DEG:
            return {}, (
                f"ArUco rotation spread {rotation_spread_deg:.3f} deg exceeds 0.500 deg"
            )
        calibration = dict(snapshot.get("calibration") or {})
        calibration_id = str(
            snapshot.get("calibration_id")
            or calibration.get("identity")
            or calibration.get("id")
            or ""
        ).strip()
        if not calibration_id:
            return {}, "ArUco snapshot has no hand-eye calibration identity"
        if calibration_id != str(acceptance["calibration_id"]):
            return {}, (
                "ArUco hand-eye calibration changed after board acceptance; "
                "use Perception -> Locate & Accept Board again"
            )
        raw_pose = snapshot.get("pose") or snapshot.get("world_pose")
        try:
            pose = _pose_from_mapping(raw_pose)
        except (KeyError, TypeError, ValueError) as exc:
            return {}, f"ArUco snapshot world pose is invalid: {exc}"
        frame_id = str(
            snapshot.get("frame_id")
            or dict(raw_pose or {}).get("frame_id")
            or ""
        ).strip()
        if frame_id != "world":
            return {}, "ArUco snapshot pose must be expressed in world"
        translation_delta_m, rotation_delta_deg = _pose_delta(
            dict(acceptance["accepted_pose"]), pose
        )
        if (
            translation_delta_m > _ASSEMBLY_BOARD_ARUCO_MAX_BASELINE_TRANSLATION_M
            or rotation_delta_deg > _ASSEMBLY_BOARD_ARUCO_MAX_BASELINE_ROTATION_DEG
        ):
            return {}, (
                f"{_ASSEMBLY_BOARD_V1} moved {translation_delta_m * 1000.0:.1f} mm / "
                f"{rotation_delta_deg:.2f} deg from the accepted {role} baseline; "
                "use Perception -> Locate & Accept Board"
            )
        return {
            "success": True,
            "destination_location": _ASSEMBLY_BOARD_V1,
            "camera_role": role,
            "frame_id": "world",
            "pose": pose,
            "captured_at": frame_captured_at,
            "sample_started_at": sample_started_at,
            "sample_count": sample_count,
            "generation": int(acceptance["accepted_generation"]),
            "accepted_at": float(acceptance["accepted_at"]),
            "calibration_id": calibration_id,
            "marker_dictionary": dictionary,
            "marker_id": marker_id,
            "marker_length_m": marker_length_m,
            "reprojection_error_px": reprojection_error_px,
            "translation_spread_m": translation_spread_m,
            "rotation_spread_deg": rotation_spread_deg,
            "translation_delta_m": translation_delta_m,
            "rotation_delta_deg": rotation_delta_deg,
            "source": "assembly_board-v1_aruco",
        }, ""

    def _request_assembly_board_v1_post_staging_acceptance(
        self,
        callback: Any,
        requested_at: float,
    ) -> tuple[dict[str, Any], str, str]:
        """Request manager-owned acceptance, then reread its persisted authority."""
        callback_error = ""
        try:
            callback(requested_at)
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            callback_error = str(exc)
        acceptance, acceptance_error = self._assembly_board_v1_aruco_acceptance()
        return acceptance, acceptance_error, callback_error

    def localize_assembly_board_v1(
        self,
        destination_location: str,
        part_name: str = "",
    ) -> dict[str, Any]:
        """Return one post-staging, stable per-arm ArUco pose without moving.

        The accepted pose is frozen by the task runtime and used for the complete
        ``place_approach``/``place_insert`` pair.
        """
        destination = str(destination_location or "").strip()
        if self.execution_mode != "physical":
            return {
                "success": True,
                "destination_location": destination,
                "camera_role": str(self.robot_name or "").strip().lower(),
                "source": "simulation_geometry",
            }
        if destination != _ASSEMBLY_BOARD_V1:
            return {
                "success": False,
                "message": (
                    f"localize_assembly_board_v1 requires destination_location "
                    f"{_ASSEMBLY_BOARD_V1!r}, received {destination or '<empty>'!r}"
                ),
            }
        requested_at = time.time()
        post_staging_accept_callback = getattr(
            self,
            "_assembly_board_v1_post_staging_accept_callback",
            None,
        )
        acceptance, acceptance_error = self._assembly_board_v1_aruco_acceptance()
        if acceptance_error and not callable(post_staging_accept_callback):
            return {"success": False, "message": acceptance_error}
        snapshot_path, _config_path = self._assembly_board_v1_aruco_paths()
        timeout_sec = max(
            0.0,
            _as_float(
                getattr(self, "_assembly_board_v1_aruco_wait_sec", None),
                _ASSEMBLY_BOARD_ARUCO_WAIT_SEC,
            ),
        )
        deadline = time.monotonic() + timeout_sec
        last_error = f"waiting for {snapshot_path}"
        while True:
            try:
                snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
                if not isinstance(snapshot, dict):
                    last_error = "ArUco snapshot is not a JSON object"
                else:
                    callback_error = ""
                    if acceptance_error and callable(post_staging_accept_callback):
                        acceptance, acceptance_error, callback_error = (
                            self._request_assembly_board_v1_post_staging_acceptance(
                                post_staging_accept_callback,
                                requested_at,
                            )
                        )
                        if not callback_error and not acceptance_error:
                            continue
                    if acceptance_error:
                        last_error = callback_error or acceptance_error
                        validated = {}
                    else:
                        validated, last_error = (
                            self._validated_assembly_board_v1_aruco_snapshot(
                                snapshot=snapshot,
                                acceptance=acceptance,
                                requested_at=requested_at,
                                now=time.time(),
                            )
                        )
                    if (
                        last_error.startswith(f"{_ASSEMBLY_BOARD_V1} moved ")
                        and callable(post_staging_accept_callback)
                    ):
                        acceptance, acceptance_error, callback_error = (
                            self._request_assembly_board_v1_post_staging_acceptance(
                                post_staging_accept_callback,
                                requested_at,
                            )
                        )
                        if callback_error or acceptance_error:
                            last_error = callback_error or acceptance_error
                        else:
                            continue
                    if not last_error:
                        validated["part_name"] = str(part_name or "").strip()
                        return validated
            except FileNotFoundError:
                last_error = f"ArUco snapshot is unavailable: {snapshot_path}"
            except (OSError, json.JSONDecodeError) as exc:
                last_error = f"could not read ArUco snapshot {snapshot_path}: {exc}"
            if time.monotonic() >= deadline:
                return {
                    "success": False,
                    "message": (
                        f"{self.robot_name} could not localize {_ASSEMBLY_BOARD_V1} "
                        f"after its observation pose: {last_error}"
                    ),
                }
            time.sleep(0.05)

    def compute_place_targets(
        self,
        pick_ctx: dict[str, Any] | None = None,
        product_geometry: dict[str, Any] | None = None,
        part_name: str = "",
        z_adjustment_m: float = 0.0,
        destination_location: str = "",
        assembly_board_v1_aruco: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        ---
        description: Compute vertical placement targets from world geometry, pick context and configured end-effector/TCP feedback without moving. The frames must agree.
        params:
          part_name: {type: string, description: "Name of the held part to place"}
          pick_ctx:
            type: object
            description: "Optional pick result or measured held-part context. Missing grasp/tool offsets invoke controller fallbacks."
            x-grounding-required: true
            x-grounding-fields: [tz, pick_tcp_z, tcp_offset_z, part_height]
            properties:
              part_name: {type: string}
              model_name: {type: string, x-binding-role: controller_identifier}
              tz: {type: number, x-frame-source: world}
              pick_tcp_z: {type: number, x-frame-source: world}
              tcp_offset_z: {type: number, description: "TCP Z minus controlled end-effector Z in world metres."}
              part_height: {type: number, exclusiveMinimum: 0, x-binding-role: vertical_part_height}
          product_geometry:
            type: object
            description: "Optional placement geometry override, not a CADMeshRecord. Explicit fields avoid configured destination lookup and geometric defaults."
            x-grounding-required: true
            x-grounding-fields: [board_center, slot_xy, slot_floor_z_m]
            properties:
              board_center:
                type: object
                description: "Origin of the slot XY offsets in world metres."
                x-grounding-fields: [x, y]
                properties:
                  x: {type: number, x-frame-source: world}
                  y: {type: number, x-frame-source: world}
                  z: {type: number, x-frame-source: world}
              slot_xy: {type: array, minItems: 2, items: {type: number}, description: "XY offsets in metres from board_center."}
              slot_floor_z_m: {type: number, x-frame-source: world, description: "Measured seating/support surface height in world metres."}
              part_height_m: {type: number, exclusiveMinimum: 0, x-binding-role: vertical_part_height, description: "Optional height override; otherwise uses pick_ctx.part_height."}
              place_tool_yaw_offset_rad: {type: number, description: "Optional simulation assembly tool yaw offset, in radians, applied about world Z relative to the grasp orientation."}
              model_name: {type: string, x-binding-role: controller_identifier}
              target_reference:
                type: object
                properties:
                  target_point: {type: string, description: "part_origin or inserted_part_origin uses target_origin_pose.z when supplied."}
                  surface_role: {type: string, description: "Describes the target surface; assembly_slot can yield distinct pre-insertion and insertion poses."}
              target_origin_pose:
                type: object
                description: "Supported final part-origin reference, not an end-effector pose."
                properties:
                  x: {type: number, x-frame-source: world}
                  y: {type: number, x-frame-source: world}
                  z: {type: number, x-frame-source: world}
          z_adjustment_m: {type: number, description: "Extra Z vertical adjustment"}
          destination_location: {type: string, description: "Optional registered controller destination token. A descriptive ontology label does not establish a resolvable token.", x-binding-role: destination_identifier}
          assembly_board_v1_aruco: {type: object, description: "Frozen per-arm assembly_board-v1 ArUco observation for physical placement"}
        preconditions: {}
        effects: {}
        ---
        Compute placement target positions from pick context + geometry without moving.

        Returns a dict with keys: slot_x, slot_y, board_top_z, place_z,
        part_height, or {"success": False, "message": ...} on failure.
        """
        if not self.wait_for_services():
            return {"success": False, "message": self._unavailable_message("services not ready")}

        pick_ctx = dict(pick_ctx or {})
        target_part_name = str(part_name or pick_ctx.get("part_name") or "")
        geo = resolve_place_geometry(
            part_name=target_part_name,
            destination_location=destination_location,
            product_geometry=product_geometry,
            execution_mode=self.execution_mode,
        )
        symbolic_destination = destination_token_from_place_inputs(
            destination_location=destination_location,
            product_geometry=product_geometry,
        )
        frozen_board_pose = dict(assembly_board_v1_aruco or {})
        world_board_pose: dict[str, float] = {}
        board_registration: dict[str, Any] = {}
        move_insert_profile: dict[str, Any] = {}
        place_insert_release_only = bool(
            self.execution_mode == "physical"
            and self.controller_config.get("place_insert_release_only") is True
        )
        place_approach_skip_board_localization = bool(
            self.execution_mode == "physical"
            and str(self.robot_name or "").strip() == "xarm6"
            and self.controller_config.get(
                "place_approach_skip_board_localization"
            )
            is True
        )
        physical_ur5e_board = (
            self.execution_mode == "physical"
            and str(self.robot_name or "").strip() == "ur5e"
            and symbolic_destination == _ASSEMBLY_BOARD_V1
            and bool(pick_ctx)
            and not place_insert_release_only
        )
        if physical_ur5e_board:
            raw_registration = geo.get(
                "assembly_board-v1_aruco_to_assembly_board-v1"
            )
            registration_ready = False
            if isinstance(raw_registration, Mapping):
                registration_calibration_id = str(
                    raw_registration.get("calibration_id") or ""
                )
                try:
                    _pose_from_mapping(raw_registration)
                except (KeyError, TypeError, ValueError):
                    pass
                else:
                    registration_ready = bool(
                        registration_calibration_id
                        and registration_calibration_id
                        == registration_calibration_id.strip()
                    )
            trial_profile = resolve_move_insert_profile(
                self.controller_config.get("parts_tuning"),
                target_part_name,
                require_qualification=False,
            )
            learned_registration_ready = bool(
                trial_profile.get("success")
                and dict(trial_profile.get("demonstration_recipe") or {})
            )
            if (
                not trial_profile.get("success")
                or not (registration_ready or learned_registration_ready)
            ):
                # place_approach remains usable without insertion commissioning.
                # Omitting the profile keeps place_insert and trials fail-closed.
                physical_ur5e_board = False
        physical_xarm6_held_part_assembly_slot = (
            self.execution_mode == "physical"
            and str(self.robot_name or "").strip() == "xarm6"
            and symbolic_destination == _ASSEMBLY_BOARD_V1
            and bool(pick_ctx)
            and not place_insert_release_only
            and str(
                dict(geo.get("target_reference") or {}).get("surface_role") or ""
            )
            == "assembly_slot"
        )
        if physical_xarm6_held_part_assembly_slot:
            return {
                "success": False,
                "message": _PHYSICAL_XARM6_ASSEMBLY_SLOT_INSERT_ERROR,
            }
        if (
            self.execution_mode == "physical"
            and symbolic_destination == _ASSEMBLY_BOARD_V1
            and not place_approach_skip_board_localization
        ):
            if not frozen_board_pose:
                return {
                    "success": False,
                    "message": (
                        "physical assembly_board-v1 placement requires a fresh frozen "
                        "localize_assembly_board_v1 observation"
                    ),
                }
            if (
                str(frozen_board_pose.get("destination_location") or "").strip()
                != _ASSEMBLY_BOARD_V1
                or str(frozen_board_pose.get("camera_role") or "").strip().lower()
                != str(self.robot_name or "").strip().lower()
                or str(frozen_board_pose.get("frame_id") or "").strip() != "world"
            ):
                return {
                    "success": False,
                    "message": "frozen assembly_board-v1 ArUco observation provenance is invalid",
                }
            try:
                _pose_from_mapping(frozen_board_pose.get("pose"))
                generation = int(frozen_board_pose.get("generation", 0) or 0)
                captured_at = float(frozen_board_pose.get("captured_at", 0.0) or 0.0)
            except (KeyError, TypeError, ValueError) as exc:
                return {
                    "success": False,
                    "message": f"frozen assembly_board-v1 ArUco pose is invalid: {exc}",
                }
            if generation < 1 or captured_at <= 0.0:
                return {
                    "success": False,
                    "message": "frozen assembly_board-v1 ArUco generation is invalid",
                }
        if physical_ur5e_board:
            configured_trial_profile = resolve_move_insert_profile(
                self.controller_config.get("parts_tuning"),
                target_part_name,
                require_qualification=False,
            )
            learned_recipe = dict(
                configured_trial_profile.get("demonstration_recipe") or {}
            )
            raw_registration = geo.get(
                "assembly_board-v1_aruco_to_assembly_board-v1"
            )
            if not learned_recipe and not isinstance(raw_registration, Mapping):
                return {
                    "success": False,
                    "message": (
                        "physical assembly_board-v1 placement requires calibrated "
                        "assembly_board-v1_aruco_to_assembly_board-v1 geometry"
                    ),
                }
            try:
                marker_pose = _pose_from_mapping(frozen_board_pose.get("pose"))
                if learned_recipe:
                    board_registration = {
                        "calibration_id": str(
                            learned_recipe.get("board_calibration_id") or ""
                        ),
                        "x": 0.0,
                        "y": 0.0,
                        "z": 0.0,
                        "qx": 0.0,
                        "qy": 0.0,
                        "qz": 0.0,
                        "qw": 1.0,
                    }
                    world_board_pose = marker_pose
                else:
                    board_registration = dict(raw_registration)
                    registration_calibration_id = str(
                        board_registration.get("calibration_id") or ""
                    ).strip()
                    if not registration_calibration_id:
                        return {
                            "success": False,
                            "message": (
                                "assembly_board-v1_aruco_to_assembly_board-v1."
                                "calibration_id is missing"
                            ),
                        }
                    marker_to_board = _pose_from_mapping(board_registration)
                    world_board_pose = _compose_pose(marker_pose, marker_to_board)
            except (KeyError, TypeError, ValueError) as exc:
                return {
                    "success": False,
                    "message": (
                        "assembly_board-v1_aruco_to_assembly_board-v1 is invalid: "
                        f"{exc}"
                    ),
                }

            raw_frozen_profile = geo.get("move_insert_profile")
            if isinstance(raw_frozen_profile, Mapping):
                frozen_profile = _validated_frozen_move_insert_profile(
                    raw_frozen_profile,
                    part_name=target_part_name,
                    profile_sha256=geo.get("move_insert_profile_sha256"),
                )
                if not frozen_profile.get("success"):
                    return frozen_profile
                frozen_qualification = dict(
                    frozen_profile.get("qualification") or {}
                )
                configured_profile = resolve_move_insert_profile(
                    self.controller_config.get("parts_tuning"),
                    target_part_name,
                    require_qualification=bool(frozen_qualification),
                )
                if (
                    not configured_profile.get("success")
                    or dict(configured_profile.get("qualification") or {})
                    != frozen_qualification
                ):
                    configured_profile = resolve_move_insert_profile(
                        self.controller_config.get("parts_tuning"),
                        target_part_name,
                        require_qualification=False,
                    )
                if not configured_profile.get("success"):
                    return configured_profile
                frozen_identity_fields = (
                    "part_name",
                    "calibration_id",
                    "shared_calibration_id",
                    "override_calibration_id",
                    "profile_sha256",
                    *_MOVE_INSERT_PROFILE_FIELDS,
                )
                mismatched_fields = [
                    field_name
                    for field_name in frozen_identity_fields
                    if frozen_profile.get(field_name)
                    != configured_profile.get(field_name)
                ]
                if mismatched_fields:
                    return {
                        "success": False,
                        "message": (
                            "frozen move_insert_profile does not match the protected "
                            "controller profile; changed fields: "
                            f"{mismatched_fields}"
                        ),
                    }
                move_insert_profile = configured_profile
            else:
                move_insert_profile = resolve_move_insert_profile(
                    self.controller_config.get("parts_tuning"),
                    target_part_name,
                )
                if not move_insert_profile.get("success"):
                    trial_profile = resolve_move_insert_profile(
                        self.controller_config.get("parts_tuning"),
                        target_part_name,
                        require_qualification=False,
                    )
                    if trial_profile.get("success"):
                        move_insert_profile = trial_profile
            if not move_insert_profile.get("success"):
                return {
                    "success": False,
                    "message": str(
                        move_insert_profile.get("message")
                        or f"move_insert profile is unavailable for {target_part_name}"
                    ),
                }
            qualification = dict(move_insert_profile.get("qualification") or {})
            if qualification:
                board_identity = dict(geo)
                board_identity.pop("move_insert_profile", None)
                board_identity.pop("move_insert_profile_sha256", None)
                board_identity.pop("move_insert_hard_caps", None)
                board_identity.pop("move_insert_hard_caps_sha256", None)
                board_geometry_sha256, board_hash_error = _canonical_json_sha256(
                    board_identity
                )
                qualification_matches_board = bool(
                    not board_hash_error
                    and qualification.get("board_calibration_id")
                    == str(frozen_board_pose.get("calibration_id") or "")
                    and qualification.get("board_geometry_sha256")
                    == board_geometry_sha256
                )
                if not qualification_matches_board:
                    move_insert_profile = resolve_move_insert_profile(
                        self.controller_config.get("parts_tuning"),
                        target_part_name,
                        require_qualification=False,
                    )
                    if not move_insert_profile.get("success"):
                        return move_insert_profile
        if symbolic_destination and not has_place_geometry_fields(geo):
            return {
                "success": False,
                "message": (
                    f"failed to resolve placement geometry for destination "
                    f"'{symbolic_destination}' and part '{target_part_name or '?'}'"
                ),
            }
        board_center = geo.get("board_center", {}) if isinstance(geo, dict) else {}
        slot_xy = geo.get("slot_xy")
        if isinstance(slot_xy, (list, tuple)) and len(slot_xy) >= 2:
            bx = _as_float(board_center.get("x"), 0.0) + _as_float(slot_xy[0], 0.0)
            by = _as_float(board_center.get("y"), 0.0) + _as_float(slot_xy[1], 0.0)
        else:
            bx = _as_float(board_center.get("x"), pick_ctx.get("tx", 0.0))
            by = _as_float(board_center.get("y"), pick_ctx.get("ty", 0.0))

        board_top_z = _as_float(
            geo.get("slot_floor_z_m"),
            _as_float(board_center.get("z"), 1.025),
        )
        target_height = _as_float(geo.get("part_height_m"), pick_ctx.get("part_height", 0.08))
        target_reference = dict(geo.get("target_reference") or {})
        target_origin_pose = dict(geo.get("target_origin_pose") or {})

        if pick_ctx:
            grasp_tcp_to_part_origin_z = _as_float(pick_ctx.get("pick_tcp_z"), 0.0) - _as_float(
                pick_ctx.get("tz"), 0.0
            )
            tcp_offset_z = _as_float(pick_ctx.get("tcp_offset_z"), float("nan"))
            if not math.isfinite(tcp_offset_z):
                tcp_offset_z = self._get_ee_tcp_world_z_offset()
        else:
            # Recovery insert macros may only know the target geometry, not the earlier pick context.
            grasp_tcp_to_part_origin_z = max(
                self.pick_tcp_z_bias_min_m,
                min(self.pick_tcp_z_bias_max_m, target_height * 0.25),
            )
            tcp_offset_z = self._get_ee_tcp_world_z_offset()

        target_point = str(target_reference.get("target_point") or "").strip()
        reference_z = target_origin_pose.get("z")
        place_part_origin_z_source = "slot_geometry"
        origin_pose = (
            pick_ctx.get("origin_pose") if isinstance(pick_ctx.get("origin_pose"), dict) else {}
        )
        pick_origin_location = str(pick_ctx.get("origin_resource_location") or "").strip()
        requested_destination = str(destination_location or "").strip()
        use_measured_origin_pose = (
            target_point in {"part_origin", "inserted_part_origin"}
            and bool(origin_pose)
            and (not requested_destination or requested_destination == pick_origin_location)
            and {"x", "y", "z"} <= set(origin_pose.keys())
        )
        if use_measured_origin_pose:
            bx = _as_float(origin_pose.get("x"), bx)
            by = _as_float(origin_pose.get("y"), by)
            reference_z = origin_pose.get("z")
            target_origin_pose = {
                "x": bx,
                "y": by,
                "z": _as_float(reference_z, board_top_z + (target_height * 0.5)),
                "source": "pick_ctx.origin_pose",
            }
            place_part_origin_z_source = "pick_ctx.origin_pose"
        if target_point in {"part_origin", "inserted_part_origin"}:
            place_part_origin_z = _as_float(reference_z, board_top_z + (target_height * 0.5))
            if place_part_origin_z_source != "pick_ctx.origin_pose":
                reference_z_value = _as_float(reference_z, math.nan)
                place_part_origin_z_source = (
                    "target_origin_pose"
                    if math.isfinite(reference_z_value)
                    else "support_geometry_fallback"
                )
        else:
            place_part_origin_z = supported_part_origin_height(
                board_top_z, target_height, self.place_surface_gap_m, self.insertion_depth_m
            )
        place_tcp_z = place_part_origin_z + grasp_tcp_to_part_origin_z
        place_z = controlled_link_height(
            place_tcp_z, tcp_offset_z, _as_float(z_adjustment_m, 0.0)
        )

        insertion_axis_world = {"x": 0.0, "y": 0.0, "z": -1.0}
        pose_orientation: dict[str, float] = {}
        raw_handoff = pick_ctx.get("held_part_handoff")
        # A repeated place_approach updates resolved descend; it does not update
        # the grasp-time tool orientation carried by the held-part handoff.
        immutable_grasp_pose = (
            raw_handoff.get("world_tool0_pose_at_grasp")
            if isinstance(raw_handoff, Mapping)
            else None
        )
        raw_held_pose = (
            immutable_grasp_pose
            if isinstance(immutable_grasp_pose, Mapping)
            else dict(pick_ctx.get("resolved_cartesian_positions") or {}).get(
                "descend"
            )
        )
        if isinstance(raw_held_pose, dict):
            try:
                held_pose = _pose_from_mapping(raw_held_pose)
            except (KeyError, TypeError, ValueError):
                held_pose = {}
            if held_pose:
                pose_orientation = {
                    field: held_pose[field] for field in ("qx", "qy", "qz", "qw")
                }

        simulation_assembly_slot = (
            self.execution_mode == "simulation"
            and str(target_reference.get("surface_role") or "") == "assembly_slot"
        )
        raw_place_tool_yaw_offset = geo.get("place_tool_yaw_offset_rad")
        if simulation_assembly_slot and raw_place_tool_yaw_offset is not None:
            try:
                place_tool_yaw_offset = float(raw_place_tool_yaw_offset)
            except (TypeError, ValueError, OverflowError):
                return {
                    "success": False,
                    "message": "place_tool_yaw_offset_rad is invalid",
                }
            if not math.isfinite(place_tool_yaw_offset):
                return {
                    "success": False,
                    "message": "place_tool_yaw_offset_rad is invalid",
                }
            base_orientation = tuple(
                pose_orientation.get(field, fallback)
                for field, fallback in zip(
                    ("qx", "qy", "qz", "qw"),
                    (0.0, 0.0, 0.0, 1.0),
                    strict=True,
                )
            )
            half_yaw = place_tool_yaw_offset * 0.5
            yaw_orientation = (
                0.0,
                0.0,
                math.sin(half_yaw),
                math.cos(half_yaw),
            )
            rotated_orientation, orientation_error = _normalized_optional_quaternion(
                *_quaternion_multiply(yaw_orientation, base_orientation)
            )
            if orientation_error or rotated_orientation is None:
                return {
                    "success": False,
                    "message": orientation_error or "place tool orientation is invalid",
                }
            pose_orientation = dict(
                zip(
                    ("qx", "qy", "qz", "qw"),
                    rotated_orientation,
                    strict=True,
                )
            )
        poses = placement_poses(
            bx, by, place_z, pose_orientation,
            simulation_assembly_slot=simulation_assembly_slot,
            insertion_depth=float(self.insertion_depth_m),
        )
        contact = geo.get('simulation_mating_contact') if simulation_assembly_slot else None
        if contact and self.controller_config.get('payload_collision', {}).get('enabled'):
            try:
                poses = self._simulation_mating_poses(
                    str(geo.get('model_name') or pick_ctx.get('model_name') or ''),
                    target_origin_pose, float(contact['target_yaw_rad']),
                )
            except (ValueError, RuntimeError) as exc:
                return {'success': False, 'message': str(exc)}
            place_z = poses['insert_pose']['z']
        insert_pose = poses["insert_pose"]
        pre_insert_pose = poses["pre_insert_pose"]
        approach_pose = poses["approach_pose"]
        move_insert_mode = "simulation_direct" if simulation_assembly_slot else ""
        # A learned seated reference is insertion authority only. Physical
        # place_approach recordings remain corrections of these nominal poses.
        place_approach_pose = deepcopy(approach_pose)
        place_target_pose = deepcopy(pre_insert_pose)
        derived_timeout_sec: float | None = None
        if physical_ur5e_board:
            geometry_result = compute_move_insert_geometry(
                part_name=target_part_name,
                product_geometry=geo,
                assembly_board_v1_aruco=frozen_board_pose,
                held_part_handoff=pick_ctx.get("held_part_handoff"),
                move_insert_profile=move_insert_profile,
                move_insert_profile_sha256=str(
                    move_insert_profile.get("profile_sha256") or ""
                ),
                z_adjustment_m=z_adjustment_m,
            )
            if not geometry_result.get("success"):
                return geometry_result
            world_board_pose = dict(geometry_result["assembly_board_v1_pose"])
            board_registration = dict(
                geometry_result["assembly_board_v1_registration"]
            )
            move_insert_profile = dict(geometry_result["move_insert_profile"])
            move_insert_mode = "force_limited"
            insertion_axis_world = dict(geometry_result["insertion_axis_world"])
            insert_pose = dict(geometry_result["insert_pose"])
            pre_insert_pose = dict(geometry_result["pre_insert_pose"])
            bx = float(geometry_result["slot_x"])
            by = float(geometry_result["slot_y"])
            board_top_z = float(geometry_result["board_top_z"])
            place_part_origin_z = float(geometry_result["place_part_origin_z"])
            place_part_origin_z_source = (
                "assembly_board-v1_aruco_to_assembly_board-v1"
            )
            place_z = float(geometry_result["place_z"])
            place_tcp_z = place_z + tcp_offset_z
            target_origin_pose = {
                **dict(geometry_result["target_origin_pose"]),
                "source": "assembly_board-v1_aruco_to_assembly_board-v1",
            }
            derived_timeout_sec = float(geometry_result["move_insert_timeout_sec"])

        result = {
            "success": True,
            "part_name": target_part_name,
            "destination_location": requested_destination,
            "slot_x": bx,
            "slot_y": by,
            "board_top_z": board_top_z,
            "place_z": place_z,
            "place_tcp_z": place_tcp_z,
            "place_part_origin_z": place_part_origin_z,
            "place_part_origin_z_source": place_part_origin_z_source,
            "approach_pose": place_approach_pose,
            "target_pose": place_target_pose,
            "pre_insert_pose": pre_insert_pose,
            "insert_pose": insert_pose,
            "insertion_axis_world": insertion_axis_world,
            "part_height": target_height,
            "tcp_offset_z": tcp_offset_z,
            "grasp_tcp_to_part_origin_z": grasp_tcp_to_part_origin_z,
            "target_reference": target_reference,
            "surface_role": str(target_reference.get("surface_role") or ""),
            "move_insert_mode": move_insert_mode,
            "target_origin_pose": target_origin_pose,
            "model_name": str(geo.get("model_name") or pick_ctx.get("model_name") or ""),
        }
        contact = geo.get('simulation_mating_contact') if simulation_assembly_slot else None
        self._simulation_mating_context = None
        if contact and self.controller_config.get('payload_collision', {}).get('enabled'):
            if any(not 0 < float(contact[key]) <= .0005
                   for key in ('axis_tolerance_m', 'max_contact_depth_m')):
                return {'success': False, 'message': 'Mating tolerance exceeds configured contact depth'}
            self._simulation_mating_context = {
                **deepcopy(contact), 'model_name': result['model_name'],
                'target_origin_pose': deepcopy(target_origin_pose),
                'pre_insert_pose': deepcopy(pre_insert_pose), 'insert_pose': deepcopy(insert_pose),
            }
            result['simulation_mating_contact'] = deepcopy(self._simulation_mating_context)
        if place_approach_skip_board_localization:
            result["place_approach_skip_board_localization"] = True
        if frozen_board_pose:
            result["assembly_board_v1_aruco"] = frozen_board_pose
        if world_board_pose:
            result["assembly_board_v1_pose"] = world_board_pose
            result["assembly_board_v1_registration"] = {
                "calibration_id": str(board_registration["calibration_id"]),
                **{
                    field: float(board_registration[field])
                    for field in ("x", "y", "z", "qx", "qy", "qz", "qw")
                },
            }
        if move_insert_profile:
            result["move_insert_profile"] = {
                key: deepcopy(value)
                for key, value in move_insert_profile.items()
                if key not in {"success", "message"}
            }
            result["move_insert_profile_sha256"] = str(
                move_insert_profile["profile_sha256"]
            )
        if derived_timeout_sec is not None:
            result["move_insert_timeout_sec"] = derived_timeout_sec
        raw_move_insert_hard_caps = geo.get("move_insert_hard_caps")
        if (
            self.execution_mode == "physical"
            and str(self.robot_name or "").strip() == "ur5e"
            and isinstance(raw_move_insert_hard_caps, Mapping)
        ):
            result["move_insert_hard_caps"] = deepcopy(
                dict(raw_move_insert_hard_caps)
            )
            result["move_insert_hard_caps_sha256"] = str(
                geo.get("move_insert_hard_caps_sha256") or ""
            )
        return result

    def get_tcp_offset_z(self) -> float:
        """Return the world-frame Z offset between EE link and TCP link."""
        if not self.wait_for_services():
            return -0.17
        return self._get_ee_tcp_world_z_offset()

    def snap_part_to_slot(
        self,
        model_name: str,
        slot_x: float,
        slot_y: float,
        part_height: float,
        board_top_z: float,
        part_origin_z: float | None = None,
        destination_location: str = "",
    ) -> bool:
        """Teleport a Gazebo model to its exact slot pose (post-placement correction)."""
        if not self.wait_for_services():
            return False
        return self._snap_part_to_slot(
            model_name,
            slot_x,
            slot_y,
            part_height,
            board_top_z,
            part_origin_z=part_origin_z,
            destination_location=destination_location,
        )

    def detect_parts(self, part_name: str | None = None) -> list[dict[str, Any]]:
        """
        ---
        description: Detect parts via perception service. Optionally filter by part name.
        params:
          part_name: {type: string, description: "Filter results to this part name. Returns all parts if omitted."}
        preconditions: {}
        effects: {}
        ---
        """
        self._last_failure_message = ""
        if not self.wait_for_services():
            if not self._last_failure_message:
                self._last_failure_message = self._unavailable_message("services not ready")
            return []
        detection_deadline = time.monotonic() + 10.0
        if not self._wait_service(
            self._detect_all_client_legacy,
            self.service_detect_all,
            detection_deadline,
        ):
            self._last_failure_message = (
                f"Camera & Perception is required for detect_parts: "
                f"{self.service_detect_all} is unavailable"
            )
            return []
        future = self._detect_all_client_legacy.call_async(self._Trigger.Request())
        result = self._wait_future(future, timeout_sec=30.0, label="detect_all_legacy")
        if not result or not result.success:
            msg = result.message if result else "timeout"
            self._last_failure_message = f"/detect_all failed: {msg}"
            self._log().error(self._last_failure_message)
            return []
        try:
            parsed = json.loads(result.message)
        except Exception as exc:
            self._last_failure_message = f"failed to parse /detect_all payload: {exc}"
            self._log().exception("Failed to parse /detect_all payload")
            return []
        parts = parsed if isinstance(parsed, list) else []
        if part_name:
            parts = [p for p in parts if p.get("part_name") == part_name]
        self._last_failure_message = ""
        return parts

    # ------------------------------------------------------------------ #
    # Standalone utility methods (called directly by UI recovery / tests)
    # ------------------------------------------------------------------ #
    def return_to_remembered_start_pose(self) -> dict[str, Any]:
        if not self.wait_for_services():
            return {
                "success": False,
                "message": self._unavailable_message("services not ready"),
            }

        if self._last_start_pose is None:
            return {"success": False, "message": "no remembered start pose available"}

        if self._cartesian_move(self._last_start_pose, "Return to remembered start pose"):
            self._last_start_pose = None
            return {"success": True, "message": "returned to remembered start pose"}
        return {"success": False, "message": "failed to return to remembered start pose"}

    def move_home(self, speed: float | None = None) -> dict[str, Any]:
        if not self.wait_for_services():
            return {
                "success": False,
                "message": self._unavailable_message("services not ready"),
            }

        home = self.named_positions.get("home")
        if not isinstance(home, (list, tuple)) or not home:
            return {"success": False, "message": "no home pose available"}

        target_positions = [float(v) for v in home]
        actual_positions, missing = self._get_arm_joint_positions(timeout_sec=0.2)
        if actual_positions is not None:
            if len(actual_positions) == len(target_positions):
                already_home = all(
                    self._angular_joint_error(actual, target) <= 0.08
                    for actual, target in zip(actual_positions, target_positions)
                )
                if already_home:
                    self._last_start_pose = None
                    return {"success": True, "message": "already at named home pose"}
        elif missing:
            self._log().warning(
                f"move_home could not confirm current joint state before homing; missing={missing}"
            )

        # Move to the explicit named joint-space home pose.
        duration_sec = self._scaled_joint_duration(self.move_home_duration_sec, speed)
        if self.execution_mode == "simulation" and self.arm_trajectory_topic:
            if self._move_joints_via_moveit(
                target_positions,
                duration_sec=duration_sec,
            ):
                self._last_start_pose = None
                return {"success": True, "message": "moved to named home pose"}
            return {
                "success": False,
                "message": self._with_last_failure("failed to move to named home pose"),
            }
        if self._arm_pub and self._publish_arm_joint_trajectory_and_wait(
            target_positions,
            duration_sec=duration_sec,
        ):
            self._last_start_pose = None
            return {"success": True, "message": "moved to named home pose"}
        if self._arm_pub:
            actual_positions, missing = self._get_arm_joint_positions(timeout_sec=0.5)
            if actual_positions is not None and len(actual_positions) == len(target_positions):
                already_home_after_attempt = all(
                    self._angular_joint_error(actual, target) <= 0.08
                    for actual, target in zip(actual_positions, target_positions)
                )
                if already_home_after_attempt:
                    self._last_start_pose = None
                    return {"success": True, "message": "already at named home pose"}
            elif missing:
                self._log().warning(
                    f"move_home could not confirm current joint state after homing attempt; missing={missing}"
                )
            if not self._exec_client:
                return {
                    "success": False,
                    "message": self._with_last_failure("failed to move to named home pose"),
                }

        # Fallback to MoveIt execute_trajectory action (for robots like
        # xarm6 that have no direct arm trajectory publisher).
        if self._exec_client and self._JointTrajectory:
            if self._move_joints_via_moveit(target_positions, duration_sec=duration_sec):
                self._last_start_pose = None
                return {"success": True, "message": "moved to named home pose"}
            actual_positions, missing = self._get_arm_joint_positions(timeout_sec=0.5)
            if actual_positions is not None and len(actual_positions) == len(target_positions):
                already_home_after_attempt = all(
                    self._angular_joint_error(actual, target) <= 0.08
                    for actual, target in zip(actual_positions, target_positions)
                )
                if already_home_after_attempt:
                    self._last_start_pose = None
                    self._log().warning(
                        "move_home execute_trajectory reported failure but current joints are already at home; treating as success"
                    )
                    return {"success": True, "message": "already at named home pose"}

        return {
            "success": False,
            "message": self._with_last_failure("failed to move to named home pose"),
        }

    def _format_moveit_error(self, code: int | None) -> str:
        if code is None:
            return "unknown error (timeout or action server failure)"
        mapping = {
            1: "SUCCESS",
            -1: "PLANNING_FAILED (No valid trajectory found. Target may be unreachable or in self-collision)",
            -2: "INVALID_MOTION_PLAN",
            -3: "MOTION_PLAN_INVALIDATED_BY_ENVIRONMENT_CHANGE",
            -4: "CONTROL_FAILED",
            -5: "UNABLE_TO_AQUIRE_SENSOR_DATA",
            -6: "TIMED_OUT",
            -7: "PREEMPTED",
            -10: "START_STATE_IN_COLLISION",
            -11: "START_STATE_VIOLATES_PATH_CONSTRAINTS",
            -12: "GOAL_IN_COLLISION (Target pose intersects with an obstacle)",
            -13: "GOAL_VIOLATES_PATH_CONSTRAINTS",
            -14: "GOAL_CONSTRAINTS_VIOLATED",
            -15: "INVALID_GROUP_NAME",
            -16: "INVALID_GOAL_CONSTRAINTS",
            -17: "INVALID_ROBOT_STATE (Robot state is outside joint limits)",
            -18: "INVALID_LINK_NAME",
            -19: "INVALID_OBJECT_NAME",
            -21: "FRAME_TRANSFORM_FAILURE",
            -22: "COLLISION_CHECKING_UNAVAILABLE",
            -23: "ROBOT_STATE_STALE",
            -24: "SENSOR_INFO_STALE",
            -31: "NO_IK_SOLUTION (Inverse Kinematics failed. Target pose is impossible to reach)",
        }
        return mapping.get(code, f"error_code={code}")

    def _cartesian_motion_only(self) -> bool:
        """Read this resource's configured Cartesian arm execution policy."""
        return (self.execution_mode == "simulation" and
                getattr(self, "controller_config", {}).get("cartesian_motion", {}).get("only") is True)

    def _move_configuration_cartesian(self, positions: list[float]) -> bool:
        """Resolve the named configuration by FK and execute a continuous TCP path."""
        from moveit_msgs.msg import RobotState
        from moveit_msgs.srv import GetPositionFK

        if not hasattr(self, "_home_fk_client"):
            self._home_fk_client = self._node.create_client(
                GetPositionFK, "/compute_fk", callback_group=self._cb_group)
        state = RobotState(is_diff=True)
        state.joint_state.name = list(self.arm_joint_names)
        state.joint_state.position = list(positions)
        request = GetPositionFK.Request(robot_state=state, fk_link_names=[self.ee_link])
        request.header.frame_id = self.frame_id
        response = self._wait_future(self._home_fk_client.call_async(request),
                                     timeout_sec=5., label="resolve Cartesian home")
        if response is None or response.error_code.val != 1 or len(response.pose_stamped) != 1:
            self._last_failure_message = "Configured home FK is unavailable"
            return False
        target = response.pose_stamped[0].pose
        if not self._move_xy_direct(target.position.x, target.position.y, target.position.z,
                                    target.orientation, "Cartesian home"):
            return False
        if not self._fresh_stable_joint_target(dict(zip(self.arm_joint_names, positions, strict=True)),
                                                tolerance=.02):
            self._last_failure_message = "Cartesian home did not reach the configured joint posture"
            return False
        return True

    def _move_joints_via_moveit(
        self,
        positions: list[float],
        duration_sec: float = 4.0,
    ) -> bool:
        """Move to joint positions through the resource-owned controller."""
        if len(positions) != len(self.arm_joint_names):
            self._log().error(
                "_move_joints_via_moveit expected "
                f"{len(self.arm_joint_names)} joints, got {len(positions)}"
            )
            return False
        if self.execution_mode == "simulation" and self.arm_trajectory_topic:
            targets = self._nearest_simulation_joint_targets(positions)
            if GazeboPickPlaceController._cartesian_motion_only(self):
                return self._move_configuration_cartesian(targets)
            return self._execute_simulation_motion_plan(
                joint_positions=targets,
                label="move_home",
            )
        try:
            from moveit_msgs.msg import RobotTrajectory
        except ImportError:
            self._log().error("moveit_msgs not available for joint move")
            return False

        traj = self._JointTrajectory()
        traj.joint_names = self._get_arm_joint_command_names()
        point = self._JointTrajectoryPoint()
        point.positions = [float(v) for v in positions]
        duration = max(0.1, float(duration_sec))
        sec = int(duration)
        nsec = int((duration - sec) * 1_000_000_000)
        point.time_from_start = self._Duration(sec=sec, nanosec=nsec)
        traj.points = [point]
        self._time_joint_target(traj)

        robot_traj = RobotTrajectory()
        robot_traj.joint_trajectory = traj

        exec_goal = self._ExecuteTrajectory.Goal()
        exec_goal.trajectory = robot_traj

        send_future = self._exec_client.send_goal_async(exec_goal)
        goal_handle = self._wait_future(send_future, timeout_sec=10.0, label="send:move_home")
        if not goal_handle or not goal_handle.accepted:
            self._log().error("move_home trajectory goal rejected")
            return False

        self._note_motion_dispatch()

        if self.execution_mode == 'simulation':
            self._simulation_goal = goal_handle
        result_future = goal_handle.get_result_async()
        try:
            result = self._wait_future(result_future, timeout_sec=30.0, label="result:move_home")
        finally:
            if self.execution_mode == 'simulation':
                if goal_handle.status not in (4, 5, 6):
                    self._cancel_simulation_goal()
                self._simulation_goal = None
        code = result.result.error_code.val if result else None
        if code != 1:
            err_msg = self._format_moveit_error(code)
            self._log().error(f"move_home execute_trajectory failed: {err_msg}")
        return code == 1

    def _nearest_simulation_joint_targets(self, positions: list[float]) -> list[float]:
        """Select equivalent arm angles near feedback within the running limits."""
        limits = self._simulation_joint_limits(self.arm_joint_names)
        observations, missing = self._get_arm_joint_positions(timeout_sec=2.0)
        if observations is None or missing:
            raise ValueError(f"Missing observed joint position: {', '.join(missing)}")
        targets = []
        for name, nominal, observed in zip(self.arm_joint_names, positions, observations, strict=True):
            bound = limits[name]
            if not math.isfinite(nominal) or not bound["lower"] <= nominal <= bound["upper"]:
                raise ValueError(f"Target exceeds joint limits: {name}")
            turns = round((observed - nominal) / (2.0 * math.pi))
            if math.isfinite(bound["lower"]):
                turns = max(turns, math.ceil((bound["lower"] - nominal) / (2.0 * math.pi)))
            if math.isfinite(bound["upper"]):
                turns = min(turns, math.floor((bound["upper"] - nominal) / (2.0 * math.pi)))
            targets.append(float(nominal + turns * 2.0 * math.pi))
        return targets

    def _execute_simulation_motion_plan(
        self,
        *,
        label: str,
        joint_positions: list[float] | None = None,
        target_pose=None,
        time_scale: float = 1.0,
    ) -> bool:
        """Plan collision-free free-space motion and use this robot's controller."""
        if GazeboPickPlaceController._cartesian_motion_only(self):
            self._last_failure_message = "Resource requires a complete Cartesian path; joint-space fallback is disabled"
            return False
        if self.execution_mode != "simulation" or not self.arm_trajectory_topic:
            return False
        from moveit_msgs.msg import (
            Constraints,
            JointConstraint,
            OrientationConstraint,
            PositionConstraint,
        )
        from shape_msgs.msg import SolidPrimitive

        if (joint_positions is None) == (target_pose is None):
            raise ValueError("A simulation motion plan requires one target type")
        request = self._GetMotionPlan.Request()
        motion = request.motion_plan_request
        motion.group_name = self.group_name
        motion.start_state.is_diff = True
        motion.allowed_planning_time = 5.0
        motion.num_planning_attempts = 3
        motion.max_velocity_scaling_factor = 1.0
        motion.max_acceleration_scaling_factor = 1.0
        constraints = Constraints()
        if joint_positions is not None:
            if len(joint_positions) != len(self.arm_joint_names):
                raise ValueError("Simulation joint target does not cover every arm joint")
            constraints.joint_constraints = [
                JointConstraint(
                    joint_name=name,
                    position=float(position),
                    tolerance_above=0.005,
                    tolerance_below=0.005,
                    weight=1.0,
                )
                for name, position in zip(
                    self.arm_joint_names, joint_positions, strict=True
                )
            ]
        else:
            position = PositionConstraint()
            position.header.frame_id = self.frame_id
            position.link_name = self.ee_link
            position.weight = 1.0
            position.constraint_region.primitives = [
                SolidPrimitive(type=SolidPrimitive.SPHERE, dimensions=[0.002])
            ]
            position.constraint_region.primitive_poses = [target_pose]
            orientation = OrientationConstraint()
            orientation.header.frame_id = self.frame_id
            orientation.link_name = self.ee_link
            orientation.orientation = target_pose.orientation
            orientation.absolute_x_axis_tolerance = 0.01
            orientation.absolute_y_axis_tolerance = 0.01
            orientation.absolute_z_axis_tolerance = 0.01
            orientation.weight = 1.0
            constraints.position_constraints = [position]
            constraints.orientation_constraints = [orientation]
        motion.goal_constraints = [constraints]

        result = None
        planning_started = time.monotonic()
        for attempt in range(1, 4):
            response = self._wait_future(
                self._motion_plan_client.call_async(request),
                timeout_sec=60.0,
                label=f"plan:{label}:free-space:{attempt}",
            )
            if response is None:
                self._last_failure_message = f"[{label}] free-space planning timed out"
                continue
            candidate = response.motion_plan_response
            if (
                candidate.error_code.val == 1
                and candidate.trajectory.joint_trajectory.points
            ):
                result = candidate
                break
            self._last_failure_message = (
                f"[{label}] collision-free planning attempt {attempt} failed: "
                f"{self._format_moveit_error(candidate.error_code.val)}"
            )
            self._log().warning(self._last_failure_message)
        self._planning_wall_time_sec += time.monotonic() - planning_started
        if result is None:
            self._log().error(self._last_failure_message)
            return False
        self._scale_trajectory_timing(result.trajectory, time_scale)
        trajectory = result.trajectory.joint_trajectory
        if not self._simulation_trajectory_is_collision_free(trajectory):
            return False
        endpoint_time = trajectory.points[-1].time_from_start
        self._trajectory_duration_sec += (
            endpoint_time.sec + endpoint_time.nanosec / 1e9
        )
        joint_endpoint_observed = self._send_simulation_joint_trajectory(
            self.arm_trajectory_topic,
            trajectory,
        )
        joint_feedback_detail = self._last_failure_message if not joint_endpoint_observed else ""
        if joint_positions is not None and not joint_endpoint_observed:
            detail = joint_feedback_detail or "controller goal failed"
            self._last_failure_message = f"[{label}] resource trajectory failed: {detail}"
            self._last_command_evidence = {
                "command_sent": True, "collision_free_motion_plan": True,
                "joint_endpoint_observed": False, "joint_feedback_detail": detail,
            }
            self._log().error(self._last_failure_message)
            return False
        controller_endpoint_observed = None
        if target_pose is not None:
            controller_endpoint_observed = self._wait_for_simulation_cartesian_endpoint(
                target_pose
            )
            if not (
                controller_endpoint_observed
                and controller_endpoint_observed["within_tolerance"]
            ):
                self._last_failure_message = (
                    f"[{label}] Cartesian endpoint not observed: "
                    f"{controller_endpoint_observed}"
                )
                self._last_command_evidence = {
                    "command_sent": True,
                    "collision_free_motion_plan": True,
                    "controller_endpoint_observed": controller_endpoint_observed,
                }
                self._log().error(self._last_failure_message)
                return False
        self._last_failure_message = ""
        self._last_command_evidence = {
            "command_sent": True,
            "controller_endpoint": self.arm_trajectory_topic.removesuffix(
                "/joint_trajectory"
            )
            + "/follow_joint_trajectory",
            "collision_free_motion_plan": True,
            "motion_path_validation": deepcopy(getattr(self, "_last_path_validation", None)),
            "controller_endpoint_observed": controller_endpoint_observed,
            "joint_endpoint_observed": joint_endpoint_observed,
            "joint_feedback_detail": joint_feedback_detail,
        }
        return True

    def _scaled_joint_duration(self, base_duration_sec: float, speed: float | None) -> float:
        scale = _as_float(speed, self.trajectory_time_scale)
        if scale <= 0.0:
            scale = self.trajectory_time_scale
        return max(0.5, float(base_duration_sec) * float(scale))

    def detach_model(self, model_name: str, *, quiet: bool = False) -> dict[str, Any]:
        if not self.wait_for_services():
            return {
                "success": False,
                "message": self._unavailable_message("services not ready"),
            }
        target_model = str(model_name or "").strip()
        if not target_model:
            return {"success": False, "message": "model name is required"}
        ok = self._detach_model_from_any_link(target_model)
        if not ok:
            ok = self._detach_part(target_model, log_failure=not quiet)
        return {
            "success": bool(ok),
            "message": "detached" if ok else f"failed to detach {target_model}",
        }

    def set_entity_pose(
        self,
        model_name: str,
        *,
        x: float,
        y: float,
        z: float,
        qx: float = 0.0,
        qy: float = 0.0,
        qz: float = 0.0,
        qw: float = 1.0,
        reference_frame: str = "world",
    ) -> dict[str, Any]:
        if not self.wait_for_services():
            return {
                "success": False,
                "message": self._unavailable_message("services not ready"),
            }
        target_model = str(model_name or "").strip()
        if not target_model:
            return {"success": False, "message": "model name is required"}
        if not self._set_state_client.wait_for_service(timeout_sec=2.0):
            return {"success": False, "message": "set_entity_state service unavailable"}

        from gazebo_msgs.msg import EntityState

        state = EntityState()
        state.name = target_model
        state.pose.position.x = float(x)
        state.pose.position.y = float(y)
        state.pose.position.z = float(z)
        state.pose.orientation.x = float(qx)
        state.pose.orientation.y = float(qy)
        state.pose.orientation.z = float(qz)
        state.pose.orientation.w = float(qw)
        state.reference_frame = str(reference_frame or "world")

        if self._link_attacher_enabled:
            for board_link in (f"anchor_{target_model}", "link"):
                try:
                    self._detach_part_from_assembly_board(target_model, board_link)
                except Exception as exc:
                    self._log().debug(
                        f"set_entity_pose board detach ignored for {target_model}:{board_link}: {exc}"
                    )

        req = self._SetEntityState.Request()
        req.state = state
        future = self._set_state_client.call_async(req)
        response = self._wait_future(
            future, timeout_sec=5.0, label=f"set_entity_pose:{target_model}"
        )
        if response and response.success:
            return {"success": True, "message": f"entity pose reset for {target_model}"}
        detail = getattr(response, "status_message", "") if response is not None else ""
        detail = str(detail or "").strip() or f"failed to set pose for {target_model}"
        return {"success": False, "message": detail}

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _log(self):
        return self._node.get_logger() if self._node else logger

    def _unavailable_message(self, default: str) -> str:
        if self._last_failure_message:
            return f"{default}: {self._last_failure_message}"
        return default

    def _with_last_failure(self, default: str) -> str:
        """Append the last controller failure to a task-level error."""
        detail = str(self._last_failure_message or "").strip()
        return f"{default}: {detail}" if detail else default

    def _wait_service(self, client, name: str, deadline: float) -> bool:
        while time.monotonic() < deadline:
            if client and client.wait_for_service(timeout_sec=0.5):
                return True
        self._log().error(f"Timed out waiting for service: {name}")
        self._last_failure_message = f"timed out waiting for service: {name}"
        return False

    def _wait_action_server(self, action_client, name: str, deadline: float) -> bool:
        while time.monotonic() < deadline:
            if action_client and action_client.wait_for_server(timeout_sec=0.5):
                return True
        self._log().error(f"Timed out waiting for action server: {name}")
        self._last_failure_message = f"timed out waiting for action server: {name}"
        return False

    def _observed_simulation_link_poses(self) -> dict | None:
        """Read one Gazebo wrist observation and apply its fixed tool transforms."""
        if getattr(self, "_shutdown_requested", False):
            self._last_failure_message = "Observed tool pose was cancelled"
            return None
        settings = self.controller_config["payload_collision"]
        source_link = settings["observation_link"]
        scoped_link = f"{self.robot_model_name}::{source_link}"
        request = self._GetEntityState.Request(name=scoped_link, reference_frame=self.frame_id)
        deadline = time.monotonic() + float(getattr(self, "tf_lookup_timeout_sec", 2.0))
        observation_error = f"Gazebo link observation timed out: {scoped_link}"
        while True:
            if getattr(self, "_shutdown_requested", False):
                self._last_failure_message = "Observed tool pose was cancelled"
                return None
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                self._last_failure_message = observation_error
                return None
            response = self._wait_future(
                self._get_state_client.call_async(request), timeout_sec=min(1.0, remaining),
                label="observe Gazebo wrist pose", timeout_log_level="debug",
            )
            if response is None:
                observation_error = f"Gazebo link observation timed out: {scoped_link}"
                continue
            if not response.success:
                self._last_failure_message = f"Gazebo link observation failed: {scoped_link}"
                return None
            stamp = response.header.stamp.sec + response.header.stamp.nanosec / 1e9
            now = self._node.get_clock().now().nanoseconds / 1e9
            if now - stamp <= 0.25:
                break
            observation_error = f"Gazebo link observation is {now - stamp:.3f}s old"
        observed = response.state.pose
        parent = {
            "x": observed.position.x, "y": observed.position.y, "z": observed.position.z,
            "qx": observed.orientation.x, "qy": observed.orientation.y,
            "qz": observed.orientation.z, "qw": observed.orientation.w,
        }
        poses = {}
        try:
            for link in (self.ee_link, self.tcp_link):
                # These tool frames are fixed descendants of the configured
                # wrist link, so delayed world TF cannot alter the observation.
                transform = self._tf_buffer.lookup_transform(
                    source_link, link, self._rclpy.time.Time(),
                ).transform
                translation, rotation = transform.translation, transform.rotation
                relative = {
                    "x": translation.x, "y": translation.y, "z": translation.z,
                    "qx": rotation.x, "qy": rotation.y, "qz": rotation.z, "qw": rotation.w,
                }
                composed = _compose_pose(parent, relative)
                pose = self._Pose()
                pose.position.x, pose.position.y, pose.position.z = (composed[k] for k in ("x", "y", "z"))
                pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = (
                    composed[k] for k in ("qx", "qy", "qz", "qw")
                )
                poses[link] = pose
        except (self._tf2_ros.LookupException, self._tf2_ros.ConnectivityException,
                self._tf2_ros.ExtrapolationException, ValueError) as exc:
            self._last_failure_message = f"Fixed tool transform is unavailable: {exc}"
            return None
        self._last_pose_observation = {
            "source": "gazebo_link_state", "link": scoped_link,
            "simulation_stamp": stamp, "pose": parent,
        }
        return poses

    def _get_ee_pose(self):
        if (self.execution_mode == "simulation"
                and getattr(self, "controller_config", {}).get("payload_collision", {}).get("enabled")):
            poses = self._observed_simulation_link_poses()
            return poses.get(self.ee_link) if poses else None
        timeout_sec = max(
            0.0,
            float(getattr(self, "tf_lookup_timeout_sec", 2.0)),
        )
        deadline = time.monotonic() + timeout_sec
        last_error: Exception | None = None
        while True:
            try:
                transform = self._tf_buffer.lookup_transform(
                    self.frame_id,
                    self.ee_link,
                    self._rclpy.time.Time(),
                )
                if self.execution_mode == "simulation":
                    observed_at = transform.header.stamp.sec + transform.header.stamp.nanosec / 1e9
                    current = self._node.get_clock().now().nanoseconds / 1e9
                    if current - observed_at > 0.25:
                        raise RuntimeError(f"End-effector feedback is {current - observed_at:.3f}s old")
                pose = self._Pose()
                pose.position.x = transform.transform.translation.x
                pose.position.y = transform.transform.translation.y
                pose.position.z = transform.transform.translation.z
                pose.orientation = transform.transform.rotation
                if str(getattr(self, "_last_failure_message", "")).startswith(
                    "tf lookup failed for "
                ):
                    self._last_failure_message = ""
                return pose
            except (
                self._tf2_ros.LookupException,
                self._tf2_ros.ConnectivityException,
                self._tf2_ros.ExtrapolationException,
                RuntimeError,
            ) as exc:
                last_error = exc
            remaining_sec = deadline - time.monotonic()
            if remaining_sec <= 0.0:
                break
            time.sleep(min(0.05, remaining_sec))

        detail = str(last_error or "transform unavailable").strip()
        self._last_failure_message = (
            f"tf lookup failed for {self.frame_id} -> {self.ee_link} "
            f"after {timeout_sec:.1f}s: {detail}"
        )
        self._log().error(self._last_failure_message)
        return None

    def _get_ee_tcp_world_z_offset(self) -> float:
        if (self.execution_mode == "simulation"
                and getattr(self, "controller_config", {}).get("payload_collision", {}).get("enabled")):
            poses = self._observed_simulation_link_poses()
            if poses is None:
                raise RuntimeError(self._last_failure_message)
            return poses[self.tcp_link].position.z - poses[self.ee_link].position.z
        try:
            ee_tf = self._tf_buffer.lookup_transform(
                self.frame_id, self.ee_link, self._rclpy.time.Time()
            )
            tcp_tf = self._tf_buffer.lookup_transform(
                self.frame_id, self.tcp_link, self._rclpy.time.Time()
            )
            return tcp_tf.transform.translation.z - ee_tf.transform.translation.z
        except Exception:
            self._log().warn("Could not get world EE-to-TCP offset, using default -0.17m")
            return -0.17

    def _on_joint_state(self, msg):
        samples = list(zip(msg.name, msg.position))
        if self.execution_mode == 'simulation':
            owned = {*self.arm_joint_names, self.gripper_joint}
            samples = [(name, position) for name, position in samples if name in owned]
        if not samples:
            return
        with self._joint_lock:
            received = time.monotonic()
            stamp = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
            if (self.execution_mode == 'simulation' and self._last_joint_sim_time is not None
                    and stamp < self._last_joint_sim_time):
                now = self._node.get_clock().now().nanoseconds / 1e9
                if now < self._last_joint_sim_time - 1.0:
                    self._joint_positions.clear()
                    self._joint_received_times.clear()
                    self._joint_sim_stamps.clear()
                    self._joint_stable_since.clear()
                    self._joint_limits_cache.clear()
                    self._last_joint_sim_time = None
            if self.execution_mode == "simulation":
                # Arm and gripper publishers need not stamp their samples at
                # the same instant. Order feedback within each joint stream.
                samples = [(name, position) for name, position in samples
                           if stamp >= self._joint_sim_stamps.get(name, -math.inf)]
                if not samples:
                    return
            self._last_joint_sim_time = max(stamp, self._last_joint_sim_time or stamp)
            for name, pos in samples:
                previous = self._joint_positions.get(name)
                if previous is None or abs(previous - pos) > 0.001:
                    self._joint_stable_since[name] = received
                self._joint_positions[name] = pos
                self._joint_received_times[name] = received
                self._joint_sim_stamps[name] = stamp
            self._joint_state_received_monotonic = received

    def _fresh_stable_joint_target(
        self,
        targets: dict[str, float],
        *,
        tolerance: float,
        stable_for_sec: float = 0.05,
    ) -> bool:
        """Return true only for a fresh, stable endpoint with no active goal."""
        if self.execution_mode != 'simulation' or self._simulation_goal is not None:
            return False
        now = time.monotonic()
        with self._joint_lock:
            for name, target in targets.items():
                value = self._joint_positions.get(name)
                received = self._joint_received_times.get(name, 0.)
                stable_since = self._joint_stable_since.get(name, now)
                if (
                    value is None
                    or now - received > 1.
                    or now - stable_since < stable_for_sec
                    or (
                        self._angular_joint_error(value, target)
                        if name in self.arm_joint_names else abs(value - target)
                    ) > tolerance
                ):
                    return False
        return True

    def _current_simulation_state_is_collision_free(self) -> bool:
        """Validate the current stable arm state before retaining it as workflow home."""

        if self.execution_mode != "simulation" or self._simulation_goal is not None:
            return False
        positions, missing = self._get_arm_joint_positions(timeout_sec=0.2)
        if positions is None or missing or self._state_validity_client is None:
            return False
        targets = dict(zip(self.arm_joint_names, positions, strict=True))
        if not self._fresh_stable_joint_target(
            targets, tolerance=0.005, stable_for_sec=0.0
        ):
            return False
        from moveit_msgs.srv import GetStateValidity

        request = GetStateValidity.Request()
        request.group_name = self.group_name
        request.robot_state.is_diff = True
        request.robot_state.joint_state.name = list(self.arm_joint_names)
        request.robot_state.joint_state.position = positions
        response = self._wait_future(
            self._state_validity_client.call_async(request),
            timeout_sec=3.0,
            label="retain observed clear pose",
            timeout_log_level="debug",
        )
        return bool(response is not None and response.valid)

    def _get_joint_position(self, joint_name: str) -> float | None:
        with self._joint_lock:
            def fresh(name: str) -> bool:
                return (self.execution_mode != 'simulation'
                        or time.monotonic() - self._joint_received_times.get(name, 0) <= 1.0)

            exact = self._joint_positions.get(joint_name)
            if exact is not None:
                return exact if fresh(joint_name) else None

            prefix = f"{self.robot_name}_"
            prefixed_name = (
                joint_name if str(joint_name).startswith(prefix) else f"{prefix}{joint_name}"
            )
            prefixed = self._joint_positions.get(prefixed_name)
            if prefixed is not None:
                return prefixed if fresh(prefixed_name) else None

            suffix_matches = [
                (name, value)
                for name, value in self._joint_positions.items()
                if str(name).endswith(str(joint_name))
            ]
            if len(suffix_matches) == 1:
                name, value = suffix_matches[0]
                return value if fresh(name) else None
            return None

    def _get_arm_joint_positions(
        self,
        *,
        timeout_sec: float = 0.0,
    ) -> tuple[list[float] | None, list[str]]:
        deadline = time.monotonic() + max(0.0, float(timeout_sec))
        missing: list[str] = []
        while True:
            values: list[float] = []
            missing = []
            for joint_name in self.arm_joint_names:
                value = self._get_joint_position(joint_name)
                if value is None:
                    missing.append(joint_name)
                else:
                    values.append(float(value))
            if not missing:
                return values, []
            if time.monotonic() >= deadline:
                return None, missing
            time.sleep(0.02)

    def _get_arm_joint_command_names(self) -> list[str]:
        with self._joint_lock:
            available_names = set(str(name) for name in self._joint_positions.keys())

        if not available_names:
            return list(self.arm_joint_names)

        resolved: list[str] = []
        prefix = f"{self.robot_name}_"
        for joint_name in self.arm_joint_names:
            exact_name = str(joint_name)
            prefixed_name = exact_name if exact_name.startswith(prefix) else f"{prefix}{exact_name}"
            if exact_name in available_names:
                resolved.append(exact_name)
            elif prefixed_name in available_names:
                resolved.append(prefixed_name)
            else:
                resolved.append(exact_name)
        return resolved

    @staticmethod
    def _angular_joint_error(actual: float, target: float) -> float:
        return abs(math.atan2(math.sin(actual - target), math.cos(actual - target)))

    def _wait_for_arm_joint_targets(
        self,
        targets: list[float],
        timeout_sec: float,
        *,
        tolerance_rad: float = 0.08,
        log_miss: bool = True,
    ) -> bool:
        if len(targets) != len(self.arm_joint_names):
            return False

        duration = (max(float(timeout_sec), getattr(self, '_last_arm_duration_sec', 0.)+2.)
                    if self.execution_mode == 'simulation' else max(0., float(timeout_sec)))
        pending = self._motion_pending(duration)
        saw_feedback = False
        last_values: list[float] | None = None
        missing: list[str] = []

        while pending():
            values, missing = self._get_arm_joint_positions(timeout_sec=0.0)
            if values is not None:
                saw_feedback = True
                last_values = list(values)
                if all(
                    self._angular_joint_error(values[index], targets[index]) <= tolerance_rad
                    for index in range(len(targets))
                ):
                    return True
            time.sleep(0.02)

        if not log_miss:
            return False

        if not saw_feedback:
            self._log().warn(f"Timed out waiting for arm joint feedback; missing={missing}")
            return False

        max_error = max(
            self._angular_joint_error(last_values[index], targets[index])
            for index in range(len(targets))
        )
        self._log().warn(
            "Timed out waiting for arm joint target; "
            f"max_error={max_error:.4f}rad tolerance={tolerance_rad:.4f}rad"
        )
        return False

    def _wait_for_gripper_target(
        self,
        target: float,
        timeout_sec: float,
        *,
        log_miss: bool = True,
    ) -> bool:
        pending = self._motion_pending(timeout_sec)
        saw_feedback = False
        last_pos = None
        while self._rclpy.ok() and pending():
            pos = self._get_joint_position(self.gripper_joint)
            if pos is not None:
                saw_feedback = True
                last_pos = pos
                if abs(pos - target) <= self.gripper_position_tol:
                    return True
            time.sleep(0.02)

        if not log_miss:
            return False
        if not saw_feedback:
            self._last_failure_message = (
                f"no joint-state feedback for '{self.gripper_joint}' while waiting gripper move"
            )
            self._log().warn(
                f"No joint-state feedback for '{self.gripper_joint}' while waiting gripper move"
            )
        else:
            self._last_failure_message = (
                f"gripper target not reached: target={target:.3f} current={float(last_pos):.3f}"
            )
            self._log().warn(
                f"Gripper target not reached: target={target:.3f} current={float(last_pos):.3f}"
            )
        return False

    def _gripper_is_open_enough(self) -> bool:
        pos = self._get_joint_position(self.gripper_joint)
        if pos is None:
            return False
        midpoint = (float(self.gripper_open) + float(self.gripper_close)) * 0.5
        tol = max(float(self.gripper_position_tol) * 2.0, 0.01)
        if self.gripper_open >= self.gripper_close:
            return float(pos) >= (midpoint - tol)
        return float(pos) <= (midpoint + tol)

    def _get_link_world_position(self, link_name: str) -> tuple[float, float, float] | None:
        link_name = str(link_name or "").strip()
        if not link_name:
            return None
        try:
            transform = self._tf_buffer.lookup_transform(
                self.frame_id,
                link_name,
                self._rclpy.time.Time(),
            )
        except Exception:
            return None
        translation = transform.transform.translation
        return (
            float(translation.x),
            float(translation.y),
            float(translation.z),
        )

    def _get_entity_world_position(
        self,
        model_name: str,
        timeout_log_level: str = "error",
    ) -> tuple[float, float, float] | None:
        target_model = str(model_name or "").strip()
        mode = str(getattr(self, "execution_mode", "") or "").strip().lower()
        if not target_model or mode == "physical":
            return None
        if not self._get_state_client or not getattr(self, "_GetEntityState", None):
            return None
        if not self._get_state_client.wait_for_service(timeout_sec=0.2):
            return None

        req = self._GetEntityState.Request()
        req.name = target_model
        req.reference_frame = str(self.frame_id or "world")
        future = self._get_state_client.call_async(req)
        response = self._wait_future(
            future,
            timeout_sec=1.0,
            label=f"get_entity_state:{target_model}",
            timeout_log_level=timeout_log_level,
        )
        if not response or not getattr(response, "success", False):
            return None
        pose = response.state.pose
        return (
            float(pose.position.x),
            float(pose.position.y),
            float(pose.position.z),
        )

    @staticmethod
    def _xyz_distance(
        a: tuple[float, float, float],
        b: tuple[float, float, float],
    ) -> float:
        return math.sqrt(
            (float(a[0]) - float(b[0])) ** 2
            + (float(a[1]) - float(b[1])) ** 2
            + (float(a[2]) - float(b[2])) ** 2
        )

    def _verify_detach_timeout_release(
        self,
        target_model: str,
        timeout_log_level: str = "error",
    ) -> bool | None:
        target_model = str(target_model or "").strip()
        if not target_model:
            return None

        link_candidates: list[str] = []
        if self._attached_link:
            link_candidates.append(str(self._attached_link))
        if self.primary_attach_link and self.primary_attach_link not in link_candidates:
            link_candidates.append(self.primary_attach_link)
        for link_name in self.attach_link_candidates:
            if link_name not in link_candidates:
                link_candidates.append(link_name)
        if not link_candidates:
            return None

        deadline = time.monotonic() + max(0.0, self.release_detach_verify_timeout_sec)
        best_min_distance: float | None = None
        best_link_name = ""
        while True:
            model_position = self._get_entity_world_position(
                target_model,
                timeout_log_level=timeout_log_level,
            )
            if model_position is None:
                return None

            min_distance: float | None = None
            min_link_name = ""
            for link_name in link_candidates:
                link_position = self._get_link_world_position(link_name)
                if link_position is None:
                    continue
                distance = self._xyz_distance(model_position, link_position)
                if min_distance is None or distance < min_distance:
                    min_distance = distance
                    min_link_name = link_name

            if min_distance is None:
                return None

            if best_min_distance is None or min_distance > best_min_distance:
                best_min_distance = min_distance
                best_link_name = min_link_name

            if min_distance > self.release_detach_verify_distance_m:
                self._log().info(
                    f"Verified detach fallback for {target_model}: closest link "
                    f"{min_link_name or '<unknown>'} is {min_distance:.3f}m away"
                )
                return True

            if time.monotonic() >= deadline:
                break
            time.sleep(self.release_detach_verify_poll_sec)

        self._log().warn(
            f"Detach fallback verification failed for {target_model}: closest link "
            f"{best_link_name or '<unknown>'} remained within "
            f"{float(best_min_distance or 0.0):.3f}m "
            f"(threshold={self.release_detach_verify_distance_m:.3f}m)"
        )
        return False

    def _gripper_command(
        self,
        position: float,
        label: str,
        move_time_s: float | None = None,
        wait_s: float | None = None,
        require_target: bool = False,
        log_target_miss: bool | None = None,
    ) -> bool:
        if not self._gripper_pub:
            self._last_failure_message = "gripper publisher is not configured"
            self._log().error("Gripper publisher is not configured")
            return False

        if self._fresh_stable_joint_target(
            {self.gripper_joint: float(position)}, tolerance=self.gripper_position_tol,
        ):
            self._last_command_evidence = {
                "command_sent": False,
                "reason": "fresh stable endpoint already observed",
                "target": float(position),
                "position": self._get_joint_position(self.gripper_joint),
            }
            self._last_failure_message = ""
            return True

        move_time_s = self.gripper_move_time_sec if move_time_s is None else float(move_time_s)
        wait_s = self.gripper_settle_sec if wait_s is None else float(wait_s)
        require_target = require_target or self.execution_mode == 'simulation'
        if log_target_miss is None:
            log_target_miss = bool(require_target)

        self._log().info(f"Gripper: {label} (position={position:.3f})")
        traj = self._JointTrajectory()
        traj.joint_names = [self.gripper_joint]
        point = self._JointTrajectoryPoint()
        point.positions = [position]
        move_time_s = max(0.05, move_time_s)
        sec = int(move_time_s)
        nsec = int((move_time_s - sec) * 1_000_000_000)
        point.time_from_start = self._Duration(sec=sec, nanosec=nsec)
        traj.points = [point]
        self._time_joint_target(traj)

        if self.execution_mode == 'simulation':
            if not self._send_simulation_joint_trajectory(self.gripper_topic, traj):
                self._last_failure_message = 'Simulation gripper controller did not acknowledge completion'
                return False
        else:
            self._gripper_pub.publish(traj)
            time.sleep(self._scaled_wall_wait_sec(0.05))
            self._gripper_pub.publish(traj)

        duration = point.time_from_start.sec + point.time_from_start.nanosec / 1e9
        feedback_timeout = max(duration + self.gripper_feedback_timeout_pad_sec, 1.0)
        reached = self._wait_for_gripper_target(
            position,
            feedback_timeout,
            log_miss=bool(log_target_miss),
        )
        if require_target and not reached:
            if log_target_miss:
                self._log().error(
                    f"Gripper command did not reach required target: target={position:.3f}"
                )
            if not self._last_failure_message:
                self._last_failure_message = (
                    f"gripper command did not reach required target {position:.3f}"
                )
            return False
        self._wait_process_time(max(0.0, wait_s))
        self._last_failure_message = ""
        self._last_command_evidence = {
            "command_sent": True,
            "target": float(position),
            "position": self._get_joint_position(self.gripper_joint),
        }
        return True

    def _wait_future(
        self,
        future,
        timeout_sec: float,
        label: str,
        timeout_log_level: str = "error",
    ):
        deadline = time.monotonic() + timeout_sec
        pending = (self._motion_pending(timeout_sec)
                   if self.execution_mode == 'simulation' and label.startswith('result:')
                   else lambda: time.monotonic() < deadline)
        completed = threading.Event()
        try:
            future.add_done_callback(lambda _future: completed.set())
        except AttributeError:
            pass
        while self._rclpy.ok() and not future.done() and pending():
            completed.wait(0.05)
        if not future.done():
            try:
                future.cancel()
            except Exception:
                pass
            message = f"[{label}] timed out"
            logger = self._log()
            level = str(timeout_log_level or "error").strip().lower()
            if level == "debug":
                log_method = (
                    getattr(logger, "debug", None)
                    or getattr(logger, "warn", None)
                    or getattr(logger, "warning", None)
                )
                if log_method:
                    log_method(message)
                else:
                    logger.error(message)
            elif level in {"warn", "warning"}:
                log_method = getattr(logger, "warn", None) or getattr(
                    logger,
                    "warning",
                    None,
                )
                if log_method:
                    log_method(message)
                else:
                    logger.error(message)
            elif level == "info" and hasattr(logger, "info"):
                logger.info(message)
            else:
                logger.error(message)
            return None
        return future.result()

    def _scale_trajectory_timing(self, solution, scale: float):
        if solution is None:
            return
        # `scale` multiplies trajectory duration:
        #   >1.0 => slower, <1.0 => faster, ==1.0 => unchanged.
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError('Trajectory timing scale must be positive and finite')
        joint_traj = solution.joint_trajectory
        if not joint_traj.points:
            return
        for point in joint_traj.points:
            total_ns = int(point.time_from_start.sec) * 1_000_000_000 + int(
                point.time_from_start.nanosec
            )
            scaled_ns = max(1, int(total_ns * scale))
            point.time_from_start.sec = scaled_ns // 1_000_000_000
            point.time_from_start.nanosec = scaled_ns % 1_000_000_000
            if point.velocities:
                point.velocities = [v / scale for v in point.velocities]
            if point.accelerations:
                point.accelerations = [a / (scale * scale) for a in point.accelerations]
        self._validate_simulation_trajectory(joint_traj)

    def _sync_part_collision(self, model_name: str, *, attached_link: str | None = None) -> bool:
        """Synchronize this robot's observed payload before acknowledging custody."""
        config = getattr(self, "controller_config", {}).get("payload_collision", {})
        if not config.get("enabled") or self.execution_mode != "simulation":
            return True
        from moveit_msgs.srv import ApplyPlanningScene, GetPlanningScene
        from moveit_msgs.msg import PlanningSceneComponents
        from cais_spade_llm.recovery_framework.part_collision import (
            collision_geometry_evidence, grasp_point_evidence, observed_part_boxes, part_scene_update,
        )

        evidence = {"model_name": model_name, "attached_link": attached_link,
                    "physical_attachment_completed": self._attached_model == model_name,
                    "collision_scene_acknowledged": False}
        self._last_command_evidence = {"payload_collision": evidence}
        if not hasattr(self, "_payload_scene_clients"):
            self._payload_scene_clients = {
                "get": self._node.create_client(GetPlanningScene, "/get_planning_scene", callback_group=self._cb_group),
                "apply": self._node.create_client(ApplyPlanningScene, "/apply_planning_scene", callback_group=self._cb_group),
            }
        def call(client, query, label):
            if not client.wait_for_service(timeout_sec=2.0):
                raise RuntimeError(f"Payload scene service unavailable: {label}")
            result = self._wait_future(client.call_async(query), timeout_sec=5.0, label=label)
            if result is None:
                raise RuntimeError(f"Payload scene observation/acknowledgement missing: {label}")
            return result
        try:
            # Observe the physical attachment transform directly. Converting a
            # world pose through MoveIt's delayed robot state can put the payload
            # far from the gripper during concurrent motion.
            reference_frame = (
                f"{self.robot_model_name}::{attached_link}" if attached_link else "world"
            )
            evidence["collision_frame"] = attached_link or "world"
            evidence["observed_reference_frame"] = reference_frame
            observed = call(self._get_state_client, self._GetEntityState.Request(
                name=model_name, reference_frame=reference_frame), "payload_pose")
            if not observed.success:
                raise ValueError(f"Cannot observe payload {model_name}")
            rows = observed_part_boxes(model_name, observed.state.pose, support_allowance=(
                float(config["support_contact_allowance_m"]) if attached_link else 0.0))
            evidence["collision_objects"] = collision_geometry_evidence(rows)
            if attached_link:
                try:
                    tcp = self._tf_buffer.lookup_transform(
                        attached_link, self.tcp_link, self._rclpy.time.Time(),
                    ).transform.translation
                except self._tf2_ros.TransformException as exc:
                    raise RuntimeError(f"Cannot observe payload grasp frame: {exc}") from exc
                evidence.update(grasp_point_evidence(
                    rows, [tcp.x, tcp.y, tcp.z], self.cartesian_position_tolerance_m,
                ))
                if not evidence["payload_at_gripper"]:
                    raise RuntimeError(f"Observed payload {model_name} is outside the gripper: "
                                       f"{evidence['tcp_to_payload_distance_m']:.4f} m")
            current = call(self._payload_scene_clients["get"], GetPlanningScene.Request(
                components=PlanningSceneComponents(components=(PlanningSceneComponents.ALLOWED_COLLISION_MATRIX
                    | PlanningSceneComponents.WORLD_OBJECT_NAMES | PlanningSceneComponents.ROBOT_STATE_ATTACHED_OBJECTS))),
                "payload_scene_observation").scene
            touch_links = [name for name in current.allowed_collision_matrix.entry_names
                           if name.startswith(config["robot_prefix"] + "rg2")]
            touch_links.extend(self.attach_link_candidates)
            update = part_scene_update(model_name, rows, attached_link=attached_link, touch_links=touch_links,
                world_ids={obj.id for obj in current.world.collision_objects},
                attached_ids={obj.object.id for obj in current.robot_state.attached_collision_objects})
            response = call(self._payload_scene_clients["apply"], ApplyPlanningScene.Request(scene=update), "payload_scene")
            if not response.success:
                raise RuntimeError(f"MoveIt rejected payload scene for {model_name}")
            evidence["collision_scene_acknowledged"] = True
            return True
        except (OSError, KeyError, TypeError, ValueError, RuntimeError) as exc:
            evidence["error"] = str(exc)
            self._last_failure_message = str(exc)
            self._log().error(str(exc))
            return False

    def _attach_part(self, model_name: str) -> bool:
        if not self._link_attacher_enabled:
            return True
        if not model_name:
            self._log().error("Cannot attach: empty model name")
            return False

        if self._attached_model and self._attached_model != model_name:
            self._detach_part(self._attached_model)

        for link_name in self.attach_link_candidates:
            req = self._attach_srv.Request()
            req.model1_name = self.robot_model_name
            req.link1_name = link_name
            req.model2_name = model_name
            req.link2_name = "link"

            future = self._attach_client.call_async(req)
            response = self._wait_future(future, timeout_sec=5.0, label=f"attach:{link_name}")
            if response is None:
                continue
            if response and response.success:
                self._attached_model = model_name
                self._attached_link = link_name
                self.clear_motion_preparation()
                return self._sync_part_collision(model_name, attached_link=link_name)

            msg = response.message if response else "no response"
            if "already attached to another link" in str(msg).lower():
                self._detach_model_from_any_link(model_name)
                continue

        self._log().error(f"Failed to attach {model_name}")
        return False

    def _detach_model_from_any_link(self, target_model: str) -> bool:
        if not self._link_attacher_enabled:
            return True
        if not target_model:
            return False
        if not self._detach_client.wait_for_service(timeout_sec=0.5):
            return False

        detached_any = False
        for link_name in self.attach_link_candidates:
            req = self._detach_srv.Request()
            req.model1_name = self.robot_model_name
            req.link1_name = link_name
            req.model2_name = target_model
            req.link2_name = "link"

            future = self._detach_client.call_async(req)
            response = self._wait_future(
                future,
                timeout_sec=self.detach_timeout_sec,
                label=f"detach-recover:{target_model}:{link_name}",
            )
            if response and response.success:
                detached_any = True

        if detached_any and self._attached_model == target_model:
            self._attached_model = None
            self._attached_link = None
            self.clear_motion_preparation()
        if detached_any:
            return self._sync_part_collision(target_model)
        return False

    def _detach_part(
        self,
        model_name: str = "",
        timeout_sec: float | None = None,
        attached_link_only: bool = False,
        log_failure: bool = True,
        timeout_log_level: str = "error",
        break_on_timeout: bool = True,
        prefer_attached_link: bool = True,
        extra_link_candidates: list[str] | tuple[str, ...] | None = None,
    ) -> bool:
        if not self._link_attacher_enabled:
            return True

        target_model = model_name or self._attached_model
        if not target_model:
            return True
        if not self._detach_client.wait_for_service(timeout_sec=0.5):
            return False
        detach_timeout = _as_float(timeout_sec, self.detach_timeout_sec)
        if detach_timeout <= 0.0:
            detach_timeout = self.detach_timeout_sec

        links_to_try: list[str] = []
        if prefer_attached_link and self._attached_link:
            links_to_try.append(self._attached_link)
        if not (attached_link_only and links_to_try):
            if self.primary_attach_link and self.primary_attach_link not in links_to_try:
                links_to_try.append(self.primary_attach_link)
            for link in self.attach_link_candidates:
                if link not in links_to_try:
                    links_to_try.append(link)
            for link in extra_link_candidates or []:
                link_name = str(link or "").strip()
                if link_name and link_name not in links_to_try:
                    links_to_try.append(link_name)
        if (
            not prefer_attached_link
            and self._attached_link
            and self._attached_link not in links_to_try
        ):
            links_to_try.append(self._attached_link)
        max_link_attempts = (
            len(links_to_try)
            if not break_on_timeout and not attached_link_only
            else max(1, self.detach_max_link_attempts)
        )
        links_to_try = links_to_try[: max(1, max_link_attempts)]

        for link_name in links_to_try:
            req = self._detach_srv.Request()
            req.model1_name = self.robot_model_name
            req.link1_name = link_name
            req.model2_name = target_model
            req.link2_name = "link"

            future = self._detach_client.call_async(req)
            response = self._wait_future(
                future,
                timeout_sec=detach_timeout,
                label=f"detach:{link_name}",
                timeout_log_level=timeout_log_level,
            )
            if response and response.success:
                self._attached_model = None
                self._attached_link = None
                return self._sync_part_collision(target_model)

            if response is None:
                if log_failure:
                    suffix = "skipping remaining links" if break_on_timeout else "trying next link"
                    self._log().error(f"Detach service timed out on {link_name}, {suffix}")
                if break_on_timeout:
                    break
                continue

        if log_failure:
            self._log().error(f"Failed to detach {target_model}")
        return False

    def _snap_part_to_slot(
        self,
        model_name: str,
        slot_x: float,
        slot_y: float,
        part_height: float,
        board_top_z: float,
        part_origin_z: float | None = None,
        destination_location: str = "",
    ) -> bool:
        if not self._set_state_client.wait_for_service(timeout_sec=2.0):
            return False

        from gazebo_msgs.msg import EntityState

        state = EntityState()
        state.name = model_name
        state.pose.position.x = slot_x
        state.pose.position.y = slot_y
        state.pose.position.z = _as_float(
            part_origin_z,
            board_top_z + (part_height * 0.5),
        )
        state.pose.orientation.w = 1.0
        mating_evidence = None
        mating = getattr(self, '_simulation_mating_context', None)
        if (self.execution_mode == 'simulation' and mating
                and mating['model_name'] == model_name):
            from cais_spade_llm.recovery_framework.part_collision import mating_pose_valid

            observed = self._wait_future(self._get_state_client.call_async(self._GetEntityState.Request(
                name=model_name, reference_frame=self.frame_id)),
                timeout_sec=5., label='observe seated mating part')
            if observed is None or not observed.success:
                return False
            p, q = observed.state.pose.position, observed.state.pose.orientation
            actual = [p.x, p.y, p.z, q.x, q.y, q.z, q.w]
            corridor = {**mating, 'start_part_z': mating['target_origin_pose']['z']}
            if not mating_pose_valid(corridor, actual):
                self._last_failure_message = 'Released mating part is not seated in its configured pose'
                self._last_command_evidence = {'seated_mating_part': {'observed_pose': actual, 'valid': False}}
                return False
            state.pose.orientation = deepcopy(q)
            mating_evidence = {'observed_pose': actual, 'valid': True, 'orientation_preserved': True}
        state.twist.linear.x = 0.0
        state.twist.linear.y = 0.0
        state.twist.linear.z = 0.0
        state.twist.angular.x = 0.0
        state.twist.angular.y = 0.0
        state.twist.angular.z = 0.0
        state.reference_frame = "world"

        self._detach_part(
            model_name,
            timeout_sec=self._simulation_release_detach_timeout_sec(),
            attached_link_only=False,
            log_failure=False,
            timeout_log_level="debug",
            break_on_timeout=False,
            prefer_attached_link=False,
            extra_link_candidates=getattr(self, "release_detach_link_candidates", []),
        )

        if not self._set_entity_state_for_snap(model_name, state):
            return False
        retain_fixture = bool(mating_evidence and mating.get('retain_fixture_attachment'))
        if destination_location in {"", "assembly_board-v1"}:
            if not self._attach_part_to_assembly_board(
                model_name,
                destination_location=destination_location,
            ):
                return False
            if not retain_fixture and not self._detach_part_from_assembly_board(model_name, "link"):
                return False
        if not retain_fixture and not self._set_entity_state_for_snap(model_name, state):
            return False
        if not self._verify_snapped_entity_position(
            model_name,
            (slot_x, slot_y, float(state.pose.position.z)),
        ):
            return False

        if not self._sync_part_collision(model_name):
            return False
        if mating_evidence is not None:
            mating_evidence['fixture_attachment_retained'] = retain_fixture
            mating_evidence['ignored_tooth_contacts'] = list(mating.get('ignore_tooth_contact_with', []))
            self._last_command_evidence['seated_mating_part'] = mating_evidence
        self._log().info(
            f"snap_to_slot stabilized {model_name} at "
            f"({slot_x:.3f}, {slot_y:.3f}, {state.pose.position.z:.3f})"
        )
        return True

    def _verify_snapped_entity_position(
        self,
        model_name: str,
        expected_position: tuple[float, float, float],
    ) -> bool:
        """Require repeated Gazebo observations at the commanded slot position."""
        samples = max(
            1,
            int(getattr(self, "snap_to_slot_observation_samples", 3) or 3),
        )
        interval_sec = max(
            0.05,
            _as_float(
                getattr(self, "snap_to_slot_observation_interval_sec", None),
                0.1,
            ),
        )
        tolerance_m = max(
            0.0005,
            _as_float(
                getattr(self, "snap_to_slot_position_tolerance_m", None),
                0.001,
            ),
        )
        for _sample_index in range(samples):
            self._wait_process_time(interval_sec)
            observed_position = self._get_entity_world_position(
                model_name,
                timeout_log_level="warn",
            )
            if observed_position is None:
                self._log().error(
                    f"snap_to_slot could not observe {model_name} after attachment"
                )
                return False
            position_error_m = self._xyz_distance(
                observed_position,
                expected_position,
            )
            if position_error_m > tolerance_m:
                self._log().error(
                    f"snap_to_slot observed {model_name} {position_error_m:.4f}m "
                    f"from its target; tolerance={tolerance_m:.4f}m"
                )
                return False
        return True

    def _set_entity_state_for_snap(self, model_name: str, state) -> bool:
        req = self._SetEntityState.Request()
        req.state = state

        attempts = max(1, 1 + int(getattr(self, "snap_to_slot_retry_count", 0) or 0))
        timeout_sec = _as_float(getattr(self, "snap_to_slot_timeout_sec", None), 5.0)
        retry_delay_sec = _as_float(getattr(self, "snap_to_slot_retry_delay_sec", None), 0.25)
        saw_success = False
        for attempt_idx in range(attempts):
            future = self._set_state_client.call_async(req)
            response = self._wait_future(
                future,
                timeout_sec=timeout_sec,
                label="snap_to_slot",
            )
            if response and response.success:
                saw_success = True
                if attempt_idx + 1 < attempts:
                    time.sleep(retry_delay_sec)
                continue
            if attempt_idx + 1 < attempts:
                self._log().warn(
                    f"snap_to_slot retry {attempt_idx + 1}/{attempts - 1} for {model_name}"
                )
                time.sleep(retry_delay_sec)
        if saw_success:
            return True
        return False

    def _attach_part_to_assembly_board(
        self,
        model_name: str,
        *,
        destination_location: str = "",
    ) -> bool:
        if not self._link_attacher_enabled:
            return True
        target_model = str(model_name or "").strip()
        if not target_model:
            return False
        destination = str(destination_location or "").strip()
        if destination and destination != "assembly_board-v1":
            return True
        if not self._attach_client.wait_for_service(timeout_sec=0.5):
            return False

        board_links = ["link"]
        for board_link in board_links:
            self._detach_part_from_assembly_board(target_model, board_link)

        last_message = ""
        for attempt_idx in range(2):
            for board_link in board_links:
                req = self._attach_srv.Request()
                req.model1_name = "assembly_board_v1"
                req.link1_name = board_link
                req.model2_name = target_model
                req.link2_name = "link"

                future = self._attach_client.call_async(req)
                response = self._wait_future(
                    future,
                    timeout_sec=5.0,
                    label=f"attach:assembly_board_v1:{board_link}:{target_model}",
                    timeout_log_level="warn",
                )
                if response and response.success:
                    if self._attached_model == target_model:
                        self._attached_model = None
                        self._attached_link = None
                    return True

                last_message = str(response.message if response else "no response")
                msg = last_message.lower()
                if "failed to find link" in msg and board_link != "link":
                    continue
                if "already attached" in msg and board_link == f"anchor_{target_model}":
                    self._detach_part(
                        target_model,
                        timeout_sec=self._simulation_release_detach_timeout_sec(),
                        attached_link_only=False,
                        log_failure=False,
                        timeout_log_level="debug",
                        break_on_timeout=False,
                        prefer_attached_link=False,
                        extra_link_candidates=getattr(self, "release_detach_link_candidates", []),
                    )
                    self._detach_part_from_assembly_board(target_model, board_link)
                    break
            if attempt_idx == 0:
                time.sleep(self._scaled_wall_wait_sec(0.05))
        self._log().error(f"Failed to attach {target_model} to assembly_board_v1: {last_message}")
        return False

    def _detach_part_from_assembly_board(self, model_name: str, board_link: str) -> bool:
        if not self._link_attacher_enabled:
            return True
        target_model = str(model_name or "").strip()
        link_name = str(board_link or "").strip()
        if not target_model or not link_name:
            return False
        if not self._detach_client.wait_for_service(timeout_sec=0.5):
            return False

        req = self._detach_srv.Request()
        req.model1_name = "assembly_board_v1"
        req.link1_name = link_name
        req.model2_name = target_model
        req.link2_name = "link"
        future = self._detach_client.call_async(req)
        response = self._wait_future(
            future,
            timeout_sec=0.5,
            label=f"detach:assembly_board_v1:{link_name}:{target_model}",
            timeout_log_level="debug",
        )
        return bool(response and response.success)

    def _observed_cartesian_endpoint_evidence(self, target) -> dict[str, Any] | None:
        """Compare the observed tool pose with a commanded Cartesian endpoint."""
        if not getattr(self, "_last_simulation_controller_succeeded", False):
            return None
        observed = self._get_ee_pose()
        if observed is None:
            return None

        position_error_m = math.dist(
            (
                observed.position.x,
                observed.position.y,
                observed.position.z,
            ),
            (
                target.position.x,
                target.position.y,
                target.position.z,
            ),
        )
        observed_quaternion = (
            observed.orientation.x,
            observed.orientation.y,
            observed.orientation.z,
            observed.orientation.w,
        )
        target_quaternion = (
            target.orientation.x,
            target.orientation.y,
            target.orientation.z,
            target.orientation.w,
        )
        observed_norm = math.sqrt(sum(value * value for value in observed_quaternion))
        target_norm = math.sqrt(sum(value * value for value in target_quaternion))
        if observed_norm <= 1e-12 or target_norm <= 1e-12:
            return None
        normalized_dot = sum(
            observed_value * target_value
            for observed_value, target_value in zip(
                observed_quaternion,
                target_quaternion,
                strict=True,
            )
        ) / (observed_norm * target_norm)
        orientation_error_rad = 2.0 * math.acos(
            min(1.0, max(-1.0, abs(normalized_dot)))
        )
        position_tolerance_m = self.cartesian_position_tolerance_m
        orientation_tolerance_rad = self.cartesian_orientation_tolerance_rad
        return {
            "observed_cartesian_endpoint": True,
            "position_error_m": position_error_m,
            "position_tolerance_m": position_tolerance_m,
            "orientation_error_rad": orientation_error_rad,
            "orientation_tolerance_rad": orientation_tolerance_rad,
            "within_tolerance": (
                position_error_m <= position_tolerance_m
                and orientation_error_rad <= orientation_tolerance_rad
            ),
        }

    def _wait_for_simulation_cartesian_endpoint(self, target) -> dict[str, Any] | None:
        """Observe the commanded pose before another primitive consumes TF."""
        if not getattr(self, "_last_simulation_controller_succeeded", False):
            return None
        pending = self._motion_pending(2.0)
        evidence = self._observed_cartesian_endpoint_evidence(target)
        while not (evidence and evidence["within_tolerance"]) and pending():
            time.sleep(0.01)
            evidence = self._observed_cartesian_endpoint_evidence(target)
        return evidence

    def _simulation_mating_poses(self, model_name: str, target_origin: dict, yaw: float) -> dict:
        """Align the actual held CAD part with its configured shaft and orientation."""
        from cais_spade_llm.recovery_framework.geometry import multiply, rotate

        if self._attached_model != model_name or not math.isfinite(yaw):
            raise ValueError('Mating placement requires the identified attached part')
        current = self._get_ee_pose()
        observed = self._wait_future(self._get_state_client.call_async(self._GetEntityState.Request(
            name=model_name, reference_frame=self.frame_id)),
            timeout_sec=5., label='observe held mating transform')
        if current is None or observed is None or not observed.success:
            raise RuntimeError('Cannot observe the held mating transform')
        p, q = observed.state.pose.position, observed.state.pose.orientation
        ee = current.orientation
        inverse = [-ee.x, -ee.y, -ee.z, ee.w]
        offset = rotate(inverse, [p.x - current.position.x, p.y - current.position.y, p.z - current.position.z])
        relative = multiply(inverse, [q.x, q.y, q.z, q.w])
        target_q = [0., 0., math.sin(yaw / 2), math.cos(yaw / 2)]
        tool_q = multiply(target_q, [-relative[0], -relative[1], -relative[2], relative[3]])
        world_offset = rotate(tool_q, offset)
        xyz = [float(target_origin[axis]) - world_offset[i] for i, axis in enumerate(('x', 'y', 'z'))]
        if not all(math.isfinite(value) for value in [*xyz, *tool_q]):
            raise ValueError('Observed mating transform is not finite')
        return placement_poses(*xyz, dict(zip(('qx', 'qy', 'qz', 'qw'), tool_q)),
                               simulation_assembly_slot=True, insertion_depth=float(self.insertion_depth_m))

    def _simulation_mating_segment(self, target) -> dict | None:
        """Authorize configured contact only for an aligned vertical insertion."""
        context = getattr(self, '_simulation_mating_context', None)
        if (self.execution_mode != 'simulation' or not context
                or self._attached_model != context['model_name']):
            return None
        from cais_spade_llm.recovery_framework.geometry import multiply, rotate
        from cais_spade_llm.recovery_framework.part_collision import mating_pose_valid

        endpoint = context['pre_insert_pose']
        tolerance = float(context['axis_tolerance_m'])
        if (math.hypot(target.position.x - endpoint['x'], target.position.y - endpoint['y']) > tolerance
                or not context['insert_pose']['z'] - tolerance <= target.position.z <= endpoint['z'] + tolerance):
            return None
        current = self._get_ee_pose()
        if current is None or math.hypot(current.position.x - target.position.x,
                                        current.position.y - target.position.y) > tolerance:
            return None
        expected_q = [endpoint[key] for key in ('qx', 'qy', 'qz', 'qw')]
        for pose in (current, target):
            q = pose.orientation
            if abs(sum(a * b for a, b in zip(expected_q, (q.x, q.y, q.z, q.w)))) < math.cos(.01):
                return None
        observed = self._wait_future(self._get_state_client.call_async(self._GetEntityState.Request(
            name=context['model_name'], reference_frame=self.frame_id)),
            timeout_sec=5., label='observe aligned mating part')
        if observed is None or not observed.success:
            return None
        p, q = observed.state.pose.position, observed.state.pose.orientation
        authorization = {**deepcopy(context), 'start_part_z': p.z}
        if not mating_pose_valid(authorization, [p.x, p.y, p.z, q.x, q.y, q.z, q.w]):
            return None
        ee = current.orientation
        inverse = [-ee.x, -ee.y, -ee.z, ee.w]
        authorization['tool_to_part_pose'] = [
            *rotate(inverse, [p.x - current.position.x, p.y - current.position.y, p.z - current.position.z]),
            *multiply(inverse, [q.x, q.y, q.z, q.w]),
        ]
        return authorization

    def _validate_simulation_mating_contact(self, authorization: dict, contacts, state) -> bool:
        """Check each contacting timed sample against the same narrow CAD corridor."""
        from moveit_msgs.srv import GetPositionFK
        from cais_spade_llm.recovery_framework.geometry import compose
        from cais_spade_llm.recovery_framework.part_collision import mating_contacts_allowed, mating_pose_valid

        if not mating_contacts_allowed(authorization, contacts):
            return False
        if not hasattr(self, '_mating_fk_client'):
            self._mating_fk_client = self._node.create_client(
                GetPositionFK, '/compute_fk', callback_group=self._cb_group)
        request = GetPositionFK.Request(robot_state=state, fk_link_names=[self.ee_link])
        request.header.frame_id = self.frame_id
        result = self._wait_future(self._mating_fk_client.call_async(request),
                                   timeout_sec=5., label='validate mating axis')
        if result is None or result.error_code.val != 1 or len(result.pose_stamped) != 1:
            return False
        pose = result.pose_stamped[0].pose
        p, q = pose.position, pose.orientation
        part = compose([p.x, p.y, p.z, q.x, q.y, q.z, q.w], authorization['tool_to_part_pose'])
        return mating_pose_valid(authorization, part)

    def _resolve_cartesian_waypoints(self, waypoints: list, *, mating_contact: dict | None):
        """Convert this primitive's explicit waypoints without a motion planner."""
        from builtin_interfaces.msg import Duration
        from moveit_msgs.srv import GetPositionIK
        from cais_spade_llm.resources.robot.cartesian_waypoints import resolve_waypoints, robot_trajectory

        current = self._get_ee_pose()
        positions, missing = self._get_arm_joint_positions(timeout_sec=2.)
        if current is None or positions is None:
            raise ValueError(f"Cartesian waypoint start observation unavailable: {missing}")
        if not hasattr(self, "_waypoint_ik_client"):
            self._waypoint_ik_client = self._node.create_client(
                GetPositionIK, "/compute_ik", callback_group=self._cb_group)
        def values(pose):
            p, q = pose.position, pose.orientation
            return [p.x, p.y, p.z, q.x, q.y, q.z, q.w]
        def solve(target, seed):
            if self._shutdown_requested:
                raise InterruptedError("Cartesian waypoint execution cancelled")
            query = GetPositionIK.Request()
            request = query.ik_request
            request.group_name, request.ik_link_name = self.group_name, self.ee_link
            request.pose_stamped.header.frame_id = self.frame_id
            request.pose_stamped.pose = self._make_pose(*target[:3], self._make_orientation(*target[3:]))
            request.robot_state.is_diff = True
            request.robot_state.joint_state.name = list(self.arm_joint_names)
            request.robot_state.joint_state.position = seed
            request.avoid_collisions = not mating_contact
            request.timeout = Duration(nanosec=200_000_000)
            response = self._wait_future(self._waypoint_ik_client.call_async(query),
                                         timeout_sec=5., label="resolve Cartesian waypoint IK")
            if response is None or response.error_code.val != 1:
                raise ValueError(f"Cartesian waypoint IK is unavailable at {target}")
            joints = dict(zip(response.solution.joint_state.name, response.solution.joint_state.position))
            return [joints[name] for name in self.arm_joint_names]
        settings = self.controller_config["cartesian_motion"]
        rows = resolve_waypoints(start_pose=values(current), start_joints=positions,
            waypoints=[values(pose) for pose in waypoints], names=self.arm_joint_names,
            limits=self._simulation_joint_limits(self.arm_joint_names), solve_ik=solve,
            linear_step=settings["linear_step_m"], angular_step=settings["angular_step_rad"],
            maximum_joint_step=settings["max_joint_step_rad"])
        return robot_trajectory(self.arm_joint_names, rows)

    def _cartesian_move(
        self,
        target,
        label: str = "",
        avoid_collisions: bool = True,
        min_fraction: float = 0.9,
        allow_partial: bool = False,
        time_scale: float | None = None,
        waypoints: list | None = None,
    ) -> bool:
        self._last_command_evidence = {"command_sent": False, "motion_method": "Cartesian waypoints"}
        mating_contact = (self._simulation_mating_segment(target)
                          if getattr(self, '_simulation_mating_context', None) else None)
        planning_started = time.monotonic()
        solution = None
        preparation_reason = "not eligible"
        if GazeboPickPlaceController._cartesian_motion_only(self):
            try:
                solution = self._resolve_cartesian_waypoints([*(waypoints or []), target], mating_contact=mating_contact)
            except (ValueError, RuntimeError) as exc:
                self._last_failure_message = f"[{label}] {exc}"
                return False
            finally:
                self._planning_wall_time_sec += time.monotonic() - planning_started
            preparation_reason = "explicit Cartesian waypoints"
        elif avoid_collisions and not allow_partial and not mating_contact and not waypoints:
            solution, preparation_reason = self._consume_prepared_cartesian(target)
        if solution is None:
            request = self._GetCartesianPath.Request()
            request.header.frame_id = self.frame_id
            request.header.stamp = self._node.get_clock().now().to_msg()
            request.group_name = self.group_name
            request.link_name = self.ee_link
            request.waypoints = [*(waypoints or []), target]
            request.max_step = 0.01
            request.jump_threshold = 2.0
            # Kinematic insertion candidates still undergo full timed collision
            # checks below, including FK and depth bounds for the intended shaft.
            request.avoid_collisions = avoid_collisions and not mating_contact
            request.start_state.is_diff = True

            future = self._cart_client.call_async(request)
            response = self._wait_future(future, timeout_sec=10.0, label=f"plan:{label}")
            self._planning_wall_time_sec += time.monotonic() - planning_started
            if response is None:
                self._last_failure_message = f"[{label}] planning response timed out"
                self._log().error(f"[{label}] planning response timed out")
                return False
            if hasattr(response, "error_code") and response.error_code.val != 1:
                self._last_failure_message = f"[{label}] Cartesian planning failed: {response.error_code.val}"
                return False
            if response.fraction < min_fraction:
                self._last_failure_message = (
                    f"[{label}] planning fraction too low: {response.fraction:.3f} < {min_fraction:.3f}"
                )
                self._log().error(
                    f"[{label}] planning fraction too low: {response.fraction:.3f} < {min_fraction:.3f}"
                )
                return False
            if response.fraction < 0.999 and not allow_partial:
                self._last_failure_message = f"[{label}] planning fraction incomplete: {response.fraction:.3f} (partial not allowed)"
                self._log().error(
                    f"[{label}] planning fraction incomplete: {response.fraction:.3f} (partial not allowed)"
                )
                return False
            solution = response.solution
        elif not GazeboPickPlaceController._cartesian_motion_only(self):
            self._planning_wall_time_sec += time.monotonic() - planning_started

        if GazeboPickPlaceController._cartesian_motion_only(self):
            points = solution.joint_trajectory.points
            maximum_step = float(self.controller_config["cartesian_motion"]["max_joint_step_rad"])
            if not points or any(abs(b - a) > maximum_step
                                 for left, right in zip(points, points[1:])
                                 for a, b in zip(left.positions, right.positions, strict=True)):
                self._last_failure_message = "Cartesian trajectory has an IK joint discontinuity"
                return False
        exec_goal = self._ExecuteTrajectory.Goal()
        if time_scale is None:
            scale = self.trajectory_time_scale
        else:
            scale = _as_float(time_scale, self.trajectory_time_scale)
        self._scale_trajectory_timing(solution, scale)
        valid = (self._simulation_trajectory_is_collision_free(
            solution.joint_trajectory, mating_contact=mating_contact) if mating_contact
            else self._simulation_trajectory_is_collision_free(solution.joint_trajectory))
        if not valid:
            return False
        endpoint_time = solution.joint_trajectory.points[-1].time_from_start
        self._trajectory_duration_sec += endpoint_time.sec + endpoint_time.nanosec / 1e9
        if self.execution_mode == "simulation" and self.arm_trajectory_topic:
            joint_endpoint_observed = self._send_simulation_joint_trajectory(
                self.arm_trajectory_topic,
                solution.joint_trajectory,
            )
            controller_endpoint_observed = self._wait_for_simulation_cartesian_endpoint(
                target
            )
            if not (
                controller_endpoint_observed
                and controller_endpoint_observed["within_tolerance"]
            ):
                detail = self._last_failure_message or "Cartesian endpoint not observed"
                self._last_failure_message = (
                    f"[{label}] resource trajectory failed: {detail}; "
                    f"endpoint={controller_endpoint_observed}"
                )
                self._last_command_evidence = {
                    "command_sent": True,
                    "controller_endpoint_observed": controller_endpoint_observed,
                }
                self._log().error(self._last_failure_message)
                return False
            if not joint_endpoint_observed:
                self._log().warning(
                    f"[{label}] accepted observed Cartesian endpoint after controller "
                    "joint-target settling mismatch: "
                    f"position_error={controller_endpoint_observed['position_error_m']:.6f}m, "
                    f"orientation_error={controller_endpoint_observed['orientation_error_rad']:.6f}rad"
                )
            self._last_failure_message = ""
            self._last_command_evidence = {
                "command_sent": True,
                "motion_method": "Cartesian waypoints",
                "waypoint_count": len(waypoints or []) + 1,
                "controller_endpoint": self.arm_trajectory_topic.removesuffix(
                    "/joint_trajectory"
                )
                + "/follow_joint_trajectory",
                "preparation_reused": preparation_reason.startswith("reused"),
                "preparation_result": preparation_reason,
                "motion_path_validation": deepcopy(getattr(self, "_last_path_validation", None)),
                "controller_endpoint_observed": controller_endpoint_observed,
            }
            return True

        exec_goal.trajectory = solution

        send_future = self._exec_client.send_goal_async(exec_goal)
        goal_handle = self._wait_future(send_future, timeout_sec=10.0, label=f"send:{label}")
        if not goal_handle or not goal_handle.accepted:
            self._last_failure_message = f"[{label}] trajectory goal rejected by execute action"
            self._log().error(f"[{label}] trajectory goal rejected by execute action")
            return False

        self._note_motion_dispatch()

        if self.execution_mode == 'simulation':
            self._simulation_goal = goal_handle
            self._start_queued_motion_preparation(solution, target.orientation)
        result_future = goal_handle.get_result_async()
        try:
            result = self._wait_future(result_future, timeout_sec=30.0, label=f"result:{label}")
        finally:
            if self.execution_mode == 'simulation':
                if goal_handle.status not in (4, 5, 6):
                    self._cancel_simulation_goal()
                self._simulation_goal = None
        code = result.result.error_code.val if result else None
        if code != 1:
            err_msg = self._format_moveit_error(code)
            self._last_failure_message = f"[{label}] execute_trajectory failed: {err_msg}"
            self._log().error(self._last_failure_message)
            return False
        self._last_failure_message = ""
        self._last_command_evidence = {
            "command_sent": True,
            "preparation_reused": preparation_reason.startswith("reused"),
            "preparation_result": preparation_reason,
        }
        return True

    def _move_xy_at_z(
        self,
        x: float,
        y: float,
        z: float,
        *,
        orientation=None,
        label: str = "move_xy_at_z",
        speed: float | None = None,
    ) -> dict[str, Any]:
        if not self.wait_for_services():
            return {"success": False, "message": self._unavailable_message("services not ready")}
        if orientation is None:
            ee = self._get_ee_pose()
            if ee is None:
                return {
                    "success": False,
                    "message": self._unavailable_message("cannot read current ee pose"),
                }
            orientation = ee.orientation
        ok = self._move_xy_direct(
            float(x),
            float(y),
            float(z),
            orientation,
            label,
            time_scale=_as_float(speed, self.trajectory_time_scale),
        )
        if not ok:
            return {
                "success": False,
                "message": self._unavailable_message(
                    f"failed to move above target ({x}, {y}, {z})"
                ),
            }
        return {"success": True, "message": f"moved above target ({x:.4f}, {y:.4f}, {z:.4f})"}

    def _move_pose_direct(
        self,
        x: float,
        y: float,
        z: float,
        *,
        orientation=None,
        label: str = "move_pose_direct",
        speed: float | None = None,
        avoid_collisions: bool = True,
        min_fraction: float = 0.9,
        allow_partial: bool = False,
    ) -> dict[str, Any]:
        if not self.wait_for_services():
            return {"success": False, "message": self._unavailable_message("services not ready")}
        if orientation is None:
            ee = self._get_ee_pose()
            if ee is None:
                return {
                    "success": False,
                    "message": self._unavailable_message("cannot read current ee pose"),
                }
            orientation = ee.orientation
        ok = self._cartesian_move(
            self._make_pose(float(x), float(y), float(z), orientation),
            label,
            avoid_collisions=avoid_collisions,
            min_fraction=min_fraction,
            allow_partial=allow_partial,
            time_scale=_as_float(speed, self.trajectory_time_scale),
        )
        if not ok:
            return {"success": False, "message": f"failed to move directly to ({x}, {y}, {z})"}
        return {"success": True, "message": f"moved directly to ({x:.4f}, {y:.4f}, {z:.4f})"}

    def _release_part_sequence(
        self,
        *,
        model_name: str,
        slot_x: float,
        slot_y: float,
        part_height: float,
        board_top_z: float,
        place_z: float,
        travel_z: float,
        orientation=None,
    ) -> dict[str, Any]:
        if not self.wait_for_services():
            return {"success": False, "message": self._unavailable_message("services not ready")}

        if orientation is None:
            ee = self._get_ee_pose()
            if ee is None:
                return {
                    "success": False,
                    "message": self._unavailable_message("cannot read current ee pose"),
                }
            orientation = ee.orientation

        released = self.release_part(str(model_name))
        if not released.get("success"):
            return {
                "success": False,
                "message": str(released.get("message") or "failed to release part"),
            }

        if model_name:
            self._snap_part_to_slot(
                str(model_name),
                float(slot_x),
                float(slot_y),
                float(part_height),
                float(board_top_z),
            )

        lift_ok = self._cartesian_move(
            self._make_pose(float(slot_x), float(slot_y), float(travel_z), orientation),
            "Lift after place",
        )
        if not lift_ok:
            lift_target = self._make_pose(
                float(slot_x), float(slot_y), float(travel_z), orientation
            )
            if self.execution_mode == "simulation":
                lift_ok = self._execute_simulation_motion_plan(
                    label="Lift after place",
                    target_pose=lift_target,
                )
            else:
                lift_ok = self._cartesian_move(
                    lift_target,
                    "Lift after place (no-collision)",
                    avoid_collisions=False,
                    min_fraction=0.70,
                    allow_partial=True,
                )

        if lift_ok:
            return {"success": True, "message": "released part and lifted clear"}
        return {"success": False, "message": "failed to lift clear after release"}

    def _move_xy_direct(
        self,
        target_x: float,
        target_y: float,
        z: float,
        orientation,
        label_prefix: str,
        *,
        time_scale: float | None = None,
    ) -> bool:
        if self._cartesian_move(
            self._make_pose(target_x, target_y, z, orientation),
            label_prefix,
            time_scale=time_scale,
        ):
            return True

        if GazeboPickPlaceController._cartesian_motion_only(self):
            if (self._last_command_evidence or {}).get("command_sent"):
                return False
            current = self._get_ee_pose()
            if current is None:
                return False
            current_xyz = [current.position.x, current.position.y, current.position.z]
            target = self._make_pose(target_x, target_y, z, orientation)
            for route in self.controller_config["cartesian_motion"].get("transit_waypoints", []):
                ordered = list(route)
                if math.dist(current_xyz, ordered[-1]) < math.dist(current_xyz, ordered[0]):
                    ordered.reverse()
                waypoints = [self._make_pose(*xyz, orientation) for xyz in ordered]
                if self._cartesian_move(target, label_prefix, time_scale=time_scale, waypoints=waypoints):
                    return True
                if (self._last_command_evidence or {}).get("command_sent"):
                    return False
            return False

        current = self._get_ee_pose()
        if current is None:
            self._log().error(f"[{label_prefix}] cannot read current EE pose for staged fallback")
            return False
        if math.isclose(float(current.position.x), float(target_x), abs_tol=1e-6) and math.isclose(
            float(current.position.y), float(target_y), abs_tol=1e-6
        ):
            if self.execution_mode == "simulation":
                return self._execute_simulation_motion_plan(
                    label=label_prefix,
                    target_pose=self._make_pose(target_x, target_y, z, orientation),
                    time_scale=time_scale or 1.0,
                )
            self._log().warn(
                f"[{label_prefix}] direct Cartesian move failed with no XY delta; "
                "retrying direct no-collision fallback"
            )
            return self._cartesian_move(
                self._make_pose(target_x, target_y, z, orientation),
                f"{label_prefix} (no-collision)",
                avoid_collisions=False,
                min_fraction=0.70,
                allow_partial=True,
                time_scale=time_scale,
            )

        if self.execution_mode == "simulation":
            self._log().warn(
                f"[{label_prefix}] direct Cartesian move failed; retrying "
                "collision-free free-space planning"
            )
            return self._execute_simulation_motion_plan(
                label=label_prefix,
                target_pose=self._make_pose(target_x, target_y, z, orientation),
                time_scale=time_scale or 1.0,
            )
        self._log().warn(
            f"[{label_prefix}] direct Cartesian move failed; retrying staged XY fallback"
        )
        for axis_order in (("x", "y"), ("y", "x")):
            if self._move_xy_axis_order(
                current=current,
                target_x=target_x,
                target_y=target_y,
                z=z,
                orientation=orientation,
                label_prefix=label_prefix,
                axis_order=axis_order,
                time_scale=time_scale,
            ):
                if axis_order == ("y", "x"):
                    self._log().info(
                        f"[{label_prefix}] staged XY fallback succeeded with axis order Y->X"
                    )
                return True
        return False

    def _move_xy_axis_order(
        self,
        *,
        current,
        target_x: float,
        target_y: float,
        z: float,
        orientation,
        label_prefix: str,
        axis_order: tuple[str, str],
        time_scale: float | None = None,
    ) -> bool:
        start_x = float(current.position.x)
        start_y = float(current.position.y)
        order_suffix = "" if axis_order == ("x", "y") else ", alt-order"
        first_axis = axis_order[0]
        first_target_x = target_x if first_axis == "x" else start_x
        first_target_y = target_y if first_axis == "y" else start_y
        leg_targets = [
            (first_axis, first_target_x, first_target_y),
            (axis_order[1], target_x, target_y),
        ]

        current_x = start_x
        current_y = start_y
        for axis_name, leg_x, leg_y in leg_targets:
            label_base = f"{label_prefix} (leg {axis_name.upper()}{order_suffix}"
            step_targets = self._split_xy_leg_targets(
                start_x=current_x,
                start_y=current_y,
                target_x=leg_x,
                target_y=leg_y,
            )
            for step_index, (step_x, step_y) in enumerate(step_targets, start=1):
                step_suffix = (
                    f", step {step_index}/{len(step_targets)}" if len(step_targets) > 1 else ""
                )
                if not self._cartesian_move(
                    self._make_pose(step_x, step_y, z, orientation),
                    f"{label_base}{step_suffix})",
                    min_fraction=0.85,
                    allow_partial=False,
                    time_scale=time_scale,
                ):
                    if not self._cartesian_move(
                        self._make_pose(step_x, step_y, z, orientation),
                        f"{label_base}{step_suffix}, no-collision)",
                        avoid_collisions=False,
                        min_fraction=0.70,
                        allow_partial=True,
                        time_scale=time_scale,
                    ):
                        return False
            current_x = leg_x
            current_y = leg_y
        return True

    def _split_xy_leg_targets(
        self,
        *,
        start_x: float,
        start_y: float,
        target_x: float,
        target_y: float,
    ) -> list[tuple[float, float]]:
        delta_x = float(target_x) - float(start_x)
        delta_y = float(target_y) - float(start_y)
        span = max(abs(delta_x), abs(delta_y))
        if span <= 1e-9:
            return []
        segments = max(1, int(math.ceil(span / self.xy_axis_step_m)))
        return [
            (
                float(start_x) + (delta_x * idx / segments),
                float(start_y) + (delta_y * idx / segments),
            )
            for idx in range(1, segments + 1)
        ]

    def _make_pose(self, x: float, y: float, z: float, orientation):
        pose = self._Pose()
        pose.position.x = float(x)
        pose.position.y = float(y)
        pose.position.z = float(z)
        pose.orientation = orientation
        return pose

    def _make_orientation(self, qx: float, qy: float, qz: float, qw: float):
        orientation = self._Pose().orientation
        orientation.x = float(qx)
        orientation.y = float(qy)
        orientation.z = float(qz)
        orientation.w = float(qw)
        return orientation


UR5E_JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]
UR5E_TRAJECTORY_TOPIC = "/ur5e_joint_trajectory_controller/joint_trajectory"
UR5E_JOINT_STATES_TOPIC = "/joint_states"

XARM6_JOINT_NAMES = [
    "xarm6_joint1",
    "xarm6_joint2",
    "xarm6_joint3",
    "xarm6_joint4",
    "xarm6_joint5",
    "xarm6_joint6",
]
XARM6_JOINT_STATES_TOPIC = "/joint_states"


class UR5eGazeboController(GazeboPickPlaceController):
    """Config-driven UR5e Gazebo controller."""

    def __init__(
        self,
        trajectory_topic: str = UR5E_TRAJECTORY_TOPIC,
        joint_states_topic: str = UR5E_JOINT_STATES_TOPIC,
        *,
        controller_config: dict[str, Any] | None = None,
        named_positions: dict[str, Any] | None = None,
        execution_mode: str = "simulation",
        arm_joint_names: list[str] | None = None,
        node_name: str | None = None,
    ) -> None:
        super().__init__(
            robot_name="ur5e",
            node_name=node_name or f"ur5e_controller_{os.getpid()}",
            controller_config=controller_config or {},
            named_positions=named_positions,
            execution_mode=execution_mode,
            arm_joint_names=arm_joint_names or UR5E_JOINT_NAMES,
            arm_trajectory_topic=trajectory_topic,
            joint_states_topic=joint_states_topic,
        )


class XArm6GazeboController(GazeboPickPlaceController):
    """Config-driven xArm6 Gazebo controller."""

    def __init__(
        self,
        *,
        trajectory_topic: str | None = None,
        joint_states_topic: str = XARM6_JOINT_STATES_TOPIC,
        controller_config: dict[str, Any] | None = None,
        named_positions: dict[str, Any] | None = None,
        execution_mode: str = "simulation",
    ) -> None:
        super().__init__(
            robot_name="xarm6",
            node_name=f"xarm6_controller_{os.getpid()}",
            controller_config=controller_config or {},
            named_positions=named_positions,
            execution_mode=execution_mode,
            arm_joint_names=XARM6_JOINT_NAMES,
            arm_trajectory_topic=trajectory_topic,
            joint_states_topic=joint_states_topic,
        )
