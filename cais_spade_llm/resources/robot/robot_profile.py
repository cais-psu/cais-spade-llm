"""Built-in robot resource profile and robot-specific bridge helpers."""

from __future__ import annotations

from copy import deepcopy
from textwrap import dedent
from typing import Any

from cais_spade_llm.resources.resource_profile import (
    ResourceProfile,
    register_resource_profile,
    resource_snapshot_carried_entity,
    resource_snapshot_field_value,
)
from cais_spade_llm.resources.robot import UR5eController, XArm6Controller


def _availability_from_state(
    raw_snapshot: dict[str, Any],
    current_state: str,
    *,
    busy_states: set[str],
) -> str:
    availability = str(raw_snapshot.get("availability", "") or "").strip().lower()
    if availability:
        return availability
    normalized_state = str(current_state or "").strip().lower()
    if normalized_state in {"faulted", "down", "offline", "error"}:
        return "unavailable"
    if normalized_state in busy_states:
        return "busy"
    return "available"


def _robot_availability(raw_snapshot: dict[str, Any], current_state: str) -> str:
    return _availability_from_state(
        raw_snapshot,
        current_state,
        busy_states={"busy", "picked", "positioned", "placed", "at_pick"},
    )


def _robot_facet(snapshot: dict[str, Any]) -> dict[str, Any]:
    return {
        "held_part": resource_snapshot_field_value(snapshot, "held_part"),
        "gripper_state": resource_snapshot_field_value(snapshot, "gripper_state"),
        "current_pose": deepcopy(resource_snapshot_field_value(snapshot, "current_pose")),
        "current_pose_ref": resource_snapshot_field_value(snapshot, "current_pose_ref"),
        "named_poses": {
            str(pose_name): str(pose_name)
            for pose_name in (resource_snapshot_field_value(snapshot, "named_poses") or [])
            if str(pose_name).strip()
        },
    }


def _robot_occupancy(current_location: Any, snapshot: dict[str, Any]) -> dict[str, Any]:
    occupancy: dict[str, Any] = {}
    if current_location not in (None, ""):
        occupancy["location"] = deepcopy(current_location)
    held_item = resource_snapshot_field_value(snapshot, "held_part")
    if held_item not in (None, ""):
        occupancy["held_parts"] = [deepcopy(held_item)]
    return occupancy


def _sync_robot_pose(agent: Any, pose: Any) -> None:
    if isinstance(pose, dict) and {"x", "y", "z"} <= set(pose.keys()):
        agent._position = {
            "x": float(pose["x"]),
            "y": float(pose["y"]),
            "z": float(pose["z"]),
        }


def _robot_carried_location(resource_jid: str, _snapshot: dict[str, Any] | None = None) -> str:
    return f"{str(resource_jid or '').strip()}_gripper"


def _robot_event_family(event: dict[str, Any]) -> str:
    event_name = str(event.get("event_name", "") or "").strip().lower()
    has_pick_verb = "pick" in event_name or "grasp" in event_name or "acquire" in event_name
    has_place_verb = (
        "place" in event_name
        or "insert" in event_name
        or "assemble" in event_name
        or "stage" in event_name
        or "return" in event_name
        or "release" in event_name
    )
    resource_delta = dict(event.get("expected_resource_delta") or {})
    delta_from = str(resource_delta.get("from", "") or "").strip().lower()
    delta_to = str(resource_delta.get("to", "") or "").strip().lower()
    part_delta = dict(event.get("expected_part_delta") or {})
    part_to = str(part_delta.get("to", "") or "").strip().lower()
    part_name = str(event.get("part_name", "") or "").strip()

    if not part_name:
        if "home" in event_name:
            return "home"
        if "clear" in event_name or delta_to == "idle":
            return "clear"
        return ""

    if has_pick_verb and has_place_verb:
        return "pick_place"
    if part_to == "assembled" or "assemble" in event_name or "insert" in event_name:
        if delta_from in {"idle", "recovery_required"} or "pick_place" in event_name:
            return "pick_place"
        return "assemble"
    if part_to == "in_gripper" or delta_to == "picked" or "repick" in event_name:
        return "pick"
    if (
        "stage" in event_name
        or "return" in event_name
        or "release" in event_name
        or part_to == "ready"
    ):
        return "stage"
    if "place" in event_name:
        return "place"
    if "pick" in event_name:
        return "pick"
    return ""


def _robot_event_contract_validator(
    *,
    event: dict[str, Any],
    resource_jid: str,
    **_: Any,
) -> str | None:
    operation_family = str(event.get("operation_family", "") or "").strip().lower()
    if operation_family not in {"clear", "home", "pick", "stage", "assemble", "place"}:
        return None

    event_name = str(event.get("event_name", "") or "").strip() or resource_jid or operation_family
    part_name = str(event.get("part_name", "") or "").strip()
    resource_delta = dict(event.get("expected_resource_delta") or {})
    delta_to = str(resource_delta.get("to", "") or "").strip().lower()
    part_delta = dict(event.get("expected_part_delta") or {})
    part_to = str(part_delta.get("to", "") or "").strip().lower()

    if operation_family in {"clear", "home"} and part_name:
        return (
            f"bridge event '{event_name}' is inconsistent for robot resources: "
            f"operation_family '{operation_family}' must not declare part_name"
        )
    if operation_family == "pick":
        if part_to and part_to != "in_gripper":
            return (
                f"bridge event '{event_name}' is inconsistent for robot resources: "
                "'pick' requires expected_part_delta.to='in_gripper'"
            )
        if delta_to and delta_to != "picked":
            return (
                f"bridge event '{event_name}' is inconsistent for robot resources: "
                "'pick' requires expected_resource_delta.to='picked'"
            )
    if operation_family == "stage":
        if part_to and part_to != "ready":
            return (
                f"bridge event '{event_name}' is inconsistent for robot resources: "
                "'stage' requires expected_part_delta.to='ready'"
            )
        if delta_to == "picked":
            return (
                f"bridge event '{event_name}' is inconsistent for robot resources: "
                "'stage' cannot keep the resource in state 'picked'"
            )
    if operation_family == "assemble":
        if part_to and part_to != "assembled":
            return (
                f"bridge event '{event_name}' is inconsistent for robot resources: "
                "'assemble' requires expected_part_delta.to='assembled'"
            )
        if delta_to == "picked":
            return (
                f"bridge event '{event_name}' is inconsistent for robot resources: "
                "'assemble' cannot keep the resource in state 'picked'"
            )
    if operation_family == "place":
        if part_to == "in_gripper":
            return (
                f"bridge event '{event_name}' is inconsistent for robot resources: "
                "'place' cannot leave the part in_gripper"
            )
        if delta_to == "picked":
            return (
                f"bridge event '{event_name}' is inconsistent for robot resources: "
                "'place' cannot keep the resource in state 'picked'"
            )
    return None


def _robot_state_projector(
    *,
    resource_entry: dict[str, Any],
    semantic_kind: str,
    part_name: str | None,
    part_delta: dict[str, Any],
    projected_part_states: dict[str, Any],
    projected_part_locations: dict[str, Any],
    resource_jid: str,
) -> None:
    if not part_name:
        return
    part_to = str(part_delta.get("to", "") or "").strip().lower()
    if semantic_kind == "pick":
        resource_entry["resource_facets"] = dict(resource_entry.get("resource_facets") or {})
        manipulator = dict(resource_entry["resource_facets"].get("manipulator") or {})
        manipulator["held_part"] = part_name
        resource_entry["resource_facets"]["manipulator"] = manipulator
        resource_entry["held_part"] = part_name
        resource_entry.setdefault("occupancy", {})["held_parts"] = [part_name]
        projected_part_locations[part_name] = _robot_carried_location(resource_jid)
        if not part_to:
            projected_part_states[part_name] = projected_part_states.get(part_name) or "in_gripper"
    elif semantic_kind == "pick_place":
        target_location = resource_entry.get("_target_location")
        resource_entry["resource_facets"] = dict(resource_entry.get("resource_facets") or {})
        manipulator = dict(resource_entry["resource_facets"].get("manipulator") or {})
        if not part_to and not target_location:
            manipulator["held_part"] = part_name
            resource_entry.setdefault("occupancy", {})["held_parts"] = [part_name]
            projected_part_locations[part_name] = _robot_carried_location(resource_jid)
            projected_part_states[part_name] = projected_part_states.get(part_name) or "in_gripper"
            resource_entry["held_part"] = part_name
        else:
            manipulator["held_part"] = None
            resource_entry.setdefault("occupancy", {})["held_parts"] = []
            resource_entry.pop("held_part", None)
        resource_entry["resource_facets"]["manipulator"] = manipulator
    elif semantic_kind in {"stage", "place", "assemble"}:
        resource_entry["resource_facets"] = dict(resource_entry.get("resource_facets") or {})
        manipulator = dict(resource_entry["resource_facets"].get("manipulator") or {})
        manipulator["held_part"] = None
        resource_entry["resource_facets"]["manipulator"] = manipulator
        resource_entry.pop("held_part", None)
        resource_entry.setdefault("occupancy", {})["held_parts"] = []


def _robot_event_state_validator(
    *,
    event: dict[str, Any],
    resource_jid: str,
    before_resource: dict[str, Any],
    after_resource: dict[str, Any],
    profile: ResourceProfile,
    **_: Any,
) -> str | None:
    part_name = str(event.get("part_name", "") or "").strip()
    if not part_name:
        return None

    semantic_kind = str(event.get("operation_family", "") or "").strip().lower()
    carried_before = str(
        resource_snapshot_carried_entity(before_resource, profile=profile) or ""
    ).strip()
    carried_after = str(
        resource_snapshot_carried_entity(after_resource, profile=profile) or ""
    ).strip()

    if semantic_kind in {"stage", "place", "assemble"}:
        if carried_before != part_name:
            if not carried_before:
                return (
                    f"bridge event '{str(event.get('event_name', '') or resource_jid).strip()}' "
                    f"is inconsistent: {resource_jid} must be carrying '{part_name}' "
                    f"before it starts"
                )
            return (
                f"bridge event '{str(event.get('event_name', '') or resource_jid).strip()}' "
                f"is inconsistent: {resource_jid} must be carrying '{part_name}' before it "
                f"starts, but projected carried entity is '{carried_before}'"
            )
        if carried_after:
            return (
                f"bridge event '{str(event.get('event_name', '') or resource_jid).strip()}' "
                f"is inconsistent: it claims '{part_name}' is released, but projected "
                f"carried entity after the event is '{carried_after}'"
            )

    if semantic_kind == "pick" and carried_after != part_name:
        return (
            f"bridge event '{str(event.get('event_name', '') or resource_jid).strip()}' "
            f"is inconsistent: it claims '{part_name}' is acquired, but projected carried "
            f"entity after the event is {carried_after!r}"
        )

    return None


def _normalized_xyz_pose(value: Any) -> dict[str, float] | None:
    if not isinstance(value, dict):
        return None
    try:
        return {
            "x": float(value["x"]),
            "y": float(value["y"]),
            "z": float(value["z"]),
        }
    except (KeyError, TypeError, ValueError):
        return None


def _normalized_orientation(value: Any) -> dict[str, float] | None:
    if not isinstance(value, dict):
        return None
    keys = ("qx", "qy", "qz", "qw")
    if not all(key in value for key in keys):
        return None
    try:
        return {key: float(value[key]) for key in keys}
    except (TypeError, ValueError):
        return None


def _normalize_detected_item_output(item: Any, *, fallback_name: str = "") -> dict[str, Any] | None:
    if not isinstance(item, dict):
        return None
    pose = _normalized_xyz_pose(item)
    if pose is None:
        return None
    output: dict[str, Any] = {
        "part_name": str(item.get("part_name") or fallback_name or "").strip(),
        "x": pose["x"],
        "y": pose["y"],
        "z": pose["z"],
        "pose": dict(pose),
    }
    orientation = _normalized_orientation(item)
    if orientation is not None:
        output.update(orientation)
        output["orientation"] = deepcopy(orientation)
        output["pose"].update(orientation)
    model_name = str(item.get("model_name") or "").strip()
    if model_name:
        output["model_name"] = model_name
    return output


def _normalize_pose_output(pose: Any) -> dict[str, Any] | None:
    xyz = _normalized_xyz_pose(pose)
    if xyz is None:
        return None
    output: dict[str, Any] = {
        "x": xyz["x"],
        "y": xyz["y"],
        "z": xyz["z"],
        "pose": dict(xyz),
    }
    orientation = _normalized_orientation(pose)
    if orientation is not None:
        output.update(orientation)
        output["orientation"] = deepcopy(orientation)
        output["pose"].update(orientation)
    return output


def _preview_detect_output(
    params: dict[str, Any],
    _snapshot: dict[str, Any],
    grounding_context: dict[str, Any],
) -> tuple[dict[str, Any] | None, str | None]:
    item_name = str(params.get("part_name") or "").strip()
    if not item_name:
        return None, "store_as requires params.part_name"
    item_info = (grounding_context or {}).get("parts", {}).get(item_name, {})
    observed_pose = _normalized_xyz_pose((item_info or {}).get("observed_pose")) or {
        "x": 0.0,
        "y": 0.0,
        "z": 0.0,
    }
    output: dict[str, Any] = {
        "part_name": item_name,
        "x": observed_pose["x"],
        "y": observed_pose["y"],
        "z": observed_pose["z"],
        "pose": dict(observed_pose),
    }
    orientation = _normalized_orientation((item_info or {}).get("observed_pose"))
    if orientation is not None:
        output.update(orientation)
        output["orientation"] = deepcopy(orientation)
        output["pose"].update(orientation)
    model_name = str(((item_info or {}).get("target") or {}).get("model_name") or "").strip()
    if model_name:
        output["model_name"] = model_name
    return output, None


def _preview_pose_output(
    _params: dict[str, Any],
    snapshot: dict[str, Any],
    grounding_context: dict[str, Any],
) -> tuple[dict[str, Any] | None, str | None]:
    raw_pose = (
        ((grounding_context or {}).get("resource") or {}).get("current_pose")
        or dict((snapshot.get("resource_facets") or {}).get("manipulator") or {}).get("current_pose")
        or snapshot.get("current_pose")
        or {}
    )
    pose = _normalize_pose_output(raw_pose)
    if pose is None:
        pose = {
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
            "qx": 0.0,
            "qy": 0.0,
            "qz": 0.0,
            "qw": 1.0,
            "orientation": {"qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0},
            "pose": {
                "x": 0.0,
                "y": 0.0,
                "z": 0.0,
                "qx": 0.0,
                "qy": 0.0,
                "qz": 0.0,
                "qw": 1.0,
            },
        }
    elif "qx" not in pose:
        orientation = {"qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0}
        pose.update(orientation)
        pose["orientation"] = deepcopy(orientation)
        pose["pose"].update(orientation)
    return pose, None


def _preview_pick_targets_output(
    params: dict[str, Any],
    snapshot: dict[str, Any],
    grounding_context: dict[str, Any],
) -> tuple[dict[str, Any] | None, str | None]:
    item_name = str(params.get("part_name") or "").strip()
    if not item_name:
        return None, "store_as requires params.part_name"
    product_geometry = params.get("product_geometry")
    if product_geometry is not None and not isinstance(product_geometry, dict):
        return None, "params.product_geometry must be an object when provided"
    item_info = (grounding_context or {}).get("parts", {}).get(item_name, {})
    observed_pose = _normalized_xyz_pose((item_info or {}).get("observed_pose")) or {
        "x": 0.0,
        "y": 0.0,
        "z": 0.0,
    }
    current_pose = (
        _normalized_xyz_pose((((grounding_context or {}).get("resource") or {}).get("current_pose")))
        or _normalized_xyz_pose(dict((snapshot.get("resource_facets") or {}).get("manipulator") or {}).get("current_pose"))
        or {"x": 0.0, "y": 0.0, "z": 0.0}
    )
    geometry = dict(product_geometry or {})
    board_center = dict(geometry.get("board_center") or {})
    board_center_z = float(board_center.get("z", 1.02) or 1.02)
    part_height = float(geometry.get("part_height_m", 0.08) or 0.08)
    tcp_offset_z = -0.17
    pick_bias = max(0.003, min(0.02, part_height * 0.25))
    min_tcp_z = float(params.get("min_pick_tcp_z_override_m") or 1.07)
    pick_tcp_z = max(float(observed_pose["z"]) + pick_bias, min_tcp_z)
    pick_z = pick_tcp_z - tcp_offset_z
    approach_height = float(params.get("approach_height_override_m", 0.2) or 0.2)
    travel_candidates = [
        float(observed_pose["z"]) + approach_height,
        board_center_z + approach_height,
        pick_z + 0.05,
    ]
    if not bool(params.get("ignore_current_height_for_travel_z", False)):
        travel_candidates.append(float(current_pose["z"]))
    travel_z = max(travel_candidates)
    return {
        "part_name": item_name,
        "model_name": str(geometry.get("model_name") or ""),
        "tx": float(observed_pose["x"]),
        "ty": float(observed_pose["y"]),
        "tz": float(observed_pose["z"]),
        "pick_z": pick_z,
        "travel_z": travel_z,
        "approach_pose": {
            "x": float(observed_pose["x"]),
            "y": float(observed_pose["y"]),
            "z": travel_z,
        },
        "target_pose": {
            "x": float(observed_pose["x"]),
            "y": float(observed_pose["y"]),
            "z": pick_z,
        },
        "part_height": part_height,
        "tcp_offset_z": tcp_offset_z,
        "pick_tcp_z": pick_tcp_z,
        "start_x": float(current_pose["x"]),
        "start_y": float(current_pose["y"]),
        "start_z": float(current_pose["z"]),
    }, None


def _preview_place_targets_output(
    params: dict[str, Any],
    _snapshot: dict[str, Any],
    grounding_context: dict[str, Any],
) -> tuple[dict[str, Any] | None, str | None]:
    pick_ctx = params.get("pick_ctx")
    if pick_ctx is not None and not isinstance(pick_ctx, dict):
        return None, "params.pick_ctx must be an object when provided"
    product_geometry = params.get("product_geometry")
    if product_geometry is not None and not isinstance(product_geometry, dict):
        return None, "params.product_geometry must be an object when provided"
    normalized_pick_ctx = dict(pick_ctx or {})
    item_name = str(params.get("part_name") or normalized_pick_ctx.get("part_name") or "").strip()
    if not item_name:
        return None, "store_as requires params.part_name or params.pick_ctx.part_name"
    geometry = dict(product_geometry or {})
    board_center = dict(geometry.get("board_center") or {})
    slot_xy = geometry.get("slot_xy")
    if isinstance(slot_xy, (list, tuple)) and len(slot_xy) >= 2:
        slot_x = float(board_center.get("x", 0.0) or 0.0) + float(slot_xy[0] or 0.0)
        slot_y = float(board_center.get("y", 0.0) or 0.0) + float(slot_xy[1] or 0.0)
    else:
        slot_x = float(board_center.get("x", normalized_pick_ctx.get("tx", 0.0)) or 0.0)
        slot_y = float(board_center.get("y", normalized_pick_ctx.get("ty", 0.0)) or 0.0)
    board_top_z = float(geometry.get("slot_floor_z_m", board_center.get("z", 1.025)) or 1.025)
    part_height = float(geometry.get("part_height_m", normalized_pick_ctx.get("part_height", 0.08)) or 0.08)
    if normalized_pick_ctx:
        grasp_tcp_to_part_origin_z = float(normalized_pick_ctx.get("pick_tcp_z", 0.0) or 0.0) - float(
            normalized_pick_ctx.get("tz", 0.0) or 0.0
        )
        tcp_offset_z = float(normalized_pick_ctx.get("tcp_offset_z", -0.17) or -0.17)
    else:
        grasp_tcp_to_part_origin_z = max(0.003, min(0.02, part_height * 0.25))
        tcp_offset_z = -0.17
    place_gap = -0.0125
    place_part_origin_z = board_top_z + (part_height * 0.5) + place_gap
    place_tcp_z = place_part_origin_z + grasp_tcp_to_part_origin_z
    place_z = place_tcp_z - tcp_offset_z + float(params.get("z_adjustment_m", 0.0) or 0.0)
    return {
        "part_name": item_name,
        "slot_x": slot_x,
        "slot_y": slot_y,
        "board_top_z": board_top_z,
        "place_z": place_z,
        "place_tcp_z": place_tcp_z,
        "approach_pose": {"x": slot_x, "y": slot_y, "z": place_z + 0.05},
        "target_pose": {"x": slot_x, "y": slot_y, "z": place_z},
        "part_height": part_height,
        "tcp_offset_z": tcp_offset_z,
        "grasp_tcp_to_part_origin_z": grasp_tcp_to_part_origin_z,
        "model_name": str(geometry.get("model_name") or normalized_pick_ctx.get("model_name") or ""),
    }, None


def _extract_detect_output(
    params: dict[str, Any],
    step_result: dict[str, Any],
) -> tuple[dict[str, Any] | None, str | None]:
    item_name = str(params.get("part_name") or "").strip()
    if not item_name:
        return None, "store_as requires params.part_name"
    items = step_result.get("data")
    if not isinstance(items, list):
        return None, "store_as expected list result data"
    if len(items) != 1:
        return None, f"store_as expected exactly one result for '{item_name}', found {len(items)}"
    output = _normalize_detected_item_output(items[0], fallback_name=item_name)
    if output is None:
        return None, "store_as result did not contain x/y/z fields"
    return output, None


def _extract_pose_output(
    _params: dict[str, Any],
    step_result: dict[str, Any],
) -> tuple[dict[str, Any] | None, str | None]:
    output = _normalize_pose_output(step_result.get("pose"))
    if output is None:
        return None, "store_as result did not contain pose x/y/z"
    return output, None


def _extract_pick_targets_output(
    _params: dict[str, Any],
    step_result: dict[str, Any],
) -> tuple[dict[str, Any] | None, str | None]:
    if not isinstance(step_result, dict) or not step_result.get("success"):
        return None, "store_as expected successful dict result"
    required_keys = {
        "part_name",
        "tx",
        "ty",
        "tz",
        "pick_z",
        "travel_z",
        "part_height",
        "tcp_offset_z",
        "pick_tcp_z",
        "start_x",
        "start_y",
        "start_z",
    }
    if not required_keys <= set(step_result.keys()):
        return None, "store_as result was missing expected keys"
    return {
        "part_name": str(step_result.get("part_name") or ""),
        "model_name": str(step_result.get("model_name") or ""),
        "tx": float(step_result["tx"]),
        "ty": float(step_result["ty"]),
        "tz": float(step_result["tz"]),
        "pick_z": float(step_result["pick_z"]),
        "travel_z": float(step_result["travel_z"]),
        "approach_pose": {
            "x": float((step_result.get("approach_pose") or {}).get("x", step_result["tx"])),
            "y": float((step_result.get("approach_pose") or {}).get("y", step_result["ty"])),
            "z": float((step_result.get("approach_pose") or {}).get("z", step_result["travel_z"])),
        },
        "target_pose": {
            "x": float((step_result.get("target_pose") or {}).get("x", step_result["tx"])),
            "y": float((step_result.get("target_pose") or {}).get("y", step_result["ty"])),
            "z": float((step_result.get("target_pose") or {}).get("z", step_result["pick_z"])),
        },
        "part_height": float(step_result["part_height"]),
        "tcp_offset_z": float(step_result["tcp_offset_z"]),
        "pick_tcp_z": float(step_result["pick_tcp_z"]),
        "start_x": float(step_result["start_x"]),
        "start_y": float(step_result["start_y"]),
        "start_z": float(step_result["start_z"]),
    }, None


def _extract_place_targets_output(
    _params: dict[str, Any],
    step_result: dict[str, Any],
) -> tuple[dict[str, Any] | None, str | None]:
    if not isinstance(step_result, dict) or not step_result.get("success"):
        return None, "store_as expected successful dict result"
    required_keys = {
        "slot_x",
        "slot_y",
        "board_top_z",
        "place_z",
        "place_tcp_z",
        "part_height",
        "tcp_offset_z",
        "grasp_tcp_to_part_origin_z",
    }
    if not required_keys <= set(step_result.keys()):
        return None, "store_as result was missing expected keys"
    return {
        "part_name": str(step_result.get("part_name") or ""),
        "slot_x": float(step_result["slot_x"]),
        "slot_y": float(step_result["slot_y"]),
        "board_top_z": float(step_result["board_top_z"]),
        "place_z": float(step_result["place_z"]),
        "place_tcp_z": float(step_result["place_tcp_z"]),
        "approach_pose": {
            "x": float((step_result.get("approach_pose") or {}).get("x", step_result["slot_x"])),
            "y": float((step_result.get("approach_pose") or {}).get("y", step_result["slot_y"])),
            "z": float((step_result.get("approach_pose") or {}).get("z", step_result["place_z"] + 0.05)),
        },
        "target_pose": {
            "x": float((step_result.get("target_pose") or {}).get("x", step_result["slot_x"])),
            "y": float((step_result.get("target_pose") or {}).get("y", step_result["slot_y"])),
            "z": float((step_result.get("target_pose") or {}).get("z", step_result["place_z"])),
        },
        "part_height": float(step_result["part_height"]),
        "tcp_offset_z": float(step_result["tcp_offset_z"]),
        "grasp_tcp_to_part_origin_z": float(step_result["grasp_tcp_to_part_origin_z"]),
        "model_name": str(step_result.get("model_name") or ""),
    }, None


def _robot_primitive_owner(agent: Any) -> Any | None:
    controller = getattr(agent, "_controller", None)
    if controller is not None:
        return controller
    scope_name = ""
    try:
        scope_name = str(agent._robot_scope_name()).strip().lower()
    except Exception:
        scope_name = str(getattr(agent, "agent_name", "")).split("@", 1)[0].lower()
    if scope_name.startswith("ur5e"):
        return UR5eController
    if scope_name.startswith("xarm6"):
        return XArm6Controller
    return None


def _robot_snapshot_builder(agent: Any) -> dict[str, Any]:
    current_pose = None
    controller = getattr(agent, "_controller", None)
    if controller is not None and str(getattr(agent, "execution_mode", "")).strip().lower() != "dry_run":
        try:
            pose_result = controller.get_current_pose()
        except Exception:
            pose_result = None
        if isinstance(pose_result, dict) and pose_result.get("success"):
            pose = dict(pose_result.get("pose") or {})
            if {"x", "y", "z"} <= set(pose.keys()):
                current_pose = {
                    "x": float(pose["x"]),
                    "y": float(pose["y"]),
                    "z": float(pose["z"]),
                }
    if current_pose is None and getattr(agent, "_bridge_pose_ref", None) is None:
        position = getattr(agent, "_position", None)
        if isinstance(position, dict) and {"x", "y", "z"} <= set(position.keys()):
            current_pose = {
                "x": float(position["x"]),
                "y": float(position["y"]),
                "z": float(position["z"]),
            }
    return {
        "resource_type": "robot",
        "current_state": str(getattr(agent, "_current_state", "") or "").strip() or "idle",
        "held_part": getattr(agent, "_held_part", None),
        "gripper_state": str(getattr(agent, "_gripper_state", "") or "").strip() or "unknown",
        "current_pose": current_pose,
        "current_pose_ref": getattr(agent, "_bridge_pose_ref", None),
        "named_poses": sorted((getattr(agent, "named_positions", {}) or {}).keys()),
    }


def _bridge_ref(compiler: Any, path: str) -> dict[str, str]:
    return compiler._bridge_ref(path)


def _part_target_info(prepared_bridge_request: dict[str, Any], *, part_name: str) -> dict[str, Any]:
    parts = dict((prepared_bridge_request.get("grounding_context") or {}).get("parts") or {})
    return dict((parts.get(str(part_name or "").strip()) or {}).get("target") or {})


def _part_pick_geometry(prepared_bridge_request: dict[str, Any], *, part_name: str) -> dict[str, Any]:
    target = _part_target_info(prepared_bridge_request, part_name=part_name)
    slot_pose = dict(target.get("slot_pose") or {})
    board_top_z = target.get("board_top_z")
    try:
        board_z = float(board_top_z if board_top_z is not None else slot_pose.get("z", 1.0))
    except (TypeError, ValueError):
        board_z = 1.0
    geometry: dict[str, Any] = {
        "board_center": {"x": 0.0, "y": 0.0, "z": board_z},
        "slot_floor_z_m": board_z,
    }
    try:
        if slot_pose.get("x") is not None and slot_pose.get("y") is not None:
            geometry["slot_xy"] = [float(slot_pose.get("x")), float(slot_pose.get("y"))]
    except (TypeError, ValueError):
        pass
    try:
        if target.get("part_height") is not None:
            geometry["part_height_m"] = float(target.get("part_height"))
    except (TypeError, ValueError):
        pass
    model_name = str(target.get("model_name") or "").strip()
    if model_name:
        geometry["model_name"] = model_name
    return geometry


def _named_pose_available(compiler: Any, *, resource_jid: str, pose_name: str) -> bool:
    resource = compiler._resource_by_jid(resource_jid)
    named_positions = getattr(resource, "named_positions", {}) if resource is not None else {}
    return bool(isinstance(named_positions, dict) and str(pose_name) in named_positions)


def _primitive_allows_start_state(
    prepared_bridge_request: dict[str, Any] | None,
    *,
    resource_jid: str,
    primitive_name: str,
    start_state: str,
) -> bool:
    bridge_resources = dict((prepared_bridge_request or {}).get("bridge_resources") or {})
    primitive_catalog = list(dict(bridge_resources.get(resource_jid) or {}).get("primitive_catalog") or [])
    for entry in primitive_catalog:
        if not isinstance(entry, dict):
            continue
        if str(entry.get("name") or "").strip() != primitive_name:
            continue
        current_state = dict(dict(entry.get("preconditions") or {}).get("current_state") or {})
        equals = str(current_state.get("equals", "") or "").strip()
        not_equals = str(current_state.get("not_equals", "") or "").strip()
        normalized_start = str(start_state or "").strip()
        if equals and normalized_start and normalized_start != equals:
            return False
        if not_equals and normalized_start and normalized_start == not_equals:
            return False
        return True
    return True


def _home_steps(compiler: Any, *, resource_jid: str, speed: float = 0.8) -> list[dict[str, Any]]:
    if _named_pose_available(compiler, resource_jid=resource_jid, pose_name="home"):
        return [{"primitive": "move_to_named_pose", "params": {"pose_name": "home", "speed": speed}}]
    return []


def _orientation_params(compiler: Any, *, alias: str) -> dict[str, Any]:
    return {
        "qx": _bridge_ref(compiler, f"/step_outputs/{alias}/pose/qx"),
        "qy": _bridge_ref(compiler, f"/step_outputs/{alias}/pose/qy"),
        "qz": _bridge_ref(compiler, f"/step_outputs/{alias}/pose/qz"),
        "qw": _bridge_ref(compiler, f"/step_outputs/{alias}/pose/qw"),
    }


def _stage_destination(prepared_bridge_request: dict[str, Any], *, event: dict[str, Any], resource_jid: str) -> str:
    item_name = str(event.get("part_name", "")).strip()
    part_delta = dict(event.get("expected_part_delta") or {})
    if str(part_delta.get("location_to", "")).strip():
        return str(part_delta.get("location_to")).strip()
    part_context = dict(
        ((prepared_bridge_request.get("grounding_context") or {}).get("parts") or {}).get(item_name) or {}
    )
    for key in ("origin_resource_location", "last_known_location", "location"):
        token = str(part_context.get(key) or "").strip()
        if token and not token.endswith("_gripper"):
            return token
    bridge_resources = prepared_bridge_request.get("bridge_resources") or {}
    resource_entry = dict((bridge_resources.get(resource_jid) or {}).get("bridge_snapshot") or {})
    manipulator = dict((resource_entry.get("resource_facets") or {}).get("manipulator") or {})
    token = str(manipulator.get("current_pose_ref") or resource_entry.get("current_pose_ref") or "").strip()
    if token:
        return token
    return f"{resource_jid}_staging"


def _transition_from_event(
    *,
    event: dict[str, Any],
    start_state: str,
    destination_location: str = "",
) -> tuple[str, dict[str, Any] | None]:
    resource_delta = dict(event.get("expected_resource_delta") or {})
    out_state = str(resource_delta.get("to", "") or "").strip() or start_state
    item_name = str(event.get("part_name", "") or "").strip()
    part_delta = dict(event.get("expected_part_delta") or {})
    if not item_name:
        return out_state, None
    part_to = str(part_delta.get("to", "") or "").strip()
    if not part_to:
        return out_state, None
    location_to = str(part_delta.get("location_to", "") or destination_location or "").strip()
    completed: dict[str, Any] = {"state": part_to}
    if part_to == "in_gripper":
        completed["location_template"] = "{resource_jid}_gripper"
    elif location_to:
        completed["location_param"] = "destination_location"
    return out_state, {"completed": completed}


def _compile_clear_macro(
    compiler: Any,
    _prepared_bridge_request: dict[str, Any] | None = None,
    *,
    event: dict[str, Any],
    resource_jid: str,
    start_state: str,
    primitive_name: str = "",
) -> dict[str, Any]:
    steps: list[dict[str, Any]] = []
    if _primitive_allows_start_state(
        _prepared_bridge_request,
        resource_jid=resource_jid,
        primitive_name="move_to_named_pose",
        start_state=start_state,
    ):
        steps = _home_steps(compiler, resource_jid=resource_jid)
    if not steps:
        steps = [
            {"primitive": "get_current_pose", "params": {}, "store_as": "current_pose"},
            {
                "primitive": "move_cartesian",
                "params": {
                    "x": _bridge_ref(compiler, "/step_outputs/current_pose/pose/x"),
                    "y": _bridge_ref(compiler, "/step_outputs/current_pose/pose/y"),
                    "z": _bridge_ref(compiler, "/step_outputs/current_pose/pose/z"),
                    "speed": 0.8,
                },
            },
        ]
    out_state = str((event.get("expected_resource_delta") or {}).get("to", "") or "idle").strip() or "idle"
    return {
        "resource_jid": resource_jid,
        "macro_name": str(event.get("event_name") or "clear_resource").strip() or "clear_resource",
        "description": str(event.get("rationale") or "Move the resource to a safe clear state.").strip(),
        "rationale": str(event.get("rationale") or "").strip(),
        "expected_start_state": start_state,
        "part_name": "",
        "task_params": {},
        "task_metadata": {
            "in_state": start_state,
            "out_state": out_state,
            "required_context_keys": [],
            "context_mapping": {},
            "part_transition": None,
        },
        "primitive_steps": steps,
    }


def _compile_pick_macro(
    compiler: Any,
    prepared_bridge_request: dict[str, Any],
    *,
    event: dict[str, Any],
    resource_jid: str,
    start_state: str,
    primitive_name: str = "",
) -> dict[str, Any]:
    part_name = str(event.get("part_name", "")).strip()
    geometry = _part_pick_geometry(prepared_bridge_request, part_name=part_name)
    out_state, part_transition = _transition_from_event(
        event=event,
        start_state=start_state,
    )
    return {
        "resource_jid": resource_jid,
        "macro_name": str(event.get("event_name") or f"pick_{part_name.lower()}").strip(),
        "description": str(event.get("rationale") or f"Acquire {part_name}.").strip(),
        "rationale": str(event.get("rationale") or "").strip(),
        "expected_start_state": start_state,
        "part_name": part_name,
        "task_params": {},
        "task_metadata": {
            "in_state": start_state,
            "out_state": out_state or "picked",
            "required_context_keys": [],
            "context_mapping": {},
            "part_transition": part_transition,
        },
        "primitive_steps": [
            {"primitive": "detect_parts", "params": {"part_name": part_name}, "store_as": "detected_part"},
            {"primitive": "get_current_pose", "params": {}, "store_as": "pre_pick_pose"},
            {
                "primitive": "compute_pick_targets",
                "params": {"part_name": part_name, "product_geometry": geometry},
                "store_as": "part_pick_targets",
            },
            {
                "primitive": "move_cartesian",
                "params": {
                    "x": _bridge_ref(compiler, "/step_outputs/detected_part/pose/x"),
                    "y": _bridge_ref(compiler, "/step_outputs/detected_part/pose/y"),
                    "z": _bridge_ref(compiler, "/step_outputs/part_pick_targets/travel_z"),
                    "speed": 1.2,
                },
            },
            {
                "primitive": "move_pose",
                "params": {
                    "x": _bridge_ref(compiler, "/step_outputs/detected_part/pose/x"),
                    "y": _bridge_ref(compiler, "/step_outputs/detected_part/pose/y"),
                    "z": _bridge_ref(compiler, "/step_outputs/part_pick_targets/pick_z"),
                    **_orientation_params(compiler, alias="pre_pick_pose"),
                    "speed": 0.8,
                },
            },
            {"primitive": "close_gripper", "params": {}},
            {
                "primitive": "attach_part",
                "params": {
                    "model_name": _bridge_ref(compiler, f"/parts/{part_name}/target/model_name"),
                    "part_name": part_name,
                },
            },
            {"primitive": "move_relative", "params": {"dx": 0.0, "dy": 0.0, "dz": 0.05, "speed": 0.8}},
        ],
    }


def _compile_release_macro(
    compiler: Any,
    prepared_bridge_request: dict[str, Any],
    *,
    event: dict[str, Any],
    resource_jid: str,
    start_state: str,
    primitive_name: str = "",
) -> dict[str, Any]:
    part_name = str(event.get("part_name", "")).strip()
    destination_location = _stage_destination(
        prepared_bridge_request,
        event=event,
        resource_jid=resource_jid,
    )
    out_state, part_transition = _transition_from_event(
        event=event,
        start_state=start_state,
        destination_location=destination_location,
    )
    return {
        "resource_jid": resource_jid,
        "macro_name": str(event.get("event_name") or f"stage_{part_name.lower()}").strip(),
        "description": str(event.get("rationale") or f"Release {part_name} at a safe intermediate location.").strip(),
        "rationale": str(event.get("rationale") or "").strip(),
        "expected_start_state": start_state,
        "part_name": part_name,
        "task_params": {"destination_location": destination_location},
        "task_metadata": {
            "in_state": start_state,
            "out_state": out_state or "idle",
            "required_context_keys": ["destination_location"] if destination_location else [],
            "context_mapping": {"location_param": "destination_location"} if destination_location else {},
            "part_transition": part_transition,
        },
        "primitive_steps": [
            {"primitive": "get_current_pose", "params": {}, "store_as": "pre_release_pose"},
            {"primitive": "move_relative", "params": {"dx": 0.0, "dy": 0.0, "dz": -0.03, "speed": 0.6}},
            {"primitive": "open_gripper", "params": {}},
            {
                "primitive": "detach_part",
                "params": {
                    "model_name": _bridge_ref(compiler, f"/parts/{part_name}/target/model_name"),
                    "assume_released_if_open": True,
                },
            },
            {"primitive": "move_relative", "params": {"dx": 0.0, "dy": 0.0, "dz": 0.08, "speed": 0.8}},
        ] + _home_steps(compiler, resource_jid=resource_jid, speed=0.8),
    }


def _compile_place_macro(
    compiler: Any,
    prepared_bridge_request: dict[str, Any],
    *,
    event: dict[str, Any],
    resource_jid: str,
    start_state: str,
    primitive_name: str = "",
) -> dict[str, Any]:
    part_name = str(event.get("part_name", "")).strip()
    part_delta = dict(event.get("expected_part_delta") or {})
    target_info = _part_target_info(prepared_bridge_request, part_name=part_name)
    destination_location = str(part_delta.get("location_to", "")).strip() or str(target_info.get("location") or "").strip()
    geometry = _part_pick_geometry(prepared_bridge_request, part_name=part_name)
    out_state, part_transition = _transition_from_event(
        event=event,
        start_state=start_state,
        destination_location=destination_location,
    )
    return {
        "resource_jid": resource_jid,
        "macro_name": str(event.get("event_name") or f"place_{part_name.lower()}").strip(),
        "description": str(event.get("rationale") or f"Place {part_name} at its destination.").strip(),
        "rationale": str(event.get("rationale") or "").strip(),
        "expected_start_state": start_state,
        "part_name": part_name,
        "task_params": {"destination_location": destination_location} if destination_location else {},
        "task_metadata": {
            "in_state": start_state,
            "out_state": out_state or "idle",
            "required_context_keys": ["destination_location"] if destination_location else [],
            "context_mapping": {"location_param": "destination_location"} if destination_location else {},
            "part_transition": part_transition,
        },
        "primitive_steps": [
            {"primitive": "get_current_pose", "params": {}, "store_as": "pre_place_pose"},
            {
                "primitive": "compute_place_targets",
                "params": {"part_name": part_name, "product_geometry": geometry},
                "store_as": "part_place_targets",
            },
            {
                "primitive": "move_cartesian",
                "params": {
                    "x": _bridge_ref(compiler, "/step_outputs/part_place_targets/slot_x"),
                    "y": _bridge_ref(compiler, "/step_outputs/part_place_targets/slot_y"),
                    "z": _bridge_ref(compiler, "/step_outputs/pre_place_pose/pose/z"),
                    "speed": 1.2,
                },
            },
            {
                "primitive": "move_pose",
                "params": {
                    "x": _bridge_ref(compiler, "/step_outputs/part_place_targets/slot_x"),
                    "y": _bridge_ref(compiler, "/step_outputs/part_place_targets/slot_y"),
                    "z": _bridge_ref(compiler, "/step_outputs/part_place_targets/place_z"),
                    **_orientation_params(compiler, alias="pre_place_pose"),
                    "speed": 0.8,
                },
            },
            {"primitive": "open_gripper", "params": {}},
            {
                "primitive": "detach_part",
                "params": {
                    "model_name": _bridge_ref(compiler, f"/parts/{part_name}/target/model_name"),
                    "assume_released_if_open": True,
                },
            },
            {"primitive": "move_relative", "params": {"dx": 0.0, "dy": 0.0, "dz": 0.08, "speed": 1.0}},
        ] + _home_steps(compiler, resource_jid=resource_jid, speed=0.8),
    }


def _compile_pick_place_macro(
    compiler: Any,
    prepared_bridge_request: dict[str, Any],
    *,
    event: dict[str, Any],
    resource_jid: str,
    start_state: str,
    primitive_name: str = "",
) -> dict[str, Any]:
    part_name = str(event.get("part_name", "")).strip()
    part_delta = dict(event.get("expected_part_delta") or {})
    target_info = _part_target_info(prepared_bridge_request, part_name=part_name)
    destination_location = str(part_delta.get("location_to", "")).strip() or str(target_info.get("location") or "").strip()
    geometry = _part_pick_geometry(prepared_bridge_request, part_name=part_name)
    out_state, part_transition = _transition_from_event(
        event=event,
        start_state=start_state,
        destination_location=destination_location,
    )
    return {
        "resource_jid": resource_jid,
        "macro_name": str(event.get("event_name") or f"recover_{part_name.lower()}").strip(),
        "description": str(event.get("rationale") or f"Recover and place {part_name}.").strip(),
        "rationale": str(event.get("rationale") or "").strip(),
        "expected_start_state": start_state,
        "part_name": part_name,
        "task_params": {"destination_location": destination_location} if destination_location else {},
        "task_metadata": {
            "in_state": start_state,
            "out_state": out_state or "idle",
            "required_context_keys": ["destination_location"] if destination_location else [],
            "context_mapping": {"location_param": "destination_location"} if destination_location else {},
            "part_transition": part_transition,
        },
        "primitive_steps": [
            {"primitive": "detect_parts", "params": {"part_name": part_name}, "store_as": "detected_part"},
            {"primitive": "get_current_pose", "params": {}, "store_as": "pre_pick_pose"},
            {
                "primitive": "compute_pick_targets",
                "params": {"part_name": part_name, "product_geometry": geometry},
                "store_as": "part_pick_targets",
            },
            {
                "primitive": "move_cartesian",
                "params": {
                    "x": _bridge_ref(compiler, "/step_outputs/detected_part/pose/x"),
                    "y": _bridge_ref(compiler, "/step_outputs/detected_part/pose/y"),
                    "z": _bridge_ref(compiler, "/step_outputs/part_pick_targets/travel_z"),
                    "speed": 1.2,
                },
            },
            {
                "primitive": "move_pose",
                "params": {
                    "x": _bridge_ref(compiler, "/step_outputs/detected_part/pose/x"),
                    "y": _bridge_ref(compiler, "/step_outputs/detected_part/pose/y"),
                    "z": _bridge_ref(compiler, "/step_outputs/part_pick_targets/pick_z"),
                    **_orientation_params(compiler, alias="pre_pick_pose"),
                    "speed": 0.8,
                },
            },
            {"primitive": "close_gripper", "params": {}},
            {
                "primitive": "attach_part",
                "params": {
                    "model_name": _bridge_ref(compiler, f"/parts/{part_name}/target/model_name"),
                    "part_name": part_name,
                },
            },
            {"primitive": "move_relative", "params": {"dx": 0.0, "dy": 0.0, "dz": 0.05, "speed": 0.8}},
            {"primitive": "get_current_pose", "params": {}, "store_as": "pre_place_pose"},
            {
                "primitive": "compute_place_targets",
                "params": {"part_name": part_name, "product_geometry": geometry},
                "store_as": "part_place_targets",
            },
            {
                "primitive": "move_cartesian",
                "params": {
                    "x": _bridge_ref(compiler, "/step_outputs/part_place_targets/slot_x"),
                    "y": _bridge_ref(compiler, "/step_outputs/part_place_targets/slot_y"),
                    "z": _bridge_ref(compiler, "/step_outputs/pre_place_pose/pose/z"),
                    "speed": 1.2,
                },
            },
            {
                "primitive": "move_pose",
                "params": {
                    "x": _bridge_ref(compiler, "/step_outputs/part_place_targets/slot_x"),
                    "y": _bridge_ref(compiler, "/step_outputs/part_place_targets/slot_y"),
                    "z": _bridge_ref(compiler, "/step_outputs/part_place_targets/place_z"),
                    **_orientation_params(compiler, alias="pre_place_pose"),
                    "speed": 0.8,
                },
            },
            {"primitive": "open_gripper", "params": {}},
            {
                "primitive": "detach_part",
                "params": {
                    "model_name": _bridge_ref(compiler, f"/parts/{part_name}/target/model_name"),
                    "assume_released_if_open": True,
                },
            },
            {"primitive": "move_relative", "params": {"dx": 0.0, "dy": 0.0, "dz": 0.08, "speed": 1.0}},
        ] + _home_steps(compiler, resource_jid=resource_jid, speed=0.8),
    }


_MANIPULATOR_PROMPT_ADDENDUM = dedent(
    """\
    MANIPULATOR COMPOSITION ADDENDUM:
    - This addendum applies only when the chosen resource exposes manipulator
      primitives such as detect_parts, compute_pick_targets, gripper actions,
      and attach/detach.
    - To acquire a part: observe it (detect_parts), compute approach geometry
      (compute_pick_targets), move above it, descend to grasp height, close
      gripper, attach, and lift away.
    - To place a part: compute destination geometry (compute_place_targets),
      move above the slot, descend to placement height, open gripper, detach,
      and lift away.
    - To release a part without placing it at a goal: descend to a safe
      release height, open gripper, detach, and retract.
    - Use get_current_pose before motion primitives that need orientation
      (qx, qy, qz, qw) to preserve the current end-effector orientation.
    """
).strip()


_MANIPULATOR_REPAIR_EXAMPLE = dedent(
    """\
    MANIPULATOR PICK/PLACE REPAIR EXAMPLE:
    - Use this only when the chosen resource exposes manipulator acquisition/release primitives.
    - Adapt resource JIDs, part names, geometry, and states to the current bridge.
    """
).strip()


ROBOT_PROFILE = ResourceProfile(
    resource_type="robot",
    snapshot_fields=("current_state", "held_part", "gripper_state"),
    facet_key="manipulator",
    facet_builder=_robot_facet,
    occupancy_builder=_robot_occupancy,
    snapshot_builder=_robot_snapshot_builder,
    availability_resolver=_robot_availability,
    primitive_owner_resolver=_robot_primitive_owner,
    sync_map={
        "held_part": "_held_part",
        "gripper_state": "_gripper_state",
        "current_pose": _sync_robot_pose,
        "current_pose_ref": "_bridge_pose_ref",
    },
    primitive_kind_map={
        "detect_parts": "observe",
        "get_current_pose": "observe",
        "compute_pick_targets": "pick",
        "compute_place_targets": "place",
        "move_to_named_pose": "home",
        "move_cartesian": "motion",
        "move_pose": "motion",
        "move_relative": "motion",
        "open_gripper": "release",
        "close_gripper": "pick",
        "attach_part": "pick",
        "detach_part": "place",
        "rotate_wrist": "orient",
    },
    event_family_resolver=_robot_event_family,
    event_contract_validator=_robot_event_contract_validator,
    compiler_map={
        "clear": _compile_clear_macro,
        "pick": _compile_pick_macro,
        "stage": _compile_release_macro,
        "place": _compile_place_macro,
        "assemble": _compile_place_macro,
        "pick_place": _compile_pick_place_macro,
    },
    state_projector=_robot_state_projector,
    event_state_validator=_robot_event_state_validator,
    capability_flags={"supports_manipulator_pick_place": True},
    observation_families=("part_detection", "resource_pose"),
    example_families=("generic_bridge", "manipulator_pick_place"),
    observation_output_schema_map={
        "detect_parts": {
            "part_name": "string",
            "pose": {"x": "number", "y": "number", "z": "number"},
            "orientation": {"qx": "number", "qy": "number", "qz": "number", "qw": "number"},
        },
        "get_current_pose": {
            "pose": {
                "x": "number",
                "y": "number",
                "z": "number",
                "qx": "number",
                "qy": "number",
                "qz": "number",
                "qw": "number",
            }
        },
        "compute_pick_targets": {
            "approach_pose": {"x": "number", "y": "number", "z": "number"},
            "target_pose": {"x": "number", "y": "number", "z": "number"},
        },
        "compute_place_targets": {
            "approach_pose": {"x": "number", "y": "number", "z": "number"},
            "target_pose": {"x": "number", "y": "number", "z": "number"},
        },
    },
    preview_output_map={
        "detect_parts": _preview_detect_output,
        "get_current_pose": _preview_pose_output,
        "compute_pick_targets": _preview_pick_targets_output,
        "compute_place_targets": _preview_place_targets_output,
    },
    extract_output_map={
        "detect_parts": _extract_detect_output,
        "get_current_pose": _extract_pose_output,
        "compute_pick_targets": _extract_pick_targets_output,
        "compute_place_targets": _extract_place_targets_output,
    },
    carried_entity_field="held_part",
    carried_entity_location_builder=_robot_carried_location,
    prompt_addendum=_MANIPULATOR_PROMPT_ADDENDUM,
    repair_example=_MANIPULATOR_REPAIR_EXAMPLE,
)
register_resource_profile(ROBOT_PROFILE)
