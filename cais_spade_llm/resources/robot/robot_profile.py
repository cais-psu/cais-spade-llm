"""Built-in robot resource profile and robot-specific recovery helpers."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from cais_spade_llm.resources.resource_profile import (
    ResourceProfile,
    register_resource_profile,
    resource_snapshot_carried_entity,
    resource_snapshot_field_value,
)
from cais_spade_llm.resources.robot import UR5eGazeboController, XArm6GazeboController
from cais_spade_llm.resources.robot.robot_primitives import (
    ROBOT_COMPILER_MAP,
    ROBOT_EVENT_FACT_CONTRACT_MAP,
    ROBOT_EVENT_FACT_KEY_MAP,
    ROBOT_EXTRACT_OUTPUT_MAP,
    ROBOT_OBSERVATION_OUTPUT_SCHEMA_MAP,
    ROBOT_PREVIEW_OUTPUT_MAP,
    ROBOT_PRIMITIVE_EVENT_TARGET_CONTRACT_MAP,
    ROBOT_PRIMITIVE_KIND_MAP,
    ROBOT_PRIMITIVE_TRACE_FACT_MAP,
    robot_capability_decompositions,
    robot_primitive_sequence_validator,
    robot_symbolic_event_family,
)


@dataclass(frozen=True)
class RobotProfile(ResourceProfile):
    """Robot-specific resource profile."""

    resource_type: str = "robot"


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


def _robot_snapshot_equivalence(
    *,
    field: str,
    actual_snapshot: dict[str, Any],
    projected_snapshot: dict[str, Any],
    actual_value: Any,
    projected_value: Any,
    profile: ResourceProfile,
) -> bool:
    if field != "current_state":
        return False
    if projected_value != "picked" or actual_value != "idle":
        return False
    actual_carried = str(
        resource_snapshot_carried_entity(actual_snapshot, profile=profile) or ""
    ).strip()
    projected_carried = str(
        resource_snapshot_carried_entity(projected_snapshot, profile=profile) or ""
    ).strip()
    actual_gripper_state = str(
        resource_snapshot_field_value(actual_snapshot, "gripper_state", profile=profile) or ""
    ).strip()
    projected_gripper_state = str(
        resource_snapshot_field_value(projected_snapshot, "gripper_state", profile=profile) or ""
    ).strip()
    return (
        bool(actual_carried)
        and actual_carried == projected_carried
        and actual_gripper_state == projected_gripper_state
        and actual_gripper_state == "closed"
    )


def _robot_event_family(event: dict[str, Any]) -> str:
    symbolic_family = robot_symbolic_event_family(event)
    if symbolic_family is not None:
        return symbolic_family

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
        if delta_from in {"idle", "failed"} or "pick_place" in event_name:
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
            f"recovery event '{event_name}' is inconsistent for robot resources: "
            f"operation_family '{operation_family}' must not declare part_name"
        )
    if operation_family == "pick":
        if part_to and part_to != "in_gripper":
            return (
                f"recovery event '{event_name}' is inconsistent for robot resources: "
                "'pick' requires expected_part_delta.to='in_gripper'"
            )
        if delta_to and delta_to != "picked":
            return (
                f"recovery event '{event_name}' is inconsistent for robot resources: "
                "'pick' requires expected_resource_delta.to='picked'"
            )
    if operation_family == "stage":
        if part_to and part_to != "ready":
            return (
                f"recovery event '{event_name}' is inconsistent for robot resources: "
                "'stage' requires expected_part_delta.to='ready'"
            )
        if delta_to == "picked":
            return (
                f"recovery event '{event_name}' is inconsistent for robot resources: "
                "'stage' cannot keep the resource in state 'picked'"
            )
    if operation_family == "assemble":
        if part_to and part_to != "assembled":
            return (
                f"recovery event '{event_name}' is inconsistent for robot resources: "
                "'assemble' requires expected_part_delta.to='assembled'"
            )
        if delta_to == "picked":
            return (
                f"recovery event '{event_name}' is inconsistent for robot resources: "
                "'assemble' cannot keep the resource in state 'picked'"
            )
    if operation_family == "place":
        if part_to == "in_gripper":
            return (
                f"recovery event '{event_name}' is inconsistent for robot resources: "
                "'place' cannot leave the part in_gripper"
            )
        if delta_to == "picked":
            return (
                f"recovery event '{event_name}' is inconsistent for robot resources: "
                "'place' cannot keep the resource in state 'picked'"
            )
    return None


def _robot_pose_in_workspace(
    pose: dict[str, Any],
    bounds: dict[str, Any],
) -> tuple[bool, list[str]]:
    violations: list[str] = []
    for axis in ("x", "y", "z"):
        value = pose.get(axis)
        if value is None:
            continue
        try:
            coord = float(value)
        except (TypeError, ValueError):
            continue
        lower = bounds.get(f"{axis}_min_m")
        upper = bounds.get(f"{axis}_max_m")
        if lower is not None and coord < float(lower):
            violations.append(f"{axis}={coord:.4f} < {axis}_min_m={float(lower):.4f}")
        if upper is not None and coord > float(upper):
            violations.append(f"{axis}={coord:.4f} > {axis}_max_m={float(upper):.4f}")
    return len(violations) == 0, violations


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
                    f"recovery event '{str(event.get('event_name', '') or resource_jid).strip()}' "
                    f"is inconsistent: {resource_jid} must be carrying '{part_name}' "
                    f"before it starts"
                )
            return (
                f"recovery event '{str(event.get('event_name', '') or resource_jid).strip()}' "
                f"is inconsistent: {resource_jid} must be carrying '{part_name}' before it "
                f"starts, but projected carried entity is '{carried_before}'"
            )
        if carried_after:
            return (
                f"recovery event '{str(event.get('event_name', '') or resource_jid).strip()}' "
                f"is inconsistent: it claims '{part_name}' is released, but projected "
                f"carried entity after the event is '{carried_after}'"
            )

    if semantic_kind == "pick" and carried_after != part_name:
        return (
            f"recovery event '{str(event.get('event_name', '') or resource_jid).strip()}' "
            f"is inconsistent: it claims '{part_name}' is acquired, but projected carried "
            f"entity after the event is {carried_after!r}"
        )

    return None


def _robot_target_location_for_event(outline_event: dict[str, Any]) -> str:
    expected_end = dict(outline_event.get("expected_end_state") or {})
    action_target = dict(outline_event.get("action_target") or {})
    return (
        str(expected_end.get("part_location") or "").strip()
        or str(outline_event.get("target_ref") or "").strip()
        or str(action_target.get("target_location") or "").strip()
    )


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
        return UR5eGazeboController
    if scope_name.startswith("xarm6"):
        return XArm6GazeboController
    return None


def _robot_snapshot_builder(agent: Any) -> dict[str, Any]:
    current_pose = None
    controller = getattr(agent, "_controller", None)
    if (
        controller is not None
        and str(getattr(agent, "execution_mode", "")).strip().lower() != "dry_run"
    ):
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
    if current_pose is None and getattr(agent, "_recovery_pose_ref", None) is None:
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
        "current_pose_ref": getattr(agent, "_recovery_pose_ref", None),
        "named_poses": sorted((getattr(agent, "named_positions", {}) or {}).keys()),
    }


ROBOT_PROFILE = RobotProfile(
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
        "current_pose_ref": "_recovery_pose_ref",
    },
    primitive_kind_map=ROBOT_PRIMITIVE_KIND_MAP,
    primitive_trace_fact_map=ROBOT_PRIMITIVE_TRACE_FACT_MAP,
    event_family_resolver=_robot_event_family,
    event_target_resolver=_robot_target_location_for_event,
    event_contract_validator=_robot_event_contract_validator,
    compiler_map=ROBOT_COMPILER_MAP,
    state_projector=_robot_state_projector,
    event_state_validator=_robot_event_state_validator,
    primitive_sequence_validator=robot_primitive_sequence_validator,
    capability_decomposition_provider=robot_capability_decompositions,
    capability_flags={"supports_manipulator_pick_place": True},
    observation_families=("part_detection", "resource_pose"),
    grounding_observation_primitives=("detect_parts",),
    grounding_observation_fact_map={
        "part_pose": {
            "entity_kind": "part",
            "primitive": "detect_parts",
            "entity_param": "part_name",
            "request_fields": ("fact_type", "entity"),
            "optional_request_fields": ("scope", "reason"),
        },
    },
    example_families=("generic_recovery", "manipulator_pick_place"),
    observation_output_schema_map=ROBOT_OBSERVATION_OUTPUT_SCHEMA_MAP,
    preview_output_map=ROBOT_PREVIEW_OUTPUT_MAP,
    extract_output_map=ROBOT_EXTRACT_OUTPUT_MAP,
    event_fact_contract_map=ROBOT_EVENT_FACT_CONTRACT_MAP,
    event_fact_key_map=ROBOT_EVENT_FACT_KEY_MAP,
    primitive_event_target_contract_map=ROBOT_PRIMITIVE_EVENT_TARGET_CONTRACT_MAP,
    expected_end_state_projection_map={
        "held_part": "held_part",
    },
    carried_entity_field="held_part",
    carried_entity_location_builder=_robot_carried_location,
    snapshot_equivalence_resolver=_robot_snapshot_equivalence,
)
register_resource_profile(ROBOT_PROFILE)
