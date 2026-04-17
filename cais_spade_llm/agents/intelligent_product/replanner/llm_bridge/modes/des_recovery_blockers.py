"""Active recovery blocker collection for the DES recovery engine."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


def _truthy_flag(row: dict[str, Any], *field_names: str) -> bool:
    return any(bool(row.get(field_name)) for field_name in field_names)


def compute_observation_blockers(
    symbolic_parts: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Identify parts whose localization is unknown or explicitly untrusted."""
    blockers: list[dict[str, Any]] = []
    for part_name, row in (symbolic_parts or {}).items():
        if not isinstance(row, dict):
            continue
        observed_pose = row.get("observed_pose")
        has_pose = isinstance(observed_pose, dict) and bool(observed_pose)
        current_location = str(row.get("current_location") or "").strip()
        last_known_location = str(row.get("last_known_location") or "").strip()
        location_basis = str(row.get("location_basis") or "").strip().lower()

        reason_parts: list[str] = []
        if _truthy_flag(
            row,
            "needs_observation",
            "requires_observation",
            "observation_required",
            "pose_untrusted",
            "location_unverified",
            "localization_unknown",
        ):
            reason_parts.append("explicit observation/localization flag is set")

        if location_basis in {"sensor_observation", "live_observation"} and not has_pose:
            reason_parts.append(
                f"location_basis='{location_basis}' but no observed_pose is stored"
            )

        if not current_location and not last_known_location and not has_pose:
            reason_parts.append("no current location, no last known location, and no observed pose")

        if reason_parts:
            blockers.append({
                "kind": "observation_required",
                "part_name": part_name,
                "reason": "; ".join(reason_parts),
                "description": (
                    f"OBSERVATION REQUIRED: part '{part_name}' has untrusted localization "
                    f"({'; '.join(reason_parts)}). Recovery must ground this part before "
                    "relying on its pose or live workspace occupancy."
                ),
            })
    return blockers


def compute_terminal_state_blockers(
    symbolic_resources: dict[str, dict[str, Any]],
    *,
    extra_terminal_state_names: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Resources currently blocked from participation in recovery."""
    extras = {str(s).strip().lower() for s in (extra_terminal_state_names or set())}
    blockers: list[dict[str, Any]] = []
    for jid, row in (symbolic_resources or {}).items():
        if not isinstance(row, dict):
            continue
        state = str(row.get("current_state") or "").strip().lower()
        reasons: list[str] = []
        if row.get("available") is False:
            reasons.append("available=False")
        for flag in ("is_blocked", "is_faulted", "is_error", "is_terminal"):
            if bool(row.get(flag)):
                reasons.append(f"{flag}=True")
        for fault_field in ("fault", "error", "error_code"):
            value = row.get(fault_field)
            if value not in (None, "", 0, False, [], {}):
                reasons.append(f"{fault_field}={value!r}")
        if state and state in extras:
            reasons.append(f"current_state='{state}' matches configured terminal-state name")
        if reasons:
            blockers.append({
                "kind": "resource_terminal_state",
                "resource_jid": jid,
                "current_state": state,
                "reason": "; ".join(reasons),
                "description": (
                    f"RESOURCE BLOCKED: '{jid}' shows terminal-state evidence "
                    f"({'; '.join(reasons)}). Recovery must clear or route around it."
                ),
            })
    return blockers


def _part_goal_reached(row: dict[str, Any]) -> bool:
    current_location = str(row.get("current_location") or "").strip()
    goal_location = str(row.get("goal_location") or "").strip()
    return bool(current_location) and bool(goal_location) and current_location == goal_location


def compute_assembly_order_blockers(
    llm_input: dict[str, Any],
    symbolic_parts: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Detect unmet predecessor dependencies from explicit requirement structure."""
    del llm_input
    blockers: list[dict[str, Any]] = []
    for part_name, row in (symbolic_parts or {}).items():
        if not isinstance(row, dict) or _part_goal_reached(row):
            continue
        predecessors = row.get("predecessor_parts") or row.get("predecessors") or []
        if isinstance(predecessors, str):
            predecessors = [predecessors]
        for predecessor in predecessors if isinstance(predecessors, list) else []:
            predecessor_name = str(predecessor or "").strip()
            if not predecessor_name:
                continue
            predecessor_row = dict((symbolic_parts or {}).get(predecessor_name) or {})
            if predecessor_row and not _part_goal_reached(predecessor_row):
                blockers.append({
                    "kind": "assembly_order",
                    "part_name": part_name,
                    "predecessor": predecessor_name,
                    "reason": (
                        f"part '{part_name}' depends on predecessor '{predecessor_name}' "
                        "which has not yet reached its goal location"
                    ),
                    "description": (
                        f"ORDER BLOCKED: '{part_name}' cannot be considered resumable until "
                        f"predecessor '{predecessor_name}' reaches its goal location."
                    ),
                })
    return blockers


def _resource_workspace_bounds(resource_entry: dict[str, Any]) -> dict[str, Any] | None:
    entry = dict(resource_entry or {})
    caps = dict(entry.get("static_capabilities") or entry.get("capabilities") or {})
    bounds = caps.get("workspace_bounds") or entry.get("workspace_bounds")
    return dict(bounds) if isinstance(bounds, dict) else None


def _pose_in_bounds(pose: dict[str, Any], bounds: dict[str, Any]) -> bool:
    try:
        x = float(pose["x"])
        y = float(pose["y"])
        z = float(pose["z"])
    except (KeyError, TypeError, ValueError):
        return False
    for axis, value in (("x", x), ("y", y), ("z", z)):
        lo = bounds.get(f"{axis}_min_m")
        hi = bounds.get(f"{axis}_max_m")
        if lo is not None and value < float(lo):
            return False
        if hi is not None and value > float(hi):
            return False
    return True


def compute_shared_workspace_blockers(
    bridge_resources: dict[str, dict[str, Any]],
    symbolic_parts: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Detect cross-resource workspace occupancy that can require transfer/resequencing."""
    blockers: list[dict[str, Any]] = []
    for part_name, row in (symbolic_parts or {}).items():
        if not isinstance(row, dict):
            continue
        assigned = str(
            row.get("assigned_resource_jid")
            or row.get("goal_resource_jid")
            or row.get("nominal_requirement_resource_jid")
            or row.get("resource_jid")
            or ""
        ).strip()
        pose = row.get("observed_pose")
        if not assigned or not isinstance(pose, dict):
            continue
        for jid, resource_entry in (bridge_resources or {}).items():
            if jid == assigned:
                continue
            bounds = _resource_workspace_bounds(dict(resource_entry or {}))
            if bounds and _pose_in_bounds(pose, bounds):
                blockers.append({
                    "kind": "shared_workspace",
                    "part_name": part_name,
                    "assigned_resource_jid": assigned,
                    "host_resource_jid": jid,
                    "reason": (
                        f"part '{part_name}' is assigned to '{assigned}' but currently lies in "
                        f"'{jid}' workspace"
                    ),
                    "description": (
                        f"WORKSPACE CONFLICT: '{part_name}' currently lies in '{jid}' workspace "
                        f"while assigned to '{assigned}'. Recovery may require part_transfer, "
                        "reassignment, or resequencing. Do not assign the observed-pose "
                        f"pick/grasp for '{part_name}' to '{assigned}' while that pose is "
                        f"outside '{assigned}' workspace; use '{jid}' for reachable recovery "
                        "or model a physically grounded part_transfer."
                    ),
                })
                break
    return blockers


def compute_reachability_blockers(
    symbolic_parts: dict[str, dict[str, Any]],
    bridge_resources: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Detect parts whose observed pose is outside every known resource workspace."""
    blockers: list[dict[str, Any]] = []
    for part_name, row in (symbolic_parts or {}).items():
        if not isinstance(row, dict):
            continue
        pose = row.get("observed_pose")
        if not isinstance(pose, dict):
            continue
        reachable_jids: list[str] = []
        for jid, resource_entry in (bridge_resources or {}).items():
            bounds = _resource_workspace_bounds(dict(resource_entry or {}))
            if bounds and _pose_in_bounds(pose, bounds):
                reachable_jids.append(jid)
        if not reachable_jids:
            blockers.append({
                "kind": "reachability",
                "part_name": part_name,
                "observed_pose": deepcopy(pose),
                "reason": f"part '{part_name}' observed pose is outside every resource workspace",
                "description": (
                    f"UNREACHABLE: '{part_name}' currently has no grounded reachable resource "
                    "for its observed pose."
                ),
            })
    return blockers


def collect_recovery_blockers(
    *,
    session_state: dict[str, Any],
    prepared_bridge_request: dict[str, Any],
) -> list[dict[str, Any]]:
    """Aggregate recovery blockers from the active synchronized symbolic state."""
    symbolic_parts = dict(session_state.get("symbolic_parts") or {})
    symbolic_resources = dict(session_state.get("symbolic_resources") or {})
    bridge_resources = dict(prepared_bridge_request.get("bridge_resources") or {})
    llm_input = dict(prepared_bridge_request.get("llm_input") or {})
    bridge_session = dict(prepared_bridge_request.get("bridge_session") or {})

    extra_terminal_state_names = bridge_session.get("extra_terminal_resource_state_names")
    if isinstance(extra_terminal_state_names, (list, tuple, set)):
        extras = {str(s).strip().lower() for s in extra_terminal_state_names if str(s).strip()}
    else:
        extras = set()

    blockers: list[dict[str, Any]] = []
    blockers.extend(compute_observation_blockers(symbolic_parts))
    blockers.extend(
        compute_terminal_state_blockers(
            symbolic_resources,
            extra_terminal_state_names=extras,
        )
    )
    blockers.extend(compute_assembly_order_blockers(llm_input, symbolic_parts))
    blockers.extend(compute_shared_workspace_blockers(bridge_resources, symbolic_parts))
    blockers.extend(compute_reachability_blockers(symbolic_parts, bridge_resources))
    return blockers


__all__ = [
    "collect_recovery_blockers",
    "compute_assembly_order_blockers",
    "compute_observation_blockers",
    "compute_reachability_blockers",
    "compute_shared_workspace_blockers",
    "compute_terminal_state_blockers",
]
