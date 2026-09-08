from __future__ import annotations

"""Measure advisory robot proximity without changing reachability or selection."""

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..ra.feasibility_validation import RobotAgentFeasibilityRuntime
    from .resource_grounding import ReachabilityCheckRecord

_BASE_FRAMES_PATH = Path(__file__).resolve().parents[2] / "config/resource_base_frames.json"


async def read_resource_proximity(
    runtime: RobotAgentFeasibilityRuntime,
    reachability: ReachabilityCheckRecord,
) -> dict[str, object]:
    """Return validated TF distance evidence, or an advisory unavailable result."""
    try:
        base_frame = _configured_base_frame(reachability.resource_symbol)
        pose = await runtime.read_resource_base_pose(
            base_frame=base_frame, target_frame=reachability.target_frame
        )
        if not isinstance(pose, Mapping) or pose.get("base_frame") != base_frame:
            raise ValueError("Robot base pose differs from its configured frame.")
        result = _proximity_from_pose(pose, reachability.to_record())
        validate_resource_proximity(result, reachability.to_record())
        return result
    except (AttributeError, KeyError, OSError, OverflowError, RuntimeError, TypeError, ValueError):
        # Missing proximity must not manufacture a distance or override a MoveIt verdict.
        return {
            "status": "unavailable",
            "feedback": "Robot base proximity is unavailable; no distance preference is supported.",
        }


def validate_resource_proximity(
    value: Mapping[str, object], reachability: Mapping[str, object]
) -> None:
    """Recompute advisory distances against the exact pinned reachability locations."""
    if not isinstance(value, Mapping):
        raise ValueError("Resource proximity is invalid.")
    if value.get("status") == "unavailable":
        if set(value) != {"status", "feedback"} or not isinstance(value["feedback"], str):
            raise ValueError("Unavailable proximity must not contain distance evidence.")
        return
    if (
        set(value) != {"status", "base_pose", "state_locations", "mean_current_state_distance_m"}
        or value["status"] != "available"
    ):
        raise ValueError("Resource proximity fields are invalid.")
    expected = _proximity_from_pose(value["base_pose"], reachability)
    if value != expected:
        raise ValueError("Resource proximity differs from its pinned base pose and locations.")
    distances = [value["mean_current_state_distance_m"]]
    distances.extend(
        entry["distance_m"] for entries in value["state_locations"].values() for entry in entries
    )
    if any(type(item) not in (int, float) or not math.isfinite(item) for item in distances):
        raise ValueError("Resource proximity distances must be finite numbers.")


def validate_allocation_proximity(
    result: Mapping[str, object], reachability: Mapping[str, object]
) -> None:
    """Validate advisory proximity when supplied in an allocation tool result."""
    if "proximity" in result:
        validate_resource_proximity(result["proximity"], reachability)


def _proximity_from_pose(
    pose: Mapping[str, object], reachability: Mapping[str, object]
) -> dict[str, object]:
    if not isinstance(pose, Mapping) or set(pose) != {
        "base_frame",
        "target_frame",
        "translation_m",
        "observed_at_ns",
    }:
        raise ValueError("Robot base pose fields are invalid.")
    if (
        pose["base_frame"] != _configured_base_frame(reachability["resource_symbol"])
        or pose["target_frame"] != reachability["target_frame"]
        or type(pose["observed_at_ns"]) is not int
        or pose["observed_at_ns"] <= 0
    ):
        raise ValueError("Robot base pose frame or timestamp is invalid.")
    base = _coordinates(pose["translation_m"])
    locations = reachability["state_locations"]
    if not isinstance(locations, Mapping) or set(locations) != {"current_state", "desired_state"}:
        raise ValueError("Proximity requires both bound location sets.")
    distances: dict[str, list[dict[str, object]]] = {}
    for state, entries in locations.items():
        if not isinstance(entries, list) or not entries:
            raise ValueError("Proximity state locations are missing.")
        # The tool's pinned reachability record already binds these handles to location refs.
        distances[state] = [
            {
                "evidence_handle": entry["evidence_handle"],
                "distance_m": math.dist(base, _coordinates(entry["translation_m"])),
            }
            for entry in entries
        ]
        if any(not math.isfinite(entry["distance_m"]) for entry in distances[state]):
            raise ValueError("Resource proximity distance is not finite.")
    current = distances["current_state"]
    return {
        "status": "available",
        "base_pose": dict(pose),
        "state_locations": distances,
        "mean_current_state_distance_m": math.fsum(
            item["distance_m"] / len(current) for item in current
        ),
    }


def _configured_base_frame(resource_symbol: str) -> str:
    frames = json.loads(_BASE_FRAMES_PATH.read_text(encoding="utf-8"))
    if not isinstance(frames, dict):
        raise ValueError("Resource base-frame configuration is invalid.")
    base_frame = frames[resource_symbol]
    if not isinstance(base_frame, str) or not base_frame or base_frame != base_frame.strip():
        raise ValueError("Configured robot base frame is invalid.")
    return base_frame


def _coordinates(value: object) -> tuple[float, float, float]:
    if (
        not isinstance(value, list)
        or len(value) != 3
        or any(type(item) not in (int, float) or not math.isfinite(item) for item in value)
    ):
        raise ValueError("Proximity coordinates must be three finite numbers.")
    return tuple(float(item) for item in value)
