"""Shared failure-context builders and scenario config helpers."""

from __future__ import annotations

import json
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any

_FAILURE_SCENARIOS_ROOT = (
    Path(__file__).resolve().parents[3] / "initialization" / "failure_scenarios"
)


def _coerce_xyz_pose(value: Any) -> dict[str, float] | None:
    if not isinstance(value, dict):
        return None
    pose: dict[str, float] = {}
    for axis in ("x", "y", "z"):
        raw_value = value.get(axis)
        if raw_value is None:
            return None
        try:
            pose[axis] = float(raw_value)
        except (TypeError, ValueError):
            return None
    return pose


@lru_cache(maxsize=32)
def _load_failure_scenario_cached(path_str: str) -> dict[str, Any]:
    payload = json.loads(Path(path_str).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"failure scenario {path_str} did not decode to an object")
    return payload


def load_failure_scenario_config(
    scenario_id: str,
    *,
    root: Path | None = None,
) -> dict[str, Any]:
    scenario_key = str(scenario_id or "").strip()
    if not scenario_key:
        raise ValueError("scenario_id is empty")
    scenario_path = (root or _FAILURE_SCENARIOS_ROOT) / f"{scenario_key}.json"
    if not scenario_path.exists():
        raise FileNotFoundError(f"failure scenario not found: {scenario_path}")
    return deepcopy(_load_failure_scenario_cached(str(scenario_path.resolve())))


def normalize_failure_observations(
    observations: dict[str, Any] | None = None,
    *,
    function_name: str = "",
    status: str = "",
    raw_failure_mode: str = "",
    state_before: dict[str, Any] | None = None,
    state_after: dict[str, Any] | None = None,
) -> dict[str, Any]:
    normalized = deepcopy(observations if isinstance(observations, dict) else {})

    for key in (
        "observed_pose",
        "pose",
        "position",
        "dropped_location",
        "last_known_position",
    ):
        pose = _coerce_xyz_pose(normalized.get(key))
        if pose is not None:
            normalized[key] = pose

    if function_name and "function_name" not in normalized:
        normalized["function_name"] = str(function_name).strip()
    if status and "status" not in normalized:
        normalized["status"] = str(status).strip()
    if raw_failure_mode and "raw_failure_mode" not in normalized:
        normalized["raw_failure_mode"] = str(raw_failure_mode).strip()
    if state_before and "state_before" not in normalized:
        normalized["state_before"] = deepcopy(state_before)
    if state_after and "state_after" not in normalized:
        normalized["state_after"] = deepcopy(state_after)

    return normalized


def failure_context_from_scenario_config(
    scenario_config: dict[str, Any] | None,
) -> dict[str, Any]:
    config = dict(scenario_config or {})
    context_template = deepcopy(config.get("failure_context") or {})
    if not isinstance(context_template, dict):
        context_template = {}

    affected_entities = deepcopy(config.get("affected_entities") or [])
    if isinstance(affected_entities, list) and affected_entities:
        context_template["affected_entities"] = affected_entities

    observations = deepcopy(config.get("observation_template") or {})
    if isinstance(observations, dict) and observations:
        context_template["observations"] = observations

    return context_template


def build_failure_event(
    *,
    failed_task_id: str = "",
    failed_resource_jid: str = "",
    failed_function_name: str = "",
    final_status: str = "",
    part_name: str = "",
    base_failure_context: dict[str, Any] | None = None,
    observations: dict[str, Any] | None = None,
    affected_entities: list[dict[str, Any]] | None = None,
    state_before: dict[str, Any] | None = None,
    state_after: dict[str, Any] | None = None,
) -> dict[str, Any]:
    failure_context = deepcopy(
        base_failure_context if isinstance(base_failure_context, dict) else {}
    )
    observed_failure_mode = str(
        failure_context.get("failure_mode")
        or (observations.get("raw_failure_mode") if isinstance(observations, dict) else "")
        or (final_status.split(":", 1)[1] if ":" in str(final_status or "") else "")
    ).strip()

    resolved_affected_entities = (
        deepcopy(affected_entities)
        if isinstance(affected_entities, list)
        else deepcopy(failure_context.get("affected_entities") or [])
    )
    if not isinstance(resolved_affected_entities, list):
        resolved_affected_entities = []
    if not resolved_affected_entities and str(part_name or "").strip():
        resolved_affected_entities = [
            {
                "entity_type": "part",
                "entity_id": str(part_name).strip(),
                "state": "unknown",
            }
        ]

    merged_observations = deepcopy(
        failure_context.get("observations")
        if isinstance(failure_context.get("observations"), dict)
        else {}
    )
    if isinstance(observations, dict):
        merged_observations.update(deepcopy(observations))
    normalized_observations = normalize_failure_observations(
        merged_observations,
        function_name=str(failed_function_name or "").strip(),
        status=str(final_status or "").strip(),
        raw_failure_mode=observed_failure_mode,
        state_before=state_before,
        state_after=state_after,
    )

    observed_failure_context = {
        key: deepcopy(value)
        for key, value in failure_context.items()
        if key not in {"affected_entities", "observations"}
    }
    if observed_failure_mode and "failure_mode" not in observed_failure_context:
        observed_failure_context["failure_mode"] = observed_failure_mode
    if resolved_affected_entities:
        observed_failure_context["affected_entities"] = resolved_affected_entities
    if normalized_observations:
        observed_failure_context["observations"] = normalized_observations

    return {
        "failed_task_id": str(failed_task_id or "").strip(),
        "failed_resource_jid": str(failed_resource_jid or "").strip(),
        "failed_function_name": str(failed_function_name or "").strip(),
        "failure_context": observed_failure_context,
    }
