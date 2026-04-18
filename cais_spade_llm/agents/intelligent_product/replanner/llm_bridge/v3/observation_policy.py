from __future__ import annotations

from copy import deepcopy
from typing import Any


def primitive_param_names(entry: dict[str, Any]) -> set[str]:
    names: set[str] = set()
    raw_params = entry.get("params")
    if isinstance(raw_params, dict):
        names.update(str(name).strip().lower() for name in raw_params if str(name).strip())
    raw_params_schema = entry.get("params_schema")
    if isinstance(raw_params_schema, dict):
        names.update(str(name).strip().lower() for name in raw_params_schema if str(name).strip())
    raw_parameters = entry.get("parameters")
    if isinstance(raw_parameters, dict):
        properties = raw_parameters.get("properties") or {}
        if isinstance(properties, dict):
            names.update(str(name).strip().lower() for name in properties if str(name).strip())
    names.update(
        str(name).strip().lower()
        for name in (entry.get("required_params") or [])
        if str(name).strip()
    )
    return names


def primitive_output_field_names(entry: dict[str, Any]) -> set[str]:
    semantics = entry.get("bridge_semantics") or {}
    schema = semantics.get("observation_output_schema") or {}
    if not isinstance(schema, dict):
        return set()
    properties = schema.get("properties") or {}
    if not isinstance(properties, dict):
        return set()
    return {
        str(name).strip().lower()
        for name in properties
        if str(name).strip()
    }


def primitive_supports_part_grounding(entry: dict[str, Any]) -> bool:
    tokens = primitive_param_names(entry) | primitive_output_field_names(entry)
    if "part_name" in tokens:
        return True
    return any(token.startswith("part") for token in tokens)


def observation_semantic_operation(
    *,
    primitive_name: str,
    entry: dict[str, Any] | None = None,
    part_name: str | None = None,
) -> str:
    primitive = str(primitive_name or "").strip()
    if primitive == "detect_parts":
        return "observe_part_pose"
    if primitive == "get_current_pose":
        return "observe_resource_pose"
    if entry and primitive_supports_part_grounding(entry):
        return "observe_part_pose"
    if part_name:
        return "observe_part_pose"
    return "observe_state"


def semantic_observation_candidates_from_grounding_candidates(
    candidate_observations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    semantic_rows: list[dict[str, Any]] = []
    for row in candidate_observations or []:
        if not isinstance(row, dict):
            continue
        part_name = str(row.get("part_name") or "").strip()
        recommended = dict(row.get("recommended_observer") or {})
        recommended_binding = {}
        resource_jid = str(recommended.get("resource_jid") or "").strip()
        primitive_name = str(recommended.get("recommended_primitive") or "").strip()
        semantic_operation = observation_semantic_operation(
            primitive_name=primitive_name,
            part_name=part_name,
        )
        if resource_jid and primitive_name:
            recommended_binding = {
                "resource_jid": resource_jid,
                "primitive": primitive_name,
            }
        binding_options: list[dict[str, Any]] = []
        for option in (row.get("observer_options") or []):
            if not isinstance(option, dict):
                continue
            option_resource = str(option.get("resource_jid") or "").strip()
            for primitive in (option.get("primitives") or []):
                primitive_token = str(primitive).strip()
                if not option_resource or not primitive_token:
                    continue
                binding_options.append({
                    "resource_jid": option_resource,
                    "primitive": primitive_token,
                    "semantic_operation": observation_semantic_operation(
                        primitive_name=primitive_token,
                        part_name=part_name,
                    ),
                })
        semantic_rows.append({
            "semantic_operation": semantic_operation,
            "part_name": part_name,
            "blocked_transition": row.get("blocked_transition"),
            "recommended_binding": recommended_binding,
            "binding_options": binding_options,
            "observation_policy_reason": row.get("observation_policy_reason"),
            "priority_score": row.get("priority_score"),
            "affected_entities": deepcopy(row.get("affected_entities") or []),
        })
    semantic_rows.sort(
        key=lambda item: (
            -int(item.get("priority_score") or 0),
            str(item.get("part_name") or ""),
        )
    )
    return semantic_rows
