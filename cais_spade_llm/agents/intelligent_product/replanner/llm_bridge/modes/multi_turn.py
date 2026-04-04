"""Multi-turn executor for the active v4 bridge."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import inspect
import json
import logging
from typing import Any, Callable
from uuid import uuid4

_logger = logging.getLogger(__name__)

from cais_spade_llm.agents.central_controller.outline_macro_safety import (
    validate_outline_macro_cca_constraints,
)
from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.bridge_artifacts import (
    write_bridge_artifacts,
)
from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.bridge_grounding_compiler import (
    compile_grounded_outline_task,
)
from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.primitive_semantics import (
    extract_step_output,
)
from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.prompts.multi_turn import (
    build_multi_turn_phase_prompt_input,
    multi_turn_phase_response_schema,
    render_multi_turn_phase_prompt,
)
from cais_spade_llm.resources.resource_profile import (
    get_resource_profile,
    get_resource_profile_for_agent,
    resource_store_as_contract,
)


_DEFAULT_MAX_TURNS = 8
_DEFAULT_MAX_OBSERVATIONS = 3
_DEFAULT_MAX_OBSERVE_BATCH = 3

_PHASE_SEQUENCE = (
    "grounding",
    "outline",
    "primitive_generation",
    "finalize",
)

_TRANSITIONS: dict[str, dict[str, str]] = {
    "grounding": {
        "observe": "grounding",
        "grounded": "outline",
    },
    "outline": {
        "outline_ready": "primitive_generation",
        "need_revision": "outline",
    },
    "primitive_generation": {
        "need_outline_revision": "outline",
        "draft_ready": "finalize",
    },
    "finalize": {
        "need_outline_revision": "outline",
        "need_primitive_revision": "primitive_generation",
        "final_ready": "finalize",
    },
}

_DECISION_COMPATIBILITY_DOWGRADES: dict[str, dict[str, str]] = {
    "outline": {
        "need_grounding": "need_revision",
    },
    "primitive_generation": {
        "need_grounding": "need_outline_revision",
    },
    "finalize": {
        "need_grounding": "need_outline_revision",
    },
}


def transition_multi_turn_phase(current_phase: str, decision: str) -> str:
    phase = str(current_phase or "").strip().lower()
    token = str(decision or "").strip().lower()
    try:
        return _TRANSITIONS[phase][token]
    except KeyError as exc:
        raise ValueError(
            f"unsupported multi-turn transition: phase={phase!r} decision={token!r}"
        ) from exc


def _normalize_phase_decision(
    *,
    current_phase: str,
    decision: str,
) -> tuple[str, dict[str, Any] | None]:
    phase = str(current_phase or "").strip().lower()
    token = str(decision or "").strip().lower()
    normalized = str(
        (_DECISION_COMPATIBILITY_DOWGRADES.get(phase) or {}).get(token) or token
    ).strip().lower()
    if not token or normalized == token:
        return token, None
    return normalized, {
        "original_decision": token,
        "normalized_decision": normalized,
        "reason": (
            "Post-grounding response contracts do not permit this decision token; "
            "the runtime normalized it to the nearest supported revision decision."
        ),
    }


def build_multi_turn_session_seed(
    prepared_bridge_request: dict[str, Any],
) -> dict[str, Any]:
    bridge_session = dict(prepared_bridge_request.get("bridge_session") or {})
    session_id = str(bridge_session.get("session_id") or "").strip() or f"mt_{uuid4().hex[:8]}"
    max_turns = max(
        1,
        int(bridge_session.get("max_turns", _DEFAULT_MAX_TURNS) or _DEFAULT_MAX_TURNS),
    )
    max_observations = max(
        0,
        int(
            bridge_session.get("max_observations", _DEFAULT_MAX_OBSERVATIONS)
            or _DEFAULT_MAX_OBSERVATIONS
        ),
    )
    max_observe_batch = max(
        1,
        min(
            3,
            int(
                bridge_session.get("max_observe_batch", _DEFAULT_MAX_OBSERVE_BATCH)
                or _DEFAULT_MAX_OBSERVE_BATCH
            ),
        ),
    )
    stop_after_phase = str(bridge_session.get("stop_after_phase") or "").strip().lower()
    if stop_after_phase not in _PHASE_SEQUENCE:
        stop_after_phase = ""
    return {
        "session_id": session_id,
        "status": "prepared_for_llm",
        "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
        "current_phase": "grounding",
        "turn_index": 0,
        "max_turns": max_turns,
        "max_observations": max_observations,
        "max_observe_batch": max_observe_batch,
        "stop_after_phase": stop_after_phase or None,
        "observation_count": 0,
        "observation_store": {},
        "observation_fact_ledger": {},
        "observation_history": [],
        "accepted_outline": None,
        "proposal_draft": None,
        "phase_feedback": [],
        "pruned_actions": [],
        "turns": [],
        "final_proposal": None,
    }


def _resource_agent_map(planner: Any) -> dict[str, Any]:
    return {
        str(getattr(agent, "jid", "")).strip(): agent
        for agent in (getattr(planner, "resource_agents", None) or [])
        if str(getattr(agent, "jid", "")).strip()
    }


def _supports_store_as(*, resource_type: str, primitive_name: str) -> bool:
    profile = get_resource_profile(resource_type or "resource")
    return (
        str(primitive_name or "").strip() in dict(profile.preview_output_map or {})
        or str(primitive_name or "").strip() in dict(profile.extract_output_map or {})
    )


def _is_grounding_observation_primitive(*, resource_type: str, primitive_name: str) -> bool:
    profile = get_resource_profile(resource_type or "resource")
    primitive_token = str(primitive_name or "").strip()
    allowlist = {
        str(item).strip()
        for item in (profile.grounding_observation_primitives or ())
        if str(item).strip()
    }
    if allowlist:
        return primitive_token in allowlist
    return True


def _store_as_contract_fields(*, resource_type: str, primitive_name: str) -> tuple[list[str], list[list[str]]]:
    profile = get_resource_profile(resource_type or "resource")
    contract = resource_store_as_contract(profile, primitive_name)
    required_params = [
        str(param).strip()
        for param in (contract.get("required_params") or [])
        if str(param).strip()
    ]
    any_of_param_sets = [
        [
            str(param).strip()
            for param in (param_set or [])
            if str(param).strip()
        ]
        for param_set in (contract.get("any_of_param_sets") or [])
        if isinstance(param_set, (list, tuple))
    ]
    return required_params, any_of_param_sets


def _params_has_path(params: dict[str, Any], dotted_path: str) -> bool:
    current: Any = params
    for token in str(dotted_path or "").split("."):
        key = str(token or "").strip()
        if not key:
            return False
        if not isinstance(current, dict) or key not in current:
            return False
        current = current.get(key)
        if current is None:
            return False
    return True


def _render_store_as_contract_error(
    required_params: list[str],
    any_of_param_sets: list[list[str]],
) -> str | None:
    if required_params:
        missing = [
            str(param).strip()
            for param in required_params
            if str(param).strip()
        ]
        if missing:
            if len(missing) == 1:
                return f"observation request requires params.{missing[0]}"
            return "observation request requires " + ", ".join(
                f"params.{name}" for name in missing
            )
    if any_of_param_sets:
        options = [
            " + ".join(f"params.{name}" for name in param_set)
            for param_set in any_of_param_sets
            if param_set
        ]
        if options:
            return "observation request requires one of: " + " or ".join(options)
    return None


def _validate_store_as_contract(
    *,
    params: dict[str, Any],
    resource_type: str,
    primitive_name: str,
) -> str | None:
    required_params, any_of_param_sets = _store_as_contract_fields(
        resource_type=resource_type,
        primitive_name=primitive_name,
    )
    missing_required = [
        param
        for param in required_params
        if not _params_has_path(params, param)
    ]
    if missing_required:
        return _render_store_as_contract_error(missing_required, [])

    if any_of_param_sets:
        has_valid_option = any(
            param_set and all(_params_has_path(params, param) for param in param_set)
            for param_set in any_of_param_sets
        )
        if not has_valid_option:
            return _render_store_as_contract_error([], any_of_param_sets)
    return None


def _sanitize_store_as_token(value: Any) -> str:
    token = "".join(
        char if str(char).isalnum() else "_"
        for char in str(value or "").strip()
    )
    while "__" in token:
        token = token.replace("__", "_")
    return token.strip("_")


def _known_observation_aliases(session_state: dict[str, Any]) -> set[str]:
    aliases = {
        str(alias).strip()
        for alias in dict(session_state.get("observation_store") or {}).keys()
        if str(alias).strip()
    }
    for raw_event in (session_state.get("observation_history") or []):
        if not isinstance(raw_event, dict):
            continue
        alias = str(raw_event.get("store_as") or "").strip()
        if alias:
            aliases.add(alias)
    for raw_fact in dict(session_state.get("observation_fact_ledger") or {}).values():
        if not isinstance(raw_fact, dict):
            continue
        for alias in (raw_fact.get("aliases") or []):
            token = str(alias).strip()
            if token:
                aliases.add(token)
    return aliases


def _default_observe_store_as(
    normalized_request: dict[str, Any],
    *,
    session_state: dict[str, Any],
    seen_aliases: set[str],
) -> str:
    fact_type = str(normalized_request.get("fact_type") or "").strip().lower()
    entity = str(normalized_request.get("entity") or "").strip()
    primitive = str(normalized_request.get("primitive") or "").strip().lower()

    if fact_type == "part_pose":
        prefix = "observed_pose"
    elif fact_type:
        prefix = _sanitize_store_as_token(fact_type) or "observed_fact"
    elif primitive:
        prefix = _sanitize_store_as_token(primitive) or "observation"
    else:
        prefix = "observation"

    entity_token = _sanitize_store_as_token(entity)
    base_alias = f"{prefix}_{entity_token}" if entity_token else prefix
    if not base_alias:
        base_alias = "observation_alias"

    used_aliases = _known_observation_aliases(session_state) | {
        str(alias).strip()
        for alias in seen_aliases
        if str(alias).strip()
    }
    if base_alias not in used_aliases:
        return base_alias

    suffix = 2
    while True:
        candidate = f"{base_alias}_{suffix}"
        if candidate not in used_aliases:
            return candidate
        suffix += 1


def _focused_observation_resource_jid(prepared_bridge_request: dict[str, Any]) -> str:
    llm_input = dict(prepared_bridge_request.get("llm_input") or {})
    allowed_surface = dict(llm_input.get("allowed_execution_surface") or {})
    focused_resource_jid = str(
        allowed_surface.get("focused_resource_jid")
        or prepared_bridge_request.get("ra_jid")
        or ""
    ).strip()
    bridge_resources = dict(prepared_bridge_request.get("bridge_resources") or {})
    if focused_resource_jid and focused_resource_jid in bridge_resources:
        return focused_resource_jid
    for row in (allowed_surface.get("resources") or []):
        if not isinstance(row, dict):
            continue
        if str(row.get("role") or "").strip().lower() != "focused":
            continue
        resource_jid = str(row.get("resource_jid") or "").strip()
        if resource_jid:
            return resource_jid
    for resource_jid in bridge_resources:
        token = str(resource_jid or "").strip()
        if token:
            return token
    return ""


def _focused_observation_resource_entry(
    prepared_bridge_request: dict[str, Any],
) -> tuple[str, dict[str, Any], str]:
    bridge_resources = dict(prepared_bridge_request.get("bridge_resources") or {})
    resource_jid = _focused_observation_resource_jid(prepared_bridge_request)
    bridge_entry = dict(bridge_resources.get(resource_jid) or {})
    resource_type = str(bridge_entry.get("resource_type") or "resource").strip() or "resource"
    return resource_jid, bridge_entry, resource_type


def _world_observation_fact_contracts(
    prepared_bridge_request: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    _, _, resource_type = _focused_observation_resource_entry(prepared_bridge_request)
    profile = get_resource_profile(resource_type or "resource")
    contracts: dict[str, dict[str, Any]] = {}
    for raw_fact_type, raw_contract in dict(profile.grounding_observation_fact_map or {}).items():
        fact_type = str(raw_fact_type or "").strip()
        if not fact_type or not isinstance(raw_contract, dict):
            continue
        primitive_name = str(raw_contract.get("primitive") or "").strip()
        entity_param = str(raw_contract.get("entity_param") or "").strip()
        if not primitive_name or not entity_param:
            continue
        _, primitive_entry, _ = _lookup_world_observation_primitive(
            prepared_bridge_request,
            primitive_name=primitive_name,
        )
        if not primitive_entry:
            continue
        output_fields = [
            str(field).strip()
            for field in (
                raw_contract.get("output_fields")
                or dict(primitive_entry.get("output_schema") or {}).keys()
            )
            if str(field).strip()
        ]
        contracts[fact_type] = {
            "fact_type": fact_type,
            "entity_kind": str(raw_contract.get("entity_kind") or "").strip() or None,
            "primitive": primitive_name,
            "entity_param": entity_param,
            "request_fields": [
                str(field).strip()
                for field in (raw_contract.get("request_fields") or ("fact_type", "entity"))
                if str(field).strip() and str(field).strip() != "store_as"
            ],
            "optional_request_fields": [
                str(field).strip()
                for field in (raw_contract.get("optional_request_fields") or ())
                if str(field).strip() and str(field).strip() != "store_as"
            ],
            "output_fields": output_fields,
        }
    return contracts


def _build_world_observation_surface(
    prepared_bridge_request: dict[str, Any],
) -> dict[str, Any]:
    _, bridge_entry, resource_type = _focused_observation_resource_entry(prepared_bridge_request)
    observation_facts: list[dict[str, Any]] = []
    for fact_type, contract in _world_observation_fact_contracts(prepared_bridge_request).items():
        row: dict[str, Any] = {
            "fact_type": fact_type,
            "request_fields": deepcopy(contract.get("request_fields") or []),
        }
        entity_kind = str(contract.get("entity_kind") or "").strip()
        if entity_kind:
            row["entity_kind"] = entity_kind
        optional_request_fields = [
            str(field).strip()
            for field in (contract.get("optional_request_fields") or [])
            if str(field).strip()
        ]
        if optional_request_fields:
            row["optional_request_fields"] = optional_request_fields
        output_fields = [
            str(field).strip()
            for field in (contract.get("output_fields") or [])
            if str(field).strip()
        ]
        if output_fields:
            row["output_fields"] = output_fields
        observation_facts.append(row)
    if observation_facts:
        return {"observation_facts": observation_facts}

    observation_primitives: list[dict[str, Any]] = []
    for primitive_entry in (bridge_entry.get("primitive_catalog") or []):
        if not isinstance(primitive_entry, dict):
            continue
        primitive_name = str(primitive_entry.get("name") or "").strip()
        if not primitive_name:
            continue
        primitive_kind = str(primitive_entry.get("primitive_kind") or "").strip().lower()
        if primitive_kind != "observe":
            continue
        if not _is_grounding_observation_primitive(
            resource_type=resource_type,
            primitive_name=primitive_name,
        ):
            continue
        if not _supports_store_as(resource_type=resource_type, primitive_name=primitive_name):
            continue
        observation_row: dict[str, Any] = {
            "name": primitive_name,
            "primitive_kind": primitive_kind,
            "required_params": [
                str(param).strip()
                for param in (primitive_entry.get("required_params") or [])
                if str(param).strip()
            ],
            "supports_store_as": True,
        }
        store_as_required_params, store_as_any_of_param_sets = _store_as_contract_fields(
            resource_type=resource_type,
            primitive_name=primitive_name,
        )
        if store_as_required_params:
            observation_row["store_as_required_params"] = store_as_required_params
        if store_as_any_of_param_sets:
            observation_row["store_as_any_of_param_sets"] = store_as_any_of_param_sets
        output_fields = [
            str(field).strip()
            for field in dict(primitive_entry.get("output_schema") or {}).keys()
            if str(field).strip()
        ]
        if output_fields:
            observation_row["output_fields"] = output_fields
        observation_primitives.append(observation_row)
    return {"observation_primitives": observation_primitives}


def _build_compact_resources(prepared_bridge_request: dict[str, Any]) -> list[dict[str, Any]]:
    llm_input = dict(prepared_bridge_request.get("llm_input") or {})
    return _build_outline_action_surface(llm_input)


def _outline_task_action_candidates(
    llm_input: dict[str, Any],
) -> tuple[list[str], dict[str, dict[str, Any]], dict[str, dict[str, dict[str, Any]]]]:
    observed_runtime_state = dict(llm_input.get("observed_runtime_state") or {})
    allowed_surface = dict(llm_input.get("allowed_execution_surface") or {})
    fault_event = dict(llm_input.get("fault_event") or {})
    modeled_gap = dict(llm_input.get("modeled_continuation_gap") or {})

    ordered_resource_jids: list[str] = []
    resources_by_jid: dict[str, dict[str, Any]] = {}
    for row in (allowed_surface.get("resources") or []):
        if not isinstance(row, dict):
            continue
        resource_jid = str(row.get("resource_jid") or "").strip()
        if not resource_jid or resource_jid in resources_by_jid:
            continue
        ordered_resource_jids.append(resource_jid)
        resources_by_jid[resource_jid] = {
            "resource_jid": resource_jid,
            "resource_type": deepcopy(row.get("resource_type")),
            "role": deepcopy(row.get("role")),
        }
    for row in (observed_runtime_state.get("resources") or []):
        if not isinstance(row, dict):
            continue
        resource_jid = str(row.get("resource_jid") or "").strip()
        if not resource_jid:
            continue
        if resource_jid not in resources_by_jid:
            ordered_resource_jids.append(resource_jid)
            resources_by_jid[resource_jid] = {
                "resource_jid": resource_jid,
                "resource_type": None,
                "role": None,
            }
        if resources_by_jid[resource_jid].get("resource_type") in (None, ""):
            resources_by_jid[resource_jid]["resource_type"] = deepcopy(
                row.get("resource_type")
            )

    modeled_action_names: set[str] = set()
    pending_task_resource_by_id: dict[str, str] = {}

    actions_by_resource: dict[str, dict[str, dict[str, Any]]] = {
        resource_jid: {} for resource_jid in ordered_resource_jids
    }

    def _merge_action(
        *,
        resource_jid: str,
        action_name: str,
        source_type: str,
        description: str = "",
        primitive_kind: str = "",
        required_params: list[str] | None = None,
    ) -> None:
        jid = str(resource_jid or "").strip()
        action = str(action_name or "").strip()
        if not jid or not action:
            return
        if jid not in actions_by_resource:
            ordered_resource_jids.append(jid)
            actions_by_resource[jid] = {}
            resources_by_jid.setdefault(
                jid,
                {
                    "resource_jid": jid,
                    "resource_type": None,
                    "role": None,
                },
            )
        entry = actions_by_resource[jid].get(action)
        if entry is None:
            entry = {
                "task_action": action,
                "source_types": [],
            }
            actions_by_resource[jid][action] = entry
        token = str(source_type or "").strip()
        if token and token not in entry["source_types"]:
            entry["source_types"].append(token)
        if description and not entry.get("description"):
            entry["description"] = description
        if primitive_kind and not entry.get("primitive_kind"):
            entry["primitive_kind"] = primitive_kind
        if required_params:
            existing = [
                str(item).strip()
                for item in (entry.get("required_params") or [])
                if str(item).strip()
            ]
            for param_name in required_params:
                token = str(param_name or "").strip()
                if token and token not in existing:
                    existing.append(token)
            if existing:
                entry["required_params"] = existing

    focused_resource_jid = str(fault_event.get("focused_resource_jid") or "").strip()
    blocked_at_function = str(fault_event.get("blocked_at_function") or "").strip()
    if focused_resource_jid and blocked_at_function:
        modeled_action_names.add(blocked_at_function)
        _merge_action(
            resource_jid=focused_resource_jid,
            action_name=blocked_at_function,
            source_type="fault_event",
        )

    for row in (modeled_gap.get("pending_nominal_tasks") or []):
        if not isinstance(row, dict):
            continue
        resource_jid = str(row.get("resource") or "").strip()
        task_action = str(row.get("function") or "").strip()
        task_id = str(row.get("id") or "").strip()
        if task_id and resource_jid:
            pending_task_resource_by_id[task_id] = resource_jid
        if not resource_jid or not task_action:
            continue
        modeled_action_names.add(task_action)
        description_parts: list[str] = []
        part_name = str(row.get("part") or "").strip()
        if part_name:
            description_parts.append(part_name)
        blocked_ids = [str(item).strip() for item in (row.get("blocked_by_condition_ids") or []) if str(item).strip()]
        if blocked_ids:
            description_parts.append("blocked continuation task")
        _merge_action(
            resource_jid=resource_jid,
            action_name=task_action,
            source_type="pending_nominal_task",
            description=" | ".join(description_parts),
        )

    for row in (modeled_gap.get("unmet_continuation_conditions") or []):
        if not isinstance(row, dict):
            continue
        task_action = str(row.get("source_function_name") or "").strip()
        if not task_action:
            continue
        modeled_action_names.add(task_action)
        candidate_resources: list[str] = []
        entity_kind = str(row.get("entity_kind") or "").strip().lower()
        entity = str(row.get("entity") or "").strip()
        if entity_kind == "resource" and entity:
            candidate_resources.append(entity)
        source_task_id = str(row.get("source_task_id") or "").strip()
        if source_task_id and source_task_id in pending_task_resource_by_id:
            candidate_resources.append(pending_task_resource_by_id[source_task_id])
        for task_id in (row.get("source_task_ids") or []):
            token = str(task_id or "").strip()
            if token and token in pending_task_resource_by_id:
                candidate_resources.append(pending_task_resource_by_id[token])
        if not candidate_resources and focused_resource_jid:
            candidate_resources.append(focused_resource_jid)
        for resource_jid in candidate_resources:
            _merge_action(
                resource_jid=resource_jid,
                action_name=task_action,
                source_type="continuation_condition",
                description=str(row.get("blocking_reason") or "").strip(),
            )

    for row in (allowed_surface.get("resources") or []):
        if not isinstance(row, dict):
            continue
        resource_jid = str(row.get("resource_jid") or "").strip()
        if not resource_jid:
            continue
        for primitive in (row.get("allowed_primitives") or []):
            if not isinstance(primitive, dict):
                continue
            action_name = str(primitive.get("name") or "").strip()
            if not action_name:
                continue
            primitive_kind = str(primitive.get("primitive_kind") or "").strip().lower()
            if action_name.startswith("compute_"):
                continue
            if (
                action_name not in modeled_action_names
                and primitive_kind not in {"pick", "place", "release", "home"}
            ):
                continue
            _merge_action(
                resource_jid=resource_jid,
                action_name=action_name,
                source_type="allowed_primitive",
                description=str(
                    primitive.get("description") or primitive.get("semantic_summary") or ""
                ).strip(),
                primitive_kind=primitive_kind,
                required_params=[
                    str(item).strip()
                    for item in (primitive.get("required_params") or [])
                    if str(item).strip()
                ],
            )

    return ordered_resource_jids, resources_by_jid, actions_by_resource


def _build_outline_action_surface(llm_input: dict[str, Any]) -> list[dict[str, Any]]:
    ordered_resource_jids, resources_by_jid, actions_by_resource = (
        _outline_task_action_candidates(llm_input)
    )
    compact_resources: list[dict[str, Any]] = []
    for resource_jid in ordered_resource_jids:
        resource_row = deepcopy(resources_by_jid.get(resource_jid) or {})
        compact_resources.append(
            {
                "resource_jid": resource_jid,
                "resource_type": deepcopy(resource_row.get("resource_type")),
                "role": deepcopy(resource_row.get("role")),
                "task_actions": [
                    deepcopy(action)
                    for action in (actions_by_resource.get(resource_jid) or {}).values()
                ],
            }
        )
    return compact_resources


def _observation_pose(payload: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    pose: dict[str, Any] = {}
    nested_pose = payload.get("pose")
    if isinstance(nested_pose, dict):
        pose.update(deepcopy(nested_pose))
    for axis in ("x", "y", "z", "qx", "qy", "qz", "qw"):
        value = payload.get(axis)
        if value is not None:
            pose[axis] = deepcopy(value)
    orientation = payload.get("orientation")
    if isinstance(orientation, dict):
        for axis in ("qx", "qy", "qz", "qw"):
            value = orientation.get(axis)
            if value is not None and axis not in pose:
                pose[axis] = deepcopy(value)
    if all(pose.get(axis) is None for axis in ("x", "y", "z")):
        return None
    return pose


def _observation_request_key(primitive_name: str, params: dict[str, Any]) -> str:
    return json.dumps(
        {
            "primitive": str(primitive_name or "").strip(),
            "params": deepcopy(params or {}),
        },
        sort_keys=True,
        default=str,
        ensure_ascii=True,
    )


def _observation_fact_key(
    fact_type: str,
    entity: str,
    scope: Any = None,
) -> str:
    payload: dict[str, Any] = {
        "fact_type": str(fact_type or "").strip(),
        "entity": str(entity or "").strip(),
    }
    if scope not in (None, "", [], {}):
        payload["scope"] = deepcopy(scope)
    return json.dumps(payload, sort_keys=True, default=str, ensure_ascii=True)


def _legacy_observation_fact_identity(
    prepared_bridge_request: dict[str, Any],
    *,
    primitive_name: str,
    params: dict[str, Any],
) -> tuple[str, str]:
    primitive_token = str(primitive_name or "").strip()
    for fact_type, contract in _world_observation_fact_contracts(prepared_bridge_request).items():
        if str(contract.get("primitive") or "").strip() != primitive_token:
            continue
        entity_param = str(contract.get("entity_param") or "").strip()
        entity = str(params.get(entity_param) or "").strip() if entity_param else ""
        if entity:
            return fact_type, entity
    return "", ""


def _known_part_names(prepared_bridge_request: dict[str, Any]) -> set[str]:
    llm_input = dict(prepared_bridge_request.get("llm_input") or {})
    return {
        str(row.get("part_name") or "").strip()
        for row in (llm_input.get("part_facts") or [])
        if isinstance(row, dict) and str(row.get("part_name") or "").strip()
    }


def _known_resource_tokens(prepared_bridge_request: dict[str, Any]) -> set[str]:
    llm_input = dict(prepared_bridge_request.get("llm_input") or {})
    observed_runtime_state = dict(llm_input.get("observed_runtime_state") or {})
    tokens: set[str] = set()
    for row in (observed_runtime_state.get("resources") or []):
        if not isinstance(row, dict):
            continue
        resource_jid = str(row.get("resource_jid") or "").strip()
        if not resource_jid:
            continue
        tokens.add(resource_jid)
        tokens.add(resource_jid.split("@", 1)[0])
    for resource_jid in dict(prepared_bridge_request.get("bridge_resources") or {}):
        token = str(resource_jid or "").strip()
        if not token:
            continue
        tokens.add(token)
        tokens.add(token.split("@", 1)[0])
    return {token for token in tokens if token}


def _resolve_observation_request(
    prepared_bridge_request: dict[str, Any],
    raw_request: dict[str, Any],
) -> tuple[dict[str, Any] | None, str | None]:
    request = dict(raw_request or {})
    fact_type = str(request.get("fact_type") or "").strip()
    entity = str(request.get("entity") or "").strip()
    reason = str(request.get("reason") or "").strip()
    scope = deepcopy(request.get("scope"))
    params = dict(request.get("params") or {})

    if fact_type:
        fact_contract = dict(
            (_world_observation_fact_contracts(prepared_bridge_request) or {}).get(fact_type) or {}
        )
        if not fact_contract:
            return None, f"world observation fact {fact_type!r} is not allowed"
        primitive_name = str(fact_contract.get("primitive") or "").strip()
        entity_param = str(fact_contract.get("entity_param") or "").strip()
        if not entity:
            return None, f"observe request fact {fact_type!r} requires entity"
        if not primitive_name or not entity_param:
            return None, f"world observation fact {fact_type!r} is not executable"
        entity_kind = str(fact_contract.get("entity_kind") or "").strip().lower()
        if entity_kind == "part":
            known_part_names = _known_part_names(prepared_bridge_request)
            if known_part_names and entity not in known_part_names:
                known_resource_tokens = _known_resource_tokens(prepared_bridge_request)
                if entity in known_resource_tokens:
                    return None, (
                        f"observe request fact {fact_type!r} expects a part entity, but "
                        f"got resource {entity!r}"
                    )
                return None, (
                    f"observe request fact {fact_type!r} has unknown part entity {entity!r}; "
                    f"known parts are {sorted(known_part_names)!r}"
                )
        param_entity = str(params.get(entity_param) or "").strip()
        if param_entity and param_entity != entity:
            return None, (
                f"observe request fact {fact_type!r} has entity {entity!r}, but "
                f"params.{entity_param}={param_entity!r}"
            )
        params[entity_param] = entity
        resource_jid, primitive_entry, resource_type = _lookup_world_observation_primitive(
            prepared_bridge_request,
            primitive_name=primitive_name,
        )
        if not primitive_entry:
            return None, f"world observation fact {fact_type!r} is not executable"
        return {
            "fact_type": fact_type,
            "entity": entity,
            "entity_kind": str(fact_contract.get("entity_kind") or "").strip() or None,
            "scope": scope,
            "reason": reason,
            "primitive": primitive_name,
            "params": deepcopy(params),
            "store_as": "",
            "resource_jid": resource_jid,
            "resource_type": resource_type,
            "fact_key": _observation_fact_key(fact_type, entity, scope),
        }, None

    primitive_name = str(request.get("primitive") or "").strip()
    if not primitive_name:
        return None, "observe_requests must include fact_type and entity"
    resource_jid, primitive_entry, resource_type = _lookup_world_observation_primitive(
        prepared_bridge_request,
        primitive_name=primitive_name,
    )
    if not primitive_entry:
        return None, f"world observation primitive {primitive_name!r} is not allowed"
    legacy_fact_type, legacy_entity = _legacy_observation_fact_identity(
        prepared_bridge_request,
        primitive_name=primitive_name,
        params=params,
    )
    return {
        "fact_type": legacy_fact_type,
        "entity": legacy_entity,
        "entity_kind": "part" if legacy_fact_type == "part_pose" and legacy_entity else None,
        "scope": scope,
        "reason": reason,
        "primitive": primitive_name,
        "params": deepcopy(params),
        "store_as": "",
        "resource_jid": resource_jid,
        "resource_type": resource_type,
        "fact_key": (
            _observation_fact_key(legacy_fact_type, legacy_entity, scope)
            if legacy_fact_type and legacy_entity
            else ""
        ),
    }, None


def _render_prior_fact_message(prior_fact: dict[str, Any]) -> str:
    fact_type = str(prior_fact.get("fact_type") or "").strip()
    entity = str(prior_fact.get("entity") or "").strip()
    message = "observation fact already succeeded earlier"
    if fact_type and entity:
        message = f"observation fact {fact_type!r} for {entity!r} already succeeded earlier"
    return message


def _is_terminal_grounding_observe_error(error: str) -> bool:
    token = str(error or "").strip().lower()
    if not token:
        return False
    terminal_prefixes = (
        "observe request fact ",
        "observe_requests must include fact_type and entity",
        "world observation fact ",
        "world observation primitive ",
        "observe request ",
        "observation request requires ",
    )
    return token.startswith(terminal_prefixes)


def _is_redundant_grounding_observe_error(error: str) -> bool:
    token = str(error or "").strip().lower()
    return bool(token) and "already succeeded earlier" in token


def _grounding_observation_fulfillment_status(
    session_state: dict[str, Any],
) -> dict[str, Any]:
    observation_fact_ledger = {
        str(fact_key).strip(): dict(row)
        for fact_key, row in dict(session_state.get("observation_fact_ledger") or {}).items()
        if str(fact_key).strip() and isinstance(row, dict)
    }
    prior_requests: list[dict[str, Any]] = []
    for raw_turn in (session_state.get("turns") or []):
        if not isinstance(raw_turn, dict):
            continue
        if str(raw_turn.get("phase") or "").strip().lower() != "grounding":
            continue
        response = dict(raw_turn.get("raw_response") or {})
        for raw_request in (response.get("observe_requests") or []):
            if not isinstance(raw_request, dict):
                continue
            fact_type = str(raw_request.get("fact_type") or "").strip()
            entity = str(raw_request.get("entity") or "").strip()
            scope = deepcopy(raw_request.get("scope"))
            fact_key = (
                _observation_fact_key(fact_type, entity, scope)
                if fact_type and entity
                else ""
            )
            fact_row = dict(observation_fact_ledger.get(fact_key) or {}) if fact_key else {}
            is_fulfilled = bool(
                fact_row
                and str(fact_row.get("validity") or "current").strip().lower() != "stale"
            )
            row = {
                "fact_type": fact_type,
                "entity": entity,
                "requested_at_turn": int(raw_turn.get("turn_index") or 0),
                "status": "fulfilled" if is_fulfilled else "pending",
            }
            if scope not in (None, "", [], {}):
                row["scope"] = scope
            prior_requests.append(
                row
            )
    unfulfilled = [row for row in prior_requests if row.get("status") != "fulfilled"]
    return {
        "prior_requests": prior_requests,
        "fulfilled_count": len(prior_requests) - len(unfulfilled),
        "unfulfilled_count": len(unfulfilled),
        "has_session_observations": bool(
            observation_fact_ledger or dict(session_state.get("observation_store") or {})
        ),
    }


def _grounding_contract_failure_empty_observe(
    *,
    session_state: dict[str, Any],
) -> dict[str, Any]:
    fulfillment = _grounding_observation_fulfillment_status(session_state)
    return {
        "status": "failed",
        "reason": "observe_without_requests",
        "observation_fulfillment": deepcopy(fulfillment),
    }


def _grounding_contract_failure_redundant_observe(
    *,
    prepared_bridge_request: dict[str, Any],
    session_state: dict[str, Any],
    observe_requests: list[dict[str, Any]],
) -> dict[str, Any]:
    fulfillment = _grounding_observation_fulfillment_status(session_state)
    observation_fact_ledger = {
        str(fact_key).strip(): dict(row)
        for fact_key, row in dict(session_state.get("observation_fact_ledger") or {}).items()
        if str(fact_key).strip() and isinstance(row, dict)
    }
    fulfilled_requests: list[dict[str, Any]] = []
    for raw_request in observe_requests:
        normalized_request, resolve_error = _resolve_observation_request(
            prepared_bridge_request,
            dict(raw_request or {}),
        )
        if resolve_error is not None or not isinstance(normalized_request, dict):
            continue
        fact_key = str(normalized_request.get("fact_key") or "").strip()
        prior_fact = dict(observation_fact_ledger.get(fact_key) or {}) if fact_key else {}
        if not prior_fact:
            continue
        if str(prior_fact.get("validity") or "current").strip().lower() == "stale":
            continue
        row = {
            "fact_type": str(normalized_request.get("fact_type") or "").strip(),
            "entity": str(normalized_request.get("entity") or "").strip(),
        }
        scope = deepcopy(normalized_request.get("scope"))
        if scope not in (None, "", [], {}):
            row["scope"] = scope
        fulfilled_requests.append(row)
    return {
        "status": "failed",
        "reason": "observe_already_fulfilled",
        "fulfilled_requests": fulfilled_requests,
        "observation_fulfillment": deepcopy(fulfillment),
    }


def _overlay_session_observations_on_llm_input(
    llm_input: dict[str, Any],
    session_state: dict[str, Any],
) -> dict[str, Any]:
    updated = deepcopy(llm_input or {})
    part_facts = [
        deepcopy(row)
        for row in (updated.get("part_facts") or [])
        if isinstance(row, dict)
    ]
    if not part_facts:
        return updated

    latest_part_observations: dict[str, dict[str, Any]] = {}
    for raw_fact in dict(session_state.get("observation_fact_ledger") or {}).values():
        if not isinstance(raw_fact, dict):
            continue
        if str(raw_fact.get("fact_type") or "").strip() != "part_pose":
            continue
        if str(raw_fact.get("validity") or "current").strip().lower() == "stale":
            continue
        output = dict(raw_fact.get("output") or {})
        part_name = str(raw_fact.get("entity") or output.get("part_name") or "").strip()
        pose = _observation_pose(output)
        if not part_name or pose is None:
            continue
        latest_part_observations[part_name] = {
            "pose": pose,
            "aliases": [
                str(alias).strip()
                for alias in (raw_fact.get("aliases") or [])
                if str(alias).strip()
            ],
            "turn_index": int(raw_fact.get("turn_index") or 0),
            "primitive": str(raw_fact.get("primitive") or "").strip(),
            "fact_type": "part_pose",
            "current_location": deepcopy(output.get("current_location")),
            "current_holder_resource_jid": (
                str(output.get("current_holder_resource_jid") or "").strip() or None
            ),
        }

    for row in (session_state.get("observation_history") or []):
        if not isinstance(row, dict):
            continue
        if str(row.get("primitive") or "").strip() != "detect_parts":
            continue
        output = dict(row.get("output") or {})
        params = dict(row.get("params") or {})
        part_name = str(output.get("part_name") or params.get("part_name") or "").strip()
        pose = _observation_pose(output)
        if not part_name or pose is None:
            continue
        latest_part_observations[part_name] = {
            "pose": pose,
            "aliases": [str(row.get("store_as") or "").strip()],
            "turn_index": int(row.get("turn_index") or 0),
            "primitive": "detect_parts",
            "fact_type": str(row.get("fact_type") or "part_pose").strip() or "part_pose",
            "current_location": deepcopy(output.get("current_location")),
            "current_holder_resource_jid": (
                str(output.get("current_holder_resource_jid") or "").strip() or None
            ),
        }

    for alias, payload in dict(session_state.get("observation_store") or {}).items():
        if not isinstance(payload, dict):
            continue
        part_name = str(payload.get("part_name") or "").strip()
        pose = _observation_pose(payload)
        if not part_name or pose is None or part_name in latest_part_observations:
            continue
        latest_part_observations[part_name] = {
            "pose": pose,
            "aliases": [str(alias or "").strip()],
            "turn_index": 0,
            "primitive": "detect_parts",
            "fact_type": "part_pose",
            "current_location": deepcopy(payload.get("current_location")),
            "current_holder_resource_jid": (
                str(payload.get("current_holder_resource_jid") or "").strip() or None
            ),
        }

    if not latest_part_observations:
        return updated

    for row in part_facts:
        part_name = str(row.get("part_name") or "").strip()
        observation_info = latest_part_observations.get(part_name)
        if not isinstance(observation_info, dict):
            continue
        pose = dict(observation_info.get("pose") or {})
        if not pose:
            continue
        row["observed_pose"] = deepcopy(pose)
        row["location_basis"] = "session_observation"
        row["observation_status"] = "observed"
        row["observed_in_session"] = True
        row["observed_by"] = str(observation_info.get("primitive") or "").strip()
        row["observed_by_primitive"] = str(observation_info.get("primitive") or "").strip()
        observed_fact_type = str(observation_info.get("fact_type") or "").strip()
        if observed_fact_type:
            row["observed_fact_type"] = observed_fact_type
        observed_aliases = [
            str(alias).strip()
            for alias in (observation_info.get("aliases") or [])
            if str(alias).strip()
        ]
        observed_store_as = observed_aliases[-1] if observed_aliases else ""
        if observed_store_as:
            row["observed_store_as"] = observed_store_as
        if observed_aliases:
            row["observed_aliases"] = observed_aliases
        observed_turn_index = int(observation_info.get("turn_index") or 0)
        if observed_turn_index > 0:
            row["observed_turn_index"] = observed_turn_index
        observed_current_location = observation_info.get("current_location")
        if observed_current_location not in (None, ""):
            row["current_location"] = deepcopy(observed_current_location)
        observed_holder_resource_jid = str(
            observation_info.get("current_holder_resource_jid") or ""
        ).strip()
        if observed_holder_resource_jid:
            row["current_holder_resource_jid"] = observed_holder_resource_jid

    updated["part_facts"] = part_facts
    return updated


def _record_observation_fact(
    session_state: dict[str, Any],
    observation_row: dict[str, Any],
) -> None:
    fact_key = str(observation_row.get("fact_key") or "").strip()
    if not fact_key:
        return
    ledger = dict(session_state.get("observation_fact_ledger") or {})
    existing = dict(ledger.get(fact_key) or {})
    aliases = [
        str(alias).strip()
        for alias in (existing.get("aliases") or [])
        if str(alias).strip()
    ]
    store_as = str(observation_row.get("store_as") or "").strip()
    if store_as and store_as not in aliases:
        aliases.append(store_as)
    ledger[fact_key] = {
        "fact_key": fact_key,
        "fact_type": str(observation_row.get("fact_type") or "").strip(),
        "entity": str(observation_row.get("entity") or "").strip(),
        "entity_kind": str(observation_row.get("entity_kind") or "").strip() or None,
        "scope": deepcopy(observation_row.get("scope")),
        "primitive": str(observation_row.get("primitive") or "").strip(),
        "params": deepcopy(observation_row.get("params") or {}),
        "output": deepcopy(observation_row.get("output") or {}),
        "turn_index": int(observation_row.get("turn_index") or 0),
        "validity": "current",
        "freshness": "current_session",
        "aliases": aliases,
    }
    session_state["observation_fact_ledger"] = ledger


def _is_pose_in_workspace(
    pose: dict[str, Any],
    bounds: dict[str, Any],
) -> tuple[bool, list[str]]:
    """Check if a pose falls within workspace bounds. Returns (is_inside, violations)."""
    violations: list[str] = []
    for axis in ("x", "y", "z"):
        val = pose.get(axis)
        if val is None:
            continue
        try:
            coord = float(val)
        except (TypeError, ValueError):
            continue
        lo = bounds.get(f"{axis}_min_m")
        hi = bounds.get(f"{axis}_max_m")
        if lo is not None and coord < float(lo):
            violations.append(f"{axis}={coord:.4f} < {axis}_min_m={float(lo):.4f}")
        if hi is not None and coord > float(hi):
            violations.append(f"{axis}={coord:.4f} > {axis}_max_m={float(hi):.4f}")
    return len(violations) == 0, violations


def _outline_task_closes_condition_ids(task: dict[str, Any]) -> list[str]:
    return [
        str(item).strip()
        for item in (task.get("closes_condition_ids") or [])
        if str(item).strip()
    ]


def _task_findings_block_projected_state(findings: list[dict[str, Any]]) -> bool:
    return bool(findings)


def _finding_claimed_condition_ids(finding: dict[str, Any]) -> list[str]:
    claimed_condition_ids = [
        str(item).strip()
        for item in (finding.get("claimed_condition_ids") or [])
        if str(item).strip()
    ]
    if claimed_condition_ids:
        return claimed_condition_ids
    condition_id = str(finding.get("condition_id") or "").strip()
    return [condition_id] if condition_id else []


def _extract_string_list(finding: dict[str, Any], key: str) -> list[str]:
    """Extract a list of non-empty stripped strings from a finding dict key."""
    return [
        str(v).strip()
        for v in (finding.get(key) or [])
        if str(v).strip()
    ]


def _outline_validation_context(
    llm_input: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    observed_runtime_state = dict(llm_input.get("observed_runtime_state") or {})
    resources_by_jid: dict[str, dict[str, Any]] = {}
    for resource in (observed_runtime_state.get("resources") or []):
        if not isinstance(resource, dict):
            continue
        jid = str(resource.get("resource_jid") or "").strip()
        if jid:
            resources_by_jid[jid] = resource

    parts_by_name: dict[str, dict[str, Any]] = {}
    for part in (llm_input.get("part_facts") or []):
        if not isinstance(part, dict):
            continue
        name = str(part.get("part_name") or "").strip()
        if name:
            parts_by_name[name] = part
    return resources_by_jid, parts_by_name


def _resource_constraint_finding(
    *,
    task: dict[str, Any],
    constraint_code: str,
    reason: str,
    resource_jid: str = "",
    part_name: str = "",
    evidence: dict[str, Any] | None = None,
    guard: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "task_id": str(task.get("outline_id") or "").strip(),
        "resource_jid": resource_jid or None,
        "part_name": part_name or None,
        "pose_source": "resource_feasibility",
        "pose": deepcopy(dict(evidence or {}).get("checked_pose")),
        "workspace_bounds": deepcopy(dict(evidence or {}).get("workspace_bounds")),
        "failed_axes": [constraint_code],
        "constraint_owner": "resource",
        "constraint_family": "resource_feasibility",
        "constraint_code": constraint_code,
        "reason": reason,
        "guard": deepcopy(guard),
        "evidence": deepcopy(evidence or {}),
    }


def _task_with_grounded_defaults(
    task: dict[str, Any],
    grounded_action: dict[str, Any],
) -> dict[str, Any]:
    normalized_task = deepcopy(task or {})
    if grounded_action.get("resource_jid") and not str(normalized_task.get("resource_jid") or "").strip():
        normalized_task["resource_jid"] = str(grounded_action.get("resource_jid") or "").strip()
    if grounded_action.get("part_name") and not str(normalized_task.get("part_name") or "").strip():
        normalized_task["part_name"] = str(grounded_action.get("part_name") or "").strip()
    return normalized_task


def _resource_part_context(
    *,
    task: dict[str, Any],
    grounded_action: dict[str, Any],
    resource_row: dict[str, Any],
    parts_by_name: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    part_name = str(grounded_action.get("part_name") or "").strip()
    action_target = dict(grounded_action.get("target") or {})
    part_context = deepcopy(dict(parts_by_name.get(part_name) or {}))
    part_context["target"] = deepcopy(action_target)
    part_context["resource_held_part"] = str(resource_row.get("held_part") or "").strip() or None
    part_context["resource_gripper_state"] = str(
        resource_row.get("gripper_state") or ""
    ).strip() or None
    part_context["named_pose"] = str(action_target.get("named_pose") or "").strip() or None
    if "pose" in action_target:
        part_context["pose"] = deepcopy(action_target.get("pose"))
    return part_context


def _validate_outline_task_resource_constraints(
    *,
    planner: Any | None,
    task: dict[str, Any],
    grounded_action: dict[str, Any],
    resources_by_jid: dict[str, dict[str, Any]],
    parts_by_name: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    resource_jid = str(grounded_action.get("resource_jid") or "").strip()
    part_name = str(grounded_action.get("part_name") or "").strip()
    resource_row = dict(resources_by_jid.get(resource_jid) or {})
    if not resource_jid or not resource_row:
        return [
            _resource_constraint_finding(
                task=task,
                constraint_code="resource_unavailable",
                reason=f"resource '{resource_jid or 'unknown'}' is not available in current bridge state",
                resource_jid=resource_jid,
                part_name=part_name,
                guard={
                    "kind": "resource_not_available",
                    "resource_jid": resource_jid,
                }
                if resource_jid
                else None,
            )
        ]

    resolver = getattr(planner, "_resource_by_jid", None) if planner is not None else None
    resource_agent = resolver(resource_jid) if callable(resolver) else None
    oracle = getattr(resource_agent, "bridge_feasibility_oracle", None)
    if not callable(oracle):
        return []

    # Planning-time feasibility should be checked against the symbolic pre-task
    # bridge state, not the live agent's current runtime state.
    resource_snapshot = deepcopy(resource_row)
    get_bridge_snapshot = getattr(resource_agent, "get_bridge_snapshot", None)
    if callable(get_bridge_snapshot):
        try:
            maybe_snapshot = get_bridge_snapshot()
        except Exception:
            maybe_snapshot = {}
        if isinstance(maybe_snapshot, dict):
            for field_name in (
                "workspace_bounds",
                "available_named_poses",
                "bridge_adapter",
                "resource_type",
                "role",
            ):
                if field_name not in resource_snapshot and field_name in maybe_snapshot:
                    resource_snapshot[field_name] = deepcopy(maybe_snapshot.get(field_name))

    try:
        oracle_result = oracle(
            operation_kind=str(grounded_action.get("operation_kind") or "").strip(),
            part_name=part_name or None,
            part_context=_resource_part_context(
                task=task,
                grounded_action=grounded_action,
                resource_row=resource_row,
                parts_by_name=parts_by_name,
            ),
            bridge_snapshot=resource_snapshot,
            grounded_action=deepcopy(grounded_action),
        )
    except Exception as exc:
        return [
            _resource_constraint_finding(
                task=task,
                constraint_code="resource_unavailable",
                reason=f"resource feasibility oracle failed: {exc}",
                resource_jid=resource_jid,
                part_name=part_name,
            )
        ]

    result = dict(oracle_result or {})
    if bool(result.get("allowed", True)):
        return []
    constraint_code = str(result.get("constraint_code") or "").strip() or "resource_unavailable"
    reason = str(result.get("reason") or "").strip() or "resource feasibility rejected the grounded action"
    return [
        _resource_constraint_finding(
            task=task,
            constraint_code=constraint_code,
            reason=reason,
            resource_jid=resource_jid,
            part_name=part_name,
            evidence=dict(result.get("evidence") or {}),
            guard=dict(result.get("guard") or {}),
        )
    ]


def _outline_part_readiness_blockers(
    *,
    resource_jid: str,
    part_name: str,
    resources_by_jid: dict[str, dict[str, Any]],
    parts_by_name: dict[str, dict[str, Any]],
) -> list[str]:
    blockers: list[str] = []
    resource_row = dict(resources_by_jid.get(resource_jid) or {})
    part_row = dict(parts_by_name.get(part_name) or {})
    held_part = str(resource_row.get("held_part") or "").strip()
    if held_part and held_part != part_name:
        blockers.append(f"resource_holds_part:{held_part}")
    gripper_state = str(resource_row.get("gripper_state") or "").strip().lower()
    if not held_part and gripper_state == "closed":
        blockers.append("gripper_closed_without_target_part")
    current_holder = str(part_row.get("current_holder_resource_jid") or "").strip()
    if current_holder and current_holder != resource_jid:
        blockers.append(f"part_currently_held_by:{current_holder}")
    return blockers


def _current_recovery_gap_condition_status(
    *,
    resources_by_jid: dict[str, dict[str, Any]],
    parts_by_name: dict[str, dict[str, Any]],
    llm_input: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    modeled_gap = dict(llm_input.get("modeled_continuation_gap") or {})
    fault_event = dict(llm_input.get("fault_event") or {})
    fallback_parts = [
        str(item).strip()
        for item in (fault_event.get("affected_part_names") or [])
        if str(item).strip()
    ]
    condition_rows: list[dict[str, Any]] = []
    unmet_condition_ids: list[str] = []
    cleared_condition_ids: list[str] = []
    for raw_condition in (modeled_gap.get("unmet_continuation_conditions") or []):
        if not isinstance(raw_condition, dict):
            continue
        condition_row = deepcopy(raw_condition)
        condition_id = str(condition_row.get("condition_id") or "").strip()
        is_cleared = _continuation_condition_cleared(
            condition_row,
            resources_by_jid=resources_by_jid,
            parts_by_name=parts_by_name,
            fallback_parts=fallback_parts,
        )
        condition_row["currently_cleared"] = is_cleared
        condition_rows.append(condition_row)
        if not condition_id:
            continue
        if is_cleared:
            cleared_condition_ids.append(condition_id)
        else:
            unmet_condition_ids.append(condition_id)
    return condition_rows, unmet_condition_ids, cleared_condition_ids


def _compact_recovery_gap_resources(
    resources_by_jid: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    compact_rows: list[dict[str, Any]] = []
    for resource_jid, raw_row in resources_by_jid.items():
        row = dict(raw_row or {})
        compact_rows.append(
            {
                "resource_jid": resource_jid,
                "current_state": deepcopy(row.get("current_state")),
                "held_part": deepcopy(row.get("held_part")),
                "gripper_state": deepcopy(row.get("gripper_state")),
                "current_location": deepcopy(row.get("current_location")),
                "current_pose": deepcopy(row.get("current_pose")),
            }
        )
    return compact_rows


def _compact_recovery_gap_parts(
    parts_by_name: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    compact_rows: list[dict[str, Any]] = []
    for part_name, raw_row in parts_by_name.items():
        row = dict(raw_row or {})
        compact_rows.append(
            {
                "part_name": part_name,
                "current_state": deepcopy(row.get("current_state")),
                "current_holder_resource_jid": deepcopy(row.get("current_holder_resource_jid")),
                "current_location": deepcopy(row.get("current_location") or row.get("location")),
                "observed_pose": deepcopy(row.get("observed_pose")),
                "goal_location": deepcopy(row.get("goal_location")),
                "pending_nominal_task_ids": deepcopy(row.get("pending_nominal_task_ids") or []),
            }
        )
    return compact_rows


def _build_recovery_gap_state(
    llm_input: dict[str, Any],
    *,
    resources_by_jid: dict[str, dict[str, Any]] | None = None,
    parts_by_name: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if resources_by_jid is None or parts_by_name is None:
        resources_by_jid, parts_by_name = _outline_validation_context(llm_input)
    condition_rows, unmet_condition_ids, _ = _current_recovery_gap_condition_status(
        resources_by_jid=resources_by_jid,
        parts_by_name=parts_by_name,
        llm_input=llm_input,
    )
    modeled_gap = dict(llm_input.get("modeled_continuation_gap") or {})
    blocked_nominal_tasks = [
        deepcopy(row)
        for row in (modeled_gap.get("pending_nominal_tasks") or [])
        if isinstance(row, dict) and list(row.get("blocked_by_condition_ids") or [])
    ]
    return {
        "resources": _compact_recovery_gap_resources(resources_by_jid),
        "parts": _compact_recovery_gap_parts(parts_by_name),
        "required_condition_ids_to_clear": deepcopy(unmet_condition_ids),
        "unmet_continuation_conditions": [
            deepcopy(row)
            for row in condition_rows
            if not bool(row.get("currently_cleared"))
        ],
        "blocked_nominal_tasks": blocked_nominal_tasks,
    }


def _build_grounded_feasibility_facts(
    llm_input: dict[str, Any],
    *,
    resources_by_jid: dict[str, dict[str, Any]] | None = None,
    parts_by_name: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    if resources_by_jid is None or parts_by_name is None:
        resources_by_jid, parts_by_name = _outline_validation_context(llm_input)
    observed_runtime_state = dict(llm_input.get("observed_runtime_state") or {})
    ordered_resource_jids = [
        str(row.get("resource_jid") or "").strip()
        for row in (observed_runtime_state.get("resources") or [])
        if isinstance(row, dict) and str(row.get("resource_jid") or "").strip()
    ]
    fault_event = dict(llm_input.get("fault_event") or {})
    target_part_names = [
        str(item).strip()
        for item in (fault_event.get("affected_part_names") or [])
        if str(item).strip()
    ]
    if not target_part_names:
        target_part_names = list(parts_by_name)
    facts: list[dict[str, Any]] = []
    for part_name in target_part_names:
        part_row = dict(parts_by_name.get(part_name) or {})
        observed_pose = dict(part_row.get("observed_pose") or {})
        if not observed_pose:
            continue
        resource_evidence: list[dict[str, Any]] = []
        for resource_jid in ordered_resource_jids:
            resource_row = dict(resources_by_jid.get(resource_jid) or {})
            bounds = dict(resource_row.get("workspace_bounds") or {})
            if bounds:
                workspace_contains_observed_pose, workspace_violations = _is_pose_in_workspace(
                    observed_pose,
                    bounds,
                )
            else:
                workspace_contains_observed_pose = False
                workspace_violations = ["missing_workspace_bounds"]
            readiness_blockers = (
                _outline_part_readiness_blockers(
                    resource_jid=resource_jid,
                    part_name=part_name,
                    resources_by_jid=resources_by_jid,
                    parts_by_name=parts_by_name,
                )
                if workspace_contains_observed_pose
                else []
            )
            resource_evidence.append(
                {
                    "resource_jid": resource_jid,
                    "workspace_bounds": deepcopy(bounds),
                    "workspace_contains_observed_pose": workspace_contains_observed_pose,
                    "workspace_violations": deepcopy(workspace_violations),
                    "currently_ready_to_acquire": bool(
                        workspace_contains_observed_pose and not readiness_blockers
                    ),
                    "readiness_blockers": deepcopy(readiness_blockers),
                }
            )
        facts.append(
            {
                "part_name": part_name,
                "pose_source": "observed_pose",
                "pose": deepcopy(observed_pose),
                "resource_evidence": resource_evidence,
            }
        )
    return facts


def _outline_action_lookup(
    llm_input: dict[str, Any],
) -> dict[str, dict[str, dict[str, Any]]]:
    _, _, actions_by_resource = _outline_task_action_candidates(llm_input)
    return actions_by_resource


def _outline_task_explicit_pose_targets(
    task: dict[str, Any],
) -> list[tuple[str, dict[str, Any]]]:
    targets: list[tuple[str, dict[str, Any]]] = []
    for state_key in ("expected_start_state", "expected_end_state"):
        state = task.get(state_key)
        if not isinstance(state, dict):
            continue
        pose = state.get("position") or state.get("pose") or state.get("current_pose")
        if not isinstance(pose, dict) or "x" not in pose:
            continue
        targets.append((state_key, dict(pose)))
    return targets


def _first_non_empty_state_value(state: dict[str, Any], *field_names: str) -> Any:
    for field_name in field_names:
        if field_name not in state:
            continue
        value = state.get(field_name)
        if value in (None, "", [], {}):
            continue
        return deepcopy(value)
    return None


def _outline_state_resource_state_token(state: dict[str, Any]) -> str:
    return str(
        _first_non_empty_state_value(state, "current_state", "state", "resource_state") or ""
    ).strip()


def _outline_state_part_state_token(
    state: dict[str, Any],
    *,
    candidate_part_name: str,
) -> str:
    state_part_name = str(state.get("part_name") or "").strip()
    if state_part_name and candidate_part_name and state_part_name != candidate_part_name:
        return ""
    token = str(
        _first_non_empty_state_value(state, "current_state", "state") or ""
    ).strip()
    if _is_part_lifecycle_state(token):
        return token
    return ""


def _outline_state_location_token(state: dict[str, Any]) -> str:
    return str(
        _first_non_empty_state_value(
            state,
            "part_location",
            "location",
            "current_location",
            "named_pose",
        )
        or ""
    ).strip()


def _outline_state_pose_value(state: dict[str, Any]) -> dict[str, Any] | None:
    pose = (
        state.get("position")
        or state.get("pose")
        or state.get("current_pose")
    )
    if not isinstance(pose, dict) or "x" not in pose:
        return None
    return dict(pose)


def _outline_resource_named_pose_token(
    *,
    state: dict[str, Any],
    action_target: dict[str, Any],
) -> str:
    return str(
        _first_non_empty_state_value(state, "named_pose", "location", "current_location")
        or action_target.get("named_pose")
        or ""
    ).strip()


def _outline_task_part_references(task: dict[str, Any]) -> list[str]:
    part_names: list[str] = []
    for candidate in (task.get("part_name"),):
        token = str(candidate or "").strip()
        if token and token not in part_names:
            part_names.append(token)
    for state_key in ("expected_start_state", "expected_end_state"):
        state = task.get(state_key)
        if not isinstance(state, dict):
            continue
        for field_name in ("part_name", "held_part"):
            token = str(state.get(field_name) or "").strip()
            if token and token not in part_names:
                part_names.append(token)
    return part_names


def _dedupe_outline_part_candidates(part_names: list[str]) -> list[str]:
    deduped: list[str] = []
    for part_name in part_names:
        token = str(part_name or "").strip()
        if token and token not in deduped:
            deduped.append(token)
    return deduped


def _outline_part_names_matching_location(
    *,
    location_token: str,
    parts_by_name: dict[str, dict[str, Any]],
    include_current: bool = True,
    include_goal: bool = True,
) -> list[str]:
    normalized_location = str(location_token or "").strip()
    if not normalized_location or normalized_location == "observed_pose":
        return []
    matches: list[str] = []
    for part_name, raw_row in parts_by_name.items():
        row = dict(raw_row or {})
        current_location = str(
            row.get("current_location") or row.get("location") or ""
        ).strip()
        goal_location = str(row.get("goal_location") or "").strip()
        if include_current and current_location and current_location == normalized_location:
            matches.append(part_name)
            continue
        if include_goal and goal_location and goal_location == normalized_location:
            matches.append(part_name)
    return _dedupe_outline_part_candidates(matches)


def _outline_task_part_binding(
    task: dict[str, Any],
    *,
    parts_by_name: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    explicit_candidates = _dedupe_outline_part_candidates(
        [
            str(task.get("part_name") or "").strip(),
            str(dict(task.get("expected_start_state") or {}).get("part_name") or "").strip(),
            str(dict(task.get("expected_end_state") or {}).get("part_name") or "").strip(),
        ]
    )
    candidates = list(explicit_candidates)
    if not candidates:
        candidates.extend(_outline_task_part_references(task))
    if not explicit_candidates:
        action_target = _outline_task_action_target(task)
        source_matches = _outline_part_names_matching_location(
            location_token=str(action_target.get("source_location") or "").strip(),
            parts_by_name=parts_by_name,
            include_current=True,
            include_goal=False,
        )
        if len(source_matches) == 1:
            candidates.extend(source_matches)
        target_matches = _outline_part_names_matching_location(
            location_token=str(action_target.get("target_location") or "").strip(),
            parts_by_name=parts_by_name,
            include_current=False,
            include_goal=True,
        )
        if len(target_matches) == 1:
            candidates.extend(target_matches)
        for state_key in ("expected_start_state", "expected_end_state"):
            state = dict(task.get(state_key) or {})
            location_matches = _outline_part_names_matching_location(
                location_token=_outline_state_location_token(state),
                parts_by_name=parts_by_name,
                include_current=True,
                include_goal=True,
            )
            if len(location_matches) == 1:
                candidates.extend(location_matches)
    deduped_candidates = _dedupe_outline_part_candidates(candidates)
    explicit_part_name = str(task.get("part_name") or "").strip()
    effective_part_name = explicit_part_name
    if not effective_part_name and len(deduped_candidates) == 1:
        effective_part_name = deduped_candidates[0]
    return {
        "candidate_part_names": deduped_candidates,
        "effective_part_name": effective_part_name,
        "is_ambiguous": len(deduped_candidates) > 1,
        "is_unbound": not deduped_candidates,
    }



def _outline_task_default_resource_pose_target(
    resource: dict[str, Any],
) -> tuple[str, dict[str, Any]] | None:
    pose = dict(resource.get("current_pose") or {})
    if not isinstance(pose, dict) or "x" not in pose:
        return None
    return "resource_current_pose", pose


def _outline_task_action_target(task: dict[str, Any]) -> dict[str, Any]:
    action_target = task.get("action_target")
    if not isinstance(action_target, dict):
        return {}
    return dict(action_target)


def _outline_task_symbolic_anchors(task: dict[str, Any]) -> list[tuple[str, str]]:
    anchors: list[tuple[str, str]] = []
    action_target = _outline_task_action_target(task)
    for field_name in ("target_location", "source_location", "named_pose"):
        token = str(action_target.get(field_name) or "").strip()
        if token:
            anchors.append((f"action_target.{field_name}", token))
    for state_key in ("expected_start_state", "expected_end_state"):
        state = task.get(state_key)
        if not isinstance(state, dict):
            continue
        for field_name in ("position", "pose", "current_pose", "location", "current_location", "named_pose"):
            value = state.get(field_name)
            token = str(value or "").strip() if not isinstance(value, dict) else ""
            if token:
                anchors.append((f"{state_key}.{field_name}", token))
    return anchors


def _outline_task_state_transition_fields(task: dict[str, Any]) -> list[str]:
    start_state = dict(task.get("expected_start_state") or {})
    end_state = dict(task.get("expected_end_state") or {})
    changed_fields: list[str] = []
    significant_fields = (
        "current_state",
        "state",
        "resource_state",
        "gripper_state",
        "held_part",
        "part_name",
        "current_holder_resource_jid",
        "location",
        "current_location",
        "part_location",
        "position",
        "pose",
        "current_pose",
        "named_pose",
    )
    for field_name in significant_fields:
        start_value = deepcopy(start_state.get(field_name))
        end_value = deepcopy(end_state.get(field_name))
        if start_value in (None, "", [], {}) and end_value in (None, "", [], {}):
            continue
        if start_value != end_value:
            changed_fields.append(field_name)
    return changed_fields


def _outline_task_has_physical_anchor(
    task: dict[str, Any],
    *,
    referenced_parts: list[str],
    explicit_pose_targets: list[tuple[str, dict[str, Any]]],
    state_transition_fields: list[str],
) -> bool:
    if referenced_parts:
        return True
    if explicit_pose_targets:
        return True
    if _outline_task_symbolic_anchors(task):
        return True
    if state_transition_fields:
        return True
    return False


def _outline_task_depends_on(task: dict[str, Any]) -> list[str]:
    return [
        str(item).strip()
        for item in (task.get("depends_on") or [])
        if str(item).strip()
    ]


def _outline_task_requirement_id(
    task: dict[str, Any],
    *,
    parts_by_name: dict[str, dict[str, Any]],
) -> str:
    action_target = _outline_task_action_target(task)
    requirement_id = str(action_target.get("requirement_id") or "").strip()
    if requirement_id:
        return requirement_id
    for part_name in _outline_task_part_references(task):
        requirement_id = str(
            dict(parts_by_name.get(part_name) or {}).get("goal_requirement_id") or ""
        ).strip()
        if requirement_id:
            return requirement_id
    return ""


def _modeled_gap_pending_nominal_tasks(llm_input: dict[str, Any] | None) -> list[dict[str, Any]]:
    modeled_gap = dict((llm_input or {}).get("modeled_continuation_gap") or {})
    return [
        dict(row)
        for row in (modeled_gap.get("pending_nominal_tasks") or [])
        if isinstance(row, dict)
    ]


def _outline_task_target_locations(task: dict[str, Any]) -> list[str]:
    action_target = _outline_task_action_target(task)
    end_state = dict(task.get("expected_end_state") or {})
    tokens = [
        str(action_target.get("target_location") or "").strip(),
        str(end_state.get("location") or "").strip(),
        str(end_state.get("current_location") or "").strip(),
        str(end_state.get("part_location") or "").strip(),
    ]
    deduped: list[str] = []
    for token in tokens:
        if token and token not in deduped:
            deduped.append(token)
    return deduped


def _outline_task_matches_pending_nominal_suffix(
    task: dict[str, Any],
    *,
    parts_by_name: dict[str, dict[str, Any]],
    llm_input: dict[str, Any] | None,
) -> bool:
    resource_jid = str(task.get("resource_jid") or "").strip()
    if not resource_jid:
        return False
    part_binding = _outline_task_part_binding(task, parts_by_name=parts_by_name)
    candidate_parts = list(part_binding.get("candidate_part_names") or [])
    effective_part_name = str(part_binding.get("effective_part_name") or "").strip()
    if effective_part_name and effective_part_name not in candidate_parts:
        candidate_parts.append(effective_part_name)
    if not candidate_parts:
        return False
    requirement_id = _outline_task_requirement_id(task, parts_by_name=parts_by_name)
    target_locations = set(_outline_task_target_locations(task))
    end_state = dict(task.get("expected_end_state") or {})
    end_state_token = str(
        end_state.get("current_state") or end_state.get("state") or ""
    ).strip().lower()
    for pending_task in _modeled_gap_pending_nominal_tasks(llm_input):
        pending_part = str(pending_task.get("part") or "").strip()
        pending_resource = str(pending_task.get("resource") or "").strip()
        if not pending_part:
            continue
        if pending_resource and pending_resource != resource_jid:
            continue
        if pending_part not in candidate_parts:
            continue
        part_row = dict(parts_by_name.get(pending_part) or {})
        goal_requirement_id = str(part_row.get("goal_requirement_id") or "").strip()
        if requirement_id and goal_requirement_id and requirement_id != goal_requirement_id:
            continue
        goal_location = str(part_row.get("goal_location") or "").strip()
        if goal_location and goal_location in target_locations:
            return True
        if goal_location and _outline_state_location_token(end_state) == goal_location:
            return True
        if end_state_token in {"placed", "assembled"}:
            return True
    return False


def _outline_task_has_part_flow(
    task: dict[str, Any],
    *,
    referenced_parts: list[str],
) -> bool:
    if not referenced_parts:
        return False
    part_tokens = {token for token in referenced_parts if token}
    start_state = dict(task.get("expected_start_state") or {})
    end_state = dict(task.get("expected_end_state") or {})
    action_target = _outline_task_action_target(task)
    for state in (start_state, end_state):
        for field_name in ("held_part", "part_name"):
            token = str(state.get(field_name) or "").strip()
            if token and token in part_tokens:
                return True
        current_state = str(state.get("current_state") or state.get("state") or "").strip().lower()
        if current_state in {"misplaced", "picked", "placed", "assembled", "in_gripper"}:
            return True
        if any(
            str(state.get(field_name) or "").strip()
            for field_name in ("location", "current_location")
        ):
            return True
    if any(
        str(action_target.get(field_name) or "").strip()
        for field_name in ("source_location", "target_location")
    ):
        return True
    return False


def _state_delta(before: Any, after: Any) -> dict[str, Any] | None:
    if before == after:
        return None
    return {"before": deepcopy(before), "after": deepcopy(after)}


def _infer_outline_macro_signature(
    task: dict[str, Any],
    *,
    resources_by_jid: dict[str, dict[str, Any]] | None,
    parts_by_name: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    resource_map = resources_by_jid or {}
    resource_jid = str(task.get("resource_jid") or "").strip()
    resource_row = dict(resource_map.get(resource_jid) or {})
    start_state = dict(task.get("expected_start_state") or {})
    end_state = dict(task.get("expected_end_state") or {})
    action_target = _outline_task_action_target(task)
    referenced_parts = _outline_task_part_references(task)
    task_part_name = str(task.get("part_name") or "").strip()
    explicit_start_part_name = str(start_state.get("part_name") or "").strip()
    explicit_end_part_name = str(end_state.get("part_name") or "").strip()

    before_resource_state = (
        _outline_state_resource_state_token(start_state)
        or str(resource_row.get("current_state") or "").strip()
    )
    after_resource_state = (
        _outline_state_resource_state_token(end_state)
        or before_resource_state
    )
    before_gripper_state = str(
        _first_non_empty_state_value(start_state, "gripper_state")
        or resource_row.get("gripper_state")
        or ""
    ).strip()
    after_gripper_state = str(
        _first_non_empty_state_value(end_state, "gripper_state")
        or before_gripper_state
        or ""
    ).strip()
    if "held_part" in start_state:
        before_held_part = str(start_state.get("held_part") or "").strip()
    else:
        before_held_part = str(resource_row.get("held_part") or "").strip()
    if "held_part" in end_state:
        after_held_part = str(end_state.get("held_part") or "").strip()
    else:
        after_held_part = before_held_part
    before_named_pose = _outline_resource_named_pose_token(
        state=start_state,
        action_target={},
    ) or str(resource_row.get("current_location") or "").strip()
    after_named_pose = _outline_resource_named_pose_token(
        state=end_state,
        action_target=action_target,
    ) or before_named_pose
    before_resource_pose = _outline_state_pose_value(start_state) or dict(
        resource_row.get("current_pose") or {}
    ) or None
    after_resource_pose = _outline_state_pose_value(end_state) or before_resource_pose

    resource_delta: dict[str, Any] = {}
    for field_name, before_value, after_value in (
        ("current_state", before_resource_state or None, after_resource_state or None),
        ("gripper_state", before_gripper_state or None, after_gripper_state or None),
        ("held_part", before_held_part or None, after_held_part or None),
        ("current_location", before_named_pose or None, after_named_pose or None),
        ("current_pose", before_resource_pose, after_resource_pose),
    ):
        delta = _state_delta(before_value, after_value)
        if delta is not None:
            resource_delta[field_name] = delta

    candidate_parts: list[str] = []
    for token in referenced_parts + [before_held_part, after_held_part]:
        normalized = str(token or "").strip()
        if normalized and normalized not in candidate_parts:
            candidate_parts.append(normalized)

    inferable_primary_part = task_part_name
    if not inferable_primary_part:
        explicit_contract_candidates = [
            token
            for token in (
                explicit_end_part_name,
                explicit_start_part_name,
                str(end_state.get("held_part") or "").strip()
                if "held_part" in end_state
                else "",
                str(start_state.get("held_part") or "").strip()
                if "held_part" in start_state
                else "",
            )
            if token
        ]
        deduped_contract_candidates: list[str] = []
        for token in explicit_contract_candidates:
            if token not in deduped_contract_candidates:
                deduped_contract_candidates.append(token)
        if (
            "held_part" in end_state
            and after_held_part
            and after_held_part != before_held_part
        ):
            inferable_primary_part = after_held_part
        elif explicit_end_part_name:
            inferable_primary_part = explicit_end_part_name
        elif explicit_start_part_name:
            inferable_primary_part = explicit_start_part_name
        elif len(deduped_contract_candidates) == 1:
            inferable_primary_part = deduped_contract_candidates[0]

    part_deltas: dict[str, dict[str, Any]] = {}
    direct_observed_pickup_parts: list[str] = []
    for part_name in candidate_parts:
        part_row = dict(parts_by_name.get(part_name) or {})
        before_holder = str(part_row.get("current_holder_resource_jid") or "").strip()
        if not before_holder and before_held_part == part_name:
            before_holder = resource_jid
        after_holder = before_holder
        explicit_end_holder = str(end_state.get("current_holder_resource_jid") or "").strip()
        if explicit_end_holder:
            after_holder = explicit_end_holder
        elif after_held_part == part_name:
            after_holder = resource_jid
        elif before_held_part == part_name and after_held_part != part_name:
            after_holder = ""

        before_part_state = str(part_row.get("current_state") or "").strip()
        explicit_start_part_state = _outline_state_part_state_token(
            start_state,
            candidate_part_name=part_name,
        )
        explicit_end_part_state = _outline_state_part_state_token(
            end_state,
            candidate_part_name=part_name,
        )
        if explicit_start_part_state:
            before_part_state = explicit_start_part_state
        after_part_state = explicit_end_part_state or before_part_state
        if after_holder == resource_jid and not after_part_state:
            after_part_state = "picked"

        before_part_location = str(
            part_row.get("current_location") or part_row.get("location") or ""
        ).strip()
        if not before_part_location and before_holder == resource_jid:
            before_part_location = f"{resource_jid}_gripper"
        after_part_location = before_part_location
        explicit_end_location = _outline_state_location_token(end_state)
        if after_holder == resource_jid:
            after_part_location = f"{resource_jid}_gripper"
        elif explicit_end_location and (
            task_part_name == part_name
            or inferable_primary_part == part_name
            or str(end_state.get("part_name") or "").strip() == part_name
        ):
            after_part_location = explicit_end_location
        elif (
            str(action_target.get("target_location") or "").strip()
            and (
                task_part_name == part_name
                or inferable_primary_part == part_name
            )
            and after_holder != resource_jid
        ):
            after_part_location = str(action_target.get("target_location") or "").strip()
        elif before_holder == resource_jid and after_holder != resource_jid:
            after_part_location = explicit_end_location

        before_part_pose = dict(part_row.get("observed_pose") or {}) or None
        after_part_pose = _outline_state_pose_value(end_state) or before_part_pose

        delta_row: dict[str, Any] = {}
        for field_name, before_value, after_value in (
            ("state", before_part_state or None, after_part_state or None),
            ("holder", before_holder or None, after_holder or None),
            ("location", before_part_location or None, after_part_location or None),
            ("pose", before_part_pose, after_part_pose),
        ):
            delta = _state_delta(before_value, after_value)
            if delta is not None:
                delta_row[field_name] = delta
        if delta_row:
            part_deltas[part_name] = delta_row
            if (
                isinstance(part_row.get("observed_pose"), dict)
                and "x" in dict(part_row.get("observed_pose") or {})
                and before_holder != resource_jid
            ):
                direct_observed_pickup_parts.append(part_name)

    changes_part_world = bool(part_deltas)
    part_intent_without_delta = False
    primary_part_for_intent = task_part_name or inferable_primary_part
    if not changes_part_world and primary_part_for_intent:
        state_scoped_part_anchor = False
        for state in (start_state, end_state):
            if str(state.get("part_name") or "").strip() != primary_part_for_intent:
                continue
            if (
                _outline_state_pose_value(state)
                or _outline_state_location_token(state)
                or str(state.get("current_holder_resource_jid") or "").strip()
                or str(state.get("held_part") or "").strip() == primary_part_for_intent
            ):
                state_scoped_part_anchor = True
                break
        part_intent_without_delta = bool(
            _outline_state_part_state_token(
                start_state,
                candidate_part_name=primary_part_for_intent,
            )
            or _outline_state_part_state_token(
                end_state,
                candidate_part_name=primary_part_for_intent,
            )
            or state_scoped_part_anchor
            or str(action_target.get("source_location") or "").strip()
            or str(action_target.get("target_location") or "").strip()
        )
        if part_intent_without_delta:
            primary_part_row = dict(parts_by_name.get(primary_part_for_intent) or {})
            if (
                isinstance(primary_part_row.get("observed_pose"), dict)
                and "x" in dict(primary_part_row.get("observed_pose") or {})
                and str(primary_part_row.get("current_holder_resource_jid") or "").strip() != resource_jid
                and primary_part_for_intent not in direct_observed_pickup_parts
            ):
                direct_observed_pickup_parts.append(primary_part_for_intent)
    illegal_holder_swap = bool(
        before_held_part and after_held_part and before_held_part != after_held_part
    )
    if changes_part_world or part_intent_without_delta:
        task_kind = "part_handling"
    else:
        task_kind = "resource_only"

    return {
        "resource_before": {
            "current_state": before_resource_state or None,
            "gripper_state": before_gripper_state or None,
            "held_part": before_held_part or None,
            "current_location": before_named_pose or None,
            "current_pose": deepcopy(before_resource_pose),
        },
        "resource_after": {
            "current_state": after_resource_state or None,
            "gripper_state": after_gripper_state or None,
            "held_part": after_held_part or None,
            "current_location": after_named_pose or None,
            "current_pose": deepcopy(after_resource_pose),
        },
        "resource_delta": resource_delta,
        "part_deltas": deepcopy(part_deltas),
        "changes_part_world": changes_part_world,
        "changes_part_holder": any("holder" in row for row in part_deltas.values()),
        "changes_part_location": any("location" in row for row in part_deltas.values()),
        "changes_part_pose": any("pose" in row for row in part_deltas.values()),
        "changes_held_part": "held_part" in resource_delta,
        "illegal_holder_swap": illegal_holder_swap,
        "direct_observed_pickup_parts": direct_observed_pickup_parts,
        "inferable_primary_part": inferable_primary_part,
        "task_kind": task_kind,
    }
def _classify_outline_task(
    task: dict[str, Any],
    *,
    resources_by_jid: dict[str, dict[str, Any]] | None = None,
    parts_by_name: dict[str, dict[str, Any]],
    llm_input: dict[str, Any] | None = None,
) -> str:
    signature = _infer_outline_macro_signature(
        task,
        resources_by_jid=resources_by_jid,
        parts_by_name=parts_by_name,
    )
    task_kind = str(signature.get("task_kind") or "resource_only")
    if task_kind == "part_handling" and _outline_task_matches_pending_nominal_suffix(
        task,
        parts_by_name=parts_by_name,
        llm_input=llm_input,
    ):
        return "continuation_resume"
    return task_kind


def _build_outline_task_type_lookup(
    outline_tasks: list[dict[str, Any]],
    *,
    resources_by_jid: dict[str, dict[str, Any]] | None = None,
    parts_by_name: dict[str, dict[str, Any]],
    llm_input: dict[str, Any] | None = None,
) -> dict[str, str]:
    task_types: dict[str, str] = {}
    for index, raw_task in enumerate(outline_tasks):
        if not isinstance(raw_task, dict):
            continue
        outline_id = str(raw_task.get("outline_id") or "").strip() or f"task_{index}"
        task_types[outline_id] = _classify_outline_task(
            raw_task,
            resources_by_jid=resources_by_jid,
            parts_by_name=parts_by_name,
            llm_input=llm_input,
        )
    return task_types


def _outline_dependency_map(outline_tasks: list[dict[str, Any]]) -> dict[str, list[str]]:
    dependency_map: dict[str, list[str]] = {}
    for index, raw_task in enumerate(outline_tasks):
        if not isinstance(raw_task, dict):
            continue
        outline_id = str(raw_task.get("outline_id") or "").strip() or f"task_{index}"
        dependency_map[outline_id] = _outline_task_depends_on(raw_task)
    return dependency_map


def _outline_dependency_reaches(
    *,
    task_id: str,
    target_id: str,
    dependency_map: dict[str, list[str]],
) -> bool:
    frontier = list(dependency_map.get(task_id) or [])
    visited: set[str] = set()
    while frontier:
        current = frontier.pop()
        if current in visited:
            continue
        if current == target_id:
            return True
        visited.add(current)
        frontier.extend(dependency_map.get(current) or [])
    return False


def _outline_rollout_tasks(outline_tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    indexed_tasks: list[tuple[int, str, dict[str, Any]]] = []
    task_index_by_id: dict[str, int] = {}
    tasks_by_id: dict[str, dict[str, Any]] = {}
    for index, raw_task in enumerate(outline_tasks):
        if not isinstance(raw_task, dict):
            continue
        outline_id = str(raw_task.get("outline_id") or "").strip() or f"task_{index}"
        indexed_tasks.append((index, outline_id, raw_task))
        task_index_by_id[outline_id] = index
        tasks_by_id[outline_id] = raw_task

    indegree = {outline_id: 0 for _, outline_id, _ in indexed_tasks}
    outgoing: dict[str, list[str]] = {outline_id: [] for _, outline_id, _ in indexed_tasks}
    dependency_map = _outline_dependency_map(outline_tasks)
    for _, outline_id, _ in indexed_tasks:
        for dependency_id in dependency_map.get(outline_id) or []:
            if dependency_id not in indegree:
                continue
            indegree[outline_id] += 1
            outgoing.setdefault(dependency_id, []).append(outline_id)

    ready = sorted(
        [outline_id for outline_id, degree in indegree.items() if degree == 0],
        key=lambda item: task_index_by_id[item],
    )
    ordered_ids: list[str] = []
    while ready:
        outline_id = ready.pop(0)
        ordered_ids.append(outline_id)
        for dependent_id in sorted(
            outgoing.get(outline_id) or [],
            key=lambda item: task_index_by_id[item],
        ):
            indegree[dependent_id] -= 1
            if indegree[dependent_id] == 0:
                ready.append(dependent_id)
                ready.sort(key=lambda item: task_index_by_id[item])

    for _, outline_id, _ in indexed_tasks:
        if outline_id not in ordered_ids:
            ordered_ids.append(outline_id)
    return [deepcopy(tasks_by_id[outline_id]) for outline_id in ordered_ids if outline_id in tasks_by_id]


def _is_part_lifecycle_state(token: str) -> bool:
    return str(token or "").strip().lower() in {
        "misplaced",
        "picked",
        "placed",
        "assembled",
        "in_gripper",
    }


def _is_resource_state_token(token: str) -> bool:
    return str(token or "").strip().lower() in {
        "failed",
        "idle",
        "busy",
        "picked",
        "ready",
        "engaged",
    }


def _outline_task_has_post_task_anchor(
    task: dict[str, Any],
    *,
    task_part_name: str,
) -> bool:
    if not task_part_name:
        return True
    end_state = dict(task.get("expected_end_state") or {})
    action_target = _outline_task_action_target(task)
    end_pose = (
        end_state.get("position")
        or end_state.get("pose")
        or end_state.get("current_pose")
    )
    if isinstance(end_pose, dict) and "x" in end_pose:
        return True
    if any(
        str(end_state.get(field_name) or "").strip()
        for field_name in ("part_location", "location", "current_location")
    ):
        return True
    if any(
        str(action_target.get(field_name) or "").strip()
        for field_name in ("target_location", "named_pose")
    ):
        return True
    if str(end_state.get("held_part") or "").strip() == task_part_name:
        return True
    if str(end_state.get("current_holder_resource_jid") or "").strip():
        return True
    return False


def _apply_outline_task_effects(
    task: dict[str, Any],
    *,
    resources_by_jid: dict[str, dict[str, Any]],
    parts_by_name: dict[str, dict[str, Any]],
    task_type: str,
    grounded_action: dict[str, Any] | None = None,
) -> None:
    grounded_action = dict(grounded_action or {})
    resource_jid = str(task.get("resource_jid") or "").strip()
    if not resource_jid or resource_jid not in resources_by_jid:
        return

    resource_row = dict(resources_by_jid.get(resource_jid) or {})
    end_state = dict(task.get("expected_end_state") or {})
    action_target = _outline_task_action_target(task)
    task_part_name = str(
        grounded_action.get("part_name")
        or _outline_task_part_binding(task, parts_by_name=parts_by_name).get("effective_part_name")
        or task.get("part_name")
        or ""
    ).strip()
    expected_effect = dict(grounded_action.get("expected_effect") or {})
    resource_effect = dict(expected_effect.get("resource") or {})
    part_effect = dict(expected_effect.get("part") or {})
    effect_scope = str(grounded_action.get("effect_scope") or "").strip().lower()

    if "current_state" in resource_effect:
        candidate_state = str(resource_effect.get("current_state") or "").strip()
        if candidate_state:
            resource_row["current_state"] = candidate_state
    elif "current_state" in end_state or "state" in end_state or "resource_state" in end_state:
        candidate_state = str(
            end_state.get("current_state")
            or end_state.get("state")
            or end_state.get("resource_state")
            or ""
        ).strip()
        if task_type == "resource_only" or _is_resource_state_token(candidate_state):
            resource_row["current_state"] = candidate_state or resource_row.get("current_state")
    if "gripper_state" in resource_effect:
        resource_row["gripper_state"] = deepcopy(resource_effect.get("gripper_state"))
    elif "gripper_state" in end_state:
        resource_row["gripper_state"] = deepcopy(end_state.get("gripper_state"))
    if "held_part" in resource_effect:
        held_part = str(resource_effect.get("held_part") or "").strip()
        resource_row["held_part"] = held_part or None
    elif "held_part" in end_state:
        held_part = str(end_state.get("held_part") or "").strip()
        resource_row["held_part"] = held_part or None
    if "location" in resource_effect:
        resource_row["current_location"] = str(
            resource_effect.get("location") or ""
        ).strip() or None
    elif "current_location" in end_state or "location" in end_state or "named_pose" in end_state:
        resource_row["current_location"] = str(
            end_state.get("current_location")
            or end_state.get("location")
            or end_state.get("named_pose")
            or ""
        ).strip() or None
    end_pose = resource_effect.get("pose")
    if not isinstance(end_pose, dict):
        end_pose = (
            end_state.get("position")
            or end_state.get("pose")
            or end_state.get("current_pose")
        )
    if isinstance(end_pose, dict):
        if "x" in end_pose:
            resource_row["current_pose"] = deepcopy(end_pose)
        elif str(end_pose.get("named_pose") or "").strip():
            resource_row["current_location"] = str(end_pose.get("named_pose") or "").strip()
    resources_by_jid[resource_jid] = deepcopy(resource_row)

    if task_type != "part_handling" or not task_part_name:
        return

    part_row = dict(parts_by_name.get(task_part_name) or {})
    if not part_row:
        return
    if effect_scope in {"part_only", "resource_and_part"} or part_effect:
        if "state" in part_effect:
            candidate_state = str(part_effect.get("state") or "").strip()
            if candidate_state:
                part_row["current_state"] = candidate_state
        if "location" in part_effect:
            part_row["current_location"] = str(
                part_effect.get("location") or ""
            ).strip() or None
        if "holder" in part_effect:
            holder = str(part_effect.get("holder") or "").strip()
            part_row["current_holder_resource_jid"] = holder or None
            if holder and "location" not in part_effect:
                part_row["current_location"] = f"{holder}_gripper"
        if "pose" in part_effect and isinstance(part_effect.get("pose"), dict):
            part_row["observed_pose"] = deepcopy(part_effect.get("pose"))
        elif any(key in part_effect for key in ("holder", "location")):
            if str(part_row.get("current_holder_resource_jid") or "").strip() or str(
                part_row.get("current_location") or ""
            ).strip():
                part_row["observed_pose"] = None
    else:
        if "current_state" in end_state or "state" in end_state:
            candidate_state = str(
                end_state.get("current_state") or end_state.get("state") or ""
            ).strip()
            if _is_part_lifecycle_state(candidate_state):
                part_row["current_state"] = candidate_state
        if "part_location" in end_state or "location" in end_state or "current_location" in end_state:
            part_row["current_location"] = str(
                end_state.get("part_location")
                or end_state.get("location")
                or end_state.get("current_location")
                or ""
            ).strip() or None
        elif str(action_target.get("target_location") or "").strip():
            part_row["current_location"] = str(action_target.get("target_location") or "").strip()
        if "held_part" in end_state:
            held_part = str(end_state.get("held_part") or "").strip()
            if held_part == task_part_name:
                part_row["current_holder_resource_jid"] = resource_jid
                part_row["current_location"] = f"{resource_jid}_gripper"
            elif not held_part:
                part_row["current_holder_resource_jid"] = None
        elif str(end_state.get("current_holder_resource_jid") or "").strip():
            part_row["current_holder_resource_jid"] = str(
                end_state.get("current_holder_resource_jid") or ""
            ).strip()
        part_end_pose = (
            end_state.get("position")
            or end_state.get("pose")
            or end_state.get("current_pose")
        )
        if isinstance(part_end_pose, dict) and "x" in part_end_pose:
            part_row["observed_pose"] = deepcopy(part_end_pose)
        elif str(part_row.get("current_holder_resource_jid") or "").strip() or str(
            part_row.get("current_location") or ""
        ).strip():
            part_row["observed_pose"] = None
    parts_by_name[task_part_name] = deepcopy(part_row)


def _extract_blocker_part_names(
    *,
    blocking_reason: str,
    parts_by_name: dict[str, dict[str, Any]],
    fallback_parts: list[str],
) -> list[str]:
    preferred_fallback = [
        str(part_name).strip()
        for part_name in fallback_parts
        if str(part_name).strip()
    ]
    if preferred_fallback:
        return preferred_fallback
    blocker_text = str(blocking_reason or "").strip().lower()
    blocker_parts = [
        part_name
        for part_name in parts_by_name
        if part_name and part_name.lower() in blocker_text
    ]
    if blocker_parts:
        return blocker_parts
    return preferred_fallback


def _continuation_prerequisite_task_ids(
    task: dict[str, Any],
    *,
    outline_tasks: list[dict[str, Any]],
    task_types_by_id: dict[str, str],
    parts_by_name: dict[str, dict[str, Any]],
    llm_input: dict[str, Any],
) -> list[str]:
    if _classify_outline_task(
        task,
        parts_by_name=parts_by_name,
        llm_input=llm_input,
    ) != "continuation_resume":
        return []
    requirement_id = _outline_task_requirement_id(task, parts_by_name=parts_by_name)
    referenced_parts = _outline_task_part_references(task)
    pending_nominal_task_ids: list[str] = []
    for part_name in referenced_parts:
        pending_nominal_task_ids.extend(
            str(item).strip()
            for item in (dict(parts_by_name.get(part_name) or {}).get("pending_nominal_task_ids") or [])
            if str(item).strip()
        )
    modeled_gap = dict(llm_input.get("modeled_continuation_gap") or {})
    unmet_conditions = [
        dict(row)
        for row in (modeled_gap.get("unmet_continuation_conditions") or [])
        if isinstance(row, dict)
    ]
    fault_event = dict(llm_input.get("fault_event") or {})
    fallback_parts = [
        str(item).strip()
        for item in (fault_event.get("affected_part_names") or [])
        if str(item).strip()
    ]
    prerequisite_ids: list[str] = []
    for condition in unmet_conditions:
        kind = str(condition.get("kind") or "").strip()
        if kind == "focused_resource_terminal_state":
            resource_jid = str(condition.get("entity") or "").strip()
            expected_state = str(condition.get("expected") or "").strip()
            for index, raw_task in enumerate(outline_tasks):
                if not isinstance(raw_task, dict):
                    continue
                outline_id = str(raw_task.get("outline_id") or "").strip() or f"task_{index}"
                if task_types_by_id.get(outline_id) != "resource_only":
                    continue
                if str(raw_task.get("resource_jid") or "").strip() != resource_jid:
                    continue
                end_state = dict(raw_task.get("expected_end_state") or {})
                if str(end_state.get("current_state") or end_state.get("state") or "").strip() == expected_state:
                    prerequisite_ids.append(outline_id)
        elif kind == "safety_blocked_suffix_task":
            source_task_id = str(condition.get("source_task_id") or "").strip()
            if pending_nominal_task_ids and source_task_id and source_task_id not in pending_nominal_task_ids:
                continue
            blocker_parts = _extract_blocker_part_names(
                blocking_reason=str(condition.get("blocking_reason") or "").strip(),
                parts_by_name=parts_by_name,
                fallback_parts=fallback_parts,
            )
            for blocker_part in blocker_parts:
                goal_location = str(
                    dict(parts_by_name.get(blocker_part) or {}).get("goal_location") or ""
                ).strip()
                for index, raw_task in enumerate(outline_tasks):
                    if not isinstance(raw_task, dict):
                        continue
                    outline_id = str(raw_task.get("outline_id") or "").strip() or f"task_{index}"
                    if task_types_by_id.get(outline_id) == "continuation_resume":
                        continue
                    if str(raw_task.get("part_name") or "").strip() != blocker_part:
                        continue
                    end_state = dict(raw_task.get("expected_end_state") or {})
                    end_current_state = str(
                        end_state.get("current_state") or end_state.get("state") or ""
                    ).strip().lower()
                    end_location = str(
                        end_state.get("location") or end_state.get("current_location") or ""
                    ).strip()
                    if end_current_state in {"placed", "assembled"} or (
                        goal_location and end_location == goal_location
                    ):
                        prerequisite_ids.append(outline_id)
    if requirement_id:
        prerequisite_ids = [
            outline_id
            for outline_id in prerequisite_ids
            if outline_id
        ]
    deduped_ids: list[str] = []
    for outline_id in prerequisite_ids:
        if outline_id and outline_id not in deduped_ids:
            deduped_ids.append(outline_id)
    return deduped_ids


def _relevant_continuation_conditions(
    task: dict[str, Any],
    *,
    parts_by_name: dict[str, dict[str, Any]],
    llm_input: dict[str, Any],
) -> list[dict[str, Any]]:
    if not _outline_task_matches_pending_nominal_suffix(
        task,
        parts_by_name=parts_by_name,
        llm_input=llm_input,
    ):
        return []
    referenced_parts = _outline_task_part_references(task)
    pending_nominal_task_ids: list[str] = []
    for part_name in referenced_parts:
        pending_nominal_task_ids.extend(
            str(item).strip()
            for item in (dict(parts_by_name.get(part_name) or {}).get("pending_nominal_task_ids") or [])
            if str(item).strip()
        )
    modeled_gap = dict(llm_input.get("modeled_continuation_gap") or {})
    relevant_conditions: list[dict[str, Any]] = []
    for raw_condition in (modeled_gap.get("unmet_continuation_conditions") or []):
        if not isinstance(raw_condition, dict):
            continue
        condition = dict(raw_condition)
        kind = str(condition.get("kind") or "").strip()
        if kind == "focused_resource_terminal_state":
            relevant_conditions.append(condition)
            continue
        if kind != "safety_blocked_suffix_task":
            continue
        source_task_id = str(condition.get("source_task_id") or "").strip()
        if pending_nominal_task_ids and source_task_id and source_task_id not in pending_nominal_task_ids:
            continue
        relevant_conditions.append(condition)
    return relevant_conditions


def _continuation_condition_cleared(
    condition: dict[str, Any],
    *,
    resources_by_jid: dict[str, dict[str, Any]],
    parts_by_name: dict[str, dict[str, Any]],
    fallback_parts: list[str],
) -> bool:
    kind = str(condition.get("kind") or "").strip()
    if kind == "focused_resource_terminal_state":
        resource_jid = str(condition.get("entity") or "").strip()
        expected_state = str(condition.get("expected") or "").strip()
        current_state = str(
            dict(resources_by_jid.get(resource_jid) or {}).get("current_state") or ""
        ).strip()
        return bool(expected_state and current_state == expected_state)
    if kind != "safety_blocked_suffix_task":
        return False
    blocker_parts = _extract_blocker_part_names(
        blocking_reason=str(condition.get("blocking_reason") or "").strip(),
        parts_by_name=parts_by_name,
        fallback_parts=fallback_parts,
    )
    if not blocker_parts:
        return False
    for blocker_part in blocker_parts:
        part_row = dict(parts_by_name.get(blocker_part) or {})
        goal_location = str(part_row.get("goal_location") or "").strip()
        current_state = str(part_row.get("current_state") or "").strip().lower()
        current_location = str(
            part_row.get("current_location") or part_row.get("location") or ""
        ).strip()
        if current_state in {"placed", "assembled"}:
            continue
        if goal_location and current_location == goal_location:
            continue
        return False
    return True


def _outline_validation_ref(finding: dict[str, Any]) -> dict[str, Any]:
    ref = {
        "task_id": str(finding.get("task_id") or "").strip(),
        "pose_source": str(finding.get("pose_source") or "").strip(),
        "failed_axes": [
            str(item).strip()
            for item in (finding.get("failed_axes") or [])
            if str(item).strip()
        ],
    }
    resource_jid = str(finding.get("resource_jid") or "").strip()
    if resource_jid:
        ref["resource_jid"] = resource_jid
    failed_reason = str(finding.get("failed_reason") or "").strip()
    if failed_reason:
        ref["failed_reason"] = failed_reason
    return ref


def _outline_validation_ref_key(finding: dict[str, Any]) -> tuple[str, str, tuple[str, ...]]:
    ref = _outline_validation_ref(finding)
    return (
        str(ref.get("task_id") or "").strip(),
        str(ref.get("pose_source") or "").strip(),
        tuple(
            str(item).strip()
            for item in (ref.get("failed_axes") or [])
            if str(item).strip()
        ),
    )


def _format_outline_validation_ref(ref: dict[str, Any]) -> str:
    task_id = str(ref.get("task_id") or "").strip()
    resource_jid = str(ref.get("resource_jid") or "").strip()
    pose_source = str(ref.get("pose_source") or "").strip()
    failed_axes = [
        str(item).strip()
        for item in (ref.get("failed_axes") or [])
        if str(item).strip()
    ]
    axis_summary = ", ".join(failed_axes) or "no failed_axes"
    if resource_jid:
        return f"task '{task_id}' on {resource_jid} ({pose_source}: {axis_summary})"
    return f"task '{task_id}' ({pose_source}: {axis_summary})"


def _extract_addressed_validation_findings(
    parsed_response: dict[str, Any],
) -> list[dict[str, Any]]:
    addressed_refs: list[dict[str, Any]] = []
    for row in (parsed_response.get("addressed_validation_findings") or []):
        if not isinstance(row, dict):
            continue
        ref = {
            "task_id": str(row.get("task_id") or "").strip(),
            "pose_source": str(row.get("pose_source") or "").strip(),
            "failed_axes": [
                str(item).strip()
                for item in (row.get("failed_axes") or [])
                if str(item).strip()
            ],
        }
        resource_jid = str(row.get("resource_jid") or "").strip()
        if resource_jid:
            ref["resource_jid"] = resource_jid
        addressed_refs.append(ref)
    return addressed_refs


def _validate_addressed_validation_findings(
    *,
    addressed_refs: list[dict[str, Any]],
    outstanding_findings: list[dict[str, Any]],
) -> tuple[list[str], list[dict[str, Any]], dict[str, Any] | None]:
    required_refs = [
        _outline_validation_ref(row)
        for row in outstanding_findings
        if isinstance(row, dict)
    ]
    if not required_refs and not addressed_refs:
        return [], [], None

    required_keys = {_outline_validation_ref_key(row) for row in required_refs}
    provided_keys = {_outline_validation_ref_key(row) for row in addressed_refs}

    missing_refs = [
        deepcopy(row)
        for row in required_refs
        if _outline_validation_ref_key(row) not in provided_keys
    ]
    unexpected_refs = [
        deepcopy(row)
        for row in addressed_refs
        if _outline_validation_ref_key(row) not in required_keys
    ]
    violations: list[str] = []
    if missing_refs:
        violations.append(
            "addressed_validation_findings is missing required refs: "
            + "; ".join(_format_outline_validation_ref(row) for row in missing_refs)
        )
    if unexpected_refs:
        violations.append(
            "addressed_validation_findings includes unexpected refs: "
            + "; ".join(_format_outline_validation_ref(row) for row in unexpected_refs)
        )

    carried_findings = [
        deepcopy(row)
        for row in outstanding_findings
        if _outline_validation_ref_key(row)
        in {_outline_validation_ref_key(item) for item in missing_refs}
    ]
    coverage = {
        "status": "failed" if violations else "passed",
        "required_addressed_validation_findings": deepcopy(required_refs),
        "provided_addressed_validation_findings": deepcopy(addressed_refs),
    }
    if violations:
        coverage["violations"] = deepcopy(violations)
    if missing_refs:
        coverage["missing_required_refs"] = deepcopy(missing_refs)
    if unexpected_refs:
        coverage["unexpected_refs"] = deepcopy(unexpected_refs)
    return violations, carried_findings, coverage


def _merge_outline_validation_findings(
    *finding_groups: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str, tuple[str, ...]]] = set()
    for group in finding_groups:
        for row in group or []:
            if not isinstance(row, dict):
                continue
            key = _outline_validation_ref_key(row)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            merged.append(deepcopy(row))
    return merged


def _outline_pruned_action_descriptor(
    task: dict[str, Any],
    *,
    resources_by_jid: dict[str, dict[str, Any]],
    parts_by_name: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    signature = _infer_outline_macro_signature(
        task,
        resources_by_jid=resources_by_jid,
        parts_by_name=parts_by_name,
    )
    action_target = _outline_task_action_target(task)
    end_state = dict(task.get("expected_end_state") or {})
    descriptor: dict[str, Any] = {
        "resource_jid": str(task.get("resource_jid") or "").strip() or None,
        "task_kind": str(signature.get("task_kind") or "").strip() or None,
        "part_name": (
            str(task.get("part_name") or "").strip()
            or str(signature.get("inferable_primary_part") or "").strip()
            or None
        ),
        "named_pose": str(action_target.get("named_pose") or "").strip() or None,
        "source_location": str(action_target.get("source_location") or "").strip() or None,
        "target_location": (
            str(action_target.get("target_location") or "").strip()
            or _outline_state_location_token(end_state)
            or None
        ),
        "end_resource_state": _outline_state_resource_state_token(end_state) or None,
        "end_gripper_state": str(end_state.get("gripper_state") or "").strip() or None,
        "end_held_part": str(end_state.get("held_part") or "").strip() or None,
    }
    return {
        key: deepcopy(value)
        for key, value in descriptor.items()
        if value not in (None, "", [], {})
    }


def _outline_pruned_action_key(action: dict[str, Any]) -> str:
    return json.dumps(
        deepcopy(action or {}),
        sort_keys=True,
        default=str,
        ensure_ascii=True,
    )


def _outline_validation_finding_status(finding: dict[str, Any]) -> str:
    constraint_owner = str(finding.get("constraint_owner") or "").strip().lower()
    if constraint_owner == "binding":
        return "binding_invalid"
    failed_axes = [
        str(item).strip()
        for item in (finding.get("failed_axes") or [])
        if str(item).strip()
    ]
    return "state_infeasible" if failed_axes or str(finding.get("constraint_code") or "").strip() else "feasible"


def _outline_validation_guard(finding: dict[str, Any]) -> dict[str, Any] | None:
    status = _outline_validation_finding_status(finding)
    if status != "state_infeasible":
        return None
    explicit_guard = dict(finding.get("guard") or {})
    if explicit_guard:
        return explicit_guard
    failed_axes = [
        str(item).strip()
        for item in (finding.get("failed_axes") or [])
        if str(item).strip()
    ]
    guard: dict[str, Any] = {"failed_axes": failed_axes}
    constraint_code = str(finding.get("constraint_code") or "").strip() or (
        failed_axes[0] if len(failed_axes) == 1 else ""
    )
    claimed_condition_ids = _finding_claimed_condition_ids(finding)
    if constraint_code in {"holder_conflict", "resource_holds_other_part"}:
        conflicting_part = str(finding.get("conflicting_part") or "").strip()
        current_holder = str(finding.get("current_holder_resource_jid") or "").strip()
        if conflicting_part:
            guard["kind"] = "resource_holds_part"
            guard["resource_jid"] = str(finding.get("resource_jid") or "").strip()
            guard["held_part"] = conflicting_part
            return guard
        if current_holder:
            guard["kind"] = "part_held_by_other"
            guard["part_name"] = str(finding.get("part_name") or "").strip()
            guard["current_holder_resource_jid"] = current_holder
            return guard
    if constraint_code == "required_part_not_held":
        guard["kind"] = "required_part_not_held"
        guard["resource_jid"] = str(finding.get("resource_jid") or "").strip()
        guard["part_name"] = str(finding.get("part_name") or "").strip()
        return guard
    if constraint_code == "source_reference_unavailable":
        guard["kind"] = "source_reference_unavailable"
        guard["resource_jid"] = str(finding.get("resource_jid") or "").strip()
        guard["part_name"] = str(finding.get("part_name") or "").strip()
        return guard
    if constraint_code == "unsupported_resource_target":
        guard["kind"] = "unsupported_resource_target"
        guard["resource_jid"] = str(finding.get("resource_jid") or "").strip()
        resource_state = str(dict(finding.get("evidence") or {}).get("grounded_action", {}).get("expected_effect", {}).get("resource", {}).get("current_state") or "").strip()
        if resource_state:
            guard["resource_state"] = resource_state
        return guard
    if constraint_code == "illegal_holder_swap":
        conflicting_part = str(finding.get("conflicting_part") or "").strip()
        if conflicting_part:
            guard["kind"] = "resource_holds_part"
            guard["resource_jid"] = str(finding.get("resource_jid") or "").strip()
            guard["held_part"] = conflicting_part
            return guard
    if constraint_code == "gripper_occupancy_conflict":
        guard["kind"] = "gripper_closed_without_target_part"
        guard["resource_jid"] = str(finding.get("resource_jid") or "").strip()
        return guard
    if constraint_code == "blocker_open":
        guard["kind"] = "condition_unmet"
        guard["condition_ids"] = [
            str(item).strip()
            for item in (
                finding.get("blocking_condition_ids")
                or dict(finding.get("evidence") or {}).get("condition_ids")
                or []
            )
            if str(item).strip()
        ]
        return guard
    if constraint_code == "condition_reopened":
        guard["kind"] = "condition_unmet"
        guard["condition_ids"] = (
            [
                str(item).strip()
                for item in (
                    finding.get("reopened_condition_ids")
                    or dict(finding.get("evidence") or {}).get("condition_ids")
                    or []
                )
                if str(item).strip()
            ]
        )
        return guard
    if constraint_code == "safety_rule_violation" and claimed_condition_ids:
        guard["kind"] = "condition_unmet"
        guard["condition_ids"] = claimed_condition_ids
        guard["rule_id"] = str(finding.get("rule_id") or "").strip()
        return guard
    if constraint_code == "named_pose_unavailable":
        guard["kind"] = "named_pose_unavailable"
        guard["resource_jid"] = str(finding.get("resource_jid") or "").strip()
        guard["named_pose"] = str(finding.get("named_pose") or "").strip()
        return guard
    failed_reason = str(finding.get("failed_reason") or "").strip()
    if constraint_code == "workspace_unreachable" or failed_reason == "physically_unreachable":
        pose = deepcopy(finding.get("pose"))
        if isinstance(pose, dict) and "x" in pose:
            guard["kind"] = "observed_pose_unreachable"
            guard["resource_jid"] = str(finding.get("resource_jid") or "").strip()
            guard["part_name"] = str(finding.get("part_name") or "").strip()
            guard["pose"] = pose
            return guard
    if failed_reason == "predicted_pose_out_of_bounds":
        guard["kind"] = "explicit_pose_out_of_bounds"
        guard["pose_source"] = str(finding.get("pose_source") or "").strip()
        guard["failed_reason"] = failed_reason
        guard["failed_axes"] = failed_axes
        guard["pose"] = deepcopy(finding.get("pose"))
        return guard
    if constraint_code == "part_not_in_current_facts":
        guard["kind"] = "part_not_in_current_facts"
        guard["part_name"] = str(finding.get("part_name") or "").strip()
        return guard
    if constraint_code == "resource_unavailable":
        guard["kind"] = "resource_not_available"
        guard["resource_jid"] = str(finding.get("resource_jid") or "").strip()
        return guard
    if constraint_code == "resource_missing_workspace_bounds":
        guard["kind"] = "resource_missing_workspace_bounds"
        guard["resource_jid"] = str(finding.get("resource_jid") or "").strip()
        return guard
    for field_name in (
        "pose_source",
        "failed_reason",
        "resource_jid",
        "part_name",
        "named_pose",
        "conflicting_part",
        "current_holder_resource_jid",
        "rule_id",
    ):
        value = finding.get(field_name)
        if value not in (None, "", [], {}):
            guard[field_name] = deepcopy(value)
    for field_name in (
        "claimed_condition_ids",
        "blocking_condition_ids",
        "reopened_condition_ids",
    ):
        values = [
            str(item).strip()
            for item in (finding.get(field_name) or [])
            if str(item).strip()
        ]
        if values:
            guard[field_name] = values
    guard["kind"] = "validation_replay"
    return guard


def _annotate_outline_validation_finding(
    finding: dict[str, Any],
) -> dict[str, Any]:
    annotated = deepcopy(finding)
    status = _outline_validation_finding_status(annotated)
    annotated["validation_status"] = status
    annotated["guard"] = _outline_validation_guard(annotated)
    annotated["validator_reason"] = _format_outline_validation_finding(annotated)
    return annotated


def _guard_matches_validation_finding(
    guard: dict[str, Any],
    finding: dict[str, Any],
) -> bool:
    guard_axes = [
        str(item).strip()
        for item in (guard.get("failed_axes") or [])
        if str(item).strip()
    ]
    finding_axes = [
        str(item).strip()
        for item in (finding.get("failed_axes") or [])
        if str(item).strip()
    ]
    if guard_axes != finding_axes:
        return False
    for field_name in (
        "pose_source",
        "failed_reason",
        "resource_jid",
        "part_name",
        "named_pose",
        "conflicting_part",
        "current_holder_resource_jid",
        "rule_id",
    ):
        expected = guard.get(field_name)
        if expected in (None, "", [], {}):
            continue
        actual = finding.get(field_name)
        if actual != expected:
            return False
    for field_name in (
        "claimed_condition_ids",
        "blocking_condition_ids",
        "reopened_condition_ids",
    ):
        expected = [
            str(item).strip()
            for item in (guard.get(field_name) or [])
            if str(item).strip()
        ]
        if not expected:
            continue
        actual = [
            str(item).strip()
            for item in (finding.get(field_name) or [])
            if str(item).strip()
        ]
        if actual != expected:
            return False
    return True


def _replay_outline_task_validation(
    task: dict[str, Any],
    *,
    resources_by_jid: dict[str, dict[str, Any]],
    parts_by_name: dict[str, dict[str, Any]],
    llm_input: dict[str, Any],
) -> list[dict[str, Any]]:
    replay_task = deepcopy(task or {})
    task_id = str(replay_task.get("outline_id") or "").strip() or "task_0"
    outline_tasks = [replay_task]
    return _analyze_outline_task_validation(
        replay_task,
        outline_tasks=outline_tasks,
        resources_by_jid=deepcopy(resources_by_jid),
        parts_by_name=deepcopy(parts_by_name),
        llm_input=llm_input,
        task_types_by_id=_build_outline_task_type_lookup(
            outline_tasks,
            resources_by_jid=resources_by_jid,
            parts_by_name=parts_by_name,
            llm_input=llm_input,
        ),
        task_index_by_id={task_id: 0},
        dependency_map={task_id: _outline_task_depends_on(replay_task)},
        active_pruned_actions=None,
    )


def _pruned_action_is_active(
    pruned_action: dict[str, Any],
    *,
    resources_by_jid: dict[str, dict[str, Any]],
    parts_by_name: dict[str, dict[str, Any]],
    llm_input: dict[str, Any],
) -> bool:
    guard = dict(pruned_action.get("guard") or {})
    if not guard:
        return False
    kind = str(guard.get("kind") or "").strip()
    if kind == "resource_holds_part":
        resource = dict(resources_by_jid.get(str(guard.get("resource_jid") or "").strip()) or {})
        return str(resource.get("held_part") or "").strip() == str(guard.get("held_part") or "").strip()
    if kind == "part_held_by_other":
        part = dict(parts_by_name.get(str(guard.get("part_name") or "").strip()) or {})
        return str(part.get("current_holder_resource_jid") or "").strip() == str(
            guard.get("current_holder_resource_jid") or ""
        ).strip()
    if kind == "gripper_closed_without_target_part":
        resource = dict(resources_by_jid.get(str(guard.get("resource_jid") or "").strip()) or {})
        held_part = str(resource.get("held_part") or "").strip()
        gripper_state = str(resource.get("gripper_state") or "").strip().lower()
        return not held_part and gripper_state == "closed"
    if kind == "required_part_not_held":
        resource = dict(resources_by_jid.get(str(guard.get("resource_jid") or "").strip()) or {})
        part_name = str(guard.get("part_name") or "").strip()
        part = dict(parts_by_name.get(part_name) or {})
        return str(resource.get("held_part") or "").strip() != part_name and str(
            part.get("current_holder_resource_jid") or ""
        ).strip() != str(guard.get("resource_jid") or "").strip()
    if kind == "source_reference_unavailable":
        task = dict(pruned_action.get("task") or {})
        if not task:
            return False
        replay_findings = _replay_outline_task_validation(
            task,
            resources_by_jid=resources_by_jid,
            parts_by_name=parts_by_name,
            llm_input=llm_input,
        )
        return any(
            dict(finding).get("constraint_code") == "source_reference_unavailable"
            for finding in replay_findings
            if isinstance(finding, dict)
        )
    if kind == "unsupported_resource_target":
        task = dict(pruned_action.get("task") or {})
        if not task:
            return False
        replay_findings = _replay_outline_task_validation(
            task,
            resources_by_jid=resources_by_jid,
            parts_by_name=parts_by_name,
            llm_input=llm_input,
        )
        return any(
            dict(finding).get("constraint_code") == "unsupported_resource_target"
            for finding in replay_findings
            if isinstance(finding, dict)
        )
    if kind == "condition_unmet":
        _, unmet_condition_ids, _ = _current_recovery_gap_condition_status(
            resources_by_jid=resources_by_jid,
            parts_by_name=parts_by_name,
            llm_input=llm_input,
        )
        tracked_ids = [
            str(item).strip()
            for item in (guard.get("condition_ids") or [])
            if str(item).strip()
        ]
        return any(condition_id in unmet_condition_ids for condition_id in tracked_ids)
    if kind == "condition_not_unmet":
        _, unmet_condition_ids, _ = _current_recovery_gap_condition_status(
            resources_by_jid=resources_by_jid,
            parts_by_name=parts_by_name,
            llm_input=llm_input,
        )
        tracked_ids = [
            str(item).strip()
            for item in (guard.get("condition_ids") or [])
            if str(item).strip()
        ]
        return bool(tracked_ids) and all(
            condition_id not in unmet_condition_ids for condition_id in tracked_ids
        )
    if kind == "named_pose_unavailable":
        resource = dict(resources_by_jid.get(str(guard.get("resource_jid") or "").strip()) or {})
        raw_named_poses = resource.get("named_poses")
        if isinstance(raw_named_poses, dict):
            available_named_poses = {
                str(pose_name).strip()
                for pose_name in raw_named_poses.keys()
                if str(pose_name).strip()
            }
        else:
            available_named_poses = {
                str(pose_name).strip()
                for pose_name in (raw_named_poses or [])
                if str(pose_name).strip()
            }
        named_pose = str(guard.get("named_pose") or "").strip()
        return bool(available_named_poses) and named_pose not in available_named_poses
    if kind == "observed_pose_unreachable":
        resource = dict(resources_by_jid.get(str(guard.get("resource_jid") or "").strip()) or {})
        bounds = dict(resource.get("workspace_bounds") or {})
        part = dict(parts_by_name.get(str(guard.get("part_name") or "").strip()) or {})
        pose = dict(part.get("observed_pose") or {})
        guarded_pose = dict(guard.get("pose") or {})
        reachable, _ = _is_pose_in_workspace(pose, bounds) if pose and bounds else (True, [])
        return bool(pose) and pose == guarded_pose and not reachable
    if kind == "explicit_pose_out_of_bounds":
        return True
    if kind == "part_not_in_current_facts":
        return str(guard.get("part_name") or "").strip() not in parts_by_name
    if kind == "resource_not_available":
        return str(guard.get("resource_jid") or "").strip() not in resources_by_jid
    if kind == "resource_missing_workspace_bounds":
        resource = dict(resources_by_jid.get(str(guard.get("resource_jid") or "").strip()) or {})
        return not isinstance(resource.get("workspace_bounds"), dict)
    task = dict(pruned_action.get("task") or {})
    if not task:
        return False
    replay_findings = _replay_outline_task_validation(
        task,
        resources_by_jid=resources_by_jid,
        parts_by_name=parts_by_name,
        llm_input=llm_input,
    )
    return any(
        _guard_matches_validation_finding(guard, finding)
        for finding in replay_findings
        if isinstance(finding, dict)
    )


def _active_pruned_actions_for_state(
    pruned_actions: list[dict[str, Any]],
    *,
    resources_by_jid: dict[str, dict[str, Any]],
    parts_by_name: dict[str, dict[str, Any]],
    llm_input: dict[str, Any],
) -> list[dict[str, Any]]:
    active_rows: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str]] = set()
    for raw_row in (pruned_actions or []):
        if not isinstance(raw_row, dict):
            continue
        if not _pruned_action_is_active(
            raw_row,
            resources_by_jid=resources_by_jid,
            parts_by_name=parts_by_name,
            llm_input=llm_input,
        ):
            continue
        action_key = _outline_pruned_action_key(dict(raw_row.get("action") or {}))
        guard_key = json.dumps(
            deepcopy(raw_row.get("guard") or {}),
            sort_keys=True,
            default=str,
            ensure_ascii=True,
        )
        dedupe_key = (action_key, guard_key)
        if dedupe_key in seen_keys:
            continue
        seen_keys.add(dedupe_key)
        active_rows.append(deepcopy(raw_row))
    return active_rows


def _build_pruned_actions(
    *,
    existing_pruned_actions: list[dict[str, Any]],
    outline_tasks: list[dict[str, Any]],
    validation_findings: list[dict[str, Any]],
    llm_input: dict[str, Any],
) -> list[dict[str, Any]]:
    resources_by_jid, parts_by_name = _outline_validation_context(llm_input)
    merged_rows = _active_pruned_actions_for_state(
        existing_pruned_actions,
        resources_by_jid=resources_by_jid,
        parts_by_name=parts_by_name,
        llm_input=llm_input,
    )
    tasks_by_id = {
        str(dict(task or {}).get("outline_id") or "").strip(): dict(task or {})
        for task in outline_tasks
        if isinstance(task, dict)
    }
    seen_keys = {
        (
            _outline_pruned_action_key(dict(row.get("action") or {})),
            json.dumps(
                deepcopy(row.get("guard") or {}),
                sort_keys=True,
                default=str,
                ensure_ascii=True,
            ),
        )
        for row in merged_rows
        if isinstance(row, dict)
    }
    for raw_finding in (validation_findings or []):
        if not isinstance(raw_finding, dict):
            continue
        finding = _annotate_outline_validation_finding(raw_finding)
        if str(finding.get("validation_status") or "").strip() != "state_infeasible":
            continue
        task_id = str(finding.get("task_id") or "").strip()
        task = dict(tasks_by_id.get(task_id) or {})
        guard = dict(finding.get("guard") or {})
        if not task or not guard:
            continue
        action = _outline_pruned_action_descriptor(
            task,
            resources_by_jid=resources_by_jid,
            parts_by_name=parts_by_name,
        )
        dedupe_key = (
            _outline_pruned_action_key(action),
            json.dumps(guard, sort_keys=True, default=str, ensure_ascii=True),
        )
        if dedupe_key in seen_keys:
            continue
        seen_keys.add(dedupe_key)
        merged_rows.append(
            {
                "action": deepcopy(action),
                "task": deepcopy(task),
                "guard": deepcopy(guard),
                "reason": str(finding.get("validator_reason") or "").strip(),
            }
        )
    return _active_pruned_actions_for_state(
        merged_rows,
        resources_by_jid=resources_by_jid,
        parts_by_name=parts_by_name,
        llm_input=llm_input,
    )


def _matching_active_pruned_action(
    task: dict[str, Any],
    *,
    pruned_actions: list[dict[str, Any]],
    resources_by_jid: dict[str, dict[str, Any]],
    parts_by_name: dict[str, dict[str, Any]],
    llm_input: dict[str, Any],
) -> dict[str, Any] | None:
    task_action = _outline_pruned_action_descriptor(
        task,
        resources_by_jid=resources_by_jid,
        parts_by_name=parts_by_name,
    )
    task_key = _outline_pruned_action_key(task_action)
    for raw_row in (pruned_actions or []):
        if not isinstance(raw_row, dict):
            continue
        if _outline_pruned_action_key(dict(raw_row.get("action") or {})) != task_key:
            continue
        if _pruned_action_is_active(
            raw_row,
            resources_by_jid=resources_by_jid,
            parts_by_name=parts_by_name,
            llm_input=llm_input,
        ):
            return deepcopy(raw_row)
    return None


# ---------------------------------------------------------------------------
# Outline validation finding formatter — lookup tables & handlers
# ---------------------------------------------------------------------------

# constraint_code → fallback message template (used when reason is empty).
# Templates are interpolated via str.format_map with the common fields dict.
_CONSTRAINT_CODE_FALLBACKS: dict[str, str] = {
    "resource_unavailable": "resource '{resource_jid}' is currently unavailable",
    "named_pose_unavailable": "requested named pose is not available on '{resource_jid}'",
    "holder_conflict": "resource assignment conflicts with current held-part state",
    "resource_holds_other_part": (
        "resource '{resource_jid}' already holds another part "
        "and cannot manipulate '{part_name}'"
    ),
    "required_part_not_held": (
        "task changes part '{part_name}' without proving that "
        "resource '{resource_jid}' currently holds it"
    ),
    "source_reference_unavailable": (
        "task requires a concrete current source reference for "
        "part '{part_name}' before the assigned resource can acquire it"
    ),
    "unsupported_resource_target": (
        "resource-only transition does not target a concrete "
        "supported recovery pose or advertised recovery state"
    ),
    "gripper_occupancy_conflict": (
        "resource '{resource_jid}' gripper state is incompatible "
        "with acquiring '{part_name}'"
    ),
    "workspace_unreachable": (
        "resource '{resource_jid}' cannot reach the grounded "
        "target for part '{part_name_display}'"
    ),
    "dependency_unsatisfied": "continuation task is missing prerequisite dependencies",
    "order_violation": "continuation task appears before prerequisite tasks",
    "blocker_open": "continuation blockers are still uncleared in symbolic state",
    "condition_reopened": (
        "projected state transition reopens previously cleared "
        "continuation conditions"
    ),
}

# failed_axes (single value) → message template needing only common fields.
_SIMPLE_AXIS_MESSAGES: dict[str, str] = {
    "missing_resource_jid": "resource_jid is missing",
    "resource_not_available": (
        "resource '{resource_jid}' is not in the available resource list"
    ),
    "missing_workspace_bounds": (
        "resource '{resource_jid}' has no workspace_bounds in current resource facts"
    ),
    "missing_physical_anchor": (
        "outline task is not a concrete physical recovery step "
        "because it has no grounded part, location, named pose, explicit pose, "
        "or concrete state anchor"
    ),
    "missing_state_transition": (
        "outline task does not describe a meaningful start-to-end state transition"
    ),
    "unbound_part_reference": (
        "part-handling task does not bind any identifiable part in its "
        "structured state"
    ),
    "unexpected_part_reference": (
        "resource-only recovery task must not name part '{part_name}' "
        "unless it directly manipulates that part"
    ),
    "gripper_occupancy_conflict": (
        "resource '{resource_jid}' gripper state is incompatible "
        "with acquiring '{part_name}'"
    ),
    "part_not_in_current_facts": "part '{part_name}' is not in current part facts",
    "missing_validation_anchor": (
        "no explicit task pose or grounded part reference is available "
        "for validation"
    ),
    "missing_post_task_anchor": (
        "part-moving task does not make the post-task holder, "
        "location, or pose of '{part_name}' explicit"
    ),
    "missing_pose_for_referenced_part": (
        "referenced part '{part_name}' has no grounded observed pose "
        "and the task provides no explicit pose for validation"
    ),
}

# failed_axes (single value) → (base_message, finding_key, suffix_format).
# If the list extracted from finding_key is non-empty, suffix_format is
# appended to base_message with the joined list. Otherwise base_message alone.
_ENRICHED_AXIS_SPECS: dict[str, tuple[str, str, str]] = {
    "ambiguous_part_reference": (
        "structured task state refers to multiple possible parts",
        "candidate_part_names",
        " ({ids}); bind the manipulated part more explicitly",
    ),
    "missing_prerequisite_dependency": (
        "continuation task is missing prerequisite dependencies",
        "required_dependency_ids",
        " on {ids}",
    ),
    "continuation_before_prerequisites": (
        "continuation task appears before prerequisite tasks",
        "required_dependency_ids",
        " {ids}",
    ),
    "continuation_blocker_not_cleared": (
        "continuation blockers are still uncleared in symbolic state",
        "blocking_condition_ids",
        " ({ids})",
    ),
    "claimed_condition_not_currently_unmet": (
        "closes_condition_ids references condition ids that are not "
        "currently unmet in recovery gap state",
        "claimed_condition_ids",
        " ({ids})",
    ),
    "claimed_condition_not_cleared": (
        "closes_condition_ids claims conditions that remain unmet "
        "after projected state transition",
        "claimed_condition_ids",
        " ({ids})",
    ),
    "reopened_continuation_condition": (
        "projected state transition reopens previously cleared "
        "continuation conditions",
        "reopened_condition_ids",
        " ({ids})",
    ),
}


def _format_axis_named_pose_not_available(
    task_id: str,
    fields: dict[str, str],
    finding: dict[str, Any],
) -> str:
    named_pose = str(finding.get("named_pose") or "").strip()
    if named_pose:
        return (
            f"task '{task_id}': resource '{fields['resource_jid']}' does not expose "
            f"named pose '{named_pose}'"
        )
    return (
        f"task '{task_id}': requested named pose is not available on "
        f"'{fields['resource_jid']}'"
    )


def _format_axis_held_part_conflict(
    task_id: str,
    fields: dict[str, str],
    finding: dict[str, Any],
) -> str:
    conflicting_part = str(finding.get("conflicting_part") or "").strip()
    conflicting_holder = str(finding.get("current_holder_resource_jid") or "").strip()
    if conflicting_part:
        return (
            f"task '{task_id}': resource '{fields['resource_jid']}' already holds "
            f"'{conflicting_part}' and cannot manipulate '{fields['part_name']}'"
        )
    if conflicting_holder:
        return (
            f"task '{task_id}': part '{fields['part_name']}' is currently held by "
            f"'{conflicting_holder}', not '{fields['resource_jid']}'"
        )
    return f"task '{task_id}': resource assignment conflicts with current held-part state"


def _format_axis_illegal_holder_swap(
    task_id: str,
    fields: dict[str, str],
    finding: dict[str, Any],
) -> str:
    conflicting_part = str(finding.get("conflicting_part") or "").strip()
    if conflicting_part and fields["part_name"]:
        return (
            f"task '{task_id}': resource '{fields['resource_jid']}' cannot swap directly "
            f"from holding '{conflicting_part}' to holding '{fields['part_name']}' in one "
            "macro-step"
        )
    return (
        f"task '{task_id}': resource '{fields['resource_jid']}' changes held parts in "
        "one macro-step without an explicit intermediate release/place"
    )


def _format_axis_safety_rule_violation(
    task_id: str,
    fields: dict[str, str],
    finding: dict[str, Any],
) -> str:
    rule_id = str(finding.get("rule_id") or "").strip()
    condition_ids = _finding_claimed_condition_ids(finding)
    reason = str(finding.get("reason") or "").strip()
    rule_text = f"safety rule '{rule_id}'" if rule_id else "an active bridge safety rule"
    if condition_ids and reason:
        return (
            f"task '{task_id}': projected macro violates {rule_text} while claimed "
            f"continuation condition {', '.join(condition_ids)} remains unsafe — {reason}"
        )
    if condition_ids:
        return (
            f"task '{task_id}': projected macro violates {rule_text} while claimed "
            f"continuation condition {', '.join(condition_ids)} remains unsafe"
        )
    if reason:
        return f"task '{task_id}': projected macro violates {rule_text} — {reason}"
    return f"task '{task_id}': projected macro violates {rule_text}"


def _format_axis_blocked_suffix_task_reused(
    task_id: str,
    fields: dict[str, str],
    finding: dict[str, Any],
) -> str:
    blocked_task_id = str(finding.get("blocked_task_id") or "").strip()
    blocked_part_name = str(finding.get("blocked_part_name") or "").strip()
    blocked_resource_jid = str(finding.get("blocked_resource_jid") or "").strip()
    condition_ids = _finding_claimed_condition_ids(finding)
    interaction_text = "the still-blocked suffix interaction"
    if blocked_part_name:
        interaction_text += f" for part '{blocked_part_name}'"
    if blocked_resource_jid:
        interaction_text += f" on resource '{blocked_resource_jid}'"
    if blocked_task_id:
        interaction_text += f" (blocked task '{blocked_task_id}')"
    if condition_ids:
        return (
            f"task '{task_id}': task reuses {interaction_text} while continuation "
            f"condition {', '.join(condition_ids)} remains unmet; the blocker must be "
            "cleared by a prerequisite recovery step first"
        )
    return (
        f"task '{task_id}': task reuses {interaction_text} while the cited safety "
        "condition remains unmet; the blocker must be cleared by a prerequisite "
        "recovery step first"
    )


# Axes that require custom logic beyond simple template interpolation.
# Signature: (task_id, fields, finding) -> str
_COMPLEX_AXIS_HANDLERS: dict[
    str,
    Callable[[str, dict[str, str], dict[str, Any]], str],
] = {
    "named_pose_not_available": _format_axis_named_pose_not_available,
    "held_part_conflict": _format_axis_held_part_conflict,
    "illegal_holder_swap": _format_axis_illegal_holder_swap,
    "safety_rule_violation": _format_axis_safety_rule_violation,
    "blocked_suffix_task_reused": _format_axis_blocked_suffix_task_reused,
}


def _format_outline_validation_finding(finding: dict[str, Any]) -> str:
    """Format a single outline validation finding into a human-readable string."""
    task_id = str(finding.get("task_id") or "").strip()
    resource_jid = str(finding.get("resource_jid") or "").strip()
    part_name = str(finding.get("part_name") or "").strip()
    pose_source = str(finding.get("pose_source") or "").strip()
    constraint_code = str(finding.get("constraint_code") or "").strip()
    constraint_owner = str(finding.get("constraint_owner") or "").strip()
    reason = str(finding.get("reason") or "").strip()
    failed_axes = _extract_string_list(finding, "failed_axes")

    fields = {
        "task_id": task_id,
        "resource_jid": resource_jid,
        "part_name": part_name,
        "part_name_display": part_name or "-",
    }

    # --- early exits for special structural cases ---
    if failed_axes == ["pruned_action"]:
        pruned_reason = str(
            finding.get("blocked_reason") or finding.get("reason") or ""
        ).strip()
        if pruned_reason:
            return f"task '{task_id}': action is currently pruned in this state — {pruned_reason}"
        return f"task '{task_id}': action is currently pruned in this state"

    if constraint_owner == "binding":
        if reason:
            return f"task '{task_id}': {reason}"
        return f"task '{task_id}': task could not be grounded into one concrete action"

    # --- constraint_code lookup (reason overrides fallback) ---
    if constraint_code in _CONSTRAINT_CODE_FALLBACKS:
        if reason:
            return f"task '{task_id}': {reason}"
        return f"task '{task_id}': {_CONSTRAINT_CODE_FALLBACKS[constraint_code].format_map(fields)}"

    # --- failed_axes dispatch (single-axis only) ---
    axis = failed_axes[0] if len(failed_axes) == 1 else ""

    if axis in _COMPLEX_AXIS_HANDLERS:
        return _COMPLEX_AXIS_HANDLERS[axis](task_id, fields, finding)

    if axis in _SIMPLE_AXIS_MESSAGES:
        return f"task '{task_id}': {_SIMPLE_AXIS_MESSAGES[axis].format_map(fields)}"

    if axis in _ENRICHED_AXIS_SPECS:
        base_msg, finding_key, suffix_fmt = _ENRICHED_AXIS_SPECS[axis]
        ids = _extract_string_list(finding, finding_key)
        if ids:
            return f"task '{task_id}': {base_msg}{suffix_fmt.format(ids=', '.join(ids))}"
        return f"task '{task_id}': {base_msg}"

    # --- generic pose-based fallbacks ---
    axes_text = ", ".join(failed_axes)
    if pose_source == "observed_pose" and part_name:
        return (
            f"task '{task_id}': {resource_jid} cannot reach {part_name} at observed pose "
            f"— {axes_text}"
        )
    if pose_source:
        return (
            f"task '{task_id}': {resource_jid} {pose_source} pose is outside workspace "
            f"— {axes_text}"
        )
    return f"task '{task_id}': validation failed — {axes_text}"


def _analyze_outline_task_validation(
    task: dict[str, Any],
    *,
    outline_tasks: list[dict[str, Any]],
    resources_by_jid: dict[str, dict[str, Any]],
    parts_by_name: dict[str, dict[str, Any]],
    llm_input: dict[str, Any],
    task_types_by_id: dict[str, str],
    task_index_by_id: dict[str, int],
    dependency_map: dict[str, list[str]],
    active_pruned_actions: list[dict[str, Any]] | None = None,
    planner: Any | None = None,
    previously_cleared_condition_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    task_id = str(task.get("outline_id") or "").strip()
    resource_jid = str(task.get("resource_jid") or "").strip()
    task_part_name = str(task.get("part_name") or "").strip()
    active_pruned_action = _matching_active_pruned_action(
        task,
        pruned_actions=list(active_pruned_actions or []),
        resources_by_jid=resources_by_jid,
        parts_by_name=parts_by_name,
        llm_input=llm_input,
    )
    if active_pruned_action is not None:
        findings.append(
            {
                "task_id": task_id,
                "resource_jid": resource_jid or None,
                "part_name": task_part_name or None,
                "pose_source": str(
                    dict(active_pruned_action.get("guard") or {}).get("pose_source") or "task_contract"
                ).strip(),
                "pose": None,
                "workspace_bounds": None,
                "failed_axes": ["pruned_action"],
                "blocked_reason": str(active_pruned_action.get("reason") or "").strip(),
            }
        )
        return findings

    grounding_result = compile_grounded_outline_task(
        task,
        resources_by_jid=resources_by_jid,
        parts_by_name=parts_by_name,
    )
    grounding_finding = grounding_result.get("finding")
    if isinstance(grounding_finding, dict):
        findings.append(deepcopy(grounding_finding))
        return findings

    grounded_action = dict(grounding_result.get("grounded_action") or {})
    normalized_task = _task_with_grounded_defaults(task, grounded_action)
    resource_jid = str(grounded_action.get("resource_jid") or normalized_task.get("resource_jid") or "").strip()
    signature = _infer_outline_macro_signature(
        normalized_task,
        resources_by_jid=resources_by_jid,
        parts_by_name=parts_by_name,
    )
    task_type = str(grounded_action.get("task_kind") or signature.get("task_kind") or "").strip() or str(
        task_types_by_id.get(task_id) or ""
    ).strip() or _classify_outline_task(
        normalized_task,
        resources_by_jid=resources_by_jid,
        parts_by_name=parts_by_name,
        llm_input=llm_input,
    )

    resource_findings = _validate_outline_task_resource_constraints(
        planner=planner,
        task=normalized_task,
        grounded_action=grounded_action,
        resources_by_jid=resources_by_jid,
        parts_by_name=parts_by_name,
    )
    findings = _merge_outline_validation_findings(findings, resource_findings)

    projected_resources = deepcopy(resources_by_jid)
    projected_parts = deepcopy(parts_by_name)
    if not _task_findings_block_projected_state(findings):
        _apply_outline_task_effects(
            normalized_task,
            resources_by_jid=projected_resources,
            parts_by_name=projected_parts,
            task_type=task_type,
            grounded_action=grounded_action,
        )

    cca_result = validate_outline_macro_cca_constraints(
        task=normalized_task,
        grounded_action=grounded_action,
        signature=signature,
        pre_resources=resources_by_jid,
        pre_parts=parts_by_name,
        projected_resources=projected_resources,
        projected_parts=projected_parts,
        llm_input=llm_input,
        outline_tasks=outline_tasks,
        task_types_by_id=task_types_by_id,
        task_index_by_id=task_index_by_id,
        dependency_map=dependency_map,
        previously_cleared_condition_ids=previously_cleared_condition_ids,
    )
    findings = _merge_outline_validation_findings(
        findings,
        list(cca_result.get("findings") or []),
    )
    return findings


def _compute_outline_validation_findings(
    outline_tasks: list[dict[str, Any]],
    llm_input: dict[str, Any],
    pruned_actions: list[dict[str, Any]] | None = None,
    planner: Any | None = None,
) -> list[dict[str, Any]]:
    resources_by_jid, parts_by_name = _outline_validation_context(llm_input)
    symbolic_resources = deepcopy(resources_by_jid)
    symbolic_parts = deepcopy(parts_by_name)
    task_types_by_id = _build_outline_task_type_lookup(
        outline_tasks,
        resources_by_jid=resources_by_jid,
        parts_by_name=parts_by_name,
        llm_input=llm_input,
    )
    task_index_by_id = {
        str(dict(task or {}).get("outline_id") or "").strip() or f"task_{index}": index
        for index, task in enumerate(outline_tasks)
        if isinstance(task, dict)
    }
    dependency_map = _outline_dependency_map(outline_tasks)
    validation_findings: list[dict[str, Any]] = []
    cleared_condition_ids: list[str] = []

    for task in _outline_rollout_tasks(outline_tasks):
        if not isinstance(task, dict):
            continue
        task_findings = _analyze_outline_task_validation(
            task,
            outline_tasks=outline_tasks,
            resources_by_jid=symbolic_resources,
            parts_by_name=symbolic_parts,
            llm_input=llm_input,
            task_types_by_id=task_types_by_id,
            task_index_by_id=task_index_by_id,
            dependency_map=dependency_map,
            active_pruned_actions=pruned_actions,
            planner=planner,
            previously_cleared_condition_ids=cleared_condition_ids,
        )
        validation_findings.extend(task_findings)
        task_id = str(task.get("outline_id") or "").strip()
        task_type = str(task_types_by_id.get(task_id) or "").strip()
        if not task_findings:
            grounding_result = compile_grounded_outline_task(
                task,
                resources_by_jid=symbolic_resources,
                parts_by_name=symbolic_parts,
            )
            _apply_outline_task_effects(
                task,
                resources_by_jid=symbolic_resources,
                parts_by_name=symbolic_parts,
                task_type=task_type,
                grounded_action=dict(grounding_result.get("grounded_action") or {}),
            )
            _, unmet_condition_ids, _ = _current_recovery_gap_condition_status(
                resources_by_jid=symbolic_resources,
                parts_by_name=symbolic_parts,
                llm_input=llm_input,
            )
            cleared_condition_ids = [
                condition_id
                for condition_id in (
                    str(dict(row).get("condition_id") or "").strip()
                    for row in (
                        dict(llm_input.get("modeled_continuation_gap") or {}).get(
                            "unmet_continuation_conditions"
                        )
                        or []
                    )
                )
                if condition_id and condition_id not in unmet_condition_ids
            ]

    return [_annotate_outline_validation_finding(finding) for finding in validation_findings]


def _validate_outline_tasks(
    outline_tasks: list[dict[str, Any]],
    llm_input: dict[str, Any],
    pruned_actions: list[dict[str, Any]] | None = None,
    planner: Any | None = None,
) -> list[str]:
    """Validate outline tasks against physical constraints and symbolic sequence state."""
    violations: list[str] = []
    for finding in _compute_outline_validation_findings(
        outline_tasks,
        llm_input,
        pruned_actions=pruned_actions,
        planner=planner,
    ):
        violations.append(_format_outline_validation_finding(finding))
    return violations


def _build_outline_validation_findings(
    outline_tasks: list[dict[str, Any]],
    llm_input: dict[str, Any],
    pruned_actions: list[dict[str, Any]] | None = None,
    planner: Any | None = None,
) -> list[dict[str, Any]]:
    return _compute_outline_validation_findings(
        outline_tasks,
        llm_input,
        pruned_actions=pruned_actions,
        planner=planner,
    )


def _build_phase_prompt_artifacts(
    prepared_bridge_request: dict[str, Any],
    session_state: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    phase = str(session_state.get("current_phase") or "grounding").strip().lower()
    llm_input = _overlay_session_observations_on_llm_input(
        deepcopy(prepared_bridge_request.get("llm_input") or {}),
        session_state,
    )
    recovery_gap_state: dict[str, Any] = {}
    grounded_feasibility_facts: list[dict[str, Any]] = []
    pruned_actions: list[dict[str, Any]] = []
    if phase == "outline":
        resources_by_jid, parts_by_name = _outline_validation_context(llm_input)
        recovery_gap_state = _build_recovery_gap_state(
            llm_input,
            resources_by_jid=resources_by_jid,
            parts_by_name=parts_by_name,
        )
        grounded_feasibility_facts = _build_grounded_feasibility_facts(
            llm_input,
            resources_by_jid=resources_by_jid,
            parts_by_name=parts_by_name,
        )
        pruned_actions = _active_pruned_actions_for_state(
            list(session_state.get("pruned_actions") or []),
            resources_by_jid=resources_by_jid,
            parts_by_name=parts_by_name,
            llm_input=llm_input,
        )
    prompt_session_state = deepcopy(session_state)
    if phase == "outline":
        prompt_session_state["pruned_actions"] = deepcopy(pruned_actions)
    prompt_input = build_multi_turn_phase_prompt_input(
        phase=phase,
        llm_input=llm_input,
        session_state=prompt_session_state,
        world_observation_surface=_build_world_observation_surface(prepared_bridge_request),
        recovery_gap_state=recovery_gap_state,
        grounded_feasibility_facts=grounded_feasibility_facts,
        pruned_actions=pruned_actions,
    )
    prompt_text = render_multi_turn_phase_prompt(prompt_input)
    return prompt_input, prompt_text


def _lookup_world_observation_primitive(
    prepared_bridge_request: dict[str, Any],
    *,
    primitive_name: str,
) -> tuple[str, dict[str, Any], str]:
    resource_jid, bridge_entry, resource_type = _focused_observation_resource_entry(
        prepared_bridge_request
    )
    for primitive_entry in (bridge_entry.get("primitive_catalog") or []):
        if not isinstance(primitive_entry, dict):
            continue
        if str(primitive_entry.get("name") or "").strip() != primitive_name:
            continue
        if str(primitive_entry.get("primitive_kind") or "").strip().lower() != "observe":
            break
        if not _is_grounding_observation_primitive(
            resource_type=resource_type,
            primitive_name=primitive_name,
        ):
            break
        if not _supports_store_as(resource_type=resource_type, primitive_name=primitive_name):
            break
        return resource_jid, primitive_entry, resource_type
    return resource_jid, {}, resource_type


async def _invoke_primitive(resource_agent: Any, primitive_name: str, params: dict[str, Any]) -> Any:
    profile = get_resource_profile_for_agent(resource_agent)
    owner = resource_agent
    if profile.primitive_owner_resolver is not None:
        try:
            owner = profile.primitive_owner_resolver(resource_agent) or resource_agent
        except Exception:
            owner = resource_agent
    fn = getattr(resource_agent, primitive_name, None)
    if not callable(fn):
        fn = getattr(owner, primitive_name, None)
    if not callable(fn):
        raise LookupError(f"primitive '{primitive_name}' is not callable on resource")
    if inspect.iscoroutinefunction(fn):
        return await fn(**params)
    return fn(**params)


async def _execute_observe_requests(
    planner: Any,
    prepared_bridge_request: dict[str, Any],
    session_state: dict[str, Any],
    observe_requests: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], str | None]:
    resource_agents = _resource_agent_map(planner)
    results: list[dict[str, Any]] = []
    seen_aliases = set(dict(session_state.get("observation_store") or {}))
    seen_fact_rows = {
        str(fact_key).strip(): dict(row)
        for fact_key, row in dict(session_state.get("observation_fact_ledger") or {}).items()
        if str(fact_key).strip() and isinstance(row, dict)
    }
    seen_request_rows = {
        _observation_request_key(
            str(row.get("primitive") or "").strip(),
            dict(row.get("params") or {}),
        ): dict(row)
        for row in (session_state.get("observation_history") or [])
        if isinstance(row, dict)
    }

    for raw_request in observe_requests:
        normalized_request, resolve_error = _resolve_observation_request(
            prepared_bridge_request,
            dict(raw_request or {}),
        )
        if resolve_error is not None or not isinstance(normalized_request, dict):
            return [], resolve_error or "failed to resolve observation request"
        primitive_name = str(normalized_request.get("primitive") or "").strip()
        params = dict(normalized_request.get("params") or {})
        request_key = _observation_request_key(primitive_name, params)
        fact_key = str(normalized_request.get("fact_key") or "").strip()
        if not primitive_name:
            return [], "observe_requests must include fact_type and entity"
        prior_fact = dict(seen_fact_rows.get(fact_key) or {}) if fact_key else {}
        if prior_fact and str(prior_fact.get("validity") or "current").strip().lower() != "stale":
            return [], _render_prior_fact_message(prior_fact)
        prior_observation = seen_request_rows.get(request_key)
        if not prior_fact and isinstance(prior_observation, dict):
            message = (
                f"observation {primitive_name!r} with params "
                f"{json.dumps(params, sort_keys=True, default=str, ensure_ascii=True)} "
                "already succeeded earlier"
            )
            return [], message
        store_as = _default_observe_store_as(
            normalized_request,
            session_state=session_state,
            seen_aliases=seen_aliases,
        )
        normalized_request["store_as"] = store_as

        resource_jid = str(normalized_request.get("resource_jid") or "").strip()
        resource_type = str(normalized_request.get("resource_type") or "").strip() or "resource"
        _, primitive_entry, _ = _lookup_world_observation_primitive(
            prepared_bridge_request,
            primitive_name=primitive_name,
        )
        if not primitive_entry:
            return [], f"world observation primitive {primitive_name!r} is not allowed"

        required_params = [
            str(param).strip()
            for param in (primitive_entry.get("required_params") or [])
            if str(param).strip()
        ]
        for required_param in required_params:
            if required_param not in params or params.get(required_param) is None:
                return [], (
                    f"observe request {primitive_name!r} on {resource_jid!r} is "
                    f"missing required param {required_param!r}"
                )
        store_as_contract_error = _validate_store_as_contract(
            params=params,
            resource_type=resource_type,
            primitive_name=primitive_name,
        )
        if store_as_contract_error is not None:
            return [], store_as_contract_error

        resource_agent = resource_agents.get(resource_jid)
        if resource_agent is None:
            return [], f"resource agent {resource_jid!r} is not available for observation"

        raw_result = await _invoke_primitive(resource_agent, primitive_name, params)
        normalized_result = (
            deepcopy(raw_result)
            if isinstance(raw_result, dict)
            else {"success": True, "data": deepcopy(raw_result)}
        )
        extracted_output, extract_error = extract_step_output(
            primitive=primitive_name,
            params=params,
            step_result=normalized_result,
            resource_type=resource_type,
        )
        if extract_error is not None or extracted_output is None:
            return [], (
                f"observation primitive {primitive_name!r} on {resource_jid!r} did not "
                f"produce a reusable output: {extract_error or 'unknown error'}"
            )

        observation_row = {
            "fact_key": fact_key,
            "fact_type": str(normalized_request.get("fact_type") or "").strip(),
            "entity": str(normalized_request.get("entity") or "").strip(),
            "entity_kind": deepcopy(normalized_request.get("entity_kind")),
            "scope": deepcopy(normalized_request.get("scope")),
            "reason": str(normalized_request.get("reason") or "").strip(),
            "primitive": primitive_name,
            "params": deepcopy(params),
            "store_as": store_as,
            "output": deepcopy(extracted_output),
        }
        results.append(observation_row)
        seen_aliases.add(store_as)
        seen_request_rows[request_key] = deepcopy(observation_row)
        if fact_key:
            seen_fact_rows[fact_key] = {
                "fact_key": fact_key,
                "fact_type": str(normalized_request.get("fact_type") or "").strip(),
                "entity": str(normalized_request.get("entity") or "").strip(),
                "entity_kind": deepcopy(normalized_request.get("entity_kind")),
                "scope": deepcopy(normalized_request.get("scope")),
                "primitive": primitive_name,
                "params": deepcopy(params),
                "output": deepcopy(extracted_output),
                "turn_index": int(session_state.get("turn_index") or 0),
                "validity": "current",
                "freshness": "current_session",
                "aliases": [store_as],
            }
    return results, None


def _build_phase_feedback(
    *,
    phase: str,
    decision: str,
    thought: str,
    detail: Any = None,
) -> dict[str, Any]:
    feedback = {
        "phase": str(phase or "").strip().lower(),
        "decision": str(decision or "").strip().lower(),
        "thought": str(thought or "").strip(),
    }
    if detail not in (None, "", [], {}):
        feedback["detail"] = deepcopy(detail)
    return feedback


def _append_turn(
    planner: Any,
    prepared_bridge_request: dict[str, Any],
    session_state: dict[str, Any],
    turn_entry: dict[str, Any],
) -> None:
    session_state["turns"].append(deepcopy(turn_entry))
    bridge_debug = deepcopy(prepared_bridge_request.get("bridge_debug") or {})
    bridge_debug["status"] = deepcopy(session_state.get("status"))
    bridge_debug["multi_turn_session"] = deepcopy(session_state)
    prepared_bridge_request["bridge_debug"] = deepcopy(bridge_debug)
    if hasattr(planner, "_set_last_bridge_debug"):
        planner._set_last_bridge_debug(bridge_debug)
    per_turn_debug_dir = str(bridge_debug.get("per_turn_debug_dir") or "").strip()
    if per_turn_debug_dir:
        per_turn_payload = {
            **deepcopy(prepared_bridge_request),
            "reasoning_mode": "multi_turn",
            "multi_turn_prompt_input": deepcopy(turn_entry.get("prompt_input")),
            "multi_turn_prompt_text": str(turn_entry.get("prompt_text") or ""),
            "multi_turn_raw_response": deepcopy(turn_entry.get("raw_response")),
            "multi_turn_session_result": deepcopy(session_state),
        }
        try:
            write_bridge_artifacts(
                per_turn_payload,
                phase_label="multi_turn",
                debug_dir=per_turn_debug_dir,
                write_latest=True,
            )
        except Exception as exc:
            _logger.warning("[MultiTurn] Failed to write per-turn artifact: %s", exc)


async def execute_multi_turn_bridge(
    planner: Any,
    prepared_bridge_request: dict[str, Any],
) -> dict[str, Any] | None:
    product_agent = getattr(planner, "product_agent", None)
    ask_llm_structured = getattr(product_agent, "ask_llm_structured", None)
    if not callable(ask_llm_structured):
        raise RuntimeError("product_agent.ask_llm_structured is required for multi-turn bridge execution")

    session_state = deepcopy(
        prepared_bridge_request.get("multi_turn_session_seed")
        or build_multi_turn_session_seed(prepared_bridge_request)
    )
    session_state["status"] = "running"

    bridge_debug = deepcopy(prepared_bridge_request.get("bridge_debug") or {})
    bridge_debug["multi_turn_session"] = deepcopy(session_state)
    prepared_bridge_request["bridge_debug"] = deepcopy(bridge_debug)
    if hasattr(planner, "_set_last_bridge_debug"):
        planner._set_last_bridge_debug(bridge_debug)

    while int(session_state.get("turn_index") or 0) < int(session_state.get("max_turns") or 0):
        session_state["turn_index"] = int(session_state.get("turn_index") or 0) + 1
        current_phase = str(session_state.get("current_phase") or "grounding").strip().lower()
        turn_idx = int(session_state.get("turn_index") or 0)
        max_turns = int(session_state.get("max_turns") or 0)
        _logger.info("[MultiTurn] Turn %d/%d | phase=%s", turn_idx, max_turns, current_phase)
        prompt_input, prompt_text = _build_phase_prompt_artifacts(
            prepared_bridge_request,
            session_state,
        )
        raw_response = await ask_llm_structured(
            prompt=prompt_text,
            response_format=multi_turn_phase_response_schema(current_phase),
        )
        parsed_response = deepcopy(raw_response if isinstance(raw_response, dict) else {})
        raw_decision = str(parsed_response.get("decision") or "").strip().lower()
        thought = str(parsed_response.get("thought") or "").strip()
        turn_entry: dict[str, Any] = {
            "turn_index": int(session_state.get("turn_index") or 0),
            "phase": current_phase,
            "prompt_input": deepcopy(prompt_input),
            "prompt_text": prompt_text,
            "raw_response": deepcopy(parsed_response),
            "thought": thought,
        }
        if raw_decision:
            turn_entry["raw_decision"] = raw_decision

        decision = ""
        decision_compatibility: dict[str, Any] | None = None
        next_phase = current_phase
        if current_phase != "outline":
            decision, decision_compatibility = _normalize_phase_decision(
                current_phase=current_phase,
                decision=raw_decision,
            )
            turn_entry["decision"] = decision
            if decision_compatibility is not None:
                _logger.warning(
                    "[MultiTurn] Turn %d/%d | decision compatibility downgrade: %s -> %s",
                    turn_idx,
                    max_turns,
                    raw_decision,
                    decision,
                )
                turn_entry["decision_compatibility"] = deepcopy(decision_compatibility)
            else:
                _logger.info(
                    "[MultiTurn] Turn %d/%d | decision=%s",
                    turn_idx,
                    max_turns,
                    decision,
                )

            try:
                next_phase = transition_multi_turn_phase(current_phase, decision)
            except ValueError as exc:
                session_state["status"] = "invalid_phase_decision"
                turn_entry["error"] = str(exc)
                _append_turn(planner, prepared_bridge_request, session_state, turn_entry)
                return None

            if decision_compatibility is not None:
                session_state["phase_feedback"].append(
                    _build_phase_feedback(
                        phase=current_phase,
                        decision=decision,
                        thought=thought,
                        detail={
                            "compatibility_downgrade": deepcopy(decision_compatibility),
                        },
                    )
                )

        if current_phase == "grounding":
            blocking_reasons = [
                str(item).strip()
                for item in (
                    parsed_response.get("blocking_reasons")
                    or parsed_response.get("blocking_summary")
                    or []
                )
                if str(item).strip()
            ]
            observe_requests = [
                {
                    key: deepcopy(value)
                    for key, value in dict(row or {}).items()
                    if key != "resource_jid"
                }
                for row in (parsed_response.get("observe_requests") or [])
                if isinstance(row, dict)
            ]
            raw_sufficient_grounding = parsed_response.get("sufficient_grounding")
            sufficient_grounding = decision == "grounded"
            observe_reason = str(parsed_response.get("observe_reason") or "").strip()
            grounded_facts = [
                str(item).strip()
                for item in (parsed_response.get("grounded_facts") or [])
                if str(item).strip()
            ]
            recovery_implications = [
                str(item).strip()
                for item in (parsed_response.get("recovery_implications") or [])
                if str(item).strip()
            ]
            grounding_decision_normalization: dict[str, Any] | None = None
            if raw_sufficient_grounding is not None:
                raw_sufficient_grounding_bool = bool(raw_sufficient_grounding)
                if raw_sufficient_grounding_bool != sufficient_grounding:
                    grounding_decision_normalization = {
                        "field": "sufficient_grounding",
                        "from": raw_sufficient_grounding_bool,
                        "to": sufficient_grounding,
                        "reason": (
                            "grounding decision controls whether grounding is sufficient; "
                            "the runtime normalized the legacy auxiliary boolean to match the decision"
                        ),
                    }
            turn_entry["blocking_reasons"] = deepcopy(blocking_reasons)
            turn_entry["observe_requests"] = deepcopy(observe_requests)
            turn_entry["grounded_facts"] = deepcopy(grounded_facts)
            turn_entry["recovery_implications"] = deepcopy(recovery_implications)
            turn_entry["sufficient_grounding"] = sufficient_grounding
            if observe_reason:
                turn_entry["observe_reason"] = observe_reason
            if grounding_decision_normalization is not None:
                turn_entry["grounding_decision_normalization"] = deepcopy(
                    grounding_decision_normalization
                )
                _logger.warning(
                    "[MultiTurn] Turn %d/%d | grounding normalization: %s",
                    turn_idx,
                    max_turns,
                    str(grounding_decision_normalization.get("reason") or "").strip(),
                )
            if decision == "observe":
                if not observe_requests:
                    grounding_contract = _grounding_contract_failure_empty_observe(
                        session_state=session_state
                    )
                    turn_entry["grounding_contract"] = deepcopy(grounding_contract)
                    session_state["grounding_contract"] = deepcopy(grounding_contract)
                    _logger.warning(
                        "[MultiTurn] Turn %d/%d | grounding contract failed: observe without "
                        "observe_requests",
                        turn_idx,
                        max_turns,
                    )
                    session_state["phase_feedback"].append(
                        _build_phase_feedback(
                            phase=current_phase,
                            decision=decision,
                            thought=thought,
                            detail={
                                "grounding_contract": deepcopy(grounding_contract),
                            },
                        )
                    )
                    _append_turn(planner, prepared_bridge_request, session_state, turn_entry)
                    continue
                _logger.info(
                    "[MultiTurn] Turn %d/%d | observe requested (%d facts) — %s",
                    turn_idx, max_turns, len(observe_requests), observe_reason or "(no reason)",
                )
            if decision == "observe":
                if not observe_reason:
                    session_state["status"] = "invalid_observe_request"
                    turn_entry["error"] = "grounding decision 'observe' requires observe_reason"
                    _append_turn(planner, prepared_bridge_request, session_state, turn_entry)
                    return None
                if len(observe_requests) > int(session_state.get("max_observe_batch") or 0):
                    session_state["status"] = "observe_batch_exhausted"
                    turn_entry["error"] = (
                        f"observe_requests count exceeds max_observe_batch="
                        f"{int(session_state.get('max_observe_batch') or 0)}"
                    )
                    _append_turn(planner, prepared_bridge_request, session_state, turn_entry)
                    return None
                remaining_observations = max(
                    0,
                    int(session_state.get("max_observations") or 0)
                    - int(session_state.get("observation_count") or 0),
                )
                if len(observe_requests) > remaining_observations:
                    session_state["status"] = "observation_budget_exhausted"
                    turn_entry["error"] = "observation budget exhausted"
                    _append_turn(planner, prepared_bridge_request, session_state, turn_entry)
                    return None
                observation_results, observe_error = await _execute_observe_requests(
                    planner,
                    prepared_bridge_request,
                    session_state,
                    observe_requests,
                )
                if observe_error is not None:
                    _logger.warning(
                        "[MultiTurn] Turn %d/%d | observe FAILED: %s",
                        turn_idx, max_turns, observe_error,
                    )
                    turn_entry["error"] = observe_error
                    if _is_redundant_grounding_observe_error(observe_error):
                        grounding_contract = _grounding_contract_failure_redundant_observe(
                            prepared_bridge_request=prepared_bridge_request,
                            session_state=session_state,
                            observe_requests=observe_requests,
                        )
                        turn_entry["grounding_contract"] = deepcopy(grounding_contract)
                        session_state["grounding_contract"] = deepcopy(grounding_contract)
                        session_state["phase_feedback"].append(
                            _build_phase_feedback(
                                phase=current_phase,
                                decision=decision,
                                thought=thought,
                                detail={
                                    "grounding_contract": deepcopy(grounding_contract),
                                    "observe_error": observe_error,
                                    "observe_requests": observe_requests,
                                },
                            )
                        )
                        _append_turn(planner, prepared_bridge_request, session_state, turn_entry)
                        continue
                    if _is_terminal_grounding_observe_error(observe_error):
                        session_state["status"] = "invalid_observe_request"
                        _append_turn(planner, prepared_bridge_request, session_state, turn_entry)
                        return None
                    session_state["phase_feedback"].append(
                        _build_phase_feedback(
                            phase=current_phase,
                            decision=decision,
                            thought=thought,
                            detail={
                                "observe_error": observe_error,
                                "observe_requests": observe_requests,
                            },
                        )
                    )
                    _append_turn(planner, prepared_bridge_request, session_state, turn_entry)
                    continue
                for result in observation_results:
                    alias = str(result.get("store_as") or "").strip()
                    session_state["observation_store"][alias] = deepcopy(
                        result.get("output") or {}
                    )
                    observation_event = {
                        "turn_index": int(session_state.get("turn_index") or 0),
                        "phase": current_phase,
                        **deepcopy(result),
                    }
                    session_state["observation_history"].append(observation_event)
                    _record_observation_fact(
                        session_state,
                        observation_event,
                    )
                session_state["observation_count"] = int(session_state.get("observation_count") or 0) + len(
                    observation_results
                )
                turn_entry["observation_results"] = deepcopy(observation_results)
                session_state["phase_feedback"].append(
                    _build_phase_feedback(
                        phase=current_phase,
                        decision=decision,
                        thought=thought,
                        detail={
                            "grounded_facts": grounded_facts,
                            "recovery_implications": recovery_implications,
                            "blocking_reasons": blocking_reasons,
                            "observe_reason": observe_reason,
                            "observe_requests": observe_requests,
                            "observation_results": observation_results,
                        },
                    )
                )
                session_state.pop("grounding_contract", None)
            else:
                _logger.info("[MultiTurn] Turn %d/%d | grounded ✓", turn_idx, max_turns)
                if not sufficient_grounding:
                    session_state["status"] = "invalid_grounding_decision"
                    turn_entry["error"] = (
                        "grounding decision 'grounded' is inconsistent with "
                        "sufficient_grounding=false"
                    )
                    _append_turn(planner, prepared_bridge_request, session_state, turn_entry)
                    return None
                session_state["phase_feedback"].append(
                    _build_phase_feedback(
                        phase=current_phase,
                        decision=decision,
                        thought=thought,
                        detail={
                            "blocking_reasons": deepcopy(blocking_reasons),
                            "grounded_facts": grounded_facts,
                            "recovery_implications": recovery_implications,
                        },
                    )
                )
                session_state.pop("grounding_contract", None)
                stop_after_phase = str(session_state.get("stop_after_phase") or "").strip().lower()
                if stop_after_phase == "grounding":
                    session_state["current_phase"] = next_phase
                    session_state["status"] = "paused_after_grounding"
                    _append_turn(planner, prepared_bridge_request, session_state, turn_entry)
                    prepared_bridge_request["multi_turn_session_result"] = deepcopy(session_state)
                    bridge_debug = deepcopy(prepared_bridge_request.get("bridge_debug") or {})
                    bridge_debug["status"] = "paused_after_grounding"
                    bridge_debug["multi_turn_session"] = deepcopy(session_state)
                    prepared_bridge_request["bridge_debug"] = deepcopy(bridge_debug)
                    if hasattr(planner, "_set_last_bridge_debug"):
                        planner._set_last_bridge_debug(bridge_debug)
                    return None

        elif current_phase == "outline":
            outstanding_outline_validation_findings = [
                deepcopy(row)
                for row in (session_state.get("outline_validation_findings") or [])
                if isinstance(row, dict)
            ]
            addressed_validation_findings = _extract_addressed_validation_findings(parsed_response)
            turn_entry["addressed_validation_findings"] = deepcopy(
                addressed_validation_findings
            )
            outline_tasks = [
                deepcopy(row)
                for row in (parsed_response.get("outline_tasks") or [])
                if isinstance(row, dict)
            ]
            turn_entry["outline_tasks"] = deepcopy(outline_tasks)
            outline_llm_input = dict(prompt_input.get("llm_input") or {})
            resources_by_jid, parts_by_name = _outline_validation_context(outline_llm_input)
            outline_task_types = [
                {
                    "outline_id": str(dict(row or {}).get("outline_id") or "").strip(),
                    "task_type": _classify_outline_task(
                        dict(row or {}),
                        resources_by_jid=resources_by_jid,
                        parts_by_name=parts_by_name,
                        llm_input=outline_llm_input,
                    ),
                }
                for row in outline_tasks
                if isinstance(row, dict)
            ]
            turn_entry["outline_task_types"] = deepcopy(outline_task_types)
            coverage_violations, carried_forward_findings, outline_revision_coverage = (
                _validate_addressed_validation_findings(
                    addressed_refs=addressed_validation_findings,
                    outstanding_findings=outstanding_outline_validation_findings,
                )
            )
            if outline_revision_coverage is not None:
                turn_entry["outline_revision_coverage"] = deepcopy(outline_revision_coverage)
            outline_violations = deepcopy(coverage_violations)
            task_outline_validation_findings = _build_outline_validation_findings(
                outline_tasks,
                outline_llm_input,
                pruned_actions=list(session_state.get("pruned_actions") or []),
                planner=planner,
            )
            task_outline_violations = [
                _format_outline_validation_finding(finding)
                for finding in task_outline_validation_findings
            ]
            outline_violations.extend(task_outline_violations)
            if outline_violations:
                outline_validation_findings = _merge_outline_validation_findings(
                    carried_forward_findings,
                    task_outline_validation_findings,
                )
                _logger.warning(
                    "[MultiTurn] Turn %d/%d | outline validation failed: %s",
                    turn_idx,
                    max_turns,
                    outline_violations,
                )
                decision = "need_revision"
                next_phase = transition_multi_turn_phase(current_phase, decision)
                turn_entry["validation_violations"] = outline_violations
                turn_entry["outline_validation_findings"] = deepcopy(
                    outline_validation_findings
                )
                outline_validation = {
                    "status": "failed",
                    "violations": deepcopy(outline_violations),
                    "findings": deepcopy(outline_validation_findings),
                }
                session_state["outline_violations"] = deepcopy(outline_violations)
                session_state.pop("outline_revision_guidance", None)
                if (
                    isinstance(outline_revision_coverage, dict)
                    and str(outline_revision_coverage.get("status") or "").strip().lower()
                    == "failed"
                ):
                    session_state["outline_revision_coverage"] = deepcopy(
                        outline_revision_coverage
                    )
                else:
                    session_state.pop("outline_revision_coverage", None)
                session_state["outline_validation_findings"] = deepcopy(
                    outline_validation_findings
                )
                session_state["outline_task_types"] = deepcopy(outline_task_types)
                session_state["pruned_actions"] = _build_pruned_actions(
                    existing_pruned_actions=list(session_state.get("pruned_actions") or []),
                    outline_tasks=outline_tasks,
                    validation_findings=task_outline_validation_findings,
                    llm_input=outline_llm_input,
                )
            else:
                decision = "outline_ready"
                next_phase = transition_multi_turn_phase(current_phase, decision)
                outline_validation = {"status": "passed"}
                session_state.pop("outline_violations", None)
                session_state.pop("outline_revision_guidance", None)
                session_state.pop("outline_revision_coverage", None)
                session_state.pop("outline_validation_findings", None)
                session_state["outline_task_types"] = deepcopy(outline_task_types)
                session_state["pruned_actions"] = _active_pruned_actions_for_state(
                    list(session_state.get("pruned_actions") or []),
                    resources_by_jid=resources_by_jid,
                    parts_by_name=parts_by_name,
                    llm_input=outline_llm_input,
                )
                session_state["accepted_outline"] = deepcopy(outline_tasks)
            _logger.info(
                "[MultiTurn] Turn %d/%d | outline runtime decision=%s",
                turn_idx,
                max_turns,
                decision,
            )
            turn_entry["decision"] = decision
            turn_entry["outline_validation"] = deepcopy(outline_validation)
            session_state["last_outline_validation"] = deepcopy(outline_validation)
            stop_after_phase = str(session_state.get("stop_after_phase") or "").strip().lower()
            if stop_after_phase == "outline" and not outline_violations:
                session_state["paused_before_phase_transition"] = next_phase
                session_state["status"] = "paused_after_outline"
                _append_turn(planner, prepared_bridge_request, session_state, turn_entry)
                prepared_bridge_request["multi_turn_session_result"] = deepcopy(session_state)
                bridge_debug = deepcopy(prepared_bridge_request.get("bridge_debug") or {})
                bridge_debug["status"] = "paused_after_outline"
                bridge_debug["multi_turn_session"] = deepcopy(session_state)
                prepared_bridge_request["bridge_debug"] = deepcopy(bridge_debug)
                if hasattr(planner, "_set_last_bridge_debug"):
                    planner._set_last_bridge_debug(bridge_debug)
                return None
            session_state["phase_feedback"].append(
                _build_phase_feedback(
                    phase=current_phase,
                    decision=decision,
                    thought=thought,
                    detail={
                        "outline_tasks": deepcopy(outline_tasks),
                        "outline_validation": deepcopy(
                            turn_entry.get("outline_validation") or {}
                        ),
                        "validation_violations": deepcopy(
                            turn_entry.get("validation_violations") or []
                        ),
                        "outline_validation_findings": deepcopy(
                            turn_entry.get("outline_validation_findings") or []
                        ),
                        "addressed_validation_findings": deepcopy(
                            turn_entry.get("addressed_validation_findings") or []
                        ),
                        "outline_revision_coverage": deepcopy(
                            turn_entry.get("outline_revision_coverage") or {}
                        ),
                    },
                )
            )

        elif current_phase == "primitive_generation":
            macro_tasks = [
                deepcopy(row)
                for row in (parsed_response.get("macro_tasks") or [])
                if isinstance(row, dict)
            ]
            turn_entry["macro_tasks"] = deepcopy(macro_tasks)
            session_state["phase_feedback"].append(
                _build_phase_feedback(
                    phase=current_phase,
                    decision=decision,
                    thought=thought,
                    detail=macro_tasks,
                )
            )
            if decision == "draft_ready":
                session_state["proposal_draft"] = {
                    "thought": thought,
                    "macro_tasks": deepcopy(macro_tasks),
                }

        elif current_phase == "finalize":
            final_proposal = deepcopy(parsed_response.get("final_proposal") or {})
            turn_entry["final_proposal"] = deepcopy(final_proposal)
            session_state["phase_feedback"].append(
                _build_phase_feedback(
                    phase=current_phase,
                    decision=decision,
                    thought=thought,
                )
            )
            if decision == "final_ready":
                session_state["final_proposal"] = deepcopy(final_proposal)
                session_state["status"] = "final_proposal_recorded"
                _append_turn(planner, prepared_bridge_request, session_state, turn_entry)
                prepared_bridge_request["multi_turn_session_result"] = deepcopy(session_state)
                return deepcopy(final_proposal)

        session_state["current_phase"] = next_phase
        session_state["status"] = "running"
        _append_turn(planner, prepared_bridge_request, session_state, turn_entry)

    session_state["status"] = "turn_budget_exhausted"
    final_phase = str(session_state.get("current_phase") or "grounding")
    _logger.warning(
        "[MultiTurn] Session ended: turn_budget_exhausted (stuck in phase=%s — no proposal generated)",
        final_phase,
    )
    prepared_bridge_request["multi_turn_session_result"] = deepcopy(session_state)
    bridge_debug = deepcopy(prepared_bridge_request.get("bridge_debug") or {})
    bridge_debug["status"] = "turn_budget_exhausted"
    bridge_debug["multi_turn_session"] = deepcopy(session_state)
    prepared_bridge_request["bridge_debug"] = deepcopy(bridge_debug)
    if hasattr(planner, "_set_last_bridge_debug"):
        planner._set_last_bridge_debug(bridge_debug)
    return None
