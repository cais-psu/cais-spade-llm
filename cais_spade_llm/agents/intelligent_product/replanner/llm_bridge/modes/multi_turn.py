"""Multi-turn bridge execution engine — one-task-at-a-time outline."""

from __future__ import annotations

import inspect
import json
import logging
from copy import deepcopy
from typing import Any

from cais_spade_llm.agents.central_controller.outline_macro_safety import (
    validate_outline_macro_cca_constraints,
)
from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.bridge_artifacts import (
    write_bridge_artifacts,
)
from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.bridge_grounding_compiler import (
    compile_grounded_outline_task,
)
from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.bridge_proposal_normalization import (
    normalize_primitive_bridge_proposal,
)
from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.modes.multi_turn_outline_state import (
    _apply_outline_task_effects,
    _build_outline_task_type_lookup,
    _infer_outline_macro_signature,
    _outline_task_depends_on,
    _task_findings_block_projected_state,
)
from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.bridge_primitives import (
    extract_step_output,
    validate_and_project_steps_with_trace,
)
from cais_spade_llm.resources.resource_primitives import (
    filter_synthesis_primitive_catalog,
)
from cais_spade_llm.resources.resource_profile import (
    get_resource_profile,
    get_resource_profile_for_agent,
)

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_MAX_TURNS = 1000  # Practical ceiling for long LLM-guided outline retries.
_DEFAULT_MAX_OBSERVATIONS = 3
_DEFAULT_MAX_OBSERVE_BATCH = 3
_DEFAULT_CANDIDATE_BOUND = 5
_DEFAULT_CANDIDATE_BOUND_CAP = 8
_CANDIDATE_PRUNE_REPEAT_THRESHOLD = 2
_MULTI_TURN_OUTLINE_CONTRACT = {
    "allowed_state_fields": [
        "resource_state",
        "held_part",
        "part_state",
        "part_location",
        "part_holder_resource_jid",
    ],
    "disallow_unknown_state_fields": True,
    "require_expected_start_match": True,
    "require_meaningful_delta": True,
    "require_release_destination_for_release": True,
    "require_carrier_for_part_relocation": True,
}

_PHASE_SEQUENCE = ("grounding", "outline", "primitive_generation", "finalize")

_DURABLE_PRUNED_CONSTRAINT_CODES = {
    "workspace_unreachable",
    "holder_conflict",
    "required_part_not_held",
    "source_reference_unavailable",
    "part_relocation_without_carrier",
    "safety_rule_violation",
    "blocker_open",
    "dependency_unsatisfied",
    "order_violation",
}

_TRANSITIONS: dict[str, dict[str, str]] = {
    "grounding": {
        "observe": "grounding",
        "grounded": "outline",
    },
    "outline": {
        "outline_ready": "primitive_generation",
        "need_revision": "outline",
        "need_next_task": "outline",
    },
    "primitive_generation": {
        "need_context": "primitive_generation",
        "primitive_steps_ready": "primitive_generation",
        "need_primitive_revision": "primitive_generation",
        "primitive_event_stuck": "primitive_generation",
        "primitive_blocked": "primitive_generation",
        "draft_ready": "finalize",
    },
    "finalize": {
        "accepted": "finalize",
        "need_outline_revision": "outline",
        "need_primitive_revision": "primitive_generation",
    },
}

# ---------------------------------------------------------------------------
# Phase transitions
# ---------------------------------------------------------------------------


def transition_multi_turn_phase(current_phase: str, decision: str) -> str:
    """Return the next phase given the current phase and a decision token."""
    phase = current_phase.strip().lower()
    token = decision.strip().lower()
    phase_map = _TRANSITIONS.get(phase, {})
    next_phase = phase_map.get(token)
    if next_phase is None:
        _logger.warning(
            "[MultiTurn] No transition for phase=%s decision=%s; staying in %s",
            phase, token, phase,
        )
        return phase
    return next_phase


# ---------------------------------------------------------------------------
# Session seed
# ---------------------------------------------------------------------------


def build_multi_turn_session_seed(
    prepared_bridge_request: dict[str, Any],
) -> dict[str, Any]:
    """Build the initial session state for a multi-turn bridge run."""
    bridge_session = dict(prepared_bridge_request.get("bridge_session") or {})
    max_turns = int(bridge_session.get("max_turns") or _DEFAULT_MAX_TURNS)
    max_observations = int(
        bridge_session.get("max_observations") or _DEFAULT_MAX_OBSERVATIONS
    )
    max_observe_batch = max(
        1, int(bridge_session.get("max_observe_batch") or _DEFAULT_MAX_OBSERVE_BATCH)
    )
    outline_mode = str(
        bridge_session.get("outline_mode") or "incremental"
    ).strip().lower()
    if outline_mode not in (
        "single_pass",
        "incremental",
        "incremental_validated",
        "incremental_candidates_validated",
    ):
        outline_mode = "incremental"
    feedback_render_style = str(
        bridge_session.get("feedback_render_style") or "des_event_diagnostic"
    ).strip().lower()
    if feedback_render_style not in {"des_event_diagnostic", "raw_code"}:
        feedback_render_style = "des_event_diagnostic"
    candidate_bound_cap = max(
        1,
        int(
            bridge_session.get("candidate_bound_cap")
            or _DEFAULT_CANDIDATE_BOUND_CAP
        ),
    )
    candidate_bound = max(
        1,
        int(
            bridge_session.get("candidate_bound")
            or _DEFAULT_CANDIDATE_BOUND
        ),
    )
    # Build symbolic resource/part state for validation tracking
    llm_input = dict(prepared_bridge_request.get("llm_input") or {})
    observed_runtime_state = dict(llm_input.get("observed_runtime_state") or {})
    symbolic_resources: dict[str, dict[str, Any]] = {}
    for row in (observed_runtime_state.get("resources") or []):
        if not isinstance(row, dict):
            continue
        jid = str(row.get("resource_jid") or "").strip()
        if jid:
            symbolic_resources[jid] = deepcopy(row)
    symbolic_parts: dict[str, dict[str, Any]] = {}
    for row in (llm_input.get("part_facts") or []):
        if not isinstance(row, dict):
            continue
        name = str(row.get("part_name") or "").strip()
        if name:
            symbolic_parts[name] = deepcopy(row)

    return {
        "current_phase": "grounding",
        "turn_index": 0,
        "max_turns": max_turns,
        "max_observations": max_observations,
        "max_observe_batch": max_observe_batch,
        "outline_mode": outline_mode,
        "feedback_render_style": feedback_render_style,
        "candidate_bound": candidate_bound,
        "candidate_bound_cap": candidate_bound_cap,
        "status": "pending",
        "turns": [],
        # Grounding state
        "observation_count": 0,
        "observation_store": {},
        "observation_history": [],
        "observation_fact_ledger": {},
        "phase_feedback": [],
        # Outline state
        "accepted_outline_prefix": [],
        "accepted_transition_prefix": [],
        "des_event_sequence": [],
        "transition_trace": [],
        "outline_lookahead": [],
        "outline_stagnation_count": 0,
        "outline_progress_signature": "",
        "pruned_actions": [],
        "outline_validation_findings": [],
        "transition_validation": {},
        "unresolved_target_predicates": [],
        "candidate_rejection_feedback": [],
        "candidate_prune_history": {},
        # Primitive-generation state
        "primitive_generation_cursor": 0,
        "accepted_primitive_program": [],
        "primitive_rejection_feedback": [],
        "primitive_served_context": {},
        "primitive_context_errors": [],
        "primitive_input_diagnostics": [],
        "primitive_event_guard": {},
        "primitive_escalation_diagnostics": [],
        # Symbolic state for validation
        "symbolic_resources": symbolic_resources,
        "symbolic_parts": symbolic_parts,
    }


# ---------------------------------------------------------------------------
# Bridge trace helpers
# ---------------------------------------------------------------------------


def _sync_des_recovery_aliases(
    session_state: dict[str, Any],
    *,
    turn_entry: dict[str, Any] | None = None,
    transition_validation: dict[str, Any] | None = None,
    unresolved_target_predicates: list[dict[str, Any]] | None = None,
) -> None:
    """Maintain DES-style debug aliases without changing parser-facing fields."""
    accepted_prefix = [
        deepcopy(row)
        for row in (session_state.get("accepted_outline_prefix") or [])
        if isinstance(row, dict)
    ]
    session_state["accepted_transition_prefix"] = deepcopy(accepted_prefix)
    session_state["des_event_sequence"] = deepcopy(accepted_prefix)
    session_state["transition_trace"] = deepcopy(accepted_prefix)

    if transition_validation is not None:
        session_state["transition_validation"] = deepcopy(transition_validation)
    if unresolved_target_predicates is not None:
        session_state["unresolved_target_predicates"] = deepcopy(
            unresolved_target_predicates
        )

    if turn_entry is None:
        return
    turn_entry["accepted_transition_prefix"] = deepcopy(accepted_prefix)
    turn_entry["des_event_sequence"] = deepcopy(accepted_prefix)
    turn_entry["transition_trace"] = deepcopy(accepted_prefix)
    if transition_validation is not None:
        turn_entry["transition_validation"] = deepcopy(transition_validation)
    if unresolved_target_predicates is not None:
        turn_entry["unresolved_target_predicates"] = deepcopy(
            unresolved_target_predicates
        )


# ---------------------------------------------------------------------------
# Observation helpers
# ---------------------------------------------------------------------------


def _resource_agent_map(planner: Any) -> dict[str, Any]:
    return {
        str(getattr(agent, "jid", "")).strip(): agent
        for agent in (getattr(planner, "resource_agents", None) or [])
        if str(getattr(agent, "jid", "")).strip()
    }


def _focused_observation_resource_jid(prepared_bridge_request: dict[str, Any]) -> str:
    llm_input = dict(prepared_bridge_request.get("llm_input") or {})
    allowed_surface = dict(llm_input.get("allowed_execution_surface") or {})
    focused_jid = str(
        allowed_surface.get("focused_resource_jid")
        or prepared_bridge_request.get("ra_jid")
        or ""
    ).strip()
    bridge_resources = dict(prepared_bridge_request.get("bridge_resources") or {})
    if focused_jid and focused_jid in bridge_resources:
        return focused_jid
    for row in (allowed_surface.get("resources") or []):
        if not isinstance(row, dict):
            continue
        if str(row.get("role") or "").strip().lower() == "focused":
            jid = str(row.get("resource_jid") or "").strip()
            if jid:
                return jid
    for jid in bridge_resources:
        token = str(jid or "").strip()
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


def _is_grounding_observation_primitive(*, resource_type: str, primitive_name: str) -> bool:
    profile = get_resource_profile(resource_type or "resource")
    allowlist = {
        str(item).strip()
        for item in (profile.grounding_observation_primitives or ())
        if str(item).strip()
    }
    return str(primitive_name or "").strip() in allowlist if allowlist else True


def _supports_observation_output(*, resource_type: str, primitive_name: str) -> bool:
    profile = get_resource_profile(resource_type or "resource")
    token = str(primitive_name or "").strip()
    return (
        token in dict(profile.preview_output_map or {})
        or token in dict(profile.extract_output_map or {})
    )


def _lookup_world_observation_primitive(
    prepared_bridge_request: dict[str, Any],
    *,
    primitive_name: str,
) -> tuple[str, dict[str, Any], str]:
    resource_jid, bridge_entry, resource_type = _focused_observation_resource_entry(
        prepared_bridge_request
    )
    for entry in (bridge_entry.get("primitive_catalog") or []):
        if not isinstance(entry, dict):
            continue
        if str(entry.get("name") or "").strip() != primitive_name:
            continue
        if str(entry.get("primitive_kind") or "").strip().lower() != "observe":
            break
        if not _is_grounding_observation_primitive(
            resource_type=resource_type, primitive_name=primitive_name,
        ):
            break
        if not _supports_observation_output(
            resource_type=resource_type, primitive_name=primitive_name,
        ):
            break
        return resource_jid, entry, resource_type
    return resource_jid, {}, resource_type


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
            prepared_bridge_request, primitive_name=primitive_name,
        )
        if not primitive_entry:
            continue
        contracts[fact_type] = {
            "fact_type": fact_type,
            "entity_kind": str(raw_contract.get("entity_kind") or "").strip() or None,
            "primitive": primitive_name,
            "entity_param": entity_param,
        }
    return contracts


def _known_part_names(prepared_bridge_request: dict[str, Any]) -> set[str]:
    llm_input = dict(prepared_bridge_request.get("llm_input") or {})
    return {
        str(row.get("part_name") or "").strip()
        for row in (llm_input.get("part_facts") or [])
        if isinstance(row, dict) and str(row.get("part_name") or "").strip()
    }


def _observation_request_key(primitive_name: str, params: dict[str, Any]) -> str:
    return json.dumps(
        {"primitive": str(primitive_name or "").strip(), "params": deepcopy(params or {})},
        sort_keys=True, default=str, ensure_ascii=True,
    )


def _observation_fact_key(fact_type: str, entity: str, scope: Any = None) -> str:
    payload: dict[str, Any] = {
        "fact_type": str(fact_type or "").strip(),
        "entity": str(entity or "").strip(),
    }
    if scope not in (None, "", [], {}):
        payload["scope"] = deepcopy(scope)
    return json.dumps(payload, sort_keys=True, default=str, ensure_ascii=True)


def _sanitize_observation_key_token(value: Any) -> str:
    raw_token = str(value or "").strip()
    if not raw_token:
        return ""
    sanitized_chars: list[str] = []
    for char in raw_token:
        sanitized_chars.append(char if char.isalnum() else "_")
    sanitized = "".join(sanitized_chars)
    while "__" in sanitized:
        sanitized = sanitized.replace("__", "_")
    return sanitized.strip("_")


def _default_observation_key(
    request: dict[str, Any],
    *,
    seen_keys: set[str],
) -> str:
    fact_type = str(request.get("fact_type") or "").strip()
    entity = str(request.get("entity") or "").strip()
    primitive_name = str(request.get("primitive") or "").strip()
    resource_jid = str(request.get("resource_jid") or "").strip()
    params = dict(request.get("params") or {})

    if fact_type == "part_pose" and entity:
        prefix = "observed_pose"
        entity_token = _sanitize_observation_key_token(entity) or entity
        base = f"{prefix}_{entity_token}"
    else:
        key_parts: list[str] = []
        prefix = (
            _sanitize_observation_key_token(fact_type)
            or _sanitize_observation_key_token(primitive_name)
            or "observation"
        )
        key_parts.append(prefix)
        if entity:
            key_parts.append(_sanitize_observation_key_token(entity) or entity)
        elif resource_jid:
            resource_token = _sanitize_observation_key_token(resource_jid)
            if resource_token:
                key_parts.append(resource_token)
        if len(key_parts) == 1 and params:
            for param_name in sorted(params.keys()):
                param_token = _sanitize_observation_key_token(param_name)
                if param_token:
                    key_parts.append(param_token)
                value = params.get(param_name)
                if isinstance(value, (str, int, float, bool)):
                    value_token = _sanitize_observation_key_token(value)
                    if value_token:
                        key_parts.append(value_token)
                if len(key_parts) >= 4:
                    break
        base = "_".join(part for part in key_parts if part) or "observation"

    observation_key = base
    counter = 2
    while observation_key in seen_keys:
        observation_key = f"{base}_{counter}"
        counter += 1
    return observation_key


def _resolve_observation_request(
    prepared_bridge_request: dict[str, Any],
    raw_request: dict[str, Any],
) -> tuple[dict[str, Any] | None, str | None]:
    """Resolve a raw observe request into a normalized form."""
    request = dict(raw_request or {})
    fact_type = str(request.get("fact_type") or "").strip()
    entity = str(request.get("entity") or "").strip()
    reason = str(request.get("reason") or "").strip()
    scope = deepcopy(request.get("scope"))
    params = dict(request.get("params") or {})

    if fact_type:
        contract = dict(
            (_world_observation_fact_contracts(prepared_bridge_request) or {}).get(fact_type) or {}
        )
        if not contract:
            return None, f"world observation fact {fact_type!r} is not allowed"
        primitive_name = str(contract.get("primitive") or "").strip()
        entity_param = str(contract.get("entity_param") or "").strip()
        if not entity:
            return None, f"observe request fact {fact_type!r} requires entity"
        if not primitive_name or not entity_param:
            return None, f"world observation fact {fact_type!r} is not executable"
        entity_kind = str(contract.get("entity_kind") or "").strip().lower()
        if entity_kind == "part":
            known_parts = _known_part_names(prepared_bridge_request)
            if known_parts and entity not in known_parts:
                return None, (
                    f"observe request fact {fact_type!r} has unknown part entity {entity!r}; "
                    f"known parts are {sorted(known_parts)!r}"
                )
        params[entity_param] = entity
        resource_jid, primitive_entry, resource_type = _lookup_world_observation_primitive(
            prepared_bridge_request, primitive_name=primitive_name,
        )
        if not primitive_entry:
            return None, f"world observation fact {fact_type!r} is not executable"
        return {
            "fact_type": fact_type,
            "entity": entity,
            "entity_kind": str(contract.get("entity_kind") or "").strip() or None,
            "scope": scope,
            "reason": reason,
            "primitive": primitive_name,
            "params": deepcopy(params),
            "resource_jid": resource_jid,
            "resource_type": resource_type,
            "fact_key": _observation_fact_key(fact_type, entity, scope),
        }, None

    # Fallback: raw primitive request
    primitive_name = str(request.get("primitive") or "").strip()
    if not primitive_name:
        return None, "observe_requests must include fact_type and entity"
    resource_jid, primitive_entry, resource_type = _lookup_world_observation_primitive(
        prepared_bridge_request, primitive_name=primitive_name,
    )
    if not primitive_entry:
        return None, f"world observation primitive {primitive_name!r} is not allowed"
    return {
        "fact_type": "",
        "entity": "",
        "entity_kind": None,
        "scope": scope,
        "reason": reason,
        "primitive": primitive_name,
        "params": deepcopy(params),
        "resource_jid": resource_jid,
        "resource_type": resource_type,
        "fact_key": "",
    }, None


async def _invoke_primitive(
    resource_agent: Any, primitive_name: str, params: dict[str, Any],
) -> Any:
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


def _record_observation_fact(
    session_state: dict[str, Any],
    observation_row: dict[str, Any],
) -> None:
    fact_key = str(observation_row.get("fact_key") or "").strip()
    if not fact_key:
        return
    ledger = dict(session_state.get("observation_fact_ledger") or {})
    existing = dict(ledger.get(fact_key) or {})
    observation_key = str(observation_row.get("observation_key") or "").strip()
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
        "observation_key": observation_key
        or str(existing.get("observation_key") or "").strip()
        or None,
    }
    session_state["observation_fact_ledger"] = ledger


def _observation_pose(payload: dict[str, Any]) -> dict[str, Any] | None:
    pose = payload.get("pose")
    if isinstance(pose, dict) and any(
        pose.get(axis) is not None for axis in ("x", "y", "z")
    ):
        return deepcopy(pose)
    if any(payload.get(axis) is not None for axis in ("x", "y", "z")):
        return {
            axis: deepcopy(payload[axis])
            for axis in ("x", "y", "z", "qx", "qy", "qz", "qw")
            if payload.get(axis) is not None
        }
    return None


def _seed_grounded_part_pose_observations(
    prepared_bridge_request: dict[str, Any],
    session_state: dict[str, Any],
) -> None:
    """Populate observation store with known part poses from part_tracker."""
    part_tracker = {
        str(name or "").strip(): dict(entry or {})
        for name, entry in dict(prepared_bridge_request.get("part_tracker") or {}).items()
        if str(name or "").strip() and isinstance(entry, dict)
    }
    if not part_tracker:
        return

    llm_part_facts = {
        str(dict(row).get("part_name") or "").strip(): dict(row)
        for row in (dict(prepared_bridge_request.get("llm_input") or {}).get("part_facts") or [])
        if isinstance(row, dict) and str(dict(row).get("part_name") or "").strip()
    }
    observation_store = dict(session_state.get("observation_store") or {})
    seen_observation_keys = set(observation_store)

    for part_name, tracker_entry in part_tracker.items():
        pose = _observation_pose(
            {"pose": deepcopy(tracker_entry.get("observed_pose"))}
        ) or _observation_pose(tracker_entry)
        if pose is None:
            continue
        observation_key = _default_observation_key(
            {
                "fact_type": "part_pose",
                "entity": part_name,
                "primitive": "grounding_context",
            },
            seen_keys=seen_observation_keys,
        )
        part_row = dict(llm_part_facts.get(part_name) or {})
        payload: dict[str, Any] = {"part_name": part_name, "pose": deepcopy(pose)}
        for axis in ("x", "y", "z", "qx", "qy", "qz", "qw"):
            value = pose.get(axis)
            if value is not None:
                payload[axis] = deepcopy(value)
        current_location = part_row.get("current_location")
        if current_location not in (None, ""):
            payload["current_location"] = deepcopy(current_location)
        holder_jid = str(part_row.get("current_holder_resource_jid") or "").strip()
        if holder_jid:
            payload["current_holder_resource_jid"] = holder_jid

        observation_store[observation_key] = deepcopy(payload)
        seen_observation_keys.add(observation_key)
        _record_observation_fact(session_state, {
            "fact_key": _observation_fact_key("part_pose", part_name, None),
            "fact_type": "part_pose",
            "entity": part_name,
            "entity_kind": "part",
            "scope": None,
            "primitive": "grounding_context",
            "params": {"part_name": part_name},
            "observation_key": observation_key,
            "output": deepcopy(payload),
            "turn_index": int(session_state.get("turn_index") or 0),
        })

    session_state["observation_store"] = observation_store


def _build_world_observation_surface(
    prepared_bridge_request: dict[str, Any],
) -> dict[str, Any]:
    """Build the observation surface shown to the LLM during grounding."""
    _, bridge_entry, resource_type = _focused_observation_resource_entry(prepared_bridge_request)
    observation_facts: list[dict[str, Any]] = []
    for fact_type, contract in _world_observation_fact_contracts(prepared_bridge_request).items():
        row: dict[str, Any] = {
            "fact_type": fact_type,
            "request_fields": ["fact_type", "entity"],
        }
        entity_kind = str(contract.get("entity_kind") or "").strip()
        if entity_kind:
            row["entity_kind"] = entity_kind
        observation_facts.append(row)
    if observation_facts:
        return {"observation_facts": observation_facts}

    # Fallback: list raw observe primitives
    observation_primitives: list[dict[str, Any]] = []
    for entry in (bridge_entry.get("primitive_catalog") or []):
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name") or "").strip()
        kind = str(entry.get("primitive_kind") or "").strip().lower()
        if kind != "observe" or not name:
            continue
        if not _is_grounding_observation_primitive(
            resource_type=resource_type, primitive_name=name,
        ):
            continue
        if not _supports_observation_output(resource_type=resource_type, primitive_name=name):
            continue
        observation_primitives.append({
            "name": name,
            "primitive_kind": kind,
            "required_params": [
                str(p).strip() for p in (entry.get("required_params") or []) if str(p).strip()
            ],
        })
    return {"observation_primitives": observation_primitives}


async def _execute_observe_requests(
    planner: Any,
    prepared_bridge_request: dict[str, Any],
    session_state: dict[str, Any],
    observe_requests: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], str | None]:
    """Execute observation requests and return (results, error)."""
    resource_agents = _resource_agent_map(planner)
    results: list[dict[str, Any]] = []
    seen_observation_keys = set(dict(session_state.get("observation_store") or {}))
    seen_fact_rows: dict[str, dict[str, Any]] = {
        str(k).strip(): dict(v)
        for k, v in dict(session_state.get("observation_fact_ledger") or {}).items()
        if str(k).strip() and isinstance(v, dict)
    }
    seen_request_rows: dict[str, dict[str, Any]] = {
        _observation_request_key(
            str(row.get("primitive") or "").strip(),
            dict(row.get("params") or {}),
        ): dict(row)
        for row in (session_state.get("observation_history") or [])
        if isinstance(row, dict)
    }

    for raw_request in observe_requests:
        normalized, resolve_error = _resolve_observation_request(
            prepared_bridge_request, dict(raw_request or {}),
        )
        if resolve_error is not None or not isinstance(normalized, dict):
            return [], resolve_error or "failed to resolve observation request"

        primitive_name = str(normalized.get("primitive") or "").strip()
        params = dict(normalized.get("params") or {})
        fact_key = str(normalized.get("fact_key") or "").strip()
        request_key = _observation_request_key(primitive_name, params)

        if not primitive_name:
            return [], "observe_requests must include fact_type and entity"

        # Check for duplicate observation
        prior_fact = dict(seen_fact_rows.get(fact_key) or {}) if fact_key else {}
        if prior_fact and str(prior_fact.get("validity") or "current").strip().lower() != "stale":
            return [], f"observation fact {fact_key!r} already exists and is current"
        if not prior_fact and request_key in seen_request_rows:
            return [], (
                f"observation {primitive_name!r} with these params already succeeded earlier"
            )

        observation_key = _default_observation_key(
            normalized,
            seen_keys=seen_observation_keys,
        )
        normalized["observation_key"] = observation_key

        # Validate primitive exists
        resource_jid = str(normalized.get("resource_jid") or "").strip()
        _, primitive_entry, _ = _lookup_world_observation_primitive(
            prepared_bridge_request, primitive_name=primitive_name,
        )
        if not primitive_entry:
            return [], f"world observation primitive {primitive_name!r} is not allowed"

        # Check required params
        for req_param in (primitive_entry.get("required_params") or []):
            req_param = str(req_param).strip()
            if req_param and (req_param not in params or params.get(req_param) is None):
                return [], (
                    f"observe request {primitive_name!r} on {resource_jid!r} "
                    f"is missing required param {req_param!r}"
                )

        # Execute
        resource_agent = resource_agents.get(resource_jid)
        if resource_agent is None:
            return [], f"resource agent {resource_jid!r} is not available for observation"

        raw_result = await _invoke_primitive(resource_agent, primitive_name, params)
        normalized_result = (
            deepcopy(raw_result) if isinstance(raw_result, dict)
            else {"success": True, "data": deepcopy(raw_result)}
        )
        resource_type = str(normalized.get("resource_type") or "resource").strip()
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
            "fact_type": str(normalized.get("fact_type") or "").strip(),
            "entity": str(normalized.get("entity") or "").strip(),
            "entity_kind": deepcopy(normalized.get("entity_kind")),
            "scope": deepcopy(normalized.get("scope")),
            "reason": str(normalized.get("reason") or "").strip(),
            "primitive": primitive_name,
            "params": deepcopy(params),
            "observation_key": observation_key,
            "output": deepcopy(extracted_output),
        }
        results.append(observation_row)
        seen_observation_keys.add(observation_key)
        seen_request_rows[request_key] = deepcopy(observation_row)
        if fact_key:
            seen_fact_rows[fact_key] = {
                **deepcopy(observation_row),
                "turn_index": int(session_state.get("turn_index") or 0),
                "validity": "current",
                "freshness": "current_session",
                "observation_key": observation_key,
            }

    return results, None


# ---------------------------------------------------------------------------
# Phase handlers
# ---------------------------------------------------------------------------


async def _handle_grounding_phase(
    *,
    session_state: dict[str, Any],
    parsed_response: dict[str, Any],
    prepared_bridge_request: dict[str, Any],
    planner: Any,
) -> tuple[str, dict[str, Any]]:
    """Handle one turn of the grounding (observe) phase.

    The LLM decides either:
    - "observe": request observations, execute them, loop back
    - "grounded": grounding is complete, transition to outline
    """
    decision = str(parsed_response.get("decision") or "").strip().lower()
    turn_entry: dict[str, Any] = {}

    # Normalize decision
    if decision not in ("observe", "grounded"):
        # Default: if observe_requests present, treat as observe; otherwise grounded
        if parsed_response.get("observe_requests"):
            decision = "observe"
        else:
            decision = "grounded"

    observe_requests = [
        dict(row) for row in (parsed_response.get("observe_requests") or [])
        if isinstance(row, dict)
    ]
    turn_entry["observe_requests"] = deepcopy(observe_requests)

    if decision == "observe":
        # Validate: must have requests
        if not observe_requests:
            turn_entry["error"] = "observe decision requires observe_requests"
            session_state["phase_feedback"].append({
                "phase": "grounding", "issue": "observe_without_requests",
            })
            return decision, turn_entry

        observation_fact_ledger = {
            str(fact_key).strip(): dict(row)
            for fact_key, row in dict(session_state.get("observation_fact_ledger") or {}).items()
            if str(fact_key).strip() and isinstance(row, dict)
        }
        unresolved_observe_requests: list[dict[str, Any]] = []
        already_fulfilled_observe_requests: list[dict[str, Any]] = []
        invalid_observe_requests: list[dict[str, Any]] = []
        for raw_request in observe_requests:
            request = dict(raw_request or {})
            normalized_request, resolve_error = _resolve_observation_request(
                prepared_bridge_request,
                request,
            )
            if resolve_error is not None or not isinstance(normalized_request, dict):
                # Track invalid requests instead of aborting the loop
                invalid_observe_requests.append({
                    "fact_type": str(request.get("fact_type") or "").strip(),
                    "entity": str(request.get("entity") or "").strip(),
                    "error": resolve_error or "failed to resolve observation request",
                })
                continue
            fact_key = str(normalized_request.get("fact_key") or "").strip()
            prior_fact = dict(observation_fact_ledger.get(fact_key) or {}) if fact_key else {}
            is_fulfilled = bool(
                prior_fact
                and str(prior_fact.get("validity") or "current").strip().lower() != "stale"
            )
            if is_fulfilled:
                already_fulfilled_observe_requests.append(
                    {
                        "fact_type": str(request.get("fact_type") or "").strip(),
                        "entity": str(request.get("entity") or "").strip(),
                        "fact_key": fact_key,
                    }
                )
            else:
                unresolved_observe_requests.append(deepcopy(request))
        if invalid_observe_requests:
            turn_entry["invalid_observe_requests"] = deepcopy(invalid_observe_requests)
        if already_fulfilled_observe_requests:
            turn_entry["already_fulfilled_observe_requests"] = deepcopy(
                already_fulfilled_observe_requests
            )
        # If every request was either already fulfilled or invalid, skip to grounded
        if not unresolved_observe_requests:
            decision = "grounded"
            reasons = []
            if already_fulfilled_observe_requests:
                reasons.append("already_fulfilled")
            if invalid_observe_requests:
                reasons.append("invalid_fact_types")
            turn_entry["grounding_override_reason"] = (
                "all_observe_requests_resolved: " + "+".join(reasons)
            )
            _seed_grounded_part_pose_observations(prepared_bridge_request, session_state)
            _logger.info(
                "[MultiTurn] Grounding override → grounded (%s)",
                turn_entry["grounding_override_reason"],
            )
            return decision, turn_entry

        # Check batch limit
        max_batch = int(session_state.get("max_observe_batch") or _DEFAULT_MAX_OBSERVE_BATCH)
        if len(unresolved_observe_requests) > max_batch:
            turn_entry["error"] = f"observe_requests count exceeds max_observe_batch={max_batch}"
            return decision, turn_entry

        # Check observation budget
        remaining = max(
            0,
            int(session_state.get("max_observations") or 0)
            - int(session_state.get("observation_count") or 0),
        )
        if len(unresolved_observe_requests) > remaining:
            turn_entry["error"] = "observation budget exhausted"
            return decision, turn_entry

        # Execute observations
        observation_results, observe_error = await _execute_observe_requests(
            planner, prepared_bridge_request, session_state, unresolved_observe_requests,
        )

        if observe_error is not None:
            _logger.warning("[MultiTurn] observe failed: %s", observe_error)
            turn_entry["error"] = observe_error
            session_state["phase_feedback"].append({
                "phase": "grounding", "issue": "observe_error", "detail": observe_error,
            })
            return decision, turn_entry

        # Store results
        for result in observation_results:
            observation_key = str(result.get("observation_key") or "").strip()
            if not observation_key:
                turn_entry["error"] = "observation result missing observation_key"
                session_state["phase_feedback"].append({
                    "phase": "grounding",
                    "issue": "observe_error",
                    "detail": "observation result missing observation_key",
                })
                return decision, turn_entry
            session_state["observation_store"][observation_key] = deepcopy(
                result.get("output") or {}
            )
            observation_event = {
                "turn_index": int(session_state.get("turn_index") or 0),
                "phase": "grounding",
                **deepcopy(result),
            }
            session_state["observation_history"].append(observation_event)
            _record_observation_fact(session_state, observation_event)

        session_state["observation_count"] = (
            int(session_state.get("observation_count") or 0) + len(observation_results)
        )
        turn_entry["observation_results"] = deepcopy(observation_results)
        _logger.info(
            "[MultiTurn] Observed %d facts", len(observation_results),
        )

    else:
        # decision == "grounded"
        _seed_grounded_part_pose_observations(prepared_bridge_request, session_state)
        _logger.info("[MultiTurn] Grounding complete")

    return decision, turn_entry


def _projected_outline_validation_context(
    *,
    session_state: dict[str, Any],
    prepared_bridge_request: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    llm_input = dict(prepared_bridge_request.get("llm_input") or {})
    observed_runtime_state = dict(llm_input.get("observed_runtime_state") or {})

    resources_by_jid: dict[str, dict[str, Any]] = {}
    for row in (observed_runtime_state.get("resources") or []):
        if not isinstance(row, dict):
            continue
        resource_jid = str(row.get("resource_jid") or "").strip()
        if resource_jid:
            resources_by_jid[resource_jid] = deepcopy(row)
    for resource_jid, row in dict(session_state.get("symbolic_resources") or {}).items():
        token = str(resource_jid or "").strip()
        if token and isinstance(row, dict):
            resources_by_jid[token] = deepcopy(row)
    bridge_resources = dict(prepared_bridge_request.get("bridge_resources") or {})
    for resource_jid, raw_entry in bridge_resources.items():
        token = str(resource_jid or "").strip()
        if not token or not isinstance(raw_entry, dict):
            continue
        entry = dict(raw_entry)
        bridge_snapshot = dict(entry.get("bridge_snapshot") or {})
        static_capabilities = dict(entry.get("static_capabilities") or {})
        resource_row = resources_by_jid.setdefault(token, {"resource_jid": token})
        for key in (
            "named_poses",
            "available_named_poses",
            "supported_recovery_states",
            "available_recovery_states",
            "reachability",
            "reachable_locations",
            "known_locations",
            "staging_areas",
            "workspace_bounds",
        ):
            if resource_row.get(key) not in (None, "", [], {}):
                continue
            if static_capabilities.get(key) not in (None, "", [], {}):
                resource_row[key] = deepcopy(static_capabilities.get(key))
            elif bridge_snapshot.get(key) not in (None, "", [], {}):
                resource_row[key] = deepcopy(bridge_snapshot.get(key))

    parts_by_name: dict[str, dict[str, Any]] = {}
    for row in (llm_input.get("part_facts") or []):
        if not isinstance(row, dict):
            continue
        part_name = str(row.get("part_name") or "").strip()
        if part_name:
            parts_by_name[part_name] = deepcopy(row)
    for part_name, row in dict(session_state.get("symbolic_parts") or {}).items():
        token = str(part_name or "").strip()
        if token and isinstance(row, dict):
            parts_by_name[token] = deepcopy(row)

    for entry in dict(session_state.get("observation_store") or {}).values():
        if not isinstance(entry, dict):
            continue
        part_name = str(entry.get("part_name") or "").strip()
        if not part_name:
            continue
        is_new_part = part_name not in parts_by_name
        part_row = parts_by_name.setdefault(part_name, {"part_name": part_name})
        pose = dict(entry.get("pose") or {})
        if not pose and entry.get("x") is not None:
            pose = {"x": entry.get("x"), "y": entry.get("y"), "z": entry.get("z")}
        if pose:
            part_row["observed_pose"] = deepcopy(pose)
        if (
            is_new_part
            and part_row.get("current_location") in (None, "")
            and entry.get("current_location") not in (None, "")
        ):
            part_row["current_location"] = deepcopy(entry.get("current_location"))
        holder = str(entry.get("current_holder_resource_jid") or "").strip()
        if (
            is_new_part
            and not str(part_row.get("current_holder_resource_jid") or "").strip()
            and holder
        ):
            part_row["current_holder_resource_jid"] = holder
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
    evidence = dict(evidence or {})
    return {
        "task_id": str(task.get("outline_id") or "").strip(),
        "resource_jid": resource_jid or None,
        "part_name": part_name or None,
        "pose_source": "resource_feasibility",
        "pose": deepcopy(evidence.get("checked_pose")),
        "workspace_bounds": deepcopy(evidence.get("workspace_bounds")),
        "failed_axes": [constraint_code],
        "constraint_owner": "resource",
        "constraint_family": "resource_feasibility",
        "constraint_code": constraint_code,
        "reason": reason,
        "guard": deepcopy(guard),
        "evidence": deepcopy(evidence),
    }


def _resource_part_context(
    *,
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


def _validate_outline_task_ra(
    *,
    planner: Any,
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
                reason=(
                    f"resource '{resource_jid or 'unknown'}' is not available in current bridge state"
                ),
                resource_jid=resource_jid,
                part_name=part_name,
                guard={"kind": "resource_not_available", "resource_jid": resource_jid}
                if resource_jid
                else None,
            )
        ]

    resource_agent = _resource_agent_map(planner).get(resource_jid)
    oracle = getattr(resource_agent, "bridge_feasibility_oracle", None)
    if not callable(oracle):
        return []

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

    constraint_code = (
        str(result.get("constraint_code") or "").strip() or "resource_unavailable"
    )
    reason = (
        str(result.get("reason") or "").strip()
        or "resource feasibility rejected the grounded action"
    )
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


def _validate_outline_task_cca(
    *,
    task: dict[str, Any],
    grounded_action: dict[str, Any],
    resources_by_jid: dict[str, dict[str, Any]],
    parts_by_name: dict[str, dict[str, Any]],
    llm_input: dict[str, Any],
    prior_findings: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    validation_trace = [deepcopy(task)]
    task_id = str(task.get("outline_id") or "").strip() or "task_0"
    task_types_by_id = _build_outline_task_type_lookup(
        validation_trace,
        resources_by_jid=resources_by_jid,
        parts_by_name=parts_by_name,
        llm_input=llm_input,
    )
    signature = _infer_outline_macro_signature(
        task,
        resources_by_jid=resources_by_jid,
        parts_by_name=parts_by_name,
    )

    projected_resources = deepcopy(resources_by_jid)
    projected_parts = deepcopy(parts_by_name)
    if not _task_findings_block_projected_state(list(prior_findings or [])):
        task_type = str(task_types_by_id.get(task_id) or "").strip()
        _apply_outline_task_effects(
            task,
            resources_by_jid=projected_resources,
            parts_by_name=projected_parts,
            task_type=task_type,
            grounded_action=grounded_action,
        )

    cca_result = validate_outline_macro_cca_constraints(
        task=deepcopy(task),
        grounded_action=deepcopy(grounded_action),
        signature=deepcopy(signature),
        pre_resources=deepcopy(resources_by_jid),
        pre_parts=deepcopy(parts_by_name),
        projected_resources=projected_resources,
        projected_parts=projected_parts,
        llm_input=deepcopy(llm_input),
        outline_tasks=validation_trace,
        task_types_by_id=deepcopy(task_types_by_id),
        task_index_by_id={task_id: 0},
        dependency_map={task_id: _outline_task_depends_on(task)},
        previously_cleared_condition_ids=None,
    )
    return [
        deepcopy(row)
        for row in (cca_result.get("findings") or [])
        if isinstance(row, dict)
    ]


def _validate_single_outline_task(
    *,
    planner: Any,
    task: dict[str, Any],
    session_state: dict[str, Any],
    prepared_bridge_request: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    resources_by_jid, parts_by_name = _projected_outline_validation_context(
        session_state=session_state,
        prepared_bridge_request=prepared_bridge_request,
    )
    grounding_result = compile_grounded_outline_task(
        task,
        resources_by_jid=resources_by_jid,
        parts_by_name=parts_by_name,
        outline_contract=deepcopy(_MULTI_TURN_OUTLINE_CONTRACT),
        location_validation_mode="strict",
    )
    finding = grounding_result.get("finding")
    if isinstance(finding, dict):
        return [deepcopy(finding)], None

    grounded_action = dict(grounding_result.get("grounded_action") or {})
    if not grounded_action:
        return [], None

    llm_input = dict(prepared_bridge_request.get("llm_input") or {})
    findings: list[dict[str, Any]] = []
    findings.extend(
        _validate_outline_task_ra(
            planner=planner,
            task=task,
            grounded_action=grounded_action,
            resources_by_jid=resources_by_jid,
            parts_by_name=parts_by_name,
        )
    )
    findings.extend(
        _validate_outline_task_cca(
            task=task,
            grounded_action=grounded_action,
            resources_by_jid=resources_by_jid,
            parts_by_name=parts_by_name,
            llm_input=llm_input,
            prior_findings=findings,
        )
    )
    return findings, grounded_action


def _outline_validation_finding_key(
    finding: dict[str, Any],
) -> tuple[str, str, str, str, str]:
    """Small stable key for de-duping validation findings."""
    evidence = dict(finding.get("evidence") or {})
    return (
        str(finding.get("constraint_owner") or "").strip().lower(),
        str(finding.get("constraint_code") or "").strip().lower(),
        str(finding.get("resource_jid") or "").strip(),
        str(finding.get("part_name") or "").strip(),
        str(evidence.get("field") or evidence.get("token") or "").strip().lower(),
    )


def _merge_outline_validation_findings(
    existing: list[dict[str, Any]],
    new: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Append new findings while de-duping equivalent active findings."""
    merged: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    for row in list(existing or []) + list(new or []):
        if not isinstance(row, dict):
            continue
        merged[_outline_validation_finding_key(row)] = deepcopy(row)
    return list(merged.values())


def _finding_still_unresolved(
    finding: dict[str, Any],
    *,
    session_state: dict[str, Any],
    prepared_bridge_request: dict[str, Any],
) -> bool:
    """Return True when a validation finding should remain visible next turn."""
    constraint_code = str(finding.get("constraint_code") or "").strip().lower()
    resource_jid = str(finding.get("resource_jid") or "").strip()
    part_name = str(finding.get("part_name") or "").strip()
    del prepared_bridge_request

    symbolic_resources = dict(session_state.get("symbolic_resources") or {})
    symbolic_parts = dict(session_state.get("symbolic_parts") or {})
    resource_row = dict(symbolic_resources.get(resource_jid) or {})
    part_row = dict(symbolic_parts.get(part_name) or {})

    if constraint_code == "workspace_unreachable":
        if not resource_jid or not part_name:
            return False
        current_holder = str(part_row.get("current_holder_resource_jid") or "").strip()
        current_state = str(part_row.get("current_state") or "").strip().lower()
        current_location = str(part_row.get("current_location") or "").strip()
        if current_holder or current_state in {"held", "in_gripper", "assembled", "placed"}:
            return False
        if current_location:
            return False
        return True

    if constraint_code == "holder_conflict":
        if not part_name or not resource_jid:
            return False
        current_holder = str(part_row.get("current_holder_resource_jid") or "").strip()
        return bool(current_holder and current_holder != resource_jid)

    if constraint_code == "required_part_not_held":
        expected_part = part_name
        if not resource_jid or not expected_part:
            return False
        actual_held = str(resource_row.get("held_part") or "").strip()
        return actual_held != expected_part

    if constraint_code == "part_relocation_without_carrier":
        if not part_name or not resource_jid:
            return False
        current_holder = str(part_row.get("current_holder_resource_jid") or "").strip()
        actual_held = str(resource_row.get("held_part") or "").strip()
        return current_holder != resource_jid and actual_held != part_name

    return False


def _prune_resolved_outline_validation_findings(
    active_findings: list[dict[str, Any]],
    *,
    session_state: dict[str, Any],
    prepared_bridge_request: dict[str, Any],
) -> list[dict[str, Any]]:
    """Keep only findings that remain unresolved after an accepted task."""
    unresolved: list[dict[str, Any]] = []
    for row in active_findings or []:
        if not isinstance(row, dict):
            continue
        if _finding_still_unresolved(
            row,
            session_state=session_state,
            prepared_bridge_request=prepared_bridge_request,
        ):
            unresolved.append(deepcopy(row))
    return unresolved


def _active_continuation_conditions(
    prepared_bridge_request: dict[str, Any],
) -> list[dict[str, Any]]:
    llm_input = dict(prepared_bridge_request.get("llm_input") or {})
    modeled_gap = dict(llm_input.get("modeled_continuation_gap") or {})
    raw_conditions = (
        modeled_gap.get("unmet_continuation_conditions")
        or modeled_gap.get("unsatisfied_conditions")
        or []
    )
    return [deepcopy(row) for row in raw_conditions if isinstance(row, dict)]


def _normalized_blocker_kind(condition: dict[str, Any]) -> str:
    kind = str(condition.get("kind") or "").strip().lower()
    if kind == "focused_resource_terminal_state":
        return "resource_terminal_state"
    return kind


def _fault_event_fallback_parts(
    prepared_bridge_request: dict[str, Any],
) -> list[str]:
    llm_input = dict(prepared_bridge_request.get("llm_input") or {})
    fault_event = dict(llm_input.get("fault_event") or {})
    return [
        str(item).strip()
        for item in (fault_event.get("affected_part_names") or [])
        if str(item).strip()
    ]


def _extract_blocker_part_names(
    *,
    blocking_reason: str,
    parts_by_name: dict[str, dict[str, Any]],
    fallback_parts: list[str],
) -> list[str]:
    blocker_text = str(blocking_reason or "").strip().lower()
    blocker_parts = [
        part_name
        for part_name in parts_by_name
        if part_name and part_name.lower() in blocker_text
    ]
    if blocker_parts:
        return blocker_parts
    return [str(part_name).strip() for part_name in fallback_parts if str(part_name).strip()]


def _extract_safety_blocker_part_names(
    *,
    blocking_reason: str,
    parts_by_name: dict[str, dict[str, Any]],
    fallback_parts: list[str],
) -> list[str]:
    blocker_text = str(blocking_reason or "").strip()
    lowered = blocker_text.lower()
    if " before " in lowered:
        prefix = blocker_text[:lowered.index(" before ")].strip()
        blocker_parts = _extract_blocker_part_names(
            blocking_reason=prefix,
            parts_by_name=parts_by_name,
            fallback_parts=[],
        )
        if blocker_parts:
            return blocker_parts
    return _extract_blocker_part_names(
        blocking_reason=blocking_reason,
        parts_by_name=parts_by_name,
        fallback_parts=fallback_parts,
    )


def _condition_expected_matches(actual: Any, expected: Any) -> bool:
    if isinstance(expected, (dict, list)):
        return actual == expected
    if expected in (None, ""):
        return actual in (None, "")
    return str(actual or "").strip() == str(expected or "").strip()


def _continuation_condition_satisfied(
    condition: dict[str, Any],
    *,
    session_state: dict[str, Any],
    prepared_bridge_request: dict[str, Any],
) -> bool:
    kind = _normalized_blocker_kind(condition)
    entity_kind = str(condition.get("entity_kind") or "").strip().lower()
    entity = str(condition.get("entity") or "").strip()
    field = str(condition.get("field") or "").strip()
    expected = condition.get("expected")

    resources_by_jid, parts_by_name = _projected_outline_validation_context(
        session_state=session_state,
        prepared_bridge_request=prepared_bridge_request,
    )

    if kind == "resource_terminal_state" and entity_kind == "resource" and entity:
        row = dict(resources_by_jid.get(entity) or {})
        if not row:
            return False
        if field == "current_state":
            return _condition_expected_matches(row.get("current_state"), expected)
        if field == "current_location":
            return _condition_expected_matches(row.get("current_location"), expected)
        if field == "held_part":
            return _condition_expected_matches(row.get("held_part"), expected)
        if field == "gripper_state":
            return _condition_expected_matches(row.get("gripper_state"), expected)
        return False

    if kind == "resource_terminal_state" and entity_kind == "part" and entity:
        row = dict(parts_by_name.get(entity) or {})
        if not row:
            return False
        if field == "current_state":
            return _condition_expected_matches(row.get("current_state"), expected)
        if field == "current_location":
            return _condition_expected_matches(row.get("current_location"), expected)
        if field == "current_holder_resource_jid":
            return _condition_expected_matches(
                row.get("current_holder_resource_jid"), expected,
            )
        return False

    if kind != "safety_blocked_suffix_task":
        return False

    blocker_parts = _extract_safety_blocker_part_names(
        blocking_reason=str(condition.get("blocking_reason") or "").strip(),
        parts_by_name=parts_by_name,
        fallback_parts=_fault_event_fallback_parts(prepared_bridge_request),
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


def _candidate_recovery_blocker_key(
    blocker: dict[str, Any],
) -> tuple[str, str, str, str, str, str]:
    return (
        str(blocker.get("kind") or "").strip().lower(),
        str(blocker.get("entity_kind") or "").strip().lower(),
        str(blocker.get("entity") or "").strip(),
        str(blocker.get("field") or "").strip(),
        str(blocker.get("expected") or "").strip(),
        str(blocker.get("blocking_rule_id") or blocker.get("source_task_id") or "").strip(),
    )


def _candidate_recovery_blocker_summary(
    blocker: dict[str, Any],
    *,
    prepared_bridge_request: dict[str, Any],
    parts_by_name: dict[str, dict[str, Any]],
) -> str:
    kind = _normalized_blocker_kind(blocker)
    if kind == "resource_terminal_state":
        entity = str(blocker.get("entity") or "").strip()
        expected = str(blocker.get("expected") or "").strip()
        field = str(blocker.get("field") or "").strip()
        if entity and expected and field == "current_state":
            return f"{entity} must reach {expected}"
        if entity and expected and field:
            return f"{entity} {field} must reach {expected}"
        return f"{entity or 'resource'} blocker remains"

    if kind == "safety_blocked_suffix_task":
        blocking_reason = str(blocker.get("blocking_reason") or "").strip()
        if blocking_reason:
            return blocking_reason
        blocker_parts = _extract_safety_blocker_part_names(
            blocking_reason=blocking_reason,
            parts_by_name=parts_by_name,
            fallback_parts=_fault_event_fallback_parts(prepared_bridge_request),
        )
        if blocker_parts:
            blocker_part = blocker_parts[0]
            goal_location = str(
                dict(parts_by_name.get(blocker_part) or {}).get("goal_location") or ""
            ).strip()
            if goal_location:
                return f"{blocker_part} must be at {goal_location} before blocked suffix can resume"
            return f"{blocker_part} must be restored before blocked suffix can resume"
        return "Blocked suffix must be cleared before continuation can resume"

    return "Recovery blocker remains"


def _active_candidate_recovery_blockers(
    *,
    session_state: dict[str, Any],
    prepared_bridge_request: dict[str, Any],
) -> list[dict[str, Any]]:
    resources_by_jid, parts_by_name = _projected_outline_validation_context(
        session_state=session_state,
        prepared_bridge_request=prepared_bridge_request,
    )
    blockers: dict[tuple[str, str, str, str, str, str], dict[str, Any]] = {}
    for condition in _active_continuation_conditions(prepared_bridge_request):
        if not isinstance(condition, dict):
            continue
        kind = _normalized_blocker_kind(condition)
        if kind not in {"resource_terminal_state", "safety_blocked_suffix_task"}:
            continue
        if _continuation_condition_satisfied(
            condition,
            session_state=session_state,
            prepared_bridge_request=prepared_bridge_request,
        ):
            continue
        blocker = deepcopy(condition)
        blocker["kind"] = kind
        blocker["summary"] = _candidate_recovery_blocker_summary(
            blocker,
            prepared_bridge_request=prepared_bridge_request,
            parts_by_name=parts_by_name,
        )
        blockers[_candidate_recovery_blocker_key(blocker)] = blocker
    return list(blockers.values())


def _no_blocker_reduction_finding(
    *,
    task: dict[str, Any],
) -> dict[str, Any]:
    return {
        "task_id": str(task.get("outline_id") or "").strip(),
        "resource_jid": str(task.get("resource_jid") or "").strip() or None,
        "part_name": str(task.get("part_name") or "").strip() or None,
        "constraint_owner": "selector",
        "constraint_family": "candidate_selection",
        "constraint_code": "no_blocker_reduction",
        "reason": "Task does not directly reduce the current recovery blockers.",
        "evidence": {"field": "recovery_blockers"},
    }


def _count_resolved_continuation_conditions(
    *,
    candidate_session_state: dict[str, Any],
    prepared_bridge_request: dict[str, Any],
) -> tuple[int, int]:
    active_conditions = _active_continuation_conditions(prepared_bridge_request)
    if not active_conditions:
        return 0, 0
    unresolved_after = [
        row for row in active_conditions
        if not _continuation_condition_satisfied(
            row,
            session_state=candidate_session_state,
            prepared_bridge_request=prepared_bridge_request,
        )
    ]
    return len(active_conditions) - len(unresolved_after), len(unresolved_after)


def _candidate_session_after_task(
    *,
    session_state: dict[str, Any],
    task: dict[str, Any],
) -> dict[str, Any]:
    candidate_session_state = deepcopy(session_state)
    _apply_task_effects_to_symbolic_state(task, candidate_session_state)
    return candidate_session_state


def _part_release_frees_resource_for_blocker(
    *,
    task: dict[str, Any],
    session_state: dict[str, Any],
    candidate_session_state: dict[str, Any],
    prepared_bridge_request: dict[str, Any],
    current_blockers: list[dict[str, Any]],
) -> bool:
    if not _task_ends_with_part_clear_of_resource(task):
        return False

    resource_jid = str(task.get("resource_jid") or "").strip()

    # Only block if THIS resource has a terminal state blocker — not if some
    # other resource does.  Allows cross-assignment recovery.
    if any(
        _normalized_blocker_kind(blocker) == "resource_terminal_state"
        and str(blocker.get("entity") or "").strip() == resource_jid
        for blocker in current_blockers
        if isinstance(blocker, dict)
    ):
        return False
    released_part = str(task.get("part_name") or "").strip()
    if not resource_jid or not released_part:
        return False

    resources_by_jid, parts_by_name = _projected_outline_validation_context(
        session_state=session_state,
        prepared_bridge_request=prepared_bridge_request,
    )
    candidate_resources_by_jid, _ = _projected_outline_validation_context(
        session_state=candidate_session_state,
        prepared_bridge_request=prepared_bridge_request,
    )
    current_resource = dict(resources_by_jid.get(resource_jid) or {})
    candidate_resource = dict(candidate_resources_by_jid.get(resource_jid) or {})

    if str(current_resource.get("held_part") or "").strip() != released_part:
        return False
    if str(candidate_resource.get("held_part") or "").strip():
        return False

    blocker_parts = _current_safety_blocker_parts(
        current_blockers=current_blockers,
        parts_by_name=parts_by_name,
        prepared_bridge_request=prepared_bridge_request,
    )

    if not blocker_parts:
        return False
    if released_part in blocker_parts:
        return False
    return True


def _current_safety_blocker_parts(
    *,
    current_blockers: list[dict[str, Any]],
    parts_by_name: dict[str, dict[str, Any]],
    prepared_bridge_request: dict[str, Any],
) -> set[str]:
    blocker_parts: set[str] = set()
    for blocker in current_blockers:
        if not isinstance(blocker, dict):
            continue
        if _normalized_blocker_kind(blocker) != "safety_blocked_suffix_task":
            continue
        blocker_parts.update(
            _extract_safety_blocker_part_names(
                blocking_reason=str(blocker.get("blocking_reason") or "").strip(),
                parts_by_name=parts_by_name,
                fallback_parts=_fault_event_fallback_parts(prepared_bridge_request),
            )
        )
    return blocker_parts


def _part_acquisition_counts_as_blocker_progress(
    *,
    task: dict[str, Any],
    session_state: dict[str, Any],
    candidate_session_state: dict[str, Any],
    prepared_bridge_request: dict[str, Any],
    current_blockers: list[dict[str, Any]],
) -> bool:
    if not _task_ends_with_part_held_by_resource(task):
        return False

    resource_jid = str(task.get("resource_jid") or "").strip()

    # Only block if the resource performing this action has a terminal state
    # blocker — not if some OTHER resource does.  This allows cross-assignment
    # (e.g. ur5e acquiring LG when xarm6 is in failed state).
    if any(
        _normalized_blocker_kind(blocker) == "resource_terminal_state"
        and str(blocker.get("entity") or "").strip() == resource_jid
        for blocker in current_blockers
        if isinstance(blocker, dict)
    ):
        return False
    acquired_part = str(task.get("part_name") or "").strip()
    if not resource_jid or not acquired_part:
        return False

    _, parts_by_name = _projected_outline_validation_context(
        session_state=session_state,
        prepared_bridge_request=prepared_bridge_request,
    )
    blocker_parts = _current_safety_blocker_parts(
        current_blockers=current_blockers,
        parts_by_name=parts_by_name,
        prepared_bridge_request=prepared_bridge_request,
    )
    if acquired_part not in blocker_parts:
        return False

    candidate_resources_by_jid, candidate_parts_by_name = _projected_outline_validation_context(
        session_state=candidate_session_state,
        prepared_bridge_request=prepared_bridge_request,
    )
    candidate_resource = dict(candidate_resources_by_jid.get(resource_jid) or {})
    candidate_part = dict(candidate_parts_by_name.get(acquired_part) or {})
    if str(candidate_resource.get("held_part") or "").strip() != acquired_part:
        return False
    if str(candidate_part.get("current_holder_resource_jid") or "").strip() != resource_jid:
        return False
    return True


def _preparatory_transit_toward_blocker(
    *,
    task: dict[str, Any],
    session_state: dict[str, Any],
    candidate_session_state: dict[str, Any],
    prepared_bridge_request: dict[str, Any],
    current_blockers: list[dict[str, Any]],
) -> bool:
    """Credit preparatory moves where a resource holds a blocker part and
    the task represents a meaningful physical transit step (e.g. moving toward
    the goal location or to an intermediate staging position).

    This prevents rejection of valid intermediate actions like "transit LG to
    approach position" when the resource already carries the blocker part.
    """
    resource_jid = str(task.get("resource_jid") or "").strip()
    part_name = str(task.get("part_name") or "").strip()
    if not resource_jid or not part_name:
        return False

    # The resource must currently hold the part.
    resources_by_jid, parts_by_name = _projected_outline_validation_context(
        session_state=session_state,
        prepared_bridge_request=prepared_bridge_request,
    )
    resource_row = dict(resources_by_jid.get(resource_jid) or {})
    if str(resource_row.get("held_part") or "").strip() != part_name:
        return False

    # The held part must be relevant to a current blocker.
    blocker_parts = _current_safety_blocker_parts(
        current_blockers=current_blockers,
        parts_by_name=parts_by_name,
        prepared_bridge_request=prepared_bridge_request,
    )
    if part_name not in blocker_parts:
        return False

    # The task must have a target_ref or change resource location — i.e. it's
    # actually commanding a physical move, not a no-op.
    target_ref = str(task.get("target_ref") or "").strip()
    action_target = dict(task.get("action_target") or {})
    target_location = str(action_target.get("target_location") or "").strip()
    end_state = dict(task.get("expected_end_state") or {})
    end_part_location = str(end_state.get("part_location") or "").strip()

    if target_ref or target_location or end_part_location:
        return True

    # Even without explicit target, if the symbolic state changes (e.g.
    # resource location moves) we credit it.
    candidate_resources_by_jid, _ = _projected_outline_validation_context(
        session_state=candidate_session_state,
        prepared_bridge_request=prepared_bridge_request,
    )
    candidate_resource = dict(candidate_resources_by_jid.get(resource_jid) or {})
    if (
        str(candidate_resource.get("current_location") or "").strip()
        != str(resource_row.get("current_location") or "").strip()
    ):
        return True

    return False


def _candidate_pruned_task_match_key(task: dict[str, Any]) -> str:
    return json.dumps(
        {
            "resource_jid": str(task.get("resource_jid") or "").strip(),
            "part_name": str(task.get("part_name") or "").strip(),
            "target_ref": str(task.get("target_ref") or "").strip(),
            "effect": _candidate_effect_match_key(task),
        },
        sort_keys=True,
        ensure_ascii=True,
    )


def _is_durable_candidate_finding(finding: dict[str, Any]) -> bool:
    constraint_code = str(finding.get("constraint_code") or "").strip().lower()
    return constraint_code in _DURABLE_PRUNED_CONSTRAINT_CODES


def _current_candidate_state_signature(
    *,
    task: dict[str, Any],
    finding: dict[str, Any],
    session_state: dict[str, Any],
    prepared_bridge_request: dict[str, Any],
) -> str:
    resources_by_jid, parts_by_name = _projected_outline_validation_context(
        session_state=session_state,
        prepared_bridge_request=prepared_bridge_request,
    )
    resource_jid = str(
        task.get("resource_jid")
        or finding.get("resource_jid")
        or ""
    ).strip()
    part_name = str(
        task.get("part_name")
        or finding.get("part_name")
        or ""
    ).strip()
    target_ref = str(task.get("target_ref") or "").strip()
    constraint_code = str(finding.get("constraint_code") or "").strip().lower()
    resource_row = dict(resources_by_jid.get(resource_jid) or {})
    part_row = dict(parts_by_name.get(part_name) or {})
    active_blockers = _active_candidate_recovery_blockers(
        session_state=session_state,
        prepared_bridge_request=prepared_bridge_request,
    )
    payload: dict[str, Any] = {
        "constraint_code": constraint_code,
        "resource_jid": resource_jid,
        "part_name": part_name,
        "target_ref": target_ref,
    }
    if constraint_code == "workspace_unreachable":
        payload.update({
            "part_current_location": str(part_row.get("current_location") or "").strip(),
            "part_current_holder_resource_jid": str(
                part_row.get("current_holder_resource_jid") or ""
            ).strip(),
            "has_observed_pose": bool(dict(part_row.get("observed_pose") or {})),
            "checked_pose": deepcopy(
                finding.get("checked_pose")
                or finding.get("pose")
                or part_row.get("observed_pose")
                or {}
            ),
            "workspace_bounds": deepcopy(
                finding.get("workspace_bounds")
                or resource_row.get("workspace_bounds")
                or {}
            ),
        })
    elif constraint_code in {
        "holder_conflict",
        "required_part_not_held",
        "part_relocation_without_carrier",
    }:
        payload.update({
            "resource_held_part": str(resource_row.get("held_part") or "").strip(),
            "part_current_holder_resource_jid": str(
                part_row.get("current_holder_resource_jid") or ""
            ).strip(),
            "part_current_location": str(part_row.get("current_location") or "").strip(),
            "part_current_state": str(part_row.get("current_state") or "").strip(),
        })
    elif constraint_code == "source_reference_unavailable":
        payload.update({
            "part_current_holder_resource_jid": str(
                part_row.get("current_holder_resource_jid") or ""
            ).strip(),
            "part_current_location": str(part_row.get("current_location") or "").strip(),
            "has_observed_pose": bool(dict(part_row.get("observed_pose") or {})),
        })
    elif constraint_code in {
        "blocker_open",
        "dependency_unsatisfied",
        "order_violation",
        "safety_rule_violation",
    }:
        payload.update({
            "condition_ids": list(finding.get("condition_ids") or []),
            "rule_id": str(finding.get("rule_id") or "").strip(),
            "active_blockers": [
                str(row.get("summary") or "").strip()
                for row in active_blockers
                if isinstance(row, dict) and str(row.get("summary") or "").strip()
            ],
        })
    else:
        payload.update({
            "resource_held_part": str(resource_row.get("held_part") or "").strip(),
            "part_current_holder_resource_jid": str(
                part_row.get("current_holder_resource_jid") or ""
            ).strip(),
            "part_current_location": str(part_row.get("current_location") or "").strip(),
        })
    return json.dumps(payload, sort_keys=True, default=str, ensure_ascii=True)


def _build_durable_pruned_action_row(
    *,
    task: dict[str, Any],
    finding: dict[str, Any],
    activation_signature: str,
    repeat_count: int,
) -> dict[str, Any]:
    return {
        "resource_jid": str(task.get("resource_jid") or "").strip(),
        "part_name": str(task.get("part_name") or "").strip(),
        "target_ref": str(task.get("target_ref") or "").strip(),
        "task": deepcopy(task),
        "action": deepcopy(task),
        "guard": deepcopy(finding),
        "reason": str(finding.get("reason") or "").strip(),
        "activation_signature": activation_signature,
        "repeat_count": max(1, int(repeat_count or 1)),
    }


def _active_pruned_actions(
    session_state: dict[str, Any],
    prepared_bridge_request: dict[str, Any],
) -> list[dict[str, Any]]:
    active_rows: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str, str]] = set()
    for raw_row in (session_state.get("pruned_actions") or []):
        if not isinstance(raw_row, dict):
            continue
        task = dict(raw_row.get("task") or raw_row.get("action") or {})
        guard = dict(raw_row.get("guard") or {})
        activation_signature = str(raw_row.get("activation_signature") or "").strip()
        if not task or not guard or not activation_signature:
            continue
        if not _is_durable_candidate_finding(guard):
            continue
        current_signature = _current_candidate_state_signature(
            task=task,
            finding=guard,
            session_state=session_state,
            prepared_bridge_request=prepared_bridge_request,
        )
        if current_signature != activation_signature:
            continue
        dedupe_key = (
            _candidate_pruned_task_match_key(task),
            str(guard.get("constraint_code") or "").strip().lower(),
            activation_signature,
        )
        if dedupe_key in seen_keys:
            continue
        seen_keys.add(dedupe_key)
        active_rows.append(deepcopy(raw_row))
    return active_rows


def _matching_active_pruned_action(
    *,
    task: dict[str, Any],
    session_state: dict[str, Any],
    prepared_bridge_request: dict[str, Any],
) -> dict[str, Any] | None:
    task_key = _candidate_pruned_task_match_key(task)
    for row in _active_pruned_actions(session_state, prepared_bridge_request):
        row_task = dict(row.get("task") or row.get("action") or {})
        if _candidate_pruned_task_match_key(row_task) != task_key:
            continue
        return deepcopy(row)
    return None


def _retarget_candidate_finding_to_task(
    finding: dict[str, Any],
    task: dict[str, Any],
) -> dict[str, Any]:
    retargeted = deepcopy(finding or {})
    outline_id = str(task.get("outline_id") or "").strip()
    resource_jid = str(task.get("resource_jid") or "").strip()
    part_name = str(task.get("part_name") or "").strip()
    if outline_id:
        retargeted["task_id"] = outline_id
    if resource_jid:
        retargeted["resource_jid"] = resource_jid
    if part_name:
        retargeted["part_name"] = part_name
    return retargeted


def _promote_durable_candidate_rejections(
    *,
    session_state: dict[str, Any],
    prepared_bridge_request: dict[str, Any],
    candidate_evaluations: list[dict[str, Any]],
) -> None:
    prune_history = dict(session_state.get("candidate_prune_history") or {})
    active_rows = _active_pruned_actions(session_state, prepared_bridge_request)
    seen_active = {
        (
            _candidate_pruned_task_match_key(dict(row.get("task") or row.get("action") or {})),
            str(dict(row.get("guard") or {}).get("constraint_code") or "").strip().lower(),
            str(row.get("activation_signature") or "").strip(),
        )
        for row in active_rows
        if isinstance(row, dict)
    }

    for evaluation in candidate_evaluations:
        if not isinstance(evaluation, dict) or bool(evaluation.get("valid")):
            continue
        task = dict(evaluation.get("task") or {})
        findings = [
            dict(item)
            for item in (evaluation.get("validation_findings") or [])
            if isinstance(item, dict)
        ]
        durable_finding = next(
            (item for item in findings if _is_durable_candidate_finding(item)),
            None,
        )
        if durable_finding is None:
            continue
        activation_signature = _current_candidate_state_signature(
            task=task,
            finding=durable_finding,
            session_state=session_state,
            prepared_bridge_request=prepared_bridge_request,
        )
        history_key = json.dumps(
            {
                "task": _candidate_pruned_task_match_key(task),
                "constraint_code": str(
                    durable_finding.get("constraint_code") or ""
                ).strip().lower(),
                "activation_signature": activation_signature,
            },
            sort_keys=True,
            ensure_ascii=True,
        )
        repeat_count = int(prune_history.get(history_key) or 0) + 1
        prune_history[history_key] = repeat_count
        if repeat_count < _CANDIDATE_PRUNE_REPEAT_THRESHOLD:
            continue
        dedupe_key = (
            _candidate_pruned_task_match_key(task),
            str(durable_finding.get("constraint_code") or "").strip().lower(),
            activation_signature,
        )
        if dedupe_key in seen_active:
            continue
        seen_active.add(dedupe_key)
        active_rows.append(
            _build_durable_pruned_action_row(
                task=task,
                finding=durable_finding,
                activation_signature=activation_signature,
                repeat_count=repeat_count,
            )
        )

    session_state["candidate_prune_history"] = prune_history
    session_state["pruned_actions"] = active_rows


def _compute_enabled_candidate_bound(
    session_state: dict[str, Any],
    prepared_bridge_request: dict[str, Any],
) -> int:
    """Return the configured candidate budget for candidate selection."""
    del prepared_bridge_request
    candidate_bound = max(
        1,
        int(session_state.get("candidate_bound") or _DEFAULT_CANDIDATE_BOUND),
    )
    candidate_bound_cap = max(
        1,
        int(session_state.get("candidate_bound_cap") or _DEFAULT_CANDIDATE_BOUND_CAP),
    )
    return min(candidate_bound, candidate_bound_cap)


def _candidate_progress_score(
    *,
    task: dict[str, Any],
    session_state: dict[str, Any],
    prepared_bridge_request: dict[str, Any],
) -> tuple[int, dict[str, int]]:
    current_blockers = _active_candidate_recovery_blockers(
        session_state=session_state,
        prepared_bridge_request=prepared_bridge_request,
    )
    candidate_session_state = _candidate_session_after_task(
        session_state=session_state,
        task=task,
    )
    remaining_blockers = _active_candidate_recovery_blockers(
        session_state=candidate_session_state,
        prepared_bridge_request=prepared_bridge_request,
    )
    current_keys = {
        _candidate_recovery_blocker_key(row)
        for row in current_blockers
        if isinstance(row, dict)
    }
    remaining_keys = {
        _candidate_recovery_blocker_key(row)
        for row in remaining_blockers
        if isinstance(row, dict)
    }
    resolved_blockers = len(current_keys - remaining_keys)
    blocker_part_acquired = 0
    if resolved_blockers == 0 and _part_acquisition_counts_as_blocker_progress(
        task=task,
        session_state=session_state,
        candidate_session_state=candidate_session_state,
        prepared_bridge_request=prepared_bridge_request,
        current_blockers=current_blockers,
    ):
        blocker_part_acquired = 1
    resource_freed_for_blocker = 0
    if resolved_blockers == 0 and blocker_part_acquired == 0 and _part_release_frees_resource_for_blocker(
        task=task,
        session_state=session_state,
        candidate_session_state=candidate_session_state,
        prepared_bridge_request=prepared_bridge_request,
        current_blockers=current_blockers,
    ):
        resource_freed_for_blocker = 1
    preparatory_transit = 0
    if (
        resolved_blockers == 0
        and blocker_part_acquired == 0
        and resource_freed_for_blocker == 0
        and _preparatory_transit_toward_blocker(
            task=task,
            session_state=session_state,
            candidate_session_state=candidate_session_state,
            prepared_bridge_request=prepared_bridge_request,
            current_blockers=current_blockers,
        )
    ):
        preparatory_transit = 1
    remaining_blocked_issues = len(remaining_keys)
    secondary_progress = blocker_part_acquired + resource_freed_for_blocker + preparatory_transit
    return (
        resolved_blockers + secondary_progress,
        {
            "resolved_direct_blockers": resolved_blockers,
            "blocker_part_acquired": blocker_part_acquired,
            "freed_resource_for_blocker": resource_freed_for_blocker,
            "preparatory_transit": preparatory_transit,
            "resolved_continuation_conditions": resolved_blockers,
            "remaining_continuation_conditions": remaining_blocked_issues,
            "remaining_blocked_issues": remaining_blocked_issues,
        },
    )


def _candidate_feedback_rows(
    candidate_evaluations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in candidate_evaluations:
        if not isinstance(row, dict) or bool(row.get("valid")):
            continue
        findings = [
            deepcopy(item)
            for item in (row.get("validation_findings") or [])
            if isinstance(item, dict)
        ]
        if not findings:
            continue
        feedback_row = {
            "candidate_index": int(row.get("candidate_index") or 0),
            "task": deepcopy(dict(row.get("task") or {})),
            "validation_findings": findings,
        }
        if isinstance(row.get("surface_task"), dict):
            feedback_row["surface_task"] = deepcopy(dict(row.get("surface_task") or {}))
        if isinstance(row.get("repaired_task"), dict):
            feedback_row["repaired_task"] = deepcopy(dict(row.get("repaired_task") or {}))
        if str(row.get("repair_applied") or "").strip():
            feedback_row["repair_applied"] = str(row.get("repair_applied") or "").strip()
        rows.append(feedback_row)
    return rows


def _finding_event_status_for_logging(finding: dict[str, Any]) -> str:
    constraint_code = str(finding.get("constraint_code") or "").strip().lower()
    constraint_family = str(finding.get("constraint_family") or "").strip().lower()
    if constraint_code in {
        "blocker_open",
        "dependency_unsatisfied",
        "order_violation",
        "invalid_dependency_reference",
        "claimed_condition_not_currently_unmet",
        "safety_rule_violation",
    } or constraint_family in {"safety", "continuation"}:
        return "blocked_by_supervisor"
    if constraint_code in {"no_state_change", "no_blocker_reduction"} or constraint_family == "candidate_selection":
        return "nonprogressing"
    return "disabled"


def _remaining_blocked_issue_counts(
    *,
    session_state: dict[str, Any],
    prepared_bridge_request: dict[str, Any],
) -> tuple[int, int]:
    remaining_blockers = len(
        _active_candidate_recovery_blockers(
            session_state=session_state,
            prepared_bridge_request=prepared_bridge_request,
        )
    )
    return 0, remaining_blockers


def _next_recovery_sequence_index(session_state: dict[str, Any]) -> int:
    return len(list(session_state.get("accepted_outline_prefix") or [])) + 1


def _candidate_outline_id(*, sequence_index: int, candidate_index: int) -> str:
    return f"RECOVERY_SEQ{sequence_index}_{candidate_index + 1}"


def _committed_outline_id(*, sequence_index: int) -> str:
    return f"RECOVERY_SEQ{sequence_index}"


def _candidate_schema_finding(
    *,
    task: dict[str, Any],
    reason: str,
    evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "task_id": str(task.get("outline_id") or "").strip(),
        "resource_jid": str(task.get("resource_jid") or "").strip() or None,
        "part_name": str(task.get("part_name") or "").strip() or None,
        "constraint_owner": "binding",
        "constraint_family": "binding",
        "constraint_code": "candidate_schema_violation",
        "reason": reason,
        "evidence": deepcopy(evidence or {}),
    }


def _candidate_named_pose_tokens(resource_row: dict[str, Any]) -> set[str]:
    tokens: set[str] = set()
    raw_named_poses = resource_row.get("named_poses")
    if isinstance(raw_named_poses, dict):
        tokens.update(
            str(token).strip()
            for token in raw_named_poses.keys()
            if str(token).strip()
        )
    else:
        tokens.update(
            str(token).strip()
            for token in (raw_named_poses or [])
            if str(token).strip()
        )
    tokens.update(
        str(token).strip()
        for token in (resource_row.get("available_named_poses") or [])
        if str(token).strip()
    )
    return tokens


def _part_current_location_token(part_row: dict[str, Any]) -> str:
    return str(part_row.get("current_location") or part_row.get("location") or "").strip()


def _part_current_holder_token(part_row: dict[str, Any]) -> str:
    return str(
        part_row.get("current_holder_resource_jid")
        or part_row.get("holder_resource_jid")
        or ""
    ).strip()


def _part_current_state_token(part_row: dict[str, Any]) -> str:
    return str(part_row.get("current_state") or part_row.get("state") or "").strip()


def _resource_current_state_token(resource_row: dict[str, Any]) -> str:
    return str(resource_row.get("current_state") or resource_row.get("state") or "").strip()


def _candidate_target_ref_from_surface_task(task: dict[str, Any]) -> str:
    if str(task.get("target_ref") or "").strip():
        return str(task.get("target_ref") or "").strip()
    action_target = dict(task.get("action_target") or {})
    return str(
        action_target.get("target_location")
        or action_target.get("named_pose")
        or ""
    ).strip()


def _task_ends_with_part_held_by_resource(task: dict[str, Any]) -> bool:
    resource_jid = str(task.get("resource_jid") or "").strip()
    part_name = str(task.get("part_name") or "").strip()
    end_state = dict(task.get("expected_end_state") or {})
    held_part = str(end_state.get("held_part") or "").strip()
    part_holder = str(end_state.get("part_holder_resource_jid") or "").strip()
    return bool(
        part_name
        and resource_jid
        and (held_part == part_name or part_holder == resource_jid)
    )


def _task_ends_with_part_clear_of_resource(task: dict[str, Any]) -> bool:
    part_name = str(task.get("part_name") or "").strip()
    if not part_name:
        return False
    end_state = dict(task.get("expected_end_state") or {})
    held_part = end_state.get("held_part")
    part_holder = end_state.get("part_holder_resource_jid")
    return bool(
        ("part_location" in end_state or str(task.get("target_ref") or "").strip())
        and (held_part in (None, ""))
        and (part_holder in (None, ""))
    )


def _candidate_effect_match_key(task: dict[str, Any]) -> dict[str, Any]:
    return {
        "expected_end_state": deepcopy(task.get("expected_end_state") or {}),
        "action_target": deepcopy(task.get("action_target") or {}),
    }


def _normalize_candidate_task(
    *,
    task: dict[str, Any],
    sequence_index: int,
    candidate_index: int,
) -> dict[str, Any]:
    raw_task = deepcopy(task or {})
    original_outline_id = str(raw_task.get("outline_id") or "").strip()
    normalized: dict[str, Any] = {
        "resource_jid": str(raw_task.get("resource_jid") or "").strip(),
    }
    event_name = str(raw_task.get("event_name") or "").strip()
    description = str(raw_task.get("description") or "").strip()
    part_name = str(raw_task.get("part_name") or "").strip()
    target_ref = _candidate_target_ref_from_surface_task(raw_task)
    if event_name:
        normalized["event_name"] = event_name
    if description:
        normalized["description"] = description
    if part_name:
        normalized["part_name"] = part_name
    if target_ref:
        normalized["target_ref"] = target_ref
    raw_start_state = raw_task.get("expected_start_state")
    if isinstance(raw_start_state, dict):
        normalized["expected_start_state"] = deepcopy(raw_start_state)
    raw_end_state = raw_task.get("expected_end_state")
    if isinstance(raw_end_state, dict):
        normalized["expected_end_state"] = deepcopy(raw_end_state)
    if original_outline_id:
        normalized["llm_outline_id"] = original_outline_id
    normalized["outline_id"] = _candidate_outline_id(
        sequence_index=sequence_index,
        candidate_index=candidate_index,
    )
    return normalized


def _candidate_state_completeness_findings(
    *,
    candidate_task: dict[str, Any],
    part_name: str,
    start_state: dict[str, Any],
    end_state: dict[str, Any],
) -> list[dict[str, Any]]:
    """Require the LLM to emit every canonical predicate key.

    Because the bridge no longer synthesizes state, any missing key would
    leave the symbolic state stale. Surface the omission as a schema
    finding so the LLM is asked to emit the field.
    """
    findings: list[dict[str, Any]] = []
    required_always = ("resource_state",)
    required_with_part = (
        "held_part",
        "part_state",
        "part_location",
        "part_holder_resource_jid",
    )
    required_keys = list(required_always) + (list(required_with_part) if part_name else [])
    for side, state in (("expected_start_state", start_state), ("expected_end_state", end_state)):
        missing = [key for key in required_keys if key not in state]
        if missing:
            findings.append(
                _candidate_schema_finding(
                    task=candidate_task,
                    reason=(
                        f"{side} is missing required predicate key(s): "
                        f"{', '.join(missing)}"
                    ),
                    evidence={"field": side, "missing": missing},
                )
            )
    return findings


def _candidate_state_consistency_findings(
    *,
    candidate_task: dict[str, Any],
    resource_jid: str,
    part_name: str,
    end_state: dict[str, Any],
) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    if part_name and "held_part" in end_state:
        held = end_state.get("held_part")
        if held not in (None, "", part_name):
            findings.append(
                _candidate_schema_finding(
                    task=candidate_task,
                    reason=(
                        f"expected_end_state.held_part '{held}' contradicts "
                        f"part_name slot '{part_name}' (must equal part_name or null)"
                    ),
                    evidence={
                        "field": "expected_end_state.held_part",
                        "held_part": held,
                        "part_name": part_name,
                    },
                )
            )
    if resource_jid and "part_holder_resource_jid" in end_state:
        holder = end_state.get("part_holder_resource_jid")
        if holder not in (None, "", resource_jid):
            findings.append(
                _candidate_schema_finding(
                    task=candidate_task,
                    reason=(
                        f"expected_end_state.part_holder_resource_jid '{holder}' "
                        f"contradicts resource_jid slot '{resource_jid}' "
                        "(must equal resource_jid or null)"
                    ),
                    evidence={
                        "field": "expected_end_state.part_holder_resource_jid",
                        "part_holder_resource_jid": holder,
                        "resource_jid": resource_jid,
                    },
                )
            )
    return findings


def _derive_candidate_outline_task(
    *,
    candidate_task: dict[str, Any],
    session_state: dict[str, Any],
    prepared_bridge_request: dict[str, Any],
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    resources_by_jid, parts_by_name = _projected_outline_validation_context(
        session_state=session_state,
        prepared_bridge_request=prepared_bridge_request,
    )
    resource_jid = str(candidate_task.get("resource_jid") or "").strip()
    event_name = str(candidate_task.get("event_name") or "").strip()
    part_name = str(candidate_task.get("part_name") or "").strip()
    target_ref = str(candidate_task.get("target_ref") or "").strip()
    description = str(candidate_task.get("description") or "").strip()
    resource_row = dict(resources_by_jid.get(resource_jid) or {})
    part_row = dict(parts_by_name.get(part_name) or {}) if part_name else {}

    if not resource_jid:
        return None, [
            _candidate_schema_finding(
                task=candidate_task,
                reason="candidate must include resource_jid",
                evidence={"field": "resource_jid"},
            )
        ]

    if not event_name:
        return None, [
            _candidate_schema_finding(
                task=candidate_task,
                reason="candidate must include event_name naming the DES event",
                evidence={"field": "event_name"},
            )
        ]

    if not description:
        return None, [
            _candidate_schema_finding(
                task=candidate_task,
                reason="candidate must include description",
                evidence={"field": "description"},
            )
        ]

    if part_name:
        if part_name not in parts_by_name:
            return None, [
                _candidate_schema_finding(
                    task=candidate_task,
                    reason=f"candidate references unknown part '{part_name}'",
                    evidence={"field": "part_name", "token": part_name},
                )
            ]

    raw_start_state = candidate_task.get("expected_start_state")
    if not isinstance(raw_start_state, dict):
        return None, [
            _candidate_schema_finding(
                task=candidate_task,
                reason="candidate must include expected_start_state object",
                evidence={"field": "expected_start_state"},
            )
        ]
    raw_end_state = candidate_task.get("expected_end_state")
    if not isinstance(raw_end_state, dict):
        return None, [
            _candidate_schema_finding(
                task=candidate_task,
                reason="candidate must include expected_end_state object",
                evidence={"field": "expected_end_state"},
            )
        ]
    start_state: dict[str, Any] = deepcopy(raw_start_state)
    end_state: dict[str, Any] = deepcopy(raw_end_state)

    completeness_findings = _candidate_state_completeness_findings(
        candidate_task=candidate_task,
        part_name=part_name,
        start_state=start_state,
        end_state=end_state,
    )
    if completeness_findings:
        return None, completeness_findings

    consistency_findings = _candidate_state_consistency_findings(
        candidate_task=candidate_task,
        resource_jid=resource_jid,
        part_name=part_name,
        end_state=end_state,
    )
    if consistency_findings:
        return None, consistency_findings

    action_target: dict[str, Any] = {}
    if part_name and not target_ref:
        current_part_location = _part_current_location_token(part_row)
        if current_part_location:
            action_target["source_location"] = current_part_location
        elif isinstance(part_row.get("observed_pose"), dict) and "x" in dict(part_row.get("observed_pose") or {}):
            action_target["source_location"] = "observed_pose"
    elif part_name and target_ref:
        action_target["target_location"] = target_ref
    elif target_ref:
        if target_ref in _candidate_named_pose_tokens(resource_row):
            action_target["named_pose"] = target_ref
        else:
            action_target["target_location"] = target_ref

    normalized_task: dict[str, Any] = {
        "outline_id": str(candidate_task.get("outline_id") or "").strip(),
        "resource_jid": resource_jid,
        "event_name": event_name,
        "description": description,
        "expected_start_state": start_state,
        "expected_end_state": end_state,
    }
    if part_name:
        normalized_task["part_name"] = part_name
    if target_ref:
        normalized_task["target_ref"] = target_ref
    if action_target:
        normalized_task["action_target"] = action_target
    if str(candidate_task.get("llm_outline_id") or "").strip():
        normalized_task["llm_outline_id"] = str(candidate_task.get("llm_outline_id") or "").strip()
    return normalized_task, []


def _commit_selected_candidate_task(
    *,
    task: dict[str, Any],
    sequence_index: int,
) -> dict[str, Any]:
    committed = deepcopy(task or {})
    committed["candidate_outline_id"] = str(committed.get("outline_id") or "").strip()
    committed["outline_id"] = _committed_outline_id(sequence_index=sequence_index)
    return committed


def _apply_task_effects_to_symbolic_state(
    task: dict[str, Any],
    session_state: dict[str, Any],
) -> None:
    """Update symbolic resource/part state from accepted task's expected_end_state.

    Pass-through only: the LLM is authoritative. No slot-based inference.
    Missing end-state keys leave the corresponding symbolic field unchanged;
    a completeness check in _derive_candidate_outline_task surfaces omissions
    at candidate-validation time.
    """
    resource_jid = str(task.get("resource_jid") or "").strip()
    part_name = str(task.get("part_name") or "").strip()
    end_state = dict(task.get("expected_end_state") or {})

    symbolic_resources = dict(session_state.get("symbolic_resources") or {})
    symbolic_parts = dict(session_state.get("symbolic_parts") or {})

    if resource_jid:
        res = symbolic_resources.setdefault(resource_jid, {"resource_jid": resource_jid})
        if "resource_state" in end_state:
            res["current_state"] = end_state["resource_state"]
        if "held_part" in end_state:
            res["held_part"] = end_state["held_part"] or None
            res["gripper_state"] = "closed" if end_state["held_part"] else "open"

    if part_name:
        part = symbolic_parts.setdefault(part_name, {"part_name": part_name})
        if "part_state" in end_state:
            part["current_state"] = end_state["part_state"]
        if "part_location" in end_state:
            part["current_location"] = end_state["part_location"] or None
        if "part_holder_resource_jid" in end_state:
            part["current_holder_resource_jid"] = (
                end_state["part_holder_resource_jid"] or None
            )

    session_state["symbolic_resources"] = symbolic_resources
    session_state["symbolic_parts"] = symbolic_parts


def _parsed_response_object(
    parsed_response: dict[str, Any],
    *,
    primary_key: str,
) -> dict[str, Any]:
    """Read one DES transition object from the canonical response key."""
    raw = parsed_response.get(primary_key)
    return dict(raw or {}) if isinstance(raw, dict) else {}


def _parsed_response_rows(
    parsed_response: dict[str, Any],
    *,
    primary_key: str,
) -> list[dict[str, Any]]:
    """Read DES transition rows from the canonical response key."""
    raw = parsed_response.get(primary_key)
    return [dict(row) for row in (raw or []) if isinstance(row, dict)]


async def _handle_outline_single_pass(
    *,
    session_state: dict[str, Any],
    parsed_response: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    """Single-pass: LLM proposes all tasks at once, accept without validation."""
    turn_entry: dict[str, Any] = {}

    transition_trace = _parsed_response_rows(
        parsed_response,
        primary_key="transition_trace",
    )
    turn_entry["transition_trace"] = deepcopy(transition_trace)

    if not transition_trace:
        turn_entry["error"] = "outline response missing transition_trace"
        _logger.warning("[MultiTurn] outline single_pass: no transition_trace")
        return "need_revision", turn_entry

    session_state["accepted_outline_prefix"] = deepcopy(transition_trace)
    _sync_des_recovery_aliases(session_state, turn_entry=turn_entry)

    _logger.info(
        "[MultiTurn] outline single_pass: accepted %d recovery events",
        len(transition_trace),
    )

    session_state["status"] = "paused_after_outline_turn"
    return "outline_ready", turn_entry


async def _handle_outline_incremental(
    *,
    session_state: dict[str, Any],
    parsed_response: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    """Incremental: one task at a time, no validation."""
    turn_entry: dict[str, Any] = {}

    next_transition = _parsed_response_object(
        parsed_response,
        primary_key="next_transition",
    )
    transition_suffix = _parsed_response_rows(
        parsed_response,
        primary_key="transition_suffix",
    )

    turn_entry["next_transition"] = deepcopy(next_transition)
    turn_entry["transition_suffix"] = deepcopy(transition_suffix)

    if not next_transition or not str(next_transition.get("outline_id") or "").strip():
        turn_entry["error"] = "outline response missing next_transition with outline_id"
        _logger.warning("[MultiTurn] outline incremental: no next_transition")
        return "need_revision", turn_entry

    accepted_prefix = list(session_state.get("accepted_outline_prefix") or [])
    accepted_prefix.append(deepcopy(next_transition))
    session_state["accepted_outline_prefix"] = accepted_prefix
    session_state["outline_lookahead"] = deepcopy(transition_suffix)
    _sync_des_recovery_aliases(session_state, turn_entry=turn_entry)

    # Track symbolic state even in non-validated mode
    _apply_task_effects_to_symbolic_state(next_transition, session_state)

    # Detect outline completion: if no lookahead remaining, the LLM
    # considers this the final recovery task → outline is ready.
    outline_complete = not transition_suffix
    decision = "outline_ready" if outline_complete else "need_next_task"

    _logger.info(
        "[MultiTurn] outline incremental: accepted event %s (prefix now %d events, complete=%s)",
        str(next_transition.get("outline_id") or "").strip(),
        len(accepted_prefix),
        outline_complete,
    )

    session_state["status"] = "paused_after_outline_turn"
    return decision, turn_entry


async def _handle_outline_incremental_validated(
    *,
    session_state: dict[str, Any],
    parsed_response: dict[str, Any],
    prepared_bridge_request: dict[str, Any],
    planner: Any,
) -> tuple[str, dict[str, Any]]:
    """Incremental with validation: one task at a time, validate before accepting."""
    turn_entry: dict[str, Any] = {}

    next_transition = _parsed_response_object(
        parsed_response,
        primary_key="next_transition",
    )
    transition_suffix = _parsed_response_rows(
        parsed_response,
        primary_key="transition_suffix",
    )

    turn_entry["next_transition"] = deepcopy(next_transition)
    turn_entry["transition_suffix"] = deepcopy(transition_suffix)

    if not next_transition or not str(next_transition.get("outline_id") or "").strip():
        turn_entry["error"] = "outline response missing next_transition with outline_id"
        _logger.warning("[MultiTurn] outline incremental_validated: no next_transition")
        return "need_revision", turn_entry

    # Validate before accepting
    findings, grounded_action = _validate_single_outline_task(
        planner=planner,
        task=next_transition,
        session_state=session_state,
        prepared_bridge_request=prepared_bridge_request,
    )
    if grounded_action:
        turn_entry["grounded_action"] = deepcopy(grounded_action)

    if findings:
        turn_entry["validation_findings"] = deepcopy(findings)
        session_state["outline_validation_findings"] = _merge_outline_validation_findings(
            list(session_state.get("outline_validation_findings") or []),
            findings,
        )
        turn_entry["transition_validation"] = {
            "status": "rejected",
            "findings": deepcopy(findings),
        }
        session_state["transition_validation"] = deepcopy(
            turn_entry["transition_validation"]
        )
        _logger.info(
            "[MultiTurn] outline incremental_validated: rejected event %s (%d findings)",
            str(next_transition.get("outline_id") or "").strip(),
            len(findings),
        )
        # Pause so the operator can inspect the validation feedback before
        # the LLM re-proposes on the next turn.
        session_state["status"] = "paused_after_outline_turn"
        return "need_revision", turn_entry

    # Validation passed — accept into prefix
    accepted_prefix = list(session_state.get("accepted_outline_prefix") or [])
    accepted_prefix.append(deepcopy(next_transition))
    session_state["accepted_outline_prefix"] = accepted_prefix
    session_state["outline_lookahead"] = deepcopy(transition_suffix)

    # Apply task effects to symbolic state for future validations
    _apply_task_effects_to_symbolic_state(next_transition, session_state)
    session_state["outline_validation_findings"] = _prune_resolved_outline_validation_findings(
        list(session_state.get("outline_validation_findings") or []),
        session_state=session_state,
        prepared_bridge_request=prepared_bridge_request,
    )
    _sync_des_recovery_aliases(
        session_state,
        turn_entry=turn_entry,
        transition_validation={"status": "passed", "findings": []},
    )

    outline_complete = not transition_suffix
    decision = "outline_ready" if outline_complete else "need_next_task"

    _logger.info(
        "[MultiTurn] outline incremental_validated: accepted event %s "
        "(prefix now %d events, complete=%s)",
        str(next_transition.get("outline_id") or "").strip(),
        len(accepted_prefix),
        outline_complete,
    )

    session_state["status"] = "paused_after_outline_turn"
    return decision, turn_entry


async def _handle_outline_incremental_candidates_validated(
    *,
    session_state: dict[str, Any],
    parsed_response: dict[str, Any],
    prepared_bridge_request: dict[str, Any],
    planner: Any,
) -> tuple[str, dict[str, Any]]:
    """Incremental with multiple candidate next tasks and deterministic selection."""
    turn_entry: dict[str, Any] = {}
    sequence_index = _next_recovery_sequence_index(session_state)
    session_state["outline_validation_findings"] = []
    session_state["pruned_actions"] = _active_pruned_actions(
        session_state,
        prepared_bridge_request,
    )

    candidate_events = [
        _normalize_candidate_task(
            task=dict(row),
            sequence_index=sequence_index,
            candidate_index=candidate_index,
        )
        for candidate_index, row in enumerate(
            _parsed_response_rows(
                parsed_response,
                primary_key="candidate_events",
            )
        )
    ]
    turn_entry["candidate_events"] = deepcopy(candidate_events)

    candidate_bound = int(
        session_state.get("candidate_bound")
        or _DEFAULT_CANDIDATE_BOUND
    )
    if not (1 <= len(candidate_events) <= candidate_bound):
        turn_entry["error"] = (
            "outline response must include 1 to "
            f"{candidate_bound} candidate_events"
        )
        _logger.warning(
            "[MultiTurn] outline incremental_candidates_validated: expected 1-%d candidate_events, got %d",
            candidate_bound,
            len(candidate_events),
        )
        return "need_revision", turn_entry

    candidate_evaluations: list[dict[str, Any]] = []
    valid_candidates: list[dict[str, Any]] = []
    for candidate_index, task in enumerate(candidate_events):
        surface_task = deepcopy(task)
        working_task = deepcopy(task)
        evaluation: dict[str, Any] = {
            "candidate_index": candidate_index,
            "surface_task": deepcopy(surface_task),
            "task": deepcopy(working_task),
        }

        pruned_row = _matching_active_pruned_action(
            task=working_task,
            session_state=session_state,
            prepared_bridge_request=prepared_bridge_request,
        )
        if pruned_row is not None:
            evaluation["valid"] = False
            evaluation["validation_findings"] = [
                _retarget_candidate_finding_to_task(
                    dict(pruned_row.get("guard") or {}),
                    working_task,
                )
            ]
            evaluation["pruned_match"] = True
            candidate_evaluations.append(evaluation)
            continue

        normalized_task, schema_findings = _derive_candidate_outline_task(
            candidate_task=working_task,
            session_state=session_state,
            prepared_bridge_request=prepared_bridge_request,
        )
        if schema_findings:
            evaluation["valid"] = False
            evaluation["validation_findings"] = deepcopy(schema_findings)
            candidate_evaluations.append(evaluation)
            continue
        evaluation["normalized_task"] = deepcopy(normalized_task)

        findings, grounded_action = _validate_single_outline_task(
            planner=planner,
            task=dict(normalized_task or {}),
            session_state=session_state,
            prepared_bridge_request=prepared_bridge_request,
        )
        evaluation["valid"] = not findings
        evaluation["validation_findings"] = deepcopy(findings)
        if grounded_action:
            evaluation["grounded_action"] = deepcopy(grounded_action)
        if findings:
            candidate_evaluations.append(evaluation)
            continue

        progress_score, progress_detail = _candidate_progress_score(
            task=dict(normalized_task or {}),
            session_state=session_state,
            prepared_bridge_request=prepared_bridge_request,
        )
        evaluation["progress_score"] = progress_score
        evaluation["progress_detail"] = deepcopy(progress_detail)
        valid_candidates.append(evaluation)
        candidate_evaluations.append(evaluation)

    progress_candidates = [
        row for row in valid_candidates
        if int(row.get("progress_score") or 0) > 0
    ]

    if not progress_candidates:
        for row in candidate_evaluations:
            if not bool(row.get("valid")):
                continue
            row["valid"] = False
            row["validation_findings"] = [
                _no_blocker_reduction_finding(task=dict(row.get("task") or {}))
            ]
        _promote_durable_candidate_rejections(
            session_state=session_state,
            prepared_bridge_request=prepared_bridge_request,
            candidate_evaluations=candidate_evaluations,
        )
        turn_entry["candidate_evaluations"] = deepcopy(candidate_evaluations)
        feedback_rows = _candidate_feedback_rows(candidate_evaluations)
        session_state["candidate_rejection_feedback"] = deepcopy(feedback_rows)
        turn_entry["candidate_rejection_feedback"] = deepcopy(feedback_rows)
        turn_entry["transition_validation"] = {
            "status": "rejected",
            "findings": deepcopy(feedback_rows),
        }
        session_state["transition_validation"] = deepcopy(
            turn_entry["transition_validation"]
        )
        # Store the LLM's reasoning from this rejected turn for next-turn feedback.
        session_state["rejected_turn_thought"] = str(
            parsed_response.get("thought") or ""
        ).strip()
        _logger.info(
            "[MultiTurn] outline incremental_candidates_validated: rejected all %d candidates",
            len(candidate_events),
        )
        # Keep a stagnation counter for diagnostics, but do not terminate the
        # LLM feedback loop here. Repeated validator feedback is part of the
        # multi-turn recovery contract.
        stagnation = int(session_state.get("outline_stagnation_count") or 0) + 1
        session_state["outline_stagnation_count"] = stagnation
        status_counts: dict[str, int] = {}
        for row in candidate_evaluations:
            if not isinstance(row, dict):
                continue
            findings = [
                dict(f)
                for f in (row.get("validation_findings") or [])
                if isinstance(f, dict)
            ]
            if not findings:
                continue
            status = _finding_event_status_for_logging(findings[0])
            status_counts[status] = int(status_counts.get(status) or 0) + 1
        status_summary = ", ".join(
            f"{status}={count}"
            for status, count in sorted(status_counts.items())
        ) or "none"
        rejection_codes = [
            str(f.get("constraint_code") or "unknown")
            for row in candidate_evaluations
            if isinstance(row, dict)
            for f in (row.get("validation_findings") or [])
            if isinstance(f, dict)
        ]
        _logger.info(
            "[MultiTurn] Stagnation %d — status_counts: %s",
            stagnation, status_summary,
        )
        _logger.debug(
            "[MultiTurn] Stagnation %d — rejection codes: %s",
            stagnation, rejection_codes,
        )
        session_state["status"] = "paused_after_outline_turn"
        return "need_revision", turn_entry
    turn_entry["candidate_evaluations"] = deepcopy(candidate_evaluations)

    # Reset stagnation on successful candidate acceptance.
    session_state["outline_stagnation_count"] = 0

    selected = max(
        progress_candidates,
        key=lambda row: (
            int(dict(row.get("progress_detail") or {}).get("resolved_direct_blockers") or 0),
            int(dict(row.get("progress_detail") or {}).get("blocker_part_acquired") or 0)
            + int(dict(row.get("progress_detail") or {}).get("freed_resource_for_blocker") or 0),
            int(dict(row.get("progress_detail") or {}).get("preparatory_transit") or 0),
            -int(dict(row.get("progress_detail") or {}).get("remaining_blocked_issues") or 0),
            -int(row.get("candidate_index") or 0),
        ),
    )
    selected_candidate_index = int(selected.get("candidate_index") or 0)
    selected_candidate_task = deepcopy(dict(selected.get("task") or {}))
    selected_normalized_task = deepcopy(dict(selected.get("normalized_task") or {}))
    selected_transition = _commit_selected_candidate_task(
        task=selected_normalized_task,
        sequence_index=sequence_index,
    )
    selected_grounded_action = deepcopy(dict(selected.get("grounded_action") or {}))

    turn_entry["selected_candidate_index"] = selected_candidate_index
    turn_entry["selected_transition"] = deepcopy(selected_transition)
    turn_entry["selected_candidate_task"] = deepcopy(selected_candidate_task)
    turn_entry["next_transition"] = deepcopy(selected_transition)
    if selected_grounded_action:
        turn_entry["grounded_action"] = deepcopy(selected_grounded_action)

    accepted_prefix = list(session_state.get("accepted_outline_prefix") or [])
    accepted_prefix.append(deepcopy(selected_transition))
    session_state["accepted_outline_prefix"] = accepted_prefix
    session_state["outline_lookahead"] = []
    session_state["candidate_rejection_feedback"] = []
    session_state["rejected_turn_thought"] = ""
    _sync_des_recovery_aliases(
        session_state,
        turn_entry=turn_entry,
        transition_validation={
            "status": "passed",
            "selected_candidate_index": selected_candidate_index,
        },
    )

    _apply_task_effects_to_symbolic_state(selected_transition, session_state)
    session_state["pruned_actions"] = _active_pruned_actions(
        session_state,
        prepared_bridge_request,
    )

    remaining_findings, remaining_conditions = _remaining_blocked_issue_counts(
        session_state=session_state,
        prepared_bridge_request=prepared_bridge_request,
    )
    outline_complete = remaining_findings == 0 and remaining_conditions == 0

    decision = "outline_ready" if outline_complete else "need_next_task"

    _logger.info(
        "[MultiTurn] outline incremental_candidates_validated: selected candidate %d (%s) "
        "(progress=%d, prefix now %d events, complete=%s)",
        selected_candidate_index + 1,
        str(selected_transition.get("outline_id") or "").strip(),
        int(selected.get("progress_score") or 0),
        len(accepted_prefix),
        outline_complete,
    )

    session_state["status"] = "paused_after_outline_turn"
    return decision, turn_entry


async def _handle_outline_phase(
    *,
    session_state: dict[str, Any],
    parsed_response: dict[str, Any],
    prepared_bridge_request: dict[str, Any],
    planner: Any,
) -> tuple[str, dict[str, Any]]:
    """Dispatch to the appropriate outline handler based on outline_mode."""
    outline_mode = str(session_state.get("outline_mode") or "incremental").strip().lower()

    if outline_mode == "single_pass":
        return await _handle_outline_single_pass(
            session_state=session_state,
            parsed_response=parsed_response,
        )
    if outline_mode == "incremental_validated":
        return await _handle_outline_incremental_validated(
            session_state=session_state,
            parsed_response=parsed_response,
            prepared_bridge_request=prepared_bridge_request,
            planner=planner,
        )
    if outline_mode == "incremental_candidates_validated":
        return await _handle_outline_incremental_candidates_validated(
            session_state=session_state,
            parsed_response=parsed_response,
            prepared_bridge_request=prepared_bridge_request,
            planner=planner,
        )
    return await _handle_outline_incremental(
        session_state=session_state,
        parsed_response=parsed_response,
    )


def _active_primitive_outline_event(
    session_state: dict[str, Any],
) -> tuple[int, dict[str, Any] | None, list[dict[str, Any]]]:
    accepted_prefix = [
        dict(row)
        for row in (session_state.get("accepted_outline_prefix") or [])
        if isinstance(row, dict)
    ]
    cursor = int(session_state.get("primitive_generation_cursor") or 0)
    if cursor < 0:
        cursor = 0
        session_state["primitive_generation_cursor"] = 0
    if cursor >= len(accepted_prefix):
        return cursor, None, accepted_prefix
    return cursor, deepcopy(accepted_prefix[cursor]), accepted_prefix


def _primitive_feedback_row(
    *,
    outline_event: dict[str, Any] | None,
    constraint_code: str,
    reason: str,
    finding: dict[str, Any] | None = None,
) -> dict[str, Any]:
    event = dict(outline_event or {})
    row: dict[str, Any] = {
        "outline_id": str(event.get("outline_id") or "").strip(),
        "resource_jid": str(event.get("resource_jid") or "").strip(),
        "part_name": str(event.get("part_name") or "").strip() or None,
        "constraint_code": str(constraint_code or "").strip() or "primitive_validation_failed",
        "reason": str(reason or "").strip() or "primitive validation failed",
    }
    if isinstance(finding, dict):
        if str(finding.get("constraint_code") or "").strip():
            row["constraint_code"] = str(finding.get("constraint_code") or "").strip()
        if str(finding.get("reason") or "").strip():
            row["reason"] = str(finding.get("reason") or "").strip()
        for key in (
            "constraint_owner",
            "constraint_family",
            "evidence",
            "guard",
            "failed_axes",
            "step_index",
            "primitive",
            "task_id",
        ):
            if key in finding and finding.get(key) not in (None, "", [], {}):
                row[key] = deepcopy(finding.get(key))
    return row


def _primitive_catalog_for_resource(
    prepared_bridge_request: dict[str, Any],
    resource_jid: str,
) -> list[dict[str, Any]]:
    bridge_resources = dict(prepared_bridge_request.get("bridge_resources") or {})
    bridge_entry = dict(bridge_resources.get(resource_jid) or {})
    return filter_synthesis_primitive_catalog(
        [
            dict(row)
            for row in (bridge_entry.get("primitive_catalog") or [])
            if isinstance(row, dict)
        ]
    )


def _primitive_resource_sequence_findings(
    *,
    outline_event: dict[str, Any],
    primitive_steps: list[dict[str, Any]],
    trace_metadata: dict[str, Any],
    start_snapshot: dict[str, Any],
    projected_snapshot: dict[str, Any],
    grounding_context: dict[str, Any],
    primitive_catalog: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    resource_type = str(
        dict(start_snapshot.get("resource_core") or {}).get("resource_type")
        or start_snapshot.get("resource_type")
        or ""
    ).strip()
    profile = get_resource_profile(resource_type or "resource")
    validator = getattr(profile, "primitive_sequence_validator", None)
    if not callable(validator):
        return []
    try:
        raw_findings = validator(
            outline_event=deepcopy(outline_event),
            primitive_steps=deepcopy(primitive_steps),
            trace_metadata=deepcopy(trace_metadata),
            start_snapshot=deepcopy(start_snapshot),
            projected_snapshot=deepcopy(projected_snapshot),
            grounding_context=deepcopy(grounding_context),
            primitive_catalog=deepcopy(primitive_catalog),
        )
    except Exception as exc:
        return [{
            "outline_id": str(outline_event.get("outline_id") or "").strip(),
            "resource_jid": str(outline_event.get("resource_jid") or "").strip(),
            "part_name": str(outline_event.get("part_name") or "").strip() or None,
            "constraint_owner": "resource",
            "constraint_family": "primitive_sequence",
            "constraint_code": "primitive_sequence_validator_error",
            "reason": f"resource primitive sequence validator failed: {exc}",
        }]
    return [
        dict(row)
        for row in (raw_findings or [])
        if isinstance(row, dict)
    ]


def _primitive_start_snapshot(
    *,
    session_state: dict[str, Any],
    prepared_bridge_request: dict[str, Any],
    outline_event: dict[str, Any],
) -> dict[str, Any]:
    resource_jid = str(outline_event.get("resource_jid") or "").strip()
    resources_by_jid, _parts_by_name = _projected_outline_validation_context(
        session_state=session_state,
        prepared_bridge_request=prepared_bridge_request,
    )
    bridge_entry = dict(
        dict(prepared_bridge_request.get("bridge_resources") or {}).get(resource_jid)
        or {}
    )
    bridge_snapshot = dict(bridge_entry.get("bridge_snapshot") or {})
    snapshot = deepcopy(dict(resources_by_jid.get(resource_jid) or {}))
    snapshot.setdefault("resource_jid", resource_jid)
    for key in (
        "resource_type",
        "current_pose",
        "current_pose_ref",
        "workspace_bounds",
        "named_poses",
        "available_named_poses",
        "reachability",
        "staging_areas",
    ):
        if snapshot.get(key) in (None, "", [], {}):
            value = bridge_entry.get(key)
            if value in (None, "", [], {}):
                value = bridge_snapshot.get(key)
            if value not in (None, "", [], {}):
                snapshot[key] = deepcopy(value)

    expected_start = dict(outline_event.get("expected_start_state") or {})
    field_map = {
        "resource_state": "current_state",
        "resource_location": "current_location",
        "held_part": "held_part",
    }
    for expected_key, snapshot_key in field_map.items():
        if expected_key in expected_start:
            snapshot[snapshot_key] = deepcopy(expected_start.get(expected_key))
    if "held_part" in expected_start:
        snapshot["gripper_state"] = (
            "closed" if expected_start.get("held_part") not in (None, "") else "open"
        )
    return snapshot


def _primitive_grounding_context(
    *,
    session_state: dict[str, Any],
    prepared_bridge_request: dict[str, Any],
    outline_event: dict[str, Any],
) -> dict[str, Any]:
    resources_by_jid, parts_by_name = _projected_outline_validation_context(
        session_state=session_state,
        prepared_bridge_request=prepared_bridge_request,
    )
    resource_jid = str(outline_event.get("resource_jid") or "").strip()
    part_name = str(outline_event.get("part_name") or "").strip()
    return {
        "active_outline_event": deepcopy(outline_event),
        "resource": deepcopy(resources_by_jid.get(resource_jid) or {}),
        "part": deepcopy(parts_by_name.get(part_name) or {}) if part_name else {},
        "resources_by_jid": deepcopy(resources_by_jid),
        "parts_by_name": deepcopy(parts_by_name),
        "observation_store": deepcopy(session_state.get("observation_store") or {}),
    }


async def _handle_primitive_generation_phase(
    *,
    session_state: dict[str, Any],
    parsed_response: dict[str, Any],
    prepared_bridge_request: dict[str, Any],
    planner: Any,
) -> tuple[str, dict[str, Any]]:
    """Handle one batched turn of the primitive generation phase.

    Expects ``parsed_response.primitive_events`` to cover every remaining
    accepted outline event in order. Validates each event sequentially with
    chained projected snapshots.

    Returns (decision, turn_entry).
    """
    cursor, active_event, accepted_prefix = _active_primitive_outline_event(session_state)
    remaining_events = accepted_prefix[cursor:] if cursor < len(accepted_prefix) else []
    turn_entry: dict[str, Any] = {
        "primitive_generation_cursor": cursor,
        "active_outline_event": deepcopy(active_event),
        "active_recovery_event": deepcopy(active_event),
        "accepted_transition_prefix": deepcopy(accepted_prefix),
        "des_event_sequence": deepcopy(accepted_prefix),
        "remaining_outline_events": deepcopy(remaining_events),
    }
    if active_event is None:
        if accepted_prefix:
            session_state["primitive_rejection_feedback"] = []
            session_state["status"] = "paused_after_primitive_generation"
            return "draft_ready", turn_entry
        feedback = [
            _primitive_feedback_row(
                outline_event={},
                constraint_code="outline_prefix_missing",
                reason="primitive generation requires an accepted outline prefix",
            )
        ]
        session_state["primitive_rejection_feedback"] = deepcopy(feedback)
        turn_entry["primitive_rejection_feedback"] = deepcopy(feedback)
        session_state["status"] = "paused_after_primitive_turn"
        return "primitive_blocked", turn_entry

    response_decision = str(parsed_response.get("decision") or "").strip()
    response_events = [
        dict(row)
        for row in (parsed_response.get("primitive_events") or [])
        if isinstance(row, dict)
    ]
    turn_entry["primitive_events"] = deepcopy(response_events)

    if response_decision in {"primitive_blocked", "need_outline_revision"}:
        feedback = [
            _primitive_feedback_row(
                outline_event=active_event,
                constraint_code="primitive_blocked",
                reason="LLM reported that one or more outline events remain blocked for primitive authoring",
            )
        ]
        session_state["primitive_rejection_feedback"] = deepcopy(feedback)
        turn_entry["primitive_rejection_feedback"] = deepcopy(feedback)
        session_state["status"] = "paused_after_primitive_blocked"
        return "primitive_blocked", turn_entry

    if response_decision not in {"primitive_steps_ready", "need_primitive_revision"}:
        feedback = [
            _primitive_feedback_row(
                outline_event=active_event,
                constraint_code="primitive_schema_violation",
                reason="decision must be primitive_steps_ready, need_primitive_revision, or primitive_blocked",
            )
        ]
        session_state["primitive_rejection_feedback"] = deepcopy(feedback)
        turn_entry["primitive_rejection_feedback"] = deepcopy(feedback)
        session_state["status"] = "paused_after_primitive_turn"
        return "need_primitive_revision", turn_entry

    if response_decision == "need_primitive_revision":
        feedback = [
            _primitive_feedback_row(
                outline_event=active_event,
                constraint_code="primitive_revision_requested",
                reason=(
                    "LLM marked the primitive batch as needing revision; return "
                    "decision primitive_steps_ready only when the batch is ready "
                    "for validation"
                ),
            )
        ]
        session_state["primitive_rejection_feedback"] = deepcopy(feedback)
        turn_entry["primitive_rejection_feedback"] = deepcopy(feedback)
        session_state["status"] = "paused_after_primitive_turn"
        return "need_primitive_revision", turn_entry

    if len(response_events) != len(remaining_events):
        feedback = [
            _primitive_feedback_row(
                outline_event=active_event,
                constraint_code="primitive_schema_violation",
                reason=(
                    f"primitive_events must contain exactly {len(remaining_events)} entries "
                    f"(one per remaining outline event); got {len(response_events)}"
                ),
            )
        ]
        session_state["primitive_rejection_feedback"] = deepcopy(feedback)
        turn_entry["primitive_rejection_feedback"] = deepcopy(feedback)
        session_state["status"] = "paused_after_primitive_turn"
        return "need_primitive_revision", turn_entry

    per_event_results: list[dict[str, Any]] = []
    accepted_rows_to_append: list[dict[str, Any]] = []
    prev_projected_snapshot: dict[str, Any] | None = None
    prev_resource_jid: str | None = None

    for idx, (expected_event, response_event) in enumerate(
        zip(remaining_events, response_events)
    ):
        outline_id = str(expected_event.get("outline_id") or "").strip()
        resource_jid = str(expected_event.get("resource_jid") or "").strip()
        response_outline_id = str(response_event.get("outline_id") or "").strip()
        response_resource_jid = str(response_event.get("resource_jid") or "").strip()
        primitive_steps = [
            dict(row)
            for row in (response_event.get("primitive_steps") or [])
            if isinstance(row, dict)
        ]

        schema_errors: list[str] = []
        if response_outline_id != outline_id:
            schema_errors.append(
                f"primitive_events[{idx}].outline_id must match {outline_id!r}"
            )
        if response_resource_jid != resource_jid:
            schema_errors.append(
                f"primitive_events[{idx}].resource_jid must match {resource_jid!r}"
            )
        if not primitive_steps:
            schema_errors.append(
                f"primitive_events[{idx}].primitive_steps must contain at least one primitive step"
            )
        for step_index, step in enumerate(primitive_steps):
            if "primitive" not in step or not str(step.get("primitive") or "").strip():
                schema_errors.append(
                    f"primitive_events[{idx}].primitive_steps[{step_index}] must include primitive"
                )
            if "params" not in step or not isinstance(step.get("params"), dict):
                schema_errors.append(
                    f"primitive_events[{idx}].primitive_steps[{step_index}] must include params object"
                )
        if schema_errors:
            feedback = [
                _primitive_feedback_row(
                    outline_event=expected_event,
                    constraint_code="primitive_schema_violation",
                    reason="; ".join(schema_errors),
                )
            ]
            session_state["primitive_rejection_feedback"] = deepcopy(feedback)
            turn_entry["primitive_rejection_feedback"] = deepcopy(feedback)
            turn_entry["per_event_results"] = per_event_results
            session_state["status"] = "paused_after_primitive_turn"
            return "need_primitive_revision", turn_entry

        if prev_projected_snapshot is not None and prev_resource_jid == resource_jid:
            start_snapshot = deepcopy(prev_projected_snapshot)
            expected_start = dict(expected_event.get("expected_start_state") or {})
            field_map = {
                "resource_state": "current_state",
                "resource_location": "current_location",
                "held_part": "held_part",
            }
            for expected_key, snapshot_key in field_map.items():
                if expected_key in expected_start:
                    start_snapshot[snapshot_key] = deepcopy(expected_start.get(expected_key))
            if "held_part" in expected_start:
                start_snapshot["gripper_state"] = (
                    "closed"
                    if expected_start.get("held_part") not in (None, "")
                    else "open"
                )
        else:
            start_snapshot = _primitive_start_snapshot(
                session_state=session_state,
                prepared_bridge_request=prepared_bridge_request,
                outline_event=expected_event,
            )

        primitive_catalog = _primitive_catalog_for_resource(
            prepared_bridge_request,
            resource_jid,
        )
        grounding_context = _primitive_grounding_context(
            session_state=session_state,
            prepared_bridge_request=prepared_bridge_request,
            outline_event=expected_event,
        )
        trace_result = validate_and_project_steps_with_trace(
            primitive_steps,
            primitive_catalog,
            start_snapshot,
            grounding_context=grounding_context,
        )
        valid = bool(trace_result.get("valid"))
        projected_snapshot = dict(trace_result.get("projected_snapshot") or {})
        validation_error = trace_result.get("validation_error")
        per_event_results.append({
            "outline_id": outline_id,
            "resource_jid": resource_jid,
            "start_snapshot": deepcopy(start_snapshot),
            "projected_snapshot": deepcopy(projected_snapshot),
            "valid": bool(valid),
            "validation_error": validation_error,
            "trace_step_count": len(trace_result.get("step_results") or []),
        })
        if not valid:
            feedback = [
                _primitive_feedback_row(
                    outline_event=expected_event,
                    constraint_code="primitive_validation_failed",
                    reason=validation_error or "primitive sequence failed validation",
                )
            ]
            session_state["primitive_rejection_feedback"] = deepcopy(feedback)
            turn_entry["primitive_rejection_feedback"] = deepcopy(feedback)
            turn_entry["per_event_results"] = per_event_results
            session_state["status"] = "paused_after_primitive_turn"
            return "need_primitive_revision", turn_entry

        resource_findings = _primitive_resource_sequence_findings(
            outline_event=expected_event,
            primitive_steps=primitive_steps,
            trace_metadata=trace_result,
            start_snapshot=start_snapshot,
            projected_snapshot=projected_snapshot,
            grounding_context=grounding_context,
            primitive_catalog=primitive_catalog,
        )
        if resource_findings:
            feedback = [
                _primitive_feedback_row(
                    outline_event=expected_event,
                    constraint_code=str(row.get("constraint_code") or "primitive_sequence_invalid"),
                    reason=str(row.get("reason") or "resource primitive sequence validation failed"),
                    finding=row,
                )
                for row in resource_findings
            ]
            session_state["primitive_rejection_feedback"] = deepcopy(feedback)
            turn_entry["primitive_rejection_feedback"] = deepcopy(feedback)
            turn_entry["per_event_results"] = per_event_results
            per_event_results[-1]["resource_validation_findings"] = deepcopy(resource_findings)
            session_state["status"] = "paused_after_primitive_turn"
            return "need_primitive_revision", turn_entry

        compatible, compatibility_error = _primitive_projected_snapshot_matches_event(
            projected_snapshot,
            expected_event,
        )
        if not compatible:
            feedback = [
                _primitive_feedback_row(
                    outline_event=expected_event,
                    constraint_code="primitive_projection_mismatch",
                    reason=compatibility_error
                    or "primitive projection does not match outline event",
                )
            ]
            session_state["primitive_rejection_feedback"] = deepcopy(feedback)
            turn_entry["primitive_rejection_feedback"] = deepcopy(feedback)
            turn_entry["per_event_results"] = per_event_results
            session_state["status"] = "paused_after_primitive_turn"
            return "need_primitive_revision", turn_entry

        accepted_rows_to_append.append({
            "outline_id": outline_id,
            "des_event_id": outline_id,
            "resource_jid": resource_jid,
            "part_name": str(expected_event.get("part_name") or "").strip() or None,
            "event_name": str(expected_event.get("event_name") or "").strip(),
            "description": str(expected_event.get("description") or "").strip(),
            "primitive_steps": deepcopy(primitive_steps),
            "projected_snapshot": deepcopy(projected_snapshot),
        })
        prev_projected_snapshot = projected_snapshot
        prev_resource_jid = resource_jid

    accepted_program = list(session_state.get("accepted_primitive_program") or [])
    accepted_program.extend(deepcopy(accepted_rows_to_append))
    session_state["accepted_primitive_program"] = deepcopy(accepted_program)
    session_state["primitive_rejection_feedback"] = []
    session_state["primitive_generation_cursor"] = cursor + len(remaining_events)
    turn_entry["per_event_results"] = per_event_results
    turn_entry["accepted_primitive_macros"] = deepcopy(accepted_rows_to_append)
    session_state["status"] = "paused_after_primitive_generation"
    return "draft_ready", turn_entry


async def _handle_finalize_phase(
    *,
    session_state: dict[str, Any],
    parsed_response: dict[str, Any],
    prepared_bridge_request: dict[str, Any],
    planner: Any,
) -> tuple[str, dict[str, Any]]:
    """Handle one turn of the finalize phase.

    Returns (decision, turn_entry).
    """
    raise NotImplementedError("finalize phase handler not yet implemented")


from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.modes.multi_turn_outline_generation import (
    _projected_outline_validation_context,
    _handle_outline_single_pass,
    _handle_outline_incremental,
    _handle_outline_incremental_validated,
    _handle_outline_incremental_candidates_validated,
    _handle_outline_phase,
)
from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.modes.multi_turn_primitive_generation import (
    _active_primitive_outline_event,
    _primitive_feedback_row,
    _primitive_catalog_for_resource,
    _primitive_resource_sequence_findings,
    _primitive_start_snapshot,
    _primitive_grounding_context,
    _primitive_projected_snapshot_matches_event,
    build_primitive_generation_prompt_context,
    _handle_primitive_generation_phase,
)


_PHASE_HANDLERS = {
    "grounding": _handle_grounding_phase,
    "outline": _handle_outline_phase,
    "primitive_generation": _handle_primitive_generation_phase,
    "finalize": _handle_finalize_phase,
}

# ---------------------------------------------------------------------------
# Prompt / schema delegation
# ---------------------------------------------------------------------------


def _build_phase_prompt(
    prepared_bridge_request: dict[str, Any],
    session_state: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    """Build prompt input and rendered prompt text for the current phase."""
    from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.modes.multi_turn_prompts import (
        build_multi_turn_phase_prompt_input,
        render_multi_turn_phase_prompt,
    )

    phase = str(session_state.get("current_phase") or "grounding").strip().lower()
    llm_input = dict(prepared_bridge_request.get("llm_input") or {})
    outline_mode = str(session_state.get("outline_mode") or "incremental").strip().lower()

    world_observation_surface: dict[str, Any] | None = None
    if phase == "grounding":
        world_observation_surface = _build_world_observation_surface(prepared_bridge_request)

    current_recovery_blockers: list[dict[str, Any]] | None = None
    if phase == "outline" and outline_mode == "incremental_candidates_validated":
        session_state["pruned_actions"] = _active_pruned_actions(
            session_state,
            prepared_bridge_request,
        )
        current_recovery_blockers = _active_candidate_recovery_blockers(
            session_state=session_state,
            prepared_bridge_request=prepared_bridge_request,
        )
        # Compute the configured candidate budget for this turn.
        candidate_bound = _compute_enabled_candidate_bound(
            session_state, prepared_bridge_request,
        )
        session_state["candidate_bound"] = candidate_bound

    prompt_input = build_multi_turn_phase_prompt_input(
        phase=phase,
        llm_input=llm_input,
        session_state=session_state,
        bridge_resources=dict(prepared_bridge_request.get("bridge_resources") or {}),
        world_observation_surface=world_observation_surface,
        current_recovery_blockers=current_recovery_blockers,
    )
    if phase == "primitive_generation":
        prompt_input.update(
            build_primitive_generation_prompt_context(
                session_state=session_state,
                prepared_bridge_request=prepared_bridge_request,
            )
        )
    prompt_text = render_multi_turn_phase_prompt(prompt_input)
    return prompt_input, prompt_text


def _get_response_schema(phase: str, session_state: dict[str, Any]) -> dict[str, Any]:
    """Return the JSON response schema for the given phase."""
    from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.modes.multi_turn_prompts import (
        multi_turn_phase_response_schema,
    )
    outline_mode = str(session_state.get("outline_mode") or "incremental").strip().lower()
    candidate_bound = None
    if phase.strip().lower() == "outline" and outline_mode == "incremental_candidates_validated":
        candidate_bound = max(
            1,
            int(
                session_state.get("candidate_bound")
                or _DEFAULT_CANDIDATE_BOUND
            ),
        )
    return multi_turn_phase_response_schema(
        phase,
        outline_mode=outline_mode,
        candidate_bound=candidate_bound,
    )


# ---------------------------------------------------------------------------
# Response artifact shaping
# ---------------------------------------------------------------------------

_ARTIFACT_TASK_KEYS = (
    "outline_id",
    "candidate_outline_id",
    "llm_outline_id",
    "resource_jid",
    "event_name",
    "description",
    "part_name",
    "target_ref",
    "action_target",
    "expected_start_state",
    "expected_end_state",
)


def _compact_artifact_task(task: Any) -> dict[str, Any]:
    if not isinstance(task, dict):
        return {}
    compact: dict[str, Any] = {}
    for key in _ARTIFACT_TASK_KEYS:
        value = task.get(key)
        if value in (None, "", [], {}):
            continue
        compact[key] = deepcopy(value)
    return compact


def _compact_artifact_finding(finding: Any) -> dict[str, Any]:
    if not isinstance(finding, dict):
        return {}
    compact: dict[str, Any] = {}
    for key in (
        "constraint_code",
        "constraint_family",
        "constraint_owner",
        "resource_jid",
        "part_name",
        "task_id",
        "reason",
    ):
        value = finding.get(key)
        if value in (None, "", [], {}):
            continue
        compact[key] = deepcopy(value)
    return compact


def _compact_artifact_feedback_rows(rows: Any) -> list[dict[str, Any]]:
    compact_rows: list[dict[str, Any]] = []
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        if "constraint_code" in row and not (
            row.get("validation_findings") or row.get("findings")
        ):
            compact_finding = _compact_artifact_finding(row)
            if compact_finding:
                compact_rows.append(compact_finding)
            continue
        compact_row: dict[str, Any] = {
            "candidate_index": int(row.get("candidate_index") or 0),
        }
        task = row.get("task")
        if not isinstance(task, dict):
            task = row.get("surface_task")
        compact_task = _compact_artifact_task(task)
        if compact_task:
            compact_row["task"] = compact_task
        findings = [
            compact_finding
            for compact_finding in (
                _compact_artifact_finding(item)
                for item in (row.get("validation_findings") or [])
            )
            if compact_finding
        ]
        if findings:
            compact_row["findings"] = findings
            compact_row["constraint_codes"] = sorted({
                str(item.get("constraint_code") or "")
                for item in findings
                if str(item.get("constraint_code") or "").strip()
            })
        compact_rows.append(compact_row)
    return compact_rows


def _compact_artifact_candidate_evaluations(rows: Any) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        task = row.get("normalized_task")
        if not isinstance(task, dict):
            task = row.get("task")
        if not isinstance(task, dict):
            task = row.get("surface_task")
        summary: dict[str, Any] = {
            "candidate_index": int(row.get("candidate_index") or 0),
            "valid": bool(row.get("valid")),
        }
        if row.get("pruned_match") is not None:
            summary["pruned_match"] = bool(row.get("pruned_match"))
        compact_task = _compact_artifact_task(task)
        if compact_task:
            summary["task"] = compact_task
        findings = [
            compact_finding
            for compact_finding in (
                _compact_artifact_finding(item)
                for item in (row.get("validation_findings") or [])
            )
            if compact_finding
        ]
        if findings:
            summary["findings"] = findings
            summary["constraint_codes"] = sorted({
                str(item.get("constraint_code") or "")
                for item in findings
                if str(item.get("constraint_code") or "").strip()
            })
        progress_detail = row.get("progress_detail")
        if isinstance(progress_detail, dict) and progress_detail:
            summary["progress_detail"] = deepcopy(progress_detail)
        summaries.append(summary)
    return summaries


def _compact_artifact_transition_validation(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    compact: dict[str, Any] = {}
    for key in ("status", "selected_candidate_index"):
        if value.get(key) not in (None, "", [], {}):
            compact[key] = deepcopy(value.get(key))
    findings = value.get("findings")
    if isinstance(findings, list) and findings:
        compact_findings = _compact_artifact_feedback_rows(findings)
        if compact_findings:
            compact["findings"] = compact_findings
    return compact


def _compact_final_output_artifact_payload(payload: dict[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for key in (
        "engine",
        "decision",
        "final_output_stage",
        "status",
        "current_phase",
        "accepted_trace_length",
        "primitive_program_complete",
    ):
        if payload.get(key) not in (None, "", [], {}):
            compact[key] = deepcopy(payload.get(key))
    transition_trace = payload.get("transition_trace")
    if not isinstance(transition_trace, list) or not transition_trace:
        transition_trace = payload.get("accepted_transition_prefix")
    if isinstance(transition_trace, list) and transition_trace:
        compact["transition_trace"] = [
            _compact_artifact_task(row)
            for row in transition_trace
            if isinstance(row, dict)
        ]
    accepted_program = payload.get("accepted_primitive_program")
    if isinstance(accepted_program, list) and accepted_program:
        compact["accepted_primitive_program"] = deepcopy(accepted_program)
    executable_trace = payload.get("executable_recovery_trace")
    if isinstance(executable_trace, list) and executable_trace:
        compact["executable_recovery_trace"] = deepcopy(executable_trace)
    return compact


def _normalized_response_artifact_payload(
    *,
    phase: str,
    session_state: dict[str, Any],
    parsed_response: dict[str, Any],
    turn_entry: dict[str, Any],
) -> dict[str, Any]:
    """Return the response payload to persist in per-turn artifacts."""
    normalized = deepcopy(parsed_response if isinstance(parsed_response, dict) else {})
    if str(turn_entry.get("decision") or "").strip():
        normalized.setdefault("decision", str(turn_entry.get("decision") or "").strip())
    if str(phase or "").strip().lower() != "outline":
        return normalized

    accepted_prefix = [
        deepcopy(row)
        for row in (session_state.get("accepted_outline_prefix") or [])
        if isinstance(row, dict)
    ]
    if accepted_prefix:
        normalized.setdefault("transition_trace", deepcopy(accepted_prefix))
    next_transition = turn_entry.get("next_transition")
    if isinstance(next_transition, dict):
        normalized.setdefault("next_transition", deepcopy(next_transition))
    if isinstance(turn_entry.get("transition_suffix"), list):
        normalized.setdefault(
            "transition_suffix",
            deepcopy(turn_entry.get("transition_suffix") or []),
        )
    if isinstance(turn_entry.get("transition_validation"), dict):
        normalized.setdefault(
            "transition_validation",
            deepcopy(turn_entry.get("transition_validation") or {}),
        )

    outline_mode = str(session_state.get("outline_mode") or "incremental").strip().lower()
    if outline_mode != "incremental_candidates_validated":
        return normalized

    payload: dict[str, Any] = {}
    thought = str(parsed_response.get("thought") or "").strip()
    if thought:
        payload["thought"] = thought
    if str(turn_entry.get("decision") or "").strip():
        payload["decision"] = str(turn_entry.get("decision") or "").strip()
    payload["candidate_events"] = [
        _compact_artifact_task(row)
        for row in (turn_entry.get("candidate_events") or [])
        if isinstance(row, dict)
    ]
    if isinstance(turn_entry.get("candidate_evaluations"), list):
        payload["candidate_evaluation_summary"] = (
            _compact_artifact_candidate_evaluations(
                turn_entry.get("candidate_evaluations") or []
            )
        )
    if "selected_candidate_index" in turn_entry:
        payload["selected_candidate_index"] = int(
            turn_entry.get("selected_candidate_index") or 0
        )
    if isinstance(turn_entry.get("selected_transition"), dict):
        payload["selected_transition"] = _compact_artifact_task(
            turn_entry.get("selected_transition")
        )
    if isinstance(turn_entry.get("selected_candidate_task"), dict):
        payload["selected_candidate_task"] = _compact_artifact_task(
            turn_entry.get("selected_candidate_task")
        )
    if isinstance(turn_entry.get("candidate_rejection_feedback"), list):
        payload["candidate_rejection_feedback"] = _compact_artifact_feedback_rows(
            turn_entry.get("candidate_rejection_feedback") or []
        )
    if accepted_prefix:
        payload["transition_trace"] = deepcopy(accepted_prefix)
    if isinstance(turn_entry.get("transition_validation"), dict):
        payload["transition_validation"] = _compact_artifact_transition_validation(
            turn_entry.get("transition_validation") or {}
        )
    return payload


def _final_output_transition_trace(payload: dict[str, Any]) -> list[dict[str, Any]]:
    transition_trace: Any = payload.get("transition_trace")
    if not isinstance(transition_trace, list) or not transition_trace:
        transition_trace = payload.get("accepted_transition_prefix")
    if not isinstance(transition_trace, list) or not transition_trace:
        transition_trace = payload.get("executable_recovery_trace")
    if not isinstance(transition_trace, list) or not transition_trace:
        transition_trace = payload.get("outline_tasks")
    if not isinstance(transition_trace, list):
        transition_trace = []
    return [
        deepcopy(row)
        for row in transition_trace
        if isinstance(row, dict)
    ]


def _resolve_multi_turn_primary_obligation(
    *,
    prepared_bridge_request: dict[str, Any],
    resource_jids: list[str],
) -> tuple[dict[str, Any] | None, str]:
    obligation_targets = [
        deepcopy(row)
        for row in (prepared_bridge_request.get("obligation_targets") or [])
        if isinstance(row, dict)
    ]
    if not obligation_targets:
        return None, ""
    if len(obligation_targets) == 1:
        return obligation_targets[0], ""

    distinct_resources = {
        str(resource_jid or "").strip()
        for resource_jid in resource_jids
        if str(resource_jid or "").strip()
    }
    if len(distinct_resources) == 1:
        resource_jid = next(iter(distinct_resources))
        matches = [
            target
            for target in obligation_targets
            if str(target.get("resource_jid") or "").strip() == resource_jid
        ]
        if len(matches) == 1:
            return matches[0], ""
        if len(matches) > 1:
            return None, (
                f"multiple obligation_targets matched resource_jid {resource_jid!r}; "
                "cannot infer primary_obligation"
            )

    return None, (
        "multiple active obligation_targets are present; cannot infer primary_obligation "
        "from multi-turn final output"
    )


def _target_location_for_part(
    *,
    part_name: str,
    prepared_bridge_request: dict[str, Any],
) -> str:
    grounding_context = dict(prepared_bridge_request.get("grounding_context") or {})
    parts = dict(grounding_context.get("parts") or {})
    part_entry = dict(parts.get(part_name) or {})
    target = dict(part_entry.get("target") or {})
    return str(target.get("location") or "").strip()


def _multi_turn_bridge_task_context(
    *,
    transition_event: dict[str, Any],
    primitive_row: dict[str, Any],
    prepared_bridge_request: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    expected_start = dict(transition_event.get("expected_start_state") or {})
    expected_end = dict(transition_event.get("expected_end_state") or {})
    action_target = dict(transition_event.get("action_target") or {})
    part_name = str(
        primitive_row.get("part_name")
        or transition_event.get("part_name")
        or ""
    ).strip()
    target_location = str(
        transition_event.get("target_ref")
        or action_target.get("target_location")
        or expected_end.get("part_location")
        or ""
    ).strip()
    origin_location = str(
        action_target.get("source_location")
        or expected_start.get("part_location")
        or ""
    ).strip()

    task_params: dict[str, Any] = {}
    if target_location:
        task_params["destination_location"] = target_location
    if origin_location and not origin_location.endswith("_gripper"):
        task_params["origin_resource_location"] = origin_location

    task_metadata: dict[str, Any] = {
        "in_state": str(expected_start.get("resource_state") or "").strip(),
        "out_state": str(expected_end.get("resource_state") or "").strip(),
        "required_context_keys": [],
        "context_mapping": {},
        "part_transition": None,
    }

    if not part_name:
        return task_params, task_metadata

    transition: dict[str, Any] = {}
    end_held = str(expected_end.get("held_part") or "").strip()
    end_part_state = str(expected_end.get("part_state") or "").strip()
    product_target_location = _target_location_for_part(
        part_name=part_name,
        prepared_bridge_request=prepared_bridge_request,
    )

    if end_held and end_held == part_name:
        transition = {
            "state": end_part_state or "in_gripper",
            "location_template": "{resource_jid}_gripper",
        }
    elif target_location:
        task_metadata["required_context_keys"] = ["destination_location"]
        task_metadata["context_mapping"] = {
            "location_param": "destination_location",
        }
        transition_state = end_part_state or "ready"
        if product_target_location and target_location == product_target_location:
            transition_state = "assembled"
        transition = {
            "state": transition_state,
            "location_param": "destination_location",
        }
    elif str(expected_end.get("part_location") or "").strip():
        task_metadata["required_context_keys"] = ["destination_location"]
        task_metadata["context_mapping"] = {
            "location_param": "destination_location",
        }
        task_params.setdefault(
            "destination_location",
            str(expected_end.get("part_location") or "").strip(),
        )
        transition = {
            "state": end_part_state or "ready",
            "location_param": "destination_location",
        }

    if transition:
        task_metadata["part_transition"] = {"completed": transition}

    return task_params, task_metadata


def _primitive_catalog_for_bridge_normalizer(
    raw_catalog: Any,
) -> list[dict[str, Any]]:
    normalized_catalog: list[dict[str, Any]] = []
    for raw_entry in raw_catalog or []:
        if not isinstance(raw_entry, dict):
            continue
        entry = deepcopy(raw_entry)
        parameters = entry.get("parameters")
        if not isinstance(parameters, dict):
            properties = {}
            for raw_name, raw_schema in dict(entry.get("params") or {}).items():
                param_name = str(raw_name or "").strip()
                if not param_name:
                    continue
                if isinstance(raw_schema, dict):
                    properties[param_name] = deepcopy(raw_schema)
                else:
                    properties[param_name] = {"type": "string"}
            entry["parameters"] = {
                "type": "object",
                "properties": properties,
                "required": [
                    str(name).strip()
                    for name in (entry.get("required_params") or [])
                    if str(name).strip()
                ],
            }
        normalized_catalog.append(entry)
    return normalized_catalog


def _bridge_resources_for_bridge_normalizer(
    raw_resources: Any,
) -> dict[str, dict[str, Any]]:
    normalized_resources: dict[str, dict[str, Any]] = {}
    for raw_jid, raw_entry in dict(raw_resources or {}).items():
        if not isinstance(raw_entry, dict):
            continue
        resource_jid = str(raw_jid or raw_entry.get("resource_jid") or "").strip()
        if not resource_jid:
            continue
        entry = deepcopy(raw_entry)
        entry["primitive_catalog"] = _primitive_catalog_for_bridge_normalizer(
            entry.get("primitive_catalog") or []
        )
        entry["execution_primitive_catalog"] = _primitive_catalog_for_bridge_normalizer(
            entry.get("execution_primitive_catalog")
            or entry.get("primitive_catalog")
            or []
        )
        normalized_resources[resource_jid] = entry
    return normalized_resources


def normalize_multi_turn_final_output_to_bridge_proposal(
    *,
    final_output_payload: dict[str, Any],
    prepared_bridge_request: dict[str, Any],
) -> dict[str, Any]:
    """Convert multi-turn final output into an executable primitive bridge proposal."""
    result: dict[str, Any] = {
        "accepted": False,
        "reason": "",
        "raw_proposal": None,
        "normalized_proposal": None,
        "final_output_stage": str(final_output_payload.get("final_output_stage") or "").strip(),
    }
    if not isinstance(final_output_payload, dict) or not final_output_payload:
        result["reason"] = "final_output_payload is missing"
        return result
    if not isinstance(prepared_bridge_request, dict) or not prepared_bridge_request:
        result["reason"] = "prepared_bridge_request is missing"
        return result

    transition_trace = _final_output_transition_trace(final_output_payload)
    accepted_program = [
        deepcopy(row)
        for row in (final_output_payload.get("accepted_primitive_program") or [])
        if isinstance(row, dict)
    ]
    if not transition_trace:
        result["reason"] = "final output did not include a transition trace"
        return result
    if not accepted_program:
        result["reason"] = "final output did not include accepted_primitive_program"
        return result

    primitives_by_outline_id = {
        str(row.get("outline_id") or "").strip(): deepcopy(row)
        for row in accepted_program
        if str(row.get("outline_id") or "").strip()
    }
    macro_tasks: list[dict[str, Any]] = []
    resource_jids: list[str] = []
    for transition_event in transition_trace:
        outline_id = str(transition_event.get("outline_id") or "").strip()
        primitive_row = dict(primitives_by_outline_id.get(outline_id) or {})
        if not primitive_row:
            result["reason"] = (
                f"accepted_primitive_program is missing an entry for outline_id {outline_id!r}"
            )
            return result
        primitive_steps = [
            deepcopy(step)
            for step in (primitive_row.get("primitive_steps") or [])
            if isinstance(step, dict)
        ]
        if not primitive_steps:
            result["reason"] = (
                f"accepted_primitive_program[{outline_id}] has no primitive_steps"
            )
            return result

        resource_jid = str(
            primitive_row.get("resource_jid")
            or transition_event.get("resource_jid")
            or ""
        ).strip()
        if not resource_jid:
            result["reason"] = (
                f"transition trace entry {outline_id!r} is missing resource_jid"
            )
            return result
        resource_jids.append(resource_jid)

        task_params, task_metadata = _multi_turn_bridge_task_context(
            transition_event=transition_event,
            primitive_row=primitive_row,
            prepared_bridge_request=prepared_bridge_request,
        )
        macro_tasks.append(
            {
                "resource_jid": resource_jid,
                "macro_name": str(
                    primitive_row.get("event_name")
                    or transition_event.get("action_name")
                    or transition_event.get("event_name")
                    or outline_id
                ).strip(),
                "description": str(
                    primitive_row.get("description")
                    or transition_event.get("description")
                    or ""
                ).strip(),
                "expected_start_state": str(
                    dict(transition_event.get("expected_start_state") or {}).get("resource_state")
                    or ""
                ).strip(),
                "part_name": str(
                    primitive_row.get("part_name")
                    or transition_event.get("part_name")
                    or ""
                ).strip(),
                "task_params": task_params,
                "task_metadata": task_metadata,
                "primitive_steps": primitive_steps,
            }
        )

    primary_obligation, obligation_error = _resolve_multi_turn_primary_obligation(
        prepared_bridge_request=prepared_bridge_request,
        resource_jids=resource_jids,
    )
    if obligation_error:
        result["reason"] = obligation_error
        return result

    raw_proposal: dict[str, Any] = {
        "description": "Executable bridge proposal derived from multi-turn final_output.",
        "macro_tasks": macro_tasks,
    }
    if primary_obligation:
        raw_proposal["primary_obligation"] = deepcopy(primary_obligation)
    result["raw_proposal"] = deepcopy(raw_proposal)

    normalized = normalize_primitive_bridge_proposal(
        raw=json.dumps(raw_proposal, default=str),
        ra_jid=str(prepared_bridge_request.get("ra_jid", "") or "").strip(),
        primitive_catalog=_primitive_catalog_for_bridge_normalizer(
            list(prepared_bridge_request.get("primitive_catalog") or [])
        ),
        bridge_snapshot=deepcopy(prepared_bridge_request.get("bridge_snapshot") or {}),
        grounding_context=deepcopy(prepared_bridge_request.get("grounding_context") or {}),
        bridge_resources=_bridge_resources_for_bridge_normalizer(
            prepared_bridge_request.get("bridge_resources") or {}
        ),
        obligation_targets=deepcopy(
            list(prepared_bridge_request.get("obligation_targets") or [])
        ),
    )
    if not isinstance(normalized, dict):
        result["reason"] = "primitive normalizer rejected the derived final_output proposal"
        return result

    result["accepted"] = True
    result["normalized_proposal"] = deepcopy(normalized)
    return result


def _build_final_output_payload(
    session_state: dict[str, Any],
    *,
    stage: str,
) -> dict[str, Any]:
    """Build the standalone final-output response shown in debug artifacts."""
    accepted_prefix = [
        deepcopy(row)
        for row in (session_state.get("accepted_outline_prefix") or [])
        if isinstance(row, dict)
    ]
    accepted_program = [
        deepcopy(row)
        for row in (session_state.get("accepted_primitive_program") or [])
        if isinstance(row, dict)
    ]
    executable_trace: list[dict[str, Any]] = []
    primitives_by_outline_id = {
        str(row.get("outline_id") or "").strip(): deepcopy(row)
        for row in accepted_program
        if isinstance(row, dict) and str(row.get("outline_id") or "").strip()
    }
    for event in accepted_prefix:
        outline_id = str(event.get("outline_id") or "").strip()
        primitive_row = dict(primitives_by_outline_id.get(outline_id) or {})
        executable_trace.append({
            "outline_id": outline_id,
            "des_event_id": outline_id,
            "event_name": str(
                event.get("event_name")
                or primitive_row.get("event_name")
                or ""
            ).strip(),
            "resource_jid": str(event.get("resource_jid") or "").strip(),
            "part_name": str(event.get("part_name") or "").strip() or None,
            "target_ref": str(event.get("target_ref") or "").strip() or None,
            "description": str(event.get("description") or "").strip(),
            "expected_start_state": deepcopy(event.get("expected_start_state") or {}),
            "expected_end_state": deepcopy(event.get("expected_end_state") or {}),
            "primitive_steps": deepcopy(primitive_row.get("primitive_steps") or []),
        })

    return {
        "engine": "multi_turn",
        "decision": "final_output_ready",
        "final_output_stage": str(stage or "").strip() or "unknown",
        "status": str(session_state.get("status") or "").strip(),
        "current_phase": str(session_state.get("current_phase") or "").strip(),
        "accepted_trace_length": len(accepted_prefix),
        "transition_trace": deepcopy(accepted_prefix),
        "accepted_primitive_program": deepcopy(accepted_program),
        "executable_recovery_trace": executable_trace,
        "primitive_program_complete": bool(accepted_prefix)
        and len(accepted_program) >= len(accepted_prefix),
    }


def _append_final_output_turn(
    *,
    planner: Any,
    prepared_bridge_request: dict[str, Any],
    session_state: dict[str, Any],
    stage: str,
) -> dict[str, Any] | None:
    """Append and persist a synthetic final-output turn without another LLM call."""
    if any(
        str(turn.get("phase") or "").strip().lower() == "final_output"
        and str(turn.get("final_output_stage") or "").strip() == stage
        for turn in (session_state.get("turns") or [])
        if isinstance(turn, dict)
    ):
        return None

    final_output = _build_final_output_payload(session_state, stage=stage)
    adapter_result = normalize_multi_turn_final_output_to_bridge_proposal(
        final_output_payload=final_output,
        prepared_bridge_request=prepared_bridge_request,
    )
    normalized_proposal = (
        deepcopy(adapter_result.get("normalized_proposal"))
        if isinstance(adapter_result.get("normalized_proposal"), dict)
        else None
    )
    final_turn_index = int(session_state.get("turn_index") or 0) + 1
    turn_entry: dict[str, Any] = {
        "turn_index": final_turn_index,
        "phase": "final_output",
        "decision": "final_output_ready",
        "final_output_stage": stage,
        "prompt_text": "",
        "llm_raw_response": {},
        "raw_response": _compact_final_output_artifact_payload(final_output),
        "phase_result": deepcopy(final_output),
    }
    if normalized_proposal is not None:
        turn_entry["normalized_proposal"] = deepcopy(normalized_proposal)
    if isinstance(adapter_result, dict):
        turn_entry["adapter_result"] = deepcopy(adapter_result)
    session_state["turn_index"] = final_turn_index
    session_state["proposal"] = deepcopy(normalized_proposal) if normalized_proposal else None
    session_state["final_output"] = deepcopy(final_output)
    session_state["final_output_adapter"] = deepcopy(adapter_result)
    session_state.setdefault("turns", []).append(deepcopy(turn_entry))

    bridge_debug = deepcopy(prepared_bridge_request.get("bridge_debug") or {})
    bridge_debug["multi_turn_session"] = deepcopy(session_state)
    bridge_debug["status"] = str(session_state.get("status") or "")
    bridge_debug["normalized_proposal"] = deepcopy(normalized_proposal)
    bridge_debug["final_output_adapter"] = deepcopy(adapter_result)
    bridge_debug["final_output"] = deepcopy(final_output)
    prepared_bridge_request["bridge_debug"] = deepcopy(bridge_debug)
    if hasattr(planner, "_set_last_bridge_debug"):
        planner._set_last_bridge_debug(bridge_debug)
    _write_per_turn_artifact(prepared_bridge_request, session_state, turn_entry)
    return final_output


# ---------------------------------------------------------------------------
# Per-turn debug artifacts
# ---------------------------------------------------------------------------


def _write_per_turn_artifact(
    prepared_bridge_request: dict[str, Any],
    session_state: dict[str, Any],
    turn_entry: dict[str, Any],
) -> None:
    bridge_debug = dict(prepared_bridge_request.get("bridge_debug") or {})
    per_turn_debug_dir = str(bridge_debug.get("per_turn_debug_dir") or "").strip()
    if not per_turn_debug_dir:
        return
    payload = {
        **deepcopy(prepared_bridge_request),
        "reasoning_mode": "multi_turn",
        "multi_turn_prompt_input": deepcopy(turn_entry.get("prompt_input")),
        "multi_turn_prompt_text": str(turn_entry.get("prompt_text") or ""),
        "multi_turn_raw_response": deepcopy(turn_entry.get("raw_response")),
        "multi_turn_session_result": deepcopy(session_state),
    }
    if isinstance(turn_entry.get("llm_raw_response"), dict):
        payload["multi_turn_llm_raw_response"] = deepcopy(turn_entry.get("llm_raw_response"))
    try:
        write_bridge_artifacts(
            payload,
            phase_label="multi_turn",
            debug_dir=per_turn_debug_dir,
            write_latest=False,
        )
    except Exception as exc:
        _logger.warning("[MultiTurn] Failed to write per-turn artifact: %s", exc)


# ---------------------------------------------------------------------------
# Main execution loop
# ---------------------------------------------------------------------------


async def execute_multi_turn_bridge(
    planner: Any,
    prepared_bridge_request: dict[str, Any],
    *,
    session_state: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Run the multi-turn bridge loop.

    Iterates through phases (grounding → outline → primitive_generation → finalize),
    calling the LLM each turn and dispatching to the appropriate phase handler.

    Pass an existing *session_state* to resume from a previous pause (e.g.
    ``paused_after_outline_turn``).  When omitted a fresh seed is built.
    """
    product_agent = getattr(planner, "product_agent", None)
    ask_llm_structured = getattr(product_agent, "ask_llm_structured", None)
    if not callable(ask_llm_structured):
        raise RuntimeError(
            "product_agent.ask_llm_structured is required for multi-turn bridge execution"
        )

    if session_state is not None:
        # Resume from a prior pause
        session_state = deepcopy(session_state)
    else:
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

        _logger.info(
            "[MultiTurn] Turn %d/%d | phase=%s",
            turn_idx, max_turns, current_phase,
        )

        # 1. Build prompt
        prompt_input, prompt_text = _build_phase_prompt(
            prepared_bridge_request, session_state,
        )

        # 2. Call LLM
        response_schema = _get_response_schema(current_phase, session_state)
        raw_response = await ask_llm_structured(
            prompt=prompt_text,
            response_format=response_schema,
        )
        parsed_response = deepcopy(raw_response if isinstance(raw_response, dict) else {})

        # 3. Dispatch to phase handler
        handler = _PHASE_HANDLERS.get(current_phase)
        if handler is None:
            _logger.error("[MultiTurn] No handler for phase=%s", current_phase)
            session_state["status"] = "error"
            break

        decision, turn_entry = await handler(
            session_state=session_state,
            parsed_response=parsed_response,
            prepared_bridge_request=prepared_bridge_request,
            planner=planner,
        )

        # 4. Record turn
        turn_entry["turn_index"] = turn_idx
        turn_entry["phase"] = current_phase
        turn_entry["prompt_text"] = prompt_text
        turn_entry["llm_raw_response"] = deepcopy(parsed_response)
        turn_entry["decision"] = decision
        turn_entry["raw_response"] = _normalized_response_artifact_payload(
            phase=current_phase,
            session_state=session_state,
            parsed_response=parsed_response,
            turn_entry=turn_entry,
        )
        session_state.setdefault("turns", []).append(deepcopy(turn_entry))

        # 5. Transition
        next_phase = transition_multi_turn_phase(current_phase, decision)
        session_state["current_phase"] = next_phase

        # 6. Check terminal conditions
        if current_phase == "finalize" and decision == "accepted":
            session_state["status"] = "completed"

        # Update debug + write per-turn artifacts
        bridge_debug["multi_turn_session"] = deepcopy(session_state)
        bridge_debug["status"] = str(session_state.get("status") or "running")
        prepared_bridge_request["bridge_debug"] = deepcopy(bridge_debug)
        if hasattr(planner, "_set_last_bridge_debug"):
            planner._set_last_bridge_debug(bridge_debug)
        _write_per_turn_artifact(prepared_bridge_request, session_state, turn_entry)

        if current_phase == "outline" and decision == "outline_ready":
            _append_final_output_turn(
                planner=planner,
                prepared_bridge_request=prepared_bridge_request,
                session_state=session_state,
                stage="outline_ready",
            )
        elif current_phase == "primitive_generation" and decision == "draft_ready":
            _append_final_output_turn(
                planner=planner,
                prepared_bridge_request=prepared_bridge_request,
                session_state=session_state,
                stage="primitive_program_ready",
            )

        if session_state.get("status") in (
            "completed", "paused_after_outline_turn",
            "paused_after_primitive_turn", "paused_after_primitive_generation",
            "paused_after_primitive_blocked",
            "paused_after_primitive_stuck",
            "des_cycle_detected", "des_deadlock",
        ):
            break

    # Stash session state for resume access
    prepared_bridge_request["multi_turn_session_state"] = deepcopy(session_state)

    # Turn budget exhausted
    if session_state.get("status") not in (
        "completed",
        "paused_after_outline_turn",
        "paused_after_primitive_turn",
        "paused_after_primitive_generation",
        "paused_after_primitive_blocked",
        "paused_after_primitive_stuck",
        "des_cycle_detected",
        "des_deadlock",
        "error",
    ):
        session_state["status"] = "des_turn_limit"
        bridge_debug["multi_turn_session"] = deepcopy(session_state)
        bridge_debug["status"] = "des_turn_limit"
        prepared_bridge_request["bridge_debug"] = deepcopy(bridge_debug)
        prepared_bridge_request["multi_turn_session_state"] = deepcopy(session_state)
        if hasattr(planner, "_set_last_bridge_debug"):
            planner._set_last_bridge_debug(bridge_debug)
        _logger.warning(
            "[MultiTurn] Turn budget exhausted (%d turns) in phase=%s",
            int(session_state.get("max_turns") or 0),
            str(session_state.get("current_phase") or ""),
        )
        return None

    # Build proposal from finalized session
    return deepcopy(session_state.get("proposal"))
