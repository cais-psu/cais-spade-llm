"""DES recovery bridge execution engine with one-task-at-a-time outlines."""

from __future__ import annotations

import inspect
import json
import logging
from copy import deepcopy
from typing import Any

from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.bridge_artifacts import (
    write_bridge_artifacts,
)
from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.modes.des_recovery_automaton import (
    compose_and_solve,
    solver_diagnostic_summary,
)
from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.modes.des_recovery_shared import (
    extract_ap_descriptors as _extract_ap_descriptors,
    extract_safety_dfas as _extract_safety_dfas,
)
from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.modes.des_recovery_outline_helpers import (
    active_candidate_recovery_blockers,
    active_des_recovery_pruned_actions,
    compute_enabled_candidate_bound,
    symbolic_state_fingerprint,
)
from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.modes.des_recovery_outline_runtime import (
    _handle_outline_phase,
)
from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.modes.des_recovery_primitive_generation import (
    _handle_primitive_generation_phase,
    build_primitive_generation_prompt_context,
)
from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.primitive_semantics import (
    extract_step_output,
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
_PHASE_SEQUENCE = ("grounding", "outline", "primitive_generation", "finalize")

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


def transition_des_recovery_phase(current_phase: str, decision: str) -> str:
    """Return the next phase given the current phase and a decision token."""
    phase = current_phase.strip().lower()
    token = decision.strip().lower()
    phase_map = _TRANSITIONS.get(phase, {})
    next_phase = phase_map.get(token)
    if next_phase is None:
        _logger.warning(
            "[DesRecovery] No transition for phase=%s decision=%s; staying in %s",
            phase, token, phase,
        )
        return phase
    return next_phase


# ---------------------------------------------------------------------------
# Session seed
# ---------------------------------------------------------------------------


def build_des_recovery_session_seed(
    prepared_bridge_request: dict[str, Any],
) -> dict[str, Any]:
    """Build the initial session state for a DES recovery bridge run."""
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
            or bridge_session.get("des_candidate_bound")
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
        "des_recovery_engine": "des_recovery",
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
        # DES plant (built incrementally as candidates are accepted)
        "des_plant": {
            "states": [],
            "initial": "",
            "marked": [],
            "events": {},
            "transitions": {},
        },
        "des_current_state": "",
        "des_trace": [],
        "des_safety_dfa_vector": (),
        "des_safety_dfas": {},
        "des_ap_descriptors": [],
        "des_visited_states": [],
    }


# ---------------------------------------------------------------------------
# DES plant helpers
# ---------------------------------------------------------------------------


def _init_des_state(
    session_state: dict[str, Any],
    planner: Any,
    prepared_bridge_request: dict[str, Any],
) -> None:
    """Initialize incremental DES plant at the grounding→outline transition."""
    initial = symbolic_state_fingerprint(
        session_state["symbolic_resources"],
        session_state["symbolic_parts"],
    )
    plant = session_state["des_plant"]
    plant["initial"] = initial
    plant["states"] = [initial]

    # Derive marked states from continuation conditions (goal = all parts at
    # goal_location).  We use a single abstract marked state "S_goal" since the
    # exact goal fingerprint is not yet known — the solver treats any plant
    # marked state as accepting.
    plant["marked"] = ["S_goal"]

    session_state["des_current_state"] = initial
    session_state["des_trace"] = []
    session_state["des_visited_states"] = [initial]

    # Cache safety DFAs and AP descriptors for the session lifetime.
    session_state["des_safety_dfas"] = _extract_safety_dfas(
        planner, prepared_bridge_request,
    )
    session_state["des_ap_descriptors"] = _extract_ap_descriptors(
        planner, prepared_bridge_request,
    )

    # Initialize safety DFA state vector — all DFAs at their initial state.
    dfas = session_state["des_safety_dfas"]
    session_state["des_safety_dfa_vector"] = tuple(
        str(dfa.get("initial") or "q0")
        for dfa in dfas.values()
    )

    _logger.info(
        "[DES] Initialized incremental plant: initial=%s, safety_dfas=%d, aps=%d",
        initial,
        len(dfas),
        len(session_state["des_ap_descriptors"]),
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
                "[DesRecovery] Grounding override → grounded (%s)",
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
            _logger.warning("[DesRecovery] observe failed: %s", observe_error)
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
            "[DesRecovery] Observed %d facts", len(observation_results),
        )

    else:
        # decision == "grounded"
        _seed_grounded_part_pose_observations(prepared_bridge_request, session_state)
        _logger.info("[DesRecovery] Grounding complete")

    return decision, turn_entry

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
    from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.modes.des_recovery_prompts import (
        build_des_recovery_phase_prompt_input,
        render_des_recovery_phase_prompt,
    )

    phase = str(session_state.get("current_phase") or "grounding").strip().lower()
    llm_input = dict(prepared_bridge_request.get("llm_input") or {})
    outline_mode = str(session_state.get("outline_mode") or "incremental").strip().lower()

    world_observation_surface: dict[str, Any] | None = None
    if phase == "grounding":
        world_observation_surface = _build_world_observation_surface(prepared_bridge_request)

    current_recovery_blockers: list[dict[str, Any]] | None = None
    if phase == "outline" and outline_mode == "incremental_candidates_validated":
        session_state["pruned_actions"] = active_des_recovery_pruned_actions(
            session_state,
            prepared_bridge_request,
        )
        current_recovery_blockers = active_candidate_recovery_blockers(
            session_state=session_state,
            prepared_bridge_request=prepared_bridge_request,
        )
        # Compute the configured candidate budget for this turn.
        candidate_bound = compute_enabled_candidate_bound(session_state)
        session_state["des_candidate_bound"] = candidate_bound

    prompt_input = build_des_recovery_phase_prompt_input(
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
    prompt_text = render_des_recovery_phase_prompt(prompt_input)
    return prompt_input, prompt_text


def _get_response_schema(phase: str, session_state: dict[str, Any]) -> dict[str, Any]:
    """Return the JSON response schema for the given phase."""
    from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.modes.des_recovery_prompts import (
        des_recovery_phase_response_schema,
    )
    outline_mode = str(session_state.get("outline_mode") or "incremental").strip().lower()
    candidate_bound = None
    if phase.strip().lower() == "outline" and outline_mode == "incremental_candidates_validated":
        candidate_bound = max(
            1,
            int(
                session_state.get("des_candidate_bound")
                or session_state.get("candidate_bound")
                or _DEFAULT_CANDIDATE_BOUND
            ),
        )
    return des_recovery_phase_response_schema(
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
    "action_name",
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
    outline_tasks = payload.get("outline_tasks")
    if not isinstance(outline_tasks, list) or not outline_tasks:
        outline_tasks = payload.get("accepted_transition_prefix")
    if isinstance(outline_tasks, list) and outline_tasks:
        compact["outline_tasks"] = [
            _compact_artifact_task(row)
            for row in outline_tasks
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
        normalized.setdefault("accepted_transition_prefix", deepcopy(accepted_prefix))
        normalized.setdefault("des_event_sequence", deepcopy(accepted_prefix))
        normalized.setdefault("transition_trace", deepcopy(accepted_prefix))
    next_transition = turn_entry.get("next_transition")
    if not isinstance(next_transition, dict):
        next_transition = turn_entry.get("next_task")
    if isinstance(next_transition, dict):
        normalized.setdefault("next_transition", deepcopy(next_transition))
        normalized.setdefault("next_recovery_event", deepcopy(next_transition))
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
        for row in (
            turn_entry.get("candidate_events")
            or turn_entry.get("candidate_transitions")
            or turn_entry.get("candidate_tasks")
            or []
        )
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
    if isinstance(turn_entry.get("selected_next_task"), dict):
        payload["selected_transition"] = _compact_artifact_task(
            turn_entry.get("selected_transition") or turn_entry.get("selected_next_task")
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
        payload["accepted_transition_prefix"] = deepcopy(accepted_prefix)
    if isinstance(turn_entry.get("transition_validation"), dict):
        payload["transition_validation"] = _compact_artifact_transition_validation(
            turn_entry.get("transition_validation") or {}
        )
    return payload


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
                or event.get("action_name")
                or primitive_row.get("event_name")
                or primitive_row.get("action_name")
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
        "engine": "des_recovery",
        "decision": "final_output_ready",
        "final_output_stage": str(stage or "").strip() or "unknown",
        "status": str(session_state.get("status") or "").strip(),
        "current_phase": str(session_state.get("current_phase") or "").strip(),
        "accepted_trace_length": len(accepted_prefix),
        "accepted_transition_prefix": deepcopy(accepted_prefix),
        "des_event_sequence": deepcopy(accepted_prefix),
        "outline_tasks": deepcopy(accepted_prefix),
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
    session_state["turn_index"] = final_turn_index
    session_state["proposal"] = deepcopy(final_output)
    session_state["final_output"] = deepcopy(final_output)
    session_state.setdefault("turns", []).append(deepcopy(turn_entry))

    bridge_debug = deepcopy(prepared_bridge_request.get("bridge_debug") or {})
    bridge_debug["des_recovery_session"] = deepcopy(session_state)
    bridge_debug["status"] = str(session_state.get("status") or "")
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
        "reasoning_mode": "des_recovery",
        "des_recovery_prompt_input": deepcopy(turn_entry.get("prompt_input")),
        "des_recovery_prompt_text": str(turn_entry.get("prompt_text") or ""),
        "des_recovery_raw_response": deepcopy(turn_entry.get("raw_response")),
        "des_recovery_session_result": deepcopy(session_state),
    }
    if isinstance(turn_entry.get("llm_raw_response"), dict):
        payload["des_recovery_llm_raw_response"] = deepcopy(turn_entry.get("llm_raw_response"))
    try:
        write_bridge_artifacts(
            payload,
            phase_label="des_recovery",
            debug_dir=per_turn_debug_dir,
            write_latest=False,
        )
    except Exception as exc:
        _logger.warning("[DesRecovery] Failed to write per-turn artifact: %s", exc)


# ---------------------------------------------------------------------------
# Main execution loop
# ---------------------------------------------------------------------------


async def execute_des_recovery_bridge(
    planner: Any,
    prepared_bridge_request: dict[str, Any],
    *,
    session_state: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Run the DES recovery bridge loop.

    Iterates through phases (grounding → outline → primitive_generation → finalize),
    calling the LLM each turn and dispatching to the appropriate phase handler.

    Pass an existing *session_state* to resume from a previous pause (e.g.
    ``paused_after_outline_turn``).  When omitted a fresh seed is built.
    """
    product_agent = getattr(planner, "product_agent", None)
    ask_llm_structured = getattr(product_agent, "ask_llm_structured", None)
    if not callable(ask_llm_structured):
        raise RuntimeError(
            "product_agent.ask_llm_structured is required for DES recovery bridge execution"
        )

    if session_state is not None:
        # Resume from a prior pause
        session_state = deepcopy(session_state)
    else:
        session_state = deepcopy(
            prepared_bridge_request.get("des_recovery_session_seed")
            or build_des_recovery_session_seed(prepared_bridge_request)
        )
    session_state["status"] = "running"

    bridge_debug = deepcopy(prepared_bridge_request.get("bridge_debug") or {})
    bridge_debug["des_recovery_session"] = deepcopy(session_state)
    prepared_bridge_request["bridge_debug"] = deepcopy(bridge_debug)
    if hasattr(planner, "_set_last_bridge_debug"):
        planner._set_last_bridge_debug(bridge_debug)

    while int(session_state.get("turn_index") or 0) < int(session_state.get("max_turns") or 0):
        session_state["turn_index"] = int(session_state.get("turn_index") or 0) + 1
        current_phase = str(session_state.get("current_phase") or "grounding").strip().lower()
        turn_idx = int(session_state.get("turn_index") or 0)
        max_turns = int(session_state.get("max_turns") or 0)

        _logger.info(
            "[DesRecovery] Turn %d/%d | phase=%s",
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
            _logger.error("[DesRecovery] No handler for phase=%s", current_phase)
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
        next_phase = transition_des_recovery_phase(current_phase, decision)
        session_state["current_phase"] = next_phase

        # 5a. Initialize DES plant at grounding→outline transition
        if current_phase == "grounding" and next_phase == "outline":
            if not session_state.get("des_plant", {}).get("initial"):
                _init_des_state(session_state, planner, prepared_bridge_request)

        # 6. Check terminal conditions
        if current_phase == "finalize" and decision == "accepted":
            session_state["status"] = "completed"

        # Update debug + write per-turn artifacts
        bridge_debug["des_recovery_session"] = deepcopy(session_state)
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
    prepared_bridge_request["des_recovery_session_state"] = deepcopy(session_state)

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
        bridge_debug["des_recovery_session"] = deepcopy(session_state)
        bridge_debug["status"] = "des_turn_limit"
        prepared_bridge_request["bridge_debug"] = deepcopy(bridge_debug)
        prepared_bridge_request["des_recovery_session_state"] = deepcopy(session_state)
        if hasattr(planner, "_set_last_bridge_debug"):
            planner._set_last_bridge_debug(bridge_debug)
        _logger.warning(
            "[DesRecovery] Turn budget exhausted (%d turns) in phase=%s",
            int(session_state.get("max_turns") or 0),
            str(session_state.get("current_phase") or ""),
        )
        return None

    # Build proposal from finalized session
    return deepcopy(session_state.get("proposal"))
