"""Multi-turn bridge execution engine v2 — one-task-at-a-time outline."""

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
from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.modes.multi_turn import (
    _apply_outline_task_effects as _v1_apply_outline_task_effects,
    _build_outline_task_type_lookup as _v1_build_outline_task_type_lookup,
    _infer_outline_macro_signature as _v1_infer_outline_macro_signature,
    _outline_task_depends_on as _v1_outline_task_depends_on,
    _task_findings_block_projected_state as _v1_task_findings_block_projected_state,
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

_DEFAULT_MAX_TURNS = 20
_DEFAULT_MAX_OBSERVATIONS = 3
_DEFAULT_MAX_OBSERVE_BATCH = 3
_OUTLINE_STAGNATION_LIMIT = 3
_V2_OUTLINE_CONTRACT = {
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
        "need_outline_revision": "outline",
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
            "[MultiTurnV2] No transition for phase=%s decision=%s; staying in %s",
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
    """Build the initial session state for a multi-turn v2 bridge run."""
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
        "multi_turn_engine": "v2",
        "current_phase": "grounding",
        "turn_index": 0,
        "max_turns": max_turns,
        "max_observations": max_observations,
        "max_observe_batch": max_observe_batch,
        "outline_mode": outline_mode,
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
        "outline_lookahead": [],
        "outline_stagnation_count": 0,
        "outline_progress_signature": "",
        "pruned_actions": [],
        "outline_validation_findings": [],
        "candidate_rejection_feedback": [],
        # Symbolic state for validation
        "symbolic_resources": symbolic_resources,
        "symbolic_parts": symbolic_parts,
    }


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


def _supports_store_as(*, resource_type: str, primitive_name: str) -> bool:
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
        if not _supports_store_as(resource_type=resource_type, primitive_name=primitive_name):
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


def _default_observe_store_as(
    request: dict[str, Any],
    *,
    session_state: dict[str, Any],
    seen_aliases: set[str],
) -> str:
    fact_type = str(request.get("fact_type") or "").strip()
    entity = str(request.get("entity") or "").strip()
    if fact_type == "part_pose" and entity:
        prefix = "observed_pose"
    elif fact_type:
        prefix = fact_type.replace(" ", "_").lower() or "observed_fact"
    else:
        prefix = "observed_fact"
    base = f"{prefix}_{entity}" if entity else prefix
    alias = base
    counter = 2
    while alias in seen_aliases:
        alias = f"{base}_{counter}"
        counter += 1
    return alias


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
            "store_as": "",
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
        "store_as": "",
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
    aliases = [
        str(a).strip() for a in (existing.get("aliases") or []) if str(a).strip()
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
    seen_aliases = set(observation_store)

    for part_name, tracker_entry in part_tracker.items():
        pose = _observation_pose(
            {"pose": deepcopy(tracker_entry.get("observed_pose"))}
        ) or _observation_pose(tracker_entry)
        if pose is None:
            continue
        alias = _default_observe_store_as(
            {"fact_type": "part_pose", "entity": part_name, "primitive": "grounding_context"},
            session_state=session_state,
            seen_aliases=seen_aliases,
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

        observation_store[alias] = deepcopy(payload)
        seen_aliases.add(alias)
        _record_observation_fact(session_state, {
            "fact_key": _observation_fact_key("part_pose", part_name, None),
            "fact_type": "part_pose",
            "entity": part_name,
            "entity_kind": "part",
            "scope": None,
            "primitive": "grounding_context",
            "params": {"part_name": part_name},
            "store_as": alias,
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
        if not _supports_store_as(resource_type=resource_type, primitive_name=name):
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
    seen_aliases = set(dict(session_state.get("observation_store") or {}))
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

        store_as = _default_observe_store_as(
            normalized, session_state=session_state, seen_aliases=seen_aliases,
        )
        normalized["store_as"] = store_as

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
            "store_as": store_as,
            "output": deepcopy(extracted_output),
        }
        results.append(observation_row)
        seen_aliases.add(store_as)
        seen_request_rows[request_key] = deepcopy(observation_row)
        if fact_key:
            seen_fact_rows[fact_key] = {
                **deepcopy(observation_row),
                "turn_index": int(session_state.get("turn_index") or 0),
                "validity": "current",
                "freshness": "current_session",
                "aliases": [store_as],
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
                "[MultiTurnV2] Grounding override → grounded (%s)",
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
            _logger.warning("[MultiTurnV2] observe failed: %s", observe_error)
            turn_entry["error"] = observe_error
            session_state["phase_feedback"].append({
                "phase": "grounding", "issue": "observe_error", "detail": observe_error,
            })
            return decision, turn_entry

        # Store results
        for result in observation_results:
            alias = str(result.get("store_as") or "").strip()
            session_state["observation_store"][alias] = deepcopy(result.get("output") or {})
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
            "[MultiTurnV2] Observed %d facts", len(observation_results),
        )

    else:
        # decision == "grounded"
        _seed_grounded_part_pose_observations(prepared_bridge_request, session_state)
        _logger.info("[MultiTurnV2] Grounding complete")

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
        part_row = parts_by_name.setdefault(part_name, {"part_name": part_name})
        pose = dict(entry.get("pose") or {})
        if not pose and entry.get("x") is not None:
            pose = {"x": entry.get("x"), "y": entry.get("y"), "z": entry.get("z")}
        if pose:
            part_row["observed_pose"] = deepcopy(pose)
        if part_row.get("current_location") in (None, "") and entry.get("current_location") not in (None, ""):
            part_row["current_location"] = deepcopy(entry.get("current_location"))
        holder = str(entry.get("current_holder_resource_jid") or "").strip()
        if not str(part_row.get("current_holder_resource_jid") or "").strip() and holder:
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
    outline_tasks = [deepcopy(task)]
    task_id = str(task.get("outline_id") or "").strip() or "task_0"
    task_types_by_id = _v1_build_outline_task_type_lookup(
        outline_tasks,
        resources_by_jid=resources_by_jid,
        parts_by_name=parts_by_name,
        llm_input=llm_input,
    )
    signature = _v1_infer_outline_macro_signature(
        task,
        resources_by_jid=resources_by_jid,
        parts_by_name=parts_by_name,
    )

    projected_resources = deepcopy(resources_by_jid)
    projected_parts = deepcopy(parts_by_name)
    if not _v1_task_findings_block_projected_state(list(prior_findings or [])):
        task_type = str(task_types_by_id.get(task_id) or "").strip()
        _v1_apply_outline_task_effects(
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
        outline_tasks=outline_tasks,
        task_types_by_id=deepcopy(task_types_by_id),
        task_index_by_id={task_id: 0},
        dependency_map={task_id: _v1_outline_task_depends_on(task)},
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
        outline_contract=deepcopy(_V2_OUTLINE_CONTRACT),
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
    """Small stable key for de-duping v2 validation findings."""
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


def _release_part_frees_resource_for_blocker(
    *,
    task: dict[str, Any],
    session_state: dict[str, Any],
    candidate_session_state: dict[str, Any],
    prepared_bridge_request: dict[str, Any],
    current_blockers: list[dict[str, Any]],
) -> bool:
    if str(task.get("action_type") or "").strip().lower() != "release_part":
        return False

    if any(
        _normalized_blocker_kind(blocker) == "resource_terminal_state"
        for blocker in current_blockers
        if isinstance(blocker, dict)
    ):
        return False

    resource_jid = str(task.get("resource_jid") or "").strip()
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


def _acquire_part_counts_as_blocker_progress(
    *,
    task: dict[str, Any],
    session_state: dict[str, Any],
    candidate_session_state: dict[str, Any],
    prepared_bridge_request: dict[str, Any],
    current_blockers: list[dict[str, Any]],
) -> bool:
    if str(task.get("action_type") or "").strip().lower() != "acquire_part":
        return False

    if any(
        _normalized_blocker_kind(blocker) == "resource_terminal_state"
        for blocker in current_blockers
        if isinstance(blocker, dict)
    ):
        return False

    resource_jid = str(task.get("resource_jid") or "").strip()
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
    if resolved_blockers == 0 and _acquire_part_counts_as_blocker_progress(
        task=task,
        session_state=session_state,
        candidate_session_state=candidate_session_state,
        prepared_bridge_request=prepared_bridge_request,
        current_blockers=current_blockers,
    ):
        blocker_part_acquired = 1
    resource_freed_for_blocker = 0
    if resolved_blockers == 0 and blocker_part_acquired == 0 and _release_part_frees_resource_for_blocker(
        task=task,
        session_state=session_state,
        candidate_session_state=candidate_session_state,
        prepared_bridge_request=prepared_bridge_request,
        current_blockers=current_blockers,
    ):
        resource_freed_for_blocker = 1
    remaining_blocked_issues = len(remaining_keys)
    secondary_progress = blocker_part_acquired + resource_freed_for_blocker
    return (
        resolved_blockers + secondary_progress,
        {
            "resolved_direct_blockers": resolved_blockers,
            "blocker_part_acquired": blocker_part_acquired,
            "freed_resource_for_blocker": resource_freed_for_blocker,
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
        rows.append({
            "candidate_index": int(row.get("candidate_index") or 0),
            "task": deepcopy(dict(row.get("task") or {})),
            "validation_findings": findings,
        })
    return rows


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


def _candidate_action_name_from_task(task: dict[str, Any]) -> str:
    action_name = str(task.get("action_name") or "").strip()
    if action_name:
        return action_name
    legacy_action_type = str(task.get("action_type") or "").strip().lower()
    if legacy_action_type:
        return legacy_action_type.replace("_", " ")
    return ""


def _candidate_target_ref_from_surface_task(task: dict[str, Any]) -> str:
    if str(task.get("target_ref") or "").strip():
        return str(task.get("target_ref") or "").strip()
    action_target = dict(task.get("action_target") or {})
    return str(
        action_target.get("target_location")
        or action_target.get("named_pose")
        or ""
    ).strip()


def _surface_candidate_hidden_action_type(
    *,
    part_name: str,
    target_ref: str,
) -> str:
    if part_name:
        return "release_part" if target_ref else "acquire_part"
    return "recover_resource"


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
    action_name = _candidate_action_name_from_task(raw_task)
    description = str(raw_task.get("description") or "").strip()
    part_name = str(raw_task.get("part_name") or "").strip()
    target_ref = _candidate_target_ref_from_surface_task(raw_task)
    if action_name:
        normalized["action_name"] = action_name
    if description:
        normalized["description"] = description
    if part_name:
        normalized["part_name"] = part_name
    if target_ref:
        normalized["target_ref"] = target_ref
    if original_outline_id:
        normalized["llm_outline_id"] = original_outline_id
    normalized["outline_id"] = _candidate_outline_id(
        sequence_index=sequence_index,
        candidate_index=candidate_index,
    )
    return normalized


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
    action_name = str(candidate_task.get("action_name") or "").strip()
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

    if not action_name:
        return None, [
            _candidate_schema_finding(
                task=candidate_task,
                reason="candidate must include action_name naming the physical intent",
                evidence={"field": "action_name"},
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

    action_type = _surface_candidate_hidden_action_type(
        part_name=part_name,
        target_ref=target_ref,
    )

    if action_type == "acquire_part":
        if part_name not in parts_by_name:
            return None, [
                _candidate_schema_finding(
                    task=candidate_task,
                    reason=f"candidate references unknown part '{part_name}'",
                    evidence={"field": "part_name", "token": part_name},
                )
            ]

    if action_type == "release_part":
        if part_name not in parts_by_name:
            return None, [
                _candidate_schema_finding(
                    task=candidate_task,
                    reason=f"candidate references unknown part '{part_name}'",
                    evidence={"field": "part_name", "token": part_name},
                )
            ]
        if not target_ref:
            return None, [
                _candidate_schema_finding(
                    task=candidate_task,
                    reason="release_part requires target_ref",
                    evidence={"field": "target_ref"},
                )
            ]

    start_state: dict[str, Any] = {}
    current_resource_state = _resource_current_state_token(resource_row)
    if current_resource_state or "current_state" in resource_row or "state" in resource_row:
        start_state["resource_state"] = current_resource_state or None
    if "held_part" in resource_row:
        start_state["held_part"] = resource_row.get("held_part") or None

    if part_name:
        current_part_state = _part_current_state_token(part_row)
        current_part_location = _part_current_location_token(part_row)
        current_part_holder = _part_current_holder_token(part_row)
        if current_part_state or "current_state" in part_row or "state" in part_row:
            start_state["part_state"] = current_part_state or None
        if current_part_location:
            start_state["part_location"] = current_part_location
        elif isinstance(part_row.get("observed_pose"), dict) and "x" in dict(part_row.get("observed_pose") or {}):
            start_state["part_location"] = "observed_pose"
        else:
            start_state["part_location"] = None
        if current_part_holder or "current_holder_resource_jid" in part_row or "holder_resource_jid" in part_row:
            start_state["part_holder_resource_jid"] = current_part_holder or None

    action_target: dict[str, Any] = {}
    end_state: dict[str, Any] = {}

    if action_type == "recover_resource":
        end_state["resource_state"] = "idle"
        if target_ref:
            if target_ref in _candidate_named_pose_tokens(resource_row):
                action_target["named_pose"] = target_ref
            else:
                action_target["target_location"] = target_ref
    elif action_type == "acquire_part":
        current_part_location = _part_current_location_token(part_row)
        if current_part_location:
            action_target["source_location"] = current_part_location
        elif isinstance(part_row.get("observed_pose"), dict) and "x" in dict(part_row.get("observed_pose") or {}):
            action_target["source_location"] = "observed_pose"
        end_state.update({
            "resource_state": "picked",
            "held_part": part_name,
            "part_location": f"{resource_jid}_gripper",
            "part_holder_resource_jid": resource_jid,
        })
    elif action_type == "release_part":
        action_target["target_location"] = target_ref
        end_state.update({
            "resource_state": "idle",
            "held_part": None,
            "part_location": target_ref,
            "part_holder_resource_jid": None,
        })

    normalized_task: dict[str, Any] = {
        "outline_id": str(candidate_task.get("outline_id") or "").strip(),
        "resource_jid": resource_jid,
        "action_name": action_name,
        "action_type": action_type,
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
    """Update symbolic resource/part state based on accepted task's expected_end_state."""
    resource_jid = str(task.get("resource_jid") or "").strip()
    part_name = str(task.get("part_name") or "").strip()
    action_type = str(task.get("action_type") or "").strip().lower()
    target_ref = str(task.get("target_ref") or "").strip()
    end_state = dict(task.get("expected_end_state") or {})

    symbolic_resources = dict(session_state.get("symbolic_resources") or {})
    symbolic_parts = dict(session_state.get("symbolic_parts") or {})

    # Update resource
    if resource_jid:
        res = symbolic_resources.setdefault(resource_jid, {"resource_jid": resource_jid})
        if "resource_state" in end_state:
            res["current_state"] = end_state["resource_state"]
        if "held_part" in end_state:
            res["held_part"] = end_state["held_part"] or None
            res["gripper_state"] = "closed" if end_state["held_part"] else "open"

    # Update part
    if part_name:
        part = symbolic_parts.setdefault(part_name, {"part_name": part_name})
        if "part_state" in end_state:
            part["current_state"] = end_state["part_state"]
        elif action_type == "acquire_part":
            part["current_state"] = "in_gripper"
        elif action_type == "release_part":
            goal_location = str(part.get("goal_location") or "").strip()
            resolved_target = str(end_state.get("part_location") or target_ref or "").strip()
            part["current_state"] = "placed" if goal_location and resolved_target == goal_location else "misplaced"
        if "part_location" in end_state:
            part["current_location"] = end_state["part_location"] or None
        elif action_type == "acquire_part" and resource_jid:
            part["current_location"] = f"{resource_jid}_gripper"
        elif action_type == "release_part":
            part["current_location"] = target_ref or None
        if "part_holder_resource_jid" in end_state:
            part["current_holder_resource_jid"] = (
                end_state["part_holder_resource_jid"] or None
            )
        elif action_type == "acquire_part":
            part["current_holder_resource_jid"] = resource_jid or None
        elif action_type == "release_part":
            part["current_holder_resource_jid"] = None
        elif "held_part" in end_state:
            part["current_holder_resource_jid"] = resource_jid if end_state["held_part"] == part_name else None
        if "held_part" in end_state and end_state["held_part"] == part_name and "part_location" not in end_state:
            part["current_location"] = f"{resource_jid}_gripper"
        elif (
            "held_part" in end_state
            and end_state["held_part"] in (None, "")
            and "part_holder_resource_jid" not in end_state
        ):
            part["current_holder_resource_jid"] = None

    session_state["symbolic_resources"] = symbolic_resources
    session_state["symbolic_parts"] = symbolic_parts


async def _handle_outline_single_pass(
    *,
    session_state: dict[str, Any],
    parsed_response: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    """Single-pass: LLM proposes all tasks at once, accept without validation."""
    turn_entry: dict[str, Any] = {}

    outline_tasks = [
        dict(row) for row in (parsed_response.get("outline_tasks") or [])
        if isinstance(row, dict)
    ]
    turn_entry["outline_tasks"] = deepcopy(outline_tasks)

    if not outline_tasks:
        turn_entry["error"] = "outline response missing outline_tasks"
        _logger.warning("[MultiTurnV2] outline single_pass: no outline_tasks")
        return "need_revision", turn_entry

    session_state["accepted_outline_prefix"] = deepcopy(outline_tasks)

    _logger.info(
        "[MultiTurnV2] outline single_pass: accepted %d tasks",
        len(outline_tasks),
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

    next_task = dict(parsed_response.get("next_task") or {})
    lookahead_tasks = [
        dict(row) for row in (parsed_response.get("lookahead_tasks") or [])
        if isinstance(row, dict)
    ]

    turn_entry["next_task"] = deepcopy(next_task)
    if lookahead_tasks:
        turn_entry["lookahead_tasks"] = deepcopy(lookahead_tasks)

    if not next_task or not str(next_task.get("outline_id") or "").strip():
        turn_entry["error"] = "outline response missing next_task with outline_id"
        _logger.warning("[MultiTurnV2] outline incremental: no next_task")
        return "need_revision", turn_entry

    accepted_prefix = list(session_state.get("accepted_outline_prefix") or [])
    accepted_prefix.append(deepcopy(next_task))
    session_state["accepted_outline_prefix"] = accepted_prefix
    session_state["outline_lookahead"] = deepcopy(lookahead_tasks)

    # Track symbolic state even in non-validated mode
    _apply_task_effects_to_symbolic_state(next_task, session_state)

    # Detect outline completion: if no lookahead remaining, the LLM
    # considers this the final recovery task → outline is ready.
    outline_complete = not lookahead_tasks
    decision = "outline_ready" if outline_complete else "need_next_task"

    _logger.info(
        "[MultiTurnV2] outline incremental: accepted task %s (prefix now %d tasks, complete=%s)",
        str(next_task.get("outline_id") or "").strip(),
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

    next_task = dict(parsed_response.get("next_task") or {})
    lookahead_tasks = [
        dict(row) for row in (parsed_response.get("lookahead_tasks") or [])
        if isinstance(row, dict)
    ]

    turn_entry["next_task"] = deepcopy(next_task)
    if lookahead_tasks:
        turn_entry["lookahead_tasks"] = deepcopy(lookahead_tasks)

    if not next_task or not str(next_task.get("outline_id") or "").strip():
        turn_entry["error"] = "outline response missing next_task with outline_id"
        _logger.warning("[MultiTurnV2] outline incremental_validated: no next_task")
        return "need_revision", turn_entry

    # Validate before accepting
    findings, grounded_action = _validate_single_outline_task(
        planner=planner,
        task=next_task,
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
        _logger.info(
            "[MultiTurnV2] outline incremental_validated: rejected task %s (%d findings)",
            str(next_task.get("outline_id") or "").strip(),
            len(findings),
        )
        # Pause so the operator can inspect the validation feedback before
        # the LLM re-proposes on the next turn.
        session_state["status"] = "paused_after_outline_turn"
        return "need_revision", turn_entry

    # Validation passed — accept into prefix
    accepted_prefix = list(session_state.get("accepted_outline_prefix") or [])
    accepted_prefix.append(deepcopy(next_task))
    session_state["accepted_outline_prefix"] = accepted_prefix
    session_state["outline_lookahead"] = deepcopy(lookahead_tasks)

    # Apply task effects to symbolic state for future validations
    _apply_task_effects_to_symbolic_state(next_task, session_state)
    session_state["outline_validation_findings"] = _prune_resolved_outline_validation_findings(
        list(session_state.get("outline_validation_findings") or []),
        session_state=session_state,
        prepared_bridge_request=prepared_bridge_request,
    )

    outline_complete = not lookahead_tasks
    decision = "outline_ready" if outline_complete else "need_next_task"

    _logger.info(
        "[MultiTurnV2] outline incremental_validated: accepted task %s "
        "(prefix now %d tasks, complete=%s)",
        str(next_task.get("outline_id") or "").strip(),
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

    candidate_tasks = [
        _normalize_candidate_task(
            task=dict(row),
            sequence_index=sequence_index,
            candidate_index=candidate_index,
        )
        for candidate_index, row in enumerate(parsed_response.get("candidate_tasks") or [])
        if isinstance(row, dict)
    ]
    turn_entry["candidate_tasks"] = deepcopy(candidate_tasks)

    if len(candidate_tasks) != 3:
        turn_entry["error"] = "outline response must include exactly 3 candidate_tasks"
        _logger.warning(
            "[MultiTurnV2] outline incremental_candidates_validated: expected 3 candidate_tasks, got %d",
            len(candidate_tasks),
        )
        return "need_revision", turn_entry

    candidate_evaluations: list[dict[str, Any]] = []
    valid_candidates: list[dict[str, Any]] = []
    for candidate_index, task in enumerate(candidate_tasks):
        evaluation: dict[str, Any] = {
            "candidate_index": candidate_index,
            "task": deepcopy(task),
        }

        normalized_task, schema_findings = _derive_candidate_outline_task(
            candidate_task=task,
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
        turn_entry["candidate_evaluations"] = deepcopy(candidate_evaluations)
        feedback_rows = _candidate_feedback_rows(candidate_evaluations)
        session_state["candidate_rejection_feedback"] = deepcopy(feedback_rows)
        turn_entry["candidate_rejection_feedback"] = deepcopy(feedback_rows)
        _logger.info(
            "[MultiTurnV2] outline incremental_candidates_validated: rejected all %d candidates",
            len(candidate_tasks),
        )
        session_state["status"] = "paused_after_outline_turn"
        return "need_revision", turn_entry
    turn_entry["candidate_evaluations"] = deepcopy(candidate_evaluations)

    selected = max(
        progress_candidates,
        key=lambda row: (
            int(dict(row.get("progress_detail") or {}).get("resolved_direct_blockers") or 0),
            int(dict(row.get("progress_detail") or {}).get("blocker_part_acquired") or 0)
            + int(dict(row.get("progress_detail") or {}).get("freed_resource_for_blocker") or 0),
            -int(dict(row.get("progress_detail") or {}).get("remaining_blocked_issues") or 0),
            -int(row.get("candidate_index") or 0),
        ),
    )
    selected_candidate_index = int(selected.get("candidate_index") or 0)
    selected_candidate_task = deepcopy(dict(selected.get("task") or {}))
    selected_normalized_task = deepcopy(dict(selected.get("normalized_task") or {}))
    selected_next_task = _commit_selected_candidate_task(
        task=selected_normalized_task,
        sequence_index=sequence_index,
    )
    selected_grounded_action = deepcopy(dict(selected.get("grounded_action") or {}))

    turn_entry["selected_candidate_index"] = selected_candidate_index
    turn_entry["selected_candidate_task"] = deepcopy(selected_candidate_task)
    turn_entry["selected_next_task"] = deepcopy(selected_next_task)
    turn_entry["next_task"] = deepcopy(selected_next_task)
    if selected_grounded_action:
        turn_entry["grounded_action"] = deepcopy(selected_grounded_action)

    accepted_prefix = list(session_state.get("accepted_outline_prefix") or [])
    accepted_prefix.append(deepcopy(selected_next_task))
    session_state["accepted_outline_prefix"] = accepted_prefix
    session_state["outline_lookahead"] = []
    session_state["candidate_rejection_feedback"] = []

    _apply_task_effects_to_symbolic_state(selected_next_task, session_state)
    remaining_findings, remaining_conditions = _remaining_blocked_issue_counts(
        session_state=session_state,
        prepared_bridge_request=prepared_bridge_request,
    )
    outline_complete = remaining_findings == 0 and remaining_conditions == 0
    decision = "outline_ready" if outline_complete else "need_next_task"

    _logger.info(
        "[MultiTurnV2] outline incremental_candidates_validated: selected candidate %d (%s) "
        "(progress=%d, prefix now %d tasks, complete=%s)",
        selected_candidate_index + 1,
        str(selected_next_task.get("outline_id") or "").strip(),
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


async def _handle_primitive_generation_phase(
    *,
    session_state: dict[str, Any],
    parsed_response: dict[str, Any],
    prepared_bridge_request: dict[str, Any],
    planner: Any,
) -> tuple[str, dict[str, Any]]:
    """Handle one turn of the primitive generation phase.

    Returns (decision, turn_entry).
    """
    raise NotImplementedError("primitive_generation phase handler not yet implemented")


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
    from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.prompts.multi_turn_v2 import (
        build_multi_turn_v2_phase_prompt_input,
        render_multi_turn_v2_phase_prompt,
    )

    phase = str(session_state.get("current_phase") or "grounding").strip().lower()
    llm_input = dict(prepared_bridge_request.get("llm_input") or {})
    outline_mode = str(session_state.get("outline_mode") or "incremental").strip().lower()

    world_observation_surface: dict[str, Any] | None = None
    if phase == "grounding":
        world_observation_surface = _build_world_observation_surface(prepared_bridge_request)

    current_recovery_blockers: list[dict[str, Any]] | None = None
    if phase == "outline" and outline_mode == "incremental_candidates_validated":
        current_recovery_blockers = _active_candidate_recovery_blockers(
            session_state=session_state,
            prepared_bridge_request=prepared_bridge_request,
        )

    prompt_input = build_multi_turn_v2_phase_prompt_input(
        phase=phase,
        llm_input=llm_input,
        session_state=session_state,
        bridge_resources=dict(prepared_bridge_request.get("bridge_resources") or {}),
        world_observation_surface=world_observation_surface,
        current_recovery_blockers=current_recovery_blockers,
    )
    prompt_text = render_multi_turn_v2_phase_prompt(prompt_input)
    return prompt_input, prompt_text


def _get_response_schema(phase: str, session_state: dict[str, Any]) -> dict[str, Any]:
    """Return the JSON response schema for the given phase."""
    from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.prompts.multi_turn_v2 import (
        multi_turn_v2_phase_response_schema,
    )
    outline_mode = str(session_state.get("outline_mode") or "incremental").strip().lower()
    return multi_turn_v2_phase_response_schema(phase, outline_mode=outline_mode)


# ---------------------------------------------------------------------------
# Response artifact shaping
# ---------------------------------------------------------------------------


def _normalized_response_artifact_payload(
    *,
    phase: str,
    session_state: dict[str, Any],
    parsed_response: dict[str, Any],
    turn_entry: dict[str, Any],
) -> dict[str, Any]:
    """Return the response payload to persist in per-turn artifacts."""
    def _sanitize_surface_task(task: Any) -> dict[str, Any]:
        sanitized = deepcopy(dict(task or {}))
        sanitized.pop("action_type", None)
        return sanitized

    normalized = deepcopy(parsed_response if isinstance(parsed_response, dict) else {})
    if str(phase or "").strip().lower() != "outline":
        return normalized

    outline_mode = str(session_state.get("outline_mode") or "incremental").strip().lower()
    if outline_mode != "incremental_candidates_validated":
        return normalized

    payload: dict[str, Any] = {}
    thought = str(parsed_response.get("thought") or "").strip()
    if thought:
        payload["thought"] = thought
    payload["candidate_tasks"] = [
        _sanitize_surface_task(row)
        for row in (turn_entry.get("candidate_tasks") or [])
        if isinstance(row, dict)
    ]
    if "selected_candidate_index" in turn_entry:
        payload["selected_candidate_index"] = int(turn_entry.get("selected_candidate_index") or 0)
    if isinstance(turn_entry.get("selected_next_task"), dict):
        payload["selected_next_task"] = _sanitize_surface_task(
            turn_entry.get("selected_next_task")
        )
    if isinstance(turn_entry.get("candidate_rejection_feedback"), list):
        payload["candidate_rejection_feedback"] = deepcopy(
            turn_entry.get("candidate_rejection_feedback") or []
        )
    return payload


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
        _logger.warning("[MultiTurnV2] Failed to write per-turn artifact: %s", exc)


# ---------------------------------------------------------------------------
# Main execution loop
# ---------------------------------------------------------------------------


async def execute_multi_turn_bridge(
    planner: Any,
    prepared_bridge_request: dict[str, Any],
    *,
    session_state: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Run the multi-turn v2 bridge loop.

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
            "[MultiTurnV2] Turn %d/%d | phase=%s",
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
            _logger.error("[MultiTurnV2] No handler for phase=%s", current_phase)
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
        turn_entry["raw_response"] = _normalized_response_artifact_payload(
            phase=current_phase,
            session_state=session_state,
            parsed_response=parsed_response,
            turn_entry=turn_entry,
        )
        turn_entry["decision"] = decision
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

        if session_state.get("status") in ("completed", "paused_after_outline_turn"):
            break

    # Stash session state for resume access
    prepared_bridge_request["multi_turn_session_state"] = deepcopy(session_state)

    # Turn budget exhausted
    if session_state.get("status") not in ("completed", "paused_after_outline_turn"):
        session_state["status"] = "turn_budget_exhausted"
        bridge_debug["multi_turn_session"] = deepcopy(session_state)
        bridge_debug["status"] = "turn_budget_exhausted"
        prepared_bridge_request["bridge_debug"] = deepcopy(bridge_debug)
        prepared_bridge_request["multi_turn_session_state"] = deepcopy(session_state)
        if hasattr(planner, "_set_last_bridge_debug"):
            planner._set_last_bridge_debug(bridge_debug)
        _logger.warning(
            "[MultiTurnV2] Turn budget exhausted (%d turns) in phase=%s",
            int(session_state.get("max_turns") or 0),
            str(session_state.get("current_phase") or ""),
        )
        return None

    # Build proposal from finalized session
    return deepcopy(session_state.get("proposal"))
