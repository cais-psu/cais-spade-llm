"""Multi-turn executor for the active v4 bridge."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import inspect
import json
from typing import Any
from uuid import uuid4

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
        "need_grounding": "grounding",
        "outline_ready": "primitive_generation",
    },
    "primitive_generation": {
        "need_grounding": "grounding",
        "need_outline_revision": "outline",
        "draft_ready": "finalize",
    },
    "finalize": {
        "need_grounding": "grounding",
        "need_outline_revision": "outline",
        "need_primitive_revision": "primitive_generation",
        "final_ready": "finalize",
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
    return {
        "session_id": session_id,
        "status": "prepared_for_llm",
        "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
        "current_phase": "grounding",
        "turn_index": 0,
        "max_turns": max_turns,
        "max_observations": max_observations,
        "max_observe_batch": max_observe_batch,
        "observation_count": 0,
        "observation_store": {},
        "observation_history": [],
        "accepted_outline": None,
        "proposal_draft": None,
        "phase_feedback": [],
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


def _build_observation_surface(
    prepared_bridge_request: dict[str, Any],
) -> dict[str, Any]:
    llm_input = dict(prepared_bridge_request.get("llm_input") or {})
    allowed_surface = dict(llm_input.get("allowed_execution_surface") or {})
    allowed_by_jid = {
        str(row.get("resource_jid") or "").strip(): dict(row)
        for row in (allowed_surface.get("resources") or [])
        if isinstance(row, dict) and str(row.get("resource_jid") or "").strip()
    }
    bridge_resources = dict(prepared_bridge_request.get("bridge_resources") or {})
    observation_resources: list[dict[str, Any]] = []
    for resource_jid, bridge_entry in bridge_resources.items():
        resource_key = str(resource_jid or "").strip()
        if not resource_key:
            continue
        allowed_entry = dict(allowed_by_jid.get(resource_key) or {})
        resource_type = str(
            bridge_entry.get("resource_type")
            or allowed_entry.get("resource_type")
            or "resource"
        ).strip()
        observation_primitives: list[dict[str, Any]] = []
        for primitive_entry in (bridge_entry.get("primitive_catalog") or []):
            if not isinstance(primitive_entry, dict):
                continue
            primitive_name = str(primitive_entry.get("name") or "").strip()
            if not primitive_name:
                continue
            if dict(primitive_entry.get("effects") or {}):
                continue
            if not _supports_store_as(resource_type=resource_type, primitive_name=primitive_name):
                continue
            observation_row: dict[str, Any] = {
                "name": primitive_name,
                "required_params": [
                    str(param).strip()
                    for param in (primitive_entry.get("required_params") or [])
                    if str(param).strip()
                ],
                "supports_store_as": True,
            }
            description = str(
                primitive_entry.get("description") or primitive_entry.get("semantic_summary") or ""
            ).strip()
            if description:
                observation_row["description"] = description
            output_fields = [
                str(field).strip()
                for field in dict(primitive_entry.get("output_schema") or {}).keys()
                if str(field).strip()
            ]
            if output_fields:
                observation_row["output_fields"] = output_fields
            observation_primitives.append(observation_row)
        if not observation_primitives:
            continue
        observation_resources.append(
            {
                "resource_jid": resource_key,
                "resource_type": resource_type,
                "role": deepcopy(allowed_entry.get("role")),
                "allowed_observation_primitives": observation_primitives,
            }
        )
    return {
        "resources": observation_resources,
    }


def _build_compact_resources(prepared_bridge_request: dict[str, Any]) -> list[dict[str, Any]]:
    llm_input = dict(prepared_bridge_request.get("llm_input") or {})
    allowed_surface = dict(llm_input.get("allowed_execution_surface") or {})
    return [
        {
            "resource_jid": str(row.get("resource_jid") or "").strip(),
            "resource_type": deepcopy(row.get("resource_type")),
            "role": deepcopy(row.get("role")),
        }
        for row in (allowed_surface.get("resources") or [])
        if isinstance(row, dict) and str(row.get("resource_jid") or "").strip()
    ]


def _build_phase_prompt_artifacts(
    prepared_bridge_request: dict[str, Any],
    session_state: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    phase = str(session_state.get("current_phase") or "grounding").strip().lower()
    prompt_input = build_multi_turn_phase_prompt_input(
        phase=phase,
        llm_input=deepcopy(prepared_bridge_request.get("llm_input") or {}),
        session_state=deepcopy(session_state),
        observation_surface=_build_observation_surface(prepared_bridge_request),
        compact_resources=_build_compact_resources(prepared_bridge_request),
    )
    prompt_text = render_multi_turn_phase_prompt(prompt_input)
    return prompt_input, prompt_text


def _lookup_observation_primitive(
    prepared_bridge_request: dict[str, Any],
    *,
    resource_jid: str,
    primitive_name: str,
) -> tuple[dict[str, Any], str]:
    bridge_resources = dict(prepared_bridge_request.get("bridge_resources") or {})
    bridge_entry = dict(bridge_resources.get(resource_jid) or {})
    resource_type = str(bridge_entry.get("resource_type") or "resource").strip() or "resource"
    for primitive_entry in (bridge_entry.get("primitive_catalog") or []):
        if not isinstance(primitive_entry, dict):
            continue
        if str(primitive_entry.get("name") or "").strip() != primitive_name:
            continue
        if dict(primitive_entry.get("effects") or {}):
            break
        if not _supports_store_as(resource_type=resource_type, primitive_name=primitive_name):
            break
        return primitive_entry, resource_type
    return {}, resource_type


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

    for raw_request in observe_requests:
        request = dict(raw_request or {})
        resource_jid = str(request.get("resource_jid") or "").strip()
        primitive_name = str(request.get("primitive") or "").strip()
        store_as = str(request.get("store_as") or "").strip()
        params = dict(request.get("params") or {})
        if not resource_jid or not primitive_name or not store_as:
            return [], "observe_requests must include resource_jid, primitive, and store_as"
        if store_as in seen_aliases:
            return [], f"duplicate observe store_as alias: {store_as!r}"

        primitive_entry, resource_type = _lookup_observation_primitive(
            prepared_bridge_request,
            resource_jid=resource_jid,
            primitive_name=primitive_name,
        )
        if not primitive_entry:
            return [], (
                f"observation primitive {primitive_name!r} is not allowed for "
                f"{resource_jid!r}"
            )

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
            "resource_jid": resource_jid,
            "primitive": primitive_name,
            "params": deepcopy(params),
            "store_as": store_as,
            "output": deepcopy(extracted_output),
        }
        results.append(observation_row)
        seen_aliases.add(store_as)
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
        prompt_input, prompt_text = _build_phase_prompt_artifacts(
            prepared_bridge_request,
            session_state,
        )
        raw_response = await ask_llm_structured(
            prompt=prompt_text,
            response_format=multi_turn_phase_response_schema(current_phase),
        )
        parsed_response = deepcopy(raw_response if isinstance(raw_response, dict) else {})
        decision = str(parsed_response.get("decision") or "").strip().lower()
        thought = str(parsed_response.get("thought") or "").strip()
        turn_entry: dict[str, Any] = {
            "turn_index": int(session_state.get("turn_index") or 0),
            "phase": current_phase,
            "prompt_input": deepcopy(prompt_input),
            "prompt_text": prompt_text,
            "raw_response": deepcopy(parsed_response),
            "decision": decision,
            "thought": thought,
        }

        try:
            next_phase = transition_multi_turn_phase(current_phase, decision)
        except ValueError as exc:
            session_state["status"] = "invalid_phase_decision"
            turn_entry["error"] = str(exc)
            _append_turn(planner, prepared_bridge_request, session_state, turn_entry)
            return None

        if current_phase == "grounding":
            observe_requests = [
                deepcopy(row)
                for row in (parsed_response.get("observe_requests") or [])
                if isinstance(row, dict)
            ]
            sufficient_grounding = bool(parsed_response.get("sufficient_grounding"))
            observe_reason = str(parsed_response.get("observe_reason") or "").strip()
            turn_entry["observe_requests"] = deepcopy(observe_requests)
            if observe_reason:
                turn_entry["observe_reason"] = observe_reason
            if decision == "observe":
                if sufficient_grounding:
                    session_state["status"] = "invalid_observe_request"
                    turn_entry["error"] = (
                        "grounding decision 'observe' is inconsistent with "
                        "sufficient_grounding=true"
                    )
                    _append_turn(planner, prepared_bridge_request, session_state, turn_entry)
                    return None
                if not observe_reason:
                    session_state["status"] = "invalid_observe_request"
                    turn_entry["error"] = "grounding decision 'observe' requires observe_reason"
                    _append_turn(planner, prepared_bridge_request, session_state, turn_entry)
                    return None
                if not observe_requests:
                    session_state["status"] = "invalid_observe_request"
                    turn_entry["error"] = "grounding decision 'observe' requires observe_requests"
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
                    session_state["status"] = "observe_execution_failed"
                    turn_entry["error"] = observe_error
                    _append_turn(planner, prepared_bridge_request, session_state, turn_entry)
                    return None
                for result in observation_results:
                    alias = str(result.get("store_as") or "").strip()
                    session_state["observation_store"][alias] = deepcopy(
                        result.get("output") or {}
                    )
                    session_state["observation_history"].append(
                        {
                            "turn_index": int(session_state.get("turn_index") or 0),
                            "phase": current_phase,
                            **deepcopy(result),
                        }
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
                            "observe_reason": observe_reason,
                            "observe_requests": observe_requests,
                            "observation_results": observation_results,
                        },
                    )
                )
            else:
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
                        detail=parsed_response.get("blocking_summary"),
                    )
                )

        elif current_phase == "outline":
            outline_tasks = [
                deepcopy(row)
                for row in (parsed_response.get("outline_tasks") or [])
                if isinstance(row, dict)
            ]
            turn_entry["outline_tasks"] = deepcopy(outline_tasks)
            session_state["phase_feedback"].append(
                _build_phase_feedback(
                    phase=current_phase,
                    decision=decision,
                    thought=thought,
                    detail=outline_tasks,
                )
            )
            if decision == "outline_ready":
                session_state["accepted_outline"] = deepcopy(outline_tasks)

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
    prepared_bridge_request["multi_turn_session_result"] = deepcopy(session_state)
    bridge_debug = deepcopy(prepared_bridge_request.get("bridge_debug") or {})
    bridge_debug["status"] = "turn_budget_exhausted"
    bridge_debug["multi_turn_session"] = deepcopy(session_state)
    prepared_bridge_request["bridge_debug"] = deepcopy(bridge_debug)
    if hasattr(planner, "_set_last_bridge_debug"):
        planner._set_last_bridge_debug(bridge_debug)
    return None
