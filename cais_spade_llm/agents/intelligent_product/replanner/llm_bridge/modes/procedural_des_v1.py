"""Procedural DES bridge mode.

This is the control condition for hybrid DES: it shares the same grounding,
solver, validation, finalization, and budget, but its plant is generated
mechanically from grounded catalog actions and currently assigned resources.
"""

from __future__ import annotations

from copy import deepcopy
import logging
from typing import Any

from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.bridge_plant_compiler import (
    compile_plant_from_llm_response,
)
from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.modes.des_recovery_common import (
    append_revision_entry,
    build_des_session_seed,
    handle_compose_and_solve as _shared_handle_compose_and_solve,
    handle_evaluate_grounding as _shared_handle_evaluate_grounding,
    handle_finalize as _shared_handle_finalize,
    handle_validate_plan as _shared_handle_validate_plan,
    write_des_per_turn_artifact,
)

_logger = logging.getLogger(__name__)

_TRANSITIONS: dict[str, dict[str, str]] = {
    "evaluate_grounding": {
        "grounding_required": "evaluate_grounding",
        "grounding_satisfied": "domain_generation",
    },
    "domain_generation": {
        "plant_valid": "compose_and_solve",
        "plant_invalid": "domain_generation",
    },
    "compose_and_solve": {
        "solved": "validate_plan",
        "unsolvable": "domain_generation",
        "safety_blocked": "domain_generation",
    },
    "validate_plan": {
        "all_feasible": "finalize",
        "infeasible": "domain_generation",
    },
    "finalize": {
        "accepted": "finalize",
    },
}


def _transition_phase(current_phase: str, decision: str) -> str:
    return _TRANSITIONS.get(current_phase, {}).get(decision, current_phase)


def build_procedural_session_seed(
    prepared_bridge_request: dict[str, Any],
) -> dict[str, Any]:
    seed = build_des_session_seed(
        prepared_bridge_request,
        engine_name="procedural_des_v1",
    )
    seed["procedural_engine"] = "des_v1"
    return seed


def _resource_owner_alias(resource_jid: str) -> str:
    token = str(resource_jid or "").strip()
    if not token:
        return ""
    return token.split("@", 1)[0]


def _current_or_goal_location(part_row: dict[str, Any]) -> str:
    return (
        str(part_row.get("current_location") or "").strip()
        or str(part_row.get("goal_location") or "").strip()
    )


def _tool_by_owner(
    tools_catalog: list[dict[str, Any]],
    *,
    owner_alias: str,
    function_name: str,
) -> dict[str, Any] | None:
    for entry in tools_catalog:
        if not isinstance(entry, dict):
            continue
        owner = str(entry.get("function_owner_agent") or "").strip()
        fn = str(entry.get("function") or "").strip()
        if owner == owner_alias and fn == function_name:
            return deepcopy(entry)
    return None


def _mechanical_state_metadata(
    part_name: str,
    part_row: dict[str, Any],
) -> dict[str, Any]:
    goal_location = str(part_row.get("goal_location") or "").strip()
    current_location = str(part_row.get("current_location") or "").strip()
    metadata: dict[str, Any] = {"atomic_bindings": {}, "derived_state_labels": []}
    if part_name and goal_location:
        metadata["atomic_bindings"][f"AtGoal({part_name}, {goal_location})"] = (
            current_location == goal_location
        )
    if part_name:
        metadata["atomic_bindings"][f"PoseKnown({part_name})"] = bool(part_row.get("observed_pose"))
    return metadata


def _build_procedural_plant(
    prepared_bridge_request: dict[str, Any],
    session_state: dict[str, Any],
) -> dict[str, Any]:
    tools_catalog = list(prepared_bridge_request.get("tools_catalog") or [])
    symbolic_parts = dict(session_state.get("symbolic_parts") or {})
    symbolic_resources = dict(session_state.get("symbolic_resources") or {})

    states: list[str] = ["s0"]
    events: list[dict[str, Any]] = []
    state_metadata: dict[str, dict[str, Any]] = {"s0": {"atomic_bindings": {}, "derived_state_labels": []}}
    marked_state_metadata: dict[str, dict[str, Any]] = {}
    cursor = "s0"
    event_index = 0

    # Fairness constraint: procedural may not invent reassignment or handoff.
    # It only uses the currently assigned/declared resource for each part.
    for part_name, part_row in symbolic_parts.items():
        if not isinstance(part_row, dict):
            continue
        goal_location = str(part_row.get("goal_location") or "").strip()
        current_location = str(part_row.get("current_location") or "").strip()
        assigned_resource_jid = str(
            part_row.get("assigned_resource_jid")
            or part_row.get("goal_resource_jid")
            or part_row.get("resource_jid")
            or ""
        ).strip()
        if not assigned_resource_jid or not goal_location or current_location == goal_location:
            continue

        owner_alias = _resource_owner_alias(assigned_resource_jid)
        state_metadata[cursor] = _mechanical_state_metadata(part_name, part_row)

        maybe_reset = dict(symbolic_resources.get(assigned_resource_jid) or {})
        current_state = str(maybe_reset.get("current_state") or "").strip().lower()
        move_home_tool = _tool_by_owner(tools_catalog, owner_alias=owner_alias, function_name="move_home")
        if move_home_tool and current_state and current_state != "idle":
            next_state = f"s{len(states)}"
            states.append(next_state)
            event_index += 1
            events.append({
                "name": f"procedural_move_home_{event_index}",
                "from": cursor,
                "to": next_state,
                "resource_jid": assigned_resource_jid,
                "action_type": "move_home",
                "description": f"Reset {assigned_resource_jid} to idle",
                "derived_state_labels": [],
            })
            state_metadata[next_state] = _mechanical_state_metadata(part_name, part_row)
            cursor = next_state

        if not part_row.get("current_holder_resource_jid"):
            pick_approach = _tool_by_owner(tools_catalog, owner_alias=owner_alias, function_name="pick_approach")
            pick_grasp = _tool_by_owner(tools_catalog, owner_alias=owner_alias, function_name="pick_grasp")
            if pick_approach:
                next_state = f"s{len(states)}"
                states.append(next_state)
                event_index += 1
                events.append({
                    "name": f"procedural_pick_approach_{event_index}",
                    "from": cursor,
                    "to": next_state,
                    "resource_jid": assigned_resource_jid,
                    "action_type": "pick_approach",
                    "part_name": part_name,
                    "target_ref": _current_or_goal_location(part_row),
                    "description": f"Approach {part_name} with assigned resource",
                    "derived_state_labels": [],
                })
                state_metadata[next_state] = _mechanical_state_metadata(part_name, part_row)
                cursor = next_state
            if pick_grasp:
                next_state = f"s{len(states)}"
                states.append(next_state)
                event_index += 1
                events.append({
                    "name": f"procedural_pick_grasp_{event_index}",
                    "from": cursor,
                    "to": next_state,
                    "resource_jid": assigned_resource_jid,
                    "action_type": "pick_grasp",
                    "part_name": part_name,
                    "target_ref": "observed_pose" if part_row.get("observed_pose") else current_location,
                    "description": f"Pick {part_name} with assigned resource",
                    "derived_state_labels": [],
                })
                state_metadata[next_state] = _mechanical_state_metadata(part_name, part_row)
                cursor = next_state

        place_approach = _tool_by_owner(tools_catalog, owner_alias=owner_alias, function_name="place_approach")
        place_insert = _tool_by_owner(tools_catalog, owner_alias=owner_alias, function_name="place_insert")
        if place_approach:
            next_state = f"s{len(states)}"
            states.append(next_state)
            event_index += 1
            events.append({
                "name": f"procedural_place_approach_{event_index}",
                "from": cursor,
                "to": next_state,
                "resource_jid": assigned_resource_jid,
                "action_type": "place_approach",
                "part_name": part_name,
                "target_ref": goal_location,
                "description": f"Carry {part_name} toward its goal location",
                "derived_state_labels": [],
            })
            state_metadata[next_state] = _mechanical_state_metadata(part_name, part_row)
            cursor = next_state
        if place_insert:
            next_state = f"s{len(states)}"
            states.append(next_state)
            event_index += 1
            events.append({
                "name": f"procedural_place_insert_{event_index}",
                "from": cursor,
                "to": next_state,
                "resource_jid": assigned_resource_jid,
                "action_type": "place_insert",
                "part_name": part_name,
                "target_ref": goal_location,
                "description": f"Place {part_name} at its goal location",
                "derived_state_labels": [],
            })
            state_metadata[next_state] = {
                "atomic_bindings": {f"AtGoal({part_name}, {goal_location})": True},
                "derived_state_labels": [],
            }
            cursor = next_state

    marked = [cursor]
    marked_state_metadata[cursor] = {
        "marking_predicate": "mechanically recovered assigned tasks are back at their goal locations"
    }
    return {
        "states": states,
        "initial": "s0",
        "marked": marked,
        "state_metadata": state_metadata,
        "marked_state_metadata": marked_state_metadata,
        "events": events,
    }


async def _handle_evaluate_grounding(
    *,
    session_state: dict[str, Any],
    parsed_response: dict[str, Any],
    prepared_bridge_request: dict[str, Any],
    planner: Any,
) -> tuple[str, dict[str, Any]]:
    del parsed_response, planner
    session_state["grounding_checkpoint"] = {
        "symbolic_parts": deepcopy(session_state.get("symbolic_parts") or {}),
        "symbolic_resources": deepcopy(session_state.get("symbolic_resources") or {}),
        "current_phase": str(session_state.get("current_phase") or ""),
        "turn_index": int(session_state.get("turn_index") or 0),
    }
    return await _shared_handle_evaluate_grounding(
        session_state=session_state,
        prepared_bridge_request=prepared_bridge_request,
        engine_name="procedural",
        session_state_key="procedural_session_state",
    )


async def _handle_procedural_domain_generation(
    *,
    session_state: dict[str, Any],
    parsed_response: dict[str, Any],
    prepared_bridge_request: dict[str, Any],
    planner: Any,
) -> tuple[str, dict[str, Any]]:
    del parsed_response, planner
    plant_response = _build_procedural_plant(prepared_bridge_request, session_state)
    compile_result = compile_plant_from_llm_response(
        plant_response,
        dict(prepared_bridge_request.get("bridge_resources") or {}),
        dict(session_state.get("symbolic_parts") or {}),
    )
    turn_entry: dict[str, Any] = {
        "compile_result_status": compile_result["status"],
        "plant_findings": deepcopy(compile_result.get("findings") or []),
    }
    if compile_result["status"] == "valid":
        session_state["current_plant"] = deepcopy(compile_result["plant"])
        session_state["plant_findings"] = []
        return "plant_valid", turn_entry
    session_state["plant_findings"] = deepcopy(compile_result.get("findings") or [])
    session_state["domain_revision_count"] = int(session_state.get("domain_revision_count") or 0) + 1
    append_revision_entry(
        session_state,
        plant=compile_result.get("plant") or plant_response,
        plant_findings=compile_result.get("findings") or [],
    )
    return "plant_invalid", turn_entry


async def _handle_compose_and_solve(
    *,
    session_state: dict[str, Any],
    parsed_response: dict[str, Any],
    prepared_bridge_request: dict[str, Any],
    planner: Any,
) -> tuple[str, dict[str, Any]]:
    del parsed_response
    decision, turn_entry = await _shared_handle_compose_and_solve(
        session_state=session_state,
        prepared_bridge_request=prepared_bridge_request,
        planner=planner,
    )
    if decision in {"unsolvable", "safety_blocked"}:
        append_revision_entry(
            session_state,
            plant=session_state.get("current_plant"),
            solver_result=session_state.get("solver_result"),
        )
    return decision, turn_entry


async def _handle_validate_plan(
    *,
    session_state: dict[str, Any],
    parsed_response: dict[str, Any],
    prepared_bridge_request: dict[str, Any],
    planner: Any,
) -> tuple[str, dict[str, Any]]:
    del parsed_response
    decision, turn_entry = await _shared_handle_validate_plan(
        session_state=session_state,
        prepared_bridge_request=prepared_bridge_request,
        planner=planner,
    )
    if decision == "infeasible":
        append_revision_entry(
            session_state,
            plant=session_state.get("current_plant"),
            solver_result=session_state.get("solver_result"),
            feasibility_findings=session_state.get("feasibility_findings"),
        )
    return decision, turn_entry


async def _handle_finalize(
    *,
    session_state: dict[str, Any],
    parsed_response: dict[str, Any],
    prepared_bridge_request: dict[str, Any],
    planner: Any,
) -> tuple[str, dict[str, Any]]:
    del parsed_response, planner
    return await _shared_handle_finalize(
        session_state=session_state,
        prepared_bridge_request=prepared_bridge_request,
        engine_name="procedural_des_v1",
    )


_PHASE_HANDLERS: dict[str, Any] = {
    "evaluate_grounding": _handle_evaluate_grounding,
    "domain_generation": _handle_procedural_domain_generation,
    "compose_and_solve": _handle_compose_and_solve,
    "validate_plan": _handle_validate_plan,
    "finalize": _handle_finalize,
}


async def execute_procedural_des_bridge(
    planner: Any,
    prepared_bridge_request: dict[str, Any],
    *,
    session_state: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    if session_state is not None:
        session_state = deepcopy(session_state)
    else:
        session_state = deepcopy(
            prepared_bridge_request.get("procedural_session_seed")
            or build_procedural_session_seed(prepared_bridge_request)
        )
    session_state["status"] = "running"

    bridge_debug = deepcopy(prepared_bridge_request.get("bridge_debug") or {})
    bridge_debug["procedural_session"] = deepcopy(session_state)
    prepared_bridge_request["bridge_debug"] = deepcopy(bridge_debug)
    if hasattr(planner, "_set_last_bridge_debug"):
        planner._set_last_bridge_debug(bridge_debug)

    max_turns = int(session_state.get("max_turns") or 6)
    while int(session_state.get("turn_index") or 0) < max_turns:
        session_state["turn_index"] = int(session_state.get("turn_index") or 0) + 1
        current_phase = str(session_state.get("current_phase") or "domain_generation").strip().lower()
        turn_idx = int(session_state.get("turn_index") or 0)

        handler = _PHASE_HANDLERS.get(current_phase)
        if handler is None:
            session_state["status"] = "error"
            break

        decision, turn_entry = await handler(
            session_state=session_state,
            parsed_response={},
            prepared_bridge_request=prepared_bridge_request,
            planner=planner,
        )

        turn_entry["turn_index"] = turn_idx
        turn_entry["phase"] = current_phase
        turn_entry["decision"] = decision
        session_state.setdefault("turns", []).append(deepcopy(turn_entry))
        session_state["current_phase"] = _transition_phase(current_phase, decision)

        if current_phase == "finalize" and decision == "accepted":
            session_state["status"] = "completed"

        bridge_debug["procedural_session"] = deepcopy(session_state)
        bridge_debug["status"] = str(session_state.get("status") or "running")
        prepared_bridge_request["bridge_debug"] = deepcopy(bridge_debug)
        if hasattr(planner, "_set_last_bridge_debug"):
            planner._set_last_bridge_debug(bridge_debug)

        is_final_turn = session_state.get("status") in ("completed", "paused_after_grounding")
        write_des_per_turn_artifact(
            prepared_bridge_request,
            session_state,
            phase_label=str(session_state.get("current_phase") or "procedural"),
            debug_session_key="procedural_session",
            write_session_transcript=is_final_turn,
        )
        if is_final_turn:
            break

    prepared_bridge_request["procedural_session_state"] = deepcopy(session_state)
    if session_state.get("status") not in ("completed", "paused_after_grounding"):
        session_state["status"] = "turn_budget_exhausted"
        prepared_bridge_request["procedural_session_state"] = deepcopy(session_state)
        bridge_debug["procedural_session"] = deepcopy(session_state)
        bridge_debug["status"] = "turn_budget_exhausted"
        prepared_bridge_request["bridge_debug"] = deepcopy(bridge_debug)
        if hasattr(planner, "_set_last_bridge_debug"):
            planner._set_last_bridge_debug(bridge_debug)
        write_des_per_turn_artifact(
            prepared_bridge_request,
            session_state,
            phase_label=str(session_state.get("current_phase") or "procedural"),
            debug_session_key="procedural_session",
            write_session_transcript=True,
        )
        return None
    if session_state.get("status") == "paused_after_grounding":
        return None
    return deepcopy(session_state.get("proposal"))


__all__ = [
    "build_procedural_session_seed",
    "execute_procedural_des_bridge",
]
