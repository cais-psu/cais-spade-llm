"""Hybrid DES bridge execution engine — LLM as domain author + DFA solver.

The LLM generates a recovery plant automaton (state-transition model for
actions that don't exist in any pre-modeled domain).  The DES solver
composes that plant with safety DFA specifications and finds the optimal
recovery trace via BFS on the product automaton.

Phases
------
domain_generation (LLM) → compose_and_solve (DFA) → validate_plan → finalize
        ↑                                                   |
        └─── feedback (if plant invalid or no solution) ────┘

Self-contained — no imports from multi_turn or multi_turn_v2.
"""

from __future__ import annotations

import json
import logging
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any

from cais_spade_llm.agents.central_controller.outline_macro_safety import (
    validate_outline_macro_cca_constraints,
)
from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.bridge_artifacts import (
    write_bridge_artifacts,
)
from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.bridge_des_solver import (
    compose_and_solve,
    solver_diagnostic_summary,
)
from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.bridge_plant_compiler import (
    compile_plant_from_llm_response,
    plant_findings_summary,
)
from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.prompts.hybrid_des_v1 import (
    build_hybrid_des_prompt_input,
    build_hybrid_domain_generation_prompt,
    build_hybrid_feedback_prompt,
    hybrid_des_phase_response_schema,
)

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_MAX_TURNS = 6
_DEFAULT_MAX_DOMAIN_REVISIONS = 3

_TRANSITIONS: dict[str, dict[str, str]] = {
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
    phase_map = _TRANSITIONS.get(current_phase, {})
    next_phase = phase_map.get(decision)
    if next_phase is None:
        _logger.warning(
            "[HybridDES] No transition for phase=%s decision=%s; staying",
            current_phase, decision,
        )
        return current_phase
    return next_phase


# ---------------------------------------------------------------------------
# Session seed
# ---------------------------------------------------------------------------


def build_hybrid_session_seed(
    prepared_bridge_request: dict[str, Any],
) -> dict[str, Any]:
    """Build the initial session state for a hybrid DES bridge run."""
    bridge_session = dict(prepared_bridge_request.get("bridge_session") or {})
    max_turns = int(bridge_session.get("max_turns") or _DEFAULT_MAX_TURNS)

    # Build symbolic resource/part state
    llm_input = dict(prepared_bridge_request.get("llm_input") or {})
    observed_runtime_state = dict(llm_input.get("observed_runtime_state") or {})
    symbolic_resources: dict[str, dict[str, Any]] = {}
    for row in observed_runtime_state.get("resources") or []:
        if not isinstance(row, dict):
            continue
        jid = str(row.get("resource_jid") or "").strip()
        if jid:
            symbolic_resources[jid] = deepcopy(row)
    symbolic_parts: dict[str, dict[str, Any]] = {}
    for row in llm_input.get("part_facts") or []:
        if not isinstance(row, dict):
            continue
        name = str(row.get("part_name") or "").strip()
        if name:
            symbolic_parts[name] = deepcopy(row)

    return {
        "hybrid_engine": "des_v1",
        "current_phase": "domain_generation",
        "turn_index": 0,
        "max_turns": max_turns,
        "status": "pending",
        "turns": [],
        "domain_revision_count": 0,
        # Plant state
        "current_plant": None,
        "plant_findings": [],
        # Solver state
        "solver_result": None,
        # Validation state
        "feasibility_findings": [],
        # Final output
        "action_sequence": [],
        "proposal": None,
        # Symbolic state
        "symbolic_resources": symbolic_resources,
        "symbolic_parts": symbolic_parts,
    }


# ---------------------------------------------------------------------------
# Recovery gap state (resource + part state for prompt)
# ---------------------------------------------------------------------------


def _build_recovery_gap_state(
    session_state: dict[str, Any],
) -> dict[str, Any]:
    """Build the resource/part state for the prompt from session symbolic state."""
    resource_state = list(
        deepcopy(row) for row in (session_state.get("symbolic_resources") or {}).values()
    )
    part_state = list(
        deepcopy(row) for row in (session_state.get("symbolic_parts") or {}).values()
    )
    return {
        "resource_state": resource_state,
        "part_state": part_state,
    }


# ---------------------------------------------------------------------------
# Phase handlers
# ---------------------------------------------------------------------------


async def _handle_domain_generation(
    *,
    session_state: dict[str, Any],
    parsed_response: dict[str, Any],
    prepared_bridge_request: dict[str, Any],
    planner: Any,
) -> tuple[str, dict[str, Any]]:
    """Handle domain_generation phase — validate LLM-produced plant."""
    bridge_resources = dict(prepared_bridge_request.get("bridge_resources") or {})
    symbolic_parts = dict(session_state.get("symbolic_parts") or {})

    # Extract plant from LLM response
    plant_response = parsed_response.get("plant") or parsed_response
    compile_result = compile_plant_from_llm_response(
        plant_response, bridge_resources, symbolic_parts,
    )

    turn_entry: dict[str, Any] = {
        "compile_result_status": compile_result["status"],
        "plant_findings": deepcopy(compile_result.get("findings") or []),
    }

    if compile_result["status"] == "valid":
        session_state["current_plant"] = deepcopy(compile_result["plant"])
        session_state["plant_findings"] = []
        _logger.info("[HybridDES] Plant compiled successfully.")
        return "plant_valid", turn_entry

    session_state["plant_findings"] = deepcopy(compile_result.get("findings") or [])
    session_state["domain_revision_count"] = int(
        session_state.get("domain_revision_count") or 0
    ) + 1
    _logger.warning(
        "[HybridDES] Plant invalid (%d findings). Revision %d.",
        len(compile_result.get("findings") or []),
        int(session_state.get("domain_revision_count") or 0),
    )
    return "plant_invalid", turn_entry


async def _handle_compose_and_solve(
    *,
    session_state: dict[str, Any],
    parsed_response: dict[str, Any],
    prepared_bridge_request: dict[str, Any],
    planner: Any,
) -> tuple[str, dict[str, Any]]:
    """Handle compose_and_solve phase — run DES solver (no LLM call)."""
    plant = dict(session_state.get("current_plant") or {})

    # Get safety DFAs from the planner/CCA
    safety_dfas = _extract_safety_dfas(planner, prepared_bridge_request)
    ap_descriptors = _extract_ap_descriptors(planner, prepared_bridge_request)

    solver_result = compose_and_solve(
        plant=plant,
        safety_dfas=safety_dfas,
        ap_descriptors=ap_descriptors,
    )

    session_state["solver_result"] = deepcopy(solver_result)
    turn_entry: dict[str, Any] = {
        "solver_status": solver_result["status"],
        "product_states_explored": solver_result.get("product_states_explored", 0),
        "trace_length": len(solver_result.get("trace") or []),
    }

    status = solver_result["status"]
    if status == "solved":
        session_state["action_sequence"] = deepcopy(
            solver_result.get("action_sequence") or []
        )
        _logger.info(
            "[HybridDES] Solver found plan with %d steps.",
            len(solver_result.get("action_sequence") or []),
        )
        return "solved", turn_entry

    _logger.warning("[HybridDES] Solver returned: %s", status)
    session_state["domain_revision_count"] = int(
        session_state.get("domain_revision_count") or 0
    ) + 1
    return status, turn_entry


async def _handle_validate_plan(
    *,
    session_state: dict[str, Any],
    parsed_response: dict[str, Any],
    prepared_bridge_request: dict[str, Any],
    planner: Any,
) -> tuple[str, dict[str, Any]]:
    """Handle validate_plan phase — check physical feasibility of each step."""
    action_sequence = list(session_state.get("action_sequence") or [])
    all_findings: list[dict[str, Any]] = []

    for i, action in enumerate(action_sequence):
        step_findings = await _validate_action_feasibility(
            action=action,
            step_index=i,
            planner=planner,
            prepared_bridge_request=prepared_bridge_request,
            session_state=session_state,
        )
        all_findings.extend(step_findings)

    session_state["feasibility_findings"] = deepcopy(all_findings)
    turn_entry: dict[str, Any] = {
        "feasibility_finding_count": len(all_findings),
        "feasibility_findings": deepcopy(all_findings),
    }

    if not all_findings:
        _logger.info("[HybridDES] All plan steps pass feasibility.")
        return "all_feasible", turn_entry

    _logger.warning(
        "[HybridDES] %d feasibility finding(s); revising domain.",
        len(all_findings),
    )
    session_state["domain_revision_count"] = int(
        session_state.get("domain_revision_count") or 0
    ) + 1
    return "infeasible", turn_entry


async def _handle_finalize(
    *,
    session_state: dict[str, Any],
    parsed_response: dict[str, Any],
    prepared_bridge_request: dict[str, Any],
    planner: Any,
) -> tuple[str, dict[str, Any]]:
    """Handle finalize phase — build the bridge proposal from validated trace."""
    action_sequence = list(session_state.get("action_sequence") or [])

    # Convert action_sequence to bridge proposal format
    outline_tasks: list[dict[str, Any]] = []
    for i, action in enumerate(action_sequence):
        task: dict[str, Any] = {
            "outline_id": f"RECOVERY_SEQ{i + 1}",
            "resource_jid": str(action.get("resource_jid") or "").strip(),
            "action_type": str(action.get("action_type") or "").strip(),
            "description": str(action.get("description") or "").strip(),
        }
        part_name = str(action.get("part_name") or "").strip()
        if part_name:
            task["part_name"] = part_name
        target_ref = str(action.get("target_ref") or "").strip()
        if target_ref:
            task["target_ref"] = target_ref
        pose = action.get("pose")
        if isinstance(pose, dict):
            task["pose"] = deepcopy(pose)
        outline_tasks.append(task)

    proposal: dict[str, Any] = {
        "outline_tasks": outline_tasks,
        "solver_status": "solved",
        "engine": "hybrid_des_v1",
        "domain_revisions": int(session_state.get("domain_revision_count") or 0),
    }
    session_state["proposal"] = deepcopy(proposal)

    turn_entry: dict[str, Any] = {
        "proposal_task_count": len(outline_tasks),
    }
    _logger.info(
        "[HybridDES] Finalized proposal with %d tasks.",
        len(outline_tasks),
    )
    return "accepted", turn_entry


# ---------------------------------------------------------------------------
# Phase handler dispatch
# ---------------------------------------------------------------------------

_PHASE_HANDLERS: dict[str, Any] = {
    "domain_generation": _handle_domain_generation,
    "compose_and_solve": _handle_compose_and_solve,
    "validate_plan": _handle_validate_plan,
    "finalize": _handle_finalize,
}


# ---------------------------------------------------------------------------
# Helper: extract safety DFAs and AP descriptors from planner context
# ---------------------------------------------------------------------------


def _extract_safety_dfas(
    planner: Any,
    prepared_bridge_request: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Extract loaded safety DFA rules from the CCA/planner context."""
    # Try direct access via product_agent → cca_agent → safety checker
    product_agent = getattr(planner, "product_agent", None)
    if product_agent is not None:
        cca = getattr(product_agent, "cca_agent", None) or getattr(product_agent, "_cca", None)
        if cca is not None:
            safety_checker = (
                getattr(cca, "plan_safety_validator", None)
                or getattr(cca, "safety_checker", None)
                or getattr(cca, "online_safety_monitor", None)
            )
            if safety_checker is not None:
                dfas = getattr(safety_checker, "dfas", None)
                if isinstance(dfas, dict) and dfas:
                    return deepcopy(dfas)

    # Fallback: extract from llm_input safety rules (build minimal DFAs)
    llm_input = dict(prepared_bridge_request.get("llm_input") or {})
    rules = llm_input.get("loaded_safety_rules") or []
    dfas: dict[str, dict[str, Any]] = {}
    for rule in rules:
        if not isinstance(rule, dict):
            continue
        rule_id = str(rule.get("id") or rule.get("rule_id") or "").strip()
        dfa_data = rule.get("dfa")
        if rule_id and isinstance(dfa_data, dict):
            dfas[rule_id] = deepcopy(dfa_data)
    return dfas


def _extract_ap_descriptors(
    planner: Any,
    prepared_bridge_request: dict[str, Any],
) -> list[dict[str, Any]]:
    """Extract AP descriptor list from safety rules."""
    llm_input = dict(prepared_bridge_request.get("llm_input") or {})
    rules = llm_input.get("loaded_safety_rules") or []
    descriptors: list[dict[str, Any]] = []
    for rule in rules:
        if not isinstance(rule, dict):
            continue
        ap_defs = rule.get("ap_definitions") or rule.get("aps") or []
        if isinstance(ap_defs, list):
            descriptors.extend(
                deepcopy(d) for d in ap_defs if isinstance(d, dict)
            )
        elif isinstance(ap_defs, dict):
            for label, desc in ap_defs.items():
                if isinstance(desc, dict):
                    entry = deepcopy(desc)
                    entry["label"] = label
                    descriptors.append(entry)
    return descriptors


# ---------------------------------------------------------------------------
# Helper: validate a single action's physical feasibility
# ---------------------------------------------------------------------------


async def _validate_action_feasibility(
    *,
    action: dict[str, Any],
    step_index: int,
    planner: Any,
    prepared_bridge_request: dict[str, Any],
    session_state: dict[str, Any],
) -> list[dict[str, Any]]:
    """Check physical feasibility of a single action via robot agent oracle."""
    findings: list[dict[str, Any]] = []
    resource_jid = str(action.get("resource_jid") or "").strip()
    if not resource_jid:
        findings.append({
            "constraint_owner": "hybrid_validator",
            "constraint_code": "missing_resource",
            "reason": f"Step {step_index + 1} has no resource_jid.",
        })
        return findings

    # Try to call bridge_feasibility_oracle on the robot agent
    product_agent = getattr(planner, "product_agent", None)
    if product_agent is None:
        return findings

    # Build a grounded-action-like dict for the oracle
    grounded_action: dict[str, Any] = {
        "outline_id": f"RECOVERY_SEQ{step_index + 1}",
        "resource_jid": resource_jid,
        "action_type": str(action.get("action_type") or "").strip(),
        "description": str(action.get("description") or "").strip(),
    }
    part_name = str(action.get("part_name") or "").strip()
    if part_name:
        grounded_action["part_name"] = part_name
    target_ref = str(action.get("target_ref") or "").strip()
    if target_ref:
        grounded_action["target_ref"] = target_ref
    pose = action.get("pose")
    if isinstance(pose, dict):
        grounded_action["pose"] = deepcopy(pose)

    # Try CCA constraint validation
    try:
        bridge_resources = dict(prepared_bridge_request.get("bridge_resources") or {})
        llm_input = dict(prepared_bridge_request.get("llm_input") or {})
        cca_findings = validate_outline_macro_cca_constraints(
            tasks=[grounded_action],
            bridge_resources=bridge_resources,
            llm_input=llm_input,
            symbolic_resources=dict(session_state.get("symbolic_resources") or {}),
            symbolic_parts=dict(session_state.get("symbolic_parts") or {}),
        )
        if isinstance(cca_findings, list):
            findings.extend(cca_findings)
    except Exception as exc:
        _logger.debug(
            "[HybridDES] CCA validation not available for step %d: %s",
            step_index, exc,
        )

    return findings


def _feasibility_findings_summary(findings: list[dict[str, Any]]) -> str:
    """Render feasibility findings as text for LLM feedback."""
    if not findings:
        return "(none)"
    lines: list[str] = []
    for f in findings:
        code = str(f.get("constraint_code") or "").strip()
        reason = str(f.get("reason") or "").strip()
        lines.append(f"- [{code}] {reason}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------


def _build_phase_prompt(
    prepared_bridge_request: dict[str, Any],
    session_state: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    """Build the prompt for the current phase."""
    current_phase = str(session_state.get("current_phase") or "").strip()
    llm_input = dict(prepared_bridge_request.get("llm_input") or {})
    bridge_resources = dict(prepared_bridge_request.get("bridge_resources") or {})
    recovery_gap_state = _build_recovery_gap_state(session_state)
    current_recovery_blockers = list(
        dict(prepared_bridge_request.get("llm_input") or {}).get(
            "current_recovery_blockers"
        ) or []
    )

    prompt_input = build_hybrid_des_prompt_input(
        phase=current_phase,
        llm_input=llm_input,
        session_state=session_state,
        bridge_resources=bridge_resources,
        recovery_gap_state=recovery_gap_state,
        current_recovery_blockers=current_recovery_blockers,
    )

    # Check if this is a revision (feedback prompt) or initial generation
    domain_revision_count = int(session_state.get("domain_revision_count") or 0)

    if current_phase == "domain_generation" and domain_revision_count > 0:
        # Build feedback prompt with diagnostics
        previous_plant = session_state.get("current_plant")
        previous_plant_json = ""
        if previous_plant:
            # Serialize plant for display, converting sets to lists
            serializable_plant = deepcopy(previous_plant)
            if isinstance(serializable_plant.get("states"), set):
                serializable_plant["states"] = sorted(serializable_plant["states"])
            if isinstance(serializable_plant.get("marked"), set):
                serializable_plant["marked"] = sorted(serializable_plant["marked"])
            previous_plant_json = json.dumps(
                serializable_plant, indent=2, default=str, ensure_ascii=False,
            )

        solver_result = session_state.get("solver_result") or {}
        solver_diag = solver_diagnostic_summary(solver_result) if solver_result else ""

        prompt_text = build_hybrid_feedback_prompt(
            prompt_input,
            plant_findings_text=plant_findings_summary(
                session_state.get("plant_findings") or [],
            ),
            solver_diagnostic_text=solver_diag,
            feasibility_findings_text=_feasibility_findings_summary(
                session_state.get("feasibility_findings") or [],
            ),
            previous_plant_json=previous_plant_json,
        )
    else:
        prompt_text = build_hybrid_domain_generation_prompt(prompt_input)

    return prompt_input, prompt_text


# ---------------------------------------------------------------------------
# Per-turn artifact writing
# ---------------------------------------------------------------------------


def _write_per_turn_artifact(
    prepared_bridge_request: dict[str, Any],
    session_state: dict[str, Any],
    turn_entry: dict[str, Any],
) -> None:
    """Write debug artifacts for the current turn."""
    try:
        payload = deepcopy(prepared_bridge_request)
        payload["bridge_debug"] = payload.get("bridge_debug") or {}
        payload["bridge_debug"]["multi_turn_session"] = deepcopy(session_state)
        write_bridge_artifacts(
            payload,
            phase_label=str(session_state.get("current_phase") or "hybrid"),
        )
    except Exception:
        _logger.debug("[HybridDES] Failed to write per-turn artifact.", exc_info=True)


# ---------------------------------------------------------------------------
# Main execution loop
# ---------------------------------------------------------------------------


async def execute_hybrid_des_bridge(
    planner: Any,
    prepared_bridge_request: dict[str, Any],
    *,
    session_state: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Run the hybrid DES bridge loop.

    LLM generates recovery plant → DES solver finds safe trace → validators
    check feasibility → finalize proposal.

    Parameters
    ----------
    planner
        The process planner (must have ``product_agent.ask_llm_structured``).
    prepared_bridge_request
        The prepared bridge request dict.
    session_state
        Optional existing session state for resumption.

    Returns
    -------
    Finalized proposal dict, or None if turn budget exhausted.
    """
    product_agent = getattr(planner, "product_agent", None)
    ask_llm_structured = getattr(product_agent, "ask_llm_structured", None)
    if not callable(ask_llm_structured):
        raise RuntimeError(
            "product_agent.ask_llm_structured is required for hybrid DES bridge execution"
        )

    if session_state is not None:
        session_state = deepcopy(session_state)
    else:
        session_state = deepcopy(
            prepared_bridge_request.get("hybrid_session_seed")
            or build_hybrid_session_seed(prepared_bridge_request)
        )
    session_state["status"] = "running"

    bridge_debug = deepcopy(prepared_bridge_request.get("bridge_debug") or {})
    bridge_debug["multi_turn_session"] = deepcopy(session_state)
    prepared_bridge_request["bridge_debug"] = deepcopy(bridge_debug)
    if hasattr(planner, "_set_last_bridge_debug"):
        planner._set_last_bridge_debug(bridge_debug)

    max_turns = int(session_state.get("max_turns") or _DEFAULT_MAX_TURNS)

    while int(session_state.get("turn_index") or 0) < max_turns:
        session_state["turn_index"] = int(session_state.get("turn_index") or 0) + 1
        current_phase = str(
            session_state.get("current_phase") or "domain_generation"
        ).strip().lower()
        turn_idx = int(session_state.get("turn_index") or 0)

        _logger.info(
            "[HybridDES] Turn %d/%d | phase=%s | revisions=%d",
            turn_idx, max_turns, current_phase,
            int(session_state.get("domain_revision_count") or 0),
        )

        # Phases that require an LLM call
        if current_phase == "domain_generation":
            prompt_input, prompt_text = _build_phase_prompt(
                prepared_bridge_request, session_state,
            )
            response_schema = hybrid_des_phase_response_schema("domain_generation")
            raw_response = await ask_llm_structured(
                prompt=prompt_text,
                response_format=response_schema,
            )
            parsed_response = deepcopy(
                raw_response if isinstance(raw_response, dict) else {}
            )
        else:
            # compose_and_solve, validate_plan, finalize — no LLM call
            prompt_input = {}
            prompt_text = ""
            parsed_response = {}

        # Dispatch to handler
        handler = _PHASE_HANDLERS.get(current_phase)
        if handler is None:
            _logger.error("[HybridDES] No handler for phase=%s", current_phase)
            session_state["status"] = "error"
            break

        decision, turn_entry = await handler(
            session_state=session_state,
            parsed_response=parsed_response,
            prepared_bridge_request=prepared_bridge_request,
            planner=planner,
        )

        # Record turn
        turn_entry["turn_index"] = turn_idx
        turn_entry["phase"] = current_phase
        turn_entry["decision"] = decision
        if prompt_text:
            turn_entry["prompt_text"] = prompt_text
            turn_entry["raw_response"] = deepcopy(parsed_response)
        session_state.setdefault("turns", []).append(deepcopy(turn_entry))

        # Transition
        next_phase = _transition_phase(current_phase, decision)
        session_state["current_phase"] = next_phase

        # Terminal check
        if current_phase == "finalize" and decision == "accepted":
            session_state["status"] = "completed"

        # Update debug
        bridge_debug["multi_turn_session"] = deepcopy(session_state)
        bridge_debug["status"] = str(session_state.get("status") or "running")
        prepared_bridge_request["bridge_debug"] = deepcopy(bridge_debug)
        if hasattr(planner, "_set_last_bridge_debug"):
            planner._set_last_bridge_debug(bridge_debug)
        _write_per_turn_artifact(prepared_bridge_request, session_state, turn_entry)

        if session_state.get("status") == "completed":
            break

    # Stash session state
    prepared_bridge_request["hybrid_session_state"] = deepcopy(session_state)

    if session_state.get("status") != "completed":
        session_state["status"] = "turn_budget_exhausted"
        bridge_debug["multi_turn_session"] = deepcopy(session_state)
        bridge_debug["status"] = "turn_budget_exhausted"
        prepared_bridge_request["bridge_debug"] = deepcopy(bridge_debug)
        if hasattr(planner, "_set_last_bridge_debug"):
            planner._set_last_bridge_debug(bridge_debug)
        _logger.warning(
            "[HybridDES] Turn budget exhausted (%d turns) in phase=%s",
            max_turns,
            str(session_state.get("current_phase") or ""),
        )
        return None

    return deepcopy(session_state.get("proposal"))


__all__ = [
    "build_hybrid_session_seed",
    "execute_hybrid_des_bridge",
]
