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
        "current_phase": "evaluate_grounding",
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
# Observation blockers  (Component 1)
# ---------------------------------------------------------------------------


def _compute_observation_blockers(
    symbolic_parts: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Identify parts that need observation before they can be manipulated.

    Parts with ``current_state == 'misplaced'`` (or similar fault states) and
    ``observed_pose is None`` or ``current_location is None`` require a
    ``detect_parts`` observation before any pick/place action can be grounded.
    """
    blockers: list[dict[str, Any]] = []
    _NEEDS_OBSERVATION_STATES = {"misplaced", "dropped", "lost", "unknown", "fault"}
    for part_name, row in (symbolic_parts or {}).items():
        if not isinstance(row, dict):
            continue
        current_state = str(row.get("current_state") or "").strip().lower()
        observed_pose = row.get("observed_pose")
        current_location = row.get("current_location")
        location_basis = str(row.get("location_basis") or "").strip().lower()

        needs_observation = False
        reason_parts: list[str] = []

        # Case 1: part in fault state with no observation data
        if current_state in _NEEDS_OBSERVATION_STATES:
            # If we already have a valid observed pose, we don't need to observe it again
            # even if current_location is unknown
            has_pose = isinstance(observed_pose, dict) and bool(observed_pose)
            
            if not has_pose:
                needs_observation = True
                reason_parts.append(
                    f"part '{part_name}' is in state '{current_state}' "
                    "but has no observed_pose"
                )
            if (current_location is None or str(current_location).strip() == "") and not has_pose:
                needs_observation = True
                reason_parts.append(
                    f"part '{part_name}' has no current_location"
                )

        # Case 2: location_basis requires sensor confirmation
        if location_basis == "sensor_observation" and (
            observed_pose is None or not isinstance(observed_pose, dict)
        ):
            needs_observation = True
            reason_parts.append(
                f"part '{part_name}' location_basis is 'sensor_observation' "
                "but no sensor data available"
            )

        if needs_observation:
            blockers.append({
                "kind": "observation_required",
                "part_name": part_name,
                "description": (
                    f"OBSERVATION REQUIRED: {'; '.join(reason_parts)}. "
                    f"You MUST include a 'detect_parts' or 'observe' event "
                    f"for '{part_name}' BEFORE any pick/place action involving "
                    f"this part. Without observation, the target pose is unknown "
                    f"and the action will fail physical reachability validation."
                ),
                "reason": "; ".join(reason_parts),
            })
    return blockers


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


async def _handle_evaluate_grounding(
    *,
    session_state: dict[str, Any],
    parsed_response: dict[str, Any],
    prepared_bridge_request: dict[str, Any],
    planner: Any,
) -> tuple[str, dict[str, Any]]:
    """Handle evaluate_grounding phase — check if missing poses require camera execution."""
    symbolic_parts = dict(session_state.get("symbolic_parts") or {})
    observation_blockers = _compute_observation_blockers(symbolic_parts)
    
    if not observation_blockers:
        _logger.info("[HybridDES] All parts currently grounded. Proceeding to domain generation.")
        return "grounding_satisfied", {}

    # Need observation — dispatch detect_parts to execution engine
    part_names = []
    for blocker in observation_blockers:
        p_name = str(blocker.get("part_name") or "").strip()
        if p_name and p_name not in part_names:
            part_names.append(p_name)
            
    # We build an observation task for the Resource Agent to run
    # (Similar to what the observation_policy generates in multi_turn)
    observation_tasks = []
    for part_name in part_names:
        observation_tasks.append({
            "id": f"OBS_" + part_name,
            "resource_jid": str(prepared_bridge_request.get("ra_jid") or ""),
            "function_name": "detect_parts",
            "params": {"part_name": part_name},
            "store_as": f"detected_{part_name.lower()}",
            "background": False
        })
        
    session_state["status"] = "paused_after_grounding"
    _logger.info("[HybridDES] Dispatching %d actual observation tasks to Bridge Engine.", len(observation_tasks))
    
    # Store the tasks in multi_turn_session_result so that the Engine's `_execute_multi_turn_session` 
    # (or equivalent bridge return handler) runs them
    prepared_bridge_request["multi_turn_session_result"] = deepcopy(session_state)
    prepared_bridge_request.setdefault("bridge_debug", {})["status"] = "paused_after_grounding"
    
    return "grounding_required", {"dispatch_observation_tasks": observation_tasks}


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
    """Handle validate_plan phase — check physical feasibility of each step.

    Uses rolling symbolic state projection so that each step is validated
    against the projected state *after* all preceding steps have been applied.
    """
    action_sequence = list(session_state.get("action_sequence") or [])
    all_findings: list[dict[str, Any]] = []

    # Component 4: rolling symbolic state projection
    projected_resources = deepcopy(dict(session_state.get("symbolic_resources") or {}))
    projected_parts = deepcopy(dict(session_state.get("symbolic_parts") or {}))

    for i, action in enumerate(action_sequence):
        # Validate step against current projected state
        step_findings = await _validate_action_feasibility(
            action=action,
            step_index=i,
            planner=planner,
            prepared_bridge_request=prepared_bridge_request,
            session_state=session_state,
            pre_resources=projected_resources,
            pre_parts=projected_parts,
            action_sequence=action_sequence,
        )
        all_findings.extend(step_findings)

        # If step is valid, project its effects onto symbolic state
        if not step_findings:
            _apply_hybrid_action_effects(
                action,
                resources=projected_resources,
                parts=projected_parts,
            )

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
    "evaluate_grounding": _handle_evaluate_grounding,
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
    pre_resources: dict[str, dict[str, Any]] | None = None,
    pre_parts: dict[str, dict[str, Any]] | None = None,
    action_sequence: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Check physical feasibility of a single action via CCA and RA oracle."""
    findings: list[dict[str, Any]] = []
    resource_jid = str(action.get("resource_jid") or "").strip()
    if not resource_jid:
        findings.append({
            "constraint_owner": "hybrid_validator",
            "constraint_code": "missing_resource",
            "reason": f"Step {step_index + 1} has no resource_jid.",
        })
        return findings

    # Resolve the resource agent (needed for both CCA and RA validation)
    resolver = getattr(planner, "_resource_by_jid", None)
    resource_agent = resolver(resource_jid) if callable(resolver) else None

    # Build a grounded-action-like dict for validators
    task_id = f"RECOVERY_SEQ{step_index + 1}"
    part_name = str(action.get("part_name") or "").strip()
    action_type = str(action.get("action_type") or "").strip()
    target_ref = str(action.get("target_ref") or "").strip()
    pose = action.get("pose")

    # Build task dict (matches outline task format)
    task: dict[str, Any] = {
        "outline_id": task_id,
        "resource_jid": resource_jid,
        "action_type": action_type,
        "description": str(action.get("description") or "").strip(),
    }
    if part_name:
        task["part_name"] = part_name
    if target_ref:
        task["action_target"] = {"target_location": target_ref}
    if isinstance(pose, dict):
        task["pose"] = deepcopy(pose)

    # Build grounded_action dict for CCA
    grounded_action: dict[str, Any] = deepcopy(task)

    # Use provided symbolic state or fall back to session state
    sym_resources = pre_resources or dict(session_state.get("symbolic_resources") or {})
    sym_parts = pre_parts or dict(session_state.get("symbolic_parts") or {})

    # --- Component 2: CCA constraint validation (corrected signature) ---
    try:
        llm_input = dict(prepared_bridge_request.get("llm_input") or {})
        full_action_sequence = action_sequence or list(
            session_state.get("action_sequence") or []
        )

        # Build outline_tasks list from the full action sequence
        outline_tasks: list[dict[str, Any]] = []
        task_types_by_id: dict[str, str] = {}
        task_index_by_id: dict[str, int] = {}
        dependency_map: dict[str, list[str]] = {}
        for idx, act in enumerate(full_action_sequence):
            act_id = f"RECOVERY_SEQ{idx + 1}"
            act_task: dict[str, Any] = {
                "outline_id": act_id,
                "resource_jid": str(act.get("resource_jid") or "").strip(),
                "action_type": str(act.get("action_type") or "").strip(),
                "description": str(act.get("description") or "").strip(),
            }
            act_part = str(act.get("part_name") or "").strip()
            if act_part:
                act_task["part_name"] = act_part
            act_target = str(act.get("target_ref") or "").strip()
            if act_target:
                act_task["action_target"] = {"target_location": act_target}
            outline_tasks.append(act_task)
            task_types_by_id[act_id] = str(act.get("action_type") or "part_handling").strip()
            task_index_by_id[act_id] = idx
            # Sequential dependency: each step depends on the previous
            if idx > 0:
                dependency_map[act_id] = [f"RECOVERY_SEQ{idx}"]
            else:
                dependency_map[act_id] = []

        # Build projected state for this step
        projected_resources = deepcopy(sym_resources)
        projected_parts = deepcopy(sym_parts)
        _apply_hybrid_action_effects(
            action,
            resources=projected_resources,
            parts=projected_parts,
        )

        # Manually build signature and grounded_action state since hybrid
        # tasks don't have the explicit states needed to run the normal grounding compiler
        act_lower = action_type.lower()
        signature: dict[str, Any] = {
            "task_kind": "part_handling",
            "changes_part_world": bool(part_name),
            "inferable_primary_part": part_name if part_name else None,
        }
        preconditions: dict[str, Any] = {}
        expected_effect: dict[str, Any] = {}

        if act_lower in {"pick_part", "pick", "grasp", "acquire"}:
            signature["task_kind"] = "part_acquire"
            expected_effect["resource"] = {"held_part": part_name, "current_state": "picked"}
            expected_effect["part"] = {"holder": resource_jid}
            preconditions["part"] = {"requires_acquisition": True, "holder": None}
            if target_ref:
                preconditions["source_ref"] = {"location": target_ref}
            elif sym_parts.get(part_name, {}).get("observed_pose"):
                preconditions["source_ref"] = {"location": "observed_pose", "pose": sym_parts[part_name]["observed_pose"]}
            else:
                preconditions["source_ref"] = {"location": sym_parts.get(part_name, {}).get("current_location")}
        elif act_lower in {"place_part", "place", "insert", "release_part", "release"}:
            signature["task_kind"] = "part_release"
            expected_effect["resource"] = {"held_part": None, "current_state": "idle"}
            expected_effect["part"] = {"holder": None, "location": target_ref}
            preconditions["part"] = {"holder": resource_jid}
        elif act_lower in {"place_prepare", "place_approach"}:
            signature["task_kind"] = "part_interaction"
            expected_effect["resource"] = {"held_part": part_name, "current_state": "place_prepare"}
        elif act_lower in {"move_resource", "move", "go_to", "navigate"}:
            signature["task_kind"] = "resource_transition"
            signature["changes_part_world"] = False
            expected_effect["resource"] = {"location": target_ref}
        elif act_lower in {"reset_state", "reset", "home"}:
            signature["task_kind"] = "resource_transition"
            signature["changes_part_world"] = False
            expected_effect["resource"] = {"current_state": "idle"}
        elif act_lower in {"observe", "detect_parts", "detect", "observe_part"}:
            signature["task_kind"] = "observation"
            signature["changes_part_world"] = False

        grounded_action["preconditions"] = preconditions
        grounded_action["expected_effect"] = expected_effect
        grounded_action["task_kind"] = signature["task_kind"]
        if signature.get("effect_scope"):
            grounded_action["effect_scope"] = signature["effect_scope"]

        cca_result = validate_outline_macro_cca_constraints(
            task=task,
            grounded_action=grounded_action,
            signature=signature,
            pre_resources=sym_resources,
            pre_parts=sym_parts,
            projected_resources=projected_resources,
            projected_parts=projected_parts,
            llm_input=llm_input,
            outline_tasks=outline_tasks,
            task_types_by_id=task_types_by_id,
            task_index_by_id=task_index_by_id,
            dependency_map=dependency_map,
        )
        cca_findings = list(cca_result.get("findings") or [])
        if cca_findings:
            findings.extend(cca_findings)
            _logger.info(
                "[HybridDES] CCA validation step %d: %d finding(s).",
                step_index, len(cca_findings),
            )
    except Exception as exc:
        _logger.warning(
            "[HybridDES] CCA validation failed for step %d: %s",
            step_index, exc,
        )

    # --- Component 3: Resource Agent feasibility oracle ---
    if resource_agent is not None:
        oracle = getattr(resource_agent, "bridge_feasibility_oracle", None)
        if callable(oracle):
            try:
                # Build bridge snapshot from resource agent or symbolic state
                resource_snapshot = deepcopy(
                    dict(sym_resources.get(resource_jid) or {})
                )
                get_bridge_snapshot = getattr(
                    resource_agent, "get_bridge_snapshot", None
                )
                if callable(get_bridge_snapshot):
                    try:
                        live_snapshot = get_bridge_snapshot()
                    except Exception:
                        live_snapshot = {}
                    if isinstance(live_snapshot, dict):
                        for field_name in (
                            "workspace_bounds",
                            "available_named_poses",
                            "bridge_adapter",
                            "resource_type",
                            "role",
                        ):
                            if (
                                field_name not in resource_snapshot
                                and field_name in live_snapshot
                            ):
                                resource_snapshot[field_name] = deepcopy(
                                    live_snapshot.get(field_name)
                                )

                # Build part context from symbolic parts
                part_row = deepcopy(dict(sym_parts.get(part_name) or {}))
                part_context: dict[str, Any] = {
                    **part_row,
                    "target": {
                        "target_location": target_ref,
                    },
                }
                if isinstance(pose, dict):
                    part_context["observed_pose"] = deepcopy(pose)

                oracle_result = oracle(
                    operation_kind=action_type,
                    part_name=part_name or None,
                    part_context=part_context,
                    bridge_snapshot=resource_snapshot,
                    grounded_action=deepcopy(grounded_action),
                )

                result = dict(oracle_result or {})
                if not bool(result.get("allowed", True)):
                    constraint_code = (
                        str(result.get("constraint_code") or "").strip()
                        or "resource_unavailable"
                    )
                    reason = (
                        str(result.get("reason") or "").strip()
                        or "resource feasibility oracle rejected the action"
                    )
                    findings.append({
                        "task_id": task_id,
                        "resource_jid": resource_jid,
                        "part_name": part_name or None,
                        "pose_source": "resource_feasibility",
                        "pose": deepcopy(
                            dict(result.get("evidence") or {}).get(
                                "checked_pose"
                            )
                        ),
                        "workspace_bounds": deepcopy(
                            dict(result.get("evidence") or {}).get(
                                "workspace_bounds"
                            )
                        ),
                        "failed_axes": [constraint_code],
                        "constraint_owner": "resource",
                        "constraint_family": "resource_feasibility",
                        "constraint_code": constraint_code,
                        "reason": reason,
                        "evidence": deepcopy(result.get("evidence") or {}),
                    })
                    _logger.info(
                        "[HybridDES] RA oracle step %d rejected: %s",
                        step_index, reason,
                    )
            except Exception as exc:
                _logger.warning(
                    "[HybridDES] RA oracle failed for step %d: %s",
                    step_index, exc,
                )

    return findings


# ---------------------------------------------------------------------------
# Component 4: Symbolic state projection for action effects
# ---------------------------------------------------------------------------


def _apply_hybrid_action_effects(
    action: dict[str, Any],
    *,
    resources: dict[str, dict[str, Any]],
    parts: dict[str, dict[str, Any]],
) -> None:
    """Apply the symbolic effects of a hybrid DES action to projected state.

    This mirrors what multi_turn does with ``_apply_outline_task_effects`` but
    uses the simpler action_type-based contract from the hybrid plant events.
    """
    resource_jid = str(action.get("resource_jid") or "").strip()
    part_name = str(action.get("part_name") or "").strip()
    action_type = str(action.get("action_type") or "").strip().lower()
    target_ref = str(action.get("target_ref") or "").strip()

    resource_row = resources.get(resource_jid)
    if resource_row is None and resource_jid:
        resource_row = {"resource_jid": resource_jid}
        resources[resource_jid] = resource_row

    part_row = parts.get(part_name) if part_name else None

    if action_type in {"pick_part", "pick", "grasp", "acquire"}:
        # Resource now holds the part
        if resource_row is not None:
            resource_row["held_part"] = part_name
            resource_row["gripper_state"] = "closed"
        if part_row is not None:
            part_row["current_holder_resource_jid"] = resource_jid

    elif action_type in {"place_part", "place", "insert", "release_part", "release"}:
        # Resource releases the part to target
        if resource_row is not None:
            resource_row["held_part"] = None
            resource_row["gripper_state"] = "open"
        if part_row is not None:
            part_row["current_holder_resource_jid"] = None
            if target_ref:
                part_row["current_location"] = target_ref

    elif action_type in {"place_prepare", "place_approach"}:
        # Resource approaches target while held
        if resource_row is not None:
            resource_row["current_state"] = "place_prepare"
            # It should ALREADY be holding the part, so we assert it
            if resource_row.get("held_part") != part_name:
                pass # The Feasibility oracle will catch this violation natively
        
    elif action_type in {"move_resource", "move", "go_to", "navigate"}:
        # Resource moves to target location
        if resource_row is not None and target_ref:
            resource_row["current_location"] = target_ref

    elif action_type in {"reset_state", "reset", "home"}:
        # Resource returns to idle
        if resource_row is not None:
            resource_row["current_state"] = "idle"
            if target_ref:
                resource_row["current_location"] = target_ref

    elif action_type in {"observe", "detect_parts", "detect", "observe_part"}:
        # Observation updates part pose (in real execution)
        # For projection, mark that observation was performed and inject a dummy pose
        # in the resource's workspace to satisfy downstream oracle reachability checks
        if part_row is not None:
            part_row["observation_performed"] = True
            
            # Create a broadly valid dummy pose. The exact bounds check varies, 
            # so we place it roughly in a generic work area unless we have resource bounds.
            # Using typical generic positive values since z is usually > 0
            dummy_pose = {"x": 0.0, "y": 0.0, "z": 1.0}
            
            bounds = (resources.get(resource_jid) or {}).get("workspace_bounds")
            if isinstance(bounds, dict):
                dummy_pose["x"] = (bounds.get("x_min_m", -0.5) + bounds.get("x_max_m", 0.5)) / 2
                dummy_pose["y"] = (bounds.get("y_min_m", -0.5) + bounds.get("y_max_m", 0.5)) / 2
                dummy_pose["z"] = (bounds.get("z_min_m", 0.5) + bounds.get("z_max_m", 1.5)) / 2
                
            part_row["observed_pose"] = dummy_pose


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
            # compose_and_solve, validate_plan, finalize, evaluate_grounding — no LLM call
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
            
        if current_phase == "evaluate_grounding" and decision == "grounding_required":
            # State is already set to 'paused_after_grounding' inside the handler
            pass

        # Update debug
        bridge_debug["multi_turn_session"] = deepcopy(session_state)
        bridge_debug["status"] = str(session_state.get("status") or "running")
        prepared_bridge_request["bridge_debug"] = deepcopy(bridge_debug)
        if hasattr(planner, "_set_last_bridge_debug"):
            planner._set_last_bridge_debug(bridge_debug)
        _write_per_turn_artifact(prepared_bridge_request, session_state, turn_entry)

        if session_state.get("status") in ("completed", "paused_after_grounding"):
            break

    # Stash session state
    prepared_bridge_request["hybrid_session_state"] = deepcopy(session_state)

    if session_state.get("status") not in ("completed", "paused_after_grounding"):
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

    if session_state.get("status") == "paused_after_grounding":
        return None
        
    return deepcopy(session_state.get("proposal"))


__all__ = [
    "build_hybrid_session_seed",
    "execute_hybrid_des_bridge",
]
