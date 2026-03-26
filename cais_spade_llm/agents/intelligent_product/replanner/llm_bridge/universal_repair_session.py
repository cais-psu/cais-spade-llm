"""TSS-enriched repair session (v3 bridge).

Implements the constraint-guided repair loop using:
  - Structured LLM output (constrained decoding via ``ask_llm_structured``)
  - Projection tool (mid-turn simulation via ``project_primitive_sequence``)
  - Bridge feedback summary (partitioned rejections)
  - Delta prompts (cached static sections on Turn 2+)
  - Turn cache with duplicate-turn suppression

Three LLM output types:
  - ``observe`` — top-level sensor request, results stored in observation_store
  - ``repair_outline`` — task/state-level recovery outline with no primitives
  - ``repair_program`` — the single universal repair artifact

Turn sequence::

    [observe]* → repair_outline → repair_program →
    [validate → reject → feedback → [observe]* → repair_outline → repair_program]* →
    approve → execute → verify

Each iteration rebuilds from scratch (with delta optimization on Turn 2+).
Only discovered constraints (condensed from rejections) and the most recent
rejected proposal persist across iterations.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.mutation_types import (
    RepairProgram,
    RepairStepKind,
    ValidatedRepairProgram,
    extract_constraint_from_rejection,
    repair_program_from_dict,
    repair_program_to_dict,
    validated_program_from_dict,
    validated_program_to_dict,
)
from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.recovery_context_builder import (
    build_grounding_assessment,
    build_recovery_context,
    part_observation_status,
    recovery_context_to_prompt_dict,
)
from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.repair_program_validator import (
    validate_repair_program,
)
from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.recovery_library import (
    RecoveryLibrary,
)
from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.outline_validation import (
    OutlineValidationResult,
    materialize_abstract_repair_order,
    materialize_outline_actions,
    outline_signature,
    validate_repair_outline,
)
from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.observation_policy import (
    observation_semantic_operation,
)
from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.primitive_semantics import (
    validate_and_project_steps,
)
from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.tss_schemas import (
    REPAIR_TURN_RESPONSE_SCHEMA,
    PROJECT_PRIMITIVE_SEQUENCE_TOOL,
    parse_structured_response,
)
from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.tss_feedback import (
    BridgeFeedbackSummary,
    summarize_validation_feedback,
    feedback_to_prompt_section,
    open_witnesses_to_prompt_section,
)
from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.tss_turn_cache import (
    TurnCache,
    ProgramDelta,
    compute_context_fingerprint,
    compute_outline_context_fingerprint,
    compute_program_fingerprint,
    diff_programs,
    can_reuse_validation_result,
)
from cais_spade_llm.resources.resource_profile import (
    get_resource_profile,
)

logger = logging.getLogger(__name__)

# Turn budget for the repair loop.
_DEFAULT_MAX_TURNS = 8

# Debug output directory — sits at the repo root.
_DEBUG_DIR: Path | None = None


def _get_debug_dir() -> Path:
    """Lazily resolve and create ``<repo_root>/debug/``."""
    global _DEBUG_DIR
    if _DEBUG_DIR is None:
        # Walk upward from this file to find the repo root (contains pyproject.toml).
        candidate = Path(__file__).resolve().parent
        for _ in range(10):
            if (candidate / "pyproject.toml").exists():
                break
            candidate = candidate.parent
        _DEBUG_DIR = candidate / "debug"
        _DEBUG_DIR.mkdir(exist_ok=True)
    return _DEBUG_DIR


def _is_empty_debug_value(value: Any) -> bool:
    return value is None or value == "" or value == [] or value == {}


def _strip_empty_debug_fields(value: Any) -> Any:
    if isinstance(value, dict):
        cleaned: dict[str, Any] = {}
        for key, child in value.items():
            compact_child = _strip_empty_debug_fields(child)
            if _is_empty_debug_value(compact_child):
                continue
            cleaned[key] = compact_child
        return cleaned
    if isinstance(value, list):
        cleaned_list = []
        for child in value:
            compact_child = _strip_empty_debug_fields(child)
            if _is_empty_debug_value(compact_child):
                continue
            cleaned_list.append(compact_child)
        return cleaned_list
    return value


def _compact_outline_debug_payload(
    outline: dict[str, Any] | None,
    *,
    accepted: bool,
) -> dict[str, Any]:
    compact_outline = deepcopy(dict(outline or {}))
    reasoning = dict(compact_outline.get("reasoning") or {})
    if accepted and list(reasoning.get("outline_actions") or []):
        reasoning.pop("abstract_repair_order", None)
    compact_outline["reasoning"] = _strip_empty_debug_fields(reasoning)
    return _strip_empty_debug_fields(compact_outline)


def _compact_outline_validation_debug_payload(
    outline_validation: dict[str, Any] | None,
) -> dict[str, Any]:
    validation_dict = deepcopy(dict(outline_validation or {}))
    if bool(validation_dict.get("is_valid")):
        terminal_phase = str(validation_dict.get("terminal_phase") or "").strip()
        would_close_bridge_if_executed = terminal_phase in {
            "resume_modeled_suffix",
            "replace_suffix",
        }
        compact_validation = {
            "is_valid": True,
            "terminal_phase": terminal_phase,
            "unresolved_bridge_goal_parts_now": list(
                validation_dict.get("unresolved_bridge_goal_parts") or []
            ),
            "would_close_bridge_if_executed": would_close_bridge_if_executed,
        }
        return _strip_empty_debug_fields(compact_validation)
    return _strip_empty_debug_fields(validation_dict)


def _compact_program_debug_payload(program: dict[str, Any] | None) -> dict[str, Any]:
    compact_program = deepcopy(dict(program or {}))
    reasoning = dict(compact_program.get("reasoning") or {})
    reasoning.pop("abstract_repair_order", None)
    reasoning.pop("transition_plan", None)
    reasoning.pop("current_state_analysis", None)
    reasoning.pop("goal_gap_analysis", None)
    compact_program["reasoning"] = _strip_empty_debug_fields(reasoning)
    return _strip_empty_debug_fields(compact_program)


def _format_debug_arg_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    if value is True:
        return "true"
    if value is False:
        return "false"
    if value is None:
        return "null"
    if isinstance(value, (int, float)):
        return str(value)
    return json.dumps(value, default=str, ensure_ascii=False, sort_keys=True)


def _direct_nominal_call_steps(program: dict[str, Any] | None) -> list[dict[str, Any]]:
    program_dict = dict(program or {})
    synthesized_fn_names = {
        str(fn.get("name") or "").strip()
        for fn in (program_dict.get("function_defs") or [])
        if isinstance(fn, dict) and str(fn.get("name") or "").strip()
    }
    direct_steps: list[dict[str, Any]] = []
    for step in (program_dict.get("steps") or []):
        if not isinstance(step, dict):
            continue
        if str(step.get("kind") or "").strip() != "call_function":
            continue
        payload = dict(step.get("payload") or {})
        function_name = str(payload.get("function_name") or "").strip()
        if function_name and function_name not in synthesized_fn_names:
            direct_steps.append(step)
    return direct_steps


def _format_direct_nominal_call_label(step: dict[str, Any]) -> str:
    payload = dict((step or {}).get("payload") or {})
    function_name = str(payload.get("function_name") or "?").strip() or "?"
    resource_jid = str(payload.get("resource_jid") or "").strip()
    args = dict(payload.get("args") or {})
    arg_text = ", ".join(
        f"{str(key)}={_format_debug_arg_value(value)}"
        for key, value in args.items()
    )
    call_name = f"{resource_jid}.{function_name}" if resource_jid else function_name
    return f"{call_name}({arg_text})" if arg_text else f"{call_name}()"


def _summarize_hybrid_repair_program(program: dict[str, Any] | None) -> str:
    program_dict = dict(program or {})
    synthesized_fn_names = [
        str(fn.get("name") or "?").strip() or "?"
        for fn in (program_dict.get("function_defs") or [])
        if isinstance(fn, dict)
    ]
    direct_nominal_calls = _direct_nominal_call_steps(program_dict)
    pieces: list[str] = []
    fn_count = len(synthesized_fn_names)
    if fn_count == 1:
        pieces.append(f"1 synthesized fn: {synthesized_fn_names[0]}")
    elif fn_count > 1:
        preview = ", ".join(synthesized_fn_names[:5])
        if fn_count > 5:
            preview += ", ..."
        pieces.append(f"{fn_count} synthesized fn: {preview}")
    else:
        pieces.append("0 synthesized fn")
    if direct_nominal_calls:
        count = len(direct_nominal_calls)
        pieces.append(f"{count} direct nominal call{'s' if count != 1 else ''}")
    return "; ".join(pieces)


def _render_hybrid_repair_program_lines(program: dict[str, Any] | None) -> list[str]:
    program_dict = dict(program or {})
    lines: list[str] = []
    for fn in (program_dict.get("function_defs") or []):
        if not isinstance(fn, dict):
            continue
        primitives = [
            str(step.get("primitive") or "?")
            for step in (fn.get("primitive_program") or [])
            if isinstance(step, dict)
        ]
        lines.append(f"fn: {str(fn.get('name') or '?').strip() or '?'}")
        lines.append(f"  primitives: {' -> '.join(primitives) if primitives else '?'}")

    for step in _direct_nominal_call_steps(program_dict):
        lines.append(f"nominal: {_format_direct_nominal_call_label(step)}")

    for step in (program_dict.get("steps") or []):
        if not isinstance(step, dict):
            continue
        if str(step.get("kind") or "").strip() == RepairStepKind.RESUME_SUFFIX.value:
            lines.append("terminal: resume_suffix")
            break
    return lines


def _repair_ready_helper_semantic_summary(name: str) -> str:
    token = str(name or "").strip()
    if token == "compute_pick_targets":
        return (
            "Produces `approach_pose` above the observed part and `target_pose` at the "
            "actual grasp pose. A grounded pickup normally reaches `target_pose` before "
            "`grasp_part`."
        )
    if token == "compute_place_targets":
        return (
            "Produces `approach_pose` above the destination and `target_pose` at the "
            "actual place pose. Ground it with `destination_location` only when that "
            "symbolic destination resolves to place geometry for the current part; "
            "otherwise provide explicit `product_geometry`. A grounded place normally "
            "reaches `approach_pose`, then `target_pose`, then `release_part`."
        )
    if token == "detect_parts":
        return (
            "Produces a fresh observed part pose. Reuse its exact `store_as` handle in "
            "later same-function `context_ref` paths."
        )
    if token == "get_current_pose":
        return (
            "Produces the current end-effector pose, including orientation, for later "
            "`move_pose` parameter binding."
        )
    return ""


def _outline_action_id_for_function_name(
    function_name: str,
    outline_actions: list[dict[str, Any]],
) -> int | None:
    token = str(function_name or "").strip()
    if not token:
        return None
    for idx, action in enumerate(outline_actions):
        action_id = str(action.get("action_id") or "").strip()
        if not action_id:
            continue
        if token == action_id or token.startswith(f"{action_id}__"):
            return idx
    return None


def _outline_match_score(
    action: dict[str, Any],
    *,
    function_name: str,
    resource_jid: str,
    args: dict[str, Any],
) -> int:
    targets = {
        str(entity).strip()
        for entity in (action.get("target_entities") or [])
        if str(entity).strip()
    }
    phase_type = str(action.get("phase_type") or "").strip().lower()
    objective = str(action.get("objective") or "").strip().lower()
    action_id = str(action.get("action_id") or "").strip().lower()
    function_token = str(function_name or "").strip().lower()
    score = 0
    if resource_jid and resource_jid in targets:
        score += 3
    for value in args.values():
        if isinstance(value, str) and value.strip() in targets:
            score += 2
    part_name = str(args.get("part_name") or "").strip()
    if part_name and part_name in targets:
        score += 2
    destination = str(args.get("destination_location") or "").strip()
    if destination and destination in targets:
        score += 2
    if function_token in {"pick_approach", "pick_grasp"}:
        if phase_type == "recover_entities":
            score += 4
        if part_name and part_name.lower() in objective:
            score += 2
        if "recover" in objective or "pick" in objective or "assemble" in objective:
            score += 1
    if function_token in {"place_approach", "place_insert"}:
        if phase_type == "recover_entities":
            score += 4
        if destination and destination.lower() in objective:
            score += 2
        if "place" in objective or "assemble" in objective or "recover" in objective:
            score += 1
    if function_token in {"move_home", "move_to_named_pose"}:
        if resource_jid and resource_jid in targets:
            if any(
                keyword in objective or keyword in action_id
                for keyword in ("park", "vacate", "clear", "retract", "home")
            ):
                score += 5
            if phase_type == "recover_entities":
                score += 2
            elif phase_type in {"resolve_safety", "restore_capability"}:
                score += 1
    return score


def _render_outline_grouped_repair_program_lines(
    program: dict[str, Any] | None,
    accepted_outline: dict[str, Any] | None,
) -> list[str]:
    program_dict = dict(program or {})
    accepted_outline_dict = dict(accepted_outline or {})
    outline_reasoning = dict(accepted_outline_dict.get("reasoning") or {})
    outline_actions = list(outline_reasoning.get("outline_actions") or [])
    if not outline_actions:
        outline_actions = materialize_outline_actions(outline_reasoning)
    if not outline_actions:
        return _render_hybrid_repair_program_lines(program_dict)

    grouped: list[list[str]] = [[] for _ in outline_actions]
    synthesized_lines_by_name: dict[str, str] = {}
    synthesized_call_names: set[str] = set()
    current_outline_idx = 0

    for fn in (program_dict.get("function_defs") or []):
        if not isinstance(fn, dict):
            continue
        fn_name = str(fn.get("name") or "?").strip() or "?"
        primitives = [
            str(step.get("primitive") or "?")
            for step in (fn.get("primitive_program") or [])
            if isinstance(step, dict)
        ]
        synthesized_lines_by_name[fn_name] = (
            f"primitives: {' -> '.join(primitives) if primitives else '?'}"
        )

    for step in (program_dict.get("steps") or []):
        if not isinstance(step, dict):
            continue
        kind = str(step.get("kind") or "").strip()
        if kind == RepairStepKind.RESUME_SUFFIX.value:
            terminal_idx = max(len(outline_actions) - 1, 0)
            grouped[terminal_idx].append("resume_suffix")
            continue
        if kind != RepairStepKind.CALL_FUNCTION.value:
            continue
        payload = dict(step.get("payload") or {})
        function_name = str(payload.get("function_name") or "").strip()
        if not function_name:
            continue
        if function_name in synthesized_lines_by_name:
            idx = _outline_action_id_for_function_name(function_name, outline_actions)
            if idx is None:
                idx = min(current_outline_idx, max(len(outline_actions) - 1, 0))
            current_outline_idx = max(current_outline_idx, idx)
            grouped[idx].append(synthesized_lines_by_name[function_name])
            synthesized_call_names.add(function_name)
            continue
        idx = _outline_action_id_for_function_name(function_name, outline_actions)
        if idx is None:
            best_idx = None
            best_score = 0
            for candidate_idx in range(current_outline_idx, len(outline_actions)):
                score = _outline_match_score(
                    outline_actions[candidate_idx],
                    function_name=function_name,
                    resource_jid=str(payload.get("resource_jid") or "").strip(),
                    args=dict(payload.get("args") or {}),
                )
                if score > best_score:
                    best_score = score
                    best_idx = candidate_idx
            idx = best_idx if best_idx is not None else min(
                current_outline_idx,
                max(len(outline_actions) - 1, 0),
            )
        current_outline_idx = max(current_outline_idx, idx)
        grouped[idx].append(_format_direct_nominal_call_label(step))

    for fn_name, line in synthesized_lines_by_name.items():
        if fn_name in synthesized_call_names:
            continue
        idx = _outline_action_id_for_function_name(fn_name, outline_actions)
        if idx is None:
            idx = min(current_outline_idx, max(len(outline_actions) - 1, 0))
        grouped[idx].append(line)

    lines: list[str] = []
    for idx, action in enumerate(outline_actions):
        action_id = str(action.get("action_id") or f"A{idx + 1}").strip()
        objective = str(action.get("objective") or "").strip()
        header = action_id if not objective else f"{action_id} — {objective}"
        lines.append(header)
        rows = grouped[idx]
        if not rows:
            lines.append("  (no explicit repair-program steps recorded)")
            continue
        for row in rows:
            lines.append(f"  {row}")
    return lines


def _compact_program_validation_debug_payload(
    validation: dict[str, Any] | None,
) -> dict[str, Any]:
    validation_dict = deepcopy(dict(validation or {}))
    rejection_rows = []
    for row in validation_dict.get("rejection_reasons") or []:
        if not isinstance(row, dict):
            continue
        compact_row = {
            "layer": row.get("layer"),
            "check": row.get("check"),
            "message": row.get("message"),
            "rule_id": row.get("rule_id"),
        }
        rejection_rows.append(_strip_empty_debug_fields(compact_row))
    is_valid = bool(validation_dict.get("is_valid", False))
    if is_valid:
        compact_validation = {
            "is_valid": True,
            "risk_level": validation_dict.get("risk_level"),
            "requires_operator_approval": validation_dict.get("requires_operator_approval"),
            "continuation_viable": validation_dict.get("continuation_viable"),
        }
    else:
        compact_validation = {
            "is_valid": False,
            "rejection_reasons": rejection_rows,
        }
    return _strip_empty_debug_fields(compact_validation)


def _compact_rejected_proposal_for_prompt(
    proposal: dict[str, Any] | None,
    *,
    prompt_mode: str,
) -> dict[str, Any]:
    proposal_dict = deepcopy(dict(proposal or {}))
    if prompt_mode != "repair_ready":
        return _strip_empty_debug_fields(proposal_dict)

    compact: dict[str, Any] = {
        "type": proposal_dict.get("type"),
        "rationale": proposal_dict.get("rationale"),
    }

    reasoning = dict(proposal_dict.get("reasoning") or {})
    compact_reasoning: dict[str, Any] = {}
    safety_check = reasoning.get("safety_check")
    if isinstance(safety_check, list) and safety_check:
        compact_reasoning["safety_check"] = safety_check
    if compact_reasoning:
        compact["reasoning"] = compact_reasoning

    function_defs = []
    for fn_def in proposal_dict.get("function_defs") or []:
        if not isinstance(fn_def, dict):
            continue
        primitive_program = []
        for step in fn_def.get("primitive_program") or []:
            if not isinstance(step, dict):
                continue
            primitive_program.append({
                "resource_jid": step.get("resource_jid"),
                "primitive": step.get("primitive"),
                "params": step.get("params"),
            })
        function_defs.append({
            "name": fn_def.get("name"),
            "intent": fn_def.get("intent"),
            "primitive_program": primitive_program,
        })
    if function_defs:
        compact["function_defs"] = function_defs

    steps = []
    for step in proposal_dict.get("steps") or []:
        if not isinstance(step, dict):
            continue
        payload = dict(step.get("payload") or {})
        steps.append({
            "kind": step.get("kind"),
            "payload": {
                "function_name": payload.get("function_name"),
                "resource_jid": payload.get("resource_jid"),
                "mutation_type": payload.get("mutation_type"),
            },
        })
    if steps:
        compact["steps"] = steps

    success_conditions = [
        dict(condition)
        for condition in (proposal_dict.get("success_conditions") or [])
        if isinstance(condition, dict)
    ]
    if success_conditions:
        compact["success_conditions"] = success_conditions

    return _strip_empty_debug_fields(compact)


def _write_turn_debug(
    session_id: str,
    turn_idx: int,
    *,
    prompt: str = "",
    policy_input: dict[str, Any] | None = None,
    turn_metrics: dict[str, Any] | None = None,
    raw_response: Any = None,
    policy_decision: Any = None,
    error: str = "",
    observation: dict[str, Any] | None = None,
    outline: dict[str, Any] | None = None,
    outline_validation: dict[str, Any] | None = None,
    program: dict[str, Any] | None = None,
    validation: dict[str, Any] | None = None,
) -> None:
    """Append one turn's data to the incremental debug text file."""
    try:
        debug_dir = _get_debug_dir()
        txt_path = debug_dir / f"v3_session_{session_id}.txt"
        sep = "=" * 80

        lines: list[str] = ["", sep, f"TURN {turn_idx}", sep]

        if prompt:
            lines.append("")
            lines.append(f"--- PROMPT ({len(prompt)} chars) ---")
            lines.append(prompt)
        elif policy_input:
            lines.append("")
            lines.append("--- POLICY INPUT ---")
            lines.append(json.dumps(policy_input, indent=2, default=str, ensure_ascii=False))

        if turn_metrics:
            lines.append("")
            lines.append("--- TURN METRICS ---")
            lines.append(json.dumps(turn_metrics, indent=2, default=str, ensure_ascii=False))

        response_block = policy_decision if policy_decision is not None else raw_response
        if (
            outline
            and policy_decision is None
            and isinstance(raw_response, dict)
            and str(raw_response.get("type") or "").strip() == "repair_outline"
        ):
            response_block = None
        if (
            program
            and policy_decision is None
            and isinstance(raw_response, dict)
            and str(raw_response.get("type") or "").strip() == "repair_program"
        ):
            response_block = None
        if response_block is not None:
            lines.append("")
            lines.append(
                "--- POLICY DECISION ---"
                if policy_decision is not None
                else "--- LLM RESPONSE ---"
            )
            if isinstance(response_block, dict):
                lines.append(json.dumps(response_block, indent=2, default=str, ensure_ascii=False))
            else:
                lines.append(str(response_block))

        if error:
            lines.append("")
            lines.append(f"--- ERROR ---")
            lines.append(error)

        if observation:
            lines.append("")
            lines.append("--- OBSERVATION ---")
            lines.append(json.dumps(observation, indent=2, default=str, ensure_ascii=False))

        compact_outline = None
        if outline:
            compact_outline = _compact_outline_debug_payload(
                outline,
                accepted=bool(
                    outline_validation and outline_validation.get("is_valid", False)
                ),
            )
        if compact_outline:
            lines.append("")
            lines.append("--- REPAIR OUTLINE ---")
            lines.append(
                json.dumps(compact_outline, indent=2, default=str, ensure_ascii=False)
            )

        if outline_validation:
            compact_outline_validation = _compact_outline_validation_debug_payload(
                outline_validation
            )
            lines.append("")
            is_valid = outline_validation.get("is_valid", False)
            lines.append(
                f"--- OUTLINE VALIDATION ({'ACCEPTED' if is_valid else 'REJECTED'}) ---"
            )
            lines.append(
                json.dumps(
                    compact_outline_validation,
                    indent=2,
                    default=str,
                    ensure_ascii=False,
                )
            )

        if program:
            lines.append("")
            lines.append("--- REPAIR PROGRAM ---")
            lines.append(
                json.dumps(
                    _compact_program_debug_payload(program),
                    indent=2,
                    default=str,
                    ensure_ascii=False,
                )
            )

        if validation:
            lines.append("")
            is_valid = validation.get("is_valid", False)
            lines.append(f"--- VALIDATION ({'ACCEPTED' if is_valid else 'REJECTED'}) ---")
            lines.append(
                json.dumps(
                    _compact_program_validation_debug_payload(validation),
                    indent=2,
                    default=str,
                    ensure_ascii=False,
                )
            )

        with open(txt_path, "a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
    except Exception:
        pass  # Never let debug I/O break the session.


_react_logger = logging.getLogger(__name__ + ".react")


def _raw_response_content(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {}
    content = raw.get("content") or raw
    if isinstance(content, str):
        try:
            content = json.loads(content)
        except Exception:
            content = {}
    return content if isinstance(content, dict) else {}


def _summarize_program_step_fact(
    step: dict[str, Any],
    fn_map: dict[str, dict[str, Any]],
) -> str:
    if not isinstance(step, dict):
        return ""
    if str(step.get("kind") or "").strip() != "call_function":
        return ""
    payload = dict(step.get("payload") or {})
    fn_name = str(payload.get("function_name") or "").strip()
    resource_jid = str(payload.get("resource_jid") or "").strip()
    if not fn_name:
        return ""
    fn_def = dict(fn_map.get(fn_name) or {})
    primitive_program = list(fn_def.get("primitive_program") or [])
    if not primitive_program:
        return f"{resource_jid}.{fn_name}" if resource_jid else fn_name
    first = primitive_program[0]
    if not isinstance(first, dict):
        return f"{resource_jid}.{fn_name}" if resource_jid else fn_name
    primitive = str(first.get("primitive") or "").strip()
    if not primitive:
        return f"{resource_jid}.{fn_name}" if resource_jid else fn_name
    fact = f"{resource_jid}.{primitive}" if resource_jid else primitive
    params = dict(first.get("params") or {})
    if primitive == "move_to_named_pose":
        pose_name = str(params.get("pose_name") or "").strip()
        if pose_name:
            fact += f" -> {pose_name}"
    elif primitive == "grasp_part":
        model_name = str(params.get("model_name") or params.get("part_name") or "").strip()
        if model_name:
            fact += f" -> {model_name}"
    elif primitive == "release_part":
        location = str(params.get("location") or params.get("place_pose") or "").strip()
        if location:
            fact += f" -> {location}"
    return fact


def _summarize_turn_thought(
    raw: Any,
    *,
    response_type: str = "",
    validation: dict[str, Any] | None = None,
) -> str:
    """Build a concise ReAct thought summary from rationale plus key blockers."""
    content = _raw_response_content(raw)
    if not content:
        return ""

    rationale = str(content.get("rationale") or "").strip()
    reasoning = dict(content.get("reasoning") or {})
    normalized_response_type = (
        str(response_type or content.get("type") or "").strip().lower()
    )
    facts: list[str] = []
    primitive_facts: list[str] = []

    def _append_fact(target: list[str], text: Any) -> None:
        sentence = str(text or "").strip()
        if not sentence:
            return
        if sentence[-1] not in ".!?":
            sentence += "."
        lowered = sentence.lower()
        if rationale and lowered in rationale.lower():
            return
        if any(
            lowered == existing.lower()
            or lowered in existing.lower()
            or existing.lower() in lowered
            for existing in target
        ):
            return
        target.append(sentence)

    if normalized_response_type == "repair_program":
        fn_map = {
            str(fn.get("name") or "").strip(): dict(fn)
            for fn in (content.get("function_defs") or [])
            if isinstance(fn, dict) and str(fn.get("name") or "").strip()
        }
        for step in content.get("steps") or []:
            if not isinstance(step, dict):
                continue
            primitive_fact = _summarize_program_step_fact(step, fn_map)
            if not primitive_fact:
                continue
            _append_fact(primitive_facts, primitive_fact)
            if len(primitive_facts) >= 3:
                break

        primitive_summary = "; ".join(primitive_facts[:3])
        validator_feedback = ""
        if isinstance(validation, dict) and not bool(validation.get("is_valid", False)):
            rejection_rows = list(validation.get("rejection_reasons") or [])
            messages = [
                str(row.get("message") or "").strip()
                for row in rejection_rows
                if isinstance(row, dict) and str(row.get("message") or "").strip()
            ]
            if messages:
                validator_feedback = "; ".join(messages[:2])

        if rationale and primitive_summary and validator_feedback:
            return (
                rationale
                + " Primitive plan: "
                + primitive_summary
                + " Validator: "
                + validator_feedback
            )
        if rationale and primitive_summary:
            return rationale + " Primitive plan: " + primitive_summary
        if primitive_summary and validator_feedback:
            return "Primitive plan: " + primitive_summary + " Validator: " + validator_feedback
        if primitive_summary:
            return "Primitive plan: " + primitive_summary
        if rationale:
            return rationale

    for line in reasoning.get("current_state_analysis") or []:
        text = str(line or "").strip()
        lower = text.lower()
        if any(
            token in lower
            for token in (
                "reachable by",
                "not reachable",
                "outside xarm6",
                "outside ur5e",
                "exceeds",
                "occupied by",
                "holding=mcp",
                "holding mcp",
            )
        ):
            _append_fact(facts, text)
        if len(facts) >= 2:
            break

    for row in reasoning.get("blocked_transitions") or []:
        if not isinstance(row, dict):
            continue
        why = str(row.get("why_state_is_insufficient") or "").strip()
        if not why:
            continue
        why_lower = why.lower()
        if any(
            token in why_lower
            for token in (
                "occupied",
                "recovery_required",
                "not reachable",
                "outside",
                "assembled before",
                "return to assembly board",
                "simultaneous",
                "co-occupancy",
                "must vacate",
                "executor not free",
                "gripper",
            )
        ):
            _append_fact(facts, why)
        if len(facts) >= 4:
            break

    if rationale and facts:
        return rationale + " Why: " + "; ".join(facts[:3])
    if rationale:
        return rationale
    if facts:
        return " ".join(facts[:3])
    return ""


def _summarize_outline_result(
    outline: dict[str, Any] | None,
    *,
    max_actions: int = 6,
) -> str:
    """Return a concise ordered summary of a task-level repair outline."""
    outline = dict(outline or {})
    reasoning = dict(outline.get("reasoning") or {})
    action_rows = list(reasoning.get("outline_actions") or [])
    labels: list[str] = []
    for row in action_rows:
        if not isinstance(row, dict):
            continue
        action_id = str(row.get("action_id") or "").strip()
        objective = str(row.get("objective") or "").strip()
        phase_type = str(row.get("phase_type") or "").strip()
        label = action_id or objective or phase_type
        if label:
            labels.append(label)
        if len(labels) >= max_actions:
            break
    total_actions = len([row for row in action_rows if isinstance(row, dict)])
    if not labels:
        phase_labels = [
            str(row.get("phase_type") or "").strip()
            for row in (reasoning.get("abstract_repair_order") or [])[:max_actions]
            if isinstance(row, dict) and str(row.get("phase_type") or "").strip()
        ]
        labels = phase_labels
        total_actions = len(
            [
                row
                for row in (reasoning.get("abstract_repair_order") or [])
                if isinstance(row, dict)
            ]
        )
    if not labels:
        return ""
    summary = " -> ".join(labels)
    if total_actions > len(labels):
        summary += " -> ..."
    return summary


def _public_observation_results(
    rows: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    public_rows: list[dict[str, Any]] = []
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        semantic_operation = str(row.get("semantic_operation") or "").strip()
        target_entity = str(row.get("target_entity") or "").strip()
        observation = row.get("observation")
        if semantic_operation and target_entity:
            public_rows.append({
                "semantic_operation": semantic_operation,
                "target_entity": target_entity,
                "observation": observation,
            })
        else:
            public_rows.append(dict(row))
    return public_rows


def _log_react_turn(turn_idx: int, turn_debug: dict[str, Any]) -> None:
    """Print a concise ReAct-style summary line for one turn."""
    try:
        response_type = turn_debug.get("response_type", "")
        error = turn_debug.get("error", "")

        raw = turn_debug.get("raw_response")
        if raw is None and turn_debug.get("auto_observe"):
            raw = turn_debug.get("policy_decision")
        thought = _summarize_turn_thought(
            raw,
            response_type=response_type,
            validation=turn_debug.get("validation"),
        )

        if thought:
            if turn_debug.get("auto_observe"):
                time_suffix = " (policy)"
            else:
                latency = turn_debug.get("llm_latency_s")
                time_suffix = (
                    f" ({latency:.1f}s)" if isinstance(latency, (int, float)) else ""
                )
            _react_logger.info("[Turn %d] Thought: %s%s", turn_idx, thought[:280], time_suffix)

        if error:
            action_label = response_type or "error"
            _react_logger.info("[Turn %d] %s: %s", turn_idx, action_label, error[:200])
            return

        if response_type == "observe":
            obs = turn_debug.get("observation") or {}
            prim = str(turn_debug.get("observe_action_label") or "").strip()
            if not prim and isinstance(raw, dict):
                content = raw.get("content") or raw
                if isinstance(content, str):
                    try:
                        content = json.loads(content)
                    except Exception:
                        content = {}
                if isinstance(content, dict):
                    requests = content.get("observe_requests")
                    if not isinstance(requests, list):
                        legacy = content.get("observe_request")
                        requests = [legacy] if isinstance(legacy, dict) else []
                    labels = [
                        _observe_request_action_label(request)
                        for request in requests[:3]
                        if isinstance(request, dict)
                    ]
                    prim = ", ".join(label for label in labels if label)
            _react_logger.info("[Turn %d] Action: observe %s", turn_idx, prim)
            results = []
            if isinstance(obs, dict):
                results = list(obs.get("results") or [])
            if results:
                summary_payload: Any = _public_observation_results(results[:3]) or results[:3]
                summary = json.dumps(summary_payload, default=str, ensure_ascii=False)
                if len(summary) > 200:
                    summary = summary[:200] + "..."
                _react_logger.info("[Turn %d] Result: %s", turn_idx, summary)

        elif response_type == "repair_outline":
            _react_logger.info("[Turn %d] Action: repair_outline", turn_idx)
            if turn_debug.get("outline_accepted"):
                outline = turn_debug.get("outline") or {}
                outline_summary = _summarize_outline_result(outline)
                if outline_summary:
                    _react_logger.info(
                        "[Turn %d] Result: ACCEPTED  outline=%s",
                        turn_idx,
                        outline_summary[:240],
                    )
                else:
                    _react_logger.info("[Turn %d] Result: ACCEPTED  task-level outline stored", turn_idx)

        elif response_type == "repair_program":
            program = turn_debug.get("program") or {}
            program_summary = _summarize_hybrid_repair_program(program)
            _react_logger.info(
                "[Turn %d] Action: repair_program (%s)",
                turn_idx, program_summary,
            )

            validation = turn_debug.get("validation") or {}
            is_valid = validation.get("is_valid", False)
            if is_valid:
                risk = validation.get("risk_level", "?")
                approval = validation.get("requires_operator_approval", False)
                _react_logger.info(
                    "[Turn %d] Accepted: risk=%s, approval=%s",
                    turn_idx, risk, approval,
                )
            else:
                reasons = validation.get("rejection_reasons", [])
                reason_msgs = [
                    str(r.get("message", "")).strip()
                    for r in reasons[:3]
                ]
                _react_logger.info(
                    "[Turn %d] Rejected: %s",
                    turn_idx, "; ".join(reason_msgs)[:200],
                )
    except Exception:
        pass  # Never let logging break the session.


_DEFAULT_MAX_OBSERVATIONS = 3
_DEFAULT_MAX_OBSERVE_BATCH = 3
_DEFAULT_REPAIR_MODE = "recover"
_DEFAULT_AUTO_OBSERVE = True


# ---------------------------------------------------------------------------
# Session state dataclass (plain dict for simplicity + JSON serialization)
# ---------------------------------------------------------------------------

def _new_repair_session(
    *,
    session_id: str = "",
    max_turns: int = _DEFAULT_MAX_TURNS,
    max_observations: int = _DEFAULT_MAX_OBSERVATIONS,
    max_observe_batch: int = _DEFAULT_MAX_OBSERVE_BATCH,
    repair_mode: str = _DEFAULT_REPAIR_MODE,
    auto_observe: bool = _DEFAULT_AUTO_OBSERVE,
) -> dict[str, Any]:
    """Create a fresh v2 repair session state dict."""
    return {
        "session_id": session_id or f"repair_{uuid4().hex[:8]}",
        "version": 3,
        "turn_index": 0,
        "max_turns": max_turns,
        "max_observations": max_observations,
        "max_observe_batch": max(1, min(3, int(max_observe_batch or _DEFAULT_MAX_OBSERVE_BATCH))),
        "repair_mode": (
            str(repair_mode or _DEFAULT_REPAIR_MODE).strip().lower()
            or _DEFAULT_REPAIR_MODE
        ),
        "auto_observe": bool(auto_observe),
        "observation_count": 0,
        "observation_store": {},
        "observation_history": [],
        "discovered_constraints": [],
        "last_rejected_proposal": None,
        "best_validated_program": None,
        "accepted_outline": None,
        "accepted_outline_context_fingerprint": "",
        "rejection_history": [],
        "v3_prompt_mode": "repair_ready",
        "status": "running",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# Outline helpers
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# UniversalRepairSessionMixin
# ---------------------------------------------------------------------------

class UniversalRepairSessionMixin:
    """Mixin providing ``run_universal_repair_session()``.

    Designed to be mixed into ``ProcessPlanner``.  Reuses
    ``self.product_agent``, ``self.logger``, ``self.resource_agents``,
    and ``self._resource_by_jid`` / ``self._set_last_bridge_debug``.
    """

    # ------------------------------------------------------------------
    # Observation execution (self-contained, no v1 dependency)
    # ------------------------------------------------------------------

    async def _execute_repair_observation(
        self,
        prepared_bridge_request: dict[str, Any],
        *,
        action: dict[str, Any],
    ) -> tuple[dict[str, Any] | None, str | None]:
        """Execute a single observation primitive on a resource.

        Simplified from the v1 ``_execute_bridge_observation_turn`` — does
        NOT call ``_refresh_bridge_grounding_context`` since v2 builds its
        own context via :func:`build_recovery_context`.

        Returns ``(observation_row, error_string)``.
        """
        resource_jid = str(action.get("resource_jid", "") or "").strip()
        primitive = str(action.get("primitive", "") or "").strip()
        params = deepcopy(action.get("params") or {})

        resource = self._resource_by_jid(resource_jid)
        if resource is None:
            return None, f"bridge observation targeted unknown resource '{resource_jid}'"
        execute_observation = getattr(resource, "execute_bridge_observation", None)
        if not callable(execute_observation):
            return None, f"resource '{resource_jid}' does not expose execute_bridge_observation"

        observation_response = execute_observation(primitive, params)
        if asyncio.iscoroutine(observation_response):
            observation_response = await observation_response
        if not isinstance(observation_response, dict):
            return None, f"bridge observation '{primitive}' returned invalid payload"
        if not observation_response.get("success", False):
            return None, str(observation_response.get("message", "") or f"{primitive} failed")

        observation = deepcopy(observation_response.get("observation") or {})
        if not isinstance(observation, dict):
            return None, f"bridge observation '{primitive}' produced no normalized observation"

        # Update resource snapshot in prepared_bridge_request.
        bridge_resources = deepcopy(prepared_bridge_request.get("bridge_resources") or {})
        resource_entry = dict(bridge_resources.get(resource_jid) or {})
        snapshot = deepcopy(
            observation_response.get("snapshot")
            or (resource.get_bridge_snapshot() if hasattr(resource, "get_bridge_snapshot") else {})
            or {}
        )
        resource_entry["bridge_snapshot"] = snapshot
        bridge_resources[resource_jid] = resource_entry
        prepared_bridge_request["bridge_resources"] = bridge_resources
        if resource_jid == str(prepared_bridge_request.get("ra_jid", "") or "").strip():
            prepared_bridge_request["bridge_snapshot"] = deepcopy(snapshot)

        # Update part tracker with observation data.
        part_tracker = deepcopy(prepared_bridge_request.get("part_tracker") or {})
        observed_part_name = str(
            observation.get("part_name") or params.get("part_name") or ""
        ).strip()
        if observed_part_name:
            entry = dict(part_tracker.get(observed_part_name) or {})
            if isinstance(observation.get("pose"), dict):
                entry["observed_pose"] = deepcopy(observation["pose"])
                entry["pose_source"] = "live_observation"
            part_tracker[observed_part_name] = entry
            prepared_bridge_request["part_tracker"] = part_tracker

        alias = str(action.get("store_as", "") or "").strip()
        if not alias:
            alias = f"observation_{int(time.monotonic())}"

        observation_row = {
            "resource_jid": resource_jid,
            "primitive": primitive,
            "params": deepcopy(params),
            "store_as": alias,
            "observation": deepcopy(observation),
            "reason_summary": str(action.get("reason_summary", "") or "").strip(),
        }
        return observation_row, None

    async def run_universal_repair_session(
        self,
        prepared_bridge_request: dict[str, Any],
        *,
        recovery_library: RecoveryLibrary | None = None,
    ) -> dict[str, Any]:
        """Run the TSS-enriched repair session (v3).

        Uses structured LLM output (constrained decoding), projection tool
        (mid-turn simulation), bridge feedback summary (partitioned
        rejections), delta prompts (cached static sections), and turn cache
        with duplicate-turn suppression.

        Parameters
        ----------
        prepared_bridge_request:
            Standard prepared bridge request dict.
        recovery_library:
            Optional persistent recovery library for candidate retrieval
            and post-execution registration.

        Returns
        -------
        dict:
            Result dict with keys: ``status``, ``validated_program``,
            ``session``, ``bridge_debug``.
        """
        # Delegate to the v3 session implementation.
        return await run_v3_repair_session(
            self, prepared_bridge_request,
            recovery_library=recovery_library,
        )

    # ------------------------------------------------------------------
    # Validation helper
    # ------------------------------------------------------------------

    def _run_repair_validation(
        self,
        *,
        program: RepairProgram,
        recovery_context: Any,
        prepared_bridge_request: dict[str, Any],
        recovery_library: RecoveryLibrary | None = None,
    ) -> ValidatedRepairProgram:
        """Run the two-layer validator on a repair program.

        Gathers the necessary inputs from the planner + bridge request
        and delegates to :func:`validate_repair_program`.
        """
        # Gather current task graph nodes.
        current_nodes = list(getattr(self, "nodes", None) or [])

        # Gather primitive catalogs from recovery context.
        primitive_catalogs = dict(recovery_context.available_primitives)

        # Gather resource snapshots.
        resource_snapshots = dict(recovery_context.resource_snapshots)

        # Gather capability flags per resource.
        capability_flags_map: dict[str, dict[str, bool]] = {}
        for ra in (getattr(self, "resource_agents", None) or []):
            jid = str(getattr(ra, "jid", "")).strip()
            if not jid:
                continue
            caps = getattr(ra, "static_capabilities", None)
            if isinstance(caps, dict):
                capability_flags_map[jid] = {
                    k: bool(v) for k, v in caps.items()
                    if isinstance(v, bool)
                }

        # Safety validator (PlanSafetyValidator).
        safety_validator = getattr(self, "plan_safety_validator", None)

        # Safety rules from bridge request.
        safety_ctx = prepared_bridge_request.get("bridge_safety_context") or {}
        safety_rules = list(safety_ctx.get("safety_rules") or [])

        # Runtime monitor state for continuation viability.
        runtime_monitor = getattr(self, "online_fsa_monitor", None)
        runtime_monitor_state: dict[str, Any] | None = None
        if runtime_monitor is not None:
            runtime_monitor_state = {
                "completed_task_ids": list(
                    getattr(runtime_monitor, "completed_task_ids", None) or []
                ),
                "running_task_ids": list(
                    getattr(runtime_monitor, "running_task_ids", None) or []
                ),
                "failed_task_ids": list(
                    getattr(runtime_monitor, "failed_task_ids", None) or []
                ),
            }

        # FSA compilation function.
        compile_fsa_fn = getattr(self, "compile_global_fsa", None)

        # Workspace bounds per resource for feasibility check.
        ws_bounds: dict[str, dict[str, float]] = {}
        for jid, snap in resource_snapshots.items():
            wb = snap.get("workspace_bounds")
            if isinstance(wb, dict):
                ws_bounds[jid] = wb

        product_geometry = dict(getattr(self, "product_geometry", None) or {})
        if not product_geometry:
            product_geometry = dict(
                getattr(getattr(self, "product_agent", None), "product_geometry", None) or {}
            )

        return validate_repair_program(
            program,
            primitive_catalogs=primitive_catalogs,
            resource_snapshots=resource_snapshots,
            current_nodes=current_nodes,
            available_task_actions=list(recovery_context.available_task_actions),
            observation_store=dict(recovery_context.observation_store),
            capability_flags_map=capability_flags_map,
            safety_validator=safety_validator,
            safety_rules=safety_rules,
            runtime_monitor_state=runtime_monitor_state,
            compile_fsa_fn=compile_fsa_fn,
            recovery_library=recovery_library,
            active_obligations=list(recovery_context.active_obligations),
            part_states=dict(recovery_context.part_states),
            product_geometry=product_geometry,
            workspace_bounds=ws_bounds if ws_bounds else None,
            grounded_environment_facts=dict(recovery_context.grounded_environment_facts or {}),
        )



# ---------------------------------------------------------------------------
# Schema section (metadata-driven)
# ---------------------------------------------------------------------------

# Payload examples keyed by RepairStepKind value.  When new kinds are added
# to the enum the prompt stays in sync automatically — just add an entry here.
_STEP_KIND_PAYLOAD_EXAMPLES: dict[str, str] = {
    RepairStepKind.CALL_FUNCTION.value: (
        '"payload": {"function_name": "...", "resource_jid": "...", "args": {}}'
    ),
    RepairStepKind.TASK_MUTATION.value: (
        '"payload": {"mutation_type": "insert|delete|reassign|replace_suffix", '
        '"target_task_ids": [...], "payload": {...}}'
    ),
    RepairStepKind.WAIT.value: (
        '"payload": {"until": {"entity_kind": "...", "entity": "...", '
        '"field": "...", "expected": "..."}}'
    ),
    RepairStepKind.RESUME_SUFFIX.value: '"payload": {}',
}


def _build_steps_schema_block() -> str:
    """Generate the ``steps`` array example from :class:`RepairStepKind`."""
    lines: list[str] = []
    for kind in RepairStepKind:
        payload_example = _STEP_KIND_PAYLOAD_EXAMPLES.get(kind.value, '"payload": {}')
        sep = "    " if not lines else "    | "
        lines.append(f'{sep}{{"kind": "{kind.value}", {payload_example}}}')
    return "  \"steps\": [\n" + "\n".join(lines) + "\n  ]"


def _valid_step_kinds_csv() -> str:
    """Comma-separated list of valid step kind values."""
    return ", ".join(f"`{k.value}`" for k in RepairStepKind)


_REPAIR_PROGRAM_SCHEMA_SECTION = (
    """\
## RepairProgram JSON Schema

```json
{
  "type": "repair_program",
  "reasoning": {
    "current_state_analysis": ["..."],
    "goal_gap_analysis": ["..."],
    "blocked_transitions": [
      {
        "transition": "...",
        "affected_entities": ["..."],
        "why_state_is_insufficient": "...",
        "requires_observation": true,
        "smallest_observation_batch": 1
      }
    ],
    "abstract_repair_order": [
      {
        "phase_type": "resolve_safety|restore_capability|free_executor|recover_entities|restore_resume_entry|adapt_goals|replace_suffix|resume_modeled_suffix",
        "objective": "...",
        "target_entities": ["..."],
        "advances_obligations": ["..."]
      }
    ],
    "safety_check": ["..."]
  },
  "function_defs": [
    {
      "name": "string (unique function name)",
      "intent": "string (what this function does)",
      "resource_constraints": {"resource_type": "string"},
      "inputs": {},
      "preconditions": {
        "field_name": {"equals": value} | {"not_equals": value} | {"exists": true}
      },
      "effects": {
        "field_name": {"set": value}
      },
      "primitive_program": [
        {
          "primitive": "string (must be in available primitives)",
          "params": {"key": "value"},
          "store_as": "optional_output_key"
        }
      ],
      "expected_post_state": {
        "field_name": "expected_value"
      }
    }
  ],
"""
    + _build_steps_schema_block()
    + """,
  "success_conditions": null,
  "rationale": "string (brief explanation of approach)"
}
```

### Binding Rules
- `store_as` writes to function-local scope
- `context_ref` resolves: (1) prior store_as outputs, (2) function inputs, (3) observation store
- Forward references to later store_as are NOT allowed
- Cross-function references are NOT allowed; use function inputs

### Mutation Types
- `insert`: Add new tasks. Payload: `{"new_tasks": [{"function_name", "resource_jid", "params", ...}]}`
- `delete`: Remove pending tasks. Provide `target_task_ids`.
- `reassign`: Move tasks to different resource. Payload: `{"replacements": [{"old_task_id", "new_resource_jid", "function_name", "params"}]}`
- `replace_suffix`: Replace all pending tasks. Payload: `{"new_suffix": [{"function_name", "resource_jid", "params", ...}]}`
"""
)

_REPAIR_PROGRAM_FORMAT_REMINDER = f"""\
## Structured Output Reminder
- The API already enforces the structured response schema. Do not restate the schema.
- Use exact repair-program keys:
  `function_defs[].name`, `function_defs[].intent`, `function_defs[].primitive_program`,
  `steps[].kind`, `steps[].payload`, and `steps[].payload.function_name/resource_jid`.
- Valid `steps[].kind` values are: {_valid_step_kinds_csv()}.
  Do not use any other kind value (e.g. `execute`, `run`).
- Do not use legacy aliases such as `function_name`, `description`, `primitives`,
  `fn`, or `resume_suffix: true` in place of the required keys.
- In repair_program, `reasoning.safety_check` is the primary reasoning artifact.
  `reasoning.abstract_repair_order`, `reasoning.current_state_analysis`,
  and `reasoning.goal_gap_analysis` are optional and should be omitted unless
  validator feedback specifically requires them.
- Do not emit extra step-by-step state narration in repair_program. The runtime
  validates directly from `function_defs` and `steps`.
- If you omit `reasoning.abstract_repair_order`, the runtime will reuse the accepted
  repair_outline phase order for validation.
- Use `rationale` to explain primitive refinement deltas only: which required
  parameters or primitive ordering changed from the accepted outline or prior
  rejection. Do not restate the accepted outline in prose.
- `success_conditions` are auto-derived from bridge-goal obligations; set to
  null or omit. You do not need to emit them.
- Emit `resume_suffix` only as the final step and only after the repair prefix
  already satisfies every must-satisfy-before-resume obligation.
- Use `task_mutation` only for pending-task edits with mutation_type
  `insert`, `delete`, `reassign`, or `replace_suffix`."""


# ---------------------------------------------------------------------------
# V3 system instruction (includes structured reasoning requirements)
# ---------------------------------------------------------------------------

_V3_GROUNDING_FIRST_SYSTEM_INSTRUCTION = """\
You are a recovery planner for a multi-robot manufacturing system.
This turn is in grounding-first mode: inspect the current grounding gaps and decide
which targeted observation to request before task-level planning.

In grounding-first mode you must emit:
  1. observe — request one to three targeted sensor observations

Do not emit `repair_outline` or `repair_program` in this stage. After the
required grounding step succeeds, the next stage is `repair_outline`.

Use the grounding gaps, candidate observations, and recent observations below.
Do not invent scenario-specific scripts. Reason from current state, obligations,
continuation needs, reachability hints, and available capabilities only.

Only observe when uncertainty blocks a concrete executable transition. Name the
blocked transition explicitly and request only the smallest observation batch
needed to unblock that transition."""

_V3_OUTLINE_READY_SYSTEM_INSTRUCTION = """\
You are a recovery planner for a multi-robot manufacturing system.
This turn is the task-level repair stage. Produce either:
  1. observe — only if additional grounding is still required
  2. repair_outline — a task/state-level recovery outline with no primitives

Do not emit function_defs, steps, or success_conditions in this stage.
Reason from current state, obligations, degraded resources, reachability, and
continuation needs. Commit to the abstract repair order before any primitive
refinement."""

_V3_REPAIR_READY_SYSTEM_INSTRUCTION = """\
You are a recovery planner for a multi-robot manufacturing system.
Reason from current state gaps and obligations. Produce either:
  1. observe — if additional grounding materially reduces uncertainty
  2. repair_program — a complete repair program with function_defs and steps

Do not hardcode scenario-specific scripts. Synthesize from current state,
relevant capabilities, trusted observations, and validator feedback.

Before primitive synthesis, commit to a task/state-level repair order:
resolve safety conflicts first, then free or restore the recovery executor,
then recover displaced entities, then restore resume-entry conditions,
and only then resume or replace the modeled suffix.

In repair_program, use `rationale` to describe primitive-level refinement deltas
from the accepted outline or prior rejection, not to restate the outline."""

_V3_OUTLINE_READY_DELTA_SYSTEM_INSTRUCTION = """\
You are continuing a repair session. Use the fresh outline-ready context below and
revise only the failing parts of the previous task-level repair outline.
Do not emit primitives yet."""

_V3_REPAIR_READY_DELTA_SYSTEM_INSTRUCTION = """\
You are continuing a repair session. Use the fresh repair-ready context below and \
revise only the failing parts of the previous proposal.
Fill only the reasoning fields that are still informative before proposing.
Use `rationale` to describe the low-level fixes you made in this revision
(for example required parameters added or primitive ordering corrected),
not to restate the accepted outline."""

_V3_OUTLINE_REASONING_INSTRUCTION = """\
## Task-Level Repair Analysis Requirements
Before proposing the outline, fill the `reasoning` object compactly:

1. `outline_actions` — this is the primary task-level artifact. Name the
   ordered recovery actions with `action_id`, `phase_type`, `objective`,
   `target_entities`, and `must_complete_before`.
2. `safety_check` — 1-2 concise lines explaining why the outline respects the
   active safety rules.
3. `current_state_analysis` and `goal_gap_analysis` are optional. Include them
   only if they add new information beyond outline_actions.
4. `abstract_repair_order` is optional. If omitted, the
   runtime will derive grouped phase blocks from `outline_actions`.

Do not fill primitive-level `transition_plan` in this stage; use null or []."""

_V3_GROUNDING_REASONING_INSTRUCTION = """\
## Grounding Analysis Requirements
Before proposing any actions, fill the `reasoning` object with compact entries:

1. `current_state_analysis` — only the resources/parts relevant to the grounding gap.
2. `goal_gap_analysis` — only the obligations directly affected by this observation choice.
3. `blocked_transitions` — name the blocked executable transition and whether observation is required.
4. `abstract_repair_order` — give only the minimal task/state phases needed to justify the next sensing or pose-independent repair step.
5. `transition_plan` — use null or [] in this stage.
6. `safety_check` — explain why the proposed sensing or repair step respects each active safety rule.

Keep every entry short and factual."""

_V3_REASONING_INSTRUCTION = """\
## Planning Analysis Requirements
Before proposing any actions, provide the shortest reasoning object that still
supports deterministic validation:

1. `safety_check` — explain why the proposal respects each safety rule.
2. Do not emit extra step-by-step state narration in repair_program. The
   runtime will validate directly from `function_defs` and `steps`.
3. `abstract_repair_order`, `current_state_analysis`,
   and `goal_gap_analysis` are optional in repair_program. Omit them unless they
   add new information beyond the accepted outline and validator feedback.
4. If you omit `abstract_repair_order`, the runtime will reuse the accepted
   repair_outline phase order.

Keep the entries compact and factual."""

_V3_REPAIR_READY_OBSERVE_REMINDER = """\
## Repair-Ready Observation Reminder
- Emit `observe` in repair_ready only when the observation policy below lists an
  admissible semantic observation candidate that would materially reduce current
  execution uncertainty.
- If you emit `observe`, use `observe_requests` with 1-3 semantic requests only.
  Do not invent concrete `primitive` names like `compute_*` or `lookup_*` as
  top-level observe requests.
- In `observe.reasoning`, include the same compact grounding fields required in
  grounding-first mode:
  `goal_gap_analysis`, `blocked_transitions`, `abstract_repair_order`,
  `transition_plan` (null or []), and `safety_check`.
- If no admissible semantic observation candidates are listed below, emit
  `repair_program`, not `observe`.
"""

_V3_REPAIR_PROGRAM_GUIDANCE = """\
## Repair Program Guidance
- Implement the accepted outline with synthesized primitive-backed functions.
- Prefer one synthesized function per accepted outline action. If one outline
  action genuinely needs multiple functions, prefix the extra names with
  `<action_id>__`.
- `steps` should call synthesized functions in accepted-outline order, then end
  with `resume_suffix` only after the repair prefix is complete.
- Do not call nominal task-level actions directly in `repair_program`. If a
  nominal action would help, synthesize its primitive effect explicitly inside a
  function_def instead.
- `effects` and `expected_post_state` must use only resource-level fields that the
  primitive catalog can verify, such as held_part, gripper_state, current_state,
  current_pose_ref, or occupancy.location.
- Part-state goals are auto-derived from obligations; do not put them in `effects`.
- Each function_def should perform one logical operation and should be assigned
  to a resource that can actually execute its primitives.
- Respect primitive affordances exactly:
  `move_to_named_pose` is empty-hand only; `grasp_part` requires empty hand;
  `release_part` requires that the resource is already holding a part.
- `move_relative` is best for short local approach/retreat motions. For loaded
  transport after `grasp_part`, prefer motion primitives whose preconditions
  allow a carried part; do not insert empty-hand-only motion after grasp.
- For non-goal staging/release actions, treat the staging token as an explicit
  release anchor unless the current grounded facts say the destination is
  geometry-backed. If a staging destination is `anchor_only`, do not use
  `compute_place_targets` from that token alone; move to the named destination
  (or another grounded anchor), descend, `release_part`, and retreat. If the
  staging facts include explicit `product_geometry` for an `anchor_only`
  destination, use that exact object with `compute_place_targets` instead of
  relying on `destination_location` resolution.
- `rationale` should name the primitive refinement delta: what low-level pieces
  were added or corrected to implement the accepted outline.
- `resume_suffix` is only allowed after the repair prefix already satisfies every
  must-satisfy-before-resume obligation in projected state.
- When placing or releasing a part, use coordinates grounded in physical
  reality. If the part context includes `origin_pose`, that is the last
  known surface where the part was successfully placed or picked from —
  prefer it over invented coordinates. Do not fabricate arbitrary poses."""


def _render_under_modeled_repair_guidance(repair_context_payload: dict[str, Any]) -> str:
    return ""


def _format_pose_tuple(pose: dict[str, Any] | None) -> str:
    row = dict(pose or {})
    try:
        return (
            f"({float(row.get('x', 0.0)):.4f}, "
            f"{float(row.get('y', 0.0)):.4f}, "
            f"{float(row.get('z', 0.0)):.4f})"
        )
    except (TypeError, ValueError):
        return "(unknown)"


def _render_observed_part_fact_lines(
    ctx_dict: dict[str, Any],
    *,
    relevant_parts: set[str] | None = None,
) -> str:
    grounded = dict(ctx_dict.get("grounded_environment_facts") or {})
    observed = dict(grounded.get("observed_part_sources") or {})
    lines: list[str] = []
    for part_name, row in observed.items():
        token = str(part_name).strip()
        if relevant_parts and token not in relevant_parts:
            continue
        info = dict(row or {})
        pose_text = _format_pose_tuple(info.get("observed_pose"))
        source = str(info.get("pose_source") or "").strip()
        line = f"- `{token}`: observed_pose={pose_text}"
        if source:
            line += f", pose_source=`{source}`"
        lines.append(line)
    if not lines:
        return ""
    return "## Observed Part Facts\n" + "\n".join(lines)


def _render_required_destination_fact_lines(
    ctx_dict: dict[str, Any],
    *,
    relevant_parts: set[str] | None = None,
) -> str:
    grounded = dict(ctx_dict.get("grounded_environment_facts") or {})
    destinations = dict(grounded.get("required_destinations") or {})
    lines: list[str] = []
    for part_name, destination in destinations.items():
        token = str(part_name).strip()
        if relevant_parts and token not in relevant_parts:
            continue
        lines.append(f"- `{token}`: destination=`{str(destination).strip()}`")
    if not lines:
        return ""
    return "## Required Destination Facts\n" + "\n".join(lines)


def _render_staging_destination_fact_lines(
    ctx_dict: dict[str, Any],
    *,
    relevant_parts: set[str] | None = None,
) -> str:
    grounded = dict(ctx_dict.get("grounded_environment_facts") or {})
    destinations = dict(grounded.get("staging_destinations") or {})
    detail_rows = {
        str(name).strip(): dict(row)
        for name, row in dict(grounded.get("staging_destination_details") or {}).items()
        if str(name).strip()
    }
    lines: list[str] = []
    saw_anchor_only = False
    saw_geometry_backed = False
    for part_name, destination in destinations.items():
        token = str(part_name).strip()
        if relevant_parts and token not in relevant_parts:
            continue
        destination_token = str(destination).strip()
        if not token or not destination_token:
            continue
        detail = detail_rows.get(token) or {}
        placement_support = str(detail.get("placement_support") or "").strip()
        line = f"- `{token}`: staging_destination=`{destination_token}`"
        if placement_support:
            line += f", placement_support=`{placement_support}`"
        lines.append(line)
        explicit_geometry = dict(detail.get("explicit_product_geometry") or {})
        if explicit_geometry:
            lines.append(
                f"- `{token}`: explicit_product_geometry={json.dumps(explicit_geometry, sort_keys=True)}"
            )
        if placement_support == "anchor_only":
            saw_anchor_only = True
        elif placement_support == "geometry_backed":
            saw_geometry_backed = True
    if not lines:
        return ""
    if saw_anchor_only:
        lines.append(
            "- `anchor_only` means the staging token is a non-goal release anchor, "
            "not a geometry-backed place target. Use an explicit destination witness "
            "such as move-to-destination -> descend -> `release_part` -> retreat, "
            "unless you also provide explicit `product_geometry`. If "
            "`explicit_product_geometry` is listed above, reuse that exact object."
        )
    if saw_geometry_backed:
        lines.append(
            "- `geometry_backed` means the symbolic staging destination can resolve "
            "place geometry for that part, so `compute_place_targets` is admissible."
        )
    return "## Staging Destination Facts\n" + "\n".join(lines)


def _render_degradation_fact_lines(
    ctx_dict: dict[str, Any],
    *,
    relevant_resources: set[str] | None = None,
) -> str:
    grounded = dict(ctx_dict.get("grounded_environment_facts") or {})
    rows = list(grounded.get("degradation_facts") or [])
    lines: list[str] = []
    for item in rows:
        if not isinstance(item, dict):
            continue
        resource_jid = str(item.get("resource_jid") or "").strip()
        if relevant_resources and resource_jid not in relevant_resources:
            continue
        parts: list[str] = [f"`{resource_jid}`"]
        state = str(item.get("resource_state") or "").strip()
        availability = str(item.get("availability") or "").strip()
        reason = str(item.get("reason") or "").strip()
        if state:
            parts.append(f"resource_state=`{state}`")
        if availability:
            parts.append(f"availability=`{availability}`")
        if reason:
            parts.append(f"reason={reason}")
        lines.append("- " + ", ".join(parts))
    if not lines:
        return ""
    return "## Resource Degradation Facts\n" + "\n".join(lines)


def _render_resource_prompt_addenda(
    recovery_context: Any,
    *,
    relevant_resource_ids: set[str] | None = None,
) -> str:
    rows: list[str] = []
    seen_types: set[str] = set()
    for resource_jid, snapshot in dict(recovery_context.resource_snapshots or {}).items():
        if relevant_resource_ids and resource_jid not in relevant_resource_ids:
            continue
        resource_type = str(snapshot.get("resource_type") or "").strip().lower()
        if not resource_type or resource_type in seen_types:
            continue
        profile = get_resource_profile(resource_type)
        addendum = str(getattr(profile, "prompt_addendum", "") or "").strip()
        if not addendum:
            continue
        seen_types.add(resource_type)
        rows.append(f"### {resource_type}\n{addendum}")
    if not rows:
        return ""
    return "## Resource Composition Addenda\n" + "\n\n".join(rows)


def _render_repair_ready_observation_policy(
    recovery_context: Any,
) -> list[tuple[str, str]]:
    assessment = build_grounding_assessment(recovery_context)
    semantic_candidates = _compact_semantic_observation_candidates(
        list(assessment.get("semantic_observation_candidates") or []),
        limit=3,
    )
    if not semantic_candidates:
        return [
            (
                "repair_ready_observation_policy",
                "## Repair-Ready Observation Policy\n"
                "No additional grounding observations are currently admissible from the "
                "current state. Emit `repair_program`, not `observe`. If you need "
                "geometry or pose helper outputs, use the available primitives inside "
                "`function_defs[].primitive_program` instead of top-level "
                "`observe_requests`.\n"
                "- Reuse observation handles exactly as they appear in `## Observation Results` "
                "when writing `context_ref` paths (for example `auto_obs_lg_t1.pose.x`).\n"
                "- If you want a new local alias such as `lg_obs`, first create it with a "
                "same-function primitive `store_as`; do not reference undeclared helper names.",
            )
        ]

    sections: list[tuple[str, str]] = [
        (
            "repair_ready_observation_policy",
            "## Repair-Ready Observation Options\n"
            "If additional grounding is still required, use only these semantic "
            "observation operations. The runtime will bind them to the best "
            "admissible observer.\n"
            + json.dumps(semantic_candidates, indent=2, default=str),
        )
    ]
    contracts = _semantic_observation_contract_rows(semantic_candidates)
    if contracts:
        sections.append(
            (
                "repair_ready_semantic_observation_contracts",
                "## Repair-Ready Semantic Observation Contracts\n"
                + json.dumps(contracts, indent=2, default=str),
            )
        )
    sections.append(
        ("repair_ready_observe_reminder", _V3_REPAIR_READY_OBSERVE_REMINDER)
    )
    return sections

_V3_GROUNDING_FIRST_RESPONSE_REMINDER = """\
## Grounding-First Response Reminder
- If you emit `observe`, use `observe_requests` with 1-3 requests.
- Prefer semantic observe requests such as
  `{"semantic_operation": "observe_part_pose", "target_entity": "LG", "params": {"part_name": "LG"}}`.
  The runtime will bind them to the recommended resource/primitive.
- Grounding-first mode permits `observe` only. The next admissible planning
  stage after grounding is `repair_outline`.
- Choose only the smallest observation batch that materially reduces grounding gaps.
- Each observe request must be tied to a named blocked transition and affected
  entity from `reasoning.blocked_transitions`.
- Every observe request must use the exact parameter names shown in the
  observation primitive contracts and include any listed required_params.
"""


def _observe_request_target_entity(request: dict[str, Any]) -> str:
    target_entity = str(request.get("target_entity") or "").strip()
    if target_entity:
        return target_entity
    params = request.get("params") or {}
    if not isinstance(params, dict):
        return ""
    return str(
        params.get("part_name")
        or params.get("target_entity")
        or params.get("resource_jid")
        or ""
    ).strip()


def _observe_request_action_label(request: dict[str, Any]) -> str:
    semantic_operation = str(request.get("semantic_operation") or "").strip()
    target_entity = _observe_request_target_entity(request)
    if semantic_operation and target_entity:
        return f"{semantic_operation}({target_entity})"
    primitive_name = str(request.get("primitive") or "").strip()
    resource = str(request.get("resource_jid") or "").strip()
    if primitive_name and resource:
        return f"{primitive_name} on {resource}"
    if primitive_name:
        return primitive_name
    return "observe"


def _bind_semantic_observe_requests(
    recovery_context: Any,
    observe_requests: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], str | None]:
    assessment = build_grounding_assessment(recovery_context)
    semantic_candidates = {
        (
            str(row.get("semantic_operation") or "").strip(),
            str(row.get("part_name") or "").strip(),
        ): dict(row)
        for row in (assessment.get("semantic_observation_candidates") or [])
        if isinstance(row, dict)
    }

    bound_requests: list[dict[str, Any]] = []
    for index, request in enumerate(observe_requests, start=1):
        normalized = deepcopy(request or {})
        params = normalized.get("params") or {}
        if not isinstance(params, dict):
            params = {}
        semantic_operation = str(normalized.get("semantic_operation") or "").strip()
        target_entity = _observe_request_target_entity(normalized)

        if not semantic_operation:
            semantic_operation = observation_semantic_operation(
                primitive_name=str(normalized.get("primitive") or "").strip(),
                part_name=target_entity or str(params.get("part_name") or "").strip(),
            )
        if semantic_operation == "observe_part_pose" and target_entity and not params.get("part_name"):
            params["part_name"] = target_entity

        normalized["semantic_operation"] = semantic_operation
        normalized["target_entity"] = target_entity
        normalized["params"] = params

        resource_jid = str(normalized.get("resource_jid") or "").strip()
        primitive = str(normalized.get("primitive") or "").strip()
        if resource_jid and primitive:
            bound_requests.append(normalized)
            continue

        if not semantic_operation:
            return [], (
                f"observe_requests[{index}] must provide either a concrete binding or a semantic_operation"
            )
        candidate = semantic_candidates.get((semantic_operation, target_entity))
        recommended_binding = dict(candidate.get("recommended_binding") or {})
        bound_resource = str(recommended_binding.get("resource_jid") or "").strip()
        bound_primitive = str(recommended_binding.get("primitive") or "").strip()
        if not bound_resource or not bound_primitive:
            return [], (
                f"observe_requests[{index}] could not be bound for "
                f"{semantic_operation}({target_entity or '?'})"
            )
        normalized["resource_jid"] = bound_resource
        normalized["primitive"] = bound_primitive
        normalized["bound_via_semantic_policy"] = True
        bound_requests.append(normalized)

    return bound_requests, None


def _auto_observe_params_for_binding(
    recovery_context: Any,
    *,
    target_entity: str,
    resource_jid: str,
    primitive_name: str,
) -> dict[str, Any] | None:
    catalog = list(
        (recovery_context.available_primitives or {}).get(resource_jid) or []
    )
    entry = next(
        (
            dict(item)
            for item in catalog
            if isinstance(item, dict)
            and str(item.get("name") or "").strip() == primitive_name
        ),
        None,
    )
    if entry is None:
        return None

    _, required_params = _observation_contract_from_entry(entry)
    params: dict[str, Any] = {}
    for raw_name in required_params:
        name = str(raw_name or "").strip()
        if not name:
            continue
        if name == "part_name":
            params[name] = target_entity
            continue
        if name in {"part_names", "targets"}:
            params[name] = [target_entity]
            continue
        if name in {"target_entity", "resource_jid"}:
            params[name] = target_entity
            continue
        return None
    return params


def _maybe_auto_observe_grounding_response(
    recovery_context: Any,
    *,
    turn_index: int,
) -> dict[str, Any] | None:
    assessment = build_grounding_assessment(recovery_context)
    required_parts = [
        str(part_name).strip()
        for part_name in (assessment.get("required_observation_parts") or [])
        if str(part_name).strip()
    ]
    semantic_candidates = [
        dict(row)
        for row in (assessment.get("semantic_observation_candidates") or [])
        if isinstance(row, dict)
    ]
    if len(required_parts) != 1 or len(semantic_candidates) != 1:
        return None

    candidate = semantic_candidates[0]
    part_name = str(candidate.get("part_name") or "").strip()
    semantic_operation = str(candidate.get("semantic_operation") or "").strip()
    recommended_binding = dict(candidate.get("recommended_binding") or {})
    resource_jid = str(recommended_binding.get("resource_jid") or "").strip()
    primitive_name = str(recommended_binding.get("primitive") or "").strip()
    if (
        not part_name
        or part_name != required_parts[0]
        or not semantic_operation
        or not resource_jid
        or not primitive_name
    ):
        return None

    params = _auto_observe_params_for_binding(
        recovery_context,
        target_entity=part_name,
        resource_jid=resource_jid,
        primitive_name=primitive_name,
    )
    if params is None:
        return None

    gap = next(
        (
            dict(row)
            for row in (assessment.get("grounding_gaps") or [])
            if isinstance(row, dict)
            and str(row.get("part_name") or "").strip() == part_name
        ),
        {},
    )
    blocked_transition = str(
        candidate.get("blocked_transition")
        or gap.get("blocked_transition")
        or f"ground {part_name}"
    ).strip()
    why_state_is_insufficient = str(
        gap.get("why_state_is_insufficient")
        or candidate.get("observation_policy_reason")
        or f"{part_name} requires a trusted observation before pose-dependent recovery."
    ).strip()
    observation_status = str(gap.get("observation_status") or "UNOBSERVED").strip()
    store_alias = f"auto_obs_{part_name.lower()}_t{turn_index}"

    return {
        "type": "observe",
        "reasoning": {
            "current_state_analysis": [
                (
                    f"{part_name} observation_status={observation_status} blocks "
                    f"'{blocked_transition}'."
                ),
            ],
            "goal_gap_analysis": [
                (
                    f"Acquire a trusted observation for {part_name} before choosing "
                    "a pose-dependent recovery action."
                ),
            ],
            "blocked_transitions": [
                {
                    "transition": blocked_transition,
                    "affected_entities": list(
                        gap.get("affected_entities") or [part_name]
                    ),
                    "why_state_is_insufficient": why_state_is_insufficient,
                    "requires_observation": True,
                    "smallest_observation_batch": int(
                        gap.get("smallest_admissible_observation_batch") or 1
                    ),
                }
            ],
            "abstract_repair_order": [],
            "transition_plan": [],
            "safety_check": [
                (
                    "The selected observation is the smallest admissible sensing "
                    "step and leaves robot state unchanged."
                )
            ],
        },
        "observe_requests": [
            {
                "semantic_operation": semantic_operation,
                "target_entity": part_name,
                "params": params,
                "store_as": store_alias,
            }
        ],
        "rationale": (
            "Deterministic grounding policy selected the single admissible "
            f"observation to unblock planning for {part_name}."
        ),
    }

_V3_OUTLINE_RESPONSE_REMINDER = """\
## Repair-Outline Response Reminder
- If you emit `repair_outline`, keep it task/state-level only.
- Do not emit `function_defs`, `steps`, or `success_conditions`.
- Include `reasoning.outline_actions` so the next turn can refine named
  task-level actions instead of recomputing the decomposition.
- Prefer one named action when a single operation discharges multiple closely
  related preparatory obligations, instead of splitting it into redundant
  "recover" and "vacate" actions unless they are truly separate steps.
- End the outline with `resume_modeled_suffix` or `replace_suffix`.
- If unresolved bridge-goal parts remain, either include `recover_entities`
  before `resume_modeled_suffix` or end with `replace_suffix`.
- If executor feasibility is blocked, the outline must resolve that before
  `recover_entities`."""


# ---------------------------------------------------------------------------
# V3 helpers
# ---------------------------------------------------------------------------

def _v3_unobserved_critical_parts(
    self: UniversalRepairSessionMixin,
    prepared_bridge_request: dict[str, Any],
    observation_history: list[dict[str, Any]],
) -> list[str]:
    """Return critical parts that have not been observed in the v3 session.

    Uses :meth:`_bridge_critical_parts` from ``BridgeSafetyMixin`` to
    identify parts that must be localized and checks the session-local
    ``observation_history`` for matching observations.
    """
    try:
        critical = self._bridge_critical_parts(prepared_bridge_request)
    except Exception:
        return []

    observed_parts: set[str] = set()
    part_tracker = dict(prepared_bridge_request.get("part_tracker") or {})
    for part_name, raw_info in part_tracker.items():
        info = raw_info if isinstance(raw_info, dict) else {}
        if part_observation_status(info) == "observed":
            token = str(part_name or "").strip()
            if token:
                observed_parts.add(token)

    for obs in observation_history:
        obs_data = obs.get("observation") or {}
        if isinstance(obs_data, dict):
            pn = str(obs_data.get("part_name", "")).strip()
            if pn:
                observed_parts.add(pn)
        params = obs.get("params") or {}
        if isinstance(params, dict):
            pn = str(params.get("part_name", "")).strip()
            if pn:
                observed_parts.add(pn)

    return [p for p in critical if p not in observed_parts]


def _compute_v3_prompt_mode(
    recovery_context: Any,
    session_state: dict[str, Any],
) -> str:
    assessment = build_grounding_assessment(recovery_context)
    accepted_outline = session_state.get("accepted_outline")
    accepted_outline_context_fingerprint = str(
        session_state.get("accepted_outline_context_fingerprint") or ""
    ).strip()
    current_outline_context_fingerprint = str(
        session_state.get("current_outline_context_fingerprint")
        or session_state.get("current_context_fingerprint")
        or ""
    ).strip()
    remaining_observations = max(
        0,
        int(session_state.get("max_observations", _DEFAULT_MAX_OBSERVATIONS))
        - int(session_state.get("observation_count", 0)),
    )
    if assessment.get("required_observation_parts") and remaining_observations > 0:
        return "grounding_first"
    if (
        isinstance(accepted_outline, dict)
        and accepted_outline
        and accepted_outline_context_fingerprint
        and accepted_outline_context_fingerprint == current_outline_context_fingerprint
    ):
        return "repair_ready"
    return "outline_ready"


def _compact_capability_rows(
    recovery_context: Any,
    *,
    relevant_resource_ids: set[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for resource_jid, catalog in (recovery_context.available_primitives or {}).items():
        if relevant_resource_ids and resource_jid not in relevant_resource_ids:
            continue
        observation_primitives: list[str] = []
        operation_kinds: set[str] = set()
        for entry in catalog or []:
            if not isinstance(entry, dict):
                continue
            primitive_name = str(entry.get("name") or "").strip()
            if not primitive_name:
                continue
            semantics = entry.get("bridge_semantics") or {}
            operation_kind = str(semantics.get("operation_kind") or "").strip()
            if operation_kind:
                operation_kinds.add(operation_kind)
            if semantics.get("top_level_observation_admissible"):
                observation_primitives.append(primitive_name)
        rows.append({
            "resource_jid": resource_jid,
            "observation_primitives": sorted(set(observation_primitives)),
            "operation_kinds": sorted(operation_kinds),
        })
    return rows


def _observation_contract_from_entry(entry: dict[str, Any]) -> tuple[dict[str, str], list[str]]:
    params_info: dict[str, str] = {}
    raw_params = entry.get("params")
    if isinstance(raw_params, dict):
        params_info = {
            str(key): value.get("type", "any") if isinstance(value, dict) else str(value)
            for key, value in raw_params.items()
            if str(key).strip()
        }
    raw_params_schema = entry.get("params_schema")
    if not params_info and isinstance(raw_params_schema, dict):
        params_info = {
            str(key): value.get("type", "any") if isinstance(value, dict) else str(value)
            for key, value in raw_params_schema.items()
            if str(key).strip()
        }
    raw_parameters = entry.get("parameters")
    if not params_info and isinstance(raw_parameters, dict):
        properties = raw_parameters.get("properties") or {}
        if isinstance(properties, dict):
            params_info = {
                str(key): value.get("type", "any") if isinstance(value, dict) else str(value)
                for key, value in properties.items()
                if str(key).strip()
            }

    required = [
        str(name)
        for name in (entry.get("required_params") or [])
        if str(name).strip()
    ]
    if not required and isinstance(raw_parameters, dict):
        required = [
            str(name)
            for name in (raw_parameters.get("required") or [])
            if str(name).strip()
        ]
    if not required and isinstance(raw_params_schema, dict):
        required = [str(name) for name in raw_params_schema.keys() if str(name).strip()]
    return params_info, required


def _observation_primitive_contract_rows(
    recovery_context: Any,
    *,
    relevant_resource_ids: set[str],
    allowed_pairs: set[tuple[str, str]] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for resource_jid, catalog in (recovery_context.available_primitives or {}).items():
        if relevant_resource_ids and resource_jid not in relevant_resource_ids:
            continue
        primitives: list[dict[str, Any]] = []
        for entry in catalog or []:
            if not isinstance(entry, dict):
                continue
            semantics = entry.get("bridge_semantics") or {}
            if not semantics.get("top_level_observation_admissible"):
                continue
            primitive_name = str(entry.get("name") or "").strip()
            if not primitive_name:
                continue
            if allowed_pairs is not None and (str(resource_jid), primitive_name) not in allowed_pairs:
                continue
            primitive_row: dict[str, Any] = {"name": primitive_name}
            params_info, required = _observation_contract_from_entry(entry)
            if required:
                primitive_row["required_params"] = required
            if params_info:
                primitive_row["params"] = params_info
            output_schema = semantics.get("observation_output_schema")
            if output_schema:
                primitive_row["output_schema"] = output_schema
            primitives.append(primitive_row)
        if primitives:
            rows.append({
                "resource_jid": resource_jid,
                "primitives": primitives,
            })
    return rows


def _compact_semantic_observation_candidates(
    rows: list[dict[str, Any]] | None,
    *,
    limit: int,
) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for row in list(rows or [])[:limit]:
        if not isinstance(row, dict):
            continue
        compact.append({
            "semantic_operation": row.get("semantic_operation"),
            "part_name": row.get("part_name"),
            "blocked_transition": row.get("blocked_transition"),
            "affected_entities": row.get("affected_entities") or [],
            "observation_policy_reason": row.get("observation_policy_reason"),
            "priority_score": row.get("priority_score"),
            "binding_policy": "runtime binds this semantic observation to the best admissible observer",
        })
    return compact


def _semantic_observation_contract_rows(
    rows: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    operations: list[str] = []
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        token = str(row.get("semantic_operation") or "").strip()
        if token and token not in operations:
            operations.append(token)

    contracts: list[dict[str, Any]] = []
    for operation in operations:
        if operation == "observe_part_pose":
            contracts.append({
                "semantic_operation": operation,
                "required_params": ["part_name"],
                "output_schema": {
                    "part_name": "string",
                    "pose": "object",
                },
            })
        elif operation == "observe_resource_pose":
            contracts.append({
                "semantic_operation": operation,
                "required_params": ["resource_jid"],
                "output_schema": {
                    "resource_jid": "string",
                    "pose": "object",
                },
            })
        else:
            contracts.append({
                "semantic_operation": operation,
                "required_params": [],
            })
    return contracts


def _compact_observation_history_rows(
    observation_history: list[dict[str, Any]] | None,
    *,
    limit: int = 5,
) -> list[dict[str, Any]]:
    compact_obs: list[dict[str, Any]] = []
    for row in list(observation_history or [])[-limit:]:
        if not isinstance(row, dict):
            continue
        compact_row: dict[str, Any] = {
            "store_as": row.get("store_as"),
            "observation": row.get("observation"),
        }
        semantic_operation = str(row.get("semantic_operation") or "").strip()
        observation = dict(row.get("observation") or {})
        params = dict(row.get("params") or {})
        target_entity = str(
            row.get("target_entity")
            or observation.get("part_name")
            or params.get("part_name")
            or ""
        ).strip()
        if not semantic_operation:
            semantic_operation = observation_semantic_operation(
                primitive_name=str(row.get("primitive") or "").strip(),
                part_name=target_entity or None,
            )
        if semantic_operation:
            compact_row["semantic_operation"] = semantic_operation
        if target_entity:
            compact_row["target_entity"] = target_entity
        if not semantic_operation:
            compact_row["primitive"] = row.get("primitive")
        compact_obs.append(compact_row)
    return compact_obs


def _compact_outline_ready_context_payload(ctx_dict: dict[str, Any]) -> dict[str, Any]:
    resources: dict[str, Any] = {}
    for resource_jid, raw in dict(ctx_dict.get("resources") or {}).items():
        info = dict(raw or {})
        compact_row: dict[str, Any] = {
            "current_state": info.get("current_state"),
        }
        if info.get("held_part") is not None:
            compact_row["held_part"] = info.get("held_part")
        occupancy = dict(info.get("occupancy") or {})
        if occupancy.get("location") is not None:
            compact_row["location"] = occupancy.get("location")
        if info.get("gripper_state") is not None:
            compact_row["gripper_state"] = info.get("gripper_state")
        if info.get("workspace_bounds") is not None:
            compact_row["workspace_bounds"] = info.get("workspace_bounds")
        resources[str(resource_jid)] = compact_row

    parts: dict[str, Any] = {}
    for part_name, raw in dict(ctx_dict.get("parts") or {}).items():
        info = dict(raw or {})
        compact_row: dict[str, Any] = {
            "state": info.get("state"),
            "location_summary": info.get("location_summary"),
        }
        if info.get("observation_status") is not None:
            compact_row["observation_status"] = info.get("observation_status")
        if info.get("pose_source") is not None:
            compact_row["pose_source"] = info.get("pose_source")
        parts[str(part_name)] = compact_row

    payload: dict[str, Any] = {
        "resources": resources,
        "parts": parts,
        "obligations": list(ctx_dict.get("obligations") or []),
        "goal": ctx_dict.get("goal"),
    }
    return payload


def _compact_auto_observe_policy_input(
    recovery_context: Any,
    *,
    remaining_observations: int,
) -> dict[str, Any]:
    assessment = build_grounding_assessment(recovery_context)
    return {
        "grounding_gaps": [
            {
                "part_name": row.get("part_name"),
                "blocked_transition": row.get("blocked_transition"),
                "affected_entities": row.get("affected_entities") or [],
                "why_state_is_insufficient": row.get("why_state_is_insufficient"),
            }
            for row in list(assessment.get("grounding_gaps") or [])[:3]
            if isinstance(row, dict)
        ],
        "semantic_observation_candidates": _compact_semantic_observation_candidates(
            list(assessment.get("semantic_observation_candidates") or []),
            limit=3,
        ),
        "required_observation_parts": list(
            assessment.get("required_observation_parts") or []
        ),
        "executor_first_parts": list(assessment.get("executor_first_parts") or []),
        "observation_budget_remaining": remaining_observations,
    }


def _recommended_observation_pairs(ctx_dict: dict[str, Any]) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for row in (ctx_dict.get("candidate_observations") or []):
        if not isinstance(row, dict):
            continue
        recommended = row.get("recommended_observer") or {}
        resource_jid = str(recommended.get("resource_jid") or "").strip()
        primitive = str(recommended.get("recommended_primitive") or "").strip()
        if resource_jid and primitive:
            pairs.add((resource_jid, primitive))
    return pairs


def _validate_observe_requests_against_catalog(
    recovery_context: Any,
    observe_requests: list[dict[str, Any]],
) -> str | None:
    catalog_lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for resource_jid, catalog in (recovery_context.available_primitives or {}).items():
        for entry in catalog or []:
            if not isinstance(entry, dict):
                continue
            primitive_name = str(entry.get("name") or "").strip()
            if primitive_name:
                catalog_lookup[(str(resource_jid), primitive_name)] = entry

    for index, request in enumerate(observe_requests, start=1):
        resource_jid = str(request.get("resource_jid") or "").strip()
        primitive = str(request.get("primitive") or "").strip()
        catalog_entry = catalog_lookup.get((resource_jid, primitive))
        if catalog_entry is None:
            return (
                f"observe_requests[{index}] targets unknown observation primitive "
                f"'{primitive}' on resource '{resource_jid}'"
            )
        params = request.get("params") or {}
        if not isinstance(params, dict):
            return f"observe_requests[{index}] params must be an object"
        _, required_params = _observation_contract_from_entry(catalog_entry)
        missing = [
            str(name)
            for name in required_params
            if str(name).strip() and params.get(str(name)) in (None, "", [])
        ]
        if missing:
            return (
                f"observe_requests[{index}] for {resource_jid}.{primitive} missing "
                f"required params: {', '.join(missing)}"
            )
    return None


def _validate_observe_requests_against_grounding_policy(
    recovery_context: Any,
    reasoning: dict[str, Any],
    observe_requests: list[dict[str, Any]],
) -> str | None:
    """Reject exploratory observe requests that do not close a named gap."""
    assessment = build_grounding_assessment(recovery_context)
    grounding_gaps = {
        str(item.get("part_name") or "").strip(): dict(item)
        for item in (assessment.get("grounding_gaps") or [])
        if str(item.get("part_name") or "").strip()
    }
    if not grounding_gaps:
        return (
            "observe is not admissible here because no active grounding gap "
            "is blocking an executable transition"
        )

    blocked_entities: set[str] = set()
    blocked_transitions = list(reasoning.get("blocked_transitions") or [])
    for item in blocked_transitions:
        if not isinstance(item, dict):
            continue
        for entity in (item.get("affected_entities") or []):
            token = str(entity or "").strip()
            if token:
                blocked_entities.add(token)

    targeted_parts: list[str] = []
    for index, request in enumerate(observe_requests, start=1):
        params = request.get("params") or {}
        part_name = str(params.get("part_name") or "").strip()
        if not part_name:
            continue
        targeted_parts.append(part_name)
        gap = grounding_gaps.get(part_name)
        if gap is None:
            return (
                f"observe_requests[{index}] targets '{part_name}', but that entity "
                "is not an active grounding gap"
            )
        if not bool(gap.get("observation_admissible", True)):
            policy_reason = str(gap.get("observation_policy_reason") or "").strip()
            if not policy_reason:
                policy_reason = "the current blocker is not a grounding problem"
            return (
                f"observe_requests[{index}] targets '{part_name}', but observe is not "
                f"admissible here because {policy_reason}"
            )
        if blocked_entities and part_name not in blocked_entities:
            return (
                f"observe_requests[{index}] targets '{part_name}', but reasoning "
                "did not name that entity in a blocked transition"
            )

        allowed_observers = {
            str(option.get("resource_jid") or "").strip(): set(
                str(name).strip()
                for name in (option.get("primitives") or [])
                if str(name).strip()
            )
            for option in (gap.get("observer_options") or [])
            if str(option.get("resource_jid") or "").strip()
        }
        resource_jid = str(request.get("resource_jid") or "").strip()
        primitive = str(request.get("primitive") or "").strip()
        if allowed_observers:
            if resource_jid not in allowed_observers:
                return (
                    f"observe_requests[{index}] uses '{resource_jid}', which is not "
                    f"an admissible observer for '{part_name}'"
                )
            if primitive and primitive not in allowed_observers[resource_jid]:
                return (
                    f"observe_requests[{index}] uses primitive '{primitive}' on "
                    f"'{resource_jid}', but that combination was not recommended "
                    f"for grounding '{part_name}'"
                )

    unique_targeted_parts = {part for part in targeted_parts if part}
    if unique_targeted_parts and len(observe_requests) > len(unique_targeted_parts):
        return (
            "observe_requests exceeds the smallest admissible observation batch "
            "for the named grounding gaps"
        )
    return None


def _compact_task_actions(
    recovery_context: Any,
    *,
    relevant_resource_ids: set[str],
    limit: int = 8,
) -> list[dict[str, Any]]:
    compact_actions: list[dict[str, Any]] = []
    for action in recovery_context.available_task_actions:
        if not isinstance(action, dict):
            continue
        resource_jid = str(action.get("resource_jid") or "").strip()
        if relevant_resource_ids and resource_jid and resource_jid not in relevant_resource_ids:
            continue
        compact_actions.append({
            "function_name": action.get("function_name"),
            "resource_jid": resource_jid,
            "in_state": action.get("in_state"),
            "out_state": action.get("out_state"),
        })
        if len(compact_actions) >= limit:
            break
    return compact_actions


def _focused_repair_resource_ids(
    ctx_dict: dict[str, Any],
    accepted_outline: dict[str, Any] | None,
) -> set[str]:
    resource_ids, _ = _focused_repair_entities(ctx_dict, accepted_outline)
    return resource_ids


def _compact_accepted_outline_payload(accepted_outline: dict[str, Any]) -> dict[str, Any]:
    accepted_reasoning = dict(accepted_outline.get("reasoning") or {})
    outline_actions = accepted_reasoning.get("outline_actions") or []
    if not outline_actions:
        outline_actions = materialize_outline_actions(accepted_reasoning)

    abstract_repair_order = list(accepted_reasoning.get("abstract_repair_order") or [])
    terminal_phase = ""
    if abstract_repair_order and isinstance(abstract_repair_order[-1], dict):
        terminal_phase = str(abstract_repair_order[-1].get("phase_type") or "").strip()

    return {
        "outline_actions": outline_actions,
        "terminal_phase": terminal_phase,
        "rationale": accepted_outline.get("rationale") or "",
    }


def _compact_accepted_outline_retry_payload(
    accepted_outline: dict[str, Any],
) -> dict[str, Any]:
    accepted_reasoning = dict(accepted_outline.get("reasoning") or {})
    outline_actions = accepted_reasoning.get("outline_actions") or []
    if not outline_actions:
        outline_actions = materialize_outline_actions(accepted_reasoning)

    compact_actions = []
    for row in outline_actions:
        if not isinstance(row, dict):
            continue
        compact_actions.append({
            "action_id": str(row.get("action_id") or "").strip(),
            "phase_type": str(row.get("phase_type") or "").strip(),
            "objective": str(row.get("objective") or "").strip(),
        })

    abstract_repair_order = list(accepted_reasoning.get("abstract_repair_order") or [])
    terminal_phase = ""
    if abstract_repair_order and isinstance(abstract_repair_order[-1], dict):
        terminal_phase = str(abstract_repair_order[-1].get("phase_type") or "").strip()

    return {
        "outline_actions": compact_actions,
        "terminal_phase": terminal_phase,
    }


def _focused_repair_entities(
    ctx_dict: dict[str, Any],
    accepted_outline: dict[str, Any] | None,
) -> tuple[set[str], set[str]]:
    resource_ids = {
        str(jid).strip()
        for jid in (ctx_dict.get("resources") or {}).keys()
        if str(jid).strip()
    }
    part_names = {
        str(name).strip()
        for name in (ctx_dict.get("parts") or {}).keys()
        if str(name).strip()
    }
    if not isinstance(accepted_outline, dict) or not accepted_outline:
        return resource_ids, part_names

    accepted_reasoning = dict(accepted_outline.get("reasoning") or {})
    focused_resources: set[str] = set()
    focused_parts: set[str] = set()

    def _collect_tokens(values: list[Any] | None) -> None:
        for value in values or []:
            token = str(value or "").strip()
            if token in resource_ids:
                focused_resources.add(token)
            if token in part_names:
                focused_parts.add(token)

    for action in accepted_reasoning.get("outline_actions") or []:
        if isinstance(action, dict):
            _collect_tokens(list(action.get("target_entities") or []))

    if not focused_resources and not focused_parts:
        for phase in accepted_reasoning.get("abstract_repair_order") or []:
            if isinstance(phase, dict):
                _collect_tokens(list(phase.get("target_entities") or []))

    if not focused_resources and not focused_parts:
        for row in accepted_reasoning.get("blocked_transitions") or []:
            if isinstance(row, dict):
                _collect_tokens(list(row.get("affected_entities") or []))

    return focused_resources or resource_ids, focused_parts or part_names


def _compact_repair_ready_context_payload(
    ctx_dict: dict[str, Any],
    accepted_outline: dict[str, Any] | None,
) -> dict[str, Any]:
    resource_ids, part_names = _focused_repair_entities(ctx_dict, accepted_outline)

    resources: dict[str, Any] = {}
    for resource_jid, raw in dict(ctx_dict.get("resources") or {}).items():
        token = str(resource_jid).strip()
        if token not in resource_ids:
            continue
        info = dict(raw or {})
        compact_row: dict[str, Any] = {
            "current_state": info.get("current_state"),
        }
        if info.get("held_part") is not None:
            compact_row["held_part"] = info.get("held_part")
        occupancy = dict(info.get("occupancy") or {})
        if occupancy.get("location") is not None:
            compact_row["location"] = occupancy.get("location")
        if info.get("gripper_state") is not None:
            compact_row["gripper_state"] = info.get("gripper_state")
        resources[token] = compact_row

    parts: dict[str, Any] = {}
    for part_name, raw in dict(ctx_dict.get("parts") or {}).items():
        token = str(part_name).strip()
        if token not in part_names:
            continue
        info = dict(raw or {})
        compact_row: dict[str, Any] = {
            "state": info.get("state"),
            "location_summary": info.get("location_summary"),
        }
        if info.get("observation_status") is not None:
            compact_row["observation_status"] = info.get("observation_status")
        if info.get("pose_source") is not None:
            compact_row["pose_source"] = info.get("pose_source")
        parts[token] = compact_row

    relevant_entities = resource_ids | part_names
    obligations: list[Any] = []
    for obligation in list(ctx_dict.get("obligations") or []):
        if isinstance(obligation, str):
            # String-formatted obligations (e.g. "[safety] ...", "[bridge-goal] LG.state ...").
            # Include safety obligations unconditionally; for others, include if any
            # relevant entity name appears in the text.
            text = obligation
            if text.startswith("[safety]"):
                obligations.append(obligation)
            elif any(ent in text for ent in relevant_entities):
                obligations.append(obligation)
            elif not relevant_entities:
                obligations.append(obligation)
        elif isinstance(obligation, dict):
            entity = str(obligation.get("entity") or "").strip()
            obligation_class = str(
                obligation.get("obligation_class") or obligation.get("type") or ""
            ).strip()
            if obligation_class == "safety" or not entity or entity in relevant_entities:
                obligations.append(obligation)

    payload: dict[str, Any] = {
        "resources": resources,
        "parts": parts,
        "obligations": obligations,
    }
    executor_first_parts = [
        str(name).strip()
        for name in (ctx_dict.get("executor_first_parts") or [])
        if str(name).strip() in part_names
    ]
    if executor_first_parts:
        payload["executor_first_parts"] = executor_first_parts
    return payload


# ---------------------------------------------------------------------------
# V3 methods on UniversalRepairSessionMixin
# ---------------------------------------------------------------------------

def _execute_projection_tool(
    self: UniversalRepairSessionMixin,
    tool_name: str,
    arguments: dict[str, Any],
    recovery_context: Any,
) -> dict[str, Any]:
    """Execute a projection tool call from the LLM.

    Delegates to :func:`validate_and_project_steps` to simulate a
    primitive sequence on a resource and return the projected snapshot.
    """
    if tool_name != "project_primitive_sequence":
        return {"error": f"Unknown tool: {tool_name}"}

    resource_jid = str(arguments.get("resource_jid", "")).strip()
    steps = arguments.get("steps") or []
    if not resource_jid:
        return {"error": "missing resource_jid"}

    catalog = (recovery_context.available_primitives or {}).get(resource_jid, [])
    snapshot = (recovery_context.resource_snapshots or {}).get(resource_jid, {})
    if not catalog:
        return {"error": f"No primitive catalog for {resource_jid}"}

    is_valid, projected, error = validate_and_project_steps(
        steps, catalog, snapshot,
    )
    return {
        "is_valid": is_valid,
        "projected_snapshot": projected,
        "error": error,
    }


# Attach to the mixin class.
UniversalRepairSessionMixin._execute_projection_tool = _execute_projection_tool  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# V3 prompt builder
# ---------------------------------------------------------------------------

def _build_v3_prompt(
    self: UniversalRepairSessionMixin,
    recovery_context: Any,
    session_state: dict[str, Any],
    *,
    turn_cache: TurnCache | None = None,
    feedback: BridgeFeedbackSummary | None = None,
    recovery_library: RecoveryLibrary | None = None,
) -> str:
    """Build a v3 prompt using soft prompt modes instead of hard phases."""
    turn_idx = session_state["turn_index"]
    max_turns = session_state["max_turns"]
    ctx_dict = recovery_context_to_prompt_dict(recovery_context)
    prompt_mode = _compute_v3_prompt_mode(recovery_context, session_state)
    session_state["v3_prompt_mode"] = prompt_mode
    is_delta = (
        prompt_mode in {"outline_ready", "repair_ready"}
        and turn_idx > 1
        and turn_cache is not None
    )

    relevant_resources = {
        str(jid).strip()
        for jid in (ctx_dict.get("resources") or {}).keys()
        if str(jid).strip()
    }
    accepted_outline = session_state.get("accepted_outline")
    repair_prompt_resources = (
        _focused_repair_resource_ids(ctx_dict, accepted_outline)
        if prompt_mode == "repair_ready"
        else relevant_resources
    )
    max_observations = int(
        session_state.get("max_observations", _DEFAULT_MAX_OBSERVATIONS)
    )
    observation_count = int(session_state.get("observation_count", 0))
    remaining_turns = max(0, max_turns - turn_idx)
    remaining_observations = max(0, max_observations - observation_count)
    max_observe_batch = max(
        1,
        min(
            3,
            int(
                session_state.get(
                    "max_observe_batch",
                    _DEFAULT_MAX_OBSERVE_BATCH,
                )
            ),
        ),
    )

    parts: list[str] = []
    section_sizes: dict[str, int] = {}

    def _append_section(label: str, content: str) -> None:
        if not content:
            return
        parts.append(content)
        section_sizes[label] = len(content)

    if prompt_mode == "grounding_first":
        _append_section("system_instruction", _V3_GROUNDING_FIRST_SYSTEM_INSTRUCTION)
    elif prompt_mode == "outline_ready" and is_delta:
        _append_section("system_instruction", _V3_OUTLINE_READY_DELTA_SYSTEM_INSTRUCTION)
    elif prompt_mode == "outline_ready":
        _append_section("system_instruction", _V3_OUTLINE_READY_SYSTEM_INSTRUCTION)
    elif is_delta:
        _append_section("system_instruction", _V3_REPAIR_READY_DELTA_SYSTEM_INSTRUCTION)
    else:
        _append_section("system_instruction", _V3_REPAIR_READY_SYSTEM_INSTRUCTION)

    _append_section(
        "turn_budget",
        f"Turn {turn_idx}/{max_turns} ({remaining_turns} turns remaining). "
        f"Prompt mode: {prompt_mode}. Observation budget remaining: "
        f"{remaining_observations}/{max_observations}. "
        f"Max observe batch: {max_observe_batch}.",
    )

    observation_history = session_state.get("observation_history") or []

    if prompt_mode == "grounding_first":
        grounding_payload: dict[str, Any] = {
            "goal": ctx_dict.get("goal"),
            "obligations": ctx_dict.get("obligations") or [],
            "grounding_gaps": ctx_dict.get("grounding_gaps") or [],
            "semantic_observation_candidates": _compact_semantic_observation_candidates(
                list(ctx_dict.get("semantic_observation_candidates") or []),
                limit=max_observe_batch,
            ),
            "observation_budget_remaining": remaining_observations,
        }
        if ctx_dict.get("required_observation_parts"):
            grounding_payload["required_observation_parts"] = (
                ctx_dict.get("required_observation_parts") or []
            )
        if ctx_dict.get("executor_first_parts"):
            grounding_payload["executor_first_parts"] = (
                ctx_dict.get("executor_first_parts") or []
            )
        _append_section(
            "grounding_context",
            "## Current Grounding Context\n"
            + json.dumps(grounding_payload, indent=2, default=str),
        )

        observation_contract_rows = _semantic_observation_contract_rows(
            grounding_payload.get("semantic_observation_candidates") or []
        )
        if observation_contract_rows:
            _append_section(
                "semantic_observation_contracts",
                "## Semantic Observation Contracts\n"
                "Use these semantic operations in observe_requests. The runtime "
                "will bind them to the best admissible observer.\n"
                + json.dumps(observation_contract_rows, indent=2, default=str),
            )

        if observation_history:
            compact_obs = _compact_observation_history_rows(observation_history)
            _append_section(
                "recent_observations",
                "## Recent Observation Results\n"
                + json.dumps(compact_obs, indent=2, default=str),
            )
        observed_fact_section = _render_observed_part_fact_lines(
            ctx_dict,
            relevant_parts={str(name).strip() for name in (ctx_dict.get("parts") or {}).keys()},
        )
        if observed_fact_section:
            _append_section("observed_part_facts", observed_fact_section)
        degradation_fact_section = _render_degradation_fact_lines(
            ctx_dict,
            relevant_resources={str(jid).strip() for jid in (ctx_dict.get("resources") or {}).keys()},
        )
        if degradation_fact_section:
            _append_section("degradation_facts", degradation_fact_section)

        _append_section("reasoning_instruction", _V3_GROUNDING_REASONING_INSTRUCTION)
        _append_section("grounding_response_reminder", _V3_GROUNDING_FIRST_RESPONSE_REMINDER)
        session_state["v3_prompt_sections"] = section_sizes
        return "\n\n".join(parts)

    if prompt_mode == "outline_ready":
        repair_context_payload = _compact_outline_ready_context_payload(ctx_dict)
        if ctx_dict.get("executor_first_parts"):
            repair_context_payload["executor_first_parts"] = ctx_dict.get("executor_first_parts")
    else:
        repair_context_payload = _compact_repair_ready_context_payload(
            ctx_dict,
            accepted_outline if isinstance(accepted_outline, dict) else None,
        )
    _append_section(
        "repair_context",
        "## Current Repair Context\n"
        + json.dumps(repair_context_payload, indent=2, default=str),
    )
    repair_context_parts = {
        str(name).strip() for name in (repair_context_payload.get("parts") or {}).keys()
    }
    repair_context_resources = {
        str(jid).strip() for jid in (repair_context_payload.get("resources") or {}).keys()
    }
    degradation_fact_section = _render_degradation_fact_lines(
        ctx_dict,
        relevant_resources=repair_context_resources,
    )
    if degradation_fact_section:
        _append_section("degradation_facts", degradation_fact_section)

    if prompt_mode == "repair_ready" and isinstance(accepted_outline, dict) and accepted_outline:
        outline_payload = (
            _compact_accepted_outline_retry_payload(accepted_outline)
            if feedback is not None
            else _compact_accepted_outline_payload(accepted_outline)
        )
        _append_section(
            "accepted_outline",
            "## Accepted Repair Outline\n"
            "The primitive-level repair_program must refine this accepted task/state outline.\n"
            + json.dumps(outline_payload, indent=2, default=str),
        )
        destination_fact_section = _render_required_destination_fact_lines(
            ctx_dict,
            relevant_parts=repair_context_parts,
        )
        if destination_fact_section:
            _append_section("required_destination_facts", destination_fact_section)
        staging_fact_section = _render_staging_destination_fact_lines(
            ctx_dict,
            relevant_parts=repair_context_parts,
        )
        if staging_fact_section:
            _append_section("staging_destination_facts", staging_fact_section)

    if prompt_mode == "outline_ready":
        if feedback is not None:
            _append_section("feedback", feedback_to_prompt_section(feedback))
        if observation_history:
            compact_obs = _compact_observation_history_rows(observation_history)
            _append_section(
                "observation_results",
                "## Observation Results\n"
                + json.dumps(compact_obs, indent=2, default=str),
            )
        observed_fact_section = _render_observed_part_fact_lines(
            ctx_dict,
            relevant_parts=repair_context_parts,
        )
        if observed_fact_section:
            _append_section("observed_part_facts", observed_fact_section)
        destination_fact_section = _render_required_destination_fact_lines(
            ctx_dict,
            relevant_parts=repair_context_parts,
        )
        if destination_fact_section:
            _append_section("required_destination_facts", destination_fact_section)
        staging_fact_section = _render_staging_destination_fact_lines(
            ctx_dict,
            relevant_parts=repair_context_parts,
        )
        if staging_fact_section:
            _append_section("staging_destination_facts", staging_fact_section)
        _append_section("outline_reasoning_instruction", _V3_OUTLINE_REASONING_INSTRUCTION)
        _append_section("outline_response_reminder", _V3_OUTLINE_RESPONSE_REMINDER)
        session_state["v3_prompt_sections"] = section_sizes
        return "\n\n".join(parts)

    if is_delta:
        assert turn_cache is not None
        excerpt = _render_primitive_catalogs_v3(
            recovery_context,
            relevant_resource_ids=repair_prompt_resources,
            repair_ready_only=True,
            heading="## Available Primitives (cached excerpt)",
        )
        if excerpt:
            _append_section("cached_catalog_excerpt", excerpt)
        reminder = turn_cache.render_schema_reminder()
        if reminder:
            if reminder.strip() != _REPAIR_PROGRAM_FORMAT_REMINDER.strip():
                _append_section("schema_reminder", reminder)
        if feedback is not None:
            _append_section("feedback", feedback_to_prompt_section(feedback))
    else:
        _append_section(
            "primitive_catalogs",
            _render_primitive_catalogs_v3(
                recovery_context,
                relevant_resource_ids=repair_prompt_resources,
                repair_ready_only=True,
            ),
        )
    addenda_section = _render_resource_prompt_addenda(
        recovery_context,
        relevant_resource_ids=repair_prompt_resources,
    )
    if addenda_section:
        _append_section("resource_addenda", addenda_section)

    for section_key, section_body in _render_repair_ready_observation_policy(
        recovery_context
    ):
        _append_section(section_key, section_body)

    _append_section("format_reminder", _REPAIR_PROGRAM_FORMAT_REMINDER)

    if recovery_library is not None and prompt_mode != "repair_ready":
        resource_types = set()
        for snap in recovery_context.resource_snapshots.values():
            rt = str(snap.get("resource_type", "")).strip()
            if rt:
                resource_types.add(rt)
        all_candidates: list[dict[str, Any]] = []
        for rt in resource_types:
            candidates = recovery_library.candidates_for_prompt(
                resource_profile_id=rt, max_entries=3,
            )
            all_candidates.extend(candidates)
        if all_candidates:
            _append_section(
                "library_candidates",
                "## Library Candidates (previously validated functions)\n"
                "You may reuse or adapt these. They will still be fully "
                "validated.\n"
                + json.dumps(all_candidates, indent=2, default=str),
            )

    if observation_history:
        compact_obs = [
            {
                "primitive": h.get("primitive"),
                "store_as": h.get("store_as"),
                "observation": h.get("observation"),
            }
            for h in observation_history[-5:]
        ]
        _append_section(
            "observation_results",
            "## Observation Results\n"
            + json.dumps(compact_obs, indent=2, default=str),
        )
        observed_fact_section = _render_observed_part_fact_lines(
            ctx_dict,
            relevant_parts=repair_context_parts,
        )
        if observed_fact_section:
            _append_section("observed_part_facts", observed_fact_section)

    if feedback is not None and not is_delta:
        _append_section("feedback", feedback_to_prompt_section(feedback))

    if feedback is not None:
        witness_section = open_witnesses_to_prompt_section(feedback)
        if witness_section:
            _append_section("open_repair_witnesses", witness_section)

    _append_section("reasoning_instruction", _V3_REASONING_INSTRUCTION)
    _append_section("repair_program_guidance", _V3_REPAIR_PROGRAM_GUIDANCE)
    session_state["v3_prompt_sections"] = section_sizes
    return "\n\n".join(parts)


def _render_primitive_catalogs_v3(
    recovery_context: Any,
    *,
    relevant_resource_ids: set[str] | None = None,
    repair_ready_only: bool = False,
    heading: str = "## Available Primitives per Resource",
) -> str:
    """Render primitive catalogs in the same format as the v2 prompt builder."""
    if repair_ready_only:
        family_rows: dict[str, dict[str, Any]] = {}
        family_order: list[str] = []
        for jid, catalog in (recovery_context.available_primitives or {}).items():
            if relevant_resource_ids and jid not in relevant_resource_ids:
                continue
            if not catalog:
                continue
            for entry in catalog:
                if not isinstance(entry, dict):
                    continue
                name = str(entry.get("name", "")).strip()
                if not name:
                    continue
                semantics = entry.get("bridge_semantics") or {}
                if name not in family_rows:
                    family_rows[name] = {
                        "name": name,
                        "supported_resources": [],
                        "required_params": [],
                        "params": {},
                        "preconditions": {},
                        "effects": {},
                        "produces_observation": False,
                        "output_schema": {},
                        "semantic_summary": "",
                    }
                    family_order.append(name)
                family = family_rows[name]
                if jid not in family["supported_resources"]:
                    family["supported_resources"].append(jid)
                for param in entry.get("required_params") or []:
                    token = str(param).strip()
                    if token and token not in family["required_params"]:
                        family["required_params"].append(token)
                for param_name, param_info in dict(entry.get("params") or {}).items():
                    token = str(param_name).strip()
                    if not token:
                        continue
                    if isinstance(param_info, dict):
                        family["params"][token] = param_info.get("type", "any")
                    else:
                        family["params"][token] = str(param_info)
                if not family["preconditions"] and isinstance(entry.get("preconditions"), dict):
                    family["preconditions"] = deepcopy(entry.get("preconditions") or {})
                if not family["effects"] and isinstance(entry.get("effects"), dict):
                    family["effects"] = deepcopy(entry.get("effects") or {})
                if semantics.get("produces_observation"):
                    family["produces_observation"] = True
                    output_schema = semantics.get("observation_output_schema")
                    if not family["output_schema"] and isinstance(output_schema, dict):
                        family["output_schema"] = deepcopy(output_schema)
                if not family["semantic_summary"]:
                    family["semantic_summary"] = str(entry.get("semantic_summary") or "").strip()
                if not family["semantic_summary"]:
                    family["semantic_summary"] = _repair_ready_helper_semantic_summary(name)
        if not family_order:
            return ""
        families = [family_rows[name] for name in family_order]
        return (
            "## Available Primitive Families\n"
            "For repair_program, reason over these shared primitive families first.\n"
            "Bind each primitive to one of its supported_resources in function_defs[].primitive_program.\n"
            "Use only the required_params and param names shown here.\n"
            "Primitives with `produces_observation=true` may write `store_as` outputs for later "
            "same-function `context_ref` use.\n"
            "Pay close attention to each family's preconditions and effects; they are validator-enforced.\n\n"
            + json.dumps(families, indent=2, default=str)
        )

    prim_sections: list[str] = []
    for jid, catalog in (recovery_context.available_primitives or {}).items():
        if relevant_resource_ids and jid not in relevant_resource_ids:
            continue
        if not catalog:
            continue
        prim_entries: list[dict[str, Any]] = []
        for entry in catalog:
            if not isinstance(entry, dict):
                continue
            name = str(entry.get("name", "")).strip()
            if not name:
                continue
            semantics = entry.get("bridge_semantics") or {}
            if repair_ready_only:
                if name.startswith("compute_"):
                    continue
                if semantics.get("produces_observation"):
                    continue
            compact: dict[str, Any] = {"name": name}
            required = entry.get("required_params") or []
            params_info = entry.get("params") or {}
            if required:
                compact["required_params"] = required
            if params_info:
                compact["params"] = {
                    k: v.get("type", "any") if isinstance(v, dict) else str(v)
                    for k, v in params_info.items()
                }
            preconds = entry.get("preconditions")
            if preconds:
                compact["preconditions"] = preconds
            effects = entry.get("effects")
            if effects:
                compact["effects"] = effects
            if semantics.get("produces_observation"):
                compact["produces_observation"] = True
                output_schema = semantics.get("observation_output_schema")
                if output_schema:
                    compact["output_schema"] = output_schema
            prim_entries.append(compact)
        if prim_entries:
            prim_sections.append(
                f"### {jid}\n"
                + json.dumps(prim_entries, indent=2, default=str)
            )
    if not prim_sections:
        return ""
    return (
        f"{heading}\n"
        "Each primitive lists its required_params, preconditions, and effects.\n"
        "Your function primitive_program MUST use only these primitives with correct params.\n"
        "Do NOT use store_as on primitives that do not have produces_observation=true.\n\n"
        + "\n\n".join(prim_sections)
    )


def _render_existing_task_actions_v3(
    recovery_context: Any,
    *,
    relevant_resource_ids: set[str] | None = None,
) -> str:
    rows = list(recovery_context.available_task_actions or [])
    if not rows:
        return ""

    relevant_owner_tokens = {
        str(jid).split("@", 1)[0].strip().lower()
        for jid in (relevant_resource_ids or set())
        if str(jid).strip()
    }
    grouped: dict[str, dict[str, Any]] = {}
    ordered_names: list[str] = []

    for row in rows:
        if not isinstance(row, dict):
            continue
        name = str(row.get("function") or row.get("function_name") or row.get("name") or "").strip()
        owner = str(row.get("function_owner_agent") or row.get("owner") or "").strip().lower()
        if not name:
            continue
        if relevant_owner_tokens and owner and owner not in relevant_owner_tokens:
            continue
        if name not in grouped:
            grouped[name] = {
                "name": name,
                "supported_agents": [],
                "in_state": str(row.get("in_state") or "").strip(),
                "out_state": str(row.get("out_state") or "").strip(),
                "required_context_keys": list(row.get("required_context_keys") or []),
                "description": str(row.get("description") or "").strip(),
            }
            ordered_names.append(name)
        entry = grouped[name]
        if owner and owner not in entry["supported_agents"]:
            entry["supported_agents"].append(owner)
        if not entry["required_context_keys"]:
            entry["required_context_keys"] = list(row.get("required_context_keys") or [])

    if not ordered_names:
        return ""

    payload = [grouped[name] for name in ordered_names]
    return (
        "## Existing Task-Level Actions\n"
        "Prefer reusing one of these nominal actions when it already matches an accepted outline action"
        " such as move_home / pick_approach / pick_grasp / place_approach / place_insert.\n"
        "Only synthesize a new function when no existing action or library candidate fits.\n\n"
        + json.dumps(payload, indent=2, default=str)
    )


# Attach to the mixin class.
UniversalRepairSessionMixin._build_v3_prompt = _build_v3_prompt  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# V3 session loop
# ---------------------------------------------------------------------------

async def run_v3_repair_session(
    self: UniversalRepairSessionMixin,
    prepared_bridge_request: dict[str, Any],
    *,
    recovery_library: RecoveryLibrary | None = None,
) -> dict[str, Any]:
    """Run the v3 TSS-enriched repair session loop.

    Same interface as :meth:`run_universal_repair_session` but uses:
    - Structured LLM output (constrained decoding)
    - Projection tool (mid-turn simulation)
    - Bridge feedback summary (partitioned rejections)
    - Delta prompts (cached static sections)
    - Turn cache with duplicate-turn suppression
    """
    if not isinstance(prepared_bridge_request, dict):
        raise ValueError("prepared bridge request is missing")

    bridge_session_cfg = dict(prepared_bridge_request.get("bridge_session") or {})
    max_turns = max(
        1,
        int(bridge_session_cfg.get("max_turns", _DEFAULT_MAX_TURNS) or _DEFAULT_MAX_TURNS),
    )
    max_observations = max(
        0,
        int(
            bridge_session_cfg.get(
                "max_observations",
                _DEFAULT_MAX_OBSERVATIONS,
            ) or _DEFAULT_MAX_OBSERVATIONS
        ),
    )
    max_observe_batch = max(
        1,
        min(
            3,
            int(
                bridge_session_cfg.get(
                    "max_observe_batch",
                    _DEFAULT_MAX_OBSERVE_BATCH,
                ) or _DEFAULT_MAX_OBSERVE_BATCH
            ),
        ),
    )
    repair_mode = (
        str(bridge_session_cfg.get("repair_mode") or _DEFAULT_REPAIR_MODE)
        .strip()
        .lower()
        or _DEFAULT_REPAIR_MODE
    )
    auto_observe = bool(
        bridge_session_cfg.get("auto_observe", _DEFAULT_AUTO_OBSERVE)
    )
    if repair_mode not in {"recover", "diagnose_first"}:
        repair_mode = _DEFAULT_REPAIR_MODE

    bridge_session_cfg["max_turns"] = max_turns
    bridge_session_cfg["max_observations"] = max_observations
    bridge_session_cfg["max_observe_batch"] = max_observe_batch
    bridge_session_cfg["repair_mode"] = repair_mode
    bridge_session_cfg["auto_observe"] = auto_observe
    prepared_bridge_request["bridge_session"] = bridge_session_cfg

    repair_session = _new_repair_session(
        max_turns=max_turns,
        max_observations=max_observations,
        max_observe_batch=max_observe_batch,
        repair_mode=repair_mode,
        auto_observe=auto_observe,
    )
    session_id = repair_session["session_id"]

    bridge_debug: dict[str, Any] = {
        "session_id": session_id,
        "session_type": "universal_repair_v3",
        "generation_started_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "turns": [],
    }
    prepared_bridge_request["bridge_debug"] = bridge_debug
    self._set_last_bridge_debug(bridge_debug)

    max_turns = int(repair_session["max_turns"])
    max_observations = int(repair_session["max_observations"])
    repair_mode = str(repair_session.get("repair_mode") or _DEFAULT_REPAIR_MODE)
    _session_t0 = time.monotonic()

    _react_logger.info(
        "[RepairV3] Session %s — mode=%s, max_turns=%d, max_observations=%d, max_observe_batch=%d",
        session_id, repair_mode, max_turns, max_observations, max_observe_batch,
    )

    # Write header to incremental debug file.
    try:
        debug_path = _get_debug_dir() / f"v3_session_{session_id}.txt"
        debug_path.write_text(
            f"V3 Repair Session — {session_id}\n"
            f"Started: {datetime.now(timezone.utc).isoformat()}\n"
            f"Repair mode: {repair_mode}\n"
            f"Max turns: {max_turns}, Max observations: {max_observations}, "
            f"Max observe batch: {max_observe_batch}\n",
            encoding="utf-8",
        )
        _react_logger.info("[RepairV3] Debug: %s", debug_path)
    except Exception:
        pass

    # Accumulators.
    discovered_constraints: list[dict[str, Any]] = list(
        repair_session["discovered_constraints"]
    )
    observation_store: dict[str, Any] = dict(repair_session["observation_store"])
    observation_history: list[dict[str, Any]] = list(
        repair_session["observation_history"]
    )
    last_rejected_proposal: dict[str, Any] | None = None
    best_validated: ValidatedRepairProgram | None = None
    rejection_history: list[dict[str, Any]] = []
    accepted_outline: dict[str, Any] | None = deepcopy(
        repair_session.get("accepted_outline")
    )
    accepted_outline_context_fingerprint = str(
        repair_session.get("accepted_outline_context_fingerprint") or ""
    ).strip()

    # V3-specific state.
    turn_cache: TurnCache | None = None
    feedback: BridgeFeedbackSummary | None = None

    while repair_session["turn_index"] < max_turns:
        repair_session["turn_index"] += 1
        turn_idx = repair_session["turn_index"]
        _turn_t0 = time.monotonic()

        # ----- Build fresh RecoveryContext -----
        _ctx_t0 = time.monotonic()
        recovery_context = build_recovery_context(
            prepared_bridge_request,
            planner=self,
            resource_agents={
                str(getattr(ra, "jid", "")).strip(): ra
                for ra in (getattr(self, "resource_agents", None) or [])
                if str(getattr(ra, "jid", "")).strip()
            },
            observation_store=observation_store,
            discovered_constraints=discovered_constraints,
        )
        context_build_s = time.monotonic() - _ctx_t0
        current_context_fingerprint = compute_context_fingerprint(recovery_context)
        current_outline_context_fingerprint = compute_outline_context_fingerprint(
            recovery_context
        )

        # ----- Build session state dict for prompt builder -----
        session_state = {
            "turn_index": turn_idx,
            "max_turns": max_turns,
            "max_observations": max_observations,
            "observation_count": repair_session["observation_count"],
            "max_observe_batch": repair_session["max_observe_batch"],
            "discovered_constraints": discovered_constraints,
            "last_rejected_proposal": last_rejected_proposal,
            "observation_history": observation_history,
            "repair_mode": repair_mode,
            "auto_observe": bool(repair_session.get("auto_observe", _DEFAULT_AUTO_OBSERVE)),
            "accepted_outline": deepcopy(accepted_outline),
            "accepted_outline_context_fingerprint": accepted_outline_context_fingerprint,
            "current_context_fingerprint": current_context_fingerprint,
            "current_outline_context_fingerprint": current_outline_context_fingerprint,
        }

        prompt_mode = _compute_v3_prompt_mode(recovery_context, session_state)
        session_state["v3_prompt_mode"] = prompt_mode
        repair_session["v3_prompt_mode"] = prompt_mode
        remaining_observations = max(
            0,
            max_observations - int(repair_session["observation_count"]),
        )

        auto_observe_response = None
        policy_input: dict[str, Any] | None = None
        if (
            bool(repair_session.get("auto_observe", _DEFAULT_AUTO_OBSERVE))
            and prompt_mode == "grounding_first"
        ):
            auto_observe_response = _maybe_auto_observe_grounding_response(
                recovery_context,
                turn_index=turn_idx,
            )
            if auto_observe_response is not None:
                policy_input = _compact_auto_observe_policy_input(
                    recovery_context,
                    remaining_observations=remaining_observations,
                )
                session_state["v3_prompt_sections"] = {}

        # ----- Build v3 prompt (full or delta) -----
        prompt = ""
        prompt_build_s = 0.0
        if auto_observe_response is None:
            _prompt_t0 = time.monotonic()
            prompt = self._build_v3_prompt(
                recovery_context=recovery_context,
                session_state=session_state,
                turn_cache=turn_cache,
                feedback=feedback,
                recovery_library=recovery_library,
            )
            prompt_build_s = time.monotonic() - _prompt_t0

        turn_debug: dict[str, Any] = {
            "turn_index": turn_idx,
            "prompt": prompt,
            "prompt_length": len(prompt),
            "policy_input": deepcopy(policy_input) if policy_input else None,
            "prompt_sections": deepcopy(session_state.get("v3_prompt_sections") or {}),
            "discovered_constraints_count": len(discovered_constraints),
            "is_delta": bool(
                session_state.get("v3_prompt_mode") in {"outline_ready", "repair_ready"}
                and turn_idx > 1
                and turn_cache is not None
            ),
            "repair_mode": repair_mode,
            "v3_prompt_mode": session_state.get("v3_prompt_mode", "repair_ready"),
            "context_build_s": round(context_build_s, 4),
            "prompt_build_s": round(prompt_build_s, 4),
        }

        def _flush_turn() -> None:
            bridge_debug["turns"].append(turn_debug)
            self._set_last_bridge_debug(bridge_debug)
            _write_turn_debug(
                session_id,
                turn_idx,
                prompt=turn_debug.get("prompt", ""),
                policy_input=turn_debug.get("policy_input"),
                turn_metrics={
                    "prompt_sections": turn_debug.get("prompt_sections") or {},
                    "prompt_length": turn_debug.get("prompt_length"),
                    "v3_prompt_mode": turn_debug.get("v3_prompt_mode"),
                    "auto_observe": turn_debug.get("auto_observe", False),
                    "context_build_s": turn_debug.get("context_build_s"),
                    "prompt_build_s": turn_debug.get("prompt_build_s"),
                    "llm_latency_s": turn_debug.get("llm_latency_s"),
                    "validation_s": turn_debug.get("validation_s"),
                },
                raw_response=turn_debug.get("raw_response"),
                policy_decision=turn_debug.get("policy_decision"),
                error=turn_debug.get("error", ""),
                observation=turn_debug.get("observation"),
                outline=turn_debug.get("outline"),
                outline_validation=turn_debug.get("outline_validation"),
                program=turn_debug.get("program"),
                validation=turn_debug.get("validation"),
            )
            _log_react_turn(turn_idx, turn_debug)

        # ----- Call LLM (structured output + tool loop) -----
        raw_response: Any
        llm_latency_s: float | None = None

        if auto_observe_response is not None:
            raw_response = auto_observe_response
            turn_debug["auto_observe"] = True
            turn_debug["policy_decision"] = deepcopy(auto_observe_response)
            self.logger.debug(
                "[RepairV3] Turn %d/%d — auto-observe selected by grounding policy",
                turn_idx,
                max_turns,
            )
        else:
            self.logger.debug(
                "[RepairV3] Turn %d/%d — sending v3 prompt to LLM (%d chars)...",
                turn_idx, max_turns, len(prompt),
            )
            try:
                _llm_t0 = time.monotonic()
                raw_response = await self.product_agent.ask_llm_structured(
                    prompt=prompt,
                    response_format=REPAIR_TURN_RESPONSE_SCHEMA,
                    tools=[PROJECT_PRIMITIVE_SEQUENCE_TOOL],
                    tool_executor=lambda name, args: self._execute_projection_tool(
                        name, args, recovery_context,
                    ),
                )
                llm_latency_s = time.monotonic() - _llm_t0
            except Exception as exc:
                bridge_debug["status"] = "exception"
                bridge_debug["exception"] = repr(exc)
                turn_debug["exception"] = repr(exc)
                _flush_turn()
                raise

        if llm_latency_s is not None:
            turn_debug["llm_latency_s"] = round(llm_latency_s, 4)
        turn_debug["session_elapsed_s"] = round(
            time.monotonic() - _session_t0, 4,
        )
        turn_debug["raw_response"] = (
            None if auto_observe_response is not None
            else (deepcopy(raw_response) if isinstance(raw_response, dict) else raw_response)
        )

        # ----- Parse structured response -----
        parsed, parse_error = parse_structured_response(
            raw_response if isinstance(raw_response, dict) else {},
        )
        if parse_error:
            self.logger.debug(
                "[RepairV3] Turn %d — parse error: %s", turn_idx, parse_error,
            )
            turn_debug["error"] = parse_error
            _flush_turn()
            continue

        response_type = str(parsed.get("type", "")).strip().lower()
        turn_debug["response_type"] = response_type
        turn_debug["reasoning"] = parsed.get("reasoning")

        # ----- Handle observe -----
        if response_type == "observe":
            if repair_session["observation_count"] >= max_observations:
                obs_error = (
                    f"observation budget exhausted ({max_observations}); "
                    f"emit a repair_program instead"
                )
                discovered_constraints.append({
                    "layer": "session",
                    "check": "observation_budget",
                    "constraint": obs_error,
                })
                turn_debug["error"] = obs_error
                _flush_turn()
                continue

            raw_observe_requests = list(parsed.get("observe_requests") or [])
            observe_requests, obs_error = _bind_semantic_observe_requests(
                recovery_context,
                raw_observe_requests,
            )
            if obs_error:
                turn_debug["error"] = obs_error
                _flush_turn()
                continue
            turn_debug["observe_action_label"] = ", ".join(
                _observe_request_action_label(request)
                for request in observe_requests[:3]
            )
            if len(observe_requests) > repair_session["max_observe_batch"]:
                obs_error = (
                    f"observe_requests count ({len(observe_requests)}) exceeds "
                    f"max_observe_batch ({repair_session['max_observe_batch']})"
                )
                turn_debug["error"] = obs_error
                _flush_turn()
                continue
            remaining_budget = (
                max_observations - repair_session["observation_count"]
            )
            if len(observe_requests) > remaining_budget:
                obs_error = (
                    f"observation budget remaining ({remaining_budget}) is "
                    f"smaller than observe_requests count ({len(observe_requests)})"
                )
                turn_debug["error"] = obs_error
                _flush_turn()
                continue
            obs_error = _validate_observe_requests_against_catalog(
                recovery_context,
                observe_requests,
            )
            if obs_error:
                turn_debug["error"] = obs_error
                _flush_turn()
                continue
            obs_error = _validate_observe_requests_against_grounding_policy(
                recovery_context,
                dict(parsed.get("reasoning") or {}),
                observe_requests,
            )
            if obs_error:
                turn_debug["error"] = obs_error
                _flush_turn()
                continue

            batch_results: list[dict[str, Any]] = []
            batch_errors: list[str] = []
            for request in observe_requests:
                action = {
                    "resource_jid": request.get("resource_jid"),
                    "primitive": request.get("primitive"),
                    "params": request.get("params") or {},
                    "store_as": request.get("store_as", ""),
                }
                obs_row, obs_error = await self._execute_repair_observation(
                    prepared_bridge_request,
                    action=action,
                )
                repair_session["observation_count"] += 1

                if obs_error:
                    batch_errors.append(obs_error)
                    self.logger.debug(
                        "[RepairV3] Turn %d — observe FAILED: %s",
                        turn_idx, obs_error,
                    )
                    continue

                alias = (
                    str(request.get("store_as", "")).strip()
                    or f"obs_{turn_idx}_{len(batch_results) + 1}"
                )
                obs_data = deepcopy((obs_row or {}).get("observation") or {})
                observation_store[alias] = obs_data
                observation_history.append({
                    "turn_index": turn_idx,
                    "semantic_operation": request.get("semantic_operation"),
                    "target_entity": request.get("target_entity"),
                    "primitive": action["primitive"],
                    "resource_jid": action["resource_jid"],
                    "params": deepcopy(action["params"]),
                    "store_as": alias,
                    "observation": obs_data,
                })
                obs_debug_row = deepcopy(obs_row or {})
                obs_debug_row["semantic_operation"] = request.get("semantic_operation")
                obs_debug_row["target_entity"] = request.get("target_entity")
                batch_results.append(obs_debug_row)
                self.logger.debug(
                    "[RepairV3] Turn %d — observe succeeded → %s",
                    turn_idx, alias,
                )

            turn_debug["observation"] = {
                "results": _public_observation_results(batch_results),
                "errors": batch_errors,
            }
            accepted_outline = None
            accepted_outline_context_fingerprint = ""
            ctx_fp = current_context_fingerprint
            rendered_catalogs: dict[str, str] = {}
            for jid, catalog in (recovery_context.available_primitives or {}).items():
                if catalog:
                    rendered_catalogs[jid] = (
                        f"### {jid}\n"
                        + json.dumps(
                            [
                                {"name": e.get("name"), "preconditions": e.get("preconditions"), "effects": e.get("effects")}
                                for e in catalog if isinstance(e, dict)
                            ],
                            indent=2, default=str,
                        )
                    )
            turn_cache = TurnCache(
                turn_index=turn_idx,
                context_fingerprint=ctx_fp,
                program_fingerprint=None,
                rendered_catalog_by_resource=rendered_catalogs,
                rendered_schema_section=_REPAIR_PROGRAM_FORMAT_REMINDER,
                last_feedback=feedback,
                last_rejected_proposal=last_rejected_proposal,
                last_validated_result=None,
            )
            if batch_errors and not batch_results:
                turn_debug["error"] = "; ".join(batch_errors)
            _flush_turn()
            if repair_mode == "diagnose_first" and (
                batch_results or batch_errors
            ):
                repair_session["status"] = "diagnose_first"
                break
            continue

        # ----- Handle repair_outline -----
        if response_type == "repair_outline":
            if session_state.get("v3_prompt_mode") == "grounding_first":
                turn_debug["error"] = (
                    "repair_outline is not admissible before the required grounding step; "
                    "emit observe first"
                )
                _flush_turn()
                continue

            normalized_reasoning = deepcopy(parsed.get("reasoning") or {})
            normalized_reasoning["outline_actions"] = materialize_outline_actions(
                normalized_reasoning
            )
            normalized_reasoning["abstract_repair_order"] = (
                materialize_abstract_repair_order(normalized_reasoning)
            )
            parsed["reasoning"] = normalized_reasoning
            outline_validation = validate_repair_outline(
                parsed,
                recovery_context=recovery_context,
            )
            turn_debug["outline_validation"] = outline_validation.to_dict()
            if not outline_validation.is_valid:
                feedback = BridgeFeedbackSummary(
                    blocked_steps=list(outline_validation.errors),
                    suggested_adaptations=[
                        "Revise the task-level repair order before proposing primitives."
                    ],
                )
                for msg in outline_validation.errors:
                    discovered_constraints.append({
                        "layer": "outline",
                        "check": "repair_outline",
                        "constraint": msg,
                    })
                turn_debug["error"] = "; ".join(outline_validation.errors)
                turn_debug["outline"] = {
                    "reasoning": deepcopy(parsed.get("reasoning") or {}),
                    "rationale": parsed.get("rationale") or "",
                }
                _flush_turn()
                continue

            accepted_outline = {
                "type": "repair_outline",
                "reasoning": deepcopy(parsed.get("reasoning") or {}),
                "rationale": str(parsed.get("rationale") or ""),
            }
            accepted_outline_context_fingerprint = current_outline_context_fingerprint
            feedback = None
            turn_debug["outline"] = deepcopy(accepted_outline)
            turn_debug["outline_accepted"] = True
            _flush_turn()
            continue

        # ----- Handle repair_program -----
        assert response_type == "repair_program"

        if session_state.get("v3_prompt_mode") == "grounding_first":
            turn_debug["error"] = (
                "repair_program is not admissible before the required grounding step; "
                "emit observe first"
            )
            _flush_turn()
            continue

        if not accepted_outline or (
            accepted_outline_context_fingerprint != current_outline_context_fingerprint
        ):
            turn_debug["error"] = (
                "repair_program requires an accepted repair_outline for the current context first"
            )
            _flush_turn()
            continue

        parsed_reasoning = dict(parsed.get("reasoning") or {})
        accepted_outline_reasoning = dict(accepted_outline.get("reasoning") or {})
        if not isinstance(parsed_reasoning.get("abstract_repair_order"), list) or not list(
            parsed_reasoning.get("abstract_repair_order") or []
        ):
            accepted_abstract_order = list(
                accepted_outline_reasoning.get("abstract_repair_order") or []
            )
            if accepted_abstract_order:
                parsed_reasoning["abstract_repair_order"] = deepcopy(
                    accepted_abstract_order
                )
        parsed["reasoning"] = parsed_reasoning

        accepted_outline_signature = outline_signature(
            accepted_outline_reasoning
        )
        program_outline_signature = outline_signature(
            parsed_reasoning
        )
        if (
            accepted_outline_signature
            and program_outline_signature != accepted_outline_signature
        ):
            turn_debug["error"] = (
                "repair_program.reasoning.abstract_repair_order must refine the accepted repair_outline without changing its phase structure"
            )
            _flush_turn()
            continue

        # Soft check: warn if critical parts are unobserved (ReAct feedback).
        _unobserved = _v3_unobserved_critical_parts(
            self, prepared_bridge_request, observation_history,
        )
        if _unobserved:
            warn_text = (
                f"Parts {', '.join(_unobserved)} have not been observed "
                f"in this session — coordinates may be inaccurate. "
                f"Consider emitting type='observe' first."
            )
            turn_debug["unobserved_warning"] = warn_text

        # Parse the repair program from structured response.
        try:
            program = repair_program_from_dict(parsed)
        except Exception as exc:
            parse_err = f"failed to parse repair_program: {exc}"
            self.logger.debug(
                "[RepairV3] Turn %d — %s", turn_idx, parse_err,
            )
            turn_debug["error"] = parse_err
            _flush_turn()
            continue

        turn_debug["program"] = repair_program_to_dict(program)
        self.logger.debug(
            "[RepairV3] Turn %d — received repair_program with %d functions, %d steps",
            turn_idx, len(program.function_defs), len(program.steps),
        )

        # ----- Duplicate check (turn cache) -----
        ctx_fp = compute_context_fingerprint(recovery_context)
        program_delta = (
            diff_programs(program, turn_cache) if turn_cache
            else ProgramDelta()
        )

        if turn_cache and can_reuse_validation_result(
            delta=program_delta,
            context_fingerprint=ctx_fp,
            cache=turn_cache,
        ):
            self.logger.debug(
                "[RepairV3] Turn %d — duplicate proposal, reusing cached result",
                turn_idx,
            )
            validated = validated_program_from_dict(
                turn_cache.last_validated_result,
            )
            turn_debug["cache_reuse"] = True
            validation_s = 0.0
        else:
            # ----- Full validation -----
            _validation_t0 = time.monotonic()
            validated = self._run_repair_validation(
                program=program,
                recovery_context=recovery_context,
                prepared_bridge_request=prepared_bridge_request,
                recovery_library=recovery_library,
            )
            validation_s = time.monotonic() - _validation_t0

        turn_debug["validation"] = validated_program_to_dict(validated)
        turn_debug["validation_s"] = round(validation_s, 4)

        # ----- Feedback summary (always, for next turn) -----
        feedback = summarize_validation_feedback(validated)

        # ----- Update turn cache -----
        # Build rendered catalog sections for caching.
        rendered_catalogs: dict[str, str] = {}
        for jid, catalog in (recovery_context.available_primitives or {}).items():
            if catalog:
                rendered_catalogs[jid] = (
                    f"### {jid}\n"
                    + json.dumps(
                        [
                            {"name": e.get("name"), "preconditions": e.get("preconditions"), "effects": e.get("effects")}
                            for e in catalog if isinstance(e, dict)
                        ],
                        indent=2, default=str,
                    )
                )

        turn_cache = TurnCache(
            turn_index=turn_idx,
            context_fingerprint=ctx_fp,
            program_fingerprint=compute_program_fingerprint(program),
            rendered_catalog_by_resource=rendered_catalogs,
            rendered_schema_section=_REPAIR_PROGRAM_FORMAT_REMINDER,
            last_feedback=feedback,
            last_rejected_proposal=repair_program_to_dict(program),
            last_validated_result=validated_program_to_dict(validated),
        )

        if validated.is_valid:
            self.logger.debug(
                "[RepairV3] Turn %d — VALID (risk=%s, approval=%s, continuation=%s)",
                turn_idx,
                validated.risk_level.value,
                validated.requires_operator_approval,
                validated.continuation_viable,
            )
            best_validated = validated
            repair_session["status"] = (
                "diagnose_first"
                if repair_mode == "diagnose_first"
                else "validated"
            )

            try:
                program_dict = repair_program_to_dict(program)
                plan_lines = [
                    f"  {line}"
                    for line in _render_outline_grouped_repair_program_lines(
                        program_dict,
                        accepted_outline,
                    )
                ]
                step_kinds = [
                    s.payload.get("function_name", s.kind.value)
                    if s.kind.value == "call_function"
                    else s.kind.value
                    for s in program.steps
                ]
                _react_logger.info(
                    "[Session] V3 Accepted plan (%d turns):\n%s\n  steps: %s",
                    turn_idx,
                    "\n".join(plan_lines),
                    " → ".join(step_kinds),
                )
            except Exception:
                pass

            _flush_turn()
            break
        else:
            # ----- Rejection: extract constraints -----
            new_constraints = [
                extract_constraint_from_rejection(r)
                for r in validated.rejection_reasons
            ]
            discovered_constraints.extend(new_constraints)
            rejection_history.append({
                "turn_index": turn_idx,
                "rejection_reasons": deepcopy(validated.rejection_reasons),
                "constraints_added": new_constraints,
                "feedback_summary": {
                    "blocked_steps": feedback.blocked_steps,
                    "unmet_obligations": feedback.unmet_obligations,
                    "violated_rules": feedback.violated_rules,
                    "open_witnesses": feedback.open_witnesses,
                    "suggested_adaptations": feedback.suggested_adaptations,
                },
            })
            last_rejected_proposal = repair_program_to_dict(program)

            reason_summary = "; ".join(
                str(r.get("message", "")).strip()
                for r in validated.rejection_reasons[:3]
            )
            self.logger.debug(
                "[RepairV3] Turn %d — REJECTED (%d reasons): %s",
                turn_idx,
                len(validated.rejection_reasons),
                reason_summary[:200],
            )
            _flush_turn()
            if repair_mode == "diagnose_first":
                repair_session["status"] = "diagnose_first"
                break
            continue

    # ----- Loop exhausted or validated -----
    session_elapsed = time.monotonic() - _session_t0

    if repair_session["status"] == "diagnose_first":
        bridge_debug["status"] = "diagnose_first"
        _react_logger.info(
            "[Session] V3 Diagnose-first stopped after %d turns (%.1fs)",
            repair_session["turn_index"], session_elapsed,
        )
    elif best_validated is None:
        repair_session["status"] = "exhausted"
        self.logger.debug(
            "[RepairV3] Session %s exhausted after %d turns (%.1fs) — "
            "escalating to operator",
            session_id, repair_session["turn_index"], session_elapsed,
        )
        _react_logger.info(
            "[Session] V3 Exhausted after %d turns (%.1fs) — escalating",
            repair_session["turn_index"], session_elapsed,
        )
        bridge_debug["status"] = "exhausted"
    else:
        bridge_debug["status"] = repair_session["status"]

    # Store final state.
    repair_session["discovered_constraints"] = discovered_constraints
    repair_session["observation_store"] = observation_store
    repair_session["observation_history"] = observation_history
    repair_session["rejection_history"] = rejection_history
    repair_session["last_rejected_proposal"] = last_rejected_proposal
    repair_session["accepted_outline"] = deepcopy(accepted_outline)
    repair_session["accepted_outline_context_fingerprint"] = (
        accepted_outline_context_fingerprint
    )
    if best_validated is not None:
        repair_session["best_validated_program"] = validated_program_to_dict(
            best_validated,
        )
    bridge_session_state = dict(prepared_bridge_request.get("bridge_session") or {})
    bridge_session_state.update({
        "turn_index": repair_session["turn_index"],
        "max_turns": repair_session["max_turns"],
        "max_observations": repair_session["max_observations"],
        "max_observe_batch": repair_session["max_observe_batch"],
        "repair_mode": repair_session.get("repair_mode", _DEFAULT_REPAIR_MODE),
        "auto_observe": repair_session.get("auto_observe", _DEFAULT_AUTO_OBSERVE),
        "v3_prompt_mode": repair_session.get("v3_prompt_mode", "repair_ready"),
        "observation_count": repair_session["observation_count"],
        "observation_history": deepcopy(observation_history),
        "last_rejected_proposal": deepcopy(last_rejected_proposal),
        "accepted_outline": deepcopy(accepted_outline),
        "accepted_outline_context_fingerprint": (
            accepted_outline_context_fingerprint
        ),
        "status": repair_session["status"],
    })
    prepared_bridge_request["bridge_session"] = bridge_session_state
    bridge_debug["session_elapsed_s"] = round(session_elapsed, 4)
    bridge_debug["total_turns"] = repair_session["turn_index"]
    bridge_debug["total_observations"] = repair_session["observation_count"]
    bridge_debug["total_constraints"] = len(discovered_constraints)
    prepared_bridge_request["bridge_debug"] = bridge_debug
    self._set_last_bridge_debug(bridge_debug)

    self.logger.debug(
        "[RepairV3] Session %s finished — status=%s, turns=%d, elapsed=%.1fs",
        session_id,
        repair_session["status"],
        repair_session["turn_index"],
        session_elapsed,
    )

    return {
        "status": repair_session["status"],
        "validated_program": (
            validated_program_to_dict(best_validated)
            if best_validated is not None
            else None
        ),
        "session": repair_session,
        "bridge_debug": bridge_debug,
    }


# Attach to the mixin class.
UniversalRepairSessionMixin.run_v3_repair_session = run_v3_repair_session  # type: ignore[attr-defined]
