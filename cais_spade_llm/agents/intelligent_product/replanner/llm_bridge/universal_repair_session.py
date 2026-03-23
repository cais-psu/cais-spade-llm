"""Universal Mutation-Synthesis repair session (v2 bridge).

Implements the constraint-guided repair loop with two LLM output types:
  - ``observe`` — top-level sensor request, results stored in observation_store
  - ``repair_program`` — the single universal repair artifact

Turn sequence::

    [observe]* → repair_program →
    [validate → reject → [observe]* → repair_program]* →
    approve → execute → verify

Each iteration rebuilds the prompt from scratch to prevent context poisoning.
Only discovered constraints (condensed from rejections) and the most recent
rejected proposal persist across iterations.

This is the primary LLM bridge session type (v2).
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.mutation_types import (
    RepairProgram,
    ValidatedRepairProgram,
    extract_constraint_from_rejection,
    repair_program_from_dict,
    repair_program_to_dict,
    validated_program_to_dict,
)
from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.recovery_context_builder import (
    build_recovery_context,
    recovery_context_to_prompt_dict,
)
from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.repair_program_validator import (
    validate_repair_program,
)
from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.recovery_library import (
    RecoveryLibrary,
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


def _write_turn_debug(
    session_id: str,
    turn_idx: int,
    *,
    prompt: str = "",
    raw_response: Any = None,
    error: str = "",
    observation: dict[str, Any] | None = None,
    program: dict[str, Any] | None = None,
    validation: dict[str, Any] | None = None,
) -> None:
    """Append one turn's data to the incremental debug text file."""
    try:
        debug_dir = _get_debug_dir()
        txt_path = debug_dir / f"v2_session_{session_id}.txt"
        sep = "=" * 80

        lines: list[str] = ["", sep, f"TURN {turn_idx}", sep]

        if prompt:
            lines.append("")
            lines.append(f"--- PROMPT ({len(prompt)} chars) ---")
            lines.append(prompt)

        if raw_response is not None:
            lines.append("")
            lines.append("--- LLM RESPONSE ---")
            if isinstance(raw_response, dict):
                lines.append(json.dumps(raw_response, indent=2, default=str, ensure_ascii=False))
            else:
                lines.append(str(raw_response))

        if error:
            lines.append("")
            lines.append(f"--- ERROR ---")
            lines.append(error)

        if observation:
            lines.append("")
            lines.append("--- OBSERVATION ---")
            lines.append(json.dumps(observation, indent=2, default=str, ensure_ascii=False))

        if program:
            lines.append("")
            lines.append("--- REPAIR PROGRAM ---")
            lines.append(json.dumps(program, indent=2, default=str, ensure_ascii=False))

        if validation:
            lines.append("")
            is_valid = validation.get("is_valid", False)
            lines.append(f"--- VALIDATION ({'ACCEPTED' if is_valid else 'REJECTED'}) ---")
            lines.append(json.dumps(validation, indent=2, default=str, ensure_ascii=False))

        with open(txt_path, "a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
    except Exception:
        pass  # Never let debug I/O break the session.


_react_logger = logging.getLogger(__name__ + ".react")


def _log_react_turn(turn_idx: int, turn_debug: dict[str, Any]) -> None:
    """Print a concise ReAct-style summary line for one turn."""
    try:
        response_type = turn_debug.get("response_type", "")
        error = turn_debug.get("error", "")

        # Extract thought (rationale from repair_program or parsed observe).
        thought = ""
        raw = turn_debug.get("raw_response")
        if isinstance(raw, dict):
            content = raw.get("content") or raw
            if isinstance(content, str):
                try:
                    content = json.loads(content)
                except Exception:
                    content = {}
            if isinstance(content, dict):
                thought = str(content.get("rationale", "")).strip()

        if thought:
            _react_logger.info("[Turn %d] Thought: %s", turn_idx, thought[:200])

        if error:
            action_label = response_type or "error"
            _react_logger.info("[Turn %d] %s: %s", turn_idx, action_label, error[:200])
            return

        if response_type == "observe":
            obs = turn_debug.get("observation") or {}
            prim = ""
            if isinstance(raw, dict):
                content = raw.get("content") or raw
                if isinstance(content, str):
                    try:
                        content = json.loads(content)
                    except Exception:
                        content = {}
                if isinstance(content, dict):
                    prim = str(content.get("primitive", "")).strip()
                    resource = str(content.get("resource_jid", "")).strip()
                    if resource:
                        prim = f"{prim} on {resource}"
            _react_logger.info("[Turn %d] Action: observe %s", turn_idx, prim)
            obs_data = obs.get("observation") if isinstance(obs, dict) else obs
            if obs_data:
                summary = json.dumps(obs_data, default=str, ensure_ascii=False)
                if len(summary) > 200:
                    summary = summary[:200] + "..."
                _react_logger.info("[Turn %d] Result: %s", turn_idx, summary)

        elif response_type == "repair_program":
            program = turn_debug.get("program") or {}
            fn_count = len(program.get("function_defs", []))
            fn_names = [
                fd.get("name", "?")
                for fd in program.get("function_defs", [])[:5]
            ]
            _react_logger.info(
                "[Turn %d] Action: repair_program (%d fn: %s)",
                turn_idx, fn_count, ", ".join(fn_names),
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


# ---------------------------------------------------------------------------
# Session state dataclass (plain dict for simplicity + JSON serialization)
# ---------------------------------------------------------------------------

def _new_repair_session(
    *,
    session_id: str = "",
    max_turns: int = _DEFAULT_MAX_TURNS,
    max_observations: int = _DEFAULT_MAX_OBSERVATIONS,
) -> dict[str, Any]:
    """Create a fresh v2 repair session state dict."""
    return {
        "session_id": session_id or f"repair_{uuid4().hex[:8]}",
        "version": 2,
        "turn_index": 0,
        "max_turns": max_turns,
        "max_observations": max_observations,
        "observation_count": 0,
        "observation_store": {},
        "observation_history": [],
        "discovered_constraints": [],
        "last_rejected_proposal": None,
        "best_validated_program": None,
        "rejection_history": [],
        "status": "running",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# LLM response parsing
# ---------------------------------------------------------------------------

def _parse_llm_response(raw: Any) -> tuple[dict[str, Any] | None, str]:
    """Parse raw LLM output into a typed response dict.

    Returns ``(parsed_dict, error_string)``.  If *error_string* is non-empty
    the response could not be parsed.
    """
    if isinstance(raw, dict):
        payload = raw
    elif isinstance(raw, str):
        text = raw.strip()
        # Strip markdown fences if present.
        if text.startswith("```"):
            lines = text.split("\n")
            lines = lines[1:]  # drop opening fence
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            return None, f"LLM response is not valid JSON: {exc}"
    else:
        return None, f"unexpected LLM response type: {type(raw).__name__}"

    if not isinstance(payload, dict):
        return None, "LLM response JSON root is not an object"

    response_type = str(payload.get("type", "")).strip().lower()
    if response_type not in ("observe", "repair_program"):
        return None, (
            f"invalid response type '{response_type}'; "
            f"expected 'observe' or 'repair_program'"
        )
    return payload, ""


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
        """Run the v2 constraint-guided universal repair loop.

        Parameters
        ----------
        prepared_bridge_request:
            Standard v1 prepared bridge request dict.  The session builds
            a fresh :class:`RecoveryContext` from this on every iteration.
        recovery_library:
            Optional persistent recovery library for candidate retrieval
            and post-execution registration.

        Returns
        -------
        dict:
            Result dict with keys: ``status``, ``validated_program``,
            ``session``, ``bridge_debug``.
        """
        if not isinstance(prepared_bridge_request, dict):
            raise ValueError("prepared bridge request is missing")

        repair_session = _new_repair_session()
        session_id = repair_session["session_id"]

        bridge_debug: dict[str, Any] = {
            "session_id": session_id,
            "session_type": "universal_repair_v2",
            "generation_started_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "running",
            "turns": [],
        }
        prepared_bridge_request["bridge_debug"] = bridge_debug
        self._set_last_bridge_debug(bridge_debug)

        max_turns = int(repair_session["max_turns"])
        max_observations = int(repair_session["max_observations"])
        _session_t0 = time.monotonic()

        _react_logger.info(
            "[RepairV2] Session %s — max_turns=%d, max_observations=%d",
            session_id, max_turns, max_observations,
        )

        # Write header to incremental debug file.
        try:
            debug_path = _get_debug_dir() / f"v2_session_{session_id}.txt"
            debug_path.write_text(
                f"V2 Repair Session — {session_id}\n"
                f"Started: {datetime.now(timezone.utc).isoformat()}\n"
                f"Max turns: {max_turns}, Max observations: {max_observations}\n",
                encoding="utf-8",
            )
            _react_logger.info("[RepairV2] Debug: %s", debug_path)
        except Exception:
            pass

        # Accumulators that persist across iterations.
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

        while repair_session["turn_index"] < max_turns:
            repair_session["turn_index"] += 1
            turn_idx = repair_session["turn_index"]
            _turn_t0 = time.monotonic()

            # ----- Build fresh RecoveryContext -----
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

            # ----- Build fresh prompt -----
            prompt = self._build_repair_program_prompt(
                recovery_context=recovery_context,
                recovery_library=recovery_library,
                discovered_constraints=discovered_constraints,
                last_rejected_proposal=last_rejected_proposal,
                observation_history=observation_history,
                turn_index=turn_idx,
                max_turns=max_turns,
            )

            turn_debug: dict[str, Any] = {
                "turn_index": turn_idx,
                "prompt": prompt,
                "prompt_length": len(prompt),
                "discovered_constraints_count": len(discovered_constraints),
            }

            def _flush_turn() -> None:
                """Append turn_debug to bridge_debug, write debug file, and log ReAct line."""
                bridge_debug["turns"].append(turn_debug)
                self._set_last_bridge_debug(bridge_debug)
                _write_turn_debug(
                    session_id,
                    turn_idx,
                    prompt=turn_debug.get("prompt", ""),
                    raw_response=turn_debug.get("raw_response"),
                    error=turn_debug.get("error", ""),
                    observation=turn_debug.get("observation"),
                    program=turn_debug.get("program"),
                    validation=turn_debug.get("validation"),
                )
                _log_react_turn(turn_idx, turn_debug)

            # ----- Call LLM -----
            self.logger.debug(
                "[RepairV2] Turn %d/%d — sending prompt to LLM (%d chars)...",
                turn_idx, max_turns, len(prompt),
            )
            llm_latency_s = 0.0
            try:
                _llm_t0 = time.monotonic()
                raw_response = await self.product_agent.ask_llm(
                    prompt=prompt,
                    with_functions=False,
                    temperature=0.0,
                )
                llm_latency_s = time.monotonic() - _llm_t0
            except Exception as exc:
                bridge_debug["status"] = "exception"
                bridge_debug["exception"] = repr(exc)
                turn_debug["exception"] = repr(exc)
                _flush_turn()
                raise

            turn_debug["llm_latency_s"] = round(llm_latency_s, 4)
            turn_debug["session_elapsed_s"] = round(
                time.monotonic() - _session_t0, 4
            )
            turn_debug["raw_response"] = deepcopy(raw_response) if isinstance(raw_response, dict) else raw_response

            # ----- Parse response -----
            parsed, parse_error = _parse_llm_response(raw_response)
            if parse_error:
                self.logger.debug(
                    "[RepairV2] Turn %d — parse error: %s", turn_idx, parse_error,
                )
                turn_debug["error"] = parse_error
                _flush_turn()
                continue

            response_type = str(parsed.get("type", "")).strip().lower()
            turn_debug["response_type"] = response_type

            # ----- Handle observe -----
            if response_type == "observe":
                if repair_session["observation_count"] >= max_observations:
                    obs_error = (
                        f"observation budget exhausted ({max_observations}); "
                        f"emit a repair_program instead"
                    )
                    self.logger.debug(
                        "[RepairV2] Turn %d — %s", turn_idx, obs_error,
                    )
                    # Add as a discovered constraint so the LLM knows.
                    discovered_constraints.append({
                        "layer": "session",
                        "check": "observation_budget",
                        "constraint": obs_error,
                    })
                    turn_debug["error"] = obs_error
                    _flush_turn()
                    continue

                obs_row, obs_error = await self._execute_repair_observation(
                    prepared_bridge_request,
                    action=parsed,
                )
                repair_session["observation_count"] += 1
                _obs_prim = str(parsed.get("primitive", "")).strip()

                if obs_error:
                    self.logger.debug(
                        "[RepairV2] Turn %d — observe %s FAILED: %s",
                        turn_idx, _obs_prim, obs_error,
                    )
                    turn_debug["error"] = obs_error
                else:
                    alias = str(parsed.get("store_as", "")).strip() or f"obs_{turn_idx}"
                    obs_data = deepcopy(
                        (obs_row or {}).get("observation") or {}
                    )
                    observation_store[alias] = obs_data
                    observation_history.append({
                        "turn_index": turn_idx,
                        "primitive": _obs_prim,
                        "store_as": alias,
                        "observation": obs_data,
                    })
                    self.logger.debug(
                        "[RepairV2] Turn %d — observe %s succeeded → %s",
                        turn_idx, _obs_prim, alias,
                    )
                    turn_debug["observation"] = deepcopy(obs_row)

                _flush_turn()
                continue

            # ----- Handle repair_program -----
            assert response_type == "repair_program"

            # Parse the repair program from LLM JSON.
            try:
                program = repair_program_from_dict(parsed)
            except Exception as exc:
                parse_err = f"failed to parse repair_program: {exc}"
                self.logger.debug(
                    "[RepairV2] Turn %d — %s", turn_idx, parse_err,
                )
                turn_debug["error"] = parse_err
                _flush_turn()
                continue

            turn_debug["program"] = repair_program_to_dict(program)
            self.logger.debug(
                "[RepairV2] Turn %d — received repair_program with %d functions, %d steps",
                turn_idx, len(program.function_defs), len(program.steps),
            )

            # ----- Validate -----
            validated = self._run_repair_validation(
                program=program,
                recovery_context=recovery_context,
                prepared_bridge_request=prepared_bridge_request,
                recovery_library=recovery_library,
            )

            turn_debug["validation"] = validated_program_to_dict(validated)

            if validated.is_valid:
                self.logger.debug(
                    "[RepairV2] Turn %d — VALID (risk=%s, approval=%s, continuation=%s)",
                    turn_idx,
                    validated.risk_level.value,
                    validated.requires_operator_approval,
                    validated.continuation_viable,
                )
                best_validated = validated
                repair_session["status"] = "validated"

                # Log high-level accepted plan summary via ReAct logger.
                try:
                    plan_lines = []
                    for fd in program.function_defs:
                        prims = [
                            p.get("primitive", "?")
                            for p in (fd.primitive_program or [])
                        ]
                        plan_lines.append(
                            f"  fn {fd.name}: {' → '.join(prims)}"
                        )
                    step_kinds = [
                        s.payload.get("function_name", s.kind.value)
                        if s.kind.value == "call_function"
                        else s.kind.value
                        for s in program.steps
                    ]
                    _react_logger.info(
                        "[Session] Accepted plan (%d turns):\n%s\n  steps: %s",
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
                })
                last_rejected_proposal = repair_program_to_dict(program)

                reason_summary = "; ".join(
                    str(r.get("message", "")).strip()
                    for r in validated.rejection_reasons[:3]
                )
                self.logger.debug(
                    "[RepairV2] Turn %d — REJECTED (%d reasons): %s",
                    turn_idx,
                    len(validated.rejection_reasons),
                    reason_summary[:200],
                )
                _flush_turn()
                continue

        # ----- Loop exhausted or validated -----
        session_elapsed = time.monotonic() - _session_t0

        if best_validated is None:
            repair_session["status"] = "exhausted"
            self.logger.debug(
                "[RepairV2] Session %s exhausted after %d turns (%.1fs) — "
                "escalating to operator",
                session_id, repair_session["turn_index"], session_elapsed,
            )
            _react_logger.info(
                "[Session] Exhausted after %d turns (%.1fs) — escalating",
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
        if best_validated is not None:
            repair_session["best_validated_program"] = validated_program_to_dict(
                best_validated
            )
        bridge_debug["session_elapsed_s"] = round(session_elapsed, 4)
        bridge_debug["total_turns"] = repair_session["turn_index"]
        bridge_debug["total_observations"] = repair_session["observation_count"]
        bridge_debug["total_constraints"] = len(discovered_constraints)
        prepared_bridge_request["bridge_debug"] = bridge_debug
        self._set_last_bridge_debug(bridge_debug)

        self.logger.debug(
            "[RepairV2] Session %s finished — status=%s, turns=%d, elapsed=%.1fs",
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

        return validate_repair_program(
            program,
            primitive_catalogs=primitive_catalogs,
            resource_snapshots=resource_snapshots,
            current_nodes=current_nodes,
            observation_store=dict(recovery_context.observation_store),
            capability_flags_map=capability_flags_map,
            safety_validator=safety_validator,
            safety_rules=safety_rules,
            runtime_monitor_state=runtime_monitor_state,
            compile_fsa_fn=compile_fsa_fn,
            recovery_library=recovery_library,
            active_obligations=list(recovery_context.active_obligations),
            part_states=dict(recovery_context.part_states),
        )

    # ------------------------------------------------------------------
    # Prompt builder
    # ------------------------------------------------------------------

    def _build_repair_program_prompt(
        self,
        *,
        recovery_context: Any,
        recovery_library: RecoveryLibrary | None = None,
        discovered_constraints: list[dict[str, Any]] | None = None,
        last_rejected_proposal: dict[str, Any] | None = None,
        observation_history: list[dict[str, Any]] | None = None,
        turn_index: int = 1,
        max_turns: int = _DEFAULT_MAX_TURNS,
    ) -> str:
        """Build a fresh prompt for the v2 repair session.

        Each iteration rebuilds from scratch with:
        1. Full RecoveryContext (always fresh)
        2. Discovered constraints from prior rejections
        3. Only the most recent rejected proposal
        4. Library candidates
        5. Observation history
        """
        sections: list[str] = []

        # ---- System instruction ----
        sections.append(
            "You are a recovery planner for a multi-robot manufacturing system.\n"
            "Reason from state gaps and obligations. Produce a repair program "
            "that closes all gaps.\n"
            "If your proposal is rejected, use the rejection reason to revise.\n"
            "\n"
            "You may output one of two response types:\n"
            '  1. {"type": "observe", "resource_jid": str, "primitive": str, '
            '"params": dict, "store_as": str}\n'
            '  2. {"type": "repair_program", "function_defs": [...], '
            '"steps": [...], "success_conditions": [...], "rationale": str}\n'
            "\n"
            "Output ONLY valid JSON. No markdown, no explanation outside JSON."
        )

        # ---- Turn budget ----
        remaining = max(0, max_turns - turn_index)
        sections.append(
            f"Turn {turn_index}/{max_turns} ({remaining} turns remaining)."
        )

        # ---- Recovery context ----
        ctx_dict = recovery_context_to_prompt_dict(recovery_context)
        sections.append(
            "## Current System State\n"
            + json.dumps(ctx_dict, indent=2, default=str)
        )

        # ---- Available primitives (with params, preconditions, effects) ----
        prim_sections: list[str] = []
        for jid, catalog in (recovery_context.available_primitives or {}).items():
            if not catalog:
                continue
            prim_entries: list[dict[str, Any]] = []
            for entry in catalog:
                if not isinstance(entry, dict):
                    continue
                name = str(entry.get("name", "")).strip()
                if not name:
                    continue
                compact: dict[str, Any] = {"name": name}
                # Required and optional params.
                required = entry.get("required_params") or []
                params_info = entry.get("params") or {}
                if required:
                    compact["required_params"] = required
                if params_info:
                    compact["params"] = {
                        k: v.get("type", "any") if isinstance(v, dict) else str(v)
                        for k, v in params_info.items()
                    }
                # Preconditions and effects (critical for the LLM).
                preconds = entry.get("preconditions")
                if preconds:
                    compact["preconditions"] = preconds
                effects = entry.get("effects")
                if effects:
                    compact["effects"] = effects
                # Whether it produces observation output.
                semantics = entry.get("bridge_semantics") or {}
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
        if prim_sections:
            sections.append(
                "## Available Primitives per Resource\n"
                "Each primitive lists its required_params, preconditions, and effects.\n"
                "Your function primitive_program MUST use only these primitives with correct params.\n"
                "Do NOT use store_as on primitives that do not have produces_observation=true.\n\n"
                + "\n\n".join(prim_sections)
            )

        # ---- Library candidates ----
        if recovery_library is not None:
            # Get candidates for each resource type in context.
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
                sections.append(
                    "## Library Candidates (previously validated functions)\n"
                    "You may reuse or adapt these. They will still be fully "
                    "validated.\n"
                    + json.dumps(all_candidates, indent=2, default=str)
                )

        # ---- Observation history ----
        if observation_history:
            compact_obs = [
                {
                    "primitive": h.get("primitive"),
                    "store_as": h.get("store_as"),
                    "observation": h.get("observation"),
                }
                for h in observation_history[-5:]
            ]
            sections.append(
                "## Observation Results\n"
                + json.dumps(compact_obs, indent=2, default=str)
            )

        # ---- Discovered constraints ----
        if discovered_constraints:
            constraint_lines = []
            for i, c in enumerate(discovered_constraints, 1):
                text = str(c.get("constraint", "")).strip()
                if text:
                    constraint_lines.append(f"  {i}. {text}")
            if constraint_lines:
                sections.append(
                    "## Discovered Constraints (from prior rejections)\n"
                    "Your repair program MUST satisfy all of these:\n"
                    + "\n".join(constraint_lines)
                )

        # ---- Last rejected proposal ----
        if last_rejected_proposal is not None:
            sections.append(
                "## Your Most Recent Rejected Proposal\n"
                "This was rejected. Revise your approach using the "
                "discovered constraints above.\n"
                + json.dumps(last_rejected_proposal, indent=2, default=str)
            )

        # ---- Repair program schema ----
        sections.append(_REPAIR_PROGRAM_SCHEMA_SECTION)

        return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# Schema section (constant)
# ---------------------------------------------------------------------------

_REPAIR_PROGRAM_SCHEMA_SECTION = """\
## RepairProgram JSON Schema

```json
{
  "type": "repair_program",
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
  "steps": [
    {"kind": "call_function", "payload": {"function_name": "...", "resource_jid": "...", "args": {}}}
    | {"kind": "task_mutation", "payload": {"mutation_type": "insert|delete|reassign|replace_suffix", "target_task_ids": [...], "payload": {...}}}
    | {"kind": "wait", "payload": {"until": {"entity_kind": "...", "entity": "...", "field": "...", "expected": "..."}}}
    | {"kind": "resume_suffix", "payload": {}}
  ],
  "success_conditions": [
    {"entity_kind": "resource|part", "entity": "...", "field": "...", "expected": "..."}
  ],
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
