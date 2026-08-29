"""Orchestrate ontology-backed Spec2Primitives PA context grounding."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from cais_spade_llm.spec2primitives.agents.pa import context_serving
from cais_spade_llm.spec2primitives.agents.pa.context_grounding import (
    PAOntologyConfig,
    ProductContextGroundingRuntime,
    producer_for_evidence_type,
    validated_grounding_producer_descriptors,
)
from cais_spade_llm.spec2primitives.agents.pa.context_interaction import (
    ProductAgentContextRuntime,
)
from cais_spade_llm.spec2primitives.agents.pa.grounding_contracts import (
    GroundingContractError,
    GroundingProducerDescriptor,
    build_product_context_view,
    load_latest_grounding_session,
    persist_pa_context_grounding_completion_v2,
    persist_product_context_view,
)
from cais_spade_llm.spec2primitives.agents.pa.product_context import (
    ABoxSnapshot,
    load_interaction_abox,
    validate_and_merge_triple_delta,
)
from cais_spade_llm.spec2primitives.ontology import TBoxSnapshot
from cais_spade_llm.spec2primitives.tools.exact_ref_resolver import (
    approved_context_ref_evidence_types,
)

logger = logging.getLogger(__name__)

_SETTINGS_PATH = Path("interaction_record/pa_context_settings.json")
_TURN_KEYS = {"turn", "product_requirement", "PA_input", "PA_output", "failure"}
_RETRIEVAL_KEYS = {
    "retrieval",
    "product_requirement",
    "needed_context",
    "context_request",
    "served_context",
    "retrieval_error",
    "failure",
}
_ASSESSMENT_KEYS = {
    "unresolved_semantic_need",
    "needed_context",
    "context understanding complete",
}
_SESSION_ASSESSMENT_KEYS = _ASSESSMENT_KEYS | {"grounding_status"}
_CLARIFICATION_KEYS = {
    "schema_version",
    "record_type",
    "product_requirement",
    "question_turn",
    "semantic_need",
    "question",
    "action",
    "reply",
    "recorded_at_ns",
    "fingerprint",
}


class _AuditedProductAgentRuntime:
    """Record exact structured PA calls made inside the assessment boundary."""

    def __init__(self, product_agent: ProductAgentContextRuntime) -> None:
        self._product_agent = product_agent
        self.calls: list[dict[str, object]] = []

    async def ask_llm_structured(
        self,
        prompt: str,
        *,
        response_format: dict[str, Any],
    ) -> dict[str, Any]:
        """Delegate one call while preserving its exact input and output."""
        record: dict[str, object] = {
            "prompt": prompt,
            "response_format": response_format,
            "output": None,
            "failure": None,
        }
        self.calls.append(record)
        try:
            output = await self._product_agent.ask_llm_structured(
                prompt,
                response_format=response_format,
            )
        except Exception as exc:
            record["failure"] = {
                "type": type(exc).__name__,
                "message": str(exc),
            }
            raise RuntimeError("ProductAgent structured assessment call failed.") from exc
        record["output"] = output
        return output


async def submit_pa_clarification_reply(
    product_agent: ProductAgentContextRuntime,
    interaction_root: Path,
    user_reply: str,
    *,
    ontology_config: PAOntologyConfig | None = None,
    grounding_runtime: ProductContextGroundingRuntime | None = None,
) -> dict[str, object]:
    """Persist one exact user-intent reply and resume the same interaction."""
    if not isinstance(user_reply, str) or not user_reply.strip():
        return _failure(
            "invalid_clarification",
            "Phase 3.4 user_reply must be a non-empty string.",
        )
    interaction_root = Path(interaction_root)
    settings, settings_error = _read_pa_context_settings(interaction_root)
    if settings_error is not None or settings is None:
        return _failure(
            "invalid_interaction",
            settings_error or "Phase 3.3 settings are unavailable.",
        )
    configuration_failure = _configuration_failure(
        ontology_config=ontology_config,
        grounding_runtime=grounding_runtime,
        max_pa_turns=settings["max_pa_turns"],
        live_observation_timeout_sec=settings["live_observation_timeout_sec"],
    )
    if configuration_failure is not None:
        return configuration_failure
    pending, pending_error = _pending_clarification(interaction_root)
    if pending_error is not None or pending is None:
        return _failure(
            "invalid_clarification",
            pending_error or "No pending user_intent clarification exists.",
        )
    question_turn = pending["question_turn"]
    if not isinstance(question_turn, int) or question_turn >= settings["max_pa_turns"]:
        return _failure(
            "pa_turn_limit_reached",
            "No ProductAgent turn remains for a Phase 3.4 clarification reply.",
        )
    record = _clarification_record(
        product_requirement=str(pending["product_requirement"]),
        question_turn=question_turn,
        semantic_need=pending["semantic_need"],
        question=str(pending["question"]),
        action="answered",
        reply=user_reply,
    )
    path = (
        interaction_root
        / "interaction_record"
        / f"clarification_{question_turn:04d}.json"
    )
    existing, existing_error = _existing_clarification_record(path)
    if existing_error is not None:
        return _failure("invalid_clarification", existing_error)
    if existing is not None:
        if existing.get("action") != "answered" or existing.get("reply") != user_reply:
            return _failure(
                "clarification_exists",
                "The pending clarification already has a different terminal action.",
            )
    else:
        try:
            _write_json_exclusive(path, record)
        except (FileExistsError, OSError, TypeError, ValueError) as exc:
            return _failure(
                "invalid_interaction",
                f"Phase 3.4 reply write failed: {type(exc).__name__}: {exc}",
            )
    return await continue_pa_context_interaction(
        product_agent,
        interaction_root,
        ontology_config=ontology_config,
        grounding_runtime=grounding_runtime,
        max_pa_turns=settings["max_pa_turns"],
        live_observation_timeout_sec=settings["live_observation_timeout_sec"],
    )


def cancel_pa_context_interaction(interaction_root: Path) -> dict[str, object]:
    """Persist an explicit cancellation for one pending clarification."""
    interaction_root = Path(interaction_root)
    pending, pending_error = _pending_clarification(interaction_root)
    if pending_error is not None or pending is None:
        return _failure(
            "invalid_clarification",
            pending_error or "No pending user_intent clarification exists.",
        )
    question_turn = pending["question_turn"]
    if not isinstance(question_turn, int):
        return _failure("invalid_clarification", "Clarification turn is invalid.")
    record = _clarification_record(
        product_requirement=str(pending["product_requirement"]),
        question_turn=question_turn,
        semantic_need=pending["semantic_need"],
        question=str(pending["question"]),
        action="cancelled",
        reply=None,
    )
    path = (
        interaction_root
        / "interaction_record"
        / f"clarification_{question_turn:04d}.json"
    )
    existing, existing_error = _existing_clarification_record(path)
    if existing_error is not None:
        return _failure("invalid_clarification", existing_error)
    if existing is not None:
        if existing.get("action") != "cancelled":
            return _failure(
                "clarification_exists",
                "The pending clarification already has a different terminal action.",
            )
    else:
        try:
            _write_json_exclusive(path, record)
        except (FileExistsError, OSError, TypeError, ValueError) as exc:
            return _failure(
                "invalid_interaction",
                f"Phase 3.4 cancellation write failed: {type(exc).__name__}: {exc}",
            )
    return {"status": "cancelled", "context understanding complete": False}


async def continue_pa_context_interaction(  # noqa: C901, PLR0912, PLR0915
    product_agent: ProductAgentContextRuntime,
    interaction_root: Path,
    *,
    ontology_config: PAOntologyConfig | None = None,
    grounding_runtime: ProductContextGroundingRuntime | None = None,
    max_pa_turns: int = 12,
    live_observation_timeout_sec: float = 5.0,
) -> dict[str, object]:
    """Interpret, merge, and reassess served context until a safe stop.

    Args:
        product_agent: Shared ProductAgent passed only to the injected Phase 4.3
            assessment boundary.
        interaction_root: Root containing successful Phase 3.1 and Phase 3.2
            records plus the initialized interaction ABox.
        ontology_config: Exact TBox path and namespace used by Phase 3.1.
        grounding_runtime: Controlled Phase 4 interpretation and assessment
            boundary.
        max_pa_turns: Maximum PA decisions including the Phase 3.1 bootstrap.
        live_observation_timeout_sec: Maximum duration of each fresh capture.

    Returns:
        A terminal persisted assessment or a structured fail-closed result.
    """
    configuration_failure = _configuration_failure(
        ontology_config=ontology_config,
        grounding_runtime=grounding_runtime,
        max_pa_turns=max_pa_turns,
        live_observation_timeout_sec=live_observation_timeout_sec,
    )
    if configuration_failure is not None:
        return configuration_failure
    if ontology_config is None or grounding_runtime is None:
        raise AssertionError("Validated ontology and grounding dependencies are required.")

    interaction_root = Path(interaction_root)
    initial, preparation_error = _read_initial_interaction(interaction_root)
    if preparation_error is not None:
        return _failure("invalid_interaction", preparation_error)
    if initial is None:
        raise AssertionError("Validated Phase 3.1 and Phase 3.2 records are required.")

    resuming_clarification = any(
        (interaction_root / "interaction_record").glob("clarification_*.json")
    )
    existing_record = (
        None
        if resuming_clarification
        else _existing_phase_3_3_record(interaction_root)
    )
    if existing_record is not None:
        return _failure(
            "interaction_exists",
            f"Phase 3.3 record already exists: {existing_record}.",
        )

    try:
        tbox = ontology_config.load_tbox()
        producer_descriptors = validated_grounding_producer_descriptors(
            grounding_runtime
        )
        abox = load_interaction_abox(interaction_root, tbox)
        context_ref_evidence_types = approved_context_ref_evidence_types()
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        logger.exception("Phase 3.3 ontology reload failed.")
        return _failure(
            "ontology_context_invalid",
            f"Phase 4.0 context could not be validated: {type(exc).__name__}: {exc}",
        )
    product_requirement = initial["product_requirement"]
    if abox.product_requirement != product_requirement:
        return _failure(
            "ontology_context_invalid",
            "The interaction ABox product_requirement does not match Phase 3.1.",
        )
    if (
        not resuming_clarification
        and abox.delta_count != 0
        and not _is_initial_provisional_grounding(interaction_root, abox)
    ):
        return _failure(
            "invalid_interaction",
            "Phase 3.3 may start only from the initialized or exact provisional ABox.",
        )

    settings = {
        "max_pa_turns": max_pa_turns,
        "live_observation_timeout_sec": float(live_observation_timeout_sec),
    }
    if resuming_clarification:
        persisted_settings, settings_error = _read_pa_context_settings(interaction_root)
        if settings_error is not None:
            return _failure("invalid_interaction", settings_error)
        if persisted_settings != settings:
            return _failure(
                "invalid_interaction",
                "Phase 3.4 must preserve the original PA context settings.",
            )
        resume_state, resume_error = _clarification_resume_state(
            interaction_root,
            product_requirement=product_requirement,
            max_pa_turns=max_pa_turns,
        )
        if resume_error is not None:
            return _failure("invalid_clarification", resume_error)
        if resume_state is None:
            raise AssertionError("Validated clarification resume state is required.")
        served_context: dict[str, object] | None = None
        attempted_evidence = list(resume_state["attempted_evidence"])
        served_contexts = list(resume_state["served_contexts"])
        served_static_refs = _served_static_refs(served_contexts)
        live_request_history = set(resume_state["live_request_history"])
        next_observation_number = _next_observation_number(served_contexts)
        clarification_history = tuple(resume_state["clarification_history"])
        start_operation_number = int(resume_state["operation_number"])
    else:
        try:
            _write_json_exclusive(interaction_root / _SETTINGS_PATH, settings)
        except FileExistsError:
            return _failure(
                "interaction_exists",
                "Phase 3.3 pa_context_settings.json already exists.",
            )
        except (OSError, TypeError, ValueError) as exc:
            return _failure(
                "invalid_interaction",
                f"Phase 3.3 settings write failed: {type(exc).__name__}: {exc}",
            )
        served_context = initial["served_context"]
        attempted_evidence = []
        served_static_refs = _served_static_refs([served_context])
        live_request_history = set()
        next_observation_number = _next_observation_number([served_context])
        clarification_history = ()
        start_operation_number = 1

    for operation_number in range(start_operation_number, max_pa_turns):
        if served_context is not None:
            interpretation = await _interpret_and_merge(
                grounding_runtime,
                interaction_root=interaction_root,
                tbox=tbox,
                abox=abox,
                producer_descriptors=producer_descriptors,
                served_context=served_context,
                attempted_evidence=tuple(attempted_evidence),
                operation_number=operation_number,
            )
            if "failure" in interpretation:
                return interpretation
            updated_abox = interpretation.get("abox")
            evidence_identifier = interpretation.get("evidence_identifier")
            if not isinstance(updated_abox, ABoxSnapshot) or not isinstance(
                evidence_identifier,
                str,
            ):
                raise AssertionError("Validated interpretation result is incomplete.")
            abox = updated_abox
            if evidence_identifier not in attempted_evidence:
                attempted_evidence.append(evidence_identifier)
            served_context = None

        turn_number = operation_number + 1
        try:
            product_context_view = build_product_context_view(
                interaction_root,
                abox,
                attempted_evidence=attempted_evidence,
                assessed_at_ns=time.time_ns(),
            )
            product_context_view_path = persist_product_context_view(
                interaction_root,
                product_context_view,
            )
        except (OSError, TypeError, ValueError) as exc:
            logger.exception("Phase 4.3 ProductContextView construction failed.")
            return _failure(
                "product_context_invalid",
                f"ProductContextView could not be validated: {type(exc).__name__}: {exc}",
            )
        assessment_input = {
            "product_context": product_context_view.to_record(),
            "product_context_ref": str(
                product_context_view_path.relative_to(interaction_root)
            ),
            "attempted_evidence": list(attempted_evidence),
            "clarification_history": [dict(item) for item in clarification_history],
            "turn": turn_number,
            "max_pa_turns": max_pa_turns,
        }
        decision_path = (
            interaction_root / "interaction_record" / f"decision_{operation_number:04d}.json"
        )
        turn_path = interaction_root / "interaction_record" / f"turn_{turn_number:04d}.json"
        if decision_path.exists() or turn_path.exists():
            return _failure(
                "interaction_exists",
                f"Phase 3.3 decision or turn {operation_number:04d} already exists.",
            )

        assessment_product_agent = _AuditedProductAgentRuntime(product_agent)
        try:
            assessment_value = await grounding_runtime.assess_product_context(
                assessment_product_agent,
                interaction_root=interaction_root,
                tbox=tbox,
                abox=abox,
                abox_view=assessment_input["product_context"],
                attempted_evidence=tuple(attempted_evidence),
                clarification_history=clarification_history,
                turn_number=turn_number,
                max_pa_turns=max_pa_turns,
            )
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            logger.exception("Phase 4.3 controlled assessment failed.")
            assessment_input["ProductAgent_calls"] = assessment_product_agent.calls
            failure = _failure(
                "assessment_failed",
                f"Controlled product-context assessment failed: {type(exc).__name__}: {exc}",
            )
            decision_failure = _write_decision_record(
                decision_path,
                operation_number=operation_number,
                turn_number=turn_number,
                product_requirement=product_requirement,
                assessment_input=assessment_input,
                assessment_output=None,
                failure=failure,
            )
            if decision_failure is not None:
                return decision_failure
            turn_failure = _write_turn_record(
                turn_path,
                turn_number=turn_number,
                product_requirement=product_requirement,
                operation_number=operation_number,
                pa_output=None,
                failure=failure,
            )
            return failure if turn_failure is None else turn_failure

        assessment_input["ProductAgent_calls"] = assessment_product_agent.calls
        try:
            assessed_abox = load_interaction_abox(interaction_root, tbox)
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            return _failure(
                "product_context_invalid",
                f"Post-assessment ABox could not be reloaded: {type(exc).__name__}: {exc}",
            )
        if assessed_abox.delta_count != abox.delta_count:
            abox = assessed_abox
            assessment_input["post_assessment_delta_count"] = abox.delta_count
        validation_error = _assessment_validation_error(
            assessment_value,
            producer_descriptors=producer_descriptors,
            context_refs=tuple(context_ref_evidence_types),
            context_ref_evidence_types=context_ref_evidence_types,
            served_static_refs=served_static_refs,
            live_request_history=live_request_history,
        )
        assessment_output = (
            dict(assessment_value) if isinstance(assessment_value, Mapping) else assessment_value
        )
        if validation_error is not None:
            failure = _failure("invalid_assessment", validation_error)
            decision_failure = _write_decision_record(
                decision_path,
                operation_number=operation_number,
                turn_number=turn_number,
                product_requirement=product_requirement,
                assessment_input=assessment_input,
                assessment_output=assessment_output,
                failure=failure,
            )
            if decision_failure is not None:
                return decision_failure
            turn_failure = _write_turn_record(
                turn_path,
                turn_number=turn_number,
                product_requirement=product_requirement,
                operation_number=operation_number,
                pa_output=assessment_output,
                failure=failure,
            )
            return failure if turn_failure is None else turn_failure
        if not isinstance(assessment_output, dict):
            raise AssertionError("Validated assessment output must be a mapping.")

        decision_failure = _write_decision_record(
            decision_path,
            operation_number=operation_number,
            turn_number=turn_number,
            product_requirement=product_requirement,
            assessment_input=assessment_input,
            assessment_output=assessment_output,
            failure=None,
        )
        if decision_failure is not None:
            return decision_failure

        pa_output = {
            "needed_context": assessment_output["needed_context"],
            "context understanding complete": assessment_output["context understanding complete"],
        }
        if "grounding_status" in assessment_output:
            pa_output["grounding_status"] = assessment_output["grounding_status"]
        terminal = _is_terminal_assessment(assessment_output)
        limit_reached = (
            turn_number == max_pa_turns
            and assessment_output["context understanding complete"] is False
            and not terminal
        )
        turn_failure_value = (
            _failure(
                "pa_turn_limit_reached",
                f"Phase 4.3 requested more context on turn {turn_number} of {max_pa_turns}.",
            )
            if limit_reached
            else None
        )
        turn_failure = _write_turn_record(
            turn_path,
            turn_number=turn_number,
            product_requirement=product_requirement,
            operation_number=operation_number,
            pa_output=pa_output,
            failure=turn_failure_value,
        )
        if turn_failure is not None:
            return turn_failure
        if turn_failure_value is not None:
            return turn_failure_value
        if terminal:
            if assessment_output["context understanding complete"] is True:
                completion_error = _persist_pa_context_grounding_completion(
                    interaction_root,
                    tbox=tbox,
                    product_requirement=product_requirement,
                    completion_turn=turn_number,
                    decision_path=decision_path,
                    attempted_evidence=tuple(attempted_evidence),
                    clarification_history=clarification_history,
                )
                if completion_error is not None:
                    return _failure("context_completion_failed", completion_error)
            return pa_output

        needed_context = assessment_output["needed_context"]
        if not isinstance(needed_context, dict):
            raise AssertionError("Non-terminal assessment must request context.")
        requested_ref = needed_context["context_ref"]
        semantic_need = assessment_output["unresolved_semantic_need"]
        if isinstance(requested_ref, str):
            served_static_refs.add(requested_ref)
        elif isinstance(semantic_need, Mapping):
            live_request_history.add(_semantic_need_key(semantic_need))

        observation_ref = f"observation_{next_observation_number:04d}"
        retrieval_result = context_serving._serve_pa_requested_context_turn(
            interaction_root,
            turn_number=turn_number,
            retrieval_number=turn_number,
            observation_ref=observation_ref,
            live_observation_timeout_sec=float(live_observation_timeout_sec),
            phase_label="Phase 3.3",
        )
        if "failure" in retrieval_result:
            return retrieval_result
        next_served_context = retrieval_result.get("served_context")
        if not isinstance(next_served_context, dict):
            return _failure(
                "retrieval_failed",
                f"retrieval_{turn_number:04d} returned no served_context.",
            )
        served_context = next_served_context
        if served_context.get("evidence_type") == "observation":
            next_observation_number += 1

    raise AssertionError("Phase 3.3 loop must stop at a validated terminal state.")


async def _interpret_and_merge(  # noqa: PLR0913
    grounding_runtime: ProductContextGroundingRuntime,
    *,
    interaction_root: Path,
    tbox: TBoxSnapshot,
    abox: ABoxSnapshot,
    producer_descriptors: tuple[GroundingProducerDescriptor, ...],
    served_context: dict[str, object],
    attempted_evidence: tuple[str, ...],
    operation_number: int,
) -> dict[str, object]:
    interpretation_path = (
        interaction_root / "interaction_record" / f"interpretation_{operation_number:04d}.json"
    )
    if interpretation_path.exists():
        return _failure(
            "interaction_exists",
            f"Phase 3.3 interpretation_{operation_number:04d} already exists.",
        )
    evidence_kind = served_context.get("evidence_type")
    if evidence_kind not in {"document", "CAD", "observation"}:
        return _failure("invalid_interaction", "served_context evidence_type is invalid.")
    evidence_identifier = _evidence_identifier(served_context)
    if evidence_identifier is None:
        return _failure("invalid_interaction", "served_context evidence ref is invalid.")
    if (
        evidence_identifier in attempted_evidence
        and load_latest_grounding_session(interaction_root) is None
    ):
        return _failure(
            "duplicate_evidence_request",
            f"Evidence was already interpreted: {evidence_identifier}.",
        )

    try:
        producer = producer_for_evidence_type(
            producer_descriptors,
            str(evidence_kind),
        )
    except (TypeError, ValueError) as exc:
        return _failure(
            "grounding_unavailable",
            f"No unambiguous controlled producer is configured for {evidence_kind}: {exc}",
        )
    authorized_evidence_refs = _authorized_evidence_refs(served_context)
    try:
        delta_value = await grounding_runtime.interpret_served_context(
            producer=producer,
            interaction_root=interaction_root,
            tbox=tbox,
            abox=abox,
            served_context=served_context,
            operation_number=operation_number,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        logger.exception("Controlled Phase 4 interpretation failed.")
        failure = _failure(
            "interpretation_failed",
            f"{producer} failed: {type(exc).__name__}: {exc}",
        )
        record_failure = _write_interpretation_record(
            interpretation_path,
            operation_number=operation_number,
            producer=producer,
            evidence_kind=str(evidence_kind),
            evidence_identifier=evidence_identifier,
            authorized_evidence_refs=authorized_evidence_refs,
            tool_output=None,
            delta_ref=None,
            failure=failure,
        )
        return failure if record_failure is None else record_failure
    if not isinstance(delta_value, Mapping):
        failure = _failure(
            "invalid_interpretation",
            "Controlled interpretation must return one JSON-shaped triple delta.",
        )
        record_failure = _write_interpretation_record(
            interpretation_path,
            operation_number=operation_number,
            producer=producer,
            evidence_kind=str(evidence_kind),
            evidence_identifier=evidence_identifier,
            authorized_evidence_refs=authorized_evidence_refs,
            tool_output=delta_value,
            delta_ref=None,
            failure=failure,
        )
        return failure if record_failure is None else record_failure
    delta = dict(delta_value)
    try:
        json.dumps(delta, ensure_ascii=False, allow_nan=False)
        merge_result = validate_and_merge_triple_delta(
            interaction_root,
            tbox,
            producer,
            delta,
            authorized_evidence_refs=authorized_evidence_refs,
        )
    except (OSError, TypeError, ValueError) as exc:
        failure = _failure(
            "invalid_interpretation",
            f"Controlled triple delta was rejected: {type(exc).__name__}: {exc}",
        )
        record_failure = _write_interpretation_record(
            interpretation_path,
            operation_number=operation_number,
            producer=producer,
            evidence_kind=str(evidence_kind),
            evidence_identifier=evidence_identifier,
            authorized_evidence_refs=authorized_evidence_refs,
            tool_output=delta,
            delta_ref=None,
            failure=failure,
        )
        return failure if record_failure is None else record_failure

    delta_ref = str(merge_result.delta_path.relative_to(interaction_root))
    record_failure = _write_interpretation_record(
        interpretation_path,
        operation_number=operation_number,
        producer=producer,
        evidence_kind=str(evidence_kind),
        evidence_identifier=evidence_identifier,
        authorized_evidence_refs=authorized_evidence_refs,
        tool_output=delta,
        delta_ref=delta_ref,
        failure=None,
    )
    if record_failure is not None:
        return record_failure
    return {
        "abox": merge_result.abox,
        "evidence_identifier": evidence_identifier,
    }


def _configuration_failure(
    *,
    ontology_config: object,
    grounding_runtime: object,
    max_pa_turns: object,
    live_observation_timeout_sec: object,
) -> dict[str, object] | None:
    if ontology_config is None or grounding_runtime is None:
        return _failure(
            "grounding_unavailable",
            "An authoritative TBox and controlled Phase 4 grounding runtime are "
            "required before Phase 3.3 can continue.",
        )
    if not isinstance(max_pa_turns, int) or isinstance(max_pa_turns, bool) or max_pa_turns < 2:
        return _failure(
            "invalid_max_pa_turns",
            "max_pa_turns must be an integer of at least 2 and includes turn_0001.",
        )
    if (
        not isinstance(live_observation_timeout_sec, (int, float))
        or isinstance(live_observation_timeout_sec, bool)
        or not math.isfinite(float(live_observation_timeout_sec))
        or live_observation_timeout_sec <= 0
    ):
        return _failure(
            "invalid_live_observation_timeout",
            "live_observation_timeout_sec must be a finite positive number.",
        )
    return None


def _assessment_validation_error(  # noqa: C901
    assessment: object,
    *,
    producer_descriptors: tuple[GroundingProducerDescriptor, ...],
    context_refs: tuple[str, ...],
    context_ref_evidence_types: Mapping[str, str],
    served_static_refs: set[str],
    live_request_history: set[str],
) -> str | None:
    if not isinstance(assessment, Mapping) or frozenset(assessment) not in {
        frozenset(_ASSESSMENT_KEYS),
        frozenset(_SESSION_ASSESSMENT_KEYS),
    }:
        return "Phase 4.3 output fields are invalid."
    complete = assessment["context understanding complete"]
    needed_context = assessment["needed_context"]
    semantic_need = assessment["unresolved_semantic_need"]
    grounding_status = assessment.get("grounding_status")
    if not isinstance(complete, bool):
        return "context understanding complete must be a boolean."
    if grounding_status is not None and grounding_status not in {
        "waiting_for_evidence",
        "waiting_for_user",
        "complete",
        "incomplete",
        "ontology_gap",
    }:
        return "grounding_status is invalid."
    if complete:
        if any(value is not None for value in (needed_context, semantic_need)):
            return "A completed assessment cannot retain a semantic need or request."
        if grounding_status is not None and grounding_status != "complete":
            return "A completed assessment requires grounding_status complete."
        return None
    if grounding_status in {"incomplete", "ontology_gap"}:
        if needed_context is not None or semantic_need is not None:
            return "A terminal grounding state cannot retain a request."
        return None
    semantic_need_error = _semantic_need_validation_error(semantic_need)
    if semantic_need_error is not None:
        return semantic_need_error
    if not isinstance(needed_context, dict):
        return "An incomplete assessment requires needed_context."
    validation_error = context_serving._needed_context_validation_error(
        needed_context,
        context_refs,
        phase_label="Phase 4.3",
    )
    if validation_error is not None:
        return validation_error

    context_ref = needed_context["context_ref"]
    clarification = needed_context["clarification_question"]
    if isinstance(clarification, str):
        if not isinstance(semantic_need, Mapping) or semantic_need.get("kind") != "user_intent":
            return "A clarification decision requires a user_intent semantic need."
        return None
    evidence_kind = (
        context_ref_evidence_types.get(context_ref)
        if isinstance(context_ref, str)
        else "observation"
    )
    descriptor_error = _descriptor_request_validation_error(
        semantic_need,
        evidence_kind=evidence_kind,
        producer_descriptors=producer_descriptors,
    )
    if descriptor_error is not None:
        return descriptor_error
    if context_ref is None and _semantic_need_key(semantic_need) in live_request_history:
        return "A repeated live observation requires a new semantic need."
    if grounding_status is not None:
        return None
    if isinstance(context_ref, str) and context_ref in served_static_refs:
        return f"Phase 4.3 context_ref was already served: {context_ref}."
    return None


def _descriptor_request_validation_error(
    semantic_need: object,
    *,
    evidence_kind: object,
    producer_descriptors: tuple[GroundingProducerDescriptor, ...],
) -> str | None:
    if not isinstance(semantic_need, Mapping) or not isinstance(evidence_kind, str):
        return "A producer request requires one structured semantic need."
    if set(semantic_need) != {"kind", "symbol", "description"}:
        return "Phase 4.3 semantic need fields are invalid."
    kind = semantic_need.get("kind")
    symbol = semantic_need.get("symbol")
    description = semantic_need.get("description")
    if kind not in {"class", "property", "individual", "typed_context_record"}:
        return "Phase 4.3 semantic need kind cannot be routed."
    if any(
        not isinstance(item, str) or not item.strip()
        for item in (symbol, description)
    ):
        return "Phase 4.3 semantic need values cannot be routed."
    matches = [
        descriptor
        for descriptor in producer_descriptors
        if evidence_kind in descriptor.accepted_evidence_types
        and (
            kind != "typed_context_record"
            or descriptor.supports_record_type(str(symbol))
        )
    ]
    if not matches:
        return (
            "No controlled producer advertises the unresolved output and "
            "requested evidence type."
        )
    return None


def _semantic_need_validation_error(semantic_need: object) -> str | None:
    if not isinstance(semantic_need, Mapping) or set(semantic_need) != {
        "kind",
        "symbol",
        "description",
    }:
        return "An incomplete assessment requires one structured unresolved_semantic_need."
    kind = semantic_need["kind"]
    symbol = semantic_need["symbol"]
    description = semantic_need["description"]
    if kind not in {"class", "property", "typed_context_record", "user_intent"}:
        return "unresolved_semantic_need kind is invalid."
    if not isinstance(symbol, str) or not symbol:
        return "unresolved_semantic_need symbol must be non-empty."
    if not isinstance(description, str) or not description.strip():
        return "unresolved_semantic_need description must be non-empty."
    return None


def _semantic_need_key(semantic_need: Mapping[str, object]) -> str:
    return json.dumps(semantic_need, sort_keys=True, ensure_ascii=False)


def _is_terminal_assessment(assessment: Mapping[str, object]) -> bool:
    if assessment["context understanding complete"] is True:
        return True
    if assessment.get("grounding_status") in {"incomplete", "ontology_gap"}:
        return True
    needed_context = assessment["needed_context"]
    return isinstance(needed_context, dict) and isinstance(
        needed_context["clarification_question"],
        str,
    )


def _evidence_identifier(served_context: Mapping[str, object]) -> str | None:
    context_ref = served_context.get("context_ref")
    if isinstance(context_ref, str):
        return context_ref
    observation_ref = served_context.get("observation_ref")
    return observation_ref if isinstance(observation_ref, str) else None


def _authorized_evidence_refs(
    served_context: Mapping[str, object],
) -> tuple[str, ...]:
    evidence_identifier = _evidence_identifier(served_context)
    if evidence_identifier is None:
        return ()
    refs = [evidence_identifier]
    if served_context.get("evidence_type") == "document":
        document_evidence = served_context.get("document_evidence")
        pages = document_evidence.get("pages") if isinstance(document_evidence, Mapping) else None
        if isinstance(pages, list):
            for page in pages:
                page_number = page.get("page") if isinstance(page, Mapping) else None
                if isinstance(page_number, int) and not isinstance(page_number, bool):
                    refs.append(f"{evidence_identifier}#page={page_number}")
    return tuple(refs)


def _read_initial_interaction(
    interaction_root: Path,
) -> tuple[dict[str, Any] | None, str | None]:
    product_requirement, needed_context, validation_error = context_serving._read_phase_3_1_records(
        interaction_root
    )
    if validation_error is not None:
        return None, validation_error
    if product_requirement is None or needed_context is None:
        return None, "Phase 3.1 records are incomplete."

    turn_path = interaction_root / "interaction_record/turn_0001.json"
    retrieval_path = interaction_root / "interaction_record/retrieval_0001.json"
    try:
        turn_record = _read_json(turn_path)
        retrieval_record = _read_json(retrieval_path)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None, "Successful Phase 3.1 and Phase 3.2 records could not be read."
    if not isinstance(turn_record, dict) or set(turn_record) != _TURN_KEYS:
        return None, "turn_0001.json fields are invalid."
    if not isinstance(retrieval_record, dict) or set(retrieval_record) != _RETRIEVAL_KEYS:
        return None, "retrieval_0001.json fields are invalid."
    if (
        retrieval_record["retrieval"] != 1
        or retrieval_record["product_requirement"] != product_requirement
        or retrieval_record["needed_context"] != needed_context
        or not isinstance(retrieval_record["context_request"], dict)
        or not isinstance(retrieval_record["served_context"], dict)
        or retrieval_record["retrieval_error"] is not None
        or retrieval_record["failure"] is not None
    ):
        return None, "retrieval_0001.json does not record successful Phase 3.2 serving."
    served_context = retrieval_record["served_context"]
    served_error = _initial_served_context_error(served_context, needed_context)
    if served_error is not None:
        return None, served_error
    return {
        "product_requirement": product_requirement,
        "served_context": served_context,
    }, None


def _initial_served_context_error(
    served_context: dict[str, object],
    needed_context: dict[str, object],
) -> str | None:
    context_ref = needed_context["context_ref"]
    if isinstance(context_ref, str):
        validation_error = context_serving._static_result_validation_error(
            {"served_context": served_context},
            context_ref,
        )
        return (
            None
            if validation_error is None
            else f"retrieval_0001 served_context is invalid: {validation_error}"
        )
    if needed_context["request_live_observation"] is not True:
        return "retrieval_0001 does not correspond to a retrievable request."
    required_keys = {
        "context_ref",
        "observation_ref",
        "evidence_type",
        "evidence_label",
        "provenance",
        "observation_evidence",
    }
    if (
        set(served_context) != required_keys
        or served_context["context_ref"] is not None
        or served_context["observation_ref"] != "observation_0001"
        or served_context["evidence_type"] != "observation"
        or served_context["evidence_label"] != "live"
        or not isinstance(served_context["provenance"], dict)
        or not isinstance(served_context["observation_evidence"], dict)
    ):
        return "retrieval_0001 live served_context is invalid."
    return None


def _read_pa_context_settings(
    interaction_root: Path,
) -> tuple[dict[str, Any] | None, str | None]:
    try:
        settings = _read_json(interaction_root / _SETTINGS_PATH)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None, "Phase 3.3 settings could not be read."
    if not isinstance(settings, dict) or set(settings) != {
        "max_pa_turns",
        "live_observation_timeout_sec",
    }:
        return None, "Phase 3.3 settings fields are invalid."
    max_pa_turns = settings["max_pa_turns"]
    timeout = settings["live_observation_timeout_sec"]
    validation_error = _configuration_failure(
        ontology_config=object(),
        grounding_runtime=object(),
        max_pa_turns=max_pa_turns,
        live_observation_timeout_sec=timeout,
    )
    if validation_error is not None:
        return None, str(validation_error["failure"]["message"])
    return {
        "max_pa_turns": max_pa_turns,
        "live_observation_timeout_sec": float(timeout),
    }, None


def _persist_pa_context_grounding_completion(  # noqa: PLR0913
    interaction_root: Path,
    *,
    tbox: TBoxSnapshot,
    product_requirement: str,
    completion_turn: int,
    decision_path: Path,
    attempted_evidence: tuple[str, ...],
    clarification_history: tuple[Mapping[str, object], ...],
) -> str | None:
    try:
        tbox.assert_unchanged()
        abox = load_interaction_abox(interaction_root, tbox)
        session = load_latest_grounding_session(interaction_root)
        if session is not None:
            if session.status != "complete":
                return "GroundingSession is not complete for Phase 3.5."
            fresh_view = build_product_context_view(
                interaction_root,
                abox,
                attempted_evidence=attempted_evidence,
                assessed_at_ns=time.time_ns(),
            )
            persist_product_context_view(interaction_root, fresh_view)
            clarification_refs: list[str] = []
            for item in clarification_history:
                question_turn = item.get("question_turn")
                if not isinstance(question_turn, int) or isinstance(
                    question_turn, bool
                ):
                    return "Clarification history contains an invalid question turn."
                path = (
                    interaction_root
                    / "interaction_record"
                    / f"clarification_{question_turn:04d}.json"
                )
                persisted, error = _existing_clarification_record(path)
                if error is not None or persisted != item:
                    return error or (
                        "Clarification history does not match its persisted record."
                    )
                if persisted.get("action") != "answered":
                    return "Cancelled clarification cannot enter Phase 3.5."
                clarification_refs.append(str(path.relative_to(interaction_root)))
            persist_pa_context_grounding_completion_v2(
                interaction_root,
                product_requirement=product_requirement,
                completion_turn=completion_turn,
                decision_ref=str(decision_path.relative_to(interaction_root)),
                product_context=fresh_view,
                clarification_refs=clarification_refs,
            )
            return None
        return "No complete GroundingSession is available for Phase 3.5."
    except (
        FileExistsError,
        GroundingContractError,
        OSError,
        TypeError,
        UnicodeDecodeError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        return f"Phase 3.5 validation failed: {type(exc).__name__}: {exc}"
    return None


def _pending_clarification(
    interaction_root: Path,
) -> tuple[dict[str, object] | None, str | None]:
    decision_paths = sorted(
        (interaction_root / "interaction_record").glob("decision_*.json")
    )
    if not decision_paths:
        return None, "No persisted Phase 4.3 decision exists."
    try:
        decision = _read_json(decision_paths[-1])
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None, "The latest Phase 4.3 decision could not be read."
    expected_keys = {
        "decision",
        "turn",
        "product_requirement",
        "Phase_4_3_input",
        "Phase_4_3_output",
        "failure",
    }
    if not isinstance(decision, dict) or set(decision) != expected_keys:
        return None, "The latest Phase 4.3 decision fields are invalid."
    if decision["failure"] is not None:
        return None, "A failed interaction cannot enter Phase 3.4."
    turn = decision["turn"]
    output = decision["Phase_4_3_output"]
    if not isinstance(turn, int) or isinstance(turn, bool):
        return None, "The latest clarification turn is invalid."
    if not isinstance(output, dict) or set(output) != _SESSION_ASSESSMENT_KEYS:
        return None, "The latest Phase 4.3 output is invalid."
    semantic_need = output["unresolved_semantic_need"]
    needed_context = output["needed_context"]
    if output["context understanding complete"] is not False:
        return None, "A completed interaction cannot enter Phase 3.4."
    if output["grounding_status"] != "waiting_for_user":
        return None, "The latest decision is not waiting for user clarification."
    if (
        not isinstance(semantic_need, dict)
        or semantic_need.get("kind") != "user_intent"
        or set(semantic_need) != {"kind", "symbol", "description"}
    ):
        return None, "Phase 3.4 requires one pending user_intent need."
    if (
        not isinstance(needed_context, dict)
        or set(needed_context)
        != {"context_ref", "request_live_observation", "clarification_question"}
        or needed_context["context_ref"] is not None
        or needed_context["request_live_observation"] is not False
        or not isinstance(needed_context["clarification_question"], str)
        or not needed_context["clarification_question"].strip()
    ):
        return None, "The latest decision has no valid clarification question."
    turn_path = interaction_root / "interaction_record" / f"turn_{turn:04d}.json"
    try:
        turn_record = _read_json(turn_path)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None, "The clarification ProductAgent turn could not be read."
    if (
        not isinstance(turn_record, dict)
        or set(turn_record) != _TURN_KEYS
        or turn_record["failure"] is not None
        or turn_record["product_requirement"] != decision["product_requirement"]
        or turn_record["PA_output"]
        != {
            "needed_context": needed_context,
            "context understanding complete": False,
            "grounding_status": "waiting_for_user",
        }
    ):
        return None, "The clarification ProductAgent turn is invalid."
    return {
        "product_requirement": decision["product_requirement"],
        "question_turn": turn,
        "semantic_need": semantic_need,
        "question": needed_context["clarification_question"],
    }, None


def _clarification_record(
    *,
    product_requirement: str,
    question_turn: int,
    semantic_need: object,
    question: str,
    action: str,
    reply: str | None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": 1,
        "record_type": "PAClarification",
        "product_requirement": product_requirement,
        "question_turn": question_turn,
        "semantic_need": semantic_need,
        "question": question,
        "action": action,
        "reply": reply,
        "recorded_at_ns": time.time_ns(),
    }
    payload["fingerprint"] = _record_fingerprint(payload)
    return payload


def _existing_clarification_record(
    path: Path,
) -> tuple[dict[str, object] | None, str | None]:
    if not path.exists():
        return None, None
    try:
        value = _read_json(path)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None, f"Clarification record is unreadable: {path.name}."
    error = _clarification_record_validation_error(value)
    return (None, error) if error is not None else (value, None)


def _clarification_record_validation_error(value: object) -> str | None:
    if not isinstance(value, dict) or set(value) != _CLARIFICATION_KEYS:
        return "Clarification record fields are invalid."
    if value["schema_version"] != 1 or value["record_type"] != "PAClarification":
        return "Clarification record identity is invalid."
    if not isinstance(value["product_requirement"], str) or not value[
        "product_requirement"
    ]:
        return "Clarification product_requirement is invalid."
    if (
        not isinstance(value["question_turn"], int)
        or isinstance(value["question_turn"], bool)
        or value["question_turn"] < 2
    ):
        return "Clarification question_turn is invalid."
    semantic_need = value["semantic_need"]
    if (
        not isinstance(semantic_need, dict)
        or set(semantic_need) != {"kind", "symbol", "description"}
        or semantic_need.get("kind") != "user_intent"
    ):
        return "Clarification semantic_need is invalid."
    if not isinstance(value["question"], str) or not value["question"].strip():
        return "Clarification question is invalid."
    action = value["action"]
    reply = value["reply"]
    if action not in {"answered", "cancelled"}:
        return "Clarification action is invalid."
    if action == "answered" and (
        not isinstance(reply, str) or not reply.strip()
    ):
        return "Answered clarification reply is invalid."
    if action == "cancelled" and reply is not None:
        return "Cancelled clarification must not contain a reply."
    if not isinstance(value["recorded_at_ns"], int) or isinstance(
        value["recorded_at_ns"], bool
    ):
        return "Clarification timestamp is invalid."
    fingerprint = value["fingerprint"]
    payload = {key: item for key, item in value.items() if key != "fingerprint"}
    if not isinstance(fingerprint, str) or fingerprint != _record_fingerprint(payload):
        return "Clarification fingerprint is invalid."
    return None


def _clarification_resume_state(  # noqa: C901
    interaction_root: Path,
    *,
    product_requirement: str,
    max_pa_turns: int,
) -> tuple[dict[str, object] | None, str | None]:
    pending, pending_error = _pending_clarification(interaction_root)
    if pending_error is not None or pending is None:
        return None, pending_error or "No clarification can be resumed."
    history: list[dict[str, object]] = []
    for path in sorted(
        (interaction_root / "interaction_record").glob("clarification_*.json")
    ):
        record, record_error = _existing_clarification_record(path)
        if record_error is not None or record is None:
            return None, record_error or f"Clarification record is invalid: {path.name}."
        if record["product_requirement"] != product_requirement:
            return None, "Clarification history product_requirement is inconsistent."
        history.append(record)
    if not history:
        return None, "The pending clarification has no persisted user reply."
    latest = history[-1]
    if latest["action"] != "answered":
        return None, "A cancelled interaction cannot be resumed."
    if (
        latest["question_turn"] != pending["question_turn"]
        or latest["semantic_need"] != pending["semantic_need"]
        or latest["question"] != pending["question"]
    ):
        return None, "The latest clarification reply does not match the pending question."
    operation_number = int(latest["question_turn"])
    if operation_number >= max_pa_turns:
        return None, "No ProductAgent turn remains after clarification."

    attempted_evidence: list[str] = []
    for path in sorted(
        (interaction_root / "interaction_record").glob("interpretation_*.json")
    ):
        try:
            record = _read_json(path)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return None, f"Interpretation history is unreadable: {path.name}."
        if (
            not isinstance(record, dict)
            or record.get("accepted") is not True
            or record.get("failure") is not None
            or not isinstance(record.get("evidence_identifier"), str)
        ):
            return None, f"Interpretation history is invalid: {path.name}."
        attempted_evidence.append(record["evidence_identifier"])

    served_contexts: list[dict[str, object]] = []
    for path in sorted(
        (interaction_root / "interaction_record").glob("retrieval_*.json")
    ):
        try:
            record = _read_json(path)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return None, f"Retrieval history is unreadable: {path.name}."
        served = record.get("served_context") if isinstance(record, dict) else None
        if not isinstance(served, dict):
            return None, f"Retrieval history is invalid: {path.name}."
        served_contexts.append(served)

    live_request_history: set[str] = set()
    for path in sorted(
        (interaction_root / "interaction_record").glob("decision_*.json")
    ):
        try:
            decision = _read_json(path)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return None, f"Decision history is unreadable: {path.name}."
        output = decision.get("Phase_4_3_output") if isinstance(decision, dict) else None
        if not isinstance(output, dict):
            continue
        needed = output.get("needed_context")
        semantic_need = output.get("unresolved_semantic_need")
        if (
            isinstance(needed, dict)
            and needed.get("request_live_observation") is True
            and isinstance(semantic_need, Mapping)
        ):
            live_request_history.add(_semantic_need_key(semantic_need))
    return {
        "operation_number": operation_number,
        "attempted_evidence": attempted_evidence,
        "served_contexts": served_contexts,
        "live_request_history": live_request_history,
        "clarification_history": history,
    }, None


def _record_fingerprint(value: Mapping[str, object]) -> str:
    serialized = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _served_static_refs(served_contexts: list[dict[str, object]]) -> set[str]:
    return {
        context_ref
        for served_context in served_contexts
        if isinstance((context_ref := served_context.get("context_ref")), str)
    }


def _next_observation_number(served_contexts: list[dict[str, object]]) -> int:
    observation_numbers = []
    for served_context in served_contexts:
        observation_ref = served_context.get("observation_ref")
        if not isinstance(observation_ref, str) or not observation_ref.startswith("observation_"):
            continue
        suffix = observation_ref.removeprefix("observation_")
        if suffix.isdigit():
            observation_numbers.append(int(suffix))
    return max(observation_numbers, default=0) + 1


def _existing_phase_3_3_record(interaction_root: Path) -> str | None:
    settings_path = interaction_root / _SETTINGS_PATH
    if settings_path.exists():
        return str(_SETTINGS_PATH)
    record_root = interaction_root / "interaction_record"
    for pattern in (
        "turn_*.json",
        "retrieval_*.json",
        "interpretation_*.json",
        "decision_*.json",
    ):
        for path in sorted(record_root.glob(pattern)):
            if path.name not in {"turn_0001.json", "retrieval_0001.json"}:
                return str(path.relative_to(interaction_root))
    return None


def _is_initial_provisional_grounding(
    interaction_root: Path,
    abox: ABoxSnapshot,
) -> bool:
    """Accept only the exact Phase 4.4 semantic delta before first retrieval."""
    if abox.delta_count != 1:
        return False
    session = load_latest_grounding_session(interaction_root)
    if (
        session is None
        or session.status != "waiting_for_evidence"
        or session.next_action.action not in {"retrieve", "inspect"}
    ):
        return False
    proposals = sorted(
        (
            Path(interaction_root)
            / "products/grounding/ontology_grounding"
        ).glob("proposal_*.json")
    )
    if len(proposals) != 1:
        return False
    proposal = _read_json(proposals[0])
    if not isinstance(proposal, Mapping) or proposal.get("status") != "accepted":
        return False
    delta = _read_json(
        Path(interaction_root) / "products/grounding/ontology/delta_0001.json"
    )
    if (
        not isinstance(delta, Mapping)
        or delta.get("producer") != "ontology_grounding"
        or delta.get("typed_context_refs") != []
        or delta.get("unresolved_evidence_needs") != []
    ):
        return False
    assertions = delta.get("assertions")
    if not isinstance(assertions, list) or len(assertions) != 3:
        return False
    feature_iri = f"{abox.namespace}medium_gear_feature"
    specification_iri = abox.specification_iri
    normalized = {
        (
            item.get("subject"),
            str(item.get("predicate", "")).rsplit("#", 1)[-1],
            (
                item.get("object", {}).get("value")
                if isinstance(item.get("object"), Mapping)
                else None
            ),
        )
        for item in assertions
        if isinstance(item, Mapping)
    }
    return normalized == {
        (feature_iri, "type", "http://PAonto.com#feature"),
        (specification_iri, "defines", feature_iri),
        (
            "https://cais-spade-llm.local/process/assembly",
            "realizes",
            feature_iri,
        ),
    }


def _write_interpretation_record(  # noqa: PLR0913
    path: Path,
    *,
    operation_number: int,
    producer: str,
    evidence_kind: str,
    evidence_identifier: str,
    authorized_evidence_refs: tuple[str, ...],
    tool_output: object,
    delta_ref: str | None,
    failure: dict[str, object] | None,
) -> dict[str, object] | None:
    return _write_record_or_failure(
        path,
        {
            "interpretation": operation_number,
            "retrieval": operation_number,
            "producer": producer,
            "evidence_kind": evidence_kind,
            "evidence_identifier": evidence_identifier,
            "authorized_evidence_refs": list(authorized_evidence_refs),
            "tool_output": tool_output,
            "accepted": failure is None,
            "delta_ref": delta_ref,
            "failure": None if failure is None else failure["failure"],
        },
        label=f"interpretation_{operation_number:04d}",
    )


def _write_decision_record(
    path: Path,
    *,
    operation_number: int,
    turn_number: int,
    product_requirement: str,
    assessment_input: Mapping[str, object],
    assessment_output: object,
    failure: dict[str, object] | None,
) -> dict[str, object] | None:
    return _write_record_or_failure(
        path,
        {
            "decision": operation_number,
            "turn": turn_number,
            "product_requirement": product_requirement,
            "Phase_4_3_input": assessment_input,
            "Phase_4_3_output": assessment_output,
            "failure": None if failure is None else failure["failure"],
        },
        label=f"decision_{operation_number:04d}",
    )


def _write_turn_record(
    path: Path,
    *,
    turn_number: int,
    product_requirement: str,
    operation_number: int,
    pa_output: object,
    failure: dict[str, object] | None,
) -> dict[str, object] | None:
    return _write_record_or_failure(
        path,
        {
            "turn": turn_number,
            "product_requirement": product_requirement,
            "PA_input": {
                "assessment_ref": (f"interaction_record/decision_{operation_number:04d}.json")
            },
            "PA_output": pa_output,
            "failure": None if failure is None else failure["failure"],
        },
        label=f"turn_{turn_number:04d}",
    )


def _write_record_or_failure(
    path: Path,
    value: object,
    *,
    label: str,
) -> dict[str, object] | None:
    try:
        _write_json_exclusive(path, value)
    except FileExistsError:
        return _failure("interaction_exists", f"Phase 3.3 {label} already exists.")
    except (OSError, TypeError, ValueError) as exc:
        return _failure(
            "invalid_interaction",
            f"{label} write failed: {type(exc).__name__}: {exc}",
        )
    return None


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _write_json_exclusive(path: Path, value: object) -> None:
    serialized = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(serialized)


def _failure(reason: str, message: str) -> dict[str, object]:
    return {"failure": {"reason": reason, "message": message}}
