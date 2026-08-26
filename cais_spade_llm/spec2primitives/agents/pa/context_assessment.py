"""Orchestrate ontology-backed Spec2Primitives PA context grounding."""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from cais_spade_llm.spec2primitives.agents.pa import context_serving
from cais_spade_llm.spec2primitives.agents.pa.context_grounding import (
    PAOntologyConfig,
    ProductContextGroundingRuntime,
    compact_abox_view,
    validated_grounding_producer_routes,
)
from cais_spade_llm.spec2primitives.agents.pa.context_interaction import (
    ProductAgentContextRuntime,
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

    existing_record = _existing_phase_3_3_record(interaction_root)
    if existing_record is not None:
        return _failure(
            "interaction_exists",
            f"Phase 3.3 record already exists: {existing_record}.",
        )

    try:
        tbox = ontology_config.load_tbox()
        producer_routes = validated_grounding_producer_routes(grounding_runtime)
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
    if abox.delta_count != 0:
        return _failure(
            "invalid_interaction",
            "Phase 3.3 must start before any interaction delta exists.",
        )

    settings = {
        "max_pa_turns": max_pa_turns,
        "live_observation_timeout_sec": float(live_observation_timeout_sec),
    }
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
    attempted_evidence: list[str] = []
    served_static_refs = _served_static_refs([served_context])
    live_request_history: set[str] = set()
    next_observation_number = _next_observation_number([served_context])

    for operation_number in range(1, max_pa_turns):
        interpretation = await _interpret_and_merge(
            grounding_runtime,
            interaction_root=interaction_root,
            tbox=tbox,
            abox=abox,
            producer_routes=producer_routes,
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
        attempted_evidence.append(evidence_identifier)

        turn_number = operation_number + 1
        assessment_input = {
            "product_context": compact_abox_view(abox),
            "attempted_evidence": list(attempted_evidence),
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
        validation_error = _assessment_validation_error(
            assessment_value,
            producer_routes=producer_routes,
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
        terminal = _is_terminal_assessment(assessment_output)
        limit_reached = turn_number == max_pa_turns and not terminal
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
        if terminal:
            return pa_output
        if turn_failure_value is not None:
            return turn_failure_value

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
    producer_routes: Mapping[str, str],
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
    if abox.delta_count != operation_number - 1:
        return _failure(
            "invalid_interaction",
            "ABox delta numbering is not aligned with Phase 3.3 retrievals.",
        )

    evidence_kind = served_context.get("evidence_type")
    if evidence_kind not in {"document", "CAD", "observation"}:
        return _failure("invalid_interaction", "served_context evidence_type is invalid.")
    evidence_identifier = _evidence_identifier(served_context)
    if evidence_identifier is None:
        return _failure("invalid_interaction", "served_context evidence ref is invalid.")
    if evidence_identifier in attempted_evidence:
        return _failure(
            "duplicate_evidence_request",
            f"Evidence was already interpreted: {evidence_identifier}.",
        )

    producer = producer_routes.get(str(evidence_kind))
    if not isinstance(producer, str):
        return _failure(
            "grounding_unavailable",
            f"No controlled producer is configured for {evidence_kind}.",
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
    producer_routes: Mapping[str, str],
    context_refs: tuple[str, ...],
    context_ref_evidence_types: Mapping[str, str],
    served_static_refs: set[str],
    live_request_history: set[str],
) -> str | None:
    if not isinstance(assessment, Mapping) or set(assessment) != _ASSESSMENT_KEYS:
        return "Phase 4.3 output fields are invalid."
    complete = assessment["context understanding complete"]
    needed_context = assessment["needed_context"]
    semantic_need = assessment["unresolved_semantic_need"]
    if not isinstance(complete, bool):
        return "context understanding complete must be a boolean."
    if complete:
        if any(value is not None for value in (needed_context, semantic_need)):
            return "A completed assessment cannot retain a semantic need or request."
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
    if evidence_kind not in producer_routes:
        return "No controlled producer is configured for the requested evidence."
    if isinstance(context_ref, str) and context_ref in served_static_refs:
        return f"Phase 4.3 context_ref was already served: {context_ref}."
    if context_ref is None and _semantic_need_key(semantic_need) in live_request_history:
        return "A repeated live observation requires a new semantic need."
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
