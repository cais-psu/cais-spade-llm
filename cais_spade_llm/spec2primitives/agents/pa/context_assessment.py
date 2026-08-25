"""Continue the Spec2Primitives PA context interaction after Phase 3.2."""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

from cais_spade_llm.spec2primitives.agents.pa import context_serving
from cais_spade_llm.spec2primitives.agents.pa.context_interaction import (
    ProductAgentContextRuntime,
)
from cais_spade_llm.spec2primitives.tools.exact_ref_resolver import (
    approved_context_refs,
)

logger = logging.getLogger(__name__)

_PRODUCT_REQUIREMENT_PATH = Path(
    "products/user_requirement/product_requirement.json"
)
_SETTINGS_PATH = Path("interaction_record/pa_context_settings.json")
_TURN_KEYS = {"turn", "product_requirement", "PA_input", "PA_output", "failure"}
_PA_INPUT_KEYS = {"prompt", "response_format"}
_RETRIEVAL_KEYS = {
    "retrieval",
    "product_requirement",
    "needed_context",
    "context_request",
    "served_context",
    "retrieval_error",
    "failure",
}
_REASSESSMENT_KEYS = {"needed_context", "context understanding complete"}


async def continue_pa_context_interaction(
    product_agent: ProductAgentContextRuntime,
    interaction_root: Path,
    *,
    max_pa_turns: int = 12,
    live_observation_timeout_sec: float = 5.0,
) -> dict[str, object]:
    """Reassess and serve PA context until the interaction pauses or completes.

    Args:
        product_agent: Shared ProductAgent viewed through the narrow structured
            composition protocol.
        interaction_root: Caller-owned `contexts/<interaction_identifier>/`
            directory containing successful Phase 3.1 and Phase 3.2 records.
        max_pa_turns: Maximum total PA turns, including `turn_0001`.
        live_observation_timeout_sec: Maximum wait for each explicitly requested
            fresh live RGB-D observation.

    Returns:
        The terminal validated PA response or a structured failure.
    """
    interaction_root = Path(interaction_root)
    initial, context_refs, preparation_failure = _prepare_context_assessment(
        interaction_root,
        max_pa_turns=max_pa_turns,
        live_observation_timeout_sec=live_observation_timeout_sec,
    )
    if preparation_failure is not None:
        return preparation_failure
    if initial is None or context_refs is None:
        raise AssertionError("Validated Phase 3.3 preparation is required.")

    product_requirement = initial["product_requirement"]
    served_contexts = [initial["served_context"]]
    recorded_decisions = [initial["PA_output"]]
    served_static_refs = _served_static_refs(served_contexts)
    next_observation_number = _next_observation_number(served_contexts)

    for turn_number in range(2, max_pa_turns + 1):
        remaining_context_refs = tuple(
            context_ref
            for context_ref in context_refs
            if context_ref not in served_static_refs
        )
        response_format = _assessment_response_format(remaining_context_refs)
        prompt = _assessment_prompt(
            product_requirement=product_requirement,
            served_contexts=served_contexts,
            recorded_decisions=recorded_decisions,
            remaining_context_refs=remaining_context_refs,
            turn_number=turn_number,
            max_pa_turns=max_pa_turns,
        )
        pa_input = {"prompt": prompt, "response_format": response_format}
        turn_path = (
            interaction_root
            / "interaction_record"
            / f"turn_{turn_number:04d}.json"
        )
        pa_output, pa_failure = await _request_pa_assessment(
            product_agent,
            turn_path=turn_path,
            turn_number=turn_number,
            product_requirement=product_requirement,
            pa_input=pa_input,
            context_refs=context_refs,
            served_static_refs=served_static_refs,
        )
        if pa_failure is not None:
            return pa_failure
        if pa_output is None:
            raise AssertionError("Validated Phase 3.3 PA output is required.")

        if _is_terminal_pa_output(pa_output):
            write_failure = _write_turn_record(
                turn_path,
                turn_number=turn_number,
                product_requirement=product_requirement,
                pa_input=pa_input,
                pa_output=pa_output,
                failure=None,
            )
            return pa_output if write_failure is None else write_failure

        if turn_number == max_pa_turns:
            failure = _failure(
                "pa_turn_limit_reached",
                f"PA requested more context on turn {turn_number} of {max_pa_turns}.",
            )
            return _write_turn_or_failure(
                turn_path,
                turn_number=turn_number,
                product_requirement=product_requirement,
                pa_input=pa_input,
                pa_output=pa_output,
                failure=failure,
            )

        write_failure = _write_turn_record(
            turn_path,
            turn_number=turn_number,
            product_requirement=product_requirement,
            pa_input=pa_input,
            pa_output=pa_output,
            failure=None,
        )
        if write_failure is not None:
            return write_failure

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

        served_context = retrieval_result.get("served_context")
        if not isinstance(served_context, dict):
            return _failure(
                "retrieval_failed",
                f"retrieval_{turn_number:04d} returned no served_context.",
            )
        served_contexts.append(served_context)
        recorded_decisions.append(pa_output)
        context_ref = served_context.get("context_ref")
        if isinstance(context_ref, str):
            served_static_refs.add(context_ref)
        if served_context.get("evidence_type") == "observation":
            next_observation_number += 1

    raise AssertionError("Phase 3.3 loop must return from an allowed terminal state.")


def _prepare_context_assessment(
    interaction_root: Path,
    *,
    max_pa_turns: object,
    live_observation_timeout_sec: object,
) -> tuple[
    dict[str, object] | None,
    tuple[str, ...] | None,
    dict[str, object] | None,
]:
    configuration_failure = _configuration_failure(
        max_pa_turns=max_pa_turns,
        live_observation_timeout_sec=live_observation_timeout_sec,
    )
    if configuration_failure is not None:
        return None, None, configuration_failure

    initial, validation_error = _read_initial_interaction(interaction_root)
    if validation_error is not None:
        return None, None, _failure("invalid_interaction", validation_error)
    if initial is None:
        raise AssertionError("Validated Phase 3.1 and Phase 3.2 records are required.")

    existing_record = _existing_phase_3_3_record(interaction_root)
    if existing_record is not None:
        return None, None, _failure(
            "interaction_exists",
            f"Phase 3.3 record already exists: {existing_record}.",
        )

    try:
        context_refs = approved_context_refs()
    except (OSError, TypeError, ValueError) as exc:
        return None, None, _failure(
            "invalid_interaction",
            f"Approved context refs could not be read: {type(exc).__name__}: {exc}",
        )

    settings = {
        "max_pa_turns": max_pa_turns,
        "live_observation_timeout_sec": float(live_observation_timeout_sec),
    }
    try:
        _write_json_exclusive(interaction_root / _SETTINGS_PATH, settings)
    except FileExistsError:
        return None, None, _failure(
            "interaction_exists",
            "Phase 3.3 pa_context_settings.json already exists.",
        )
    except (OSError, TypeError, ValueError) as exc:
        return None, None, _failure(
            "invalid_interaction",
            f"Phase 3.3 settings write failed: {type(exc).__name__}: {exc}",
        )
    return initial, context_refs, None


def _configuration_failure(
    *,
    max_pa_turns: object,
    live_observation_timeout_sec: object,
) -> dict[str, object] | None:
    if (
        not isinstance(max_pa_turns, int)
        or isinstance(max_pa_turns, bool)
        or max_pa_turns < 2
    ):
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


async def _request_pa_assessment(
    product_agent: ProductAgentContextRuntime,
    *,
    turn_path: Path,
    turn_number: int,
    product_requirement: str,
    pa_input: dict[str, object],
    context_refs: tuple[str, ...],
    served_static_refs: set[str],
) -> tuple[dict[str, object] | None, dict[str, object] | None]:
    try:
        pa_output = await product_agent.ask_llm_structured(
            str(pa_input["prompt"]),
            response_format=pa_input["response_format"],
        )
    except Exception as exc:
        logger.exception("Phase 3.3 ProductAgent structured call failed.")
        failure = _failure(
            "pa_call_failed",
            f"ProductAgent ask_llm_structured failed: {type(exc).__name__}: {exc}",
        )
        return None, _write_turn_or_failure(
            turn_path,
            turn_number=turn_number,
            product_requirement=product_requirement,
            pa_input=pa_input,
            pa_output=None,
            failure=failure,
        )

    validation_error = _assessment_validation_error(
        pa_output,
        context_refs=context_refs,
        served_static_refs=served_static_refs,
    )
    if validation_error is not None:
        failure = _failure("invalid_pa_response", validation_error)
        return None, _write_turn_or_failure(
            turn_path,
            turn_number=turn_number,
            product_requirement=product_requirement,
            pa_input=pa_input,
            pa_output=pa_output,
            failure=failure,
        )
    if not isinstance(pa_output, dict):
        raise AssertionError("Validated Phase 3.3 output must be a mapping.")
    return pa_output, None


def _is_terminal_pa_output(pa_output: dict[str, object]) -> bool:
    if pa_output["context understanding complete"] is True:
        return True
    needed_context = pa_output["needed_context"]
    if not isinstance(needed_context, dict):
        raise AssertionError("Validated needed_context must be a mapping.")
    return isinstance(needed_context["clarification_question"], str)


def _read_initial_interaction(
    interaction_root: Path,
) -> tuple[dict[str, object] | None, str | None]:
    product_requirement, needed_context, validation_error = (
        context_serving._read_phase_3_1_records(interaction_root)
    )
    if validation_error is not None:
        return None, validation_error
    if product_requirement is None or needed_context is None:
        return None, "Phase 3.1 records are incomplete."
    if needed_context["clarification_question"] is not None:
        return None, "Phase 3.1 requested clarification instead of retrievable context."

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
        or isinstance(retrieval_record["retrieval"], bool)
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
        "PA_output": turn_record["PA_output"],
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


def _assessment_prompt(
    *,
    product_requirement: str,
    served_contexts: list[dict[str, object]],
    recorded_decisions: list[object],
    remaining_context_refs: tuple[str, ...],
    turn_number: int,
    max_pa_turns: int,
) -> str:
    pa_context = {
        "product_requirement": product_requirement,
        "turn": turn_number,
        "max_pa_turns": max_pa_turns,
        "served_contexts": served_contexts,
        "recorded_PA_decisions": recorded_decisions,
        "retrieval_errors": [],
        "remaining_approved_context_refs": list(remaining_context_refs),
        "request_live_observation_available": True,
        "downstream_contract": {
            "Phase 4": (
                "PA grounds target_feature, target pose, insertion axis, and "
                "tolerances from the collected product evidence."
            ),
            "Phase 5": "PA creates the robot-independent assembly plan.",
            "Phase 6": (
                "PA sends grounded assembly tasks and required outcomes to RA."
            ),
            "Phase 7": (
                "RA retrieves fresh robot state and its resource-owned primitive "
                "catalog."
            ),
        },
        "permitted_response_shapes": {
            "approved context_ref": {
                "needed_context": {
                    "context_ref": "<one remaining exact context_ref>",
                    "request_live_observation": False,
                    "clarification_question": None,
                },
                "context understanding complete": False,
            },
            "fresh live RGB-D observation": {
                "needed_context": {
                    "context_ref": None,
                    "request_live_observation": True,
                    "clarification_question": None,
                },
                "context understanding complete": False,
            },
            "user clarification": {
                "needed_context": {
                    "context_ref": None,
                    "request_live_observation": False,
                    "clarification_question": "<one focused non-empty question>",
                },
                "context understanding complete": False,
            },
            "context understanding complete": {
                "needed_context": None,
                "context understanding complete": True,
            },
        },
    }
    return (
        "Assess the accumulated Spec2Primitives product context against the exact "
        "product_requirement. Decide whether one more permitted context item is "
        "needed, one focused user clarification is required, or context understanding "
        "is complete for the Phase 4 handoff. Sufficient context must cover the "
        "requested component, its assembly destination and receiving feature, and "
        "the current arrangement when relevant. Do not declare completion while a "
        "required ambiguity, contradiction, context request, or failed retrieval "
        "remains. Remaining metric grounding belongs to Phase 4. PA-to-RA messaging "
        "belongs to Phase 6, and robot state plus the resource-owned primitive catalog "
        "belong to RA in Phase 7. Do not contact RA, retrieve the primitive catalog, "
        "ground metric geometry, create an assembly plan, author primitive_steps, or "
        "execute robot behavior. Choose exactly one permitted response shape. Return "
        "only the structured decision, not hidden reasoning.\n\nPA input:\n"
        f"{json.dumps(pa_context, indent=2, ensure_ascii=False, allow_nan=False)}"
    )


def _assessment_response_format(
    remaining_context_refs: tuple[str, ...],
) -> dict[str, Any]:
    return {
        "name": "spec2primitives_context_assessment",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["needed_context", "context understanding complete"],
            "properties": {
                "needed_context": {
                    "anyOf": [
                        {"type": "null"},
                        {
                            "type": "object",
                            "additionalProperties": False,
                            "required": [
                                "context_ref",
                                "request_live_observation",
                                "clarification_question",
                            ],
                            "properties": {
                                "context_ref": {
                                    "enum": [None, *remaining_context_refs]
                                },
                                "request_live_observation": {"type": "boolean"},
                                "clarification_question": {
                                    "anyOf": [
                                        {"type": "string"},
                                        {"type": "null"},
                                    ]
                                },
                            },
                        },
                    ]
                },
                "context understanding complete": {"type": "boolean"},
            },
        },
    }


def _assessment_validation_error(
    pa_output: object,
    *,
    context_refs: tuple[str, ...],
    served_static_refs: set[str],
) -> str | None:
    if not isinstance(pa_output, dict) or set(pa_output) != _REASSESSMENT_KEYS:
        return "PA output must contain only needed_context and context understanding complete."
    complete = pa_output["context understanding complete"]
    needed_context = pa_output["needed_context"]
    if not isinstance(complete, bool):
        return "context understanding complete must be a boolean."
    if complete is True:
        if needed_context is not None:
            return "Completed context understanding cannot include needed_context."
        return None
    if not isinstance(needed_context, dict):
        return "Incomplete context understanding must include needed_context."
    validation_error = context_serving._needed_context_validation_error(
        needed_context,
        context_refs,
        phase_label="Phase 3.3",
    )
    if validation_error is not None:
        return validation_error
    context_ref = needed_context["context_ref"]
    if isinstance(context_ref, str) and context_ref in served_static_refs:
        return f"Phase 3.3 context_ref was already served: {context_ref}."
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
        if not isinstance(observation_ref, str) or not observation_ref.startswith(
            "observation_"
        ):
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
    for pattern in ("turn_*.json", "retrieval_*.json"):
        for path in sorted(record_root.glob(pattern)):
            if path.name not in {"turn_0001.json", "retrieval_0001.json"}:
                return str(path.relative_to(interaction_root))
    return None


def _write_turn_or_failure(
    turn_path: Path,
    *,
    turn_number: int,
    product_requirement: str,
    pa_input: dict[str, object],
    pa_output: object,
    failure: dict[str, object],
) -> dict[str, object]:
    write_failure = _write_turn_record(
        turn_path,
        turn_number=turn_number,
        product_requirement=product_requirement,
        pa_input=pa_input,
        pa_output=pa_output,
        failure=failure,
    )
    return failure if write_failure is None else write_failure


def _write_turn_record(
    turn_path: Path,
    *,
    turn_number: int,
    product_requirement: str,
    pa_input: dict[str, object],
    pa_output: object,
    failure: dict[str, object] | None,
) -> dict[str, object] | None:
    try:
        _write_json_exclusive(
            turn_path,
            {
                "turn": turn_number,
                "product_requirement": product_requirement,
                "PA_input": pa_input,
                "PA_output": pa_output,
                "failure": None if failure is None else failure["failure"],
            },
        )
    except FileExistsError:
        return _failure(
            "interaction_exists",
            f"Phase 3.3 turn_{turn_number:04d} already exists.",
        )
    except (OSError, TypeError, ValueError) as exc:
        return _failure(
            "invalid_interaction",
            f"turn_{turn_number:04d} write failed: {type(exc).__name__}: {exc}",
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
