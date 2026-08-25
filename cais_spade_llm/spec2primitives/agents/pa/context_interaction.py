"""Run the first Spec2Primitives PA needed-context decision."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Protocol

from cais_spade_llm.spec2primitives.tools.exact_ref_resolver import (
    approved_context_refs,
)

logger = logging.getLogger(__name__)

_PRODUCT_REQUIREMENT_PATH = Path(
    "products/user_requirement/product_requirement.json"
)
_FIRST_TURN_PATH = Path("interaction_record/turn_0001.json")
_NEEDED_CONTEXT_KEYS = {
    "context_ref",
    "request_live_observation",
    "clarification_question",
}


class ProductAgentContextRuntime(Protocol):
    """Expose the one shared ProductAgent operation used by Phase 3.1."""

    async def ask_llm_structured(
        self,
        prompt: str,
        *,
        response_format: dict[str, Any],
    ) -> dict[str, Any]:
        """Return one structured PA response."""
        ...


async def start_pa_context_interaction(
    product_agent: ProductAgentContextRuntime,
    interaction_root: Path,
    product_requirement: str,
) -> dict[str, object]:
    """Record a product requirement and request PA's first context decision.

    Args:
        product_agent: Shared ProductAgent viewed through the narrow Phase 3.1
            composition protocol.
        interaction_root: Caller-owned `contexts/<interaction_identifier>/`
            directory.
        product_requirement: Exact user-supplied product requirement.

    Returns:
        The validated `needed_context` response or a structured failure.
    """
    if not isinstance(product_requirement, str) or not product_requirement.strip():
        return _failure(
            "invalid_product_requirement",
            "product_requirement must contain non-whitespace text.",
        )

    requirement_path = interaction_root / _PRODUCT_REQUIREMENT_PATH
    turn_path = interaction_root / _FIRST_TURN_PATH
    if requirement_path.exists() or turn_path.exists():
        return _failure(
            "interaction_exists",
            "Phase 3.1 records already exist for this interaction_root.",
        )

    context_refs = approved_context_refs()
    response_format = _needed_context_response_format(context_refs)
    prompt = _needed_context_prompt(product_requirement, context_refs)
    pa_input = {
        "prompt": prompt,
        "response_format": response_format,
    }

    try:
        _write_json_exclusive(
            requirement_path,
            {"product_requirement": product_requirement},
        )
    except FileExistsError:
        return _failure(
            "interaction_exists",
            "Phase 3.1 records already exist for this interaction_root.",
        )

    try:
        pa_output = await product_agent.ask_llm_structured(
            prompt,
            response_format=response_format,
        )
    except Exception as exc:
        logger.exception("Phase 3.1 ProductAgent structured call failed.")
        failure = _failure(
            "pa_call_failed",
            f"ProductAgent ask_llm_structured failed: {type(exc).__name__}: {exc}",
        )
        _write_first_turn(turn_path, product_requirement, pa_input, None, failure)
        return failure

    validation_error = _needed_context_validation_error(pa_output, context_refs)
    if validation_error is not None:
        failure = _failure("invalid_pa_response", validation_error)
        _write_first_turn(
            turn_path,
            product_requirement,
            pa_input,
            pa_output,
            failure,
        )
        return failure

    _write_first_turn(
        turn_path,
        product_requirement,
        pa_input,
        pa_output,
        None,
    )
    return pa_output


def _needed_context_prompt(
    product_requirement: str,
    context_refs: tuple[str, ...],
) -> str:
    pa_context = {
        "product_requirement": product_requirement,
        "approved_context_refs": list(context_refs),
        "request_live_observation_available": True,
        "permitted_request_shapes": {
            "approved context_ref": {
                "context_ref": "<one exact approved_context_ref>",
                "request_live_observation": False,
                "clarification_question": None,
            },
            "fresh live RGB-D observation": {
                "context_ref": None,
                "request_live_observation": True,
                "clarification_question": None,
            },
            "user clarification": {
                "context_ref": None,
                "request_live_observation": False,
                "clarification_question": "<one focused non-empty question>",
            },
        },
    }
    return (
        "Make only the first Spec2Primitives needed_context decision. "
        "No document, CAD, or observation evidence has been served. Retrieve "
        "permitted context before asking for clarification unless the exact "
        "product_requirement itself is ambiguous. Treat user expertise as unknown. "
        "Choose exactly one permitted request shape. Do not return context "
        "understanding complete, grounding, an assembly plan, or primitive_steps.\n\n"
        "PA input:\n"
        f"{json.dumps(pa_context, indent=2, ensure_ascii=False)}"
    )


def _needed_context_response_format(
    context_refs: tuple[str, ...],
) -> dict[str, Any]:
    return {
        "name": "spec2primitives_needed_context",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["needed_context"],
            "properties": {
                "needed_context": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "context_ref",
                        "request_live_observation",
                        "clarification_question",
                    ],
                    "properties": {
                        "context_ref": {
                            "enum": [None, *context_refs],
                        },
                        "request_live_observation": {"type": "boolean"},
                        "clarification_question": {
                            "anyOf": [
                                {"type": "string"},
                                {"type": "null"},
                            ]
                        },
                    },
                }
            },
        },
    }


def _needed_context_validation_error(
    pa_output: object,
    context_refs: tuple[str, ...],
) -> str | None:
    if not isinstance(pa_output, dict) or set(pa_output) != {"needed_context"}:
        return "PA output must contain only needed_context."

    needed_context = pa_output["needed_context"]
    if (
        not isinstance(needed_context, dict)
        or set(needed_context) != _NEEDED_CONTEXT_KEYS
    ):
        return "needed_context fields do not match the Phase 3.1 response shape."

    context_ref = needed_context["context_ref"]
    request_live_observation = needed_context["request_live_observation"]
    clarification_question = needed_context["clarification_question"]

    if context_ref is not None and not isinstance(context_ref, str):
        return "context_ref must be null or an approved exact ref."
    if isinstance(context_ref, str) and context_ref not in context_refs:
        return "context_ref is not an approved exact ref."
    if not isinstance(request_live_observation, bool):
        return "request_live_observation must be a boolean."
    if clarification_question is not None and not isinstance(
        clarification_question, str
    ):
        return "clarification_question must be null or a string."
    if isinstance(clarification_question, str) and not clarification_question.strip():
        return "clarification_question must contain non-whitespace text."

    active_values = sum(
        (
            context_ref is not None,
            request_live_observation is True,
            clarification_question is not None,
        )
    )
    if active_values != 1:
        return "needed_context must contain exactly one active decision."
    return None


def _write_first_turn(
    turn_path: Path,
    product_requirement: str,
    pa_input: dict[str, Any],
    pa_output: object,
    failure: dict[str, object] | None,
) -> None:
    _write_json_exclusive(
        turn_path,
        {
            "turn": 1,
            "product_requirement": product_requirement,
            "PA_input": pa_input,
            "PA_output": pa_output,
            "failure": None if failure is None else failure["failure"],
        },
    )


def _write_json_exclusive(path: Path, value: object) -> None:
    serialized = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(serialized)


def _failure(reason: str, message: str) -> dict[str, object]:
    return {
        "failure": {
            "reason": reason,
            "message": message,
        }
    }
