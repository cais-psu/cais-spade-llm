"""Start one native tool-using Spec2Primitives ProductAgent interaction."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Awaitable, Callable, Mapping
from pathlib import Path
from typing import Any, Protocol

from cais_spade_llm.spec2primitives.agents.pa.context_grounding import (
    PAOntologyConfig,
    ProductContextGroundingRuntime,
    validated_grounding_producer_descriptors,
)
from cais_spade_llm.spec2primitives.agents.pa.grounding_contracts import (
    build_product_context_view,
    persist_pa_context_grounding_completion_v7,
    persist_product_context_view,
)
from cais_spade_llm.spec2primitives.agents.pa.product_context import (
    initialize_interaction_abox,
    load_interaction_abox,
)

logger = logging.getLogger(__name__)

_PRODUCT_REQUIREMENT_PATH = Path("products/user_requirement/product_requirement.json")
_FIRST_TURN_PATH = Path("interaction_record/turn_0001.json")
_GROUNDING_VALIDATION_CODES = frozenset(
    {
        "unsupported_process",
        "invalid_target_feature",
        "evidence_reference_invalid",
        "location_evidence_unavailable",
        "no_reachable_resource",
        "invalid_resource_selection",
    }
)


class ProductAgentContextRuntime(Protocol):
    """Expose the shared ProductAgent's controlled structured-call boundary."""

    async def ask_llm_structured(
        self,
        prompt: str,
        *,
        response_format: dict[str, Any],
        tools: list[dict[str, Any]] | None = None,
        tool_executor: Callable[[str, Mapping[str, object]], Awaitable[Mapping[str, object]]]
        | None = None,
        max_tool_rounds: int = 3,
    ) -> dict[str, Any]:
        """Return one structured PA response after optional controlled tool use."""
        ...


async def start_pa_context_interaction(
    product_agent: ProductAgentContextRuntime,
    interaction_root: Path,
    product_requirement: str,
    *,
    ontology_config: PAOntologyConfig | None = None,
    grounding_runtime: ProductContextGroundingRuntime | None = None,
) -> dict[str, object]:
    """Record the requirement and run one native ProductAgent investigation."""
    if not isinstance(product_requirement, str) or not product_requirement.strip():
        return _failure(
            "invalid_product_requirement",
            "product_requirement must contain non-whitespace text.",
        )
    if ontology_config is None or grounding_runtime is None:
        return _failure(
            "grounding_unavailable",
            "An authoritative TBox and controlled grounding runtime are required.",
        )

    root = Path(interaction_root)
    requirement_path = root / _PRODUCT_REQUIREMENT_PATH
    turn_path = root / _FIRST_TURN_PATH
    if requirement_path.exists() or turn_path.exists():
        return _failure(
            "interaction_exists",
            "ProductAgent records already exist for this interaction_root.",
        )
    try:
        _write_json_exclusive(
            requirement_path,
            {"product_requirement": product_requirement},
        )
        tbox = ontology_config.load_tbox()
        validated_grounding_producer_descriptors(grounding_runtime)
        abox = initialize_interaction_abox(root, product_requirement, tbox)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        logger.exception("Spec2Primitives grounding initialization failed.")
        failure = _failure(
            "grounding_initialization_failed",
            f"Grounding initialization failed: {type(exc).__name__}: {exc}",
        )
        _write_first_turn(turn_path, product_requirement, None, None, failure)
        return failure

    product_context = build_product_context_view(
        root,
        abox,
        attempted_evidence=(),
        assessed_at_ns=time.time_ns(),
    )
    product_context_path = persist_product_context_view(root, product_context)
    pa_input = {
        "mode": "native_tool_grounding",
        "product_context_ref": product_context_path.relative_to(root).as_posix(),
    }
    try:
        assessment = await grounding_runtime.ground_product_context(
            product_agent,
            interaction_root=root,
            tbox=tbox,
            abox=abox,
            product_context=product_context.to_record(),
            max_pa_turns=12,
        )
        if not isinstance(assessment, Mapping):
            raise RuntimeError("Native ProductAgent grounding returned an invalid result.")
        pa_output = dict(assessment)
    except Exception as exc:
        logger.exception("Native ProductAgent grounding failed.")
        failure = _failure(
            "pa_call_failed",
            f"ProductAgent grounding failed: {type(exc).__name__}: {exc}",
            diagnostic=_pa_call_diagnostic(exc),
        )
        _write_first_turn(turn_path, product_requirement, pa_input, None, failure)
        return failure

    validation_error = _native_output_validation_error(pa_output)
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
    _write_first_turn(turn_path, product_requirement, pa_input, pa_output, None)

    if pa_output["grounding_status"] == "complete":
        try:
            fresh_abox = load_interaction_abox(root, tbox)
            fresh_view = build_product_context_view(
                root,
                fresh_abox,
                attempted_evidence=tuple(
                    item for item in pa_output.get("tool_call_refs", []) if isinstance(item, str)
                ),
                assessed_at_ns=time.time_ns(),
            )
            persist_product_context_view(root, fresh_view)
            if pa_output.get("resource_assignment_status") == "complete":
                persist_pa_context_grounding_completion_v7(
                    root,
                    tbox=tbox,
                    product_requirement=product_requirement,
                    completion_turn=1,
                    decision_ref=turn_path.relative_to(root).as_posix(),
                    product_context=fresh_view,
                    ontology_projection_ref=str(pa_output["ontology_projection_ref"]),
                    resource_selection_ref=str(pa_output["resource_selection_ref"]),
                    tool_call_refs=tuple(pa_output["tool_call_refs"]),
                )
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            logger.exception("Native grounding completion failed.")
            return _failure(
                "context_completion_failed",
                f"Grounding completion failed: {type(exc).__name__}: {exc}",
            )
    return pa_output


def _native_output_validation_error(value: Mapping[str, object]) -> str | None:
    status = value.get("grounding_status")
    if status not in {"complete", "incomplete", "clarification_required"}:
        return "Native PA output has an invalid grounding_status."
    if status == "clarification_required" and not isinstance(
        value.get("clarification_question"), str
    ):
        return "Native PA clarification output is invalid."
    if status == "incomplete" and not isinstance(value.get("insufficient_evidence"), str):
        return "Native PA insufficient-evidence output is invalid."
    validation_code = value.get("grounding_validation_code")
    if status == "incomplete" and validation_code not in _GROUNDING_VALIDATION_CODES:
        return "Native PA grounding-validation code is invalid."
    if status != "incomplete" and validation_code is not None:
        return "Native PA grounding-validation code is invalid."
    if value.get("unmet_grounding_obligation") is not None:
        return "Native PA output cannot contain a grounding obligation."
    if status == "complete":
        if not isinstance(value.get("ontology_projection_ref"), str):
            return "Native completion ontology projection is invalid."
        resource_error = _resource_assignment_output_validation_error(value)
        if resource_error is not None:
            return resource_error
        tool_refs = value.get("tool_call_refs")
        if not isinstance(tool_refs, list) or not all(isinstance(item, str) for item in tool_refs):
            return "Native completion tool-call references are invalid."
    return None


def _resource_assignment_output_validation_error(
    value: Mapping[str, object],
) -> str | None:
    """Validate the optional resource-assignment portion of semantic completion."""
    status = value.get("resource_assignment_status")
    if status != "complete":
        return "Native completion resource-assignment status is invalid."
    resource_selection_ref = value.get("resource_selection_ref")
    if not isinstance(resource_selection_ref, str):
        return "Native completion resource selection is invalid."
    if value.get("resource_assignment_validation_code") is not None:
        return "Native completion resource-assignment validation code is invalid."
    return None


def _write_first_turn(
    turn_path: Path,
    product_requirement: str,
    pa_input: dict[str, Any] | None,
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


def _pa_call_diagnostic(error: Exception) -> dict[str, object]:
    """Return safe structured metadata for one failed ProductAgent call."""
    cause = error
    while isinstance(cause.__cause__, Exception):
        cause = cause.__cause__
    body = getattr(cause, "body", None)
    if isinstance(body, dict) and isinstance(body.get("error"), dict):
        body = body["error"]
    body_message = body.get("message") if isinstance(body, dict) else None
    message = body_message if isinstance(body_message, str) else str(cause)

    def metadata(name: str) -> object:
        value = getattr(cause, name, None)
        if value is None and isinstance(body, dict):
            value = body.get(name)
        return value if isinstance(value, (int, str)) and not isinstance(value, bool) else None

    return {
        "stage": "native ProductAgent grounding",
        "exception": type(cause).__name__,
        "status_code": metadata("status_code"),
        "request_id": metadata("request_id"),
        "error_type": metadata("type"),
        "param": metadata("param"),
        "code": metadata("code"),
        "message": message[:2000],
    }


def _failure(
    reason: str,
    message: str,
    *,
    diagnostic: dict[str, object] | None = None,
) -> dict[str, object]:
    failure: dict[str, object] = {"reason": reason, "message": message}
    if diagnostic is not None:
        failure["diagnostic"] = diagnostic
    return {"failure": failure}
