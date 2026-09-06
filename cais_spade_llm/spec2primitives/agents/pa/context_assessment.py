from __future__ import annotations

"""Resume native ProductAgent grounding after operator clarification."""


import hashlib
import json
import logging
import math
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from cais_spade_llm.spec2primitives.agents.pa.context_grounding import (
    PAOntologyConfig,
    ProductContextGroundingRuntime,
    validated_grounding_producer_descriptors,
)
from cais_spade_llm.spec2primitives.agents.pa.context_interaction import (
    ProductAgentContextRuntime,
    _grounding_progress_validation_error,
    _resource_assignment_output_validation_error,
)
from cais_spade_llm.spec2primitives.agents.pa.grounding_contracts import (
    build_product_context_view,
    persist_pa_context_grounding_completion,
    persist_product_context_view,
)
from cais_spade_llm.spec2primitives.agents.pa.product_context import (
    load_interaction_abox,
)

logger = logging.getLogger(__name__)

_REQUIREMENT_PATH = Path("products/user_requirement/product_requirement.json")
_RECORD_ROOT = Path("interaction_record")
_GROUNDING_VALIDATION_CODES = frozenset(
    {
        "unsupported_process",
        "grounding_budget_exhausted",
        "grounding_no_progress",
        "invalid_target_feature",
        "evidence_reference_invalid",
        "location_evidence_unavailable",
        "no_reachable_resource",
        "reachability_validation_unavailable",
        "invalid_resource_selection",
    }
)


async def submit_pa_clarification_reply(
    product_agent: ProductAgentContextRuntime,
    interaction_root: Path,
    user_reply: str,
    *,
    ontology_config: PAOntologyConfig | None = None,
    grounding_runtime: ProductContextGroundingRuntime | None = None,
) -> dict[str, object]:
    """Record an exact operator reply and resume native grounding."""
    if not isinstance(user_reply, str) or not user_reply.strip():
        return _failure("invalid_clarification", "user_reply must be non-empty.")
    root = Path(interaction_root)
    pending, error = _pending_clarification(root)
    if error is not None or pending is None:
        return _failure(
            "invalid_clarification",
            error or "No pending ProductAgent clarification exists.",
        )
    question_turn = int(pending["turn"])
    record = _clarification_record(
        product_requirement=str(pending["product_requirement"]),
        question_turn=question_turn,
        question=str(pending["question"]),
        action="answered",
        reply=user_reply,
    )
    path = root / _RECORD_ROOT / f"clarification_{question_turn:04d}.json"
    existing, existing_error = _read_clarification(path)
    if existing_error is not None:
        return _failure("invalid_clarification", existing_error)
    if existing is not None:
        if existing.get("action") != "answered" or existing.get("reply") != user_reply:
            return _failure(
                "clarification_exists",
                "The clarification already has a different terminal action.",
            )
    else:
        try:
            _write_json_exclusive(path, record)
        except (FileExistsError, OSError, TypeError, ValueError) as exc:
            return _failure(
                "invalid_interaction",
                f"Clarification write failed: {type(exc).__name__}: {exc}",
            )
    return await continue_pa_context_interaction(
        product_agent,
        root,
        ontology_config=ontology_config,
        grounding_runtime=grounding_runtime,
    )


def cancel_pa_context_interaction(interaction_root: Path) -> dict[str, object]:
    """Record cancellation without invoking ProductAgent."""
    root = Path(interaction_root)
    pending, error = _pending_clarification(root)
    if error is not None or pending is None:
        return _failure(
            "invalid_clarification",
            error or "No pending ProductAgent clarification exists.",
        )
    question_turn = int(pending["turn"])
    record = _clarification_record(
        product_requirement=str(pending["product_requirement"]),
        question_turn=question_turn,
        question=str(pending["question"]),
        action="cancelled",
        reply=None,
    )
    path = root / _RECORD_ROOT / f"clarification_{question_turn:04d}.json"
    existing, existing_error = _read_clarification(path)
    if existing_error is not None:
        return _failure("invalid_clarification", existing_error)
    if existing is not None:
        if existing.get("action") != "cancelled":
            return _failure(
                "clarification_exists",
                "The clarification already has a different terminal action.",
            )
    else:
        try:
            _write_json_exclusive(path, record)
        except (FileExistsError, OSError, TypeError, ValueError) as exc:
            return _failure(
                "invalid_interaction",
                f"Clarification cancellation failed: {type(exc).__name__}: {exc}",
            )
    return {"status": "cancelled", "grounding_status": "incomplete"}


async def continue_pa_context_interaction(
    product_agent: ProductAgentContextRuntime,
    interaction_root: Path,
    *,
    ontology_config: PAOntologyConfig | None = None,
    grounding_runtime: ProductContextGroundingRuntime | None = None,
    max_pa_turns: int = 12,
    live_observation_timeout_sec: float = 5.0,
) -> dict[str, object]:
    """Resume the native investigation after a persisted clarification reply.

    The observation timeout remains in this public signature for compatibility;
    native retrieval enforces the evidence-service timeout internally.
    """
    configuration_error = _configuration_error(
        ontology_config,
        grounding_runtime,
        max_pa_turns,
        live_observation_timeout_sec,
    )
    if configuration_error is not None:
        return _failure("grounding_unavailable", configuration_error)
    if ontology_config is None or grounding_runtime is None:
        raise AssertionError("Validated grounding dependencies are required.")

    root = Path(interaction_root)
    pending, error = _pending_clarification(root)
    if error is not None or pending is None:
        return _failure(
            "invalid_clarification",
            error or "No pending ProductAgent clarification exists.",
        )
    question_turn = int(pending["turn"])
    clarification_path = root / _RECORD_ROOT / f"clarification_{question_turn:04d}.json"
    clarification, clarification_error = _read_clarification(clarification_path)
    if clarification_error is not None or clarification is None:
        return _failure(
            "invalid_clarification",
            clarification_error or "The clarification has not been answered.",
        )
    if clarification.get("action") == "cancelled":
        return {"status": "cancelled", "grounding_status": "incomplete"}
    if clarification.get("action") != "answered":
        return _failure("invalid_clarification", "The clarification is not answered.")

    next_turn = question_turn + 1
    if next_turn > max_pa_turns:
        return _failure("pa_turn_limit_reached", "No clarification turn remains.")
    turn_path = root / _RECORD_ROOT / f"turn_{next_turn:04d}.json"
    if turn_path.exists():
        existing = _read_json(turn_path)
        output = existing.get("PA_output") if isinstance(existing, Mapping) else None
        return (
            dict(output)
            if isinstance(output, Mapping)
            else _failure("invalid_interaction", "Existing ProductAgent turn is invalid.")
        )

    try:
        requirement = _product_requirement(root)
        tbox = ontology_config.load_tbox()
        validated_grounding_producer_descriptors(grounding_runtime)
        abox = load_interaction_abox(root, tbox)
        clarification_history = _answered_clarifications(root, next_turn)
        view = build_product_context_view(
            root,
            abox,
            attempted_evidence=_retrieved_evidence_ids(root),
            assessed_at_ns=time.time_ns(),
        )
        view_path = persist_product_context_view(root, view)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        logger.exception("Native clarification preparation failed.")
        return _failure(
            "invalid_interaction",
            f"Clarification preparation failed: {type(exc).__name__}: {exc}",
        )

    pa_input = {
        "mode": "native_tool_grounding",
        "product_context_ref": view_path.relative_to(root).as_posix(),
        "clarification_refs": [
            (_RECORD_ROOT / f"clarification_{int(item['question_turn']):04d}.json").as_posix()
            for item in clarification_history
        ],
    }
    try:
        result = await grounding_runtime.ground_product_context(
            product_agent,
            interaction_root=root,
            tbox=tbox,
            abox=abox,
            product_context=view.to_record(),
            max_pa_turns=max_pa_turns,
            clarification_history=clarification_history,
        )
        if not isinstance(result, Mapping):
            raise RuntimeError("Native ProductAgent grounding returned an invalid result.")
        output = dict(result)
        output_error = _native_output_error(output)
        if output_error is not None:
            raise RuntimeError(output_error)
    except Exception as exc:
        logger.exception("Native ProductAgent clarification failed.")
        failure = _failure(
            "pa_call_failed",
            f"ProductAgent grounding failed: {type(exc).__name__}: {exc}",
        )
        _write_turn(turn_path, next_turn, requirement, pa_input, None, failure)
        return failure

    _write_turn(turn_path, next_turn, requirement, pa_input, output, None)
    if output["grounding_status"] == "complete":
        try:
            fresh_abox = load_interaction_abox(root, tbox)
            fresh_view = build_product_context_view(
                root,
                fresh_abox,
                attempted_evidence=_retrieved_evidence_ids(root),
                assessed_at_ns=time.time_ns(),
            )
            persist_product_context_view(root, fresh_view)
            _persist_resource_assignment_completion(
                root,
                tbox=tbox,
                product_requirement=requirement,
                completion_turn=next_turn,
                decision_ref=turn_path.relative_to(root).as_posix(),
                product_context=fresh_view,
                output=output,
            )
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            logger.exception("Native clarification completion failed.")
            return _failure(
                "context_completion_failed",
                f"Grounding completion failed: {type(exc).__name__}: {exc}",
            )
    return output


def _pending_clarification(root: Path) -> tuple[dict[str, object] | None, str | None]:
    paths = sorted((root / _RECORD_ROOT).glob("turn_*.json"))
    if not paths:
        return None, "No ProductAgent turn exists."
    try:
        turn = _read_json(paths[-1])
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return None, f"Latest ProductAgent turn could not be read: {exc}."
    output = turn.get("PA_output") if isinstance(turn, Mapping) else None
    if (
        not isinstance(output, Mapping)
        or output.get("grounding_status") != "clarification_required"
    ):
        return None, "The latest ProductAgent turn is not awaiting clarification."
    question = output.get("clarification_question")
    number = turn.get("turn")
    requirement = turn.get("product_requirement")
    if (
        not isinstance(question, str)
        or not question.strip()
        or not isinstance(number, int)
        or isinstance(number, bool)
        or not isinstance(requirement, str)
    ):
        return None, "Pending clarification payload is invalid."
    return {"turn": number, "question": question, "product_requirement": requirement}, None


def _clarification_record(
    *,
    product_requirement: str,
    question_turn: int,
    question: str,
    action: str,
    reply: str | None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "record_type": "PAClarification",
        "product_requirement": product_requirement,
        "question_turn": question_turn,
        "question": question,
        "action": action,
        "reply": reply,
        "recorded_at_ns": time.time_ns(),
    }
    return {**payload, "fingerprint": _fingerprint(payload)}


def _read_clarification(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    if not path.exists():
        return None, None
    try:
        value = _read_json(path)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return None, f"Clarification could not be read: {exc}."
    if not isinstance(value, dict):
        return None, "Clarification must be an object."
    payload = {key: item for key, item in value.items() if key != "fingerprint"}
    if (
        (value.get("record_type") != "PAClarification")
        or (value.get("action") not in {"answered", "cancelled"})
        or (not isinstance(value.get("product_requirement"), str))
        or (not isinstance(value.get("question"), str))
        or (not isinstance(value.get("question_turn"), int))
        or (isinstance(value.get("question_turn"), bool))
        or (value.get("fingerprint") != _fingerprint(payload))
    ):
        return None, "Clarification is invalid or changed."
    reply = value.get("reply")
    if (value["action"] == "answered" and not isinstance(reply, str)) or (
        value["action"] == "cancelled" and reply is not None
    ):
        return None, "Clarification action and reply are inconsistent."
    return value, None


def _answered_clarifications(root: Path, before_turn: int) -> tuple[Mapping[str, object], ...]:
    records: list[Mapping[str, object]] = []
    for path in sorted((root / _RECORD_ROOT).glob("clarification_*.json")):
        record, error = _read_clarification(path)
        if error is not None or record is None:
            raise ValueError(error or f"Clarification is unavailable: {path.name}.")
        if int(record["question_turn"]) >= before_turn:
            raise ValueError("Clarification turn order is invalid.")
        if record["action"] == "answered":
            records.append(record)
    return tuple(records)


def _product_requirement(root: Path) -> str:
    value = _read_json(root / _REQUIREMENT_PATH)
    requirement = value.get("product_requirement") if isinstance(value, Mapping) else None
    if not isinstance(requirement, str) or not requirement.strip():
        raise ValueError("Product requirement record is invalid.")
    return requirement


def _retrieved_evidence_ids(root: Path) -> tuple[str, ...]:
    evidence_ids: list[str] = []
    for path in sorted((root / _RECORD_ROOT).glob("tool_call_*.json")):
        try:
            value = _read_json(path)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        resolved = value.get("resolved_evidence") if isinstance(value, Mapping) else None
        evidence_id = resolved.get("evidence_id") if isinstance(resolved, Mapping) else None
        if (
            isinstance(evidence_id, str)
            and value.get("failure") is None
            and evidence_id not in evidence_ids
        ):
            evidence_ids.append(evidence_id)
    return tuple(evidence_ids)


def _tool_call_refs(root: Path) -> tuple[str, ...]:
    return tuple(
        path.relative_to(root).as_posix()
        for path in sorted((root / _RECORD_ROOT).glob("tool_call_*.json"))
    )


def _native_output_error(value: Mapping[str, object]) -> str | None:
    progress_error = _grounding_progress_validation_error(value.get("grounding_progress"))
    if progress_error is not None:
        return progress_error
    status = value.get("grounding_status")
    if status not in {"complete", "incomplete", "clarification_required"}:
        return "Native ProductAgent output has an invalid grounding_status."
    if status == "clarification_required" and not isinstance(
        value.get("clarification_question"), str
    ):
        return "Native clarification output is invalid."
    if status == "incomplete" and not isinstance(value.get("insufficient_evidence"), str):
        return "Native insufficient-evidence output is invalid."
    validation_code = value.get("grounding_validation_code")
    if status == "incomplete" and validation_code not in _GROUNDING_VALIDATION_CODES:
        return "Native grounding-validation code is invalid."
    if status != "incomplete" and validation_code is not None:
        return "Native grounding-validation code is invalid."
    if value.get("unmet_grounding_obligation") is not None:
        return "Native output cannot contain a grounding obligation."
    if status == "complete":
        if not isinstance(value.get("ontology_projection_ref"), str) or not isinstance(
            value.get("tool_call_refs"),
            list,
        ):
            return "Native completion output is invalid."
        resource_error = _resource_assignment_output_validation_error(value)
        if resource_error is not None:
            return resource_error
    return None


def _persist_resource_assignment_completion(
    root: Path,
    *,
    tbox: object,
    product_requirement: str,
    completion_turn: int,
    decision_ref: str,
    product_context: object,
    output: Mapping[str, object],
) -> None:
    """Persist only after target grounding and resource assignment complete."""
    if output.get("resource_assignment_status") != "complete":
        return
    persist_pa_context_grounding_completion(
        root,
        tbox=tbox,
        product_requirement=product_requirement,
        completion_turn=completion_turn,
        decision_ref=decision_ref,
        product_context=product_context,
        ontology_projection_ref=str(output["ontology_projection_ref"]),
        resource_selection_ref=str(output["resource_selection_ref"]),
        tool_call_refs=tuple(
            item for item in output.get("tool_call_refs", []) if isinstance(item, str)
        ),
    )


def _configuration_error(
    ontology_config: object,
    grounding_runtime: object,
    max_pa_turns: object,
    timeout: object,
) -> str | None:
    if ontology_config is None or grounding_runtime is None:
        return "An authoritative TBox and controlled grounding runtime are required."
    if not isinstance(max_pa_turns, int) or isinstance(max_pa_turns, bool) or max_pa_turns < 2:
        return "max_pa_turns must be an integer of at least 2."
    if (
        not isinstance(timeout, (int, float))
        or isinstance(timeout, bool)
        or not math.isfinite(float(timeout))
        or float(timeout) <= 0
    ):
        return "live_observation_timeout_sec must be finite and positive."
    return None


def _write_turn(
    path: Path,
    turn: int,
    requirement: str,
    pa_input: Mapping[str, object],
    pa_output: Mapping[str, object] | None,
    failure: Mapping[str, object] | None,
) -> None:
    _write_json_exclusive(
        path,
        {
            "turn": turn,
            "product_requirement": requirement,
            "PA_input": dict(pa_input),
            "PA_output": None if pa_output is None else dict(pa_output),
            "failure": None if failure is None else failure["failure"],
        },
    )


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _write_json_exclusive(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8") as stream:
        stream.write(serialized)


def _fingerprint(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _failure(reason: str, message: str) -> dict[str, object]:
    return {"failure": {"reason": reason, "message": message}}
