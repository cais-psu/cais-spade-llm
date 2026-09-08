"""Serve context requested by a recorded Spec2Primitives PA turn."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from cais_spade_llm.spec2primitives.agents.pa.grounding_contracts import (
    GroundingContractError,
    ProductContextView,
)
from cais_spade_llm.spec2primitives.tools.exact_ref_resolver import (
    approved_context_ref_evidence_types,
    approved_context_refs,
    resolve_context_ref,
)
from cais_spade_llm.spec2primitives.tools.observation_context import (
    ObservationContextError,
    read_observation_bundle,
)
from cais_spade_llm.spec2primitives.tools.rgb_d_cad_grounding.gazebo_observation_provider import (
    GazeboObservationProviderError,
    capture_gazebo_observation,
)

logger = logging.getLogger(__name__)

_PRODUCT_REQUIREMENT_PATH = Path("products/user_requirement/product_requirement.json")
_FIRST_TURN_PATH = Path("interaction_record/turn_0001.json")
_RETRIEVAL_RECORD_PATH = Path("interaction_record/retrieval_0001.json")
_SERVED_REFERENCES_ROOT = Path("products/served_references")
_OBSERVATIONS_ROOT = Path("products/observations")
_NEEDED_CONTEXT_KEYS = {
    "context_ref",
    "request_live_observation",
    "clarification_question",
}
_FIRST_TURN_KEYS = {
    "turn",
    "product_requirement",
    "PA_input",
    "PA_output",
    "failure",
}
_FIRST_PA_INPUT_KEYS = {"prompt", "response_format"}
_FIRST_PRODUCTION_PA_INPUT_KEYS = {"product_context_ref", "product_context"}
_LATER_PA_INPUT_KEYS = {"assessment_ref"}
_PROVENANCE_KEYS = {"repository_path", "source_url"}
_REJECTION_KEYS = {"context_ref", "reason", "message"}


def serve_pa_requested_context(
    interaction_root: Path,
    *,
    live_observation_timeout_sec: float = 5.0,
) -> dict[str, object]:
    """Serve the exact context selected by the completed Phase 3.1 turn.

    Args:
        interaction_root: Caller-owned `contexts/<interaction_identifier>/`
            directory containing the Phase 3.1 records.
        live_observation_timeout_sec: Maximum wait for a requested fresh live
            RGB-D observation.

    Returns:
        The exact `served_context` result or a structured failure.
    """
    return _serve_pa_requested_context_turn(
        Path(interaction_root),
        turn_number=1,
        retrieval_number=1,
        observation_ref="observation_0001",
        live_observation_timeout_sec=live_observation_timeout_sec,
        phase_label="Phase 3.2",
    )


def retrieve_pa_evidence(
    interaction_root: Path,
    *,
    product_requirement: str,
    evidence_type: str,
    context_ref: str | None,
    retrieval_number: int,
    live_observation_timeout_sec: float = 5.0,
    audit_directory: Path | None = None,
) -> dict[str, object]:
    """Serve one controller-resolved native ``retrieve`` tool request."""
    root = Path(interaction_root)
    directory = audit_directory or root / "interaction_record"
    if not directory.resolve().is_relative_to(root.resolve()):
        raise ValueError("Retrieval audit directory must remain inside its interaction.")
    retrieval_path = directory / f"retrieval_{retrieval_number:04d}.json"
    if retrieval_path.exists():
        return _failure(
            "interaction_exists",
            f"Native retrieval {retrieval_number:04d} already exists.",
        )
    if evidence_type in {"document", "CAD"}:
        approved = approved_context_ref_evidence_types()
        if context_ref is None or approved.get(context_ref) != evidence_type:
            return _failure(
                "unauthorized_evidence",
                "The resolved static evidence is not currently approved.",
            )
        needed_context = {
            "context_ref": context_ref,
            "request_live_observation": False,
            "clarification_question": None,
        }
        return _serve_static_context(
            root,
            retrieval_path,
            product_requirement,
            needed_context,
            context_ref,
            phase_label="native retrieve",
            reuse_served_reference=True,
        )
    if evidence_type == "observation" and context_ref is None:
        needed_context = {
            "context_ref": None,
            "request_live_observation": True,
            "clarification_question": None,
        }
        return _serve_live_observation(
            root,
            retrieval_path,
            product_requirement,
            needed_context,
            live_observation_timeout_sec,
            observation_ref=f"observation_{retrieval_number:04d}",
            phase_label="native retrieve",
        )
    return _failure(
        "unauthorized_evidence",
        "The resolved evidence type is not available to retrieve.",
    )


def _serve_pa_requested_context_turn(
    interaction_root: Path,
    *,
    turn_number: int,
    retrieval_number: int,
    observation_ref: str,
    live_observation_timeout_sec: float,
    phase_label: str,
    reuse_served_static_reference: bool = False,
) -> dict[str, object]:
    """Serve the exact request from one persisted PA turn."""
    retrieval_record_path = (
        interaction_root / "interaction_record" / f"retrieval_{retrieval_number:04d}.json"
    )
    if retrieval_record_path.exists():
        return _failure(
            "interaction_exists",
            f"{phase_label} retrieval_{retrieval_number:04d} already exists for "
            "this interaction_root.",
        )

    product_requirement, needed_context, validation_error = _read_recorded_pa_request(
        interaction_root,
        turn_number=turn_number,
    )
    if validation_error is not None:
        return _failure("invalid_interaction", validation_error)
    if product_requirement is None or needed_context is None:
        raise AssertionError("Validated PA records must contain both values.")

    context_ref = needed_context["context_ref"]
    if isinstance(context_ref, str):
        return _serve_static_context(
            interaction_root,
            retrieval_record_path,
            product_requirement,
            needed_context,
            context_ref,
            phase_label=phase_label,
            reuse_served_reference=reuse_served_static_reference,
        )
    if needed_context["request_live_observation"] is True:
        return _serve_live_observation(
            interaction_root,
            retrieval_record_path,
            product_requirement,
            needed_context,
            live_observation_timeout_sec,
            observation_ref=observation_ref,
            phase_label=phase_label,
        )

    failure = _failure(
        "context_not_requested",
        "Phase 3.1 requested user clarification instead of retrievable context.",
    )
    record_failure = _write_retrieval_record_or_return_failure(
        retrieval_record_path,
        product_requirement=product_requirement,
        needed_context=needed_context,
        context_request={
            "clarification_question": needed_context["clarification_question"],
        },
        served_context=None,
        retrieval_error=None,
        failure=failure,
        phase_label=phase_label,
    )
    return failure if record_failure is None else record_failure


def _serve_static_context(
    interaction_root: Path,
    retrieval_record_path: Path,
    product_requirement: str,
    needed_context: dict[str, object],
    context_ref: str,
    *,
    phase_label: str,
    reuse_served_reference: bool,
) -> dict[str, object]:
    served_reference_path = interaction_root / _SERVED_REFERENCES_ROOT / f"{context_ref}.json"
    context_request = {"context_ref": context_ref}
    if served_reference_path.exists():
        if reuse_served_reference:
            return _reuse_served_static_context(
                served_reference_path,
                retrieval_record_path,
                product_requirement=product_requirement,
                needed_context=needed_context,
                context_request=context_request,
                context_ref=context_ref,
                phase_label=phase_label,
            )
        return _failure(
            "interaction_exists",
            f"{phase_label} served reference already exists for {context_ref}.",
        )

    try:
        result = resolve_context_ref(context_request)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        logger.exception("Phase 3.2 exact context_ref retrieval failed.")
        return _record_retrieval_failure(
            retrieval_record_path,
            product_requirement=product_requirement,
            needed_context=needed_context,
            context_request=context_request,
            retrieval_error=_retrieval_error(
                context_ref=context_ref,
                observation_ref=None,
                reason="resolver_failed",
                message=f"resolve_context_ref failed: {type(exc).__name__}: {exc}",
            ),
            phase_label=phase_label,
        )

    rejection = _resolver_rejection(result, context_ref)
    if rejection is not None:
        return _record_retrieval_failure(
            retrieval_record_path,
            product_requirement=product_requirement,
            needed_context=needed_context,
            context_request=context_request,
            retrieval_error=rejection,
            phase_label=phase_label,
        )

    validation_error = _static_result_validation_error(result, context_ref)
    if validation_error is not None:
        return _record_retrieval_failure(
            retrieval_record_path,
            product_requirement=product_requirement,
            needed_context=needed_context,
            context_request=context_request,
            retrieval_error=_retrieval_error(
                context_ref=context_ref,
                observation_ref=None,
                reason="invalid_resolver_response",
                message=validation_error,
            ),
            phase_label=phase_label,
        )

    served_context = result["served_context"]
    try:
        _write_json_exclusive(served_reference_path, result)
    except FileExistsError:
        if reuse_served_reference:
            return _reuse_served_static_context(
                served_reference_path,
                retrieval_record_path,
                product_requirement=product_requirement,
                needed_context=needed_context,
                context_request=context_request,
                context_ref=context_ref,
                phase_label=phase_label,
            )
        return _failure(
            "interaction_exists",
            f"{phase_label} served reference already exists for {context_ref}.",
        )
    except (OSError, TypeError, ValueError) as exc:
        return _record_retrieval_failure(
            retrieval_record_path,
            product_requirement=product_requirement,
            needed_context=needed_context,
            context_request=context_request,
            retrieval_error=_retrieval_error(
                context_ref=context_ref,
                observation_ref=None,
                reason="record_write_failed",
                message=f"Served reference write failed: {type(exc).__name__}: {exc}",
            ),
            phase_label=phase_label,
        )

    record_failure = _write_retrieval_record_or_return_failure(
        retrieval_record_path,
        product_requirement=product_requirement,
        needed_context=needed_context,
        context_request=context_request,
        served_context=served_context,
        retrieval_error=None,
        failure=None,
        phase_label=phase_label,
    )
    return result if record_failure is None else record_failure


def _reuse_served_static_context(
    served_reference_path: Path,
    retrieval_record_path: Path,
    *,
    product_requirement: str,
    needed_context: dict[str, object],
    context_request: dict[str, object],
    context_ref: str,
    phase_label: str,
) -> dict[str, object]:
    """Reuse one validated static source without resolving or overwriting it."""
    try:
        result = _read_json(served_reference_path)
    except (OSError, TypeError, ValueError) as exc:
        return _record_retrieval_failure(
            retrieval_record_path,
            product_requirement=product_requirement,
            needed_context=needed_context,
            context_request=context_request,
            retrieval_error=_retrieval_error(
                context_ref=context_ref,
                observation_ref=None,
                reason="invalid_served_reference",
                message=(
                    f"Stored served reference could not be read: "
                    f"{type(exc).__name__}: {exc}"
                ),
            ),
            phase_label=phase_label,
        )

    validation_error = _static_result_validation_error(result, context_ref)
    if validation_error is not None:
        return _record_retrieval_failure(
            retrieval_record_path,
            product_requirement=product_requirement,
            needed_context=needed_context,
            context_request=context_request,
            retrieval_error=_retrieval_error(
                context_ref=context_ref,
                observation_ref=None,
                reason="invalid_served_reference",
                message=f"Stored served reference is invalid: {validation_error}",
            ),
            phase_label=phase_label,
        )
    if not isinstance(result, dict) or not isinstance(result["served_context"], dict):
        raise AssertionError("Validated served static context must be a mapping.")

    record_failure = _write_retrieval_record_or_return_failure(
        retrieval_record_path,
        product_requirement=product_requirement,
        needed_context=needed_context,
        context_request=context_request,
        served_context=result["served_context"],
        retrieval_error=None,
        failure=None,
        phase_label=phase_label,
    )
    return result if record_failure is None else record_failure


def _serve_live_observation(
    interaction_root: Path,
    retrieval_record_path: Path,
    product_requirement: str,
    needed_context: dict[str, object],
    live_observation_timeout_sec: float,
    *,
    observation_ref: str,
    phase_label: str,
) -> dict[str, object]:
    observations_root = interaction_root / _OBSERVATIONS_ROOT
    observation_path = observations_root / observation_ref
    if observation_path.exists():
        return _failure(
            "interaction_exists",
            f"{phase_label} observation already exists for {observation_ref}.",
        )

    context_request = {
        "request_live_observation": True,
        "observation_ref": observation_ref,
        "live_observation_timeout_sec": live_observation_timeout_sec,
    }
    try:
        captured_path = capture_gazebo_observation(
            observations_root,
            observation_ref,
            timeout_sec=live_observation_timeout_sec,
        )
    except GazeboObservationProviderError as exc:
        logger.exception("Phase 3.2 live observation capture failed.")
        return _record_retrieval_failure(
            retrieval_record_path,
            product_requirement=product_requirement,
            needed_context=needed_context,
            context_request=context_request,
            retrieval_error=_retrieval_error(
                context_ref=None,
                observation_ref=observation_ref,
                reason=exc.reason,
                message=str(exc),
            ),
            phase_label=phase_label,
        )
    except ObservationContextError as exc:
        logger.exception("Phase 3.2 live observation storage failed.")
        return _record_retrieval_failure(
            retrieval_record_path,
            product_requirement=product_requirement,
            needed_context=needed_context,
            context_request=context_request,
            retrieval_error=_retrieval_error(
                context_ref=None,
                observation_ref=observation_ref,
                reason="observation_storage_failed",
                message=str(exc),
            ),
            phase_label=phase_label,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        logger.exception("Phase 3.2 live observation request failed.")
        return _record_retrieval_failure(
            retrieval_record_path,
            product_requirement=product_requirement,
            needed_context=needed_context,
            context_request=context_request,
            retrieval_error=_retrieval_error(
                context_ref=None,
                observation_ref=observation_ref,
                reason="capture_failed",
                message=(f"capture_gazebo_observation failed: {type(exc).__name__}: {exc}"),
            ),
            phase_label=phase_label,
        )

    try:
        served_context = _live_served_context(
            interaction_root,
            observation_path,
            Path(captured_path),
            observation_ref,
        )
    except (OSError, TypeError, ValueError, ObservationContextError) as exc:
        return _record_retrieval_failure(
            retrieval_record_path,
            product_requirement=product_requirement,
            needed_context=needed_context,
            context_request=context_request,
            retrieval_error=_retrieval_error(
                context_ref=None,
                observation_ref=observation_ref,
                reason="invalid_observation_bundle",
                message=f"Captured observation is invalid: {type(exc).__name__}: {exc}",
            ),
            phase_label=phase_label,
        )

    result = {"served_context": served_context}
    record_failure = _write_retrieval_record_or_return_failure(
        retrieval_record_path,
        product_requirement=product_requirement,
        needed_context=needed_context,
        context_request=context_request,
        served_context=served_context,
        retrieval_error=None,
        failure=None,
        phase_label=phase_label,
    )
    return result if record_failure is None else record_failure


def _read_phase_3_1_records(
    interaction_root: Path,
) -> tuple[str | None, dict[str, object] | None, str | None]:
    """Read the persisted Phase 3.1 request for the public Phase 3.2 API."""
    return _read_recorded_pa_request(interaction_root, turn_number=1)


def _read_recorded_pa_request(
    interaction_root: Path,
    *,
    turn_number: int,
) -> tuple[str | None, dict[str, object] | None, str | None]:
    requirement_path = interaction_root / _PRODUCT_REQUIREMENT_PATH
    turn_path = interaction_root / "interaction_record" / f"turn_{turn_number:04d}.json"
    try:
        requirement_record = _read_json(requirement_path)
        turn_record = _read_json(turn_path)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None, None, f"Completed turn_{turn_number:04d} records could not be read."

    if not isinstance(requirement_record, dict) or set(requirement_record) != {
        "product_requirement"
    }:
        return None, None, "product_requirement.json fields are invalid."
    product_requirement = requirement_record["product_requirement"]
    if not isinstance(product_requirement, str) or not product_requirement.strip():
        return None, None, "The recorded product_requirement is invalid."

    if not isinstance(turn_record, dict) or set(turn_record) != _FIRST_TURN_KEYS:
        return None, None, f"turn_{turn_number:04d}.json fields are invalid."
    if (
        not isinstance(turn_record["turn"], int)
        or isinstance(turn_record["turn"], bool)
        or turn_record["turn"] != turn_number
    ):
        return None, None, f"turn_{turn_number:04d}.json has an invalid turn number."
    if turn_record["product_requirement"] != product_requirement:
        return None, None, "PA product_requirement records do not match."
    if turn_record["failure"] is not None:
        return None, None, f"turn_{turn_number:04d} did not complete successfully."

    pa_input = turn_record["PA_input"]
    pa_input_error = _pa_input_validation_error(pa_input, turn_number)
    if pa_input_error is not None:
        return None, None, pa_input_error

    pa_output = turn_record["PA_output"]
    expected_output_keys = (
        {"needed_context"}
        if turn_number == 1
        else {
            "needed_context",
            "context understanding complete",
            "grounding_status",
        }
    )
    if not isinstance(pa_output, dict) or set(pa_output) != expected_output_keys:
        return None, None, f"turn_{turn_number:04d} PA_output is invalid."
    if turn_number > 1 and pa_output["context understanding complete"] is not False:
        return None, None, f"turn_{turn_number:04d} does not contain a context request."
    if turn_number > 1 and pa_output["grounding_status"] != "waiting_for_evidence":
        return None, None, f"turn_{turn_number:04d} grounding_status is invalid."
    needed_context = pa_output["needed_context"]
    if not isinstance(needed_context, dict):
        return None, None, f"turn_{turn_number:04d} needed_context is invalid."

    try:
        context_refs = approved_context_refs()
    except (OSError, TypeError, ValueError):
        return None, None, "Approved context refs could not be read."
    validation_error = _needed_context_validation_error(
        needed_context,
        context_refs,
        phase_label="Phase 3.1" if turn_number == 1 else "Phase 3.3",
    )
    if validation_error is not None:
        return None, None, validation_error
    return product_requirement, needed_context, None


def _pa_input_validation_error(pa_input: object, turn_number: int) -> str | None:
    if not isinstance(pa_input, dict):
        return f"turn_{turn_number:04d} PA_input is invalid."
    if turn_number == 1:
        if (
            set(pa_input) == _FIRST_PRODUCTION_PA_INPUT_KEYS
            and pa_input.get("product_context_ref")
            == "products/grounding/product_context/view_0000.json"
            and isinstance(pa_input.get("product_context"), dict)
        ):
            try:
                view = ProductContextView.from_mapping(pa_input["product_context"])
            except GroundingContractError as exc:
                return f"turn_0001 ProductContextView is invalid: {exc}"
            if view.delta_count != 0:
                return "turn_0001 ProductContextView must precede evidence deltas."
            return None
        if (
            set(pa_input) != _FIRST_PA_INPUT_KEYS
            or not isinstance(pa_input["prompt"], str)
            or not isinstance(pa_input["response_format"], dict)
        ):
            return "turn_0001 PA_input is invalid."
        return None
    expected_assessment_ref = f"interaction_record/decision_{turn_number - 1:04d}.json"
    if (
        set(pa_input) != _LATER_PA_INPUT_KEYS
        or pa_input.get("assessment_ref") != expected_assessment_ref
    ):
        return f"turn_{turn_number:04d} assessment_ref is invalid."
    return None


def _needed_context_validation_error(
    needed_context: dict[str, object],
    context_refs: tuple[str, ...],
    *,
    phase_label: str = "Phase 3.1",
) -> str | None:
    if set(needed_context) != _NEEDED_CONTEXT_KEYS:
        return f"{phase_label} needed_context fields are invalid."

    context_ref = needed_context["context_ref"]
    request_live_observation = needed_context["request_live_observation"]
    clarification_question = needed_context["clarification_question"]
    if context_ref is not None and (
        not isinstance(context_ref, str) or context_ref not in context_refs
    ):
        return f"{phase_label} context_ref is not an approved exact ref."
    if not isinstance(request_live_observation, bool):
        return f"{phase_label} request_live_observation is invalid."
    if clarification_question is not None and (
        not isinstance(clarification_question, str) or not clarification_question.strip()
    ):
        return f"{phase_label} clarification_question is invalid."

    active_values = sum(
        (
            context_ref is not None,
            request_live_observation is True,
            clarification_question is not None,
        )
    )
    if active_values != 1:
        return f"{phase_label} needed_context must contain exactly one active decision."
    return None


def _resolver_rejection(
    result: object,
    context_ref: str,
) -> dict[str, object] | None:
    if not isinstance(result, dict) or set(result) != {"rejection"}:
        return None
    rejection = result["rejection"]
    if (
        not isinstance(rejection, dict)
        or set(rejection) != _REJECTION_KEYS
        or rejection["context_ref"] != context_ref
        or not isinstance(rejection["reason"], str)
        or not rejection["reason"]
        or not isinstance(rejection["message"], str)
        or not rejection["message"]
    ):
        return _retrieval_error(
            context_ref=context_ref,
            observation_ref=None,
            reason="invalid_resolver_response",
            message="resolve_context_ref returned a malformed rejection.",
        )
    return {
        "context_ref": context_ref,
        "observation_ref": None,
        "reason": rejection["reason"],
        "message": rejection["message"],
    }


def _static_result_validation_error(
    result: object,
    context_ref: str,
) -> str | None:
    if not isinstance(result, dict) or set(result) != {"served_context"}:
        return "resolve_context_ref must return only served_context or rejection."
    served_context = result["served_context"]
    if not isinstance(served_context, dict):
        return "served_context must be a mapping."
    if served_context.get("context_ref") != context_ref:
        return "served_context context_ref does not match the exact request."
    evidence_type = served_context.get("evidence_type")
    expected_keys = (
        {"context_ref", "evidence_type", "provenance", "document_evidence"}
        if evidence_type == "document"
        else {"context_ref", "evidence_type", "provenance", "CAD_evidence"}
    )
    if evidence_type not in {"document", "CAD"} or set(served_context) != expected_keys:
        return "served_context fields do not match document or CAD evidence."
    provenance = served_context["provenance"]
    if (
        not isinstance(provenance, dict)
        or set(provenance) != _PROVENANCE_KEYS
        or not all(isinstance(value, str) and value for value in provenance.values())
    ):
        return "served_context provenance is invalid."
    evidence_key = "document_evidence" if evidence_type == "document" else "CAD_evidence"
    if not isinstance(served_context[evidence_key], dict):
        return f"served_context {evidence_key} is invalid."
    return None


def _live_served_context(
    interaction_root: Path,
    expected_path: Path,
    captured_path: Path,
    observation_ref: str,
) -> dict[str, object]:
    expected_resolved = expected_path.resolve()
    if captured_path.resolve() != expected_resolved:
        raise ObservationContextError(
            f"Captured observation path does not match {observation_ref}."
        )

    observation_bundle = read_observation_bundle(expected_resolved)
    if (
        observation_bundle.observation_ref != observation_ref
        or observation_bundle.evidence_label != "live"
    ):
        raise ObservationContextError(
            f"Captured observation must be {observation_ref} with evidence label live."
        )

    manifest_path = expected_resolved / "manifest.json"
    manifest = _read_json(manifest_path)
    cameras = manifest.get("cameras") if isinstance(manifest, dict) else None
    if not isinstance(cameras, list):
        raise ObservationContextError("Captured observation cameras are invalid.")

    bundle_relative_path = expected_resolved.relative_to(interaction_root.resolve())
    artifact_references = []
    for camera in cameras:
        if not isinstance(camera, dict):
            raise ObservationContextError("Captured camera manifest is invalid.")
        camera_id = camera.get("camera_id")
        rgb_artifact = camera.get("rgb_artifact")
        depth_artifact = camera.get("depth_artifact")
        if not all(
            isinstance(value, str) and value for value in (camera_id, rgb_artifact, depth_artifact)
        ):
            raise ObservationContextError("Captured artifact reference is invalid.")
        artifact_references.append(
            {
                "camera_id": camera_id,
                "rgb_artifact": str(bundle_relative_path / rgb_artifact),
                "depth_artifact": str(bundle_relative_path / depth_artifact),
            }
        )

    manifest_relative_path = bundle_relative_path / "manifest.json"
    return {
        "context_ref": None,
        "observation_ref": observation_ref,
        "evidence_type": "observation",
        "evidence_label": "live",
        "provenance": {"manifest_path": str(manifest_relative_path)},
        "observation_evidence": {
            "manifest": manifest,
            "artifact_references": artifact_references,
        },
    }


def _record_retrieval_failure(
    retrieval_record_path: Path,
    *,
    product_requirement: str,
    needed_context: dict[str, object],
    context_request: dict[str, object],
    retrieval_error: dict[str, object],
    phase_label: str = "Phase 3.2",
) -> dict[str, object]:
    failure = _failure(
        "retrieval_failed",
        str(retrieval_error["message"]),
        retrieval_error=retrieval_error,
    )
    record_failure = _write_retrieval_record_or_return_failure(
        retrieval_record_path,
        product_requirement=product_requirement,
        needed_context=needed_context,
        context_request=context_request,
        served_context=None,
        retrieval_error=retrieval_error,
        failure=failure,
        phase_label=phase_label,
    )
    return failure if record_failure is None else record_failure


def _write_retrieval_record_or_return_failure(
    retrieval_record_path: Path,
    *,
    product_requirement: str,
    needed_context: dict[str, object],
    context_request: dict[str, object],
    served_context: object,
    retrieval_error: dict[str, object] | None,
    failure: dict[str, object] | None,
    phase_label: str = "Phase 3.2",
) -> dict[str, object] | None:
    retrieval_number = int(retrieval_record_path.stem.removeprefix("retrieval_"))
    try:
        _write_json_exclusive(
            retrieval_record_path,
            {
                "retrieval": retrieval_number,
                "product_requirement": product_requirement,
                "needed_context": needed_context,
                "context_request": context_request,
                "served_context": served_context,
                "retrieval_error": retrieval_error,
                "failure": None if failure is None else failure["failure"],
            },
        )
    except FileExistsError:
        return _failure(
            "interaction_exists",
            f"{phase_label} retrieval_{retrieval_number:04d} already exists for "
            "this interaction_root.",
        )
    except (OSError, TypeError, ValueError) as exc:
        return _failure(
            "retrieval_failed",
            f"{phase_label} retrieval record write failed: {type(exc).__name__}: {exc}",
            retrieval_error=_retrieval_error(
                context_ref=None,
                observation_ref=None,
                reason="record_write_failed",
                message=(
                    f"retrieval_{retrieval_number:04d} write failed: {type(exc).__name__}: {exc}"
                ),
            ),
        )
    return None


def _retrieval_error(
    *,
    context_ref: str | None,
    observation_ref: str | None,
    reason: str,
    message: str,
) -> dict[str, object]:
    return {
        "context_ref": context_ref,
        "observation_ref": observation_ref,
        "reason": reason,
        "message": message,
    }


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _write_json_exclusive(path: Path, value: object) -> None:
    serialized = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(serialized)


def _failure(
    reason: str,
    message: str,
    *,
    retrieval_error: dict[str, object] | None = None,
) -> dict[str, object]:
    result: dict[str, object] = {
        "failure": {
            "reason": reason,
            "message": message,
        }
    }
    if retrieval_error is not None:
        result["retrieval_error"] = retrieval_error
    return result
