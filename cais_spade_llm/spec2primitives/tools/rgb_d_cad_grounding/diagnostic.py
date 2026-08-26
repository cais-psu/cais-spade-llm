"""Run controlled RGB-D preprocessing and automatic segmentation paths."""

from __future__ import annotations

import json
import math
import os
import time
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Protocol

from cais_spade_llm.spec2primitives.tools.exact_ref_resolver import (
    approved_cad_refs,
    resolve_context_ref,
)
from cais_spade_llm.spec2primitives.tools.observation_context import (
    CAMERA_IDS,
    ObservationContextError,
    read_observation_bundle,
)
from cais_spade_llm.spec2primitives.tools.rgb_d_cad_grounding.gazebo_observation_provider import (
    GazeboObservationProviderError,
    capture_gazebo_observation,
)
from cais_spade_llm.spec2primitives.tools.rgb_d_cad_grounding.preprocessor import (
    GeometryPreprocessingError,
    preprocess_served_geometry,
)
from cais_spade_llm.spec2primitives.tools.rgb_d_cad_grounding.segmenter import (
    RGBDSegmentationError,
    segment_preprocessed_observation,
)
from cais_spade_llm.spec2primitives.tools.rgb_d_cad_grounding.size_correspondence import (
    CADSizeAssociationError,
    associate_segmented_candidate_by_size,
)

_DIAGNOSTIC_RECORD = Path("interaction_record/rgb_d_cad_preprocessing_diagnostic.json")
_PIPELINE_RECORD = Path("interaction_record/rgbd_segmentation_pipeline.json")
_LATEST_STATUS_RECORD = Path("rgbd_segmentation_status.json")
_OBSERVATION_REF = "observation_0001"
_AUTOMATIC_CAPTURE_TIMEOUT_SEC = 5.0
_STATUS_KEYS = {
    "schema_version",
    "record_type",
    "status",
    "source_candidate_count",
    "assembly_candidate_count",
    "identity",
    "CAD_correspondence",
    "location",
    "pose",
    "failure",
    "updated_at_ns",
}
_LEGACY_STATUS_KEYS = _STATUS_KEYS - {"location"}


class ObservationCaptureRuntime(Protocol):
    """Narrow injected boundary for one request-scoped RGB-D capture."""

    def capture(
        self,
        observations_root: Path,
        observation_ref: str,
        *,
        timeout_sec: float,
    ) -> Path:
        """Capture and persist one exact observation bundle."""
        ...


class LiveGazeboObservationCaptureRuntime:
    """Delegate only to the existing request-scoped Gazebo capture tool."""

    def capture(
        self,
        observations_root: Path,
        observation_ref: str,
        *,
        timeout_sec: float,
    ) -> Path:
        """Capture one fresh live four-camera observation bundle."""
        return capture_gazebo_observation(
            observations_root,
            observation_ref,
            timeout_sec=timeout_sec,
        )


def run_automatic_rgbd_segmentation_pipeline(
    *,
    contexts_root: Path,
    capture_runtime: ObservationCaptureRuntime,
) -> dict[str, object]:
    """Capture, preprocess, and segment one observation without operator parameters.

    This observation-only supporting path never loads CAD and never performs
    identity, correspondence, pose, context assessment, planning, or execution.
    """
    root = Path(contexts_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    interaction_root = root / f"rgbd_segmentation_{uuid.uuid4().hex}"
    pipeline_record_path = interaction_root / _PIPELINE_RECORD
    _write_latest_status(root, _status_record("running"))

    preprocessing_result = None
    segmentation_result = None
    try:
        capture = getattr(capture_runtime, "capture", None)
        if not callable(capture):
            raise TypeError("capture_runtime must provide a callable capture operation.")
        observations_root = interaction_root / "products/observations"
        captured_path = capture(
            observations_root,
            _OBSERVATION_REF,
            timeout_sec=_AUTOMATIC_CAPTURE_TIMEOUT_SEC,
        )
        served_observation = _served_observation_context(
            interaction_root,
            Path(captured_path),
            _OBSERVATION_REF,
        )
        preprocessing_result = preprocess_served_geometry(
            interaction_root=interaction_root,
            served_context=served_observation,
            operation_number=1,
        )
        segmentation_result = segment_preprocessed_observation(
            interaction_root=interaction_root,
            observation_record_path=preprocessing_result.record_path,
            segmentation_number=1,
        )
    except (
        GazeboObservationProviderError,
        GeometryPreprocessingError,
        ObservationContextError,
        RGBDSegmentationError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        reason = (
            exc.reason
            if isinstance(exc, GazeboObservationProviderError)
            else "automatic_observation_pipeline_rejected"
        )
        result = _automatic_pipeline_result(
            status="failed",
            interaction_root=interaction_root,
            pipeline_record_path=pipeline_record_path,
            preprocessing_result=preprocessing_result,
            segmentation_result=segmentation_result,
            failure={"reason": reason, "message": f"{type(exc).__name__}: {exc}"},
        )
        _write_record_if_absent(pipeline_record_path, result)
        _write_latest_status(root, _status_record("failed", failure=result["failure"]))
        return result

    result = _automatic_pipeline_result(
        status="ready",
        interaction_root=interaction_root,
        pipeline_record_path=pipeline_record_path,
        preprocessing_result=preprocessing_result,
        segmentation_result=segmentation_result,
        failure=None,
    )
    _write_record_if_absent(pipeline_record_path, result)
    _write_latest_status(
        root,
        _status_record(
            "ready",
            source_candidate_count=segmentation_result.source_candidate_count,
            assembly_candidate_count=segmentation_result.assembly_candidate_count,
        ),
    )
    return result


def read_rgbd_segmentation_status(contexts_root: Path) -> dict[str, object]:
    """Read the compact automatic RGB-D status, defaulting safely to idle."""
    path = Path(contexts_root).resolve() / _LATEST_STATUS_RECORD
    if not path.is_file():
        return _status_record("idle", updated_at_ns=0)
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return _status_record(
            "failed",
            failure={
                "reason": "status_record_invalid",
                "message": f"{type(exc).__name__}: {exc}",
            },
        )
    if isinstance(record, dict) and set(record) == _LEGACY_STATUS_KEYS:
        record["location"] = "not_evaluated"
    if not _valid_status_record(record):
        return _status_record(
            "failed",
            failure={
                "reason": "status_record_invalid",
                "message": "The persisted RGB-D status fields are invalid.",
            },
        )
    return record


def run_cad_size_association_pipeline(
    *,
    contexts_root: Path,
    interaction_root: Path,
    segmentation_record_path: Path,
    cad_record_path: Path,
    correspondence_number: int = 1,
) -> dict[str, object]:
    """Run size association and update only the compact read-only UI status."""
    root = Path(contexts_root).resolve()
    interaction = Path(interaction_root).resolve()
    try:
        interaction.relative_to(root)
    except ValueError as exc:
        raise ValueError("interaction_root must be inside contexts_root.") from exc
    current_status = read_rgbd_segmentation_status(root)
    source_count = int(current_status["source_candidate_count"])
    assembly_count = int(current_status["assembly_candidate_count"])
    _write_latest_status(
        root,
        _status_record(
            "running",
            source_candidate_count=source_count,
            assembly_candidate_count=assembly_count,
            CAD_correspondence="running",
            location="running",
        ),
    )
    try:
        association = associate_segmented_candidate_by_size(
            interaction_root=interaction,
            segmentation_record_path=segmentation_record_path,
            cad_record_path=cad_record_path,
            correspondence_number=correspondence_number,
        )
    except (CADSizeAssociationError, OSError, RuntimeError, TypeError, ValueError) as exc:
        failure = {
            "reason": "cad_size_association_rejected",
            "message": f"{type(exc).__name__}: {exc}",
        }
        _write_latest_status(
            root,
            _status_record(
                "failed",
                source_candidate_count=source_count,
                assembly_candidate_count=assembly_count,
                CAD_correspondence="failed",
                location="failed",
                failure=failure,
            ),
        )
        return {
            "status": "failed",
            "interaction_root": str(interaction),
            "correspondence_record_path": None,
            "CAD_correspondence": "failed",
            "location": "failed",
            "pose": "not_evaluated",
            "failure": failure,
        }

    _write_latest_status(
        root,
        _status_record(
            "ready",
            source_candidate_count=source_count,
            assembly_candidate_count=assembly_count,
            CAD_correspondence=association.CAD_correspondence,
            location=association.location,
        ),
    )
    return {
        "status": "ready",
        "interaction_root": str(interaction),
        "correspondence_record_path": str(association.record_path),
        "CAD_correspondence": association.CAD_correspondence,
        "location": association.location,
        "pose": "not_evaluated",
        "failure": None,
    }


def run_rgbd_cad_preprocessing_diagnostic(
    *,
    interaction_root: Path,
    cad_ref: str,
    capture_timeout_sec: float,
    capture_runtime: ObservationCaptureRuntime,
) -> dict[str, object]:
    """Preprocess one approved CAD mesh and one fresh RGB-D observation.

    The diagnostic validates preprocessing records only. It does not initialize
    an ABox, perform correspondence, estimate a pose, or assess completion.
    """
    root = Path(interaction_root).resolve()
    record_path = root / _DIAGNOSTIC_RECORD
    validation_error = _input_validation_error(cad_ref, capture_timeout_sec)
    if validation_error is not None:
        return _failure_result(
            cad_ref,
            record_path,
            "invalid_diagnostic_request",
            validation_error,
        )
    if record_path.exists():
        return _failure_result(
            cad_ref,
            record_path,
            "diagnostic_exists",
            "The preprocessing diagnostic already has a result record.",
        )

    cad_result = None
    observation_result = None
    try:
        resolved_cad = resolve_context_ref({"context_ref": cad_ref})
        served_cad = resolved_cad.get("served_context")
        if not isinstance(served_cad, Mapping):
            raise GeometryPreprocessingError("The exact approved CAD source could not be served.")
        cad_result = preprocess_served_geometry(
            interaction_root=root,
            served_context=served_cad,
            operation_number=1,
        )

        observations_root = root / "products/observations"
        captured_path = capture_runtime.capture(
            observations_root,
            _OBSERVATION_REF,
            timeout_sec=capture_timeout_sec,
        )
        served_observation = _served_observation_context(
            root,
            Path(captured_path),
            _OBSERVATION_REF,
        )
        observation_result = preprocess_served_geometry(
            interaction_root=root,
            served_context=served_observation,
            operation_number=2,
        )
    except (
        GazeboObservationProviderError,
        GeometryPreprocessingError,
        ObservationContextError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        reason = (
            exc.reason
            if isinstance(exc, GazeboObservationProviderError)
            else "preprocessing_rejected"
        )
        result = _failure_result(
            cad_ref,
            record_path,
            reason,
            f"{type(exc).__name__}: {exc}",
            cad_result=cad_result,
            observation_result=observation_result,
        )
        _write_record_if_absent(record_path, result)
        return result

    cad_record = cad_result.record
    observation_record = observation_result.record
    cameras = observation_record["cameras"]
    result = {
        "status": "preprocessed",
        "cad_ref": cad_ref,
        "observation_ref": _OBSERVATION_REF,
        "capture_timeout_sec": float(capture_timeout_sec),
        "CAD": {
            "triangle_count": cad_record["triangle_count"],
            "bounds_m": cad_record["bounds_m"],
            "record_path": str(cad_result.record_path),
            "artifact_paths": [str(path) for path in cad_result.artifact_paths],
            "delta": cad_result.delta,
        },
        "RGB_D": {
            "cameras": [
                {
                    "camera_id": camera["camera_id"],
                    "frame": camera["frame"],
                    "point_count": camera["point_count"],
                    "depth_range_m": camera["depth_range_m"],
                }
                for camera in cameras
            ],
            "record_path": str(observation_result.record_path),
            "artifact_paths": [str(path) for path in observation_result.artifact_paths],
            "delta": observation_result.delta,
        },
        "correspondence": "not_evaluated",
        "pose": "not_evaluated",
        "diagnostic_record_path": str(record_path),
        "failure": None,
    }
    _write_record_if_absent(record_path, result)
    return result


def _served_observation_context(
    interaction_root: Path,
    captured_path: Path,
    observation_ref: str,
) -> dict[str, object]:
    expected_path = (interaction_root / "products/observations" / observation_ref).resolve()
    if captured_path.resolve() != expected_path:
        raise ObservationContextError(
            "Captured observation path does not match the diagnostic request."
        )
    bundle = read_observation_bundle(expected_path)
    if bundle.observation_ref != observation_ref or bundle.evidence_label != "live":
        raise ObservationContextError("Diagnostic observation must be a fresh live bundle.")
    manifest_path = expected_path / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ObservationContextError("Diagnostic observation manifest could not be read.") from exc
    cameras = manifest.get("cameras") if isinstance(manifest, dict) else None
    if not isinstance(cameras, list):
        raise ObservationContextError("Diagnostic observation cameras are invalid.")
    bundle_ref = expected_path.relative_to(interaction_root)
    artifact_references = []
    for camera_id in CAMERA_IDS:
        camera = next(
            (
                item
                for item in cameras
                if isinstance(item, Mapping) and item.get("camera_id") == camera_id
            ),
            None,
        )
        if not isinstance(camera, Mapping):
            raise ObservationContextError("Diagnostic observation camera manifest is incomplete.")
        rgb_artifact = camera.get("rgb_artifact")
        depth_artifact = camera.get("depth_artifact")
        if not all(isinstance(value, str) and value for value in (rgb_artifact, depth_artifact)):
            raise ObservationContextError("Diagnostic observation artifact reference is invalid.")
        artifact_references.append(
            {
                "camera_id": camera_id,
                "rgb_artifact": str(bundle_ref / str(rgb_artifact)),
                "depth_artifact": str(bundle_ref / str(depth_artifact)),
            }
        )
    return {
        "context_ref": None,
        "observation_ref": observation_ref,
        "evidence_type": "observation",
        "evidence_label": "live",
        "provenance": {"manifest_path": str(bundle_ref / "manifest.json")},
        "observation_evidence": {
            "manifest": manifest,
            "artifact_references": artifact_references,
        },
    }


def _input_validation_error(cad_ref: object, timeout_sec: object) -> str | None:
    try:
        cad_refs = approved_cad_refs()
    except (OSError, TypeError, ValueError):
        return "Approved CAD refs could not be loaded."
    if not isinstance(cad_ref, str) or cad_ref not in cad_refs:
        return "cad_ref must be one exact approved CAD ref."
    if (
        isinstance(timeout_sec, bool)
        or not isinstance(timeout_sec, (int, float))
        or not math.isfinite(float(timeout_sec))
        or not 1.0 <= float(timeout_sec) <= 60.0
    ):
        return "capture_timeout_sec must be between 1 and 60 seconds."
    return None


def _failure_result(
    cad_ref: object,
    record_path: Path,
    reason: str,
    message: str,
    *,
    cad_result: object = None,
    observation_result: object = None,
) -> dict[str, object]:
    return {
        "status": "rejected",
        "cad_ref": cad_ref,
        "observation_ref": _OBSERVATION_REF,
        "CAD_record_path": (str(cad_result.record_path) if cad_result is not None else None),
        "RGB_D_record_path": (
            str(observation_result.record_path) if observation_result is not None else None
        ),
        "correspondence": "not_evaluated",
        "pose": "not_evaluated",
        "diagnostic_record_path": str(record_path),
        "failure": {"reason": reason, "message": message},
    }


def _automatic_pipeline_result(
    *,
    status: str,
    interaction_root: Path,
    pipeline_record_path: Path,
    preprocessing_result: object,
    segmentation_result: object,
    failure: object,
) -> dict[str, object]:
    source_candidate_count = (
        segmentation_result.source_candidate_count
        if segmentation_result is not None
        else 0
    )
    assembly_candidate_count = (
        segmentation_result.assembly_candidate_count
        if segmentation_result is not None
        else 0
    )
    return {
        "status": status,
        "observation_ref": _OBSERVATION_REF,
        "interaction_root": str(interaction_root),
        "preprocessing_record_path": (
            str(preprocessing_result.record_path)
            if preprocessing_result is not None
            else None
        ),
        "segmentation_record_path": (
            str(segmentation_result.record_path)
            if segmentation_result is not None
            else None
        ),
        "source_candidate_count": source_candidate_count,
        "assembly_candidate_count": assembly_candidate_count,
        "identity": "not_evaluated",
        "CAD_correspondence": "not_evaluated",
        "pose": "not_evaluated",
        "pipeline_record_path": str(pipeline_record_path),
        "failure": failure,
    }


def _status_record(
    status: str,
    *,
    source_candidate_count: int = 0,
    assembly_candidate_count: int = 0,
    CAD_correspondence: str = "not_evaluated",
    location: str = "not_evaluated",
    failure: object = None,
    updated_at_ns: int | None = None,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "record_type": "RGBDSegmentationStatus",
        "status": status,
        "source_candidate_count": source_candidate_count,
        "assembly_candidate_count": assembly_candidate_count,
        "identity": "not_evaluated",
        "CAD_correspondence": CAD_correspondence,
        "location": location,
        "pose": "not_evaluated",
        "failure": failure,
        "updated_at_ns": time.time_ns() if updated_at_ns is None else updated_at_ns,
    }


def _valid_status_record(value: object) -> bool:
    if not isinstance(value, dict) or set(value) != _STATUS_KEYS:
        return False
    status = value["status"]
    correspondence_states = {
        "not_evaluated",
        "running",
        "accepted",
        "ambiguous",
        "rejected",
        "failed",
    }
    location_states = {
        "not_evaluated",
        "running",
        "available",
        "ambiguous",
        "unavailable",
        "failed",
    }
    if (
        value["schema_version"] != 1
        or value["record_type"] != "RGBDSegmentationStatus"
        or not isinstance(status, str)
        or status not in {"idle", "running", "ready", "failed"}
        or value["identity"] != "not_evaluated"
        or not isinstance(value["CAD_correspondence"], str)
        or value["CAD_correspondence"] not in correspondence_states
        or not isinstance(value["location"], str)
        or value["location"] not in location_states
        or value["pose"] != "not_evaluated"
    ):
        return False
    counts = (value["source_candidate_count"], value["assembly_candidate_count"])
    if any(isinstance(count, bool) or not isinstance(count, int) or count < 0 for count in counts):
        return False
    updated_at_ns = value["updated_at_ns"]
    if isinstance(updated_at_ns, bool) or not isinstance(updated_at_ns, int) or updated_at_ns < 0:
        return False
    failure = value["failure"]
    if status == "failed":
        return (
            isinstance(failure, dict)
            and set(failure) == {"reason", "message"}
            and all(isinstance(item, str) and item for item in failure.values())
        )
    return failure is None


def _write_latest_status(contexts_root: Path, record: Mapping[str, object]) -> None:
    if not _valid_status_record(record):
        raise ValueError("RGB-D segmentation status is invalid.")
    path = contexts_root / _LATEST_STATUS_RECORD
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary_path.open("x", encoding="utf-8") as stream:
            json.dump(record, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
        os.replace(temporary_path, path)
    except (OSError, TypeError, ValueError):
        temporary_path.unlink(missing_ok=True)
        raise


def _write_record_if_absent(path: Path, record: Mapping[str, object]) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
