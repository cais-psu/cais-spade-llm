"""Convert one accepted camera-frame CAD pose into a requested robot frame."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from cais_spade_llm.spec2primitives.tools.rgb_d_cad_grounding.pose_estimation import (
    _load_pose_inputs,
)
from cais_spade_llm.spec2primitives.tools.rgb_d_cad_grounding.size_correspondence import (
    _GROUNDING_ROOT,
    _PRODUCER,
    CADSizeAssociationError,
    _load_json_record,
    _relative_ref,
    _sha256_path,
    _validate_hashed_ref,
    _validate_positive_integer,
    _write_json,
)

_CALIBRATION_RECORD_KEYS = {
    "schema_version",
    "record_type",
    "producer",
    "calibration_number",
    "calibration_id",
    "method",
    "source_frame",
    "target_frame",
    "coordinate_convention",
    "stored_units",
    "target_from_camera_transform",
    "translation_m",
    "rotation_matrix",
    "quaternion_xyzw",
    "validity",
    "provenance",
    "payload_sha256",
}
_POSE_RECORD_KEYS = {
    "schema_version",
    "record_type",
    "producer",
    "pose_number",
    "method",
    "parameters",
    "source_correspondence",
    "CAD",
    "segmentation",
    "ranked_pose_hypotheses",
    "qualified_pose_hypotheses",
    "selected_candidate",
    "CAD_correspondence",
    "location",
    "pose",
    "coordinate_frame",
    "cross_camera_fusion",
    "robot_frame_conversion",
}
_PROVENANCE_KEYS = {"source", "source_sha256"}
_VALIDITY_KEYS = {"valid_from_ns", "valid_until_ns"}
_ACCEPTED_CANDIDATE_KEYS = {
    "camera_id",
    "role",
    "frame",
    "candidate_id",
    "point_count",
    "CAD_origin_translation_m",
    "rotation_matrix",
    "quaternion_xyzw",
    "camera_from_CAD_transform",
    "registration",
}
_CANDIDATE_IDENTITY_KEYS = {
    "camera_id",
    "role",
    "frame",
    "candidate_id",
    "point_count",
}
_COORDINATE_CONVENTION = "target_from_source_left_multiplication"
_CALIBRATION_METHOD = "injected_approved_camera_to_robot_calibration"
_CONVERSION_METHOD = "homogeneous_transform_composition"
_TRANSFORM_ATOL = 1e-8


class CameraToRobotCalibrationError(ValueError):
    """Raised when an injected camera-to-robot calibration is invalid."""


class RobotFrameConversionError(ValueError):
    """Raised when a camera-frame CAD pose cannot be converted safely."""


@dataclass(frozen=True)
class CameraToRobotCalibrationResult:
    """Return one persisted camera-to-robot calibration record."""

    record_path: Path
    calibration_id: str
    source_frame: str
    target_frame: str
    record: Mapping[str, object]


@dataclass(frozen=True)
class RobotFramePoseResult:
    """Return one persisted robot-frame CAD pose result."""

    record_path: Path
    CAD_correspondence: str
    location: str
    pose: str
    robot_frame_conversion: str
    target_frame: str
    robot_frame_pose: Mapping[str, object] | None
    record: Mapping[str, object]


@dataclass(frozen=True)
class _ValidatedPose:
    path: Path
    record: Mapping[str, object]
    observation_timestamp_ns: int
    source_frame: str | None
    camera_from_CAD: np.ndarray | None


@dataclass(frozen=True)
class _ValidatedCalibration:
    path: Path
    record: Mapping[str, object]
    source_frame: str
    target_frame: str
    target_from_camera: np.ndarray


def record_camera_to_robot_calibration(
    *,
    interaction_root: Path,
    calibration_id: str,
    source_frame: str,
    target_frame: str,
    target_from_camera_transform: Sequence[Sequence[float]],
    valid_from_ns: int,
    valid_until_ns: int | None,
    provenance_source: str,
    provenance_sha256: str,
    calibration_number: int = 1,
) -> CameraToRobotCalibrationResult:
    """Validate and atomically persist one injected extrinsic calibration.

    The caller remains the authority that approves the measured calibration.
    This function only validates its rigid-transform contract and records its
    exact provenance; it does not derive values from simulator state.
    """
    try:
        _validate_positive_integer(calibration_number, "calibration_number")
        calibration_identifier = _nonempty_string(
            calibration_id,
            "calibration_id",
            error_type=CameraToRobotCalibrationError,
        )
        camera_frame = _nonempty_string(
            source_frame,
            "source_frame",
            error_type=CameraToRobotCalibrationError,
        )
        robot_frame = _nonempty_string(
            target_frame,
            "target_frame",
            error_type=CameraToRobotCalibrationError,
        )
        if camera_frame == robot_frame:
            raise CameraToRobotCalibrationError(
                "source_frame and target_frame must be different."
            )
        transformation = _rigid_transform(
            target_from_camera_transform,
            "target_from_camera_transform",
            CameraToRobotCalibrationError,
        )
        validity = _validity(
            valid_from_ns,
            valid_until_ns,
            error_type=CameraToRobotCalibrationError,
        )
        source = _nonempty_string(
            provenance_source,
            "provenance_source",
            error_type=CameraToRobotCalibrationError,
        )
        if not _is_sha256(provenance_sha256):
            raise CameraToRobotCalibrationError("provenance_sha256 is invalid.")
    except CADSizeAssociationError as exc:
        raise CameraToRobotCalibrationError(str(exc)) from exc

    root = Path(interaction_root).resolve()
    destination = root / _GROUNDING_ROOT / f"calibration_{calibration_number:04d}"
    if destination.exists():
        raise CameraToRobotCalibrationError(
            f"Camera-to-robot calibration {calibration_number:04d} already exists."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        temporary_root = Path(
            tempfile.mkdtemp(prefix=".calibration-", dir=destination.parent)
        )
    except OSError as exc:
        raise CameraToRobotCalibrationError(
            "Camera-to-robot calibration temporary directory could not be created."
        ) from exc

    try:
        payload = _calibration_payload(
            calibration_number=calibration_number,
            calibration_id=calibration_identifier,
            source_frame=camera_frame,
            target_frame=robot_frame,
            transformation=transformation,
            validity=validity,
            provenance={"source": source, "source_sha256": provenance_sha256},
        )
        record = dict(payload)
        record["payload_sha256"] = _payload_sha256(payload)
        _write_json(temporary_root / "calibration_record.json", record)
        if destination.exists():
            raise CameraToRobotCalibrationError(
                f"Camera-to-robot calibration {calibration_number:04d} already exists."
            )
        temporary_root.rename(destination)
    except CameraToRobotCalibrationError:
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise
    except (OSError, TypeError, ValueError) as exc:
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise CameraToRobotCalibrationError(
            f"Camera-to-robot calibration persistence failed: {type(exc).__name__}: {exc}"
        ) from exc

    return CameraToRobotCalibrationResult(
        record_path=destination / "calibration_record.json",
        calibration_id=calibration_identifier,
        source_frame=camera_frame,
        target_frame=robot_frame,
        record=record,
    )


def transform_camera_pose_to_robot_frame(
    *,
    interaction_root: Path,
    pose_record_path: Path,
    calibration_record_path: Path,
    target_frame: str,
    conversion_number: int = 1,
) -> RobotFramePoseResult:
    """Convert one camera-frame CAD pose into one exact requested robot frame."""
    try:
        _validate_positive_integer(conversion_number, "conversion_number")
        requested_target = _nonempty_string(target_frame, "target_frame")
        root = Path(interaction_root).resolve()
        pose_input = _load_pose(root, pose_record_path)
        calibration_input = _load_calibration(
            root,
            calibration_record_path,
            observation_timestamp_ns=pose_input.observation_timestamp_ns,
        )
        if calibration_input.target_frame != requested_target:
            raise RobotFrameConversionError(
                "Calibration target_frame does not match the requested target_frame."
            )
        if (
            pose_input.source_frame is not None
            and calibration_input.source_frame != pose_input.source_frame
        ):
            raise RobotFrameConversionError(
                "Calibration source_frame does not match the camera pose frame."
            )
    except CADSizeAssociationError as exc:
        raise RobotFrameConversionError(str(exc)) from exc

    destination = root / _GROUNDING_ROOT / f"robot_pose_{conversion_number:04d}"
    if destination.exists():
        raise RobotFrameConversionError(
            f"Robot-frame pose conversion {conversion_number:04d} already exists."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        temporary_root = Path(
            tempfile.mkdtemp(prefix=".robot-pose-", dir=destination.parent)
        )
    except OSError as exc:
        raise RobotFrameConversionError(
            "Robot-frame pose temporary directory could not be created."
        ) from exc

    try:
        robot_frame_pose = _compose_robot_frame_pose(
            pose_input.camera_from_CAD,
            calibration_input.target_from_camera,
        )
        conversion_state = str(pose_input.record["pose"])
        record = _robot_frame_pose_record(
            root=root,
            conversion_number=conversion_number,
            pose_input=pose_input,
            calibration_input=calibration_input,
            robot_frame_pose=robot_frame_pose,
            conversion_state=conversion_state,
        )
        _write_json(temporary_root / "robot_frame_pose_record.json", record)
        if destination.exists():
            raise RobotFrameConversionError(
                f"Robot-frame pose conversion {conversion_number:04d} already exists."
            )
        temporary_root.rename(destination)
    except RobotFrameConversionError:
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise RobotFrameConversionError(
            f"Robot-frame pose conversion failed: {type(exc).__name__}: {exc}"
        ) from exc

    return RobotFramePoseResult(
        record_path=destination / "robot_frame_pose_record.json",
        CAD_correspondence=str(record["CAD_correspondence"]),
        location=str(record["location"]),
        pose=str(record["pose"]),
        robot_frame_conversion=conversion_state,
        target_frame=requested_target,
        robot_frame_pose=robot_frame_pose,
        record=record,
    )


def _calibration_payload(
    *,
    calibration_number: int,
    calibration_id: str,
    source_frame: str,
    target_frame: str,
    transformation: np.ndarray,
    validity: Mapping[str, object],
    provenance: Mapping[str, object],
) -> dict[str, object]:
    rotation = transformation[:3, :3]
    return {
        "schema_version": 1,
        "record_type": "CameraToRobotCalibrationRecord",
        "producer": _PRODUCER,
        "calibration_number": calibration_number,
        "calibration_id": calibration_id,
        "method": _CALIBRATION_METHOD,
        "source_frame": source_frame,
        "target_frame": target_frame,
        "coordinate_convention": _COORDINATE_CONVENTION,
        "stored_units": "m",
        "target_from_camera_transform": _float_matrix(transformation),
        "translation_m": _float_vector(transformation[:3, 3]),
        "rotation_matrix": _float_matrix(rotation),
        "quaternion_xyzw": _float_vector(Rotation.from_matrix(rotation).as_quat()),
        "validity": dict(validity),
        "provenance": dict(provenance),
    }


def _load_pose(interaction_root: Path, record_path: Path) -> _ValidatedPose:
    path, relative, record = _load_json_record(interaction_root, record_path)
    if path.name != "pose_record.json" or set(record) != _POSE_RECORD_KEYS:
        raise RobotFrameConversionError("CAD pose estimation record fields are invalid.")
    pose_number = record["pose_number"]
    try:
        _validate_positive_integer(pose_number, "pose_number")
    except CADSizeAssociationError as exc:
        raise RobotFrameConversionError(str(exc)) from exc
    expected_ref = _GROUNDING_ROOT / f"pose_{pose_number:04d}" / "pose_record.json"
    if relative != expected_ref:
        raise RobotFrameConversionError("CAD pose estimation record path is invalid.")
    if (
        record["schema_version"] != 1
        or record["record_type"] != "CADPoseEstimationRecord"
        or record["producer"] != _PRODUCER
        or record["method"] != "principal_axis_multistart_point_to_point_ICP"
        or not isinstance(record["parameters"], Mapping)
        or not isinstance(record["ranked_pose_hypotheses"], list)
        or not isinstance(record["qualified_pose_hypotheses"], list)
        or record["cross_camera_fusion"] != "not_evaluated"
        or record["robot_frame_conversion"] != "not_evaluated"
    ):
        raise RobotFrameConversionError("CAD pose estimation identity is invalid.")

    correspondence_path = _validate_hashed_ref(
        interaction_root,
        record["source_correspondence"],
        "CAD pose source correspondence",
    )
    pose_inputs = _load_pose_inputs(interaction_root, correspondence_path)
    if (
        pose_inputs.correspondence_path != correspondence_path
        or record["CAD"] != pose_inputs.correspondence_record["CAD"]
        or record["segmentation"] != pose_inputs.correspondence_record["segmentation"]
    ):
        raise RobotFrameConversionError("CAD pose estimation provenance is inconsistent.")

    source_frame, camera_from_CAD = _validated_pose_decision(record)
    observation_timestamp_ns = _observation_timestamp_ns(interaction_root, record)
    return _ValidatedPose(
        path=path,
        record=record,
        observation_timestamp_ns=observation_timestamp_ns,
        source_frame=source_frame,
        camera_from_CAD=camera_from_CAD,
    )


def _validated_pose_decision(
    record: Mapping[str, object],
) -> tuple[str | None, np.ndarray | None]:
    CAD_correspondence = record["CAD_correspondence"]
    location = record["location"]
    pose = record["pose"]
    selected = record["selected_candidate"]
    coordinate_frame = record["coordinate_frame"]
    valid_states = {
        ("accepted", "available", "accepted"),
        ("accepted", "available", "ambiguous"),
        ("accepted", "ambiguous", "ambiguous"),
        ("ambiguous", "ambiguous", "ambiguous"),
        ("rejected", "unavailable", "rejected"),
    }
    if (CAD_correspondence, location, pose) not in valid_states:
        raise RobotFrameConversionError("CAD pose estimation state is invalid.")

    if pose == "accepted":
        if not isinstance(selected, Mapping) or set(selected) != _ACCEPTED_CANDIDATE_KEYS:
            raise RobotFrameConversionError("Accepted CAD pose candidate is invalid.")
        frame = _nonempty_string(selected["frame"], "selected candidate frame")
        if coordinate_frame != frame:
            raise RobotFrameConversionError("Accepted CAD pose frame is inconsistent.")
        transformation = _rigid_transform(
            selected["camera_from_CAD_transform"],
            "camera_from_CAD_transform",
            RobotFrameConversionError,
        )
        _validate_derived_pose(selected, transformation, "camera-frame CAD pose")
        return frame, transformation

    if selected is None:
        if coordinate_frame is not None:
            raise RobotFrameConversionError("Unselected CAD pose frame must be null.")
        return None, None
    if not isinstance(selected, Mapping) or set(selected) != _CANDIDATE_IDENTITY_KEYS:
        raise RobotFrameConversionError("Ambiguous CAD pose candidate is invalid.")
    frame = _nonempty_string(selected["frame"], "selected candidate frame")
    if coordinate_frame != frame:
        raise RobotFrameConversionError("Ambiguous CAD pose frame is inconsistent.")
    return frame, None


def _observation_timestamp_ns(
    interaction_root: Path,
    pose_record: Mapping[str, object],
) -> int:
    segmentation = pose_record["segmentation"]
    if not isinstance(segmentation, Mapping):
        raise RobotFrameConversionError("CAD pose segmentation provenance is invalid.")
    segmentation_path = _validate_hashed_ref(
        interaction_root,
        segmentation.get("record"),
        "CAD pose segmentation record",
    )
    _, _, segmentation_record = _load_json_record(interaction_root, segmentation_path)
    preprocessing_path = _validate_hashed_ref(
        interaction_root,
        segmentation_record.get("source_record"),
        "Segmentation source preprocessing record",
    )
    _, _, preprocessing_record = _load_json_record(
        interaction_root,
        preprocessing_path,
    )
    manifest_path = _validate_hashed_ref(
        interaction_root,
        preprocessing_record.get("source_manifest"),
        "Observation source manifest",
    )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RobotFrameConversionError("Observation source manifest is invalid.") from exc
    timestamp = manifest.get("captured_at_ns") if isinstance(manifest, Mapping) else None
    if isinstance(timestamp, bool) or not isinstance(timestamp, int) or timestamp < 0:
        raise RobotFrameConversionError("Observation capture timestamp is invalid.")
    if manifest.get("observation_ref") != segmentation.get("observation_ref"):
        raise RobotFrameConversionError("Observation source manifest is inconsistent.")
    return timestamp


def _load_calibration(
    interaction_root: Path,
    record_path: Path,
    *,
    observation_timestamp_ns: int,
) -> _ValidatedCalibration:
    try:
        path, relative, record = _load_json_record(interaction_root, record_path)
    except CADSizeAssociationError as exc:
        raise RobotFrameConversionError(str(exc)) from exc
    if path.name != "calibration_record.json" or set(record) != _CALIBRATION_RECORD_KEYS:
        raise RobotFrameConversionError("Camera-to-robot calibration fields are invalid.")
    calibration_number = record["calibration_number"]
    try:
        _validate_positive_integer(calibration_number, "calibration_number")
    except CADSizeAssociationError as exc:
        raise RobotFrameConversionError(str(exc)) from exc
    expected_ref = (
        _GROUNDING_ROOT
        / f"calibration_{calibration_number:04d}"
        / "calibration_record.json"
    )
    if relative != expected_ref:
        raise RobotFrameConversionError("Camera-to-robot calibration path is invalid.")
    if (
        record["schema_version"] != 1
        or record["record_type"] != "CameraToRobotCalibrationRecord"
        or record["producer"] != _PRODUCER
        or record["method"] != _CALIBRATION_METHOD
        or record["coordinate_convention"] != _COORDINATE_CONVENTION
        or record["stored_units"] != "m"
    ):
        raise RobotFrameConversionError("Camera-to-robot calibration identity is invalid.")
    _nonempty_string(record["calibration_id"], "calibration_id")
    source_frame = _nonempty_string(record["source_frame"], "source_frame")
    target_frame = _nonempty_string(record["target_frame"], "target_frame")
    if source_frame == target_frame:
        raise RobotFrameConversionError(
            "Calibration source_frame and target_frame must be different."
        )
    transformation = _rigid_transform(
        record["target_from_camera_transform"],
        "target_from_camera_transform",
        RobotFrameConversionError,
    )
    _validate_derived_pose(record, transformation, "camera-to-robot calibration")
    validity = record["validity"]
    if not isinstance(validity, Mapping) or set(validity) != _VALIDITY_KEYS:
        raise RobotFrameConversionError("Camera-to-robot calibration validity is invalid.")
    valid_values = _validity(
        validity["valid_from_ns"],
        validity["valid_until_ns"],
        error_type=RobotFrameConversionError,
    )
    if valid_values != validity:
        raise RobotFrameConversionError("Camera-to-robot calibration validity is inconsistent.")
    valid_until = valid_values["valid_until_ns"]
    if observation_timestamp_ns < valid_values["valid_from_ns"] or (
        valid_until is not None and observation_timestamp_ns > valid_until
    ):
        raise RobotFrameConversionError(
            "Camera-to-robot calibration is not valid at the observation timestamp."
        )
    provenance = record["provenance"]
    if not isinstance(provenance, Mapping) or set(provenance) != _PROVENANCE_KEYS:
        raise RobotFrameConversionError("Camera-to-robot calibration provenance is invalid.")
    _nonempty_string(provenance["source"], "calibration provenance source")
    if not _is_sha256(provenance["source_sha256"]):
        raise RobotFrameConversionError(
            "Camera-to-robot calibration provenance hash is invalid."
        )
    payload = {key: value for key, value in record.items() if key != "payload_sha256"}
    if not _is_sha256(record["payload_sha256"]) or record[
        "payload_sha256"
    ] != _payload_sha256(payload):
        raise RobotFrameConversionError(
            "Camera-to-robot calibration payload hash is invalid."
        )
    return _ValidatedCalibration(
        path=path,
        record=record,
        source_frame=source_frame,
        target_frame=target_frame,
        target_from_camera=transformation,
    )


def _compose_robot_frame_pose(
    camera_from_CAD: np.ndarray | None,
    target_from_camera: np.ndarray,
) -> dict[str, object] | None:
    if camera_from_CAD is None:
        return None
    robot_from_CAD = target_from_camera @ camera_from_CAD
    if not _is_rigid_transform(robot_from_CAD):
        raise RobotFrameConversionError("Composed robot-frame transform is invalid.")
    rotation = robot_from_CAD[:3, :3]
    return {
        "CAD_origin_translation_m": _float_vector(robot_from_CAD[:3, 3]),
        "rotation_matrix": _float_matrix(rotation),
        "quaternion_xyzw": _float_vector(Rotation.from_matrix(rotation).as_quat()),
        "robot_from_CAD_transform": _float_matrix(robot_from_CAD),
    }


def _robot_frame_pose_record(
    *,
    root: Path,
    conversion_number: int,
    pose_input: _ValidatedPose,
    calibration_input: _ValidatedCalibration,
    robot_frame_pose: Mapping[str, object] | None,
    conversion_state: str,
) -> dict[str, object]:
    pose_record = pose_input.record
    return {
        "schema_version": 1,
        "record_type": "RobotFramePoseRecord",
        "producer": _PRODUCER,
        "conversion_number": conversion_number,
        "method": _CONVERSION_METHOD,
        "parameters": {"coordinate_convention": _COORDINATE_CONVENTION},
        "source_pose": {
            "ref": _relative_ref(root, pose_input.path),
            "sha256": _sha256_path(pose_input.path),
        },
        "source_calibration": {
            "ref": _relative_ref(root, calibration_input.path),
            "sha256": _sha256_path(calibration_input.path),
            "payload_sha256": calibration_input.record["payload_sha256"],
            "calibration_id": calibration_input.record["calibration_id"],
        },
        "CAD": pose_record["CAD"],
        "segmentation": pose_record["segmentation"],
        "observation_timestamp_ns": pose_input.observation_timestamp_ns,
        "source_frame": pose_input.source_frame,
        "target_frame": calibration_input.target_frame,
        "robot_frame_pose": robot_frame_pose,
        "CAD_correspondence": pose_record["CAD_correspondence"],
        "location": pose_record["location"],
        "pose": pose_record["pose"],
        "cross_camera_fusion": pose_record["cross_camera_fusion"],
        "robot_frame_conversion": conversion_state,
    }


def _validate_derived_pose(
    record: Mapping[str, object],
    transformation: np.ndarray,
    label: str,
) -> None:
    translation = _finite_array(record.get("translation_m", record.get("CAD_origin_translation_m")), (3,), f"{label} translation")
    rotation = _finite_array(record.get("rotation_matrix"), (3, 3), f"{label} rotation")
    quaternion = _finite_array(record.get("quaternion_xyzw"), (4,), f"{label} quaternion")
    if (
        not np.allclose(translation, transformation[:3, 3], rtol=0.0, atol=_TRANSFORM_ATOL)
        or not np.allclose(rotation, transformation[:3, :3], rtol=0.0, atol=_TRANSFORM_ATOL)
    ):
        raise RobotFrameConversionError(f"{label} derived values are inconsistent.")
    expected_quaternion = Rotation.from_matrix(transformation[:3, :3]).as_quat()
    if not (
        np.allclose(quaternion, expected_quaternion, rtol=0.0, atol=_TRANSFORM_ATOL)
        or np.allclose(quaternion, -expected_quaternion, rtol=0.0, atol=_TRANSFORM_ATOL)
    ):
        raise RobotFrameConversionError(f"{label} quaternion is inconsistent.")


def _validity(
    valid_from_ns: object,
    valid_until_ns: object,
    *,
    error_type: type[ValueError],
) -> dict[str, int | None]:
    if (
        isinstance(valid_from_ns, bool)
        or not isinstance(valid_from_ns, int)
        or valid_from_ns < 0
    ):
        raise error_type("valid_from_ns must be a nonnegative integer.")
    if valid_until_ns is not None and (
        isinstance(valid_until_ns, bool)
        or not isinstance(valid_until_ns, int)
        or valid_until_ns < valid_from_ns
    ):
        raise error_type(
            "valid_until_ns must be null or an integer at least valid_from_ns."
        )
    return {"valid_from_ns": valid_from_ns, "valid_until_ns": valid_until_ns}


def _rigid_transform(
    value: object,
    label: str,
    error_type: type[ValueError],
) -> np.ndarray:
    transformation = _finite_array(value, (4, 4), label, error_type=error_type)
    if not _is_rigid_transform(transformation):
        raise error_type(f"{label} must be a rigid homogeneous transform.")
    return transformation


def _is_rigid_transform(value: np.ndarray) -> bool:
    if value.shape != (4, 4) or not np.isfinite(value).all():
        return False
    rotation = value[:3, :3]
    return bool(
        np.allclose(value[3], [0.0, 0.0, 0.0, 1.0], rtol=0.0, atol=_TRANSFORM_ATOL)
        and np.allclose(rotation.T @ rotation, np.eye(3), rtol=0.0, atol=1e-6)
        and math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1e-6)
    )


def _finite_array(
    value: object,
    shape: tuple[int, ...],
    label: str,
    *,
    error_type: type[ValueError] = RobotFrameConversionError,
) -> np.ndarray:
    if _contains_boolean(value):
        raise error_type(f"{label} contains a boolean value.")
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise error_type(f"{label} is invalid.") from exc
    if result.shape != shape or not np.isfinite(result).all():
        raise error_type(f"{label} must contain finite values with shape {shape}.")
    return result


def _contains_boolean(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return True
    if isinstance(value, np.ndarray):
        return value.dtype == np.bool_
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return any(_contains_boolean(item) for item in value)
    return False


def _nonempty_string(
    value: object,
    label: str,
    *,
    error_type: type[ValueError] = RobotFrameConversionError,
) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise error_type(f"{label} must be a nonempty exact string.")
    return value


def _payload_sha256(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _is_sha256(value: object) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _float_vector(values: np.ndarray) -> list[float]:
    return [float(value) for value in values]


def _float_matrix(values: np.ndarray) -> list[list[float]]:
    return [_float_vector(row) for row in values]
