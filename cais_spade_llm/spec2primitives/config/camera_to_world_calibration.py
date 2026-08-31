"""Load operator-approved camera-to-world calibration manifests."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cais_spade_llm.spec2primitives.tools.rgb_d_cad_grounding import (
    CameraToRobotCalibrationError,
    CameraToRobotCalibrationResult,
    record_camera_to_robot_calibration,
)

_ROOT_KEYS = {"schema_version", "calibrations"}
_CALIBRATION_KEYS = {
    "calibration_id",
    "source_frame",
    "target_frame",
    "target_from_camera_transform",
    "valid_from_ns",
    "valid_until_ns",
    "provenance_source",
    "provenance_sha256",
}
_TARGET_FRAME = "world"
_TRANSFORM_ATOL = 1e-8
DEFAULT_GAZEBO_CAMERA_TO_WORLD_CALIBRATION_PATH = Path(__file__).with_name(
    "gazebo_camera_to_world_calibration.json"
)


@dataclass(frozen=True)
class _ApprovedCalibration:
    calibration_id: str
    source_frame: str
    target_frame: str
    target_from_camera_transform: tuple[tuple[float, ...], ...]
    valid_from_ns: int
    valid_until_ns: int | None
    provenance_source: str
    provenance_sha256: str


@dataclass(frozen=True)
class ApprovedCameraToWorldCalibrationRuntime:
    """Materialize only transforms approved in one validated manifest."""

    manifest_path: Path
    calibrations: tuple[_ApprovedCalibration, ...]

    def materialize_camera_to_world_calibration(
        self,
        *,
        interaction_root: Path,
        grounding_record_path: Path,
        source_frame: str,
        target_frame: str,
        calibration_number: int,
    ) -> CameraToRobotCalibrationResult:
        """Persist the exact approved transform for one selected camera frame.

        Args:
            interaction_root: Root of the active interaction.
            grounding_record_path: Accepted correspondence or camera-pose evidence.
            source_frame: Exact selected camera frame.
            target_frame: Required target frame, which must remain ``world``.
            calibration_number: Interaction-local calibration sequence number.

        Returns:
            The persisted calibration record.

        Raises:
            CameraToRobotCalibrationError: If the request has no approved match.
        """
        root = Path(interaction_root).resolve()
        grounding_path = Path(grounding_record_path).resolve()
        try:
            grounding_path.relative_to(root)
            grounding_record = json.loads(grounding_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CameraToRobotCalibrationError(
                "Grounding record is not valid interaction-local evidence."
            ) from exc
        if not isinstance(grounding_record, Mapping):
            raise CameraToRobotCalibrationError(
                "Grounding record cannot support the requested calibration."
            )
        record_type = grounding_record.get("record_type")
        selected = grounding_record.get("selected_candidate")
        record_frame = (
            selected.get("frame")
            if record_type == "CADSizeCorrespondenceRecord"
            and isinstance(selected, Mapping)
            else grounding_record.get("coordinate_frame")
        )
        valid_identity = (
            (record_type == "CADSizeCorrespondenceRecord" and grounding_record.get("schema_version") == 1)
            or (record_type == "CADPoseEstimationRecord" and grounding_record.get("schema_version") == 2)
        )
        if (
            not valid_identity
            or grounding_record.get("CAD_correspondence") != "accepted"
            or grounding_record.get("location") != "available"
            or record_frame != source_frame
        ):
            raise CameraToRobotCalibrationError(
                "Grounding record cannot support the requested calibration."
            )
        if target_frame != _TARGET_FRAME:
            raise CameraToRobotCalibrationError(
                "Camera-to-world calibration target_frame must be world."
            )
        selected = next(
            (
                calibration
                for calibration in self.calibrations
                if calibration.source_frame == source_frame
            ),
            None,
        )
        if selected is None:
            raise CameraToRobotCalibrationError(
                "No approved camera-to-world calibration is configured for "
                f"source_frame {source_frame}."
            )
        return record_camera_to_robot_calibration(
            interaction_root=root,
            calibration_id=selected.calibration_id,
            source_frame=selected.source_frame,
            target_frame=selected.target_frame,
            target_from_camera_transform=selected.target_from_camera_transform,
            valid_from_ns=selected.valid_from_ns,
            valid_until_ns=selected.valid_until_ns,
            provenance_source=selected.provenance_source,
            provenance_sha256=selected.provenance_sha256,
            calibration_number=calibration_number,
        )


def load_camera_to_world_calibration_runtime(
    manifest_path: Path,
) -> ApprovedCameraToWorldCalibrationRuntime:
    """Read and strictly validate one approved calibration manifest.

    Args:
        manifest_path: Operator-supplied versioned calibration manifest.

    Returns:
        An immutable runtime containing the approved frame transforms.

    Raises:
        OSError: If the manifest cannot be read.
        ValueError: If the manifest or any calibration entry is invalid.
    """
    path = Path(manifest_path).resolve()
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("Camera-to-world calibration manifest is malformed.") from exc
    if not isinstance(value, Mapping) or set(value) != _ROOT_KEYS:
        raise ValueError("Camera-to-world calibration manifest fields are invalid.")
    if value["schema_version"] != 1:
        raise ValueError(
            "Camera-to-world calibration manifest schema_version must be 1."
        )
    entries = value["calibrations"]
    if not isinstance(entries, list):
        raise ValueError("Camera-to-world calibrations must be a list.")

    calibrations = tuple(
        _validated_calibration(entry, index=index)
        for index, entry in enumerate(entries)
    )
    source_frames = [entry.source_frame for entry in calibrations]
    if len(source_frames) != len(set(source_frames)):
        raise ValueError(
            "Camera-to-world calibration source_frame values must be unique."
        )
    return ApprovedCameraToWorldCalibrationRuntime(
        manifest_path=path,
        calibrations=calibrations,
    )


def _validated_calibration(
    value: object,
    *,
    index: int,
) -> _ApprovedCalibration:
    label = f"calibrations[{index}]"
    if not isinstance(value, Mapping) or set(value) != _CALIBRATION_KEYS:
        raise ValueError(f"{label} fields are invalid.")
    calibration_id = _nonempty_string(
        value["calibration_id"],
        f"{label}.calibration_id",
    )
    source_frame = _nonempty_string(value["source_frame"], f"{label}.source_frame")
    target_frame = _nonempty_string(value["target_frame"], f"{label}.target_frame")
    if target_frame != _TARGET_FRAME:
        raise ValueError(f"{label}.target_frame must be world.")
    if source_frame == target_frame:
        raise ValueError(f"{label}.source_frame must differ from world.")
    transform = _rigid_transform(
        value["target_from_camera_transform"],
        f"{label}.target_from_camera_transform",
    )
    valid_from_ns, valid_until_ns = _validity_bounds(
        value["valid_from_ns"],
        value["valid_until_ns"],
        label=label,
    )
    provenance_source = _nonempty_string(
        value["provenance_source"],
        f"{label}.provenance_source",
    )
    provenance_sha256 = value["provenance_sha256"]
    if not _is_sha256(provenance_sha256):
        raise ValueError(f"{label}.provenance_sha256 is invalid.")
    return _ApprovedCalibration(
        calibration_id=calibration_id,
        source_frame=source_frame,
        target_frame=target_frame,
        target_from_camera_transform=tuple(
            tuple(float(item) for item in row) for row in transform
        ),
        valid_from_ns=valid_from_ns,
        valid_until_ns=valid_until_ns,
        provenance_source=provenance_source,
        provenance_sha256=str(provenance_sha256),
    )


def _rigid_transform(value: object, label: str) -> np.ndarray:
    if _contains_boolean(value):
        raise ValueError(f"{label} contains a boolean value.")
    try:
        transform = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} is invalid.") from exc
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError(f"{label} must contain finite values with shape (4, 4).")
    rotation = transform[:3, :3]
    if not (
        np.allclose(
            transform[3],
            [0.0, 0.0, 0.0, 1.0],
            rtol=0.0,
            atol=_TRANSFORM_ATOL,
        )
        and np.allclose(
            rotation.T @ rotation,
            np.eye(3),
            rtol=0.0,
            atol=1e-6,
        )
        and math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1e-6)
    ):
        raise ValueError(f"{label} must be a rigid homogeneous transform.")
    return transform


def _validity_bounds(
    valid_from_ns: object,
    valid_until_ns: object,
    *,
    label: str,
) -> tuple[int, int | None]:
    if (
        isinstance(valid_from_ns, bool)
        or not isinstance(valid_from_ns, int)
        or valid_from_ns < 0
    ):
        raise ValueError(f"{label}.valid_from_ns must be a nonnegative integer.")
    if valid_until_ns is not None and (
        isinstance(valid_until_ns, bool)
        or not isinstance(valid_until_ns, int)
        or valid_until_ns < valid_from_ns
    ):
        raise ValueError(
            f"{label}.valid_until_ns must be null or at least valid_from_ns."
        )
    return valid_from_ns, valid_until_ns


def _nonempty_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{label} must be a nonempty exact string.")
    return value


def _is_sha256(value: object) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _contains_boolean(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return True
    if isinstance(value, np.ndarray):
        return value.dtype == np.bool_
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        return any(_contains_boolean(item) for item in value)
    return False
