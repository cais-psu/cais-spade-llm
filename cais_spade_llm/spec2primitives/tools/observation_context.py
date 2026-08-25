"""Validate and store offline Spec2Primitives RGB-D observation bundles."""

from __future__ import annotations

import json
import math
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

CAMERA_IDS = (
    "cam_mk3",
    "cam_mk4_1",
    "cam_mk4_2",
    "cam_assembly",
)

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
MAX_RGB_DEPTH_SKEW_NS = 50_000_000
MAX_BUNDLE_SKEW_NS = 250_000_000
EVIDENCE_LABELS = ("fixture", "replay", "live")

_OBSERVATION_REF_PATTERN = re.compile(r"^observation_[A-Za-z0-9_-]+$")
_MANIFEST_KEYS = {
    "observation_ref",
    "evidence_label",
    "captured_at_ns",
    "bundle_skew_ns",
    "cameras",
}
_CAMERA_MANIFEST_KEYS = {
    "camera_id",
    "rgb_artifact",
    "depth_artifact",
    "rgb_encoding",
    "depth_encoding",
    "depth_units",
    "rgb_timestamp_ns",
    "depth_timestamp_ns",
    "timestamp_difference_ns",
    "rgb_frame",
    "depth_frame",
    "depth_registered_to_rgb",
    "valid_depth_fraction",
    "camera_calibration",
}
_CALIBRATION_KEYS = {
    "width",
    "height",
    "frame",
    "distortion_model",
    "D",
    "K",
    "R",
    "P",
    "extrinsics_available",
}


class ObservationContextError(ValueError):
    """Raised when an observation bundle violates the offline contract."""


@dataclass(frozen=True)
class CameraCalibration:
    """Approved intrinsic calibration recorded for one RGB-D camera."""

    width: int
    height: int
    frame: str
    distortion_model: str
    D: tuple[float, ...]
    K: tuple[float, ...]
    R: tuple[float, ...]
    P: tuple[float, ...]
    extrinsics_available: bool = False


@dataclass(frozen=True)
class CameraObservation:
    """Synchronized RGB, metric depth, and calibration for one camera."""

    camera_id: str
    rgb: np.ndarray = field(repr=False, compare=False)
    depth_m: np.ndarray = field(repr=False, compare=False)
    rgb_timestamp_ns: int
    depth_timestamp_ns: int
    rgb_frame: str
    depth_frame: str
    camera_calibration: CameraCalibration
    depth_registered_to_rgb: bool = True
    rgb_encoding: str = "rgb8"
    depth_encoding: str = "32FC1"
    depth_units: str = "m"

    @property
    def timestamp_difference_ns(self) -> int:
        """Return the absolute RGB/depth timestamp difference."""
        return abs(self.rgb_timestamp_ns - self.depth_timestamp_ns)

    @property
    def valid_depth_fraction(self) -> float:
        """Return the fraction of finite positive metric-depth pixels."""
        if not isinstance(self.depth_m, np.ndarray) or self.depth_m.size == 0:
            return 0.0
        valid = np.isfinite(self.depth_m) & (self.depth_m > 0)
        return float(np.count_nonzero(valid) / self.depth_m.size)


@dataclass(frozen=True)
class ObservationBundle:
    """Complete four-camera fixture, replay, or live observation context."""

    observation_ref: str
    evidence_label: str
    captured_at_ns: int
    camera_observations: tuple[CameraObservation, ...]

    @property
    def bundle_skew_ns(self) -> int:
        """Return the difference between the earliest and latest sensor stamp."""
        timestamps = [
            timestamp
            for observation in self.camera_observations
            for timestamp in (
                observation.rgb_timestamp_ns,
                observation.depth_timestamp_ns,
            )
        ]
        return max(timestamps) - min(timestamps) if timestamps else 0


def write_observation_bundle(
    observations_root: Path,
    observation_bundle: ObservationBundle,
) -> Path:
    """Validate and atomically write one observation bundle.

    Args:
        observations_root: Caller-owned directory for observation artifacts.
        observation_bundle: Complete offline observation evidence to store.

    Returns:
        The created observation bundle directory.

    Raises:
        ObservationContextError: If validation or artifact writing fails.
    """
    _validate_bundle(observation_bundle)

    observations_root = Path(observations_root).resolve()
    destination = observations_root / observation_bundle.observation_ref
    if destination.exists():
        raise ObservationContextError(
            f"Observation bundle already exists: {observation_bundle.observation_ref}"
        )

    observations_root.mkdir(parents=True, exist_ok=True)
    try:
        temporary_path = Path(
            tempfile.mkdtemp(
                prefix=f".{observation_bundle.observation_ref}-",
                dir=observations_root,
            )
        )
    except OSError as exc:
        raise ObservationContextError("Observation temporary directory failed.") from exc

    try:
        manifest = _write_bundle_files(temporary_path, observation_bundle)
        (temporary_path / "manifest.json").write_text(
            json.dumps(manifest, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        if destination.exists():
            raise ObservationContextError(
                f"Observation bundle already exists: {observation_bundle.observation_ref}"
            )
        temporary_path.rename(destination)
    except ObservationContextError:
        shutil.rmtree(temporary_path, ignore_errors=True)
        raise
    except (OSError, TypeError, ValueError, cv2.error) as exc:
        shutil.rmtree(temporary_path, ignore_errors=True)
        raise ObservationContextError("Observation bundle write failed.") from exc
    return destination


def read_observation_bundle(bundle_path: Path) -> ObservationBundle:
    """Validate and reload one stored observation bundle.

    Args:
        bundle_path: Directory containing a bundle `manifest.json` and artifacts.

    Returns:
        The reconstructed observation bundle with RGB and metric depth arrays.

    Raises:
        ObservationContextError: If the manifest or artifacts are invalid.
    """
    bundle_path = Path(bundle_path).resolve()
    if not bundle_path.is_dir():
        raise ObservationContextError("Observation bundle directory is missing.")

    try:
        manifest = json.loads(
            (bundle_path / "manifest.json").read_text(encoding="utf-8")
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ObservationContextError("Observation manifest could not be read.") from exc

    try:
        observation_bundle = _bundle_from_manifest(bundle_path, manifest)
        _validate_bundle(observation_bundle)
        _validate_derived_manifest_values(manifest, observation_bundle)
    except ObservationContextError:
        raise
    except (KeyError, OSError, TypeError, ValueError, cv2.error) as exc:
        raise ObservationContextError("Observation bundle is malformed.") from exc
    return observation_bundle


def _validate_bundle(observation_bundle: ObservationBundle) -> None:
    if not isinstance(observation_bundle, ObservationBundle):
        raise ObservationContextError("Expected an ObservationBundle.")
    if not _OBSERVATION_REF_PATTERN.fullmatch(observation_bundle.observation_ref):
        raise ObservationContextError("observation_ref has an unsafe shape.")
    if observation_bundle.evidence_label not in EVIDENCE_LABELS:
        raise ObservationContextError("Unknown observation evidence label.")
    _require_nonnegative_int("captured_at_ns", observation_bundle.captured_at_ns)
    if not isinstance(observation_bundle.camera_observations, tuple):
        raise ObservationContextError("camera_observations must be a tuple.")

    camera_ids = [
        observation.camera_id
        for observation in observation_bundle.camera_observations
        if isinstance(observation, CameraObservation)
    ]
    if len(camera_ids) != len(observation_bundle.camera_observations):
        raise ObservationContextError("Every camera entry must be a CameraObservation.")
    if len(camera_ids) != len(set(camera_ids)):
        raise ObservationContextError("Camera identifiers must not be duplicated.")
    if set(camera_ids) != set(CAMERA_IDS) or len(camera_ids) != len(CAMERA_IDS):
        raise ObservationContextError("All four exact camera identifiers are required.")

    for observation in observation_bundle.camera_observations:
        _validate_camera_observation(observation)
    if observation_bundle.bundle_skew_ns > MAX_BUNDLE_SKEW_NS:
        raise ObservationContextError("Cross-camera bundle skew exceeds 250 ms.")


def _validate_camera_observation(observation: CameraObservation) -> None:
    if observation.camera_id not in CAMERA_IDS:
        raise ObservationContextError("Unknown camera identifier.")
    if observation.rgb_encoding != "rgb8":
        raise ObservationContextError("RGB encoding must be rgb8.")
    if observation.depth_encoding != "32FC1":
        raise ObservationContextError("Depth encoding must be 32FC1.")
    if observation.depth_units != "m":
        raise ObservationContextError("Depth units must be metres using m.")
    if not isinstance(observation.rgb, np.ndarray):
        raise ObservationContextError("RGB evidence must be a NumPy array.")
    if observation.rgb.shape != (IMAGE_HEIGHT, IMAGE_WIDTH, 3):
        raise ObservationContextError("RGB shape must be 480 x 640 x 3.")
    if observation.rgb.dtype != np.uint8:
        raise ObservationContextError("RGB dtype must be uint8.")
    if not isinstance(observation.depth_m, np.ndarray):
        raise ObservationContextError("Depth evidence must be a NumPy array.")
    if observation.depth_m.shape != (IMAGE_HEIGHT, IMAGE_WIDTH):
        raise ObservationContextError("Depth shape must be 480 x 640.")
    if observation.depth_m.dtype != np.float32:
        raise ObservationContextError("Depth dtype must be float32.")
    if observation.valid_depth_fraction <= 0:
        raise ObservationContextError("Depth must contain a finite positive pixel.")
    if observation.depth_registered_to_rgb is not True:
        raise ObservationContextError("Depth must be registered to RGB.")
    if not observation.rgb_frame or not observation.depth_frame:
        raise ObservationContextError("RGB and depth frames must be non-empty.")

    _require_nonnegative_int("rgb_timestamp_ns", observation.rgb_timestamp_ns)
    _require_nonnegative_int("depth_timestamp_ns", observation.depth_timestamp_ns)
    if observation.timestamp_difference_ns > MAX_RGB_DEPTH_SKEW_NS:
        raise ObservationContextError("RGB/depth timestamp difference exceeds 50 ms.")
    _validate_calibration(observation.camera_calibration)


def _validate_calibration(calibration: CameraCalibration) -> None:
    if not isinstance(calibration, CameraCalibration):
        raise ObservationContextError("camera_calibration is required.")
    if calibration.width != IMAGE_WIDTH or calibration.height != IMAGE_HEIGHT:
        raise ObservationContextError("Calibration dimensions must be 640 x 480.")
    if not calibration.frame or not calibration.distortion_model:
        raise ObservationContextError(
            "Calibration frame and distortion_model must be non-empty."
        )
    _validate_numeric_tuple("D", calibration.D, minimum_length=1)
    _validate_numeric_tuple("K", calibration.K, expected_length=9)
    _validate_numeric_tuple("R", calibration.R, expected_length=9)
    _validate_numeric_tuple("P", calibration.P, expected_length=12)
    if calibration.extrinsics_available is not False:
        raise ObservationContextError(
            "Phase 1.1 requires extrinsics_available to remain false."
        )


def _validate_numeric_tuple(
    name: str,
    values: tuple[float, ...],
    *,
    expected_length: int | None = None,
    minimum_length: int | None = None,
) -> None:
    if not isinstance(values, tuple):
        raise ObservationContextError(f"Calibration {name} must be a tuple.")
    if expected_length is not None and len(values) != expected_length:
        raise ObservationContextError(
            f"Calibration {name} must contain {expected_length} values."
        )
    if minimum_length is not None and len(values) < minimum_length:
        raise ObservationContextError(
            f"Calibration {name} must contain at least {minimum_length} value."
        )
    for value in values:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float, np.integer, np.floating))
            or not math.isfinite(float(value))
        ):
            raise ObservationContextError(
                f"Calibration {name} values must be finite numbers."
            )


def _require_nonnegative_int(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ObservationContextError(f"{name} must be a non-negative integer.")


def _write_bundle_files(
    bundle_path: Path,
    observation_bundle: ObservationBundle,
) -> dict[str, object]:
    cameras = []
    observations = {
        observation.camera_id: observation
        for observation in observation_bundle.camera_observations
    }
    for camera_id in CAMERA_IDS:
        observation = observations[camera_id]
        rgb_artifact = f"{camera_id}_rgb.png"
        depth_artifact = f"{camera_id}_depth_m.npy"

        rgb_bgr = cv2.cvtColor(observation.rgb, cv2.COLOR_RGB2BGR)
        if not cv2.imwrite(str(bundle_path / rgb_artifact), rgb_bgr):
            raise ObservationContextError(f"Failed to write {rgb_artifact}.")
        np.save(
            bundle_path / depth_artifact,
            observation.depth_m,
            allow_pickle=False,
        )
        cameras.append(
            _camera_manifest(observation, rgb_artifact, depth_artifact)
        )

    return {
        "observation_ref": observation_bundle.observation_ref,
        "evidence_label": observation_bundle.evidence_label,
        "captured_at_ns": observation_bundle.captured_at_ns,
        "bundle_skew_ns": observation_bundle.bundle_skew_ns,
        "cameras": cameras,
    }


def _camera_manifest(
    observation: CameraObservation,
    rgb_artifact: str,
    depth_artifact: str,
) -> dict[str, object]:
    calibration = observation.camera_calibration
    return {
        "camera_id": observation.camera_id,
        "rgb_artifact": rgb_artifact,
        "depth_artifact": depth_artifact,
        "rgb_encoding": observation.rgb_encoding,
        "depth_encoding": observation.depth_encoding,
        "depth_units": observation.depth_units,
        "rgb_timestamp_ns": observation.rgb_timestamp_ns,
        "depth_timestamp_ns": observation.depth_timestamp_ns,
        "timestamp_difference_ns": observation.timestamp_difference_ns,
        "rgb_frame": observation.rgb_frame,
        "depth_frame": observation.depth_frame,
        "depth_registered_to_rgb": observation.depth_registered_to_rgb,
        "valid_depth_fraction": observation.valid_depth_fraction,
        "camera_calibration": {
            "width": calibration.width,
            "height": calibration.height,
            "frame": calibration.frame,
            "distortion_model": calibration.distortion_model,
            "D": list(calibration.D),
            "K": list(calibration.K),
            "R": list(calibration.R),
            "P": list(calibration.P),
            "extrinsics_available": calibration.extrinsics_available,
        },
    }


def _bundle_from_manifest(
    bundle_path: Path,
    manifest: Any,
) -> ObservationBundle:
    if not isinstance(manifest, dict) or set(manifest) != _MANIFEST_KEYS:
        raise ObservationContextError("Observation manifest fields are invalid.")
    observation_ref = manifest["observation_ref"]
    if not isinstance(observation_ref, str) or observation_ref != bundle_path.name:
        raise ObservationContextError(
            "Manifest observation_ref must match its directory."
        )
    camera_entries = manifest["cameras"]
    if not isinstance(camera_entries, list):
        raise ObservationContextError("Manifest cameras must be a list.")

    camera_observations = tuple(
        _camera_from_manifest(bundle_path, camera_entry)
        for camera_entry in camera_entries
    )
    return ObservationBundle(
        observation_ref=observation_ref,
        evidence_label=str(manifest["evidence_label"]),
        captured_at_ns=_manifest_int(manifest["captured_at_ns"]),
        camera_observations=camera_observations,
    )


def _camera_from_manifest(
    bundle_path: Path,
    camera_entry: Any,
) -> CameraObservation:
    if not isinstance(camera_entry, dict) or set(camera_entry) != _CAMERA_MANIFEST_KEYS:
        raise ObservationContextError("Camera manifest fields are invalid.")
    camera_id = camera_entry["camera_id"]
    if not isinstance(camera_id, str) or camera_id not in CAMERA_IDS:
        raise ObservationContextError("Unknown camera identifier in manifest.")

    rgb_path = _artifact_path(
        bundle_path,
        camera_entry["rgb_artifact"],
        f"{camera_id}_rgb.png",
    )
    depth_path = _artifact_path(
        bundle_path,
        camera_entry["depth_artifact"],
        f"{camera_id}_depth_m.npy",
    )
    rgb_bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if rgb_bgr is None:
        raise ObservationContextError(f"RGB artifact is unreadable for {camera_id}.")
    rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
    depth_m = np.load(depth_path, allow_pickle=False)

    calibration_entry = camera_entry["camera_calibration"]
    if (
        not isinstance(calibration_entry, dict)
        or set(calibration_entry) != _CALIBRATION_KEYS
    ):
        raise ObservationContextError("Camera calibration fields are invalid.")
    calibration = CameraCalibration(
        width=_manifest_int(calibration_entry["width"]),
        height=_manifest_int(calibration_entry["height"]),
        frame=_manifest_string(calibration_entry["frame"]),
        distortion_model=_manifest_string(
            calibration_entry["distortion_model"]
        ),
        D=_manifest_numeric_tuple(calibration_entry["D"]),
        K=_manifest_numeric_tuple(calibration_entry["K"]),
        R=_manifest_numeric_tuple(calibration_entry["R"]),
        P=_manifest_numeric_tuple(calibration_entry["P"]),
        extrinsics_available=_manifest_bool(
            calibration_entry["extrinsics_available"]
        ),
    )

    return CameraObservation(
        camera_id=camera_id,
        rgb=rgb,
        depth_m=depth_m,
        rgb_timestamp_ns=_manifest_int(camera_entry["rgb_timestamp_ns"]),
        depth_timestamp_ns=_manifest_int(camera_entry["depth_timestamp_ns"]),
        rgb_frame=_manifest_string(camera_entry["rgb_frame"]),
        depth_frame=_manifest_string(camera_entry["depth_frame"]),
        camera_calibration=calibration,
        depth_registered_to_rgb=_manifest_bool(
            camera_entry["depth_registered_to_rgb"]
        ),
        rgb_encoding=_manifest_string(camera_entry["rgb_encoding"]),
        depth_encoding=_manifest_string(camera_entry["depth_encoding"]),
        depth_units=_manifest_string(camera_entry["depth_units"]),
    )


def _artifact_path(bundle_path: Path, value: Any, expected_name: str) -> Path:
    if not isinstance(value, str) or value != expected_name:
        raise ObservationContextError("Observation artifact path is invalid.")
    artifact_path = (bundle_path / value).resolve()
    try:
        artifact_path.relative_to(bundle_path)
    except ValueError as exc:
        raise ObservationContextError(
            "Observation artifact escapes its bundle directory."
        ) from exc
    if not artifact_path.is_file():
        raise ObservationContextError("Observation artifact is missing.")
    return artifact_path


def _manifest_string(value: Any) -> str:
    if not isinstance(value, str):
        raise ObservationContextError("Manifest value must be a string.")
    return value


def _manifest_int(value: Any) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ObservationContextError("Manifest value must be an integer.")
    return value


def _manifest_bool(value: Any) -> bool:
    if not isinstance(value, bool):
        raise ObservationContextError("Manifest value must be a boolean.")
    return value


def _manifest_numeric_tuple(value: Any) -> tuple[float, ...]:
    if not isinstance(value, list):
        raise ObservationContextError("Calibration values must be a list.")
    numeric_values = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ObservationContextError("Calibration values must be numeric.")
        numeric_values.append(float(item))
    return tuple(numeric_values)


def _validate_derived_manifest_values(
    manifest: dict[str, Any],
    observation_bundle: ObservationBundle,
) -> None:
    if _manifest_int(manifest["bundle_skew_ns"]) != observation_bundle.bundle_skew_ns:
        raise ObservationContextError("Manifest bundle_skew_ns is inconsistent.")
    observations = {
        observation.camera_id: observation
        for observation in observation_bundle.camera_observations
    }
    for camera_entry in manifest["cameras"]:
        observation = observations[camera_entry["camera_id"]]
        if (
            _manifest_int(camera_entry["timestamp_difference_ns"])
            != observation.timestamp_difference_ns
        ):
            raise ObservationContextError(
                "Manifest timestamp_difference_ns is inconsistent."
            )
        valid_depth_fraction = camera_entry["valid_depth_fraction"]
        if (
            isinstance(valid_depth_fraction, bool)
            or not isinstance(valid_depth_fraction, (int, float))
            or not math.isclose(
                float(valid_depth_fraction),
                observation.valid_depth_fraction,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise ObservationContextError(
                "Manifest valid_depth_fraction is inconsistent."
            )
