"""Estimate the physical ``assembly_board-v1`` ArUco tag pose per camera role."""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from collections import deque
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

ARUCO_DICTIONARY_NAME = "DICT_ARUCO_ORIGINAL"
ARUCO_MARKER_ID = 70
DEFAULT_MARKER_LENGTH_M = 0.076
STABILITY_SAMPLE_COUNT = 10
STABILITY_CANDIDATE_COUNT = STABILITY_SAMPLE_COUNT + 1
MAXIMUM_TRANSLATION_SPREAD_M = 0.002
MAXIMUM_ROTATION_SPREAD_DEG = 0.5
MAXIMUM_REPROJECTION_ERROR_PX = 1.0
MINIMUM_IPPE_AMBIGUITY_MARGIN_PX = 0.25
MAXIMUM_FRAME_AGE_SEC = 2.0
MAXIMUM_FRAME_FUTURE_DELTA_SEC = 1.0
CHILD_FRAME_ID = "assembly_board-v1_aruco"
RESOURCE_LOCATION = "assembly_board-v1"


class ArucoLocalizationError(RuntimeError):
    """Raised when a frame cannot produce a safe ``assembly_board-v1`` tag pose."""


@dataclass(frozen=True)
class CameraCalibration:
    """Validated color-camera parameters obtained from one ROS ``CameraInfo``."""

    camera_matrix: np.ndarray
    distortion: np.ndarray
    frame_id: str
    width: int
    height: int
    distortion_model: str

    @property
    def identity(self) -> str:
        """Return a stable digest for the exact ``CameraInfo`` values."""
        digest = hashlib.sha256()
        digest.update(np.asarray(self.camera_matrix, dtype=np.float64).tobytes())
        digest.update(np.asarray(self.distortion, dtype=np.float64).tobytes())
        digest.update(
            f"{self.frame_id}|{self.width}|{self.height}|{self.distortion_model}".encode()
        )
        return digest.hexdigest()


@dataclass(frozen=True)
class CalibrationProvenance:
    """Accepted hand-eye identity and its parent-to-optical transform."""

    identity: str
    path: str
    sha256: str
    camera_role: str
    parent_frame: str
    method: str
    parent_to_camera: np.ndarray


@dataclass(frozen=True)
class ArucoPoseEstimate:
    """One ambiguity-checked IPPE pose of marker ID 70 in the camera frame."""

    camera_to_aruco: np.ndarray
    corners: np.ndarray
    reprojection_error_px: float
    alternative_reprojection_error_px: float
    ambiguity_margin_px: float
    minimum_corner_depth_m: float


@dataclass(frozen=True)
class _AcceptedSample:
    captured_at: float
    world_to_aruco: np.ndarray
    estimate: ArucoPoseEstimate
    world_to_parent: np.ndarray
    world_to_camera: np.ndarray


def _atomic_json_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def camera_calibration_from_info(message: Any) -> CameraCalibration:
    """Validate and copy the color camera parameters from a ROS ``CameraInfo``."""
    try:
        camera_matrix = np.asarray(message.k, dtype=np.float64).reshape(3, 3)
        distortion = np.asarray(message.d, dtype=np.float64).reshape(-1)
        frame_id = str(message.header.frame_id)
        width = int(message.width)
        height = int(message.height)
        distortion_model = str(message.distortion_model)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ArucoLocalizationError(f"invalid RealSense CameraInfo: {exc}") from exc
    if (
        width <= 0
        or height <= 0
        or not frame_id
        or camera_matrix[0, 0] <= 0.0
        or camera_matrix[1, 1] <= 0.0
        or not np.all(np.isfinite(camera_matrix))
        or not np.all(np.isfinite(distortion))
    ):
        raise ArucoLocalizationError("invalid RealSense CameraInfo values")
    return CameraCalibration(
        camera_matrix=camera_matrix,
        distortion=distortion,
        frame_id=frame_id,
        width=width,
        height=height,
        distortion_model=distortion_model,
    )


def ros_stamp_to_epoch_seconds(
    stamp: Any,
    *,
    now_sec: float | None = None,
    maximum_age_sec: float = MAXIMUM_FRAME_AGE_SEC,
    maximum_future_delta_sec: float = MAXIMUM_FRAME_FUTURE_DELTA_SEC,
) -> float:
    """Convert a physical ROS header stamp and require Unix-epoch compatibility."""
    try:
        captured_at = float(stamp.sec) + float(stamp.nanosec) / 1e9
    except (AttributeError, TypeError, ValueError) as exc:
        raise ArucoLocalizationError("color frame timestamp is invalid") from exc
    current = time.time() if now_sec is None else float(now_sec)
    age_sec = current - captured_at
    if (
        not math.isfinite(captured_at)
        or captured_at <= 0.0
        or not math.isfinite(current)
        or age_sec > float(maximum_age_sec)
        or age_sec < -float(maximum_future_delta_sec)
    ):
        raise ArucoLocalizationError(
            "color frame timestamp is not Unix-epoch compatible with time.time()"
        )
    return captured_at


def _transform_from_payload(payload: dict[str, Any]) -> np.ndarray:
    try:
        translation = payload["translation"]
        quaternion = payload["quaternion"]
        x = float(quaternion["x"])
        y = float(quaternion["y"])
        z = float(quaternion["z"])
        w = float(quaternion["w"])
        values = np.array(
            [
                float(translation["x"]),
                float(translation["y"]),
                float(translation["z"]),
                x,
                y,
                z,
                w,
            ],
            dtype=np.float64,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ArucoLocalizationError("hand-eye transform is incomplete") from exc
    if not np.all(np.isfinite(values)):
        raise ArucoLocalizationError("hand-eye transform contains a non-finite value")
    norm = float(np.linalg.norm(values[3:]))
    if norm <= 0.0:
        raise ArucoLocalizationError("hand-eye transform has a zero-length quaternion")
    x, y, z, w = values[3:] / norm
    rotation = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = values[:3]
    return transform


def load_calibration_provenance(
    path: str | Path,
    *,
    camera_role: str,
    parent_frame: str,
) -> CalibrationProvenance:
    """Load one accepted, role-matched hand-eye calibration."""
    expanded = Path(path).expanduser()
    try:
        raw = expanded.read_bytes()
    except OSError as exc:
        raise ArucoLocalizationError(f"hand-eye calibration unavailable: {expanded}") from exc
    try:
        payload = yaml.safe_load(raw) or {}
    except yaml.YAMLError as exc:
        raise ArucoLocalizationError(f"invalid hand-eye calibration: {expanded}") from exc
    validation = payload.get("validation")
    if not isinstance(validation, dict) or not bool(validation.get("accepted", False)):
        raise ArucoLocalizationError("hand-eye calibration is not marked accepted")
    identity = str(payload.get("calibration_id") or "").strip()
    if not identity:
        raise ArucoLocalizationError("hand-eye calibration has no calibration_id")
    if str(payload.get("camera_role") or "") != camera_role:
        raise ArucoLocalizationError(
            f"hand-eye calibration camera_role does not match {camera_role}"
        )
    if str(payload.get("parent_frame") or "") != parent_frame:
        raise ArucoLocalizationError(
            f"hand-eye calibration parent_frame does not match {parent_frame}"
        )
    transform_payload = payload.get("parent_to_camera_color_optical_frame")
    if not isinstance(transform_payload, dict):
        raise ArucoLocalizationError(
            "hand-eye calibration is missing parent_to_camera_color_optical_frame"
        )
    return CalibrationProvenance(
        identity=identity,
        path=str(expanded),
        sha256=hashlib.sha256(raw).hexdigest(),
        camera_role=camera_role,
        parent_frame=parent_frame,
        method=str(payload.get("method") or ""),
        parent_to_camera=_transform_from_payload(transform_payload),
    )


def matrix_from_transform_message(message: Any) -> np.ndarray:
    """Convert a ROS ``TransformStamped`` or ``Transform`` into a matrix."""
    transform = getattr(message, "transform", message)
    return _transform_from_payload(
        {
            "translation": {
                "x": transform.translation.x,
                "y": transform.translation.y,
                "z": transform.translation.z,
            },
            "quaternion": {
                "x": transform.rotation.x,
                "y": transform.rotation.y,
                "z": transform.rotation.z,
                "w": transform.rotation.w,
            },
        }
    )


def _validated_rigid_transform(value: Any, *, name: str) -> np.ndarray:
    try:
        transform = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ArucoLocalizationError(f"{name} is not a numeric 4x4 transform") from exc
    if transform.shape != (4, 4):
        raise ArucoLocalizationError(f"{name} must be a 4x4 transform")
    if not np.all(np.isfinite(transform)):
        raise ArucoLocalizationError(f"{name} contains a non-finite value")
    if not np.allclose(
        transform[3],
        np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
        rtol=0.0,
        atol=1e-9,
    ):
        raise ArucoLocalizationError(f"{name} has an invalid homogeneous row")
    rotation = transform[:3, :3]
    if not np.allclose(
        rotation.T @ rotation,
        np.eye(3, dtype=np.float64),
        rtol=0.0,
        atol=1e-6,
    ):
        raise ArucoLocalizationError(f"{name} rotation is not orthonormal")
    determinant = float(np.linalg.det(rotation))
    if not math.isclose(determinant, 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ArucoLocalizationError(f"{name} rotation is not proper")
    return transform


def transform_camera_optical_point_to_assembly_board_v1(
    point_camera_m: Any,
    *,
    point_frame_id: str,
    camera_frame_id: str,
    camera_to_aruco: Any,
    assembly_board_v1_aruco_to_assembly_board_v1: Any,
) -> np.ndarray:
    """Transform one exact optical-frame point into ``assembly_board-v1``.

    Args:
        point_camera_m: Three-dimensional part point in the camera optical frame.
        point_frame_id: Exact frame identifier carried with ``point_camera_m``.
        camera_frame_id: Exact optical frame from the RealSense ``CameraInfo``.
        camera_to_aruco: Pose of ``assembly_board-v1_aruco`` in the camera frame.
        assembly_board_v1_aruco_to_assembly_board_v1: Pose of
            ``assembly_board-v1`` in ``assembly_board-v1_aruco``.

    Returns:
        The three-dimensional part point in ``assembly_board-v1`` coordinates.

    Raises:
        ArucoLocalizationError: If the frames do not match exactly, the point is
            invalid, or either transform is not a finite rigid transform.
    """
    if not isinstance(point_frame_id, str) or not point_frame_id:
        raise ArucoLocalizationError("camera optical-frame point frame_id is missing")
    if not isinstance(camera_frame_id, str) or not camera_frame_id:
        raise ArucoLocalizationError("RealSense CameraInfo frame_id is missing")
    if point_frame_id != camera_frame_id:
        raise ArucoLocalizationError(
            "camera optical-frame point frame_id does not exactly match RealSense "
            f"CameraInfo: {point_frame_id!r} != {camera_frame_id!r}"
        )
    try:
        point = np.asarray(point_camera_m, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ArucoLocalizationError(
            "camera optical-frame part point is not a numeric three-vector"
        ) from exc
    if point.shape != (3,):
        raise ArucoLocalizationError(
            "camera optical-frame part point must be a three-vector"
        )
    if not np.all(np.isfinite(point)):
        raise ArucoLocalizationError(
            "camera optical-frame part point contains a non-finite value"
        )

    marker_pose_in_camera = _validated_rigid_transform(
        camera_to_aruco,
        name="camera_to_aruco",
    )
    board_pose_in_marker = _validated_rigid_transform(
        assembly_board_v1_aruco_to_assembly_board_v1,
        name="assembly_board-v1_aruco_to_assembly_board-v1",
    )
    point_camera = np.append(point, 1.0)
    point_board = (
        np.linalg.inv(board_pose_in_marker)
        @ np.linalg.inv(marker_pose_in_camera)
        @ point_camera
    )
    if not np.all(np.isfinite(point_board)) or not math.isclose(
        float(point_board[3]),
        1.0,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ArucoLocalizationError(
            "camera optical-frame part point produced invalid assembly_board-v1 coordinates"
        )
    return point_board[:3]


def detect_assembly_board_v1_aruco(image: np.ndarray) -> np.ndarray:
    """Return the four refined corners of exactly one original-dictionary ID 70."""
    if not hasattr(cv2, "aruco"):
        raise ArucoLocalizationError("OpenCV ArUco support is unavailable")
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
    parameters = cv2.aruco.DetectorParameters()
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    if hasattr(cv2.aruco, "ArucoDetector"):
        corners, identifiers, _ = cv2.aruco.ArucoDetector(
            dictionary,
            parameters,
        ).detectMarkers(image)
    else:
        corners, identifiers, _ = cv2.aruco.detectMarkers(
            image,
            dictionary,
            parameters=parameters,
        )
    if identifiers is None:
        raise ArucoLocalizationError("assembly_board-v1 ArUco ID 70 is not visible")
    matches = [
        np.asarray(corner, dtype=np.float64).reshape(4, 2)
        for corner, identifier in zip(corners, identifiers.reshape(-1), strict=True)
        if int(identifier) == ARUCO_MARKER_ID
    ]
    if not matches:
        raise ArucoLocalizationError("assembly_board-v1 ArUco ID 70 is not visible")
    if len(matches) != 1:
        raise ArucoLocalizationError("multiple assembly_board-v1 ArUco ID 70 markers are visible")
    return matches[0]


def _marker_object_points(marker_length_m: float) -> np.ndarray:
    half = float(marker_length_m) * 0.5
    return np.array(
        [
            [-half, half, 0.0],
            [half, half, 0.0],
            [half, -half, 0.0],
            [-half, -half, 0.0],
        ],
        dtype=np.float64,
    )


def estimate_camera_to_aruco(
    corners: np.ndarray,
    camera: CameraCalibration,
    *,
    marker_length_m: float = DEFAULT_MARKER_LENGTH_M,
    maximum_reprojection_error_px: float = MAXIMUM_REPROJECTION_ERROR_PX,
    minimum_ambiguity_margin_px: float = MINIMUM_IPPE_AMBIGUITY_MARGIN_PX,
) -> ArucoPoseEstimate:
    """Solve ID 70 with IPPE square and reject unsafe planar-pose ambiguity."""
    if not math.isfinite(marker_length_m) or marker_length_m <= 0.0:
        raise ArucoLocalizationError("assembly_board-v1 ArUco marker length is invalid")
    image_points = np.asarray(corners, dtype=np.float64).reshape(4, 2)
    object_points = _marker_object_points(marker_length_m)
    try:
        solved, rvecs, tvecs, _ = cv2.solvePnPGeneric(
            object_points,
            image_points,
            camera.camera_matrix,
            camera.distortion,
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )
    except cv2.error as exc:
        raise ArucoLocalizationError(f"ArUco IPPE pose solve failed: {exc}") from exc
    if not solved or len(rvecs) < 2 or len(tvecs) < 2:
        raise ArucoLocalizationError("ArUco IPPE did not return both planar pose solutions")

    candidates: list[tuple[float, float, np.ndarray]] = []
    for rvec, tvec in zip(rvecs, tvecs, strict=True):
        rotation, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64))
        translation = np.asarray(tvec, dtype=np.float64).reshape(3)
        corner_depths = (rotation @ object_points.T + translation[:, None])[2]
        if not np.all(np.isfinite(corner_depths)) or float(np.min(corner_depths)) <= 0.0:
            continue
        projected, _ = cv2.projectPoints(
            object_points,
            rvec,
            tvec,
            camera.camera_matrix,
            camera.distortion,
        )
        differences = projected.reshape(4, 2) - image_points
        error = float(np.sqrt(np.mean(np.sum(np.square(differences), axis=1))))
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = rotation
        transform[:3, 3] = translation
        candidates.append((error, float(np.min(corner_depths)), transform))
    if len(candidates) < 2:
        raise ArucoLocalizationError("ArUco IPPE has fewer than two positive-depth solutions")
    candidates.sort(key=lambda candidate: candidate[0])
    best_error, minimum_depth, camera_to_aruco = candidates[0]
    alternative_error = candidates[1][0]
    if not math.isfinite(best_error) or best_error > maximum_reprojection_error_px:
        raise ArucoLocalizationError(
            "assembly_board-v1 ArUco reprojection error "
            f"{best_error:.3f} px exceeds {maximum_reprojection_error_px:.3f} px"
        )
    ambiguity_margin = alternative_error - best_error
    if not math.isfinite(ambiguity_margin) or ambiguity_margin < minimum_ambiguity_margin_px:
        raise ArucoLocalizationError(
            "assembly_board-v1 ArUco IPPE pose is ambiguous: "
            f"margin {ambiguity_margin:.3f} px is below "
            f"{minimum_ambiguity_margin_px:.3f} px"
        )
    return ArucoPoseEstimate(
        camera_to_aruco=camera_to_aruco,
        corners=image_points,
        reprojection_error_px=best_error,
        alternative_reprojection_error_px=alternative_error,
        ambiguity_margin_px=ambiguity_margin,
        minimum_corner_depth_m=minimum_depth,
    )


def _rotation_distance_deg(first: np.ndarray, second: np.ndarray) -> float:
    relative = first.T @ second
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def _maximum_pairwise_spreads(samples: list[_AcceptedSample]) -> tuple[float, float]:
    translation_spread = 0.0
    rotation_spread = 0.0
    for index, first in enumerate(samples):
        for second in samples[index + 1 :]:
            translation_spread = max(
                translation_spread,
                float(np.linalg.norm(first.world_to_aruco[:3, 3] - second.world_to_aruco[:3, 3])),
            )
            rotation_spread = max(
                rotation_spread,
                _rotation_distance_deg(
                    first.world_to_aruco[:3, :3],
                    second.world_to_aruco[:3, :3],
                ),
            )
    return translation_spread, rotation_spread


def _stability_window(
    samples: list[_AcceptedSample],
) -> tuple[list[_AcceptedSample], float, float]:
    """Select ten mutually consistent samples while retaining the newest frame."""
    if len(samples) <= STABILITY_SAMPLE_COUNT:
        translation_spread, rotation_spread = _maximum_pairwise_spreads(samples)
        return samples, translation_spread, rotation_spread

    newest_index = len(samples) - 1
    stable_selection: tuple[list[_AcceptedSample], float, float] | None = None
    closest_selection: tuple[list[_AcceptedSample], float, float] | None = None
    closest_score = (math.inf, math.inf)
    for earlier_indices in combinations(
        range(newest_index),
        STABILITY_SAMPLE_COUNT - 1,
    ):
        selected = [samples[index] for index in (*earlier_indices, newest_index)]
        translation_spread, rotation_spread = _maximum_pairwise_spreads(selected)
        if (
            translation_spread <= MAXIMUM_TRANSLATION_SPREAD_M
            and rotation_spread <= MAXIMUM_ROTATION_SPREAD_DEG
        ):
            # combinations are ordered oldest-to-newest, so retaining the last
            # passing selection favors the freshest complete ten-frame window.
            stable_selection = selected, translation_spread, rotation_spread
            continue
        score = (
            max(
                translation_spread / MAXIMUM_TRANSLATION_SPREAD_M,
                rotation_spread / MAXIMUM_ROTATION_SPREAD_DEG,
            ),
            translation_spread / MAXIMUM_TRANSLATION_SPREAD_M
            + rotation_spread / MAXIMUM_ROTATION_SPREAD_DEG,
        )
        if score < closest_score:
            closest_score = score
            closest_selection = selected, translation_spread, rotation_spread
    if stable_selection is not None:
        return stable_selection
    if closest_selection is not None:
        return closest_selection
    translation_spread, rotation_spread = _maximum_pairwise_spreads(samples)
    return samples, translation_spread, rotation_spread


def _mean_transform(samples: list[_AcceptedSample]) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, 3] = np.mean(
        [sample.world_to_aruco[:3, 3] for sample in samples],
        axis=0,
    )
    summed = np.sum([sample.world_to_aruco[:3, :3] for sample in samples], axis=0)
    left, _, right = np.linalg.svd(summed)
    rotation = left @ right
    if np.linalg.det(rotation) < 0.0:
        left[:, -1] *= -1.0
        rotation = left @ right
    transform[:3, :3] = rotation
    return transform


def _quaternion_from_matrix(rotation: np.ndarray) -> tuple[float, float, float, float]:
    matrix = np.asarray(rotation, dtype=np.float64)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quaternion = (
            (matrix[2, 1] - matrix[1, 2]) / scale,
            (matrix[0, 2] - matrix[2, 0]) / scale,
            (matrix[1, 0] - matrix[0, 1]) / scale,
            0.25 * scale,
        )
    else:
        axis = int(np.argmax(np.diag(matrix)))
        if axis == 0:
            scale = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            quaternion = (
                0.25 * scale,
                (matrix[0, 1] + matrix[1, 0]) / scale,
                (matrix[0, 2] + matrix[2, 0]) / scale,
                (matrix[2, 1] - matrix[1, 2]) / scale,
            )
        elif axis == 1:
            scale = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            quaternion = (
                (matrix[0, 1] + matrix[1, 0]) / scale,
                0.25 * scale,
                (matrix[1, 2] + matrix[2, 1]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
            )
        else:
            scale = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            quaternion = (
                (matrix[0, 2] + matrix[2, 0]) / scale,
                (matrix[1, 2] + matrix[2, 1]) / scale,
                0.25 * scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
            )
    norm = float(np.linalg.norm(quaternion))
    return tuple(float(value / norm) for value in quaternion)


def _transform_payload(transform: np.ndarray) -> dict[str, Any]:
    x, y, z, w = _quaternion_from_matrix(transform[:3, :3])
    return {
        "translation": {
            "x": float(transform[0, 3]),
            "y": float(transform[1, 3]),
            "z": float(transform[2, 3]),
        },
        "quaternion": {"x": x, "y": y, "z": z, "w": w},
    }


class AssemblyBoardV1ArucoLocalizer:
    """Maintain and atomically publish one role's ten-frame stable tag pose."""

    def __init__(
        self,
        *,
        camera_role: str,
        world_frame: str,
        parent_frame: str,
        marker_length_m: float,
        output_path: Path,
    ) -> None:
        if camera_role not in ("ur5e", "xarm6"):
            raise ValueError("assembly_board-v1 ArUco localization requires ur5e or xarm6")
        if not math.isfinite(marker_length_m) or marker_length_m <= 0.0:
            raise ValueError("assembly_board-v1 ArUco marker length must be positive")
        self.camera_role = camera_role
        self.world_frame = world_frame
        self.parent_frame = parent_frame
        self.marker_length_m = float(marker_length_m)
        self.output_path = output_path
        self._samples: deque[_AcceptedSample] = deque(maxlen=STABILITY_CANDIDATE_COUNT)
        self._sample_identity: tuple[str, str] | None = None

    def invalidate(
        self,
        error: str,
        *,
        frame_captured_at: float | None = None,
        calibration: CalibrationProvenance | None = None,
        camera: CameraCalibration | None = None,
    ) -> dict[str, Any]:
        """Clear prior pose authority and publish an invalid status immediately."""
        self._samples.clear()
        self._sample_identity = None
        payload = self._payload(
            valid=False,
            visible=False,
            error=str(error),
            frame_captured_at=frame_captured_at,
            calibration=calibration,
            camera=camera,
        )
        _atomic_json_write(self.output_path, payload)
        return payload

    def observe(
        self,
        *,
        image: np.ndarray,
        frame_captured_at: float,
        camera: CameraCalibration,
        calibration: CalibrationProvenance,
        world_to_parent: np.ndarray,
    ) -> tuple[dict[str, Any], np.ndarray]:
        """Process one timestamped image and publish its current stability state."""
        height, width = image.shape[:2]
        if (width, height) != (camera.width, camera.height):
            raise ArucoLocalizationError(
                "RealSense CameraInfo dimensions do not match the color frame: "
                f"CameraInfo={camera.width}x{camera.height}, color={width}x{height}"
            )
        if not math.isfinite(frame_captured_at) or frame_captured_at <= 0.0:
            raise ArucoLocalizationError("color frame timestamp is invalid")
        corners = detect_assembly_board_v1_aruco(image)
        estimate = estimate_camera_to_aruco(
            corners,
            camera,
            marker_length_m=self.marker_length_m,
        )
        world_to_parent = np.asarray(world_to_parent, dtype=np.float64).reshape(4, 4)
        if not np.all(np.isfinite(world_to_parent)):
            raise ArucoLocalizationError("timestamped world-to-parent TF is invalid")
        world_to_camera = world_to_parent @ calibration.parent_to_camera
        world_to_aruco = world_to_camera @ estimate.camera_to_aruco
        identity = (calibration.identity, camera.identity)
        if self._sample_identity != identity:
            self._samples.clear()
            self._sample_identity = identity
        if self._samples and frame_captured_at <= self._samples[-1].captured_at:
            raise ArucoLocalizationError("color frame timestamp is not newer than the last sample")
        self._samples.append(
            _AcceptedSample(
                captured_at=float(frame_captured_at),
                world_to_aruco=world_to_aruco,
                estimate=estimate,
                world_to_parent=world_to_parent,
                world_to_camera=world_to_camera,
            )
        )
        samples, translation_spread, rotation_spread = _stability_window(
            list(self._samples)
        )
        window_complete = len(samples) == STABILITY_SAMPLE_COUNT
        stable = (
            window_complete
            and translation_spread <= MAXIMUM_TRANSLATION_SPREAD_M
            and rotation_spread <= MAXIMUM_ROTATION_SPREAD_DEG
        )
        if not window_complete:
            error = (
                "assembly_board-v1 ArUco stability window incomplete: "
                f"{len(samples)}/{STABILITY_SAMPLE_COUNT} frames"
            )
        elif not stable:
            error = (
                "assembly_board-v1 ArUco pose is unstable: "
                f"{translation_spread * 1000.0:.3f} mm / "
                f"{rotation_spread:.3f} deg"
            )
        else:
            error = ""
        payload = self._payload(
            valid=stable,
            visible=True,
            error=error,
            frame_captured_at=frame_captured_at,
            calibration=calibration,
            camera=camera,
            samples=samples,
        )
        _atomic_json_write(self.output_path, payload)
        return payload, corners

    def _payload(
        self,
        *,
        valid: bool,
        visible: bool,
        error: str,
        frame_captured_at: float | None,
        calibration: CalibrationProvenance | None,
        camera: CameraCalibration | None,
        samples: list[_AcceptedSample] | None = None,
    ) -> dict[str, Any]:
        samples = list(self._samples) if samples is None else samples
        if samples:
            translation_spread_m, rotation_spread_deg = _maximum_pairwise_spreads(
                samples
            )
        else:
            translation_spread_m = None
            rotation_spread_deg = None
        pose = _mean_transform(samples) if valid else None
        latest = samples[-1] if samples else None
        reprojection_errors = [sample.estimate.reprojection_error_px for sample in samples]
        pose_transform = _transform_payload(pose) if pose is not None else None
        pose_translation = pose_transform["translation"] if pose_transform else None
        pose_quaternion = pose_transform["quaternion"] if pose_transform else None
        return {
            "schema_version": 1,
            "updated_at": time.time(),
            "camera_role": self.camera_role,
            "resource_location": RESOURCE_LOCATION,
            "valid": bool(valid),
            "visible": bool(visible),
            "stable": bool(valid),
            "world_pose_ready": bool(valid),
            "frame_id": self.world_frame,
            "child_frame_id": CHILD_FRAME_ID,
            "timestamp_domain": "unix_epoch_seconds",
            "sample_started_at": samples[0].captured_at if samples else None,
            "frame_captured_at": frame_captured_at,
            "sample_count": len(samples),
            "required_sample_count": STABILITY_SAMPLE_COUNT,
            "marker": {
                "dictionary": ARUCO_DICTIONARY_NAME,
                "id": ARUCO_MARKER_ID,
                "marker_length_m": self.marker_length_m,
            },
            "marker_dictionary": ARUCO_DICTIONARY_NAME,
            "marker_id": ARUCO_MARKER_ID,
            "marker_length_m": self.marker_length_m,
            "pose": (
                {
                    "frame_id": self.world_frame,
                    "child_frame_id": CHILD_FRAME_ID,
                    "x": pose_translation["x"],
                    "y": pose_translation["y"],
                    "z": pose_translation["z"],
                    "qx": pose_quaternion["x"],
                    "qy": pose_quaternion["y"],
                    "qz": pose_quaternion["z"],
                    "qw": pose_quaternion["w"],
                }
                if pose is not None
                else None
            ),
            "stability": {
                "window_frame_count": len(samples),
                "required_window_frame_count": STABILITY_SAMPLE_COUNT,
                "translation_spread_m": translation_spread_m,
                "rotation_spread_deg": rotation_spread_deg,
                "maximum_translation_spread_m": MAXIMUM_TRANSLATION_SPREAD_M,
                "maximum_rotation_spread_deg": MAXIMUM_ROTATION_SPREAD_DEG,
            },
            "quality": {
                "median_reprojection_error_px": (
                    float(np.median(reprojection_errors)) if reprojection_errors else None
                ),
                "latest_reprojection_error_px": (
                    latest.estimate.reprojection_error_px if latest else None
                ),
                "latest_alternative_reprojection_error_px": (
                    latest.estimate.alternative_reprojection_error_px if latest else None
                ),
                "latest_ambiguity_margin_px": (
                    latest.estimate.ambiguity_margin_px if latest else None
                ),
                "latest_minimum_corner_depth_m": (
                    latest.estimate.minimum_corner_depth_m if latest else None
                ),
                "maximum_reprojection_error_px": MAXIMUM_REPROJECTION_ERROR_PX,
                "minimum_ippe_ambiguity_margin_px": MINIMUM_IPPE_AMBIGUITY_MARGIN_PX,
            },
            "reprojection_error_px": (latest.estimate.reprojection_error_px if latest else None),
            "translation_spread_m": translation_spread_m,
            "rotation_spread_deg": rotation_spread_deg,
            "calibration_id": calibration.identity if calibration else None,
            "calibration": {
                "identity": calibration.identity if calibration else None,
                "path": calibration.path if calibration else None,
                "sha256": calibration.sha256 if calibration else None,
                "camera_role": calibration.camera_role if calibration else None,
                "parent_frame": calibration.parent_frame if calibration else self.parent_frame,
                "method": calibration.method if calibration else None,
                "camera_info_identity": camera.identity if camera else None,
                "camera_info_frame_id": camera.frame_id if camera else None,
                "camera_info_width": camera.width if camera else None,
                "camera_info_height": camera.height if camera else None,
                "distortion_model": camera.distortion_model if camera else None,
            },
            "tf": {
                "captured_at": latest.captured_at if latest else None,
                "world_frame": self.world_frame,
                "parent_frame": self.parent_frame,
                "camera_frame": camera.frame_id if camera else None,
                "world_to_parent": (_transform_payload(latest.world_to_parent) if latest else None),
                "world_to_camera": (_transform_payload(latest.world_to_camera) if latest else None),
            },
            "last_error": str(error),
        }
