"""ROS2 RealSense and Roboflow gear-pose service node."""

from __future__ import annotations

import json
import logging
import math
import os
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from dotenv import load_dotenv

from .assembly_board_v1_aruco import (
    ARUCO_MARKER_ID,
    DEFAULT_MARKER_LENGTH_M,
    MAXIMUM_REPROJECTION_ERROR_PX,
    MAXIMUM_ROTATION_SPREAD_DEG,
    MAXIMUM_TRANSLATION_SPREAD_M,
    MINIMUM_IPPE_AMBIGUITY_MARGIN_PX,
    STABILITY_SAMPLE_COUNT,
    ArucoLocalizationError,
    ArucoPoseEstimate,
    CameraCalibration,
    camera_calibration_from_info,
    detect_assembly_board_v1_aruco,
    estimate_camera_to_aruco,
    transform_camera_optical_point_to_assembly_board_v1,
)
from .assembly_board_v1_aruco import (
    MAXIMUM_FRAME_AGE_SEC as MAXIMUM_ARUCO_WINDOW_AGE_SEC,
)
from .realsense_pose_estimator import (
    CalibrationError,
    ColorIntrinsics,
    DepthQualityError,
    RigidTransform,
    constrain_gear_center_to_table_plane,
    deproject_pixel,
    gear_center_world_point,
    load_hand_eye_calibration,
    pose_motion,
    quaternion_matrix,
    robust_surface_depth,
    table_surface_z_from_calibration,
)
from .roboflow_detector import (
    UNSUPPORTED_PARTS,
    DuplicateDetectionError,
    RoboflowConfigurationError,
    RoboflowGearDetector,
    RoboflowResponseError,
    RoboflowSettings,
)

logger = logging.getLogger(__name__)

DEFAULT_SNAPSHOT_PATH = Path("/tmp/cais_physical_perception.json")
DEFAULT_PREVIEW_ROOT = Path("/tmp/cais_perception_previews")
DEFAULT_ASSEMBLY_BOARD_V1_GEOMETRY_PATH = (
    Path(__file__).resolve().parents[3]
    / "specification/products/geometry/assembly_board-v1.json"
)
STATIONARY_INSPECTION_PARTS = ("SG", "MG")
STATIONARY_INSPECTION_UNSUPPORTED_PART_NAMES = (
    "LG",
    "SRP",
    "MRP",
    "LRP",
    "SCP",
    "MCP",
    "LCP",
)
STATIONARY_INSPECTION_XY_TOLERANCE_M = 0.010
STATIONARY_INSPECTION_SEATING_TOLERANCE_M = 0.005
STATIONARY_INSPECTION_FRAME_ID = "assembly_board-v1"
STATIONARY_REGISTRATION_NAME = "assembly_board-v1_aruco_to_assembly_board-v1"
STATIONARY_REGISTRATION_UNAVAILABLE = (
    "assembly_board-v1_aruco_to_assembly_board-v1 is not configured; "
    "stationary assembly inspection is unavailable."
)


@dataclass(frozen=True)
class _StationaryArucoObservation:
    """One quality-checked ID 70 pose from a synchronized stationary frame."""

    stamp_ns: int
    captured_at: float
    camera: CameraCalibration
    estimate: ArucoPoseEstimate


def _ros_stamp_values(stamp: Any) -> tuple[int, float]:
    """Return an exact ROS stamp key and its epoch-seconds representation."""
    try:
        sec = int(stamp.sec)
        nanosec = int(stamp.nanosec)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ArucoLocalizationError("color frame timestamp is invalid") from exc
    if sec <= 0 or nanosec < 0 or nanosec >= 1_000_000_000:
        raise ArucoLocalizationError("color frame timestamp is invalid")
    return sec * 1_000_000_000 + nanosec, float(sec) + float(nanosec) / 1e9


def _rotation_distance_deg(first: np.ndarray, second: np.ndarray) -> float:
    """Return the shortest rotation distance between two rotation matrices."""
    relative = first.T @ second
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def _stationary_aruco_window_quality(
    observations: list[_StationaryArucoObservation],
) -> dict[str, float | int]:
    """Validate one exact-frame, recent, ten-sample ID 70 stability window."""
    if len(observations) != STABILITY_SAMPLE_COUNT:
        raise ArucoLocalizationError(
            "assembly_board-v1 ArUco stability window incomplete: "
            f"{len(observations)}/{STABILITY_SAMPLE_COUNT} frames"
        )
    if observations[-1].captured_at - observations[0].captured_at > (
        MAXIMUM_ARUCO_WINDOW_AGE_SEC
    ):
        raise ArucoLocalizationError(
            "assembly_board-v1 ArUco stability window is stale: "
            f"{observations[-1].captured_at - observations[0].captured_at:.3f} s "
            f"exceeds {MAXIMUM_ARUCO_WINDOW_AGE_SEC:.3f} s"
        )
    translation_spread_m = 0.0
    rotation_spread_deg = 0.0
    for index, first in enumerate(observations):
        for second in observations[index + 1 :]:
            translation_spread_m = max(
                translation_spread_m,
                float(
                    np.linalg.norm(
                        first.estimate.camera_to_aruco[:3, 3]
                        - second.estimate.camera_to_aruco[:3, 3]
                    )
                ),
            )
            rotation_spread_deg = max(
                rotation_spread_deg,
                _rotation_distance_deg(
                    first.estimate.camera_to_aruco[:3, :3],
                    second.estimate.camera_to_aruco[:3, :3],
                ),
            )
    if (
        translation_spread_m > MAXIMUM_TRANSLATION_SPREAD_M
        or rotation_spread_deg > MAXIMUM_ROTATION_SPREAD_DEG
    ):
        raise ArucoLocalizationError(
            "assembly_board-v1 ArUco pose is unstable: "
            f"{translation_spread_m * 1000.0:.3f} mm / "
            f"{rotation_spread_deg:.3f} deg"
        )
    return {
        "sample_count": len(observations),
        "required_sample_count": STABILITY_SAMPLE_COUNT,
        "sample_started_at": observations[0].captured_at,
        "window_age_sec": observations[-1].captured_at - observations[0].captured_at,
        "translation_spread_m": translation_spread_m,
        "maximum_translation_spread_m": MAXIMUM_TRANSLATION_SPREAD_M,
        "rotation_spread_deg": rotation_spread_deg,
        "maximum_rotation_spread_deg": MAXIMUM_ROTATION_SPREAD_DEG,
    }


def _registration_matrix(registration: dict[str, Any]) -> np.ndarray:
    """Return the configured marker-to-board transform without filling null values."""
    try:
        translation = tuple(float(registration[field]) for field in ("x", "y", "z"))
        quaternion = tuple(
            float(registration[field]) for field in ("qx", "qy", "qz", "qw")
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise CalibrationError(STATIONARY_REGISTRATION_UNAVAILABLE) from exc
    values = np.asarray((*translation, *quaternion), dtype=np.float64)
    if not np.all(np.isfinite(values)) or float(np.linalg.norm(values[3:])) <= 0.0:
        raise CalibrationError(STATIONARY_REGISTRATION_UNAVAILABLE)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = quaternion_matrix(quaternion)
    transform[:3, 3] = translation
    return transform


def _load_stationary_inspection_geometry(path: str | Path) -> dict[str, Any]:
    """Load physical SG/MG slots and marker registration from the existing geometry."""
    expanded = Path(path).expanduser()
    try:
        payload = json.loads(expanded.read_text(encoding="utf-8"))
        physical = payload["real"]
        board = physical["assembly_board"]
        parts = physical["parts"]
        slots = board["slots"]
        heights = parts["heights_m"]
        board_center_z_m = float(board["center"]["z"])
        slot_floor_z_m = float(board["slot_floor_z_m"])
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise CalibrationError(
            f"stationary assembly inspection geometry is invalid: {expanded}: {exc}"
        ) from exc
    expected_top_surfaces: dict[str, tuple[float, float, float]] = {}
    for part_name in STATIONARY_INSPECTION_PARTS:
        try:
            slot = slots[part_name]
            values = (
                float(slot[0]),
                float(slot[1]),
                slot_floor_z_m - board_center_z_m + float(heights[part_name]),
            )
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise CalibrationError(
                f"stationary assembly inspection {part_name} geometry is invalid"
            ) from exc
        if not all(math.isfinite(value) for value in values):
            raise CalibrationError(
                f"stationary assembly inspection {part_name} geometry is not finite"
            )
        expected_top_surfaces[part_name] = values

    registration = board.get(STATIONARY_REGISTRATION_NAME)
    if not isinstance(registration, dict):
        registration = {}
    calibration_id = str(registration.get("calibration_id") or "")
    try:
        marker_to_board = _registration_matrix(registration)
    except CalibrationError:
        registration_configured = False
        marker_to_board = None
        registration_error = STATIONARY_REGISTRATION_UNAVAILABLE
    else:
        registration_configured = bool(
            calibration_id and calibration_id == calibration_id.strip()
        )
        registration_error = (
            "" if registration_configured else STATIONARY_REGISTRATION_UNAVAILABLE
        )
        if not registration_configured:
            marker_to_board = None
    return {
        "path": str(expanded),
        "expected_top_surfaces": expected_top_surfaces,
        "registration": registration,
        "registration_calibration_id": calibration_id,
        "registration_configured": registration_configured,
        "registration_error": registration_error,
        "marker_to_board": marker_to_board,
    }


def _stationary_aruco_evidence(
    observation: _StationaryArucoObservation,
    window_quality: dict[str, float | int],
    *,
    marker_length_m: float,
) -> dict[str, Any]:
    """Return exact-frame marker quality and stable-window evidence."""
    estimate = observation.estimate
    return {
        "marker_id": ARUCO_MARKER_ID,
        "marker_length_m": float(marker_length_m),
        "captured_at": observation.captured_at,
        "frame_stamp_ns": observation.stamp_ns,
        "visible": True,
        "valid": True,
        "stable": True,
        "camera_info_frame_id": observation.camera.frame_id,
        "camera_info_identity": observation.camera.identity,
        "reprojection_error_px": estimate.reprojection_error_px,
        "alternative_reprojection_error_px": (
            estimate.alternative_reprojection_error_px
        ),
        "ambiguity_margin_px": estimate.ambiguity_margin_px,
        "minimum_corner_depth_m": estimate.minimum_corner_depth_m,
        "maximum_reprojection_error_px": MAXIMUM_REPROJECTION_ERROR_PX,
        "minimum_ippe_ambiguity_margin_px": MINIMUM_IPPE_AMBIGUITY_MARGIN_PX,
        **window_quality,
    }


def _expected_stationary_part_evidence(
    geometry: dict[str, Any],
    *,
    detected_parts: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Return SG/MG expected top surfaces for unavailable inspection evidence."""
    expected_top_surfaces = dict(geometry.get("expected_top_surfaces") or {})
    detected_parts = set() if detected_parts is None else detected_parts
    evidence: list[dict[str, Any]] = []
    for part_name in STATIONARY_INSPECTION_PARTS:
        expected = expected_top_surfaces.get(part_name)
        evidence.append(
            {
                "part_name": part_name,
                "detected": part_name in detected_parts,
                "success": False,
                "expected": (
                    {
                        "frame_id": STATIONARY_INSPECTION_FRAME_ID,
                        "x": float(expected[0]),
                        "y": float(expected[1]),
                        "z": float(expected[2]),
                        "point": "top_surface",
                    }
                    if expected is not None
                    else None
                ),
                "observed": None,
                "xy_error_m": None,
                "seating_error_m": None,
            }
        )
    return evidence


def _stationary_inspection_unavailable_payload(
    message: str,
    *,
    geometry: dict[str, Any] | None = None,
    marker_length_m: float = DEFAULT_MARKER_LENGTH_M,
    captured_at: float | None = None,
    aruco: dict[str, Any] | None = None,
    rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Return a fail-closed stationary inspection status with stable wire fields."""
    geometry = {} if geometry is None else geometry
    row_parts = {
        str(row.get("part_name") or "")
        for row in (rows or [])
        if isinstance(row, dict)
    }
    registration_configured = bool(geometry.get("registration_configured", False))
    marker = {
        "marker_id": ARUCO_MARKER_ID,
        "marker_length_m": float(marker_length_m),
        "captured_at": captured_at,
        "frame_stamp_ns": None,
        "visible": False,
        "valid": False,
        "stable": False,
        "camera_info_frame_id": None,
        "camera_info_identity": None,
        "reprojection_error_px": None,
        "alternative_reprojection_error_px": None,
        "ambiguity_margin_px": None,
        "minimum_corner_depth_m": None,
        "maximum_reprojection_error_px": MAXIMUM_REPROJECTION_ERROR_PX,
        "minimum_ippe_ambiguity_margin_px": MINIMUM_IPPE_AMBIGUITY_MARGIN_PX,
        "sample_count": 0,
        "required_sample_count": STABILITY_SAMPLE_COUNT,
        "sample_started_at": None,
        "window_age_sec": None,
        "translation_spread_m": None,
        "maximum_translation_spread_m": MAXIMUM_TRANSLATION_SPREAD_M,
        "rotation_spread_deg": None,
        "maximum_rotation_spread_deg": MAXIMUM_ROTATION_SPREAD_DEG,
    }
    if aruco is not None:
        marker.update(aruco)
    return {
        "available": False,
        "success": False,
        "diagnostic_only": True,
        "message": str(message),
        "frame_id": STATIONARY_INSPECTION_FRAME_ID,
        "captured_at": captured_at,
        "geometry_path": str(geometry.get("path") or ""),
        "unsupported_part_names": list(STATIONARY_INSPECTION_UNSUPPORTED_PART_NAMES),
        "xy_tolerance_m": STATIONARY_INSPECTION_XY_TOLERANCE_M,
        "seating_tolerance_m": STATIONARY_INSPECTION_SEATING_TOLERANCE_M,
        "registration": {
            "name": STATIONARY_REGISTRATION_NAME,
            "configured": registration_configured,
            "calibration_id": str(
                geometry.get("registration_calibration_id") or ""
            ),
            "geometry_path": str(geometry.get("path") or ""),
            "transform": dict(geometry.get("registration") or {}),
        },
        "aruco": marker,
        "parts": _expected_stationary_part_evidence(
            geometry,
            detected_parts=row_parts,
        ),
    }


def _stationary_inspection_from_rows(
    rows: list[dict[str, Any]],
    *,
    geometry: dict[str, Any],
    observation: _StationaryArucoObservation,
    aruco: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate raw exact-frame SG/MG top surfaces in ``assembly_board-v1``."""
    if not bool(geometry.get("registration_configured", False)):
        return _stationary_inspection_unavailable_payload(
            str(geometry.get("registration_error") or STATIONARY_REGISTRATION_UNAVAILABLE),
            geometry=geometry,
            marker_length_m=float(aruco["marker_length_m"]),
            captured_at=observation.captured_at,
            aruco=aruco,
            rows=rows,
        )
    marker_to_board = geometry.get("marker_to_board")
    expected_top_surfaces = dict(geometry.get("expected_top_surfaces") or {})
    evidence: list[dict[str, Any]] = []
    failures: list[str] = []
    for part_name in STATIONARY_INSPECTION_PARTS:
        matching = [row for row in rows if row.get("part_name") == part_name]
        expected = expected_top_surfaces[part_name]
        expected_payload = {
            "frame_id": STATIONARY_INSPECTION_FRAME_ID,
            "x": float(expected[0]),
            "y": float(expected[1]),
            "z": float(expected[2]),
            "point": "top_surface",
        }
        if not matching:
            evidence.append(
                {
                    "part_name": part_name,
                    "detected": False,
                    "success": False,
                    "expected": expected_payload,
                    "observed": None,
                    "xy_error_m": None,
                    "seating_error_m": None,
                }
            )
            failures.append(f"{part_name} was not detected")
            continue
        if len(matching) != 1:
            raise DuplicateDetectionError(
                f"stationary inspection received multiple accepted {part_name} detections"
            )
        row = matching[0]
        camera_point = np.array(
            [row["camera_x"], row["camera_y"], row["camera_z"]],
            dtype=np.float64,
        )
        observed = transform_camera_optical_point_to_assembly_board_v1(
            camera_point,
            point_frame_id=str(row.get("camera_frame_id") or ""),
            camera_frame_id=observation.camera.frame_id,
            camera_to_aruco=observation.estimate.camera_to_aruco,
            assembly_board_v1_aruco_to_assembly_board_v1=marker_to_board,
        )
        xy_error_m = float(np.linalg.norm(observed[:2] - np.asarray(expected[:2])))
        seating_error_m = abs(float(observed[2]) - float(expected[2]))
        accepted = (
            xy_error_m <= STATIONARY_INSPECTION_XY_TOLERANCE_M
            and seating_error_m <= STATIONARY_INSPECTION_SEATING_TOLERANCE_M
        )
        part_evidence = {
            "part_name": part_name,
            "detected": True,
            "success": accepted,
            "expected": expected_payload,
            "observed": {
                "frame_id": STATIONARY_INSPECTION_FRAME_ID,
                "x": float(observed[0]),
                "y": float(observed[1]),
                "z": float(observed[2]),
                "point": "top_surface",
            },
            "xy_error_m": xy_error_m,
            "seating_error_m": seating_error_m,
            "confidence": float(row["confidence"]),
            "depth_sample_count": int(row["depth_sample_count"]),
            "depth_mad_m": float(row["depth_mad_m"]),
        }
        evidence.append(part_evidence)
        row["stationary_inspection"] = part_evidence
        if xy_error_m > STATIONARY_INSPECTION_XY_TOLERANCE_M:
            failures.append(
                f"{part_name} XY error {xy_error_m * 1000.0:.2f} mm exceeds "
                f"{STATIONARY_INSPECTION_XY_TOLERANCE_M * 1000.0:.2f} mm"
            )
        if seating_error_m > STATIONARY_INSPECTION_SEATING_TOLERANCE_M:
            failures.append(
                f"{part_name} seating error {seating_error_m * 1000.0:.2f} mm exceeds "
                f"{STATIONARY_INSPECTION_SEATING_TOLERANCE_M * 1000.0:.2f} mm"
            )
    success = not failures
    return {
        "available": True,
        "success": success,
        "diagnostic_only": True,
        "message": (
            "stationary assembly inspection passed for SG and MG."
            if success
            else "; ".join(failures) + "."
        ),
        "frame_id": STATIONARY_INSPECTION_FRAME_ID,
        "captured_at": observation.captured_at,
        "geometry_path": str(geometry.get("path") or ""),
        "unsupported_part_names": list(STATIONARY_INSPECTION_UNSUPPORTED_PART_NAMES),
        "xy_tolerance_m": STATIONARY_INSPECTION_XY_TOLERANCE_M,
        "seating_tolerance_m": STATIONARY_INSPECTION_SEATING_TOLERANCE_M,
        "registration": {
            "name": STATIONARY_REGISTRATION_NAME,
            "configured": True,
            "calibration_id": str(geometry["registration_calibration_id"]),
            "geometry_path": str(geometry.get("path") or ""),
            "transform": dict(geometry.get("registration") or {}),
        },
        "aruco": aruco,
        "parts": evidence,
    }


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Replace a JSON snapshot without exposing partial content to Gazebo."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def _atomic_write_jpeg(path: Path, image: np.ndarray) -> None:
    """Replace a JPEG without exposing a partially encoded frame."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.jpg")
    try:
        if not cv2.imwrite(
            str(temporary),
            image,
            [cv2.IMWRITE_JPEG_QUALITY, 86],
        ):
            raise RuntimeError(f"failed to encode detection preview: {path}")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def annotate_detection_frame(
    color_image: np.ndarray,
    detections: list[dict[str, Any]],
) -> np.ndarray:
    """Draw accepted gear boxes on the exact image used for inference."""
    image = np.asarray(color_image).copy()
    image_height, image_width = image.shape[:2]
    if not detections:
        cv2.rectangle(image, (8, 8), (360, 42), (30, 30, 30), -1)
        cv2.putText(
            image,
            "No accepted SG/MG detection",
            (16, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return image

    for detection in detections:
        bbox = detection.get("bbox") or {}
        center_x = float(bbox.get("center_x", 0.0))
        center_y = float(bbox.get("center_y", 0.0))
        width = max(0.0, float(bbox.get("width", 0.0)))
        height = max(0.0, float(bbox.get("height", 0.0)))
        x1 = max(0, min(image_width - 1, int(round(center_x - width / 2.0))))
        y1 = max(0, min(image_height - 1, int(round(center_y - height / 2.0))))
        x2 = max(0, min(image_width - 1, int(round(center_x + width / 2.0))))
        y2 = max(0, min(image_height - 1, int(round(center_y + height / 2.0))))
        if x2 <= x1 or y2 <= y1:
            continue
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 210, 0), 2)
        text = (
            f"{detection.get('part_name', '')} | {detection.get('label', '')} | "
            f"{float(detection.get('confidence', 0.0)) * 100.0:.1f}%"
        )
        (text_width, text_height), baseline = cv2.getTextSize(
            text,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            2,
        )
        text_y = max(text_height + baseline + 4, y1)
        background_x2 = min(image_width - 1, x1 + text_width + 8)
        cv2.rectangle(
            image,
            (x1, text_y - text_height - baseline - 4),
            (background_x2, text_y + baseline),
            (0, 120, 0),
            -1,
        )
        cv2.putText(
            image,
            text,
            (x1 + 4, text_y - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return image


def _transform_from_message(transform: Any) -> RigidTransform:
    translation = transform.transform.translation
    rotation = transform.transform.rotation
    return RigidTransform(
        translation=(float(translation.x), float(translation.y), float(translation.z)),
        quaternion=(float(rotation.x), float(rotation.y), float(rotation.z), float(rotation.w)),
    )


class RealSenseRoboflowNode:
    """Own synchronized frames, validated detections, services, and the twin snapshot."""

    def __init__(self) -> None:
        try:
            import message_filters
            import rclpy
            from cv_bridge import CvBridge
            from rclpy.callback_groups import ReentrantCallbackGroup
            from rclpy.node import Node
            from sensor_msgs.msg import CameraInfo, Image
            from std_srvs.srv import Trigger
        except ImportError as exc:
            raise RuntimeError(
                "ROS2 perception imports are unavailable; source ROS2 and install RealSense dependencies"
            ) from exc

        self._rclpy = rclpy
        self._Trigger = Trigger
        self.node: Any = Node("realsense_roboflow_perception")
        self._callback_group = ReentrantCallbackGroup()
        self._bridge = CvBridge()
        self._lock = threading.RLock()
        self._latest_frame: tuple[Any, np.ndarray, np.ndarray, ColorIntrinsics] | None = None
        self._latest_intrinsics: ColorIntrinsics | None = None
        self._latest_intrinsics_image_size: tuple[int, int] | None = None
        self._latest_camera_calibration: CameraCalibration | None = None
        self._last_rows: list[dict[str, Any]] = []
        self._last_error = "waiting for synchronized RealSense frames"
        self._roboflow_model_validated = False
        self._last_inference_latency_ms: float | None = None
        self._inference_lock = threading.Lock()

        self._declare_parameters()
        self.world_frame = str(self.node.get_parameter("world_frame").value)
        self.tool_frame = str(self.node.get_parameter("tool_frame").value)
        self.camera_link_frame = str(self.node.get_parameter("camera_link_frame").value)
        self.camera_optical_frame = str(self.node.get_parameter("camera_optical_frame").value)
        self.minimum_confidence = float(self.node.get_parameter("minimum_confidence").value)
        self.background_rate_hz = float(self.node.get_parameter("background_rate_hz").value)
        self.on_demand_inference_wait_sec = float(
            self.node.get_parameter("on_demand_inference_wait_sec").value
        )
        self.camera_role = str(self.node.get_parameter("camera_role").value)
        self._initialize_stationary_inspection()
        self.snapshot_path = Path(str(self.node.get_parameter("snapshot_path").value)).expanduser()
        self.preview_dir = (
            Path(str(self.node.get_parameter("preview_root").value)).expanduser()
            / self.camera_role
        )
        calibration_path = str(self.node.get_parameter("hand_eye_config").value)
        table_plane_path = str(self.node.get_parameter("table_plane_config").value).strip()

        self._calibration: dict[str, Any] = {}
        self._table_plane_path: Path | None = None
        self._table_plane_calibration: dict[str, Any] = {}
        self._table_surface_z_m: float | None = None
        self._table_plane_mtime_ns: int | None = None
        self._initialize_role_pose_pipeline(
            calibration_path=calibration_path,
            table_plane_path=table_plane_path,
        )
        self._detector = RoboflowGearDetector(
            RoboflowSettings.from_environment(),
            minimum_confidence=self.minimum_confidence,
        )

        self._color_subscriber = message_filters.Subscriber(
            self.node,
            Image,
            str(self.node.get_parameter("color_topic").value),
        )
        self._depth_subscriber = message_filters.Subscriber(
            self.node,
            Image,
            str(self.node.get_parameter("aligned_depth_topic").value),
        )
        self._camera_info_subscription = self.node.create_subscription(
            CameraInfo,
            str(self.node.get_parameter("camera_info_topic").value),
            self._on_camera_info,
            10,
        )
        self._synchronizer = message_filters.ApproximateTimeSynchronizer(
            [self._color_subscriber, self._depth_subscriber],
            queue_size=10,
            slop=0.08,
        )
        self._synchronizer.registerCallback(self._on_synchronized_frame)

        detect_all_service = str(self.node.get_parameter("detect_all_service").value)
        detect_part_service = str(self.node.get_parameter("detect_part_service").value)
        self.node.create_service(
            Trigger,
            detect_all_service,
            self._detect_all_service,
            callback_group=self._callback_group,
        )
        self.node.create_service(
            Trigger,
            detect_part_service,
            self._detect_part_service,
            callback_group=self._callback_group,
        )
        if self.camera_role != "stationary":
            table_plane_measurement_service = str(
                self.node.get_parameter("table_plane_measurement_service").value
            ).strip() or f"/perception/{self.camera_role}/table_plane_measurement"
            self.node.create_service(
                Trigger,
                table_plane_measurement_service,
                self._table_plane_measurement_service,
                callback_group=self._callback_group,
            )
        if self._canonical_services_enabled():
            if detect_all_service != "/detect_all":
                self.node.create_service(
                    Trigger,
                    "/detect_all",
                    self._detect_all_service,
                    callback_group=self._callback_group,
                )
            if detect_part_service != "/detect_part":
                self.node.create_service(
                    Trigger,
                    "/detect_part",
                    self._detect_part_service,
                    callback_group=self._callback_group,
                )
        if self.background_rate_hz > 0.0:
            self.node.create_timer(
                1.0 / self.background_rate_hz,
                self._background_detection,
                callback_group=self._callback_group,
            )
        self._write_snapshot([])

    def _initialize_role_pose_pipeline(
        self,
        *,
        calibration_path: str,
        table_plane_path: str,
    ) -> None:
        """Keep stationary inspection independent from calibration and TF."""
        if self.camera_role == "stationary":
            return
        self._initialize_world_pose_pipeline(
            calibration_path=calibration_path,
            table_plane_path=table_plane_path,
        )

    def _canonical_services_enabled(self) -> bool:
        """Never publish board-frame stationary rows through canonical services."""
        return self.camera_role != "stationary" and bool(
            self.node.get_parameter("publish_canonical_services").value
        )

    def _initialize_world_pose_pipeline(
        self,
        *,
        calibration_path: str,
        table_plane_path: str,
    ) -> None:
        """Load calibration and TF publishers for canonical robot-camera roles."""
        try:
            import tf2_ros
            from geometry_msgs.msg import TransformStamped
        except ImportError as exc:
            raise RuntimeError(
                "ROS2 TF perception imports are unavailable; source ROS2"
            ) from exc
        self._tf2_ros = tf2_ros
        self._TransformStamped = TransformStamped
        self._calibration = load_hand_eye_calibration(calibration_path)
        self._table_plane_path = Path(table_plane_path or calibration_path).expanduser()
        self._table_plane_calibration = (
            load_hand_eye_calibration(self._table_plane_path)
            if self._table_plane_path != Path(calibration_path).expanduser()
            else self._calibration
        )
        self._table_surface_z_m = table_surface_z_from_calibration(
            self._table_plane_calibration,
            world_frame=self.world_frame,
            required=False,
        )
        self._table_plane_mtime_ns = self._table_plane_path.stat().st_mtime_ns
        self._tf_buffer = tf2_ros.Buffer(
            cache_time=self._rclpy.duration.Duration(seconds=30.0)
        )
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self.node)
        self._static_broadcaster = tf2_ros.StaticTransformBroadcaster(self.node)
        self._publish_tool_to_camera_link()

    def _initialize_stationary_inspection(self) -> None:
        """Load stationary-only ID 70 and SG/MG inspection configuration."""
        self._stationary_aruco_observations: deque[_StationaryArucoObservation] = deque(
            maxlen=STABILITY_SAMPLE_COUNT * 2
        )
        self._stationary_aruco_error = "waiting for assembly_board-v1 ArUco ID 70"
        self._stationary_aruco_error_stamp_ns: int | None = None
        self._stationary_marker_length_m = float(
            self.node.get_parameter("assembly_board_v1_marker_length_m").value
        )
        if (
            not math.isfinite(self._stationary_marker_length_m)
            or self._stationary_marker_length_m <= 0.0
        ):
            raise ValueError("assembly_board-v1 ArUco marker length must be positive")
        self._stationary_inspection_geometry: dict[str, Any] = {}
        self._stationary_inspection_geometry_error = ""
        if self.camera_role != "stationary":
            self._last_stationary_inspection: dict[str, Any] = {}
            return
        geometry_path = str(
            self.node.get_parameter("assembly_board_v1_geometry_path").value
        )
        try:
            self._stationary_inspection_geometry = (
                _load_stationary_inspection_geometry(geometry_path)
            )
        except CalibrationError as exc:
            self._stationary_inspection_geometry_error = str(exc)
        initial_inspection_error = (
            self._stationary_inspection_geometry_error
            or str(
                self._stationary_inspection_geometry.get("registration_error")
                or "waiting for a fresh stable assembly_board-v1 ArUco ID 70 window"
            )
        )
        self._last_stationary_inspection = _stationary_inspection_unavailable_payload(
            initial_inspection_error,
            geometry=self._stationary_inspection_geometry,
            marker_length_m=self._stationary_marker_length_m,
        )

    def _declare_parameters(self) -> None:
        self.node.declare_parameter("color_topic", "/camera/camera/color/image_raw")
        self.node.declare_parameter(
            "aligned_depth_topic",
            "/camera/camera/aligned_depth_to_color/image_raw",
        )
        self.node.declare_parameter("camera_info_topic", "/camera/camera/color/camera_info")
        self.node.declare_parameter("camera_role", "ur5e")
        self.node.declare_parameter("world_frame", "world")
        self.node.declare_parameter("tool_frame", "tool0")
        self.node.declare_parameter("camera_link_frame", "camera_link")
        self.node.declare_parameter("camera_optical_frame", "camera_color_optical_frame")
        self.node.declare_parameter("minimum_confidence", 0.70)
        self.node.declare_parameter("minimum_depth_samples", 25)
        self.node.declare_parameter("maximum_depth_mad_m", 0.003)
        self.node.declare_parameter("maximum_frame_age_sec", 1.0)
        self.node.declare_parameter("background_rate_hz", 0.2)
        self.node.declare_parameter("on_demand_inference_wait_sec", 12.0)
        self.node.declare_parameter("snapshot_path", str(DEFAULT_SNAPSHOT_PATH))
        self.node.declare_parameter("preview_root", str(DEFAULT_PREVIEW_ROOT))
        self.node.declare_parameter("detect_all_service", "/perception/ur5e/detect_all")
        self.node.declare_parameter("detect_part_service", "/perception/ur5e/detect_part")
        self.node.declare_parameter("table_plane_measurement_service", "")
        self.node.declare_parameter("publish_canonical_services", True)
        self.node.declare_parameter(
            "hand_eye_config",
            os.path.expanduser(
                os.environ.get(
                    "REALSENSE_HAND_EYE_CONFIG",
                    "~/.config/cais-spade-llm/ur5e_realsense_hand_eye.yaml",
                )
            ),
        )
        self.node.declare_parameter("table_plane_config", "")
        self.node.declare_parameter(
            "assembly_board_v1_geometry_path",
            str(DEFAULT_ASSEMBLY_BOARD_V1_GEOMETRY_PATH),
        )
        self.node.declare_parameter(
            "assembly_board_v1_marker_length_m",
            DEFAULT_MARKER_LENGTH_M,
        )
        self.node.declare_parameter("target_part", "")

    def _publish_tool_to_camera_link(self) -> None:
        transform = self._calibration.get("parent_to_camera_link") or self._calibration.get(
            "tool0_to_camera_link"
        )
        if not isinstance(transform, dict):
            raise CalibrationError("calibration is missing parent_to_camera_link")
        translation = transform.get("translation", {})
        quaternion = transform.get("quaternion", {})
        message = self._TransformStamped()
        message.header.stamp = self.node.get_clock().now().to_msg()
        message.header.frame_id = self.tool_frame
        message.child_frame_id = self.camera_link_frame
        message.transform.translation.x = float(translation["x"])
        message.transform.translation.y = float(translation["y"])
        message.transform.translation.z = float(translation["z"])
        message.transform.rotation.x = float(quaternion["x"])
        message.transform.rotation.y = float(quaternion["y"])
        message.transform.rotation.z = float(quaternion["z"])
        message.transform.rotation.w = float(quaternion["w"])
        self._static_broadcaster.sendTransform(message)

    def _on_camera_info(self, info_msg: Any) -> None:
        """Cache color intrinsics independently from the image synchronizer."""
        try:
            camera = camera_calibration_from_info(info_msg)
            intrinsics = ColorIntrinsics(
                fx=float(camera.camera_matrix[0, 0]),
                fy=float(camera.camera_matrix[1, 1]),
                cx=float(camera.camera_matrix[0, 2]),
                cy=float(camera.camera_matrix[1, 2]),
            )
            image_size = (camera.width, camera.height)
        except (ArucoLocalizationError, AttributeError, IndexError, TypeError, ValueError) as exc:
            self._reject_camera_info(f"invalid RealSense CameraInfo: {exc}")
            return
        if (
            not all(np.isfinite(value) for value in vars(intrinsics).values())
            or intrinsics.fx <= 0.0
            or intrinsics.fy <= 0.0
            or image_size[0] <= 0
            or image_size[1] <= 0
        ):
            self._reject_camera_info(
                "invalid RealSense CameraInfo intrinsics or image size"
            )
            return
        camera_changed = False
        with self._lock:
            self._latest_intrinsics = intrinsics
            self._latest_intrinsics_image_size = image_size
            if (
                self._latest_camera_calibration is not None
                and self._latest_camera_calibration.identity != camera.identity
            ):
                self._stationary_aruco_observations.clear()
                camera_changed = True
            self._latest_camera_calibration = camera
        if camera_changed:
            self._set_stationary_inspection_unavailable(
                "RealSense CameraInfo changed; a fresh stable ten-frame "
                "assembly_board-v1 ArUco ID 70 window is required"
            )

    def _reject_camera_info(self, message: str) -> None:
        """Clear cached intrinsics and ID 70 authority after invalid CameraInfo."""
        with self._lock:
            self._latest_intrinsics = None
            self._latest_intrinsics_image_size = None
            self._latest_camera_calibration = None
            self._stationary_aruco_observations.clear()
            self._stationary_aruco_error = str(message)
            self._stationary_aruco_error_stamp_ns = None
        self._last_error = str(message)
        self._set_stationary_inspection_unavailable(str(message))

    def _on_synchronized_frame(self, color_msg: Any, depth_msg: Any) -> None:
        """Store one synchronized color/depth pair using the latest CameraInfo."""
        with self._lock:
            intrinsics = self._latest_intrinsics
            intrinsics_image_size = self._latest_intrinsics_image_size
            camera = self._latest_camera_calibration
        if intrinsics is None or intrinsics_image_size is None or camera is None:
            self._last_error = "waiting for RealSense CameraInfo"
            return
        color = self._bridge.imgmsg_to_cv2(color_msg, desired_encoding="bgr8")
        depth = self._bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
        color_height, color_width = color.shape[:2]
        if intrinsics_image_size != (color_width, color_height):
            message = (
                "RealSense CameraInfo dimensions do not match the color frame: "
                f"CameraInfo={intrinsics_image_size[0]}x{intrinsics_image_size[1]}, "
                f"color={color_width}x{color_height}"
            )
            with self._lock:
                self._stationary_aruco_observations.clear()
                self._stationary_aruco_error = message
                self._stationary_aruco_error_stamp_ns = None
            self._last_error = message
            self._set_stationary_inspection_unavailable(message)
            return
        depth_array = np.asarray(depth)
        if self.camera_role == "stationary" and (
            depth_array.ndim != 2
            or depth_array.shape != (color_height, color_width)
        ):
            message = (
                "stationary aligned depth dimensions do not match the exact color frame: "
                f"depth={depth_array.shape}, color={(color_height, color_width)}"
            )
            with self._lock:
                self._stationary_aruco_observations.clear()
                self._stationary_aruco_error = message
                self._stationary_aruco_error_stamp_ns = None
            self._last_error = message
            self._set_stationary_inspection_unavailable(message)
            return
        if depth_array.dtype == np.uint16:
            depth_m = depth_array.astype(np.float32) * 0.001
        else:
            depth_m = depth_array.astype(np.float32)
        if self.camera_role == "stationary":
            self._observe_stationary_aruco(
                stamp=color_msg.header.stamp,
                color=color,
                camera=camera,
            )
        with self._lock:
            first_frame = self._latest_frame is None
            self._latest_frame = (color_msg.header.stamp, color.copy(), depth_m.copy(), intrinsics)
        if first_frame:
            self._last_error = "waiting for Roboflow model inference"
            self._write_snapshot(self._last_rows)

    def _observe_stationary_aruco(
        self,
        *,
        stamp: Any,
        color: np.ndarray,
        camera: CameraCalibration,
    ) -> None:
        """Add one exact synchronized frame to the bounded stationary ID 70 window."""
        stamp_ns: int | None = None
        try:
            stamp_ns, captured_at = _ros_stamp_values(stamp)
            corners = detect_assembly_board_v1_aruco(color)
            estimate = estimate_camera_to_aruco(
                corners,
                camera,
                marker_length_m=self._stationary_marker_length_m,
            )
        except (ArucoLocalizationError, cv2.error, ValueError) as exc:
            with self._lock:
                self._stationary_aruco_observations.clear()
                self._stationary_aruco_error = str(exc)
                self._stationary_aruco_error_stamp_ns = stamp_ns
            return
        observation = _StationaryArucoObservation(
            stamp_ns=stamp_ns,
            captured_at=captured_at,
            camera=camera,
            estimate=estimate,
        )
        with self._lock:
            observations = self._stationary_aruco_observations
            if observations and (
                stamp_ns <= observations[-1].stamp_ns
                or camera.identity != observations[-1].camera.identity
            ):
                observations.clear()
            observations.append(observation)
            self._stationary_aruco_error = ""
            self._stationary_aruco_error_stamp_ns = None

    def _stationary_aruco_for_frame(
        self,
        stamp: Any,
    ) -> tuple[_StationaryArucoObservation, dict[str, Any]]:
        """Require the exact inference frame to end a fresh stable ID 70 window."""
        stamp_ns, _captured_at = _ros_stamp_values(stamp)
        with self._lock:
            observations = list(self._stationary_aruco_observations)
            error = self._stationary_aruco_error
            error_stamp_ns = self._stationary_aruco_error_stamp_ns
        if error_stamp_ns == stamp_ns:
            raise ArucoLocalizationError(error)
        exact_index = next(
            (
                index
                for index, observation in enumerate(observations)
                if observation.stamp_ns == stamp_ns
            ),
            None,
        )
        if exact_index is None:
            raise ArucoLocalizationError(
                "exact detection frame has no assembly_board-v1 ArUco ID 70 observation"
            )
        first_index = max(0, exact_index - STABILITY_SAMPLE_COUNT + 1)
        window = observations[first_index : exact_index + 1]
        quality = _stationary_aruco_window_quality(window)
        observation = observations[exact_index]
        if observation.camera.frame_id != self.camera_optical_frame:
            raise ArucoLocalizationError(
                "RealSense CameraInfo frame_id does not exactly match configured "
                f"camera_optical_frame: {observation.camera.frame_id!r} != "
                f"{self.camera_optical_frame!r}"
            )
        return observation, _stationary_aruco_evidence(
            observation,
            quality,
            marker_length_m=self._stationary_marker_length_m,
        )

    def _set_stationary_inspection_unavailable(
        self,
        message: str,
        *,
        captured_at: float | None = None,
        aruco: dict[str, Any] | None = None,
        rows: list[dict[str, Any]] | None = None,
    ) -> None:
        """Clear stationary inspection authority while retaining exact-frame evidence."""
        if getattr(self, "camera_role", "") != "stationary":
            return
        previous = getattr(self, "_last_stationary_inspection", {})
        if captured_at is None and isinstance(previous.get("captured_at"), (int, float)):
            captured_at = float(previous["captured_at"])
        if aruco is None and previous.get("captured_at") == captured_at:
            previous_aruco = previous.get("aruco")
            if isinstance(previous_aruco, dict):
                aruco = previous_aruco
        self._last_stationary_inspection = (
            _stationary_inspection_unavailable_payload(
                message,
                geometry=self._stationary_inspection_geometry,
                marker_length_m=self._stationary_marker_length_m,
                captured_at=captured_at,
                aruco=aruco,
                rows=rows,
            )
        )

    def _lookup_transform(self, target: str, source: str, stamp: Any | None = None) -> RigidTransform:
        lookup_time = self._rclpy.time.Time() if stamp is None else self._rclpy.time.Time.from_msg(stamp)
        try:
            transform = self._tf_buffer.lookup_transform(
                target,
                source,
                lookup_time,
                timeout=self._rclpy.duration.Duration(seconds=1.0),
            )
        except self._tf2_ros.TransformException as exc:
            raise CalibrationError(
                f"TF unavailable for {target} <- {source}: {exc}"
            ) from exc
        return _transform_from_message(transform)

    def _frame_copy(
        self,
        *,
        captured_after_sec: float | None = None,
        wait_timeout_sec: float = 2.0,
    ) -> tuple[Any, np.ndarray, np.ndarray, ColorIntrinsics]:
        """Return a current frame, optionally waiting for a newer capture."""
        deadline = time.monotonic() + max(0.1, float(wait_timeout_sec))
        while True:
            with self._lock:
                frame = self._latest_frame
            if frame is not None:
                stamp, color, depth_m, intrinsics = frame
                captured_at = float(stamp.sec) + float(stamp.nanosec) / 1e9
                if captured_after_sec is None or captured_at > float(captured_after_sec):
                    age = self.node.get_clock().now().nanoseconds / 1e9 - captured_at
                    maximum_age = float(
                        self.node.get_parameter("maximum_frame_age_sec").value
                    )
                    if age > maximum_age:
                        raise RuntimeError(f"RealSense frame is stale ({age:.2f} s)")
                    return stamp, color.copy(), depth_m.copy(), intrinsics
            if time.monotonic() >= deadline:
                if frame is None:
                    raise RuntimeError("no synchronized RealSense color/depth frame")
                raise RuntimeError(
                    "no synchronized RealSense color/depth frame was captured after "
                    "the detection request"
                )
            time.sleep(0.01)

    def _reload_table_plane_calibration(self) -> None:
        """Apply an accepted table-plane update without restarting perception."""
        if self.camera_role == "stationary" or self._table_plane_path is None:
            return
        try:
            mtime_ns = self._table_plane_path.stat().st_mtime_ns
        except OSError:
            return
        if mtime_ns == self._table_plane_mtime_ns:
            return
        calibration = load_hand_eye_calibration(self._table_plane_path)
        surface_z_m = table_surface_z_from_calibration(
            calibration,
            world_frame=self.world_frame,
            required=False,
        )
        self._table_plane_calibration = calibration
        self._table_surface_z_m = surface_z_m
        self._table_plane_mtime_ns = mtime_ns
        self.node.get_logger().info(
            f"Reloaded table-plane calibration: surface_z_m={surface_z_m}"
        )

    def _wait_for_stationary_tool_pose(
        self,
        *,
        timeout_sec: float = 2.0,
    ) -> RigidTransform:
        """Wait for two stable world-to-tool0 samples before inference."""
        deadline = time.monotonic() + max(0.15, float(timeout_sec))
        previous = self._lookup_transform(self.world_frame, self.tool_frame)
        consecutive_stable_samples = 0
        last_translation_m = 0.0
        last_rotation_deg = 0.0
        while True:
            time.sleep(0.15)
            current = self._lookup_transform(self.world_frame, self.tool_frame)
            last_translation_m, last_rotation_deg = pose_motion(previous, current)
            if last_translation_m <= 0.001 and last_rotation_deg <= 0.5:
                consecutive_stable_samples += 1
                if consecutive_stable_samples >= 2:
                    return current
            else:
                consecutive_stable_samples = 0
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"{self.camera_role} did not become stationary within "
                    f"{float(timeout_sec):.2f} s; last motion "
                    f"({last_translation_m * 1000.0:.2f} mm, "
                    f"{last_rotation_deg:.2f} deg); inference was not started"
                )
            previous = current

    def _write_detection_progress(self, stage: str, *, message: str = "") -> None:
        """Publish the current no-motion detection stage for the operator UI."""
        preview_dir = getattr(self, "preview_dir", None)
        if not isinstance(preview_dir, Path):
            return
        try:
            _atomic_write_json(
                preview_dir / "detection_status.json",
                {
                    "updated_at": time.time(),
                    "camera_role": self.camera_role,
                    "stage": str(stage or ""),
                    "message": str(message or ""),
                    "visual_detection_ready": False,
                    "world_pose_ready": False,
                },
            )
        except OSError as exc:
            self.node.get_logger().warning(f"Detection progress update failed: {exc}")

    def _stationary_marker_for_detection(
        self,
        stamp: Any,
        *,
        captured_at: float,
    ) -> tuple[_StationaryArucoObservation | None, dict[str, Any] | None]:
        """Freeze exact-frame ID 70 evidence before Roboflow inference."""
        if self.camera_role != "stationary":
            return None, None
        try:
            observation, aruco = self._stationary_aruco_for_frame(stamp)
        except ArucoLocalizationError as exc:
            self._set_stationary_inspection_unavailable(
                str(exc),
                captured_at=captured_at,
            )
            self._write_detection_progress("failed", message=str(exc))
            raise
        self._set_stationary_inspection_unavailable(
            "waiting for accepted SG/MG depth evidence from the exact detection frame",
            captured_at=captured_at,
            aruco=aruco,
        )
        return observation, aruco

    def _inspect_stationary_rows(
        self,
        rows: list[dict[str, Any]],
        *,
        observation: _StationaryArucoObservation | None,
        aruco: dict[str, Any] | None,
    ) -> None:
        """Attach board-relative evidence without granting robot motion authority."""
        if self.camera_role != "stationary":
            return
        if observation is None or aruco is None:
            raise ArucoLocalizationError(
                "exact detection frame has no stable assembly_board-v1 ArUco ID 70 evidence"
            )
        self._last_stationary_inspection = _stationary_inspection_from_rows(
            rows,
            geometry=self._stationary_inspection_geometry,
            observation=observation,
            aruco=aruco,
        )

    def _require_stationary_inspection_registration(self) -> None:
        """Reject stationary inference before producing any unregistered rows."""
        if self.camera_role != "stationary" or bool(
            self._stationary_inspection_geometry.get(
                "registration_configured",
                False,
            )
        ):
            return
        message = str(
            self._stationary_inspection_geometry.get("registration_error")
            or self._stationary_inspection_geometry_error
            or STATIONARY_REGISTRATION_UNAVAILABLE
        )
        self._set_stationary_inspection_unavailable(message)
        raise CalibrationError(message)

    def _prepare_detection_pose(self) -> tuple[RigidTransform | None, str]:
        """Prepare canonical world-pose checks; stationary needs no world pose."""
        if self.camera_role == "stationary":
            return None, ""
        self._reload_table_plane_calibration()
        try:
            return self._wait_for_stationary_tool_pose(), ""
        except CalibrationError as exc:
            return None, str(exc)
        except RuntimeError as exc:
            self._write_detection_progress("failed", message=str(exc))
            raise

    def _detection_camera_to_world(
        self,
        stamp: Any,
        *,
        tool_before: RigidTransform | None,
        pose_error: str,
    ) -> RigidTransform | None:
        """Validate canonical TF motion while leaving stationary TF-free."""
        if self.camera_role == "stationary":
            return None
        if pose_error:
            raise CalibrationError(pose_error)
        if tool_before is None:
            raise CalibrationError(
                f"TF unavailable for {self.world_frame} <- {self.tool_frame}"
            )
        camera_to_world = self._lookup_transform(
            self.world_frame,
            self.camera_optical_frame,
            stamp,
        )
        tool_after = self._lookup_transform(self.world_frame, self.tool_frame)
        translation_m, rotation_deg = pose_motion(tool_before, tool_after)
        if translation_m > 0.001 or rotation_deg > 0.5:
            raise RuntimeError(
                f"{self.camera_role} moved during inference "
                f"({translation_m * 1000.0:.2f} mm, {rotation_deg:.2f} deg)"
            )
        return camera_to_world

    def _detection_row(
        self,
        detection: Any,
        *,
        depth_m: np.ndarray,
        intrinsics: ColorIntrinsics,
        captured_at: float,
        stationary_observation: _StationaryArucoObservation | None,
        camera_to_world: RigidTransform | None,
        constrain_table_plane: bool,
    ) -> dict[str, Any]:
        """Build one board-relative stationary row or canonical world row."""
        estimate = robust_surface_depth(
            depth_m,
            detection,
            minimum_samples=int(
                self.node.get_parameter("minimum_depth_samples").value
            ),
            maximum_mad_m=float(
                self.node.get_parameter("maximum_depth_mad_m").value
            ),
        )
        camera_point = deproject_pixel(
            detection.center_x,
            detection.center_y,
            estimate.depth_m,
            intrinsics,
        )
        if self.camera_role == "stationary":
            if stationary_observation is None:
                raise ArucoLocalizationError(
                    "exact detection frame has no stable assembly_board-v1 "
                    "ArUco ID 70 evidence"
                )
            output_point = transform_camera_optical_point_to_assembly_board_v1(
                camera_point,
                point_frame_id=stationary_observation.camera.frame_id,
                camera_frame_id=stationary_observation.camera.frame_id,
                camera_to_aruco=stationary_observation.estimate.camera_to_aruco,
                assembly_board_v1_aruco_to_assembly_board_v1=(
                    self._stationary_inspection_geometry["marker_to_board"]
                ),
            )
            output_frame = STATIONARY_INSPECTION_FRAME_ID
            output_evidence = {
                "observed_top_surface_z": float(output_point[2]),
            }
        else:
            if camera_to_world is None:
                raise CalibrationError(
                    f"TF unavailable for {self.world_frame} <- {self.camera_optical_frame}"
                )
            observed_world_point = gear_center_world_point(
                camera_point,
                camera_to_world,
            )
            output_point = observed_world_point
            if constrain_table_plane and self._table_surface_z_m is not None:
                output_point = constrain_gear_center_to_table_plane(
                    observed_world_point,
                    self._table_surface_z_m,
                )
            output_frame = self.world_frame
            output_evidence = {
                "observed_center_z": float(observed_world_point[2]),
                "table_surface_z_m": self._table_surface_z_m,
            }
        return {
            "part_name": detection.part_name,
            "model_name": detection.model_name,
            "x": float(output_point[0]),
            "y": float(output_point[1]),
            "z": float(output_point[2]),
            "qx": 0.0,
            "qy": 0.0,
            "qz": 0.0,
            "qw": 1.0,
            "frame_id": output_frame,
            "confidence": detection.confidence,
            "captured_at": captured_at,
            "source": "realsense_roboflow",
            "model_id": self._detector.settings.model_id,
            "camera_x": float(camera_point[0]),
            "camera_y": float(camera_point[1]),
            "camera_z": float(camera_point[2]),
            "camera_frame_id": (
                stationary_observation.camera.frame_id
                if stationary_observation is not None
                else self.camera_optical_frame
            ),
            "depth_sample_count": estimate.sample_count,
            "depth_mad_m": estimate.mad_m,
            "bbox": {
                "center_x": detection.center_x,
                "center_y": detection.center_y,
                "width": detection.width,
                "height": detection.height,
            },
            **output_evidence,
        }

    def _run_detection(
        self,
        *,
        constrain_table_plane: bool = True,
        publish_executable_snapshot: bool = True,
        wait_for_inference_sec: float = 0.0,
    ) -> list[dict[str, Any]]:
        wait_sec = max(0.0, float(wait_for_inference_sec))
        if wait_sec > 0.0 and self._inference_lock.locked():
            self._write_detection_progress(
                "settling",
                message="Waiting for the active Roboflow inference before fresh /detect_all.",
            )
        acquired = (
            self._inference_lock.acquire(timeout=wait_sec)
            if wait_sec > 0.0
            else self._inference_lock.acquire(blocking=False)
        )
        if not acquired:
            if wait_sec > 0.0:
                message = (
                    "Roboflow inference did not become available within "
                    f"{wait_sec:.1f} s"
                )
                self._write_detection_progress("failed", message=message)
                raise RuntimeError(message)
            raise RuntimeError("Roboflow inference is already running")
        try:
            if publish_executable_snapshot:
                self._last_rows = []
            stationary = self.camera_role == "stationary"
            self._require_stationary_inspection_registration()
            self._write_detection_progress("settling")
            tool_before, pose_error = self._prepare_detection_pose()

            detection_requested_at = time.time()
            stamp, color, depth_m, intrinsics = self._frame_copy(
                captured_after_sec=detection_requested_at
            )
            captured_at = float(stamp.sec) + float(stamp.nanosec) / 1e9
            stationary_observation, stationary_aruco = (
                self._stationary_marker_for_detection(
                    stamp,
                    captured_at=captured_at,
                )
            )

            self._write_detection_progress("detection")
            inference_started = time.monotonic()
            detections = self._detector.detect(color)
            self._last_inference_latency_ms = (
                time.monotonic() - inference_started
            ) * 1000.0
            self._roboflow_model_validated = True
            annotations = [
                {
                    "part_name": detection.part_name,
                    "model_name": detection.model_name,
                    "label": detection.label,
                    "confidence": detection.confidence,
                    "bbox": {
                        "center_x": detection.center_x,
                        "center_y": detection.center_y,
                        "width": detection.width,
                        "height": detection.height,
                    },
                }
                for detection in detections
            ]
            self._write_detection_preview(
                color,
                annotations,
                captured_at,
                world_pose_ready=False,
                pose_error=(
                    "stationary inspection pending exact-frame SG/MG depth evidence"
                    if stationary
                    else pose_error or "world pose validation pending"
                ),
            )

            rows: list[dict[str, Any]] = []
            try:
                camera_to_world = self._detection_camera_to_world(
                    stamp,
                    tool_before=tool_before,
                    pose_error=pose_error,
                )

                for detection in detections:
                    rows.append(
                        self._detection_row(
                            detection,
                            depth_m=depth_m,
                            intrinsics=intrinsics,
                            captured_at=captured_at,
                            stationary_observation=stationary_observation,
                            camera_to_world=camera_to_world,
                            constrain_table_plane=constrain_table_plane,
                        )
                    )
                self._inspect_stationary_rows(
                    rows,
                    observation=stationary_observation,
                    aruco=stationary_aruco,
                )
            except (
                ArucoLocalizationError,
                CalibrationError,
                DepthQualityError,
                DuplicateDetectionError,
                RuntimeError,
                ValueError,
                KeyError,
            ) as exc:
                self._set_stationary_inspection_unavailable(
                    str(exc),
                    captured_at=captured_at,
                    aruco=stationary_aruco,
                    rows=rows,
                )
                self._write_detection_status(
                    annotations,
                    captured_at,
                    color,
                    world_pose_ready=False,
                    pose_error=str(exc),
                )
                raise

            if publish_executable_snapshot:
                self._last_rows = rows
                self._last_error = ""
                self._write_detection_status(
                    annotations,
                    captured_at,
                    color,
                    world_pose_ready=not stationary,
                    pose_error="",
                )
                self._write_snapshot(rows)
            else:
                self._write_detection_status(
                    annotations,
                    captured_at,
                    color,
                    world_pose_ready=False,
                    pose_error=(
                        "stationary inspection-only measurement; snapshot remains unchanged"
                        if stationary
                        else (
                            "table-plane calibration measurement only; executable world "
                            "pose remains unchanged"
                        )
                    ),
                )
            return rows
        finally:
            self._inference_lock.release()

    def _write_detection_preview(
        self,
        color: np.ndarray,
        annotations: list[dict[str, Any]],
        captured_at: float,
        *,
        world_pose_ready: bool,
        pose_error: str,
    ) -> None:
        """Write exact-frame annotations without affecting executable detection."""
        try:
            annotated = annotate_detection_frame(color, annotations)
            _atomic_write_jpeg(self.preview_dir / "detection.jpg", annotated)
            self._write_detection_status(
                annotations,
                captured_at,
                color,
                world_pose_ready=world_pose_ready,
                pose_error=pose_error,
            )
        except (OSError, RuntimeError, ValueError, cv2.error) as exc:
            self.node.get_logger().warning(f"Detection preview update failed: {exc}")

    def _write_detection_status(
        self,
        annotations: list[dict[str, Any]],
        captured_at: float,
        color: np.ndarray,
        *,
        world_pose_ready: bool,
        pose_error: str,
    ) -> None:
        """Write visual detection and executable world-pose readiness separately."""
        stationary_inspection_ready = bool(
            self.camera_role == "stationary"
            and self._last_stationary_inspection.get("available", False)
        )
        payload = {
            "updated_at": time.time(),
            "captured_at": captured_at,
            "camera_role": self.camera_role,
            "stage": (
                "completed"
                if world_pose_ready or stationary_inspection_ready
                else "failed"
            ),
            "detection_count": len(annotations),
            "detections": annotations,
            "visual_detection_ready": True,
            "world_pose_ready": bool(world_pose_ready),
            "stationary_inspection_ready": stationary_inspection_ready,
            "pose_error": str(pose_error or ""),
            "roboflow_latency_ms": self._last_inference_latency_ms,
            "image_width": int(color.shape[1]),
            "image_height": int(color.shape[0]),
        }
        if self.camera_role == "stationary":
            payload["stationary_inspection"] = self._last_stationary_inspection
        _atomic_write_json(self.preview_dir / "detection_status.json", payload)

    def _write_snapshot(self, rows: list[dict[str, Any]]) -> None:
        with self._lock:
            frame = self._latest_frame
        frame_time = None
        if frame is not None:
            stamp = frame[0]
            frame_time = float(stamp.sec) + float(stamp.nanosec) / 1e9
        validation = self._calibration.get("validation", {})
        calibration = (
            {
                "mode": "inspection_only",
                "world_pose_required": False,
                "identity": None,
            }
            if self.camera_role == "stationary"
            else {
                "identity": str(self._calibration.get("calibration_id", "unknown")),
                "median_reprojection_error_px": validation.get(
                    "median_reprojection_error_px"
                ),
                "fixed_board_translation_rms_m": validation.get(
                    "fixed_board_translation_rms_m"
                ),
                "fixed_board_rotation_rms_deg": validation.get(
                    "fixed_board_rotation_rms_deg"
                ),
            }
        )
        payload = {
            "updated_at": time.time(),
            "camera_role": self.camera_role,
            "inspection_only": self.camera_role == "stationary",
            "frame_captured_at": frame_time,
            "detections": rows,
            "last_error": self._last_error,
            "realsense_connected": frame is not None,
            "roboflow_model_configured": True,
            "roboflow_ready": self._roboflow_model_validated,
            "model_id": self._detector.settings.model_id,
            "roboflow_latency_ms": self._last_inference_latency_ms,
            "table_plane_ready": (
                self.camera_role != "stationary"
                and self._table_surface_z_m is not None
            ),
            "table_plane": (
                None
                if self.camera_role == "stationary"
                else self._table_plane_calibration.get("table_plane")
            ),
            "calibration": calibration,
            "unsupported": UNSUPPORTED_PARTS,
        }
        if self.camera_role == "stationary":
            payload["stationary_inspection"] = self._last_stationary_inspection
        _atomic_write_json(self.snapshot_path, payload)

    def _service_result(self, response: Any, *, target_part: str = "") -> Any:
        target = str(target_part or "").strip().upper()
        if (
            getattr(self, "camera_role", "") == "stationary"
            and target in STATIONARY_INSPECTION_UNSUPPORTED_PART_NAMES
        ):
            response.success = False
            response.message = f"stationary assembly inspection does not support {target}"
            return response
        if target in UNSUPPORTED_PARTS:
            response.success = False
            response.message = UNSUPPORTED_PARTS[target]
            return response
        try:
            rows = self._run_detection(
                wait_for_inference_sec=max(
                    0.0,
                    float(getattr(self, "on_demand_inference_wait_sec", 12.0)),
                )
            )
            if target:
                rows = [row for row in rows if row.get("part_name") == target]
                if not rows:
                    response.success = False
                    response.message = f"{target} was not detected"
                    return response
                response.message = json.dumps({"detected": True, **rows[0]})
            else:
                response.message = json.dumps(rows)
            response.success = True
        except (
            CalibrationError,
            DepthQualityError,
            DuplicateDetectionError,
            RoboflowConfigurationError,
            RoboflowResponseError,
            RuntimeError,
            ValueError,
            KeyError,
            OSError,
        ) as exc:
            self._set_stationary_inspection_unavailable(str(exc))
            self._last_error = str(exc)
            self._last_rows = []
            self._write_snapshot([])
            response.success = False
            response.message = str(exc)
        return response

    def _detect_all_service(self, _request: Any, response: Any) -> Any:
        return self._service_result(response)

    def _detect_part_service(self, _request: Any, response: Any) -> Any:
        target = str(self.node.get_parameter("target_part").value)
        return self._service_result(response, target_part=target)

    def _table_plane_measurement_service(self, _request: Any, response: Any) -> Any:
        """Return transformed measurements without bypassing executable pose validation."""
        try:
            rows = self._run_detection(
                constrain_table_plane=False,
                publish_executable_snapshot=False,
                wait_for_inference_sec=max(
                    0.0,
                    float(getattr(self, "on_demand_inference_wait_sec", 12.0)),
                ),
            )
            response.success = True
            response.message = json.dumps(rows)
        except (
            CalibrationError,
            DepthQualityError,
            DuplicateDetectionError,
            RoboflowConfigurationError,
            RoboflowResponseError,
            RuntimeError,
            ValueError,
            KeyError,
            OSError,
        ) as exc:
            self._set_stationary_inspection_unavailable(str(exc))
            response.success = False
            response.message = str(exc)
        return response

    def _background_detection(self) -> None:
        if self._latest_frame is None or self._inference_lock.locked():
            return
        try:
            self._run_detection()
        except (
            CalibrationError,
            DepthQualityError,
            DuplicateDetectionError,
            RoboflowConfigurationError,
            RoboflowResponseError,
            RuntimeError,
            ValueError,
            KeyError,
            OSError,
        ) as exc:
            if str(exc) == "Roboflow inference is already running":
                return
            self._set_stationary_inspection_unavailable(str(exc))
            self._last_error = str(exc)
            self._last_rows = []
            self._write_snapshot([])
            self.node.get_logger().warning(f"Background gear detection rejected: {exc}")


def main() -> None:
    """Run the physical perception node in a multithreaded ROS2 executor."""
    load_dotenv()
    logging.basicConfig(level=logging.INFO)
    try:
        import rclpy
        from rclpy.executors import MultiThreadedExecutor
    except ImportError as exc:
        raise SystemExit("ROS2 Python is unavailable; source /opt/ros/humble/setup.bash") from exc

    rclpy.init()
    wrapper: RealSenseRoboflowNode | None = None
    try:
        wrapper = RealSenseRoboflowNode()
        executor = MultiThreadedExecutor(num_threads=4)
        executor.add_node(wrapper.node)
        executor.spin()
    finally:
        if wrapper is not None:
            wrapper.node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
