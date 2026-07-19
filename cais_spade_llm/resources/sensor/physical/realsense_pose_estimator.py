"""Depth filtering, deprojection, transformation, and hand-eye loading helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .roboflow_detector import GearBoundingBox


class DepthQualityError(RuntimeError):
    """Raised when a detection has no stable gear-surface depth cluster."""


class CalibrationError(RuntimeError):
    """Raised when the required hand-eye calibration is absent or invalid."""


@dataclass(frozen=True)
class ColorIntrinsics:
    """Pinhole intrinsics for the aligned RealSense color image."""

    fx: float
    fy: float
    cx: float
    cy: float


@dataclass(frozen=True)
class RigidTransform:
    """Translation and quaternion for a child-frame pose in its parent frame."""

    translation: tuple[float, float, float]
    quaternion: tuple[float, float, float, float]


@dataclass(frozen=True)
class DepthEstimate:
    """Stable surface depth and its quality statistics."""

    depth_m: float
    sample_count: int
    mad_m: float


def load_hand_eye_calibration(path: str | Path) -> dict[str, Any]:
    """Load and validate the accepted hand-eye calibration YAML."""
    expanded = Path(path).expanduser()
    if not expanded.is_file():
        raise CalibrationError(f"hand-eye calibration not found: {expanded}")
    with expanded.open(encoding="utf-8") as stream:
        payload = yaml.safe_load(stream) or {}
    transform = payload.get("parent_to_camera_color_optical_frame") or payload.get(
        "tool0_to_camera_color_optical_frame"
    )
    validation = payload.get("validation")
    if not isinstance(transform, dict) or not isinstance(validation, dict):
        raise CalibrationError("camera calibration is missing transform or validation data")
    if not bool(validation.get("accepted", False)):
        raise CalibrationError("hand-eye calibration is not marked accepted")
    return payload


def table_surface_z_from_calibration(
    calibration: dict[str, Any],
    *,
    world_frame: str = "world",
    required: bool = True,
) -> float | None:
    """Return the accepted physical table surface height in the requested frame."""
    table_plane = calibration.get("table_plane")
    if not isinstance(table_plane, dict):
        if required:
            raise CalibrationError(
                "table-plane calibration is missing; run calibrate_hand_eye table-plane"
            )
        return None
    if not bool(table_plane.get("accepted", False)):
        raise CalibrationError("table-plane calibration is not marked accepted")
    if str(table_plane.get("world_frame") or "") != world_frame:
        raise CalibrationError(
            "table-plane calibration frame mismatch: "
            f"expected {world_frame}, received {table_plane.get('world_frame')}"
        )
    try:
        surface_z_m = float(table_plane["surface_z_m"])
        mad_m = float(table_plane["mad_m"])
        frame_count = int(table_plane["frame_count"])
    except (KeyError, TypeError, ValueError) as exc:
        raise CalibrationError("table-plane calibration is incomplete") from exc
    if not math.isfinite(surface_z_m) or not math.isfinite(mad_m):
        raise CalibrationError("table-plane calibration contains a non-finite value")
    if frame_count < 10 or mad_m > 0.002:
        raise CalibrationError(
            "table-plane calibration does not satisfy the 10-frame / 2 mm MAD contract"
        )
    return surface_z_m


def _bounded_box(
    depth_shape: tuple[int, int],
    box: GearBoundingBox,
    shrink_fraction: float,
) -> tuple[int, int, int, int]:
    height, width = depth_shape
    scale = 1.0 - float(shrink_fraction)
    half_w = max(1.0, box.width * scale * 0.5)
    half_h = max(1.0, box.height * scale * 0.5)
    x0 = max(0, int(math.floor(box.center_x - half_w)))
    x1 = min(width, int(math.ceil(box.center_x + half_w)))
    y0 = max(0, int(math.floor(box.center_y - half_h)))
    y1 = min(height, int(math.ceil(box.center_y + half_h)))
    if x1 <= x0 or y1 <= y0:
        raise DepthQualityError(f"{box.part_name} bounding box is outside the depth image")
    return x0, x1, y0, y1


def robust_surface_depth(
    depth_m: np.ndarray,
    box: GearBoundingBox,
    *,
    shrink_fraction: float = 0.10,
    minimum_samples: int = 25,
    maximum_mad_m: float = 0.003,
    cluster_bin_m: float = 0.005,
) -> DepthEstimate:
    """Select the closest populated depth cluster inside a shrunken box."""
    if depth_m.ndim != 2:
        raise DepthQualityError("aligned depth image must be two-dimensional")
    x0, x1, y0, y1 = _bounded_box(depth_m.shape, box, shrink_fraction)
    values = np.asarray(depth_m[y0:y1, x0:x1], dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values) & (values > 0.0)]
    if values.size < minimum_samples:
        raise DepthQualityError(
            f"{box.part_name} has {values.size} populated depth samples; {minimum_samples} required"
        )

    bin_ids = np.floor(values / float(cluster_bin_m)).astype(np.int64)
    unique_bins, counts = np.unique(bin_ids, return_counts=True)
    populated = unique_bins[counts >= minimum_samples]
    if populated.size == 0:
        raise DepthQualityError(f"{box.part_name} has no populated surface-depth cluster")
    closest_bin = int(np.min(populated))
    cluster = values[np.abs(bin_ids - closest_bin) <= 1]
    if cluster.size < minimum_samples:
        raise DepthQualityError(f"{box.part_name} closest depth cluster is too sparse")
    median = float(np.median(cluster))
    mad = float(np.median(np.abs(cluster - median)))
    if mad > maximum_mad_m:
        raise DepthQualityError(
            f"{box.part_name} depth MAD {mad * 1000.0:.2f} mm exceeds 3 mm"
        )
    return DepthEstimate(depth_m=median, sample_count=int(cluster.size), mad_m=mad)


def deproject_pixel(
    pixel_x: float,
    pixel_y: float,
    depth_m: float,
    intrinsics: ColorIntrinsics,
) -> np.ndarray:
    """Deproject one aligned color pixel to camera optical-frame metres."""
    if intrinsics.fx <= 0.0 or intrinsics.fy <= 0.0 or depth_m <= 0.0:
        raise ValueError("invalid camera intrinsics or depth")
    return np.array(
        [
            (float(pixel_x) - intrinsics.cx) * depth_m / intrinsics.fx,
            (float(pixel_y) - intrinsics.cy) * depth_m / intrinsics.fy,
            depth_m,
        ],
        dtype=np.float64,
    )


def quaternion_matrix(quaternion: tuple[float, float, float, float]) -> np.ndarray:
    """Return the 3x3 rotation matrix for an x/y/z/w quaternion."""
    x, y, z, w = (float(value) for value in quaternion)
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 0.0:
        raise ValueError("zero-length quaternion")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def transform_point(point: np.ndarray, transform: RigidTransform) -> np.ndarray:
    """Transform an optical-frame point into the transform parent frame."""
    translation = np.asarray(transform.translation, dtype=np.float64)
    return quaternion_matrix(transform.quaternion) @ np.asarray(point, dtype=np.float64) + translation


def gear_center_world_point(
    point_camera: np.ndarray,
    camera_to_world: RigidTransform,
    *,
    gear_height_m: float = 0.020,
) -> np.ndarray:
    """Transform the observed top point and shift it to the gear model center."""
    point_world = transform_point(point_camera, camera_to_world)
    point_world[2] -= float(gear_height_m) * 0.5
    return point_world


def constrain_gear_center_to_table_plane(
    observed_center_world: np.ndarray,
    table_surface_z_m: float,
    *,
    gear_height_m: float = 0.020,
    maximum_surface_error_m: float = 0.005,
) -> np.ndarray:
    """Validate observed Z against the table and return a plane-constrained center."""
    constrained = np.asarray(observed_center_world, dtype=np.float64).copy()
    observed_surface_z = float(constrained[2]) - float(gear_height_m) * 0.5
    surface_error_m = abs(observed_surface_z - float(table_surface_z_m))
    if surface_error_m > float(maximum_surface_error_m):
        raise CalibrationError(
            "observed gear surface disagrees with the calibrated table plane by "
            f"{surface_error_m * 1000.0:.2f} mm"
        )
    constrained[2] = float(table_surface_z_m) + float(gear_height_m) * 0.5
    return constrained


def pose_motion(
    before: RigidTransform,
    after: RigidTransform,
) -> tuple[float, float]:
    """Return translation metres and shortest quaternion rotation degrees."""
    translation = float(
        np.linalg.norm(np.asarray(after.translation) - np.asarray(before.translation))
    )
    q1 = np.asarray(before.quaternion, dtype=np.float64)
    q2 = np.asarray(after.quaternion, dtype=np.float64)
    q1 /= np.linalg.norm(q1)
    q2 /= np.linalg.norm(q2)
    angle = math.degrees(2.0 * math.acos(float(np.clip(abs(np.dot(q1, q2)), -1.0, 1.0))))
    return translation, angle
