"""Segment preprocessed RGB-D point clouds into unlabeled geometry candidates."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import tempfile
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from cais_spade_llm.spec2primitives.tools.observation_context import (
    CAMERA_IDS,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
)

_GROUNDING_ROOT = Path("products/grounding/rgb_d_cad_grounding")
_PRODUCER = "rgb_d_cad_grounding"

_RANSAC_SEED = 0
_RANSAC_ITERATIONS = 256
_MAXIMUM_RANSAC_SAMPLE_POINTS = 10_000
_PLANE_DISTANCE_THRESHOLD_M = 0.005
_MINIMUM_PLANE_INLIER_FRACTION = 0.20
_MINIMUM_PLANE_INLIER_POINTS = 100
_DEPTH_NEIGHBOR_THRESHOLD_M = 0.020
_MINIMUM_CANDIDATE_POINTS = 50
_MAXIMUM_CANDIDATES = 32

_SOURCE_RECORD_KEYS = {
    "schema_version",
    "record_type",
    "producer",
    "operation_number",
    "evidence_type",
    "evidence_refs",
    "observation_ref",
    "evidence_label",
    "source_manifest",
    "coordinate_convention",
    "stored_units",
    "cross_camera_fusion",
    "extrinsics_available",
    "cameras",
    "correspondence",
    "pose",
}
_CAMERA_RECORD_KEYS = {
    "camera_id",
    "frame",
    "rgb_frame",
    "depth_frame",
    "rgb_timestamp_ns",
    "depth_timestamp_ns",
    "point_count",
    "depth_range_m",
    "projection",
    "source_artifacts",
    "point_cloud_artifact",
}


class RGBDSegmentationError(ValueError):
    """Raised when RGB-D segmentation input or persistence is invalid."""


@dataclass(frozen=True)
class RGBDSegmentationResult:
    """Return one persisted segmentation record and compact candidate counts."""

    record_path: Path
    artifact_paths: tuple[Path, ...]
    candidate_count: int
    record: Mapping[str, object]


@dataclass(frozen=True)
class _Plane:
    normal: np.ndarray
    offset: float
    inliers: np.ndarray
    rms_distance_m: float


def segment_preprocessed_observation(
    *,
    interaction_root: Path,
    observation_record_path: Path,
    segmentation_number: int = 1,
) -> RGBDSegmentationResult:
    """Segment one validated Phase 4.2A observation record atomically.

    Every observation view uses the same geometry-only policy. A dominant
    plane removes the support surface and points behind it; it does not assign
    a product-state role to any retained candidate.
    """
    _validate_segmentation_number(segmentation_number)
    root = Path(interaction_root).resolve()
    source_path, source_record = _load_source_record(root, observation_record_path)
    destination = root / _GROUNDING_ROOT / f"segmentation_{segmentation_number:04d}"
    if destination.exists():
        raise RGBDSegmentationError(f"RGB-D segmentation {segmentation_number:04d} already exists.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        temporary_root = Path(tempfile.mkdtemp(prefix=".segmentation-", dir=destination.parent))
    except OSError as exc:
        raise RGBDSegmentationError(
            "RGB-D segmentation temporary directory could not be created."
        ) from exc

    try:
        camera_records = []
        artifact_names = []
        candidate_count = 0
        source_cameras = {camera["camera_id"]: camera for camera in source_record["cameras"]}
        for observation_index, camera_id in enumerate(CAMERA_IDS, start=1):
            camera_record = source_cameras[camera_id]
            points_m, pixels_uv, artifact_path = _load_point_cloud(
                root,
                camera_record,
                operation_number=int(source_record["operation_number"]),
            )
            labels, candidates, plane_record = _segment_camera(
                observation_index,
                points_m,
                pixels_uv,
            )
            artifact_name = f"{camera_id}_candidate_labels.npy"
            label_path = temporary_root / artifact_name
            np.save(label_path, labels, allow_pickle=False)
            view_candidate_count = len(candidates)
            candidate_count += view_candidate_count
            camera_records.append(
                {
                    "observation_handle": f"view_{observation_index:04d}",
                    "camera_id": camera_id,
                    "frame": camera_record["frame"],
                    "input_point_count": int(points_m.shape[0]),
                    "source_point_cloud": {
                        "ref": _relative_ref(root, artifact_path),
                        "sha256": _sha256_path(artifact_path),
                    },
                    "source_artifacts": camera_record["source_artifacts"],
                    "support_plane": plane_record,
                    "candidate_count": view_candidate_count,
                    "candidate_state": (
                        "unresolved" if view_candidate_count == 0 else "candidates_available"
                    ),
                    "candidates": candidates,
                    "label_mask_artifact": {
                        "ref": _relative_ref(root, destination / artifact_name),
                        "sha256": _sha256_path(label_path),
                        "shape": [IMAGE_HEIGHT, IMAGE_WIDTH],
                        "dtype": "uint16",
                    },
                    "identity": "not_evaluated",
                    "CAD_correspondence": "not_evaluated",
                    "pose": "not_evaluated",
                }
            )
            artifact_names.append(artifact_name)

        record = {
            "schema_version": 2,
            "record_type": "RGBDSegmentationRecord",
            "producer": _PRODUCER,
            "segmentation_number": segmentation_number,
            "observation_ref": source_record["observation_ref"],
            "source_record": {
                "ref": _relative_ref(root, source_path),
                "sha256": _sha256_path(source_path),
            },
            "parameters": _parameter_record(),
            "candidate_count": candidate_count,
            "candidate_state": ("unresolved" if candidate_count == 0 else "candidates_available"),
            "cameras": camera_records,
            "cross_camera_fusion": "not_evaluated",
            "identity": "not_evaluated",
            "CAD_correspondence": "not_evaluated",
            "pose": "not_evaluated",
        }
        _write_json(temporary_root / "segmentation_record.json", record)
        if destination.exists():
            raise RGBDSegmentationError(
                f"RGB-D segmentation {segmentation_number:04d} already exists."
            )
        temporary_root.rename(destination)
    except RGBDSegmentationError:
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise
    except (OSError, TypeError, ValueError, np.linalg.LinAlgError) as exc:
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise RGBDSegmentationError(
            f"RGB-D segmentation failed: {type(exc).__name__}: {exc}"
        ) from exc

    return RGBDSegmentationResult(
        record_path=destination / "segmentation_record.json",
        artifact_paths=tuple(destination / name for name in artifact_names),
        candidate_count=candidate_count,
        record=record,
    )


def _load_source_record(
    interaction_root: Path,
    record_path: Path,
) -> tuple[Path, dict[str, Any]]:
    path = Path(record_path).resolve()
    try:
        relative = path.relative_to(interaction_root)
    except ValueError as exc:
        raise RGBDSegmentationError(
            "Observation preprocessing record is outside the interaction root."
        ) from exc
    if not relative.parts or Path(*relative.parts[:3]) != Path(
        "products/grounding/rgb_d_cad_grounding"
    ):
        raise RGBDSegmentationError(
            "Observation preprocessing record is outside the geometry product directory."
        )
    if path.name != "geometry_record.json" or not path.is_file():
        raise RGBDSegmentationError("Observation preprocessing record is missing.")
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RGBDSegmentationError("Observation preprocessing record could not be read.") from exc
    if not isinstance(record, dict) or set(record) != _SOURCE_RECORD_KEYS:
        raise RGBDSegmentationError("Observation preprocessing record fields are invalid.")
    if (
        record["schema_version"] != 1
        or record["record_type"] != "ColoredPointCloudSetRecord"
        or record["producer"] != _PRODUCER
        or record["evidence_type"] != "observation"
        or record["stored_units"] != "m"
        or record["coordinate_convention"] != "+x right, +y down, +z forward"
        or record["extrinsics_available"] is not False
        or record["cross_camera_fusion"] != "not_evaluated"
        or record["correspondence"] != "not_evaluated"
        or record["pose"] != "not_evaluated"
    ):
        raise RGBDSegmentationError("Observation preprocessing record identity is invalid.")
    operation_number = record["operation_number"]
    if (
        isinstance(operation_number, bool)
        or not isinstance(operation_number, int)
        or operation_number <= 0
        or relative
        != _GROUNDING_ROOT / f"operation_{operation_number:04d}" / "geometry_record.json"
    ):
        raise RGBDSegmentationError("Observation preprocessing record path is invalid.")
    observation_ref = record["observation_ref"]
    if (
        not isinstance(observation_ref, str)
        or not observation_ref
        or Path(observation_ref).name != observation_ref
        or record["evidence_refs"] != [observation_ref]
        or not isinstance(record["evidence_label"], str)
        or record["evidence_label"] not in {"fixture", "replay", "live"}
    ):
        raise RGBDSegmentationError("Observation preprocessing evidence identity is invalid.")
    source_manifest = record["source_manifest"]
    expected_manifest_ref = Path("products/observations") / observation_ref / "manifest.json"
    if (
        not isinstance(source_manifest, Mapping)
        or set(source_manifest) != {"ref", "sha256"}
        or source_manifest["ref"] != str(expected_manifest_ref)
    ):
        raise RGBDSegmentationError("Observation source manifest metadata is invalid.")
    manifest_path = _resolved_ref(interaction_root, source_manifest["ref"])
    if not manifest_path.is_file() or source_manifest["sha256"] != _sha256_path(manifest_path):
        raise RGBDSegmentationError("Observation source manifest hash is invalid.")
    cameras = record["cameras"]
    if not isinstance(cameras, list) or len(cameras) != len(CAMERA_IDS):
        raise RGBDSegmentationError("Observation preprocessing cameras are invalid.")
    camera_ids = []
    for camera in cameras:
        if not isinstance(camera, dict) or set(camera) != _CAMERA_RECORD_KEYS:
            raise RGBDSegmentationError("Observation camera record fields are invalid.")
        camera_ids.append(camera.get("camera_id"))
    if tuple(camera_ids) != CAMERA_IDS:
        raise RGBDSegmentationError("Observation camera record order is invalid.")
    return path, record


def _load_point_cloud(
    interaction_root: Path,
    camera_record: Mapping[str, object],
    *,
    operation_number: int,
) -> tuple[np.ndarray, np.ndarray, Path]:
    artifact = camera_record["point_cloud_artifact"]
    if not isinstance(artifact, Mapping) or set(artifact) != {"ref", "sha256", "arrays"}:
        raise RGBDSegmentationError("Point-cloud artifact metadata is invalid.")
    artifact_path = _resolved_ref(interaction_root, artifact["ref"])
    camera_id = camera_record["camera_id"]
    expected_ref = (
        _GROUNDING_ROOT / f"operation_{operation_number:04d}" / f"{camera_id}_point_cloud.npz"
    )
    if artifact["ref"] != str(expected_ref):
        raise RGBDSegmentationError("Point-cloud artifact ref is invalid.")
    if not artifact_path.is_file() or artifact["sha256"] != _sha256_path(artifact_path):
        raise RGBDSegmentationError("Point-cloud artifact hash is invalid.")
    source_artifacts = camera_record["source_artifacts"]
    if not isinstance(source_artifacts, Mapping) or set(source_artifacts) != {"rgb", "depth"}:
        raise RGBDSegmentationError("Observation source artifact metadata is invalid.")
    expected_names = {
        "rgb": f"{camera_id}_rgb.png",
        "depth": f"{camera_id}_depth_m.npy",
    }
    for kind, expected_name in expected_names.items():
        source = source_artifacts[kind]
        if not isinstance(source, Mapping) or set(source) != {"ref", "sha256"}:
            raise RGBDSegmentationError("Observation source artifact metadata is invalid.")
        source_path = _resolved_ref(interaction_root, source["ref"])
        source_relative = Path(str(source["ref"]))
        if (
            source_path.name != expected_name
            or source_relative.parts[:2] != ("products", "observations")
            or not source_path.is_file()
            or source["sha256"] != _sha256_path(source_path)
        ):
            raise RGBDSegmentationError("Observation source artifact hash is invalid.")
    try:
        with np.load(artifact_path, allow_pickle=False) as arrays:
            if set(arrays.files) != {"points_m", "colors_rgb", "pixels_uv"}:
                raise RGBDSegmentationError("Point-cloud arrays are invalid.")
            points_m = np.array(arrays["points_m"], copy=True)
            colors_rgb = np.array(arrays["colors_rgb"], copy=True)
            pixels_uv = np.array(arrays["pixels_uv"], copy=True)
    except (OSError, TypeError, ValueError) as exc:
        raise RGBDSegmentationError("Point-cloud artifact could not be loaded.") from exc
    point_count = camera_record["point_count"]
    if isinstance(point_count, bool) or not isinstance(point_count, int) or point_count < 1:
        raise RGBDSegmentationError("Point-cloud point count is invalid.")
    expected_arrays = {
        "points_m": {"shape": [point_count, 3], "dtype": "float32"},
        "colors_rgb": {"shape": [point_count, 3], "dtype": "uint8"},
        "pixels_uv": {"shape": [point_count, 2], "dtype": "uint16"},
    }
    if (
        artifact["arrays"] != expected_arrays
        or points_m.dtype != np.float32
        or colors_rgb.dtype != np.uint8
        or pixels_uv.dtype != np.uint16
        or points_m.ndim != 2
        or points_m.shape[1:] != (3,)
        or colors_rgb.shape != points_m.shape
        or pixels_uv.shape != (points_m.shape[0], 2)
        or point_count != points_m.shape[0]
        or not np.isfinite(points_m).all()
        or np.any(points_m[:, 2] <= 0)
    ):
        raise RGBDSegmentationError("Point-cloud array shape or values are invalid.")
    if (
        np.any(pixels_uv[:, 0] >= IMAGE_WIDTH)
        or np.any(pixels_uv[:, 1] >= IMAGE_HEIGHT)
        or np.unique(pixels_uv, axis=0).shape[0] != pixels_uv.shape[0]
    ):
        raise RGBDSegmentationError("Point-cloud pixels are invalid.")
    return points_m, pixels_uv, artifact_path


def _segment_camera(
    observation_index: int,
    points_m: np.ndarray,
    pixels_uv: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, object]], dict[str, object]]:
    plane = _dominant_plane(points_m)
    if plane is None:
        eligible_indices = np.empty(0, dtype=np.int64)
        plane_record: dict[str, object] = {
            "status": "unavailable",
            "candidate_filtering_applied": False,
            "retained_point_count": 0,
        }
    else:
        signed_distances = points_m @ plane.normal + plane.offset
        camera_side_sign = -1.0 if plane.offset < 0.0 else 1.0
        eligible_indices = np.flatnonzero(
            signed_distances * camera_side_sign > _PLANE_DISTANCE_THRESHOLD_M
        ).astype(np.int64)
        plane_record = {
            "status": "detected",
            "candidate_filtering_applied": True,
            "candidate_side": "camera_side",
            "retained_point_count": int(eligible_indices.size),
            "normal": _float_list(plane.normal),
            "offset_m": float(plane.offset),
            "inlier_count": int(np.count_nonzero(plane.inliers)),
            "inlier_fraction": float(np.mean(plane.inliers)),
            "rms_distance_m": plane.rms_distance_m,
        }

    clusters = _depth_connected_clusters(points_m, pixels_uv, eligible_indices)
    ordered = sorted(
        clusters,
        key=lambda indices: (
            -int(indices.size),
            int(pixels_uv[indices, 1].min()),
            int(pixels_uv[indices, 0].min()),
        ),
    )[:_MAXIMUM_CANDIDATES]
    labels = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.uint16)
    candidates = []
    for candidate_id, indices in enumerate(ordered, start=1):
        candidate_points = points_m[indices]
        candidate_pixels = pixels_uv[indices]
        labels[candidate_pixels[:, 1], candidate_pixels[:, 0]] = candidate_id
        candidates.append(
            {
                "candidate_handle": (f"candidate_{observation_index:04d}_{candidate_id:04d}"),
                "candidate_id": candidate_id,
                "point_count": int(indices.size),
                "pixel_bounds_uv": {
                    "minimum": [
                        int(candidate_pixels[:, 0].min()),
                        int(candidate_pixels[:, 1].min()),
                    ],
                    "maximum": [
                        int(candidate_pixels[:, 0].max()),
                        int(candidate_pixels[:, 1].max()),
                    ],
                },
                "bounds_m": {
                    "minimum": _float_list(candidate_points.min(axis=0)),
                    "maximum": _float_list(candidate_points.max(axis=0)),
                },
                "centroid_m": _float_list(candidate_points.mean(axis=0, dtype=np.float64)),
                "depth_range_m": {
                    "minimum": float(candidate_points[:, 2].min()),
                    "maximum": float(candidate_points[:, 2].max()),
                },
                "identity": "not_evaluated",
                "CAD_correspondence": "not_evaluated",
                "pose": "not_evaluated",
            }
        )
    return labels, candidates, plane_record


def _dominant_plane(points_m: np.ndarray) -> _Plane | None:
    if points_m.shape[0] < max(3, _MINIMUM_PLANE_INLIER_POINTS):
        return None
    rng = np.random.default_rng(_RANSAC_SEED)
    sample_size = min(points_m.shape[0], _MAXIMUM_RANSAC_SAMPLE_POINTS)
    if sample_size == points_m.shape[0]:
        sample = points_m
    else:
        sample = points_m[rng.choice(points_m.shape[0], size=sample_size, replace=False)]
    best_inliers: np.ndarray | None = None
    best_count = 0
    for _ in range(_RANSAC_ITERATIONS):
        indices = rng.choice(sample.shape[0], size=3, replace=False)
        first, second, third = sample[indices].astype(np.float64)
        normal = np.cross(second - first, third - first)
        length = float(np.linalg.norm(normal))
        if not math.isfinite(length) or length <= 1e-12:
            continue
        normal /= length
        offset = -float(np.dot(normal, first))
        inliers = np.abs(sample @ normal + offset) <= _PLANE_DISTANCE_THRESHOLD_M
        count = int(np.count_nonzero(inliers))
        if count > best_count:
            best_inliers = inliers
            best_count = count
    if (
        best_inliers is None
        or best_count < min(_MINIMUM_PLANE_INLIER_POINTS, sample_size)
        or best_count / sample_size < _MINIMUM_PLANE_INLIER_FRACTION
    ):
        return None
    inlier_points = sample[best_inliers].astype(np.float64)
    center = inlier_points.mean(axis=0)
    _, _, right_vectors = np.linalg.svd(inlier_points - center, full_matrices=False)
    normal = right_vectors[-1]
    normal /= np.linalg.norm(normal)
    dominant_axis = int(np.argmax(np.abs(normal)))
    if normal[dominant_axis] < 0:
        normal = -normal
    offset = -float(np.dot(normal, center))
    distances = np.abs(points_m @ normal + offset)
    inliers = distances <= _PLANE_DISTANCE_THRESHOLD_M
    inlier_count = int(np.count_nonzero(inliers))
    if (
        inlier_count < _MINIMUM_PLANE_INLIER_POINTS
        or inlier_count / points_m.shape[0] < _MINIMUM_PLANE_INLIER_FRACTION
    ):
        return None
    rms_distance_m = float(np.sqrt(np.mean(np.square(distances[inliers]))))
    return _Plane(
        normal=normal,
        offset=offset,
        inliers=inliers,
        rms_distance_m=rms_distance_m,
    )


def _depth_connected_clusters(
    points_m: np.ndarray,
    pixels_uv: np.ndarray,
    eligible_indices: np.ndarray,
) -> list[np.ndarray]:
    if eligible_indices.size == 0:
        return []
    point_indices = np.full((IMAGE_HEIGHT, IMAGE_WIDTH), -1, dtype=np.int32)
    eligible_pixels = pixels_uv[eligible_indices]
    point_indices[eligible_pixels[:, 1], eligible_pixels[:, 0]] = eligible_indices
    visited = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=bool)
    clusters = []
    for start_index in eligible_indices:
        start_u, start_v = (int(value) for value in pixels_uv[start_index])
        if visited[start_v, start_u]:
            continue
        visited[start_v, start_u] = True
        queue = deque([(start_u, start_v)])
        cluster = []
        while queue:
            u, v = queue.popleft()
            point_index = int(point_indices[v, u])
            cluster.append(point_index)
            depth = float(points_m[point_index, 2])
            for neighbor_v in range(max(0, v - 1), min(IMAGE_HEIGHT, v + 2)):
                for neighbor_u in range(max(0, u - 1), min(IMAGE_WIDTH, u + 2)):
                    if visited[neighbor_v, neighbor_u] or (neighbor_u == u and neighbor_v == v):
                        continue
                    neighbor_index = int(point_indices[neighbor_v, neighbor_u])
                    if neighbor_index < 0:
                        continue
                    if (
                        abs(float(points_m[neighbor_index, 2]) - depth)
                        > _DEPTH_NEIGHBOR_THRESHOLD_M
                    ):
                        continue
                    visited[neighbor_v, neighbor_u] = True
                    queue.append((neighbor_u, neighbor_v))
        if len(cluster) >= _MINIMUM_CANDIDATE_POINTS:
            clusters.append(np.asarray(cluster, dtype=np.int64))
    return clusters


def _parameter_record() -> dict[str, object]:
    return {
        "ransac_seed": _RANSAC_SEED,
        "ransac_iterations": _RANSAC_ITERATIONS,
        "maximum_ransac_sample_points": _MAXIMUM_RANSAC_SAMPLE_POINTS,
        "plane_distance_threshold_m": _PLANE_DISTANCE_THRESHOLD_M,
        "minimum_plane_inlier_fraction": _MINIMUM_PLANE_INLIER_FRACTION,
        "minimum_plane_inlier_points": _MINIMUM_PLANE_INLIER_POINTS,
        "depth_neighbor_threshold_m": _DEPTH_NEIGHBOR_THRESHOLD_M,
        "minimum_candidate_points": _MINIMUM_CANDIDATE_POINTS,
        "maximum_candidates": _MAXIMUM_CANDIDATES,
    }


def _validate_segmentation_number(value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RGBDSegmentationError("segmentation_number must be a positive integer.")


def _resolved_ref(root: Path, value: object) -> Path:
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise RGBDSegmentationError("Geometry artifact ref is invalid.")
    path = (root / value).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise RGBDSegmentationError("Geometry artifact is outside the interaction root.") from exc
    return path


def _relative_ref(root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError as exc:
        raise RGBDSegmentationError("Geometry artifact is outside the interaction root.") from exc


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _float_list(values: np.ndarray) -> list[float]:
    result = [float(value) for value in values]
    if not all(math.isfinite(value) for value in result):
        raise RGBDSegmentationError("Derived segmentation geometry is non-finite.")
    return result


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
