"""Associate segmented RGB-D candidates with one approved CAD size."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from cais_spade_llm.spec2primitives.tools.exact_ref_resolver import approved_cad_path
from cais_spade_llm.spec2primitives.tools.observation_context import (
    CAMERA_IDS,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
)
from cais_spade_llm.spec2primitives.tools.rgb_d_cad_grounding.segmenter import (
    _load_source_record,
)

_GROUNDING_ROOT = Path("products/grounding/rgb_d_cad_grounding")
_PRODUCER = "rgb_d_cad_grounding"
_DIMENSION_ERROR_LIMIT = 0.15
_UNIQUENESS_MARGIN = 0.10
_MINIMUM_MEASURED_DIMENSION_M = 1e-4
_SEGMENTATION_PARAMETERS = {
    "ransac_seed": 0,
    "ransac_iterations": 256,
    "maximum_ransac_sample_points": 10_000,
    "plane_distance_threshold_m": 0.005,
    "minimum_plane_inlier_fraction": 0.2,
    "minimum_plane_inlier_points": 100,
    "depth_neighbor_threshold_m": 0.02,
    "minimum_candidate_points": 50,
    "maximum_candidates": 32,
}

_CAD_RECORD_KEYS = {
    "schema_version",
    "record_type",
    "producer",
    "operation_number",
    "evidence_type",
    "evidence_refs",
    "source",
    "coordinate_frame",
    "stored_units",
    "triangle_count",
    "vertex_count",
    "bounds_m",
    "vertex_centroid_m",
    "artifacts",
    "correspondence",
    "pose",
}
_SEGMENTATION_RECORD_KEYS = {
    "schema_version",
    "record_type",
    "producer",
    "segmentation_number",
    "observation_ref",
    "source_record",
    "parameters",
    "candidate_count",
    "candidate_state",
    "cameras",
    "cross_camera_fusion",
    "identity",
    "CAD_correspondence",
    "pose",
}
_SEGMENTATION_CAMERA_KEYS = {
    "observation_handle",
    "camera_id",
    "frame",
    "input_point_count",
    "source_point_cloud",
    "source_artifacts",
    "support_plane",
    "candidate_count",
    "candidate_state",
    "candidates",
    "label_mask_artifact",
    "identity",
    "CAD_correspondence",
    "pose",
}
_SEGMENTATION_CANDIDATE_KEYS = {
    "candidate_handle",
    "candidate_id",
    "point_count",
    "pixel_bounds_uv",
    "bounds_m",
    "centroid_m",
    "depth_range_m",
    "identity",
    "CAD_correspondence",
    "pose",
}


class CADSizeAssociationError(ValueError):
    """Raised when size association input or persistence is invalid."""


@dataclass(frozen=True)
class CADSizeAssociationResult:
    """Return one persisted size-based correspondence result."""

    record_path: Path
    CAD_correspondence: str
    location: str
    selected_candidate: Mapping[str, object] | None
    record: Mapping[str, object]


@dataclass(frozen=True)
class _CADInput:
    record_path: Path
    context_ref: str
    dimensions_m: np.ndarray
    source_sha256: str
    mesh_ref: str
    mesh_sha256: str
    triangles_m: np.ndarray


def associate_segmented_candidate_by_size(
    *,
    interaction_root: Path,
    segmentation_record_path: Path,
    cad_record_path: Path,
    correspondence_number: int = 1,
) -> CADSizeAssociationResult:
    """Associate one approved CAD size with segmented camera-local candidates.

    The result contains a measured candidate center, not a robot pick point.
    Rotation and complete pose remain unevaluated.
    """
    _validate_positive_integer(correspondence_number, "correspondence_number")
    root = Path(interaction_root).resolve()
    cad_input = _load_cad_input(root, cad_record_path)
    segmentation_path, segmentation_record, candidate_inputs = _load_candidates(
        root,
        segmentation_record_path,
    )
    destination = root / _GROUNDING_ROOT / f"correspondence_{correspondence_number:04d}"
    if destination.exists():
        raise CADSizeAssociationError(
            f"CAD size correspondence {correspondence_number:04d} already exists."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        temporary_root = Path(tempfile.mkdtemp(prefix=".correspondence-", dir=destination.parent))
    except OSError as exc:
        raise CADSizeAssociationError(
            "CAD size correspondence temporary directory could not be created."
        ) from exc

    try:
        ranked_candidates = _rank_candidates(candidate_inputs, cad_input.dimensions_m)
        correspondence, location, selected, plausible = _association_decision(ranked_candidates)
        record = {
            "schema_version": 2,
            "record_type": "CADSizeCorrespondenceRecord",
            "producer": _PRODUCER,
            "correspondence_number": correspondence_number,
            "method": "two_largest_principal_dimensions",
            "parameters": {
                "dimension_error_limit": _DIMENSION_ERROR_LIMIT,
                "uniqueness_margin": _UNIQUENESS_MARGIN,
                "minimum_measured_dimension_m": _MINIMUM_MEASURED_DIMENSION_M,
            },
            "CAD": {
                "context_ref": cad_input.context_ref,
                "record": {
                    "ref": _relative_ref(root, cad_input.record_path),
                    "sha256": _sha256_path(cad_input.record_path),
                },
                "source_sha256": cad_input.source_sha256,
                "mesh": {
                    "ref": cad_input.mesh_ref,
                    "sha256": cad_input.mesh_sha256,
                },
                "compared_dimensions_m": _float_list(cad_input.dimensions_m),
            },
            "segmentation": {
                "observation_ref": segmentation_record["observation_ref"],
                "record": {
                    "ref": _relative_ref(root, segmentation_path),
                    "sha256": _sha256_path(segmentation_path),
                },
            },
            "ranked_candidates": ranked_candidates,
            "plausible_candidates": plausible,
            "selected_candidate": selected,
            "CAD_correspondence": correspondence,
            "location": location,
            "pose": "not_evaluated",
            "cross_camera_fusion": "not_evaluated",
        }
        _write_json(temporary_root / "correspondence_record.json", record)
        if destination.exists():
            raise CADSizeAssociationError(
                f"CAD size correspondence {correspondence_number:04d} already exists."
            )
        temporary_root.rename(destination)
    except CADSizeAssociationError:
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise
    except (OSError, TypeError, ValueError, np.linalg.LinAlgError) as exc:
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise CADSizeAssociationError(
            f"CAD size correspondence failed: {type(exc).__name__}: {exc}"
        ) from exc

    return CADSizeAssociationResult(
        record_path=destination / "correspondence_record.json",
        CAD_correspondence=correspondence,
        location=location,
        selected_candidate=selected,
        record=record,
    )


def _load_cad_input(interaction_root: Path, record_path: Path) -> _CADInput:
    path, relative, record = _load_json_record(interaction_root, record_path)
    if path.name != "geometry_record.json" or set(record) != _CAD_RECORD_KEYS:
        raise CADSizeAssociationError("CAD preprocessing record fields are invalid.")
    operation_number = record["operation_number"]
    _validate_positive_integer(operation_number, "CAD operation_number")
    expected_record_ref = (
        _GROUNDING_ROOT / f"operation_{operation_number:04d}" / "geometry_record.json"
    )
    if relative != expected_record_ref:
        raise CADSizeAssociationError("CAD preprocessing record path is invalid.")
    if (
        record["schema_version"] != 1
        or record["record_type"] != "CADMeshRecord"
        or record["producer"] != _PRODUCER
        or record["evidence_type"] != "CAD"
        or record["coordinate_frame"] != "CAD_local"
        or record["stored_units"] != "m"
        or record["correspondence"] != "not_evaluated"
        or record["pose"] != "not_evaluated"
    ):
        raise CADSizeAssociationError("CAD preprocessing record identity is invalid.")

    context_ref, source_sha256 = _validated_cad_source(record)
    triangles_m, mesh_ref, mesh_sha256 = _validated_cad_mesh(
        interaction_root,
        record,
        operation_number=operation_number,
    )
    bounds_minimum = triangles_m.min(axis=(0, 1))
    bounds_maximum = triangles_m.max(axis=(0, 1))
    bounds_size = bounds_maximum - bounds_minimum
    bounds = record["bounds_m"]
    if not isinstance(bounds, dict) or set(bounds) != {"minimum", "maximum", "size"}:
        raise CADSizeAssociationError("CAD bounds metadata is invalid.")
    for key, expected in (
        ("minimum", bounds_minimum),
        ("maximum", bounds_maximum),
        ("size", bounds_size),
    ):
        value = _finite_vector(bounds[key], f"CAD bounds {key}")
        if not np.allclose(value, expected, rtol=0.0, atol=1e-7):
            raise CADSizeAssociationError("CAD bounds do not match the mesh artifact.")
    centroid = _finite_vector(record["vertex_centroid_m"], "CAD vertex centroid")
    if not np.allclose(centroid, triangles_m.mean(axis=(0, 1)), rtol=0.0, atol=1e-7):
        raise CADSizeAssociationError("CAD centroid does not match the mesh artifact.")
    dimensions = np.sort(bounds_size.astype(np.float64))[::-1][:2]
    if dimensions.shape != (2,) or np.any(dimensions <= _MINIMUM_MEASURED_DIMENSION_M):
        raise CADSizeAssociationError("CAD compared dimensions are invalid.")
    return _CADInput(
        record_path=path,
        context_ref=context_ref,
        dimensions_m=dimensions,
        source_sha256=source_sha256,
        mesh_ref=mesh_ref,
        mesh_sha256=mesh_sha256,
        triangles_m=triangles_m,
    )


def _validated_cad_source(record: Mapping[str, object]) -> tuple[str, str]:
    source = record["source"]
    if not isinstance(source, dict) or set(source) != {
        "context_ref",
        "provenance",
        "source_units",
        "source_sha256",
    }:
        raise CADSizeAssociationError("CAD source metadata is invalid.")
    context_ref = source["context_ref"]
    source_sha256 = source["source_sha256"]
    if (
        not isinstance(context_ref, str)
        or not context_ref
        or record["evidence_refs"] != [context_ref]
        or source["source_units"] != "mm"
        or not _is_sha256(source_sha256)
    ):
        raise CADSizeAssociationError("CAD source identity is invalid.")
    try:
        approved_source = approved_cad_path(context_ref)
    except (OSError, TypeError, ValueError) as exc:
        raise CADSizeAssociationError("CAD source is not an exact approved ref.") from exc
    if _sha256_path(approved_source) != source_sha256:
        raise CADSizeAssociationError("Approved CAD source hash is invalid.")
    return context_ref, source_sha256


def _validated_cad_mesh(
    interaction_root: Path,
    record: Mapping[str, object],
    *,
    operation_number: int,
) -> tuple[np.ndarray, str, str]:
    artifacts = record["artifacts"]
    if not isinstance(artifacts, dict) or set(artifacts) != {"mesh"}:
        raise CADSizeAssociationError("CAD mesh metadata is invalid.")
    mesh = artifacts["mesh"]
    if not isinstance(mesh, dict) or set(mesh) != {"ref", "sha256", "arrays"}:
        raise CADSizeAssociationError("CAD mesh artifact metadata is invalid.")
    expected_mesh_ref = _GROUNDING_ROOT / f"operation_{operation_number:04d}" / "cad_mesh.npz"
    if mesh["ref"] != str(expected_mesh_ref) or not _is_sha256(mesh["sha256"]):
        raise CADSizeAssociationError("CAD mesh artifact ref is invalid.")
    mesh_path = _resolved_ref(interaction_root, mesh["ref"])
    if not mesh_path.is_file() or _sha256_path(mesh_path) != mesh["sha256"]:
        raise CADSizeAssociationError("CAD mesh artifact hash is invalid.")
    try:
        with np.load(mesh_path, allow_pickle=False) as arrays:
            if set(arrays.files) != {"triangles_m", "facet_normals"}:
                raise CADSizeAssociationError("CAD mesh arrays are invalid.")
            triangles_m = np.array(arrays["triangles_m"], copy=True)
            facet_normals = np.array(arrays["facet_normals"], copy=True)
    except (OSError, TypeError, ValueError) as exc:
        raise CADSizeAssociationError("CAD mesh artifact could not be loaded.") from exc
    triangle_count = record["triangle_count"]
    if (
        isinstance(triangle_count, bool)
        or not isinstance(triangle_count, int)
        or triangle_count <= 0
        or triangles_m.dtype != np.float32
        or facet_normals.dtype != np.float32
        or triangles_m.shape != (triangle_count, 3, 3)
        or facet_normals.shape != (triangle_count, 3)
        or record["vertex_count"] != triangle_count * 3
        or not np.isfinite(triangles_m).all()
        or not np.isfinite(facet_normals).all()
    ):
        raise CADSizeAssociationError("CAD mesh array shape or values are invalid.")
    expected_arrays = {
        "triangles_m": {"shape": list(triangles_m.shape), "dtype": "float32"},
        "facet_normals": {"shape": list(facet_normals.shape), "dtype": "float32"},
    }
    if mesh["arrays"] != expected_arrays:
        raise CADSizeAssociationError("CAD mesh array metadata is invalid.")
    return triangles_m, str(mesh["ref"]), str(mesh["sha256"])


def _load_candidates(
    interaction_root: Path,
    record_path: Path,
) -> tuple[Path, dict[str, Any], list[dict[str, object]]]:
    path, relative, record = _load_json_record(interaction_root, record_path)
    if path.name != "segmentation_record.json" or set(record) != _SEGMENTATION_RECORD_KEYS:
        raise CADSizeAssociationError("Segmentation record fields are invalid.")
    segmentation_number = record["segmentation_number"]
    _validate_positive_integer(segmentation_number, "segmentation_number")
    expected_record_ref = (
        _GROUNDING_ROOT / f"segmentation_{segmentation_number:04d}" / "segmentation_record.json"
    )
    if relative != expected_record_ref:
        raise CADSizeAssociationError("Segmentation record path is invalid.")
    if (
        record["schema_version"] != 2
        or record["record_type"] != "RGBDSegmentationRecord"
        or record["producer"] != _PRODUCER
        or record["cross_camera_fusion"] != "not_evaluated"
        or record["identity"] != "not_evaluated"
        or record["CAD_correspondence"] != "not_evaluated"
        or record["pose"] != "not_evaluated"
        or record["parameters"] != _SEGMENTATION_PARAMETERS
    ):
        raise CADSizeAssociationError("Segmentation record identity is invalid.")
    source_record = record["source_record"]
    if not isinstance(source_record, Mapping) or set(source_record) != {"ref", "sha256"}:
        raise CADSizeAssociationError("Segmentation source record metadata is invalid.")
    source_path = _validate_hashed_ref(
        interaction_root,
        source_record,
        "Segmentation source record",
    )
    try:
        validated_source_path, validated_source = _load_source_record(
            interaction_root,
            source_path,
        )
    except (OSError, TypeError, ValueError) as exc:
        raise CADSizeAssociationError("Observation preprocessing record is invalid.") from exc
    if validated_source_path != source_path:
        raise CADSizeAssociationError("Segmentation source record path is invalid.")
    source_cameras = {
        source_camera["camera_id"]: source_camera for source_camera in validated_source["cameras"]
    }
    operation_number = int(validated_source["operation_number"])
    cameras = record["cameras"]
    if not isinstance(cameras, list) or len(cameras) != len(CAMERA_IDS):
        raise CADSizeAssociationError("Segmentation camera records are invalid.")

    candidate_inputs = []
    total_candidates = 0
    for camera_index, camera_id in enumerate(CAMERA_IDS):
        camera = cameras[camera_index]
        camera_candidates = _load_camera_candidates(
            interaction_root,
            camera,
            camera_id=camera_id,
            camera_index=camera_index,
            segmentation_number=segmentation_number,
            source_camera=source_cameras[camera_id],
            operation_number=operation_number,
        )
        candidate_inputs.extend(camera_candidates)
        total_candidates += len(camera_candidates)
    if record["candidate_count"] != total_candidates or record["candidate_state"] != (
        "unresolved" if total_candidates == 0 else "candidates_available"
    ):
        raise CADSizeAssociationError("Segmentation aggregate counts are invalid.")
    return path, record, candidate_inputs


def _load_camera_candidates(
    interaction_root: Path,
    camera: object,
    *,
    camera_id: str,
    camera_index: int,
    segmentation_number: int,
    source_camera: Mapping[str, object],
    operation_number: int,
) -> list[dict[str, object]]:
    if not isinstance(camera, dict) or set(camera) != _SEGMENTATION_CAMERA_KEYS:
        raise CADSizeAssociationError("Segmentation camera fields are invalid.")
    if (
        camera["observation_handle"] != f"view_{camera_index + 1:04d}"
        or camera["camera_id"] != camera_id
        or not isinstance(camera["frame"], str)
        or not camera["frame"]
        or camera["identity"] != "not_evaluated"
        or camera["CAD_correspondence"] != "not_evaluated"
        or camera["pose"] != "not_evaluated"
        or camera["frame"] != source_camera["frame"]
        or camera["input_point_count"] != source_camera["point_count"]
        or camera["source_artifacts"] != source_camera["source_artifacts"]
    ):
        raise CADSizeAssociationError("Segmentation camera identity is invalid.")
    source_point_cloud = camera["source_point_cloud"]
    source_artifact = source_camera["point_cloud_artifact"]
    if (
        not isinstance(source_point_cloud, Mapping)
        or set(source_point_cloud) != {"ref", "sha256"}
        or not isinstance(source_artifact, Mapping)
        or source_point_cloud.get("ref") != source_artifact.get("ref")
        or source_point_cloud.get("sha256") != source_artifact.get("sha256")
    ):
        raise CADSizeAssociationError("Segmentation point-cloud provenance is invalid.")
    _validate_source_artifacts(interaction_root, camera, camera_id=camera_id)
    points_m, pixels_uv = _load_camera_points(
        interaction_root,
        camera,
        source_artifact=source_artifact,
        operation_number=operation_number,
    )
    labels = _load_camera_labels(
        interaction_root,
        camera,
        segmentation_number=segmentation_number,
    )
    candidate_count = camera["candidate_count"]
    if (
        isinstance(candidate_count, bool)
        or not isinstance(candidate_count, int)
        or candidate_count < 0
    ):
        raise CADSizeAssociationError("Segmentation candidate count is invalid.")
    candidates = camera["candidates"]
    if not isinstance(candidates, list) or len(candidates) != candidate_count:
        raise CADSizeAssociationError("Segmentation candidates are invalid.")
    expected_candidate_state = "unresolved" if candidate_count == 0 else "candidates_available"
    if camera["candidate_state"] != expected_candidate_state:
        raise CADSizeAssociationError("Segmentation candidate state is invalid.")
    point_labels = labels[pixels_uv[:, 1], pixels_uv[:, 0]]
    if int(labels.max(initial=0)) > candidate_count:
        raise CADSizeAssociationError("Segmentation label values are invalid.")
    result = []
    for candidate_index, candidate in enumerate(candidates, start=1):
        result.append(
            _load_candidate(
                camera,
                candidate,
                camera_id=camera_id,
                camera_index=camera_index,
                candidate_index=candidate_index,
                points_m=points_m,
                pixels_uv=pixels_uv,
                point_labels=point_labels,
            )
        )
    return result


def _load_candidate(
    camera: Mapping[str, object],
    candidate: object,
    *,
    camera_id: str,
    camera_index: int,
    candidate_index: int,
    points_m: np.ndarray,
    pixels_uv: np.ndarray,
    point_labels: np.ndarray,
) -> dict[str, object]:
    if not isinstance(candidate, dict) or set(candidate) != _SEGMENTATION_CANDIDATE_KEYS:
        raise CADSizeAssociationError("Segmentation candidate fields are invalid.")
    if (
        candidate["candidate_handle"] != f"candidate_{camera_index + 1:04d}_{candidate_index:04d}"
        or candidate["candidate_id"] != candidate_index
        or candidate["identity"] != "not_evaluated"
        or candidate["CAD_correspondence"] != "not_evaluated"
        or candidate["pose"] != "not_evaluated"
    ):
        raise CADSizeAssociationError("Segmentation candidate identity is invalid.")
    candidate_points = points_m[point_labels == candidate_index]
    candidate_pixels = pixels_uv[point_labels == candidate_index]
    point_count = candidate["point_count"]
    if (
        isinstance(point_count, bool)
        or not isinstance(point_count, int)
        or point_count <= 0
        or candidate_points.shape != (point_count, 3)
    ):
        raise CADSizeAssociationError("Segmentation candidate points are invalid.")
    partial_visibility = _validate_candidate_summary(
        candidate,
        candidate_points,
        candidate_pixels,
    )
    return {
        "observation_handle": camera["observation_handle"],
        "camera_id": camera_id,
        "camera_order": camera_index,
        "frame": camera["frame"],
        "candidate_handle": candidate["candidate_handle"],
        "candidate_id": candidate_index,
        "point_count": point_count,
        "points_m": candidate_points,
        "partial_visibility": partial_visibility,
    }


def _validate_source_artifacts(
    interaction_root: Path,
    camera: Mapping[str, object],
    *,
    camera_id: str,
) -> None:
    artifacts = camera["source_artifacts"]
    if not isinstance(artifacts, Mapping) or set(artifacts) != {"rgb", "depth"}:
        raise CADSizeAssociationError("Segmentation source artifact metadata is invalid.")
    expected_names = {
        "rgb": f"{camera_id}_rgb.png",
        "depth": f"{camera_id}_depth_m.npy",
    }
    for kind, expected_name in expected_names.items():
        path = _validate_hashed_ref(
            interaction_root,
            artifacts[kind],
            f"Segmentation {kind} source",
        )
        if path.name != expected_name:
            raise CADSizeAssociationError("Segmentation source artifact ref is invalid.")


def _load_camera_points(
    interaction_root: Path,
    camera: Mapping[str, object],
    *,
    source_artifact: Mapping[str, object],
    operation_number: int,
) -> tuple[np.ndarray, np.ndarray]:
    source = camera["source_point_cloud"]
    point_cloud_path = _validate_hashed_ref(
        interaction_root,
        source,
        "Segmentation point cloud",
    )
    expected_ref = (
        _GROUNDING_ROOT
        / f"operation_{operation_number:04d}"
        / f"{camera['camera_id']}_point_cloud.npz"
    )
    if source.get("ref") != str(expected_ref):
        raise CADSizeAssociationError("Segmentation point-cloud ref is invalid.")
    try:
        with np.load(point_cloud_path, allow_pickle=False) as arrays:
            if set(arrays.files) != {"points_m", "colors_rgb", "pixels_uv"}:
                raise CADSizeAssociationError("Segmentation point-cloud arrays are invalid.")
            points_m = np.array(arrays["points_m"], copy=True)
            colors_rgb = np.array(arrays["colors_rgb"], copy=True)
            pixels_uv = np.array(arrays["pixels_uv"], copy=True)
    except (OSError, TypeError, ValueError) as exc:
        raise CADSizeAssociationError("Segmentation point cloud could not be loaded.") from exc
    input_count = camera["input_point_count"]
    expected_arrays = {
        "points_m": {"shape": [input_count, 3], "dtype": "float32"},
        "colors_rgb": {"shape": [input_count, 3], "dtype": "uint8"},
        "pixels_uv": {"shape": [input_count, 2], "dtype": "uint16"},
    }
    if (
        isinstance(input_count, bool)
        or not isinstance(input_count, int)
        or input_count <= 0
        or points_m.dtype != np.float32
        or colors_rgb.dtype != np.uint8
        or pixels_uv.dtype != np.uint16
        or points_m.shape != (input_count, 3)
        or colors_rgb.shape != (input_count, 3)
        or pixels_uv.shape != (input_count, 2)
        or not np.isfinite(points_m).all()
        or np.any(points_m[:, 2] <= 0)
        or np.any(pixels_uv[:, 0] >= IMAGE_WIDTH)
        or np.any(pixels_uv[:, 1] >= IMAGE_HEIGHT)
        or np.unique(pixels_uv, axis=0).shape[0] != input_count
        or source_artifact.get("arrays") != expected_arrays
    ):
        raise CADSizeAssociationError("Segmentation point-cloud values are invalid.")
    return points_m, pixels_uv


def _load_camera_labels(
    interaction_root: Path,
    camera: Mapping[str, object],
    *,
    segmentation_number: int,
) -> np.ndarray:
    artifact = camera["label_mask_artifact"]
    if not isinstance(artifact, dict) or set(artifact) != {
        "ref",
        "sha256",
        "shape",
        "dtype",
    }:
        raise CADSizeAssociationError("Segmentation label metadata is invalid.")
    camera_id = camera["camera_id"]
    expected_ref = (
        _GROUNDING_ROOT
        / f"segmentation_{segmentation_number:04d}"
        / f"{camera_id}_candidate_labels.npy"
    )
    if (
        artifact["ref"] != str(expected_ref)
        or artifact["shape"] != [IMAGE_HEIGHT, IMAGE_WIDTH]
        or artifact["dtype"] != "uint16"
    ):
        raise CADSizeAssociationError("Segmentation label artifact ref is invalid.")
    label_path = _validate_hashed_ref(interaction_root, artifact, "Segmentation label mask")
    try:
        labels = np.load(label_path, allow_pickle=False)
    except (OSError, TypeError, ValueError) as exc:
        raise CADSizeAssociationError("Segmentation label mask could not be loaded.") from exc
    if labels.dtype != np.uint16 or labels.shape != (IMAGE_HEIGHT, IMAGE_WIDTH):
        raise CADSizeAssociationError("Segmentation label mask values are invalid.")
    return labels


def _validate_candidate_summary(
    candidate: Mapping[str, object],
    points_m: np.ndarray,
    pixels_uv: np.ndarray,
) -> bool:
    bounds = candidate["bounds_m"]
    if not isinstance(bounds, Mapping) or set(bounds) != {"minimum", "maximum"}:
        raise CADSizeAssociationError("Segmentation candidate bounds are invalid.")
    minimum = _finite_vector(bounds["minimum"], "candidate minimum")
    maximum = _finite_vector(bounds["maximum"], "candidate maximum")
    centroid = _finite_vector(candidate["centroid_m"], "candidate centroid")
    pixel_bounds = candidate["pixel_bounds_uv"]
    depth_range = candidate["depth_range_m"]
    if (
        not isinstance(pixel_bounds, Mapping)
        or set(pixel_bounds) != {"minimum", "maximum"}
        or not isinstance(depth_range, Mapping)
        or set(depth_range) != {"minimum", "maximum"}
    ):
        raise CADSizeAssociationError("Segmentation candidate summary is invalid.")
    pixel_minimum = _integer_pixel(pixel_bounds["minimum"], "candidate pixel minimum")
    pixel_maximum = _integer_pixel(pixel_bounds["maximum"], "candidate pixel maximum")
    expected_pixel_minimum = pixels_uv.min(axis=0)
    expected_pixel_maximum = pixels_uv.max(axis=0)
    depth_minimum = _finite_number(depth_range["minimum"], "candidate depth minimum")
    depth_maximum = _finite_number(depth_range["maximum"], "candidate depth maximum")
    if (
        not np.allclose(minimum, points_m.min(axis=0), rtol=0.0, atol=1e-6)
        or not np.allclose(maximum, points_m.max(axis=0), rtol=0.0, atol=1e-6)
        or not np.allclose(
            centroid,
            points_m.mean(axis=0, dtype=np.float64),
            rtol=0.0,
            atol=1e-6,
        )
        or not np.array_equal(pixel_minimum, expected_pixel_minimum)
        or not np.array_equal(pixel_maximum, expected_pixel_maximum)
        or not math.isclose(depth_minimum, float(points_m[:, 2].min()), abs_tol=1e-6)
        or not math.isclose(depth_maximum, float(points_m[:, 2].max()), abs_tol=1e-6)
    ):
        raise CADSizeAssociationError("Segmentation candidate summary is inconsistent.")
    return bool(
        pixel_minimum[0] == 0
        or pixel_minimum[1] == 0
        or pixel_maximum[0] == IMAGE_WIDTH - 1
        or pixel_maximum[1] == IMAGE_HEIGHT - 1
    )


def _rank_candidates(
    candidates: list[dict[str, object]],
    cad_dimensions_m: np.ndarray,
) -> list[dict[str, object]]:
    ranked = []
    for candidate in candidates:
        points_m = candidate["points_m"]
        if not isinstance(points_m, np.ndarray):
            raise CADSizeAssociationError("Candidate point array is invalid.")
        partial_visibility = candidate["partial_visibility"] is True
        measurement = None if partial_visibility else _measure_candidate(points_m)
        if measurement is None:
            dimensions = None
            errors = None
            score = None
            measurement_status = "partial_visibility" if partial_visibility else "unreliable"
            within_tolerance = False
            center = _float_list(np.median(points_m, axis=0))
        else:
            dimensions_array, center_array = measurement
            errors_array = np.abs(dimensions_array - cad_dimensions_m) / cad_dimensions_m
            dimensions = _float_list(dimensions_array)
            errors = _float_list(errors_array)
            score = float(errors_array.mean())
            measurement_status = "measured"
            within_tolerance = bool(np.all(errors_array <= _DIMENSION_ERROR_LIMIT))
            center = _float_list(center_array)
        ranked.append(
            {
                "observation_handle": candidate["observation_handle"],
                "camera_id": candidate["camera_id"],
                "camera_order": candidate["camera_order"],
                "frame": candidate["frame"],
                "candidate_handle": candidate["candidate_handle"],
                "candidate_id": candidate["candidate_id"],
                "point_count": candidate["point_count"],
                "measurement_status": measurement_status,
                "observed_dimensions_m": dimensions,
                "dimension_errors": errors,
                "mean_dimension_error": score,
                "within_size_tolerance": within_tolerance,
                "candidate_center_m": center,
            }
        )
    ranked.sort(
        key=lambda item: (
            item["mean_dimension_error"] is None,
            math.inf if item["mean_dimension_error"] is None else item["mean_dimension_error"],
            item["camera_order"],
            item["candidate_id"],
        )
    )
    for rank, candidate in enumerate(ranked, start=1):
        candidate["rank"] = rank
        candidate.pop("camera_order")
    return ranked


def _measure_candidate(points_m: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    if points_m.shape[0] < 3:
        return None
    center = np.median(points_m.astype(np.float64), axis=0)
    centered = points_m.astype(np.float64) - points_m.mean(axis=0, dtype=np.float64)
    covariance = centered.T @ centered / points_m.shape[0]
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    projected = centered @ eigenvectors[:, order]
    dimensions = np.sort(np.ptp(projected, axis=0))[::-1][:2]
    if (
        dimensions.shape != (2,)
        or not np.isfinite(dimensions).all()
        or dimensions[1] < _MINIMUM_MEASURED_DIMENSION_M
    ):
        return None
    return dimensions, center


def _association_decision(
    ranked_candidates: list[dict[str, object]],
) -> tuple[str, str, dict[str, object] | None, list[dict[str, object]]]:
    reliable = [
        candidate
        for candidate in ranked_candidates
        if isinstance(candidate["mean_dimension_error"], float)
    ]
    plausible = [candidate for candidate in reliable if candidate["within_size_tolerance"]]
    if not plausible:
        return "rejected", "unavailable", None, []
    best = plausible[0]
    if len(reliable) > 1:
        best_score = float(best["mean_dimension_error"])
        second_score = float(reliable[1]["mean_dimension_error"])
        if second_score - best_score < _UNIQUENESS_MARGIN:
            return "ambiguous", "ambiguous", None, plausible
    selected = {
        "observation_handle": best["observation_handle"],
        "camera_id": best["camera_id"],
        "frame": best["frame"],
        "candidate_handle": best["candidate_handle"],
        "candidate_id": best["candidate_id"],
        "candidate_center_m": best["candidate_center_m"],
        "observed_dimensions_m": best["observed_dimensions_m"],
        "dimension_errors": best["dimension_errors"],
        "mean_dimension_error": best["mean_dimension_error"],
    }
    return "accepted", "available", selected, plausible


def _load_json_record(
    interaction_root: Path,
    record_path: Path,
) -> tuple[Path, Path, dict[str, Any]]:
    path = Path(record_path).resolve()
    try:
        relative = path.relative_to(interaction_root)
    except ValueError as exc:
        raise CADSizeAssociationError("Geometry record is outside the interaction root.") from exc
    if relative.parts[:3] != _GROUNDING_ROOT.parts or not path.is_file():
        raise CADSizeAssociationError("Geometry record path is invalid.")
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CADSizeAssociationError("Geometry record could not be read.") from exc
    if not isinstance(record, dict):
        raise CADSizeAssociationError("Geometry record must be an object.")
    return path, relative, record


def _validate_hashed_ref(
    interaction_root: Path,
    value: object,
    label: str,
) -> Path:
    if not isinstance(value, Mapping) or not {"ref", "sha256"}.issubset(value):
        raise CADSizeAssociationError(f"{label} metadata is invalid.")
    if not _is_sha256(value["sha256"]):
        raise CADSizeAssociationError(f"{label} hash is invalid.")
    path = _resolved_ref(interaction_root, value["ref"])
    if not path.is_file() or _sha256_path(path) != value["sha256"]:
        raise CADSizeAssociationError(f"{label} hash is invalid.")
    return path


def _resolved_ref(root: Path, value: object) -> Path:
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise CADSizeAssociationError("Geometry artifact ref is invalid.")
    path = (root / value).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise CADSizeAssociationError("Geometry artifact is outside the interaction root.") from exc
    return path


def _relative_ref(root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError as exc:
        raise CADSizeAssociationError("Geometry artifact is outside the interaction root.") from exc


def _validate_positive_integer(value: object, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise CADSizeAssociationError(f"{label} must be a positive integer.")


def _finite_vector(value: object, label: str) -> np.ndarray:
    if not isinstance(value, list) or len(value) != 3:
        raise CADSizeAssociationError(f"{label} must contain three values.")
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise CADSizeAssociationError(f"{label} is invalid.") from exc
    if not np.isfinite(result).all():
        raise CADSizeAssociationError(f"{label} is non-finite.")
    return result


def _integer_pixel(value: object, label: str) -> np.ndarray:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
    ):
        raise CADSizeAssociationError(f"{label} must contain two integers.")
    result = np.asarray(value, dtype=np.int64)
    if result[0] < 0 or result[0] >= IMAGE_WIDTH or result[1] < 0 or result[1] >= IMAGE_HEIGHT:
        raise CADSizeAssociationError(f"{label} is outside the image.")
    return result


def _finite_number(value: object, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise CADSizeAssociationError(f"{label} must be finite.")
    return float(value)


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _float_list(values: np.ndarray) -> list[float]:
    result = [float(value) for value in values]
    if not all(math.isfinite(value) for value in result):
        raise CADSizeAssociationError("Derived size association value is non-finite.")
    return result


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
