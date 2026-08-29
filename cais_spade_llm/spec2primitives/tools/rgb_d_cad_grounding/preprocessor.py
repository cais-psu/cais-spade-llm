"""Preprocess approved CAD and RGB-D evidence into typed geometry records."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import struct
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from cais_spade_llm.spec2primitives.tools.exact_ref_resolver import (
    approved_cad_path,
)
from cais_spade_llm.spec2primitives.tools.observation_context import (
    CAMERA_IDS,
    CameraCalibration,
    CameraObservation,
    ObservationContextError,
    read_observation_bundle,
)

_GROUNDING_ROOT = Path("products/grounding/rgb_d_cad_grounding")
_OBSERVATIONS_ROOT = Path("products/observations")
_PRODUCER = "rgb_d_cad_grounding"
_SUPPORTED_DISTORTION_MODELS = {"plumb_bob", "rational_polynomial"}
_SERVED_BOUNDS_ATOL_MM = 5e-7
_CAD_CONTEXT_KEYS = {"context_ref", "evidence_type", "provenance", "CAD_evidence"}
_OBSERVATION_CONTEXT_KEYS = {
    "context_ref",
    "observation_ref",
    "evidence_type",
    "evidence_label",
    "provenance",
    "observation_evidence",
}
_PROVENANCE_KEYS = {"repository_path", "source_url"}
_CAD_EVIDENCE_KEYS = {
    "filename",
    "units",
    "source_sha256",
    "triangle_count",
    "bounds_mm",
}
_OBSERVATION_PROVENANCE_KEYS = {"manifest_path"}
_OBSERVATION_EVIDENCE_KEYS = {"manifest", "artifact_references"}
_ARTIFACT_REFERENCE_KEYS = {"camera_id", "rgb_artifact", "depth_artifact"}


class GeometryPreprocessingError(ValueError):
    """Raised when CAD or RGB-D evidence cannot be preprocessed safely."""


@dataclass(frozen=True)
class GeometryPreprocessingResult:
    """Return one typed record and its untrusted generic delta."""

    evidence_type: str
    evidence_ref: str
    record_path: Path
    artifact_paths: tuple[Path, ...]
    record: Mapping[str, object]
    delta: Mapping[str, object]


def preprocess_served_geometry(
    *,
    interaction_root: Path,
    served_context: Mapping[str, object],
    operation_number: int,
) -> GeometryPreprocessingResult:
    """Preprocess one exact served CAD or observation context atomically.

    The resulting delta contains typed context refs and unresolved records only.
    It never proposes factual RDF assertions, correspondence, or a pose.
    """
    _validate_operation_number(operation_number)
    root = Path(interaction_root).resolve()
    destination = root / _GROUNDING_ROOT / f"operation_{operation_number:04d}"
    if destination.exists():
        raise GeometryPreprocessingError(
            f"Geometry preprocessing operation {operation_number:04d} already exists."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        temporary_root = Path(tempfile.mkdtemp(prefix=".preprocessing-", dir=destination.parent))
    except OSError as exc:
        raise GeometryPreprocessingError(
            "Geometry preprocessing temporary directory could not be created."
        ) from exc

    try:
        evidence_type = served_context.get("evidence_type")
        if evidence_type == "CAD":
            evidence_ref, record, artifact_names = _preprocess_cad(
                served_context,
                temporary_root=temporary_root,
                destination=destination,
                interaction_root=root,
                operation_number=operation_number,
            )
        elif evidence_type == "observation":
            evidence_ref, record, artifact_names = _preprocess_observation(
                served_context,
                temporary_root=temporary_root,
                destination=destination,
                interaction_root=root,
                operation_number=operation_number,
            )
        else:
            raise GeometryPreprocessingError(
                "Geometry preprocessing accepts only CAD or observation evidence."
            )
        record_path = temporary_root / "geometry_record.json"
        _write_json(record_path, record)
        if destination.exists():
            raise GeometryPreprocessingError(
                f"Geometry preprocessing operation {operation_number:04d} already exists."
            )
        temporary_root.rename(destination)
    except GeometryPreprocessingError:
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise
    except (
        OSError,
        ObservationContextError,
        TypeError,
        ValueError,
        cv2.error,
    ) as exc:
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise GeometryPreprocessingError(
            f"Geometry preprocessing failed: {type(exc).__name__}: {exc}"
        ) from exc

    final_record_path = destination / "geometry_record.json"
    final_artifacts = tuple(destination / name for name in artifact_names)
    record_ref = _relative_ref(root, final_record_path)
    delta = {
        "assertions": [],
        "uncertainty": [],
        "unresolved_evidence_needs": [
            {
                "description": ("CAD correspondence has not been evaluated in Phase 4.2A."),
                "evidence_refs": [evidence_ref],
            },
            {
                "description": "Object pose has not been evaluated in Phase 4.2A.",
                "evidence_refs": [evidence_ref],
            },
        ],
        "typed_context_refs": [record_ref],
    }
    return GeometryPreprocessingResult(
        evidence_type=evidence_type,
        evidence_ref=evidence_ref,
        record_path=final_record_path,
        artifact_paths=final_artifacts,
        record=record,
        delta=delta,
    )


def _preprocess_cad(
    served_context: Mapping[str, object],
    *,
    temporary_root: Path,
    destination: Path,
    interaction_root: Path,
    operation_number: int,
) -> tuple[str, dict[str, object], tuple[str, ...]]:
    if set(served_context) != _CAD_CONTEXT_KEYS:
        raise GeometryPreprocessingError("Served CAD context fields are invalid.")
    context_ref = _nonempty_string(served_context["context_ref"], "CAD context_ref")
    provenance = served_context["provenance"]
    if (
        not isinstance(provenance, Mapping)
        or set(provenance) != _PROVENANCE_KEYS
        or not all(isinstance(value, str) and value for value in provenance.values())
    ):
        raise GeometryPreprocessingError("Served CAD provenance is invalid.")
    evidence = served_context["CAD_evidence"]
    if not isinstance(evidence, Mapping) or set(evidence) != _CAD_EVIDENCE_KEYS:
        raise GeometryPreprocessingError("Served CAD evidence fields are invalid.")
    if evidence["filename"] != context_ref or evidence["units"] != "mm":
        raise GeometryPreprocessingError("Served CAD identity or units are invalid.")

    source_path = approved_cad_path(context_ref)
    source_bytes = source_path.read_bytes()
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    if evidence["source_sha256"] != source_sha256:
        raise GeometryPreprocessingError(
            "Approved CAD source changed after its exact ref was served."
        )
    vertices_mm, facet_normals, triangle_count = _load_binary_stl(source_bytes)
    if evidence["triangle_count"] != triangle_count:
        raise GeometryPreprocessingError(
            "Served CAD triangle count does not match the approved mesh."
        )
    bounds_minimum_mm = vertices_mm.min(axis=(0, 1))
    bounds_maximum_mm = vertices_mm.max(axis=(0, 1))
    _validate_served_bounds(
        evidence["bounds_mm"],
        bounds_minimum_mm,
        bounds_maximum_mm,
    )

    triangles_m = np.asarray(vertices_mm * np.float32(0.001), dtype=np.float32)
    bounds_minimum = triangles_m.min(axis=(0, 1))
    bounds_maximum = triangles_m.max(axis=(0, 1))
    bounds_size = bounds_maximum - bounds_minimum

    artifact_name = "cad_mesh.npz"
    artifact_path = temporary_root / artifact_name
    np.savez_compressed(
        artifact_path,
        triangles_m=triangles_m,
        facet_normals=facet_normals,
    )
    artifact_ref = _relative_ref(interaction_root, destination / artifact_name)
    record = {
        "schema_version": 1,
        "record_type": "CADMeshRecord",
        "producer": _PRODUCER,
        "operation_number": operation_number,
        "evidence_type": "CAD",
        "evidence_refs": [context_ref],
        "source": {
            "context_ref": context_ref,
            "provenance": dict(provenance),
            "source_units": "mm",
            "source_sha256": source_sha256,
        },
        "coordinate_frame": "CAD_local",
        "stored_units": "m",
        "triangle_count": triangle_count,
        "vertex_count": triangle_count * 3,
        "bounds_m": {
            "minimum": _float_list(bounds_minimum),
            "maximum": _float_list(bounds_maximum),
            "size": _float_list(bounds_size),
        },
        "vertex_centroid_m": _float_list(triangles_m.mean(axis=(0, 1))),
        "artifacts": {
            "mesh": {
                "ref": artifact_ref,
                "sha256": _sha256_path(artifact_path),
                "arrays": {
                    "triangles_m": {
                        "shape": list(triangles_m.shape),
                        "dtype": str(triangles_m.dtype),
                    },
                    "facet_normals": {
                        "shape": list(facet_normals.shape),
                        "dtype": str(facet_normals.dtype),
                    },
                },
            }
        },
        "correspondence": "not_evaluated",
        "pose": "not_evaluated",
    }
    return context_ref, record, (artifact_name,)


def _preprocess_observation(
    served_context: Mapping[str, object],
    *,
    temporary_root: Path,
    destination: Path,
    interaction_root: Path,
    operation_number: int,
) -> tuple[str, dict[str, object], tuple[str, ...]]:
    if set(served_context) != _OBSERVATION_CONTEXT_KEYS:
        raise GeometryPreprocessingError("Served observation context fields are invalid.")
    if served_context["context_ref"] is not None:
        raise GeometryPreprocessingError("Observation context_ref must remain null.")
    observation_ref = _nonempty_string(
        served_context["observation_ref"],
        "observation_ref",
    )
    evidence_label = _nonempty_string(
        served_context["evidence_label"],
        "observation evidence_label",
    )
    manifest_path, manifest_ref = _validated_manifest_path(
        interaction_root,
        observation_ref,
        served_context["provenance"],
    )
    observation_evidence = served_context["observation_evidence"]
    if (
        not isinstance(observation_evidence, Mapping)
        or set(observation_evidence) != _OBSERVATION_EVIDENCE_KEYS
    ):
        raise GeometryPreprocessingError("Served observation evidence fields are invalid.")
    try:
        persisted_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GeometryPreprocessingError("Observation manifest could not be read.") from exc
    if observation_evidence["manifest"] != persisted_manifest:
        raise GeometryPreprocessingError(
            "Served observation manifest does not match persisted evidence."
        )
    bundle = read_observation_bundle(manifest_path.parent)
    if bundle.observation_ref != observation_ref or bundle.evidence_label != evidence_label:
        raise GeometryPreprocessingError(
            "Served observation identity does not match the persisted bundle."
        )
    artifact_references = _validated_artifact_references(
        interaction_root,
        manifest_path.parent,
        observation_evidence["artifact_references"],
    )

    camera_records: list[dict[str, object]] = []
    artifact_names: list[str] = []
    for camera_id in CAMERA_IDS:
        observation = next(
            value for value in bundle.camera_observations if value.camera_id == camera_id
        )
        artifact_name = f"{camera_id}_point_cloud.npz"
        artifact_path = temporary_root / artifact_name
        points_m, colors_rgb, pixels_uv = _deproject_camera(observation)
        np.savez_compressed(
            artifact_path,
            points_m=points_m,
            colors_rgb=colors_rgb,
            pixels_uv=pixels_uv,
        )
        source_refs = artifact_references[camera_id]
        camera_records.append(
            _camera_record(
                observation,
                points_m=points_m,
                artifact_path=artifact_path,
                artifact_ref=_relative_ref(
                    interaction_root,
                    destination / artifact_name,
                ),
                source_refs=source_refs,
            )
        )
        artifact_names.append(artifact_name)

    record = {
        "schema_version": 1,
        "record_type": "ColoredPointCloudSetRecord",
        "producer": _PRODUCER,
        "operation_number": operation_number,
        "evidence_type": "observation",
        "evidence_refs": [observation_ref],
        "observation_ref": observation_ref,
        "evidence_label": evidence_label,
        "source_manifest": {
            "ref": manifest_ref,
            "sha256": _sha256_path(manifest_path),
        },
        "coordinate_convention": "+x right, +y down, +z forward",
        "stored_units": "m",
        "cross_camera_fusion": "not_evaluated",
        "extrinsics_available": False,
        "cameras": camera_records,
        "correspondence": "not_evaluated",
        "pose": "not_evaluated",
    }
    return observation_ref, record, tuple(artifact_names)


def _load_binary_stl(source_bytes: bytes) -> tuple[np.ndarray, np.ndarray, int]:
    if len(source_bytes) < 84:
        raise GeometryPreprocessingError("Binary STL header is incomplete.")
    triangle_count = struct.unpack_from("<I", source_bytes, 80)[0]
    if triangle_count == 0 or len(source_bytes) != 84 + triangle_count * 50:
        raise GeometryPreprocessingError("Binary STL triangle layout is invalid.")
    facet_dtype = np.dtype(
        [
            ("normal", "<f4", (3,)),
            ("vertices", "<f4", (3, 3)),
            ("attribute", "<u2"),
        ]
    )
    facets = np.frombuffer(
        source_bytes,
        dtype=facet_dtype,
        count=triangle_count,
        offset=84,
    )
    vertices_mm = np.array(facets["vertices"], dtype=np.float32, copy=True)
    facet_normals = np.array(facets["normal"], dtype=np.float32, copy=True)
    if not np.isfinite(vertices_mm).all() or not np.isfinite(facet_normals).all():
        raise GeometryPreprocessingError("Binary STL contains non-finite geometry.")
    return vertices_mm, facet_normals, triangle_count


def _deproject_camera(
    observation: CameraObservation,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    calibration = observation.camera_calibration
    _validate_projection_calibration(calibration)
    valid_depth = np.isfinite(observation.depth_m) & (observation.depth_m > 0)
    rows, columns = np.nonzero(valid_depth)
    if rows.size == 0:
        raise GeometryPreprocessingError(f"{observation.camera_id} contains no valid metric depth.")
    pixels = np.column_stack((columns, rows)).astype(np.float32)
    camera_matrix = np.asarray(calibration.K, dtype=np.float64).reshape(3, 3)
    distortion = np.asarray(calibration.D, dtype=np.float64)
    normalized = cv2.undistortPoints(
        pixels.reshape(-1, 1, 2),
        camera_matrix,
        distortion,
    ).reshape(-1, 2)
    depths = observation.depth_m[rows, columns].astype(np.float32, copy=False)
    points_m = np.column_stack(
        (
            normalized[:, 0] * depths,
            normalized[:, 1] * depths,
            depths,
        )
    ).astype(np.float32, copy=False)
    if not np.isfinite(points_m).all():
        raise GeometryPreprocessingError(
            f"{observation.camera_id} deprojection produced non-finite points."
        )
    colors_rgb = np.array(observation.rgb[rows, columns], dtype=np.uint8, copy=True)
    pixels_uv = np.column_stack((columns, rows)).astype(np.uint16)
    return points_m, colors_rgb, pixels_uv


def _validate_projection_calibration(calibration: CameraCalibration) -> None:
    if calibration.distortion_model not in _SUPPORTED_DISTORTION_MODELS:
        raise GeometryPreprocessingError(
            f"Unsupported distortion model: {calibration.distortion_model}"
        )
    expected_lengths = {
        "plumb_bob": {4, 5},
        "rational_polynomial": {8, 12, 14},
    }
    if len(calibration.D) not in expected_lengths[calibration.distortion_model]:
        raise GeometryPreprocessingError(
            "Camera distortion coefficients do not match the distortion model."
        )
    camera_matrix = np.asarray(calibration.K, dtype=np.float64).reshape(3, 3)
    if (
        not np.isfinite(camera_matrix).all()
        or camera_matrix[0, 0] <= 0
        or camera_matrix[1, 1] <= 0
        or not math.isclose(camera_matrix[2, 2], 1.0)
    ):
        raise GeometryPreprocessingError("Camera intrinsic matrix is invalid.")
    if calibration.extrinsics_available is not False:
        raise GeometryPreprocessingError(
            "Phase 4.2A requires cross-camera extrinsics to remain unavailable."
        )


def _camera_record(
    observation: CameraObservation,
    *,
    points_m: np.ndarray,
    artifact_path: Path,
    artifact_ref: str,
    source_refs: Mapping[str, Path],
) -> dict[str, object]:
    calibration = observation.camera_calibration
    return {
        "camera_id": observation.camera_id,
        "frame": calibration.frame,
        "rgb_frame": observation.rgb_frame,
        "depth_frame": observation.depth_frame,
        "rgb_timestamp_ns": observation.rgb_timestamp_ns,
        "depth_timestamp_ns": observation.depth_timestamp_ns,
        "point_count": int(points_m.shape[0]),
        "depth_range_m": {
            "minimum": float(points_m[:, 2].min()),
            "maximum": float(points_m[:, 2].max()),
        },
        "projection": {
            "distortion_model": calibration.distortion_model,
            "D": list(calibration.D),
            "K": list(calibration.K),
            "R": list(calibration.R),
            "P": list(calibration.P),
            "depth_registered_to_rgb": observation.depth_registered_to_rgb,
            "extrinsics_available": calibration.extrinsics_available,
        },
        "source_artifacts": {
            "rgb": {
                "ref": _relative_ref(source_refs["root"], source_refs["rgb"]),
                "sha256": _sha256_path(source_refs["rgb"]),
            },
            "depth": {
                "ref": _relative_ref(source_refs["root"], source_refs["depth"]),
                "sha256": _sha256_path(source_refs["depth"]),
            },
        },
        "point_cloud_artifact": {
            "ref": artifact_ref,
            "sha256": _sha256_path(artifact_path),
            "arrays": {
                "points_m": {"shape": list(points_m.shape), "dtype": "float32"},
                "colors_rgb": {"shape": [points_m.shape[0], 3], "dtype": "uint8"},
                "pixels_uv": {"shape": [points_m.shape[0], 2], "dtype": "uint16"},
            },
        },
    }


def _validated_manifest_path(
    interaction_root: Path,
    observation_ref: str,
    provenance: object,
) -> tuple[Path, str]:
    if not isinstance(provenance, Mapping) or set(provenance) != _OBSERVATION_PROVENANCE_KEYS:
        raise GeometryPreprocessingError("Observation provenance fields are invalid.")
    manifest_ref = _nonempty_string(provenance["manifest_path"], "manifest_path")
    if Path(manifest_ref).is_absolute() or ".." in Path(manifest_ref).parts:
        raise GeometryPreprocessingError("Observation manifest ref is unsafe.")
    expected_ref = _OBSERVATIONS_ROOT / observation_ref / "manifest.json"
    if Path(manifest_ref) != expected_ref:
        raise GeometryPreprocessingError("Observation manifest ref does not match observation_ref.")
    manifest_path = (interaction_root / manifest_ref).resolve()
    try:
        manifest_path.relative_to((interaction_root / _OBSERVATIONS_ROOT).resolve())
    except ValueError as exc:
        raise GeometryPreprocessingError(
            "Observation manifest is outside its permitted directory."
        ) from exc
    if not manifest_path.is_file():
        raise GeometryPreprocessingError("Observation manifest is missing.")
    return manifest_path, manifest_ref


def _validated_artifact_references(
    interaction_root: Path,
    bundle_path: Path,
    value: object,
) -> dict[str, dict[str, Path]]:
    if not isinstance(value, list) or len(value) != len(CAMERA_IDS):
        raise GeometryPreprocessingError("Observation artifact references are invalid.")
    result: dict[str, dict[str, Path]] = {}
    for item in value:
        if not isinstance(item, Mapping) or set(item) != _ARTIFACT_REFERENCE_KEYS:
            raise GeometryPreprocessingError("Camera artifact reference fields are invalid.")
        camera_id = item["camera_id"]
        if camera_id not in CAMERA_IDS or camera_id in result:
            raise GeometryPreprocessingError("Camera artifact reference identity is invalid.")
        rgb_ref = _nonempty_string(item["rgb_artifact"], "rgb_artifact")
        depth_ref = _nonempty_string(item["depth_artifact"], "depth_artifact")
        expected_rgb = bundle_path / f"{camera_id}_rgb.png"
        expected_depth = bundle_path / f"{camera_id}_depth_m.npy"
        rgb_path = (interaction_root / rgb_ref).resolve()
        depth_path = (interaction_root / depth_ref).resolve()
        if rgb_path != expected_rgb.resolve() or depth_path != expected_depth.resolve():
            raise GeometryPreprocessingError(
                "Camera artifact refs do not match the persisted observation."
            )
        if not rgb_path.is_file() or not depth_path.is_file():
            raise GeometryPreprocessingError("Observation artifact is missing.")
        result[str(camera_id)] = {
            "root": interaction_root,
            "rgb": rgb_path,
            "depth": depth_path,
        }
    if set(result) != set(CAMERA_IDS):
        raise GeometryPreprocessingError("All exact camera artifact refs are required.")
    return result


def _validate_served_bounds(
    value: object,
    minimum_mm: np.ndarray,
    maximum_mm: np.ndarray,
) -> None:
    if not isinstance(value, Mapping) or set(value) != {"minimum", "maximum", "size"}:
        raise GeometryPreprocessingError("Served CAD bounds are invalid.")
    try:
        served_minimum = np.asarray(value["minimum"], dtype=np.float64)
        served_maximum = np.asarray(value["maximum"], dtype=np.float64)
        served_size = np.asarray(value["size"], dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise GeometryPreprocessingError("Served CAD bounds are invalid.") from exc
    if any(
        array.shape != (3,) or not np.isfinite(array).all()
        for array in (
            served_minimum,
            served_maximum,
            served_size,
        )
    ):
        raise GeometryPreprocessingError("Served CAD bounds are invalid.")
    approved_minimum = np.asarray(minimum_mm, dtype=np.float64)
    approved_maximum = np.asarray(maximum_mm, dtype=np.float64)
    approved_size = approved_maximum - approved_minimum
    # The resolver serializes source-unit STL bounds to six decimal places.
    # Validate before float32 metre conversion so its rounding cannot reject
    # metadata that still matches the exact approved source mesh.
    if not (
        np.allclose(
            served_minimum,
            approved_minimum,
            rtol=0.0,
            atol=_SERVED_BOUNDS_ATOL_MM,
        )
        and np.allclose(
            served_maximum,
            approved_maximum,
            rtol=0.0,
            atol=_SERVED_BOUNDS_ATOL_MM,
        )
        and np.allclose(
            served_size,
            approved_size,
            rtol=0.0,
            atol=_SERVED_BOUNDS_ATOL_MM,
        )
    ):
        raise GeometryPreprocessingError("Served CAD bounds do not match the approved mesh.")


def _validate_operation_number(value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise GeometryPreprocessingError("operation_number must be a positive integer.")


def _nonempty_string(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise GeometryPreprocessingError(f"{field_name} must be an exact non-empty string.")
    return value


def _relative_ref(root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError as exc:
        raise GeometryPreprocessingError(
            "Geometry artifact is outside the interaction root."
        ) from exc


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _float_list(values: Sequence[float] | np.ndarray) -> list[float]:
    result = [float(value) for value in values]
    if not all(math.isfinite(value) for value in result):
        raise GeometryPreprocessingError("Derived geometry contains non-finite values.")
    return result


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
