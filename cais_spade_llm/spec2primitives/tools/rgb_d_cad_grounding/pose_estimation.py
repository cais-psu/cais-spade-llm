"""Estimate a camera-frame CAD pose for neutral segmented candidates."""

from __future__ import annotations

import math
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import permutations, product
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from cais_spade_llm.spec2primitives.tools.rgb_d_cad_grounding.size_correspondence import (
    _GROUNDING_ROOT,
    _PRODUCER,
    CADSizeAssociationError,
    _association_decision,
    _load_cad_input,
    _load_candidates,
    _load_json_record,
    _rank_candidates,
    _relative_ref,
    _sha256_path,
    _validate_hashed_ref,
    _validate_positive_integer,
    _write_json,
)

_CORRESPONDENCE_RECORD_KEYS = {
    "schema_version",
    "record_type",
    "producer",
    "correspondence_number",
    "method",
    "parameters",
    "CAD",
    "segmentation",
    "ranked_candidates",
    "plausible_candidates",
    "selected_candidate",
    "CAD_correspondence",
    "location",
    "pose",
    "cross_camera_fusion",
}
_MINIMUM_VOXEL_SIZE_M = 0.00025
_MAXIMUM_VOXEL_SIZE_M = 0.0025
_VOXELS_ACROSS_SMALLEST_DIMENSION = 20.0
_MINIMUM_DOWNSAMPLED_POINTS = 30
_MAXIMUM_ICP_ITERATIONS = 40
_ICP_TRIM_FRACTION = 0.90
_ICP_TRANSLATION_TOLERANCE_M = 1e-7
_ICP_ROTATION_TOLERANCE_DEGREES = 0.01
_INLIER_DISTANCE_FACTOR = 1.5
_MINIMUM_REGISTRATION_FITNESS = 0.70
_MAXIMUM_RMSE_FACTOR = 1.5
_CANDIDATE_FITNESS_MARGIN = 0.10
_ROTATION_FITNESS_MARGIN = 0.05
_ROTATION_CLUSTER_DEGREES = 5.0
_TRANSLATION_CLUSTER_FACTOR = 2.0
_MINIMUM_SURFACE_SAMPLES = 2_000
_MAXIMUM_SURFACE_SAMPLES = 8_000


class CADPoseEstimationError(ValueError):
    """Raised when camera-frame CAD pose estimation cannot run safely."""


@dataclass(frozen=True)
class CADPoseEstimationResult:
    """Return one persisted CAD pose-estimation result."""

    record_path: Path
    CAD_correspondence: str
    location: str
    pose: str
    selected_candidate: Mapping[str, object] | None
    record: Mapping[str, object]


@dataclass(frozen=True)
class _PoseInputs:
    correspondence_path: Path
    correspondence_record: Mapping[str, object]
    cad_triangles_m: np.ndarray
    cad_dimensions_m: np.ndarray
    candidate_inputs: tuple[Mapping[str, object], ...]


@dataclass(frozen=True)
class _RegistrationHypothesis:
    observation_handle: str
    camera_id: str
    camera_order: int
    frame: str
    candidate_handle: str
    candidate_id: int
    point_count: int
    initialization: int
    transformation: np.ndarray
    fitness: float
    inlier_rmse_m: float
    inlier_count: int
    voxel_size_m: float


def estimate_camera_frame_pose(
    *,
    interaction_root: Path,
    correspondence_record_path: Path,
    pose_number: int = 1,
) -> CADPoseEstimationResult:
    """Estimate one CAD pose from size-plausible neutral candidates.

    The returned transformation maps the exact approved CAD-local frame into
    the selected camera optical frame. It is not a robot-frame pose or pick
    point, and ambiguous registrations remain explicit.
    """
    try:
        _validate_positive_integer(pose_number, "pose_number")
        root = Path(interaction_root).resolve()
        inputs = _load_pose_inputs(root, correspondence_record_path)
    except CADSizeAssociationError as exc:
        raise CADPoseEstimationError(str(exc)) from exc

    destination = root / _GROUNDING_ROOT / f"pose_{pose_number:04d}"
    if destination.exists():
        raise CADPoseEstimationError(f"CAD pose estimation {pose_number:04d} already exists.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        temporary_root = Path(tempfile.mkdtemp(prefix=".pose-", dir=destination.parent))
    except OSError as exc:
        raise CADPoseEstimationError(
            "CAD pose estimation temporary directory could not be created."
        ) from exc

    try:
        hypotheses = _estimate_hypotheses(inputs)
        decision = _pose_decision(hypotheses, inputs.cad_triangles_m)
        record = _pose_record(
            root=root,
            pose_number=pose_number,
            inputs=inputs,
            hypotheses=hypotheses,
            decision=decision,
        )
        _write_json(temporary_root / "pose_record.json", record)
        if destination.exists():
            raise CADPoseEstimationError(f"CAD pose estimation {pose_number:04d} already exists.")
        temporary_root.rename(destination)
    except CADPoseEstimationError:
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise CADPoseEstimationError(
            f"CAD pose estimation failed: {type(exc).__name__}: {exc}"
        ) from exc

    return CADPoseEstimationResult(
        record_path=destination / "pose_record.json",
        CAD_correspondence=str(decision["CAD_correspondence"]),
        location=str(decision["location"]),
        pose=str(decision["pose"]),
        selected_candidate=decision["selected_candidate"],
        record=record,
    )


def _load_pose_inputs(
    interaction_root: Path,
    record_path: Path,
) -> _PoseInputs:
    path, relative, record = _load_json_record(interaction_root, record_path)
    if path.name != "correspondence_record.json" or set(record) != _CORRESPONDENCE_RECORD_KEYS:
        raise CADSizeAssociationError("CAD size correspondence record fields are invalid.")
    correspondence_number = record["correspondence_number"]
    _validate_positive_integer(correspondence_number, "correspondence_number")
    expected_ref = (
        _GROUNDING_ROOT
        / f"correspondence_{correspondence_number:04d}"
        / "correspondence_record.json"
    )
    if relative != expected_ref:
        raise CADSizeAssociationError("CAD size correspondence record path is invalid.")
    if (
        record["schema_version"] != 2
        or record["record_type"] != "CADSizeCorrespondenceRecord"
        or record["producer"] != _PRODUCER
        or record["method"] != "two_largest_principal_dimensions"
        or record["pose"] != "not_evaluated"
        or record["cross_camera_fusion"] != "not_evaluated"
    ):
        raise CADSizeAssociationError("CAD size correspondence identity is invalid.")

    cad_metadata = record["CAD"]
    segmentation_metadata = record["segmentation"]
    if (
        not isinstance(cad_metadata, Mapping)
        or set(cad_metadata)
        != {"context_ref", "record", "source_sha256", "mesh", "compared_dimensions_m"}
        or not isinstance(segmentation_metadata, Mapping)
        or set(segmentation_metadata) != {"observation_ref", "record"}
    ):
        raise CADSizeAssociationError("CAD size correspondence provenance is invalid.")
    cad_path = _validate_hashed_ref(
        interaction_root,
        cad_metadata["record"],
        "CAD preprocessing record",
    )
    segmentation_path = _validate_hashed_ref(
        interaction_root,
        segmentation_metadata["record"],
        "Segmentation record",
    )
    cad_input = _load_cad_input(interaction_root, cad_path)
    validated_segmentation_path, segmentation_record, candidate_inputs = _load_candidates(
        interaction_root,
        segmentation_path,
    )
    ranked = _rank_candidates(candidate_inputs, cad_input.dimensions_m)
    correspondence, location, selected, plausible = _association_decision(ranked)
    expected_cad_metadata = {
        "context_ref": cad_input.context_ref,
        "record": {
            "ref": _relative_ref(interaction_root, cad_input.record_path),
            "sha256": _sha256_path(cad_input.record_path),
        },
        "source_sha256": cad_input.source_sha256,
        "mesh": {"ref": cad_input.mesh_ref, "sha256": cad_input.mesh_sha256},
        "compared_dimensions_m": [float(value) for value in cad_input.dimensions_m],
    }
    expected_segmentation_metadata = {
        "observation_ref": segmentation_record["observation_ref"],
        "record": {
            "ref": _relative_ref(interaction_root, validated_segmentation_path),
            "sha256": _sha256_path(validated_segmentation_path),
        },
    }
    if (
        cad_metadata != expected_cad_metadata
        or segmentation_metadata != expected_segmentation_metadata
        or record["ranked_candidates"] != ranked
        or record["plausible_candidates"] != plausible
        or record["selected_candidate"] != selected
        or record["CAD_correspondence"] != correspondence
        or record["location"] != location
    ):
        raise CADSizeAssociationError("CAD size correspondence record is inconsistent.")

    plausible_keys = {
        (candidate["observation_handle"], candidate["candidate_handle"]) for candidate in plausible
    }
    eligible = tuple(
        candidate
        for candidate in candidate_inputs
        if (
            candidate["observation_handle"],
            candidate["candidate_handle"],
        )
        in plausible_keys
    )
    return _PoseInputs(
        correspondence_path=path,
        correspondence_record=record,
        cad_triangles_m=cad_input.triangles_m,
        cad_dimensions_m=cad_input.dimensions_m,
        candidate_inputs=eligible,
    )


def _estimate_hypotheses(inputs: _PoseInputs) -> list[_RegistrationHypothesis]:
    hypotheses = []
    for candidate in inputs.candidate_inputs:
        points_m = candidate["points_m"]
        if not isinstance(points_m, np.ndarray):
            raise CADPoseEstimationError("Candidate point array is invalid.")
        hypotheses.extend(
            _register_candidate(
                inputs.cad_triangles_m,
                points_m,
                observation_handle=str(candidate["observation_handle"]),
                camera_id=str(candidate["camera_id"]),
                camera_order=int(candidate["camera_order"]),
                frame=str(candidate["frame"]),
                candidate_handle=str(candidate["candidate_handle"]),
                candidate_id=int(candidate["candidate_id"]),
            )
        )
    hypotheses.sort(
        key=lambda item: (
            -item.fitness,
            item.inlier_rmse_m,
            item.camera_order,
            item.candidate_id,
            item.initialization,
        )
    )
    return hypotheses


def _register_candidate(
    triangles_m: np.ndarray,
    candidate_points_m: np.ndarray,
    *,
    observation_handle: str,
    camera_id: str,
    camera_order: int,
    frame: str,
    candidate_handle: str,
    candidate_id: int,
) -> list[_RegistrationHypothesis]:
    voxel_size_m = _voxel_size(triangles_m)
    sample_count = _surface_sample_count(triangles_m, voxel_size_m)
    model_points = _voxel_downsample(
        _sample_mesh_surface(triangles_m, sample_count),
        voxel_size_m,
    )
    observed_points = _voxel_downsample(candidate_points_m, voxel_size_m)
    if (
        model_points.shape[0] < _MINIMUM_DOWNSAMPLED_POINTS
        or observed_points.shape[0] < _MINIMUM_DOWNSAMPLED_POINTS
    ):
        return []
    hypotheses = []
    for initialization, initial in enumerate(
        _principal_axis_initializations(model_points, observed_points),
        start=1,
    ):
        transformation = _refine_point_to_point_icp(
            model_points,
            observed_points,
            initial,
        )
        if not _valid_transformation(transformation):
            continue
        fitness, rmse, inlier_count = _observed_fit(
            model_points,
            observed_points,
            transformation,
            maximum_distance_m=voxel_size_m * _INLIER_DISTANCE_FACTOR,
        )
        hypotheses.append(
            _RegistrationHypothesis(
                observation_handle=observation_handle,
                camera_id=camera_id,
                camera_order=camera_order,
                frame=frame,
                candidate_handle=candidate_handle,
                candidate_id=candidate_id,
                point_count=int(candidate_points_m.shape[0]),
                initialization=initialization,
                transformation=transformation,
                fitness=fitness,
                inlier_rmse_m=rmse,
                inlier_count=inlier_count,
                voxel_size_m=voxel_size_m,
            )
        )
    return hypotheses


def _voxel_downsample(points_m: np.ndarray, voxel_size_m: float) -> np.ndarray:
    points = points_m.astype(np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise CADPoseEstimationError("Registration point array is invalid.")
    voxel_indices = np.floor(points / voxel_size_m).astype(np.int64)
    _, inverse = np.unique(voxel_indices, axis=0, return_inverse=True)
    sums = np.zeros((int(inverse.max(initial=-1)) + 1, 3), dtype=np.float64)
    counts = np.zeros(sums.shape[0], dtype=np.int64)
    np.add.at(sums, inverse, points)
    np.add.at(counts, inverse, 1)
    return sums / counts[:, None]


def _principal_axis_initializations(
    model_points_m: np.ndarray,
    observed_points_m: np.ndarray,
) -> tuple[np.ndarray, ...]:
    model_center, model_basis = _principal_frame(model_points_m)
    observed_center, observed_basis = _principal_frame(observed_points_m)
    results = []
    for permutation in permutations(range(3)):
        permutation_matrix = np.eye(3)[:, permutation]
        for signs in product((-1.0, 1.0), repeat=3):
            axis_transform = permutation_matrix @ np.diag(signs)
            if np.linalg.det(axis_transform) <= 0:
                continue
            rotation = observed_basis @ axis_transform @ model_basis.T
            translation = observed_center - rotation @ model_center
            transformation = np.eye(4, dtype=np.float64)
            transformation[:3, :3] = rotation
            transformation[:3, 3] = translation
            results.append(transformation)
    return tuple(results)


def _principal_frame(points_m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = points_m.mean(axis=0)
    centered = points_m - center
    covariance = centered.T @ centered / points_m.shape[0]
    _, eigenvectors = np.linalg.eigh(covariance)
    basis = eigenvectors[:, ::-1]
    if np.linalg.det(basis) < 0:
        basis[:, -1] *= -1.0
    return center, basis


def _refine_point_to_point_icp(
    model_points_m: np.ndarray,
    observed_points_m: np.ndarray,
    initial_transformation: np.ndarray,
) -> np.ndarray:
    from scipy.spatial import cKDTree

    transformation = initial_transformation.copy()
    keep_count = max(
        _MINIMUM_DOWNSAMPLED_POINTS,
        math.ceil(observed_points_m.shape[0] * _ICP_TRIM_FRACTION),
    )
    for _ in range(_MAXIMUM_ICP_ITERATIONS):
        transformed_model = model_points_m @ transformation[:3, :3].T + transformation[:3, 3]
        distances, indices = cKDTree(transformed_model).query(observed_points_m, k=1)
        retained = np.argsort(distances, kind="stable")[:keep_count]
        updated = _rigid_transform(
            model_points_m[indices[retained]],
            observed_points_m[retained],
        )
        translation_change = float(np.linalg.norm(updated[:3, 3] - transformation[:3, 3]))
        relative = transformation[:3, :3].T @ updated[:3, :3]
        cosine = min(1.0, max(-1.0, (float(np.trace(relative)) - 1.0) * 0.5))
        rotation_change = math.degrees(math.acos(cosine))
        transformation = updated
        if (
            translation_change <= _ICP_TRANSLATION_TOLERANCE_M
            and rotation_change <= _ICP_ROTATION_TOLERANCE_DEGREES
        ):
            break
    return transformation


def _rigid_transform(source_points_m: np.ndarray, target_points_m: np.ndarray) -> np.ndarray:
    source_center = source_points_m.mean(axis=0)
    target_center = target_points_m.mean(axis=0)
    covariance = (source_points_m - source_center).T @ (target_points_m - target_center)
    left, _, right_transpose = np.linalg.svd(covariance)
    rotation = right_transpose.T @ left.T
    if np.linalg.det(rotation) < 0:
        right_transpose[-1] *= -1.0
        rotation = right_transpose.T @ left.T
    transformation = np.eye(4, dtype=np.float64)
    transformation[:3, :3] = rotation
    transformation[:3, 3] = target_center - rotation @ source_center
    return transformation


def _voxel_size(triangles_m: np.ndarray) -> float:
    bounds = np.ptp(triangles_m.reshape(-1, 3).astype(np.float64), axis=0)
    positive = bounds[bounds > 0]
    if positive.size != 3:
        raise CADPoseEstimationError("CAD mesh dimensions are invalid for registration.")
    proposed = float(positive.min() / _VOXELS_ACROSS_SMALLEST_DIMENSION)
    return max(_MINIMUM_VOXEL_SIZE_M, min(_MAXIMUM_VOXEL_SIZE_M, proposed))


def _surface_sample_count(triangles_m: np.ndarray, voxel_size_m: float) -> int:
    edges_a = triangles_m[:, 1].astype(np.float64) - triangles_m[:, 0]
    edges_b = triangles_m[:, 2].astype(np.float64) - triangles_m[:, 0]
    total_area = float(np.linalg.norm(np.cross(edges_a, edges_b), axis=1).sum() * 0.5)
    if not math.isfinite(total_area) or total_area <= 0:
        raise CADPoseEstimationError("CAD mesh surface area is invalid.")
    proposed = math.ceil(total_area / max(voxel_size_m**2, 1e-12))
    return max(_MINIMUM_SURFACE_SAMPLES, min(_MAXIMUM_SURFACE_SAMPLES, proposed))


def _sample_mesh_surface(triangles_m: np.ndarray, sample_count: int) -> np.ndarray:
    triangles = triangles_m.astype(np.float64)
    areas = np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
        axis=1,
    )
    cumulative = np.cumsum(areas)
    if cumulative.size == 0 or cumulative[-1] <= 0:
        raise CADPoseEstimationError("CAD mesh has no sampleable surface.")
    positions = (np.arange(sample_count, dtype=np.float64) + 0.5) / sample_count
    triangle_indices = np.searchsorted(cumulative / cumulative[-1], positions, side="right")
    sequence = np.arange(1, sample_count + 1, dtype=np.float64)
    u = np.mod(sequence * 0.6180339887498949, 1.0)
    v = np.mod(sequence * 0.4142135623730950, 1.0)
    sqrt_u = np.sqrt(u)
    weights = np.column_stack((1.0 - sqrt_u, sqrt_u * (1.0 - v), sqrt_u * v))
    selected = triangles[triangle_indices]
    return np.einsum("ni,nij->nj", weights, selected)


def _observed_fit(
    model_points_m: np.ndarray,
    observed_points_m: np.ndarray,
    transformation: np.ndarray,
    *,
    maximum_distance_m: float,
) -> tuple[float, float, int]:
    from scipy.spatial import cKDTree

    transformed_model = model_points_m @ transformation[:3, :3].T + transformation[:3, 3]
    distances, _ = cKDTree(transformed_model).query(
        observed_points_m.astype(np.float64),
        k=1,
    )
    inliers = distances <= maximum_distance_m
    count = int(inliers.sum())
    if count == 0:
        return 0.0, math.inf, 0
    return (
        float(count / observed_points_m.shape[0]),
        float(np.sqrt(np.mean(np.square(distances[inliers])))),
        count,
    )


def _valid_transformation(value: np.ndarray) -> bool:
    if value.shape != (4, 4) or not np.isfinite(value).all():
        return False
    rotation = value[:3, :3]
    return bool(
        np.allclose(value[3], [0.0, 0.0, 0.0, 1.0], rtol=0.0, atol=1e-8)
        and np.allclose(rotation.T @ rotation, np.eye(3), rtol=0.0, atol=1e-5)
        and math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1e-5)
    )


def _pose_decision(
    hypotheses: list[_RegistrationHypothesis],
    cad_triangles_m: np.ndarray,
) -> dict[str, object]:
    qualified = [
        item
        for item in hypotheses
        if item.fitness >= _MINIMUM_REGISTRATION_FITNESS
        and item.inlier_rmse_m <= item.voxel_size_m * _MAXIMUM_RMSE_FACTOR
    ]
    if not qualified:
        return {
            "CAD_correspondence": "rejected",
            "location": "unavailable",
            "pose": "rejected",
            "selected_candidate": None,
            "qualified_hypotheses": [],
        }
    best = qualified[0]
    competing_candidate = next(
        (
            item
            for item in qualified[1:]
            if (item.camera_id, item.candidate_id) != (best.camera_id, best.candidate_id)
        ),
        None,
    )
    if (
        competing_candidate is not None
        and best.fitness - competing_candidate.fitness < _CANDIDATE_FITNESS_MARGIN
    ):
        return {
            "CAD_correspondence": "ambiguous",
            "location": "ambiguous",
            "pose": "ambiguous",
            "selected_candidate": None,
            "qualified_hypotheses": qualified,
        }

    cad_centroid_m = cad_triangles_m.reshape(-1, 3).mean(axis=0)
    competing_rotations = [
        item
        for item in qualified[1:]
        if (item.camera_id, item.candidate_id) == (best.camera_id, best.candidate_id)
        and not _same_pose(best, item)
        and best.fitness - item.fitness < _ROTATION_FITNESS_MARGIN
    ]
    base_candidate = _candidate_identity(best)
    best_centroid_translation_m = _transformed_point(
        best.transformation,
        cad_centroid_m,
    )
    if competing_rotations:
        centroid_translations_m = [
            best_centroid_translation_m,
            *[
                _transformed_point(item.transformation, cad_centroid_m)
                for item in competing_rotations
            ],
        ]
        centroid_is_stable = all(
            float(np.linalg.norm(best_centroid_translation_m - centroid))
            <= best.voxel_size_m * _TRANSLATION_CLUSTER_FACTOR
            for centroid in centroid_translations_m[1:]
        )
        if centroid_is_stable:
            base_candidate["CAD_centroid_translation_m"] = _float_vector(
                np.mean(centroid_translations_m, axis=0)
            )
        return {
            "CAD_correspondence": "accepted",
            "location": "available" if centroid_is_stable else "ambiguous",
            "pose": "ambiguous",
            "selected_candidate": base_candidate,
            "qualified_hypotheses": qualified,
        }

    selected = dict(base_candidate)
    selected["CAD_centroid_translation_m"] = _float_vector(best_centroid_translation_m)
    selected["CAD_origin_translation_m"] = _float_vector(best.transformation[:3, 3])
    selected["rotation_matrix"] = [_float_vector(row) for row in best.transformation[:3, :3]]
    selected["quaternion_xyzw"] = _float_vector(
        Rotation.from_matrix(best.transformation[:3, :3]).as_quat()
    )
    selected["camera_from_CAD_transform"] = [_float_vector(row) for row in best.transformation]
    selected["registration"] = _registration_metrics(best)
    return {
        "CAD_correspondence": "accepted",
        "location": "available",
        "pose": "accepted",
        "selected_candidate": selected,
        "qualified_hypotheses": qualified,
    }


def _transformed_point(transformation: np.ndarray, point: np.ndarray) -> np.ndarray:
    return transformation[:3, :3] @ point + transformation[:3, 3]


def _same_pose(first: _RegistrationHypothesis, second: _RegistrationHypothesis) -> bool:
    translation_distance = float(
        np.linalg.norm(first.transformation[:3, 3] - second.transformation[:3, 3])
    )
    relative = first.transformation[:3, :3].T @ second.transformation[:3, :3]
    cosine = min(1.0, max(-1.0, (float(np.trace(relative)) - 1.0) * 0.5))
    angle_degrees = math.degrees(math.acos(cosine))
    return bool(
        translation_distance <= first.voxel_size_m * _TRANSLATION_CLUSTER_FACTOR
        and angle_degrees <= _ROTATION_CLUSTER_DEGREES
    )


def _pose_record(
    *,
    root: Path,
    pose_number: int,
    inputs: _PoseInputs,
    hypotheses: list[_RegistrationHypothesis],
    decision: Mapping[str, object],
) -> dict[str, object]:
    qualified = decision["qualified_hypotheses"]
    if not isinstance(qualified, list):
        raise CADPoseEstimationError("Qualified pose hypotheses are invalid.")
    correspondence = inputs.correspondence_record
    return {
        "schema_version": 3,
        "record_type": "CADPoseEstimationRecord",
        "producer": _PRODUCER,
        "pose_number": pose_number,
        "method": "principal_axis_multistart_point_to_point_ICP",
        "parameters": {
            "voxel_size_range_m": [
                _MINIMUM_VOXEL_SIZE_M,
                _MAXIMUM_VOXEL_SIZE_M,
            ],
            "principal_axis_initializations": 24,
            "maximum_ICP_iterations": _MAXIMUM_ICP_ITERATIONS,
            "ICP_trim_fraction": _ICP_TRIM_FRACTION,
            "inlier_distance_factor": _INLIER_DISTANCE_FACTOR,
            "minimum_registration_fitness": _MINIMUM_REGISTRATION_FITNESS,
            "maximum_RMSE_factor": _MAXIMUM_RMSE_FACTOR,
            "candidate_fitness_margin": _CANDIDATE_FITNESS_MARGIN,
            "rotation_fitness_margin": _ROTATION_FITNESS_MARGIN,
            "rotation_cluster_degrees": _ROTATION_CLUSTER_DEGREES,
        },
        "source_correspondence": {
            "ref": _relative_ref(root, inputs.correspondence_path),
            "sha256": _sha256_path(inputs.correspondence_path),
        },
        "CAD": correspondence["CAD"],
        "segmentation": correspondence["segmentation"],
        "ranked_pose_hypotheses": [_hypothesis_record(item) for item in hypotheses],
        "qualified_pose_hypotheses": [_hypothesis_record(item) for item in qualified],
        "selected_candidate": decision["selected_candidate"],
        "CAD_correspondence": decision["CAD_correspondence"],
        "location": decision["location"],
        "pose": decision["pose"],
        "coordinate_frame": (
            None
            if decision["selected_candidate"] is None
            else decision["selected_candidate"]["frame"]
        ),
        "cross_camera_fusion": "not_evaluated",
        "robot_frame_conversion": "not_evaluated",
    }


def _candidate_identity(item: _RegistrationHypothesis) -> dict[str, object]:
    return {
        "observation_handle": item.observation_handle,
        "camera_id": item.camera_id,
        "frame": item.frame,
        "candidate_handle": item.candidate_handle,
        "candidate_id": item.candidate_id,
        "point_count": item.point_count,
    }


def _hypothesis_record(item: _RegistrationHypothesis) -> dict[str, object]:
    record = _candidate_identity(item)
    record.update(
        {
            "initialization": item.initialization,
            "camera_from_CAD_transform": [_float_vector(row) for row in item.transformation],
            "registration": _registration_metrics(item),
        }
    )
    return record


def _registration_metrics(item: _RegistrationHypothesis) -> dict[str, object]:
    return {
        "fitness": item.fitness,
        "inlier_RMSE_m": item.inlier_rmse_m,
        "inlier_count": item.inlier_count,
        "voxel_size_m": item.voxel_size_m,
    }


def _float_vector(values: np.ndarray) -> list[float]:
    result = [float(value) for value in values]
    if not all(math.isfinite(value) for value in result):
        raise CADPoseEstimationError("Pose result contains a non-finite value.")
    return result
