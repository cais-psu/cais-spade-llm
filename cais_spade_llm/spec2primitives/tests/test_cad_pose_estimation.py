"""Tests for simple generalized camera-frame CAD pose estimation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from cais_spade_llm.spec2primitives.tests.test_cad_size_correspondence import (
    _prepare_inputs,
    _size_bundle,
)
from cais_spade_llm.spec2primitives.tools.rgb_d_cad_grounding import (
    CADPoseEstimationError,
    associate_segmented_candidate_by_size,
    estimate_camera_frame_pose,
    read_rgbd_segmentation_status,
    run_cad_pose_estimation_pipeline,
)
from cais_spade_llm.spec2primitives.tools.rgb_d_cad_grounding import (
    pose_estimation as pose_module,
)
from cais_spade_llm.spec2primitives.tools.rgb_d_cad_grounding.pose_estimation import (
    _register_candidate,
    _RegistrationHypothesis,
    _sample_mesh_surface,
)


def test_asymmetric_surface_registration_recovers_known_camera_frame_pose() -> None:
    triangles = _asymmetric_tetrahedron()
    model_points = _sample_mesh_surface(triangles, 5_000)
    expected_rotation = Rotation.from_euler(
        "xyz",
        [21.0, -17.0, 38.0],
        degrees=True,
    ).as_matrix()
    expected_translation = np.asarray([0.12, -0.04, 0.75])
    observed_points = model_points @ expected_rotation.T + expected_translation

    hypotheses = _register_candidate(
        triangles,
        observed_points,
        camera_id="cam_mk3",
        camera_order=0,
        frame="cam_mk3_optical_frame",
        candidate_id=1,
    )

    assert len(hypotheses) == 24
    best = sorted(
        hypotheses,
        key=lambda item: (-item.fitness, item.inlier_rmse_m, item.initialization),
    )[0]
    rotation_error_degrees = np.degrees(
        Rotation.from_matrix(
            best.transformation[:3, :3].T @ expected_rotation
        ).magnitude()
    )
    assert best.fitness == pytest.approx(1.0)
    assert rotation_error_degrees < 1.0
    np.testing.assert_allclose(
        best.transformation[:3, 3],
        expected_translation,
        atol=5e-4,
    )


def test_unique_shape_fit_persists_complete_camera_frame_pose(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    correspondence_path = _prepare_correspondence(tmp_path, (0.042,))
    monkeypatch.setattr(pose_module, "_register_candidate", _accepted_registration)

    result = estimate_camera_frame_pose(
        interaction_root=tmp_path,
        correspondence_record_path=correspondence_path,
    )

    assert result.CAD_correspondence == "accepted"
    assert result.location == "available"
    assert result.pose == "accepted"
    assert result.selected_candidate is not None
    assert result.selected_candidate["frame"] == "cam_mk3_optical_frame"
    assert len(result.selected_candidate["quaternion_xyzw"]) == 4
    record = _read_json(result.record_path)
    assert record == result.record
    assert record["schema_version"] == 2
    assert record["record_type"] == "CADPoseEstimationRecord"
    assert record["method"] == "principal_axis_multistart_point_to_point_ICP"
    assert record["coordinate_frame"] == "cam_mk3_optical_frame"
    assert record["cross_camera_fusion"] == "not_evaluated"
    assert record["robot_frame_conversion"] == "not_evaluated"
    assert len(result.selected_candidate["CAD_centroid_translation_m"]) == 3
    assert record["source_correspondence"]["ref"].endswith(
        "correspondence_0001/correspondence_record.json"
    )


def test_shape_fit_resolves_same_size_candidates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    correspondence_path = _prepare_correspondence(tmp_path, (0.042, 0.042))

    def register(*args: object, **kwargs: object) -> list[_RegistrationHypothesis]:
        candidate_id = int(kwargs["candidate_id"])
        fitness = 0.94 if candidate_id == 2 else 0.72
        return [_hypothesis(kwargs, fitness=fitness)]

    monkeypatch.setattr(pose_module, "_register_candidate", register)

    result = estimate_camera_frame_pose(
        interaction_root=tmp_path,
        correspondence_record_path=correspondence_path,
    )

    assert result.CAD_correspondence == "accepted"
    assert result.pose == "accepted"
    assert result.selected_candidate is not None
    assert result.selected_candidate["candidate_id"] == 2


def test_similar_shape_fits_remain_candidate_ambiguous(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    correspondence_path = _prepare_correspondence(tmp_path, (0.042, 0.042))

    def register(*args: object, **kwargs: object) -> list[_RegistrationHypothesis]:
        candidate_id = int(kwargs["candidate_id"])
        return [_hypothesis(kwargs, fitness=0.91 - candidate_id * 0.01)]

    monkeypatch.setattr(pose_module, "_register_candidate", register)

    result = estimate_camera_frame_pose(
        interaction_root=tmp_path,
        correspondence_record_path=correspondence_path,
    )

    assert result.CAD_correspondence == "ambiguous"
    assert result.location == "ambiguous"
    assert result.pose == "ambiguous"
    assert result.selected_candidate is None


def test_symmetric_rotation_hypotheses_remain_pose_ambiguous(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    correspondence_path = _prepare_correspondence(tmp_path, (0.042,))
    inputs = pose_module._load_pose_inputs(tmp_path.resolve(), correspondence_path)
    cad_centroid_m = inputs.cad_triangles_m.reshape(-1, 3).mean(axis=0)
    camera_centroid_m = cad_centroid_m + np.asarray([0.12, -0.04, 0.75])

    def register(*args: object, **kwargs: object) -> list[_RegistrationHypothesis]:
        first = _hypothesis(kwargs, fitness=0.92)
        rotated = np.eye(4)
        rotated[:3, :3] = Rotation.from_euler("z", 90.0, degrees=True).as_matrix()
        rotated[:3, 3] = camera_centroid_m - rotated[:3, :3] @ cad_centroid_m
        second = _hypothesis(
            kwargs,
            fitness=0.90,
            initialization=2,
            transformation=rotated,
        )
        return [first, second]

    monkeypatch.setattr(pose_module, "_register_candidate", register)

    result = estimate_camera_frame_pose(
        interaction_root=tmp_path,
        correspondence_record_path=correspondence_path,
    )

    assert result.CAD_correspondence == "accepted"
    assert result.location == "available"
    assert result.pose == "ambiguous"
    assert result.selected_candidate is not None
    assert len(result.record["qualified_pose_hypotheses"]) == 2
    np.testing.assert_allclose(
        result.selected_candidate["CAD_centroid_translation_m"],
        camera_centroid_m,
    )
    first_origin = result.record["qualified_pose_hypotheses"][0][
        "camera_from_CAD_transform"
    ][:3]
    second_origin = result.record["qualified_pose_hypotheses"][1][
        "camera_from_CAD_transform"
    ][:3]
    assert not np.allclose(
        np.asarray(first_origin)[:, 3],
        np.asarray(second_origin)[:, 3],
    )
    assert "rotation_matrix" not in result.selected_candidate
    assert "CAD_origin_translation_m" not in result.selected_candidate


def test_symmetric_rotations_with_different_centroids_keep_location_ambiguous(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    correspondence_path = _prepare_correspondence(tmp_path, (0.042,))
    inputs = pose_module._load_pose_inputs(tmp_path.resolve(), correspondence_path)
    cad_centroid_m = inputs.cad_triangles_m.reshape(-1, 3).mean(axis=0)

    def register(*args: object, **kwargs: object) -> list[_RegistrationHypothesis]:
        first = _hypothesis(kwargs, fitness=0.92)
        rotated = np.eye(4)
        rotated[:3, :3] = Rotation.from_euler("z", 90.0, degrees=True).as_matrix()
        camera_centroid_m = (
            first.transformation[:3, :3] @ cad_centroid_m
            + first.transformation[:3, 3]
        )
        rotated[:3, 3] = (
            camera_centroid_m
            + np.asarray([0.01, 0.0, 0.0])
            - rotated[:3, :3] @ cad_centroid_m
        )
        return [
            first,
            _hypothesis(
                kwargs,
                fitness=0.90,
                initialization=2,
                transformation=rotated,
            ),
        ]

    monkeypatch.setattr(pose_module, "_register_candidate", register)

    result = estimate_camera_frame_pose(
        interaction_root=tmp_path,
        correspondence_record_path=correspondence_path,
    )

    assert result.CAD_correspondence == "accepted"
    assert result.location == "ambiguous"
    assert result.pose == "ambiguous"
    assert result.selected_candidate is not None
    assert "CAD_centroid_translation_m" not in result.selected_candidate


def test_zero_candidates_persist_rejected_pose_without_registration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    correspondence_path = _prepare_correspondence(
        tmp_path,
        (),
        zero_candidates=True,
    )

    def unexpected_registration(*args: object, **kwargs: object) -> object:
        raise AssertionError("Registration must not run without a plausible source candidate.")

    monkeypatch.setattr(pose_module, "_register_candidate", unexpected_registration)

    result = estimate_camera_frame_pose(
        interaction_root=tmp_path,
        correspondence_record_path=correspondence_path,
    )

    assert result.CAD_correspondence == "rejected"
    assert result.location == "unavailable"
    assert result.pose == "rejected"
    assert result.record["ranked_pose_hypotheses"] == []


def test_tampered_correspondence_rejects_without_partial_pose_output(
    tmp_path: Path,
) -> None:
    correspondence_path = _prepare_correspondence(tmp_path, (0.042,))
    record = _read_json(correspondence_path)
    record["ranked_candidates"][0]["point_count"] += 1
    correspondence_path.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(CADPoseEstimationError, match="inconsistent"):
        estimate_camera_frame_pose(
            interaction_root=tmp_path,
            correspondence_record_path=correspondence_path,
        )

    grounding_root = tmp_path / "products/grounding/rgb_d_cad_grounding"
    assert not (grounding_root / "pose_0001").exists()
    assert list(grounding_root.glob(".pose-*")) == []


def test_registration_failure_cleans_up_and_existing_pose_is_not_overwritten(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    correspondence_path = _prepare_correspondence(tmp_path, (0.042,))

    def failed_registration(*args: object, **kwargs: object) -> object:
        raise RuntimeError("controlled registration failure")

    monkeypatch.setattr(pose_module, "_register_candidate", failed_registration)
    with pytest.raises(CADPoseEstimationError, match="controlled registration failure"):
        estimate_camera_frame_pose(
            interaction_root=tmp_path,
            correspondence_record_path=correspondence_path,
        )
    grounding_root = tmp_path / "products/grounding/rgb_d_cad_grounding"
    assert list(grounding_root.glob(".pose-*")) == []

    monkeypatch.setattr(pose_module, "_register_candidate", _accepted_registration)
    first = estimate_camera_frame_pose(
        interaction_root=tmp_path,
        correspondence_record_path=correspondence_path,
    )
    before = first.record_path.read_bytes()
    with pytest.raises(CADPoseEstimationError, match="already exists"):
        estimate_camera_frame_pose(
            interaction_root=tmp_path,
            correspondence_record_path=correspondence_path,
        )
    assert first.record_path.read_bytes() == before


def test_status_wrapper_exposes_pose_state_without_coordinates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    interaction_root = tmp_path / "interaction"
    correspondence_path = _prepare_correspondence(interaction_root, (0.042,))
    monkeypatch.setattr(pose_module, "_register_candidate", _accepted_registration)

    result = run_cad_pose_estimation_pipeline(
        contexts_root=tmp_path,
        interaction_root=interaction_root,
        correspondence_record_path=correspondence_path,
    )

    assert result == {
        "status": "ready",
        "interaction_root": str(interaction_root.resolve()),
        "pose_record_path": result["pose_record_path"],
        "CAD_correspondence": "accepted",
        "location": "available",
        "pose": "accepted",
        "failure": None,
    }
    assert "selected_candidate" not in result
    assert "camera_from_CAD_transform" not in result
    status = read_rgbd_segmentation_status(tmp_path)
    assert status["pose"] == "accepted"
    assert "selected_candidate" not in status
    assert "camera_from_CAD_transform" not in status


def _prepare_correspondence(
    interaction_root: Path,
    diameters_m: tuple[float, ...],
    *,
    zero_candidates: bool = False,
) -> Path:
    segmentation_path, cad_path = _prepare_inputs(
        interaction_root,
        _size_bundle(diameters_m, zero_candidates=zero_candidates),
    )
    result = associate_segmented_candidate_by_size(
        interaction_root=interaction_root,
        segmentation_record_path=segmentation_path,
        cad_record_path=cad_path,
    )
    return result.record_path


def _accepted_registration(
    triangles_m: np.ndarray,
    candidate_points_m: np.ndarray,
    *,
    camera_id: str,
    camera_order: int,
    frame: str,
    candidate_id: int,
) -> list[_RegistrationHypothesis]:
    del triangles_m
    values = {
        "camera_id": camera_id,
        "camera_order": camera_order,
        "frame": frame,
        "candidate_id": candidate_id,
        "point_count": candidate_points_m.shape[0],
    }
    return [_hypothesis(values, fitness=0.94)]


def _hypothesis(
    values: Any,
    *,
    fitness: float,
    initialization: int = 1,
    transformation: np.ndarray | None = None,
) -> _RegistrationHypothesis:
    resolved_transform = np.eye(4) if transformation is None else transformation.copy()
    if transformation is None:
        resolved_transform[:3, 3] = [0.12, -0.04, 0.75]
    return _RegistrationHypothesis(
        camera_id=str(values["camera_id"]),
        camera_order=int(values["camera_order"]),
        frame=str(values["frame"]),
        candidate_id=int(values["candidate_id"]),
        point_count=int(values.get("point_count", 100)),
        initialization=initialization,
        transformation=resolved_transform,
        fitness=fitness,
        inlier_rmse_m=0.0002,
        inlier_count=90,
        voxel_size_m=0.001,
    )


def _asymmetric_tetrahedron() -> np.ndarray:
    vertices = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.041, 0.0, 0.0],
            [0.0, 0.027, 0.0],
            [0.0, 0.0, 0.019],
        ],
        dtype=np.float32,
    )
    return np.asarray(
        [
            [vertices[0], vertices[2], vertices[1]],
            [vertices[0], vertices[1], vertices[3]],
            [vertices[0], vertices[3], vertices[2]],
            [vertices[1], vertices[2], vertices[3]],
        ],
        dtype=np.float32,
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
