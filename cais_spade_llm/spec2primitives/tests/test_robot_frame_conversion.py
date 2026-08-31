"""Tests for dynamic camera-to-robot CAD pose conversion."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from cais_spade_llm.spec2primitives.tests.test_cad_pose_estimation import (
    _accepted_registration,
    _hypothesis,
    _prepare_correspondence,
)
from cais_spade_llm.spec2primitives.tools.rgb_d_cad_grounding import (
    CameraToRobotCalibrationError,
    RobotFrameConversionError,
    estimate_camera_frame_pose,
    read_rgbd_segmentation_status,
    record_camera_to_robot_calibration,
    run_robot_frame_pose_conversion_pipeline,
    transform_camera_pose_to_robot_frame,
    transform_correspondence_location_to_robot_frame,
)
from cais_spade_llm.spec2primitives.tools.rgb_d_cad_grounding import (
    frame_conversion as conversion_module,
)
from cais_spade_llm.spec2primitives.tools.rgb_d_cad_grounding import (
    pose_estimation as pose_module,
)

_CAMERA_FRAME = "cam_mk3_optical_frame"
_ROBOT_FRAME = "robot_base"
_OBSERVATION_TIMESTAMP_NS = 3_000_004_000
_PROVENANCE_SHA256 = "a" * 64


def test_correspondence_center_becomes_location_without_pose_estimation(
    tmp_path: Path,
) -> None:
    correspondence_path = _prepare_correspondence(tmp_path, (0.042,))
    correspondence = _read_json(correspondence_path)
    selected = correspondence["selected_candidate"]
    robot_from_camera = np.eye(4)
    robot_from_camera[:3, :3] = Rotation.from_euler(
        "z", 30.0, degrees=True
    ).as_matrix()
    robot_from_camera[:3, 3] = [0.8, -0.2, 0.4]
    calibration = _calibration(tmp_path, robot_from_camera)

    result = transform_correspondence_location_to_robot_frame(
        interaction_root=tmp_path,
        correspondence_record_path=correspondence_path,
        calibration_record_path=calibration.record_path,
        target_frame=_ROBOT_FRAME,
    )

    expected = (
        robot_from_camera[:3, :3]
        @ np.asarray(selected["candidate_center_m"], dtype=np.float64)
        + robot_from_camera[:3, 3]
    )
    np.testing.assert_allclose(result.translated_location_m, expected, atol=1e-12)
    assert result.record["robot_frame_conversion"] == "accepted"
    assert result.record["record_type"] == "RobotFrameLocationRecord"
    assert result.record["schema_version"] == 1
    assert result.record["CAD_correspondence"] == "accepted"
    assert result.record["location"] == "available"
    assert "pose" not in result.record
    assert "rotation_matrix" not in result.record
    assert result.record["source_correspondence"]["sha256"] == _sha256(
        correspondence_path
    )
    assert result.record["source_calibration"]["sha256"] == _sha256(
        calibration.record_path
    )


def test_rotated_camera_transform_recovers_known_robot_frame_pose(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pose = _accepted_pose(tmp_path, monkeypatch)
    robot_from_camera = np.eye(4)
    robot_from_camera[:3, :3] = Rotation.from_euler(
        "z", 90.0, degrees=True
    ).as_matrix()
    robot_from_camera[:3, 3] = [1.0, 2.0, 3.0]
    calibration = _calibration(tmp_path, robot_from_camera)

    result = transform_camera_pose_to_robot_frame(
        interaction_root=tmp_path,
        pose_record_path=pose.record_path,
        calibration_record_path=calibration.record_path,
        target_frame=_ROBOT_FRAME,
    )

    expected = robot_from_camera @ np.asarray(
        pose.selected_candidate["camera_from_CAD_transform"],
        dtype=np.float64,
    )
    assert result.robot_frame_conversion == "accepted"
    assert result.target_frame == _ROBOT_FRAME
    assert result.robot_frame_pose is not None
    np.testing.assert_allclose(
        result.robot_frame_pose["robot_from_CAD_transform"],
        expected,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        result.robot_frame_pose["CAD_origin_translation_m"],
        expected[:3, 3],
        atol=1e-12,
    )
    expected_centroid = (
        robot_from_camera[:3, :3]
        @ np.asarray(
            pose.selected_candidate["CAD_centroid_translation_m"],
            dtype=np.float64,
        )
        + robot_from_camera[:3, 3]
    )
    np.testing.assert_allclose(
        result.robot_frame_pose["CAD_centroid_translation_m"],
        expected_centroid,
        atol=1e-12,
    )
    record = _read_json(result.record_path)
    assert record == result.record
    assert record["schema_version"] == 2
    assert record["record_type"] == "RobotFramePoseRecord"
    assert record["source_frame"] == _CAMERA_FRAME
    assert record["target_frame"] == _ROBOT_FRAME
    assert record["observation_timestamp_ns"] == _OBSERVATION_TIMESTAMP_NS
    assert record["source_pose"]["sha256"] == _sha256(pose.record_path)
    assert record["source_calibration"]["sha256"] == _sha256(
        calibration.record_path
    )


def test_different_dynamic_camera_poses_use_the_same_conversion_code(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    correspondence_path = _prepare_correspondence(tmp_path, (0.042,))
    monkeypatch.setattr(pose_module, "_register_candidate", _accepted_registration)
    first_pose = estimate_camera_frame_pose(
        interaction_root=tmp_path,
        correspondence_record_path=correspondence_path,
        pose_number=1,
    )

    second_camera_from_CAD = np.eye(4)
    second_camera_from_CAD[:3, :3] = Rotation.from_euler(
        "xyz", [10.0, -20.0, 30.0], degrees=True
    ).as_matrix()
    second_camera_from_CAD[:3, 3] = [0.35, 0.08, 0.61]

    def second_registration(*args: object, **kwargs: object) -> list[object]:
        del args
        return [
            _hypothesis(
                kwargs,
                fitness=0.94,
                transformation=second_camera_from_CAD,
            )
        ]

    monkeypatch.setattr(pose_module, "_register_candidate", second_registration)
    second_pose = estimate_camera_frame_pose(
        interaction_root=tmp_path,
        correspondence_record_path=correspondence_path,
        pose_number=2,
    )
    robot_from_camera = np.eye(4)
    robot_from_camera[:3, 3] = [0.8, -0.2, 0.4]
    calibration = _calibration(tmp_path, robot_from_camera)

    first = transform_camera_pose_to_robot_frame(
        interaction_root=tmp_path,
        pose_record_path=first_pose.record_path,
        calibration_record_path=calibration.record_path,
        target_frame=_ROBOT_FRAME,
        conversion_number=1,
    )
    second = transform_camera_pose_to_robot_frame(
        interaction_root=tmp_path,
        pose_record_path=second_pose.record_path,
        calibration_record_path=calibration.record_path,
        target_frame=_ROBOT_FRAME,
        conversion_number=2,
    )

    assert first.robot_frame_pose != second.robot_frame_pose
    np.testing.assert_allclose(
        second.robot_frame_pose["robot_from_CAD_transform"],
        robot_from_camera @ second_camera_from_CAD,
        atol=1e-12,
    )


@pytest.mark.parametrize(
    ("transform", "message"),
    [
        (np.diag([1.0, 1.0, -1.0, 1.0]), "rigid homogeneous"),
        (np.full((4, 4), np.nan), "finite values"),
        (np.eye(3), "shape"),
    ],
)
def test_calibration_writer_rejects_invalid_transforms_without_output(
    tmp_path: Path,
    transform: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(CameraToRobotCalibrationError, match=message):
        _calibration(tmp_path, transform)

    grounding_root = tmp_path / "products/grounding/rgb_d_cad_grounding"
    assert not (grounding_root / "calibration_0001").exists()


@pytest.mark.parametrize(
    ("target_frame", "valid_from_ns", "valid_until_ns", "expected"),
    [
        ("different_robot_base", 0, None, "target_frame"),
        (_ROBOT_FRAME, _OBSERVATION_TIMESTAMP_NS + 1, None, "observation timestamp"),
        (_ROBOT_FRAME, 0, _OBSERVATION_TIMESTAMP_NS - 1, "observation timestamp"),
    ],
)
def test_frame_mismatch_and_invalid_calibration_window_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target_frame: str,
    valid_from_ns: int,
    valid_until_ns: int | None,
    expected: str,
) -> None:
    pose = _accepted_pose(tmp_path, monkeypatch)
    calibration = _calibration(
        tmp_path,
        np.eye(4),
        valid_from_ns=valid_from_ns,
        valid_until_ns=valid_until_ns,
    )

    with pytest.raises(RobotFrameConversionError, match=expected):
        transform_camera_pose_to_robot_frame(
            interaction_root=tmp_path,
            pose_record_path=pose.record_path,
            calibration_record_path=calibration.record_path,
            target_frame=target_frame,
        )

    _assert_no_robot_pose_output(tmp_path)


def test_source_frame_mismatch_and_missing_calibration_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pose = _accepted_pose(tmp_path, monkeypatch)
    calibration = record_camera_to_robot_calibration(
        interaction_root=tmp_path,
        calibration_id="calibration-v1",
        source_frame="another_camera_optical_frame",
        target_frame=_ROBOT_FRAME,
        target_from_camera_transform=np.eye(4),
        valid_from_ns=0,
        valid_until_ns=None,
        provenance_source="approved_hand_eye_calibration",
        provenance_sha256=_PROVENANCE_SHA256,
    )
    with pytest.raises(RobotFrameConversionError, match="source_frame"):
        transform_camera_pose_to_robot_frame(
            interaction_root=tmp_path,
            pose_record_path=pose.record_path,
            calibration_record_path=calibration.record_path,
            target_frame=_ROBOT_FRAME,
        )
    with pytest.raises(RobotFrameConversionError, match="path is invalid"):
        transform_camera_pose_to_robot_frame(
            interaction_root=tmp_path,
            pose_record_path=pose.record_path,
            calibration_record_path=tmp_path / "missing.json",
            target_frame=_ROBOT_FRAME,
        )
    _assert_no_robot_pose_output(tmp_path)


def test_tampered_calibration_payload_and_pose_hash_chain_are_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pose = _accepted_pose(tmp_path, monkeypatch)
    calibration = _calibration(tmp_path, np.eye(4))
    calibration_record = _read_json(calibration.record_path)
    calibration_record["calibration_id"] = "tampered"
    calibration.record_path.write_text(json.dumps(calibration_record), encoding="utf-8")

    with pytest.raises(RobotFrameConversionError, match="payload hash"):
        transform_camera_pose_to_robot_frame(
            interaction_root=tmp_path,
            pose_record_path=pose.record_path,
            calibration_record_path=calibration.record_path,
            target_frame=_ROBOT_FRAME,
        )

    source_correspondence = tmp_path / pose.record["source_correspondence"]["ref"]
    correspondence_record = _read_json(source_correspondence)
    correspondence_record["location"] = "tampered"
    source_correspondence.write_text(json.dumps(correspondence_record), encoding="utf-8")
    with pytest.raises(RobotFrameConversionError, match="hash is invalid"):
        transform_camera_pose_to_robot_frame(
            interaction_root=tmp_path,
            pose_record_path=pose.record_path,
            calibration_record_path=calibration.record_path,
            target_frame=_ROBOT_FRAME,
        )
    _assert_no_robot_pose_output(tmp_path)


def test_location_only_pose_converts_centroid_without_fabricating_rotation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ambiguous_root = tmp_path / "ambiguous"
    correspondence_path = _prepare_correspondence(ambiguous_root, (0.042,))
    inputs = pose_module._load_pose_inputs(
        ambiguous_root.resolve(),
        correspondence_path,
    )
    cad_centroid_m = inputs.cad_triangles_m.reshape(-1, 3).mean(axis=0)
    camera_centroid_m = cad_centroid_m + np.asarray([0.12, -0.04, 0.75])

    def ambiguous_registration(*args: object, **kwargs: object) -> list[object]:
        del args
        first = _hypothesis(kwargs, fitness=0.92)
        rotated = np.eye(4)
        rotated[:3, :3] = Rotation.from_euler("z", 90.0, degrees=True).as_matrix()
        rotated[:3, 3] = camera_centroid_m - rotated[:3, :3] @ cad_centroid_m
        return [
            first,
            _hypothesis(
                kwargs,
                fitness=0.90,
                initialization=2,
                transformation=rotated,
            ),
        ]

    monkeypatch.setattr(pose_module, "_register_candidate", ambiguous_registration)
    ambiguous_pose = estimate_camera_frame_pose(
        interaction_root=ambiguous_root,
        correspondence_record_path=correspondence_path,
    )
    robot_from_camera = np.eye(4)
    robot_from_camera[:3, :3] = Rotation.from_euler(
        "z", 30.0, degrees=True
    ).as_matrix()
    robot_from_camera[:3, 3] = [0.8, -0.2, 0.4]
    ambiguous_calibration = _calibration(ambiguous_root, robot_from_camera)
    ambiguous_result = transform_camera_pose_to_robot_frame(
        interaction_root=ambiguous_root,
        pose_record_path=ambiguous_pose.record_path,
        calibration_record_path=ambiguous_calibration.record_path,
        target_frame=_ROBOT_FRAME,
    )

    expected_centroid_m = (
        robot_from_camera[:3, :3] @ camera_centroid_m
        + robot_from_camera[:3, 3]
    )
    assert ambiguous_result.location == "available"
    assert ambiguous_result.pose == "ambiguous"
    assert ambiguous_result.robot_frame_conversion == "accepted"
    assert ambiguous_result.robot_frame_pose is not None
    assert set(ambiguous_result.robot_frame_pose) == {
        "CAD_centroid_translation_m"
    }
    np.testing.assert_allclose(
        ambiguous_result.robot_frame_pose["CAD_centroid_translation_m"],
        expected_centroid_m,
        atol=1e-12,
    )
    assert "rotation_matrix" not in ambiguous_result.robot_frame_pose
    assert "robot_from_CAD_transform" not in ambiguous_result.robot_frame_pose


def test_ambiguous_location_and_rejected_pose_propagate_without_coordinates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ambiguous_root = tmp_path / "ambiguous"
    correspondence_path = _prepare_correspondence(ambiguous_root, (0.042,))

    def ambiguous_registration(*args: object, **kwargs: object) -> list[object]:
        del args
        first = _hypothesis(kwargs, fitness=0.92)
        rotated = np.eye(4)
        rotated[:3, :3] = Rotation.from_euler("z", 90.0, degrees=True).as_matrix()
        rotated[:3, 3] = first.transformation[:3, 3]
        return [
            first,
            _hypothesis(
                kwargs,
                fitness=0.90,
                initialization=2,
                transformation=rotated,
            ),
        ]

    monkeypatch.setattr(pose_module, "_register_candidate", ambiguous_registration)
    ambiguous_pose = estimate_camera_frame_pose(
        interaction_root=ambiguous_root,
        correspondence_record_path=correspondence_path,
    )
    ambiguous_calibration = _calibration(ambiguous_root, np.eye(4))
    ambiguous_result = transform_camera_pose_to_robot_frame(
        interaction_root=ambiguous_root,
        pose_record_path=ambiguous_pose.record_path,
        calibration_record_path=ambiguous_calibration.record_path,
        target_frame=_ROBOT_FRAME,
    )

    rejected_root = tmp_path / "rejected"
    rejected_correspondence = _prepare_correspondence(
        rejected_root,
        (),
        zero_candidates=True,
    )
    rejected_pose = estimate_camera_frame_pose(
        interaction_root=rejected_root,
        correspondence_record_path=rejected_correspondence,
    )
    rejected_calibration = _calibration(rejected_root, np.eye(4))
    rejected_result = transform_camera_pose_to_robot_frame(
        interaction_root=rejected_root,
        pose_record_path=rejected_pose.record_path,
        calibration_record_path=rejected_calibration.record_path,
        target_frame=_ROBOT_FRAME,
    )

    assert ambiguous_result.location == "ambiguous"
    assert ambiguous_result.robot_frame_conversion == "ambiguous"
    assert ambiguous_result.robot_frame_pose is None
    assert ambiguous_result.record["robot_frame_pose"] is None
    assert rejected_result.robot_frame_conversion == "rejected"
    assert rejected_result.robot_frame_pose is None
    assert rejected_result.record["source_frame"] is None


def test_deterministic_reruns_atomic_cleanup_and_no_overwrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = []
    for name in ("first", "second"):
        root = tmp_path / name
        pose = _accepted_pose(root, monkeypatch)
        calibration = _calibration(root, np.eye(4))
        result = transform_camera_pose_to_robot_frame(
            interaction_root=root,
            pose_record_path=pose.record_path,
            calibration_record_path=calibration.record_path,
            target_frame=_ROBOT_FRAME,
        )
        records.append(result.record)
    assert records[0] == records[1]

    cleanup_root = tmp_path / "cleanup"
    pose = _accepted_pose(cleanup_root, monkeypatch)
    calibration = _calibration(cleanup_root, np.eye(4))
    original_write = conversion_module._write_json

    def failed_write(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise OSError("controlled write failure")

    monkeypatch.setattr(conversion_module, "_write_json", failed_write)
    with pytest.raises(RobotFrameConversionError, match="controlled write failure"):
        transform_camera_pose_to_robot_frame(
            interaction_root=cleanup_root,
            pose_record_path=pose.record_path,
            calibration_record_path=calibration.record_path,
            target_frame=_ROBOT_FRAME,
        )
    grounding_root = cleanup_root / "products/grounding/rgb_d_cad_grounding"
    assert not (grounding_root / "robot_pose_0001").exists()
    assert list(grounding_root.glob(".robot-pose-*")) == []

    monkeypatch.setattr(conversion_module, "_write_json", original_write)
    first = transform_camera_pose_to_robot_frame(
        interaction_root=cleanup_root,
        pose_record_path=pose.record_path,
        calibration_record_path=calibration.record_path,
        target_frame=_ROBOT_FRAME,
    )
    before = first.record_path.read_bytes()
    with pytest.raises(RobotFrameConversionError, match="already exists"):
        transform_camera_pose_to_robot_frame(
            interaction_root=cleanup_root,
            pose_record_path=pose.record_path,
            calibration_record_path=calibration.record_path,
            target_frame=_ROBOT_FRAME,
        )
    assert first.record_path.read_bytes() == before


def test_status_wrapper_exposes_conversion_state_without_coordinates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    interaction_root = tmp_path / "interaction"
    pose = _accepted_pose(interaction_root, monkeypatch)
    calibration = _calibration(interaction_root, np.eye(4))

    result = run_robot_frame_pose_conversion_pipeline(
        contexts_root=tmp_path,
        interaction_root=interaction_root,
        pose_record_path=pose.record_path,
        calibration_record_path=calibration.record_path,
        target_frame=_ROBOT_FRAME,
    )

    assert result == {
        "status": "ready",
        "interaction_root": str(interaction_root.resolve()),
        "robot_frame_pose_record_path": result["robot_frame_pose_record_path"],
        "CAD_correspondence": "accepted",
        "location": "available",
        "pose": "accepted",
        "robot_frame_conversion": "accepted",
        "failure": None,
    }
    assert "target_frame" not in result
    assert "robot_frame_pose" not in result
    assert "robot_from_CAD_transform" not in result
    status = read_rgbd_segmentation_status(tmp_path)
    assert status["robot_frame_conversion"] == "accepted"
    assert "target_frame" not in status
    assert "robot_frame_pose" not in status
    assert "robot_from_CAD_transform" not in status


def _accepted_pose(
    interaction_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Any:
    correspondence_path = _prepare_correspondence(interaction_root, (0.042,))
    monkeypatch.setattr(pose_module, "_register_candidate", _accepted_registration)
    return estimate_camera_frame_pose(
        interaction_root=interaction_root,
        correspondence_record_path=correspondence_path,
    )


def _calibration(
    interaction_root: Path,
    transformation: np.ndarray,
    *,
    valid_from_ns: int = 0,
    valid_until_ns: int | None = None,
) -> Any:
    return record_camera_to_robot_calibration(
        interaction_root=interaction_root,
        calibration_id="calibration-v1",
        source_frame=_CAMERA_FRAME,
        target_frame=_ROBOT_FRAME,
        target_from_camera_transform=transformation,
        valid_from_ns=valid_from_ns,
        valid_until_ns=valid_until_ns,
        provenance_source="approved_hand_eye_calibration",
        provenance_sha256=_PROVENANCE_SHA256,
    )


def _assert_no_robot_pose_output(interaction_root: Path) -> None:
    grounding_root = (
        interaction_root / "products/grounding/rgb_d_cad_grounding"
    )
    assert not (grounding_root / "robot_pose_0001").exists()
    assert list(grounding_root.glob(".robot-pose-*")) == []


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
