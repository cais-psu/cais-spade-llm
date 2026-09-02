"""Tests for operator-approved camera-to-world calibration manifests."""

from __future__ import annotations

import hashlib
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from cais_spade_llm.spec2primitives.config import (
    DEFAULT_GAZEBO_CAMERA_TO_WORLD_CALIBRATION_PATH,
    load_camera_to_world_calibration_runtime,
)
from cais_spade_llm.spec2primitives.tools.rgb_d_cad_grounding import (
    CameraToRobotCalibrationError,
)

_CAMERA_FRAMES = (
    "cam_mk3_link",
    "cam_mk4_1_link",
    "cam_mk4_2_link",
    "cam_assembly_link",
)
_CAMERA_FRAME = _CAMERA_FRAMES[0]
_PROVENANCE_SHA256 = "a" * 64
_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_GAZEBO_WORLD_PATH = (
    _REPOSITORY_ROOT
    / "ros2/cais_lab_robotics/worlds/table_spec2primitives.world"
)
_GAZEBO_PROVENANCE_PATH = DEFAULT_GAZEBO_CAMERA_TO_WORLD_CALIBRATION_PATH.with_name(
    "gazebo_camera_to_world_calibration_provenance.json"
)
_MODEL_BY_FRAME = {
    "cam_mk3_link": "cam_mk3",
    "cam_mk4_1_link": "cam_mk4_1",
    "cam_mk4_2_link": "cam_mk4_2",
    "cam_assembly_link": "cam_assembly",
}


def _entry(
    source_frame: str,
    transformation: np.ndarray,
    *,
    target_frame: str = "world",
    valid_from_ns: int = 0,
    valid_until_ns: int | None = None,
    provenance_sha256: str = _PROVENANCE_SHA256,
) -> dict[str, object]:
    return {
        "calibration_id": "approved-world-v1",
        "source_frame": source_frame,
        "target_frame": target_frame,
        "target_from_camera_transform": transformation.tolist(),
        "valid_from_ns": valid_from_ns,
        "valid_until_ns": valid_until_ns,
        "provenance_source": "operator_approved_calibration_report",
        "provenance_sha256": provenance_sha256,
    }


@pytest.mark.parametrize("source_frame", _CAMERA_FRAMES)
def test_manifest_materializes_exact_runtime_selected_world_transform(
    tmp_path: Path,
    source_frame: str,
) -> None:
    interaction_root = tmp_path / "interaction"
    pose_path = _write_camera_location_pose(interaction_root, source_frame)
    transforms: dict[str, np.ndarray] = {}
    for index, frame in enumerate(_CAMERA_FRAMES, start=1):
        transform = np.eye(4)
        transform[:3, 3] = [float(index), 0.0, 0.0]
        transforms[frame] = transform
    manifest_path = _write_manifest(
        tmp_path / "camera_to_world.json",
        [_entry(frame, transforms[frame]) for frame in _CAMERA_FRAMES],
    )
    runtime = load_camera_to_world_calibration_runtime(manifest_path)

    result = runtime.materialize_camera_to_world_calibration(
        interaction_root=interaction_root,
        grounding_record_path=pose_path,
        source_frame=source_frame,
        target_frame="world",
        calibration_number=1,
    )

    assert runtime.manifest_path == manifest_path.resolve()
    assert result.source_frame == source_frame
    assert result.target_frame == "world"
    assert result.record["calibration_id"] == "approved-world-v1"
    assert result.record["target_from_camera_transform"] == transforms[
        source_frame
    ].tolist()
    assert result.record["validity"] == {
        "valid_from_ns": 0,
        "valid_until_ns": None,
    }
    assert result.record["provenance"] == {
        "source": "operator_approved_calibration_report",
        "source_sha256": _PROVENANCE_SHA256,
    }


def test_manifest_missing_selected_camera_fails_closed_without_output(
    tmp_path: Path,
) -> None:
    interaction_root = tmp_path / "interaction"
    pose_path = _write_camera_location_pose(interaction_root, _CAMERA_FRAME)
    runtime = load_camera_to_world_calibration_runtime(
        _write_manifest(
            tmp_path / "camera_to_world.json",
            [_entry("another_camera_frame", np.eye(4))],
        )
    )

    with pytest.raises(
        CameraToRobotCalibrationError,
        match="No approved camera-to-world calibration.*cam_mk3_link",
    ):
        runtime.materialize_camera_to_world_calibration(
            interaction_root=interaction_root,
                grounding_record_path=pose_path,
            source_frame=_CAMERA_FRAME,
            target_frame="world",
            calibration_number=1,
        )

    grounding_root = interaction_root / "products/grounding/rgb_d_cad_grounding"
    assert not (grounding_root / "calibration_0001").exists()


def test_default_gazebo_manifest_matches_pinned_fixed_camera_poses() -> None:
    manifest = json.loads(
        DEFAULT_GAZEBO_CAMERA_TO_WORLD_CALIBRATION_PATH.read_text(
            encoding="utf-8"
        )
    )
    provenance = json.loads(_GAZEBO_PROVENANCE_PATH.read_text(encoding="utf-8"))
    assert provenance["source_world"] == {
        "repository_path": (
            "ros2/cais_lab_robotics/worlds/table_spec2primitives.world"
        ),
        "sha256": hashlib.sha256(_GAZEBO_WORLD_PATH.read_bytes()).hexdigest(),
    }
    provenance_sha256 = hashlib.sha256(
        _GAZEBO_PROVENANCE_PATH.read_bytes()
    ).hexdigest()

    world = ET.parse(_GAZEBO_WORLD_PATH).getroot().find("world")
    assert world is not None
    models = {model.get("name"): model for model in world.findall("model")}
    camera_body_from_optical = Rotation.from_euler(
        "xyz",
        [-math.pi / 2.0, 0.0, -math.pi / 2.0],
    ).as_matrix()
    entries = {
        entry["source_frame"]: entry for entry in manifest["calibrations"]
    }
    assert set(entries) == set(_CAMERA_FRAMES)
    for source_frame, model_name in _MODEL_BY_FRAME.items():
        pose = [float(value) for value in models[model_name].findtext("pose").split()]
        expected = np.eye(4)
        expected[:3, :3] = (
            Rotation.from_euler("xyz", pose[3:]).as_matrix()
            @ camera_body_from_optical
        )
        expected[:3, 3] = pose[:3]
        entry = entries[source_frame]
        np.testing.assert_allclose(
            entry["target_from_camera_transform"],
            expected,
            rtol=0.0,
            atol=1e-12,
        )
        assert entry["target_frame"] == "world"
        assert entry["valid_from_ns"] == 0
        assert entry["valid_until_ns"] is None
        assert entry["provenance_sha256"] == provenance_sha256


def test_default_gazebo_manifest_projects_optical_axis_downward() -> None:
    manifest = json.loads(
        DEFAULT_GAZEBO_CAMERA_TO_WORLD_CALIBRATION_PATH.read_text(
            encoding="utf-8"
        )
    )
    expected_centers = {
        "cam_mk3_link": (-0.4, 0.0, 1.4),
        "cam_mk4_1_link": (0.4, 0.3, 1.4),
        "cam_mk4_2_link": (0.4, -0.3, 1.4),
        "cam_assembly_link": (0.0, 0.0, 1.55),
    }
    for entry in manifest["calibrations"]:
        source_frame = entry["source_frame"]
        camera_x, camera_y, camera_z = expected_centers[source_frame]
        optical_axis_point = np.asarray([0.0, 0.0, camera_z - 1.02, 1.0])
        world_point = (
            np.asarray(entry["target_from_camera_transform"])
            @ optical_axis_point
        )
        np.testing.assert_allclose(
            world_point,
            [camera_x, camera_y, 1.02, 1.0],
            rtol=0.0,
            atol=2e-6,
        )


@pytest.mark.parametrize(
    ("entries", "message"),
    [
        (
            [_entry(_CAMERA_FRAME, np.eye(4)), _entry(_CAMERA_FRAME, np.eye(4))],
            "source_frame values must be unique",
        ),
        (
            [_entry(_CAMERA_FRAME, np.diag([1.0, 1.0, -1.0, 1.0]))],
            "rigid homogeneous transform",
        ),
        (
            [_entry(_CAMERA_FRAME, np.eye(4), target_frame="robot_base")],
            "target_frame must be world",
        ),
        (
            [_entry(_CAMERA_FRAME, np.eye(4), valid_from_ns=2, valid_until_ns=1)],
            "valid_until_ns",
        ),
        (
            [_entry(_CAMERA_FRAME, np.eye(4), provenance_sha256="untrusted")],
            "provenance_sha256",
        ),
    ],
)
def test_manifest_rejects_duplicate_malformed_or_unpinned_entries(
    tmp_path: Path,
    entries: list[dict[str, object]],
    message: str,
) -> None:
    manifest_path = _write_manifest(tmp_path / "camera_to_world.json", entries)

    with pytest.raises(ValueError, match=message):
        load_camera_to_world_calibration_runtime(manifest_path)


def _write_camera_location_pose(
    interaction_root: Path,
    source_frame: str,
) -> Path:
    path = (
        interaction_root
        / "products/grounding/rgb_d_cad_grounding/pose_0001/pose_record.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": 3,
                "record_type": "CADPoseEstimationRecord",
                "CAD_correspondence": "accepted",
                "location": "available",
                "pose": "ambiguous",
                "coordinate_frame": source_frame,
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_manifest(path: Path, entries: list[dict[str, object]]) -> Path:
    path.write_text(
        json.dumps({"schema_version": 1, "calibrations": entries}),
        encoding="utf-8",
    )
    return path
