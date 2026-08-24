from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import yaml

from cais_spade_llm.resources.sensor.physical.assembly_board_v1_aruco import (
    ARUCO_DICTIONARY_NAME,
    ARUCO_MARKER_ID,
    CHILD_FRAME_ID,
    DEFAULT_MARKER_LENGTH_M,
    ArucoLocalizationError,
    AssemblyBoardV1ArucoLocalizer,
    CalibrationProvenance,
    CameraCalibration,
    camera_calibration_from_info,
    detect_assembly_board_v1_aruco,
    estimate_camera_to_aruco,
    load_calibration_provenance,
    ros_stamp_to_epoch_seconds,
    transform_camera_optical_point_to_assembly_board_v1,
)


def _camera() -> CameraCalibration:
    return CameraCalibration(
        camera_matrix=np.array(
            [[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        ),
        distortion=np.zeros(5, dtype=np.float64),
        frame_id="camera_color_optical_frame",
        width=640,
        height=480,
        distortion_model="plumb_bob",
    )


def _provenance() -> CalibrationProvenance:
    return CalibrationProvenance(
        identity="accepted-calibration-id",
        path="/tmp/accepted-hand-eye.yaml",
        sha256="a" * 64,
        camera_role="ur5e",
        parent_frame="tool0",
        method="CALIB_HAND_EYE_PARK",
        parent_to_camera=np.eye(4, dtype=np.float64),
    )


def _transform(
    *,
    translation: tuple[float, float, float] = (0.0, 0.0, 0.0),
    rotation_deg: float = 0.0,
) -> np.ndarray:
    angle = np.deg2rad(rotation_deg)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    transform[:3, 3] = translation
    return transform


def _marker_image() -> np.ndarray:
    camera = _camera()
    half = DEFAULT_MARKER_LENGTH_M * 0.5
    object_points = np.array(
        [
            [-half, half, 0.0],
            [half, half, 0.0],
            [half, -half, 0.0],
            [-half, -half, 0.0],
        ],
        dtype=np.float64,
    )
    destination, _ = cv2.projectPoints(
        object_points,
        np.array([0.3, 0.2, 0.1], dtype=np.float64),
        np.array([0.02, -0.01, 0.5], dtype=np.float64),
        camera.camera_matrix,
        camera.distortion,
    )
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
    marker = cv2.aruco.generateImageMarker(dictionary, ARUCO_MARKER_ID, 240)
    source = np.array(
        [[0.0, 239.0], [239.0, 239.0], [239.0, 0.0], [0.0, 0.0]],
        dtype=np.float32,
    )
    homography = cv2.getPerspectiveTransform(
        source,
        destination.reshape(4, 2).astype(np.float32),
    )
    image = cv2.warpPerspective(
        marker,
        homography,
        (camera.width, camera.height),
        flags=cv2.INTER_NEAREST,
        borderValue=255,
    )
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


def test_original_dictionary_id_70_is_detected_and_solved_with_ippe() -> None:
    corners = detect_assembly_board_v1_aruco(_marker_image())
    estimate = estimate_camera_to_aruco(corners, _camera())

    assert corners.shape == (4, 2)
    assert estimate.camera_to_aruco[:3, 3] == pytest.approx(
        [0.02, -0.01, 0.5],
        abs=0.002,
    )
    assert estimate.reprojection_error_px <= 1.0
    assert estimate.ambiguity_margin_px >= 0.25
    assert estimate.minimum_corner_depth_m > 0.0


def test_camera_motion_keeps_part_point_fixed_in_assembly_board_v1() -> None:
    board_point = np.array([0.0, 0.08, 0.015], dtype=np.float64)
    registration = _transform(
        translation=(0.11, -0.04, 0.006),
        rotation_deg=12.0,
    )
    camera_poses = (
        _transform(translation=(0.02, -0.01, 0.48), rotation_deg=-8.0),
        _transform(translation=(-0.035, 0.025, 0.53), rotation_deg=19.0),
    )

    for camera_to_aruco in camera_poses:
        camera_point = (
            camera_to_aruco @ registration @ np.append(board_point, 1.0)
        )[:3]
        observed_board_point = transform_camera_optical_point_to_assembly_board_v1(
            camera_point,
            point_frame_id="camera_color_optical_frame",
            camera_frame_id="camera_color_optical_frame",
            camera_to_aruco=camera_to_aruco,
            assembly_board_v1_aruco_to_assembly_board_v1=registration,
        )

        assert observed_board_point == pytest.approx(board_point, abs=1e-12)


def test_part_point_requires_exact_realsense_camera_info_frame() -> None:
    with pytest.raises(ArucoLocalizationError, match="does not exactly match"):
        transform_camera_optical_point_to_assembly_board_v1(
            np.array([0.0, 0.0, 0.5], dtype=np.float64),
            point_frame_id="stationary_camera_color_optical_frame",
            camera_frame_id="camera_color_optical_frame",
            camera_to_aruco=np.eye(4, dtype=np.float64),
            assembly_board_v1_aruco_to_assembly_board_v1=np.eye(
                4,
                dtype=np.float64,
            ),
        )


@pytest.mark.parametrize(
    ("registration", "error"),
    [
        (np.eye(3, dtype=np.float64), "must be a 4x4 transform"),
        (
            np.array(
                [
                    [1.0, 0.0, 0.0, np.nan],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            ),
            "contains a non-finite value",
        ),
        (
            np.array(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.1, 1.0],
                ],
                dtype=np.float64,
            ),
            "has an invalid homogeneous row",
        ),
        (
            np.diag([2.0, 1.0, 1.0, 1.0]),
            "rotation is not orthonormal",
        ),
        (
            np.diag([-1.0, 1.0, 1.0, 1.0]),
            "rotation is not proper",
        ),
    ],
)
def test_invalid_assembly_board_v1_registration_fails_closed(
    registration: np.ndarray,
    error: str,
) -> None:
    with pytest.raises(
        ArucoLocalizationError,
        match=f"assembly_board-v1_aruco_to_assembly_board-v1 {error}",
    ):
        transform_camera_optical_point_to_assembly_board_v1(
            np.array([0.0, 0.0, 0.5], dtype=np.float64),
            point_frame_id="camera_color_optical_frame",
            camera_frame_id="camera_color_optical_frame",
            camera_to_aruco=np.eye(4, dtype=np.float64),
            assembly_board_v1_aruco_to_assembly_board_v1=registration,
        )


def test_nonfinite_camera_to_aruco_fails_closed() -> None:
    camera_to_aruco = np.eye(4, dtype=np.float64)
    camera_to_aruco[0, 3] = np.inf

    with pytest.raises(
        ArucoLocalizationError,
        match="camera_to_aruco contains a non-finite value",
    ):
        transform_camera_optical_point_to_assembly_board_v1(
            np.array([0.0, 0.0, 0.5], dtype=np.float64),
            point_frame_id="camera_color_optical_frame",
            camera_frame_id="camera_color_optical_frame",
            camera_to_aruco=camera_to_aruco,
            assembly_board_v1_aruco_to_assembly_board_v1=np.eye(
                4,
                dtype=np.float64,
            ),
        )


def test_ippe_rejects_frontal_pose_ambiguity() -> None:
    half = DEFAULT_MARKER_LENGTH_M * 0.5
    object_points = np.array(
        [
            [-half, half, 0.0],
            [half, half, 0.0],
            [half, -half, 0.0],
            [-half, -half, 0.0],
        ],
        dtype=np.float64,
    )
    camera = _camera()
    corners, _ = cv2.projectPoints(
        object_points,
        np.array([np.pi, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.5]),
        camera.camera_matrix,
        camera.distortion,
    )

    with pytest.raises(ArucoLocalizationError, match="IPPE pose is ambiguous"):
        estimate_camera_to_aruco(corners, camera)


def test_ippe_rejects_nonpositive_corner_depth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def negative_depth_solutions(*_args: object, **_kwargs: object) -> tuple[object, ...]:
        rvecs = (np.zeros((3, 1)), np.zeros((3, 1)))
        tvecs = (
            np.array([[0.0], [0.0], [-0.5]]),
            np.array([[0.0], [0.0], [-0.6]]),
        )
        return True, rvecs, tvecs, np.zeros((2, 1))

    monkeypatch.setattr(cv2, "solvePnPGeneric", negative_depth_solutions)
    with pytest.raises(ArucoLocalizationError, match="positive-depth"):
        estimate_camera_to_aruco(np.zeros((4, 2)), _camera())


def test_ten_frame_stability_and_failure_invalidation(tmp_path: Path) -> None:
    output = tmp_path / "ur5e" / "assembly_board-v1_aruco.json"
    localizer = AssemblyBoardV1ArucoLocalizer(
        camera_role="ur5e",
        world_frame="world",
        parent_frame="tool0",
        marker_length_m=DEFAULT_MARKER_LENGTH_M,
        output_path=output,
    )
    image = _marker_image()
    payload: dict[str, object] = {}
    for index in range(10):
        world_to_parent = np.eye(4, dtype=np.float64)
        world_to_parent[0, 3] = index * 0.0001
        payload, _ = localizer.observe(
            image=image,
            frame_captured_at=100.0 + index * 0.2,
            camera=_camera(),
            calibration=_provenance(),
            world_to_parent=world_to_parent,
        )

    assert payload["valid"] is True
    assert payload["visible"] is True
    assert payload["stable"] is True
    assert payload["world_pose_ready"] is True
    assert payload["sample_started_at"] == 100.0
    assert payload["frame_captured_at"] == pytest.approx(101.8)
    assert payload["sample_count"] == 10
    assert payload["required_sample_count"] == 10
    assert payload["frame_id"] == "world"
    assert payload["child_frame_id"] == CHILD_FRAME_ID
    assert payload["marker"] == {
        "dictionary": ARUCO_DICTIONARY_NAME,
        "id": ARUCO_MARKER_ID,
        "marker_length_m": DEFAULT_MARKER_LENGTH_M,
    }
    assert payload["pose"]["frame_id"] == "world"
    assert set(payload["pose"]) == {
        "frame_id",
        "child_frame_id",
        "x",
        "y",
        "z",
        "qx",
        "qy",
        "qz",
        "qw",
    }
    assert payload["stability"]["window_frame_count"] == 10
    assert payload["stability"]["translation_spread_m"] <= 0.002
    assert payload["stability"]["rotation_spread_deg"] <= 0.5
    assert payload["calibration"]["identity"] == "accepted-calibration-id"
    assert payload["calibration_id"] == "accepted-calibration-id"
    assert payload["calibration"]["camera_info_identity"] == _camera().identity
    assert payload["tf"]["captured_at"] == pytest.approx(101.8)
    assert json.loads(output.read_text(encoding="utf-8"))["valid"] is True

    invalid = localizer.invalidate(
        "assembly_board-v1 ArUco ID 70 is not visible",
        frame_captured_at=102.0,
        calibration=_provenance(),
        camera=_camera(),
    )
    persisted = json.loads(output.read_text(encoding="utf-8"))
    assert invalid["valid"] is False
    assert invalid["visible"] is False
    assert invalid["stable"] is False
    assert invalid["world_pose_ready"] is False
    assert persisted["valid"] is False
    assert persisted["pose"] is None
    assert persisted["sample_started_at"] is None
    assert persisted["sample_count"] == 0
    assert persisted["last_error"] == "assembly_board-v1 ArUco ID 70 is not visible"


def test_unstable_ten_frame_window_is_not_authoritative(tmp_path: Path) -> None:
    localizer = AssemblyBoardV1ArucoLocalizer(
        camera_role="xarm6",
        world_frame="world",
        parent_frame="link_eef",
        marker_length_m=DEFAULT_MARKER_LENGTH_M,
        output_path=tmp_path / "assembly_board-v1_aruco.json",
    )
    provenance = CalibrationProvenance(
        **{
            **vars(_provenance()),
            "camera_role": "xarm6",
            "parent_frame": "link_eef",
        }
    )
    for index in range(10):
        world_to_parent = np.eye(4, dtype=np.float64)
        world_to_parent[0, 3] = index * 0.0005
        payload, _ = localizer.observe(
            image=_marker_image(),
            frame_captured_at=200.0 + index,
            camera=_camera(),
            calibration=provenance,
            world_to_parent=world_to_parent,
        )

    assert payload["valid"] is False
    assert payload["pose"] is None
    assert payload["sample_count"] == 10
    assert "pose is unstable" in payload["last_error"]
    assert payload["stability"]["translation_spread_m"] > 0.002


def test_one_rotation_outlier_requires_a_new_consistent_tenth_sample(
    tmp_path: Path,
) -> None:
    localizer = AssemblyBoardV1ArucoLocalizer(
        camera_role="ur5e",
        world_frame="world",
        parent_frame="tool0",
        marker_length_m=DEFAULT_MARKER_LENGTH_M,
        output_path=tmp_path / "assembly_board-v1_aruco.json",
    )
    payload: dict[str, object] = {}
    for index in range(10):
        angle_deg = 0.8 if index == 5 else 0.0
        angle_rad = np.deg2rad(angle_deg)
        world_to_parent = np.eye(4, dtype=np.float64)
        world_to_parent[:3, :3] = np.array(
            [
                [np.cos(angle_rad), -np.sin(angle_rad), 0.0],
                [np.sin(angle_rad), np.cos(angle_rad), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        payload, _ = localizer.observe(
            image=_marker_image(),
            frame_captured_at=300.0 + index,
            camera=_camera(),
            calibration=_provenance(),
            world_to_parent=world_to_parent,
        )

    assert payload["valid"] is False
    assert payload["sample_count"] == 10
    assert payload["rotation_spread_deg"] == pytest.approx(0.8)

    payload, _ = localizer.observe(
        image=_marker_image(),
        frame_captured_at=310.0,
        camera=_camera(),
        calibration=_provenance(),
        world_to_parent=np.eye(4, dtype=np.float64),
    )

    assert payload["valid"] is True
    assert payload["sample_count"] == 10
    assert payload["sample_started_at"] == pytest.approx(300.0)
    assert payload["frame_captured_at"] == pytest.approx(310.0)
    assert payload["rotation_spread_deg"] <= 0.5


def test_newest_rotation_change_is_not_discarded_as_an_outlier(tmp_path: Path) -> None:
    localizer = AssemblyBoardV1ArucoLocalizer(
        camera_role="ur5e",
        world_frame="world",
        parent_frame="tool0",
        marker_length_m=DEFAULT_MARKER_LENGTH_M,
        output_path=tmp_path / "assembly_board-v1_aruco.json",
    )
    for index in range(10):
        payload, _ = localizer.observe(
            image=_marker_image(),
            frame_captured_at=400.0 + index,
            camera=_camera(),
            calibration=_provenance(),
            world_to_parent=np.eye(4, dtype=np.float64),
        )
    assert payload["valid"] is True

    angle_rad = np.deg2rad(0.8)
    moved_world_to_parent = np.eye(4, dtype=np.float64)
    moved_world_to_parent[:3, :3] = np.array(
        [
            [np.cos(angle_rad), -np.sin(angle_rad), 0.0],
            [np.sin(angle_rad), np.cos(angle_rad), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    payload, _ = localizer.observe(
        image=_marker_image(),
        frame_captured_at=410.0,
        camera=_camera(),
        calibration=_provenance(),
        world_to_parent=moved_world_to_parent,
    )

    assert payload["valid"] is False
    assert payload["sample_count"] == 10
    assert payload["frame_captured_at"] == pytest.approx(410.0)
    assert payload["rotation_spread_deg"] == pytest.approx(0.8)


def test_camera_info_and_hand_eye_provenance_are_exact(tmp_path: Path) -> None:
    info = SimpleNamespace(
        k=[600.0, 0.0, 320.0, 0.0, 600.0, 240.0, 0.0, 0.0, 1.0],
        d=[0.1, -0.2, 0.0, 0.0, 0.0],
        header=SimpleNamespace(frame_id="camera_color_optical_frame"),
        width=640,
        height=480,
        distortion_model="plumb_bob",
    )
    camera = camera_calibration_from_info(info)
    assert camera.camera_matrix[0, 0] == 600.0
    assert camera.distortion.tolist() == info.d

    path = tmp_path / "ur5e_realsense_hand_eye.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "calibration_id": "stable-calibration-id",
                "camera_role": "ur5e",
                "parent_frame": "tool0",
                "method": "CALIB_HAND_EYE_PARK",
                "parent_to_camera_color_optical_frame": {
                    "translation": {"x": 0.01, "y": 0.02, "z": 0.03},
                    "quaternion": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                },
                "validation": {"accepted": True},
            }
        ),
        encoding="utf-8",
    )
    provenance = load_calibration_provenance(
        path,
        camera_role="ur5e",
        parent_frame="tool0",
    )
    assert provenance.identity == "stable-calibration-id"
    assert provenance.path == str(path)
    assert provenance.parent_to_camera[:3, 3].tolist() == [0.01, 0.02, 0.03]
    assert len(provenance.sha256) == 64

    with pytest.raises(ArucoLocalizationError, match="camera_role does not match"):
        load_calibration_provenance(
            path,
            camera_role="xarm6",
            parent_frame="tool0",
        )


def test_ros_stamp_must_share_time_time_epoch() -> None:
    stamp = SimpleNamespace(sec=1_800_000_000, nanosec=250_000_000)
    assert ros_stamp_to_epoch_seconds(
        stamp,
        now_sec=1_800_000_001.0,
    ) == pytest.approx(1_800_000_000.25)
    assert ros_stamp_to_epoch_seconds(
        stamp,
        now_sec=1_799_999_999.25,
    ) == pytest.approx(1_800_000_000.25)
    assert ros_stamp_to_epoch_seconds(
        stamp,
        now_sec=1_800_000_002.25,
    ) == pytest.approx(1_800_000_000.25)
    with pytest.raises(ArucoLocalizationError, match="Unix-epoch compatible"):
        ros_stamp_to_epoch_seconds(stamp, now_sec=1_799_999_999.249)
    with pytest.raises(ArucoLocalizationError, match="Unix-epoch compatible"):
        ros_stamp_to_epoch_seconds(stamp, now_sec=1_800_000_002.251)
    with pytest.raises(ArucoLocalizationError, match="Unix-epoch compatible"):
        ros_stamp_to_epoch_seconds(stamp, now_sec=100.0)
