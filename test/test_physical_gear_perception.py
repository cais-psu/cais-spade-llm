from __future__ import annotations

import importlib.util
import json
import math
import sys
import threading
import time
import xml.etree.ElementTree as ET
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from cais_spade_llm.resources.robot.robot_primitives import _normalize_detected_item_output
from cais_spade_llm.resources.sensor.camera_module import CameraModule
from cais_spade_llm.resources.sensor.physical import calibrate_hand_eye, calibration_pose_replay
from cais_spade_llm.resources.sensor.physical.assembly_board_v1_aruco import (
    ArucoLocalizationError,
    ArucoPoseEstimate,
    CameraCalibration,
)
from cais_spade_llm.resources.sensor.physical.realsense_pose_estimator import (
    CalibrationError,
    ColorIntrinsics,
    DepthQualityError,
    RigidTransform,
    constrain_gear_center_to_table_plane,
    deproject_pixel,
    gear_center_world_point,
    load_hand_eye_calibration,
    pose_motion,
    robust_surface_depth,
    table_surface_z_from_calibration,
)
from cais_spade_llm.resources.sensor.physical.realsense_roboflow_node import (
    STATIONARY_REGISTRATION_UNAVAILABLE,
    RealSenseRoboflowNode,
    _load_stationary_inspection_geometry,
    _stationary_aruco_evidence,
    _stationary_aruco_window_quality,
    _stationary_inspection_from_rows,
    _StationaryArucoObservation,
)
from cais_spade_llm.resources.sensor.physical.roboflow_detector import (
    DuplicateDetectionError,
    GearBoundingBox,
    RoboflowGearDetector,
    RoboflowResponseError,
    RoboflowSettings,
    parse_gear_predictions,
)

ROOT = Path(__file__).resolve().parents[1]


def test_calibration_preview_records_missing_move_action(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeNode:
        def __init__(self, _name: str) -> None:
            pass

        def destroy_node(self) -> None:
            pass

    class FakeActionClient:
        def __init__(self, _node: object, _action: object, _name: str) -> None:
            pass

        @staticmethod
        def wait_for_server(*, timeout_sec: float) -> bool:
            assert timeout_sec == 10.0
            return False

    fake_rclpy = SimpleNamespace(
        init=lambda: None,
        ok=lambda: True,
        shutdown=lambda: None,
    )
    monkeypatch.setitem(sys.modules, "rclpy", fake_rclpy)
    monkeypatch.setitem(
        sys.modules,
        "moveit_msgs.action",
        SimpleNamespace(MoveGroup=object),
    )
    monkeypatch.setitem(
        sys.modules,
        "rclpy.action",
        SimpleNamespace(ActionClient=FakeActionClient),
    )
    monkeypatch.setitem(sys.modules, "rclpy.node", SimpleNamespace(Node=FakeNode))

    poses_path = tmp_path / "poses.yaml"
    poses_path.write_text(yaml.safe_dump({"poses": [{} for _ in range(20)]}), encoding="utf-8")
    status_path = tmp_path / "status.json"
    with pytest.raises(RuntimeError, match="/move_action is unavailable"):
        calibration_pose_replay.replay(
            SimpleNamespace(
                confirmed=False,
                preview_only=True,
                poses=str(poses_path),
                control=str(tmp_path / "control.json"),
                status=str(status_path),
                camera_role="xarm6",
                planning_group="xarm6",
                capture_command=[],
            )
        )

    status = json.loads(status_path.read_text(encoding="utf-8"))
    assert status["state"] == "preview_failed"
    assert status["pose_index"] == 0
    assert status["pose_count"] == 20
    assert status["error"] == (
        "/move_action is unavailable; start the correct hardware MoveIt stack"
    )


def _twin_sync_module() -> object:
    path = ROOT / "ros2/cais_lab_robotics/scripts/physical_part_twin_sync.py"
    spec = importlib.util.spec_from_file_location("physical_part_twin_sync_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _module_from_path(name: str, relative_path: str) -> object:
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _prediction(label: str, confidence: float = 0.9) -> dict[str, float | str]:
    return {
        "class": label,
        "confidence": confidence,
        "x": 50.0,
        "y": 50.0,
        "width": 40.0,
        "height": 40.0,
    }


def _box() -> GearBoundingBox:
    return GearBoundingBox(
        part_name="SG",
        model_name="gear_small",
        label="small_gear",
        confidence=0.9,
        center_x=50.0,
        center_y=50.0,
        width=40.0,
        height=40.0,
    )


def test_exact_roboflow_mapping_and_lg_blocking() -> None:
    rows = parse_gear_predictions(
        [
            {
                "predictions": {
                    "predictions": [
                        _prediction("small_gear"),
                        _prediction("medium_gear", 0.8),
                        _prediction("large_gear"),
                        _prediction("small_rectangular_pin"),
                    ]
                }
            }
        ]
    )
    assert [(row.part_name, row.model_name) for row in rows] == [
        ("SG", "gear_small"),
        ("MG", "gear_medium"),
    ]


def test_low_confidence_and_duplicate_detections_are_rejected() -> None:
    assert parse_gear_predictions({"predictions": [_prediction("small_gear", 0.69)]}) == []
    with pytest.raises(DuplicateDetectionError):
        parse_gear_predictions(
            {"predictions": [_prediction("small_gear"), _prediction("small_gear")]}
        )


def test_model_response_mismatch_is_explicit() -> None:
    with pytest.raises(RoboflowResponseError, match="predictions"):
        parse_gear_predictions({"output": []})


def test_model_receives_numpy_image_and_model_id() -> None:
    class FakeClient:
        image: object | None = None
        model_id = ""

        def infer(self, image: object, *, model_id: str) -> dict[str, list[object]]:
            self.image = image
            self.model_id = model_id
            return {"predictions": []}

    client = FakeClient()
    detector = RoboflowGearDetector(
        RoboflowSettings(api_key="test"),
        client=client,
    )
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    assert detector.detect(image) == []
    assert client.image is image
    assert client.model_id == "hrc-assembly-gph6m/5"


def test_network_error_does_not_create_a_detection() -> None:
    class FailingClient:
        def infer(self, _image: object, *, model_id: str) -> object:
            assert model_id == "hrc-assembly-gph6m/5"
            raise OSError("network unavailable?api_key=test-secret")

    detector = RoboflowGearDetector(
        RoboflowSettings(api_key="test-secret"),
        client=FailingClient(),
    )
    with pytest.raises(RoboflowResponseError, match=r"api_key=\[redacted\]") as error:
        detector.detect(np.zeros((8, 8, 3), dtype=np.uint8))
    assert "test-secret" not in str(error.value)


def test_depth_filter_ignores_hole_table_and_sparse_outliers() -> None:
    depth = np.full((100, 100), 1.0, dtype=np.float32)
    depth[32:68, 32:68] = 0.500
    depth[45:55, 45:55] = 0.0
    depth[34:38, 34:38] = 0.100
    estimate = robust_surface_depth(depth, _box())
    assert estimate.depth_m == pytest.approx(0.500, abs=0.001)
    assert estimate.sample_count >= 25
    assert estimate.mad_m <= 0.003


def test_depth_quality_gates_sparse_and_unstable_regions() -> None:
    sparse = np.zeros((100, 100), dtype=np.float32)
    sparse[48:52, 48:52] = 0.5
    with pytest.raises(DepthQualityError):
        robust_surface_depth(sparse, _box())

    unstable = np.zeros((100, 100), dtype=np.float32)
    unstable[32:68, 32:68] = np.where(
        np.indices((36, 36)).sum(axis=0) % 2 == 0,
        0.500,
        0.508,
    )
    with pytest.raises(DepthQualityError):
        robust_surface_depth(unstable, _box())


def test_deprojection_transform_and_top_to_center_correction() -> None:
    point = deproject_pixel(
        330.0,
        250.0,
        0.5,
        ColorIntrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0),
    )
    assert point.tolist() == pytest.approx([0.01, 0.01, 0.5])
    world = gear_center_world_point(
        point,
        RigidTransform(
            translation=(1.0, 2.0, 3.0),
            quaternion=(0.0, 0.0, 0.0, 1.0),
        ),
    )
    assert world.tolist() == pytest.approx([1.01, 2.01, 3.49])


def test_table_plane_estimation_rejects_outliers_and_constrains_center(
    tmp_path: Path,
) -> None:
    frames = [
        [
            {
                "part_name": "SG",
                "frame_id": "world",
                "z": 1.038 + index * 0.00005,
            },
            {
                "part_name": "MG",
                "frame_id": "world",
                "z": 1.0384 + index * 0.00005,
            },
        ]
        for index in range(10)
    ]
    frames[0].append(
        {"part_name": "SG", "frame_id": "world", "z": 1.200}
    )
    table_plane = calibrate_hand_eye.estimate_table_plane(frames)
    assert table_plane["accepted"] is True
    assert table_plane["frame_count"] == 10
    assert table_plane["rejected_sample_count"] == 1
    assert table_plane["surface_z_m"] == pytest.approx(1.028425, abs=0.0003)
    assert table_plane["mad_m"] <= 0.002
    assert table_plane["timestamp"] > 0.0

    calibration_path = tmp_path / "calibration.yaml"
    calibration_path.write_text(
        yaml.safe_dump(
            {
                "calibration_id": "existing-hand-eye",
                "validation": {"accepted": True},
            }
        ),
        encoding="utf-8",
    )
    updated = calibrate_hand_eye.write_table_plane_calibration(
        calibration_path,
        table_plane,
    )
    assert updated["calibration_id"] == "existing-hand-eye"
    assert updated["table_plane"]["sample_count"] == 20
    assert updated["table_plane"]["timestamp"] == table_plane["timestamp"]

    calibration = {"table_plane": table_plane}
    surface_z = table_surface_z_from_calibration(calibration)
    assert surface_z == pytest.approx(table_plane["surface_z_m"])
    constrained = constrain_gear_center_to_table_plane(
        np.array([0.1, -0.2, surface_z + 0.011]),
        surface_z,
    )
    assert constrained.tolist() == pytest.approx([0.1, -0.2, surface_z + 0.010])


def test_table_plane_rejects_class_disagreement_and_preserves_previous_file(
    tmp_path: Path,
) -> None:
    frames = [
        [
            {"part_name": "SG", "frame_id": "world", "z": 1.038},
            {"part_name": "MG", "frame_id": "world", "z": 1.044},
        ]
        for _ in range(10)
    ]
    with pytest.raises(RuntimeError, match="SG/MG median disagreement"):
        calibrate_hand_eye.estimate_table_plane(frames)

    calibration_path = tmp_path / "calibration.yaml"
    previous = {
        "validation": {"accepted": True},
        "table_plane": {"accepted": True, "surface_z_m": 1.02},
    }
    calibration_path.write_text(yaml.safe_dump(previous), encoding="utf-8")
    with pytest.raises(RuntimeError, match="rejected"):
        calibrate_hand_eye.write_table_plane_calibration(
            calibration_path,
            {"accepted": False},
        )
    assert yaml.safe_load(calibration_path.read_text(encoding="utf-8")) == previous


def test_table_plane_constraint_rejects_inconsistent_observed_height() -> None:
    with pytest.raises(CalibrationError, match="disagrees"):
        constrain_gear_center_to_table_plane(
            np.array([0.0, 0.0, 1.050]),
            1.028,
        )


def test_robot_movement_gate_measurements() -> None:
    before = RigidTransform((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
    half_angle = math.radians(0.5)
    after = RigidTransform((0.0011, 0.0, 0.0), (0.0, 0.0, math.sin(half_angle), math.cos(half_angle)))
    translation_m, rotation_deg = pose_motion(before, after)
    assert translation_m == pytest.approx(0.0011)
    assert rotation_deg == pytest.approx(1.0)


def test_calibration_loader_requires_accepted_validation(tmp_path: Path) -> None:
    missing = tmp_path / "missing.yaml"
    with pytest.raises(CalibrationError):
        load_hand_eye_calibration(missing)
    path = tmp_path / "hand_eye.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "tool0_to_camera_color_optical_frame": {},
                "validation": {"accepted": False},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(CalibrationError, match="not marked accepted"):
        load_hand_eye_calibration(path)


def _hand_eye_samples() -> dict[str, list[dict[str, object]]]:
    identity = np.eye(4).tolist()
    return {
        "samples": [
            {
                "base_to_tool": identity,
                "camera_to_board": identity,
                "camera_link_to_optical": identity,
                "reprojection_error_px": 0.1,
            }
            for _ in range(20)
        ]
    }


def test_hand_eye_uses_park_with_horaud_agreement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeOpenCV:
        CALIB_HAND_EYE_PARK = 1
        CALIB_HAND_EYE_HORAUD = 2

        def __init__(self) -> None:
            self.methods: list[int] = []

        def calibrateHandEye(self, *_args: object, method: int) -> tuple[np.ndarray, np.ndarray]:
            self.methods.append(method)
            return np.eye(3), np.zeros((3, 1))

    fake_cv = FakeOpenCV()
    monkeypatch.setattr(calibrate_hand_eye, "_opencv", lambda: fake_cv)
    samples = tmp_path / "samples.json"
    samples.write_text(json.dumps(_hand_eye_samples()), encoding="utf-8")
    output = tmp_path / "calibration.yaml"

    payload = calibrate_hand_eye.solve_calibration(samples, output)

    assert fake_cv.methods == [fake_cv.CALIB_HAND_EYE_PARK, fake_cv.CALIB_HAND_EYE_HORAUD]
    assert payload["method"] == "CALIB_HAND_EYE_PARK"
    assert payload["validation"]["cross_check_method"] == "CALIB_HAND_EYE_HORAUD"
    assert payload["validation"]["accepted"] is True


def test_hand_eye_rejects_park_horaud_disagreement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeOpenCV:
        CALIB_HAND_EYE_PARK = 1
        CALIB_HAND_EYE_HORAUD = 2

        def calibrateHandEye(
            self,
            *_args: object,
            method: int,
        ) -> tuple[np.ndarray, np.ndarray]:
            translation = np.zeros((3, 1))
            if method == self.CALIB_HAND_EYE_HORAUD:
                translation[0, 0] = 0.002
            return np.eye(3), translation

    monkeypatch.setattr(calibrate_hand_eye, "_opencv", lambda: FakeOpenCV())
    samples = tmp_path / "samples.json"
    samples.write_text(json.dumps(_hand_eye_samples()), encoding="utf-8")

    with pytest.raises(RuntimeError, match="PARK/HORAUD disagreement"):
        calibrate_hand_eye.solve_calibration(samples, tmp_path / "calibration.yaml")


def test_xarm6_calibration_uses_controller_pose_and_active_offset() -> None:
    capture = object.__new__(calibrate_hand_eye._CaptureNode)
    capture.camera_role = "xarm6"
    capture.world_frame = "world"
    capture.tool_frame = "link_eef"
    capture.xarm6_robot_state_received_monotonic = time.monotonic()
    capture.xarm6_robot_state = SimpleNamespace(
        pose=[100.0, 200.0, 300.0, 0.0, 0.0, math.pi / 2.0],
        offset=[0.0, 0.0, 100.0, 0.0, 0.0, 0.0],
    )
    capture.transform = lambda target, source: (
        np.eye(4)
        if (target, source) == ("world", "link_base")
        else pytest.fail(f"unexpected TF lookup: {target} -> {source}")
    )

    world_to_eef = capture.calibration_tool_pose("world", "link_eef")

    assert world_to_eef[:3, 3] == pytest.approx([0.1, 0.2, 0.2])
    assert world_to_eef[:3, :3] == pytest.approx(
        calibrate_hand_eye._xarm6_pose_matrix(
            [0.0, 0.0, 0.0, 0.0, 0.0, math.pi / 2.0]
        )[:3, :3]
    )


def test_xarm6_calibration_rejects_stale_controller_pose() -> None:
    capture = object.__new__(calibrate_hand_eye._CaptureNode)
    capture.camera_role = "xarm6"
    capture.world_frame = "world"
    capture.tool_frame = "link_eef"
    capture.xarm6_robot_state = SimpleNamespace(
        pose=[0.0] * 6,
        offset=[0.0] * 6,
    )
    capture.xarm6_robot_state_received_monotonic = time.monotonic() - 3.0

    with pytest.raises(RuntimeError, match="fresh /xarm6/xarm/robot_states"):
        capture.calibration_tool_pose("world", "link_eef")


def test_xarm6_solve_rejects_legacy_tf_samples(tmp_path: Path) -> None:
    samples = tmp_path / "samples.json"
    samples.write_text(json.dumps(_hand_eye_samples()), encoding="utf-8")

    with pytest.raises(RuntimeError, match="legacy TF robot pose"):
        calibrate_hand_eye.solve_calibration(
            samples,
            tmp_path / "calibration.yaml",
            camera_role="xarm6",
            parent_frame="link_eef",
        )


def test_xarm6_solve_accepts_controller_fk_samples(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeOpenCV:
        CALIB_HAND_EYE_PARK = 1
        CALIB_HAND_EYE_HORAUD = 2

        @staticmethod
        def calibrateHandEye(
            *_args: object,
            method: int,
        ) -> tuple[np.ndarray, np.ndarray]:
            assert method in {1, 2}
            return np.eye(3), np.zeros((3, 1))

    monkeypatch.setattr(calibrate_hand_eye, "_opencv", lambda: FakeOpenCV())
    payload = _hand_eye_samples()
    for sample in payload["samples"]:
        sample["robot_pose_source"] = "xarm6_controller_fk"
    samples = tmp_path / "samples.json"
    samples.write_text(json.dumps(payload), encoding="utf-8")

    calibration = calibrate_hand_eye.solve_calibration(
        samples,
        tmp_path / "calibration.yaml",
        camera_role="xarm6",
        parent_frame="link_eef",
    )

    assert calibration["validation"]["accepted"] is True


def test_physical_perception_converts_missing_tf_to_calibration_error() -> None:
    class FakeTF2:
        class TransformException(Exception):
            pass

    class FakeBuffer:
        def lookup_transform(self, *_args: object, **_kwargs: object) -> object:
            raise FakeTF2.TransformException("world does not exist")

    class FakeRclpy:
        class time:
            class Time:
                pass

        class duration:
            class Duration:
                def __init__(self, *, seconds: float) -> None:
                    self.seconds = seconds

    perception = object.__new__(RealSenseRoboflowNode)
    perception._rclpy = FakeRclpy()
    perception._tf2_ros = FakeTF2()
    perception._tf_buffer = FakeBuffer()

    with pytest.raises(CalibrationError, match="TF unavailable for world <- tool0"):
        perception._lookup_transform("world", "tool0")


def test_missing_world_tf_still_writes_visual_detection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    perception = object.__new__(RealSenseRoboflowNode)
    perception._inference_lock = threading.Lock()
    perception._last_rows = [{"part_name": "old"}]
    perception._last_inference_latency_ms = None
    perception._roboflow_model_validated = False
    perception.camera_role = "ur5e"
    perception.world_frame = "world"
    perception.tool_frame = "tool0"
    perception.camera_optical_frame = "camera_color_optical_frame"
    stamp = SimpleNamespace(sec=10, nanosec=0)
    color = np.zeros((100, 100, 3), dtype=np.uint8)
    depth = np.full((100, 100), 0.5, dtype=np.float32)
    perception._frame_copy = lambda **_kwargs: (
        stamp,
        color,
        depth,
        ColorIntrinsics(fx=100.0, fy=100.0, cx=50.0, cy=50.0),
    )
    perception._lookup_transform = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        CalibrationError("TF unavailable for world <- tool0: world does not exist")
    )
    detector_calls: list[np.ndarray] = []
    perception._detector = SimpleNamespace(
        detect=lambda image: detector_calls.append(image) or [_box()],
        settings=SimpleNamespace(model_id="hrc-assembly-gph6m/5"),
    )
    previews: list[dict[str, object]] = []
    statuses: list[dict[str, object]] = []
    perception._write_detection_preview = lambda *_args, **kwargs: previews.append(kwargs)
    perception._write_detection_status = lambda *_args, **kwargs: statuses.append(kwargs)
    perception._reload_table_plane_calibration = lambda: None
    monkeypatch.setattr(
        "cais_spade_llm.resources.sensor.physical.realsense_roboflow_node.time.sleep",
        lambda _seconds: None,
    )

    with pytest.raises(CalibrationError, match="TF unavailable for world <- tool0"):
        perception._run_detection()

    assert len(detector_calls) == 1
    assert detector_calls[0] is color
    assert previews[0]["world_pose_ready"] is False
    assert "world <- tool0" in str(previews[0]["pose_error"])
    assert statuses[-1]["world_pose_ready"] is False
    assert perception._last_rows == []


def test_valid_tf_keeps_world_pose_payload_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    perception = object.__new__(RealSenseRoboflowNode)
    perception._inference_lock = threading.Lock()
    perception._last_rows = []
    perception._last_error = ""
    perception._last_inference_latency_ms = None
    perception._roboflow_model_validated = False
    perception._table_surface_z_m = None
    perception.camera_role = "ur5e"
    perception.world_frame = "world"
    perception.tool_frame = "tool0"
    perception.camera_optical_frame = "camera_color_optical_frame"
    stamp = SimpleNamespace(sec=10, nanosec=0)
    color = np.zeros((100, 100, 3), dtype=np.uint8)
    depth = np.full((100, 100), 0.5, dtype=np.float32)
    perception._frame_copy = lambda **_kwargs: (
        stamp,
        color,
        depth,
        ColorIntrinsics(fx=100.0, fy=100.0, cx=50.0, cy=50.0),
    )
    perception._lookup_transform = lambda *_args, **_kwargs: RigidTransform(
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )
    perception._detector = SimpleNamespace(
        detect=lambda _image: [_box()],
        settings=SimpleNamespace(model_id="hrc-assembly-gph6m/5"),
    )
    perception.node = SimpleNamespace(
        get_parameter=lambda name: SimpleNamespace(
            value={"minimum_depth_samples": 25, "maximum_depth_mad_m": 0.003}[name]
        )
    )
    statuses: list[dict[str, object]] = []
    snapshots: list[list[dict[str, object]]] = []
    perception._write_detection_preview = lambda *_args, **_kwargs: None
    perception._write_detection_status = lambda *_args, **kwargs: statuses.append(kwargs)
    perception._write_snapshot = lambda rows: snapshots.append(rows)
    perception._reload_table_plane_calibration = lambda: None
    monkeypatch.setattr(
        "cais_spade_llm.resources.sensor.physical.realsense_roboflow_node.time.sleep",
        lambda _seconds: None,
    )

    rows = perception._run_detection()

    assert rows[0]["part_name"] == "SG"
    assert rows[0]["model_name"] == "gear_small"
    assert rows[0]["frame_id"] == "world"
    assert rows[0]["z"] == pytest.approx(0.49)
    assert statuses[-1]["world_pose_ready"] is True
    assert snapshots[-1] == rows

    perception._table_surface_z_m = 1.0
    perception._last_rows = [{"part_name": "existing_executable_pose"}]
    statuses.clear()
    snapshots.clear()
    measurement_rows = perception._run_detection(
        constrain_table_plane=False,
        publish_executable_snapshot=False,
    )

    assert measurement_rows[0]["z"] == pytest.approx(0.49)
    assert perception._last_rows == [{"part_name": "existing_executable_pose"}]
    assert snapshots == []
    assert statuses[-1]["world_pose_ready"] is False
    assert "calibration measurement only" in str(statuses[-1]["pose_error"])


def _stationary_camera_calibration(
    *,
    frame_id: str = "stationary_camera_color_optical_frame",
    width: int = 100,
) -> CameraCalibration:
    return CameraCalibration(
        camera_matrix=np.array(
            [[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        ),
        distortion=np.zeros(5, dtype=np.float64),
        frame_id=frame_id,
        width=width,
        height=100,
        distortion_model="plumb_bob",
    )


def _stationary_aruco_observation(
    captured_at: float,
    *,
    camera_to_aruco: np.ndarray | None = None,
    camera: CameraCalibration | None = None,
) -> _StationaryArucoObservation:
    transform = (
        np.eye(4, dtype=np.float64)
        if camera_to_aruco is None
        else np.asarray(camera_to_aruco, dtype=np.float64)
    )
    return _StationaryArucoObservation(
        stamp_ns=int(round(captured_at * 1e9)),
        captured_at=captured_at,
        camera=camera or _stationary_camera_calibration(),
        estimate=ArucoPoseEstimate(
            camera_to_aruco=transform,
            corners=np.zeros((4, 2), dtype=np.float64),
            reprojection_error_px=0.2,
            alternative_reprojection_error_px=0.8,
            ambiguity_margin_px=0.6,
            minimum_corner_depth_m=0.45,
        ),
    )


def _configured_stationary_geometry(tmp_path: Path) -> dict[str, object]:
    geometry_path = tmp_path / "assembly_board-v1.json"
    payload = json.loads(
        (ROOT / "cais_spade_llm/specification/products/geometry/assembly_board-v1.json")
        .read_text(encoding="utf-8")
    )
    payload["real"]["assembly_board"][
        "assembly_board-v1_aruco_to_assembly_board-v1"
    ] = {
        "calibration_id": "stationary-board-registration-v1",
        "x": 0.0,
        "y": 0.0,
        "z": 0.0,
        "qx": 0.0,
        "qy": 0.0,
        "qz": 0.0,
        "qw": 1.0,
    }
    geometry_path.write_text(json.dumps(payload), encoding="utf-8")
    return _load_stationary_inspection_geometry(geometry_path)


def _stationary_row(part_name: str, point: tuple[float, float, float]) -> dict[str, object]:
    return {
        "part_name": part_name,
        "camera_x": point[0],
        "camera_y": point[1],
        "camera_z": point[2],
        "camera_frame_id": "stationary_camera_color_optical_frame",
        "confidence": 0.9,
        "depth_sample_count": 100,
        "depth_mad_m": 0.001,
    }


def test_stationary_geometry_reports_null_registration_unavailable() -> None:
    geometry = _load_stationary_inspection_geometry(
        ROOT / "cais_spade_llm/specification/products/geometry/assembly_board-v1.json"
    )

    assert geometry["registration_configured"] is False
    assert geometry["registration_error"] == STATIONARY_REGISTRATION_UNAVAILABLE
    assert geometry["expected_top_surfaces"]["SG"] == pytest.approx(
        (-0.1, 0.08, 0.025)
    )
    assert geometry["expected_top_surfaces"]["MG"] == pytest.approx(
        (0.0, 0.08, 0.025)
    )


def test_stationary_aruco_window_requires_fresh_stable_ten_frames() -> None:
    stable = [_stationary_aruco_observation(100.0 + index * 0.1) for index in range(10)]
    quality = _stationary_aruco_window_quality(stable)
    assert quality["sample_count"] == 10
    assert quality["translation_spread_m"] == pytest.approx(0.0)

    moved_transform = np.eye(4, dtype=np.float64)
    moved_transform[0, 3] = 0.0021
    unstable = [*stable[:-1], _stationary_aruco_observation(100.9, camera_to_aruco=moved_transform)]
    with pytest.raises(ArucoLocalizationError, match="pose is unstable"):
        _stationary_aruco_window_quality(unstable)

    stale = [_stationary_aruco_observation(100.0 + index * 0.25) for index in range(10)]
    with pytest.raises(ArucoLocalizationError, match="window is stale"):
        _stationary_aruco_window_quality(stale)


def test_stationary_aruco_requires_exact_stamp_and_resets_for_camera_info_change() -> None:
    perception = object.__new__(RealSenseRoboflowNode)
    perception._lock = threading.RLock()
    perception.camera_role = "stationary"
    perception.camera_optical_frame = "stationary_camera_color_optical_frame"
    perception._stationary_marker_length_m = 0.076
    perception._stationary_inspection_geometry = {}
    perception._last_stationary_inspection = {}
    observations = [
        _stationary_aruco_observation(100.0 + index * 0.1) for index in range(10)
    ]
    perception._stationary_aruco_observations = deque(observations, maxlen=20)
    perception._stationary_aruco_error = ""
    perception._stationary_aruco_error_stamp_ns = None

    stamp = SimpleNamespace(sec=100, nanosec=900_000_000)
    exact, evidence = perception._stationary_aruco_for_frame(stamp)
    assert exact.stamp_ns == observations[-1].stamp_ns
    assert evidence["captured_at"] == pytest.approx(100.9)
    assert evidence["sample_count"] == 10
    with pytest.raises(ArucoLocalizationError, match="exact detection frame"):
        perception._stationary_aruco_for_frame(
            SimpleNamespace(sec=100, nanosec=950_000_000)
        )

    perception._latest_camera_calibration = observations[-1].camera
    perception._latest_intrinsics = None
    perception._latest_intrinsics_image_size = None
    changed_info = SimpleNamespace(
        k=[100.0, 0.0, 50.0, 0.0, 100.0, 50.0, 0.0, 0.0, 1.0],
        d=[0.0] * 5,
        header=SimpleNamespace(frame_id="changed_camera_color_optical_frame"),
        width=100,
        height=100,
        distortion_model="plumb_bob",
    )
    perception._on_camera_info(changed_info)
    assert not perception._stationary_aruco_observations
    assert "CameraInfo changed" in perception._last_stationary_inspection["message"]

    perception._stationary_aruco_observations.extend(observations)
    invalid_info = SimpleNamespace(**vars(changed_info))
    invalid_info.k = [0.0, 0.0, 50.0, 0.0, 100.0, 50.0, 0.0, 0.0, 1.0]
    perception._on_camera_info(invalid_info)
    assert perception._latest_camera_calibration is None
    assert perception._latest_intrinsics is None
    assert not perception._stationary_aruco_observations
    assert "invalid RealSense CameraInfo" in perception._last_stationary_inspection[
        "message"
    ]


def test_stationary_inspection_compensates_for_settled_camera_movement(
    tmp_path: Path,
) -> None:
    geometry = _configured_stationary_geometry(tmp_path)
    board_points = {
        "SG": np.array([-0.1, 0.08, 0.025], dtype=np.float64),
        "MG": np.array([0.0, 0.08, 0.025], dtype=np.float64),
    }
    fixed_observation = _stationary_aruco_observation(100.0)
    fixed_rows = [
        _stationary_row(part_name, tuple(point))
        for part_name, point in board_points.items()
    ]
    fixed = _stationary_inspection_from_rows(
        fixed_rows,
        geometry=geometry,
        observation=fixed_observation,
        aruco=_stationary_aruco_evidence(
            fixed_observation,
            _stationary_aruco_window_quality(
                [_stationary_aruco_observation(99.1 + index * 0.1) for index in range(10)]
            ),
            marker_length_m=0.076,
        ),
    )

    angle = math.radians(1.0)
    moved_transform = np.array(
        [
            [math.cos(angle), -math.sin(angle), 0.0, 0.004],
            [math.sin(angle), math.cos(angle), 0.0, -0.003],
            [0.0, 0.0, 1.0, 0.006],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    moved_observation = _stationary_aruco_observation(
        101.0,
        camera_to_aruco=moved_transform,
    )
    moved_rows = []
    for part_name, board_point in board_points.items():
        camera_point = moved_transform @ np.append(board_point, 1.0)
        moved_rows.append(_stationary_row(part_name, tuple(camera_point[:3])))
    moved = _stationary_inspection_from_rows(
        moved_rows,
        geometry=geometry,
        observation=moved_observation,
        aruco={"marker_length_m": 0.076, "captured_at": 101.0},
    )

    assert fixed["success"] is True
    assert moved["success"] is True
    for part in moved["parts"]:
        assert part["observed"]["x"] == pytest.approx(part["expected"]["x"])
        assert part["observed"]["y"] == pytest.approx(part["expected"]["y"])
        assert part["observed"]["z"] == pytest.approx(part["expected"]["z"])


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ([_stationary_row("SG", (-0.1, 0.08, 0.025))], "MG was not detected"),
        (
            [
                _stationary_row("SG", (0.0, 0.08, 0.025)),
                _stationary_row("MG", (-0.1, 0.08, 0.025)),
            ],
            "XY error",
        ),
        (
            [
                _stationary_row("SG", (-0.0899, 0.08, 0.025)),
                _stationary_row("MG", (0.0, 0.08, 0.025)),
            ],
            "XY error 10.10 mm",
        ),
        (
            [
                _stationary_row("SG", (-0.1, 0.08, 0.025)),
                _stationary_row("MG", (0.0, 0.08, 0.0301)),
            ],
            "seating error 5.10 mm",
        ),
    ],
)
def test_stationary_inspection_rejects_missing_wrong_or_unseated_parts(
    tmp_path: Path,
    rows: list[dict[str, object]],
    message: str,
) -> None:
    geometry = _configured_stationary_geometry(tmp_path)
    observation = _stationary_aruco_observation(100.0)
    inspection = _stationary_inspection_from_rows(
        rows,
        geometry=geometry,
        observation=observation,
        aruco={"marker_length_m": 0.076, "captured_at": 100.0},
    )
    assert inspection["available"] is True
    assert inspection["success"] is False
    assert message in inspection["message"]


def test_stationary_inspection_rejects_duplicate_accepted_detection(
    tmp_path: Path,
) -> None:
    geometry = _configured_stationary_geometry(tmp_path)
    rows = [
        _stationary_row("SG", (-0.1, 0.08, 0.025)),
        _stationary_row("SG", (-0.1, 0.08, 0.025)),
        _stationary_row("MG", (0.0, 0.08, 0.025)),
    ]
    with pytest.raises(DuplicateDetectionError, match="multiple accepted SG"):
        _stationary_inspection_from_rows(
            rows,
            geometry=geometry,
            observation=_stationary_aruco_observation(100.0),
            aruco={"marker_length_m": 0.076, "captured_at": 100.0},
        )


def test_stationary_run_uses_raw_depth_without_world_pose_or_tf(
    tmp_path: Path,
) -> None:
    perception = object.__new__(RealSenseRoboflowNode)
    perception._inference_lock = threading.Lock()
    perception._last_rows = []
    perception._last_error = ""
    perception._last_inference_latency_ms = None
    perception._roboflow_model_validated = False
    perception._table_surface_z_m = 1.0
    perception.camera_role = "stationary"
    perception.world_frame = "world"
    perception.tool_frame = "world"
    perception.camera_optical_frame = "stationary_camera_color_optical_frame"
    perception._stationary_marker_length_m = 0.076
    perception._stationary_inspection_geometry = _configured_stationary_geometry(
        tmp_path
    )
    perception._last_stationary_inspection = {}
    stamp = SimpleNamespace(sec=10, nanosec=0)
    color = np.zeros((100, 100, 3), dtype=np.uint8)
    depth = np.full((100, 100), 0.5, dtype=np.float32)
    perception._frame_copy = lambda **_kwargs: (
        stamp,
        color,
        depth,
        ColorIntrinsics(fx=100.0, fy=100.0, cx=50.0, cy=50.0),
    )
    marker_transform = np.eye(4, dtype=np.float64)
    marker_transform[2, 3] = 0.475
    window = [
        _stationary_aruco_observation(
            9.1 + index * 0.1,
            camera_to_aruco=marker_transform,
        )
        for index in range(10)
    ]
    observation = window[-1]
    marker_evidence = _stationary_aruco_evidence(
        observation,
        _stationary_aruco_window_quality(window),
        marker_length_m=0.076,
    )
    perception._stationary_aruco_for_frame = lambda _stamp: (
        observation,
        marker_evidence,
    )
    perception._wait_for_stationary_tool_pose = lambda: pytest.fail(
        "stationary inspection requested a tool pose"
    )
    perception._lookup_transform = lambda *_args, **_kwargs: pytest.fail(
        "stationary inspection requested TF"
    )
    perception._detector = SimpleNamespace(
        detect=lambda _image: [
            GearBoundingBox(
                part_name="SG",
                model_name="gear_small",
                label="small_gear",
                confidence=0.9,
                center_x=30.0,
                center_y=66.0,
                width=10.0,
                height=10.0,
            ),
            GearBoundingBox(
                part_name="MG",
                model_name="gear_medium",
                label="medium_gear",
                confidence=0.9,
                center_x=50.0,
                center_y=66.0,
                width=10.0,
                height=10.0,
            ),
        ],
        settings=SimpleNamespace(model_id="hrc-assembly-gph6m/5"),
    )
    perception.node = SimpleNamespace(
        get_parameter=lambda name: SimpleNamespace(
            value={"minimum_depth_samples": 25, "maximum_depth_mad_m": 0.003}[name]
        )
    )
    snapshots: list[list[dict[str, object]]] = []
    perception._write_detection_preview = lambda *_args, **_kwargs: None
    perception._write_detection_status = lambda *_args, **_kwargs: None
    perception._write_snapshot = lambda rows: snapshots.append(rows)
    perception._reload_table_plane_calibration = lambda: pytest.fail(
        "stationary inspection loaded table-plane calibration"
    )

    rows = perception._run_detection()

    assert len(rows) == 2
    assert rows[0]["frame_id"] == "assembly_board-v1"
    assert rows[0]["x"] == pytest.approx(-0.1)
    assert rows[0]["y"] == pytest.approx(0.08)
    assert rows[0]["z"] == pytest.approx(0.025)
    assert "observed_center_z" not in rows[0]
    assert "table_surface_z_m" not in rows[0]
    assert rows[0]["stationary_inspection"]["observed"]["z"] == pytest.approx(
        0.025
    )
    assert perception._last_stationary_inspection["success"] is True
    assert perception._last_stationary_inspection["diagnostic_only"] is True
    assert perception._last_stationary_inspection["aruco"]["captured_at"] == 10.0
    assert snapshots[-1] == rows


def test_stationary_role_skips_calibration_and_disables_canonical_services() -> None:
    perception = object.__new__(RealSenseRoboflowNode)
    perception.camera_role = "stationary"
    perception.node = SimpleNamespace(
        get_parameter=lambda name: SimpleNamespace(
            value={"publish_canonical_services": True}[name]
        )
    )
    perception._initialize_world_pose_pipeline = lambda **_kwargs: pytest.fail(
        "stationary startup loaded calibration or initialized TF"
    )

    perception._initialize_role_pose_pipeline(
        calibration_path="/missing/stationary_realsense_extrinsic.yaml",
        table_plane_path="/missing/table_plane.yaml",
    )

    assert perception._canonical_services_enabled() is False

    calls: list[dict[str, str]] = []
    perception.camera_role = "ur5e"
    perception._initialize_world_pose_pipeline = lambda **kwargs: calls.append(kwargs)
    perception._initialize_role_pose_pipeline(
        calibration_path="/configured/ur5e.yaml",
        table_plane_path="/configured/table_plane.yaml",
    )
    assert calls == [
        {
            "calibration_path": "/configured/ur5e.yaml",
            "table_plane_path": "/configured/table_plane.yaml",
        }
    ]
    assert perception._canonical_services_enabled() is True


def test_stationary_null_registration_publishes_no_detection_rows() -> None:
    perception = object.__new__(RealSenseRoboflowNode)
    perception._inference_lock = threading.Lock()
    perception._last_rows = [{"part_name": "old", "frame_id": "world"}]
    perception.camera_role = "stationary"
    perception._stationary_marker_length_m = 0.076
    perception._stationary_inspection_geometry = _load_stationary_inspection_geometry(
        ROOT / "cais_spade_llm/specification/products/geometry/assembly_board-v1.json"
    )
    perception._stationary_inspection_geometry_error = ""
    perception._last_stationary_inspection = {}
    perception._frame_copy = lambda **_kwargs: pytest.fail(
        "null registration reached frame acquisition"
    )
    snapshots: list[list[dict[str, object]]] = []
    perception._write_snapshot = lambda rows: snapshots.append(rows)
    response = SimpleNamespace(success=True, message="")

    result = perception._service_result(response)

    assert result.success is False
    assert "is not configured" in result.message
    assert perception._last_rows == []
    assert snapshots == [[]]
    assert perception._last_stationary_inspection["available"] is False
    assert perception._last_stationary_inspection["registration"]["configured"] is False


def test_stationary_status_and_snapshot_are_inspection_only(tmp_path: Path) -> None:
    perception = object.__new__(RealSenseRoboflowNode)
    perception.camera_role = "stationary"
    perception._last_stationary_inspection = {
        "available": True,
        "success": False,
        "diagnostic_only": True,
        "message": "MG was not detected.",
    }
    perception._last_inference_latency_ms = 12.0
    perception.preview_dir = tmp_path / "preview"
    perception._write_detection_status(
        [],
        100.0,
        np.zeros((10, 10, 3), dtype=np.uint8),
        world_pose_ready=False,
        pose_error="",
    )
    status = json.loads(
        (perception.preview_dir / "detection_status.json").read_text(encoding="utf-8")
    )
    assert status["stage"] == "completed"
    assert status["world_pose_ready"] is False
    assert status["stationary_inspection_ready"] is True

    perception._lock = threading.RLock()
    perception._latest_frame = None
    perception._calibration = {}
    perception._table_surface_z_m = None
    perception._table_plane_calibration = {}
    perception._last_error = ""
    perception._roboflow_model_validated = True
    perception._detector = SimpleNamespace(
        settings=SimpleNamespace(model_id="hrc-assembly-gph6m/5")
    )
    perception.snapshot_path = tmp_path / "snapshot.json"
    perception._write_snapshot(
        [
            {
                "part_name": "SG",
                "frame_id": "assembly_board-v1",
                "x": -0.1,
                "y": 0.08,
                "z": 0.025,
            }
        ]
    )
    snapshot = json.loads(perception.snapshot_path.read_text(encoding="utf-8"))
    assert snapshot["inspection_only"] is True
    assert snapshot["calibration"] == {
        "mode": "inspection_only",
        "world_pose_required": False,
        "identity": None,
    }
    assert snapshot["table_plane_ready"] is False
    assert snapshot["table_plane"] is None
    assert snapshot["detections"][0]["frame_id"] == "assembly_board-v1"


def test_stationary_detect_part_explicitly_rejects_pin_parts() -> None:
    perception = object.__new__(RealSenseRoboflowNode)
    perception.camera_role = "stationary"
    response = SimpleNamespace(success=True, message="")

    result = perception._service_result(response, target_part="MCP")

    assert result.success is False
    assert result.message == "stationary assembly inspection does not support MCP"


def test_pose_rejection_clears_executable_detection_rows() -> None:
    perception = object.__new__(RealSenseRoboflowNode)
    perception._last_rows = [{"part_name": "SG", "frame_id": "world"}]
    perception._run_detection = lambda **_kwargs: (_ for _ in ()).throw(
        CalibrationError("TF unavailable for world <- tool0")
    )
    snapshots: list[list[dict[str, object]]] = []
    perception._write_snapshot = lambda rows: snapshots.append(rows)
    response = SimpleNamespace(success=True, message="")

    result = perception._service_result(response)

    assert result.success is False
    assert "TF unavailable" in result.message
    assert perception._last_rows == []
    assert snapshots == [[]]


@pytest.mark.parametrize(
    ("model_name", "mesh_name", "diameter_m", "offset"),
    [
        ("gear_small", "Gear_Small.STL", 0.0218776093, "-0.21316049955 0.1518883057 0.0697159157 3.141592653589793 0 0"),
        ("gear_medium", "Gear_Medium.STL", 0.0419959259, "-0.21316049955 0.18188829805 0.0697159157 3.141592653589793 0 0"),
        ("gear_large", "Gear_Large.STL", 0.0619944, "-0.2131604996 0.231888298 0.0697159157 3.141592653589793 0 0"),
    ],
)
def test_gazebo_gear_model_contract(
    model_name: str,
    mesh_name: str,
    diameter_m: float,
    offset: str,
) -> None:
    model_dir = ROOT / "ros2" / "cais_lab_robotics" / "models" / model_name
    root = ET.parse(model_dir / "model.sdf").getroot()
    model = root.find("model")
    assert model is not None and model.get("name") == model_name
    link = model.find("link")
    assert link is not None and link.get("name") == "link"
    radius = float(link.findtext("collision/geometry/cylinder/radius", "0"))
    assert radius * 2.0 == pytest.approx(diameter_m)
    assert float(link.findtext("collision/geometry/cylinder/length", "0")) == 0.02
    assert link.findtext("visual/geometry/mesh/scale") == "0.001 0.001 0.001"
    assert link.findtext("visual/pose") == offset
    assert (model_dir / "meshes" / mesh_name).is_file()


def test_world_and_mode_contracts_for_loose_parts() -> None:
    world = ET.parse(ROOT / "ros2/cais_lab_robotics/worlds/table.world").getroot()
    includes = {
        include.findtext("name"): include.findtext("uri") for include in world.iter("include")
    }
    assert includes["gear_small"] == "model://gear_small"
    assert includes["gear_medium"] == "model://gear_medium"
    assert includes["gear_large"] == "model://gear_large"
    single_world = (
        ROOT / "ros2/cais_lab_robotics/worlds/single_table.world"
    ).read_text(encoding="utf-8")
    assert "libgazebo_ros_state.so" in single_world
    assert "libgazebo_link_attacher.so" in single_world

    commands = (ROOT / "cais_spade_llm/ui/ros2_processes.py").read_text(encoding="utf-8")
    assert "gazebo_dual" in commands and "include_loose_parts:=true" in commands
    assert "gazebo_dual_passive" in commands and "include_loose_parts:=false" in commands


def test_passive_worlds_align_only_the_physical_ur5e_table() -> None:
    single_launch = _module_from_path(
        "ur5e_rg2_gazebo_alignment_test",
        "ros2/cais_lab_robotics/launch/ur5e_rg2_gazebo.launch.py",
    )
    single_generated = Path(
        single_launch._aligned_single_table_world(
            ROOT / "ros2/cais_lab_robotics/worlds/single_table.world",
            1.028,
        )
    )
    try:
        root = ET.parse(single_generated).getroot()
        work_table = next(
            include
            for include in root.iter("include")
            if include.findtext("name") == "work_table"
        )
        assert work_table.findtext("pose") == "0.0 0.0 0.013000000 0 0 0"
    finally:
        single_generated.unlink()

    dual_launch = _module_from_path(
        "xarm6_ur5e_gazebo_alignment_test",
        "ros2/cais_lab_robotics/launch/xarm6_ur5e_gazebo.launch.py",
    )
    dual_generated = Path(
        dual_launch._filtered_world(
            ROOT / "ros2/cais_lab_robotics/worlds/table.world",
            include_assembly_parts=True,
            include_loose_parts=False,
            table_surface_z_m=1.028,
        )
    )
    try:
        root = ET.parse(dual_generated).getroot()
        poses = {
            include.findtext("name"): include.findtext("pose")
            for include in root.iter("include")
        }
        assert poses["table_ur5e"] == "0.0 -0.40 0.013000000 0 0 0"
        assert poses["table_xarm6"] == "0.0 0.40 0 0 0 0"
    finally:
        dual_generated.unlink()


def test_dual_digital_twin_world_hides_only_prusa_printers_and_assembly_board() -> None:
    dual_launch = _module_from_path(
        "xarm6_ur5e_gazebo_digital_twin_world_test",
        "ros2/cais_lab_robotics/launch/xarm6_ur5e_gazebo.launch.py",
    )
    source_world = ROOT / "ros2/cais_lab_robotics/worlds/table.world"
    source_model_names = {
        element.get("name")
        if element.tag == "model"
        else str(element.findtext("name") or "").strip()
        for element in ET.parse(source_world).getroot().iter()
        if element.tag in {"model", "include"}
    }
    hidden_model_names = {
        "assembly_board_v1",
        "prusa_mk3",
        "prusa_mk4_1",
        "prusa_mk4_2",
    }
    assert hidden_model_names <= source_model_names

    dual_generated = Path(
        dual_launch._filtered_world(
            source_world,
            include_assembly_parts=True,
            include_loose_parts=False,
            include_prusa_printers_and_assembly_board=False,
            table_surface_z_m=1.028,
        )
    )
    try:
        root = ET.parse(dual_generated).getroot()
        generated_model_names = {
            element.get("name")
            if element.tag == "model"
            else str(element.findtext("name") or "").strip()
            for element in root.iter()
            if element.tag in {"model", "include"}
        }
        assert hidden_model_names.isdisjoint(generated_model_names)
        assert {
            "table_xarm6",
            "table_ur5e",
            "cam_mk3",
            "cam_mk4_1",
            "cam_mk4_2",
            "cam_assembly",
        } <= generated_model_names
        assert any(
            plugin.get("name") == "gazebo_ros_state"
            for plugin in root.iter("plugin")
        )
        assert any(
            plugin.get("name") == "gazebo_link_attacher"
            for plugin in root.iter("plugin")
        )
    finally:
        dual_generated.unlink()


@pytest.mark.parametrize(
    ("module_name", "launch_path"),
    [
        (
            "ur5e_rg2_gazebo_passive_collision_test",
            "ros2/cais_lab_robotics/launch/ur5e_rg2_gazebo.launch.py",
        ),
        (
            "xarm6_ur5e_gazebo_passive_collision_test",
            "ros2/cais_lab_robotics/launch/xarm6_ur5e_gazebo.launch.py",
        ),
    ],
)
def test_passive_ur5e_mount_collision_is_removed_without_changing_other_geometry(
    module_name: str,
    launch_path: str,
) -> None:
    module = _module_from_path(module_name, launch_path)
    root = ET.fromstring(
        """
        <robot name="test">
          <link name="ur5e_base_link_inertia">
            <visual name="base_visual"/>
            <collision name="base_collision"/>
            <inertial><mass value="4.0"/></inertial>
          </link>
          <link name="ur5e_shoulder_link">
            <collision name="shoulder_collision"/>
          </link>
        </robot>
        """
    )

    module._strip_passive_ur5e_mount_collision(root, "ur5e_")

    base = root.find("link[@name='ur5e_base_link_inertia']")
    shoulder = root.find("link[@name='ur5e_shoulder_link']")
    assert base is not None
    assert base.find("collision") is None
    assert base.find("visual") is not None
    assert base.find("inertial") is not None
    assert shoulder is not None and shoulder.find("collision") is not None


def test_passive_ur5e_mount_collision_suppression_is_passive_only() -> None:
    single_source = (
        ROOT / "ros2/cais_lab_robotics/launch/ur5e_rg2_gazebo.launch.py"
    ).read_text(encoding="utf-8")
    dual_source = (
        ROOT / "ros2/cais_lab_robotics/launch/xarm6_ur5e_gazebo.launch.py"
    ).read_text(encoding="utf-8")

    assert "_build_ur5e_rg2_description(\n        controllers_yaml,\n        passive=passive," in single_source
    assert "if passive:\n        _strip_passive_ur5e_mount_collision" in dual_source


def test_twin_sync_spawns_updates_and_freezes_held_parts(tmp_path: Path) -> None:
    module = _twin_sync_module()
    sync = object.__new__(module.PhysicalPartTwinSync)
    sync._degraded_reason = ""
    sync._waiting_reason = ""
    sync._mirrored_models = set()
    sync.deadband_m = 0.002
    sync._held_models = {"gear_medium"}
    sync._release_after = {}
    sync._process_ownership = lambda: None
    sync._accepted_rows = lambda: [
        {"part_name": "SG", "model_name": "gear_small", "captured_at": 10.0},
        {"part_name": "MG", "model_name": "gear_medium", "captured_at": 10.0},
    ]
    sync._entity_state = lambda model_name: None if model_name == "gear_small" else object()
    calls: list[tuple[str, str]] = []
    sync._spawn = lambda model_name, _row: calls.append(("spawn", model_name)) or True
    sync._update = lambda model_name, _row, _state: calls.append(("update", model_name)) or True
    status_path = tmp_path / "status.json"
    sync.status_path = status_path

    sync.sync_once()

    assert calls == [("spawn", "gear_small")]
    status = json.loads(status_path.read_text(encoding="utf-8"))
    assert status["mirrored_models"] == ["gear_small"]
    assert status["held_models"] == ["gear_medium"]
    assert status["state"] == "mirrored"
    assert status["synchronizer_pid"] > 0
    assert status["heartbeat_at"] == status["updated_at"]


def test_twin_sync_waits_for_late_validated_pose_then_spawns(tmp_path: Path) -> None:
    module = _twin_sync_module()
    snapshot_path = tmp_path / "perception.json"
    status_path = tmp_path / "status.json"
    snapshot_path.write_text(
        json.dumps(
            {
                "table_plane_ready": True,
                "detections": [],
                "last_error": (
                    "observed gear surface disagrees with the calibrated table plane by 6.15 mm"
                ),
            }
        ),
        encoding="utf-8",
    )
    sync = object.__new__(module.PhysicalPartTwinSync)
    sync.snapshot_path = snapshot_path
    sync.status_path = status_path
    sync.maximum_detection_age_sec = 15.0
    sync._degraded_reason = ""
    sync._waiting_reason = ""
    sync._mirrored_models = set()
    sync._held_models = set()
    sync._release_after = {}
    sync.deadband_m = 0.002
    sync._process_ownership = lambda: None
    sync._entity_state = lambda _model_name: None
    spawned: list[str] = []
    sync._spawn = lambda model_name, _row: spawned.append(model_name) or True

    sync.sync_once()

    waiting = json.loads(status_path.read_text(encoding="utf-8"))
    assert spawned == []
    assert waiting["state"] == "waiting"
    assert waiting["waiting_reason"] == (
        "world pose rejected: observed gear surface disagrees with the calibrated table plane "
        "by 6.15 mm"
    )

    captured_at = time.time()
    snapshot_path.write_text(
        json.dumps(
            {
                "table_plane_ready": True,
                "detections": [
                    {
                        "part_name": "SG",
                        "model_name": "gear_small",
                        "captured_at": captured_at,
                    }
                ],
                "last_error": "",
            }
        ),
        encoding="utf-8",
    )

    sync.sync_once()

    mirrored = json.loads(status_path.read_text(encoding="utf-8"))
    assert spawned == ["gear_small"]
    assert mirrored["state"] == "mirrored"
    assert mirrored["mirrored_models"] == ["gear_small"]
    assert mirrored["waiting_reason"] == ""


def test_twin_sync_singleton_lock_rejects_second_process(tmp_path: Path) -> None:
    module = _twin_sync_module()
    lock_path = tmp_path / "part_sync.lock"
    first = module._acquire_singleton_lock(lock_path)
    assert first is not None
    try:
        assert module._acquire_singleton_lock(lock_path) is None
    finally:
        module.fcntl.flock(first.fileno(), module.fcntl.LOCK_UN)
        first.close()


def test_bridge_starts_part_sync_only_after_gazebo_services_and_reconciles_late() -> None:
    bridge = (ROOT / "cais_spade_llm/ui/bridge.py").read_text(encoding="utf-8")
    start = bridge.index("    def _start_digital_twin_part_sync_when_ready(")
    end = bridge.index("    def _reconcile_physical_part_twin_sync(", start)
    method = bridge[start:end]

    services = '["/spawn_entity", "/get_entity_state", "/set_entity_state"]'
    assert services in method
    assert method.index(services) < method.index("_stop_stale_physical_part_twin_sync()")
    assert method.index("_stop_stale_physical_part_twin_sync()") < method.index(
        "_start_tracked_ros2_command("
    )
    assert "waiting to spawn" in method
    assert "_reconcile_physical_part_twin_sync" in bridge
    assert "def _active_digital_twin_target_from_status(" in bridge
    assert "self._physical_part_twin_reconcile_lock" in bridge
    assert "self._active_digital_twin_target_from_status()" in bridge
    assert "timeout_sec=3.0" in bridge

    manager = (ROOT / "cais_spade_llm/ui/perception_manager.py").read_text(
        encoding="utf-8"
    )
    assert "self._reconcile_part_twin_sync(key)" in manager


def test_twin_sync_does_not_fight_vertical_settling_inside_xy_deadband() -> None:
    module = _twin_sync_module()
    sync = object.__new__(module.PhysicalPartTwinSync)
    sync.deadband_m = 0.002
    sync._pose_from_row = lambda _row: SimpleNamespace(
        position=SimpleNamespace(x=0.001, y=0.0, z=1.038)
    )
    state = SimpleNamespace(
        pose=SimpleNamespace(position=SimpleNamespace(x=0.0, y=0.0, z=1.025))
    )
    sync._call = lambda *_args, **_kwargs: pytest.fail("vertical-only drift was updated")

    assert sync._update("gear_small", {}, state) is True


def test_twin_sync_ownership_attach_release_and_freshness(tmp_path: Path) -> None:
    module = _twin_sync_module()
    sync = object.__new__(module.PhysicalPartTwinSync)
    sync.ownership_path = tmp_path / "ownership.json"
    sync._last_ownership_sequence = ""
    sync._held_models = set()
    sync._release_after = {}
    actions: list[tuple[str, str]] = []
    sync._attach = lambda model_name: actions.append(("attach", model_name)) or True
    sync._detach = lambda model_name: actions.append(("detach", model_name)) or True

    sync.ownership_path.write_text(
        json.dumps(
            {
                "sequence": "1",
                "occurred_at": 10.0,
                "action": "held",
                "model_name": "gear_small",
            }
        ),
        encoding="utf-8",
    )
    sync._process_ownership()
    sync.ownership_path.write_text(
        json.dumps(
            {
                "sequence": "2",
                "occurred_at": 20.0,
                "action": "released",
                "model_name": "gear_small",
            }
        ),
        encoding="utf-8",
    )
    sync._process_ownership()

    assert actions == [("attach", "gear_small"), ("detach", "gear_small")]
    assert sync._release_after == {"gear_small": 20.0}


def test_recovery_detection_fact_preserves_world_pose_metadata() -> None:
    row = {
        "part_name": "SG",
        "model_name": "gear_small",
        "x": 0.1,
        "y": 0.2,
        "z": 0.3,
        "qx": 0.0,
        "qy": 0.0,
        "qz": 0.0,
        "qw": 1.0,
        "frame_id": "world",
        "confidence": 0.91,
        "captured_at": 100.0,
        "source": "realsense_roboflow",
        "model_id": "hrc-assembly-gph6m/5",
    }
    fact = _normalize_detected_item_output(row)
    assert fact is not None
    assert fact["pose"]["x"] == 0.1
    assert fact["frame_id"] == "world"
    assert fact["confidence"] == 0.91
    assert fact["source"] == "realsense_roboflow"
    assert fact["model_name"] == "gear_small"


def test_physical_runtime_has_no_yolo_placeholder_branch() -> None:
    controller = (
        ROOT / "cais_spade_llm/resources/robot/gazebo_pick_place_controller.py"
    ).read_text(encoding="utf-8")
    recovery = (
        ROOT
        / "cais_spade_llm/agents/intelligent_product/product_recovery_controller.py"
    ).read_text(encoding="utf-8")
    assert "Physical mode detect_parts is unavailable" not in controller
    assert "todo_yolo_placeholder" not in recovery
    assert 'perception_backend in {"", "none", "yolo"}' not in recovery


def test_yolo_camera_targets_realsense_parameter_node(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PERCEPTION_NODE_NAME", raising=False)
    camera = CameraModule(backend="yolo")
    assert camera._perception_node_name == "/realsense_roboflow_perception"


def test_physical_snapshot_client_rejects_error_and_stale_detection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cais_spade_llm.resources.sensor.physical import detect_all_service

    snapshot = tmp_path / "perception.json"
    monkeypatch.setattr(detect_all_service, "SNAPSHOT_PATH", snapshot)
    row = {
        "part_name": "SG",
        "model_name": "gear_small",
        "captured_at": 10.0,
        "x": 0.1,
        "y": 0.2,
        "z": 0.3,
    }
    snapshot.write_text(
        json.dumps(
            {
                "updated_at": 20.0,
                "last_error": "",
                "table_plane_ready": True,
                "detections": [row],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(detect_all_service.time, "time", lambda: 20.0)
    assert detect_all_service.detect_all()["SG"]["x"] == 0.1

    row["captured_at"] = 9.0
    snapshot.write_text(
        json.dumps(
            {
                "updated_at": 20.0,
                "last_error": "",
                "table_plane_ready": True,
                "detections": [row],
            }
        ),
        encoding="utf-8",
    )
    assert detect_all_service.detect_all() == {}

    row["captured_at"] = 20.0
    snapshot.write_text(
        json.dumps(
            {
                "updated_at": 20.0,
                "last_error": "UR5e moved during inference",
                "table_plane_ready": True,
                "detections": [row],
            }
        ),
        encoding="utf-8",
    )
    assert detect_all_service.detect_all() == {}

    snapshot.write_text(
        json.dumps(
            {
                "updated_at": 20.0,
                "last_error": "",
                "table_plane_ready": False,
                "detections": [row],
            }
        ),
        encoding="utf-8",
    )
    assert detect_all_service.detect_all() == {}
