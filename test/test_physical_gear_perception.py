from __future__ import annotations

import importlib.util
import json
import math
import threading
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from cais_spade_llm.resources.robot.robot_primitives import _normalize_detected_item_output
from cais_spade_llm.resources.sensor.camera_module import CameraModule
from cais_spade_llm.resources.sensor.physical import calibrate_hand_eye
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
    RealSenseRoboflowNode,
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
    perception._frame_copy = lambda: (
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
    perception._frame_copy = lambda: (
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


def test_pose_rejection_clears_executable_detection_rows() -> None:
    perception = object.__new__(RealSenseRoboflowNode)
    perception._last_rows = [{"part_name": "SG", "frame_id": "world"}]
    perception._run_detection = lambda: (_ for _ in ()).throw(
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
