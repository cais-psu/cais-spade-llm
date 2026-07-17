from __future__ import annotations

import importlib.util
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

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
    deproject_pixel,
    gear_center_world_point,
    load_hand_eye_calibration,
    pose_motion,
    robust_surface_depth,
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


@pytest.mark.parametrize(
    ("model_name", "mesh_name", "diameter_m", "offset"),
    [
        ("gear_small", "Gear_Small.STL", 0.0218776093, "-0.21316049955 -0.1518883057 -0.0697159157 0 0 0"),
        ("gear_medium", "Gear_Medium.STL", 0.0419959259, "-0.21316049955 -0.18188829805 -0.0697159157 0 0 0"),
        ("gear_large", "Gear_Large.STL", 0.0619944, "-0.2131604996 -0.231888298 -0.0697159157 0 0 0"),
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


def test_twin_sync_spawns_updates_and_freezes_held_parts(tmp_path: Path) -> None:
    module = _twin_sync_module()
    sync = object.__new__(module.PhysicalPartTwinSync)
    sync._degraded_reason = ""
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
        json.dumps({"updated_at": 20.0, "last_error": "", "detections": [row]}),
        encoding="utf-8",
    )
    monkeypatch.setattr(detect_all_service.time, "time", lambda: 20.0)
    assert detect_all_service.detect_all()["SG"]["x"] == 0.1

    row["captured_at"] = 9.0
    snapshot.write_text(
        json.dumps({"updated_at": 20.0, "last_error": "", "detections": [row]}),
        encoding="utf-8",
    )
    assert detect_all_service.detect_all() == {}

    row["captured_at"] = 20.0
    snapshot.write_text(
        json.dumps(
            {
                "updated_at": 20.0,
                "last_error": "UR5e moved during inference",
                "detections": [row],
            }
        ),
        encoding="utf-8",
    )
    assert detect_all_service.detect_all() == {}
