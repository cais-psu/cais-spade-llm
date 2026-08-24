"""Focused contracts for the three-camera Perception operator foundation."""

from __future__ import annotations

import inspect
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from cais_spade_llm.resources.sensor.physical import realsense_preview_node
from cais_spade_llm.resources.sensor.physical.calibrate_hand_eye import (
    _CaptureNode,
    solve_stationary_calibration,
)
from cais_spade_llm.resources.sensor.physical.realsense_preview_node import depth_preview
from cais_spade_llm.resources.sensor.physical.realsense_roboflow_node import (
    annotate_detection_frame,
)
from cais_spade_llm.ui import perception_manager as manager_module
from cais_spade_llm.ui.pages.perception import _PerceptionPage
from cais_spade_llm.ui.perception_manager import (
    CAMERA_ROLES,
    PerceptionManager,
    parse_realsense_devices,
    parse_usbipd_realsense_devices,
    validate_camera_config,
)


class _FakeBridge:
    _ROS2_ENV = ""

    def __init__(self) -> None:
        self.commands: list[tuple[str, str, int | None]] = []
        self.stopped: list[str] = []
        self.running: set[str] = set()
        self.active_hardware_stack = ""
        self.selected_hardware_stack = ""

    @staticmethod
    def _default_ros_domain_id() -> int:
        return 0

    @staticmethod
    def _ros2_domain_export(_domain_id: int | None) -> str:
        return ""

    def _start_tracked_ros2_command(
        self,
        process_name: str,
        command: str,
        *,
        ros_domain_id: int | None = None,
    ) -> None:
        self.commands.append((process_name, command, ros_domain_id))
        self.running.add(process_name)

    def ros2_proc_status(self, name: str) -> str:
        return "running" if name in self.running else "stopped"

    def ros2_stop(self, name: str) -> None:
        self.stopped.append(name)
        self.running.discard(name)

    def _active_normal_hardware_stack(self) -> str:
        return self.active_hardware_stack

    def _selected_normal_hardware_stack(self) -> str:
        return self.selected_hardware_stack


def test_realsense_preview_uses_two_executor_workers_and_stops_before_destroy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[object] = []
    fake_node = object()

    class FakeRCLError(Exception):
        pass

    class FakePreview:
        node = fake_node

        def __init__(self, **kwargs: object) -> None:
            events.append(("preview", kwargs))

        @staticmethod
        def destroy() -> None:
            events.append("preview.destroy")

    class FakeExecutor:
        def __init__(self, *, num_threads: int) -> None:
            events.append(("executor", num_threads))

        @staticmethod
        def add_node(node: object) -> None:
            events.append(("executor.add_node", node))

        @staticmethod
        def spin() -> None:
            events.append("executor.spin")
            raise KeyboardInterrupt

        @staticmethod
        def shutdown() -> None:
            events.append("executor.shutdown")

    fake_rclpy = SimpleNamespace(
        init=lambda *, args: events.append(("rclpy.init", args)),
        ok=lambda: True,
        shutdown=lambda: events.append("rclpy.shutdown"),
        spin=lambda _node: pytest.fail("rclpy.spin must not run the preview node"),
    )
    monkeypatch.setitem(sys.modules, "rclpy", fake_rclpy)
    monkeypatch.setitem(
        sys.modules,
        "rclpy.executors",
        SimpleNamespace(MultiThreadedExecutor=FakeExecutor),
    )
    monkeypatch.setitem(
        sys.modules,
        "rclpy._rclpy_pybind11",
        SimpleNamespace(RCLError=FakeRCLError),
    )
    args = SimpleNamespace(
        camera_role="ur5e",
        color_topic="/camera/color",
        depth_topic="/camera/depth",
        output_root=tmp_path,
        camera_info_topic="/camera/info",
        world_frame="world",
        parent_frame="tool0",
        camera_optical_frame="camera_color_optical_frame",
        hand_eye_config=None,
        marker_length_m=0.076,
        maximum_rate_hz=5.0,
        expected_rate_hz=6.0,
    )
    monkeypatch.setattr(realsense_preview_node, "RealSensePreviewNode", FakePreview)
    monkeypatch.setattr(
        realsense_preview_node,
        "_build_parser",
        lambda: SimpleNamespace(parse_known_args=lambda: (args, ["--ros-args"])),
    )

    realsense_preview_node.main()

    assert ("executor", 2) in events
    assert ("executor.add_node", fake_node) in events
    assert events.index("executor.shutdown") < events.index("preview.destroy")
    assert events.index("preview.destroy") < events.index("rclpy.shutdown")


@pytest.fixture
def perception_manager(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> PerceptionManager:
    monkeypatch.setattr(manager_module, "CONFIG_PATH", tmp_path / "perception_cameras.yaml")
    monkeypatch.setattr(manager_module, "DIAGNOSTIC_ROOT", tmp_path / "diagnostics")
    monkeypatch.setattr(manager_module, "PREVIEW_ROOT", tmp_path / "previews")
    monkeypatch.setattr(manager_module, "SNAPSHOT_ROOT", tmp_path / "snapshots")
    monkeypatch.setattr(manager_module, "CAMERA_STARTUP_GRACE_SEC", 0.0)
    manager = PerceptionManager(
        _FakeBridge(),
        project_root=Path.cwd(),
        venv_python=Path("/tmp/cais-venv-python"),
    )
    payload = manager.config()
    for role in CAMERA_ROLES:
        payload["cameras"][role]["calibration_path"] = str(
            tmp_path / f"{role}_calibration.yaml"
        )
    manager_module._atomic_yaml_write(manager.config_path, payload)
    monkeypatch.setattr(manager, "_ensure_assigned_camera_available", lambda _serial: None)
    monkeypatch.setattr(
        manager,
        "_stop_stale_role_processes",
        lambda _role, _camera, _process: None,
    )
    return manager


def test_exact_camera_roles_and_duplicate_serial_rejection() -> None:
    assert CAMERA_ROLES == ("ur5e", "xarm6", "stationary")
    payload = {
        "cameras": {
            role: {
                "serial": "same" if role != "stationary" else "",
                "camera_namespace": role,
                "camera_name": role,
                "parent_frame": "world",
                "calibration_path": "/tmp/calibration.yaml",
            }
            for role in CAMERA_ROLES
        }
    }
    with pytest.raises(ValueError, match="same RealSense serial"):
        validate_camera_config(payload)


def _write_assembly_board_v1_snapshot(
    manager: PerceptionManager,
    role: str,
    *,
    frame_captured_at: float | None = None,
    pose: dict[str, float] | None = None,
    **overrides: object,
) -> Path:
    snapshot_path = manager._assembly_board_v1_aruco_path(role)
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "camera_role": role,
        "resource_location": "assembly_board-v1",
        "valid": True,
        "visible": True,
        "stable": True,
        "world_pose_ready": True,
        "frame_id": "world",
        "sample_started_at": time.time() - 0.5,
        "frame_captured_at": frame_captured_at or time.time(),
        "sample_count": 10,
        "required_sample_count": 10,
        "marker_dictionary": "DICT_ARUCO_ORIGINAL",
        "marker_id": 70,
        "marker_length_m": 0.076,
        "pose": pose
        or {
            "frame_id": "world",
            "x": 0.4,
            "y": -0.2,
            "z": 1.0,
            "qx": 0.0,
            "qy": 0.0,
            "qz": 0.0,
            "qw": 1.0,
        },
        "reprojection_error_px": 0.4,
        "translation_spread_m": 0.001,
        "rotation_spread_deg": 0.25,
        "calibration_id": f"{role}-calibration",
        "last_error": "",
    }
    payload.update(overrides)
    snapshot_path.write_text(json.dumps(payload), encoding="utf-8")
    calibration_path = Path(manager.config()["cameras"][role]["calibration_path"])
    calibration_path.write_text(
        yaml.safe_dump(
            {
                "calibration_id": str(payload.get("calibration_id") or ""),
                "validation": {"accepted": True},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return snapshot_path


def _write_stationary_inspection_geometry(
    manager: PerceptionManager,
    root: Path,
    *,
    configured: bool,
) -> Path:
    geometry_path = (
        root
        / "cais_spade_llm/specification/products/geometry/assembly_board-v1.json"
    )
    geometry_path.parent.mkdir(parents=True, exist_ok=True)
    registration = {
        "calibration_id": "board-registration" if configured else None,
        "x": 0.0 if configured else None,
        "y": 0.0 if configured else None,
        "z": 0.0 if configured else None,
        "qx": 0.0 if configured else None,
        "qy": 0.0 if configured else None,
        "qz": 0.0 if configured else None,
        "qw": 1.0 if configured else None,
    }
    geometry_path.write_text(
        json.dumps(
            {
                "real": {
                    "assembly_board": {
                        "assembly_board-v1_aruco_to_assembly_board-v1": registration,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    manager.project_root = root
    return geometry_path


def _write_stationary_inspection_result(
    manager: PerceptionManager,
    inspection: dict[str, object],
    *,
    preview_world_pose_ready: bool = False,
) -> None:
    captured_at = float(inspection.get("captured_at") or time.time() + 0.5)
    preview_dir = manager_module.PREVIEW_ROOT / "stationary"
    preview_dir.mkdir(parents=True, exist_ok=True)
    (preview_dir / "detection_status.json").write_text(
        json.dumps(
            {
                "updated_at": time.time() + 0.5,
                "captured_at": captured_at,
                "visual_detection_ready": True,
                "world_pose_ready": preview_world_pose_ready,
                "pose_error": "",
                "detections": [],
            }
        ),
        encoding="utf-8",
    )
    snapshot_path = manager._snapshot_path("stationary")
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path.write_text(
        json.dumps(
            {
                "updated_at": time.time() + 0.5,
                "frame_captured_at": captured_at,
                "camera_role": "stationary",
                "detections": [],
                "last_error": "",
                "stationary_inspection": {
                    "frame_id": "assembly_board-v1",
                    "captured_at": captured_at,
                    "xy_tolerance_m": 0.01,
                    "seating_tolerance_m": 0.005,
                    "registration": {
                        "name": "assembly_board-v1_aruco_to_assembly_board-v1",
                        "configured": True,
                        "calibration_id": "board-registration",
                    },
                    "aruco": {"marker_id": 70, "marker_length_m": 0.076},
                    "parts": [],
                    **inspection,
                },
            }
        ),
        encoding="utf-8",
    )


def test_post_staging_acceptance_allowed_for_unaccepted_hand_eye_role(
    perception_manager: PerceptionManager,
) -> None:
    snapshot_path = _write_assembly_board_v1_snapshot(perception_manager, "ur5e")
    snapshot_path.unlink()

    status = perception_manager.assembly_board_v1_aruco_status("ur5e")

    assert status["accepted_generation"] == 0
    assert status["active_calibration_ready"] is True
    assert status["configured_marker_length_m"] == pytest.approx(0.076)
    assert status["post_staging_acceptance_allowed"] is True


def test_stationary_id70_status_is_diagnostic_and_never_accepts_a_world_baseline(
    perception_manager: PerceptionManager,
    tmp_path: Path,
) -> None:
    _write_stationary_inspection_geometry(
        perception_manager,
        tmp_path / "configured",
        configured=True,
    )
    _write_assembly_board_v1_snapshot(
        perception_manager,
        "stationary",
        marker_id=999,
        sample_count=99,
    )
    captured_at = time.time() + 0.5
    _write_stationary_inspection_result(
        perception_manager,
        {
            "available": True,
            "success": True,
            "message": "stationary assembly inspection passed for SG and MG.",
            "captured_at": captured_at,
            "aruco": {
                "marker_id": 70,
                "marker_length_m": 0.076,
                "captured_at": captured_at,
                "visible": True,
                "valid": True,
                "stable": True,
                "sample_count": 10,
                "required_sample_count": 10,
                "reprojection_error_px": 0.4,
                "maximum_reprojection_error_px": 1.0,
                "translation_spread_m": 0.001,
                "maximum_translation_spread_m": 0.003,
                "rotation_spread_deg": 0.2,
                "maximum_rotation_spread_deg": 1.0,
            },
        },
    )

    status = perception_manager.assembly_board_v1_aruco_status("stationary")

    assert status["source"] == "stationary_inspection"
    assert status["marker_id"] == 70
    assert status["sample_count"] == 10
    assert status["marker_quality_ready"] is True
    assert status["stationary_diagnostic_only"] is True
    assert status["frame_id"] == "assembly_board-v1"
    assert status["world_pose_ready"] is False
    assert status["inspection_only"] is True
    assert status["calibration_required"] is False
    assert status["inspection_available"] is True
    assert status["calibration_ready"] is False
    assert status["active_calibration_ready"] is False
    assert status["ready_to_accept"] is False
    assert status["accepted_baseline_ready"] is False
    assert status["movement_evidence_valid"] is False
    assert status["translation_delta_m"] is None
    assert status["rotation_delta_deg"] is None
    assert "no world or robot placement authority" in status["accepted_baseline_error"]
    with pytest.raises(ValueError, match="diagnostic only"):
        perception_manager.locate_and_accept_assembly_board_v1("stationary")


def test_stationary_id70_status_uses_current_unconfigured_registration_reason(
    perception_manager: PerceptionManager,
    tmp_path: Path,
) -> None:
    _write_stationary_inspection_geometry(
        perception_manager,
        tmp_path / "unconfigured",
        configured=False,
    )
    _write_stationary_inspection_result(
        perception_manager,
        {
            "available": True,
            "success": True,
            "message": "stale success",
        },
    )

    status = perception_manager.assembly_board_v1_aruco_status("stationary")

    expected = (
        "assembly_board-v1_aruco_to_assembly_board-v1 is not configured; "
        "stationary assembly inspection is unavailable."
    )
    assert status["inspection_available"] is False
    assert status["inspection_success"] is False
    assert status["inspection_message"] == expected
    assert status["error"] == expected


def test_explicitly_persisted_movement_block_remains_through_occlusion_until_accept(
    perception_manager: PerceptionManager,
) -> None:
    snapshot_path = _write_assembly_board_v1_snapshot(perception_manager, "ur5e")
    first = perception_manager.locate_and_accept_assembly_board_v1("ur5e")
    payload = perception_manager.config()
    payload["assembly_board-v1_aruco"]["roles"]["ur5e"]["movement_blocked"] = True
    manager_module._atomic_yaml_write(perception_manager.config_path, payload)
    snapshot_path.unlink()

    status = perception_manager.assembly_board_v1_aruco_status("ur5e")

    assert status["accepted"] is True
    assert status["calibration_identity_matches"] is True
    assert status["movement_blocked"] is True
    assert status["accepted_baseline_ready"] is False
    assert "moved more than 10 mm or 2 deg" in status["accepted_baseline_error"]
    assert status["post_staging_acceptance_allowed"] is True

    _write_assembly_board_v1_snapshot(
        perception_manager,
        "ur5e",
        pose=dict(first["pose"]),
    )
    second = perception_manager.locate_and_accept_assembly_board_v1("ur5e")

    assert second["accepted_generation"] == 2
    assert second["movement_blocked"] is False
    assert second["accepted_baseline_ready"] is True
    saved = perception_manager.config()["assembly_board-v1_aruco"]["roles"]["ur5e"]
    assert saved["movement_blocked"] is False


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("accepted_generation", 0),
        ("accepted_pose", {}),
        ("accepted_at", None),
        ("calibration_id", ""),
    ],
)
def test_post_staging_acceptance_rejects_malformed_existing_acceptance(
    perception_manager: PerceptionManager,
    field: str,
    value: object,
) -> None:
    _write_assembly_board_v1_snapshot(perception_manager, "ur5e")
    perception_manager.locate_and_accept_assembly_board_v1("ur5e")
    payload = perception_manager.config()
    role_config = payload["assembly_board-v1_aruco"]["roles"]["ur5e"]
    role_config["movement_blocked"] = True
    role_config[field] = value
    manager_module._atomic_yaml_write(perception_manager.config_path, payload)

    status = perception_manager.assembly_board_v1_aruco_status("ur5e")

    assert status["post_staging_acceptance_allowed"] is False


def test_post_staging_acceptance_requires_hand_eye_and_76_mm_configuration(
    perception_manager: PerceptionManager,
) -> None:
    snapshot_path = _write_assembly_board_v1_snapshot(perception_manager, "xarm6")
    snapshot_path.unlink()
    payload = perception_manager.config()
    payload["cameras"]["xarm6"]["calibration_mode"] = "stationary"
    manager_module._atomic_yaml_write(perception_manager.config_path, payload)

    assert perception_manager.assembly_board_v1_aruco_status("xarm6")[
        "post_staging_acceptance_allowed"
    ] is False

    payload = perception_manager.config()
    payload["cameras"]["xarm6"]["calibration_mode"] = "hand_eye"
    payload["assembly_board-v1_aruco"]["marker_length_m"] = 0.075
    manager_module._atomic_yaml_write(perception_manager.config_path, payload)

    assert perception_manager.assembly_board_v1_aruco_status("xarm6")[
        "post_staging_acceptance_allowed"
    ] is False


def test_cross_view_movement_evidence_does_not_persist_movement_block(
    perception_manager: PerceptionManager,
) -> None:
    _write_assembly_board_v1_snapshot(perception_manager, "ur5e")

    first = perception_manager.locate_and_accept_assembly_board_v1("ur5e")

    assert first["success"] is True
    assert first["accepted_generation"] == 1
    config = perception_manager.config()["assembly_board-v1_aruco"]
    assert config["marker_length_m"] == pytest.approx(0.076)
    assert config["roles"]["ur5e"]["accepted_pose"] == pytest.approx(first["pose"])
    assert config["roles"]["xarm6"]["accepted_generation"] == 0

    cross_view_angle_rad = math.radians(3.5)
    cross_view_pose = {
        **first["pose"],
        "qz": math.sin(cross_view_angle_rad / 2.0),
        "qw": math.cos(cross_view_angle_rad / 2.0),
    }
    snapshot_path = _write_assembly_board_v1_snapshot(
        perception_manager,
        "ur5e",
        pose=cross_view_pose,
    )
    cross_view = perception_manager.assembly_board_v1_aruco_status("ur5e")
    assert cross_view["translation_delta_m"] == pytest.approx(0.0)
    assert cross_view["rotation_delta_deg"] == pytest.approx(3.5)
    assert cross_view["movement_evidence_valid"] is True
    assert cross_view["excessive_movement"] is True
    assert cross_view["movement_blocked"] is False
    assert cross_view["accepted_baseline_ready"] is True
    assert cross_view["accepted_baseline_error"] == ""
    assert perception_manager.config()["assembly_board-v1_aruco"]["roles"][
        "ur5e"
    ]["movement_blocked"] is False

    cross_view_snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    cross_view_snapshot.update(
        {
            "valid": False,
            "visible": False,
            "stable": False,
            "world_pose_ready": False,
            "pose": None,
            "sample_count": 0,
            "last_error": "assembly_board-v1 ArUco ID 70 is not visible",
        }
    )
    snapshot_path.write_text(json.dumps(cross_view_snapshot), encoding="utf-8")

    occluded = perception_manager.assembly_board_v1_aruco_status("ur5e")
    assert occluded["movement_evidence_valid"] is False
    assert occluded["excessive_movement"] is False
    assert occluded["movement_blocked"] is False
    assert occluded["accepted_baseline_ready"] is True
    assert occluded["accepted_baseline_error"] == ""
    assert occluded["accepted_generation"] == first["accepted_generation"]
    assert occluded["accepted_pose"] == pytest.approx(first["pose"])


def test_stale_movement_evidence_does_not_relatch_a_new_accepted_generation(
    perception_manager: PerceptionManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_assembly_board_v1_snapshot(perception_manager, "ur5e")
    first = perception_manager.locate_and_accept_assembly_board_v1("ur5e")
    moved_pose = {**first["pose"], "x": float(first["pose"]["x"]) + 0.011}
    _write_assembly_board_v1_snapshot(perception_manager, "ur5e", pose=moved_pose)
    original_movement = perception_manager._assembly_board_v1_movement
    generation_advanced = False

    def _advance_generation_before_latch(
        accepted_pose: dict[str, object],
        live_pose: dict[str, object],
    ) -> tuple[float | None, float | None]:
        nonlocal generation_advanced
        movement = original_movement(accepted_pose, live_pose)
        if not generation_advanced:
            generation_advanced = True
            with perception_manager._assembly_board_v1_config_lock:
                payload = perception_manager.config()
                role_config = payload["assembly_board-v1_aruco"]["roles"]["ur5e"]
                role_config.update(
                    {
                        "accepted_generation": 2,
                        "accepted_pose": dict(live_pose),
                        "accepted_at": time.time(),
                        "frame_captured_at": time.time(),
                        "movement_blocked": False,
                    }
                )
                manager_module._atomic_yaml_write(
                    perception_manager.config_path,
                    payload,
                )
        return movement

    monkeypatch.setattr(
        perception_manager,
        "_assembly_board_v1_movement",
        _advance_generation_before_latch,
    )

    perception_manager.assembly_board_v1_aruco_status("ur5e")

    saved = perception_manager.config()["assembly_board-v1_aruco"]["roles"]["ur5e"]
    assert saved["accepted_generation"] == 2
    assert saved["movement_blocked"] is False


@pytest.mark.parametrize("snapshot_role", [None, "", "xarm6"])
def test_assembly_board_v1_snapshot_requires_explicit_matching_camera_role(
    perception_manager: PerceptionManager,
    snapshot_role: str | None,
) -> None:
    _write_assembly_board_v1_snapshot(
        perception_manager,
        "ur5e",
        camera_role=snapshot_role,
    )

    status = perception_manager.assembly_board_v1_aruco_status("ur5e")

    assert status["camera_role"] == str(snapshot_role or "")
    assert status["ready_to_accept"] is False
    with pytest.raises(RuntimeError, match="snapshot camera_role"):
        perception_manager.locate_and_accept_assembly_board_v1("ur5e")


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"marker_dictionary": "DICT_4X4_50"}, "DICT_ARUCO_ORIGINAL"),
        ({"marker_id": 23}, "marker ID must be 70"),
        ({"sample_count": 9}, "9/10 samples"),
        ({"reprojection_error_px": 1.001}, "at most 1 px"),
        ({"translation_spread_m": 0.002001}, "at most 2 mm"),
        ({"rotation_spread_deg": 0.501}, "at most 0.5 deg"),
    ],
)
def test_assembly_board_v1_acceptance_enforces_exact_quality_contract(
    perception_manager: PerceptionManager,
    overrides: dict[str, object],
    expected: str,
) -> None:
    _write_assembly_board_v1_snapshot(perception_manager, "xarm6", **overrides)

    with pytest.raises(RuntimeError, match=expected):
        perception_manager.locate_and_accept_assembly_board_v1("xarm6")


def test_post_staging_acceptance_requires_a_new_stability_window(
    perception_manager: PerceptionManager,
) -> None:
    requested_at = time.time()
    _write_assembly_board_v1_snapshot(
        perception_manager,
        "ur5e",
        sample_started_at=requested_at - 0.001,
    )

    with pytest.raises(RuntimeError, match="before the post-staging request"):
        perception_manager.locate_and_accept_assembly_board_v1(
            "ur5e",
            minimum_sample_started_at=requested_at,
        )
    assert perception_manager.config()["assembly_board-v1_aruco"]["roles"]["ur5e"][
        "accepted_generation"
    ] == 0

    _write_assembly_board_v1_snapshot(
        perception_manager,
        "ur5e",
        sample_started_at=requested_at,
    )
    accepted = perception_manager.locate_and_accept_assembly_board_v1(
        "ur5e",
        minimum_sample_started_at=requested_at,
    )

    assert accepted["success"] is True
    assert accepted["accepted_generation"] == 1


@pytest.mark.parametrize("sample_started_at", [None, float("nan")])
def test_post_staging_acceptance_requires_finite_sample_started_at(
    perception_manager: PerceptionManager,
    sample_started_at: float | None,
) -> None:
    requested_at = time.time()
    _write_assembly_board_v1_snapshot(
        perception_manager,
        "ur5e",
        sample_started_at=sample_started_at,
    )

    with pytest.raises(RuntimeError, match="sample_started_at is missing or invalid"):
        perception_manager.locate_and_accept_assembly_board_v1(
            "ur5e",
            minimum_sample_started_at=requested_at,
        )


def test_acceptance_without_post_staging_minimum_retains_existing_behavior(
    perception_manager: PerceptionManager,
) -> None:
    _write_assembly_board_v1_snapshot(
        perception_manager,
        "ur5e",
        sample_started_at=None,
    )

    accepted = perception_manager.locate_and_accept_assembly_board_v1("ur5e")

    assert accepted["success"] is True
    assert accepted["accepted_generation"] == 1


def test_assembly_board_v1_snapshot_age_limit_is_two_seconds(
    perception_manager: PerceptionManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = time.time()
    monkeypatch.setattr(manager_module.time, "time", lambda: now)
    _write_assembly_board_v1_snapshot(
        perception_manager,
        "ur5e",
        frame_captured_at=now - 2.0,
    )
    assert perception_manager.assembly_board_v1_aruco_status("ur5e")[
        "ready_to_accept"
    ]

    _write_assembly_board_v1_snapshot(
        perception_manager,
        "ur5e",
        frame_captured_at=now - 2.001,
    )
    with pytest.raises(RuntimeError, match="frame is stale"):
        perception_manager.locate_and_accept_assembly_board_v1("ur5e")


def test_assembly_board_v1_calibration_change_blocks_old_acceptance(
    perception_manager: PerceptionManager,
) -> None:
    _write_assembly_board_v1_snapshot(perception_manager, "xarm6")
    perception_manager.locate_and_accept_assembly_board_v1("xarm6")
    _write_assembly_board_v1_snapshot(
        perception_manager,
        "xarm6",
        calibration_id="replacement-calibration",
    )

    status = perception_manager.assembly_board_v1_aruco_status("xarm6")

    assert status["calibration_changed"] is True
    assert status["calibration_identity_matches"] is False
    assert status["movement_blocked"] is True
    assert status["accepted_baseline_ready"] is False
    assert "calibration identity changed" in status["accepted_baseline_error"]
    assert status["placement_blocked"] is True
    assert status["ready_to_accept"] is True
    assert status["post_staging_acceptance_allowed"] is False


def test_missing_live_tag_does_not_claim_that_the_accepted_board_moved(
    perception_manager: PerceptionManager,
) -> None:
    snapshot_path = _write_assembly_board_v1_snapshot(perception_manager, "ur5e")
    perception_manager.locate_and_accept_assembly_board_v1("ur5e")
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    snapshot.update(
        {
            "valid": False,
            "visible": False,
            "stable": False,
            "world_pose_ready": False,
            "pose": None,
            "sample_count": 0,
            "reprojection_error_px": None,
            "translation_spread_m": None,
            "rotation_spread_deg": None,
            "last_error": "assembly_board-v1 ArUco ID 70 is not visible",
        }
    )
    snapshot_path.write_text(json.dumps(snapshot), encoding="utf-8")

    status = perception_manager.assembly_board_v1_aruco_status("ur5e")

    assert status["ready_to_accept"] is False
    assert status["movement_evidence_valid"] is False
    assert status["translation_delta_m"] is None
    assert status["rotation_delta_deg"] is None
    assert status["movement_blocked"] is False
    assert status["accepted_baseline_ready"] is True
    assert status["accepted_baseline_error"] == ""
    assert status["placement_blocked"] is False


def test_missing_live_snapshot_uses_active_calibration_without_claiming_movement(
    perception_manager: PerceptionManager,
) -> None:
    snapshot_path = _write_assembly_board_v1_snapshot(perception_manager, "ur5e")
    perception_manager.locate_and_accept_assembly_board_v1("ur5e")
    snapshot_path.unlink()

    status = perception_manager.assembly_board_v1_aruco_status("ur5e")

    assert status["movement_evidence_valid"] is False
    assert status["movement_blocked"] is False
    assert status["accepted_baseline_ready"] is True
    assert status["accepted_baseline_error"] == ""
    assert status["placement_blocked"] is False
    assert status["placement_blocked_reason"] == ""


def test_active_calibration_change_blocks_baseline_without_live_tag(
    perception_manager: PerceptionManager,
) -> None:
    snapshot_path = _write_assembly_board_v1_snapshot(perception_manager, "ur5e")
    perception_manager.locate_and_accept_assembly_board_v1("ur5e")
    snapshot_path.unlink()
    calibration_path = Path(
        perception_manager.config()["cameras"]["ur5e"]["calibration_path"]
    )
    calibration_path.write_text(
        "calibration_id: replacement-calibration\nvalidation:\n  accepted: true\n",
        encoding="utf-8",
    )

    status = perception_manager.assembly_board_v1_aruco_status("ur5e")

    assert status["movement_evidence_valid"] is False
    assert status["calibration_changed"] is True
    assert status["accepted_baseline_ready"] is False
    assert "calibration identity changed" in status["accepted_baseline_error"]
    assert status["placement_blocked"] is True


def test_perception_page_distinguishes_unknown_movement_from_blocking() -> None:
    class _Label:
        def __init__(self) -> None:
            self.text = ""
            self.style = ""

        def classes(self, *, replace: str) -> None:
            self.style = replace

    page = object.__new__(_PerceptionPage)
    labels = {name: _Label() for name in ("source", "quality", "movement", "pose", "error")}
    page.assembly_board_v1_labels = {"ur5e": labels}

    page._refresh_assembly_board_v1_aruco(
        "ur5e",
        {
            "camera_role": "ur5e",
            "accepted_generation": 1,
            "accepted_baseline_ready": True,
            "accepted_baseline_error": "",
            "movement_evidence_valid": False,
            "error": "assembly_board-v1 ArUco ID 70 is not visible",
        },
    )

    assert "movement from accepted pose=not evaluated" in labels["movement"].text
    assert "accepted baseline=USABLE" in labels["movement"].text
    assert labels["movement"].style == "text-xs text-amber-700"


def test_marker_length_change_invalidates_both_accepted_role_baselines(
    perception_manager: PerceptionManager,
) -> None:
    for role in ("ur5e", "xarm6"):
        _write_assembly_board_v1_snapshot(perception_manager, role)
        perception_manager.locate_and_accept_assembly_board_v1(role)

    legacy = perception_manager.config()
    legacy["assembly_board-v1_aruco"]["marker_length_m"] = 0.075
    manager_module._atomic_yaml_write(perception_manager.config_path, legacy)

    payload = perception_manager.save_assembly_board_v1_marker_length(0.076)

    for role in ("ur5e", "xarm6"):
        accepted = payload["assembly_board-v1_aruco"]["roles"][role]
        assert accepted["accepted_pose"] == {}
        assert accepted["accepted_at"] is None
        assert accepted["calibration_id"] == ""

    with pytest.raises(ValueError, match="exactly 0.076 m"):
        perception_manager.save_assembly_board_v1_marker_length(0.075)


def test_xarm6_calibration_uses_physical_link_eef(
    perception_manager: PerceptionManager,
) -> None:
    camera = perception_manager.config()["cameras"]["xarm6"]
    command = perception_manager._calibration_capture_command("xarm6")

    assert camera["parent_frame"] == "link_eef"
    assert command[command.index("--tool-frame") + 1] == "link_eef"


def test_xarm6_calibration_replay_reports_preview_failure(
    perception_manager: PerceptionManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    status_path = Path("/tmp/cais_xarm6_calibration_replay_status.json")
    previous = status_path.read_bytes() if status_path.exists() else None
    manager_module._atomic_json_write(
        status_path,
        {
            "state": "preview_failed",
            "pose_index": 0,
            "pose_count": 25,
            "error": "/move_action is unavailable; start the correct hardware MoveIt stack",
        },
    )
    monkeypatch.setattr(perception_manager, "_pose_count", lambda _camera: 25)
    try:
        error = perception_manager.start_calibration_replay("xarm6", confirmed=True)
    finally:
        if previous is None:
            status_path.unlink(missing_ok=True)
        else:
            status_path.write_bytes(previous)

    assert error == (
        "xarm6 calibration preview failed: /move_action is unavailable; "
        "start the correct hardware MoveIt stack"
    )


def test_perception_uses_hardware_domain_for_externally_detected_twin(
    perception_manager: PerceptionManager,
) -> None:
    bridge = perception_manager.bridge
    bridge._DIGITAL_TWIN_HARDWARE_PROCESS_NAMES = set()
    bridge._active_digital_twin_target_from_status = lambda: "ur5e only"
    bridge._digital_twin_domain_ids = lambda: {"hardware": 42}

    assert perception_manager._domain_id() == 42


def test_perception_uses_xarm6_domain_for_xarm6_capture(
    perception_manager: PerceptionManager,
) -> None:
    bridge = perception_manager.bridge
    bridge._DIGITAL_TWIN_HARDWARE_PROCESS_NAMES = set()
    bridge._active_digital_twin_target_from_status = lambda: "dual robots"
    bridge._digital_twin_domain_ids = lambda: {
        "hardware": 42,
        "hardware_xarm6": 42,
        "hardware_ur5e": 43,
    }
    bridge._digital_twin_target = lambda _target: {"multiple_hardware_domains": True}
    bridge._digital_twin_hardware_domain_id = (
        lambda _target, robot, domains: domains[f"hardware_{robot}"]
    )

    assert perception_manager._domain_id("xarm6") == 42
    assert perception_manager._domain_id("ur5e") == 43


def test_xarm6_calibration_pose_uses_hardware_stack_snapshot(
    perception_manager: PerceptionManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[tuple[str, str, int]] = []

    def _snapshot(
        robot: str,
        *,
        source: str,
        hardware_domain_id: int,
    ) -> dict[str, object]:
        observed.append((robot, source, hardware_domain_id))
        return {
            "joint_names": ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"],
            "positions": [0.1, -0.2, 0.3, -0.4, 0.5, -0.6],
        }

    perception_manager.bridge._snapshot_robot_waypoint = _snapshot
    monkeypatch.setattr(
        perception_manager,
        "_run_ros_command",
        lambda *_args, **_kwargs: pytest.fail(
            "xArm6 capture must not depend on one hard-coded joint-state topic"
        ),
    )

    joint_state = perception_manager._read_joint_state("xarm6")

    assert joint_state["names"] == [
        "joint1",
        "joint2",
        "joint3",
        "joint4",
        "joint5",
        "joint6",
    ]
    assert joint_state["positions"] == [0.1, -0.2, 0.3, -0.4, 0.5, -0.6]
    assert observed == [("xarm6", "hardware", 0)]


def test_xarm6_calibration_pose_reports_hardware_snapshot_error(
    perception_manager: PerceptionManager,
) -> None:
    perception_manager.bridge._snapshot_robot_waypoint = lambda *_args, **_kwargs: {
        "error": "hardware /joint_states has no xarm6 arm joints yet"
    }

    with pytest.raises(RuntimeError, match="xarm6 hardware joint feedback is unavailable"):
        perception_manager._read_joint_state("xarm6")


def test_ur5e_calibration_pose_uses_read_only_rtde_without_joint_states(
    perception_manager: PerceptionManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    class _FakeRTDEReceive:
        def __init__(self, robot_ip: str) -> None:
            events.append(f"connect:{robot_ip}")

        @staticmethod
        def getActualQ() -> list[float]:
            events.append("getActualQ")
            return [0.1, -1.0, -2.0, -1.2, 1.5, 0.0]

        @staticmethod
        def disconnect() -> None:
            events.append("disconnect")

    perception_manager.bridge.get_hardware_ips = lambda: {"ur5e": "192.168.1.172"}
    monkeypatch.setitem(
        sys.modules,
        "rtde_receive",
        SimpleNamespace(RTDEReceiveInterface=_FakeRTDEReceive),
    )
    monkeypatch.setattr(
        perception_manager,
        "_run_ros_command",
        lambda *_args, **_kwargs: pytest.fail("UR5e capture must not require /joint_states"),
    )

    joint_state = perception_manager._read_joint_state("ur5e")

    assert joint_state["names"] == [
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    ]
    assert joint_state["positions"] == [0.1, -1.0, -2.0, -1.2, 1.5, 0.0]
    assert events == ["connect:192.168.1.172", "getActualQ", "disconnect"]


def test_ur5e_calibration_starts_only_read_only_state_and_tf_monitor(
    perception_manager: PerceptionManager,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    perception_manager.bridge.get_hardware_ips = lambda: {"ur5e": "192.168.1.172"}
    monitor_status = tmp_path / "ur5e_calibration_monitor.json"
    monkeypatch.setattr(
        manager_module,
        "UR5E_CALIBRATION_MONITOR_STATUS",
        monitor_status,
    )
    original_start = perception_manager.bridge._start_tracked_ros2_command

    def _start(
        process_name: str,
        command: str,
        *,
        ros_domain_id: int | None = None,
    ) -> None:
        original_start(process_name, command, ros_domain_id=ros_domain_id)
        if process_name == "ur5e_calibration_rtde_monitor":
            monitor_status.write_text(
                json.dumps(
                    {
                        "rtde_receive_connected": True,
                        "rtde_control_connected": False,
                        "joint_states_fresh": True,
                        "message": "UR5e read-only calibration monitoring ready in Local Control",
                    }
                ),
                encoding="utf-8",
            )

    monkeypatch.setattr(
        perception_manager.bridge,
        "_start_tracked_ros2_command",
        _start,
    )
    monkeypatch.setattr(
        perception_manager,
        "_run_ros_command",
        lambda *_args, **_kwargs: pytest.fail(
            "fresh monitor status must not depend on ros2 topic echo"
        ),
    )
    monkeypatch.setattr(manager_module.time, "sleep", lambda _seconds: None)

    perception_manager._ensure_ur5e_calibration_monitor()

    commands = {
        name: (command, domain)
        for name, command, domain in perception_manager.bridge.commands
    }
    rtde_command, rtde_domain = commands["ur5e_calibration_rtde_monitor"]
    state_command, state_domain = commands["ur5e_calibration_state_publisher"]
    assert "--monitor-only" in rtde_command
    assert "rtde_control" not in rtde_command
    assert "launch_move_group:=false launch_rviz:=false" in state_command
    assert "RG2" not in state_command
    assert rtde_domain == 0
    assert state_domain == 0


@pytest.mark.parametrize("hardware_stack", ["xarm6", "ur5e", "dual robots"])
def test_normal_ur5e_hardware_state_publisher_stops_calibration_state_publisher(
    perception_manager: PerceptionManager,
    hardware_stack: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge = perception_manager.bridge
    bridge.active_hardware_stack = hardware_stack
    bridge.running.update(
        {
            "hardware_robot_state_publisher",
            "ur5e_calibration_state_publisher",
        }
    )
    monkeypatch.setattr(
        perception_manager,
        "_ur5e_joint_state_publisher_running",
        lambda: True,
    )
    monkeypatch.setattr(
        perception_manager,
        "_run_ros_command",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout="",
            stderr="",
        ),
    )

    perception_manager._ensure_ur5e_calibration_monitor()

    assert "ur5e_calibration_state_publisher" in bridge.stopped
    assert "ur5e_calibration_state_publisher" not in {
        name for name, _command, _domain in bridge.commands
    }


@pytest.mark.parametrize("hardware_stack", ["xarm6", "ur5e", "dual robots"])
@pytest.mark.parametrize("calibration_publisher_running", [False, True])
def test_starting_ur5e_hardware_stack_reserves_full_tf_authority(
    perception_manager: PerceptionManager,
    hardware_stack: str,
    calibration_publisher_running: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge = perception_manager.bridge
    bridge.selected_hardware_stack = hardware_stack
    if calibration_publisher_running:
        bridge.running.add("ur5e_calibration_state_publisher")
    monkeypatch.setattr(
        perception_manager,
        "_ur5e_joint_state_publisher_running",
        lambda: True,
    )
    monkeypatch.setattr(
        perception_manager,
        "_run_ros_command",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout="",
            stderr="",
        ),
    )

    perception_manager._ensure_ur5e_calibration_monitor()

    assert "ur5e_calibration_state_publisher" not in bridge.running
    assert "ur5e_calibration_state_publisher" not in {
        name for name, _command, _domain in bridge.commands
    }
    assert (
        "ur5e_calibration_state_publisher" in bridge.stopped
    ) is calibration_publisher_running


def test_xarm6_hardware_state_publisher_replaces_ur5e_calibration_tf(
    perception_manager: PerceptionManager,
) -> None:
    bridge = perception_manager.bridge
    bridge.active_hardware_stack = "xarm6"
    bridge.running.add("hardware_robot_state_publisher")

    assert perception_manager._ur5e_full_state_publisher_running() is True


def test_failed_main_rtde_process_does_not_suppress_read_only_monitor(
    perception_manager: PerceptionManager,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    main_status = tmp_path / "ur5e_rtde_trajectory_status.json"
    main_status.write_text(
        json.dumps(
            {
                "updated_at": time.time(),
                "state": "failed",
                "rtde_receive_connected": False,
                "joint_states_fresh": False,
                "rtde_reset_required": True,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(manager_module, "UR5E_RTDE_TRAJECTORY_STATUS", main_status)
    perception_manager.bridge.running.add("hardware_ur5e_rtde_trajectory_server")

    assert perception_manager._ur5e_joint_state_publisher_running() is False


def test_calibration_rejects_near_duplicate_pose_and_accepts_new_pose() -> None:
    names = [
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    ]
    existing = {"names": names, "positions": [0.0] * 6}
    near_duplicate = {"names": names, "positions": [0.01] * 6}
    new_pose = {"names": names, "positions": [0.0, 0.0, 0.1, 0.0, 0.0, 0.0]}

    with pytest.raises(RuntimeError, match="not a new calibration pose"):
        PerceptionManager._require_new_reviewed_pose("ur5e", near_duplicate, [existing])

    PerceptionManager._require_new_reviewed_pose("ur5e", new_pose, [existing])


def test_missing_charuco_board_is_an_operator_warning_without_traceback() -> None:
    result = SimpleNamespace(
        returncode=1,
        stdout="",
        stderr=(
            "Traceback (most recent call last):\n"
            "RuntimeError: fewer than four ChArUco markers are visible"
        ),
    )

    message = PerceptionManager._calibration_capture_error(result)

    assert "ChArUco board is not visible enough" in message
    assert "Place the complete board in the color view" in message
    assert "Traceback" not in message


def test_stationary_calibration_frame_error_names_stationary_capture() -> None:
    result = SimpleNamespace(
        returncode=1,
        stdout="",
        stderr="RuntimeError: color image or CameraInfo is unavailable",
    )

    message = PerceptionManager._calibration_capture_error(result, "stationary")

    assert message.startswith("stationary color image or CameraInfo is unavailable")
    assert "UR5e" not in message


def test_ur5e_calibration_missing_tf_reports_local_control_workflow() -> None:
    class _LookupException(Exception):
        pass

    class _ConnectivityException(Exception):
        pass

    class _ExtrapolationException(Exception):
        pass

    def _missing_transform(*_args: object, **_kwargs: object) -> object:
        raise _LookupException('"world" does not exist')

    capture = object.__new__(_CaptureNode)
    capture.camera_role = "ur5e"
    capture.tf2_ros = SimpleNamespace(
        LookupException=_LookupException,
        ConnectivityException=_ConnectivityException,
        ExtrapolationException=_ExtrapolationException,
    )
    capture.rclpy = SimpleNamespace(
        time=SimpleNamespace(Time=lambda: object()),
        duration=SimpleNamespace(Duration=lambda **_kwargs: object()),
    )
    capture.tf_buffer = SimpleNamespace(lookup_transform=_missing_transform)

    with pytest.raises(RuntimeError, match="read-only UR5e calibration monitor"):
        capture.transform("world", "tool0")


def test_realsense_and_usbipd_discovery_parsing() -> None:
    devices = parse_realsense_devices(
        """
        Device info:
            Name                          : Intel RealSense D435
            Serial Number                 : 103422070738
            Firmware Version              : 5.12.10
            Physical Port                 : /sys/devices/video0
            Usb Type Descriptor           : 3.2
        """
    )
    assert devices == [
        {
            "model": "Intel RealSense D435",
            "serial": "103422070738",
            "firmware": "5.12.10",
            "physical_port": "/sys/devices/video0",
            "usb_type": "3.2",
        }
    ]
    rows = parse_usbipd_realsense_devices(
        "7-2    8086:0b07  Intel(R) RealSense(TM) Depth Camera 435  Shared\n"
    )
    assert rows[0]["busid"] == "7-2"
    assert rows[0]["vid_pid"] == "8086:0b07"
    assert rows[0]["description"] == "Intel(R) RealSense(TM) Depth Camera 435"
    assert rows[0]["state"] == "Shared"


def test_failed_realsense_discovery_preserves_last_successful_inventory(
    perception_manager: PerceptionManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    perception_manager._devices_cache = [
        {"serial": "103422070738", "model": "Intel RealSense D435"}
    ]
    perception_manager._devices_cache_at = 0.0
    monkeypatch.setattr(manager_module.shutil, "which", lambda _name: "/usr/bin/rs-enum")
    monkeypatch.setattr(
        perception_manager,
        "_run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=1,
            stdout="",
            stderr="could not initialize udev monitor\n",
        ),
    )

    devices = perception_manager.discover_devices(force=True)

    assert devices == [{"serial": "103422070738", "model": "Intel RealSense D435"}]
    assert perception_manager._device_discovery_error == "could not initialize udev monitor"


def test_failed_realsense_discovery_is_rate_limited(
    perception_manager: PerceptionManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    now = [100.0]
    perception_manager._devices_cache_at = 0.0
    monkeypatch.setattr(manager_module.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(manager_module.shutil, "which", lambda _name: "/usr/bin/rs-enum")

    def _run(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return SimpleNamespace(returncode=1, stdout="", stderr="device disconnected\n")

    monkeypatch.setattr(perception_manager, "_run", _run)

    assert perception_manager.discover_devices() == []
    now[0] = 101.0
    assert perception_manager.discover_devices() == []
    assert calls == 1

    now[0] = 111.0
    assert perception_manager.discover_devices() == []
    assert calls == 2


def test_camera_assignment_options_retain_saved_serials_when_discovery_is_empty() -> None:
    class _Select:
        def __init__(self) -> None:
            self.options: dict[str, str] = {"": "Unassigned"}
            self.value = ""
            self.update_count = 0

        def update(self) -> None:
            self.update_count += 1

    page = object.__new__(_PerceptionPage)
    page.assigned_serials = {
        "ur5e": "103422070738",
        "xarm6": "048522073304",
        "stationary": "",
    }
    page.serial_selects = {role: _Select() for role in CAMERA_ROLES}

    page._set_serial_options([])

    for select in page.serial_selects.values():
        assert "103422070738" in select.options
        assert "048522073304" in select.options
        assert select.update_count == 1
    assert page.serial_selects["ur5e"].value == "103422070738"
    assert page.serial_selects["xarm6"].value == "048522073304"
    assert page.serial_selects["stationary"].value == ""


def test_bound_realsense_rows_attach_until_assigned_serial_is_visible(
    perception_manager: PerceptionManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device_checks = 0
    attached: list[str] = []

    def discover_devices(*, force: bool = False) -> list[dict[str, str]]:
        del force
        nonlocal device_checks
        device_checks += 1
        return [] if device_checks == 1 else [{"serial": "100"}]

    monkeypatch.setattr(perception_manager, "discover_devices", discover_devices)
    monkeypatch.setattr(
        perception_manager,
        "discover_wsl_attachments",
        lambda force=False: [
            {"busid": "7-1", "state": "Shared"},
            {"busid": "7-2", "state": "Attached"},
            {"busid": "7-3", "state": "Not shared"},
        ],
    )
    monkeypatch.setattr(
        perception_manager,
        "attach_wsl_camera",
        lambda busid: attached.append(busid),
    )
    error = PerceptionManager._ensure_assigned_camera_available(
        perception_manager,
        "100",
    )
    assert error is None
    assert attached == ["7-1"]


def test_not_shared_realsense_requires_one_time_bind(
    perception_manager: PerceptionManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(manager_module, "USB_ATTACH_WAIT_SEC", 0.0)
    monkeypatch.setattr(
        perception_manager,
        "discover_devices",
        lambda force=False: [],
    )
    monkeypatch.setattr(
        perception_manager,
        "discover_wsl_attachments",
        lambda force=False: [{"busid": "7-3", "state": "Not shared"}],
    )
    error = PerceptionManager._ensure_assigned_camera_available(
        perception_manager,
        "100",
    )
    assert error is not None
    assert "one-time command" in error
    assert "usbipd bind --busid 7-3" in error


def test_attach_not_shared_camera_returns_exact_administrator_command(
    perception_manager: PerceptionManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        perception_manager,
        "discover_wsl_attachments",
        lambda force=False: [
            {
                "busid": "6-1",
                "description": "Intel(R) RealSense(TM) Depth Camera 435",
                "state": "Not shared",
            }
        ],
    )
    monkeypatch.setattr(
        perception_manager,
        "_run",
        lambda *_args, **_kwargs: pytest.fail(
            "Not shared must not attempt attachment or request administrator credentials"
        ),
    )

    error = perception_manager.attach_wsl_camera("6-1")

    assert error == (
        "RealSense BUSID 6-1 is Not shared. In Administrator Windows PowerShell run: "
        "usbipd bind --busid 6-1. Then refresh WSL USB and attach it."
    )


def test_camera_status_reports_windows_wsl_and_assignment_counts(
    perception_manager: PerceptionManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    perception_manager.save_assignments(
        {"ur5e": "103422070738", "xarm6": "", "stationary": ""}
    )
    monkeypatch.setattr(
        perception_manager,
        "discover_devices",
        lambda force=False: [
            {
                "serial": "103422070738",
                "model": "Intel RealSense D435",
                "usb_type": "2.1",
            }
        ],
    )
    monkeypatch.setattr(
        perception_manager,
        "discover_wsl_attachments",
        lambda force=False: [
            {"busid": "6-1", "state": "Not shared"},
            {"busid": "6-2", "state": "Attached"},
        ],
    )
    monkeypatch.setattr(
        perception_manager,
        "preflight",
        lambda: {"ready": True, "checks": {}, "setup_message": "ready"},
    )
    stationary_preview_dir = manager_module.PREVIEW_ROOT / "stationary"
    stationary_preview_dir.mkdir(parents=True, exist_ok=True)
    (stationary_preview_dir / "detection_status.json").write_text(
        json.dumps(
            {
                "captured_at": time.time(),
                "world_pose_ready": True,
            }
        ),
        encoding="utf-8",
    )

    status = perception_manager.status()

    assert status["camera_inventory"] == {
        "windows_d435_devices": 2,
        "wsl_attached_devices": 1,
        "wsl_discovered_devices": 1,
        "assigned_roles": 1,
        "assigned_role_names": ["ur5e"],
        "total_roles": 3,
    }
    assert status["preflight"]["camera_inventory"] == status["camera_inventory"]
    assert status["cameras"]["ur5e"]["serial"] == "103422070738"
    assert status["cameras"]["xarm6"]["serial"] == ""
    stationary = status["cameras"]["stationary"]
    assert stationary["serial"] == ""
    assert stationary["inspection_only"] is True
    assert stationary["calibration_required"] is False
    assert stationary["calibration_ready"] is True
    assert stationary["calibration"]["required"] is False
    assert stationary["world_pose_ready"] is False
    assert stationary["tf_ready"] is False
    assert stationary["detection_preview"]["world_pose_ready"] is False
    assert stationary["detection_preview"]["tf_ready"] is False
    assert stationary["perception"]["world_pose_ready"] is False
    assert stationary["perception"]["tf_ready"] is False


def test_camera_log_reports_specific_driver_error(tmp_path: Path) -> None:
    log_path = tmp_path / "realsense_camera.log"
    log_path.write_text(
        "\x1b[31m[ERROR] parameter serial_no has Wrong parameter type\x1b[0m\n"
        "[ERROR] process has died\n",
        encoding="utf-8",
    )
    assert PerceptionManager._last_process_error(log_path) == (
        "[ERROR] parameter serial_no has Wrong parameter type"
    )

    log_path.write_text(
        "RealSense warning: The device has been disconnected\n",
        encoding="utf-8",
    )
    assert "device has been disconnected" in PerceptionManager._last_process_error(
        log_path
    )


def test_camera_topics_and_launch_identities(perception_manager: PerceptionManager) -> None:
    perception_manager.save_assignments(
        {"ur5e": "100", "xarm6": "200", "stationary": "300"}
    )
    for role in CAMERA_ROLES:
        assert perception_manager.start_camera(role) is None
    commands = {name: command for name, command, _domain in perception_manager.bridge.commands}
    assert "camera_namespace:=camera camera_name:=camera" in commands["realsense_camera"]
    assert (
        "camera_namespace:=xarm6_camera camera_name:=xarm6_camera"
        in commands["realsense_camera_xarm6"]
    )
    assert (
        "camera_namespace:=stationary_camera camera_name:=stationary_camera"
        in commands["realsense_camera_stationary"]
    )
    assert "/camera/camera/color/image_raw" in commands["realsense_preview_ur5e"]
    assert "rgb_camera.color_profile:=640x480x6" in commands["realsense_camera"]
    assert "depth_module.depth_profile:=640x480x6" in commands["realsense_camera"]
    assert (
        "/xarm6_camera/xarm6_camera/aligned_depth_to_color/image_raw"
        in commands["realsense_preview_xarm6"]
    )


def test_camera_start_cleans_matching_orphans_before_usb_attachment(
    perception_manager: PerceptionManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    perception_manager.save_assignments({"ur5e": "100"})
    actions: list[str] = []
    monkeypatch.setattr(
        perception_manager,
        "_stop_stale_role_processes",
        lambda _role, _camera, process: actions.append(f"cleanup:{process}"),
    )
    monkeypatch.setattr(
        perception_manager,
        "_ensure_assigned_camera_available",
        lambda _serial: actions.append("attach"),
    )

    assert perception_manager.start_camera("ur5e") is None
    assert actions[:3] == ["cleanup:camera", "cleanup:preview", "attach"]

    actions.clear()
    assert perception_manager.start_camera("ur5e") is None
    assert actions == ["attach"]


def test_stale_process_fragments_are_isolated_by_camera_role(
    perception_manager: PerceptionManager,
) -> None:
    camera = perception_manager.config()["cameras"]["xarm6"]
    assert PerceptionManager._stale_process_fragments("xarm6", camera, "camera") == (
        "realsense2_camera_node",
        "-r __node:=xarm6_camera",
        "-r __ns:=/xarm6_camera",
    )
    assert "--camera-role xarm6" in PerceptionManager._stale_process_fragments(
        "xarm6",
        camera,
        "preview",
    )
    assert "detect_all_service:=/perception/xarm6/detect_all" in (
        PerceptionManager._stale_process_fragments("xarm6", camera, "perception")
    )


def test_stale_process_scan_finds_child_when_group_leader_is_gone(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = tmp_path / "123"
    process.mkdir()
    process.joinpath("cmdline").write_bytes(
        b"/opt/ros/humble/lib/realsense2_camera/realsense2_camera_node\0"
        b"--ros-args\0-r\0__node:=camera\0-r\0__ns:=/camera\0"
    )
    monkeypatch.setattr(manager_module.os, "getpgrp", lambda: 999)
    monkeypatch.setattr(manager_module.os, "getpgid", lambda _pid: 456)

    groups = PerceptionManager._matching_stale_process_groups(
        ("realsense2_camera_node", "-r __node:=camera", "-r __ns:=/camera"),
        proc_root=tmp_path,
    )

    assert groups == {456}


def test_camera_recovery_exhausts_and_explicit_disconnect_cancels_retry(
    perception_manager: PerceptionManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    perception_manager.save_assignments({"ur5e": "100"})
    assert perception_manager.start_camera("ur5e") is None
    preview_dir = manager_module.PREVIEW_ROOT / "ur5e"
    preview_dir.mkdir(parents=True, exist_ok=True)
    (preview_dir / "status.json").write_text(
        json.dumps({"frame_captured_at": time.time() - 30.0}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        perception_manager,
        "_start_camera_stack",
        lambda _role: "assigned RealSense is still unavailable",
    )
    now = time.time() + 30.0
    perception_manager.reconcile_connections(now=now)
    for attempt in range(3):
        perception_manager._recovery["ur5e"]["next_retry_at"] = now
        perception_manager.reconcile_connections(now=now)
        assert perception_manager._recovery["ur5e"]["attempt_count"] == attempt + 1
    assert perception_manager._recovery["ur5e"]["state"] == "degraded"

    command_count = len(perception_manager.bridge.commands)
    perception_manager.stop_camera("ur5e")
    perception_manager.reconcile_connections(now=now + 100.0)
    assert len(perception_manager.bridge.commands) == command_count
    assert perception_manager._recovery["ur5e"]["state"] == "disconnected"


def test_camera_waits_for_first_preview_frame_during_usb2_startup(
    perception_manager: PerceptionManager,
) -> None:
    perception_manager.save_assignments({"ur5e": "100"})
    assert perception_manager.start_camera("ur5e") is None
    connected_at = float(perception_manager._recovery["ur5e"]["connected_at"])
    stopped_before = list(perception_manager.bridge.stopped)

    perception_manager.reconcile_connections(now=connected_at + 10.0)
    assert perception_manager._recovery["ur5e"]["state"] == "connecting"
    assert perception_manager.bridge.stopped == stopped_before

    perception_manager.reconcile_connections(
        now=connected_at + manager_module.CAMERA_FIRST_FRAME_GRACE_SEC + 1.0
    )
    assert perception_manager._recovery["ur5e"]["state"] == "waiting"
    assert "realsense_camera" in perception_manager.bridge.stopped


def test_camera_recovery_restores_operator_started_detection(
    perception_manager: PerceptionManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    perception_manager.save_assignments({"ur5e": "100"})
    assert perception_manager.start_camera("ur5e") is None
    perception_manager._desired_perception.add("ur5e")
    preview_dir = manager_module.PREVIEW_ROOT / "ur5e"
    preview_dir.mkdir(parents=True, exist_ok=True)
    (preview_dir / "status.json").write_text(
        json.dumps({"frame_captured_at": time.time() - 30.0}),
        encoding="utf-8",
    )

    restarted: list[str] = []
    perception_restarted: list[str] = []
    monkeypatch.setattr(
        perception_manager,
        "_start_camera_stack",
        lambda role: restarted.append(role),
    )
    monkeypatch.setattr(
        perception_manager,
        "_start_perception_stack",
        lambda role: perception_restarted.append(role),
    )

    now = time.time() + 30.0
    perception_manager.reconcile_connections(now=now)
    perception_manager._recovery["ur5e"]["next_retry_at"] = now
    perception_manager.reconcile_connections(now=now)

    assert restarted == ["ur5e"]
    assert perception_restarted == ["ur5e"]
    assert perception_manager._recovery["ur5e"]["state"] == "connecting"


def test_delayed_first_frame_still_starts_requested_perception(
    perception_manager: PerceptionManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    perception_manager.save_assignments({"ur5e": "100"})
    monkeypatch.setattr(
        perception_manager,
        "_wait_for_camera_frame",
        lambda _role: "ur5e camera did not produce a synchronized frame within 20 seconds",
    )

    error = perception_manager.start_detection("ur5e")

    assert error is not None
    assert "ur5e" in perception_manager._desired_perception
    preview_dir = manager_module.PREVIEW_ROOT / "ur5e"
    preview_dir.mkdir(parents=True, exist_ok=True)
    now = time.time()
    (preview_dir / "status.json").write_text(
        json.dumps({"frame_captured_at": now}),
        encoding="utf-8",
    )
    started: list[str] = []
    monkeypatch.setattr(
        perception_manager,
        "_start_perception_stack",
        lambda role: started.append(role),
    )

    perception_manager.reconcile_connections(now=now)

    assert started == ["ur5e"]
    assert perception_manager._recovery["ur5e"]["state"] == "connected"


def test_reset_camera_cleans_orphans_before_restarting_detection(
    perception_manager: PerceptionManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    perception_manager.save_assignments({"ur5e": "100"})
    actions: list[str] = []
    monkeypatch.setattr(
        perception_manager,
        "stop_detection",
        lambda role: actions.append(f"stop:{role}"),
    )
    monkeypatch.setattr(
        perception_manager,
        "_stop_stale_role_processes",
        lambda _role, _camera, process: actions.append(f"cleanup:{process}"),
    )
    monkeypatch.setattr(manager_module.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(
        perception_manager,
        "start_detection",
        lambda role: actions.append(f"start:{role}"),
    )

    assert perception_manager.reset_camera("ur5e") is None
    assert actions == [
        "stop:ur5e",
        "cleanup:perception",
        "cleanup:preview",
        "cleanup:camera",
        "start:ur5e",
    ]


def test_role_specific_services_keep_ur5e_canonical(
    perception_manager: PerceptionManager,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ROBOFLOW_API_KEY", "test-only")
    monkeypatch.setattr(
        perception_manager,
        "_ensure_ur5e_calibration_monitor",
        lambda **_kwargs: None,
    )
    payload = perception_manager.config()
    for role in ("ur5e", "xarm6"):
        calibration = tmp_path / f"{role}.yaml"
        calibration.write_text("validation:\n  accepted: true\n", encoding="utf-8")
        payload["cameras"][role]["calibration_path"] = str(calibration)
    stationary_calibration = tmp_path / "stationary-does-not-exist.yaml"
    payload["cameras"]["stationary"]["calibration_path"] = str(
        stationary_calibration
    )
    manager_module._atomic_yaml_write(perception_manager.config_path, payload)

    for role in CAMERA_ROLES:
        assert perception_manager.start_perception(role) is None
    commands = {name: command for name, command, _domain in perception_manager.bridge.commands}
    assert "detect_all_service:=/perception/ur5e/detect_all" in commands["physical_perception"]
    assert "publish_canonical_services:=true" in commands["physical_perception"]
    assert "background_rate_hz:=0.2" in commands["physical_perception"]
    assert "preview_root:=" in commands["physical_perception"]
    assert (
        "detect_all_service:=/perception/xarm6/detect_all"
        in commands["physical_perception_xarm6"]
    )
    assert "publish_canonical_services:=false" in commands["physical_perception_xarm6"]
    assert "background_rate_hz:=0.0" in commands["physical_perception_stationary"]
    assert f"hand_eye_config:={stationary_calibration}" not in commands[
        "physical_perception_stationary"
    ]
    assert "table_plane_config:=" not in commands["physical_perception_stationary"]
    assert "hand_eye_config:=" in commands["physical_perception"]
    assert "table_plane_config:=" in commands["physical_perception"]
    assert "hand_eye_config:=" in commands["physical_perception_xarm6"]
    assert "table_plane_config:=" in commands["physical_perception_xarm6"]
    assert (
        "assembly_board_v1_geometry_path:="
        in commands["physical_perception_stationary"]
    )
    assert (
        "assembly_board_v1_marker_length_m:=0.076"
        in commands["physical_perception_stationary"]
    )
    assert "assembly_board_v1_geometry_path:=" not in commands["physical_perception"]
    assert "publish_canonical_services:=false" in commands[
        "physical_perception_stationary"
    ]


def test_stationary_start_collects_id70_window_without_immediate_test_detection(
    perception_manager: PerceptionManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(perception_manager, "start_camera", lambda _role: None)
    monkeypatch.setattr(
        perception_manager,
        "_wait_for_camera_frame",
        lambda _role: None,
    )
    monkeypatch.setattr(perception_manager, "start_perception", lambda _role: None)
    monkeypatch.setattr(
        perception_manager,
        "test_detection",
        lambda _role: pytest.fail(
            "stationary startup must not evaluate the ID 70 window before 10 frames"
        ),
    )

    assert perception_manager.start_detection("stationary") is None


def test_start_and_stop_detection_own_the_complete_camera_stack(
    perception_manager: PerceptionManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    perception_manager.save_assignments({"ur5e": "100"})
    preview_dir = manager_module.PREVIEW_ROOT / "ur5e"
    preview_dir.mkdir(parents=True, exist_ok=True)
    (preview_dir / "detection.jpg").write_bytes(b"old detection")
    (preview_dir / "detection_status.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(perception_manager, "_wait_for_camera_frame", lambda _role: None)
    monkeypatch.setattr(perception_manager, "start_perception", lambda _role: None)
    monkeypatch.setattr(
        perception_manager,
        "test_detection",
        lambda _role: {
            "success": True,
            "visual_detection_ready": True,
            "world_pose_ready": False,
            "message": "2D detection completed; world pose unavailable",
        },
    )

    assert perception_manager.start_detection("ur5e") is None
    assert "ur5e" in perception_manager._desired_connected
    assert not (preview_dir / "detection.jpg").exists()

    for filename in ("color.jpg", "depth.jpg", "detection.jpg"):
        (preview_dir / filename).write_bytes(b"frame")
    perception_manager.stop_detection("ur5e")

    assert "ur5e" not in perception_manager._desired_connected
    assert "realsense_camera" in perception_manager.bridge.stopped
    assert "realsense_preview_ur5e" in perception_manager.bridge.stopped
    assert "physical_perception" in perception_manager.bridge.stopped
    assert "realsense_calibration_ur5e" in perception_manager.bridge.stopped
    assert not any(preview_dir.glob("*.jpg"))


def test_ui_test_detection_accepts_visual_result_when_world_pose_is_blocked(
    perception_manager: PerceptionManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preview_dir = manager_module.PREVIEW_ROOT / "ur5e"
    preview_dir.mkdir(parents=True, exist_ok=True)
    (preview_dir / "detection_status.json").write_text(
        json.dumps(
            {
                "updated_at": time.time() + 1.0,
                "captured_at": time.time(),
                "visual_detection_ready": True,
                "world_pose_ready": False,
                "pose_error": "TF unavailable for world <- tool0",
                "detections": [
                    {
                        "part_name": "MG",
                        "model_name": "gear_medium",
                        "label": "medium_gear",
                        "confidence": 0.923,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    snapshot_path = perception_manager._snapshot_path("ur5e")
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path.write_text(
        json.dumps(
            {
                "last_error": "TF unavailable for world <- tool0",
                "detections": [],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        manager_module.subprocess,
        "run",
        lambda *_args, **_kwargs: type(
            "Result",
            (),
            {"returncode": 0, "stdout": "", "stderr": ""},
        )(),
    )

    result = perception_manager.test_detection("ur5e")

    assert result["success"] is True
    assert result["visual_detection_ready"] is True
    assert result["world_pose_ready"] is False
    assert result["detections"] == []
    assert result["visual_detections"][0]["label"] == "medium_gear"
    assert "world pose unavailable" in result["message"]


def test_stationary_test_detection_fails_closed_for_unconfigured_registration(
    perception_manager: PerceptionManager,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_stationary_inspection_geometry(
        perception_manager,
        tmp_path / "unconfigured",
        configured=False,
    )
    _write_stationary_inspection_result(
        perception_manager,
        {
            "available": True,
            "success": True,
            "message": "stale success must not bypass current configuration",
        },
    )
    monkeypatch.setattr(
        manager_module.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout="",
            stderr="",
        ),
    )

    result = perception_manager.test_detection("stationary")

    expected = (
        "assembly_board-v1_aruco_to_assembly_board-v1 is not configured; "
        "stationary assembly inspection is unavailable."
    )
    assert result["success"] is False
    assert result["stationary_inspection_ready"] is False
    assert result["message"] == expected
    assert result["stationary_inspection"]["message"] == expected
    assert result["diagnostic_only"] is True
    assert result["canonical_authority"] is False
    assert result["robot_motion_requested"] is False


def test_stationary_test_detection_exposes_exact_frame_sg_mg_inspection(
    perception_manager: PerceptionManager,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_stationary_inspection_geometry(
        perception_manager,
        tmp_path / "configured",
        configured=True,
    )
    captured_at = time.time() + 0.5
    parts = [
        {
            "part_name": "SG",
            "detected": True,
            "success": True,
            "expected": {
                "frame_id": "assembly_board-v1",
                "x": -0.1,
                "y": 0.08,
                "z": 0.03,
                "point": "top_surface",
            },
            "observed": {
                "frame_id": "assembly_board-v1",
                "x": -0.096,
                "y": 0.08,
                "z": 0.032,
                "point": "top_surface",
            },
            "xy_error_m": 0.004,
            "seating_error_m": 0.002,
        },
        {
            "part_name": "MG",
            "detected": True,
            "success": True,
            "expected": {
                "frame_id": "assembly_board-v1",
                "x": 0.0,
                "y": 0.08,
                "z": 0.03,
                "point": "top_surface",
            },
            "observed": {
                "frame_id": "assembly_board-v1",
                "x": 0.003,
                "y": 0.08,
                "z": 0.029,
                "point": "top_surface",
            },
            "xy_error_m": 0.003,
            "seating_error_m": -0.001,
        },
    ]
    _write_stationary_inspection_result(
        perception_manager,
        {
            "available": True,
            "success": True,
            "message": "SG and MG assembly inspection passed",
            "captured_at": captured_at,
            "aruco": {
                "marker_id": 70,
                "marker_length_m": 0.076,
                "captured_at": captured_at,
                "visible": True,
                "valid": True,
                "sample_count": 10,
                "required_sample_count": 10,
                "reprojection_error_px": 0.4,
                "translation_spread_m": 0.001,
                "rotation_spread_deg": 0.2,
            },
            "parts": parts,
        },
        preview_world_pose_ready=True,
    )
    monkeypatch.setattr(
        manager_module.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout="",
            stderr="",
        ),
    )

    result = perception_manager.test_detection("stationary")

    assert result["success"] is True
    assert result["world_pose_ready"] is False
    assert result["stationary_inspection_ready"] is True
    assert result["message"] == "SG and MG assembly inspection passed"
    assert result["stationary_inspection"]["xy_tolerance_m"] == pytest.approx(0.01)
    assert result["stationary_inspection"]["seating_tolerance_m"] == pytest.approx(
        0.005
    )
    assert result["stationary_inspection"]["parts"] == parts
    assert result["stationary_inspection"]["aruco"]["sample_count"] == 10
    assert result["stationary_inspection"]["unsupported_part_names"] == [
        "LG",
        "SRP",
        "MRP",
        "LRP",
        "SCP",
        "MCP",
        "LCP",
    ]


@pytest.mark.parametrize(
    "message",
    [
        "assembly_board-v1 ArUco ID 70 is not visible",
        "multiple assembly_board-v1 ArUco ID 70 markers are visible",
        "assembly_board-v1 ArUco stability window has 9/10 samples",
        "assembly_board-v1 ArUco pose is ambiguous",
    ],
)
def test_stationary_test_detection_preserves_fail_closed_marker_reason(
    perception_manager: PerceptionManager,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    message: str,
) -> None:
    _write_stationary_inspection_geometry(
        perception_manager,
        tmp_path / "configured",
        configured=True,
    )
    _write_stationary_inspection_result(
        perception_manager,
        {
            "available": False,
            "success": False,
            "message": message,
        },
    )
    monkeypatch.setattr(
        manager_module.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout="",
            stderr="",
        ),
    )

    result = perception_manager.test_detection("stationary")

    assert result["success"] is False
    assert result["stationary_inspection_ready"] is False
    assert result["message"] == message


def test_cross_camera_warning_threshold_is_ten_millimetres() -> None:
    cameras = {
        "ur5e": {
            "perception": {
                "detections": [
                    {"part_name": "SG", "frame_id": "world", "x": 0.0, "y": 0.0, "z": 1.0}
                ]
            }
        },
        "stationary": {
            "perception": {
                "detections": [
                    {
                        "part_name": "SG",
                        "frame_id": "world",
                        "x": 0.011,
                        "y": 0.0,
                        "z": 1.0,
                    }
                ]
            }
        },
    }
    rows = PerceptionManager._cross_camera_comparisons(cameras)
    assert rows[0]["disagreement_m"] == pytest.approx(0.011)
    assert rows[0]["warning"] is True
    cameras["stationary"]["inspection_only"] = True
    assert PerceptionManager._cross_camera_comparisons(cameras) == []


def test_stale_preview_and_external_viewer_lifecycle(
    perception_manager: PerceptionManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    perception_manager.save_assignments({"ur5e": "100"})
    monkeypatch.setattr(perception_manager, "discover_devices", lambda: [])
    preview_dir = manager_module.PREVIEW_ROOT / "ur5e"
    preview_dir.mkdir(parents=True)
    (preview_dir / "status.json").write_text(
        json.dumps({"frame_captured_at": time.time() - 10.0}),
        encoding="utf-8",
    )
    status = perception_manager.status()["cameras"]["ur5e"]
    assert status["frame_age_sec"] >= 9.0
    assert status["ros_topic_ready"] is False
    perception_manager.stop_camera("ur5e")
    assert "realsense_viewer_ur5e" in perception_manager.bridge.stopped
    assert "realsense_preview_ur5e" in perception_manager.bridge.stopped


def test_invalid_or_missing_wsl_attachment_is_safe(
    perception_manager: PerceptionManager,
) -> None:
    assert parse_usbipd_realsense_devices("") == []
    assert perception_manager.attach_wsl_camera("not-a-busid") == (
        "select a valid RealSense BUSID first"
    )


def test_calibration_activate_and_rollback_preserve_previous(
    perception_manager: PerceptionManager,
    tmp_path: Path,
) -> None:
    config = perception_manager.config()
    active = tmp_path / "ur5e_realsense_hand_eye.yaml"
    config["cameras"]["ur5e"]["calibration_path"] = str(active)
    manager_module._atomic_yaml_write(perception_manager.config_path, config)
    active.write_text("calibration_id: old\nvalidation:\n  accepted: true\n", encoding="utf-8")
    candidate = tmp_path / "ur5e_realsense_hand_eye.candidate.yaml"
    candidate.write_text(
        "calibration_id: new\nvalidation:\n  accepted: true\n",
        encoding="utf-8",
    )
    perception_manager.activate_calibration("ur5e")
    assert yaml.safe_load(active.read_text(encoding="utf-8"))["calibration_id"] == "new"
    perception_manager.rollback_calibration("ur5e")
    assert yaml.safe_load(active.read_text(encoding="utf-8"))["calibration_id"] == "old"


def test_stationary_sample_reset_archives_and_starts_an_empty_exact_role_set(
    perception_manager: PerceptionManager,
    tmp_path: Path,
) -> None:
    samples_path = tmp_path / "stationary_camera_samples.json"
    previous = {
        "camera_role": "stationary",
        "world_frame": "world",
        "tool_frame": "world",
        "stationary_camera": True,
        "samples": [{"captured_at": 123.0}],
    }
    samples_path.write_text(json.dumps(previous), encoding="utf-8")
    config = perception_manager.config()
    config["cameras"]["stationary"]["samples_path"] = str(samples_path)
    manager_module._atomic_yaml_write(perception_manager.config_path, config)

    result = perception_manager.reset_stationary_calibration_samples()

    archive_path = Path(result["archive_path"])
    assert archive_path.is_file()
    assert json.loads(archive_path.read_text(encoding="utf-8")) == previous
    assert json.loads(samples_path.read_text(encoding="utf-8")) == {
        "camera_role": "stationary",
        "world_frame": "world",
        "tool_frame": "world",
        "stationary_camera": True,
        "samples": [],
    }
    assert result["sample_count"] == 0
    assert result["robot_motion_requested"] is False


def test_stationary_capture_rejects_another_roles_existing_sample_set(
    perception_manager: PerceptionManager,
    tmp_path: Path,
) -> None:
    samples_path = tmp_path / "stationary_camera_samples.json"
    samples_path.write_text(
        json.dumps({"camera_role": "ur5e", "samples": []}),
        encoding="utf-8",
    )
    config = perception_manager.config()
    config["cameras"]["stationary"]["samples_path"] = str(samples_path)
    manager_module._atomic_yaml_write(perception_manager.config_path, config)

    with pytest.raises(RuntimeError, match=r"Archive \+ Reset Samples"):
        perception_manager.save_pose_and_capture("stationary")


def test_stationary_extrinsic_solver_uses_surveyed_world_pose(tmp_path: Path) -> None:
    identity = np.eye(4).tolist()
    samples = {
        "samples": [
            {
                "camera_to_board": identity,
                "camera_link_to_optical": identity,
                "reprojection_error_px": 0.1,
            }
            for _index in range(10)
        ]
    }
    samples_path = tmp_path / "stationary_samples.json"
    samples_path.write_text(json.dumps(samples), encoding="utf-8")
    output = tmp_path / "stationary_realsense_extrinsic.yaml"
    result = solve_stationary_calibration(
        samples_path,
        output,
        board_world_pose={
            "configured": True,
            "x": 0.2,
            "y": -0.1,
            "z": 1.2,
            "roll": 0.0,
            "pitch": 0.0,
            "yaw": 0.0,
        },
    )
    assert result["parent_frame"] == "world"
    assert result["accepted_pose_count"] == 10
    assert result["parent_to_camera_link"]["translation"]["z"] == pytest.approx(1.2)
    assert yaml.safe_load(output.read_text(encoding="utf-8"))["validation"]["accepted"]


def test_stationary_extrinsic_solver_rejects_fewer_than_ten_samples(
    tmp_path: Path,
) -> None:
    identity = np.eye(4).tolist()
    samples_path = tmp_path / "stationary_samples.json"
    samples_path.write_text(
        json.dumps(
            {
                "camera_role": "stationary",
                "samples": [
                    {
                        "camera_to_board": identity,
                        "camera_link_to_optical": identity,
                        "reprojection_error_px": 0.1,
                    }
                    for _index in range(9)
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="at least 10 accepted stationary"):
        solve_stationary_calibration(
            samples_path,
            tmp_path / "stationary_realsense_extrinsic.yaml",
            board_world_pose={
                "configured": True,
                "x": 0.2,
                "y": -0.1,
                "z": 1.2,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
            },
        )


def test_stationary_extrinsic_solver_rejects_inconsistent_fixed_camera_samples(
    tmp_path: Path,
) -> None:
    identity = np.eye(4)
    moved = identity.copy()
    moved[0, 3] = 0.1
    samples_path = tmp_path / "stationary_samples.json"
    samples_path.write_text(
        json.dumps(
            {
                "camera_role": "stationary",
                "samples": [
                    {
                        "camera_to_board": (moved if index == 9 else identity).tolist(),
                        "camera_link_to_optical": identity.tolist(),
                        "reprojection_error_px": 0.1,
                    }
                    for index in range(10)
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="stationary calibration rejected.*translation RMS"):
        solve_stationary_calibration(
            samples_path,
            tmp_path / "stationary_realsense_extrinsic.yaml",
            board_world_pose={
                "configured": True,
                "x": 0.2,
                "y": -0.1,
                "z": 1.2,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
            },
        )


def test_preview_and_page_contracts_do_not_invoke_inference_or_motion() -> None:
    preview_source = inspect.getsource(
        __import__(
            "cais_spade_llm.resources.sensor.physical.realsense_preview_node",
            fromlist=["RealSensePreviewNode"],
        )
    )
    page_source = Path("cais_spade_llm/ui/pages/perception.py").read_text(encoding="utf-8")
    app_source = Path("cais_spade_llm/ui/app.py").read_text(encoding="utf-8")
    assert "RoboflowGearDetector" not in preview_source
    assert "AssemblyBoardV1ArucoLocalizer(" in preview_source
    assert 'if camera_role in ("ur5e", "xarm6")' in preview_source
    assert "Camera & Perception" in page_source
    assert '"Perception", "/perception"' in app_source
    assert "/perception/stream/{camera_role}/{stream_name}" in app_source
    assert "/perception/frame/{camera_role}/{stream_name}" in app_source
    assert '"detection",' in app_source
    assert "/perception/frame/{role}/detection" in page_source
    assert 'ui.label("Detection view")' in page_source
    assert 'ui.element("img")' in page_source
    assert "ui.image(detection_url)" not in page_source
    assert "ui.image(color_url)" not in page_source
    assert "ui.image(depth_url)" not in page_source
    assert "_NO_PERCEPTION_FRAME_JPEG" in app_source
    assert '("no-perception-frame", 0)' in app_source
    assert "Last detection result" not in page_source
    assert '"Connect"' not in page_source
    assert '"Disconnect"' not in page_source
    assert "Start All Detection" in page_source
    assert "Stop All Detection" in page_source
    assert "fallback_path" in app_source
    assert "USB2 allowed; move to USB3.x if frames become stale or drop" in page_source
    assert "Windows D435:" in page_source
    assert "WSL attached:" in page_source
    assert "assigned roles:" in page_source
    assert "usbipd bind --busid {busid}" in page_source
    assert "row.get('state', 'Unknown')" in page_source
    assert "self._set_serial_options(devices)" in page_source
    assert "Preview Automatic Calibration" in page_source
    assert "I Confirm — Plan and Replay" in page_source
    assert "Local Control through read-only" in page_source
    assert "MoveIt and RG2 control remain stopped" in page_source
    assert "camera is inspection-only" in page_source
    assert "does not use this ChArUco calibration workflow" in page_source
    assert "Capture Sample" not in page_source
    assert "Archive + Reset Samples" not in page_source
    assert "Save Surveyed Pose" not in page_source
    assert "_render_stationary_board_pose" not in page_source
    assert "Stationary ID 70 (diagnostic only)" in page_source
    assert "SG/MG slot XY error (10 mm limit)" in page_source
    assert "Stationary assembly inspection supports SG and MG only" in page_source
    assert "Calibration live view — raw color" in page_source
    assert "Enlarge Calibration View" in page_source
    assert 'color_url = f"/perception/frame/{role}/color"' in page_source
    assert "calibration raw color" in page_source
    assert "ui.timer(0.5, self._refresh_images)" in page_source
    assert 'image.props["src"] = f"{source}?v={self.image_refresh_sequence}"' in page_source
    assert "ChArUco board={'visible' if charuco_visible else 'NOT VISIBLE'}" in page_source
    assert depth_preview(np.zeros((2, 3), dtype=np.uint16)).shape == (2, 3, 3)


def test_detection_annotation_draws_clipped_boxes_and_empty_result_banner() -> None:
    image = np.zeros((100, 120, 3), dtype=np.uint8)
    annotated = annotate_detection_frame(
        image,
        [
            {
                "part_name": "MG",
                "label": "medium_gear",
                "confidence": 0.923,
                "bbox": {
                    "center_x": 5.0,
                    "center_y": 50.0,
                    "width": 40.0,
                    "height": 50.0,
                },
            }
        ],
    )
    empty = annotate_detection_frame(image, [])
    assert annotated.shape == image.shape
    assert empty.shape == image.shape
    assert np.count_nonzero(annotated) > 0
    assert np.count_nonzero(empty) > 0


def test_node_declares_isolated_and_canonical_service_controls() -> None:
    source = Path(
        "cais_spade_llm/resources/sensor/physical/realsense_roboflow_node.py"
    ).read_text(encoding="utf-8")
    assert '"detect_all_service", "/perception/ur5e/detect_all"' in source
    assert '"detect_part_service", "/perception/ur5e/detect_part"' in source
    assert '"table_plane_measurement_service", ""' in source
    assert 'f"/perception/{self.camera_role}/table_plane_measurement"' in source
    assert "self._table_plane_measurement_service" in source
    assert '"publish_canonical_services", True' in source
    assert 'Trigger,\n                    "/detect_all"' in source
    assert 'Trigger,\n                    "/detect_part"' in source


def test_table_plane_calibration_uses_non_executable_measurement_service() -> None:
    calibration = Path(
        "cais_spade_llm/resources/sensor/physical/calibrate_hand_eye.py"
    ).read_text(encoding="utf-8")
    manager = Path("cais_spade_llm/ui/perception_manager.py").read_text(encoding="utf-8")

    service = "/perception/ur5e/table_plane_measurement"
    assert f'measurement_service: str = "{service}"' in calibration
    assert 'client = node.create_client(Trigger, service_name)' in calibration
    assert f'"--service {service}"' in manager
