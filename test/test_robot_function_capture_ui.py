"""Focused tests for predefined physical robot-function position capture."""

from __future__ import annotations

import asyncio
import inspect
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from cais_spade_llm.resources.robot import robot_task_runtime
from cais_spade_llm.ui import bridge as bridge_module
from cais_spade_llm.ui.bridge import SystemBridge
from cais_spade_llm.ui.pages import control

UR5E_JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]


def _world_base_pose(robot: str = "ur5e") -> dict[str, object]:
    return {
        "frame_id": "world",
        "child_frame_id": "link_base" if robot == "xarm6" else "base_link",
        "x": 0.0,
        "y": -0.5 if robot == "xarm6" else 0.5,
        "z": 1.021,
        "qx": 0.0,
        "qy": 0.0,
        "qz": 0.0,
        "qw": 1.0,
    }


def test_function_templates_come_from_exact_robot_task_steps() -> None:
    expected = {
        "pick_approach": [
            ("move_to_origin_resource_location", "move_to_named_pose", False),
            ("detect_parts", "detect_parts", False),
            ("compute_pick_targets", "compute_pick_targets", False),
            ("open_gripper", "open_gripper", False),
            ("move_above_part", "move_cartesian", True),
            ("descend", "move_cartesian", True),
        ],
        "pick_grasp": [
            ("grasp_part", "grasp_part", False),
            ("delay_after_grasp", "delay", False),
            ("lift", "move_relative", False),
        ],
        "place_approach": [
            ("move_to_destination_location", "move_to_named_pose", False),
            ("localize_assembly_board_v1", "localize_assembly_board_v1", False),
            ("compute_place_targets", "compute_place_targets", False),
            ("move_above_destination", "move_cartesian", True),
            ("descend", "move_cartesian", True),
        ],
        "place_insert": [
            ("delay_before_release", "delay", False),
            ("release_part", "release_part", False),
            ("delay_after_release", "delay", False),
            ("snap_part_to_slot", "snap_part_to_slot", False),
            ("lift", "move_relative", False),
        ],
        "move_home": [("move_home", "move_to_named_pose", False)],
    }

    assert set(SystemBridge.digital_twin_function_names()) == set(expected)
    for function_name, steps in expected.items():
        template = SystemBridge.digital_twin_function_template(function_name)
        assert [
            (step["step_name"], step["primitive"], step["recordable"]) for step in template
        ] == steps
    pick_template = SystemBridge.digital_twin_function_template("pick_approach")
    assert {
        step["parameter_source"]
        for step in pick_template
        if step["step_name"] in {"move_above_part", "descend"}
    } == {
        "Computed live by compute_pick_targets unless an optional axis source is saved."
    }
    assert {
        (step["step_name"], step["required"])
        for step in pick_template
        if step["recordable"]
    } == {("move_above_part", False), ("descend", False)}


def test_capture_readiness_uses_selected_ur5e_domain_without_control() -> None:
    bridge = object.__new__(SystemBridge)
    domains = {
        "gazebo": 41,
        "hardware": 42,
        "hardware_xarm6": 42,
        "hardware_ur5e": 43,
    }
    monitor_domains: list[int] = []
    bridge._digital_twin_target = lambda _target: {"hardware": ("ur5e",)}
    bridge._digital_twin_domain_ids = lambda: domains
    bridge._digital_twin_hardware_domain_id = lambda _cfg, robot, values: values[
        f"hardware_{robot}"
    ]
    bridge.perception_manager = SimpleNamespace(
        ensure_ur5e_calibration_monitor=lambda *, domain_id: monitor_domains.append(domain_id)
    )
    bridge._read_json_file = lambda _path: {
        "rtde_receive_connected": True,
        "joint_states_fresh": True,
        "rtde_control_connected": False,
    }

    def _snapshot(
        robot: str,
        *,
        source: str,
        hardware_domain_id: int,
        include_world_tool_pose: bool,
    ) -> dict[str, object]:
        assert (robot, source, hardware_domain_id, include_world_tool_pose) == (
            "ur5e",
            "hardware",
            43,
            True,
        )
        return {
            "positions": [0.1] * 6,
            "joint_names": list(UR5E_JOINT_NAMES),
            "pose": {
                "frame_id": "world",
                "child_frame_id": "tool0",
                "x": 0.4,
                "y": 0.3,
                "z": 1.2,
                "qx": 0.0,
                "qy": 0.0,
                "qz": 0.0,
                "qw": 1.0,
            },
        }

    bridge._snapshot_robot_waypoint = _snapshot

    result = bridge.digital_twin_function_capture_readiness("ur5e only", "ur5e")

    assert result["success"] is True
    assert result["hardware_domain_id"] == 43
    assert result["rtde_receive_connected"] is True
    assert result["joint_states_fresh"] is True
    assert result["world_tool0_ready"] is True
    assert result["rtde_control_connected"] is False
    assert result["blocked_reason"] == ""
    assert monitor_domains == [43]


def test_capture_readiness_reports_rtde_failure_instead_of_rg2_only_joint() -> None:
    bridge = object.__new__(SystemBridge)
    bridge._digital_twin_target = lambda _target: {"hardware": ("ur5e",)}
    bridge._digital_twin_domain_ids = lambda: {"hardware": 42}
    bridge._digital_twin_hardware_domain_id = lambda *_args: 42
    bridge.perception_manager = SimpleNamespace(
        ensure_ur5e_calibration_monitor=lambda *, domain_id: None
    )
    failed_status = {
        "updated_at": time.time(),
        "state": "failed",
        "blocked_reason": "UR5e RTDE feedback transport failed: End of file.",
        "rtde_receive_connected": False,
        "joint_states_fresh": False,
        "rtde_control_connected": False,
        "rtde_reset_required": True,
    }
    bridge._read_json_file = lambda path: (
        failed_status
        if path == bridge_module._UR5E_RTDE_TRAJECTORY_STATUS
        else {}
    )
    bridge._snapshot_robot_waypoint = lambda *_args, **_kwargs: {
        "error": (
            "hardware /joint_states has no ur5e arm joints yet. "
            "Seen joints: ur5e_rg2_finger_width."
        )
    }

    result = bridge.digital_twin_function_capture_readiness("ur5e only", "ur5e")

    assert result["success"] is False
    assert result["blocked_reason"] == (
        "UR5e RTDE feedback transport failed: End of file."
    )
    assert "ur5e_rg2_finger_width" not in result["blocked_reason"]


def test_preview_detection_uses_selected_ur5e_hardware_domain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge = object.__new__(SystemBridge)
    domains = {
        "gazebo": 41,
        "hardware": 42,
        "hardware_xarm6": 42,
        "hardware_ur5e": 43,
    }
    selected_domains: list[int] = []
    bridge._digital_twin_target = lambda target: (
        {"hardware": ("ur5e",)} if target == "ur5e only" else None
    )
    bridge._digital_twin_domain_ids = lambda: domains
    bridge._digital_twin_hardware_domain_id = lambda _cfg, robot, values: values[
        f"hardware_{robot}"
    ]
    bridge._ros2_domain_export = lambda domain_id: selected_domains.append(domain_id) or ""
    bridge.ros2_proc_status = lambda _name: "stopped"
    bridge.physical_perception_status = lambda: {
        "updated_at": time.time() + 1.0,
        "last_error": "",
        "detections": [],
    }
    monkeypatch.setattr(
        bridge_module.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=0, stderr="", stdout=""),
    )

    result = bridge.test_physical_detection("ur5e only")

    assert result["success"] is True
    assert selected_domains == [43]


def test_physical_perception_readiness_uses_live_camera_frame(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = time.time()
    snapshot_path = tmp_path / "cais_physical_perception.json"
    snapshot_path.write_text(
        json.dumps(
            {
                "frame_captured_at": now - 30.0,
                "last_error": "",
                "realsense_connected": True,
                "roboflow_ready": True,
                "table_plane_ready": True,
            }
        ),
        encoding="utf-8",
    )
    camera_status_path = tmp_path / "status.json"
    camera_status_path.write_text(
        json.dumps({"frame_captured_at": now, "last_error": ""}),
        encoding="utf-8",
    )
    calibration_path = tmp_path / "hand_eye.yaml"
    calibration_path.write_text(
        """validation:
  accepted: true
table_plane:
  accepted: true
  world_frame: world
  frame_count: 10
  mad_m: 0.001
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(bridge_module, "_PHYSICAL_PERCEPTION_SNAPSHOT", snapshot_path)
    monkeypatch.setattr(bridge_module, "_UR5E_CAMERA_STATUS", camera_status_path)
    monkeypatch.setenv("ROBOFLOW_API_KEY", "test-only")

    bridge = object.__new__(SystemBridge)
    bridge.perception_manager = SimpleNamespace(
        config=lambda: {
            "cameras": {"ur5e": {"calibration_path": str(calibration_path)}}
        }
    )
    bridge.ros2_proc_status = lambda _name: "running"

    status = bridge.physical_perception_status()
    ready, reason = bridge.physical_perception_ready()

    assert status["frame_age_sec"] < 1.0
    assert status["snapshot_frame_age_sec"] >= 29.0
    assert status["frame_captured_at"] == now
    assert status["snapshot_frame_captured_at"] == now - 30.0
    assert ready is True
    assert reason == ""


def test_physical_perception_readiness_rejects_stale_live_camera_frame(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = time.time()
    snapshot_path = tmp_path / "cais_physical_perception.json"
    snapshot_path.write_text(
        json.dumps(
            {
                "frame_captured_at": now,
                "last_error": "",
                "realsense_connected": True,
                "roboflow_ready": True,
                "table_plane_ready": True,
            }
        ),
        encoding="utf-8",
    )
    camera_status_path = tmp_path / "status.json"
    camera_status_path.write_text(
        json.dumps({"frame_captured_at": now - 30.0, "last_error": ""}),
        encoding="utf-8",
    )
    calibration_path = tmp_path / "hand_eye.yaml"
    calibration_path.write_text(
        """validation:
  accepted: true
table_plane:
  accepted: true
  world_frame: world
  frame_count: 10
  mad_m: 0.001
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(bridge_module, "_PHYSICAL_PERCEPTION_SNAPSHOT", snapshot_path)
    monkeypatch.setattr(bridge_module, "_UR5E_CAMERA_STATUS", camera_status_path)
    monkeypatch.setenv("ROBOFLOW_API_KEY", "test-only")

    bridge = object.__new__(SystemBridge)
    bridge.perception_manager = SimpleNamespace(
        config=lambda: {
            "cameras": {"ur5e": {"calibration_path": str(calibration_path)}}
        }
    )
    bridge.ros2_proc_status = lambda _name: "running"

    ready, reason = bridge.physical_perception_ready()

    assert ready is False
    assert "synchronized RealSense color/depth stream is stale" in reason


def test_capture_upserts_exact_step_with_world_pose_and_joints() -> None:
    bridge = object.__new__(SystemBridge)
    bridge._digital_twin_function_steps = {}
    bridge._digital_twin_record_lock = threading.Lock()
    bridge._ur5e_robot_function_execution_lock = threading.Lock()
    bridge._ur5e_robot_function_preflight_lock = threading.Lock()
    bridge._ur5e_robot_function_execution_active = None
    bridge._robot_function_validate_request = lambda *_args: ({}, "")
    bridge._robot_function_capture_source = lambda *_args: "hardware"
    motion_calls: list[str] = []
    agent_motion_lock = threading.Lock()
    resource_agent = SimpleNamespace(
        execution_mode="physical",
        _controller=SimpleNamespace(
            move_cartesian=lambda **_kwargs: motion_calls.append("move_cartesian"),
            move_to_named_pose=lambda **_kwargs: motion_calls.append("move_to_named_pose"),
        ),
        _robot_motion_lock=agent_motion_lock,
    )
    bridge._physical_ur5e_robot_agent = lambda: resource_agent
    board_reference = {
        "kind": "destination_target",
        "frame_id": "world",
        "name": "assembly_board-v1",
        "position_m": {"x": 0.4, "y": 0.3, "z": 1.0},
        "pose": {
            "x": 0.4,
            "y": 0.3,
            "z": 1.0,
            "qx": 0.0,
            "qy": 0.0,
            "qz": 0.7071067811865476,
            "qw": 0.7071067811865476,
        },
        "source": "assembly_board-v1_aruco",
        "captured_at": time.time(),
        "camera_role": "ur5e",
        "generation": 1,
        "calibration_id": "ur5e-calibration",
        "marker_length_m": 0.076,
        "observation_source": "accepted",
    }
    capture_index = {"value": 0}

    def _capture_snapshot(*_args: object) -> dict[str, object]:
        capture_index["value"] += 1
        x = 0.4 + capture_index["value"] / 100.0
        return {
            "success": True,
            "blocked_reason": "",
            "waypoint": {
                "world_base_pose": _world_base_pose(),
                "joint_names": list(UR5E_JOINT_NAMES),
                "positions": [float(capture_index["value"])] * 6,
                "pose": {
                    "frame_id": "world",
                    "child_frame_id": "tool0",
                    "x": x,
                    "y": 0.3,
                    "z": 1.2,
                    "qx": 0.0,
                    "qy": 0.0,
                    "qz": 0.0,
                    "qw": 1.0,
                },
            },
        }

    bridge._robot_function_capture_snapshot = _capture_snapshot
    resolution_calls: list[dict[str, object]] = []

    def _resolve(*_args: object, **kwargs: object) -> dict[str, object]:
        assert bridge._ur5e_robot_function_execution_lock.locked()
        assert bridge._ur5e_robot_function_preflight_lock.locked()
        assert agent_motion_lock.locked()
        assert kwargs["resource_agent"] is resource_agent
        resolution_calls.append(dict(kwargs))
        return {
            "success": True,
            "computed_position": {
                "frame_id": "world",
                "x": 0.4,
                "y": 0.3,
                "z": 1.0,
            },
            "computed_reference": dict(board_reference),
            "computed_source": "capture",
            "computed_at": time.time(),
            "resolved_position": {"x": 0.4, "y": 0.3, "z": 1.0},
        }

    bridge._resolve_robot_function_position = _resolve
    bridge._robot_function_computed_pose = lambda **_kwargs: pytest.fail(
        "Capture Pose must not read a preview-time computed-pose cache"
    )

    bridge._ur5e_robot_function_execution_lock.acquire()
    bridge._ur5e_robot_function_execution_active = "place_approach"
    busy = bridge.digital_twin_capture_function_step(
        "ur5e only",
        "ur5e",
        "place_approach",
        "assembly_board-v1",
        "move_above_destination",
        "move_cartesian",
        part_name="MG",
    )
    bridge._ur5e_robot_function_execution_lock.release()
    bridge._ur5e_robot_function_execution_active = None
    first = bridge.digital_twin_capture_function_step(
        "ur5e only",
        "ur5e",
        "place_approach",
        "assembly_board-v1",
        "move_above_destination",
        "move_cartesian",
        part_name="MG",
    )
    second = bridge.digital_twin_capture_function_step(
        "ur5e only",
        "ur5e",
        "place_approach",
        "assembly_board-v1",
        "move_above_destination",
        "move_cartesian",
        part_name="MG",
    )

    assert busy["success"] is False
    assert busy["active_function"] == "place_approach"
    assert first["success"] is True
    assert second["success"] is True
    assert second["count"] == 1
    assert len(resolution_calls) == 2
    assert all(call["computed_source"] == "capture" for call in resolution_calls)
    assert motion_calls == []
    assert first["computed_position_m"] == pytest.approx(
        {"x": 0.4, "y": 0.3, "z": 1.0}
    )
    assert first["current_position_m"] == pytest.approx(
        {"x": 0.41, "y": 0.3, "z": 1.2}
    )
    assert first["relative_position_m"] == pytest.approx(
        {"x": 0.0, "y": -0.01, "z": 0.2}
    )
    assert first["relative_pose"] == pytest.approx(
        {
            "x": 0.0,
            "y": -0.01,
            "z": 0.2,
            "qx": 0.0,
            "qy": 0.0,
            "qz": -0.7071067811865476,
            "qw": 0.7071067811865476,
        }
    )
    buffered = bridge.digital_twin_list_function_buffer_steps(
        "ur5e only",
        "ur5e",
        "place_approach",
        "assembly_board-v1",
        part_name="MG",
    )
    assert len(buffered) == 1
    assert buffered[0]["step_name"] == "move_above_destination"
    assert buffered[0]["primitive"] == "move_cartesian"
    assert buffered[0]["params"]["x"] == pytest.approx(0.42)
    assert buffered[0]["pose"]["frame_id"] == "world"
    assert buffered[0]["joint_positions"] == [2.0] * 6
    assert buffered[0]["position_sources"] == {
        "x": "captured_relative",
        "y": "captured_relative",
        "z": "captured_relative",
    }
    assert buffered[0]["relative_position_m"] == pytest.approx(
        {"x": 0.0, "y": -0.02, "z": 0.2}
    )


def test_hardware_stack_generation_clears_computed_pose_cache() -> None:
    bridge = object.__new__(SystemBridge)
    bridge._digital_twin_record_lock = threading.Lock()
    bridge._digital_twin_function_computed_poses = {"cached": {"x": 1.0}}
    bridge._hardware_stack_lifecycle_generation = 4

    generation = bridge._next_hardware_stack_generation()

    assert generation == 5
    assert bridge._digital_twin_function_computed_poses == {}


def test_computed_pose_provenance_rejects_calibration_identity_change() -> None:
    cached = {
        "hardware_stack_generation": 7,
        "calibration_identity": "calibration-a",
        "target_context": {"target": "ur5e only"},
        "world_base_pose": _world_base_pose(),
    }
    current = {
        **cached,
        "calibration_identity": "calibration-b",
    }

    assert (
        SystemBridge._robot_function_computed_pose_provenance_error(cached, current)
        == "the calibration identity changed"
    )


def test_capture_resolves_without_preview_cache_and_does_not_move() -> None:
    bridge = object.__new__(SystemBridge)
    bridge._digital_twin_function_steps = {}
    bridge._digital_twin_record_lock = threading.Lock()
    bridge._ur5e_robot_function_execution_lock = threading.Lock()
    bridge._ur5e_robot_function_preflight_lock = threading.Lock()
    bridge._ur5e_robot_function_execution_active = None
    bridge._robot_function_validate_request = lambda *_args: ({}, "")
    bridge._robot_function_capture_source = lambda *_args: "hardware"
    motion_calls: list[str] = []
    resource_agent = SimpleNamespace(
        execution_mode="physical",
        _controller=SimpleNamespace(
            move_cartesian=lambda **_kwargs: motion_calls.append("move_cartesian"),
            move_to_named_pose=lambda **_kwargs: motion_calls.append("move_to_named_pose"),
        ),
    )
    bridge._physical_ur5e_robot_agent = lambda: resource_agent
    bridge._robot_function_capture_snapshot = lambda *_args: {
        "success": True,
        "waypoint": {
            "world_base_pose": _world_base_pose(),
            "joint_names": list(UR5E_JOINT_NAMES),
            "positions": [0.0] * 6,
            "pose": {
                "frame_id": "world",
                "child_frame_id": "tool0",
                "x": 0.4,
                "y": 0.3,
                "z": 1.2,
                "qx": 0.0,
                "qy": 0.0,
                "qz": 0.0,
                "qw": 1.0,
            },
        },
    }
    resolution_calls: list[dict[str, object]] = []

    def _resolve(*_args: object, **kwargs: object) -> dict[str, object]:
        resolution_calls.append(dict(kwargs))
        return {
            "success": True,
            "computed_position": {"x": 0.4, "y": 0.3, "z": 1.2},
            "computed_reference": {
                "kind": "detected_part",
                "frame_id": "world",
                "name": "MG",
                "position_m": {"x": 0.4, "y": 0.3, "z": 1.2},
                "source": "live_detection",
                "captured_at": time.time(),
            },
            "computed_source": "capture",
            "computed_at": time.time(),
            "resolved_position": {"x": 0.4, "y": 0.3, "z": 1.2},
        }

    bridge._resolve_robot_function_position = _resolve
    bridge._robot_function_computed_pose = lambda **_kwargs: pytest.fail(
        "Capture Pose must not read a preview-time computed-pose cache"
    )

    result = bridge.digital_twin_capture_function_step(
        "ur5e only",
        "ur5e",
        "pick_approach",
        "prusa-mk4-2",
        "descend",
        "move_cartesian",
        part_name="MG",
    )

    assert result["success"] is True
    assert len(resolution_calls) == 1
    assert resolution_calls[0]["computed_source"] == "capture"
    assert motion_calls == []
    assert len(bridge._digital_twin_function_steps) == 1


def test_capture_rejects_staging_pose_far_from_computed_pose_without_buffering() -> None:
    bridge = object.__new__(SystemBridge)
    bridge._digital_twin_function_steps = {}
    bridge._digital_twin_record_lock = threading.Lock()
    bridge._ur5e_robot_function_execution_lock = threading.Lock()
    bridge._ur5e_robot_function_preflight_lock = threading.Lock()
    bridge._ur5e_robot_function_execution_active = None
    bridge._robot_function_validate_request = lambda *_args: ({}, "")
    bridge._robot_function_capture_source = lambda *_args: "hardware"
    bridge._physical_ur5e_robot_agent = lambda: SimpleNamespace(
        execution_mode="physical",
        _controller=object(),
    )
    bridge._robot_function_capture_snapshot = lambda *_args: {
        "success": True,
        "waypoint": {
            "world_base_pose": _world_base_pose(),
            "joint_names": list(UR5E_JOINT_NAMES),
            "positions": [0.0] * 6,
            "pose": {
                "frame_id": "world",
                "child_frame_id": "tool0",
                "x": 0.322814,
                "y": -0.126268,
                "z": 1.266863,
                "qx": 0.0,
                "qy": 0.0,
                "qz": 0.0,
                "qw": 1.0,
            },
        },
    }
    bridge._resolve_robot_function_position = lambda *_args, **_kwargs: {
        "success": True,
        "computed_position": {"x": 0.323193, "y": 0.373448, "z": 1.256266},
        "computed_reference": {
            "kind": "detected_part",
            "frame_id": "world",
            "name": "MG",
            "position_m": {"x": 0.323193, "y": 0.373448, "z": 1.256266},
            "source": "live_detection",
            "captured_at": time.time(),
        },
        "computed_source": "capture",
        "computed_at": time.time(),
        "resolved_position": {"x": 0.323193, "y": 0.373448, "z": 1.256266},
    }

    result = bridge.digital_twin_capture_function_step(
        "ur5e only",
        "ur5e",
        "pick_approach",
        "prusa-mk4-2",
        "descend",
        "move_cartesian",
        part_name="MG",
    )

    assert result["success"] is False
    assert "Y=-499.7 mm" in result["message"]
    assert "250 mm calibration limit" in result["message"]
    assert "No pose was buffered" in result["message"]
    assert result["relative_position_m"] == pytest.approx(
        {"x": -0.000379, "y": -0.499716, "z": 0.010597}
    )
    assert bridge._digital_twin_function_steps == {}


def test_pick_reference_capture_requests_unfiltered_detections() -> None:
    bridge = object.__new__(SystemBridge)
    bridge._robot_function_product_geometry_for_part = lambda _part_name: {}
    detect_arguments: list[str | None] = []

    def detect_parts(part_name: str | None = None) -> list[dict[str, object]]:
        detect_arguments.append(part_name)
        return [
            {
                "part_name": "SG",
                "frame_id": "world",
                "x": 0.1,
                "y": 0.2,
                "z": 0.3,
                "captured_at": time.time(),
            }
        ]

    controller = SimpleNamespace(
        _last_failure_message="",
        _last_start_pose=None,
        detect_parts=detect_parts,
        compute_pick_targets=lambda **_kwargs: {"success": True},
    )

    result = bridge._robot_function_relative_reference(
        resource_agent=SimpleNamespace(_controller=controller),
        function_name="pick_approach",
        name="prusa-mk4-2",
        step_name="descend",
        part_name="MG",
    )

    assert result["success"] is False
    assert detect_arguments == [None]
    assert "latest accepted detections were ['SG']" in result["message"]
    assert "detected=[]" not in result["message"]


def test_pick_reference_capture_explains_empty_validated_frame() -> None:
    bridge = object.__new__(SystemBridge)
    bridge._robot_function_product_geometry_for_part = lambda _part_name: {}
    controller = SimpleNamespace(
        _last_failure_message="",
        _last_start_pose=None,
        detect_parts=lambda _part_name=None: [],
        compute_pick_targets=lambda **_kwargs: {"success": True},
    )

    result = bridge._robot_function_relative_reference(
        resource_agent=SimpleNamespace(_controller=controller),
        function_name="pick_approach",
        name="prusa-mk4-2",
        step_name="descend",
        part_name="MG",
    )

    assert result["success"] is False
    assert "latest validated camera frame contained no accepted part detections" in result[
        "message"
    ]
    assert "existing saved position is unchanged" in result["message"]
    assert "detected=[]" not in result["message"]


def test_place_reference_uses_accepted_pose_when_tag_is_occluded() -> None:
    bridge = object.__new__(SystemBridge)
    bridge._robot_function_product_geometry_for_part = lambda _part_name: {}
    accepted_pose = {
        "x": 0.4,
        "y": 0.3,
        "z": 1.0,
        "qx": 0.0,
        "qy": 0.0,
        "qz": 0.0,
        "qw": 1.0,
    }
    compute_arguments: list[dict[str, object]] = []

    def _compute_place_targets(**kwargs: object) -> dict[str, object]:
        compute_arguments.append(dict(kwargs))
        return {
            "success": True,
            "approach_pose": {"x": 0.4, "y": 0.3, "z": 1.2},
            "target_pose": {"x": 0.4, "y": 0.3, "z": 1.1},
        }

    resource_agent = SimpleNamespace(
        agent_name="ur5e",
        _task_ctx={},
        _controller=SimpleNamespace(
            _last_start_pose=None,
            compute_place_targets=_compute_place_targets,
        ),
    )
    bridge.perception_manager = SimpleNamespace(
        assembly_board_v1_aruco_status=lambda role: {
            "camera_role": role,
            "accepted": True,
            "accepted_baseline_ready": True,
            "accepted_generation": 3,
            "accepted_pose": dict(accepted_pose),
            "accepted_at": time.time() - 30.0,
            "accepted_frame_captured_at": time.time() - 31.0,
            "accepted_calibration_id": "ur5e-calibration",
            "calibration_id": "",
            "ready_to_accept": False,
            "visible": False,
            "movement_blocked": True,
            "marker_length_m": 0.076,
            "dictionary": "DICT_ARUCO_ORIGINAL",
            "marker_id": 70,
        }
    )

    result = bridge._robot_function_relative_reference(
        resource_agent=resource_agent,
        function_name="place_approach",
        name="assembly_board-v1",
        step_name="descend",
        part_name="MG",
        robot="ur5e",
    )

    assert result["success"] is True, result
    assert result["reference"]["pose"] == pytest.approx(accepted_pose)
    assert result["reference"]["observation_source"] == "accepted"
    assert result["reference"]["calibration_id"] == "ur5e-calibration"
    frozen = compute_arguments[0]["assembly_board_v1_aruco"]
    assert isinstance(frozen, dict)
    assert frozen["pose"] == pytest.approx(accepted_pose)
    assert frozen["observation_source"] == "accepted"


def test_capture_resolution_allows_an_older_accepted_pose_when_tag_is_occluded() -> None:
    bridge = object.__new__(SystemBridge)
    bridge._robot_function_validate_request = lambda *_args: ({}, "")
    bridge._physical_robot_function_cartesian_error = lambda _robot: ""
    accepted_at = time.time() - 60.0
    board_pose = {
        "x": 0.4,
        "y": 0.3,
        "z": 1.0,
        "qx": 0.0,
        "qy": 0.0,
        "qz": 0.0,
        "qw": 1.0,
    }
    reference = {
        "kind": "destination_target",
        "frame_id": "world",
        "name": "assembly_board-v1",
        "position_m": {"x": 0.4, "y": 0.3, "z": 1.0},
        "pose": dict(board_pose),
        "source": "assembly_board-v1_aruco",
        "captured_at": accepted_at,
        "camera_role": "ur5e",
        "generation": 3,
        "calibration_id": "ur5e-calibration",
    }
    bridge._robot_function_relative_reference = lambda **_kwargs: {
        "success": True,
        "reference": dict(reference),
        "computed": {
            "success": True,
            "destination_location": "assembly_board-v1",
            "approach_pose": {"x": 0.4, "y": 0.3, "z": 1.2},
            "target_pose": {"x": 0.4, "y": 0.3, "z": 1.1},
        },
        "assembly_board_v1_aruco": {
            "destination_location": "assembly_board-v1",
            "camera_role": "ur5e",
            "generation": 3,
            "calibration_id": "ur5e-calibration",
            "captured_at": accepted_at,
            "frame_id": "world",
            "pose": dict(board_pose),
        },
    }
    bridge._cache_robot_function_computed_pose = lambda **kwargs: {
        "computed_pose": dict(kwargs["computed_position"]),
        "relative_reference": dict(kwargs["relative_reference"]),
    }
    resource_agent = SimpleNamespace(
        agent_name="ur5e",
        execution_mode="physical",
        _controller=object(),
        _task_ctx={},
    )

    result = bridge._resolve_robot_function_position(
        "ur5e only",
        "ur5e",
        "place_approach",
        "assembly_board-v1",
        "descend",
        part_name="MG",
        readiness={"success": True},
        resource_agent=resource_agent,
        computed_source="capture",
    )

    assert result["success"] is True, result
    assert result["computed_reference"]["captured_at"] == pytest.approx(accepted_at)
    assert {
        axis: result["resolved_position"][axis] for axis in ("x", "y", "z")
    } == pytest.approx({"x": 0.4, "y": 0.3, "z": 1.1})


def test_place_reference_prefers_current_stable_tag_pose() -> None:
    bridge = object.__new__(SystemBridge)
    bridge._robot_function_product_geometry_for_part = lambda _part_name: {}
    live_pose = {
        "x": 0.405,
        "y": 0.302,
        "z": 1.0,
        "qx": 0.0,
        "qy": 0.0,
        "qz": 0.0,
        "qw": 1.0,
    }
    resource_agent = SimpleNamespace(
        agent_name="xarm6",
        _task_ctx={},
        _controller=SimpleNamespace(
            _last_start_pose=None,
            compute_place_targets=lambda **_kwargs: {
                "success": True,
                "approach_pose": {"x": 0.405, "y": 0.302, "z": 1.2},
                "target_pose": {"x": 0.405, "y": 0.302, "z": 1.1},
            },
        ),
    )
    bridge.perception_manager = SimpleNamespace(
        assembly_board_v1_aruco_status=lambda role: {
            "camera_role": role,
            "accepted": True,
            "accepted_baseline_ready": True,
            "accepted_generation": 7,
            "accepted_pose": {**live_pose, "x": 0.4},
            "accepted_frame_captured_at": time.time() - 20.0,
            "accepted_calibration_id": "xarm6-calibration",
            "calibration_id": "xarm6-calibration",
            "ready_to_accept": True,
            "movement_blocked": False,
            "pose": dict(live_pose),
            "frame_captured_at": time.time(),
            "marker_length_m": 0.076,
            "dictionary": "DICT_ARUCO_ORIGINAL",
            "marker_id": 70,
            "sample_count": 10,
            "reprojection_error_px": 0.2,
            "translation_spread_m": 0.0004,
            "rotation_spread_deg": 0.1,
        }
    )

    result = bridge._robot_function_relative_reference(
        resource_agent=resource_agent,
        function_name="place_approach",
        name="assembly_board-v1",
        step_name="descend",
        part_name="MG",
        robot="xarm6",
    )

    assert result["success"] is True, result
    assert result["reference"]["pose"] == pytest.approx(live_pose)
    assert result["reference"]["observation_source"] == "live_stable"
    assert result["reference"]["camera_role"] == "xarm6"


@pytest.mark.parametrize(
    ("status_update", "message"),
    [
        ({"accepted": False}, "Locate & Accept Board"),
        (
            {
                "calibration_id": "replacement-calibration",
                "accepted_calibration_id": "ur5e-calibration",
            },
            "calibration identity changed",
        ),
        (
            {"ready_to_accept": True, "movement_blocked": True},
            "moved more than 10 mm or 2 deg",
        ),
    ],
)
def test_place_reference_rejects_missing_acceptance_calibration_change_or_known_movement(
    status_update: dict[str, object],
    message: str,
) -> None:
    bridge = object.__new__(SystemBridge)
    bridge._robot_function_product_geometry_for_part = lambda _part_name: {}
    accepted_pose = {
        "x": 0.4,
        "y": 0.3,
        "z": 1.0,
        "qx": 0.0,
        "qy": 0.0,
        "qz": 0.0,
        "qw": 1.0,
    }
    status = {
        "accepted": True,
        "accepted_baseline_ready": True,
        "accepted_generation": 1,
        "accepted_pose": accepted_pose,
        "accepted_frame_captured_at": time.time() - 5.0,
        "accepted_calibration_id": "ur5e-calibration",
        "calibration_id": "ur5e-calibration",
        "ready_to_accept": False,
        "movement_blocked": False,
        "marker_length_m": 0.076,
    }
    status.update(status_update)
    bridge.perception_manager = SimpleNamespace(
        assembly_board_v1_aruco_status=lambda _role: dict(status)
    )
    resource_agent = SimpleNamespace(
        agent_name="ur5e",
        _task_ctx={},
        _controller=SimpleNamespace(
            _last_start_pose=None,
            compute_place_targets=lambda **_kwargs: pytest.fail(
                "invalid board state must fail before target computation"
            ),
        ),
    )

    result = bridge._robot_function_relative_reference(
        resource_agent=resource_agent,
        function_name="place_approach",
        name="assembly_board-v1",
        step_name="descend",
        part_name="MG",
        robot="ur5e",
    )

    assert result["success"] is False
    assert message in result["message"]


def test_xarm6_capture_uses_world_link_eef_relative_xyz() -> None:
    bridge = object.__new__(SystemBridge)
    bridge._digital_twin_function_steps = {}
    bridge._digital_twin_record_lock = threading.Lock()
    bridge._ur5e_robot_function_execution_lock = threading.Lock()
    bridge._ur5e_robot_function_preflight_lock = threading.Lock()
    bridge._ur5e_robot_function_execution_active = None
    bridge._robot_function_validate_request = lambda *_args: ({}, "")
    bridge._robot_function_capture_source = lambda *_args: "hardware"
    bridge._physical_xarm6_robot_agent = lambda: SimpleNamespace(
        execution_mode="physical",
        _controller=object(),
    )
    bridge._robot_function_capture_snapshot = lambda *_args: {
        "success": True,
        "waypoint": {
            "world_base_pose": _world_base_pose("xarm6"),
            "joint_names": [f"joint{index}" for index in range(1, 7)],
            "positions": [0.1] * 6,
            "pose": {
                "frame_id": "world",
                "child_frame_id": "link_eef",
                "x": 0.35,
                "y": -0.45,
                "z": 1.15,
                "qx": 0.0,
                "qy": 0.0,
                "qz": 1.0,
                "qw": 0.0,
            },
        },
    }
    bridge._resolve_robot_function_position = lambda *_args, **_kwargs: {
        "success": True,
        "computed_position": {"x": 0.30, "y": -0.40, "z": 1.05},
        "computed_reference": {
            "kind": "detected_part",
            "frame_id": "world",
            "name": "RG",
            "position_m": {"x": 0.30, "y": -0.40, "z": 1.05},
            "source": "live_detection",
            "captured_at": time.time(),
        },
        "computed_source": "capture",
        "computed_at": time.time(),
        "resolved_position": {"x": 0.30, "y": -0.40, "z": 1.05},
    }

    result = bridge.digital_twin_capture_function_step(
        "dual robots",
        "xarm6",
        "pick_approach",
        "prusa-mk4-1",
        "descend",
        "move_cartesian",
        part_name="RG",
    )

    assert result["success"] is True
    assert result["step"]["waypoint"]["pose"]["child_frame_id"] == "link_eef"
    assert result["step"]["relative_position_m"] == pytest.approx(
        {"x": 0.05, "y": -0.05, "z": 0.10}
    )


def test_pick_capture_persists_exact_context_axis_sources_and_hardware_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(bridge_module, "_ROBOT_TAUGHT_FUNCTIONS_DIR", tmp_path)
    bridge = object.__new__(SystemBridge)
    bridge._digital_twin_function_steps = {}
    bridge._digital_twin_record_lock = threading.Lock()
    bridge._ur5e_robot_function_execution_lock = threading.Lock()
    bridge._ur5e_robot_function_preflight_lock = threading.Lock()
    bridge._ur5e_robot_function_execution_active = None
    bridge._robot_function_validate_request = lambda *_args: ({}, "")
    bridge._robot_function_capture_source = lambda *_args: "hardware"
    bridge._robot_function_storage_source = lambda *_args: "hardware"
    bridge._physical_ur5e_robot_agent = lambda: SimpleNamespace(
        execution_mode="physical",
        _controller=object(),
    )
    bridge._resolve_robot_function_position = lambda *_args, **_kwargs: {
        "success": True,
        "computed_position": {"x": 0.36, "y": 0.30, "z": 1.10},
        "computed_reference": {
            "kind": "detected_part",
            "frame_id": "world",
            "name": "MG",
            "position_m": {"x": 0.36, "y": 0.30, "z": 1.10},
            "source": "live_detection",
            "captured_at": time.time(),
        },
        "computed_source": "capture",
        "computed_at": time.time(),
        "resolved_position": {"x": 0.36, "y": 0.30, "z": 1.10},
    }
    bridge._robot_function_capture_snapshot = lambda *_args: {
        "success": True,
        "blocked_reason": "",
        "waypoint": {
            "world_base_pose": _world_base_pose(),
            "joint_names": list(UR5E_JOINT_NAMES),
            "positions": [0.1 * index for index in range(6)],
            "pose": {
                "frame_id": "world",
                "child_frame_id": "tool0",
                "x": 0.41,
                "y": 0.32,
                "z": 1.23456789,
                "qx": 0.0,
                "qy": 0.70710678,
                "qz": 0.0,
                "qw": 0.70710678,
            },
        },
    }

    captured = bridge.digital_twin_capture_function_step(
        "ur5e only",
        "ur5e",
        "pick_approach",
        "prusa-mk4-2",
        "descend",
        "move_cartesian",
        part_name="MG",
    )
    saved = bridge.digital_twin_save_function_position(
        "ur5e only",
        "ur5e",
        "pick_approach",
        "prusa-mk4-2",
        "descend",
        part_name="MG",
    )
    recaptured = bridge.digital_twin_capture_function_step(
        "ur5e only",
        "ur5e",
        "pick_approach",
        "prusa-mk4-2",
        "descend",
        "move_cartesian",
        part_name="MG",
    )

    assert captured["success"] is True
    assert captured["step"]["position_sources"] == {
        "x": "captured_relative",
        "y": "captured_relative",
        "z": "captured_relative",
    }
    assert captured["step"]["relative_position_m"] == pytest.approx(
        {"x": 0.05, "y": 0.02, "z": 0.13456789}
    )
    assert saved["success"] is True
    assert recaptured["success"] is True
    path = tmp_path / "ur5e/pick_approach/prusa-mk4-2__MG__hardware.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["name"] == "prusa-mk4-2"
    assert payload["part_name"] == "MG"
    step = payload["steps"][0]
    assert step["position_sources"] == {
        "x": "captured_relative",
        "y": "captured_relative",
        "z": "captured_relative",
    }
    assert step["relative_position_m"] == pytest.approx(
        {"x": 0.05, "y": 0.02, "z": 0.13456789}
    )
    assert step["relative_reference"]["kind"] == "detected_part"
    assert step["waypoint"]["joint_positions"] == pytest.approx(
        [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    )
    assert step["waypoint"]["pose"]["z"] == pytest.approx(1.23456789)


def test_save_position_replaces_same_step_in_hardware_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(bridge_module, "_ROBOT_TAUGHT_FUNCTIONS_DIR", tmp_path)
    bridge = object.__new__(SystemBridge)
    bridge._digital_twin_record_lock = threading.Lock()
    key = bridge._robot_function_buffer_key(
        "ur5e only",
        "ur5e",
        "place_approach",
        "assembly_board-v1",
        "MG",
    )
    bridge._digital_twin_function_steps = {
        key: [
            {
                "step_name": "move_above_destination",
                "primitive": "move_cartesian",
                "params": {
                    "x": 0.5,
                    "y": 0.3,
                    "z": 1.2,
                    "qx": 0.0,
                    "qy": 0.0,
                    "qz": 0.0,
                    "qw": 1.0,
                },
                "waypoint": {
                    "pose": {"frame_id": "world", "x": 0.5},
                    "joint_names": list(UR5E_JOINT_NAMES),
                    "joint_positions": [1.0] * 6,
                    "source": "hardware",
                },
            }
        ]
    }
    bridge._robot_function_validate_request = lambda *_args: ({}, "")
    bridge._robot_function_storage_source = lambda *_args: "hardware"
    bridge._robot_function_capture_source = lambda *_args: "hardware"
    path = tmp_path / "ur5e/place_approach/assembly_board-v1__MG__hardware.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "name": "assembly_board-v1",
                "part_name": "MG",
                "steps": [
                    {
                        "step_name": "move_above_destination",
                        "primitive": "move_cartesian",
                        "params": {"x": 0.1},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = bridge.digital_twin_save_function_position(
        "ur5e only",
        "ur5e",
        "place_approach",
        "assembly_board-v1",
        "move_above_destination",
        part_name="MG",
    )

    assert result["success"] is True
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert len(payload["steps"]) == 1
    assert payload["steps"][0]["params"]["x"] == pytest.approx(0.5)
    assert payload["name"] == "assembly_board-v1"
    assert payload["part_name"] == "MG"
    assert bridge._digital_twin_function_steps == {}


def test_test_position_uses_runtime_validated_move_cartesian_and_requires_confirmation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cais_spade_llm.resources.robot import robot_task_runtime

    monkeypatch.setattr(robot_task_runtime, "_TAUGHT_FUNCTIONS_ROOT", tmp_path)
    called: list[dict[str, float]] = []
    control_status = {
        "rtde_receive_connected": True,
        "joint_states_fresh": True,
        "rtde_control_connected": False,
    }

    class _Controller:
        @staticmethod
        def move_cartesian(**params: float) -> dict[str, object]:
            called.append(params)
            return {"success": True, "message": "moved"}

    controller = _Controller()
    bridge = object.__new__(SystemBridge)
    bridge._ur5e_robot_function_execution_lock = threading.Lock()
    bridge._ur5e_robot_function_execution_active = None
    bridge._robot_function_validate_request = lambda *_args: ({"hardware": ("ur5e",)}, "")
    bridge._digital_twin_robot_function_target_error = lambda *_args: ""
    bridge._physical_robot_function_cartesian_error = lambda _robot: ""
    bridge._robot_function_capture_snapshot = lambda *_args: {"success": True}
    bridge._robot_function_relative_reference = lambda **_kwargs: {
        "success": True,
        "reference": {
            "kind": "destination_target",
            "frame_id": "world",
            "name": "assembly_board-v1",
            "position_m": {"x": 0.4, "y": 0.3, "z": 1.1},
            "pose": {
                "x": 0.4,
                "y": 0.3,
                "z": 1.1,
                "qx": 0.0,
                "qy": 0.0,
                "qz": 0.0,
                "qw": 1.0,
            },
            "source": "assembly_board-v1_aruco",
            "captured_at": time.time(),
            "camera_role": "ur5e",
            "generation": 1,
        },
        "computed": {
            "success": True,
            "part_name": "MG",
            "destination_location": "assembly_board-v1",
            "approach_pose": {"x": 0.4, "y": 0.3, "z": 1.2},
            "target_pose": {"x": 0.4, "y": 0.3, "z": 1.1},
        },
        "assembly_board_v1_aruco": {
            "destination_location": "assembly_board-v1",
            "camera_role": "ur5e",
            "generation": 1,
            "captured_at": time.time(),
            "sample_started_at": time.time() - 0.5,
            "frame_id": "world",
            "pose": {
                "x": 0.4,
                "y": 0.3,
                "z": 1.1,
                "qx": 0.0,
                "qy": 0.0,
                "qz": 0.0,
                "qw": 1.0,
            },
            "calibration_id": "ur5e-calibration",
        },
    }

    def _motion_readiness(
        _target: str,
        _cfg: dict[str, object],
        resource_agent: object | None = None,
    ) -> tuple[dict[str, object], str]:
        assert resource_agent is agent
        readiness = {
            "rtde_receive_connected": True,
            "joint_states_fresh": True,
            "rtde_control_connected": control_status["rtde_control_connected"],
            "trajectory_action_ready": control_status["rtde_control_connected"],
        }
        error = "" if control_status["rtde_control_connected"] else "Remote Control is not ready."
        return readiness, error

    bridge._digital_twin_ur5e_motion_readiness = _motion_readiness
    agent = SimpleNamespace(
        agent_name="ur5e",
        jid="ur5e@localhost",
        execution_mode="physical",
        _controller=controller,
        _robot_motion_lock=threading.Lock(),
        static_capabilities={
            "gripper_reach": {
                "frame": "world",
                "origin_pose": {"x": 0.0, "y": 0.0, "z": 0.0},
                "max_xy_radius_m": 2.0,
                "z_min_m": 0.0,
                "z_max_m": 2.0,
                "tolerance_m": 0.0,
            }
        },
        _is_pose_in_workspace=lambda _pose: (True, "ready"),
    )
    bridge.resource_agents = []
    bridge._ur5e_robot_function_agent = agent

    def _recorded_step(step_name: str, z: float) -> dict[str, object]:
        params = {
            "x": 0.4,
            "y": 0.3,
            "z": z,
            "qx": 0.0,
            "qy": 0.0,
            "qz": 0.0,
            "qw": 1.0,
        }
        return {
            "step_name": step_name,
            "primitive": "move_cartesian",
            "params": dict(params),
            "capture_source": "hardware",
            "position_sources": {
                "x": "captured_relative",
                "y": "captured_relative",
                "z": "captured_relative",
            },
            "relative_position_m": {
                "x": 0.0,
                "y": 0.0,
                "z": z - 1.1,
            },
            "relative_pose": {
                "x": 0.0,
                "y": 0.0,
                "z": z - 1.1,
                "qx": 0.0,
                "qy": 0.0,
                "qz": 0.0,
                "qw": 1.0,
            },
            "relative_reference": {
                "kind": "destination_target",
                "frame_id": "world",
                "name": "assembly_board-v1",
                "position_m": {"x": 0.4, "y": 0.3, "z": 1.1},
                "pose": {
                    "x": 0.4,
                    "y": 0.3,
                    "z": 1.1,
                    "qx": 0.0,
                    "qy": 0.0,
                    "qz": 0.0,
                    "qw": 1.0,
                },
                "source": "assembly_board-v1_aruco",
                "captured_at": time.time(),
                "camera_role": "ur5e",
                "generation": 1,
            },
            "waypoint": {
                "pose": {
                    "frame_id": "world",
                    "child_frame_id": "tool0",
                    **params,
                },
                "joint_names": list(UR5E_JOINT_NAMES),
                "joint_positions": [0.0] * 6,
                "source": "hardware",
            },
        }

    path = tmp_path / "ur5e/place_approach/assembly_board-v1__MG__hardware.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "robot": "ur5e",
                "function_name": "place_approach",
                "name": "assembly_board-v1",
                "part_name": "MG",
                "capture_source": "hardware",
                "steps": [
                    _recorded_step("move_above_destination", 1.2),
                    _recorded_step("descend", 1.1),
                ],
            }
        ),
        encoding="utf-8",
    )

    blocked = bridge.digital_twin_test_function_position(
        "ur5e only",
        "ur5e",
        "place_approach",
        "assembly_board-v1",
        "descend",
        part_name="MG",
    )
    remote_control_blocked = bridge.digital_twin_test_function_position(
        "ur5e only",
        "ur5e",
        "place_approach",
        "assembly_board-v1",
        "descend",
        confirmed=True,
        part_name="MG",
    )
    control_status["rtde_control_connected"] = True
    bridge._ur5e_robot_function_execution_lock.acquire()
    bridge._ur5e_robot_function_execution_active = "pick_approach"
    motion_busy = bridge.digital_twin_test_function_position(
        "ur5e only",
        "ur5e",
        "place_approach",
        "assembly_board-v1",
        "descend",
        confirmed=True,
        part_name="MG",
    )
    bridge._ur5e_robot_function_execution_lock.release()
    bridge._ur5e_robot_function_execution_active = None
    moved = bridge.digital_twin_test_function_position(
        "ur5e only",
        "ur5e",
        "place_approach",
        "assembly_board-v1",
        "descend",
        confirmed=True,
        part_name="MG",
    )
    invalid_payload = json.loads(path.read_text(encoding="utf-8"))
    invalid_payload["steps"].append(dict(invalid_payload["steps"][0]))
    path.write_text(json.dumps(invalid_payload), encoding="utf-8")
    invalid_recording = bridge.digital_twin_test_function_position(
        "ur5e only",
        "ur5e",
        "place_approach",
        "assembly_board-v1",
        "descend",
        confirmed=True,
        part_name="MG",
    )

    assert blocked["success"] is False
    assert remote_control_blocked["success"] is False
    assert "Remote Control" in remote_control_blocked["message"]
    assert motion_busy["success"] is False
    assert motion_busy["active_function"] == "pick_approach"
    assert invalid_recording["success"] is False
    assert "duplicate physical position step_name" in invalid_recording["message"]
    assert called == [
        {
            "x": 0.4,
            "y": 0.3,
            "z": 1.1,
            "qx": 0.0,
            "qy": 0.0,
            "qz": 0.0,
            "qw": 1.0,
        }
    ]
    assert moved == {"success": True, "message": "moved"}


def test_prepare_function_position_waits_for_cached_agent_lifecycle() -> None:
    async def _exercise() -> None:
        bridge = object.__new__(SystemBridge)
        bridge._digital_twin_target = lambda _target: {"hardware": ("ur5e",)}
        bridge._digital_twin_robot_function_target_error = lambda *_args: ""
        bridge._ur5e_robot_function_agent_lifecycle_lock = asyncio.Lock()
        agent = SimpleNamespace(
            execution_mode="physical",
            _controller=object(),
        )
        prepared: list[tuple[str, str]] = []

        async def _ensure_locked(target: str, robot: str) -> tuple[object, str]:
            prepared.append((target, robot))
            return agent, ""

        bridge._ensure_ur5e_robot_function_agent_locked = _ensure_locked
        await bridge._ur5e_robot_function_agent_lifecycle_lock.acquire()
        preparation = asyncio.create_task(
            bridge.digital_twin_prepare_function_position("ur5e only", "ur5e")
        )
        await asyncio.sleep(0)
        assert preparation.done() is False

        bridge._ur5e_robot_function_agent_lifecycle_lock.release()
        result = await preparation

        assert result["success"] is True
        assert "no motion was requested" in result["message"]
        assert prepared == [("ur5e only", "ur5e")]

    asyncio.run(_exercise())


def test_prepare_function_capture_is_read_only_and_skips_cartesian_motion_gate() -> None:
    async def _exercise() -> None:
        bridge = object.__new__(SystemBridge)
        bridge._digital_twin_target = lambda _target: {"hardware": ("ur5e",)}
        bridge._robot_function_capture_source = lambda *_args: "hardware"
        agent = SimpleNamespace(execution_mode="physical", _controller=object())
        bridge._ensure_ur5e_robot_function_agent = (
            lambda *_args: asyncio.sleep(0, result=(agent, ""))
        )
        bridge._physical_robot_function_cartesian_error = lambda _robot: (_ for _ in ()).throw(
            AssertionError("read-only Capture Pose must not require Cartesian motion")
        )

        result = await bridge.digital_twin_prepare_function_capture(
            "ur5e only",
            "ur5e",
        )

        assert result["success"] is True
        assert "no motion was requested" in result["message"]

    asyncio.run(_exercise())


def test_prepare_function_position_does_not_create_controller_before_stack_running() -> None:
    async def _exercise() -> None:
        bridge = object.__new__(SystemBridge)
        bridge._digital_twin_target = lambda _target: {"hardware": ("ur5e",)}
        bridge._digital_twin_robot_function_target_error = lambda *_args: (
            "ur5e Hardware Stack is starting; wait for final readiness."
        )

        async def _unexpected_ensure(*_args: object) -> tuple[object, str]:
            pytest.fail("a stopped or starting Hardware Stack must not create a controller")

        bridge._ensure_ur5e_robot_function_agent = _unexpected_ensure

        result = await bridge.digital_twin_prepare_function_position(
            "ur5e only",
            "ur5e",
        )

        assert result == {
            "success": False,
            "message": "ur5e Hardware Stack is starting; wait for final readiness.",
        }

    asyncio.run(_exercise())


def test_prepare_function_position_surfaces_preparation_and_controller_errors() -> None:
    async def _exercise() -> None:
        bridge = object.__new__(SystemBridge)
        bridge._digital_twin_target = lambda _target: {"hardware": ("ur5e",)}
        bridge._digital_twin_robot_function_target_error = lambda *_args: ""
        outcomes = [
            (None, "physical ur5e controller prewarm failed: action unavailable"),
            (SimpleNamespace(execution_mode="simulation", _controller=object()), ""),
            (SimpleNamespace(execution_mode="physical", _controller=None), ""),
        ]

        async def _ensure(_target: str, _robot: str) -> tuple[object | None, str]:
            return outcomes.pop(0)

        bridge._ensure_ur5e_robot_function_agent = _ensure

        prewarm = await bridge.digital_twin_prepare_function_position(
            "ur5e only",
            "ur5e",
        )
        nonphysical = await bridge.digital_twin_prepare_function_position(
            "ur5e only",
            "ur5e",
        )
        missing_controller = await bridge.digital_twin_prepare_function_position(
            "ur5e only",
            "ur5e",
        )

        assert prewarm == {
            "success": False,
            "message": "physical ur5e controller prewarm failed: action unavailable",
        }
        assert nonphysical["success"] is False
        assert "not in Physical mode" in nonphysical["message"]
        assert missing_controller == {
            "success": False,
            "message": "The physical ur5e controller is unavailable.",
        }

    asyncio.run(_exercise())


def test_mg_preview_resolves_live_xyz_with_captured_orientation_and_diagnostics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(robot_task_runtime, "_TAUGHT_FUNCTIONS_ROOT", tmp_path)
    path = tmp_path / "ur5e/pick_approach/prusa-mk4-2__MG__hardware.json"
    path.parent.mkdir(parents=True)
    captured_quaternion = {
        "qx": 0.6728028804150565,
        "qy": 0.7397684883486975,
        "qz": -0.005807808849261562,
        "qw": 0.0067184155763625976,
    }
    captured_pose = {
        "frame_id": "world",
        "child_frame_id": "tool0",
        "x": 9.0,
        "y": 8.0,
        "z": 1.2657060858127773,
        **captured_quaternion,
    }
    path.write_text(
        json.dumps(
            {
                "robot": "ur5e",
                "function_name": "pick_approach",
                "name": "prusa-mk4-2",
                "part_name": "MG",
                "capture_source": "hardware",
                "steps": [
                    {
                        "step_name": "descend",
                        "primitive": "move_cartesian",
                        "params": captured_pose,
                        "capture_source": "hardware",
                        "position_sources": {
                            "x": "captured_relative",
                            "y": "captured_relative",
                            "z": "captured_relative",
                        },
                        "relative_position_m": {"x": 0.0, "y": 0.0, "z": 0.1},
                        "relative_reference": {
                            "kind": "detected_part",
                            "frame_id": "world",
                            "name": "MG",
                            "position_m": {"x": 0.1, "y": 0.2, "z": 0.3},
                            "source": "live_detection",
                            "captured_at": time.time(),
                        },
                        "waypoint": {
                            "pose": captured_pose,
                            "joint_names": list(UR5E_JOINT_NAMES),
                            "joint_positions": [0.0] * 6,
                            "source": "hardware",
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    resource_agent = SimpleNamespace(
        agent_name="ur5e",
        execution_mode="physical",
        _controller=object(),
        static_capabilities={
            "gripper_reach": {
                "frame": "world",
                "origin_pose": {"x": 0.0, "y": 0.0, "z": 0.0},
                "max_xy_radius_m": 2.0,
                "z_min_m": 0.0,
                "z_max_m": 2.0,
                "tolerance_m": 0.0,
            }
        },
        _is_pose_in_workspace=lambda _pose: (True, "ready"),
    )
    bridge = object.__new__(SystemBridge)
    bridge._robot_function_validate_request = lambda *_args: ({}, "")
    bridge._physical_ur5e_robot_agent = lambda: resource_agent
    bridge._physical_robot_function_cartesian_error = lambda _robot: ""
    bridge._robot_function_capture_snapshot = lambda *_args: {"success": True}
    bridge._robot_function_relative_reference = lambda **_kwargs: {
        "success": True,
        "reference": {
            "kind": "detected_part",
            "frame_id": "world",
            "name": "MG",
            "position_m": {"x": 0.1, "y": 0.2, "z": 0.3},
            "source": "live_detection",
            "captured_at": time.time(),
        },
        "computed": {
            "part_name": "MG",
            "tx": 0.1,
            "ty": 0.2,
            "tz": 0.3,
            "frame_id": "world",
            "captured_at": time.time(),
            "approach_pose": {"x": 0.1, "y": 0.2, "z": 0.8},
            "target_pose": {"x": 0.1, "y": 0.2, "z": 0.4},
            "table_surface_z_m": 0.20435,
            "tcp_offset_z": -0.17,
            "pick_tool0_z_adjustment_m": 0.005,
            "tooth_height_m": 0.01,
            "part_height": 0.02,
            "tooth_clearance_m": 0.002,
            "minimum_hub_overlap_m": 0.006,
            "open_inner_pad_lower_z_from_tcp_m": 0.01751,
            "closed_inner_pad_lower_z_from_tcp_m": -0.00865,
            "closed_inner_pad_upper_z_from_tcp_m": 0.0211,
        },
    }

    result = bridge.digital_twin_preview_function_position(
        "ur5e only",
        "ur5e",
        "pick_approach",
        "prusa-mk4-2",
        "descend",
        part_name="MG",
    )

    assert result["success"] is True, result
    assert result["resolved_position"] == pytest.approx(
        {"x": 0.1, "y": 0.2, "z": 0.4, **captured_quaternion}
    )
    assert result["diagnostics"]["finger_tooth_clearance_m"] == pytest.approx(
        0.002
    )
    assert result["diagnostics"]["finger_hub_overlap_m"] == pytest.approx(0.008)


def test_physical_function_position_agent_prefers_running_agent() -> None:
    bridge = object.__new__(SystemBridge)
    cached = SimpleNamespace(execution_mode="physical")
    running = SimpleNamespace(
        agent_name="ur5e",
        jid="ur5e@localhost",
        execution_mode="physical",
    )
    bridge._ur5e_robot_function_agent = cached
    bridge.resource_agents = [running]

    assert bridge._physical_ur5e_robot_agent() is running


def test_position_file_operations_reject_empty_pick_place_location() -> None:
    bridge = object.__new__(SystemBridge)
    bridge._robot_function_validate_request = lambda *_args: ({}, "")

    save = bridge.digital_twin_save_function_position(
        "ur5e only",
        "ur5e",
        "pick_approach",
        "",
        "move_above_part",
    )
    clear = bridge.digital_twin_clear_function_position(
        "ur5e only",
        "ur5e",
        "place_approach",
        "",
        "descend",
        part_name="MG",
    )
    test = bridge.digital_twin_test_function_position(
        "ur5e only",
        "ur5e",
        "pick_approach",
        "",
        "descend",
        confirmed=True,
    )

    assert save == {"success": False, "message": "origin_resource_location is empty"}
    assert clear == {"success": False, "message": "destination_location is empty"}
    assert test == {"success": False, "message": "origin_resource_location is empty"}


def test_predefined_ui_has_no_free_form_function_or_step_inputs() -> None:
    source = inspect.getsource(control._predefined_function_record_body)

    assert "digital_twin_function_names" in source
    assert "digital_twin_function_template" in source
    assert "function name" not in source
    assert "step name" not in source
    assert "step_1" not in source
    assert "Capture Pose" in source
    assert "Save/Replace Pose" in source
    assert "Position Recording" in source
    assert "recording_container.set_visibility(bool(recordable_steps))" in source
    assert "Preview Resolved Pose" not in source
    assert "Apply Axis Sources" not in source
    assert "Computed Z offset (mm)" not in source
    assert "Apply and Save Z Offset" not in source
    assert "Capture Orientation" not in source
    assert "Saved calibration XYZ offset" in source
    assert "Recapture required" not in source
    assert "This legacy position has no captured_relative XYZ" not in source
    assert 'label="origin_resource_location"' in source
    assert 'label="destination_location"' in source
    assert 'label="part_name"' in source
    assert "digital_twin_robot_function_execution_readiness" in source
    assert "digital_twin_prepare_function_position" in source
    assert "keeps the captured quaternion unchanged" in source
    assert "_prepare_function_execution_runtime" not in source
    assert "digital_twin_execute_robot_function" in source
    assert 'or execution.get("preparing")' in source
    assert "Preview Target" not in source
    assert "Check Capture Readiness" not in source


def test_place_recording_requires_exact_part_name_and_validates_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(bridge_module, "_ROBOT_TAUGHT_FUNCTIONS_DIR", tmp_path)
    bridge = object.__new__(SystemBridge)
    bridge._digital_twin_target = lambda _target: {"hardware": ("ur5e",)}
    bridge._robot_function_storage_source = lambda *_args: "hardware"

    missing = bridge.digital_twin_function_info(
        "ur5e only",
        "ur5e",
        "place_approach",
        "assembly_board-v1",
    )
    assert missing == {"success": False, "message": "part_name is empty"}

    path = tmp_path / "ur5e/place_approach/assembly_board-v1__MG__hardware.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "robot": "ur5e",
                "function_name": "place_approach",
                "name": "assembly_board-v1",
                "part_name": "SG",
                "steps": [],
            }
        ),
        encoding="utf-8",
    )

    payload, loaded_path, error = bridge._robot_function_file_payload(
        "ur5e only",
        "ur5e",
        "place_approach",
        "assembly_board-v1",
        part_name="MG",
    )

    assert payload is None
    assert loaded_path == path
    assert error == "part_name mismatch in taught function file"


def test_place_recording_paths_and_buffers_are_isolated_by_exact_part_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(bridge_module, "_ROBOT_TAUGHT_FUNCTIONS_DIR", tmp_path)
    bridge = object.__new__(SystemBridge)
    bridge._digital_twin_record_lock = threading.Lock()
    mg_key = bridge._robot_function_buffer_key(
        "ur5e only",
        "ur5e",
        "place_approach",
        "assembly_board-v1",
        "MG",
    )
    sg_key = bridge._robot_function_buffer_key(
        "ur5e only",
        "ur5e",
        "place_approach",
        "assembly_board-v1",
        "SG",
    )
    bridge._digital_twin_function_steps = {
        mg_key: [{"step_name": "move_above_destination", "primitive": "move_cartesian"}],
        sg_key: [{"step_name": "descend", "primitive": "move_cartesian"}],
    }

    assert mg_key != sg_key
    assert (
        bridge._robot_function_path(
            "ur5e",
            "place_approach",
            "assembly_board-v1",
            "hardware",
            "MG",
        ).name
        == "assembly_board-v1__MG__hardware.json"
    )
    assert (
        bridge._robot_function_path(
            "ur5e",
            "place_approach",
            "assembly_board-v1",
            "hardware",
            "SG",
        ).name
        == "assembly_board-v1__SG__hardware.json"
    )
    assert [
        step["step_name"]
        for step in bridge.digital_twin_list_function_buffer_steps(
            "ur5e only",
            "ur5e",
            "place_approach",
            "assembly_board-v1",
            part_name="MG",
        )
    ] == ["move_above_destination"]
    assert [
        step["step_name"]
        for step in bridge.digital_twin_list_function_buffer_steps(
            "ur5e only",
            "ur5e",
            "place_approach",
            "assembly_board-v1",
            part_name="SG",
        )
    ] == ["descend"]


def test_place_delete_and_replay_apis_thread_exact_part_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(bridge_module, "_ROBOT_TAUGHT_FUNCTIONS_DIR", tmp_path)
    bridge = object.__new__(SystemBridge)
    bridge._digital_twin_target = lambda _target: {"hardware": ("ur5e",)}
    bridge._robot_function_storage_source = lambda *_args: "hardware"
    path = tmp_path / "ur5e/place_approach/assembly_board-v1__MG__hardware.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "robot": "ur5e",
                "function_name": "place_approach",
                "name": "assembly_board-v1",
                "part_name": "MG",
                "steps": [],
            }
        ),
        encoding="utf-8",
    )

    deleted = bridge.digital_twin_delete_function(
        "ur5e only",
        "ur5e",
        "place_approach",
        "assembly_board-v1",
        part_name="MG",
    )
    assert deleted["success"] is True
    assert not path.exists()

    seen: list[tuple[str, str]] = []

    def _file_payload(
        _target: str,
        _robot: str,
        function_name: str,
        _name: str,
        *,
        part_name: str = "",
    ) -> tuple[None, None, str]:
        seen.append((function_name, part_name))
        return None, None, "expected test stop"

    bridge._robot_function_file_payload = _file_payload
    replay = bridge.digital_twin_replay_function(
        "ur5e only",
        "ur5e",
        "place_approach",
        "assembly_board-v1",
        part_name="MG",
    )
    replay_step = bridge.digital_twin_replay_function_step(
        "ur5e only",
        "ur5e",
        "place_approach",
        "assembly_board-v1",
        0,
        part_name="MG",
    )

    assert replay == {"success": False, "message": "expected test stop"}
    assert replay_step == {"success": False, "message": "expected test stop"}
    assert seen == [("place_approach", "MG"), ("place_approach", "MG")]


def test_preview_pick_target_is_read_only_and_uses_only_selected_detection() -> None:
    compute_calls: list[dict[str, object]] = []

    class _Controller:
        _last_start_pose = "before preview"

        @staticmethod
        def compute_pick_targets(**params: object) -> dict[str, object]:
            compute_calls.append(params)
            _Controller._last_start_pose = "changed by compute_pick_targets"
            return {
                "success": True,
                "travel_z": 1.31,
                "pick_z": 1.12,
            }

    controller = _Controller()
    bridge = object.__new__(SystemBridge)
    bridge.resource_agents = [
        SimpleNamespace(
            agent_name="ur5e",
            jid="ur5e@localhost",
            execution_mode="physical",
            _controller=controller,
        )
    ]
    bridge._robot_function_validate_request = lambda *_args: ({}, "")
    bridge._robot_function_capture_source = lambda *_args: "hardware"
    bridge._robot_function_capture_snapshot = lambda *_args: {
        "success": True,
        "rtde_receive_connected": True,
        "joint_states_fresh": True,
        "world_tool0_ready": True,
        "rtde_control_connected": False,
        "blocked_reason": "",
    }
    bridge._robot_function_product_geometry_for_part = lambda part_name: {
        "part_height_m": 0.02,
        "model_name": "gear_medium",
        "part_name": part_name,
    }
    captured_at = time.time() - 0.2
    bridge.test_physical_detection = lambda _target=None: {
        "success": True,
        "message": "Detection validated; no robot motion was requested.",
        "detections": [
            {
                "part_name": "SG",
                "frame_id": "world",
                "x": 0.1,
                "y": 0.2,
                "z": 1.0,
                "confidence": 0.91,
                "captured_at": captured_at,
                "table_surface_z_m": 1.0,
            },
            {
                "part_name": "MG",
                "frame_id": "world",
                "x": 0.4,
                "y": 0.3,
                "z": 1.02,
                "confidence": 0.96,
                "captured_at": captured_at,
                "table_surface_z_m": 1.0,
            },
        ],
        "status": {
            "updated_at": captured_at,
            "calibration_ready": True,
            "table_plane_ready": True,
        },
    }

    result = bridge.digital_twin_preview_pick_target("ur5e only", "ur5e", "MG")

    assert result["success"] is True
    assert result["ready"] is True
    assert result["part_name"] == "MG"
    assert result["confidence"] == pytest.approx(0.96)
    assert result["world_pose"] == {
        "frame_id": "world",
        "x": 0.4,
        "y": 0.3,
        "z": 1.02,
    }
    assert result["travel_z"] == pytest.approx(1.31)
    assert result["pick_z"] == pytest.approx(1.12)
    assert result["rtde_control_required"] is False
    assert result["readiness"]["rtde_receive_connected"] is True
    assert result["readiness"]["joint_states_fresh"] is True
    assert result["readiness"]["world_tool0_ready"] is True
    assert result["readiness"]["rtde_control_required"] is False
    assert controller._last_start_pose == "before preview"
    assert len(compute_calls) == 1
    assert compute_calls[0]["part_name"] == "MG"
    assert compute_calls[0]["detected_parts"] == [
        {
            "part_name": "MG",
            "frame_id": "world",
            "x": 0.4,
            "y": 0.3,
            "z": 1.02,
            "confidence": 0.96,
            "captured_at": captured_at,
            "table_surface_z_m": 1.0,
        }
    ]


def test_preview_pick_target_stops_before_detection_when_robot_state_is_not_ready() -> None:
    bridge = object.__new__(SystemBridge)
    bridge._robot_function_validate_request = lambda *_args: ({}, "")
    bridge._robot_function_capture_source = lambda *_args: "hardware"
    bridge._robot_function_capture_snapshot = lambda *_args: {
        "success": False,
        "rtde_receive_connected": True,
        "joint_states_fresh": False,
        "world_tool0_ready": False,
        "rtde_control_connected": False,
        "blocked_reason": "UR5e joint feedback is not fresh.",
    }
    bridge.test_physical_detection = lambda *_args: pytest.fail(
        "detection must not run before read-only robot readiness"
    )

    result = bridge.digital_twin_preview_pick_target("ur5e only", "ur5e", "MG")

    assert result["success"] is False
    assert result["ready"] is False
    assert result["blocked_reason"] == "UR5e joint feedback is not fresh."
    assert result["readiness"]["rtde_receive_connected"] is True
    assert result["readiness"]["joint_states_fresh"] is False
    assert result["readiness"]["world_tool0_ready"] is False
    assert result["readiness"]["rtde_control_required"] is False


def test_snapshot_world_tool_pose_is_optional_and_backward_compatible() -> None:
    script = (
        Path(__file__).resolve().parents[1] / "ros2/cais_lab_robotics/scripts/digital_twin_sync.py"
    ).read_text(encoding="utf-8")

    assert 'parser.add_argument("--include-world-tool-pose", action="store_true")' in script
    assert 'self.pose_child_frame = "link_eef" if robot == "xarm6" else "tool0"' in script
    assert 'self.tf_buffer.lookup_transform(\n                            "world",' in script
    assert "self.pose_child_frame," in script
    assert "self.world_base_child_frame," in script
    assert "include_world_tool_pose: bool = False" in script
