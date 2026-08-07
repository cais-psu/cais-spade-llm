"""Focused tests for predefined physical robot-function position capture."""

from __future__ import annotations

import inspect
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from cais_spade_llm.ui import bridge as bridge_module
from cais_spade_llm.ui.bridge import SystemBridge
from cais_spade_llm.ui.pages import control


def test_function_templates_come_from_exact_robot_task_steps() -> None:
    expected = {
        "pick_approach": [
            ("move_to_origin_resource_location", "move_to_named_pose", False),
            ("detect_parts", "detect_parts", False),
            ("compute_pick_targets", "compute_pick_targets", False),
            ("open_gripper", "open_gripper", False),
            ("move_above_part", "move_cartesian", False),
            ("descend", "move_cartesian", False),
        ],
        "pick_grasp": [
            ("grasp_part", "grasp_part", False),
            ("delay_after_grasp", "delay", False),
            ("lift", "move_relative", False),
        ],
        "place_approach": [
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
    } == {"Computed live by compute_pick_targets."}


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
            "joint_names": [f"joint_{index}" for index in range(6)],
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
    capture_index = {"value": 0}

    def _capture_snapshot(*_args: object) -> dict[str, object]:
        capture_index["value"] += 1
        x = 0.4 + capture_index["value"] / 100.0
        return {
            "success": True,
            "blocked_reason": "",
            "waypoint": {
                "joint_names": [f"joint_{index}" for index in range(6)],
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
                    "joint_names": [f"joint_{index}" for index in range(6)],
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

    def _motion_readiness(
        _target: str,
        _cfg: dict[str, object],
    ) -> tuple[dict[str, object], str]:
        readiness = {
            "rtde_receive_connected": True,
            "joint_states_fresh": True,
            "rtde_control_connected": control_status["rtde_control_connected"],
            "trajectory_action_ready": control_status["rtde_control_connected"],
        }
        error = "" if control_status["rtde_control_connected"] else "Remote Control is not ready."
        return readiness, error

    bridge._digital_twin_ur5e_motion_readiness = _motion_readiness
    bridge.resource_agents = [
        SimpleNamespace(
            agent_name="ur5e",
            jid="ur5e@localhost",
            execution_mode="physical",
            _controller=controller,
            _robot_motion_lock=threading.Lock(),
        )
    ]

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
            "waypoint": {
                "pose": {
                    "frame_id": "world",
                    "child_frame_id": "tool0",
                    **params,
                },
                "joint_names": [f"joint_{index}" for index in range(6)],
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
    assert "Capture Position" in source
    assert "Save/Replace Position" in source
    assert "Position Recording" in source
    assert "recording_container.set_visibility(bool(required))" in source
    assert 'label="origin_resource_location"' in source
    assert 'label="destination_location"' in source
    assert 'label="part_name"' in source
    assert "digital_twin_robot_function_execution_readiness" in source
    assert "digital_twin_execute_robot_function" in source
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
    assert (
        'lookup_transform(\n                        "world",\n                        "tool0"'
        in script
    )
    assert "include_world_tool_pose: bool = False" in script
