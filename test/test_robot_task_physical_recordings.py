"""Focused tests for registry-backed physical Cartesian position recordings."""

from __future__ import annotations

import asyncio
import hashlib
import json
import threading
import time
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from cais_spade_llm.resources.robot import robot_task_runtime
from cais_spade_llm.resources.robot.robot_tasks import (
    execute_robot_task,
    robot_task_names,
    robot_task_registry,
)

_MOVE_INSERT_MODEL_MAP = {
    "SG": "gear_small",
    "MG": "gear_medium",
    "LG": "gear_large",
    "SCP": "circ_pin_small",
    "MCP": "circ_pin_medium",
    "LCP": "circ_pin_large",
}


def _move_insert_hard_caps_sha256(hard_caps: dict[str, Any]) -> str:
    canonical_hard_caps = json.dumps(
        {
            name: float(value)
            for name, value in sorted(hard_caps.items())
        },
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(canonical_hard_caps.encode("utf-8")).hexdigest()


def _force_depth_profile() -> dict[str, list[float]]:
    normalized_depth = [index / 15.0 for index in range(16)]
    return {
        "normalized_depth": normalized_depth,
        "axial_force_upper_n": [20.0] * 16,
        "lateral_force_upper_n": [10.0] * 16,
        "active_tcp_torque_upper_nm": [2.0] * 16,
    }


def test_move_insert_result_sanitizer_preserves_planned_backoff_evidence() -> None:
    sanitized = robot_task_runtime._sanitized_move_insert_result(
        {
            "relief_backoff_m": 0.0002,
            "relief_planned_backoff_m": 0.0005,
            "total_relief_backoff_m": 0.0012,
        }
    )

    assert sanitized == {
        "relief_backoff_m": 0.0002,
        "relief_planned_backoff_m": 0.0005,
        "total_relief_backoff_m": 0.0012,
    }


def test_move_insert_result_sanitizer_does_not_duplicate_complete_server_trace() -> None:
    sanitized = robot_task_runtime._sanitized_move_insert_result(
        {
            "trial_id": "move-insert-test",
            "server_trace_id": "move-insert-test",
            "server_trace_path": (
                "/tmp/cais_ur5e_insert_trials/move-insert-test/trace.jsonl"
            ),
            "server_trace_sha256": "a" * 64,
            "server_trace_status": "complete",
            "server_trace_complete": True,
            "server_trace_sample_count": 2_304,
            "disengagement_cycle_count": 1,
            "last_disengagement_reason": (
                "MG disengagement timed out after 0.300 s before contact cleared"
            ),
            "disengagement_withdrawal_m": 0.000047,
            "disengagement_contact_cleared": False,
            "disengagement_stationary_confirmed": True,
            "feedback_trace": [
                {
                    "timestamp": 1.0,
                    "phase": "seating",
                    "insertion_depth_m": 0.018,
                },
                {
                    "timestamp": 2.0,
                    "phase": "disengaging",
                    "insertion_depth_m": 0.016,
                },
            ],
        }
    )

    assert sanitized["server_trace_id"] == "move-insert-test"
    assert sanitized["server_trace_status"] == "complete"
    assert sanitized["server_trace_complete"] is True
    assert sanitized["server_trace_sample_count"] == 2_304
    assert sanitized["disengagement_cycle_count"] == 1
    assert sanitized["disengagement_contact_cleared"] is False
    assert sanitized["disengagement_stationary_confirmed"] is True
    assert sanitized["max_insertion_depth_m"] == pytest.approx(0.018)
    assert sanitized["feedback_trace"] == []


def test_move_insert_result_sanitizer_keeps_trace_when_metadata_is_not_complete() -> None:
    sanitized = robot_task_runtime._sanitized_move_insert_result(
        {
            "server_trace_id": "move-insert-test",
            "server_trace_path": (
                "/tmp/cais_ur5e_insert_trials/move-insert-test/trace.jsonl"
            ),
            "server_trace_sha256": "a" * 64,
            "server_trace_status": "failed",
            "server_trace_complete": False,
            "server_trace_sample_count": 2,
            "feedback_trace": [
                {"timestamp": 1.0, "insertion_depth_m": 0.018},
                {"timestamp": 2.0, "insertion_depth_m": 0.016},
            ],
        }
    )

    assert sanitized["server_trace_status"] == "failed"
    assert sanitized["server_trace_complete"] is False
    assert sanitized["max_insertion_depth_m"] == pytest.approx(0.018)
    assert len(sanitized["feedback_trace"]) == 2


def test_robot_agent_rejects_overlapping_registered_robot_tasks() -> None:
    from cais_spade_llm.agents.resource_agent.robot_agent import RobotAgent

    agent = object.__new__(RobotAgent)
    agent.agent_name = "ur5e"
    agent._robot_motion_lock = threading.Lock()
    agent._robot_motion_lock.acquire()
    try:
        result = asyncio.run(RobotAgent._execute_registered_robot_task(agent, "move_home"))
    finally:
        agent._robot_motion_lock.release()

    assert result == {
        "status": "blocked",
        "content": "ur5e is already executing a robot task.",
    }


def test_robot_agent_supervised_pre_execute_block_reports_known_no_motion() -> None:
    from cais_spade_llm.agents.resource_agent.robot_agent import RobotAgent

    agent = object.__new__(RobotAgent)
    agent.agent_name = "ur5e"
    agent._robot_motion_lock = threading.Lock()

    result = asyncio.run(
        RobotAgent._execute_place_insert_move_insert_trial(
            agent,
            lambda: "move_insert identity changed before dispatch",
            destination_location="assembly_board-v1",
            part_name="MG",
            trial_id="move-insert-test",
        )
    )

    assert result == {
        "status": "blocked",
        "content": "move_insert identity changed before dispatch",
        "manual_pre_execute_blocked": True,
        "trial_id": "move-insert-test",
        "motion_settled": True,
        "dispatch_attempted": False,
        "move_insert_result": {
            "success": False,
            "trial_id": "move-insert-test",
            "state_uncertain": False,
            "motion_settled": True,
            "dispatch_attempted": False,
        },
    }
    assert agent._robot_motion_lock.acquire(blocking=False)
    agent._robot_motion_lock.release()


def test_robot_agent_passes_operator_confirmation_outside_task_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cais_spade_llm.agents.resource_agent import robot_agent as robot_agent_module
    from cais_spade_llm.agents.resource_agent.robot_agent import RobotAgent

    agent = object.__new__(RobotAgent)
    agent.agent_name = "ur5e"
    agent._robot_motion_lock = threading.Lock()
    agent._controller = None
    observed: dict[str, Any] = {}

    async def _execute(
        actual_agent: Any,
        task_name: str,
        authority: object | None = None,
        operator_confirmed_held_part: bool = False,
        operator_confirmed_held_part_handoff: dict[str, Any] | None = None,
        /,
        **kwargs: Any,
    ) -> dict[str, Any]:
        observed.update(
            {
                "agent": actual_agent,
                "task_name": task_name,
                "authority": authority,
                "operator_confirmed_held_part": operator_confirmed_held_part,
                "operator_confirmed_held_part_handoff": (
                    operator_confirmed_held_part_handoff
                ),
                "kwargs": kwargs,
            }
        )
        return {"status": "completed"}

    monkeypatch.setattr(robot_agent_module, "execute_robot_task", _execute)
    handoff = {"source": "operator_confirmed_pick_approach_recording"}

    result = asyncio.run(
        agent._execute_registered_robot_task_for_manual_function_execution(
            "place_approach",
            operator_confirmed_held_part=True,
            operator_confirmed_held_part_handoff=handoff,
            destination_location="assembly_board-v1",
            part_name="MG",
        )
    )

    assert result == {"status": "completed"}
    assert observed == {
        "agent": agent,
        "task_name": "place_approach",
        "authority": robot_task_runtime._MANUAL_FUNCTION_EXECUTION_AUTHORITY,
        "operator_confirmed_held_part": True,
        "operator_confirmed_held_part_handoff": handoff,
        "kwargs": {
            "destination_location": "assembly_board-v1",
            "part_name": "MG",
        },
    }
    assert agent._robot_motion_lock.locked() is False


class _Logger:
    def warning(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def error(self, *_args: Any, **_kwargs: Any) -> None:
        return None


@pytest.mark.parametrize(
    ("part_name", "model_name"),
    list(_MOVE_INSERT_MODEL_MAP.items()),
)
def test_physical_ur5e_release_event_keeps_exact_part_model_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    part_name: str,
    model_name: str,
) -> None:
    from cais_spade_llm.agents.resource_agent import robot_agent as robot_agent_module

    agent = SimpleNamespace(
        execution_mode="physical",
        name="ur5e",
        _held_part=part_name,
        logger=_Logger(),
    )
    event_path = tmp_path / "cais_physical_part_ownership.json"
    monkeypatch.setattr(robot_agent_module, "Path", lambda _path: event_path)

    robot_agent_module.RobotAgent._emit_physical_part_ownership_event(
        agent,
        "release_part",
        {"part_name": part_name},
    )

    event = json.loads(event_path.read_text(encoding="utf-8"))
    assert str(event.pop("sequence")).isdigit()
    assert float(event.pop("occurred_at")) > 0.0
    assert event == {
        "action": "released",
        "part_name": part_name,
        "model_name": model_name,
        "source": "hardware_ur5e",
    }


class _Agent:
    def __init__(self, *, execution_mode: str, robot: str = "ur5e") -> None:
        self.execution_mode = execution_mode
        self.agent_name = robot
        self.logger = _Logger()
        self._held_part: str | None = None
        self._current_state = "idle"
        self._position = {"x": 0.0, "y": 0.0, "z": 0.0}
        self._gripper_state = "open"
        self._recovery_pose_ref: str | None = None
        self._task_ctx: dict[str, Any] = {}
        self._controller = self
        child_frame_id = "link_eef" if robot == "xarm6" else "tool0"
        tcp_link = "link_tcp" if robot == "xarm6" else "ur5e_rg2_gripper_tcp"
        joint_names = (
            [f"joint{index}" for index in range(1, 7)]
            if robot == "xarm6"
            else [
                "shoulder_pan_joint",
                "shoulder_lift_joint",
                "elbow_joint",
                "wrist_1_joint",
                "wrist_2_joint",
                "wrist_3_joint",
            ]
        )
        self.controller_config = {
            "arm_joint_names": joint_names,
            "move_group": {
                "frame_id": "world",
                "ee_link": child_frame_id,
                "tcp_link": tcp_link,
            },
        }
        self.named_positions = {
            "home": [0.0] * 6,
            "prusa-mk4-1": [0.0] * 6,
            "prusa-mk4-2": [0.0] * 6,
            "assembly_board-v1": [0.0] * 6,
        }
        self.primitive_calls: list[tuple[str, dict[str, Any]]] = []
        self.failures: list[dict[str, Any]] = []
        self.assembly_board_v1_aruco_pose = {
            "x": 0.5,
            "y": 0.6,
            "z": 0.3,
            "qx": 0.0,
            "qy": 0.0,
            "qz": 0.0,
            "qw": 1.0,
        }
        self.assembly_board_v1_aruco_generation = 7
        self.accepted_assembly_board_v1_aruco_generation = 7
        self.assembly_board_v1_aruco_calibration_id = f"{robot}-calibration"
        self.accepted_assembly_board_v1_aruco_calibration_id = (
            self.assembly_board_v1_aruco_calibration_id
        )
        self.static_capabilities = {
            "workspace_bounds": {
                "x_min_m": -10.0,
                "x_max_m": 10.0,
                "y_min_m": -10.0,
                "y_max_m": 10.0,
                "z_min_m": -10.0,
                "z_max_m": 10.0,
            },
            "gripper_reach": {
                "frame": "world",
                "origin_pose": {"x": 0.0, "y": 0.0, "z": 0.0},
                "max_xy_radius_m": 20.0,
                "z_min_m": -10.0,
                "z_max_m": 10.0,
                "tolerance_m": 0.0,
            },
        }

    def _robot_scope_name(self) -> str:
        return self.agent_name

    async def _execute_primitive(
        self,
        primitive: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        self.primitive_calls.append((primitive, deepcopy(params)))
        if primitive == "detect_parts":
            return {
                "success": True,
                "data": [
                    {
                        "part_name": str(params.get("part_name") or "MG"),
                        "model_name": "gear_medium",
                        "x": 0.1,
                        "y": 0.2,
                        "z": 0.3,
                        "frame_id": "world",
                        "captured_at": time.time(),
                    }
                ],
            }
        if primitive == "compute_pick_targets":
            detected_parts = list(params.get("detected_parts") or [])
            target = dict(detected_parts[0]) if detected_parts else {}
            return {
                "success": True,
                "part_name": str(target.get("part_name") or params.get("part_name") or ""),
                "model_name": str(target.get("model_name") or "gear_medium"),
                "tx": float(target.get("x", 0.1)),
                "ty": float(target.get("y", 0.2)),
                "tz": float(target.get("z", 0.3)),
                "pick_z": 0.4,
                "travel_z": 0.8,
                "part_height": 0.08,
                "tcp_offset_z": -0.17,
                "pick_tcp_z": 0.4,
                "source_stl": "/actual/Gear_Medium.STL",
                "source_stl_sha256": "a" * 64,
                "hub_up": True,
                "hub_diameter_m": 0.03,
                "hub_height_m": 0.01,
                "tooth_diameter_m": 0.042,
                "tooth_height_m": 0.01,
                "grasp_width_m": 0.028,
                "tooth_clearance_m": 0.002,
                "minimum_hub_overlap_m": 0.006,
                "finger_tooth_clearance_m": 0.002,
                "finger_hub_overlap_m": 0.008,
                "pick_z_adjustment_m": 0.001,
                "pick_tool0_z_adjustment_m": 0.005,
                "open_gripper_position": 0.11,
                "mg_gripper_close_position": 0.047,
                "open_inner_pad_lower_z_from_tcp_m": 0.01751,
                "open_inner_pad_upper_z_from_tcp_m": 0.04726,
                "closed_inner_pad_lower_z_from_tcp_m": -0.00865,
                "closed_inner_pad_upper_z_from_tcp_m": 0.0211,
                "predicted_closing_z_displacement_m": -0.02616,
                "gripper_close_position": 0.37,
                "start_x": 0.0,
                "start_y": 0.0,
                "start_z": 0.0,
                "frame_id": str(target.get("frame_id") or "world"),
                "captured_at": float(target.get("captured_at") or time.time()),
            }
        if primitive == "compute_place_targets":
            return {
                "success": True,
                "slot_x": 0.5,
                "slot_y": 0.6,
                "board_top_z": 0.2,
                "place_z": 0.3,
                "part_height": 0.08,
                "destination_location": str(params.get("destination_location") or ""),
            }
        if primitive == "localize_assembly_board_v1":
            return {
                "success": True,
                "destination_location": str(params.get("destination_location") or ""),
                "camera_role": self.agent_name,
                "generation": self.assembly_board_v1_aruco_generation,
                "calibration_id": self.assembly_board_v1_aruco_calibration_id,
                "captured_at": time.time(),
                "frame_id": "world",
                "pose": deepcopy(self.assembly_board_v1_aruco_pose),
            }
        if primitive == "move_cartesian":
            return {
                "success": True,
                "absolute_position": {
                    "x": float(params["x"]),
                    "y": float(params["y"]),
                    "z": float(params["z"]),
                    "qx": float(params.get("qx", 0.0)),
                    "qy": float(params.get("qy", 0.0)),
                    "qz": float(params.get("qz", 0.0)),
                    "qw": float(params.get("qw", 1.0)),
                },
            }
        if primitive == "move_insert":
            return {
                "success": True,
                "absolute_position": deepcopy(params["target_pose"]),
                "final_tool0_pose_valid": True,
                "engagement_detected": True,
                "seated_detected": True,
                "profile_sha256": params.get("profile_sha256"),
                "hard_caps_sha256": params.get("hard_caps_sha256"),
                "trial_id": params.get("trial_id", ""),
                "state_uncertain": False,
                "motion_settled": True,
                "feedback_trace": [
                    {
                        "timestamp": 1.0,
                        "phase": "settling",
                        "insertion_depth_m": 0.01,
                        "actual_tcp_force": [0.0, 0.0, 5.0, 0.0, 0.0, 0.1],
                        "actual_tool0_velocity": {
                            "vx": 0.0,
                            "vy": 0.0,
                            "vz": 0.0,
                            "wx": 0.0,
                            "wy": 0.0,
                            "wz": 0.0,
                        },
                        "engagement_detected": True,
                        "seated_detected": True,
                    }
                ],
            }
        return {"success": True, "message": primitive}

    def _assembly_board_v1_aruco_acceptance(self) -> tuple[dict[str, Any], str]:
        return {
            "accepted_generation": self.accepted_assembly_board_v1_aruco_generation,
            "calibration_id": self.accepted_assembly_board_v1_aruco_calibration_id,
        }, ""

    async def _maybe_inject_failure(self, **_kwargs: Any) -> None:
        return None

    async def _simulate_action(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def _task_failure(
        self,
        message: str,
        *,
        step: str,
        observations: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        failure = {
            "status": "failed",
            "content": message,
            "step": step,
            "observations": deepcopy(observations or {}),
        }
        self.failures.append(failure)
        return failure

    def _is_pose_in_workspace(self, _pose: dict[str, Any]) -> tuple[bool, str]:
        return True, "pose within workspace bounds"


def _recorded_step(
    step_name: str,
    pose: dict[str, float],
    *,
    child_frame_id: str = "tool0",
) -> dict[str, Any]:
    joint_names = (
        [f"joint{index}" for index in range(1, 7)]
        if child_frame_id == "link_eef"
        else [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ]
    )
    return {
        "step_name": step_name,
        "primitive": "move_cartesian",
        "params": deepcopy(pose),
        "capture_source": "hardware",
        "waypoint": {
            "pose": {
                "frame_id": "world",
                "child_frame_id": child_frame_id,
                **deepcopy(pose),
            },
            "joint_names": joint_names,
            "joint_positions": [0.0] * 6,
            "source": "hardware",
        },
    }


def _relative_recorded_step(
    step_name: str,
    pose: dict[str, float],
    *,
    relative_position_m: dict[str, float],
    reference_kind: str,
    reference_name: str,
    reference_position_m: dict[str, float],
    child_frame_id: str = "tool0",
    relative_pose: dict[str, float] | None = None,
) -> dict[str, Any]:
    step = _recorded_step(step_name, pose, child_frame_id=child_frame_id)
    step["position_sources"] = {
        "x": "captured_relative",
        "y": "captured_relative",
        "z": "captured_relative",
    }
    step["relative_position_m"] = deepcopy(relative_position_m)
    is_place_recording = reference_kind == "destination_target"
    step["relative_reference"] = {
        "kind": reference_kind,
        "frame_id": "world",
        "name": reference_name,
        "position_m": deepcopy(reference_position_m),
        "source": (
            "live_detection"
            if reference_kind == "detected_part"
            else "assembly_board-v1_aruco"
        ),
        "captured_at": time.time(),
    }
    if is_place_recording:
        camera_role = "xarm6" if child_frame_id == "link_eef" else "ur5e"
        step["relative_reference"].update(
            {
                "camera_role": camera_role,
                "generation": 3,
                "calibration_id": f"{camera_role}-calibration",
                "pose": deepcopy(
                    {
                        **reference_position_m,
                        "qx": 0.0,
                        "qy": 0.0,
                        "qz": 0.0,
                        "qw": 1.0,
                    }
                ),
            }
        )
    step["relative_pose"] = deepcopy(
        relative_pose
        or {
            **relative_position_m,
            "qx": pose["qx"],
            "qy": pose["qy"],
            "qz": pose["qz"],
            "qw": pose["qw"],
        }
    )
    step["computed_pose"] = robot_task_runtime._compose_se3(
        pose,
        robot_task_runtime._inverse_se3(step["relative_pose"]),
    )
    step["computed_position_m"] = {
        field: float(step["computed_pose"][field]) for field in ("x", "y", "z")
    }
    step["computed_source"] = "test"
    step["computed_at"] = time.time()
    step["confirmed"] = True
    return step


def _write_recording(
    root: Path,
    *,
    function_name: str,
    location: str,
    part_name: str,
    steps: list[dict[str, Any]],
    robot: str = "ur5e",
) -> Path:
    _ = location, part_name
    path = root / function_name / "default__hardware.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    child_frame_id = "link_eef" if robot == "xarm6" else "tool0"
    tcp_link = "link_tcp" if robot == "xarm6" else "ur5e_rg2_gripper_tcp"
    existing = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    robots = dict(existing.get("robots") or {})
    robots[robot] = {
        "frame_id": "world",
        "ee_link": child_frame_id,
        "tcp_link": tcp_link,
        "steps": steps,
    }
    path.write_text(
        json.dumps(
            {
                "function_name": function_name,
                "capture_source": "hardware",
                "robots": robots,
            }
        ),
        encoding="utf-8",
    )
    return path


def _place_recording(
    root: Path,
    *,
    part_name: str = "MG",
) -> tuple[dict[str, float], dict[str, float]]:
    above = {
        "x": 0.51,
        "y": 0.58,
        "z": 1.21,
        "qx": 0.0,
        "qy": 0.70710678,
        "qz": 0.0,
        "qw": 0.70710678,
    }
    target = {
        "x": 0.51,
        "y": 0.58,
        "z": 0.31,
        "qx": 0.0,
        "qy": 0.70710678,
        "qz": 0.0,
        "qw": 0.70710678,
    }
    _write_recording(
        root,
        function_name="place_approach",
        location="assembly_board-v1",
        part_name=part_name,
        steps=[
            _relative_recorded_step(
                "move_above_destination",
                above,
                relative_position_m={"x": 0.01, "y": -0.02, "z": 0.01},
                reference_kind="destination_target",
                reference_name="assembly_board-v1",
                reference_position_m={"x": 0.5, "y": 0.6, "z": 0.3},
            ),
            _relative_recorded_step(
                "descend",
                target,
                relative_position_m={"x": 0.01, "y": -0.02, "z": 0.01},
                reference_kind="destination_target",
                reference_name="assembly_board-v1",
                reference_position_m={"x": 0.5, "y": 0.6, "z": 0.3},
            ),
        ],
    )
    return above, target


def _operator_confirmed_held_part_handoff(
    part_name: str = "MG",
) -> dict[str, Any]:
    model_name = _MOVE_INSERT_MODEL_MAP[part_name]
    world_tool0 = {
        "x": 0.3,
        "y": 0.4,
        "z": 1.2,
        "qx": 0.0,
        "qy": 0.0,
        "qz": 0.0,
        "qw": 1.0,
    }
    world_part = {
        "x": 0.3,
        "y": 0.4,
        "z": 1.0,
        "qx": 0.0,
        "qy": 0.0,
        "qz": 0.0,
        "qw": 1.0,
    }
    recording_sha256 = "a" * 64
    reference_captured_at = time.time() - 60.0
    current_tf_stamp_sec = time.time()
    return {
        "robot": "ur5e",
        "destination_location": "assembly_board-v1",
        "part_name": part_name,
        "model_name": model_name,
        "origin_resource_location": "prusa-mk4-2",
        "frame_id": "world",
        "tool_frame": "tool0",
        "part_frame": "held_part_origin",
        "source": "operator_confirmed_pick_approach_recording",
        "orientation_source": "realsense_roboflow_identity",
        "captured_at": current_tf_stamp_sec,
        "current_world_tool0_pose": {
            "x": 0.4,
            "y": 0.5,
            "z": 1.3,
            "qx": 0.0,
            "qy": 0.0,
            "qz": 0.0,
            "qw": 1.0,
        },
        "current_tf_stamp_sec": current_tf_stamp_sec,
        "pick_approach_recording_path": (
            "/tmp/pick_approach/default__hardware.json"
        ),
        "pick_approach_recording_sha256": recording_sha256,
        "pick_approach_recorded_at": time.time() - 30.0,
        "world_tool0_pose_at_grasp": world_tool0,
        "world_held_part_pose_at_grasp": world_part,
        "tool0_to_held_part": {
            "x": 0.0,
            "y": 0.0,
            "z": -0.2,
            "qx": 0.0,
            "qy": 0.0,
            "qz": 0.0,
            "qw": 1.0,
        },
        "origin_pose_provenance": {
            "frame_id": "world",
            "part_name": part_name,
            "model_name": model_name,
            "source": "operator_confirmed_pick_approach_recording",
            "orientation_source": "realsense_roboflow_identity",
            "captured_at": reference_captured_at,
            "pick_approach_recording_sha256": recording_sha256,
        },
    }


def _set_retained_held_part_handoff(
    agent: _Agent,
    *,
    part_name: str = "MG",
    model_name: str = "gear_medium",
) -> dict[str, Any]:
    world_tool0 = {
        "x": 0.3,
        "y": 0.4,
        "z": 1.2,
        "qx": 0.0,
        "qy": 0.0,
        "qz": 0.0,
        "qw": 1.0,
    }
    world_part = {**world_tool0, "z": 1.0}
    provenance = {
        "frame_id": "world",
        "part_name": part_name,
        "model_name": model_name,
        "source": "live_detection",
    }
    handoff = {
        "part_name": part_name,
        "model_name": model_name,
        "origin_resource_location": "prusa-mk4-2",
        "frame_id": "world",
        "tool_frame": "tool0",
        "part_frame": "held_part_origin",
        "source": "pick_grasp",
        "world_tool0_pose_at_grasp": world_tool0,
        "world_held_part_pose_at_grasp": world_part,
        "tool0_to_held_part": {
            "x": 0.0,
            "y": 0.0,
            "z": -0.2,
            "qx": 0.0,
            "qy": 0.0,
            "qz": 0.0,
            "qw": 1.0,
        },
        "origin_pose_provenance": provenance,
    }
    task_context = dict(agent._task_ctx)
    retained_positions = dict(
        task_context.get("resolved_cartesian_positions") or {}
    )
    retained_positions.setdefault("descend", deepcopy(world_tool0))
    task_context.update(
        {
            "part_name": part_name,
            "model_name": model_name,
            "origin_resource_location": "prusa-mk4-2",
            "origin_pose": deepcopy(world_part),
            "origin_pose_provenance": deepcopy(provenance),
            "resolved_cartesian_positions": retained_positions,
            "held_part_handoff": deepcopy(handoff),
        }
    )
    agent._task_ctx = task_context
    return handoff


def test_five_function_definitions_preserve_exact_primitive_composition() -> None:
    assert robot_task_names() == (
        "pick_approach",
        "pick_grasp",
        "place_approach",
        "move_home",
        "place_insert",
    )
    registry = robot_task_registry()
    assert {
        name: [(step.id, step.op) for step in task.program.steps] for name, task in registry.items()
    } == {
        "pick_approach": [
            ("move_to_origin_resource_location", "move_to_named_pose"),
            ("detect_parts", "detect_parts"),
            ("compute_pick_targets", "compute_pick_targets"),
            ("open_gripper", "open_gripper"),
            ("move_above_part", "move_cartesian"),
            ("descend", "move_cartesian"),
        ],
        "pick_grasp": [
            ("grasp_part", "grasp_part"),
            ("lift", "move_relative"),
        ],
        "place_approach": [
            ("move_to_destination_location", "move_to_named_pose"),
            ("localize_assembly_board_v1", "localize_assembly_board_v1"),
            ("compute_place_targets", "compute_place_targets"),
            ("move_above_destination", "move_cartesian"),
            ("descend", "move_cartesian"),
        ],
        "place_insert": [
            ("move_insert", "move_insert"),
            ("release_part", "release_part"),
            ("snap_part_to_slot", "snap_part_to_slot"),
            ("lift", "move_relative"),
        ],
        "move_home": [("move_home", "move_to_named_pose")],
    }
    assert all(task.source == "robot_tasks.py" for task in registry.values())
    pick_rows = registry["pick_approach"].program.render_recovery_steps()
    compute_pick_targets_row = next(
        row for row in pick_rows if row["primitive"] == "compute_pick_targets"
    )
    assert compute_pick_targets_row["params"] == {
        "part_name": "<PART>",
        "product_geometry": "<PRODUCT_GEOMETRY>",
    }
    assert "detected_parts" not in repr(compute_pick_targets_row)
    for name, task in registry.items():
        assert task.handler is not None
        assert task.handler.__name__ == name
        assert task.handler.__qualname__ == f"RobotAgent.{name}"
        assert task.handler.__tool_spec__ is task
        assert task.handler.__doc__ == task.rendered_docstring()
    assert {
        (task.name, step.id)
        for task in registry.values()
        for step in task.program.steps
        if step.physical_position_required
    } == set()


def test_physical_pick_passes_one_detection_to_compute_and_uses_live_targets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(robot_task_runtime, "_TAUGHT_FUNCTIONS_ROOT", tmp_path)
    agent = _Agent(execution_mode="physical")

    result = asyncio.run(
        execute_robot_task(
            agent,
            "pick_approach",
            origin_resource_location="prusa-mk4-2",
            part_name="MG",
            speed=0.12,
        )
    )

    assert result["status"] == "completed", result
    assert [primitive for primitive, _params in agent.primitive_calls] == [
        "move_to_named_pose",
        "detect_parts",
        "compute_pick_targets",
        "open_gripper",
        "move_cartesian",
        "move_cartesian",
    ]
    assert agent.primitive_calls[0][1] == {
        "pose_name": "prusa-mk4-2",
        "speed": 0.12,
    }
    detected_parts = agent.primitive_calls[1][1]
    compute_params = agent.primitive_calls[2][1]
    assert detected_parts == {"part_name": "MG"}
    assert len(compute_params["detected_parts"]) == 1
    assert {
        field: compute_params["detected_parts"][0][field]
        for field in ("part_name", "model_name", "x", "y", "z", "frame_id")
    } == {
        "part_name": "MG",
        "model_name": "gear_medium",
        "x": 0.1,
        "y": 0.2,
        "z": 0.3,
        "frame_id": "world",
    }
    assert compute_params["detected_parts"][0]["captured_at"] > 0.0
    move_calls = [
        params for primitive, params in agent.primitive_calls if primitive == "move_cartesian"
    ]
    assert len(move_calls) == 2
    assert move_calls[0]["x"] == pytest.approx(0.1)
    assert move_calls[0]["y"] == pytest.approx(0.2)
    assert move_calls[0]["z"] == pytest.approx(0.8)
    assert move_calls[0]["speed"] == pytest.approx(0.12)
    assert move_calls[1] == pytest.approx(
        {"x": 0.1, "y": 0.2, "z": 0.4, "speed": 0.12}
    )
    assert agent._task_ctx["travel_z"] == pytest.approx(0.8)
    assert agent._task_ctx["source_stl"] == "/actual/Gear_Medium.STL"
    assert agent._task_ctx["hub_diameter_m"] == pytest.approx(0.03)
    assert agent._task_ctx["finger_tooth_clearance_m"] == pytest.approx(0.002)
    assert agent._task_ctx["finger_hub_overlap_m"] == pytest.approx(0.008)
    assert agent._task_ctx["pick_z_adjustment_m"] == pytest.approx(0.001)
    assert agent._task_ctx["pick_tool0_z_adjustment_m"] == pytest.approx(0.005)
    assert agent._task_ctx["predicted_closing_z_displacement_m"] == pytest.approx(
        -0.02616
    )
    assert agent._task_ctx["gripper_close_position"] == pytest.approx(0.37)
    assert agent._position == pytest.approx({"x": 0.1, "y": 0.2, "z": 0.4})

    agent.primitive_calls.clear()
    grasp_result = asyncio.run(
        execute_robot_task(
            agent,
            "pick_grasp",
            origin_resource_location="prusa-mk4-2",
            part_name="MG",
        )
    )
    assert grasp_result["status"] == "completed"
    assert [primitive for primitive, _params in agent.primitive_calls] == [
        "grasp_part",
        "move_relative",
    ]
    assert agent.primitive_calls[0][1]["position"] == pytest.approx(0.37)
    lift_params = agent.primitive_calls[-1][1]
    assert lift_params["dz"] == pytest.approx(0.4)
    handoff = agent._task_ctx["held_part_handoff"]
    assert handoff["part_name"] == "MG"
    assert handoff["frame_id"] == "world"
    assert handoff["tool_frame"] == "tool0"
    assert handoff["part_frame"] == "held_part_origin"
    assert handoff["world_tool0_pose_at_grasp"] == pytest.approx(
        {
            "x": 0.1,
            "y": 0.2,
            "z": 0.4,
            "qx": 0.0,
            "qy": 0.0,
            "qz": 0.0,
            "qw": 1.0,
        }
    )
    assert handoff["world_held_part_pose_at_grasp"] == pytest.approx(
        {
            "x": 0.1,
            "y": 0.2,
            "z": 0.3,
            "qx": 0.0,
            "qy": 0.0,
            "qz": 0.0,
            "qw": 1.0,
        }
    )
    assert handoff["tool0_to_held_part"] == pytest.approx(
        {
            "x": 0.0,
            "y": 0.0,
            "z": -0.1,
            "qx": 0.0,
            "qy": 0.0,
            "qz": 0.0,
            "qw": 1.0,
        }
    )


def test_physical_pick_move_above_failure_reports_completed_motion_steps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(robot_task_runtime, "_TAUGHT_FUNCTIONS_ROOT", tmp_path)
    agent = _Agent(execution_mode="physical")
    execute_primitive = agent._execute_primitive

    async def reject_move_above(
        primitive: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        if primitive == "move_cartesian":
            agent.primitive_calls.append((primitive, deepcopy(params)))
            return {
                "success": False,
                "message": "UR5e RTDE Cartesian target rejected: test reach limit",
            }
        return await execute_primitive(primitive, params)

    agent._execute_primitive = reject_move_above  # type: ignore[method-assign]

    result = asyncio.run(
        execute_robot_task(
            agent,
            "pick_approach",
            origin_resource_location="prusa-mk4-2",
            part_name="MG",
        )
    )

    assert result["status"] == "failed"
    assert result["step"] == "pick_approach.move_above_part"
    assert "UR5e RTDE Cartesian target rejected: test reach limit" in result["content"]
    assert "Completed motion steps: move_to_origin_resource_location" in result["content"]
    assert "move_above_part was requested but did not complete" in result["content"]
    assert "descend was not commanded" in result["content"]
    assert [primitive for primitive, _params in agent.primitive_calls].count(
        "move_cartesian"
    ) == 1


def test_physical_mg_pick_uses_relative_xyz_and_exact_captured_orientation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(robot_task_runtime, "_TAUGHT_FUNCTIONS_ROOT", tmp_path)
    agent = _Agent(execution_mode="physical")
    execute_primitive = agent._execute_primitive

    async def consistent_mg_geometry(
        primitive: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        result = await execute_primitive(primitive, params)
        if primitive == "compute_pick_targets":
            result.update(
                {
                    "table_surface_z_m": 0.20435,
                    "tcp_offset_z": -0.17,
                    "part_height": 0.02,
                }
            )
        return result

    agent._execute_primitive = consistent_mg_geometry  # type: ignore[method-assign]
    captured = {
        "x": 0.12,
        "y": 0.18,
        "z": 0.401,
        "qx": 0.0,
        "qy": 0.70710678,
        "qz": 0.0,
        "qw": 0.70710678,
    }
    descend = _relative_recorded_step(
        "descend",
        captured,
        relative_position_m={"x": 0.02, "y": -0.02, "z": 0.001},
        reference_kind="detected_part",
        reference_name="MG",
        reference_position_m={"x": 0.1, "y": 0.2, "z": 0.3},
    )
    _write_recording(
        tmp_path,
        function_name="pick_approach",
        location="prusa-mk4-2",
        part_name="MG",
        steps=[descend],
    )

    result = asyncio.run(
        execute_robot_task(
            agent,
            "pick_approach",
            origin_resource_location="prusa-mk4-2",
            part_name="MG",
        )
    )

    assert result["status"] == "completed", result
    descend_call = [
        params
        for primitive, params in agent.primitive_calls
        if primitive == "move_cartesian"
    ][-1]
    assert descend_call == pytest.approx(
        {
            "x": captured["x"],
            "y": captured["y"],
            "z": captured["z"],
            "qx": captured["qx"],
            "qy": captured["qy"],
            "qz": captured["qz"],
            "qw": captured["qw"],
            "speed": None,
        }
    )
    assert agent._position == pytest.approx(
        {"x": captured["x"], "y": captured["y"], "z": captured["z"]}
    )
    assert agent._task_ctx["pick_z"] == pytest.approx(captured["z"])
    assert agent._task_ctx["pick_tool0_z_adjustment_m"] == pytest.approx(0.005)
    assert agent._task_ctx["finger_tooth_clearance_m"] == pytest.approx(0.003)
    assert agent._task_ctx["finger_hub_overlap_m"] == pytest.approx(0.007)
    assert agent._task_ctx["cartesian_position_sources"]["descend"] == {
        "x": "captured_relative",
        "y": "captured_relative",
        "z": "captured_relative",
    }


def test_physical_pick_relative_xyz_moves_with_current_part_xyz(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(robot_task_runtime, "_TAUGHT_FUNCTIONS_ROOT", tmp_path)
    agent = _Agent(execution_mode="physical")
    descend = _relative_recorded_step(
        "descend",
        {
            "x": 0.15,
            "y": 0.18,
            "z": 0.41,
            "qx": 0.0,
            "qy": 0.70710678,
            "qz": 0.0,
            "qw": 0.70710678,
        },
        relative_position_m={"x": 0.05, "y": -0.02, "z": 0.01},
        reference_kind="detected_part",
        reference_name="RG",
        reference_position_m={"x": 0.1, "y": 0.2, "z": 0.3},
    )
    _write_recording(
        tmp_path,
        function_name="pick_approach",
        location="prusa-mk4-2",
        part_name="RG",
        steps=[descend],
    )

    result = asyncio.run(
        execute_robot_task(
            agent,
            "pick_approach",
            origin_resource_location="prusa-mk4-2",
            part_name="RG",
        )
    )

    assert result["status"] == "completed"
    move_calls = [
        params
        for primitive, params in agent.primitive_calls
        if primitive == "move_cartesian"
    ]
    assert move_calls[-1] == pytest.approx(
        {
            "x": 0.15,
            "y": 0.18,
            "z": 0.41,
            "qx": 0.0,
            "qy": 0.70710678,
            "qz": 0.0,
            "qw": 0.70710678,
            "speed": None,
        }
    )


def test_computed_pose_calibration_replays_offset_from_current_computed_pose(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(robot_task_runtime, "_TAUGHT_FUNCTIONS_ROOT", tmp_path)
    agent = _Agent(execution_mode="physical")
    original_execute = agent._execute_primitive
    computed_updates: list[
        tuple[str, dict[str, dict[str, Any]], dict[str, Any], float]
    ] = []
    agent._robot_task_computed_pose_callback = (
        lambda task_name, positions, reference, computed_at: computed_updates.append(
            (task_name, positions, reference, computed_at)
        )
    )

    async def moved_part(
        primitive: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        result = await original_execute(primitive, params)
        if primitive == "detect_parts":
            result["data"][0].update({"x": 0.13, "y": 0.16, "z": 0.32})
        return result

    agent._execute_primitive = moved_part  # type: ignore[method-assign]
    captured_quaternion = {
        "qx": 0.0,
        "qy": 0.70710678,
        "qz": 0.0,
        "qw": 0.70710678,
    }
    descend = _relative_recorded_step(
        "descend",
        {"x": 0.12, "y": 0.18, "z": 0.41, **captured_quaternion},
        relative_position_m={"x": 0.02, "y": -0.02, "z": 0.01},
        reference_kind="detected_part",
        reference_name="RG",
        reference_position_m={"x": 0.1, "y": 0.2, "z": 0.3},
    )
    descend["computed_position_m"] = {"x": 0.1, "y": 0.2, "z": 0.4}
    descend["computed_source"] = "preview"
    descend["computed_at"] = time.time()
    _write_recording(
        tmp_path,
        function_name="pick_approach",
        location="prusa-mk4-2",
        part_name="RG",
        steps=[descend],
    )

    result = asyncio.run(
        execute_robot_task(
            agent,
            "pick_approach",
            origin_resource_location="prusa-mk4-2",
            part_name="RG",
        )
    )

    assert result["status"] == "completed", result
    descend_call = [
        params
        for primitive, params in agent.primitive_calls
        if primitive == "move_cartesian"
    ][-1]
    assert descend_call == pytest.approx(
        {
            "x": 0.15,
            "y": 0.14,
            "z": 0.41,
            **captured_quaternion,
            "speed": None,
        }
    )
    assert agent._task_ctx["computed_cartesian_positions"]["descend"] == pytest.approx(
        {"x": 0.13, "y": 0.16, "z": 0.4}
    )
    assert computed_updates
    assert computed_updates[0][0] == "pick_approach"
    assert computed_updates[0][1]["descend"] == pytest.approx(
        {"x": 0.13, "y": 0.16, "z": 0.4}
    )


@pytest.mark.parametrize(
    ("robot", "child_frame_id"),
    [("ur5e", "tool0"), ("xarm6", "link_eef")],
)
def test_relative_pick_target_tracks_identical_part_delta_for_both_hardware_robots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    robot: str,
    child_frame_id: str,
) -> None:
    monkeypatch.setattr(robot_task_runtime, "_TAUGHT_FUNCTIONS_ROOT", tmp_path)
    agent = _Agent(execution_mode="physical", robot=robot)
    original_execute = agent._execute_primitive

    async def moved_part(
        primitive: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        result = await original_execute(primitive, params)
        if primitive == "detect_parts":
            detection = result["data"][0]
            detection.update({"x": 0.13, "y": 0.16, "z": 0.32})
        return result

    agent._execute_primitive = moved_part  # type: ignore[method-assign]
    captured_quaternion = {
        "qx": 0.0,
        "qy": 0.70710678,
        "qz": 0.0,
        "qw": 0.70710678,
    }
    descend = _relative_recorded_step(
        "descend",
        {"x": 0.15, "y": 0.18, "z": 0.41, **captured_quaternion},
        relative_position_m={"x": 0.05, "y": -0.02, "z": 0.01},
        reference_kind="detected_part",
        reference_name="RG",
        reference_position_m={"x": 0.1, "y": 0.2, "z": 0.3},
        child_frame_id=child_frame_id,
    )
    _write_recording(
        tmp_path,
        function_name="pick_approach",
        location="prusa-mk4-1" if robot == "xarm6" else "prusa-mk4-2",
        part_name="RG",
        steps=[descend],
        robot=robot,
    )

    result = asyncio.run(
        execute_robot_task(
            agent,
            "pick_approach",
            origin_resource_location=(
                "prusa-mk4-1" if robot == "xarm6" else "prusa-mk4-2"
            ),
            part_name="RG",
        )
    )

    assert result["status"] == "completed", result
    descend_call = [
        params
        for primitive, params in agent.primitive_calls
        if primitive == "move_cartesian"
    ][-1]
    assert descend_call == pytest.approx(
        {
            "x": 0.18,
            "y": 0.14,
            "z": 0.41,
            **captured_quaternion,
            "speed": None,
        }
    )


def test_xarm6_with_no_matching_recording_ignores_ur5e_pick_correction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(robot_task_runtime, "_TAUGHT_FUNCTIONS_ROOT", tmp_path)
    _write_recording(
        tmp_path,
        function_name="pick_approach",
        location="prusa-mk4-2",
        part_name="MG",
        robot="ur5e",
        steps=[
            _relative_recorded_step(
                "descend",
                {
                    "x": 1.1,
                    "y": 1.2,
                    "z": 1.4,
                    "qx": 0.0,
                    "qy": 1.0,
                    "qz": 0.0,
                    "qw": 0.0,
                },
                relative_position_m={"x": 1.0, "y": 1.0, "z": 1.0},
                reference_kind="detected_part",
                reference_name="MG",
                reference_position_m={"x": 0.1, "y": 0.2, "z": 0.4},
            )
        ],
    )
    agent = _Agent(execution_mode="physical", robot="xarm6")

    result = asyncio.run(
        execute_robot_task(
            agent,
            "pick_approach",
            origin_resource_location="prusa-mk4-1",
            part_name="MG",
        )
    )

    assert result["status"] == "completed"
    move_calls = [
        params
        for primitive, params in agent.primitive_calls
        if primitive == "move_cartesian"
    ]
    assert move_calls == pytest.approx(
        [
            {"x": 0.1, "y": 0.2, "z": 0.8, "speed": None},
            {"x": 0.1, "y": 0.2, "z": 0.4, "speed": None},
        ]
    )


def test_xarm6_pick_correction_learned_with_sg_applies_to_mg(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(robot_task_runtime, "_TAUGHT_FUNCTIONS_ROOT", tmp_path)
    correction_orientation = {
        "qx": 0.0,
        "qy": 0.70710678,
        "qz": 0.0,
        "qw": 0.70710678,
    }
    _write_recording(
        tmp_path,
        function_name="pick_approach",
        location="prusa-mk4-1",
        part_name="SG",
        robot="xarm6",
        steps=[
            _relative_recorded_step(
                "descend",
                {"x": 0.12, "y": 0.19, "z": 0.401, **correction_orientation},
                relative_position_m={"x": 0.02, "y": -0.01, "z": 0.001},
                reference_kind="detected_part",
                reference_name="SG",
                reference_position_m={"x": 0.1, "y": 0.2, "z": 0.4},
                child_frame_id="link_eef",
            )
        ],
    )
    agent = _Agent(execution_mode="physical", robot="xarm6")
    execute_primitive = agent._execute_primitive

    async def consistent_mg_geometry(
        primitive: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        result = await execute_primitive(primitive, params)
        if primitive == "compute_pick_targets":
            result.update(
                {
                    "table_surface_z_m": 0.20435,
                    "tcp_offset_z": -0.17,
                    "part_height": 0.02,
                }
            )
        return result

    agent._execute_primitive = consistent_mg_geometry  # type: ignore[method-assign]

    result = asyncio.run(
        execute_robot_task(
            agent,
            "pick_approach",
            origin_resource_location="prusa-mk4-1",
            part_name="MG",
        )
    )

    assert result["status"] == "completed", result
    assert agent._task_ctx["computed_cartesian_positions"]["descend"] == pytest.approx(
        {"x": 0.1, "y": 0.2, "z": 0.4}
    )
    assert agent._task_ctx["resolved_cartesian_positions"]["descend"] == pytest.approx(
        {"x": 0.12, "y": 0.19, "z": 0.401, **correction_orientation}
    )


def test_configured_future_robot_executes_pick_without_robot_name_allowlist(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(robot_task_runtime, "_TAUGHT_FUNCTIONS_ROOT", tmp_path)
    _write_recording(
        tmp_path,
        function_name="pick_approach",
        location="prusa-mk4-2",
        part_name="SG",
        robot="future_robot",
        steps=[
            _relative_recorded_step(
                "descend",
                {
                    "x": 0.11,
                    "y": 0.2,
                    "z": 0.401,
                    "qx": 0.0,
                    "qy": 0.0,
                    "qz": 0.0,
                    "qw": 1.0,
                },
                relative_position_m={"x": 0.01, "y": 0.0, "z": 0.001},
                reference_kind="detected_part",
                reference_name="SG",
                reference_position_m={"x": 0.1, "y": 0.2, "z": 0.4},
            )
        ],
    )
    agent = _Agent(execution_mode="physical", robot="future_robot")
    execute_primitive = agent._execute_primitive

    async def consistent_mg_geometry(
        primitive: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        result = await execute_primitive(primitive, params)
        if primitive == "compute_pick_targets":
            result.update(
                {
                    "table_surface_z_m": 0.20435,
                    "tcp_offset_z": -0.17,
                    "part_height": 0.02,
                }
            )
        return result

    agent._execute_primitive = consistent_mg_geometry  # type: ignore[method-assign]

    result = asyncio.run(
        execute_robot_task(
            agent,
            "pick_approach",
            origin_resource_location="prusa-mk4-2",
            part_name="MG",
        )
    )

    assert result["status"] == "completed", result
    assert agent._task_ctx["resolved_cartesian_positions"]["descend"] == pytest.approx(
        {
            "x": 0.11,
            "y": 0.2,
            "z": 0.401,
            "qx": 0.0,
            "qy": 0.0,
            "qz": 0.0,
            "qw": 1.0,
        }
    )


def test_relative_pick_rejects_stale_current_detection_before_cartesian_motion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(robot_task_runtime, "_TAUGHT_FUNCTIONS_ROOT", tmp_path)
    agent = _Agent(execution_mode="physical")
    original_execute = agent._execute_primitive

    async def stale_reference(
        primitive: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        result = await original_execute(primitive, params)
        if primitive == "compute_pick_targets":
            result["captured_at"] = time.time() - 30.0
        return result

    agent._execute_primitive = stale_reference  # type: ignore[method-assign]
    descend = _relative_recorded_step(
        "descend",
        {
            "x": 0.1,
            "y": 0.2,
            "z": 0.41,
            "qx": 0.0,
            "qy": 0.0,
            "qz": 0.0,
            "qw": 1.0,
        },
        relative_position_m={"x": 0.0, "y": 0.0, "z": 0.01},
        reference_kind="detected_part",
        reference_name="RG",
        reference_position_m={"x": 0.1, "y": 0.2, "z": 0.3},
    )
    _write_recording(
        tmp_path,
        function_name="pick_approach",
        location="prusa-mk4-2",
        part_name="RG",
        steps=[descend],
    )

    result = asyncio.run(
        execute_robot_task(
            agent,
            "pick_approach",
            origin_resource_location="prusa-mk4-2",
            part_name="RG",
        )
    )

    assert result["status"] == "failed"
    assert "current detected part reference is stale" in result["content"]
    assert not any(
        primitive == "move_cartesian" for primitive, _params in agent.primitive_calls
    )


def test_relative_pick_rejects_resolved_pose_outside_gripper_reach(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(robot_task_runtime, "_TAUGHT_FUNCTIONS_ROOT", tmp_path)
    agent = _Agent(execution_mode="physical")
    agent.static_capabilities["gripper_reach"]["max_xy_radius_m"] = 0.2
    descend = _relative_recorded_step(
        "descend",
        {
            "x": 1.1,
            "y": 0.2,
            "z": 0.4,
            "qx": 0.0,
            "qy": 0.0,
            "qz": 0.0,
            "qw": 1.0,
        },
        relative_position_m={"x": 1.0, "y": 0.0, "z": 0.0},
        reference_kind="detected_part",
        reference_name="RG",
        reference_position_m={"x": 0.1, "y": 0.2, "z": 0.3},
    )
    _write_recording(
        tmp_path,
        function_name="pick_approach",
        location="prusa-mk4-2",
        part_name="RG",
        steps=[descend],
    )

    result = asyncio.run(
        execute_robot_task(
            agent,
            "pick_approach",
            origin_resource_location="prusa-mk4-2",
            part_name="RG",
        )
    )

    assert result["status"] == "failed"
    assert "outside gripper_reach" in result["content"]
    assert not any(
        primitive == "move_cartesian" for primitive, _params in agent.primitive_calls
    )


def test_physical_mg_relative_descend_reports_hub_overlap_without_rejecting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(robot_task_runtime, "_TAUGHT_FUNCTIONS_ROOT", tmp_path)
    agent = _Agent(execution_mode="physical")
    execute_primitive = agent._execute_primitive

    async def consistent_mg_geometry(
        primitive: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        result = await execute_primitive(primitive, params)
        if primitive == "compute_pick_targets":
            result.update(
                {
                    "table_surface_z_m": 0.20435,
                    "tcp_offset_z": -0.17,
                    "part_height": 0.02,
                }
            )
        return result

    agent._execute_primitive = consistent_mg_geometry  # type: ignore[method-assign]
    descend = _relative_recorded_step(
        "descend",
        {
            "x": 0.1,
            "y": 0.2,
            "z": 0.405,
            "qx": 0.0,
            "qy": 0.70710678,
            "qz": 0.0,
            "qw": 0.70710678,
        },
        relative_position_m={"x": 0.0, "y": 0.0, "z": 0.005},
        reference_kind="detected_part",
        reference_name="MG",
        reference_position_m={"x": 0.1, "y": 0.2, "z": 0.3},
    )
    _write_recording(
        tmp_path,
        function_name="pick_approach",
        location="prusa-mk4-2",
        part_name="MG",
        steps=[descend],
    )

    result = asyncio.run(
        execute_robot_task(
            agent,
            "pick_approach",
            origin_resource_location="prusa-mk4-2",
            part_name="MG",
        )
    )

    assert result["status"] == "completed"
    assert agent._task_ctx["finger_hub_overlap_m"] == pytest.approx(0.003)
    assert [
        primitive for primitive, _params in agent.primitive_calls
    ].count("move_cartesian") == 2


def test_migrated_ur5e_pick_correction_is_confirmed_and_applied() -> None:
    recording_path = (
        robot_task_runtime._TAUGHT_FUNCTIONS_ROOT
        / "pick_approach"
        / "default__hardware.json"
    )
    recording = json.loads(recording_path.read_text(encoding="utf-8"))
    recorded_by_name = {
        str(step["step_name"]): dict(step)
        for step in recording["robots"]["ur5e"]["steps"]
    }
    assert set(recorded_by_name) == {"descend"}
    assert recorded_by_name["descend"]["confirmed"] is True
    assert recorded_by_name["descend"]["relative_position_m"] == pytest.approx(
        {
            axis: recorded_by_name["descend"]["waypoint"]["pose"][axis]
            - recorded_by_name["descend"]["computed_position_m"][axis]
            for axis in ("x", "y", "z")
        }
    )
    agent = _Agent(execution_mode="physical")
    execute_primitive = agent._execute_primitive

    async def consistent_mg_geometry(
        primitive: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        result = await execute_primitive(primitive, params)
        if primitive == "compute_pick_targets":
            result.update(
                {
                    "table_surface_z_m": 0.20435,
                    "tcp_offset_z": -0.17,
                    "part_height": 0.02,
                }
            )
        return result

    agent._execute_primitive = consistent_mg_geometry  # type: ignore[method-assign]

    result = asyncio.run(
        execute_robot_task(
            agent,
            "pick_approach",
            origin_resource_location="prusa-mk4-2",
            part_name="MG",
        )
    )

    assert result["status"] == "completed"
    move_calls = [
        params
        for primitive, params in agent.primitive_calls
        if primitive == "move_cartesian"
    ]
    assert move_calls[0]["z"] == pytest.approx(0.8)
    correction = recorded_by_name["descend"]["relative_position_m"]
    expected_descend = {
        "x": 0.1 + float(correction["x"]),
        "y": 0.2 + float(correction["y"]),
        "z": 0.4 + float(correction["z"]),
        **{
            field: float(recorded_by_name["descend"]["waypoint"]["pose"][field])
            for field in ("qx", "qy", "qz", "qw")
        },
    }
    assert correction["z"] == pytest.approx(0.01158770164919698)
    assert move_calls[1] == pytest.approx(
        {**expected_descend, "speed": None}
    )
    assert all(axis not in move_calls[0] for axis in ("qx", "qy", "qz", "qw"))
    assert agent._task_ctx["resolved_cartesian_positions"]["descend"] == pytest.approx(
        expected_descend
    )


def test_pick_without_optional_correction_keeps_legacy_xyz_result_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(robot_task_runtime, "_TAUGHT_FUNCTIONS_ROOT", tmp_path)
    agent = _Agent(execution_mode="physical")
    execute_primitive = agent._execute_primitive

    async def legacy_move_cartesian_result(
        primitive: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        result = await execute_primitive(primitive, params)
        if primitive == "move_cartesian":
            result.pop("absolute_position", None)
        return result

    agent._execute_primitive = legacy_move_cartesian_result  # type: ignore[method-assign]

    result = asyncio.run(
        execute_robot_task(
            agent,
            "pick_approach",
            origin_resource_location="prusa-mk4-2",
            part_name="MG",
        )
    )

    assert result["status"] == "completed", result
    assert agent._task_ctx["resolved_cartesian_positions"]["descend"] == pytest.approx(
        {"x": 0.1, "y": 0.2, "z": 0.4}
    )


def test_pick_without_optional_correction_retains_controller_full_pose(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(robot_task_runtime, "_TAUGHT_FUNCTIONS_ROOT", tmp_path)
    agent = _Agent(execution_mode="physical")
    execute_primitive = agent._execute_primitive
    preserved_orientation = {
        "qx": 0.5,
        "qy": -0.5,
        "qz": 0.5,
        "qw": 0.5,
    }

    async def controller_full_pose(
        primitive: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        result = await execute_primitive(primitive, params)
        if primitive == "move_cartesian":
            result["absolute_position"] = {
                "x": float(params["x"]),
                "y": float(params["y"]),
                "z": float(params["z"]),
                **preserved_orientation,
            }
        return result

    agent._execute_primitive = controller_full_pose  # type: ignore[method-assign]

    result = asyncio.run(
        execute_robot_task(
            agent,
            "pick_approach",
            origin_resource_location="prusa-mk4-2",
            part_name="MG",
        )
    )

    assert result["status"] == "completed", result
    move_calls = [
        params
        for primitive, params in agent.primitive_calls
        if primitive == "move_cartesian"
    ]
    assert all(axis not in move_calls[-1] for axis in ("qx", "qy", "qz", "qw"))
    assert agent._task_ctx["resolved_cartesian_positions"]["descend"] == pytest.approx(
        {
            "x": 0.1,
            "y": 0.2,
            "z": 0.4,
            **preserved_orientation,
        }
    )


def test_unconfirmed_absolute_pick_recording_is_ignored(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(robot_task_runtime, "_TAUGHT_FUNCTIONS_ROOT", tmp_path)
    agent = _Agent(execution_mode="physical")
    captured = {
        "x": 0.1,
        "y": 0.2,
        "z": 0.398,
        "qx": 0.0,
        "qy": 0.0,
        "qz": 0.0,
        "qw": 1.0,
    }
    descend = _recorded_step("descend", captured)
    descend["position_sources"] = {
        "x": "computed",
        "y": "computed",
        "z": "captured",
    }
    descend["manual_position_m"] = {}
    _write_recording(
        tmp_path,
        function_name="pick_approach",
        location="prusa-mk4-2",
        part_name="MG",
        steps=[descend],
    )

    execute_primitive = agent._execute_primitive

    async def consistent_mg_geometry(
        primitive: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        result = await execute_primitive(primitive, params)
        if primitive == "compute_pick_targets":
            result.update(
                {
                    "table_surface_z_m": 0.20235,
                    "tcp_offset_z": -0.17,
                    "part_height": 0.02,
                }
            )
        return result

    agent._execute_primitive = consistent_mg_geometry  # type: ignore[method-assign]

    result = asyncio.run(
        execute_robot_task(
            agent,
            "pick_approach",
            origin_resource_location="prusa-mk4-2",
            part_name="MG",
        )
    )

    assert result["status"] == "completed"
    move_calls = [
        params
        for primitive, params in agent.primitive_calls
        if primitive == "move_cartesian"
    ]
    assert move_calls[-1]["z"] == pytest.approx(0.4)
    assert all(axis not in move_calls[-1] for axis in ("qx", "qy", "qz", "qw"))


def test_physical_mg_zero_hub_overlap_does_not_reject_taught_descend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(robot_task_runtime, "_TAUGHT_FUNCTIONS_ROOT", tmp_path)
    agent = _Agent(execution_mode="physical")
    execute_primitive = agent._execute_primitive

    async def insufficient_hub_overlap(
        primitive: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        result = await execute_primitive(primitive, params)
        if primitive == "compute_pick_targets":
            result.update(
                {
                    "table_surface_z_m": 0.15,
                    "tcp_offset_z": -0.17,
                    "part_height": 0.02,
                }
            )
        return result

    agent._execute_primitive = insufficient_hub_overlap  # type: ignore[method-assign]
    descend = _relative_recorded_step(
        "descend",
        {
            "x": 0.1,
            "y": 0.2,
            "z": 0.4,
            "qx": 0.0,
            "qy": 0.0,
            "qz": 0.0,
            "qw": 1.0,
        },
        relative_position_m={"x": 0.0, "y": 0.0, "z": 0.0},
        reference_kind="detected_part",
        reference_name="MG",
        reference_position_m={"x": 0.1, "y": 0.2, "z": 0.3},
    )
    _write_recording(
        tmp_path,
        function_name="pick_approach",
        location="prusa-mk4-2",
        part_name="MG",
        steps=[descend],
    )

    result = asyncio.run(
        execute_robot_task(
            agent,
            "pick_approach",
            origin_resource_location="prusa-mk4-2",
            part_name="MG",
        )
    )

    assert result["status"] == "completed"
    assert agent._task_ctx["finger_hub_overlap_m"] == pytest.approx(0.0)
    assert [primitive for primitive, _params in agent.primitive_calls] == [
        "move_to_named_pose",
        "detect_parts",
        "compute_pick_targets",
        "open_gripper",
        "move_cartesian",
        "move_cartesian",
    ]


def test_pick_grasp_requires_pick_approach_gripper_close_position() -> None:
    agent = _Agent(execution_mode="physical")
    agent._current_state = "at_pick"
    agent._task_ctx = {
        "part_name": "MG",
        "origin_resource_location": "prusa-mk4-2",
        "model_name": "gear_medium",
        "travel_z": 0.8,
    }

    result = asyncio.run(
        execute_robot_task(
            agent,
            "pick_grasp",
            origin_resource_location="prusa-mk4-2",
            part_name="MG",
        )
    )

    assert result == {
        "status": "blocked",
        "content": "pick_approach has not established a gripper_close_position.",
    }
    assert agent.primitive_calls == []


def test_physical_pick_target_rejection_stops_before_open_gripper_or_motion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(robot_task_runtime, "_TAUGHT_FUNCTIONS_ROOT", tmp_path)
    agent = _Agent(execution_mode="physical")
    execute_primitive = agent._execute_primitive

    async def reject_pick_target(
        primitive: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        if primitive == "compute_pick_targets":
            agent.primitive_calls.append((primitive, deepcopy(params)))
            return {"success": False, "message": "physical detection is stale"}
        return await execute_primitive(primitive, params)

    agent._execute_primitive = reject_pick_target  # type: ignore[method-assign]

    result = asyncio.run(
        execute_robot_task(
            agent,
            "pick_approach",
            origin_resource_location="prusa-mk4-2",
            part_name="MG",
        )
    )

    assert result["status"] == "failed"
    assert result["step"] == "pick_approach.compute_pick_targets"
    assert "physical detection is stale" in result["content"]
    assert [primitive for primitive, _params in agent.primitive_calls] == [
        "move_to_named_pose",
        "detect_parts",
        "compute_pick_targets",
    ]


def test_physical_pick_staging_failure_stops_before_detection_or_gripper() -> None:
    agent = _Agent(execution_mode="physical")
    execute_primitive = agent._execute_primitive

    async def reject_staging(
        primitive: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        if primitive == "move_to_named_pose":
            agent.primitive_calls.append((primitive, deepcopy(params)))
            return {"success": False, "message": "staging trajectory failed"}
        return await execute_primitive(primitive, params)

    agent._execute_primitive = reject_staging  # type: ignore[method-assign]

    result = asyncio.run(
        execute_robot_task(
            agent,
            "pick_approach",
            origin_resource_location="prusa-mk4-2",
            part_name="MG",
        )
    )

    assert result["status"] == "failed"
    assert result["step"] == "pick_approach.move_to_origin_resource_location"
    assert "staging trajectory failed" in result["content"]
    assert [primitive for primitive, _params in agent.primitive_calls] == ["move_to_named_pose"]
    assert agent._current_state == "idle"


def test_physical_pick_detection_failure_stops_after_staging_before_gripper() -> None:
    agent = _Agent(execution_mode="physical")
    execute_primitive = agent._execute_primitive
    failure = (
        "ur5e did not become stationary within 2.00 s; last motion "
        "(2.00 mm, 0.60 deg); inference was not started"
    )

    async def reject_detection(
        primitive: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        if primitive == "detect_parts":
            agent.primitive_calls.append((primitive, deepcopy(params)))
            return {"success": False, "message": failure}
        return await execute_primitive(primitive, params)

    agent._execute_primitive = reject_detection  # type: ignore[method-assign]

    result = asyncio.run(
        execute_robot_task(
            agent,
            "pick_approach",
            origin_resource_location="prusa-mk4-2",
            part_name="MG",
        )
    )

    assert result["status"] == "failed"
    assert result["step"] == "pick_approach.detect_parts"
    assert failure in result["content"]
    assert [primitive for primitive, _params in agent.primitive_calls] == [
        "move_to_named_pose",
        "detect_parts",
    ]
    assert agent._current_state == "idle"


def test_pick_approach_requires_idle_before_physical_staging() -> None:
    agent = _Agent(execution_mode="physical")
    agent._current_state = "at_pick"

    result = asyncio.run(
        execute_robot_task(
            agent,
            "pick_approach",
            origin_resource_location="prusa-mk4-2",
            part_name="MG",
        )
    )

    assert result == {
        "status": "blocked",
        "content": "pick_approach requires the robot to be idle.",
    }
    assert agent.primitive_calls == []


def test_manual_pick_approach_skips_only_resource_state_guard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(robot_task_runtime, "_TAUGHT_FUNCTIONS_ROOT", tmp_path)
    agent = _Agent(execution_mode="physical")
    agent._current_state = "picked"

    result = asyncio.run(
        execute_robot_task(
            agent,
            "pick_approach",
            robot_task_runtime._MANUAL_FUNCTION_EXECUTION_AUTHORITY,
            origin_resource_location="prusa-mk4-2",
            part_name="MG",
        )
    )

    assert result["status"] == "completed"
    assert agent._current_state == "at_pick"
    assert [primitive for primitive, _params in agent.primitive_calls] == [
        "move_to_named_pose",
        "detect_parts",
        "compute_pick_targets",
        "open_gripper",
        "move_cartesian",
        "move_cartesian",
    ]


def test_manual_pick_approach_keeps_held_part_guard() -> None:
    agent = _Agent(execution_mode="physical")
    agent._current_state = "picked"
    agent._held_part = "MG"

    result = asyncio.run(
        execute_robot_task(
            agent,
            "pick_approach",
            robot_task_runtime._MANUAL_FUNCTION_EXECUTION_AUTHORITY,
            origin_resource_location="prusa-mk4-2",
            part_name="MG",
        )
    )

    assert result == {
        "status": "blocked",
        "content": "Cannot move-to-pick while already holding a part.",
    }
    assert agent.primitive_calls == []


def test_manual_authority_cannot_be_supplied_as_a_task_keyword() -> None:
    agent = _Agent(execution_mode="physical")
    agent._current_state = "picked"

    result = asyncio.run(
        execute_robot_task(
            agent,
            "pick_approach",
            manual_function_execution_authority=(
                robot_task_runtime._MANUAL_FUNCTION_EXECUTION_AUTHORITY
            ),
            origin_resource_location="prusa-mk4-2",
            part_name="MG",
        )
    )

    assert result == {
        "status": "blocked",
        "content": "pick_approach requires the robot to be idle.",
    }
    assert agent.primitive_calls == []


@pytest.mark.parametrize(
    ("robot", "origin_resource_location"),
    [("ur5e", "prusa-mk4-2"), ("xarm6", "prusa-mk4-1")],
)
def test_manual_pick_grasp_skips_only_resource_state_for_both_robots(
    robot: str,
    origin_resource_location: str,
) -> None:
    agent = _Agent(execution_mode="physical", robot=robot)
    agent._current_state = "idle"
    agent._task_ctx = {
        "part_name": "MG",
        "origin_resource_location": origin_resource_location,
        "model_name": "gear_medium",
        "gripper_close_position": 0.37,
        "travel_z": 0.8,
    }

    result = asyncio.run(
        execute_robot_task(
            agent,
            "pick_grasp",
            robot_task_runtime._MANUAL_FUNCTION_EXECUTION_AUTHORITY,
            origin_resource_location=origin_resource_location,
            part_name="MG",
        )
    )

    assert result["status"] == "completed"
    assert agent._current_state == "picked"
    assert [primitive for primitive, _params in agent.primitive_calls] == [
        "grasp_part",
        "move_relative",
    ]


def test_manual_place_functions_skip_only_resource_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(robot_task_runtime, "_TAUGHT_FUNCTIONS_ROOT", tmp_path)
    _place_recording(tmp_path)
    agent = _Agent(execution_mode="physical")
    agent._current_state = "idle"
    agent._held_part = "MG"
    agent._gripper_state = "closed"
    _set_retained_held_part_handoff(agent)

    approach_result = asyncio.run(
        execute_robot_task(
            agent,
            "place_approach",
            robot_task_runtime._MANUAL_FUNCTION_EXECUTION_AUTHORITY,
            destination_location="assembly_board-v1",
            part_name="MG",
        )
    )

    assert approach_result["status"] == "completed"
    assert agent._current_state == "positioned"
    assert set(agent._task_ctx["resolved_cartesian_positions"]["descend"]) == {
        "x",
        "y",
        "z",
        "qx",
        "qy",
        "qz",
        "qw",
    }
    compute_params = next(
        params
        for primitive, params in agent.primitive_calls
        if primitive == "compute_place_targets"
    )
    assert compute_params["pick_ctx"]["manual_function_execution"] is True
    assert "manual_function_execution" not in agent._task_ctx

    agent._current_state = "picked"
    agent.primitive_calls.clear()
    insert_result = asyncio.run(
        execute_robot_task(
            agent,
            "place_insert",
            robot_task_runtime._MANUAL_FUNCTION_EXECUTION_AUTHORITY,
            destination_location="assembly_board-v1",
            part_name="MG",
        )
    )

    assert insert_result["status"] == "blocked"
    assert "cannot release or lift" in insert_result["content"]
    assert agent._current_state == "picked"
    assert agent._held_part == "MG"
    assert agent._gripper_state == "closed"
    assert agent.primitive_calls == []


def test_manual_place_approach_runs_independently_without_held_part(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(robot_task_runtime, "_TAUGHT_FUNCTIONS_ROOT", tmp_path)
    _place_recording(tmp_path)
    agent = _Agent(execution_mode="physical")
    agent._current_state = "positioned"
    agent._task_ctx = {
        "destination_location": "assembly_board-v1",
        "travel_z": 1.2,
    }
    retained_task_context = deepcopy(agent._task_ctx)

    result = asyncio.run(
        execute_robot_task(
            agent,
            "place_approach",
            robot_task_runtime._MANUAL_FUNCTION_EXECUTION_AUTHORITY,
            destination_location="assembly_board-v1",
            part_name="MG",
        )
    )

    assert result == {
        "status": "completed",
        "content": (
            "Completed independent place_approach with held_part empty; "
            "the RobotAgent pick/place context was preserved."
        ),
    }
    assert agent._current_state == "positioned"
    assert agent._held_part is None
    assert agent._gripper_state == "open"
    assert agent._task_ctx == retained_task_context
    compute_params = next(
        params
        for primitive, params in agent.primitive_calls
        if primitive == "compute_place_targets"
    )
    assert compute_params["pick_ctx"] == {}


@pytest.mark.parametrize(
    ("part_name", "model_name"),
    list(_MOVE_INSERT_MODEL_MAP.items()),
)
def test_operator_confirmed_place_approach_adopts_real_custody(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    part_name: str,
    model_name: str,
) -> None:
    monkeypatch.setattr(robot_task_runtime, "_TAUGHT_FUNCTIONS_ROOT", tmp_path)
    _place_recording(tmp_path, part_name=part_name)
    agent = _Agent(execution_mode="physical")
    agent._current_state = "positioned"
    agent._position = {"x": 0.2, "y": -0.3, "z": 0.4}
    agent._task_ctx = {
        "destination_location": "assembly_board-v1",
        "travel_z": 1.2,
    }
    handoff = _operator_confirmed_held_part_handoff(part_name)

    result = asyncio.run(
        execute_robot_task(
            agent,
            "place_approach",
            robot_task_runtime._MANUAL_FUNCTION_EXECUTION_AUTHORITY,
            True,
            handoff,
            destination_location="assembly_board-v1",
            part_name=part_name,
        )
    )

    assert result == {
        "status": "completed",
        "content": (
            "Completed place_approach after adopting operator-confirmed held_part "
            f"{part_name!r} from the confirmed pick_approach.descend handoff; "
            f"{part_name} remains "
            "clamped. place_approach did not require move_insert; Supervised Test "
            "move_insert remains blocked until its protected recipe and calibration "
            "are ready."
        ),
        "operator_confirmed_held_part": True,
        "operator_held_part": part_name,
        "held_part_handoff_adopted": True,
        "move_insert_trial_context_ready": False,
        "move_insert_authorized": False,
    }
    assert agent._current_state == "positioned"
    assert agent._held_part == part_name
    assert agent._gripper_state == "closed"
    assert agent._position == pytest.approx({"x": 0.51, "y": 0.58, "z": 0.31})
    assert agent._task_ctx["part_name"] == part_name
    assert agent._task_ctx["model_name"] == model_name
    assert agent._task_ctx["held_part_handoff"] == handoff
    assert agent._task_ctx["operator_confirmed_held_part"] is True
    assert agent._task_ctx["pick_approach_recording_sha256"] == "a" * 64
    compute_params = next(
        params
        for primitive, params in agent.primitive_calls
        if primitive == "compute_place_targets"
    )
    assert compute_params["part_name"] == part_name
    assert compute_params["pick_ctx"]["held_part_handoff"] == handoff
    assert compute_params["pick_ctx"]["operator_confirmed_held_part"] is True
    assert compute_params["pick_ctx"]["manual_function_execution"] is True
    assert "manual_function_execution" not in agent._task_ctx

    trial_result = asyncio.run(
        robot_task_runtime.execute_place_insert_move_insert_trial(
            agent,
            destination_location="assembly_board-v1",
            part_name=part_name,
            trial_id="move-insert-test",
        )
    )
    assert trial_result["status"] == "blocked"
    assert "physical move_insert trial profile" in trial_result["content"]


def test_operator_confirmation_without_handoff_blocks_before_motion() -> None:
    agent = _Agent(execution_mode="physical")

    result = asyncio.run(
        execute_robot_task(
            agent,
            "place_approach",
            robot_task_runtime._MANUAL_FUNCTION_EXECUTION_AUTHORITY,
            True,
            destination_location="assembly_board-v1",
            part_name="MG",
        )
    )

    assert result == {
        "status": "blocked",
        "content": "operator-confirmed held-part handoff is missing",
    }
    assert agent._held_part is None
    assert agent._gripper_state == "open"
    assert agent._task_ctx == {}
    assert agent.primitive_calls == []


@pytest.mark.parametrize(
    ("requested_part_name", "handoff_part_name"),
    [
        ("SG", "MG"),
        ("MG", "LG"),
        ("LG", "SCP"),
        ("SCP", "MCP"),
        ("MCP", "LCP"),
        ("LCP", "SG"),
    ],
)
def test_operator_confirmed_handoff_never_crosses_exact_part_or_model_identity(
    requested_part_name: str,
    handoff_part_name: str,
) -> None:
    agent = _Agent(execution_mode="physical")
    handoff = _operator_confirmed_held_part_handoff(handoff_part_name)

    result = asyncio.run(
        execute_robot_task(
            agent,
            "place_approach",
            robot_task_runtime._MANUAL_FUNCTION_EXECUTION_AUTHORITY,
            True,
            handoff,
            destination_location="assembly_board-v1",
            part_name=requested_part_name,
        )
    )

    assert result["status"] == "blocked"
    assert "held-part handoff identity is invalid" in result["content"]
    assert agent._held_part is None
    assert agent._gripper_state == "open"
    assert agent._task_ctx == {}
    assert agent.primitive_calls == []


@pytest.mark.parametrize(
    ("robot", "part_name"),
    [
        ("xarm6", "MG"),
        ("ur5e", "SRP"),
        ("ur5e", "MRP"),
        ("ur5e", "LRP"),
    ],
)
def test_operator_confirmed_handoff_keeps_xarm6_and_rectangular_parts_blocked(
    robot: str,
    part_name: str,
) -> None:
    agent = _Agent(execution_mode="physical", robot=robot)
    handoff = _operator_confirmed_held_part_handoff()

    result = asyncio.run(
        execute_robot_task(
            agent,
            "place_approach",
            robot_task_runtime._MANUAL_FUNCTION_EXECUTION_AUTHORITY,
            True,
            handoff,
            destination_location="assembly_board-v1",
            part_name=part_name,
        )
    )

    assert result["status"] == "blocked"
    assert "available only for physical ur5e" in result["content"]
    assert agent._held_part is None
    assert agent._gripper_state == "open"
    assert agent.primitive_calls == []


def test_operator_confirmed_handoff_does_not_require_gripper_command_history(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(robot_task_runtime, "_TAUGHT_FUNCTIONS_ROOT", tmp_path)
    _place_recording(tmp_path)
    agent = _Agent(execution_mode="physical")
    handoff = _operator_confirmed_held_part_handoff()

    result = asyncio.run(
        execute_robot_task(
            agent,
            "place_approach",
            robot_task_runtime._MANUAL_FUNCTION_EXECUTION_AUTHORITY,
            True,
            handoff,
            destination_location="assembly_board-v1",
            part_name="MG",
        )
    )

    assert result["status"] == "completed", result
    assert result["move_insert_trial_context_ready"] is False
    assert result["move_insert_authorized"] is False
    assert agent._held_part == "MG"
    assert agent._gripper_state == "closed"


@pytest.mark.parametrize(
    ("field", "replacement", "expected_message"),
    [
        (
            "source",
            "pick_grasp",
            "held-part handoff identity is invalid",
        ),
        (
            "orientation_source",
            "unknown",
            "held-part handoff identity is invalid",
        ),
        (
            "pick_approach_recording_sha256",
            "not-a-sha",
            "recording SHA-256 is invalid",
        ),
        (
            "captured_at",
            1.0,
            "current TF evidence is stale or invalid",
        ),
    ],
)
def test_operator_confirmed_handoff_rejects_invalid_evidence_before_motion(
    field: str,
    replacement: object,
    expected_message: str,
) -> None:
    agent = _Agent(execution_mode="physical")
    handoff = _operator_confirmed_held_part_handoff()
    handoff[field] = replacement

    result = asyncio.run(
        execute_robot_task(
            agent,
            "place_approach",
            robot_task_runtime._MANUAL_FUNCTION_EXECUTION_AUTHORITY,
            True,
            handoff,
            destination_location="assembly_board-v1",
            part_name="MG",
        )
    )

    assert result["status"] == "blocked"
    assert expected_message in result["content"]
    assert agent._held_part is None
    assert agent._task_ctx == {}
    assert agent.primitive_calls == []


def test_operator_confirmed_handoff_rejects_nonrecomposable_transform() -> None:
    agent = _Agent(execution_mode="physical")
    handoff = _operator_confirmed_held_part_handoff()
    handoff["tool0_to_held_part"]["x"] = 0.01

    result = asyncio.run(
        execute_robot_task(
            agent,
            "place_approach",
            robot_task_runtime._MANUAL_FUNCTION_EXECUTION_AUTHORITY,
            True,
            handoff,
            destination_location="assembly_board-v1",
            part_name="MG",
        )
    )

    assert result == {
        "status": "blocked",
        "content": (
            "operator-confirmed tool0 -> held part transform does not recompose "
            "the confirmed pick poses"
        ),
    }
    assert agent._held_part is None
    assert agent._task_ctx == {}
    assert agent.primitive_calls == []


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("current_world_tool0_pose", {"x": True}),
        (
            "tool0_to_held_part",
            {
                "x": 0.0,
                "y": 0.0,
                "z": -0.2,
                "qx": 0.0,
                "qy": 0.0,
                "qz": 0.0,
                "qw": 1.0,
                "unexpected": 0.0,
            },
        ),
    ],
)
def test_operator_confirmed_handoff_rejects_nonexact_pose_evidence(
    field: str,
    replacement: dict[str, object],
) -> None:
    agent = _Agent(execution_mode="physical")
    handoff = _operator_confirmed_held_part_handoff()
    handoff[field] = replacement

    result = asyncio.run(
        execute_robot_task(
            agent,
            "place_approach",
            robot_task_runtime._MANUAL_FUNCTION_EXECUTION_AUTHORITY,
            True,
            handoff,
            destination_location="assembly_board-v1",
            part_name="MG",
        )
    )

    assert result == {
        "status": "blocked",
        "content": f"operator-confirmed held-part {field} is invalid",
    }
    assert agent._held_part is None
    assert agent._task_ctx == {}
    assert agent.primitive_calls == []


def test_operator_adoption_retains_custody_when_place_approach_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(robot_task_runtime, "_TAUGHT_FUNCTIONS_ROOT", tmp_path)
    _place_recording(tmp_path)
    agent = _Agent(execution_mode="physical")
    execute_primitive = agent._execute_primitive

    async def fail_after_adoption(
        primitive: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        if primitive == "move_to_named_pose":
            agent.primitive_calls.append((primitive, deepcopy(params)))
            return {"success": False, "message": "staging failed"}
        return await execute_primitive(primitive, params)

    agent._execute_primitive = fail_after_adoption  # type: ignore[method-assign]
    handoff = _operator_confirmed_held_part_handoff()

    result = asyncio.run(
        execute_robot_task(
            agent,
            "place_approach",
            robot_task_runtime._MANUAL_FUNCTION_EXECUTION_AUTHORITY,
            True,
            handoff,
            destination_location="assembly_board-v1",
            part_name="MG",
        )
    )

    assert result["status"] == "failed"
    assert "staging failed" in result["content"]
    assert agent._current_state == "picked"
    assert agent._held_part == "MG"
    assert agent._gripper_state == "closed"
    assert agent._task_ctx["held_part_handoff"] == handoff


def test_operator_confirmed_place_approach_cannot_replace_active_pick_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(robot_task_runtime, "_TAUGHT_FUNCTIONS_ROOT", tmp_path)
    _place_recording(tmp_path)
    agent = _Agent(execution_mode="physical")
    agent._current_state = "at_pick"
    agent._task_ctx = {
        "part_name": "MG",
        "origin_resource_location": "prusa-mk4-2",
    }

    result = asyncio.run(
        execute_robot_task(
            agent,
            "place_approach",
            robot_task_runtime._MANUAL_FUNCTION_EXECUTION_AUTHORITY,
            True,
            _operator_confirmed_held_part_handoff(),
            destination_location="assembly_board-v1",
            part_name="MG",
        )
    )

    assert result == {
        "status": "blocked",
        "content": (
            "operator_confirmed_held_part cannot replace active pick_grasp custody "
            "or context; clear or complete the active pick sequence first."
        ),
    }
    assert agent._held_part is None
    assert agent._current_state == "at_pick"
    assert agent.primitive_calls == []


def test_operator_confirmed_handoff_cannot_bypass_sequential_execution() -> None:
    agent = _Agent(execution_mode="physical")

    result = asyncio.run(
        execute_robot_task(
            agent,
            "place_approach",
            None,
            True,
            _operator_confirmed_held_part_handoff(),
            destination_location="assembly_board-v1",
            part_name="MG",
        )
    )

    assert result == {
        "status": "blocked",
        "content": (
            "operator_confirmed_held_part is available only for manual "
            "place_approach commissioning."
        ),
    }
    assert agent._held_part is None
    assert agent._task_ctx == {}
    assert agent.primitive_calls == []


@pytest.mark.parametrize(
    ("execution_mode", "robot", "destination_location", "part_name"),
    [
        ("simulation", "ur5e", "assembly_board-v1", "MG"),
        ("physical", "xarm6", "assembly_board-v1", "MG"),
        ("physical", "ur5e", "inspection_station", "MG"),
        ("physical", "ur5e", "assembly_board-v1", "SRP"),
        ("physical", "ur5e", "assembly_board-v1", "mg"),
    ],
)
def test_operator_confirmed_place_approach_is_exactly_scoped(
    execution_mode: str,
    robot: str,
    destination_location: str,
    part_name: str,
) -> None:
    agent = _Agent(execution_mode=execution_mode, robot=robot)

    result = asyncio.run(
        execute_robot_task(
            agent,
            "place_approach",
            robot_task_runtime._MANUAL_FUNCTION_EXECUTION_AUTHORITY,
            True,
            destination_location=destination_location,
            part_name=part_name,
        )
    )

    assert result == {
        "status": "blocked",
        "content": (
            "operator_confirmed_held_part is available only for physical ur5e "
            "place_approach at assembly_board-v1 with exact part_name 'SG', "
            "'MG', 'LG', 'SCP', 'MCP', or 'LCP'."
        ),
    }
    assert agent.primitive_calls == []


@pytest.mark.parametrize("operator_confirmation", [1, "true", None])
def test_operator_confirmed_place_approach_requires_boolean_identity(
    operator_confirmation: object,
) -> None:
    agent = _Agent(execution_mode="physical")

    result = asyncio.run(
        execute_robot_task(
            agent,
            "place_approach",
            robot_task_runtime._MANUAL_FUNCTION_EXECUTION_AUTHORITY,
            operator_confirmation,  # type: ignore[arg-type]
            destination_location="assembly_board-v1",
            part_name="MG",
        )
    )

    assert result == {
        "status": "blocked",
        "content": "operator_confirmed_held_part must be a boolean.",
    }
    assert agent.primitive_calls == []


def test_manual_place_approach_keeps_held_part_guard_for_active_pick_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(robot_task_runtime, "_TAUGHT_FUNCTIONS_ROOT", tmp_path)
    _place_recording(tmp_path)
    agent = _Agent(execution_mode="physical")
    agent._current_state = "at_pick"
    agent._task_ctx = {
        "part_name": "MG",
        "origin_resource_location": "prusa-mk4-2",
    }

    result = asyncio.run(
        execute_robot_task(
            agent,
            "place_approach",
            robot_task_runtime._MANUAL_FUNCTION_EXECUTION_AUTHORITY,
            destination_location="assembly_board-v1",
            part_name="MG",
        )
    )

    assert result == {
        "status": "blocked",
        "content": "Cannot move-loaded without holding a part.",
    }
    assert agent.primitive_calls == []


def test_manual_place_insert_blocks_empty_held_part_at_assembly_board() -> None:
    agent = _Agent(execution_mode="physical")
    agent._current_state = "idle"
    agent._position = {"x": 0.2, "y": -0.3, "z": 0.4}

    result = asyncio.run(
        execute_robot_task(
            agent,
            "place_insert",
            robot_task_runtime._MANUAL_FUNCTION_EXECUTION_AUTHORITY,
            destination_location="assembly_board-v1",
            part_name="MG",
        )
    )

    assert result == {
        "status": "blocked",
        "content": (
            "Physical place_insert at assembly_board-v1 requires a held part and the "
            "retained pick_grasp/place_approach context. Run those functions first, "
            "then use Supervised Test move_insert until the exact part is confirmed."
        ),
    }
    assert agent._current_state == "idle"
    assert agent._held_part is None
    assert agent._gripper_state == "open"
    assert agent._task_ctx == {}
    assert agent.primitive_calls == []


def test_manual_place_insert_keeps_release_only_for_non_assembly_destination() -> None:
    agent = _Agent(execution_mode="physical")
    agent._current_state = "idle"
    agent._position = {"x": 0.2, "y": -0.3, "z": 0.4}

    result = asyncio.run(
        execute_robot_task(
            agent,
            "place_insert",
            robot_task_runtime._MANUAL_FUNCTION_EXECUTION_AUTHORITY,
            destination_location="inspection_station",
            part_name="MG",
        )
    )

    assert result["status"] == "completed", result
    assert [primitive for primitive, _params in agent.primitive_calls] == [
        "release_part",
        "move_relative",
    ]


def test_sequence_place_insert_still_requires_held_part() -> None:
    agent = _Agent(execution_mode="physical")
    agent._current_state = "positioned"
    agent._task_ctx = {"destination_location": "assembly_board-v1"}

    result = asyncio.run(
        execute_robot_task(
            agent,
            "place_insert",
            destination_location="assembly_board-v1",
            part_name="MG",
        )
    )

    assert result == {
        "status": "blocked",
        "content": "No part currently held; run pick_grasp first.",
    }
    assert agent.primitive_calls == []


@pytest.mark.parametrize(
    ("current_state", "task_part", "task_origin", "expected_message"),
    [
        (
            "idle",
            "MG",
            "prusa-mk4-2",
            "pick_grasp requires pick_approach to finish at at_pick.",
        ),
        (
            "at_pick",
            "MG",
            "prusa-mk3",
            "pick_approach has not established the requested origin context.",
        ),
        (
            "at_pick",
            "SG",
            "prusa-mk4-2",
            "pick_approach has not established the requested part context.",
        ),
    ],
)
def test_pick_grasp_requires_at_pick_and_exact_pick_context(
    current_state: str,
    task_part: str,
    task_origin: str,
    expected_message: str,
) -> None:
    agent = _Agent(execution_mode="physical")
    agent._current_state = current_state
    agent._task_ctx = {
        "part_name": task_part,
        "origin_resource_location": task_origin,
        "model_name": "gear_medium",
        "gripper_close_position": 0.37,
        "travel_z": 0.8,
    }

    result = asyncio.run(
        execute_robot_task(
            agent,
            "pick_grasp",
            origin_resource_location="prusa-mk4-2",
            part_name="MG",
        )
    )

    assert result == {"status": "blocked", "content": expected_message}
    assert agent.primitive_calls == []


def test_physical_place_uses_computed_targets_without_recording(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(robot_task_runtime, "_TAUGHT_FUNCTIONS_ROOT", tmp_path)
    agent = _Agent(execution_mode="physical")
    agent._held_part = "MG"
    agent._current_state = "picked"
    _set_retained_held_part_handoff(agent)

    result = asyncio.run(
        execute_robot_task(
            agent,
            "place_approach",
            destination_location="assembly_board-v1",
            part_name="MG",
        )
    )

    assert result["status"] == "completed"
    move_calls = [
        params
        for primitive, params in agent.primitive_calls
        if primitive == "move_cartesian"
    ]
    assert move_calls == pytest.approx(
        [
            {"x": 0.5, "y": 0.6, "z": 1.2, "speed": None},
            {"x": 0.5, "y": 0.6, "z": 0.3, "speed": None},
        ]
    )
    assert "cartesian_position_sources" not in agent._task_ctx


def test_repeat_physical_place_approach_retains_immutable_handoff_and_pre_insert(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(robot_task_runtime, "_TAUGHT_FUNCTIONS_ROOT", tmp_path)
    _place_recording(tmp_path)
    recording_path = tmp_path / "place_approach/default__hardware.json"
    recording_sha256 = hashlib.sha256(recording_path.read_bytes()).hexdigest()
    agent = _Agent(execution_mode="physical")
    agent._held_part = "MG"
    agent._current_state = "picked"
    agent._gripper_state = "closed"
    retained_handoff = deepcopy(_set_retained_held_part_handoff(agent))
    execute_primitive = agent._execute_primitive

    async def fresh_place_targets(
        primitive: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        result = await execute_primitive(primitive, params)
        if primitive == "compute_place_targets":
            orientation = {"qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0}
            result["approach_pose"] = {
                "x": 0.5,
                "y": 0.6,
                "z": 1.2,
                **orientation,
            }
            result["target_pose"] = {
                "x": 0.5,
                "y": 0.6,
                "z": 0.3,
                **orientation,
            }
        return result

    agent._execute_primitive = fresh_place_targets  # type: ignore[method-assign]

    first = asyncio.run(
        execute_robot_task(
            agent,
            "place_approach",
            destination_location="assembly_board-v1",
            part_name="MG",
        )
    )

    assert first["status"] == "completed", first
    first_moves = [
        deepcopy(params)
        for primitive, params in agent.primitive_calls
        if primitive == "move_cartesian"
    ]
    assert len(first_moves) == 2
    assert agent._task_ctx["held_part_handoff"] == retained_handoff
    assert agent._task_ctx["resolved_cartesian_positions"]["descend"] == pytest.approx(
        {key: value for key, value in first_moves[-1].items() if key != "speed"}
    )
    assert agent._task_ctx["resolved_cartesian_positions"]["descend"] != (
        retained_handoff["world_tool0_pose_at_grasp"]
    )

    agent._position = {"x": -0.4, "y": 0.2, "z": 1.7}
    agent.primitive_calls.clear()
    repeated = asyncio.run(
        execute_robot_task(
            agent,
            "place_approach",
            robot_task_runtime._MANUAL_FUNCTION_EXECUTION_AUTHORITY,
            destination_location="assembly_board-v1",
            part_name="MG",
        )
    )

    assert repeated["status"] == "completed", repeated
    assert [primitive for primitive, _params in agent.primitive_calls] == [
        "move_to_named_pose",
        "localize_assembly_board_v1",
        "compute_place_targets",
        "move_cartesian",
        "move_cartesian",
    ]
    repeated_moves = [
        params
        for primitive, params in agent.primitive_calls
        if primitive == "move_cartesian"
    ]
    assert repeated_moves == pytest.approx(first_moves)
    assert agent._task_ctx["held_part_handoff"] == retained_handoff
    assert agent._task_ctx["resolved_cartesian_positions"]["descend"] == pytest.approx(
        {key: value for key, value in repeated_moves[-1].items() if key != "speed"}
    )
    assert agent._position == pytest.approx(
        {field: repeated_moves[-1][field] for field in ("x", "y", "z")}
    )
    assert hashlib.sha256(recording_path.read_bytes()).hexdigest() == recording_sha256


def test_current_rotated_tool_recording_replays_exact_poses_without_rewriting_artifacts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recording_root = Path(robot_task_runtime.__file__).resolve().parent / "taught_functions"
    recording_path = recording_root / "place_approach/default__hardware.json"
    assert recording_path.is_file()
    monkeypatch.setattr(
        robot_task_runtime,
        "_TAUGHT_FUNCTIONS_ROOT",
        recording_root,
    )
    demonstration_root = (
        Path.home()
        / ".local/share/cais-spade-llm/move_insert_demonstrations"
        / "insertion-demonstration-1787242384762-ed399f4b"
    )
    ur5e_manifest_path = (
        Path(robot_task_runtime.__file__).resolve().parents[2]
        / "initialization/resources/robot_ur5e.json"
    )
    assert ur5e_manifest_path.is_file()
    protected_artifacts = [recording_path, ur5e_manifest_path]
    if demonstration_root.is_dir():
        protected_artifacts.extend(
            sorted(path for path in demonstration_root.iterdir() if path.is_file())
        )

    def artifact_sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as artifact:
            for chunk in iter(lambda: artifact.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    hashes_before = {
        path: artifact_sha256(path) for path in protected_artifacts
    }
    recording_payload = json.loads(recording_path.read_text(encoding="utf-8"))
    recorded_steps = {
        str(step["step_name"]): step
        for step in recording_payload["robots"]["ur5e"]["steps"]
    }
    recorded_board_pose = {
        "x": 0.13561381662639274,
        "y": 0.04309333253741284,
        "z": 1.0415224148537343,
        "qx": -0.018736954151793554,
        "qy": -0.008108115985439925,
        "qz": 0.017140870818836546,
        "qw": 0.9996446246300993,
    }
    expected_recorded_poses: dict[str, dict[str, float]] = {}
    for step_name in ("move_above_destination", "descend"):
        recorded_step = recorded_steps[step_name]
        board_delta = robot_task_runtime._compose_se3(
            recorded_board_pose,
            robot_task_runtime._inverse_se3(
                dict(recorded_step["relative_reference"]["pose"])
            ),
        )
        expected_recorded_poses[step_name] = robot_task_runtime._compose_se3(
            board_delta,
            dict(recorded_step["params"]),
        )
    learned_pre_insert = {
        "x": -0.010342198499527616,
        "y": 0.09341850781101455,
        "z": 1.3238394745903925,
        "qx": -0.663122428233365,
        "qy": -0.7484753640188675,
        "qz": -0.0014406298419920869,
        "qw": 0.007155362769842532,
    }
    exact_insert = {
        "x": -0.010507877505830253,
        "y": 0.09190332781107517,
        "z": 1.3009418270545603,
        "qx": -0.663122428233365,
        "qy": -0.7484753640188675,
        "qz": -0.0014406298419920869,
        "qw": 0.007155362769842532,
    }
    insertion_axis = {
        "x": -0.007219656683339478,
        "y": -0.06602574253157188,
        "z": -0.9977918008685627,
    }
    profile = {
        "part_name": "MG",
        "calibration_id": "insertion-demonstration-7345d1d91fe5322c",
        "shared_calibration_id": "insertion-demonstration-7345d1d91fe5322c",
        "override_calibration_id": None,
        "profile_sha256": "b" * 64,
        "pre_insert_offset_m": 0.022948322000541744,
        "contact_speed_m_s": 0.005,
        "contact_force_delta_n": 4.0,
        "engagement_progress_m": 0.005277777080359542,
        "insertion_force_n": 15.0,
        "spiral_radius_m": 0.0015,
        "spiral_pitch_m": 0.0005,
        "spiral_speed_m_s": 0.002,
        "spiral_acceleration_m_s2": 0.02,
        "max_axial_force_n": 25.0,
        "max_lateral_force_n": 12.0,
        "max_torque_nm": 1.5,
        "tilt_tolerance_rad": 0.03490658503988659,
        "seated_depth_tolerance_m": 0.0011474161000270872,
        "settle_time_sec": 0.5,
    }
    hard_caps = {
        "insert_max_travel_m": 0.05,
        "insert_start_position_tolerance_m": 0.003,
        "insert_start_orientation_tolerance_rad": 0.034906585,
        "insert_max_timeout_sec": 30.0,
        "insert_max_contact_search_radius_m": 0.01,
    }
    hard_caps_sha256 = _move_insert_hard_caps_sha256(hard_caps)
    agent = _Agent(execution_mode="physical")
    agent._held_part = "MG"
    agent._current_state = "picked"
    agent._gripper_state = "closed"
    agent.assembly_board_v1_aruco_pose = deepcopy(recorded_board_pose)
    agent.assembly_board_v1_aruco_generation = 14
    agent.accepted_assembly_board_v1_aruco_generation = 14
    agent.assembly_board_v1_aruco_calibration_id = (
        "ce2da819-e953-4a84-b809-6d32e72cf9a8"
    )
    agent.accepted_assembly_board_v1_aruco_calibration_id = (
        agent.assembly_board_v1_aruco_calibration_id
    )
    _set_retained_held_part_handoff(agent)
    execute_primitive = agent._execute_primitive

    async def current_move_insert_geometry(
        primitive: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        result = await execute_primitive(primitive, params)
        if primitive == "compute_place_targets":
            result.update(
                {
                    "part_name": "MG",
                    "model_name": "gear_medium",
                    "destination_location": "assembly_board-v1",
                    "approach_pose": {
                        **learned_pre_insert,
                        "z": 1.425369,
                    },
                    "target_pose": {
                        **learned_pre_insert,
                        "z": 1.394955,
                    },
                    "pre_insert_pose": deepcopy(learned_pre_insert),
                    "insert_pose": deepcopy(exact_insert),
                    "insertion_axis_world": deepcopy(insertion_axis),
                    "move_insert_mode": "force_limited_trial",
                    "move_insert_profile": deepcopy(profile),
                    "move_insert_profile_sha256": "b" * 64,
                    "move_insert_hard_caps_sha256": hard_caps_sha256,
                    "surface_role": "assembly_slot",
                }
            )
        return result

    agent._execute_primitive = current_move_insert_geometry  # type: ignore[method-assign]

    result = asyncio.run(
        execute_robot_task(
            agent,
            "place_approach",
            destination_location="assembly_board-v1",
            part_name="MG",
            product_geometry={
                "move_insert_hard_caps": deepcopy(hard_caps),
                "move_insert_hard_caps_sha256": hard_caps_sha256,
            },
        )
    )

    assert result["status"] == "completed", result
    move_calls = [
        params
        for primitive, params in agent.primitive_calls
        if primitive == "move_cartesian"
    ]
    assert len(move_calls) == 2
    assert move_calls[0] == pytest.approx(
        {**expected_recorded_poses["move_above_destination"], "speed": None}
    )
    assert move_calls[1] == pytest.approx(
        {**expected_recorded_poses["descend"], "speed": None}
    )
    assert all(call["z"] not in {1.425369, 1.394955} for call in move_calls)
    assert agent._task_ctx["resolved_cartesian_positions"][
        "descend"
    ] == pytest.approx(expected_recorded_poses["descend"])
    assert agent._task_ctx["pre_insert_pose"] == pytest.approx(
        expected_recorded_poses["descend"]
    )
    assert {
        field: agent._task_ctx["insert_pose"][field]
        for field in ("x", "y", "z")
    } == pytest.approx(
        {field: exact_insert[field] for field in ("x", "y", "z")}
    )
    assert {
        field: agent._task_ctx["insert_pose"][field]
        for field in ("qx", "qy", "qz", "qw")
    } == pytest.approx(
        {
            field: expected_recorded_poses["descend"][field]
            for field in ("qx", "qy", "qz", "qw")
        }
    )
    assert agent._task_ctx["move_insert_boundary_ready"] is True
    boundary_metrics = agent._task_ctx["move_insert_boundary_metrics"]
    assert boundary_metrics["insertion_depth_m"] > 0.0
    assert boundary_metrics["insertion_travel_m"] <= hard_caps[
        "insert_max_travel_m"
    ]
    assert boundary_metrics["learned_start_position_error_m"] > hard_caps[
        "insert_start_position_tolerance_m"
    ]
    assert boundary_metrics["learned_start_orientation_error_rad"] > hard_caps[
        "insert_start_orientation_tolerance_rad"
    ]
    assert {
        path: artifact_sha256(path) for path in protected_artifacts
    } == hashes_before


def test_nominal_place_approach_clears_stale_move_insert_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(robot_task_runtime, "_TAUGHT_FUNCTIONS_ROOT", tmp_path)
    _place_recording(tmp_path)
    agent = _Agent(execution_mode="physical")
    agent._held_part = "MG"
    agent._current_state = "picked"
    agent._gripper_state = "closed"
    retained_handoff = deepcopy(_set_retained_held_part_handoff(agent))
    stale_fields = {
        "move_insert_profile": {"part_name": "MG"},
        "move_insert_profile_sha256": "a" * 64,
        "move_insert_mode": "force_limited",
        "move_insert_timeout_sec": 12.0,
        "pre_insert_pose": {"stale": True},
        "insert_pose": {"stale": True},
        "insertion_axis_world": {"x": 0.0, "y": 0.0, "z": -1.0},
        "move_insert_hard_caps": {"insert_max_travel_m": 0.05},
        "move_insert_boundary_ready": True,
        "move_insert_boundary_error": "stale",
        "move_insert_boundary_metrics": {"insertion_depth_m": 0.01},
        "move_insert_qualification_error": "stale",
    }
    agent._task_ctx.update(deepcopy(stale_fields))

    result = asyncio.run(
        execute_robot_task(
            agent,
            "place_approach",
            destination_location="assembly_board-v1",
            part_name="MG",
        )
    )

    assert result["status"] == "completed", result
    assert agent._task_ctx["held_part_handoff"] == retained_handoff
    assert "descend" in agent._task_ctx["resolved_cartesian_positions"]
    for field in stale_fields:
        assert field not in agent._task_ctx


@pytest.mark.parametrize(
    ("mutation", "expected_message"),
    [
        (
            lambda context: context.pop("held_part_handoff"),
            "held_part_handoff is missing",
        ),
        (
            lambda context: context["held_part_handoff"].update(
                {"part_name": "LG"}
            ),
            "held_part_handoff identity is invalid",
        ),
        (
            lambda context: context["held_part_handoff"].pop(
                "world_tool0_pose_at_grasp"
            ),
            "world_tool0_pose_at_grasp is missing",
        ),
        (
            lambda context: context["held_part_handoff"][
                "tool0_to_held_part"
            ].update({"x": 0.01}),
            "does not reconstruct its frozen pick poses",
        ),
        (
            lambda context: context["held_part_handoff"][
                "origin_pose_provenance"
            ].update({"frame_id": "base"}),
            "held_part_handoff provenance is invalid",
        ),
    ],
)
def test_physical_place_approach_rejects_invalid_immutable_handoff_before_motion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: Any,
    expected_message: str,
) -> None:
    monkeypatch.setattr(robot_task_runtime, "_TAUGHT_FUNCTIONS_ROOT", tmp_path)
    _place_recording(tmp_path)
    agent = _Agent(execution_mode="physical")
    agent._held_part = "MG"
    agent._current_state = "picked"
    agent._gripper_state = "closed"
    _set_retained_held_part_handoff(agent)
    mutation(agent._task_ctx)
    retained_context = deepcopy(agent._task_ctx)

    result = asyncio.run(
        execute_robot_task(
            agent,
            "place_approach",
            destination_location="assembly_board-v1",
            part_name="MG",
        )
    )

    assert result["status"] == "failed"
    assert result["step"] == "place_approach.held_part_handoff_preflight"
    assert expected_message in result["content"]
    assert agent.primitive_calls == []
    assert agent._current_state == "picked"
    assert agent._held_part == "MG"
    assert agent._gripper_state == "closed"
    assert agent._task_ctx == retained_context


def test_physical_place_correction_captured_with_mg_applies_to_lg(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(robot_task_runtime, "_TAUGHT_FUNCTIONS_ROOT", tmp_path)
    _place_recording(tmp_path, part_name="MG")
    agent = _Agent(execution_mode="physical")
    agent._held_part = "LG"
    agent._current_state = "picked"
    _set_retained_held_part_handoff(
        agent,
        part_name="LG",
        model_name="gear_large",
    )

    result = asyncio.run(
        execute_robot_task(
            agent,
            "place_approach",
            destination_location="assembly_board-v1",
            part_name="LG",
        )
    )

    assert result["status"] == "completed"
    move_calls = [
        params
        for primitive, params in agent.primitive_calls
        if primitive == "move_cartesian"
    ]
    assert move_calls[0] == pytest.approx(
        {
            "x": 0.51,
            "y": 0.58,
            "z": 1.21,
            "speed": None,
            "qx": 0.0,
            "qy": 0.70710678,
            "qz": 0.0,
            "qw": 0.70710678,
        }
    )


def test_generalized_place_payload_has_no_part_or_location_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(robot_task_runtime, "_TAUGHT_FUNCTIONS_ROOT", tmp_path)
    _place_recording(tmp_path, part_name="MG")
    path = tmp_path / "place_approach" / "default__hardware.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert "part_name" not in payload
    assert "name" not in payload
    assert set(payload["robots"]) == {"ur5e"}


def test_physical_place_insert_without_move_insert_profile_never_releases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(robot_task_runtime, "_TAUGHT_FUNCTIONS_ROOT", tmp_path)
    _place_recording(tmp_path)
    agent = _Agent(execution_mode="physical")
    agent._held_part = "MG"
    agent._current_state = "picked"
    agent._gripper_state = "closed"
    agent._position = {"x": 0.2, "y": -0.3, "z": 0.7}
    agent._task_ctx = {
        "part_name": "MG",
        "model_name": "gear_medium",
        "origin_resource_location": "prusa-mk4-2",
        "travel_z": 0.7,
    }
    _set_retained_held_part_handoff(agent)

    approach_result = asyncio.run(
        execute_robot_task(
            agent,
            "place_approach",
            destination_location="assembly_board-v1",
            part_name="MG",
            speed=0.1,
        )
    )
    assert approach_result["status"] == "completed"
    approach_moves = [
        params
        for primitive, params in agent.primitive_calls
        if primitive == "move_cartesian"
    ]
    assert len(approach_moves) == 2
    assert agent._task_ctx["travel_z"] == pytest.approx(approach_moves[0]["z"])
    assert agent._position == pytest.approx(
        {
            field: approach_moves[-1][field]
            for field in ("x", "y", "z")
        }
    )

    agent.primitive_calls.clear()
    insert_result = asyncio.run(
        execute_robot_task(
            agent,
            "place_insert",
            destination_location="assembly_board-v1",
            part_name="MG",
        )
    )
    assert insert_result["status"] == "blocked"
    assert "cannot release or lift" in insert_result["content"]
    assert agent.primitive_calls == []
    assert agent._held_part == "MG"
    assert agent._gripper_state != "open"


def test_place_approach_requires_exact_picked_resource_state() -> None:
    agent = _Agent(execution_mode="physical")
    agent._held_part = "MG"
    agent._current_state = "at_pick"

    result = asyncio.run(
        execute_robot_task(
            agent,
            "place_approach",
            destination_location="assembly_board-v1",
            part_name="MG",
        )
    )

    assert result == {
        "status": "blocked",
        "content": "place_approach requires pick_grasp to finish at picked.",
    }
    assert agent.primitive_calls == []


def test_place_insert_requires_exact_positioned_resource_state() -> None:
    agent = _Agent(execution_mode="physical")
    agent._held_part = "MG"
    agent._current_state = "picked"
    agent._task_ctx = {
        "destination_location": "assembly_board-v1",
        "model_name": "gear_medium",
        "travel_z": 0.69,
    }

    result = asyncio.run(
        execute_robot_task(
            agent,
            "place_insert",
            destination_location="assembly_board-v1",
            part_name="MG",
        )
    )

    assert result == {
        "status": "blocked",
        "content": "place_insert requires place_approach to finish at positioned.",
    }
    assert agent.primitive_calls == []


@pytest.mark.parametrize(
    ("mutation", "expected_message"),
    [
        (lambda steps: steps.append(deepcopy(steps[0])), "duplicate physical position step_name"),
        (
            lambda steps: steps[0]["waypoint"]["pose"].update({"frame_id": "base"}),
            "physical position frame must be world",
        ),
        (
            lambda steps: steps[0]["waypoint"]["pose"].update({"child_frame_id": "wrist_3_link"}),
            "physical position child frame must be tool0",
        ),
        (
            lambda steps: steps[0].update({"primitive": "move_relative"}),
            "physical position primitive mismatch",
        ),
        (
            lambda steps: steps[0]["waypoint"].update(
                {"joint_names": ["joint_0"], "joint_positions": [0.0]}
            ),
            "physical position joints do not match the configured ur5e hardware set",
        ),
        (
            lambda steps: steps[0]["waypoint"]["pose"].update({"x": float("nan")})
            or steps[0]["params"].update({"x": float("nan")}),
            "physical position x is not finite",
        ),
        (
            lambda steps: steps[0]["waypoint"]["pose"].update(
                {"qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 0.0}
            )
            or steps[0]["params"].update({"qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 0.0}),
            "physical position quaternion is zero",
        ),
        (
            lambda steps: steps[0]["relative_reference"].update({"camera_role": "xarm6"}),
            "relative_reference.camera_role does not match the executing robot",
        ),
        (
            lambda steps: steps[0]["relative_reference"].update({"frame_id": "base"}),
            "relative_reference.frame_id must be world",
        ),
        (
            lambda steps: steps[0]["relative_reference"].pop("calibration_id"),
            "relative_reference.calibration_id is missing or invalid",
        ),
        (
            lambda steps: steps[0]["relative_reference"].pop("pose"),
            "relative_reference.pose is missing",
        ),
        (
            lambda steps: steps[0]["relative_reference"]["pose"].update(
                {"z": 0.31}
            ),
            "relative_reference.pose does not match position_m",
        ),
        (
            lambda steps: steps[0]["relative_position_m"].update({"x": float("nan")}),
            "relative_position_m.x is not finite",
        ),
        (
            lambda steps: steps[0].pop("relative_pose"),
            "relative_pose for place_approach.move_above_destination is missing",
        ),
        (
            lambda steps: steps[0].pop("computed_pose"),
            "Unsafe legacy place_approach.move_above_destination correction: "
            "computed_pose is missing",
        ),
        (
            lambda steps: steps[0]["relative_pose"].update({"x": 0.0}),
            "relative_pose translation does not reconstruct the captured pose",
        ),
    ],
)
def test_invalid_physical_recording_fails_closed_before_motion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: Any,
    expected_message: str,
) -> None:
    monkeypatch.setattr(robot_task_runtime, "_TAUGHT_FUNCTIONS_ROOT", tmp_path)
    _place_recording(tmp_path)
    recording_path = tmp_path / "place_approach/default__hardware.json"
    steps = json.loads(recording_path.read_text(encoding="utf-8"))["robots"]["ur5e"][
        "steps"
    ]
    mutation(steps)
    _write_recording(
        tmp_path,
        function_name="place_approach",
        location="assembly_board-v1",
        part_name="MG",
        steps=steps,
    )
    agent = _Agent(execution_mode="physical")
    agent._held_part = "MG"
    agent._current_state = "picked"

    result = asyncio.run(
        execute_robot_task(
            agent,
            "place_approach",
            destination_location="assembly_board-v1",
            part_name="MG",
        )
    )

    assert result["status"] == "failed"
    assert expected_message in result["content"]
    assert agent.primitive_calls == []


def test_simulation_ignores_hardware_recordings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(robot_task_runtime, "_TAUGHT_FUNCTIONS_ROOT", tmp_path)
    agent = _Agent(execution_mode="simulation")

    result = asyncio.run(
        execute_robot_task(
            agent,
            "pick_approach",
            origin_resource_location="prusa-mk4-2",
            part_name="MG",
        )
    )

    assert result["status"] == "completed"
    assert [primitive for primitive, _params in agent.primitive_calls] == [
        "detect_parts",
        "compute_pick_targets",
        "open_gripper",
        "move_cartesian",
        "move_cartesian",
    ]
    move_calls = [
        params for primitive, params in agent.primitive_calls if primitive == "move_cartesian"
    ]
    assert len(move_calls) == 2
    assert move_calls[0]["x"] == pytest.approx(0.1)
    assert "qx" not in move_calls[0]


def test_simulation_place_keeps_static_targets_without_board_localization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(robot_task_runtime, "_TAUGHT_FUNCTIONS_ROOT", tmp_path)
    agent = _Agent(execution_mode="simulation")
    agent._held_part = "MG"
    agent._current_state = "picked"

    result = asyncio.run(
        execute_robot_task(
            agent,
            "place_approach",
            destination_location="assembly_board-v1",
            part_name="MG",
        )
    )

    assert result["status"] == "completed"
    assert [primitive for primitive, _params in agent.primitive_calls] == [
        "compute_place_targets",
        "move_cartesian",
        "move_cartesian",
    ]
    assert agent.primitive_calls[0][1]["assembly_board_v1_aruco"] is None
    assert agent.primitive_calls[1][1] == pytest.approx(
        {"x": 0.5, "y": 0.6, "z": 1.2, "speed": None}
    )
    assert agent.primitive_calls[2][1] == pytest.approx(
        {"x": 0.5, "y": 0.6, "z": 0.3, "speed": None}
    )


def test_physical_place_applies_robot_correction_to_computed_world_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(robot_task_runtime, "_TAUGHT_FUNCTIONS_ROOT", tmp_path)
    agent = _Agent(execution_mode="physical")
    agent._held_part = "MG"
    agent._current_state = "picked"
    _set_retained_held_part_handoff(agent)
    square_root_half = 2.0**-0.5
    agent.assembly_board_v1_aruco_pose = {
        "x": 1.0,
        "y": 2.0,
        "z": 0.3,
        "qx": 0.0,
        "qy": 0.0,
        "qz": square_root_half,
        "qw": square_root_half,
    }
    agent.assembly_board_v1_aruco_generation = 9
    relative_orientation = {
        "qx": square_root_half,
        "qy": 0.0,
        "qz": 0.0,
        "qw": square_root_half,
    }
    recording_path = _write_recording(
        tmp_path,
        function_name="place_approach",
        location="assembly_board-v1",
        part_name="MG",
        steps=[
            _relative_recorded_step(
                "move_above_destination",
                {"x": 0.6, "y": 0.6, "z": 0.5, **relative_orientation},
                relative_position_m={"x": 0.1, "y": 0.0, "z": 0.2},
                relative_pose={
                    "x": 0.1,
                    "y": 0.0,
                    "z": 0.2,
                    **relative_orientation,
                },
                reference_kind="destination_target",
                reference_name="assembly_board-v1",
                reference_position_m={"x": 0.5, "y": 0.6, "z": 0.3},
            ),
            _relative_recorded_step(
                "descend",
                {"x": 0.6, "y": 0.6, "z": 0.3, **relative_orientation},
                relative_position_m={"x": 0.1, "y": 0.0, "z": 0.0},
                relative_pose={
                    "x": 0.1,
                    "y": 0.0,
                    "z": 0.0,
                    **relative_orientation,
                },
                reference_kind="destination_target",
                reference_name="assembly_board-v1",
                reference_position_m={"x": 0.5, "y": 0.6, "z": 0.3},
            ),
        ],
    )
    recorded_payload = json.loads(recording_path.read_text(encoding="utf-8"))
    recording_sha256 = hashlib.sha256(recording_path.read_bytes()).hexdigest()
    assert {
        step["relative_reference"]["generation"]
        for step in recorded_payload["robots"]["ur5e"]["steps"]
    } == {3}

    result = asyncio.run(
        execute_robot_task(
            agent,
            "place_approach",
            destination_location="assembly_board-v1",
            part_name="MG",
            speed=0.08,
        )
    )

    assert result["status"] == "completed", result
    assert [primitive for primitive, _params in agent.primitive_calls[:3]] == [
        "move_to_named_pose",
        "localize_assembly_board_v1",
        "compute_place_targets",
    ]
    assert agent.primitive_calls[0][1] == {
        "pose_name": "assembly_board-v1",
        "speed": 0.08,
    }
    localization = agent.primitive_calls[2][1]["assembly_board_v1_aruco"]
    assert localization["camera_role"] == "ur5e"
    assert localization["generation"] == 9
    assert localization["pose"] == pytest.approx(agent.assembly_board_v1_aruco_pose)
    move_calls = [
        params for primitive, params in agent.primitive_calls if primitive == "move_cartesian"
    ]
    assert len(move_calls) == 2
    assert move_calls[0] == pytest.approx(
        {
            "x": 1.0,
            "y": 2.1,
            "z": 0.5,
            "qx": 0.5,
            "qy": 0.5,
            "qz": 0.5,
            "qw": 0.5,
            "speed": 0.08,
        }
    )
    assert move_calls[1] == pytest.approx(
        {
            "x": 1.0,
            "y": 2.1,
            "z": 0.3,
            "qx": 0.5,
            "qy": 0.5,
            "qz": 0.5,
            "qw": 0.5,
            "speed": 0.08,
        }
    )
    assert agent._task_ctx["assembly_board_v1_aruco_generation"] == 9
    assert agent._task_ctx["assembly_board_v1_aruco"] == localization

    agent._current_state = "picked"
    agent._position = {"x": -0.4, "y": 0.7, "z": 1.6}
    agent.primitive_calls.clear()
    repeated = asyncio.run(
        execute_robot_task(
            agent,
            "place_approach",
            destination_location="assembly_board-v1",
            part_name="MG",
            speed=0.08,
        )
    )

    assert repeated["status"] == "completed", repeated
    repeated_moves = [
        params
        for primitive, params in agent.primitive_calls
        if primitive == "move_cartesian"
    ]
    assert len(repeated_moves) == len(move_calls)
    for repeated_move, first_move in zip(
        repeated_moves,
        move_calls,
        strict=True,
    ):
        assert repeated_move == pytest.approx(first_move)
    assert agent._task_ctx["resolved_cartesian_positions"][
        "descend"
    ] == pytest.approx(
        {key: value for key, value in move_calls[-1].items() if key != "speed"}
    )
    assert hashlib.sha256(recording_path.read_bytes()).hexdigest() == recording_sha256


def test_place_correction_with_rotated_tool_replays_world_z_upward(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(robot_task_runtime, "_TAUGHT_FUNCTIONS_ROOT", tmp_path)
    half_sqrt = 2.0**-0.5
    computed_pose = {
        "x": 0.5,
        "y": 0.6,
        "z": 0.5,
        "qx": 0.0,
        "qy": half_sqrt,
        "qz": 0.0,
        "qw": half_sqrt,
    }
    captured_pose = {
        **computed_pose,
        "z": computed_pose["z"] + 0.051,
    }
    relative_pose = robot_task_runtime._compose_se3(
        robot_task_runtime._inverse_se3(computed_pose),
        captured_pose,
    )
    assert relative_pose["x"] == pytest.approx(-0.051)
    assert relative_pose["z"] == pytest.approx(0.0, abs=1e-9)
    recorded_step = _relative_recorded_step(
        "move_above_destination",
        captured_pose,
        relative_position_m={"x": 0.0, "y": 0.0, "z": 0.051},
        relative_pose=relative_pose,
        reference_kind="destination_target",
        reference_name="assembly_board-v1",
        reference_position_m={"x": 0.5, "y": 0.6, "z": 0.3},
    )
    assert recorded_step["computed_pose"] == pytest.approx(computed_pose)
    _write_recording(
        tmp_path,
        function_name="place_approach",
        location="assembly_board-v1",
        part_name="MG",
        steps=[recorded_step],
    )
    agent = _Agent(execution_mode="physical")
    agent._held_part = "MG"
    agent._current_state = "picked"
    _set_retained_held_part_handoff(agent)
    execute_primitive = agent._execute_primitive

    async def rotated_targets(
        primitive: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        result = await execute_primitive(primitive, params)
        if primitive == "compute_place_targets":
            result["approach_pose"] = deepcopy(computed_pose)
            result["target_pose"] = {**computed_pose, "z": 0.3}
        return result

    agent._execute_primitive = rotated_targets  # type: ignore[method-assign]
    result = asyncio.run(
        execute_robot_task(
            agent,
            "place_approach",
            destination_location="assembly_board-v1",
            part_name="MG",
        )
    )

    assert result["status"] == "completed", result
    move_calls = [
        params
        for primitive, params in agent.primitive_calls
        if primitive == "move_cartesian"
    ]
    assert move_calls[0]["x"] == pytest.approx(0.5)
    assert move_calls[0]["y"] == pytest.approx(0.6)
    assert move_calls[0]["z"] == pytest.approx(0.551)
    assert move_calls[0]["z"] > computed_pose["z"]


def test_place_correction_orientation_mismatch_blocks_only_move_insert(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(robot_task_runtime, "_TAUGHT_FUNCTIONS_ROOT", tmp_path)
    half_sqrt = 2.0**-0.5
    corrected_pre_insert = {
        "x": 0.502,
        "y": 0.6,
        "z": 0.31,
        "qx": half_sqrt,
        "qy": 0.0,
        "qz": 0.0,
        "qw": half_sqrt,
    }
    _write_recording(
        tmp_path,
        function_name="place_approach",
        location="assembly_board-v1",
        part_name="MG",
        steps=[
            _relative_recorded_step(
                "descend",
                corrected_pre_insert,
                relative_position_m={"x": 0.002, "y": 0.0, "z": 0.0},
                relative_pose={
                    "x": 0.002,
                    "y": 0.0,
                    "z": 0.0,
                    "qx": half_sqrt,
                    "qy": 0.0,
                    "qz": 0.0,
                    "qw": half_sqrt,
                },
                reference_kind="destination_target",
                reference_name="assembly_board-v1",
                reference_position_m={"x": 0.5, "y": 0.6, "z": 0.3},
            )
        ],
    )
    agent = _Agent(execution_mode="physical")
    agent._held_part = "MG"
    agent._current_state = "picked"
    agent._gripper_state = "closed"
    agent._task_ctx = {
        "part_name": "MG",
        "origin_resource_location": "prusa-mk4-2",
    }
    _set_retained_held_part_handoff(agent)
    profile = {
        "part_name": "MG",
        "calibration_id": "insert-calibration-1",
        "shared_calibration_id": "insert-calibration-1",
        "override_calibration_id": None,
        "profile_sha256": "a" * 64,
        "pre_insert_offset_m": 0.01,
        "contact_speed_m_s": 0.01,
        "contact_force_delta_n": 2.0,
        "engagement_progress_m": 0.003,
        "insertion_force_n": 5.0,
        "spiral_radius_m": 0.0,
        "spiral_pitch_m": 0.0005,
        "spiral_speed_m_s": 0.002,
        "spiral_acceleration_m_s2": 0.02,
        "max_axial_force_n": 20.0,
        "max_lateral_force_n": 10.0,
        "max_torque_nm": 2.0,
        "tilt_tolerance_rad": 0.1,
        "seated_depth_tolerance_m": 0.001,
        "settle_time_sec": 0.2,
    }
    hard_caps = {
        "insert_max_travel_m": 0.05,
        "insert_start_position_tolerance_m": 0.003,
        "insert_start_orientation_tolerance_rad": 0.1,
        "insert_max_timeout_sec": 30.0,
        "insert_max_contact_search_radius_m": 0.01,
    }
    hard_caps_sha256 = _move_insert_hard_caps_sha256(hard_caps)
    execute_primitive = agent._execute_primitive

    async def insertion_targets(
        primitive: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        result = await execute_primitive(primitive, params)
        if primitive == "compute_place_targets":
            result.update(
                {
                    "part_name": "MG",
                    "model_name": "gear_medium",
                    "destination_location": "assembly_board-v1",
                    "approach_pose": {
                        "x": 0.5,
                        "y": 0.6,
                        "z": 0.36,
                        "qx": 0.0,
                        "qy": 0.0,
                        "qz": 0.0,
                        "qw": 1.0,
                    },
                    "target_pose": {
                        "x": 0.5,
                        "y": 0.6,
                        "z": 0.31,
                        "qx": 0.0,
                        "qy": 0.0,
                        "qz": 0.0,
                        "qw": 1.0,
                    },
                    "pre_insert_pose": {
                        "x": 0.5,
                        "y": 0.6,
                        "z": 0.31,
                        "qx": 0.0,
                        "qy": 0.0,
                        "qz": 0.0,
                        "qw": 1.0,
                    },
                    "insert_pose": {
                        "x": 0.5,
                        "y": 0.6,
                        "z": 0.30,
                        "qx": 0.0,
                        "qy": 0.0,
                        "qz": 0.0,
                        "qw": 1.0,
                    },
                    "insertion_axis_world": {"x": 0.0, "y": 0.0, "z": -1.0},
                    "move_insert_mode": "force_limited",
                    "move_insert_profile": deepcopy(profile),
                    "move_insert_profile_sha256": "a" * 64,
                    "move_insert_hard_caps_sha256": hard_caps_sha256,
                    "surface_role": "assembly_slot",
                }
            )
        return result

    agent._execute_primitive = insertion_targets  # type: ignore[method-assign]

    approach_result = asyncio.run(
        execute_robot_task(
            agent,
            "place_approach",
            destination_location="assembly_board-v1",
            part_name="MG",
            product_geometry={
                "move_insert_hard_caps": hard_caps,
                "move_insert_hard_caps_sha256": hard_caps_sha256,
            },
        )
    )

    assert approach_result["status"] == "completed", approach_result
    assert agent._task_ctx["resolved_cartesian_positions"][
        "descend"
    ] == pytest.approx(corrected_pre_insert)
    assert agent._task_ctx["pre_insert_pose"] == pytest.approx(
        corrected_pre_insert
    )
    assert agent._task_ctx["insert_pose"] == pytest.approx(
        {
            "x": 0.5,
            "y": 0.6,
            "z": 0.30,
            "qx": 0.0,
            "qy": 0.0,
            "qz": 0.0,
            "qw": 1.0,
        }
    )
    assert agent._task_ctx["move_insert_boundary_ready"] is False
    assert "orientation" in agent._task_ctx["move_insert_boundary_error"]
    agent.primitive_calls.clear()

    insert_result = asyncio.run(
        robot_task_runtime.execute_place_insert_move_insert_trial(
            agent,
            destination_location="assembly_board-v1",
            part_name="MG",
            trial_id="move-insert-test",
        )
    )

    assert insert_result["status"] == "blocked"
    assert "move_insert boundary is not ready" in insert_result["content"]
    assert agent.primitive_calls == []


def _move_insert_boundary_inputs() -> tuple[
    dict[str, float],
    dict[str, Any],
    dict[str, float],
]:
    start_pose = {
        "x": 0.0,
        "y": 0.0,
        "z": 0.02,
        "qx": 0.0,
        "qy": 0.0,
        "qz": 0.0,
        "qw": 1.0,
    }
    targets = {
        "pre_insert_pose": deepcopy(start_pose),
        "insert_pose": {**start_pose, "z": 0.0},
        "insertion_axis_world": {"x": 0.0, "y": 0.0, "z": -1.0},
        "move_insert_profile": {
            "part_name": "MG",
            "pre_insert_offset_m": 0.02,
            "contact_speed_m_s": 0.005,
            "engagement_progress_m": 0.005,
            "spiral_radius_m": 0.001,
            "spiral_pitch_m": 0.0005,
            "spiral_speed_m_s": 0.002,
            "spiral_acceleration_m_s2": 0.02,
            "seated_depth_tolerance_m": 0.001,
            "settle_time_sec": 0.5,
        },
    }
    hard_caps = {
        "insert_max_travel_m": 0.05,
        "insert_start_position_tolerance_m": 0.003,
        "insert_start_orientation_tolerance_rad": 0.1,
        "insert_max_timeout_sec": 30.0,
        "insert_max_contact_search_radius_m": 0.01,
    }
    targets["move_insert_hard_caps_sha256"] = _move_insert_hard_caps_sha256(
        hard_caps
    )
    return start_pose, targets, hard_caps


def test_move_insert_boundary_uses_exact_resolved_start_and_learned_target() -> None:
    start_pose, targets, hard_caps = _move_insert_boundary_inputs()

    boundary, error = robot_task_runtime._validated_move_insert_boundary(
        start_pose=start_pose,
        targets=targets,
        raw_hard_caps=hard_caps,
    )

    assert error == ""
    assert boundary["pre_insert_pose"] == pytest.approx(start_pose)
    assert boundary["insert_pose"] == pytest.approx(targets["insert_pose"])
    assert boundary["insertion_depth_m"] == pytest.approx(0.02)
    assert boundary["insertion_travel_m"] == pytest.approx(0.02)
    assert boundary["lateral_error_m"] == pytest.approx(0.0)
    assert boundary["learned_start_position_error_m"] == pytest.approx(0.0)
    assert boundary["learned_start_orientation_error_rad"] == pytest.approx(0.0)


def test_move_insert_boundary_derives_hash_and_retains_all_live_hard_caps() -> None:
    start_pose, targets, hard_caps = _move_insert_boundary_inputs()
    hard_caps["insert_max_insertion_force_n"] = 25.0
    targets["move_insert_hard_caps_sha256"] = "0" * 64

    first_boundary, first_error = robot_task_runtime._validated_move_insert_boundary(
        start_pose=start_pose,
        targets=targets,
        raw_hard_caps=hard_caps,
    )

    assert first_error == ""
    assert first_boundary["move_insert_hard_caps"] == hard_caps
    assert first_boundary["move_insert_hard_caps_sha256"] == (
        _move_insert_hard_caps_sha256(hard_caps)
    )

    repeated_targets = {
        **targets,
        **first_boundary,
    }
    second_boundary, second_error = robot_task_runtime._validated_move_insert_boundary(
        start_pose=start_pose,
        targets=repeated_targets,
        raw_hard_caps=first_boundary["move_insert_hard_caps"],
    )

    assert second_error == ""
    assert second_boundary["move_insert_hard_caps"] == hard_caps
    assert second_boundary["move_insert_hard_caps_sha256"] == (
        first_boundary["move_insert_hard_caps_sha256"]
    )


def test_move_insert_boundary_keeps_place_approach_descend_independent() -> None:
    start_pose, targets, hard_caps = _move_insert_boundary_inputs()
    start_pose["z"] = 0.025
    targets["pre_insert_pose"].update(
        {
            "x": 0.004,
            "qz": 0.0599640065,
            "qw": 0.9982005399,
        }
    )
    targets["move_insert_profile"]["pre_insert_offset_m"] = 0.03

    boundary, error = robot_task_runtime._validated_move_insert_boundary(
        start_pose=start_pose,
        targets=targets,
        raw_hard_caps=hard_caps,
    )

    assert error == ""
    assert boundary["pre_insert_pose"] == pytest.approx(start_pose)
    assert boundary["insert_pose"] == pytest.approx(
        {
            **targets["insert_pose"],
            "z": float(start_pose["z"]) - 0.03,
        }
    )
    assert boundary["insertion_depth_m"] == pytest.approx(0.03)
    assert boundary["insertion_travel_m"] == pytest.approx(0.03)
    assert boundary["learned_start_position_error_m"] > hard_caps[
        "insert_start_position_tolerance_m"
    ]
    assert boundary["learned_start_orientation_error_rad"] > hard_caps[
        "insert_start_orientation_tolerance_rad"
    ]


@pytest.mark.parametrize(
    "part_name",
    ("SG", "MG", "LG", "SCP", "MCP", "LCP"),
)
def test_move_insert_boundary_uses_each_recipe_depth_from_current_descend(
    part_name: str,
) -> None:
    start_pose, targets, hard_caps = _move_insert_boundary_inputs()
    current_start_depth_m = 0.027978430538150152
    demonstrated_insertion_depth_m = 0.024130104415888313
    start_pose["z"] = current_start_depth_m
    targets["pre_insert_pose"]["z"] = demonstrated_insertion_depth_m
    targets["move_insert_profile"]["part_name"] = part_name
    targets["move_insert_profile"]["pre_insert_offset_m"] = (
        demonstrated_insertion_depth_m
    )

    boundary, error = robot_task_runtime._validated_move_insert_boundary(
        start_pose=start_pose,
        targets=targets,
        raw_hard_caps=hard_caps,
    )

    assert error == ""
    assert boundary["pre_insert_pose"] == pytest.approx(start_pose)
    assert boundary["insert_pose"]["z"] == pytest.approx(
        current_start_depth_m - demonstrated_insertion_depth_m
    )
    assert boundary["insertion_depth_m"] == pytest.approx(
        demonstrated_insertion_depth_m
    )
    assert boundary["insertion_travel_m"] == pytest.approx(
        demonstrated_insertion_depth_m
    )
    assert boundary["learned_start_position_error_m"] == pytest.approx(
        current_start_depth_m - demonstrated_insertion_depth_m
    )


@pytest.mark.parametrize(
    ("mutation", "expected_message"),
    [
        (
            lambda _start, targets, _caps: targets["insert_pose"].update(
                {"x": 0.011}
            ),
            "insert_max_contact_search_radius_m",
        ),
        (
            lambda _start, _targets, caps: caps.update(
                {"insert_max_travel_m": 0.01}
            ),
            "insert_max_travel_m",
        ),
        (
            lambda _start, _targets, caps: caps.pop("insert_max_timeout_sec"),
            "insert_max_timeout_sec is invalid",
        ),
    ],
)
def test_move_insert_boundary_failures_are_explicit(
    mutation: Any,
    expected_message: str,
) -> None:
    start_pose, targets, hard_caps = _move_insert_boundary_inputs()
    mutation(start_pose, targets, hard_caps)
    targets["move_insert_hard_caps_sha256"] = _move_insert_hard_caps_sha256(
        hard_caps
    )

    boundary, error = robot_task_runtime._validated_move_insert_boundary(
        start_pose=start_pose,
        targets=targets,
        raw_hard_caps=hard_caps,
    )

    assert boundary == {}
    assert expected_message in error


def test_physical_non_board_place_uses_cartesian_fallback_without_named_position(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(robot_task_runtime, "_TAUGHT_FUNCTIONS_ROOT", tmp_path)
    quaternion = {"qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0}
    above = {"x": 0.4, "y": -0.2, "z": 0.7, **quaternion}
    target = {"x": 0.4, "y": -0.2, "z": 0.35, **quaternion}
    _write_recording(
        tmp_path,
        function_name="place_approach",
        location="inspection_station",
        part_name="MG",
        steps=[
            _recorded_step("move_above_destination", above),
            _recorded_step("descend", target),
        ],
    )
    agent = _Agent(execution_mode="physical")
    agent._held_part = "MG"
    agent._current_state = "picked"

    approach_result = asyncio.run(
        execute_robot_task(
            agent,
            "place_approach",
            destination_location="inspection_station",
            part_name="MG",
            speed=0.1,
        )
    )

    assert approach_result["status"] == "completed", approach_result
    assert [primitive for primitive, _params in agent.primitive_calls] == [
        "compute_place_targets",
        "move_cartesian",
        "move_cartesian",
    ]
    assert agent.primitive_calls[1][1] == pytest.approx(
        {"x": 0.5, "y": 0.6, "z": 1.2, "speed": 0.1}
    )
    assert agent.primitive_calls[2][1] == pytest.approx(
        {"x": 0.5, "y": 0.6, "z": 0.3, "speed": 0.1}
    )
    assert "assembly_board_v1_aruco" not in agent._task_ctx

    agent.primitive_calls.clear()
    insert_result = asyncio.run(
        execute_robot_task(
            agent,
            "place_insert",
            destination_location="inspection_station",
            part_name="MG",
        )
    )
    assert insert_result["status"] == "completed", insert_result
    assert [primitive for primitive, _params in agent.primitive_calls] == [
        "release_part",
        "move_relative",
    ]


@pytest.mark.parametrize(
    ("invalid_field", "invalid_value", "expected_message"),
    [
        ("camera_role", "xarm6", "camera_role does not match the executing robot"),
        ("captured_at", time.time() - 30.0, "ArUco pose is stale"),
    ],
)
def test_physical_place_rejects_invalid_role_specific_aruco_after_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    invalid_field: str,
    invalid_value: Any,
    expected_message: str,
) -> None:
    monkeypatch.setattr(robot_task_runtime, "_TAUGHT_FUNCTIONS_ROOT", tmp_path)
    _place_recording(tmp_path)
    agent = _Agent(execution_mode="physical")
    agent._held_part = "MG"
    agent._current_state = "picked"
    _set_retained_held_part_handoff(agent)
    execute_primitive = agent._execute_primitive

    async def invalid_localization(
        primitive: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        result = await execute_primitive(primitive, params)
        if primitive == "localize_assembly_board_v1":
            result[invalid_field] = invalid_value
        return result

    agent._execute_primitive = invalid_localization  # type: ignore[method-assign]
    result = asyncio.run(
        execute_robot_task(
            agent,
            "place_approach",
            destination_location="assembly_board-v1",
            part_name="MG",
        )
    )

    assert result["status"] == "failed"
    assert result["step"] == "place_approach.localize_assembly_board_v1"
    assert expected_message in result["content"]
    assert "Completed motion steps: move_to_destination_location" in result["content"]
    assert "move_above_destination and descend were not commanded" in result["content"]
    assert [primitive for primitive, _params in agent.primitive_calls] == [
        "move_to_named_pose",
        "localize_assembly_board_v1",
    ]


def test_physical_place_insert_rejects_changed_board_generation_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(robot_task_runtime, "_TAUGHT_FUNCTIONS_ROOT", tmp_path)
    _place_recording(tmp_path)
    agent = _Agent(execution_mode="physical")
    agent._held_part = "MG"
    agent._current_state = "picked"
    _set_retained_held_part_handoff(agent)
    approach_result = asyncio.run(
        execute_robot_task(
            agent,
            "place_approach",
            destination_location="assembly_board-v1",
            part_name="MG",
        )
    )
    assert approach_result["status"] == "completed"
    _install_force_limited_place_insert_context(agent)
    agent._task_ctx["assembly_board_v1_aruco_generation"] = 8
    agent.primitive_calls.clear()

    result = asyncio.run(
        execute_robot_task(
            agent,
            "place_insert",
            destination_location="assembly_board-v1",
            part_name="MG",
        )
    )

    assert result["status"] == "failed"
    assert result["step"] == "place_insert.assembly_board_v1_aruco_generation_lock"
    assert "generation lock changed" in result["content"]
    assert result["observations"]["move_insert_dispatched"] is False
    assert agent.primitive_calls == []


@pytest.mark.parametrize(
    ("field", "replacement", "expected_message"),
    [
        (
            "accepted_assembly_board_v1_aruco_generation",
            8,
            "accepted generation changed after place_approach",
        ),
        (
            "accepted_assembly_board_v1_aruco_calibration_id",
            "replacement-calibration",
            "calibration identity changed after place_approach",
        ),
    ],
)
def test_physical_place_insert_rejects_reaccepted_or_recalibrated_board(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    replacement: object,
    expected_message: str,
) -> None:
    monkeypatch.setattr(robot_task_runtime, "_TAUGHT_FUNCTIONS_ROOT", tmp_path)
    _place_recording(tmp_path)
    agent = _Agent(execution_mode="physical")
    agent._held_part = "MG"
    agent._current_state = "picked"
    _set_retained_held_part_handoff(agent)
    approach_result = asyncio.run(
        execute_robot_task(
            agent,
            "place_approach",
            destination_location="assembly_board-v1",
            part_name="MG",
        )
    )
    assert approach_result["status"] == "completed"
    _install_force_limited_place_insert_context(agent)
    setattr(agent, field, replacement)
    agent.primitive_calls.clear()

    result = asyncio.run(
        execute_robot_task(
            agent,
            "place_insert",
            destination_location="assembly_board-v1",
            part_name="MG",
        )
    )

    assert result["status"] == "failed"
    assert result["step"] == "place_insert.assembly_board_v1_aruco_generation_lock"
    assert expected_message in result["content"]
    assert result["observations"]["move_insert_dispatched"] is False
    assert agent.primitive_calls == []


@pytest.mark.parametrize(
    ("field", "replacement", "expected_message"),
    [
        (
            "accepted_assembly_board_v1_aruco_generation",
            8,
            "accepted generation changed after place_approach",
        ),
        (
            "accepted_assembly_board_v1_aruco_calibration_id",
            "replacement-calibration",
            "calibration identity changed after place_approach",
        ),
    ],
)
def test_physical_place_insert_rechecks_board_lock_after_release_delay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    replacement: object,
    expected_message: str,
) -> None:
    monkeypatch.setattr(robot_task_runtime, "_TAUGHT_FUNCTIONS_ROOT", tmp_path)
    _place_recording(tmp_path)
    agent = _Agent(execution_mode="physical")
    agent._held_part = "MG"
    agent._current_state = "picked"
    _set_retained_held_part_handoff(agent)
    approach_result = asyncio.run(
        execute_robot_task(
            agent,
            "place_approach",
            destination_location="assembly_board-v1",
            part_name="MG",
        )
    )
    assert approach_result["status"] == "completed"
    _install_force_limited_place_insert_context(agent)
    execute_primitive = agent._execute_primitive

    async def change_board_lock_after_delay(
        primitive: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        result = await execute_primitive(primitive, params)
        if primitive == "move_insert":
            setattr(agent, field, replacement)
        return result

    agent._execute_primitive = change_board_lock_after_delay  # type: ignore[method-assign]
    agent.primitive_calls.clear()

    result = asyncio.run(
        execute_robot_task(
            agent,
            "place_insert",
            destination_location="assembly_board-v1",
            part_name="MG",
        )
    )

    assert result["status"] == "failed"
    assert result["step"] == "place_insert.assembly_board_v1_aruco_generation_lock"
    assert expected_message in result["content"]
    assert result["observations"]["move_insert_dispatched"] is True
    assert [primitive for primitive, _params in agent.primitive_calls] == [
        "move_insert",
    ]


def test_xarm6_independent_place_approach_applies_its_relative_correction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(robot_task_runtime, "_TAUGHT_FUNCTIONS_ROOT", tmp_path)
    agent = _Agent(execution_mode="physical", robot="xarm6")
    quaternion = {"qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0}
    _write_recording(
        tmp_path,
        function_name="place_approach",
        location="assembly_board-v1",
        part_name="MG",
        robot="xarm6",
        steps=[
            _relative_recorded_step(
                "move_above_destination",
                {"x": 0.5, "y": 0.6, "z": 0.5, **quaternion},
                relative_position_m={"x": 0.0, "y": 0.0, "z": 0.2},
                reference_kind="destination_target",
                reference_name="assembly_board-v1",
                reference_position_m={"x": 0.5, "y": 0.6, "z": 0.3},
                child_frame_id="link_eef",
            ),
            _relative_recorded_step(
                "descend",
                {"x": 0.5, "y": 0.6, "z": 0.3, **quaternion},
                relative_position_m={"x": 0.0, "y": 0.0, "z": 0.0},
                reference_kind="destination_target",
                reference_name="assembly_board-v1",
                reference_position_m={"x": 0.5, "y": 0.6, "z": 0.3},
                child_frame_id="link_eef",
            ),
        ],
    )

    result = asyncio.run(
        execute_robot_task(
            agent,
            "place_approach",
            robot_task_runtime._MANUAL_FUNCTION_EXECUTION_AUTHORITY,
            destination_location="assembly_board-v1",
            part_name="MG",
        )
    )

    assert result["status"] == "completed"
    move_calls = [
        params for primitive, params in agent.primitive_calls if primitive == "move_cartesian"
    ]
    assert move_calls == pytest.approx(
        [
            {"x": 0.5, "y": 0.6, "z": 0.5, "speed": None, **quaternion},
            {"x": 0.5, "y": 0.6, "z": 0.3, "speed": None, **quaternion},
        ]
    )


@pytest.mark.parametrize("task_name", ["place_approach", "place_insert"])
def test_xarm6_held_part_assembly_slot_blocks_before_approach_or_release(
    task_name: str,
) -> None:
    agent = _Agent(execution_mode="physical", robot="xarm6")
    agent._held_part = "MG"
    agent._current_state = "picked" if task_name == "place_approach" else "positioned"
    agent._gripper_state = "closed"
    agent._task_ctx = {
        "part_name": "MG",
        "destination_location": "assembly_board-v1",
        "surface_role": "assembly_slot",
    }

    result = asyncio.run(
        execute_robot_task(
            agent,
            task_name,
            destination_location="assembly_board-v1",
            part_name="MG",
        )
    )

    assert result == {
        "status": "blocked",
        "content": (
            "physical xarm6 assembly_board-v1 assembly_slot insertion with a held "
            "part is blocked: move_insert is available only for ur5e. Select ur5e "
            "for Assembly."
        ),
    }
    assert agent.primitive_calls == []
    assert agent._held_part == "MG"
    assert agent._gripper_state == "closed"


def test_xarm6_non_assembly_slot_keeps_legacy_release_without_move_insert() -> None:
    agent = _Agent(execution_mode="physical", robot="xarm6")
    agent._held_part = "MG"
    agent._current_state = "positioned"
    agent._gripper_state = "closed"
    agent._task_ctx = {
        "part_name": "MG",
        "destination_location": "inspection_station",
        "surface_role": "inspection_surface",
        "travel_z": 0.8,
    }

    result = asyncio.run(
        execute_robot_task(
            agent,
            "place_insert",
            destination_location="inspection_station",
            part_name="MG",
        )
    )

    assert result["status"] == "completed", result
    assert [primitive for primitive, _params in agent.primitive_calls] == [
        "release_part",
        "move_relative",
    ]


def _force_limited_place_insert_context(
    start_pose: dict[str, float],
    *,
    part_name: str = "MG",
) -> dict[str, Any]:
    model_name = _MOVE_INSERT_MODEL_MAP[part_name]
    hard_caps = {
        "insert_max_travel_m": 0.05,
        "insert_start_position_tolerance_m": 0.003,
        "insert_start_orientation_tolerance_rad": 0.1,
        "insert_max_timeout_sec": 30.0,
        "insert_max_contact_search_radius_m": 0.01,
    }
    hard_caps_sha256 = _move_insert_hard_caps_sha256(hard_caps)
    return {
        "part_name": part_name,
        "model_name": model_name,
        "origin_resource_location": "prusa-mk4-2",
        "destination_location": "assembly_board-v1",
        "travel_z": 0.8,
        "assembly_board_v1_aruco": {
            "destination_location": "assembly_board-v1",
            "camera_role": "ur5e",
            "generation": 7,
            "calibration_id": "ur5e-calibration",
            "captured_at": time.time(),
            "frame_id": "world",
            "pose": {
                "x": 0.5,
                "y": 0.6,
                "z": 0.3,
                "qx": 0.0,
                "qy": 0.0,
                "qz": 0.0,
                "qw": 1.0,
            },
        },
        "assembly_board_v1_aruco_generation": 7,
        "resolved_cartesian_positions": {"descend": deepcopy(start_pose)},
        "insert_pose": {**start_pose, "z": 0.30},
        "insertion_axis_world": {"x": 0.0, "y": 0.0, "z": -1.0},
        "move_insert_profile_sha256": "a" * 64,
        "move_insert_mode": "force_limited",
        "move_insert_timeout_sec": 9.0,
        "move_insert_boundary_ready": True,
        "move_insert_boundary_error": "",
        "move_insert_hard_caps": hard_caps,
        "move_insert_hard_caps_sha256": hard_caps_sha256,
        "held_part_handoff": {
            "part_name": part_name,
            "model_name": model_name,
            "origin_resource_location": "prusa-mk4-2",
            "frame_id": "world",
            "tool_frame": "tool0",
            "part_frame": "held_part_origin",
            "source": "pick_grasp",
            "world_tool0_pose_at_grasp": {
                "x": 0.5,
                "y": 0.6,
                "z": 0.51,
                "qx": 0.0,
                "qy": 0.0,
                "qz": 0.0,
                "qw": 1.0,
            },
            "world_held_part_pose_at_grasp": {
                "x": 0.5,
                "y": 0.6,
                "z": 0.31,
                "qx": 0.0,
                "qy": 0.0,
                "qz": 0.0,
                "qw": 1.0,
            },
            "tool0_to_held_part": {
                "x": 0.0,
                "y": 0.0,
                "z": -0.2,
                "qx": 0.0,
                "qy": 0.0,
                "qz": 0.0,
                "qw": 1.0,
            },
            "origin_pose_provenance": {
                "frame_id": "world",
                "part_name": part_name,
                "model_name": model_name,
                "source": "live_detection",
            },
        },
        "move_insert_profile": {
            "part_name": part_name,
            "calibration_id": "insert-calibration-1",
            "shared_calibration_id": "insert-calibration-1",
            "override_calibration_id": None,
            "profile_sha256": "a" * 64,
            "pre_insert_offset_m": 0.01,
            "contact_speed_m_s": 0.002,
            "contact_force_delta_n": 2.0,
            "engagement_progress_m": 0.003,
            "insertion_force_n": 5.0,
            "spiral_radius_m": 0.001,
            "spiral_pitch_m": 0.0005,
            "spiral_speed_m_s": 0.002,
            "spiral_acceleration_m_s2": 0.02,
            "max_axial_force_n": 20.0,
            "max_lateral_force_n": 10.0,
            "max_torque_nm": 2.0,
            "force_depth_profile": _force_depth_profile(),
            "tilt_tolerance_rad": 0.1,
            "seated_depth_tolerance_m": 0.001,
            "settle_time_sec": 0.2,
            "demonstration_recipe": {
                "hard_caps_sha256": hard_caps_sha256,
            },
        },
    }


def _install_force_limited_place_insert_context(agent: _Agent) -> None:
    retained_context = deepcopy(agent._task_ctx)
    start_pose = deepcopy(
        dict(retained_context.get("resolved_cartesian_positions") or {}).get(
            "descend"
        )
    )
    force_context = _force_limited_place_insert_context(start_pose)
    for field in (
        "assembly_board_v1_aruco",
        "assembly_board_v1_aruco_generation",
        "held_part_handoff",
        "part_name",
        "model_name",
        "origin_resource_location",
        "travel_z",
    ):
        if field in retained_context:
            force_context[field] = deepcopy(retained_context[field])
    agent._task_ctx = force_context
    agent._gripper_state = "closed"


@pytest.mark.parametrize(
    "move_insert_mode",
    [None, "", "simulation_direct", "stale_force_limited"],
)
def test_normal_physical_assembly_slot_never_releases_without_force_limited_mode(
    move_insert_mode: str | None,
) -> None:
    agent = _Agent(execution_mode="physical")
    agent._held_part = "MG"
    agent._current_state = "positioned"
    agent._gripper_state = "closed"
    start_pose = {
        "x": 0.5,
        "y": 0.6,
        "z": 0.31,
        "qx": 0.0,
        "qy": 0.0,
        "qz": 0.0,
        "qw": 1.0,
    }
    agent._position = deepcopy(start_pose)
    agent._task_ctx = _force_limited_place_insert_context(start_pose)
    if move_insert_mode is None:
        agent._task_ctx.pop("move_insert_mode")
    else:
        agent._task_ctx["move_insert_mode"] = move_insert_mode

    result = asyncio.run(
        execute_robot_task(
            agent,
            "place_insert",
            destination_location="assembly_board-v1",
            part_name="MG",
        )
    )

    assert result["status"] == "blocked"
    assert "cannot release or lift" in result["content"]
    assert agent.primitive_calls == []
    assert agent._current_state == "positioned"
    assert agent._held_part == "MG"
    assert agent._gripper_state == "closed"


@pytest.mark.parametrize(
    ("mutate_context", "expected_message"),
    [
        (
            lambda context: context.update(
                {
                    "move_insert_boundary_ready": False,
                    "move_insert_boundary_error": "fresh hard caps are unavailable",
                }
            ),
            "move_insert boundary is not ready",
        ),
        (
            lambda context: context["move_insert_profile"].update(
                {"part_name": "LG"}
            ),
            "profile part identity is invalid",
        ),
        (
            lambda context: context["held_part_handoff"][
                "tool0_to_held_part"
            ].update({"z": -0.1}),
            "does not reconstruct its frozen pick poses",
        ),
    ],
)
def test_normal_physical_assembly_slot_revalidates_retained_insertion_context(
    mutate_context: Any,
    expected_message: str,
) -> None:
    agent = _Agent(execution_mode="physical")
    agent._held_part = "MG"
    agent._current_state = "positioned"
    agent._gripper_state = "closed"
    start_pose = {
        "x": 0.5,
        "y": 0.6,
        "z": 0.31,
        "qx": 0.0,
        "qy": 0.0,
        "qz": 0.0,
        "qw": 1.0,
    }
    agent._position = deepcopy(start_pose)
    agent._task_ctx = _force_limited_place_insert_context(start_pose)
    mutate_context(agent._task_ctx)

    result = asyncio.run(
        execute_robot_task(
            agent,
            "place_insert",
            destination_location="assembly_board-v1",
            part_name="MG",
        )
    )

    assert result["status"] == "failed"
    assert result["step"] == "place_insert.assembly_board_v1_aruco_generation_lock"
    assert expected_message in result["content"]
    assert agent.primitive_calls == []
    assert agent._current_state == "positioned"
    assert agent._held_part == "MG"
    assert agent._gripper_state == "closed"


def test_move_insert_dispatch_derives_hard_caps_sha256_from_retained_caps() -> None:
    agent = _Agent(execution_mode="physical")
    agent._held_part = "MG"
    agent._current_state = "positioned"
    agent._gripper_state = "closed"
    start_pose = {
        "x": 0.5,
        "y": 0.6,
        "z": 0.31,
        "qx": 0.0,
        "qy": 0.0,
        "qz": 0.0,
        "qw": 1.0,
    }
    agent._position = deepcopy(start_pose)
    agent._task_ctx = _force_limited_place_insert_context(start_pose)
    expected_hard_caps_sha256 = _move_insert_hard_caps_sha256(
        agent._task_ctx["move_insert_hard_caps"]
    )
    agent._task_ctx["move_insert_hard_caps_sha256"] = "0" * 64
    agent._task_ctx["move_insert_profile"]["demonstration_recipe"][
        "hard_caps_sha256"
    ] = "1" * 64
    result = asyncio.run(
        robot_task_runtime.execute_place_insert_move_insert_trial(
            agent,
            destination_location="assembly_board-v1",
            part_name="MG",
            trial_id="move-insert-test",
        )
    )

    assert result["status"] == "completed", result
    assert [primitive for primitive, _params in agent.primitive_calls] == [
        "move_insert"
    ]
    assert agent.primitive_calls[0][1]["hard_caps_sha256"] == (
        expected_hard_caps_sha256
    )
    assert agent._task_ctx["move_insert_hard_caps_sha256"] == (
        expected_hard_caps_sha256
    )


@pytest.mark.parametrize("part_name", ["SG", "MG", "LG", "SCP", "MCP", "LCP"])
def test_supervised_move_insert_trial_is_exact_part_neutral_and_runs_no_suffix(
    part_name: str,
) -> None:
    agent = _Agent(execution_mode="physical")
    agent._held_part = part_name
    agent._current_state = "positioned"
    agent._gripper_state = "closed"
    start_pose = {
        "x": 0.5,
        "y": 0.6,
        "z": 0.31,
        "qx": 0.0,
        "qy": 0.0,
        "qz": 0.0,
        "qw": 1.0,
    }
    agent._position = deepcopy(start_pose)
    agent._task_ctx = _force_limited_place_insert_context(
        start_pose,
        part_name=part_name,
    )
    agent._task_ctx["move_insert_mode"] = "force_limited_trial"

    result = asyncio.run(
        robot_task_runtime.execute_place_insert_move_insert_trial(
            agent,
            destination_location="assembly_board-v1",
            part_name=part_name,
            trial_id="move-insert-test",
        )
    )

    assert result["status"] == "completed", result
    assert result["trial_id"] == "move-insert-test"
    assert result["trial_ready_for_confirmation"] is True
    assert [primitive for primitive, _params in agent.primitive_calls] == [
        "move_insert"
    ]
    assert agent._current_state == "positioned"
    assert agent._held_part == part_name
    assert agent._gripper_state == "closed"
    assert "move_insert_trial_id" not in agent._task_ctx
    assert agent.primitive_calls[0][1]["trial_id"] == "move-insert-test"
    assert agent._task_ctx["move_insert_trial_result"]["seated_detected"] is True
    assert agent._task_ctx["move_insert_trial_result"]["feedback_trace"] == [
        {
            "timestamp": 1.0,
            "phase": "settling",
            "insertion_depth_m": 0.01,
            "actual_tcp_force": [0.0, 0.0, 5.0, 0.0, 0.0, 0.1],
            "actual_tool0_velocity": {
                "vx": 0.0,
                "vy": 0.0,
                "vz": 0.0,
                "wx": 0.0,
                "wy": 0.0,
                "wz": 0.0,
            },
            "engagement_detected": True,
            "seated_detected": True,
        }
    ]


def test_supervised_move_insert_trial_rejects_missing_trial_id_before_motion() -> None:
    agent = _Agent(execution_mode="physical")
    agent._held_part = "MG"
    agent._current_state = "positioned"
    agent._gripper_state = "closed"

    result = asyncio.run(
        robot_task_runtime.execute_place_insert_move_insert_trial(
            agent,
            destination_location="assembly_board-v1",
            part_name="MG",
            trial_id="",
        )
    )

    assert result == {
        "status": "blocked",
        "content": "Supervised Test move_insert trial_id is missing or invalid.",
    }
    assert agent.primitive_calls == []


def test_confirm_move_insert_trial_rejects_unsettled_result_before_release() -> None:
    agent = _Agent(execution_mode="physical")
    agent._held_part = "MG"
    agent._current_state = "positioned"
    agent._gripper_state = "closed"
    start_pose = {
        "x": 0.5,
        "y": 0.6,
        "z": 0.31,
        "qx": 0.0,
        "qy": 0.0,
        "qz": 0.0,
        "qw": 1.0,
    }
    agent._position = deepcopy(start_pose)
    agent._task_ctx = _force_limited_place_insert_context(start_pose)
    trial = asyncio.run(
        robot_task_runtime.execute_place_insert_move_insert_trial(
            agent,
            destination_location="assembly_board-v1",
            part_name="MG",
            trial_id="move-insert-test",
        )
    )
    retained_result = deepcopy(agent._task_ctx["move_insert_trial_result"])
    retained_result["motion_settled"] = False
    retained_sha256 = robot_task_runtime._move_insert_result_sha256(
        retained_result
    )
    agent._task_ctx["move_insert_trial_result"] = retained_result
    agent._task_ctx["move_insert_trial_result_sha256"] = retained_sha256
    agent.primitive_calls.clear()

    result = asyncio.run(
        robot_task_runtime.complete_place_insert_after_move_insert_trial(
            agent,
            destination_location="assembly_board-v1",
            part_name="MG",
            expected_move_insert_result_sha256=retained_sha256,
        )
    )

    assert trial["status"] == "completed"
    assert result["status"] == "blocked"
    assert "stationary settlement evidence" in result["content"]
    assert agent.primitive_calls == []
    assert agent._held_part == "MG"
    assert agent._gripper_state == "closed"


@pytest.mark.parametrize("move_insert_mode", ["", "simulation_direct"])
def test_supervised_move_insert_trial_rejects_context_without_physical_trial_mode(
    move_insert_mode: str,
) -> None:
    agent = _Agent(execution_mode="physical")
    agent._held_part = "MG"
    agent._current_state = "positioned"
    agent._gripper_state = "closed"
    start_pose = {
        "x": 0.5,
        "y": 0.6,
        "z": 0.31,
        "qx": 0.0,
        "qy": 0.0,
        "qz": 0.0,
        "qw": 1.0,
    }
    agent._task_ctx = _force_limited_place_insert_context(start_pose)
    agent._task_ctx["move_insert_mode"] = move_insert_mode

    trial = asyncio.run(
        robot_task_runtime.execute_place_insert_move_insert_trial(
            agent,
            destination_location="assembly_board-v1",
            part_name="MG",
            trial_id="move-insert-test",
        )
    )
    completion = asyncio.run(
        robot_task_runtime.complete_place_insert_after_move_insert_trial(
            agent,
            destination_location="assembly_board-v1",
            part_name="MG",
            expected_move_insert_result_sha256="a" * 64,
        )
    )

    assert trial["status"] == "blocked"
    assert trial["trial_id"] == "move-insert-test"
    assert trial["motion_settled"] is True
    assert trial["dispatch_attempted"] is False
    assert trial["move_insert_result"] == {
        "success": False,
        "trial_id": "move-insert-test",
        "state_uncertain": False,
        "motion_settled": True,
        "dispatch_attempted": False,
    }
    assert completion["status"] == "blocked"
    assert "physical move_insert trial profile" in trial["content"]
    assert agent.primitive_calls == []


def test_place_approach_recording_qualification_downgrades_only_after_change(
    tmp_path: Path,
) -> None:
    recording_path = tmp_path / "default__hardware.json"
    recording_path.write_bytes(b'{"recording":"confirmed"}\n')
    recording_sha256 = hashlib.sha256(recording_path.read_bytes()).hexdigest()
    targets = {
        "move_insert_mode": "force_limited",
        "move_insert_profile": {
            "qualification": {
                "place_approach_recording_sha256": recording_sha256,
            }
        },
    }

    current, current_error = (
        robot_task_runtime._apply_place_approach_recording_qualification(
            targets,
            recording_path,
        )
    )
    recording_path.write_bytes(b'{"recording":"changed"}\n')
    stale, stale_error = (
        robot_task_runtime._apply_place_approach_recording_qualification(
            targets,
            recording_path,
        )
    )

    assert current_error == ""
    assert current["move_insert_mode"] == "force_limited"
    assert current["move_insert_profile"]["qualification"]
    assert stale_error == ""
    assert stale["move_insert_mode"] == "force_limited_trial"
    assert stale["move_insert_profile"]["qualification"] == {}
    assert "recording changed" in stale["move_insert_qualification_error"]
    assert targets["move_insert_mode"] == "force_limited"


def test_normal_place_insert_cannot_use_unqualified_trial_profile() -> None:
    agent = _Agent(execution_mode="physical")
    agent._held_part = "MG"
    agent._current_state = "positioned"
    agent._gripper_state = "closed"
    start_pose = {
        "x": 0.5,
        "y": 0.6,
        "z": 0.31,
        "qx": 0.0,
        "qy": 0.0,
        "qz": 0.0,
        "qw": 1.0,
    }
    agent._position = deepcopy(start_pose)
    agent._task_ctx = _force_limited_place_insert_context(start_pose)
    agent._task_ctx["move_insert_mode"] = "force_limited_trial"

    result = asyncio.run(
        robot_task_runtime.execute_robot_task(
            agent,
            "place_insert",
            robot_task_runtime._MANUAL_FUNCTION_EXECUTION_AUTHORITY,
            destination_location="assembly_board-v1",
            part_name="MG",
        )
    )

    assert result["status"] == "blocked"
    assert "Supervised Test move_insert and Confirm Completion" in result["content"]
    assert agent.primitive_calls == []
    assert agent._current_state == "positioned"
    assert agent._held_part == "MG"
    assert agent._gripper_state == "closed"


@pytest.mark.parametrize("part_name", list(_MOVE_INSERT_MODEL_MAP))
def test_confirm_move_insert_trial_runs_suffix_once_without_move_insert(
    part_name: str,
) -> None:
    agent = _Agent(execution_mode="physical")
    agent._held_part = part_name
    agent._current_state = "positioned"
    agent._gripper_state = "closed"
    start_pose = {
        "x": 0.5,
        "y": 0.6,
        "z": 0.31,
        "qx": 0.0,
        "qy": 0.0,
        "qz": 0.0,
        "qw": 1.0,
    }
    insert_pose = {**start_pose, "z": 0.30}
    agent._position = deepcopy(start_pose)
    agent._task_ctx = _force_limited_place_insert_context(
        start_pose,
        part_name=part_name,
    )
    trial = asyncio.run(
        robot_task_runtime.execute_place_insert_move_insert_trial(
            agent,
            destination_location="assembly_board-v1",
            part_name=part_name,
            trial_id="move-insert-test",
        )
    )
    agent._get_ee_pose = lambda: SimpleNamespace(
        position=SimpleNamespace(
            x=insert_pose["x"],
            y=insert_pose["y"],
            z=insert_pose["z"],
        ),
        orientation=SimpleNamespace(
            x=insert_pose["qx"],
            y=insert_pose["qy"],
            z=insert_pose["qz"],
            w=insert_pose["qw"],
        ),
    )
    agent.primitive_calls.clear()

    result = asyncio.run(
        robot_task_runtime.complete_place_insert_after_move_insert_trial(
            agent,
            destination_location="assembly_board-v1",
            part_name=part_name,
            expected_move_insert_result_sha256=trial[
                "move_insert_result_sha256"
            ],
        )
    )

    assert result["status"] == "completed", result
    primitives = [primitive for primitive, _params in agent.primitive_calls]
    assert primitives == ["release_part", "move_relative"]
    assert "move_insert" not in primitives
    assert agent._current_state == "placed"
    assert agent._held_part is None
    assert agent._gripper_state == "open"


def test_confirm_move_insert_trial_commits_irreversible_release_before_lift_failure() -> None:
    agent = _Agent(execution_mode="physical")
    agent._held_part = "MG"
    agent._current_state = "positioned"
    agent._gripper_state = "closed"
    start_pose = {
        "x": 0.5,
        "y": 0.6,
        "z": 0.31,
        "qx": 0.0,
        "qy": 0.0,
        "qz": 0.0,
        "qw": 1.0,
    }
    insert_pose = {**start_pose, "z": 0.30}
    agent._position = deepcopy(start_pose)
    agent._task_ctx = _force_limited_place_insert_context(start_pose)
    trial = asyncio.run(
        robot_task_runtime.execute_place_insert_move_insert_trial(
            agent,
            destination_location="assembly_board-v1",
            part_name="MG",
            trial_id="move-insert-test",
        )
    )
    agent._get_ee_pose = lambda: SimpleNamespace(
        position=SimpleNamespace(
            x=insert_pose["x"],
            y=insert_pose["y"],
            z=insert_pose["z"],
        ),
        orientation=SimpleNamespace(
            x=insert_pose["qx"],
            y=insert_pose["qy"],
            z=insert_pose["qz"],
            w=insert_pose["qw"],
        ),
    )
    execute_primitive = agent._execute_primitive

    async def fail_lift(
        primitive: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        if primitive == "move_relative":
            agent.primitive_calls.append((primitive, deepcopy(params)))
            return {"success": False, "message": "lift failed after release"}
        return await execute_primitive(primitive, params)

    agent._execute_primitive = fail_lift  # type: ignore[method-assign]
    agent.primitive_calls.clear()

    result = asyncio.run(
        robot_task_runtime.complete_place_insert_after_move_insert_trial(
            agent,
            destination_location="assembly_board-v1",
            part_name="MG",
            expected_move_insert_result_sha256=trial[
                "move_insert_result_sha256"
            ],
        )
    )

    assert result["status"] == "failed"
    assert result["completed_steps"] == ["release_part"]
    assert agent._held_part is None
    assert agent._gripper_state == "open"
    assert agent._current_state == "positioned"
    assert agent._task_ctx
    assert "move_insert" not in [
        primitive for primitive, _params in agent.primitive_calls
    ]


def test_normal_place_insert_commits_irreversible_release_before_lift_failure() -> None:
    agent = _Agent(execution_mode="physical")
    agent._held_part = "MG"
    agent._current_state = "positioned"
    agent._gripper_state = "closed"
    start_pose = {
        "x": 0.5,
        "y": 0.6,
        "z": 0.31,
        "qx": 0.0,
        "qy": 0.0,
        "qz": 0.0,
        "qw": 1.0,
    }
    insert_pose = {**start_pose, "z": 0.30}
    agent._position = deepcopy(start_pose)
    agent._task_ctx = _force_limited_place_insert_context(start_pose)
    agent._get_ee_pose = lambda: SimpleNamespace(
        position=SimpleNamespace(
            x=insert_pose["x"],
            y=insert_pose["y"],
            z=insert_pose["z"],
        ),
        orientation=SimpleNamespace(
            x=insert_pose["qx"],
            y=insert_pose["qy"],
            z=insert_pose["qz"],
            w=insert_pose["qw"],
        ),
    )
    execute_primitive = agent._execute_primitive

    async def fail_lift(
        primitive: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        if primitive == "move_relative":
            agent.primitive_calls.append((primitive, deepcopy(params)))
            return {"success": False, "message": "lift failed after release"}
        return await execute_primitive(primitive, params)

    agent._execute_primitive = fail_lift  # type: ignore[method-assign]

    result = asyncio.run(
        execute_robot_task(
            agent,
            "place_insert",
            destination_location="assembly_board-v1",
            part_name="MG",
        )
    )

    assert result["status"] == "failed"
    assert result["step"] == "place_insert.lift"
    assert "release_part already completed" in result["content"]
    assert [primitive for primitive, _params in agent.primitive_calls] == [
        "move_insert",
        "release_part",
        "move_relative",
    ]
    assert agent._held_part is None
    assert agent._gripper_state == "open"
    assert agent._current_state == "positioned"
    assert agent._task_ctx


@pytest.mark.parametrize("part_name", list(_MOVE_INSERT_MODEL_MAP))
def test_move_insert_failure_stops_before_release_and_preserves_held_state(
    part_name: str,
) -> None:
    agent = _Agent(execution_mode="physical")
    agent._held_part = part_name
    agent._current_state = "positioned"
    agent._gripper_state = "closed"
    start_pose = {
        "x": 0.5,
        "y": 0.6,
        "z": 0.31,
        "qx": 0.0,
        "qy": 0.0,
        "qz": 0.0,
        "qw": 1.0,
    }
    agent._position = deepcopy(start_pose)
    agent._task_ctx = _force_limited_place_insert_context(
        start_pose,
        part_name=part_name,
    )
    execute_primitive = agent._execute_primitive

    async def fail_move_insert(
        primitive: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        if primitive == "move_insert":
            agent.primitive_calls.append((primitive, deepcopy(params)))
            return {
                "success": False,
                "message": "spiral search exhausted",
                "state_uncertain": False,
            }
        return await execute_primitive(primitive, params)

    agent._execute_primitive = fail_move_insert  # type: ignore[method-assign]

    result = asyncio.run(
        execute_robot_task(
            agent,
            "place_insert",
            destination_location="assembly_board-v1",
            part_name=part_name,
        )
    )

    assert result["status"] == "failed"
    assert result["step"] == "place_insert.move_insert"
    assert "release_part and lift were not commanded" in result["content"]
    assert [primitive for primitive, _params in agent.primitive_calls] == [
        "move_insert"
    ]
    assert agent._current_state == "positioned"
    assert agent._held_part == part_name
    assert agent._gripper_state == "closed"
    assert result["observations"]["move_insert_result"] == {
        "success": False,
        "message": "spiral search exhausted",
        "state_uncertain": False,
    }


def test_move_insert_revalidates_matching_fresh_pose_before_release() -> None:
    agent = _Agent(execution_mode="physical")
    agent._held_part = "MG"
    agent._current_state = "positioned"
    agent._gripper_state = "closed"
    start_pose = {
        "x": 0.5,
        "y": 0.6,
        "z": 0.31,
        "qx": 0.0,
        "qy": 0.0,
        "qz": 0.0,
        "qw": 1.0,
    }
    insert_pose = {**start_pose, "z": 0.30}
    agent._position = deepcopy(start_pose)
    agent._task_ctx = _force_limited_place_insert_context(start_pose)
    agent._get_ee_pose = lambda: SimpleNamespace(
        position=SimpleNamespace(
            x=insert_pose["x"],
            y=insert_pose["y"],
            z=insert_pose["z"],
        ),
        orientation=SimpleNamespace(
            x=insert_pose["qx"],
            y=insert_pose["qy"],
            z=insert_pose["qz"],
            w=insert_pose["qw"],
        ),
    )

    result = asyncio.run(
        execute_robot_task(
            agent,
            "place_insert",
            destination_location="assembly_board-v1",
            part_name="MG",
        )
    )

    assert result["status"] == "completed", result
    assert [primitive for primitive, _params in agent.primitive_calls] == [
        "move_insert",
        "release_part",
        "move_relative",
    ]
    assert agent._current_state == "placed"
    assert agent._held_part is None
    assert agent._gripper_state == "open"


def test_move_insert_blocks_release_when_fresh_pose_moved_outside_tolerance() -> None:
    agent = _Agent(execution_mode="physical")
    agent._held_part = "MG"
    agent._current_state = "positioned"
    agent._gripper_state = "closed"
    start_pose = {
        "x": 0.5,
        "y": 0.6,
        "z": 0.31,
        "qx": 0.0,
        "qy": 0.0,
        "qz": 0.0,
        "qw": 1.0,
    }
    insert_pose = {**start_pose, "z": 0.30}
    agent._position = deepcopy(start_pose)
    agent._task_ctx = _force_limited_place_insert_context(start_pose)
    agent._get_ee_pose = lambda: SimpleNamespace(
        position=SimpleNamespace(
            x=insert_pose["x"] + 0.002,
            y=insert_pose["y"],
            z=insert_pose["z"],
        ),
        orientation=SimpleNamespace(
            x=insert_pose["qx"],
            y=insert_pose["qy"],
            z=insert_pose["qz"],
            w=insert_pose["qw"],
        ),
    )

    result = asyncio.run(
        execute_robot_task(
            agent,
            "place_insert",
            destination_location="assembly_board-v1",
            part_name="MG",
        )
    )

    assert result["status"] == "failed"
    assert result["step"] == "place_insert.assembly_board_v1_aruco_generation_lock"
    assert "moved outside seated tolerances" in result["content"]
    assert [primitive for primitive, _params in agent.primitive_calls] == [
        "move_insert",
    ]
    assert agent._current_state == "positioned"
    assert agent._held_part == "MG"
    assert agent._gripper_state == "closed"
    assert agent._position == pytest.approx(insert_pose)
    assert agent._task_ctx["resolved_cartesian_positions"][
        "move_insert"
    ] == pytest.approx(insert_pose)
    assert agent._task_ctx["move_insert_result"]["success"] is True
    assert agent._task_ctx["move_insert_result"]["engagement_detected"] is True
    assert agent._task_ctx["move_insert_result"]["seated_detected"] is True


def test_simulation_assembly_slot_dispatches_simulation_direct_before_release() -> None:
    agent = _Agent(execution_mode="simulation")
    agent._held_part = "MG"
    agent._current_state = "positioned"
    agent._gripper_state = "closed"
    pre_insert_pose = {
        "x": 0.5,
        "y": 0.6,
        "z": 0.31,
        "qx": 0.0,
        "qy": 0.0,
        "qz": 0.0,
        "qw": 1.0,
    }
    insert_pose = {**pre_insert_pose, "z": 0.30}
    agent._task_ctx = {
        "part_name": "MG",
        "destination_location": "assembly_board-v1",
        "surface_role": "assembly_slot",
        "move_insert_mode": "simulation_direct",
        "resolved_cartesian_positions": {"descend": pre_insert_pose},
        "insert_pose": insert_pose,
        "travel_z": 0.8,
        "model_name": "gear_medium",
        "slot_x": 0.5,
        "slot_y": 0.6,
        "part_height": 0.02,
        "board_top_z": 0.29,
        "place_part_origin_z": 0.30,
    }

    result = asyncio.run(
        execute_robot_task(
            agent,
            "place_insert",
            destination_location="assembly_board-v1",
            part_name="MG",
        )
    )

    assert result["status"] == "completed"
    assert [primitive for primitive, _params in agent.primitive_calls] == [
        "move_insert",
        "release_part",
        "snap_part_to_slot",
        "move_relative",
    ]
