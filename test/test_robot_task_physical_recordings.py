"""Focused tests for registry-backed physical Cartesian position recordings."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from cais_spade_llm.resources.robot import robot_task_runtime
from cais_spade_llm.resources.robot.robot_tasks import (
    execute_robot_task,
    robot_task_names,
    robot_task_registry,
)


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


class _Logger:
    def warning(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def error(self, *_args: Any, **_kwargs: Any) -> None:
        return None


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
    path = root / robot / function_name / f"{location}__{part_name}__hardware.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "robot": robot,
                "function_name": function_name,
                "name": location,
                "part_name": part_name,
                "capture_source": "hardware",
                "steps": steps,
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
        "x": -0.11,
        "y": 0.42,
        "z": 0.69,
        "qx": 0.0,
        "qy": 0.70710678,
        "qz": 0.0,
        "qw": 0.70710678,
    }
    target = {
        "x": -0.11,
        "y": 0.42,
        "z": 0.36,
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
                relative_position_m={"x": -0.61, "y": -0.18, "z": 0.39},
                reference_kind="destination_target",
                reference_name="assembly_board-v1",
                reference_position_m={"x": 0.5, "y": 0.6, "z": 0.3},
            ),
            _relative_recorded_step(
                "descend",
                target,
                relative_position_m={"x": -0.61, "y": -0.18, "z": 0.06},
                reference_kind="destination_target",
                reference_name="assembly_board-v1",
                reference_position_m={"x": 0.5, "y": 0.6, "z": 0.3},
            ),
        ],
    )
    return above, target


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
            ("delay_after_grasp", "delay"),
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
            ("delay_before_release", "delay"),
            ("release_part", "release_part"),
            ("delay_after_release", "delay"),
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
    } == {
        ("place_approach", "move_above_destination"),
        ("place_approach", "descend"),
    }


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

    assert result["status"] == "completed"
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
    assert move_calls[1] == pytest.approx({"x": 0.1, "y": 0.2, "z": 0.4})
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
        "delay",
        "move_relative",
    ]
    assert agent.primitive_calls[0][1]["position"] == pytest.approx(0.37)
    lift_params = agent.primitive_calls[-1][1]
    assert lift_params["dz"] == pytest.approx(0.4)


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
        relative_position_m={"x": 0.02, "y": -0.02, "z": 0.101},
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
        relative_position_m={"x": 0.05, "y": -0.02, "z": 0.11},
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
        relative_position_m={"x": 0.05, "y": -0.02, "z": 0.11},
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
            "z": 0.43,
            **captured_quaternion,
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
        relative_position_m={"x": 0.0, "y": 0.0, "z": 0.11},
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
        relative_position_m={"x": 1.0, "y": 0.0, "z": 0.1},
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
        relative_position_m={"x": 0.0, "y": 0.0, "z": 0.105},
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


def test_current_prusa_mk4_2_mg_recording_allows_cleared_descend() -> None:
    recording_path = (
        robot_task_runtime._TAUGHT_FUNCTIONS_ROOT
        / "ur5e"
        / "pick_approach"
        / "prusa-mk4-2__MG__hardware.json"
    )
    recording = json.loads(recording_path.read_text(encoding="utf-8"))
    recorded_by_name = {
        str(step["step_name"]): dict(step) for step in recording["steps"]
    }
    assert set(
        dict(recorded_by_name["move_above_part"]["position_sources"]).values()
    ) == {"computed"}
    assert "descend" not in recorded_by_name
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
    assert {
        axis: move_calls[1][axis] for axis in ("x", "y", "z")
    } == pytest.approx(
        {
            "x": 0.1,
            "y": 0.2,
            "z": 0.4,
        }
    )
    recorded_pose = dict(recorded_by_name["move_above_part"]["waypoint"]["pose"])
    assert {
        axis: move_calls[0][axis] for axis in ("qx", "qy", "qz", "qw")
    } == pytest.approx(
        {axis: recorded_pose[axis] for axis in ("qx", "qy", "qz", "qw")}
    )
    assert agent._task_ctx["finger_tooth_clearance_m"] == pytest.approx(0.002)
    assert agent._task_ctx["finger_hub_overlap_m"] == pytest.approx(0.008)


def test_physical_mg_captured_z_is_rejected_before_any_primitive(
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
    assert move_calls[-1]["z"] == pytest.approx(0.398)
    assert {
        axis: move_calls[-1][axis] for axis in ("qx", "qy", "qz", "qw")
    } == pytest.approx({"qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0})


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
        relative_position_m={"x": 0.0, "y": 0.0, "z": 0.1},
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
        "delay",
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

    assert insert_result["status"] == "completed"
    assert agent._current_state == "placed"
    assert [primitive for primitive, _params in agent.primitive_calls] == [
        "delay",
        "release_part",
        "delay",
        "move_relative",
    ]


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


def test_physical_place_preflight_fails_before_any_primitive_for_missing_recording(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(robot_task_runtime, "_TAUGHT_FUNCTIONS_ROOT", tmp_path)
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
    assert "physical position file not found" in result["content"]
    assert "assembly_board-v1__MG__hardware.json" in result["content"]
    assert result["step"] == "place_approach.physical_position_preflight"
    assert agent.primitive_calls == []


def test_physical_place_does_not_fall_back_to_another_parts_recording(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(robot_task_runtime, "_TAUGHT_FUNCTIONS_ROOT", tmp_path)
    _place_recording(tmp_path, part_name="MG")
    agent = _Agent(execution_mode="physical")
    agent._held_part = "LG"
    agent._current_state = "picked"

    result = asyncio.run(
        execute_robot_task(
            agent,
            "place_approach",
            destination_location="assembly_board-v1",
            part_name="LG",
        )
    )

    assert result["status"] == "failed"
    assert "assembly_board-v1__LG__hardware.json" in result["content"]
    assert agent.primitive_calls == []


def test_physical_place_rejects_payload_part_name_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(robot_task_runtime, "_TAUGHT_FUNCTIONS_ROOT", tmp_path)
    _place_recording(tmp_path, part_name="MG")
    path = tmp_path / "ur5e" / "place_approach" / "assembly_board-v1__MG__hardware.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["part_name"] = "LG"
    path.write_text(json.dumps(payload), encoding="utf-8")
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
    assert "physical position part_name mismatch" in result["content"]
    assert agent.primitive_calls == []


def test_physical_place_threads_recorded_height_into_unrecorded_place_insert_lift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(robot_task_runtime, "_TAUGHT_FUNCTIONS_ROOT", tmp_path)
    above, target = _place_recording(tmp_path)
    agent = _Agent(execution_mode="physical")
    agent._held_part = "MG"
    agent._current_state = "picked"
    agent._position = {"x": 0.2, "y": -0.3, "z": 0.7}
    agent._task_ctx = {
        "part_name": "MG",
        "model_name": "gear_medium",
        "origin_resource_location": "prusa-mk4-2",
        "travel_z": 0.7,
    }

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
    assert agent._task_ctx["travel_z"] == pytest.approx(above["z"])
    assert agent._position == pytest.approx({"x": target["x"], "y": target["y"], "z": target["z"]})

    agent.primitive_calls.clear()
    insert_result = asyncio.run(
        execute_robot_task(
            agent,
            "place_insert",
            destination_location="assembly_board-v1",
            part_name="MG",
        )
    )
    assert insert_result["status"] == "completed"
    assert [primitive for primitive, _params in agent.primitive_calls] == [
        "delay",
        "release_part",
        "delay",
        "move_relative",
    ]
    assert agent.primitive_calls[-1][1]["dz"] == pytest.approx(above["z"] - target["z"])


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
            "physical position joints are incomplete",
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
            lambda steps: steps[0]["relative_reference"].update({"name": "wrong-location"}),
            "relative_reference.name mismatch",
        ),
        (
            lambda steps: steps[0]["relative_reference"].update({"frame_id": "base"}),
            "relative_reference.frame_id must be world",
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
            lambda steps: steps[0]["relative_pose"].update({"x": 0.0}),
            "relative_pose translation does not reconstruct the captured pose",
        ),
        (
            lambda steps: steps[0]["relative_reference"].update(
                {"source": "computed_destination"}
            ),
            "relative_reference.source must be assembly_board-v1_aruco",
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
    recording_path = (
        tmp_path / "ur5e/place_approach/assembly_board-v1__MG__hardware.json"
    )
    steps = json.loads(recording_path.read_text(encoding="utf-8"))["steps"]
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
        {"x": 0.5, "y": 0.6, "z": 0.3}
    )


def test_physical_place_stages_then_rotates_full_relative_pose_with_frozen_aruco(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(robot_task_runtime, "_TAUGHT_FUNCTIONS_ROOT", tmp_path)
    agent = _Agent(execution_mode="physical")
    agent._held_part = "MG"
    agent._current_state = "picked"
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
    assert {
        step["relative_reference"]["generation"]
        for step in recorded_payload["steps"]
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
    assert move_calls == pytest.approx(
        [
            {
                "x": 1.0,
                "y": 2.1,
                "z": 0.5,
                "qx": 0.5,
                "qy": 0.5,
                "qz": 0.5,
                "qw": 0.5,
                "speed": 0.08,
            },
            {
                "x": 1.0,
                "y": 2.1,
                "z": 0.3,
                "qx": 0.5,
                "qy": 0.5,
                "qz": 0.5,
                "qw": 0.5,
            },
        ]
    )
    assert agent._task_ctx["assembly_board_v1_aruco_generation"] == 9
    assert agent._task_ctx["assembly_board_v1_aruco"] == localization


def test_physical_non_board_place_preserves_static_recording_without_aruco(
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
        "move_to_named_pose",
        "compute_place_targets",
        "move_cartesian",
        "move_cartesian",
    ]
    assert agent.primitive_calls[0][1] == {
        "pose_name": "inspection_station",
        "speed": 0.1,
    }
    assert agent.primitive_calls[2][1] == pytest.approx({**above, "speed": 0.1})
    assert agent.primitive_calls[3][1] == pytest.approx(target)
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
        "delay",
        "release_part",
        "delay",
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
    approach_result = asyncio.run(
        execute_robot_task(
            agent,
            "place_approach",
            destination_location="assembly_board-v1",
            part_name="MG",
        )
    )
    assert approach_result["status"] == "completed"
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
    approach_result = asyncio.run(
        execute_robot_task(
            agent,
            "place_approach",
            destination_location="assembly_board-v1",
            part_name="MG",
        )
    )
    assert approach_result["status"] == "completed"
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
    approach_result = asyncio.run(
        execute_robot_task(
            agent,
            "place_approach",
            destination_location="assembly_board-v1",
            part_name="MG",
        )
    )
    assert approach_result["status"] == "completed"
    execute_primitive = agent._execute_primitive

    async def change_board_lock_after_delay(
        primitive: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        result = await execute_primitive(primitive, params)
        if primitive == "delay":
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
    assert [primitive for primitive, _params in agent.primitive_calls] == ["delay"]


def test_xarm6_replays_relative_place_recording_with_link_eef_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(robot_task_runtime, "_TAUGHT_FUNCTIONS_ROOT", tmp_path)
    agent = _Agent(execution_mode="physical", robot="xarm6")
    agent._held_part = "MG"
    agent._current_state = "picked"
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
            {"x": 0.5, "y": 0.6, "z": 0.3, **quaternion},
        ]
    )
