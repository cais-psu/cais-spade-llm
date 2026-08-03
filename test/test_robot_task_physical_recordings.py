"""Focused tests for registry-backed physical Cartesian position recordings."""

from __future__ import annotations

import asyncio
import json
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
        self.primitive_calls: list[tuple[str, dict[str, Any]]] = []
        self.failures: list[dict[str, Any]] = []

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
                "gripper_close_position": 0.37,
                "start_x": 0.0,
                "start_y": 0.0,
                "start_z": 0.0,
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
        return {"success": True, "message": primitive}

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


def _recorded_step(step_name: str, pose: dict[str, float]) -> dict[str, Any]:
    return {
        "step_name": step_name,
        "primitive": "move_cartesian",
        "params": deepcopy(pose),
        "capture_source": "hardware",
        "waypoint": {
            "pose": {
                "frame_id": "world",
                "child_frame_id": "tool0",
                **deepcopy(pose),
            },
            "joint_names": ["joint_1"],
            "joint_positions": [0.0],
            "source": "hardware",
        },
    }


def _write_recording(
    root: Path,
    *,
    function_name: str,
    location: str,
    part_name: str,
    steps: list[dict[str, Any]],
) -> Path:
    path = root / "ur5e" / function_name / f"{location}__{part_name}__hardware.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "robot": "ur5e",
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
            _recorded_step("move_above_destination", above),
            _recorded_step("descend", target),
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
    assert compute_params["detected_parts"] == [
        {
            "part_name": "MG",
            "model_name": "gear_medium",
            "x": 0.1,
            "y": 0.2,
            "z": 0.3,
        }
    ]
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
    assert [primitive for primitive, _params in agent.primitive_calls] == [
        "move_to_named_pose"
    ]
    assert agent._current_state == "idle"


def test_physical_pick_detection_failure_stops_after_staging_before_gripper() -> None:
    agent = _Agent(execution_mode="physical")
    execute_primitive = agent._execute_primitive

    async def reject_detection(
        primitive: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        if primitive == "detect_parts":
            agent.primitive_calls.append((primitive, deepcopy(params)))
            return {"success": False, "message": "MG was not detected"}
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
    assert "MG was not detected" in result["content"]
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
    ],
)
def test_invalid_physical_recording_fails_closed_before_motion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: Any,
    expected_message: str,
) -> None:
    monkeypatch.setattr(robot_task_runtime, "_TAUGHT_FUNCTIONS_ROOT", tmp_path)
    above, target = _place_recording(tmp_path)
    steps = [
        _recorded_step("move_above_destination", above),
        _recorded_step("descend", target),
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


def test_xarm6_recordable_physical_task_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(robot_task_runtime, "_TAUGHT_FUNCTIONS_ROOT", tmp_path)
    agent = _Agent(execution_mode="physical", robot="xarm6")
    agent._held_part = "MG"

    result = asyncio.run(
        execute_robot_task(
            agent,
            "place_approach",
            destination_location="assembly_board-v1",
            part_name="MG",
        )
    )

    assert result["status"] == "failed"
    assert "currently supported only for ur5e" in result["content"]
    assert agent.primitive_calls == []
