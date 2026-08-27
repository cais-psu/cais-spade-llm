"""Regression coverage for the physical place_insert release-only demo override."""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from cais_spade_llm.resources.robot.gazebo_pick_place_controller import (
    GazeboPickPlaceController,
)
from cais_spade_llm.resources.robot.hardware_pick_place_controller import (
    UR5eHardwareController,
)
from cais_spade_llm.resources.robot.robot_task_runtime import execute_robot_task
from cais_spade_llm.ui.bridge import SystemBridge

_ROOT = Path(__file__).resolve().parents[1]
_IDENTITY_POSE = {
    "x": 0.0,
    "y": 0.0,
    "z": 0.0,
    "qx": 0.0,
    "qy": 0.0,
    "qz": 0.0,
    "qw": 1.0,
}


class _PlaceInsertAgent:
    """Minimal RobotAgent-compatible owner for no-hardware runtime tests."""

    def __init__(
        self,
        robot: str,
        *,
        execution_mode: str = "physical",
        override: Any = True,
        primitive_failures: set[str] | None = None,
    ) -> None:
        self.agent_name = robot
        self.execution_mode = execution_mode
        self.controller_config = {"place_insert_release_only": override}
        self._held_part = "MG"
        self._current_state = "positioned"
        self._position = {"x": 0.4, "y": 0.1, "z": 0.5}
        self._gripper_state = "closed"
        self._recovery_pose_ref = None
        self._task_ctx = self._place_context(robot, execution_mode)
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.progress: list[str] = []
        self.primitive_failures = set(primitive_failures or set())
        self.logger = logging.getLogger(f"test.place_insert_release_only.{robot}")
        self._robot_task_progress_callback = self._record_progress
        self._controller = SimpleNamespace(
            controller_config=self.controller_config,
            _assembly_board_v1_aruco_acceptance=lambda: (
                {"accepted_generation": 7, "calibration_id": "board-calibration"},
                "",
            ),
        )

    @staticmethod
    def _place_context(robot: str, execution_mode: str) -> dict[str, Any]:
        handoff = {
            "part_name": "MG",
            "model_name": "gear_medium",
            "origin_resource_location": "prusa-mk4-2",
            "frame_id": "world",
            "tool_frame": "tool0",
            "part_frame": "held_part_origin",
            "source": "pick_grasp",
            "world_tool0_pose_at_grasp": dict(_IDENTITY_POSE),
            "world_held_part_pose_at_grasp": dict(_IDENTITY_POSE),
            "tool0_to_held_part": dict(_IDENTITY_POSE),
            "origin_pose_provenance": {
                "frame_id": "world",
                "part_name": "MG",
                "model_name": "gear_medium",
            },
        }
        context = {
            "part_name": "MG",
            "model_name": "gear_medium",
            "origin_resource_location": "prusa-mk4-2",
            "destination_location": "assembly_board-v1",
            "surface_role": "assembly_slot",
            "travel_z": 0.58,
            "held_part_handoff": handoff,
            "assembly_board_v1_aruco": {
                "destination_location": "assembly_board-v1",
                "camera_role": robot,
                "generation": 7,
                "calibration_id": "board-calibration",
                "captured_at": time.time(),
                "frame_id": "world",
                "pose": dict(_IDENTITY_POSE),
            },
            "assembly_board_v1_aruco_generation": 7,
        }
        if execution_mode == "simulation":
            context["move_insert_mode"] = "simulation_direct"
        return context

    def _record_progress(self, task_name: str, step_id: str) -> None:
        self.progress.append(f"{task_name}.{step_id}")

    async def _execute_primitive(
        self,
        primitive: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        self.calls.append((primitive, dict(params)))
        if primitive in self.primitive_failures:
            return {"success": False, "message": f"{primitive} failed for test"}
        if primitive == "move_insert":
            return {"success": True, "absolute_position": dict(_IDENTITY_POSE)}
        return {"success": True}

    async def _maybe_inject_failure(self, **_kwargs: Any) -> None:
        return None

    @staticmethod
    def _task_failure(
        message: str,
        *,
        step: str,
        observations: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "status": "failed",
            "content": message,
            "failure_context": {
                "step": step,
                "observations": dict(observations),
            },
        }


class _XArmPlaceApproachAgent(_PlaceInsertAgent):
    """Physical xarm6 place_approach owner with no board-localization primitive."""

    def __init__(self) -> None:
        super().__init__("xarm6")
        self.controller_config["place_approach_skip_board_localization"] = True
        self._held_part = "SG"
        self._current_state = "picked"
        self._gripper_state = "closed"
        self.named_positions: dict[str, Any] = {}
        self._task_ctx["part_name"] = "SG"
        self._task_ctx["model_name"] = "gear_small"
        self._task_ctx["held_part_handoff"]["part_name"] = "SG"
        self._task_ctx["held_part_handoff"]["model_name"] = "gear_small"
        self._task_ctx["held_part_handoff"]["origin_pose_provenance"][
            "part_name"
        ] = "SG"
        self._task_ctx["held_part_handoff"]["origin_pose_provenance"][
            "model_name"
        ] = "gear_small"
        self._task_ctx.pop("assembly_board_v1_aruco")
        self._task_ctx.pop("assembly_board_v1_aruco_generation")

    async def _execute_primitive(
        self,
        primitive: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        if primitive == "localize_assembly_board_v1":
            raise AssertionError("xarm6 demo dispatched board localization")
        self.calls.append((primitive, dict(params)))
        if primitive == "compute_place_targets":
            return {
                "success": True,
                "part_name": "SG",
                "model_name": "gear_small",
                "destination_location": "assembly_board-v1",
                "approach_pose": {
                    **_IDENTITY_POSE,
                    "x": -0.1,
                    "y": 0.08,
                    "z": 1.30,
                },
                "target_pose": {
                    **_IDENTITY_POSE,
                    "x": -0.1,
                    "y": 0.08,
                    "z": 1.25,
                },
                "pre_insert_pose": {
                    **_IDENTITY_POSE,
                    "x": -0.1,
                    "y": 0.08,
                    "z": 1.25,
                },
                "insert_pose": {
                    **_IDENTITY_POSE,
                    "x": -0.1,
                    "y": 0.08,
                    "z": 1.25,
                },
                "move_insert_mode": "",
                "place_approach_skip_board_localization": True,
            }
        return {"success": True}


def _run_place_insert(agent: _PlaceInsertAgent) -> dict[str, Any]:
    return asyncio.run(
        execute_robot_task(
            agent,
            "place_insert",
            None,
            False,
            None,
            destination_location="assembly_board-v1",
            part_name="MG",
        )
    )


def _run_place_approach(agent: _XArmPlaceApproachAgent) -> dict[str, Any]:
    return asyncio.run(
        execute_robot_task(
            agent,
            "place_approach",
            None,
            False,
            None,
            destination_location="assembly_board-v1",
            part_name="SG",
        )
    )


@pytest.mark.parametrize("robot", ["xarm6", "ur5e"])
def test_physical_release_only_skips_move_insert_and_completes(robot: str) -> None:
    agent = _PlaceInsertAgent(robot)

    result = _run_place_insert(agent)

    assert result["status"] == "completed"
    assert result["place_insert_release_only"] is True
    assert result["move_insert_dispatched"] is False
    assert [primitive for primitive, _params in agent.calls] == [
        "release_part",
        "move_relative",
    ]
    assert "place_insert.move_insert" not in agent.progress
    assert agent._held_part is None
    assert agent._gripper_state == "open"
    assert agent._current_state == "placed"
    assert agent._task_ctx == {}
    assert result["placed_location"] == "assembly_board-v1"
    assert "without move_insert" in result["content"]


def test_release_failure_does_not_lift_or_clear_custody() -> None:
    agent = _PlaceInsertAgent("ur5e", primitive_failures={"release_part"})

    result = _run_place_insert(agent)

    assert result["status"] == "failed"
    assert result["place_insert_release_only"] is True
    assert result["move_insert_dispatched"] is False
    assert [primitive for primitive, _params in agent.calls] == ["release_part"]
    assert agent._held_part == "MG"
    assert agent._gripper_state == "closed"
    assert "lift was not commanded" in result["content"]


def test_lift_failure_retains_irreversible_release_accounting() -> None:
    agent = _PlaceInsertAgent("xarm6", primitive_failures={"move_relative"})

    result = _run_place_insert(agent)

    assert result["status"] == "failed"
    assert result["place_insert_release_only"] is True
    assert result["move_insert_dispatched"] is False
    assert [primitive for primitive, _params in agent.calls] == [
        "release_part",
        "move_relative",
    ]
    assert agent._held_part is None
    assert agent._gripper_state == "open"
    assert "release_part already completed" in result["content"]
    assert "no longer clamped" in result["content"]


@pytest.mark.parametrize("robot", ["xarm6", "ur5e"])
@pytest.mark.parametrize("override", [False, None, "true", 1])
def test_non_true_override_restores_strict_physical_blocks(
    robot: str,
    override: Any,
) -> None:
    agent = _PlaceInsertAgent(robot, override=override)

    result = _run_place_insert(agent)

    assert result["status"] == "blocked"
    assert agent.calls == []
    if robot == "xarm6":
        assert "physical xarm6 assembly_board-v1 assembly_slot insertion" in result[
            "content"
        ]
    else:
        assert "cannot release or lift" in result["content"]


@pytest.mark.parametrize("robot", ["xarm6", "ur5e"])
def test_missing_override_restores_strict_physical_blocks(robot: str) -> None:
    agent = _PlaceInsertAgent(robot)
    agent.controller_config.pop("place_insert_release_only")

    result = _run_place_insert(agent)

    assert result["status"] == "blocked"
    assert agent.calls == []


def test_simulation_keeps_simulation_direct_move_insert_path() -> None:
    agent = _PlaceInsertAgent("ur5e", execution_mode="simulation")

    result = _run_place_insert(agent)

    assert result["status"] == "completed"
    assert "place_insert_release_only" not in result
    assert [primitive for primitive, _params in agent.calls] == [
        "move_insert",
        "release_part",
        "snap_part_to_slot",
        "move_relative",
    ]
    assert "place_insert.move_insert" in agent.progress


def test_manifests_enable_exact_physical_boolean() -> None:
    for robot in ("xarm6", "ur5e"):
        path = (
            _ROOT
            / "cais_spade_llm"
            / "initialization"
            / "resources"
            / f"robot_{robot}.json"
        )
        resource = json.loads(path.read_text(encoding="utf-8"))

        assert resource[robot]["real"]["controller"][
            "place_insert_release_only"
        ] is True

    xarm6_path = (
        _ROOT
        / "cais_spade_llm"
        / "initialization"
        / "resources"
        / "robot_xarm6.json"
    )
    xarm6_resource = json.loads(xarm6_path.read_text(encoding="utf-8"))
    assert xarm6_resource["xarm6"]["real"]["controller"][
        "place_approach_skip_board_localization"
    ] is True


@pytest.mark.parametrize("robot", ["xarm6", "ur5e"])
def test_release_only_place_targets_do_not_require_move_insert_geometry(
    robot: str,
) -> None:
    controller = SimpleNamespace(
        wait_for_services=lambda: True,
        execution_mode="physical",
        robot_name=robot,
        controller_config={"place_insert_release_only": True},
        pick_tcp_z_bias_min_m=0.0,
        pick_tcp_z_bias_max_m=0.1,
        place_surface_gap_m=0.0,
        insertion_depth_m=0.0,
        _get_ee_tcp_world_z_offset=lambda: 0.0,
    )
    frozen_board = {
        "destination_location": "assembly_board-v1",
        "camera_role": robot,
        "generation": 7,
        "calibration_id": "board-calibration",
        "captured_at": time.time(),
        "frame_id": "world",
        "pose": dict(_IDENTITY_POSE),
    }

    result = GazeboPickPlaceController.compute_place_targets(
        controller,
        pick_ctx={
            "part_name": "MG",
            "model_name": "gear_medium",
            "tx": 0.0,
            "ty": 0.0,
            "tz": 0.0,
            "pick_tcp_z": 0.01,
            "part_height": 0.02,
        },
        part_name="MG",
        destination_location="assembly_board-v1",
        assembly_board_v1_aruco=frozen_board,
    )

    assert result["success"] is True
    assert result["assembly_board_v1_aruco"] == frozen_board
    assert result["move_insert_mode"] == ""
    assert "move_insert_profile" not in result


def test_xarm6_demo_place_targets_do_not_require_board_localization() -> None:
    controller = SimpleNamespace(
        wait_for_services=lambda: True,
        execution_mode="physical",
        robot_name="xarm6",
        controller_config={
            "place_insert_release_only": True,
            "place_approach_skip_board_localization": True,
        },
        pick_tcp_z_bias_min_m=0.0,
        pick_tcp_z_bias_max_m=0.1,
        place_surface_gap_m=0.0,
        insertion_depth_m=0.0,
        _get_ee_tcp_world_z_offset=lambda: 0.0,
    )

    result = GazeboPickPlaceController.compute_place_targets(
        controller,
        pick_ctx={
            "part_name": "SG",
            "model_name": "gear_small",
            "tx": 0.4,
            "ty": -0.3,
            "tz": 1.04,
            "pick_tcp_z": 1.05,
            "part_height": 0.02,
        },
        part_name="SG",
        destination_location="assembly_board-v1",
        assembly_board_v1_aruco=None,
    )

    assert result["success"] is True
    assert result["part_name"] == "SG"
    assert result["place_approach_skip_board_localization"] is True
    assert "assembly_board_v1_aruco" not in result


def test_xarm6_demo_place_approach_skips_localization_before_progress() -> None:
    agent = _XArmPlaceApproachAgent()

    result = _run_place_approach(agent)

    assert result["status"] == "completed"
    assert "place_approach.localize_assembly_board_v1" not in agent.progress
    assert "localize_assembly_board_v1" not in [
        primitive for primitive, _params in agent.calls
    ]
    assert [primitive for primitive, _params in agent.calls] == [
        "compute_place_targets",
        "move_cartesian",
        "move_cartesian",
    ]
    assert agent._current_state == "positioned"
    assert agent._held_part == "SG"


def test_xarm6_release_only_accepts_context_without_board_generation() -> None:
    agent = _PlaceInsertAgent("xarm6")
    agent.controller_config["place_approach_skip_board_localization"] = True
    agent._task_ctx.pop("assembly_board_v1_aruco")
    agent._task_ctx.pop("assembly_board_v1_aruco_generation")

    result = _run_place_insert(agent)

    assert result["status"] == "completed"
    assert [primitive for primitive, _params in agent.calls] == [
        "release_part",
        "move_relative",
    ]


def _release_only_readiness_bridge() -> SystemBridge:
    bridge = object.__new__(SystemBridge)
    bridge._physical_place_insert_release_only = lambda *_args, **_kwargs: True
    bridge._digital_twin_assembly_lifecycle_error = lambda: ""
    bridge._digital_twin_assembly_request_error = lambda *_args: ""
    bridge._digital_twin_assembly_correction_error = lambda *_args, **_kwargs: ""
    bridge._digital_twin_assembly_state_error = lambda *_args, **_kwargs: ""
    bridge._digital_twin_place_approach_recording_error = lambda *_args: (
        "/tmp/confirmed-place.json",
        "",
    )
    bridge._digital_twin_assembly_board_v1_accepted_status = (
        lambda *_args, **_kwargs: ({"accepted_generation": 7}, "")
    )
    bridge._ur5e_named_position_error = lambda *_args: ""
    bridge._xarm6_named_position_error = lambda *_args: ""
    bridge._configured_named_position_error = lambda *_args: ""
    bridge._ur5e_robot_function_execution_lock = None

    agent = SimpleNamespace(executables={})

    def function() -> None:
        return None

    for function_name in SystemBridge._ASSEMBLY_FUNCTION_ORDER:
        agent.executables[function_name] = function
        setattr(agent, function_name, function)

    async def preflight(*_args: Any, **_kwargs: Any) -> tuple[Any, ...]:
        return agent, {}, {"hardware_ready": True}, ""

    bridge._digital_twin_robot_function_execution_preflight_async = preflight

    def insertion_gate_called(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("release-only readiness called an insertion-only gate")

    bridge.digital_twin_move_insert_settings = insertion_gate_called
    bridge._move_insert_normal_qualification_readiness = insertion_gate_called
    bridge._digital_twin_move_insert_live_readiness = insertion_gate_called
    bridge._digital_twin_assembly_move_insert_geometry_readiness = insertion_gate_called
    return bridge


@pytest.mark.parametrize("robot", ["xarm6", "ur5e"])
def test_assembly_readiness_accepts_both_without_insertion_gates(robot: str) -> None:
    bridge = _release_only_readiness_bridge()

    result = asyncio.run(
        SystemBridge.digital_twin_assembly_readiness(
            bridge,
            "digital_twin",
            robot,
            origin_resource_location="prusa-mk4-2",
            destination_location="assembly_board-v1",
            part_name="MG",
        )
    )

    assert result["success"] is True
    assert result["ready"] is True
    assert result["place_insert_release_only"] is True
    assert result["move_insert_dispatched"] is False
    assert result["assembly_step_count"] == 5
    assert "skip move_insert" in result["message"]


def test_xarm6_assembly_request_remains_strict_without_override() -> None:
    bridge = object.__new__(SystemBridge)
    bridge._physical_place_insert_release_only = lambda *_args, **_kwargs: False

    error = SystemBridge._digital_twin_assembly_request_error(
        bridge,
        "digital_twin",
        "xarm6",
        "prusa-mk4-2",
        "assembly_board-v1",
        "MG",
    )

    assert "requires robot 'ur5e'" in error


def test_xarm6_assembly_request_is_accepted_with_override() -> None:
    bridge = object.__new__(SystemBridge)
    bridge._physical_place_insert_release_only = lambda *_args, **_kwargs: True
    bridge._digital_twin_robot_function_request_error = (
        lambda *_args, **_kwargs: ({}, "")
    )

    error = SystemBridge._digital_twin_assembly_request_error(
        bridge,
        "digital_twin",
        "xarm6",
        "prusa-mk4-2",
        "assembly_board-v1",
        "MG",
    )

    assert error == ""


@pytest.mark.parametrize("robot", ["xarm6", "ur5e"])
def test_run_assembly_keeps_five_functions_under_one_release_only_run(
    robot: str,
) -> None:
    bridge = object.__new__(SystemBridge)
    bridge._physical_place_insert_release_only = lambda *_args, **_kwargs: True
    bridge._digital_twin_assembly_lifecycle_error = lambda: ""
    bridge._digital_twin_assembly_request_error = lambda *_args: ""
    bridge._digital_twin_assembly_correction_error = lambda *_args, **_kwargs: ""
    bridge._digital_twin_assembly_state_error = lambda *_args, **_kwargs: ""
    bridge._digital_twin_assembly_resolved_descend_error = lambda *_args: ""
    bridge._xarm6_cartesian_session_mode = lambda: "off"
    lock = threading.Lock()
    bridge._get_robot_function_execution_lock = lambda: lock
    bridge._ur5e_robot_function_execution_active = None
    bridge._ur5e_robot_function_execution_stage = ""
    bridge._ur5e_robot_function_execution_started_at = 0.0
    bridge._robot_function_execution_robot = ""
    bridge._robot_function_execution_active_step = ""
    bridge._assembly_move_insert_profile_sha256 = ""
    bridge._assembly_move_insert_effective = {}
    resource_agent = SimpleNamespace(_current_state="idle")
    calls: list[str] = []
    prior_completion_markers: list[float] = []
    resulting_states = {
        "pick_approach": "positioned",
        "pick_grasp": "picked",
        "place_approach": "positioned",
        "place_insert": "placed",
        "move_home": "idle",
    }

    async def execute_function(
        execution_context: dict[str, Any],
        _target: str,
        _robot: str,
        function_name: str,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        calls.append(function_name)
        prior_completion_markers.append(
            float(
                execution_context.get(
                    "previous_function_completed_monotonic",
                    0.0,
                )
                or 0.0
            )
        )
        execution_context["resource_agent"] = resource_agent
        resource_agent._current_state = resulting_states[function_name]
        return {
            "success": True,
            "status": "completed",
            "function_name": function_name,
            **(
                {
                    "place_insert_release_only": True,
                    "move_insert_dispatched": False,
                }
                if function_name == "place_insert"
                else {}
            ),
        }

    bridge._digital_twin_execute_robot_function_with_held_lock = execute_function

    def insertion_gate_called(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("release-only Assembly called an insertion-only gate")

    bridge.digital_twin_move_insert_settings = insertion_gate_called
    bridge._move_insert_normal_qualification_readiness = insertion_gate_called
    bridge._digital_twin_move_insert_live_readiness = insertion_gate_called
    bridge._digital_twin_assembly_move_insert_geometry_readiness = insertion_gate_called

    result = asyncio.run(
        SystemBridge.digital_twin_execute_assembly(
            bridge,
            "digital_twin",
            robot,
            origin_resource_location="prusa-mk4-2",
            destination_location="assembly_board-v1",
            part_name="MG",
            confirmed=True,
        )
    )

    assert result["success"] is True
    assert result["status"] == "completed"
    assert calls == list(SystemBridge._ASSEMBLY_FUNCTION_ORDER)
    assert result["completed_functions"] == calls
    assert result["place_insert_release_only"] is True
    assert result["move_insert_dispatched"] is False
    assert "does not claim physical insertion or seating" in result["message"]
    assert prior_completion_markers[0] == 0.0
    assert all(marker > 0.0 for marker in prior_completion_markers[1:])
    assert lock.acquire(blocking=False) is True
    lock.release()


def _assembly_function_bridge() -> SystemBridge:
    bridge = object.__new__(SystemBridge)
    bridge._physical_place_insert_release_only = lambda *_args, **_kwargs: False
    bridge._get_robot_function_execution_lock = threading.Lock
    bridge._ur5e_robot_function_execution_active = None
    bridge._ur5e_robot_function_execution_stage = ""
    bridge._ur5e_robot_function_execution_started_at = 0.0
    bridge._robot_function_execution_robot = ""
    bridge._robot_function_execution_active_step = ""
    bridge._assembly_move_insert_profile_sha256 = ""
    bridge._assembly_move_insert_effective = {}
    return bridge


def test_assembly_continuation_reuses_prepared_agent_preflight() -> None:
    bridge = _assembly_function_bridge()
    resource_agent = object()
    prepared_calls: list[Any] = []

    async def full_preflight(*_args: Any, **_kwargs: Any) -> tuple[Any, ...]:
        raise AssertionError("Assembly continuation repeated full agent preparation")

    async def prepared_preflight(
        *_args: Any,
        assembly_resource_agent: Any | None = None,
        **_kwargs: Any,
    ) -> tuple[Any, ...]:
        prepared_calls.append(assembly_resource_agent)
        return None, {}, {"fresh_readiness_checked": True}, "test stop"

    bridge._digital_twin_robot_function_execution_preflight_async = full_preflight
    bridge._digital_twin_robot_function_execution_preflight_prepared_async = (
        prepared_preflight
    )
    execution_context = {
        "bridge": bridge,
        "resource_agent": resource_agent,
        "previous_function_completed_monotonic": time.monotonic() - 0.01,
    }

    result = asyncio.run(
        SystemBridge._digital_twin_execute_robot_function_with_held_lock(
            bridge,
            execution_context,
            "digital_twin",
            "ur5e",
            "pick_grasp",
            origin_resource_location="prusa-mk4-2",
            part_name="MG",
        )
    )

    assert prepared_calls == [resource_agent]
    assert result["success"] is False
    assert result["assembly_continuation_preflight"] is True
    assert result["fresh_readiness_checked"] is True
    assert result["preflight_duration_sec"] >= 0.0
    assert result["message"] == "test stop"


def test_first_assembly_function_keeps_full_preflight() -> None:
    bridge = _assembly_function_bridge()
    full_calls: list[str] = []

    async def full_preflight(
        _target: str,
        _robot: str,
        function_name: str,
        *_args: Any,
        **_kwargs: Any,
    ) -> tuple[Any, ...]:
        full_calls.append(function_name)
        return None, {}, {"fresh_readiness_checked": True}, "test stop"

    async def prepared_preflight(*_args: Any, **_kwargs: Any) -> tuple[Any, ...]:
        raise AssertionError("First Assembly function skipped full preparation")

    bridge._digital_twin_robot_function_execution_preflight_async = full_preflight
    bridge._digital_twin_robot_function_execution_preflight_prepared_async = (
        prepared_preflight
    )
    execution_context = {"bridge": bridge, "resource_agent": None}

    result = asyncio.run(
        SystemBridge._digital_twin_execute_robot_function_with_held_lock(
            bridge,
            execution_context,
            "digital_twin",
            "ur5e",
            "pick_approach",
            origin_resource_location="prusa-mk4-2",
            part_name="MG",
        )
    )

    assert full_calls == ["pick_approach"]
    assert result["success"] is False
    assert result["assembly_continuation_preflight"] is False
    assert result["fresh_readiness_checked"] is True
    assert result["preflight_duration_sec"] >= 0.0


def test_assembly_continuation_reports_function_boundary_timing() -> None:
    bridge = _assembly_function_bridge()

    async def manual_function(
        function_name: str,
        manual_pre_execute: Any,
        _post_staging_callback: Any,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        assert function_name == "move_home"
        assert manual_pre_execute() == ""
        return {"status": "completed", "content": "move_home completed"}

    resource_agent = SimpleNamespace(
        _current_state="idle",
        _task_ctx={},
        _execute_registered_robot_task_for_manual_function_execution=(
            manual_function
        ),
    )

    async def prepared_preflight(*_args: Any, **_kwargs: Any) -> tuple[Any, ...]:
        return resource_agent, {}, {"fresh_readiness_checked": True}, ""

    async def run_on_agent_runtime(awaitable: Any) -> Any:
        return await awaitable

    bridge._digital_twin_robot_function_execution_preflight_prepared_async = (
        prepared_preflight
    )
    bridge._digital_twin_robot_function_execution_preflight_async = (
        lambda *_args, **_kwargs: pytest.fail(
            "Assembly continuation repeated full preflight"
        )
    )
    bridge._digital_twin_assembly_state_error = lambda *_args, **_kwargs: ""
    bridge._record_ur5e_robot_function_result = lambda *_args, **_kwargs: None
    bridge._record_xarm6_robot_function_result = lambda *_args, **_kwargs: None
    bridge._run_on_agent_runtime = run_on_agent_runtime
    previous_completion = time.monotonic() - 0.05
    execution_context = {
        "bridge": bridge,
        "resource_agent": resource_agent,
        "origin_resource_location": "prusa-mk4-2",
        "destination_location": "assembly_board-v1",
        "part_name": "MG",
        "previous_function_completed_monotonic": previous_completion,
    }

    result = asyncio.run(
        SystemBridge._digital_twin_execute_robot_function_with_held_lock(
            bridge,
            execution_context,
            "digital_twin",
            "ur5e",
            "move_home",
        )
    )

    assert result["success"] is True
    assert result["assembly_continuation_preflight"] is True
    assert result["preflight_duration_sec"] >= 0.0
    assert result["pre_dispatch_duration_sec"] >= result["preflight_duration_sec"]
    assert result["inter_function_delay_sec"] >= 0.04
    assert result["function_duration_sec"] >= result["pre_dispatch_duration_sec"]


class _ReadyActionClient:
    """Action client proving prepared readiness does not enter a blocking wait."""

    def __init__(self) -> None:
        self.wait_calls = 0

    def server_is_ready(self) -> bool:
        return True

    def wait_for_server(self, *, timeout_sec: float) -> bool:
        self.wait_calls += 1
        raise AssertionError(f"unexpected blocking wait_for_server({timeout_sec})")


class _DiscoveringActionClient:
    """Action client proving an unready prepared client keeps bounded discovery."""

    def __init__(self) -> None:
        self.wait_timeouts: list[float] = []

    def server_is_ready(self) -> bool:
        return False

    def wait_for_server(self, *, timeout_sec: float) -> bool:
        self.wait_timeouts.append(timeout_sec)
        return True


def test_prepared_action_client_readiness_uses_immediate_server_state() -> None:
    client = _ReadyActionClient()

    ready, error = SystemBridge._prepared_action_client_ready(
        client,
        action_name="/prepared/action",
    )

    assert ready is True
    assert error == ""
    assert client.wait_calls == 0


def test_ur5e_commands_reuse_prepared_action_client_state() -> None:
    client = _ReadyActionClient()

    ready = UR5eHardwareController._prepared_action_client_ready(
        client,
        timeout_sec=2.0,
    )

    assert ready is True
    assert client.wait_calls == 0


def test_ur5e_commands_keep_bounded_action_discovery_fallback() -> None:
    client = _DiscoveringActionClient()

    ready = UR5eHardwareController._prepared_action_client_ready(
        client,
        timeout_sec=2.0,
    )

    assert ready is True
    assert client.wait_timeouts == [2.0]


def test_ur5e_initial_readiness_uses_prepared_action_clients_without_wait() -> None:
    controller = object.__new__(UR5eHardwareController)
    arm_client = _ReadyActionClient()
    gripper_client = _ReadyActionClient()
    cartesian_client = _ReadyActionClient()
    controller.init = lambda: True
    controller._services_ready = False
    controller._last_failure_message = ""
    controller._ur5e_hardware_trajectory_client = arm_client
    controller._rg2_action_client = gripper_client
    controller._ur5e_hardware_cartesian_client = cartesian_client
    controller._ur5e_hardware_trajectory_action = "/arm"
    controller._rg2_action_name = "/gripper"
    controller._ur5e_hardware_cartesian_action = "/cartesian"
    controller._get_arm_joint_positions = lambda **_kwargs: ([0.0] * 6, [])

    ready = UR5eHardwareController.wait_for_services(controller, timeout_sec=8.0)

    assert ready is True
    assert controller._services_ready is True
    assert arm_client.wait_calls == 0
    assert gripper_client.wait_calls == 0
    assert cartesian_client.wait_calls == 0


def _xarm6_feedback_bridge() -> SystemBridge:
    bridge = object.__new__(SystemBridge)
    bridge._robot_function_hardware_domain_id = lambda *_args: 42
    bridge._prepared_action_client_ready = lambda *_args, **_kwargs: (True, "")
    bridge._robot_function_execution_pose_readiness = lambda *_args: {
        "world_tool0_ready": True,
    }
    return bridge


def test_xarm6_readiness_reuses_fresh_prepared_controller_feedback() -> None:
    bridge = _xarm6_feedback_bridge()
    bridge._snapshot_robot_waypoint = lambda *_args, **_kwargs: pytest.fail(
        "Fresh prepared feedback must not spawn a hardware snapshot"
    )
    controller = SimpleNamespace(
        _xarm6_hardware_trajectory_action="/xarm6_traj_controller/follow_joint_trajectory",
        _xarm6_hardware_trajectory_client=object(),
        _get_arm_joint_positions=lambda **_kwargs: ([0.0] * 6, []),
        _joint_state_received_monotonic=time.monotonic(),
        _services_ready=True,
    )

    readiness, error = SystemBridge._digital_twin_xarm6_motion_readiness(
        bridge,
        "digital_twin",
        {},
        SimpleNamespace(_controller=controller),
    )

    assert error == ""
    assert readiness["joint_states_fresh"] is True
    assert readiness["joint_state_source"] == "prepared_controller"
    assert readiness["actual_positions_rad"] == [0.0] * 6
    assert 0.0 <= readiness["joint_state_age_sec"] <= 2.0


def test_xarm6_readiness_fails_closed_on_stale_prepared_feedback() -> None:
    bridge = _xarm6_feedback_bridge()
    bridge._snapshot_robot_waypoint = lambda *_args, **_kwargs: pytest.fail(
        "Stale prepared feedback must fail closed without a second ROS process"
    )
    controller = SimpleNamespace(
        _xarm6_hardware_trajectory_action="/xarm6_traj_controller/follow_joint_trajectory",
        _xarm6_hardware_trajectory_client=object(),
        _get_arm_joint_positions=lambda **_kwargs: ([0.0] * 6, []),
        _joint_state_received_monotonic=time.monotonic() - 3.0,
        _services_ready=True,
    )

    readiness, error = SystemBridge._digital_twin_xarm6_motion_readiness(
        bridge,
        "digital_twin",
        {},
        SimpleNamespace(_controller=controller),
    )

    assert readiness["joint_states_fresh"] is False
    assert readiness["joint_state_source"] == "prepared_controller"
    assert readiness["joint_state_age_sec"] > 2.0
    assert error == "xArm6 joint feedback is not fresh."


def test_xarm6_uncertainty_flag_does_not_replace_live_preflight_gates() -> None:
    bridge = object.__new__(SystemBridge)
    agent = SimpleNamespace(_controller=None)
    bridge._xarm6_robot_function_state_uncertain = True
    bridge._digital_twin_target = lambda _target: {}
    bridge._digital_twin_robot_function_target_error = lambda *_args: ""
    bridge._physical_robot_function_cartesian_error = lambda _robot: ""
    bridge._physical_robot_agent = lambda _robot: agent
    bridge._running_physical_xarm6_robot_agent = lambda: agent

    _agent, _call_kwargs, _readiness, error = (
        SystemBridge._digital_twin_robot_function_execution_preflight(
            bridge,
            "dual robots",
            "xarm6",
            "pick_approach",
            "prusa-mk4-1",
            "",
            "SG",
            assembly_resource_agent=agent,
        )
    )

    assert error == "The physical xarm6 controller is unavailable."
    assert "complete move_home" not in error


def test_dual_assembly_readiness_uses_fixed_demo_pair() -> None:
    bridge = object.__new__(SystemBridge)
    bridge._ur5e_robot_function_execution_lock = threading.Lock()
    bridge._digital_twin_assembly_lifecycle_error = lambda: ""
    calls: list[tuple[str, str, dict[str, Any]]] = []

    async def readiness(
        target: str,
        robot: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        calls.append((target, robot, dict(kwargs)))
        await asyncio.sleep(0)
        return {"success": True, "ready": True, "robot": robot}

    bridge.digital_twin_assembly_readiness = readiness

    result = asyncio.run(
        SystemBridge.digital_twin_dual_assembly_readiness(
            bridge,
            "dual robots",
        )
    )

    assert result["success"] is True
    assert calls == [
        (
            "dual robots",
            "xarm6",
            {
                "origin_resource_location": "prusa-mk4-1",
                "destination_location": "assembly_board-v1",
                "part_name": "SG",
            },
        ),
        (
            "dual robots",
            "ur5e",
            {
                "origin_resource_location": "prusa-mk4-2",
                "destination_location": "assembly_board-v1",
                "part_name": "MG",
            },
        ),
    ]


def test_dual_assembly_executes_both_robot_runs_concurrently() -> None:
    bridge = object.__new__(SystemBridge)
    global_lock = threading.Lock()
    bridge._ur5e_robot_function_execution_lock = global_lock
    bridge._ur5e_robot_function_execution_active = None
    bridge._ur5e_robot_function_execution_stage = ""
    bridge._ur5e_robot_function_execution_started_at = 0.0
    bridge._robot_function_execution_robot = ""
    bridge._robot_function_execution_active_step = ""
    bridge._assembly_move_insert_profile_sha256 = ""
    bridge._assembly_move_insert_effective = {}

    async def dual_readiness(_target: str) -> dict[str, Any]:
        return {"success": True, "ready": True}

    started = 0
    both_started = asyncio.Event()
    calls: list[tuple[str, str, dict[str, Any], threading.Lock]] = []

    async def execute_assembly(
        target: str,
        robot: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        nonlocal started
        calls.append(
            (
                target,
                robot,
                dict(kwargs),
                SystemBridge._get_robot_function_execution_lock(bridge),
            )
        )
        started += 1
        if started == 2:
            both_started.set()
        await asyncio.wait_for(both_started.wait(), timeout=0.5)
        return {"success": True, "status": "completed", "robot": robot}

    bridge.digital_twin_dual_assembly_readiness = dual_readiness
    bridge.digital_twin_execute_assembly = execute_assembly

    result = asyncio.run(
        asyncio.wait_for(
            SystemBridge.digital_twin_execute_dual_assembly(
                bridge,
                "dual robots",
                confirmed=True,
            ),
            timeout=1.0,
        )
    )

    assert result["success"] is True
    assert {robot for _target, robot, _kwargs, _lock in calls} == {
        "xarm6",
        "ur5e",
    }
    call_by_robot = {robot: kwargs for _target, robot, kwargs, _lock in calls}
    assert call_by_robot["xarm6"]["origin_resource_location"] == "prusa-mk4-1"
    assert call_by_robot["xarm6"]["part_name"] == "SG"
    assert call_by_robot["ur5e"]["origin_resource_location"] == "prusa-mk4-2"
    assert call_by_robot["ur5e"]["part_name"] == "MG"
    child_locks = [lock for _target, _robot, _kwargs, lock in calls]
    assert child_locks[0] is not child_locks[1]
    assert all(lock is not global_lock for lock in child_locks)
    assert global_lock.acquire(blocking=False) is True
    global_lock.release()


def test_control_page_exposes_fixed_dual_assembly_demo_button() -> None:
    source = (
        _ROOT / "cais_spade_llm" / "ui" / "pages" / "control.py"
    ).read_text(encoding="utf-8")

    assert '"Run Dual Assembly"' in source
    assert "digital_twin_dual_assembly_readiness" in source
    assert "digital_twin_execute_dual_assembly" in source
    assert 'preferred_origin = (\n                "prusa-mk4-1"' in source
    assert 'preferred_part = "SG" if robot == "xarm6" else "MG"' in source
