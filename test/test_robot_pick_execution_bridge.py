"""Focused tests for guarded physical UR5e robot-function execution."""

from __future__ import annotations

import asyncio
import json
import math
import threading
import time
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from cais_spade_llm.agents.resource_agent import robot_agent as robot_agent_module
from cais_spade_llm.agents.resource_agent.robot_agent import RobotAgent
from cais_spade_llm.resources.robot import robot_task_runtime
from cais_spade_llm.ui.bridge import SystemBridge

ROOT = Path(__file__).resolve().parents[1]
ACTUAL_MG_STL = str(
    (ROOT / "ros2/cais_lab_robotics/cad_models/Gear_Medium.STL").resolve()
)
ACTUAL_MG_STL_SHA256 = "73d2c5d06497db2042ec6624a45558398ee47c9e55ccd87d49184c179a501507"
MANUAL_DESCEND_POSE = {
    "x": 0.3,
    "y": 0.4,
    "z": 1.1,
    "qx": 0.0,
    "qy": 1.0,
    "qz": 0.0,
    "qw": 0.0,
}


def _mg_product_geometry(part_name: str = "MG") -> dict[str, Any]:
    return {
        "part_name": part_name,
        "model_name": "gear_medium",
        "part_height_m": 0.02,
        "source_stl": ACTUAL_MG_STL,
        "source_stl_sha256": ACTUAL_MG_STL_SHA256,
        "hub_up": True,
        "hub_diameter_m": 0.03,
        "hub_height_m": 0.01,
        "tooth_diameter_m": 0.042,
        "tooth_height_m": 0.01,
        "grasp_width_m": 0.028,
        "tooth_clearance_m": 0.002,
        "minimum_hub_overlap_m": 0.006,
    }


def _mg_task_context() -> dict[str, Any]:
    return {
        **_mg_product_geometry(),
        "origin_resource_location": "prusa-mk4-2",
        "pick_tcp_z": 1.0323,
        "finger_tooth_clearance_m": 0.003,
        "finger_hub_overlap_m": 0.007,
        "pick_z_adjustment_m": 0.001,
        "pick_tool0_z_adjustment_m": 0.005,
        "open_gripper_position": 0.11,
        "mg_gripper_close_position": 0.047,
        "open_inner_pad_lower_z_from_tcp_m": 0.01751,
        "open_inner_pad_upper_z_from_tcp_m": 0.04726,
        "closed_inner_pad_lower_z_from_tcp_m": -0.00865,
        "closed_inner_pad_upper_z_from_tcp_m": 0.0211,
        "predicted_closing_z_displacement_m": -0.02616,
        "gripper_close_position": 0.047,
        "travel_z": 1.2,
        "resolved_cartesian_positions": {
            "descend": deepcopy(MANUAL_DESCEND_POSE),
        },
    }


class _PhysicalController:
    gripper_open = 0.11
    gripper_close = 0.02

    def __init__(self) -> None:
        self.gripper_calls: list[tuple[str, float | None]] = []
        self.close_success = True
        self.reopen_success = True
        self._last_failure_message = ""

    def close_gripper(self, position: float | None = None) -> bool:
        self.gripper_calls.append(("close_gripper", position))
        if not self.close_success:
            self._last_failure_message = "close failed"
        return self.close_success

    def open_gripper(self) -> bool:
        self.gripper_calls.append(("open_gripper", None))
        if not self.reopen_success:
            self._last_failure_message = "reopen failed"
        return self.reopen_success

    @staticmethod
    def _physical_stl_pick_readiness(geometry: dict[str, Any]) -> dict[str, Any]:
        if not geometry.get("source_stl"):
            return {"success": False, "message": "actual source_stl is missing"}
        return {
            "success": True,
            **geometry,
            "gripper_close_position": 0.047,
            "finger_tooth_clearance_m": 0.003,
            "finger_hub_overlap_m": 0.007,
            "pick_z_adjustment_m": 0.001,
            "open_gripper_position": 0.11,
            "mg_gripper_close_position": 0.047,
            "predicted_closing_z_displacement_m": -0.02616,
        }


class _PhysicalUR5eAgent:
    def __init__(
        self,
        *,
        state: str = "idle",
        held_part: str | None = None,
        gripper_state: str = "open",
    ) -> None:
        self.agent_name = "ur5e"
        self.jid = "ur5e@localhost"
        self.execution_mode = "physical"
        self._controller = _PhysicalController()
        self.named_positions = {
            "home": [0.0, -1.0, -2.0, -1.5, 1.5, 0.0],
            "prusa-mk4-2": [0.1, -0.8, -2.1, -1.6, 1.5, -3.1],
            "assembly_board-v1": [0.2, -0.9, -2.0, -1.4, 1.5, -3.0],
        }
        self._current_state = state
        self._held_part = held_part
        self._gripper_state = gripper_state
        self._position: dict[str, float] = {"x": 0.0, "y": 0.0, "z": 1.0}
        self._task_ctx: dict[str, Any] = {}
        self._robot_motion_lock = threading.Lock()
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.executables = {
            name: getattr(self, name)
            for name in (
                "pick_approach",
                "pick_grasp",
                "place_approach",
                "place_insert",
                "move_home",
            )
        }

    @staticmethod
    def is_alive() -> bool:
        return True

    async def pick_approach(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("pick_approach", kwargs))
        self._task_ctx = {
            **_mg_task_context(),
            "part_name": kwargs["part_name"],
            "origin_resource_location": kwargs["origin_resource_location"],
        }
        self._current_state = "at_pick"
        return {"status": "completed", "content": "Arrived at the live pick target."}

    async def pick_grasp(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("pick_grasp", kwargs))
        self._held_part = kwargs["part_name"]
        self._gripper_state = "closed"
        self._current_state = "picked"
        return {"status": "completed", "content": "Picked MG."}

    async def place_approach(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("place_approach", kwargs))
        self._task_ctx["destination_location"] = kwargs["destination_location"]
        self._current_state = "positioned"
        return {"status": "completed", "content": "Reached assembly_board-v1."}

    async def place_insert(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("place_insert", kwargs))
        self._held_part = None
        self._gripper_state = "open"
        self._current_state = "placed"
        return {"status": "completed", "content": "Assembled MG."}

    async def move_home(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("move_home", kwargs))
        self._current_state = "idle"
        return {"status": "completed", "content": "At home position."}

    async def _execute_registered_robot_task_for_manual_function_execution(
        self,
        function_name: str,
        pre_execute: Any = None,
        post_staging_acceptance: Any = None,
        /,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if callable(pre_execute):
            pre_execute_error = str(pre_execute() or "").strip()
            if pre_execute_error:
                return {
                    "status": "blocked",
                    "content": pre_execute_error,
                    "manual_pre_execute_blocked": True,
                }
        callback_name = "_assembly_board_v1_post_staging_accept_callback"
        install_callback = bool(
            callable(post_staging_acceptance)
            and function_name == "place_approach"
            and kwargs.get("destination_location") == "assembly_board-v1"
        )
        if install_callback:
            setattr(self._controller, callback_name, post_staging_acceptance)
        try:
            return await getattr(self, function_name)(**kwargs)
        finally:
            if install_callback:
                delattr(self._controller, callback_name)


def _healthy_status(target: str) -> dict[str, Any]:
    return {
        "target": target,
        "repair_needed": False,
        "repair_reason": "",
        "gazebo": {"status": "running"},
        "hardware": {"overall": "running"},
        "sync/status": {"process_status": "running", "state": "mirroring"},
    }


def _ready_bridge(agent: _PhysicalUR5eAgent) -> SystemBridge:
    bridge = object.__new__(SystemBridge)
    bridge.system_running = True
    bridge.execution_mode = "physical"
    bridge.selected_product = "product.json"
    bridge.resource_agents = [agent]
    bridge._ur5e_robot_function_execution_lock = threading.Lock()
    bridge._ur5e_robot_function_preflight_lock = threading.Lock()
    bridge._ur5e_robot_function_execution_active = None
    bridge._ur5e_robot_function_execution_stage = ""
    bridge._ur5e_robot_function_execution_started_at = 0.0
    bridge._ur5e_robot_function_state_uncertain = False
    bridge._MG_CLOSE_TEST_HOLD_S = 0.0
    bridge._digital_twin_sim_mode = lambda _target: "monitor"
    bridge._robot_function_capture_source = lambda _target, _cfg: "hardware"
    bridge._digital_twin_direction = lambda _target: "hardware -> gazebo"
    bridge.digital_twin_statuses = lambda: {
        target: _healthy_status(target) for target in ("ur5e only", "dual robots")
    }
    bridge._digital_twin_ur5e_motion_readiness = lambda _target, _cfg, _agent=None: (
        {
            "hardware_domain_id": 42,
            "trajectory_action_ready": True,
            "rtde_receive_connected": True,
            "joint_states_fresh": True,
            "rtde_control_connected": True,
        },
        "",
    )
    bridge._digital_twin_ur5e_gripper_readiness = lambda _target, _domain, _agent=None: (
        {"gripper_action_ready": True},
        "",
    )
    bridge._robot_function_capture_snapshot = lambda _target, _robot: {
        "success": True,
        "world_tool0_ready": True,
        "blocked_reason": "",
        "waypoint": {
            "source": "hardware",
            "pose": {
                **deepcopy(MANUAL_DESCEND_POSE),
                "frame_id": "world",
                "child_frame_id": "link_eef" if _robot == "xarm6" else "tool0",
            },
        },
    }
    bridge.physical_perception_ready = lambda: (True, "")
    bridge._robot_function_product_geometry_for_part = _mg_product_geometry
    bridge._digital_twin_place_approach_recording_error = lambda _agent, _destination, _part: (
        "/tmp/assembly_board-v1__MG__hardware.json",
        "",
    )
    bridge.perception_manager = SimpleNamespace(
        assembly_board_v1_aruco_status=lambda role: {
            "camera_role": role,
            "accepted": True,
            "accepted_baseline_ready": True,
            "accepted_baseline_error": "",
            "accepted_generation": 1,
            "accepted_calibration_id": f"{role}-calibration",
            "active_calibration_id": f"{role}-calibration",
            "calibration_changed": False,
            "movement_evidence_valid": False,
            "movement_blocked": False,
        }
    )

    async def _run_on_agent_runtime(coroutine: Any) -> Any:
        return await coroutine

    bridge._run_on_agent_runtime = _run_on_agent_runtime
    return bridge


def _agent_for(function_name: str) -> _PhysicalUR5eAgent:
    if function_name == "pick_approach":
        return _PhysicalUR5eAgent()
    if function_name == "pick_grasp":
        agent = _PhysicalUR5eAgent(state="at_pick")
        agent._task_ctx = _mg_task_context()
        return agent
    if function_name == "place_approach":
        return _PhysicalUR5eAgent(state="picked", held_part="MG", gripper_state="closed")
    if function_name == "place_insert":
        agent = _PhysicalUR5eAgent(state="positioned", held_part="MG", gripper_state="closed")
        agent._task_ctx = {
            "destination_location": "assembly_board-v1",
            "travel_z": 1.2,
            "assembly_board_v1_aruco_generation": 1,
            "assembly_board_v1_aruco": {
                "destination_location": "assembly_board-v1",
                "camera_role": "ur5e",
                "generation": 1,
                "calibration_id": "ur5e-calibration",
            },
            "resolved_cartesian_positions": {
                "descend": deepcopy(MANUAL_DESCEND_POSE),
            },
        }
        return agent
    return _PhysicalUR5eAgent(state="placed")


def _run_to_thread_inline(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _inline(function: Any, *args: Any, **kwargs: Any) -> Any:
        return function(*args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", _inline)


@pytest.mark.parametrize(
    ("function_name", "arguments", "expected_kwargs"),
    [
        (
            "pick_approach",
            {"origin_resource_location": "prusa-mk4-2", "part_name": "MG"},
            {
                "origin_resource_location": "prusa-mk4-2",
                "part_name": "MG",
                "product_geometry": _mg_product_geometry(),
            },
        ),
        (
            "pick_grasp",
            {"origin_resource_location": "prusa-mk4-2", "part_name": "MG"},
            {"origin_resource_location": "prusa-mk4-2", "part_name": "MG"},
        ),
        (
            "place_approach",
            {"destination_location": "assembly_board-v1", "part_name": "MG"},
            {
                "destination_location": "assembly_board-v1",
                "part_name": "MG",
                "product_geometry": _mg_product_geometry(),
            },
        ),
        (
            "place_insert",
            {"destination_location": "assembly_board-v1", "part_name": "MG"},
            {"destination_location": "assembly_board-v1", "part_name": "MG"},
        ),
        ("move_home", {}, {}),
    ],
)
def test_generic_execution_dispatches_each_exact_generated_method(
    function_name: str,
    arguments: dict[str, str],
    expected_kwargs: dict[str, Any],
) -> None:
    agent = _agent_for(function_name)
    bridge = _ready_bridge(agent)
    result = asyncio.run(
        bridge.digital_twin_execute_robot_function(
            "dual robots",
            "ur5e",
            function_name,
            **arguments,
            confirmed=True,
        )
    )

    assert result["success"] is True
    assert agent.calls == [(function_name, expected_kwargs)]


def test_robot_agent_manual_function_execution_uses_private_identity_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = object.__new__(RobotAgent)
    agent.agent_name = "ur5e"
    agent._robot_motion_lock = threading.Lock()
    observed: dict[str, Any] = {}

    async def _execute(
        actual_agent: Any,
        task_name: str,
        authority: object | None = None,
        /,
        **kwargs: Any,
    ) -> dict[str, Any]:
        observed.update(
            {
                "agent": actual_agent,
                "task_name": task_name,
                "authority": authority,
                "kwargs": kwargs,
            }
        )
        return {"status": "completed"}

    monkeypatch.setattr(robot_agent_module, "execute_robot_task", _execute)
    pre_execute_lock_states: list[bool] = []

    def _pre_execute() -> str:
        pre_execute_lock_states.append(agent._robot_motion_lock.locked())
        return ""

    result = asyncio.run(
        agent._execute_registered_robot_task_for_manual_function_execution(
            "pick_approach",
            _pre_execute,
            origin_resource_location="prusa-mk4-2",
            part_name="MG",
        )
    )

    assert result == {"status": "completed"}
    assert observed == {
        "agent": agent,
        "task_name": "pick_approach",
        "authority": robot_task_runtime._MANUAL_FUNCTION_EXECUTION_AUTHORITY,
        "kwargs": {
            "origin_resource_location": "prusa-mk4-2",
            "part_name": "MG",
        },
    }
    assert pre_execute_lock_states == [True]
    assert agent._robot_motion_lock.locked() is False

    observed.clear()
    blocked = asyncio.run(
        agent._execute_registered_robot_task_for_manual_function_execution(
            "pick_grasp",
            lambda: "current TCP moved",
            origin_resource_location="prusa-mk4-2",
            part_name="MG",
        )
    )
    assert blocked == {
        "status": "blocked",
        "content": "current TCP moved",
        "manual_pre_execute_blocked": True,
    }
    assert observed == {}


def test_robot_agent_scopes_post_staging_acceptance_to_its_motion_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = object.__new__(RobotAgent)
    agent.agent_name = "ur5e"
    agent._robot_motion_lock = threading.Lock()
    agent._controller = SimpleNamespace()

    def callback(_requested_at: float) -> None:
        pass

    async def _execute(
        actual_agent: Any,
        task_name: str,
        _authority: object | None = None,
        /,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        assert task_name == "place_approach"
        assert actual_agent._robot_motion_lock.locked() is True
        assert (
            actual_agent._controller._assembly_board_v1_post_staging_accept_callback
            is callback
        )
        return {"status": "completed"}

    monkeypatch.setattr(robot_agent_module, "execute_robot_task", _execute)

    result = asyncio.run(
        agent._execute_registered_robot_task_for_manual_function_execution(
            "place_approach",
            None,
            callback,
            destination_location="assembly_board-v1",
            part_name="MG",
        )
    )

    assert result == {"status": "completed"}
    assert not hasattr(
        agent._controller,
        "_assembly_board_v1_post_staging_accept_callback",
    )
    assert agent._robot_motion_lock.locked() is False
    assert agent._robot_motion_lock.locked() is False


@pytest.mark.parametrize("target", ["ur5e only", "dual robots"])
def test_readiness_accepts_each_hardware_led_ur5e_monitor_target(target: str) -> None:
    agent = _agent_for("pick_grasp")
    bridge = _ready_bridge(agent)

    result = asyncio.run(
        bridge.digital_twin_robot_function_execution_readiness(
            target,
            "ur5e",
            "pick_grasp",
            origin_resource_location="prusa-mk4-2",
            part_name="MG",
        )
    )

    assert result["ready"] is True
    assert result["target"] == target


def test_pick_approach_readiness_blocks_without_actual_mg_stl_geometry() -> None:
    agent = _agent_for("pick_approach")
    bridge = _ready_bridge(agent)
    bridge._robot_function_product_geometry_for_part = lambda _part: {
        "part_name": "MG",
        "model_name": "gear_medium",
        "part_height_m": 0.02,
    }

    result = asyncio.run(
        bridge.digital_twin_robot_function_execution_readiness(
            "dual robots",
            "ur5e",
            "pick_approach",
            origin_resource_location="prusa-mk4-2",
            part_name="MG",
        )
    )

    assert result["ready"] is False
    assert "actual source_stl" in result["message"]
    assert agent.calls == []


def test_pick_approach_readiness_ignores_manual_sequence_state_without_mutation() -> None:
    agent = _PhysicalUR5eAgent(state="picked")
    agent._task_ctx = _mg_task_context()
    retained_context = deepcopy(agent._task_ctx)
    bridge = _ready_bridge(agent)

    result = asyncio.run(
        bridge.digital_twin_robot_function_execution_readiness(
            "dual robots",
            "ur5e",
            "pick_approach",
            origin_resource_location="prusa-mk4-2",
            part_name="MG",
        )
    )

    assert result["ready"] is True
    assert "pick_approach_reset_to_idle" not in result
    assert "state 'idle'" not in result["message"]
    assert agent._current_state == "picked"
    assert agent._task_ctx == retained_context
    assert agent.calls == []


def test_confirmed_pick_approach_dispatches_manual_mode_without_state_rewrite() -> None:
    agent = _PhysicalUR5eAgent(state="picked")
    agent._task_ctx = _mg_task_context()
    retained_context = deepcopy(agent._task_ctx)
    agent._recovery_pose_ref = "stale-pick"
    observed: dict[str, Any] = {}

    async def _pick_approach_manual(**kwargs: Any) -> dict[str, Any]:
        observed["state"] = agent._current_state
        observed["task_ctx"] = deepcopy(agent._task_ctx)
        observed["recovery_pose_ref"] = agent._recovery_pose_ref
        agent.calls.append(("pick_approach", kwargs))
        agent._current_state = "at_pick"
        return {"status": "completed", "content": "Arrived at the live pick target."}

    agent.pick_approach = _pick_approach_manual  # type: ignore[method-assign]
    bridge = _ready_bridge(agent)

    result = asyncio.run(
        bridge.digital_twin_execute_robot_function(
            "dual robots",
            "ur5e",
            "pick_approach",
            origin_resource_location="prusa-mk4-2",
            part_name="MG",
            confirmed=True,
        )
    )

    assert result["success"] is True
    assert observed == {
        "state": "picked",
        "task_ctx": retained_context,
        "recovery_pose_ref": "stale-pick",
    }
    assert agent._current_state == "at_pick"


def test_pick_approach_manual_run_still_requires_an_empty_gripper() -> None:
    agent = _PhysicalUR5eAgent(
        state="picked",
        held_part="MG",
        gripper_state="closed",
    )
    agent._task_ctx = _mg_task_context()
    bridge = _ready_bridge(agent)

    result = asyncio.run(
        bridge.digital_twin_robot_function_execution_readiness(
            "dual robots",
            "ur5e",
            "pick_approach",
            origin_resource_location="prusa-mk4-2",
            part_name="MG",
        )
    )

    assert result["ready"] is False
    assert "empty ur5e gripper" in result["message"]
    assert agent._current_state == "picked"
    assert agent._task_ctx
    assert agent.calls == []


def test_pick_approach_manual_readiness_does_not_rewrite_uncertain_sequence_state() -> None:
    agent = _PhysicalUR5eAgent(state="picked")
    agent._task_ctx = _mg_task_context()
    bridge = _ready_bridge(agent)
    bridge._ur5e_robot_function_state_uncertain = True

    result = asyncio.run(
        bridge.digital_twin_robot_function_execution_readiness(
            "dual robots",
            "ur5e",
            "pick_approach",
            origin_resource_location="prusa-mk4-2",
            part_name="MG",
        )
    )

    assert result["ready"] is True
    assert "state is uncertain" not in result["message"]
    assert "pick_approach_reset_to_idle" not in result
    assert agent._current_state == "picked"
    assert agent._task_ctx
    assert agent.calls == []


@pytest.mark.parametrize(
    ("function_name", "manual_state", "arguments"),
    [
        (
            "pick_approach",
            "positioned",
            {"origin_resource_location": "prusa-mk4-2", "part_name": "MG"},
        ),
        (
            "pick_grasp",
            "idle",
            {"origin_resource_location": "prusa-mk4-2", "part_name": "MG"},
        ),
        (
            "place_approach",
            "idle",
            {"destination_location": "assembly_board-v1", "part_name": "MG"},
        ),
        (
            "place_insert",
            "picked",
            {"destination_location": "assembly_board-v1", "part_name": "MG"},
        ),
    ],
)
def test_manual_function_readiness_ignores_only_sequence_state(
    function_name: str,
    manual_state: str,
    arguments: dict[str, str],
) -> None:
    agent = _agent_for(function_name)
    agent._current_state = manual_state
    bridge = _ready_bridge(agent)

    result = asyncio.run(
        bridge.digital_twin_robot_function_execution_readiness(
            "dual robots",
            "ur5e",
            function_name,
            **arguments,
        )
    )

    assert result["ready"] is True, result
    assert agent._current_state == manual_state
    assert agent.calls == []


def test_manual_pick_grasp_requires_pick_approach_descend_pose() -> None:
    agent = _agent_for("pick_grasp")
    agent._current_state = "idle"
    agent._task_ctx.pop("resolved_cartesian_positions")
    bridge = _ready_bridge(agent)

    result = asyncio.run(
        bridge.digital_twin_robot_function_execution_readiness(
            "dual robots",
            "ur5e",
            "pick_grasp",
            origin_resource_location="prusa-mk4-2",
            part_name="MG",
        )
    )

    assert result["ready"] is False
    assert "pick_approach.descend" in result["message"]
    assert "Run pick_approach before pick_grasp" in result["message"]
    assert agent.calls == []


def test_manual_place_insert_rejects_tcp_moved_from_place_approach_descend() -> None:
    agent = _agent_for("place_insert")
    agent._current_state = "picked"
    bridge = _ready_bridge(agent)
    moved_pose = {**deepcopy(MANUAL_DESCEND_POSE), "x": MANUAL_DESCEND_POSE["x"] + 0.003}
    bridge._robot_function_capture_snapshot = lambda _target, _robot: {
        "success": True,
        "world_tool0_ready": True,
        "blocked_reason": "",
        "waypoint": {
            "source": "hardware",
            "pose": {
                **moved_pose,
                "frame_id": "world",
                "child_frame_id": "tool0",
            },
        },
    }

    result = asyncio.run(
        bridge.digital_twin_robot_function_execution_readiness(
            "dual robots",
            "ur5e",
            "place_insert",
            destination_location="assembly_board-v1",
            part_name="MG",
        )
    )

    assert result["ready"] is False
    assert "place_approach.descend" in result["message"]
    assert "3.00 mm" in result["message"]
    assert "limit 2.00 mm" in result["message"]
    assert agent.calls == []


def test_manual_pick_grasp_rechecks_tcp_after_acquiring_agent_motion_lock() -> None:
    agent = _agent_for("pick_grasp")
    agent._current_state = "idle"
    bridge = _ready_bridge(agent)
    poses = iter(
        (
            deepcopy(MANUAL_DESCEND_POSE),
            {
                **deepcopy(MANUAL_DESCEND_POSE),
                "x": MANUAL_DESCEND_POSE["x"] + 0.003,
            },
        )
    )

    def _snapshot(_target: str, _robot: str) -> dict[str, Any]:
        return {
            "success": True,
            "world_tool0_ready": True,
            "blocked_reason": "",
            "waypoint": {
                "source": "hardware",
                "pose": {
                    **next(poses),
                    "frame_id": "world",
                    "child_frame_id": "tool0",
                },
            },
        }

    bridge._robot_function_capture_snapshot = _snapshot

    result = asyncio.run(
        bridge.digital_twin_execute_robot_function(
            "dual robots",
            "ur5e",
            "pick_grasp",
            origin_resource_location="prusa-mk4-2",
            part_name="MG",
            confirmed=True,
        )
    )

    assert result["success"] is False
    assert result["status"] == "blocked"
    assert "pick_approach.descend" in result["message"]
    assert agent.calls == []
    assert agent._robot_motion_lock.locked() is False
    assert bridge._ur5e_robot_function_state_uncertain is False


@pytest.mark.parametrize(
    ("robot", "child_frame_id", "position_tolerance_m", "orientation_tolerance_rad"),
    [
        ("ur5e", "tool0", 0.002, 0.0349065850),
        ("xarm6", "link_eef", 0.003, 0.0523598776),
    ],
)
def test_manual_dependent_pose_uses_role_tolerance_and_quaternion_sign(
    robot: str,
    child_frame_id: str,
    position_tolerance_m: float,
    orientation_tolerance_rad: float,
) -> None:
    bridge = object.__new__(SystemBridge)
    expected = {
        "x": 0.0,
        "y": 0.1,
        "z": 1.0,
        "qx": 0.0,
        "qy": 0.0,
        "qz": 0.0,
        "qw": 1.0,
    }
    current = {
        **expected,
        "x": expected["x"] + position_tolerance_m,
        "qw": -1.0,
        "frame_id": "world",
        "child_frame_id": child_frame_id,
    }
    bridge._robot_function_execution_pose_readiness = lambda *_args: {
        "success": True,
        "world_tool0_ready": True,
        "waypoint": {"source": "hardware", "pose": deepcopy(current)},
    }
    controller = SimpleNamespace(
        _xarm6_cartesian_position_tolerance_m=position_tolerance_m,
        _xarm6_cartesian_orientation_tolerance_rad=orientation_tolerance_rad,
    )
    resource_agent = SimpleNamespace(_controller=controller)
    task_context = {
        "resolved_cartesian_positions": {"descend": deepcopy(expected)}
    }
    motion_readiness = {
        "cartesian_position_tolerance_m": position_tolerance_m,
        "cartesian_orientation_tolerance_rad": orientation_tolerance_rad,
    }

    readiness, error = bridge._manual_dependent_function_pose_error(
        "dual robots",
        robot,
        "pick_grasp",
        resource_agent,
        task_context,
        motion_readiness,
    )

    assert error == ""
    assert readiness["manual_pose_translation_error_m"] == pytest.approx(
        position_tolerance_m
    )
    assert readiness["manual_pose_rotation_error_rad"] == pytest.approx(0.0)

    current["x"] += 1e-5
    _readiness, moved_error = bridge._manual_dependent_function_pose_error(
        "dual robots",
        robot,
        "pick_grasp",
        resource_agent,
        task_context,
        motion_readiness,
    )
    assert "position error" in moved_error

    current["x"] = expected["x"]
    half_angle = 0.5 * (orientation_tolerance_rad + 1e-4)
    current.update({"qz": math.sin(half_angle), "qw": math.cos(half_angle)})
    _readiness, rotated_error = bridge._manual_dependent_function_pose_error(
        "dual robots",
        robot,
        "pick_grasp",
        resource_agent,
        task_context,
        motion_readiness,
    )
    assert "rotation error" in rotated_error


def test_pick_grasp_readiness_blocks_invalid_stl_grounded_context() -> None:
    agent = _agent_for("pick_grasp")
    agent._task_ctx["source_stl"] = "/tmp/not-the-actual-mg.STL"
    bridge = _ready_bridge(agent)

    result = asyncio.run(
        bridge.digital_twin_robot_function_execution_readiness(
            "dual robots",
            "ur5e",
            "pick_grasp",
            origin_resource_location="prusa-mk4-2",
            part_name="MG",
        )
    )

    assert result["ready"] is False
    assert "actual MG geometry must come from" in result["message"]
    assert agent.calls == []


def test_pick_grasp_readiness_allows_zero_fingertip_hub_overlap() -> None:
    agent = _agent_for("pick_grasp")
    agent._task_ctx["finger_hub_overlap_m"] = 0.0
    bridge = _ready_bridge(agent)

    result = asyncio.run(
        bridge.digital_twin_robot_function_execution_readiness(
            "dual robots",
            "ur5e",
            "pick_grasp",
            origin_resource_location="prusa-mk4-2",
            part_name="MG",
        )
    )

    assert result["ready"] is True
    assert "overlap" not in result["message"]
    assert agent.calls == []


def test_mg_close_test_closes_once_holds_and_reopens_without_arm_motion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _run_to_thread_inline(monkeypatch)
    agent = _agent_for("pick_grasp")
    bridge = _ready_bridge(agent)

    result = asyncio.run(
        bridge.digital_twin_execute_mg_close_test(
            "ur5e only",
            "ur5e",
            "prusa-mk4-2",
            confirmed=True,
        )
    )

    assert result["success"] is True
    assert result["gripper_close_position"] == pytest.approx(0.047)
    assert result["hold_sec"] == pytest.approx(0.0)
    assert agent._controller.gripper_calls == [
        ("close_gripper", pytest.approx(0.047)),
        ("open_gripper", None),
    ]
    assert agent.calls == []
    assert agent._current_state == "at_pick"
    assert agent._gripper_state == "open"
    assert bridge._ur5e_robot_function_state_uncertain is False


@pytest.mark.parametrize(
    ("part_name", "gripper_close_position"),
    [("SG", 0.061), ("MG", 0.047), ("LG", 0.033)],
)
def test_gripper_close_test_uses_each_retained_part_position_without_arm_motion(
    part_name: str,
    gripper_close_position: float,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _run_to_thread_inline(monkeypatch)
    agent = _agent_for("pick_grasp")
    agent._task_ctx.update(
        {
            "part_name": part_name,
            "gripper_close_position": gripper_close_position,
        }
    )
    bridge = _ready_bridge(agent)

    result = asyncio.run(
        bridge.digital_twin_execute_gripper_close_test(
            "ur5e only",
            "ur5e",
            "prusa-mk4-2",
            part_name,
            confirmed=True,
        )
    )

    assert result["success"] is True
    assert result["part_name"] == part_name
    assert result["gripper_close_position"] == pytest.approx(gripper_close_position)
    assert agent._controller.gripper_calls == [
        ("close_gripper", pytest.approx(gripper_close_position)),
        ("open_gripper", None),
    ]
    assert agent.calls == []
    assert agent._current_state == "at_pick"
    assert agent._gripper_state == "open"


@pytest.mark.parametrize(
    ("close_success", "reopen_success", "failed_action"),
    [
        (False, True, "close"),
        (True, False, "reopen"),
    ],
)
def test_mg_close_test_reopens_in_finally_and_failed_cycle_requires_move_home(
    close_success: bool,
    reopen_success: bool,
    failed_action: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _run_to_thread_inline(monkeypatch)
    agent = _agent_for("pick_grasp")
    agent._controller.close_success = close_success
    agent._controller.reopen_success = reopen_success
    bridge = _ready_bridge(agent)

    result = asyncio.run(
        bridge.digital_twin_execute_mg_close_test(
            "dual robots",
            "ur5e",
            "prusa-mk4-2",
            confirmed=True,
        )
    )

    assert result["success"] is False
    assert failed_action in result["message"]
    assert "complete move_home" not in result["message"]
    assert "inspect the UR5e" in result["message"]
    assert agent._controller.gripper_calls[-1] == ("open_gripper", None)
    assert agent.calls == []
    assert bridge._ur5e_robot_function_state_uncertain is True

    readiness = asyncio.run(
        bridge.digital_twin_robot_function_execution_readiness(
            "dual robots",
            "ur5e",
            "pick_grasp",
            origin_resource_location="prusa-mk4-2",
            part_name="MG",
        )
    )
    assert readiness["ready"] is True
    assert "state is uncertain" not in readiness["message"]


def test_mg_close_test_requires_at_pick_empty_mg_context_and_motion_locks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _run_to_thread_inline(monkeypatch)
    idle_agent = _PhysicalUR5eAgent(state="idle")
    idle_bridge = _ready_bridge(idle_agent)

    idle = asyncio.run(
        idle_bridge.digital_twin_execute_mg_close_test(
            "ur5e only",
            "ur5e",
            "prusa-mk4-2",
            confirmed=True,
        )
    )

    assert idle["success"] is False
    assert "active pick context" in idle["message"]
    assert idle_agent._controller.gripper_calls == []

    closed_agent = _agent_for("pick_grasp")
    closed_agent._gripper_state = "closed"
    closed_bridge = _ready_bridge(closed_agent)
    closed = asyncio.run(
        closed_bridge.digital_twin_execute_mg_close_test(
            "ur5e only",
            "ur5e",
            "prusa-mk4-2",
            confirmed=True,
        )
    )
    assert closed["success"] is False
    assert "empty RG2 to be open" in closed["message"]
    assert closed_agent._controller.gripper_calls == []

    locked_agent = _agent_for("pick_grasp")
    locked_bridge = _ready_bridge(locked_agent)
    locked_agent._robot_motion_lock.acquire()
    try:
        locked = asyncio.run(
            locked_bridge.digital_twin_execute_mg_close_test(
                "ur5e only",
                "ur5e",
                "prusa-mk4-2",
                confirmed=True,
            )
        )
    finally:
        locked_agent._robot_motion_lock.release()

    assert locked["success"] is False
    assert "already executing" in locked["message"]
    assert locked_agent._controller.gripper_calls == []


def test_readiness_waits_for_a_running_ur5e_mirror_to_recover() -> None:
    bridge = _ready_bridge(_agent_for("move_home"))
    bridge._ROBOT_FUNCTION_MIRROR_RECOVERY_TIMEOUT_S = 1.0
    calls = [0]

    def _statuses() -> dict[str, dict[str, Any]]:
        calls[0] += 1
        if calls[0] == 1:
            return {
                "ur5e only": {
                    "target": "ur5e only",
                    "repair_needed": True,
                    "repair_reason": "ur5e mirror is waiting",
                    "gazebo": {"status": "running"},
                    "hardware": {"overall": "running"},
                    "sync/status": {
                        "process_status": "running",
                        "state": "waiting",
                    },
                }
            }
        return {"ur5e only": _healthy_status("ur5e only")}

    bridge.digital_twin_statuses = _statuses

    error = asyncio.run(
        bridge._wait_for_digital_twin_robot_function_target_error(
            "ur5e only",
            bridge._DIGITAL_TWIN_TARGETS["ur5e only"],
        )
    )

    assert error == ""
    assert calls[0] == 2


def test_readiness_does_not_wait_for_a_stopped_ur5e_mirror_process() -> None:
    bridge = _ready_bridge(_agent_for("move_home"))
    bridge._ROBOT_FUNCTION_MIRROR_RECOVERY_TIMEOUT_S = 1.0
    calls = [0]

    def _statuses() -> dict[str, dict[str, Any]]:
        calls[0] += 1
        return {
            "ur5e only": {
                "target": "ur5e only",
                "repair_needed": True,
                "repair_reason": "hardware -> gazebo mirror process is stopped",
                "gazebo": {"status": "running"},
                "hardware": {"overall": "running"},
                "sync/status": {
                    "process_status": "stopped",
                    "state": "waiting",
                },
            }
        }

    bridge.digital_twin_statuses = _statuses

    error = asyncio.run(
        bridge._wait_for_digital_twin_robot_function_target_error(
            "ur5e only",
            bridge._DIGITAL_TWIN_TARGETS["ur5e only"],
        )
    )

    assert "requires Repair Twin" in error
    assert calls[0] == 1


@pytest.mark.parametrize(
    ("target", "robot", "function_name", "arguments", "message"),
    [
        ("xarm only", "ur5e", "move_home", {}, "ur5e is not part"),
        ("unknown", "ur5e", "move_home", {}, "unknown digital twin target"),
        ("dual robots", "UR5E", "move_home", {}, "unknown robot"),
        ("dual robots", "ur5e", "Pick_Approach", {}, "unknown robot function"),
        (
            "dual robots",
            "ur5e",
            "pick_grasp",
            {
                "origin_resource_location": "prusa-mk4-2",
                "destination_location": "assembly_board-v1",
                "part_name": "MG",
            },
            "does not accept destination_location",
        ),
        (
            "dual robots",
            "ur5e",
            "move_home",
            {"part_name": "MG"},
            "does not accept location or part arguments",
        ),
    ],
)
def test_readiness_rejects_nonexact_or_irrelevant_requests(
    target: str,
    robot: str,
    function_name: str,
    arguments: dict[str, str],
    message: str,
) -> None:
    bridge = _ready_bridge(_PhysicalUR5eAgent())

    result = asyncio.run(
        bridge.digital_twin_robot_function_execution_readiness(
            target,
            robot,
            function_name,
            **arguments,
        )
    )

    assert result["ready"] is False
    assert message in result["message"]


@pytest.mark.parametrize("function_name", ["pick_approach", "place_approach"])
def test_approach_functions_use_only_configured_product_when_unselected(
    function_name: str,
) -> None:
    agent = _agent_for(function_name)
    bridge = _ready_bridge(agent)
    bridge.selected_product = ""
    product_file = "/tmp/assembly_board-v1.json"
    bridge.list_product_files = lambda: [product_file]
    arguments = (
        {"origin_resource_location": "prusa-mk4-2", "part_name": "MG"}
        if function_name == "pick_approach"
        else {"destination_location": "assembly_board-v1", "part_name": "MG"}
    )

    readiness = asyncio.run(
        bridge.digital_twin_robot_function_execution_readiness(
            "dual robots",
            "ur5e",
            function_name,
            **arguments,
        )
    )
    result = asyncio.run(
        bridge.digital_twin_execute_robot_function(
            "dual robots",
            "ur5e",
            function_name,
            **arguments,
            confirmed=True,
        )
    )

    assert readiness["ready"] is True
    assert readiness["selected_product"] == product_file
    assert result["success"] is True
    assert agent.calls[0][0] == function_name
    assert agent.calls[0][1]["part_name"] == "MG"
    assert agent.calls[0][1]["product_geometry"]["part_name"] == "MG"


@pytest.mark.parametrize("function_name", ["pick_approach", "place_approach"])
def test_approach_functions_require_selection_when_multiple_products_are_configured(
    function_name: str,
) -> None:
    agent = _agent_for(function_name)
    bridge = _ready_bridge(agent)
    bridge.selected_product = ""
    bridge.list_product_files = lambda: ["product-a.json", "product-b.json"]
    bridge._robot_function_product_geometry_for_part = lambda _part: pytest.fail(
        "geometry lookup must not choose between multiple products"
    )
    arguments = (
        {"origin_resource_location": "prusa-mk4-2", "part_name": "MG"}
        if function_name == "pick_approach"
        else {"destination_location": "assembly_board-v1", "part_name": "MG"}
    )

    result = asyncio.run(
        bridge.digital_twin_execute_robot_function(
            "dual robots",
            "ur5e",
            function_name,
            **arguments,
            confirmed=True,
        )
    )

    assert result["success"] is False
    assert "Select a product" in result["message"]
    assert agent.calls == []


def test_preflight_requires_exact_executable_membership() -> None:
    agent = _agent_for("pick_grasp")
    agent.executables.pop("pick_grasp")
    bridge = _ready_bridge(agent)

    result = asyncio.run(
        bridge.digital_twin_execute_pick_grasp(
            "dual robots",
            "ur5e",
            "prusa-mk4-2",
            "MG",
            confirmed=True,
        )
    )

    assert result["success"] is False
    assert "does not expose executable pick_grasp" in result["message"]
    assert agent.calls == []


@pytest.mark.parametrize(
    ("function_name", "pose_name", "pose", "arguments"),
    [
        (
            "pick_approach",
            "prusa-mk4-2",
            [0.0] * 5,
            {"origin_resource_location": "prusa-mk4-2", "part_name": "MG"},
        ),
        (
            "move_home",
            "home",
            [0.0, 0.0, 0.0, 0.0, 0.0, float("nan")],
            {},
        ),
    ],
)
def test_named_position_gates_reject_incomplete_or_nonfinite_joints(
    function_name: str,
    pose_name: str,
    pose: list[float],
    arguments: dict[str, str],
) -> None:
    agent = _agent_for(function_name)
    agent.named_positions[pose_name] = pose
    bridge = _ready_bridge(agent)

    result = asyncio.run(
        bridge.digital_twin_robot_function_execution_readiness(
            "dual robots",
            "ur5e",
            function_name,
            **arguments,
        )
    )

    assert result["ready"] is False
    assert "joint values" in result["message"]
    assert agent.calls == []


@pytest.mark.parametrize("function_name", ["place_approach", "place_insert"])
def test_place_functions_require_logically_closed_gripper(function_name: str) -> None:
    agent = _agent_for(function_name)
    agent._gripper_state = "open"
    bridge = _ready_bridge(agent)
    arguments = {"destination_location": "assembly_board-v1", "part_name": "MG"}

    result = asyncio.run(
        bridge.digital_twin_execute_robot_function(
            "dual robots",
            "ur5e",
            function_name,
            **arguments,
            confirmed=True,
        )
    )

    assert result["success"] is False
    assert "gripper_state 'closed'" in result["message"]
    assert agent.calls == []


@pytest.mark.parametrize(
    "status",
    [
        {
            "accepted": False,
            "accepted_generation": 0,
            "accepted_baseline_ready": False,
            "accepted_baseline_error": (
                "Locate & Accept Board for ur5e before using assembly_board-v1."
            ),
            "post_staging_acceptance_allowed": True,
        },
        {
            "accepted": True,
            "accepted_generation": 5,
            "accepted_baseline_ready": False,
            "accepted_baseline_error": (
                "assembly_board-v1 moved more than 10 mm or 2 deg from the accepted "
                "ur5e pose."
            ),
            "movement_blocked": True,
            "post_staging_acceptance_allowed": True,
        },
    ],
)
def test_place_approach_allows_post_staging_board_acceptance(
    status: dict[str, Any],
) -> None:
    agent = _agent_for("place_approach")
    bridge = _ready_bridge(agent)
    bridge.perception_manager.assembly_board_v1_aruco_status = lambda _role: dict(status)

    result = asyncio.run(
        bridge.digital_twin_execute_robot_function(
            "dual robots",
            "ur5e",
            "place_approach",
            destination_location="assembly_board-v1",
            part_name="MG",
            confirmed=True,
        )
    )

    assert result["success"] is True
    assert agent.calls == [
        (
            "place_approach",
            {
                "destination_location": "assembly_board-v1",
                "part_name": "MG",
                "product_geometry": _mg_product_geometry(),
            },
        )
    ]


def test_place_approach_still_blocks_post_staging_acceptance_after_calibration_change() -> None:
    agent = _agent_for("place_approach")
    bridge = _ready_bridge(agent)
    bridge.perception_manager.assembly_board_v1_aruco_status = lambda _role: {
        "accepted": True,
        "accepted_generation": 5,
        "accepted_baseline_ready": False,
        "accepted_baseline_error": "The active ur5e calibration identity changed.",
        "calibration_changed": True,
        "post_staging_acceptance_allowed": False,
    }

    result = asyncio.run(
        bridge.digital_twin_execute_robot_function(
            "dual robots",
            "ur5e",
            "place_approach",
            destination_location="assembly_board-v1",
            part_name="MG",
            confirmed=True,
        )
    )

    assert result["success"] is False
    assert "calibration identity changed" in result["message"]
    assert agent.calls == []


def test_place_approach_post_staging_acceptance_is_scoped_to_confirmed_runtime() -> None:
    agent = _agent_for("place_approach")
    bridge = _ready_bridge(agent)
    calls: list[tuple[str, object]] = []

    bridge.perception_manager.assembly_board_v1_aruco_status = lambda role: {
        "camera_role": role,
        "accepted": True,
        "accepted_generation": 5,
        "accepted_baseline_ready": False,
        "accepted_baseline_error": "board movement requires post-staging acceptance",
        "movement_blocked": True,
        "calibration_changed": False,
        "post_staging_acceptance_allowed": True,
    }

    def _accept(role: str, *, minimum_sample_started_at: float | None = None) -> dict[str, Any]:
        calls.append((role, minimum_sample_started_at))
        return {"success": True, "accepted_generation": 6}

    bridge.perception_manager.locate_and_accept_assembly_board_v1 = _accept

    async def _place_approach(**kwargs: Any) -> dict[str, Any]:
        callback = getattr(
            agent._controller,
            "_assembly_board_v1_post_staging_accept_callback",
        )
        assert callable(callback)
        accepted = callback(123.5)
        assert accepted["accepted_generation"] == 6
        agent.calls.append(("place_approach", kwargs))
        return {"status": "completed", "content": "Reached assembly_board-v1."}

    agent.place_approach = _place_approach

    result = asyncio.run(
        bridge.digital_twin_execute_robot_function(
            "dual robots",
            "ur5e",
            "place_approach",
            destination_location="assembly_board-v1",
            part_name="MG",
            confirmed=True,
        )
    )

    assert result["success"] is True
    assert calls == [("ur5e", 123.5)]
    assert not hasattr(
        agent._controller,
        "_assembly_board_v1_post_staging_accept_callback",
    )


def test_place_approach_accepts_occluded_but_usable_board_baseline() -> None:
    agent = _agent_for("place_approach")
    bridge = _ready_bridge(agent)
    bridge.perception_manager.assembly_board_v1_aruco_status = lambda _role: {
        "accepted": True,
        "accepted_baseline_ready": True,
        "accepted_baseline_error": "",
        "accepted_generation": 1,
        "accepted_calibration_id": "ur5e-calibration",
        "visible": False,
        "movement_evidence_valid": False,
        "movement_blocked": False,
    }

    result = asyncio.run(
        bridge.digital_twin_robot_function_execution_readiness(
            "dual robots",
            "ur5e",
            "place_approach",
            destination_location="assembly_board-v1",
            part_name="MG",
        )
    )

    assert result["ready"] is True


def test_place_insert_rejects_reaccepted_board_before_confirmation() -> None:
    agent = _agent_for("place_insert")
    bridge = _ready_bridge(agent)
    bridge.perception_manager.assembly_board_v1_aruco_status = lambda _role: {
        "accepted": True,
        "accepted_baseline_ready": True,
        "accepted_baseline_error": "",
        "accepted_generation": 2,
        "accepted_calibration_id": "ur5e-calibration",
    }

    result = asyncio.run(
        bridge.digital_twin_robot_function_execution_readiness(
            "dual robots",
            "ur5e",
            "place_insert",
            destination_location="assembly_board-v1",
            part_name="MG",
        )
    )

    assert result["ready"] is False
    assert "accepted generation changed after place_approach" in result["message"]
    assert agent.calls == []


@pytest.mark.parametrize(
    ("bridge_method", "manager_method", "expected"),
    [
        (
            "perception_locate_and_accept_assembly_board_v1",
            "locate_and_accept_assembly_board_v1",
            {"success": True, "accepted_generation": 2},
        ),
        (
            "perception_activate_calibration",
            "activate_calibration",
            Path("/tmp/ur5e_realsense_hand_eye.yaml"),
        ),
        (
            "perception_rollback_calibration",
            "rollback_calibration",
            Path("/tmp/ur5e_realsense_hand_eye.yaml"),
        ),
    ],
)
def test_board_acceptance_and_calibration_identity_changes_use_execution_lock(
    bridge_method: str,
    manager_method: str,
    expected: object,
) -> None:
    bridge = _ready_bridge(_agent_for("place_approach"))
    calls: list[str] = []

    def _change(role: str) -> object:
        assert bridge._ur5e_robot_function_execution_lock.locked() is True
        calls.append(role)
        return expected

    setattr(bridge.perception_manager, manager_method, _change)

    result = getattr(bridge, bridge_method)("ur5e")

    assert result == expected
    assert calls == ["ur5e"]
    assert bridge._ur5e_robot_function_execution_lock.acquire(blocking=False) is True
    bridge._ur5e_robot_function_execution_lock.release()


@pytest.mark.parametrize(
    ("bridge_method", "manager_method", "operation_name"),
    [
        (
            "perception_locate_and_accept_assembly_board_v1",
            "locate_and_accept_assembly_board_v1",
            "Locate & Accept Board (xarm6)",
        ),
        (
            "perception_activate_calibration",
            "activate_calibration",
            "Activate calibration (xarm6)",
        ),
        (
            "perception_rollback_calibration",
            "rollback_calibration",
            "Rollback calibration (xarm6)",
        ),
    ],
)
def test_board_acceptance_and_calibration_identity_changes_refuse_during_execution(
    bridge_method: str,
    manager_method: str,
    operation_name: str,
) -> None:
    bridge = _ready_bridge(_agent_for("place_approach"))
    calls: list[str] = []
    setattr(
        bridge.perception_manager,
        manager_method,
        lambda role: calls.append(role),
    )
    bridge._ur5e_robot_function_execution_active = "place_approach"
    bridge._ur5e_robot_function_execution_lock.acquire()
    try:
        with pytest.raises(RuntimeError) as error:
            getattr(bridge, bridge_method)("xarm6")
    finally:
        bridge._ur5e_robot_function_execution_lock.release()

    assert str(error.value) == (
        f"{operation_name} is unavailable while physical robot function execution is active: "
        "place_approach."
    )
    assert calls == []


def test_board_acceptance_releases_execution_lock_after_perception_failure() -> None:
    bridge = _ready_bridge(_agent_for("place_approach"))

    def _fail(_role: str) -> dict[str, Any]:
        raise RuntimeError("fresh stable ArUco observation is unavailable")

    bridge.perception_manager.locate_and_accept_assembly_board_v1 = _fail

    with pytest.raises(RuntimeError, match="fresh stable ArUco observation is unavailable"):
        bridge.perception_locate_and_accept_assembly_board_v1("ur5e")

    assert bridge._ur5e_robot_function_execution_lock.acquire(blocking=False) is True
    bridge._ur5e_robot_function_execution_lock.release()


def test_automatic_board_acceptance_is_idempotent_after_one_accepted_generation() -> None:
    bridge = _ready_bridge(_agent_for("place_approach"))
    calls: list[str] = []

    def _status(role: str) -> dict[str, Any]:
        assert bridge._ur5e_robot_function_execution_lock.locked() is True
        calls.append(f"status:{role}")
        return {
            "camera_role": role,
            "accepted": False,
            "accepted_generation": 4,
            "accepted_baseline_ready": True,
        }

    def _accept(role: str) -> dict[str, Any]:
        calls.append(f"accept:{role}")
        return {"success": True, "accepted_generation": 5}

    bridge.perception_manager.assembly_board_v1_aruco_status = _status
    bridge.perception_manager.locate_and_accept_assembly_board_v1 = _accept

    result = bridge.perception_locate_and_accept_assembly_board_v1(
        "ur5e",
        only_if_unaccepted=True,
    )

    assert result == {
        "camera_role": "ur5e",
        "accepted": False,
        "accepted_generation": 4,
        "accepted_baseline_ready": True,
        "success": True,
        "auto_accepted": False,
    }
    assert calls == ["status:ur5e"]


def test_automatic_board_acceptance_accepts_one_unaccepted_observation() -> None:
    bridge = _ready_bridge(_agent_for("place_approach"))
    calls: list[str] = []

    def _status(role: str) -> dict[str, Any]:
        assert bridge._ur5e_robot_function_execution_lock.locked() is True
        calls.append(f"status:{role}")
        return {
            "camera_role": role,
            "accepted": False,
            "accepted_generation": 0,
            "ready_to_accept": True,
        }

    def _accept(role: str) -> dict[str, Any]:
        assert bridge._ur5e_robot_function_execution_lock.locked() is True
        calls.append(f"accept:{role}")
        return {
            "success": True,
            "camera_role": role,
            "accepted": True,
            "accepted_generation": 1,
        }

    bridge.perception_manager.assembly_board_v1_aruco_status = _status
    bridge.perception_manager.locate_and_accept_assembly_board_v1 = _accept

    result = bridge.perception_locate_and_accept_assembly_board_v1(
        "xarm6",
        only_if_unaccepted=True,
    )

    assert result == {
        "success": True,
        "camera_role": "xarm6",
        "accepted": True,
        "accepted_generation": 1,
        "auto_accepted": True,
    }
    assert calls == ["status:xarm6", "accept:xarm6"]


def test_solving_calibration_candidate_remains_unlocked() -> None:
    bridge = _ready_bridge(_agent_for("place_approach"))
    expected = Path("/tmp/ur5e_realsense_hand_eye.candidate.yaml")
    bridge.perception_manager.solve_calibration = lambda role: expected
    bridge._ur5e_robot_function_execution_active = "place_insert"
    bridge._ur5e_robot_function_execution_lock.acquire()
    try:
        assert bridge.perception_solve_calibration("ur5e") == expected
    finally:
        bridge._ur5e_robot_function_execution_lock.release()


def test_gripper_functions_require_exact_target_domain_rg2_action() -> None:
    agent = _agent_for("pick_grasp")
    bridge = _ready_bridge(agent)
    bridge._digital_twin_ur5e_gripper_readiness = lambda target, domain, _agent=None: (
        {"gripper_action_ready": False},
        (
            "/ur5e_rg2_gripper_traj_controller/follow_joint_trajectory is unavailable "
            f"on ROS_DOMAIN_ID={domain} for {target}"
        ),
    )

    result = asyncio.run(
        bridge.digital_twin_execute_pick_grasp(
            "ur5e only",
            "ur5e",
            "prusa-mk4-2",
            "MG",
            confirmed=True,
        )
    )

    assert result["success"] is False
    assert "ROS_DOMAIN_ID=42" in result["message"]
    assert agent.calls == []


def test_motion_and_gripper_probes_use_exact_actions_and_target_domain() -> None:
    bridge = object.__new__(SystemBridge)
    bridge._digital_twin_domain_ids = lambda: {
        "gazebo": 41,
        "hardware": 42,
        "hardware_xarm6": 42,
        "hardware_ur5e": 43,
    }
    bridge._digital_twin_hardware_domain_id = lambda _cfg, _robot, _domains: 43
    rtde_status = {
        "updated_at": time.time(),
        "ros_domain_id": 43,
        "action_name": "/cais_ur5e_rtde_trajectory_controller/follow_joint_trajectory",
        "rtde_receive_connected": True,
        "joint_states_fresh": True,
        "rtde_control_connected": True,
    }
    bridge._ur5e_rtde_trajectory_status = lambda: dict(rtde_status)
    bridge._ur5e_rtde_result_timeout_requires_repair = lambda _status: False
    probes: list[tuple[str, int | None]] = []

    def _wait(action: str, **kwargs: Any) -> None:
        probes.append((action, kwargs.get("ros_domain_id")))
        return None

    bridge._wait_for_ros_action = _wait
    motion, motion_error = bridge._digital_twin_ur5e_motion_readiness(
        "dual robots", {"hardware": ("xarm6", "ur5e")}
    )
    gripper, gripper_error = bridge._digital_twin_ur5e_gripper_readiness("dual robots", 43)

    assert motion_error == ""
    assert gripper_error == ""
    assert motion["trajectory_action_ready"] is True
    assert gripper["gripper_action_ready"] is True
    assert probes == [
        ("/cais_ur5e_rtde_trajectory_controller/follow_joint_trajectory", 43),
        ("/ur5e_rg2_gripper_traj_controller/follow_joint_trajectory", 43),
    ]
    rtde_status["ros_domain_id"] = 42
    _wrong_domain, wrong_domain_error = bridge._digital_twin_ur5e_motion_readiness(
        "dual robots", {"hardware": ("xarm6", "ur5e")}
    )
    assert "belongs to ROS_DOMAIN_ID=42" in wrong_domain_error
    assert "requested ROS_DOMAIN_ID=43" in wrong_domain_error
    assert len(probes) == 2


def _recorded_step(step_name: str, z: float) -> dict[str, Any]:
    pose = {
        "x": -0.11,
        "y": 0.42,
        "z": z,
        "qx": 0.0,
        "qy": 0.70710678,
        "qz": 0.0,
        "qw": 0.70710678,
    }
    return {
        "step_name": step_name,
        "primitive": "move_cartesian",
        "params": deepcopy(pose),
        "capture_source": "hardware",
        "position_sources": {
            "x": "captured_relative",
            "y": "captured_relative",
            "z": "captured_relative",
        },
        "relative_position_m": {
            "x": -0.61,
            "y": -0.18,
            "z": z - 0.3,
        },
        "relative_pose": {
            "x": -0.61,
            "y": -0.18,
            "z": z - 0.3,
            "qx": 0.0,
            "qy": 0.70710678,
            "qz": 0.0,
            "qw": 0.70710678,
        },
        "relative_reference": {
            "kind": "destination_target",
            "frame_id": "world",
            "name": "assembly_board-v1",
            "position_m": {"x": 0.5, "y": 0.6, "z": 0.3},
            "pose": {
                "x": 0.5,
                "y": 0.6,
                "z": 0.3,
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
                **deepcopy(pose),
            },
            "joint_names": [
                "shoulder_pan_joint",
                "shoulder_lift_joint",
                "elbow_joint",
                "wrist_1_joint",
                "wrist_2_joint",
                "wrist_3_joint",
            ],
            "joint_positions": [0.0] * 6,
            "source": "hardware",
        },
    }


def test_place_readiness_runs_complete_runtime_recording_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(robot_task_runtime, "_TAUGHT_FUNCTIONS_ROOT", tmp_path)
    path = tmp_path / "ur5e" / "place_approach" / "assembly_board-v1__MG__hardware.json"
    path.parent.mkdir(parents=True)
    payload = {
        "robot": "ur5e",
        "function_name": "place_approach",
        "name": "assembly_board-v1",
        "part_name": "MG",
        "capture_source": "hardware",
        "steps": [
            _recorded_step("move_above_destination", 0.69),
            _recorded_step("descend", 0.36),
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    agent = _agent_for("place_approach")
    bridge = _ready_bridge(agent)
    del bridge._digital_twin_place_approach_recording_error

    async def _check_both_recordings() -> tuple[dict[str, Any], dict[str, Any]]:
        ready_result = await bridge.digital_twin_robot_function_execution_readiness(
            "dual robots",
            "ur5e",
            "place_approach",
            destination_location="assembly_board-v1",
            part_name="MG",
        )
        payload["steps"][1] = deepcopy(payload["steps"][0])
        path.write_text(json.dumps(payload), encoding="utf-8")
        blocked_result = await bridge.digital_twin_robot_function_execution_readiness(
            "dual robots",
            "ur5e",
            "place_approach",
            destination_location="assembly_board-v1",
            part_name="MG",
        )
        return ready_result, blocked_result

    ready, blocked = asyncio.run(_check_both_recordings())
    assert ready["ready"] is True
    assert blocked["ready"] is False
    assert "duplicate physical position step_name" in blocked["message"]


def test_confirmation_and_generalized_motion_lock_block_before_preflight() -> None:
    agent = _agent_for("move_home")
    bridge = _ready_bridge(agent)
    unconfirmed = asyncio.run(
        bridge.digital_twin_execute_robot_function("dual robots", "ur5e", "move_home")
    )
    assert "Explicit operator confirmation" in unconfirmed["message"]

    bridge._ur5e_robot_function_execution_lock.acquire()
    bridge._ur5e_robot_function_execution_active = "place_insert"
    busy = asyncio.run(
        bridge.digital_twin_execute_robot_function(
            "dual robots", "ur5e", "move_home", confirmed=True
        )
    )
    bridge._ur5e_robot_function_execution_lock.release()
    assert busy["active_function"] == "place_insert"
    assert agent.calls == []


@pytest.mark.parametrize("failure_kind", ["result", "exception"])
def test_post_dispatch_failure_warns_that_physical_state_may_have_changed(
    failure_kind: str,
) -> None:
    agent = _agent_for("move_home")

    async def _fail(**_kwargs: Any) -> dict[str, Any]:
        if failure_kind == "exception":
            raise RuntimeError("RTDE transport ended")
        return {
            "status": "failed",
            "content": "move_to_named_pose failed",
            "failure_context": {"observations": {"step": "move_home.move_home"}},
        }

    agent.move_home = _fail
    bridge = _ready_bridge(agent)
    result = asyncio.run(
        bridge.digital_twin_execute_robot_function(
            "dual robots", "ur5e", "move_home", confirmed=True
        )
    )

    assert result["success"] is False
    assert "Physical state may have changed" in result["message"]
    assert "inspect the robot and recover before retrying" in result["message"]
    if failure_kind == "result":
        assert "Failed step: move_home.move_home" in result["message"]


def test_ui_cancellation_keeps_generalized_lock_until_runtime_finishes() -> None:
    async def _exercise() -> None:
        agent = _agent_for("pick_approach")
        bridge = _ready_bridge(agent)
        started = asyncio.Event()
        finish = asyncio.Event()

        async def _slow_pick_approach(**kwargs: Any) -> dict[str, Any]:
            agent.calls.append(("pick_approach", kwargs))
            started.set()
            await finish.wait()
            return {"status": "completed", "content": "motion complete"}

        agent.pick_approach = _slow_pick_approach
        execution = asyncio.create_task(
            bridge.digital_twin_execute_pick_approach(
                "dual robots",
                "ur5e",
                "prusa-mk4-2",
                "MG",
                confirmed=True,
            )
        )
        await started.wait()
        execution.cancel()
        with pytest.raises(asyncio.CancelledError):
            await execution

        assert bridge._ur5e_robot_function_execution_lock.acquire(blocking=False) is False
        assert bridge._ur5e_robot_function_execution_active == "pick_approach"

        finish.set()
        for _attempt in range(20):
            await asyncio.sleep(0)
            if bridge._ur5e_robot_function_execution_lock.acquire(blocking=False):
                bridge._ur5e_robot_function_execution_lock.release()
                break
        else:
            pytest.fail("UR5e lock was not released after the agent runtime completed")
        assert bridge._ur5e_robot_function_execution_active is None

    asyncio.run(_exercise())


def test_ui_cancellation_keeps_generalized_lock_until_preflight_finishes() -> None:
    async def _exercise() -> None:
        agent = _agent_for("move_home")
        bridge = _ready_bridge(agent)
        started = threading.Event()
        finish = threading.Event()
        original_preflight = bridge._digital_twin_robot_function_execution_preflight

        def _slow_preflight(*args: Any) -> tuple[Any | None, dict[str, Any], dict[str, Any], str]:
            started.set()
            assert finish.wait(timeout=2.0)
            return original_preflight(*args)

        bridge._digital_twin_robot_function_execution_preflight = _slow_preflight
        execution = asyncio.create_task(
            bridge.digital_twin_execute_robot_function(
                "dual robots",
                "ur5e",
                "move_home",
                confirmed=True,
            )
        )
        for _attempt in range(100):
            if started.is_set():
                break
            await asyncio.sleep(0.01)
        else:
            pytest.fail("UR5e preflight worker did not start")

        execution.cancel()
        with pytest.raises(asyncio.CancelledError):
            await execution
        assert bridge._ur5e_robot_function_execution_lock.acquire(blocking=False) is False
        assert bridge._ur5e_robot_function_execution_active == "move_home"

        finish.set()
        for _attempt in range(100):
            await asyncio.sleep(0.01)
            if bridge._ur5e_robot_function_execution_lock.acquire(blocking=False):
                bridge._ur5e_robot_function_execution_lock.release()
                break
        else:
            pytest.fail("UR5e lock was not released after preflight completed")
        assert bridge._ur5e_robot_function_execution_active is None
        assert agent.calls == []

    asyncio.run(_exercise())


def test_robot_function_preflights_are_serialized() -> None:
    async def _exercise() -> None:
        bridge = _ready_bridge(_agent_for("move_home"))
        first_started = threading.Event()
        release_first = threading.Event()
        second_started = threading.Event()
        call_count = 0
        active_count = 0
        max_active = 0
        count_lock = threading.Lock()

        def _preflight(*_args: Any) -> tuple[Any | None, dict[str, Any], dict[str, Any], str]:
            nonlocal call_count, active_count, max_active
            with count_lock:
                call_count += 1
                call_number = call_count
                active_count += 1
                max_active = max(max_active, active_count)
            if call_number == 1:
                first_started.set()
                assert release_first.wait(timeout=2.0)
            else:
                second_started.set()
            with count_lock:
                active_count -= 1
            return None, {}, {}, "expected test stop"

        bridge._digital_twin_robot_function_execution_preflight = _preflight
        first = asyncio.create_task(
            bridge.digital_twin_robot_function_execution_readiness(
                "dual robots", "ur5e", "move_home"
            )
        )
        for _attempt in range(100):
            if first_started.is_set():
                break
            await asyncio.sleep(0.01)
        else:
            pytest.fail("first UR5e preflight did not start")
        second = asyncio.create_task(
            bridge.digital_twin_robot_function_execution_readiness(
                "dual robots", "ur5e", "move_home"
            )
        )
        await asyncio.sleep(0.05)
        assert second_started.is_set() is False
        release_first.set()
        await asyncio.gather(first, second)
        assert second_started.is_set() is True
        assert max_active == 1

    asyncio.run(_exercise())


def test_robot_function_preflight_has_a_bounded_wait() -> None:
    bridge = _ready_bridge(_agent_for("move_home"))
    bridge._ROBOT_FUNCTION_PREFLIGHT_TIMEOUT_S = 0.05
    finish = threading.Event()

    def _hung_preflight(
        *_args: Any,
    ) -> tuple[Any | None, dict[str, Any], dict[str, Any], str]:
        assert finish.wait(timeout=2.0)
        return None, {}, {}, "expected test stop"

    bridge._digital_twin_robot_function_execution_preflight = _hung_preflight
    result = asyncio.run(
        bridge.digital_twin_robot_function_execution_readiness("dual robots", "ur5e", "move_home")
    )
    finish.set()

    assert result["ready"] is False
    assert "readiness timed out after 0.05s" in result["message"]


@pytest.mark.parametrize(
    ("function_name", "location_kwargs"),
    [
        (
            "pick_grasp",
            {"origin_resource_location": "prusa-mk4-2", "part_name": "MG"},
        ),
        (
            "place_insert",
            {"destination_location": "assembly_board-v1", "part_name": "MG"},
        ),
    ],
)
def test_relative_lift_functions_require_finite_positive_return_height(
    function_name: str,
    location_kwargs: dict[str, str],
) -> None:
    agent = _agent_for(function_name)
    agent._task_ctx.pop("travel_z")
    bridge = _ready_bridge(agent)

    missing = asyncio.run(
        bridge.digital_twin_robot_function_execution_readiness(
            "dual robots",
            "ur5e",
            function_name,
            **location_kwargs,
        )
    )
    agent._task_ctx["travel_z"] = agent._position["z"]
    nonpositive = asyncio.run(
        bridge.digital_twin_robot_function_execution_readiness(
            "dual robots",
            "ur5e",
            function_name,
            **location_kwargs,
        )
    )

    assert missing["ready"] is False
    assert "task_ctx.travel_z and position.z" in missing["message"]
    assert nonpositive["ready"] is False
    assert "positive lift" in nonpositive["message"]
    assert agent.calls == []


def test_part_options_cover_all_four_exact_part_functions() -> None:
    bridge = object.__new__(SystemBridge)
    bridge.product_geometry_slots_for_product = lambda: ["MG", "SG"]

    for function_name in (
        "pick_approach",
        "pick_grasp",
        "place_approach",
        "place_insert",
    ):
        assert bridge.digital_twin_function_part_options(function_name) == ["MG", "SG"]
    assert bridge.digital_twin_function_part_options("move_home") == []


def test_dual_hardware_function_execution_uses_selected_robot_cartesian_readiness() -> None:
    bridge = object.__new__(SystemBridge)
    bridge._selected_normal_hardware_stack = lambda: "dual robots"
    bridge._teleop_smooth_session = None
    bridge._hardware_cartesian_readiness = {
        "xarm6": {
            "cartesian_jog_ready": True,
            "cartesian_function_ready": True,
            "message": "xArm6 Cartesian frame validation ready",
        },
        "ur5e": {
            "cartesian_jog_ready": False,
            "cartesian_function_ready": False,
            "message": "world -> tool0 disagrees with live RTDE TCP",
        },
    }

    assert bridge._physical_robot_function_cartesian_error("xarm6") == ""
    assert bridge._physical_robot_function_cartesian_error("ur5e") == (
        "Cartesian frame validation failed: "
        "world -> tool0 disagrees with live RTDE TCP"
    )
    bridge._teleop_smooth_session = {"robot": "xarm6", "axis": "z"}
    assert bridge._physical_robot_function_cartesian_error("xarm6") == (
        "Release Cartesian Smooth Hold before Function Execution. "
        "Active robot=xarm6 axis=World Z."
    )
