"""Focused tests for guarded physical UR5e pick execution from SystemBridge."""

from __future__ import annotations

import asyncio
import threading
from typing import Any

import pytest

from cais_spade_llm.ui.bridge import SystemBridge


class _PhysicalUR5eAgent:
    def __init__(self, *, state: str = "idle") -> None:
        self.agent_name = "ur5e"
        self.jid = "ur5e@localhost"
        self.execution_mode = "physical"
        self._controller = object()
        self.named_positions = {
            "prusa-mk4-2": [0.1, -0.8, -2.1, -1.6, 1.5, -3.1],
        }
        self._current_state = state
        self._held_part = None
        self._task_ctx: dict[str, Any] = {}
        self.calls: list[tuple[str, dict[str, Any]]] = []

    @staticmethod
    def is_alive() -> bool:
        return True

    async def pick_approach(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("pick_approach", kwargs))
        self._task_ctx = {
            "part_name": kwargs["part_name"],
            "origin_resource_location": kwargs["origin_resource_location"],
            "gripper_close_position": 0.04,
        }
        self._current_state = "at_pick"
        return {"status": "completed", "content": "Arrived at the live pick target."}

    async def pick_grasp(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("pick_grasp", kwargs))
        self._held_part = kwargs["part_name"]
        self._current_state = "picked"
        return {"status": "completed", "content": "Picked MG."}


def _ready_bridge(agent: _PhysicalUR5eAgent) -> SystemBridge:
    bridge = object.__new__(SystemBridge)
    bridge.system_running = True
    bridge.execution_mode = "physical"
    bridge.resource_agents = [agent]
    bridge._ur5e_pick_execution_lock = threading.Lock()
    bridge._ur5e_pick_execution_active = None
    cfg = {"robot": "dual robots", "hardware": ("xarm6", "ur5e")}
    bridge._robot_function_validate_request = lambda *_args: (cfg, "")
    bridge._digital_twin_is_dual_robots = lambda _cfg: True
    bridge._digital_twin_sim_mode = lambda _target: "monitor"
    bridge._robot_function_capture_source = lambda _target, _cfg: "hardware"
    bridge._digital_twin_direction = lambda _target: "hardware -> gazebo"
    bridge.digital_twin_statuses = lambda: {
        "dual robots": {
            "repair_needed": False,
            "repair_reason": "",
            "gazebo": {"status": "running"},
            "hardware": {"overall": "running"},
            "sync/status": {
                "process_status": "running",
                "state": "mirroring",
            },
        }
    }
    bridge.teleop_named_position_readiness = lambda _robot: (
        True,
        "UR5e trajectory interface and joint feedback are ready",
    )
    bridge.physical_perception_ready = lambda: (True, "")
    bridge._robot_function_product_geometry_for_part = lambda part_name: {
        "part_name": part_name,
        "model_name": "gear_medium",
        "part_height_m": 0.02,
    }

    async def _run_on_agent_runtime(coroutine: Any) -> Any:
        return await coroutine

    bridge._run_on_agent_runtime = _run_on_agent_runtime
    return bridge


def test_pick_approach_dispatches_exact_generated_method_with_product_geometry() -> None:
    agent = _PhysicalUR5eAgent()
    bridge = _ready_bridge(agent)
    caller_thread = threading.get_ident()
    preflight_threads: list[int] = []
    original_preflight = bridge._digital_twin_pick_execution_preflight

    def _record_preflight_thread(*args: Any) -> tuple[Any | None, dict[str, Any], str]:
        preflight_threads.append(threading.get_ident())
        return original_preflight(*args)

    bridge._digital_twin_pick_execution_preflight = _record_preflight_thread

    result = asyncio.run(
        bridge.digital_twin_execute_pick_approach(
            "dual robots",
            "ur5e",
            "prusa-mk4-2",
            "MG",
            confirmed=True,
        )
    )

    assert result["success"] is True
    assert result["status"] == "completed"
    assert result["state"] == "at_pick"
    assert result["message"] == "Arrived at the live pick target."
    assert agent.calls == [
        (
            "pick_approach",
            {
                "origin_resource_location": "prusa-mk4-2",
                "part_name": "MG",
                "product_geometry": {
                    "part_name": "MG",
                    "model_name": "gear_medium",
                    "part_height_m": 0.02,
                },
            },
        )
    ]
    assert bridge._ur5e_pick_execution_active is None
    assert bridge._ur5e_pick_execution_lock.acquire(blocking=False) is True
    bridge._ur5e_pick_execution_lock.release()
    assert preflight_threads and preflight_threads[0] != caller_thread


def test_pick_execution_requires_explicit_confirmation_before_preflight() -> None:
    agent = _PhysicalUR5eAgent()
    bridge = _ready_bridge(agent)
    bridge._digital_twin_pick_execution_preflight = lambda *_args: pytest.fail(
        "preflight must not run before operator confirmation"
    )

    result = asyncio.run(
        bridge.digital_twin_execute_pick_approach(
            "dual robots",
            "ur5e",
            "prusa-mk4-2",
            "MG",
        )
    )

    assert result["success"] is False
    assert "Explicit operator confirmation" in result["message"]
    assert agent.calls == []


def test_pick_approach_rejects_invalid_named_origin_before_perception_or_motion() -> None:
    agent = _PhysicalUR5eAgent()
    agent.named_positions["prusa-mk4-2"] = [0.1] * 5
    bridge = _ready_bridge(agent)
    bridge.teleop_named_position_readiness = lambda _robot: pytest.fail(
        "trajectory readiness must follow named-position validation"
    )
    bridge.physical_perception_ready = lambda: pytest.fail(
        "perception must follow named-position validation"
    )

    result = asyncio.run(
        bridge.digital_twin_execute_pick_approach(
            "dual robots",
            "ur5e",
            "prusa-mk4-2",
            "MG",
            confirmed=True,
        )
    )

    assert result["success"] is False
    assert "exactly six UR5e joint values" in result["message"]
    assert agent.calls == []


def test_pick_approach_requires_trajectory_and_perception_readiness() -> None:
    agent = _PhysicalUR5eAgent()
    bridge = _ready_bridge(agent)
    bridge.physical_perception_ready = lambda: (
        False,
        "Physical mode is blocked: table_plane_ready is false.",
    )

    result = asyncio.run(
        bridge.digital_twin_execute_pick_approach(
            "dual robots",
            "ur5e",
            "prusa-mk4-2",
            "MG",
            confirmed=True,
        )
    )

    assert result["success"] is False
    assert "table_plane_ready" in result["message"]
    assert agent.calls == []


def test_pick_grasp_requires_matching_at_pick_context_and_skips_new_detection_gate() -> None:
    agent = _PhysicalUR5eAgent(state="at_pick")
    agent._task_ctx = {
        "part_name": "MG",
        "origin_resource_location": "prusa-mk4-2",
        "gripper_close_position": 0.04,
    }
    bridge = _ready_bridge(agent)
    bridge.physical_perception_ready = lambda: pytest.fail(
        "pick_grasp must use the established pick context without a new detection"
    )

    result = asyncio.run(
        bridge.digital_twin_execute_pick_grasp(
            "dual robots",
            "ur5e",
            "prusa-mk4-2",
            "MG",
            confirmed=True,
        )
    )

    assert result["success"] is True
    assert result["state"] == "picked"
    assert agent.calls[0][0] == "pick_grasp"
    assert agent.calls[0][1]["origin_resource_location"] == "prusa-mk4-2"
    assert agent.calls[0][1]["part_name"] == "MG"
    assert agent.calls[0][1]["product_geometry"]["model_name"] == "gear_medium"


@pytest.mark.parametrize(
    ("context", "message"),
    [
        (
            {"part_name": "SG", "origin_resource_location": "prusa-mk4-2"},
            "part_name does not match",
        ),
        (
            {"part_name": "MG", "origin_resource_location": "prusa-mk3"},
            "origin_resource_location does not match",
        ),
    ],
)
def test_pick_grasp_rejects_mismatched_pick_context(
    context: dict[str, str],
    message: str,
) -> None:
    agent = _PhysicalUR5eAgent(state="at_pick")
    agent._task_ctx = context
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
    assert message in result["message"]
    assert agent.calls == []


def test_pick_execution_rejects_a_second_ur5e_motion_instead_of_queueing() -> None:
    agent = _PhysicalUR5eAgent()
    bridge = _ready_bridge(agent)
    bridge._ur5e_pick_execution_lock.acquire()
    bridge._ur5e_pick_execution_active = "pick_approach"

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
    assert result["active_function"] == "pick_approach"
    assert "already active" in result["message"]
    assert agent.calls == []
    bridge._ur5e_pick_execution_lock.release()


def test_pick_execution_requires_dual_robots_hardware_led_monitor_mode() -> None:
    agent = _PhysicalUR5eAgent()
    bridge = _ready_bridge(agent)

    result = asyncio.run(
        bridge.digital_twin_execute_pick_approach(
            "ur5e only",
            "ur5e",
            "prusa-mk4-2",
            "MG",
            confirmed=True,
        )
    )

    assert result["success"] is False
    assert "dual robots digital twin" in result["message"]
    assert agent.calls == []


@pytest.mark.parametrize(
    ("status_update", "message"),
    [
        (
            {"repair_needed": True, "repair_reason": "ur5e mirror is waiting"},
            "requires Repair Twin",
        ),
        (
            {"sync/status": {"process_status": "running", "state": "waiting"}},
            "is not mirroring",
        ),
    ],
)
def test_pick_execution_requires_healthy_passive_twin(
    status_update: dict[str, Any],
    message: str,
) -> None:
    agent = _PhysicalUR5eAgent()
    bridge = _ready_bridge(agent)
    healthy = bridge.digital_twin_statuses()["dual robots"]
    healthy.update(status_update)
    bridge.digital_twin_statuses = lambda: {"dual robots": healthy}

    result = asyncio.run(
        bridge.digital_twin_execute_pick_approach(
            "dual robots",
            "ur5e",
            "prusa-mk4-2",
            "MG",
            confirmed=True,
        )
    )

    assert result["success"] is False
    assert message in result["message"]
    assert agent.calls == []


def test_ui_cancellation_keeps_ur5e_lock_until_agent_runtime_finishes() -> None:
    async def _exercise() -> None:
        agent = _PhysicalUR5eAgent()
        bridge = _ready_bridge(agent)
        started = asyncio.Event()
        finish = asyncio.Event()

        async def _slow_pick_approach(**kwargs: Any) -> dict[str, Any]:
            agent.calls.append(("pick_approach", kwargs))
            started.set()
            await finish.wait()
            agent._current_state = "at_pick"
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

        assert bridge._ur5e_pick_execution_lock.acquire(blocking=False) is False
        assert bridge._ur5e_pick_execution_active == "pick_approach"

        finish.set()
        for _attempt in range(20):
            await asyncio.sleep(0)
            if bridge._ur5e_pick_execution_lock.acquire(blocking=False):
                bridge._ur5e_pick_execution_lock.release()
                break
        else:
            pytest.fail("UR5e lock was not released after the agent runtime completed")
        assert bridge._ur5e_pick_execution_active is None

    asyncio.run(_exercise())
