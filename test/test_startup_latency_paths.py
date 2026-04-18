import asyncio
import os
import threading
import time

os.environ.setdefault("OPENAI_API_KEY", "test")

from cais_spade_llm.agents.intelligent_product.product_agent import ProductAgent
from cais_spade_llm.ui.bridge import SystemBridge


def _bare_product_agent() -> ProductAgent:
    agent = ProductAgent.__new__(ProductAgent)
    agent.agent_name = "widget"
    agent.jid = "widget@localhost"
    agent.kickoff_result = {
        "success": False,
        "message": "kickoff pending",
        "retries_used": 0,
        "retries_max": 0,
        "violated_rules": [],
        "witness_count": 0,
        "updated_at_utc": "",
        "product_name": "widget",
        "product_jid": "widget@localhost",
        "stage": "kickoff",
        "alert": None,
    }
    agent.startup_readiness = {
        "startup_ready": False,
        "success": False,
        "continuing": True,
        "execution_blocked": True,
        "message": "kickoff pending",
        "retries_used": 0,
        "retries_max": 0,
        "violated_rules": [],
        "witness_count": 0,
        "updated_at_utc": "",
        "product_name": "widget",
        "product_jid": "widget@localhost",
        "stage": "kickoff",
        "alert": None,
        "used_precomputed_bundle": False,
    }
    agent._kickoff_result_event = asyncio.Event()
    agent._startup_readiness_event = asyncio.Event()
    return agent


def test_wait_for_startup_readiness_timeout_returns_not_ready() -> None:
    agent = _bare_product_agent()

    result = asyncio.run(agent.wait_for_startup_readiness(timeout=0.01))

    assert result["startup_ready"] is False
    assert result["success"] is False
    assert result["execution_blocked"] is True
    assert "timed out" in result["message"].lower()


def test_startup_readiness_unblocks_before_final_kickoff_result() -> None:
    agent = _bare_product_agent()
    agent._set_startup_readiness(
        startup_ready=True,
        success=False,
        continuing=True,
        message="widget: auto-replan running in background.",
        retries_used=1,
        retries_max=3,
        violations=[{"violated_rule_id": "SAFE_1"}],
    )

    result = asyncio.run(agent.wait_for_startup_readiness(timeout=0.01))

    assert result["startup_ready"] is True
    assert result["success"] is False
    assert result["continuing"] is True
    assert result["execution_blocked"] is True
    assert result["violated_rules"] == ["SAFE_1"]
    assert agent._kickoff_result_event.is_set() is False


def test_final_kickoff_result_does_not_imply_startup_ready_by_itself() -> None:
    agent = _bare_product_agent()
    agent._set_kickoff_result(
        success=False,
        message="widget: kickoff failed before first validation.",
        retries_used=0,
        retries_max=3,
        violations=[],
    )

    assert agent._kickoff_result_event.is_set() is True
    assert agent._startup_readiness_event.is_set() is False


def test_simulation_start_ready_allows_force_while_prewarm_is_inflight() -> None:
    bridge = SystemBridge.__new__(SystemBridge)
    bridge._gazebo_prewarm_lock = threading.Lock()
    bridge._gazebo_prewarm_done = threading.Event()
    bridge._gazebo_prewarm_thread = None
    bridge._gazebo_prewarm_pending = {"xarm6"}
    bridge._sim_ready_probe_inflight = False
    bridge._sim_ready_cache = (False, "pending")
    bridge._sim_ready_cache_ts = 0.0
    bridge._any_running = lambda names: True
    bridge._probe_sim_services = lambda timeout_sec=3.0: (True, "")

    ready, message = bridge.simulation_start_ready(force=True)

    assert ready is True
    assert "background" in message.lower()
    assert bridge._sim_ready_cache[0] is True
    assert bridge._sim_ready_cache_ts <= time.monotonic()
