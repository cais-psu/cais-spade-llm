from __future__ import annotations

import asyncio
import os
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("OPENAI_API_KEY", "sk-local-test")

REPO_ROOT = Path(__file__).resolve().parents[1]
PKG_ROOT = REPO_ROOT / "cais_spade_llm"
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from cais_spade_llm.agents.intelligent_product.product_agent import ProductAgent
from cais_spade_llm.resources.sensor.camera_module import CameraModule
from cais_spade_llm.ui.bridge import SystemBridge


def _make_product_agent(tmp_path: Path) -> tuple[ProductAgent, list]:
    resource = SimpleNamespace(jid="ur5e@localhost", static_capabilities={})
    agent = ProductAgent(
        "assembly_board-v1@localhost",
        "none",
        name="assembly_board-v1",
        resource_jids=["ur5e@localhost"],
        resource_agents=[resource],
        product_specification_file="cais_spade_llm/specification/products/requirements/assembly_board-v1.txt",
        safety_file="cais_spade_llm/specification/safety/safety_requirements.txt",
        camera=CameraModule(backend="none"),
        replan_mode="des",
    )
    agent.plan_path = tmp_path / "plan.json"
    agent.global_fsa_path = tmp_path / "global_fsa.json"
    agent.product_state_path = tmp_path / "product_state.json"
    agent.resource_state_path = tmp_path / "resource_state.json"

    sent_messages: list = []

    async def _fake_send(msg):
        sent_messages.append(msg)

    agent.send = _fake_send  # type: ignore[assignment]
    agent._ensure_plan_result_inbox = lambda: None  # type: ignore[assignment]
    agent._persist_plan_snapshot = lambda: None  # type: ignore[assignment]
    agent._persist_product_state = lambda: None  # type: ignore[assignment]
    agent._persist_resource_state = lambda: None  # type: ignore[assignment]
    agent.process_planner.compile_global_fsa = lambda: setattr(  # type: ignore[assignment]
        agent.process_planner,
        "global_fsa",
        {
            "A": {"X": ["S0"], "E": [], "Tr": [], "x0": "S0", "Xm": ["S0"]},
            "meta": {},
        },
    )
    agent.process_planner.save_global_fsa = lambda path: None  # type: ignore[assignment]
    agent.process_planner.nodes = []
    agent.process_planner.global_fsa = {
        "A": {"X": ["S0"], "E": [], "Tr": [], "x0": "S0", "Xm": ["S0"]},
        "meta": {},
    }
    return agent, sent_messages


def _runtime_violation(task_id: str = "REQ_1_T4") -> list[dict]:
    return [
        {
            "type": "inevitable_violation",
            "failed_task_id": task_id,
            "affected_task_ids": [task_id],
            "violated_rule_id": "SAFE_1",
        }
    ]


def test_runtime_des_recovery_starts_session_and_validates(tmp_path):
    agent, sent_messages = _make_product_agent(tmp_path)
    planner_calls: list[tuple[list[dict], dict]] = []

    async def _fake_online_replan(violations, system_coordination_state=None):
        planner_calls.append((list(violations), dict(system_coordination_state or {})))
        return {
            "plan_changed": True,
            "used_llm_bridge": False,
            "human_required": False,
            "message": "DES recovery candidate generated.",
        }

    async def _unexpected_offline(*_args, **_kwargs):
        raise AssertionError("offline replanner should not be used in runtime DES mode")

    agent.process_planner.replan_with_feedback_online = _fake_online_replan  # type: ignore[assignment]
    agent.process_planner.replan_with_feedback_offline = _unexpected_offline  # type: ignore[assignment]

    async def _run() -> None:
        recovery = await agent._handle_runtime_des_replan_request(
            reason="inevitable_violation",
            failed_task_id="REQ_1_T4",
            violations=_runtime_violation(),
            system_coordination_state={"resource_states": {"ur5e@localhost": {"current_state": "idle"}}},
        )
        assert recovery["status"] == "validating"
        assert recovery["attempts_used"] == 1
        assert recovery["resolution_class"] == "none"
        assert agent.runtime_repair_state == "repairing"

    asyncio.run(_run())

    assert len(planner_calls) == 1
    assert planner_calls[0][1]["resource_states"]["ur5e@localhost"]["current_state"] == "idle"
    assert len(sent_messages) == 1
    assert sent_messages[0].metadata.get("type") == "plan_safety_check"


def test_runtime_des_validation_failure_retries_without_offline_replan(tmp_path):
    agent, sent_messages = _make_product_agent(tmp_path)
    planner_results = [
        {
            "plan_changed": True,
            "used_llm_bridge": False,
            "human_required": False,
            "message": "first DES candidate",
        },
        {
            "plan_changed": True,
            "used_llm_bridge": True,
            "human_required": False,
            "message": "second DES candidate",
            "bridge_summary": ["move_home"],
        },
    ]
    planner_call_count = {"count": 0}

    async def _fake_online_replan(_violations, system_coordination_state=None):
        planner_call_count["count"] += 1
        assert isinstance(system_coordination_state, dict)
        return planner_results.pop(0)

    async def _unexpected_offline(*_args, **_kwargs):
        raise AssertionError("offline replanner should not be used in runtime DES mode")

    agent.process_planner.replan_with_feedback_online = _fake_online_replan  # type: ignore[assignment]
    agent.process_planner.replan_with_feedback_offline = _unexpected_offline  # type: ignore[assignment]

    async def _run() -> None:
        await agent._handle_runtime_des_replan_request(
            reason="inevitable_violation",
            failed_task_id="REQ_1_T4",
            violations=_runtime_violation(),
            system_coordination_state={"resource_states": {}},
        )
        handled = await agent._handle_runtime_plan_validation_result(
            ok=False,
            violations=_runtime_violation("REQ_1_T4"),
        )
        assert handled is True
        recovery = agent.get_runtime_recovery()
        assert recovery["status"] == "validating"
        assert recovery["attempts_used"] == 2
        assert recovery["used_llm_bridge"] is True

    asyncio.run(_run())

    assert planner_call_count["count"] == 2
    assert len(sent_messages) == 2


def test_runtime_des_validation_success_resolves_session(tmp_path):
    agent, _sent_messages = _make_product_agent(tmp_path)

    async def _fake_online_replan(_violations, system_coordination_state=None):
        assert isinstance(system_coordination_state, dict)
        return {
            "plan_changed": True,
            "used_llm_bridge": False,
            "human_required": False,
            "message": "DES recovery candidate generated.",
        }

    agent.process_planner.replan_with_feedback_online = _fake_online_replan  # type: ignore[assignment]

    async def _run() -> None:
        await agent._handle_runtime_des_replan_request(
            reason="inevitable_violation",
            failed_task_id="REQ_1_T4",
            violations=_runtime_violation(),
            system_coordination_state={"resource_states": {}},
        )
        handled = await agent._handle_runtime_plan_validation_result(ok=True, violations=[])
        assert handled is True
        recovery = agent.get_runtime_recovery()
        assert recovery["status"] == "resolved"
        assert recovery["resolution_class"] == "des_only"
        assert recovery["attempts_used"] == 1
        assert agent.runtime_repair_state == "idle"

    asyncio.run(_run())


class _BridgeProductAgent:
    def __init__(self) -> None:
        self.jid = "assembly_board-v1@localhost"
        self.runtime_recovery = {
            "product_name": "assembly_board-v1",
            "product_jid": self.jid,
            "replan_mode": "des",
            "status": "human_required",
            "resolution_class": "human_required",
            "message": "Need operator help.",
            "history": [],
        }
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    @property
    def loop(self):
        return self._loop

    def get_runtime_recovery(self) -> dict:
        return dict(self.runtime_recovery)

    async def submit_runtime_recovery_guidance(self, message: str) -> dict:
        self.runtime_recovery["operator_guidance"] = str(message)
        self.runtime_recovery["history"] = list(self.runtime_recovery.get("history") or []) + [
            {"status": "human_required", "message": f"Operator guidance recorded: {message}"}
        ]
        return self.get_runtime_recovery()

    async def retry_runtime_recovery_des(self) -> dict:
        self.runtime_recovery["status"] = "des_search"
        self.runtime_recovery["resolution_class"] = "none"
        self.runtime_recovery["message"] = "Retrying DES recovery."
        return self.get_runtime_recovery()

    def close(self) -> None:
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=2.0)
        self._loop.close()


def test_bridge_runtime_recovery_actions_use_product_agent_loop():
    bridge = SystemBridge()
    fake_product = _BridgeProductAgent()
    bridge.system_running = True
    bridge.product_agents = [fake_product]

    try:
        recoveries = bridge.get_runtime_recoveries()
        assert len(recoveries) == 1
        assert recoveries[0]["status"] == "human_required"

        guidance = bridge.submit_runtime_recovery_guidance(fake_product.jid, "operator note")
        assert guidance["operator_guidance"] == "operator note"

        retried = bridge.retry_runtime_recovery_des(fake_product.jid)
        assert retried["status"] == "des_search"
        assert retried["resolution_class"] == "none"
    finally:
        fake_product.close()

