from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("OPENAI_API_KEY", "sk-local-test")

REPO_ROOT = Path(__file__).resolve().parents[1]
PKG_ROOT = REPO_ROOT / "cais_spade_llm"
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from cais_spade_llm.agents.intelligent_product.product_agent import ProductAgent
from cais_spade_llm.agents.intelligent_product.replanner.resource_bidding import _tool_signature
from cais_spade_llm.resources.sensor.camera_module import CameraModule
from cais_spade_llm.ui.bridge import SystemBridge


def _make_product_agent(
    tmp_path: Path,
    *,
    resource_agents: list | None = None,
    resource_jids: list[str] | None = None,
) -> tuple[ProductAgent, list]:
    resources = resource_agents or [SimpleNamespace(jid="ur5e@localhost", static_capabilities={})]
    agent = ProductAgent(
        "assembly_board-v1@localhost",
        "none",
        name="assembly_board-v1",
        resource_jids=resource_jids or [str(getattr(resource, "jid", "")) for resource in resources],
        resource_agents=resources,
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

    async def _fake_send_agent_message(msg, trace_category="agent"):
        sent_messages.append(msg)

    agent.send = _fake_send  # type: ignore[assignment]
    agent._send_agent_message = _fake_send_agent_message  # type: ignore[assignment]
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


def _minimal_recovery_tools() -> list[dict]:
    return [
        {
            "function": "move_home",
            "function_owner_agent": "ur5e",
            "process": "assembly",
            "resource_type": "robot",
            "in_state": "any",
            "out_state": "idle",
            "context": [],
            "params": {},
            "description": "Robot arm move to its home position.",
        },
        {
            "function": "place_approach",
            "function_owner_agent": "ur5e",
            "process": "assembly",
            "resource_type": "robot",
            "in_state": "picked",
            "out_state": "positioned",
            "part_in_state": "in_gripper",
            "context_mapping": {
                "location_param": "destination_location",
                "location_type": "reachable_location",
            },
            "part_transition": {
                "completed": {
                    "state": "in_transit",
                    "location_template": "{resource_jid}_gripper",
                }
            },
            "params": {},
            "description": "Move the loaded part to its destination location.",
        },
        {
            "function": "place_insert",
            "function_owner_agent": "ur5e",
            "process": "assembly",
            "resource_type": "robot",
            "in_state": "positioned",
            "out_state": "placed",
            "part_in_state": "in_transit",
            "context_mapping": {
                "location_param": "destination_location",
                "location_type": "current_location",
            },
            "part_transition": {
                "completed": {
                    "state": "assembled",
                    "location_param": "destination_location",
                }
            },
            "params": {},
            "description": "Assemble the currently held part at its final destination.",
        },
    ]


def _disconnected_recovery_tools() -> list[dict]:
    tools = _minimal_recovery_tools()
    tools.append(
        {
            "function": "clear_fault",
            "function_owner_agent": "ur5e",
            "process": "assembly",
            "resource_type": "robot",
            "in_state": "faulted",
            "out_state": "idle",
            "context": [],
            "params": {},
            "description": "Recover a resource from an abstract faulted state.",
        }
    )
    return tools


def _write_tools_catalog(tmp_path: Path, tools_catalog: list[dict]) -> Path:
    tools_path = tmp_path / "tools.json"
    tools_path.write_text(json.dumps(tools_catalog, indent=2), encoding="utf-8")
    return tools_path


def _runtime_mutex_violation(candidate_tools: list[dict]) -> list[dict]:
    return [
        {
            "type": "safety_block",
            "failed_task_id": "REQ_2_T3",
            "affected_task_ids": ["REQ_2_T3", "REQ_2_T4"],
            "safety_ctx": {
                "violated_rule": "SAFE_1",
                "violated_rule_id": "SAFE_1",
                "obligation_targets": [
                    {
                        "rule_id": "SAFE_1",
                        "resource_jid": "ur5e@localhost",
                        "candidate_tools": candidate_tools,
                        "required_event_aps": [],
                        "required_state_aps": [],
                        "current_resource_state": "picked",
                    }
                ],
            },
        }
    ]


def _configure_des_runtime_agent(
    tmp_path: Path,
    *,
    nodes: list[dict],
    part_tracker: dict[str, dict[str, object]],
    tools_catalog: list[dict] | None = None,
) -> tuple[ProductAgent, list, list[dict]]:
    tools_catalog = list(tools_catalog or _minimal_recovery_tools())
    tools_path = _write_tools_catalog(tmp_path, tools_catalog)
    ProductAgent.configure_shared_tools_catalogue(tools_path)

    resource = SimpleNamespace(
        jid="ur5e@localhost",
        static_capabilities={"reachability": ["Assembly Station"], "staging_areas": {}},
    )
    agent, sent_messages = _make_product_agent(
        tmp_path,
        resource_agents=[resource],
        resource_jids=["ur5e@localhost"],
    )
    agent.process_planner.nodes = deepcopy(nodes)
    agent.part_tracker = deepcopy(part_tracker)
    return agent, sent_messages, tools_catalog


def _candidate_tool(tool: dict, *, resource_jid: str) -> dict:
    return {
        "function_name": tool["function"],
        "resource_jid": resource_jid,
        "tool_signature": _tool_signature(tool),
        "in_state": tool["in_state"],
        "out_state": tool["out_state"],
        "description": tool["description"],
        "matched_event_aps": [],
        "matched_state_aps": [],
    }


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

    async def _fake_online_replan(violations, system_coordination_state=None, bridge_feedback=""):
        planner_calls.append((list(violations), dict(system_coordination_state or {})))
        assert bridge_feedback == ""
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


def test_product_requeues_blocked_task_after_retry_ready(tmp_path):
    agent, _sent_messages = _make_product_agent(tmp_path)
    agent.process_planner.nodes = [
        {
            "id": "REQ_2_T3",
            "type": "task",
            "status": "blocked",
            "function_name": "place_approach",
            "resource_jid": "xarm6@localhost",
            "params": {"part_name": "LRP"},
        }
    ]
    agent.task_states["REQ_2_T3"] = "blocked"

    reactivated = agent._handle_task_retry_ready(["REQ_2_T3"])

    assert reactivated == 1
    assert agent.process_planner.nodes[0]["status"] == "pending"
    assert agent.task_states["REQ_2_T3"] == "pending"
    assert agent.execution_timeline[-1]["task_id"] == "REQ_2_T3"
    assert agent.execution_timeline[-1]["status"] == "requeued"


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

    async def _fake_online_replan(_violations, system_coordination_state=None, bridge_feedback=""):
        planner_call_count["count"] += 1
        assert isinstance(system_coordination_state, dict)
        assert isinstance(bridge_feedback, str)
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

    async def _fake_online_replan(_violations, system_coordination_state=None, bridge_feedback=""):
        assert isinstance(system_coordination_state, dict)
        assert bridge_feedback == ""
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


def test_project_resource_suffix_state_projects_full_snapshot(tmp_path):
    nodes = [
        {
            "id": "REQ_1_T3",
            "type": "task",
            "function_name": "place_approach",
            "params": {"part_name": "MCP", "destination_location": "Assembly Station"},
            "resource_jid": "ur5e@localhost",
            "sequence_index": 2,
            "status": "running",
            "predecessors": [],
            "successors": ["REQ_1_T4"],
        },
        {
            "id": "REQ_1_T4",
            "type": "task",
            "function_name": "place_insert",
            "params": {"part_name": "MCP", "destination_location": "Assembly Station"},
            "resource_jid": "ur5e@localhost",
            "sequence_index": 3,
            "status": "pending",
            "predecessors": ["REQ_1_T3"],
            "successors": [],
        },
    ]
    part_tracker = {"MCP": {"state": "ready", "location": "ur5e@localhost_gripper"}}
    agent, _sent_messages, tools_catalog = _configure_des_runtime_agent(
        tmp_path,
        nodes=nodes,
        part_tracker=part_tracker,
    )

    try:
        projected, anchor_task_id, failure_reason = agent.process_planner._project_resource_suffix_state(
            resource_jid="ur5e@localhost",
            tools_catalog=tools_catalog,
            resource_states={"ur5e@localhost": {"current_state": "picked", "held_part": "MCP"}},
            default_resource_state="idle",
            part_states={"MCP": "ready"},
            part_locations={"MCP": "ur5e@localhost_gripper"},
            goal_state="assembled",
        )
        assert failure_reason == ""
        assert anchor_task_id == "REQ_1_T4"
        assert projected is not None
        assert projected["resource_state"] == "placed"
        assert projected["current_part"] is None
        assert projected["current_location"] is None
        assert projected["part_states"]["MCP"] == "assembled"
        assert projected["part_locations"]["MCP"] == "Assembly Station"
    finally:
        ProductAgent.configure_shared_tools_catalogue()


def test_runtime_des_recovery_uses_projected_suffix_and_skips_bridge(tmp_path):
    nodes = [
        {
            "id": "REQ_2_T2",
            "type": "task",
            "function_name": "pick_grasp",
            "params": {"part_name": "LRP", "origin_resource_location": "prusa-mk4-1"},
            "resource_jid": "xarm6@localhost",
            "sequence_index": 1,
            "status": "completed",
            "predecessors": [],
            "successors": ["REQ_2_T3"],
        },
        {
            "id": "REQ_1_T3",
            "type": "task",
            "function_name": "place_approach",
            "params": {"part_name": "MCP", "destination_location": "Assembly Station"},
            "resource_jid": "ur5e@localhost",
            "sequence_index": 2,
            "status": "running",
            "predecessors": [],
            "successors": ["REQ_1_T4"],
        },
        {
            "id": "REQ_1_T4",
            "type": "task",
            "function_name": "place_insert",
            "params": {"part_name": "MCP", "destination_location": "Assembly Station"},
            "resource_jid": "ur5e@localhost",
            "sequence_index": 3,
            "status": "pending",
            "predecessors": ["REQ_1_T3"],
            "successors": [],
        },
        {
            "id": "REQ_2_T3",
            "type": "task",
            "function_name": "place_approach",
            "params": {"part_name": "LRP", "destination_location": "Assembly Station"},
            "resource_jid": "xarm6@localhost",
            "sequence_index": 4,
            "status": "blocked",
            "predecessors": ["REQ_2_T2"],
            "successors": [],
        },
    ]
    part_tracker = {"MCP": {"state": "ready", "location": "ur5e@localhost_gripper"}}
    agent, sent_messages, tools_catalog = _configure_des_runtime_agent(
        tmp_path,
        nodes=nodes,
        part_tracker=part_tracker,
    )
    move_home_tool = next(tool for tool in tools_catalog if tool["function"] == "move_home")
    candidate_tools = [_candidate_tool(move_home_tool, resource_jid="ur5e@localhost")]
    agent.ask_llm = lambda **_kwargs: (_ for _ in ()).throw(AssertionError("bridge should not be used"))  # type: ignore[assignment]

    async def _run() -> dict:
        return await agent._handle_runtime_des_replan_request(
            reason="safety_block",
            failed_task_id="REQ_2_T3",
            violations=_runtime_mutex_violation(candidate_tools),
            system_coordination_state={
                "resource_states": {
                    "ur5e@localhost": {"current_state": "picked", "held_part": "MCP"},
                    "xarm6@localhost": {"current_state": "picked", "held_part": "LRP"},
                }
            },
        )

    try:
        recovery = asyncio.run(_run())
        assert recovery["status"] == "validating"
        assert recovery["used_llm_bridge"] is False
        assert recovery["bridge_proposal"] is None
        inserted = [
            node for node in agent.process_planner.nodes
            if str(node.get("change_reason", "")).startswith("INSERTION: DES recovery")
        ]
        assert [node.get("function_name") for node in inserted] == ["move_home"]
        move_home = inserted[0]
        assert move_home.get("resource_jid") == "ur5e@localhost"
        assert move_home.get("predecessors") == ["REQ_1_T4"]
        blocked_task = next(node for node in agent.process_planner.nodes if node.get("id") == "REQ_2_T3")
        assert blocked_task.get("status") == "pending"
        assert "REQ_2_T2" in blocked_task.get("predecessors", [])
        assert move_home.get("id") in blocked_task.get("predecessors", [])
        assert len(sent_messages) == 1
        assert sent_messages[0].metadata.get("type") == "plan_safety_check"
    finally:
        ProductAgent.configure_shared_tools_catalogue()


def test_runtime_des_recovery_falls_back_to_bridge_when_no_modeled_path_exists(tmp_path):
    nodes = [
        {
            "id": "REQ_1_T3",
            "type": "task",
            "function_name": "place_approach",
            "params": {"part_name": "MCP", "destination_location": "Assembly Station"},
            "resource_jid": "ur5e@localhost",
            "sequence_index": 2,
            "status": "running",
            "predecessors": [],
            "successors": [],
        },
    ]
    part_tracker = {"MCP": {"state": "in_gripper", "location": "ur5e@localhost_gripper"}}
    agent, sent_messages, tools_catalog = _configure_des_runtime_agent(
        tmp_path,
        nodes=nodes,
        part_tracker=part_tracker,
        tools_catalog=_disconnected_recovery_tools(),
    )
    clear_fault_tool = next(tool for tool in tools_catalog if tool["function"] == "clear_fault")
    candidate_tools = [_candidate_tool(clear_fault_tool, resource_jid="ur5e@localhost")]

    async def _bridge_prompt(**_kwargs):
        return json.dumps(
            {
                "function_name": "bridge_recovery_macro",
                "resource_jid": "ur5e@localhost",
                "description": "Operator-approved recovery macro",
                "rationale": "No modeled recovery path was available.",
                "macro_steps": [{"function_name": "move_home", "params": {}}],
            }
        )

    agent.ask_llm = _bridge_prompt  # type: ignore[assignment]

    async def _run() -> dict:
        return await agent._handle_runtime_des_replan_request(
            reason="safety_block",
            failed_task_id="REQ_2_T3",
            violations=_runtime_mutex_violation(candidate_tools),
            system_coordination_state={
                "resource_states": {
                    "ur5e@localhost": {"current_state": "picked", "held_part": "MCP"},
                    "xarm6@localhost": {"current_state": "picked", "held_part": "LRP"},
                }
            },
        )

    try:
        recovery = asyncio.run(_run())
        assert recovery["status"] == "llm_bridge"
        assert recovery["used_llm_bridge"] is True
        assert recovery["bridge_approval_state"] == "pending"
        assert recovery["bridge_proposal"]["macro_steps"][0]["function_name"] == "move_home"
        assert sent_messages == []
    finally:
        ProductAgent.configure_shared_tools_catalogue()


def test_runtime_des_recovery_skips_live_duplicate_suffix_fallback(tmp_path):
    nodes = [
        {
            "id": "REQ_1_T3",
            "type": "task",
            "function_name": "place_approach",
            "params": {"part_name": "MCP", "destination_location": "Assembly Station"},
            "resource_jid": "ur5e@localhost",
            "sequence_index": 2,
            "status": "running",
            "predecessors": [],
            "successors": ["REQ_1_T4"],
        },
        {
            "id": "REQ_1_T4",
            "type": "task",
            "function_name": "place_insert",
            "params": {"part_name": "MCP", "destination_location": "Assembly Station"},
            "resource_jid": "ur5e@localhost",
            "sequence_index": 3,
            "status": "pending",
            "predecessors": ["REQ_1_T3"],
            "successors": [],
        },
    ]
    part_tracker = {"MCP": {"state": "ready", "location": "ur5e@localhost_gripper"}}
    agent, sent_messages, tools_catalog = _configure_des_runtime_agent(
        tmp_path,
        nodes=nodes,
        part_tracker=part_tracker,
    )
    move_home_tool = next(tool for tool in tools_catalog if tool["function"] == "move_home")
    candidate_tools = [_candidate_tool(move_home_tool, resource_jid="ur5e@localhost")]

    async def _bridge_prompt(**_kwargs):
        return json.dumps(
            {
                "function_name": "bridge_recovery_macro",
                "resource_jid": "ur5e@localhost",
                "description": "Operator-approved recovery macro",
                "rationale": "Projection stayed inconsistent, so no catalog-valid residual path was available.",
                "macro_steps": [{"function_name": "move_home", "params": {}}],
            }
        )

    agent.ask_llm = _bridge_prompt  # type: ignore[assignment]

    async def _run() -> dict:
        return await agent._handle_runtime_des_replan_request(
            reason="safety_block",
            failed_task_id="REQ_2_T3",
            violations=_runtime_mutex_violation(candidate_tools),
            system_coordination_state={
                "resource_states": {
                    "ur5e@localhost": {"current_state": "idle", "held_part": None},
                    "xarm6@localhost": {"current_state": "picked", "held_part": "LRP"},
                }
            },
        )

    try:
        recovery = asyncio.run(_run())
        assert recovery["status"] == "llm_bridge"
        assert recovery["used_llm_bridge"] is True
        inserted = [
            node for node in agent.process_planner.nodes
            if str(node.get("change_reason", "")).startswith("INSERTION: DES recovery")
        ]
        assert inserted == []
        assert sent_messages == []
    finally:
        ProductAgent.configure_shared_tools_catalogue()


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
