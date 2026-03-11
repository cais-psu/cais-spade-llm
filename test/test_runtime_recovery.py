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
from cais_spade_llm.agents.intelligent_product.replanner.primitive_semantics import (
    get_robot_bridge_snapshot,
)
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


class _DummyBridgeRobot:
    _BRIDGE_PRIMITIVES = frozenset(
        {
            "move_cartesian",
            "move_relative",
            "move_to_named_pose",
            "open_gripper",
            "close_gripper",
            "detect_parts",
            "attach_part",
            "detach_part",
            "get_current_pose",
        }
    )

    def __init__(self, jid: str, scope_name: str, *, current_state: str, held_part: str | None) -> None:
        self.jid = jid
        self.agent_name = scope_name
        self.static_capabilities = {"reachability": ["Assembly Station"], "staging_areas": {}}
        self._controller = None
        self.execution_mode = "dry_run"
        self._current_state = current_state
        self._held_part = held_part
        self._gripper_state = "closed" if held_part else "open"
        self._position = {"x": 0.1, "y": 0.2, "z": 0.3}
        self._bridge_pose_ref = None
        self.named_positions = {"home": [0, 1, 2, 3, 4, 5]}
        self._scope_name = scope_name

    def _robot_scope_name(self) -> str:
        return self._scope_name

    def get_bridge_snapshot(self) -> dict[str, object]:
        return get_robot_bridge_snapshot(self)


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


def _configure_bridge_runtime_agent(
    tmp_path: Path,
    *,
    nodes: list[dict],
    part_tracker: dict[str, dict[str, object]],
    robot: _DummyBridgeRobot,
    tools_catalog: list[dict] | None = None,
) -> tuple[ProductAgent, list, list[dict]]:
    tools_catalog = list(tools_catalog or _minimal_recovery_tools())
    tools_path = _write_tools_catalog(tmp_path, tools_catalog)
    ProductAgent.configure_shared_tools_catalogue(tools_path)

    agent, sent_messages = _make_product_agent(
        tmp_path,
        resource_agents=[robot],
        resource_jids=[str(robot.jid)],
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


async def _deliver_ack(
    agent: ProductAgent,
    *,
    task_id: str,
    status: str = "completed",
    sender: str = "ur5e@localhost",
    content: str = "",
    observations: dict[str, Any] | None = None,
) -> None:
    inbox = agent._AckInbox()
    inbox.agent = agent  # type: ignore[attr-defined]

    async def _fake_receive(timeout: float = 0.5):
        assert timeout == 0.5
        payload: dict[str, Any] = {"task_id": task_id, "status": status}
        if content:
            payload["content"] = content
        if isinstance(observations, dict):
            payload["observations"] = observations
        return SimpleNamespace(
            body=json.dumps(payload),
            sender=sender,
        )

    inbox.receive = _fake_receive  # type: ignore[assignment]
    await inbox.run()


def _activate_bridge_sequence(
    agent: ProductAgent,
    *,
    bridge_sequence_id: str,
    bridge_task_ids: list[str],
    failed_task_id: str,
    violations: list[dict[str, Any]],
    system_coordination_state: dict[str, Any],
    trigger: str = "safety_block",
) -> dict[str, Any]:
    return agent._set_runtime_recovery(
        reset=True,
        status="resolved",
        resolution_class="des_with_llm_bridge",
        trigger=trigger,
        failed_task_id=failed_task_id,
        message="Plan validation passed; approved bridge sequence is executing.",
        attempts_used=1,
        attempts_max=agent._runtime_repair_max_attempts,
        used_llm_bridge=True,
        bridge_proposal=None,
        bridge_approval_state="approved",
        active_bridge_sequence={
            "bridge_sequence_id": bridge_sequence_id,
            "bridge_task_ids": list(bridge_task_ids),
            "bridge_sequence_length": len(bridge_task_ids),
            "trigger": trigger,
            "failed_task_id": failed_task_id,
            "violations": deepcopy(violations),
            "system_coordination_state": deepcopy(system_coordination_state),
            "state": "executing",
        },
        bridge_feedback_history=[],
        violations=[],
    )


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


def test_bridge_macro_compilation_threads_part_name_and_task_params(tmp_path):
    agent, _sent_messages = _make_product_agent(tmp_path)

    inserted = agent.process_planner._apply_primitive_bridge_proposal(
        {
            "macro_name": "recover_sg",
            "resource_jid": "ur5e@localhost",
            "expected_start_state": "recovery_required",
            "expected_snapshot": {"current_state": "recovery_required"},
            "part_name": "SG",
            "task_params": {"destination_location": "assembly_board-v1"},
            "task_metadata": {
                "in_state": "recovery_required",
                "out_state": "idle",
                "required_context_keys": ["destination_location"],
                "context_mapping": {"location_param": "destination_location"},
                "part_transition": {
                    "completed": {
                        "state": "assembled",
                        "location_param": "destination_location",
                    }
                },
            },
            "primitive_steps": [
                {"primitive": "open_gripper", "params": {}},
            ],
        }
    )

    assert len(inserted) == 1
    node = inserted[0]
    assert node["function_name"] == "execute_recovery_macro"
    assert node["params"]["part_name"] == "SG"
    assert node["params"]["destination_location"] == "assembly_board-v1"
    assert node["part_name"] == "SG"
    assert node["part_transition"]["completed"]["location_param"] == "destination_location"


def test_bridge_request_uses_whole_system_bridge_context(tmp_path):
    resources = [
        _DummyBridgeRobot(
            "ur5e@localhost",
            "ur5e",
            current_state="picked",
            held_part="MCP",
        ),
        _DummyBridgeRobot(
            "xarm6@localhost",
            "xarm6",
            current_state="idle",
            held_part=None,
        ),
    ]
    agent, _sent_messages = _make_product_agent(
        tmp_path,
        resource_agents=resources,
        resource_jids=["ur5e@localhost", "xarm6@localhost"],
    )
    captured_prompt: dict[str, str] = {}

    async def _bridge_prompt(**kwargs):
        captured_prompt["value"] = str(kwargs["prompt"])
        return json.dumps(
            {
                "macro_tasks": [
                    {
                        "resource_jid": "ur5e@localhost",
                        "macro_name": "stabilize_pick",
                        "expected_start_state": "picked",
                        "task_metadata": {
                            "in_state": "picked",
                            "out_state": "picked",
                            "required_context_keys": [],
                            "context_mapping": {},
                            "part_transition": None,
                        },
                        "primitive_steps": [
                            {"primitive": "open_gripper", "params": {}},
                        ],
                    }
                ]
            }
        )

    agent.ask_llm = _bridge_prompt  # type: ignore[assignment]

    proposal = asyncio.run(
        agent.process_planner._request_bridge_proposal(
            stuck_state={
                "resource_state": "picked",
                "current_part": "MCP",
                "current_location": None,
                "part_states": {"MCP": "in_gripper"},
                "part_locations": {"MCP": "ur5e@localhost_gripper"},
            },
            P_id=["MCP"],
            ra_jid="ur5e@localhost",
            goal_state="assembled",
            tools_catalog=_minimal_recovery_tools(),
            part_tracker={"MCP": {"state": "in_gripper", "location": "ur5e@localhost_gripper"}},
            obligation_targets=[],
            bridge_feedback="",
            resource_states={
                "ur5e@localhost": {"current_state": "picked", "held_part": "MCP"},
                "xarm6@localhost": {"current_state": "idle", "held_part": None},
            },
            default_resource_state="idle",
            part_states={"MCP": "in_gripper"},
            part_locations={"MCP": "ur5e@localhost_gripper"},
        )
    )

    assert proposal is not None
    assert "WHOLE-SYSTEM BRIDGE RESOURCES" in captured_prompt["value"]
    assert '"ur5e@localhost"' in captured_prompt["value"]
    assert '"xarm6@localhost"' in captured_prompt["value"]
    assert "pending_tasks" in captured_prompt["value"]


def test_approve_runtime_bridge_proposal_compiles_ordered_macro_tasks_and_gates_blocked_tasks(tmp_path):
    agent, _sent_messages = _make_product_agent(tmp_path)
    agent.process_planner.nodes = [
        {
            "id": "REQ_2_T3",
            "type": "task",
            "status": "blocked",
            "function_name": "place_approach",
            "params": {"part_name": "LRP", "destination_location": "Assembly Station"},
            "resource_jid": "xarm6@localhost",
            "sequence_index": 3,
            "predecessors": ["REQ_2_T2"],
            "successors": [],
        },
        {
            "id": "REQ_9_T1",
            "type": "task",
            "status": "pending",
            "function_name": "move_home",
            "params": {},
            "resource_jid": "ur5e@localhost",
            "sequence_index": 9,
            "predecessors": [],
            "successors": [],
        },
    ]

    proposal = {
        "primary_obligation": {
            "rule_id": "SAFE_1",
            "resource_jid": "xarm6@localhost",
        },
        "macro_tasks": [
            {
                "resource_jid": "ur5e@localhost",
                "macro_name": "stash_mcp",
                "expected_start_state": "picked",
                "expected_snapshot": {"current_state": "picked"},
                "part_name": "MCP",
                "task_params": {"destination_location": "buffer_a"},
                "task_metadata": {
                    "in_state": "picked",
                    "out_state": "idle",
                    "required_context_keys": ["destination_location"],
                    "context_mapping": {"location_param": "destination_location"},
                    "part_transition": {
                        "completed": {
                            "state": "ready",
                            "location_param": "destination_location",
                        }
                    },
                },
                "projected_snapshot": {
                    "current_state": "idle",
                    "held_part": "MCP",
                    "gripper_state": "open",
                },
                "projected_part_entry": {
                    "state": "ready",
                    "location": "buffer_a",
                },
                "primitive_steps": [{"primitive": "open_gripper", "params": {}}],
            },
            {
                "resource_jid": "xarm6@localhost",
                "macro_name": "clear_zone",
                "expected_start_state": "placed",
                "expected_snapshot": {"current_state": "placed"},
                "task_metadata": {
                    "in_state": "placed",
                    "out_state": "idle",
                    "required_context_keys": [],
                    "context_mapping": {},
                    "part_transition": None,
                },
                "projected_snapshot": {
                    "current_state": "idle",
                    "held_part": None,
                    "gripper_state": "open",
                    "current_pose_ref": "home",
                },
                "primitive_steps": [{"primitive": "move_to_named_pose", "params": {"pose_name": "home"}}],
            },
        ],
    }

    agent.runtime_recovery = {
        "status": "llm_bridge",
        "bridge_proposal": deepcopy(proposal),
        "failed_task_id": "REQ_2_T3",
    }
    agent._runtime_recovery_context = {
        "failed_task_id": "REQ_2_T3",
        "violations": _runtime_mutex_violation([]),
        "trigger": "safety_block",
        "system_coordination_state": {},
    }

    async def _fake_validation_check() -> None:
        return None

    agent._send_runtime_plan_validation_check = _fake_validation_check  # type: ignore[assignment]

    recovery = asyncio.run(agent.approve_runtime_bridge_proposal())

    inserted = [
        node
        for node in agent.process_planner.nodes
        if str(node.get("change_reason", "")).startswith("INSERTION: Approved bridge recovery macro")
    ]
    assert len(inserted) == 2
    assert inserted[0]["function_name"] == "execute_recovery_macro"
    assert inserted[1]["predecessors"] == [inserted[0]["id"]]
    assert inserted[0]["params"]["destination_location"] == "buffer_a"
    assert inserted[0]["part_name"] == "MCP"
    assert inserted[0]["bridge_sequence_id"] == inserted[1]["bridge_sequence_id"]
    assert inserted[0]["bridge_sequence_index"] == 1
    assert inserted[1]["bridge_sequence_index"] == 2
    assert inserted[0]["bridge_sequence_length"] == 2
    assert inserted[0]["projected_snapshot"]["current_state"] == "idle"
    assert inserted[0]["projected_part_entry"]["location"] == "buffer_a"
    assert inserted[1]["primary_obligation"]["rule_id"] == "SAFE_1"

    blocked_task = next(node for node in agent.process_planner.nodes if node.get("id") == "REQ_2_T3")
    assert inserted[-1]["id"] in blocked_task.get("predecessors", [])

    unrelated_task = next(node for node in agent.process_planner.nodes if node.get("id") == "REQ_9_T1")
    assert inserted[-1]["id"] not in unrelated_task.get("predecessors", [])

    assert recovery["status"] == "validating"
    assert recovery["bridge_approval_state"] == "approved"
    assert recovery["active_bridge_sequence"]["bridge_sequence_id"] == inserted[0]["bridge_sequence_id"]
    assert recovery["active_bridge_sequence"]["bridge_task_ids"] == [inserted[0]["id"], inserted[1]["id"]]


def test_ack_updates_bridge_part_tracker_from_bridge_part_name_stash(tmp_path):
    agent, _sent_messages = _make_product_agent(tmp_path)
    agent.part_tracker = {"MCP": {"state": "in_gripper", "location": "ur5e@localhost_gripper"}}
    agent.process_planner.nodes = [
        {
            "id": "RECOVERY_BRIDGE_1",
            "type": "task",
            "status": "completed",
            "function_name": "execute_recovery_macro",
            "resource_jid": "ur5e@localhost",
            "params": {
                "macro_name": "stash_mcp",
                "primitive_steps": [{"primitive": "open_gripper", "params": {}}],
                "part_name": "MCP",
                "destination_location": "prusa-mk4-2",
            },
            "part_transition": {
                "completed": {
                    "state": "ready",
                    "location_param": "destination_location",
                }
            },
        }
    ]

    asyncio.run(_deliver_ack(agent, task_id="RECOVERY_BRIDGE_1"))

    assert agent.process_planner.nodes[0]["status"] == "completed"
    assert agent.part_tracker["MCP"]["state"] == "ready"
    assert agent.part_tracker["MCP"]["location"] == "prusa-mk4-2"
    assert agent.part_tracker["MCP"]["last_successful_task"] == "RECOVERY_BRIDGE_1"


def test_ack_updates_bridge_part_tracker_from_bridge_part_name_recovery_place(tmp_path):
    agent, _sent_messages = _make_product_agent(tmp_path)
    agent.part_tracker = {"SG": {"state": "misplaced", "location": "ur5e_region"}}
    agent.process_planner.nodes = [
        {
            "id": "RECOVERY_BRIDGE_2",
            "type": "task",
            "status": "completed",
            "function_name": "execute_recovery_macro",
            "resource_jid": "ur5e@localhost",
            "params": {
                "macro_name": "recover_sg",
                "primitive_steps": [{"primitive": "open_gripper", "params": {}}],
                "part_name": "SG",
                "destination_location": "assembly_board-v1",
            },
            "part_transition": {
                "completed": {
                    "state": "assembled",
                    "location_param": "destination_location",
                }
            },
        }
    ]

    asyncio.run(_deliver_ack(agent, task_id="RECOVERY_BRIDGE_2"))

    assert agent.process_planner.nodes[0]["status"] == "completed"
    assert agent.part_tracker["SG"]["state"] == "assembled"
    assert agent.part_tracker["SG"]["location"] == "assembly_board-v1"
    assert agent.part_tracker["SG"]["last_successful_task"] == "RECOVERY_BRIDGE_2"


def test_bridge_ack_matching_post_state_trims_tail_and_revalidates_direct_resume(tmp_path):
    robot = _DummyBridgeRobot(
        "ur5e@localhost",
        "ur5e",
        current_state="idle",
        held_part=None,
    )
    nodes = [
        {
            "id": "REQ_1_T4",
            "type": "task",
            "status": "blocked",
            "function_name": "move_home",
            "params": {},
            "resource_jid": "ur5e@localhost",
            "sequence_index": 1,
            "predecessors": ["RECOVERY_BRIDGE_2"],
            "successors": [],
        },
        {
            "id": "RECOVERY_BRIDGE_1",
            "type": "task",
            "status": "pending",
            "function_name": "execute_recovery_macro",
            "resource_jid": "ur5e@localhost",
            "params": {"macro_name": "clear_fault", "primitive_steps": []},
            "bridge_sequence_id": "BRIDGESEQ_TEST",
            "bridge_sequence_index": 1,
            "bridge_sequence_length": 2,
            "projected_snapshot": {
                "current_state": "idle",
                "held_part": None,
                "gripper_state": "open",
            },
            "predecessors": [],
            "successors": ["RECOVERY_BRIDGE_2"],
        },
        {
            "id": "RECOVERY_BRIDGE_2",
            "type": "task",
            "status": "pending",
            "function_name": "execute_recovery_macro",
            "resource_jid": "ur5e@localhost",
            "params": {"macro_name": "hold_tail", "primitive_steps": []},
            "bridge_sequence_id": "BRIDGESEQ_TEST",
            "bridge_sequence_index": 2,
            "bridge_sequence_length": 2,
            "projected_snapshot": {
                "current_state": "idle",
                "held_part": None,
                "gripper_state": "open",
            },
            "predecessors": ["RECOVERY_BRIDGE_1"],
            "successors": [],
        },
    ]
    agent, _sent_messages, _tools_catalog = _configure_bridge_runtime_agent(
        tmp_path,
        nodes=nodes,
        part_tracker={},
        robot=robot,
    )
    validation_calls: list[bool] = []

    async def _fake_validation_check() -> None:
        validation_calls.append(True)

    async def _unexpected_des(*_args, **_kwargs):
        raise AssertionError("DES handoff should not run when blocked task is directly resumable")

    agent._send_runtime_plan_validation_check = _fake_validation_check  # type: ignore[assignment]
    agent.process_planner.replan_with_feedback_des = _unexpected_des  # type: ignore[assignment]
    _activate_bridge_sequence(
        agent,
        bridge_sequence_id="BRIDGESEQ_TEST",
        bridge_task_ids=["RECOVERY_BRIDGE_1", "RECOVERY_BRIDGE_2"],
        failed_task_id="REQ_1_T4",
        violations=_runtime_violation("REQ_1_T4"),
        system_coordination_state={
            "resource_states": {
                "ur5e@localhost": {"current_state": "picked", "held_part": "MCP"},
            }
        },
    )

    try:
        asyncio.run(_deliver_ack(agent, task_id="RECOVERY_BRIDGE_1"))

        assert agent.process_planner._find_node("RECOVERY_BRIDGE_2") is None
        blocked_task = agent.process_planner._find_node("REQ_1_T4")
        assert blocked_task is not None
        assert blocked_task["status"] == "pending"
        assert "RECOVERY_BRIDGE_2" not in blocked_task.get("predecessors", [])
        assert agent.runtime_recovery["status"] == "validating"
        assert agent.runtime_recovery["active_bridge_sequence"] is None
        assert validation_calls == [True]
        assert agent._runtime_recovery_context["system_coordination_state"]["resource_states"]["ur5e@localhost"]["current_state"] == "idle"
    finally:
        ProductAgent.configure_shared_tools_catalogue()


def test_bridge_ack_matching_post_state_trims_tail_and_inserts_des_patch(tmp_path):
    robot = _DummyBridgeRobot(
        "ur5e@localhost",
        "ur5e",
        current_state="idle",
        held_part=None,
    )
    nodes = [
        {
            "id": "REQ_1_T4",
            "type": "task",
            "status": "blocked",
            "function_name": "place_insert",
            "params": {"part_name": "MCP", "destination_location": "Assembly Station"},
            "resource_jid": "ur5e@localhost",
            "sequence_index": 1,
            "predecessors": ["RECOVERY_BRIDGE_2"],
            "successors": [],
        },
        {
            "id": "RECOVERY_BRIDGE_1",
            "type": "task",
            "status": "pending",
            "function_name": "execute_recovery_macro",
            "resource_jid": "ur5e@localhost",
            "params": {"macro_name": "clear_fault", "primitive_steps": []},
            "bridge_sequence_id": "BRIDGESEQ_TEST",
            "bridge_sequence_index": 1,
            "bridge_sequence_length": 2,
            "projected_snapshot": {
                "current_state": "idle",
                "held_part": None,
                "gripper_state": "open",
            },
            "predecessors": [],
            "successors": ["RECOVERY_BRIDGE_2"],
        },
        {
            "id": "RECOVERY_BRIDGE_2",
            "type": "task",
            "status": "pending",
            "function_name": "execute_recovery_macro",
            "resource_jid": "ur5e@localhost",
            "params": {"macro_name": "tail_macro", "primitive_steps": []},
            "bridge_sequence_id": "BRIDGESEQ_TEST",
            "bridge_sequence_index": 2,
            "bridge_sequence_length": 2,
            "projected_snapshot": {
                "current_state": "idle",
                "held_part": None,
                "gripper_state": "open",
            },
            "predecessors": ["RECOVERY_BRIDGE_1"],
            "successors": [],
        },
    ]
    agent, _sent_messages, _tools_catalog = _configure_bridge_runtime_agent(
        tmp_path,
        nodes=nodes,
        part_tracker={"MCP": {"state": "in_transit", "location": "Assembly Station"}},
        robot=robot,
    )
    validation_calls: list[bool] = []

    async def _fake_validation_check() -> None:
        validation_calls.append(True)

    async def _fake_des(
        violations,
        system_coordination_state=None,
        bridge_feedback="",
        *,
        allow_bridge_fallback=True,
        ignored_task_ids=None,
    ):
        assert bridge_feedback == ""
        assert allow_bridge_fallback is False
        assert set(ignored_task_ids or set()) == {"RECOVERY_BRIDGE_2"}
        des_task = {
            "id": "RECOVERY_DES_1",
            "function_name": "move_home",
            "params": {},
            "resource_jid": "ur5e@localhost",
            "sequence_index": 99,
            "predecessors": [],
            "successors": [],
            "change_reason": "INSERTION: DES recovery — move_home on ur5e@localhost",
        }
        blocked = agent.process_planner._find_node("REQ_1_T4")
        preds = list(dict.fromkeys(list(blocked.get("predecessors", [])) + ["RECOVERY_DES_1"]))
        agent.process_planner._apply_replan_patch(
            [
                des_task,
                {
                    "id": "REQ_1_T4",
                    "predecessors": preds,
                    "change_reason": "MODIFICATION: gate REQ_1_T4 after RECOVERY_DES_1",
                },
            ]
        )
        return {
            "plan_changed": True,
            "used_llm_bridge": False,
            "human_required": False,
            "message": "DES recovery produced 1 task.",
        }

    agent._send_runtime_plan_validation_check = _fake_validation_check  # type: ignore[assignment]
    agent.process_planner.replan_with_feedback_des = _fake_des  # type: ignore[assignment]
    _activate_bridge_sequence(
        agent,
        bridge_sequence_id="BRIDGESEQ_TEST",
        bridge_task_ids=["RECOVERY_BRIDGE_1", "RECOVERY_BRIDGE_2"],
        failed_task_id="REQ_1_T4",
        violations=_runtime_violation("REQ_1_T4"),
        system_coordination_state={
            "resource_states": {
                "ur5e@localhost": {"current_state": "positioned", "held_part": "MCP"},
            }
        },
    )

    try:
        asyncio.run(_deliver_ack(agent, task_id="RECOVERY_BRIDGE_1"))

        assert agent.process_planner._find_node("RECOVERY_BRIDGE_2") is None
        assert agent.process_planner._find_node("RECOVERY_DES_1") is not None
        blocked_task = agent.process_planner._find_node("REQ_1_T4")
        assert blocked_task is not None
        assert blocked_task["status"] == "pending"
        assert "RECOVERY_DES_1" in blocked_task.get("predecessors", [])
        assert "RECOVERY_BRIDGE_2" not in blocked_task.get("predecessors", [])
        assert agent.runtime_recovery["status"] == "validating"
        assert validation_calls == [True]
    finally:
        ProductAgent.configure_shared_tools_catalogue()


def test_bridge_ack_matching_post_state_continues_tail_when_des_still_missing(tmp_path):
    robot = _DummyBridgeRobot(
        "ur5e@localhost",
        "ur5e",
        current_state="idle",
        held_part=None,
    )
    nodes = [
        {
            "id": "REQ_1_T4",
            "type": "task",
            "status": "blocked",
            "function_name": "place_insert",
            "params": {"part_name": "MCP", "destination_location": "Assembly Station"},
            "resource_jid": "ur5e@localhost",
            "sequence_index": 1,
            "predecessors": ["RECOVERY_BRIDGE_2"],
            "successors": [],
        },
        {
            "id": "RECOVERY_BRIDGE_1",
            "type": "task",
            "status": "pending",
            "function_name": "execute_recovery_macro",
            "resource_jid": "ur5e@localhost",
            "params": {"macro_name": "clear_fault", "primitive_steps": []},
            "bridge_sequence_id": "BRIDGESEQ_TEST",
            "bridge_sequence_index": 1,
            "bridge_sequence_length": 2,
            "projected_snapshot": {
                "current_state": "idle",
                "held_part": None,
                "gripper_state": "open",
            },
            "predecessors": [],
            "successors": ["RECOVERY_BRIDGE_2"],
        },
        {
            "id": "RECOVERY_BRIDGE_2",
            "type": "task",
            "status": "pending",
            "function_name": "execute_recovery_macro",
            "resource_jid": "ur5e@localhost",
            "params": {"macro_name": "tail_macro", "primitive_steps": []},
            "bridge_sequence_id": "BRIDGESEQ_TEST",
            "bridge_sequence_index": 2,
            "bridge_sequence_length": 2,
            "projected_snapshot": {
                "current_state": "idle",
                "held_part": None,
                "gripper_state": "open",
            },
            "predecessors": ["RECOVERY_BRIDGE_1"],
            "successors": [],
        },
    ]
    agent, _sent_messages, _tools_catalog = _configure_bridge_runtime_agent(
        tmp_path,
        nodes=nodes,
        part_tracker={"MCP": {"state": "in_transit", "location": "Assembly Station"}},
        robot=robot,
    )
    validation_calls: list[bool] = []

    async def _fake_validation_check() -> None:
        validation_calls.append(True)

    async def _fake_des(*_args, **_kwargs):
        return {
            "plan_changed": False,
            "used_llm_bridge": False,
            "human_required": False,
            "des_recovery_missing": True,
            "message": "DES reevaluation found no modeled continuation.",
        }

    agent._send_runtime_plan_validation_check = _fake_validation_check  # type: ignore[assignment]
    agent.process_planner.replan_with_feedback_des = _fake_des  # type: ignore[assignment]
    _activate_bridge_sequence(
        agent,
        bridge_sequence_id="BRIDGESEQ_TEST",
        bridge_task_ids=["RECOVERY_BRIDGE_1", "RECOVERY_BRIDGE_2"],
        failed_task_id="REQ_1_T4",
        violations=_runtime_violation("REQ_1_T4"),
        system_coordination_state={
            "resource_states": {
                "ur5e@localhost": {"current_state": "positioned", "held_part": "MCP"},
            }
        },
    )

    try:
        asyncio.run(_deliver_ack(agent, task_id="RECOVERY_BRIDGE_1"))

        assert agent.process_planner._find_node("RECOVERY_BRIDGE_2") is not None
        assert agent.runtime_recovery["status"] == "resolved"
        assert agent.runtime_recovery["active_bridge_sequence"]["bridge_sequence_id"] == "BRIDGESEQ_TEST"
        assert validation_calls == []
    finally:
        ProductAgent.configure_shared_tools_catalogue()


def test_bridge_ack_final_macro_without_des_continuation_fails_closed(tmp_path):
    robot = _DummyBridgeRobot(
        "ur5e@localhost",
        "ur5e",
        current_state="idle",
        held_part=None,
    )
    nodes = [
        {
            "id": "REQ_1_T4",
            "type": "task",
            "status": "blocked",
            "function_name": "place_insert",
            "params": {"part_name": "MCP", "destination_location": "Assembly Station"},
            "resource_jid": "ur5e@localhost",
            "sequence_index": 1,
            "predecessors": ["RECOVERY_BRIDGE_1"],
            "successors": [],
        },
        {
            "id": "RECOVERY_BRIDGE_1",
            "type": "task",
            "status": "pending",
            "function_name": "execute_recovery_macro",
            "resource_jid": "ur5e@localhost",
            "params": {"macro_name": "last_macro", "primitive_steps": []},
            "bridge_sequence_id": "BRIDGESEQ_TEST",
            "bridge_sequence_index": 1,
            "bridge_sequence_length": 1,
            "projected_snapshot": {
                "current_state": "idle",
                "held_part": None,
                "gripper_state": "open",
            },
            "predecessors": [],
            "successors": [],
        },
    ]
    agent, _sent_messages, _tools_catalog = _configure_bridge_runtime_agent(
        tmp_path,
        nodes=nodes,
        part_tracker={"MCP": {"state": "in_transit", "location": "Assembly Station"}},
        robot=robot,
    )

    async def _fake_des(*_args, **_kwargs):
        return {
            "plan_changed": False,
            "used_llm_bridge": False,
            "human_required": False,
            "des_recovery_missing": True,
            "message": "DES reevaluation found no modeled continuation.",
        }

    agent.process_planner.replan_with_feedback_des = _fake_des  # type: ignore[assignment]
    _activate_bridge_sequence(
        agent,
        bridge_sequence_id="BRIDGESEQ_TEST",
        bridge_task_ids=["RECOVERY_BRIDGE_1"],
        failed_task_id="REQ_1_T4",
        violations=_runtime_violation("REQ_1_T4"),
        system_coordination_state={
            "resource_states": {
                "ur5e@localhost": {"current_state": "positioned", "held_part": "MCP"},
            }
        },
    )

    try:
        asyncio.run(_deliver_ack(agent, task_id="RECOVERY_BRIDGE_1"))

        assert agent.runtime_recovery["status"] == "human_required"
        assert agent.runtime_recovery["active_bridge_sequence"]["state"] == "failed"
        assert "no valid continuation" in agent.runtime_recovery["message"]
    finally:
        ProductAgent.configure_shared_tools_catalogue()


def test_bridge_ack_snapshot_mismatch_fails_closed_and_trims_tail(tmp_path):
    robot = _DummyBridgeRobot(
        "ur5e@localhost",
        "ur5e",
        current_state="idle",
        held_part=None,
    )
    nodes = [
        {
            "id": "REQ_1_T4",
            "type": "task",
            "status": "blocked",
            "function_name": "move_home",
            "params": {},
            "resource_jid": "ur5e@localhost",
            "sequence_index": 1,
            "predecessors": ["RECOVERY_BRIDGE_2"],
            "successors": [],
        },
        {
            "id": "RECOVERY_BRIDGE_1",
            "type": "task",
            "status": "pending",
            "function_name": "execute_recovery_macro",
            "resource_jid": "ur5e@localhost",
            "params": {"macro_name": "diverge", "primitive_steps": []},
            "bridge_sequence_id": "BRIDGESEQ_TEST",
            "bridge_sequence_index": 1,
            "bridge_sequence_length": 2,
            "projected_snapshot": {
                "current_state": "placed",
                "held_part": None,
                "gripper_state": "open",
            },
            "predecessors": [],
            "successors": ["RECOVERY_BRIDGE_2"],
        },
        {
            "id": "RECOVERY_BRIDGE_2",
            "type": "task",
            "status": "pending",
            "function_name": "execute_recovery_macro",
            "resource_jid": "ur5e@localhost",
            "params": {"macro_name": "tail_macro", "primitive_steps": []},
            "bridge_sequence_id": "BRIDGESEQ_TEST",
            "bridge_sequence_index": 2,
            "bridge_sequence_length": 2,
            "projected_snapshot": {
                "current_state": "idle",
                "held_part": None,
                "gripper_state": "open",
            },
            "predecessors": ["RECOVERY_BRIDGE_1"],
            "successors": [],
        },
    ]
    agent, _sent_messages, _tools_catalog = _configure_bridge_runtime_agent(
        tmp_path,
        nodes=nodes,
        part_tracker={},
        robot=robot,
    )
    _activate_bridge_sequence(
        agent,
        bridge_sequence_id="BRIDGESEQ_TEST",
        bridge_task_ids=["RECOVERY_BRIDGE_1", "RECOVERY_BRIDGE_2"],
        failed_task_id="REQ_1_T4",
        violations=_runtime_violation("REQ_1_T4"),
        system_coordination_state={
            "resource_states": {
                "ur5e@localhost": {"current_state": "picked", "held_part": "MCP"},
            }
        },
    )

    try:
        asyncio.run(_deliver_ack(agent, task_id="RECOVERY_BRIDGE_1"))

        assert agent.process_planner._find_node("RECOVERY_BRIDGE_2") is None
        assert agent.runtime_recovery["status"] == "human_required"
        assert "diverged from its approved projected post-state" in agent.runtime_recovery["message"]
    finally:
        ProductAgent.configure_shared_tools_catalogue()


def test_bridge_ack_failed_runtime_macro_fails_closed_with_observations(tmp_path):
    robot = _DummyBridgeRobot(
        "ur5e@localhost",
        "ur5e",
        current_state="idle",
        held_part=None,
    )
    nodes = [
        {
            "id": "RECOVERY_BRIDGE_1",
            "type": "task",
            "status": "pending",
            "function_name": "execute_recovery_macro",
            "resource_jid": "ur5e@localhost",
            "params": {"macro_name": "runtime_semantic_check", "primitive_steps": []},
            "bridge_sequence_id": "BRIDGESEQ_TEST",
            "bridge_sequence_index": 1,
            "bridge_sequence_length": 2,
            "projected_snapshot": {
                "current_state": "idle",
                "held_part": None,
                "gripper_state": "open",
            },
            "predecessors": [],
            "successors": ["RECOVERY_BRIDGE_2"],
        },
        {
            "id": "RECOVERY_BRIDGE_2",
            "type": "task",
            "status": "pending",
            "function_name": "execute_recovery_macro",
            "resource_jid": "ur5e@localhost",
            "params": {"macro_name": "tail_macro", "primitive_steps": []},
            "bridge_sequence_id": "BRIDGESEQ_TEST",
            "bridge_sequence_index": 2,
            "bridge_sequence_length": 2,
            "projected_snapshot": {
                "current_state": "idle",
                "held_part": None,
                "gripper_state": "open",
            },
            "predecessors": ["RECOVERY_BRIDGE_1"],
            "successors": [],
        },
    ]
    agent, _sent_messages, _tools_catalog = _configure_bridge_runtime_agent(
        tmp_path,
        nodes=nodes,
        part_tracker={},
        robot=robot,
    )
    _activate_bridge_sequence(
        agent,
        bridge_sequence_id="BRIDGESEQ_TEST",
        bridge_task_ids=["RECOVERY_BRIDGE_1", "RECOVERY_BRIDGE_2"],
        failed_task_id="REQ_1_T4",
        violations=_runtime_violation("REQ_1_T4"),
        system_coordination_state={"resource_states": {"ur5e@localhost": {"current_state": "idle"}}},
    )

    try:
        asyncio.run(
            _deliver_ack(
                agent,
                task_id="RECOVERY_BRIDGE_1",
                status="failed",
                content="runtime semantic validation failed",
                observations={"semantic_error": "missing required param 'z'"},
            )
        )

        assert agent.process_planner._find_node("RECOVERY_BRIDGE_2") is None
        assert agent.runtime_recovery["status"] == "human_required"
        assert "runtime semantic validation failed" in agent.runtime_recovery["message"]
        assert agent.runtime_recovery["active_bridge_sequence"]["last_observations"]["semantic_error"] == "missing required param 'z'"
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
