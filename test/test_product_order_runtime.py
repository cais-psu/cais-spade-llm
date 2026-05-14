from __future__ import annotations

import asyncio
import json
import logging
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

from cais_spade_llm.agents.central_controller.central_controller_agent import (
    CentralControllerAgent,
)
from cais_spade_llm.agents.central_controller.online_fsa_monitor import OnlineFsaMonitor
from cais_spade_llm.agents.central_controller.plan_safety_validator import PlanSafetyValidator
from cais_spade_llm.agents.central_controller.safety_logic import SafetyLogic
from cais_spade_llm.agents.intelligent_product.process_planner import ProcessPlanner
from cais_spade_llm.agents.intelligent_product.product_agent import ProductAgent
from cais_spade_llm.agents.intelligent_product.product_recovery_controller import (
    ProductRecoveryController,
)
from cais_spade_llm.product.order import validate_product_order
from cais_spade_llm.ui.bridge import SystemBridge
from cais_spade_llm.ui.components.dag_graph import nodes_to_mermaid


ROOT = Path(__file__).resolve().parents[1]
GEOMETRY_PATH = ROOT / "cais_spade_llm/specification/products/geometry/assembly_board-v1.json"
TOOLS_PATH = ROOT / "cais_spade_llm/initialization/tools.json"


def _geometry() -> dict:
    payload = json.loads(GEOMETRY_PATH.read_text(encoding="utf-8"))
    return dict(payload["gazebo"])


def _tools_catalog() -> list[dict]:
    return json.loads(TOOLS_PATH.read_text(encoding="utf-8"))


def _order(**overrides):
    payload = {
        "product": "assembly_board-v1",
        "product_jid": "assembly_board-v1@localhost",
        "quantity": 1,
        "objective": "assemble assembly_board-v1",
    }
    payload.update(overrides)
    return payload


def _resources():
    return [
        SimpleNamespace(
            jid="ur5e@localhost",
            static_capabilities={
                "reachability": ["prusa-mk4-1", "prusa-mk3", "assembly_board-v1"],
                "gripper_reach": {
                    "frame": "world",
                    "origin_pose": {"x": 0.0, "y": 0.50, "z": 1.021},
                    "max_xy_radius_m": 0.70,
                    "z_min_m": 0.85,
                    "z_max_m": 1.60,
                    "tolerance_m": 0.01,
                },
                "staging_areas": {
                    "prusa-mk3": {"anchor_pose": {"x": -0.4, "y": 0.0, "z": 1.04}},
                    "prusa-mk4-1": {"anchor_pose": {"x": 0.4, "y": 0.3, "z": 1.04}},
                },
            },
        ),
        SimpleNamespace(
            jid="xarm6@localhost",
            static_capabilities={
                "reachability": ["prusa-mk4-2", "prusa-mk3", "assembly_board-v1"],
                "gripper_reach": {
                    "frame": "world",
                    "origin_pose": {"x": 0.0, "y": -0.50, "z": 1.021},
                    "max_xy_radius_m": 0.70,
                    "z_min_m": 0.90,
                    "z_max_m": 1.50,
                    "tolerance_m": 0.01,
                },
                "staging_areas": {
                    "prusa-mk3": {"anchor_pose": {"x": -0.4, "y": 0.0, "z": 1.04}},
                    "prusa-mk4-2": {"anchor_pose": {"x": 0.4, "y": -0.3, "z": 1.04}},
                },
            },
        ),
    ]


def _planner(
    order_payload: dict,
    safety_text: str = "",
    *,
    resources: list[SimpleNamespace] | None = None,
    tools_catalog: list[dict] | None = None,
) -> ProcessPlanner:
    agent = SimpleNamespace(
        jid="assembly_board-v1@localhost",
        logger=logging.getLogger("test.product_order"),
        product_geometry=_geometry(),
        product_order_file="",
        tools_catalog=_tools_catalog() if tools_catalog is None else tools_catalog,
    )
    planner = ProcessPlanner(agent, _resources() if resources is None else resources)
    planner.build_from_product_order(order_payload, safety_text=safety_text)
    return planner


def _lg_before_mcp_validator() -> PlanSafetyValidator:
    rules = [
        {
            "id": "SAFE_LG_BEFORE_MCP",
            "raw_text": "LG must be placed before MCP",
            "ltlf": "not MCP.place_insert.start until LG.place_insert.done",
            "aps": [
                {
                    "label": "ap001",
                    "full": "ap/task/LG/any/place_insert/any",
                    "function": "place_insert",
                },
                {
                    "label": "ap002",
                    "full": "ap/task/MCP/any/place_insert/any",
                    "function": "place_insert",
                },
            ],
        }
    ]
    validator = PlanSafetyValidator(
        rules=rules,
        dfa_map={
            "SAFE_LG_BEFORE_MCP": 'digraph { init -> q0; q0 -> q0 [label="true"]; }'
        },
        tools_catalog=_tools_catalog(),
    )
    validator.dfas["SAFE_LG_BEFORE_MCP"] = {
            "initial": "q0",
            "transitions": {
                "q0": [
                    ["ap002", "violation"],
                    ["ap001", "q1"],
                    ["true", "q0"],
                ],
                "q1": [["true", "q1"]],
                "violation": [["true", "violation"]],
            },
            "violation_state": "violation",
            "accepting_states": ["q0", "q1"],
            "ap_symbols": ["ap001", "ap002"],
        }
    return validator


def _runtime_planner(
    order_payload: dict,
    safety_text: str = "",
    *,
    resources: list[SimpleNamespace] | None = None,
    tools_catalog: list[dict] | None = None,
) -> ProcessPlanner:
    agent = SimpleNamespace(
        jid="assembly_board-v1@localhost",
        logger=logging.getLogger("test.product_order"),
        product_geometry=_geometry(),
        product_order_file="",
        tools_catalog=_tools_catalog() if tools_catalog is None else tools_catalog,
    )
    planner = ProcessPlanner(agent, _resources() if resources is None else resources)
    planner.build_product_order_runtime_skeleton(order_payload, safety_text=safety_text)
    return planner


def _system_plan_row(planner: ProcessPlanner, part_name: str) -> dict:
    return next(
        row
        for row in planner.last_product_order_artifact["system_plan"]
        if row["part"] == part_name
    )


def _candidate_rows(row: dict, resource_jid: str) -> list[dict]:
    return [
        candidate
        for candidate in row["product_bidding"]["candidates"]
        if candidate["resource_jid"] == resource_jid
    ]


def _retry_ready_agent(node_status: str = "blocked"):
    agent = SimpleNamespace(
        process_planner=SimpleNamespace(
            nodes=[
                {
                    "type": "task",
                    "id": "REQ_2_T3",
                    "status": node_status,
                }
            ]
        ),
        task_states={},
        execution_timeline=[],
        cca_jid="cca@localhost",
    )
    ProductRecoveryController(agent).bind_methods()
    return agent


def _safety_mutex_dfa_dot() -> str:
    return """
digraph MONA_DFA {
  init -> 1;
  node [shape = doublecircle]; 1;
  node [shape = circle]; 1;
  1 -> 1 [label="~ap001"];
  1 -> 2 [label="ap001"];
  2 -> 2 [label="true"];
}
"""


def _write_safety_mutex_artifacts(
    tmp_path: Path,
    *,
    safety_text: str = "robots must not overlap in assembly board placement",
    cached_text: str | None = None,
    include_dfa: bool = True,
) -> tuple[Path, Path]:
    safety_file = tmp_path / "safety_mutex.txt"
    safety_file.write_text(safety_text, encoding="utf-8")
    logic_path = tmp_path / "cca_safety_logic.json"
    logic_path.write_text(
        json.dumps(
            {
                "preview_interpretation_summary": "",
                "safety_text_sha256": SafetyLogic.compute_safety_text_sha256(
                    cached_text if cached_text is not None else safety_text
                ),
                "rules": [
                    {
                        "id": "SAFE_1",
                        "raw_text": safety_text,
                        "ltlf": "G (!ap001)",
                        "aps": [
                            {
                                "label": "ap001",
                                "full": "ap/task/any/ur5e/place_approach/any",
                                "function": "place_approach",
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    if include_dfa:
        (tmp_path / "SAFE_1_dfa.dot").write_text(
            _safety_mutex_dfa_dot(),
            encoding="utf-8",
        )
    return safety_file, logic_path


def _fast_path_agent(
    tmp_path: Path,
    *,
    cached_text: str | None = None,
    include_dfa: bool = True,
):
    safety_file, logic_path = _write_safety_mutex_artifacts(
        tmp_path,
        cached_text=cached_text,
        include_dfa=include_dfa,
    )
    agent = SimpleNamespace(
        logger=logging.getLogger("test.product_fast_path"),
        safety_file=safety_file,
        safety_logic_path=logic_path,
        safety_text_has_requirements=True,
        _runtime_safety_fast_path_cache={},
        _runtime_safety_history_cache={},
        tools_catalog=[],
        runtime_recovery={},
        _runtime_recovery_context={},
        _runtime_bridge_validation_policy="validated",
        _runtime_repair_fail_streak=0,
        _runtime_repair_max_attempts=3,
        _geometry_for_part=lambda part_name: {},
    )
    ProductRecoveryController(agent).bind_methods()
    return agent


def _bind_product_runtime_payload_methods(agent):
    agent._fsa_task_ids = ProductAgent._fsa_task_ids
    for name in (
        "_build_plan_validation_payload",
        "_filter_runtime_context_completed_task_ids_for_fsa",
        "_safety_event_history_cache_key",
        "_build_runtime_plan_context",
        "_build_safety_event_history",
    ):
        setattr(agent, name, getattr(ProductAgent, name).__get__(agent, type(agent)))
    return agent


def _payload_agent(planner: ProcessPlanner, execution_timeline: list[dict]):
    agent = SimpleNamespace(
        jid="assembly_board-v1@localhost",
        logger=logging.getLogger("test.product_payload"),
        process_planner=planner,
        execution_timeline=list(execution_timeline),
        _runtime_safety_fast_path_cache={},
        _runtime_safety_history_cache={},
        _runtime_recovery_context={},
        runtime_recovery={},
        _runtime_bridge_validation_policy="validated",
        _runtime_repair_fail_streak=0,
        _runtime_repair_max_attempts=3,
        safety_file=None,
        safety_logic_path=Path("missing"),
        tools_catalog=[],
        _geometry_for_part=lambda part_name: {},
    )
    ProductRecoveryController(agent).bind_methods()
    return _bind_product_runtime_payload_methods(agent)


def test_task_retry_ready_requeues_blocked_task():
    agent = _retry_ready_agent("blocked")

    reactivated = agent._handle_task_retry_ready(["REQ_2_T3"])

    assert reactivated == 1
    assert agent.process_planner.nodes[0]["status"] == "pending"
    assert agent.task_states["REQ_2_T3"] == "pending"
    assert agent.execution_timeline[-1]["task_id"] == "REQ_2_T3"
    assert agent.execution_timeline[-1]["status"] == "requeued"


def test_task_retry_ready_waits_for_delayed_blocked_ack():
    agent = _retry_ready_agent("accepted")

    reactivated = agent._handle_task_retry_ready(["REQ_2_T3"])

    assert reactivated == 0
    assert agent.process_planner.nodes[0]["status"] == "accepted"
    assert agent._pending_task_retry_ready_ids == {"REQ_2_T3"}

    agent.process_planner.nodes[0]["status"] = "blocked"
    reactivated = agent._handle_task_retry_ready(["REQ_2_T3"])

    assert reactivated == 1
    assert agent.process_planner.nodes[0]["status"] == "pending"
    assert agent.task_states["REQ_2_T3"] == "pending"
    assert agent._pending_task_retry_ready_ids == set()

    reactivated = agent._handle_task_retry_ready(["REQ_2_T3"])

    assert reactivated == 0
    assert agent._pending_task_retry_ready_ids == set()


def test_runtime_safety_fast_path_only_for_ap_empty_task(tmp_path: Path):
    agent = _fast_path_agent(tmp_path)
    task_node = {
        "type": "task",
        "id": "REQ_2_T2",
        "function_name": "pick_grasp",
        "resource_jid": "ur5e@localhost",
        "params": {"part_name": "MRP"},
    }

    ap_sets = agent._runtime_safety_ap_sets_for_task(
        task_node,
        dict(task_node["params"], task_id="REQ_2_T2"),
    )
    params = agent._dispatch_params_for_task_node(task_node)

    assert ap_sets == {"candidate_aps": [], "predicted_state_aps": []}
    assert params["start_safety_mode"] == "fast_path"


def test_runtime_safety_fast_path_keeps_ap_relevant_task_on_cca_check(tmp_path: Path):
    agent = _fast_path_agent(tmp_path)
    task_node = {
        "type": "task",
        "id": "REQ_2_T3",
        "function_name": "place_approach",
        "resource_jid": "ur5e@localhost",
        "params": {"part_name": "MRP"},
    }

    ap_sets = agent._runtime_safety_ap_sets_for_task(
        task_node,
        dict(task_node["params"], task_id="REQ_2_T3"),
    )
    params = agent._dispatch_params_for_task_node(task_node)

    assert ap_sets == {"candidate_aps": ["ap001"], "predicted_state_aps": []}
    assert "start_safety_mode" not in params


def test_assembly_board_place_insert_forces_cca_check_when_safety_requirements_loaded(tmp_path: Path):
    agent = _fast_path_agent(tmp_path)
    task_node = {
        "type": "task",
        "id": "REQ_1_T4",
        "function_name": "place_insert",
        "resource_jid": "ur5e@localhost",
        "params": {
            "part_name": "MG",
            "destination_location": "assembly_board-v1",
        },
    }

    params = agent._dispatch_params_for_task_node(task_node)

    assert params["start_safety_mode"] == "cca_check"


def test_assembly_board_place_insert_uses_fast_path_without_safety_requirements(tmp_path: Path):
    agent = _fast_path_agent(tmp_path)
    agent.safety_text_has_requirements = False
    task_node = {
        "type": "task",
        "id": "REQ_1_T4",
        "function_name": "place_insert",
        "resource_jid": "ur5e@localhost",
        "params": {
            "part_name": "MG",
            "destination_location": "assembly_board-v1",
        },
    }

    params = agent._dispatch_params_for_task_node(task_node)

    assert params["start_safety_mode"] == "fast_path"


def _run_plan_executor_guard_case(
    *,
    task_node: dict,
    planner_nodes: list[dict],
    safety_text_has_requirements: bool,
):
    sent_messages: list[dict] = []
    fake_agent = SimpleNamespace(
        resource_jids=["xarm6@localhost", "ur5e@localhost"],
        logger=logging.getLogger("test.product_order_executor_guard"),
        task_states={},
        runtime_recovery={},
        process_planner=SimpleNamespace(nodes=planner_nodes),
        safety_text_has_requirements=safety_text_has_requirements,
        _runtime_recovery_blocks_execution=lambda: False,
        _next_dispatchable_task_node=lambda: task_node,
        _active_bridge_blocks_nominal_dispatch=lambda: False,
        _active_bridge_sequence=lambda: None,
        _reconstruct_active_bridge_sequence_for_validation=lambda: None,
        _bridge_sequence_task_ids=lambda ids: list(ids or []),
        _set_runtime_recovery=lambda **kwargs: None,
        _compose_task_msg=lambda **kwargs: dict(kwargs),
        _dispatch_params_for_task_node=lambda task_node: dict(task_node.get("params") or {}),
    )
    executor = ProductAgent._PlanExecutor()
    executor.agent = fake_agent  # type: ignore[attr-defined]

    async def _fake_send(msg: dict) -> None:
        sent_messages.append(dict(msg))

    executor.send = _fake_send  # type: ignore[method-assign]
    asyncio.run(executor.run())
    return sent_messages, fake_agent


def test_plan_executor_allows_cross_resource_assembly_board_place_without_safety_requirements():
    task_node = {
        "type": "task",
        "id": "REQ_2_T4",
        "status": "pending",
        "function_name": "place_insert",
        "resource_jid": "ur5e@localhost",
        "params": {
            "part_name": "MRP",
            "destination_location": "assembly_board-v1",
        },
    }
    active_node = {
        "type": "task",
        "id": "REQ_4_T3",
        "status": "running",
        "function_name": "place_approach",
        "resource_jid": "xarm6@localhost",
        "params": {
            "part_name": "SRP",
            "destination_location": "assembly_board-v1",
        },
    }

    sent_messages, fake_agent = _run_plan_executor_guard_case(
        task_node=task_node,
        planner_nodes=[task_node, active_node],
        safety_text_has_requirements=False,
    )

    assert sent_messages
    assert str(sent_messages[0].get("task_id") or "").strip() == "REQ_2_T4"
    assert str(task_node.get("status") or "").strip() == "dispatched"
    assert fake_agent.task_states["REQ_2_T4"] == "dispatched"


def test_plan_executor_allows_cross_resource_assembly_board_place_with_safety_requirements():
    task_node = {
        "type": "task",
        "id": "REQ_2_T4",
        "status": "pending",
        "function_name": "place_insert",
        "resource_jid": "ur5e@localhost",
        "params": {
            "part_name": "MRP",
            "destination_location": "assembly_board-v1",
        },
    }
    active_node = {
        "type": "task",
        "id": "REQ_4_T3",
        "status": "running",
        "function_name": "place_approach",
        "resource_jid": "xarm6@localhost",
        "params": {
            "part_name": "SRP",
            "destination_location": "assembly_board-v1",
        },
    }

    sent_messages, fake_agent = _run_plan_executor_guard_case(
        task_node=task_node,
        planner_nodes=[task_node, active_node],
        safety_text_has_requirements=True,
    )

    assert sent_messages
    assert str(sent_messages[0].get("task_id") or "").strip() == "REQ_2_T4"
    assert str(task_node.get("status") or "").strip() == "dispatched"
    assert fake_agent.task_states["REQ_2_T4"] == "dispatched"


def test_plan_executor_keeps_same_resource_guard_without_safety_requirements():
    task_node = {
        "type": "task",
        "id": "REQ_2_T4",
        "status": "pending",
        "function_name": "place_insert",
        "resource_jid": "ur5e@localhost",
        "params": {
            "part_name": "MRP",
            "destination_location": "assembly_board-v1",
        },
    }
    active_node = {
        "type": "task",
        "id": "REQ_3_T3",
        "status": "running",
        "function_name": "place_approach",
        "resource_jid": "ur5e@localhost",
        "params": {
            "part_name": "LRP",
            "destination_location": "assembly_board-v1",
        },
    }

    sent_messages, fake_agent = _run_plan_executor_guard_case(
        task_node=task_node,
        planner_nodes=[task_node, active_node],
        safety_text_has_requirements=False,
    )

    assert sent_messages == []
    assert str(task_node.get("status") or "").strip() == "pending"
    assert "REQ_2_T4" not in fake_agent.task_states


def test_runtime_safety_fast_path_fails_closed_for_stale_cache(tmp_path: Path):
    agent = _fast_path_agent(tmp_path, cached_text="old safety text")
    task_node = {
        "type": "task",
        "id": "REQ_2_T2",
        "function_name": "pick_grasp",
        "resource_jid": "ur5e@localhost",
        "params": {"part_name": "MRP"},
    }

    params = agent._dispatch_params_for_task_node(task_node)

    assert agent._runtime_safety_ap_sets_for_task(task_node, params) is None
    assert "start_safety_mode" not in params


def test_runtime_safety_fast_path_fails_closed_for_missing_dfa(tmp_path: Path):
    agent = _fast_path_agent(tmp_path, include_dfa=False)
    task_node = {
        "type": "task",
        "id": "REQ_2_T2",
        "function_name": "pick_grasp",
        "resource_jid": "ur5e@localhost",
        "params": {"part_name": "MRP"},
    }

    params = agent._dispatch_params_for_task_node(task_node)

    assert agent._runtime_safety_ap_sets_for_task(task_node, params) is None
    assert "start_safety_mode" not in params


def test_safety_event_history_omits_ap_empty_tasks_when_classifier_available(tmp_path: Path):
    agent = _fast_path_agent(tmp_path)
    _bind_product_runtime_payload_methods(agent)
    agent.process_planner = SimpleNamespace(
        nodes=[
            {
                "type": "task",
                "id": "REQ_EMPTY",
                "function_name": "pick_grasp",
                "resource_jid": "ur5e@localhost",
                "params": {"part_name": "MRP"},
            },
            {
                "type": "task",
                "id": "REQ_RELEVANT",
                "function_name": "place_approach",
                "resource_jid": "ur5e@localhost",
                "params": {"part_name": "MRP"},
            },
        ]
    )
    agent.execution_timeline = [
        {"task_id": "REQ_EMPTY", "status": "completed", "timestamp": "1"},
        {"task_id": "REQ_RELEVANT", "status": "completed", "timestamp": "2"},
    ]

    history = agent._build_safety_event_history()
    events = [row["event"] for row in history]

    assert "REQ_EMPTY.start" not in events
    assert "REQ_EMPTY.done" not in events
    assert events == ["REQ_RELEVANT.start", "REQ_RELEVANT.done"]


def test_safety_event_history_keeps_full_history_when_classifier_stale(tmp_path: Path):
    agent = _fast_path_agent(tmp_path, cached_text="old safety text")
    _bind_product_runtime_payload_methods(agent)
    agent.process_planner = SimpleNamespace(
        nodes=[
            {
                "type": "task",
                "id": "REQ_EMPTY",
                "function_name": "pick_grasp",
                "resource_jid": "ur5e@localhost",
                "params": {"part_name": "MRP"},
            }
        ]
    )
    agent.execution_timeline = [
        {"task_id": "REQ_EMPTY", "status": "completed", "timestamp": "1"},
    ]

    history = agent._build_safety_event_history()
    events = [row["event"] for row in history]

    assert events == ["REQ_EMPTY.start", "REQ_EMPTY.done"]


def test_current_safety_cache_loads_only_when_hash_and_dfa_match(tmp_path: Path):
    safety_file, logic_path = _write_safety_mutex_artifacts(tmp_path)
    controller = SimpleNamespace(logger=logging.getLogger("test.safety_cache"))
    safety_logic = SafetyLogic(controller, safety_file)
    cca = SimpleNamespace(
        logger=logging.getLogger("test.cca_cache"),
        safety_logic_path=logic_path,
        tools_catalog=[],
        resource_agents=[],
        safety_rules=[],
        safety_monitor=None,
        _seed_safety_monitor_resource_states=lambda: None,
    )

    loaded = CentralControllerAgent._load_current_safety_cache(
        cca,
        safety_logic,
        safety_file.read_text(encoding="utf-8"),
    )

    assert loaded is True
    assert [rule["id"] for rule in cca.safety_rules] == ["SAFE_1"]
    assert cca.safety_monitor is not None


def test_current_safety_cache_rejects_stale_hash(tmp_path: Path):
    safety_file, logic_path = _write_safety_mutex_artifacts(
        tmp_path,
        cached_text="old safety text",
    )
    controller = SimpleNamespace(logger=logging.getLogger("test.safety_cache"))
    safety_logic = SafetyLogic(controller, safety_file)
    cca = SimpleNamespace(
        logger=logging.getLogger("test.cca_cache"),
        safety_logic_path=logic_path,
        tools_catalog=[],
        resource_agents=[],
        safety_rules=[],
        safety_monitor=None,
        _seed_safety_monitor_resource_states=lambda: None,
    )

    loaded = CentralControllerAgent._load_current_safety_cache(
        cca,
        safety_logic,
        safety_file.read_text(encoding="utf-8"),
    )

    assert loaded is False
    assert cca.safety_monitor is None


def test_product_order_parts_omitted_means_all_geometry_slots():
    validated = validate_product_order(_order(), _geometry())
    assert validated.selected_parts == [
        "SG",
        "MG",
        "LG",
        "SRP",
        "MRP",
        "LRP",
        "SCP",
        "MCP",
        "LCP",
    ]
    assert validated.payload["parts"] == "all"


def test_product_order_parts_all_means_all_geometry_slots():
    validated = validate_product_order(_order(parts="all"), _geometry())
    assert len(validated.selected_parts) == 9


def test_product_order_specific_parts_are_preserved():
    validated = validate_product_order(_order(parts=["LG", "MCP"]), _geometry())
    assert validated.selected_parts == ["LG", "MCP"]
    assert validated.payload["parts"] == ["LG", "MCP"]


def test_product_order_empty_parts_are_rejected():
    with pytest.raises(ValueError, match="parts list must not be empty"):
        validate_product_order(_order(parts=[]), _geometry())


def test_product_order_unknown_parts_are_rejected():
    with pytest.raises(ValueError, match="unknown product order part"):
        validate_product_order(_order(parts=["LG", "UNKNOWN"]), _geometry())


def test_product_order_constraints_are_rejected():
    with pytest.raises(ValueError, match="must not include constraints"):
        validate_product_order(
            _order(parts="all", constraints=["LG must be placed before MCP"]),
            _geometry(),
        )


def test_product_order_planner_builds_all_nine_parts():
    planner = _planner(_order(parts="all"))
    task_nodes = [node for node in planner.nodes if node.get("type") == "task"]
    assert len(task_nodes) == 45
    assert {node["params"].get("part_name") for node in task_nodes if node["function_name"] != "move_home"} == {
        "SG",
        "MG",
        "LG",
        "SRP",
        "MRP",
        "LRP",
        "SCP",
        "MCP",
        "LCP",
    }


def test_product_order_planner_builds_specific_subset():
    planner = _planner(_order(parts=["LG", "MCP"]))
    task_nodes = [node for node in planner.nodes if node.get("type") == "task"]
    assert len(task_nodes) == 10
    assert {node["params"].get("part_name") for node in task_nodes if node["function_name"] != "move_home"} == {
        "LG",
        "MCP",
    }


def test_product_order_planner_records_product_bidding_artifact():
    planner = _planner(_order(parts=["LG"]))
    system_plan = planner.last_product_order_artifact["system_plan"]
    assert len(system_plan) == 1
    product_bidding = system_plan[0]["product_bidding"]
    assert product_bidding["selected_bid"]["event_count"] == 5
    assert product_bidding["selected_bid"]["events"][-1]["function_name"] == "move_home"
    assert any(candidate["status"] == "selected" for candidate in product_bidding["candidates"])
    assert any(candidate["status"] == "incomplete" for candidate in product_bidding["candidates"])


def test_product_order_bidding_distributes_multiple_parts_by_bid_load():
    planner = _planner(_order(parts=["SG", "MG", "LG", "SRP"]))
    system_plan = planner.last_product_order_artifact["system_plan"]
    resource_counts = Counter(row["resource_jid"] for row in system_plan)
    assert resource_counts == Counter({"ur5e@localhost": 2, "xarm6@localhost": 2})


def test_product_order_bidding_keeps_small_gear_on_ur5e_from_prusa_mk3():
    planner = _planner(_order(parts=["SG"]))
    row = _system_plan_row(planner, "SG")

    assert row["resource_jid"] == "ur5e@localhost"
    assert row["source_location"] == "prusa-mk3"
    rejected = _candidate_rows(row, "xarm6@localhost")
    assert rejected
    assert any(
        "source pose outside gripper_reach" in candidate["reason"]
        for candidate in rejected
    )


@pytest.mark.parametrize("part_name", ["MG", "MRP", "MCP"])
def test_product_order_bidding_keeps_medium_set_on_ur5e_side(part_name: str):
    planner = _planner(_order(parts=[part_name]))
    row = _system_plan_row(planner, part_name)

    assert row["resource_jid"] == "ur5e@localhost"
    assert row["source_location"] == "prusa-mk4-1"
    rejected = _candidate_rows(row, "xarm6@localhost")
    assert rejected
    assert all(candidate["status"] == "incomplete" for candidate in rejected)
    assert any(
        "source pose outside gripper_reach" in candidate["reason"]
        for candidate in rejected
    )


@pytest.mark.parametrize("part_name", ["LRP", "LCP"])
def test_product_order_bidding_keeps_large_set_on_xarm6_side(part_name: str):
    planner = _planner(_order(parts=[part_name]))
    row = _system_plan_row(planner, part_name)

    assert row["resource_jid"] == "xarm6@localhost"
    assert row["source_location"] == "prusa-mk4-2"
    rejected = _candidate_rows(row, "ur5e@localhost")
    assert rejected
    assert all(candidate["status"] == "incomplete" for candidate in rejected)
    assert any(
        "source pose outside gripper_reach" in candidate["reason"]
        for candidate in rejected
    )


def test_product_order_source_pose_gripper_reach_filter_representative_parts():
    planner = _runtime_planner(_order(parts=["MG"]))
    resources = {resource.jid: resource for resource in _resources()}
    xarm_reach = resources["xarm6@localhost"].static_capabilities["gripper_reach"]
    ur5e_reach = resources["ur5e@localhost"].static_capabilities["gripper_reach"]

    sg_pose, sg_error = planner._product_order_source_pose(
        part_name="SG",
        source_location="prusa-mk3",
    )
    mg_pose, mg_error = planner._product_order_source_pose(
        part_name="MG",
        source_location="prusa-mk4-1",
    )
    lrp_pose, lrp_error = planner._product_order_source_pose(
        part_name="LRP",
        source_location="prusa-mk4-2",
    )

    assert sg_error == ""
    assert mg_error == ""
    assert lrp_error == ""
    assert planner._product_order_pose_in_gripper_reach(
        pose=sg_pose,
        gripper_reach=xarm_reach,
    )[0] is False
    assert planner._product_order_pose_in_gripper_reach(
        pose=sg_pose,
        gripper_reach=ur5e_reach,
    )[0] is True
    assert planner._product_order_pose_in_gripper_reach(
        pose=mg_pose,
        gripper_reach=ur5e_reach,
    )[0] is True
    assert planner._product_order_pose_in_gripper_reach(
        pose=lrp_pose,
        gripper_reach=xarm_reach,
    )[0] is True
    assert planner._product_order_pose_in_gripper_reach(
        pose=lrp_pose,
        gripper_reach=ur5e_reach,
    )[0] is False


def test_product_order_bidding_source_selection_respects_source_anchor():
    resources = [
        SimpleNamespace(
            jid="ur5e@localhost",
            static_capabilities={
                "reachability": ["prusa-mk4-1", "prusa-mk4-2", "assembly_board-v1"],
                "gripper_reach": {
                    "frame": "world",
                    "origin_pose": {"x": 0.0, "y": 0.50, "z": 1.021},
                    "max_xy_radius_m": 1.0,
                    "z_min_m": 0.85,
                    "z_max_m": 1.60,
                    "tolerance_m": 0.01,
                },
                "staging_areas": {
                    "prusa-mk4-2": {"anchor_pose": {"x": 0.4, "y": -0.3, "z": 1.04}},
                    "prusa-mk4-1": {"anchor_pose": {"x": 0.4, "y": 0.3, "z": 1.04}},
                },
            },
        )
    ]
    planner = _planner(_order(parts=["LG"]), resources=resources)
    system_plan = planner.last_product_order_artifact["system_plan"]
    assert system_plan[0]["resource_jid"] == "ur5e@localhost"
    assert system_plan[0]["source_location"] == "prusa-mk4-2"


def test_product_order_bidding_requires_complete_idle_bid():
    tools_without_move_home = [
        row for row in _tools_catalog() if row.get("function") != "move_home"
    ]
    with pytest.raises(ValueError, match="no complete product bid"):
        _planner(_order(parts=["LG"]), tools_catalog=tools_without_move_home)


def test_product_order_runtime_skeleton_creates_zero_task_nodes_until_commit():
    planner = _runtime_planner(_order(parts=["LG", "MCP"]))
    assert planner.nodes == []
    assert planner.global_fsa is None
    artifact = planner.last_product_order_artifact
    assert artifact["pending_product_order_parts"] == ["LG", "MCP"]
    assert artifact["committed_product_order_parts"] == []
    assert artifact["completed_product_order_parts"] == []
    assert artifact["derived_nodes"] == []


def test_product_order_runtime_commit_first_ready_part_compiles_fsa():
    planner = _runtime_planner(_order(parts=["LG", "MCP"]))
    assert planner.ready_product_order_parts() == ["LG", "MCP"]

    record = planner.commit_product_order_part("LG")
    assert record["part"] == "LG"
    assert len(record["task_ids"]) == 5
    assert all(node["status"] == "pending_validation" for node in planner.nodes)
    assert planner.last_product_order_artifact["bid_evidence_by_part"]["LG"]["selected_bid"]["event_count"] == 5

    fsa = planner.recompile_committed_product_order_fsa()
    assert fsa
    assert planner.global_fsa is fsa


def test_product_order_runtime_completed_part_unlocks_constrained_next_part():
    planner = _runtime_planner(
        _order(parts=["LG", "MCP"]),
        safety_text="LG must be placed before MCP",
    )
    assert planner.ready_product_order_parts() == ["LG"]

    planner.commit_product_order_part("LG")
    for node in planner.nodes:
        node["status"] = "completed"
    assert planner.ready_product_order_parts() == ["MCP"]
    assert planner.last_product_order_artifact["completed_product_order_parts"] == ["LG"]


def test_product_order_runtime_defers_selected_part_until_place_before_predecessor_done():
    planner = _runtime_planner(
        _order(parts=["LRP", "LCP"]),
        safety_text="LCP must be placed before LRP",
    )

    assert planner.last_product_order_artifact["pending_product_order_parts"] == ["LRP", "LCP"]
    assert planner.ready_product_order_parts() == ["LCP"]

    with pytest.raises(ValueError, match="product-order part LRP is not ready"):
        planner.commit_product_order_part("LRP")

    assert planner.last_product_order_artifact["pending_product_order_parts"] == ["LRP", "LCP"]
    assert planner.last_product_order_artifact["committed_product_order_parts"] == []
    assert all(node.get("product_order_part") != "LRP" for node in planner.nodes)

    lcp = planner.commit_product_order_part("LCP")
    assert lcp["part"] == "LCP"
    assert {node.get("product_order_part") for node in planner.nodes} == {"LCP"}
    for node in planner.nodes:
        if node.get("product_order_part") == "LCP":
            node["status"] = "completed"

    assert planner.ready_product_order_parts() == ["LRP"]
    assert planner.last_product_order_artifact["completed_product_order_parts"] == ["LCP"]

    lrp = planner.commit_product_order_part("LRP")
    assert lrp["part"] == "LRP"
    assert any(node.get("product_order_part") == "LRP" for node in planner.nodes)


def test_product_order_runtime_busy_resource_is_excluded_from_bidding():
    planner = _runtime_planner(_order(parts=["LG"]))
    record = planner.commit_product_order_part(
        "LG",
        unavailable_resource_jids={"ur5e@localhost"},
    )
    assert record["resource_jid"] == "xarm6@localhost"


def test_product_order_runtime_availability_change_before_dispatch_rolls_back_and_rebids():
    planner = _runtime_planner(_order(parts=["MG"]))
    first = planner.commit_product_order_part("MG", status="pending")
    first_resource = first["resource_jid"]

    removed = planner.rollback_product_order_committed_parts(["MG"])
    assert removed == ["MG"]
    assert planner.nodes == []
    assert planner.ready_product_order_parts() == ["MG"]

    with pytest.raises(ValueError, match="no available product bid resources"):
        planner.commit_product_order_part(
            "MG",
            unavailable_resource_jids={first_resource},
            status="pending_validation",
        )
    assert planner.nodes == []
    assert planner.ready_product_order_parts() == ["MG"]


def test_product_order_runtime_running_committed_part_is_never_reassigned():
    planner = _runtime_planner(_order(parts=["LG"]))
    record = planner.commit_product_order_part("LG", status="pending")
    first_task_id = record["task_ids"][0]
    first_node = next(node for node in planner.nodes if node["id"] == first_task_id)
    first_node["status"] = "running"

    removed = planner.rollback_product_order_committed_parts(["LG"])
    assert removed == []
    assert planner.nodes
    assert planner.last_product_order_artifact["committed_product_order_parts"] == ["LG"]


def test_active_window_fsa_excludes_completed_nodes_and_keeps_unfinished_nodes():
    planner = _runtime_planner(_order(parts=["LG", "MCP"]))
    lg = planner.commit_product_order_part("LG", status="pending")
    for node in planner.nodes:
        if node["id"] in lg["task_ids"]:
            node["status"] = "completed"
    planner.mark_product_order_part_completed("LG")

    mcp = planner.commit_product_order_part("MCP", status="pending_validation")
    fsa = planner.recompile_committed_product_order_fsa()
    assert fsa
    transition_task_ids = {
        str(t.get("task_id") or "")
        for t in fsa["A"]["Tr"]
        if str(t.get("task_id") or "")
    }
    assert not set(lg["task_ids"]) & transition_task_ids
    assert set(mcp["task_ids"]) <= transition_task_ids


def test_active_window_plan_validation_payload_filters_completed_task_ids():
    planner = _runtime_planner(_order(parts=["LG", "MCP"]))
    lg = planner.commit_product_order_part("LG", status="pending")
    for node in planner.nodes:
        if node["id"] in lg["task_ids"]:
            node["status"] = "completed"
    planner.mark_product_order_part_completed("LG")
    planner.commit_product_order_part("MCP", status="pending_validation")
    planner.recompile_committed_product_order_fsa()
    agent = _payload_agent(
        planner,
        [
            {"task_id": task_id, "status": "completed", "timestamp": str(index)}
            for index, task_id in enumerate(lg["task_ids"])
        ],
    )

    active_payload = agent._build_plan_validation_payload(
        validation_scope="active_window",
        composition_backend="explicit_fsa_dfa",
    )
    full_payload = agent._build_plan_validation_payload()

    assert active_payload["runtime_context"]["completed_task_ids"] == []
    assert full_payload["runtime_context"]["completed_task_ids"] == lg["task_ids"]


def test_active_window_restore_filtered_completed_history_has_no_warning(caplog):
    planner = _runtime_planner(_order(parts=["LG", "MCP"]))
    lg = planner.commit_product_order_part("LG", status="pending")
    for node in planner.nodes:
        if node["id"] in lg["task_ids"]:
            node["status"] = "completed"
    planner.mark_product_order_part_completed("LG")
    planner.commit_product_order_part("MCP", status="pending_validation")
    fsa = planner.recompile_committed_product_order_fsa()
    monitor = OnlineFsaMonitor(fsa)

    with caplog.at_level(logging.WARNING, logger="OnlineFsaMonitor"):
        monitor.restore_runtime_progress(
            completed_task_ids=monitor.filter_task_ids(lg["task_ids"]),
        )

    assert "Could not fully restore completed task" not in caplog.text


def test_active_window_monitor_restore_replays_task_progress_not_state_alias():
    planner = _runtime_planner(_order(parts=["MG", "MRP", "LRP", "MCP", "LCP"]))
    mrp = planner.commit_product_order_part("MRP", status="pending")
    lrp = planner.commit_product_order_part("LRP", status="pending")
    for node in planner.nodes:
        if node["id"] in lrp["task_ids"][:-1]:
            node["status"] = "completed"
        elif node["id"] == lrp["task_ids"][-1]:
            node["status"] = "running"
        elif node["id"] == mrp["task_ids"][0]:
            node["status"] = "running"

    old_fsa = planner.recompile_committed_product_order_fsa()
    old_monitor = OnlineFsaMonitor(old_fsa)
    old_monitor.restore_runtime_progress(
        running_task_ids=[mrp["task_ids"][0], lrp["task_ids"][-1]],
    )
    old_monitor.process_event(
        event_type="done",
        task_id=lrp["task_ids"][-1],
        function_name="move_home",
        resource_jid=lrp["resource_jid"],
        status="completed",
    )
    assert "xarm6@localhost=(k=1,idle)" in str(old_monitor.current_state)

    for node in planner.nodes:
        if node["id"] == lrp["task_ids"][-1]:
            node["status"] = "completed"
    planner.mark_product_order_part_completed("LRP")
    planner.commit_product_order_part("LCP", status="pending")
    new_fsa = planner.recompile_committed_product_order_fsa()

    new_monitor = OnlineFsaMonitor(new_fsa)
    assert new_monitor.has_state(old_monitor.current_state)
    new_monitor.restore_runtime_progress(running_task_ids=[mrp["task_ids"][0]])

    assert "ur5e@localhost=(k=0,run=REQ_2_T1:pick_approach)" in str(
        new_monitor.current_state
    )
    assert "xarm6@localhost=(k=0,idle)" in str(new_monitor.current_state)
    assert "xarm6@localhost=(k=1,idle)" not in str(new_monitor.current_state)


def test_dashboard_current_task_dag_nodes_show_only_active_product_order_window():
    planner = _runtime_planner(_order(parts=["LG", "MCP"]))
    lg = planner.commit_product_order_part("LG", status="pending")
    for node in planner.nodes:
        if node["id"] in lg["task_ids"]:
            node["status"] = "completed"
    planner.mark_product_order_part_completed("LG")

    mcp = planner.commit_product_order_part("MCP", status="pending_validation")
    bridge = SystemBridge()
    bridge.product_agents = [SimpleNamespace(process_planner=planner)]

    nodes = bridge.get_current_task_dag_nodes()
    visible_ids = {node["id"] for node in nodes}
    assert not set(lg["task_ids"]) & visible_ids
    assert set(mcp["task_ids"]) <= visible_ids
    assert {node.get("product_order_part") for node in nodes} == {"MCP"}


def test_gazebo_dual_launch_command_keeps_default_args_without_fast_forward():
    bridge = SystemBridge()

    assert (
        bridge._render_ros2_launch_cmd("gazebo_dual")
        == "ros2 launch xarm_gazebo dual_moveit_gazebo.launch.py"
    )


def test_gazebo_dual_launch_command_adds_fast_forward_args():
    bridge = SystemBridge()

    assert bridge._render_ros2_launch_cmd(
        "gazebo_dual",
        fast_forward_simulation=True,
    ) == (
        "ros2 launch xarm_gazebo dual_moveit_gazebo.launch.py "
        "fast_sim:=true launch_rviz:=false"
    )


def test_dashboard_current_task_dag_nodes_keep_completed_tasks_for_current_part():
    planner = _runtime_planner(_order(parts=["MCP"]))
    mcp = planner.commit_product_order_part("MCP", status="pending")
    first_task_id = mcp["task_ids"][0]
    for node in planner.nodes:
        if node["id"] == first_task_id:
            node["status"] = "completed"
        elif node["id"] == mcp["task_ids"][1]:
            node["status"] = "running"

    bridge = SystemBridge()
    bridge.product_agents = [SimpleNamespace(process_planner=planner)]

    nodes = bridge.get_current_task_dag_nodes()
    by_id = {node["id"]: node for node in nodes}
    assert set(mcp["task_ids"]) <= set(by_id)
    assert by_id[first_task_id]["status"] == "completed"
    assert by_id[mcp["task_ids"][1]]["status"] == "running"


def test_dashboard_current_task_dag_nodes_overlay_live_task_states():
    planner = _runtime_planner(_order(parts=["MCP"]))
    mcp = planner.commit_product_order_part("MCP", status="pending")
    live_task_id = mcp["task_ids"][0]
    bridge = SystemBridge()
    bridge.product_agents = [
        SimpleNamespace(
            process_planner=planner,
            task_states={live_task_id: "accepted"},
        )
    ]

    nodes = bridge.get_current_task_dag_nodes()
    by_id = {node["id"]: node for node in nodes}
    assert by_id[live_task_id]["status"] == "accepted"


def test_task_dag_mermaid_styles_accepted_status():
    graph = nodes_to_mermaid(
        [
            {
                "id": "REQ_1_T1",
                "type": "task",
                "function_name": "pick_approach",
                "status": "accepted",
                "predecessors": [],
            }
        ]
    )

    assert "classDef accepted" in graph
    assert 'REQ_1_T1["REQ_1_T1' in graph
    assert ":::accepted" in graph


def test_dashboard_current_task_dag_nodes_prune_hidden_edges():
    planner = _runtime_planner(
        _order(parts=["LG", "MCP"]),
        safety_text="LG must be placed before MCP",
    )
    lg = planner.commit_product_order_part("LG", status="pending")
    for node in planner.nodes:
        if node["id"] in lg["task_ids"]:
            node["status"] = "completed"
    planner.mark_product_order_part_completed("LG")
    planner.commit_product_order_part("MCP", status="pending_validation")

    bridge = SystemBridge()
    bridge.product_agents = [SimpleNamespace(process_planner=planner)]

    nodes = bridge.get_current_task_dag_nodes()
    visible_ids = {node["id"] for node in nodes}
    assert visible_ids
    for node in nodes:
        assert set(node.get("predecessors") or []) <= visible_ids
        assert set(node.get("successors") or []) <= visible_ids


def test_dashboard_current_task_dag_nodes_empty_when_product_order_window_complete():
    planner = _runtime_planner(_order(parts=["LG"]))
    lg = planner.commit_product_order_part("LG", status="pending")
    for node in planner.nodes:
        if node["id"] in lg["task_ids"]:
            node["status"] = "completed"
    planner.mark_product_order_part_completed("LG")

    bridge = SystemBridge()
    bridge.product_agents = [SimpleNamespace(process_planner=planner)]

    assert bridge.get_current_task_dag_nodes() == []


def test_dashboard_current_task_dag_nodes_fall_back_to_raw_non_product_order_nodes():
    raw_nodes = [
        {
            "id": "REQ_RAW_T1",
            "type": "task",
            "function_name": "pick_approach",
            "status": "completed",
            "predecessors": [],
            "successors": [],
        }
    ]
    bridge = SystemBridge()
    bridge.product_agents = [
        SimpleNamespace(process_planner=SimpleNamespace(nodes=raw_nodes))
    ]

    assert bridge.get_current_task_dag_nodes() == raw_nodes


def test_active_window_safety_history_allows_mcp_after_completed_lg():
    planner = _runtime_planner(
        _order(parts=["LG", "MCP"]),
        safety_text="LG must be placed before MCP",
    )
    lg = planner.commit_product_order_part("LG", status="pending")
    lg_place = next(
        node
        for node in planner.nodes
        if node["id"] in lg["task_ids"] and node["function_name"] == "place_insert"
    )
    for node in planner.nodes:
        if node["id"] in lg["task_ids"]:
            node["status"] = "completed"
    planner.mark_product_order_part_completed("LG")

    mcp = planner.commit_product_order_part("MCP", status="pending_validation")
    fsa = planner.recompile_committed_product_order_fsa()
    assert fsa

    validator = _lg_before_mcp_validator()
    ok, violations = validator.validate_active_window_fsa(
        fsa=fsa,
        plan={"nodes": planner.nodes},
        product_jid="assembly_board-v1@localhost",
        runtime_context={
            "safety_event_history": [
                {
                    "task_id": lg_place["id"],
                    "suffix": "start",
                    "function_name": "place_insert",
                    "part_name": "LG",
                    "resource_jid": lg_place["resource_jid"],
                    "params": dict(lg_place["params"]),
                },
                {
                    "task_id": lg_place["id"],
                    "suffix": "done",
                    "function_name": "place_insert",
                    "part_name": "LG",
                    "resource_jid": lg_place["resource_jid"],
                    "params": dict(lg_place["params"]),
                },
            ],
        },
    )
    assert ok, violations
    assert mcp["task_ids"]


def test_active_window_rejects_mcp_without_completed_lg_history():
    planner = _runtime_planner(_order(parts=["MCP"]))
    planner.commit_product_order_part("MCP", status="pending_validation")
    fsa = planner.recompile_committed_product_order_fsa()
    assert fsa

    validator = _lg_before_mcp_validator()
    ok, violations = validator.validate_active_window_fsa(
        fsa=fsa,
        plan={"nodes": planner.nodes},
        product_jid="assembly_board-v1@localhost",
        runtime_context={"safety_event_history": []},
    )
    assert not ok
    assert violations
    assert violations[0]["violated_rule_id"] == "SAFE_LG_BEFORE_MCP"
    assert violations[0]["validation_scope"] == "active_window"


def test_safety_ordering_constraint_blocks_mcp_place_until_lg_place_done():
    planner = _planner(
        _order(parts=["LG", "MCP"]),
        safety_text="LG must be placed before MCP",
    )
    lg_place = next(
        node
        for node in planner.nodes
        if node.get("function_name") == "place_insert"
        and node.get("params", {}).get("part_name") == "LG"
    )
    mcp_place = next(
        node
        for node in planner.nodes
        if node.get("function_name") == "place_insert"
        and node.get("params", {}).get("part_name") == "MCP"
    )
    assert lg_place["id"] in mcp_place["predecessors"]

    simulation = SystemBridge._simulate_order_dry_run(planner.nodes)
    blocked_rows = [
        row
        for step in simulation["simulated_event_trace"]
        for row in step.get("blocked_operations", [])
    ]
    assert any(
        row["task_id"] == mcp_place["id"]
        and lg_place["id"] in row.get("missing_predecessors", [])
        for row in blocked_rows
    )
    assert simulation["monitor_state"]["status"] == "completed"
