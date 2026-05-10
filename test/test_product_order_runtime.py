from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

from cais_spade_llm.agents.central_controller.plan_safety_validator import PlanSafetyValidator
from cais_spade_llm.agents.intelligent_product.process_planner import ProcessPlanner
from cais_spade_llm.product.order import validate_product_order
from cais_spade_llm.ui.bridge import SystemBridge


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
                "reachability": ["prusa-mk4-2", "prusa-mk3", "assembly_board-v1"],
                "staging_areas": {"prusa-mk3": {}, "prusa-mk4-2": {}},
            },
        ),
        SimpleNamespace(
            jid="xarm6@localhost",
            static_capabilities={
                "reachability": ["prusa-mk4-1", "prusa-mk3", "assembly_board-v1"],
                "staging_areas": {"prusa-mk3": {}, "prusa-mk4-1": {}},
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
    assert any(candidate["status"] == "rejected" for candidate in product_bidding["candidates"])


def test_product_order_bidding_distributes_multiple_parts_by_bid_load():
    planner = _planner(_order(parts=["SG", "MG", "LG", "SRP"]))
    system_plan = planner.last_product_order_artifact["system_plan"]
    resource_counts = Counter(row["resource_jid"] for row in system_plan)
    assert resource_counts == Counter({"ur5e@localhost": 2, "xarm6@localhost": 2})


def test_product_order_bidding_source_selection_uses_bid_score_not_suffix_preference():
    resources = [
        SimpleNamespace(
            jid="ur5e@localhost",
            static_capabilities={
                "reachability": ["prusa-mk4-1", "prusa-mk4-2", "assembly_board-v1"],
                "staging_areas": {"prusa-mk4-2": {}, "prusa-mk4-1": {}},
            },
        )
    ]
    planner = _planner(_order(parts=["LG"]), resources=resources)
    system_plan = planner.last_product_order_artifact["system_plan"]
    assert system_plan[0]["resource_jid"] == "ur5e@localhost"
    assert system_plan[0]["source_location"] == "prusa-mk4-1"


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


def test_product_order_runtime_busy_resource_is_excluded_from_bidding():
    planner = _runtime_planner(_order(parts=["LG"]))
    record = planner.commit_product_order_part(
        "LG",
        unavailable_resource_jids={"ur5e@localhost"},
    )
    assert record["resource_jid"] == "xarm6@localhost"


def test_product_order_runtime_availability_change_before_dispatch_rolls_back_and_rebids():
    planner = _runtime_planner(_order(parts=["LG"]))
    first = planner.commit_product_order_part("LG", status="pending")
    first_resource = first["resource_jid"]

    removed = planner.rollback_product_order_committed_parts(["LG"])
    assert removed == ["LG"]
    assert planner.nodes == []
    assert planner.ready_product_order_parts() == ["LG"]

    second = planner.commit_product_order_part(
        "LG",
        unavailable_resource_jids={first_resource},
        status="pending_validation",
    )
    assert second["resource_jid"] != first_resource
    assert all(node["status"] == "pending_validation" for node in planner.nodes)


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
