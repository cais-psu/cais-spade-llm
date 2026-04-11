from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cais_spade_llm.agents.intelligent_product.process_planner import ProcessPlanner


def _noop(*_args: object, **_kwargs: object) -> None:
    return None


def _task(
    task_id: str,
    resource_jid: str,
    sequence_index: int,
    *,
    predecessors: list[str] | None = None,
    requirement_id: str | None = None,
    function_name: str = "place_approach",
    part_name: str | None = None,
) -> dict[str, object]:
    params: dict[str, object] = {}
    if part_name:
        params["part_name"] = part_name
    return {
        "id": task_id,
        "type": "task",
        "requirement_id": requirement_id,
        "resource_jid": resource_jid,
        "function_name": function_name,
        "sequence_index": sequence_index,
        "status": "pending",
        "predecessors": list(predecessors or []),
        "successors": [],
        "params": params,
    }


def _with_successors(nodes: list[dict[str, object]]) -> list[dict[str, object]]:
    prepared: list[dict[str, object]] = []
    for node in nodes:
        cloned = dict(node)
        cloned["predecessors"] = list(node.get("predecessors", []) or [])
        cloned["successors"] = []
        cloned["params"] = dict(node.get("params", {}) or {})
        prepared.append(cloned)

    by_id = {str(node["id"]): node for node in prepared}
    for node in prepared:
        for predecessor_id in node.get("predecessors", []) or []:
            by_id[str(predecessor_id)]["successors"].append(node["id"])
    return prepared


def _make_planner(nodes: list[dict[str, object]]) -> ProcessPlanner:
    logger = SimpleNamespace(
        debug=_noop,
        info=_noop,
        warning=_noop,
        exception=_noop,
    )
    product = SimpleNamespace(logger=logger, tools_catalog=[])
    planner = ProcessPlanner(product_agent=product, resource_agents=[])
    planner.nodes = nodes
    return planner


def _single_resource_start_order(fsa: dict[str, object]) -> list[str]:
    transitions = fsa["A"]["Tr"]
    return [
        str(t["task_id"])
        for t in transitions
        if str(t.get("event", "")).endswith(".start")
    ]


def _resource_task_order(fsa: dict[str, object], resource_jid: str) -> list[str]:
    return list(fsa["meta"]["resource_task_order"][resource_jid])


def test_compile_global_fsa_reorders_direct_same_resource_dependency() -> None:
    nodes = _with_successors(
        [
            _task(
                "TASK_A",
                "xarm6@localhost",
                0,
                predecessors=["TASK_B"],
                part_name="SG",
            ),
            _task("TASK_B", "xarm6@localhost", 1, part_name="LCP"),
        ]
    )

    fsa = _make_planner(nodes).compile_global_fsa()

    assert _single_resource_start_order(fsa) == ["TASK_B", "TASK_A"]
    assert fsa["A"]["Xm"][0] in fsa["A"]["X"]


def test_compile_global_fsa_handles_indirect_shared_resource_deadlock_case() -> None:
    nodes = _with_successors(
        [
            _task(
                "TASK_SG",
                "xarm6@localhost",
                2,
                predecessors=["TASK_LCP"],
                part_name="SG",
            ),
            _task(
                "TASK_MRP",
                "ur5e@localhost",
                2,
                predecessors=["TASK_SG"],
                part_name="MRP",
            ),
            _task("TASK_LCP", "ur5e@localhost", 3, part_name="LCP"),
        ]
    )

    fsa = _make_planner(nodes).compile_global_fsa()

    assert fsa["A"]["Xm"][0] in fsa["A"]["X"]
    assert any(t["task_id"] == "TASK_SG" for t in fsa["A"]["Tr"])
    assert any(t["task_id"] == "TASK_MRP" for t in fsa["A"]["Tr"])


def test_compile_global_fsa_preserves_sequence_tie_break_for_unrelated_tasks() -> None:
    nodes = _with_successors(
        [
            _task("TASK_B", "xarm6@localhost", 0, part_name="MRP"),
            _task("TASK_A", "xarm6@localhost", 0, part_name="LCP"),
            _task("TASK_C", "xarm6@localhost", 1, part_name="SG"),
        ]
    )

    fsa = _make_planner(nodes).compile_global_fsa()

    assert _single_resource_start_order(fsa) == ["TASK_A", "TASK_B", "TASK_C"]
    assert fsa["A"]["Xm"][0] in fsa["A"]["X"]


def test_compile_global_fsa_keeps_same_requirement_tasks_contiguous_per_resource() -> None:
    nodes = _with_successors(
        [
            _task(
                "REQ_1_T1",
                "xarm6@localhost",
                0,
                requirement_id="REQ_1",
                function_name="pick_approach",
            ),
            _task(
                "REQ_1_T2",
                "xarm6@localhost",
                1,
                requirement_id="REQ_1",
                predecessors=["REQ_1_T1"],
                function_name="pick_grasp",
            ),
            _task(
                "REQ_2_T1",
                "xarm6@localhost",
                0,
                requirement_id="REQ_2",
                function_name="pick_approach",
            ),
            _task(
                "REQ_2_T2",
                "xarm6@localhost",
                1,
                requirement_id="REQ_2",
                predecessors=["REQ_2_T1"],
                function_name="pick_grasp",
            ),
        ]
    )

    planner = _make_planner(nodes)
    fsa = planner.compile_global_fsa()

    assert _resource_task_order(fsa, "xarm6@localhost") == [
        "REQ_1_T1",
        "REQ_1_T2",
        "REQ_2_T1",
        "REQ_2_T2",
    ]
    assert planner.next_ready_task()["id"] == "REQ_1_T1"

    by_id = {str(node["id"]): node for node in planner.nodes}
    by_id["REQ_1_T1"]["status"] = "completed"
    assert planner.next_ready_task()["id"] == "REQ_1_T2"


def test_compile_global_fsa_reorders_whole_requirement_block_when_dependency_demands_it() -> None:
    nodes = _with_successors(
        [
            _task(
                "REQ_1_T1",
                "xarm6@localhost",
                0,
                requirement_id="REQ_1",
                function_name="pick_approach",
            ),
            _task(
                "REQ_1_T2",
                "xarm6@localhost",
                1,
                requirement_id="REQ_1",
                predecessors=["REQ_1_T1", "REQ_2_T2"],
                function_name="place_approach",
            ),
            _task(
                "REQ_2_T1",
                "xarm6@localhost",
                0,
                requirement_id="REQ_2",
                function_name="pick_approach",
            ),
            _task(
                "REQ_2_T2",
                "xarm6@localhost",
                1,
                requirement_id="REQ_2",
                predecessors=["REQ_2_T1"],
                function_name="place_approach",
            ),
        ]
    )

    planner = _make_planner(nodes)
    fsa = planner.compile_global_fsa()

    assert _resource_task_order(fsa, "xarm6@localhost") == [
        "REQ_2_T1",
        "REQ_2_T2",
        "REQ_1_T1",
        "REQ_1_T2",
    ]
    assert planner.next_ready_task()["id"] == "REQ_2_T1"


def test_next_ready_task_waits_on_same_resource_block_instead_of_switching_requirements() -> None:
    nodes = _with_successors(
        [
            _task(
                "REQ_1_T1",
                "xarm6@localhost",
                0,
                requirement_id="REQ_1",
                function_name="pick_approach",
            ),
            _task(
                "REQ_1_T2",
                "xarm6@localhost",
                1,
                requirement_id="REQ_1",
                predecessors=["REQ_1_T1", "REQ_3_T1"],
                function_name="place_approach",
            ),
            _task(
                "REQ_2_T1",
                "xarm6@localhost",
                0,
                requirement_id="REQ_2",
                function_name="pick_approach",
            ),
            _task(
                "REQ_3_T1",
                "ur5e@localhost",
                0,
                requirement_id="REQ_3",
                function_name="pick_approach",
            ),
        ]
    )

    planner = _make_planner(nodes)
    planner.compile_global_fsa()
    by_id = {str(node["id"]): node for node in planner.nodes}

    by_id["REQ_1_T1"]["status"] = "completed"
    assert planner.next_ready_task()["id"] == "REQ_3_T1"

    by_id["REQ_3_T1"]["status"] = "completed"
    assert planner.next_ready_task()["id"] == "REQ_1_T2"


def test_compile_global_fsa_keeps_different_resources_parallel_from_x0() -> None:
    nodes = _with_successors(
        [
            _task(
                "REQ_1_T1",
                "xarm6@localhost",
                0,
                requirement_id="REQ_1",
                function_name="pick_approach",
            ),
            _task(
                "REQ_2_T1",
                "ur5e@localhost",
                0,
                requirement_id="REQ_2",
                function_name="pick_approach",
            ),
        ]
    )

    fsa = _make_planner(nodes).compile_global_fsa()
    x0 = fsa["A"]["x0"]
    startable = {
        str(transition["task_id"])
        for transition in fsa["A"]["Tr"]
        if transition["from"] == x0 and str(transition["event"]).endswith(".start")
    }

    assert startable == {"REQ_1_T1", "REQ_2_T1"}


def test_same_resource_block_cycle_error_includes_task_evidence() -> None:
    nodes = _with_successors(
        [
            _task(
                "REQ_2_T1",
                "ur5e@localhost",
                0,
                requirement_id="REQ_2",
                function_name="pick_approach",
            ),
            _task(
                "REQ_2_T2",
                "ur5e@localhost",
                1,
                requirement_id="REQ_2",
                predecessors=["REQ_2_T1", "REQ_3_T1"],
                function_name="place_approach",
            ),
            _task(
                "REQ_3_T1",
                "ur5e@localhost",
                0,
                requirement_id="REQ_3",
                function_name="pick_approach",
            ),
            _task(
                "REQ_3_T2",
                "ur5e@localhost",
                1,
                requirement_id="REQ_3",
                predecessors=["REQ_3_T1", "REQ_2_T1"],
                function_name="place_approach",
            ),
        ]
    )

    with pytest.raises(ValueError) as excinfo:
        _make_planner(nodes).compile_global_fsa()

    message = str(excinfo.value)
    assert "ur5e@localhost" in message
    assert "REQ_2" in message
    assert "REQ_3" in message
    assert "REQ_2 -> REQ_3 via REQ_2_T1 -> REQ_3_T2" in message
    assert "REQ_3 -> REQ_2 via REQ_3_T1 -> REQ_2_T2" in message
