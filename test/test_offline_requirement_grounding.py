from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cais_spade_llm.agents.intelligent_product.process_planner import ProcessPlanner
from cais_spade_llm.bundles.bundle_compiler import BundleCompiler


def _load_json(path: Path) -> dict[str, Any] | list[Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_tools_catalog() -> list[dict[str, Any]]:
    payload = _load_json(ROOT / "cais_spade_llm" / "initialization" / "tools.json")
    if not isinstance(payload, list):
        raise TypeError("tools catalogue must be a list")
    return [row for row in payload if isinstance(row, dict)]


def _load_resource(name: str) -> SimpleNamespace:
    path = ROOT / "cais_spade_llm" / "initialization" / "resources" / f"robot_{name}.json"
    if not path.exists():
        path = ROOT / "cais_spade_llm" / "initialization" / "resources" / f"robot_{name.replace('-', '_')}.json"
    payload = _load_json(path)
    if not isinstance(payload, dict):
        raise TypeError(f"resource payload for {name} must be an object")
    meta = payload.get(name) or payload.get(name.replace("_", "-"))
    if not isinstance(meta, dict):
        raise KeyError(f"resource payload missing key {name}")
    gazebo = meta.get("gazebo", {})
    static_capabilities = gazebo.get("static_capabilities", {}) if isinstance(gazebo, dict) else {}
    return SimpleNamespace(
        jid=str(meta.get("jid") or f"{name}@localhost"),
        static_capabilities=static_capabilities if isinstance(static_capabilities, dict) else {},
    )


class FakeProductAgent:
    def __init__(self, *, tools_catalog: list[dict[str, Any]], name: str = "assembly_board-v1") -> None:
        self.jid = f"{name}@localhost"
        self.name = name
        self.tools_catalog = tools_catalog
        self.logger = logging.getLogger(f"test.{name}")
        if not self.logger.handlers:
            self.logger.addHandler(logging.NullHandler())


def _build_planner() -> ProcessPlanner:
    product_agent = FakeProductAgent(tools_catalog=_load_tools_catalog())
    resources = [_load_resource("xarm6"), _load_resource("ur5e")]
    return ProcessPlanner(product_agent, resources)


def test_requirement_grounding_corrects_unique_sources_and_preserves_shared_destination(tmp_path: Path) -> None:
    planner = _build_planner()
    requirements_path = tmp_path / "requirements.json"
    requirements_path.write_text(
        json.dumps(
            {
                "nodes": [
                    {
                        "id": "REQ_1",
                        "type": "requirement",
                        "raw_text": "ur5e assemble MCP from prusa-mk4-2 to the Assembly Station.",
                        "phase": "ASSEMBLY",
                        "process_type": "PICK_PLACE",
                        "product": "MCP",
                        "context": {
                            "origin": "prusa-mk4-2",
                            "destination": "Assembly Station",
                            "resource": "ur5e",
                            "part_name": "MCP",
                        },
                    }
                ]
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    planner.load_requirements(requirements_path)
    planner.nodes = [
        {
            "id": "REQ_1_T1",
            "type": "task",
            "requirement_id": "REQ_1",
            "function_name": "pick_approach",
            "params": {"part_name": "MCP", "product_jid": "assembly_board-v1@localhost"},
            "resource_jid": "xarm6@localhost",
        },
        {
            "id": "REQ_1_T2",
            "type": "task",
            "requirement_id": "REQ_1",
            "function_name": "place_insert",
            "params": {"part_name": "MCP", "product_jid": "assembly_board-v1@localhost"},
            "resource_jid": "ur5e@localhost",
        },
        {
            "id": "REQ_1_T3",
            "type": "task",
            "requirement_id": "REQ_1",
            "function_name": "move_home",
            "params": {"product_jid": "assembly_board-v1@localhost"},
            "resource_jid": "xarm6@localhost",
        },
    ]

    summary = planner.apply_offline_requirement_grounding()

    by_id = {node["id"]: node for node in planner.nodes}
    assert by_id["REQ_1_T1"]["resource_jid"] == "ur5e@localhost"
    assert by_id["REQ_1_T1"]["params"]["origin_resource_location"] == "prusa-mk4-2"
    assert by_id["REQ_1_T2"]["resource_jid"] == "ur5e@localhost"
    assert by_id["REQ_1_T2"]["params"]["destination_location"] == "assembly_board-v1"

    assert summary["corrected_task_count"] == 2
    assert summary["invalid_task_count"] == 0
    assert summary["unresolved_task_count"] == 1

    findings = {row["task_id"]: row for row in summary["findings"]}
    assert findings["REQ_1_T1"]["status"] == "corrected"
    assert findings["REQ_1_T2"]["status"] == "corrected"
    assert findings["REQ_1_T3"]["status"] == "unresolved"


def test_requirement_grounding_uses_workspace_bounds_when_coordinates_are_explicit(tmp_path: Path) -> None:
    planner = _build_planner()
    requirements_path = tmp_path / "requirements.json"
    requirements_path.write_text(
        json.dumps(
            {
                "nodes": [
                    {
                        "id": "REQ_3",
                        "type": "requirement",
                        "raw_text": "assemble SG into the Assembly Station",
                        "phase": "ASSEMBLY",
                        "process_type": "PICK_PLACE",
                        "product": "SG",
                        "context": {
                            "part_name": "SG",
                        },
                    }
                ]
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    planner.load_requirements(requirements_path)
    planner.nodes = [
        {
            "id": "REQ_3_T1",
            "type": "task",
            "requirement_id": "REQ_3",
            "function_name": "place_approach",
            "params": {
                "part_name": "SG",
                "target_pose": {"x": 0.4, "y": 0.8, "z": 1.05},
                "product_jid": "assembly_board-v1@localhost",
            },
            "resource_jid": "xarm6@localhost",
        }
    ]

    summary = planner.apply_offline_requirement_grounding()

    assert planner.nodes[0]["resource_jid"] == "ur5e@localhost"
    assert summary["corrected_task_count"] == 1
    assert summary["invalid_task_count"] == 0
    assert summary["findings"][0]["status"] == "corrected"
    assert summary["findings"][0]["allowed_resource_jids"] == ["ur5e@localhost"]


def test_s2_xarm6_2_can_pick_mg_from_prusa_mk3_2() -> None:
    product_agent = FakeProductAgent(tools_catalog=_load_tools_catalog())
    planner = ProcessPlanner(
        product_agent,
        [_load_resource("xarm6"), _load_resource("ur5e"), _load_resource("xarm6-2")],
    )
    planner.nodes = [
        {
            "id": "REQ_4_T1",
            "type": "task",
            "requirement_id": "REQ_4",
            "function_name": "pick_approach",
            "params": {
                "origin_resource_location": "prusa-mk3-2",
                "part_name": "MG",
                "product_jid": "assembly_board-v1@localhost",
            },
            "resource_jid": "xarm6-2@localhost",
        },
        {
            "id": "REQ_4_T2",
            "type": "task",
            "requirement_id": "REQ_4",
            "function_name": "pick_grasp",
            "params": {
                "origin_resource_location": "prusa-mk3-2",
                "part_name": "MG",
                "product_jid": "assembly_board-v1@localhost",
            },
            "resource_jid": "xarm6-2@localhost",
        },
    ]

    summary = planner.apply_offline_requirement_grounding()

    assert summary["invalid_task_count"] == 0
    findings = {row["task_id"]: row for row in summary["findings"]}
    assert findings["REQ_4_T1"]["allowed_resource_jids"] == ["xarm6-2@localhost"]
    assert findings["REQ_4_T2"]["allowed_resource_jids"] == ["xarm6-2@localhost"]
    assert findings["REQ_4_T1"]["status"] == "unchanged"
    assert findings["REQ_4_T2"]["status"] == "unchanged"


def test_run_offline_repair_loop_stops_on_grounding_invalid_before_validator() -> None:
    class DummyPlanner:
        def __init__(self) -> None:
            self.nodes = [{"id": "REQ_9_T1", "type": "task"}]
            self.global_fsa = {"states": [], "transitions": []}
            self.last_grounding_summary = {
                "corrected_task_count": 0,
                "invalid_task_count": 1,
                "unresolved_task_count": 0,
                "findings": [
                    {
                        "task_id": "REQ_9_T1",
                        "status": "invalid",
                        "reason": "no resource reachability matches grounded location",
                    }
                ],
            }

        def get_last_grounding_summary(self) -> dict[str, Any]:
            return dict(self.last_grounding_summary)

    class DummyValidator:
        def validate_plan_fsa(self, **_: Any) -> tuple[bool, list[dict[str, Any]]]:
            raise AssertionError("validator should not run when grounding is already invalid")

    product_agent = SimpleNamespace(process_planner=DummyPlanner())
    payload = asyncio.run(
        BundleCompiler.run_offline_repair_loop(
            product_agent=product_agent,
            validator=DummyValidator(),
            product_jid="assembly_board-v1@localhost",
            auto_replan_max_attempts=5,
        )
    )

    assert payload["ok"] is False
    assert payload["stop_reason"] == "grounding_invalid"
    assert payload["witness_count"] == 0
    assert payload["grounding_summary"]["invalid_task_count"] == 1
    assert payload["repair_history"][0]["stop_reason"] == "grounding_invalid"
    assert payload["repair_history"][0]["validation_call_index"] == 0
