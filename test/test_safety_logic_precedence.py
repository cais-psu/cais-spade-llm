from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cais_spade_llm.agents.central_controller.plan_safety_validator import PlanSafetyValidator
from cais_spade_llm.agents.central_controller.safety_logic import SafetyLogic
from cais_spade_llm.experiments.offline_study import OfflineStudyRunner


class _DummyController:
    def __init__(self, llm_payload: dict) -> None:
        self.logger = logging.getLogger("test_safety_logic_precedence")
        self.tools_catalog = [
            {
                "function_owner_agent": "xarm6",
                "function": "place_approach",
                "process": "assembly",
            },
            {
                "function_owner_agent": "ur5e",
                "function": "place_approach",
                "process": "assembly",
            },
        ]
        self._llm_payload = llm_payload

    async def ask_llm(self, **_: object) -> str:
        return json.dumps(self._llm_payload)


def _safe_1_rule() -> dict:
    return {
        "id": "SAFE_1",
        "constraint_type": "precedence",
        "raw_text": "LG by xarm6 must place_approach first before MCP by ur5e to the assembly station.",
        "process": "assembly",
        "event": "place_approach",
        "product": ["LG", "MCP"],
        "resources": ["xarm6", "ur5e"],
        "context": {"location": "assembly_station"},
    }


def _expected_aps() -> tuple[str, str]:
    earlier = "ap_event/assembly/lg/xarm6/place_approach/location=assembly_station"
    later = "ap_event/assembly/mcp/ur5e/place_approach/location=assembly_station"
    return earlier, later


def _safe_1_rule_resourceless_parts() -> dict:
    return {
        "id": "SAFE_1",
        "constraint_type": "precedence",
        "raw_text": "SG must place_approach first before MRP place_approach to the assembly station.",
        "process": "assembly",
        "event": "place_approach",
        "product": ["sg", "mrp"],
        "resources": [],
        "context": {"location": "assembly_station"},
    }


def _expected_resourceless_part_aps() -> tuple[str, str]:
    earlier = "ap_event/assembly/sg/any/place_approach/location=assembly_station"
    later = "ap_event/assembly/mrp/any/place_approach/location=assembly_station"
    return earlier, later


def test_llm_build_safety_logic_prefers_deterministic_precedence_formula() -> None:
    earlier_raw = "ap/assembly/LG/xarm6/place_approach/location=assembly_station"
    later_raw = "ap/assembly/MCP/ur5e/place_approach/location=assembly_station"
    controller = _DummyController(
        {
            "rules": [
                {
                    "id": "SAFE_1",
                    "aps": [earlier_raw, later_raw],
                    "ltlf": f"G (({later_raw} -> {earlier_raw}))",
                }
            ]
        }
    )
    logic = SafetyLogic(
        controller,
        ROOT / "cais_spade_llm" / "specification" / "safety" / "safety_case_llm_bridge.txt",
    )
    logic.rules = [_safe_1_rule()]

    compiled = asyncio.run(logic._llm_build_safety_logic())

    earlier, later = _expected_aps()
    assert compiled["SAFE_1"]["aps"] == [earlier, later]
    assert compiled["SAFE_1"]["ltlf"] == f"((!{later}) U {earlier})"


def test_formula_ast_precedence_is_compiled_as_ordering_not_implication() -> None:
    controller = _DummyController({"rules": []})
    logic = SafetyLogic(
        controller,
        ROOT / "cais_spade_llm" / "specification" / "safety" / "safety_case_llm_bridge.txt",
    )
    rule = _safe_1_rule()
    formula_ast = {
        "op": "->",
        "left": {
            "type": "ap_event_atom",
            "function": "place_approach",
            "product": "MCP",
            "resource": "ur5e",
            "context": {"location": "assembly_station"},
        },
        "right": {
            "type": "ap_event_atom",
            "function": "place_approach",
            "product": "LG",
            "resource": "xarm6",
            "context": {"location": "assembly_station"},
        },
    }

    compiled = logic._compile_formula_ast_for_rule(rule, formula_ast)

    earlier, later = _expected_aps()
    assert compiled["aps"] == [later, earlier]
    assert compiled["ltlf"] == f"((!{later}) U {earlier})"


def test_formula_ast_resourceless_parts_compile_with_per_part_product_slot() -> None:
    """Multi-part precedence rule with no concrete resources must compile via
    atom-level ``product`` discrimination (resource="any"), not via
    ``resource_var``. Mirrors the s1_rpc_232_safety.txt failure where SG/MRP/LCP
    are product parts, not robots."""
    controller = _DummyController({"rules": []})
    logic = SafetyLogic(
        controller,
        ROOT / "cais_spade_llm" / "specification" / "safety" / "safety_case_llm_bridge.txt",
    )
    rule = _safe_1_rule_resourceless_parts()
    formula_ast = {
        "op": "U",
        "left": {
            "op": "!",
            "arg": {
                "type": "ap_event_atom",
                "function": "place_approach",
                "product": "mrp",
                "resource": "any",
                "context": {"location": "assembly_station"},
            },
        },
        "right": {
            "type": "ap_event_atom",
            "function": "place_approach",
            "product": "sg",
            "resource": "any",
            "context": {"location": "assembly_station"},
        },
    }

    compiled = logic._compile_formula_ast_for_rule(rule, formula_ast)

    earlier, later = _expected_resourceless_part_aps()
    assert compiled["aps"] == [later, earlier]
    assert compiled["ltlf"] == f"((!{later}) U {earlier})"


def test_precedence_dfa_dot_accepts_ordered_trace_and_rejects_reversed_trace() -> None:
    rule = {
        "id": "SAFE_1",
        "raw_text": "LG by xarm6 must place_approach first before MCP by ur5e to the assembly station.",
        "aps": [
            {
                "label": "ap004",
                "full": "ap_event/assembly/mcp/ur5e/place_approach/destination=assembly_board-v1",
            },
            {
                "label": "ap003",
                "full": "ap_event/assembly/lg/xarm6/place_approach/destination=assembly_board-v1",
            },
        ],
        "ltlf": "((!ap004) U ap003)",
        "context": {"destination": "assembly_board-v1"},
    }
    dfa_dot = OfflineStudyRunner._precedence_dfa_dot("ap003", "ap004")
    validator = PlanSafetyValidator(
        rules=[rule],
        dfa_map={"SAFE_1": dfa_dot},
        tools_catalog=[
            {"function_owner_agent": "xarm6", "function": "place_approach", "process": "assembly"},
            {"function_owner_agent": "ur5e", "function": "place_approach", "process": "assembly"},
        ],
    )

    ordered_plan = {
        "nodes": [
            {
                "id": "REQ_1_T3",
                "type": "task",
                "resource_jid": "xarm6@localhost",
                "function_name": "place_approach",
                "params": {"destination_location": "assembly_board-v1", "part_name": "LG"},
            },
            {
                "id": "REQ_2_T3",
                "type": "task",
                "resource_jid": "ur5e@localhost",
                "function_name": "place_approach",
                "params": {"destination_location": "assembly_board-v1", "part_name": "MCP"},
            },
        ]
    }
    ordered_fsa = {
        "A": {
            "x0": "x0",
            "Xm": ["x4"],
            "Tr": [
                {
                    "from": "x0",
                    "to": "x1",
                    "event": "REQ_1_T3.start",
                    "task_id": "REQ_1_T3",
                    "resource_jid": "xarm6@localhost",
                    "function_name": "place_approach",
                    "params": {"destination_location": "assembly_board-v1", "part_name": "LG"},
                },
                {
                    "from": "x1",
                    "to": "x2",
                    "event": "REQ_1_T3.done",
                    "task_id": "REQ_1_T3",
                    "resource_jid": "xarm6@localhost",
                    "function_name": "place_approach",
                    "params": {"destination_location": "assembly_board-v1", "part_name": "LG"},
                },
                {
                    "from": "x2",
                    "to": "x3",
                    "event": "REQ_2_T3.start",
                    "task_id": "REQ_2_T3",
                    "resource_jid": "ur5e@localhost",
                    "function_name": "place_approach",
                    "params": {"destination_location": "assembly_board-v1", "part_name": "MCP"},
                },
                {
                    "from": "x3",
                    "to": "x4",
                    "event": "REQ_2_T3.done",
                    "task_id": "REQ_2_T3",
                    "resource_jid": "ur5e@localhost",
                    "function_name": "place_approach",
                    "params": {"destination_location": "assembly_board-v1", "part_name": "MCP"},
                },
            ],
        }
    }
    ok_ordered, violations_ordered = validator.validate_plan_fsa(
        fsa=ordered_fsa,
        plan=ordered_plan,
        product_jid="assembly_board-v1@localhost",
    )
    assert ok_ordered is True
    assert violations_ordered == []

    reversed_fsa = {
        "A": {
            "x0": "x0",
            "Xm": ["x4"],
            "Tr": [
                {
                    "from": "x0",
                    "to": "x1",
                    "event": "REQ_2_T3.start",
                    "task_id": "REQ_2_T3",
                    "resource_jid": "ur5e@localhost",
                    "function_name": "place_approach",
                    "params": {"destination_location": "assembly_board-v1", "part_name": "MCP"},
                },
                {
                    "from": "x1",
                    "to": "x2",
                    "event": "REQ_2_T3.done",
                    "task_id": "REQ_2_T3",
                    "resource_jid": "ur5e@localhost",
                    "function_name": "place_approach",
                    "params": {"destination_location": "assembly_board-v1", "part_name": "MCP"},
                },
                {
                    "from": "x2",
                    "to": "x3",
                    "event": "REQ_1_T3.start",
                    "task_id": "REQ_1_T3",
                    "resource_jid": "xarm6@localhost",
                    "function_name": "place_approach",
                    "params": {"destination_location": "assembly_board-v1", "part_name": "LG"},
                },
                {
                    "from": "x3",
                    "to": "x4",
                    "event": "REQ_1_T3.done",
                    "task_id": "REQ_1_T3",
                    "resource_jid": "xarm6@localhost",
                    "function_name": "place_approach",
                    "params": {"destination_location": "assembly_board-v1", "part_name": "LG"},
                },
            ],
        }
    }
    ok_reversed, violations_reversed = validator.validate_plan_fsa(
        fsa=reversed_fsa,
        plan=ordered_plan,
        product_jid="assembly_board-v1@localhost",
    )
    assert ok_reversed is False
    assert violations_reversed[0]["violated_rule_id"] == "SAFE_1"
