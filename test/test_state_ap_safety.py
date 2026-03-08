from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("OPENAI_API_KEY", "sk-local-test")

REPO_ROOT = Path(__file__).resolve().parents[1]
PKG_ROOT = REPO_ROOT / "cais_spade_llm"
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from cais_spade_llm.agents.central_controller.offline_safety_validator import (
    OfflineSafetyValidator,
)
from cais_spade_llm.agents.central_controller.online_safety_monitor import (
    OnlineSafetyMonitor,
)


class _FakeController:
    def __init__(self, tools_catalog: list[dict]) -> None:
        self.logger = logging.getLogger("test.state_ap_safety")
        self.tools_catalog = tools_catalog

    async def ask_llm(self, **_: object) -> str:
        raise RuntimeError("ask_llm should not be called in this unit test")

    def _static_caps_overview(self) -> str:
        return "resource: ur5e, xarm6"


def _tools_catalog() -> list[dict]:
    return [
        {
            "function": "place_approach",
            "function_owner_agent": "ur5e",
            "process": "assembly",
            "in_state": "picked",
            "out_state": "positioned",
            "required_context_keys": ["destination"],
            "context_mapping": {"location_param": "destination_location"},
        },
        {
            "function": "place_insert",
            "function_owner_agent": "ur5e",
            "process": "assembly",
            "in_state": "positioned",
            "out_state": "placed",
            "required_context_keys": ["destination"],
            "context_mapping": {"location_param": "destination_location"},
        },
        {
            "function": "move_home",
            "function_owner_agent": "ur5e",
            "process": "assembly",
            "in_state": "any",
            "out_state": "idle",
        },
        {
            "function": "place_approach",
            "function_owner_agent": "xarm6",
            "process": "assembly",
            "in_state": "picked",
            "out_state": "positioned",
            "required_context_keys": ["destination"],
            "context_mapping": {"location_param": "destination_location"},
        },
        {
            "function": "place_insert",
            "function_owner_agent": "xarm6",
            "process": "assembly",
            "in_state": "positioned",
            "out_state": "placed",
            "required_context_keys": ["destination"],
            "context_mapping": {"location_param": "destination_location"},
        },
        {
            "function": "move_home",
            "function_owner_agent": "xarm6",
            "process": "assembly",
            "in_state": "any",
            "out_state": "idle",
        },
    ]


def _board_mutex_rule() -> dict:
    return {
        "id": "SAFE_1",
        "raw_text": "both arms should not be at the assembly board at the same time",
        "process": "assembly",
        "product": [],
        "resources": ["ur5e", "xarm6"],
        "context": {"destination": "assembly_board-v1"},
        "aps": [
            {
                "label": "ap001",
                "full": "ap_event/assembly/any/ur5e/place_approach/destination=assembly_board-v1",
            },
            {
                "label": "ap002",
                "full": "ap_state/assembly/any/ur5e/positioned/destination=assembly_board-v1",
            },
            {
                "label": "ap003",
                "full": "ap_state/assembly/any/ur5e/placed/destination=assembly_board-v1",
            },
            {
                "label": "ap004",
                "full": "ap_event/assembly/any/xarm6/place_approach/destination=assembly_board-v1",
            },
            {
                "label": "ap005",
                "full": "ap_state/assembly/any/xarm6/positioned/destination=assembly_board-v1",
            },
            {
                "label": "ap006",
                "full": "ap_state/assembly/any/xarm6/placed/destination=assembly_board-v1",
            },
        ],
        "ltlf": "G !((ap001 | ap002 | ap003) & (ap004 | ap005 | ap006))",
    }


def _board_mutex_dot() -> str:
    return "\n".join(
        [
            "digraph MONA_DFA {",
            "  init -> 1;",
            '  1 -> 2 [label="((ap001 | ap002 | ap003) & (ap004 | ap005 | ap006))"];',
            '  1 -> 1 [label="true"];',
            '  2 -> 2 [label="true"];',
            "}",
        ]
    )


def test_ap_selector_expands_destination_selector_to_entry_event_and_state_aps(tmp_path: Path) -> None:
    pytest.importorskip("ltlf2dfa")
    from cais_spade_llm.agents.central_controller.safety_logic import SafetyLogic

    logic = SafetyLogic(_FakeController(_tools_catalog()), tmp_path / "safety.txt")
    rule = {
        "id": "SAFE_1",
        "raw_text": "both arms should not be at the assembly board at the same time",
        "process": "assembly",
        "product": [],
        "resources": ["ur5e"],
        "context": {"destination": "assembly_board-v1"},
    }
    selector = {
        "type": "ap_selector",
        "resource": "ur5e",
        "match": {"context": {"destination": "assembly_board-v1"}},
        "include_entry_events": True,
        "include_state_aps": True,
    }

    compiled = logic._compile_formula_ast_for_rule(rule, selector)

    assert compiled["aps"] == [
        "ap_event/assembly/any/ur5e/place_approach/destination=assembly_board-v1",
        "ap_state/assembly/any/ur5e/positioned/destination=assembly_board-v1",
        "ap_state/assembly/any/ur5e/placed/destination=assembly_board-v1",
    ]


def test_resource_var_selector_grounds_only_matching_resource_type(tmp_path: Path) -> None:
    pytest.importorskip("ltlf2dfa")
    from cais_spade_llm.agents.central_controller.safety_logic import SafetyLogic

    tools_catalog = [
        {
            "function": "stage_part",
            "function_owner_agent": "arm_a",
            "process": "transfer",
            "resource_type": "robot",
            "in_state": "idle",
            "out_state": "staged",
            "required_context_keys": ["machine"],
        },
        {
            "function": "start_print",
            "function_owner_agent": "printer_a",
            "process": "printing",
            "resource_type": "printer",
            "in_state": "loaded",
            "out_state": "printing",
            "required_context_keys": ["machine"],
        },
        {
            "function": "finish_print",
            "function_owner_agent": "printer_a",
            "process": "printing",
            "resource_type": "printer",
            "in_state": "printing",
            "out_state": "printed",
            "required_context_keys": ["machine"],
        },
        {
            "function": "start_print",
            "function_owner_agent": "printer_b",
            "process": "printing",
            "resource_type": "printer",
            "in_state": "loaded",
            "out_state": "printing",
            "required_context_keys": ["machine"],
        },
        {
            "function": "finish_print",
            "function_owner_agent": "printer_b",
            "process": "printing",
            "resource_type": "printer",
            "in_state": "printing",
            "out_state": "printed",
            "required_context_keys": ["machine"],
        },
    ]

    logic = SafetyLogic(_FakeController(tools_catalog), tmp_path / "safety.txt")
    rule = {
        "id": "SAFE_2",
        "raw_text": "matched printers should not print in the same machine area",
        "process": "printing",
        "product": [],
        "resources": ["any"],
        "context": {"machine": "printer_cell_1"},
    }
    formula_ast = {
        "op": "G",
        "arg": {
            "type": "ap_selector",
            "resource_var": "$r",
            "match": {
                "resource_type": "printer",
                "process": "printing",
                "context": {"machine": "printer_cell_1"},
            },
            "include_entry_events": True,
            "include_state_aps": True,
        },
    }

    compiled = logic._compile_formula_ast_for_rule(rule, formula_ast)

    assert all("/printer_a/" in ap or "/printer_b/" in ap for ap in compiled["aps"])
    assert all("/arm_a/" not in ap for ap in compiled["aps"])
    assert "printer_a" in compiled["ltlf"]
    assert "printer_b" in compiled["ltlf"]


def test_online_monitor_blocks_predicted_state_overlap_and_releases_after_move_home() -> None:
    rule = _board_mutex_rule()
    monitor = OnlineSafetyMonitor(
        {"SAFE_1": _board_mutex_dot()},
        [rule],
        tools_catalog=_tools_catalog(),
    )

    monitor.process_finish_event(
        {
            "resource_jid": "ur5e@localhost",
            "function_name": "place_approach",
            "params": {"destination_location": "assembly_board-v1"},
            "current_state": "positioned",
        }
    )

    allowed, info = monitor.process_start_event(
        {
            "resource_jid": "xarm6@localhost",
            "function_name": "place_approach",
            "params": {"destination_location": "assembly_board-v1"},
        }
    )
    assert allowed is False
    assert info["violated_rule"] == "SAFE_1"

    monitor.process_finish_event(
        {
            "resource_jid": "ur5e@localhost",
            "function_name": "move_home",
            "params": {},
            "current_state": "idle",
        }
    )

    allowed, info = monitor.process_start_event(
        {
            "resource_jid": "xarm6@localhost",
            "function_name": "place_approach",
            "params": {"destination_location": "assembly_board-v1"},
        }
    )
    assert allowed is True
    assert "SAFE_1" not in info.get("violated_rule_id", "")


def test_offline_validator_blocks_plan_branch_on_predicted_state_overlap() -> None:
    rule = _board_mutex_rule()
    validator = OfflineSafetyValidator(
        rules=[rule],
        dfa_map={"SAFE_1": _board_mutex_dot()},
        tools_catalog=_tools_catalog(),
    )

    fsa = {
        "A": {
            "X": [
                "(ur5e@localhost=(k=0,idle),xarm6@localhost=(k=0,idle))",
                "(ur5e@localhost=(k=0,run=U1:place_approach),xarm6@localhost=(k=0,idle))",
                "(ur5e@localhost=(k=1,idle),xarm6@localhost=(k=0,idle))",
                "(ur5e@localhost=(k=1,idle),xarm6@localhost=(k=0,run=X1:place_approach))",
            ],
            "E": ["U1.start", "U1.done", "X1.start"],
            "Tr": [
                {
                    "from": "(ur5e@localhost=(k=0,idle),xarm6@localhost=(k=0,idle))",
                    "event": "U1.start",
                    "to": "(ur5e@localhost=(k=0,run=U1:place_approach),xarm6@localhost=(k=0,idle))",
                    "task_id": "U1",
                    "resource_jid": "ur5e@localhost",
                    "function_name": "place_approach",
                    "params": {"destination_location": "assembly_board-v1"},
                    "in_state": "picked",
                    "out_state": "positioned",
                },
                {
                    "from": "(ur5e@localhost=(k=0,run=U1:place_approach),xarm6@localhost=(k=0,idle))",
                    "event": "U1.done",
                    "to": "(ur5e@localhost=(k=1,idle),xarm6@localhost=(k=0,idle))",
                    "task_id": "U1",
                    "resource_jid": "ur5e@localhost",
                    "function_name": "place_approach",
                    "params": {"destination_location": "assembly_board-v1"},
                    "in_state": "picked",
                    "out_state": "positioned",
                },
                {
                    "from": "(ur5e@localhost=(k=1,idle),xarm6@localhost=(k=0,idle))",
                    "event": "X1.start",
                    "to": "(ur5e@localhost=(k=1,idle),xarm6@localhost=(k=0,run=X1:place_approach))",
                    "task_id": "X1",
                    "resource_jid": "xarm6@localhost",
                    "function_name": "place_approach",
                    "params": {"destination_location": "assembly_board-v1"},
                    "in_state": "picked",
                    "out_state": "positioned",
                },
            ],
            "x0": "(ur5e@localhost=(k=0,idle),xarm6@localhost=(k=0,idle))",
            "Xm": ["(ur5e@localhost=(k=1,idle),xarm6@localhost=(k=0,run=X1:place_approach))"],
        }
    }

    ok, violations = validator.validate_fsa_offline(fsa=fsa, plan=None, product_jid="assembly_board-v1")

    assert ok is False
    assert violations
    assert violations[0]["violated_rule_id"] == "SAFE_1"
    assert violations[0]["witness_events"] == ["U1.start", "U1.done", "X1.start"]
