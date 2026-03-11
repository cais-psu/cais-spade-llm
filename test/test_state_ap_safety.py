from __future__ import annotations

import asyncio
import json
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

from cais_spade_llm.agents.central_controller.plan_safety_validator import (
    PlanSafetyValidator,
)
from cais_spade_llm.agents.central_controller.online_fsa_monitor import OnlineFsaMonitor
from cais_spade_llm.agents.central_controller.online_safety_monitor import (
    OnlineSafetyMonitor,
)
from cais_spade_llm.agents.central_controller.online_safety_supervisor import (
    OnlineSafetySupervisor,
)


class _FakeController:
    def __init__(self, tools_catalog: list[dict], responses: list[dict] | None = None) -> None:
        self.logger = logging.getLogger("test.state_ap_safety")
        self.tools_catalog = tools_catalog
        self._responses = [json.dumps(r) for r in (responses or [])]

    async def ask_llm(self, **_: object) -> str:
        if not self._responses:
            raise RuntimeError("no mock response left")
        return self._responses.pop(0)

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


def _pick_and_place_tools_catalog() -> list[dict]:
    return [
        {
            "function": "pick_approach",
            "function_owner_agent": "ur5e",
            "process": "assembly",
            "resource_type": "robot",
            "in_state": "idle",
            "out_state": "at_pick",
            "required_context_keys": ["origin"],
            "context_mapping": {
                "location_param": "origin_resource_location",
                "location_type": "part_location",
            },
        },
        {
            "function": "pick_grasp",
            "function_owner_agent": "ur5e",
            "process": "assembly",
            "resource_type": "robot",
            "in_state": "at_pick",
            "out_state": "picked",
            "required_context_keys": ["origin"],
            "context_mapping": {
                "location_param": "origin_resource_location",
                "location_type": "current_location",
            },
        },
        {
            "function": "place_approach",
            "function_owner_agent": "ur5e",
            "process": "assembly",
            "resource_type": "robot",
            "in_state": "picked",
            "out_state": "positioned",
            "required_context_keys": ["destination"],
            "context_mapping": {
                "location_param": "destination_location",
                "location_type": "reachable_location",
            },
        },
        {
            "function": "place_insert",
            "function_owner_agent": "ur5e",
            "process": "assembly",
            "resource_type": "robot",
            "in_state": "positioned",
            "out_state": "placed",
            "required_context_keys": ["destination"],
            "context_mapping": {
                "location_param": "destination_location",
                "location_type": "current_location",
            },
        },
    ]


def _load_case3_llm_bridge_bundle() -> tuple[list[dict], dict[str, str], dict, dict, list[dict]]:
    bundle_root = (
        REPO_ROOT
        / "cais_spade_llm"
        / "user_verified_plan"
        / "bundles"
        / "case3_llm_bridge"
    )
    safety_logic = json.loads((bundle_root / "safety" / "cca_safety_logic.json").read_text())
    plan = json.loads((bundle_root / "plan" / "twopart_assembly_llm_bridge_plan.json").read_text())
    fsa = json.loads(
        (bundle_root / "plan" / "twopart_assembly_llm_bridge_global_fsa.json").read_text()
    )
    tools_catalog = json.loads((bundle_root / "catalog" / "tools.json").read_text())
    dfa_map = {
        str(rule["id"]): (bundle_root / "safety" / f"{rule['id']}_dfa.dot").read_text()
        for rule in safety_logic["rules"]
        if rule.get("id")
    }
    return safety_logic["rules"], dfa_map, plan, fsa, tools_catalog


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
            "  node [shape = doublecircle]; 1;",
            "  node [shape = circle]; 2;",
            "  init -> 1;",
            '  1 -> 2 [label="((ap001 | ap002 | ap003) & (ap004 | ap005 | ap006))"];',
            '  1 -> 1 [label="true"];',
            '  2 -> 2 [label="true"];',
            "}",
        ]
    )


def _return_home_rule() -> dict:
    return {
        "id": "SAFE_R",
        "raw_text": "resource should move home after place_insert",
        "process": "assembly",
        "product": [],
        "resources": ["ur5e"],
        "context": None,
        "aps": [
            {
                "label": "ap001",
                "full": "ap_event/assembly/any/ur5e/move_home/any",
            },
            {
                "label": "ap002",
                "full": "ap_event/assembly/any/ur5e/place_insert/any",
            },
        ],
        "ltlf": "G (ap002 -> F ap001)",
    }


def _return_home_dot() -> str:
    return "\n".join(
        [
            "digraph MONA_DFA {",
            "  node [shape = doublecircle]; 1;",
            "  node [shape = circle]; 2;",
            "  init -> 1;",
            '  1 -> 2 [label="ap002 & ~ap001"];',
            '  1 -> 1 [label="ap001 | ~ap002"];',
            '  2 -> 1 [label="ap001"];',
            '  2 -> 2 [label="~ap001"];',
            "}",
        ]
    )


def _response_plan_fsa_with_move_home() -> dict:
    return {
        "A": {
            "X": [
                "(ur5e@localhost=(k=0,idle))",
                "(ur5e@localhost=(k=1,idle))",
                "(ur5e@localhost=(k=1,run=H1:move_home))",
                "(ur5e@localhost=(k=2,idle))",
            ],
            "E": ["I1.done", "H1.start", "H1.done"],
            "Tr": [
                {
                    "from": "(ur5e@localhost=(k=0,idle))",
                    "event": "I1.done",
                    "to": "(ur5e@localhost=(k=1,idle))",
                    "task_id": "I1",
                    "resource_jid": "ur5e@localhost",
                    "function_name": "place_insert",
                    "params": {},
                    "in_state": "positioned",
                    "out_state": "placed",
                },
                {
                    "from": "(ur5e@localhost=(k=1,idle))",
                    "event": "H1.start",
                    "to": "(ur5e@localhost=(k=1,run=H1:move_home))",
                    "task_id": "H1",
                    "resource_jid": "ur5e@localhost",
                    "function_name": "move_home",
                    "params": {},
                    "in_state": "placed",
                    "out_state": "idle",
                },
                {
                    "from": "(ur5e@localhost=(k=1,run=H1:move_home))",
                    "event": "H1.done",
                    "to": "(ur5e@localhost=(k=2,idle))",
                    "task_id": "H1",
                    "resource_jid": "ur5e@localhost",
                    "function_name": "move_home",
                    "params": {},
                    "in_state": "placed",
                    "out_state": "idle",
                },
            ],
            "x0": "(ur5e@localhost=(k=0,idle))",
            "Xm": ["(ur5e@localhost=(k=2,idle))"],
        }
    }


def _response_plan_fsa_missing_move_home() -> dict:
    return {
        "A": {
            "X": ["(ur5e@localhost=(k=0,idle))", "(ur5e@localhost=(k=1,idle))"],
            "E": ["I1.done"],
            "Tr": [
                {
                    "from": "(ur5e@localhost=(k=0,idle))",
                    "event": "I1.done",
                    "to": "(ur5e@localhost=(k=1,idle))",
                    "task_id": "I1",
                    "resource_jid": "ur5e@localhost",
                    "function_name": "place_insert",
                    "params": {},
                    "in_state": "positioned",
                    "out_state": "placed",
                },
            ],
            "x0": "(ur5e@localhost=(k=0,idle))",
            "Xm": ["(ur5e@localhost=(k=1,idle))"],
        }
    }


def _mutex_waiting_plan_fsa() -> dict:
    return {
        "A": {
            "X": [
                "(ur5e@localhost=(k=0,idle),xarm6@localhost=(k=0,idle))",
                "(ur5e@localhost=(k=1,idle),xarm6@localhost=(k=0,idle))",
                "(ur5e@localhost=(k=1,run=H1:move_home),xarm6@localhost=(k=0,idle))",
                "(ur5e@localhost=(k=2,idle),xarm6@localhost=(k=0,idle))",
                "(ur5e@localhost=(k=1,idle),xarm6@localhost=(k=0,run=X1:place_approach))",
            ],
            "E": ["U1.done", "H1.start", "H1.done", "X1.start"],
            "Tr": [
                {
                    "from": "(ur5e@localhost=(k=0,idle),xarm6@localhost=(k=0,idle))",
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
                    "event": "H1.start",
                    "to": "(ur5e@localhost=(k=1,run=H1:move_home),xarm6@localhost=(k=0,idle))",
                    "task_id": "H1",
                    "resource_jid": "ur5e@localhost",
                    "function_name": "move_home",
                    "params": {},
                    "in_state": "positioned",
                    "out_state": "idle",
                },
                {
                    "from": "(ur5e@localhost=(k=1,run=H1:move_home),xarm6@localhost=(k=0,idle))",
                    "event": "H1.done",
                    "to": "(ur5e@localhost=(k=2,idle),xarm6@localhost=(k=0,idle))",
                    "task_id": "H1",
                    "resource_jid": "ur5e@localhost",
                    "function_name": "move_home",
                    "params": {},
                    "in_state": "positioned",
                    "out_state": "idle",
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
            "Xm": ["(ur5e@localhost=(k=2,idle),xarm6@localhost=(k=0,idle))"],
        }
    }


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


def test_formula_ast_compile_failure_logs_rule_and_selector_details(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    pytest.importorskip("ltlf2dfa")
    from cais_spade_llm.agents.central_controller.safety_logic import SafetyLogic

    parse_payload = {
        "rules": [
            {
                "id": "SAFE_1",
                "raw_text": "both arms should not enter into assembly board to place their parts at the same time.",
                "constraint_type": "no_concurrent_place_in_assembly_board",
                "process": "assembly",
                "product": [],
                "resources": ["ur5e"],
                "resource_types": None,
                "event": "place_approach",
                "context": {"destination": "assembly_board-v1"},
            }
        ]
    }
    logic_payload = {
        "rules": [
            {
                "id": "SAFE_1",
                "formula_ast": {
                    "type": "ap_selector",
                    "resource": "ur5e",
                    "match": {
                        "process": "printing",
                        "context": {"destination": "assembly_board-v1"},
                    },
                    "include_entry_events": True,
                    "include_state_aps": True,
                },
            }
        ]
    }

    logic = SafetyLogic(
        _FakeController(_tools_catalog(), responses=[parse_payload, logic_payload]),
        tmp_path / "safety.txt",
    )

    with caplog.at_level(logging.ERROR, logger="test.state_ap_safety"):
        with pytest.raises(
            RuntimeError,
            match="did not match any tool rows before context grounding",
        ) as excinfo:
            asyncio.run(logic.build_safety_rules_and_logic("dummy text"))

    message = str(excinfo.value)
    assert "selector=" in message
    assert '"process": "printing"' in message
    assert "before context grounding" in message
    assert "formula_ast=" in caplog.text
    assert '"process": "printing"' in caplog.text
    assert '"id": "SAFE_1"' in caplog.text


def test_ap_selector_repairs_generic_location_to_unique_canonical_context_role(
    tmp_path: Path,
) -> None:
    pytest.importorskip("ltlf2dfa")
    from cais_spade_llm.agents.central_controller.safety_logic import SafetyLogic

    logic = SafetyLogic(_FakeController(_tools_catalog()), tmp_path / "safety.txt")
    rule = {
        "id": "SAFE_CTX_1",
        "raw_text": "resource should not enter the same workspace slice twice",
        "process": "assembly",
        "product": [],
        "resources": ["ur5e"],
        "context": {"destination": "assembly_board-v1"},
    }
    selector = {
        "type": "ap_selector",
        "resource": "ur5e",
        "match": {
            "functions": ["place_approach", "place_insert"],
            "context": {"location": "assembly_board-v1"},
        },
        "include_entry_events": True,
        "include_state_aps": True,
    }

    compiled = logic._compile_formula_ast_for_rule(rule, selector)

    assert compiled["aps"] == [
        "ap_event/assembly/any/ur5e/place_approach/destination=assembly_board-v1",
        "ap_state/assembly/any/ur5e/positioned/destination=assembly_board-v1",
        "ap_state/assembly/any/ur5e/placed/destination=assembly_board-v1",
    ]


def test_ap_selector_rejects_ambiguous_generic_context_role(tmp_path: Path) -> None:
    pytest.importorskip("ltlf2dfa")
    from cais_spade_llm.agents.central_controller.safety_logic import SafetyLogic

    logic = SafetyLogic(
        _FakeController(_pick_and_place_tools_catalog()),
        tmp_path / "safety.txt",
    )
    rule = {
        "id": "SAFE_CTX_2",
        "raw_text": "robot should not be in the same location slice twice",
        "process": "assembly",
        "product": [],
        "resources": ["ur5e"],
        "context": None,
    }
    selector = {
        "type": "ap_selector",
        "resource": "ur5e",
        "match": {"context": {"location": "assembly_board-v1"}},
        "include_entry_events": True,
        "include_state_aps": True,
    }

    with pytest.raises(RuntimeError, match="unresolved context keys"):
        logic._compile_formula_ast_for_rule(rule, selector)


def test_ap_selector_rejects_mixed_context_families_in_same_branch(tmp_path: Path) -> None:
    pytest.importorskip("ltlf2dfa")
    from cais_spade_llm.agents.central_controller.safety_logic import SafetyLogic

    logic = SafetyLogic(
        _FakeController(_pick_and_place_tools_catalog()),
        tmp_path / "safety.txt",
    )
    rule = {
        "id": "SAFE_CTX_3",
        "raw_text": "robot should not straddle incompatible work slices",
        "process": "assembly",
        "product": [],
        "resources": ["ur5e"],
        "context": None,
    }
    selector = {
        "type": "ap_selector",
        "resource": "ur5e",
        "match": {
            "functions": ["pick_approach", "place_approach"],
            "states": ["at_pick", "positioned"],
        },
        "include_entry_events": True,
        "include_state_aps": True,
    }

    with pytest.raises(RuntimeError, match="incompatible tool families"):
        logic._compile_formula_ast_for_rule(rule, selector)


def test_formula_ast_rejects_multiple_distinct_resource_vars(tmp_path: Path) -> None:
    pytest.importorskip("ltlf2dfa")
    from cais_spade_llm.agents.central_controller.safety_logic import SafetyLogic

    logic = SafetyLogic(_FakeController(_tools_catalog()), tmp_path / "safety.txt")
    rule = {
        "id": "SAFE_RV_1",
        "raw_text": "both arms should not be at the same station at the same time",
        "process": "assembly",
        "product": [],
        "resources": ["ur5e", "xarm6"],
        "context": {"destination": "assembly_board-v1"},
    }
    formula_ast = {
        "op": "G",
        "arg": {
            "op": "!",
            "arg": {
                "op": "&",
                "args": [
                    {
                        "type": "ap_selector",
                        "resource_var": "$r1",
                        "match": {
                            "functions": ["place_approach", "place_insert"],
                            "context": {"destination": "assembly_board-v1"},
                        },
                        "include_entry_events": True,
                        "include_state_aps": True,
                    },
                    {
                        "type": "ap_selector",
                        "resource_var": "$r2",
                        "match": {
                            "functions": ["place_approach", "place_insert"],
                            "context": {"destination": "assembly_board-v1"},
                        },
                        "include_entry_events": True,
                        "include_state_aps": True,
                    },
                ],
            },
        },
    }

    with pytest.raises(RuntimeError, match="multiple distinct resource_var"):
        logic._compile_formula_ast_for_rule(rule, formula_ast)


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


def test_parsed_resource_types_survive_and_constrain_resource_var_grounding(tmp_path: Path) -> None:
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

    parse_payload = {
        "rules": [
            {
                "id": "SAFE_3",
                "raw_text": "printers should not be in the same machine area while printing",
                "constraint_type": "printer_mutex",
                "process": "printing",
                "product": [],
                "resources": ["any"],
                "resource_types": ["printer"],
                "event": None,
                "context": {"machine": "printer_cell_1"},
            }
        ]
    }
    logic_payload = {
        "rules": [
            {
                "id": "SAFE_3",
                "formula_ast": {
                    "op": "G",
                    "arg": {
                        "op": "!",
                        "arg": {
                            "op": "&",
                            "args": [
                                {
                                    "type": "ap_selector",
                                    "resource": "printer_a",
                                    "match": {
                                        "resource_type": "printer",
                                        "process": "printing",
                                        "context": {"machine": "printer_cell_1"},
                                    },
                                    "include_entry_events": True,
                                    "include_state_aps": True,
                                },
                                {
                                    "type": "ap_selector",
                                    "resource": "printer_b",
                                    "match": {
                                        "resource_type": "printer",
                                        "process": "printing",
                                        "context": {"machine": "printer_cell_1"},
                                    },
                                    "include_entry_events": True,
                                    "include_state_aps": True,
                                },
                            ],
                        },
                    },
                },
            }
        ]
    }

    logic = SafetyLogic(
        _FakeController(tools_catalog, responses=[parse_payload, logic_payload]),
        tmp_path / "safety.txt",
    )
    asyncio.run(logic.build_safety_rules_and_logic("dummy text"))

    rule = logic.rules[0]
    full_aps = [str(ap.get("full", "")) for ap in rule.get("aps", [])]

    assert rule.get("resource_types") == ["printer"]
    assert all("/printer_a/" in ap or "/printer_b/" in ap for ap in full_aps)
    assert all("/arm_a/" not in ap for ap in full_aps)


def test_build_safety_rules_rejects_degenerate_single_resource_mutex_conjuncts(
    tmp_path: Path,
) -> None:
    pytest.importorskip("ltlf2dfa")
    from cais_spade_llm.agents.central_controller.safety_logic import SafetyLogic

    parse_payload = {
        "rules": [
            {
                "id": "SAFE_4",
                "raw_text": "both arms should not enter into assembly board to place their parts at the same time.",
                "constraint_type": "no_concurrent_place_in_assembly_board",
                "process": "assembly",
                "product": [],
                "resources": ["ur5e", "xarm6"],
                "resource_types": ["robot"],
                "event": "place_approach",
                "context": {"destination": "assembly_board-v1"},
            }
        ]
    }
    logic_payload = {
        "rules": [
            {
                "id": "SAFE_4",
                "formula_ast": {
                    "op": "G",
                    "arg": {
                        "op": "!",
                        "arg": {
                            "op": "&",
                            "args": [
                                {
                                    "type": "ap_selector",
                                    "resource_var": "$r",
                                    "match": {
                                        "functions": ["place_approach", "place_insert"],
                                        "context": {"destination": "assembly_board-v1"},
                                    },
                                    "include_entry_events": True,
                                    "include_state_aps": True,
                                },
                                {
                                    "type": "ap_selector",
                                    "resource_var": "$r",
                                    "match": {
                                        "functions": ["place_approach", "place_insert"],
                                        "context": {"destination": "assembly_board-v1"},
                                    },
                                    "include_entry_events": True,
                                    "include_state_aps": True,
                                },
                            ],
                        },
                    },
                },
            }
        ]
    }

    logic = SafetyLogic(
        _FakeController(_tools_catalog(), responses=[parse_payload, logic_payload]),
        tmp_path / "safety.txt",
    )

    with pytest.raises(RuntimeError, match="degenerates into independent single-resource conjuncts"):
        asyncio.run(logic.build_safety_rules_and_logic("dummy text"))


def test_build_safety_rules_keeps_cross_resource_mutex_as_one_rule(tmp_path: Path) -> None:
    pytest.importorskip("ltlf2dfa")
    from cais_spade_llm.agents.central_controller.safety_logic import SafetyLogic

    parse_payload = {
        "rules": [
            {
                "id": "SAFE_5",
                "raw_text": "both arms should not be inside the assembly board station at the same time.",
                "constraint_type": "no_simultaneous_presence_in_station",
                "process": "assembly",
                "product": [],
                "resources": ["ur5e", "xarm6"],
                "resource_types": ["robot"],
                "event": None,
                "context": {"destination": "assembly_board-v1"},
            }
        ]
    }
    logic_payload = {
        "rules": [
            {
                "id": "SAFE_5",
                "formula_ast": {
                    "op": "G",
                    "arg": {
                        "op": "!",
                        "arg": {
                            "op": "&",
                            "args": [
                                {
                                    "type": "ap_selector",
                                    "resource": "ur5e",
                                    "match": {
                                        "functions": ["place_approach", "place_insert"],
                                        "context": {"location": "assembly_board-v1"},
                                    },
                                    "include_entry_events": True,
                                    "include_state_aps": True,
                                },
                                {
                                    "type": "ap_selector",
                                    "resource": "xarm6",
                                    "match": {
                                        "functions": ["place_approach", "place_insert"],
                                        "context": {"location": "assembly_board-v1"},
                                    },
                                    "include_entry_events": True,
                                    "include_state_aps": True,
                                },
                            ],
                        },
                    },
                },
            }
        ]
    }

    logic = SafetyLogic(
        _FakeController(_tools_catalog(), responses=[parse_payload, logic_payload]),
        tmp_path / "safety.txt",
    )
    asyncio.run(logic.build_safety_rules_and_logic("dummy text"))

    assert [rule["id"] for rule in logic.rules] == ["SAFE_5"]
    assert "ur5e" in logic.logic_raw["SAFE_5"]["ltlf"]
    assert "xarm6" in logic.logic_raw["SAFE_5"]["ltlf"]
    assert all("/destination=assembly_board-v1" in ap for ap in logic.logic_raw["SAFE_5"]["aps"])


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
    validator = PlanSafetyValidator(
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

    ok, violations = validator.validate_plan_fsa(fsa=fsa, plan=None, product_jid="assembly_board-v1")

    assert ok is False
    assert violations
    assert violations[0]["violated_rule_id"] == "SAFE_1"
    assert violations[0]["witness_events"] == ["U1.start", "U1.done", "X1.start"]


def test_offline_validator_rejects_missing_eventual_successor_at_marked_state() -> None:
    rule = _return_home_rule()
    validator = PlanSafetyValidator(
        rules=[rule],
        dfa_map={"SAFE_R": _return_home_dot()},
        tools_catalog=_tools_catalog(),
    )

    ok, violations = validator.validate_plan_fsa(
        fsa=_response_plan_fsa_missing_move_home(),
        plan=None,
        product_jid="assembly_board-v1@localhost",
    )

    assert ok is False
    assert violations
    assert violations[0]["violated_rule_id"] == "SAFE_R"
    assert violations[0]["witness_events"] == ["I1.done"]


def test_winning_set_reports_pending_obligation_and_safe_suffix() -> None:
    rule = _return_home_rule()
    dfa_map = {"SAFE_R": _return_home_dot()}
    fsa = _response_plan_fsa_with_move_home()
    validator = PlanSafetyValidator(
        rules=[rule],
        dfa_map=dfa_map,
        tools_catalog=_tools_catalog(),
    )
    winning = validator.compute_winning_set(fsa=fsa, plan=None)

    safety_monitor = OnlineSafetyMonitor(dfa_map, [rule], tools_catalog=_tools_catalog())
    fsa_monitor = OnlineFsaMonitor(fsa)
    supervisor = OnlineSafetySupervisor(
        winning_set_data=winning,
        fsa_monitor=fsa_monitor,
        safety_monitor=safety_monitor,
    )

    assert supervisor.classify()["status"] == "safe"

    fsa_monitor.process_event(
        event_type="done",
        task_id="I1",
        function_name="place_insert",
        resource_jid="ur5e@localhost",
        status="completed",
    )
    safety_monitor.process_finish_event(
        {
            "resource_jid": "ur5e@localhost",
            "function_name": "place_insert",
            "params": {},
            "current_state": "placed",
        }
    )

    diagnosis = supervisor.classify()
    assert diagnosis["status"] == "pending_obligation"
    assert diagnosis["rule_ids"] == ["SAFE_R"]
    assert diagnosis["safe_next_task_ids"] == ["H1"]
    assert diagnosis["safe_suffix_hint"]
    assert diagnosis["safe_suffix_hint"][0]["task_id"] == "H1"


def test_supervisor_detects_inevitable_violation_when_required_successor_missing() -> None:
    rule = _return_home_rule()
    dfa_map = {"SAFE_R": _return_home_dot()}
    fsa = _response_plan_fsa_missing_move_home()
    validator = PlanSafetyValidator(
        rules=[rule],
        dfa_map=dfa_map,
        tools_catalog=_tools_catalog(),
    )
    winning = validator.compute_winning_set(fsa=fsa, plan=None)

    safety_monitor = OnlineSafetyMonitor(dfa_map, [rule], tools_catalog=_tools_catalog())
    fsa_monitor = OnlineFsaMonitor(fsa)
    supervisor = OnlineSafetySupervisor(
        winning_set_data=winning,
        fsa_monitor=fsa_monitor,
        safety_monitor=safety_monitor,
    )

    fsa_monitor.process_event(
        event_type="done",
        task_id="I1",
        function_name="place_insert",
        resource_jid="ur5e@localhost",
        status="completed",
    )
    safety_monitor.process_finish_event(
        {
            "resource_jid": "ur5e@localhost",
            "function_name": "place_insert",
            "params": {},
            "current_state": "placed",
        }
    )

    diagnosis = supervisor.classify()
    assert diagnosis["status"] == "inevitable_violation"
    assert diagnosis["rule_ids"] == ["SAFE_R"]
    assert diagnosis["safe_next_task_ids"] == []


def test_supervisor_blocks_start_when_required_successor_is_absent_from_suffix() -> None:
    rule = _return_home_rule()
    dfa_map = {"SAFE_R": _return_home_dot()}
    fsa = {
        "A": {
            "X": [
                "(ur5e@localhost=(k=0,idle))",
                "(ur5e@localhost=(k=0,run=I1:place_insert))",
                "(ur5e@localhost=(k=1,idle))",
            ],
            "E": ["I1.start", "I1.done"],
            "Tr": [
                {
                    "from": "(ur5e@localhost=(k=0,idle))",
                    "event": "I1.start",
                    "to": "(ur5e@localhost=(k=0,run=I1:place_insert))",
                    "task_id": "I1",
                    "resource_jid": "ur5e@localhost",
                    "function_name": "place_insert",
                    "params": {},
                    "in_state": "positioned",
                    "out_state": "placed",
                },
                {
                    "from": "(ur5e@localhost=(k=0,run=I1:place_insert))",
                    "event": "I1.done",
                    "to": "(ur5e@localhost=(k=1,idle))",
                    "task_id": "I1",
                    "resource_jid": "ur5e@localhost",
                    "function_name": "place_insert",
                    "params": {},
                    "in_state": "positioned",
                    "out_state": "placed",
                },
            ],
            "x0": "(ur5e@localhost=(k=0,idle))",
            "Xm": ["(ur5e@localhost=(k=1,idle))"],
        }
    }

    validator = PlanSafetyValidator(
        rules=[rule],
        dfa_map=dfa_map,
        tools_catalog=_tools_catalog(),
    )
    winning = validator.compute_winning_set(fsa=fsa, plan=None)

    safety_monitor = OnlineSafetyMonitor(dfa_map, [rule], tools_catalog=_tools_catalog())
    fsa_monitor = OnlineFsaMonitor(fsa)
    supervisor = OnlineSafetySupervisor(
        winning_set_data=winning,
        fsa_monitor=fsa_monitor,
        safety_monitor=safety_monitor,
    )

    allowed, diagnosis = supervisor.check_candidate(
        {
            "task_id": "I1",
            "resource_jid": "ur5e@localhost",
            "function_name": "place_insert",
            "params": {},
        }
    )

    assert allowed is False
    assert diagnosis["status"] == "inevitable_violation"
    assert diagnosis["safe_next_task_ids"] == []


def test_reactive_supervisor_allows_start_until_missing_successor_obligation_triggers() -> None:
    rule = _return_home_rule()
    dfa_map = {"SAFE_R": _return_home_dot()}
    fsa = {
        "A": {
            "X": [
                "(ur5e@localhost=(k=0,idle))",
                "(ur5e@localhost=(k=0,run=I1:place_insert))",
                "(ur5e@localhost=(k=1,idle))",
            ],
            "E": ["I1.start", "I1.done"],
            "Tr": [
                {
                    "from": "(ur5e@localhost=(k=0,idle))",
                    "event": "I1.start",
                    "to": "(ur5e@localhost=(k=0,run=I1:place_insert))",
                    "task_id": "I1",
                    "resource_jid": "ur5e@localhost",
                    "function_name": "place_insert",
                    "params": {},
                    "in_state": "positioned",
                    "out_state": "placed",
                },
                {
                    "from": "(ur5e@localhost=(k=0,run=I1:place_insert))",
                    "event": "I1.done",
                    "to": "(ur5e@localhost=(k=1,idle))",
                    "task_id": "I1",
                    "resource_jid": "ur5e@localhost",
                    "function_name": "place_insert",
                    "params": {},
                    "in_state": "positioned",
                    "out_state": "placed",
                },
            ],
            "x0": "(ur5e@localhost=(k=0,idle))",
            "Xm": ["(ur5e@localhost=(k=1,idle))"],
        }
    }

    validator = PlanSafetyValidator(
        rules=[rule],
        dfa_map=dfa_map,
        tools_catalog=_tools_catalog(),
    )
    winning = validator.compute_winning_set(fsa=fsa, plan=None)

    safety_monitor = OnlineSafetyMonitor(dfa_map, [rule], tools_catalog=_tools_catalog())
    fsa_monitor = OnlineFsaMonitor(fsa)
    supervisor = OnlineSafetySupervisor(
        winning_set_data=winning,
        fsa_monitor=fsa_monitor,
        safety_monitor=safety_monitor,
        enforcement_mode="reactive",
    )

    assert supervisor.classify()["status"] == "safe"

    start_event = {
        "task_id": "I1",
        "resource_jid": "ur5e@localhost",
        "function_name": "place_insert",
        "params": {},
    }
    allowed, diagnosis = supervisor.check_candidate(start_event)
    assert allowed is True
    assert diagnosis["status"] == "deferred_monitoring"

    fsa_monitor.process_event(
        event_type="start",
        task_id="I1",
        function_name="place_insert",
        resource_jid="ur5e@localhost",
        status="running",
    )
    allowed, _ = safety_monitor.process_start_event(start_event)
    assert allowed is True

    assert supervisor.classify()["status"] == "safe"

    fsa_monitor.process_event(
        event_type="done",
        task_id="I1",
        function_name="place_insert",
        resource_jid="ur5e@localhost",
        status="completed",
    )
    safety_monitor.process_finish_event(
        {
            "task_id": "I1",
            "resource_jid": "ur5e@localhost",
            "function_name": "place_insert",
            "params": {},
            "current_state": "placed",
        }
    )

    diagnosis = supervisor.classify(event_kind="done")
    assert diagnosis["status"] == "inevitable_violation"
    assert diagnosis["rule_ids"] == ["SAFE_R"]


def test_supervisor_blocks_candidate_outside_winning_set_but_current_state_remains_safe() -> None:
    rule = _board_mutex_rule()
    dfa_map = {"SAFE_1": _board_mutex_dot()}
    fsa = _mutex_waiting_plan_fsa()
    validator = PlanSafetyValidator(
        rules=[rule],
        dfa_map=dfa_map,
        tools_catalog=_tools_catalog(),
    )
    winning = validator.compute_winning_set(fsa=fsa, plan=None)

    safety_monitor = OnlineSafetyMonitor(dfa_map, [rule], tools_catalog=_tools_catalog())
    fsa_monitor = OnlineFsaMonitor(fsa)
    supervisor = OnlineSafetySupervisor(
        winning_set_data=winning,
        fsa_monitor=fsa_monitor,
        safety_monitor=safety_monitor,
    )

    fsa_monitor.process_event(
        event_type="done",
        task_id="U1",
        function_name="place_approach",
        resource_jid="ur5e@localhost",
        status="completed",
    )
    safety_monitor.process_finish_event(
        {
            "resource_jid": "ur5e@localhost",
            "function_name": "place_approach",
            "params": {"destination_location": "assembly_board-v1"},
            "current_state": "positioned",
        }
    )

    current = supervisor.classify()
    assert current["status"] == "safe"
    assert current["safe_next_task_ids"] == ["H1"]

    allowed, diagnosis = supervisor.check_candidate(
        {
            "task_id": "X1",
            "resource_jid": "xarm6@localhost",
            "function_name": "place_approach",
            "params": {"destination_location": "assembly_board-v1"},
        }
    )
    assert allowed is False
    assert diagnosis["status"] == "blocked_candidate"
    assert diagnosis["safe_next_task_ids"] == ["H1"]


def test_supervisor_detects_inevitable_violation_after_unmodeled_drop_failure() -> None:
    dfa_map: dict[str, str] = {}
    fsa = {
        "A": {
            "X": [
                "(ur5e@localhost=(k=0,idle))",
                "(ur5e@localhost=(k=0,run=I1:place_insert))",
                "(ur5e@localhost=(k=1,idle))",
            ],
            "E": ["I1.start", "I1.done"],
            "Tr": [
                {
                    "from": "(ur5e@localhost=(k=0,idle))",
                    "event": "I1.start",
                    "to": "(ur5e@localhost=(k=0,run=I1:place_insert))",
                    "task_id": "I1",
                    "resource_jid": "ur5e@localhost",
                    "function_name": "place_insert",
                    "params": {},
                    "in_state": "positioned",
                    "out_state": "placed",
                },
                {
                    "from": "(ur5e@localhost=(k=0,run=I1:place_insert))",
                    "event": "I1.done",
                    "to": "(ur5e@localhost=(k=1,idle))",
                    "task_id": "I1",
                    "resource_jid": "ur5e@localhost",
                    "function_name": "place_insert",
                    "params": {},
                    "in_state": "positioned",
                    "out_state": "placed",
                },
            ],
            "x0": "(ur5e@localhost=(k=0,idle))",
            "Xm": ["(ur5e@localhost=(k=1,idle))"],
        }
    }

    validator = PlanSafetyValidator(rules=[], dfa_map=dfa_map, tools_catalog=_tools_catalog())
    winning = validator.compute_winning_set(fsa=fsa, plan=None)

    safety_monitor = OnlineSafetyMonitor(dfa_map, [], tools_catalog=_tools_catalog())
    fsa_monitor = OnlineFsaMonitor(fsa)
    supervisor = OnlineSafetySupervisor(
        winning_set_data=winning,
        fsa_monitor=fsa_monitor,
        safety_monitor=safety_monitor,
    )

    start_event = {
        "task_id": "I1",
        "resource_jid": "ur5e@localhost",
        "function_name": "place_insert",
        "params": {},
    }
    allowed, _ = supervisor.check_candidate(start_event)
    assert allowed is True

    fsa_monitor.process_event(
        event_type="start",
        task_id="I1",
        function_name="place_insert",
        resource_jid="ur5e@localhost",
        status="running",
    )
    allowed, _ = safety_monitor.process_start_event(start_event)
    assert allowed is True

    fsa_monitor.process_event(
        event_type="fail",
        task_id="I1",
        function_name="place_insert",
        resource_jid="ur5e@localhost",
        status="failed:slippage",
    )
    safety_monitor.process_fail_event(
        {
            "task_id": "I1",
            "resource_jid": "ur5e@localhost",
            "function_name": "place_insert",
            "params": {},
            "status": "failed:slippage",
            "current_state": "recovery_required",
        }
    )

    diagnosis = supervisor.classify()
    assert diagnosis["status"] == "inevitable_violation"
    assert diagnosis["rule_ids"] == []
    assert diagnosis["safe_next_task_ids"] == []


def test_reactive_supervisor_replans_after_unmodeled_drop_failure() -> None:
    dfa_map: dict[str, str] = {}
    fsa = {
        "A": {
            "X": [
                "(ur5e@localhost=(k=0,idle))",
                "(ur5e@localhost=(k=0,run=I1:place_insert))",
                "(ur5e@localhost=(k=1,idle))",
            ],
            "E": ["I1.start", "I1.done"],
            "Tr": [
                {
                    "from": "(ur5e@localhost=(k=0,idle))",
                    "event": "I1.start",
                    "to": "(ur5e@localhost=(k=0,run=I1:place_insert))",
                    "task_id": "I1",
                    "resource_jid": "ur5e@localhost",
                    "function_name": "place_insert",
                    "params": {},
                    "in_state": "positioned",
                    "out_state": "placed",
                },
                {
                    "from": "(ur5e@localhost=(k=0,run=I1:place_insert))",
                    "event": "I1.done",
                    "to": "(ur5e@localhost=(k=1,idle))",
                    "task_id": "I1",
                    "resource_jid": "ur5e@localhost",
                    "function_name": "place_insert",
                    "params": {},
                    "in_state": "positioned",
                    "out_state": "placed",
                },
            ],
            "x0": "(ur5e@localhost=(k=0,idle))",
            "Xm": ["(ur5e@localhost=(k=1,idle))"],
        }
    }

    validator = PlanSafetyValidator(rules=[], dfa_map=dfa_map, tools_catalog=_tools_catalog())
    winning = validator.compute_winning_set(fsa=fsa, plan=None)

    safety_monitor = OnlineSafetyMonitor(dfa_map, [], tools_catalog=_tools_catalog())
    fsa_monitor = OnlineFsaMonitor(fsa)
    supervisor = OnlineSafetySupervisor(
        winning_set_data=winning,
        fsa_monitor=fsa_monitor,
        safety_monitor=safety_monitor,
        enforcement_mode="reactive",
    )

    start_event = {
        "task_id": "I1",
        "resource_jid": "ur5e@localhost",
        "function_name": "place_insert",
        "params": {},
    }
    allowed, diagnosis = supervisor.check_candidate(start_event)
    assert allowed is True
    assert diagnosis["status"] == "safe"

    fsa_monitor.process_event(
        event_type="start",
        task_id="I1",
        function_name="place_insert",
        resource_jid="ur5e@localhost",
        status="running",
    )
    allowed, _ = safety_monitor.process_start_event(start_event)
    assert allowed is True

    fsa_monitor.process_event(
        event_type="fail",
        task_id="I1",
        function_name="place_insert",
        resource_jid="ur5e@localhost",
        status="failed:slippage",
    )
    safety_monitor.process_fail_event(
        {
            "task_id": "I1",
            "resource_jid": "ur5e@localhost",
            "function_name": "place_insert",
            "params": {},
            "status": "failed:slippage",
            "current_state": "recovery_required",
        }
    )

    diagnosis = supervisor.classify(event_kind="fail")
    assert diagnosis["status"] == "inevitable_violation"
    assert diagnosis["rule_ids"] == []


def test_winning_set_initial_state_includes_idle_catalog_resources() -> None:
    fsa = {
        "A": {
            "X": [
                "(xarm6@localhost=(k=0,idle))",
                "(xarm6@localhost=(k=0,run=I1:pick_approach))",
                "(xarm6@localhost=(k=1,idle))",
            ],
            "E": ["I1.start", "I1.done"],
            "Tr": [
                {
                    "from": "(xarm6@localhost=(k=0,idle))",
                    "event": "I1.start",
                    "to": "(xarm6@localhost=(k=0,run=I1:pick_approach))",
                    "task_id": "I1",
                    "resource_jid": "xarm6@localhost",
                    "function_name": "pick_approach",
                    "params": {"part_name": "MCP"},
                    "in_state": "idle",
                    "out_state": "at_pick",
                },
                {
                    "from": "(xarm6@localhost=(k=0,run=I1:pick_approach))",
                    "event": "I1.done",
                    "to": "(xarm6@localhost=(k=1,idle))",
                    "task_id": "I1",
                    "resource_jid": "xarm6@localhost",
                    "function_name": "pick_approach",
                    "params": {"part_name": "MCP"},
                    "in_state": "idle",
                    "out_state": "at_pick",
                },
            ],
            "x0": "(xarm6@localhost=(k=0,idle))",
            "Xm": ["(xarm6@localhost=(k=1,idle))"],
        }
    }

    validator = PlanSafetyValidator(
        rules=[],
        dfa_map={},
        tools_catalog=_tools_catalog(),
    )
    winning = validator.compute_winning_set(fsa=fsa, plan=None)

    assert winning["initial_resource_states"] == {
        "ur5e@localhost": {"current_state": "idle", "params": {}},
        "xarm6@localhost": {"current_state": "idle", "params": {}},
    }

    safety_monitor = OnlineSafetyMonitor({}, [], tools_catalog=_tools_catalog())
    safety_monitor.seed_resource_states(
        {
            "xarm6@localhost": {"current_state": "idle"},
            "ur5e@localhost": {"current_state": "idle"},
        }
    )
    fsa_monitor = OnlineFsaMonitor(fsa)
    supervisor = OnlineSafetySupervisor(
        winning_set_data=winning,
        fsa_monitor=fsa_monitor,
        safety_monitor=safety_monitor,
    )

    diagnosis = supervisor.classify()
    assert diagnosis["status"] == "safe"

    allowed, diagnosis = supervisor.check_candidate(
        {
            "task_id": "I1",
            "resource_jid": "xarm6@localhost",
            "function_name": "pick_approach",
            "params": {"part_name": "MCP"},
        }
    )
    assert allowed is True
    assert diagnosis["status"] == "safe"


def test_supervisor_matches_case3_runtime_state_despite_enriched_finish_params() -> None:
    rules, dfa_map, plan, fsa, tools_catalog = _load_case3_llm_bridge_bundle()
    validator = PlanSafetyValidator(
        rules=rules,
        dfa_map=dfa_map,
        tools_catalog=tools_catalog,
    )
    winning = validator.compute_winning_set(fsa=fsa, plan=plan)

    safety_monitor = OnlineSafetyMonitor(dfa_map, rules, tools_catalog=tools_catalog)
    safety_monitor.seed_resource_states(
        {
            "ur5e@localhost": {"current_state": "idle"},
            "xarm6@localhost": {"current_state": "idle"},
        }
    )
    fsa_monitor = OnlineFsaMonitor(fsa)
    supervisor = OnlineSafetySupervisor(
        winning_set_data=winning,
        fsa_monitor=fsa_monitor,
        safety_monitor=safety_monitor,
        enforcement_mode="reactive",
    )

    fsa_monitor.process_event(
        event_type="start",
        task_id="REQ_2_T1",
        function_name="pick_approach",
        resource_jid="xarm6@localhost",
        status="running",
    )
    allowed, _ = safety_monitor.process_start_event(
        {
            "resource_jid": "xarm6@localhost",
            "function_name": "pick_approach",
            "params": {
                "origin_resource_location": "prusa-mk4-1",
                "part_name": "LCP",
                "speed": None,
                "product_jid": "assembly_board-v1@localhost",
                "task_id": "REQ_2_T1",
                "product_geometry": {
                    "slot_xy": [0.1, -0.08],
                    "part_height_m": 0.1,
                    "model_name": "circ_pin_large",
                    "slot_floor_z_m": 1.025,
                    "board_center": {"x": 0.0, "y": 0.0, "z": 1.02},
                },
            },
        }
    )
    assert allowed is True

    fsa_monitor.process_event(
        event_type="start",
        task_id="REQ_1_T1",
        function_name="pick_approach",
        resource_jid="ur5e@localhost",
        status="running",
    )
    allowed, _ = safety_monitor.process_start_event(
        {
            "resource_jid": "ur5e@localhost",
            "function_name": "pick_approach",
            "params": {
                "origin_resource_location": "prusa-mk4-2",
                "part_name": "MCP",
                "speed": None,
                "product_jid": "assembly_board-v1@localhost",
                "task_id": "REQ_1_T1",
                "product_geometry": {
                    "slot_xy": [0.0, -0.08],
                    "part_height_m": 0.08,
                    "model_name": "circ_pin_medium",
                    "slot_floor_z_m": 1.025,
                    "board_center": {"x": 0.0, "y": 0.0, "z": 1.02},
                },
            },
        }
    )
    assert allowed is True

    fsa_monitor.process_event(
        event_type="done",
        task_id="REQ_1_T1",
        function_name="pick_approach",
        resource_jid="ur5e@localhost",
        status="completed",
    )
    safety_monitor.process_finish_event(
        {
            "resource_jid": "ur5e@localhost",
            "function_name": "pick_approach",
            "params": {
                "origin_resource_location": "prusa-mk4-2",
                "part_name": "MCP",
                "speed": None,
                "product_jid": "assembly_board-v1@localhost",
                "task_id": "REQ_1_T1",
                "product_geometry": {
                    "slot_xy": [0.0, -0.08],
                    "part_height_m": 0.08,
                    "model_name": "circ_pin_medium",
                    "slot_floor_z_m": 1.025,
                    "board_center": {"x": 0.0, "y": 0.0, "z": 1.02},
                },
            },
            "current_state": "at_pick",
        }
    )

    diagnosis = supervisor.classify(event_kind="done")
    assert diagnosis["status"] == "pending_obligation"
    assert diagnosis["rule_ids"] == ["SAFE_1"]
    assert diagnosis["safe_next_task_ids"] == ["REQ_1_T2"]
