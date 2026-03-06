from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from pathlib import Path

os.environ.setdefault("OPENAI_API_KEY", "sk-local-test")

REPO_ROOT = Path(__file__).resolve().parents[1]
PKG_ROOT = REPO_ROOT / "cais_spade_llm"
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from cais_spade_llm.agents.central_controller.safety_logic import SafetyLogic


class _FakeController:
    def __init__(self, responses: list[dict], tools_catalog: list[dict]) -> None:
        self.logger = logging.getLogger("test.ltlf_generation")
        self.tools_catalog = tools_catalog
        self._responses = [json.dumps(r) for r in responses]

    async def ask_llm(self, **_: object) -> str:
        if not self._responses:
            raise RuntimeError("no mock response left")
        return self._responses.pop(0)

    def _static_caps_overview(self) -> str:
        return "resource: ur5e, xarm6"


def _build_logic(
    tmp_path: Path,
    *,
    tools_catalog: list[dict],
    parse_payload: dict,
    logic_payload: dict,
) -> SafetyLogic:
    controller = _FakeController([parse_payload, logic_payload], tools_catalog)
    logic = SafetyLogic(controller, tmp_path / "safety.txt")
    asyncio.run(logic.build_safety_rules_and_logic("dummy text"))
    return logic


def _normalized(text: str) -> str:
    return str(text or "").replace(" ", "")


def test_precedence_formula_is_compiled_from_grounded_aps(tmp_path: Path) -> None:
    tools_catalog = [
        {
            "function": "place_insert",
            "function_owner_agent": "ur5e",
            "process": "assembly",
        }
    ]
    parse_payload = {
        "rules": [
            {
                "id": "SAFE_1",
                "raw_text": "MCP must be placed before LRP is placed.",
                "constraint_type": "ordering_place_before",
                "process": "assembly",
                "product": ["mcp", "lrp"],
                "resources": ["any"],
                "event": "place_insert",
                "context": None,
            }
        ]
    }
    logic_payload = {
        "rules": [
            {
                "id": "SAFE_1",
                "aps": [
                    "ap/assembly/mcp/any/place_insert/any",
                    "ap/assembly/lrp/any/place_insert/any",
                ],
                "ltlf": "G (ap/assembly/mcp/any/place_insert/any)",
            }
        ]
    }

    logic = _build_logic(
        tmp_path,
        tools_catalog=tools_catalog,
        parse_payload=parse_payload,
        logic_payload=logic_payload,
    )

    assert _normalized(logic.rules[0]["ltlf"]) == "((!ap001)Uap002)"


def test_mutex_formula_is_compiled_from_grounded_aps(tmp_path: Path) -> None:
    tools_catalog = [
        {
            "function": "place_approach",
            "function_owner_agent": "ur5e",
            "process": "assembly",
        },
        {
            "function": "place_approach",
            "function_owner_agent": "xarm6",
            "process": "assembly",
        },
    ]
    parse_payload = {
        "rules": [
            {
                "id": "SAFE_1",
                "raw_text": "ur5e and xarm6 should not approach the station at the same time.",
                "constraint_type": "no_concurrent_place_approach",
                "process": "assembly",
                "product": [],
                "resources": ["ur5e", "xarm6"],
                "event": "place_approach",
                "context": {"destination": "assembly_board-v1"},
            }
        ]
    }
    logic_payload = {
        "rules": [
            {
                "id": "SAFE_1",
                "aps": [
                    "ap/assembly/any/ur5e/place_approach/legacy",
                    "ap/assembly/any/xarm6/place_approach/legacy",
                ],
                "ltlf": "ap/assembly/any/ur5e/place_approach/legacy",
            }
        ]
    }

    logic = _build_logic(
        tmp_path,
        tools_catalog=tools_catalog,
        parse_payload=parse_payload,
        logic_payload=logic_payload,
    )

    assert _normalized(logic.rules[0]["ltlf"]) == "G!(ap001&ap002)"


def test_response_formula_is_compiled_by_event_overlap_when_rule_event_is_missing(
    tmp_path: Path,
) -> None:
    tools_catalog = [
        {
            "function": "place_insert",
            "function_owner_agent": "ur5e",
            "process": "assembly",
        },
        {
            "function": "move_home",
            "function_owner_agent": "ur5e",
            "process": "assembly",
        },
        {
            "function": "place_insert",
            "function_owner_agent": "xarm6",
            "process": "assembly",
        },
        {
            "function": "move_home",
            "function_owner_agent": "xarm6",
            "process": "assembly",
        },
    ]
    parse_payload = {
        "rules": [
            {
                "id": "SAFE_1",
                "raw_text": "xarm6 and ur5e should move back home after inserting the pins.",
                "constraint_type": "post_insertion_return_home",
                "process": "assembly",
                "product": ["pins"],
                "resources": ["ur5e", "xarm6"],
                "event": None,
                "context": None,
            }
        ]
    }
    logic_payload = {
        "rules": [
            {
                "id": "SAFE_1",
                "aps": [
                    "ap/assembly/pins/ur5e/place_insert/any",
                    "ap/assembly/pins/ur5e/move_home/any",
                    "ap/assembly/pins/xarm6/place_insert/any",
                    "ap/assembly/pins/xarm6/move_home/any",
                ],
                "ltlf": "G (ap/assembly/pins/ur5e/place_insert/any)",
            }
        ]
    }

    logic = _build_logic(
        tmp_path,
        tools_catalog=tools_catalog,
        parse_payload=parse_payload,
        logic_payload=logic_payload,
    )

    assert _normalized(logic.rules[0]["ltlf"]) == "G((ap002->Fap001)&(ap004->Fap003))"


def test_absence_formula_is_compiled_from_grounded_aps(tmp_path: Path) -> None:
    tools_catalog = [
        {
            "function": "pick_approach",
            "function_owner_agent": "ur5e",
            "process": "assembly",
        }
    ]
    parse_payload = {
        "rules": [
            {
                "id": "SAFE_1",
                "raw_text": "Never perform pick approach in the forbidden zone.",
                "constraint_type": "forbidden_pick_approach",
                "process": "assembly",
                "product": ["sg"],
                "resources": ["ur5e"],
                "event": "pick_approach",
                "context": {"zone": "forbidden"},
            }
        ]
    }
    logic_payload = {
        "rules": [
            {
                "id": "SAFE_1",
                "aps": [
                    "ap/assembly/sg/ur5e/pick_approach/legacy",
                ],
                "ltlf": "ap/assembly/sg/ur5e/pick_approach/legacy",
            }
        ]
    }

    logic = _build_logic(
        tmp_path,
        tools_catalog=tools_catalog,
        parse_payload=parse_payload,
        logic_payload=logic_payload,
    )

    assert _normalized(logic.rules[0]["ltlf"]) == "G!(ap001)"


def test_unrecognized_family_preserves_llm_formula(tmp_path: Path) -> None:
    tools_catalog = [
        {
            "function": "pick_approach",
            "function_owner_agent": "ur5e",
            "process": "assembly",
        },
        {
            "function": "move_home",
            "function_owner_agent": "ur5e",
            "process": "assembly",
        },
    ]
    parse_payload = {
        "rules": [
            {
                "id": "SAFE_1",
                "raw_text": "The approach event has a custom relation to homing.",
                "constraint_type": "custom_relation",
                "process": "assembly",
                "product": ["sg"],
                "resources": ["ur5e"],
                "event": None,
                "context": None,
            }
        ]
    }
    logic_payload = {
        "rules": [
            {
                "id": "SAFE_1",
                "aps": [
                    "ap/assembly/sg/ur5e/pick_approach/any",
                    "ap/assembly/sg/ur5e/move_home/any",
                ],
                "ltlf": (
                    "G (ap/assembly/sg/ur5e/pick_approach/any"
                    " -> X ap/assembly/sg/ur5e/move_home/any)"
                ),
            }
        ]
    }

    logic = _build_logic(
        tmp_path,
        tools_catalog=tools_catalog,
        parse_payload=parse_payload,
        logic_payload=logic_payload,
    )

    assert _normalized(logic.rules[0]["ltlf"]) == "G(ap002->Xap001)"
