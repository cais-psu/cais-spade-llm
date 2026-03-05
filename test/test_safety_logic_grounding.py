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
        self.logger = logging.getLogger("test.safety_logic_grounding")
        self.tools_catalog = tools_catalog
        self._responses = [json.dumps(r) for r in responses]

    async def ask_llm(self, **_: object) -> str:
        if not self._responses:
            raise RuntimeError("no mock response left")
        return self._responses.pop(0)

    def _static_caps_overview(self) -> str:
        return "resource: ur5e"


def test_safety_logic_rewrites_unsupported_ap_events(tmp_path: Path) -> None:
    tools_catalog = [
        {
            "function": "pick_approach",
            "function_owner_agent": "ur5e",
            "process": "assembly",
        },
        {
            "function": "pick_grasp",
            "function_owner_agent": "ur5e",
            "process": "assembly",
        },
    ]

    parse_payload = {
        "rules": [
            {
                "id": "SAFE_1",
                "raw_text": "Do not approach SG while entering forbidden zone.",
                "constraint_type": "forbidden_overlap",
                "process": "assembly",
                "product": ["SG"],
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
                "aps": ["ap/assembly/sg/ur5e/enter_zone/forbidden"],
                "ltlf": "G !(ap/assembly/sg/ur5e/enter_zone/forbidden)",
            }
        ]
    }

    controller = _FakeController([parse_payload, logic_payload], tools_catalog)
    logic = SafetyLogic(controller, tmp_path / "safety.txt")
    msg = asyncio.run(logic.build_safety_rules_and_logic("dummy text"))

    assert "LTLf logic generation failed." not in msg
    assert logic.rules
    rule = logic.rules[0]
    assert rule.get("event") == "pick_approach"
    full_aps = [str(ap.get("full", "")) for ap in rule.get("aps", []) if isinstance(ap, dict)]
    assert full_aps
    assert all("/pick_approach/" in ap for ap in full_aps)
    assert all("enter_zone" not in ap for ap in full_aps)
    assert "enter_zone" not in str(rule.get("ltlf", ""))


def test_safety_logic_blocks_ungrounded_events(tmp_path: Path) -> None:
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
                "raw_text": "Never enter restricted zone.",
                "constraint_type": "zone_restriction",
                "process": "assembly",
                "product": ["SG"],
                "resources": ["ur5e"],
                "event": "enter_zone",
                "context": {"zone": "restricted"},
            }
        ]
    }
    logic_payload = {
        "rules": [
            {
                "id": "SAFE_1",
                "aps": ["ap/assembly/sg/ur5e/enter_zone/restricted"],
                "ltlf": "G !(ap/assembly/sg/ur5e/enter_zone/restricted)",
            }
        ]
    }

    controller = _FakeController([parse_payload, logic_payload], tools_catalog)
    logic = SafetyLogic(controller, tmp_path / "safety.txt")
    try:
        asyncio.run(logic.build_safety_rules_and_logic("dummy text"))
        assert False, "expected safety logic grounding to fail"
    except RuntimeError as exc:
        assert "unsupported events" in str(exc) or "safety logic generation failed" in str(exc)
    assert logic.logic_raw == {}
