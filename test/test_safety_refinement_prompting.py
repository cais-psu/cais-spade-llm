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
from cais_spade_llm.prompts import (
    build_safety_logic_prompt,
    build_safety_parse_prompt,
)


class _FakeController:
    def __init__(self, responses: list[dict], tools_catalog: list[dict]) -> None:
        self.logger = logging.getLogger("test.safety_refinement_prompting")
        self.tools_catalog = tools_catalog
        self._responses = [json.dumps(r) for r in responses]
        self.prompts: list[str] = []

    async def ask_llm(self, **kwargs: object) -> str:
        self.prompts.append(str(kwargs.get("prompt", "")))
        if not self._responses:
            raise RuntimeError("no mock response left")
        return self._responses.pop(0)

    def _static_caps_overview(self) -> str:
        return "resource: ur5e"


def test_build_safety_prompts_include_refinement_feedback_and_previous_preview() -> None:
    tools_catalog = [
        {
            "function": "place_insert",
            "function_owner_agent": "ur5e",
            "process": "assembly",
        }
    ]
    previous_preview_rules = [
        {
            "id": "SAFE_1",
            "raw_text": "Legacy rule",
            "constraint_type": "old_mutex",
            "aps": ["ap/assembly/mcp/any/place_insert/any"],
            "ltlf": "G !(ap/assembly/mcp/any/place_insert/any)",
        }
    ]
    feedback = "Interpret the station action as place_insert and use precedence."

    parse_prompt = build_safety_parse_prompt(
        "MCP must be placed before LRP is placed.",
        tools_catalog,
        capability_overview="resource: ur5e",
        refinement_feedback=feedback,
        previous_preview_rules=previous_preview_rules,
    )
    logic_prompt = build_safety_logic_prompt(
        [
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
        ],
        tools_catalog,
        refinement_feedback=feedback,
        previous_preview_rules=previous_preview_rules,
    )

    assert "HUMAN-IN-THE-LOOP REFINEMENT CONTEXT" in parse_prompt
    assert feedback in parse_prompt
    assert '"constraint_type": "old_mutex"' in parse_prompt
    assert 'Do NOT invent generic stand-ins like "location"' in parse_prompt
    assert "Do NOT encode temporal semantics as boolean context" in parse_prompt
    assert "HUMAN-IN-THE-LOOP REFINEMENT CONTEXT" in logic_prompt
    assert feedback in logic_prompt
    assert '"constraint_type": "old_mutex"' in logic_prompt
    assert "Do NOT emit multiple distinct `resource_var` names" in logic_prompt
    assert "Do not replace a canonical key with a generic key like `location`." in logic_prompt


def test_safety_logic_passes_refinement_context_into_both_llm_calls(tmp_path: Path) -> None:
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
    previous_preview_rules = [
        {
            "id": "SAFE_1",
            "raw_text": "MCP and LRP are mutually exclusive.",
            "constraint_type": "old_mutex",
            "aps": ["ap/assembly/mcp/any/place_insert/any"],
            "ltlf": "G !(ap/assembly/mcp/any/place_insert/any)",
        }
    ]
    feedback = "This is an ordering rule. MCP must happen before LRP."

    controller = _FakeController([parse_payload, logic_payload], tools_catalog)
    logic = SafetyLogic(controller, tmp_path / "safety.txt")
    asyncio.run(
        logic.build_safety_rules_and_logic(
            "dummy text",
            refinement_feedback=feedback,
            previous_preview_rules=previous_preview_rules,
        )
    )

    assert len(controller.prompts) == 2
    assert feedback in controller.prompts[0]
    assert feedback in controller.prompts[1]
    assert '"constraint_type": "old_mutex"' in controller.prompts[0]
    assert '"constraint_type": "old_mutex"' in controller.prompts[1]
