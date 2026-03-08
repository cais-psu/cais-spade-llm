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

from cais_spade_llm.agents.central_controller.base_safety_checker import BaseSafetyChecker
from cais_spade_llm.agents.central_controller.safety_logic import SafetyLogic


class _FakeController:
    def __init__(self, responses: list[dict], tools_catalog: list[dict]) -> None:
        self.logger = logging.getLogger("test.ppr_ap_context_serialization")
        self.tools_catalog = tools_catalog
        self._responses = [json.dumps(r) for r in responses]

    async def ask_llm(self, **_: object) -> str:
        if not self._responses:
            raise RuntimeError("no mock response left")
        return self._responses.pop(0)

    def _static_caps_overview(self) -> str:
        return "resource: ur5e"


def _logic_with_tools(tmp_path: Path) -> SafetyLogic:
    controller = _FakeController([], [])
    return SafetyLogic(controller, tmp_path / "safety.txt")


def test_context_serializer_sorts_and_percent_encodes(tmp_path: Path) -> None:
    logic = _logic_with_tools(tmp_path)

    token = logic._serialize_context_object(
        {
            "precondition_state": "ready now",
            "destination": "assembly/board-v1",
        }
    )

    assert token == "destination=assembly%2Fboard-v1&precondition_state=ready%20now"


def test_context_normalization_keeps_only_flat_scalar_bindings(tmp_path: Path) -> None:
    logic = _logic_with_tools(tmp_path)

    normalized = logic._normalize_context_object(
        {
            "destination": "assembly_board-v1",
            "priority": 2,
            "blocking": True,
            "nested": {"phase": "start"},
            "tags": ["hot"],
            "empty": "   ",
            "none_value": None,
        }
    )

    assert normalized == {
        "destination": "assembly_board-v1",
        "priority": "2",
        "blocking": "true",
    }


def test_safety_logic_rebuilds_ap_context_from_structured_rule_context(tmp_path: Path) -> None:
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
                "raw_text": "MCP must be inserted only when ready at the board.",
                "constraint_type": "gated_insert",
                "process": "assembly",
                "product": ["MCP"],
                "resources": ["ur5e"],
                "event": "place_insert",
                "context": {
                    "precondition_state": "ready now",
                    "destination": "assembly/board-v1",
                },
            }
        ]
    }
    logic_payload = {
        "rules": [
            {
                "id": "SAFE_1",
                "aps": ["ap/assembly/mcp/ur5e/place_insert/legacy_context"],
                "ltlf": "G !(ap/assembly/mcp/ur5e/place_insert/legacy_context)",
            }
        ]
    }

    controller = _FakeController([parse_payload, logic_payload], tools_catalog)
    logic = SafetyLogic(controller, tmp_path / "safety.txt")
    asyncio.run(logic.build_safety_rules_and_logic("dummy text"))

    expected_context = "destination=assembly%2Fboard-v1&precondition_state=ready%20now"
    rule = logic.rules[0]
    full_aps = [str(ap.get("full", "")) for ap in rule.get("aps", []) if isinstance(ap, dict)]

    assert rule.get("context") == {
        "destination": "assembly/board-v1",
        "precondition_state": "ready now",
    }
    assert full_aps == [f"ap_event/assembly/mcp/ur5e/place_insert/{expected_context}"]
    assert expected_context in str(logic.logic_raw["SAFE_1"]["ltlf"])
    assert "legacy_context" not in str(logic.logic_raw["SAFE_1"]["ltlf"])


def test_base_safety_checker_matches_composite_context_against_rule_context() -> None:
    checker = BaseSafetyChecker(
        dfa_map={},
        safety_rules=[
            {
                "context": {
                    "destination": "assembly/board-v1",
                    "precondition_state": "ready",
                },
                "aps": [
                    {
                        "label": "ap001",
                        "full": (
                            "ap/assembly/mcp/ur5e/place_insert/"
                            "destination=assembly%2Fboard-v1&precondition_state=ready"
                        ),
                    }
                ],
            }
        ],
    )

    labels = checker._map_task_to_aps(
        resource_jid="ur5e@localhost",
        function_name="place_insert",
        params={
            "part_name": "MCP",
            "destination_location": "assembly/board-v1",
        },
    )

    assert labels == ["ap001"]


def test_base_safety_checker_requires_all_composite_pairs_when_param_keys_exist() -> None:
    checker = BaseSafetyChecker(
        dfa_map={},
        safety_rules=[
            {
                "context": {},
                "aps": [
                    {
                        "label": "ap001",
                        "full": (
                            "ap/assembly/mcp/ur5e/place_insert/"
                            "destination_location=assembly_board-v1&speed=0.5"
                        ),
                    }
                ],
            }
        ],
    )

    matched = checker._map_task_to_aps(
        resource_jid="ur5e@localhost",
        function_name="place_insert",
        params={
            "part_name": "MCP",
            "destination_location": "assembly_board-v1",
            "speed": 0.5,
        },
    )
    mismatched = checker._map_task_to_aps(
        resource_jid="ur5e@localhost",
        function_name="place_insert",
        params={
            "part_name": "MCP",
            "destination_location": "assembly_board-v1",
            "speed": 0.8,
        },
    )

    assert matched == ["ap001"]
    assert mismatched == []


def test_base_safety_checker_preserves_legacy_single_token_context_matching() -> None:
    checker = BaseSafetyChecker(
        dfa_map={},
        safety_rules=[
            {
                "context": {},
                "aps": [
                    {
                        "label": "ap001",
                        "full": "ap/assembly/mcp/ur5e/place_insert/assembly_board-v1",
                    }
                ],
            }
        ],
    )

    labels = checker._map_task_to_aps(
        resource_jid="ur5e@localhost",
        function_name="place_insert",
        params={
            "part_name": "MCP",
            "destination_location": "assembly_board-v1",
        },
    )

    assert labels == ["ap001"]
