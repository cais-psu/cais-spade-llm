from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

from cais_spade_llm.agents.central_controller.safety_logic import SafetyLogic

TOOLS_CATALOG = [
    {
        "function": "place_insert",
        "function_owner_agent": "ur5e",
        "process": "assembly",
        "resource_type": "robot",
        "in_state": "positioned",
        "out_state": "placed",
        "required_context_keys": ["destination"],
    },
    {
        "function": "place_insert",
        "function_owner_agent": "xarm6",
        "process": "assembly",
        "resource_type": "robot",
        "in_state": "positioned",
        "out_state": "placed",
        "required_context_keys": ["destination"],
    },
]


def _safety_logic(tmp_path: Path, response: dict) -> SafetyLogic:
    async def ask_llm(**kwargs):
        del kwargs
        return json.dumps(response)

    controller = SimpleNamespace(
        logger=logging.getLogger("test.safety_logic"),
        tools_catalog=TOOLS_CATALOG,
        ask_llm=ask_llm,
    )
    logic = SafetyLogic(controller, tmp_path / "safety.txt")
    logic.rules = [
        {
            "id": "SAFE_TEXT",
            "raw_text": "LCP before LRP and same time until destination is clear",
            "constraint_type": "mutex",
            "process": "assembly",
            "product": ["lcp", "lrp"],
            "resources": ["ur5e", "xarm6"],
            "resource_types": ["robot"],
            "event": "place_insert",
            "context": {"destination": "assembly_board-v1"},
        }
    ]
    return logic


def test_safety_logic_preserves_llm_ltlf_without_semantic_fallback(tmp_path: Path):
    ap_lcp = "ap_event/assembly/lcp/ur5e/place_insert/destination=assembly_board-v1"
    ap_lrp = "ap_event/assembly/lrp/xarm6/place_insert/destination=assembly_board-v1"
    llm_ltlf = f"F ({ap_lrp})"
    logic = _safety_logic(
        tmp_path,
        {
            "rules": [
                {
                    "id": "SAFE_TEXT",
                    "aps": [ap_lcp, ap_lrp],
                    "ltlf": llm_ltlf,
                }
            ]
        },
    )

    result = asyncio.run(logic._llm_build_safety_logic())

    assert result["SAFE_TEXT"]["aps"] == [ap_lcp, ap_lrp]
    assert result["SAFE_TEXT"]["ltlf"] == llm_ltlf


def test_safety_logic_missing_ltlf_fails_instead_of_synthesizing(tmp_path: Path):
    ap_lcp = "ap_event/assembly/lcp/ur5e/place_insert/destination=assembly_board-v1"
    logic = _safety_logic(
        tmp_path,
        {
            "rules": [
                {
                    "id": "SAFE_TEXT",
                    "aps": [ap_lcp],
                    "ltlf": "",
                }
            ]
        },
    )

    with pytest.raises(RuntimeError, match="SAFE_TEXT.*no ltlf"):
        asyncio.run(logic._llm_build_safety_logic())
