"""Safety preview generation preserves exact product and AP state symbols."""

from __future__ import annotations

import asyncio
import json
import logging
from functools import lru_cache
from itertools import product
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from cais_spade_llm.agents.central_controller.base_safety_checker import BaseSafetyChecker
from cais_spade_llm.agents.central_controller.reviewed_primitive_program_safety import (
    _compile_formula,
)
from cais_spade_llm.agents.central_controller.safety_logic import SafetyLogic
from cais_spade_llm.recovery_framework.environment_composition import state_labels
from cais_spade_llm.resources.environment_models import build_environment_models

_ROOT = Path(__file__).resolve().parents[1]
_REFERENCE = _ROOT / "cais_spade_llm/specification/safety/assembly_board-v1_preview_reference.json"


@lru_cache(maxsize=2)
def _reference_checker(kind: str) -> tuple[dict, BaseSafetyChecker]:
    reference = json.loads(_REFERENCE.read_text())
    rule = next(row["rule"] for row in reference["previews"] if row["kind"] == kind)
    dot = _compile_formula(rule["ltlf"], {ap["label"] for ap in rule["aps"]})
    return rule, BaseSafetyChecker({rule["id"]: dot}, [rule])


def _accepts(checker: BaseSafetyChecker, trace: tuple[frozenset[str], ...]) -> bool:
    state = checker.dfas["SAFE_1"]["initial"]
    for values in trace:
        state = checker.transition_evidence("SAFE_1", state, values)["to"]
    return state in checker.dfas["SAFE_1"]["accepting_states"]


def _logic(payload: dict) -> SafetyLogic:
    tools = [
        {
            "function_owner_agent": "ur5e-3@localhost",
            "function": "place_approach",
            "process": "assembly",
            "in_state": "picked",
            "out_state": "positioned",
        },
        {
            "function_owner_agent": "ur5e-4@localhost",
            "function": "place_insert",
            "process": "assembly",
            "in_state": "positioned",
            "out_state": "placed",
        },
    ]
    agent = SimpleNamespace(
        logger=logging.getLogger(__name__),
        tools_catalog=tools,
        ask_llm=AsyncMock(return_value=json.dumps({"rules": [payload]})),
    )
    logic = SafetyLogic(agent, "unused_safety_preview.txt")
    logic.rules = [
        {
            "id": "SAFE_1",
            "process": "assembly",
            "resources": ["ur5e-3", "ur5e-4"],
            "product": ["KET4_Square_4mm", "gear_small"],
        }
    ]
    return logic


@pytest.mark.parametrize("route", ["formula_ast", "raw"])
def test_preview_routes_preserve_product_symbols_through_labeling(route: str) -> None:
    entry = (
        "ap_event/assembly/KET4_Square_4mm/ur5e-3/"
        "place_approach/destination=assembly_board-v1"
    )
    completion = (
        "ap_state/assembly/gear_small/ur5e-4/"
        "placed/destination=assembly_board-v1"
    )
    payload = {"id": "SAFE_1"}
    if route == "formula_ast":
        payload["formula_ast"] = {
            "op": "U",
            "left": {
                "op": "!",
                "arg": {
                    "type": "ap_event_atom",
                    "function": "place_approach",
                    "resource": "ur5e-3",
                    "product": "KET4_Square_4mm",
                    "context": {"destination": "assembly_board-v1"},
                },
            },
            "right": {
                "type": "ap_state_atom",
                "state": "placed",
                "resource": "ur5e-4",
                "product": "gear_small",
                "context": {"destination": "assembly_board-v1"},
            },
        }
    else:
        payload.update(aps=[entry, completion], ltlf=f"(!({entry}) U {completion})")
    logic = _logic(payload)

    logic.logic_raw = asyncio.run(logic._llm_build_safety_logic())
    assert logic.logic_raw["SAFE_1"]["aps"] == [entry, completion]
    logic._apply_labels_into_rules()

    assert logic.rules[0]["aps"] == [
        {"label": "ap001", "full": entry},
        {"label": "ap002", "full": completion},
    ]
    assert "ket4_square_4mm" not in json.dumps(logic.logic_raw)
    assert "ap001" in logic.rules[0]["ltlf"]
    assert "ap002" in logic.rules[0]["ltlf"]
    logic.controller_agent.ask_llm.assert_awaited_once()


@pytest.mark.parametrize("product", ["KET4_Square_4mm", "gear_small"])
@pytest.mark.parametrize("state", ["part_state=assembled", "resource_location=assembly_board-v1"])
def test_ast_state_descriptor_preserves_rule_product_and_field_state(
    product: str, state: str
) -> None:
    logic = _logic({"id": "SAFE_1"})
    rule = {**logic.rules[0], "product": [product]}
    formula, aps = logic._compile_ast_state_atom(
        rule,
        {"type": "ap_state_atom", "state": state, "resource": "ur5e-4"},
    )

    assert formula == f"ap_state/assembly/{product}/ur5e-4/{state}/any"
    assert aps == [formula]
    assert logic._ap_segments(formula) == {
        "prefix": "ap_state",
        "process": "assembly",
        "product": product,
        "resource": "ur5e-4",
        "event": state,
        "context": "any",
    }


def test_raw_route_retains_existing_rejection_of_unsupported_field_state() -> None:
    ap = "ap_state/assembly/gear_small/ur5e-4/part_state=assembled/any"
    logic = _logic({"id": "SAFE_1", "aps": [ap], "ltlf": f"F ({ap})"})

    with pytest.raises(RuntimeError, match="produced no grounded APs"):
        asyncio.run(logic._llm_build_safety_logic())


@pytest.mark.parametrize("kind", ["mutex", "precedence"])
def test_reference_dfa_matches_independent_boolean_traces(kind: str) -> None:
    rule, checker = _reference_checker(kind)
    labels = [ap["label"] for ap in rule["aps"]]
    alphabet = [frozenset(label for label, flag in zip(labels, flags, strict=True) if flag)
                for flags in product((False, True), repeat=len(labels))]
    checked = 0
    for length in range(1, 4):
        for trace in product(alphabet, repeat=length):
            if kind == "mutex":
                expected = all(not (values & {"ap001", "ap002"}
                                    and values & {"ap003", "ap004"}) for values in trace)
            else:
                completed_before = False
                expected = True
                for values in trace:
                    if values & {"ap001", "ap002"} and not completed_before:
                        expected = False
                        break
                    completed_before |= bool(values & {"ap003", "ap004"})
            assert _accepts(checker, trace) == expected, (kind, trace)
            checked += 1
    assert checked == 4368


def test_reference_bindings_come_from_declared_board_capabilities() -> None:
    reference = json.loads(_REFERENCE.read_text())
    models = build_environment_models(json.loads((_ROOT / reference["scene_source"]).read_text()))
    actors = sorted(rid for rid, model in models.items() if any(
        event["event_name"] == "place_approach"
        and event["parameter_bindings"].get("resource_id", {}).get("equals") == rid
        and event["parameter_bindings"].get("destination_location", {}).get("equals")
        == "assembly_board-v1" for event in model["events"]
    ))
    assert actors == reference["declared_board_actors"]
    assert reference["scene_resources"] == sorted(models)
    for row in reference["previews"]:
        rule = row["rule"]
        source = _REFERENCE.parent / row["safety_file"]
        assert source.read_text().strip() == "[Safety Requirements]\n- " + rule["raw_text"]
        assert rule["resources"] == actors
        for ap in rule["aps"]:
            prefix, _, _, actor, symbol, _ = ap["full"].split("/", 5)
            if prefix == "ap_event":
                assert any(event["function_name"] == symbol for event in models[actor]["events"])
            else:
                field, value = symbol.split("=", 1)
                assert value in models[actor]["state_variables"][field]["domain"]


@pytest.mark.parametrize("state", ["positioned", "placed", "failed"])
def test_mutex_preserves_board_occupancy_after_release_or_failure(state: str) -> None:
    _, checker = _reference_checker("mutex")
    occupied = checker._map_state_to_aps("ur5e-4@localhost", state, {
        "resource_location": "assembly_board-v1", "held_part": None,
    })
    entering = checker._map_task_to_aps("ur5e-3@localhost", "place_approach", {
        "part_name": "KET4_Square_4mm", "destination_location": "assembly_board-v1",
    })
    assert occupied == ["ap004"]
    assert entering == ["ap001"]
    assert _accepts(checker, (frozenset(occupied),))
    assert not _accepts(checker, (frozenset(occupied), frozenset(occupied + entering)))


def test_precedence_uses_acknowledged_part_state_and_remembers_completion() -> None:
    _, checker = _reference_checker("precedence")
    params = {"part_name": "gear_small", "destination_location": "assembly_board-v1"}
    assert checker._map_task_to_aps("ur5e-4@localhost", "place_insert", params) == []
    done = state_labels(checker,
                        {"ur5e-4": {"resource_state": "placed", "held_part": None}},
                        {"gear_small": {"state": "assembled", "location": "assembly_board-v1"}},
                        {"ur5e-4": "ur5e-4@localhost"}, {"ur5e-4": params})
    assert done == frozenset({"ap004"})
    entry = frozenset(checker._map_task_to_aps("ur5e-3@localhost", "place_approach", {
        "part_name": "KET4_Square_4mm", "destination_location": "assembly_board-v1",
    }))
    assert entry == frozenset({"ap001"})
    assert not _accepts(checker, (frozenset(), entry))
    assert not _accepts(checker, (done | entry,))
    assert _accepts(checker, (done, frozenset(), entry))
    assert _accepts(checker, (frozenset(),))


@pytest.mark.parametrize("part, destination, state", [
    ("gear_large", "assembly_board-v1", "assembled"),
    ("gear_small", "M1", "assembled"),
    ("gear_small", "assembly_board-v1", "in_transit"),
])
def test_precedence_rejects_wrong_or_unfinished_completion_witness(
    part: str, destination: str, state: str,
) -> None:
    _, checker = _reference_checker("precedence")
    assert checker._map_state_to_aps("ur5e-4@localhost", "placed", {
        "part_name": part, "destination_location": destination, "part_state": state,
    }) == []


def test_reference_native_aps_do_not_claim_unknown_recovery_events() -> None:
    _, checker = _reference_checker("precedence")
    assert checker._map_task_to_aps("ur5e-3@localhost", "execute_recovery_macro", {
        "part_name": "KET4_Square_4mm", "destination_location": "assembly_board-v1",
    }) == []


@pytest.mark.parametrize("kind", ["mutex", "precedence"])
def test_reference_entry_does_not_confuse_board_origin_with_destination(kind: str) -> None:
    _, checker = _reference_checker(kind)
    assert checker._map_task_to_aps("ur5e-3@localhost", "place_approach", {
        "part_name": "KET4_Square_4mm", "origin_resource_location": "assembly_board-v1",
        "destination_location": "Exit",
    }) == []


def test_reference_completion_requires_the_declared_destination_context() -> None:
    _, checker = _reference_checker("precedence")
    assert checker._map_state_to_aps("ur5e-4@localhost", "placed", {
        "part_name": "gear_small", "destination_location": "M1",
        "part_location": "assembly_board-v1", "part_state": "assembled",
    }) == []
