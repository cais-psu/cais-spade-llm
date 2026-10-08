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
from cais_spade_llm.agents.central_controller.ppr_ap import (
    ap_record, parse_ap_record, parse_ap_definition, canonical_ap_key, make_ap_definition,
)
from ppr_ap_migration import migrate_ppr_fixture
from cais_spade_llm.recovery_framework.environment_composition import state_labels
from cais_spade_llm.resources.environment_models import build_environment_models

_ROOT = Path(__file__).resolve().parents[1]
_REFERENCE = _ROOT / "test/fixtures/legacy_task_ap_preview.json"


@lru_cache(maxsize=2)
def _reference_checker(kind: str) -> tuple[dict, BaseSafetyChecker]:
    reference = migrate_ppr_fixture(json.loads(_REFERENCE.read_text()))
    rule = next(row["rule"] for row in reference["previews"] if row["kind"] == kind)
    dot = _compile_formula(rule["ltlf"], {ap["label"] for ap in rule["aps"]})
    tools = [{"function_owner_agent": resource + "@localhost", "function": function,
              "process": "assembly"}
             for resource in ("ur5e-3", "ur5e-4")
             for function in ("place_approach", "place_insert")]
    return rule, BaseSafetyChecker({rule["id"]: dot}, [rule], tools_catalog=tools)


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
        '{"event":{"arguments":{"destination":"assembly_board-v1"},"symbol":"place_approach"},"kind":"ap_event","process":"assembly","product":"KET4_Square_4mm","resource":"ur5e-3"}'
    )
    completion = (
        '{"kind":"ap_state","process":"assembly","product":"gear_small","resource":"ur5e-4","state":{"arguments":{"destination":"assembly_board-v1"},"symbol":"placed"}}'
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
                    "arguments": {"destination": "assembly_board-v1"},
                },
            },
            "right": {
                "type": "ap_state_atom",
                "state": "placed",
                "resource": "ur5e-4",
                "product": "gear_small",
                "arguments": {"destination": "assembly_board-v1"},
            },
        }
    else:
        payload.update(aps=[entry, completion], ltlf="(!ap001 U ap002)")
    logic = _logic(payload)

    logic.logic_raw = asyncio.run(logic._llm_build_safety_logic())
    assert logic.logic_raw["SAFE_1"]["aps"] == [entry, completion]
    logic._apply_labels_into_rules()

    assert logic.rules[0]["aps"] == [
        ap_record("ap001", parse_ap_definition(entry), ""),
        ap_record("ap002", parse_ap_definition(completion), ""),
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

    assert formula == canonical_ap_key(make_ap_definition("ap_state", product, "assembly", "ur5e-4", state))
    assert aps == [formula]
    assert logic._ap_segments(formula) == {
        "prefix": "ap_state",
        "process": "assembly",
        "product": product,
        "resource": "ur5e-4",
        "event": state,
        "arguments": {},
    }


def test_raw_route_retains_existing_rejection_of_unsupported_field_state() -> None:
    ap = '{"kind":"ap_state","process":"assembly","product":"gear_small","resource":"ur5e-4","state":{"arguments":{},"symbol":"part_state=assembled"}}'
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
    reference = migrate_ppr_fixture(json.loads(_REFERENCE.read_text()))
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
        assert rule["raw_text"]
        assert rule["resources"] == actors
        for ap in rule["aps"]:
            definition = parse_ap_record(ap)
            prefix, actor = definition["kind"], definition["resource"]
            symbol = definition["state" if prefix == "ap_state" else "event"]["symbol"]
            if prefix == "ap_event":
                assert any(event["function_name"] == symbol for event in models[actor]["events"])
            else:
                field, value = symbol.split("=", 1)
                assert value in models[actor]["state_variables"][field]["domain"]


@pytest.mark.parametrize("state", ["positioned", "placed", "failed"])
def test_mutex_preserves_board_occupancy_after_release_or_failure(state: str) -> None:
    _, checker = _reference_checker("mutex")
    occupied = checker._map_state_to_aps("ur5e-4@localhost", state, {
        "resource_location": "assembly_board-v1", "held_part": None, "process": "assembly",
    })
    entering = checker._map_task_to_aps("ur5e-3@localhost", "place_approach", {
        "part_name": "KET4_Square_4mm", "destination_location": "assembly_board-v1", "process": "assembly",
    })
    assert occupied == ["ap004"]
    assert entering == ["ap001"]
    assert _accepts(checker, (frozenset(occupied),))
    assert not _accepts(checker, (frozenset(occupied), frozenset(occupied + entering)))


def test_precedence_uses_acknowledged_part_state_and_remembers_completion() -> None:
    _, checker = _reference_checker("precedence")
    params = {"part_name": "gear_small", "destination_location": "assembly_board-v1", "process": "assembly"}
    assert checker._map_task_to_aps("ur5e-4@localhost", "place_insert", params) == []
    done = state_labels(checker,
                        {"ur5e-4": {"resource_state": "placed", "held_part": None}},
                        {"gear_small": {"state": "assembled", "location": "assembly_board-v1"}},
                        {"ur5e-4": "ur5e-4@localhost"}, {"ur5e-4": params})
    assert done == frozenset({"ap004"})
    entry = frozenset(checker._map_task_to_aps("ur5e-3@localhost", "place_approach", {
        "part_name": "KET4_Square_4mm", "destination_location": "assembly_board-v1", "process": "assembly",
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
        "part_name": "KET4_Square_4mm", "destination_location": "assembly_board-v1", "process": "assembly",
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


def test_context_is_rejected_in_new_authoring_and_saved_aps() -> None:
    logic = _logic({"id": "SAFE_1"})
    with pytest.raises(ValueError, match="context"):
        logic._compile_ast_state_atom(logic.rules[0], {
            "type": "ap_state_atom", "resource": "ur5e-4", "state": "placed",
            "context": {"destination": "assembly_board-v1"},
        })
    with pytest.raises(ValueError, match="recompile"):
        parse_ap_definition("ap_state/assembly/any/ur5e-4/placed/any")


def test_spatial_ap_cannot_be_inferred_from_stored_task_location() -> None:
    definition = make_ap_definition("ap_state", "*", "*", "KMR", "any",
                                    {"region": "assembly_board-v1"})
    checker = BaseSafetyChecker({}, [{"id": "mutex", "aps": [ap_record("ap001", definition, "")]}])
    with pytest.raises(ValueError, match="Physical AP"):
        checker._map_state_to_aps("KMR@localhost", "positioned", {
            "resource_location": "assembly_board-v1"})


def test_registered_function_grounds_process_without_caller_supplied_label():
    from cais_spade_llm.agents.central_controller.base_safety_checker import BaseSafetyChecker
    from cais_spade_llm.agents.central_controller.ppr_ap import ap_record, make_ap_definition
    definition = make_ap_definition("ap_event", "gear_small", "Assembly", "KMR", "place_insert")
    checker = BaseSafetyChecker({}, [{"id": "exact", "aps": [ap_record("ap001", definition, "exact")]}],
        tools_catalog=[{"function_owner_agent": "KMR", "function": "place_insert", "process": "Assembly"}])
    checker.resource_bindings = {"kmr@localhost": "KMR"}
    assert checker._map_task_to_aps("kmr@localhost", "place_insert", {"part_name": "gear_small"}) == ["ap001"]
    with pytest.raises(ValueError, match="disagrees"):
        checker._map_task_to_aps("kmr@localhost", "place_insert", {"part_name": "gear_small", "process": "assembly"})
    assert checker._map_task_to_aps("kmr@localhost", "another_function",
        {"part_name": "gear_small", "process": "Assembly", "function_name": "place_insert"}) == []
    with pytest.raises(ValueError, match="process.*evidence"):
        BaseSafetyChecker({}, checker.safety_rules)._map_task_to_aps(
            "KMR", "place_insert", {"part_name": "gear_small"})


@pytest.mark.parametrize("formula", ["G !(ap001 & ap999)", "G !(ap001 & undeclared)"])
def test_raw_authoring_rejects_undeclared_propositions(formula):
    definition = make_ap_definition("ap_event", "*", "assembly", "ur5e-3", "place_approach")
    logic = _logic({"id": "SAFE_1", "aps": [definition], "ltlf": formula})
    with pytest.raises(ValueError, match="labels must exactly match"):
        asyncio.run(logic._llm_build_safety_logic())


@pytest.mark.parametrize("state", ["never_registered", "unknown_field=assembled", "part_state="])
def test_ast_authoring_rejects_unregistered_state_conditions(state):
    logic = _logic({"id": "SAFE_1"})
    with pytest.raises(RuntimeError, match="no registered state condition"):
        logic._compile_ast_state_atom(logic.rules[0], {
            "type": "ap_state_atom", "resource": "ur5e-4", "state": state})


@pytest.mark.parametrize("route", ["formula_ast", "raw"])
def test_authoring_rejects_process_scope_inconsistent_with_registered_event(route):
    logic = _logic({"id": "SAFE_1"})
    if route == "formula_ast":
        operation = lambda: logic._compile_ast_event_atom(logic.rules[0], {
            "type": "ap_event_atom", "resource": "ur5e-3",
            "function": "place_approach", "process": "wrong_process"})
    else:
        definition = make_ap_definition("ap_event", "*", "wrong_process", "ur5e-3", "place_approach")
        logic = _logic({"id": "SAFE_1", "aps": [definition], "ltlf": "G !ap001"})
        operation = lambda: asyncio.run(logic._llm_build_safety_logic())
    with pytest.raises(ValueError, match="process disagrees"):
        operation()


def test_event_atom_does_not_borrow_function_from_another_resource():
    logic = _logic({"id": "SAFE_1"})
    with pytest.raises(RuntimeError, match="unregistered"):
        logic._compile_ast_event_atom(logic.rules[0], {
            "type": "ap_event_atom", "resource": "ur5e-4", "function": "place_approach"})


def test_ast_physical_region_condition_keeps_its_separate_grounding_contract():
    logic = _logic({"id": "SAFE_1"})
    key, _ = logic._compile_ast_state_atom(logic.rules[0], {
        "type": "ap_state_atom", "resource": "*", "process": "*",
        "product": "*", "state": "any@assembly_board-v1"})
    assert parse_ap_definition(key)["state"] == {
        "symbol": "any", "arguments": {"region": "assembly_board-v1"}}


def test_raw_label_substitution_never_rewrites_fixed_argument_symbols():
    first = make_ap_definition("ap_event", "*", "assembly", "ur5e-3", "place_approach",
                               {"tag": "ap002"})
    second = make_ap_definition("ap_event", "*", "assembly", "ur5e-4", "place_insert")
    logic = _logic({"id": "SAFE_1", "aps": [first, second], "ltlf": "G !(ap001 & ap002)"})
    logic.logic_raw = asyncio.run(logic._llm_build_safety_logic())
    logic._apply_labels_into_rules()
    assert {ap["full"] for ap in logic.rules[0]["aps"]} == {
        canonical_ap_key(first), canonical_ap_key(second)}
    assert logic.rules[0]["ltlf"] == "G !(ap001 & ap002)"
