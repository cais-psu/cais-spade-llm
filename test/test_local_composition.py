"""Completion-aware composition and atomic admission acceptance models."""

from __future__ import annotations

import json
from copy import deepcopy

import pytest

from cais_spade_llm.agents.central_controller.local_composition import (
    Action, AnalysisLimit, Budget, Edge, Scope, analyze, analyze_state, nonblocking_region,
)
from cais_spade_llm.agents.central_controller.online_safety_monitor import OnlineSafetyMonitor


def monitor(dot: str = "", aps: list | None = None) -> OnlineSafetyMonitor:
    return OnlineSafetyMonitor({"SPEC": dot} if dot else {},
                               [{"id": "SPEC", "aps": aps or []}] if dot else [])


class TablePlant:
    """Finite nominal fixtures with explicit guards, writes, claims and objectives."""

    def __init__(self, initial, rows, goal, *, selected=None):
        self.initial = deepcopy(initial)
        self.rows = rows
        self.goal = goal
        self.selected = set(selected or rows)

    def action(self, name):
        row = self.rows[name]
        return Action(name, {"resource_id": name, "event_id": name, "event_name": name,
                             "parameters": {"part_name": row.get("part", name)}},
                      frozenset(row.get("claims", [name])),
                      frozenset(row.get("labels", [])))

    def select(self, candidate, running, checker, budget, *, full):
        selected = set(self.rows) if full else self.selected
        selected |= {candidate.key, *(action.key for action in running)}
        return Scope(resources=set(selected), products=set(selected), rules=set(checker.dfas),
                     tasks=selected, reasons=[{"kind": "nominal_fixture", "tasks": sorted(selected)}])

    def actions(self, state, scope, budget):
        for name in sorted(scope.tasks):
            if name in self.rows:
                yield self.action(name)

    def project(self, state, action):
        row = self.rows[action.key]
        if not row["guard"](state):
            raise ValueError("guard")
        result = deepcopy(state)
        result.update(row["effect"](state))
        return result

    def labels(self, state):
        return frozenset(state.get("aps", []))

    def key(self, state, scope):
        return json.dumps(state, sort_keys=True)

    def complete(self, state, scope):
        return self.goal(state)


def assembly_plant():
    rows = {}
    for part in ("gear_small", "KET4_Square_4mm"):
        rows[part + ":enter"] = {
            "part": part, "claims": ["entry"],
            "guard": lambda s, p=part: s["assembly_board-v1"] is None and not s[p],
            "effect": lambda s, p=part: {"assembly_board-v1": p},
        }
        rows[part + ":assemble"] = {
            "part": part, "claims": ["assembly"],
            "guard": lambda s, p=part: s["assembly_board-v1"] == p and not s[p],
            "effect": lambda s, p=part: {p: True},
            "labels": ["ap1"] if part == "gear_small" else ["ap2"],
        }
        rows[part + ":release"] = {
            "part": part, "claims": ["release"],
            "guard": lambda s, p=part: s["assembly_board-v1"] == p and s[p],
            "effect": lambda s: {"assembly_board-v1": None},
        }
    return TablePlant({"assembly_board-v1": None, "gear_small": False, "KET4_Square_4mm": False},
                      rows, lambda s: s["gear_small"] and s["KET4_Square_4mm"]
                      and s["assembly_board-v1"] is None)


PRECEDENCE = """
digraph DFA {
    node [shape = doublecircle]; 0; 1;
    init -> 0;
    0 -> 2 [label="ap2"];
    0 -> 1 [label="ap1 & !ap2"];
    0 -> 0 [label="!ap1 & !ap2"];
    1 -> 1 [label="true"];
    2 -> 2 [label="true"];
}
"""


@pytest.mark.parametrize("part,expected", [("KET4_Square_4mm", "held"), ("gear_small", "allowed")])
def test_assembly_lookahead_and_full_reference(part, expected):
    plant, checker = assembly_plant(), monitor(PRECEDENCE)
    candidate = plant.action(part + ":enter")
    assert checker.online_safety_validation(list(candidate.labels))[0]
    local = analyze(plant, checker, dict(checker.current_states), candidate)
    full = analyze(plant, checker, dict(checker.current_states), candidate, full=True)
    assert local.status == full.status == expected
    assert local.graph and local.scope.tasks
    assert checker.current_states == {"SPEC": "0"}
    assert plant.initial["assembly_board-v1"] is None
    if expected == "held":
        assert local.witness


def _whole_scope(plant, checker):
    return Scope(resources=set(plant.rows), products=set(plant.rows), rules=set(checker.dfas),
                 tasks=set(plant.rows))


def test_a_state_check_needs_no_candidate_and_reads_the_nonblocking_region():
    plant, checker = assembly_plant(), monitor(PRECEDENCE)
    result = analyze_state(plant, checker, dict(checker.current_states), _whole_scope(plant, checker))
    assert result.status == "nonblocking" and result.root in result.winning
    # With KET4 already on the board, its assembly must come before the gear's, which the rule forbids.
    plant.initial["assembly_board-v1"] = "KET4_Square_4mm"
    result = analyze_state(plant, checker, dict(checker.current_states), _whole_scope(plant, checker))
    assert (result.status, result.reason) == ("blocking", "no_joint_completion")
    assert result.root not in result.winning


def test_a_state_check_that_cannot_finish_raises_rather_than_returning_a_verdict():
    plant, checker = assembly_plant(), monitor(PRECEDENCE)
    with pytest.raises(AnalysisLimit, match="state_limit"):
        analyze_state(plant, checker, dict(checker.current_states), _whole_scope(plant, checker),
                      budget=Budget(max_states=2))


def test_allowed_assembly_strategy_finishes_and_releases():
    plant, checker = assembly_plant(), monitor(PRECEDENCE)
    result = analyze(plant, checker, dict(checker.current_states),
                     plant.action("gear_small:enter"))
    root = result.root
    for name in ("gear_small:enter", "gear_small:assemble", "gear_small:release",
                 "KET4_Square_4mm:enter", "KET4_Square_4mm:assemble", "KET4_Square_4mm:release"):
        start = next(edge for edge in result.graph[root] if edge.action == name and edge.kind == "start")
        assert start.target in result.winning
        # The start contributes running APs, without changing values or history.
        assert result.nodes[start.target].state == result.nodes[root].state
        assert result.nodes[start.target].monitors == result.nodes[root].monitors
        root = next(edge.target for edge in result.graph[start.target]
                    if edge.action == name and edge.kind == "done")
        assert root in result.winning
    assert plant.complete(result.nodes[root].state, result.scope)


def test_unsafe_uncontrollable_completion_cannot_be_hidden_by_a_successful_order():
    plant = TablePlant({"a": False, "b": False}, {
        "a": {"guard": lambda s: not s["a"],
              "effect": lambda s: {"a": True, "aps": ["ap1"] if not s["b"] else []}},
        "b": {"guard": lambda s: not s["b"], "effect": lambda s: {"b": True}},
    }, lambda s: s["a"] and s["b"])
    checker = monitor("""
    digraph DFA {
        node [shape = doublecircle]; 0; init -> 0;
        0 -> 1 [label="ap1"]; 0 -> 0 [label="!ap1"]; 1 -> 1 [label="true"];
    }""")
    # Both were already started: b-before-a can finish, a-before-b is unsafe.
    result = analyze(plant, checker, dict(checker.current_states),
                     plant.action("b"), (plant.action("a"), plant.action("b")))
    assert result.root not in result.winning
    assert any(edge.kind == "done" and not edge.controllable and edge.target is None
               for edge in result.graph[result.root])
    assert any(edge.target in result.winning for edge in result.graph[result.root])


def test_fixed_point_repeats_coaccessibility_after_uncontrollable_removal():
    # Removing B invalidates C's only completion path, then A's forced successor.
    graph = {
        "A": [Edge("C", "forced", "done", False), Edge("G", "choice", "start", True)],
        "C": [Edge("B", "choice", "start", True)],
        "B": [Edge(None, "unsafe", "done", False), Edge("G", "choice", "start", True)],
        "G": [],
    }
    assert nonblocking_region(graph, {"G"}, Budget()) == {"G"}


@pytest.mark.parametrize("needs_assembly", [False, True])
def test_M1_Conveyor_Buffer_capacity_completion(needs_assembly):
    buffer = "Buffer For Machined parts"
    rows = {
        "M1": {"claims": ["M1"], "guard": lambda s: s["M1"] == "loaded",
               "effect": lambda s: {"M1": "completed"}},
        "release_M1": {"claims": ["M1", "Conveyor"],
                       "guard": lambda s: s["M1"] == "completed" and s["Conveyor"] is None,
                       "effect": lambda s: {"M1": None, "Conveyor": "part"}},
        "Conveyor": {"claims": ["Conveyor", buffer],
                     "guard": lambda s: s["Conveyor"] == "old" and s[buffer] is None,
                     "effect": lambda s: {"Conveyor": None, buffer: "old"}},
    }
    if needs_assembly:
        rows["assembly"] = {"claims": [buffer], "guard": lambda s: s[buffer] == "resident",
                            "effect": lambda s: {buffer: None, "assembled": True}}
    plant = TablePlant({"M1": "loaded", "Conveyor": "old",
                        buffer: "resident" if needs_assembly else None},
                       rows, lambda s: s["M1"] is None and s["Conveyor"] == "part")
    result = analyze(plant, monitor(), {}, plant.action("M1"))
    assert result.status == "allowed"
    assert ("assembly" in result.scope.tasks) is needs_assembly
    assert analyze(plant, monitor(), {}, plant.action("M1"), full=True).status == result.status


def test_independent_future_work_is_not_a_completion_obligation():
    rows = {
        "local": {"guard": lambda s: not s["local"], "effect": lambda s: {"local": True}},
        "independent": {"guard": lambda s: not s["independent"],
                        "effect": lambda s: {"independent": True}},
    }
    plant = TablePlant({"local": False, "independent": False}, rows,
                       lambda s: s["local"], selected={"local"})
    local = analyze(plant, monitor(), {}, plant.action("local"))
    full = analyze(plant, monitor(), {}, plant.action("local"), full=True)
    assert local.status == full.status == "allowed"
    assert "independent" not in local.scope.tasks
    assert len(local.nodes) < len(full.nodes)


def test_later_started_action_completion_is_also_uncontrollable():
    plant = TablePlant({"x": 0}, {
        "advance": {"guard": lambda s: s["x"] == 0, "effect": lambda s: {"x": 1}},
        "finish": {"guard": lambda s: s["x"] == 1, "effect": lambda s: {"x": 2}},
    }, lambda s: s["x"] == 2)
    result = analyze(plant, monitor(), {}, plant.action("advance"))
    assert result.status == "allowed"
    assert any(edge.action == "finish" and edge.kind == "done" and not edge.controllable
               for edges in result.graph.values() for edge in edges)


def test_limits_never_return_a_partial_permission():
    plant = assembly_plant()
    result = analyze(plant, monitor(), {}, plant.action("gear_small:enter"),
                     budget=Budget(max_states=2))
    assert result.status == "inconclusive" and result.reason == "state_limit"
    assert not result.winning
    ticks = iter([0.0, 3.0, 3.0])
    result = analyze(plant, monitor(), {}, plant.action("gear_small:enter"),
                     budget=Budget(clock=lambda: next(ticks)))
    assert result.status == "inconclusive" and result.reason == "time_limit"
    # Exhaustion doesn't alter the model or poison a separately completed proof.
    assert analyze(plant, monitor(), {}, plant.action("gear_small:enter")).status == "allowed"


def test_overlapping_running_APs_survive_one_completion():
    plant = TablePlant({"a": False, "b": False}, {
        name: {"labels": ["ap1"], "guard": lambda s, n=name: not s[n],
               "effect": lambda s, n=name: {n: True}}
        for name in ("a", "b")
    }, lambda s: s["a"] and s["b"])
    checker = monitor("""
    digraph DFA { node [shape = doublecircle]; 0; init -> 0;
      0 -> 0 [label="ap1"]; 0 -> 1 [label="!ap1"]; 1 -> 0 [label="true"]; }
    """)
    result = analyze(plant, checker, dict(checker.current_states), plant.action("a"),
                     (plant.action("a"), plant.action("b")))
    for edge in result.graph[result.root]:
        assert edge.target is not None
        successor = result.nodes[edge.target]
        assert len(successor.running) == 1
        assert successor.running[0].labels == {"ap1"}


def test_terminal_empty_event_changes_only_analysis_copy():
    checker = monitor("""
    digraph DFA { node [shape = doublecircle]; 1; init -> 0;
      0 -> 1 [label="true"]; 1 -> 1 [label="true"]; }
    """)
    plant = TablePlant({"done": False}, {
        "finish": {"guard": lambda s: not s["done"], "effect": lambda s: {"done": True}},
    }, lambda s: s["done"])
    result = analyze(plant, checker, dict(checker.current_states), plant.action("finish"))
    assert result.status == "allowed"
    assert checker.current_states == {"SPEC": "0"}
    assert not checker.running_aps

def _specification_check(plant, checker, name, running=()):
    from cais_spade_llm.recovery_framework.environment_admission import specification_check

    return specification_check(plant, checker, dict(checker.current_states), plant.action(name),
                               tuple(plant.action(item) for item in running), Budget(seconds=5))


def test_nominal_admission_holds_a_start_whose_finish_breaks_a_specification():
    plant = assembly_plant()
    plant.initial["assembly_board-v1"] = "KET4_Square_4mm"
    checker = monitor(PRECEDENCE)
    # KET4's assembly would complete before gear_small's (ap2 before ap1): no accepting state stays reachable.
    held = _specification_check(plant, checker, "KET4_Square_4mm:assemble")
    assert (held.status, held.reason) == ("held", "specification_violation")
    assert held.witness[0]["rule_checks"][0]["reason"] == "accepting_state_unreachable"
    plant.initial["assembly_board-v1"] = "gear_small"
    assert _specification_check(plant, checker, "gear_small:assemble").status == "allowed"


def test_nominal_admission_holds_a_start_that_needs_a_running_reservation():
    plant = assembly_plant()
    checker = monitor()
    held = _specification_check(plant, checker, "gear_small:enter", running=("KET4_Square_4mm:enter",))
    assert (held.status, held.reason) == ("held", "running_reservation_conflict")
    assert held.scope.reasons[0]["reservations"] == ["entry"]


def test_nominal_admission_without_specifications_searches_no_plant_completion():
    plant = assembly_plant()
    plant.goal = lambda _state: False
    # Eq. 13 checks the monitors only; with none, an enabled start is allowed even if the plant cannot finish.
    allowed = _specification_check(plant, monitor(), "gear_small:enter")
    assert (allowed.status, allowed.nodes) == ("allowed", {})
