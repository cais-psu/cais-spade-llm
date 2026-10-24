from __future__ import annotations

"""Physical admission of nominal starts and their native sends against predefined specifications."""

import asyncio
import time
from copy import deepcopy
from threading import RLock
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from test_continuous_motion import configuration

from cais_spade_llm.agents.central_controller.base_safety_checker import BaseSafetyChecker
from cais_spade_llm.agents.central_controller.reviewed_primitive_program_safety import (
    _compile_formula,
)
from cais_spade_llm.recovery_framework.nominal_physical_gate import (
    _UNKNOWN,
    NominalPhysicalGate,
    _entries,
)

MUTEX = "G !(ap001 & ap002)"
ORDER = "(!ap001 U (ap002 & !ap001)) | G !ap001"


def _rule(rule_id: str, formula: str, aps: dict) -> dict:
    return {"rule_id": rule_id, "id": rule_id, "aps": [{"label": label} for label in aps],
            "ap_kinds": aps, "dfa_dot": _compile_formula(formula, set(aps))}


def _board_mutex() -> dict:
    return _rule("board_mutex", MUTEX, {"ap001": ("resource_region", {"resource": "r1", "region": "board"}),
                                         "ap002": ("resource_region", {"resource": "r2", "region": "board"})})


def _gate(rules: list, *, tracker=None, pending=None) -> NominalPhysicalGate:
    gate = object.__new__(NominalPhysicalGate)
    gate.lock = RLock()
    models = {rid: {"events": [{"event_id": 1, "event_name": "event", "controllable": True}]} for rid in ("r1", "r2")}
    gate.runtime = SimpleNamespace(context=SimpleNamespace(
        admission_lock=gate.lock, part_tracker=tracker or {}, pending_tasks=pending or {}, models=models))
    gate.regions = {"board": [[-1, 1]] * 3, "M1": [[5, 6]] * 3}
    gate.resources = {"r1": {}, "r2": {}}
    gate.rules = rules
    gate.checker = BaseSafetyChecker({rule["rule_id"]: rule["dfa_dot"] for rule in rules}, rules)
    gate.states = {rule["rule_id"]: {gate.checker.dfas[rule["rule_id"]]["initial"]} for rule in rules}
    gate.states_observed = 0.0
    gate.work, gate.history, gate.part_models = {}, [], {}
    gate.wait_timeout_s, gate.poll_s = 0.05, 0.01
    return gate


def _observation(**occupied) -> dict:
    return {"record": {"observed_monotonic": time.monotonic()}, "parts": {},
            "occupancy": {rid: {"board": occupied.get(rid) == "board", "M1": occupied.get(rid) == "M1"}
                          for rid in ("r1", "r2")}}


def _prediction(gate, *, task_id="t1", rid="r1", regions=("board",), part=None, part_regions=(),
                completions=frozenset(), observation=None) -> dict:
    still = {name: (False,) for name in gate.regions}
    moving = {name: _UNKNOWN if name in regions else (False,) for name in gate.regions}
    carried = {name: _UNKNOWN if name in part_regions else (False,) for name in gate.regions} if part else {}
    cells = _entries([gate._cell("at", rid, still, part, deepcopy(still) if part else {}),
                      gate._cell("between", rid, moving, part, carried),
                      gate._cell("at", rid, moving, part, carried, completions)])
    return {"task": {"task_id": task_id, "resource_id": rid, "event_name": "event", "parameters": {}},
            "observation": observation or _observation(), "cells": cells, "regions": set(regions),
            "part_regions": {part: set(part_regions)} if part else {}, "completions": completions,
            "prediction_time_s": 0.0}


@pytest.mark.parametrize("claimed,expected", [("board", "held"), ("M1", "allowed")])
def test_a_running_claim_holds_only_starts_that_may_enter_its_region(claimed, expected):
    gate = _gate([_board_mutex()])
    gate.claim(_prediction(gate, task_id="other", rid="r2", regions=(claimed,)))
    result = gate.decide(_prediction(gate))
    assert result["status"] == expected, result
    if expected == "held":
        assert result["rule_id"] == "board_mutex"
    assert gate.history[-1]["status"] == expected


def test_a_claim_is_released_only_by_a_snapshot_taken_after_its_execution_ended():
    gate = _gate([_board_mutex()])
    gate.claim(_prediction(gate, task_id="other", rid="r2"))
    before = _observation()
    gate.retire("other")
    assert gate.decide(_prediction(gate, observation=before))["status"] == "held"
    assert gate.decide(_prediction(gate, observation=_observation(r2="board")))["status"] == "held"
    # Resting outside the board, the finished robot no longer blocks it.
    assert gate.decide(_prediction(gate, observation=_observation()))["status"] == "allowed"
    assert "other" not in gate.work  # released, and its task is no longer pending


def _entry_rule() -> dict:
    return _rule("trim_before_entry", ORDER, {
        "ap001": ("part_region_entry", {"part": "KET4", "region": "board"}),
        "ap002": ("process_result_completed", {"part": "KET4", "process": "trim", "result": "square"})})


@pytest.mark.parametrize("completion,expected", [("none", "held"), ("running", "held"), ("done", "allowed")])
def test_a_carried_part_enters_a_region_only_after_its_required_completion(completion, expected):
    tracker = {"KET4": {"processCompleted": [{"process": "trim", "result": "square"}]
                        if completion == "done" else []}}
    gate = _gate([_entry_rule()], tracker=tracker, pending={"trim": {"task_id": "trim", "resource_id": "r2", "event_id": 1}})
    if completion == "running":
        gate.claim(_prediction(gate, task_id="trim", rid="r2", regions=(),
                               completions=frozenset({("KET4", "trim", "result", "square")})))
    prediction = _prediction(gate, regions=(), part="KET4", part_regions=("board",))
    assert gate.decide(prediction)["status"] == expected


@pytest.mark.parametrize("square_done,expected", [(False, "held"), (True, "allowed")])
def test_a_start_is_checked_against_the_completions_its_finish_applies(square_done, expected):
    processing = _rule("square_before_circle", ORDER, {
        "ap001": ("process_result_completed", {"part": "RGOCG4", "process": "trim", "result": "circle"}),
        "ap002": ("process_result_completed", {"part": "KET4", "process": "trim", "result": "square"})})
    tracker = {"KET4": {"processCompleted": [{"process": "trim", "result": "square"}] if square_done else []},
               "RGOCG4": {"processCompleted": []}}
    gate = _gate([processing], tracker=tracker)
    prediction = _prediction(gate, rid="r2", regions=(),
                             completions=frozenset({("RGOCG4", "trim", "result", "circle")}))
    assert gate.decide(prediction)["status"] == expected


def _send_case(gate, monkeypatch, cells_regions=("board",)):
    owner = SimpleNamespace(agent_name="r1", _controller=SimpleNamespace(arm_trajectory_topic="/arm"))
    cells = _prediction(gate, regions=cells_regions)["cells"]
    monkeypatch.setattr(gate, "_observe", lambda: _observation())
    monkeypatch.setattr(gate, "_sweep", lambda *args: deepcopy(cells))
    return owner


def test_a_send_extends_its_claim_and_commits_the_monitor_states(monkeypatch):
    gate = _gate([_board_mutex()])
    gate.claim(_prediction(gate, regions=()))
    owner = _send_case(gate, monkeypatch)
    gate.guard_send(owner, "/gripper", object())
    assert not gate.history
    gate.guard_send(owner, "/arm", object())
    assert gate.work["t1"]["regions"] == {"board"}
    assert gate.history[-1]["kind"] == "send" and gate.history[-1]["status"] == "allowed"


def test_a_held_send_waits_stopped_and_faults_at_its_timeout(monkeypatch):
    gate = _gate([_board_mutex()])
    gate.claim(_prediction(gate, regions=()))
    gate.claim(_prediction(gate, task_id="other", rid="r2"))
    owner = _send_case(gate, monkeypatch)
    with pytest.raises(ValueError, match="Nominal send held"):
        gate.guard_send(owner, "/arm", object())
    assert gate.work["t1"]["regions"] == set()
    assert gate.history[-1]["status"] == "held" and gate.history[-1]["waited_s"] >= gate.wait_timeout_s


def test_a_waiting_send_proceeds_once_the_other_claim_is_released(monkeypatch):
    gate = _gate([_board_mutex()])
    gate.claim(_prediction(gate, regions=()))
    gate.claim(_prediction(gate, task_id="other", rid="r2"))
    gate.wait_timeout_s = 5.0
    owner = _send_case(gate, monkeypatch)
    sleeps = []

    def sleep(seconds):
        sleeps.append(seconds)
        gate.retire("other")

    monkeypatch.setattr("cais_spade_llm.recovery_framework.nominal_physical_gate.time.sleep", sleep)
    gate.guard_send(owner, "/arm", object())
    assert sleeps and gate.work["t1"]["regions"] == {"board"}
    assert gate.history[-1]["waited_s"] > 0


def _trajectory_message(last: float):
    def point(position, sec):
        return SimpleNamespace(positions=[position], velocities=[0.0], accelerations=[], effort=[],
                               time_from_start=SimpleNamespace(sec=sec, nanosec=0))
    return SimpleNamespace(joint_names=["slide"], points=[point(0.0, 0), point(last, 1)],
                           header=SimpleNamespace(frame_id="world", stamp=SimpleNamespace(sec=0, nanosec=0)))


def test_a_native_trajectory_is_swept_through_a_region_and_refined_around_it():
    gate = _gate([_board_mutex()])
    gate.regions = {"board": [[0.45, 0.55], [-1, 1], [-1, 1]]}
    gate.resources = {"r1": {"model": "slider"}}
    controller = SimpleNamespace(recovery_safety_observer=SimpleNamespace(motion_configuration=configuration()),
                                 _attached_model=None)
    owner = SimpleNamespace(agent_name="r1", _controller=controller)
    record = {"models": {"slider": {"joints": {"slide": {"position": 0.0}},
                                    "links": {"world": {"pose": [0, 0, 0, 0, 0, 0, 1]}}}}}
    cells = gate._sweep(owner, _trajectory_message(1.0), {"record": record})
    values = [cell["occupancy"]["board"] for _, cell in cells]
    assert values[0] == (False,) and values[-1] == (False,)
    assert any(True in value for value in values)
    assert len(cells) > 3 and all(cell["part"] is None for _, cell in cells)
    away = gate._sweep(owner, _trajectory_message(-1.0), {"record": record})
    assert all(cell["occupancy"]["board"] == (False,) for _, cell in away)


def _coordinator(gate_result="allowed", commit_result="allowed"):
    from cais_spade_llm.agents.central_controller.recovery_composition_admission import (
        RecoveryCompositionAdmission,
    )

    task = {"task_id": "t1", "resource_id": "r1", "event_name": "pick_approach", "parameters": {}, "run_id": "run"}
    coordinator = RecoveryCompositionAdmission()
    native = SimpleNamespace(check=AsyncMock(return_value={"status": "allowed", "snapshot_revision": 0}),
                             commit_prepared=Mock(return_value={"status": commit_result}))
    coordinator.runtime = SimpleNamespace(product_jid="product", context=SimpleNamespace(
        run_id="run", pending_tasks={"t1": task}))
    coordinator.cca = SimpleNamespace(_environment_admission=lambda _: native)
    coordinator.nominal_gate = SimpleNamespace(predict=Mock(return_value={"prediction": True}),
                                               decide=Mock(return_value={"status": gate_result}), claim=Mock())
    return coordinator, native, task


def test_a_nominal_preview_runs_only_the_symbolic_check():
    coordinator, native, task = _coordinator()
    result = asyncio.run(coordinator.check_nominal(task, "product", commit=False))
    assert result["status"] == "allowed"
    native.check.assert_awaited_once_with(task, commit=False)
    coordinator.nominal_gate.predict.assert_not_called()


@pytest.mark.parametrize("gate_result,commit_result,granted", [
    ("allowed", "allowed", True), ("held", "allowed", False), ("allowed", "inconclusive", False)])
def test_a_nominal_start_commits_its_symbolic_grant_and_physical_claim_together(gate_result, commit_result, granted):
    coordinator, native, task = _coordinator(gate_result, commit_result)
    result = asyncio.run(coordinator.check_nominal(task, "product", commit=True))
    assert (result["status"] == "allowed") is granted, result
    assert native.commit_prepared.called is (gate_result == "allowed")
    assert coordinator.nominal_gate.claim.called is granted
    if granted:
        assert result["recovery_composition_grant"] == {"nominal_native": True, "nominal_task": task,
                                                        "run_id": "run"}


def test_controller_sends_route_to_a_recovery_grant_or_the_nominal_check():
    coordinator, _, _ = _coordinator()
    owner = SimpleNamespace(agent_name="r1")
    provider = SimpleNamespace(dispatch_command="command", dispatch_guard=Mock(return_value="command"))
    coordinator.nominal_gate.guard_send = Mock()
    guard = coordinator._dispatch_guard(owner, provider)
    assert guard("/arm", "trajectory") == "command"
    provider.dispatch_command = None
    assert guard("/arm", "trajectory") is None
    coordinator.nominal_gate.guard_send.assert_called_once_with(owner, "/arm", "trajectory")
    coordinator.nominal_gate = None
    with pytest.raises(ValueError, match="lacks its exact live physical grant"):
        guard("/arm", "trajectory")


def test_a_kmr_task_request_needs_its_admitted_nominal_claim():
    coordinator, _, _ = _coordinator()
    owner = SimpleNamespace(agent_name="KMR")
    provider = SimpleNamespace(request_guard=Mock())
    coordinator.nominal_gate.admitted = lambda task_id, rid: task_id == "t1" and rid == "KMR"
    guard = coordinator._request_guard(owner, provider)
    guard({"mode": "environment_task", "pending": {"task_id": "t1"}})
    with pytest.raises(ValueError, match="admitted nominal task"):
        guard({"mode": "environment_task", "pending": {"task_id": "t2"}})
    guard({"mode": "probe"})
    provider.request_guard.assert_called_once_with({"mode": "probe"})


def _processing_rule() -> dict:
    return _rule("square_before_circle", ORDER, {
        "ap001": ("process_result_completed", {"part": "RGOCG4", "process": "trim", "result": "circle"}),
        "ap002": ("process_result_completed", {"part": "KET4", "process": "trim", "result": "square"})})


def test_an_observation_drops_monitor_states_the_recorded_history_rules_out():
    tracker = {"KET4": {"processCompleted": []}, "RGOCG4": {"processCompleted": []}}
    gate = _gate([_processing_rule()], tracker=tracker)
    waiting = set(gate.states["square_before_circle"])
    # A send keeps every state along its word, so the state before the square trim stays possible.
    tracker["KET4"]["processCompleted"].append({"process": "trim", "result": "square"})
    gate._observe_states(_observation())
    done = set(gate.states["square_before_circle"])
    gate.states["square_before_circle"] = waiting | done
    tracker["RGOCG4"]["processCompleted"].append({"process": "trim", "result": "circle"})
    assert gate.decide(_prediction(gate, regions=()))["status"] == "allowed"
    assert gate.states["square_before_circle"] == done


def test_an_observed_violation_holds_every_start():
    tracker = {"KET4": {"processCompleted": []},
               "RGOCG4": {"processCompleted": [{"process": "trim", "result": "circle"}]}}
    gate = _gate([_processing_rule()], tracker=tracker)
    result = gate.decide(_prediction(gate, regions=()))
    assert result["status"] == "held" and result["reason"] == "observed_history_violates_specification"


def test_a_job_start_applies_the_completion_its_uncontrollable_end_records(monkeypatch):
    import cais_spade_llm.resources.environment_models as environment_models

    gate = _gate([_processing_rule()], tracker={"RGOCG4": {"processCompleted": []}})
    context = gate.runtime.context
    context.models["r2"]["events"] = [
        {"event_id": 1, "event_name": "start_job", "controllable": True},
        {"event_id": 2, "event_name": "complete_job", "controllable": False},
        {"event_id": 3, "event_name": "start_job_done", "controllable": False, "start_event_id": 1}]
    context.snapshot, context.product_name, context.requirements, context.started_tasks = dict, "product", {}, {}
    circle = {"RGOCG4": {"processCompleted": [{"process": "trim", "result": "circle"}]}}
    monkeypatch.setattr(environment_models, "project_activity",
                        lambda models, values, products, task, *args, **kwargs: (values, products))
    monkeypatch.setattr(environment_models, "project_transition", lambda models, values, products, task, *args, **kwargs:
                        (values, circle if task["event_name"] == "complete_job" else products))
    task = {"task_id": "job", "resource_id": "r2", "event_id": 1, "event_name": "start_job", "parameters": {}}
    assert gate._completions(task) == {("RGOCG4", "trim", "result", "circle")}
    context.pending_tasks = {"end": {**task, "task_id": "end", "event_id": 2, "event_name": "complete_job"}}
    assert gate._base(_observation(), None)["completions"] == {("RGOCG4", "trim", "result", "circle"): _UNKNOWN}
