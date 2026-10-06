"""Finite offline recovery composition, grounded safety, and preserved history."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from functools import cache
from pathlib import Path

import pytest

from cais_spade_llm.agents.central_controller.local_composition import Budget
from cais_spade_llm.agents.central_controller.offline_safety_grounding import (
    _prepare_grounded_primitive_trace,
)

_FIXTURE = Path(__file__).parent / "fixtures/KMR_assembly_board-v1_recovery/storage_interruption"
_PLACEMENT = "KMR_STORAGE_INTERRUPTION_SEQ3"


def _case() -> dict:
    document = json.loads((_FIXTURE / "composition_evidence.json").read_text())
    reference = document["base_evidence"]
    path = _FIXTURE / reference["path"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == reference["sha256"]
    grounding = json.loads(path.read_text())["inputs"]
    grounding.update(document["grounding_input_overrides"])
    return {"grounding_inputs": grounding, **document["inputs"]}


def _analyze(case: dict, **kwargs) -> dict:
    from cais_spade_llm.agents.central_controller.offline_recovery_composition import (
        analyze_grounded_recovery_composition,
    )

    kwargs.setdefault("budget", Budget(seconds=30))
    return analyze_grounded_recovery_composition(**case, **kwargs)


def _formula(case: dict, formula: str) -> None:
    next(row for row in case["grounding_inputs"]["catalog"]["specifications"]
         if row["id"] == "KET4_Square_4mm_trim_precedence")["formula"] = formula


def _ledger(value, records: list[dict]) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "KET4_Square_4mm" and isinstance(child, dict) and "processCompleted" in child:
                child["processCompleted"] = deepcopy(records)
            _ledger(child, records)
    elif isinstance(value, list):
        for child in value:
            _ledger(child, records)


def _prepared(case: dict, choice: dict) -> dict:
    inputs = deepcopy(case["grounding_inputs"])
    inputs.update({key: deepcopy(choice[key]) for key in
                   ("programs", "stationary", "task_evidence", "state_evidence") if key in choice})
    boundaries = sorted({time for row in case["event_start_choices"] for time in row["starts"].values()})
    return _prepare_grounded_primitive_trace(**inputs, observation_boundaries=boundaries)


def _finite_truth(formula: str, word: list[dict[str, bool]]) -> bool:
    """Evaluate finite words directly, independently of MONA and the graph solver."""
    from ltlf2dfa.parser.ltlf import LTLfParser

    @cache
    def truth(node, index):
        kind = type(node).__name__
        if kind == "LTLfAtomic":
            return word[index][str(node)]
        if kind in {"LTLfTrue", "LTLfFalse"}:
            return kind == "LTLfTrue"
        if kind == "LTLfNot":
            return not truth(node.f, index)
        if kind in {"LTLfAlways", "LTLfEventually"}:
            values = (truth(node.f, pos) for pos in range(index, len(word)))
            return all(values) if kind == "LTLfAlways" else any(values)
        operands = node.formulas
        if kind == "LTLfAnd":
            return all(truth(item, index) for item in operands)
        if kind == "LTLfOr":
            return any(truth(item, index) for item in operands)
        if kind == "LTLfImplies":
            assert len(operands) == 2
            return not truth(operands[0], index) or truth(operands[1], index)
        if kind == "LTLfEquivalence":
            return len({truth(item, index) for item in operands}) == 1
        if kind == "LTLfUntil":
            assert len(operands) == 2
            return any(truth(operands[1], end)
                       and all(truth(operands[0], pos) for pos in range(index, end))
                       for end in range(index, len(word)))
        raise AssertionError(f"Reference does not implement {kind}")

    return truth(LTLfParser()(formula), 0)


def _exhaustive_reference(case: dict) -> dict[str, bool]:
    """Enumerate every supplied schedule; each admitted event then has fixed progress."""
    result = {}
    for choice in case["event_start_choices"]:
        trace = _prepared(case, choice)
        result[choice["id"]] = all(
            _finite_truth(rule["formula"], [values[rule["rule_id"]] for values in trace["valuations"]])
            for rule in trace["rules"]
        )
    return result


@pytest.fixture(scope="module")
def baseline() -> dict:
    return _analyze(_case())


def test_composition_restores_checkpoint_and_preserves_delivery(baseline) -> None:
    assert baseline["status"] == "allowed", baseline.get("reason")
    assert len(baseline["scope"]["included_resources"]) == 12
    assert len(baseline["scope"]["included_specifications"]) == 67
    assert baseline["explored_states"] > 0
    case = _case()
    before = deepcopy(case)
    result = _analyze(case)
    assert case == before
    assert result["problem_id"] == baseline["problem_id"]
    projected = result["completion_witness"]["projected_snapshot"]
    assert projected["resources"]["KMR"]["held_part"] == "KET8_Square_8mm"
    assert projected["parts"]["KET4_Square_4mm"]["contained_by"] == "assembly_board-v1"
    assert projected["parts"]["KET4_Square_4mm"]["processCompleted"] == [{"process": "trim", "result": "square"}]
    assert projected["parts"]["KET8_Square_8mm"]["processCompleted"] == []
    for key in ("nominal_tasks", "task_statuses", "resume_entry_task_ids_by_resource", "resumable_task_ids_by_resource"):
        assert projected[key] == case["grounding_inputs"]["snapshot"][key]


def test_composition_immediate_start_held_and_explicit_wait_allowed(baseline) -> None:
    assert len(baseline["choices"]) == 1
    assert baseline["choices"][0]["event_ids"] == ["KMR_STORAGE_INTERRUPTION_SEQ1"]
    at_ten = next(row for row in baseline["decision_prefixes"] if row["time"] == 10)
    start = next(row for row in at_ten["choices"] if _PLACEMENT in row["event_ids"])
    wait = next(row for row in at_ten["choices"] if row["kind"] == "wait")
    assert start["status"] == "held"
    assert wait["status"] == "allowed"
    replay = _analyze(_case(), accepted_prefix=at_ten["accepted_prefix"])
    assert replay["status"] == "allowed"
    assert replay["choices"] == at_ten["choices"]


def test_composition_internal_conflict_cannot_pause_after_admission() -> None:
    case = _case()
    case["event_start_choices"] = case["event_start_choices"][:1]
    result = _analyze(case)
    assert result["status"] == "held", result.get("reason")
    counterexample = result["counterexample"][-1]
    assert counterexample
    assert any(row.get("source", {}).get("outline_id") == _PLACEMENT
               for row in counterexample["active_steps"])
    assert 11 < counterexample["time"] < 12


def test_composition_clear_occupancy_permits_fixed_recovery() -> None:
    case = _case()
    case["event_start_choices"] = case["event_start_choices"][1:]
    assert _analyze(case)["status"] == "allowed"


@pytest.mark.parametrize("formula", ["G(ap001 -> ap002)", "F ap001", "ap002 U ap001", "G ap002 & F ap001"])
def test_composition_agrees_with_independent_exhaustive_reference(formula) -> None:
    case = _case()
    _formula(case, formula)
    expected = _exhaustive_reference(case)
    assert expected == {"immediate_KMR_place": False, "delayed_KMR_place": True}
    result = _analyze(case)
    assert result["status"] == "allowed"
    decision = next(row for row in result["decision_prefixes"] if row["time"] == 10)
    assert {row["kind"]: row["status"] for row in decision["choices"]} == {"start": "held", "wait": "allowed"}


@pytest.mark.parametrize("formula", ["F ap001", "ap002 U ap001"])
def test_composition_pending_formula_survives_event_boundaries(formula) -> None:
    case = _case()
    _formula(case, formula)
    result = _analyze(case)
    at_ten = next(row for row in result["decision_prefixes"] if row["time"] == 10)
    replay = _analyze(case, accepted_prefix=at_ten["accepted_prefix"])
    assert replay["status"] == "allowed"
    assert any("KET4_Square_4mm_trim_precedence" in rule for rule in replay["pending_rule_ids"])
    assert result["completion_witness"]["pending_rule_ids"] == []


def test_composition_unresolved_eventual_requirement_is_inconclusive() -> None:
    case = _case()
    _formula(case, "F !ap002")
    result = _analyze(case)
    assert result["status"] == "inconclusive", result
    assert result["pending_rule_ids"]
    assert not result.get("completion_witness")


def test_composition_missing_trim_held_with_consistent_snapshots() -> None:
    case = _case()
    case["event_start_choices"] = case["event_start_choices"][1:]
    _ledger(case, [])
    result = _analyze(case)
    assert result["status"] == "held", result.get("reason")
    assert "KET4_Square_4mm_trim_precedence" in result["counterexample"][-1]["violation"]["rule_id"]


@pytest.mark.parametrize("missing", ["geometry", "stationary", "custody", "trajectory", "running", "step", "completion"])
def test_composition_incomplete_evidence_never_passes(missing) -> None:
    case = _case()
    if missing == "geometry":
        del case["grounding_inputs"]["geometry"]["resources"]["M2"]
    elif missing == "stationary":
        del case["event_start_choices"][1]["stationary"]["M2"]
    elif missing == "custody":
        case["grounding_inputs"]["snapshot"]["resources"]["KMR"]["held_part"] = "KET4_Square_4mm"
    elif missing == "trajectory":
        del case["event_start_choices"][1]["programs"][0]["step_results"][11]["model_evidence"]["trajectory"]
    elif missing == "running":
        case["event_start_choices"][1]["programs"].pop()
    elif missing == "step":
        case["recovery_events"][2]["primitive_step_indices"].pop()
    else:
        case["completion"] = {}
    result = _analyze(case)
    assert result["status"] == "inconclusive", result.get("reason")
    assert not result.get("completion_witness")


@pytest.mark.parametrize("formula", ["X ap001", "G(ap002 -> X ap001)", "WX ap001", "ap002 R ap001"])
def test_composition_rejects_unsupported_temporal_operators(formula) -> None:
    case = _case()
    _formula(case, formula)
    result = _analyze(case)
    assert result["status"] == "inconclusive"
    assert "Boolean/G/F/U" in result["reason"]


@pytest.mark.parametrize("mutation", ["problem", "duplicate", "gap", "foreign_trace"])
def test_composition_incompatible_history_never_resets_monitors(baseline, mutation) -> None:
    prefix = deepcopy(next(row for row in baseline["decision_prefixes"] if row["time"] == 10)["accepted_prefix"])
    if mutation == "problem":
        prefix["problem_id"] = "another frozen problem"
    elif mutation == "duplicate":
        prefix["path"].append(prefix["path"][-1])
    elif mutation == "gap":
        prefix["path"].pop(0)
    else:
        prefix = {"clock_version": "grounded_primitive_observations_joint_trace_v1", "next_observation_index": 10}
    before = deepcopy(prefix)
    result = _analyze(_case(), accepted_prefix=prefix)
    assert result["status"] == "inconclusive"
    assert prefix == before


@pytest.mark.parametrize("kind", ["states", "time"])
def test_composition_budget_exhaustion_is_inconclusive(kind) -> None:
    budget = Budget(max_states=2, seconds=30) if kind == "states" else Budget(seconds=0)
    result = _analyze(_case(), budget=budget)
    assert result["status"] == "inconclusive"
    assert result["reason"] == ("state_limit" if kind == "states" else "time_limit")
    assert not result.get("completion_witness")


def test_composition_retains_nominal_budget_defaults() -> None:
    budget = Budget()
    assert budget.max_states == 20_000
    assert budget.seconds == 2.0


def test_composition_mutex_instances_keep_distinct_values() -> None:
    case = _case()
    choice = case["event_start_choices"][0]
    prepared = _prepared(case, choice)
    bindings = {tuple(row["binding"].get("resources", [])): row["rule_id"] for row in prepared["rules"]}
    conflicting = bindings[("KMR", "ur5e-3")]
    clear = bindings[("M1", "M2")]
    at_conflict = next(row for row in prepared["valuations"] if row[conflicting] == {"ap001": True, "ap002": True})
    assert at_conflict[clear] == {"ap001": False, "ap002": False}
    case["event_start_choices"] = [choice]
    result = _analyze(case)
    assert result["status"] == "held"
    assert result["counterexample"][-1]["violation"]["rule_id"] == conflicting


def test_composition_new_event_names_preserve_verdict_and_distinct_provenance() -> None:
    case = _case()
    original = _analyze(case)
    symbols = {}
    for row in case["recovery_events"]:
        for field in ("outline_id", "event_name"):
            symbols[row[field]] = "independently_authored_" + row[field]

    def rename(value):
        if isinstance(value, dict):
            return {symbols.get(key, key): rename(child) for key, child in value.items()}
        if isinstance(value, list):
            return [rename(child) for child in value]
        return symbols.get(value, value) if isinstance(value, str) else value

    renamed = rename(case)
    result = _analyze(renamed)
    assert original["status"] == result["status"] == "allowed"
    assert original["problem_id"] != result["problem_id"]
    for old, new in zip(case["event_start_choices"], renamed["event_start_choices"], strict=True):
        assert _prepared(case, old)["valuations"] == _prepared(renamed, new)["valuations"]
    renamed["event_start_choices"] = renamed["event_start_choices"][:1]
    held = _analyze(renamed)
    assert held["status"] == "held"
    assert any(row.get("source", {}).get("outline_id") == symbols[_PLACEMENT]
               for row in held["counterexample"][-1]["active_steps"])


def test_composition_all22_steps_keep_parameters_and_original_sources() -> None:
    case = _case()
    original = json.loads((_FIXTURE / "automatic_grounding_evidence.json").read_text())["inputs"]["programs"][0]
    for choice in case["event_start_choices"]:
        program = choice["programs"][0]
        assert program["primitive_steps"] == original["primitive_steps"]
        assert len(program["step_results"]) == 22
        for before, after in zip(original["step_results"], program["step_results"], strict=True):
            assert before["source"] == after["source"]
            assert before["resolved_params"] == after["resolved_params"]
        assert [len(row["primitive_step_indices"]) for row in case["recovery_events"]] == [5, 5, 4, 8]


def test_composition_running_work_cannot_change_between_alternatives() -> None:
    case = _case()
    result = case["event_start_choices"][1]["programs"][1]["step_results"][0]
    result["model_evidence"]["trajectory"][1]["time"] = 11
    assert _analyze(case)["status"] == "inconclusive"


def test_composition_wait_requires_explicit_stationary_coverage() -> None:
    case = _case()
    case["event_start_choices"][1]["stationary"]["KMR"] = [[24, 26]]
    result = _analyze(case)
    assert result["status"] == "inconclusive"
    assert not result.get("completion_witness")


def test_composition_unavailable_compiler_cannot_use_cached_permission(monkeypatch) -> None:
    import shutil

    monkeypatch.setattr(shutil, "which", lambda executable: None)
    result = _analyze(_case())
    assert result["status"] == "inconclusive"
    assert "MONA" in result["reason"]


def test_composition_consumes_joint_observations_once_without_terminal_tick(baseline) -> None:
    witness = baseline["completion_witness"]
    observations = witness["observations"]
    steps = [row for row in witness["path"] if row["kind"] == "observation"]
    assert len(steps) == len(observations) == witness["observation_count"]
    assert [row["time_exact"] for row in steps] == [row["time_exact"] for row in observations]
    assert len({row["time_exact"] for row in steps}) == len(steps)
    assert steps[-1]["time"] == 26
    assert all(len(row["rule_checks"]) == 67 and row["controllable"] is False for row in steps)
    assert all(row["kind"] in {"start", "wait"} for row in witness["path"] if row["controllable"])
    assert witness["completed_running_task_ids"] == ["UR5E3_ASSEMBLY_BOARD_V1_RUNNING_WITHDRAWAL"]


@pytest.mark.parametrize("field", ["duration", "trajectory", "source"])
def test_composition_future_choice_cannot_change_committed_event(field) -> None:
    case = _case()
    # Both choices commit to the same event behavior, including after their split.
    choice = case["event_start_choices"][1]
    step = choice["programs"][0]["step_results"][21]
    if field == "duration":
        step["end_time"] = 23.9
        step["model_evidence"]["trajectory"][-1]["time"] = 23.9
        choice["stationary"]["KMR"][-1] = [23.9, 26]
    elif field == "trajectory":
        step["model_evidence"]["trajectory"].insert(1, {"time": 23.2, "pose": step["start_snapshot"]["current_pose"]})
    else:
        step["source"]["step_index"] = 99
    assert _analyze(case)["status"] == "inconclusive"


@pytest.mark.parametrize("field", ["unsupported_completion_fact", "current_state", "resource_location"])
def test_composition_unknown_completion_field_is_unavailable(field) -> None:
    case = _case()
    case["completion"]["resources"]["KMR"][field] = case["grounding_inputs"]["snapshot"]["resources"]["KMR"].get(field, True)
    assert _analyze(case)["status"] == "inconclusive"


def _running_state_requirement(case: dict) -> None:
    inputs = case["grounding_inputs"]
    context = "destination=assembly_board-v1"
    inputs["catalog"]["specifications"].append({
        "id": "ur5e-3_placed", "requirement": "ur5e-3 eventually reaches placed.", "formula": "F ap007",
        "aps": [{"label": "ap007", "full": f"ap_state/assembly/any/recovery-resource-3/placed/{context}",
                 "meaning": "The resource-owned resource_state is placed in the bound context."}],
    })
    inputs["requirement_scopes"].append({
        "specification": "ur5e-3_placed", "ap_groundings": {
            "ap007": {"source": "resource_state", "resource_id": "ur5e-3", "resource_symbol": "recovery-resource-3",
                      "process": "assembly", "product": "any", "context": context,
                      "state_field": "resource_state", "state_value": "placed"},
        },
    })
    identity = {"resource_id": "ur5e-3", "process": "assembly", "product": "KET4_Square_4mm", "context": context}
    task = case["running_work"][0]
    inputs["task_evidence"] = {
        "complete": True, "source_kind": "synthetic", "horizon": [0, 26],
        "events": [{**identity, "task_id": task["task_id"], "function": task["event_name"], "start_time": 0, "end_time": 12}],
    }
    inputs["state_evidence"] = {
        "complete": True, "source_kind": "synthetic", "horizon": [0, 26],
        "initial": [{**identity, "values": {"resource_state": "idle"}}],
        "updates": [{**identity, "task_id": task["task_id"], "time": 12, "values": {"resource_state": "placed"}}],
    }


def test_composition_fixed_task_state_completion_is_observed_jointly() -> None:
    case = _case()
    _running_state_requirement(case)
    result = _analyze(case)
    assert result["status"] == "allowed", result.get("reason")
    path = result["completion_witness"]["path"]
    before = next(row for row in path if row["kind"] == "observation" and row["time"] == 10)
    completed = [row for row in path if row["kind"] == "observation" and row["time"] == 12]
    assert len(completed) == 1
    def values(row):
        return next(check["ap_values"] for check in row["rule_checks"] if check["rule_id"].startswith("ur5e-3_placed:"))

    assert values(before) == {"ap007": False}
    assert values(completed[0]) == {"ap007": True}


def test_composition_wait_cannot_select_another_resources_completion_state() -> None:
    case = _case()
    _running_state_requirement(case)
    case["event_start_choices"][0]["state_evidence"] = deepcopy(case["grounding_inputs"]["state_evidence"])
    case["event_start_choices"][0]["state_evidence"]["updates"][0]["values"]["resource_state"] = "positioned"
    result = _analyze(case)
    assert result["status"] == "inconclusive", result.get("reason")
    assert not result.get("completion_witness")


def test_composition_distinct_des_event_id_is_preserved() -> None:
    case = _case()
    identity = "KMR_STORAGE_INTERRUPTION_SEQ3_DES_EVENT"
    case["recovery_events"][2]["des_event_id"] = identity
    programs = [case["grounding_inputs"]["programs"][0]]
    programs += [choice["programs"][0] for choice in case["event_start_choices"]]
    for program in programs:
        for step in program["step_results"][10:14]:
            step["source"]["des_event_id"] = identity
    result = _analyze(case)
    assert result["status"] == "allowed", result.get("reason")
    assert any(step.get("source", {}).get("des_event_id") == identity
               for observation in result["completion_witness"]["observations"] for step in observation["active_steps"])


def _native_context(case: dict, formula: str = "F ap001", aps: list | None = None) -> dict:
    from cais_spade_llm.agents.central_controller.online_safety_monitor import OnlineSafetyMonitor
    from cais_spade_llm.agents.central_controller.reviewed_primitive_program_safety import (
        _compile_formula,
    )

    placement = next(row for row in case["recovery_events"] if row["outline_id"] == _PLACEMENT)
    aps = aps or [{"label": "ap001", "full": f"ap_event/any/any/KMR/{placement['event_name']}/any"}]
    rule = {"id": "native_rule", "formula": formula, "aps": aps}
    monitor = OnlineSafetyMonitor({"native_rule": _compile_formula(formula, {row["label"] for row in aps})}, [rule])
    resources = {rid: {"resource_state": "idle"} for rid in case["grounding_inputs"]["snapshot"]["resources"]}
    jids = {rid: rid + "@localhost" for rid in resources}
    monitor.resource_bindings = {jid: rid for rid, jid in jids.items()}
    tasks = {}
    for row in case["recovery_events"] + case["running_work"]:
        identity = row.get("outline_id", row.get("task_id"))
        task = {"task_id": identity, "resource_id": row["resource_id"],
                "event_name": row["event_name"], "parameters": {"part_name": "KET4_Square_4mm"}}
        tasks[identity] = {"task": task, "participants": [row["resource_id"]],
                           "resource_updates": {}, "product_updates": {}}
        if row in case["running_work"]:
            monitor.running_aps.update(monitor._map_task_to_aps(jids[row["resource_id"]], row["event_name"], task["parameters"]))
    return {"monitor": monitor, "current_states": dict(monitor.current_states),
            "resources": resources, "products": {}, "contexts": {}, "jids": jids, "tasks": tasks}


def test_native_history_pending_advances_only_on_task_completion() -> None:
    case = _case()
    context = _native_context(case)
    before = deepcopy(context["monitor"].current_states)
    result = _analyze(case, task_monitor_context=context)
    assert result["status"] == "allowed", result["reason"]
    assert context["monitor"].current_states == before
    assert result["task_pending_rule_ids"] == [{"scope_id": None, "rule_id": "native_rule"}]
    path = result["completion_witness"]["path"]
    started = next(row for row in path if _PLACEMENT in row.get("event_ids", []))
    assert started["task_monitor_state"]["monitors"][0]["state"]["states"] == before
    completed = next(row for row in path if row.get("event_id") == _PLACEMENT and row["kind"] == "task_completion")
    assert completed["controllable"] is False
    assert completed["task_rule_checks"][0]["accepting"] is True
    assert context["monitor"].running_aps == set()


def test_native_existing_obligation_is_not_reset() -> None:
    case = _case()
    placement = case["recovery_events"][2]["event_name"]
    context = _native_context(case, "G (ap001 -> F ap002)", [
        {"label": "ap001", "full": "ap_event/any/any/KMR/prior_task/any"},
        {"label": "ap002", "full": f"ap_event/any/any/KMR/{placement}/any"},
    ])
    monitor = context["monitor"]
    state = monitor.transition_evidence("native_rule", monitor.current_states["native_rule"], frozenset({"ap001"}))["to"]
    monitor.current_states["native_rule"] = context["current_states"]["native_rule"] = state
    result = _analyze(case, task_monitor_context=context)
    assert result["status"] == "allowed", result["reason"]
    assert result["task_monitor_state_before"]["monitors"][0]["state"]["states"] == {"native_rule": state}
    assert result["task_pending_rule_ids"]
    assert result["completion_witness"]["task_monitor_state_after"]["monitors"][0]["state"]["states"] != {"native_rule": state}


def test_native_scopes_preserve_duplicate_rule_and_ap_identifiers() -> None:
    case = _case()
    main = _native_context(case)
    scope = _native_context(case, "G !ap001")
    records = [{"scope_id": None, "monitor": main.pop("monitor"), "current_states": main.pop("current_states")},
               {"scope_id": "exact_recovery_safety_scope", "monitor": scope["monitor"], "current_states": scope["current_states"]}]
    result = _analyze(case, task_monitor_context={**main, "monitors": records})
    assert result["status"] == "held", result["reason"]
    states = result["task_monitor_state_before"]["monitors"]
    assert [row["scope_id"] for row in states] == [None, "exact_recovery_safety_scope"]
    assert all(set(row["state"]["states"]) == {"native_rule"} for row in states)
    assert any(check["scope_id"] == "exact_recovery_safety_scope" and check["status"] == "rejected"
               for edge in result["graph"]["edges"] for check in edge.get("task_rule_checks", []))


def test_native_and_physical_checks_require_the_same_continuation() -> None:
    case = _case()
    placement = case["recovery_events"][2]["event_name"]
    context = _native_context(case, "(!ap002) U ap001", [
        {"label": "ap001", "full": f"ap_event/any/any/KMR/{placement}/any"},
        {"label": "ap002", "full": "ap_state/any/any/ur5e-3/resource_state=placed/any"},
    ])
    identity = case["running_work"][0]["task_id"]
    context["tasks"][identity]["resource_updates"] = {"ur5e-3": {"resource_state": "placed"}}
    result = _analyze(case, task_monitor_context=context)
    assert result["status"] == "held", result["reason"]
    assert result["completion_witness"] is None
    assert any(edge.get("violation") for edge in result["graph"]["edges"] if edge["kind"] == "observation")
    assert any(edge.get("violation") for edge in result["graph"]["edges"] if edge["kind"] == "task_completion")


def test_native_simultaneous_completion_orders_are_all_uncontrollable() -> None:
    case = _case()
    case["event_start_choices"] = [case["event_start_choices"][1]]
    choice = case["event_start_choices"][0]
    running = case["running_work"][0]
    running["end_time"] = 24
    program = next(row for row in choice["programs"] if row["resource_id"] == "ur5e-3")
    trace = program["step_results"][0]
    trace["end_time"] = 24
    trace["model_evidence"]["trajectory"].append({"time": 24, "pose": deepcopy(trace["model_evidence"]["trajectory"][-1]["pose"])})
    choice["stationary"]["ur5e-3"] = [[24, 26]]
    context = _native_context(case, "G (ap001 -> ap002)", [
        {"label": "ap001", "full": "ap_state/any/any/ur5e-3/resource_state=placed/any"},
        {"label": "ap002", "full": "ap_state/any/any/KMR/resource_state=placed/any"},
    ])
    context["tasks"][running["task_id"]]["resource_updates"] = {"ur5e-3": {"resource_state": "placed"}}
    context["tasks"][case["recovery_events"][-1]["outline_id"]]["resource_updates"] = {"KMR": {"resource_state": "placed"}}
    result = _analyze(case, task_monitor_context=context)
    assert result["status"] == "held", result["reason"]
    edges = [row for row in result["graph"]["edges"] if row["kind"] == "task_completion" and row["time"] == 24]
    assert all(row["controllable"] is False for row in edges)
    assert any(row.get("violation") for row in edges)
    assert any(len({other["event_id"] for other in edges if other["source"] == row["source"]}) == 2 for row in edges)


@pytest.mark.parametrize("missing", ["current_states", "tasks", "running_aps"])
def test_native_incomplete_history_cannot_pass(missing) -> None:
    case = _case()
    context = _native_context(case)
    if missing == "running_aps":
        context["monitor"].running_aps = {"unprovided_running_AP"}
    else:
        context[missing].clear()
    result = _analyze(case, task_monitor_context=context)
    assert result["status"] == "inconclusive"


@pytest.mark.parametrize("formula", ["X ap001", "WX ap001", "ap001 R ap001"])
def test_native_next_and_release_formulas_are_unavailable(formula) -> None:
    case = _case()
    context = _native_context(case, formula)
    context["monitor"].safety_rules[0]["ltlf"] = context["monitor"].safety_rules[0].pop("formula")
    result = _analyze(case, task_monitor_context=context)
    assert result["status"] == "inconclusive"
    assert "Boolean/G/F/U" in result["reason"]


def test_native_custody_effect_cannot_disagree_with_its_physical_completion() -> None:
    case = _case()
    context = _native_context(case)
    context["resources"]["KMR"]["held_part"] = "KET8_Square_8mm"
    first = case["recovery_events"][0]["outline_id"]
    context["tasks"][first]["resource_updates"] = {"KMR": {"held_part": "KET4_Square_4mm"}}
    result = _analyze(case, task_monitor_context=context)
    assert result["status"] == "inconclusive"
    assert "custody" in result["reason"]


def test_native_actual_macro_function_is_not_replaced_by_generated_event_name() -> None:
    case = _case()
    context = _native_context(case, "G !ap001", [
        {"label": "ap001", "full": "ap_event/any/any/KMR/execute_recovery_macro/any"},
    ])
    for identity in (row["outline_id"] for row in case["recovery_events"]):
        task = context["tasks"][identity]["task"]
        task["function_name"] = "execute_recovery_macro"
        task["parameters"]["event_name"] = task["event_name"]
    result = _analyze(case, task_monitor_context=context)
    assert result["status"] == "held", result["reason"]
    assert result["choices"][0]["status"] == "held"


@pytest.mark.parametrize("formula", ["F !ap001", "!ap001"])
def test_native_excluded_completion_does_not_consume_an_empty_scope_tick(formula) -> None:
    case = _case()
    context = _native_context(case, formula, [
        {"label": "ap001", "full": "ap_event/any/any/KMR/move_to_resource/any"},
    ])
    before = dict(context["current_states"])
    context["monitors"] = [{"scope_id": None, "monitor": context.pop("monitor"),
                            "current_states": context.pop("current_states"), "event_ids": []}]
    result = _analyze(case, task_monitor_context=context)
    assert result["status"] == ("inconclusive" if formula.startswith("F") else "allowed"), result["reason"]
    assert result["task_pending_rule_ids"] == ([{"scope_id": None, "rule_id": "native_rule"}]
                                               if formula.startswith("F") else [])
    assert all(row["task_monitor_state"]["monitors"][0]["state"]["states"] == before
               for row in result["graph"]["edges"] if row["kind"] == "task_completion")
    assert all(row.get("task_rule_checks", []) == [] for row in result["graph"]["edges"])


def test_native_scope_eligibility_changes_invalidate_accepted_prefix_identity() -> None:
    case = _case()
    context = _native_context(case)
    result = _analyze(case, task_monitor_context=context)
    token = result["decision_prefixes"][0]["accepted_prefix"]
    context["event_ids"] = [case["recovery_events"][2]["outline_id"]]
    changed = _analyze(case, task_monitor_context=context, accepted_prefix=token)
    assert changed["status"] == "inconclusive"
    assert "different frozen" in changed["reason"]


def test_native_existing_M1_obligation_requires_its_missing_continuation() -> None:
    case = _case()
    context = _native_context(case, "F ap001", [
        {"label": "ap001", "full": "ap_event/any/any/KMR/move_to_resource/destination=M1"},
    ])
    before = deepcopy(case["grounding_inputs"]["snapshot"]["task_statuses"])
    result = _analyze(case, task_monitor_context=context)
    assert result["status"] == "inconclusive", result["reason"]
    assert result["reason"] == "required_continuation_outside_supplied_behavior"
    assert result["task_pending_rule_ids"] == [{"scope_id": None, "rule_id": "native_rule"}]
    assert result["completion_witness"] is None
    assert case["grounding_inputs"]["snapshot"]["task_statuses"] == before


def test_native_decision_uses_completion_endpoint_without_an_extra_physical_tick() -> None:
    case = _case()
    result = _analyze(case, task_monitor_context=_native_context(case))
    assert result["status"] == "allowed", result["reason"]
    prepared = _prepared(case, case["event_start_choices"][1])
    endpoint = next(row for row in prepared["observations"] if row["time"] == 5)
    nodes = [row for row in result["graph"]["nodes"] if row["phase"] == "decision" and row["time_exact"] == "5"]
    assert nodes and all(row["state"]["resources"] == endpoint["resources"] for row in nodes)
    assert len([row for row in result["completion_witness"]["path"]
                if row["kind"] == "observation" and row["time"] == 5]) == 1


def _physical_checkpoint(case: dict) -> dict:
    from cais_spade_llm.agents.central_controller._recovery_monitor_history import (
        physical_trace_fingerprint,
    )
    from cais_spade_llm.agents.central_controller.base_safety_checker import BaseSafetyChecker

    inputs = deepcopy(case["grounding_inputs"])
    choice = case["event_start_choices"][0]
    inputs.update({key: deepcopy(choice[key]) for key in ("programs", "stationary")})
    boundaries = [inputs["horizon"][0], *(time for row in case["event_start_choices"] for time in row["starts"].values())]
    prepared = _prepare_grounded_primitive_trace(**inputs, observation_boundaries=boundaries)
    checker = BaseSafetyChecker({row["rule_id"]: row["dfa_dot"] for row in prepared["rules"]}, prepared["rules"])
    states = {rule: checker.transition_evidence(rule, dfa["initial"], frozenset(
        label for label, value in prepared["valuations"][0][rule].items() if value))["to"]
              for rule, dfa in checker.dfas.items()}
    return {"version": 1, "grounding_inputs": inputs, "observation_boundaries": boundaries,
            "observation_count": 1, "trace_fingerprint": physical_trace_fingerprint(prepared), "safety_dfa_states": states}


def test_physical_checkpoint_replays_history_and_does_not_repeat_boundary() -> None:
    case = _case()
    checkpoint = _physical_checkpoint(case)
    result = _analyze(case, physical_checkpoint=checkpoint)
    assert result["status"] == "allowed", result["reason"]
    assert result["safety_dfa_states_before"] == checkpoint["safety_dfa_states"]
    observations = [row for row in result["completion_witness"]["path"] if row["kind"] == "observation"]
    assert all(row["time"] > 0 for row in observations)


@pytest.mark.parametrize("mutation", ["state", "fingerprint", "gap", "formula"])
def test_physical_checkpoint_rejects_unverified_or_incompatible_history(mutation) -> None:
    case = _case()
    checkpoint = _physical_checkpoint(case)
    if mutation == "state":
        checkpoint["safety_dfa_states"][next(iter(checkpoint["safety_dfa_states"]))] = "unverified_state"
    elif mutation == "fingerprint":
        checkpoint["trace_fingerprint"] = "different_trace"
    elif mutation == "gap":
        checkpoint["observation_count"] = 2
    else:
        _formula(case, "G ap002")
    result = _analyze(case, physical_checkpoint=checkpoint)
    assert result["status"] == "inconclusive"
    assert result["completion_witness"] is None


def _assembly_completion_case(completion_time: float = 12, *, recovery: bool = False) -> tuple[dict, dict]:
    """Author a declared gear_small effect alongside the existing synthetic KMR plan."""
    from cais_spade_llm.agents.central_controller.reviewed_primitive_program_safety import (
        _ENTRY,
        _MEANINGS,
        _TARGET_COMPLETED,
    )
    from cais_spade_llm.resources.environment_models import build_environment_models

    case = _case()
    case["event_start_choices"] = case["event_start_choices"][1:]
    inputs, choice = case["grounding_inputs"], case["event_start_choices"][0]
    target = "Gear_Plate/Gear_Shaft_1"
    gear = {
        "frame": "world", "current_pose": [0, 0, 1.04, 0, 0, 0, 1],
        "contained_by": "assembly_board-v1", "state": "positioned", "target": None,
        "processCompleted": [{"process": "print_part"}], "processCompleted_complete": True,
        "processCompleted_evidence": {"complete": True, "source_kind": "synthetic",
                                      "checkpoint": "before_gear_small_assembly_completion"},
        "stationary_until": 26,
    }
    inputs["snapshot"]["parts"]["gear_small"] = gear
    inputs["snapshot"]["resources"]["assembly_board-v1"]["contained_parts"].append("gear_small")
    case["completion"]["resources"]["assembly_board-v1"]["contained_parts"].insert(0, "gear_small")
    inputs["geometry"]["parts"]["gear_small"] = {
        "frame": "world", "footprint": [[-0.02, 0.02]] * 3, "target": target,
    }
    inputs["catalog"]["specifications"].append({
        "id": "gear_small_before_KET4_Square_4mm", "requirement": "gear_small before KET4_Square_4mm",
        "formula": "G !ap002 | (!ap002 U (ap001 & !ap002))",
        "aps": [{"label": "ap001", "full": _TARGET_COMPLETED, "meaning": _MEANINGS[_TARGET_COMPLETED]},
                {"label": "ap002", "full": _ENTRY, "meaning": _MEANINGS[_ENTRY]}],
    })
    inputs["requirement_scopes"].append({
        "specification": "gear_small_before_KET4_Square_4mm", "physical_ap_bindings": {
            "ap001": {"part": "gear_small", "process": "assembly", "target": target},
            "ap002": {"part": "KET4_Square_4mm", "region": "assembly_board-v1"},
        },
    })
    # Make the first exact contact occur at 13.5, independently of monitor ticks.
    choice["programs"][0]["step_results"][11]["model_evidence"]["trajectory"].insert(
        1, {"time": 13.5, "pose": [0, 0, 1.145, 1, 0, 0, 0]})
    work = case["running_work"][0]
    work["end_time"] = completion_time
    identity = work["task_id"]
    program = choice["programs"][1]
    trace = program["step_results"][0]
    trace["end_time"] = completion_time
    if completion_time > 12:
        trace["model_evidence"]["trajectory"].append({
            "time": completion_time, "pose": deepcopy(trace["model_evidence"]["trajectory"][-1]["pose"]),
        })
    choice["stationary"]["ur5e-3"] = [[completion_time, 26]]
    if recovery:
        case["running_work"] = []
        case["recovery_events"].append({
            "outline_id": identity, "resource_id": work["resource_id"], "event_name": work["event_name"],
            "primitive_step_indices": work["primitive_step_indices"], "predecessors": [],
        })
        choice["starts"][identity] = 0
        trace["source"].update(outline_id=identity, des_event_id=identity + "_DES")
    native = _native_context(case, "G (ap001 | !ap001)")
    native["products"] = deepcopy(inputs["snapshot"]["parts"])
    models = build_environment_models(inputs["scene"])
    declared = next(row for row in models["ur5e-3"]["events"]
                    if row["event_name"] == "place_insert"
                    and row["parameter_bindings"]["destination_location"].get("equals") == "assembly_board-v1")
    params = {key: row["equals"] for key, row in declared["parameter_bindings"].items() if "equals" in row}
    params.update(part_name="gear_small", target=target)
    declaration = {"event_id": declared["event_id"], "event_name": declared["event_name"], "parameters": params}
    effect = {"state": "assembled", "target": target,
              "processCompleted": [{"process": "assembly", "target": target}]}
    binding = native["tasks"][identity]
    binding["task"]["parameters"] = {"part_name": "gear_small"}
    binding["declared_task"] = declaration
    binding["product_updates"] = {"gear_small": {
        **deepcopy(effect), "processCompleted": [*deepcopy(gear["processCompleted"]), *deepcopy(effect["processCompleted"])],
    }}
    choice["task_evidence"] = {
        "complete": True, "source_kind": "synthetic", "horizon": [0, 26], "events": [{
            "task_id": identity, "resource_id": "ur5e-3", "function": work["event_name"],
            "process": "assembly", "product": "gear_small", "context": "destination=assembly_board-v1",
            "start_time": 0, "end_time": completion_time, "declared_task": deepcopy(declaration),
        }],
    }
    choice["product_effect_evidence"] = {
        "complete": True, "source_kind": "synthetic", "horizon": [0, 26], "updates": [{
            "task_id": identity, "resource_id": "ur5e-3", "time": completion_time, "kind": "predicted",
            "declaration": deepcopy(declaration), "product_effects": {"gear_small": deepcopy(effect)},
        }],
    }
    case["completion"]["parts"]["gear_small"] = deepcopy(binding["product_updates"]["gear_small"])
    return case, native


@pytest.mark.parametrize("recovery", [False, True], ids=["running", "recovery"])
@pytest.mark.parametrize("completion_time,expected", [(12, "allowed"), (13.5, "held"), (14, "held")])
def test_native_assembly_completion_must_precede_KET4_entry(recovery, completion_time, expected) -> None:
    case, native = _assembly_completion_case(completion_time, recovery=recovery)
    result = _analyze(case, task_monitor_context=native)
    assert result["status"] == expected, result["reason"]
    if expected == "held":
        rejected = result["counterexample"][-1]
        assert rejected["time"] == 13.5
        assert rejected["violation"]["rule_id"].startswith("gear_small_before_KET4_Square_4mm:")


def test_native_assembly_projection_preserves_unrelated_history_without_committing() -> None:
    case, native = _assembly_completion_case()
    before_case = deepcopy(case)
    before_native = deepcopy({key: value for key, value in native.items() if key != "monitor"})
    states = deepcopy(native["monitor"].current_states)
    result = _analyze(case, task_monitor_context=native)
    assert result["status"] == "allowed", result["reason"]
    projected = result["completion_witness"]["projected_snapshot"]
    assert projected["parts"]["gear_small"]["processCompleted"] == [
        {"process": "print_part"}, {"process": "assembly", "target": "Gear_Plate/Gear_Shaft_1"},
    ]
    assert projected["parts"]["gear_small"]["product_effect_evidence"][0]["kind"] == "predicted"
    for part in ("KET4_Square_4mm", "KET8_Square_8mm"):
        assert projected["parts"][part]["processCompleted"] == before_case["grounding_inputs"]["snapshot"]["parts"][part]["processCompleted"]
    for key in ("nominal_tasks", "task_statuses", "resume_entry_task_ids_by_resource", "resumable_task_ids_by_resource"):
        assert projected[key] == before_case["grounding_inputs"]["snapshot"][key]
    assert case == before_case
    assert {key: value for key, value in native.items() if key != "monitor"} == before_native
    assert native["monitor"].current_states == states


@pytest.mark.parametrize("mutation", ["native_context", "native_effect", "declaration", "time", "missing_ledger", "prediction_flag"])
def test_product_effects_require_exact_native_completion_authority(mutation) -> None:
    case, native = _assembly_completion_case()
    choice = case["event_start_choices"][0]
    identity = case["running_work"][0]["task_id"]
    if mutation == "native_context":
        native = None
    elif mutation == "native_effect":
        native["tasks"][identity]["product_updates"] = {}
    elif mutation == "declaration":
        native["tasks"][identity]["declared_task"]["event_name"] = "another_event"
    elif mutation == "time":
        choice["product_effect_evidence"]["updates"][0]["time"] = 11
    elif mutation == "missing_ledger":
        del choice["product_effect_evidence"]
    else:
        case["grounding_inputs"]["allow_predicted_product_effects"] = True
    result = _analyze(case, task_monitor_context=native)
    assert result["status"] == "inconclusive", result["reason"]
    assert result["completion_witness"] is None


def test_declared_product_effects_keep_generated_event_symbols_independent() -> None:
    case, native = _assembly_completion_case(recovery=True)
    original = _analyze(case, task_monitor_context=native)
    symbols = {row[key]: "new_authored_" + row[key] for row in case["recovery_events"]
               for key in ("outline_id", "event_name")}

    def renamed(value):
        if isinstance(value, dict):
            return {symbols.get(key, key): renamed(child) for key, child in value.items()}
        if isinstance(value, list):
            return [renamed(child) for child in value]
        return symbols.get(value, value) if isinstance(value, str) else value

    result = _analyze(renamed(case), task_monitor_context=renamed(native))
    assert original["status"] == result["status"] == "allowed", result["reason"]
    assert original["problem_id"] != result["problem_id"]
    assert original["completion_witness"]["projected_snapshot"]["parts"]["gear_small"]["processCompleted"] == result["completion_witness"]["projected_snapshot"]["parts"]["gear_small"]["processCompleted"]


def test_scheduling_choices_preserve_declared_completion_effects_at_shifted_times() -> None:
    case, native = _assembly_completion_case(recovery=True)
    delayed = deepcopy(case["event_start_choices"][0])
    identity = case["recovery_events"][-1]["outline_id"]
    delayed["id"] = "delayed_gear_small_completion"
    delayed["starts"][identity] = 1
    trace = delayed["programs"][1]["step_results"][0]
    trace["start_time"] += 1
    trace["end_time"] += 1
    for point in trace["model_evidence"]["trajectory"]:
        point["time"] += 1
    delayed["stationary"]["ur5e-3"] = [[0, 1], [13, 26]]
    task = delayed["task_evidence"]["events"][0]
    task["start_time"], task["end_time"] = 1, 13
    delayed["product_effect_evidence"]["updates"][0]["time"] = 13
    case["event_start_choices"].append(delayed)
    before = deepcopy(case)
    result = _analyze(case, task_monitor_context=native)
    assert result["status"] == "allowed", result["reason"]
    assert len(result["choices"]) == 2
    assert {row["status"] for row in result["choices"]} == {"allowed"}
    assert case == before
    delayed["product_effect_evidence"]["updates"][0]["product_effects"]["gear_small"]["target"] = "another_target"
    rejected = _analyze(case, task_monitor_context=native)
    assert rejected["status"] == "inconclusive"
    assert rejected["completion_witness"] is None


@pytest.mark.parametrize("planned_kind", ["predicted", "acknowledged"])
@pytest.mark.parametrize("mutation", [None, "predicted", "missing_ack", "time", "task", "effect", "extra"])
def test_product_effect_replay_requires_actual_acknowledgement(mutation, planned_kind) -> None:
    from cais_spade_llm.agents.central_controller._recovery_monitor_history import (
        acknowledged_product_effects_for_comparison,
    )

    case, _ = _assembly_completion_case()
    effect = deepcopy(case["event_start_choices"][0]["product_effect_evidence"]["updates"][0])
    effect["source_kind"] = "synthetic"
    effect["kind"] = planned_kind
    expected = {"parts": {"gear_small": {"product_effect_evidence": [effect]}}}
    observed = deepcopy(expected)
    actual = observed["parts"]["gear_small"]["product_effect_evidence"][0]
    actual["kind"] = "acknowledged"
    acknowledged = {effect["task_id"]}
    if mutation == "predicted":
        actual["kind"] = "predicted"
    elif mutation == "missing_ack":
        acknowledged.clear()
    elif mutation == "time":
        actual["time"] += 1
    elif mutation == "task":
        actual["task_id"] = "another_task"
    elif mutation == "effect":
        actual["product_effects"]["gear_small"]["target"] = "another_target"
    elif mutation == "extra":
        observed["parts"]["gear_small"]["product_effect_evidence"].append(deepcopy(actual))
    before = deepcopy(observed)
    if mutation and not (mutation == "missing_ack" and planned_kind == "acknowledged"):
        with pytest.raises(ValueError):
            acknowledged_product_effects_for_comparison(expected, observed, acknowledged)
    else:
        assert acknowledged_product_effects_for_comparison(expected, observed, acknowledged) == expected
    assert observed == before
