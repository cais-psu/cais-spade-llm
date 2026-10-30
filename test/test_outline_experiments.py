from __future__ import annotations

"""Study 1 experiment records: the independent outline audit and the batch runner."""

import asyncio
import hashlib
import json
import threading
from copy import deepcopy
from pathlib import Path

import pytest

from cais_spade_llm.agents.shared_information.llm_request_records import RequestRecords
from cais_spade_llm.recovery_framework import (
    outline_audit,
    outline_draft,
    outline_experiments,
    recovery_evaluation,
    recovery_observations,
)

GROUNDED = {"thought": "Use the retained facts.", "decision": "grounded", "observations": []}
DONE = {"thought": "The failure is overcome.", "decision": "recovery_complete"}
SCENARIO = "Machining breakdown during part processing"


@pytest.fixture
def observations(tmp_path):
    entry = {"resource_jid": "owner", "resource_type": "resource",
        "static_capabilities": {"reachability": ["M2"], "capabilities": []},
        "recovery_snapshot": {"resource_jid": "owner", "resource_state": "idle", "resource_location": "M2"},
        "recovery_des_model": {"state_variables": {
            "resource_state": {"type": "string", "scope": "resource"},
            "resource_location": {"type": "string", "scope": "resource"},
            "part_name": {"type": ["string", "null"], "scope": "resource"},
            "part_location": {"type": "string", "scope": "part"}},
            "events": [
                {"event_name": "machine_part", "parameter_bindings": {"part_name": {"equals": "KET4_Square_4mm"}},
                 "updates": {"resource_state": {"set": "completed"}}},
                {"event_name": "resume_task", "guards": {"resource_state": {"equals": "completed"}}}]}}
    record = {"scenario": SCENARIO, "run_id": "capture", "jids": {"M2": "owner"},
        "pending_nodes": [
            {"id": "replaced", "function_name": "machine_part", "params": {"resource_id": "M2", "part_name": "KET4_Square_4mm"}},
            {"id": "resume", "function_name": "resume_task", "params": {"resource_id": "M2"}}],
        "captured_at": "recorded-time", "offline_ready": False, "missing": ["physical_evidence"],
        "prepared_recovery_request": {"ra_jid": "owner", "recovery_resources": {"owner": entry},
            "marked_reentry_context": {"pending_suffixes": [
                {"role": "recovery_replaced", "resource_jid": "owner", "pending_task_ids": ["replaced"]},
                {"role": "resume_suffix", "resource_jid": "owner", "entry_task_id": "resume"}]},
            "llm_input": {"fault_event": {"focused_resource_jid": "owner", "affected_part_names": ["KET4_Square_4mm"]},
                "observed_runtime_state": {"resources": [entry["recovery_snapshot"]]},
                "part_facts": [{"part_name": "KET4_Square_4mm", "current_location": "M2"}]}}}
    source = tmp_path / "capture" / "observations.json"
    source.parent.mkdir()
    source.write_text(json.dumps(record))
    record.update(source_path=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    return record


def event(name, state):
    return {"outline_id": name, "event_name": name, "resource": "M2", "part_name": "KET4_Square_4mm",
            "expected_end_state": {"resource_state": state}, "rationale": "Supplied test step."}


def reply_for(prompt, schema):
    """Answer any configuration's request: the grounding turn, a whole outline, or one event per turn."""
    if schema.get("name") == "multi_turn_grounding_response":
        return GROUNDED
    if "events" in schema["schema"]["properties"]:
        return {"thought": "Whole outline.", "events": [event("finish", "completed"), event("release", "idle")]}
    if "Recovery Events So Far" in prompt:
        return DONE
    return {"thought": "Next step.", "candidate_events": [event("finish", "completed")]}


def install_provider(monkeypatch, answer=reply_for):
    """Record replies at the same boundary as real provider replies; ``answer`` may block or raise."""
    calls = []

    async def recorded(self, prompt, **kwargs):
        self._last_structured_request = {}
        calls.append(prompt)
        result = await answer(prompt, kwargs["response_format"]) if asyncio.iscoroutinefunction(answer) \
            else answer(prompt, kwargs["response_format"])
        RequestRecords(self._last_structured_request).submit(lambda **payload: result, {"prompt": prompt, **kwargs},
                                                             tool_round=0, attempt=1)
        return result

    monkeypatch.setattr(outline_draft.SavedRecoveryProduct, "ask_llm_structured", recorded)
    return calls


def stand_in_checks(monkeypatch, *, ra=None, cca=None, nonblocking=None):
    """Replace the three checks; each verdict function gets the event name and returns a status or a result."""
    from cais_spade_llm.recovery_framework import cca_nonblocking, cca_safety, ra_feasibility

    monkeypatch.setattr(ra_feasibility, "recorded_cell", lambda observations, floor: object())
    monkeypatch.setattr(cca_safety, "capture_inputs", lambda observations, files, assumptions=None: {})
    monkeypatch.setattr(cca_nonblocking, "capture_inputs", lambda observations: {})

    def check_ra(event, session, prepared, cell):
        verdict = (ra or (lambda name: "FEASIBLE"))(event["event_name"])
        result = verdict if isinstance(verdict, dict) else {
            "feasibility_status": verdict, "conditions": [] if verdict == "FEASIBLE" else [
                {"condition": "resource_state", "status": verdict}], "capability": []}
        return {"resource_jid": event["resource_jid"], "event_name": event["event_name"], "primitive_set": [],
                "reasons": [] if result["feasibility_status"] == "FEASIBLE" else ["the RA fact"], **result}

    def check_cca(event, session, prepared, capture, feasibility):
        status = (cca or (lambda name: "SAFE"))(event["event_name"])
        result = {"resource_jid": event["resource_jid"], "event_name": event["event_name"], "steps": [],
                  "safety_status": status, "reasons": [] if status == "SAFE" else ["the CCA fact"]}
        return {**result, "state_after": {"monitors": {}, "robots": {}}} if status == "SAFE" else result

    def check_completion(session, prepared, capture):
        status = (nonblocking or (lambda: "NONBLOCKING"))()
        return {"nonblocking_status": status, "reasons": [] if status == "NONBLOCKING" else ["M2 cannot finish KET4"],
                "blocked": [], "evidence": {}}

    monkeypatch.setattr(ra_feasibility, "check_event", check_ra)
    monkeypatch.setattr(cca_safety, "check_event", check_cca)
    monkeypatch.setattr(cca_nonblocking, "check_completion", check_completion)


def finished_draft(observations, root, monkeypatch, **flags):
    install_provider(monkeypatch)
    draft = outline_draft.OutlineDraft(observations, [], root=root, **flags)
    asyncio.run(draft.generate())
    assert draft.record["status"] == "draft", draft.record["message"]
    return draft.directory


# The audit's classification.

@pytest.mark.parametrize("result, stages", [
    ({"conditions": [{"status": "INFEASIBLE"}]}, ("INFEASIBLE", "FEASIBLE")),
    # Every primitive that could make the value fails a precondition: the discrete check refuses it.
    ({"conditions": [{"status": "INFEASIBLE", "producers": [
        {"primitive": "grasp_part", "makes_value": True, "status": "INFEASIBLE",
         "reason": "grasp_part needs held_part = null; projected held_part is P1"}]}]}, ("INFEASIBLE", "FEASIBLE")),
    # One primitive makes the value with its preconditions holding; only its pose constraint refuses it.
    ({"conditions": [{"status": "INFEASIBLE", "producers": [
        {"primitive": "grasp_part", "makes_value": True, "status": "INFEASIBLE",
         "reason": "grasp_part needs held_part = null; projected held_part is P1"},
        {"primitive": "move_base", "makes_value": True, "status": "NEEDS_CONTEXT",
         "reason": "move_base: pose has no orientation to check"}]}]}, ("FEASIBLE", "NEEDS_CONTEXT")),
    ({"conditions": [{"status": "INFEASIBLE", "producers": [
        {"primitive": "release_part", "status": "INFEASIBLE", "reason": "release_part: makes held_part null"}]}]},
     ("INFEASIBLE", "FEASIBLE")),
    # A primitive with several alternatives names the one refused by its declared values.
    ({"conditions": [{"status": "INFEASIBLE", "producers": [
        {"primitive": "start_job", "alternative": 1, "makes_value": True, "status": "INFEASIBLE",
         "reason": "start_job (result square) needs selected_program = square; projected selected_program is circle"}]}]},
     ("INFEASIBLE", "FEASIBLE")),
    ({"conditions": [{"status": "INFEASIBLE", "producers": [
        {"primitive": "move_base", "alternative": 0, "makes_value": True, "status": "INFEASIBLE",
         "reason": "move_base: KMR's footprint at {x: -1.500, y: 0.500, yaw: 3.142} overlaps Conveyor on the floor map"}]}]},
     ("FEASIBLE", "INFEASIBLE")),
    # The RA's own record of what refused a producer is read before its reason text.
    ({"conditions": [{"status": "INFEASIBLE", "producers": [
        {"primitive": "place_part", "makes_value": True, "status": "INFEASIBLE", "refused_by": "pose_constraint",
         "reason": "place_part: no configuration reaches the stated pose"},
        {"primitive": "drop_part", "makes_value": True, "status": "INFEASIBLE", "refused_by": "preconditions",
         "reason": "drop_part needs held_part = P1; projected held_part is null"}]}]}, ("FEASIBLE", "INFEASIBLE")),
    ({"conditions": [{"status": "INFEASIBLE", "producers": [
        {"primitive": "drop_part", "makes_value": True, "status": "INFEASIBLE", "refused_by": "preconditions",
         "reason": "drop_part: an unusual reason form"}]}]}, ("INFEASIBLE", "FEASIBLE")),
    ({"capability": [{"check": "capacity", "status": "INFEASIBLE"}]}, ("INFEASIBLE", "FEASIBLE")),
    ({"capability": [{"check": "pick", "status": "INFEASIBLE"}, {"check": "capacity", "status": "FEASIBLE"}]},
     ("FEASIBLE", "INFEASIBLE")),
    # A device that reports itself unavailable refuses every primitive by a shared precondition: discrete.
    ({"capability": [{"check": "primitives", "refused_by": "preconditions", "status": "INFEASIBLE",
                      "reason": "Conveyor's primitives need availability = available"}]}, ("INFEASIBLE", "FEASIBLE")),
    # Taking and putting down a part in one event needs the gripper's grasp and then its release.
    ({"capability": [{"check": "handling", "refused_by": "preconditions", "status": "INFEASIBLE"}]},
     ("INFEASIBLE", "FEASIBLE")),
    ({"capability": [{"check": "a later check", "status": "NEEDS_CONTEXT"}]}, ("NEEDS_CONTEXT", "FEASIBLE")),
    # A refusal from rows the split does not know stays symbolic rather than being lost.
    ({"feasibility_status": "INFEASIBLE"}, ("INFEASIBLE", "FEASIBLE")),
])
def test_ra_results_split_into_symbolic_and_physical_parts(result, stages):
    assert tuple(outline_audit.ra_stages(result).values()) == stages


@pytest.mark.parametrize("symbolic, physical, safety, expected", [
    ("INFEASIBLE", "FEASIBLE", None, ("infeasible", "ra_feasibility")),
    # The first stage that does not pass decides, even when a later stage has a violation.
    ("NEEDS_CONTEXT", "INFEASIBLE", None, ("unresolved", "ra_feasibility")),
    ("FEASIBLE", "INFEASIBLE", None, ("physically_unsupported", "physical_evidence")),
    ("FEASIBLE", "NEEDS_CONTEXT", None, ("unresolved", "physical_evidence")),
    ("FEASIBLE", "FEASIBLE", "UNSAFE", ("unsafe", "cca_safety")),
    ("FEASIBLE", "FEASIBLE", "NEEDS_CONTEXT", ("unresolved", "cca_safety")),
    ("FEASIBLE", "FEASIBLE", "SAFE", None),
])
def test_classification_follows_the_fixed_stage_order(symbolic, physical, safety, expected):
    rows = [{"status": symbolic}]
    capability = [{"check": "pick", "status": physical}]
    # Inside a stage a violation beats missing evidence.
    if symbolic == "INFEASIBLE":
        rows.append({"status": "NEEDS_CONTEXT"})
    ra = {"feasibility_status": "FEASIBLE", "conditions": rows, "capability": capability}
    cca = None if safety is None else {"safety_status": safety}
    assert outline_audit.classify_event(ra, cca) == expected


@pytest.mark.parametrize("checks, outcome, stage, first", [
    ({}, "passed", None, None),
    ({"ra": lambda name: "INFEASIBLE" if name == "release" else "FEASIBLE"}, "infeasible", "ra_feasibility", 2),
    ({"ra": lambda name: {"feasibility_status": "INFEASIBLE", "conditions": [],
                          "capability": [{"check": "place", "status": "INFEASIBLE"}]}},
     "physically_unsupported", "physical_evidence", 1),
    ({"cca": lambda name: "UNSAFE"}, "unsafe", "cca_safety", 1),
    ({"nonblocking": lambda: "BLOCKING"}, "no_nominal_continuation", "nonblocking", None),
])
def test_audit_classifies_a_finished_draft_by_its_first_failing_check(observations, tmp_path, monkeypatch,
                                                                       checks, outcome, stage, first):
    directory = finished_draft(observations, tmp_path, monkeypatch, single_pass_outline=True)
    before = sorted(path.name for path in directory.iterdir())
    stand_in_checks(monkeypatch, **checks)
    audit = outline_audit.audit_draft(directory, observations, [], [])
    assert (audit["status"], audit["outcome"], audit["failing_stage"], audit["first_failing_event"]) == (
        "audited", outcome, stage, first)
    assert audit["event_count"] == 2 and audit["replay_mismatches"] == []
    assert audit["audited_events"] == (first or 2)
    # The audit only adds its own record beside the draft.
    assert sorted(path.name for path in directory.iterdir()) == sorted([*before, "audit.json"])
    assert json.loads((directory / "audit.json").read_text())["outcome"] == outcome


def test_a_nonblocking_search_that_runs_out_is_unresolved_not_a_refusal(observations, tmp_path, monkeypatch):
    directory = finished_draft(observations, tmp_path, monkeypatch)

    def ran_out():
        raise RuntimeError("the composition search reached its 900 s budget")

    stand_in_checks(monkeypatch, nonblocking=ran_out)
    audit = outline_audit.audit_draft(directory, observations, [], [])
    assert (audit["outcome"], audit["failing_stage"]) == ("unresolved", "nonblocking")
    assert "900 s budget" in audit["nonblocking"]["error"]


def test_an_unfinished_draft_did_not_finish_and_keeps_its_stop_cause(observations, tmp_path, monkeypatch):
    install_provider(monkeypatch, lambda prompt, schema: GROUNDED if schema.get("name") ==
                     "multi_turn_grounding_response" else {"thought": "Unreadable."})
    draft = outline_draft.OutlineDraft(observations, [], root=tmp_path, single_pass_outline=True)
    asyncio.run(draft.generate())
    audit = outline_audit.audit_draft(draft.directory, observations, [], [])
    assert (audit["status"], audit["outcome"]) == ("audited", "did_not_finish")
    assert audit["stop_cause"]["cause"] == "error"


# The batch runner.

def test_configurations_are_labels_for_exact_draft_flags():
    checks = {"check_ra_feasibility": True, "check_cca_safety": True, "check_nonblocking": True}
    rules = {"safety_rules_to_llm": "prompt"}
    assert {
        "LLM, single-pass": {"single_pass_outline": True}, "LLM + rules, single-pass": {"single_pass_outline": True, **rules},
        "LLM, incremental": {}, "LLM + rules, incremental": rules,
        "Framework, single-pass": {"single_pass_outline": True, **checks}, "Framework, incremental": checks} == outline_experiments.CONFIGURATIONS


@pytest.mark.parametrize("change, message", [
    ({"scenarios": []}, "at least one failure scenario"),
    ({"scenarios": ["Unknown"]}, "No retained capture"),
    ({"attempts": {"Framework, incremental": 21}}, "from 0 to 20"),
    ({"attempts": {}}, "at least one configuration"),
    ({"model": " "}, "model"),
    ({"model": "gpt-4o", "reasoning_effort": "medium"}, "unsupported for gpt-4o"),
    ({"reasoning_effort": None}, "Choose a reasoning effort for gpt-5.4"),
    ({"max_turns": 0}, "Max turns"),
])
def test_batch_settings_are_checked_before_anything_runs(change, message):
    settings = {"scenarios": [SCENARIO], **deepcopy(outline_experiments.DEFAULT_SETTINGS), **change}
    with pytest.raises(ValueError, match=message):
        outline_experiments.checked_settings(settings, {SCENARIO: "observations.json"})


def test_the_plan_shuffles_the_configurations_still_due_in_each_repetition():
    attempts = {"LLM, single-pass": 1, "LLM, incremental": 2, "Framework, single-pass": 1, "Framework, incremental": 2}
    plan = outline_experiments.planned_attempts(["A", "B"], attempts, seed=7)
    assert plan == outline_experiments.planned_attempts(["A", "B"], attempts, seed=7)
    assert len(plan) == 12 and [row["attempt_id"] for row in plan][:2] == ["attempt_001", "attempt_002"]
    first = [row["configuration"] for row in plan if row["scenario"] == "A" and row["repetition"] == 1]
    second = [row["configuration"] for row in plan if row["scenario"] == "A" and row["repetition"] == 2]
    assert sorted(first) == sorted(attempts) and sorted(second) == ["Framework, incremental", "LLM, incremental"]
    orders = {tuple(row["configuration"] for row in outline_experiments.planned_attempts(["A"], attempts, seed))
              for seed in range(20)}
    assert len(orders) > 1


@pytest.fixture
def frozen(monkeypatch, tmp_path):
    """Freeze one small input file instead of the whole repository; return it so a test can change it."""
    path = tmp_path / "frozen_input.json"
    path.write_text("{}")
    monkeypatch.setattr(recovery_evaluation, "frozen_files", lambda captures: {str(path): recovery_evaluation.file_hash(path)})
    monkeypatch.setattr(recovery_evaluation, "implementation_inventory", lambda: [])
    monkeypatch.setattr(recovery_evaluation, "_implementation_files", lambda root: set())
    monkeypatch.setattr(outline_experiments, "changed_since_start", lambda files: [])
    return path


def run_batch(runner, settings, observations, *, during=None):
    async def run():
        await runner.start(settings, captures={SCENARIO: observations["source_path"]})
        if during is not None:
            await during(runner)
        await runner._task
    asyncio.run(run())
    return outline_experiments.read_batch(runner.root / runner.snapshot()["batch_id"] / "batch.json")


def test_a_batch_runs_and_audits_every_attempt_and_keeps_every_record(observations, tmp_path, monkeypatch, frozen):
    monkeypatch.setattr(recovery_observations, "read_observations", lambda path: deepcopy(observations))
    calls = install_provider(monkeypatch)
    stand_in_checks(monkeypatch)
    runner = outline_experiments.OutlineExperiments(root=tmp_path / "experiments")
    settings = {"scenarios": [SCENARIO], "attempts": dict.fromkeys(outline_experiments.CONFIGURATIONS, 1),
                "model": "gpt-5.4", "reasoning_effort": "medium", "max_turns": 12}
    batch = run_batch(runner, settings, observations)
    assert batch["status"] == "finished" and batch["settings"]["max_turns"] == 12
    assert batch["held_constant"]["token_budget"] == "not set"
    assert batch["configurations"] == outline_experiments.CONFIGURATIONS
    rows = {row["configuration"]: row for row in batch["rows"]}
    assert set(rows) == set(outline_experiments.CONFIGURATIONS)
    assert {row["status"] for row in rows.values()} == {"finished"}
    assert {row["outcome"] for row in rows.values()} == {"passed"}
    assert (rows["LLM, single-pass"]["rejections_ra_feasibility"], rows["Framework, incremental"]["rejections_ra_feasibility"]) == (
        "–", 0)
    assert rows["Framework, single-pass"]["rewrites"] == 0 and rows["Framework, incremental"]["revisions"] == 0
    assert rows["LLM, incremental"]["outline_length"] == 1 and rows["LLM, single-pass"]["outline_length"] == 2
    assert {row["model"] for row in rows.values()} == {"gpt-5.4"}
    for row in rows.values():
        directory = Path(row["draft_path"]).parent
        assert directory.is_relative_to(runner.root / batch["batch_id"] / "attempts")
        assert (directory / "audit.json").is_file()
        record = json.loads((directory / "draft.json").read_text())
        assert record["max_turns"] == 12 and record["llm_settings"]["model_source"] == "override"
        flags = outline_experiments.CONFIGURATIONS[row["configuration"]]
        assert record["single_pass_outline"] is flags.get("single_pass_outline", False)
        assert record["ra_feasibility_check"] is flags.get("check_ra_feasibility", False)
        assert record["safety_rules_to_llm"] == flags.get("safety_rules_to_llm", "hidden")
    # Grounding plus two turns for each incremental attempt; grounding plus one outline for single-pass.
    assert len(calls) == 3 * 3 + 3 * 2
    summary = outline_experiments.summarize(batch["rows"])
    pooled = [row for row in summary["outcomes"] if row["scenario"] == "All scenarios"]
    assert [row["configuration"] for row in pooled] == list(outline_experiments.CONFIGURATIONS)
    assert {row["passed"] for row in pooled} == {1}
    assert [row["configuration"] for row in summary["rejections"] if row["scenario"] == "All scenarios"] == [
        "Framework, single-pass", "Framework, incremental"]
    text = outline_experiments.rows_csv(batch["rows"])
    assert text.splitlines()[0].split(",")[:3] == ["attempt_id", "scenario", "configuration"]
    assert len(text.splitlines()) == 1 + len(outline_experiments.CONFIGURATIONS)


def test_stop_ends_the_running_attempt_and_leaves_the_rest_not_started(observations, tmp_path, monkeypatch, frozen):
    monkeypatch.setattr(recovery_observations, "read_observations", lambda path: deepcopy(observations))
    entered, release = threading.Event(), threading.Event()

    def blocking(prompt, schema):
        if schema.get("name") != "multi_turn_grounding_response":
            entered.set()
            release.wait(10)
        return reply_for(prompt, schema)

    install_provider(monkeypatch, blocking)
    runner = outline_experiments.OutlineExperiments(root=tmp_path / "experiments")
    settings = {"scenarios": [SCENARIO], "attempts": {"LLM, incremental": 3}, "model": "gpt-5.4",
                "reasoning_effort": "medium", "max_turns": 12}

    async def stop_once_generating(runner):
        await asyncio.to_thread(entered.wait, 10)
        runner.stop()
        release.set()

    batch = run_batch(runner, settings, observations, during=stop_once_generating)
    assert batch["status"] == "stopped"
    assert [row["status"] for row in batch["rows"]] == ["finished", "not started", "not started"]
    assert (batch["rows"][0]["draft_status"], batch["rows"][0]["outcome"]) == ("stopped", "did_not_finish")


def test_a_changed_input_stops_the_batch_after_the_attempt_it_changed_under(observations, tmp_path, monkeypatch, frozen):
    monkeypatch.setattr(recovery_observations, "read_observations", lambda path: deepcopy(observations))

    def answer(prompt, schema):
        frozen.write_text('{"changed": true}')
        return reply_for(prompt, schema)

    install_provider(monkeypatch, answer)
    runner = outline_experiments.OutlineExperiments(root=tmp_path / "experiments")
    settings = {"scenarios": [SCENARIO], "attempts": {"LLM, single-pass": 2}, "model": "gpt-5.4",
                "reasoning_effort": "medium", "max_turns": 12}
    batch = run_batch(runner, settings, observations)
    assert batch["status"] == "inputs_changed"
    assert [row["status"] for row in batch["rows"]] == ["finished", "not started"]
    assert batch["rows"][0]["message"].startswith("Inputs changed during this attempt")


def test_an_attempt_that_cannot_start_is_recorded_as_failed(observations, tmp_path, monkeypatch, frozen):
    def unreadable(path):
        raise ValueError("capture is not readable")

    monkeypatch.setattr(recovery_observations, "read_observations", unreadable)
    runner = outline_experiments.OutlineExperiments(root=tmp_path / "experiments")
    settings = {"scenarios": [SCENARIO], "attempts": {"Framework, incremental": 2}, "model": "gpt-5.4",
                "reasoning_effort": "medium", "max_turns": 12}
    batch = run_batch(runner, settings, observations)
    assert batch["status"] == "failed"
    assert [(row["status"], row["outcome"]) for row in batch["rows"]] == [
        ("failed", "did_not_finish"), ("not started", "not recorded")]
    assert outline_experiments.summarize(batch["rows"])["outcomes"][0]["started"] == 1


def test_code_changed_after_the_ui_started_refuses_a_batch(observations, tmp_path, monkeypatch, frozen):
    monkeypatch.setattr(outline_experiments, "changed_since_start", lambda files: ["cais_spade_llm/x.py"])
    runner = outline_experiments.OutlineExperiments(root=tmp_path / "experiments")
    settings = {"scenarios": [SCENARIO], "attempts": {"Framework, incremental": 1}, "model": "gpt-5.4",
                "reasoning_effort": "medium", "max_turns": 12}
    with pytest.raises(ValueError, match="restart the UI"):
        asyncio.run(runner.start(settings, captures={SCENARIO: observations["source_path"]}))
    assert not (tmp_path / "experiments").exists()


def test_a_batch_left_running_reads_as_interrupted(tmp_path):
    path = tmp_path / "batch_1" / "batch.json"
    path.parent.mkdir()
    path.write_text(json.dumps({"batch_id": "batch_1", "status": "running", "plan": [{}, {}], "rows": [
        {"attempt_id": "attempt_001", "status": "running", "outcome": "not recorded"}]}))
    batch = outline_experiments.read_batch(path)
    assert batch["status"] == "interrupted"
    assert (batch["rows"][0]["status"], batch["rows"][0]["outcome"]) == ("interrupted", "did_not_finish")
    assert outline_experiments.read_batch(path, live="batch_1")["status"] == "running"
    assert outline_experiments.list_batches(tmp_path)[0]["status"] == "interrupted"
    # Reading never rewrites the record.
    assert json.loads(path.read_text())["status"] == "running"


def test_rejections_are_counted_by_the_check_that_refused():
    symbolic = {"feasibility_status": "INFEASIBLE", "conditions": [{"status": "INFEASIBLE"}]}
    physical = {"feasibility_status": "INFEASIBLE", "capability": [{"check": "pick", "status": "INFEASIBLE"}]}
    full = {"ra_feasibility_check": True,
            "ra_feasibility_checks": [{"results": [symbolic, physical, {"feasibility_status": "FEASIBLE"}]}],
            "cca_safety_checks": [{"results": [{"safety_status": "UNSAFE"}, {"safety_status": "SAFE"}]}],
            "cca_nonblocking_checks": [{"nonblocking_status": "BLOCKING"}, {"nonblocking_status": "NONBLOCKING"}]}
    assert outline_experiments.rejections(full) == {"ra_feasibility": 1, "physical_evidence": 1, "cca_safety": 1,
                                                    "nonblocking": 1}
    single = {"ra_feasibility_check": True, "single_pass_outline": True, "single_pass_checks": [
        {"accepted": False, "checks": [{"event_index": 0, "ra_feasibility": [physical]}]},
        {"accepted": False, "checks": [{"event_index": 0, "ra_feasibility": [{"feasibility_status": "FEASIBLE"}],
                                        "cca_safety": [{"safety_status": "UNSAFE"}]}]},
        {"accepted": False, "checks": [{"event_index": 0}, {"completion": {"nonblocking_status": "BLOCKING"}}]},
        {"accepted": True, "checks": []}]}
    assert outline_experiments.rejections(single) == {"ra_feasibility": 0, "physical_evidence": 1, "cca_safety": 1,
                                                      "nonblocking": 1}
    assert outline_experiments.rejections({"ra_feasibility_check": False}) is None


def test_a_batch_saved_under_an_earlier_configuration_name_keeps_its_rejections():
    counted = {f"rejections_{check}": 1 for check in outline_experiments.REJECTION_CHECKS}
    rows = [{"attempt_id": "attempt_001", "scenario": SCENARIO, "configuration": "Full framework", "status": "finished",
             "outcome": "passed", **counted, "revisions": 2, "rewrites": "–"},
            {"attempt_id": "attempt_002", "scenario": SCENARIO, "configuration": "Incremental LLM", "status": "finished",
             "outcome": "unsafe", **dict.fromkeys(counted, "–"), "revisions": "–", "rewrites": "–"}]
    summary = outline_experiments.summarize(rows)
    assert [row["configuration"] for row in summary["outcomes"]] == ["Full framework", "Full framework",
                                                                     "Incremental LLM", "Incremental LLM"]
    assert [(row["configuration"], row["scenario"], row["cca_safety"], row["revisions or rewrites"])
            for row in summary["rejections"]] == [("Full framework", SCENARIO, 1, 2),
                                                  ("Full framework", "All scenarios", 1, 2)]

def test_a_row_says_not_recorded_for_what_its_records_lack(tmp_path):
    directory = tmp_path / "draft_x"
    directory.mkdir()
    (directory / "draft.json").write_text(json.dumps({"status": "draft", "transition_trace": [{}], "usage": []}))
    row = outline_experiments.attempt_row({"attempt_id": "attempt_001", "scenario": SCENARIO,
                                           "configuration": "LLM, incremental", "repetition": 1,
                                           "status": "finished"}, directory)
    assert (row["outcome"], row["tokens"], row["outline_length"], row["rewrites"]) == (
        "not audited", "not recorded", 1, "–")


def test_audit_missing_audits_finished_attempts_once_their_inputs_still_match(observations, tmp_path, monkeypatch,
                                                                            frozen):
    monkeypatch.setattr(recovery_observations, "read_observations", lambda path: deepcopy(observations))
    install_provider(monkeypatch)
    stand_in_checks(monkeypatch)
    audit_draft = outline_audit.audit_draft
    # As when the UI stopped between an attempt's generation and its audit.
    monkeypatch.setattr(outline_audit, "audit_draft", lambda *args, **kwargs: None)
    runner = outline_experiments.OutlineExperiments(root=tmp_path / "experiments")
    settings = {"scenarios": [SCENARIO], "attempts": {"LLM, single-pass": 1}, "model": "gpt-5.4",
                "reasoning_effort": "medium", "max_turns": 12}
    batch = run_batch(runner, settings, observations)
    assert batch["rows"][0]["outcome"] == "not audited"
    monkeypatch.setattr(outline_audit, "audit_draft", audit_draft)

    async def audit():
        await runner.audit_missing(batch["batch_id"])
        await runner._task

    asyncio.run(audit())
    audited = outline_experiments.read_batch(runner.root / batch["batch_id"] / "batch.json")
    assert audited["rows"][0]["outcome"] == "passed" and audited["status"] == "finished"
    frozen.write_text('{"changed": true}')
    with pytest.raises(ValueError, match="Frozen evaluation input changed"):
        asyncio.run(runner.audit_missing(batch["batch_id"]))


def test_a_second_start_while_the_first_is_freezing_its_inputs_is_refused(observations, tmp_path, monkeypatch, frozen):
    monkeypatch.setattr(recovery_observations, "read_observations", lambda path: deepcopy(observations))
    install_provider(monkeypatch)
    runner = outline_experiments.OutlineExperiments(root=tmp_path / "experiments")
    settings = {"scenarios": [SCENARIO], "attempts": {"LLM, single-pass": 1}, "model": "gpt-5.4",
                "reasoning_effort": "medium", "max_turns": 12}

    async def run():
        first = asyncio.create_task(runner.start(settings, captures={SCENARIO: observations["source_path"]}))
        await asyncio.sleep(0)
        with pytest.raises(ValueError, match="already running"):
            await runner.start(settings, captures={SCENARIO: observations["source_path"]})
        await first
        await runner._task

    asyncio.run(run())
    assert len(list((tmp_path / "experiments").glob("batch_*"))) == 1


def test_an_attempt_interrupted_after_its_outline_waits_for_its_audit(observations, tmp_path, monkeypatch, frozen):
    monkeypatch.setattr(recovery_observations, "read_observations", lambda path: deepcopy(observations))
    install_provider(monkeypatch)
    stand_in_checks(monkeypatch)
    audit_draft = outline_audit.audit_draft
    monkeypatch.setattr(outline_audit, "audit_draft", lambda *args, **kwargs: None)
    runner = outline_experiments.OutlineExperiments(root=tmp_path / "experiments")
    settings = {"scenarios": [SCENARIO], "attempts": {"LLM, single-pass": 1}, "model": "gpt-5.4",
                "reasoning_effort": "medium", "max_turns": 12}
    batch = run_batch(runner, settings, observations)
    # As left on disk when the UI stopped during the attempt's audit.
    path = runner.root / batch["batch_id"] / "batch.json"
    saved = json.loads(path.read_text())
    saved["status"], saved["rows"][0]["status"] = "running", "running"
    path.write_text(json.dumps(saved))
    interrupted = outline_experiments.read_batch(path)
    assert (interrupted["status"], interrupted["rows"][0]["status"], interrupted["rows"][0]["outcome"]) == (
        "interrupted", "interrupted", "not audited")
    monkeypatch.setattr(outline_audit, "audit_draft", audit_draft)

    async def audit():
        await runner.audit_missing(batch["batch_id"])
        await runner._task

    asyncio.run(audit())
    audited = outline_experiments.read_batch(path)
    assert (audited["rows"][0]["status"], audited["rows"][0]["outcome"]) == ("interrupted", "passed")
    assert outline_experiments.summarize(audited["rows"])["outcomes"][0]["started"] == 1


def test_stop_while_an_attempt_is_prepared_starts_nothing(observations, tmp_path, monkeypatch, frozen):
    preparing, release = threading.Event(), threading.Event()

    def slow_read(path):
        preparing.set()
        release.wait(10)
        return deepcopy(observations)

    monkeypatch.setattr(recovery_observations, "read_observations", slow_read)
    calls = install_provider(monkeypatch)
    runner = outline_experiments.OutlineExperiments(root=tmp_path / "experiments")
    settings = {"scenarios": [SCENARIO], "attempts": {"LLM, incremental": 2}, "model": "gpt-5.4",
                "reasoning_effort": "medium", "max_turns": 12}

    async def stop_while_preparing(runner):
        await asyncio.to_thread(preparing.wait, 10)
        runner.stop()
        release.set()

    batch = run_batch(runner, settings, observations, during=stop_while_preparing)
    assert batch["status"] == "stopped" and calls == []
    assert [row["status"] for row in batch["rows"]] == ["not started", "not started"]
    assert outline_experiments.list_batches(runner.root)[0]["done"] == 0
    assert not (runner.root / batch["batch_id"] / "attempts").exists()


def test_stop_during_the_last_attempt_records_a_stopped_batch(observations, tmp_path, monkeypatch, frozen):
    monkeypatch.setattr(recovery_observations, "read_observations", lambda path: deepcopy(observations))
    entered, release = threading.Event(), threading.Event()

    def blocking(prompt, schema):
        if schema.get("name") != "multi_turn_grounding_response":
            entered.set()
            release.wait(10)
        return reply_for(prompt, schema)

    install_provider(monkeypatch, blocking)
    runner = outline_experiments.OutlineExperiments(root=tmp_path / "experiments")
    settings = {"scenarios": [SCENARIO], "attempts": {"LLM, single-pass": 1}, "model": "gpt-5.4",
                "reasoning_effort": "medium", "max_turns": 12}

    async def stop_once_generating(runner):
        await asyncio.to_thread(entered.wait, 10)
        runner.stop()
        release.set()

    batch = run_batch(runner, settings, observations, during=stop_once_generating)
    assert batch["status"] == "stopped"
    assert outline_experiments.list_batches(runner.root)[0]["done"] == 1


@pytest.mark.parametrize("record, revisions, rewrites", [
    # Each outline after the first rewrites a refused one; the last refusal may end the attempt unrevised.
    ({"single_pass_outline": True, "single_pass_checks": [{"accepted": False}, {"accepted": False},
                                                          {"accepted": True}]}, "–", 2),
    ({"single_pass_outline": True, "single_pass_checks": [{"accepted": False}] * 3}, "–", 2),
    ({"turn_decision_order": ["need_context", "need_revision", "need_next_task", "need_revision"]}, 1, "–"),
    ({"turn_decision_order": ["need_revision", "need_context", "need_revision", "outline_ready"]}, 2, "–"),
    # Records from before the decision order was kept count their refused turns.
    ({"turn_decisions": {"need_revision": 2}}, 2, "–"),
])
def test_revisions_and_rewrites_count_only_proposals_made_after_a_refusal(tmp_path, record, revisions, rewrites):
    directory = tmp_path / "draft_x"
    directory.mkdir()
    (directory / "draft.json").write_text(json.dumps({"status": "draft_incomplete", "ra_feasibility_check": True,
                                                      **record}))
    row = outline_experiments.attempt_row({"attempt_id": "attempt_001", "status": "finished"}, directory)
    assert (row["revisions"], row["rewrites"]) == (revisions, rewrites)
