from __future__ import annotations

"""Verify diagnostic progression through the existing multi_turn orchestration."""

import asyncio
import hashlib
import json
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes import (
    multi_turn, multi_turn_context, multi_turn_diagnostic, multi_turn_validated_event,
)
from cais_spade_llm.recovery_framework import outline_draft


@pytest.fixture
def captured(tmp_path):
    variables = {name: {"scope": scope, "type": kind} for name, scope, kind in (
        ("resource_state", "resource", "string"), ("resource_location", "resource", "string"),
        ("held_part", "resource", ["string", "null"]), ("part_name", "resource", ["string", "null"]),
        ("job_state", "resource", "string"), ("selected_program", "resource", ["string", "null"]),
        ("active_job", "resource", ["string", "null"]), ("part_state", "part", "string"),
        ("part_location", "part", "string"))}
    events = [{"event_name": "machine_part", "parameter_bindings": {
        "part_name": {"equals": part}, "process": {"equals": "trim"}, "result": {"equals": result}},
        "guards": {"resource_state": {"equals": "loaded"}, "part_name": {"equals_from_param": "part_name"}},
        "updates": {"resource_state": {"set": "completed"}},
        "capability_transition": {"source": {"part_location": "M2"}},
        "product_effects": {"processCompleted": [{"process": "trim", "result": {"set_from_param": "result"}}]}}
        for part, result in (("KET4_Square_4mm", "square"), ("RGOCG4-50_Round_4mm", "circle"))]
    resources = {jid: {"resource_jid": jid, "resource_type": "resource",
        "static_capabilities": {"reachability": ["M1", "M2"], "gripper_reach": 99,
            "capabilities": ["place_release"]},
        "recovery_snapshot": {"resource_jid": jid, "resource_state": "loaded" if jid == "machine" else "idle",
            "resource_location": "M2", "held_part": None, "part_name": "RGOCG4-50_Round_4mm" if jid == "machine" else None,
            "job_state": "running", "active_job": "private_job"},
        "recovery_des_model": {"state_variables": deepcopy(variables), "events": deepcopy(events) if jid == "machine" else []}}
        for jid in ("machine", "arm")}
    record = {"scenario": "Machining breakdown during part processing", "run_id": "supplied_fixture",
        "captured_at": "recorded-time", "jids": {"M2": "machine"},
        "pending_nodes": [{"id": identifier, "function_name": "machine_part", "params": {
            "resource_id": location, "part_name": part, "process": "trim", "result": result}}
            for identifier, location, part, result in (
                ("environment_21", "M1", "KET4_Square_4mm", "square"),
                ("environment_23", "M2", "RGOCG4-50_Round_4mm", "circle"))],
        "prepared_recovery_request": {"ra_jid": "machine", "recovery_resources": resources,
            "marked_reentry_context": {"pending_suffixes": [
                {"role": "recovery_replaced", "pending_task_ids": ["environment_21"]},
                {"role": "resume_suffix", "resource_jid": "machine", "entry_task_id": "environment_23"}]},
            "llm_input": {"fault_event": {"focused_resource_jid": "machine"},
                "observed_runtime_state": {"resources": [row["recovery_snapshot"] for row in resources.values()]},
                "part_facts": [{"part_name": part, "part_state": "loaded", "current_location": location}
                    for part, location in (("KET4_Square_4mm", "M1"), ("RGOCG4-50_Round_4mm", "M2"))]}}}
    path = tmp_path / "observations.json"
    path.write_text(json.dumps(record))
    record.update(source_path=str(path), source_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    return record


def event(identifier, jid, part, **end):
    return {"outline_id": identifier, "event_name": identifier, "resource_jid": jid,
        "part_name": part, "expected_end_state": end, "rationale": "Supplied verification assumption; not validated."}


def sequence():
    return [
        event("pause", "machine", None, job_state="paused"),
        event("stage", "arm", "RGOCG4-50_Round_4mm", resource_state="ready", held_part=None,
              part_state="placed", part_location="M2 staging tray"),
        event("transfer", "arm", "KET4_Square_4mm", resource_state="ready", held_part=None,
              part_state="loaded", part_location="M2"),
        event("finish", "machine", "KET4_Square_4mm", resource_state="completed", job_state="completed",
              part_name="KET4_Square_4mm", selected_program="square"),
        event("unload", "arm", "KET4_Square_4mm", resource_state="ready", held_part=None,
              part_state="placed", part_location="Conveyor"),
        event("restore", "arm", "RGOCG4-50_Round_4mm", resource_state="ready", held_part=None,
              part_state="loaded", part_location="M2"),
    ]


def test_real_multi_turn_retrieval_and_six_events_bypass_all_admission(captured, tmp_path, monkeypatch):
    from cais_spade_llm.recovery_framework import primitive_composition

    forbidden = Mock(side_effect=AssertionError("Diagnostic must not invoke validators or execution"))
    monkeypatch.setattr(primitive_composition, "start_composition_attempt", forbidden)
    replies = [{"thought": "Use recorded facts.", "decision": "grounded", "observations": []},
        {"thought": "Request facts.", "decision": "need_context",
         "context_requests": ["/resources/machine/snapshot", "/safety_rules"]},
        *[{"thought": "Propose one task-level event.", "candidate_events": [row]} for row in sequence()],
        {"thought": "The failure is overcome.", "decision": "recovery_complete"}]
    provider = AsyncMock(side_effect=replies)
    monkeypatch.setattr(outline_draft.SavedRecoveryProduct, "ask_llm_structured", provider)
    execute = AsyncMock(wraps=multi_turn.execute_multi_turn_recovery)
    monkeypatch.setattr(multi_turn, "execute_multi_turn_recovery", execute)
    before = deepcopy(captured)
    draft = outline_draft.OutlineDraft(captured, [], root=tmp_path / "drafts")
    asyncio.run(draft.generate())
    assert draft.record["status"] == "draft", draft.record["message"]
    assert len(draft.record["transition_trace"]) == 6
    assert draft.record["provider_calls"] == provider.await_count == 9
    assert execute.await_count >= 6
    assert draft.record["nominal_handoff"]["complete"] is True
    assert draft.record["nominal_handoff"]["assumptions"]
    assert captured == before
    assert "/resources/machine/snapshot" in draft.record["retrievals"]
    assert "checks" not in draft.record
    assert draft.record["execution_authorized"] is False
    forbidden.assert_not_called()
    for call in provider.call_args_list:
        prompt = call.kwargs["prompt"] if "prompt" in call.kwargs else call.args[0]
        assert not any(token in prompt for token in ("gripper_reach", "place_release", "active_job", "private_job"))


@pytest.mark.parametrize("reverse", [False, True])
def test_llm_choice_is_committed_and_candidates_stay_isolated(captured, reverse):
    prepared, _ = outline_draft._inputs(captured)
    session = multi_turn.build_multi_turn_session_seed(prepared)
    candidates = [event("noop", "machine", None, resource_state="loaded"), sequence()[0]]
    if reverse:
        candidates.reverse()
    before = deepcopy(prepared)
    choice = next(index for index, row in enumerate(candidates) if row["outline_id"] == "pause")
    status, audit = asyncio.run(multi_turn_validated_event.handle_validated_candidates(
        session=session, prepared=prepared, planner=SimpleNamespace(product_agent=SimpleNamespace()),
        response={"thought": "Compare supplied alternatives.", "candidate_events": candidates,
                  "selected_candidate_index": choice}))
    assert status == "need_next_task"
    assert audit["llm_selected_candidate_index"] == choice
    assert len(session["accepted_outline_prefix"]) == 1
    assert session["accepted_outline_prefix"][0]["event_name"] == "pause"
    assert session["symbolic_resources"]["machine"]["job_state"] == "paused"
    assert prepared == before
    assert audit["candidate_events"] == candidates and audit["transition_validation"]["status"] == "not_checked"


def test_nothing_checks_the_llm_reply_or_its_chosen_event(captured):
    prepared, _ = outline_draft._inputs(captured)
    session = multi_turn.build_multi_turn_session_seed(prepared)

    def propose(response):
        return asyncio.run(multi_turn_validated_event.handle_validated_candidates(
            session=session, prepared=prepared, planner=SimpleNamespace(product_agent=SimpleNamespace()),
            response=response))[0]

    candidates = [event("noop", "machine", None, resource_state="loaded"), sequence()[0]]
    # A reply code cannot read raises; the draft records the error.
    with pytest.raises(KeyError):
        propose({"thought": "Two candidates, no choice.", "candidate_events": candidates})
    # A repeat, a variable the resource does not declare and a stated params are applied as stated.
    for row in (candidates[0], candidates[0], event("slot", "machine", None, zone_1_part="KET4_Square_4mm"),
                {**sequence()[0], "params": {"x": 1.0}}):
        assert propose({"thought": "Choose it.", "candidate_events": [row]}) == "need_next_task"
    assert [row["event_name"] for row in session["accepted_outline_prefix"]] == ["noop", "noop", "slot", "pause"]
    assert session["accepted_outline_prefix"][2]["expected_end_state"] == {"zone_1_part": "KET4_Square_4mm"}
    # Recovery is complete when the LLM says so, whatever else the reply carries.
    assert propose({"thought": "Done.", "decision": "recovery_complete", "candidate_events": candidates}) == "outline_ready"


def test_handoff_requires_replaced_work_and_restored_part(captured):
    prepared, _ = outline_draft._inputs(captured)
    session = multi_turn.build_multi_turn_session_seed(prepared)
    assert multi_turn_diagnostic.handoff(session, prepared)["complete"] is False
    for row in sequence()[:-1]:
        multi_turn_diagnostic.apply_tentative_effects(multi_turn_diagnostic.event_record(row, session, prepared),
                                                      session, prepared)
        assert multi_turn_diagnostic.handoff(session, prepared)["complete"] is False
    multi_turn_diagnostic.apply_tentative_effects(multi_turn_diagnostic.event_record(sequence()[-1], session, prepared),
                                                  session, prepared)
    assert multi_turn_diagnostic.handoff(session, prepared)["complete"] is True
    assert session["symbolic_parts"]["KET4_Square_4mm"]["processCompleted"] == [{"process": "trim", "result": "square"}]


def test_diagnostic_flag_defaults_off(captured):
    prepared, _ = outline_draft._inputs(captured)
    prepared["recovery_session"].pop("outline_diagnostic_bypass")
    assert multi_turn.build_multi_turn_session_seed(prepared)["outline_diagnostic_bypass"] is False


def test_a_position_is_a_pose_condition(captured):
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes.multi_turn_prompts import (
        _accepted_transition_prefix_for_prompt,
    )

    prepared, _ = outline_draft._inputs(captured)
    prepared["recovery_resources"]["arm"]["recovery_des_model"]["events"] = [{"event_name": "move_to_location",
        "updates": {"resource_location": {"set_from_param": "observed"}},
        "parameter_bindings": {name: {"required": True, "type": "number"} for name in ("x", "y", "yaw")}}]
    multi_turn_diagnostic.declare_derived_conditions(prepared, {})
    session = multi_turn.build_multi_turn_session_seed(prepared)
    drive = event("drive", "arm", None, resource_state="idle", pose={"x": -1.2, "y": 1.15, "yaw": -1.5708})

    def propose(row):
        return asyncio.run(multi_turn_validated_event.handle_validated_candidates(
            session=session, prepared=prepared, planner=SimpleNamespace(product_agent=SimpleNamespace()),
            response={"thought": "Explore a pose.", "candidate_events": [row]}))[0]

    # A position is the pose condition; an event carries no params.
    assert propose({**drive, "params": {"x": -1.2}}) == "need_next_task"
    assert session["symbolic_resources"]["arm"]["current_pose"] == {"x": -1.2, "y": 1.15, "yaw": -1.5708}
    elsewhere = {**drive, "outline_id": "elsewhere",
                 "expected_end_state": {**drive["expected_end_state"], "pose": {"x": -1.0, "y": 1.15, "yaw": -1.5708}}}
    assert propose(elsewhere) == "need_next_task"
    assert session["symbolic_resources"]["arm"]["current_pose"] == {"x": -1.0, "y": 1.15, "yaw": -1.5708}
    assert all("params" not in row for row in _accepted_transition_prefix_for_prompt(session["accepted_outline_prefix"]))


def _arm(jid, state, location, held, pose):
    variables = {name: {"scope": scope, "type": kind} for name, scope, kind in (
        ("resource_state", "resource", "string"), ("resource_location", "resource", "string"),
        ("held_part", "resource", ["string", "null"]), ("part_state", "part", "string"),
        ("part_location", "part", ["string", "null"]))}
    approach = {"event_name": "place_approach", "parameter_bindings": {"destination_location": {"location_type": "reachable_location"}},
        "guards": {"resource_state": {"equals": "picked"}, "held_part": {"equals_from_param": "part_name"}},
        "updates": {"resource_state": {"set": "positioned"}, "part_state": {"set": "in_transit"}}}
    return {"resource_jid": jid, "resource_type": "robot",
        "static_capabilities": {"reachability": ["cell", "buffer"], "gripper_reach": {"frame": "world",
            "origin_pose": {"x": pose["x"], "y": pose["y"], "z": 1.0}, "max_xy_radius_m": 0.5,
            "z_min_m": 0.8, "z_max_m": 1.6, "tolerance_m": 0.01}},
        "recovery_snapshot": {"resource_jid": jid, "resource_state": state, "resource_location": location,
            "held_part": held, "current_pose": pose},
        "recovery_des_model": {"state_variables": variables, "events": [approach]}}


@pytest.fixture
def shared_cell(tmp_path):
    """Two arms share a cell; the interrupted arm is measured inside it."""
    resources = {"arm_a": _arm("arm_a", "picked", "buffer", "peg", {"x": 1.0, "y": 0.0, "z": 1.3}),
                 "arm_b": _arm("arm_b", "failed", "station", None, {"x": 0.0, "y": 0.0, "z": 1.2})}
    region = {"label": "ap001", "definition": {"kind": "ap_state", "process": "*", "product": "*",
        "resource": "$first_resource", "state": {"arguments": {"region": "$region"}, "symbol": "any"}}}
    rule = {"id": "SAFE_cell_mutex", "requirement": "At most one resource may occupy cell.",
            "formula": "G !(ap001 & ap002)", "aps": [region, {**region, "label": "ap002",
                "definition": {**region["definition"], "resource": "$second_resource"}}]}
    spec = tmp_path / "safety_cell_mutex.txt"
    spec.write_text(json.dumps({"mode": "predefined", "version": 2, "catalog": {"version": 2, "specifications": [rule]},
                                "requirement_scopes": [{"specification": "SAFE_cell_mutex", "region": "cell"}]}))
    conditions = [
        {"role": "resume_suffix", "entity_kind": "resource", "entity": "arm_a", "field": "current_state", "expected": "picked"},
        {"role": "resume_suffix", "entity_kind": "resource", "entity": "arm_a", "field": "held_part", "expected": "peg"},
        {"role": "recovery_replaced", "entity_kind": "part", "entity": "gear", "field": "state", "expected": "assembled"},
        {"role": "recovery_replaced", "entity_kind": "part", "entity": "gear", "field": "location", "expected": "cell"}]
    record = {"scenario": "Part slippage", "run_id": "supplied_fixture", "captured_at": "recorded-time",
        "jids": {"arm_a": "arm_a", "arm_b": "arm_b"},
        "physical_evidence": {"geometry": {"regions": {"cell": {"frame": "world",
            "bounds": [[-0.25, 0.25], [-0.25, 0.25], [1.0, 1.4]]}}}},
        "pending_nodes": [
            {"id": "replaced_approach", "function_name": "place_approach",
             "params": {"resource_id": "arm_b", "destination_location": "cell", "part_name": "gear"}},
            {"id": "resume_approach", "function_name": "place_approach",
             "params": {"resource_id": "arm_a", "destination_location": "cell", "part_name": "peg"}}],
        "prepared_recovery_request": {"ra_jid": "arm_b", "recovery_resources": resources,
            "marked_reentry_context": {
                "pending_suffixes": [
                    {"role": "recovery_replaced", "resource_jid": "arm_b", "pending_task_ids": ["replaced_approach"]},
                    {"role": "resume_suffix", "resource_jid": "arm_a", "entry_task_id": "resume_approach"}],
                "marked_reentry_conditions": conditions, "unmet_reentry_conditions": conditions[2:]},
            "llm_input": {"fault_event": {"focused_resource_jid": "arm_b", "blocked_at_function": "place_approach",
                                          "affected_part_names": ["gear"]},
                "observed_runtime_state": {"resources": [row["recovery_snapshot"] for row in resources.values()]},
                "part_facts": [{"part_name": "peg", "part_state": "in_gripper", "current_location": "arm_a",
                                "current_holder_resource_jid": "arm_a"},
                               {"part_name": "gear", "part_state": "unknown", "current_location": None,
                                "observed_pose": {"x": 0.9, "y": 0.1, "z": 1.0}}]}}}
    path = tmp_path / "observations.json"
    path.write_text(json.dumps(record))
    record.update(source_path=str(path), source_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    return record, [str(spec)]


def _apply(rows, session, prepared):
    for row in rows:
        task = multi_turn_diagnostic.event_record(row, session, prepared)
        multi_turn_diagnostic.apply_tentative_effects(task, session, prepared)
        session.setdefault("accepted_outline_prefix", []).append(task)


def _cell_recovery():
    return [
        event("clear", "arm_b", None, resource_state="idle", resource_location="home"),
        event("set_down", "arm_a", "peg", resource_state="placed", held_part=None, part_state="ready", part_location="buffer"),
        event("take", "arm_a", "gear", resource_state="picked", held_part="gear", part_state="in_gripper", part_location="arm_a"),
        event("insert", "arm_a", "gear", resource_state="placed", held_part=None, part_state="assembled", part_location="cell",
              resource_location="cell"),
        event("retake", "arm_a", "peg", resource_state="picked", held_part="peg", part_state="in_gripper", part_location="arm_a",
              resource_location="buffer"),
    ]


def _draft(shared_cell, tmp_path):
    record, specs = shared_cell
    draft = outline_draft.OutlineDraft(record, specs, root=tmp_path / "drafts")
    return draft.prepared, multi_turn.build_multi_turn_session_seed(draft.prepared)


def test_recorded_handoff_conditions_finish_without_replaying_interrupted_steps(shared_cell, tmp_path):
    prepared, session = _draft(shared_cell, tmp_path)
    assert multi_turn_diagnostic.unresolved_conditions(session, prepared)
    for index, row in enumerate(_cell_recovery()):
        _apply([row], session, prepared)
        state = multi_turn_diagnostic.handoff(session, prepared)
        assert state["complete"] is (index == 4), (row["outline_id"], state)
    # The interrupted owner step was never reproduced; the recorded outcome is what counts.
    assert state["remaining_replaced_task_ids"] == ["replaced_approach"]
    assert multi_turn_diagnostic.unresolved_conditions(session, prepared) == set()


def test_a_resumed_entry_no_declared_event_binds_is_stated_as_not_checked(shared_cell, tmp_path):
    record, specs = shared_cell
    request = record["prepared_recovery_request"]
    request["recovery_resources"]["belt"] = {"resource_jid": "belt", "resource_type": "resource",
        "recovery_snapshot": {"resource_jid": "belt", "resource_state": "idle"},
        "recovery_des_model": {"state_variables": {"resource_state": {"scope": "resource", "type": "string"}}, "events": []}}
    request["marked_reentry_context"]["pending_suffixes"].append(
        {"role": "resume_suffix", "resource_jid": "belt", "entry_task_id": "resume_belt"})
    record["pending_nodes"].append({"id": "resume_belt", "function_name": "advance_conveyor",
                                    "params": {"resource_id": "belt", "part_name": "bolt"}})
    record["jids"]["belt"] = "belt"
    prepared, session = _draft((record, specs), tmp_path)
    _apply(_cell_recovery(), session, prepared)
    state = multi_turn_diagnostic.handoff(session, prepared)
    # The bound entry still decides completion; the unbound one is stated instead of silently dropped.
    assert state["complete"] is True and state["required_nominal_entry_ids"] == ["nominal_entry:resume_approach"]
    assert state["assumptions"] == [{"task_id": "resume_belt", "reason": "no declared event of belt binds the resumed "
                                     "advance_conveyor; whether nominal work continues there is not checked"}]


def test_prompt_states_recorded_facts_for_every_robot_without_tuned_targets(shared_cell, tmp_path, monkeypatch):
    record, specs = shared_cell
    provider = AsyncMock(side_effect=[{"thought": "Use recorded facts.", "decision": "grounded", "observations": []},
                                      RuntimeError("stop after the first outline prompt")])
    monkeypatch.setattr(outline_draft.SavedRecoveryProduct, "ask_llm_structured", provider)
    draft = outline_draft.OutlineDraft(record, specs, root=tmp_path / "drafts")
    asyncio.run(draft.generate())
    prompt = provider.call_args_list[1].kwargs.get("prompt") or provider.call_args_list[1].args[0]
    assert "Prefer approximately" not in prompt
    # The goal comes first; bulky declarations are listed by ref, not inlined.
    headers = ("Failure", "Recovery Goal", "Resources", "Parts", "Safety Rules", "Candidate Rules",
               "Knowledge Base")
    assert [prompt.index(f"\n{header}\n") for header in headers] == sorted(prompt.index(f"\n{header}\n") for header in headers)
    assert "Diagnostic Mode" not in prompt and "Resource Capabilities" not in prompt
    assert "- arm_b did not complete place_approach" in prompt
    assert "- arm_b: measured pose (0.000, 0.000, 1.200), inside cell" in prompt
    # Which resources can act at the measured pose is left to be looked up and reasoned out.
    assert "Declared coverage" not in prompt and "covers:" not in prompt
    assert "\nRecovery Goal\nWhat the interrupted work was meant to achieve." in prompt
    assert "- gear: state assembled, location cell (normally arm_b's place_approach)\n" in prompt
    assert "\narm_b (robot)  failed; at station; holds none; pose (0.000, 0.000, 1.200), inside cell; " in prompt
    assert "\narm_a (robot)  picked; at buffer; holds peg; " in prompt
    assert "\ngear  not held, not at any station, measured at (0.900, 0.100, 1.000)\n" in prompt
    assert "misplaced" not in prompt and "robot states:" not in prompt and "Part states:" not in prompt
    assert "[[-0.25, 0.25]" not in prompt and "max_xy_radius_m" not in prompt
    assert "- /parts/<name>/pose_relations: " in prompt and "- /regions: " in prompt
    assert "/safety_rules" not in prompt and "gripper_reach" not in prompt
    assert "Resource validators check" not in prompt
    session = multi_turn.build_multi_turn_session_seed(draft.prepared)
    for jid in ("arm_a", "arm_b"):
        fact = multi_turn_context.resolve_outline_context(draft.prepared, session, f"/resources/{jid}/gripper_reach")
        assert fact["value"]["max_xy_radius_m"] == 0.5 and "tolerance_m" in fact["interpretation"]
    region = multi_turn_context.resolve_outline_context(draft.prepared, session, "/regions/cell")
    assert region["value"]["bounds"] == [[-0.25, 0.25], [-0.25, 0.25], [1.0, 1.4]]


def test_the_prompt_asks_for_a_plan_and_dates_the_failure_facts_once_events_exist(shared_cell, tmp_path):
    prepared, session = _draft(shared_cell, tmp_path)
    session["current_phase"] = "outline"
    before = multi_turn._build_phase_prompt(prepared, session)[1]
    assert "\nFailure\nWhat was detected and measured, stated the same way for every resource.\n" in before
    _apply(_cell_recovery()[1:2], session, prepared)
    after = multi_turn._build_phase_prompt(prepared, session)[1]
    # The measured failure facts stay, but they predate the state the planned events lead to.
    assert ("\nFailure\nWhat was detected and measured at the failure, before your events. Resources and Parts "
            "below show the state after them.\n") in after
    assert "- arm_b did not complete place_approach" in after
    for prompt in (before, after):
        assert "- `neighbor_end_state`: when moving the part changes another resource" in prompt
        assert ("Until normal production takes over, only your recovery events change the planned state; a job "
                "already running keeps running until an event pauses or stops it.") in prompt
        assert ("Plan how to get from the current state to the Recovery Goal. Propose 1–5 ways to do the next step "
                "of that plan, then choose one with `selected_candidate_index`.") in prompt
        assert "Give your plan in `thought` (two short sentences)" in prompt
    # Drafts offer neighbor_end_state right after the event's own end state.
    event_schema = multi_turn._get_response_schema("outline", session)["schema"]["$defs"]["outline_event"]
    names = list(event_schema["properties"])
    assert names[names.index("expected_end_state") + 1] == "neighbor_end_state"


def test_a_changed_resource_snapshot_leaves_out_its_failure_time_values(shared_cell, tmp_path):
    record, specs = deepcopy(shared_cell[0]), shared_cell[1]
    for jid, value in (("arm_a", "closed"), ("arm_b", "open")):
        record["prepared_recovery_request"]["recovery_resources"][jid]["recovery_snapshot"]["gripper_state"] = value
    prepared, session = _draft((record, specs), tmp_path)
    set_down = event("set_down", "arm_a", "peg", resource_state="placed", held_part=None, part_state="ready",
                     part_location="buffer")
    # Accepted through the real handler, which switches lookups to the projected rows.
    decision, _ = asyncio.run(multi_turn_validated_event.handle_validated_candidates(
        session=session, prepared=prepared, planner=SimpleNamespace(product_agent=SimpleNamespace()),
        response={"thought": "Set the peg down.", "candidate_events": [set_down], "selected_candidate_index": 0}))
    assert decision != "need_revision"
    changed = multi_turn_context.resolve_outline_context(prepared, session, "/resources/arm_a/snapshot")["value"]
    unchanged = multi_turn_context.resolve_outline_context(prepared, session, "/resources/arm_b/snapshot")["value"]
    # The released peg leaves no recorded closed gripper behind; an untouched arm keeps its values.
    assert "gripper_state" not in changed and changed["resource_state"] == "placed"
    assert unchanged["gripper_state"] == "open"


def test_handoff_status_is_recorded_for_review_and_never_told_to_the_llm(shared_cell, tmp_path):
    prepared, session = _draft(shared_cell, tmp_path)
    # The gear is assembled, but arm_a never retakes the peg its resumed step needs.
    _apply(_cell_recovery()[1:4], session, prepared)
    session["current_phase"] = "outline"
    prompt = multi_turn._build_phase_prompt(prepared, session)[1]
    assert "cannot take over" not in prompt and "Goal reached" not in prompt
    decision, audit = asyncio.run(multi_turn_validated_event.handle_validated_candidates(
        session=session, prepared=prepared, planner=SimpleNamespace(product_agent=SimpleNamespace()),
        response={"thought": "The failure is overcome.", "decision": "recovery_complete"}))
    # The LLM's declaration ends the recovery; why nominal work cannot take over is only recorded.
    assert decision == "outline_ready" and session["status"] == "ready_for_primitive_generation"
    handoff = audit["diagnostic_nominal_handoff"]
    assert handoff["complete"] is False and handoff["declared_complete_by_llm"] is True
    assert handoff["not_connected_reasons"] == ["no sequence of up to 6 declared nominal events reaches its next task"]


def test_pose_relations_treat_every_resource_by_its_declarations():
    regions = {"cell": {"bounds": [[-1, 1], [-1, 1], [0, 2]]}, "M1.workholding": {"bounds": [[4, 5], [4, 5], [0, 2]]}}
    envelope = {"origin_pose": {"x": 0.0, "y": 0.0}, "max_xy_radius_m": 0.5, "z_min_m": 0.5, "z_max_m": 1.5,
                "tolerance_m": 0.01}
    locations = {"arm": {"cell"}, "station": {"M1"}, "mobile": {"Storage", "M1"}, "silent": set()}
    station = multi_turn_diagnostic.pose_relations({"x": 4.5, "y": 4.5, "z": 1.0}, regions, locations, {"arm": envelope})
    assert station == {"containing_regions": ["M1.workholding"],
                       "by_declared_location": {"mobile": ["M1"], "station": ["M1"]},
                       "envelopes": {"arm": {"xy_distance_m": 6.364, "max_xy_radius_m": 0.5, "z_m": 1.0,
                                             "z_range_m": [0.5, 1.5], "tolerance_m": 0.01}},
                       "not_evaluated": ["silent"]}
    near = multi_turn_diagnostic.pose_relations({"x": 0.3, "y": 0.0, "z": 1.0}, regions, locations, {"arm": envelope})
    assert near["containing_regions"] == ["cell"] and near["by_declared_location"] == {"arm": ["cell"]}
    # Envelopes state distance and height; whether the pose fits is left to the reader.
    assert near["envelopes"]["arm"]["xy_distance_m"] == 0.3
    far = multi_turn_diagnostic.pose_relations({"x": 0.6, "y": 0.0, "z": 1.0}, regions, locations, {"arm": envelope})
    assert far["envelopes"]["arm"]["xy_distance_m"] == 0.6
    # A resource that moves to coordinates is not judged by its declared stations.
    moving = multi_turn_diagnostic.pose_relations({"x": 0.6, "y": 0.0, "z": 1.0}, regions, locations, {"arm": envelope},
                                                  {"mobile"})
    assert moving["not_evaluated"] == ["mobile", "silent"]


def test_a_pose_lies_only_in_regions_whose_every_bounded_axis_it_states():
    regions = {"cell": {"bounds": [[-1, 1], [-1, 1], [1.0, 1.4]]}, "floor": {"bounds": [[-1, 1], [-1, 1]]}}
    locations = {"arm": {"cell"}, "mobile": {"cell"}}
    # A planar pose states no height: it lies in no region bounded in height, only in one bounding x and y alone.
    planar = multi_turn_diagnostic.pose_relations({"x": 0.0, "y": 0.0, "yaw": 1.57}, regions, locations, {}, {"mobile"})
    assert planar["containing_regions"] == ["floor"] and planar["by_declared_location"] == {}
    assert planar["not_evaluated"] == ["mobile"]
    # The same holds for a part measured only in x and y.
    assert multi_turn_diagnostic.pose_relations({"x": 0.0, "y": 0.0}, regions, {}, {})["containing_regions"] == ["floor"]
    full = multi_turn_diagnostic.pose_relations({"x": 0.0, "y": 0.0, "z": 1.2}, regions, locations, {}, {"mobile"})
    assert full["containing_regions"] == ["cell", "floor"]
    assert full["by_declared_location"] == {"arm": ["cell"], "mobile": ["cell"]} and full["not_evaluated"] == []


def test_diagnostic_refs_serve_declarations_and_derived_pose_relations(shared_cell, tmp_path):
    prepared, session = _draft(shared_cell, tmp_path)
    resolve = multi_turn_context.resolve_outline_context
    gear = resolve(prepared, session, "/parts/gear/pose_relations")
    assert gear["source"] == "model_calculation" and gear["basis"]
    assert gear["value"]["observed_pose"] == {"x": 0.9, "y": 0.1, "z": 1.0}
    assert gear["value"]["containing_regions"] == [] and "by_declared_envelope" not in gear["value"]
    assert {jid: row["xy_distance_m"] for jid, row in gear["value"]["envelopes"].items()} == {"arm_a": 0.141, "arm_b": 0.906}
    arm_b = resolve(prepared, session, "/resources/arm_b/pose_relations")["value"]
    assert arm_b["containing_regions"] == ["cell"]
    assert arm_b["by_declared_location"] == {"arm_a": ["cell"], "arm_b": ["cell"]}
    assert resolve(prepared, session, "/safety_rules/SAFE_cell_mutex")["value"]["id"] == "SAFE_cell_mutex"
    assert "world frame" in resolve(prepared, session, "/geometry")["interpretation"]
    listed = [ref for ref, _ in multi_turn_context.diagnostic_refs(prepared)]
    assert "/resources/<jid>/primitives" not in listed
    for ref in ("/resources/arm_a/primitives", "/geometry/frame/x", "/resources/unknown/gripper_reach"):
        with pytest.raises(ValueError, match="NEEDS_CONTEXT"):
            resolve(prepared, session, ref)


def test_roster_shows_the_declared_locations_of_a_resource_without_reachability(shared_cell, tmp_path):
    record, specs = shared_cell
    mobile = _arm("mobile", "idle", "Storage", None, {"x": 5.0, "y": 5.0, "z": 0.7})
    mobile["static_capabilities"] = {}
    mobile["recovery_des_model"]["state_variables"]["resource_location"]["domain"] = [None, "Storage", "cell"]
    record["prepared_recovery_request"]["recovery_resources"]["mobile"] = mobile
    prepared, session = _draft((record, specs), tmp_path)
    session["current_phase"] = "outline"
    prompt = multi_turn._build_phase_prompt(prepared, session)[1]
    # Without recorded runtime state the resource stays under Other Resources, with its declared locations.
    # Without recorded runtime state the resource is still listed, with its declared locations.
    assert "\nmobile (robot)  state not recorded; holds none; capabilities: reachable locations Storage, cell\n" in prompt
    assert "field_sources" not in prompt and "evaluation_support" not in prompt
    relations = multi_turn_context.resolve_outline_context(prepared, session, "/resources/arm_b/pose_relations")["value"]
    assert relations["by_declared_location"]["mobile"] == ["cell"]


def _candidate_limit(value):
    if isinstance(value, dict):
        if "candidate_events" in value.get("properties", {}):
            return value["properties"]["candidate_events"].get("maxItems")
        return next((limit for item in value.values() if (limit := _candidate_limit(item)) is not None), None)
    return None


def test_llm_always_chooses_a_candidate_in_a_draft(shared_cell, tmp_path, monkeypatch):
    record, specs = shared_cell
    provider = AsyncMock(side_effect=[{"thought": "Use recorded facts.", "decision": "grounded", "observations": []},
                                      RuntimeError("stop after the first outline prompt")])
    monkeypatch.setattr(outline_draft.SavedRecoveryProduct, "ask_llm_structured", provider)
    draft = outline_draft.OutlineDraft(record, specs, root=tmp_path / "drafts")
    asyncio.run(draft.generate())
    call = provider.call_args_list[1]
    prompt = call.kwargs.get("prompt") or call.args[0]
    assert _candidate_limit(call.kwargs) == 5
    assert "selected_candidate_index" in json.dumps(call.kwargs["response_format"])
    assert "recovery_complete" in json.dumps(call.kwargs["response_format"])
    # Chat completions reject function tools with reasoning effort, so facts are requested by JSON.
    assert "tools" not in call.kwargs
    assert "then choose one with `selected_candidate_index`" in prompt
    assert "Do not select or rank candidates" not in prompt
    # Code applies neighbor_end_state unchecked, so every draft offers it.
    assert "neighbor_end_state" in json.dumps(call.kwargs["response_format"])
    assert "- `neighbor_end_state`: when moving the part changes another resource" in prompt
    # No validator checks a draft, so a recorded check flag never takes the choice from the LLM.
    session = multi_turn.build_multi_turn_session_seed(draft.prepared)
    assert multi_turn_diagnostic.llm_selects_next_event({**session, "check_cca_safety": True}) is True


def _resume_only_cell(shared_cell):
    """Record only the resumed work's entry, as schema-2 captures do, with a final insert step."""
    record, specs = deepcopy(shared_cell[0]), shared_cell[1]
    insert = {"event_name": "place_insert", "parameter_bindings": {},
              "guards": {"resource_state": {"equals": "picked"}},
              "updates": {"resource_state": {"set": "placed"}, "held_part": {"set": None},
                          "part_state": {"set": "assembled"}, "part_location": {"set_from_param": "destination_location"}}}
    request = record["prepared_recovery_request"]
    for resource in request["recovery_resources"].values():
        resource["recovery_des_model"]["events"].append(deepcopy(insert))
    record["pending_nodes"].insert(1, {"id": "replaced_insert", "function_name": "place_insert",
        "params": {"resource_id": "arm_b", "destination_location": "cell", "part_name": "gear"}})
    request["marked_reentry_context"]["pending_suffixes"][0]["pending_task_ids"] = ["replaced_approach", "replaced_insert"]
    conditions = request["marked_reentry_context"]["marked_reentry_conditions"][:2]
    request["marked_reentry_context"].update(marked_reentry_conditions=conditions, unmet_reentry_conditions=[])
    return record, specs


def test_a_replaced_step_its_model_cannot_describe_is_done_when_its_part_reaches_its_destination(shared_cell, tmp_path):
    record, specs = _resume_only_cell(shared_cell)
    next(node for node in record["pending_nodes"] if node["id"] == "replaced_insert")["function_name"] = "place_release"
    # Another resource's event of the same name must not complete it by its own fixed effects.
    record["prepared_recovery_request"]["recovery_resources"]["arm_a"]["recovery_des_model"]["events"].append(
        {"event_name": "place_release", "parameter_bindings": {}, "guards": {},
         "updates": {"resource_state": {"set": "placed"}, "held_part": {"set": None}}})
    prepared, session = _draft((record, specs), tmp_path)
    # The goal is the part at the step's destination, and the step is open work, not an assumption.
    assert facts_goal(prepared, session) == ["- gear: part_location cell (normally arm_b's place_release)"]
    open_work, assumed = multi_turn_diagnostic.open_replaced_work(session, prepared)
    assert open_work == {"replaced_insert"} and assumed == []
    recovery = _cell_recovery()
    drop = event("drop", "arm_a", "gear", resource_state="placed", held_part=None, part_state="ready",
                 part_location="buffer")
    # The part must rest at the destination: held there by its actor does not complete the step.
    hold_there = event("hold_there", "arm_a", "gear", resource_state="positioned", held_part="gear",
                       part_state="in_gripper", part_location="cell")
    _apply([*recovery[:3], drop, hold_there], session, prepared)
    assert multi_turn_diagnostic.open_replaced_work(session, prepared)[0] == {"replaced_insert"}

    def claims():
        return sorted(session.get("diagnostic_replaced_task_claims", {}))

    assert claims() == []
    _apply(recovery[3:4], session, prepared)
    assert multi_turn_diagnostic.open_replaced_work(session, prepared)[0] == set()
    # The claim is kept for the handoff record; it is not a value of any resource or part, so the table omits it.
    assert claims() == ["replaced_insert"]
    assert not any(row["entity"].startswith("interrupted ")
                   for row in session["accepted_outline_prefix"][-1]["condition_changes"])
    # A later move away reopens the work rather than keeping a stale claim.
    _apply([event("lift_again", "arm_a", "gear", resource_state="picked", held_part="gear", part_state="in_gripper",
                  part_location="arm_a")], session, prepared)
    assert multi_turn_diagnostic.open_replaced_work(session, prepared)[0] == {"replaced_insert"}


def facts_goal(prepared, session):
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes import multi_turn_outline_facts as facts

    return facts.goal_lines(prepared, session)


def test_resume_only_conditions_still_require_the_replaced_final_outcome(shared_cell, tmp_path):
    prepared, session = _draft(_resume_only_cell(shared_cell), tmp_path)
    recovery = _cell_recovery()
    # Clearing the cell satisfies the recorded entry, but gear is not assembled.
    _apply(recovery[:1], session, prepared)
    state = multi_turn_diagnostic.handoff(session, prepared)
    assert state["complete"] is False and state["open_replaced_task_ids"] == ["replaced_insert"]
    assert "replaced_work:replaced_insert" in multi_turn_diagnostic.unresolved_conditions(session, prepared)
    for index, row in enumerate(recovery[1:], start=1):
        _apply([row], session, prepared)
        assert multi_turn_diagnostic.handoff(session, prepared)["complete"] is (index == 4), row["outline_id"]


def test_template_states_a_machine_fault_from_recorded_facts(captured):
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes import multi_turn_outline_facts

    prepared, _ = outline_draft._inputs(captured)
    prepared["llm_input"]["fault_event"].update(blocked_at_task_id="environment_21", blocked_at_function="machine_part",
        detected_signals=[{"signal": "task_outcome", "resource_id": "M2", "task_id": "environment_21",
                           "part_name": "KET4_Square_4mm", "job_state": "interrupted", "processing_fraction": 0.07},
                          {"signal": "device_fault_report", "resource_id": "M2", "available": False,
                           "reported_by": "device controller"}])
    session = multi_turn.build_multi_turn_session_seed(prepared)
    template = multi_turn_outline_facts.outline_template(prepared, session)
    assert template["failure"] == [
        "- M1 did not complete machine_part of KET4_Square_4mm (process trim, result square)",
        "- task_outcome (M2, KET4_Square_4mm, task environment_21): job_state interrupted, processing_fraction 0.07",
        "- device_fault_report (M2): available false, reported_by device controller",
        "- KET4_Square_4mm: loaded, at M1"]
    assert template["goal"] == ["- KET4_Square_4mm: process trim, result square (normally M1's machine_part)"]
    assert template["resources"] == [
        "M2 (resource)  loaded; at M2; holds none; part_name RGOCG4-50_Round_4mm; job_state running; capabilities: reachable "
        "locations M1, M2; trim (circle, square)",
        "arm (resource)  idle; at M2; holds none; job_state running; capabilities: reachable locations M1, M2"]
    resources = "\n".join(template["resources"])
    # Owner bookkeeping and nominal event names are not offered.
    assert "active_job" not in resources and "private_job" not in resources and "machine_part" not in resources


def test_template_states_a_part_only_failure_and_its_goal(shared_cell, tmp_path):
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes import multi_turn_outline_facts

    prepared, session = _draft(shared_cell, tmp_path)
    prepared["llm_input"]["fault_event"] = {"detected_signals": [{"signal": "state_discrepancy", "part_name": "gear",
        "expected_location": "cell", "measured_position": [0.9, 0.1, 1.0]}]}
    prepared["marked_reentry_context"].update(pending_suffixes=[], marked_reentry_conditions=[])
    template = multi_turn_outline_facts.outline_template(prepared, session)
    assert template["failure"] == [
        "- state_discrepancy (gear): expected_location cell, measured_position (0.9, 0.1, 1)",
        "- gear: not held, not at any station, measured at (0.900, 0.100, 1.000)"]
    assert template["goal"] == ["- gear: location cell"]
    assert "gear  not held, not at any station, measured at (0.900, 0.100, 1.000)" in template["parts"]


def test_resource_names_round_trip_through_prompt_schema_and_response(captured):
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes import multi_turn_outline_facts as facts

    prepared, _ = outline_draft._inputs(captured)
    names = facts.resource_names(prepared)
    assert facts.with_resource_names('machine_part by "machine" at /resources/machine/snapshot', names) == (
        'machine_part by "M2" at /resources/M2/snapshot')
    schema = facts.schema_with_resource_names({"properties": {"resource_jid": {"enum": ["machine"]}},
                                               "required": ["resource_jid"]}, names)
    assert schema == {"properties": {"resource": {"enum": ["M2"]}}, "required": ["resource"]}
    response = facts.response_with_jids({"thought": "t", "context_requests": ["/resources/M2/snapshot"],
        "candidate_events": [
            {"outline_id": "load", "resource": "M2", "part_name": "KET4_Square_4mm",
             "expected_end_state": {"part_location": "M2", "part_state": "loaded"}},
            {"outline_id": "hold", "resource": "M2", "part_name": "KET4_Square_4mm",
             "expected_end_state": {"held_part": "KET4_Square_4mm", "part_location": "M2"}}]}, prepared)
    assert response["context_requests"] == ["/resources/machine/snapshot"]
    load, hold = response["candidate_events"]
    assert load["resource_jid"] == "machine" and "resource" not in load
    # A station named like its resource stays a location; only taking hold maps to the JID.
    assert load["expected_end_state"]["part_location"] == "M2"
    assert hold["expected_end_state"]["part_location"] == "machine"
    placed = facts.response_with_jids({"thought": "t", "candidate_events": [
        {"outline_id": "unload", "resource": "M2", "part_name": "KET4_Square_4mm", "expected_end_state": {},
         "neighbor_end_state": {"M2": {"part_name": None}}}]}, prepared)
    assert placed["candidate_events"][0]["neighbor_end_state"] == {"machine": {"part_name": None}}



def test_events_so_far_are_listed_with_the_reply_format_key_and_resource_names(captured, tmp_path):
    prepared = outline_draft.OutlineDraft(captured, [], root=tmp_path / "drafts").prepared
    session = multi_turn.build_multi_turn_session_seed(prepared)
    session["current_phase"] = "outline"
    _apply(sequence()[:1], session, prepared)
    prompt = multi_turn._build_phase_prompt(prepared, session)[1]
    listed = prompt.split("\nRecovery Events So Far (planned, not executed)\n", 1)[1].split(
        "\nResources and Parts already include these events.", 1)[0]
    reply_keys = multi_turn._get_response_schema("outline", session)["schema"]["$defs"]["outline_event"]["properties"]
    # Each event is listed as a reply states it: under `resource`, by the resource's name.
    assert json.loads(listed) == [{"outline_id": "pause", "event_name": "pause", "resource": "M2"}]
    assert "resource" in reply_keys and "resource_jid" not in reply_keys


def test_a_lookup_domain_names_a_resource_listed_by_jid_and_by_name_once(captured):
    prepared, _ = outline_draft._inputs(captured)
    # The arm's part locations list the M2 station by name and the machine M2 itself by JID.
    prepared["recovery_resources"]["arm"]["recovery_des_model"]["state_variables"]["part_location"]["domain"] = [
        "M2", "machine", None]
    session = multi_turn.build_multi_turn_session_seed(prepared)
    session.update(current_phase="outline", outline_requested_refs=[
        "/resources/arm/state_variables", "/parts/KET4_Square_4mm/state_variables"])
    retrieved = multi_turn._build_phase_prompt(prepared, session)[1].split("\nRetrieved context\n", 1)[1]
    replies = dict(line.split(": ", 1) for line in retrieved.splitlines() if line.startswith("/"))
    for ref in ("/resources/arm/state_variables", "/parts/KET4_Square_4mm/state_variables"):
        assert json.loads(replies[ref])["value"]["part_location"]["domain"] == ["M2", None], ref
    # The recorded reply keeps both declared tokens; only the prompt states them by name.
    fact = multi_turn_context.resolve_outline_context(prepared, session, "/resources/arm/state_variables")
    assert fact["value"]["part_location"]["domain"] == ["M2", "machine", None]


def test_a_lookup_reply_may_carry_an_empty_candidate_list_and_a_choice_index(captured):
    prepared, _ = outline_draft._inputs(captured)
    session = multi_turn.build_multi_turn_session_seed(prepared)
    status, turn = asyncio.run(multi_turn_validated_event.handle_validated_candidates(
        session=session, prepared=prepared, planner=SimpleNamespace(product_agent=SimpleNamespace()),
        response={"thought": "Look up the machine first.", "decision": "need_context",
                  "context_requests": ["/resources/M2/snapshot"], "candidate_events": [],
                  "selected_candidate_index": 0}))
    assert status == "need_context" and turn["context_errors"] == []
    assert list(turn["served_context"]) == ["/resources/machine/snapshot"]
    assert session.get("accepted_outline_prefix", []) == []



def test_lookup_errors_name_what_is_wrong():
    error = multi_turn_context._context_request_error
    assert error({"thought": "t", "context_requests": []}) == "need_context requires context_requests with 1 to 8 references"
    assert error({"thought": "t", "context_requests": ["/a"] * 9}) == "need_context allows at most 8 references; 9 were requested"
    assert error({"thought": "t", "context_requests": ["parts"]}) == (
        "each context request must be a reference starting with /: parts")
    assert error({"thought": "t", "context_requests": ["/parts"], "candidate_events": [{}]}) == (
        "need_context cannot also carry candidate_events; propose events in a later reply")
    assert error({"thought": "t", "context_requests": ["/parts"]}) is None


def test_feedback_rows_name_their_event_and_carry_no_placeholders():
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes import multi_turn_prompts as prompts

    custody = {"constraint_owner": "resource", "constraint_code": "custody_discontinuity", "reason": "arm_a already holds peg",
               "outline_id": "take", "resource_jid": "arm_a", "part_name": "gear"}
    shown = prompts._outline_feedback_for_prompt(custody, set())
    assert shown["reason"] == "arm_a already holds peg"
    assert prompts._render_des_event_diagnostic(task={}, finding=shown) == (
        "- candidate_event=take/arm_a/gear | constraint_code=custody_discontinuity | reason=arm_a already holds peg")
    assert prompts._render_des_event_diagnostic(task={}, finding={"constraint_code": "x", "reason": "y"}) == (
        "- constraint_code=x | reason=y")

def test_draft_rows_name_their_resource_without_changing_the_saved_record(captured, tmp_path):
    draft = outline_draft.OutlineDraft(captured, [], root=tmp_path / "drafts")
    draft.record["transition_trace"] = [{"event_name": "load", "resource_jid": "machine", "part_name": "KET4_Square_4mm",
                                         "expected_end_state": {"part_location": "machine", "part_state": "loaded"},
                                         "neighbor_end_state": {"machine": {"part_name": "KET4_Square_4mm"}}}]
    saved = deepcopy(draft.record)
    row, = draft.snapshot()["outline_draft"]["transition_trace"]
    assert row["resource"] == "M2" and row["expected_end_state"] == {"part_location": "M2", "part_state": "loaded"}
    assert row["neighbor_end_state"] == {"M2": {"part_name": "KET4_Square_4mm"}}
    assert draft.record == saved

def test_resource_rows_state_set_family_entries_without_function_names():
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes import multi_turn_outline_facts as facts

    model = {"events": [
        {"event_name": "move_to_resource", "updates": {"resource_location": {"set_from_param": "target_resource"}},
         "parameter_bindings": {"target_resource": {"location_type": "reachable_location"}}},
        {"event_name": "move_to_location", "updates": {"resource_location": {"set_from_param": "observed"}},
         "parameter_bindings": {name: {"required": True, "type": "number"} for name in ("x", "y", "yaw")}}]}
    assert facts._coordinate_parameters(model) == ["x", "y", "yaw"]
    assert facts._coordinate_parameters({"events": model["events"][:1]}) == []
    entry = {"recovery_des_model": {"state_variables": {"resource_location": {"domain": [None, "Storage", "M1"]}}}}
    # Every resource states its declared capabilities by the same rule; one that declares none has no entry.
    prepared = {"recovery_resources": {"mobile": entry, "arm": entry, "station": {}},
                "diagnostic_reach": {"arm": {"max_xy_radius_m": 0.5}}}
    session = {"recovery_des_models": {"mobile": model}}
    assert facts._capabilities("mobile", prepared, session) == (
        "capabilities: reachable locations Storage, M1; moves to any position (x, y, yaw)")
    assert facts._capabilities("arm", prepared, session) == "capabilities: reachable locations Storage, M1; workspace bounds"
    assert facts._capabilities("station", prepared, session) == ""
    variables = {name: {} for name in ("belt_stopped", "part_location.gear", "part_location.peg", "inventory.peg",
                                       "inventory.gear", "active_job")} | {"task_ctx.part_name": {"private": True}}
    session = {"symbolic_resources": {"conveyor": {"resource_state": "unknown", "belt_stopped": True,
        "part_location.gear": "loading_position_1", "part_location.peg": None, "inventory.peg": True,
        "inventory.gear": False, "active_job": "private_job"}},
        "recovery_des_models": {"conveyor": {"state_variables": variables}}}
    row = facts._resource_row("conveyor", {"recovery_resources": {"conveyor": {"resource_type": "resource"}}}, session,
                              {"conveyor": "Conveyor"})
    # A resource that declares no resource_state shows no state word, and a family shows its name pattern.
    assert row == ("Conveyor (resource); belt_stopped true; part_location.<part name>: gear loading_position_1; "
                   "inventory.<part name>: peg")


def test_a_reachable_place_states_where_its_own_events_put_a_part_placed_there():
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes import multi_turn_outline_facts as facts

    def placement(resource: str, place: str, field: str, value: object) -> dict:
        return {"event_name": "place_release", "updates": {field: {"set": value}},
                "parameter_bindings": {"resource_id": {"equals": resource}, "destination_location": {"equals": place}}}

    belt_events = [placement("arm_1", "Belt", "part_location.{part_name}", "loading_position_1"),
                   placement("arm_2", "Belt", "part_location.{part_name}", "loading_position_2")]
    prepared = {"observation_jids": {"Belt": "belt", "Storage": "storage", "arm_1": "a1", "arm_2": "a2", "arm_3": "a3"},
        "recovery_resources": {
            "a1": {"static_capabilities": {"reachability": ["M1", "Belt"]}},
            "a2": {"static_capabilities": {"reachability": ["M2", "Belt", "Storage"]}},
            "a3": {"static_capabilities": {"reachability": ["Belt"]}},
            # A draft keeps the owner's declared events; a full entry carries them in its owner evidence.
            "belt": {"environment_model_events": belt_events},
            "storage": {"resource_evidence": {"resource_id": "Storage", "environment_models": {"Storage": {"events": [
                placement("arm_2", "Storage", "inventory.{part_name}", True)]}}}}}}
    placed_on_belt = "capabilities: reachable locations M1, Belt; places parts on Belt at loading_position_1"
    assert facts._capabilities("a1", prepared, {}) == placed_on_belt
    # A per-part flag is not a position, and a resource the place declares no placement for gets no entry.
    assert facts._capabilities("a2", prepared, {}) == (
        "capabilities: reachable locations M2, Belt, Storage; places parts on Belt at loading_position_2")
    assert facts._capabilities("a3", prepared, {}) == "capabilities: reachable locations Belt"
    prepared["recovery_resources"]["belt"] = {"resource_evidence": {"resource_id": "Belt",
                                                                    "environment_models": {"Belt": {"events": belt_events}}}}
    assert facts._capabilities("a1", prepared, {}) == placed_on_belt


def test_a_machine_keeps_each_job_by_its_part_so_a_paused_one_stays_in_view():
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes import multi_turn_outline_facts as facts

    jobs = {"job_1": {"part_name": "blank", "job_state": "running", "processing_fraction": 0.05}}
    prepared = {"observation_jids": {"mill": "mill"}, "recovery_resources": {"mill": {
        "resource_type": "resource", "static_capabilities": {"processing_observations": {"jobs": jobs}}}}}
    session = {"symbolic_resources": {"mill": {"resource_state": "loaded", "part_name": "blank", "job_state": "running"}},
               "recovery_des_models": {"mill": {"state_variables": {}}}}
    names = facts.resource_names(prepared)
    assert facts._resource_row("mill", prepared, session, names).endswith("jobs: blank running (processing_fraction 0.05)")
    # A stated job_state is the job of the part the mill holds.
    multi_turn_diagnostic._project_machine_jobs({"resource_jid": "mill", "expected_end_state": {"job_state": "paused"}},
                                                session, prepared)
    # Another part's job runs to completion; the paused job stays listed by its part.
    session["symbolic_resources"]["mill"]["part_name"] = "peg"
    multi_turn_diagnostic._project_machine_jobs({"resource_jid": "mill", "expected_end_state": {"job_state": "completed"}},
                                                session, prepared)
    assert facts._resource_row("mill", prepared, session, names).endswith("jobs: blank paused, peg completed")
    # No active job is not a job state: the retained jobs keep theirs.
    multi_turn_diagnostic._project_machine_jobs({"resource_jid": "mill", "expected_end_state": {"job_state": None}},
                                                session, prepared)
    assert facts._resource_row("mill", prepared, session, names).endswith("jobs: blank paused, peg completed")
    # Holding no part, the mill's job_state is its last retained job's.
    session["symbolic_resources"]["mill"]["part_name"] = None
    multi_turn_diagnostic._project_machine_jobs({"resource_jid": "mill", "expected_end_state": {"job_state": "stopped"}},
                                                session, prepared)
    assert [job["job_state"] for job in multi_turn_diagnostic.machine_jobs("mill", session, prepared)] == ["paused", "stopped"]


def test_a_resource_states_its_declared_job_control_flags():
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes import multi_turn_outline_facts as facts

    job_control = {"version": 1, "resume_after_reload": True, "restart_stopped_jobs": False,
                   "semantics": "active simulation time"}
    prepared = {"recovery_resources": {"machine": {"static_capabilities": {"job_control": job_control}}, "station": {}}}
    # Only the declared flags are stated, by name and value; the version and description are not flags.
    assert facts._capabilities("machine", prepared, {}) == (
        "capabilities: job control: resume_after_reload true, restart_stopped_jobs false")
    assert facts._capabilities("station", prepared, {}) == ""


def test_a_signal_kind_never_seen_before_is_stated_with_all_its_recorded_values(shared_cell, tmp_path):
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes import multi_turn_outline_facts

    prepared, session = _draft(shared_cell, tmp_path)
    prepared["llm_input"]["fault_event"]["detected_signals"] = [
        {"signal": "torque_spike", "resource_id": "arm_a", "joint": "wrist_3", "peak_nm": 41.25, "over_limit": True},
        {"signal": "surface_scratch", "part_name": "peg", "expected_finish": "smooth", "measured_depth_mm": 0.31}]
    template = multi_turn_outline_facts.outline_template(prepared, session)
    assert "- torque_spike (arm_a): joint wrist_3, peak_nm 41.25, over_limit true" in template["failure"]
    assert "- surface_scratch (peg): expected_finish smooth, measured_depth_mm 0.31" in template["failure"]
    # The newly observed resource and part enter the failure scope like any other.
    assert {"arm_a", "arm_b"} <= set(template["resource_jids"]) and "peg" in template["part_names"]


def test_context_replies_repeat_only_what_current_state_shows_and_rules_carry_no_scenario_hint(shared_cell, tmp_path):
    prepared, session = _draft(shared_cell, tmp_path)
    session.update(current_phase="outline", outline_requested_refs=["/resources/arm_b/snapshot", "/resources"])
    prompt = multi_turn._build_phase_prompt(prepared, session)[1]
    retrieved = prompt.split("\nRetrieved context\n", 1)[1]
    assert "Current DES State" not in retrieved and "Resource Capabilities" not in retrieved
    assert "Fields already shown above: " in retrieved
    # The resource roster is not displayed in this prompt, so its reply keeps every value.
    roster = next(line for line in retrieved.splitlines() if line.startswith("/resources: "))
    assert '"value": [' in roster
    assert "Look up any fact or variable you rely on that is not shown." in prompt
    assert "measured pose can be reached" not in prompt and "may omit `resource_location`" not in prompt


def test_goal_states_the_declared_outcome_of_the_interrupted_work(shared_cell, tmp_path):
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes import multi_turn_outline_facts as facts

    prepared, session = _draft(_resume_only_cell(shared_cell), tmp_path)
    # The final interrupted task declares part effects, resolved with its own parameters.
    assert facts.outline_template(prepared, session)["goal"] == [
        "- gear: part_state assembled, part_location cell (normally arm_b's place_insert)"]


def test_each_goal_names_the_recorded_task_that_normally_produces_it():
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes import (
        multi_turn_outline_facts as facts,
    )

    trim = {"event_name": "trim_part", "parameter_bindings": {"process": {"equals": "trim"}},
            "updates": {"resource_state": {"set": "completed"}}}
    arm_states = {"resource_state": {"scope": "resource", "type": "string", "domain": ["idle", "failed"]}}
    prepared = {"observation_jids": {"mill": "mill@host", "arm": "arm@host"},
        "recovery_resources": {"mill@host": {}, "arm@host": {}},
        "saved_pending_nodes": [
            {"id": "trim", "function_name": "trim_part",
             "params": {"resource_id": "mill", "part_name": "blank", "process": "trim"}},
            {"id": "load", "function_name": "place_part",
             "params": {"resource_id": "arm", "part_name": "peg", "destination_location": "mill"}}],
        "marked_reentry_context": {"pending_suffixes": [
            {"role": "recovery_replaced", "resource_jid": "mill@host", "pending_task_ids": ["trim"]},
            {"role": "recovery_replaced", "resource_jid": "arm@host", "pending_task_ids": ["load"]}]},
        "llm_input": {"fault_event": {"detected_signals": [
            {"signal": "device_fault_report", "resource_id": "mill", "available": False},
            {"signal": "gripper_check", "resource_id": "arm", "resource_state": "failed"}]}}}
    session = {"symbolic_resources": {"mill@host": {"resource_state": "loaded", "availability": "unavailable"},
                                      "arm@host": {"resource_state": "failed"}},
               "recovery_des_models": {"mill@host": {"state_variables": {}, "events": [trim]},
                                       "arm@host": {"state_variables": arm_states, "events": []}}}
    # Each outcome names the recorded task that normally produces it, and the current value of each
    # resource that task needs, as its actor or as a place it names, while a signal reports it faulted.
    assert facts.goal_lines(prepared, session) == [
        "- blank: process trim (normally mill's trim_part; mill unavailable)",
        "- peg: part_location mill (normally arm's place_part; arm failed; mill unavailable)"]
    # A fault left for a declared state is no longer stated.
    session["symbolic_resources"]["arm@host"]["resource_state"] = "idle"
    assert facts.goal_lines(prepared, session)[1] == "- peg: part_location mill (normally arm's place_part; mill unavailable)"



def test_a_new_state_name_still_completes_the_interrupted_part_outcome(shared_cell, tmp_path):
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes import multi_turn_outline_facts

    prepared, session = _draft(_resume_only_cell(shared_cell), tmp_path)
    recovery = _cell_recovery()
    insert = {**recovery[3], "expected_end_state": {**recovery[3]["expected_end_state"],
                                                    "resource_state": "recovery_assembly_complete"}}
    _apply([*recovery[:3], insert], session, prepared)
    assert multi_turn_diagnostic.handoff(session, prepared)["open_replaced_task_ids"] == []
    assert multi_turn_diagnostic.goal_reached(session, prepared) is True
    # Reaching the goal changes nothing the LLM is told; only its declaration ends the recovery.
    assert multi_turn_outline_facts.goal_lines(prepared, session) == [
        "- gear: part_state assembled, part_location cell (normally arm_b's place_insert)"]



def _with_pick_events(cell):
    """Give the shared-cell arms the declared nominal moves a connection can use."""
    record, specs = deepcopy(cell[0]), cell[1]
    for jid, resource in record["prepared_recovery_request"]["recovery_resources"].items():
        resource["recovery_des_model"]["events"] += [
            {"event_name": "move_home", "controllable": True, "guards": {"held_part": {"equals": None}},
             "updates": {"resource_state": {"set": "idle"}, "resource_location": {"set": "home"}}},
            {"event_name": "pick_approach", "controllable": True, "requires_part_binding": True,
             "guards": {"resource_state": {"equals": "idle"}, "held_part": {"equals": None}},
             "updates": {"resource_state": {"set": "at_pick"},
                         "resource_location": {"set_from_param": "origin_resource_location"}},
             "parameter_bindings": {"origin_resource_location": {"location_type": "part_location"}}},
            {"event_name": "pick_grasp", "controllable": True, "requires_part_binding": True,
             "guards": {"resource_state": {"equals": "at_pick"}, "held_part": {"equals": None}},
             "updates": {"resource_state": {"set": "picked"}, "held_part": {"set_from_param": "part_name"},
                         "part_state": {"set": "in_gripper"}, "part_location": {"set": jid}}}]
    return record, specs


def test_nominal_events_take_over_once_the_interrupted_outcome_holds(shared_cell, tmp_path):
    prepared, session = _draft(_with_pick_events(_resume_only_cell(shared_cell)), tmp_path)
    recovery = _cell_recovery()
    _apply(recovery[:3], session, prepared)
    planner = SimpleNamespace(product_agent=SimpleNamespace())
    decision, _ = asyncio.run(multi_turn_validated_event.handle_validated_candidates(
        session=session, prepared=prepared, planner=planner,
        response={"thought": "Insert the gear.", "candidate_events": [recovery[3]]}))
    # Reaching the outcome does not end the recovery; the LLM decides that.
    assert decision == "need_next_task"
    decision, audit = asyncio.run(multi_turn_validated_event.handle_validated_candidates(
        session=session, prepared=prepared, planner=planner,
        response={"thought": "The failure is overcome.", "decision": "recovery_complete"}))
    # The LLM never retakes the peg; the declared nominal moves reach the next nominal task.
    assert decision == "outline_ready" and session["status"] == "ready_for_primitive_generation"
    connection = audit["diagnostic_nominal_handoff"]["nominal_connection"]
    assert [(step["resource_jid"], step["event_name"], step.get("part_name")) for step in connection] == [
        ("arm_a", "move_home", None), ("arm_a", "pick_approach", "peg"), ("arm_a", "pick_grasp", "peg")]
    assert {step["candidate_source"] for step in connection} == {"nominal_connection"}


def test_a_nominal_step_s_own_bookkeeping_guards_are_recorded_as_assumed(shared_cell, tmp_path):
    record, specs = _with_pick_events(_resume_only_cell(shared_cell))
    arm_a = record["prepared_recovery_request"]["recovery_resources"]["arm_a"]["recovery_des_model"]
    next(row for row in arm_a["events"] if row["event_name"] == "pick_grasp")["guards"]["task_ctx.part_name"] = {
        "equals_from_param": "part_name"}
    prepared, session = _draft((record, specs), tmp_path)
    _apply(_cell_recovery()[:4], session, prepared)
    decision, audit = asyncio.run(multi_turn_validated_event.handle_validated_candidates(
        session=session, prepared=prepared, planner=SimpleNamespace(product_agent=SimpleNamespace()),
        response={"thought": "The failure is overcome.", "decision": "recovery_complete"}))
    handoff = audit["diagnostic_nominal_handoff"]
    assert decision == "outline_ready" and handoff["complete"] is True
    assert [step["event_name"] for step in handoff["nominal_connection"]] == ["move_home", "pick_approach", "pick_grasp"]
    # The search never reads the arm's own bookkeeping, so the handoff states it as assumed, not checked.
    assert handoff["assumptions"] == [{"outline_id": "NOMINAL_CONNECTION_3", "field": "task_ctx.part_name",
        "reason": "the nominal pick_grasp of arm_a needs its own bookkeeping task_ctx.part_name"}]


def test_a_resource_still_reported_faulted_never_connects(shared_cell, tmp_path):
    prepared, session = _draft(_with_pick_events(_resume_only_cell(shared_cell)), tmp_path)
    _apply(_cell_recovery()[:4], session, prepared)
    prepared["llm_input"]["fault_event"]["detected_signals"] = [
        {"signal": "device_fault_report", "resource_id": "arm_a", "resource_state": "placed"}]
    assert multi_turn_diagnostic.faulted_resources(session, prepared) == {"arm_a"}
    assert multi_turn_diagnostic.nominal_connection(session, prepared) is None
    assert multi_turn_diagnostic.connection_blockers(session, prepared) == [
        "arm_a must do the resumed place_approach but is still reported faulted"]


def test_recovery_complete_ends_the_recovery_and_records_whether_nominal_work_can_take_over(shared_cell, tmp_path):
    planner = SimpleNamespace(product_agent=SimpleNamespace())
    declare = {"thought": "The failure is overcome.", "decision": "recovery_complete"}
    prepared, session = _draft(_with_pick_events(_resume_only_cell(shared_cell)), tmp_path / "declared")
    prepared["llm_input"]["fault_event"]["detected_signals"] = [
        {"signal": "device_fault_report", "resource_id": "arm_b", "resource_state": "failed"}]
    decision, audit = asyncio.run(multi_turn_validated_event.handle_validated_candidates(
        session=session, prepared=prepared, planner=planner, response=declare))
    # A wrong declaration still ends the recovery; the reasons are recorded for review, not fed back.
    assert decision == "outline_ready"
    handoff = audit["diagnostic_nominal_handoff"]
    assert handoff["complete"] is False and handoff["not_connected_reasons"] == [
        "the outcome stated in Recovery Goal does not hold, and no sequence of up to 6 declared nominal events produces it",
        "arm_b did the interrupted work and is still reported faulted"]
    # Where the interrupted step has no declared event, its part at its destination decides; nominal events connect.
    record, specs = _with_pick_events(_resume_only_cell(shared_cell))
    next(node for node in record["pending_nodes"] if node["id"] == "replaced_insert")["function_name"] = "place_release"
    prepared, session = _draft((record, specs), tmp_path / "undeclared")
    _apply(_cell_recovery()[:4], session, prepared)
    assert multi_turn_diagnostic.goal_reached(session, prepared) is True
    decision, audit = asyncio.run(multi_turn_validated_event.handle_validated_candidates(
        session=session, prepared=prepared, planner=planner, response=declare))
    assert decision == "outline_ready"
    assert [step["event_name"] for step in audit["diagnostic_nominal_handoff"]["nominal_connection"]] == [
        "move_home", "pick_approach", "pick_grasp"]


def test_nominal_work_may_redo_the_interrupted_task_once_its_part_is_at_a_declared_place(shared_cell, tmp_path):
    planner = SimpleNamespace(product_agent=SimpleNamespace())
    declare = {"thought": "Nominal work can finish it.", "decision": "recovery_complete"}
    prepared, session = _draft(_with_pick_events(_resume_only_cell(shared_cell)), tmp_path / "loose")
    # The gear lies at no declared place, so no nominal pick can take it.
    decision, audit = asyncio.run(multi_turn_validated_event.handle_validated_candidates(
        session=session, prepared=prepared, planner=planner, response=declare))
    assert decision == "outline_ready" and audit["diagnostic_nominal_handoff"]["complete"] is False
    prepared, session = _draft(_with_pick_events(_resume_only_cell(shared_cell)), tmp_path / "placed")
    _apply([event("gear_to_buffer", "arm_b", "gear", part_state="ready", part_location="buffer")], session, prepared)
    decision, audit = asyncio.run(multi_turn_validated_event.handle_validated_candidates(
        session=session, prepared=prepared, planner=planner, response=declare))
    assert decision == "outline_ready"
    # The interrupted insert is redone with its recorded parameters.
    steps = audit["diagnostic_nominal_handoff"]["nominal_connection"]
    assert [(step["resource_jid"], step["event_name"], step.get("part_name"), step.get("task_id")) for step in steps] == [
        ("arm_b", "move_home", None, None), ("arm_b", "pick_approach", "gear", None),
        ("arm_b", "pick_grasp", "gear", None), ("arm_b", "place_insert", "gear", "replaced_insert")]
    # Each nominal step records only the variables it changes.
    assert [step["expected_end_state"] for step in steps] == [
        {"resource_state": "idle", "resource_location": "home"},
        {"resource_state": "at_pick", "resource_location": "buffer"},
        {"resource_state": "picked", "held_part": "gear", "part_state": "in_gripper", "part_location": "arm_b"},
        {"resource_state": "placed", "held_part": None, "part_state": "assembled", "part_location": "cell"}]


def _with_buffer_place(cell):
    """Add a buffer whose own declared events say arm_a picks only from its zone_2, and move parts there."""
    record, specs = _with_pick_events(_resume_only_cell(cell))
    zones = {name: {"scope": "resource", "type": ["string", "null"], "domain": [None, "peg", "gear"]}
             for name in ("zone_1_part", "zone_2_part")}
    advance = {"event_name": "advance_part", "controllable": True,
               "parameter_bindings": {"resource_id": {"equals": "buffer"}},
               "guards": {"zone_1_part": {"equals_from_param": "part_name"}, "zone_2_part": {"equals": None}},
               "updates": {"zone_1_part": {"set": None}, "zone_2_part": {"set_from_param": "part_name"}}}
    picks = [{"event_name": name, "guards": {"zone_2_part": {"equals_from_param": "part_name"}},
              "updates": {"zone_2_part": {"set": None}} if name == "pick_grasp" else {},
              "parameter_bindings": {"resource_id": {"equals": "arm_a"}, "origin_resource_location": {"equals": "buffer"},
                                     "part_name": {"required": True}}}
             for name in ("pick_approach", "pick_grasp")]
    snapshot = {"resource_jid": "buffer", "zone_1_part": None, "zone_2_part": None}
    request = record["prepared_recovery_request"]
    request["recovery_resources"]["buffer"] = {
        "resource_jid": "buffer", "resource_type": "resource", "recovery_snapshot": snapshot,
        "recovery_des_model": {"state_variables": zones, "events": [advance]},
        "resource_evidence": {"resource_id": "buffer", "environment_models": {"buffer": {"events": [advance, *picks]}}}}
    request["llm_input"]["observed_runtime_state"]["resources"].append(deepcopy(snapshot))
    record["jids"]["buffer"] = "buffer"
    return record, specs


def test_a_place_s_declared_guards_hold_nominal_picks_until_its_own_events_bring_the_part(shared_cell, tmp_path):
    prepared, session = _draft(_with_buffer_place(shared_cell), tmp_path)
    recovery = _cell_recovery()
    # The LLM sets the peg down in the buffer's zone_1; the buffer's own events say arm_a picks from zone_2.
    recovery[1] = {**recovery[1], "neighbor_end_state": {"buffer": {"zone_1_part": "peg"}}}
    _apply(recovery[:4], session, prepared)
    connection = multi_turn_diagnostic.nominal_connection(session, prepared)
    # arm_a's own pick model has no guard about the buffer; the buffer's does, so the buffer acts first.
    assert [(task["resource_jid"], task["event_name"], task.get("part_name")) for task in connection] == [
        ("arm_a", "move_home", None), ("buffer", "advance_part", "peg"), ("arm_a", "pick_approach", "peg"),
        ("arm_a", "pick_grasp", "peg")]
    # A joint step changes the buffer by the buffer's own copy of it, so the zone it leaves is cleared by the model.
    assert connection[-1]["neighbor_end_state"] == {"buffer": {"zone_2_part": None}}
    assert ("buffer", "zone_2_part", None, "model") in [
        (row["entity"], row["variable"], row["value"], row["by"]) for row in connection[-1]["condition_changes"]]


def test_a_new_state_name_does_not_clear_a_reported_fault(shared_cell, tmp_path):
    prepared, session = _draft(shared_cell, tmp_path)
    prepared["llm_input"]["fault_event"]["detected_signals"] = [
        {"signal": "device_fault_report", "resource_id": "arm_b", "resource_state": "failed"}]
    session["recovery_des_models"]["arm_b"]["state_variables"]["resource_state"]["domain"] = ["idle", "picked", "failed"]
    for state, faulted in (("recovery_standby", True), ("failed", True), ("idle", False)):
        session["symbolic_resources"]["arm_b"]["resource_state"] = state
        assert ("arm_b" in multi_turn_diagnostic.faulted_resources(session, prepared)) is faulted, state


def test_custody_is_never_enforced_and_a_part_placed_elsewhere_is_released(shared_cell, tmp_path):
    planner = SimpleNamespace(product_agent=SimpleNamespace())
    prepared, session = _draft(shared_cell, tmp_path / "second_part")
    take = event("take", "arm_a", "gear", resource_state="picked", held_part="gear", part_state="in_gripper",
                 part_location="arm_a")
    decision, audit = asyncio.run(multi_turn_validated_event.handle_validated_candidates(
        session=session, prepared=prepared, planner=planner, response={"thought": "t", "candidate_events": [take]}))
    # Nothing judges the LLM's event: arm_a takes the gear while it still holds the peg, and no note is kept.
    assert decision == "need_next_task" and [row["event_name"] for row in session["accepted_outline_prefix"]] == ["take"]
    assert "custody_notes" not in json.dumps(audit)
    # Placing a held part elsewhere releases it, even when the end state omits held_part.
    prepared, session = _draft(shared_cell, tmp_path / "released")
    set_down = event("set_down", "arm_a", "peg", resource_state="idle", part_state="ready", part_location="buffer")
    decision, _ = asyncio.run(multi_turn_validated_event.handle_validated_candidates(
        session=session, prepared=prepared, planner=planner, response={"thought": "t", "candidate_events": [set_down]}))
    assert decision == "need_next_task" and session["symbolic_resources"]["arm_a"]["held_part"] is None
    # Each changed variable records who set it: the LLM's end state, then what code derived from it.
    changes = session["accepted_outline_prefix"][-1]["condition_changes"]
    assert [(row["entity"], row["variable"], row["value"], row["by"]) for row in changes] == [
        ("arm_a", "resource_state", "idle", "LLM"), ("peg", "part_state", "ready", "LLM"),
        ("peg", "part_location", "buffer", "LLM"), ("arm_a", "held_part", None, "code"), ("peg", "held_by", None, "code")]
    assert changes[3]["reason"] == "peg was placed elsewhere"
    # The part no longer records its former holder either, so the prompt states where it lies.
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes import multi_turn_outline_facts as facts

    assert facts._part_text("peg", prepared, session, facts.resource_names(prepared)) == "ready, at buffer"


def test_a_resource_that_takes_a_part_holds_it_whatever_location_it_states(shared_cell, tmp_path):
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes import multi_turn_outline_facts as facts

    planner = SimpleNamespace(product_agent=SimpleNamespace())
    for directory, part, holder_before in (("loose", "gear", None), ("handover", "peg", "arm_a")):
        prepared, session = _draft(shared_cell, tmp_path / directory)
        # A take may give the part no location (live draft_df4f2479: the KMR took KET4 with part_location null).
        take = event("take", "arm_b", part, resource_state="picked", held_part=part, part_location=None)
        decision, _ = asyncio.run(multi_turn_validated_event.handle_validated_candidates(
            session=session, prepared=prepared, planner=planner, response={"thought": "t", "candidate_events": [take]}))
        assert decision == "need_next_task", directory
        part_row = session["symbolic_parts"][part]
        assert (part_row["part_holder_resource_jid"], part_row["current_holder_resource_jid"]) == ("arm_b", "arm_b")
        assert facts._part_text(part, prepared, session, facts.resource_names(prepared)).endswith("held by arm_b")
        # Only a former holder is released; the taker's own holding is the LLM's value, not code's.
        holding = [(row["entity"], row["variable"], row["value"], row.get("reason"))
                   for row in session["accepted_outline_prefix"][-1]["condition_changes"]
                   if row["by"] == "code" and row["variable"] in ("held_part", "held_by")]
        assert holding == ([(holder_before, "held_part", None, f"{part} is now held by arm_b")] if holder_before else [])


def _belt_cell(tmp_path, *, belt_state="failed"):
    """arm_a must load the peg onto a belt reported broken; the buffer and arm_b resume from the buffer."""
    belt = {"resource_jid": "belt", "resource_type": "resource",
            "recovery_snapshot": {"resource_jid": "belt", "resource_state": belt_state},
            "recovery_des_model": {"state_variables": {"resource_state": {"scope": "resource", "type": "string"}},
                                   "events": [{"event_name": "advance_belt", "updates": {"resource_state": {"set": "running"}}}]}}
    buffer = {"resource_jid": "buffer", "resource_type": "resource",
              "recovery_snapshot": {"resource_jid": "buffer", "resource_state": "idle", "zone_1_part": None},
              "recovery_des_model": {"state_variables": {"resource_state": {"scope": "resource", "type": "string"},
                                                         "zone_1_part": {"scope": "resource", "type": ["string", "null"]}},
                                     "events": [{"event_name": "advance_part",
                                                 "guards": {"zone_1_part": {"equals_from_param": "part_name"}},
                                                 "updates": {"zone_1_part": {"set": None}}}]}}
    resources = {"arm_a": _arm("arm_a", "picked", "station", "peg", {"x": 3.0, "y": 0.0, "z": 1.3}),
                 "arm_b": _arm("arm_b", "idle", "buffer", None, {"x": 6.0, "y": 0.0, "z": 1.3}), "belt": belt,
                 "buffer": buffer}
    conditions = [
        {"role": "resume_suffix", "entity_kind": "resource", "entity": "arm_a", "field": "current_state", "expected": "picked",
         "source_task_id": "load"},
        {"role": "resume_suffix", "entity_kind": "part", "entity": "peg", "field": "state", "expected": "in_gripper",
         "source_task_id": "load"},
        {"role": "resume_suffix", "entity_kind": "resource", "entity": "arm_b", "field": "current_state", "expected": "idle",
         "source_task_id": "pick"},
        {"role": "resume_suffix", "entity_kind": "part", "entity": "peg", "field": "location", "expected": "buffer",
         "source_task_id": "pick"}]
    record = {"scenario": "Conveyor breakdown", "run_id": "supplied_fixture", "captured_at": "recorded-time",
        "jids": {"arm_a": "arm_a", "arm_b": "arm_b", "belt": "belt", "buffer": "buffer"},
        "pending_nodes": [
            {"id": "load", "function_name": "place_approach",
             "params": {"resource_id": "arm_a", "destination_location": "belt", "part_name": "peg"}},
            {"id": "advance", "function_name": "advance_belt", "params": {"resource_id": "belt", "delivered_part": "peg"}},
            {"id": "shift", "function_name": "advance_part", "params": {"resource_id": "buffer", "part_name": "peg"}},
            {"id": "pick", "function_name": "pick_approach",
             "params": {"resource_id": "arm_b", "origin_resource_location": "buffer", "part_name": "peg"}}],
        "prepared_recovery_request": {"ra_jid": "belt", "recovery_resources": resources,
            "marked_reentry_context": {"pending_suffixes": [
                {"role": "resume_suffix", "resource_jid": "arm_a", "entry_task_id": "load", "pending_task_ids": ["load"],
                 "parts": ["peg"], "required_location": "belt"},
                {"role": "recovery_replaced", "resource_jid": "belt", "pending_task_ids": ["advance"], "parts": []},
                {"role": "resume_suffix", "resource_jid": "buffer", "entry_task_id": "shift", "pending_task_ids": ["shift"],
                 "parts": ["peg"]},
                {"role": "resume_suffix", "resource_jid": "arm_b", "entry_task_id": "pick", "pending_task_ids": ["pick"],
                 "parts": ["peg"]}],
                "marked_reentry_conditions": conditions, "unmet_reentry_conditions": conditions[3:]},
            "llm_input": {"fault_event": {"focused_resource_jid": "belt", "detected_signals": [
                    {"signal": "device_fault_report", "resource_id": "belt", "resource_state": "failed"}]},
                "observed_runtime_state": {"resources": [row["recovery_snapshot"] for row in resources.values()]},
                "part_facts": [{"part_name": "peg", "part_state": "in_gripper", "current_location": "arm_a",
                                "current_holder_resource_jid": "arm_a"}]}}}
    path = tmp_path / "observations.json"
    path.write_text(json.dumps(record))
    record.update(source_path=str(path), source_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    return record, []


def test_resumed_work_that_needs_a_faulted_resource_is_replaced_and_the_goal_is_the_handover(tmp_path):
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes import multi_turn_outline_facts as facts

    draft = outline_draft.OutlineDraft(*_belt_cell(tmp_path), root=tmp_path / "drafts")
    prepared = draft.prepared
    session = multi_turn.build_multi_turn_session_seed(prepared)
    # arm_a is healthy, but its next task loads the broken belt, so it is replaced work now.
    assert draft.record["resumed_tasks_replaced"] == ["load"]
    roles = [(row["role"], row["resource_jid"]) for row in prepared["marked_reentry_context"]["pending_suffixes"]]
    assert roles == [("recovery_replaced", "arm_a"), ("recovery_replaced", "belt"), ("resume_suffix", "buffer"),
                     ("resume_suffix", "arm_b")]
    assert [(row["entity"], row["field"]) for row in multi_turn_diagnostic.recorded_conditions(prepared)] == [
        ("arm_b", "current_state"), ("peg", "location")]
    # The goal is what the peg's next remaining step needs, not the broken belt's own fields: the
    # buffer slot its declared guard requires. arm_b's later pick follows from nominal work.
    assert [identifier for _, identifier in multi_turn_diagnostic.resume_entries(prepared)] == ["shift"]
    assert facts.goal_lines(prepared, session) == ["- buffer: zone_1_part peg (normally belt's advance_belt; belt failed)"]
    assert multi_turn_diagnostic.goal_reached(session, prepared) is False
    _apply([event("peg_to_buffer", "arm_a", "peg", resource_state="placed", held_part=None, part_state="ready",
                  part_location="buffer")], session, prepared)
    assert multi_turn_diagnostic.goal_reached(session, prepared) is False
    _apply([event("peg_received", "buffer", "peg", zone_1_part="peg")], session, prepared)
    assert multi_turn_diagnostic.goal_reached(session, prepared) is True


def test_resumed_work_stays_when_it_needs_no_faulted_resource(shared_cell, tmp_path):
    record, specs = _resume_only_cell(shared_cell)
    record["prepared_recovery_request"]["llm_input"]["fault_event"]["detected_signals"] = [
        {"signal": "device_fault_report", "resource_id": "arm_b", "resource_state": "failed"}]
    draft = outline_draft.OutlineDraft(record, specs, root=tmp_path / "drafts")
    assert "resumed_tasks_replaced" not in draft.record
    assert draft.prepared["marked_reentry_context"] == record["prepared_recovery_request"]["marked_reentry_context"]
    # A healthy belt leaves the recorded resume point unchanged.
    (tmp_path / "healthy").mkdir()
    healthy = outline_draft.OutlineDraft(*_belt_cell(tmp_path / "healthy", belt_state="idle"), root=tmp_path / "drafts")
    assert "resumed_tasks_replaced" not in healthy.record


def test_a_resource_lookup_lists_only_the_tasks_normal_production_can_still_run(tmp_path):
    record, specs = _belt_cell(tmp_path)
    resources = record["prepared_recovery_request"]["recovery_resources"]
    for node in record["pending_nodes"]:
        resources[node["params"]["resource_id"]].setdefault("pending_tasks", []).append({**node, "status": "pending"})
    prepared = outline_draft.OutlineDraft(record, specs, root=tmp_path / "drafts").prepared
    session = multi_turn.build_multi_turn_session_seed(prepared)

    def pending(state, jid):
        return [task["id"] for task in multi_turn_context.resolve_outline_context(
            prepared, state, f"/resources/{jid}/pending_tasks")["value"]]

    # arm_a's load and the belt's own advance both need the broken belt, so only the recovery replaces them.
    assert {jid: pending(session, jid) for jid in ("arm_a", "belt", "buffer", "arm_b")} == {
        "arm_a": [], "belt": [], "buffer": ["shift"], "arm_b": ["pick"]}
    # Validated runs keep every recorded task.
    assert pending({**session, "outline_diagnostic_bypass": False}, "arm_a") == ["load"]
    # Once an event takes the belt out of its reported fault, nominal work may redo that work again.
    _apply([event("restart_belt", "belt", None, resource_state="running")], session, prepared)
    assert (pending(session, "arm_a"), pending(session, "belt")) == (["load"], ["advance"])


def test_nominal_work_restarts_at_each_parts_earliest_remaining_step():
    prepared = {"saved_pending_nodes": [
            {"id": "carry", "function_name": "advance", "params": {"resource_id": "belt", "next_locations": {"peg": "end"}}},
            {"id": "shift", "function_name": "advance_part", "params": {"resource_id": "buffer", "part_name": "peg"}},
            {"id": "home", "function_name": "move_home", "params": {"resource_id": "arm"}}],
        "marked_reentry_context": {"pending_suffixes": [
                {"role": "resume_suffix", "resource_jid": "buffer", "entry_task_id": "shift"},
                {"role": "resume_suffix", "resource_jid": "belt", "entry_task_id": "carry"},
                {"role": "resume_suffix", "resource_jid": "arm", "entry_task_id": "home"}],
            "marked_reentry_conditions": [
                {"role": "resume_suffix", "entity_kind": "part", "entity": "peg", "field": "location", "expected": "buffer",
                 "source_task_id": "shift"},
                {"role": "resume_suffix", "entity_kind": "resource", "entity": "arm", "field": "current_state",
                 "expected": "idle", "source_task_id": "home"}]},
        "llm_input": {"part_facts": [{"part_name": "peg"}]}}
    # The belt names the peg only inside next_locations, yet it is the peg's earliest remaining step;
    # the buffer's later step and its recorded condition follow from nominal work.
    assert [identifier for _, identifier in multi_turn_diagnostic.resume_entries(prepared)] == ["carry", "home"]
    assert [row["source_task_id"] for row in multi_turn_diagnostic.entry_conditions(prepared)] == ["home"]


def test_a_choice_by_index_accepts_candidates_that_share_a_label(captured):
    prepared, _ = outline_draft._inputs(captured)
    session = multi_turn.build_multi_turn_session_seed(prepared)
    first, second = event("noop", "machine", None, resource_state="loaded"), sequence()[0]
    second = {**second, "outline_id": first["outline_id"]}
    status, _ = asyncio.run(multi_turn_validated_event.handle_validated_candidates(
        session=session, prepared=prepared, planner=SimpleNamespace(product_agent=SimpleNamespace()),
        response={"thought": "Two options under one label.", "candidate_events": [first, second], "selected_candidate_index": 1}))
    assert status == "need_next_task" and session["accepted_outline_prefix"][0]["event_name"] == second["event_name"]


def test_a_base_motion_primitive_locates_a_mobile_resource_by_its_pose():
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes.multi_turn_outline_facts import (
        _coordinate_parameters,
    )
    from cais_spade_llm.function_analyzer import FunctionAnalyzer
    from cais_spade_llm.recovery_framework.kmr_primitives import KMRPrimitives

    # The KMR declares where move_base takes its base; no declared event shortcuts the travel.
    effects = FunctionAnalyzer._extract_yaml_frontmatter(KMRPrimitives.move_base)["effects"]
    prepared = {"recovery_resources": {"KMR": {"recovery_des_model": {"state_variables": {}, "events": []}}},
                "llm_input": {}}
    assert multi_turn_diagnostic.declare_derived_conditions(
        prepared, {"KMR": [{"name": "move_base", "effects": effects}]}) == {"KMR": ["pose"]}
    model = prepared["recovery_resources"]["KMR"]["recovery_des_model"]
    assert model["state_variables"]["pose"]["fields"] == ["x", "y", "yaw"]
    assert model["state_variables"]["pose"]["locates_resource"] is True
    assert _coordinate_parameters(model) == ["x", "y", "yaw"]

def test_conditions_come_from_each_resources_model_and_primitives():
    cartesian = {"name": "move_cartesian", "effects": {"current_pose": {"pose_absolute_from_params": ["x", "y", "z"]}}}
    mover_event = {"event_name": "move_to_location", "updates": {"resource_location": {"set_from_param": "observed"}},
                   "parameter_bindings": {name: {"required": True, "type": "number"} for name in ("x", "y", "yaw")}}
    machine_event = {"event_name": "machine_part", "updates": {"resource_state": {"set": "completed"}},
                     "product_effects": {"processCompleted": [{"process": "trim", "result": {"set_from_param": "result"}}]}}
    prepared = {"recovery_resources": {
            "arm": {"recovery_des_model": {"state_variables": {"held_part": {"scope": "resource"}}}},
            "mover": {"recovery_des_model": {"state_variables": {}, "events": [mover_event]}},
            "machine": {"recovery_des_model": {"state_variables": {}, "events": [machine_event]}},
            "station": {"recovery_des_model": {"state_variables": {"resource_state": {"scope": "resource"}}}}},
        "part_tracker": {"peg": {"processCompleted": [{"process": "trim", "result": "square"}]}},
        "llm_input": {"part_facts": [{"part_name": "peg"}]}}
    added = multi_turn_diagnostic.declare_derived_conditions(prepared, {"arm": [cartesian], "mover": [{"name": "move_base"}]})
    # The same rule for every resource: a primitive's pose effect, else coordinate inputs; a declared process record.
    assert added == {"arm": ["pose"], "mover": ["pose"], "machine": ["processCompleted"]}
    variables = {jid: entry["recovery_des_model"]["state_variables"] for jid, entry in prepared["recovery_resources"].items()}
    assert variables["arm"]["pose"]["fields"] == ["x", "y", "z"] and variables["mover"]["pose"]["fields"] == ["x", "y", "yaw"]
    assert variables["machine"]["processCompleted"]["scope"] == "part" and "pose" not in variables["station"]
    assert prepared["llm_input"]["part_facts"][0]["processCompleted"] == [{"process": "trim", "result": "square"}]
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes.multi_turn_prompts import _outline_state_field_schema

    assert _outline_state_field_schema("pose", variables["mover"]["pose"]) == {
        "type": "object", "properties": {name: {"type": "number"} for name in ("x", "y", "yaw")},
        "required": ["x", "y", "yaw"], "additionalProperties": False}


def test_only_an_event_that_adds_to_a_process_record_makes_it_a_resource_condition(captured):
    prepared, _ = outline_draft._inputs(captured)
    # Every declared event carries a processCompleted list; an empty one leaves the record as it is.
    prepared["recovery_resources"]["arm"]["recovery_des_model"]["events"] = [
        {"event_name": "carry", "updates": {"resource_state": {"set": "carrying"}},
         "product_effects": {"processCompleted": []}}]
    assert multi_turn_diagnostic.declare_derived_conditions(prepared, {}) == {"machine": ["processCompleted"]}
    session = multi_turn.build_multi_turn_session_seed(prepared)
    part = multi_turn_context.resolve_outline_context(prepared, session, "/parts/KET4_Square_4mm/state_variables")
    assert part["value"]["processCompleted"]["set_by"] == ["machine"]
    # Only the resource whose events add to the record lists it among the variables the LLM looks up.
    offered = {jid for jid in prepared["recovery_resources"]
               if "processCompleted" in multi_turn_context.resolve_outline_context(
                   prepared, session, f"/resources/{jid}/state_variables")["value"]}
    assert offered == {"machine"}


def test_a_process_record_is_a_part_condition(captured):
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes import multi_turn_outline_facts as facts

    prepared, _ = outline_draft._inputs(captured)
    for model_event in prepared["recovery_resources"]["machine"]["recovery_des_model"]["events"]:
        model_event["product_effects"] = {"processCompleted": [{"process": "trim", "result": {"set_from_param": "result"}}]}
    multi_turn_diagnostic.declare_derived_conditions(prepared, {})
    session = multi_turn.build_multi_turn_session_seed(prepared)
    part = multi_turn_context.resolve_outline_context(prepared, {**session, "outline_diagnostic_bypass": True},
                                                      "/parts/KET4_Square_4mm/state_variables")
    assert "machine" in part["value"]["processCompleted"]["set_by"] and part["values_rule"]
    _apply([event("trimmed", "machine", "KET4_Square_4mm", resource_state="completed",
                  processCompleted=[{"process": "trim", "result": "square"}])], session, prepared)
    names = facts.resource_names(prepared)
    assert facts._part_text("KET4_Square_4mm", prepared, session, names).endswith("; processCompleted: trim square")


def test_a_stated_pose_moves_a_resource_out_of_a_shared_area(shared_cell, tmp_path):
    prepared, _ = _draft(shared_cell, tmp_path)
    cartesian = {"name": "move_cartesian", "effects": {"current_pose": {"pose_absolute_from_params": ["x", "y", "z"]}}}
    multi_turn_diagnostic.declare_derived_conditions(prepared, {"arm_a": [cartesian], "arm_b": [cartesian]})
    session = multi_turn.build_multi_turn_session_seed(prepared)
    _apply(_cell_recovery()[1:], session, prepared)

    def regions():
        return multi_turn_context.resolve_outline_context(prepared, session, "/resources/arm_b/pose_relations")["value"][
            "containing_regions"]

    assert regions() == ["cell"]
    # arm_b never changed its location, but its stated pose now lies outside the cell.
    _apply([event("clear_by_pose", "arm_b", None, resource_state="idle", pose={"x": 3.0, "y": 3.0, "z": 1.2})],
           session, prepared)
    assert regions() == []
    # Code keeps the location consistent with the stated pose and says why.
    assert [(row["variable"], row["value"], row["by"], row.get("reason"))
            for row in session["accepted_outline_prefix"][-1]["condition_changes"]] == [
        ("resource_state", "idle", "LLM", None), ("pose", {"x": 3.0, "y": 3.0, "z": 1.2}, "LLM", None),
        ("resource_location", None, "code", "no named place's recorded area contains the stated pose")]
    _apply([event("return_by_pose", "arm_b", None, resource_state="idle", pose={"x": 0.0, "y": 0.1, "z": 1.2})],
           session, prepared)
    assert session["symbolic_resources"]["arm_b"]["resource_location"] == "cell"
    assert session["accepted_outline_prefix"][-1]["condition_changes"][-1]["reason"] == "the stated pose is inside cell"


def _pose_draft(shared_cell, tmp_path):
    """Draft the shared cell with both arms declaring the pose their primitives' current_pose effect sets."""
    prepared, _ = _draft(shared_cell, tmp_path)
    cartesian = {"name": "move_cartesian", "effects": {"current_pose": {"pose_absolute_from_params": ["x", "y", "z"]}}}
    multi_turn_diagnostic.declare_derived_conditions(prepared, {"arm_a": [cartesian], "arm_b": [cartesian]})
    return prepared, multi_turn.build_multi_turn_session_seed(prepared)


def test_a_first_stated_pose_starts_from_the_recorded_pose(shared_cell, tmp_path):
    prepared, session = _pose_draft(shared_cell, tmp_path)
    _apply([event("clear_by_pose", "arm_b", None, resource_state="idle", pose={"x": 3.0, "y": 3.0, "z": 1.2})],
           session, prepared)
    # The pose the Resources row shows is the event's start condition, so nothing is assumed about it.
    task = session["accepted_outline_prefix"][-1]
    assert task["expected_start_state"]["pose"] == {"x": 0.0, "y": 0.0, "z": 1.2}
    assert "diagnostic_assumptions" not in task
    # A resource that records no pose has no start value for it.
    session["symbolic_resources"]["arm_a"].pop("current_pose")
    task = multi_turn_diagnostic.event_record(
        event("lift", "arm_a", None, resource_state="picked", pose={"x": 1.0, "y": 0.0, "z": 1.35}), session, prepared)
    assert "pose" not in task["expected_start_state"] and "diagnostic_assumptions" not in task


def test_a_pose_an_earlier_event_stated_is_never_shown_as_measured(shared_cell, tmp_path):
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes import (
        multi_turn_outline_facts as facts,
    )

    prepared, session = _pose_draft(shared_cell, tmp_path)
    names = facts.resource_names(prepared)
    # Moved by its location only, a resource keeps the pose the capture measured, marked as left behind.
    _apply([event("clear", "arm_b", None, resource_state="idle", resource_location="home")], session, prepared)
    assert "; last measured pose (0.000, 0.000, 1.200), inside cell; " in facts._resource_row("arm_b", prepared, session, names)
    # A pose an event stated stays the LLM's value, also once a later location change leaves it behind.
    _apply([event("reach", "arm_a", None, resource_state="picked", pose={"x": 0.1, "y": 0.0, "z": 1.3})], session, prepared)
    assert "; pose (0.100, 0.000, 1.300), inside cell; " in facts._resource_row("arm_a", prepared, session, names)
    _apply([event("back", "arm_a", None, resource_state="picked", resource_location="buffer")], session, prepared)
    row = facts._resource_row("arm_a", prepared, session, names)
    assert "; last stated pose (0.100, 0.000, 1.300), inside cell; " in row and "measured" not in row


@pytest.mark.parametrize("states_release", [True, False])
def test_a_part_keeps_a_stated_pose_only_while_the_stating_resource_holds_it(shared_cell, tmp_path, states_release):
    prepared, session = _pose_draft(shared_cell, tmp_path)
    release = {"held_part": None} if states_release else {}
    # The stated pose is the arm's own after it sets the peg down, not where the peg rests.
    _apply([event("set_down", "arm_a", "peg", resource_state="placed", part_state="ready", part_location="buffer",
                  pose={"x": 1.0, "y": 0.0, "z": 1.5}, **release)], session, prepared)
    peg = session["symbolic_parts"]["peg"]
    assert session["symbolic_resources"]["arm_a"]["held_part"] is None
    assert peg["projected_pose"] is None and peg["observed_pose_current"] is False
    # A part the arm holds moves with it, so it takes the arm's stated pose.
    _apply([event("take", "arm_a", "gear", resource_state="picked", held_part="gear", part_state="in_gripper",
                  part_location="arm_a", pose={"x": 0.9, "y": 0.1, "z": 1.0})], session, prepared)
    assert session["symbolic_parts"]["gear"]["projected_pose"] == {"x": 0.9, "y": 0.1, "z": 1.0}


def test_a_planar_pose_is_not_placed_in_an_area_bounded_in_height(shared_cell, tmp_path):
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes import multi_turn_outline_facts as facts

    record, specs = shared_cell
    mobile = _arm("mobile", "idle", "home", None, {"x": 3.0, "y": 3.0, "z": 0.7})
    mobile["static_capabilities"] = {}
    del mobile["recovery_snapshot"]["current_pose"]
    mobile["recovery_des_model"]["state_variables"]["resource_location"]["domain"] = [None, "home", "cell"]
    mobile["recovery_des_model"]["events"].append({"event_name": "move_to_location",
        "updates": {"resource_location": {"set_from_param": "observed"}},
        "parameter_bindings": {name: {"required": True, "type": "number"} for name in ("x", "y", "yaw")}})
    request = record["prepared_recovery_request"]
    request["recovery_resources"]["mobile"] = mobile
    request["llm_input"]["observed_runtime_state"]["resources"].append(mobile["recovery_snapshot"])
    record["jids"]["mobile"] = "mobile"
    prepared, session = _draft((record, specs), tmp_path)
    # The cell's x and y bounds contain the stated position; its height bound cannot be tested.
    _apply([event("drive", "mobile", None, resource_state="idle", pose={"x": 0.0, "y": 0.1, "yaw": 0.0})],
           session, prepared)
    row = facts._resource_row("mobile", prepared, session, facts.resource_names(prepared))
    assert "; pose (0.000, 0.100, yaw 0.000);" in row and "inside" not in row
    assert [change["value"] for change in session["accepted_outline_prefix"][-1]["condition_changes"]
            if change["variable"] == "resource_location"] == [None]
    relations = multi_turn_context.resolve_outline_context(prepared, session, "/resources/mobile/pose_relations")["value"]
    assert relations["containing_regions"] == [] and "mobile" in relations["not_evaluated"]


def test_a_place_a_part_enters_has_only_its_slots_for_that_part_and_its_state_unknown(tmp_path):
    record, specs = _belt_cell(tmp_path)
    buffer = record["prepared_recovery_request"]["recovery_resources"]["buffer"]
    buffer["recovery_des_model"]["state_variables"].update(
        zone_1_part={"scope": "resource", "type": ["string", "null"], "domain": [None, "peg"]},
        job_state={"scope": "resource", "type": "string", "domain": ["idle", "running"]})
    buffer["recovery_snapshot"]["job_state"] = "idle"
    prepared = outline_draft.OutlineDraft(record, specs, root=tmp_path / "drafts").prepared
    session = multi_turn.build_multi_turn_session_seed(prepared)
    _apply([event("peg_to_buffer", "arm_a", "peg", resource_state="placed", held_part=None, part_state="ready",
                  part_location="buffer")], session, prepared)
    # The event does not state how the buffer changed, so only what could record the peg is unknown.
    assert [(row["variable"], row["value"], row["by"], row["reason"])
            for row in session["accepted_outline_prefix"][-1]["condition_changes"] if row["entity"] == "buffer"] == [
        ("resource_state", "unknown", "code", "peg arrived there; its own change was not stated"),
        ("zone_1_part", "unknown", "code", "peg arrived there; its own change was not stated")]
    assert session["symbolic_resources"]["buffer"]["job_state"] == "idle"


def test_a_part_s_own_entry_on_a_place_reads_not_stated_until_the_event_states_it(tmp_path):
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes import multi_turn_context as context
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes import multi_turn_outline_facts as facts

    record, specs = _belt_cell(tmp_path)
    belt = record["prepared_recovery_request"]["recovery_resources"]["belt"]
    belt["recovery_des_model"]["state_variables"]["part_location.peg"] = {
        "scope": "resource", "type": ["string", "null"], "domain": [None, "loading_position_1", "loading_position_2"]}
    belt["recovery_snapshot"]["part_location.peg"] = None
    prepared = outline_draft.OutlineDraft(record, specs, root=tmp_path / "drafts").prepared
    names = facts.resource_names(prepared)
    unstated, stated = (multi_turn.build_multi_turn_session_seed(prepared) for _ in range(2))
    # An empty record is shown with the values it can take, so a part move's change is in view.
    assert facts._resource_row("belt", prepared, unstated, names).endswith(
        "part_location.<part name>: none (values loading_position_1, loading_position_2)")
    load = event("peg_on_belt", "arm_a", "peg", resource_state="placed", held_part=None, part_state="ready",
                 part_location="belt")
    _apply([load], unstated, prepared)
    _apply([{**load, "neighbor_end_state": {"belt": {"part_location.peg": "loading_position_1"}}}], stated, prepared)
    # Unstated, the peg's own entry reads "not stated" instead of keeping its old empty value.
    assert "part_location.peg" in unstated["diagnostic_unknown_owner_fields"]["belt"]
    assert "part_location.<part name>: peg not stated" in facts._resource_row("belt", prepared, unstated, names)
    assert context.public_resource_snapshot(prepared, unstated, "belt")["part_location.peg"] == "not stated"
    assert "part_location.<part name>: peg loading_position_1" in facts._resource_row("belt", prepared, stated, names)


def test_a_resource_lookup_states_current_values_and_what_events_left_unstated(tmp_path):
    record, specs = _belt_cell(tmp_path)
    buffer = record["prepared_recovery_request"]["recovery_resources"]["buffer"]
    buffer["recovery_des_model"]["state_variables"]["zone_1_part"]["domain"] = [None, "peg"]
    prepared = outline_draft.OutlineDraft(record, specs, root=tmp_path / "drafts").prepared
    cartesian = {"name": "move_cartesian", "effects": {"current_pose": {"pose_absolute_from_params": ["x", "y", "z"]}}}
    multi_turn_diagnostic.declare_derived_conditions(prepared, {"arm_a": [cartesian]})
    session = multi_turn.build_multi_turn_session_seed(prepared)

    def current(jid, state=session):
        variables = multi_turn_context.resolve_outline_context(prepared, state, f"/resources/{jid}/state_variables")
        return {name: row["current_value"] for name, row in variables["value"].items() if "current_value" in row}

    def accept(row):
        decision, _ = asyncio.run(multi_turn_validated_event.handle_validated_candidates(
            session=session, prepared=prepared, planner=SimpleNamespace(product_agent=SimpleNamespace()),
            response={"thought": "Next step.", "candidate_events": [row], "selected_candidate_index": 0}))
        assert decision != "need_revision"

    # Recorded values; pose is the measured pose in its declared fields; part variables stay on the parts.
    assert current("arm_a") == {"resource_state": "picked", "resource_location": "station", "held_part": "peg",
                                "pose": {"x": 3.0, "y": 0.0, "z": 1.3}}
    accept(event("peg_to_buffer", "arm_a", "peg", resource_state="placed", held_part=None, part_state="ready",
                 part_location="buffer", resource_location="buffer"))
    # The buffer's own change was not stated, and a move by location alone leaves the pose unstated.
    assert current("buffer") == {"resource_state": "not stated", "zone_1_part": "not stated"}
    assert current("arm_a") == {"resource_state": "placed", "resource_location": "buffer", "held_part": None,
                                "pose": "not stated"}
    accept(event("lift", "arm_a", None, resource_state="idle", pose={"x": 3.0, "y": 0.5, "z": 1.4}))
    assert current("arm_a")["pose"] == {"x": 3.0, "y": 0.5, "z": 1.4}
    # Validated runs serve the declarations only, as their prompt says.
    assert current("arm_a", {**session, "outline_diagnostic_bypass": False}) == {}


def test_a_snapshot_reads_not_stated_for_what_an_accepted_event_left_unstated(tmp_path):
    record, specs = _belt_cell(tmp_path)
    buffer = record["prepared_recovery_request"]["recovery_resources"]["buffer"]
    buffer["recovery_des_model"]["state_variables"]["zone_1_part"]["domain"] = [None, "peg"]
    prepared = outline_draft.OutlineDraft(record, specs, root=tmp_path / "drafts").prepared
    session = multi_turn.build_multi_turn_session_seed(prepared)
    set_down = event("peg_to_buffer", "arm_a", "peg", resource_state="placed", held_part=None, part_state="ready",
                     part_location="buffer")
    # Accepted through the real handler, which switches lookups to the projected rows.
    decision, _ = asyncio.run(multi_turn_validated_event.handle_validated_candidates(
        session=session, prepared=prepared, planner=SimpleNamespace(product_agent=SimpleNamespace()),
        response={"thought": "Set the peg down.", "candidate_events": [set_down], "selected_candidate_index": 0}))
    assert decision != "need_revision" and "buffer" in session["derived_resource_snapshots"]
    snapshot = multi_turn_context.resolve_outline_context(prepared, session, "/resources/buffer/snapshot")["value"]
    assert (snapshot["resource_state"], snapshot["zone_1_part"]) == ("not stated", "not stated")


def test_a_place_inside_a_resource_changes_only_the_owner_record_its_events_declare(tmp_path):
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes import multi_turn_outline_facts as facts

    record, specs = _belt_cell(tmp_path)
    buffer = record["prepared_recovery_request"]["recovery_resources"]["buffer"]
    buffer["recovery_des_model"]["state_variables"]["staging_part"] = {
        "scope": "resource", "type": ["string", "null"], "domain": [None, "peg"]}
    buffer["recovery_snapshot"]["staging_part"] = None
    # The buffer's own declared events record a part another resource places on its tray.
    buffer["resource_evidence"] = {"resource_id": "buffer", "environment_models": {"buffer": {"events": [
        {"event_name": "place_release", "updates": {"staging_part": {"set_from_param": "part_name"}},
         "parameter_bindings": {"resource_id": {"equals": "arm_a"}, "destination_location": {"equals": "buffer tray"}}}]}}}
    prepared = outline_draft.OutlineDraft(record, specs, root=tmp_path / "drafts").prepared
    names = facts.resource_names(prepared)
    unstated, stated = (multi_turn.build_multi_turn_session_seed(prepared) for _ in range(2))
    assert "staging_part none (buffer tray)" in facts._resource_row("buffer", prepared, unstated, names)
    to_tray = event("peg_to_tray", "arm_a", "peg", resource_state="placed", held_part=None, part_state="ready",
                    part_location="buffer tray")
    _apply([to_tray], unstated, prepared)
    _apply([{**to_tray, "neighbor_end_state": {"buffer": {"staging_part": "peg"}}}], stated, prepared)
    # Only the tray's record is not stated; the rest of the buffer stays as it was.
    assert [(row["variable"], row["value"]) for row in unstated["accepted_outline_prefix"][-1]["condition_changes"]
            if row["entity"] == "buffer"] == [("staging_part", "unknown")]
    assert unstated["symbolic_resources"]["buffer"]["resource_state"] == "idle"
    assert "staging_part not stated (buffer tray)" in facts._resource_row("buffer", prepared, unstated, names)
    assert stated["symbolic_resources"]["buffer"]["staging_part"] == "peg"
    assert not stated.get("diagnostic_unknown_owner_fields", {}).get("buffer")
    # Naming the buffer states its change, so values left out stay as they are, as the rule says.
    named = multi_turn.build_multi_turn_session_seed(prepared)
    _apply([{**to_tray, "neighbor_end_state": {"buffer": {"resource_state": "idle"}}}], named, prepared)
    assert not named.get("diagnostic_unknown_owner_fields", {}).get("buffer")


def test_an_event_may_state_how_moving_its_part_changes_another_resource(tmp_path):
    record, specs = _belt_cell(tmp_path)
    prepared = outline_draft.OutlineDraft(record, specs, root=tmp_path / "drafts").prepared
    unstated, stated = (multi_turn.build_multi_turn_session_seed(prepared) for _ in range(2))
    set_down = event("peg_to_buffer", "arm_a", "peg", resource_state="placed", held_part=None, part_state="ready",
                     part_location="buffer")
    _apply([set_down], unstated, prepared)
    _apply([{**set_down, "neighbor_end_state": {"buffer": {"zone_1_part": "peg"}}}], stated, prepared)
    # The stated slot is the LLM's value; nothing else about the buffer is marked unknown or changed.
    assert [(row["variable"], row["value"], row["by"]) for row in stated["accepted_outline_prefix"][-1]["condition_changes"]
            if row["entity"] == "buffer"] == [("zone_1_part", "peg", "LLM")]
    assert stated["symbolic_resources"]["buffer"]["zone_1_part"] == "peg"
    assert stated["symbolic_resources"]["buffer"]["resource_state"] == "idle"
    # The buffer's next step needs the peg in zone 1, so only the stated slot lets normal production take over.
    assert multi_turn_diagnostic.handoff(unstated, prepared)["complete"] is False
    handoff = multi_turn_diagnostic.handoff(stated, prepared)
    assert handoff["complete"] is True and handoff["assumptions"] == []


def test_a_resumed_entry_needs_what_its_place_declares_for_it(tmp_path):
    record, specs = _belt_cell(tmp_path)
    request = record["prepared_recovery_request"]
    context = request["marked_reentry_context"]
    # Nominal work restarts at arm_b's pick from the buffer, which declares arm_b picks only from zone_1.
    context["pending_suffixes"] = [row for row in context["pending_suffixes"] if row["resource_jid"] != "buffer"]
    request["recovery_resources"]["arm_b"]["recovery_des_model"]["events"].append(
        {"event_name": "pick_approach", "parameter_bindings": {"origin_resource_location": {"location_type": "part_location"}},
         "guards": {"resource_state": {"equals": "idle"}, "held_part": {"equals": None}},
         "updates": {"resource_state": {"set": "at_pick"}, "resource_location": {"set_from_param": "origin_resource_location"}}})
    buffer = request["recovery_resources"]["buffer"]
    buffer["recovery_des_model"]["state_variables"]["zone_2_part"] = {"scope": "resource", "type": ["string", "null"]}
    buffer["recovery_snapshot"]["zone_2_part"] = None
    buffer["resource_evidence"] = {"resource_id": "buffer", "environment_models": {"buffer": {"events": [
        {"event_name": "pick_approach", "guards": {"zone_1_part": {"equals_from_param": "part_name"}}, "updates": {},
         "parameter_bindings": {"resource_id": {"equals": "arm_b"}, "origin_resource_location": {"equals": "buffer"}}}]}}}
    prepared = outline_draft.OutlineDraft(record, specs, root=tmp_path / "drafts").prepared
    set_down = event("peg_to_buffer", "arm_a", "peg", resource_state="placed", held_part=None, part_state="ready",
                     part_location="buffer")
    wrong_zone, right_zone = (multi_turn.build_multi_turn_session_seed(prepared) for _ in range(2))
    _apply([{**set_down, "neighbor_end_state": {"buffer": {"zone_2_part": "peg"}}}], wrong_zone, prepared)
    _apply([{**set_down, "neighbor_end_state": {"buffer": {"zone_1_part": "peg"}}}], right_zone, prepared)
    # The recorded entry conditions hold either way; only the buffer's own declared guard tells them apart.
    wrong = multi_turn_diagnostic.handoff(wrong_zone, prepared)
    assert wrong["unmet_handoff_conditions"] == [] and wrong["complete"] is False
    assert multi_turn_diagnostic.handoff(right_zone, prepared)["complete"] is True


def test_a_stated_holding_change_on_another_resource_is_also_the_parts(shared_cell, tmp_path):
    prepared, session = _draft(shared_cell, tmp_path)
    give = {**event("give", "arm_a", "peg", resource_state="idle", held_part=None, part_state="in_gripper",
                    part_location="arm_b"), "neighbor_end_state": {"arm_b": {"held_part": "peg"}}}
    _apply([give], session, prepared)
    # Who holds the part agrees on the resource and on the part.
    part = session["symbolic_parts"]["peg"]
    assert session["symbolic_resources"]["arm_b"]["held_part"] == "peg"
    assert part["part_holder_resource_jid"] == part["current_holder_resource_jid"] == "arm_b"
    # A take-over that states the former holder's release moves the holding back on both sides.
    take = {**event("take", "arm_a", "peg", resource_state="picked", held_part="peg", part_state="in_gripper",
                    part_location="arm_a"), "neighbor_end_state": {"arm_b": {"held_part": None}}}
    _apply([take], session, prepared)
    part = session["symbolic_parts"]["peg"]
    assert (session["symbolic_resources"]["arm_a"]["held_part"], session["symbolic_resources"]["arm_b"]["held_part"]) == (
        "peg", None)
    assert part["part_holder_resource_jid"] == part["current_holder_resource_jid"] == "arm_a"
    # A stated taker stays the holder even when the part is given no location; only the giver is released.
    prepared, session = _draft(shared_cell, tmp_path / "no_location")
    give = {**event("give", "arm_a", "peg", resource_state="idle", part_location=None),
            "neighbor_end_state": {"arm_b": {"held_part": "peg"}}}
    _apply([give], session, prepared)
    part = session["symbolic_parts"]["peg"]
    assert (session["symbolic_resources"]["arm_a"]["held_part"], session["symbolic_resources"]["arm_b"]["held_part"]) == (
        None, "peg")
    assert part["part_holder_resource_jid"] == part["current_holder_resource_jid"] == "arm_b"
    assert [(row["entity"], row["variable"], row["value"], row.get("reason"))
            for row in session["accepted_outline_prefix"][-1]["condition_changes"] if row["by"] == "code"] == [
        ("arm_a", "held_part", None, "peg is now held by arm_b")]


def test_the_draft_table_lists_each_end_state_value_and_marks_what_code_added():
    from cais_spade_llm.ui.components.part_slippage_recovery import draft_rows

    draft = {"transition_trace": [{"event_name": "set_down", "resource": "ur5e-3", "part_name": "KET4",
                                   "expected_end_state": {"part_location": "buffer"}, "rationale": "free the gripper",
                                   "condition_changes": [
                                       {"entity": "KET4", "variable": "part_location", "value": "buffer", "by": "LLM"},
                                       {"entity": "ur5e-3", "variable": "held_part", "value": None, "by": "code",
                                        "reason": "KET4 was placed elsewhere"}]}],
             "nominal_handoff": {"nominal_connection": [{"event_name": "pick_grasp", "resource": "ur5e-3",
                                                         "part_name": "KET4", "expected_end_state": {"held_part": "KET4"}}]}}
    rows = draft_rows(draft)
    # Only recovery events are rows; the nominal steps after them stay in the handoff record.
    assert [row["event_name"] for row in rows] == ["set_down"]
    # The LLM's values carry no label; a value code added says so, with its reason.
    assert rows[0]["end_state"].splitlines() == ["KET4  part_location = buffer",
                                                  "ur5e-3  held_part = none  — code (KET4 was placed elsewhere)"]
    # Older drafts listed claims of interrupted work as rows; the table leaves them out.
    claimed = draft_rows({"transition_trace": [{"event_name": "set_down", "resource": "ur5e-3", "condition_changes": [
        {"entity": "KET4", "variable": "part_location", "value": "buffer", "by": "LLM"},
        {"entity": "interrupted place_insert", "variable": "completed", "value": True, "by": "code",
         "reason": "its declared outcome now holds"}]}]})
    assert claimed[0]["end_state"] == "KET4  part_location = buffer"
    # A draft recorded before per-variable attribution shows the LLM's end state as written.
    earlier = draft_rows({"transition_trace": [{"event_name": "set_down", "resource": "ur5e-3",
                                                "expected_end_state": {"part_location": "buffer"}}]})
    assert earlier[0]["end_state"] == "ur5e-3  part_location = buffer"

