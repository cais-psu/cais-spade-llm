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
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes import multi_turn_nominal_evidence
    from cais_spade_llm.recovery_framework import offline_recovery, outline_endpoint_evidence, primitive_composition

    forbidden = Mock(side_effect=AssertionError("Diagnostic must not invoke validators or execution"))
    for module, name in ((multi_turn_validated_event, "validate_event"),
                         (multi_turn_nominal_evidence, "validate_nominal_continuation"),
                         (offline_recovery, "create_offline_recovery"),
                         (outline_endpoint_evidence, "project_bounded_successor"),
                         (primitive_composition, "start_composition_attempt"),
                         (outline_draft.context, "public_outline_safety_context")):
        monkeypatch.setattr(module, name, forbidden)
    replies = [{"thought": "Use recorded facts.", "decision": "grounded", "observations": []},
        {"thought": "Request facts.", "decision": "need_context",
         "context_requests": ["/resources/machine/snapshot", "/safety_rules"]},
        *[{"thought": "Propose one task-level event.", "candidate_events": [row]} for row in sequence()]]
    provider = AsyncMock(side_effect=replies)
    monkeypatch.setattr(outline_draft.SavedRecoveryProduct, "ask_llm_structured", provider)
    execute = AsyncMock(wraps=multi_turn.execute_multi_turn_recovery)
    monkeypatch.setattr(multi_turn, "execute_multi_turn_recovery", execute)
    before = deepcopy(captured)
    draft = outline_draft.OutlineDraft(captured, [], root=tmp_path / "drafts")
    asyncio.run(draft.generate())
    assert draft.record["status"] == "draft", draft.record["message"]
    assert len(draft.record["transition_trace"]) == 6
    assert draft.record["provider_calls"] == provider.await_count == 8
    assert execute.await_count >= 6
    assert draft.record["nominal_handoff"]["complete"] is True
    assert draft.record["nominal_handoff"]["assumptions"]
    assert captured == before
    assert "/resources/machine/snapshot" in draft.record["retrievals"]
    assert set(draft.record["checks"].values()) == {"not_checked"}
    assert draft.record["execution_authorized"] is False
    forbidden.assert_not_called()
    for call in provider.call_args_list:
        prompt = call.kwargs["prompt"] if "prompt" in call.kwargs else call.args[0]
        assert not any(token in prompt for token in ("gripper_reach", "place_release", "active_job", "private_job"))


@pytest.mark.parametrize("reverse", [False, True])
def test_diagnostic_candidates_are_isolated_and_order_independent(captured, reverse):
    prepared, _ = outline_draft._inputs(captured)
    session = multi_turn.build_multi_turn_session_seed(prepared)
    candidates = [event("noop", "machine", None, resource_state="loaded"), sequence()[0]]
    if reverse:
        candidates.reverse()
    before = deepcopy(prepared)
    status, audit = asyncio.run(multi_turn_validated_event.handle_validated_candidates(
        session=session, prepared=prepared, planner=SimpleNamespace(product_agent=SimpleNamespace()),
        response={"thought": "Compare supplied alternatives.", "candidate_events": candidates}))
    assert status == "need_next_task"
    assert len(session["accepted_outline_prefix"]) == 1
    assert session["accepted_outline_prefix"][0]["event_name"] == "pause"
    assert session["symbolic_resources"]["machine"]["job_state"] == "paused"
    assert prepared == before
    assert all(row["validation_audit"]["transition_validation"]["status"] == "not_checked" for row in audit["candidate_evaluations"])


def test_handoff_requires_replaced_work_and_restored_part(captured):
    prepared, _ = outline_draft._inputs(captured)
    session = multi_turn.build_multi_turn_session_seed(prepared)
    assert multi_turn_diagnostic.handoff(session, prepared)["complete"] is False
    for row in sequence()[:-1]:
        task, findings = multi_turn._derive_candidate_outline_task(
            candidate_task=row, session_state=session, prepared_recovery_request=prepared)
        assert not findings
        multi_turn_diagnostic.apply_tentative_effects(task, session, prepared)
        assert multi_turn_diagnostic.handoff(session, prepared)["complete"] is False
    task, _ = multi_turn._derive_candidate_outline_task(
        candidate_task=sequence()[-1], session_state=session, prepared_recovery_request=prepared)
    multi_turn_diagnostic.apply_tentative_effects(task, session, prepared)
    assert multi_turn_diagnostic.handoff(session, prepared)["complete"] is True
    assert session["symbolic_parts"]["KET4_Square_4mm"]["processCompleted"] == [{"process": "trim", "result": "square"}]


def test_diagnostic_flag_defaults_off(captured):
    prepared, _ = outline_draft._inputs(captured)
    prepared["recovery_session"].pop("outline_diagnostic_bypass")
    assert multi_turn.build_multi_turn_session_seed(prepared)["outline_diagnostic_bypass"] is False


def test_repeated_candidates_stop_without_mutating_prefix(captured):
    prepared, _ = outline_draft._inputs(captured)
    session = multi_turn.build_multi_turn_session_seed(prepared)
    prefix = deepcopy(session["accepted_outline_prefix"])
    for _ in range(3):
        decision, _ = asyncio.run(multi_turn_validated_event.handle_validated_candidates(
            session=session, prepared=prepared, planner=SimpleNamespace(product_agent=SimpleNamespace()),
            response={"thought": "Unchanged state.", "candidate_events": [event("same", "machine", None, resource_state="loaded")]}))
        assert decision == "need_revision"
    assert session["status"] == "selection_unresolved"
    assert session["accepted_outline_prefix"] == prefix


def test_explicit_event_params_are_kept_and_only_exact_repeats_are_excluded(captured):
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes.multi_turn_prompts import (
        _accepted_transition_prefix_for_prompt,
    )

    prepared, _ = outline_draft._inputs(captured)
    prepared["recovery_resources"]["arm"]["recovery_des_model"]["events"] = [{"event_name": "move_to_location",
        "parameter_bindings": {name: {"required": True, "type": "number"} for name in ("x", "y", "yaw")}}]
    session = multi_turn.build_multi_turn_session_seed(prepared)
    drive = {**event("drive", "arm", None, resource_state="idle"), "params": {"x": -1.2, "y": 1.15, "yaw": -1.5708}}
    validated = {**session, "outline_diagnostic_bypass": False}
    _, findings = multi_turn._derive_candidate_outline_task(
        candidate_task=drive, session_state=validated, prepared_recovery_request=prepared)
    assert "only when outline validation is bypassed" in findings[0]["reason"]
    task, findings = multi_turn._derive_candidate_outline_task(
        candidate_task=drive, session_state=session, prepared_recovery_request=prepared)
    assert findings == [] and task["params"] == drive["params"]
    assert multi_turn._finite_json_object({"count": 10 ** 400})
    for bad in ({"x": float("nan")}, [1.0, 2.0], {"pose": {"x": float("inf")}}):
        _, findings = multi_turn._derive_candidate_outline_task(
            candidate_task={**drive, "params": bad}, session_state=session, prepared_recovery_request=prepared)
        assert findings and findings[0]["evidence"]["field"] == "params"

    def propose(row):
        return asyncio.run(multi_turn_validated_event.handle_validated_candidates(
            session=session, prepared=prepared, planner=SimpleNamespace(product_agent=SimpleNamespace()),
            response={"thought": "Explore a pose.", "candidate_events": [row]}))[0]

    # Arguments the resource does not declare are not exploration.
    assert propose({**drive, "outline_id": "undeclared", "params": {"attempt": 2}}) == "need_revision"
    # The modeled state is unchanged; the chosen pose is the progress, and the same pose twice is a repeat.
    assert propose(drive) != "need_revision"
    assert propose({**drive, "outline_id": "again"}) == "need_revision"
    assert propose({**drive, "outline_id": "elsewhere", "params": {"x": -1.0, "y": 1.15, "yaw": -1.5708}}) != "need_revision"
    assert [row["params"] for row in session["accepted_outline_prefix"]] == [
        {"x": -1.2, "y": 1.15, "yaw": -1.5708}, {"x": -1.0, "y": 1.15, "yaw": -1.5708}]
    assert _accepted_transition_prefix_for_prompt(session["accepted_outline_prefix"])[0]["params"] == drive["params"]


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
            "llm_input": {"fault_event": {"focused_resource_jid": "arm_b", "blocked_at_function": "place_approach"},
                "observed_runtime_state": {"resources": [row["recovery_snapshot"] for row in resources.values()]},
                "part_facts": [{"part_name": "peg", "part_state": "in_gripper", "current_location": "arm_a",
                                "current_holder_resource_jid": "arm_a"},
                               {"part_name": "gear", "part_state": "misplaced", "current_location": None,
                                "observed_pose": {"x": 0.9, "y": 0.1, "z": 1.0}}]}}}
    path = tmp_path / "observations.json"
    path.write_text(json.dumps(record))
    record.update(source_path=str(path), source_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    return record, [str(spec)]


def _apply(rows, session, prepared):
    for row in rows:
        task, findings = multi_turn._derive_candidate_outline_task(
            candidate_task=row, session_state=session, prepared_recovery_request=prepared)
        assert not findings and not multi_turn_diagnostic.custody_findings(task, session)
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


def test_shared_area_rule_holds_handoff_until_measured_occupant_leaves(shared_cell, tmp_path):
    prepared, session = _draft(shared_cell, tmp_path)
    _apply(_cell_recovery()[1:], session, prepared)
    state = multi_turn_diagnostic.handoff(session, prepared)
    assert state["conditions_met"] is True and state["complete"] is False
    assert state["safety_conflicts"][0]["resource_jids"] == ["arm_a", "arm_b"]
    _apply(_cell_recovery()[:1], session, prepared)
    assert multi_turn_diagnostic.handoff(session, prepared)["complete"] is True


def test_custody_rejects_a_second_part_and_a_part_held_elsewhere(shared_cell, tmp_path):
    prepared, session = _draft(shared_cell, tmp_path)
    for row, rejected in ((event("grab", "arm_a", "gear", held_part="gear", resource_state="picked"), True),
                          (event("grab", "arm_b", "peg", held_part="peg", resource_state="picked"), True),
                          (event("move", "arm_b", "gear", part_state="assembled", part_location="cell",
                                 resource_state="placed"), False)):
        task, _ = multi_turn._derive_candidate_outline_task(
            candidate_task=row, session_state=session, prepared_recovery_request=prepared)
        assert bool(multi_turn_diagnostic.custody_findings(task, session)) is rejected, row


def test_prompt_states_recorded_facts_for_every_robot_without_tuned_targets(shared_cell, tmp_path, monkeypatch):
    record, specs = shared_cell
    provider = AsyncMock(side_effect=[{"thought": "Use recorded facts.", "decision": "grounded", "observations": []},
                                      RuntimeError("stop after the first outline prompt")])
    monkeypatch.setattr(outline_draft.SavedRecoveryProduct, "ask_llm_structured", provider)
    draft = outline_draft.OutlineDraft(record, specs, root=tmp_path / "drafts")
    asyncio.run(draft.generate())
    prompt = provider.call_args_list[1].kwargs.get("prompt") or provider.call_args_list[1].args[0]
    assert "Prefer approximately" not in prompt
    # The handoff obligation comes first; bulky declarations are listed by ref, not inlined.
    assert (prompt.index("Fault Event") < prompt.index("Recovery Goals") < prompt.index("Resource Capabilities")
            < prompt.index("Diagnostic Mode") < prompt.index("Context retrieval"))
    assert prompt.count("Recovery Goals") == 1
    assert '"entity": "gear", "entity_kind": "part", "expected": "assembled", "field": "state", "holds": false' in prompt
    assert "Interrupted event: place_approach of arm_b did not complete." in prompt
    assert "[[-0.25, 0.25]" not in prompt and "max_xy_radius_m" not in prompt
    assert "- /resources/<jid>/gripper_reach: " in prompt and "- /regions/<name>: " in prompt
    assert "Resource validators check" not in prompt
    session = multi_turn.build_multi_turn_session_seed(draft.prepared)
    for jid in ("arm_a", "arm_b"):
        fact = multi_turn_context.resolve_outline_context(draft.prepared, session, f"/resources/{jid}/gripper_reach")
        assert fact["value"]["max_xy_radius_m"] == 0.5 and "tolerance_m" in fact["interpretation"]
    region = multi_turn_context.resolve_outline_context(draft.prepared, session, "/regions/cell")
    assert region["value"]["bounds"] == [[-0.25, 0.25], [-0.25, 0.25], [1.0, 1.4]]


def test_prompt_states_the_handoff_blocker_once_every_condition_holds(shared_cell, tmp_path):
    prepared, session = _draft(shared_cell, tmp_path)
    _apply(_cell_recovery()[1:], session, prepared)
    session["current_phase"] = "outline"
    prompt = multi_turn._build_phase_prompt(prepared, session)[1]
    goals = prompt.split("Recovery Goals\n", 1)[1].split("\n\n", 1)[0]
    assert '"holds": false' not in goals
    assert ("Nominal handoff is blocked in the projected state: SAFE_cell_mutex: arm_a, arm_b would occupy cell"
            in goals)


def test_pose_relations_treat_every_resource_by_its_declarations():
    regions = {"cell": {"bounds": [[-1, 1], [-1, 1], [0, 2]]}, "M1.workholding": {"bounds": [[4, 5], [4, 5], [0, 2]]}}
    envelope = {"origin_pose": {"x": 0.0, "y": 0.0}, "max_xy_radius_m": 0.5, "z_min_m": 0.5, "z_max_m": 1.5,
                "tolerance_m": 0.01}
    locations = {"arm": {"cell"}, "station": {"M1"}, "mobile": {"Storage", "M1"}, "silent": set()}
    station = multi_turn_diagnostic.pose_relations({"x": 4.5, "y": 4.5, "z": 1.0}, regions, locations, {"arm": envelope})
    assert station == {"containing_regions": ["M1.workholding"],
                       "by_declared_location": {"mobile": ["M1"], "station": ["M1"]},
                       "by_declared_envelope": [], "not_evaluated": ["silent"]}
    near = multi_turn_diagnostic.pose_relations({"x": 0.3, "y": 0.0, "z": 1.0}, regions, locations, {"arm": envelope})
    assert near["containing_regions"] == ["cell"] and near["by_declared_location"] == {"arm": ["cell"]}
    assert near["by_declared_envelope"] == ["arm"]
    far = multi_turn_diagnostic.pose_relations({"x": 0.6, "y": 0.0, "z": 1.0}, regions, locations, {"arm": envelope})
    assert far["by_declared_envelope"] == []


def test_diagnostic_refs_serve_declarations_and_derived_pose_relations(shared_cell, tmp_path):
    prepared, session = _draft(shared_cell, tmp_path)
    resolve = multi_turn_context.resolve_outline_context
    gear = resolve(prepared, session, "/parts/gear/pose_relations")
    assert gear["source"] == "model_calculation" and gear["basis"]
    assert gear["value"]["observed_pose"] == {"x": 0.9, "y": 0.1, "z": 1.0}
    assert gear["value"]["containing_regions"] == [] and gear["value"]["by_declared_envelope"] == ["arm_a"]
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
    roster = prompt.split("Resource Capabilities\n", 1)[1].split("\n\n", 1)[0].splitlines()
    row = next(json.loads(line) for line in roster if '"resource_jid": "mobile"' in line)
    assert row["resource_location_domain"] == ["Storage", "cell"]
    assert "field_sources" not in row and "evaluation_support" not in row
    relations = multi_turn_context.resolve_outline_context(prepared, session, "/resources/arm_b/pose_relations")["value"]
    assert relations["by_declared_location"]["mobile"] == ["cell"]


def _candidate_limit(value):
    if isinstance(value, dict):
        if "candidate_events" in value.get("properties", {}):
            return value["properties"]["candidate_events"].get("maxItems")
        return next((limit for item in value.values() if (limit := _candidate_limit(item)) is not None), None)
    return None


@pytest.mark.parametrize("check_cca_safety", [False, True])
def test_llm_chooses_one_event_only_when_no_validator_is_enabled(shared_cell, tmp_path, monkeypatch, check_cca_safety):
    record, specs = shared_cell
    provider = AsyncMock(side_effect=[{"thought": "Use recorded facts.", "decision": "grounded", "observations": []},
                                      RuntimeError("stop after the first outline prompt")])
    monkeypatch.setattr(outline_draft.SavedRecoveryProduct, "ask_llm_structured", provider)
    draft = outline_draft.OutlineDraft(record, specs, root=tmp_path / "drafts", check_cca_safety=check_cca_safety)
    if check_cca_safety:
        monkeypatch.setattr(outline_draft.SavedRecoveryProduct, "ask_llm_structured", provider)
    asyncio.run(draft.generate())
    call = provider.call_args_list[1]
    prompt = call.kwargs.get("prompt") or call.args[0]
    assert _candidate_limit(call.kwargs) == (5 if check_cca_safety else 1)
    assert ("exactly one `candidate_events` item" in prompt) is not check_cca_safety
    assert ("Do not select or rank candidates" in prompt) is check_cca_safety
