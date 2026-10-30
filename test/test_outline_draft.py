from __future__ import annotations

"""Generation-only drafts never enter recovery admission or execution."""

import asyncio
import hashlib
import json
import threading
import time
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from jsonschema import Draft202012Validator

from cais_spade_llm.recovery_framework import outline_draft
from cais_spade_llm.recovery_framework.part_slippage_example import PartSlippageExample
from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes import multi_turn
from cais_spade_llm.agents.shared_information.llm_request_records import RequestRecords


GROUNDED = {"thought": "Use the retained facts.", "decision": "grounded", "observations": []}
# Only the LLM ends a draft, by declaring the recovery complete.
DONE = {"thought": "The failure is overcome.", "decision": "recovery_complete"}


def scripted_provider(monkeypatch, responses):
    """Replace provider transport while retaining framework request recording."""
    provider = AsyncMock(side_effect=[GROUNDED, *responses])
    install_provider(monkeypatch, provider)
    return provider


def install_provider(monkeypatch, provider):
    """Record supplied replies at the same boundary as real provider replies."""
    async def recorded(self, prompt, **kwargs):
        self._last_structured_request = {}
        records = RequestRecords(self._last_structured_request)
        try:
            result = await provider(prompt, **kwargs)
        except json.JSONDecodeError:
            records.submit(lambda **payload: "?", {"prompt": prompt, **kwargs}, tool_round=0, attempt=1)
            raise
        records.submit(lambda **payload: result, {"prompt": prompt, **kwargs}, tool_round=0, attempt=1)
        return result
    monkeypatch.setattr(outline_draft.SavedRecoveryProduct, "ask_llm_structured", recorded)


@pytest.fixture
def observations(tmp_path):
    entry = {"resource_jid": "owner", "resource_type": "resource",
        "static_capabilities": {"reachability": ["M2"], "gripper_reach": 99,
            "capabilities": ["place_release"], "evaluation_support": {"additional_bindings": ["private_function"]}},
        "recovery_snapshot": {"resource_jid": "owner", "resource_state": "idle", "resource_location": "M2"},
        "recovery_des_model": {"state_variables": {
            "resource_state": {"type": "string", "scope": "resource"},
            "resource_location": {"type": "string", "scope": "resource"},
            "active_job": {"type": ["string", "null"], "scope": "resource"},
            "part_name": {"type": ["string", "null"], "scope": "resource"},
            "part_location": {"type": "string", "scope": "part"},
            "private_program": {"type": "string", "scope": "resource", "private": True}},
            "events": [
                {"event_name": "machine_part", "parameter_bindings": {"part_name": {"equals": "KET4_Square_4mm"}},
                 "updates": {"resource_state": {"set": "completed"}}},
                {"event_name": "resume_task", "guards": {"resource_state": {"equals": "completed"}}}]}}
    record = {"scenario": "Machining breakdown during part processing", "run_id": "capture",
        "jids": {"M2": "owner"},
        "pending_nodes": [
            {"id": "replaced", "function_name": "machine_part", "params": {"resource_id": "M2", "part_name": "KET4_Square_4mm"}},
            {"id": "resume", "function_name": "resume_task", "params": {"resource_id": "M2"}}],
        "captured_at": "recorded-time", "offline_ready": False, "missing": ["physical_evidence"],
        "prepared_recovery_request": {"ra_jid": "owner", "recovery_resources": {"owner": entry},
            "marked_reentry_context": {"pending_suffixes": [
                {"role": "recovery_replaced", "resource_jid": "owner", "pending_task_ids": ["replaced"]},
                {"role": "resume_suffix", "resource_jid": "owner", "entry_task_id": "resume"}]},
            "multi_turn_session_state": {"accepted_outline_prefix": [{"event_name": "DO_NOT_COPY"}]},
            "llm_input": {"fault_event": {"focused_resource_jid": "owner", "affected_part_names": ["KET4_Square_4mm"]},
                "observed_runtime_state": {"resources": [entry["recovery_snapshot"]]},
                "part_facts": [{"part_name": "KET4_Square_4mm", "current_location": "M2"}]}}}
    source = tmp_path / "observations.json"
    source.write_text(json.dumps(record))
    record.update(source_path=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    return record


def response():
    # Drafts propose two to five ways to do the next step, most preferred first.
    return {"thought": "A supplied test draft, not a generated recovery.",
            "candidate_events": [{
                "outline_id": "draft_step", "event_name": "Arbitrary task event", "resource": "M2",
                "part_name": "KET4_Square_4mm", "expected_end_state": {"resource_state": "completed"},
                "rationale": "Assumption remains unvalidated."}, {
                "outline_id": "draft_alternative", "event_name": "Another task event", "resource": "M2",
                "part_name": "KET4_Square_4mm", "expected_end_state": {"resource_state": "idle"},
                "rationale": "An alternative the draft does not choose."}]}


def test_draft_retrieval_generation_and_reload_never_validate(observations, tmp_path, monkeypatch):
    from cais_spade_llm.recovery_framework import primitive_composition

    forbidden = Mock(side_effect=AssertionError("No admission, projection, or composition in draft generation"))
    monkeypatch.setattr(primitive_composition, "start_composition_attempt", forbidden)
    original = deepcopy(observations)
    provider = scripted_provider(monkeypatch, [{"thought": "Need facts", "decision": "need_context",
        "context_requests": ["/resources/owner/snapshot", "/safety_rules", "/parts/KET4_Square_4mm/observed_pose"]}, response(), DONE])
    draft = outline_draft.OutlineDraft(observations, [], root=tmp_path / "drafts")
    asyncio.run(draft.generate())
    assert draft.record["status"] == "draft"
    assert draft.record["transition_trace"][0]["event_name"] == response()["candidate_events"][0]["event_name"]
    assert draft.record["provider_calls"] == 4
    assert "/resources/owner/snapshot" in draft.record["retrievals"]
    assert "checks" not in draft.record and "validation_checks" not in draft.record
    assert "accepted_outline_prefix" not in draft.record
    assert draft.record["pipeline"] == "execute_multi_turn_recovery"
    assert draft.snapshot()["outline"] == [] and draft.snapshot()["execution_authorized"] is False
    assert observations == original
    assert outline_draft.load_draft(draft.directory / "draft.json", observations, root=tmp_path / "drafts") == draft.record
    forbidden.assert_not_called()
    for call in provider.call_args_list:
        assert call.kwargs["max_attempts"] == 1
        assert call.kwargs["include_agent_instructions"] is False
        assert not any(token in call.args[0] if call.args else token in call.kwargs["prompt"] for token in (
            "place_release", "gripper_reach", "private_function", "private_program", "DO_NOT_COPY"))


def test_approached_part_is_kept_from_the_llm(observations, tmp_path):
    # Only the KMR's nominal Storage pick sets approached_part; no recovery primitive does (user decision, 2026-10-30).
    entry = observations["prepared_recovery_request"]["recovery_resources"]["owner"]
    entry["recovery_des_model"]["state_variables"]["approached_part"] = {"type": ["string", "null"], "scope": "resource"}
    entry["recovery_snapshot"]["approached_part"] = "KET4_Square_4mm"
    draft = outline_draft.OutlineDraft(observations, [], root=tmp_path)
    assert "approached_part" not in draft.prepared["recovery_resources"]["owner"]["recovery_des_model"]["state_variables"]
    session = multi_turn.build_multi_turn_session_seed(draft.prepared)
    session["current_phase"] = "outline"
    _, prompt = multi_turn._build_phase_prompt(draft.prepared, session)
    assert "approached_part" not in prompt


def test_draft_equal_reachability_context_and_schema_expressiveness(observations, tmp_path):
    prepared = observations["prepared_recovery_request"]
    prepared["recovery_resources"]["second"] = deepcopy(prepared["recovery_resources"]["owner"])
    prepared["recovery_resources"]["second"]["recovery_snapshot"]["resource_jid"] = "second"
    draft = outline_draft.OutlineDraft(observations, [], root=tmp_path)
    session = multi_turn.build_multi_turn_session_seed(draft.prepared)
    session["current_phase"] = "outline"
    payload, prompt = multi_turn._build_phase_prompt(draft.prepared, session)
    # Both resources are stated by name with the same declared location; neither is preferred.
    assert "\nM2 (resource)  idle; resource_location M2; capabilities: reachable locations M2\n" in prompt
    assert "\nsecond (resource)  state not recorded; capabilities: reachable locations M2\n" in prompt
    assert session.get("outline_requested_refs", []) == []
    schema = multi_turn._get_response_schema("outline", session)
    assert not list(Draft202012Validator(schema["schema"]).iter_errors(response()))
    assert schema["schema"]["$defs"]["outline_event"]["properties"]["expected_end_state"]["type"] == "object"
    assert "transition_trace" not in schema["schema"]["properties"]


def test_draft_states_the_position_a_place_declares_for_a_resource_placing_there(observations, tmp_path):
    prepared = observations["prepared_recovery_request"]
    prepared["recovery_resources"]["owner"]["static_capabilities"]["reachability"] = ["M2", "Belt"]
    belt = deepcopy(prepared["recovery_resources"]["owner"])
    belt.update(resource_jid="belt", static_capabilities={}, resource_evidence={"resource_id": "Belt",
        "environment_models": {"Belt": {"events": [{"event_name": "place_release",
            "updates": {"part_location.{part_name}": {"set": "loading_position_1"}},
            "parameter_bindings": {"resource_id": {"equals": "M2"}, "destination_location": {"equals": "Belt"}}}]}}})
    belt["recovery_snapshot"]["resource_jid"] = "belt"
    prepared["recovery_resources"]["belt"] = belt
    observations["jids"]["Belt"] = "belt"
    draft = outline_draft.OutlineDraft(observations, [], root=tmp_path)
    session = multi_turn.build_multi_turn_session_seed(draft.prepared)
    session["current_phase"] = "outline"
    _, prompt = multi_turn._build_phase_prompt(draft.prepared, session)
    # The draft keeps the place's declared events, so the robot line states where its placement puts a part.
    assert ("\nM2 (resource)  idle; resource_location M2; capabilities: reachable locations M2, Belt; "
            "places parts on Belt at loading_position_1\n") in prompt
    assert "place_release" not in prompt


def test_draft_states_a_resource_s_declared_job_control(observations, tmp_path):
    prepared = observations["prepared_recovery_request"]
    prepared["recovery_resources"]["owner"]["static_capabilities"]["job_control"] = {
        "version": 1, "resume_after_reload": True, "restart_stopped_jobs": False}
    draft = outline_draft.OutlineDraft(observations, [], root=tmp_path)
    session = multi_turn.build_multi_turn_session_seed(draft.prepared)
    session["current_phase"] = "outline"
    _, prompt = multi_turn._build_phase_prompt(draft.prepared, session)
    # The draft keeps the declared job control, so the LLM can tell a resumable pause from a final stop.
    assert ("\nM2 (resource)  idle; resource_location M2; capabilities: reachable locations M2; "
            "job control: a paused job continues from where it was paused, also after its part is unloaded "
            "and loaded back in; a stopped job cannot be restarted\n") in prompt


@pytest.mark.parametrize("responses, expected_calls, status", [
    ([response(), DONE], 3, "draft"),
    # Nothing checks a reply; one code cannot read ends the draft with its error.
    ([{}, response(), DONE], 2, "draft_incomplete"),
    # A repeated lookup is answered, so only the turn budget ends the draft.
    ([{"thought": "Need facts", "decision": "need_context", "context_requests": ["/resources"]}] * 8, 6, "draft_incomplete"),
    ([json.JSONDecodeError("bad JSON", "?", 0), response()], 2, "draft_incomplete"),
])
def test_draft_budget_and_formatting_failures(observations, tmp_path, monkeypatch, responses, expected_calls, status):
    provider = scripted_provider(monkeypatch, responses)
    draft = outline_draft.OutlineDraft(observations, [], root=tmp_path)
    assert draft.prepared["recovery_session"]["max_turns"] == outline_draft.DRAFT_MAX_TURNS == 40
    draft.prepared["recovery_session"]["max_turns"] = 6
    asyncio.run(draft.generate())
    assert provider.await_count == expected_calls
    assert draft.record["status"] == status
    assert len(list(draft.directory.glob("recovery_outline/requests/*/*_request.json"))) == expected_calls


def test_draft_stop_retains_response_without_committing(observations, tmp_path, monkeypatch):
    async def run():
        entered, release = asyncio.Event(), asyncio.Event()
        async def provider(prompt, **kwargs):
            entered.set()
            await release.wait()
            return response()
        transport = AsyncMock(side_effect=provider)
        install_provider(monkeypatch, transport)
        draft = outline_draft.OutlineDraft(observations, [], root=tmp_path)
        task = asyncio.create_task(draft.generate())
        await entered.wait()
        draft.stop()
        release.set()
        await task
        assert draft.record["status"] == "stopped"
        assert draft.record["transition_trace"] == []
        assert list(draft.directory.glob("recovery_outline/requests/*/*_response.json"))
        assert transport.await_count == 1
    asyncio.run(run())


def test_draft_reload_rejects_changed_source_and_accepted_labels(observations, tmp_path, monkeypatch):
    scripted_provider(monkeypatch, [response(), DONE])
    draft = outline_draft.OutlineDraft(observations, [], root=tmp_path)
    asyncio.run(draft.generate())
    path = draft.directory / "draft.json"
    record = json.loads(path.read_text())
    # A draft saved with an outline check enabled is not an unvalidated draft.
    record["validation_checks"] = {"check_cca_safety": True}
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="unvalidated"):
        outline_draft.load_draft(path, observations, root=tmp_path)
    draft._save()
    from pathlib import Path
    Path(observations["source_path"]).write_text("changed")
    with pytest.raises(ValueError, match="source observations changed"):
        outline_draft.load_draft(path, observations, root=tmp_path)


def test_example_draft_actions_work_without_offline_worker(observations, tmp_path, monkeypatch):
    from cais_spade_llm.recovery_framework import part_slippage_example
    actual = outline_draft.OutlineDraft
    provider = scripted_provider(monkeypatch, [response(), DONE])
    def factory(*args, **kwargs):
        draft = actual(*args, **kwargs, root=tmp_path / "drafts")
        return draft
    monkeypatch.setattr(outline_draft, "OutlineDraft", factory)
    original_latest, original_load = outline_draft.latest_draft, outline_draft.load_draft
    monkeypatch.setattr(outline_draft, "latest_draft", lambda obs: original_latest(obs, root=tmp_path / "drafts"))
    monkeypatch.setattr(outline_draft, "load_draft", lambda path, obs, **kwargs: original_load(path, obs, root=tmp_path / "drafts"))
    monkeypatch.setattr(part_slippage_example, "read_observations", lambda path: deepcopy(observations))
    example = PartSlippageExample(SimpleNamespace(system_running=False))
    example.source, example.scenario = "offline", observations["scenario"]
    example.observation_path = observations["source_path"]
    example.safety_files = []
    async def run():
        await example.action("generate_outline_only")
        await example._draft._task
        assert example.snapshot()["outline_draft"]["status"] == "draft"
        saved = deepcopy(example.snapshot()["outline_draft"])
        for action in ("continue_outline", "generate_primitives", "execute", "test_in_gazebo"):
            with pytest.raises(ValueError, match="review only"):
                await example.action(action)
        await example.action("load_draft")
        assert provider.await_count == 3
        assert example.snapshot()["outline_draft"] == saved
        assert example.snapshot()["can_load_draft"] is True
    asyncio.run(run())


def test_start_fresh_keeps_empty_prefix_and_locks_generated_choices(observations, tmp_path, monkeypatch):
    from cais_spade_llm.recovery_framework import part_slippage_example

    actual = outline_draft.OutlineDraft
    monkeypatch.setattr(outline_draft, "OutlineDraft", lambda *args, **kwargs: actual(*args, **kwargs, root=tmp_path / "drafts"))
    monkeypatch.setattr(part_slippage_example, "read_observations", lambda path: deepcopy(observations))
    provider = scripted_provider(monkeypatch, [response(), DONE])
    example = PartSlippageExample(SimpleNamespace(system_running=False))
    example.set_options(source="offline", scenario=observations["scenario"], safety_files=[],
        observation_path=observations["source_path"])
    first = example.prepare_offline()
    assert first["status"] == "draft_ready" and first["outline_draft"]["transition_trace"] == []
    assert first["can_generate_outline"] is True
    assert json.loads((example._draft.directory / "draft.json").read_text()) == example._draft.record
    provider.assert_not_called()

    async def run():
        fresh = await example.action("start_fresh")
        assert fresh["run_id"] != first["run_id"]
        assert fresh["outline"] == fresh["primitives"] == fresh["outline_draft"]["transition_trace"] == []
        assert json.loads((example._draft.directory / "draft.json").read_text())["provider_calls"] == 0
        await example.action("generate_outline")
        assert example.snapshot()["busy"] is True
        await example._draft._task
        # The generated draft keeps the safety selections it was generated with.
        with pytest.raises(ValueError, match="Reset example"):
            example.set_options(source="offline", scenario=observations["scenario"], safety_files=[],
                observation_path=observations["source_path"])
        state = await example.action("start_fresh")
        assert state["can_generate_outline"] is True
        assert state["outline_draft"]["provider_calls"] == 0
        assert provider.await_count == 3

    asyncio.run(run())


def test_deferred_preferences_do_not_load_prepare_or_write(observations, tmp_path, monkeypatch):
    from cais_spade_llm.recovery_framework import part_slippage_example
    from cais_spade_llm.recovery_framework.recovery_case_safety import default_safety_files

    forbidden = Mock(side_effect=AssertionError("Preferences must stay lightweight"))
    monkeypatch.setattr(part_slippage_example, "read_observations", forbidden)
    monkeypatch.setattr(part_slippage_example, "observation_summary", forbidden)
    monkeypatch.setattr(outline_draft, "OutlineDraft", forbidden)
    example = PartSlippageExample(SimpleNamespace(system_running=False))
    defaults = default_safety_files(observations["scenario"])
    started = time.monotonic()
    for index in range(30):
        state = example.set_options(source="offline", scenario=observations["scenario"],
            safety_files=[] if index % 2 else defaults, observation_path=observations["source_path"],
            defer_preparation=True)
        assert state["can_generate_outline"] and not state["busy"]
        assert not state.get("run_id")
    assert time.monotonic() - started < 1
    forbidden.assert_not_called()
    assert list(tmp_path.iterdir()) == [tmp_path / "observations.json"]


def test_generate_outline_fresh_freezes_choices_and_keeps_ui_responsive(observations, tmp_path, monkeypatch):
    from cais_spade_llm.recovery_framework import part_slippage_example

    actual = outline_draft.OutlineDraft
    monkeypatch.setattr(outline_draft, "OutlineDraft", lambda *args, **kwargs: actual(*args, **kwargs, root=tmp_path / "drafts"))
    entered, release = threading.Event(), threading.Event()
    threads = []

    def read(path):
        threads.append(threading.get_ident())
        entered.set()
        assert release.wait(3)
        return deepcopy(observations)

    monkeypatch.setattr(part_slippage_example, "read_observations", read)
    provider = scripted_provider(monkeypatch, [response(), DONE, GROUNDED, response(), DONE])
    example = PartSlippageExample(SimpleNamespace(system_running=False))
    options = dict(source="offline", scenario=observations["scenario"], safety_files=[],
        observation_path=observations["source_path"], defer_preparation=True)
    example.set_options(**options)
    options["safety_files"] = list(example.safety_files)

    async def run():
        state = await example.action("generate_outline_fresh")
        assert state["busy"]
        for _ in range(100):
            if entered.is_set():
                break
            await asyncio.sleep(.005)
        assert entered.is_set()
        provider.assert_not_called()
        with pytest.raises(ValueError):
            example.set_options(**{**options, "safety_files": []})
        with pytest.raises(ValueError):
            await example.action("generate_outline_fresh")
        release.set()
        await example._task
        first = example.snapshot()
        assert first["outline_draft"]["status"] == "draft"
        assert not first["busy"]
        record = deepcopy(first["outline_draft"])
        example.set_options(**options)
        assert example.snapshot()["outline_draft"] == record
        await example.action("generate_outline_fresh")
        await example._task
        second = example.snapshot()
        assert second["run_id"] != first["run_id"]
        assert len(second["outline_draft"]["transition_trace"]) == 1
        assert len(threads) == 2 and all(t != threading.get_ident() for t in threads)
        assert provider.await_count == 6

    asyncio.run(run())


def test_stage_stop_on_a_fresh_outline_stops_before_any_provider_call(observations, tmp_path, monkeypatch):
    from cais_spade_llm.recovery_framework import part_slippage_example

    actual = outline_draft.OutlineDraft
    monkeypatch.setattr(outline_draft, "OutlineDraft", lambda *args, **kwargs: actual(*args, **kwargs, root=tmp_path / "drafts"))
    entered, release = threading.Event(), threading.Event()

    def read(path):
        entered.set()
        assert release.wait(3)
        return deepcopy(observations)

    monkeypatch.setattr(part_slippage_example, "read_observations", read)
    provider = scripted_provider(monkeypatch, [response()])
    example = PartSlippageExample(SimpleNamespace(system_running=False))
    example.set_options(source="offline", scenario=observations["scenario"], safety_files=[],
        observation_path=observations["source_path"], defer_preparation=True)

    async def run():
        await example.action("generate_outline_fresh")
        try:
            assert await asyncio.to_thread(entered.wait, 3)
            assert example.snapshot()["generating_stage"] == "outline"
            await example.action("stop_generation")
            assert example.snapshot()["generating_stage"] == ""
        finally:
            release.set()
        await asyncio.gather(example._task, return_exceptions=True)
        assert example.snapshot()["status"] == "stopped"
        assert example.snapshot()["can_generate_outline"]

    asyncio.run(run())
    provider.assert_not_called()
    assert not (tmp_path / "drafts").exists()


def test_scene_geometry_describes_layout_without_routes_docks_or_assignments():
    scene = {
        "robots": [{"resource_id": "arm_a", "base_xyz": [0.0, 0.5, 1.0], "prefix": "a_"}],
        "machines": [{"resource_id": "M1", "world_pose": [-4.0, 1.9, 0, 0, 0, 0], "enclosure_dimensions_m": [0.5, 0.5, 0.5],
                      "stand_height_m": 1.0, "KMR_docking_pose": [-4.7, 2.2, 0, 0, 0, 0], "handling_robot": "arm_a",
                      "staging_pose": [-4.6, 1.1, 1.0, 0, 0, 0], "conveyor_loading_pose": [-4.0, 0.5, 1.0, 0, 0, 0]}],
        "Conveyor": {"world_pose": [-2.0, 0.5, 0, 0, 0, 0], "length": 4.0, "width": 0.3, "surface_height": 1.0,
                     "normal_output_transfer": {"to": "Buffer"}, "output_handling_robot": "arm_a"},
        "Storage": {"world_pose": [-6.0, 2.0, 0, 0, 0, 0], "size": [1.0, 1.0, 0.7], "KMR_docking_pose": [-5.5, 2.0, 0, 0, 0, 0],
                    "KMR_pick_docking_poses": {"peg": [-5.5, 2.1, 0, 0, 0, 0]}},
        "Mobile": {"initial_pose": [0, 0, 0, 0, 0, 0], "platform_overall_dimensions_m": [1.2, 0.7, 0.7],
                   "arm_mount_xyz": [-0.25, 0.0, 0.7], "arm_reach_m": 0.8, "predefined_routes": [{"resources": ["Storage", "M1"]}],
                   "predefined_route_waypoints": {}},
    }
    observations = {"inputs": {"scene": scene},
                    "physical_evidence": {"resources": {"Mobile": {"base_pose": [-5.5, 2.0, 0.0, 0.0, 0.0, 0.0, 1.0]}}}}
    geometry = outline_draft.scene_geometry(observations)
    assert geometry["stations"]["M1"] == {"world_pose": [-4.0, 1.9, 0, 0, 0, 0], "enclosure_dimensions_m": [0.5, 0.5, 0.5],
                                          "stand_height_m": 1.0}
    assert geometry["stations"]["Conveyor"] == {"world_pose": [-2.0, 0.5, 0, 0, 0, 0], "length": 4.0, "width": 0.3,
                                                "surface_height": 1.0}
    assert geometry["robot_bases"] == {"arm_a": [0.0, 0.5, 1.0]}
    assert geometry["mobile_bases"]["Mobile"] == {"platform_overall_dimensions_m": [1.2, 0.7, 0.7],
        "arm_mount_xyz": [-0.25, 0.0, 0.7], "arm_reach_m": 0.8, "base_pose_xy_yaw": [-5.5, 2.0, 0.0]}
    assert "Conveyor" in geometry["floor_obstacles_xy"] and len(geometry["floor_extent_xy"]) == 4
    text = json.dumps(geometry)
    for hint in ("docking", "handling_robot", "route", "staging_pose", "conveyor_loading_pose", "normal_output_transfer",
                 "output_handling_robot"):
        assert hint not in text


def test_ra_feasibility_toggle_reaches_the_next_offline_draft(observations, monkeypatch):
    from cais_spade_llm.recovery_framework import part_slippage_example

    # The fixture records no physical evidence for the RA's cell, so the draft is recorded, not built.
    created = Mock(side_effect=ValueError("draft recorded"))
    monkeypatch.setattr(outline_draft, "OutlineDraft", created)
    monkeypatch.setattr(part_slippage_example, "read_observations", lambda path: deepcopy(observations))
    example = PartSlippageExample(SimpleNamespace(system_running=False))
    example.set_options(source="offline", scenario=observations["scenario"], safety_files=[],
        observation_path=observations["source_path"], defer_preparation=True)
    assert example.snapshot()["check_ra_feasibility"] is False
    with pytest.raises(ValueError, match="on or off"):
        example.set_ra_feasibility_check("on")
    assert example.set_ra_feasibility_check(True)["check_ra_feasibility"] is True

    async def run():
        await example.action("generate_outline_fresh")
        await example._task

    asyncio.run(run())
    assert created.call_args.kwargs["check_ra_feasibility"] is True
    assert example.snapshot()["status"] == "blocked"


def test_one_cca_toggle_needs_the_ra_check_and_runs_both_cca_checks(observations, monkeypatch):
    from cais_spade_llm.recovery_framework import part_slippage_example

    created = Mock(side_effect=ValueError("draft recorded"))
    monkeypatch.setattr(outline_draft, "OutlineDraft", created)
    monkeypatch.setattr(part_slippage_example, "read_observations", lambda path: deepcopy(observations))
    example = PartSlippageExample(SimpleNamespace(system_running=False))
    example.set_options(source="offline", scenario=observations["scenario"], safety_files=[],
        observation_path=observations["source_path"], defer_preparation=True)
    assert example.snapshot()["check_cca"] is False
    with pytest.raises(ValueError, match="RA feasibility check first"):
        example.set_cca_check(True)
    example.set_ra_feasibility_check(True)
    assert example.set_cca_check(True)["check_cca"] is True
    # The CCA safety check places robots from the RA's reach results, so it cannot stay on alone.
    assert example.set_ra_feasibility_check(False)["check_cca"] is False
    example.set_ra_feasibility_check(True)
    example.set_cca_check(True)

    async def run():
        await example.action("generate_outline_fresh")
        await example._task

    asyncio.run(run())
    checks = ("check_ra_feasibility", "check_nonblocking", "check_cca_safety")
    assert {key: created.call_args.kwargs[key] for key in checks} == dict.fromkeys(checks, True)
    assert created.call_args.kwargs["cca_assumptions"] == []


def test_safety_rules_lookup_toggle_offers_the_rules_only_as_a_lookup(observations, monkeypatch):
    from cais_spade_llm.recovery_framework import part_slippage_example

    created = Mock(side_effect=ValueError("draft recorded"))
    monkeypatch.setattr(outline_draft, "OutlineDraft", created)
    monkeypatch.setattr(part_slippage_example, "read_observations", lambda path: deepcopy(observations))
    example = PartSlippageExample(SimpleNamespace(system_running=False))
    example.set_options(source="offline", scenario=observations["scenario"], safety_files=[],
        observation_path=observations["source_path"], defer_preparation=True)
    assert example.snapshot()["safety_rules_lookup"] is False
    with pytest.raises(ValueError, match="on or off"):
        example.set_safety_rules_lookup("on")

    async def run():
        await example.action("generate_outline_fresh")
        await example._task

    for enabled, setting in ((False, "hidden"), (True, "knowledge_base")):
        assert example.set_safety_rules_lookup(enabled)["safety_rules_lookup"] is enabled
        asyncio.run(run())
        assert created.call_args.kwargs["safety_rules_to_llm"] == setting


def test_claude_token_usage_reads_like_openai_usage():
    usage = {"input_tokens": 100, "output_tokens": 40, "cache_read_input_tokens": 300, "cache_creation_input_tokens": 20}
    mapped = outline_draft._token_usage(usage)
    assert {key: mapped[key] for key in ("prompt_tokens", "completion_tokens", "total_tokens")} == {
        "prompt_tokens": 420, "completion_tokens": 40, "total_tokens": 460}
    assert mapped["prompt_tokens_details"] == {"cached_tokens": 300} and mapped["input_tokens"] == 100
    openai = {"prompt_tokens": 5, "completion_tokens": 1}
    assert outline_draft._token_usage(openai) is openai


LOOKUP = {"thought": "Need facts", "decision": "need_context", "context_requests": ["/resources/owner/snapshot"]}


def single_pass_reply(*steps):
    # A single-pass reply is the whole outline: every event, in order, and nothing to choose.
    return {"thought": "A supplied test outline, not a generated recovery.", "events": [
        {"outline_id": f"R{index}", "event_name": name, "resource": "M2", "part_name": "KET4_Square_4mm",
         "expected_end_state": {"resource_state": state}, "rationale": "Supplied test step."}
        for index, (name, state) in enumerate(steps, 1)]}


def _stand_in_checks(monkeypatch, refused=(), unsafe=(), blocking=0, seen=None):
    """Replace the RA, CCA and nonblocking checks with stand-ins keyed by event name.

    ``seen`` collects, for each RA call, the event and the applied RA records the check could read.
    """
    from cais_spade_llm.recovery_framework import cca_nonblocking, cca_safety, ra_feasibility

    monkeypatch.setattr(ra_feasibility, "recorded_cell", lambda observations, floor: object())
    monkeypatch.setattr(cca_safety, "capture_inputs", lambda observations, files, assumptions=None: {})
    monkeypatch.setattr(cca_nonblocking, "capture_inputs", lambda observations: {})
    verdicts = iter([cca_nonblocking.BLOCKING] * blocking)

    def nonblocking(session, prepared, capture):
        status = next(verdicts, cca_nonblocking.NONBLOCKING)
        return {"nonblocking_status": status, "reasons": [] if status == cca_nonblocking.NONBLOCKING
                else ["normal production cannot finish KET4"], "blocked": [], "evidence": {}}

    def ra(event, session, prepared, cell):
        if seen is not None:
            seen.append((event["event_name"], [row["results"][0]["event_name"] for row in session.get("ra_feasibility_checks", [])
                                               if row["applied_candidate_index"] is not None]))
        status = ra_feasibility.INFEASIBLE if event["event_name"] in refused else ra_feasibility.FEASIBLE
        return {"feasibility_status": status, "resource_jid": event["resource_jid"], "event_name": event["event_name"],
                "conditions": [], "capability": [], "primitive_set": [],
                "reasons": ["the RA fact"] if status != ra_feasibility.FEASIBLE else []}

    def cca(event, session, prepared, capture, feasibility):
        if event["event_name"] in unsafe:
            return {"resource_jid": event["resource_jid"], "event_name": event["event_name"], "steps": [],
                    "safety_status": cca_safety.UNSAFE, "reasons": ["the CCA fact"]}
        return {"resource_jid": event["resource_jid"], "event_name": event["event_name"], "steps": [],
                "safety_status": cca_safety.SAFE, "reasons": [], "state_after": {"monitors": {}, "robots": {}}}

    monitor = Mock(side_effect=nonblocking)
    monkeypatch.setattr(ra_feasibility, "check_event", ra)
    monkeypatch.setattr(cca_safety, "check_event", cca)
    monkeypatch.setattr(cca_nonblocking, "check_completion", monitor)
    return monitor


def _outline_prompts(provider):
    return [call.args[0] for call in provider.call_args_list[1:]]


def test_single_pass_draft_applies_the_whole_reply_after_lookups(observations, tmp_path, monkeypatch):
    reply = single_pass_reply(("finish", "completed"), ("release", "idle"))
    provider = scripted_provider(monkeypatch, [LOOKUP, reply])
    draft = outline_draft.OutlineDraft(observations, [], root=tmp_path / "drafts", single_pass_outline=True)
    asyncio.run(draft.generate())
    assert draft.record["status"] == "draft", draft.record["message"]
    # Grounding, one lookup and the outline itself: no completion turn follows the outline.
    assert provider.await_count == draft.record["provider_calls"] == 3
    assert [row["event_name"] for row in draft.record["transition_trace"]] == ["finish", "release"]
    assert [row["outline_id"] for row in draft.record["transition_trace"]] == ["RECOVERY_SEQ1", "RECOVERY_SEQ2"]
    assert {row["resource_jid"] for row in draft.record["transition_trace"]} == {"owner"}
    assert draft.record["nominal_handoff"]["declared_complete_by_llm"] is True
    assert "/resources/owner/snapshot" in draft.record["retrievals"]
    assert draft.record["single_pass_outline"] is True and draft.record["message"].startswith("Unvalidated single-pass")
    assert draft.record["stop_cause"]["cause"] == "declared_complete"
    assert draft.record["turn_decisions"] == {"need_context": 1, "outline_ready": 1}
    assert draft.record["single_pass_checks"][0]["accepted"] is True
    schema = provider.call_args_list[2].kwargs["response_format"]["schema"]
    assert not list(Draft202012Validator(schema).iter_errors(reply))
    assert "candidate_events" not in schema["properties"] and "selected_candidate_index" not in schema["properties"]
    assert schema["properties"]["decision"]["enum"] == [None, "need_context"]
    event = schema["$defs"]["outline_event"]["properties"]
    assert list(event)[list(event).index("expected_end_state") + 1] == "neighbor_end_state"
    prompt = _outline_prompts(provider)[-1]
    assert "Give the whole plan in `events`: every recovery event, in the order they happen." in prompt
    assert "\nEvent Rules\n" in prompt and "Candidate Rules" not in prompt and "Propose 2–5" not in prompt


def test_single_pass_prompt_differs_from_the_incremental_one_only_in_its_reply_format(observations, tmp_path):
    prompts = []
    for single_pass in (False, True):
        draft = outline_draft.OutlineDraft(deepcopy(observations), [], root=tmp_path, single_pass_outline=single_pass)
        session = multi_turn.build_multi_turn_session_seed(draft.prepared)
        session["current_phase"] = "outline"
        prompts.append(multi_turn._build_phase_prompt(draft.prepared, session)[1])
    incremental, single_pass = prompts
    for incremental_text, single_pass_text in (
        ("Propose 2–5 different ways to do the next step of that plan, most preferred first.",
         "Give the whole plan in `events`: every recovery event, in the order they happen."),
        ("\nCandidate Rules\n", "\nEvent Rules\n"),
        ("- One event per candidate, by one resource: ", "- Each event, by one resource: "),
        ("- When you judge that normal production can carry on its remaining work from the state your events leave, "
         "return only `thought` and `decision: \"recovery_complete\"`; normal production then takes over.",
         "- End `events` where you judge that normal production can carry on its remaining work from the state your "
         "events leave; normal production then takes over."),
    ):
        assert incremental.count(incremental_text) == 1 and single_pass.count(single_pass_text) == 1
        single_pass = single_pass.replace(single_pass_text, incremental_text)
    assert single_pass == incremental


def test_single_pass_with_checks_rewrites_a_refused_outline_from_the_failure_state(observations, tmp_path, monkeypatch):
    seen = []
    monitor = _stand_in_checks(monkeypatch, refused={"release"}, seen=seen)
    provider = scripted_provider(monkeypatch, [single_pass_reply(("finish", "completed"), ("release", "idle")),
                                               single_pass_reply(("finish", "completed"), ("unload", "idle"))])
    draft = outline_draft.OutlineDraft(observations, [], root=tmp_path / "drafts", single_pass_outline=True,
        check_ra_feasibility=True, check_cca_safety=True, check_nonblocking=True)
    asyncio.run(draft.generate())
    assert draft.record["status"] == "draft", draft.record["message"]
    assert [row["event_name"] for row in draft.record["transition_trace"]] == ["finish", "unload"]
    # Each event's RA record is written before the next event is checked (the RA reads rest poses from it),
    # and the rewrite starts without the refused outline's records.
    assert seen == [("finish", []), ("release", ["finish"]), ("finish", []), ("unload", ["finish"])]
    # The nonblocking check judges only the outline whose events all passed.
    assert monitor.call_count == 1
    rewrite = _outline_prompts(provider)[1]
    assert "\nYour Last Outline (refused)\n" in rewrite and "It was refused at event 2 of 2 (`R2`)" in rewrite
    assert '"event_name":"release"' in rewrite.replace(" ", "") and "reason=the RA fact" in rewrite
    # Nothing from the refused outline stays in the planning state the rewrite starts from.
    assert "Recovery Events So Far" not in rewrite and "Recorded at the failure." in rewrite
    assert [row["refused_event_index"] for row in draft.record["single_pass_checks"]] == [1, None]
    assert [row["accepted"] for row in draft.record["single_pass_checks"]] == [False, True]
    assert [row["applied_candidate_index"] for row in draft.record["ra_feasibility_checks"]] == [0, 0]
    assert len(draft.record["cca_safety_checks"]) == 2 and len(draft.record["cca_nonblocking_checks"]) == 1
    assert draft.record["turn_decisions"] == {"need_revision": 1, "outline_ready": 1}


def test_single_pass_nonblocking_refusal_names_the_end_of_the_outline(observations, tmp_path, monkeypatch):
    _stand_in_checks(monkeypatch, blocking=1)
    reply = single_pass_reply(("finish", "completed"))
    provider = scripted_provider(monkeypatch, [reply, reply])
    draft = outline_draft.OutlineDraft(observations, [], root=tmp_path / "drafts", single_pass_outline=True,
        check_ra_feasibility=True, check_cca_safety=True, check_nonblocking=True)
    asyncio.run(draft.generate())
    assert draft.record["status"] == "draft"
    rewrite = _outline_prompts(provider)[1]
    assert "It was refused after its last event: normal production cannot carry on from the state it leaves." in rewrite
    assert "reason=normal production cannot finish KET4" in rewrite
    assert [row["refused_event_index"] for row in draft.record["single_pass_checks"]] == [None, None]
    assert [row["accepted"] for row in draft.record["single_pass_checks"]] == [False, True]


@pytest.mark.parametrize("reply", [
    {"thought": "Done.", "decision": "recovery_complete"},
    {"thought": "Nothing to do.", "events": []},
    {"thought": "No events at all."},
])
def test_single_pass_reply_code_cannot_read_ends_the_draft(observations, tmp_path, monkeypatch, reply):
    provider = scripted_provider(monkeypatch, [reply, single_pass_reply(("finish", "completed"))])
    draft = outline_draft.OutlineDraft(observations, [], root=tmp_path, single_pass_outline=True)
    asyncio.run(draft.generate())
    assert provider.await_count == 2
    assert draft.record["status"] == "draft_incomplete"
    assert draft.record["stop_cause"]["cause"] == "error" and draft.record["stop_cause"]["exception"] == "ValueError"


def test_single_pass_lookup_may_carry_an_empty_events_list(observations, tmp_path, monkeypatch):
    provider = scripted_provider(monkeypatch, [{**LOOKUP, "events": []}, single_pass_reply(("finish", "completed"))])
    draft = outline_draft.OutlineDraft(observations, [], root=tmp_path, single_pass_outline=True)
    asyncio.run(draft.generate())
    assert draft.record["status"] == "draft" and provider.await_count == 3
    assert "/resources/owner/snapshot" in draft.record["retrievals"]


def test_an_error_after_lookups_still_records_them_and_the_budget(observations, tmp_path, monkeypatch):
    # Lookups run inside one multi_turn call, so the reply that cannot be read raises in that same call.
    provider = scripted_provider(monkeypatch, [LOOKUP, {"thought": "Done.", "decision": "recovery_complete"}])
    draft = outline_draft.OutlineDraft(observations, [], root=tmp_path, single_pass_outline=True, max_turns=80)
    asyncio.run(draft.generate())
    assert draft.record["status"] == "draft_incomplete" and provider.await_count == 3
    assert draft.record["retrievals"] == ["/resources/owner/snapshot"]
    assert draft.record["stop_cause"] == {"cause": "error", "exception": "ValueError", "turn_index": 3, "max_turns": 80}
    assert draft.record["turn_decision_order"] == ["need_context"]


def _checked_single_pass_draft(observations, tmp_path, monkeypatch, *, checks=True):
    """Generate a finished single-pass draft with stand-in checks, all on or all off."""
    _stand_in_checks(monkeypatch)
    scripted_provider(monkeypatch, [single_pass_reply(("finish", "completed"))])
    draft = outline_draft.OutlineDraft(observations, [], root=tmp_path / "drafts", single_pass_outline=True,
        check_ra_feasibility=checks, check_cca_safety=checks, check_nonblocking=checks)
    asyncio.run(draft.generate())
    assert draft.record["status"] == "draft", draft.record["message"]
    return draft


def test_checked_drafts_lists_only_finished_drafts_every_check_accepted(observations, tmp_path, monkeypatch):
    checked = _checked_single_pass_draft(observations, tmp_path, monkeypatch)
    _checked_single_pass_draft(observations, tmp_path, monkeypatch, checks=False)
    rows = outline_draft.checked_drafts(observations["source_path"], root=tmp_path / "drafts")
    assert [row["run_id"] for row in rows] == [checked.record["run_id"]]
    assert rows[0]["events"] == 1 and rows[0]["single_pass_outline"] is True
    assert outline_draft.checked_drafts(str(tmp_path / "other.json"), root=tmp_path / "drafts") == []


def test_a_loaded_checked_draft_composes_for_review_without_changing_the_draft(observations, tmp_path, monkeypatch):
    from pathlib import Path

    from cais_spade_llm.recovery_framework import part_slippage_example, primitive_composition

    owner = observations["prepared_recovery_request"]["recovery_resources"]["owner"]
    owner["execution_primitive_catalog"] = [
        {"name": "move_to_named_pose", "params": {}, "effects": {"resource_state": {"set": "completed"}}}]
    owner["resource_evidence"] = {"state": {"_task_ctx": {"tz": 0.1}}}
    draft = _checked_single_pass_draft(observations, tmp_path, monkeypatch)
    # The supplied capture has no scene to rebuild the runtime models and read the current declarations from.
    from cais_spade_llm.resources import resource_primitives

    observations["inputs"] = {"scene": {}}
    monkeypatch.setattr(primitive_composition, "_composition_physical_evidence",
                        lambda observations, prepared: {"environment_models": {"M2": {}}})
    monkeypatch.setattr(resource_primitives, "current_primitive_declarations", lambda scene: {"M2": {
        "move_to_named_pose": [{"preconditions": {}, "effects": {"resource_state": {"set": "completed"}}}]}})
    path = draft.directory / "draft.json"
    saved = path.read_bytes()
    original_load, original_checked = outline_draft.load_draft, outline_draft.checked_drafts
    monkeypatch.setattr(outline_draft, "load_draft", lambda path, obs, **kwargs: original_load(path, obs, root=tmp_path / "drafts"))
    monkeypatch.setattr(outline_draft, "checked_drafts", lambda source: original_checked(source, root=tmp_path / "drafts"))
    monkeypatch.setattr(part_slippage_example, "read_observations", lambda source: deepcopy(observations))
    actual = primitive_composition.DraftComposition
    monkeypatch.setattr(primitive_composition, "DraftComposition",
                        lambda *args, **kwargs: actual(*args, **kwargs, root=tmp_path / "compositions"))

    async def compose(planner, prepared, *, session_state):
        assert prepared["_stop_after_multi_turn_phase"] == "primitive"
        assert session_state["current_phase"] == "primitive_generation"
        # The capture's recorded resources check their own programs, with their recorded contracts.
        assert [owner.jid for owner in planner.resource_agents] == ["owner"]
        assert prepared["recovery_resources"]["owner"]["execution_primitive_catalog"]
        # Each owner checks its program against its recorded evidence and the capture's physical evidence.
        assert prepared["recovery_resources"]["owner"]["resource_evidence"] == {"state": {"_task_ctx": {"tz": 0.1}}}
        assert prepared["physical_evidence"] == {"environment_models": {"M2": {}}}
        prepared["multi_turn_session_state"] = {**session_state, "final_output": {"primitive_program_complete": True},
                                                 "accepted_primitive_program": [{"outline_id": "R1"}]}

    monkeypatch.setattr(multi_turn, "execute_multi_turn_recovery", AsyncMock(side_effect=compose))
    monkeypatch.setattr(multi_turn, "build_multi_turn_recovery_proposal",
                        Mock(return_value={"accepted": True, "recovery_proposal": {"macro_tasks": []}}))
    example = PartSlippageExample(SimpleNamespace(system_running=False))
    example.source, example.scenario = "offline", observations["scenario"]
    example.observation_path = observations["source_path"]

    async def run():
        assert [row["path"] for row in example.snapshot()["checked_drafts"]] == [str(path)]
        with pytest.raises(ValueError, match="listed checked drafts"):
            example.load_checked_draft(str(tmp_path / "drafts" / "draft_other" / "draft.json"))
        example.load_checked_draft(str(path))
        await example._task
        state = example.snapshot()
        assert state["outline_draft"]["run_id"] == draft.record["run_id"]
        assert state["can_generate_composition"] is True, state["composition_readiness"]
        await example.action("generate_primitives")
        await example._composition._task
        state = example.snapshot()
        assert state["primitive_status"] == "completed", state.get("message")
        assert state["primitives"] == [{"outline_id": "R1"}] and state["execution_authorized"] is False
        assert Path(state["composition_path"]).is_file()
        assert Path(state["composition_path"]).is_relative_to(tmp_path / "compositions")
        with pytest.raises(ValueError, match="review only"):
            await example.action("execute")

    asyncio.run(run())
    assert path.read_bytes() == saved


def test_a_checked_draft_the_current_checks_refuse_loads_for_review_only(observations, tmp_path, monkeypatch):
    from cais_spade_llm.recovery_framework import part_slippage_example, primitive_composition

    observations["prepared_recovery_request"]["recovery_resources"]["owner"]["execution_primitive_catalog"] = [
        {"name": "move_to_named_pose", "params": {}, "effects": {"resource_state": {"set": "completed"}}}]
    draft = _checked_single_pass_draft(observations, tmp_path, monkeypatch)
    monkeypatch.setattr(primitive_composition, "_composition_physical_evidence", lambda observations, prepared: {})
    # The code changed after the draft: the RA now refuses its event.
    _stand_in_checks(monkeypatch, refused={"finish"})
    original_load, original_checked = outline_draft.load_draft, outline_draft.checked_drafts
    monkeypatch.setattr(outline_draft, "load_draft", lambda path, obs, **kwargs: original_load(path, obs, root=tmp_path / "drafts"))
    monkeypatch.setattr(outline_draft, "checked_drafts", lambda source: original_checked(source, root=tmp_path / "drafts"))
    monkeypatch.setattr(part_slippage_example, "read_observations", lambda source: deepcopy(observations))
    actual = primitive_composition.DraftComposition
    monkeypatch.setattr(primitive_composition, "DraftComposition",
                        lambda *args, **kwargs: actual(*args, **kwargs, root=tmp_path / "compositions"))
    example = PartSlippageExample(SimpleNamespace(system_running=False))
    example.source, example.scenario = "offline", observations["scenario"]
    example.observation_path = observations["source_path"]

    async def run():
        example.load_checked_draft(str(draft.directory / "draft.json"))
        await example._task
        state = example.snapshot()
        assert state["outline_draft"]["run_id"] == draft.record["run_id"]
        assert state["can_generate_composition"] is False and state["primitive_status"] == "unavailable"
        assert "event 1 is refused: the RA fact" in state["composition_readiness"]["reasons"][0]
        with pytest.raises(ValueError, match="NEEDS_CONTEXT"):
            await example.action("generate_primitives")

    asyncio.run(run())
    assert not (tmp_path / "compositions").exists()


def test_composition_takes_the_recorded_interface_and_the_current_contracts():
    from cais_spade_llm.recovery_framework.primitive_composition import _declared_catalog

    recorded = [
        {"name": "start_job", "params": {"job_id": {"type": "string"}}, "required_params": ["job_id"],
         "description": "Start a job.", "preconditions": {}, "effects": {}, "semantic_summary": "stale"},
        {"name": "detect_parts", "params": {}, "output_schema": {"type": "object"}, "effects": {"stale": {"set": 1}}},
        {"name": "open_gripper", "params": {}, "effects": {}}]
    declared = {
        "start_job": [{"preconditions": {"selected_program": {"equals": program}}, "parameters": {"result": {"enum": [program]}},
                       "effects": {"resource_state": {"set": "running"}},
                       "product_effects": {"current_process": {"set_status": "running"}}} for program in ("circle", "square")],
        "detect_parts": [],
        "complete_job": [{"controllable": False, "after_sec": {"$evidence": "job.processing_time_sec"},
                          "effects": {"resource_state": {"set": "completed"}},
                          "product_effects": {"processCompleted": [{"process": "trim", "result": {"set_from_param": "result"}}]}}]}
    rows = _declared_catalog(recorded, declared)
    # One row per declared alternative, with the recorded interface; nothing the code does not offer for recovery.
    assert [row["name"] for row in rows] == ["start_job", "start_job", "detect_parts", "complete_job"]
    assert rows[1]["params"] == {"job_id": {"type": "string"}} and rows[1]["required_params"] == ["job_id"]
    assert rows[1]["preconditions"] == {"selected_program": {"equals": "square"}}
    assert rows[1]["product_effects"] == {"current_process": {"set_status": "running"}}
    assert rows[1]["semantic_summary"] == "Start a job. (pre: selected_program; effects: resource_state)"
    # A primitive its executor leaves undeclared keeps its interface and declares nothing.
    assert rows[2]["effects"] == {} and rows[2]["output_schema"] == {"type": "object"}
    # An uncontrollable outcome no executor records keeps its declaration; its parameters are those its effects read.
    outcome = rows[3]
    assert outcome["controllable"] is False and outcome["after_sec"] == {"$evidence": "job.processing_time_sec"}
    assert outcome["params"] == {"result": {}} and outcome["required_params"] == ["result"]
    assert "output_schema" not in outcome and "description" not in outcome
