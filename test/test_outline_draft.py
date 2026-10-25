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
    return {"thought": "A supplied test draft, not a generated recovery.", "candidate_events": [{
        "outline_id": "draft_step", "event_name": "Arbitrary task event", "resource": "M2",
        "part_name": "KET4_Square_4mm", "expected_end_state": {"resource_state": "completed"},
        "rationale": "Assumption remains unvalidated."}]}


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
            "job control: resume_after_reload true, restart_stopped_jobs false\n") in prompt


@pytest.mark.parametrize("responses, expected_calls, status", [
    ([response(), DONE], 3, "draft"),
    # Nothing checks a reply; one code cannot read ends the draft with its error.
    ([{}, response(), DONE], 2, "draft_incomplete"),
    ([{"thought": "Need facts", "decision": "need_context", "context_requests": ["/resources"]}] * 8, 3, "draft_incomplete"),
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
