from __future__ import annotations

"""Part slippage stage modes, review ownership and simulation-only execution."""

import asyncio
import hashlib
import json
import threading
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from cais_spade_llm.recovery_framework.part_slippage_example import (
    PartSlippageExample,
    example_for_bridge,
    example_setup,
)
from cais_spade_llm.recovery_framework.part_slippage_recovery import (
    PartSlippageRecovery,
    diagnostic_setup,
)
from cais_spade_llm.ui.recovery_evidence import list_examples, read_stage_record, stage_records


def _accepted_composition_inputs():
    event = {"outline_id": "step1", "resource_jid": "owner", "event_name": "move_home",
             "part_name": None, "expected_start_state": {"mode": "failed"},
             "expected_end_state": {"mode": "idle"}}
    return {"recovery_resources": {"owner": {
        "recovery_des_model": {"state_variables": {"mode": {"scope": "resource"}}},
        "execution_primitive_catalog": [{"name": "move_to_named_pose", "params": {},
                                           "effects": {"mode": {"set": "idle"}}}]}},
        "multi_turn_session_state": {"current_phase": "primitive_generation", "accepted_outline_prefix": [event]}}


@pytest.fixture
def recovery(tmp_path, monkeypatch):
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes import multi_turn

    setup = example_setup()
    runtime = SimpleNamespace(
        diagnostic_cca_bypass=True, setup=setup, path=tmp_path, agent=SimpleNamespace(process_planner=object()),
        context=SimpleNamespace(run_id="test-run", revision=4, admission_lock=threading.RLock(), environment_model={},
                                inputs={}, resources={}, part_tracker={}, initial_product_states={}),
        jids={},
        nominal_idle=asyncio.Event(), queue_save=Mock(), stopped=True, _calculation_stop=threading.Event(),
        outcome={"status": "blocked"}, active_executions=set(),
    )
    runtime.nominal_idle.set()
    session = PartSlippageRecovery(runtime)
    runtime.part_slippage_recovery = session

    async def prepare(feedback=""):
        directory = session._new_artifact_directory()
        session._context_revision = runtime.context.revision
        session._proposal = {}
        session._outline_request = {}
        session._prepared = {**_accepted_composition_inputs(),
                             "recovery_debug": {"artifact_directory": str(directory)}, "recovery_feedback": feedback}

    async def generate(planner, prepared, *, session_state):
        stage = prepared["_stop_after_multi_turn_phase"]
        session_state = deepcopy(session_state or {})
        if stage == "outline":
            session_state.update(_accepted_composition_inputs()["multi_turn_session_state"])
        else:
            session_state.update(final_output={"primitive_program_complete": True}, accepted_primitive_program=[{"primitive_steps": [{"primitive": "move_to_named_pose"}]}])
        prepared["multi_turn_session_state"] = session_state

    monkeypatch.setattr(session, "_prepare", prepare)
    monkeypatch.setattr(multi_turn, "execute_multi_turn_recovery", AsyncMock(side_effect=generate))
    monkeypatch.setattr(multi_turn, "build_multi_turn_recovery_proposal", Mock(return_value={
        "accepted": True, "recovery_proposal": {"macro_tasks": [{"outline_id": "step1"}]},
    }))
    return session


@pytest.mark.parametrize("outline_mode", ["auto", "manual"])
@pytest.mark.parametrize("primitive_mode", ["auto", "manual"])
def test_stage_modes_stop_before_execution(recovery, monkeypatch, outline_mode, primitive_mode):
    execute = AsyncMock()
    monkeypatch.setattr("cais_spade_llm.recovery_framework.slippage_execution.execute_generated_recovery", execute)
    recovery._state.update(outline_mode=outline_mode, primitive_mode=primitive_mode)

    async def run():
        recovery.begin({"injection_status": "completed"})
        await recovery._task
        if outline_mode == "manual":
            assert recovery.snapshot()["status"] == "outline_ready"
            recovery.action("generate_outline")
            await recovery._task
            assert recovery.snapshot()["status"] == "outline_review"
            recovery.action("continue_outline")
            await recovery._task
        if primitive_mode == "manual":
            assert recovery.snapshot()["status"] == "primitive_ready"
            recovery.action("generate_primitives")
            await recovery._task
        assert recovery.snapshot()["status"] == "ready_to_execute"
        execute.assert_not_called()
        assert recovery.snapshot()["execution_authorized"] is False
        with pytest.raises(ValueError, match="does not authorize execution"):
            recovery.action("execute")
        # Retain coverage of the pre-composition execution and nominal-resume path.
        recovery._state.pop("composition_validation_version")
        recovery._proposal.pop("composition_validation_version", None)
        recovery.action("execute")
        with pytest.raises(ValueError, match="already running"):
            recovery.action("execute")
        await recovery._task
        execute.assert_awaited_once()
        assert recovery.snapshot()["status"] == "resuming"
        recovery.runtime.outcome = {"status": "completed"}
        recovery.nominal_finished()
        assert recovery.snapshot()["status"] == "completed"
        assert recovery.snapshot()["safety_status"] == "skipped"
    asyncio.run(run())


def test_outline_evidence_retry_does_not_generate_or_replace_llm_capture(recovery, monkeypatch):
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes import multi_turn_validated_event

    async def retry(*, prepared, session, planner):
        assert session["projected_physical_evidence"]["event_evidence"] == [{"source": "owner preview"}]
        assert session["projected_physical_evidence"]["scene_object_bindings"] == {"part": {"object_id": "part/link/collision"}}
        assert session["projected_physical_evidence"]["support_collision_object_bindings"] == {"table/link/surface": "table"}
        session["accepted_outline_prefix"] = [{"outline_id": "existing_proposal"}]
        session.pop("pending_outline_validation")
        return "need_next_task", {"source_turn_index": 2, "transition_validation": {"status": "passed"}}

    async def run():
        await recovery._prepare()
        recovery._state["validation_contract_version"] = 2
        recovery._prepared["multi_turn_session_state"] = {
            "pending_outline_validation": {"response": {"thought": "recorded", "next_transition": {"outline_id": "existing_proposal"}}}}
        directory = Path(recovery.snapshot()["artifact_directory"])
        original = directory / "recovery_outline" / "turns" / "turn_00002_outline.json"
        original.parent.mkdir(parents=True)
        original.write_text('{"turn_index":2,"phase":"outline","status":"NEEDS_CONTEXT"}')
        before = original.read_bytes()
        await recovery.retry_outline_validation({"event_evidence": [{"source": "owner preview"}],
            "physical_evidence": {"scene_object_bindings": {"part": {"object_id": "part/link/collision"}},
                                  "support_collision_object_bindings": {"table/link/surface": "table"}}})
        assert recovery.snapshot()["status"] == "outline_ready"
        assert original.read_bytes() == before
        turns = directory / "recovery_outline"
        from cais_spade_llm.ui.recovery_evidence import recovery_turns
        grouped = recovery_turns(turns, "recovery_outline")
        assert len(grouped) == 1
        assert len(grouped[0]["files"]) == 2

    monkeypatch.setattr(multi_turn_validated_event, "retry_pending_event", retry, raising=False)
    asyncio.run(run())


def test_retry_archives_nominal_proof_before_resume_checkpoint(recovery, monkeypatch):
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes import multi_turn_validated_event
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.recovery_artifacts import (
        nominal_continuation_context_binding, read_nominal_continuation,
    )

    proof = {"complete": False, "status": "NEEDS_CONTEXT", "failures": [{"task": {"part_name": None}}]}
    pending = {"kind": "nominal", "source_turn_index": 2, "task": {"part_name": None},
        "projected_physical_evidence": {"snapshot": {"recorded": [1, 2]}}}

    async def retry(*, session, **kwargs):
        assert session["pending_outline_validation"] == pending
        session["nominal_continuation_evidence"] = deepcopy(proof)
        return "need_context", {"source_turn_index": 2, "nominal_continuation_evidence": deepcopy(proof)}

    async def run():
        await recovery._prepare()
        recovery._prepared.update(nominal_continuation_artifact_version=1,
            source_observation_fingerprint="a" * 64,
            multi_turn_session_state={"validation_contract_version": 2, "turn_index": 2,
                                      "pending_outline_validation": deepcopy(pending)})
        await recovery.retry_outline_validation({})
        directory = Path(recovery.snapshot()["artifact_directory"])
        state = recovery._prepared["multi_turn_session_state"]
        assert state["pending_outline_validation"] == pending
        summary = state["nominal_continuation_evidence"]
        assert "artifact_reference" in summary
        assert read_nominal_continuation(summary, directory=directory / "recovery_outline",
            context_binding=nominal_continuation_context_binding(state, recovery._prepared)) == proof
        assert len(list((directory / "recovery_outline/turns").glob("*_validation_*.json"))) == 1
        resume = json.loads((directory / "validation_resume.json").read_text())
        assert resume["multi_turn_session_state"]["nominal_continuation_evidence"] == summary
        assert recovery.snapshot()["status"] == "blocked"

    monkeypatch.setattr(multi_turn_validated_event, "retry_pending_event", retry)
    asyncio.run(run())


def test_offline_nominal_context_error_keeps_the_recorded_reason(recovery, monkeypatch):
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery import recovery_artifacts
    from cais_spade_llm.recovery_framework import outline_gazebo_evidence

    proof = {"status": "NEEDS_CONTEXT", "complete": False, "reason": "Nominal continuation is not established",
             "failures": [{"task_id": "environment_21", "reason": "Missing recorded parameter binding"}]}
    monkeypatch.setattr(recovery_artifacts, "read_nominal_continuation", lambda *args, **kwargs: deepcopy(proof))
    preview = Mock(side_effect=AssertionError("No task is available for a motion preview"))
    monkeypatch.setattr(outline_gazebo_evidence, "pending_outline_event", preview)

    async def run():
        await recovery._prepare()
        recovery.runtime.source_observation_path = "recorded/observations.json"
        recovery._prepared["multi_turn_session_state"] = {"pending_outline_validation": {"kind": "nominal"},
            "accepted_outline_prefix": [{"outline_id": "accepted"}], "nominal_continuation_evidence": proof}
        before = deepcopy(recovery._prepared["multi_turn_session_state"])
        with pytest.raises(ValueError, match="NEEDS_CONTEXT:.*Missing recorded parameter binding"):
            await recovery._prepare_offline_outline_evidence(set())
        assert recovery._prepared["multi_turn_session_state"] == before
        preview.assert_not_called()

    asyncio.run(run())


def test_generation_archives_nominal_proof_before_debug_publication(tmp_path, monkeypatch):
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes import multi_turn
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.recovery_artifacts import read_nominal_continuation

    proof = {"complete": False, "status": "not_ready", "failures": [{"task": {"part_name": None}}]}
    prepared = {"nominal_continuation_artifact_version": 1, "source_observation_fingerprint": "a" * 64,
        "recovery_debug": {"per_turn_debug_dir": str(tmp_path)},
        "multi_turn_session_seed": {"current_phase": "outline", "turn_index": 0, "max_turns": 1,
                                    "validation_contract_version": 2}}
    published = []

    async def handle(*, session_state, **kwargs):
        session_state["nominal_continuation_evidence"] = deepcopy(proof)
        session_state["status"] = "paused_after_outline_turn"
        return "need_next_task", {"nominal_continuation_evidence": deepcopy(proof)}

    async def recorded(call, **kwargs):
        return await call()

    def publish(value):
        saved = value["multi_turn_session"]
        if "nominal_continuation_evidence" in saved:
            assert "artifact_reference" in saved["nominal_continuation_evidence"]
            assert "artifact_reference" in saved["turns"][-1]["nominal_continuation_evidence"]
        published.append(deepcopy(value))

    planner = SimpleNamespace(product_agent=SimpleNamespace(ask_llm_structured=AsyncMock(return_value={})),
                              _set_last_recovery_debug=publish)
    monkeypatch.setattr(multi_turn, "_build_phase_prompt", lambda *_: ({}, "Recorded unchanged request"))
    monkeypatch.setattr(multi_turn, "recorded_structured_call", recorded)
    monkeypatch.setattr(multi_turn, "_write_per_turn_artifact", lambda *_: None)
    monkeypatch.setattr(multi_turn, "_artifact_response_payload", lambda **kwargs: kwargs["parsed_response"])
    monkeypatch.setitem(multi_turn._PHASE_HANDLERS, "outline", handle)
    asyncio.run(multi_turn.execute_multi_turn_recovery(planner, prepared))
    assert len(published) == 2
    summary = prepared["multi_turn_session_state"]["nominal_continuation_evidence"]
    assert read_nominal_continuation(summary, directory=tmp_path / "recovery_outline") == proof
    summary["status"] = "caller mutation"
    assert published[-1]["multi_turn_session"]["nominal_continuation_evidence"]["status"] == "not_ready"


@pytest.mark.parametrize("broken", [None, "source", "file", "context"])
def test_saved_nominal_archive_requires_verified_source_and_context(recovery, tmp_path, broken):
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.recovery_artifacts import archive_nominal_continuation
    from cais_spade_llm.recovery_framework.recovery_observations import observation_fingerprint

    source = tmp_path / "original_generation"
    recovery._observations = {"run_id": "captured", "context_revision": 3}
    state = {"validation_contract_version": 2, "turn_index": 2, "symbolic_parts": {"part": {"part_state": "ready"}},
        "nominal_continuation_evidence": {"complete": True, "status": "passed", "tasks": [], "execution_authorized": False}}
    prepared = {"nominal_continuation_artifact_version": 1, "validation_contract_version": 2,
        "source_observation_fingerprint": observation_fingerprint(recovery._observations),
        "recovery_debug": {"per_turn_debug_dir": str(source)}, "multi_turn_session_state": state}
    turn = {"turn_index": 2, "nominal_continuation_evidence": deepcopy(state["nominal_continuation_evidence"])}
    path = archive_nominal_continuation(prepared, state, turn, decision="outline_ready")
    if broken == "source":
        recovery._observations["context_revision"] += 1
    elif broken == "file":
        path.write_bytes(path.read_bytes() + b" ")
    elif broken == "context":
        state["symbolic_parts"]["part"]["part_state"] = "failed"
    destination = recovery._new_artifact_directory()
    if broken:
        with pytest.raises(ValueError):
            recovery._install_prepared(prepared, destination, "", new_attempt=False, source_directory=source)
    else:
        recovery._install_prepared(prepared, destination, "", new_attempt=False, source_directory=source)
        assert recovery._prepared == prepared


def test_primitive_regeneration_retains_verified_nominal_archive(recovery, tmp_path):
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.recovery_artifacts import (
        archive_nominal_continuation, read_nominal_continuation, nominal_continuation_context_binding,
    )

    source = tmp_path / "original_generation"
    proof = {"complete": True, "status": "passed", "tasks": [], "execution_authorized": False}
    inputs = _accepted_composition_inputs()
    state = {**inputs["multi_turn_session_state"], "validation_contract_version": 2, "turn_index": 2,
             "nominal_continuation_evidence": deepcopy(proof)}
    prepared = {**inputs, "nominal_continuation_artifact_version": 1, "validation_contract_version": 2,
        "source_observation_fingerprint": "a" * 64, "recovery_session": {},
        "recovery_debug": {"per_turn_debug_dir": str(source)}, "multi_turn_session_state": state}
    turn = {"turn_index": 2, "nominal_continuation_evidence": deepcopy(proof)}
    path = archive_nominal_continuation(prepared, state, turn, decision="outline_ready")
    original = path.read_bytes()
    recovery.offline = True
    recovery._outline_request = deepcopy(prepared)
    recovery._state.update(status="offline_completed", validation_contract_version=2)

    async def run():
        recovery.action("regenerate_primitives")
        await recovery._task

    asyncio.run(run())
    assert recovery.snapshot()["status"] == "offline_completed"
    directory = Path(recovery.snapshot()["artifact_directory"]) / "recovery_outline"
    saved = recovery._prepared["multi_turn_session_state"]["nominal_continuation_evidence"]
    assert read_nominal_continuation(saved, directory=directory,
        context_binding=nominal_continuation_context_binding(recovery._prepared["multi_turn_session_state"], recovery._prepared)) == proof
    assert path.read_bytes() == original


@pytest.mark.parametrize("field", ["scene_object_bindings", "support_collision_object_bindings"])
def test_preview_retry_rejects_changed_object_mapping_without_session_mutation(recovery, field):
    async def run():
        await recovery._prepare()
        recovery._prepared["multi_turn_session_state"] = {
            "pending_outline_validation": {"kind": "event"},
            "projected_physical_evidence": {field: {"object": {"source": "recorded"}}}}
        before = deepcopy(recovery._prepared)
        with pytest.raises(ValueError, match="changes an existing"):
            await recovery.retry_outline_validation({"physical_evidence": {field: {"object": {"source": "different"}}}})
        assert recovery._prepared == before

    asyncio.run(run())


def test_outline_checkpoint_requires_recorded_joints_and_matches_facts():
    from cais_spade_llm.recovery_framework.outline_gazebo_evidence import checkpoint_differences

    state = {"held_part": "part", "joint_names": ["joint"], "joint_positions": [1.]}
    record = {
        "jids": {"robot": "robot@localhost"}, "inputs": {}, "program_revision": "r1",
        "resource_states": {"robot": {"held_part": "part"}}, "part_tracker": {}, "pending_nodes": [],
        "context_revision": 4, "acknowledgements": {}, "transitions": [], "monitor_history": {"active": False},
        "physical_evidence": {"snapshot": {"resources": {"robot": deepcopy(state)}},
            "resources": {"robot": {"observation_state": state, "attachment": {"model_name": "part"}}},
            "resource_models": {"robot": {"version": 1}}, "geometry": {"frame": "world"}},
    }
    fresh = deepcopy(record)
    assert checkpoint_differences(record, fresh) == []
    for field, changed in (("context_revision", 5), ("acknowledgements", {"task": {"status": "completed"}}),
                           ("transitions", [{"event_name": "different_event"}]), ("monitor_history", {"active": True})):
        other = {**deepcopy(record), field: changed}
        assert f"{field} changed; use new observations" in checkpoint_differences(record, other)
        other.pop(field)
        assert f"Missing {field}; use new observations" in checkpoint_differences(record, other)
        other[field] = None
        assert f"Unavailable {field}; use new observations" in checkpoint_differences(record, other)
    assert "context_revision changed; use new observations" in checkpoint_differences(
        record, {**deepcopy(record), "context_revision": 4.0})
    fresh["physical_evidence"]["resources"]["robot"]["observation_state"]["joint_positions"] = [2.]
    assert "robot.joint_positions changed" in checkpoint_differences(record, fresh)
    del record["physical_evidence"]["resources"]["robot"]["observation_state"]["joint_positions"]
    assert "Missing robot.joint_positions" in checkpoint_differences(record, fresh)


@pytest.mark.parametrize("physical_source", ["observations", "prepared", "projected", "pending"])
def test_explicit_archived_preview_keeps_recorded_facts_and_does_not_recapture(tmp_path, monkeypatch, physical_source):
    from cais_spade_llm.recovery_framework import outline_gazebo_evidence as preview
    from cais_spade_llm.recovery_framework.recovery_observations import fingerprint

    physical = {"snapshot": {"resources": {"robot": {"joint_positions": [1.]}}}}
    physical["evidence_fingerprint"] = fingerprint(physical)
    scene = {key: {"saved": True} for key in ("planning_scene", "start_state")}
    scene.update(status="captured", physical_evidence_fingerprint=physical["evidence_fingerprint"],
                 scene_fingerprint="scene", model_fingerprint="model", robot_description="saved URDF",
                 robot_description_semantic="saved SRDF")
    observations = {"physical_evidence": physical, "planning_scene_capture": scene,
                    "resource_evidence": {"robot@localhost": {"state": {"_current_state": "picked"}}}}
    event = {"outline_id": "proposal", "resource_jid": "robot@localhost"}
    session = {"pending_outline_validation": {"response": {"next_transition": event}}}
    recovery = SimpleNamespace(_prepared={"multi_turn_session_state": session,
        "recovery_resources": {"robot@localhost": {"recovery_snapshot": {}, "recovery_des_model": {}}}},
        _observations=observations, _generation=1, _stopped=False,
        snapshot=lambda: {"artifact_directory": str(tmp_path)})
    expected_physical = deepcopy(physical)
    if physical_source != "observations":
        expected_physical["physical_preview_contract_version"] = 2
        recovery._prepared["physical_evidence"] = deepcopy(expected_physical)
    if physical_source in {"projected", "pending"}:
        expected_physical["snapshot"]["resources"]["robot"]["joint_positions"] = [1.1]
        session["projected_physical_evidence"] = deepcopy(expected_physical)
    if physical_source == "pending":
        expected_physical["snapshot"]["resources"]["robot"]["joint_positions"] = [1.2]
        session["pending_outline_validation"]["projected_physical_evidence"] = deepcopy(expected_physical)
    owner = SimpleNamespace(jid="robot@localhost", prepare_recovery_outline_event_evidence=AsyncMock(
        return_value={"status": "prepared", "command_sent": False, "event_evidence": []}))
    runtime = SimpleNamespace(setup={"execution_mode": "simulation"}, conveyor_fault=SimpleNamespace(status="triggered"),
        stopped=True, active_executions=set(), context=SimpleNamespace(reservations={}, revision=4),
        part_slippage_recovery=SimpleNamespace(_task=None), resource_agents=[owner])
    live_capture = Mock(side_effect=AssertionError("Archived prediction must not recapture live physics"))
    monkeypatch.setattr(preview, "capture_observations", live_capture)
    original = deepcopy(observations)
    source_fingerprint = fingerprint(observations)
    original_prepared = deepcopy(recovery._prepared)
    result = asyncio.run(preview.prepare_gazebo_evidence(recovery, runtime))
    assert result["status"] == "prepared"
    assert result["evidence_source"] == "saved planning scene"
    assert result["physical_execution_verified"] is False
    assert result["execution_authorized"] is False
    assert "resource_evidence_capture" not in result["evidence"]
    assert owner.prepare_recovery_outline_event_evidence.call_args.kwargs["observations"] == original
    provided = owner.prepare_recovery_outline_event_evidence.call_args.kwargs["physical_evidence"]
    assert provided == expected_physical
    from cais_spade_llm.resources.recovery_geometry import outline_event_binding

    assert outline_event_binding(event, {}, {}, provided) == outline_event_binding(event, {}, {}, expected_physical)
    assert recovery._prepared == original_prepared
    assert observations == original
    assert fingerprint(observations) == source_fingerprint
    live_capture.assert_not_called()
    observations["physical_evidence"]["snapshot"]["resources"]["robot"]["joint_positions"] = [2.]
    with pytest.raises(ValueError, match="unchanged physical capture"):
        asyncio.run(preview.prepare_gazebo_evidence(recovery, runtime))


def test_owner_evidence_capture_preserves_null_and_has_no_controller_calls():
    from cais_spade_llm.recovery_framework.recovery_observations import capture_resource_evidence

    controller = SimpleNamespace(controller_config={"move_group": {"frame_id": "world"}}, move_cartesian=Mock())
    owner = SimpleNamespace(agent_name="robot", jid="robot@localhost", _held_part=None, _current_state="idle",
                            _position={"x": 1.}, _controller=controller)
    runtime = SimpleNamespace(context=SimpleNamespace(run_id="capture", revision=8, inputs={}), resource_agents=[owner],
                              setup={"execution_mode": "simulation"})
    result = capture_resource_evidence(runtime, {"resources": {"robot": {"attachment": {"model_name": None}}}})
    assert result["robot@localhost"]["state"]["_held_part"] is None
    assert result["robot@localhost"]["state"]["_current_state"] == "idle"
    assert "_task_ctx" not in result["robot@localhost"]["state"]
    assert result["robot@localhost"]["controller_config"] == controller.controller_config
    assert result["robot@localhost"]["attachment"]["model_name"] is None
    controller.move_cartesian.assert_not_called()


@pytest.mark.parametrize("state", [{}, {"resource_state": None}, {"resource_state": "available"}])
def test_fresh_preparation_uses_recorded_owner_state_without_inventing_resource_state(state):
    from cais_spade_llm.recovery_framework.outline_gazebo_evidence import _fresh_preparation

    planner = SimpleNamespace(_build_resource_search_state=Mock(return_value={}),
                              prepare_recovery_request=AsyncMock(return_value={"recovery_resources": {}}))
    runtime = SimpleNamespace(agent=SimpleNamespace(process_planner=planner))
    fresh = {"part_tracker": {}, "jids": {"Storage": "storage@localhost"},
             "resource_states": {"Storage": state},
             "resource_evidence": {"storage@localhost": {"state": {"_current_state": "idle"}}}}
    original = deepcopy(fresh)
    template = {"ra_jid": "storage@localhost", "P_id": [], "goal_state": "assembled",
                "tools_catalog": [], "obligation_targets": [], "failure_context_raw": {}}
    if "resource_state" in state and state["resource_state"] is None:
        with pytest.raises(ValueError, match="recorded resource_state or owner _current_state"):
            asyncio.run(_fresh_preparation(runtime, fresh, template))
        planner.prepare_recovery_request.assert_not_called()
    else:
        asyncio.run(_fresh_preparation(runtime, fresh, template))
        received = planner.prepare_recovery_request.call_args.kwargs["resource_states"]["storage@localhost"]
        assert received["current_state"] == state.get("resource_state", "idle")
    assert fresh == original


def test_fresh_preparation_retains_declared_fields_without_resource_state():
    from cais_spade_llm.recovery_framework.outline_gazebo_evidence import _fresh_preparation

    planner = SimpleNamespace(_build_resource_search_state=Mock(return_value={}),
                              prepare_recovery_request=AsyncMock(return_value={"recovery_resources": {}}))
    runtime = SimpleNamespace(agent=SimpleNamespace(process_planner=planner))
    values = {"zone_4_part": None}
    fresh = {"part_tracker": {}, "jids": {"Buffer For Machined parts": "buffer@localhost"},
             "resource_states": {"Buffer For Machined parts": values},
             "resource_evidence": {"buffer@localhost": {"state": {}, "snapshot": values}}}
    template = {"ra_jid": "buffer@localhost", "P_id": [], "goal_state": "assembled",
                "tools_catalog": [], "obligation_targets": [], "failure_context_raw": {}}
    asyncio.run(_fresh_preparation(runtime, fresh, template))
    received = planner.prepare_recovery_request.call_args.kwargs
    assert received["recorded_resource_snapshots"]["buffer@localhost"] == values
    assert received["resource_states"]["buffer@localhost"] == values
    assert received["default_resource_state"] == ""


@pytest.mark.parametrize("conflict", [False, True])
def test_fresh_preparation_uses_frozen_kmr_snapshot_despite_recorded_joints(monkeypatch, conflict):
    from cais_spade_llm.recovery_framework.outline_gazebo_evidence import _fresh_preparation
    from cais_spade_llm.resources.robot import robot_profile

    builder = Mock(side_effect=AssertionError("KMR must use its own recorded snapshot"))
    monkeypatch.setattr(robot_profile, "_robot_snapshot_builder", builder)
    values = {"resource_state": "idle", "held_part": None, "base_pose": [1., 2., 0.],
              "joint_names": ["joint_1"], "joint_positions": [0.], "optional": None}
    owner = SimpleNamespace(agent_name="KMR", jid="kmr@localhost", static_capabilities={"resource_type": "kmr"},
                            get_recovery_snapshot=Mock(side_effect=AssertionError("Do not read the owner again")))
    planner = SimpleNamespace(_build_resource_search_state=Mock(return_value={}),
                              prepare_recovery_request=AsyncMock(return_value={"recovery_resources": {}}))
    runtime = SimpleNamespace(agent=SimpleNamespace(process_planner=planner), resource_agents=[owner])
    fresh = {"part_tracker": {}, "jids": {"KMR": "kmr@localhost"}, "resource_states": {"KMR": deepcopy(values)},
             "unavailable_resources": ["KMR"], "inputs": {"scene": {"robots": []}},
             "resource_evidence": {"kmr@localhost": {"snapshot": deepcopy(values)}},
             "physical_evidence": {"snapshot": {"resources": {"KMR": deepcopy(values)}}}}
    if conflict:
        fresh["resource_evidence"]["kmr@localhost"]["snapshot"]["joint_positions"] = [False]
    original = deepcopy(fresh)
    template = {"ra_jid": "kmr@localhost", "P_id": [], "goal_state": "", "tools_catalog": [],
                "obligation_targets": [], "failure_context_raw": {}}
    if conflict:
        with pytest.raises(ValueError, match="conflicting captured KMR.joint_positions"):
            asyncio.run(_fresh_preparation(runtime, fresh, template))
        planner.prepare_recovery_request.assert_not_called()
    else:
        asyncio.run(_fresh_preparation(runtime, fresh, template))
        received = planner.prepare_recovery_request.call_args.kwargs["recorded_resource_snapshots"]["kmr@localhost"]
        assert received == {**values, "availability": "unavailable"}
        received["base_pose"][0] = 8.
        assert fresh["resource_evidence"]["kmr@localhost"]["snapshot"] == values
    assert fresh == original
    builder.assert_not_called()
    owner.get_recovery_snapshot.assert_not_called()


@pytest.mark.parametrize('defect', [None, 'custody', 'pose', 'frame', 'revision', 'run'])
def test_frozen_kmr_measurements_preserve_history_and_reject_conflicts(defect):
    from cais_spade_llm.recovery_framework.recovery_observations import reconcile_resource_snapshot

    old = [1., 2., 3., 0., 0., 0., 1.]
    new = [2., 3., 4., 0., 0., 0., 1.]
    owner = {'snapshot': {'current_pose': old, 'held_part': None, 'last_execution_evidence': {'old': True}},
             'state': {'_primitive_state': {'current_pose': old}}, 'source': {'run_id': 'run', 'revision': 5}}
    state = {'current_pose': new, 'held_part': None}
    physical = {'provenance': {'run_id': 'run', 'context_revision': 5},
        'snapshot': {'frame': 'world', 'resources': {'KMR': deepcopy(state)}},
        'resources': {'KMR': {'observation_state': deepcopy(state), 'current_pose': new, 'frame': 'world',
            'source': 'registered owner / GETRECOVERYSTATE', 'simulation_time': 90., 'observed_monotonic': 100.}}}
    if defect == 'custody':
        owner['snapshot']['held_part'] = 'part'
    elif defect == 'pose':
        physical['resources']['KMR']['current_pose'] = old
    elif defect == 'frame':
        physical['resources']['KMR']['frame'] = 'another-frame'
    elif defect == 'revision':
        physical['provenance']['context_revision'] = 6
    elif defect == 'run':
        physical['provenance']['run_id'] = 'another-run'
    before = deepcopy((owner, physical))
    result = reconcile_resource_snapshot(owner, physical, resource_id='KMR', run_id='run', revision=5, inputs={})
    assert bool(result['conflicts']) is (defect is not None)
    assert result['owner_snapshot']['current_pose'] == old
    assert result['snapshot']['current_pose'] == new
    assert 'last_execution_evidence' not in result['snapshot']
    assert result['state']['_primitive_state']['current_pose'] == old
    assert (owner, physical) == before


def test_failed_preparation_keeps_raw_capture_and_error(tmp_path, monkeypatch):
    from cais_spade_llm.recovery_framework import outline_gazebo_evidence as module

    fresh = {'captured_at': 'recorded-time', 'context_revision': 8, 'resource_evidence': {'KMR': {'snapshot': {'held_part': None}}}}
    path = tmp_path / 'observations.json'
    async def fail(runtime, capture, template):
        assert json.loads(path.read_text()) == fresh
        raise ValueError('KMR.current_pose: captured value differs from observation source')
    monkeypatch.setattr(module, '_fresh_preparation', fail)
    with pytest.raises(ValueError, match='KMR.current_pose'):
        asyncio.run(module._prepare_saved_capture(object(), fresh, {}, path))
    assert json.loads(path.read_text()) == fresh
    error = json.loads((tmp_path / 'preparation_error.json').read_text())
    assert error['context_revision'] == 8
    assert 'KMR.current_pose' in error['reason']


def test_fresh_preparation_rejects_changed_revision():
    from cais_spade_llm.recovery_framework.outline_gazebo_evidence import _fresh_preparation

    planner = SimpleNamespace(_build_resource_search_state=Mock(return_value={}),
        prepare_recovery_request=AsyncMock(return_value={'recovery_resources': {}}))
    runtime = SimpleNamespace(agent=SimpleNamespace(process_planner=planner), context=SimpleNamespace(revision=9))
    fresh = {'part_tracker': {}, 'jids': {}, 'resource_states': {}, 'context_revision': 8}
    template = {'ra_jid': 'owner', 'P_id': [], 'goal_state': '', 'tools_catalog': [],
                'obligation_targets': [], 'failure_context_raw': {}}
    with pytest.raises(ValueError, match='changed during preparation'):
        asyncio.run(_fresh_preparation(runtime, fresh, template))


def test_fresh_preparation_rebuilds_actual_failure_observations_without_mutating_capture():
    from cais_spade_llm.recovery_framework.outline_gazebo_evidence import _fresh_preparation

    planner = SimpleNamespace(_build_resource_search_state=Mock(return_value={}),
                              prepare_recovery_request=AsyncMock(return_value={"recovery_resources": {}}))
    runtime = SimpleNamespace(agent=SimpleNamespace(process_planner=planner))
    fresh = {"scenario": "Part slippage", "part_tracker": {"gear_small": {"state": "misplaced", "location": None}},
             "jids": {"ur5e-3": "robot@localhost"}, "resource_states": {"ur5e-3": {"held_part": None}},
             "failure": {"part_name": "gear_small", "observed_drop_pose": {"x": 100., "y": 100., "z": .05},
                         "resource_values_before": {"ur5e-3": {"held_part": "gear_small"}}}}
    template = {"ra_jid": "robot@localhost", "P_id": ["gear_small"], "goal_state": "", "tools_catalog": [],
                "obligation_targets": [], "failure_context_raw": {"failed_task_id": "pending_2",
                    "failed_function_name": "move_cartesian", "failure_context": {
                        "observations": {"observed_drop_pose": {"x": .1}}}}}
    original, original_template = deepcopy(fresh), deepcopy(template)
    asyncio.run(_fresh_preparation(runtime, fresh, template))
    received = planner.prepare_recovery_request.call_args.kwargs
    failure = received["failure_context"]
    assert failure["failed_task_id"] == "pending_2"
    assert failure["failed_function_name"] == "move_cartesian"
    assert failure["failure_context"]["observations"]["observed_drop_pose"] == fresh["failure"]["observed_drop_pose"]
    assert failure["failure_context"]["observations"]["state_before"] == {"held_part": "gear_small"}
    assert failure["failure_context"]["observations"]["state_after"] == {"held_part": None, "part_state": "misplaced", "part_location": None}
    assert received["resource_states"]["robot@localhost"] == {"held_part": None}
    assert received["default_resource_state"] == ""
    assert fresh == original
    assert template == original_template


def test_owner_evidence_capture_satisfies_default_pick_calculation_without_calls():
    from cais_spade_llm.recovery_framework.recovery_observations import capture_resource_evidence
    from cais_spade_llm.resources.robot.gazebo_pick_place_controller import GazeboPickPlaceController
    from cais_spade_llm.resources.robot.recovery_capability import calculate_recovery_pick_target

    controller = GazeboPickPlaceController.__new__(GazeboPickPlaceController)
    controller.pick_z_adjustments_m = {"part": .01}
    controller.controller_config = {"motion": {"pick_tcp_z_bias_min_m": .003, "pick_tcp_z_bias_max_m": .02,
        "approach_height_m": .2, "min_pick_tcp_z_m": 1.07}}
    controller.wait_for_services = Mock(side_effect=AssertionError("live controller access"))
    owner = SimpleNamespace(agent_name="robot", jid="robot@localhost", _controller=controller,
                            static_capabilities={"resource_type": "robot"})
    geometry = {"parts": {"heights_m": {"part": .05}, "model_map": {"part": "part"}},
                "assembly_board": {"slots": {"part": [.1, .2]}, "center": {"z": .975}}}
    runtime = SimpleNamespace(context=SimpleNamespace(run_id="capture", revision=8, inputs={"geometry": geometry}),
                              resource_agents=[owner], setup={"execution_mode": "simulation"})
    pose = [0., 0., 1.5, 0., 1., 0., 0.]
    physical = {"resources": {"robot": {"frame": "world", "tcp_offset_z": -.218,
        "tcp_transform_source": "same observation", "controlled_link": "tool0", "tcp_link": "tcp"}},
        "snapshot": {"resources": {"robot": {"current_pose": pose}},
                     "parts": {"part": {"current_pose": [.2, .1, 1., 0., 0., 0., 1.]}}}}
    evidence = capture_resource_evidence(runtime, physical)["robot@localhost"]
    assert evidence["pick_geometry"]["part"]["part_height_m"] == .05
    assert evidence["pick_policy"]["part"]["source"].startswith("GazeboPickPlaceController.compute_pick_targets")
    result = calculate_recovery_pick_target(task={"resource_jid": "robot@localhost", "part_name": "part"},
        recovery_snapshot={"current_pose": pose}, resource_evidence=evidence)
    assert result["status"] == "calculated", result
    assert result["target_pose"]["z"] == pytest.approx(1.288)
    controller.wait_for_services.assert_not_called()


def test_robot_capture_retains_readiness_without_another_pose_query():
    from cais_spade_llm.recovery_framework.recovery_observations import capture_resource_evidence

    controller = SimpleNamespace(is_usable=Mock(return_value=True), controller_config={},
        get_current_pose=Mock(side_effect=AssertionError('second pose read')))
    owner = SimpleNamespace(agent_name='robot', jid='robot@localhost', _controller=controller,
        static_capabilities={'resource_type': 'robot'}, execution_mode='simulation')
    context = SimpleNamespace(run_id='capture', revision=8,
        inputs={'scene': {'robots': [{'resource_id': 'robot'}]}})
    runtime = SimpleNamespace(context=context, resource_agents=[owner], setup={'execution_mode': 'simulation'})
    pose = [0., 0., 1., 0., 0., 0., 1.]
    physical = {'snapshot': {'resources': {'robot': {'current_pose': pose, 'held_part': None}}}}
    evidence = capture_resource_evidence(runtime, physical)['robot@localhost']
    assert evidence['snapshot']['controller_ready'] is True
    assert evidence['owner_snapshot']['controller_ready'] is True
    assert evidence['snapshot']['current_pose'] == pose
    controller.get_current_pose.assert_not_called()
    controller.is_usable.assert_called_once()


def test_owner_evidence_capture_retains_mesh_width_gripper_and_attachment_sources():
    from cais_spade_llm.recovery_framework.recovery_observations import capture_resource_evidence
    from cais_spade_llm.resources.robot.gazebo_pick_place_controller import GazeboPickPlaceController

    controller = GazeboPickPlaceController.__new__(GazeboPickPlaceController)
    controller.pick_z_adjustments_m = {}
    controller.controller_config = {
        "gripper": {"open": .11, "close": .015},
        "attach": {"robot_model_name": "dual_robot", "primary_attach_link": "ur5e_3_rg2_gripper_tcp"},
        "services": {"attach": "/link_attacher_node/attach"},
    }
    controller.wait_for_services = Mock(side_effect=AssertionError("live controller access"))
    owner = SimpleNamespace(agent_name="robot", jid="robot@localhost", _controller=controller)
    geometry = {"parts": {"model_map": {"gear_small": "gear_small"}, "heights_m": {"gear_small": .02}},
                "assembly_board": {"slots": {"gear_small": [0., 0.]}}}
    runtime = SimpleNamespace(context=SimpleNamespace(run_id="capture", revision=8, inputs={"geometry": geometry}),
                              resource_agents=[owner], setup={"execution_mode": "simulation"})
    identity = {"status": "recorded", "source_files": {"model.sdf": "model_hash"}}
    physical = {"parts": {"gear_small": {"collision_scene_identity": identity}},
                "snapshot": {"parts": {"gear_small": {"model_name": "gear_small"}}}}
    evidence = capture_resource_evidence(runtime, physical)["robot@localhost"]
    assert evidence["pick_geometry"]["gear_small"]["width_m"] == pytest.approx(.021877609252929697)
    source = evidence["pick_geometry_sources"]["gear_small"]
    assert source["kind"] == "local_owner_calculation" and source["frame"] == "part"
    assert source["source_files"] and source["owner_model_fingerprint"]
    assert source["recorded_model_identity"] == identity
    assert evidence["controller_config"] == controller.controller_config
    controller.wait_for_services.assert_not_called()


@pytest.mark.parametrize("width", [None, 0., .035])
def test_owner_grasp_geometry_preserves_declared_width_without_fallback(width, monkeypatch):
    from cais_spade_llm.resources.robot import gazebo_pick_place_controller as module

    controller = module.GazeboPickPlaceController.__new__(module.GazeboPickPlaceController)
    lookup = Mock(side_effect=AssertionError("must retain declared value"))
    monkeypatch.setattr(module, "_model_footprint_width_from_gazebo_world", lookup)
    recorded = {"grasp_width_m": width}
    result = controller.recovery_grasp_geometry(model_name="gear_small", product_geometry=recorded)
    assert result["geometry"] == recorded
    assert result["source"]["kind"] == "recorded_product_geometry"
    lookup.assert_not_called()


def test_tcp_transform_uses_fixed_captured_joint_declarations_without_live_tf():
    from cais_spade_llm.recovery_framework.recovery_observations import _captured_tcp_transform, capture_resource_evidence

    physical = {"controlled_link": "tool0", "tcp_link": "tcp", "tcp_offset_z": -.218}
    configuration = {"description_sha256": "captured-description", "joints": [
        {"name": "mount", "parent": "tool0", "child": "gripper", "type": "fixed",
         "xyz": [0., 0., 0.], "rpy": [0., 0., 0.]},
        {"name": "tcp_joint", "parent": "gripper", "child": "tcp", "type": "fixed",
         "xyz": [0., 0., .218], "rpy": [0., 0., 0.]},
    ]}
    owner = SimpleNamespace(agent_name="robot", jid="robot@localhost", _controller=SimpleNamespace(controller_config={}))
    runtime = SimpleNamespace(context=SimpleNamespace(run_id="capture", revision=1, inputs={}),
                              resource_agents=[owner], setup={"execution_mode": "simulation"})
    record = capture_resource_evidence(runtime, {"resources": {"robot": physical},
        "resource_models": {"robot": {"configuration": configuration}}})["robot@localhost"]
    result = record["motion_config"]["tcp_transform"]
    assert result["translation"] == [0., 0., .218]
    assert result["rotation"] == [0., 0., 0., 1.]
    assert result["parent_frame"] == "tool0" and result["child_frame"] == "tcp" and result["fixed"]
    assert result["source"]["joints"] == configuration["joints"]
    assert result["source"]["description_sha256"] == "captured-description"
    configuration["joints"][-1]["type"] = "revolute"
    with pytest.raises(ValueError, match="fully declared fixed"):
        _captured_tcp_transform(physical, configuration, source_path="captured-model")
    assert record["motion_config"]["tcp_transform"] == result


def test_outline_preview_rejects_hardware_before_observation(recovery, monkeypatch):
    from cais_spade_llm.recovery_framework import outline_gazebo_evidence

    capture = Mock(side_effect=AssertionError("must not observe hardware"))
    monkeypatch.setattr(outline_gazebo_evidence, "capture_observations", capture)
    recovery.runtime.setup["execution_mode"] = "physical"
    with pytest.raises(ValueError, match="requires simulation"):
        asyncio.run(outline_gazebo_evidence.prepare_gazebo_evidence(recovery, recovery.runtime))
    capture.assert_not_called()


def test_outline_preview_duplicate_reset_and_stop_are_owned(recovery, monkeypatch):
    from cais_spade_llm.recovery_framework import outline_gazebo_evidence

    async def run():
        await recovery._prepare()
        recovery._state.update(status="blocked", validation_contract_version=2)
        recovery._prepared["multi_turn_session_state"] = {"pending_outline_validation": {
            "response": {"next_transition": {"outline_id": "proposal"}}}}
        bridge = SimpleNamespace(system_running=True, _simulation_launch_key=lambda: "launch")
        async def dispatch(coroutine):
            return await coroutine
        bridge._run_on_agent_runtime = dispatch
        example = PartSlippageExample(bridge)
        example.source, example._offline = "offline", recovery
        example._runtime = SimpleNamespace(launch_identity="launch")
        started, ended = asyncio.Event(), asyncio.Event()
        async def preview(*args, cancel_event=None):
            started.set()
            try:
                while not cancel_event.is_set():
                    await asyncio.sleep(.01)
                await asyncio.sleep(.02)
                assert example.snapshot()["busy"]
                raise asyncio.CancelledError()
            finally:
                ended.set()
        monkeypatch.setattr(outline_gazebo_evidence, "prepare_gazebo_evidence", preview)
        await example.action("prepare_gazebo_evidence")
        await started.wait()
        with pytest.raises(ValueError, match="already running"):
            await example.action("prepare_gazebo_evidence")
        with pytest.raises(ValueError, match="already running"):
            await example.action("reset")
        await example.action("stop")
        assert ended.is_set()
        assert not example.snapshot()["busy"]
        assert example.snapshot()["status"] == "stopped"
    asyncio.run(run())


@pytest.mark.parametrize("mode", ["simulation", "physical"])
def test_pose_feasibility_plans_without_dispatch_only_in_simulation(mode):
    from cais_spade_llm.agents.resource_agent.robot_agent import RobotAgent

    pose = SimpleNamespace(position=SimpleNamespace(x=0., y=0., z=1.),
                           orientation=SimpleNamespace(x=0., y=0., z=0., w=1.))
    point = SimpleNamespace(positions=[0.] * 6, time_from_start=SimpleNamespace(sec=1, nanosec=0))
    trajectory = SimpleNamespace(joint_names=[str(i) for i in range(6)], points=[point])
    controller = SimpleNamespace(
        execution_mode=mode, _get_ee_pose=Mock(return_value=pose),
        _prepare_cartesian_motion=Mock(return_value=(SimpleNamespace(joint_trajectory=trajectory), "current targets")),
        move_cartesian=Mock(),
    )
    owner = SimpleNamespace(jid="recovery-resource-3@localhost", static_capabilities={},
                            execution_mode=mode, _controller=controller,
                            environment_runtime=SimpleNamespace(diagnostic_cca_bypass=True, setup=example_setup()))
    result = RobotAgent.check_recovery_physical_feasibility(
        owner, part_context={}, recovery_snapshot={},
        grounded_action={"function_name": "move_cartesian", "effect_scope": "resource_only",
                         "target": {"pose": {"x": .2, "y": .3, "z": 1.2}}},
    )
    controller.move_cartesian.assert_not_called()
    assert pose.position.x == 0.
    if mode == "simulation":
        assert result["allowed"] is True
        assert result["evidence"]["native_plan"]["command_sent"] is False
        controller._prepare_cartesian_motion.assert_called_once()
    else:
        assert result["feasibility_status"] == "NEEDS_CONTEXT"
        controller._prepare_cartesian_motion.assert_not_called()


@pytest.mark.parametrize("observed_x", [.2, .4])
def test_assembly_completion_requires_observed_configured_placement(monkeypatch, observed_x):
    from cais_spade_llm.recovery_framework.slippage_execution import _reconcile_completion

    part, rid = "KET4_Square_4mm", "ur5e-3"
    actor = SimpleNamespace(valuation={"resource_state": "picked", "held_part": part}, revision=0)
    context = SimpleNamespace(
        run_id="trial", resources={rid: actor},
        models={rid: {"state_variables": {"resource_location": {"domain": ["assembly_board-v1"]}}}},
        geometry={part: {"model_name": part, "target": "square_peg_4mm"}}, product_name="assembly_board-v1",
        part_tracker={part: {"state": "in_gripper", "location": rid, "processCompleted": []}},
        admission_lock=threading.RLock(), revision=0, transitions=[],
    )
    context.snapshot = lambda: {rid: deepcopy(actor.valuation)}
    owner = SimpleNamespace(agent_name=rid, _controller=SimpleNamespace(_attached_model=None),
                            _held_part=None, _current_state="placed")
    runtime = SimpleNamespace(context=context, agent=SimpleNamespace())
    monkeypatch.setattr("cais_spade_llm.recovery_framework.failure_effects.observe_robot",
                        AsyncMock(return_value={"x": .2, "y": .3, "z": 1.4}))
    monkeypatch.setattr("cais_spade_llm.recovery_framework.failure_effects._observe_part",
                        lambda *args: {"x": observed_x, "y": .3, "z": 1.})
    monkeypatch.setattr("cais_spade_llm.recovery_framework.workflow_execution._part_geometry",
                        lambda *args: {"target_origin_pose": {"x": .2, "y": .3, "z": 1.}})
    task = {"part_name": part, "outline_id": "RECOVERY_SEQ1", "macro_name": "place_part",
            "expected_end_state": {"part_state": "assembled", "resource_location": "assembly_board-v1"}}
    if observed_x == .4:
        with pytest.raises(ValueError, match="not observed"):
            asyncio.run(_reconcile_completion(runtime, owner, task, "native-task", {}))
        assert context.part_tracker[part]["state"] == "in_gripper"
        assert context.transitions == []
    else:
        asyncio.run(_reconcile_completion(runtime, owner, task, "native-task", {}))
        assert context.part_tracker[part]["processCompleted"] == [{"process": "assembly", "target": "square_peg_4mm"}]
        record = context.transitions[0]
        assert record["acknowledgement"]["evidence"] == "resource"
        assert record["safety_status"] == "skipped"


def test_pick_feasibility_uses_controller_target_extraction(monkeypatch):
    from cais_spade_llm.agents.resource_agent.robot_agent import RobotAgent

    raw = {"success": True, "part_name": "gear_small", "model_name": "gear_small",
           "tx": 0., "ty": .3, "tz": 1.025, "pick_z": 1.248, "travel_z": 1.5,
           "part_height": .01, "tcp_offset_z": -.218, "pick_tcp_z": 1.03,
           "start_x": .1, "start_y": .2, "start_z": 1.4}
    pose = SimpleNamespace(position=SimpleNamespace(x=.1, y=.2, z=1.4),
                           orientation=SimpleNamespace(x=0., y=0., z=0., w=1.))
    point = SimpleNamespace(positions=[0.] * 6, time_from_start=SimpleNamespace(sec=1, nanosec=0))
    controller = SimpleNamespace(
        execution_mode="simulation", compute_pick_targets=Mock(return_value=raw),
        _get_ee_pose=Mock(return_value=pose),
        _prepare_cartesian_motion=Mock(return_value=(SimpleNamespace(joint_trajectory=SimpleNamespace(
            joint_names=[str(i) for i in range(6)], points=[point])), "current targets")),
    )
    owner = SimpleNamespace(jid="recovery-resource-4@localhost", static_capabilities={},
                            execution_mode="simulation", _controller=controller,
                            environment_runtime=SimpleNamespace(diagnostic_cca_bypass=True,
                                                                setup=example_setup(), context=object()))
    monkeypatch.setattr("cais_spade_llm.recovery_framework.workflow_execution._part_geometry", lambda *args: {})
    result = RobotAgent.check_recovery_physical_feasibility(
        owner, part_context={}, recovery_snapshot={"held_part": None, "gripper_state": "open"},
        grounded_action={"function_name": "retrieve_part", "part_name": "gear_small",
                         "effect_scope": "resource_and_part",
                         "preconditions": {"part": {"requires_acquisition": True},
                                           "source_ref": {"location": "observed_pose", "pose": {"x": 0., "y": .3, "z": 1.025}}}},
    )
    assert result["allowed"] is True
    args = controller._prepare_cartesian_motion.call_args
    assert args.args[0].position.z == 1.248
    assert args.kwargs["waypoints"][0].position.z == 1.5


def test_regeneration_invalidates_downstream_and_keeps_previous_records(recovery):
    async def run():
        recovery.begin({"injection_status": "completed"})
        await recovery._task
        original = recovery.snapshot()["artifact_directory"]
        recovery._state["outline_mode"] = "manual"
        recovery.action("regenerate_outline", feedback="Keep the observed custody")
        with pytest.raises(ValueError, match="does not authorize execution"):
            recovery.action("execute")
        await recovery._task
        state = recovery.snapshot()
        assert state["status"] == "outline_review" and state["primitives"] == []
        assert recovery._proposal == {}
        assert state["artifact_directory"] != original
        assert (Path(original) / "proposal.json").is_file()
        assert recovery._prepared["recovery_feedback"] == "Keep the observed custody"
        recovery.action("continue_outline")
        await recovery._task
        recovery.action("regenerate_primitives", feedback="Review the destination")
        await recovery._task
        assert recovery.snapshot()["status"] == "ready_to_execute"
        assert recovery._prepared["recovery_feedback"] == "Review the destination"
        assert recovery.snapshot()["execution_authorized"] is False
    asyncio.run(run())


def test_stop_cancels_generation_and_rejects_stale_actions(recovery, monkeypatch):
    started = asyncio.Event()

    async def pending(stage):
        started.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(recovery, "_run_stage", pending)

    async def run():
        recovery.begin({"injection_status": "completed"})
        await started.wait()
        recovery.stop()
        await asyncio.gather(recovery._task, return_exceptions=True)
        assert recovery.snapshot()["status"] == "stopped"
        with pytest.raises(ValueError):
            recovery.action("execute")
    asyncio.run(run())


def test_composition_keeps_outline_evidence_out_of_active_checkpoint(recovery):
    from cais_spade_llm.recovery_framework.primitive_composition import start_composition_attempt

    async def prepare():
        await recovery._prepare()
        recovery.offline = True
        recovery._prepared["physical_evidence"] = {"recorded": [1, 2, 3]}
        state = recovery._prepared["multi_turn_session_state"]
        state.update(projected_physical_evidence={"outline_end": [4, 5, 6]},
                     outline_monitor_context={"accepted": [7, 8, 9]},
                     base_symbolic_resources={"owner": {"mode": "failed"}})
        recovery._outline_request = deepcopy(recovery._prepared)
        original = deepcopy(recovery._outline_request)
        await start_composition_attempt(recovery)
        active = recovery._prepared["multi_turn_session_state"]
        assert "projected_physical_evidence" not in active
        assert "outline_monitor_context" not in active
        assert active["base_symbolic_resources"] == state["base_symbolic_resources"]
        assert recovery._prepared["physical_evidence"] == original["physical_evidence"]
        assert recovery._outline_request == original
        saved = Path(recovery.snapshot()["artifact_directory"]) / "outline_request.json"
        assert json.loads(saved.read_text()) == original

    asyncio.run(prepare())


def test_stop_during_composition_preparation_does_not_publish_worker_state(recovery, monkeypatch):
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes import multi_turn

    started, release = threading.Event(), threading.Event()

    def preview(prepared, state):
        assert state["turns"] == []
        started.set()
        assert release.wait(5)
        return {}, "Not sent"

    monkeypatch.setattr(multi_turn, "_build_phase_prompt", preview)

    async def run():
        await recovery._prepare()
        recovery.offline = True
        recovery._prepared["multi_turn_session_state"]["turns"] = [{"phase": "outline", "validation": "preserved"}]
        recovery._outline_request = deepcopy(recovery._prepared)
        original = deepcopy(recovery._outline_request)
        prepared = recovery._prepared
        recovery._state.update(status="outline_review", primitive_mode="manual")
        recovery.action("continue_outline")
        try:
            assert await asyncio.to_thread(started.wait, 5)
            assert recovery.snapshot()["primitive_status"] == "preparing"
            recovery.stop()
        finally:
            release.set()
        await asyncio.gather(recovery._task, return_exceptions=True)
        assert recovery.snapshot()["status"] == "stopped"
        assert recovery._prepared is prepared
        assert recovery._outline_request == original
        assert recovery.snapshot()["execution_authorized"] is False

    asyncio.run(run())


def test_context_change_blocks_execution_and_generation_errors_are_retained(recovery, monkeypatch):
    execute = AsyncMock()
    monkeypatch.setattr("cais_spade_llm.recovery_framework.slippage_execution.execute_generated_recovery", execute)

    async def run():
        recovery.begin({"injection_status": "completed"})
        await recovery._task
        recovery._state.pop("composition_validation_version")
        recovery._proposal.pop("composition_validation_version", None)
        recovery.runtime.context.revision += 1
        recovery.action("execute")
        await recovery._task
        assert recovery.snapshot()["status"] == "blocked"
        execute.assert_not_called()
        monkeypatch.setattr(recovery, "_run_stage", AsyncMock(side_effect=ValueError("captured error")))
        recovery.action("regenerate_outline")
        await recovery._task
        assert recovery.snapshot()["message"] == "ValueError: captured error"
    asyncio.run(run())


def test_preparation_attribute_error_retains_observations_and_allows_retry(recovery, monkeypatch):
    previous = recovery._prepare
    monkeypatch.setattr(recovery, "_prepare", AsyncMock(side_effect=AttributeError("missing resource field")))

    async def run():
        recovery.begin({"injection_status": "completed", "checkpoint": "during_place_lowering"})
        with pytest.raises(AttributeError):
            await recovery._task
        await asyncio.sleep(0)
        state = recovery.snapshot()
        assert state["status"] == "blocked"
        assert "missing resource field" in state["message"]
        assert json.loads((recovery.path / "observations.json").read_text())["failure"]["checkpoint"] == "during_place_lowering"
        assert state["observations"]["failure"]["checkpoint"] == "during_place_lowering"
        monkeypatch.setattr(recovery, "_prepare", previous)
        recovery.action("prepare")
        await recovery._task
        assert recovery.snapshot()["status"] == "ready_to_execute"
    asyncio.run(run())


def test_malformed_outline_type_uses_preparation_error_without_provider_call(recovery, monkeypatch):
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes import multi_turn
    from cais_spade_llm.agents.shared_information import llm_agent

    previous = recovery._prepare
    provider = Mock(side_effect=AssertionError("Preparation must not call the provider"))
    monkeypatch.setattr(llm_agent._client.chat.completions, "create", provider)
    captured = []

    async def prepare(feedback=""):
        await previous(feedback)
        captured.append((recovery.path / "observations.json").read_bytes())
        multi_turn._get_response_schema("outline", {
            "validation_contract_version": 2, "outline_mode": "incremental_validated",
            "recovery_des_models": {"recovery-resource-6@localhost": {"state_variables": {
                "active_job": {"type": None, "domain": ["environment_23"]}}}},
        })

    monkeypatch.setattr(recovery, "_prepare", prepare)

    async def run():
        recovery.begin({"injection_status": "completed", "checkpoint": "during_processing"})
        await recovery._task
        state = recovery.snapshot()
        assert state["status"] == "blocked" and state["outline_status"] == "failed"
        assert state["message"] == "ValueError: Invalid declared type for outline field 'active_job': None"
        assert (recovery.path / "observations.json").read_bytes() == captured[0]
        error = json.loads((Path(state["artifact_directory"]) / "stage_error.json").read_text())
        assert error["error"] == state["message"]
        provider.assert_not_called()

    asyncio.run(run())


def test_preparation_exception_without_message_retains_its_type(recovery, monkeypatch):
    monkeypatch.setattr(recovery, "_prepare", AsyncMock(side_effect=NotImplementedError()))

    async def run():
        recovery.begin({"injection_status": "completed", "checkpoint": "during_processing_halfway"})
        await recovery._task
        assert recovery.snapshot()["status"] == "blocked"
        assert recovery.snapshot()["message"] == "NotImplementedError: "
        assert json.loads((recovery.path / "observations.json").read_text())["failure"]["checkpoint"] == "during_processing_halfway"

    asyncio.run(run())


@pytest.mark.parametrize("scenario", ["Part slippage", "Conveyor breakdown", "ur5e-1 breakdown", "Machining breakdown during part processing"])
def test_selected_failure_setup_retains_simulation_only_policy(scenario):
    setup = example_setup(scenario=scenario, safety_files=["recorded-for-later"])
    assert setup["failure_scenario"]["scenario"] == scenario
    assert diagnostic_setup(setup)
    assert setup["selected_safety_file"] == "__NONE__"
    assert setup["selected_safety_files"] == ["recorded-for-later"]
    if scenario == "Machining breakdown during part processing":
        assert {"M2", "ur5e-2"}.issubset(setup["permitted_resources"])
        assert {"M2", "ur5e-2"}.issubset(setup["recovery_permitted_resources"])
    elif scenario != "Part slippage":
        assert not {"M2", "ur5e-2"}.intersection(setup["permitted_resources"])
        assert {"M2", "ur5e-2"}.issubset(setup["recovery_permitted_resources"])
    assert setup["nominal_permitted_resources"] == setup["permitted_resources"]
    setup["execution_mode"] = "physical"
    assert not diagnostic_setup(setup)


def test_machine_safety_defaults_reach_ui_state_without_overwriting_saved_selections():
    from cais_spade_llm.recovery_framework.recovery_case_safety import default_safety_files

    scenario = "Machining breakdown during part processing"
    expected = default_safety_files(scenario)
    example = PartSlippageExample(SimpleNamespace(system_running=False))
    selected = example.set_options(source="gazebo", scenario=scenario, safety_files=example.safety_files)
    assert selected["selected_safety_files"] == expected
    assert example_setup(scenario=scenario)["selected_safety_files"] == expected
    selected = example.set_options(source="offline", scenario=scenario, safety_files=expected)
    assert selected["selected_safety_files"] == expected
    assert selected["outline_mode"] == selected["primitive_mode"] == "manual"
    old = ["cais_spade_llm/specification/safety/safety_none.txt"]
    selected = example.set_options(source="offline", scenario=scenario, safety_files=old)
    assert selected["selected_safety_files"] == old
    assert example.snapshot()["selected_safety_files"] == old
    assert example_setup(scenario=scenario, safety_files=old)["selected_safety_files"] == old
    assert example_setup(scenario=scenario, safety_files=[])["selected_safety_files"] == []


def test_part_slippage_safety_defaults_preserve_manual_choices_and_deferred_policy():
    from cais_spade_llm.recovery_framework import ROOT

    expected = [
        "cais_spade_llm/specification/safety/safety_assembly_board-v1_mutex.txt",
        "cais_spade_llm/specification/safety/safety_assembly_board-v1_gear_small_before_KET4_Square_4mm.txt",
    ]
    example = PartSlippageExample(SimpleNamespace(system_running=False))
    assert example.snapshot()["selected_safety_files"] == expected
    setup = example_setup()
    assert setup["selected_safety_files"] == expected
    assert setup["selected_safety_file"] == "__NONE__" and diagnostic_setup(setup)
    assert example_setup(safety_files=[])["selected_safety_files"] == []
    from cais_spade_llm.recovery_framework.recovery_case_safety import default_safety_files
    assert example_setup(scenario="Conveyor breakdown")["selected_safety_files"] == default_safety_files("Conveyor breakdown")

    example.set_options(source="gazebo", scenario="Part slippage", safety_files=[])
    assert example.snapshot()["selected_safety_files"] == []
    example.set_options(source="offline", scenario="Part slippage", safety_files=[])
    assert example.snapshot()["selected_safety_files"] == []
    example.set_options(source="gazebo", scenario="Conveyor breakdown", safety_files=[])
    selected = example.set_options(source="gazebo", scenario="Part slippage", safety_files=[expected[0]])
    assert selected["selected_safety_files"] == expected
    assert selected["status"] == "idle" and selected["safety_status"] == "skipped"
    absolute_mutex = str(ROOT / expected[0])
    example.set_options(source="gazebo", scenario="Conveyor breakdown", safety_files=[absolute_mutex])
    selected = example.set_options(source="gazebo", scenario="Part slippage", safety_files=[absolute_mutex])
    assert selected["selected_safety_files"] == [absolute_mutex, expected[1]]


@pytest.mark.parametrize("change", [
    {"execution_mode": "physical"}, {"execution_mode": "hardware"},
    {"diagnostic_cca_bypass": False}, {"diagnostic_cca_bypass": "true"},
    {"runtime_recovery_validation_policy": "validated"}, {"selected_safety_file": "safety.txt"},
    {"failure_scenario": {"scenario": "Robot breakdown"}},
])
def test_diagnostic_policy_requires_the_exact_simulation_settings(change):
    setup = example_setup()
    assert diagnostic_setup(setup)
    setup.update(change)
    assert not diagnostic_setup(setup)


def test_example_duplicate_start_reconnect_stop_and_reset(monkeypatch):
    bridge = SimpleNamespace(system_running=False, _starting=False, _stopping=False,
                             get_conveyor_fault=Mock(return_value={}), stop_system=AsyncMock(),
                             simulation_environment_running=Mock(return_value=False))
    example = example_for_bridge(bridge)
    started = asyncio.Event()

    async def start():
        started.set()
        while not example._cancelled:
            await asyncio.sleep(.001)

    monkeypatch.setattr(example, "_start", start)

    async def run():
        assert example.start()["status"] == "starting"
        with pytest.raises(ValueError, match="already started"):
            example.start()
        await started.wait()
        assert example_for_bridge(bridge) is example
        await example.action("stop")
        await example._task
        assert example.snapshot()["status"] == "stopped"
        await example.action("reset")
        await example._task
        assert example.snapshot()["status"] == "idle"
        assert example.snapshot()["outline_mode"] == example.snapshot()["primitive_mode"] == "auto"
    asyncio.run(run())


def test_saved_generations_and_exact_captures_cannot_cross_runs(tmp_path):
    from cais_spade_llm.agents.shared_information.llm_request_records import RequestRecords, capture_requests

    monitor = tmp_path / "monitor"
    for run_id in ("first", "second"):
        recovery = monitor / "environment_runs" / run_id / "recovery"
        stage = recovery / "generation_0001/recovery_outline"
        stage.mkdir(parents=True)
        (recovery / "run.json").write_text(json.dumps({"run_id": run_id, "status": "ready_to_execute"}))
        with capture_requests(directory=stage, run_id=run_id, stage="outline", turn=1):
            writer = RequestRecords({})
            writer.submit(lambda **payload: {"raw": run_id}, {"messages": [{"role": "user", "content": run_id}]}, tool_round=1, attempt=1)
    rows = list_examples(monitor / "debug")
    assert len(rows) == 2
    for row in rows:
        directory = row["stages"]["recovery_outline"]
        path = stage_records(directory, "recovery_outline")[0]
        captures = read_stage_record(directory, path)["captures"]
        assert {capture["record"]["run_id"] for capture in captures} == {row["path"].parents[1].name}
        assert {capture["record"]["kind"] for capture in captures} == {"provider_request", "provider_response"}


def test_robot_snapshot_uses_the_observed_environment_location():
    from cais_spade_llm.resources.robot.robot_profile import _robot_snapshot_builder

    owner = SimpleNamespace(environment_context=SimpleNamespace(valuation={
        "resource_location": "Buffer For Machined parts",
    }))
    assert _robot_snapshot_builder(owner)["current_location"] == "Buffer For Machined parts"
    owner.environment_context.valuation["resource_location"] = None
    assert _robot_snapshot_builder(owner)["current_location"] is None


def test_interrupted_assembly_precedes_cached_move_home(recovery):
    rid = "ur5e-3"
    recovery.runtime.jids = {rid: "recovery-resource-3@localhost"}
    def task(event, **parameters):
        return {"resource_id": rid, "event_id": event, "event_name": event, "parameters": parameters}

    place = task("place_approach", part_name="KET4_Square_4mm", destination_location="assembly_board-v1")
    recovery._failure = {
        "pending_tasks": [],
        "continuations": {
            "resource:ur5e-3": {"path": [task("move_home")]},
            "KET4_Square_4mm": {"path": [place, task("place_insert", part_name="KET4_Square_4mm")]},
        },
    }
    nodes = recovery._pending_nodes()
    assert [node["function_name"] for node in nodes] == ["place_approach", "place_insert", "move_home"]
    assert nodes[0]["params"] == place["parameters"]


def test_preparation_passes_observed_failure_and_declared_contracts_without_guidance(recovery, monkeypatch):
    from cais_spade_llm.recovery_framework import ROOT
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes import multi_turn
    from cais_spade_llm.recovery_framework.recovery_observations import capture_observations

    rid, part = "ur5e-4", "gear_small"
    context = recovery.runtime.context
    context.inputs = {"scene": json.loads((ROOT / "cais_spade_llm/initialization/recovery_framework_gazebo.json").read_text())}
    context.resources = {rid: SimpleNamespace(valuation={"resource_state": "failed", "held_part": None,
                                                        "resource_location": "3D Printing Station"})}
    context.part_tracker = {part: {"state": "misplaced", "location": None,
                                  "observed_pose": {"x": .0, "y": .3, "z": 1.025}}}
    context.initial_product_states = {part: {"location": "3D Printing Station"}}
    context.selected_parts = [part]
    recovery.runtime.jids = {rid: "recovery-resource-4@localhost"}
    recovery.runtime.resource_agents = [SimpleNamespace(agent_name=rid, jid=recovery.runtime.jids[rid],
                                                      executables={"place_approach": Mock(), "move_home": Mock()})]
    recovery.runtime.conveyor_fault = SimpleNamespace(configuration={
        "resource_id": rid, "part_name": part, "event_name": "place_insert",
    })
    recovery._failure = {"injection_status": "completed", "pending_tasks": [{"task_id": "environment_8", "resource_id": rid,
        "event_id": "place_approach", "event_name": "place_approach", "parameters": {"part_name": part}}]}
    recovery._observations = capture_observations(recovery.runtime, recovery._failure)
    prepare = AsyncMock(return_value={"recovery_session": {}, "recovery_debug": {}, "llm_input": {}, "ra_jid": recovery.runtime.jids[rid],
                                     "recovery_resources": {recovery.runtime.jids[rid]: {"recovery_snapshot": {}}}})
    recovery.agent.process_planner = SimpleNamespace(prepare_recovery_request=prepare,
                                                    _build_resource_search_state=Mock(return_value={}))
    monkeypatch.setattr(multi_turn, "build_multi_turn_session_seed", lambda value: {})
    asyncio.run(PartSlippageRecovery._prepare(recovery))
    captured = prepare.call_args.kwargs
    assert captured["recovery_feedback"] == ""
    assert captured["failure_context"]["failed_function_name"] == "place_approach"
    assert captured["failure_context"]["failure_context"]["affected_entities"][0]["state"] == "misplaced"
    assert captured["part_tracker"][part]["observed_pose"] == context.part_tracker[part]["observed_pose"]
    place = next(row for row in captured["tools_catalog"] if row["function"] == "place_approach")
    assert place["function_owner_agent"] == recovery.runtime.jids[rid]
    assert place["in_state"] == "picked"


@pytest.mark.parametrize("scenario", ["Conveyor breakdown", "ur5e-1 breakdown", "Machining breakdown during part processing"])
@pytest.mark.parametrize("mode", ["auto", "manual"])
def test_other_failures_compose_from_ready_contracts(recovery, scenario, mode):
    recovery.runtime.setup = example_setup(scenario=scenario, outline_mode=mode, primitive_mode="manual")
    recovery._state.update(scenario=scenario, outline_only=False, outline_mode=mode, primitive_mode="manual")

    async def run():
        recovery.begin({"injection_status": "completed", "part_name": "recorded-part"})
        await recovery._task
        if mode == "manual":
            assert recovery.snapshot()["status"] == "outline_ready"
            recovery.action("generate_outline")
            await recovery._task
            assert recovery.snapshot()["status"] == "outline_review"
            recovery.action("continue_outline")
            await recovery._task
        assert recovery.snapshot()["status"] == "primitive_ready"
        assert recovery.snapshot()["outline"] == _accepted_composition_inputs()["multi_turn_session_state"]["accepted_outline_prefix"]
        assert recovery._observations["scenario"] == scenario
        recovery.action("generate_primitives")
        await recovery._task
        assert recovery.snapshot()["status"] == "ready_to_execute"
        assert recovery.snapshot()["primitives"]
        assert recovery.snapshot()["execution_authorized"] is False
        with pytest.raises(ValueError, match="does not authorize execution"):
            recovery.action("execute")
        with pytest.raises(ValueError, match="does not authorize execution"):
            await recovery._execute()
    asyncio.run(run())


@pytest.mark.parametrize("scenario", ["Part slippage", "Conveyor breakdown", "ur5e-1 breakdown", "Machining breakdown during part processing"])
def test_failure_pause_preserves_recovery_but_operator_stop_cancels_it(scenario):
    from cais_spade_llm.recovery_framework.environment_runtime import EnvironmentRuntime

    recovery = SimpleNamespace(stop=Mock())
    reader = SimpleNamespace(close=Mock())
    runtime = SimpleNamespace(part_slippage_recovery=recovery, recovery_observation_preparation=SimpleNamespace(reader=reader),
        admission=None, _close_calculations=Mock(), resource_agents=[], active_executions=set(), outcome={},
        context=SimpleNamespace(admission_lock=threading.RLock(), pending_tasks={}, environment_model={}, exploration_models={}))
    EnvironmentRuntime.stop(runtime, scenario, preserve_recovery=True)
    assert runtime.stopped and runtime.context.environment_model["closed"] is True
    recovery.stop.assert_not_called()
    reader.close.assert_not_called()
    EnvironmentRuntime.stop(runtime, scenario)
    recovery.stop.assert_called_once_with(scenario)
    reader.close.assert_called_once()


@pytest.mark.parametrize("flag", ["recovery_checkpoint_capturing", "recovery_capture_frozen"])
def test_operator_stop_retains_checkpoint_reader_until_unfreeze(flag):
    from cais_spade_llm.recovery_framework.environment_runtime import EnvironmentRuntime

    reader = SimpleNamespace(close=Mock())
    recovery = SimpleNamespace(stop=Mock())
    runtime = SimpleNamespace(part_slippage_recovery=recovery,
        recovery_observation_preparation=SimpleNamespace(reader=reader),
        admission=None, _close_calculations=Mock(), resource_agents=[], active_executions=set(), outcome={},
        context=SimpleNamespace(admission_lock=threading.RLock(), pending_tasks={}, environment_model={}, exploration_models={}))
    setattr(runtime, flag, True)
    EnvironmentRuntime.stop(runtime, "Stopped by operator")
    recovery.stop.assert_called_once_with("Stopped by operator")
    reader.close.assert_not_called()
    assert runtime.recovery_observation_preparation.reader is reader


@pytest.mark.parametrize("scenario,rid", [
    ("Conveyor breakdown", "Conveyor"), ("ur5e-1 breakdown", "ur5e-1"),
    ("Machining breakdown during part processing", "M1"),
])
@pytest.mark.parametrize("checkpoint_completed", [False, True])
def test_nonrobot_preparation_preserves_owner_facts_and_actual_failure(recovery, monkeypatch, scenario, rid, checkpoint_completed):
    from cais_spade_llm.recovery_framework.recovery_observations import capture_observations, offline_preparation_missing

    part = "KET4_Square_4mm"
    recovery.runtime.setup = example_setup(scenario=scenario)
    recovery._state.update(scenario=scenario, outline_only=True)
    context = recovery.runtime.context
    context.inputs = {"scene": {"robots": []}}
    context.resources = {name: SimpleNamespace(valuation={"part_name": None, "belt_stopped": True}) for name in (rid, "KMR", "M2")}
    context.unavailable_resources = {rid}
    context.permitted_resources = [rid, "KMR"]
    context.part_tracker = {part: {"state": "ready", "location": "M1", "processCompleted": []}}
    context.selected_parts = [part]
    context.requirements = {part: [{"processesToComplete": [{"process": "trim", "result": "square"}]}]}
    recovery.runtime.jids = {name: name + "@localhost" for name in context.resources}
    owners = []
    for name, actor in context.resources.items():
        owners.append(SimpleNamespace(agent_name=name, jid=recovery.runtime.jids[name],
            get_recovery_snapshot=Mock(return_value=deepcopy(actor.valuation)),
            get_recovery_resource_evidence=Mock(return_value={"environment_valuation": {name: deepcopy(actor.valuation)}}),
            recovery_des_model=Mock(return_value={"events": [{"event_name": "declared_function", "parameter_bindings": {"part_name": {"type": "string"}}}]})))
    recovery.runtime.resource_agents = owners
    recovery.runtime.conveyor_fault = SimpleNamespace(configuration={"resource_id": rid})
    recovery._failure = {"injection_status": "completed", "part_name": part, "pending_tasks": [{
        "task_id": "recorded-task", "resource_id": rid, "event_id": "recorded-event", "event_name": "recorded_function",
        "parameters": {"part_name": part}}]}
    recovery._observations = capture_observations(recovery.runtime, recovery._failure, include_physical=False)
    if checkpoint_completed:
        recovery._failure.update(task_id="recorded-task", checkpoint="observed-checkpoint")
        recovery._observations["failure"] = deepcopy(recovery._failure)
        recovery._observations["acknowledgements"] = {"recorded-task": {
            "task_id": "recorded-task", "resource_id": rid, "event_name": "recorded_function",
            "status": "completed", "evidence": "resource", "run_id": recovery._observations["run_id"]}}
    prepared = {"recovery_session": {}, "recovery_debug": {}, "llm_input": {}, "ra_jid": recovery.runtime.jids[rid],
        "recovery_resources": {jid: {"recovery_snapshot": {}} for jid in recovery.runtime.jids.values()}}
    prepare = AsyncMock(return_value=prepared)
    recovery.agent.process_planner = SimpleNamespace(prepare_recovery_request=prepare, _build_resource_search_state=Mock(return_value={}))
    monkeypatch.setattr(recovery, "_install_prepared", Mock())
    asyncio.run(PartSlippageRecovery._prepare(recovery))
    args = prepare.call_args.kwargs
    assert args["failure_context"]["failed_function_name"] == ("" if checkpoint_completed else "recorded_function")
    assert args["failure_context"]["failed_task_id"] == ("" if checkpoint_completed else "recorded-task")
    if checkpoint_completed:
        checkpoint = args["failure_context"]["failure_context"]["observations"]["checkpoint_context"]
        assert checkpoint["name"] == "observed-checkpoint"
        assert checkpoint["completed_task"]["task_id"] == "recorded-task"
        assert checkpoint["completed_task"]["status"] == "completed"
    assert args["failure_context"]["failure_context"]["failure_mode"] == scenario
    assert args["goal_state"] == "" and args["recovery_feedback"] == ""
    assert set(args["recorded_resource_snapshots"]) == set(recovery.runtime.jids.values())
    assert args["recorded_resource_snapshots"][rid + "@localhost"]["part_name"] is None
    assert args["recorded_resource_snapshots"][rid + "@localhost"]["availability"] == "unavailable"
    assert args["recorded_resource_snapshots"]["M2@localhost"]["availability"] == "available"
    assert "held_part" not in args["resource_states"][rid + "@localhost"]
    assert {row["function_owner_agent"] for row in args["tools_catalog"]} == set(recovery.runtime.jids.values())
    assert recovery._observations["unavailable_resources"] == [rid]
    assert "M2" in recovery._observations["recovery_permitted_resources"]
    assert "M2" not in recovery._observations["nominal_permitted_resources"]
    assert recovery._observations["requirements"] == context.requirements
    assert not offline_preparation_missing(recovery._observations)
    recovery._failure["part_name"] = None
    recovery._observations.pop("prepared_recovery_request")
    with pytest.raises(ValueError, match="recorded affected part"):
        asyncio.run(PartSlippageRecovery._prepare(recovery))


def test_regeneration_records_operator_assistance_without_changing_capture(recovery):
    async def run():
        recovery.begin({"injection_status": "completed"})
        await recovery._task
        before = deepcopy(recovery._observations)
        recovery._state["status"] = "outline_review"
        recovery.action("regenerate_outline", feedback="Recorded operator feedback")
        await recovery._task
        assert recovery.snapshot()["operator_assistance"][-1]["feedback"] == "Recorded operator feedback"
        assert recovery._observations == before
    asyncio.run(run())


def test_selected_capture_must_match_the_failure_scenario(monkeypatch):
    example = PartSlippageExample(SimpleNamespace(system_running=False))
    monkeypatch.setattr("cais_spade_llm.recovery_framework.part_slippage_example.read_observations",
                        lambda _: {"scenario": "ur5e-1 breakdown"})
    with pytest.raises(ValueError, match="different failure scenario"):
        example.set_options(source="offline", scenario="Conveyor breakdown", safety_files=[], observation_path="capture")
    assert example.source == "gazebo" and not example.observation_path


@pytest.mark.parametrize("mode", ["simulation", "physical"])
def test_lift_witness_after_slippage_requires_controller_target_evidence(monkeypatch, mode):
    from cais_spade_llm.agents.resource_agent.robot_agent import RobotAgent

    observed_pose = {"x": .0, "y": .3, "z": 1.025}
    controller = SimpleNamespace(execution_mode=mode, compute_pick_targets=Mock(return_value={
        "success": True, "pick_z": 1.248, "travel_z": 1.298,
    }))
    owner = SimpleNamespace(
        execution_mode=mode, _controller=controller, _task_ctx={},
        environment_runtime=SimpleNamespace(diagnostic_cca_bypass=True, setup=example_setup(),
            context=SimpleNamespace(geometry={"gear_small": {"model_name": "gear_small"}})),
        check_recovery_primitive_feasibility=Mock(return_value={"allowed": True, "feasibility_status": "FEASIBLE"}),
    )
    monkeypatch.setattr("cais_spade_llm.recovery_framework.workflow_execution._part_geometry", lambda *args: {})
    model = {
        "state_variables": {"resource_state": {"domain": ["idle", "picked"]}},
        "events": [{"event_name": "pick_grasp", "updates": {"resource_state": {"set": "picked"}},
                    "primitive_support": [{"primitive": "move_relative", "params": {
                        "dz": {"$state": "_task_ctx", "path": ["travel_z"]},
                    }}]}],
        "primitive_catalog": [{"name": "move_relative", "params": {"dz": {"type": "number"}},
                               "required_params": ["dz"]}],
    }
    result = RobotAgent.validate_recovery_primitive_support(
        owner, task={"part_name": "gear_small", "expected_start_state": {"resource_state": "idle"},
                     "expected_end_state": {"resource_state": "picked"}},
        recovery_des_model=model, recovery_snapshot={"current_pose": {"x": .029, "y": .145, "z": 1.289}},
        physical_input={"part_context": {"observed_pose": observed_pose}},
    )
    if mode == "simulation":
        assert result["allowed"] is True
        assert result["primitive_support"]["parameter_evidence"]["dz"] == pytest.approx(.009)
        assert result["primitive_support"]["pick_target_evidence"]["success"] is True
        assert controller.compute_pick_targets.call_args.kwargs["target_pose"] == observed_pose
        assert owner._task_ctx == {}
    else:
        assert result["feasibility_status"] == "NEEDS_CONTEXT"
        controller.compute_pick_targets.assert_not_called()


@pytest.mark.parametrize("owner_mode,controller_mode", [("physical", "physical"), ("simulation", "physical")])
def test_execution_rejects_hardware_even_when_the_session_requests_bypass(recovery, owner_mode, controller_mode):
    from cais_spade_llm.recovery_framework.slippage_execution import execute_generated_recovery

    owner = SimpleNamespace(jid="recovery-resource-4@localhost", agent_name="ur5e-4",
                            execution_mode=owner_mode, _controller=SimpleNamespace(execution_mode=controller_mode),
                            execute_recovery_macro=AsyncMock())
    recovery.runtime.resource_agents = [owner]
    recovery.runtime.context.pending_tasks = {}
    recovery.runtime.context.reservations = {}
    recovery.runtime.context.permitted_resources = ["ur5e-4"]
    recovery._proposal = {"macro_tasks": [{"outline_id": "step1", "resource_jid": str(owner.jid)}]}
    with pytest.raises(ValueError, match="prepared simulation controller"):
        asyncio.run(execute_generated_recovery(recovery))
    owner.execute_recovery_macro.assert_not_called()


def test_stop_restores_the_previous_bridge_settings():
    bridge = SimpleNamespace(system_running=False, _starting=False, _stopping=False,
                             execution_mode="physical", selected_safety_file="later-safety.txt",
                             runtime_recovery_validation_policy="validated",
                             get_conveyor_fault=Mock(return_value={}), stop_system=AsyncMock())
    example = PartSlippageExample(bridge)
    async def run():
        example._saved_settings = {"execution_mode": bridge.execution_mode,
                                   "selected_safety_file": bridge.selected_safety_file,
                                   "runtime_recovery_validation_policy": bridge.runtime_recovery_validation_policy}
        bridge.execution_mode = "simulation"
        bridge.selected_safety_file = "__NONE__"
        bridge.runtime_recovery_validation_policy = "no_validation"
        await example.action("stop")
        assert bridge.execution_mode == "physical"
        assert bridge.selected_safety_file == "later-safety.txt"
        assert bridge.runtime_recovery_validation_policy == "validated"
    asyncio.run(run())


def test_feedback_is_in_the_sent_outline_prompt():
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes.multi_turn import _build_phase_prompt

    payload, prompt = _build_phase_prompt({"recovery_feedback": "Keep gear_small custody observed"},
                                          {"current_phase": "grounding"})
    assert payload["recovery_feedback"] == "Keep gear_small custody observed"
    assert "Operator feedback:\nKeep gear_small custody observed" in prompt


def test_safety_prompt_snapshot_is_per_attempt_and_does_not_enable_validation(recovery, tmp_path):
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes.multi_turn import (
        _build_phase_prompt,
    )

    path = tmp_path / "selected.txt"
    specification = {"id": "SAFE_shared_area_mutex", "requirement": "At most one resource may occupy assembly_board-v1."}
    path.write_text(json.dumps({"catalog": {"specifications": [specification]}}))
    original_bytes = path.read_bytes()
    recovery.runtime.setup["selected_safety_files"] = [str(path)]
    recovery._observations = {"captured_at": 123, "run_id": "source-run", "context_revision": 4}
    observations_before = deepcopy(recovery._observations)
    prepared = {"recovery_session": {}, "recovery_debug": {}, "llm_input": {"loaded_safety_rules": []},
                "recovery_safety_context": {}}
    directory = recovery._new_artifact_directory()
    recovery._install_prepared(deepcopy(prepared), directory, "")
    installed = deepcopy(recovery._prepared)
    record = json.loads((directory / "inputs.json").read_text())["prepared_recovery_request"]
    snapshot = [{**specification, "source_path": str(path), "source_sha256": hashlib.sha256(original_bytes).hexdigest()}]
    assert record["prompt_safety_rules"] == snapshot
    assert json.loads((directory / "prompt_safety_rules.json").read_text()) == {
        "prompt_safety_rules": snapshot, "safety_status": "skipped"}
    assert record["llm_input"]["loaded_safety_rules"] == []
    assert record["recovery_safety_context"] == {}
    assert recovery.snapshot()["safety_status"] == "skipped"
    assert diagnostic_setup(recovery.runtime.setup)
    assert recovery._observations == observations_before
    assert path.read_bytes() == original_bytes

    path.write_text(json.dumps({"catalog": {"specifications": [{**specification, "requirement": "Changed requirement"}]}}))
    _, frozen_prompt = _build_phase_prompt(installed, {"current_phase": "outline"})
    assert specification["requirement"] in frozen_prompt
    assert "Changed requirement" not in frozen_prompt
    recovery._install_prepared(deepcopy(prepared), recovery._new_artifact_directory(), "")
    assert recovery._prepared["prompt_safety_rules"][0]["requirement"] == "Changed requirement"
    recovery.runtime.setup["selected_safety_files"] = []
    recovery._install_prepared(deepcopy(prepared), recovery._new_artifact_directory(), "")
    assert recovery._prepared["prompt_safety_rules"] == []
    assert json.loads((directory / "inputs.json").read_text())["prepared_recovery_request"] == record


@pytest.mark.parametrize("payload", [None, {}, {"catalog": {"specifications": []}},
                                     {"catalog": {"specifications": [{"id": "SAFE_shared_area_mutex"}]}}])
def test_invalid_selected_safety_specification_keeps_observations(recovery, tmp_path, payload):
    path = tmp_path / "selected.txt"
    if payload is not None:
        path.write_text(json.dumps(payload))
    recovery.runtime.setup["selected_safety_files"] = [str(path)]
    recovery._observations = {"captured_at": 123, "run_id": "source-run"}
    before = deepcopy(recovery._observations)
    with pytest.raises(ValueError, match="NEEDS_CONTEXT:.*selected safety specification"):
        recovery._install_prepared({"recovery_session": {}, "recovery_debug": {}}, recovery._new_artifact_directory(), "")
    assert recovery._observations == before
    assert recovery.snapshot()["safety_status"] == "skipped"


def test_outline_preparation_records_exact_rule_evidence_needs_without_claiming_pass(recovery):
    from cais_spade_llm.recovery_framework.part_slippage_example import _PART_SLIPPAGE_SAFETY_FILES

    recovery.runtime.setup["selected_safety_files"] = list(_PART_SLIPPAGE_SAFETY_FILES)
    recovery._observations = {"captured_at": 123, "run_id": "source-run", "context_revision": 4}
    original = deepcopy(recovery._observations)
    directory = recovery._new_artifact_directory()
    recovery._install_prepared({"recovery_session": {}, "recovery_debug": {}}, directory, "")
    record = json.loads((directory / "outline_evidence_requirements.json").read_text())
    rules = {row["id"]: row for row in record["rules"]}
    assert set(rules) == {"SAFE_shared_area_mutex", "SAFE_gear_small_before_KET4_Square_4mm"}
    assert rules["SAFE_shared_area_mutex"]["aps"][0]["required_fields"] == [
        "geometry.resources", "geometry.regions", "snapshot.resources", "resource_models", "geometry.parts", "snapshot.parts"]
    assert "snapshot.parts.processCompleted_evidence" in rules["SAFE_gear_small_before_KET4_Square_4mm"]["aps"][1]["required_fields"]
    assert all(ap["status"] == "evidence_required" for row in rules.values() for ap in row["aps"])
    assert record["execution_authorized"] is False
    assert recovery.snapshot()["outline_safety_status"] == "pending"
    assert recovery._observations == original


@pytest.mark.parametrize("offline", [False, True])
def test_new_attempt_pins_primitive_support_semantics_without_changing_capture(recovery, offline):
    recovery.offline = offline
    recovery.runtime.setup["selected_safety_files"] = []
    recovery._observations = {"run_id": "captured-run", "context_revision": 4}
    captured = deepcopy(recovery._observations)
    prepared = {"validation_contract_version": 2, "recovery_session": {}, "recovery_debug": {},
                "multi_turn_session_state": {"turn_index": 7},
                "physical_evidence": {"physical_preview_contract_version": 2}}
    original = deepcopy(prepared)
    directory = recovery._new_artifact_directory()
    recovery._install_prepared(deepcopy(prepared), directory, "")
    record = json.loads((directory / "inputs.json").read_text())["prepared_recovery_request"]
    assert record["primitive_support_semantics"] == "successor_primitive_union"
    assert record["multi_turn_session_seed"]["primitive_support_semantics"] == "successor_primitive_union"
    assert record["validation_contract_version"] == 2
    assert record["physical_evidence"]["physical_preview_contract_version"] == 7
    assert record["physical_evidence"]["owner_projection_clock_version"] == 1
    assert record["public_safety_context_version"] == 1
    assert record["multi_turn_session_seed"]["public_safety_context_version"] == 1
    assert "multi_turn_session_state" not in record
    assert recovery.snapshot()["primitive_support_semantics"] == "successor_primitive_union"
    assert recovery.snapshot()["execution_authorized"] is False
    assert recovery._observations == captured
    assert prepared == original
    with pytest.raises(ValueError, match="does not authorize execution"):
        recovery.action("execute")


@pytest.mark.parametrize("version", [None, 1, 2, 3, 4, 5, 6, 7])
def test_restored_attempt_keeps_recorded_physical_preview_contract_version(recovery, version):
    saved = {"validation_contract_version": 2, "physical_evidence": {"recorded": True},
             "multi_turn_session_state": {"turn_index": 3}}
    if version is not None:
        saved["physical_evidence"]["physical_preview_contract_version"] = version
    before = deepcopy(saved)
    recovery._install_prepared(saved, recovery._new_artifact_directory(), "", new_attempt=False)
    assert recovery._prepared == before
    assert recovery._prepared["physical_evidence"].get("physical_preview_contract_version") == version


@pytest.mark.parametrize("version", [None, 0, 1])
def test_restored_attempt_keeps_recorded_public_safety_context_version(recovery, version):
    saved = {"validation_contract_version": 2,
             "multi_turn_session_seed": {"turn_index": 1},
             "multi_turn_session_state": {"turn_index": 3}}
    if version is not None:
        saved["public_safety_context_version"] = version
        saved["multi_turn_session_seed"]["public_safety_context_version"] = version
        saved["multi_turn_session_state"]["public_safety_context_version"] = version
    before = deepcopy(saved)
    recovery._install_prepared(saved, recovery._new_artifact_directory(), "", new_attempt=False)
    assert recovery._prepared == before
    assert recovery._prepared.get("public_safety_context_version") == version


@pytest.mark.parametrize("conflict", [False, True])
def test_new_attempt_records_owner_capabilities_before_scene_binding(recovery, conflict):
    recovery.runtime.setup["selected_safety_files"] = []
    capabilities = {"gripper_reach": {"frame": "world", "max_xy_radius_m": 0.7}}
    physical = {"jids": {"ur5e-3": "owner@localhost"}, "resource_models": {"ur5e-3": {}}}
    if conflict:
        physical["resource_models"]["ur5e-3"]["static_capabilities"] = {"gripper_reach": None}
    recovery._observations = {"physical_evidence": physical}
    before = deepcopy(recovery._observations)
    prepared = {"recovery_session": {}, "recovery_debug": {},
                "recovery_resources": {"owner@localhost": {"static_capabilities": capabilities}}}
    directory = recovery._new_artifact_directory()
    if conflict:
        with pytest.raises(ValueError, match="static_capabilities conflict"):
            recovery._install_prepared(prepared, directory, "")
    else:
        recovery._install_prepared(prepared, directory, "")
        captured = prepared["physical_evidence"]["resource_models"]["ur5e-3"]["static_capabilities"]
        assert captured == capabilities
        capabilities["gripper_reach"]["max_xy_radius_m"] = 1
        assert captured["gripper_reach"]["max_xy_radius_m"] == 0.7
    assert recovery._observations == before


@pytest.mark.parametrize("marked", [False, True])
def test_session_seed_and_resume_preserve_attempt_primitive_support_semantics(marked):
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes import multi_turn

    prepared = {"validation_contract_version": 2, "recovery_session": {}, "recovery_debug": {}}
    if marked:
        prepared["primitive_support_semantics"] = "successor_primitive_union"
    seed = multi_turn.build_multi_turn_session_seed(prepared)
    assert ("primitive_support_semantics" in seed) is marked
    if marked:
        assert seed["primitive_support_semantics"] == "successor_primitive_union"
    seed.update(max_turns=0, status="paused_after_outline_turn")
    # A resumed session owns its marker even if a newer preparation is supplied.
    prepared["primitive_support_semantics"] = "successor_primitive_union"
    llm = AsyncMock(side_effect=AssertionError("A resumed zero-turn session must not call the LLM"))
    planner = SimpleNamespace(product_agent=SimpleNamespace(ask_llm_structured=llm))
    asyncio.run(multi_turn.execute_multi_turn_recovery(planner, prepared, session_state=seed))
    assert ("primitive_support_semantics" in prepared["multi_turn_session_state"]) is marked
    assert prepared["multi_turn_session_state"]["validation_contract_version"] == 2
    llm.assert_not_called()


@pytest.mark.parametrize("resumed,turn_index,max_turns", [(False, 0, 1), (True, 4, 5),
                                                        (False, 0, 0), (True, 4, 4)])
def test_generation_debug_sync_covers_start_resume_and_exhausted_budgets(
    resumed, turn_index, max_turns, monkeypatch,
):
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes import multi_turn

    seed = {"current_phase": "outline", "turn_index": turn_index, "max_turns": max_turns,
        "validation_contract_version": 2, "status": "paused_after_outline_turn", "facts": {"pose": [0, 1, 2]}}
    original = deepcopy(seed)
    prepared = {"multi_turn_session_seed": seed, "recovery_debug": {"retained": {"capture": "source"}}}
    saved, progress = [], []

    async def report(*, session_state, status_label, **kwargs):
        # Every observable progress callback already has detached current debug
        # snapshots, including the first callback before any provider request.
        if status_label == "running":
            assert prepared["multi_turn_session_state"]["turn_index"] == turn_index + 1
            assert prepared["recovery_debug"]["multi_turn_session"] == saved[-1]["multi_turn_session"]
        session_state["facts"]["pose"].append("callback mutation")
        progress.append(status_label)

    async def ask(*args, **kwargs):
        assert saved[-1]["multi_turn_session"]["facts"]["pose"] == [0, 1, 2]
        return {"thought": "Recorded facts"}

    async def recorded(call, **kwargs):
        return await call()

    async def handle(*, session_state, **kwargs):
        assert session_state["facts"]["pose"] == [0, 1, 2]
        session_state["facts"]["pose"].append("derived")
        session_state["status"] = "paused_after_outline_turn"
        return "need_next_task", {}

    llm = AsyncMock(side_effect=ask)
    planner = SimpleNamespace(product_agent=SimpleNamespace(ask_llm_structured=llm,
        report_runtime_recovery_turn_progress=report), _set_last_recovery_debug=lambda value: saved.append(deepcopy(value)))
    monkeypatch.setattr(multi_turn, "_build_phase_prompt", lambda *args: ({}, "Recorded facts"))
    monkeypatch.setattr(multi_turn, "recorded_structured_call", recorded)
    monkeypatch.setattr(multi_turn, "_write_per_turn_artifact", lambda *args: None)
    monkeypatch.setattr(multi_turn, "_artifact_response_payload", lambda **kwargs: deepcopy(kwargs["parsed_response"]))
    monkeypatch.setitem(multi_turn._PHASE_HANDLERS, "outline", handle)
    asyncio.run(multi_turn.execute_multi_turn_recovery(planner, prepared, session_state=seed if resumed else None))
    assert seed == original
    current = prepared["multi_turn_session_state"]
    if turn_index < max_turns:
        assert current["status"] == "paused_after_outline_turn"
        assert current["facts"]["pose"] == [0, 1, 2, "derived"]
        assert [row["multi_turn_session"]["turn_index"] for row in saved] == [turn_index + 1] * 2
        assert progress == ["running", "waiting_for_llm", "response_received", "decision"]
        llm.assert_awaited_once()
        assert saved[0]["multi_turn_session"]["facts"]["pose"] == [0, 1, 2]
    else:
        assert current["status"] == "des_turn_limit"
        assert len(saved) == 1 and saved[0]["status"] == "des_turn_limit"
        assert progress == []
        llm.assert_not_called()
    debug = prepared["recovery_debug"]["multi_turn_session"]
    assert debug == current and saved[-1]["multi_turn_session"] == current
    current["facts"]["pose"].append("caller mutation")
    assert "caller mutation" not in debug["facts"]["pose"]
    debug["facts"]["pose"].append("debug mutation")
    assert "debug mutation" not in saved[-1]["multi_turn_session"]["facts"]["pose"]


def test_generation_provider_failure_preserves_current_detached_debug(monkeypatch):
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes import multi_turn

    seed = {"current_phase": "outline", "turn_index": 4, "max_turns": 5,
        "validation_contract_version": 2, "facts": {"pose": [0, 1, 2]}}
    original = deepcopy(seed)
    prepared, saved = {"recovery_debug": {}}, []

    async def recorded(*args, **kwargs):
        assert prepared["multi_turn_session_state"]["turn_index"] == 5
        assert saved[-1]["multi_turn_session"]["turn_index"] == 5
        raise RuntimeError("Provider unavailable")

    planner = SimpleNamespace(product_agent=SimpleNamespace(ask_llm_structured=AsyncMock()),
        _set_last_recovery_debug=lambda value: saved.append(deepcopy(value)))
    monkeypatch.setattr(multi_turn, "_build_phase_prompt", lambda *args: ({}, "Recorded facts"))
    monkeypatch.setattr(multi_turn, "recorded_structured_call", recorded)
    with pytest.raises(RuntimeError, match="Provider unavailable"):
        asyncio.run(multi_turn.execute_multi_turn_recovery(planner, prepared, session_state=seed))
    assert len(saved) == 1 and seed == original
    prepared["multi_turn_session_state"]["facts"]["pose"].append("mutation")
    assert prepared["recovery_debug"]["multi_turn_session"]["facts"]["pose"] == [0, 1, 2]
    assert saved[0]["multi_turn_session"]["facts"]["pose"] == [0, 1, 2]


@pytest.mark.parametrize("next_failure", [False, True])
def test_successful_context_retrieval_publishes_once_before_next_await(monkeypatch, next_failure):
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes import multi_turn

    seed = {"current_phase": "outline", "turn_index": 0, "max_turns": 3,
        "validation_contract_version": 2, "status": "running", "facts": {"pose": [0, 1, 2]}}
    prepared = {"recovery_debug": {}}
    published, artifacts = [], []

    async def progress(*, session_state, status_label, **kwargs):
        if status_label == "running":
            current = prepared["multi_turn_session_state"]
            assert current == session_state
            assert prepared["recovery_debug"]["multi_turn_session"] == current
            assert published[-1]["multi_turn_session"] == current
            assert session_state is not current
            if current["turn_index"] == 2:
                assert current["facts"]["pose"] == [0, 1, 2, "retrieved"]
                assert current["turns"][0]["request_artifact_path"] == "recorded-request"
        session_state["facts"]["pose"].append("callback mutation")

    async def recorded(*args, **kwargs):
        if next_failure and prepared["multi_turn_session_state"]["turn_index"] == 2:
            raise RuntimeError("Provider unavailable after retrieval")
        return {"decision": "need_context"}

    async def handle(*, session_state, **kwargs):
        assert "callback mutation" not in session_state["facts"]["pose"]
        if session_state["turn_index"] == 1:
            session_state["facts"]["pose"].append("retrieved")
            return "need_context", {"served_context": {"/recorded": {"value": None}}}
        session_state["status"] = "NEEDS_CONTEXT"
        return "need_context", {"context_errors": [{"reason": "No additional recorded fact"}]}

    def write(prepared_request, state, turn):
        artifacts.append(deepcopy(state))
        turn["request_artifact_path"] = "recorded-request"
        state["turns"][-1]["request_artifact_path"] = "recorded-request"

    planner = SimpleNamespace(product_agent=SimpleNamespace(ask_llm_structured=AsyncMock(),
        report_runtime_recovery_turn_progress=progress),
        _set_last_recovery_debug=lambda value: published.append(deepcopy(value)))
    monkeypatch.setattr(multi_turn, "_build_phase_prompt", lambda *args: ({}, "Recorded facts"))
    monkeypatch.setattr(multi_turn, "recorded_structured_call", recorded)
    monkeypatch.setattr(multi_turn, "_write_per_turn_artifact", write)
    monkeypatch.setattr(multi_turn, "_artifact_response_payload", lambda **kwargs: deepcopy(kwargs["parsed_response"]))
    monkeypatch.setitem(multi_turn._PHASE_HANDLERS, "outline", handle)
    if next_failure:
        with pytest.raises(RuntimeError, match="Provider unavailable after retrieval"):
            asyncio.run(multi_turn.execute_multi_turn_recovery(planner, prepared, session_state=seed))
    else:
        asyncio.run(multi_turn.execute_multi_turn_recovery(planner, prepared, session_state=seed))
    assert [row["multi_turn_session"]["turn_index"] for row in published] == ([1, 2] if next_failure else [1, 2, 2])
    assert artifacts[0]["facts"]["pose"] == [0, 1, 2, "retrieved"]
    assert seed["facts"]["pose"] == [0, 1, 2]
    prepared["multi_turn_session_state"]["facts"]["pose"].append("caller mutation")
    assert "caller mutation" not in prepared["recovery_debug"]["multi_turn_session"]["facts"]["pose"]
    assert "caller mutation" not in published[-1]["multi_turn_session"]["facts"]["pose"]


@pytest.mark.parametrize("marked", [False, True])
def test_saved_result_load_preserves_attempt_marker_and_validation_records(recovery, tmp_path, monkeypatch, marked):
    from cais_spade_llm.recovery_framework import offline_recovery, part_slippage_example
    from cais_spade_llm.recovery_framework.recovery_observations import observation_fingerprint

    source = tmp_path / "saved-run"
    directory = source / "generation_0001"
    (directory / "recovery_outline").mkdir(parents=True)
    observations = {"run_id": "saved-run", "scenario": "Part slippage", "failure": {"injection_status": "completed"},
                    "context_revision": 4, "prepared_recovery_request": {}}
    observation_path = source / "observations.json"
    observation_path.write_text(json.dumps(observations))
    prepared = {"validation_contract_version": 2, "recovery_session": {}, "recovery_debug": {},
                "multi_turn_session_state": {"current_phase": "primitive_generation", "turn_index": 8}}
    proposal = {"validation_contract_version": 2, "offline_recovery": True,
                "source_observation_fingerprint": observation_fingerprint(observations)}
    if marked:
        prepared["primitive_support_semantics"] = "successor_primitive_union"
        prepared["multi_turn_session_state"]["primitive_support_semantics"] = "successor_primitive_union"
        proposal["primitive_support_semantics"] = "successor_primitive_union"
    (directory / "outline_request.json").write_text(json.dumps(prepared))
    (directory / "proposal.json").write_text(json.dumps(proposal))
    (directory / "inputs.json").write_text(json.dumps({"prepared_recovery_request": prepared}))
    originals = {
        "recovery_outline/validation.json": '{"validation_contract_version": 2, "allowed": false}\n',
        "outline_evidence_requirements.json": '{"execution_authorized": false}\n',
        "prompt_safety_rules.json": '{"prompt_safety_rules": [], "safety_status": "skipped"}\n',
    }
    for name, content in originals.items():
        (directory / name).write_text(content)
    (source / "run.json").write_text(json.dumps({"status": "offline_completed", "artifact_directory": str(directory),
                                                "outline": [{"outline_id": "recorded-event"}], "primitives": []}))
    recovery.offline = True
    recovery._observations = deepcopy(observations)
    recovery._state.update(source="offline", primitive_support_semantics="successor_primitive_union")
    factory = Mock(return_value=recovery)
    monkeypatch.setattr(offline_recovery, "create_offline_recovery", factory)
    monkeypatch.setattr(part_slippage_example, "read_observations", lambda path: deepcopy(observations))
    example = PartSlippageExample(SimpleNamespace(system_running=False))
    example.source = "offline"
    example.observation_path = str(observation_path)
    result = example.load_offline_result()
    assert result["validation_contract_version"] == 2
    assert ("primitive_support_semantics" in result) is marked
    assert recovery._prepared == prepared
    assert recovery._proposal == proposal
    assert recovery._prepared["multi_turn_session_state"]["turn_index"] == 8
    loaded = Path(result["artifact_directory"])
    for name, content in originals.items():
        assert (loaded / name).read_text() == content
        assert (directory / name).read_text() == content
    assert (loaded / "inputs.json").read_bytes() == (directory / "inputs.json").read_bytes()
    assert json.loads(observation_path.read_text()) == observations
    with pytest.raises(ValueError, match="does not authorize execution"):
        recovery.action("execute")


@pytest.mark.parametrize("marked", [False, True])
def test_primitive_regeneration_preserves_the_outline_attempt_marker(recovery, marked):
    prepared = {**_accepted_composition_inputs(), "validation_contract_version": 2,
                "recovery_session": {}, "recovery_debug": {}}
    if marked:
        prepared["primitive_support_semantics"] = "successor_primitive_union"
        prepared["multi_turn_session_state"]["primitive_support_semantics"] = "successor_primitive_union"
    recovery.offline = True
    recovery._outline_request = deepcopy(prepared)
    recovery._state.update(status="offline_completed", validation_contract_version=2)

    async def regenerate():
        recovery.action("regenerate_primitives")
        await recovery._task

    asyncio.run(regenerate())
    assert ("primitive_support_semantics" in recovery._prepared) is marked
    assert ("primitive_support_semantics" in recovery._proposal) is marked
    assert recovery._outline_request == prepared
    assert recovery._proposal["validation_contract_version"] == 2
    assert recovery._proposal["execution_authorized"] is False


def test_outline_regeneration_pins_new_semantics_and_retains_saved_inputs(recovery, monkeypatch):
    from cais_spade_llm.recovery_framework import part_slippage_recovery

    captured = {"run_id": "captured-run", "context_revision": 4,
                "prepared_recovery_request": {"validation_contract_version": 2,
                    "recovery_session": {}, "recovery_debug": {}}}
    recovery.offline = True
    recovery._observations = deepcopy(captured)
    recovery.runtime.setup["selected_safety_files"] = []
    recovery._prepared = deepcopy(captured["prepared_recovery_request"])
    recovery._state.update(status="outline_review", validation_contract_version=2)
    monkeypatch.setattr(recovery, "_prepare", PartSlippageRecovery._prepare.__get__(recovery))
    monkeypatch.setattr(part_slippage_recovery, "offline_preparation_missing", lambda observations: [])

    async def regenerate():
        recovery.action("regenerate_outline")
        await recovery._task

    asyncio.run(regenerate())
    assert recovery.snapshot()["status"] == "outline_review"
    assert recovery._prepared["primitive_support_semantics"] == "successor_primitive_union"
    assert recovery._outline_request["primitive_support_semantics"] == "successor_primitive_union"
    assert recovery._observations == captured
    assert recovery._proposal == {}


def test_generic_prepared_request_does_not_upgrade_validation_semantics():
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.recovery_session import (
        RecoverySessionMixin,
    )

    planner = SimpleNamespace(
        _recovery_primitive_context=AsyncMock(return_value=([], {}, {})),
        _trim_pending_tasks=Mock(return_value={}), _resolve_recovery_reasoning_mode=Mock(return_value="multi_turn"),
        _recovery_outline_experiment_settings=Mock(return_value={}), _recovery_loaded_safety_rules=Mock(return_value=[]),
        _failure_anchor_from_payload=Mock(return_value={}), _recovery_requirement_inventory=Mock(return_value={}),
        _recovery_grounding_context=Mock(return_value={}), _recovery_marked_reentry_context=Mock(return_value={}),
        _derive_recovery_safety_constraints=Mock(return_value={}), _build_context_summary=Mock(return_value={}),
        _build_llm_input=Mock(return_value={}), logger=Mock(),
    )
    prepared = asyncio.run(RecoverySessionMixin.prepare_recovery_request(
        planner, stuck_state={}, P_id=[], ra_jid="recovery-resource-3@localhost", goal_state="assembled",
        tools_catalog=[], part_tracker={}, obligation_targets=[], recovery_feedback="", resource_states={},
        default_resource_state="idle", part_states={}, part_locations={},
    ))
    assert "primitive_support_semantics" not in prepared
    assert "primitive_support_semantics" not in prepared["multi_turn_session_seed"]
    assert "validation_contract_version" not in prepared


def test_initial_capture_keeps_owner_loop_responsive_and_persists_failure_first(recovery, monkeypatch):
    from cais_spade_llm.recovery_framework import recovery_observations

    started, finish = threading.Event(), threading.Event()
    caller_thread = threading.get_ident()

    def capture(runtime):
        assert threading.get_ident() != caller_thread
        assert json.loads((recovery.path / "observations.json").read_text())["failure"]["checkpoint"] == "recorded checkpoint"
        started.set()
        assert finish.wait(3.)
        return {"status": "supported", "snapshot": {"resources": {}, "parts": {}}}

    monkeypatch.setattr(recovery_observations, "capture_recovery_physical_evidence", capture)

    async def run():
        recovery.begin({"injection_status": "completed", "checkpoint": "recorded checkpoint"}, generate_outline=False)
        assert recovery._observations["physical_evidence"]["status"] == "pending"
        for _ in range(100):
            if started.is_set():
                break
            await asyncio.sleep(.005)
        assert started.is_set()
        assert not recovery._task.done()
        finish.set()
        await recovery._task
        assert recovery._observations["physical_evidence"]["status"] == "supported"
        assert recovery.snapshot()["status"] == "outline_ready"
    asyncio.run(run())


@pytest.mark.parametrize("available", [True, False])
def test_planning_scene_archive_is_separate_from_physics_and_never_plans(recovery, monkeypatch, available):
    from cais_spade_llm.recovery_framework import recovery_motion_preview, recovery_observations

    physical = {"snapshot": {"resources": {}, "parts": {}}, "evidence_fingerprint": "physics",
                "provenance": {"captured_monotonic": 1.}}
    monkeypatch.setattr(recovery_observations, "capture_recovery_physical_evidence", lambda runtime: deepcopy(physical))
    recovery.runtime.context.inputs["scene"] = {"safety_preparation": {"outline_preview": {
        "planning_scene_service": "/get_planning_scene", "preview_parameter_service": "/preview/get_parameters",
        "timeout_seconds": 5.}}}
    scene = {"planning_scene": {"is_diff": False}, "captured_at_unix": 2., "model_fingerprint": "model"}
    reader = AsyncMock(return_value=scene) if available else AsyncMock(side_effect=TimeoutError("scene unavailable"))
    monkeypatch.setattr(recovery_motion_preview, "capture_preview_scene", reader)
    planner = AsyncMock(side_effect=AssertionError("capture cannot plan"))
    monkeypatch.setattr(recovery_motion_preview, "preview_recovery_motion", planner)

    async def run():
        recovery.begin({"injection_status": "completed"}, generate_outline=False)
        await recovery._task
        assert recovery.snapshot()["status"] == "outline_ready"
        assert recovery._observations["physical_evidence"] == physical
        saved = json.loads((recovery.path / "observations.json").read_text())
        archive = saved["planning_scene_capture"]
        assert archive["compatibility"] == "not_checked"
        assert archive["physical_evidence_fingerprint"] == "physics"
        assert archive["status"] == ("captured" if available else "NEEDS_CONTEXT")
        if not available:
            assert archive["reason"] == "scene unavailable"
        reader.assert_awaited_once()
        planner.assert_not_called()
    asyncio.run(run())


@pytest.mark.parametrize("action", ["stop", "changed_revision"])
def test_initial_capture_discards_results_after_stop_or_changed_facts(recovery, monkeypatch, action):
    from cais_spade_llm.recovery_framework import recovery_observations

    started, finish = threading.Event(), threading.Event()
    def capture(runtime):
        started.set()
        assert finish.wait(3.)
        return {"status": "supported", "snapshot": {"resources": {}, "parts": {}}}
    monkeypatch.setattr(recovery_observations, "capture_recovery_physical_evidence", capture)
    async def run():
        recovery.begin({"injection_status": "completed"}, generate_outline=False)
        for _ in range(100):
            if started.is_set():
                break
            await asyncio.sleep(.005)
        assert started.is_set()
        if action == "stop":
            recovery.stop()
            await asyncio.sleep(0)
            assert not recovery._task.done()
        else:
            recovery.runtime.context.revision += 1
        finish.set()
        if action == "stop":
            with pytest.raises(asyncio.CancelledError):
                await recovery._task
        else:
            await recovery._task
            assert recovery.snapshot()["status"] == "blocked"
            assert "changed during physical capture" in recovery.snapshot()["message"]
        assert recovery._observations["physical_evidence"]["status"] == "pending"
    asyncio.run(run())


def _offline_preview_pending(recovery, monkeypatch):
    from cais_spade_llm.recovery_framework import recovery_motion_preview, recovery_observations

    jid = "recovery-resource-3@localhost"
    task = {"outline_id": "authored_event", "event_name": "new_event", "resource_jid": jid,
            "part_name": None, "expected_end_state": {"held_part": None}}
    physical = {"snapshot": {"resources": {"ur5e-3": {"held_part": None}}, "parts": {}},
                "resource_models": {}, "jids": {"ur5e-3": jid}}
    facts = {"execution_mode": "simulation", "resource_type": "robot", "snapshot": {"held_part": None}}
    recovery.offline = True
    recovery.runtime.source_observation_path = str(recovery.path / "source-observations.json")
    recovery._observations = {"run_id": "source", "physical_evidence": deepcopy(physical)}
    recovery._prepared = {"validation_contract_version": 2, "physical_evidence": deepcopy(physical),
        "recovery_debug": {"artifact_directory": str(recovery.path)}, "recovery_session": {},
        "recovery_resources": {jid: {"recovery_snapshot": {"held_part": None},
            "recovery_des_model": {"state_variables": {"held_part": {"scope": "resource"}}}}},
        "multi_turn_session_state": {"current_phase": "outline", "status": "paused_after_outline_turn",
            "accepted_outline_prefix": [], "projected_physical_evidence": deepcopy(physical)}}
    recovery._state.update(status="outline_ready", source="offline", validation_contract_version=2,
                           artifact_directory=str(recovery.path), outline_mode="manual", primitive_mode="manual",
                           execution_authorized=False)
    owner = SimpleNamespace(agent_name="ur5e-3", get_recovery_resource_evidence=lambda: deepcopy(facts))
    recovery.agent.resources = {jid: owner}
    archive = {"manifest_sha256": "a" * 64, "source_path": recovery.runtime.source_observation_path}
    read = Mock(return_value=archive)
    native = AsyncMock(return_value={"status": "PREVIEW_COMPLETE", "command_sent": False})
    monkeypatch.setattr(recovery_observations, "read_native_preview_model", read)
    monkeypatch.setattr(recovery_observations, "native_preview_observations", lambda observations, model: deepcopy(observations))
    monkeypatch.setattr(recovery_motion_preview, "preview_recovery_motion_from_files", native)
    return task, physical, read, native


def test_offline_manual_generate_retries_same_event_and_nominal_before_another_llm(recovery, monkeypatch):
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes import multi_turn
    from cais_spade_llm.recovery_framework import outline_gazebo_evidence

    task, physical, read, native = _offline_preview_pending(recovery, monkeypatch)
    original = deepcopy((task, physical, recovery._observations, recovery._prepared["recovery_resources"]))
    generated, checked = [], []

    async def generate(planner, prepared, *, session_state):
        assert not session_state.get("pending_outline_validation")
        generated.append(deepcopy(task))
        session_state.update(status="NEEDS_CONTEXT", pending_outline_validation={"kind": "event",
            "task": deepcopy(task), "response": {"next_transition": deepcopy(task)}, "source_turn_index": 1})

    async def prepare(owner, **kwargs):
        checked.append(deepcopy(kwargs["task"]))
        assert recovery.snapshot()["status"] == "generating_outline"
        assert recovery.snapshot()["outline_status"] == "generating"
        assert kwargs["allow_live_capture"] is False
        assert kwargs["observations"] == original[2]
        response = await kwargs["preview_call"]({"proposal": deepcopy(kwargs["task"])}, execution_mode="simulation",
            action_name="/private_preview", timeout_seconds=60, cancel_event=kwargs["cancel_event"])
        for key in ("task", "recovery_snapshot", "recovery_des_model", "physical_evidence", "resource_evidence", "observations"):
            kwargs[key]["private_calculation"] = True
        return {"status": "prepared", "native_result": response}

    async def retry(evidence):
        assert evidence["status"] == "prepared"
        session = recovery._prepared["multi_turn_session_state"]
        if session["pending_outline_validation"]["kind"] == "event":
            session["accepted_outline_prefix"] = [deepcopy(task)]
            nominal = {**deepcopy(task), "outline_id": "recorded_nominal_task", "event_name": "declared_nominal_event"}
            session["pending_outline_validation"] = {"kind": "nominal", "task": nominal,
                "recovery_snapshot": {"held_part": None}, "projected_physical_evidence": deepcopy(physical)}
            recovery._publish(status="blocked", outline_status="NEEDS_CONTEXT")
        else:
            session.pop("pending_outline_validation")
            session.update(current_phase="primitive_generation", status="ready_for_primitive_generation")

    monkeypatch.setattr(multi_turn, "execute_multi_turn_recovery", generate)
    monkeypatch.setattr(outline_gazebo_evidence, "prepare_owner_event_evidence", prepare)
    monkeypatch.setattr(recovery, "retry_outline_validation", retry)

    async def run():
        recovery.action("generate_outline")
        await recovery._task

    asyncio.run(run())
    assert generated == [task]
    assert [row["outline_id"] for row in checked] == ["authored_event", "recorded_nominal_task"]
    assert native.await_count == 2 and read.call_count == 2
    for call in native.await_args_list:
        assert "action_name" not in call.kwargs and call.kwargs["model_archive"]["manifest_sha256"] == "a" * 64
    assert (task, physical, recovery._observations, recovery._prepared["recovery_resources"]) == original
    assert recovery.snapshot()["status"] == "outline_review"
    assert recovery.snapshot()["execution_authorized"] is False


def test_offline_preview_stops_same_bound_proposal_without_new_information(recovery, monkeypatch):
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes import multi_turn
    from cais_spade_llm.recovery_framework import outline_gazebo_evidence

    task, _, read, native = _offline_preview_pending(recovery, monkeypatch)
    session = recovery._prepared["multi_turn_session_state"]
    session["pending_outline_validation"] = {"kind": "event", "task": task,
        "response": {"next_transition": deepcopy(task)}}
    original = deepcopy(session)
    provider = AsyncMock(side_effect=AssertionError("Pending validation called the provider"))
    prepare = AsyncMock(return_value={"status": "NEEDS_CONTEXT", "reason": "Required recorded fact is absent"})
    retry = AsyncMock()
    monkeypatch.setattr(multi_turn, "execute_multi_turn_recovery", provider)
    monkeypatch.setattr(outline_gazebo_evidence, "prepare_owner_event_evidence", prepare)
    monkeypatch.setattr(recovery, "retry_outline_validation", retry)
    with pytest.raises(ValueError, match="Required recorded fact is absent"):
        asyncio.run(recovery._run_stage("outline"))
    assert prepare.await_count == 1 and retry.await_count == 0
    assert read.call_count == 1
    assert recovery._prepared["multi_turn_session_state"] == original
    assert recovery.snapshot()['outline_status'] == 'NEEDS_CONTEXT'
    assert recovery.snapshot()['outline_safety_status'] == 'unresolved'
    provider.assert_not_called()
    native.assert_not_called()


@pytest.mark.parametrize("source_missing", [False, True])
def test_offline_pending_preview_requires_verified_source_before_provider_or_native(recovery, monkeypatch, source_missing):
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes import multi_turn

    task, _, read, native = _offline_preview_pending(recovery, monkeypatch)
    recovery._prepared["multi_turn_session_state"]["pending_outline_validation"] = {"kind": "event", "task": task}
    if source_missing:
        recovery.runtime.source_observation_path = None
    else:
        read.side_effect = ValueError("Native preview model archive has stale capture or model bindings")
    provider = AsyncMock(side_effect=AssertionError("Pending validation called the provider"))
    monkeypatch.setattr(multi_turn, "execute_multi_turn_recovery", provider)
    with pytest.raises(ValueError, match="saved observation path|stale capture"):
        asyncio.run(recovery._run_stage("outline"))
    assert recovery._prepared["multi_turn_session_state"]["accepted_outline_prefix"] == []
    provider.assert_not_called()
    native.assert_not_called()


@pytest.mark.parametrize("phase", ["provider", "validation"])
def test_offline_stop_drains_worker_without_committing_private_state(recovery, monkeypatch, phase):
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes import multi_turn, multi_turn_validated_event

    task, _, _, _ = _offline_preview_pending(recovery, monkeypatch)
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    original = deepcopy(recovery._prepared)

    async def held_operation(prepared, session):
        prepared["private_worker_field"] = "must not publish"
        session["accepted_outline_prefix"] = [deepcopy(task)]
        entered.set()
        try:
            assert release.wait(5), "test did not release the private worker"
            return "need_next_task", {"source_turn_index": 1}
        finally:
            finished.set()

    async def generate(planner, prepared, *, session_state):
        await held_operation(prepared, session_state)

    async def validate(*, prepared, session, planner):
        return await held_operation(prepared, session)

    monkeypatch.setattr(multi_turn, "execute_multi_turn_recovery", generate)
    monkeypatch.setattr(multi_turn_validated_event, "retry_pending_event", validate)
    if phase == "validation":
        recovery._prepared["multi_turn_session_state"]["pending_outline_validation"] = {"kind": "event", "task": task}
        original = deepcopy(recovery._prepared)

    async def run():
        if phase == "provider":
            recovery._schedule(recovery._generate_outline)
        else:
            recovery._schedule(lambda: recovery.retry_outline_validation({}))
        owned = recovery._task
        try:
            for _ in range(100):
                if entered.is_set():
                    break
                await asyncio.sleep(0.01)
            assert entered.is_set()
            recovery.stop()
            await asyncio.sleep(0)
            assert recovery.snapshot()["status"] == "stopped"
            assert not owned.done() and not finished.is_set()
            assert recovery._prepared.get("private_worker_field") is None
            assert recovery._prepared["multi_turn_session_state"] == original["multi_turn_session_state"]
            with pytest.raises(ValueError, match="Reset example"):
                recovery._schedule(lambda: recovery.retry_outline_validation({}))
            recovery.stop()
            await asyncio.sleep(0)
        finally:
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await owned
        assert finished.is_set() and owned.done()
        assert recovery._prepared.get("private_worker_field") is None
        assert recovery._prepared["multi_turn_session_state"] == original["multi_turn_session_state"]

    asyncio.run(run())


def test_offline_owner_preview_never_captures_missing_scene(recovery, monkeypatch):
    from cais_spade_llm.recovery_framework import outline_gazebo_evidence, recovery_motion_preview
    from cais_spade_llm.resources import resource_safety_preparation

    fields = {"action_name": "/private_preview", "planning_scene_service": "/recorded_scene",
        "preview_parameter_service": "/recorded_parameters", "max_translation_step": 0.01,
        "max_rotation_step": 0.05, "max_joint_step": 0.35, "velocity_scaling": 0.1,
        "acceleration_scaling": 0.1, "timeout_seconds": 60}
    evidence = {"execution_mode": "simulation", "resource_type": "robot",
                "inputs": {"scene": {"safety_preparation": {"outline_preview": fields}}}}
    live = AsyncMock(side_effect=AssertionError("Offline preview captured live state"))
    file_preview = AsyncMock(side_effect=AssertionError("Missing scene reached native preview"))
    monkeypatch.setattr(recovery_motion_preview, "capture_preview_scene", live)
    monkeypatch.setattr(recovery_motion_preview, "preview_recovery_motion", live)
    monkeypatch.setattr(resource_safety_preparation, "prepare_outline_event_preview_request",
        lambda **kwargs: {"status": "ready", "motion_requests": [{"mode": "cartesian"}]})
    result = asyncio.run(outline_gazebo_evidence.prepare_owner_event_evidence(SimpleNamespace(),
        task={}, recovery_snapshot={}, recovery_des_model={}, physical_evidence={}, resource_evidence=evidence,
        observations={}, preview_call=file_preview, allow_live_capture=False))
    assert result["status"] == "NEEDS_CONTEXT"
    assert "Saved planning_scene_capture is missing" in result["reason"]
    assert result["command_sent"] is False and result["execution_authorized"] is False
    live.assert_not_called()
    file_preview.assert_not_called()


@pytest.mark.parametrize("rid", ["M1", "M2", "Conveyor", "3D Printing Station"])
def test_owner_evidence_keeps_resource_projection_and_missing_facts(rid, monkeypatch):
    from cais_spade_llm.recovery_framework import outline_gazebo_evidence, recovery_motion_preview
    from cais_spade_llm.resources import resource_safety_preparation

    forbidden = Mock(side_effect=AssertionError("Resource projection reached robot or live preparation"))
    for module, name in ((resource_safety_preparation, "prepare_outline_event_preview_request"),
                         (recovery_motion_preview, "capture_preview_scene"),
                         (recovery_motion_preview, "preview_scene_fingerprint")):
        monkeypatch.setattr(module, name, forbidden)
    request = {"task": {"event_name": "authored_event", "resource_jid": rid + "@localhost",
                         "part_name": None, "expected_end_state": {}},
               "recovery_snapshot": {"part_name": None}, "recovery_des_model": {},
               "physical_evidence": {"snapshot": {"resources": {rid: {"part_name": None}}}},
               "resource_evidence": {"resource_type": "resource", "execution_mode": "simulation"},
               "observations": {}}
    original = deepcopy(request)
    missing = {"status": "needs_context", "unresolved": ["Recorded owner interval is unavailable"],
               "context_requests": [f"/resources/{rid}@localhost/outline_event_evidence"],
               "command_sent": False, "execution_authorized": False}
    def project(**kwargs):
        assert kwargs["physical_input"]["physical_evidence"] == original["physical_evidence"]
        kwargs["task"]["private_calculation"] = True
        kwargs["recovery_snapshot"]["private_calculation"] = True
        return deepcopy(missing)
    owner = SimpleNamespace(project_recovery_outline_event=Mock(side_effect=project))
    result = asyncio.run(outline_gazebo_evidence.prepare_owner_event_evidence(
        owner, **request, allow_live_capture=False))
    assert result["status"] == "NEEDS_CONTEXT"
    assert result["geometry_evidence"] == missing
    assert result["context_requests"] == missing["context_requests"]
    assert result["reason"] == missing["unresolved"][0]
    assert request == original
    owner.project_recovery_outline_event.assert_called_once()
    forbidden.assert_not_called()
    absent = asyncio.run(outline_gazebo_evidence.prepare_owner_event_evidence(
        SimpleNamespace(), **request, allow_live_capture=False))
    assert absent["status"] == "NEEDS_CONTEXT"
    assert absent["reason"] == "Resource-owned event projection is unavailable"


@pytest.mark.parametrize("resource_type", ["robot", "kmr", "resource"])
def test_owner_evidence_does_not_preview_incompatible_checkpoints(resource_type, monkeypatch):
    from cais_spade_llm.recovery_framework import outline_gazebo_evidence, kmr_outline_evidence
    from cais_spade_llm.resources import resource_safety_preparation

    forbidden = Mock(side_effect=AssertionError("Incompatible records reached physical projection"))
    monkeypatch.setattr(resource_safety_preparation, "prepare_outline_event_preview_request", forbidden)
    monkeypatch.setattr(kmr_outline_evidence, "prepare_kmr_event_evidence", forbidden)
    compatibility = {"status": "NEEDS_CONTEXT", "checks": [
        {"resource_id": "M2", "status": "NEEDS_CONTEXT", "reason": "Recorded observation times differ"}]}
    owner = SimpleNamespace(project_recovery_outline_event=forbidden)
    result = asyncio.run(outline_gazebo_evidence.prepare_owner_event_evidence(owner,
        task={}, recovery_snapshot={}, recovery_des_model={}, observations={},
        physical_evidence={"processing_capture_compatibility": compatibility},
        resource_evidence={"resource_type": resource_type, "execution_mode": "simulation"},
        allow_live_capture=False))
    assert result["status"] == "NEEDS_CONTEXT"
    assert result["capture_compatibility"]["capture_diagnostic"] == compatibility
    assert result["capture_compatibility"]["checks"][0]["ref"] == "/physical_evidence/outline_safety_specifications"
    assert result["reason"] == "Selected AP dependencies were not recorded"
    assert not result["execution_authorized"] and not result["command_sent"]
    forbidden.assert_not_called()


def test_owner_evidence_retains_kmr_preview_dispatch(monkeypatch):
    from cais_spade_llm.recovery_framework import outline_gazebo_evidence, kmr_outline_evidence
    from cais_spade_llm.resources import resource_safety_preparation

    robot = Mock(side_effect=AssertionError("KMR used the robot preview fallback"))
    kmr = AsyncMock(return_value={"status": "NEEDS_CONTEXT", "reason": "Recorded base motion is unavailable"})
    monkeypatch.setattr(resource_safety_preparation, "prepare_outline_event_preview_request", robot)
    monkeypatch.setattr(kmr_outline_evidence, "prepare_kmr_event_evidence", kmr)
    result = asyncio.run(outline_gazebo_evidence.prepare_owner_event_evidence(SimpleNamespace(),
        task={}, recovery_snapshot={}, recovery_des_model={}, physical_evidence={}, observations={},
        resource_evidence={"resource_type": "kmr", "resource_id": "KMR", "environment_models": {"KMR": {}},
                           "execution_mode": "simulation"}, allow_live_capture=False))
    assert result == kmr.return_value
    kmr.assert_awaited_once()
    assert kmr.await_args.kwargs["allow_live_capture"] is False
    robot.assert_not_called()


def test_offline_owner_evidence_needs_no_robot_archive_and_preserves_pending_event(recovery, monkeypatch):
    from cais_spade_llm.recovery_framework import outline_gazebo_evidence

    task, _, read, native = _offline_preview_pending(recovery, monkeypatch)
    owner = recovery.agent.resources[task["resource_jid"]]
    facts = owner.get_recovery_resource_evidence()
    facts["resource_type"] = "resource"
    owner.get_recovery_resource_evidence = lambda: deepcopy(facts)
    owner.project_recovery_outline_event = Mock(return_value={"status": "needs_context",
        "unresolved": ["Recorded owner interval is unavailable"], "context_requests": [],
        "execution_authorized": False, "command_sent": False})
    source = Path(recovery.runtime.source_observation_path)
    source.write_text(json.dumps(recovery._observations))
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    recovery._prepared["attempt_provenance"] = {"capture": {"source_sha256": digest}}
    session = recovery._prepared["multi_turn_session_state"]
    session.update(outline_monitor_context={"existing": True}, pending_outline_validation={
        "kind": "event", "task": task, "source_turn_index": 5, "response": {"next_transition": deepcopy(task)}})
    original = deepcopy(session)
    read.side_effect = AssertionError("Nonrobot owner requires a robot archive")
    retry = AsyncMock(side_effect=AssertionError("Missing owner evidence cannot retry as success"))
    monkeypatch.setattr(recovery, "retry_outline_validation", retry)
    with pytest.raises(ValueError, match="Recorded owner interval is unavailable"):
        asyncio.run(recovery._prepare_offline_outline_evidence(set()))
    assert recovery._prepared["multi_turn_session_state"] == original
    assert recovery.snapshot()["outline_status"] == "NEEDS_CONTEXT"
    assert hashlib.sha256(source.read_bytes()).hexdigest() == digest
    owner.project_recovery_outline_event.assert_called_once()
    read.assert_not_called()
    native.assert_not_called()
    records = list(recovery.path.glob("offline_validation/*/evidence.json"))
    assert len(records) == 1 and json.loads(records[0].read_text())["status"] == "NEEDS_CONTEXT"
    source.write_text("{}")
    with pytest.raises(ValueError, match="saved observations changed"):
        asyncio.run(recovery._prepare_offline_outline_evidence(set()))
    owner.project_recovery_outline_event.assert_called_once()


def test_offline_stop_during_nominal_archive_retains_artifact_without_committing(recovery, monkeypatch):
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery import recovery_artifacts
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes import multi_turn_validated_event

    task, _, _, _ = _offline_preview_pending(recovery, monkeypatch)
    recovery._prepared.update(nominal_continuation_artifact_version=1,
        source_observation_fingerprint="a" * 64, saved_pending_nodes=[])
    current = recovery._prepared["multi_turn_session_state"]
    current.update(validation_contract_version=2, turn_index=1,
        pending_outline_validation={"kind": "event", "task": deepcopy(task)})
    before = deepcopy(recovery._prepared)
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    archive = recovery_artifacts.archive_nominal_continuation
    written = []

    async def validate(*, prepared, session, planner):
        proof = {"complete": True, "status": "passed", "tasks": [], "replaced_task_ids": [],
                 "replacement_evidence": {}, "execution_authorized": False}
        session.pop("pending_outline_validation")
        session.update(accepted_outline_prefix=[deepcopy(task)], status="ready_for_primitive_generation",
            nominal_continuation_evidence=deepcopy(proof))
        return "outline_ready", {"source_turn_index": 1, "nominal_continuation_evidence": deepcopy(proof)}

    def persist(prepared, session, turn, *, decision):
        entered.set()
        try:
            assert release.wait(5), "test did not release the archive writer"
            path = archive(prepared, session, turn, decision=decision)
            written.append(path)
            return path
        finally:
            finished.set()

    monkeypatch.setattr(multi_turn_validated_event, "retry_pending_event", validate)
    monkeypatch.setattr(recovery_artifacts, "archive_nominal_continuation", persist)

    async def run():
        recovery._schedule(lambda: recovery.retry_outline_validation({}))
        owned = recovery._task
        try:
            for _ in range(100):
                if entered.is_set():
                    break
                await asyncio.sleep(0.01)
            assert entered.is_set()
            assert recovery._prepared == before
            recovery.stop()
            await asyncio.sleep(0)
            assert recovery.snapshot()["status"] == "stopped"
            assert not owned.done() and not finished.is_set()
            assert recovery._prepared == before
            with pytest.raises(ValueError, match="Reset example"):
                recovery._schedule(lambda: recovery.retry_outline_validation({}))
        finally:
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await owned
        assert owned.done() and finished.is_set()

    asyncio.run(run())
    assert recovery._prepared == before
    assert recovery.snapshot()["status"] == "stopped" and recovery.snapshot()["outline"] == []
    assert recovery.snapshot()["execution_authorized"] is False
    assert len(written) == 1 and written[0] is not None and written[0].is_file()
    retained = json.loads(written[0].read_text())
    assert retained["nominal_continuation_evidence"]["complete"] is True
    resume = json.loads((recovery.path / "validation_resume.json").read_text())
    assert resume["multi_turn_session_state"]["accepted_outline_prefix"] == [task]
    assert not (recovery.path / "outline.json").exists()
    assert not (recovery.path / "outline_request.json").exists()


def test_attempt_provenance_is_pinned_once_and_preserved_on_restore(recovery, monkeypatch):
    from cais_spade_llm.recovery_framework import diagnostics, fingerprint
    from cais_spade_llm.recovery_framework.recovery_observations import observation_fingerprint

    implementation = Mock(return_value='recorded-implementation')
    monkeypatch.setattr(diagnostics, 'validator_fingerprint', implementation)
    recovery.runtime.setup['selected_safety_files'] = []
    recovery._observations = {'run_id': 'source-run', 'captured_at': 'source-time', 'context_revision': 5,
                             'inputs': {'recorded': None}, 'planning_scene_capture': {'model_fingerprint': 'recorded-model'}}
    recovery.path.mkdir(parents=True, exist_ok=True)
    source = recovery.path / 'observations.json'
    source.write_text(json.dumps(recovery._observations))
    before = source.read_bytes()
    prepared = {'recovery_session': {}, 'recovery_debug': {}}
    directory = recovery._new_artifact_directory()
    recovery._install_prepared(prepared, directory, '')
    recorded = deepcopy(recovery._prepared)
    provenance = recorded['attempt_provenance']
    assert provenance['capture'] == {'run_id': 'source-run', 'captured_at': 'source-time', 'context_revision': 5,
        'observation_fingerprint': observation_fingerprint(recovery._observations),
        'source_sha256': hashlib.sha256(before).hexdigest()}
    assert provenance['configuration_fingerprint'] == fingerprint({'recorded': None})
    assert provenance['model_fingerprint'] == 'recorded-model'
    assert provenance['implementation_fingerprint'] == 'recorded-implementation'
    assert provenance['unassisted'] is True
    assert source.read_bytes() == before
    for _ in range(3):
        assert recovery.snapshot()['attempt_provenance'] == provenance
    implementation.assert_called_once_with()
    implementation.side_effect = AssertionError('Restoring history must not hash current implementation')
    restored = recovery._new_artifact_directory()
    recovery._install_prepared(recorded, restored, '', new_attempt=False, source_directory=directory)
    assert recovery.snapshot()['attempt_provenance'] == provenance
    assert json.loads((restored / 'attempt_provenance.json').read_text()) == provenance
    assert recovery.snapshot()['provenance_path'] == str(restored / 'attempt_provenance.json')


def test_historical_attempt_does_not_receive_current_provenance(recovery, monkeypatch):
    from cais_spade_llm.recovery_framework import diagnostics

    monkeypatch.setattr(diagnostics, 'validator_fingerprint', Mock(side_effect=AssertionError('Historical record changed')))
    recovery._state.update(attempt_provenance={'version': 1}, provenance_path='previous')
    directory = recovery._new_artifact_directory()
    recovery._install_prepared({'validation_contract_version': 2}, directory, '', new_attempt=False)
    assert recovery.snapshot()['attempt_provenance'] == {}
    assert recovery.snapshot()['provenance_path'] == ''
    assert not (directory / 'attempt_provenance.json').exists()


def test_operator_assisted_attempt_provenance_is_excluded_from_unassisted_results(recovery):
    recovery.runtime.setup['selected_safety_files'] = []
    recovery._observations = {'run_id': 'recorded'}
    assistance = {'action': 'regenerate_outline', 'feedback': 'operator supplied text', 'generation': 1}
    recovery._state['operator_assistance'] = [assistance]
    recovery._install_prepared({'recovery_session': {}, 'recovery_debug': {}}, recovery._new_artifact_directory(), assistance['feedback'])
    assert recovery.snapshot()['attempt_provenance']['unassisted'] is False
    assert recovery.snapshot()['attempt_provenance']['operator_assistance'] == [assistance]


@pytest.mark.parametrize('new_attempt', [True, False])
def test_new_attempt_restores_declared_owner_fields_without_rewriting_capture_or_history(recovery, new_attempt):
    jid = 'recovery-resource-6@localhost'
    declared = {'resource_state': {'scope': 'resource'}, 'part_name': {'scope': 'resource'},
                'staging_part': {'scope': 'resource'}}
    recorded = {'resource_state': 'idle', 'part_name': None}
    owner = SimpleNamespace(get_recovery_snapshot=Mock(return_value=deepcopy(recorded)),
        recovery_des_model=Mock(return_value={'descriptor_fingerprint': 'actual-owner-descriptor',
            'state_variables': deepcopy(declared), 'events': [], 'primitive_catalog': [{'name': 'dwell', 'params': {}, 'required_params': []}]}),
        static_capabilities={'process_capabilities': {'trim': {'supported_results': ['square']}}})
    recovery.offline = True
    recovery.agent.resources = {jid: owner}
    recovery.runtime.setup['selected_safety_files'] = []
    recovery._observations = {'run_id': 'capture', 'resource_evidence': {jid: {
        'resource_id': 'M2', 'environment_models': {'M2': {'state_variables': declared}}, 'snapshot': deepcopy(recorded)}}}
    owner.get_recovery_resource_evidence = Mock(return_value=deepcopy(recovery._observations['resource_evidence'][jid]))
    original = deepcopy(recovery._observations)
    prepared = {'recovery_session': {}, 'recovery_debug': {}, 'validation_contract_version': 2,
        'ra_jid': jid, 'recovery_resources': {jid: {'resource_type': 'resource',
            'recovery_snapshot': {'current_state': 'unknown'}, 'static_capabilities': {},
            'recovery_des_model': {'state_variables': declared}}},
        'llm_input': {'observed_runtime_state': {'resources': [{'resource_jid': jid,
            'current_state': 'unknown', 'staging_part': 'must not survive absent owner fact'}]}}}
    before = deepcopy(prepared)
    recovery._install_prepared(prepared, recovery._new_artifact_directory(), '', new_attempt=new_attempt)
    assert recovery._observations == original
    if new_attempt:
        entry = recovery._prepared['recovery_resources'][jid]
        assert entry['recovery_snapshot']['resource_state'] == 'idle'
        assert entry['recovery_snapshot']['part_name'] is None
        assert 'staging_part' not in entry['recovery_snapshot']
        assert entry['recovery_des_model_fingerprint'] == 'actual-owner-descriptor'
        assert entry['static_capabilities'] == owner.static_capabilities
        assert entry['execution_primitive_catalog'][0]['name'] == 'dwell'
        assert entry['recovery_adapter']['supports_executable_recovery'] is True
        assert recovery.snapshot()['execution_authorized'] is False
        assert recovery._prepared['recovery_session']['outline_mode'] == 'incremental_candidates_validated'
        assert recovery._prepared['multi_turn_session_seed']['outline_mode'] == 'incremental_candidates_validated'
        current = recovery._prepared['llm_input']['observed_runtime_state']['resources'][0]
        assert current['resource_state'] == 'idle' and current['part_name'] is None
        assert 'staging_part' not in current
    else:
        assert recovery._prepared == before
        owner.get_recovery_snapshot.assert_not_called()
        owner.recovery_des_model.assert_not_called()


def test_full_validation_choices_stay_locked_when_generation_pauses_without_an_event(recovery, monkeypatch):
    from cais_spade_llm.recovery_framework.part_slippage_example import PartSlippageExample
    from cais_spade_llm.recovery_framework.outline_checks import CHECK_FIELDS

    choices = dict.fromkeys(CHECK_FIELDS, True)
    recovery._publish(status="outline_ready", outline=[], primitives=[], validation_checks=choices)
    example = PartSlippageExample(SimpleNamespace(system_running=False))
    example.source, example.safety_files = "offline", []
    example.validation_checks, example._offline = choices, recovery

    async def paused():
        recovery._publish(status="blocked", outline_status="NEEDS_CONTEXT", outline=[])

    monkeypatch.setattr(recovery, "_generate_outline", paused)

    async def run():
        await example.action("generate_outline")
        assert example.snapshot()["validation_checks_locked"] is True
        await recovery._task
        assert example.snapshot()["outline"] == []
        assert example.snapshot()["validation_checks_locked"] is True
        with pytest.raises(ValueError, match="Start fresh"):
            example.set_options(source="offline", scenario="Part slippage", safety_files=[],
                check_ra_feasibility=False, check_physical_evidence=False, check_cca_safety=False)

    asyncio.run(run())


def test_offline_primitive_generate_includes_review_preparation(recovery, monkeypatch):
    order = []
    recovery.offline = True
    recovery._state["status"] = "outline_review"

    async def prepare():
        order.append("prepare")
        recovery._publish(status="primitive_ready")

    async def generate():
        order.append("generate")

    monkeypatch.setattr(recovery, "_continue_outline", prepare)
    monkeypatch.setattr(recovery, "_generate_primitives", generate)

    async def run():
        recovery.action("generate_primitives")
        await recovery._task

    asyncio.run(run())
    assert order == ["prepare", "generate"]


def test_offline_primitive_generate_keeps_partial_outline_gate(recovery, monkeypatch):
    recovery.offline = True
    recovery._state["status"] = "outline_review"
    recovery._outline_request = {"multi_turn_session_state": {
        "outline_diagnostic_bypass": True, "current_phase": "primitive_generation", "accepted_outline_prefix": []}}
    generate = AsyncMock()
    monkeypatch.setattr(recovery, "_generate_primitives", generate)

    async def run():
        recovery.action("generate_primitives")
        await recovery._task

    asyncio.run(run())
    assert recovery.snapshot()["status"] == "blocked"
    assert "full validation is required" in recovery.snapshot()["message"]
    generate.assert_not_called()
