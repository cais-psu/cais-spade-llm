from __future__ import annotations

"""Part slippage stage modes, review ownership and simulation-only execution."""

import asyncio
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


def _accept_outline(recovery):
    """Install the accepted outline that composition reads; the live workflow stops at outline_ready."""
    recovery._outline_request = deepcopy(recovery._prepared)
    recovery._publish(status="outline_review", outline_status="completed",
                      outline=recovery._prepared["multi_turn_session_state"]["accepted_outline_prefix"])


@pytest.mark.parametrize("outline_mode", ["auto", "manual"])
@pytest.mark.parametrize("primitive_mode", ["auto", "manual"])
def test_stage_modes_stop_before_execution(recovery, monkeypatch, outline_mode, primitive_mode):
    execute = AsyncMock()
    monkeypatch.setattr("cais_spade_llm.recovery_framework.slippage_execution.execute_generated_recovery", execute)
    recovery._state.update(outline_mode=outline_mode, primitive_mode=primitive_mode)

    async def run():
        recovery.begin({"injection_status": "completed"})
        await recovery._task
        assert recovery.snapshot()["status"] == "outline_ready"
        with pytest.raises(ValueError, match="unavailable"):
            recovery.action("generate_outline")
        _accept_outline(recovery)
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


def _live_grounding_case(recovery, monkeypatch, attachments: list, *, checkpoint: bool = False) -> tuple:
    """A robot reported a fault after losing its peg; each physical capture reads the next attachment."""
    context = recovery.runtime.context
    context.resources = {"arm_a": SimpleNamespace(valuation={"resource_state": "picked", "held_part": "peg"}, revision=1)}
    context.part_tracker = {"peg": {"location": "arm_a", "current_process": None}}
    failure = {"injection_status": "completed",
               "resource_values_before": {"arm_a": {"resource_state": "picked", "held_part": "peg"}},
               "part_tracker_before": {"peg": {"location": "arm_a", "current_process": None}},
               "pending_tasks": [{"task_id": "t2", "resource_id": "arm_a", "event_id": "place", "event_name": "place",
                                  "parameters": {"part_name": "peg"}}], "continuations": {}}

    def physical(model_name):
        # A dict also reports another resource's controller fault in that capture.
        faults = model_name.get("faults", {}) if isinstance(model_name, dict) else {}
        model_name = model_name.get("model_name") if isinstance(model_name, dict) else model_name
        resources = {} if model_name == "unobserved" else {"arm_a": {"attachment": {"model_name": model_name}}}
        resources.update({rid: {"controller_hold_observation": {"fault_reason": reason}} for rid, reason in faults.items()})
        return {"resources": resources, "parts": {"peg": {"name": "peg", "pose": [3.0, 3.0, 1.0, 0, 0, 0, 1]}},
                "geometry": {"regions": {}, "parts": {}}, "physics_evidence": {"attachments": []}}

    observations = {"schema_version": 2, "run_id": "test-run", "jids": {"arm_a": "jid-a"}, "acknowledgements": {},
                    "failure": failure, "fault_reports": {"arm_a": {"reported_state": "failed"}},
                    "part_tracker": deepcopy(context.part_tracker),
                    "resource_states": {"arm_a": {"resource_state": "picked", "held_part": "peg"}},
                    "context_revision": context.revision,
                    "physical_evidence": physical(attachments[0]) if checkpoint else {"status": "pending"}}
    if checkpoint:
        recovery.runtime.recovery_checkpoint_observations = deepcopy(observations)
    monkeypatch.setattr("cais_spade_llm.recovery_framework.part_slippage_recovery.capture_observations",
                        lambda runtime, failure, include_physical=True: deepcopy(observations))
    trackers = []

    async def capture_physical():
        trackers.append(deepcopy(context.part_tracker))
        recovery._observations["physical_evidence"] = physical(attachments[len(trackers) - 1])
        return recovery._observations["physical_evidence"]

    monkeypatch.setattr(recovery, "_capture_physical_evidence", capture_physical)

    async def run():
        recovery.begin(failure)
        await recovery._task

    return context, trackers, run


def test_live_capture_grounds_beliefs_then_retakes_belief_dependent_evidence(recovery, monkeypatch):
    from cais_spade_llm.recovery_framework.capture_files import BELIEFS_NAME

    # The robot gripper is empty and the peg lies where it fell.
    context, trackers, run = _live_grounding_case(recovery, monkeypatch, [None, None])
    asyncio.run(run())
    # Held by the robot, then located only by its measured pose.
    assert [row["peg"]["location"] for row in trackers] == ["arm_a", None]
    assert context.resources["arm_a"].valuation == {"resource_state": "failed", "held_part": None}
    assert context.part_tracker["peg"]["observed_pose"] == {"x": 3.0, "y": 3.0, "z": 1.0, "qx": 0, "qy": 0, "qz": 0,
                                                           "qw": 1}
    assert recovery._observations["context_revision"] == context.revision == 5
    path = Path(recovery.snapshot()["observation_path"])
    assert "part_tracker" not in json.loads(path.read_text())
    views = json.loads((path.parent / BELIEFS_NAME).read_text())["views"]
    assert views["part_tracker"]["peg"]["location"] is None and "state" not in views["part_tracker"]["peg"]
    assert views["resource_states"]["arm_a"] == {"resource_state": "failed", "held_part": None}


def test_live_grounding_settles_when_the_first_capture_missed_the_robot(recovery, monkeypatch):
    # Stage 1 on October 15: the first capture had no observation of the slipping robot.
    context, trackers, run = _live_grounding_case(recovery, monkeypatch, ["unobserved", None, None])
    asyncio.run(run())
    assert [row["peg"]["location"] for row in trackers] == ["arm_a", "arm_a", None]
    assert context.resources["arm_a"].valuation == {"resource_state": "failed", "held_part": None}
    assert context.part_tracker["peg"]["location"] is None
    passes = json.loads((recovery.path / "grounding_passes.json").read_text())["passes"]
    assert [row["signals"] for row in passes] == [["B2:arm_a"], ["B2:arm_a", "C1:arm_a"], ["B2:arm_a", "C1:arm_a"]]


def test_live_grounding_keeps_only_what_the_latest_captures_agree_on(recovery, monkeypatch):
    # The first capture missed the peg; two retakes measure it in the gripper.
    context, trackers, run = _live_grounding_case(recovery, monkeypatch, [None, "peg", "peg"])
    asyncio.run(run())
    assert len(trackers) == 3
    assert context.resources["arm_a"].valuation == {"resource_state": "failed", "held_part": "peg"}
    assert context.part_tracker["peg"] == {"location": "arm_a", "current_process": None}
    passes = json.loads((recovery.path / "grounding_passes.json").read_text())["passes"]
    assert [row["signals"] for row in passes] == [["B2:arm_a", "C1:arm_a"], ["B2:arm_a"], ["B2:arm_a"]]


def test_live_grounding_restores_beliefs_when_retaken_captures_never_agree(recovery, monkeypatch):
    context, trackers, run = _live_grounding_case(recovery, monkeypatch, [None, "peg", None, "peg"])
    asyncio.run(run())
    assert len(trackers) == 4
    assert recovery.snapshot()["status"] == "blocked"
    assert "did not settle" in recovery.snapshot()["message"]
    assert context.resources["arm_a"].valuation == {"resource_state": "picked", "held_part": "peg"}
    assert context.part_tracker["peg"] == {"location": "arm_a", "current_process": None}
    assert recovery._observations["part_tracker"]["peg"] == {"location": "arm_a", "current_process": None}


def test_live_grounding_stops_when_a_retake_grounds_a_different_fault(recovery, monkeypatch):
    retake = {"model_name": None, "faults": {"arm_b": "joint limit"}}
    context, trackers, run = _live_grounding_case(recovery, monkeypatch, [None, retake])
    asyncio.run(run())
    assert "grounds a different fault" in recovery.snapshot()["message"]
    assert context.resources["arm_a"].valuation == {"resource_state": "picked", "held_part": "peg"}


def test_frozen_checkpoint_grounding_never_changes_beliefs(recovery, monkeypatch):
    context, trackers, run = _live_grounding_case(recovery, monkeypatch, [None], checkpoint=True)
    prepare = AsyncMock()
    monkeypatch.setattr(recovery, "_prepare", prepare)
    asyncio.run(run())
    # The frozen record is kept as captured and preparation continues; offline grounding applies the fault.
    assert trackers == []
    prepare.assert_awaited_once()
    assert context.resources["arm_a"].valuation == {"resource_state": "picked", "held_part": "peg"}
    assert context.part_tracker["peg"] == {"location": "arm_a", "current_process": None}
    assert context.revision == 4


def test_generation_archives_nominal_proof_before_debug_publication(tmp_path, monkeypatch):
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes import multi_turn
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.recovery_artifacts import (
        read_nominal_continuation,
    )

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


def test_primitive_regeneration_retains_verified_nominal_archive(recovery, tmp_path):
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.recovery_artifacts import (
        archive_nominal_continuation,
        nominal_continuation_context_binding,
        read_nominal_continuation,
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
    assert failure["failure_context"]["observations"]["state_after"] == {"held_part": None, "part_location": None}
    assert received["resource_states"]["robot@localhost"] == {"held_part": None}
    assert received["default_resource_state"] == ""
    assert fresh == original
    assert template == original_template


def test_owner_evidence_capture_satisfies_default_pick_calculation_without_calls():
    from cais_spade_llm.recovery_framework.recovery_observations import capture_resource_evidence
    from cais_spade_llm.resources.robot.gazebo_pick_place_controller import (
        GazeboPickPlaceController,
    )
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
    from cais_spade_llm.resources.robot.gazebo_pick_place_controller import (
        GazeboPickPlaceController,
    )

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
    from cais_spade_llm.recovery_framework.recovery_observations import (
        _captured_tcp_transform,
        capture_resource_evidence,
    )

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


def _kmr_reconcile_fixture(primitive_state: dict, held_before=None):
    part = "KET4_Square_4mm"
    kmr = SimpleNamespace(valuation={"resource_state": "carrying" if held_before else "idle", "held_part": held_before,
                                     "approached_part": None, "resource_location": "M1"}, revision=0)
    machine = SimpleNamespace(valuation={"resource_state": "completed", "part_name": part}, revision=0)
    pickup = {"event_name": "pick_grasp", "participants": ["ur5e-1", "M1"],
              "parameter_bindings": {"resource_id": {"equals": "ur5e-1"}, "origin_resource_location": {"equals": "M1"}},
              "guards": {"resource_state": {"equals": "completed"}, "part_name": {"equals_from_param": "part_name"}},
              "updates": {"resource_state": {"set": "idle"}, "part_name": {"set": None}}}
    context = SimpleNamespace(
        run_id="trial", resources={"KMR": kmr, "M1": machine},
        models={"KMR": {"state_variables": {"resource_location": {"domain": [None, "Storage", "M1", "M2"]},
                                            "approached_part": {}}, "events": []},
                "M1": {"state_variables": {}, "events": [pickup]}},
        part_tracker={part: {"location": "M1", "current_process": None, "processCompleted": []}},
        admission_lock=threading.RLock(), revision=0, transitions=[])
    context.snapshot = lambda: {"KMR": deepcopy(kmr.valuation), "M1": deepcopy(machine.valuation)}
    owner = SimpleNamespace(agent_name="KMR", _primitive_state=primitive_state)
    return part, context, SimpleNamespace(context=context, agent=SimpleNamespace()), owner


def test_KMR_pickup_reconciles_its_measured_custody_and_the_source_owner_consequence():
    from cais_spade_llm.recovery_framework.slippage_execution import _reconcile_mobile_completion

    part, context, runtime, owner = _kmr_reconcile_fixture(
        {"held_part": "KET4_Square_4mm", "resource_location": "M1", "base_pose": [-4.7, 2.2, 1.5708]})
    task = {"part_name": part, "outline_id": "RECOVERY_SEQ2", "macro_name": "pick_part",
            "expected_end_state": {"held_part": part}}
    observed = asyncio.run(_reconcile_mobile_completion(runtime, owner, task, "kmr-task", {}))
    assert observed == {"base_pose": [-4.7, 2.2, 1.5708], "resource_location": "M1", "held_part": part}
    assert context.resources["KMR"].valuation["resource_state"] == "carrying"
    assert context.resources["M1"].valuation == {"resource_state": "idle", "part_name": None}
    assert context.part_tracker[part]["location"] == "KMR"
    assert context.transitions[0]["acknowledgement"]["evidence"] == "resource"


def test_KMR_coordinate_move_reconciles_a_location_no_dock_names():
    from cais_spade_llm.recovery_framework.slippage_execution import _reconcile_mobile_completion

    part, context, runtime, owner = _kmr_reconcile_fixture(
        {"held_part": "KET4_Square_4mm", "resource_location": None, "base_pose": [-1.2, 1.15, -1.5708]},
        held_before="KET4_Square_4mm")
    context.part_tracker[part].update(location="KMR")
    task = {"part_name": part, "outline_id": "RECOVERY_SEQ3", "macro_name": "move_to_location",
            "expected_end_state": {"resource_location": None}}
    asyncio.run(_reconcile_mobile_completion(runtime, owner, task, "kmr-drive", {}))
    assert context.resources["KMR"].valuation["resource_location"] is None
    assert context.resources["M1"].valuation["part_name"] == part
    assert context.part_tracker[part]["location"] == "KMR"


@pytest.mark.parametrize("state,message", [
    ({"held_part": None, "resource_location": None}, "observed resting place"),
    ({"held_part": "KET4_Square_4mm", "resource_location": "Conveyor"}, "resource_location=Conveyor"),
])
def test_KMR_release_and_undeclared_locations_are_not_committed(state, message):
    from cais_spade_llm.recovery_framework.slippage_execution import _reconcile_mobile_completion

    part, context, runtime, owner = _kmr_reconcile_fixture(state, held_before="KET4_Square_4mm")
    before = deepcopy(context.resources["KMR"].valuation)
    with pytest.raises(ValueError, match=message):
        asyncio.run(_reconcile_mobile_completion(runtime, owner, {"part_name": part, "outline_id": "RECOVERY_SEQ4",
                                                                  "macro_name": "release"}, "kmr-release", {}))
    assert context.resources["KMR"].valuation == before and context.transitions == []


def _kmr_release_fixture(pose):
    from cais_spade_llm.product.environment import EnvironmentProductContext
    from cais_spade_llm.ui.recovery_setup import load_setup, validate_setup

    setup = load_setup()
    inputs = validate_setup(setup)
    context = EnvironmentProductContext(inputs["scene"], inputs["product_order"], inputs["geometry"],
                                        setup["permitted_resources"])
    part = "KET4_Square_4mm"
    context.resources["KMR"].valuation.update(resource_state="carrying", held_part=part, resource_location=None)
    context.part_tracker[part].update(location="KMR")
    owner = SimpleNamespace(agent_name="KMR", _primitive_state={"held_part": None, "resource_location": None,
                                                               "base_pose": [-3.0, -0.12, 0.0]},
                            _released_part={"part_name": part, "pose": pose, "settled": True})
    task = {"part_name": part, "outline_id": "RECOVERY_SEQ5", "macro_name": "release_part", "expected_end_state": {}}
    return part, context, SimpleNamespace(context=context, agent=SimpleNamespace()), owner, task


def test_KMR_release_onto_the_belt_is_recorded_where_the_part_rests():
    from cais_spade_llm.recovery_framework.slippage_execution import _reconcile_mobile_completion

    part, context, runtime, owner, task = _kmr_release_fixture([-3.0, 0.5, 1.016, 0.0, 0.0, 0.0, 1.0])
    observed = asyncio.run(_reconcile_mobile_completion(runtime, owner, task, "kmr-release", {}))
    belt = context.resources["Conveyor"].valuation
    assert belt[f"part_location.{part}"] == "after loading_position_1" and belt[f"part_order.{part}"] == 0
    assert context.part_tracker[part]["location"] == "Conveyor" and "state" not in context.part_tracker[part]
    assert context.part_tracker[part]["observed_pose"]["x"] == -3.0
    assert observed["released"]["receiver"] == "Conveyor"
    assert context.resources["KMR"].valuation["held_part"] is None


def test_KMR_release_into_a_buffer_zone_and_onto_an_undeclared_surface():
    from cais_spade_llm.recovery_framework.slippage_execution import _reconcile_mobile_completion

    part, context, runtime, owner, task = _kmr_release_fixture([-0.56, 0.5, 1.017, 0.0, 0.0, 0.0, 1.0])
    asyncio.run(_reconcile_mobile_completion(runtime, owner, task, "kmr-release", {}))
    assert context.resources["Buffer For Machined parts"].valuation["zone_2_part"] == part
    assert context.part_tracker[part]["location"] == "Buffer For Machined parts"

    part, context, runtime, owner, task = _kmr_release_fixture([0.25, 0.45, 1.025, 0.0, 0.0, 0.0, 1.0])
    asyncio.run(_reconcile_mobile_completion(runtime, owner, task, "kmr-release", {}))
    assert context.part_tracker[part]["location"] is None and "state" not in context.part_tracker[part]
    assert context.part_tracker[part]["observed_pose"] == {"x": 0.25, "y": 0.45, "z": 1.025, "qx": 0.0, "qy": 0.0,
                                                           "qz": 0.0, "qw": 1.0}


@pytest.mark.parametrize("pose,receiver,check", [
    ([-4.13, 1.82, 1.06, 0.0, 0.0, 0.0, 1.0], "M1",
     lambda context, part: context.resources["M1"].valuation["part_name"] == part
     and context.resources["M1"].valuation["resource_state"] == "loaded"),
    ([-4.6, 1.1, 1.0, 0.0, 0.0, 0.0, 1.0], "M1",
     lambda context, part: context.resources["M1"].valuation["staging_part"] == part
     and context.part_tracker[part]["location"] == "M1 staging tray"),
    ([-6.25, 1.88, 0.645, 0.0, 0.0, 0.0, 1.0], "Storage",
     lambda context, part: context.resources["Storage"].valuation[f"inventory.{part}"] is True),
])
def test_KMR_release_uses_the_receiving_owner_declared_consequence(pose, receiver, check):
    from cais_spade_llm.recovery_framework.slippage_execution import _reconcile_mobile_completion

    part, context, runtime, owner, task = _kmr_release_fixture(pose)
    context.resources["Storage"].valuation[f"inventory.{part}"] = False
    context.resources["M1"].valuation.update(resource_state="idle", part_name=None)
    observed = asyncio.run(_reconcile_mobile_completion(runtime, owner, task, "kmr-release", {}))
    assert observed["released"]["receiver"] == receiver and check(context, part)
    assert owner._released_part is None


def test_KMR_release_still_moving_when_observation_ended_is_located_only_by_its_pose():
    from cais_spade_llm.recovery_framework.slippage_execution import _reconcile_mobile_completion

    part, context, runtime, owner, task = _kmr_release_fixture([-0.56, 0.5, 1.017, 0.0, 0.0, 0.0, 1.0])
    owner._released_part["settled"] = False
    asyncio.run(_reconcile_mobile_completion(runtime, owner, task, "kmr-release", {}))
    assert context.part_tracker[part]["location"] is None
    assert context.resources["Buffer For Machined parts"].valuation["zone_2_part"] is None


@pytest.mark.parametrize("location", ["Storage", None])
def test_KMR_pickup_from_Storage_or_of_a_misplaced_part(location):
    from cais_spade_llm.recovery_framework.slippage_execution import _reconcile_mobile_completion

    part, context, runtime, owner, task = _kmr_release_fixture([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    context.resources["KMR"].valuation.update(resource_state="idle", held_part=None)
    context.resources["Storage"].valuation[f"inventory.{part}"] = location is not None
    context.part_tracker[part].update(state="ready" if location else "misplaced", location=location)
    owner._primitive_state = {"held_part": part, "resource_location": "Storage" if location else None}
    asyncio.run(_reconcile_mobile_completion(runtime, owner, {**task, "macro_name": "pick_part"}, "kmr-pick", {}))
    assert context.part_tracker[part]["location"] == "KMR"
    assert context.resources["Storage"].valuation[f"inventory.{part}"] is False
    with pytest.raises(ValueError, match="not the generated event"):
        part, context, runtime, owner, task = _kmr_release_fixture([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        context.resources["KMR"].valuation.update(resource_state="idle", held_part=None)
        owner._primitive_state = {"held_part": part, "resource_location": "Storage"}
        asyncio.run(_reconcile_mobile_completion(runtime, owner, {**task, "part_name": "RGOCG4-50_Round_4mm"},
                                                 "kmr-pick", {}))


def test_KMR_release_into_an_occupied_zone_or_onto_a_moving_belt_is_not_committed():
    from cais_spade_llm.recovery_framework.slippage_execution import _reconcile_mobile_completion

    part, context, runtime, owner, task = _kmr_release_fixture([-0.56, 0.5, 1.017, 0.0, 0.0, 0.0, 1.0])
    context.resources["Buffer For Machined parts"].valuation["zone_2_part"] = "RGOCG4-50_Round_4mm"
    with pytest.raises(ValueError, match="already occupied"):
        asyncio.run(_reconcile_mobile_completion(runtime, owner, task, "kmr-release", {}))
    part, context, runtime, owner, task = _kmr_release_fixture([-3.0, 0.5, 1.016, 0.0, 0.0, 0.0, 1.0])
    context.resources["Conveyor"].valuation["belt_stopped"] = False
    with pytest.raises(ValueError, match="stopped and unreserved"):
        asyncio.run(_reconcile_mobile_completion(runtime, owner, task, "kmr-release", {}))
    assert context.transitions == [] and context.part_tracker[part]["location"] == "KMR"


@pytest.mark.parametrize("observed_x", [.2, .4])
def test_assembly_completion_requires_observed_configured_placement(monkeypatch, observed_x):
    from cais_spade_llm.recovery_framework.slippage_execution import _reconcile_completion

    part, rid = "KET4_Square_4mm", "ur5e-3"
    actor = SimpleNamespace(valuation={"resource_state": "picked", "held_part": part}, revision=0)
    context = SimpleNamespace(
        run_id="trial", resources={rid: actor},
        models={rid: {"state_variables": {"resource_location": {"domain": ["assembly_board-v1"]}}}},
        geometry={part: {"model_name": part, "target": "square_peg_4mm"}}, product_name="assembly_board-v1",
        part_tracker={part: {"location": rid, "current_process": None, "processCompleted": []}},
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
            "expected_end_state": {"part_location": "assembly_board-v1", "resource_location": "assembly_board-v1"}}
    if observed_x == .4:
        with pytest.raises(ValueError, match="not observed"):
            asyncio.run(_reconcile_completion(runtime, owner, task, "native-task", {}))
        assert context.part_tracker[part]["location"] == rid
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
        with pytest.raises(ValueError, match="unavailable"):
            recovery.action("regenerate_outline", feedback="Keep the observed custody")
        _accept_outline(recovery)
        recovery.action("continue_outline")
        await recovery._task
        original = recovery.snapshot()["artifact_directory"]
        with pytest.raises(ValueError, match="does not authorize execution"):
            recovery.action("execute")
        recovery.action("regenerate_primitives", feedback="Review the destination")
        await recovery._task
        state = recovery.snapshot()
        assert state["status"] == "ready_to_execute"
        assert state["artifact_directory"] != original
        assert (Path(original) / "proposal.json").is_file()
        assert recovery._prepared["recovery_feedback"] == "Review the destination"
        assert state["execution_authorized"] is False
    asyncio.run(run())


def test_stop_cancels_generation_and_rejects_stale_actions(recovery, monkeypatch):
    started = asyncio.Event()

    async def pending(stage):
        started.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(recovery, "_run_stage", pending)

    async def run():
        recovery.begin({"injection_status": "completed"})
        await recovery._task
        _accept_outline(recovery)
        recovery.action("continue_outline")
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


def test_stage_stop_keeps_the_accepted_outline_for_a_new_gazebo_composition(recovery, monkeypatch):
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes import multi_turn

    started = asyncio.Event()
    finish = multi_turn.execute_multi_turn_recovery.side_effect

    async def pending(planner, prepared, *, session_state):
        if prepared["_stop_after_multi_turn_phase"] == "outline":
            return await finish(planner, prepared, session_state=session_state)
        started.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(multi_turn, "execute_multi_turn_recovery", AsyncMock(side_effect=pending))
    recovery._state.update(outline_mode="auto", primitive_mode="auto")

    async def run():
        recovery.begin({"injection_status": "completed"})
        await recovery._task
        _accept_outline(recovery)
        recovery.action("continue_outline")
        await started.wait()
        outline = deepcopy(recovery._outline_request)
        stopped_directory = recovery.snapshot()["artifact_directory"]
        assert recovery.generating_stage() == "primitive"
        recovery.stop_generation()
        await asyncio.gather(recovery._task, return_exceptions=True)
        state = recovery.snapshot()
        assert (state["status"], state["primitive_status"]) == ("outline_review", "stopped")
        assert state["outline"] == outline["multi_turn_session_state"]["accepted_outline_prefix"]
        assert recovery._outline_request == outline
        composed = []
        generate_primitives = recovery._generate_primitives

        async def counted():
            composed.append(True)
            await generate_primitives()

        monkeypatch.setattr(recovery, "_generate_primitives", counted)
        multi_turn.execute_multi_turn_recovery.side_effect = finish
        # Gazebo Auto composes inside Continue; Generate must not compose twice.
        recovery.action("generate_primitives")
        await recovery._task
        assert composed == [True]
        assert recovery.snapshot()["status"] == "ready_to_execute"
        assert recovery.snapshot()["artifact_directory"] != stopped_directory
    asyncio.run(run())


def test_stage_stop_during_offline_composition_drains_and_renews_cancellation(recovery, monkeypatch):
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes import multi_turn

    started, release = threading.Event(), threading.Event()

    def preview(prepared, state):
        started.set()
        assert release.wait(5)
        return {}, "Not sent"

    monkeypatch.setattr(multi_turn, "_build_phase_prompt", preview)

    async def run():
        await recovery._prepare()
        recovery.offline = True
        recovery._outline_request = deepcopy(recovery._prepared)
        original = deepcopy(recovery._outline_request)
        recovery._state.update(status="outline_review", primitive_mode="manual")
        recovery.action("generate_primitives")
        try:
            assert await asyncio.to_thread(started.wait, 5)
            recovery.stop_generation()
            cancel = recovery._offline_cancel
            assert cancel.is_set()
            assert recovery.snapshot()["primitive_status"] == "preparing"
        finally:
            release.set()
        await asyncio.gather(recovery._task, return_exceptions=True)
        state = recovery.snapshot()
        assert (state["status"], state["primitive_status"]) == ("outline_review", "stopped")
        assert recovery._outline_request == original
        assert recovery._stopped is False
        assert recovery._offline_cancel is not cancel and not recovery._offline_cancel.is_set()
        monkeypatch.setattr(multi_turn, "_build_phase_prompt", lambda prepared, state: ({}, "Not sent"))
        recovery.action("generate_primitives")
        await recovery._task
        assert recovery.snapshot()["status"] == "offline_completed"
        assert recovery.agent._composition_cancel_event is recovery._offline_cancel
    asyncio.run(run())


def test_context_change_blocks_execution_and_generation_errors_are_retained(recovery, monkeypatch):
    execute = AsyncMock()
    monkeypatch.setattr("cais_spade_llm.recovery_framework.slippage_execution.execute_generated_recovery", execute)

    async def run():
        recovery.begin({"injection_status": "completed"})
        await recovery._task
        _accept_outline(recovery)
        recovery.action("continue_outline")
        await recovery._task
        recovery._state.pop("composition_validation_version")
        recovery._proposal.pop("composition_validation_version", None)
        recovery.runtime.context.revision += 1
        recovery.action("execute")
        await recovery._task
        assert recovery.snapshot()["status"] == "blocked"
        execute.assert_not_called()
        monkeypatch.setattr(recovery, "_run_stage", AsyncMock(side_effect=ValueError("captured error")))
        recovery.action("regenerate_primitives")
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
        assert recovery.snapshot()["status"] == "outline_ready"
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
        # The legacy schema reads declared types; the contract-2 schema gives only the event's form.
        multi_turn._get_response_schema("outline", {
            "outline_mode": "incremental_validated",
            "recovery_des_models": {"recovery-resource-6@localhost": {"state_variables": {
                "resource_state": {"scope": "resource"},
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
    if scenario in {"Machining breakdown during part processing", "ur5e-1 breakdown"}:
        # Both trims run, so the three-part order includes RGOCG4 on M2.
        assert {"M2", "ur5e-2"}.issubset(setup["permitted_resources"])
        assert {"M2", "ur5e-2"}.issubset(setup["recovery_permitted_resources"])
        assert setup["selected_product_order_file"].endswith("assembly_board-v1-machine-breakdown.json")
    elif scenario == "Conveyor breakdown":
        assert not {"M2", "ur5e-2"}.intersection(setup["permitted_resources"])
        assert {"M2", "ur5e-2"}.issubset(setup["recovery_permitted_resources"])
    assert setup["nominal_permitted_resources"] == setup["permitted_resources"]
    setup["execution_mode"] = "physical"
    assert not diagnostic_setup(setup)


def test_default_observations_are_schema_2_captures_of_their_own_scenario():
    from cais_spade_llm.recovery_framework.part_slippage_example import default_observation_paths
    from cais_spade_llm.ui.recovery_setup import FAILURE_SCENARIOS

    defaults = default_observation_paths()
    if not defaults:
        pytest.skip("retained captures are not present")
    assert set(defaults) <= set(FAILURE_SCENARIOS)
    for scenario, path in defaults.items():
        raw = json.loads(Path(path).read_text())
        assert (raw["scenario"], raw["schema_version"]) == (scenario, 2)


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
    from cais_spade_llm.recovery_framework.recovery_case_safety import default_safety_files

    expected = default_safety_files("Part slippage")
    assert expected[0] == "cais_spade_llm/specification/safety/safety_assembly_board-v1_mutex.txt"
    example = PartSlippageExample(SimpleNamespace(system_running=False))
    assert example.snapshot()["selected_safety_files"] == expected
    setup = example_setup()
    assert setup["selected_safety_files"] == expected
    assert setup["selected_safety_file"] == "__NONE__" and diagnostic_setup(setup)
    assert example_setup(safety_files=[])["selected_safety_files"] == []
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
    assert selected["selected_safety_files"] == [absolute_mutex, *expected[1:]]


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
    from cais_spade_llm.agents.shared_information.llm_request_records import (
        RequestRecords,
        capture_requests,
    )

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
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes import multi_turn
    from cais_spade_llm.recovery_framework import ROOT
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
    # A part is stated by where it is and its process; a recorded handling label is not passed on.
    assert "state" not in captured["failure_context"]["failure_context"]["affected_entities"][0]
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
        assert recovery.snapshot()["status"] == "outline_ready"
        _accept_outline(recovery)
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
    from cais_spade_llm.recovery_framework.recovery_observations import (
        capture_observations,
        offline_preparation_missing,
    )

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
        _accept_outline(recovery)
        recovery.action("regenerate_primitives", feedback="Recorded operator feedback")
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
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes.multi_turn import (
        _build_phase_prompt,
    )

    payload, prompt = _build_phase_prompt({"recovery_feedback": "Keep gear_small custody observed"},
                                          {"current_phase": "grounding"})
    assert payload["recovery_feedback"] == "Keep gear_small custody observed"
    assert "Operator feedback:\nKeep gear_small custody observed" in prompt


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
