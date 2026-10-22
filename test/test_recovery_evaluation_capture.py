from __future__ import annotations

"""Capture-only isolation and factual readiness for the four existing failures."""

import ast
import asyncio
import hashlib
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from cais_spade_llm.recovery_framework import evaluation_capture as capture


def checkpoint_motion_fixture():
    """Local cancellation fixture; never supplied to recovery generation."""
    worker = SimpleNamespace(cancel=AsyncMock(), last_result={"cancelled": True})
    machine_worker = SimpleNamespace(cancel=AsyncMock(), machine_control=AsyncMock())
    controller = SimpleNamespace(_cancel_simulation_goal=Mock())
    declarations = {
        "KMR": {"model": "KMR", "root_link": "KMR_base_link", "controller_node": "/KMR/arm"},
        "ur5e-2": {"model": "dual_robot", "root_link": "ur5e_2_base_link", "controller_node": "/ur5e_2/arm"},
        "M2": {"model": "M2"}}
    runtime = SimpleNamespace(setup={"execution_mode": "simulation"}, stopped=True,
        recovery_checkpoint_capturing=True, part_slippage_recovery=SimpleNamespace(_stopped=False),
        context=SimpleNamespace(run_id="run", revision=3,
            pending_tasks={"task": {"resource_id": "KMR", "target_resource": "Storage"}},
            inputs={"scene": {"machines": [{"resource_id": "M2"}], "safety_preparation": {
                "resources": declarations, "outline_preview": {"timeout_seconds": 1.}}}}),
        resource_agents=[SimpleNamespace(agent_name="KMR", workflow_worker=worker),
                         SimpleNamespace(agent_name="M2", workflow_worker=machine_worker),
                         SimpleNamespace(agent_name="ur5e-2", _controller=controller)])
    native = {"instance_id": "world", "simulation_time": 12., "world_paused": False,
        "models": {model: {"links": {link: {"pose": [1., 2., 0., 0., 0., 0., 1.],
            "linear_velocity": [0., 0., 0.], "angular_velocity": [0., 0., 0.]}}}
            for model, link in (("KMR", "KMR_base_link"), ("dual_robot", "ur5e_2_base_link"))}}
    reader = SimpleNamespace(idle_goals=Mock(return_value={"KMR": {"holding": True}}),
        snapshot=Mock(return_value=native), set_world_paused=Mock())
    return runtime, reader, worker, machine_worker, controller


def test_checkpoint_cancels_handling_before_freeze_without_machine_job_commands(tmp_path):
    from cais_spade_llm.recovery_framework.recovery_observations import _cancel_checkpoint_motion

    runtime, reader, worker, machine_worker, controller = checkpoint_motion_fixture()
    pending = deepcopy(runtime.context.pending_tasks)

    def idle():
        worker.cancel.assert_awaited_once()
        controller._cancel_simulation_goal.assert_called_once()
        return {"KMR": {"holding": True}}

    reader.idle_goals.side_effect = idle
    result = asyncio.run(_cancel_checkpoint_motion(runtime, reader, tmp_path))
    assert result["status"] == "observed_stopped"
    assert result["native_observation"]["world_paused"] is False
    assert result["pending_tasks"] == runtime.context.pending_tasks == pending
    assert result["machine_commands"] == []
    assert result["resources"]["KMR"]["worker_result"] == {"cancelled": True}
    machine_worker.cancel.assert_not_called()
    machine_worker.machine_control.assert_not_called()
    reader.set_world_paused.assert_not_called()
    assert json.loads((tmp_path / "checkpoint_motion_cancellation.json").read_text()) == result


def test_checkpoint_waits_for_observed_stop_after_cancellation_acknowledgement(tmp_path):
    from cais_spade_llm.recovery_framework.recovery_observations import _cancel_checkpoint_motion

    runtime, reader, *_ = checkpoint_motion_fixture()
    stopped = reader.snapshot.return_value
    moving = deepcopy(stopped)
    moving["models"]["KMR"]["links"]["KMR_base_link"]["linear_velocity"][0] = .4
    reader.snapshot.side_effect = [moving, stopped]
    result = asyncio.run(_cancel_checkpoint_motion(runtime, reader, tmp_path))
    assert reader.snapshot.call_count == 2
    assert result["native_observation"] == stopped
    assert "last_observation_error" not in result


def test_checkpoint_records_fixed_root_description_and_unmodified_native_motion(tmp_path):
    from cais_spade_llm.recovery_framework.recovery_observations import _cancel_checkpoint_motion

    runtime, reader, _, machine_worker, _ = checkpoint_motion_fixture()
    declaration = runtime.context.inputs["scene"]["safety_preparation"]["resources"]["ur5e-2"]
    declaration.update(description_node="/robot_state_publisher", stationary_linear_speed_max=.001,
                       stationary_angular_speed_max=.001)
    description = ('<robot name="fixture"><link name="world"/><link name="ur5e_2_base_link"/>'
                   '<joint name="fixture_mount" type="fixed"><parent link="world"/>'
                   '<child link="ur5e_2_base_link"/></joint></robot>')
    reader.parameter = Mock(return_value=description)
    native = reader.snapshot.return_value
    root = native["models"]["dual_robot"]["links"]["ur5e_2_base_link"]
    # These fixed-root magnitudes reproduce the rejected Gazebo checkpoint.
    root.update(linear_velocity=[-5.83e-8, -1.04e-7, -3.0827e-6],
                angular_velocity=[.00037117, .00092689, -1.527e-6])
    before = deepcopy(native)
    result = asyncio.run(_cancel_checkpoint_motion(runtime, reader, tmp_path))
    assert result["status"] == "observed_stopped"
    assert result["native_observation"] == native == before
    assert result["robot_descriptions"] == {"/robot_state_publisher": description}
    assert result["resources"]["ur5e-2"]["root_motion"] == {
        "root_link": "ur5e_2_base_link", "fixed_to_world": True,
        "linear_speed_max": .001, "angular_speed_max": .001,
        "description_sha256": hashlib.sha256(description.encode()).hexdigest(),
        "description_node": "/robot_state_publisher"}
    reader.parameter.assert_called_once_with("/robot_state_publisher", "robot_description")
    machine_worker.cancel.assert_not_called()
    machine_worker.machine_control.assert_not_called()
    reader.set_world_paused.assert_not_called()


@pytest.mark.parametrize("velocity", [[.0008, .0008, 0.], [.002, 0., 0.],
                                     [float("nan"), 0., 0.], [False, 0., 0.]])
def test_checkpoint_fixed_root_still_requires_declared_speed_bound(tmp_path, velocity):
    from cais_spade_llm.recovery_framework.recovery_observations import _cancel_checkpoint_motion

    runtime, reader, *_ = checkpoint_motion_fixture()
    configuration = runtime.context.inputs["scene"]["safety_preparation"]
    configuration["outline_preview"]["timeout_seconds"] = 0.
    configuration["resources"]["ur5e-2"].update(description_node="/robot_state_publisher",
        stationary_linear_speed_max=.001, stationary_angular_speed_max=.001)
    reader.parameter = Mock(return_value=(
        '<robot name="fixture"><link name="world"/><link name="ur5e_2_base_link"/>'
        '<joint name="fixture_mount" type="fixed"><parent link="world"/>'
        '<child link="ur5e_2_base_link"/></joint></robot>'))
    reader.snapshot.return_value["models"]["dual_robot"]["links"]["ur5e_2_base_link"][
        "angular_velocity"] = velocity
    with pytest.raises(ValueError, match="Handling root has not stopped: ur5e-2/angular_velocity"):
        asyncio.run(_cancel_checkpoint_motion(runtime, reader, tmp_path))
    assert json.loads((tmp_path / "checkpoint_motion_cancellation.json").read_text())["status"] == "failed"


def test_checkpoint_uses_mobile_physics_description_instead_of_planning_mount(tmp_path):
    from cais_spade_llm.recovery_framework.recovery_observations import _cancel_checkpoint_motion

    runtime, reader, *_ = checkpoint_motion_fixture()
    configuration = runtime.context.inputs["scene"]["safety_preparation"]
    configuration["outline_preview"]["timeout_seconds"] = 0.
    configuration["resources"]["KMR"].update(description_node="/planning_description",
        physics_description_node="/KMR/robot_state_publisher",
        stationary_linear_speed_max=.001, stationary_angular_speed_max=.001)
    reader.parameter = Mock(return_value='<robot name="fixture"><link name="KMR_base_link"/></robot>')
    reader.snapshot.return_value["models"]["KMR"]["links"]["KMR_base_link"]["linear_velocity"][0] = 1e-9
    with pytest.raises(ValueError, match="Handling root has not stopped: KMR/linear_velocity"):
        asyncio.run(_cancel_checkpoint_motion(runtime, reader, tmp_path))
    reader.parameter.assert_called_once_with("/KMR/robot_state_publisher", "robot_description")
    result = json.loads((tmp_path / "checkpoint_motion_cancellation.json").read_text())
    assert result["resources"]["KMR"]["root_motion"]["linear_speed_max"] == 0.
    assert result["resources"]["KMR"]["root_motion"]["fixed_to_world"] is False


@pytest.mark.parametrize("description", [
    '<robot',
    '<robot><link name="root"/><link name="root"/></robot>',
    '<robot><link name="root"/><joint type="fixed"><parent link="world"/><child link="root"/></joint></robot>',
    '<robot><link name="root"/><joint type="fixed"><parent link="root"/><child link="root"/></joint></robot>',
    '<robot><link name="root"/><link name="world"/>'
    '<joint type="fixed"><parent link="world"/><child link="root"/></joint>'
    '<joint type="fixed"><parent link="world"/><child link="root"/></joint></robot>',
])
def test_checkpoint_fixed_root_requires_unambiguous_complete_description(description):
    from cais_spade_llm.recovery_framework.recovery_observations import _checkpoint_root_motion

    with pytest.raises(ValueError):
        _checkpoint_root_motion({"root_link": "root", "stationary_linear_speed_max": .001,
                                 "stationary_angular_speed_max": .001}, description)


@pytest.mark.parametrize("defect", ["moving", "missing_velocity", "bool_velocity", "frozen", "active_goal"])
def test_checkpoint_does_not_mistake_acknowledgement_or_frozen_time_for_stop(tmp_path, defect):
    from cais_spade_llm.recovery_framework.recovery_observations import _cancel_checkpoint_motion

    runtime, reader, *_ = checkpoint_motion_fixture()
    runtime.context.inputs["scene"]["safety_preparation"]["outline_preview"]["timeout_seconds"] = 0.
    root = reader.snapshot.return_value["models"]["KMR"]["links"]["KMR_base_link"]
    if defect == "moving":
        root["linear_velocity"][0] = .1
    elif defect == "missing_velocity":
        root.pop("angular_velocity")
    elif defect == "bool_velocity":
        root["angular_velocity"][2] = False
    elif defect == "frozen":
        reader.snapshot.return_value["world_paused"] = True
    else:
        reader.idle_goals.side_effect = ValueError("Controller has an active goal")
    with pytest.raises(ValueError):
        asyncio.run(_cancel_checkpoint_motion(runtime, reader, tmp_path))
    result = json.loads((tmp_path / "checkpoint_motion_cancellation.json").read_text())
    assert result["status"] == "failed" and result["error"]
    reader.set_world_paused.assert_not_called()


@pytest.mark.parametrize("defect", ["hardware", "dispatch", "operator_stop", "owner"])
def test_checkpoint_cancellation_respects_run_ownership_and_operator_stop(tmp_path, defect):
    from cais_spade_llm.recovery_framework.recovery_observations import _cancel_checkpoint_motion

    runtime, reader, worker, machine_worker, _ = checkpoint_motion_fixture()
    if defect == "hardware":
        runtime.setup["execution_mode"] = "hardware"
    elif defect == "dispatch":
        runtime.stopped = False
    elif defect == "operator_stop":
        runtime.part_slippage_recovery._stopped = True
    else:
        runtime.resource_agents = [owner for owner in runtime.resource_agents if owner.agent_name != "KMR"]
    with pytest.raises(asyncio.CancelledError if defect == "operator_stop" else ValueError):
        asyncio.run(_cancel_checkpoint_motion(runtime, reader, tmp_path))
    machine_worker.cancel.assert_not_called()
    if defect in {"hardware", "dispatch", "owner"}:
        worker.cancel.assert_not_called()


def test_machine_capture_freezes_after_handling_stop_and_retains_both_times(tmp_path, monkeypatch):
    from cais_spade_llm.recovery_framework import recovery_observations as observations

    runtime, reader, worker, machine_worker, _ = checkpoint_motion_fixture()
    runtime.part_slippage_recovery.path = tmp_path
    sequence = []
    native = reader.snapshot.return_value
    fault_native = deepcopy(native)
    fault_native["models"]["KMR"]["links"]["KMR_base_link"]["linear_velocity"][0] = .3

    async def stop_motion():
        sequence.append("cancel_handling")

    def freeze(paused):
        assert paused is True and sequence == ["cancel_handling"]
        sequence.append("freeze")
        return {**deepcopy(native), "world_paused": True, "simulation_time": 13.}

    frozen = {**deepcopy(native), "world_paused": True, "simulation_time": 13.}
    reader.snapshot.side_effect = [fault_native, native, frozen]
    reader.set_world_paused.side_effect = freeze
    worker.cancel.side_effect = stop_motion
    provider = SimpleNamespace(reader=reader, initialize=Mock())
    runtime.recovery_observation_preparation = provider
    jobs = {"run_id": "run", "resource_id": "M2", "active_job": "circle-job", "jobs": {"circle-job": {
        "job_state": "running", "process_completed": False}}, "simulation_clock_sec": 13.}

    async def observe_job(**request):
        assert sequence == ["cancel_handling", "freeze"]
        sequence.append("observe_job")
        return {"status": "observed", "command_sent": False, "checkpoint_id": request["checkpoint_id"],
            "processing_observations": deepcopy(jobs), "native_source": {
                "instance_id": "world", "simulation_time": 13., "world_paused": True}}

    machine_worker.observe_machine_checkpoint = observe_job
    fault = {"resource_id": "M1", "actual_interruption_simulation_sec": 11.5,
        "processing_observations": {"M2": {"simulation_clock_sec": 11.5, "job_state": "running"}}}

    def captured(_runtime, evidence):
        sequence.append("capture")
        return {"run_id": "run", "context_revision": 3, "jids": {"M2": "M2@localhost"},
            "processing_observations": deepcopy(evidence["checkpoint_processing_observations"]),
            "physical_evidence": {"resources": {"M2": {"simulation_time": 13.,
                "native_physics_source": {"instance_id": "world"}}}}}

    async def scene(**kwargs):
        return {"scene_fingerprint": "scene"}

    runtime.context.inputs["scene"]["safety_preparation"]["outline_preview"].update(
        planning_scene_service="/recorded/scene", preview_parameter_service="/recorded/parameters")
    monkeypatch.setattr(observations, "capture_observations", captured)
    monkeypatch.setattr(observations, "recorded_processing_fields", Mock())
    monkeypatch.setattr("cais_spade_llm.recovery_framework.recovery_motion_preview.capture_preview_scene", scene)
    result = asyncio.run(observations.capture_machine_checkpoint(runtime, fault))
    assert sequence == ["cancel_handling", "freeze", "observe_job", "capture"]
    assert result["capture_checkpoint"]["fault_simulation_time"] == 11.5
    assert result["capture_checkpoint"]["simulation_time"] == 13.
    assert result["failure_processing_observations"] == fault["processing_observations"]
    assert result["processing_observations"]["M2"] == jobs
    assert result["checkpoint_motion_cancellation"]["status"] == "observed_stopped"
    assert observations.processing_capture_compatibility(result)["status"] == "supported"
    original = json.loads((tmp_path / "checkpoint_before_motion_cancellation.json").read_text())
    assert original["native_observation"] == fault_native
    machine_worker.cancel.assert_not_called()
    machine_worker.machine_control.assert_not_called()
    assert runtime.context.pending_tasks["task"]["target_resource"] == "Storage"


@pytest.mark.parametrize('scenario', list(capture.CHECKPOINTS))
def test_capture_verification_requires_exact_reconciled_checkpoint(scenario, monkeypatch):
    monkeypatch.setattr(capture, '_verify_robot_breakdown_capture', Mock())
    row = {'scenario': scenario, 'run_id': 'owned-run', 'part_tracker': {'recorded-part': {}},
           'setup': {'execution_mode': 'simulation', 'failure_scenario': {
               'scenario': scenario, 'checkpoint': capture.CHECKPOINTS[scenario], 'resource_id': 'recorded-owner'}},
           'failure': {'checkpoint': capture.CHECKPOINTS[scenario], 'run_id': 'owned-run',
                       'injection_status': 'completed', 'part_name': 'recorded-part'}}
    original = deepcopy(row)
    result = capture.verify_capture(row, scenario)
    assert result['checkpoint'] == capture.CHECKPOINTS[scenario]
    assert result['source_paths'] == ['/failure', '/setup/failure_scenario', '/part_tracker/recorded-part']
    assert row == original
    for key, value in [('checkpoint', 'another-checkpoint'), ('run_id', 'another-run'),
                       ('injection_status', 'requested'), ('part_name', None)]:
        changed = deepcopy(row)
        changed['failure'][key] = value
        with pytest.raises(ValueError):
            capture.verify_capture(changed, scenario)


def _robot_breakdown_capture_record():
    part = 'KET4_Square_4mm'
    records = {}
    for rid, name in (('M1', 'machine_part'), ('ur5e-1', 'pick_approach')):
        records[name] = {'acknowledgement': {'resource_id': rid, 'event_name': name,
            'run_id': 'owned-run', 'task_id': name, 'evidence': 'resource',
            'parameters': {'part_name': part, 'origin_resource_location': 'M1'}},
            'before': {'M1': {'resource_state': 'completed', 'part_name': part}},
            'observations': {'controller_result': {'status': 'completed'}}}
    row = {'scenario': 'ur5e-1 breakdown', 'run_id': 'owned-run', 'context_revision': 3,
        'setup': {'execution_mode': 'simulation', 'failure_scenario': {
            'scenario': 'ur5e-1 breakdown', 'checkpoint': 'after_M1_processing_before_pick',
            'resource_id': 'ur5e-1'}},
        'failure': {'checkpoint': 'after_M1_processing_before_pick', 'run_id': 'owned-run',
            'injection_status': 'completed', 'part_name': part, **deepcopy(records),
            'resource_values_before': {'ur5e-1': {'resource_state': 'at_pick', 'held_part': None}}},
        'part_tracker': {part: {'location': 'M1'}},
        'resource_states': {'M1': {'resource_state': 'completed', 'part_name': part},
                            'ur5e-1': {'resource_state': 'failed', 'held_part': None}},
        'transitions': list(records.values()), 'unavailable_resources': [], 'pending_nodes': [],
        'physical_evidence': {'snapshot': {'resources': {'ur5e-1': {'held_part': None}}},
            'resources': {'ur5e-1': {'custody_complete': True}}, 'geometry': {}, 'resource_models': {},
            'physics_evidence': {'instance_id': 'world', 'simulation_time': 84., 'world_paused': True}},
        'capture_checkpoint': {'instance_id': 'world', 'simulation_time': 84., 'world_paused': True,
                               'native_observations_unchanged': True}}
    return row


@pytest.mark.parametrize('defect', [None, 'state', 'custody', 'physical_custody', 'location', 'origin',
    'run', 'observations', 'source_record', 'unavailable', 'frozen', 'time', 'mutation', 'occupancy'])
def test_robot_capture_requires_failed_M1_approach_and_frozen_occupancy(monkeypatch, defect):
    from cais_spade_llm.recovery_framework.recovery_observations import checkpoint_fact_fingerprint
    from cais_spade_llm.resources import recovery_geometry

    row = _robot_breakdown_capture_record()
    approach = row['failure']['pick_approach']
    if defect == 'state':
        row['resource_states']['ur5e-1']['resource_state'] = 'idle'
    elif defect == 'custody':
        row['resource_states']['ur5e-1']['held_part'] = 'KET4_Square_4mm'
    elif defect == 'physical_custody':
        row['physical_evidence']['resources']['ur5e-1']['custody_complete'] = False
    elif defect == 'location':
        row['part_tracker']['KET4_Square_4mm']['location'] = 'ur5e-1'
    elif defect == 'origin':
        approach['acknowledgement']['parameters']['origin_resource_location'] = 'Storage'
    elif defect == 'run':
        approach['acknowledgement']['run_id'] = 'previous-run'
    elif defect == 'observations':
        approach['observations'] = {}
    elif defect == 'source_record':
        row['transitions'] = []
    elif defect == 'unavailable':
        row['unavailable_resources'] = ['ur5e-1']
    elif defect == 'frozen':
        row['physical_evidence']['physics_evidence']['world_paused'] = False
    elif defect == 'time':
        row['physical_evidence']['physics_evidence']['simulation_time'] = 85.
    row['capture_checkpoint']['facts_fingerprint'] = checkpoint_fact_fingerprint(row)
    if defect == 'mutation':
        row['physical_evidence']['snapshot']['changed'] = True
    monkeypatch.setattr(recovery_geometry, '_checkpoint', lambda *_args, **_kwargs: {
        'region_occupancy': {'M1.front_access': {'ur5e-1': defect != 'occupancy'}}})
    if defect is None:
        assert capture.verify_capture(row, 'ur5e-1 breakdown')['part_name'] == 'KET4_Square_4mm'
    else:
        with pytest.raises(ValueError):
            capture.verify_capture(row, 'ur5e-1 breakdown')


@pytest.mark.parametrize('defect', [None, 'source', 'condition', 'm1_active', 'm2_finished', 'kmr_moving', 'kmr_holding'])
def test_gripper_fault_capture_requires_overlap_reporting_device_and_idle_KMR(monkeypatch, defect):
    from cais_spade_llm.recovery_framework.recovery_observations import checkpoint_fact_fingerprint
    from cais_spade_llm.resources import recovery_geometry

    part = 'KET4_Square_4mm'
    row = _robot_breakdown_capture_record()
    row['capture_checkpoint']['machine_observations'] = {'M1': {'status': 'observed'}, 'M2': {'status': 'observed'}}
    row['fault_reports'] = {'ur5e-1': {'reported_state': 'failed', 'source': 'gripper object detection',
                                       'gripper': 'not working (hardware fault, needs repair)'}}
    row['processing_observations'] = {
        'M1': {'active_job': None, 'jobs': {'trim-1': {'part_name': part, 'process_completed': True,
                                                         'job_state': 'completed'}}},
        'M2': {'active_job': 'trim-2', 'jobs': {'trim-2': {'part_name': 'RGOCG4-50_Round_4mm',
                                                            'process_completed': False, 'job_state': 'running'}}}}
    row['resource_states']['KMR'] = {'resource_state': 'idle', 'held_part': None, 'resource_location': 'Storage'}
    row['inputs'] = {'scene': {'Storage': {'KMR_docking_pose': [-5.54, 2.13, 0, 0, 0, 1.5708]},
                               'machines': [{'resource_id': 'M1', 'KMR_docking_pose': [-4.7, 2.2, 0, 0, 0, 1.5708]}]}}
    row['physical_evidence']['resources']['KMR'] = {'base_pose': [-5.54, 2.13, 0.0, 0.0, 0.0, 0.7071, 0.7071]}
    if defect == 'source':
        row['fault_reports']['ur5e-1']['source'] = 'robot controller'
    elif defect == 'condition':
        del row['fault_reports']['ur5e-1']['gripper']
    elif defect == 'm1_active':
        row['processing_observations']['M1']['active_job'] = 'trim-1'
    elif defect == 'm2_finished':
        row['processing_observations']['M2']['jobs']['trim-2'].update(process_completed=True, job_state='completed')
    elif defect == 'kmr_moving':
        row['physical_evidence']['resources']['KMR']['base_pose'][:2] = [-3.0, 1.0]
    elif defect == 'kmr_holding':
        row['resource_states']['KMR'].update(resource_state='carrying', held_part='RGOCG4-50_Round_4mm')
    row['capture_checkpoint']['facts_fingerprint'] = checkpoint_fact_fingerprint(row)
    monkeypatch.setattr(recovery_geometry, '_checkpoint', lambda *_args, **_kwargs: {
        'region_occupancy': {'M1.workholding': {'ur5e-1': True}}})
    if defect is None:
        assert capture.verify_capture(row, 'ur5e-1 breakdown')['part_name'] == part
    else:
        with pytest.raises(ValueError):
            capture.verify_capture(row, 'ur5e-1 breakdown')


@pytest.mark.parametrize('changed', [False, True])
def test_robot_capture_freezes_after_observed_stop_and_rejects_changed_native_facts(tmp_path, monkeypatch, changed):
    from cais_spade_llm.recovery_framework import recovery_observations

    sequence = []
    native = {'instance_id': 'world', 'simulation_time': 84., 'world_paused': True,
              'models': {}, 'observed_monotonic': 1.}
    after = {**deepcopy(native), 'observed_monotonic': 2.}
    if changed:
        after['simulation_time'] = 85.
    reader = SimpleNamespace(set_world_paused=Mock(side_effect=lambda _paused: sequence.append('freeze') or native),
                             snapshot=Mock(return_value=after))
    runtime = SimpleNamespace(stopped=True, nominal_idle=SimpleNamespace(is_set=lambda: True),
        context=SimpleNamespace(revision=3, run_id='owned-run'),
        recovery_observation_preparation=SimpleNamespace(reader=reader),
        part_slippage_recovery=SimpleNamespace(_failure={'part_name': 'KET4_Square_4mm'}))

    async def cancel(_runtime, _reader, _directory):
        assert _runtime.recovery_checkpoint_capturing is True
        sequence.append('observed_stop')
        return {'status': 'observed_stopped'}

    def observe(_runtime, _failure):
        assert _runtime.recovery_capture_frozen is True
        sequence.append('observe')
        return _robot_breakdown_capture_record()

    monkeypatch.setattr(recovery_observations, '_cancel_checkpoint_motion', cancel)
    monkeypatch.setattr(recovery_observations, 'capture_observations', observe)
    monkeypatch.setattr(capture, 'stationary_readiness', lambda _row: {'ready': True})
    original = {'pending_nodes': [{'id': 'unfinished'}], 'additional_observations': {}}
    if changed:
        with pytest.raises(ValueError, match='physical checkpoint changed'):
            asyncio.run(capture._freeze_robot_capture(runtime, original, tmp_path))
    else:
        result = asyncio.run(capture._freeze_robot_capture(runtime, original, tmp_path))
        assert result['pending_nodes'] == original['pending_nodes']
        assert result['capture_checkpoint']['world_paused'] is True
        assert result['checkpoint_motion_cancellation']['status'] == 'observed_stopped'
    assert sequence == ['observed_stop', 'freeze', 'observe']
    assert runtime.recovery_checkpoint_capturing is False


def test_robot_capture_keeps_processing_acknowledgement_separate_from_machine_job_records(monkeypatch):
    from cais_spade_llm.recovery_framework import recovery_observations

    context = SimpleNamespace(run_id='owned-run', revision=3, inputs={}, part_tracker={},
        initial_product_states={}, resources={'ur5e-1': SimpleNamespace(valuation={'resource_state': 'failed'})})
    runtime = SimpleNamespace(context=context, jids={'ur5e-1': 'ur5e-1@localhost'},
        setup={'failure_scenario': {'scenario': 'ur5e-1 breakdown', 'resource_id': 'ur5e-1'}})
    failure = {'processing_observations': {'controller_result': {'process_completed': True}},
               'pick_approach': {'observations': {'controller_result': {'status': 'completed'}}}}
    monkeypatch.setattr(recovery_observations, 'capture_recovery_physical_evidence', lambda _runtime: {})
    monkeypatch.setattr(recovery_observations, 'capture_resource_evidence', lambda *_args: {})
    result = recovery_observations.capture_observations(runtime, failure)
    assert result['failure'] == failure
    assert 'processing_observations' not in result
    assert result['resource_states']['ur5e-1']['resource_state'] == 'failed'


def test_robot_capture_wall_budget_override_is_private_and_preserves_other_scenarios(tmp_path, monkeypatch):
    from cais_spade_llm.recovery_framework import evaluation_conditions

    original = {'KMR': {'arm_controller': '/KMR/arm'}, 'machines': [
        {'resource_id': rid, 'simulation_process': {'processing_time_sec': 5.}} for rid in ('M1', 'M2')]}
    source = tmp_path / 'scene.json'
    source.write_text(json.dumps(original))
    monkeypatch.setattr(evaluation_conditions, 'ROOT', tmp_path)
    setup = {'scene_file': 'scene.json', 'simulation': {'speed': 1}}
    result = evaluation_conditions.condition_setup(setup, 'ur5e-1 breakdown', 'normal', tmp_path)
    scene = json.loads(Path(result['scene_file']).read_text())
    assert scene['KMR']['arm_motion_wall_timeout_sec'] == 360.
    assert [row['simulation_process']['processing_time_sec'] for row in scene['machines']] == [90., 60.]
    assert json.loads(source.read_text()) == original
    assert setup == {'scene_file': 'scene.json', 'simulation': {'speed': 1}}
    # The Conveyor capture runs the same KMR motions, so it shares only the wall budget.
    result = evaluation_conditions.condition_setup(setup, 'Conveyor breakdown', 'normal', tmp_path)
    scene = json.loads(Path(result['scene_file']).read_text())
    assert scene['KMR']['arm_motion_wall_timeout_sec'] == 360.
    assert [row['simulation_process']['processing_time_sec'] for row in scene['machines']] == [5., 5.]
    # Machining's shorter M1 trim finishes KET4 first, and M2 keeps trimming at the fault.
    for condition, programs in (('normal', {'circle', 'square'}), ('changed', {'circle'})):
        original['machines'][1]['program_options'] = {'circle': {}, 'square': {}}
        source.write_text(json.dumps(original))
        result = evaluation_conditions.condition_setup(
            setup, 'Machining breakdown during part processing', condition, tmp_path)
        scene = json.loads(Path(result['scene_file']).read_text())
        assert scene['KMR']['arm_motion_wall_timeout_sec'] == 360.
        assert [row['simulation_process']['processing_time_sec'] for row in scene['machines']] == [60., 90.]
        assert set(scene['machines'][1]['program_options']) == programs
        assert json.loads(source.read_text()) == original


@pytest.mark.parametrize('timeout', [None, 360., 0., float('nan')])
def test_kmr_arm_wall_budget_preserves_the_trajectory_and_requires_positive_finite_values(timeout):
    import math

    from cais_spade_llm.recovery_framework import kmr_gazebo

    tree = ast.parse(Path(kmr_gazebo.__file__).read_text())
    function = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == 'execute')
    configuration = {'arm_controller': '/KMR/arm'}
    if timeout is not None:
        configuration['arm_motion_wall_timeout_sec'] = timeout
    action = Mock()
    namespace = {'kmr': configuration, 'math': math, 'action': action,
                 'FollowJointTrajectory': SimpleNamespace(Goal=lambda: SimpleNamespace())}
    exec(compile(ast.Module(body=[function], type_ignores=[]), kmr_gazebo.__file__, 'exec'), namespace)
    trajectory = SimpleNamespace(joint_trajectory=SimpleNamespace(points=[object()]))
    if timeout is not None and (not math.isfinite(timeout) or timeout <= 0):
        with pytest.raises(ValueError, match='finite and positive'):
            namespace['execute'](trajectory)
        action.assert_not_called()
    else:
        namespace['execute'](trajectory)
        args, kwargs = action.call_args
        assert args[1] == '/KMR/arm/follow_joint_trajectory'
        assert args[2].trajectory is trajectory.joint_trajectory
        assert kwargs['timeout'] == (120. if timeout is None else timeout)


def test_stationary_readiness_does_not_ignore_unresolved_child_evidence():
    row = {'jids': {'ur5e-3': 'owner@localhost'}, 'captured_at': 'recorded', 'context_revision': 4,
           'physical_evidence': {'snapshot': {'resources': {'ur5e-3': {'held_part': None}},
               'parts': {'gear_small': {}}}, 'resources': {'ur5e-3': {}}, 'resource_models': {},
               'stationary_parts': [], 'stationary_part_evidence': {'gear_small': {
                   'status': 'NEEDS_CONTEXT', 'reason': 'Observed part is moving'}}, 'unresolved': []}}
    result = capture.stationary_readiness(row)
    assert not result['ready']
    assert 'stationary_part_evidence/gear_small' in result['missing']
    assert result['parts']['gear_small']['reason'] == 'Observed part is moving'
    row['physical_evidence']['stationary_part_evidence']['gear_small']['status'] = 'supported'
    row['physical_evidence']['stationary_parts'] = ['gear_small']
    assert capture.stationary_readiness(row)['ready']


def _conveyor_capture_record():
    parts = {'KET4_Square_4mm': 'ur5e-1', 'gear_small': 'ur5e-4'}
    pickups = {}
    for part, rid in parts.items():
        origin = 'M1' if rid == 'ur5e-1' else '3D Printing Station'
        pickups[rid] = {'acknowledgement': {'run_id': 'owned-run', 'resource_id': rid,
            'event_name': 'pick_grasp', 'evidence': 'resource',
            'parameters': {'part_name': part, 'origin_resource_location': origin}},
            'before': {origin: {'resource_state': 'completed', 'part_name': part}},
            'observations': {'controller_result': {'status': 'completed'}}}
    return {'scenario': 'Conveyor breakdown', 'run_id': 'owned-run',
        'setup': {'execution_mode': 'simulation', 'failure_scenario': {
            'scenario': 'Conveyor breakdown', 'resource_id': 'Conveyor',
            'checkpoint': 'after_M1_pick_before_release', 'part_name': 'KET4_Square_4mm',
            'additional_condition': {'resource_id': 'ur5e-4', 'part_name': 'gear_small'}}},
        'failure': {'checkpoint': 'after_M1_pick_before_release', 'run_id': 'owned-run',
            'part_name': 'KET4_Square_4mm', 'injection_status': 'completed', 'pickups': pickups},
        'resource_states': {rid: {'held_part': part} for part, rid in parts.items()},
        'part_tracker': {part: {'location': rid, 'processCompleted': []} for part, rid in parts.items()},
        'unavailable_resources': ['Conveyor'],
        'inputs': {'scene': {'Conveyor': {'transport_enabled': False}}},
        'physical_evidence': {'snapshot': {}, 'geometry': {}, 'resource_models': {}}}


@pytest.mark.parametrize('defect', [None, 'custody', 'origin', 'run', 'completed', 'available', 'transport', 'assembly', 'inside'])
def test_conveyor_capture_requires_both_pickups_unfinished_assembly_and_clear_printed_part(monkeypatch, defect):
    from cais_spade_llm.resources import recovery_geometry

    row = _conveyor_capture_record()
    peer = row['failure']['pickups']['ur5e-4']['acknowledgement']
    if defect == 'custody':
        row['resource_states']['ur5e-4']['held_part'] = None
    elif defect == 'origin':
        peer['parameters']['origin_resource_location'] = 'Buffer For Machined parts'
    elif defect == 'run':
        peer['run_id'] = 'another-run'
    elif defect == 'completed':
        row['failure']['pickups']['ur5e-1']['before']['M1']['resource_state'] = 'processing'
    elif defect == 'available':
        row['unavailable_resources'] = []
    elif defect == 'transport':
        row['inputs']['scene']['Conveyor']['transport_enabled'] = True
    elif defect == 'assembly':
        row['part_tracker']['gear_small']['processCompleted'] = [{'process': 'assembly'}]
    observed = {'part_region_occupancy': {'assembly_board-v1': {'gear_small': defect == 'inside'}}}
    monkeypatch.setattr(recovery_geometry, '_checkpoint', lambda *_args, **_kwargs: observed)
    if defect is None:
        assert capture.verify_capture(row, 'Conveyor breakdown')['part_name'] == 'KET4_Square_4mm'
    else:
        with pytest.raises(ValueError, match='Conveyor capture'):
            capture.verify_capture(row, 'Conveyor breakdown')


def _conveyor_capture_without_peer():
    row = _conveyor_capture_record()
    del row['setup']['failure_scenario']['additional_condition']
    del row['failure']['pickups']['ur5e-4']
    row.update(schema_version=capture.RAW_SCHEMA_VERSION,
               fault_reports={'Conveyor': {'available': False, 'source': 'device controller'}},
               requirements={'KET4_Square_4mm': [], 'gear_small': []})
    row['failure']['pending_tasks'] = []
    row['resource_states']['ur5e-4'] = {'held_part': None}
    row['part_tracker']['gear_small'] = {'location': 'assembly_board-v1', 'processCompleted': [
        {'process': 'print_part'}, {'process': 'assembly', 'target': 'Gear_Plate/Gear_Shaft_1'}]}
    row['physical_evidence'].update(product_name='assembly_board-v1', parts={
        name: {'name': model} for name, model in (('assembly_board-v1', 'assembly_board_v1'),
                                                  ('gear_small', 'gear_small'), ('KET4_Square_4mm', 'KET4_Square_4mm'))},
        physics_evidence={'attachments': [
            {'model1': 'dual_robot', 'model2': 'KET4_Square_4mm', 'joint_evidence': {'fixed_joint': True}},
            {'model1': 'assembly_board_v1', 'model2': 'gear_small', 'joint_evidence': {'fixed_joint': True}}]})
    return row


@pytest.mark.parametrize('defect', [None, 'pending', 'unassembled', 'unfixed', 'gripper', 'report', 'assembled', 'pickup'])
def test_conveyor_capture_without_held_peer_requires_a_settled_cell(defect):
    row = _conveyor_capture_without_peer()
    attachments = row['physical_evidence']['physics_evidence']['attachments']
    if defect == 'pending':
        row['failure']['pending_tasks'] = [{'resource_id': 'KMR', 'event_name': 'move_to_resource'}]
    elif defect == 'unassembled':
        row['part_tracker']['gear_small']['processCompleted'] = [{'process': 'print_part'}]
    elif defect == 'unfixed':
        attachments.pop()
    elif defect == 'gripper':
        attachments[-1]['model1'] = 'dual_robot'
    elif defect == 'report':
        row['fault_reports'] = {}
    elif defect == 'assembled':
        row['part_tracker']['KET4_Square_4mm']['processCompleted'] = [{'process': 'assembly'}]
    elif defect == 'pickup':
        row['failure']['pickups'] = {}
    if defect is None:
        assert capture.verify_capture(row, 'Conveyor breakdown')['part_name'] == 'KET4_Square_4mm'
    else:
        with pytest.raises(ValueError, match='Conveyor capture'):
            capture.verify_capture(row, 'Conveyor breakdown')


def test_stationary_readiness_requires_complete_joint_and_custody_observations():
    row = {'jids': {'KMR': 'owner@localhost'}, 'captured_at': 'recorded', 'context_revision': 4,
           'physical_evidence': {'snapshot': {'resources': {'KMR': {'held_part': 'part',
               'joint_names': ['arm_joint'], 'joint_positions': [1.]}}, 'parts': {'part': {}}},
               'resources': {'KMR': {'custody_complete': True}}, 'stationary_parts': [],
               'stationary_part_evidence': {}, 'resource_models': {'KMR': {'configuration': {'joint_names': ['arm_joint']}}}}}
    assert capture.stationary_readiness(row)['ready']
    row['physical_evidence']['resources']['KMR']['custody_complete'] = False
    assert not capture.stationary_readiness(row)['ready']
    row['physical_evidence']['resources']['KMR']['custody_complete'] = True
    row['physical_evidence']['snapshot']['resources']['KMR']['joint_positions'] = []
    assert not capture.stationary_readiness(row)['ready']


def test_capture_waits_for_owned_preparation_and_nominal_idle():
    async def run():
        idle = asyncio.Event()
        finished = asyncio.create_task(asyncio.sleep(0))
        owner = SimpleNamespace(_task=finished, _state={'status': 'outline_ready'},
            _observations={'prepared_recovery_request': {'recorded': True}}, _failure={'recorded': True})
        runtime = SimpleNamespace(part_slippage_recovery=owner, stopped=True, nominal_idle=idle)
        assert not capture.recovery_capture_ready(runtime)
        await finished
        assert not capture.recovery_capture_ready(runtime)
        idle.set()
        assert capture.recovery_capture_ready(runtime)
        owner._state['status'] = 'blocked'
        owner._state['message'] = 'Missing actual evidence'
        with pytest.raises(RuntimeError, match='Missing actual evidence'):
            capture.recovery_capture_ready(runtime)
        owner._task = None
        owner._state.update(status='stopped', message='KMR readiness failed')
        with pytest.raises(RuntimeError, match='KMR readiness failed'):
            capture.recovery_capture_ready(runtime)
    asyncio.run(run())


def test_capture_guards_reject_llm_and_recovery_dispatch(monkeypatch):
    from cais_spade_llm.agents.shared_information.llm_agent import LlmAgent
    from cais_spade_llm.recovery_framework import slippage_execution
    from cais_spade_llm.recovery_framework.part_slippage_recovery import PartSlippageRecovery

    methods = [(LlmAgent, name) for name in ('ask_llm', 'ask_llm_structured')]
    methods += [(PartSlippageRecovery, name) for name in ('_run_stage', '_generate_primitives', '_execute')]
    methods += [(slippage_execution, 'execute_generated_recovery')]
    for owner, name in methods:
        monkeypatch.setattr(owner, name, getattr(owner, name))
    calls = capture.install_capture_guards()
    for owner, name in methods:
        with pytest.raises(RuntimeError, match='Capture-only guard rejected'):
            asyncio.run(getattr(owner, name)())
    assert len(calls) == 6


def test_private_workspace_leaves_shared_monitor_and_sources_unchanged(tmp_path, monkeypatch):
    root = tmp_path / 'root'
    package = root / 'cais_spade_llm'
    (package / 'monitor').mkdir(parents=True)
    (package / 'log').mkdir()
    (package / 'resources').mkdir()
    (root / '.git').mkdir()
    (root / 'AGENTS.md').write_text('fixed symbols')
    original = package / 'monitor' / 'record.json'
    original.write_text('original')
    monkeypatch.setattr(capture, 'ROOT', root)
    output = tmp_path / 'capture'
    workspace = capture.prepare_private_workspace(output)
    (workspace / 'cais_spade_llm/monitor/record.json').write_text('private')
    assert original.read_text() == 'original'
    assert not (workspace / '.git').exists()
    assert (workspace / 'cais_spade_llm/resources').resolve() == package / 'resources'


def test_preflight_refuses_existing_runtime_without_cleanup(monkeypatch):
    monkeypatch.setattr(capture, 'active_processes', lambda: [{'pid': 12, 'comm': 'gzserver'}])
    probe = SimpleNamespace(settimeout=Mock(), connect_ex=Mock(return_value=1))
    class Socket:
        def __enter__(self):
            return probe
        def __exit__(self, *args):
            return False
    monkeypatch.setattr(capture.socket, 'socket', Socket)
    with pytest.raises(RuntimeError, match='separate UI/runtime'):
        capture.preflight(11379, 89)


def test_condition_setup_keeps_scene_override_private(tmp_path, monkeypatch):
    from cais_spade_llm.recovery_framework import evaluation_conditions, part_slippage_example

    source = tmp_path / 'source.json'
    source.write_text('{"recorded":true}')
    isolated = tmp_path / 'isolated scene.json'
    isolated.write_text('{"recorded":false}')
    monkeypatch.setattr(part_slippage_example, 'example_setup', lambda **kwargs: {'scene_file': str(source)})
    setup = Mock(return_value={'scene_file': str(isolated)})
    monkeypatch.setattr(evaluation_conditions, 'condition_setup', setup)
    bridge = SimpleNamespace(ROS2_LAUNCH_CMDS={'gazebo_dual': 'ros2 launch world > log 2>&1'})
    capture.install_condition_setup(bridge, tmp_path, 'Conveyor breakdown', 'changed')
    result = part_slippage_example.example_setup(scenario='Conveyor breakdown')
    assert result['scene_file'] == str(isolated)
    assert "robots_file:='" + str(isolated) + "'" in bridge.ROS2_LAUNCH_CMDS['gazebo_dual']
    assert source.read_text() == '{"recorded":true}'
    record = json.loads((tmp_path / 'capture_setup.json').read_text())
    assert record['scene_sha256'] == capture.sha256_file(isolated)
    with pytest.raises(ValueError, match='changed its selected scenario'):
        part_slippage_example.example_setup(scenario='Part slippage')


def test_selected_seed_reproduces_only_a_randomly_timed_failure(tmp_path, monkeypatch):
    from cais_spade_llm.recovery_framework import evaluation_conditions, part_slippage_example

    scene = tmp_path / 'scene.json'
    scene.write_text('{}')
    drawn = {'scene_file': str(scene), 'failure_scenario': {'checkpoint': 'during_processing', 'seed': 7}}
    monkeypatch.setattr(part_slippage_example, 'example_setup', lambda **kwargs: deepcopy(drawn))
    monkeypatch.setattr(evaluation_conditions, 'condition_setup', lambda setup, *args: setup)
    bridge = SimpleNamespace(ROS2_LAUNCH_CMDS={'gazebo_dual': 'ros2 launch world > log 2>&1'})
    scenario = 'Machining breakdown during part processing'
    capture.install_condition_setup(bridge, tmp_path, scenario, 'normal', 4000168541)
    assert part_slippage_example.example_setup(scenario=scenario)['failure_scenario']['seed'] == 4000168541
    recorded = json.loads((tmp_path / 'capture_setup.json').read_text())
    assert recorded['setup']['failure_scenario']['seed'] == 4000168541
    drawn['failure_scenario'] = {'checkpoint': 'during_place_lowering'}
    monkeypatch.setattr(part_slippage_example, 'example_setup', lambda **kwargs: deepcopy(drawn))
    capture.install_condition_setup(bridge, tmp_path, 'Part slippage', 'normal', 1)
    with pytest.raises(ValueError, match='no randomly chosen time'):
        part_slippage_example.example_setup(scenario='Part slippage')


@pytest.mark.parametrize('scenario', list(capture.CHECKPOINTS))
def test_capture_launch_keeps_kmr_map_and_navigation_services(tmp_path, scenario):
    bridge = SimpleNamespace(ROS2_LAUNCH_CMDS={'gazebo_dual': 'original'}, _ROS2_ENV='')
    capture.configure_bridge(bridge, tmp_path, 89, 11379, scenario)
    command = bridge.ROS2_LAUNCH_CMDS['gazebo_dual']
    assert 'launch_nav2:=true' in command
    assert 'recovery_observations:=true' in command
    assert 'run_perception:=false' in command
    assert 'ROS_DOMAIN_ID=89' in bridge._ROS2_ENV
    assert 'GAZEBO_MASTER_URI=http://127.0.0.1:11379' in bridge._ROS2_ENV
    assert bridge._sim_readiness_diagnostic_path == tmp_path / 'readiness_probes.jsonl'


def test_development_capture_never_receives_an_evaluation_manifest(tmp_path, monkeypatch):
    from cais_spade_llm.recovery_framework import recovery_evaluation

    freeze = Mock(side_effect=AssertionError('Development must not reserve an evaluation implementation'))
    monkeypatch.setattr(recovery_evaluation, 'frozen_files', freeze)
    assert capture.capture_implementation(tmp_path, 'development') is None
    assert not (tmp_path / 'implementation_manifest.json').exists()
    with pytest.raises(ValueError, match='development or evaluation'):
        capture.capture_implementation(tmp_path, 'promoted')
    freeze.assert_not_called()


@pytest.mark.parametrize('change', [None, 'content', 'inventory'])
def test_evaluation_capture_pins_exact_implementation_and_inventory(tmp_path, monkeypatch, change):
    from cais_spade_llm.recovery_framework import recovery_evaluation

    root, output = tmp_path / 'source', tmp_path / 'capture'
    root.mkdir()
    output.mkdir()
    source = root / 'controller.py'
    source.write_text('unchanged controller')
    files = {source}
    original_freeze, original_verify = recovery_evaluation.frozen_files, recovery_evaluation.verify_frozen
    monkeypatch.setattr(capture, 'ROOT', root)
    monkeypatch.setattr(recovery_evaluation, '_implementation_files', lambda selected: files.copy())
    monkeypatch.setattr(recovery_evaluation, 'frozen_files', lambda captures: original_freeze(captures, root=root))
    monkeypatch.setattr(recovery_evaluation, 'verify_frozen', lambda record: original_verify(record, root=root))
    manifest = capture.capture_implementation(output, 'evaluation')
    assert manifest == {'files': {'controller.py': capture.sha256_file(source)}, 'implementation_files': ['controller.py']}
    assert json.loads((output / 'implementation_manifest.json').read_text()) == manifest
    if change == 'content':
        source.write_text('changed controller')
    elif change == 'inventory':
        extra = root / 'new.py'
        extra.write_text('new implementation')
        files.add(extra)
    if change is not None:
        with pytest.raises(ValueError, match='Frozen'):
            recovery_evaluation.verify_frozen(manifest)
    else:
        recovery_evaluation.verify_frozen(manifest)


def _capture_with_kmr_map(monkeypatch):
    import hashlib

    from cais_spade_llm.agents.central_controller import recovery_case_evidence
    from cais_spade_llm.recovery_framework import evaluation_conditions as conditions

    # These cases exercise map and coordinate checks. The owning evidence
    # suite also tests this boundary with complete controller/physics records.
    monkeypatch.setattr(recovery_case_evidence, 'recorded_base_hold', Mock(return_value={}))

    source = conditions.ROOT / 'ros2/cais_lab_robotics/scripts/kmr_base_controller.py'
    source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
    controller = {'source': '/KMR/recovery_state', 'controller_node': '/KMR_base_controller',
        'controller_source_sha256': source_sha256, 'instance_id': 'captured-launch',
        'occupancy_map': {'frame_id': 'world', 'stamp': {'sec': 1, 'nanosec': 0},
            'map_load_time': {'sec': 0, 'nanosec': 0}, 'width': 100, 'height': 80,
            'resolution': .1, 'origin': [-5., -4., 0., 0., 0., 0., 1.],
            'cell_count': 8000, 'cells_sha256': 'a' * 64}}
    description = '''<robot name="recorded">
      <joint name="KMR_base_x_joint" type="prismatic"><parent link="world"/>
        <child link="KMR_base_x_link"/><axis xyz="1 0 0"/><limit lower="-20" upper="20"/></joint>
      <joint name="KMR_base_y_joint" type="prismatic"><parent link="KMR_base_x_link"/>
        <child link="KMR_base_y_link"/><axis xyz="0 1 0"/><limit lower="-20" upper="20"/></joint>
      <joint name="KMR_base_yaw_joint" type="continuous"><parent link="KMR_base_y_link"/>
        <child link="KMR_base_link"/><axis xyz="0 0 1"/></joint>
      </robot>'''
    base_pose = [0., 0., 0., 0., 0., 0., 1.]
    return {'physical_evidence': {'snapshot': {'resources': {'KMR': {'base_pose': base_pose}}},
        'resource_models': {'KMR': {'configuration': {'root': 'KMR_base_link',
            'description_sha256': hashlib.sha256(description.encode()).hexdigest()}}},
        'resources': {'KMR': {'base_hold_evidence': {'controller_observation': controller,
            'controller_source_sha256': source_sha256, 'launch_id': 'captured-launch', 'root_pose': base_pose}}}},
        'planning_scene_capture': {'robot_description': description}}


def test_changed_drop_bound_connects_mobile_base_limits_to_recorded_controller_map(monkeypatch):
    from cais_spade_llm.recovery_framework import evaluation_conditions as conditions

    record = _capture_with_kmr_map(monkeypatch)
    original = deepcopy(record)
    result = conditions._recorded_base_reach(record)
    assert result['map_bounds'] == {'x': [-5., 5.], 'y': [-4., 4.]}
    assert result['planning_joint_limits']['KMR_base_x_joint'] == [-20., 20.]
    assert record == original
    from cais_spade_llm.agents.central_controller import recovery_case_evidence

    recovery_case_evidence.recorded_base_hold.assert_called_once_with(resource_id='KMR',
        physical_evidence=record['physical_evidence'], model=record['physical_evidence']['resource_models']['KMR']['configuration'])


@pytest.mark.parametrize('field,value', [('occupancy_map', None),
    ('controller_source_sha256', '0' * 64), ('instance_id', 'another-launch'),
    ('source', '/another/recovery_state')])
def test_changed_drop_bound_rejects_missing_or_unbound_mobile_evidence(monkeypatch, field, value):
    from cais_spade_llm.recovery_framework import evaluation_conditions as conditions

    record = _capture_with_kmr_map(monkeypatch)
    record['physical_evidence']['resources']['KMR']['base_hold_evidence']['controller_observation'][field] = value
    with pytest.raises(ValueError, match='NEEDS_CONTEXT'):
        conditions._recorded_base_reach(record)


@pytest.mark.parametrize('field,value', [('frame_id', 'unresolved-frame'), ('cell_count', 1),
    ('cells_sha256', None), ('resolution', True), ('width', 1000), ('stamp', None),
    ('origin', [-5., -4., 0., 0., 0., .5, .5])])
def test_changed_drop_bound_requires_complete_map_within_planning_coordinates(monkeypatch, field, value):
    from cais_spade_llm.recovery_framework import evaluation_conditions as conditions

    record = _capture_with_kmr_map(monkeypatch)
    observation = record['physical_evidence']['resources']['KMR']['base_hold_evidence']['controller_observation']
    observation['occupancy_map'][field] = value
    if field == 'width':
        observation['occupancy_map']['cell_count'] = value * observation['occupancy_map']['height']
    with pytest.raises(ValueError, match='NEEDS_CONTEXT'):
        conditions._recorded_base_reach(record)


def test_changed_drop_bound_cannot_silently_omit_another_mobile_resource(monkeypatch):
    from cais_spade_llm.recovery_framework import evaluation_conditions as conditions

    record = _capture_with_kmr_map(monkeypatch)
    record['physical_evidence']['snapshot']['resources']['another-owner'] = {'base_pose': [0.] * 7}
    with pytest.raises(ValueError, match='complete mobile resource'):
        conditions._recorded_base_reach(record)


def test_changed_drop_bound_requires_exact_connection_between_map_and_base_axes(monkeypatch):
    import hashlib

    from cais_spade_llm.recovery_framework import evaluation_conditions as conditions

    record = _capture_with_kmr_map(monkeypatch)
    record['planning_scene_capture']['robot_description'] = record['planning_scene_capture'][
        'robot_description'].replace('axis xyz="1 0 0"', 'axis xyz="0 1 0"')
    record['physical_evidence']['resource_models']['KMR']['configuration']['description_sha256'] = hashlib.sha256(
        record['planning_scene_capture']['robot_description'].encode()).hexdigest()
    with pytest.raises(ValueError, match='coordinates are not connected'):
        conditions._recorded_base_reach(record)


@pytest.mark.parametrize('frozen,stationary', [(True, True), (False, True), (True, False)])
def test_capture_publishes_the_scene_from_its_checkpoint(tmp_path, monkeypatch, frozen, stationary):
    from cais_spade_llm.recovery_framework import (
        environment_runtime,
        evaluation_conditions,
        outline_gazebo_evidence,
        part_slippage_example,
        recovery_motion_preview,
        recovery_observations,
    )
    from cais_spade_llm.ui import bridge as bridge_module

    scene = {'scene_fingerprint': 'recorded-scene', 'model_fingerprint': 'recorded-model'}
    observations = {'run_id': 'run', 'captured_at': 'recorded', 'context_revision': 2,
        'prepared_recovery_request': {'recorded': True}, 'jids': {},
        'physical_evidence': {'resources': {}, 'evidence_fingerprint': 'physical'},
        'planning_scene_capture': deepcopy(scene)}
    if frozen:
        observations['capture_checkpoint'] = {'checkpoint_id': 'checkpoint'}
    recovery = SimpleNamespace(_state={'status': 'outline_ready'}, _observations=observations,
        _failure={'recorded': True}, _pending_nodes=lambda: [])
    runtime = SimpleNamespace(part_slippage_recovery=recovery, stopped=True, jids={},
        setup={'execution_mode': 'simulation'}, context=SimpleNamespace(inputs={'scene': {
            'safety_preparation': {'outline_preview': {'planning_scene_service': 'scene',
                'preview_parameter_service': 'parameters', 'timeout_seconds': 10}}}}))
    example = SimpleNamespace(_runtime=runtime, _status={}, set_options=Mock(), _restore_settings=Mock())

    async def run_owned(awaitable):
        return await awaitable

    shared = tmp_path / 'shared_environment_runs'
    monkeypatch.setattr(environment_runtime, 'RUN_DIRECTORY', shared)
    previous = environment_runtime.LatestReport(shared, 'machine-run')
    previous.save({'retained': True})
    previous_sha256 = capture.sha256_file(shared / 'latest/run.json')

    async def start_failure():
        report = environment_runtime.LatestReport(environment_runtime.RUN_DIRECTORY, 'run')
        report.save({'owned': True})

    bridge = SimpleNamespace(set_part_slippage_modes=Mock(), start_failure_example=AsyncMock(side_effect=start_failure),
        part_slippage_action=AsyncMock(), ros2_stop_all=Mock(), _run_on_agent_runtime=run_owned,
        _ros2_procs={}, system_running=False, last_error='')
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(capture, 'preflight', lambda *_args: {})
    monkeypatch.setattr(capture, 'preserved_files', lambda: {})
    monkeypatch.setattr(capture, 'prepare_private_workspace', lambda _directory: tmp_path)
    monkeypatch.setattr(capture, 'install_capture_guards', lambda: [])
    monkeypatch.setattr(capture, 'configure_bridge', lambda *_args: None)
    monkeypatch.setattr(capture, 'install_condition_setup', lambda *_args: None)
    monkeypatch.setattr(capture, 'recovery_capture_ready', lambda _runtime: True)
    monkeypatch.setattr(capture, 'stationary_readiness', lambda _record: {
        'ready': stationary, 'missing': [] if stationary else ['recorded active owner']})
    monkeypatch.setattr(capture, 'verify_capture', lambda *_args: {'checkpoint': 'recorded'})
    monkeypatch.setattr(bridge_module, 'SystemBridge', lambda: bridge)
    monkeypatch.setattr(part_slippage_example, 'example_for_bridge', lambda _bridge: example)
    for name in ('_USER_VERIFIED_PLAN', '_TOOLS_OUT', '_MONITOR', '_LOG_DIR', '_RECOVERY_RUNTIME_DATA_DIR'):
        monkeypatch.setattr(bridge_module, name, getattr(bridge_module, name))
    monkeypatch.setattr(recovery_observations, 'RECORDS', tmp_path)
    observe = Mock(return_value=deepcopy(observations))
    monkeypatch.setattr(recovery_observations, 'capture_observations', observe)
    monkeypatch.setattr(recovery_observations, 'offline_preparation_missing', lambda _record: [])
    monkeypatch.setattr(recovery_observations, 'save_native_preview_model_from_capture', AsyncMock(return_value={}))
    monkeypatch.setattr(outline_gazebo_evidence, '_prepare_saved_capture', AsyncMock(return_value={'recorded': True}))
    preview = AsyncMock(return_value=deepcopy(scene))
    monkeypatch.setattr(recovery_motion_preview, 'capture_preview_scene', preview)
    monkeypatch.setattr(evaluation_conditions, 'verify_capture_condition', lambda *_args: {'recorded': True})
    args = SimpleNamespace(output=tmp_path / 'attempt', purpose='development', gazebo_port=11379,
        ros_domain=89, scenario='Machining breakdown during part processing', condition='normal',
        timeout=60, settle_timeout=60)
    # Restore the harness's process-local environment changes after this test.
    for key in ('ROS_DOMAIN_ID', 'GAZEBO_MASTER_URI', 'CAIS_KEEP_GAZEBO_ON_EXIT'):
        monkeypatch.delenv(key, raising=False)
    result = asyncio.run(capture.run_capture(args))
    assert result['status'] == 'captured'
    assert result['scene_fingerprint'] == scene['scene_fingerprint']
    assert result['model_fingerprint'] == scene['model_fingerprint']
    assert result['stationary_readiness']['ready'] is stationary
    assert capture.sha256_file(shared / 'latest/run.json') == previous_sha256
    assert json.loads((args.output / 'environment_runs/latest/run.json').read_text())['run_id'] == 'run'
    if not stationary:
        assert result['stationary_readiness']['missing'] == ['recorded active owner']
    assert preview.await_count == observe.call_count == (0 if frozen else 1)
    bridge.part_slippage_action.assert_awaited_once_with('stop')
    bridge.ros2_stop_all.assert_called_once_with(reason='owned_fresh_capture_complete')


@pytest.mark.parametrize('defect', [None, 'window_owner', 'image_write', 'unrendered', 'x11_error'])
def test_scene_image_launches_only_owned_gui_at_stopped_checkpoint(tmp_path, monkeypatch, defect):
    import cv2
    import numpy as np

    observations = tmp_path / 'observations.json'
    observations.write_text('{}')
    recorded = {'run_id': 'run', 'captured_at': 'checkpoint',
                'physical_evidence': {'evidence_fingerprint': 'physical'}}
    runtime = SimpleNamespace(stopped=True, nominal_idle=SimpleNamespace(is_set=lambda: True),
        conveyor_fault=SimpleNamespace(snapshot=lambda: {'visual': {'status': 'completed'}}))
    process = Mock(pid=1234)
    process.poll.return_value = None
    launch = Mock(return_value=process)
    bridge = SimpleNamespace(_ROS2_ENV='owned-environment; ', _ros2_procs={},
        _register_ui_process=Mock(), ros2_stop=Mock())
    frame = np.tile(np.array([[0, 255]], dtype=np.uint8), (120, 80))
    rendered = np.repeat(frame[:, :, None], 3, axis=2)
    blank = np.zeros_like(rendered)
    frames = [blank, blank]
    if defect == 'x11_error':
        frames.insert(0, RuntimeError('X11 BadMatch'))

    def scene_frame(*_args):
        if defect == 'unrendered':
            return blank
        value = frames.pop(0) if frames else rendered
        if isinstance(value, RuntimeError):
            raise value
        return value

    frame_reader = Mock(side_effect=scene_frame)
    clock = iter(range(0, 2000, 100 if defect == 'unrendered' else 1))

    def image_write(path, frame):
        if defect == 'image_write':
            return False
        assert frame.shape == (120, 160, 3)
        capture.Path(path).write_bytes(b'owned-window-fixture')
        return True

    monkeypatch.setattr(capture.subprocess, 'Popen', launch)
    environment = {'GAZEBO_MODEL_PATH': '/owned/models:/owned/share',
                   'GAZEBO_PLUGIN_PATH': '/owned/lib', 'GAZEBO_MASTER_URI': 'http://127.0.0.1:11381'}
    monkeypatch.setattr(capture, '_owned_gazebo_environment', lambda _bridge: (2222, environment))
    monkeypatch.setattr(capture.subprocess, 'check_output', lambda command, **_kwargs:
        '0xb "Gazebo"' if command[0] == 'xwininfo' else '_NET_WM_PID = 5678')
    monkeypatch.setattr(capture.os, 'getpgid', lambda pid: 4321 if pid == 1234 or defect != 'window_owner' else 9999)
    monkeypatch.setattr(capture.time, 'sleep', lambda _seconds: None)
    monkeypatch.setattr(capture.time, 'monotonic', lambda: next(clock))
    monkeypatch.setattr(capture, '_capture_gazebo_frame', frame_reader)
    monkeypatch.setattr(cv2, 'imwrite', image_write)
    if defect in {None, 'x11_error'}:
        result = capture.capture_scene_image(bridge, runtime, recorded, observations)
        assert result['run_id'] == 'run'
        assert result['source_sha256'] == capture.sha256_file(observations)
        assert result['window_process_group'] == 4321
        assert result['server_pid'] == 2222
        assert result['width'] == 160 and result['height'] == 120
        assert result['rendering']['verified'] is True
        assert frame_reader.call_count > 2
        assert result['physical_evidence_fingerprint'] == 'physical'
        assert result['execution_authorized'] is False
        assert json.loads((tmp_path / 'failure_scene.json').read_text()) == result
    else:
        error = ValueError if defect == 'window_owner' else TimeoutError if defect == 'unrendered' else RuntimeError
        with pytest.raises(error):
            capture.capture_scene_image(bridge, runtime, recorded, observations)
        assert not (tmp_path / 'failure_scene.png').exists()
    launch.assert_called_once()
    assert launch.call_args.kwargs['start_new_session'] is True
    assert launch.call_args.kwargs['env'] == environment
    bridge._register_ui_process.assert_called_once()
    assert bridge._ros2_procs['capture_gazebo_gui'] is process
    bridge.ros2_stop.assert_called_once_with('capture_gazebo_gui', reason='owned_failure_scene_saved')


@pytest.mark.parametrize('return_code', [1, -11])
def test_scene_frame_child_failure_cannot_exit_capture_process(tmp_path, monkeypatch, return_code):
    child = Mock(return_value=SimpleNamespace(returncode=return_code, stderr='X11 BadMatch'))
    monkeypatch.setattr(capture.subprocess, 'run', child)
    with pytest.raises(RuntimeError, match='Gazebo frame read failed'):
        capture._capture_gazebo_frame(11, tmp_path / 'frame.png')
    assert child.call_args.args[0][0] == capture.sys.executable
    assert child.call_args.kwargs['timeout'] == 10.


@pytest.mark.parametrize('defect', ['running', 'busy', 'marker'])
def test_scene_image_requires_final_checkpoint_before_gui_launch(tmp_path, monkeypatch, defect):
    runtime = SimpleNamespace(stopped=defect != 'running',
        nominal_idle=SimpleNamespace(is_set=lambda: defect != 'busy'),
        conveyor_fault=SimpleNamespace(snapshot=lambda: {'visual': {
            'status': 'pending' if defect == 'marker' else 'completed'}}))
    launch = Mock()
    monkeypatch.setattr(capture.subprocess, 'Popen', launch)
    with pytest.raises(ValueError):
        capture.capture_scene_image(SimpleNamespace(), runtime, {}, tmp_path / 'observations.json')
    launch.assert_not_called()


@pytest.mark.parametrize('defect', [None, 'missing', 'multiple'])
def test_scene_gui_inherits_only_the_owned_gazebo_server_environment(tmp_path, monkeypatch, defect):
    process = Mock(pid=1234)
    process.poll.return_value = None
    bridge = SimpleNamespace(_ros2_procs={'gazebo_dual': process})
    groups = {1234: 4321, 5678: 9999, 2222: 4321, 3333: 4321}
    monkeypatch.setattr(capture.os, 'getpgid', lambda pid: groups[pid])
    server_ids = [5678] + ([] if defect == 'missing' else [2222]) + ([3333] if defect == 'multiple' else [])
    for pid in server_ids:
        directory = tmp_path / str(pid)
        directory.mkdir()
        (directory / 'comm').write_text('gzserver\n')
        (directory / 'environ').write_bytes(
            b'GAZEBO_MODEL_PATH=/owned/share:/owned/models\0'
            b'GAZEBO_PLUGIN_PATH=/owned/lib\0'
            b'GAZEBO_MASTER_URI=http://127.0.0.1:11381\0')
    if defect is not None:
        with pytest.raises(ValueError, match='exactly one owned'):
            capture._owned_gazebo_environment(bridge, proc_root=tmp_path)
    else:
        pid, environment = capture._owned_gazebo_environment(bridge, proc_root=tmp_path)
        assert pid == 2222
        assert environment == {'GAZEBO_MODEL_PATH': '/owned/share:/owned/models',
                               'GAZEBO_PLUGIN_PATH': '/owned/lib',
                               'GAZEBO_MASTER_URI': 'http://127.0.0.1:11381'}
