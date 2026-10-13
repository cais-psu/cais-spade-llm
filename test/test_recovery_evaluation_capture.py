from __future__ import annotations

"""Capture-only isolation and factual readiness for the four existing failures."""

import asyncio
import json
from copy import deepcopy
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
def test_capture_verification_requires_exact_reconciled_checkpoint(scenario):
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
    from cais_spade_llm.recovery_framework.part_slippage_recovery import PartSlippageRecovery
    from cais_spade_llm.recovery_framework import slippage_execution

    methods = [(LlmAgent, name) for name in ('ask_llm', 'ask_llm_structured')]
    methods += [(PartSlippageRecovery, name) for name in ('_run_stage', '_generate_outline', '_generate_primitives', '_execute')]
    methods += [(slippage_execution, 'execute_generated_recovery')]
    for owner, name in methods:
        monkeypatch.setattr(owner, name, getattr(owner, name))
    calls = capture.install_capture_guards()
    for owner, name in methods:
        with pytest.raises(RuntimeError, match='Capture-only guard rejected'):
            asyncio.run(getattr(owner, name)())
    assert len(calls) == 7


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

    from cais_spade_llm.recovery_framework import evaluation_conditions as conditions
    from cais_spade_llm.agents.central_controller import recovery_case_evidence

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
    import cais_spade_llm.ui_main
    from cais_spade_llm.ui import bridge as bridge_module
    from cais_spade_llm.recovery_framework import (
        evaluation_conditions, outline_gazebo_evidence, part_slippage_example,
        recovery_motion_preview, recovery_observations,
    )

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

    bridge = SimpleNamespace(set_part_slippage_modes=Mock(), start_failure_example=AsyncMock(),
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
    if not stationary:
        assert result['stationary_readiness']['missing'] == ['recorded active owner']
    assert preview.await_count == observe.call_count == (0 if frozen else 1)
    bridge.part_slippage_action.assert_awaited_once_with('stop')
    bridge.ros2_stop_all.assert_called_once_with(reason='owned_fresh_capture_complete')
