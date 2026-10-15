from __future__ import annotations

"""Simulation job controls, retained observations and detached owner validation."""

from copy import deepcopy
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from cais_spade_llm.recovery_framework import SCENE_PATH
from cais_spade_llm.resources.machine.machine_jobs import (
    MachineJobs, MACHINE_PRIMITIVES, processing_failure_time, project_job_command,
    job_public_state, public_processing_observations,
    interrupted_pickup_conditions,
)


@pytest.fixture
def machine():
    return deepcopy(json.loads(SCENE_PATH.read_text())['machines'][1])


def observed(part='RGOCG4-50_Round_4mm'):
    return {'part_name': part, 'correct_part_present': True, 'robot_access_clear': True}


def start(owner, *, identity='circle-job', part='RGOCG4-50_Round_4mm', result='circle', now=100):
    return owner.command('start_job', {'job_id': identity, 'task_id': identity, 'part_name': part,
        'process': 'trim', 'result': result}, command_id=identity, now=now, observations=observed(part))


def test_start_ack_is_not_completion_and_completion_is_recorded_once(machine):
    owner = MachineJobs(machine, 'run')
    ack = start(owner)
    assert ack['process_completed'] is False and ack['job_state'] == 'running'
    assert owner.observe(101, observed()) is None
    duration = machine['simulation_process']['processing_time_sec']
    completed = owner.observe(100 + duration, observed())
    assert completed['process_completed'] and completed['processing_active_sec'] == duration
    assert owner.observe(100 + duration + 1, observed()) is None


def test_pause_unload_other_job_reload_and_resume_remaining_time(machine):
    owner = MachineJobs(machine, 'run')
    start(owner)
    owner.command('pause_job', {'job_id': 'circle-job'}, command_id='pause', now=101, observations=observed())
    owner.command('select_program', {'program': 'square'}, command_id='select-square', now=102, observations={})
    start(owner, identity='square-job', part='KET4_Square_4mm', result='square', now=103)
    duration = machine['simulation_process']['processing_time_sec']
    owner.observe(103 + duration, observed('KET4_Square_4mm'))
    owner.command('select_program', {'program': 'circle'}, command_id='select-circle', now=104+duration, observations={})
    owner.command('resume_job', {'job_id': 'circle-job'}, command_id='resume', now=105+duration, observations=observed())
    assert owner.record['jobs']['circle-job']['processing_active_sec'] == 1
    completed = owner.observe(105 + duration + duration - 1, observed())
    assert completed['job_id'] == 'circle-job'
    assert completed['processing_fraction'] == 1
    assert completed['simulation_paused_sec'] == 4 + duration
    assert len(owner.record['jobs']) == 2


@pytest.mark.parametrize('primitive', ['resume_job', 'start_job'])
def test_stopped_job_cannot_resume_or_restart_without_declaration(machine, primitive):
    owner = MachineJobs(machine, 'run')
    start(owner)
    owner.command('stop_job', {'job_id': 'circle-job'}, command_id='stop', now=101, observations=observed())
    before = owner.snapshot()
    with pytest.raises(ValueError):
        if primitive == 'resume_job':
            owner.command(primitive, {'job_id': 'circle-job'}, command_id='new', now=102, observations=observed())
        else:
            start(owner, identity='new', now=102)
    assert owner.snapshot() == before


def test_duplicate_conflicting_and_stale_commands(machine):
    owner = MachineJobs(machine, 'run')
    ack = start(owner)
    assert start(owner, now=101) == ack
    before = owner.snapshot()
    with pytest.raises(ValueError, match='conflicting'):
        start(owner, result='square')
    with pytest.raises(ValueError, match='backwards'):
        owner.command('pause_job', {'job_id': 'circle-job'}, command_id='pause', now=99, observations=observed())
    with pytest.raises(ValueError, match='Stale'):
        MachineJobs(machine, 'another-run', before)
    assert owner.snapshot() == before


def test_wrong_part_and_changed_tool_do_not_change_record(machine):
    owner = MachineJobs(machine, 'run')
    start(owner)
    owner.command('pause_job', {'job_id': 'circle-job'}, command_id='pause', now=101, observations=observed())
    before = owner.snapshot()
    with pytest.raises(ValueError, match='part'):
        owner.command('resume_job', {'job_id': 'circle-job'}, command_id='resume', now=102, observations=observed('gear_small'))
    assert owner.snapshot() == before
    owner.programs['square']['tool'] = 'incompatible'
    with pytest.raises(ValueError, match='different tool'):
        owner.command('select_program', {'program': 'square'}, command_id='program', now=102, observations={})
    assert owner.snapshot() == before


def test_completion_wins_pause_race_and_shutdown_does_not_resume(machine):
    owner = MachineJobs(machine, 'run')
    start(owner)
    with pytest.raises(ValueError, match='active running'):
        owner.command('pause_job', {'job_id': 'circle-job'}, command_id='pause',
            now=100+machine['simulation_process']['processing_time_sec'], observations=observed())
    assert owner.record['jobs']['circle-job']['process_completed'] is True
    second = MachineJobs(machine, 'second')
    start(second)
    second.observe(101, observed())
    second.interrupt('operator Stop')
    restored = MachineJobs(machine, 'second', second.snapshot())
    assert restored.record['active_job'] is None
    assert restored.record['jobs']['circle-job']['processing_active_sec'] == 1
    assert restored.observe(200, observed()) is None


@pytest.mark.parametrize('resource_id', ['M1', 'M2'])
@pytest.mark.parametrize('reason', ['operator Stop', 'Machining breakdown during part processing'])
def test_interrupted_pickup_uses_owner_quiescence_without_claiming_geometry(machine, resource_id, reason):
    machine['resource_id'] = resource_id
    owner = MachineJobs(machine, 'recorded-run')
    start(owner)
    owner.observe(101, observed())
    owner.interrupt(reason)
    record = owner.snapshot()
    record['jobs']['circle-job']['observations'].pop('robot_access_clear')
    valuation = {'resource_state': 'loaded', 'part_name': 'RGOCG4-50_Round_4mm', **job_public_state(record)}
    before = deepcopy((machine, record, valuation))
    result = interrupted_pickup_conditions(machine, record, valuation, valuation['part_name'], run_id='recorded-run')
    assert result['guards'] == {'resource_state': {'equals': 'loaded'}, 'job_state': {'equals': 'interrupted'},
                                'active_job': {'equals': None}}
    assert result['simulation_processing_quiescent'] is True
    assert result['physical_access_validated'] is False
    assert result['process_completed'] is False and result['execution_authorized'] is False
    assert (machine, record, valuation) == before


@pytest.mark.parametrize('damage,exception', [('missing', KeyError), ('running', ValueError),
    ('wrong_part', ValueError), ('run', KeyError), ('model', KeyError), ('configuration', KeyError),
    ('job_identity', KeyError), ('ambiguous', KeyError), ('inconsistent', KeyError), ('missing_state', KeyError)])
def test_interrupted_pickup_rejects_active_conflicting_or_incomplete_owner_evidence(machine, damage, exception):
    owner = MachineJobs(machine, 'recorded-run')
    start(owner)
    if damage != 'running':
        owner.interrupt('operator Stop')
    record = owner.snapshot()
    valuation = {'resource_state': 'loaded', 'part_name': 'RGOCG4-50_Round_4mm', **job_public_state(record)}
    if damage == 'missing':
        record = {}
    elif damage == 'wrong_part':
        valuation['part_name'] = 'KET4_Square_4mm'
    elif damage == 'run':
        record['run_id'] = 'another-run'
    elif damage == 'model':
        record['model_fingerprint'] = 'changed'
    elif damage == 'configuration':
        record['jobs']['circle-job']['configuration']['tool'] = 'changed'
    elif damage == 'job_identity':
        record['jobs']['circle-job']['job_id'] = 'unrecorded'
    elif damage == 'ambiguous':
        record['jobs']['another-job'] = deepcopy(record['jobs']['circle-job'])
    elif damage == 'inconsistent':
        valuation['active_job'] = 'circle-job'
    elif damage == 'missing_state':
        del valuation['job_state']
    before = deepcopy((record, valuation))
    with pytest.raises(exception):
        interrupted_pickup_conditions(machine, record, valuation, 'RGOCG4-50_Round_4mm', run_id='recorded-run')
    assert (record, valuation) == before


def test_detached_command_uses_same_logic_and_leaves_capture_immutable(machine, monkeypatch):
    import socket
    monkeypatch.setattr(socket, 'socket', lambda *a, **kw: pytest.fail('offline attempted network'))
    owner = MachineJobs(machine, 'run')
    start(owner)
    owner.observe(101, observed())
    capture = owner.snapshot()
    original = deepcopy(capture)
    projected, ack = project_job_command(machine, capture, 'pause_job', {'job_id': 'circle-job'}, observed(), command_id='pause')
    assert job_public_state(projected)['job_state'] == 'paused'
    assert ack['process_completed'] is False and capture == original
    public = public_processing_observations(projected)
    assert 'commands' not in public and 'configuration' not in public['jobs']['circle-job']


@pytest.mark.parametrize('resource_id', ['M1', 'M2'])
def test_inactive_program_selection_needs_no_clock_and_cannot_advance_processing(machine, resource_id):
    machine = deepcopy(next(row for row in json.loads(SCENE_PATH.read_text())['machines']
                            if row['resource_id'] == resource_id))
    owner = MachineJobs(machine, 'run')
    owner.command('select_program', {'program': 'circle'}, command_id='circle', now=99, observations={})
    start(owner)
    owner.command('pause_job', {'job_id': 'circle-job'}, command_id='pause', now=101, observations=observed())
    capture = owner.snapshot()
    del capture['simulation_clock_sec']
    before = deepcopy(capture)
    projected, reply = project_job_command(machine, capture, 'select_program', {'program': 'square'}, {}, command_id='select')
    assert projected['selected_program'] == 'square'
    assert projected['jobs'] == before['jobs']
    assert projected['active_job'] is None and 'simulation_clock_sec' not in projected
    assert reply['simulation_clock_sec'] is None and reply['process_completed'] is False
    assert capture == before
    restored = MachineJobs(machine, 'run', projected)
    assert restored.command('select_program', {'program': 'square'}, command_id='select', now=None, observations={}) == reply
    with pytest.raises(ValueError, match='conflicting'):
        restored.command('select_program', {'program': 'circle'}, command_id='select', now=None, observations={})


@pytest.mark.parametrize('primitive', ['start_job', 'pause_job', 'resume_job', 'stop_job'])
def test_processing_commands_still_require_observed_clock(machine, primitive):
    capture = MachineJobs(machine, 'run').snapshot()
    before = deepcopy(capture)
    with pytest.raises(ValueError, match='observed simulation_clock_sec'):
        project_job_command(machine, capture, primitive, {}, {}, command_id='command')
    assert capture == before


@pytest.mark.parametrize('resume', [False, True])
def test_relative_owner_duration_keeps_observed_time_unknown(machine, resume):
    from cais_spade_llm.resources.machine.machine_jobs import project_job_completion

    owner = MachineJobs(machine, 'run')
    if resume:
        start(owner)
        owner.command('pause_job', {'job_id': 'circle-job'}, command_id='pause',
                      now=101, observations=observed())
    capture = owner.snapshot()
    capture.pop('simulation_clock_sec', None)
    before = deepcopy(capture)
    params = {'job_id': 'circle-job'} if resume else {
        'job_id': 'circle-job', 'task_id': 'circle-job', 'part_name': 'RGOCG4-50_Round_4mm',
        'process': 'trim', 'result': 'circle'}
    projected, reply = project_job_command(machine, capture, 'resume_job' if resume else 'start_job',
        params, observed(), command_id='calculation', allow_relative_time=True)
    assert 'simulation_clock_sec' not in projected and 'simulation_clock_sec' not in reply
    assert projected['calculation_time']['absolute_time_known'] is False
    job = projected['jobs']['circle-job']
    assert 'simulation_paused_sec' not in job and 'simulation_elapsed_sec' not in job
    assert 'simulation_clock_sec' not in job
    duration = job['processing_time_sec'] - job['processing_active_sec']
    completed = project_job_completion(machine, projected, duration, observed())
    assert completed['jobs']['circle-job']['process_completed'] is True
    assert completed['jobs']['circle-job']['processing_fraction'] == 1
    assert 'simulation_clock_sec' not in completed
    assert 'simulation_completed_sec' not in completed['jobs']['circle-job']
    assert capture == before


@pytest.mark.parametrize('damage', ['active', 'unknown', 'access', 'program'])
def test_relative_owner_time_preserves_required_processing_checks(machine, damage):
    owner = MachineJobs(machine, 'run')
    if damage in {'active', 'unknown'}:
        start(owner)
        if damage == 'unknown':
            owner.record['active_job'] = None
            owner.record['jobs']['circle-job']['job_state'] = None
    capture = owner.snapshot()
    capture.pop('simulation_clock_sec', None)
    params = {'job_id': 'new-job', 'task_id': 'new-job', 'part_name': 'RGOCG4-50_Round_4mm',
              'process': 'trim', 'result': 'unsupported' if damage == 'program' else 'circle'}
    observation = observed()
    if damage == 'access':
        observation['robot_access_clear'] = False
    before = deepcopy(capture)
    with pytest.raises(ValueError):
        project_job_command(machine, capture, 'start_job', params, observation,
                            command_id='calculation', allow_relative_time=True)
    assert capture == before


@pytest.mark.parametrize('damage', ['active_job', 'running_job', 'unknown_job', 'different_tool', 'unvalidated', 'unknown_program'])
def test_program_selection_without_clock_preserves_owner_constraints(machine, damage):
    owner = MachineJobs(machine, 'run')
    start(owner)
    owner.command('pause_job', {'job_id': 'circle-job'}, command_id='pause', now=101, observations=observed())
    owner.record.pop('simulation_clock_sec')
    program = 'square'
    if damage == 'active_job':
        owner.record['active_job'] = 'circle-job'
    elif damage == 'running_job':
        owner.record['jobs']['circle-job']['job_state'] = 'running'
    elif damage == 'unknown_job':
        owner.record['jobs']['circle-job'].pop('job_state')
    elif damage == 'different_tool':
        owner.programs['square']['tool'] = 'different'
    elif damage == 'unvalidated':
        owner.programs['square']['program']['validated'] = False
    else:
        program = 'unknown'
    before = owner.snapshot()
    with pytest.raises(ValueError):
        owner.command('select_program', {'program': program}, command_id='select', now=None, observations={})
    assert owner.snapshot() == before


def _machine_access_case(resource_id, *, projected=True):
    from cais_spade_llm.agents.shared_information.recovery_validation_protocol import recovery_validation_fingerprint as fingerprint
    from test_recovery_outline_evidence import _commanded_hold_case

    scene = deepcopy(json.loads(SCENE_PATH.read_text()))
    machine = next(row for row in scene['machines'] if row['resource_id'] == resource_id)
    part = 'KET4_Square_4mm'
    pose = [*machine['workholding_pose'][:3], 0., 0., 0., 1.]
    handling = machine['handling_robot']
    robot = next(row for row in scene['robots'] if row['resource_id'] == handling)
    robot.update(prefix='', clearance_observation_link='body')
    scene['KMR']['task_execution']['clearance_link_bounds'] = {'body': [{'center': [0., 0., 0.], 'radius': .1}]}
    snapshot = {'resource_state': 'loaded', 'part_name': part}
    physical = {'endpoint_evidence_policy_version': 2, 'provenance': {'kind': 'projected', 'binding': {'id': 'current'}},
        'snapshot': {'frame': 'world', 'resources': {resource_id: deepcopy(snapshot)},
            'parts': {part: {'current_pose': pose, 'location': resource_id}}},
        'geometry': {'frame': 'world', 'resources': {resource_id: {'stationary_only': True}}},
        'resources': {resource_id: {'model_static': True}}, 'resource_models': {},
        'stationary_part_evidence': {part: {'status': 'supported', 'current_pose': deepcopy(pose),
            'source': {'kind': 'supplied verification case'}}}}
    for peer in ('KMR', handling):
        _, _, state, model, resource = _commanded_hold_case()
        model['configuration']['root_xyz'] = [pose[0] + 1., *pose[1:3]]
        state['current_pose'] = [pose[0] + 1.005, *pose[1:]]
        resource['observation_state'] = deepcopy(state)
        physical['snapshot']['resources'][peer] = state
        physical['resources'][peer] = resource
        physical['resource_models'][peer] = model
    if projected:
        state = physical['snapshot']['resources']['KMR']
        resource = physical['resources']['KMR']
        resource.pop('controller_hold_activity')
        resource['projected_controller_hold'] = {'kind': 'projected',
            'binding': deepcopy(physical['provenance']['binding']),
            'joint_names': deepcopy(state['joint_names']), 'positions': deepcopy(state['joint_positions'])}
        resource['projected_endpoint'] = {'binding': deepcopy(physical['provenance']['binding']),
            'state_fingerprint': fingerprint(state),
            'configuration_fingerprint': fingerprint(physical['resource_models']['KMR']['configuration'])}
    evidence = {'resource_id': resource_id, 'inputs': {'scene': scene}}
    task = {'outline_id': 'provider_proposal', 'event_name': 'machine_part', 'resource_jid': resource_id + '@localhost',
        'part_name': part, 'expected_end_state': {'resource_state': 'completed'}}
    return evidence, snapshot, task, physical, part


@pytest.mark.parametrize('resource_id', ['M1', 'M2'])
@pytest.mark.parametrize('projected', [False, True])
def test_machine_access_uses_supported_geometry_without_changing_observations(resource_id, projected):
    from cais_spade_llm.resources.environment_recovery import _machine_access_projection, _machine_command_projection

    evidence, snapshot, task, physical, part = _machine_access_case(resource_id, projected=projected)
    machine = next(row for row in evidence['inputs']['scene']['machines'] if row['resource_id'] == resource_id)
    owner = MachineJobs(machine, 'run')
    owner.command('select_program', {'program': 'square'}, command_id='program', now=100., observations={})
    evidence['processing_observations'] = owner.snapshot()
    physical['processing_observations'] = {resource_id: owner.snapshot()}
    before = deepcopy((evidence, snapshot, task, physical))
    access = _machine_access_projection(evidence, snapshot, task, physical, part)
    assert access['correct_part_present'] is True and access['robot_access_clear'] is True
    assert access['source']['kind'] == 'projected' and len(access['source']['resources']) == 2
    assert access['physical_execution_verified'] is False and access['execution_authorized'] is False
    assert 'observed_at_simulation_sec' not in access
    record, reply = _machine_command_projection(evidence, snapshot, 'start_job',
        {'part_name': part, 'process': 'trim', 'result': 'square'}, task, physical)
    assert reply['job_state'] == 'running' and reply['process_completed'] is False
    assert record['jobs'][record['active_job']]['observations'] == access
    assert (evidence, snapshot, task, physical) == before


@pytest.mark.parametrize('resource_id', ['M1', 'M2'])
@pytest.mark.parametrize('condition', ['part_missing_support', 'part_location', 'part_moved', 'held',
    'KMR_occupied', 'handling_robot_occupied', 'KMR_ambiguous', 'handling_robot_ambiguous',
    'missing_link', 'stale_endpoint', 'missing_hold', 'invalidated_geometry'])
def test_machine_access_keeps_constraints_and_missing_evidence_distinct(resource_id, condition):
    from cais_spade_llm.agents.shared_information.recovery_validation_protocol import recovery_validation_fingerprint as fingerprint
    from cais_spade_llm.resources.environment_recovery import _machine_access_projection

    evidence, snapshot, task, physical, part = _machine_access_case(resource_id)
    machine = next(row for row in evidence['inputs']['scene']['machines'] if row['resource_id'] == resource_id)
    handling = machine['handling_robot']
    if condition == 'part_missing_support':
        physical['stationary_part_evidence'].pop(part)
    elif condition == 'part_location':
        physical['snapshot']['parts'][part]['location'] = 'Storage'
    elif condition == 'part_moved':
        physical['snapshot']['parts'][part]['current_pose'][0] += .2
        physical['stationary_part_evidence'][part]['current_pose'] = deepcopy(physical['snapshot']['parts'][part]['current_pose'])
    elif condition == 'held':
        physical['snapshot']['resources'][handling]['held_part'] = part
    elif condition in {'KMR_occupied', 'handling_robot_occupied', 'KMR_ambiguous', 'handling_robot_ambiguous'}:
        peer = 'KMR' if condition.startswith('KMR') else handling
        model = physical['resource_models'][peer]['configuration']
        margin = machine['KMR_process_clearance_m'] + .1 if peer == 'KMR' else .12
        model['root_xyz'][0] = machine['workholding_pose'][0] + (margin if condition.endswith('ambiguous') else .01)
        if peer == 'KMR':
            physical['resources'][peer]['projected_endpoint']['configuration_fingerprint'] = fingerprint(model)
    elif condition == 'missing_link':
        evidence['inputs']['scene']['KMR']['task_execution']['clearance_link_bounds']['missing'] = [{'center': [0., 0., 0.], 'radius': .1}]
    elif condition == 'stale_endpoint':
        physical['resources']['KMR']['projected_endpoint']['binding'] = {'id': 'old'}
    elif condition == 'missing_hold':
        physical['resources'][handling].pop('controller_hold_model')
    else:
        physical['invalidated_geometry'] = {'resources': {handling: {'reason': 'moved'}}}
    before = deepcopy((evidence, snapshot, task, physical))
    if condition.endswith('occupied'):
        assert _machine_access_projection(evidence, snapshot, task, physical, part)['robot_access_clear'] is False
    elif condition == 'missing_link':
        with pytest.raises(KeyError, match='missing'):
            _machine_access_projection(evidence, snapshot, task, physical, part)
    else:
        with pytest.raises(ValueError) as rejected:
            _machine_access_projection(evidence, snapshot, task, physical, part)
        assert ('NEEDS_CONTEXT:' in str(rejected.value)) is (condition not in {'part_moved', 'held'})
    assert (evidence, snapshot, task, physical) == before


def test_seeded_random_failure_uses_observed_overlap(machine):
    records = {}
    for rid in ('M1', 'M2'):
        declaration = deepcopy(machine)
        declaration['resource_id'] = rid
        owner = MachineJobs(declaration, 'run')
        start(owner)
        owner.observe(101, observed())
        records[rid] = owner.snapshot()
    kwargs = dict(now=101, seed=71, observation_interval_sec=.25)
    selected = processing_failure_time(records, ['M1', 'M2'], **kwargs)
    assert selected == processing_failure_time(records, ['M1', 'M2'], **kwargs)
    assert 101 < selected['interruption_simulation_sec'] < selected['overlap_end_simulation_sec']
    assert processing_failure_time({'M1': records['M1']}, ['M1', 'M2'], **kwargs) is None
    with pytest.raises(ValueError, match='Insufficient'):
        processing_failure_time(records, ['M1', 'M2'], **{**kwargs, 'now': 105})


def test_new_case_and_programs_do_not_change_other_defaults():
    from cais_spade_llm.recovery_framework.part_slippage_example import example_setup
    from cais_spade_llm.resources.gazebo_programs import validate_resource_programs
    scene = json.loads(SCENE_PATH.read_text())
    validate_resource_programs(scene)
    setup = example_setup(scenario='Machining breakdown during part processing')
    assert {'M1', 'M2', 'ur5e-2'} <= set(setup['permitted_resources'])
    assert setup['failure_scenario']['checkpoint'] == 'during_processing'
    assert len(setup['selected_safety_files']) == 2
    assert example_setup()['failure_scenario']['scenario'] == 'Part slippage'
    assert all(set(scene['resource_programs']['resources'][rid]['primitives']) == set(MACHINE_PRIMITIVES)
               for rid in ('M1', 'M2'))


def test_worker_commands_remain_responsive_during_machine_part(machine, tmp_path):
    from cais_spade_llm.recovery_framework.workflow_gazebo import WorkflowPrimitiveRunner
    scene = json.loads(SCENE_PATH.read_text())
    clock = [100.]
    position = SimpleNamespace(**dict(zip(('x', 'y', 'z'), machine['workholding_pose'][:3])))
    state = SimpleNamespace(pose=SimpleNamespace(position=position))
    control = tmp_path / 'control.json'
    sent = []
    def wait(duration, observe):
        begin = clock[0]
        clock[0] += duration
        if clock[0] >= 101 and not sent:
            control.write_text(json.dumps({'id': 'pause', 'request': {'run_id':'run', 'resource_id':'M2',
                'primitive':'pause_job', 'parameters':{'job_id':'task'}}}))
            sent.append('pause')
        elif clock[0] >= 102 and sent == ['pause']:
            reply = json.loads((tmp_path/'control_result.json').read_text())
            assert reply['result']['job_state'] == 'paused'
            assert reply['result']['process_completed'] is False
            control.write_text(json.dumps({'id': 'resume', 'request': {'run_id':'run', 'resource_id':'M2',
                'primitive':'resume_job', 'parameters':{'job_id':'task'}}}))
            sent.append('resume')
        observe(duration)
        assert clock[0] < 120
        return begin, clock[0]
    runner = WorkflowPrimitiveRunner({'scene':scene, 'geometry':{'RGOCG4-50_Round_4mm':{'model_name':'part'}},
        '_control_path':str(control), '_jobs_path':str(tmp_path/'jobs.json'),
        'task':{'task_id':'task', 'run_id':'run', 'resource_id':'M2', 'event_name':'machine_part',
                'parameters':{'part_name':'RGOCG4-50_Round_4mm', 'process':'trim', 'result':'circle'}}},
        entity=lambda _: state, set_pose=lambda *_:pytest.fail('job control dispatched motion'),
        robot_clearance=lambda *_:{'ur5e-2':1.}, current_clock=lambda:clock[0], wait_simulation=wait)
    runner._kmr_clearance = lambda:{'link':1.}
    result = runner.execute()
    assert sent == ['pause','resume']
    job = result['processing_observations']['jobs']['task']
    assert job['processing_active_sec'] == machine['simulation_process']['processing_time_sec']
    assert job['simulation_elapsed_sec'] > job['processing_active_sec']
    assert job['simulation_paused_sec'] >= 1.


def test_new_machine_ra_derives_pause_without_order_or_completion(machine):
    from cais_spade_llm.product.environment import EnvironmentProductContext
    from cais_spade_llm.recovery_framework import ROOT, PRODUCT_PATH, read_json
    from cais_spade_llm.resources.environment_recovery import environment_owner_evidence, environment_recovery_descriptor, check_environment_primitive_capability
    from cais_spade_llm.resources.recovery_feasibility import validate_primitive_support
    meta = next(iter(read_json(PRODUCT_PATH).values()))
    context = EnvironmentProductContext(scene=read_json(SCENE_PATH), product_order=read_json(ROOT/meta['product_order_file']),
                                        geometry=read_json(ROOT/meta['product_geometry_file'])['gazebo'])
    jobs = MachineJobs(machine, context.run_id)
    start(jobs)
    jobs.observe(101, observed())
    owner = SimpleNamespace(agent_name='M2', jid='M2@localhost', environment_context=context.resources['M2'],
        workflow_worker=SimpleNamespace(read_machine_jobs=jobs.snapshot), environment_runtime=SimpleNamespace(context=context, setup={}))
    context.resources['M2'].valuation.update(resource_state='loaded', part_name='RGOCG4-50_Round_4mm')
    evidence = environment_owner_evidence(owner)
    snapshot = evidence['snapshot']
    descriptor = environment_recovery_descriptor(evidence, snapshot, resource_jid=str(owner.jid))
    proposal = {'event_name':'new_pause_event', 'resource_jid':str(owner.jid), 'part_name':None,
        'expected_start_state':{'job_state':'running'},
        'expected_end_state':{'job_state':'paused', 'active_job':None}, 'params':{'job_id':'circle-job'}}
    original = deepcopy(evidence)
    result = validate_primitive_support(task=proposal, recovery_des_model=descriptor, recovery_snapshot=snapshot,
        validate_primitive=lambda **kw:check_environment_primitive_capability(**kw, task=proposal,
            recovery_snapshot=snapshot, resource_evidence=evidence))
    assert result['feasibility_status'] == 'FEASIBLE', result
    assert evidence == original and jobs.record['jobs']['circle-job']['process_completed'] is False
    assert descriptor['primitive_support_semantics'] == 'successor_primitive_union'


def test_active_processing_cannot_silently_freeze_during_another_event(machine):
    from cais_spade_llm.resources.machine.machine_jobs import missing_processing_projection
    owner = MachineJobs(machine, 'run')
    start(owner)
    records = {'M2': owner.snapshot()}
    before = {'processing_observations': records}
    proposal = {'status':'projected', 'observations':[{'time':0}, {'time':2}],
                'successor_evidence':{'processing_observations':deepcopy(records)}}
    assert missing_processing_projection(before, proposal) == ['M2/jobs/circle-job']
    assert before['processing_observations'] == records
    proposal['observations'][-1]['time'] = 0
    assert missing_processing_projection(before, proposal) == []


def test_processing_prompt_keeps_facts_without_command_history(machine):
    from cais_spade_llm.recovery_framework.part_slippage_recovery import _recorded_checkpoint
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes.multi_turn_prompts import _slim_fault_event
    owner = MachineJobs(machine, 'run')
    start(owner)
    owner.observe(101, observed())
    record = owner.snapshot()
    record['route_hints'] = 'LEAK_SENTINEL'
    record['jobs']['circle-job']['reference_recovery'] = 'LEAK_SENTINEL'
    source = {'failure':{'checkpoint':'during_processing'}, 'processing_observations':{'M2':record}}
    original = deepcopy(source)
    result = _slim_fault_event({'checkpoint':_recorded_checkpoint(source)})
    text = json.dumps(result)
    assert 'LEAK_SENTINEL' not in text and 'commands' not in text
    assert result['checkpoint']['processing_observations']['M2']['jobs']['circle-job']['processing_active_sec'] == 1
    assert source == original


@pytest.mark.parametrize('rid,part,result_name,job_id,next_program', [
    ('M1', 'KET4_Square_4mm', 'square', 'square-job', 'circle'),
    ('M2', 'RGOCG4-50_Round_4mm', 'circle', 'circle-job', 'square'),
])
@pytest.mark.parametrize('offline', [False, True])
@pytest.mark.parametrize('missing_peer_geometry', [False, True])
@pytest.mark.parametrize('endpoint_version', [None, 2])
def test_pause_and_program_selection_have_separate_owner_projections(
    monkeypatch, rid, part, result_name, job_id, next_program, offline, missing_peer_geometry, endpoint_version,
):
    import asyncio
    from cais_spade_llm.agents.shared_information.recovery_validation_protocol import recovery_validation_fingerprint as fingerprint
    from cais_spade_llm.product.environment import EnvironmentProductContext
    from cais_spade_llm.recovery_framework import ROOT, PRODUCT_PATH, read_json
    from cais_spade_llm.recovery_framework.offline_recovery import SavedRecoveryResource
    from cais_spade_llm.recovery_framework.outline_gazebo_evidence import prepare_owner_event_evidence
    from cais_spade_llm.recovery_framework.workflow_execution import WorkflowResourceAgent
    from cais_spade_llm.resources import resource_safety_preparation
    from cais_spade_llm.resources.environment_recovery import environment_recovery_descriptor, project_environment_event
    from test_environment_recovery import static_machine_projection_input
    from test_recovery_outline_evidence import _commanded_hold_case
    robot_preview = Mock(side_effect=AssertionError('Machine job control requested a robot preview'))
    monkeypatch.setattr(resource_safety_preparation, 'prepare_outline_event_preview_request', robot_preview)
    meta = next(iter(read_json(PRODUCT_PATH).values()))
    scene = read_json(SCENE_PATH)
    context = EnvironmentProductContext(scene=scene, product_order=read_json(ROOT/meta['product_order_file']),
                                        geometry=read_json(ROOT/meta['product_geometry_file'])['gazebo'])
    owner, evidence, snapshot, _model, physical = static_machine_projection_input(context, rid, part)
    _, _, peer_state, peer_model, _ = _commanded_hold_case()
    for peer in ('KMR', 'ur5e-2'):
        physical['resource_models'][peer] = deepcopy(peer_model)
        physical['snapshot']['resources'][peer].update({
            key: deepcopy(peer_state[key]) for key in ('joint_names', 'joint_positions')})
    declaration = next(row for row in scene['machines'] if row['resource_id'] == rid)
    jobs = MachineJobs(declaration, context.run_id)
    start(jobs, identity=job_id, part=part, result=result_name)
    jobs.observe(101, observed(part))
    evidence['processing_observations'] = jobs.snapshot()
    snapshot.update(job_public_state(jobs.snapshot()))
    evidence['environment_valuation'][rid].update(job_public_state(jobs.snapshot()))
    physical['environment_valuation'][rid].update(job_public_state(jobs.snapshot()))
    physical['snapshot']['resources'][rid].update(job_public_state(jobs.snapshot()))
    physical['processing_observations'] = {rid: jobs.snapshot()}
    physical['geometry']['regions'] = deepcopy(scene['safety_preparation']['regions'])
    for peer, values in physical['snapshot']['resources'].items():
        if 'held_part' in values:
            physical['geometry']['resources'][peer]['stationary_only'] = False
    for values in physical['snapshot']['parts'].values():
        values.update(processCompleted_complete=True, processCompleted_evidence={
            'complete': True, 'source_kind': 'recorded', 'checkpoint': 'local fixture'})
    if endpoint_version == 2:
        physical['endpoint_evidence_policy_version'] = endpoint_version
        physical['resources'][rid].pop('model_static')
        physical['geometry']['resources'][rid].pop('component_bounds')
    if missing_peer_geometry:
        for collection in (physical['snapshot']['resources'], physical['geometry']['resources'], physical['resources']):
            del collection['ur5e-4']
    original = deepcopy(physical)
    for params, end in [({'job_id':job_id}, {'job_state':'paused', 'active_job':None}),
                         ({'program':next_program}, {'selected_program':next_program})]:
        model = environment_recovery_descriptor(evidence, snapshot, resource_jid=owner.jid)
        task = {'event_name':'provider_proposal', 'resource_jid':owner.jid, 'part_name':None,
                'params':params, 'expected_start_state':{}, 'expected_end_state':end}
        if 'selected_program' in end:
            from cais_spade_llm.recovery_framework.outline_endpoint_evidence import _occupancy_binding, _readiness

            for peer in ('KMR', 'ur5e-2'):
                physical['resources'][peer]['projected_endpoint'] = {
                    'binding': deepcopy(physical['provenance']['binding']),
                    'state_fingerprint': fingerprint(physical['snapshot']['resources'][peer]),
                    'configuration_fingerprint': fingerprint(physical['resource_models'][peer]['configuration']),
                    'condition': 'successful_owner_action_completion', 'physical_execution_verified': False,
                }
            if endpoint_version == 2:
                physical['endpoint_occupancy']['binding'] = _occupancy_binding(physical)
                if not offline and not missing_peer_geometry:
                    for stale_field in ('binding', 'state_fingerprint', 'configuration_fingerprint'):
                        changed = deepcopy(physical)
                        changed['resources']['KMR']['projected_endpoint'][stale_field] = 'stale'
                        changed['endpoint_occupancy']['binding'] = _occupancy_binding(changed)
                        stale = project_environment_event(task=task, recovery_snapshot=snapshot, recovery_des_model=model,
                            physical_input={'resource_evidence': evidence, 'physical_evidence': changed})
                        assert stale['status'] == 'projected'
                        assert stale['successor_evidence']['resources']['KMR']['projected_endpoint'][stale_field] == 'stale'
                        with pytest.raises(ValueError, match='Projected starting configuration is stale'):
                            _readiness(stale['successor_evidence'], 'KMR')
        if endpoint_version == 2 and 'selected_program' in end:
            physical['processing_observations'][rid].pop('simulation_clock_sec')
            changed = deepcopy(physical)
            changed['snapshot']['parts'][part]['current_pose'][0] += .25
            stale = project_environment_event(task=task, recovery_snapshot=snapshot, recovery_des_model=model,
                physical_input={'resource_evidence': evidence, 'physical_evidence': changed})
            assert stale['status'] == 'needs_context'
            assert any('Retained endpoint occupancy is stale' in reason for reason in stale['unresolved'])
        expected = project_environment_event(task=task, recovery_snapshot=snapshot, recovery_des_model=model,
            physical_input={'resource_evidence':evidence, 'physical_evidence':physical})
        if offline:
            validating_owner = SavedRecoveryResource(owner.jid, {'resource_type': 'resource',
                'recovery_snapshot': snapshot, 'recovery_des_model': model, 'resource_evidence': evidence},
                {'jids': {rid: owner.jid}})
        else:
            validating_owner = owner
            validating_owner.project_recovery_outline_event = (
                WorkflowResourceAgent.project_recovery_outline_event.__get__(owner))
        before = deepcopy((task, snapshot, model, physical, evidence))
        prepared = asyncio.run(prepare_owner_event_evidence(validating_owner, task=task,
            recovery_snapshot=snapshot, recovery_des_model=model, physical_evidence=physical,
            resource_evidence=evidence, observations={}, allow_live_capture=False))
        assert prepared['status'] == 'prepared', prepared
        assert prepared['geometry_evidence'] == expected
        assert (task, snapshot, model, physical, evidence) == before
        result = prepared['geometry_evidence']
        assert result['status'] == 'projected', result
        assert 'owner_safety_evidence_unresolved' not in result, result.get('owner_safety_evidence_unresolved')
        assert [row['owner_completion_applied'] for row in result['observations']] == [False, True]
        assert result['observations'][0]['state_snapshot'][rid] != result['observations'][-1]['state_snapshot'][rid]
        from cais_spade_llm.agents.central_controller.outline_evidence_safety import _owner_ledgers

        tampered = deepcopy(result['observations'][0])
        tampered['owner_completion_applied'] = True
        with pytest.raises(ValueError, match='another physical checkpoint'):
            _owner_ledgers(tampered, result['owner_safety_evidence']['resource_models'], result['geometry']['parts'])
        assert result['command_sent'] is False and result['execution_authorized'] is False
        assert result['observations'][-1]['time'] == 0
        if 'selected_program' in end:
            for peer in ('KMR', 'ur5e-2'):
                endpoint = result['successor_evidence']['resources'][peer]['projected_endpoint']
                assert endpoint['binding'] == result['successor_evidence']['provenance']['binding']
                assert endpoint['preserved_endpoint']['fingerprint'] == fingerprint(physical['resources'][peer]['projected_endpoint'])
                assert endpoint['physical_execution_verified'] is False
                if endpoint_version == 2:
                    readiness = _readiness(result['successor_evidence'], peer)
                    assert readiness['kind'] == 'conditional_successor' and readiness['status'] == 'missing'
        if endpoint_version == 2:
            assert 'model_static' not in result['successor_evidence']['resources'][rid]
            assert 'component_bounds' not in result['successor_evidence']['geometry']['resources'][rid]
            assert result['observations'][0]['occupancy_possibilities'] == result['observations'][-1]['occupancy_possibilities']
            if 'selected_program' in end:
                assert 'simulation_clock_sec' not in result['successor_evidence']['processing_observations'][rid]
        if missing_peer_geometry:
            from cais_spade_llm.agents.central_controller.outline_evidence_safety import validate_outline_evidence_safety

            specification = read_json(ROOT / 'cais_spade_llm/specification/safety/safety_M2_mutex.txt')
            cca = validate_outline_evidence_safety(projection=result, catalog=specification['catalog'],
                requirement_scopes=specification['requirement_scopes'], prospective_activation=True)
            assert cca['validation_status'] == 'NEEDS_CONTEXT'
            assert not cca['is_safe']
            assert 'ur5e-4' in result['owner_safety_evidence']['resource_models']
        else:
            from cais_spade_llm.agents.central_controller.outline_evidence_safety import validate_outline_evidence_safety

            for filename in ('safety_M2_mutex.txt',
                             'safety_KET4_Square_4mm_before_RGOCG4-50_Round_4mm_processing.txt'):
                specification = read_json(ROOT / 'cais_spade_llm/specification/safety' / filename)
                cca = validate_outline_evidence_safety(projection=result, catalog=specification['catalog'],
                    requirement_scopes=specification['requirement_scopes'], prospective_activation=True)
                assert cca['is_safe'], cca['reason']
        physical, snapshot = result['successor_evidence'], result['projected_resource_snapshot']
    configuration = physical['processing_observations'][rid]['current_configuration']
    assert configuration['tool'] == declaration['current_configuration']['tool']
    assert configuration['workholding'] == declaration['current_configuration']['workholding']
    assert physical['snapshot']['resources']['ur5e-2'] == original['snapshot']['resources']['ur5e-2']
    assert evidence['processing_observations']['jobs'][job_id]['job_state'] == 'running'
    assert physical['processing_observations'][rid]['jobs'][job_id]['job_state'] == 'paused'
    assert physical['processing_observations'][rid]['jobs'][job_id]['process_completed'] is False
    assert physical['snapshot']['parts'] == original['snapshot']['parts']
    robot_preview.assert_not_called()


@pytest.mark.parametrize('rid,part,result_name,job_id,processing_safe', [
    ('M1', 'KET4_Square_4mm', 'square', 'square-job', True),
    ('M2', 'RGOCG4-50_Round_4mm', 'circle', 'circle-job', False),
])
@pytest.mark.parametrize('partial', [False, True])
def test_endpoint_nominal_machine_completion_uses_remaining_job_time(
        rid, part, result_name, job_id, processing_safe, partial, monkeypatch):
    from cais_spade_llm.agents.central_controller.outline_evidence_safety import validate_outline_evidence_safety
    from cais_spade_llm.product.environment import EnvironmentProductContext
    from cais_spade_llm.recovery_framework import ROOT, PRODUCT_PATH, read_json
    from cais_spade_llm.recovery_framework.outline_validation import load_outline_specifications
    from cais_spade_llm.resources.environment_recovery import environment_recovery_descriptor, project_environment_event
    from test_environment_recovery import static_machine_projection_input

    monkeypatch.setattr('cais_spade_llm.recovery_framework.recovery_motion_preview.preview_recovery_motion_from_files',
                        Mock(side_effect=AssertionError('Machine completion requested a robot trajectory')))
    meta = next(iter(read_json(PRODUCT_PATH).values()))
    scene = read_json(SCENE_PATH)
    context = EnvironmentProductContext(scene=scene, product_order=read_json(ROOT/meta['product_order_file']),
        geometry=read_json(ROOT/meta['product_geometry_file'])['gazebo'])
    owner, evidence, snapshot, _, physical = static_machine_projection_input(context, rid, part)
    declaration = next(row for row in scene['machines'] if row['resource_id'] == rid)
    jobs = MachineJobs(declaration, context.run_id)
    start(jobs, identity=job_id, part=part, result=result_name)
    jobs.observe(101, observed(part))
    evidence['processing_observations'] = jobs.snapshot()
    for values in (snapshot, evidence['environment_valuation'][rid],
                   physical['environment_valuation'][rid], physical['snapshot']['resources'][rid]):
        values.update(job_public_state(jobs.snapshot()))
    specifications = load_outline_specifications([
        'cais_spade_llm/specification/safety/safety_M2_mutex.txt',
        'cais_spade_llm/specification/safety/safety_KET4_Square_4mm_before_RGOCG4-50_Round_4mm_processing.txt',
    ], ROOT)
    physical.update(processing_observations={rid: jobs.snapshot()}, endpoint_evidence_policy_version=2,
                    outline_safety_specifications=specifications)
    if partial:
        physical['outline_evidence_policy_version'] = 2
        physical['resources']['KMR'] = {'model_static': False, 'idle': False}
        physical['geometry']['resources']['KMR']['stationary_only'] = False
    physical['geometry']['regions'] = deepcopy(scene['safety_preparation']['regions'])
    for peer, values in physical['snapshot']['resources'].items():
        if 'held_part' in values:
            physical['geometry']['resources'][peer]['stationary_only'] = False
    for values in physical['snapshot']['parts'].values():
        values.update(processCompleted_complete=True, processCompleted_evidence={
            'complete': True, 'source_kind': 'recorded', 'checkpoint': 'local fixture'})
    model = environment_recovery_descriptor(evidence, snapshot, resource_jid=owner.jid)
    task = {'event_name': 'nominal_processing', 'resource_jid': owner.jid, 'part_name': part,
        'required_task_contract': 'machine_part', 'params': {'process': 'trim', 'result': result_name},
        'expected_end_state': {'resource_state': 'completed', 'part_state': 'completed'}}
    original = deepcopy((physical, evidence, snapshot))
    projection = project_environment_event(task=task, recovery_snapshot=snapshot, recovery_des_model=model,
        physical_input={'resource_evidence': evidence, 'physical_evidence': physical})
    assert projection['status'] == 'projected', projection
    assert projection['observations'][-1]['time'] == declaration['simulation_process']['processing_time_sec'] - 1
    completed = projection['successor_evidence']['processing_observations'][rid]['jobs'][job_id]
    assert completed['process_completed'] is True and completed['job_state'] == 'completed'
    if partial:
        assert 'KMR' in projection['successor_evidence']['invalidated_geometry']['resources']
        assert all(values['KMR'] == [False, True] for point in projection['observations'][1:]
                   for values in point['occupancy_possibilities'].values())
    assert (physical, evidence, snapshot) == original
    assert projection['command_sent'] is False and projection['execution_authorized'] is False
    for specification, expected in zip(specifications, (True, processing_safe), strict=True):
        document = specification['document']
        cca = validate_outline_evidence_safety(projection=projection, catalog=document['catalog'],
            requirement_scopes=document['requirement_scopes'], prospective_activation=True)
        assert cca['is_safe'] is expected, cca


def test_random_worker_failure_preserves_both_jobs_before_shutdown(tmp_path):
    from cais_spade_llm.recovery_framework.workflow_gazebo import WorkflowPrimitiveRunner
    scene = json.loads(SCENE_PATH.read_text())
    machines = {row['resource_id']: row for row in scene['machines']}
    peer = MachineJobs(machines['M2'], 'run')
    start(peer)
    paths = {rid: str(tmp_path/(rid+'.json')) for rid in machines}
    clock = [100.]
    fault = tmp_path/'fault.json'
    fault.write_text(json.dumps({'run_id':'run','resource_id':'M1','status':'armed'}))
    position = SimpleNamespace(**dict(zip(('x','y','z'), machines['M1']['workholding_pose'][:3])))
    state = SimpleNamespace(pose=SimpleNamespace(position=position))
    def publish_peer():
        from pathlib import Path
        Path(paths['M2']).write_text(json.dumps(peer.snapshot()))
    publish_peer()
    def wait(duration, observe):
        before = clock[0]
        clock[0] += duration
        peer.observe(clock[0], observed())
        publish_peer()
        observe(duration)
        return before, clock[0]
    runner = WorkflowPrimitiveRunner({'scene':scene, 'geometry':{'KET4_Square_4mm':{'model_name':'part'}},
        '_jobs_path':paths['M1'], 'task':{'run_id':'run','resource_id':'M1','task_id':'square-job',
            'event_name':'machine_part','parameters':{'part_name':'KET4_Square_4mm','process':'trim','result':'square'}},
        'failure_injection':{'run_id':'run','resource_id':'M1','task_id':'square-job', 'seed':11,
            'checkpoint':'during_processing','control_path':str(fault),'processing_paths':paths,
            'evidence_path':str(tmp_path/'interruption.json')}},
        entity=lambda _:state, set_pose=lambda *_:pytest.fail('processing dispatched motion'),
        robot_clearance=lambda *_:{'ur5e-1':1.}, current_clock=lambda:clock[0], wait_simulation=wait)
    runner._kmr_clearance = lambda:{'link':1.}
    with pytest.raises(RuntimeError, match='Machining breakdown'):
        runner.execute()
    capture = json.loads((tmp_path/'interruption.json').read_text())
    assert capture['actual_interruption_simulation_sec'] >= capture['chosen_timing']['interruption_simulation_sec']
    for rid, record in capture['processing_observations'].items():
        job = record['jobs'][record['active_job']]
        assert job['job_state'] == 'running' and job['process_completed'] is False
        assert 0 < job['processing_fraction'] < 1
    before = deepcopy(capture)
    peer.interrupt('shutdown')
    runner.machine_jobs.interrupt('shutdown')
    assert capture == before
    assert peer.record['active_job'] is None


def machine_capture(machine):
    """Build the recorded null snapshot and running worker state from one run."""
    jobs = MachineJobs(machine, 'run')
    start(jobs)
    fields = job_public_state(jobs.snapshot())
    owner = {'snapshot': deepcopy(fields), 'state': deepcopy(fields),
        'environment_valuation': {'M2': deepcopy(fields)}, 'processing_observations': jobs.snapshot(),
        'source': {'run_id': 'run', 'revision': 7, 'kind': 'recorded_owner_state'}}
    state = {'active_job': None, 'job_state': None, 'selected_program': 'circle', 'current_pose': [0, 0, 0, 0, 0, 0, 1]}
    physical = {'snapshot': {'frame': 'world', 'resources': {'M2': deepcopy(state)}},
        'provenance': {'run_id': 'run', 'context_revision': 7},
        'resources': {'M2': {'observation_state': deepcopy(state), 'frame': 'world',
            'source': 'registered owner / GETRECOVERYSTATE', 'simulation_time': 102., 'observed_monotonic': 500.}}}
    inputs = {'scene': {'machines': [machine]}}
    return owner, physical, inputs


def test_physical_refresh_preserves_worker_fields_and_raw_nulls(machine):
    from cais_spade_llm.recovery_framework.recovery_observations import capture_resource_evidence

    evidence, physical, inputs = machine_capture(machine)
    original = deepcopy((evidence, physical))
    owner = SimpleNamespace(agent_name='M2', jid='M2@localhost',
        get_recovery_resource_evidence=Mock(return_value=evidence))
    runtime = SimpleNamespace(context=SimpleNamespace(run_id='run', revision=7, inputs=inputs),
                              resource_agents=[owner], setup={'execution_mode': 'simulation'})
    for measured in ({'status': 'pending'}, physical):
        result = capture_resource_evidence(runtime, measured)['M2@localhost']
        assert result['snapshot']['active_job'] == 'circle-job'
        assert result['snapshot']['job_state'] == 'running'
        assert result['conflicts'] == []
    assert result['physical_snapshot']['active_job'] is None
    assert result['snapshot_sources']['active_job']['revision'] == evidence['processing_observations']['revision']
    assert (evidence, physical) == original


def test_absent_worker_record_does_not_mean_no_active_job(machine):
    from cais_spade_llm.recovery_framework.recovery_observations import reconcile_resource_snapshot
    from cais_spade_llm.recovery_framework.offline_recovery import SavedRecoveryResource

    evidence, physical, inputs = machine_capture(machine)
    evidence.pop('processing_observations')
    result = reconcile_resource_snapshot(evidence, physical, resource_id='M2', run_id='run', revision=7, inputs=inputs)
    assert all(field not in result['snapshot'] for field in ('active_job', 'job_state', 'selected_program'))
    assert result['physical_snapshot']['active_job'] is None
    assert result['missing'][0]['ref'] == 'processing_observations/M2'
    entry = {'recovery_snapshot': {'active_job': None, 'job_state': None, 'selected_program': 'circle'},
             'recovery_des_model': {'state_variables': {name: {'scope': 'resource'}
                                   for name in ('active_job', 'job_state', 'selected_program')}}}
    observations = {'jids': {'M2': 'M2@localhost'}, 'resource_evidence': {'M2@localhost': result}}
    saved = SavedRecoveryResource('M2@localhost', entry, observations)
    assert all(field not in saved.get_recovery_snapshot() for field in ('active_job', 'job_state', 'selected_program'))


@pytest.mark.parametrize('defect', ['run_id', 'resource_id', 'revision', 'active_job', 'job_id', 'model_fingerprint'])
def test_capture_checks_machine_job_identity_and_revision(machine, defect):
    from cais_spade_llm.recovery_framework.recovery_observations import recorded_processing_fields

    evidence, _, inputs = machine_capture(machine)
    record = evidence['processing_observations']
    if defect == 'job_id':
        record['jobs']['circle-job']['job_id'] = 'another-job'
    else:
        record[defect] = -1 if defect == 'revision' else 'another-value'
    with pytest.raises(ValueError):
        recorded_processing_fields(record, resource_id='M2', run_id='run', inputs=inputs)


@pytest.mark.parametrize('value', [None, 'another-job'])
def test_capture_keeps_authoritative_machine_disagreement(machine, value):
    from cais_spade_llm.recovery_framework.recovery_observations import reconcile_resource_snapshot

    evidence, physical, inputs = machine_capture(machine)
    evidence['snapshot']['active_job'] = value
    result = reconcile_resource_snapshot(evidence, physical, resource_id='M2', run_id='run', revision=7, inputs=inputs)
    assert result['conflicts'][0]['recorded'] == value
    assert result['conflicts'][0]['observed'] == 'circle-job'


def test_observation_repair_is_immutable_and_does_not_use_shutdown(machine):
    from cais_spade_llm.recovery_framework.recovery_observations import derive_recorded_observations

    evidence, physical, inputs = machine_capture(machine)
    evidence['environment_models'] = {'M2': {}}
    evidence['snapshot'] = deepcopy(physical['snapshot']['resources']['M2'])
    jobs = deepcopy(evidence['processing_observations'])
    record = {'run_id': 'run', 'context_revision': 7, 'jids': {'M2': 'M2@localhost'}, 'inputs': inputs,
        'resource_evidence': {'M2@localhost': evidence}, 'physical_evidence': physical,
        'processing_observations': {'M2': jobs}, 'processing_shutdown': {'M2': {'active_job': None}},
        'failure': {'processing_observations': {'M2': deepcopy(jobs)}}}
    before = deepcopy(record)
    result = derive_recorded_observations(record, source_path='/original/observations.json', source_sha256='original-hash')
    assert record == before
    assert result['physical_evidence'] == before['physical_evidence']
    assert result['resource_evidence']['M2@localhost']['snapshot']['active_job'] == 'circle-job'
    assert result['processing_shutdown'] == before['processing_shutdown']
    assert result['observation_repair']['source_sha256'] == 'original-hash'
    record['failure']['processing_observations']['M2']['revision'] += 1
    with pytest.raises(ValueError, match='Conflicting recorded processing'):
        derive_recorded_observations(record, source_path='/original', source_sha256='hash')


def test_processing_and_later_geometry_do_not_claim_one_observation(machine):
    from cais_spade_llm.recovery_framework.recovery_observations import processing_capture_compatibility

    evidence, physical, _ = machine_capture(machine)
    observations = {'processing_observations': {'M2': evidence['processing_observations']}, 'physical_evidence': physical}
    result = processing_capture_compatibility(observations)
    assert result['status'] == 'NEEDS_CONTEXT'
    assert result['checks'][0]['processing_simulation_time'] == 100
    assert result['checks'][0]['physical_simulation_times'] == {'M2': 102}
    assert result['processing_shutdown_used'] is False
    physical['resources']['M2']['simulation_time'] = 100.
    assert processing_capture_compatibility(observations)['status'] == 'NEEDS_CONTEXT'
    assert 'Equal simulation times' in processing_capture_compatibility(observations)['checks'][0]['reason']


def frozen_machine_capture(machine):
    """Local checkpoint fixture; never provider input or a historical capture."""
    from cais_spade_llm.recovery_framework.recovery_observations import checkpoint_fact_fingerprint

    evidence, physical, inputs = machine_capture(machine)
    jobs = MachineJobs(machine, 'run', record=evidence['processing_observations'])
    jobs.observe(102., observed())
    record = jobs.snapshot()
    physical['resources']['M2']['native_physics_source'] = {'instance_id': 'owned-world'}
    capture = {'run_id': 'run', 'context_revision': 7, 'inputs': inputs,
        'jids': {'M2': 'M2@localhost'}, 'processing_observations': {'M2': record},
        'resource_evidence': {'M2@localhost': evidence}, 'physical_evidence': physical,
        'planning_scene_capture': {'scene_fingerprint': 'recorded-scene'}}
    capture['capture_checkpoint'] = {'version': 1, 'checkpoint_id': 'checkpoint',
        'run_id': 'run', 'context_revision': 7, 'simulation_time': 102., 'world_paused': True,
        'instance_id': 'owned-world', 'native_observations_unchanged': True,
        'facts_fingerprint': checkpoint_fact_fingerprint(capture), 'machine_observations': {
            'M2': {'status': 'observed', 'command_sent': False, 'checkpoint_id': 'checkpoint',
                   'processing_observations': deepcopy(record), 'native_source': {
                       'world_paused': True, 'simulation_time': 102., 'instance_id': 'owned-world'}}}}
    return capture


def test_verified_frozen_checkpoint_preserves_running_job_and_shutdown(machine):
    from cais_spade_llm.recovery_framework.recovery_observations import processing_capture_compatibility

    captured = frozen_machine_capture(machine)
    captured['processing_shutdown'] = {'M2': {'active_job': None, 'job_state': 'interrupted'}}
    before = deepcopy(captured)
    assert processing_capture_compatibility(captured)['status'] == 'supported'
    assert captured == before
    assert captured['processing_observations']['M2']['active_job'] == 'circle-job'


@pytest.mark.parametrize('defect', ['time', 'revision', 'snapshot', 'model', 'job', 'scene', 'source', 'unfrozen'])
def test_frozen_checkpoint_rejects_changed_or_unbound_evidence(machine, defect):
    from cais_spade_llm.recovery_framework.recovery_observations import processing_capture_compatibility

    captured = frozen_machine_capture(machine)
    if defect == 'time':
        captured['physical_evidence']['resources']['M2']['simulation_time'] = 102.00001
    elif defect == 'revision':
        captured['context_revision'] += 1
    elif defect == 'snapshot':
        captured['physical_evidence']['snapshot']['resources']['M2']['held_part'] = 'different-part'
    elif defect == 'model':
        captured['inputs']['scene']['machines'][0]['selected_program'] = 'square'
    elif defect == 'job':
        captured['processing_observations']['M2']['jobs']['circle-job']['job_state'] = 'completed'
    elif defect == 'scene':
        captured['planning_scene_capture']['scene_fingerprint'] = 'different-scene'
    elif defect == 'source':
        captured['capture_checkpoint']['machine_observations']['M2']['native_source']['instance_id'] = 'other-world'
    else:
        captured['capture_checkpoint']['world_paused'] = False
    assert processing_capture_compatibility(captured)['status'] == 'NEEDS_CONTEXT'


@pytest.mark.parametrize('clock,completed', [(101., False), (105., True)])
def test_worker_checkpoint_observes_processing_without_job_commands(machine, tmp_path, clock, completed):
    from cais_spade_llm.recovery_framework.workflow_gazebo import WorkflowPrimitiveRunner

    owner = MachineJobs(machine, 'run')
    start(owner)
    commands = deepcopy(owner.record['commands'])
    control = tmp_path/'control.json'
    runner = WorkflowPrimitiveRunner({'scene': {}, '_control_path': str(control),
        '_jobs_path': str(tmp_path/'jobs.json'), 'task': {'run_id': 'run', 'resource_id': 'M2', 'parameters': {}}},
        entity=Mock(side_effect=AssertionError('unexpected entity mutation')),
        set_pose=Mock(side_effect=AssertionError('motion forbidden')), robot_clearance=Mock(),
        current_clock=lambda: clock, wait_simulation=Mock(), machine_jobs=owner,
        checkpoint_observation=lambda: {'instance_id': 'world', 'simulation_time': clock, 'world_paused': True})
    runner._job_observation = lambda part: observed(part)
    control.write_text(json.dumps({'id': 'read', 'request': {'run_id': 'run', 'resource_id': 'M2',
        'checkpoint_id': 'capture', 'observation_only': True}}))
    runner._poll_machine_control()
    reply = json.loads((tmp_path/'control_result.json').read_text())['result']
    assert reply['status'] == 'observed' and reply['command_sent'] is False
    record = reply['processing_observations']
    assert record['commands'] == commands
    assert record['jobs']['circle-job']['process_completed'] is completed
    assert record['jobs']['circle-job']['job_state'] == ('completed' if completed else 'running')
    revision = record['revision']
    runner._poll_machine_control()
    runner.tick_machine()
    assert owner.record['revision'] == revision


def test_checkpoint_readiness_rejects_legacy_equal_timestamps(machine):
    from cais_spade_llm.recovery_framework.recovery_observations import (
        offline_preparation_missing, processing_capture_compatibility,
    )

    captured = frozen_machine_capture(machine)
    captured.pop('capture_checkpoint')
    captured.update(scenario='Machining breakdown during part processing',
        failure={'injection_status': 'completed'}, prepared_recovery_request={
            'llm_input': {}, 'recovery_session': {}, 'recovery_debug': {}, 'recovery_resources': {}})
    compatibility = processing_capture_compatibility(captured)
    assert compatibility['status'] == 'NEEDS_CONTEXT'
    assert any('Equal simulation times' in row['reason'] for row in compatibility['checks'])
    assert 'prepared_recovery_request.recovery_resources' in offline_preparation_missing(captured)


def test_missing_capture_compatibility_does_not_overwrite_ra_feasible():
    from cais_spade_llm.recovery_framework.outline_validation import validate_resource_outline_payload

    projection = Mock(return_value={'status': 'projected', 'observations': [], 'interval_complete': True})
    owner = SimpleNamespace(jid='M2@localhost', get_recovery_snapshot=lambda: {},
        recovery_des_model=lambda **_: {}, project_recovery_outline_event=projection,
        validate_recovery_primitive_support=lambda **_: {'allowed': True, 'feasibility_status': 'FEASIBLE'})
    result = validate_resource_outline_payload(owner, {'primitive_support_semantics': 'successor_primitive_union',
        'candidates': [{'task': {'resource_jid': 'M2@localhost'}, 'physical_input': {'physical_evidence': {
            'processing_capture_compatibility': {'status': 'NEEDS_CONTEXT', 'checks': [
                {'status': 'NEEDS_CONTEXT', 'reason': 'Recorded observation times differ'}]}}}}]}, offline=True)['results'][0]
    assert result['feasibility_status'] == 'FEASIBLE'
    assert result['geometry_evidence']['status'] == 'needs_context'
    assert result['geometry_evidence']['interval_complete'] is False
    assert result['execution_authorized'] is False
    projection.assert_called_once()
    assert result['geometry_evidence']['capture_compatibility']['checks'][0]['ref'] == '/physical_evidence/outline_safety_specifications'


def test_unrelated_processing_does_not_veto_owner_projection(machine):
    from cais_spade_llm.recovery_framework.recovery_observations import projection_capture_compatibility

    jobs = MachineJobs(machine, 'run')
    start(jobs)
    physical = {'processing_observations': {'M2': jobs.snapshot()}, 'outline_safety_specifications': [],
        'processing_capture_compatibility': {'status': 'NEEDS_CONTEXT', 'checks': [
            {'resource_id': 'M2', 'status': 'NEEDS_CONTEXT', 'reason': 'No shared checkpoint'}]}}
    before = deepcopy(physical)
    checked = projection_capture_compatibility(physical=physical, task={'resource_jid': 'Conveyor@localhost'},
        resource_evidence={'resource_id': 'Conveyor'}, projection={'observations': [{'time': 0}, {'time': 0}]})
    assert checked['status'] == 'supported' and checked['required_resources'] == []
    assert physical == before
    checked = projection_capture_compatibility(physical=physical, task={'resource_jid': 'Conveyor@localhost'},
        resource_evidence={'resource_id': 'Conveyor'}, projection={'observations': [{'time': 0}, {'time': 1}]})
    assert checked['status'] == 'NEEDS_CONTEXT' and checked['required_resources'] == ['M2']
    assert 'processing_active_sec' in checked['checks'][0]['missing_fields']
    physical['processing_capture_compatibility']['integrity_errors'] = [
        {'status': 'NEEDS_CONTEXT', 'reason': 'Recorded checkpoint facts changed'}]
    checked = projection_capture_compatibility(physical=physical, task={'resource_jid': 'Conveyor@localhost'},
        resource_evidence={'resource_id': 'Conveyor'}, projection={'observations': [{'time': 0}, {'time': 0}]})
    assert checked['status'] == 'NEEDS_CONTEXT'
    assert checked['checks'][0]['reason'] == 'Recorded checkpoint facts changed'


@pytest.mark.parametrize('elapsed,can_complete', [(1., False), (4., True), (5., True)])
def test_processing_time_bound_does_not_invent_control_state(machine, elapsed, can_complete):
    from cais_spade_llm.recovery_framework.recovery_observations import projection_capture_compatibility

    jobs = MachineJobs(machine, 'run')
    start(jobs)
    jobs.observe(101, observed())
    physical = {'processing_observations': {'M2': jobs.snapshot()}, 'outline_safety_specifications': [],
        'resources': {'M2': {'simulation_time': 101. + elapsed}},
        'processing_capture_compatibility': {'status': 'NEEDS_CONTEXT', 'checks': []}}
    before = deepcopy(physical)
    result = projection_capture_compatibility(physical=physical, task={'resource_jid': 'M2@localhost'},
        resource_evidence={'resource_id': 'M2', 'inputs': {'scene': {'machines': [machine]}}})
    check = result['checks'][0]
    assert check['calculations'][0]['completion_possible'] is can_complete
    assert check['calculations'][0]['kind'] == 'calculated_bound'
    assert {'active_job', 'job_state'} <= set(check['missing_fields'])
    assert ('process_completed' in check['missing_fields']) is can_complete
    assert result['status'] == 'NEEDS_CONTEXT' and physical == before


def test_unchanged_timestamp_does_not_certify_owner_control_or_stale_model(machine):
    from cais_spade_llm.recovery_framework.recovery_observations import projection_capture_compatibility

    jobs = MachineJobs(machine, 'run')
    start(jobs)
    physical = {'processing_observations': {'M2': jobs.snapshot()}, 'outline_safety_specifications': [],
        'resources': {'M2': {'simulation_time': 100.}},
        'processing_capture_compatibility': {'status': 'NEEDS_CONTEXT', 'checks': []}}
    evidence = {'resource_id': 'M2', 'inputs': {'scene': {'machines': [machine]}}}
    assert projection_capture_compatibility(physical=physical, task={}, resource_evidence=evidence)['status'] == 'NEEDS_CONTEXT'
    physical['provenance'] = {'run_id': 'different-run'}
    with pytest.raises(ValueError, match='Stale machine job record: run_id'):
        projection_capture_compatibility(physical=physical, task={}, resource_evidence=evidence)
    physical.pop('provenance')
    physical['processing_observations']['M2']['model_fingerprint'] = 'stale'
    with pytest.raises(ValueError, match='Stale machine job record'):
        projection_capture_compatibility(physical=physical, task={}, resource_evidence=evidence)


def test_selected_completion_aps_require_matching_job_facts_only(machine):
    from cais_spade_llm.agents.central_controller.ppr_ap import ap_record, make_ap_definition
    from cais_spade_llm.recovery_framework.recovery_observations import projection_capture_compatibility

    jobs = MachineJobs(machine, 'run')
    start(jobs)
    ap = ap_record('ap001', make_ap_definition('ap_state', 'RGOCG4-50_Round_4mm', 'trim', 'any',
        'processCompleted', {'result': 'circle'}), 'declared completion')
    physical = {'processing_observations': {'M2': jobs.snapshot()},
        'outline_safety_specifications': [{'document': {'catalog': {'specifications': [{'id': 'fixture', 'aps': [ap]}]}}}],
        'resources': {'M2': {'simulation_time': 101.}},
        'processing_capture_compatibility': {'status': 'NEEDS_CONTEXT', 'checks': []}}
    result = projection_capture_compatibility(physical=physical, task={},
        resource_evidence={'resource_id': 'Conveyor', 'inputs': {'scene': {'machines': [machine]}}})
    assert result['status'] == 'supported'
    assert result['required_resources'] == ['M2']
    assert result['checks'][0]['required_fields'] == ['process_completed']
    ap = ap_record('ap001', make_ap_definition('ap_state', 'any', 'any', 'M2',
        'job_state=paused'), 'declared job state')
    physical['outline_safety_specifications'][0]['document']['catalog']['specifications'][0]['aps'] = [ap]
    result = projection_capture_compatibility(physical=physical, task={},
        resource_evidence={'resource_id': 'Conveyor', 'inputs': {'scene': {'machines': [machine]}}})
    assert result['status'] == 'NEEDS_CONTEXT'
    assert 'job_state' in result['checks'][0]['missing_fields']
