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
    job_public_state, machine_public_state, public_processing_observations,
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


def _machine_access_case(resource_id):
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
    evidence = {'resource_id': resource_id, 'inputs': {'scene': scene}}
    task = {'outline_id': 'provider_proposal', 'event_name': 'machine_part', 'resource_jid': resource_id + '@localhost',
        'part_name': part, 'expected_end_state': {'resource_state': 'completed'}}
    return evidence, snapshot, task, physical, part


@pytest.mark.parametrize('resource_id', ['M1', 'M2'])
def test_machine_access_uses_supported_geometry_without_changing_observations(resource_id):
    from cais_spade_llm.resources.environment_recovery import _machine_access_projection, _machine_command_projection

    evidence, snapshot, task, physical, part = _machine_access_case(resource_id)
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
    'missing_link', 'missing_hold'])
def test_machine_access_keeps_constraints_and_missing_evidence_distinct(resource_id, condition):
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
    elif condition == 'missing_link':
        evidence['inputs']['scene']['KMR']['task_execution']['clearance_link_bounds']['missing'] = [{'center': [0., 0., 0.], 'radius': .1}]
    elif condition == 'missing_hold':
        physical['resources'][handling].pop('controller_hold_model')
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
    from cais_spade_llm.recovery_framework.recovery_case_safety import default_safety_files
    from cais_spade_llm.resources.gazebo_programs import validate_resource_programs
    scene = json.loads(SCENE_PATH.read_text())
    validate_resource_programs(scene)
    setup = example_setup(scenario='Machining breakdown during part processing')
    assert {'M1', 'M2', 'ur5e-2'} <= set(setup['permitted_resources'])
    assert setup['failure_scenario']['checkpoint'] == 'during_processing'
    assert setup['selected_safety_files'] == default_safety_files('Machining breakdown during part processing')
    assert example_setup()['failure_scenario']['scenario'] == 'Part slippage'
    assert all(set(scene['resource_programs']['resources'][rid]['primitives']) == set(MACHINE_PRIMITIVES)
               for rid in ('M1', 'M2'))


def test_worker_commands_remain_responsive_while_a_completion_is_observed(machine, tmp_path):
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
    def runner(event_name, task_id, jobs=None):
        worker = WorkflowPrimitiveRunner({'scene':scene, 'geometry':{'RGOCG4-50_Round_4mm':{'model_name':'part'}},
            '_control_path':str(control), '_jobs_path':str(tmp_path/'jobs.json'),
            'task':{'task_id':task_id, 'run_id':'run', 'resource_id':'M2', 'event_name':event_name,
                    'parameters':{'part_name':'RGOCG4-50_Round_4mm', 'process':'trim', 'result':'circle'}}},
            entity=lambda _: state, set_pose=lambda *_:pytest.fail('job control dispatched motion'),
            robot_clearance=lambda *_:{'ur5e-2':1.}, current_clock=lambda:clock[0], wait_simulation=wait,
            machine_jobs=jobs)
        worker._kmr_clearance = lambda:{'link':1.}
        return worker
    # The machine names the job after the start command; its completion is then observed.
    starter = runner('start_job', 'task')
    starter.execute()
    result = runner('complete_job', 'observe', starter.machine_jobs).execute()
    assert sent == ['pause','resume']
    job = result['processing_observations']['jobs']['task']
    assert job['processing_active_sec'] == machine['simulation_process']['processing_time_sec']
    assert job['simulation_elapsed_sec'] > job['processing_active_sec']
    assert job['simulation_paused_sec'] >= 1.


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
    def worker(event_name, task_id, jobs=None, fault_request=None):
        runner = WorkflowPrimitiveRunner({'scene':scene, 'geometry':{'KET4_Square_4mm':{'model_name':'part'}},
            '_jobs_path':paths['M1'], 'task':{'run_id':'run','resource_id':'M1','task_id':task_id,
                'event_name':event_name,'parameters':{'part_name':'KET4_Square_4mm','process':'trim','result':'square'}},
            **({'failure_injection': fault_request} if fault_request else {})},
            entity=lambda _:state, set_pose=lambda *_:pytest.fail('processing dispatched motion'),
            robot_clearance=lambda *_:{'ur5e-1':1.}, current_clock=lambda:clock[0], wait_simulation=wait,
            machine_jobs=jobs)
        runner._kmr_clearance = lambda:{'link':1.}
        return runner
    starter = worker('start_job', 'square-job')
    starter.execute()
    # The breakdown is injected while the job's completion is observed.
    runner = worker('complete_job', 'observe-square', starter.machine_jobs, {
        'run_id':'run','resource_id':'M1','task_id':'observe-square', 'seed':11,
        'checkpoint':'during_processing','control_path':str(fault),'processing_paths':paths,
        'evidence_path':str(tmp_path/'interruption.json')})
    with pytest.raises(RuntimeError, match='Machining breakdown'):
        runner.execute()
    capture = json.loads((tmp_path/'interruption.json').read_text())
    # The evidence names the observing task; the job keeps the task that started it.
    assert capture['task_id'] == 'observe-square' and capture['job_id'] == 'square-job'
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


def test_job_control_is_the_machine_automaton_and_each_part_keeps_its_job_state():
    from cais_spade_llm.resources.environment_models import build_environment_models

    m1 = build_environment_models(json.loads(SCENE_PATH.read_text()))['M1']
    variables = m1['state_variables']
    assert 'active_job' not in variables and variables['job_state']['scope'] == 'part'
    own = {(event['event_name'], event['guards']['resource_state']['equals'], event['updates']['resource_state']['set'])
           for event in m1['events'] if event['parameter_bindings']['resource_id'] == {'equals': 'M1'}
           and 'equals' in event['guards'].get('resource_state', {}) and 'resource_state' in event['updates']}
    assert {('start_job', 'loaded', 'running'), ('pause_job', 'running', 'paused'), ('resume_job', 'paused', 'running'),
            ('stop_job', 'paused', 'stopped'), ('complete_job', 'running', 'completed')} <= own
    # Completion is observed, never commanded, and an interrupted job is the machine's declared fault state.
    assert m1['fault_states'] == ['interrupted']
    assert next(event for event in m1['events'] if event['event_name'] == 'complete_job')['controllable'] is False
    # The job of the held part decides the machine's state; an interrupted job is the observed fault state.
    record = {'active_job': None, 'selected_program': 'square',
              'jobs': {'job': {'part_name': 'KET4_Square_4mm', 'job_state': 'interrupted'}}}
    assert machine_public_state(record, {'part_name': 'KET4_Square_4mm', 'current_state': 'loaded'}) == {
        'selected_program': 'square', 'resource_state': 'interrupted', 'current_state': 'interrupted'}
    assert machine_public_state(record, {'part_name': None}) == {'selected_program': 'square'}

def test_job_controlled_machining_plans_a_commanded_start_and_an_observed_completion():
    from cais_spade_llm.resources.environment_models import build_environment_models, candidates, feasibility

    m1 = build_environment_models(json.loads(SCENE_PATH.read_text()))['M1']
    part, desired = 'KET4_Square_4mm', {'process': 'trim', 'result': 'square'}
    tasks = {task['event_name']: task for task in candidates(m1, {}, part, desired, [desired])}
    # No combined machining event remains; the machine names the job it starts, so start_job takes no job_id.
    assert set(tasks) == {'start_job', 'complete_job'}
    assert tasks['start_job']['parameters'] == {'resource_id': 'M1', 'part_name': part, 'process': 'trim',
                                                'result': 'square'}
    assert tasks['complete_job']['parameters']['result'] == 'square'
    # The command is what the machine's program and tooling must support.
    assert feasibility(m1, tasks['start_job'], {})[0] != 'INFEASIBLE'
    circle = {**tasks['start_job'], 'parameters': {**tasks['start_job']['parameters'], 'result': 'circle'}}
    assert feasibility(m1, circle, {}) == ('INFEASIBLE', ['Current program cannot establish the requested result'])

def machine_capture(machine):
    """Build the recorded null snapshot and running worker state from one run."""
    jobs = MachineJobs(machine, 'run')
    start(jobs)
    # The running job is the machine's own state.
    fields = machine_public_state(jobs.snapshot(), {})
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
        assert result['snapshot']['resource_state'] == 'running'
        assert result['conflicts'] == []
    assert result['physical_snapshot']['active_job'] is None
    assert result['snapshot_sources']['resource_state']['revision'] == evidence['processing_observations']['revision']
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


@pytest.mark.parametrize('value', ['loaded', 'paused'])
def test_capture_keeps_authoritative_machine_disagreement(machine, value):
    from cais_spade_llm.recovery_framework.recovery_observations import reconcile_resource_snapshot

    evidence, physical, inputs = machine_capture(machine)
    evidence['snapshot']['resource_state'] = value
    result = reconcile_resource_snapshot(evidence, physical, resource_id='M2', run_id='run', revision=7, inputs=inputs)
    assert result['conflicts'][0]['recorded'] == value
    assert result['conflicts'][0]['observed'] == 'running'


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
    assert result['resource_evidence']['M2@localhost']['snapshot']['resource_state'] == 'running'
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
