"""Resource registration and owner-modeled motion without robot assumptions."""

from __future__ import annotations

import json
import time
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest
from test_gazebo_safety_preparation import _case as _robot_case
from test_gazebo_safety_preparation import _coverage, _harness, _start
from test_gazebo_safety_preparation import controller as _controller_fixture
from test_primitive_program_safety import _grounded_check
from test_simulation_timing import cartesian_motion_controller  # noqa: F401

from cais_spade_llm.agents.central_controller.local_composition import Budget
from cais_spade_llm.agents.central_controller.offline_recovery_composition import (
    analyze_grounded_recovery_composition,
)
from cais_spade_llm.agents.resource_agent.resource_agent import ResourceAgent
from cais_spade_llm.recovery_framework import fingerprint
from cais_spade_llm.recovery_framework.gazebo_safety_preparation import (
    _runtime_record,
    prepare_and_check,
)
from cais_spade_llm.resources.environment_models import build_environment_models
from cais_spade_llm.resources.resource_safety_preparation import (
    LiveCommandLedger,
    PreparedRobotEvidence,
    PrimitiveModel,
    _validate_controller_contract,
    cartesian_coverage_fingerprint,
    register_resource_provider,
    validate_joint_trajectory,
    validate_prepared_start,
)
from cais_spade_llm.resources.robot.gazebo_pick_place_controller import prepare_ur5e_motion


@pytest.fixture
def controller(cartesian_motion_controller):  # noqa: F811 - imported pytest fixture
    return _controller_fixture.__wrapped__(cartesian_motion_controller)


class _Provider:
    version = 1

    def validate_programs(self, *, resource_id, model, bundle):
        assert resource_id == 'test_resource'
        if bundle != _bundle():
            raise ValueError('Test provider cannot validate these action contracts')

    def create_owner(self, *, resource_id, jid, **kwargs):
        return SimpleNamespace(agent_name=resource_id, jid=jid, preparation_only=True)


def _bundle():
    return {'functions': {'test_motion': {'function_name': 'test_motion', 'status': 'implemented',
        'program': {'entry_state': 'ready', 'success_state': 'ready',
                    'steps': [{'id': 'motion', 'op': 'test_motion', 'params': {}}]}}},
        'primitives': {'test_motion': {'status': 'executable', 'recovery_selectable': False}}}


_PROVIDER = _Provider()
register_resource_provider('test_resource_provider', _PROVIDER)


def _additional(case):
    model = {'resource_id': 'test_resource', 'resource_type': 'test_resource_type',
             'assignments': {}, 'state_variables': {'resource_state': {'domain': ['ready']}},
             'current_valuation': {'resource_state': 'ready'},
             'marked_state_conditions': [{'resource_state': {'equals': 'ready'}}],
             'events': [{'event_id': 910001, 'event_name': 'test_motion',
                         'parameter_bindings': {'resource_id': {'equals': 'test_resource'}},
                         'participants': ['test_resource'], 'guards': {'resource_state': {'equals': 'ready'}},
                         'updates': {'resource_state': {'set': 'ready'}},
                         'product_effects': {'processCompleted': []}, 'controllable': True, 'observable': True}]}
    case['scene']['resource_models'] = {'test_resource': {'model': model, 'owner_provider': 'test_resource_provider'}}
    case['scene']['resource_programs']['resources']['test_resource'] = _bundle()
    case['snapshot']['resources']['test_resource'] = {'current_pose': [9, 0, 1, 0, 0, 0, 1],
        'resource_state': 'ready', 'contained_parts': []}
    case['geometry']['resources']['test_resource'] = {
        'frame': 'world', 'footprint': [[-.01, .01]] * 3, 'stationary_only': True}
    case['stationary']['test_resource'] = [[0, 1]]


def _prepared_case(controller):
    composition, _ = _robot_case(controller)
    case = composition['grounding_inputs']
    program = case['programs'][0]
    prepared = prepare_ur5e_motion(controller, primitive='move_cartesian',
        params=program['primitive_steps'][0]['params'], start=_start(controller), binding={})
    _coverage(program, prepared)
    return case


def test_new_resource_type_is_discovered_and_expands_mutex_without_rule_changes(controller):
    case = _prepared_case(controller)
    definitions = deepcopy(case['catalog'])
    original = build_environment_models(case['scene'])
    _additional(case)
    models = build_environment_models(case['scene'])
    assert list(models) == [*original, 'test_resource']
    assert all(models[rid]['events'] == original[rid]['events'] for rid in original)
    result = _grounded_check(case, trace_complete=True)
    assert result['status'] == 'satisfied', result['reason']
    assert len([row for row in result['bindings'] if row['rule_id'].startswith('SAFE_shared_area_mutex:')]) == 78
    case['snapshot']['resources']['test_resource']['current_pose'] = [1.1, 0, 1, 0, 0, 0, 1]
    rejected = _grounded_check(case, trace_complete=True)
    assert rejected['status'] == 'violated', rejected['reason']
    binding = next(row for row in rejected['bindings'] if row['rule_id'] == rejected['counterexample']['rule_id'])
    assert 'test_resource' in binding['resources']
    assert case['catalog'] == definitions


@pytest.mark.parametrize('change', ['provider', 'geometry', 'duplicate', 'participants', 'event_id', 'program'])
def test_additional_resources_cannot_bypass_registration_or_coverage(controller, change):
    case = _prepared_case(controller)
    _additional(case)
    row = case['scene']['resource_models']['test_resource']
    if change == 'provider':
        row['owner_provider'] = 'unregistered'
    elif change == 'geometry':
        del case['geometry']['resources']['test_resource']
    elif change == 'duplicate':
        case['scene']['resource_models']['M1'] = case['scene']['resource_models'].pop('test_resource')
    elif change == 'participants':
        row['model']['events'][0]['participants'].append('M1')
    elif change == 'event_id':
        row['model']['events'][0]['event_id'] = build_environment_models(_prepared_case(controller)['scene'])['M1']['events'][0]['event_id']
    else:
        case['scene']['resource_programs']['resources']['test_resource']['functions']['test_motion']['program']['steps'][0]['op'] = 'unreviewed'
    result = _grounded_check(case, trace_complete=True)
    assert result['status'] == 'unavailable', result


def test_owner_factory_preserves_existing_identities_and_uses_registered_provider(controller, monkeypatch):
    from cais_spade_llm.recovery_framework import workflow_execution as workflow

    def stub(jid, password, **kwargs):
        return SimpleNamespace(jid=jid, agent_name=kwargs.get('name', 'KMR'))
    for name in ('RobotAgent', 'KMRResourceAgent', 'WorkflowResourceAgent'):
        monkeypatch.setattr(workflow, name, stub)
    monkeypatch.setattr(workflow, 'GazeboWorker', lambda **kwargs: None)
    case = _prepared_case(controller)
    prior = workflow.create_environment_resource_agents(case['scene'], build_environment_models(case['scene']), None)
    _additional(case)
    owners = workflow.create_environment_resource_agents(case['scene'], build_environment_models(case['scene']), None)
    assert [(o.agent_name, o.jid) for o in owners[:-1]] == [(o.agent_name, o.jid) for o in prior]
    assert owners[-1].agent_name == 'test_resource'
    assert owners[-1].preparation_only is True


def test_additional_model_keeps_existing_parameterized_inventory_declarations(controller):
    case = _prepared_case(controller)
    _additional(case)
    original = build_environment_models(case['scene'])
    model = case['scene']['resource_models']['test_resource']['model']
    model['state_variables'].update(deepcopy(original['Storage']['state_variables']))
    inventory = dict.fromkeys(original['Storage']['current_valuation'], False)
    model['current_valuation'].update(inventory)
    result = build_environment_models(case['scene'])['test_resource']
    assert result['state_variables'] == model['state_variables']
    assert result['current_valuation'] == {'resource_state': 'ready', **inventory}
    model['current_valuation'].pop(next(iter(inventory)))
    with pytest.raises(ValueError, match='registered part references'):
        build_environment_models(case['scene'])


def _effects(*, primitive, params, evidence, start_time, end_time, configuration):
    if primitive != configuration['primitive']:
        raise ValueError('Unknown owner action')
    if configuration.get('joint_names'):
        validate_joint_trajectory(evidence['joint_trajectory'], configuration['joint_names'], float(end_time - start_time))
        if set(params) != {'target'} or evidence['trajectory'][-1]['pose'] != params['target']:
            raise ValueError('Native target differs from its motion evidence')
        coverage = evidence['cartesian_coverage']
        if (coverage['joint_trajectory_fingerprint'] != fingerprint(evidence['joint_trajectory'])
                or coverage['trajectory_fingerprint'] != cartesian_coverage_fingerprint(evidence['trajectory'], start_time)
                or coverage['covers_resource_envelope'] is not True):
            raise ValueError('Missing motion coverage')
    elif primitive == 'dwell':
        if set(params) != {'duration_sec'} or params['duration_sec'] != float(end_time - start_time):
            raise ValueError('Dwell duration differs from its evidence')
    elif primitive == 'move_relative':
        if set(params) != {'speed_mps'} or params['speed_mps'] <= 0:
            raise ValueError('Unsupported native movement parameters')
        path = next(iter(evidence['part_trajectories'].values()))
        distance = sum(((b-a)**2 for a,b in zip(path[0]['pose'][:3], path[-1]['pose'][:3], strict=True))) ** .5
        if abs(distance / float(end_time-start_time) - params['speed_mps']) > 1e-9:
            raise ValueError('Part path contradicts native speed')
    return {'trajectory': deepcopy(evidence.get('trajectory')), 'base_trajectory': None,
            'part_trajectories': deepcopy(evidence.get('part_trajectories', {})),
            'transfers': deepcopy(evidence.get('transfers', [])),
            'resource_updates': deepcopy(evidence.get('resource_updates', {}))}


def _seven_joint_case(controller):
    case, _ = _robot_case(controller)
    inputs = case['grounding_inputs']
    for field in ('snapshot', 'geometry'):
        rows = inputs[field]['resources']
        rows['KMR'], rows['ur5e-3'] = rows['ur5e-3'], rows['KMR']
    inputs['snapshot']['resources']['KMR'].update(resource_id='KMR', resource_jid='KMR@localhost')
    inputs['stationary']['KMR'], inputs['stationary']['ur5e-3'] = [], [[0, 1]]
    names = ['test_joint_' + str(i) for i in range(7)]
    program = inputs['programs'][0]
    program['resource_id'] = 'KMR'
    source = program['step_results'][0]['source']
    source.update(resource_id='KMR', resource_jid='KMR@localhost')
    source.update(outline_id='KMR_TEST_EVENT', des_event_id='KMR_TEST_DES', event_name='KMR_TEST_RECOVERY')
    trace = program['step_results'][0]
    params = {'target': trace['model_evidence']['trajectory'][-1]['pose']}
    program['primitive_steps'][0].update(params=deepcopy(params), source=deepcopy(source))
    trace.update(resolved_params=deepcopy(params))
    joint = {'joint_names': names, 'duration_ns': 1_000_000_000, 'points': [
        {'positions': [float(i)] * 7, 'time_from_start': {'sec': i, 'nanosec': 0}} for i in (0, 1)]}
    evidence = trace['model_evidence']
    evidence.update(primitive_contract='test_motion', joint_trajectory=joint,
                    cartesian_coverage={'joint_trajectory_fingerprint': fingerprint(joint),
                        'trajectory_fingerprint': cartesian_coverage_fingerprint(evidence['trajectory'], 0),
                        'covers_resource_envelope': True})
    model = PrimitiveModel('test_motion', 1, {'primitive': 'move_cartesian', 'joint_names': names}, _effects)
    event = case['recovery_events'][0]
    event.update({key: source[key] for key in ('outline_id','des_event_id','event_name','resource_id')})
    choice = case['event_start_choices'][0]
    choice.update(programs=deepcopy(inputs['programs']), stationary=deepcopy(inputs['stationary']), starts={source['outline_id']: 0})
    case['completion']['resources'] = {'KMR': {'current_pose': params['target'], 'held_part': 'KET4_Square_4mm'}}
    return case, {'KMR': model}


def test_non_ur5e_native_contract_matches_physical_aps_and_composition(controller):
    reference = _grounded_check(_prepared_case(controller), trace_complete=True)
    case, models = _seven_joint_case(controller)
    result = _grounded_check(case['grounding_inputs'], primitive_models=models, trace_complete=True)
    assert result['status'] == reference['status'] == 'satisfied', result['reason']
    def precedence(result):
        return [r['ap_values'] for r in result['rule_checks'] if 'before_KET4' in r['rule_id']]
    assert precedence(result) == precedence(reference)
    composition = analyze_grounded_recovery_composition(**case, primitive_models=models, budget=Budget(seconds=30))
    assert composition['status'] == 'allowed', composition['reason']
    assert composition['completion_witness']['completed_event_ids'] == ['KMR_TEST_EVENT']
    json.dumps(composition)


def test_changed_owner_contract_invalidates_trace_and_composition_continuations(controller):
    case, models = _seven_joint_case(controller)
    prefix = _grounded_check(case['grounding_inputs'], primitive_models=models, observation_slice=[0, 1])
    assert prefix['status'] == 'prefix_checked', prefix['reason']
    changed = {'KMR': replace(models['KMR'], version=2)}
    result = _grounded_check(case['grounding_inputs'], primitive_models=changed, continuation=prefix['continuation'])
    assert result['status'] == 'unavailable'
    first = analyze_grounded_recovery_composition(**case, primitive_models=models, budget=Budget(seconds=30))
    prior = first['decision_prefixes'][0]['accepted_prefix']
    result = analyze_grounded_recovery_composition(**case, primitive_models=changed, accepted_prefix=prior, budget=Budget(seconds=30))
    assert result['status'] == 'inconclusive'


@pytest.mark.parametrize('change', ['missing', 'contract', 'joints', 'coverage', 'time_gap'])
def test_unknown_or_incomplete_owner_models_fail_closed(controller, change):
    case, models = _seven_joint_case(controller)
    evidence = case['grounding_inputs']['programs'][0]['step_results'][0]['model_evidence']
    if change == 'missing':
        models = {}
    elif change == 'contract':
        evidence['primitive_contract'] = 'unknown'
    elif change == 'joints':
        evidence['joint_trajectory']['joint_names'].reverse()
    elif change == 'time_gap':
        evidence['joint_trajectory']['points'][0]['time_from_start']['nanosec'] = 1
    else:
        evidence['cartesian_coverage']['covers_resource_envelope'] = False
    result = _grounded_check(case['grounding_inputs'], primitive_models=models, trace_complete=True)
    assert result['status'] == 'unavailable', result


def _equipment_case(controller, resource, primitive):
    case = _prepared_case(controller)
    case['programs'] = []
    case['stationary'] = {rid: [[0, 1]] for rid in case['snapshot']['resources'] if rid in case['geometry']['resources']}
    part = 'KET4_Square_4mm'
    robot = case['snapshot']['resources']['ur5e-3']
    robot.update(held_part=None, grasp_transform=None, gripper_state='open', current_pose=[8, 0, 1, 0, 0, 0, 1])
    state = case['snapshot']['resources'][resource]
    state.update(contained_parts=[part])
    case['snapshot']['parts'][part].update(contained_by=resource, stationary_until=0)
    start = deepcopy(case['snapshot']['parts'][part]['current_pose'])
    end = [*start]
    end[0] += .5 if resource == 'Buffer For Machined parts' else 2.
    if primitive == 'dwell':
        end = start
        case['snapshot']['parts'][part]['stationary_until'] = 1
    duration = 5 if primitive == 'dwell' else 1
    if duration != 1:
        case['horizon'] = [0, duration]
        case['stationary'] = {rid: [[0, duration]] for rid in case['stationary']}
        for row in case['snapshot']['parts'].values():
            row['stationary_until'] = duration
    params = {'duration_sec': duration} if primitive == 'dwell' else {'speed_mps': end[0]-start[0]}
    evidence = {'frame': 'world', 'primitive_contract': resource,
                'part_trajectories': {} if primitive == 'dwell' else {part: [{'time': 0, 'pose': start}, {'time': 1, 'pose': end}]}}
    step = {'step_index': 0, 'primitive': primitive, 'resolved_params': deepcopy(params), 'start_time': 0, 'end_time': duration,
            'model_evidence': evidence, 'source': {'outline_id': 'equipment_case', 'step_index': 0}}
    case['programs'] = [{'resource_id': resource, 'primitive_steps': [{'primitive': primitive, 'params': params}], 'step_results': [step]}]
    case['stationary'][resource] = []
    return case, {resource: PrimitiveModel(resource, 1, {'primitive': primitive}, _effects)}


def _machine_completion(controller):
    case, models = _equipment_case(controller, 'M1', 'dwell')
    part = 'KET4_Square_4mm'
    model = build_environment_models(case['scene'])['M1']
    event = next(row for row in model['events'] if row['event_name'] == 'machine_part')
    params = {key: value['equals'] for key, value in event['parameter_bindings'].items() if 'equals' in value}
    params.update(part_name=part, result='square')
    declaration = {'event_id': event['event_id'], 'event_name': event['event_name'], 'parameters': params}
    case['snapshot']['parts'][part]['processCompleted'] = []
    case['snapshot']['resources']['M1'].update(resource_state='loaded', part_name=part)
    case['programs'][0]['step_results'][0]['model_evidence']['resource_updates'] = {'resource_state': 'completed'}
    models['M1'] = replace(models['M1'], configuration={'primitive': 'dwell',
        'state_variables': model['state_variables'],
        'requirements': {part: [{'processesToComplete': [{'process': 'trim', 'result': 'square'}]}]}})
    case['task_evidence'] = {'complete': True, 'source_kind': 'synthetic', 'horizon': [0, 5],
        'events': [{'task_id': 'test_machining', 'resource_id': 'M1', 'function': 'machine_part',
                    'process': 'trim', 'product': part, 'context': 'any', 'start_time': 0, 'end_time': 5,
                    'declared_task': declaration}]}
    case['product_effect_evidence'] = {'complete': True, 'source_kind': 'synthetic', 'horizon': [0, 5],
        'updates': [{'task_id': 'test_machining', 'resource_id': 'M1', 'time': 5, 'kind': 'acknowledged',
                    'declaration': deepcopy(declaration),
                    'product_effects': {part: {'processCompleted': [{'process': 'trim', 'result': 'square'}]}}}]}
    return case, models


def test_declared_machine_completion_uses_native_guards_and_preserves_actual_history(controller):
    case, models = _machine_completion(controller)
    before = deepcopy(case)
    result = _grounded_check(case, primitive_models=models, trace_complete=True)
    assert result['status'] == 'satisfied', result['reason']
    final = result['projected_snapshot']
    assert final['resources']['M1']['resource_state'] == 'completed'
    assert final['parts']['KET4_Square_4mm']['processCompleted'] == [{'process': 'trim', 'result': 'square'}]
    assert final['parts']['KET4_Square_4mm']['contained_by'] == 'M1'
    assert case == before


def test_machine_process_completion_has_one_native_and_physical_composition_witness(controller):
    from cais_spade_llm.agents.central_controller.base_safety_checker import BaseSafetyChecker
    from cais_spade_llm.agents.central_controller.reviewed_primitive_program_safety import _COMPLETED, _MEANINGS
    from cais_spade_llm.agents.central_controller.ppr_ap import ap_record, parse_ap_definition

    case, models = _machine_completion(controller)
    part = 'KET4_Square_4mm'
    record = {'process': 'trim', 'result': 'square'}
    full = _COMPLETED
    case['catalog']['specifications'].append({'id': 'test_pending_trim', 'requirement': 'Complete trim with result square.',
        'formula': 'F ap001', 'aps': [ap_record('ap001', parse_ap_definition(full), _MEANINGS[full])]})
    case['requirement_scopes'].append({'specification': 'test_pending_trim',
        'physical_ap_bindings': {'ap001': {'part': part, **record}}})
    source = {'outline_id': 'test_machining_recovery', 'des_event_id': 'test_machining_des',
              'event_name': 'machine_part', 'step_index': 0}
    program = case['programs'][0]
    program['primitive_steps'][0]['source'] = deepcopy(source)
    program['step_results'][0]['source'] = deepcopy(source)
    event = {key: source[key] for key in ('outline_id', 'des_event_id', 'event_name')}
    event.update(resource_id='M1', primitive_step_indices=[0], predecessors=[])
    declaration = case['task_evidence']['events'][0]['declared_task']
    case['product_effect_evidence']['updates'][0]['kind'] = 'predicted'
    native = {'monitor': BaseSafetyChecker({}, []), 'current_states': {},
              'resources': deepcopy(case['snapshot']['resources']), 'products': deepcopy(case['snapshot']['parts']),
              'contexts': {}, 'jids': {rid: rid + '@localhost' for rid in case['snapshot']['resources']},
              'tasks': {event['outline_id']: {'task': {'task_id': 'test_machining', 'resource_id': 'M1', **declaration},
                  'participants': ['M1'], 'resource_updates': {'M1': {'resource_state': 'completed'}},
                  'product_updates': {part: {'processCompleted': [record]}}}}}
    before = deepcopy(case)
    result = analyze_grounded_recovery_composition(grounding_inputs=case, recovery_events=[event], running_work=[],
        event_start_choices=[{'id': 'start', 'starts': {event['outline_id']: 0},
                              'programs': deepcopy(case['programs']), 'stationary': deepcopy(case['stationary'])}],
        completion={'resources': {'M1': {'resource_state': 'completed'}}, 'parts': {part: {'processCompleted': [record]}}},
        primitive_models=models, task_monitor_context=native, budget=Budget(seconds=30))
    assert result['status'] == 'allowed', result['reason']
    assert result['completion_witness']['completed_event_ids'] == [event['outline_id']]
    assert native['products'][part]['processCompleted'] == []
    assert case == before


@pytest.mark.parametrize('change', ['guard', 'effect', 'declaration', 'program', 'requirements', 'predicted', 'task', 'part', 'process'])
def test_owner_process_and_state_effects_cannot_bypass_task_declarations(controller, change):
    case, models = _machine_completion(controller)
    step = case['programs'][0]['step_results'][0]
    if change == 'guard':
        case['snapshot']['resources']['M1']['resource_state'] = 'idle'
    elif change == 'effect':
        step['model_evidence']['resource_updates']['resource_state'] = 'idle'
    elif change == 'declaration':
        case['product_effect_evidence']['updates'][0]['declaration']['event_id'] = 9999
    elif change == 'program':
        step['resolved_params']['duration_sec'] = 4
        case['programs'][0]['primitive_steps'][0]['params']['duration_sec'] = 4
    elif change == 'requirements':
        config = deepcopy(models['M1'].configuration)
        config['requirements']['KET4_Square_4mm'][0]['processesToComplete'][0]['result'] = 'round'
        models['M1'] = replace(models['M1'], configuration=config)
    elif change == 'predicted':
        case['product_effect_evidence']['updates'][0]['kind'] = 'predicted'
    elif change == 'task':
        case.pop('task_evidence')
    else:
        case['task_evidence']['events'][0]['product' if change == 'part' else 'process'] = 'gear_small' if change == 'part' else 'assembly'
    result = _grounded_check(case, primitive_models=models, trace_complete=True)
    assert result['status'] == 'unavailable', result['reason']


def test_owner_state_update_without_acknowledged_task_is_unavailable(controller):
    case, models = _machine_completion(controller)
    case.pop('product_effect_evidence')
    case.pop('task_evidence')
    assert _grounded_check(case, primitive_models=models, trace_complete=True)['status'] == 'unavailable'


@pytest.mark.parametrize('completed', [False, True])
def test_native_state_evidence_and_physical_observations_share_completion_boundary(controller, completed):
    case, models = _machine_completion(controller)
    identity = {'resource_id': 'M1', 'process': 'trim', 'product': 'KET4_Square_4mm', 'context': 'any'}
    case['state_evidence'] = {'complete': True, 'source_kind': 'synthetic', 'horizon': [0, 5],
        'initial': [{**identity, 'values': {'resource_state': 'loaded'}}],
        'updates': [{**identity, 'time': 5, 'task_id': 'test_machining',
                     'values': {'resource_state': 'completed' if completed else 'loaded'}}]}
    result = _grounded_check(case, primitive_models=models, trace_complete=True)
    assert result['status'] == ('satisfied' if completed else 'unavailable'), result['reason']


def test_owner_model_cannot_return_a_verdict_or_invent_process_completion(controller):
    case, models = _machine_completion(controller)
    original = models['M1']
    def verdict(**kwargs):
        return {**_effects(**kwargs), 'is_safe': True}
    models['M1'] = replace(original, evaluate=verdict)
    assert _grounded_check(case, primitive_models=models, trace_complete=True)['status'] == 'unavailable'
    models['M1'] = original
    case['programs'][0]['step_results'][0]['model_evidence']['resource_updates']['processCompleted'] = [{'process': 'trim', 'result': 'square'}]
    assert _grounded_check(case, primitive_models=models, trace_complete=True)['status'] == 'unavailable'


def test_buffer_declared_zone_update_and_part_motion_need_no_gripper(controller):
    case, models = _equipment_case(controller, 'Buffer For Machined parts', 'move_relative')
    rid, part = 'Buffer For Machined parts', 'KET4_Square_4mm'
    model = build_environment_models(case['scene'])[rid]
    event = next(row for row in model['events'] if row['event_name'] == 'advance_part')
    params = {key: row['equals'] for key, row in event['parameter_bindings'].items() if 'equals' in row}
    params['part_name'] = part
    declaration = {'event_id': event['event_id'], 'event_name': event['event_name'], 'parameters': params}
    case['snapshot']['resources'][rid].update(zone_1_part=part, zone_2_part=None)
    case['programs'][0]['step_results'][0]['model_evidence']['resource_updates'] = {'zone_1_part': None, 'zone_2_part': part}
    models[rid] = replace(models[rid], configuration={**models[rid].configuration, 'state_variables': model['state_variables']})
    case['task_evidence'] = {'complete': True, 'source_kind': 'synthetic', 'horizon': [0, 1],
        'events': [{'task_id': 'test_buffer_transfer', 'resource_id': rid, 'function': 'advance_part',
                    'process': 'transport', 'product': part, 'context': 'any', 'start_time': 0, 'end_time': 1,
                    'declared_task': declaration}]}
    result = _grounded_check(case, primitive_models=models, trace_complete=True)
    assert result['status'] == 'satisfied', result['reason']
    final = result['projected_snapshot']['resources'][rid]
    assert final['zone_1_part'] is None and final['zone_2_part'] == part
    assert final['contained_parts'] == [part] and 'gripper_state' not in final


def test_standalone_reviewed_checker_uses_exact_owner_contract(controller):
    from test_primitive_program_safety import _reviewed_case

    from cais_spade_llm.agents.central_controller.reviewed_primitive_program_safety import (
        validate_reviewed_primitive_program_safety,
    )

    case, models = _seven_joint_case(controller)
    inputs = case['grounding_inputs']
    population = deepcopy(_reviewed_case()['applicability']['scene_resources'])
    scopes = deepcopy(inputs['requirement_scopes'])
    scopes[0]['bindings'] = 'all_pairs'
    applicability = {'version': 1,
        'scene_resources': population,
        'participant_resource_types': sorted({row['resource_type'] for row in population}),
        'regions': {'assembly_board-v1': {'resources': sorted(row['resource_id'] for row in population), 'excluded_resources': []}},
        'rules': scopes}
    kwargs = {key: deepcopy(inputs[key]) for key in ('programs','snapshot','geometry','horizon','stationary','catalog')}
    result = validate_reviewed_primitive_program_safety(**kwargs, applicability=applicability,
                                                      primitive_models=models, trace_complete=True)
    assert result['status'] == 'satisfied', result['reason']


def _transfer_effects(**kwargs):
    evidence = kwargs['evidence']
    if kwargs['primitive'] != 'move_relative' or kwargs['params'] != kwargs['configuration']['params']:
        raise ValueError('Unsupported native transfer action')
    return {key: deepcopy(evidence.get(key, default)) for key, default in {
        'trajectory': None, 'base_trajectory': None, 'part_trajectories': {},
        'transfers': [], 'resource_updates': {}}.items()}


@pytest.mark.parametrize('change', [None, 'missing_peer', 'different_time', 'duplicate', 'custody'])
def test_containment_transfer_requires_both_exact_participants_without_gripper_custody(controller, change):
    case, models = _equipment_case(controller, 'Conveyor', 'move_relative')
    source = case['programs'][0]
    target = deepcopy(source)
    target['resource_id'] = 'Buffer For Machined parts'
    target['primitive_steps'][0]['params'] = {'speed_mps': .5}
    target['step_results'][0]['resolved_params'] = {'speed_mps': .5}
    evidence = target['step_results'][0]['model_evidence']
    evidence.update(primitive_contract='Buffer For Machined parts', part_trajectories={})
    transfer = {'part_name': 'KET4_Square_4mm', 'from_resource': 'Conveyor', 'to_resource': 'Buffer For Machined parts'}
    for program in (source, target):
        program['step_results'][0]['model_evidence']['transfers'] = [deepcopy(transfer)]
    models[target['resource_id']] = PrimitiveModel('Buffer For Machined parts', 1, {'params': {'speed_mps': .5}}, _transfer_effects)
    case['snapshot']['resources'][target['resource_id']]['contained_parts'] = []
    case['stationary'][target['resource_id']] = []
    case['programs'].append(target)
    if change == 'missing_peer':
        evidence['transfers'] = []
    elif change == 'different_time':
        target['step_results'][0]['end_time'] = .5
        case['stationary'][target['resource_id']] = [[.5, 1]]
    elif change == 'duplicate':
        evidence['transfers'].append(deepcopy(transfer))
    elif change == 'custody':
        case['snapshot']['resources'][target['resource_id']]['contained_parts'].append('KET4_Square_4mm')
    result = _grounded_check(case, primitive_models=models, trace_complete=True)
    if change:
        assert result['status'] == 'unavailable', result['reason']
    else:
        assert result['status'] == 'satisfied', result['reason']
        snapshot = result['projected_snapshot']
        assert snapshot['parts']['KET4_Square_4mm']['contained_by'] == 'Buffer For Machined parts'
        assert snapshot['resources']['Conveyor']['contained_parts'] == []
        assert snapshot['resources']['Buffer For Machined parts']['contained_parts'] == ['KET4_Square_4mm']
        assert all(not parts for row in result['observations'] for parts in row['carried_parts'].values())


@pytest.mark.parametrize('resource,primitive', [('M1','dwell'), ('Conveyor','move_relative'), ('Buffer For Machined parts','move_relative')])
def test_equipment_uses_native_contracts_without_gripper_fields(controller, resource, primitive):
    case, models = _equipment_case(controller, resource, primitive)
    before = deepcopy(case)
    result = _grounded_check(case, primitive_models=models, trace_complete=True)
    assert result['status'] == 'satisfied', result['reason']
    assert not {'held_part','gripper_state','grasp_transform'} & result['projected_snapshot']['resources'][resource].keys()
    assert result['projected_snapshot']['parts']['KET4_Square_4mm']['processCompleted'] == before['snapshot']['parts']['KET4_Square_4mm']['processCompleted']
    if primitive == 'move_relative':
        assert any(r['part_region_occupancy']['assembly_board-v1']['KET4_Square_4mm'] for r in result['observations'])
        assert all(not r['region_occupancy']['assembly_board-v1'][resource] for r in result['observations'])
    assert case == before


def test_transport_conflicting_stationary_or_missing_path_is_unavailable(controller):
    case, models = _equipment_case(controller, 'Conveyor', 'move_relative')
    case['snapshot']['parts']['KET4_Square_4mm']['stationary_until'] = 1
    result = _grounded_check(case, primitive_models=models, trace_complete=True)
    assert result['status'] == 'unavailable'
    case['snapshot']['parts']['KET4_Square_4mm']['stationary_until'] = 0
    case['programs'][0]['step_results'][0]['model_evidence']['part_trajectories']['KET4_Square_4mm'].pop()
    assert _grounded_check(case, primitive_models=models, trace_complete=True)['status'] == 'unavailable'


def test_resource_defaults_cannot_manufacture_preparation():
    owner = SimpleNamespace(agent_name='test_resource', jid='test@localhost')
    assert ResourceAgent.capture_recovery_safety_state(owner)['status'] == 'NEEDS_CONTEXT'
    result = ResourceAgent.prepare_recovery_safety_program(owner, {'resource_id':'test_resource'}, {'checkpoint_id':'test'})
    assert result['status'] == 'NEEDS_CONTEXT'
    with pytest.raises(ValueError, match='unavailable'):
        ResourceAgent.validate_recovery_safety_step(owner, {}, {})


@pytest.mark.parametrize('change', [None, 'preparation_id', 'checkpoint', 'contract'])
def test_non_ur5e_owner_prepares_and_checks_without_a_robot_controller(controller, tmp_path, change):
    bridge, runtime, _ = _harness(controller)
    case, models = _seven_joint_case(controller)
    inputs = case['grounding_inputs']
    inputs['scene']['safety_geometry'] = deepcopy(inputs['geometry'])
    parts = inputs['snapshot']['parts']
    runtime.context.inputs['scene'] = inputs['scene']
    runtime.context.part_tracker = deepcopy(parts)
    runtime.context.snapshot = lambda: deepcopy(inputs['snapshot']['resources'])
    owners = []
    for rid, state in inputs['snapshot']['resources'].items():
        owner = SimpleNamespace(agent_name=rid, jid=rid + '@localhost', execution_mode='dry_run', environment_runtime=runtime)
        owner.get_recovery_physical_snapshot = lambda state=state: {'snapshot': deepcopy(state)}
        owner.capture_recovery_safety_state = lambda max_age=2., state=state: {
            'observation_state': deepcopy(state), 'custody_complete': True,
            'launch_id': 'synthetic-launch', 'observed_monotonic': time.monotonic()}
        owner.recovery_safety_configuration = lambda rid=rid: {'contract': None if rid not in models else models[rid].descriptor()}
        owner.get_recovery_safety_primitive_model = lambda rid=rid: models.get(rid)
        def reader(name, *, max_age):
            return {'name': name, 'frame': 'world', 'pose': deepcopy(parts[name]['current_pose']),
                    'observed_monotonic': time.monotonic(), 'simulation_time': 0., 'simulation_stamp': 0.}
        owner.get_recovery_entity_reader = lambda: reader
        owners.append(owner)
    bridge.resource_agents = runtime.resource_agents = owners
    owner = next(row for row in owners if row.agent_name == 'KMR')
    program = inputs['programs'][0]
    request = {'recovery_id': 'test_resource_preparation', 'programs': [
        {'resource_id': 'KMR', 'primitive_steps': deepcopy(program['primitive_steps'])}]}
    def prepare(program, checkpoint):
        record = {'status': 'prepared', 'primitive': program['primitive_steps'][0]['primitive'],
                  'params': deepcopy(program['primitive_steps'][0]['params']),
                  'binding': {'checkpoint_id': checkpoint['checkpoint_id'],
                              'program_fingerprint': fingerprint(program), 'source': program['primitive_steps'][0]['source']}}
        record['preparation_id'] = fingerprint(record)
        return {'status': 'prepared', 'resource_id': 'KMR', 'checkpoint_id': checkpoint['checkpoint_id'],
                'program': deepcopy(program), 'program_fingerprint': fingerprint(program), 'steps': [record]}
    owner.prepare_recovery_safety_program = prepare
    def validate(planned, step):
        if planned['primitive'] != step['primitive'] or planned['params'] != step['resolved_params']:
            raise ValueError('Changed native test parameters')
    owner.validate_recovery_safety_step = validate
    def evidence(*, checkpoint, prepared, request):
        value = deepcopy(case)
        row = value['event_start_choices'][0]['programs'][0]['step_results'][0]['model_evidence']
        row['preparation_id'] = 'forged' if change == 'preparation_id' else prepared[0]['steps'][0]['preparation_id']
        if change == 'contract':
            row['primitive_contract'] = 'forged'
        return {'synthetic': True, 'checkpoint_id': 'forged' if change == 'checkpoint' else checkpoint['checkpoint_id'],
                'prepared_fingerprint': fingerprint(prepared), 'composition_inputs': value,
                'physical_rule_activation': 'prospective'}
    bridge.cca.recovery_safety_preparation_provider = evidence
    before = _runtime_record(runtime, bridge.cca)
    result = prepare_and_check(bridge, request, output_root=tmp_path, max_age=30, budget=Budget(seconds=30))
    if change:
        assert result['status'] in {'NEEDS_CONTEXT', 'inconclusive'}, result
    else:
        assert result['status'] == 'allowed', result.get('unresolved') or result.get('analysis')
        assert result['primitive_models']['KMR']['configuration']['joint_names'] == ['test_joint_' + str(i) for i in range(7)]
    assert result['dispatch_authorized'] is False
    assert all(not hasattr(row, '_controller') for row in owners)
    assert _runtime_record(runtime, bridge.cca) == before
    assert runtime.context.pending_tasks == {'M1_delivery': {'status': 'pending'}}
    assert not bridge.cca.recovery_composition_admissions
    controller._send_simulation_joint_trajectory.assert_not_called()


def test_registered_new_action_reaches_composition_without_shared_action_dispatch(controller):
    case, _ = _equipment_case(controller, 'M1', 'dwell')
    # Keep the established scene and requirements; only the additional resource
    # declares the new action and supplies its physical effects.
    _additional(case)
    case['horizon'] = [0, 1]
    case['stationary'] = {rid: [[0, 1]] for rid in case['snapshot']['resources']}
    for row in case['snapshot']['parts'].values():
        row['stationary_until'] = 1
    case['snapshot']['resources']['test_resource']['current_pose'][0] = .8
    case['geometry']['resources']['test_resource'].pop('stationary_only')
    source = {'outline_id': 'test_resource_recovery', 'des_event_id': 'test_resource_des',
              'event_name': 'test_resource_event', 'step_index': 0}
    first = case['snapshot']['resources']['test_resource']['current_pose']
    last = [1.5, *first[1:]]
    case['programs'] = [{'resource_id': 'test_resource',
        'primitive_steps': [{'primitive': 'test_motion', 'params': {}, 'source': source}],
        'step_results': [{'step_index': 0, 'primitive': 'test_motion', 'resolved_params': {}, 'source': source,
            'start_time': 0, 'end_time': 1, 'model_evidence': {'frame': 'world', 'primitive_contract': 'test_resource',
                'trajectory': [{'time': 0, 'pose': first}, {'time': 1, 'pose': last}]}}]}]
    case['stationary']['test_resource'] = []
    models = {'test_resource': PrimitiveModel('test_resource', 1, {'primitive': 'test_motion'}, _effects)}
    event = {key: source[key] for key in ('outline_id', 'des_event_id', 'event_name')}
    event.update(resource_id='test_resource', primitive_step_indices=[0], predecessors=[])
    composition = {'grounding_inputs': case, 'recovery_events': [event], 'running_work': [],
        'event_start_choices': [{'id': 'start', 'starts': {'test_resource_recovery': 0},
                                'programs': deepcopy(case['programs']), 'stationary': deepcopy(case['stationary'])}],
        'completion': {'resources': {'test_resource': {'current_pose': last}}, 'parts': {}}}
    before = deepcopy(composition)
    result = analyze_grounded_recovery_composition(**composition, primitive_models=models, budget=Budget(seconds=30))
    assert result['status'] == 'allowed', result['reason']
    assert len(result['scope']['included_specifications']) == 79
    case['snapshot']['resources']['ur5e-4']['current_pose'] = [1.1, 0, 1, 0, 0, 0, 1]
    result = analyze_grounded_recovery_composition(**composition, primitive_models=models, budget=Budget(seconds=30))
    assert result['status'] == 'held', result['reason']
    assert case['catalog'] == before['grounding_inputs']['catalog']


def _native_preparation_case():
    owner = SimpleNamespace(agent_name='ur5e-3', _controller=SimpleNamespace(execution_mode='simulation'))
    provider = PreparedRobotEvidence(owner, LiveCommandLedger())
    planned = {'primitive': 'move_cartesian', 'observation_status': 'prepared',
               'joint_trajectory': {'joint_names': ['slide']},
               'start': {'joint_positions': [0.]}}
    planned['preparation_id'] = fingerprint(planned)
    prepared = {'status': 'prepared', 'resource_id': owner.agent_name, 'steps': [planned]}
    provider.register_program(prepared)
    checkpoint = {'checkpoint_id': 'checkpoint', 'unresolved': [], 'observations': {
        owner.agent_name: {'physical': {'idle': True, 'custody_complete': True,
            'attachment': {'model_name': None}, 'observation_state': {'held_part': None}}}}}
    native = {'operation': 'prepare', 'accepted': True, 'reason': '',
        'instance_id': 'instance', 'expected_instance_id': 'instance',
        'command_revision': 3, 'expected_command_revision': 3,
        'contract_revision': 0, 'expected_contract_revision': 0, 'rejected_commands': 0,
        'binding_fingerprint': 'binding', 'requested_binding_fingerprint': 'binding',
        'joint_names': ['slide'], 'observed_positions': [0.], 'observed_velocities': [0.],
        'stationary': False, 'reservation_token': '', 'physical_execution_verified': False,
        'physical_execution_reason': 'gazebo_physics_containment_and_stopping_unverified',
        'contract_state': 'unlocked', 'observed_stationary': True, 'stationary_samples': 3,
        'simulation_time': 1., 'checkpoint_simulation_time': 1., 'elapsed_wall_s': .1,
        'maximum_observed_update_period': .001, 'maximum_observed_position_error': [0.]}
    return provider, prepared, checkpoint, native


@pytest.mark.parametrize(('fields', 'reason'), [
    ({'instance_id': '', 'expected_instance_id': ''}, 'identity is unavailable'),
    ({'binding_fingerprint': None, 'requested_binding_fingerprint': None}, 'identity is unavailable'),
    ({'command_revision': True, 'expected_command_revision': True}, 'revision or count'),
    ({'contract_revision': -1, 'expected_contract_revision': -1}, 'revision or count'),
    ({'rejected_commands': False}, 'revision or count'),
    ({'reason': 'Native owner snapshot changed'}, 'conflicting reason'),
    ({'reservation_token': None}, 'unexpectedly acquired execution authority'),
    ({'physical_execution_reason': ''}, 'unavailability reason'),
    ({'maximum_observed_update_period': float('nan')}, 'sample diagnostics'),
    ({'maximum_observed_update_period': -1.}, 'sample diagnostics'),
    ({'maximum_observed_update_period': True}, 'sample diagnostics'),
    ({'maximum_observed_position_error': []}, 'sample diagnostics'),
    ({'maximum_observed_position_error': [float('inf')]}, 'sample diagnostics'),
    ({'maximum_observed_position_error': [-.001]}, 'sample diagnostics'),
])
def test_native_preparation_rejects_self_consistent_invalid_provenance(fields, reason):
    _, prepared, _, native = _native_preparation_case()
    native.update(fields)
    with pytest.raises(ValueError, match=reason):
        _validate_controller_contract(native, prepared['steps'][0])


@pytest.mark.parametrize('change', ['command', 'reservation', 'preparation', 'checkpoint'])
def test_native_preparation_query_cannot_outlive_local_evidence(monkeypatch, change):
    from cais_spade_llm.resources.robot import gazebo_pick_place_controller

    provider, prepared, checkpoint, native = _native_preparation_case()
    if change == 'reservation':
        identifier = provider.ledger.prepare(resource_id='ur5e-3', task_id='pending', command={},
                                             owner_identity={'preparation_id': 'pending'})
    before_revision = provider.ledger.revision

    def query(*args):
        if change == 'command':
            provider.ledger.prepare(resource_id='ur5e-3', task_id='other', command={},
                                    owner_identity={'preparation_id': 'other'})
        elif change == 'reservation':
            provider.ledger.authorize(identifier, reservation_token='concurrent',
                                      expected_revision=provider.ledger.revision)
        elif change == 'preparation':
            provider.steps[prepared['steps'][0]['preparation_id']]['start']['joint_positions'] = [1.]
        else:
            checkpoint['checkpoint_id'] = 'another checkpoint'
        return deepcopy(native)

    monkeypatch.setattr(gazebo_pick_place_controller, '_prepare_controller_contract', query)
    result = provider.prepare_execution_coverage(prepared=prepared, checkpoint=checkpoint)
    assert result['status'] == 'NEEDS_CONTEXT' and not result['native_preparation_verified']
    assert 'changed during native preparation' in result['reason']
    assert result['native_contract'] == native and result['command_ledger_revision'] == before_revision
    assert not result['physical_execution_verified'] and not result['command_sent']


@pytest.mark.parametrize('change', ['unresolved', 'observations', 'owner', 'attachment', 'observation_state'])
def test_native_preparation_requires_explicit_checkpoint_and_empty_custody(monkeypatch, change):
    from unittest.mock import Mock

    from cais_spade_llm.resources.robot import gazebo_pick_place_controller

    provider, prepared, checkpoint, _ = _native_preparation_case()
    if change in {'unresolved', 'observations'}:
        checkpoint.pop(change)
    elif change == 'owner':
        checkpoint['observations'].clear()
    else:
        checkpoint['observations']['ur5e-3']['physical'].pop(change)
    query = Mock()
    monkeypatch.setattr(gazebo_pick_place_controller, '_prepare_controller_contract', query)
    result = provider.prepare_execution_coverage(prepared=prepared, checkpoint=checkpoint)
    assert result['status'] == 'NEEDS_CONTEXT' and not result['native_preparation_verified']
    assert not result['dispatch_authorized'] and not result['command_sent']
    query.assert_not_called()


@pytest.mark.parametrize('observed_error', [0., 100.])
def test_native_preparation_samples_never_establish_physical_bounds(monkeypatch, observed_error):
    from cais_spade_llm.resources.robot import gazebo_pick_place_controller

    provider, prepared, checkpoint, native = _native_preparation_case()
    native['maximum_observed_position_error'] = [observed_error]
    monkeypatch.setattr(gazebo_pick_place_controller, '_prepare_controller_contract',
                        lambda *args: deepcopy(native))
    before = deepcopy(prepared), deepcopy(checkpoint), provider.ledger.snapshot()
    result = provider.prepare_execution_coverage(prepared=prepared, checkpoint=checkpoint)
    assert result['native_preparation_verified'] is True
    assert result['physical_execution_reason'] == native['physical_execution_reason']
    assert result['missing_physical_evidence'] == [
        'verified tracking bounds', 'future stationary containment bounds', 'failure stopping bounds']
    assert result['prepared_fingerprint'] == fingerprint(prepared)
    assert result['checkpoint_fingerprint'] == fingerprint(checkpoint)
    assert result['status'] == 'NEEDS_CONTEXT' and not result['physical_execution_verified']
    assert not result['dispatch_authorized'] and not result['command_sent']
    assert before == (prepared, checkpoint, provider.ledger.snapshot())


def _modeled_execution_case():
    from test_continuous_motion import configuration, trajectory

    provider, prepared, checkpoint, native = _native_preparation_case()
    model = configuration()
    model['joint_position_error'] = {'slide': .001}
    controller = provider.owner._controller
    controller.controller_config = {'id': 'owner'}
    controller.recovery_safety_observer = SimpleNamespace(
        motion_configuration=deepcopy(model), configuration={'controller_node': '/arm'})
    planned = prepared['steps'][0]
    planned.pop('preparation_id')
    planned['joint_trajectory'] = trajectory(0., 1., velocities=None)
    planned['continuous_motion'] = {'joint_trajectory': deepcopy(planned['joint_trajectory']),
                                    'configuration': model}
    planned['configuration_fingerprint'] = fingerprint(controller.controller_config)
    planned['preparation_id'] = fingerprint(planned)
    provider.register_program(prepared)
    checkpoint['model_execution'] = True
    physical = checkpoint['observations']['ur5e-3']['physical']
    physical['idle'] = False
    physical['stationary_contract'] = {'kind': 'idle_commanded_hold', 'requires_no_active_goals': True}
    checkpoint['controller_goals'] = {'/arm/recovery_state': {
        'instance_id': 'instance', 'controller': '/arm', 'command_revision': 3, 'contract_revision': 0,
        'holding': True, 'has_active_goal': False, 'has_pending_goal': False, 'contract_state': 'unlocked',
        'reservation_token': '', 'observed_stationary': False, 'stationary_samples': 0,
        'velocities': [.032], 'positions': [.000004], 'simulation_time': 1.,
    }}
    native.update(accepted=False, reason='Native owner has not observed a stationary hold',
                  observed_stationary=False, stationary_samples=0, observed_velocities=[.032])
    return provider, prepared, checkpoint, native


@pytest.mark.parametrize('native_available', [True, False])
def test_modeled_execution_does_not_require_certified_physics_or_exact_sampled_stillness(monkeypatch, native_available):
    from cais_spade_llm.resources.robot import gazebo_pick_place_controller

    provider, prepared, checkpoint, native = _modeled_execution_case()
    before = deepcopy(prepared), deepcopy(checkpoint), provider.ledger.snapshot()

    def query(*args):
        if not native_available:
            raise ValueError('Native motion contract service unavailable')
        return deepcopy(native)

    monkeypatch.setattr(gazebo_pick_place_controller, '_prepare_controller_contract', query)
    result = provider.prepare_execution_coverage(prepared=prepared, checkpoint=checkpoint)
    assert result['status'] == 'prepared' and result['model_execution_verified'] is True
    assert not result['native_preparation_verified'] and not result['physical_execution_verified']
    assert not result['dispatch_authorized'] and not result['command_sent']
    assumptions = result['model_execution_assumptions']
    assert assumptions['continuous_motion'] == prepared['steps'][0]['continuous_motion']
    assert assumptions['stationary_contracts']['ur5e-3'] == checkpoint['observations']['ur5e-3']['physical']['stationary_contract']
    assert assumptions['controller_goals']['/arm/recovery_state']['command_revision'] == 3
    assert not {'positions', 'velocities', 'stationary_samples'} & assumptions['controller_goals']['/arm/recovery_state'].keys()
    assert result['native_preparation_reason']
    assert before == (prepared, checkpoint, provider.ledger.snapshot())


@pytest.mark.parametrize('change', [
    'detached', 'boolean', 'controller_active', 'controller_pending', 'not_holding', 'native_reserved',
    'native_restarted', 'native_command_changed', 'stationary_contract', 'registered_model', 'controller_config',
])
def test_modeled_execution_requires_its_exact_owner_assumptions(monkeypatch, change):
    from cais_spade_llm.resources.robot import gazebo_pick_place_controller

    provider, prepared, checkpoint, native = _modeled_execution_case()
    state = checkpoint['controller_goals']['/arm/recovery_state']
    if change == 'detached':
        checkpoint.pop('model_execution')
    elif change == 'boolean':
        checkpoint['model_execution'] = 1
    elif change == 'controller_active':
        state['has_active_goal'] = True
    elif change == 'controller_pending':
        state['has_pending_goal'] = True
    elif change == 'not_holding':
        state['holding'] = False
    elif change == 'native_reserved':
        state['reservation_token'] = 'another grant'
    elif change == 'native_restarted':
        native['instance_id'] = 'restarted'
    elif change == 'native_command_changed':
        native['command_revision'] += 1
    elif change == 'stationary_contract':
        checkpoint['observations']['ur5e-3']['physical'].pop('stationary_contract')
    elif change == 'registered_model':
        provider.owner._controller.recovery_safety_observer.motion_configuration['joint_position_error']['slide'] = .1
    else:
        provider.owner._controller.controller_config['id'] = 'changed owner'
    monkeypatch.setattr(gazebo_pick_place_controller, '_prepare_controller_contract', lambda *args: deepcopy(native))
    result = provider.prepare_execution_coverage(prepared=prepared, checkpoint=checkpoint)
    assert result['status'] == 'NEEDS_CONTEXT' and not result['model_execution_verified']
    assert not result['dispatch_authorized'] and not result['command_sent']


def test_modeled_execution_detects_ledger_change_even_when_native_diagnostic_is_rejected(monkeypatch):
    from cais_spade_llm.resources.robot import gazebo_pick_place_controller

    provider, prepared, checkpoint, native = _modeled_execution_case()

    def query(*args):
        provider.ledger.prepare(resource_id='ur5e-3', task_id='concurrent', command={},
                                owner_identity={'preparation_id': 'concurrent'})
        return deepcopy(native)

    monkeypatch.setattr(gazebo_pick_place_controller, '_prepare_controller_contract', query)
    result = provider.prepare_execution_coverage(prepared=prepared, checkpoint=checkpoint)
    assert result['status'] == 'NEEDS_CONTEXT' and not result['model_execution_verified']
    assert 'changed during native preparation' in result['reason']


def _modeled_start_case():
    provider, prepared, _, _ = _modeled_execution_case()
    motion = deepcopy(prepared['steps'][0]['continuous_motion'])
    motion['configuration']['joint_position_error'] = {'slide': .01}
    start = {'model_execution': True, 'joint_names': ['slide'], 'joint_positions': [0.],
             'frame': 'world', 'launch_id': 'launch', 'attachment': {'model_name': None},
             'current_pose': [0., 0., 0., 0., 0., 0., 1.],
             'component_bounds': [{'id': 'observed::body::collision', 'link': 'body', 'bounds': [[-.05, .05]] * 3}],
             'geometry_source': {'configuration': 'retained'}, 'custody_complete': True,
             'observation_state': {'held_part': None, 'current_pose': [0., 0., 0., 0., 0., 0., 1.]}}
    observed = deepcopy(start)
    observed['joint_positions'] = [.004]
    observed['current_pose'][0] = .004
    observed['observation_state']['current_pose'][0] = .004
    observed['component_bounds'][0]['bounds'][0] = [-.046, .054]
    return start, observed, motion


def test_modeled_start_accepts_only_declared_joint_and_geometry_enclosures():
    start, observed, motion = _modeled_start_case()
    before = deepcopy(start), deepcopy(observed), deepcopy(motion)
    validate_prepared_start(observed, start, continuous_motion=motion)
    assert before == (start, observed, motion)
    start.pop('model_execution')
    with pytest.raises(ValueError, match='moved after physical motion preparation'):
        validate_prepared_start(observed, start, continuous_motion=motion)


@pytest.mark.parametrize(('change', 'reason'), [
    ('joint', 'joint_position_error'), ('pose', 'initial FK enclosure'),
    ('orientation', 'initial FK enclosure'), ('collision', 'observed::body::collision'),
    ('extra_collision', 'observed::unmodeled::collision'), ('missing_collision', 'component_bounds'),
    ('custody', 'observation_state'), ('configuration', 'geometry_source'),
    ('missing_assumption', 'joint_position_error'), ('root', 'root_pose'),
])
def test_modeled_start_rejects_observations_outside_the_exact_assumptions(change, reason):
    start, observed, motion = _modeled_start_case()
    if change == 'joint':
        observed['joint_positions'] = [.011]
    elif change == 'pose':
        observed['current_pose'][0] = .011
    elif change == 'orientation':
        observed['current_pose'][3:] = [0., 0., .1, (1. - .1 ** 2) ** .5]
    elif change == 'collision':
        observed['component_bounds'][0]['bounds'][0][1] = .061
    elif change == 'extra_collision':
        observed['component_bounds'].append({'id': 'observed::unmodeled::collision', 'bounds': [[1., 2.]] * 3})
    elif change == 'missing_collision':
        observed['component_bounds'] = []
    elif change == 'custody':
        observed['observation_state']['held_part'] = 'gear_small'
    elif change == 'configuration':
        observed['geometry_source']['configuration'] = 'changed'
    elif change == 'missing_assumption':
        motion['configuration'].pop('joint_position_error')
    else:
        start['root_pose'] = [0., 0., 0., 0., 0., 0., 1.]
        observed['root_pose'] = [.1, 0., 0., 0., 0., 0., 1.]
    with pytest.raises(ValueError, match=reason):
        validate_prepared_start(observed, start, continuous_motion=motion)


def test_modeled_joint_error_records_observed_moveit_parameter_without_changing_it(monkeypatch):
    pytest.importorskip('rcl_interfaces.srv')
    from unittest.mock import Mock
    from test_continuous_motion import configuration

    from cais_spade_llm.resources import continuous_geometry
    from cais_spade_llm.resources.robot.gazebo_pick_place_controller import GazeboPickPlaceController

    monkeypatch.setattr(continuous_geometry, 'load_continuous_geometry', lambda *args, **kwargs: configuration())
    response = SimpleNamespace(values=[SimpleNamespace(type=3, double_value=.01)])
    client = SimpleNamespace(wait_for_service=lambda **kwargs: True, call_async=Mock(return_value=response))
    node = SimpleNamespace(create_client=Mock(return_value=client), destroy_client=Mock())
    controller = SimpleNamespace(_node=node, _cb_group=None, _wait_future=lambda value, **kwargs: value,
                                 ee_link='body', arm_joint_names=['slide'])
    observer = SimpleNamespace(configuration={'root_link': 'world', 'model': 'dual_robot',
        'description_node': '/robot_state_publisher', 'controller_node': '/arm'},
        provider=SimpleNamespace(model_execution=True, reader=SimpleNamespace(parameter=lambda *args: 'fixture')))
    snapshot = {'models': {'dual_robot': {'joints': {}, 'links': {'world': {'pose': [0, 0, 0, 0, 0, 0, 1]}}}}}
    GazeboPickPlaceController.configure_recovery_safety_observer(controller, observer, snapshot)
    assert client.call_async.call_args.args[0].names == ['trajectory_execution.allowed_start_tolerance']
    assert node.create_client.call_args.args[1] == '/move_group/get_parameters'
    assert observer.motion_configuration['joint_position_error'] == {'slide': .01}
    assert observer.motion_configuration['joint_position_error_source'] == {
        'node': '/move_group', 'parameter': 'trajectory_execution.allowed_start_tolerance',
        'value': .01, 'physical_execution_verified': False}
    node.destroy_client.assert_called_once_with(client)


@pytest.mark.parametrize(('first_ns', 'last_ns', 'index'), [(1_000_000, 1_000_000_000, 0), (0, 0, 1)])
def test_invalid_native_timing_is_preserved_in_failed_preparation(controller, monkeypatch, first_ns, last_ns, index):
    messages = pytest.importorskip('trajectory_msgs.msg')
    from cais_spade_llm.resources.robot.gazebo_pick_place_controller import GazeboPickPlaceController

    trajectory = messages.JointTrajectory(joint_names=controller.arm_joint_names)
    for value, nanos in ((0., first_ns), (.1, last_ns)):
        point = messages.JointTrajectoryPoint(positions=[value] * len(controller.arm_joint_names))
        point.time_from_start.sec, point.time_from_start.nanosec = divmod(nanos, 1_000_000_000)
        trajectory.points.append(point)
    plan = SimpleNamespace(joint_trajectory=trajectory, multi_dof_joint_trajectory=SimpleNamespace(points=[]))
    monkeypatch.setattr(GazeboPickPlaceController, '_prepare_cartesian_motion', lambda *args, **kwargs: (plan,))
    result = prepare_ur5e_motion(controller, primitive='move_cartesian',
        params={'x': 1.5, 'y': 0., 'z': 1.}, start=_start(controller), binding={})
    assert result['status'] == 'NEEDS_CONTEXT' and result['command_sent'] is False
    assert f'point_index={index}' in result['reason']
    assert f'time_from_start_ns={first_ns if index == 0 else last_ns}' in result['reason']
    raw = result['joint_trajectory']
    assert raw['points'][0]['time_from_start']['nanosec'] == first_ns
    assert raw['duration_ns'] == last_ns
    assert len(raw['points']) == 2
    assert trajectory.points[0].time_from_start.nanosec == first_ns
    controller._send_simulation_joint_trajectory.assert_not_called()


@pytest.mark.parametrize('change', [None, 'position', 'velocity'])
def test_preparation_materializes_only_observed_one_nanosecond_initial_hold(controller, monkeypatch, change):
    messages = pytest.importorskip('trajectory_msgs.msg')
    from cais_spade_llm.resources.resource_safety_preparation import trajectory_record
    from cais_spade_llm.resources.robot.gazebo_pick_place_controller import GazeboPickPlaceController, trajectory_message

    trajectory = messages.JointTrajectory(joint_names=controller.arm_joint_names)
    for value, nanos in ((0., 1), (.1, 1_000_000_000)):
        point = messages.JointTrajectoryPoint(positions=[value] * len(controller.arm_joint_names),
            velocities=[0.] * len(controller.arm_joint_names), accelerations=[0.] * len(controller.arm_joint_names))
        point.time_from_start.sec, point.time_from_start.nanosec = divmod(nanos, 1_000_000_000)
        trajectory.points.append(point)
    if change == 'position':
        trajectory.points[0].positions[0] = .001
    elif change == 'velocity':
        trajectory.points[0].velocities[0] = .001
    original = trajectory_record(trajectory, validate=False)
    plan = SimpleNamespace(joint_trajectory=trajectory, multi_dof_joint_trajectory=SimpleNamespace(points=[]))
    monkeypatch.setattr(GazeboPickPlaceController, '_prepare_cartesian_motion', lambda *args, **kwargs: (plan,))
    result = prepare_ur5e_motion(controller, primitive='move_cartesian',
        params={'x': 1.5, 'y': 0., 'z': 1.}, start=_start(controller), binding={})
    assert result['planned_joint_trajectory'] == original
    assert trajectory_record(trajectory, validate=False) == original
    if change is None:
        assert result['status'] == 'prepared'
        retained = result['joint_trajectory']
        assert retained['points'][0]['time_from_start'] == {'sec': 0, 'nanosec': 0}
        assert retained['points'][0]['positions'] == _start(controller)['joint_positions']
        assert retained['points'][1:] == original['points']
        assert retained['duration_ns'] == original['duration_ns']
        assert trajectory_record(trajectory_message(retained)) == retained
        assert result['trajectory_preparation']['planned_joint_trajectory_fingerprint'] == fingerprint(original)
        assert result['trajectory_preparation']['prepared_joint_trajectory_fingerprint'] == fingerprint(retained)
    else:
        assert result['status'] == 'NEEDS_CONTEXT'
        assert 'does not represent the observed stationary start' in result['reason']
        assert result['joint_trajectory'] == original
    controller._send_simulation_joint_trajectory.assert_not_called()


@pytest.mark.parametrize(('allowance', 'offset', 'succeeds'), [
    (.01, .006, True), (.001, .002, False), (0., 0., True), (0., .000001, False),
    (.01, float('nan'), False),
])
def test_prepared_endpoint_uses_fresh_observation_and_retained_joint_error(allowance, offset, succeeds):
    pytest.importorskip('trajectory_msgs.msg')
    from cais_spade_llm.resources.robot.gazebo_pick_place_controller import execute_prepared_robot_step

    start, _, continuous = _modeled_start_case()
    continuous['configuration']['joint_position_error'] = {'slide': allowance}
    continuous['joint_trajectory']['header'] = {'frame_id': 'world', 'stamp': {'sec': 0, 'nanosec': 0}}
    continuous['joint_trajectory']['points'][-1]['positions'] = [.1]
    planned = {'status': 'prepared', 'observation_status': 'prepared', 'start': start,
               'configuration_fingerprint': fingerprint({}), 'binding': {'model_execution': True},
               'continuous_motion': continuous, 'joint_trajectory': deepcopy(continuous['joint_trajectory'])}
    planned['preparation_id'] = fingerprint(planned)
    before = deepcopy(planned)
    current = deepcopy(start)
    current['simulation_time'] = 42.
    calls = []

    def capture():
        calls.append('capture')
        return deepcopy(current)

    def dispatch(topic, trajectory, **kwargs):
        calls.append('dispatch')
        assert kwargs == {'joint_position_error': {'slide': allowance}}
        assert list(trajectory.points[-1].positions) == [.1]
        return True

    def snapshot(*, refresh):
        assert refresh is True and calls[-1] == 'dispatch'
        calls.append('refresh')
        current['joint_positions'] = [.1 + offset]
        # Actual receipt time is retained; it need not equal the modeled end.
        current['simulation_time'] = 43.25
        return deepcopy(current)

    controller = SimpleNamespace(execution_mode='simulation', controller_config={}, arm_joint_names=['slide'],
        arm_trajectory_topic='/arm/joint_trajectory', capture_recovery_safety_state=capture,
        _send_simulation_joint_trajectory=dispatch,
        recovery_safety_observer=SimpleNamespace(provider=SimpleNamespace(reader=SimpleNamespace(snapshot=snapshot))))
    result = execute_prepared_robot_step(controller, planned)
    assert result['success'] is succeeds
    assert result['observations']['simulation_time'] == 43.25
    assert calls == ['capture', 'dispatch', 'refresh', 'capture']
    assert planned == before and result['joint_trajectory'] == planned['joint_trajectory']


@pytest.mark.parametrize(('allowance', 'observed', 'succeeds'), [
    (.01, .006, True), (.001, .002, False), (0., 0., True), (0., .000001, False),
    (None, .004, True), (None, .006, False),
])
def test_native_result_endpoint_poll_uses_only_the_declared_joint_error(allowance, observed, succeeds, monkeypatch):
    messages = pytest.importorskip('trajectory_msgs.msg')
    from unittest.mock import Mock
    from cais_spade_llm.resources.robot.gazebo_pick_place_controller import GazeboPickPlaceController

    trajectory = messages.JointTrajectory(joint_names=['joint'])
    trajectory.points = [messages.JointTrajectoryPoint(positions=[0.])]
    result = SimpleNamespace(status=4, result=SimpleNamespace(error_code=0))
    goal = SimpleNamespace(accepted=True, status=4, get_result_async=lambda: result)
    client = SimpleNamespace(wait_for_server=lambda **kwargs: True, send_goal_async=lambda _: goal)
    polls = iter([True, False])
    controller = SimpleNamespace(
        _simulation_joint_clients={'/arm/follow_joint_trajectory': client},
        _wait_future=lambda future, *args: future, _get_joint_position=lambda name: observed,
        _motion_pending=lambda _: lambda: next(polls), _simulation_goal=None,
        arm_joint_names=['joint'], _angular_joint_error=GazeboPickPlaceController._angular_joint_error,
        _note_motion_dispatch=Mock(), _cancel_simulation_goal=Mock(),
        _node=SimpleNamespace(get_clock=lambda: SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=0))),
    )
    monkeypatch.setattr(time, 'sleep', lambda _: None)
    errors = None if allowance is None else {'joint': allowance}
    assert GazeboPickPlaceController._send_simulation_joint_trajectory(
        controller, '/arm/joint_trajectory', trajectory, joint_position_error=errors) is succeeds
    assert controller._last_simulation_controller_succeeded is True


@pytest.mark.parametrize(('change', 'reason'), [
    (None, None), ('cross_region', 'Fixed component occupancy changed'),
    ('unknown', 'Fixed component occupancy changed or is unknown'),
    ('missing_link', 'literal native link identities'), ('unknown_link', 'literal native link identities'),
    ('missing_regions', 'model_execution_regions'), ('missing_hold', 'commanded stationary model'),
    ('moving_escape', 'outside the configured initial enclosure'),
    ('link_changed', 'native link population changed'),
    ('captured_cross_region', 'Fixed component occupancy changed'),
    ('masked_cross_region', 'Fixed component occupancy changed'),
])
def test_fixed_component_start_requires_independent_definite_region_occupancy(change, reason):
    start, observed, motion = _modeled_start_case()
    motion['configuration']['joints'].append({'name': 'base_fixed', 'parent': 'world', 'child': 'base_inertia',
        'type': 'fixed', 'axis': [1., 0., 0.], 'xyz': [0., 0., 0.], 'rpy': [0., 0., 0.]})
    motion['configuration']['components'].append({'id': 'configured_base', 'link': 'base_inertia', 'bounds': [[-.1, .1]] * 3})
    fixed = {'id': 'native::base::collision', 'link': 'world', 'bounds': [[-.1, .1]] * 3}
    regions = {'assembly_board-v1': {'frame': 'world', 'bounds': [[.2, .3], [-1., 1.], [-1., 1.]]}}
    if change == 'masked_cross_region':
        motion['configuration']['components'][-1]['bounds'][0] = [-.4, -.3]
        fixed['bounds'][0] = [-.4, -.3]
        regions['assembly_board-v1']['bounds'][0] = [-.02, .02]
    for state in (start, observed):
        state['component_bounds'].append(deepcopy(fixed))
        state['model_execution_regions'] = deepcopy(regions)
        state['stationary_contract'] = {'kind': 'idle_commanded_hold', 'requires_no_running_tasks': True,
            'requires_no_active_goals': True, 'future_execution_tracking': 'not_established'}
    # Deliberately outside the retained base's floating-point world enclosure,
    # while preserving its definite occupancy in every configured region.
    observed['component_bounds'][-1]['bounds'][0][0] -= 1e-13
    if change == 'cross_region':
        observed['component_bounds'][-1]['bounds'][0] = [.21, .3]
    elif change == 'captured_cross_region':
        start['component_bounds'][-1]['bounds'][0] = [.21, .3]
    elif change == 'unknown':
        start['model_execution_regions']['assembly_board-v1']['bounds'][0] = [.1, .2]
    elif change == 'missing_link':
        observed['component_bounds'][-1].pop('link')
    elif change == 'unknown_link':
        observed['component_bounds'][-1]['link'] = 'unknown native link'
    elif change == 'missing_regions':
        start.pop('model_execution_regions')
    elif change == 'missing_hold':
        observed.pop('stationary_contract')
    elif change == 'moving_escape':
        observed['component_bounds'][0]['bounds'][0] = [-.05, .08]
    elif change == 'link_changed':
        observed['component_bounds'][0]['link'] = 'world'
    elif change == 'masked_cross_region':
        observed['component_bounds'][-1]['bounds'][0] = [-.01, .01]
    before = deepcopy((start, observed, motion))
    if reason is None:
        validate_prepared_start(observed, start, continuous_motion=motion)
    else:
        with pytest.raises(ValueError, match=reason):
            validate_prepared_start(observed, start, continuous_motion=motion)
    assert (start, observed, motion) == before
