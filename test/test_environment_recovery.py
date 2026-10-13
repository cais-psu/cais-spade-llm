from __future__ import annotations

"""Recorded owner capability preparation for all failure scenario resources."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from cais_spade_llm.agents.resource_agent.resource_agent import ResourceAgent
from cais_spade_llm.product.environment import EnvironmentProductContext
from cais_spade_llm.recovery_framework import ROOT, SCENE_PATH, PRODUCT_PATH, read_json
from cais_spade_llm.recovery_framework.offline_recovery import SavedRecoveryResource
from cais_spade_llm.resources.environment_recovery import (
    environment_owner_evidence, environment_owner_snapshot, environment_recovery_descriptor,
    check_environment_primitive_capability, project_environment_event,
)
from cais_spade_llm.resources.recovery_feasibility import derive_primitive_union


@pytest.fixture
def owner_context():
    meta = next(iter(read_json(PRODUCT_PATH).values()))
    scene = read_json(SCENE_PATH)
    from cais_spade_llm.resources.workflow_task_programs import workflow_task_program
    for machine in scene['machines']:
        machine['simulation_process'].pop('job_control', None)
        program = workflow_task_program('machine_part')
        function = {'function_name': program.pop('function_name'), 'status': program.pop('status'), 'program': program}
        scene['resource_programs']['resources'][machine['resource_id']] = {'functions': {'machine_part': function},
            'primitives': {'dwell': {'status': 'executable', 'recovery_selectable': False}}}
    return EnvironmentProductContext(scene=scene,
        product_order=read_json(ROOT / meta['product_order_file']),
        geometry=read_json(ROOT / meta['product_geometry_file'])['gazebo'])


def owner_for(context, rid):
    owner = SimpleNamespace(agent_name=rid, jid=rid + '@localhost', environment_context=context.resources[rid],
        environment_runtime=SimpleNamespace(context=context, setup={'execution_mode': 'simulation',
            'permitted_resources': ['M1'], 'recovery_permitted_resources': list(context.resources)}))
    return owner


def machine_proposal(rid='M1', result='square'):
    return {'event_name': 'proposed_event', 'resource_jid': rid + '@localhost', 'part_name': 'KET4_Square_4mm',
            'params': {'result': result}, 'expected_start_state': {'resource_state': 'loaded'},
            'expected_end_state': {'resource_state': 'completed', 'part_state': 'ready'}}


def test_raw_owner_snapshot_does_not_fabricate_idle_or_custody(owner_context):
    owner = owner_for(owner_context, 'M1')
    owner.environment_context.valuation.pop('resource_state')
    snapshot = environment_owner_snapshot(owner)
    assert 'current_state' not in snapshot and 'resource_state' not in snapshot and 'held_part' not in snapshot
    evidence = environment_owner_evidence(owner)
    assert 'M2' in evidence['recovery_permitted_resources']
    assert evidence['environment_models'] == owner_context.models
    assert 'M2' in evidence['environment_models']


def test_owner_snapshot_rejects_conflicting_retained_values(owner_context):
    owner = owner_for(owner_context, 'KMR')
    owner._primitive_state = {'held_part': 'KET4_Square_4mm'}
    with pytest.raises(ValueError, match='Conflicting retained'):
        environment_owner_snapshot(owner)


@pytest.mark.parametrize('rid', ['KMR', 'M1', 'M2', 'Conveyor', 'Buffer For Machined parts'])
def test_detached_legacy_snapshot_restores_only_recorded_declared_fields(owner_context, rid):
    evidence = environment_owner_evidence(owner_for(owner_context, rid))
    raw = environment_owner_snapshot(owner_for(owner_context, rid))
    model = environment_recovery_descriptor(evidence, raw, resource_jid=rid + '@localhost')
    entry = {'resource_type': 'resource', 'recovery_snapshot': {'current_state': 'legacy', 'resource_core': {}},
             'recovery_des_model': model}
    observations = {'jids': {rid: rid + '@localhost'}, 'resource_evidence': {rid + '@localhost': evidence}}
    original = deepcopy(observations)
    owner = SavedRecoveryResource(rid + '@localhost', entry, observations)
    result = owner.get_recovery_snapshot()
    assert result['current_state'] == 'legacy'
    for field, value in evidence['environment_valuation'][rid].items():
        assert field in result and result[field] == value and type(result[field]) is type(value)
    assert 'part_state' not in result and 'part_location' not in result
    assert observations == original and entry['recovery_snapshot'] == {'current_state': 'legacy', 'resource_core': {}}
    sources = owner.get_recovery_resource_evidence()['sources']
    assert all('recovery_snapshot/' + field in sources for field in evidence['environment_valuation'][rid])


@pytest.mark.parametrize('recorded,prepared', [(None, 'part'), (False, 0), (0, 0.)])
def test_detached_snapshot_preserves_null_and_rejects_typed_conflicts(recorded, prepared):
    evidence = {'resource_id': 'Conveyor', 'snapshot': {'belt_stopped': recorded},
        'environment_models': {'Conveyor': {'state_variables': {'belt_stopped': {'scope': 'resource'}}}}}
    owner = SavedRecoveryResource('owner', {'resource_type': 'resource',
        'recovery_snapshot': {'belt_stopped': prepared}, 'recovery_des_model': {'state_variables': {}}},
        {'jids': {'Conveyor': 'owner'}, 'resource_evidence': {'owner': evidence}})
    with pytest.raises(ValueError, match='Conflicting recorded resource state'):
        owner.get_recovery_snapshot()


def test_detached_snapshot_does_not_invent_absent_declared_values():
    evidence = {'resource_id': 'Conveyor', 'snapshot': {},
        'environment_models': {'Conveyor': {'state_variables': {'belt_stopped': {'scope': 'resource'}}}}}
    owner = SavedRecoveryResource('owner', {'resource_type': 'resource',
        'recovery_snapshot': {'current_state': 'legacy'}, 'recovery_des_model': {'state_variables': {}}},
        {'jids': {'Conveyor': 'owner'}, 'resource_evidence': {'owner': evidence}})
    assert 'belt_stopped' not in owner.get_recovery_snapshot()


def test_detached_machine_capabilities_are_exact_recorded_declarations(owner_context):
    evidence = environment_owner_evidence(owner_for(owner_context, 'M2'))
    owner = SavedRecoveryResource('M2@localhost', {'resource_type': 'resource',
        'recovery_snapshot': {'current_state': 'legacy'}},
        {'jids': {'M2': 'M2@localhost'}, 'resource_evidence': {'M2@localhost': evidence}})
    source = evidence['environment_models']['M2']
    machine = next(row for row in evidence['inputs']['scene']['machines'] if row['resource_id'] == 'M2')
    assert owner.static_capabilities['process_capabilities'] == source['process_capabilities']
    assert owner.static_capabilities['current_configuration'] == source['current_configuration']
    assert owner.static_capabilities['program_options'] == machine['program_options']
    assert not {'events', 'primitive_catalog', 'assignments'} & set(owner.static_capabilities)


def test_saved_mobile_resource_locations_remain_discoverable(owner_context):
    from cais_spade_llm.resources.environment_recovery import environment_declared_capabilities

    evidence = environment_owner_evidence(owner_for(owner_context, 'KMR'))
    original = deepcopy(evidence)
    capabilities = environment_declared_capabilities(evidence)
    assert capabilities['reachable_locations'] == ['Storage', 'M1', 'M2']
    assert not {'events', 'primitive_support', 'predefined_routes'} & capabilities.keys()
    assert evidence == original


@pytest.mark.parametrize(('rid', 'status'), [('M1', 'FEASIBLE'), ('M2', 'INFEASIBLE')])
def test_machine_uses_current_program_not_capability_result_only(owner_context, rid, status):
    owner = owner_for(owner_context, rid)
    evidence = environment_owner_evidence(owner)
    snapshot = environment_owner_snapshot(owner)
    proposal = machine_proposal(rid)
    model = environment_recovery_descriptor(evidence, snapshot, resource_jid=owner.jid)
    support = derive_primitive_union(task=proposal, recovery_des_model=model, recovery_snapshot=snapshot)
    assert [row['primitive'] for row in support['primitives']] == ['dwell']
    assert support['deferred_valuation_fields'] == ['part_state']
    assert all(row['parameter_bindings']['result'] == 'square' for row in support['matching_transitions'])
    original = deepcopy(evidence)
    checked = check_environment_primitive_capability(primitive='dwell', params={}, task=proposal,
        recovery_snapshot=snapshot, resource_evidence=evidence)
    assert checked['feasibility_status'] == status, checked
    assert checked['evidence']['command_sent'] is False
    assert evidence == original
    if rid == 'M2':
        assert 'Current program cannot establish' in checked['reason']
    assert model['primitive_catalog'][0]['recovery_selectable'] is False


def test_live_and_detached_use_same_pure_machine_validation(owner_context):
    owner = owner_for(owner_context, 'M1')
    evidence = environment_owner_evidence(owner)
    snapshot = environment_owner_snapshot(owner)
    model = environment_recovery_descriptor(evidence, snapshot, resource_jid=owner.jid)
    entry = {'resource_type': 'resource', 'recovery_snapshot': snapshot, 'recovery_des_model': model,
             'resource_evidence': evidence}
    saved = SavedRecoveryResource(owner.jid, entry, {'jids': {'M1': owner.jid}, 'run_id': 'run'})
    proposal = machine_proposal()
    result = saved.validate_recovery_primitive_support(task=proposal, recovery_des_model=model,
        recovery_snapshot=snapshot, physical_input={'primitive_support_semantics': 'successor_primitive_union'})
    assert result['feasibility_status'] == 'FEASIBLE', result
    projection = saved.project_recovery_outline_event(task=proposal, recovery_snapshot=snapshot,
        recovery_des_model=model, physical_input={})
    assert projection['status'] == 'needs_context'
    assert not projection['interval_complete'] and not projection['execution_authorized']


def test_unavailable_owner_remains_discoverable_but_rejected(owner_context):
    owner = owner_for(owner_context, 'M1')
    owner_context.unavailable_resources.add('M1')
    evidence = environment_owner_evidence(owner)
    snapshot = environment_owner_snapshot(owner)
    assert 'M1' in evidence['environment_models']
    checked = check_environment_primitive_capability(primitive='dwell', params={}, task=machine_proposal(),
        recovery_snapshot=snapshot, resource_evidence=evidence)
    assert checked['feasibility_status'] == 'INFEASIBLE'
    assert checked['reason'] == 'Resource is recorded unavailable'


def test_kmr_keeps_exact_seven_joint_and_pose_formats(owner_context):
    from cais_spade_llm.recovery_framework.kmr_recovery import check_kmr_primitive_capability
    owner = owner_for(owner_context, 'KMR')
    evidence = environment_owner_evidence(owner)
    snapshot = environment_owner_snapshot(owner)
    result = check_kmr_primitive_capability(primitive='move_cartesian', params={'x': 0, 'y': 0, 'z': 1},
        task={'resource_jid': owner.jid}, recovery_snapshot=snapshot, resource_evidence=evidence)
    assert result['feasibility_status'] == 'NEEDS_CONTEXT'
    assert any('joint_positions' in value for value in result['context_requests'])
    model = environment_recovery_descriptor(evidence, snapshot, resource_jid=owner.jid)
    assert model['model_complete'] is True
    cartesian = next(row for row in model['primitive_catalog'] if row['name'] == 'move_cartesian')
    assert set(cartesian['params']) == {'target', 'waypoints', 'seed'}
    assert cartesian['required_params'] == ['target']


def test_typed_bindings_validate_after_successor_binding_and_retain_missing(owner_context):
    owner = owner_for(owner_context, 'KMR')
    evidence = environment_owner_evidence(owner)
    snapshot = environment_owner_snapshot(owner)
    model = environment_recovery_descriptor(evidence, snapshot, resource_jid=owner.jid)
    proposal = {'resource_jid': owner.jid, 'part_name': None, 'params': {},
                'expected_end_state': {'resource_location': 'M1'}}
    support = derive_primitive_union(task=proposal, recovery_des_model=model, recovery_snapshot=snapshot)
    assert support['matching_transitions']
    coordinate = next(row for row in support['matching_transitions'] if row['event_name'] == 'move_to_location')
    assert 'x' not in coordinate['parameter_bindings']
    assert coordinate['parameter_bindings']['observed_resource_location'] == 'M1'
    proposal['params']['x'] = False
    support = derive_primitive_union(task=proposal, recovery_des_model=model, recovery_snapshot=snapshot)
    assert not any(row['event_name'] == 'move_to_location' for row in support['matching_transitions'])


def test_kmr_location_projection_binds_recorded_dock_without_committing_or_planning(owner_context, monkeypatch):
    from cais_spade_llm.recovery_framework.kmr_outline_evidence import _declarations
    from cais_spade_llm.recovery_framework.kmr_recovery import prepare_kmr_primitive_parameters

    owner = owner_for(owner_context, 'KMR')
    owner.environment_context.valuation['resource_location'] = 'M2'
    evidence = environment_owner_evidence(owner)
    snapshot = environment_owner_snapshot(owner)
    model = environment_recovery_descriptor(evidence, snapshot, resource_jid=owner.jid)
    proposal = {'event_name': 'unseen_event', 'resource_jid': owner.jid, 'part_name': None,
                'expected_end_state': {'resource_location': 'M1', 'resource_state': 'idle'}}
    original = deepcopy((proposal, snapshot, model, evidence))
    prepared = prepare_kmr_primitive_parameters(task=proposal, recovery_des_model=model,
        recovery_snapshot=snapshot, resource_evidence=evidence, resource_jid=owner.jid)
    calculation = next(row for row in prepared['parameter_calculations'] if row['event_name'] == 'move_to_location')
    assert 'x' not in calculation['parameter_bindings']
    assert 'x' in calculation['resolved_parameter_bindings']
    assert calculation['parameter_sources']['x']['resource_location'] == 'M1'
    candidates = _declarations(proposal, model, evidence, {})
    location = next(row for row in candidates if row['task']['event_name'] == 'move_to_location')
    assert location['task']['parameters'] == calculation['resolved_parameter_bindings']
    seen = []
    def unresolved(**kwargs):
        seen.append(deepcopy(kwargs['declared']))
        return {'status': 'NEEDS_CONTEXT', 'reason': 'motion is unavailable'}
    monkeypatch.setattr('cais_spade_llm.recovery_framework.kmr_outline_evidence.project_kmr_event', unresolved)
    result = project_environment_event(task=proposal, recovery_snapshot=snapshot, recovery_des_model=model,
                                     physical_input={'resource_evidence': evidence})
    assert any(row['task'] == location['task'] for row in seen)
    assert result['status'] == 'needs_context'
    assert (proposal, snapshot, model, evidence) == original


def static_machine_projection_input(context, rid='M1', part='KET4_Square_4mm'):
    owner = owner_for(context, rid)
    context.resources['Storage'].valuation['inventory.' + part] = False
    context.resources[rid].valuation.update(resource_state='loaded', part_name=part)
    context.part_tracker[part].update(state='loaded', location=rid)
    evidence = environment_owner_evidence(owner)
    snapshot = environment_owner_snapshot(owner)
    model = environment_recovery_descriptor(evidence, snapshot, resource_jid=owner.jid)
    resources = deepcopy(evidence['environment_valuation'])
    parts = {name: {**deepcopy(value), 'current_pose': [0., 0., 0., 0., 0., 0., 1.]}
             for name, value in evidence['environment_products'].items()}
    physical = {'jids': {rid: rid + '@localhost' for rid in resources},
        'snapshot': {'frame': 'world', 'resources': resources, 'parts': parts},
        'geometry': {'frame': 'world', 'regions': {},
            'resources': {rid: {'stationary_only': True, 'component_bounds': [
                {'id': rid, 'bounds': [[0., 1.], [0., 1.], [0., 1.]]}]} for rid in resources},
            'parts': {name: {'local_bounds': [[0., .01], [0., .01], [0., .01]]} for name in parts}},
        'resource_models': {}, 'resources': {rid: {'model_static': True} for rid in resources},
        'environment_valuation': deepcopy(evidence['environment_valuation']),
        'products': deepcopy(evidence['environment_products']),
        'stationary_part_evidence': {name: {'status': 'supported', 'current_pose': row['current_pose'],
            'source': {'kind': 'recorded', 'fingerprint': 'fixture'}} for name, row in parts.items()}}
    return owner, evidence, snapshot, model, physical


def test_missing_owner_geometry_uses_saved_native_bounds_without_live_observation(monkeypatch):
    from cais_spade_llm.recovery_framework import gazebo_safety_preparation
    from cais_spade_llm.recovery_framework.recovery_observations import restore_recorded_resource_geometry

    monkeypatch.setattr(gazebo_safety_preparation.GazeboSafetyReader, '__init__',
        lambda *_args, **_kwargs: pytest.fail('Offline calculation requested Gazebo'))
    collision = {'id': 'dual_robot::ur5e_4_wrist_3_link::collision', 'bounds': [[0, 1], [0, 1], [0, 1]]}
    record = {'instance_id': 'recorded', 'attachment_revision': 9, 'attachment_complete': True,
        'attachments': [], 'simulation_time': 12., 'observed_monotonic': 123.,
        'models': {'dual_robot': {'static': False, 'links': {
            'ur5e_4_wrist_3_link': {'collisions': [collision]}}}}}
    physical = {'snapshot': {'frame': 'world', 'resources': {}},
        'environment_models': {'ur5e-4': {}}, 'environment_valuation': {'ur5e-4': {'held_part': None}},
        'geometry': {'frame': 'world', 'resources': {}}, 'physics_evidence': record}
    inputs = {'scene': {'safety_preparation': {'resources': {
        'ur5e-4': {'model': 'dual_robot', 'link_prefix': 'ur5e_4_', 'reference_link': 'ur5e_4_tool0'}}}}}
    original = deepcopy(physical)
    result = restore_recorded_resource_geometry(physical, inputs)
    assert result['snapshot']['resources']['ur5e-4']['held_part'] is None
    assert 'current_pose' not in result['snapshot']['resources']['ur5e-4']
    assert result['recorded_geometry_supplements']['ur5e-4']['coverage'] == 'endpoint'
    assert result['recorded_geometry_supplements']['ur5e-4']['motion_interval_supported'] is False
    assert result['resources']['ur5e-4']['simulation_time'] == 12.
    assert 'idle' not in result['resources']['ur5e-4']
    assert physical == original
    # A second calculation neither duplicates evidence nor changes its source.
    assert restore_recorded_resource_geometry(result, inputs) == result
    physical['physics_evidence']['attachments'] = [{'model1': 'dual_robot', 'link1': 'ur5e_4_wrist_3_link',
        'model2': 'gear_small', 'link2': 'link'}]
    rejected = restore_recorded_resource_geometry(physical, inputs)
    assert rejected['recorded_geometry_supplements']['ur5e-4']['status'] == 'NEEDS_CONTEXT'
    assert rejected['snapshot']['resources'] == {}


def test_unresolved_participant_is_not_removed_from_discovery():
    from cais_spade_llm.recovery_framework.recovery_observations import restore_recorded_resource_geometry

    physical = {'snapshot': {'resources': {}}, 'environment_models': {'ur5e-4': {}},
        'physics_evidence': {'instance_id': 'recorded', 'attachment_complete': True, 'models': {}}}
    result = restore_recorded_resource_geometry(physical, {})
    assert result['environment_models'] == physical['environment_models']
    assert result['recorded_geometry_supplements']['ur5e-4']['status'] == 'NEEDS_CONTEXT'


def test_static_machine_projection_records_supported_interval_and_declared_completion(owner_context):
    from cais_spade_llm.agents.central_controller.outline_evidence_safety import bind_outline_owner_safety_evidence
    owner, evidence, snapshot, model, physical = static_machine_projection_input(owner_context)
    task = machine_proposal()
    task['expected_end_state']['part_state'] = 'completed'
    original = deepcopy(physical)
    result = project_environment_event(task=task, recovery_snapshot=snapshot, recovery_des_model=model,
        physical_input={'resource_evidence': evidence, 'physical_evidence': physical})
    assert result['status'] == 'projected', result
    assert result['interval_complete'] is True
    assert [row['resources']['M1']['resource_state'] for row in result['observations']] == ['loaded', 'loaded', 'completed']
    assert result['owner_safety_evidence']['task_evidence']['events'][0]['function'] == 'machine_part'
    assert result['observations'][1]['continuous_interval'] == ['0', str(result['observations'][-1]['time'])]
    assert result['successor_evidence']['products']['KET4_Square_4mm']['processCompleted']
    assert physical == original
    bound = bind_outline_owner_safety_evidence(result)
    assert bound['observations'][1]['owner_safety_evidence']['interval_complete'] is True
    assert not result['execution_authorized']


@pytest.mark.parametrize('missing', ['peer_hold', 'part_support', 'part_state'])
def test_machine_projection_keeps_missing_or_conflicting_evidence_unresolved(owner_context, missing):
    owner, evidence, snapshot, model, physical = static_machine_projection_input(owner_context)
    task = machine_proposal()
    task['expected_end_state']['part_state'] = 'completed'
    if missing == 'peer_hold':
        physical['resources']['ur5e-3']['model_static'] = False
    elif missing == 'part_support':
        physical['stationary_part_evidence'].pop('KET4_Square_4mm')
    else:
        task['expected_end_state']['part_state'] = 'ready'
    original = deepcopy(physical)
    result = project_environment_event(task=task, recovery_snapshot=snapshot, recovery_des_model=model,
        physical_input={'resource_evidence': evidence, 'physical_evidence': physical})
    assert result['status'] == ('rejected' if missing == 'part_state' else 'needs_context')
    assert not result['interval_complete'] and not result['execution_authorized']
    assert physical == original
    if missing != 'part_state':
        assert result['declared_projections']


def test_kmr_custody_model_uses_captured_joint_limits_and_part_widths(owner_context):
    from cais_spade_llm.recovery_framework.kmr_live_safety import primitive_model
    owner = owner_for(owner_context, 'KMR')
    joint = owner_context.inputs['scene']['KMR']['gripper_joint']
    owner.recovery_safety_description = f'<robot><joint name="{joint}" type="prismatic"><limit lower="0" upper="0.11"/></joint></robot>'
    owner.recovery_safety_observer = SimpleNamespace(motion_configuration={
        'joint_names': ['joint_a1'], 'joints': [{'name': joint, 'frozen_position': .11}]})
    result = primitive_model(owner)
    record = result.configuration['custody_by_part']
    assert record['default']['joint_positions']['open'][joint] == .11
    assert record['default']['joint_positions']['closed'][joint] == .015
    assert record['KET16_Square_16mm']['joint_positions']['closed'][joint] == .024
    assert record['default']['joint_bounds'][joint] == [0., .11]
    assert record['default']['release_settling_modeled'] is False
    assert record['default']['attachment_link'] == owner_context.inputs['scene']['KMR']['task_execution']['attach_link']


def transport_projection_input(context, rid):
    part = 'KET4_Square_4mm'
    context.resources['Storage'].valuation['inventory.' + part] = False
    if rid == 'Conveyor':
        context.resources[rid].valuation.update({'part_location.' + part: 'loading_position_1', 'part_order.' + part: 0})
        pose = [*context.inputs['scene']['machines'][0]['conveyor_loading_pose'][:3], 0., 0., 0., 1.]
    else:
        context.resources[rid].valuation['zone_1_part'] = part
        pose = [*context.inputs['scene'][rid]['slot_poses'][0][:3], 0., 0., 0., 1.]
    context.part_tracker[part].update(state='ready', location=rid)
    owner = owner_for(context, rid)
    evidence, snapshot = environment_owner_evidence(owner), environment_owner_snapshot(owner)
    model = environment_recovery_descriptor(evidence, snapshot, resource_jid=owner.jid)
    resources = deepcopy(evidence['environment_valuation'])
    parts = {name: {**deepcopy(value), 'model_name': name,
        'current_pose': (deepcopy(pose) if name == part else [5., 5., 0., 0., 0., 0., 1.])}
        for name, value in evidence['environment_products'].items()}
    physical = {'jids': {name: name + '@localhost' for name in resources},
        'snapshot': {'frame': 'world', 'resources': resources, 'parts': parts},
        'geometry': {'frame': 'world', 'regions': {},
            'resources': {name: {'stationary_only': True, 'component_bounds': [
                {'id': name, 'bounds': [[10., 11.], [10., 11.], [10., 11.]]}]} for name in resources},
            'parts': {name: {'local_bounds': [[-.05, .05], [-.05, .05], [-.05, .05]]} for name in parts}},
        'resource_models': {}, 'resources': {name: {'model_static': True} for name in resources},
        'environment_valuation': deepcopy(evidence['environment_valuation']),
        'products': deepcopy(evidence['environment_products']),
        'stationary_part_evidence': {name: {'status': 'supported', 'current_pose': row['current_pose'],
            'source': {'kind': 'recorded', 'fingerprint': 'fixture'}} for name, row in parts.items()},
        'physics_evidence': {'instance_id': 'test', 'simulation_time': 1., 'gravity': [0., 0., -9.81],
            'physics_engine': 'ode', 'contact_surface_layer': 0., 'contacts': [], 'models': {
                part: {'static': False, 'pose': deepcopy(pose), 'links': {'link': {
                    'mass': .1, 'pose': deepcopy(pose), 'center_of_mass': pose[:3],
                    'linear_velocity': [0., 0., 0.], 'angular_velocity': [0., 0., 0.],
                    'collisions': [{'id': part + '::collision', 'pose': deepcopy(pose), 'contact_min_depth': 0.,
                        'geometry': {'type': 'box', 'size': [.1, .1, .1]}}]}}},
                'support': {'static': True, 'links': {'link': {'collisions': [{
                    'id': 'support::collision', 'pose': [pose[0] + 1., pose[1], pose[2] - .1, 0., 0., 0., 1.],
                    'contact_min_depth': 0., 'geometry': {'type': 'box', 'size': [8., 2., .1]}}]}}}}},
        'part_support_policy': {'kind': 'gravity_supported_rest', 'requires_no_active_goals': True,
            'future_execution_tracking': 'not_established', 'stationary_linear_speed_max': .001,
            'stationary_angular_speed_max': .001}}
    if rid == 'Conveyor':
        task = {'resource_jid': owner.jid, 'part_name': part, 'event_name': 'new_transport', 'params': {},
            'expected_end_state': {'part_location.' + part: 'loading_position_2'}}
    else:
        task = {'resource_jid': owner.jid, 'part_name': part, 'event_name': 'new_transport', 'params': {},
            'expected_end_state': {'zone_1_part': None, 'zone_2_part': part}}
    return task, evidence, snapshot, model, physical


@pytest.mark.parametrize('rid', ['Conveyor', 'Buffer For Machined parts'])
def test_transport_projects_shared_displacement_and_retains_peer_scene(owner_context, rid, monkeypatch):
    import asyncio
    from unittest.mock import Mock
    from cais_spade_llm.recovery_framework.outline_gazebo_evidence import prepare_owner_event_evidence
    from cais_spade_llm.resources import resource_safety_preparation

    task, evidence, snapshot, model, physical = transport_projection_input(owner_context, rid)
    robot_preview = Mock(side_effect=AssertionError('Transport requested a robot preview'))
    monkeypatch.setattr(resource_safety_preparation, 'prepare_outline_event_preview_request', robot_preview)
    support = derive_primitive_union(task=task, recovery_des_model=model, recovery_snapshot=snapshot)
    assert support['matching_transitions'], support
    source = deepcopy(physical)
    projected = project_environment_event(task=task, recovery_snapshot=snapshot, recovery_des_model=model,
        physical_input={'physical_evidence': physical, 'resource_evidence': evidence})
    assert projected['status'] == 'projected', projected
    assert 'owner_safety_evidence_unresolved' not in projected, projected.get('owner_safety_evidence_unresolved')
    assert projected['interval_complete'] is True
    assert projected['observations'][1]['continuous_cells']
    part = task['part_name']
    assert projected['projected_snapshot']['parts'][part]['current_pose'][0] > physical['snapshot']['parts'][part]['current_pose'][0]
    assert projected['projected_snapshot']['resources']['ur5e-3'] == physical['snapshot']['resources']['ur5e-3']
    assert projected['successor_evidence']['stationary_part_evidence'][part]['status'] == 'supported'
    assert not projected['execution_authorized'] and not projected['command_sent']
    owner = SavedRecoveryResource(task['resource_jid'], {'resource_type': 'resource',
        'recovery_snapshot': snapshot, 'recovery_des_model': model, 'resource_evidence': evidence},
        {'jids': {rid: task['resource_jid']}})
    prepared = asyncio.run(prepare_owner_event_evidence(owner, task=task, recovery_snapshot=snapshot,
        recovery_des_model=model, physical_evidence=physical, resource_evidence=evidence,
        observations={}, allow_live_capture=False))
    assert prepared['status'] == 'prepared', prepared
    assert prepared['geometry_evidence'] == projected
    robot_preview.assert_not_called()
    assert physical == source


def test_transport_missing_support_is_unresolved_without_symbolic_only_projection(owner_context):
    task, evidence, snapshot, model, physical = transport_projection_input(owner_context, 'Buffer For Machined parts')
    del physical['physics_evidence']
    projected = project_environment_event(task=task, recovery_snapshot=snapshot, recovery_des_model=model,
        physical_input={'physical_evidence': physical, 'resource_evidence': evidence})
    assert projected['status'] == 'needs_context'
    assert projected['observations'] == []
    assert any('support' in value for value in projected['unresolved'])


@pytest.mark.parametrize('primitive,params', [('move_joints', {'joints': [0.] * 7}), ('rotate_joint', {'joint_name': 'joint_a1', 'delta_deg': 1.})])
def test_kmr_capability_keeps_declared_cartesian_mode(owner_context, primitive, params):
    from cais_spade_llm.recovery_framework.kmr_recovery import check_kmr_primitive_capability
    evidence = environment_owner_evidence(owner_for(owner_context, 'KMR'))
    names = owner_context.inputs['scene']['KMR']['arm_joint_names']
    evidence['snapshot'].update(joint_names=names, joint_positions=[0.] * 7,
        continuous_motion_configuration={'joints': [{'name': name, 'position_bounds': [-3., 3.]} for name in names]})
    result = check_kmr_primitive_capability(primitive=primitive, params=params,
        task={'resource_jid': 'KMR@localhost'}, recovery_snapshot={}, resource_evidence=evidence)
    assert result['feasibility_status'] == 'INFEASIBLE'
    assert 'Cartesian' in result['reason']


def test_kmr_home_missing_custody_does_not_mean_empty(owner_context):
    from cais_spade_llm.recovery_framework.kmr_recovery import check_kmr_primitive_capability
    evidence = environment_owner_evidence(owner_for(owner_context, 'KMR'))
    names = owner_context.inputs['scene']['KMR']['arm_joint_names']
    evidence['snapshot'].update(joint_names=names, joint_positions=[0.] * 7,
        continuous_motion_configuration={'joints': [{'name': name, 'position_bounds': [-3., 3.]} for name in names]})
    evidence['snapshot'].pop('held_part')
    result = check_kmr_primitive_capability(primitive='move_to_named_pose', params={'pose_name': 'home'},
        task={'resource_jid': 'KMR@localhost'}, recovery_snapshot={}, resource_evidence=evidence)
    assert result['feasibility_status'] == 'NEEDS_CONTEXT'
    assert any('held_part' in ref for ref in result['context_requests'])


@pytest.mark.parametrize('case,status', [('empty', 'FEASIBLE'), ('missing_custody', 'NEEDS_CONTEXT'),
    ('loaded', 'FEASIBLE'), ('missing_transform', 'NEEDS_CONTEXT'), ('conflicting_transform', 'INFEASIBLE'),
    ('conflicting_part', 'INFEASIBLE'), ('empty_with_transform', 'INFEASIBLE'), ('empty_attached', 'INFEASIBLE')])
def test_kmr_base_parameters_use_recorded_custody(owner_context, case, status):
    from cais_spade_llm.recovery_framework.kmr_recovery import check_kmr_primitive_capability

    evidence = environment_owner_evidence(owner_for(owner_context, 'KMR'))
    names = owner_context.inputs['scene']['KMR']['arm_joint_names']
    snapshot = evidence['snapshot']
    snapshot.update(joint_names=names, joint_positions=[0.] * 7,
        base_pose=[0., 0., 0., 0., 0., 0., 1.], held_part=None,
        continuous_motion_configuration={'joints': [{'name': name, 'position_bounds': [-3., 3.]} for name in names]})
    params = {'target_pose': [1., 2., 0.], 'waypoints': []}
    transform = [0., 0., .2, 0., 0., 0., 1.]
    if case in {'loaded', 'missing_transform', 'conflicting_transform', 'conflicting_part'}:
        snapshot['held_part'] = 'KET4_Square_4mm'
        evidence['attachment'] = {'attached': True, 'held_part': snapshot['held_part'], 'grasp_transform': transform}
        params['transform'] = deepcopy(transform)
    if case == 'missing_transform':
        evidence['attachment'].pop('grasp_transform')
    elif case == 'conflicting_transform':
        params['transform'][2] = .3
    elif case == 'conflicting_part':
        evidence['attachment']['held_part'] = 'gear_small'
    elif case == 'missing_custody':
        snapshot.pop('held_part')
    elif case == 'empty_with_transform':
        params['transform'] = transform
    elif case == 'empty_attached':
        evidence['attachment'] = {'attached': True, 'held_part': 'gear_small'}
    before = deepcopy((snapshot, evidence, params))
    result = check_kmr_primitive_capability(primitive='move_base', params=params,
        task={'resource_jid': 'KMR@localhost'}, recovery_snapshot=snapshot, resource_evidence=evidence)
    assert result['feasibility_status'] == status, result
    assert not result['evidence']['command_sent']
    assert (snapshot, evidence, params) == before


def test_kmr_preparation_records_optional_transform_omission_per_contributor(owner_context):
    from cais_spade_llm.recovery_framework.kmr_recovery import prepare_kmr_primitive_parameters
    from cais_spade_llm.resources.environment_recovery import environment_primitive_catalog

    evidence = environment_owner_evidence(owner_for(owner_context, 'KMR'))
    snapshot = deepcopy(evidence['snapshot'])
    snapshot['held_part'] = None
    model = {'state_variables': {'resource_state': {'scope': 'resource', 'domain': ['idle', 'positioned']}},
        'events': [{'event_name': 'move_to_resource',
            'parameter_bindings': {'source_resource': {'equals': source}, 'target_resource': {'equals': 'Storage'}},
            'guards': {}, 'updates': {'resource_state': {'set': 'positioned'}},
            'primitive_support': [{'primitive': 'move_base', 'params': {
                'target_pose': {'$state': 'base_target_pose'}, 'waypoints': {'$state': 'base_waypoints'},
                'transform': {'$state': 'previous', 'path': ['grasp_transform']}}}]} for source in ('M1', 'M2')],
        'primitive_catalog': environment_primitive_catalog(evidence)}
    task = {'event_name': 'arbitrary_recovery_identifier', 'part_name': None,
            'expected_end_state': {'resource_state': 'positioned'}}
    before = deepcopy((task, snapshot, evidence, model))
    result = prepare_kmr_primitive_parameters(task=task, recovery_des_model=model, recovery_snapshot=snapshot,
        resource_evidence=evidence, resource_jid='KMR@localhost')
    witnesses = result['parameter_witnesses']
    assert len(witnesses) == 2
    assert {row['parameter_bindings']['source_resource'] for row in witnesses} == {'M1', 'M2'}
    assert all(row['omitted_params'] == ['transform'] and 'transform' not in row['params'] for row in witnesses)
    assert len({row['snapshot_fingerprint'] for row in witnesses}) == 1
    assert (task, snapshot, evidence, model) == before


def test_machine_projection_does_not_claim_unsupported_product_field(owner_context):
    owner, evidence, snapshot, model, physical = static_machine_projection_input(owner_context)
    task = machine_proposal()
    task['expected_end_state'] = {'resource_state': 'completed', 'new_part_claim': True}
    model['state_variables']['new_part_claim'] = {'scope': 'part', 'domain': [True, False]}
    result = project_environment_event(task=task, recovery_snapshot=snapshot, recovery_des_model=model,
        physical_input={'resource_evidence': evidence, 'physical_evidence': physical})
    assert result['status'] == 'needs_context'
    assert result['observations'] == []


@pytest.mark.parametrize(('rid', 'field'), [('Conveyor', 'speed_mps'), ('Buffer For Machined parts', 'speed_mps')])
def test_transport_does_not_substitute_task_parameters_for_declared_program(owner_context, rid, field):
    task, evidence, snapshot, model, physical = transport_projection_input(owner_context, rid)
    task['params'][field] = .01
    result = project_environment_event(task=task, recovery_snapshot=snapshot, recovery_des_model=model,
        physical_input={'resource_evidence': evidence, 'physical_evidence': physical})
    assert result['status'] == 'needs_context'
    assert any('declared owner transport program' in value for value in result['unresolved'])


def test_accepted_kmr_projection_updates_capability_facts_without_mutating_capture(owner_context):
    from cais_spade_llm.resources.environment_recovery import kmr_validation_evidence
    owner = owner_for(owner_context, 'KMR')
    evidence = environment_owner_evidence(owner)
    old = deepcopy(evidence)
    snapshot = environment_owner_snapshot(owner)
    snapshot['held_part'] = 'KET4_Square_4mm'
    physical = {'jids': {'KMR': owner.jid}, 'provenance': {'kind': 'projected', 'binding': {'proposal': 'accepted'},
        'contract_fingerprint': 'complete-owner-contract'},
        'snapshot': {'resources': {'KMR': {**snapshot, 'grasp_transform': [0., 0., .1, 0., 0., 0., 1.]}},
                     'parts': {'KET4_Square_4mm': {'current_pose': [0., 0., 1., 0., 0., 0., 1.]}}},
        'resource_models': {'KMR': {'configuration': {'captured_and_projected': True}}}}
    result = kmr_validation_evidence(evidence, snapshot, physical)
    assert result['attachment']['attached'] is True
    assert result['attachment']['held_part'] == snapshot['held_part']
    assert result['snapshot']['continuous_motion_configuration'] == physical['resource_models']['KMR']['configuration']
    assert result['observed_parts']['KET4_Square_4mm']['kind'] == 'projected'
    assert evidence == old
    physical['snapshot']['resources']['KMR']['held_part'] = None
    with pytest.raises(ValueError, match='public snapshot conflicts'):
        kmr_validation_evidence(evidence, snapshot, physical)
    physical['snapshot']['resources']['KMR']['held_part'] = snapshot['held_part']
    physical['snapshot']['resources']['KMR'].pop('grasp_transform')
    result = kmr_validation_evidence(evidence, snapshot, physical)
    assert 'attachment' not in result


def test_projected_kmr_identity_uses_recorded_resource_core(owner_context):
    from cais_spade_llm.resources.environment_recovery import kmr_validation_evidence

    owner = owner_for(owner_context, 'KMR')
    evidence = environment_owner_evidence(owner)
    snapshot = environment_owner_snapshot(owner)
    snapshot.pop('resource_jid')
    snapshot['resource_core'] = {'resource_jid': owner.jid}
    physical = {'jids': {'KMR': owner.jid}, 'provenance': {'kind': 'projected',
        'binding': {'proposal': 'accepted'}, 'contract_fingerprint': 'complete-owner-contract'},
        'snapshot': {'resources': {'KMR': deepcopy(snapshot)}, 'parts': {}},
        'resource_models': {'KMR': {'configuration': {'captured_and_projected': True}}}}
    old = deepcopy(evidence)
    assert kmr_validation_evidence(evidence, snapshot, physical)['snapshot']['resource_core'] == snapshot['resource_core']
    assert evidence == old
    snapshot['resource_core']['resource_jid'] = 'another@localhost'
    with pytest.raises(ValueError, match='identity differs'):
        kmr_validation_evidence(evidence, snapshot, physical)
    snapshot['resource_core']['resource_jid'] = owner.jid
    snapshot['resource_jid'] = None
    with pytest.raises(ValueError, match='identity differs'):
        kmr_validation_evidence(evidence, snapshot, physical)


def test_conveyor_motion_does_not_report_boundary_stopped_state_throughout_interval(owner_context):
    from cais_spade_llm.agents.central_controller.recovery_case_evidence import transport_belt_stopped
    task, evidence, snapshot, model, physical = transport_projection_input(owner_context, 'Conveyor')
    projected = project_environment_event(task=task, recovery_snapshot=snapshot, recovery_des_model=model,
        physical_input={'physical_evidence': physical, 'resource_evidence': evidence})
    first, during, last = projected['observations']
    assert first['state_snapshot']['Conveyor']['belt_stopped'] is True
    assert during['state_snapshot']['Conveyor']['belt_stopped'] is True
    assert transport_belt_stopped('Conveyor', first) == [True]
    assert transport_belt_stopped('Conveyor', last) == [True]
    duration = last['time']
    assert transport_belt_stopped('Conveyor', during, interval=[duration/3, 2*duration/3]) == [False]
    assert transport_belt_stopped('Conveyor', during) == [False, True]
    assert len(during['continuous_cells']) <= 16
    changed = deepcopy(during)
    changed['owner_safety_evidence']['resource_evidence']['Conveyor']['transport_motion']['speed_mps'] *= 2
    with pytest.raises(ValueError, match='validated owner interval'):
        transport_belt_stopped('Conveyor', changed)


def test_conveyor_no_transfer_inventory_is_derived_from_exact_conditional_declaration(owner_context):
    from cais_spade_llm.agents.central_controller.recovery_case_evidence import case_ap_value
    task, evidence, snapshot, model, physical = transport_projection_input(owner_context, 'Conveyor')
    projected = project_environment_event(task=task, recovery_snapshot=snapshot, recovery_des_model=model,
        physical_input={'physical_evidence': physical, 'resource_evidence': evidence})
    for observation in projected['observations']:
        record = observation['owner_safety_evidence']['resource_evidence']['Conveyor']
        assert record['handoffs'] == {'complete': True, 'records': []}
        assert case_ap_value('handoff_identity', 'Conveyor', observation) == [True]
        assert case_ap_value('handoff_acknowledged', 'Conveyor', observation) == [True]
        buffer = observation['owner_safety_evidence']['resource_evidence']['Buffer For Machined parts']
        assert buffer['handoffs'] == {'complete': True, 'records': []}
        assert case_ap_value('handoff_identity', 'Buffer For Machined parts', observation) == [True]
        assert case_ap_value('handoff_acknowledged', 'Buffer For Machined parts', observation) == [True]
    tampered = deepcopy(projected['observations'][0])
    tampered['owner_safety_evidence']['resource_evidence']['Conveyor']['handoff_declarations'][0]['event_name'] = 'invented'
    with pytest.raises(ValueError, match='recorded owner model'):
        case_ap_value('handoff_identity', 'Conveyor', tampered)


def test_conveyor_recomputes_collection_support_from_projected_snapshot_but_checks_base_model(owner_context, monkeypatch):
    from cais_spade_llm.recovery_framework.outline_validation import validate_resource_outline_payload
    from cais_spade_llm.agents.shared_information.recovery_validation_protocol import recovery_validation_fingerprint
    owner = owner_for(owner_context, 'Conveyor')
    evidence, snapshot = environment_owner_evidence(owner), environment_owner_snapshot(owner)
    model = environment_recovery_descriptor(evidence, snapshot, resource_jid=owner.jid)
    entry = {'resource_type': 'resource', 'resource_evidence': evidence, 'recovery_snapshot': snapshot,
             'recovery_des_model': model}
    saved = SavedRecoveryResource(owner.jid, entry, {'run_id': 'saved', 'jids': {'Conveyor': owner.jid}})
    projected = deepcopy(snapshot)
    part = 'KET4_Square_4mm'
    projected.update({'part_location.' + part: 'loading_position_1', 'part_order.' + part: 0})
    proposal = {'event_name': 'new_event', 'resource_jid': owner.jid, 'part_name': part, 'params': {},
               'expected_end_state': {'part_location.' + part: 'loading_position_2'}}
    seen = []
    def isolated_projection(**kwargs):
        seen.append(deepcopy(kwargs['recovery_des_model']))
        return {'status': 'needs_context', 'interval_complete': False, 'observations': []}
    monkeypatch.setattr(saved, 'project_recovery_outline_event', isolated_projection)
    payload = {'primitive_support_semantics': 'successor_primitive_union', 'candidates': [{'task': proposal,
        'physical_input': {'base_model_fingerprint': recovery_validation_fingerprint(model),
            'base_snapshot_fingerprint': recovery_validation_fingerprint(snapshot), 'projected_snapshot': projected}}]}
    response = validate_resource_outline_payload(saved, payload, offline=True)
    assert response['results'][0]['feasibility_status'] == 'FEASIBLE', response
    assert response['recovery_des_model'] == seen[0] == model
    assert response['results'][0]['successor_calculation']['snapshot_fingerprint'] == recovery_validation_fingerprint(projected)
    payload['candidates'][0]['physical_input']['base_model_fingerprint'] = 'changed'
    rejected = validate_resource_outline_payload(saved, payload, offline=True)
    assert rejected['results'][0]['feasibility_status'] == 'NEEDS_CONTEXT'
    assert len(seen) == 1


def test_conveyor_delivery_keeps_pending_custody_and_binds_projected_acknowledgement(owner_context):
    from cais_spade_llm.agents.central_controller.recovery_case_evidence import case_ap_value
    from cais_spade_llm.resources.environment_models import owners
    task, evidence, snapshot, model, physical = transport_projection_input(owner_context, 'Conveyor')
    part = task['part_name']
    for values in (snapshot, evidence['environment_valuation']['Conveyor'],
                   physical['environment_valuation']['Conveyor'], physical['snapshot']['resources']['Conveyor']):
        values['part_location.' + part] = 'output_nest'
    x = evidence['inputs']['scene']['Conveyor']['output_nest_pose'][0]
    physical['snapshot']['parts'][part]['current_pose'][0] = x
    physics = physical['physics_evidence']['models'][part]
    physics['pose'][0] = x
    physics['links']['link']['pose'][0] = x
    physics['links']['link']['center_of_mass'][0] = x
    physics['links']['link']['collisions'][0]['pose'][0] = x
    physical['stationary_part_evidence'][part]['current_pose'][0] = x
    custody = owners(evidence['environment_models'], physical['environment_valuation'], physical['products'], evidence['product_name'])
    for rid, row in physical['snapshot']['resources'].items():
        row['contained_parts'] = [name for name, holder in custody.items() if holder == rid]
    task['params'] = {'delivered_part': part, 'next_locations': {}}
    task['expected_end_state'] = {'part_location.' + part: None}
    model = environment_recovery_descriptor(evidence, snapshot, resource_jid=task['resource_jid'])
    result = project_environment_event(task=task, recovery_snapshot=snapshot, recovery_des_model=model,
        physical_input={'physical_evidence': physical, 'resource_evidence': evidence})
    assert result['status'] == 'projected', result.get('unresolved')
    assert 'owner_safety_evidence_unresolved' not in result
    for index, observation in enumerate(result['observations']):
        record = observation['owner_safety_evidence']['resource_evidence']['Conveyor']['handoffs']['records'][0]
        assert record['phase'] == ('completed' if index == 2 else 'pending')
        assert case_ap_value('handoff_identity', 'Conveyor', observation) == [True]
        assert case_ap_value('handoff_acknowledged', 'Conveyor', observation) == [True]
        buffer = observation['owner_safety_evidence']['resource_evidence']['Buffer For Machined parts']['handoffs']['records'][0]
        assert buffer['task_id'] != record['task_id']
        assert case_ap_value('handoff_identity', 'Buffer For Machined parts', observation) == [True]
        assert case_ap_value('handoff_acknowledged', 'Buffer For Machined parts', observation) == [True]
    assert result['resource_valuation_updates']['Buffer For Machined parts']['zone_1_part'] == part
    changed = deepcopy(result['observations'][-1])
    changed['owner_safety_evidence']['resource_evidence']['Conveyor']['handoffs']['records'][0]['acknowledgement']['owner_effect_fingerprint'] = 'incorrect'
    with pytest.raises(ValueError, match='exact validated owner effect'):
        case_ap_value('handoff_acknowledged', 'Conveyor', changed)
