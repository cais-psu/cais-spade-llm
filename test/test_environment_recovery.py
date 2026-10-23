from __future__ import annotations

"""Recorded owner capability preparation for all failure scenario resources."""

from copy import deepcopy
import math
from types import SimpleNamespace

import pytest

from cais_spade_llm.agents.resource_agent.resource_agent import ResourceAgent
from cais_spade_llm.product.environment import EnvironmentProductContext
from cais_spade_llm.recovery_framework import ROOT, SCENE_PATH, PRODUCT_PATH, read_json
from cais_spade_llm.recovery_framework.offline_recovery import SavedRecoveryResource
from cais_spade_llm.resources.environment_recovery import (
    environment_owner_evidence, environment_owner_snapshot, environment_recovery_descriptor,
    check_environment_primitive_capability,
)
from cais_spade_llm.resources.recovery_feasibility import derive_primitive_union


@pytest.mark.parametrize('defect', [None, 'attachment', 'transform', 'part'])
@pytest.mark.parametrize('location', ['Storage', 'M1', 'M2'])
def test_placement_target_capability_keeps_recorded_custody_constraints(owner_context, defect, location):
    from cais_spade_llm.recovery_framework.geometry import compose, quaternion
    from cais_spade_llm.recovery_framework.kmr_recovery import check_kmr_primitive_capability

    owner = owner_for(owner_context, 'KMR')
    evidence = environment_owner_evidence(owner)
    names = owner_context.inputs['scene']['KMR']['arm_joint_names']
    part = 'RGOCG4-50_Round_4mm'
    snapshot = {**environment_owner_snapshot(owner), 'held_part': part, 'joint_names': names,
        'joint_positions': [0.] * 7, 'current_pose': [0., 0., .7, 0., 0., 0., 1.],
        'continuous_motion_configuration': {'joints': [{'name': name, 'position_bounds': [-3., 3.]} for name in names]}}
    task = {'resource_jid': owner.jid}
    scene = evidence['inputs']['scene']
    destination = scene['Storage'] if location == 'Storage' else next(row for row in scene['machines'] if row['resource_id'] == location)
    dock = destination.get('KMR_pick_docking_poses', {}).get(part, destination['KMR_docking_pose'])
    yaw = dock[2] if len(dock) == 3 else dock[5]
    snapshot.update(resource_location=location, base_pose=[dock[0], dock[1], 0., *quaternion([0., 0., yaw])])
    snapshot['current_pose'][3:] = [1., 0., 0., 0.]
    transform = [0., 0., .025, 1., 0., 0., 0.]
    evidence['attachment'] = {'attached': True, 'held_part': part, 'grasp_transform': transform[:]}
    task['part_name'] = part
    if defect == 'attachment':
        evidence['attachment']['attached'] = False
    elif defect == 'transform':
        transform[2] += .01
    elif defect == 'part':
        task['part_name'] = 'KET4_Square_4mm'
    result = check_kmr_primitive_capability(primitive='compute_place_targets', params={'transform': transform},
        task=task, recovery_snapshot=snapshot, resource_evidence=evidence)
    assert result['feasibility_status'] == ('NEEDS_CONTEXT' if defect == 'attachment'
        else 'INFEASIBLE' if defect else 'FEASIBLE'), result
    if defect is None:
        actual = result['evidence']['calculated_targets']
        fixture = destination['slots'][part] if location == 'Storage' else destination['workholding_pose']
        placed = compose(actual['target'], transform)
        assert placed == pytest.approx([*fixture[:3], *quaternion(fixture[3:])])



def test_an_empty_belt_keeps_its_movement_and_each_resource_keeps_the_shared_events_it_takes_part_in(owner_context):
    def descriptor(rid):
        owner = owner_for(owner_context, rid)
        return environment_recovery_descriptor(environment_owner_evidence(owner), environment_owner_snapshot(owner),
                                               resource_jid=owner.jid)

    conveyor = descriptor('Conveyor')
    # The belt's contents change during recovery, so its movement is bound against the projected valuation later.
    advances = [event for event in conveyor['events'] if event['event_name'] == 'advance_conveyor']
    assert advances and conveyor['model_complete']
    assert all(event['collection_binding'] == 'projected_valuation' and 'equals' not in event['parameter_bindings']['next_locations']
               and event['primitive_support'] for event in advances)
    # A robot's placement onto the belt is the Conveyor's shared event: its own updates, no program it runs.
    placements = [event for event in conveyor['shared_events'] if event['event_name'] == 'place_release']
    assert {event['parameter_bindings']['resource_id']['equals'] for event in placements} == {'ur5e-1', 'ur5e-2'}
    assert all('program' not in event and 'primitive_support' not in event and 'part_location.{part_name}' in event['updates']
               for event in placements)
    storage = descriptor('Storage')
    assert storage['events'] == []
    assert {event['event_name'] for event in storage['shared_events']} == {'pick_approach', 'pick_part'}
    assert all(field.startswith('inventory.') for event in storage['shared_events']
               for field in (*event['guards'], *event['updates']))

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


@pytest.mark.parametrize("rid,primitive,params", [
    ("M1", "dwell", {"duration_sec": 5.}),
    ("Conveyor", "move_relative", {"speed_mps": .01}),
    ("KMR", "move_to_named_pose", {"pose_name": "home"}),
])
def test_composition_defers_owner_programs_without_primitive_effect_contracts(owner_context, rid, primitive, params):
    from cais_spade_llm.resources.environment_recovery import environment_primitive_catalog

    owner = owner_for(owner_context, rid)
    snapshot = environment_owner_snapshot(owner)
    evidence = environment_owner_evidence(owner)
    model = environment_recovery_descriptor(evidence, snapshot, resource_jid=owner.jid)
    entry = {"resource_type": "resource", "resource_evidence": evidence, "recovery_snapshot": snapshot,
             "recovery_des_model": model, "execution_primitive_catalog": environment_primitive_catalog(evidence)}
    saved = SavedRecoveryResource(owner.jid, entry, {"jids": {rid: owner.jid}})
    task = {"outline_id": "declared_owner", "resource_jid": owner.jid, "event_name": "declared_owner",
            "part_name": None, "expected_start_state": {}, "expected_end_state": {
                "belt_stopped": True} if rid == "Conveyor" else {"resource_state": "idle"}}
    result = saved.validate_recovery_primitive_composition(task=task,
        primitive_steps=[{"primitive": primitive, "params": params}],
        prepared={"recovery_resources": {owner.jid: entry}}, recovery_snapshot=snapshot,
        part_context={}, grounding_context={})
    assert result["feasibility_status"] == "NEEDS_CONTEXT", result
    assert not result["execution_authorized"] and not result["derived_end_state"]
    if rid in {"M1", "Conveyor"}:
        assert "held_part" not in snapshot and "current_pose" not in snapshot


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


def test_a_location_change_matches_the_declared_moves_between_docks_only(owner_context):
    owner = owner_for(owner_context, 'KMR')
    evidence = environment_owner_evidence(owner)
    snapshot = environment_owner_snapshot(owner)
    model = environment_recovery_descriptor(evidence, snapshot, resource_jid=owner.jid)
    proposal = {'resource_jid': owner.jid, 'part_name': None, 'params': {},
                'expected_end_state': {'resource_location': 'M1'}}
    support = derive_primitive_union(task=proposal, recovery_des_model=model, recovery_snapshot=snapshot)
    # Travel to any other position is composed from move_base; no declared event shortcuts it.
    assert {row['event_name'] for row in support['matching_transitions']} == {'move_to_resource'}


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


@pytest.mark.parametrize('location', ['M1', 'M2'])
def test_uncalibrated_pickup_uses_recorded_owner_settings_and_exact_workholding(owner_context, location):
    from cais_spade_llm.recovery_framework.geometry import quaternion
    from cais_spade_llm.recovery_framework.kmr_recovery import pick_targets, pickup_docking_pose, transport_pose

    scene = deepcopy(owner_context.inputs['scene'])
    declaration = next(row for row in scene['machines'] if row['resource_id'] == location)
    dock, fixture = declaration['KMR_docking_pose'], declaration['workholding_pose']
    base = [*dock[:3], *quaternion(dock[3:])]
    observed = [*fixture[:3], *quaternion(fixture[3:])]
    current = transport_pose(scene, base, [*fixture[:3], 1., 0., 0., 0.])
    original = deepcopy(scene)
    targets = pick_targets(scene, 'KET4_Square_4mm', observed, current, base, location=location)
    settings = scene['KMR']['task_execution']
    assert targets['target'][:3] == [observed[0], observed[1], observed[2] + settings['grasp_height_m']]
    assert targets['target'][3:] == settings['pick_orientation_xyzw']
    assert targets['approach'][2] == targets['target'][2] + settings['approach_clearance_m']
    assert targets['seed'] == settings['ik_seed_configurations'][0]
    assert pickup_docking_pose(scene, location, 'KET4_Square_4mm') == [dock[0], dock[1], dock[5]]
    assert scene == original
    observed[0] += .1
    with pytest.raises(ValueError, match='workholding location'):
        pick_targets(scene, 'KET4_Square_4mm', observed, current, base, location=location)


@pytest.mark.parametrize('defect', ['partial_calibration', 'missing_fixture', 'missing_orientation', 'ambiguous_location'])
def test_additional_pickup_never_fills_missing_or_conflicting_location_records(owner_context, defect):
    from cais_spade_llm.recovery_framework.geometry import quaternion
    from cais_spade_llm.recovery_framework.kmr_recovery import pick_targets, transport_pose

    scene = deepcopy(owner_context.inputs['scene'])
    machine = next(row for row in scene['machines'] if row['resource_id'] == 'M1')
    dock, fixture = machine['KMR_docking_pose'], machine['workholding_pose']
    base = [*dock[:3], *quaternion(dock[3:])]
    observed = [*fixture[:3], *quaternion(fixture[3:])]
    current = transport_pose(scene, base, [*fixture[:3], 1., 0., 0., 0.])
    if defect == 'partial_calibration':
        machine['KMR_pick_orientations_xyzw'] = {'KET4_Square_4mm': [[1., 0., 0., 0.]]}
    elif defect == 'missing_fixture':
        machine.pop('workholding_pose')
    elif defect == 'missing_orientation':
        scene['KMR']['task_execution'].pop('pick_orientation_xyzw')
    else:
        scene['M1'] = deepcopy(machine)
    with pytest.raises(KeyError):
        pick_targets(scene, 'KET4_Square_4mm', observed, current, base, location='M1')


@pytest.mark.parametrize('defect', ['', 'unsupported_target', 'unverified_solver_output'])
def test_kmr_calculated_endpoint_requires_independent_fk_and_preserves_joint_limits(owner_context, monkeypatch, defect):
    from cais_spade_llm.recovery_framework import kmr_recovery
    from cais_spade_llm.resources.continuous_motion import interval, kinematic_transforms, representative_pose

    names = [f'joint_a{i}' for i in range(1, 8)]
    config = {'root': 'KMR_base_link', 'reference_link': 'tool', 'root_xyz': [0., 0., 0.],
        'root_quaternion': [0., 0., 0., 1.], 'joint_names': names, 'interpolation': 'splines',
        'joints': [{'name': name, 'parent': 'KMR_base_link' if i == 0 else 'link' + str(i),
                    'child': 'tool' if i == 6 else 'link' + str(i + 1), 'type': 'revolute', 'axis': [0., 0., 1.],
                    'xyz': [0., 0., .1], 'rpy': [0., 0., 0.], 'position_bounds': [-3., 3.]}
                   for i, name in enumerate(names)],
        'components': [{'id': 'tool', 'link': 'tool', 'bounds': [[-.02, .02]] * 3}]}
    state = {'joint_names': names, 'joint_positions': [0.] * 7, 'held_part': None,
             'current_pose': [0., 0., .7, 0., 0., 0., 1.]}
    target = [0., 0., .7, 0., 0., math.sin(.4), math.cos(.4)]
    if defect == 'unsupported_target':
        target[0] = 10.
    elif defect == 'unverified_solver_output':
        monkeypatch.setattr(kmr_recovery, '_calculate_cartesian_joints',
                            lambda *args: {'joint_positions': [0.] * 7, 'solver_status': 1})
    evidence = {'inputs': deepcopy(owner_context.inputs), 'snapshot': state,
                'continuous_motion_configuration': config}
    before = deepcopy(evidence)
    result = kmr_recovery.check_kmr_primitive_capability(primitive='move_cartesian', params={'target': target},
        task={'resource_jid': 'KMR@localhost'}, recovery_snapshot=state, resource_evidence=evidence)
    assert evidence == before
    if defect:
        assert result['feasibility_status'] == 'NEEDS_CONTEXT', result
        return
    assert result['feasibility_status'] == 'FEASIBLE', result
    record = result['evidence']['endpoint_calculation']
    assert record['execution_authorized'] is False and record['collision_or_path_validated'] is False
    joints = result['evidence']['joint_positions']
    assert all(-3. <= joint <= 3. for joint in joints)
    checked = representative_pose(kinematic_transforms(config, {
        name: interval(value) for name, value in zip(state['joint_names'], joints, strict=True)})['tool'])
    assert math.dist(checked[:3], target[:3]) < 1e-5
    assert abs(sum(a*b for a, b in zip(checked[3:], target[3:], strict=True))) > 1 - 1e-10


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


@pytest.mark.parametrize("pose,receiver", [
    ([-4.13, 1.82, 1.06], ("M1", "M1")),
    ([-4.6, 1.1, 1.0], ("M1", "M1 staging tray")),
    ([-6.25, 1.88, 0.645], ("Storage", "Storage")),
    ([-6.25, 1.98, 0.645], None),
    ([-0.56, 0.5, 1.017], ("Buffer For Machined parts", "zone_2_part")),
    ([-0.32, 0.45, 1.02], ("Buffer For Machined parts", "zone_4_part")),
    ([-3.0, 0.5, 1.015], ("Conveyor", "after loading_position_1")),
    ([-2.05, 0.52, 1.016], ("Conveyor", "loading_position_2")),
    ([-4.6, 0.5, 1.015], None),
    ([-3.0, 0.5, 1.2], None),
    ([0.25, 0.45, 1.025], None),
])
def test_released_part_receiver_comes_from_its_settled_position(pose, receiver):
    from cais_spade_llm.resources.environment_recovery import observed_receiver
    from cais_spade_llm.resources.nominal_des import build_nominal_resource_des_models

    scene = read_json(SCENE_PATH)
    conveyor = build_nominal_resource_des_models(scene)["Conveyor"]
    assert observed_receiver(scene, conveyor, "KET4_Square_4mm", pose) == receiver
    assert observed_receiver(scene, None, "KET4_Square_4mm", [-3.0, 0.5, 1.015]) is None
