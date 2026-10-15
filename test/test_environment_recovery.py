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
    check_environment_primitive_capability, project_environment_event,
)
from cais_spade_llm.resources.recovery_feasibility import derive_primitive_union


def selective_movement_case(context, *, spatial=False, carried=False):
    """Use real owner declarations with selected exact completion predicates."""
    from cais_spade_llm.recovery_framework.outline_validation import load_outline_specifications

    _, _, _, _, physical = static_machine_projection_input(context)
    owner = owner_for(context, 'KMR')
    context.resources['KMR'].valuation['resource_location'] = 'M2'
    if carried:
        part = 'RGOCG4-50_Round_4mm'
        context.resources['Storage'].valuation['inventory.' + part] = False
        context.resources['KMR'].valuation.update(held_part=part, resource_state='carrying')
        context.part_tracker[part].update(location='KMR', state='in_gripper')
    evidence = environment_owner_evidence(owner)
    snapshot = environment_owner_snapshot(owner)
    snapshot.update(current_pose=[0., 0., 1., 0., 0., 0., 1.], base_pose=[0., 0., 0., 0., 0., 0., 1.])
    evidence['snapshot'] = deepcopy(snapshot)
    model = environment_recovery_descriptor(evidence, snapshot, resource_jid=owner.jid)
    physical.update(environment_models=deepcopy(evidence['environment_models']),
        environment_valuation=deepcopy(evidence['environment_valuation']), products=deepcopy(evidence['environment_products']))
    physical['snapshot']['resources'] = deepcopy(evidence['environment_valuation'])
    physical['snapshot']['resources']['KMR'].update(deepcopy(snapshot))
    physical['resources']['KMR'] = {'model_static': False, 'idle': True}
    physical['resource_models']['KMR'] = {'configuration': {'captured': True}}
    physical['geometry']['resources']['KMR']['stationary_only'] = False
    for name, values in evidence['environment_products'].items():
        physical['snapshot']['parts'][name].update(deepcopy(values), processCompleted_complete=True,
            processCompleted_evidence={'source_kind': 'recorded', 'complete': True, 'checkpoint': 'fixture'})
    if spatial:
        for name in ('front_access', 'side_access', 'workholding'):
            physical['geometry']['regions']['M2.' + name] = {'bounds': [[4, 5], [4, 5], [4, 5]]}
        for name, value in physical['snapshot']['resources'].items():
            if 'held_part' not in value:
                continue
            physical['geometry']['resources'][name]['stationary_only'] = False
    names = ['safety_KET4_Square_4mm_before_RGOCG4-50_Round_4mm_processing.txt']
    if spatial:
        names.append('safety_M2_mutex.txt')
    specifications = load_outline_specifications(['cais_spade_llm/specification/safety/' + name for name in names], ROOT)
    task = {'outline_id': 'selective-test', 'event_name': 'arbitrary_proposed_identifier',
        'resource_jid': owner.jid, 'part_name': None,
        'rationale': 'Use the declared owner successor.',
        'expected_end_state': {'resource_location': 'M1', 'resource_state': snapshot['resource_state']}}
    inputs = {'physical_evidence': physical, 'resource_evidence': evidence, 'projected_snapshot': snapshot,
              'outline_safety_specifications': specifications}
    return task, snapshot, model, inputs


def selected_projection(case):
    from cais_spade_llm.recovery_framework.outline_evidence_policy import (
        event_evidence_requirements, project_selected_successor, evidence_coverage)

    task, snapshot, model, inputs = case
    requirements = event_evidence_requirements(task=task, snapshot=snapshot, model=model,
        physical_input=inputs, specifications=inputs['outline_safety_specifications'])
    projection = project_selected_successor(task=task, snapshot=snapshot, model=model,
        physical_input=inputs, requirements=requirements)
    if projection is not None:
        projection.update(outline_evidence_policy_version=1, evidence_requirements=requirements)
        projection['evidence_coverage'] = evidence_coverage(requirements=requirements, projection=projection)
    return requirements, projection


def test_selected_processing_movement_uses_owner_effects_without_motion_preview(owner_context, monkeypatch):
    from cais_spade_llm.recovery_framework.outline_evidence_policy import check_evidence_policy
    from cais_spade_llm.recovery_framework.outline_validation import validate_resource_outline_payload, validate_safety_outline_payload

    case = selective_movement_case(owner_context)
    task, snapshot, model, inputs = case
    original = deepcopy(case)
    owner = SimpleNamespace(jid=task['resource_jid'], get_recovery_snapshot=lambda: snapshot,
        recovery_des_model=lambda **_: model,
        validate_recovery_primitive_support=lambda **kw: {'allowed': True, 'feasibility_status': 'FEASIBLE',
            'primitive_support': derive_primitive_union(task=kw['task'], recovery_des_model=kw['recovery_des_model'],
                                                       recovery_snapshot=kw['recovery_snapshot'])},
        project_recovery_outline_event=lambda **_: pytest.fail('Unnecessary geometry projection'))
    monkeypatch.setattr('cais_spade_llm.recovery_framework.kmr_outline_evidence.project_kmr_event',
                        lambda **_: pytest.fail('Unnecessary native geometry path'))
    payload = {'validation_contract_version': 2, 'primitive_support_semantics': 'successor_primitive_union',
        'outline_evidence_policy_version': 1,
        'candidates': [{'task': task, 'physical_input': inputs}]}
    ra = validate_resource_outline_payload(owner, payload)['results'][0]
    assert ra['allowed'], ra
    projection = ra['geometry_evidence']
    assert projection['status'] == 'projected', projection.get('unresolved')
    assert projection['interval_complete'] is False
    assert projection['elapsed_time']['seconds'] is None
    assert projection['projected_resource_snapshot']['resource_location'] == 'M1'
    assert 'current_pose' not in projection['projected_resource_snapshot']
    assert projection['successor_evidence']['invalidated_evidence_history']
    assert check_evidence_policy(task=task, snapshot=snapshot, model=model, physical_input=inputs,
        specifications=inputs['outline_safety_specifications'], projection=projection)['status'] == 'covered'
    cca = validate_safety_outline_payload({'outline_evidence_policy_version': 1,
        'candidates': [{'safety_input': {'projection': projection,
            'task': task, 'candidate_fingerprint': ra['candidate_fingerprint'],
            'specifications': inputs['outline_safety_specifications'], 'monitor_context': {},
            'evidence_input': {'task': task, 'snapshot': snapshot, 'model': model, 'physical_input': inputs}}}]})['results'][0]
    assert cca['is_safe'], cca.get('reason')
    assert cca['monitor_context_after']
    assert case == original


def test_selected_processing_does_not_freeze_relevant_running_job(owner_context):
    case = selective_movement_case(owner_context)
    case[3]['physical_evidence']['processing_observations'] = {'M2': {'active_job': 'running',
        'jobs': {'running': {'job_state': 'running', 'part_name': 'RGOCG4-50_Round_4mm'}}}}
    required, projection = selected_projection(case)
    assert required['interacting_jobs'][0]['relevant'] is True
    assert projection['status'] == 'needs_context'
    assert projection['evidence_coverage']['status'] == 'missing'
    assert 'progression' in projection['unresolved'][0]


def test_selected_mixed_mutex_requires_motion_even_for_clear_destination(owner_context):
    case = selective_movement_case(owner_context, spatial=True)
    required, projection = selected_projection(case)
    assert any(row['kind'] == 'resource_region' for row in required['aps'])
    assert not required['conservative'], required['unresolved']
    assert projection is None


@pytest.fixture
def forbid_validation_trajectories(monkeypatch):
    """Fail at generation, sampling, and replay boundaries of the old path."""
    def forbidden(*args, **kwargs):
        pytest.fail('Endpoint policy invoked full trajectory generation, sampling, or replay')

    for path in (
        'cais_spade_llm.recovery_framework.kmr_outline_evidence.project_kmr_event',
        'cais_spade_llm.recovery_framework.kmr_outline_evidence._apply_motion',
        'cais_spade_llm.recovery_framework.kmr_outline_evidence._motion_request',
        'cais_spade_llm.recovery_framework.recovery_motion_preview.preview_recovery_motion_from_files',
        'cais_spade_llm.recovery_framework.recovery_motion_preview.read_cartesian_route',
        'cais_spade_llm.resources.continuous_motion.ContinuousMotion.__init__',
        'cais_spade_llm.resources.recovery_geometry._motion_checkpoint',
        'cais_spade_llm.resources.resource_safety_preparation.prepare_recovery_outline_preview',
    ):
        monkeypatch.setattr(path, forbidden)


def bounded_movement_case(context, *, carried=False):
    """Combine real owner declarations with explicit controller and geometry fixtures."""
    from test_kmr_outline_evidence import motion_fixture
    from test_recovery_case_safety import _base_hold_input
    from test_recovery_outline_evidence import _commanded_hold_case

    task, snapshot, _, inputs = selective_movement_case(context, spatial=True, carried=carried)
    physical, evidence = inputs['physical_evidence'], inputs['resource_evidence']
    motion, _, _ = motion_fixture()
    _, base, _, _ = _base_hold_input()
    _, _, _, peer_model, peer_hold = _commanded_hold_case()
    configuration = motion['resource_models']['KMR']['configuration']
    configuration['root'] = configuration['joints'][0]['parent'] = 'base'
    configuration['description_sha256'] = base['resource_models']['KMR']['configuration']['description_sha256']
    snapshot.update({key: deepcopy(value) for key, value in motion['snapshot']['resources']['KMR'].items()
                     if key not in {'held_part'}})
    snapshot['base_pose'] = [0., 0., 0., 0., 0., 0., 1.]
    if carried:
        snapshot['grasp_transform'] = [0., 0., .2, 0., 0., 0., 1.]
        evidence['attachment'] = {'attached': True, 'held_part': snapshot['held_part'],
                                  'grasp_transform': snapshot['grasp_transform']}
    else:
        evidence['attachment'] = {'attached': False}
    physical['snapshot']['resources']['KMR'] = deepcopy(snapshot)
    physical['resource_models']['KMR'] = {'configuration': configuration}
    physical['resources']['KMR'] = deepcopy(peer_hold)
    physical['resources']['KMR'].update(model_static=False, observation_state=deepcopy(snapshot),
        base_hold_evidence=base['resources']['KMR']['base_hold_evidence'])
    physical['resources']['KMR']['controller_hold_observation'].update(
        joint_names=snapshot['joint_names'], commanded_positions=snapshot['joint_positions'])
    for rid in physical['snapshot']['resources']:
        if not rid.startswith('ur5e-'):
            continue
        row = physical['snapshot']['resources'][rid]
        row.update(joint_names=['slide'], joint_positions=[.005])
        physical['resource_models'][rid] = deepcopy(peer_model)
        physical['resources'][rid] = deepcopy(peer_hold)
        physical['resources'][rid].update(model_static=False, observation_state=deepcopy(row))
    evidence.update(snapshot=deepcopy(snapshot), continuous_motion_configuration=deepcopy(configuration))
    physical['endpoint_evidence_policy_version'] = 2
    model = environment_recovery_descriptor(evidence, snapshot, resource_jid=task['resource_jid'])
    entry = {'resource_type': 'kmr', 'resource_evidence': evidence,
             'recovery_snapshot': snapshot, 'recovery_des_model': model}
    owner = SavedRecoveryResource(task['resource_jid'], entry,
        {'jids': physical['jids'], 'resource_evidence': {task['resource_jid']: evidence}})
    inputs['resource_evidence'] = owner.get_recovery_resource_evidence()
    return owner, task, snapshot, owner.recovery_des_model(snapshot=snapshot), inputs


def bounded_validation(case, *, session=None):
    from cais_spade_llm.recovery_framework.outline_validation import validate_resource_outline_payload, validate_safety_outline_payload

    owner, task, snapshot, model, inputs = case
    ra = validate_resource_outline_payload(owner, {'validation_contract_version': 2,
        'primitive_support_semantics': 'successor_primitive_union', 'outline_evidence_policy_version': 1,
        'candidates': [{'task': task, 'physical_input': inputs}]})['results'][0]
    assert ra['allowed'], ra.get('reason')
    cca = validate_safety_outline_payload({'outline_evidence_policy_version': 1,
        'candidates': [{'safety_input': {'projection': ra['geometry_evidence'],
            'task': task, 'candidate_fingerprint': ra['candidate_fingerprint'],
            'specifications': inputs['outline_safety_specifications'],
            'monitor_context': (session or {}).get('outline_monitor_context', {}),
            'evidence_input': {'task': task, 'snapshot': snapshot, 'model': model, 'physical_input': inputs}}}]})['results'][0]
    return ra, cca


def bounded_storage_case(context, *, gripper_state='open', change_endpoint=False):
    """Supply a downward arm whose recorded home and transport endpoints agree."""
    context.inputs['scene']['KMR']['task_execution']['cartesian_waypoints'][
        'transport_tcp_position_in_base_frame_m'] = [0., 0., .7]
    if change_endpoint:
        context.inputs['scene']['KMR']['task_execution']['cartesian_motion_only'] = False
    _, task, snapshot, _, inputs = bounded_movement_case(context)
    task['expected_end_state']['resource_location'] = 'Storage'
    physical, evidence = inputs['physical_evidence'], inputs['resource_evidence']
    configuration = physical['resource_models']['KMR']['configuration']
    configuration['joints'][0].update(xyz=[0., 0., 1.3], rpy=[math.pi, 0., 0.])
    snapshot.update(current_pose=[0., 0., .7, 1., 0., 0., 0.], gripper_state=gripper_state)
    physical['snapshot']['resources']['KMR'] = deepcopy(snapshot)
    evidence.update(snapshot=deepcopy(snapshot), continuous_motion_configuration=deepcopy(configuration))
    if change_endpoint:
        event = next(row for row in evidence['environment_models']['KMR']['events']
                     if row['event_name'] == 'move_to_location')
        event['program']['steps'].append({'id': 'joint_target', 'op': 'move_joints',
                                         'params': {'joints': [0.]*6 + [.2]}})
        physical['environment_models'] = deepcopy(evidence['environment_models'])
    model = environment_recovery_descriptor(evidence, snapshot, resource_jid=task['resource_jid'])
    entry = {'resource_type': 'kmr', 'resource_evidence': evidence,
             'recovery_snapshot': snapshot, 'recovery_des_model': model}
    owner = SavedRecoveryResource(task['resource_jid'], entry,
        {'jids': physical['jids'], 'resource_evidence': {task['resource_jid']: evidence}})
    inputs['resource_evidence'] = owner.get_recovery_resource_evidence()
    return owner, task, snapshot, model, inputs


def test_bounded_kmr_matching_declarations_retain_both_proofs(owner_context, forbid_validation_trajectories, monkeypatch):
    from cais_spade_llm.resources import environment_recovery

    case = bounded_storage_case(owner_context)
    original = deepcopy(case[1:])
    ra, cca = bounded_validation(case)
    projection = ra['geometry_evidence']
    assert cca['is_safe'], cca.get('reason')
    assert projection['status'] == 'projected'
    assert projection['owner_declaration_comparison']['status'] == 'equivalent'
    assert {row['owner_successor']['task']['event_name'] for row in projection['owner_alternatives']} == {
        'move_to_resource', 'move_to_location'}
    assert all(row['status'] == 'projected' for row in projection['owner_alternatives'])
    assert {len(row['endpoint_evidence']['steps']) for row in projection['owner_alternatives']} == {1, 2}
    assert projection['projected_resource_snapshot']['resource_location'] == 'Storage'
    assert case[1:] == original
    declared = environment_recovery.declared_environment_successors

    def reverse(**kwargs):
        rows, failures = declared(**kwargs)
        return list(reversed(rows)), list(reversed(failures))

    monkeypatch.setattr(environment_recovery, 'declared_environment_successors', reverse)
    again_ra, again_cca = bounded_validation(case)
    assert again_ra['geometry_evidence'] == projection
    assert again_cca == cca


@pytest.mark.parametrize('case', ['guard', 'missing', 'search_limit', 'incomplete_model'])
def test_bounded_owner_guard_conflict_is_separate_from_missing_context_and_safety(owner_context, monkeypatch, case):
    from cais_spade_llm.recovery_framework.outline_endpoint_evidence import project_bounded_successor
    from cais_spade_llm.recovery_framework.outline_validation import validate_safety_outline_payload
    from cais_spade_llm.resources import environment_recovery

    _, task, snapshot, model, inputs = bounded_storage_case(owner_context)
    task['expected_end_state'].pop('resource_state', None)
    evidence = inputs['resource_evidence']
    for event in evidence['environment_models']['KMR']['events']:
        if event['event_name'] in {'move_to_location', 'move_to_resource'}:
            event['guards']['resource_state'] = {'equals': 'positioned'}
    if case == 'missing':
        snapshot.pop('resource_state')
        inputs['physical_evidence']['environment_valuation']['KMR'].pop('resource_state')
    elif case in {'search_limit', 'incomplete_model'}:
        derive = environment_recovery.derive_primitive_union

        def incomplete(**kwargs):
            result = derive(**kwargs)
            if case == 'search_limit':
                result['search']['limit_reached'] = True
            else:
                result['model_complete'] = False
            return result

        monkeypatch.setattr(environment_recovery, 'derive_primitive_union', incomplete)
    model = environment_recovery_descriptor(evidence, snapshot, resource_jid=task['resource_jid'])
    original = deepcopy(inputs)
    projection = project_bounded_successor(task=task, snapshot=snapshot, model=model,
        physical_input=inputs, requirements={})
    cca = validate_safety_outline_payload({'candidates': [{'safety_input': {
        'projection': projection, 'specifications': inputs['outline_safety_specifications']}}]})['results'][0]
    if case != 'guard':
        assert projection['status'] == 'needs_context'
        assert not projection.get('physical_capability_rejected')
    else:
        assert projection['status'] == 'rejected', projection.get('unresolved')
        assert projection['physical_capability_rejected'] is True
        assert any(row.get('field') == 'resource_state' and row['expected'] == 'positioned'
                   and row['actual'] == 'idle' for row in projection['capability_result']['guard_failures'])
        assert cca['formal_safety_status'] == 'not_checked'
        assert cca['checks'] == []
    assert inputs == original


@pytest.mark.parametrize('nominal', [False, True])
def test_recovery_projection_defers_nominal_predecessor_guards_only(owner_context, nominal):
    from cais_spade_llm.resources.environment_recovery import declared_environment_successors

    _, task, snapshot, _, inputs = bounded_storage_case(owner_context)
    inputs['outline_evidence_policy_version'] = 2
    task['expected_end_state'].pop('resource_state', None)
    evidence = inputs['resource_evidence']
    for event in evidence['environment_models']['KMR']['events']:
        if event['event_name'] in {'move_to_location', 'move_to_resource'}:
            event['guards']['resource_state'] = {'equals': 'positioned'}
    model = environment_recovery_descriptor(evidence, snapshot, resource_jid=task['resource_jid'])
    if nominal:
        task['required_task_contract'] = 'move_to_location'
    before = deepcopy(inputs)
    candidates, failures = declared_environment_successors(task=task, recovery_snapshot=snapshot,
        recovery_des_model=model, physical_input=inputs)
    if nominal:
        assert not candidates and failures
    else:
        assert candidates and not failures
        assert all(row['composition_prerequisites'] for row in candidates)
        assert all(any(local['guards'].get('resource_state') == {'equals': 'positioned'}
                       for local in row['composition_prerequisites']) for row in candidates)
    assert inputs == before


@pytest.mark.parametrize('peer', ['Storage', 'M1', 'M2'])
@pytest.mark.parametrize('case', ['static', 'custody', 'dynamic', 'body_effect', 'other_actor'])
def test_shared_passive_participation_preserves_only_unchanged_static_body_geometry(owner_context, peer, case):
    from cais_spade_llm.recovery_framework.outline_evidence_policy import _movement_effects
    from cais_spade_llm.resources.environment_recovery import declared_environment_successors

    _, task, snapshot, model, inputs = bounded_movement_case(owner_context)
    candidates, _ = declared_environment_successors(task=task, recovery_snapshot=snapshot,
        recovery_des_model=model, physical_input=inputs)
    declared = candidates[0]
    physical = inputs['physical_evidence']
    event = deepcopy(next(row for row in inputs['resource_evidence']['environment_models']['KMR']['events']
        if row['event_id'] == declared['task']['event_id']))
    event['participants'].append(peer)
    shared = deepcopy(event)
    shared['updates'] = {'current_pose': {'set': [1., 0., 0., 0., 0., 0., 1.]}} if case == 'body_effect' else {}
    if case == 'custody':
        shared['updates'] = ({'inventory.RGOCG4-50_Round_4mm': {'set': False}} if peer == 'Storage'
            else {'resource_state': {'set': 'idle'}, 'part_name': {'set': None}})
    if case == 'other_actor':
        shared['parameter_bindings']['resource_id'] = {'equals': peer}
    physical['environment_models'][peer]['events'].append(shared)
    physical['resources'][peer]['model_static'] = case != 'dynamic'
    physical['snapshot']['resources'][peer]['current_pose'] = [0., 0., 0., 0., 0., 0., 1.]
    original = deepcopy(physical)
    result = _movement_effects(task=task, snapshot=snapshot, model=model, physical_input=inputs,
        requirements={'aps': [], 'interacting_jobs': []}, declared=declared, event=event)
    assert result is not None
    successor = result['successor_evidence']
    if case in {'static', 'custody'}:
        assert successor['geometry']['resources'][peer] == physical['geometry']['resources'][peer]
        assert successor['snapshot']['resources'][peer]['current_pose'] == physical['snapshot']['resources'][peer]['current_pose']
    else:
        assert 'current_pose' not in successor['snapshot']['resources'][peer]
    assert physical == original


@pytest.mark.parametrize('change', ['different_endpoint', 'missing_gripper', 'closed_gripper', 'old_policy'])
def test_bounded_kmr_does_not_select_one_favorable_declaration(owner_context, forbid_validation_trajectories, change):
    case = bounded_storage_case(owner_context, change_endpoint=change == 'different_endpoint',
        gripper_state={'missing_gripper': None, 'closed_gripper': 'closed'}.get(change, 'open'))
    if change == 'old_policy':
        case[4]['physical_evidence']['endpoint_evidence_policy_version'] = 1
    original = deepcopy(case[1:])
    ra, cca = bounded_validation(case)
    projection = ra['geometry_evidence']
    assert projection['status'] == 'needs_context'
    assert not cca['is_safe']
    assert cca['formal_safety_status'] == 'not_checked'
    assert projection['observations'] == []
    assert 'successor_evidence' not in projection
    assert cca['monitor_context_before'] == cca['monitor_context_after'] == {}
    assert case[1:] == original
    if change == 'different_endpoint':
        assert 'different endpoints, effects, or predicate evidence' in projection['unresolved'][0]
        assert all(row['status'] == 'projected' for row in projection['owner_alternatives'])
    elif change == 'old_policy':
        assert 'requires one supported owner declaration' in projection['unresolved'][0]
    else:
        assert 'additional endpoint evidence' in projection['unresolved'][0]
        assert {row['status'] for row in projection['owner_alternatives']} == {'projected', 'needs_context'}


@pytest.mark.parametrize('change', ['endpoint', 'declaration', 'omitted_alternative'])
def test_bounded_kmr_cca_recalculates_every_declaration(owner_context, forbid_validation_trajectories, change):
    from cais_spade_llm.recovery_framework.outline_evidence_policy import check_evidence_policy

    _, task, snapshot, model, inputs = case = bounded_storage_case(owner_context)
    ra, cca = bounded_validation(case)
    assert cca['is_safe'], cca.get('reason')
    projection = deepcopy(ra['geometry_evidence'])
    if change == 'endpoint':
        projection['owner_alternatives'][0]['endpoint_evidence']['resource_snapshot']['base_pose'][0] += 1.
    elif change == 'declaration':
        projection['owner_alternatives'][0]['owner_successor']['task']['event_id'] += 1
    else:
        projection['owner_alternatives'].pop()
    with pytest.raises(ValueError, match='independently reproduced'):
        check_evidence_policy(task=task, snapshot=snapshot, model=model, physical_input=inputs,
            specifications=inputs['outline_safety_specifications'], projection=projection)


def test_bounded_kmr_nominal_contract_keeps_its_exact_declaration(owner_context, forbid_validation_trajectories):
    case = bounded_storage_case(owner_context)
    case[1]['required_task_contract'] = 'move_to_location'
    ra, cca = bounded_validation(case)
    projection = ra['geometry_evidence']
    assert cca['is_safe'], cca.get('reason')
    assert projection['owner_successor']['task']['event_name'] == 'move_to_location'
    assert 'owner_alternatives' not in projection


@pytest.mark.parametrize('carried', [False, True])
@pytest.mark.parametrize('base_ready', [False, True])
def test_bounded_kmr_real_ra_cca_and_next_successor(owner_context, forbid_validation_trajectories, carried, base_ready):
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes.multi_turn_validated_event import _project
    from cais_spade_llm.recovery_framework.geometry import compose

    case = bounded_movement_case(owner_context, carried=carried)
    owner, task, snapshot, model, inputs = case
    if not base_ready:
        inputs['physical_evidence']['resources']['KMR'].pop('base_hold_evidence')
    for rid, row in inputs['physical_evidence']['geometry']['resources'].items():
        if 'held_part' in inputs['resource_evidence']['environment_models'][rid]['state_variables']:
            row.pop('stationary_only', None)
    for field in ('resource_jid', 'resource_core'):
        inputs['physical_evidence']['snapshot']['resources']['KMR'].pop(field, None)
    inputs['physical_evidence']['snapshot']['resources']['KMR']['observed_pose'] = deepcopy(snapshot['current_pose'])
    inputs['physical_evidence']['geometry']['resources']['KMR']['current_pose'] = deepcopy(snapshot['current_pose'])
    original = deepcopy(inputs)
    ra, cca = bounded_validation(case)
    projection = ra['geometry_evidence']
    assert projection['status'] == 'projected', projection.get('unresolved')
    assert cca['is_safe'], cca.get('reason')
    assert projection['interval_evidence']['unknown_resources'] == ['KMR']
    assert all(values['KMR'] == [False, True]
               for values in projection['observations'][1]['occupancy_possibilities'].values())
    assert projection['endpoint_evidence']['status'] == 'supported'
    assert projection['readiness_evidence']['status'] == ('supported' if base_ready else 'missing')
    assert projection['readiness_evidence']['blocks_planning'] is False
    assert not projection['execution_authorized'] and not projection['command_sent']
    assert 'observed_pose' not in projection['endpoint_evidence']['resource_snapshot']
    assert 'current_pose' not in projection['successor_evidence']['geometry']['resources']['KMR']
    assert inputs == original
    session = {'outline_evidence_policy_version': 1, 'projected_physical_evidence': inputs['physical_evidence']}
    _project(task, ra, cca, session)
    next_snapshot = session['derived_resource_snapshots'][task['resource_jid']]
    assert next_snapshot['resource_jid'] == task['resource_jid']
    assert next_snapshot['resource_location'] == 'M1'
    assert next_snapshot['base_pose'] != snapshot['base_pose']
    assert next_snapshot['joint_positions'] == snapshot['joint_positions']
    if carried:
        part = projection['successor_evidence']['snapshot']['parts'][snapshot['held_part']]
        assert part['current_pose'] == pytest.approx(compose(next_snapshot['current_pose'], snapshot['grasp_transform']))
    second = {**deepcopy(task), 'outline_id': 'second-move', 'expected_end_state': {
        'resource_location': 'M2', 'resource_state': snapshot['resource_state']}}
    next_inputs = {**deepcopy(inputs), 'physical_evidence': session['projected_physical_evidence'],
                   'projected_snapshot': next_snapshot}
    next_ra, next_cca = bounded_validation((owner, second, next_snapshot, model, next_inputs), session=session)
    assert next_ra['geometry_evidence']['status'] == 'projected', next_ra['geometry_evidence']['unresolved']
    assert next_cca['is_safe'], next_cca.get('reason')
    assert next_ra['geometry_evidence']['readiness_evidence']['status'] == 'missing'
    assert next_ra['geometry_evidence']['readiness_evidence']['kind'] == 'conditional_successor'
    assert not next_ra['geometry_evidence']['execution_authorized']
    end = next_ra['geometry_evidence']['endpoint_evidence']['resource_snapshot']
    dock = next(row['KMR_docking_pose'] for row in inputs['resource_evidence']['inputs']['scene']['machines']
                if row['resource_id'] == 'M2')
    assert end['base_pose'][:2] == pytest.approx(dock[:2])
    assert next_ra['geometry_evidence']['observations'][0]['resources']['KMR']['base_pose'] == next_snapshot['base_pose']


@pytest.mark.parametrize('policy_version', [1, 2])
@pytest.mark.parametrize('base_status', ['missing', 'active', 'supported'])
def test_bounded_kmr_separates_execution_readiness_from_safety(owner_context, forbid_validation_trajectories,
                                                             policy_version, base_status):
    case = bounded_movement_case(owner_context)
    physical = case[4]['physical_evidence']
    physical['endpoint_evidence_policy_version'] = policy_version
    if base_status == 'missing':
        physical['resources']['KMR'].pop('base_hold_evidence')
    elif base_status == 'active':
        physical['resources']['KMR']['base_hold_evidence']['controller_observation']['has_active_goal'] = True
    original = deepcopy(case[4])
    ra, cca = bounded_validation(case)
    projection = ra['geometry_evidence']
    assert projection['endpoint_evidence_policy_version'] == policy_version
    assert projection['endpoint_evidence']['status'] == 'supported'
    assert projection['readiness_evidence']['status'] == ('supported' if base_status == 'supported' else 'missing')
    assert cca['is_safe'] is (policy_version == 2 or base_status == 'supported')
    assert not ra['execution_authorized'] and not cca['execution_authorized']
    assert not projection['execution_authorized'] and not projection['command_sent']
    assert case[4] == original
    if policy_version == 2:
        assert projection['status'] == 'projected'
        assert projection['evidence_coverage']['status'] == 'covered'
        assert projection['readiness_evidence']['scope'] == 'execution'
        assert projection['readiness_evidence']['blocks_planning'] is False
        assert projection['unresolved'] == []
    elif base_status != 'supported':
        assert projection['status'] == 'needs_context'
        assert cca['monitor_context_after'] == cca['monitor_context_before'] == {}


@pytest.mark.parametrize('missing', ['peer_hold', 'overlapping_peer', 'running_job', 'unsupported'])
def test_bounded_kmr_gaps_retain_endpoint_and_never_request_trajectory(owner_context, forbid_validation_trajectories, missing):
    case = bounded_movement_case(owner_context)
    physical = case[4]['physical_evidence']
    physical['resources']['KMR'].pop('base_hold_evidence')
    if missing == 'peer_hold':
        physical['resources']['ur5e-4'].pop('stationary_contract')
    elif missing == 'overlapping_peer':
        physical['resource_models']['ur5e-4']['configuration']['root_xyz'] = [4.5, 4.5, 4.5]
    elif missing == 'running_job':
        physical['processing_observations'] = {'M2': {'jobs': {'running': {
            'job_state': 'running', 'part_name': 'RGOCG4-50_Round_4mm'}}}}
    else:
        from cais_spade_llm.recovery_framework.outline_validation import load_outline_specifications
        case[4]['outline_safety_specifications'] += load_outline_specifications([
            'cais_spade_llm/specification/safety/safety_assembly_board-v1_gear_small_before_KET4_Square_4mm.txt'], ROOT)
    original = deepcopy(case[4])
    ra, cca = bounded_validation(case)
    assert ra['geometry_evidence']['endpoint_evidence']['status'] == 'supported'
    assert not cca['is_safe']
    assert cca['monitor_context_after'] == cca['monitor_context_before'] == {}
    assert ra['geometry_evidence']['requires_projection'] is False
    assert case[4] == original


def test_bounded_kmr_rejects_conflicting_endpoint_identity(owner_context, forbid_validation_trajectories):
    case = bounded_movement_case(owner_context)
    case[4]['physical_evidence']['snapshot']['resources']['KMR']['resource_jid'] = 'another-owner@localhost'
    ra, cca = bounded_validation(case)
    assert ra['geometry_evidence']['status'] == 'needs_context'
    assert 'Endpoint owner identity' in ra['geometry_evidence']['unresolved'][0]
    assert not cca['is_safe']
    assert cca['monitor_context_after'] == cca['monitor_context_before'] == {}


@pytest.mark.parametrize('field,value', [('part_state', 'held'), ('part_location', 'KMR@localhost')])
def test_bounded_owner_effect_conflicts_precede_endpoint_calculation(owner_context, monkeypatch, field, value):
    from cais_spade_llm.recovery_framework import outline_endpoint_evidence

    case = bounded_movement_case(owner_context)
    task = case[1]
    task['part_name'] = 'KET4_Square_4mm'
    task['expected_end_state'][field] = value
    before = deepcopy(case[4])
    monkeypatch.setattr(outline_endpoint_evidence, '_kmr_endpoint',
        lambda **kwargs: pytest.fail('Conflicting public effects must be checked before endpoint calculation'))
    ra, cca = bounded_validation(case)
    projection = ra['geometry_evidence']
    assert projection['status'] == 'rejected' and projection['owner_effect_rejected']
    assert any(row['field'] == field and row['requested'] == value
               for row in projection['owner_effect_result']['conflicts'])
    assert cca['formal_safety_status'] == 'not_checked'
    assert not cca['is_safe'] and cca['checks'] == []
    assert cca['monitor_context_before'] == cca['monitor_context_after'] == {}
    assert case[4] == before


@pytest.mark.parametrize('change', ['endpoint', 'coverage', 'interval', 'binding', 'region', 'cache', 'readiness', 'execution', 'policy', 'participants'])
def test_bounded_kmr_cca_recalculates_changed_evidence(owner_context, forbid_validation_trajectories, change):
    from cais_spade_llm.recovery_framework.outline_evidence_policy import check_evidence_policy

    _, task, snapshot, model, inputs = case = bounded_movement_case(owner_context)
    ra, cca = bounded_validation(case)
    assert cca['is_safe'], cca.get('reason')
    projection = ra['geometry_evidence']
    if change == 'endpoint':
        projection['endpoint_evidence']['resource_snapshot']['base_pose'][0] += 1
    elif change == 'coverage':
        projection['evidence_coverage']['aps'][0]['status'] = 'missing'
    elif change == 'interval':
        projection['observations'][1]['occupancy_possibilities']['M2.front_access']['KMR'] = [False]
    elif change == 'binding':
        inputs['outline_safety_specifications'][0]['document']['requirement_scopes'][0]['physical_ap_bindings']['ap001']['result'] = 'square'
    elif change == 'region':
        inputs['physical_evidence']['geometry']['regions']['M2.front_access']['bounds'][0] = [0., 10.]
    elif change == 'cache':
        inputs['physical_evidence']['endpoint_occupancy'] = {'binding': 'stale'}
    elif change == 'readiness':
        projection['readiness_evidence']['execution_authorized'] = True
    elif change == 'execution':
        projection['execution_authorized'] = True
    elif change == 'participants':
        projection['owner_safety_evidence']['resource_models']['KMR']['state_variables'].pop('held_part')
    else:
        inputs['physical_evidence']['endpoint_evidence_policy_version'] = 1
    with pytest.raises(ValueError, match='requirements|coverage|reproduced'):
        check_evidence_policy(task=task, snapshot=snapshot, model=model, physical_input=inputs,
            specifications=inputs['outline_safety_specifications'], projection=projection)


def test_bounded_coordinates_need_no_named_dock(owner_context, forbid_validation_trajectories):
    import math

    case = bounded_movement_case(owner_context, carried=True)
    case[1]['params'] = {'x': 1.4, 'y': -2.8, 'yaw': .3, 'observed_resource_location': None}
    case[1]['expected_end_state']['resource_location'] = None
    ra, cca = bounded_validation(case)
    assert cca['is_safe'], cca.get('reason')
    endpoint = ra['geometry_evidence']['endpoint_evidence']['resource_snapshot']
    assert endpoint['base_pose'] == pytest.approx([1.4, -2.8, 0., 0., 0., math.sin(.15), math.cos(.15)])
    assert endpoint['held_part'] == case[2]['held_part']


@pytest.mark.parametrize('carried', [False, True])
def test_additional_location_uses_one_bound_calculation_through_ra_physical_cca(owner_context, forbid_validation_trajectories, carried):
    import json
    from cais_spade_llm.recovery_framework.kmr_calculation import validate_calculation_records
    from cais_spade_llm.recovery_framework.outline_validation import validate_resource_outline_payload, validate_safety_outline_payload

    owner, task, snapshot, model, inputs = bounded_movement_case(owner_context, carried=carried)
    evidence = inputs['resource_evidence']
    evidence['inputs']['scene']['inspection table'] = {'KMR_docking_pose': [1.4, -2.8, 0., 0., 0., .3]}
    task['expected_end_state']['resource_location'] = 'inspection table'
    owner = SavedRecoveryResource(task['resource_jid'], {'resource_type': 'kmr', 'resource_evidence': evidence,
        'recovery_snapshot': snapshot, 'recovery_des_model': model}, {'jids': inputs['physical_evidence']['jids']})
    inputs['resource_evidence'] = owner.get_recovery_resource_evidence()
    original = deepcopy((model, inputs))
    response = validate_resource_outline_payload(owner, {'recovery_session_id': 'attempt-A', 'validation_contract_version': 2,
        'primitive_support_semantics': 'successor_primitive_union', 'outline_evidence_policy_version': 1,
        'candidates': [{'task': task, 'physical_input': inputs}]})
    ra = response['results'][0]
    assert ra['allowed'], ra
    assert ra['geometry_evidence']['status'] == 'projected', ra['geometry_evidence'].get('unresolved')
    calculation = ra['calculation_model']
    assert 'inspection table' in calculation['state_variables']['resource_location']['domain']
    assert calculation['state_variables']['resource_location']['domain_complete'] is True
    cca = validate_safety_outline_payload({'outline_evidence_policy_version': 1, 'candidates': [{'safety_input': {
        'task': task, 'projection': ra['geometry_evidence'], 'candidate_fingerprint': ra['candidate_fingerprint'],
        'specifications': inputs['outline_safety_specifications'], 'monitor_context': {},
        'evidence_input': {'task': task, 'snapshot': snapshot, 'model': calculation,
                           'physical_input': ra['calculation_physical_input']}}}]})['results'][0]
    assert cca['is_safe'], cca
    assert (model, inputs) == original
    assert response['recovery_des_model'] == model
    successor = json.loads(json.dumps(ra['geometry_evidence']['successor_evidence']))
    validate_calculation_records(successor, session_id='attempt-A', inputs=evidence['inputs'])
    restored = deepcopy(successor)
    restored['additional_binding_records'][0]['calculations']['move_to_location']['x'] = 99.
    with pytest.raises(ValueError, match='dependencies changed'):
        validate_calculation_records(restored, session_id='attempt-A', inputs=evidence['inputs'])
    changed_inputs = deepcopy(evidence['inputs'])
    changed_inputs['scene']['inspection table']['KMR_docking_pose'][0] = 99.
    with pytest.raises(ValueError, match='dependencies changed'):
        validate_calculation_records(successor, session_id='attempt-A', inputs=changed_inputs)
    with pytest.raises(ValueError, match='dependencies changed'):
        validate_calculation_records(successor, session_id='attempt-B', inputs=evidence['inputs'])
    successor['environment_models']['KMR']['state_variables']['held_part']['domain'] = []
    with pytest.raises(ValueError, match='model changed'):
        validate_calculation_records(successor, session_id='attempt-A', inputs=evidence['inputs'])
    assert not ra['execution_authorized'] and not cca['execution_authorized']

    # The PA must send CCA exactly the model and evidence produced by the owner.
    import asyncio
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes.multi_turn_validated_event import validate_event, _project

    async def validate_resource(*, resource_jid, payload):
        assert resource_jid == task['resource_jid']
        return validate_resource_outline_payload(owner, payload)

    async def validate_safety(*, payload):
        return validate_safety_outline_payload(payload)

    planner = SimpleNamespace(product_agent=SimpleNamespace(request_recovery_outline_physical_validation=validate_resource,
        request_recovery_outline_safety_validation=validate_safety))
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes.multi_turn import build_multi_turn_session_seed

    prepared = {'offline_recovery': True, 'recovery_session': {'session_id': 'attempt-A'},
        'primitive_support_semantics': 'successor_primitive_union', 'outline_evidence_policy_version': 1,
        'physical_evidence': deepcopy(inputs['physical_evidence']),
        'outline_safety_specifications': inputs['outline_safety_specifications'],
        'recovery_resources': {task['resource_jid']: {'recovery_des_model': model, 'recovery_snapshot': snapshot,
            'resource_evidence': inputs['resource_evidence']}}}
    session = build_multi_turn_session_seed(prepared)
    checked_ra, checked_cca = asyncio.run(validate_event(task=task, prepared=prepared, session=session, planner=planner, turn={}))
    assert checked_ra['allowed'] and checked_cca['is_safe'], checked_cca.get('reason')
    assert session['accepted_outline_prefix'] == []
    assert not session['projected_physical_evidence'].get('additional_binding_records')
    _project(task, checked_ra, checked_cca, session)
    assert session['projected_physical_evidence']['additional_binding_records']
    from cais_spade_llm.recovery_framework.offline_recovery import SavedRecoveryProduct

    product = SavedRecoveryProduct({'jids': inputs['physical_evidence']['jids']}, [owner])
    product.restore_recovery_resource_evidence(session, source_fingerprint='unchanged-source')
    retained = deepcopy(session['projected_physical_evidence'])
    next_task = {**deepcopy(task), 'outline_id': 'return-after-additional-binding',
                 'expected_end_state': {**task['expected_end_state'], 'resource_location': 'M2'}}
    reused_ra, reused_cca = asyncio.run(validate_event(task=next_task, prepared=prepared, session=session,
        planner=planner, turn={}))
    assert reused_ra['allowed'] and reused_cca['is_safe'], reused_cca.get('reason')
    audit = reused_ra['additional_binding_record']
    assert audit['task'] == next_task and audit['calculations'] == {}
    assert audit['sources'] == [{'additional_binding_fingerprint': retained['additional_binding_records'][-1]['fingerprint']}]
    assert reused_ra['calculation_physical_input']['physical_evidence']['environment_models'] == retained['environment_models']
    assert session['projected_physical_evidence'] == retained
    _project(next_task, reused_ra, reused_cca, session)
    product.restore_recovery_resource_evidence(session, source_fingerprint='unchanged-source')
    assert len(session['projected_physical_evidence']['additional_binding_records']) == 2
    session['session_id'] = 'attempt-B'
    with pytest.raises(ValueError, match='dependencies changed'):
        product.restore_recovery_resource_evidence(session, source_fingerprint='unchanged-source')


@pytest.mark.parametrize('missing', ['geometry', 'ambiguous', 'custody', 'attachment'])
def test_additional_location_missing_evidence_leaves_inputs_unchanged(owner_context, missing):
    from cais_spade_llm.recovery_framework.outline_validation import validate_resource_outline_payload

    owner, task, snapshot, model, inputs = bounded_movement_case(owner_context, carried=missing == 'attachment')
    scene = inputs['resource_evidence']['inputs']['scene']
    scene['inspection table'] = {'KMR_docking_pose': [1., 2., 0., 0., 0., 0.]}
    task['expected_end_state']['resource_location'] = 'inspection table'
    if missing == 'geometry':
        del scene['inspection table']
    elif missing == 'ambiguous':
        scene['machines'].append({'resource_id': 'inspection table', 'KMR_docking_pose': [2., 2., 0., 0., 0., 0.]})
    elif missing == 'custody':
        for row in (snapshot, inputs['resource_evidence']['snapshot']):
            row.pop('held_part', None)
    else:
        inputs['resource_evidence'].pop('attachment', None)
    owner = SavedRecoveryResource(task['resource_jid'], {'resource_type': 'kmr', 'resource_evidence': inputs['resource_evidence'],
        'recovery_snapshot': snapshot, 'recovery_des_model': model}, {'jids': inputs['physical_evidence']['jids']})
    inputs['resource_evidence'] = owner.get_recovery_resource_evidence()
    before = deepcopy(inputs)
    result = validate_resource_outline_payload(owner, {'recovery_session_id': 'attempt-A', 'validation_contract_version': 2,
        'primitive_support_semantics': 'successor_primitive_union', 'outline_evidence_policy_version': 1,
        'candidates': [{'task': task, 'physical_input': inputs}]})['results'][0]
    assert result['feasibility_status'] == 'NEEDS_CONTEXT', result
    assert inputs == before


@pytest.mark.parametrize('operation', ['pick_part', 'place_release'])
@pytest.mark.parametrize('case', ['supported', 'missing_geometry', 'occupied'])
def test_additional_pickup_and_placement_require_their_own_geometry_and_peer_conditions(owner_context, operation, case):
    from cais_spade_llm.recovery_framework.kmr_calculation import prepare_calculation
    from cais_spade_llm.recovery_framework.geometry import quaternion

    owner = owner_for(owner_context, 'KMR')
    evidence = environment_owner_evidence(owner)
    scene = evidence['inputs']['scene']
    part = 'KET4_Square_4mm'
    pickup = operation == 'pick_part'
    location = 'M1' if pickup else 'M1 staging tray'
    machine = next(row for row in scene['machines'] if row['resource_id'] == 'M1')
    if pickup:
        for key in ('KMR_pick_orientations_xyzw', 'KMR_pick_approach_clearance_m', 'KMR_pick_arm_configurations', 'KMR_pick_docking_poses'):
            machine[key] = deepcopy(scene['Storage'][key])
    else:
        scene[location] = {key: deepcopy(machine[key]) for key in ('KMR_docking_pose', 'workholding_pose')}
    dock = machine['KMR_docking_pose']
    snapshot = {**environment_owner_snapshot(owner), 'resource_location': location,
        'resource_state': 'at_pick' if pickup else 'positioned', 'held_part': None if pickup else part,
        'approached_part': part if pickup else None, 'base_pose': [*dock[:3], *quaternion(dock[3:])],
        'current_pose': [dock[0], dock[1], 1.5, 1., 0., 0., 0.]}
    evidence['snapshot'] = deepcopy(snapshot)
    evidence['observed_parts'] = {part: {'observed_pose': [dock[0], dock[1], .5, 0., 0., 0., 1.]}}
    evidence['environment_products'][part]['location'] = location if pickup else 'KMR'
    evidence['attachment'] = {'attached': not pickup, 'held_part': part,
                              'grasp_transform': [0., 0., .025, 1., 0., 0., 0.]}
    evidence['environment_valuation']['M1'].update(resource_state='completed', part_name=part if pickup else None,
                                                  staging_part=None)
    evidence['environment_valuation']['KMR'] = deepcopy(snapshot)
    if case == 'missing_geometry':
        if pickup:
            machine.pop('KMR_pick_orientations_xyzw')
        else:
            scene[location].pop('workholding_pose')
    elif case == 'occupied':
        if pickup:
            evidence['environment_valuation']['M1']['part_name'] = 'RGOCG4-50_Round_4mm'
        else:
            evidence['environment_valuation']['M1']['staging_part'] = 'RGOCG4-50_Round_4mm'
    model = environment_recovery_descriptor(evidence, snapshot, resource_jid=owner.jid)
    task = {'outline_id': 'supplied-case', 'event_name': 'supplied-case', 'resource_jid': owner.jid,
        'part_name': part, 'expected_end_state': {'resource_location': location, 'held_part': part if pickup else None,
                                                'resource_state': 'carrying' if pickup else 'idle'}}
    inputs = {'resource_evidence': evidence, 'physical_evidence': {}}
    before = deepcopy((model, inputs))
    if case != 'supported':
        with pytest.raises(KeyError if case == 'missing_geometry' else ValueError):
            prepare_calculation(task=task, model=model, snapshot=snapshot, physical_input=inputs, session_id='attempt')
    else:
        calculated, private, record = prepare_calculation(task=task, model=model, snapshot=snapshot,
            physical_input=inputs, session_id='attempt')
        assert record and operation in record['calculations']
        support = derive_primitive_union(task=task, recovery_des_model=calculated, recovery_snapshot=snapshot)
        assert support['matching_transitions']
        assert not record['command_sent'] and not record['execution_authorized']
        assert private['resource_evidence']['environment_models']['M1'] != evidence['environment_models']['M1']
    assert (model, inputs) == before


def test_place_support_for_other_locations_does_not_hide_missing_recipient_conditions(owner_context):
    from cais_spade_llm.recovery_framework.kmr_calculation import prepare_calculation

    _, task, snapshot, model, inputs = bounded_movement_case(owner_context, carried=True)
    part = snapshot['held_part']
    location = 'inspection table'
    inputs['resource_evidence']['inputs']['scene'][location] = {
        'KMR_docking_pose': [1., 2., 0., 0., 0., 0.], 'workholding_pose': [1., 2., 1., 0., 0., 0.]}
    snapshot['resource_location'] = location
    task.update(part_name=part, expected_end_state={'held_part': None, 'resource_state': 'idle', 'part_location': location})
    original = deepcopy((model, snapshot, inputs))
    with pytest.raises(KeyError, match='Recorded peer custody and access conditions for inspection table'):
        prepare_calculation(task=task, model=model, snapshot=snapshot, physical_input=inputs, session_id='attempt')
    assert (model, snapshot, inputs) == original


def test_carried_part_context_does_not_turn_additional_movement_into_pickup(owner_context):
    from cais_spade_llm.recovery_framework.kmr_calculation import prepare_calculation

    _, task, snapshot, model, inputs = bounded_movement_case(owner_context, carried=True)
    part = snapshot['held_part']
    inputs['resource_evidence']['inputs']['scene']['inspection table'] = {'KMR_docking_pose': [1., 2., 0., 0., 0., 0.]}
    task.update(part_name=part, expected_end_state={'held_part': part, 'resource_state': 'carrying',
                                                  'resource_location': 'inspection table'})
    _, _, record = prepare_calculation(task=task, model=model, snapshot=snapshot, physical_input=inputs, session_id='attempt')
    assert set(record['calculations']) == {'move_to_location'}


@pytest.mark.parametrize('defect', [None, 'attachment', 'transform', 'part'])
@pytest.mark.parametrize('location', ['Storage', 'M1', 'M2'])
def test_placement_target_capability_keeps_recorded_custody_constraints(owner_context, defect, location):
    from cais_spade_llm.recovery_framework.geometry import compose, quaternion
    from cais_spade_llm.recovery_framework.kmr_recovery import check_kmr_primitive_capability

    _, task, snapshot, _, inputs = bounded_movement_case(owner_context, carried=True)
    evidence = inputs['resource_evidence']
    scene = evidence['inputs']['scene']
    part = snapshot['held_part']
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
        from cais_spade_llm.recovery_framework.kmr_outline_evidence import _calculation

        physical = deepcopy(inputs['physical_evidence'])
        physical['snapshot']['resources']['KMR'] = deepcopy(snapshot)
        physical['snapshot']['resources']['KMR']['grasp_transform'] = transform[:]
        physical['resource_models']['KMR']['configuration'].update(
            root_xyz=snapshot['base_pose'][:3], root_quaternion=snapshot['base_pose'][3:])
        actual = _calculation({'op': 'compute_place_targets'}, {'transform': transform},
            {'task': {'parameters': {'destination_location': location, 'part_name': part}}}, physical, evidence)
        assert actual == result['evidence']['calculated_targets']
        fixture = destination['slots'][part] if location == 'Storage' else destination['workholding_pose']
        placed = compose(actual['target'], transform)
        assert placed == pytest.approx([*fixture[:3], *quaternion(fixture[3:])])


@pytest.mark.parametrize('part', ['KET4_Square_4mm', 'RGOCG4-50_Round_4mm'])
@pytest.mark.parametrize('defect', [None, 'occupied', 'missing_inventory', 'missing_slot'])
def test_storage_receiving_conditions_preserve_exact_inventory_and_part_geometry(owner_context, part, defect):
    from cais_spade_llm.recovery_framework.geometry import compose, quaternion
    from cais_spade_llm.recovery_framework.kmr_calculation import prepare_calculation

    owner = owner_for(owner_context, 'KMR')
    evidence = environment_owner_evidence(owner)
    scene = evidence['inputs']['scene']
    storage = scene['Storage']
    dock = storage['KMR_pick_docking_poses'][part]
    snapshot = {**environment_owner_snapshot(owner), 'resource_location': 'Storage', 'held_part': part,
        'resource_state': 'positioned', 'base_pose': [dock[0], dock[1], 0., *quaternion([0., 0., dock[2]])],
        'current_pose': [dock[0], dock[1], 1.5, 1., 0., 0., 0.]}
    evidence['snapshot'] = deepcopy(snapshot)
    evidence['attachment'] = {'attached': True, 'held_part': part,
        'grasp_transform': [0., 0., .025, 1., 0., 0., 0.]}
    evidence['environment_valuation']['Storage']['inventory.' + part] = False
    evidence['environment_products'][part]['location'] = 'KMR'
    if defect == 'occupied':
        evidence['environment_valuation']['Storage']['inventory.' + part] = True
    elif defect == 'missing_inventory':
        evidence['environment_valuation']['Storage'].pop('inventory.' + part)
    elif defect == 'missing_slot':
        storage['slots'].pop(part)
    model = environment_recovery_descriptor(evidence, snapshot, resource_jid=owner.jid)
    task = {'outline_id': 'supplied-verification', 'event_name': 'supplied-verification',
        'resource_jid': owner.jid, 'part_name': part,
        'expected_end_state': {'held_part': None, 'part_location': 'Storage', 'resource_state': 'idle'}}
    inputs = {'resource_evidence': evidence,
        'physical_evidence': {'environment_valuation': deepcopy(evidence['environment_valuation'])}}
    original = deepcopy((model, snapshot, inputs))
    if defect:
        with pytest.raises(ValueError if defect == 'occupied' else KeyError):
            prepare_calculation(task=task, model=model, snapshot=snapshot, physical_input=inputs, session_id='test')
    else:
        calculated, private, record = prepare_calculation(task=task, model=model, snapshot=snapshot,
            physical_input=inputs, session_id='test')
        target = record['calculations']['place_release']['target']
        deposited = compose(target, evidence['attachment']['grasp_transform'])
        assert deposited[:3] == pytest.approx(storage['slots'][part][:3], abs=1e-12)
        support = derive_primitive_union(task=task, recovery_des_model=calculated, recovery_snapshot=snapshot)
        assert any(row['parameter_bindings'].get('destination_location') == 'Storage'
                   for row in support['matching_transitions'])
        peer = private['physical_evidence']['environment_models']['Storage']['events'][-1]
        assert peer['guards'] == {'inventory.' + part: {'equals': False}}
        assert peer['updates'] == {'inventory.' + part: {'set': True}}
        assert peer['product_effects'] == {'processCompleted': []}
    assert (model, snapshot, inputs) == original


@pytest.mark.parametrize('resource_id', ['M1', 'M2'])
def test_additional_pickup_retains_interrupted_peer_conditions_without_changing_capture(resource_id):
    from cais_spade_llm.recovery_framework.kmr_calculation import _handoff, _check_peer_conditions
    from cais_spade_llm.resources.machine.machine_jobs import MachineJobs, job_public_state

    meta = next(iter(read_json(PRODUCT_PATH).values()))
    context = EnvironmentProductContext(scene=read_json(SCENE_PATH),
        product_order=read_json(ROOT / meta['product_order_file']),
        geometry=read_json(ROOT / meta['product_geometry_file'])['gazebo'])
    evidence = environment_owner_evidence(owner_for(context, 'KMR'))
    scene, models = evidence['inputs']['scene'], evidence['environment_models']
    machine = next(row for row in scene['machines'] if row['resource_id'] == resource_id)
    part = 'KET4_Square_4mm' if resource_id == 'M1' else 'RGOCG4-50_Round_4mm'
    owner = MachineJobs(machine, 'recorded-run')
    owner.command('start_job', {'job_id': 'job', 'task_id': 'job', 'part_name': part, 'process': 'trim',
        'result': machine['selected_program']}, command_id='start', now=100,
        observations={'part_name': part, 'correct_part_present': True, 'robot_access_clear': True})
    owner.interrupt('operator Stop')
    record = owner.snapshot()
    valuation = deepcopy(evidence['environment_valuation'])
    valuation[resource_id].update(resource_state='loaded', part_name=part, **job_public_state(record))
    physical = {'environment_valuation': valuation, 'processing_observations': {resource_id: record},
                'processing_observation_source': {'run_id': 'recorded-run'}}
    before = deepcopy((models, scene, physical))
    rows, sources = _handoff(models, 'pick_approach', resource_id, part, scene=scene, physical=physical)
    _check_peer_conditions(rows, valuation, part, resource_id)
    supported = [row[resource_id] for row in rows if row[resource_id]['guards'].get('job_state') == {'equals': 'interrupted'}]
    assert supported and sources
    assert all(row['guards']['active_job'] == {'equals': None} and row['updates'] == {} for row in supported)
    assert all(source['machine_pickup_conditions']['physical_access_validated'] is False for source in sources)
    assert (models, scene, physical) == before
    physical['processing_observations'] = {}
    with pytest.raises(KeyError, match='job identity'):
        _handoff(models, 'pick_approach', resource_id, part, scene=scene, physical=physical)


def test_bounded_processing_then_spatial_does_not_resurrect_invalidated_peer(owner_context, forbid_validation_trajectories):
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes.multi_turn_validated_event import _project
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes.multi_turn_context import public_resource_snapshot

    owner, task, snapshot, model, inputs = bounded_movement_case(owner_context)
    all_specs = deepcopy(inputs['outline_safety_specifications'])
    inputs['outline_safety_specifications'] = all_specs[:1]
    inputs['physical_evidence']['resources']['ur5e-4'].pop('stationary_contract')
    ra, cca = bounded_validation((owner, task, snapshot, model, inputs))
    assert cca['is_safe'], cca.get('reason')
    session = {'outline_evidence_policy_version': 1, 'projected_physical_evidence': inputs['physical_evidence']}
    _project(task, ra, cca, session)
    successor = session['projected_physical_evidence']
    assert 'ur5e-4' in successor['invalidated_geometry']['resources']
    assert 'configuration' not in successor['resource_models']['ur5e-4']
    assert 'component_bounds' not in successor['geometry']['resources']['ur5e-4']
    assert successor['invalidated_evidence_history']
    peer_jid = successor['jids']['ur5e-4']
    prepared = {'recovery_resources': {peer_jid: {'recovery_snapshot': {
        'current_pose': [9., 9., 9., 0., 0., 0., 1.]}}}}
    assert 'current_pose' not in public_resource_snapshot(prepared, session, peer_jid)
    second = {**deepcopy(task), 'outline_id': 'spatial-check', 'expected_end_state': {
        'resource_location': 'M2', 'resource_state': 'idle'}}
    next_snapshot = session['derived_resource_snapshots'][task['resource_jid']]
    next_inputs = {**deepcopy(inputs), 'physical_evidence': successor, 'projected_snapshot': next_snapshot,
                   'outline_safety_specifications': all_specs}
    before = deepcopy(session)
    next_ra, next_cca = bounded_validation((owner, second, next_snapshot, model, next_inputs), session=session)
    assert next_ra['geometry_evidence']['endpoint_evidence']['status'] == 'supported'
    assert 'ur5e-4' in next_ra['geometry_evidence']['interval_evidence']['missing_peer_bounds']
    assert not next_cca['is_safe']
    assert next_cca['monitor_context_after'] == before['outline_monitor_context']
    assert session == before


def test_bounded_exact_nominal_task_cannot_be_manufactured(owner_context, forbid_validation_trajectories):
    from cais_spade_llm.recovery_framework.outline_evidence_policy import event_evidence_requirements, project_selected_successor

    _, task, snapshot, model, inputs = bounded_movement_case(owner_context)
    task['required_task_contract'] = 'place_part'
    required = event_evidence_requirements(task=task, snapshot=snapshot, model=model,
        physical_input=inputs, specifications=inputs['outline_safety_specifications'])
    projection = project_selected_successor(task=task, snapshot=snapshot, model=model,
        physical_input=inputs, requirements=required)
    assert projection['status'] == 'needs_context'
    assert projection['execution_authorized'] is False
    assert projection['requires_projection'] is False


def test_bounded_preview_entrypoints_cannot_start_trajectory(owner_context, forbid_validation_trajectories):
    import asyncio
    from cais_spade_llm.recovery_framework.outline_gazebo_evidence import prepare_owner_event_evidence
    from cais_spade_llm.recovery_framework.kmr_outline_evidence import prepare_kmr_event_evidence
    from cais_spade_llm.recovery_framework.part_slippage_recovery import PartSlippageRecovery

    owner, task, snapshot, model, inputs = bounded_movement_case(owner_context)
    physical = inputs['physical_evidence']
    physical['outline_safety_specifications'] = inputs['outline_safety_specifications']
    kwargs = {'task': task, 'recovery_snapshot': snapshot, 'recovery_des_model': model,
        'resource_evidence': inputs['resource_evidence'], 'physical_evidence': physical, 'observations': {}}
    result = asyncio.run(prepare_owner_event_evidence(owner, **kwargs))
    assert result['geometry_evidence']['endpoint_evidence']['status'] == 'supported'
    with pytest.raises(ValueError, match='disables KMR trajectory'):
        asyncio.run(prepare_kmr_event_evidence(owner, **kwargs))
    recovery = SimpleNamespace(_prepared={'physical_evidence': physical})
    with pytest.raises(ValueError, match='trajectory fallback is disabled'):
        asyncio.run(PartSlippageRecovery._prepare_offline_outline_evidence(recovery, set()))


@pytest.mark.parametrize('change', ['coverage', 'binding', 'successor', 'owner', 'state'])
def test_selected_cca_recomputes_requirements_and_coverage(owner_context, change):
    from cais_spade_llm.recovery_framework.outline_evidence_policy import check_evidence_policy
    case = selective_movement_case(owner_context)
    _, projection = selected_projection(case)
    task, snapshot, model, inputs = case
    if change == 'coverage':
        projection['evidence_coverage']['aps'][0]['status'] = 'missing'
    elif change == 'binding':
        inputs['outline_safety_specifications'][0]['document']['requirement_scopes'][0]['physical_ap_bindings']['ap001']['result'] = 'square'
    elif change == 'successor':
        projection['projected_resource_snapshot']['resource_location'] = 'M2'
    elif change == 'owner':
        model['source_model_fingerprint'] = 'changed'
    else:
        inputs['physical_evidence']['snapshot']['parts']['RGOCG4-50_Round_4mm']['processCompleted'] = [{'process': 'trim', 'result': 'circle'}]
    with pytest.raises(ValueError, match='requirements|coverage|reproduced'):
        check_evidence_policy(task=task, snapshot=snapshot, model=model, physical_input=inputs,
            specifications=inputs['outline_safety_specifications'], projection=projection)


def test_selective_carried_geometry_is_retired_and_cannot_return_from_capture(owner_context):
    from cais_spade_llm.recovery_framework.outline_evidence_policy import sanitize_validation_input, invalidate_session_facts
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes.multi_turn_context import public_resource_snapshot
    case = selective_movement_case(owner_context, carried=True)
    _, projection = selected_projection(case)
    assert projection is not None
    successor = projection['successor_evidence']
    assert 'RGOCG4-50_Round_4mm' in successor['invalidated_geometry']['parts']
    assert 'current_pose' not in successor['snapshot']['parts']['RGOCG4-50_Round_4mm']
    inputs = deepcopy(case[3])
    inputs['physical_evidence'] = successor
    inputs['projected_snapshot'] = projection['projected_resource_snapshot']
    sanitize_validation_input(inputs, case[0]['resource_jid'])
    assert 'base_pose' not in inputs['resource_evidence']['snapshot']
    session = {'symbolic_resources': {case[0]['resource_jid']: deepcopy(case[1])},
        'derived_resource_snapshots': {case[0]['resource_jid']: deepcopy(case[1])},
        'projected_physical_evidence': successor, 'validation_contract_version': 2}
    invalidate_session_facts(session, successor)
    prepared = {'recovery_resources': {case[0]['resource_jid']: {'recovery_snapshot': case[1],
        'recovery_des_model': case[2]}}}
    public = public_resource_snapshot(prepared, session, case[0]['resource_jid'])
    assert 'current_pose' not in public and 'projected_pose' not in public


def test_selective_successor_does_not_supply_exact_nominal_completion(owner_context):
    case = selective_movement_case(owner_context)
    case[0]['required_task_contract'] = {'task': 'exact nominal obligation'}
    _, projection = selected_projection(case)
    assert projection is None


@pytest.mark.parametrize('missing', ['scope', 'predicate'])
def test_selective_unknown_dependencies_retain_conservative_projection(owner_context, missing):
    from cais_spade_llm.agents.central_controller.ppr_ap import canonical_ap_key

    case = selective_movement_case(owner_context)
    record = case[3]['outline_safety_specifications'][0]
    if missing == 'scope':
        record['document']['requirement_scopes'] = []
    else:
        ap = record['document']['catalog']['specifications'][0]['aps'][0]
        ap['definition']['state']['symbol'] = 'unimplemented_predicate'
        ap['full'] = canonical_ap_key(ap['definition'])
    required, projection = selected_projection(case)
    assert required['conservative'] and required['unresolved']
    assert projection is None


def test_selective_unrelated_running_job_and_elapsed_clock_are_not_frozen(owner_context):
    case = selective_movement_case(owner_context)
    case[3]['physical_evidence']['processing_observations'] = {
        'M2': {'active_job': 'job', 'simulation_clock_sec': 4., 'jobs': {
            'job': {'job_state': 'running', 'part_name': 'gear_small'}}},
        'M1': {'active_job': None, 'jobs': {}, 'simulation_clock_sec': 4.}}
    _, projection = selected_projection(case)
    assert projection['status'] == 'projected'
    successor = projection['successor_evidence']
    assert 'M2' in successor['unresolved_processing']
    assert 'M2' not in successor['processing_observations']
    assert 'simulation_clock_sec' not in successor['processing_observations']['M1']
    assert successor['elapsed_time_obligations'][0]['last_observed_clocks']['M1'] == 4.
    from cais_spade_llm.recovery_framework.outline_evidence_policy import sanitize_validation_input

    later = {'physical_evidence': successor, 'projected_snapshot': {'active_job': 'job', 'job_state': 'running'},
        'resource_evidence': {'snapshot': {'active_job': 'job', 'job_state': 'running'},
            'processing_observations': deepcopy(case[3]['physical_evidence']['processing_observations']['M2'])}}
    sanitize_validation_input(later, 'M2@localhost')
    assert not later['projected_snapshot']
    assert 'processing_observations' not in later['resource_evidence']
    assert not later['resource_evidence']['snapshot']


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


@pytest.mark.parametrize("rid,location", [("ur5e-1", "M1"), ("ur5e-2", "M2")])
@pytest.mark.parametrize("nominal", [False, True])
def test_robot_successor_keeps_deferred_guards_separate_from_owner_effects(owner_context, rid, location, nominal):
    from cais_spade_llm.recovery_framework.outline_endpoint_evidence import _robot_successors
    from cais_spade_llm.resources.robot.robot_task_recovery import robot_recovery_des_descriptor

    part = "KET4_Square_4mm"
    owner_context.resources["Storage"].valuation["inventory." + part] = False
    owner_context.resources[location].valuation.update(part_name=part, resource_state="loaded")
    owner_context.part_tracker[part].update(location=location, state="ready")
    owner = owner_for(owner_context, rid)
    evidence = environment_owner_evidence(owner)
    snapshot = {"resource_state": "idle", "resource_location": "home", "held_part": None}
    model = robot_recovery_des_descriptor(resource_jid=owner.jid, snapshot=snapshot,
        task_names=["pick_grasp"], reachable_locations=[location])
    task = {"outline_id": "supplied_verification_case", "event_name": "arbitrary_pickup",
        "resource_jid": owner.jid, "part_name": part,
        "expected_end_state": {"resource_state": "picked", "resource_location": location, "held_part": part}}
    if nominal:
        task["required_task_contract"] = "pick_grasp"
    physical = {"environment_valuation": deepcopy(evidence["environment_valuation"]),
                "products": deepcopy(evidence["environment_products"])}
    original = deepcopy((task, snapshot, model, evidence, physical))
    candidates, failures = _robot_successors(task=task, snapshot=snapshot, model=model,
        evidence=evidence, physical=physical)
    if nominal:
        assert not candidates and failures
    else:
        assert len(candidates) == 1, failures
        result = candidates[0]
        assert result["environment_valuation"][rid]["held_part"] == part
        assert result["environment_valuation"][location]["part_name"] is None
        assert result["public_part_updates"]["part_location"] == owner.jid
        assert result["environment_products"][part]["location"] == rid
        assert result["owner_declarations"][0]["composition_prerequisites"]
        evidence["environment_models"][rid]["events"].reverse()
        reversed_candidates, _ = _robot_successors(task=task, snapshot=snapshot, model=model,
            evidence=evidence, physical=physical)
        assert [row["environment_valuation"] for row in reversed_candidates] == [result["environment_valuation"]]
        evidence["environment_models"][rid]["events"].reverse()
    assert (task, snapshot, model, evidence, physical) == original


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


@pytest.mark.parametrize('endpoint_policy_version', [None, 2])
def test_static_machine_projection_records_supported_interval_and_declared_completion(
        owner_context, endpoint_policy_version, forbid_validation_trajectories):
    from cais_spade_llm.agents.central_controller.outline_evidence_safety import bind_outline_owner_safety_evidence
    owner, evidence, snapshot, model, physical = static_machine_projection_input(owner_context)
    task = machine_proposal()
    task['expected_end_state']['part_state'] = 'completed'
    task['required_task_contract'] = 'machine_part'
    if endpoint_policy_version is not None:
        physical.update(endpoint_evidence_policy_version=endpoint_policy_version, outline_safety_specifications=[])
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


@pytest.mark.parametrize('endpoint_policy_version', [None, 2])
@pytest.mark.parametrize('missing', ['peer_hold', 'part_support', 'part_state'])
def test_machine_projection_keeps_missing_or_conflicting_evidence_unresolved(
        owner_context, missing, endpoint_policy_version, forbid_validation_trajectories):
    owner, evidence, snapshot, model, physical = static_machine_projection_input(owner_context)
    task = machine_proposal()
    task['expected_end_state']['part_state'] = 'completed'
    if endpoint_policy_version is not None:
        physical.update(endpoint_evidence_policy_version=endpoint_policy_version, outline_safety_specifications=[])
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
    if missing == 'peer_hold':
        assert any('ur5e-3' in reason for reason in result['unresolved'])
    elif missing == 'part_support':
        assert any('stationary part evidence: KET4_Square_4mm' in reason for reason in result['unresolved'])
    else:
        assert result['owner_effect_rejected'] is True
        assert result['owner_effect_result']['field'] == 'part_state'
    if missing != 'part_state' and endpoint_policy_version is None:
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
    from test_kmr_outline_evidence import motion_fixture
    from cais_spade_llm.recovery_framework import kmr_recovery
    from cais_spade_llm.resources.continuous_motion import interval, kinematic_transforms, representative_pose

    physical, _, _ = motion_fixture()
    config = physical['resource_models']['KMR']['configuration']
    state = physical['snapshot']['resources']['KMR']
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


@pytest.mark.parametrize('defect', ['', 'missing_contract', 'outside_bounds'])
def test_pickup_open_gripper_projects_recorded_joint_configuration(monkeypatch, defect):
    from test_kmr_outline_evidence import motion_fixture
    from cais_spade_llm.recovery_framework import kmr_outline_evidence
    from cais_spade_llm.recovery_framework.outline_endpoint_evidence import _kmr_endpoint

    physical, _, _ = motion_fixture()
    physical.update(endpoint_evidence_policy_version=2, jids={'KMR': 'KMR@localhost'})
    state = physical['snapshot']['resources']['KMR']
    state['gripper_state'] = 'open'
    config = physical['resource_models']['KMR']['configuration']
    config['joints'].append({'name': 'finger', 'parent': 'tool', 'child': 'finger_link', 'type': 'prismatic',
        'axis': [1., 0., 0.], 'xyz': [0., 0., 0.], 'rpy': [0., 0., 0.], 'frozen_position': .1})
    config['fixed_joint_positions'] = {'finger': .1}
    contract = {'joint_positions': {'open': {'finger': .2}}, 'joint_bounds': {'finger': [0., .3]},
                'gripper': {'open': .2}, 'source': 'supplied recorded owner gripper contract'}
    physical['resource_models']['KMR']['primitive_model'] = {'configuration': {'custody_by_part': {'default': contract}}}
    if defect == 'missing_contract':
        physical['resource_models']['KMR'].pop('primitive_model')
    elif defect == 'outside_bounds':
        contract['joint_positions']['open']['finger'] = .4
    evidence = {'inputs': {'scene': {'KMR': {'arm_joint_names': state['joint_names'], 'gripper_stroke_m': .2}}}}
    monkeypatch.setattr(kmr_outline_evidence, '_initial_state', lambda *args: {})
    monkeypatch.setattr(kmr_outline_evidence, '_resolve_step', lambda *args: {'status': 'resolved', 'params': {}})
    kwargs = dict(task={'resource_jid': 'KMR@localhost'}, snapshot=state,
        model={'resource_jid': 'KMR@localhost'}, evidence=evidence, physical=physical, declared={},
        event={'program': {'steps': [{'id': 'open', 'op': 'open_gripper', 'store_as': 'open'}]}})
    before = deepcopy(physical)
    if defect:
        with pytest.raises((KeyError, ValueError)):
            _kmr_endpoint(**kwargs)
    else:
        endpoint, steps = _kmr_endpoint(**kwargs)
        assert endpoint['resource_models']['KMR']['configuration']['fixed_joint_positions']['finger'] == .2
        assert steps[0]['output']['configuration_after']['fixed_joint_positions']['finger'] == .2
        assert steps[0]['output']['source'] == contract['source']
        assert endpoint['snapshot']['resources']['KMR']['held_part'] is None
    assert physical == before


@pytest.mark.parametrize('operation,defect', [('grasp_part', ''), ('release_part', ''),
    ('grasp_part', 'distance'), ('grasp_part', 'custody'), ('grasp_part', 'bounds'),
    ('grasp_part', 'contract'), ('release_part', 'support'), ('release_part', 'transform')])
def test_kmr_custody_endpoints_preserve_pose_and_require_recorded_effects(monkeypatch, operation, defect):
    from test_kmr_outline_evidence import motion_fixture
    from cais_spade_llm.recovery_framework import kmr_outline_evidence
    from cais_spade_llm.recovery_framework.outline_endpoint_evidence import _kmr_endpoint

    physical, _, _ = motion_fixture()
    physical.update(endpoint_evidence_policy_version=2, jids={'KMR': 'KMR@localhost'})
    state = physical['snapshot']['resources']['KMR']
    releasing = operation == 'release_part'
    part = 'gear_small'
    pose = [0., 0., .7, 0., 0., 0., 1.]
    transform = [0., 0., 0., 0., 0., 0., 1.]
    state.update(gripper_state='closed' if releasing else 'open', held_part=part if releasing else None,
                 grasp_transform=deepcopy(transform) if releasing else None)
    physical['snapshot']['resources']['M1']['contained_parts'] = [] if releasing else [part]
    physical['snapshot']['parts'][part] = {'model_name': part, 'current_pose': deepcopy(pose),
                                         'contained_by': None if releasing else 'M1'}
    physical['geometry']['parts'][part] = {'local_bounds': [[-.02, .02]] * 3}
    physical['physics_evidence'] = {'instance_id': 'supplied-custody-case', 'simulation_time': 1.,
        'physics_engine': 'ode', 'gravity': [0., 0., -9.81],
        'contact_surface_layer': 0., 'contacts': [], 'models': {
            part: {'static': False, 'pose': deepcopy(pose), 'links': {'link': {
                'mass': .1, 'pose': deepcopy(pose), 'center_of_mass': pose[:3],
                'linear_velocity': [0.] * 3, 'angular_velocity': [0.] * 3,
                'collisions': [{'id': part + '::collision', 'pose': deepcopy(pose), 'contact_min_depth': 0.,
                                'geometry': {'type': 'box', 'size': [.04] * 3}}]}}},
            'support': {'static': True, 'links': {'link': {'collisions': [{'id': 'support::collision',
                'pose': [0., 0., .63, 0., 0., 0., 1.], 'contact_min_depth': 0.,
                'geometry': {'type': 'box', 'size': [1., 1., .1]}}]}}}}}
    physical['part_support_policy'] = {'kind': 'gravity_supported_rest', 'requires_no_active_goals': True,
        'future_execution_tracking': 'not_established', 'stationary_linear_speed_max': .001,
        'stationary_angular_speed_max': .001}
    config = physical['resource_models']['KMR']['configuration']
    config['joints'].append({'name': 'finger', 'parent': 'tool', 'child': 'finger_link', 'type': 'prismatic',
        'axis': [1., 0., 0.], 'xyz': [0., 0., 0.], 'rpy': [0., 0., 0.],
        'frozen_position': .1 if releasing else .2})
    contract = {'kind': 'gazebo_link_attacher', 'preserves_part_pose_on_attachment': True,
        'joint_positions': {'open': {'finger': .2}, 'closed': {'finger': .1}},
        'joint_bounds': {'finger': [0., .3]}, 'gripper': {'open': .2, 'closed': .1},
        'source': 'supplied recorded custody contract'}
    physical['resource_models']['KMR']['primitive_model'] = {'configuration': {'custody_by_part': {'default': contract}}}
    evidence = {'inputs': {'scene': {'KMR': {'arm_joint_names': state['joint_names']}}},
                'observed_parts': {part: {'current_pose': deepcopy(pose)}},
                'attachment': {'attached': releasing, 'held_part': part if releasing else None,
                               'grasp_transform': deepcopy(transform)}}
    params = {'transform': deepcopy(transform)} if releasing else {'initial': deepcopy(pose)}
    if defect == 'distance':
        state['current_pose'][0] += 1.
    elif defect == 'custody':
        physical['snapshot']['resources']['M1']['held_part'] = part
    elif defect == 'bounds':
        contract['joint_positions']['closed']['finger'] = .4
    elif defect == 'contract':
        contract.pop('preserves_part_pose_on_attachment')
    elif defect == 'support':
        physical['physics_evidence']['models'].pop('support')
    elif defect == 'transform':
        params['transform'][0] += .1
    monkeypatch.setattr(kmr_outline_evidence, '_initial_state', lambda *args: {})
    monkeypatch.setattr(kmr_outline_evidence, '_resolve_step',
                        lambda *args: {'status': 'resolved', 'params': deepcopy(params)})
    before = deepcopy(physical)
    kwargs = dict(task={'resource_jid': 'KMR@localhost', 'part_name': part}, snapshot=state,
        model={'resource_jid': 'KMR@localhost'}, evidence=evidence, physical=physical,
        declared={'task': {'parameters': {'part_name': part, 'destination_location': 'M1'}}},
        event={'program': {'steps': [{'id': operation, 'op': operation, 'store_as': operation}]}})
    if defect:
        with pytest.raises((KeyError, ValueError)):
            _kmr_endpoint(**kwargs)
    else:
        endpoint, steps = _kmr_endpoint(**kwargs)
        assert endpoint['snapshot']['resources']['KMR']['held_part'] == (None if releasing else part)
        assert endpoint['snapshot']['resources']['M1']['contained_parts'] == ([part] if releasing else [])
        assert endpoint['snapshot']['parts'][part]['current_pose'] == pytest.approx(pose)
        assert not steps[0]['output']['physical_execution_verified']
        assert not steps[0]['output']['collision_or_path_validated']
        if releasing:
            assert endpoint['stationary_part_evidence'][part]['status'] == 'supported'
        else:
            assert steps[0]['output']['support_departure_status'] == 'not_checked'
            assert part not in endpoint.get('stationary_part_evidence', {})
    assert physical == before


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
