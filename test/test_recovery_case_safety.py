from __future__ import annotations

"""Recorded case safety definitions and conservative owner-evidence checks."""

import json
from copy import deepcopy
from pathlib import Path

import pytest

from cais_spade_llm.agents.central_controller.outline_evidence_safety import (
    bind_outline_owner_safety_evidence, validate_outline_evidence_safety,
)
from cais_spade_llm.agents.central_controller.predefined_safety import compile_predefined_safety
from cais_spade_llm.agents.central_controller.recovery_case_evidence import case_ap_value
from cais_spade_llm.agents.central_controller.reviewed_primitive_program_safety import _digest
from cais_spade_llm.recovery_framework.recovery_case_safety import default_safety_files

ROOT = Path(__file__).resolve().parents[1]
SAFETY = 'cais_spade_llm/specification/safety/'
PLANT_SAFETY_FILES = [SAFETY + name for name in (
    'safety_assembly_board-v1_mutex.txt', 'safety_M1_mutex.txt', 'safety_M2_mutex.txt',
    'safety_assembly_board-v1_trim_before_entry.txt',
    'safety_assembly_board-v1_gear_small_before_KET4_Square_4mm.txt',
    'safety_assembly_board-v1_KET4_Square_4mm_before_RGOCG4-50_Round_4mm.txt',
    'safety_KET4_Square_4mm_before_RGOCG4-50_Round_4mm_processing.txt')]
SCENARIOS = ['Part slippage', 'Conveyor breakdown', 'ur5e-1 breakdown',
             'Machining breakdown during part processing']


def _spec(name):
    return json.loads((ROOT / SAFETY / name).read_text())


def _document(identifier):
    doc = json.loads((ROOT / 'cais_spade_llm/specification/safety/safety_Conveyor_breakdown.txt').read_text())
    return {**doc, 'catalog': {'version': 2, 'specifications': [r for r in doc['catalog']['specifications']
            if r['id'] == identifier]}, 'requirement_scopes': [r for r in doc['requirement_scopes']
            if r['specification'] == identifier]}


def _projection(inside=False):
    resources = {'M1': {'resource_state': 'loaded'}, 'ur5e-1': {'held_part': None}}
    model = {'M1': {'state_variables': {'resource_state': {'domain': ['loaded', 'completed']}}},
             'ur5e-1': {'state_variables': {'held_part': {'type': ['null', 'string'], 'reference': 'part_name'}}}}
    initial = [{'resource_id': rid, 'process': '*', 'product': '*', 'context': 'any', 'values': values}
               for rid, values in resources.items()]
    event = {'task_id': 'machine_task', 'resource_id': 'M1', 'process': '*', 'product': '*',
             'context': 'any', 'function': 'machine_part', 'start_time': 0, 'end_time': 1}
    update = {**initial[0], 'time': 1, 'task_id': 'machine_task', 'values': {'resource_state': 'completed'}}
    observations = []
    for index, time in enumerate(('0', '1/2', '1')):
        state = deepcopy(resources)
        if index == 2:
            state['M1']['resource_state'] = 'completed'
        observation = {'frame': 'world', 'resources': deepcopy(state), 'parts': {}, 'time': index,
                       'time_exact': time, 'phase': 'between' if index == 1 else 'at',
                       'state_snapshot': state,
                       'region_occupancy': {'M1.front_access': {'M1': True, 'ur5e-1': inside}},
                       'part_region_occupancy': {'M1.front_access': {}}}
        if index == 1:
            observation['continuous_interval'] = ['0', '1']
        observations.append(observation)
    return {'status': 'projected', 'interval_complete': True, 'observations': observations,
            'geometry': {'frame': 'world', 'resources': {r: {} for r in resources}, 'parts': {},
                         'regions': {'M1.front_access': {'bounds': [[0,1],[0,1],[0,1]]}}},
            'owner_safety_evidence': {'resource_models': model, 'interval_complete': True,
                'task_evidence': {'complete': True, 'source_kind': 'resource_owned_projection',
                                  'horizon': [0,1], 'events': [event]},
                'state_evidence': {'complete': True, 'source_kind': 'resource_owned_projection',
                                   'horizon': [0,1], 'initial': initial, 'updates': [update]}}}


def _check(projection, identifier='SAFE_M1_front_access_ur5e-1_interlock', **kwargs):
    doc = _document(identifier)
    return validate_outline_evidence_safety(projection=projection, catalog=doc['catalog'],
        requirement_scopes=doc['requirement_scopes'], prospective_activation=True, **kwargs)


@pytest.mark.parametrize('scenario', SCENARIOS)
def test_case_defaults_compile_given_rules_without_model_calls(scenario):
    selected = default_safety_files(scenario)
    assert selected == PLANT_SAFETY_FILES
    identifiers = []
    for path in selected:
        rules, dfas = compile_predefined_safety(json.loads((ROOT / path).read_text()))
        assert len(rules) == len(dfas) == 1
        identifiers.append(rules[0]['id'])
    assert identifiers == [
        'SAFE_shared_area_mutex', 'SAFE_M1_mutex', 'SAFE_M2_mutex',
        'SAFE_trim_before_assembly_board-v1_entry',
        'SAFE_gear_small_before_KET4_Square_4mm', 'SAFE_KET4_Square_4mm_before_RGOCG4-50_Round_4mm',
        'SAFE_KET4_Square_4mm_before_RGOCG4-50_Round_4mm_processing']
    selected.clear()
    assert default_safety_files(scenario) == PLANT_SAFETY_FILES


def test_previous_machine_safety_file_remains_available():
    path = ROOT / 'cais_spade_llm/specification/safety/safety_Machining_breakdown_during_part_processing.txt'
    rules, dfas = compile_predefined_safety(json.loads(path.read_text()))
    assert len(rules) == len(dfas) == 77
    assert 'SAFE_KMR_docking' in dfas


@pytest.mark.parametrize('first_region', ['M1.front_access', 'M1.side_access', 'M1.workholding'])
@pytest.mark.parametrize('second_region', ['M1.front_access', 'M1.side_access', 'M1.workholding'])
@pytest.mark.parametrize('second_inside,expected', [(False, 'FEASIBLE'), (True, 'INFEASIBLE'),
                                                  (None, 'NEEDS_CONTEXT')])
def test_m1_mutex_counts_failed_robot_across_all_access_areas(first_region, second_region, second_inside, expected):
    resources = {'M1': {'resource_state': 'completed'},
                 'ur5e-1': {'resource_state': 'failed', 'held_part': None},
                 'KMR': {'held_part': None}, 'ur5e-4': {'held_part': None}}
    regions = ['M1.front_access', 'M1.side_access', 'M1.workholding']
    occupancy = {region: {rid: rid == 'M1' for rid in resources} for region in regions}
    occupancy[first_region]['ur5e-1'] = True
    occupancy[second_region]['KMR'] = second_inside
    projection = {'status': 'projected', 'interval_complete': True,
        'observations': [{'time': 0, 'time_exact': '0', 'phase': 'at', 'frame': 'world',
                          'resources': resources, 'parts': {}, 'region_occupancy': occupancy}],
        'geometry': {'frame': 'world', 'parts': {},
                     'resources': {rid: {'stationary_only': rid == 'M1'} for rid in resources},
                     'regions': {region: {'bounds': [[0, 1]] * 3} for region in regions}}}
    doc = _spec('safety_M1_mutex.txt')
    result = validate_outline_evidence_safety(projection=projection, catalog=doc['catalog'],
        requirement_scopes=doc['requirement_scopes'], prospective_activation=True)
    assert result['validation_status'] == expected
    if second_inside is not None:
        population = result['coverage']['scope_populations'][0]
        assert population['resources'] == ['KMR', 'ur5e-1', 'ur5e-4']
        assert [row['resource_id'] for row in population['excluded_resources']] == ['M1']
        assert len(result['rule_checks']) == 3


def test_previous_conveyor_safety_file_remains_available():
    path = ROOT / 'cais_spade_llm/specification/safety/safety_Conveyor_breakdown.txt'
    rules, dfas = compile_predefined_safety(json.loads(path.read_text()))
    assert len(rules) == len(dfas) == 77
    assert 'SAFE_KMR_docking' in dfas


_CONVEYOR_RULES = ('safety_assembly_board-v1_mutex.txt',
                   'safety_assembly_board-v1_KET4_Square_4mm_before_gear_small.txt')


def _conveyor_document(index):
    return _spec(_CONVEYOR_RULES[index])


def _conveyor_assembly_projection(trace):
    robots = ['ur5e-3', 'ur5e-4', 'KMR']
    parts = ['KET4_Square_4mm', 'gear_small']
    observations = []
    for time, (completed, gear_inside, robot_inside, kmr_inside) in enumerate(trace):
        observations.append({
            'time': time, 'time_exact': str(time), 'phase': 'at', 'frame': 'world',
            'resources': {rid: {'held_part': None} for rid in robots},
            'parts': {part: {
                'processCompleted': ([{'process': 'assembly',
                    'target': 'GMC_Laser_Plate_Virtual/KET4_Square_4mm'}]
                    if completed and part == 'KET4_Square_4mm' else []),
                'processCompleted_complete': True,
                'processCompleted_evidence': {'complete': True, 'source_kind': 'recorded',
                                              'checkpoint': 'capture'},
            } for part in parts},
            'region_occupancy': {'assembly_board-v1': {
                'ur5e-3': robot_inside, 'ur5e-4': gear_inside, 'KMR': kmr_inside}},
            'part_region_occupancy': {'assembly_board-v1': {
                'KET4_Square_4mm': completed, 'gear_small': gear_inside}},
        })
    return {'status': 'projected', 'interval_complete': True, 'observations': observations,
            'geometry': {'frame': 'world', 'resources': {rid: {} for rid in robots},
                         'regions': {'assembly_board-v1': {'bounds': [[0, 1]] * 3}},
                         'parts': {
                             'KET4_Square_4mm': {'target': 'GMC_Laser_Plate_Virtual/KET4_Square_4mm'},
                             'gear_small': {'target': 'Gear_Plate/Gear_Shaft_1'},
                         }}}


def _conveyor_check(projection, index=None, **kwargs):
    doc = _conveyor_document(1 if index is None else index)
    if index is None:
        mutex = _conveyor_document(0)
        doc['catalog']['specifications'] += mutex['catalog']['specifications']
        doc['requirement_scopes'] += mutex['requirement_scopes']
    return validate_outline_evidence_safety(
        projection=projection, catalog=doc['catalog'], requirement_scopes=doc['requirement_scopes'],
        prospective_activation=True, **kwargs)


def _trim_entry_projection(trace):
    """Observe every trimmed peg; each step gives the recorded trims and the parts on the board."""
    scopes = _spec('safety_assembly_board-v1_trim_before_entry.txt')['requirement_scopes']
    parts = [scope['physical_ap_bindings']['ap001']['part'] for scope in scopes]
    observations = []
    for time, (trims, entered) in enumerate(trace):
        observations.append({
            'time': time, 'time_exact': str(time), 'phase': 'at', 'frame': 'world',
            'resources': {'ur5e-3': {'held_part': None}},
            'parts': {part: {
                'processCompleted': [{'process': 'trim', 'result': trims[part]}] if part in trims else [],
                'processCompleted_complete': True,
                'processCompleted_evidence': {'complete': True, 'source_kind': 'recorded',
                                              'checkpoint': 'capture'},
            } for part in parts},
            'region_occupancy': {'assembly_board-v1': {'ur5e-3': False}},
            'part_region_occupancy': {'assembly_board-v1': {part: part in entered for part in parts}},
        })
    return {'status': 'projected', 'interval_complete': True, 'observations': observations,
            'geometry': {'frame': 'world', 'resources': {'ur5e-3': {}},
                         'regions': {'assembly_board-v1': {'bounds': [[0, 1]] * 3}},
                         'parts': {part: {} for part in parts}}}


def test_trim_before_entry_covers_every_trimmed_peg_and_no_gear():
    doc = _spec('safety_assembly_board-v1_trim_before_entry.txt')
    rule, = doc['catalog']['specifications']
    assert rule['formula'] == '(!ap001 U (ap002 & !ap001)) | G !ap001'
    bindings = [scope['physical_ap_bindings'] for scope in doc['requirement_scopes']]
    assert [(row['ap001']['part'], row['ap002']['result']) for row in bindings] == [
        ('KET4_Square_4mm', 'square'), ('KET8_Square_8mm', 'square'),
        ('KET12_Square_12mm', 'square'), ('KET16_Square_16mm', 'square'),
        ('RGOCG4-50_Round_4mm', 'circle'), ('RGOCG8-50_8mm', 'circle'),
        ('RGOCG12-50_12mm', 'circle'), ('RGOCG16-50_16mm', 'circle')]
    assert all(row['ap001']['part'] == row['ap002']['part'] and row['ap002']['process'] == 'trim'
               and row['ap001']['region'] == 'assembly_board-v1' for row in bindings)


KET4, RGOCG4 = 'KET4_Square_4mm', 'RGOCG4-50_Round_4mm'


@pytest.mark.parametrize('trace,expected', [
    ([({}, set())], 'FEASIBLE'),
    ([({}, set()), ({KET4: 'square'}, set()), ({KET4: 'square'}, {KET4})], 'FEASIBLE'),
    ([({}, set()), ({KET4: 'square'}, {KET4})], 'INFEASIBLE'),
    ([({}, set()), ({}, {KET4})], 'INFEASIBLE'),
    ([({}, set()), ({KET4: 'square'}, set()), ({KET4: 'square'}, {RGOCG4})], 'INFEASIBLE'),
    ([({}, set()), ({RGOCG4: 'square'}, set()), ({RGOCG4: 'square'}, {RGOCG4})], 'INFEASIBLE'),
    ([({}, set()), ({RGOCG4: 'circle'}, set()), ({RGOCG4: 'circle'}, {RGOCG4})], 'FEASIBLE'),
])
def test_trim_before_entry_checks_each_peg_against_its_own_trim(trace, expected):
    doc = _spec('safety_assembly_board-v1_trim_before_entry.txt')
    result = validate_outline_evidence_safety(
        projection=_trim_entry_projection(trace), catalog=doc['catalog'],
        requirement_scopes=doc['requirement_scopes'], prospective_activation=True)
    assert result['validation_status'] == expected


def test_conveyor_mutex_and_retired_ordering_specs_compile_independently():
    from cais_spade_llm.recovery_framework.part_slippage_recovery import _prompt_safety_rules

    paths = [SAFETY + name for name in _CONVEYOR_RULES]
    defaults = default_safety_files('Conveyor breakdown')
    assert paths[0] in defaults and paths[1] not in defaults
    for index, identifier in enumerate(['SAFE_shared_area_mutex', 'SAFE_KET4_Square_4mm_before_gear_small']):
        rules, dfas = compile_predefined_safety(_conveyor_document(index))
        assert len(rules) == len(dfas) == 1
        assert rules[0]['id'] == identifier
    assert [row['requirement'] for row in _prompt_safety_rules(paths)] == [
        'At most one resource may occupy the assembly_board-v1 destination area at any time.',
        'gear_small may enter assembly_board-v1 only after assembly of KET4_Square_4mm is completed.']


@pytest.mark.parametrize('trace,expected', [
    ([(False, False, False, False)], 'FEASIBLE'),
    ([(False, False, False, False), (True, False, True, False), (True, True, False, False)], 'FEASIBLE'),
    ([(False, False, False, False), (False, True, False, False)], 'INFEASIBLE'),
    ([(False, False, False, False), (True, True, False, False)], 'INFEASIBLE'),
])
def test_conveyor_ordering_requires_earlier_exact_assembly_completion(trace, expected):
    assert _conveyor_check(_conveyor_assembly_projection(trace), 1)['validation_status'] == expected


@pytest.mark.parametrize('record', [
    {'process': 'trim', 'result': 'square'},
    {'process': 'assembly', 'target': 'Gear_Plate/Gear_Shaft_1'},
])
def test_conveyor_ordering_rejects_machining_or_another_assembly_target(record):
    projection = _conveyor_assembly_projection([(False, False, False, False),
                                      (True, False, False, False), (True, True, False, False)])
    for row in projection['observations'][1:]:
        row['parts']['KET4_Square_4mm']['processCompleted'] = [record]
    assert _conveyor_check(projection, 1)['validation_status'] == 'INFEASIBLE'


def test_conveyor_ordering_retains_completion_history_and_missing_evidence():
    projection = _conveyor_assembly_projection([(False, False, False, False),
                                      (True, False, False, False), (True, True, False, False)])
    accepted = _conveyor_check({**projection, 'observations': projection['observations'][:2]}, 1)
    assert accepted['validation_status'] == 'FEASIBLE'
    result = _conveyor_check({**projection, 'observations': projection['observations'][1:]}, 1,
                            monitor_context=accepted['monitor_context_after'])
    assert result['validation_status'] == 'FEASIBLE'
    assert len(result['monitor_context_after']['observations']) == 3
    projection['observations'][0]['parts']['KET4_Square_4mm']['processCompleted_complete'] = False
    assert _conveyor_check(projection, 1)['validation_status'] == 'NEEDS_CONTEXT'


@pytest.mark.parametrize('robot_inside,kmr_inside,expected', [
    (False, False, 'FEASIBLE'), (True, False, 'FEASIBLE'),
    (False, True, 'FEASIBLE'), (True, True, 'INFEASIBLE'),
])
def test_conveyor_mutex_includes_kmr_occupancy(robot_inside, kmr_inside, expected):
    projection = _conveyor_assembly_projection([(False, False, robot_inside, kmr_inside)])
    result = _conveyor_check(projection, 0)
    assert result['validation_status'] == expected
    assert len(result['rule_checks']) == 3


def test_conveyor_mutex_keeps_missing_kmr_geometry_unresolved():
    projection = _conveyor_assembly_projection([(False, False, True, False)])
    del projection['observations'][0]['region_occupancy']['assembly_board-v1']['KMR']
    assert _conveyor_check(projection, 0)['validation_status'] == 'NEEDS_CONTEXT'


@pytest.mark.parametrize('trace,expected', [
    ([(False, False, False, False), (False, True, False, False)], 'INFEASIBLE'),
    ([(False, False, False, False), (True, False, True, False), (True, True, True, False)], 'INFEASIBLE'),
    ([(False, False, False, False), (True, False, False, True), (True, True, False, True)], 'INFEASIBLE'),
    ([(False, False, False, False), (True, False, True, False), (True, True, False, False)], 'FEASIBLE'),
])
def test_conveyor_checks_ordering_and_mutex_together(trace, expected):
    assert _conveyor_check(_conveyor_assembly_projection(trace))['validation_status'] == expected


_MACHINE_RULES = ('safety_M2_mutex.txt', 'safety_KET4_Square_4mm_before_RGOCG4-50_Round_4mm_processing.txt')


def _machine_document(index):
    return _spec(_MACHINE_RULES[index])


def _machine_check(projection, index=0, **kwargs):
    doc = _machine_document(index)
    return validate_outline_evidence_safety(
        projection=projection, catalog=doc['catalog'], requirement_scopes=doc['requirement_scopes'],
        prospective_activation=True, **kwargs)


def _machine_projection():
    robots = ['ur5e-2', 'KMR', 'ur5e-1']
    resources = {rid: {'held_part': None} for rid in robots}
    resources['M2'] = {'resource_state': 'loaded'}
    regions = ['M2.front_access', 'M2.side_access', 'M2.workholding']
    parts = {part: {'processCompleted': [], 'processCompleted_complete': True,
                    'processCompleted_evidence': {'complete': True, 'source_kind': 'recorded',
                                                  'checkpoint': 'capture'}}
             for part in ('KET4_Square_4mm', 'RGOCG4-50_Round_4mm')}
    observation = {'time': 0, 'time_exact': '0', 'phase': 'at', 'frame': 'world',
                   'resources': resources, 'parts': parts,
                   'region_occupancy': {region: {rid: rid == 'M2' for rid in resources} for region in regions}}
    return {'status': 'projected', 'interval_complete': True, 'observations': [observation],
            'geometry': {'frame': 'world', 'resources': {rid: {'stationary_only': rid == 'M2'} for rid in resources},
                         'regions': {region: {'bounds': [[0, 1]] * 3} for region in regions},
                         'parts': {part: {} for part in parts}}}


def _processing_observations(completions):
    projection = _machine_projection()
    observations = []
    for time, (square, circle) in enumerate(completions):
        row = deepcopy(projection['observations'][0])
        row.update(time=time, time_exact=str(time))
        for part, completed, result in [('KET4_Square_4mm', square, 'square'),
                                         ('RGOCG4-50_Round_4mm', circle, 'circle')]:
            row['parts'][part]['processCompleted'] = [{'process': 'trim', 'result': result}] if completed else []
            row['parts'][part]['location'] = 'M1' if time == 0 else 'M2'
        observations.append(row)
    projection['observations'] = observations
    return projection


def test_machine_specs_are_independent_compilable_specs_without_recovery_instructions():
    from cais_spade_llm.recovery_framework.part_slippage_recovery import _prompt_safety_rules

    paths = [SAFETY + name for name in _MACHINE_RULES]
    assert set(paths) <= set(default_safety_files('Machining breakdown during part processing'))
    for index in range(2):
        rules, dfas = compile_predefined_safety(_machine_document(index))
        assert len(rules) == len(dfas) == 1
    assert [row['requirement'] for row in _prompt_safety_rules(paths)] == [
        'At most one resource accessing M2 may occupy its protected area at any time.',
        'RGOCG4-50_Round_4mm must not complete trim with result circle until KET4_Square_4mm has completed trim with result square.']


@pytest.mark.parametrize('completions,expected', [
    ([(False, False)], 'FEASIBLE'),
    ([(False, False), (True, False), (True, True)], 'FEASIBLE'),
    ([(False, False), (False, True)], 'INFEASIBLE'),
    ([(False, False), (True, True)], 'INFEASIBLE'),
])
def test_machine_precedence_checks_exact_processing_completion_across_transfers(completions, expected):
    assert _machine_check(_processing_observations(completions), 1)['validation_status'] == expected


def test_machine_precedence_rejects_wrong_process_result_and_missing_evidence():
    projection = _processing_observations([(False, False), (True, False), (True, True)])
    for row in projection['observations'][1:]:
        row['parts']['KET4_Square_4mm']['processCompleted'] = [{'process': 'trim', 'result': 'circle'}]
    assert _machine_check(projection, 1)['validation_status'] == 'INFEASIBLE'
    projection['observations'][0]['parts']['KET4_Square_4mm']['processCompleted_complete'] = False
    result = _machine_check(projection, 1)
    assert result['validation_status'] == 'NEEDS_CONTEXT'
    assert 'checkpoint evidence' in result['reason']


def test_machine_precedence_retains_history_between_events():
    projection = _processing_observations([(False, False), (True, False), (True, True)])
    first = {**projection, 'observations': projection['observations'][:2]}
    accepted = _machine_check(first, 1)
    assert accepted['validation_status'] == 'FEASIBLE'
    second = {**projection, 'observations': projection['observations'][1:]}
    result = _machine_check(second, 1, monitor_context=accepted['monitor_context_after'])
    assert result['validation_status'] == 'FEASIBLE'
    assert len(result['monitor_context_after']['observations']) == 3
    assert result['execution_authorized'] is False


def test_machine_mutex_binds_alternatives_across_openings_and_excludes_fixed_fixture():
    projection = _machine_projection()
    row = projection['observations'][0]
    row['region_occupancy']['M2.front_access']['ur5e-1'] = True
    result = _machine_check(projection)
    assert result['validation_status'] == 'FEASIBLE'
    population = result['coverage']['scope_populations'][0]
    assert population['resources'] == ['KMR', 'ur5e-1', 'ur5e-2']
    assert [item['resource_id'] for item in population['excluded_resources']] == ['M2']
    assert len(result['rule_checks']) >= 3
    row['region_occupancy']['M2.side_access']['KMR'] = True
    assert _machine_check(projection)['validation_status'] == 'INFEASIBLE'


def test_machine_mutex_preserves_occupancy_uncertainty_and_missing_population():
    projection = _machine_projection()
    row = projection['observations'][0]
    row['region_occupancy']['M2.front_access']['ur5e-2'] = True
    row['occupancy_possibilities'] = {
        region: {rid: [inside] for rid, inside in values.items()}
        for region, values in row['region_occupancy'].items()}
    row['region_occupancy']['M2.side_access']['KMR'] = None
    row['occupancy_possibilities']['M2.side_access']['KMR'] = [False, True]
    assert _machine_check(projection)['validation_status'] == 'NEEDS_CONTEXT'
    projection = _machine_projection()
    del projection['geometry']['resources']['M2']['stationary_only']
    result = _machine_check(projection)
    assert result['validation_status'] == 'NEEDS_CONTEXT'
    assert 'participant declaration is missing' in result['reason']
    projection['geometry']['resources'].pop('M2')
    assert _machine_check(projection)['validation_status'] == 'NEEDS_CONTEXT'


def test_machine_mutex_rejects_false_stationary_exclusion_and_missing_region():
    projection = _machine_projection()
    projection['geometry']['resources']['KMR']['stationary_only'] = True
    assert _machine_check(projection)['validation_status'] == 'NEEDS_CONTEXT'
    projection = _machine_projection()
    projection['geometry']['regions'].pop('M2.side_access')
    assert _machine_check(projection)['validation_status'] == 'NEEDS_CONTEXT'


def test_machine_mutex_counts_carried_part_geometry_with_clear_robot_body():
    from cais_spade_llm.resources.recovery_geometry import _checkpoint

    projection = _machine_projection()
    snapshot = projection['observations'][0]
    geometry = projection['geometry']
    models = {}
    for resource, offset in [('ur5e-2', -1), ('KMR', .5), ('ur5e-1', -3)]:
        models[resource] = {'configuration': {
            'joint_names': ['slide'], 'interpolation': 'splines', 'root': 'world',
            'root_xyz': [offset, .5, .5], 'root_rpy': [0, 0, 0], 'reference_link': 'body',
            'joints': [{'name': 'slide', 'parent': 'world', 'child': 'body', 'type': 'prismatic',
                        'xyz': [0, 0, 0], 'rpy': [0, 0, 0], 'axis': [1, 0, 0]}],
            'components': [{'id': 'body_box', 'link': 'body', 'bounds': [[-.05, .05]] * 3}]}}
        snapshot['resources'][resource].update(joint_names=['slide'], joint_positions=[0],
                                              current_pose=[offset, .5, .5, 0, 0, 0, 1], grasp_transform=None)
    geometry['resources']['M2']['component_bounds'] = [{'id': 'fixture', 'bounds': [[.1, .2]] * 3}]
    part = 'RGOCG4-50_Round_4mm'
    snapshot['parts'] = {part: {'current_pose': [-2, .5, .5, 0, 0, 0, 1]}}
    geometry['parts'] = {part: {'local_bounds': [[-.05, .05]] * 3}}
    clear = _checkpoint(snapshot, geometry, models, index=0)
    assert clear['region_occupancy']['M2.front_access']['ur5e-2'] is False
    projection['observations'] = [clear]
    assert _machine_check(projection)['validation_status'] == 'FEASIBLE'
    snapshot['resources']['ur5e-2'].update(held_part=part, grasp_transform=[1.5, 0, 0, 0, 0, 0, 1])
    carried = _checkpoint(snapshot, geometry, models, index=0)
    assert carried['region_occupancy']['M2.front_access']['ur5e-2'] is True
    projection['observations'] = [carried]
    assert _machine_check(projection)['validation_status'] == 'INFEASIBLE'


def test_machine_mutex_owner_declarations_and_monitor_history_are_bound():
    projection = _machine_projection()
    models = {rid: {'state_variables': {'held_part': {}}} for rid in ('KMR', 'ur5e-2', 'ur5e-1')}
    models['M2'] = {'state_variables': {'resource_state': {}}}
    projection['owner_safety_evidence'] = {'resource_models': models}
    for resource in ('KMR', 'ur5e-2', 'ur5e-1'):
        projection['geometry']['resources'][resource].pop('stationary_only')
    accepted = _machine_check(projection)
    assert accepted['validation_status'] == 'FEASIBLE'
    changed = deepcopy(projection)
    changed['owner_safety_evidence']['resource_models']['KMR']['state_variables']['extra'] = {}
    result = _machine_check(changed, monitor_context=accepted['monitor_context_after'])
    assert result['validation_status'] == 'NEEDS_CONTEXT'
    assert 'monitor_context_after' not in result
    projection['owner_safety_evidence']['resource_models'].pop('KMR')
    assert _machine_check(projection)['validation_status'] == 'NEEDS_CONTEXT'


def test_part_slippage_defaults_use_the_plant_wide_set():
    assert default_safety_files('Part slippage') == PLANT_SAFETY_FILES
    with pytest.raises(ValueError):
        default_safety_files('part slippage')


def _case_provider_input(*, held=None):
    model = {'joint_names':['joint_a1'], 'interpolation':'splines', 'root':'base',
        'root_xyz':[0,0,0], 'root_rpy':[0,0,0], 'reference_link':'body',
        'joints':[{'name':'joint_a1','parent':'base','child':'body','type':'prismatic',
                   'xyz':[0,0,0],'rpy':[0,0,0],'axis':[1,0,0]}],
        'components':[{'id':'body_box','link':'body','bounds':[[-.05,.05]]*3}]}
    configuration = {'arm_joint_names':['joint_a1'], 'parked_arm_configuration':[0],
        'base_control':{'arm_parked_tolerance_rad':.075,'position_tolerance_m':.005,'yaw_tolerance_rad':.035},
        'minimum_clearance_m':.01}
    state = {'joint_names':['joint_a1'],'joint_positions':[0], 'base_pose':[0,0,0,0,0,0,1],
             'held_part':held, 'grasp_transform':[0,0,0,0,0,0,1]}
    observation = {'frame':'world','resources':{'KMR':state},'parts':{},'phase':'between',
                   'time_exact':'0','continuous_interval':['0','1']}
    if held:
        observation['parts'][held] = {'current_pose':[0,0,0,0,0,0,1], 'contained_by':None}
    physical = {'snapshot':deepcopy(observation), 'resources':{'KMR':{}},
        'geometry':{'resources':{'KMR':{}},'parts':{held:{'local_bounds':[[-.02,.02]]*3}} if held else {}},
        'resource_models':{'KMR':{'configuration':model}},
        'physics_evidence':{'models':{'KMR':{'joints':{'joint_a1':{'position':0,'velocity':0}},
            'links':{'base':{'pose':[0,0,0,0,0,0,1],'linear_velocity':[0,0,0],'angular_velocity':[0,0,0]}}}}}}
    trajectory = {'joint_names':['joint_a1'],'duration_ns':1_000_000_000,'points':[
        {'positions':[0.], 'velocities':[0.], 'time_from_start':{'sec':0,'nanosec':0}},
        {'positions':[1.], 'velocities':[0.], 'time_from_start':{'sec':1,'nanosec':0}}]}
    return observation,physical,configuration,trajectory


def test_case_provider_calculates_spline_interior_without_claiming_base_hold():
    from cais_spade_llm.agents.central_controller.recovery_case_evidence import calculate_case_resource_evidence
    observation,physical,configuration,trajectory = _case_provider_input()
    before=deepcopy((observation,physical,configuration,trajectory))
    record=calculate_case_resource_evidence(resource_id='KMR',observation=observation,
        physical_evidence=physical,configuration=configuration,joint_trajectory=trajectory)
    low,high=record['joint_velocity_bounds']['joint_a1']
    assert low <= 0 and high >= 1.5  # Endpoint velocity is zero; the interior moves.
    assert 'base_velocity_bounds' not in record
    assert '/resources/KMR/base_hold_evidence' in record['missing_facts']
    assert (observation,physical,configuration,trajectory)==before
    observation['continuous_interval']=['1/4','3/4']
    interior=calculate_case_resource_evidence(resource_id='KMR',observation=observation,
        physical_evidence=physical,configuration=configuration,joint_trajectory=trajectory)
    assert interior['joint_velocity_bounds']['joint_a1'][0] > 0


def test_case_provider_measured_point_requires_exact_sample_and_velocities():
    from cais_spade_llm.agents.central_controller.recovery_case_evidence import calculate_case_resource_evidence
    observation,physical,configuration,_ = _case_provider_input()
    observation.update(phase='at')
    observation.pop('continuous_interval')
    record=calculate_case_resource_evidence(resource_id='KMR',observation=observation,
        physical_evidence=physical,configuration=configuration,docking_target='M1',
        scene_configuration={'machines':[{'resource_id':'M1','KMR_docking_pose':[0,0,0,0,0,0]}]})
    assert record['base_velocity_bounds']=={'x':[0.,0.],'y':[0.,0.],'yaw':[0.,0.]}
    assert record['docking_targets']=={'M1':[0,0,0,0,0,0]}
    physical['physics_evidence']['models']['KMR']['links']['base'].pop('linear_velocity')
    physical['physics_evidence']['models']['KMR']['joints']['joint_a1']['position']=1
    missing=calculate_case_resource_evidence(resource_id='KMR',observation=observation,
        physical_evidence=physical,configuration=configuration)
    assert 'joint_position_bounds' not in missing and 'base_velocity_bounds' not in missing
    assert '/resources/KMR/joint_position_bounds' in missing['missing_facts']


def test_case_provider_complete_world_geometry_and_attachment_clearance():
    from cais_spade_llm.agents.central_controller.recovery_case_evidence import calculate_case_resource_evidence
    observation,physical,configuration,trajectory = _case_provider_input(held='part')
    pose={'position':{'x':3.,'y':0.,'z':0.},'orientation':{'x':0.,'y':0.,'z':0.,'w':1.}}
    scene={'world':{'collision_objects':[{'id':'obstacle','header':{'frame_id':'world'},
        'primitives':[{'type':1,'dimensions':[.1,.1,.1]}],'primitive_poses':[pose]}]},
        'robot_state':{'attached_collision_objects':[{'object':{'id':'part/link/collision'},'touch_links':['body']}]}}
    physical['scene_object_bindings']={'part':{'objects':[{'object_id':'part/link/collision'}]}}
    record=calculate_case_resource_evidence(resource_id='KMR',observation=observation,
        physical_evidence=physical,configuration=configuration,joint_trajectory=trajectory,planning_scene=scene)
    assert record['collision_object_ids']==['obstacle']
    assert record['carried_part_clearance']['pairs_complete'] is True
    assert record['carried_part_clearance']['pairs'][0]['distance_bounds'][0] > 1
    scene['world']['collision_objects'][0]['planes']=[{}]
    with pytest.raises(ValueError,match='unsupported'):
        calculate_case_resource_evidence(resource_id='KMR',observation=observation,
            physical_evidence=physical,configuration=configuration,joint_trajectory=trajectory,planning_scene=scene)


def test_owner_ledger_cannot_disagree_with_present_physical_state():
    projection=_projection()
    projection['observations'][-1]['resources']['M1']['resource_state']='loaded'
    assert _check(projection)['validation_status']=='NEEDS_CONTEXT'


def test_machine_interlock_uses_declared_task_interval_not_processing_state():
    original = _projection()
    assert _check(original)['validation_status'] == 'FEASIBLE'
    blocked = _check(_projection(True))
    assert blocked['validation_status'] == 'INFEASIBLE'
    assert blocked['counterexample']['ap_values'] == {'ap001': True, 'ap002': True}
    assert 'monitor_context_after' not in blocked
    assert 'owner_safety_evidence' not in original['observations'][0]


@pytest.mark.parametrize('mutation', ['missing_ledger','partial_ledger','stale_state','changed_type','missing_resource','crossed_boundary','missing_interval'])
def test_machine_interlock_incomplete_owner_evidence_never_passes(mutation):
    projection = _projection()
    evidence = projection['owner_safety_evidence']
    if mutation == 'missing_ledger': evidence['task_evidence'] = None
    elif mutation == 'partial_ledger': evidence['task_evidence']['complete'] = False
    elif mutation == 'stale_state': projection['observations'][1]['state_snapshot']['M1']['resource_state'] = 'completed'
    elif mutation == 'changed_type': evidence['state_evidence']['initial'][0]['values']['resource_state'] = None
    elif mutation == 'missing_resource': evidence['state_evidence']['initial'].pop()
    elif mutation == 'crossed_boundary': evidence['task_evidence']['events'][0]['start_time'] = .25
    else: projection['interval_complete'] = False
    result = _check(projection)
    assert result['validation_status'] == 'NEEDS_CONTEXT'
    assert not result['is_safe']
    assert 'monitor_context_after' not in result


def test_structured_monitor_history_is_replayed_and_stale_bindings_fail():
    projection = _projection()
    first = _check(projection)
    assert first['is_safe']
    context = first['monitor_context_after']
    changed = deepcopy(context)
    changed['observations'][0]['state_snapshot']['M1']['resource_state'] = 'completed'
    result = _check(projection, monitor_context=changed)
    assert result['validation_status'] == 'NEEDS_CONTEXT'
    assert 'another physical checkpoint' in result['reason']
    assert context == first['monitor_context_after']


def _capacity(alternatives):
    projection = _projection()
    projection.pop('owner_safety_evidence')
    region = 'M1.staging'
    projection['geometry']['regions'] = {region: {'bounds': [[0,1],[0,1],[0,1]], 'capacity': 1}}
    projection['geometry']['parts'] = {name: {} for name in alternatives}
    for observation in projection['observations']:
        observation['parts'] = {name: {} for name in alternatives}
        observation['part_occupancy_possibilities'] = {region: alternatives}
    return projection


@pytest.mark.parametrize('alternatives,expected', [({'a':[True],'b':[False]},'FEASIBLE'),
    ({'a':[True],'b':[True]},'INFEASIBLE'), ({'a':[True],'b':[False,True]},'NEEDS_CONTEXT')])
def test_capacity_counts_full_part_population_and_preserves_uncertainty(alternatives, expected):
    assert _check(_capacity(alternatives), 'SAFE_M1_staging_capacity')['validation_status'] == expected


def test_capacity_requires_exact_declared_capacity_and_complete_population():
    projection = _capacity({'a':[True]})
    projection['geometry']['regions']['M1.staging']['capacity'] = 2
    assert _check(projection, 'SAFE_M1_staging_capacity')['validation_status'] == 'NEEDS_CONTEXT'
    projection['geometry']['regions']['M1.staging']['capacity'] = 1
    projection['observations'][0]['parts']['omitted'] = {}
    assert _check(projection, 'SAFE_M1_staging_capacity')['validation_status'] == 'NEEDS_CONTEXT'


def _raw(resource='KMR'):
    observation = {'time_exact':'0','phase':'at','frame':'world',
                   'resources': {resource:{'held_part':None},'M1':{'held_part':None}},
                   'parts': {'part':{}}, 'state_snapshot': {}}
    configuration = {'arm_joint_names':['joint_a1','joint_a2'], 'parked_arm_configuration':[0,1],
                     'base_control':{'arm_parked_tolerance_rad':.075,'position_tolerance_m':.005,'yaw_tolerance_rad':.035},
                     'minimum_clearance_m':.01, 'handoff_functions':['transfer_part'],
                     'clearance_resources':[resource]}
    record = {'source': {'kind':'projected','fingerprint':'captured-source'},'interval':['0','0'],
              'configuration':configuration,'base_velocity_bounds':{axis:[0,0] for axis in ('x','y','yaw')},
              'joint_position_bounds':{'joint_a1':[0,0],'joint_a2':[1,1]},
              'joint_velocity_bounds':{'joint_a1':[0,0],'joint_a2':[0,0]},
              'base_pose_bounds':{'x':[0,0],'y':[0,0],'yaw':[0,0]},
              'docking_target':'M1','docking_targets':{'M1':[0,0,0,0,0,0]},'held_part':None,
              'collision_object_ids':['obstacle'], 'handoffs':{'complete':True,'records':[]}}
    checkpoint = {key:observation[key] for key in ('resources','parts','frame','state_snapshot')}
    observation['owner_safety_evidence'] = {'observation_fingerprint':_digest(checkpoint),
        'resource_evidence':{resource:record},'resource_configurations':{resource:deepcopy(configuration)},
        'collision_object_ids':['obstacle'],'task_evidence':{'events':[]}}
    return observation, record


@pytest.mark.parametrize('kind,expected', [('base_moving',[False]),('arm_moving',[False]),
    ('arm_parked',[True]),('docked',[True]),('carried_part_clearance',[True])])
def test_kmr_predicates_use_recorded_bounds(kind,expected):
    observation,_ = _raw()
    assert case_ap_value(kind,'KMR',observation) == expected


def test_kmr_motion_and_posture_keep_interval_alternatives():
    observation,record = _raw()
    record['base_velocity_bounds']['x'] = [-.01,.01]
    assert case_ap_value('base_moving','KMR',observation) == [False,True]
    record['joint_position_bounds']['joint_a1'] = [0,.1]
    assert case_ap_value('arm_parked','KMR',observation) == [False,True]
    record['joint_position_bounds']['joint_a1'] = [.1,.2]
    assert case_ap_value('arm_parked','KMR',observation) == [False]
    record['base_pose_bounds']['x'] = [.1,.2]
    assert case_ap_value('docked','KMR',observation) == [False]


@pytest.mark.parametrize('mutation', ['stale','point_only','missing_joint','claimed_bool'])
def test_kmr_missing_or_stale_bounds_cannot_be_predicate_values(mutation):
    observation,record = _raw()
    if mutation == 'stale': observation['resources']['KMR']['held_part'] = 'part'
    elif mutation == 'point_only': observation.update(phase='between',continuous_interval=['0','1'])
    elif mutation == 'missing_joint': record['joint_position_bounds'].pop('joint_a2')
    else: record['joint_position_bounds'] = True
    with pytest.raises((ValueError,TypeError)):
        case_ap_value('arm_parked','KMR',observation)


def _handoff():
    observation,record = _raw('Conveyor')
    observation['resources']['M1']['held_part'] = 'part'
    checkpoint = {key:observation[key] for key in ('resources','parts','frame','state_snapshot')}
    observation['owner_safety_evidence']['observation_fingerprint'] = _digest(checkpoint)
    task={'task_id':'transfer','resource_id':'Conveyor','function':'transfer_part','start_time':-1,'end_time':0}
    observation['owner_safety_evidence']['task_evidence']['events'] = [task]
    row={'task_id':'transfer','part_name':'part','source_resource':'Conveyor','destination_resource':'M1',
         'phase':'completed','custody_before':{'Conveyor':['part'],'M1':[]},
         'custody_after':{'Conveyor':[],'M1':['part']},
         'acknowledgement':{'kind':'recorded','source_fingerprint':'sample','task_id':'transfer',
                           'part_name':'part','source_resource':'Conveyor','destination_resource':'M1',
                           'source_departed':True,'destination_arrived':True}}
    record['handoffs']['records'] = [row]
    record['robot_clear']={'pairs_complete':True,'pairs':[{'body':'Conveyor','obstacle':'obstacle','distance_bounds':[.02,.03]}]}
    return observation,record,row


def test_handoff_identity_acknowledgement_clearance_and_complete_pair_inventory():
    observation,record,row = _handoff()
    for kind in ('handoff_identity','handoff_acknowledged','robot_clear'):
        assert case_ap_value(kind,'Conveyor',observation) == [True]
    row['custody_before']['M1'] = ['part']
    assert case_ap_value('handoff_identity','Conveyor',observation) == [False]
    row['acknowledgement']['part_name'] = 'different'
    assert case_ap_value('handoff_acknowledged','Conveyor',observation) == [False]
    record['robot_clear']['pairs'][0]['distance_bounds'] = [-.01,-.001]
    assert case_ap_value('robot_clear','Conveyor',observation) == [False]
    record['robot_clear']['pairs'] = []
    with pytest.raises(ValueError,match='omit'):
        case_ap_value('robot_clear','Conveyor',observation)


@pytest.mark.parametrize('mutation', ['missing','wrong_task','command_argument','projection_without_effect'])
def test_completed_handoff_missing_evidence_stays_unresolved(mutation):
    observation,record,row = _handoff()
    if mutation == 'missing': row.pop('acknowledgement')
    elif mutation == 'wrong_task': record['handoffs']['records'] = []
    elif mutation == 'command_argument': row['acknowledgement']['kind'] = 'params'
    else: row['acknowledgement']['kind'] = 'projected'
    with pytest.raises(ValueError):
        case_ap_value('handoff_acknowledged','Conveyor',observation)


def test_scene_regions_and_M2_program_option_preserve_existing_configuration():
    scene=json.loads((ROOT/'cais_spade_llm/initialization/recovery_framework_gazebo.json').read_text())
    machines={m['resource_id']:m for m in scene['machines']}
    assert machines['M2']['program_options']['square'] == machines['M1']['current_configuration']
    assert machines['M2']['current_configuration']['program']['effects'] == [{'process':'trim','result':'circle'}]
    regions=scene['safety_preparation']['regions']
    assert regions['M1.side_access']['source']['link'] == 'side_access'
    assert regions['M1.side_access']['source']['clearance_m'] == machines['M1']['KMR_process_clearance_m']
    for name in ('M1.staging','M2.staging','Buffer For Machined parts.zone_1','Conveyor.loading_position_2'):
        assert regions[name]['capacity'] == 1


def test_parameterized_state_declarations_use_exact_modeled_part_symbols():
    projection = _projection()
    owner = projection['owner_safety_evidence']
    owner['resource_models']['M1']['state_variables']['inventory.{part_name}'] = {'type':'boolean'}
    owner['state_evidence']['initial'][0]['values']['inventory.part_with.dot'] = True
    projection['geometry']['parts']['part_with.dot'] = {}
    for observation in projection['observations']:
        observation['parts']['part_with.dot'] = {}
        observation['state_snapshot']['M1']['inventory.part_with.dot'] = True
    result = _check(projection)
    assert result['validation_status'] == 'FEASIBLE'
    assert 'inventory.{part_name}' in owner['resource_models']['M1']['state_variables']
    assert 'inventory.part_with.dot' in result['projection']['owner_safety_evidence']['resource_models']['M1']['state_variables']
    projection['geometry']['parts'].clear()
    assert _check(projection)['validation_status'] == 'NEEDS_CONTEXT'


def test_owner_config_and_handoff_custody_cannot_change_under_existing_binding():
    observation,record = _raw()
    record['configuration']['base_control']['arm_parked_tolerance_rad'] = 100
    with pytest.raises(ValueError,match='configuration'):
        case_ap_value('arm_parked','KMR',observation)
    observation,record,row = _handoff()
    row['custody_after']['M1'] = []
    with pytest.raises(ValueError,match='physical checkpoint'):
        case_ap_value('handoff_identity','Conveyor',observation)
    row['phase'] = 'pending'
    with pytest.raises(ValueError,match='completion time'):
        case_ap_value('handoff_acknowledged','Conveyor',observation)


def _validated_builder_input():
    projection = _projection()
    ledger = projection.pop('owner_safety_evidence')
    before = {row['resource_id']:deepcopy(row['values']) for row in ledger['state_evidence']['initial']}
    after = deepcopy(projection['observations'][-1]['state_snapshot'])
    projection['observations'][-1]['resources']['M1']['resource_state'] = 'completed'
    projection['successor_evidence'] = {'environment_valuation':after}
    task={'resource_jid':'machine@localhost','event_name':'independently_proposed_identifier'}
    projection['provenance']={'binding':{'resource_jid':task['resource_jid']},'contract_fingerprint':'checked'}
    physical={'environment_models':ledger['resource_models'],'environment_valuation':before,
              'jids':{'M1':'machine@localhost','ur5e-1':'robot@localhost'},
              'resources':{'M1':{'idle':True},'ur5e-1':{'idle':True}}}
    return projection,task,physical


def test_validated_owner_builder_preserves_exact_identifier_and_complete_states():
    from cais_spade_llm.agents.central_controller.outline_evidence_safety import bind_validated_outline_projection
    projection,task,physical = _validated_builder_input()
    before = deepcopy(projection)
    result=bind_validated_outline_projection(projection,task=task,physical_evidence=physical)
    assert result['owner_safety_evidence']['task_evidence']['events'][0]['function'] == task['event_name']
    assert result['owner_safety_evidence']['state_evidence']['updates'][0]['values'] == {'resource_state':'completed'}
    assert projection == before


@pytest.mark.parametrize('change',['intermediate','missing_field','peer_update','peer_activity','identity'])
def test_validated_owner_builder_declines_incomplete_effects(change):
    from cais_spade_llm.agents.central_controller.outline_evidence_safety import bind_validated_outline_projection
    projection,task,physical = _validated_builder_input()
    if change=='intermediate': projection['observations'][1]['resources']['M1']['resource_state']='completed'
    elif change=='missing_field': projection['observations'][1]['resources']['M1'].pop('resource_state')
    elif change=='peer_update':
        projection['successor_evidence']['environment_valuation']['ur5e-1']['held_part']='part'
        projection['observations'][-1]['resources']['ur5e-1']['held_part']='part'
    elif change=='peer_activity': physical['resources']['ur5e-1']['idle']=False
    else: task['resource_jid']='another@localhost'
    with pytest.raises(ValueError):
        bind_validated_outline_projection(projection,task=task,physical_evidence=physical)


def _conveyor_projection(robot='KMR', *, reserved='ur5e-1', stopped=True, inside=True):
    projection=_projection()
    owner=projection['owner_safety_evidence']
    values={'Conveyor':{'belt_stopped':stopped,'loading_reserved_by':reserved},robot:{'held_part':None}}
    owner['resource_models']={
        'Conveyor':{'state_variables':{'belt_stopped':{'type':'boolean'},
                    'loading_reserved_by':{'domain':[None,'ur5e-1','ur5e-2']}}},
        robot:{'state_variables':{'held_part':{'type':['null','string'],'reference':'part_name'}}}}
    owner['task_evidence']['events']=[]
    owner['state_evidence']['updates']=[]
    owner['state_evidence']['initial']=[{'resource_id':rid,'process':'*','product':'*','context':'any','values':deepcopy(row)}
                                       for rid,row in values.items()]
    region='Conveyor.loading_position_1'
    projection['geometry']['resources']={rid:{} for rid in values}
    projection['geometry']['regions']={region:{'bounds':[[0,1],[0,1],[0,1]],'capacity':1}}
    for observation in projection['observations']:
        observation['state_snapshot']=deepcopy(values)
        observation['resources']=deepcopy(values)
        observation['region_occupancy']={region:{'Conveyor':True,robot:inside}}
    return projection


@pytest.mark.parametrize('robot,reserved,stopped,inside,expected',[
    ('ur5e-1','ur5e-1',True,True,'FEASIBLE'),
    ('ur5e-1','ur5e-2',True,True,'INFEASIBLE'),
    ('ur5e-1',None,True,True,'INFEASIBLE'),
    ('ur5e-1','ur5e-1',False,True,'INFEASIBLE'),
    ('KMR','ur5e-1',True,True,'INFEASIBLE'),
    ('ur5e-4','ur5e-1',True,True,'INFEASIBLE'),
    ('KMR',None,False,False,'FEASIBLE'),
])
def test_conveyor_loading_covers_every_robot_without_inventing_reservation_eligibility(robot,reserved,stopped,inside,expected):
    projection=_conveyor_projection(robot,reserved=reserved,stopped=stopped,inside=inside)
    assert _check(projection,'SAFE_Conveyor_loading_position_1_'+robot)['validation_status'] == expected


def test_loading_reserved_state_needs_exact_ledger_and_typed_null():
    projection=_conveyor_projection('ur5e-1',reserved=None)
    del projection['observations'][0]['state_snapshot']['Conveyor']['loading_reserved_by']
    assert _check(projection,'SAFE_Conveyor_loading_position_1_ur5e-1')['validation_status']=='NEEDS_CONTEXT'


def test_public_structured_APs_bind_recorded_owner_identity_without_private_model(monkeypatch):
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes.multi_turn_context import public_outline_safety_context
    from cais_spade_llm.resources import recovery_geometry
    projection=_projection()
    owner=projection['owner_safety_evidence']
    physical={'snapshot':projection['observations'][0],'geometry':projection['geometry'],
              'resource_models':{rid:{} for rid in owner['resource_models']},
              'environment_models':owner['resource_models'],
              'jids':{'M1':'machine@localhost','ur5e-1':'robot@localhost'}}
    monkeypatch.setattr(recovery_geometry,'_checkpoint',lambda *a,**kw:{'occupancy_possibilities':{
        'M1.front_access':{'M1':[True],'ur5e-1':[False]}},'part_occupancy_possibilities':{'M1.front_access':{}}})
    prepared={'physical_evidence':physical,'outline_safety_specifications':[{
        'document':_document('SAFE_M1_front_access_ur5e-1_interlock'),'source_path':'selected','source_sha256':'recorded'}]}
    result=public_outline_safety_context(prepared,{})
    ap=result['instances'][0]['aps'][0]
    assert ap['binding']=={'resource':'M1','process':'*','part':'*'}
    assert 'owner_state_models_fingerprint' in result['source']
    assert not result['missing_context']
    assert 'state_variables' not in json.dumps(result)
    physical.pop('environment_models')
    missing=public_outline_safety_context(prepared,{})
    assert missing['missing_context'] and not missing['instances']


def test_conveyor_motion_profile_overrides_boundary_stopped_without_rewriting_DES():
    projection=_conveyor_projection('ur5e-1',reserved='ur5e-1',stopped=True,inside=True)
    contract={'kind':'resource_owned_shared_transport','displacement_m':.25,'speed_mps':1.,
              'acceleration_mps2':1.,'duration_sec':1.}
    source={'kind':'projected','fingerprint':_digest(contract)}
    projection['owner_interval_contract']=contract
    projection['owner_safety_evidence']['resource_configurations']={'Conveyor':{}}
    region='Conveyor.loading_position_1'
    for observation in projection['observations']:
        observation['owner_safety_evidence']={'resource_evidence':{'Conveyor':{
            'configuration':{},'source':source,
            'interval':observation.get('continuous_interval',[observation['time_exact']]*2),
            'transport_motion':{'source':source,'interval':[0,1],**{k:v for k,v in contract.items() if k!='kind'}}}}}
    middle=projection['observations'][1]
    middle['continuous_cells']=[{'interval':[a,b],
        'occupancy_possibilities':{region:{'Conveyor':[True],'ur5e-1':[True]}}}
        for a,b in [('0','1/4'),('1/4','3/4'),('3/4','1')]]
    result=_check(projection,'SAFE_Conveyor_loading_position_1_ur5e-1')
    assert result['validation_status']=='INFEASIBLE'
    assert result['counterexample']['definite_violation'] is True
    assert result['counterexample']['ap_values']['ap002']==[False]
    assert all(row['state_snapshot']['Conveyor']['belt_stopped'] is True for row in projection['observations'])
    projection['observations'][1]['owner_safety_evidence']['resource_evidence']['Conveyor']['transport_motion']['speed_mps']=.5
    assert _check(projection,'SAFE_Conveyor_loading_position_1_ur5e-1')['validation_status']=='NEEDS_CONTEXT'


def test_case_provider_does_not_reuse_captured_velocities_after_projection():
    from cais_spade_llm.agents.central_controller.recovery_case_evidence import calculate_case_resource_evidence
    observation,physical,configuration,_=_case_provider_input()
    observation['phase']='at'
    physical['provenance']={'kind':'projected'}
    result=calculate_case_resource_evidence(resource_id='KMR',observation=observation,
        physical_evidence=physical,configuration=configuration)
    assert 'base_velocity_bounds' not in result and 'joint_velocity_bounds' not in result


def test_conditional_handoff_inventory_uses_explicit_null_and_exact_owner_declaration():
    observation,record=_raw('Conveyor')
    event={'event_id':78,'event_name':'advance_conveyor','parameter_bindings':{'delivered_part':{'equals':None}},
           'participants':['Conveyor','Buffer For Machined parts'],
           'collection_effects':{'validation_source':'cais_spade_llm.resources.nominal_conveyor'}}
    record['handoff_declarations']=[event]
    task={'task_id':'shared','resource_id':'Conveyor','function':'advance_conveyor','start_time':-1,'end_time':0,
          'declared_task':{'resource_id':'Conveyor','event_id':78,'event_name':'advance_conveyor','parameters':{'delivered_part':None}}}
    observation['owner_safety_evidence']['task_evidence']['events']=[task]
    observation['owner_safety_evidence']['resource_event_fingerprints']={'Conveyor':_digest([event])}
    assert case_ap_value('handoff_identity','Conveyor',observation)==[True]
    assert case_ap_value('handoff_acknowledged','Conveyor',observation)==[True]
    task['declared_task']['parameters'].pop('delivered_part')
    with pytest.raises(ValueError,match='explicit delivered_part'):
        case_ap_value('handoff_identity','Conveyor',observation)


def test_per_checkpoint_collision_inventory_is_preserved():
    projection=_projection()
    for index,point in enumerate(projection['observations']):
        point['owner_safety_evidence']={'collision_object_ids':['recorded-'+str(index)]}
    result=bind_outline_owner_safety_evidence(projection)
    assert [row['owner_safety_evidence']['collision_object_ids'] for row in result['observations']]==[
        ['recorded-0'],['recorded-1'],['recorded-2']]


def _custody_overlay_projection():
    projection=_projection()
    part='part'
    projection['geometry']['parts']={part:{}}
    owner=projection['owner_safety_evidence']
    owner['resource_models']['ur5e-1']['state_variables']['gripper_state']={'domain':['open','closed']}
    owner['state_evidence']['initial'][1]['values']['gripper_state']='open'
    step={'id':'grasp','op':'grasp_part','params':{'part_name':part}}
    projection['owner_interval_contract']={'owner_declaration':{'program':{'steps':[step]}},
        'declared_task':{'resource_id':'ur5e-1','parameters':{'part_name':part}},'preview_fingerprint':'native-program'}
    for index,point in enumerate(projection['observations']):
        point['parts']={part:{}}
        point['resources']['ur5e-1']['gripper_state']='open' if index==0 else 'closed'
        point['state_snapshot']['ur5e-1']['gripper_state']='open'
        if index:
            point['resources']['ur5e-1']['held_part']=part
            point['owner_safety_evidence']={
                'physical_state_updates':{'ur5e-1':{'held_part':part,'gripper_state':'closed'}},
                'physical_state_sources':{'ur5e-1':{'step_fingerprint':_digest(step),'op':'grasp_part',
                    'source_fingerprint':'native-program'}}}
    return projection


def test_supported_intermediate_custody_overlay_does_not_invent_task_duration():
    projection=_custody_overlay_projection()
    before=deepcopy(projection['owner_safety_evidence'])
    result=bind_outline_owner_safety_evidence(projection)
    assert result['observations'][1]['state_snapshot']['ur5e-1']=={'held_part':'part','gripper_state':'closed'}
    assert result['owner_safety_evidence']['task_evidence']==before['task_evidence']
    assert result['owner_safety_evidence']['state_evidence']==before['state_evidence']
    assert projection['observations'][1]['state_snapshot']['ur5e-1']['held_part'] is None


@pytest.mark.parametrize('change',['source','step','part','operation','physical'])
def test_custody_overlay_rejects_unbound_or_conflicting_effect(change):
    projection=_custody_overlay_projection()
    point=projection['observations'][1]
    supplied=point['owner_safety_evidence']
    if change=='source': supplied['physical_state_sources']['ur5e-1']['source_fingerprint']='different'
    elif change=='step': supplied['physical_state_sources']['ur5e-1']['step_fingerprint']='different'
    elif change=='part': supplied['physical_state_updates']['ur5e-1']['held_part']=None
    elif change=='operation': supplied['physical_state_sources']['ur5e-1']['op']='close_gripper'
    else: point['resources']['ur5e-1']['held_part']=None
    with pytest.raises(ValueError): bind_outline_owner_safety_evidence(projection)


def _base_hold_input():
    from hashlib import sha256
    observation,physical,configuration,trajectory=_case_provider_input()
    description='<robot><gazebo reference="base"><kinematic>true</kinematic><gravity>false</gravity></gazebo><gazebo><plugin filename="libgazebo_ros_planar_move.so"><robot_base_frame>base</robot_base_frame><ros><namespace>/KMR</namespace></ros></plugin></gazebo></robot>'
    model=physical['resource_models']['KMR']['configuration']
    model.update(root_quaternion=[0,0,0,1],description_sha256=sha256(description.encode()).hexdigest())
    hold={'version':1,'robot_description':description,'controller_source_sha256':'a'*64,
          'root_link':'base','root_pose':[0,0,0,0,0,0,1], 'linear_velocity':[0.,0.,0.],
          'angular_velocity':[0.,0.,0.], 'physics_source':{'simulation_time':1.},'launch_id':'launch','max_age':2.,
          'controller_observation':{'controller_source_sha256':'a'*64,'controller_node':'/KMR_base_controller',
              'source':'/KMR/recovery_state','has_active_goal':False,'has_pending_goal':False,
              'last_output':[0.,0.,0.],'inactive_goal_output':[0.,0.,0.],'simulation_time':1.,'instance_id':'launch'}}
    physical['resources']['KMR']['base_hold_evidence']=hold
    return observation,physical,configuration,trajectory


def test_base_hold_is_calculated_from_exact_controller_and_kinematic_model():
    from cais_spade_llm.agents.central_controller.recovery_case_evidence import calculate_case_resource_evidence
    observation,physical,configuration,trajectory=_base_hold_input()
    result=calculate_case_resource_evidence(resource_id='KMR',observation=observation,
        physical_evidence=physical,configuration=configuration,joint_trajectory=trajectory)
    assert result['base_velocity_bounds']=={axis:[0.,0.] for axis in ('x','y','yaw')}
    assert result['source']['base_hold']['physical_execution_verified'] is False
    assert 'No new base goals' in result['source']['base_hold']['assumptions'][0]


def _base_hold_separate_descriptions():
    from hashlib import sha256
    from xml.etree import ElementTree

    observation, physical, configuration, trajectory = _base_hold_input()
    hold = physical['resources']['KMR']['base_hold_evidence']
    tree = ElementTree.fromstring(hold['robot_description'])
    tree.set('name', 'KMR')
    tree.insert(0, ElementTree.fromstring('<link name="base"><collision><geometry><box size="1 1 1"/></geometry></collision></link>'))
    tree.insert(1, ElementTree.fromstring('<link name="tool"/>'))
    tree.insert(2, ElementTree.fromstring('<joint name="joint_a1" type="revolute"><parent link="base"/><child link="tool"/><axis xyz="0 0 1"/><limit lower="-2" upper="2" effort="1" velocity="1"/></joint>'))
    physics = ElementTree.tostring(tree, encoding='unicode')
    for element in tree.findall('gazebo'):
        tree.remove(element)
    tree.set('name', 'dual_robot')
    planning = ElementTree.tostring(tree, encoding='unicode')
    hold.update(version=2, robot_description=planning, physics_robot_description=physics,
        physics_description_node='/KMR/robot_state_publisher', physics_description_sha256=sha256(physics.encode()).hexdigest())
    controller = hold['controller_observation']
    owned = ['/KMR/dock/_action/status', '/KMR/move_base/_action/status',
        '/KMR/validated_navigate_to_pose/_action/status', '/navigate_to_pose/_action/status',
        '/KMR/validated_follow_path/_action/status']
    gated = ['/KMR/navigate_to_pose/_action/status', '/KMR/follow_path/_action/status']
    controller['owned_activity'] = {'has_active_goal':False, 'has_pending_goal':False, 'holding':True}
    controller['command_gate'] = {'version':1, 'closed':True, 'input_topic':'/KMR/nav_cmd_vel',
        'output_topic':'/KMR/cmd_vel', 'owned_status_topics':owned, 'gated_status_topics':gated}
    controller['command_gate_evidence'] = {
        'status':'supported', 'source_sha256':controller['controller_source_sha256'],
        'controller_node':controller['controller_node'], 'controller_instance_id':controller['instance_id'],
        'output_publisher_gid':[1], 'output_publisher_node':controller['controller_node'], 'output_publisher_count':1,
        'owned_activity':deepcopy(controller['owned_activity']), 'command_gate':deepcopy(controller['command_gate']),
        'declaration':{'input_topic':'/KMR/nav_cmd_vel', 'output_topic':'/KMR/cmd_vel',
            'owned_status_topics':owned, 'status_topics':gated, 'controller_node':controller['controller_node'],
            'source_package':'cais_lab_robotics', 'source_executable':'kmr_base_controller.py'},
        'nav2_goal_observations':{topic:{'status':'unknown'} for topic in gated},
        'all_nav2_goals_idle':None, 'physical_execution_verified':False,
    }
    physical['resource_models']['KMR']['configuration']['description_sha256'] = sha256(planning.encode()).hexdigest()
    return observation, physical, configuration, trajectory


def test_base_hold_uses_physical_description_bound_to_same_planning_subtree():
    from cais_spade_llm.agents.central_controller.recovery_case_evidence import recorded_base_hold

    _, physical, _, _ = _base_hold_separate_descriptions()
    before = deepcopy(physical)
    hold = physical['resources']['KMR']['base_hold_evidence']
    assert '<gazebo' not in hold['robot_description']
    result = recorded_base_hold(resource_id='KMR', physical_evidence=physical,
        model=physical['resource_models']['KMR']['configuration'])
    assert result['base_velocity_bounds'] == {axis:[0.,0.] for axis in ('x','y','yaw')}
    assert result['source']['physical_execution_verified'] is False
    assert physical == before


def _capture_with_checked_base_map():
    from hashlib import sha256
    from xml.etree import ElementTree
    from cais_spade_llm.recovery_framework import evaluation_conditions

    _, physical, _, _ = _base_hold_separate_descriptions()
    hold = physical['resources']['KMR']['base_hold_evidence']
    tree = ElementTree.fromstring(hold['physics_robot_description'])
    tree.find("link[@name='base']").set('name', 'KMR_base_link')
    tree.find("joint[@name='joint_a1']/parent").set('link', 'KMR_base_link')
    tree.find("gazebo[@reference='base']").set('reference', 'KMR_base_link')
    tree.find('gazebo/plugin/robot_base_frame').text = 'KMR_base_link'
    hold['physics_robot_description'] = ElementTree.tostring(tree, encoding='unicode')
    hold['physics_description_sha256'] = sha256(hold['physics_robot_description'].encode()).hexdigest()
    for row in tree.findall('gazebo'):
        tree.remove(row)
    tree.set('name', 'dual_robot')
    for name in ('world', 'KMR_base_x_link', 'KMR_base_y_link'):
        ElementTree.SubElement(tree, 'link', {'name':name})
    for name, kind, parent, child, axis in (
        ('KMR_base_x_joint', 'prismatic', 'world', 'KMR_base_x_link', '1 0 0'),
        ('KMR_base_y_joint', 'prismatic', 'KMR_base_x_link', 'KMR_base_y_link', '0 1 0'),
        ('KMR_base_yaw_joint', 'continuous', 'KMR_base_y_link', 'KMR_base_link', '0 0 1'),
    ):
        joint = ElementTree.SubElement(tree, 'joint', {'name':name, 'type':kind})
        ElementTree.SubElement(joint, 'parent', {'link':parent})
        ElementTree.SubElement(joint, 'child', {'link':child})
        ElementTree.SubElement(joint, 'axis', {'xyz':axis})
        limits = {'lower':'-20', 'upper':'20'} if kind == 'prismatic' else {}
        ElementTree.SubElement(joint, 'limit', {**limits, 'effort':'100', 'velocity':'1'})
    description = ElementTree.tostring(tree, encoding='unicode')
    hold.update(robot_description=description, root_link='KMR_base_link')
    model = physical['resource_models']['KMR']['configuration']
    model.update(root='KMR_base_link', description_sha256=sha256(description.encode()).hexdigest())
    source = evaluation_conditions.ROOT/'ros2/cais_lab_robotics/scripts/kmr_base_controller.py'
    source_sha256 = sha256(source.read_bytes()).hexdigest()
    hold['controller_source_sha256'] = source_sha256
    controller = hold['controller_observation']
    controller['controller_source_sha256'] = source_sha256
    controller['command_gate_evidence']['source_sha256'] = source_sha256
    controller['occupancy_map'] = {'frame_id':'world', 'stamp':{'sec':1, 'nanosec':0},
        'map_load_time':{'sec':0, 'nanosec':0}, 'width':100, 'height':80, 'resolution':.1,
        'origin':[-5., -4., 0., 0., 0., 0., 1.], 'cell_count':8000, 'cells_sha256':'a'*64}
    return {'physical_evidence':physical, 'planning_scene_capture':{'robot_description':description}}


def test_recorded_base_reach_uses_checked_command_authority_and_physical_model():
    from cais_spade_llm.recovery_framework.evaluation_conditions import _recorded_base_reach

    captured = _capture_with_checked_base_map()
    before = deepcopy(captured)
    result = _recorded_base_reach(captured)
    assert result['map_bounds'] == {'x':[-5.,5.], 'y':[-4.,4.]}
    assert result['planning_joint_limits']['KMR_base_x_joint'] == [-20.,20.]
    assert captured == before


@pytest.mark.parametrize('change', ['authority', 'physical_model', 'planning_hash', 'base_pose',
                                   'yaw_missing', 'yaw_child', 'yaw_axis'])
def test_recorded_base_reach_rejects_unbound_authority_model_or_complete_base_chain(change):
    from hashlib import sha256
    from xml.etree import ElementTree
    from cais_spade_llm.recovery_framework.evaluation_conditions import _recorded_base_reach

    captured = _capture_with_checked_base_map()
    physical = captured['physical_evidence']
    hold = physical['resources']['KMR']['base_hold_evidence']
    if change == 'authority': hold['controller_observation'].pop('command_gate_evidence')
    elif change == 'physical_model': hold.pop('physics_robot_description')
    elif change == 'planning_hash':
        captured['planning_scene_capture']['robot_description'] += '\n'
    elif change == 'base_pose': physical['snapshot']['resources']['KMR']['base_pose'][0] = 1.
    else:
        tree = ElementTree.fromstring(captured['planning_scene_capture']['robot_description'])
        joint = tree.find("joint[@name='KMR_base_yaw_joint']")
        if change == 'yaw_missing': tree.remove(joint)
        elif change == 'yaw_child': joint.find('child').set('link', 'KMR_base_x_link')
        else: joint.find('axis').set('xyz', '1 0 0')
        description = ElementTree.tostring(tree, encoding='unicode')
        captured['planning_scene_capture']['robot_description'] = hold['robot_description'] = description
        physical['resource_models']['KMR']['configuration']['description_sha256'] = sha256(description.encode()).hexdigest()
    with pytest.raises(ValueError, match='NEEDS_CONTEXT'):
        _recorded_base_reach(captured)


@pytest.mark.parametrize('change', ['missing', 'fingerprint', 'owner', 'geometry', 'joint', 'dynamic',
                                   'gravity', 'plugin', 'source', 'duplicate', 'version'])
def test_base_hold_separate_description_rejects_missing_or_incompatible_physical_model(change):
    from hashlib import sha256
    from cais_spade_llm.agents.central_controller.recovery_case_evidence import recorded_base_hold

    _, physical, _, _ = _base_hold_separate_descriptions()
    hold = physical['resources']['KMR']['base_hold_evidence']
    if change == 'missing': hold.pop('physics_robot_description')
    elif change == 'fingerprint': hold['physics_description_sha256'] = 'b'*64
    elif change == 'source': hold['physics_description_node'] = ''
    elif change == 'version': hold['version'] = True
    else:
        old, new = {
            'owner': ('name="KMR"', 'name="another"'),
            'geometry': ('size="1 1 1"', 'size="2 1 1"'),
            'joint': ('xyz="0 0 1"', 'xyz="1 0 0"'),
            'dynamic': ('<kinematic>true', '<kinematic>false'),
            'gravity': ('<gravity>false', '<gravity>true'),
            'plugin': ('libgazebo_ros_planar_move.so', 'another.so'),
            'duplicate': ('</robot>', '<link name="base"/></robot>'),
        }[change]
        hold['physics_robot_description'] = hold['physics_robot_description'].replace(old, new)
        hold['physics_description_sha256'] = sha256(hold['physics_robot_description'].encode()).hexdigest()
    with pytest.raises(ValueError):
        recorded_base_hold(resource_id='KMR', physical_evidence=physical,
            model=physical['resource_models']['KMR']['configuration'])


@pytest.mark.parametrize('external', ['unknown', 'active'])
def test_recorded_command_gate_does_not_relabel_external_nav2_goal_activity(external):
    from cais_spade_llm.agents.central_controller.recovery_case_evidence import recorded_base_hold

    _, physical, _, _ = _base_hold_separate_descriptions()
    controller = physical['resources']['KMR']['base_hold_evidence']['controller_observation']
    controller.update(has_active_goal=True, holding=False, nav2_follow_path_active=True)
    if external == 'active':
        controller['command_gate_evidence']['nav2_goal_observations']['/KMR/follow_path/_action/status'] = {
            'status':'observed', 'publisher_gid':[2], 'goals':[{'id':[3], 'status':2}]}
    before = deepcopy(physical)
    result = recorded_base_hold(resource_id='KMR', physical_evidence=physical,
        model=physical['resource_models']['KMR']['configuration'])
    assert result['base_velocity_bounds'] == {axis:[0.,0.] for axis in ('x','y','yaw')}
    assert physical == before
    assert controller['command_gate_evidence']['all_nav2_goals_idle'] is None


@pytest.mark.parametrize('change', ['missing', 'open', 'source', 'instance', 'owner', 'publisher',
    'publisher_count', 'publisher_boolean', 'owned', 'output', 'zero', 'global_idle', 'declaration', 'plugin'])
def test_recorded_command_gate_requires_exact_authority_and_planar_input(change):
    from hashlib import sha256
    from cais_spade_llm.agents.central_controller.recovery_case_evidence import recorded_base_hold

    _, physical, _, _ = _base_hold_separate_descriptions()
    hold = physical['resources']['KMR']['base_hold_evidence']
    controller = hold['controller_observation']
    proof = controller['command_gate_evidence']
    if change == 'missing': controller.pop('command_gate_evidence')
    elif change == 'open': controller['command_gate']['closed'] = proof['command_gate']['closed'] = False
    elif change == 'source': proof['source_sha256'] = 'b'*64
    elif change == 'instance': proof['controller_instance_id'] = 'another'
    elif change == 'owner': proof['controller_node'] = '/another'
    elif change == 'publisher': proof['output_publisher_node'] = '/another'
    elif change == 'publisher_count': proof['output_publisher_count'] = 2
    elif change == 'publisher_boolean': proof['output_publisher_gid'] = [True]
    elif change == 'owned':
        controller['owned_activity']['has_active_goal'] = proof['owned_activity']['has_active_goal'] = True
    elif change == 'output':
        controller['command_gate']['output_topic'] = proof['command_gate']['output_topic'] = proof['declaration']['output_topic'] = '/another'
    elif change == 'zero': controller['last_output'][0] = .01
    elif change == 'global_idle': proof['all_nav2_goals_idle'] = True
    elif change == 'declaration': proof['declaration']['status_topics'] = []
    else:
        hold['physics_robot_description'] = hold['physics_robot_description'].replace(
            '<namespace>/KMR</namespace>', '<namespace>/KMR</namespace><remapping>cmd_vel:=elsewhere</remapping>')
        hold['physics_description_sha256'] = sha256(hold['physics_robot_description'].encode()).hexdigest()
    with pytest.raises(ValueError):
        recorded_base_hold(resource_id='KMR', physical_evidence=physical,
            model=physical['resource_models']['KMR']['configuration'])


@pytest.mark.parametrize('change',['missing','velocity','command','active','pending','code','model','identity','age','pose','boolean'])
def test_base_hold_missing_or_changed_facts_remain_unresolved(change):
    from cais_spade_llm.agents.central_controller.recovery_case_evidence import calculate_case_resource_evidence
    observation,physical,configuration,trajectory=_base_hold_input()
    hold=physical['resources']['KMR']['base_hold_evidence']
    if change=='missing': physical['resources']['KMR'].pop('base_hold_evidence')
    elif change=='velocity': hold['linear_velocity'][0]=.001
    elif change=='command': hold['controller_observation']['last_output'][0]=.001
    elif change=='active': hold['controller_observation']['has_active_goal']=True
    elif change=='pending': hold['controller_observation']['has_pending_goal']=True
    elif change=='code': hold['controller_observation']['controller_source_sha256']='b'*64
    elif change=='model': hold['robot_description']=hold['robot_description'].replace('<kinematic>true','<kinematic>false')
    elif change=='identity': hold['controller_observation']['instance_id']='different'
    elif change=='age': hold['controller_observation']['simulation_time']=10
    elif change=='pose': physical['resource_models']['KMR']['configuration']['root_xyz'][0]=1
    else: hold['controller_observation']['last_output'][0]=False
    result=calculate_case_resource_evidence(resource_id='KMR',observation=observation,
        physical_evidence=physical,configuration=configuration,joint_trajectory=trajectory)
    assert 'base_velocity_bounds' not in result
    assert '/resources/KMR/base_hold_evidence' in result['missing_facts']


def _stationary_peer_input():
    from hashlib import sha256
    observation, physical, configuration, _ = _base_hold_input()
    hold = physical['resources']['KMR']['base_hold_evidence']
    hold['robot_description'] = hold['robot_description'].replace('<robot>',
        '<robot><joint name="joint_a1"><limit velocity="2.5"/></joint>')
    physical['resource_models']['KMR']['configuration']['description_sha256'] = sha256(hold['robot_description'].encode()).hexdigest()
    state = physical['snapshot']['resources']['KMR']
    owner = physical['resources']['KMR']
    activity = {'holding': True, 'has_active_goal': False, 'has_pending_goal': False}
    owner.update(observation_state=deepcopy(state), idle=True,
        stationary_contract={'kind':'idle_commanded_hold','requires_no_running_tasks':True,
            'requires_no_active_goals':True,'future_execution_tracking':'not_established'},
        controller_hold_observation={**activity, 'instance_id':'launch','joint_names':['joint_a1'],
            'commanded_positions':[0.], 'commanded_positions_source':'native position command interfaces'},
        controller_hold_activity={'arm':deepcopy(activity)},
        controller_hold_model={'value':.01,'kind':'conditional model assumption',
            'parameter':'trajectory_execution.allowed_start_tolerance','physical_execution_verified':False,
            'future_execution_tracking':'not_established'})
    first = deepcopy(observation)
    first.update(phase='at', time_exact='0', state_snapshot={'KMR':{'resource_location':'Storage'}})
    first.pop('continuous_interval')
    between = deepcopy(observation)
    between['state_snapshot'] = deepcopy(first['state_snapshot'])
    last = deepcopy(first)
    last['time_exact'] = '1'
    projection = {'status':'projected','interval_complete':True,'observations':[first,between,last]}
    scene = {'KMR':configuration, 'Storage':{'KMR_docking_pose':[0,0,0,0,0,0]}}
    return projection,physical,scene


def test_stationary_kmr_peer_binds_conditional_bounds_without_inventing_zero_arm_motion():
    from cais_spade_llm.agents.central_controller.recovery_case_evidence import bind_case_projection_evidence
    projection,physical,scene = _stationary_peer_input()
    before = deepcopy((projection,physical,scene))
    result = bind_case_projection_evidence(projection, physical_evidence=physical, scene_configuration=scene)
    first,between,last = [row['owner_safety_evidence']['resource_evidence']['KMR'] for row in result['observations']]
    assert first['source']['kind'] == 'recorded' and first['joint_velocity_bounds'] == {'joint_a1':[0.,0.]}
    assert between['source']['kind'] == last['source']['kind'] == 'projected'
    assert between['joint_velocity_bounds'] == {'joint_a1':[-2.5,2.5]}
    assert between['joint_position_bounds']['joint_a1'][0] < -.01
    assert between['base_velocity_bounds'] == {axis:[0.,0.] for axis in ('x','y','yaw')}
    assert between['docking_targets'] == {'Storage':[0,0,0,0,0,0]}
    assert 'carried_part_clearance' not in between
    assert (projection,physical,scene) == before


@pytest.mark.parametrize('change', ['active', 'base_missing', 'velocity_missing', 'joint_changed', 'base_changed', 'held_changed'])
def test_stationary_kmr_peer_does_not_fill_missing_or_changed_evidence(change):
    from cais_spade_llm.agents.central_controller.recovery_case_evidence import bind_case_projection_evidence
    from hashlib import sha256
    projection,physical,scene = _stationary_peer_input()
    owner = physical['resources']['KMR']
    if change == 'active': owner['controller_hold_observation']['has_active_goal'] = True
    elif change == 'base_missing': owner.pop('base_hold_evidence')
    elif change == 'velocity_missing':
        hold = owner['base_hold_evidence']
        hold['robot_description'] = hold['robot_description'].replace(' velocity="2.5"','')
        physical['resource_models']['KMR']['configuration']['description_sha256'] = sha256(hold['robot_description'].encode()).hexdigest()
    elif change == 'joint_changed': projection['observations'][1]['resources']['KMR']['joint_positions'] = [.1]
    elif change == 'base_changed': projection['observations'][1]['resources']['KMR']['base_pose'][0] = .1
    else: projection['observations'][1]['resources']['KMR']['held_part'] = 'part'
    result = bind_case_projection_evidence(projection, physical_evidence=physical, scene_configuration=scene)
    envelope = result['observations'][1]['owner_safety_evidence']
    record = envelope['resource_evidence'].get('KMR', {})
    assert 'joint_velocity_bounds' not in record
    if change == 'base_missing': assert 'base_velocity_bounds' not in record
    if change.endswith('changed'): assert 'KMR' not in envelope['resource_evidence'] and envelope['missing_facts']


def test_stationary_kmr_peer_preserves_supplied_owner_motion_record():
    from cais_spade_llm.agents.central_controller.recovery_case_evidence import bind_case_projection_evidence
    projection,physical,scene = _stationary_peer_input()
    projection['observations'][1]['owner_safety_evidence'] = {'resource_evidence':{'KMR':{'exact':'owner record'}}}
    result = bind_case_projection_evidence(projection, physical_evidence=physical, scene_configuration=scene)
    assert result['observations'][1]['owner_safety_evidence']['resource_evidence']['KMR'] == {'exact':'owner record'}


@pytest.mark.parametrize('changed_code', [False, True])
def test_kmr_capture_records_base_source_and_exact_root_without_commands(tmp_path, monkeypatch, changed_code):
    import sys
    from hashlib import sha256
    from types import ModuleType, SimpleNamespace
    from cais_spade_llm.recovery_framework import kmr_live_safety

    observation,physical,_,_ = _base_hold_separate_descriptions()
    model = physical['resource_models']['KMR']['configuration']
    installed = tmp_path/'lib/cais_lab_robotics/kmr_base_controller.py'
    installed.parent.mkdir(parents=True)
    installed.write_text('captured controller code')
    controller = deepcopy(physical['resources']['KMR']['base_hold_evidence']['controller_observation'])
    controller['controller_source_sha256'] = 'b'*64 if changed_code else sha256(installed.read_bytes()).hexdigest()
    controller['command_gate_evidence']['source_sha256'] = controller['controller_source_sha256']
    description = physical['resources']['KMR']['base_hold_evidence']['robot_description']
    physics_description = physical['resources']['KMR']['base_hold_evidence']['physics_robot_description']
    physics_model = deepcopy(physical['physics_evidence']['models']['KMR'])
    physics_model['pose'] = [99,99,99,0,0,0,1]  # A model origin is not the configured root link.
    record = {'models':{'KMR':physics_model},'instance_id':'physics','simulation_time':1.,'observed_monotonic':1.}
    reads = []
    def controller_state(node):
        reads.append(node)
        return deepcopy(controller)
    reader = SimpleNamespace(parameter=lambda node,name: physics_description if node=='/KMR/robot_state_publisher'
        else description if name=='robot_description' else 'splines',
        snapshot=lambda **kwargs:deepcopy(record),controller_state=controller_state)
    def capture(**kwargs):
        assert kwargs['record'] == record
        return {**deepcopy(kwargs['native']),'observation_state':{'held_part':None}}
    observer = SimpleNamespace(motion_configuration=model,configuration={'model':'KMR','description_node':'description',
        'physics_description_node':'/KMR/robot_state_publisher', 'controller_node':'arm'},
        provider=SimpleNamespace(reader=reader,launch_id='launch'),capture=capture)
    package = ModuleType('ament_index_python')
    packages = ModuleType('ament_index_python.packages')
    packages.get_package_prefix = lambda name:str(tmp_path)
    monkeypatch.setitem(sys.modules,'ament_index_python',package)
    monkeypatch.setitem(sys.modules,'ament_index_python.packages',packages)
    result = kmr_live_safety.capture_state(SimpleNamespace(recovery_safety_observer=observer))
    assert reads == ['/KMR']
    assert result['base_pose'] == result['observation_state']['base_pose'] == [0,0,0,0,0,0,1]
    if changed_code:
        assert result['base_hold_evidence']['status'] == 'NEEDS_CONTEXT'
    else:
        assert result['base_hold_evidence']['version'] == 2
        assert result['base_hold_evidence']['physics_robot_description'] == physics_description
        assert result['base_hold_evidence']['controller_observation'] == controller
        assert result['base_hold_evidence']['physics_source']['instance_id'] == 'physics'


def _transfer_corridor_input():
    from cais_spade_llm.agents.shared_information.recovery_validation_protocol import recovery_validation_fingerprint
    projection,physical,scene = _stationary_peer_input()
    for rid,x in [('Conveyor',2.),('Buffer For Machined parts',3.)]:
        physical['snapshot']['resources'][rid] = {}
        physical['resource_models'][rid] = {}
        physical['resources'][rid] = {'model_static':True}
        physical['geometry']['resources'][rid] = {'stationary_only':True,
            'component_bounds':[{'id':rid,'bounds':[[x-.1,x+.1],[-.1,.1],[-.1,.1]]}]}
        scene[rid] = {'transfer_requirements':['robot clear of transfer corridor']}
        for point in projection['observations']:
            point['resources'][rid] = {}
    part = {'current_pose':[2.,0.,.12,0.,0.,0.,1.],'contained_by':'Conveyor'}
    physical['snapshot']['parts']['part'] = deepcopy(part)
    physical['geometry']['parts']['part'] = {'local_bounds':[[-.02,.02]]*3}
    for point in projection['observations']:
        point['parts']['part'] = deepcopy(part)
    contract = {'kind':'resource_owned_shared_transport','event':{'participants':['Conveyor','Buffer For Machined parts']},
        'part_names':['part'],'displacement_m':1.,
        'physical_start_fingerprint':recovery_validation_fingerprint(physical)}
    projection['owner_interval_contract'] = contract
    projection['resource_models'] = deepcopy(physical['resource_models'])
    return projection,physical,scene


def test_transfer_corridor_clearance_covers_full_robot_and_keeps_overlap_unresolved():
    from cais_spade_llm.agents.central_controller.recovery_case_evidence import _transport_clearance, _transport_clearance_value
    from cais_spade_llm.agents.shared_information.recovery_validation_protocol import recovery_validation_fingerprint
    projection,physical,scene = _transfer_corridor_input()
    before = deepcopy((projection,physical,scene))
    proof = _transport_clearance(projection,physical,scene)
    point = deepcopy(projection['observations'][1])
    point['owner_safety_evidence'] = {'owner_interval_contract':projection['owner_interval_contract'],
        'resource_kinematic_model_fingerprints':{'KMR':_digest(physical['resource_models']['KMR']['configuration'])}}
    assert _transport_clearance_value(proof,point) == [True]
    assert proof['physical_execution_verified'] is False
    assert (projection,physical,scene) == before
    model = physical['resource_models']['KMR']['configuration']
    model['components'].append({'id':'arm_over_corridor','link':'body','bounds':[[2.,2.2],[-.1,.1],[-.1,.1]]})
    projection['owner_interval_contract']['physical_start_fingerprint'] = recovery_validation_fingerprint(physical)
    proof = _transport_clearance(projection,physical,scene)
    point['owner_safety_evidence']['resource_kinematic_model_fingerprints']['KMR'] = _digest(model)
    point['owner_safety_evidence']['owner_interval_contract'] = projection['owner_interval_contract']
    assert point['resources']['KMR']['base_pose'][0] == 0  # The full arm overlaps despite the root/TCP being clear.
    assert _transport_clearance_value(proof,point) == [False,True]


@pytest.mark.parametrize('change',['missing_model','missing_geometry','missing_hold','missing_base','missing_declaration','changed_peer'])
def test_transfer_corridor_needs_complete_population_and_supported_peer_geometry(change):
    from cais_spade_llm.agents.central_controller.recovery_case_evidence import _transport_clearance
    from cais_spade_llm.agents.shared_information.recovery_validation_protocol import recovery_validation_fingerprint
    projection,physical,scene = _transfer_corridor_input()
    if change=='missing_model': physical['resource_models'].pop('KMR')
    elif change=='missing_geometry': physical['resource_models']['KMR']['configuration']['components']=[]
    elif change=='missing_hold': physical['resources']['KMR'].pop('controller_hold_model')
    elif change=='missing_base': physical['resources']['KMR'].pop('base_hold_evidence')
    elif change=='missing_declaration':
        scene['Conveyor']['transfer_requirements']=[]
        scene['Buffer For Machined parts']['transfer_requirements']=[]
    else: projection['observations'][1]['resources']['KMR']['joint_positions']=[.1]
    projection['owner_interval_contract']['physical_start_fingerprint'] = recovery_validation_fingerprint(physical)
    with pytest.raises(ValueError):
        _transport_clearance(projection,physical,scene)


@pytest.mark.parametrize('change',['omitted_robot','omitted_pair','changed_distance','wrong_source'])
def test_transfer_corridor_consumer_rejects_incomplete_or_changed_calculation(change):
    from cais_spade_llm.agents.central_controller.recovery_case_evidence import _transport_clearance, _transport_clearance_value
    projection,physical,scene = _transfer_corridor_input()
    proof = _transport_clearance(projection,physical,scene)
    point = deepcopy(projection['observations'][1])
    point['owner_safety_evidence'] = {'owner_interval_contract':projection['owner_interval_contract'],
        'resource_kinematic_model_fingerprints':{'KMR':_digest(physical['resource_models']['KMR']['configuration'])}}
    if change=='omitted_robot': proof['robot_resources']=[]
    elif change=='omitted_pair': proof['pairs'].pop()
    elif change=='changed_distance': proof['pairs'][0]['distance_bounds']=[100.,200.]
    else: proof['source_fingerprint']='different'
    with pytest.raises(ValueError):
        _transport_clearance_value(proof,point)


def test_transfer_corridor_provider_attaches_raw_proof_to_each_handoff_participant():
    from cais_spade_llm.agents.central_controller.recovery_case_evidence import bind_case_projection_evidence
    projection,physical,scene = _transfer_corridor_input()
    for point in projection['observations']:
        point['owner_safety_evidence']={'resource_evidence':{
            rid:{'handoffs':{'complete':True,'records':[]}} for rid in ('Conveyor','Buffer For Machined parts')}}
    result = bind_case_projection_evidence(projection,physical_evidence=physical,scene_configuration=scene)
    for point in result['observations']:
        rows=point['owner_safety_evidence']['resource_evidence']
        assert rows['Conveyor']['robot_clear']==rows['Buffer For Machined parts']['robot_clear']
        assert rows['Conveyor']['robot_clear']['pairs_complete'] is True


def _ur_custody_ledger_input():
    projection = _custody_overlay_projection()
    ledger = projection.pop('owner_safety_evidence')
    before = {row['resource_id']:deepcopy(row['values']) for row in ledger['state_evidence']['initial']}
    after = deepcopy(before)
    after['ur5e-1'].update(held_part='part',gripper_state='closed')
    for point in projection['observations']:
        point['resources']['M1']['resource_state']='loaded'
    step = projection['owner_interval_contract']['owner_declaration']['program']['steps'][0]
    projection['owner_interval_contract'].pop('owner_declaration')
    projection['owner_interval_contract'].update(projection_clock_version=1,source_contract_fingerprint='owner contract',
        owner_declarations=[{'composition_contract':{'steps':[step]}}])
    task={'resource_jid':'robot@localhost','event_name':'arbitrary_identifier','part_name':'part'}
    projection['provenance']={'binding':{'resource_jid':task['resource_jid']},'contract_fingerprint':'validated'}
    projection['successor_evidence']={'environment_valuation':after}
    projection['projected_snapshot']=deepcopy(projection['observations'][-1])
    projection['resource_updates']={'held_part':'part','gripper_state':'closed'}
    physical={'environment_models':ledger['resource_models'],'environment_valuation':before,
        'jids':{'M1':'machine@localhost','ur5e-1':'robot@localhost'},
        'resources':{'M1':{'idle':True},'ur5e-1':{'idle':True}}}
    return projection,task,physical


def test_validated_ur_custody_ledger_uses_declared_steps_at_intermediate_checkpoint():
    from cais_spade_llm.agents.central_controller.outline_evidence_safety import bind_validated_outline_projection
    projection,task,physical=_ur_custody_ledger_input()
    before=deepcopy((projection,physical))
    result=bind_validated_outline_projection(projection,task=task,physical_evidence=physical)
    assert result['observations'][1]['state_snapshot']['ur5e-1']=={'held_part':'part','gripper_state':'closed'}
    assert result['owner_safety_evidence']['task_evidence']['events'][0]['function']=='arbitrary_identifier'
    assert (projection,physical)==before


@pytest.mark.parametrize('change',['declaration','source','clock','task_owner'])
def test_validated_ur_custody_ledger_rejects_unbound_declaration_or_clock(change):
    from cais_spade_llm.agents.central_controller.outline_evidence_safety import bind_validated_outline_projection
    projection,task,physical=_ur_custody_ledger_input()
    contract=projection['owner_interval_contract']
    if change=='declaration': contract['owner_declarations'][0]['composition_contract']['steps']=[]
    elif change=='source': contract['preview_fingerprint']='different'
    elif change=='clock': contract.pop('projection_clock_version')
    else: contract['declared_task']['resource_id']='M1'
    with pytest.raises(ValueError):
        bind_validated_outline_projection(projection,task=task,physical_evidence=physical)


def _ur_cross_owner_ledger_input():
    from cais_spade_llm.resources.recovery_geometry import _owner_product_custody
    projection,task,physical=_ur_custody_ledger_input()
    rid='Buffer For Machined parts'
    physical['environment_models'][rid]={'state_variables':{'zone_1_part':{'type':['null','string'],'reference':'part_name'}}}
    physical['environment_valuation'][rid]={'zone_1_part':'part'}
    physical['resources'][rid]={'idle':True}
    physical['jids'][rid]='buffer@localhost'
    physical['snapshot']={'parts':{'part':{'contained_by':rid}}}
    projection['geometry']['resources'][rid]={}
    for i,point in enumerate(projection['observations']):
        point['resources'][rid]={'zone_1_part':None if i==2 else 'part'}
        point['parts']['part']['contained_by']=rid if i==0 else None
    projection['projected_snapshot']=deepcopy(projection['observations'][-1])
    _,after,witness=_owner_product_custody(task=task,state=projection['projected_snapshot'],evidence=physical,
        resource_updates=projection['resource_updates'],support={},rid='ur5e-1')
    projection['successor_evidence']['environment_valuation']=after
    projection['product_custody_evidence']=witness
    return projection,task,physical


def test_owner_custody_recomputed_before_cross_owner_task_effect_ledger():
    from cais_spade_llm.agents.central_controller.outline_evidence_safety import bind_validated_outline_projection
    projection,task,physical=_ur_cross_owner_ledger_input()
    result=bind_validated_outline_projection(projection,task=task,physical_evidence=physical)
    ledger=result['owner_safety_evidence']
    events=ledger['task_evidence']['events']
    assert {row['resource_id'] for row in events}=={'ur5e-1','Buffer For Machined parts'}
    buffer=next(row for row in ledger['state_evidence']['updates'] if row['resource_id']=='Buffer For Machined parts')
    assert buffer['values']=={'zone_1_part':None} and buffer['time']==1.
    assert buffer['task_id'] in {row['task_id'] for row in events}


@pytest.mark.parametrize('change',['witness','peer_value','source_occupancy'])
def test_owner_custody_ledger_rejects_unsupported_cross_owner_effect(change):
    from cais_spade_llm.agents.central_controller.outline_evidence_safety import bind_validated_outline_projection
    projection,task,physical=_ur_cross_owner_ledger_input()
    if change=='witness': projection['product_custody_evidence']['resource_model_fingerprint']='different'
    elif change=='peer_value':
        projection['successor_evidence']['environment_valuation']['Buffer For Machined parts']['zone_1_part']='part'
        projection['observations'][-1]['resources']['Buffer For Machined parts']['zone_1_part']='part'
    else: physical['snapshot']['parts']['part']['contained_by']='missing'
    with pytest.raises(ValueError):
        bind_validated_outline_projection(projection,task=task,physical_evidence=physical)


def test_owner_ledger_duplicate_final_time_retains_the_same_verified_completion():
    from cais_spade_llm.agents.central_controller.outline_evidence_safety import bind_validated_outline_projection
    projection,task,physical=_ur_custody_ledger_input()
    physical['environment_models']['ur5e-1']['state_variables']['resource_state']={'type':'string','domain':['idle','picked']}
    physical['environment_valuation']['ur5e-1']['resource_state']='idle'
    projection['successor_evidence']['environment_valuation']['ur5e-1']['resource_state']='picked'
    for index,point in enumerate(projection['observations']):
        point['resources']['ur5e-1']['resource_state']='picked' if index==2 else 'idle'
    projection['observations'].insert(-1,deepcopy(projection['observations'][-1]))
    result=bind_validated_outline_projection(projection,task=task,physical_evidence=physical)
    assert [point['state_snapshot']['ur5e-1']['resource_state'] for point in result['observations']]==[
        'idle','idle','picked','picked']
    projection['observations'][-2]['resources']['ur5e-1']['held_part']=None
    with pytest.raises(ValueError):
        bind_validated_outline_projection(projection,task=task,physical_evidence=physical)


def _shared_participant_handoff_input(part):
    source={'event_id':80,'event_name':'advance_conveyor','participants':['Conveyor','Buffer For Machined parts'],
        'parameter_bindings':{'delivered_part':{'equals':part}},'product_effects':{},'guards':{},'updates':{},
        'collection_effects':{'validation_source':'cais_spade_llm.resources.nominal_conveyor'}}
    local=deepcopy(source)
    local.pop('collection_effects')
    local['guards']={'zone_1_part':{'equals':None}}
    local['updates']={'zone_1_part':{'set_from_param':'delivered_part'}} if part is not None else {}
    task={'task_id':'buffer_task','resource_id':'Buffer For Machined parts','function':'advance_conveyor',
        'start_time':0,'end_time':1,'declared_task':{'resource_id':'Conveyor','event_id':80,
        'event_name':'advance_conveyor','parameters':{'delivered_part':part}}}
    envelope={'owner_interval_contract':{'kind':'resource_owned_shared_transport','event':source},
        'resource_event_fingerprints':{'Conveyor':_digest([source]),'Buffer For Machined parts':_digest([local])},
        'resource_evidence':{'Conveyor':{'handoff_declarations':[source]},
                             'Buffer For Machined parts':{'handoff_declarations':[local]}},
        'state_evidence':{'initial':[{'resource_id':'Conveyor','values':{}},
            {'resource_id':'Buffer For Machined parts','values':{'zone_1_part':None}}],
            'updates':([{'resource_id':'Buffer For Machined parts','task_id':'buffer_task',
                         'time':1,'values':{'zone_1_part':part}}] if part is not None else [])}}
    return local,task,{'owner_safety_evidence':envelope}


@pytest.mark.parametrize('part',[None,'part'])
def test_shared_handoff_participant_uses_exact_owner_collection_and_local_effects(part):
    from cais_spade_llm.agents.central_controller.recovery_case_evidence import _shared_handoff_event
    local,task,point=_shared_participant_handoff_input(part)
    before=deepcopy((local,task,point))
    assert _shared_handoff_event(local,task,point)==point['owner_safety_evidence']['owner_interval_contract']['event']
    assert (local,task,point)==before


@pytest.mark.parametrize('change',['source','source_hash','local_binding','participant','guard','effect','missing_effect'])
def test_shared_handoff_participant_rejects_unbound_or_conflicting_owner_facts(change):
    from cais_spade_llm.agents.central_controller.recovery_case_evidence import _shared_handoff_event
    local,task,point=_shared_participant_handoff_input('part')
    envelope=point['owner_safety_evidence']
    if change=='source': envelope['owner_interval_contract']['event']=dict(envelope['owner_interval_contract']['event'],event_id=81)
    elif change=='source_hash': envelope['resource_event_fingerprints']['Conveyor']='changed'
    elif change=='local_binding': local['parameter_bindings']={'delivered_part':{'equals':None}}
    elif change=='participant': envelope['resource_evidence'].pop('Buffer For Machined parts')
    elif change=='guard': envelope['state_evidence']['initial'][1]['values']['zone_1_part']='occupied'
    elif change=='effect': envelope['state_evidence']['updates'][0]['values']['zone_1_part']='other'
    else: envelope['state_evidence']['updates']=[]
    with pytest.raises(ValueError):
        _shared_handoff_event(local,task,point)
