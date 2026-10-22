from __future__ import annotations

"""Recorded case safety definitions and conservative owner-evidence checks."""

import json
from copy import deepcopy
from pathlib import Path

import pytest

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


def test_previous_conveyor_safety_file_remains_available():
    path = ROOT / 'cais_spade_llm/specification/safety/safety_Conveyor_breakdown.txt'
    rules, dfas = compile_predefined_safety(json.loads(path.read_text()))
    assert len(rules) == len(dfas) == 77
    assert 'SAFE_KMR_docking' in dfas


_CONVEYOR_RULES = ('safety_assembly_board-v1_mutex.txt',
                   'safety_assembly_board-v1_KET4_Square_4mm_before_gear_small.txt')


def _conveyor_document(index):
    return _spec(_CONVEYOR_RULES[index])


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


_MACHINE_RULES = ('safety_M2_mutex.txt', 'safety_KET4_Square_4mm_before_RGOCG4-50_Round_4mm_processing.txt')


def _machine_document(index):
    return _spec(_MACHINE_RULES[index])


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
