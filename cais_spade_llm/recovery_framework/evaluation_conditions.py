from __future__ import annotations

"""Create and verify recorded evaluation conditions without recovery guidance.

These capture-only operations change an owned paused Gazebo experiment. They
are never called by offline generation or exported through public retrieval.
"""

import asyncio
import hashlib
import math
import time
import xml.etree.ElementTree as ET
from copy import deepcopy
from pathlib import Path

from cais_spade_llm.recovery_framework import ROOT
from cais_spade_llm.recovery_framework.diagnostics import read_object, write_record


def condition_setup(setup: dict, scenario: str, condition: str, directory: Path) -> dict:
    """Keep variant machine program declarations in a separate configuration."""
    result = deepcopy(setup)
    if condition not in {'normal', 'changed'}:
        raise ValueError('Select normal or changed')
    if scenario != 'Machining breakdown during part processing' or condition == 'normal':
        return result
    scene = read_object(ROOT / setup['scene_file'])
    machine = next(row for row in scene['machines'] if row['resource_id'] == 'M2')
    # Disable the declared program in this independent simulated installation;
    # its current circle program and physical tool/fixture remain unchanged.
    options = machine.get('program_options', {})
    if 'square' not in options:
        raise ValueError('The normal M2 declaration does not provide program_options.square')
    del options['square']
    destination = Path(directory) / 'recovery_framework_gazebo.json'
    write_record(destination, scene)
    result['scene_file'] = str(destination.resolve())
    return result


def _controller(runtime):
    if runtime.setup.get('execution_mode') != 'simulation' or not runtime.stopped or not runtime.nominal_idle.is_set():
        raise ValueError('Evaluation changes require this owned paused simulation checkpoint')
    for owner in runtime.resource_agents:
        controller = getattr(owner, '_controller', None)
        if controller is not None and getattr(controller, '_node', None) is not None:
            return controller
    raise ValueError('NEEDS_CONTEXT: no registered simulation observation controller')


def _call(controller, service_type, endpoint: str, request):
    client = controller._node.create_client(service_type, endpoint, callback_group=controller._cb_group)
    try:
        if not client.wait_for_service(timeout_sec=5):
            raise RuntimeError('Evaluation service unavailable: ' + endpoint)
        response = controller._wait_future(client.call_async(request), timeout_sec=15, label='evaluation_condition')
        if response is None or not response.success:
            raise RuntimeError('Evaluation service did not acknowledge: ' + endpoint)
        return response
    finally:
        controller._node.destroy_client(client)


def _block_side_access(runtime, controller) -> dict:
    from gazebo_msgs.srv import SpawnEntity
    from moveit_msgs.msg import CollisionObject, PlanningScene
    from moveit_msgs.srv import ApplyPlanningScene
    from shape_msgs.msg import SolidPrimitive

    machine = next(row for row in runtime.context.inputs['scene']['machines'] if row['resource_id'] == 'M1')
    query = controller._GetEntityState.Request(
        name=machine['gazebo_model'] + '::' + machine['side_access_link'], reference_frame='world')
    observed = controller._wait_future(controller._get_state_client.call_async(query),
        timeout_sec=5, label='evaluation_side_access')
    if observed is None or not observed.success:
        raise ValueError('NEEDS_CONTEXT: the actual M1 side_access pose was not observed')
    pose = observed.state.pose
    dimensions = [2 * float(machine['KMR_process_clearance_m'])] * 3
    name = 'evaluation_side_access_obstacle'
    xml = ('<sdf version="1.6"><model name="' + name + '"><static>true</static><link name="link">'
           '<collision name="collision"><geometry><box><size>' + ' '.join(map(str, dimensions)) +
           '</size></box></geometry></collision><visual name="visual"><geometry><box><size>' +
           ' '.join(map(str, dimensions)) + '</size></box></geometry></visual></link></model></sdf>')
    request = SpawnEntity.Request(name=name, xml=xml, initial_pose=pose, reference_frame='world')
    _call(controller, SpawnEntity, '/spawn_entity', request)
    query = controller._GetEntityState.Request(name=name, reference_frame='world')
    measured = controller._wait_future(controller._get_state_client.call_async(query),
        timeout_sec=5, label='evaluation_obstacle_observation')
    if measured is None or not measured.success:
        raise ValueError('NEEDS_CONTEXT: spawned obstruction has no actual pose observation')
    obj = CollisionObject(id=name + '/link/collision')
    obj.header.frame_id = 'world'
    obj.operation = CollisionObject.ADD
    obj.primitives = [SolidPrimitive(type=SolidPrimitive.BOX, dimensions=dimensions)]
    obj.primitive_poses = [measured.state.pose]
    scene = PlanningScene(is_diff=True)
    scene.robot_state.is_diff = True
    scene.world.collision_objects = [obj]
    _call(controller, ApplyPlanningScene, '/apply_planning_scene', ApplyPlanningScene.Request(scene=scene))
    def values(value):
        return [value.position.x, value.position.y, value.position.z,
                value.orientation.x, value.orientation.y, value.orientation.z, value.orientation.w]
    record = {'resource_id': 'M1', 'link': 'side_access', 'frame': 'world',
              'access_pose': values(pose), 'model_name': name, 'observed_pose': values(measured.state.pose),
              'size': dimensions, 'collision_object_id': obj.id, 'sdf': xml,
              'source': '/gazebo/get_entity_state', 'collision_scene_acknowledged': True,
              'observed_at_unix': time.time()}
    runtime.recovery_additional_observations = {'M1.side_access': deepcopy(record)}
    with runtime.context.admission_lock:
        runtime.context.revision += 1
    return record


async def apply_changed_condition(runtime, scenario: str) -> dict:
    """Apply only the requested changed condition and retain measured results."""
    controller = _controller(runtime)
    if scenario == 'Conveyor breakdown':
        with runtime.context.admission_lock:
            runtime.context.unavailable_resources.add('KMR')
            runtime.context.revision += 1
        return {'resource_id': 'KMR', 'unavailable_resources': sorted(runtime.context.unavailable_resources),
                'source': 'owned runtime resource availability', 'context_revision': runtime.context.revision}
    if scenario == 'ur5e-1 breakdown':
        return await asyncio.to_thread(_block_side_access, runtime, controller)
    if scenario == 'Machining breakdown during part processing':
        machine = next(row for row in runtime.context.inputs['scene']['machines'] if row['resource_id'] == 'M2')
        if 'square' in machine.get('program_options', {}):
            raise ValueError('Changed M2 configuration is not installed in this runtime')
        return {'resource_id': 'M2', 'current_configuration': deepcopy(machine['current_configuration']),
                'program_options': deepcopy(machine.get('program_options', {})),
                'source': runtime.setup['scene_file']}
    if scenario != 'Part slippage':
        raise ValueError('Select an existing failure scenario')
    from cais_spade_llm.recovery_framework.failure_effects import _observe_part

    recovery = runtime.part_slippage_recovery
    part = recovery._failure['part_name']
    model = runtime.context.geometry[part]['model_name']
    if any(actor.valuation.get('held_part') == part for actor in runtime.context.resources.values()):
        raise ValueError('Changed drop requires an actually detached part')
    requested = {'x': 100., 'y': 100., 'z': .05, 'qx': 0., 'qy': 0., 'qz': 0., 'qw': 1.}
    moved = await asyncio.to_thread(controller.set_entity_pose, model, **requested)
    if not moved.get('success'):
        raise RuntimeError('Gazebo rejected the changed drop condition')
    previous, stable = None, 0
    for _ in range(100):
        observed = await asyncio.to_thread(_observe_part, controller, model)
        if previous is not None and math.dist([previous[key] for key in ('x', 'y', 'z')],
                                             [observed[key] for key in ('x', 'y', 'z')]) <= .002:
            stable += 1
            if stable >= 8:
                break
        else:
            stable = 0
        previous = observed
        await asyncio.sleep(.1)
    else:
        raise ValueError('NEEDS_CONTEXT: changed dropped part did not become stationary')
    if not await asyncio.to_thread(controller._sync_part_collision, model):
        raise ValueError('Changed drop collision scene was not acknowledged')
    with runtime.context.admission_lock:
        runtime.context.part_tracker[part]['observed_pose'] = deepcopy(observed)
        runtime.context.revision += 1
    before = deepcopy(recovery._failure.get('observed_drop_pose'))
    recovery._failure['observed_drop_pose'] = deepcopy(observed)
    return {'part_name': part, 'model_name': model, 'previous_observed_drop_pose': before,
            'requested_pose': requested, 'observed_drop_pose': observed, 'set_entity_pose': moved,
            'collision_scene': deepcopy(getattr(controller, '_last_command_evidence', {})),
            'source': '/gazebo/get_entity_state', 'context_revision': runtime.context.revision}


def _kinematic_radius(robot_description: str) -> float:
    """Bound every recorded link origin for all configurations in finite URDF limits."""
    root = ET.fromstring(robot_description)
    parents = {}
    for joint in root.findall('joint'):
        origin = joint.find('origin')
        translation = [float(value) for value in (origin.get('xyz', '0 0 0') if origin is not None else '0 0 0').split()]
        bound = math.sqrt(sum(value * value for value in translation))
        kind = joint.get('type')
        if kind == 'prismatic':
            limit = joint.find('limit')
            if limit is None:
                raise ValueError('NEEDS_CONTEXT: prismatic reach is not bounded')
            bound += max(abs(float(limit.get('lower', 'nan'))), abs(float(limit.get('upper', 'nan'))))
        elif kind not in {'fixed', 'revolute', 'continuous'}:
            raise ValueError('NEEDS_CONTEXT: unsupported mobile joint reach: ' + str(kind))
        if not math.isfinite(bound):
            raise ValueError('NEEDS_CONTEXT: non-finite reach bound')
        parents[joint.find('child').get('link')] = (joint.find('parent').get('link'), bound)
    def distance(link, visited):
        if link in visited:
            raise ValueError('Invalid kinematic parent cycle')
        if link not in parents:
            return 0.
        parent, bound = parents[link]
        return bound + distance(parent, visited | {link})
    if not parents:
        raise ValueError('NEEDS_CONTEXT: recorded kinematic model is missing')
    return max(distance(link, set()) for link in parents)


def _recorded_base_reach(captured: dict) -> dict:
    """Bind the planning base limits to the owner's recorded map checks.

    The KMR planning joints represent its measured base pose; their limits
    alone do not constrain the Gazebo base. The captured controller admits
    base goals and stopping sweeps only inside this recorded map. This check
    establishes the workspace used by the evaluation condition, not a safe
    trajectory or an execution permission.
    """
    from cais_spade_llm.agents.central_controller.recovery_case_evidence import recorded_base_hold

    physical = captured['physical_evidence']
    mobile = {name for name, row in physical['snapshot']['resources'].items()
              if 'base_pose' in row}
    if mobile != {'KMR'}:
        raise ValueError('NEEDS_CONTEXT: complete mobile resource reach declarations are unavailable')
    evidence = physical['resources'].get('KMR', {}).get('base_hold_evidence', {})
    description = captured['planning_scene_capture']['robot_description']
    configuration = physical.get('resource_models', {}).get('KMR', {}).get('configuration')
    if (not isinstance(configuration, dict)
            or configuration.get('description_sha256') != hashlib.sha256(description.encode()).hexdigest()
            or configuration.get('root') != 'KMR_base_link'
            or physical['snapshot']['resources']['KMR'].get('base_pose') != evidence.get('root_pose')):
        raise ValueError('NEEDS_CONTEXT: KMR planning description or captured base pose differs')
    try:
        recorded_base_hold(resource_id='KMR', physical_evidence=physical, model=configuration)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError('NEEDS_CONTEXT: KMR physical base and command authority are unavailable: ' + str(exc)) from exc
    controller = evidence.get('controller_observation', {})
    source_path = ROOT / 'ros2/cais_lab_robotics/scripts/kmr_base_controller.py'
    source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
    if (controller.get('source') != '/KMR/recovery_state'
            or controller.get('controller_node') != '/KMR_base_controller'
            or controller.get('controller_source_sha256') != source_sha256
            or evidence.get('controller_source_sha256') != source_sha256
            or not evidence.get('launch_id')
            or controller.get('instance_id') != evidence['launch_id']):
        raise ValueError('NEEDS_CONTEXT: recorded KMR map lacks its controller and capture binding')
    record = controller.get('occupancy_map')
    if not isinstance(record, dict) or record.get('frame_id') != 'world':
        raise ValueError('NEEDS_CONTEXT: recorded KMR world map is unavailable')
    width, height, resolution = (record.get(key) for key in ('width', 'height', 'resolution'))
    origin = record.get('origin')
    cells_sha256 = record.get('cells_sha256')
    if (type(width) is not int or type(height) is not int or width <= 0 or height <= 0
            or type(resolution) not in (int, float) or not math.isfinite(resolution) or resolution <= 0
            or type(record.get('cell_count')) is not int or record['cell_count'] != width * height
            or not isinstance(cells_sha256, str) or len(cells_sha256) != 64
            or any(value not in '0123456789abcdef' for value in cells_sha256)
            or not isinstance(origin, list) or len(origin) != 7
            or any(type(value) not in (int, float) or not math.isfinite(value) for value in origin)
            or origin[2:6] != [0., 0., 0., 0.] or abs(origin[6]) != 1.):
        raise ValueError('NEEDS_CONTEXT: recorded KMR map extent is incomplete or unsupported')
    for name in ('stamp', 'map_load_time'):
        stamp = record.get(name)
        if (not isinstance(stamp, dict) or type(stamp.get('sec')) is not int
                or type(stamp.get('nanosec')) is not int or not 0 <= stamp['nanosec'] < 1_000_000_000):
            raise ValueError('NEEDS_CONTEXT: recorded KMR map timestamps are missing')
    bounds = {'x': [origin[0], origin[0] + width * resolution],
              'y': [origin[1], origin[1] + height * resolution]}
    robot = ET.fromstring(description)
    declarations = {}
    for axis, parent, child, direction in (
        ('x', 'world', 'KMR_base_x_link', '1 0 0'),
        ('y', 'KMR_base_x_link', 'KMR_base_y_link', '0 1 0'),
    ):
        name = 'KMR_base_' + axis + '_joint'
        matches = robot.findall("joint[@name='" + name + "']")
        if len(matches) != 1:
            raise ValueError('NEEDS_CONTEXT: exact KMR planning base declarations are unavailable')
        joint = matches[0]
        if (joint.get('type') != 'prismatic' or joint.find('parent') is None
                or joint.find('child') is None or joint.find('axis') is None
                or joint.find('limit') is None or joint.find('origin') is not None
                or joint.find('parent').get('link') != parent
                or joint.find('child').get('link') != child
                or joint.find('axis').get('xyz') != direction):
            raise ValueError('NEEDS_CONTEXT: KMR map and planning coordinates are not connected')
        limits = [float(joint.find('limit').get(key, 'nan')) for key in ('lower', 'upper')]
        if (not all(math.isfinite(value) for value in limits + bounds[axis])
                or not limits[0] <= bounds[axis][0] < bounds[axis][1] <= limits[1]):
            raise ValueError('NEEDS_CONTEXT: KMR map extends beyond the recorded planning reach bound')
        declarations[name] = limits
    yaw = robot.findall("joint[@name='KMR_base_yaw_joint']")
    if (len(yaw) != 1 or yaw[0].get('type') != 'continuous'
            or yaw[0].find('parent') is None or yaw[0].find('child') is None
            or yaw[0].find('axis') is None or yaw[0].find('origin') is not None
            or yaw[0].find('parent').get('link') != 'KMR_base_y_link'
            or yaw[0].find('child').get('link') != configuration['root']
            or yaw[0].find('axis').get('xyz') != '0 0 1'):
        raise ValueError('NEEDS_CONTEXT: KMR planning base chain does not connect to its physical root')
    return {'map_bounds': bounds, 'map_cells_sha256': cells_sha256,
            'planning_joint_limits': declarations, 'controller_source_sha256': source_sha256,
            'source_reference': '/physical_evidence/resources/KMR/base_hold_evidence/controller_observation',
            'scope': 'Recorded controller base goals and stopping sweeps within the captured world map',
            'assumptions': ['The captured map and controller/model configuration remain unchanged; changes invalidate this bound'],
            'physical_execution_verified': False}


def verify_capture_condition(captured: dict, scenario: str, condition: str) -> dict:
    """Recheck the selected condition from saved observations without live calls."""
    if captured.get('scenario') != scenario or captured.get('evidence_source') != 'Gazebo':
        raise ValueError('Condition requires the selected actual Gazebo capture')
    if condition not in {'normal', 'changed'}:
        raise ValueError('Select normal or changed')
    failure = captured.get('failure') or {}
    if failure.get('injection_status') != 'completed':
        raise ValueError('Failure injection did not complete')
    if captured.get('setup', {}).get('execution_mode') != 'simulation':
        raise ValueError('Evaluation is restricted to simulation')
    result = {'scenario': scenario, 'condition': condition, 'context_revision': captured.get('context_revision'),
              'capture_time': captured.get('captured_at'), 'source_references': ['/failure', '/context_revision'],
              'execution_authorized': False}
    changed = condition == 'changed'
    if scenario == 'Conveyor breakdown':
        unavailable = captured.get('unavailable_resources')
        if not isinstance(unavailable, list) or ('KMR' in unavailable) != changed:
            raise ValueError('Captured KMR availability does not establish the selected condition')
        result.update(unavailable_resources=unavailable, source_references=['/unavailable_resources'])
    elif scenario == 'Machining breakdown during part processing':
        machine = next(row for row in captured['inputs']['scene']['machines'] if row['resource_id'] == 'M2')
        configurations = [machine['current_configuration'], *machine.get('program_options', {}).values()]
        required = {row['result'] for requirement in captured['requirements'][failure['part_name']]
                    for row in requirement.get('processesToComplete', []) if row.get('process') == 'trim'}
        if not required:
            raise ValueError('NEEDS_CONTEXT: required machining results are not recorded')
        effects = {row.get('result') for config in configurations
                   for row in config.get('program', {}).get('effects', []) if row.get('process') == 'trim'
                   and config.get('program', {}).get('validated') is True
                   and all(config.get(key) == value for key, value in config['program'].get('required_configuration', {}).items())}
        compatible = bool(required) and required <= effects
        if compatible == changed:
            raise ValueError('Captured M2 programs do not establish the selected configuration condition')
        result.update(required_results=sorted(required), declared_results=sorted(effects),
                      source_references=['/inputs/scene/machines', '/requirements'])
    elif scenario == 'ur5e-1 breakdown':
        observation = captured.get('additional_observations', {}).get('M1.side_access')
        if changed:
            if not isinstance(observation, dict) or observation.get('collision_scene_acknowledged') is not True:
                raise ValueError('NEEDS_CONTEXT: side_access obstruction was not observed and acknowledged')
            point, pose, size = observation['access_pose'], observation['observed_pose'], observation['size']
            if math.dist(point[:3], pose[:3]) > min(size) / 2:
                raise ValueError('Observed obstruction does not cover side_access')
            objects = captured['planning_scene_capture']['planning_scene']['world']['collision_objects']
            match = [row for row in objects if row.get('id') == observation['collision_object_id']]
            if (len(match) != 1 or len(match[0].get('primitives', [])) != 1
                    or match[0]['primitives'][0]['type'] != 1
                    or match[0]['primitives'][0]['dimensions'] != size
                    or match[0].get('header', {}).get('frame_id') != 'world'
                    or len(match[0].get('primitive_poses', [])) != 1):
                raise ValueError('Planning scene does not retain the exact observed obstruction')
            recorded_pose = match[0]['primitive_poses'][0]
            values = [recorded_pose['position'][axis] for axis in ('x', 'y', 'z')]
            values += [recorded_pose['orientation'][axis] for axis in ('x', 'y', 'z', 'w')]
            if values != pose:
                raise ValueError('Planning obstacle pose differs from the actual observation')
            result.update(obstruction=deepcopy(observation), source_references=['/additional_observations/M1.side_access', '/planning_scene_capture'])
        elif observation:
            raise ValueError('Normal capture contains the changed obstruction')
    elif scenario == 'Part slippage':
        pose = failure.get('observed_drop_pose')
        if not isinstance(pose, dict) or any(type(pose.get(key)) not in {int, float} or not math.isfinite(pose[key]) for key in ('x', 'y', 'z')):
            raise ValueError('NEEDS_CONTEXT: actual dropped part pose is missing')
        base_reach = _recorded_base_reach(captured)
        origin_radius = _kinematic_radius(captured['planning_scene_capture']['robot_description'])
        bounds = [entry['bounds'] for resource in captured['physical_evidence']['geometry']['resources'].values()
                  for entry in resource.get('component_bounds', [])]
        if not bounds:
            raise ValueError('NEEDS_CONTEXT: recorded tool and resource geometry bounds are missing')
        observed_geometry_radius = max(math.sqrt(sum(max(abs(value) for value in axis) ** 2 for axis in box)) for box in bounds)
        # For any rigidly linked point, its link-relative radius is bounded by
        # the current point and link radii; add the possible future link radius.
        radius = 2 * origin_radius + observed_geometry_radius
        distance = math.sqrt(sum(pose[key] ** 2 for key in ('x', 'y', 'z')))
        if (distance > radius) != changed:
            raise ValueError('Observed drop does not establish the selected finite kinematic reach condition')
        result.update(observed_drop_pose=pose, link_origin_radius_m=origin_radius,
                      supported_geometry_radius_m=radius, observed_distance_m=distance,
                      base_reach=base_reach,
                      reach_scope='recorded controller map bounds, finite joint limits, fixed transforms, and resource/tool geometry',
                      source_references=['/failure/observed_drop_pose', '/planning_scene_capture/robot_description',
                          base_reach['source_reference']])
    else:
        raise ValueError('Select an existing failure scenario')
    return result
