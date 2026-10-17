from __future__ import annotations

"""KMR owner calculations over explicitly supplied, recorded observations."""

import math
from contextvars import ContextVar
from copy import deepcopy

from cais_spade_llm.recovery_framework.geometry import compose, quaternion, rotate

_cartesian_joint_calculation_cache: ContextVar[dict | None] = ContextVar(
    "_cartesian_joint_calculation_cache", default=None)


def _vector(value, length: int) -> bool:
    return (isinstance(value, list) and len(value) == length
            and all(type(item) in (int, float) and math.isfinite(item) for item in value))


def _pose(value) -> bool:
    return _vector(value, 7) and math.isclose(sum(item * item for item in value[3:]), 1., abs_tol=1e-5)


def recorded_scene_payloads(observations: dict, scene_capture: dict) -> dict:
    """Bind KMR's released object IDs using completed, recorded owner commands.

    The SDF producer uses one ID per collision. KMR's ``part_collision`` uses
    the exact part name for its released box. An unacknowledged SDF declaration
    is not evidence that those SDF IDs were installed after a KMR release.
    This calculation preserves both identities and never changes observations.
    """
    from cais_spade_llm.agents.shared_information.recovery_validation_protocol import recovery_validation_fingerprint

    physical = observations.get('physical_evidence', {})
    scene = scene_capture['planning_scene']
    world = scene.get('world', {}).get('collision_objects', [])
    attached = scene.get('robot_state', {}).get('attached_collision_objects', [])
    present = [item['id'] for item in world] + [item['object']['id'] for item in attached]
    result = {}
    for part, measured in physical.get('parts', {}).items():
        identity = measured.get('collision_scene_identity') or {}
        declared = identity.get('collision_objects') or []
        model = identity.get('model_name')
        if (identity.get('status') != 'recorded' or identity.get('collision_scene_acknowledged') is not False
                or model != part or measured.get('name') != model or len(declared) != 1
                or any(item['id'] in present for item in declared) or present.count(part) != 1):
            continue
        source = declared[0]
        if source.get('model') != model or not all(source.get(key) for key in ('link', 'collision')):
            continue
        # A model-name object cannot stand for an unrecorded compound body.
        native = physical.get('physics_evidence', {}).get('models', {}).get(model, {})
        collisions = [row for link in native.get('links', {}).values() for row in link.get('collisions', [])]
        if len(collisions) != 1 or collisions[0].get('id') != '::'.join((model, source['link'], source['collision'])):
            continue
        records = []
        for index, transition in enumerate(observations.get('transitions', [])):
            acknowledgement = transition.get('acknowledgement') or {}
            execution = transition.get('observations') or {}
            observed = execution.get('observations') or {}
            task_id = acknowledgement.get('task_id')
            if (acknowledgement.get('run_id') != observations.get('run_id') or not task_id
                    or acknowledgement.get('resource_id') != 'KMR' or acknowledgement.get('part_name') != part
                    or acknowledgement.get('status') != 'completed' or execution.get('status') != 'completed'
                    or execution.get('resource_id') != 'KMR' or execution.get('task_id') != task_id
                    or observed.get('part_name') != part or observed.get('attached') is not False):
                continue
            for step_index, step in enumerate(observed.get('primitive_results', [])):
                outcome = step.get('result') or {}
                if (step.get('resource_id') == 'KMR' and step.get('task_id') == task_id
                        and step.get('primitive') == 'release_part' and step.get('status') == 'completed'
                        and outcome.get('attached') is False and 'held_part' in outcome
                        and outcome['held_part'] is None):
                    records.append({'ref': f'/transitions/{index}/observations/observations/primitive_results/{step_index}',
                                    'task_id': task_id, 'revision': acknowledgement.get('revision'),
                                    'acknowledgement_fingerprint': recovery_validation_fingerprint(acknowledgement),
                                    'primitive_result_fingerprint': recovery_validation_fingerprint(step)})
        if not records:
            continue
        result[part] = {**deepcopy(identity), 'collision_objects': [{**deepcopy(source), 'id': part}],
            'source': 'KMR.part_collision released object identity and recorded completed release_part',
            'declared_scene_identity': deepcopy(identity), 'recorded_owner_commands': records,
            'scene_fingerprint': scene_capture.get('scene_fingerprint'),
            'calculation': {'kind': 'saved owner identity binding', 'run_id': observations['run_id'],
                            'capture_revision': observations.get('context_revision')}}
    return result


def transport_pose(scene: dict, base_pose: list, current_pose: list) -> list:
    """Calculate the declared transport TCP using the measured base and orientation."""
    if not _pose(base_pose) or not _pose(current_pose):
        raise ValueError('KMR transport requires complete xyz/xyzw observations')
    config = scene['KMR']['task_execution']
    offset = config['cartesian_waypoints']['transport_tcp_position_in_base_frame_m']
    if not _vector(offset, 3):
        raise ValueError('KMR transport TCP offset is unavailable')
    target = compose(base_pose, [*offset, 0., 0., 0., 1.])
    if rotate(current_pose[3:], [0., 0., 1.])[2] > -math.cos(config['orientation_tolerance_rad']):
        raise ValueError('KMR transport requires a downward observed gripper')
    return [*target[:3], *current_pose[3:]]


def transfer_waypoints(scene: dict, base_pose: list, start: list, target: list) -> list:
    """Use the same declared KMR downward transfer geometry in both modes."""
    from cais_spade_llm.recovery_framework.kmr_motion import downward_transfer_waypoints
    from cais_spade_llm.resources.robot.cartesian_waypoints import sample_segment

    if not all(_pose(value) for value in (base_pose, start, target)):
        raise ValueError('KMR transfer requires complete xyz/xyzw poses')
    kmr = scene['KMR']
    config = kmr['task_execution']
    mount = compose(base_pose, [*kmr['arm_mount_xyz'], *quaternion(kmr['arm_mount_rpy'])])
    settings = config['cartesian_waypoints']
    waypoints = downward_transfer_waypoints(start, target, mount, settings)
    middle = settings.get('transfer_orientation_waypoint_xyzw_in_base_frame')
    if (middle and waypoints and abs(sum(a*b for a, b in zip(start[3:], target[3:], strict=True)))
            < math.sin(kmr['base_control']['yaw_tolerance_rad'] + config['orientation_tolerance_rad'])):
        middle = compose(base_pose, [0., 0., 0., *middle])[3:]
        half = len(waypoints)//2
        first = sample_segment(start, [*start[:3], *middle], linear_step=1e6,
                               angular_step=1e6, minimum_samples=half)
        second = sample_segment([*target[:3], *middle], target, linear_step=1e6,
                                angular_step=1e6, minimum_samples=len(waypoints)-half-1)
        for waypoint, orientation in zip(waypoints, [*first, *second[1:]], strict=True):
            waypoint[3:] = orientation[3:]
    return waypoints


def recorded_location(scene: dict, location: str) -> tuple[dict, str]:
    """Resolve one exact recorded location; ambiguity never selects a source."""
    declarations = []
    if isinstance(scene.get(location), dict):
        declarations.append((scene[location], '/inputs/scene/' + location))
    for key, value in scene.items():
        if isinstance(value, list):
            declarations.extend((row, f'/inputs/scene/{key}/{index}') for index, row in enumerate(value)
                                if isinstance(row, dict) and row.get('resource_id') == location)
    if len(declarations) != 1:
        raise KeyError('One exact recorded location is required: ' + str(location))
    return declarations[0]


def pickup_docking_pose(scene: dict, location: str, part: str) -> list:
    """Read the part-specific dock or the exact location's recorded common dock."""
    source, _ = recorded_location(scene, location)
    if 'KMR_pick_docking_poses' in source:
        pose = source['KMR_pick_docking_poses'][part]
        if not _vector(pose, 3):
            raise ValueError('Recorded KMR pickup dock requires three finite coordinates')
        return deepcopy(pose)
    dock = source['KMR_docking_pose']
    if not _vector(dock, 6) or any(abs(value) > 1e-9 for value in dock[3:5]):
        raise ValueError('Recorded KMR docking pose requires a planar base pose')
    return [dock[0], dock[1], dock[5]]


def pick_targets(scene: dict, part: str, observed: list, current_pose: list,
                 base_pose: list, initial: list | None = None, *, location: str = 'Storage') -> dict:
    """Calculate pickup targets from the same facts used by the execution owner."""
    from cais_spade_llm.recovery_framework.kmr_gazebo import _nearest_pick_orientation

    if not all(_pose(value) for value in (observed, current_pose, base_pose)):
        raise ValueError('KMR pickup requires complete recorded part, TCP, and base poses')
    if initial is not None and (not _pose(initial) or math.dist(observed[:3], initial[:3]) > .01):
        raise ValueError('Storage part moved before target computation')
    config = scene['KMR']['task_execution']
    source, _ = recorded_location(scene, location)
    calibrated = ('KMR_pick_orientations_xyzw', 'KMR_pick_approach_clearance_m', 'KMR_pick_arm_configurations')
    if any(field in source for field in calibrated):
        choices = source['KMR_pick_orientations_xyzw'][part]
        clearance_m = source['KMR_pick_approach_clearance_m'][part]
        seed = source['KMR_pick_arm_configurations'][part]
    else:
        # These are owner-calculated targets, not inherited geometric proof.
        # Incomplete location-specific calibration must not use this branch.
        fixture = source['workholding_pose']
        tolerance = config['position_tolerance_m']
        if not _vector(fixture, 6) or not _vector([tolerance], 1) or tolerance <= 0:
            raise KeyError('Recorded workholding pose and position tolerance')
        if math.dist(observed[:3], fixture[:3]) > tolerance:
            raise ValueError('Recorded part pose differs from the requested workholding location')
        pickup_docking_pose(scene, location, part)
        choices = [config['pick_orientation_xyzw']]
        clearance_m = config['approach_clearance_m']
        seed = config['ik_seed_configurations'][0]
    if not _vector([clearance_m], 1) or clearance_m <= 0 or not _vector(seed, 7):
        raise KeyError('Recorded pickup clearance and seven-joint seed')
    orientation = _nearest_pick_orientation(current_pose[3:], choices, config['orientation_tolerance_rad'])
    target = [*observed[:3], *orientation]
    target[2] += config['grasp_height_m']
    approach = target.copy()
    approach[2] += clearance_m
    clearance = target.copy()
    clearance[2] += config['minimum_pick_lift_m']
    lift = transport_pose(scene, base_pose, target)
    if lift[2] < clearance[2]:
        raise ValueError('Saved KMR carrying height does not clear the picked part')
    return {'target': target, 'approach': approach, 'lift': lift, 'lift_waypoints': [clearance],
            'retreat': lift.copy(), 'seed': list(seed),
            'carrying': list(config['carrying_arm_configuration'])}


def place_targets(scene: dict, machine: dict, transform: list, current_pose: list,
                  base_pose: list, fixture: list | None, *, part_name: str | None = None) -> dict:
    """Calculate placement poses without establishing clearance or supported rest.

    Recorded slots and workholding specify the destination part pose. Preserve
    that pose; the separate support calculation must establish resting contact.
    """
    from cais_spade_llm.recovery_framework.kmr_gazebo import _at_dock

    dock = machine['KMR_docking_pose']
    slot = 'slots' in machine
    if slot:
        declared = machine['slots'][part_name]
        if fixture is not None and fixture != declared:
            raise ValueError('KMR placement fixture conflicts with the recorded slot')
        fixture = declared
        dock = machine.get('KMR_pick_docking_poses', {}).get(part_name, dock)
    if fixture is None:
        raise KeyError('Recorded placement fixture or slot')
    if not all(_pose(value) for value in (transform, current_pose, base_pose)) or not _vector(fixture, 6):
        raise ValueError('KMR placement needs observed attachment, TCP, base and declared fixture')
    if not _at_dock(base_pose, dock):
        raise ValueError('KMR is not docked at the declared destination')
    return release_targets(scene, transform, current_pose, base_pose, [*fixture[:3], *quaternion(fixture[3:])])


def release_targets(scene: dict, transform: list, current_pose: list, base_pose: list,
                    destination: list) -> dict:
    """Calculate TCP poses that put the held part at a world pose, wherever the base stands.

    Args:
        scene: The captured owner configuration.
        transform: The observed grasp transform from TCP to part.
        current_pose: The observed TCP pose.
        base_pose: The observed base pose as position and quaternion.
        destination: The part's intended world pose as position and quaternion.

    Returns:
        Approach, target, retreat and transport poses. Reachability is decided
        by the motion planner and resting contact by observation after release.
    """
    from cais_spade_llm.recovery_framework.kmr_gazebo import inverse

    if not all(_pose(value) for value in (transform, current_pose, base_pose, destination)):
        raise ValueError('KMR release needs observed attachment, TCP, base and a destination pose')
    config = scene['KMR']['task_execution']
    target = compose(destination, inverse(transform))
    if rotate(target[3:], [0., 0., 1.])[2] > -.999:
        raise ValueError('KMR placement requires a downward-facing grasp')
    approach = target.copy()
    approach[2] = max(target[2] + config['minimum_pick_lift_m'],
                      config['cartesian_waypoints']['transport_tcp_position_in_base_frame_m'][2])
    retreat = target.copy()
    retreat[2] = max(target[2] + config['release_offsets_m'][0][2], approach[2])
    transport = transport_pose(scene, base_pose, current_pose)
    return {'target': target, 'approach': approach, 'retreat': retreat, 'destination': destination,
            'transfer_waypoints': transfer_waypoints(scene, base_pose, current_pose, approach),
            'transport': transport, 'withdraw_waypoints': transfer_waypoints(scene, base_pose, retreat, transport)}


def _calculate_cartesian_joints(config: dict, names: list, target: list,
                                seeds: list[list], limits: dict) -> dict | None:
    """Search bounded owner kinematics; the caller independently checks its FK.

    An unsuccessful finite search is missing evidence. It does not establish
    that the requested pose is unreachable, collision-free, or executable.
    """
    import numpy as np
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation
    from .kmr_motion import bounded_joints
    from cais_spade_llm.agents.shared_information.recovery_validation_protocol import recovery_validation_fingerprint

    cache = _cartesian_joint_calculation_cache.get()
    identity = recovery_validation_fingerprint([config, names, target, seeds, limits]) if cache is not None else None
    if cache is not None and identity in cache:
        return deepcopy(cache[identity])

    by_child = {row['child']: row for row in config['joints']}
    chain, child = [], config['reference_link']
    while child != config['root']:
        if child not in by_child or any(row['child'] == child for row in chain):
            return None
        row = by_child[child]
        if ('frozen_interval' in row and row['frozen_interval'][0] != row['frozen_interval'][1]
                or row['type'] not in {'fixed', 'revolute', 'continuous', 'prismatic'}):
            return None
        chain.append(row)
        child = row['parent']
    chain.reverse()
    if set(names) - {row['name'] for row in chain}:
        return None
    root = np.eye(4)
    root[:3, :3] = (Rotation.from_quat(config['root_quaternion']).as_matrix()
                       if 'root_quaternion' in config else Rotation.from_euler('xyz', config['root_rpy']).as_matrix())
    root[:3, 3] = config['root_xyz']
    origins = []
    for row in chain:
        matrix = np.eye(4)
        matrix[:3, :3] = Rotation.from_euler('xyz', row['rpy']).as_matrix()
        matrix[:3, 3] = row['xyz']
        origins.append(matrix)
    goal = Rotation.from_quat(target[3:])

    def residual(joints):
        values = dict(zip(names, joints, strict=True))
        matrix = root.copy()
        for row, origin in zip(chain, origins, strict=True):
            matrix = matrix @ origin
            position = values.get(row['name'], row.get('frozen_position', row.get('frozen_interval', [0.])[0]))
            change = np.eye(4)
            axis = np.asarray(row['axis'], dtype=float)
            if row['type'] in {'revolute', 'continuous'}:
                norm = np.linalg.norm(axis)
                if not norm:
                    raise ValueError('Recorded kinematic joint axis is zero')
                change[:3, :3] = Rotation.from_rotvec(axis / norm * position).as_matrix()
            elif row['type'] == 'prismatic':
                change[:3, 3] = axis * position
            matrix = matrix @ change
        return np.concatenate((matrix[:3, 3] - target[:3],
            (goal.inv() * Rotation.from_matrix(matrix[:3, :3])).as_rotvec()))

    tried = set()
    for seed in seeds:
        if not _vector(seed, len(names)) or not bounded_joints(names, seed, limits) or tuple(seed) in tried:
            continue
        tried.add(tuple(seed))
        fit = least_squares(residual, seed,
            bounds=([limits[name]['lower'] for name in names], [limits[name]['upper'] for name in names]),
            method='dogbox', max_nfev=500, ftol=1e-12, xtol=1e-12, gtol=1e-12)
        error = residual(fit.x)
        if np.linalg.norm(error[:3]) <= 1e-5 and np.linalg.norm(error[3:]) <= 1e-5:
            result = {'joint_positions': fit.x.tolist(), 'source': 'resource-owned bounded inverse kinematics',
                'solver': 'scipy.optimize.least_squares', 'method': 'dogbox', 'evaluations': int(fit.nfev),
                'solver_status': int(fit.status), 'seed': deepcopy(seed), 'max_nfev': 500,
                'collision_or_path_validated': False, 'execution_authorized': False}
            if cache is not None:
                cache[identity] = deepcopy(result)
            return result
    if cache is not None:
        cache[identity] = None
    return None


def check_kmr_primitive_capability(*, primitive: str, params: dict, task: dict,
                                   recovery_snapshot: dict, resource_evidence: dict) -> dict:
    """Check KMR capabilities without planning, transport, or live observations."""
    from cais_spade_llm.resources.environment_recovery import _result

    snapshot, evidence = recovery_snapshot, resource_evidence
    def answer(status, reason, **facts):
        return _result(status, reason, evidence, snapshot, primitive=primitive, params=deepcopy(params), **facts)
    root = f'/resources/{task.get("resource_jid")}/resource_evidence'
    def missing(reason, *refs):
        return {**answer('NEEDS_CONTEXT', reason), 'context_requests': [root + '/' + ref for ref in refs]}
    scene = evidence.get('inputs', {}).get('scene', {})
    kmr = scene.get('KMR', {})
    physical = evidence.get('snapshot', {})
    state = {**deepcopy(physical), **deepcopy(snapshot)}
    names, joints = state.get('joint_names'), state.get('joint_positions')
    config = state.get('continuous_motion_configuration') or evidence.get('continuous_motion_configuration')
    if (not isinstance(names, list) or names != kmr.get('arm_joint_names') or len(names) != 7
            or not _vector(joints, 7) or not isinstance(config, dict)):
        return missing('KMR capability requires its recorded seven-joint configuration and kinematic model',
                       'snapshot/joint_names', 'snapshot/joint_positions', 'snapshot/continuous_motion_configuration')
    from cais_spade_llm.recovery_framework.kmr_motion import bounded_joints
    limits = {row['name']: dict(zip(('lower', 'upper'), row['position_bounds'], strict=True))
              for row in config.get('joints', []) if 'position_bounds' in row}
    if not bounded_joints(names, joints, limits):
        return answer('INFEASIBLE', 'Recorded KMR arm configuration violates declared joint bounds')
    if primitive in {'open_gripper', 'close_gripper'}:
        if not _vector([kmr.get('gripper_stroke_m')], 1) or kmr['gripper_stroke_m'] <= 0:
            return missing('Declared gripper stroke is unavailable', 'inputs/scene/KMR/gripper_stroke_m')
        return answer('FEASIBLE', 'Declared KMR gripper capability is available')
    part = params.get('part_name', task.get('part_name'))
    if primitive in {'release_part', 'detach_part'}:
        held = state.get('held_part')
        if 'held_part' not in state:
            return missing('Recorded KMR custody is unavailable', 'snapshot/held_part')
        if held is None:
            return answer('INFEASIBLE', 'Recorded KMR has no held part')
        attachment = evidence.get('attachment') or {}
        transform = attachment.get('grasp_transform', state.get('grasp_transform'))
        if attachment.get('attached') is not True or not _pose(transform):
            return missing('Recorded KMR attachment transform is required', 'attachment')
        if params.get('transform') is not None and params['transform'] != transform:
            return answer('INFEASIBLE', 'Requested transform conflicts with recorded KMR attachment')
        return answer('FEASIBLE', 'Recorded custody and attachment support release capability; placement remains separate', held_part=held)
    if primitive == 'compute_place_targets':
        attachment = evidence.get('attachment') or {}
        transform = attachment.get('grasp_transform', state.get('grasp_transform'))
        if (part is None or state.get('held_part') is None or attachment.get('attached') is not True
                or not _pose(transform)):
            return missing('KMR placement calculation needs recorded part custody and attachment',
                           'snapshot/held_part', 'attachment')
        if state['held_part'] != part or params.get('transform') != transform:
            return answer('INFEASIBLE', 'KMR placement conflicts with recorded part custody or attachment')
        try:
            machine, source = recorded_location(scene, state['resource_location'])
            targets = place_targets(scene, machine, transform, state['current_pose'], state['base_pose'],
                                    machine.get('workholding_pose'), part_name=part)
        except KeyError as exc:
            return missing('KMR placement calculation lacks recorded input: ' + str(exc), 'inputs/scene', 'snapshot')
        except ValueError as exc:
            return answer('INFEASIBLE', str(exc))
        return answer('FEASIBLE', 'KMR owner calculated placement targets; access and rest remain separate',
                      calculated_targets=targets, source=source)
    if primitive in {'grasp_part', 'attach_part', 'detect_parts', 'compute_pick_targets'}:
        if part is None:
            return missing('The originating part binding is required', 'state/part_name')
        recorded = evidence.get('observed_parts', {}).get(part, {})
        pose = recorded.get('observed_pose', recorded.get('current_pose'))
        if not _pose(pose):
            return missing('Recorded KMR part observation is unavailable', 'observed_parts/' + part)
        if primitive in {'grasp_part', 'attach_part'}:
            if 'held_part' not in state:
                return missing('Recorded KMR custody is unavailable', 'snapshot/held_part')
            if state['held_part'] is not None and state['held_part'] != part:
                return answer('INFEASIBLE', 'KMR already holds a conflicting part')
        if primitive == 'compute_pick_targets':
            try:
                location = recorded.get('resource_location', state.get('resource_location'))
                if not isinstance(location, str):
                    return missing('KMR pickup requires an exact recorded location', 'snapshot/resource_location')
                targets = pick_targets(scene, part, pose, state['current_pose'], state['base_pose'], params.get('initial'), location=location)
            except KeyError as exc:
                return missing('KMR pickup calculation lacks recorded input: ' + str(exc), 'snapshot', 'inputs/scene/Storage')
            except ValueError as exc:
                return answer('INFEASIBLE', str(exc))
            return answer('FEASIBLE', 'KMR owner calculated pickup targets from recorded facts', calculated_targets=targets)
        return answer('FEASIBLE', 'Recorded part identity and owner capability are available; custody outcome remains separate')
    if primitive in {'move_joints', 'rotate_joint'}:
        if kmr.get('task_execution', {}).get('cartesian_motion_only') is True:
            return answer('INFEASIBLE', 'Recorded KMR task_execution requires Cartesian arm motion')
        target = params.get('joints')
        if primitive == 'rotate_joint':
            if params.get('joint_name') not in names or not _vector([params.get('delta_deg')], 1):
                return answer('INFEASIBLE', 'KMR relative joint parameters are invalid')
            target = deepcopy(joints)
            target[names.index(params['joint_name'])] += math.radians(params['delta_deg'])
        if not _vector(target, 7):
            return answer('INFEASIBLE', 'KMR joint target must contain seven finite values')
        return answer('FEASIBLE' if bounded_joints(names, target, limits) else 'INFEASIBLE',
                      'Checked exact KMR joint target against declared bounds; motion interval remains separate')
    if primitive == 'move_to_named_pose':
        if params.get('pose_name') not in {'home', 'transport'}:
            return answer('INFEASIBLE', 'Unknown KMR arm pose')
        if params['pose_name'] == 'home' and 'held_part' not in state:
            return missing('KMR home requires explicitly recorded custody', 'snapshot/held_part')
        if params['pose_name'] == 'home' and state['held_part'] is not None:
            return answer('INFEASIBLE', 'KMR downward home requires an empty gripper')
        try:
            target = transport_pose(scene, state['base_pose'], state['current_pose'])
        except (KeyError, ValueError) as exc:
            return missing(str(exc), 'snapshot/base_pose', 'snapshot/current_pose', 'inputs/scene/KMR')
        return answer('FEASIBLE', 'KMR owner calculated its declared transport target; motion remains separate', target=target)
    if primitive == 'move_base':
        if not _vector(params.get('target_pose'), 3):
            return answer('INFEASIBLE', 'KMR move_base requires three finite world-frame coordinates')
        waypoints = params.get('waypoints') or []
        if not isinstance(waypoints, list) or not all(_vector(value, 3) for value in waypoints):
            return answer('INFEASIBLE', 'KMR base waypoints require three finite world-frame coordinates')
        if not _pose(state.get('base_pose')):
            return missing('KMR base observation is unavailable', 'snapshot/base_pose')
        if 'held_part' not in state:
            return missing('KMR custody is unavailable', 'snapshot/held_part')
        attachment = resource_evidence.get('attachment') or {}
        if state['held_part'] is None:
            if attachment.get('attached') is True or params.get('transform') is not None:
                return answer('INFEASIBLE', 'Empty KMR custody conflicts with an attachment or requested transform')
        else:
            transform = attachment.get('grasp_transform')
            if (attachment.get('attached') is not True or not _pose(transform)
                    or not _pose(params.get('transform'))):
                return missing('Loaded KMR movement requires a recorded attachment transform',
                               'attachment/attached', 'attachment/grasp_transform')
            identities = [attachment[key] for key in ('part_name', 'held_part') if key in attachment]
            if not identities:
                return missing('Loaded KMR attachment part identity is unavailable', 'attachment/held_part')
            if any(value != state['held_part'] for value in identities) or params['transform'] != transform:
                return answer('INFEASIBLE', 'KMR movement attachment conflicts with recorded custody or requested transform')
        return answer('FEASIBLE', 'KMR base capability has bound coordinates and custody; route and clearance remain separate')
    if primitive in {'move_cartesian', 'move_relative'}:
        target = params.get('target')
        if primitive == 'move_relative':
            if not _vector(params.get('offset'), 3) or not _pose(state.get('current_pose')):
                return missing('KMR relative target needs recorded TCP and three offsets', 'snapshot/current_pose')
            target = [*[value + delta for value, delta in zip(state['current_pose'][:3], params['offset'], strict=True)],
                      *state['current_pose'][3:]]
        if not _pose(target):
            return answer('INFEASIBLE', 'KMR Cartesian target requires xyz/xyzw with unit orientation')
        if params.get('seed') is not None and (not _vector(params['seed'], 7) or not bounded_joints(names, params['seed'], limits)):
            return answer('INFEASIBLE', 'KMR Cartesian seed violates the seven-joint declaration')
        from cais_spade_llm.resources.continuous_motion import interval, kinematic_transforms, representative_pose
        from cais_spade_llm.agents.shared_information.recovery_validation_protocol import recovery_validation_fingerprint

        candidates = [joints]
        if params.get('seed') is not None:
            candidates.append(params['seed'])
        for recorded in evidence.get('primitive_capability_evidence', {}).get(primitive, []):
            if (recorded.get('configuration_fingerprint') == recovery_validation_fingerprint(config)
                    and recorded.get('source') and recorded.get('params') == params
                    and _vector(recorded.get('joint_positions'), 7)):
                candidates.append(recorded['joint_positions'])
        calculated = None
        for index in range(2):
            if index == 1:
                calculated = _calculate_cartesian_joints(config, names, target,
                    [*candidates, kmr.get('task_execution', {}).get('carrying_arm_configuration'),
                     *kmr.get('task_execution', {}).get('ik_seed_configurations', [])], limits)
                if calculated is None:
                    break
                candidates = [calculated['joint_positions']]
            for candidate in candidates:
                if not bounded_joints(names, candidate, limits):
                    continue
                transforms = kinematic_transforms(config, {name: interval(value)
                    for name, value in zip(names, candidate, strict=True)})
                pose = representative_pose(transforms[config['reference_link']])
                # Independently check a calculated witness with the same owner
                # FK used for recorded witnesses; a solver status is not proof.
                if (math.dist(pose[:3], target[:3]) <= 1e-5
                        and abs(sum(a*b for a, b in zip(pose[3:], target[3:], strict=True))) >= 1 - 1e-10):
                    return answer('FEASIBLE',
                        ('Calculated' if calculated else 'Recorded') + ' KMR joint witness reproduces the requested TCP by owner FK; motion interval remains separate',
                        joint_positions=deepcopy(candidate), calculated_pose=pose,
                        configuration_fingerprint=recovery_validation_fingerprint(config),
                        **({'endpoint_calculation': calculated} if calculated else {}))
        return missing('Bounded KMR endpoint search found no independently verified joint witness; reachability is unresolved',
                       'primitive_capability_evidence/' + primitive)
    return missing('KMR owner calculation requires additional recorded parameters', 'state', 'primitive_capability_evidence/' + primitive)


def route_between(scene: dict, part: str | None, source: str, target: str) -> list[list[float]]:
    """Return the exact declared KMR route without selecting a recovery event."""
    for row in scene['KMR']['predefined_route_waypoints']:
        if row['resources'] == [source, target]:
            route = deepcopy(row['poses'])
            if source == 'Storage':
                pick_pose = (scene['Storage'].get('KMR_pick_docking_poses') or {}).get(part)
                if pick_pose and math.dist(pick_pose[:2], route[0][:2]) > .01:
                    route = [list(pick_pose), *route[1:]] if target == 'M2' else [list(pick_pose), *route]
            return route
        if row['resources'] == [target, source]:
            return list(reversed(deepcopy(row['poses'])))
    raise ValueError(f'No configured KMR route from {source} to {target}')


def coordinate_parameters(bindings: dict, scene: dict) -> tuple[dict, dict]:
    """Bind a requested named destination to its declared docking coordinates.

    Args:
        bindings: The proposed contributor's parameters, never a previous task.
        scene: The captured owner configuration.

    Returns:
        Independent parameter values and their declaration sources. This does
        not select a destination, calculate a route, or establish arrival.
    """
    parameters, sources = deepcopy(bindings), {}
    location = bindings.get('observed_resource_location')
    if location is not None:
        try:
            declaration, reference = recorded_location(scene, location)
        except KeyError as exc:
            raise ValueError('KMR requested location lacks one declared docking pose: ' + str(location)) from exc
        pose = declaration.get('KMR_docking_pose')
        reference += '/KMR_docking_pose'
        if not _vector(pose, 6):
            raise ValueError('KMR requested location lacks one declared docking pose: ' + str(location))
        for name, value in zip(('x', 'y', 'yaw'), (pose[0], pose[1], pose[5]), strict=True):
            if name not in parameters:
                parameters[name] = value
                sources[name] = {'ref': reference, 'value': value, 'resource_location': location}
        # Use the same named-destination test as move_base. Supplied coordinates
        # are preserved, including explicit null; conflicting values are errors.
        values = [parameters[name] for name in ('x', 'y', 'yaw')]
        if (not _vector(values, 3) or math.dist(values[:2], pose[:2]) > .001
                or abs(math.atan2(math.sin(values[2] - pose[5]), math.cos(values[2] - pose[5]))) > .001):
            raise ValueError('KMR coordinate parameters conflict with the declared destination: ' + str(location))
    if not _vector([parameters.get(name) for name in ('x', 'y', 'yaw')], 3):
        raise ValueError('KMR coordinate movement requires finite x, y, yaw or a declared named destination')
    return parameters, sources


def prepare_kmr_primitive_parameters(*, task: dict, recovery_des_model: dict,
                                     recovery_snapshot: dict, resource_evidence: dict,
                                     resource_jid: str) -> dict:
    """Resolve each contributor independently using the same recorded snapshot."""
    from cais_spade_llm.agents.shared_information.recovery_validation_protocol import recovery_validation_fingerprint
    from cais_spade_llm.resources.recovery_feasibility import derive_primitive_union
    from cais_spade_llm.resources.recovery_parameter_evidence import prepare_resource_primitive_parameters

    support = derive_primitive_union(task=task, recovery_des_model=recovery_des_model, recovery_snapshot=recovery_snapshot)
    witnesses, missing, conflicts, calculations = [], [], [], []
    scene = resource_evidence.get('inputs', {}).get('scene', {})
    snapshot = {**deepcopy(resource_evidence.get('snapshot', {})), **deepcopy(recovery_snapshot)}
    root = f'/resources/{resource_jid}/resource_evidence'
    seen = set()
    for transition in support['matching_transitions']:
        bindings = transition['parameter_bindings']
        key = recovery_validation_fingerprint([transition['event_name'], bindings])
        if key in seen:
            continue
        seen.add(key)
        evidence = deepcopy(resource_evidence)
        state = evidence.setdefault('state', {})
        part = bindings.get('part_name')
        row = evidence.get('observed_parts', {}).get(part, {})
        pose = row.get('observed_pose', row.get('current_pose'))
        outputs = []
        resolved_bindings, binding_sources = deepcopy(bindings), {}
        before_state = deepcopy(state)
        try:
            if part is not None and _pose(pose):
                state['initial'] = deepcopy(pose)
            if transition['event_name'] == 'pick_approach':
                location = bindings['origin_resource_location']
                declaration, _ = recorded_location(scene, location)
                state['pickup_pose'] = pickup_docking_pose(scene, location, part)
                state['pickup_waypoints'] = []  # The declared owner dispatch uses no intermediate pickup waypoints.
            if transition['event_name'] in {'pick_approach', 'pick_part'}:
                targets = pick_targets(scene, part, pose, snapshot['current_pose'], snapshot['base_pose'],
                                       location=bindings['origin_resource_location'])
                state.setdefault('previous', {})['pick_targets'] = deepcopy(targets)
                outputs.append({'event_name': transition['event_name'], 'parameter_bindings': deepcopy(bindings),
                                'store_as': 'pick', 'output': targets, 'source': root + '/observed_parts/' + part})
            if transition['event_name'] == 'move_to_resource':
                target, source = bindings['target_resource'], bindings['source_resource']
                destination = (scene['Storage']['KMR_docking_pose'] if target == 'Storage' else
                               next(row['KMR_docking_pose'] for row in scene['machines'] if row['resource_id'] == target))
                state['base_target_pose'] = [destination[0], destination[1], destination[5]]
                state['base_waypoints'] = route_between(scene, part, source, target)[1:-1]
            elif transition['event_name'] == 'move_to_location':
                resolved_bindings, binding_sources = coordinate_parameters(bindings, scene)
                state['base_target_pose'] = [resolved_bindings[key] for key in ('x', 'y', 'yaw')]
                state['base_waypoints'] = []
            attachment = evidence.get('attachment') or {}
            if 'held_part' in snapshot and snapshot['held_part'] is None and attachment.get('attached') is not True:
                evidence.setdefault('parameter_applicability', {}).setdefault('move_base', {})['transform'] = {
                    'applicable': False,
                    'reason': 'Recorded held_part is null; the owner requires transform only for loaded movement',
                    'source': root + '/snapshot/held_part',
                    'snapshot_fingerprint': recovery_validation_fingerprint(snapshot),
                }
            if attachment.get('attached') is True and _pose(attachment.get('grasp_transform')):
                if snapshot.get('held_part') is not None and part in (None, snapshot['held_part']):
                    state.setdefault('previous', {})['grasp_transform'] = deepcopy(attachment['grasp_transform'])
            if transition['event_name'] in {'place_approach', 'place_release'}:
                machine, reference = recorded_location(scene, bindings['destination_location'])
                transform = state['previous']['grasp_transform']
                targets = place_targets(scene, machine, transform, snapshot['current_pose'],
                                        snapshot['base_pose'], machine.get('workholding_pose'), part_name=part)
                state.setdefault('previous', {})['place_targets'] = deepcopy(targets)
                outputs.append({'event_name': transition['event_name'], 'parameter_bindings': deepcopy(bindings),
                                'store_as': 'place', 'output': targets, 'source': root + reference})
        except (KeyError, ValueError, TypeError, StopIteration) as exc:
            missing.append({'ref': root + '/snapshot', 'event_name': transition['event_name'],
                            'parameter_bindings': deepcopy(bindings), 'reason': str(exc)})
        # Per-contributor state is scoped to the recorded task and part. Its
        # calculation never replaces another contributor's recorded state.
        for name in before_state.keys() & state.keys():
            if recovery_validation_fingerprint(before_state[name]) != recovery_validation_fingerprint(state[name]):
                conflicts.append({'ref': root + '/state/' + name, 'event_name': transition['event_name'],
                                  'parameter_bindings': deepcopy(bindings), 'reason': 'Calculated KMR state conflicts with its recorded source'})
        descriptor = deepcopy(recovery_des_model)
        descriptor['events'] = [event for event in descriptor['events']
            if event['event_name'] == transition['event_name'] and all(
                name not in bindings or 'equals' not in declaration or
                recovery_validation_fingerprint(declaration['equals']) == recovery_validation_fingerprint(bindings[name])
                for name, declaration in event.get('parameter_bindings', {}).items())]
        for event in descriptor['events']:
            if (resource_evidence.get('endpoint_evidence_policy_version') in (1, 2)
                    and event['event_name'] == 'move_to_location' and snapshot.get('held_part') is not None):
                steps = event.get('program_variants', {}).get('carrying', {}).get('steps', [])
                original = event.get('primitive_support', [])
                if (steps and [(row['id'], row['op']) for row in steps]
                        == [(row['id'], row['primitive']) for row in original]):
                    # The carrying variant adds the owner's attachment argument;
                    # it does not change the independently derived primitive set.
                    event['primitive_support'] = [dict(deepcopy(row), primitive=row['op']) for row in steps]
        prepared = prepare_resource_primitive_parameters(task=task, recovery_des_model=descriptor,
            recovery_snapshot=recovery_snapshot, resource_evidence=evidence, resource_jid=resource_jid,
            calculated_step_outputs=outputs)
        for witness in prepared['parameter_witnesses']:
            witness['model_fingerprint'] = recovery_validation_fingerprint(recovery_des_model)
            witness['source_fingerprint'] = recovery_validation_fingerprint(resource_evidence)
        witnesses.extend(prepared['parameter_witnesses'])
        missing.extend(prepared['missing_facts'])
        conflicts.extend(prepared['conflicts'])
        calculations.append({'event_name': transition['event_name'], 'parameter_bindings': deepcopy(bindings),
            'resolved_parameter_bindings': resolved_bindings, 'parameter_sources': binding_sources,
            'calculated_state': deepcopy(state), 'step_outputs': outputs,
            'source_fingerprint': recovery_validation_fingerprint(resource_evidence),
            'snapshot_fingerprint': recovery_validation_fingerprint(recovery_snapshot), 'command_sent': False})
    return {'parameter_witnesses': witnesses, 'missing_facts': missing, 'conflicts': conflicts,
            'context_requests': list(dict.fromkeys(row['ref'] for row in missing)),
            'calculated_resource_evidence': deepcopy(resource_evidence),
            'resource_evidence_fingerprint': recovery_validation_fingerprint(resource_evidence),
            'parameter_calculations': calculations}
