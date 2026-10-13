from __future__ import annotations

"""KMR owner calculations over explicitly supplied, recorded observations."""

import math
from copy import deepcopy

from cais_spade_llm.recovery_framework.geometry import compose, quaternion, rotate


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


def pick_targets(scene: dict, part: str, observed: list, current_pose: list,
                 base_pose: list, initial: list | None = None) -> dict:
    """Calculate pickup targets from the same facts used by the execution owner."""
    from cais_spade_llm.recovery_framework.kmr_gazebo import _nearest_pick_orientation

    if not all(_pose(value) for value in (observed, current_pose, base_pose)):
        raise ValueError('KMR pickup requires complete recorded part, TCP, and base poses')
    if initial is not None and (not _pose(initial) or math.dist(observed[:3], initial[:3]) > .01):
        raise ValueError('Storage part moved before target computation')
    config = scene['KMR']['task_execution']
    orientation = _nearest_pick_orientation(current_pose[3:], scene['Storage']['KMR_pick_orientations_xyzw'][part],
                                            config['orientation_tolerance_rad'])
    target = [*observed[:3], *orientation]
    target[2] += config['grasp_height_m']
    approach = target.copy()
    approach[2] += scene['Storage']['KMR_pick_approach_clearance_m'][part]
    clearance = target.copy()
    clearance[2] += config['minimum_pick_lift_m']
    lift = transport_pose(scene, base_pose, target)
    if lift[2] < clearance[2]:
        raise ValueError('Saved KMR carrying height does not clear the picked part')
    return {'target': target, 'approach': approach, 'lift': lift, 'lift_waypoints': [clearance],
            'retreat': lift.copy(), 'seed': list(scene['Storage']['KMR_pick_arm_configurations'][part]),
            'carrying': list(config['carrying_arm_configuration'])}


def place_targets(scene: dict, machine: dict, transform: list, current_pose: list,
                  base_pose: list, fixture: list) -> dict:
    """Calculate placement poses without establishing workholding clearance."""
    from cais_spade_llm.recovery_framework.kmr_gazebo import _at_dock, inverse

    if not all(_pose(value) for value in (transform, current_pose, base_pose)) or not _vector(fixture, 6):
        raise ValueError('KMR placement needs observed attachment, TCP, base and declared fixture')
    if not _at_dock(base_pose, machine['KMR_docking_pose']):
        raise ValueError('KMR is not docked at the declared destination')
    config = scene['KMR']['task_execution']
    destination = [*fixture[:3], *quaternion(fixture[3:])]
    target = compose(destination, inverse(transform))
    target[2] += .003
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
                targets = pick_targets(scene, part, pose, state['current_pose'], state['base_pose'], params.get('initial'))
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
        for candidate in candidates:
            if not bounded_joints(names, candidate, limits):
                continue
            transforms = kinematic_transforms(config, {name: interval(value)
                for name, value in zip(names, candidate, strict=True)})
            calculated = representative_pose(transforms[config['reference_link']])
            # This is an endpoint kinematic witness, not a collision or path proof.
            if (math.dist(calculated[:3], target[:3]) <= 1e-5
                    and abs(sum(a*b for a, b in zip(calculated[3:], target[3:], strict=True))) >= 1 - 1e-10):
                return answer('FEASIBLE', 'Recorded KMR joint witness reproduces the requested TCP by owner FK; motion interval remains separate',
                              joint_positions=deepcopy(candidate), calculated_pose=calculated,
                              configuration_fingerprint=recovery_validation_fingerprint(config))
        return missing('KMR Cartesian capability requires recorded inverse-kinematic or matching native path evidence',
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
    declarations = [(row['KMR_docking_pose'], f'/inputs/scene/machines/{index}/KMR_docking_pose')
                    for index, row in enumerate(scene.get('machines', []))
                    if row.get('resource_id') == location and 'KMR_docking_pose' in row]
    if location == 'Storage' and 'KMR_docking_pose' in scene.get('Storage', {}):
        declarations.append((scene['Storage']['KMR_docking_pose'], '/inputs/scene/Storage/KMR_docking_pose'))
    if location is not None:
        if len(declarations) != 1 or not _vector(declarations[0][0], 6):
            raise ValueError('KMR requested location lacks one declared docking pose: ' + str(location))
        pose, reference = declarations[0]
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
                state['pickup_pose'] = deepcopy(scene['Storage']['KMR_pick_docking_poses'][part])
                state['pickup_waypoints'] = []  # The declared owner dispatch uses no intermediate pickup waypoints.
            if transition['event_name'] in {'pick_approach', 'pick_part'}:
                targets = pick_targets(scene, part, pose, snapshot['current_pose'], snapshot['base_pose'])
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
                machine = next(row for row in scene['machines'] if row['resource_id'] == bindings['destination_location'])
                transform = state['previous']['grasp_transform']
                targets = place_targets(scene, machine, transform, snapshot['current_pose'],
                                        snapshot['base_pose'], machine['workholding_pose'])
                state.setdefault('previous', {})['place_targets'] = deepcopy(targets)
                outputs.append({'event_name': transition['event_name'], 'parameter_bindings': deepcopy(bindings),
                                'store_as': 'place', 'output': targets, 'source': root + '/inputs/scene/machines'})
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
