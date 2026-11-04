from __future__ import annotations

"""KMR owner calculations over explicitly supplied, recorded observations."""

import math
from collections.abc import Callable
from copy import deepcopy

from cais_spade_llm.recovery_framework.geometry import compose, quaternion, rotate

# The bounded joint search lives with every resource's reach check; these names stay for callers and tests.
from cais_spade_llm.resources.kinematic_reach import (  # noqa: F401
    JOINT_SEARCH_CACHE as _cartesian_joint_calculation_cache,
)
from cais_spade_llm.resources.kinematic_reach import (
    bounded_joint_search as _calculate_cartesian_joints,
)
from cais_spade_llm.resources.kinematic_reach import pose_matrix, reach, with_root

# How far a release's approach and retreat may move toward the arm, in order, when straight above is out of reach.
RELEASE_SHIFTS_M = (.02, .04, .06, .08, .1)


def _vector(value, length: int) -> bool:
    return (isinstance(value, list) and len(value) == length
            and all(type(item) in (int, float) and math.isfinite(item) for item in value))


def _pose(value) -> bool:
    return _vector(value, 7) and math.isclose(sum(item * item for item in value[3:]), 1., abs_tol=1e-5)


def _pose_list(value):
    """Return a pose as ``[x, y, z, qx, qy, qz, qw]``, whether recorded so or projected by its named fields."""
    if isinstance(value, dict):
        return [value.get(axis) for axis in ('x', 'y', 'z', 'qx', 'qy', 'qz', 'qw')]
    return value


def kmr_docks(scene: dict) -> dict[str, list[float]]:
    """Return the KMR's declared docks as ``[x, y, yaw]``, by the place each belongs to."""
    rows = {**scene, **{row.get('resource_id'): row for row in scene.get('machines', [])}}
    return {name: [row['KMR_docking_pose'][0], row['KMR_docking_pose'][1], row['KMR_docking_pose'][5]]
            for name, row in rows.items() if isinstance(row, dict) and _vector(row.get('KMR_docking_pose'), 6)}


def kmr_place_at(scene: dict, base_pose: list) -> str | None:
    """Name the place whose declared dock a base stands on, for a part's own dock too; None when it stands on none."""
    from cais_spade_llm.recovery_framework.kmr_gazebo import _at_dock

    rows = {**scene, **{row.get('resource_id'): row for row in scene.get('machines', [])}}
    for name, row in rows.items():
        if not isinstance(row, dict):
            continue
        docks = [dock for dock in (row.get('KMR_docking_pose'), *(row.get('KMR_pick_docking_poses') or {}).values())
                 if isinstance(dock, list)]
        if any(_at_dock(base_pose, dock) for dock in docks):
            return name
    return None


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


KMR_NAMED_POSES = ("transport", "home")


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


def pick_heights(scene: dict, part: str | None, location: str | None, target_z: float) -> dict[str, float]:
    """Return the tool heights a pick at ``target_z`` comes in at and first lifts to."""
    config = scene['KMR']['task_execution']
    rows = {**scene, **{row.get('resource_id'): row for row in scene.get('machines', [])}}
    calibrated = (rows.get(location) or {}).get('KMR_pick_approach_clearance_m') or {}
    return {'approach': target_z + calibrated.get(part, config['approach_clearance_m']),
            'clearance': target_z + config['minimum_pick_lift_m']}


def release_heights(scene: dict, target_z: float) -> dict[str, float]:
    """Return the tool heights a release at ``target_z`` comes in and leaves at.

    The arm stays at its carrying height until it is over the spot, which a low one such as a Storage slot needs.
    """
    config = scene['KMR']['task_execution']
    approach = max(target_z + config['minimum_pick_lift_m'],
                   config['cartesian_waypoints']['transport_tcp_position_in_base_frame_m'][2])
    return {'approach': approach, 'retreat': max(target_z + config['release_offsets_m'][0][2], approach)}


def tool_height(scene: dict, part_z: float) -> float:
    """Return the tool height that grasps or releases a part at ``part_z``."""
    return part_z + scene['KMR']['task_execution']['grasp_height_m']


def transfer_heights(scene: dict, transfer: str, part: str | None, place: str | None, part_z: float) -> list[float]:
    """Return the tool heights above a part at ``part_z`` that a ``pick`` or ``place`` of it needs, highest first."""
    target_z = tool_height(scene, part_z)
    heights = pick_heights(scene, part, place, target_z) if transfer == 'pick' else release_heights(scene, target_z)
    return sorted(set(heights.values()), reverse=True)


def pick_targets(scene: dict, part: str, observed: list, current_pose: list,
                 base_pose: list, initial: list | None = None, *, location: str | None = 'Storage') -> dict:
    """Calculate pickup targets from the same facts used by the execution owner.

    With no ``location``, the part lies somewhere the KMR has no calibration or dock for, such as a staging tray,
    and the targets come from its observed pose alone.
    """
    from cais_spade_llm.recovery_framework.kmr_gazebo import _nearest_pick_orientation

    if not all(_pose(value) for value in (observed, current_pose, base_pose)):
        raise ValueError('KMR pickup requires complete recorded part, TCP, and base poses')
    if initial is not None and (not _pose(initial) or math.dist(observed[:3], initial[:3]) > .01):
        raise ValueError('Storage part moved before target computation')
    config = scene['KMR']['task_execution']
    source = {} if location is None else recorded_location(scene, location)[0]
    calibrated = ('KMR_pick_orientations_xyzw', 'KMR_pick_approach_clearance_m', 'KMR_pick_arm_configurations')
    if any(field in source for field in calibrated):
        choices = source['KMR_pick_orientations_xyzw'][part]
        clearance_m = source['KMR_pick_approach_clearance_m'][part]
        seed = source['KMR_pick_arm_configurations'][part]
    else:
        # These are owner-calculated targets, not inherited geometric proof.
        # Incomplete location-specific calibration must not use this branch.
        if location is not None:
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
    heights = pick_heights(scene, part, location, target[2])
    approach = target.copy()
    approach[2] = heights['approach']
    clearance = target.copy()
    clearance[2] = heights['clearance']
    lift = transport_pose(scene, base_pose, target)
    if lift[2] < clearance[2]:
        raise ValueError('Saved KMR carrying height does not clear the picked part')
    # A part away from where the arm faces, such as in a machine beside the dock, is reached by turning the arm about
    # its mount, as a release is; a straight line would wind the arm back over itself. A part in front of the arm,
    # as at Storage, turns less than the configured minimum and has no waypoints.
    return {'target': target, 'approach': approach,
            'approach_waypoints': transfer_waypoints(scene, base_pose, current_pose, approach),
            'lift': lift, 'lift_waypoints': [clearance, *transfer_waypoints(scene, base_pose, clearance, lift)],
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


def release_shift(scene: dict, base_pose: list, target: list, reaches: Callable[[list], bool]) -> list[float] | None:
    """Return how far a release's approach and retreat move toward the arm, in world x and y.

    A release comes straight down on its target where the arm reaches the point above it. The gripper stays downward,
    which costs the arm reach as it rises, so near the end of its reach it can touch a spot and not the point above.
    There the approach and retreat start closer to the arm and the last move slants; the target and the gripper's
    orientation stay.

    Args:
        scene: The captured owner configuration.
        base_pose: The base pose as position and quaternion.
        target: The release's TCP pose.
        reaches: Whether the arm reaches a TCP pose from this base.

    Returns:
        The shift, or None when no approach within ``RELEASE_SHIFTS_M`` is reachable.
    """
    arm = compose(base_pose, [*scene['KMR']['arm_mount_xyz'], 0., 0., 0., 1.])
    heights = set(release_heights(scene, target[2]).values())
    distance = math.dist(arm[:2], target[:2])

    def shift(length: float) -> list[float]:
        return [(arm[axis] - target[axis]) * length / distance for axis in range(2)] if length and distance else [0., 0.]

    def fits(length: float) -> bool:
        x, y = (target[axis] + shift(length)[axis] for axis in range(2))
        return all(reaches([x, y, z, *target[3:]]) for z in heights)

    if fits(0.):
        return shift(0.)
    if not fits(RELEASE_SHIFTS_M[-1]):
        return None
    return shift(next(length for length in RELEASE_SHIFTS_M if fits(length)))


def release_targets(scene: dict, transform: list, current_pose: list, base_pose: list,
                    destination: list, *, reaches: Callable[[list], bool] | None = None) -> dict:
    """Calculate TCP poses that put the held part at a world pose, wherever the base stands.

    Args:
        scene: The captured owner configuration.
        transform: The observed grasp transform from TCP to part.
        current_pose: The observed TCP pose.
        base_pose: The observed base pose as position and quaternion.
        destination: The part's intended world pose as position and quaternion.
        reaches: Whether the arm reaches a TCP pose from this base, which lets ``release_shift`` move the approach
            and retreat; without it they stay straight above the target.

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
    heights = release_heights(scene, target[2])
    approach = target.copy()
    approach[2] = heights['approach']
    retreat = target.copy()
    retreat[2] = heights['retreat']
    shift = release_shift(scene, base_pose, target, reaches) if reaches is not None else None
    for pose in (approach, retreat):
        pose[:2] = [value + delta for value, delta in zip(pose[:2], shift or [0., 0.], strict=True)]
    transport = transport_pose(scene, base_pose, current_pose)
    return {'target': target, 'approach': approach, 'retreat': retreat, 'destination': destination,
            'transfer_waypoints': transfer_waypoints(scene, base_pose, current_pose, approach),
            'transport': transport, 'withdraw_waypoints': transfer_waypoints(scene, base_pose, retreat, transport)}


def _arm_reaches(config: dict, base_pose: list) -> Callable[[list], bool]:
    """Return whether the arm reaches a TCP pose from a base, by the search the RA's reach check uses."""
    return lambda pose: reach(config, pose, root=pose_matrix(base_pose))['feasibility_status'] == 'FEASIBLE'


def _standing_at(state: dict, evidence: dict) -> str | None:
    """Name where the base stands now, which a base move earlier in the same program has changed."""
    base = state.get('base_pose')
    if _pose(base):
        return kmr_place_at(evidence.get('inputs', {}).get('scene', {}), base)
    return state.get('resource_location')


def _release_pose(params: dict, task: dict, state: dict, transform: list, resource_evidence: dict,
                  physical_evidence: dict | None) -> list | None:
    """Return the world pose a release puts the held part at, or None for the dock's own recorded fixture.

    An explicit ``part_pose`` is that pose; with no quaternion the part keeps the orientation the grasp holds it
    at. Otherwise the event's stated ``part_location`` names the place, and, as for an arm, the recorded geometry
    of that place gives the pose, in the zone the event names.

    Raises:
        KeyError, ValueError, TypeError, IndexError: The recorded geometry does not give one supported pose.
    """
    from cais_spade_llm.resources.robot.recovery_capability import (
        _destination_geometry,
        _stated_zone_origin,
        _support_place_pose,
    )

    explicit = params.get('part_pose')
    if explicit is not None:
        if not isinstance(explicit, list) or not (_vector(explicit, 3) or _vector(explicit, 7)):
            raise ValueError('KMR part_pose needs finite x, y, z and an optional quaternion')
        return explicit if len(explicit) == 7 else [*explicit, *compose(state['current_pose'], transform)[3:]]
    destination = (task.get('expected_end_state') or {}).get('part_location')
    if not physical_evidence or not isinstance(destination, str) or destination == _standing_at(state, resource_evidence):
        return None
    target = _destination_geometry(resource_evidence, destination, task.get('part_name'))
    if target is None:
        raise KeyError('placement geometry of ' + destination)
    zone = _stated_zone_origin(task, destination, resource_evidence, physical_evidence)
    if zone is not None:
        target['target_origin_pose'] = zone
    return _support_place_pose(part=task.get('part_name'), destination=destination, target=target,
                               physical_evidence=physical_evidence)['part_pose']


def check_kmr_primitive_capability(*, primitive: str, params: dict, task: dict,
                                   recovery_snapshot: dict, resource_evidence: dict,
                                   physical_evidence: dict | None = None) -> dict:
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
    # The symbolic current_pose of a KMR stopped between places is its base; the tool pose stays in its evidence.
    if not _pose(state.get('current_pose')) and 'current_pose' in physical:
        state['current_pose'] = deepcopy(physical['current_pose'])
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
            destination = _release_pose(params, task, state, transform, evidence, physical_evidence)
        except (KeyError, ValueError, TypeError, IndexError) as exc:
            return missing('KMR placement calculation lacks recorded input: ' + str(exc), 'inputs/scene', 'snapshot')
        try:
            if destination is not None:
                source = task.get('expected_end_state', {}).get('part_location', 'part_pose')
                targets = release_targets(scene, transform, state['current_pose'], state['base_pose'], destination,
                                          reaches=_arm_reaches(config, state['base_pose']))
            else:
                machine, source = recorded_location(scene, _standing_at(state, evidence))
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
        pose = _pose_list(recorded.get('observed_pose', recorded.get('current_pose')))
        if not _pose(pose):
            return missing('Recorded KMR part observation is unavailable', 'observed_parts/' + part)
        if primitive in {'grasp_part', 'attach_part'}:
            if 'held_part' not in state:
                return missing('Recorded KMR custody is unavailable', 'snapshot/held_part')
            if state['held_part'] is not None and state['held_part'] != part:
                return answer('INFEASIBLE', 'KMR already holds a conflicting part')
        if primitive == 'compute_pick_targets':
            try:
                location = recorded.get('resource_location', _standing_at(state, evidence))
                # A part the event starts from elsewhere, such as a tray beside the dock, is at no calibrated spot.
                if not isinstance(location, str) or (task.get('expected_start_state') or {}).get('part_location', location) != location:
                    location = None
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
        if params.get('pose_name') not in KMR_NAMED_POSES:
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

        # The recorded chain stands where the base was captured; an accepted move_base has left it since.
        base = state.get('base_pose')
        if _pose(base) and 'root_xyz' in config and (list(base[:3]) != list(config['root_xyz'])
                                                     or list(base[3:]) != list(config.get('root_quaternion', []))):
            config = with_root(config, pose_matrix(base))
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
                    path, failure = _cartesian_path(scene, config, names, limits, state, target, params.get('waypoints'))
                    if failure:
                        return answer('INFEASIBLE', failure, joint_positions=deepcopy(candidate))
                    return answer('FEASIBLE',
                        ('Calculated' if calculated else 'Recorded') + ' KMR joint witness reproduces the requested TCP by owner FK'
                        + ('; the Cartesian path resolves as the KMR plans it' if path else '; motion interval remains separate'),
                        joint_positions=deepcopy(candidate), calculated_pose=pose,
                        configuration_fingerprint=recovery_validation_fingerprint(config),
                        **({'path': path} if path else {}),
                        **({'endpoint_calculation': calculated} if calculated else {}))
        return missing('Bounded KMR endpoint search found no independently verified joint witness; reachability is unresolved',
                       'primitive_capability_evidence/' + primitive)
    return missing('KMR owner calculation requires additional recorded parameters', 'state', 'primitive_capability_evidence/' + primitive)


def _cartesian_path(scene: dict, config: dict, names: list, limits: dict, state: dict, target: list,
                    waypoints: list | None) -> tuple[dict | None, str | None]:
    """Resolve a Cartesian move from the recorded tool pose and joints as the KMR's own planner does.

    The waypoints, the joint-step limit and the timing are the KMR worker's (``kmr_gazebo.plan_motion``); the IK is
    the bounded joint search the reach checks use. A path that cannot be followed is refused with the fact; one that
    can is reported with its samples and simulated time. Nothing here bounds the time or the posture.

    Returns:
        The path summary and None, or None and the reason the path cannot be followed. Without a recorded base and
        joints there is no start to resolve from, and both are None.
    """
    from cais_spade_llm.recovery_framework import ROOT
    from cais_spade_llm.recovery_framework.kmr_motion import joint_limits
    from cais_spade_llm.resources.kinematic_reach import bounded_joint_search
    from cais_spade_llm.resources.robot.cartesian_waypoints import fixed_orientation_waypoints, resolve_waypoints

    from cais_spade_llm.resources.continuous_motion import interval, kinematic_transforms, representative_pose

    joints, base = state.get('joint_positions'), state.get('base_pose')
    if not _pose(base) or not _vector(joints, len(names)):
        return None, None
    # As the KMR's planner does, the path starts where its joints put the tool, not at a remembered tool pose.
    start = representative_pose(kinematic_transforms(config, {name: interval(value) for name, value in
                                                              zip(names, joints, strict=True)})[config['reference_link']])
    settings = scene['KMR']['task_execution']
    steps = settings['cartesian_waypoints']
    if waypoints:
        targets = [*deepcopy(waypoints), list(target)]
    else:
        middle = steps.get('transfer_orientation_waypoint_xyzw_in_base_frame')
        targets = fixed_orientation_waypoints(
            start, target, [], rotation_point=transport_pose(scene, base, start)[:3],
            angular_step=steps.get('angular_step_rad', .05),
            rotation_midpoint=compose(base, [0., 0., 0., *middle])[3:] if middle else None) or [list(target)]

    def solve(pose: list, seed: list) -> list:
        found = bounded_joint_search(config, names, [float(value) for value in pose], [list(seed)], limits)
        if found is None:
            raise ValueError(f'no joint configuration reaches {[round(value, 3) for value in pose[:3]]}')
        return found['joint_positions']

    try:
        timing = joint_limits(ROOT / 'ros2/cais_lab_robotics/urdf/KMR_recovery.urdf.xacro', names,
                              settings['joint_acceleration_limits'])
        rows = resolve_waypoints(start_pose=start, start_joints=joints, waypoints=targets, names=names, limits=timing,
                                 solve_ik=solve, linear_step=steps['linear_step_m'],
                                 angular_step=steps.get('angular_step_rad', .05),
                                 maximum_joint_step=steps['max_joint_step_rad'])
    except ValueError as exc:
        return None, f'The KMR cannot follow the Cartesian path to {[round(value, 3) for value in target[:3]]}: {exc}'
    return {'waypoints': len(targets), 'ik_samples': len(rows),
            'simulated_duration_s': round(rows[-1]['time_from_start'], 2),
            'final_joint_positions': [round(value, 4) for value in rows[-1]['positions']]}, None


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
