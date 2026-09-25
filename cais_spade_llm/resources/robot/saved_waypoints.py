"""Load prepared Gazebo joint waypoints without solving or timing a new path."""

from __future__ import annotations

import hashlib
import json
import math
from functools import lru_cache
from pathlib import Path

from cais_spade_llm.recovery_framework import ROOT
from cais_spade_llm.resources.robot.cartesian_waypoints import (
    _coefficients, _derivative, _extrema, _validate_limits, robot_trajectory,
)

SAVED_WAYPOINTS_FILE = 'cais_spade_llm/initialization/recovery_framework_waypoints.json'
SOURCE_FILES = (
    'cais_spade_llm/initialization/recovery_framework_gazebo.json',
    'cais_spade_llm/initialization/resources/robot_ur5e.json',
    'cais_spade_llm/specification/products/geometry/assembly_board-v1-recovery-framework.json',
    'ros2/cais_lab_robotics/urdf/KMR_recovery.urdf.xacro',
    'ros2/cais_lab_robotics/launch/dual_moveit_gazebo.launch.py',
)


def source_fingerprints() -> dict[str, str]:
    """Identify the saved layout, geometry and robot configurations."""
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in SOURCE_FILES}


def poses_match(left: list[float], right: list[float]) -> bool:
    """Accept only finite poses within the saved-route observation tolerances."""
    if (len(left) != 7 or len(right) != 7
            or not all(math.isfinite(v) for v in [*left, *right])):
        return False
    norms = [math.sqrt(sum(v * v for v in pose[3:])) for pose in (left, right)]
    return bool(min(norms) > 1e-9 and math.dist(left[:3], right[:3]) <= .005
                and abs(sum(a * b for a, b in zip(left[3:], right[3:]))) / math.prod(norms)
                >= math.cos(.02 / 2))


def validate_points(points: list[dict], names: list[str], limits: dict) -> None:
    """Validate saved timing, including controller interpolation, without retiming."""
    _validate_limits(names, limits)
    if len(points) < 2:
        raise ValueError('Saved waypoints require a complete timed motion')
    previous = None
    for row in points:
        if (any(len(row.get(field, [])) != len(names)
                or not all(math.isfinite(v) for v in row[field])
                for field in ('positions', 'velocities', 'accelerations'))
                or not math.isfinite(row.get('time_from_start', math.nan))):
            raise ValueError('Saved waypoints have invalid joint values or timing')
        if previous is None:
            if row['time_from_start'] != 0:
                raise ValueError('Saved waypoints must start at zero')
        else:
            dt = row['time_from_start'] - previous['time_from_start']
            for index, name in enumerate(names):
                c = _coefficients(previous, row, index)
                bounds = _extrema(c)
                velocity = _derivative(c)
                acceleration = _derivative(velocity)
                if (min(bounds) < limits[name]['lower'] - 1e-7
                        or max(bounds) > limits[name]['upper'] + 1e-7
                        or max(abs(v) for v in _extrema(velocity)) / dt > limits[name]['velocity'] * 1.00001
                        or max(abs(a) for a in _extrema(acceleration)) / dt**2 > limits[name]['acceleration'] * 1.00001):
                    raise ValueError(f'Saved waypoints exceed configured joint limits: {name}')
        previous = row
    if any(abs(v) > 1e-8 for row in (points[0], points[-1])
           for field in ('velocities', 'accelerations') for v in row[field]):
        raise ValueError('Saved waypoints must stop at both ends')


@lru_cache(maxsize=8)
def _read_saved(path: str, stamp: int, size: int, sources: tuple) -> dict:
    del stamp, size
    payload = json.loads(Path(path).read_text(encoding='utf-8'))
    if payload.get('version') != 1 or payload.get('execution_mode') != 'simulation':
        raise ValueError('Saved waypoints are not a Gazebo simulation recording')
    if payload.get('failures'):
        raise ValueError('Saved waypoint preparation is incomplete')
    if payload.get('source_fingerprints') != dict(sources):
        raise ValueError('Saved waypoints do not match the configured Gazebo layout; prepare them before production')
    for resource in payload['resources'].values():
        for route in resource['routes']:
            validate_points(route['points'], resource['joint_names'], resource['limits'])
            if not all(poses_match(route[field], route[field]) for field in ('start_pose', 'target_pose')):
                raise ValueError('Saved waypoint poses are invalid')
    payload['sha256'] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return payload


def require_saved_waypoints(scene: dict) -> dict:
    """Check all configured robot recordings before production can start."""
    references = {robot['resource_id']: robot['cartesian_motion'] for robot in scene['robots']}
    references['KMR'] = scene['KMR']['task_execution']['cartesian_waypoints']
    counts = {}
    for resource_id, settings in references.items():
        reference = settings.get('saved_waypoints_file')
        if not reference:
            raise ValueError(f'{resource_id}: no saved waypoints configured; runtime IK is disabled')
        path = ROOT / reference
        try:
            stat = path.stat()
            payload = _read_saved(str(path), stat.st_mtime_ns, stat.st_size,
                                  tuple(source_fingerprints().items()))
        except OSError as exc:
            raise ValueError(f'{resource_id}: saved waypoints unavailable; prepare them before production') from exc
        resource = payload['resources'].get(resource_id, {})
        if not resource.get('routes'):
            raise ValueError(f'{resource_id}: saved waypoint preparation is incomplete')
        counts[resource_id] = len(resource['routes'])
    return counts


def saved_motion(*, settings: dict, resource_id: str, names: list[str], limits: dict,
                 start_pose: list[float], start_joints: list[float], target_pose: list[float],
                 frame_id: str = 'world', execution_mode: str = 'simulation'):
    """Select an existing motion with matching observed start; never calculate a path.

    Args:
        settings: Resource configuration naming its saved waypoint file.
        resource_id: Exact resource identifier in that file.
        names: Owned joint names in controller order.
        limits: Limits read from the current robot model.
        start_pose: Fresh controlled-link pose in the recording frame.
        start_joints: Fresh owned joint positions.
        target_pose: Requested controlled-link pose in the same frame.
        frame_id: Frame used for recorded Cartesian poses.
        execution_mode: Only simulation may consume this recording.

    Returns:
        The saved ROS command and its recording evidence.
    """
    if execution_mode != 'simulation':
        raise ValueError('Saved Gazebo waypoints cannot execute in hardware mode')
    reference = settings.get('saved_waypoints_file')
    if not reference:
        raise ValueError(f'{resource_id}: no saved waypoints configured; runtime IK is disabled')
    path = ROOT / reference
    try:
        stat = path.stat()
        payload = _read_saved(str(path), stat.st_mtime_ns, stat.st_size,
                              tuple(source_fingerprints().items()))
    except OSError as exc:
        raise ValueError(f'{resource_id}: saved waypoints unavailable; prepare them before production') from exc
    resource = payload['resources'].get(resource_id)
    if (resource is None or resource['joint_names'] != list(names)
            or resource['frame_id'] != frame_id or resource['limits'] != limits):
        raise ValueError(f'{resource_id}: saved waypoints do not match the current robot and limits')
    if len(start_joints) != len(names) or not all(math.isfinite(v) for v in start_joints):
        raise ValueError(f'{resource_id}: fresh start joints are unavailable')
    candidates = [row for row in resource['routes']
                  if poses_match(row['start_pose'], start_pose)
                  and poses_match(row['target_pose'], target_pose)
                  and max(abs(a - b) for a, b in zip(row['points'][0]['positions'], start_joints)) <= .02]
    if not candidates:
        raise ValueError(f'{resource_id}: no saved waypoints match the observed start and requested target; runtime IK is disabled')
    selected = min(candidates, key=lambda row: sum(
        abs(a - b) for a, b in zip(row['points'][0]['positions'], start_joints)))
    return robot_trajectory(names, selected['points']), {
        'saved_waypoints_file': reference, 'saved_waypoints_sha256': payload['sha256'],
        'saved_waypoints_id': selected['id'], 'runtime_ik': False,
        'runtime_time_parameterization': False,
    }
