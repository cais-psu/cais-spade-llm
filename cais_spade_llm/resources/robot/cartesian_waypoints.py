"""Convert explicit Cartesian waypoints to continuous robot joint commands."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence


def sample_segment(start: Sequence[float], end: Sequence[float], *,
                   linear_step: float, angular_step: float, minimum_samples: int = 1) -> list[list[float]]:
    """Interpolate XYZ and the shortest quaternion arc without choosing a route."""
    if (len(start) != 7 or len(end) != 7 or not all(math.isfinite(v) for v in [*start, *end])
            or min(linear_step, angular_step) <= 0):
        raise ValueError('Cartesian waypoints require finite XYZ/xyzw poses and positive spacing')
    quaternions = []
    for pose in (start, end):
        norm = math.sqrt(sum(v * v for v in pose[3:]))
        if norm < 1e-9:
            raise ValueError('Cartesian waypoint orientation is unavailable')
        quaternions.append([v / norm for v in pose[3:]])
    left, right = quaternions
    dot = sum(a * b for a, b in zip(left, right))
    if dot < 0:
        right, dot = [-v for v in right], -dot
    angle = math.acos(min(1., dot))
    count = max(minimum_samples, 1, math.ceil(math.dist(start[:3], end[:3]) / linear_step),
                math.ceil(2 * angle / angular_step))
    result = []
    for index in range(count + 1):
        fraction = index / count
        if angle < 1e-8:
            q = left
        else:
            q = [(math.sin((1 - fraction) * angle) * a + math.sin(fraction * angle) * b)
                 / math.sin(angle) for a, b in zip(left, right)]
        result.append([*(a + (b - a) * fraction for a, b in zip(start[:3], end[:3])), *q])
    return result


def continuous_joints(values: Sequence[float], previous: Sequence[float],
                      names: Sequence[str], limits: dict, maximum_step: float) -> list[float]:
    """Keep the nearest valid revolute representation and reject an IK branch jump."""
    if len(values) != len(names) or len(previous) != len(names):
        raise ValueError('Waypoint IK did not return every owned joint')
    result = []
    for name, value, before in zip(names, values, previous, strict=True):
        if not math.isfinite(value):
            raise ValueError(f'Waypoint IK is not finite: {name}')
        turns = round((before - value) / (2 * math.pi))
        choices = [value + 2 * math.pi * k for k in (turns - 1, turns, turns + 1)]
        choices = [q for q in choices if limits[name]['lower'] <= q <= limits[name]['upper']]
        if not choices:
            raise ValueError(f'Waypoint IK exceeds joint limits: {name}')
        selected = min(choices, key=lambda q: abs(q - before))
        if abs(selected - before) > maximum_step:
            raise ValueError(f'Waypoint IK changes joint branch: {name}')
        result.append(selected)
    return result


def _segment_timing(positions: list[list[float]], names: Sequence[str], limits: dict) -> list[dict]:
    """Time one supplied segment, stopping at its endpoints within configured limits."""
    count = len(positions) - 1
    derivatives, second = [], []
    for index in range(count + 1):
        lo, hi = max(0, index - 1), min(count, index + 1)
        derivatives.append([(b - a) * count / (hi - lo)
                            for a, b in zip(positions[lo], positions[hi])])
        second.append([(positions[min(count, index + 1)][j] - 2 * positions[index][j]
                        + positions[max(0, index - 1)][j]) * count**2
                       if 0 < index < count else 0. for j in range(len(names))])
    rows, duration = [], .05
    for index, joints in enumerate(positions):
        fraction = index / count
        lo, hi = 0., 1.
        for _ in range(40):
            u = (lo + hi) / 2
            if 10 * u**3 - 15 * u**4 + 6 * u**5 < fraction:
                lo = u
            else:
                hi = u
        u = 0. if index == 0 else 1. if index == count else (lo + hi) / 2
        ds = 30 * u**2 * (1 - u)**2
        dds = 60 * u * (1 - 3 * u + 2 * u**2)
        velocity = [dq * ds for dq in derivatives[index]]
        acceleration = [ddq * ds**2 + dq * dds for ddq, dq in zip(second[index], derivatives[index])]
        for name, v, a in zip(names, velocity, acceleration):
            duration = max(duration, abs(v) / limits[name]['velocity'],
                           math.sqrt(abs(a) / limits[name]['acceleration']))
        rows.append({'positions': list(joints), 'velocities': velocity,
                     'accelerations': acceleration, 'time_from_start': u})
    # Include endpoint-only segments and every connecting displacement in the bound.
    for left, right in zip(rows, rows[1:]):
        dt = right['time_from_start'] - left['time_from_start']
        for name, a, b in zip(names, left['positions'], right['positions']):
            duration = max(duration, 1.875 * abs(b - a) / limits[name]['velocity'],
                           math.sqrt(5.78 * abs(b - a) / limits[name]['acceleration']),
                           abs(b - a) / dt / limits[name]['velocity'])
    for row in rows:
        row['time_from_start'] *= duration
        row['velocities'] = [v / duration for v in row['velocities']]
        row['accelerations'] = [a / duration**2 for a in row['accelerations']]
    return rows


def resolve_waypoints(*, start_pose: Sequence[float], start_joints: Sequence[float],
                      waypoints: Sequence[Sequence[float]], names: Sequence[str], limits: dict,
                      solve_ik: Callable[[list[float], list[float]], Sequence[float]],
                      linear_step: float = .01, angular_step: float = .05,
                      maximum_joint_step: float = .35) -> list[dict]:
    """Resolve only the supplied route, seeding every IK sample from its predecessor.

    Collision and observation checks belong to the resource controller. This
    conversion never searches for an alternative route or executes partial work.
    """
    if not waypoints:
        raise ValueError('Cartesian execution requires at least one waypoint')
    previous_pose, previous = list(start_pose), list(start_joints)
    result, elapsed = [], 0.
    for target in waypoints:
        samples = sample_segment(previous_pose, target, linear_step=linear_step, angular_step=angular_step)
        joints = [previous]
        for sample in samples[1:]:
            previous = continuous_joints(solve_ik(sample, previous), previous, names, limits, maximum_joint_step)
            joints.append(previous)
        timed = _segment_timing(joints, names, limits)
        for row in timed[(1 if result else 0):]:
            row['time_from_start'] += elapsed
            result.append(row)
        elapsed = result[-1]['time_from_start']
        previous_pose = list(target)
    return result


def robot_trajectory(names: Sequence[str], rows: list[dict]):
    """Build the owned controller's ROS trajectory after complete waypoint conversion."""
    from builtin_interfaces.msg import Duration
    from moveit_msgs.msg import RobotTrajectory
    from trajectory_msgs.msg import JointTrajectoryPoint

    result = RobotTrajectory()
    result.joint_trajectory.joint_names = list(names)
    for row in rows:
        sec, nanosec = divmod(round(row['time_from_start'] * 1e9), 1_000_000_000)
        result.joint_trajectory.points.append(JointTrajectoryPoint(
            positions=row['positions'], velocities=row['velocities'], accelerations=row['accelerations'],
            time_from_start=Duration(sec=sec, nanosec=nanosec)))
    return result
