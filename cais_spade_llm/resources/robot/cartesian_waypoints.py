"""Convert explicit Cartesian waypoints to continuous robot joint commands."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator, Sequence


def fixed_orientation_waypoints(
    start: Sequence[float], target: Sequence[float], clearance: Sequence[Sequence[float]],
    *, rotation_point: Sequence[float], angular_step: float,
    rotation_midpoint: Sequence[float] | None = None,
) -> list[list[float]]:
    """Translate at fixed orientation and turn only at a saved clearance point.

    Args:
        start: Fresh XYZ/xyzw pose in the planning frame.
        target: Computed XYZ/xyzw endpoint in the same frame.
        clearance: Ordered saved XYZ clearance points.
        rotation_point: Saved XYZ location for a required orientation change.
        angular_step: Maximum orientation interpolation interval in radians.
        rotation_midpoint: Saved quaternion disambiguating a required half turn.

    Returns:
        Cartesian poses, including the endpoint, with duplicate poses removed.
    """
    sample_segment(start, target, linear_step=1e6, angular_step=angular_step)
    route = [list(point) for point in clearance]
    if any(len(point) != 3 or not all(math.isfinite(v) for v in point)
           for point in [*route, rotation_point]):
        raise ValueError('Cartesian clearance requires finite saved XYZ points')
    dot = sum(a * b for a, b in zip(start[3:], target[3:], strict=True))
    turning = abs(dot) < math.cos(.001 / 2)
    if turning and list(rotation_point) not in route:
        if math.dist(target[:3], rotation_point) < 1e-6:
            route.append(list(rotation_point))
        else:
            route.insert(0, list(rotation_point))
    result: list[list[float]] = []
    previous = list(start)

    def append(pose: Sequence[float]) -> None:
        nonlocal previous
        if (math.dist(previous[:3], pose[:3]) < 1e-9
                and abs(sum(a * b for a, b in zip(previous[3:], pose[3:], strict=True))) > 1. - 1e-12):
            return
        result.append(list(pose))
        previous = list(pose)

    orientation = list(start[3:])
    for xyz in route:
        append([*xyz, *orientation])
        if turning and xyz == list(rotation_point):
            orientations = ([rotation_midpoint] if rotation_midpoint is not None and abs(dot) < .01 else [])
            for quaternion in [*orientations, target[3:]]:
                for pose in sample_segment(previous, [*xyz, *quaternion],
                                           linear_step=1e6, angular_step=angular_step)[1:]:
                    append(pose)
            orientation = list(target[3:])
            turning = False
    append([*target[:3], *orientation])
    return result


def _validate_limits(names: Sequence[str], limits: dict) -> None:
    if not names or len(set(names)) != len(names):
        raise ValueError('Cartesian execution requires distinct owned joint names')
    for name in names:
        row = limits.get(name, {})
        if (not all(key in row and math.isfinite(row[key])
                    for key in ('lower', 'upper', 'velocity', 'acceleration'))
                or row['lower'] >= row['upper']
                or min(row['velocity'], row['acceleration']) <= 0):
            raise ValueError(f'Invalid Cartesian joint limits: {name}')


def _coefficients(left: dict, right: dict, index: int) -> list[float]:
    """Match the controller's quintic interpolation on a unit time interval."""
    dt = right['time_from_start'] - left['time_from_start']
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError('Cartesian trajectory time must increase')
    c0 = left['positions'][index]
    c1 = dt * left['velocities'][index]
    c2 = .5 * dt**2 * left['accelerations'][index]
    position = right['positions'][index] - c0 - c1 - c2
    velocity = dt * right['velocities'][index] - c1 - 2 * c2
    acceleration = dt**2 * right['accelerations'][index] - 2 * c2
    return [c0, c1, c2, 10 * position - 4 * velocity + .5 * acceleration,
            -15 * position + 7 * velocity - acceleration,
            6 * position - 3 * velocity + .5 * acceleration]


def _derivative(coefficients: Sequence[float]) -> list[float]:
    return [index * value for index, value in enumerate(coefficients)][1:]


def _value(coefficients: Sequence[float], fraction: float) -> float:
    result = 0.
    for coefficient in reversed(coefficients):
        result = result * fraction + coefficient
    return result


def _extrema(coefficients: Sequence[float]) -> list[float]:
    import numpy as np

    roots = np.polynomial.polynomial.polyroots(_derivative(coefficients))
    fractions = [0., 1., *(float(root.real) for root in roots
                          if abs(root.imag) < 1e-8 and 0 < root.real < 1)]
    return [_value(coefficients, fraction) for fraction in fractions]


def trajectory_samples(trajectory, *, maximum_joint_step: float = .05) -> Iterator[list[float]]:
    """Sample controller interpolation with a bound on joint travel per interval.

    This uses position, velocity and acceleration at both ends, including
    excursions between equal endpoint positions. Collision checks remain owned
    by the resource controller and run before any trajectory is dispatched.
    """
    if not math.isfinite(maximum_joint_step) or maximum_joint_step <= 0:
        raise ValueError('Cartesian collision sampling requires positive finite spacing')
    previous = None
    for point in trajectory.points:
        current = {'positions': list(point.positions), 'velocities': list(point.velocities),
                   'accelerations': list(point.accelerations),
                   'time_from_start': point.time_from_start.sec + point.time_from_start.nanosec / 1e9}
        if previous is None:
            yield current['positions']
        else:
            coefficients = [_coefficients(previous, current, index)
                            for index in range(len(trajectory.joint_names))]
            speed = max(abs(v) for row in coefficients for v in _extrema(_derivative(row)))
            count = max(1, math.ceil(speed / maximum_joint_step))
            for index in range(1, count + 1):
                yield [_value(row, index / count) for row in coefficients]
        previous = current


def sample_segment(start: Sequence[float], end: Sequence[float], *,
                   linear_step: float, angular_step: float, minimum_samples: int = 1) -> list[list[float]]:
    """Interpolate XYZ and the shortest quaternion arc without choosing a route."""
    if (len(start) != 7 or len(end) != 7 or not all(math.isfinite(v) for v in [*start, *end])
            or not all(math.isfinite(v) and v > 0 for v in (linear_step, angular_step))
            or type(minimum_samples) is not int or minimum_samples < 1):
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
    _validate_limits(names, limits)
    if (not math.isfinite(maximum_step) or maximum_step <= 0
            or len(values) != len(names) or len(previous) != len(names)):
        raise ValueError('Waypoint IK did not return every owned joint')
    result = []
    for name, value, before in zip(names, values, previous, strict=True):
        if not math.isfinite(value) or not math.isfinite(before):
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
        for index, name in enumerate(names):
            coefficients = _coefficients(left, right, index)
            bounds = _extrema(coefficients)
            if min(bounds) < limits[name]['lower'] - 1e-9 or max(bounds) > limits[name]['upper'] + 1e-9:
                raise ValueError(f'Cartesian controller interpolation exceeds joint limits: {name}; '
                                 f'positions={left["positions"][index], right["positions"][index]}; '
                                 f'extrema={min(bounds), max(bounds)}')
            velocity = _derivative(coefficients)
            acceleration = _derivative(velocity)
            duration = max(duration,
                           max(abs(v) for v in _extrema(velocity)) / dt / limits[name]['velocity'],
                           math.sqrt(max(abs(a) for a in _extrema(acceleration))
                                     / dt**2 / limits[name]['acceleration']))
    duration *= 1.000001
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
    _validate_limits(names, limits)
    if (len(start_joints) != len(names)
            or any(not math.isfinite(value) or not limits[name]['lower'] <= value <= limits[name]['upper']
                   for name, value in zip(names, start_joints))
            or not math.isfinite(maximum_joint_step) or maximum_joint_step <= 0):
        raise ValueError('Cartesian execution requires finite bounded start joints and positive joint spacing')
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
