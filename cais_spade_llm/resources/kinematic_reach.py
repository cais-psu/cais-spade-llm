from __future__ import annotations

"""Whether a resource can move the thing it moves to a pose, from its own kinematic chain.

Every moving resource declares its motion as a chain of joints in the configured kinematics format
(``continuous_motion`` configurations, recorded per resource in each capture): joints with parent,
child, origin and axis, a root pose and the moving joint names. The check is the same for every chain:
a bounded search for joint positions within the joint limits that put the chain's reference link at
the target. It says whether the pose is reachable, not how to get there; the motion planner plans the
path at execution.
"""

import math
import re
from collections import deque
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

_POSITION_TOLERANCE_M = 1e-4
_ORIENTATION_TOLERANCE_RAD = 1e-3
# Orientation error counts like the position error of a point this far from the axis, in metres.
_ORIENTATION_WEIGHT_M = 0.2
_STARTS = 8
# The fixed base-pose grid: spacing, headings per position, and the floor cell of the connection check.
_GRID_STEP_M = 0.25
_GRID_HEADINGS = 8
_FLOOR_CELL_M = 0.05


def pose_matrix(pose: Sequence[float]) -> np.ndarray:
    """Return a 4x4 transform for a pose ``[x, y, z, qx, qy, qz, qw]``."""
    frame = np.eye(4)
    frame[:3, :3] = Rotation.from_quat(pose[3:7]).as_matrix()
    frame[:3, 3] = pose[:3]
    return frame


def urdf_joint_limits(robot_description: str) -> dict[str, tuple[float, float]]:
    """Read every revolute and prismatic joint's position limits from a URDF.

    Args:
        robot_description: The URDF text, such as a capture's recorded ``robot_description``.

    Returns:
        Lower and upper position limit per joint name.
    """
    joints = re.finditer(r'<joint\s+name="([^"]+)"\s+type="(?:revolute|prismatic)"\s*>(.*?)</joint>',
                         robot_description, re.S)
    limits = {}
    for joint in joints:
        limit = re.search(r'<limit\b[^>]*>', joint.group(2))
        lower = re.search(r'lower="([^"]+)"', limit.group(0)) if limit else None
        upper = re.search(r'upper="([^"]+)"', limit.group(0)) if limit else None
        if lower and upper:
            limits[joint.group(1)] = (float(lower.group(1)), float(upper.group(1)))
    return limits


def chain_frames(configuration: Mapping[str, Any], positions: Mapping[str, float],
                 root: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Return every link frame of a configured chain in the world for given joint positions.

    Args:
        configuration: The resource's configured kinematics.
        positions: Positions of the moving joints; other joints keep their fixed positions.
        root: A root pose replacing the configured one, such as a mobile base's candidate pose.

    Returns:
        A 4x4 world transform per link name.
    """
    if root is None:
        root = np.eye(4)
        root[:3, :3] = Rotation.from_quat(configuration["root_quaternion"]).as_matrix()
        root[:3, 3] = configuration["root_xyz"]
    fixed = configuration.get("fixed_joint_positions", {})
    frames = {configuration["root"]: root}
    for row in configuration["joints"]:
        frame = np.eye(4)
        frame[:3, :3] = Rotation.from_euler("xyz", row["rpy"]).as_matrix()
        frame[:3, 3] = row["xyz"]
        frame = frames[row["parent"]] @ frame
        position = positions.get(row["name"], fixed.get(row["name"], 0.0))
        motion = np.eye(4)
        if row["type"] in {"revolute", "continuous"}:
            motion[:3, :3] = Rotation.from_rotvec(np.asarray(row["axis"], dtype=float) * position).as_matrix()
        elif row["type"] == "prismatic":
            motion[:3, 3] = np.asarray(row["axis"], dtype=float) * position
        frames[row["child"]] = frame @ motion
    return frames


def _chain_bound(configuration: Mapping[str, Any], limits: Mapping[str, tuple[float, float]],
                 root: np.ndarray | None) -> tuple[np.ndarray, float]:
    # The reference link never lies farther from the first moving joint than the summed link offsets
    # plus the travel of every sliding joint.
    rows = {row["child"]: row for row in configuration["joints"]}
    path, link = [], configuration["reference_link"]
    while link in rows:
        path.append(rows[link])
        link = rows[link]["parent"]
    path.reverse()
    moving = set(configuration["joint_names"])
    first = next(index for index, row in enumerate(path) if row["name"] in moving)
    anchor = chain_frames(configuration, {}, root)[path[first]["child"]][:3, 3]
    travel = sum(max(abs(value) for value in limits[row["name"]]) for row in path[first:]
                 if row["type"] == "prismatic" and row["name"] in moving)
    return anchor, float(travel + sum(np.linalg.norm(row["xyz"]) for row in path[first + 1:]))


def reach(configuration: Mapping[str, Any], limits: Mapping[str, tuple[float, float]],
          target_pose: Sequence[float], *, root: np.ndarray | None = None,
          starts: int = _STARTS) -> list[float] | None:
    """Return moving-joint positions within the limits that put the reference link at a world pose.

    Args:
        configuration: The resource's configured kinematics, with its ``reference_link``.
        limits: Position limits per moving joint, for example from ``urdf_joint_limits``.
        target_pose: The reference link's target pose ``[x, y, z, qx, qy, qz, qw]`` in the world.
        root: A root pose replacing the configured one, such as a mobile base's candidate pose.
        starts: How many of the fixed starting points to try.

    Returns:
        Joint positions in ``joint_names`` order, or None when no position within the limits reaches
        the pose. Missing limits make the answer None rather than unbounded.
    """
    names = configuration["joint_names"]
    if any(name not in limits for name in names):
        return None
    target = pose_matrix(target_pose)
    anchor, length = _chain_bound(configuration, limits, root)
    if np.linalg.norm(target[:3, 3] - anchor) > length:
        return None
    low = np.array([limits[name][0] for name in names], dtype=float)
    high = np.array([limits[name][1] for name in names], dtype=float)
    reference = configuration["reference_link"]
    goal = Rotation.from_matrix(target[:3, :3])

    def error(positions: np.ndarray) -> tuple[np.ndarray, Rotation]:
        frame = chain_frames(configuration, dict(zip(names, positions, strict=True)), root)[reference]
        return frame[:3, 3] - target[:3, 3], goal.inv() * Rotation.from_matrix(frame[:3, :3])

    def residual(positions: np.ndarray) -> np.ndarray:
        offset, turn = error(positions)
        return np.concatenate([offset, _ORIENTATION_WEIGHT_M * turn.as_rotvec()])

    # Fixed starting points, evenly spread through every joint's range, keep the answer reproducible.
    for index in range(starts):
        fraction = (0.5 + 0.37 * index) % 1.0
        start = np.clip(low + (high - low) * fraction, low + 1e-9, high - 1e-9)
        solution = least_squares(residual, start, bounds=(low, high), xtol=1e-12, ftol=1e-12, gtol=1e-12,
                                 max_nfev=400)
        offset, turn = error(solution.x)
        if np.linalg.norm(offset) < _POSITION_TOLERANCE_M and turn.magnitude() < _ORIENTATION_TOLERANCE_RAD:
            return [float(value) for value in solution.x]
    return None


def _corners(x: float, y: float, yaw: float, size: Sequence[float]) -> list[tuple[float, float]]:
    half_length, half_width, c, s = size[0] / 2, size[1] / 2, math.cos(yaw), math.sin(yaw)
    return [(x + c * dx - s * dy, y + s * dx + c * dy)
            for dx, dy in ((half_length, half_width), (-half_length, half_width),
                           (-half_length, -half_width), (half_length, -half_width))]


def _separated(first: list[tuple[float, float]], second: list[tuple[float, float]]) -> bool:
    # Two convex outlines are apart when some edge normal of either separates their projections.
    for outline in (first, second):
        for index, (x0, y0) in enumerate(outline):
            x1, y1 = outline[(index + 1) % len(outline)]
            normal = (y0 - y1, x1 - x0)
            a = [normal[0] * x + normal[1] * y for x, y in first]
            b = [normal[0] * x + normal[1] * y for x, y in second]
            if max(a) < min(b) or max(b) < min(a):
                return True
    return False


def _box(bounds: Sequence[float]) -> list[tuple[float, float]]:
    x0, x1, y0, y1 = bounds
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def free_base_poses(floor: Mapping[str, Any], footprint: Sequence[float], start_xy: Sequence[float],
                    around_xy: Sequence[float], *, docks: Sequence[Sequence[float]] = (),
                    radius_m: float = 1.5) -> list[list[float]]:
    """Return candidate planar base poses near a point, in a fixed order, where the base fits and can get to.

    Declared docks come first, then a grid around the point, nearest first, at evenly spaced headings. A
    pose is kept when the footprint, a rectangle centred on the base, lies on the floor clear of every
    obstacle, and when its position connects to the start through floor wide enough for the footprint's
    narrow side. That connection is a necessary condition only; the base's own planner plans the route.

    Args:
        floor: ``floor_extent_xy`` [x0, x1, y0, y1] and ``floor_obstacles_xy`` (name to [x0, x1, y0, y1]).
        footprint: The base's length and width in metres.
        start_xy: The base's current position.
        around_xy: The point to work at, such as a place target.
        docks: Declared base poses ``[x, y, yaw]`` to try first.
        radius_m: How far from the point to place candidates.

    Returns:
        Base poses ``[x, y, yaw]``.
    """
    x0, x1, y0, y1 = floor["floor_extent_xy"]
    obstacles = [_box(bounds) for bounds in floor["floor_obstacles_xy"].values()]
    clearance = min(footprint[:2]) / 2

    def open_cell(x: float, y: float) -> bool:
        if not (x0 + clearance <= x <= x1 - clearance and y0 + clearance <= y <= y1 - clearance):
            return False
        return all(math.hypot(x - min(max(x, bx0), bx1), y - min(max(y, by0), by1)) > clearance
                   for bx0, bx1, by0, by1 in floor["floor_obstacles_xy"].values())

    def cell(x: float, y: float) -> tuple[int, int]:
        return round((x - x0) / _FLOOR_CELL_M), round((y - y0) / _FLOOR_CELL_M)

    columns, rows = cell(x1, y1)
    connected, frontier = set(), deque()
    if open_cell(*start_xy[:2]):
        connected.add(cell(*start_xy[:2]))
        frontier.append(cell(*start_xy[:2]))
    while frontier:
        i, j = frontier.popleft()
        for neighbour in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)):
            if (neighbour not in connected and 0 <= neighbour[0] <= columns and 0 <= neighbour[1] <= rows
                    and open_cell(x0 + neighbour[0] * _FLOOR_CELL_M, y0 + neighbour[1] * _FLOOR_CELL_M)):
                connected.add(neighbour)
                frontier.append(neighbour)

    def fits(x: float, y: float, yaw: float) -> bool:
        corners = _corners(x, y, yaw, footprint)
        return (all(x0 <= cx <= x1 and y0 <= cy <= y1 for cx, cy in corners)
                and all(_separated(corners, obstacle) for obstacle in obstacles) and cell(x, y) in connected)

    steps = int(radius_m / _GRID_STEP_M)
    grid = sorted(((around_xy[0] + i * _GRID_STEP_M, around_xy[1] + j * _GRID_STEP_M)
                   for i in range(-steps, steps + 1) for j in range(-steps, steps + 1)
                   if math.hypot(i, j) * _GRID_STEP_M <= radius_m),
                  key=lambda point: (math.hypot(point[0] - around_xy[0], point[1] - around_xy[1]), point))
    candidates = [list(dock[:3]) for dock in docks] + [
        [x, y, 2 * math.pi * index / _GRID_HEADINGS - math.pi] for x, y in grid for index in range(_GRID_HEADINGS)]
    return [pose for pose in candidates if fits(*pose)]


def reach_from_base(configuration: Mapping[str, Any], limits: Mapping[str, tuple[float, float]],
                    target_pose: Sequence[float], base_poses: Sequence[Sequence[float]], *,
                    base_height_m: float = 0.0, tries: int = 12,
                    starts: int = 2) -> tuple[list[float], list[float]] | None:
    """Return a candidate base pose from which a mounted chain reaches a pose, with its joints.

    The search is bounded: candidates are tried in order of how close they bring the chain's first moving
    joint to the target (ties keep the given order), and only the first ``tries`` within the chain's reach.

    Args:
        configuration: The mounted chain, whose root is the base link.
        limits: Position limits per moving joint.
        target_pose: The reference link's target pose in the world.
        base_poses: Candidate planar base poses ``[x, y, yaw]``, for example from ``free_base_poses``.
        base_height_m: The base link's height, which driving does not change.
        tries: How many candidates to try.
        starts: How many fixed starting points to try at each base pose.

    Returns:
        The base pose and the chain's joint positions, or None when no candidate reaches the pose.
    """
    roots, ranked = {}, []
    for index, (x, y, yaw) in enumerate(base_poses):
        roots[index] = pose_matrix([x, y, base_height_m, *Rotation.from_euler("z", yaw).as_quat()])
        anchor, length = _chain_bound(configuration, limits, roots[index])
        distance = float(np.linalg.norm(np.asarray(target_pose[:3]) - anchor))
        if distance <= length:
            ranked.append((distance, index))
    for _, index in sorted(ranked)[:tries]:
        joints = reach(configuration, limits, target_pose, root=roots[index], starts=starts)
        if joints is not None:
            return list(base_poses[index][:3]), joints
    return None
