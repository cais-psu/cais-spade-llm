from __future__ import annotations

"""Whether a resource can move the thing it moves to a pose, from its own kinematic chain.

Every moving resource declares its motion as a chain of joints in the configured kinematics format:
joints with parent, child, origin, axis and ``position_bounds``, a root pose, the moving joint names and
the reference link it moves. Captures record each arm's chain in ``physical_evidence.resource_models``.
One bounded search answers for every chain: joint positions within the bounds that put the reference
link at the target. Only a target beyond the chain's whole length is proven unreachable; a search that
finds nothing leaves reachability unresolved. Paths are planned at execution.
"""

import math
from collections import deque
from collections.abc import Mapping, Sequence
from contextvars import ContextVar
from copy import deepcopy
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

# One calculation per identical request while a caller sets it, as a validation run does.
JOINT_SEARCH_CACHE: ContextVar[dict | None] = ContextVar("_cartesian_joint_calculation_cache", default=None)
# Fixed extra starting points, as fractions of every joint's range, keep the answer reproducible.
_START_FRACTIONS = (0.5, 0.87, 0.24, 0.61, 0.98, 0.35)
# The fixed base-pose grid: spacing, headings per position, and the floor cell of the connection check.
_GRID_STEP_M = 0.25
_GRID_HEADINGS = 8
_FLOOR_CELL_M = 0.05


def _finite_vector(value: Any, length: int) -> bool:
    return (isinstance(value, list) and len(value) == length
            and all(type(item) in (int, float) and math.isfinite(item) for item in value))


def pose_matrix(pose: Sequence[float]) -> np.ndarray:
    """Return a 4x4 transform for a pose ``[x, y, z, qx, qy, qz, qw]``."""
    frame = np.eye(4)
    frame[:3, :3] = Rotation.from_quat(pose[3:7]).as_matrix()
    frame[:3, 3] = pose[:3]
    return frame


def bounded_joints(names: Sequence[str], positions: Sequence[float], limits: dict) -> bool:
    """Check exact bounded joint values without wrapping angles into another branch."""
    return len(names) == len(positions) and all(
        name in limits
        and math.isfinite(value)
        and limits[name]["lower"] <= value <= limits[name]["upper"]
        for name, value in zip(names, positions, strict=True)
    )


def bounded_joint_search(config: dict, names: list, target: list,  # noqa: C901 - moved unchanged
                         seeds: list[list], limits: dict) -> dict | None:
    """Search bounded owner kinematics; the caller independently checks its FK.

    An unsuccessful finite search is missing evidence. It does not establish
    that the requested pose is unreachable, collision-free, or executable.
    """
    from scipy.optimize import least_squares

    from cais_spade_llm.agents.shared_information.recovery_validation_protocol import (
        recovery_validation_fingerprint,
    )

    cache = JOINT_SEARCH_CACHE.get()
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
        if not _finite_vector(seed, len(names)) or not bounded_joints(names, seed, limits) or tuple(seed) in tried:
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


def chain_limits(configuration: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    """Return the moving joints' ``position_bounds`` as ``{name: {"lower": ..., "upper": ...}}``."""
    names = set(configuration["joint_names"])
    return {row["name"]: {"lower": row["position_bounds"][0], "upper": row["position_bounds"][1]}
            for row in configuration["joints"] if row["name"] in names and "position_bounds" in row}


def with_root(configuration: Mapping[str, Any], root: np.ndarray | None) -> dict:
    """Return the chain with its root moved to a pose, such as a mobile base's candidate pose."""
    if root is None:
        return dict(configuration)
    return {**configuration, "root_xyz": root[:3, 3].tolist(),
            "root_quaternion": Rotation.from_matrix(root[:3, :3]).as_quat().tolist()}


def chain_frames(configuration: Mapping[str, Any], positions: Mapping[str, float]) -> dict[str, np.ndarray]:
    """Return every link frame of a configured chain in the world for given joint positions.

    Args:
        configuration: The resource's configured kinematics.
        positions: Positions of the moving joints; other joints keep their fixed positions.

    Returns:
        A 4x4 world transform per link name.
    """
    root = np.eye(4)
    root[:3, :3] = (Rotation.from_quat(configuration["root_quaternion"]).as_matrix()
                    if "root_quaternion" in configuration
                    else Rotation.from_euler("xyz", configuration["root_rpy"]).as_matrix())
    root[:3, 3] = configuration["root_xyz"]
    frames = {configuration["root"]: root}
    for row in configuration["joints"]:
        frame = np.eye(4)
        frame[:3, :3] = Rotation.from_euler("xyz", row["rpy"]).as_matrix()
        frame[:3, 3] = row["xyz"]
        frame = frames[row["parent"]] @ frame
        position = positions.get(row["name"], row.get("frozen_position", 0.0))
        motion = np.eye(4)
        if row["type"] in {"revolute", "continuous"}:
            motion[:3, :3] = Rotation.from_rotvec(np.asarray(row["axis"], dtype=float) * position).as_matrix()
        elif row["type"] == "prismatic":
            motion[:3, 3] = np.asarray(row["axis"], dtype=float) * position
        frames[row["child"]] = frame @ motion
    return frames


def fixed_starts(names: Sequence[str], limits: Mapping[str, Mapping[str, float]]) -> list[list[float]]:
    """Return the fixed extra starting points of a joint search, spread through every joint's range."""
    return [[limits[name]["lower"] + (limits[name]["upper"] - limits[name]["lower"]) * fraction for name in names]
            for fraction in _START_FRACTIONS]


def beyond_reach(configuration: Mapping[str, Any], target_xyz: Sequence[float]) -> bool:
    """Return whether a point lies farther from the chain's first moving joint than the chain can extend.

    The reference link never lies farther than the summed link offsets plus every sliding joint's travel,
    so a point beyond that is proven unreachable.
    """
    rows = {row["child"]: row for row in configuration["joints"]}
    path, link = [], configuration["reference_link"]
    while link in rows:
        path.append(rows[link])
        link = rows[link]["parent"]
    path.reverse()
    moving = set(configuration["joint_names"])
    first = next(index for index, row in enumerate(path) if row["name"] in moving)
    anchor = chain_frames(configuration, {})[path[first]["child"]][:3, 3]
    travel = sum(max(abs(value) for value in row["position_bounds"]) for row in path[first:]
                 if row["type"] == "prismatic" and row["name"] in moving)
    length = travel + sum(np.linalg.norm(row["xyz"]) for row in path[first + 1:])
    return bool(np.linalg.norm(np.asarray(target_xyz[:3], dtype=float) - anchor) > length)


def reach(configuration: Mapping[str, Any], target_pose: Sequence[float], *, root: np.ndarray | None = None,
          seeds: Sequence[Sequence[float]] = ()) -> dict[str, Any]:
    """Check whether the chain can put its reference link at a world pose, within its joint bounds.

    Args:
        configuration: The resource's configured kinematics, with ``position_bounds`` on its moving joints.
        target_pose: The reference link's target pose ``[x, y, z, qx, qy, qz, qw]`` in the world.
        root: A root pose replacing the configured one, such as a mobile base's candidate pose.
        seeds: Recorded joint positions to start from first, such as the current or a named pose.

    Returns:
        ``feasibility_status`` FEASIBLE with ``joint_positions``; INFEASIBLE when the target is beyond the
        whole chain; NEEDS_CONTEXT when the bounds are incomplete or the bounded search finds nothing.
    """
    chain = with_root(configuration, root)
    names = list(chain["joint_names"])
    limits = chain_limits(chain)
    if set(names) - set(limits):
        return {"feasibility_status": "NEEDS_CONTEXT", "reason": "Recorded joint bounds are incomplete"}
    if beyond_reach(chain, target_pose):
        return {"feasibility_status": "INFEASIBLE", "reason": "The target lies beyond the whole chain"}
    found = bounded_joint_search(chain, names, [float(value) for value in target_pose[:7]],
                                 [list(seed) for seed in seeds] + fixed_starts(names, limits), limits)
    if found is None:
        return {"feasibility_status": "NEEDS_CONTEXT",
                "reason": "The bounded joint search found no configuration; reachability is unresolved"}
    return {"feasibility_status": "FEASIBLE", "joint_positions": found["joint_positions"], "calculation": found}


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


def reach_from_base(configuration: Mapping[str, Any], target_pose: Sequence[float],
                    base_poses: Sequence[Sequence[float]], *, base_height_m: float = 0.0,
                    tries: int = 12) -> dict[str, Any]:
    """Check whether a mounted chain reaches a pose from some candidate base pose.

    The search is bounded: candidates are tried in order of how close they bring the chain's first moving
    joint to the target (ties keep the given order), and only the first ``tries`` not beyond the chain.

    Args:
        configuration: The mounted chain, whose root is the base link.
        target_pose: The reference link's target pose in the world.
        base_poses: Candidate planar base poses ``[x, y, yaw]``, for example from ``free_base_poses``.
        base_height_m: The base link's height, which driving does not change.
        tries: How many candidates to try.

    Returns:
        ``feasibility_status`` FEASIBLE with ``base_pose`` and ``joint_positions``, else NEEDS_CONTEXT: a
        finite set of base poses never proves the target unreachable.
    """
    ranked = []
    for index, (x, y, yaw) in enumerate(base_poses):
        root = pose_matrix([x, y, base_height_m, *Rotation.from_euler("z", yaw).as_quat()])
        chain = with_root(configuration, root)
        if not beyond_reach(chain, target_pose):
            first = next(row["child"] for row in chain["joints"] if row["name"] in set(chain["joint_names"]))
            anchor = chain_frames(chain, {})[first][:3, 3]
            ranked.append((float(np.linalg.norm(np.asarray(target_pose[:3]) - anchor)), index, root))
    for _, index, root in sorted(ranked, key=lambda row: row[:2])[:tries]:
        checked = reach(configuration, target_pose, root=root)
        if checked["feasibility_status"] == "FEASIBLE":
            return {**checked, "base_pose": list(base_poses[index][:3])}
    return {"feasibility_status": "NEEDS_CONTEXT",
            "reason": "No candidate base pose gave a configuration; reachability is unresolved"}
