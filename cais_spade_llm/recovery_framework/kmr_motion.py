"""Joint limits and trajectory comparisons for the observed KMR delivery worker."""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from pathlib import Path
from typing import Any


def joint_limits(path: Path, names: Sequence[str], accelerations: Sequence[float]) -> dict:
    """Read bounded joints from the same URDF used by Gazebo and MoveIt."""
    root = ET.parse(path).getroot()
    if len(names) != len(accelerations):
        raise ValueError("KMR acceleration limits must cover every arm joint")
    limits = {}
    for name, acceleration in zip(names, accelerations, strict=True):
        element = root.find(f"joint[@name='{name}']/limit")
        if element is None:
            raise ValueError(f"Missing KMR joint limits: {name}")
        limits[name] = {key: float(element.get(key)) for key in ("lower", "upper", "velocity")}
        limits[name]["acceleration"] = float(acceleration)
        if (
            not all(math.isfinite(value) for value in limits[name].values())
            or limits[name]["lower"] >= limits[name]["upper"]
            or min(limits[name]["velocity"], acceleration) <= 0
        ):
            raise ValueError(f"Invalid KMR joint limits: {name}")
    return limits


def bounded_joints(names: Sequence[str], positions: Sequence[float], limits: dict) -> bool:
    """Check exact bounded joint values without wrapping angles into another branch."""
    return len(names) == len(positions) and all(
        name in limits
        and math.isfinite(value)
        and limits[name]["lower"] <= value <= limits[name]["upper"]
        for name, value in zip(names, positions, strict=True)
    )


def seconds(duration: Any) -> float:
    """Read a ROS duration without importing ROS into the UI or tests."""
    return duration.sec + duration.nanosec / 1e9


def retime_trajectory(trajectory: Any, limits: dict, velocity: float, acceleration: float) -> None:
    """Apply requested scaling and enforce joint limits on a MoveIt timed path."""
    if not (0.0 < velocity <= 1.0 and 0.0 < acceleration <= 1.0):
        raise ValueError("KMR trajectory scaling must be in (0, 1]")
    names, points = trajectory.joint_names, trajectory.points
    if not points or len(names) != len(limits) or set(names) != set(limits):
        raise ValueError("KMR trajectory must contain every arm joint")
    scale = max(1.0 / velocity, 1.0 / math.sqrt(acceleration))
    previous = None
    for point in points:
        if not bounded_joints(names, point.positions, limits):
            raise ValueError("KMR trajectory exceeds a bounded joint limit")
        if len(point.velocities) != len(names) or len(point.accelerations) != len(names):
            raise ValueError("KMR trajectory lacks timed joint dynamics")
        for name, v, a in zip(names, point.velocities, point.accelerations, strict=True):
            if not math.isfinite(v) or not math.isfinite(a):
                raise ValueError("KMR trajectory contains non-finite dynamics")
            scale = max(
                scale,
                abs(v) / (limits[name]["velocity"] * velocity),
                math.sqrt(abs(a) / (limits[name]["acceleration"] * acceleration)),
            )
        if previous is not None:
            dt = seconds(point.time_from_start) - seconds(previous.time_from_start)
            if dt <= 0:
                raise ValueError("KMR trajectory time must increase")
            for index, name in enumerate(names):
                scale = max(
                    scale,
                    abs(point.positions[index] - previous.positions[index])
                    / dt
                    / (limits[name]["velocity"] * velocity),
                    math.sqrt(
                        abs(point.velocities[index] - previous.velocities[index])
                        / dt
                        / (limits[name]["acceleration"] * acceleration)
                    ),
                )
        elif seconds(point.time_from_start) < 0:
            raise ValueError("KMR trajectory starts before zero")
        previous = point
    for point in points:
        nanoseconds = round(seconds(point.time_from_start) * scale * 1e9)
        point.time_from_start.sec, point.time_from_start.nanosec = divmod(
            nanoseconds, 1_000_000_000
        )
        point.velocities = [value / scale for value in point.velocities]
        point.accelerations = [value / scale**2 for value in point.accelerations]


def trajectory_cost(trajectories: Sequence[Any]) -> tuple[float, float]:
    """Rank complete feasible motions by duration, then total unwrapped joint travel."""
    duration = sum(seconds(t.points[-1].time_from_start) for t in trajectories)
    travel = sum(
        abs(b - a)
        for t in trajectories
        for first, last in zip(t.points, t.points[1:], strict=False)
        for a, b in zip(first.positions, last.positions, strict=True)
    )
    return duration, travel


def downward_transfer_waypoints(start: Sequence[float], target: Sequence[float],
                                center: Sequence[float], settings: dict) -> list[list[float]]:
    """Construct the configured TCP clearance arc while preserving downward tilt."""
    from cais_spade_llm.resources.robot.cartesian_waypoints import sample_segment
    from cais_spade_llm.recovery_framework.geometry import rotate

    if any(rotate(list(pose[3:]), [0., 0., 1.])[2] > -math.cos(.02) for pose in (start, target)):
        raise ValueError('KMR transfer endpoints must face downward')
    left = math.atan2(start[1] - center[1], start[0] - center[0])
    right = math.atan2(target[1] - center[1], target[0] - center[0])
    angle = math.atan2(math.sin(right - left), math.cos(right - left))
    if abs(angle) < settings['minimum_turn_angle_rad']:
        return []
    count = max(1, math.ceil(abs(angle) / settings['turn_step_rad']))
    poses = sample_segment(start, target, linear_step=1e6, angular_step=1e6, minimum_samples=count)
    radius, height = settings['turn_radius_m'], max(start[2], target[2])
    for index, pose in enumerate(poses):
        angle_at_sample = left + angle * index / count
        pose[:3] = [center[0] + radius * math.cos(angle_at_sample),
                    center[1] + radius * math.sin(angle_at_sample), height]
    return poses
