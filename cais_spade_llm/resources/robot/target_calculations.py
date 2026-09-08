from __future__ import annotations

"""Pure vertical target arithmetic shared by runtime and proposal calculations.

Callers own input provenance, units, frames, defaults and branch admission. These
functions perform no perception, configuration lookup, controller initialization,
state mutation or robot action.
"""


def vertical_pick_bias(part_height: float, minimum: float, maximum: float) -> float:
    """Return the controller's bounded vertical grasp bias in metres."""
    return max(minimum, min(maximum, part_height * 0.25))


def controlled_link_height(tcp_z: float, tcp_offset_z: float, adjustment: float) -> float:
    """Convert a TCP height to controlled-link height using an explicit offset."""
    return tcp_z - tcp_offset_z + adjustment


def pick_travel_height(
    object_z: float,
    support_z: float,
    pick_z: float,
    approach_height: float,
    current_ee_z: float | None,
) -> float:
    """Return the existing vertical travel policy without prescribing a movement."""
    candidates = [object_z + approach_height, support_z + approach_height, pick_z + 0.05]
    if current_ee_z is not None:
        candidates.append(current_ee_z)
    return max(candidates)


def supported_part_origin_height(
    support_z: float, part_height: float, surface_gap: float, insertion_depth: float
) -> float:
    """Return the runtime's support-based part-origin policy."""
    return max(
        support_z + part_height * 0.5 + (surface_gap - insertion_depth),
        support_z + part_height * 0.5,
    )


def placement_poses(
    x: float,
    y: float,
    place_z: float,
    orientation: dict[str, float],
    *,
    simulation_assembly_slot: bool,
    insertion_depth: float,
) -> dict[str, dict[str, float]]:
    """Return the existing conditional placement waypoints as plain values."""
    insert_pose = {"x": x, "y": y, "z": place_z, **orientation}
    pre_insert_pose = dict(insert_pose)
    approach_pose = {"x": x, "y": y, "z": place_z + 0.05, **orientation}
    if simulation_assembly_slot:
        if not orientation:
            insert_pose.update(qx=0.0, qy=0.0, qz=0.0, qw=1.0)
        pre_insert_pose = {**insert_pose, "z": place_z + max(0.0, insertion_depth)}
        approach_pose = {**pre_insert_pose, "z": pre_insert_pose["z"] + 0.05}
    return {
        "insert_pose": insert_pose,
        "pre_insert_pose": pre_insert_pose,
        "approach_pose": approach_pose,
        "target_pose": dict(pre_insert_pose),
    }
