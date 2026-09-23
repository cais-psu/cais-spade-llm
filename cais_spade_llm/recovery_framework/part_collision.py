"""Collision-scene updates for parts owned by an executing resource."""

from __future__ import annotations

import math
from typing import Any

from cais_spade_llm.recovery_framework import ROOT
from cais_spade_llm.recovery_framework.geometry import collision_boxes, compose, rotate


def observed_part_boxes(model_name: str, pose: Any, *, support_allowance: float = 0.0) -> list[dict]:
    """Read the model's collision bounds in the observed Gazebo pose frame.

    Args:
        model_name: Exact Gazebo model identifier.
        pose: Observed pose in world or in the owning attachment link.
        support_allowance: Configured contact depth at the payload's bottom face.

    Returns:
        Collision boxes for this model only.
    """
    position, orientation = pose.position, pose.orientation
    transform = [position.x, position.y, position.z,
                 orientation.x, orientation.y, orientation.z, orientation.w]
    rows = [row for row in collision_boxes(
        ROOT / 'ros2/cais_lab_robotics/worlds/table_recovery_framework.world',
        ROOT / 'ros2/cais_lab_robotics/models', {model_name: transform},
    ) if row['id'].startswith(model_name + '/')]
    if not rows:
        raise ValueError(f'No collision geometry for {model_name}')
    for row in rows:
        if not 0.0 <= support_allowance < row['size'][2]:
            raise ValueError(f'Invalid support contact allowance for {model_name}')
        row['size'][2] -= support_allowance
        row['pose'] = compose(row['pose'], [0., 0., support_allowance / 2, 0., 0., 0., 1.])
    return rows


def grasp_point_evidence(rows: list[dict], point: list[float], tolerance_m: float) -> dict:
    """Check that the observed payload reaches the gripper's grasp point.

    Args:
        rows: Collision boxes in the same frame as point.
        point: Gripper TCP coordinates in that frame.
        tolerance_m: Existing Cartesian position tolerance.

    Returns:
        Distance evidence and whether the TCP is inside a payload bound.
    """
    distances = []
    for row in rows:
        pose, size = row["pose"], row["size"]
        local = rotate([-pose[3], -pose[4], -pose[5], pose[6]],
                       [point[i] - pose[i] for i in range(3)])
        distances.append(math.sqrt(sum(max(0., abs(local[i]) - size[i] / 2.)**2
                                       for i in range(3))))
    distance = min(distances, default=math.inf)
    return {"tcp_to_payload_distance_m": distance, "tolerance_m": tolerance_m,
            "payload_at_gripper": math.isfinite(distance) and distance <= tolerance_m}


def part_scene_update(
    model_name: str,
    rows: list[dict],
    *,
    attached_link: str | None = None,
    touch_links: list[str] | None = None,
    world_ids: set[str] | None = None,
    attached_ids: set[str] | None = None,
) -> Any:
    """Build a MoveIt diff affecting only the executing resource's part.

    Args:
        model_name: Exact Gazebo model identifier.
        rows: Observed boxes in attached_link, or world for released parts.
        attached_link: Owning robot link, or None for a released world object.
        touch_links: Owning gripper links allowed to contact its payload.
        world_ids: Observed world object identifiers.
        attached_ids: Observed attached object identifiers.

    Returns:
        A PlanningScene diff; other resources' objects are untouched.
    """
    from geometry_msgs.msg import Pose
    from moveit_msgs.msg import AttachedCollisionObject, CollisionObject, PlanningScene
    from shape_msgs.msg import SolidPrimitive

    scene = PlanningScene(is_diff=True)
    scene.robot_state.is_diff = True
    # KMR uses the bare model identifier for its payload. Remove it when custody
    # passes to a UR resource, together with any old CAD world representation.
    identifiers = {model_name, *(row['id'] for row in rows)}
    # MoveIt transfers an ADD attachment out of the world automatically. A
    # redundant REMOVE reports failure after that transfer has already happened.
    if model_name in (world_ids or set()):
        scene.world.collision_objects = [CollisionObject(id=model_name, operation=CollisionObject.REMOVE)]
    if not attached_link:
        scene.robot_state.attached_collision_objects = [
            AttachedCollisionObject(object=CollisionObject(
                id=identifier, operation=CollisionObject.REMOVE))
            for identifier in sorted(identifiers.intersection(attached_ids or set()))
        ]
    for row in rows:
        pose = Pose()
        pose.position.x, pose.position.y, pose.position.z = row['pose'][:3]
        pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = row['pose'][3:]
        obj = CollisionObject(id=row['id'], operation=CollisionObject.ADD)
        obj.header.frame_id = attached_link or 'world'
        obj.primitives = [SolidPrimitive(type=SolidPrimitive.BOX, dimensions=row['size'])]
        obj.primitive_poses = [pose]
        if attached_link:
            scene.robot_state.attached_collision_objects.append(AttachedCollisionObject(
                link_name=attached_link, touch_links=list(touch_links or []), object=obj))
        else:
            scene.world.collision_objects.append(obj)
    return scene
