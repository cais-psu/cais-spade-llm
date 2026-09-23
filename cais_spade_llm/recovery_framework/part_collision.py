"""Collision-scene updates for parts owned by an executing resource."""

from __future__ import annotations

import math
from typing import Any
from copy import deepcopy

from cais_spade_llm.recovery_framework import ROOT
from cais_spade_llm.recovery_framework.geometry import collision_boxes, rotate


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
        ROOT / 'ros2/cais_lab_robotics/models', {model_name: transform}, exact_models={model_name},
    ) if row['id'].startswith(model_name + '/')]
    if not rows:
        raise ValueError(f'No collision geometry for {model_name}')
    for row in rows:
        if not 0.0 <= support_allowance < row['size'][2]:
            raise ValueError(f'Invalid support contact allowance for {model_name}')
        row['support_contact_allowance_m'] = support_allowance
        original_height = row['size'][2]
        row['size'][2] -= support_allowance
        if 'cylinder' in row:
            row['cylinder'][0] = row['size'][2]
        if 'mesh' in row:
            row['mesh'] = deepcopy(row['mesh'])
            for vertex in row['mesh']['vertices']:
                vertex[2] *= row['size'][2] / original_height
        # CAD meshes may face down in their model; clearance follows the part's
        # support axis, not the mesh's local Z axis.
        offset = rotate(transform[3:], [0., 0., support_allowance / 2])
        row['pose'][:3] = [row['pose'][i] + offset[i] for i in range(3)]
    return rows


def collision_geometry_evidence(rows: list[dict]) -> list[dict]:
    """Record observed transforms and exact CAD provenance without copying triangles."""
    evidence = []
    for row in rows:
        item = {key: deepcopy(value) for key, value in row.items() if key != 'mesh'}
        if 'mesh' in row:
            mesh = row['mesh']
            item['mesh'] = {
                **{key: deepcopy(value) for key, value in mesh.items()
                   if key not in {'vertices', 'triangles'}},
                'vertex_count': len(mesh['vertices']), 'triangle_count': len(mesh['triangles']),
            }
        evidence.append(item)
    return evidence


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


def collision_object(row: dict, frame: str = "world") -> Any:
    """Build a MoveIt collision object preserving configured mating geometry."""
    from geometry_msgs.msg import Pose
    from moveit_msgs.msg import CollisionObject
    from shape_msgs.msg import SolidPrimitive

    pose = Pose()
    pose.position.x, pose.position.y, pose.position.z = row['pose'][:3]
    pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = row['pose'][3:]
    obj = CollisionObject(id=row['id'], operation=CollisionObject.ADD)
    obj.header.frame_id = frame
    if 'mesh' in row:
        from geometry_msgs.msg import Point
        from shape_msgs.msg import Mesh, MeshTriangle

        obj.meshes = [Mesh(
            vertices=[Point(x=v[0], y=v[1], z=v[2]) for v in row['mesh']['vertices']],
            triangles=[MeshTriangle(vertex_indices=v) for v in row['mesh']['triangles']],
        )]
        obj.mesh_poses = [pose]
    else:
        primitive = (SolidPrimitive(type=SolidPrimitive.CYLINDER, dimensions=row['cylinder'])
                     if 'cylinder' in row else SolidPrimitive(type=SolidPrimitive.BOX, dimensions=row['size']))
        obj.primitives = [primitive]
        obj.primitive_poses = [pose]
    return obj


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
    from moveit_msgs.msg import AttachedCollisionObject, CollisionObject, PlanningScene

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
        obj = collision_object(row, attached_link or 'world')
        if attached_link:
            scene.robot_state.attached_collision_objects.append(AttachedCollisionObject(
                link_name=attached_link, touch_links=list(touch_links or []), object=obj))
        else:
            scene.world.collision_objects.append(obj)
    return scene


def mating_pose_valid(authorization: dict, pose: list[float]) -> bool:
    """Bound permitted shaft contact to the observed, upright insertion corridor."""
    target = authorization['target_origin_pose']
    tolerance = authorization['axis_tolerance_m']
    if 'target_yaw_rad' in authorization:
        yaw = math.atan2(2 * (pose[6] * pose[5] + pose[3] * pose[4]),
                         1 - 2 * (pose[4] ** 2 + pose[5] ** 2))
        error = yaw - authorization['target_yaw_rad']
        if abs(math.atan2(math.sin(error), math.cos(error))) > .002:
            return False
    return (all(math.isfinite(value) for value in pose)
            and math.hypot(pose[0] - target['x'], pose[1] - target['y']) <= tolerance
            and target['z'] - tolerance <= pose[2] <= authorization['start_part_z'] + tolerance
            and 1 - 2 * (pose[3] ** 2 + pose[4] ** 2) >= math.cos(.02))


def mating_contacts_allowed(authorization: dict, contacts: list) -> bool:
    """Check shaft contact and the explicitly configured tooth-physics exemption."""
    owned = authorization['model_name'] + '/link/collision'
    expected = {owned, authorization['target_collision_object']}
    ignored_pairs = [{owned, other} for other in authorization.get('ignore_tooth_contact_with', [])]
    return bool(contacts) and all(
        math.isfinite(contact.depth) and contact.depth >= 0
        and (({contact.contact_body_1, contact.contact_body_2} == expected
              and contact.depth <= authorization['max_contact_depth_m'])
             or {contact.contact_body_1, contact.contact_body_2} in ignored_pairs)
        for contact in contacts
    )
