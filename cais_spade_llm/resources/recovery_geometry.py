from __future__ import annotations

"""Pure geometric projection of resource-owned recovery evidence."""

from copy import deepcopy
import hashlib
import math
import xml.etree.ElementTree as ET
from fractions import Fraction
from typing import Any

from cais_spade_llm.agents.shared_information.recovery_validation_protocol import (
    recovery_validation_fingerprint,
)

from cais_spade_llm.resources.continuous_motion import (
    ContinuousMotion,
    _matrix_product,
    interval,
    kinematic_transforms,
    occupancy,
    quaternion_transform,
    representative_pose,
    transformed_box,
)
from cais_spade_llm.resources.primitive_observations_3d import _compose, _plain, _quaternion_product, _unit_pose


def owner_cartesian_endpoint_precision() -> dict:
    """Declare the native solver's numerical endpoint contract, not tracking error."""
    return {"source": "owner-calculated Cartesian target", "position_precision": 1e-13,
            "orientation_precision": 1e-13}


def _cartesian_endpoint_residual(actual: list, expected: list) -> dict:
    actual_pose = _unit_pose(actual, "projected owner endpoint")
    expected_pose = _unit_pose(expected, "owner-calculated target")
    position = math.sqrt(sum(float(actual_pose[i] - expected_pose[i]) ** 2 for i in range(3)))
    relative = _quaternion_product(expected_pose[3:],
        tuple(-value for value in actual_pose[3:6]) + (actual_pose[6],))
    orientation = 2 * math.atan2(math.sqrt(sum(float(value) ** 2 for value in relative[:3])),
                                 abs(float(relative[3])))
    return {"position_residual": position, "orientation_residual": orientation}


def _support_vector(value: Any, name: str) -> list[float]:
    if (not isinstance(value, list) or len(value) != 3
            or any(type(item) not in (int, float) or not math.isfinite(item) for item in value)):
        raise ValueError(f"Recorded {name} must contain three finite coordinates")
    return value


def _support_polygon(points: list[list[float]], center: list[float], gravity: list[float]) -> dict:
    """Check static equilibrium using the gravity projection and contact hull."""
    length = math.sqrt(sum(value * value for value in gravity))
    if length == 0:
        raise ValueError("A gravity-supported rest model requires nonzero gravity")
    up = [-value / length for value in gravity]
    axis = min(range(3), key=lambda index: abs(up[index]))
    horizontal = [float(index == axis) - up[index] * up[axis] for index in range(3)]
    scale = math.sqrt(sum(value * value for value in horizontal))
    horizontal = [value / scale for value in horizontal]
    vertical = [up[1] * horizontal[2] - up[2] * horizontal[1],
                up[2] * horizontal[0] - up[0] * horizontal[2],
                up[0] * horizontal[1] - up[1] * horizontal[0]]
    projected = sorted(set((sum(a * b for a, b in zip(point, horizontal, strict=True)),
                            sum(a * b for a, b in zip(point, vertical, strict=True))) for point in points))
    if len(projected) < 3:
        raise ValueError("Static support requires at least three distinct contact points")

    def cross(first, second, third):
        return ((second[0] - first[0]) * (third[1] - first[1])
                - (second[1] - first[1]) * (third[0] - first[0]))

    halves = []
    for population in (projected, list(reversed(projected))):
        hull = []
        for point in population:
            while len(hull) >= 2 and cross(hull[-2], hull[-1], point) <= 0:
                hull.pop()
            hull.append(point)
        halves.append(hull[:-1])
    hull = halves[0] + halves[1]
    projected_center = [sum(a * b for a, b in zip(center, basis, strict=True))
                        for basis in (horizontal, vertical)]
    if len(hull) < 3 or any(cross(first, second, projected_center) <= 0
            for first, second in zip(hull, [*hull[1:], hull[0]], strict=True)):
        raise ValueError("The gravity-projected center of mass is not strictly inside the support polygon")
    return {"gravity": gravity, "center_of_mass": center, "contact_polygon": [list(point) for point in hull],
            "projected_center_of_mass": projected_center, "up": up}


def disabled_part_observation(*, model_name: str, physics: dict, owners_idle: bool) -> dict:
    """Read the conditional ODE hold state of a recorded disabled rigid body.

    ODE skips disabled bodies during integration. New contacts, owner commands,
    forces, or attachment changes invalidate this model; it does not establish
    a physical rest state or future execution tracking. ``owners_idle`` can
    describe a supported conditional commanded hold, not measured zero motion.
    """
    result = {"status": "NEEDS_CONTEXT", "kind": "Gazebo ODE disabled body",
              "physical_execution_verified": False, "future_execution_tracking": "not_established"}
    try:
        if physics.get("physics_engine") != "ode" or owners_idle is not True:
            raise ValueError("Disabled-body prediction requires recorded ODE physics and idle owners")
        model = physics["models"][model_name]
        if (model.get("static") is not False or type(model.get("joint_count")) is not int or model["joint_count"] != 0
                or len(model.get("links", {})) != 1):
            raise ValueError("Disabled-body prediction requires one dynamic rigid link with no model joints")
        if physics.get("attachment_complete") is not True or not isinstance(physics.get("attachments"), list):
            raise ValueError("Disabled-body prediction requires complete recorded attachment evidence")
        if any(model_name in (row.get("model1"), row.get("model2")) for row in physics["attachments"]):
            raise ValueError("An attached body needs its owning resource's geometry projection")
        link = next(iter(model["links"].values()))
        if link.get("enabled") is not False:
            raise ValueError("The recorded ODE body is enabled")
        for field in ("linear_velocity", "angular_velocity"):
            if any(value != 0 for value in _support_vector(link[field], field)):
                raise ValueError("The disabled body has conflicting nonzero recorded motion")
        own = {row["id"] for row in link["collisions"]}
        if not own:
            raise ValueError("Recorded part collision identities are unavailable")
        collisions = {collision["id"]: body.get("static") for body in physics["models"].values()
            for body_link in body.get("links", {}).values() for collision in body_link.get("collisions", [])}
        for row in physics["contacts"]:
            first, second = row.get("collision1"), row.get("collision2")
            if first in own or second in own:
                other = second if first in own else first
                if collisions.get(other) is not True:
                    raise ValueError("A disabled body touching another dynamic body has an unresolved wake-up condition")
        return {**result, "status": "supported", "source": {
            "service": "GETRECOVERYSTATE", "instance_id": physics["instance_id"],
            "simulation_time": physics["simulation_time"]},
            "physics_engine": "ode", "enabled": False, "joint_count": 0,
            "model_assumptions": ["The recorded ODE body remains disabled",
                "No new contact, force, owner command, or attachment change wakes the body"],
            "reason": "Recorded disabled ODE body is excluded from integration until a wake-up condition occurs"}
    except (KeyError, ValueError, TypeError, IndexError) as exc:
        return {**result, "reason": str(exc)}


def fixed_attached_part_observation(*, model_name: str, current_pose: list,
                                    physics: dict, owners_idle: bool) -> dict:
    """Bind one observed fixed joint to an observed static endpoint.

    This is conditional rigid-constraint geometry, independent of contact or
    sampled zero speed. More than one attachment hop needs a separate model.
    """
    result = {"status": "NEEDS_CONTEXT", "kind": "recorded fixed attachment to static endpoint",
              "physical_execution_verified": False, "future_execution_tracking": "not_established"}
    try:
        if (owners_idle is not True or physics.get("attachment_complete") is not True
                or type(physics.get("attachment_evidence_version")) is not int
                or physics["attachment_evidence_version"] != 1
                or type(physics.get("attachment_revision")) is not int or physics["attachment_revision"] < 0
                or not isinstance(physics.get("instance_id"), str) or not physics["instance_id"]
                or type(physics.get("simulation_time")) not in (int, float)
                or not math.isfinite(physics["simulation_time"])):
            raise ValueError("Fixed-attachment support requires complete current native joint observations and idle owners")
        records = physics.get("attachments")
        if not isinstance(records, list):
            raise ValueError("Fixed-attachment graph is unavailable")
        models = physics["models"]
        adjacent = []
        for row in records:
            if not isinstance(row, dict):
                raise ValueError("Fixed-attachment graph contains an incomplete endpoint")
            endpoints = []
            for side in (1, 2):
                name, link = row.get("model"+str(side)), row.get("link"+str(side))
                if (not isinstance(name, str) or not name or not isinstance(link, str) or not link
                        or name not in models or link not in models[name].get("links", {})):
                    raise ValueError("Fixed-attachment graph contains an unresolved model or link")
                endpoints.append((name, link))
            if endpoints[0] == endpoints[1] or endpoints[0][0] == endpoints[1][0]:
                raise ValueError("Self attachments do not establish static support")
            if any(name == model_name for name, _ in endpoints):
                adjacent.append((row, endpoints))
        if len(adjacent) != 1:
            raise ValueError("Static support requires one unambiguous attachment for the part")
        row, endpoints = adjacent[0]
        part_side = 0 if endpoints[0][0] == model_name else 1
        other_model, other_link = endpoints[1-part_side]
        model, anchor = models[model_name], models[other_model]
        if (model.get("static") is not False or len(model.get("links", {})) != 1
                or type(model.get("joint_count")) is not int
                or model["joint_count"] != int(part_side == 0)):
            raise ValueError("Fixed-attachment support requires one dynamic rigid part link with no other joints")
        if anchor.get("static") is not True:
            raise ValueError("The directly attached endpoint is not observed static; additional attachment hops need context")
        if any(other is not row and other_model in (other["model1"], other["model2"])
               and model_name in (other["model1"], other["model2"]) for other in records):
            raise ValueError("Conflicting attachment paths do not establish a static endpoint")
        native = row.get("joint_evidence")
        if (not isinstance(native, dict) or native.get("status") != "observed"
                or native.get("fixed_joint") is not True
                or not isinstance(native.get("joint_name"), str) or not native["joint_name"]
                or native.get("joint_scoped_name") != row["model1"]+"::"+native["joint_name"]):
            raise ValueError("The recorded attachment lacks an observed live fixed joint")
        poses = []
        for key, (name, link_name) in zip(("parent", "child"), endpoints, strict=True):
            declared = native.get(key)
            observed = models[name]["links"][link_name]
            if (not isinstance(declared, dict) or declared.get("model") != name
                    or declared.get("link") != link_name
                    or declared.get("scoped_name") != name+"::"+link_name
                    or recovery_validation_fingerprint(declared.get("pose")) != recovery_validation_fingerprint(observed.get("pose"))):
                raise ValueError("Live attachment endpoints differ from the same physics snapshot")
            poses.append(_unit_pose(observed["pose"], "fixed attachment endpoint pose"))
            for field in ("linear_velocity", "angular_velocity"):
                if any(value != 0 for value in _support_vector(observed[field], field)):
                    raise ValueError("Recorded motion conflicts with fixed attachment to a static endpoint")
        pose = _unit_pose(current_pose, "fixed attached part pose")
        if recovery_validation_fingerprint(current_pose) != recovery_validation_fingerprint(model.get("pose")):
            raise ValueError("Part pose and fixed attachment belong to different observations")

        def relative(first, second):
            norm = sum(value * value for value in first[3:])
            rotation = tuple(-value / norm for value in first[3:6]) + (first[6] / norm,)
            rotated = _quaternion_product(_quaternion_product(rotation, tuple(-value for value in first[:3]) + (Fraction(0),)), first[3:])
            return _compose(rotated[:3] + rotation, second)

        measured = _plain(_unit_pose(native.get("parent_to_child"), "recorded fixed-joint transform"))
        calculated = [_plain(value) for value in relative(poses[0], poses[1])]
        residual = _cartesian_endpoint_residual(measured, calculated)
        if any(value > 1e-12 for value in residual.values()):
            raise ValueError("Recorded fixed-joint transform conflicts with its observed endpoint poses")
        return {**result, "status": "supported", "model_name": model_name,
            "current_pose": deepcopy(current_pose), "attachment": deepcopy(row),
            "static_endpoint": {"model": other_model, "link": other_link,
                "pose": deepcopy(models[other_model]["links"][other_link]["pose"])},
            "static_endpoint_to_part": [_plain(value) for value in relative(poses[1-part_side], pose)],
            "transform_residual": residual, "transform_numerical_precision": 1e-12,
            "source": {"service": "GETRECOVERYSTATE", "instance_id": physics["instance_id"],
                "simulation_time": physics["simulation_time"], "attachment_revision": physics["attachment_revision"],
                "attachment_evidence_version": physics["attachment_evidence_version"],
                "physics_fingerprint": recovery_validation_fingerprint(physics)},
            "model_assumptions": ["The observed native fixed joint and static endpoint remain unchanged",
                "No external pose command or attachment change acts on either endpoint"],
            "reason": "Observed native fixed joint connects the rigid part to an observed static endpoint"}
    except (KeyError, ValueError, TypeError, IndexError) as exc:
        return {**result, "reason": str(exc)}


def gravity_supported_part(*, model_name: str, physics: dict, policy: dict,
                           owners_idle: bool) -> dict:
    """Calculate conditional rest from captured static contacts and mass geometry.

    The declared model represents measured motion within its configured idle
    limits by static equilibrium. It is a model prediction, not a future Gazebo
    tracking guarantee, and is valid only while the support and other owners
    remain unchanged and no external force or contact is introduced.
    ``owners_idle`` includes separately supported conditional commanded holds;
    their observed link motion and enclosing geometry remain in owner evidence.
    """
    result = {"status": "NEEDS_CONTEXT", "kind": "gravity_supported_rest",
              "physical_execution_verified": False, "future_execution_tracking": "not_established"}
    try:
        if (policy.get("kind") != "gravity_supported_rest" or policy.get("requires_no_active_goals") is not True
                or policy.get("future_execution_tracking") != "not_established" or owners_idle is not True):
            raise ValueError("The captured scene has no applicable gravity-supported rest contract")
        limits = {key: policy[key] for key in ("stationary_linear_speed_max", "stationary_angular_speed_max")}
        if any(type(value) not in (int, float) or not math.isfinite(value) or value < 0 for value in limits.values()):
            raise ValueError("Stationary support motion limits must be declared and finite")
        gravity = _support_vector(physics["gravity"], "gravity")
        length = math.sqrt(sum(value * value for value in gravity))
        if length == 0:
            raise ValueError("A gravity-supported rest model requires nonzero gravity")
        up = [-value / length for value in gravity]
        model = physics["models"][model_name]
        if model.get("static") is not False or len(model.get("links", {})) != 1:
            raise ValueError("Gravity-supported rest requires one recorded dynamic rigid link")
        link = next(iter(model["links"].values()))
        mass = link["mass"]
        if type(mass) not in (int, float) or not math.isfinite(mass) or mass <= 0:
            raise ValueError("Recorded dynamic link mass must be positive")
        center = _support_vector(link["center_of_mass"], "center_of_mass")
        for field, limit in (("linear_velocity", limits["stationary_linear_speed_max"]),
                             ("angular_velocity", limits["stationary_angular_speed_max"])):
            values = _support_vector(link[field], field)
            if math.sqrt(sum(value * value for value in values)) > limit:
                raise ValueError("Recorded part motion exceeds the declared stationary support limits")
        collisions = {collision["id"]: (name, body.get("static"))
            for name, body in physics["models"].items() for body_link in body.get("links", {}).values()
            for collision in body_link.get("collisions", [])}
        own = {collision["id"] for collision in link.get("collisions", [])}
        if not own:
            raise ValueError("Recorded part collision identities are unavailable")
        contacts, supports = [], set()
        for row in physics["contacts"]:
            first, second = row.get("collision1"), row.get("collision2")
            if first not in own and second not in own:
                continue
            other = second if first in own else first
            if other not in collisions or collisions[other][1] is not True:
                raise ValueError("The part has contact with an unsupported moving body")
            positions, normals = row.get("positions"), row.get("normals")
            if not isinstance(positions, list) or not positions or not isinstance(normals, list) or len(positions) != len(normals):
                raise ValueError("Static support contact geometry is incomplete")
            for position, normal in zip(positions, normals, strict=True):
                position = _support_vector(position, "contact position")
                normal = _support_vector(normal, "contact normal")
                norm = math.sqrt(sum(value * value for value in normal))
                # This tolerance covers floating-point unit-vector arithmetic;
                # a tilted support needs a separate friction model.
                if norm == 0 or not math.isclose(abs(sum(a * b for a, b in zip(normal, up, strict=True))) / norm,
                                               1., rel_tol=1e-12, abs_tol=1e-12):
                    raise ValueError("Nonhorizontal support requires a declared friction and contact model")
                if sum((center[index] - position[index]) * up[index] for index in range(3)) <= 0:
                    raise ValueError("A supporting contact must lie below the center of mass along gravity")
                contacts.append(position)
            supports.add(other)
        witness = _support_polygon(contacts, center, gravity)
        return {**result, "status": "supported", **witness, "support_collisions": sorted(supports),
                "policy_fingerprint": recovery_validation_fingerprint(policy),
                "source": {"service": "GETRECOVERYSTATE", "instance_id": physics["instance_id"],
                           "simulation_time": physics["simulation_time"]},
                "model_assumptions": ["Captured motion within the declared idle limits is represented by rest",
                    "The recorded static support and gravity remain unchanged",
                    "No external force, new contact, or active owner command acts on the part"]}
    except (KeyError, ValueError, TypeError, IndexError) as exc:
        return {**result, "reason": str(exc)}


def _part_support_geometry(evidence: dict, state: dict, part_name: str) -> dict:
    record = evidence['part_support_geometry']
    body = {key: value for key, value in record.items() if key != 'fingerprint'}
    if (record.get('version') != 1 or record.get('fingerprint') != recovery_validation_fingerprint(body)
            or record['policy_fingerprint'] != recovery_validation_fingerprint(evidence.get('part_support_policy'))
            or record['meshes_fingerprint'] != recovery_validation_fingerprint(evidence.get('physics_meshes'))):
        raise ValueError('Recorded support geometry dependencies changed')
    part = record['parts'][part_name]
    if (part['model_name'] != state['parts'][part_name]['model_name'] or part['geometry_fingerprint']
            != recovery_validation_fingerprint(evidence['geometry']['parts'][part_name])):
        raise ValueError('Recorded part geometry changed')
    resources = {name for name, source in record['resources'].items()
        if source['current_pose'] == state['resources'].get(name, {}).get('current_pose')
        and source['geometry_fingerprint'] == recovery_validation_fingerprint(evidence['geometry']['resources'].get(name))}
    physics = deepcopy(record['physics'])
    # Other dynamic parts and all old contacts/attachments are outside this
    # calculation. This reference supplies shapes, never current observations.
    physics['models'] = {name: row for name, row in physics['models'].items()
                         if name == part['model_name'] or name in resources}
    physics['excluded_support_resources'] = sorted(set(record['resources']) - resources)
    return physics


def projected_part_support(*, part_name: str, state: dict, evidence: dict) -> dict:
    """Calculate resting contact for a proposed deposit from exact collision shapes.

    Rigid boxes, upright cylinders, and exact captured meshes may contact
    recorded horizontal box faces. Collision AABBs alone never establish
    contact or stability. The contact margin comes from the physics record.
    """
    from cais_spade_llm.recovery_framework.geometry import compose, rotate

    result = {"status": "NEEDS_CONTEXT", "physical_execution_verified": False}
    try:
        physics = (deepcopy(evidence['physics_evidence']) if evidence.get('physics_evidence')
                   else _part_support_geometry(evidence, state, part_name))
        physics_fingerprint = recovery_validation_fingerprint(physics)
        policy = evidence["part_support_policy"]
        if physics.get("physics_engine") != "ode":
            raise ValueError("Projected support requires recorded ODE contact settings")
        surface_layer = physics.get("contact_surface_layer")
        sdf = physics.get("physics_configuration", {}).get("sdf")
        if sdf is not None:
            try:
                declared = ET.fromstring(sdf).findtext("./ode/constraints/contact_surface_layer")
                if declared is None:
                    raise ValueError("Recorded physics configuration lacks contact_surface_layer")
                recorded_layer = float(declared)
            except (ET.ParseError, TypeError) as exc:
                raise ValueError("Recorded physics configuration is invalid") from exc
            if "contact_surface_layer" in physics and surface_layer != recorded_layer:
                raise ValueError("Recorded contact_surface_layer values conflict")
            surface_layer = recorded_layer
        if type(surface_layer) not in (int, float) or not math.isfinite(surface_layer) or surface_layer < 0:
            raise ValueError("Recorded contact_surface_layer is unavailable or invalid")
        part = state["parts"][part_name]
        name, pose = part["model_name"], list(_unit_pose(part["current_pose"], "projected part pose"))
        recorded_pose = deepcopy(part["current_pose"])
        model = physics["models"][name]
        original = list(_unit_pose(model["pose"], "recorded model pose"))
        inverse_rotation = [-original[3], -original[4], -original[5], original[6]]
        inverse = [*rotate(inverse_rotation, [-value for value in original[:3]]), *inverse_rotation]
        transform = compose(pose, inverse)
        gravity = _support_vector(physics["gravity"], "gravity")
        scale = math.sqrt(sum(value * value for value in gravity))
        if not scale:
            raise ValueError("Projected support requires nonzero recorded gravity")
        up = [-value / scale for value in gravity]
        if len(model.get("links", {})) != 1:
            raise ValueError("Projected support requires one rigid recorded link")
        link = next(iter(model["links"].values()))
        center = _support_vector(link["center_of_mass"], "center_of_mass")
        link["center_of_mass"] = compose(transform, [*center, 0., 0., 0., 1.])[:3]
        link["pose"] = compose(transform, link["pose"])
        # The caller verifies a stopped motion endpoint before using this model.
        link.update(linear_velocity=[0., 0., 0.], angular_velocity=[0., 0., 0.])
        model["pose"] = deepcopy(recorded_pose)
        own = {row["id"] for row in link["collisions"]}
        physics["contacts"] = [row for row in physics.get("contacts", [])
                               if not own.intersection((row.get("collision1"), row.get("collision2")))]
        surfaces = []
        for other_name, other_model in physics["models"].items():
            if other_name == name or other_model.get("static") is not True:
                continue
            for other_link in other_model.get("links", {}).values():
                for collision in other_link.get("collisions", []):
                    shape = collision.get("geometry") or {}
                    if shape.get("type") != "box":
                        continue
                    sizes = _support_vector(shape.get("size"), "support box size")
                    surface_pose = list(_unit_pose(collision["pose"], "support collision pose"))
                    for axis in range(3):
                        for direction in (-1., 1.):
                            normal = rotate(surface_pose[3:], [direction if index == axis else 0. for index in range(3)])
                            if not all(math.isclose(value, target, rel_tol=1e-12, abs_tol=1e-12)
                                       for value, target in zip(normal, up, strict=True)):
                                continue
                            point = compose(surface_pose,
                                [*(direction * sizes[index] / 2 if index == axis else 0. for index in range(3)),
                                 0., 0., 0., 1.])[:3]
                            surfaces.append((collision, sizes, surface_pose, axis, point))
        contact_margins = []
        for collision in link["collisions"]:
            collision["pose"] = compose(transform, collision["pose"])
            shape = collision.get("geometry") or {}
            collision_pose = list(_unit_pose(collision["pose"], "part collision pose"))
            if shape.get("type") == "box":
                sizes = _support_vector(shape.get("size"), "part box size")
                vertices = [[x * sizes[0] / 2, y * sizes[1] / 2, z * sizes[2] / 2]
                            for x in (-1., 1.) for y in (-1., 1.) for z in (-1., 1.)]
            elif shape.get("type") == "cylinder":
                radius, length = shape.get("radius"), shape.get("length")
                if any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0
                       for value in (radius, length)):
                    raise ValueError("Recorded cylinder geometry is incomplete")
                axis = rotate(collision_pose[3:], [0., 0., 1.])
                if not math.isclose(abs(sum(a * b for a, b in zip(axis, up, strict=True))),
                                    1., rel_tol=1e-12, abs_tol=1e-12):
                    raise ValueError("Tilted cylinder settling has no recorded projection model")
                # An inscribed polygon is a conservative subset of a circular face.
                vertices = [[radius * math.cos(index * math.pi / 8), radius * math.sin(index * math.pi / 8), z * length / 2]
                            for index in range(16) for z in (-1., 1.)]
            elif shape.get("type") == "mesh":
                mesh = evidence.get("physics_meshes", {}).get(shape.get("uri"))
                scale = _support_vector(shape.get("scale"), "mesh scale")
                if (not isinstance(mesh, dict) or mesh.get("uri") != shape.get("uri")
                        or not mesh.get("source_sha256") or mesh.get("source_kind") != "recorded_collision_mesh"
                        or not isinstance(mesh.get("vertices"), list) or not mesh["vertices"]
                        or len(mesh["vertices"]) != mesh.get("vertex_count") or any(value <= 0 for value in scale)):
                    raise ValueError("Projected support requires the exact captured collision mesh vertices and scale")
                vertices = [[value * scale[index] for index, value in enumerate(_support_vector(vertex, "mesh vertex"))]
                            for vertex in mesh["vertices"]]
            else:
                raise ValueError("Projected support requires exact recorded collision geometry")
            points = [compose(collision_pose, [*vertex, 0., 0., 0., 1.])[:3] for vertex in vertices]
            for surface, sizes, surface_pose, axis, plane in surfaces:
                depths = (collision.get("contact_min_depth"), surface.get("contact_min_depth"), surface_layer)
                if any(type(value) not in (int, float) or not math.isfinite(value) or value < 0 for value in depths):
                    raise ValueError("Recorded contact_min_depth is unavailable or invalid")
                # Gazebo's documented contact allowance is capped by both
                # surfaces and the world setting. It cannot bridge an air gap.
                margin = min(depths)
                distances = [sum((point[index] - plane[index]) * up[index] for index in range(3)) for point in points]
                if min(distances) < -margin - 1e-12:
                    continue
                inverse_surface = [-surface_pose[3], -surface_pose[4], -surface_pose[5], surface_pose[6]]
                contact_points = []
                contact_depths = []
                for point, distance in zip(points, distances, strict=True):
                    if distance > 1e-12 or distance < -margin - 1e-12:
                        continue
                    local = rotate(inverse_surface, [point[index] - surface_pose[index] for index in range(3)])
                    if all(abs(local[index]) <= sizes[index] / 2 for index in range(3) if index != axis):
                        contact_points.append([point[index] - distance * up[index] for index in range(3)])
                        contact_depths.append(max(-distance, 0.))
                if contact_points:
                    contact_margins.append({"collision1": collision["id"], "collision2": surface["id"],
                        "contact_min_depth1": depths[0], "contact_min_depth2": depths[1],
                        "contact_surface_layer": surface_layer, "penetration_allowance": margin})
                    physics["contacts"].append({"collision1": collision["id"], "collision2": surface["id"],
                        "positions": contact_points, "normals": [up[:] for _ in contact_points],
                        "depths": contact_depths})
        supported = gravity_supported_part(model_name=name, physics=physics, policy=policy, owners_idle=True)
        supported.update(current_pose=deepcopy(recorded_pose), part_name=part_name,
            source={"kind": "projected", "calculation": "recorded rigid collision geometry and static support",
                    "physics_fingerprint": physics_fingerprint,
                    "recorded_geometry_reference_fingerprint": (None if evidence.get('physics_evidence')
                        else evidence['part_support_geometry']['fingerprint']),
                    "excluded_support_resources": physics.get('excluded_support_resources', []),
                    "part_pose": deepcopy(recorded_pose), "policy_fingerprint": recovery_validation_fingerprint(policy),
                    "contact_margins": contact_margins},
            physics_evidence=physics)
        return supported
    except (KeyError, ValueError, TypeError, IndexError) as exc:
        return {**result, "reason": str(exc)}


def _hold_controller_activity(physical: dict, configuration: dict, robot_description: str) -> tuple[dict, dict]:
    """Scope a fixed root's hold to controllers that can change its joint tree.

    Unknown controller interfaces retain the global check. A mobile root also
    retains it; a fixed root is established from the bound complete description,
    never from a resource name or a captured pose alone.
    """
    if hashlib.sha256(robot_description.encode()).hexdigest() != configuration.get("description_sha256"):
        raise ValueError("Hold controller scope differs from the captured robot description")
    robot = ET.fromstring(robot_description)
    declarations = robot.findall("joint")
    joints = {joint.get("name"): joint for joint in declarations}
    if (robot.tag != "robot" or len(joints) != len(declarations) or None in joints
            or any(joint.find("child") is None or joint.find("parent") is None for joint in declarations)):
        raise ValueError("Hold controller scope needs an unambiguous complete joint tree")
    parents = {joint.find("child").get("link"): joint for joint in declarations}
    if len(parents) != len(declarations) or None in parents:
        raise ValueError("Hold controller scope has ambiguous child links")
    link, visited = configuration["root"], set()
    activity = physical["controller_hold_activity"]
    while link in parents:
        if link in visited:
            raise ValueError("Hold controller scope has a cyclic root chain")
        visited.add(link)
        joint = parents[link]
        if joint.get("type") != "fixed":
            return activity, {}
        link = joint.find("parent").get("link")
    if link != "world" or "base_pose" in physical.get("observation_state", {}):
        return activity, {}
    owned = {joint["name"] for joint in configuration["joints"]}
    if not owned.issubset(joints):
        raise ValueError("Hold controller scope omits declared owner joints")
    selected, excluded = {}, {}
    for source, record in activity.items():
        names = record.get("joint_names")
        if isinstance(names, list) and names and len(set(names)) == len(names) and set(names).issubset(joints):
            if owned.intersection(names):
                selected[source] = record
            else:
                excluded[source] = "Controller joints are outside the owner's fixed-root joint tree"
        elif record.get("navigation_enabled") is True and not names:
            # Base navigation cannot move a root connected to world solely by
            # fixed joints in this model. It still needs its own motion checks.
            excluded[source] = "Base navigation does not actuate the owner's fixed-root joint tree"
        else:
            selected[source] = record
    controller = physical["controller_hold_observation"]
    if not any(row.get("controller") == controller.get("controller")
               and row.get("instance_id") == controller.get("instance_id")
               and row.get("joint_names") == configuration["joint_names"] for row in selected.values()):
        raise ValueError("Hold controller scope lacks the recorded owner's controller")
    return selected, {"description_sha256": configuration["description_sha256"],
        "fixed_root": configuration["root"], "checked_controllers": sorted(selected),
        "excluded_controllers": excluded}


def commanded_hold_evidence(*, resource_id: str, state: dict, model: dict, physical: dict,
                            robot_description: str | None = None, include_observation_roundoff: bool = False) -> dict:
    """Bound a conditional model hold using saved commands and declared geometry.

    MoveIt's start tolerance is reused only as the existing model assumption;
    it is not a controller tracking guarantee. Observed motion is not relabelled
    idle, and the recorded state must lie inside the assumed envelope.

    Args:
        resource_id: Exact recorded owner identifier.
        state: Starting physical resource state.
        model: Recorded owner geometry and joint declarations.
        physical: Recorded observations and commanded hold contract.
        robot_description: Optional bound description for fixed-root controller
            scoping. Omitting it preserves the legacy global activity check.
        include_observation_roundoff: Enclose passive observations within eight
            representable floating-point steps of a limit. Defaults to legacy
            strict comparison; no observed value is clipped to a limit.

    Returns:
        Independently calculated conditional bounds or missing-evidence reasons.
    """
    result = {"status": "NEEDS_CONTEXT", "resource_id": resource_id,
              "physical_execution_verified": False, "future_execution_tracking": "not_established"}
    try:
        contract = physical["stationary_contract"]
        if contract != {"kind": "idle_commanded_hold", "requires_no_running_tasks": True,
                "requires_no_active_goals": True, "future_execution_tracking": "not_established"}:
            raise ValueError("The owner's declared commanded hold model is unavailable")
        controller = physical["controller_hold_observation"]
        activity = physical["controller_hold_activity"]
        if not isinstance(activity, dict) or not activity:
            raise ValueError("Complete recorded controller activity is unavailable")
        if robot_description is not None:
            activity, scope = _hold_controller_activity(physical, model["configuration"], robot_description)
            if scope:
                result["controller_activity_scope"] = scope
        for record in activity.values():
            if ("command_gate" in record or "command_gate_evidence" in record
                    or "owned_activity" in record or record.get("navigation_enabled") is True):
                from cais_spade_llm.agents.central_controller.recovery_case_evidence import recorded_command_gate

                idle = recorded_command_gate(record)
            else:
                idle = record
            if (idle.get("holding") is not True or idle.get("has_active_goal") is not False
                    or idle.get("has_pending_goal") is not False):
                raise ValueError("Another recorded controller has active or unconfirmed work")
        if (controller.get("holding") is not True or controller.get("has_active_goal") is not False
                or controller.get("has_pending_goal") is not False or not controller.get("instance_id")
                or controller.get("contract_state", "unlocked") != "unlocked"
                or controller.get("reservation_token", "") != ""):
            raise ValueError("Recorded controller has no unreserved commanded hold")
        source = physical["controller_hold_model"]
        error = source["value"]
        if (source.get("kind") != "conditional model assumption"
                or source.get("parameter") != "trajectory_execution.allowed_start_tolerance"
                or source.get("physical_execution_verified") is not False
                or source.get("future_execution_tracking") != "not_established"
                or type(error) not in (int, float) or not math.isfinite(error) or error < 0):
            raise ValueError("The explicit model hold envelope is unavailable")
        configuration = model["configuration"]
        if "base_pose" in state:
            from cais_spade_llm.agents.central_controller.recovery_case_evidence import recorded_base_hold

            base_hold = recorded_base_hold(resource_id=resource_id,
                physical_evidence={"resources": {resource_id: physical}}, model=configuration)
            base_pose = state["base_pose"]
            if (not isinstance(base_pose, list) or len(base_pose) != 7
                    or any(type(value) not in (int, float) or not math.isfinite(value) for value in base_pose)
                    or base_pose != physical["base_hold_evidence"]["root_pose"]):
                raise ValueError("Recorded base hold differs from the saved base pose")
            result["base_hold"] = base_hold
        names = configuration["joint_names"]
        projected = physical.get("projected_controller_hold")
        if projected is not None:
            if (projected.get("kind") != "projected" or not projected.get("binding")
                    or projected.get("joint_names") != names
                    or projected.get("positions") != state["joint_positions"]):
                raise ValueError("Projected controller hold does not match the accepted state")
            commands = projected["positions"]
            command_source = deepcopy(projected)
        else:
            if (controller.get("joint_names") != names or state.get("joint_names") != names
                    or state.get("joint_positions") != physical.get("observation_state", {}).get("joint_positions")
                    or controller.get("commanded_positions_source") != "native position command interfaces"):
                raise ValueError("Recorded controller hold differs from the saved joint identities or state")
            commands = controller.get("commanded_positions")
            command_source = {key: deepcopy(controller.get(key)) for key in
                ("source", "instance_id", "simulation_time", "sequence", "command_revision", "contract_revision")}
        if (not isinstance(commands, list) or len(commands) != len(names)
                or any(type(value) not in (int, float) or not math.isfinite(value) for value in commands)):
            raise ValueError("Complete finite native hold targets are unavailable")
        bounds = {name: [math.nextafter(value - error, -math.inf), math.nextafter(value + error, math.inf)]
                  for name, value in zip(names, commands, strict=True)}
        if any(not bounds[name][0] <= value <= bounds[name][1]
               for name, value in zip(names, state["joint_positions"], strict=True)):
            raise ValueError("Recorded joint state lies outside the declared model hold envelope")
        for joint in configuration["joints"]:
            if joint["type"] == "fixed" or joint["name"] in bounds:
                continue
            declared = joint.get("position_bounds")
            if not isinstance(declared, list) or len(declared) != 2:
                raise ValueError("Passive joint has no declared position bounds: " + joint["name"])
            low, high = interval(*declared)
            observed = joint["frozen_position"]
            if not low <= observed <= high:
                rounding = 8 * math.ulp(max(abs(low), abs(high))) if include_observation_roundoff else 0.
                if not low - rounding <= observed <= high + rounding:
                    raise ValueError("Passive joint observation exceeds declared position bounds: " + joint["name"])
                result.setdefault("observation_roundoff", {})[joint["name"]] = {
                    "declared_bounds": [low, high], "observed": observed, "rounding_bound": rounding}
                low, high = min(low, math.nextafter(observed, -math.inf)), max(high, math.nextafter(observed, math.inf))
            bounds[joint["name"]] = [low, high]
        result.update(status="supported", joint_intervals=bounds, joint_names=deepcopy(names),
            start_joint_positions=deepcopy(state["joint_positions"]), command_source=command_source,
            model_fingerprint=recovery_validation_fingerprint(configuration), bound_source=deepcopy(source),
            model_assumptions=["The recorded commanded target is retained without new commands",
                "Commanded joints remain within the explicit joint_position_error model envelope",
                "Passive joints remain within their recorded URDF position limits"],
            observed_idle=physical.get("idle"))
        return result
    except (KeyError, ValueError, TypeError, IndexError, ET.ParseError) as exc:
        return {**result, "reason": str(exc)}


def _joint_geometry(row: dict, model: dict, *, joint_intervals: dict | None = None) -> tuple[list, list]:
    configuration = model["configuration"]
    names, positions = row["joint_names"], row["joint_positions"]
    if names != configuration["joint_names"] or len(names) != len(positions) or len(set(names)) != len(names):
        raise ValueError("Recorded joint identities do not match the resource model")
    values = {name: interval(value) for name, value in zip(names, positions, strict=True)}
    if joint_intervals is not None:
        declared = {joint["name"] for joint in configuration["joints"] if joint["type"] != "fixed"}
        if set(joint_intervals) != declared:
            raise ValueError("Conditional hold intervals must cover every movable joint")
        values = {name: interval(*bounds) for name, bounds in joint_intervals.items()}
    known, used = {configuration["root"]}, set()
    for joint in configuration["joints"]:
        if joint["parent"] not in known or joint["child"] in known:
            raise ValueError("Recorded joint model must be an ordered tree")
        known.add(joint["child"])
        if joint["type"] not in {"fixed", "revolute", "continuous", "prismatic"}:
            raise ValueError("Unsupported recorded joint type")
        if joint["type"] != "fixed":
            if joint["name"] in values:
                if joint["name"] in names:
                    used.add(joint["name"])
            elif "frozen_position" not in joint and "frozen_interval" not in joint:
                raise ValueError("Recorded joint model has an unresolved joint")
    if used != set(names) or not configuration["components"]:
        raise ValueError("Recorded joints and component geometry must be complete")
    matrices = kinematic_transforms(configuration, values)
    boxes = [transformed_box(matrices[item["link"]], item["bounds"], item["id"])
             for item in configuration["components"]]
    return boxes, matrices[configuration["reference_link"]]


def _resource_boxes(result: dict, geometry: dict, models: dict) -> tuple[dict, dict, dict]:
    boxes, matrices, held = {}, {}, {}
    for rid, row in result["resources"].items():
        if rid in models and "configuration" in models[rid]:
            boxes[rid], matrices[rid] = _joint_geometry(row, models[rid])
        else:
            components = geometry["resources"][rid].get("component_bounds")
            if not components:
                raise ValueError(f"Complete recorded component bounds are missing for {rid}")
            boxes[rid] = [{"id": item["id"],
                           "minimum": [[interval(*pair)[0]] * 2 for pair in item["bounds"]],
                           "maximum": [[interval(*pair)[1]] * 2 for pair in item["bounds"]]}
                          for item in components]
            if any(len(item["minimum"]) != 3 for item in boxes[rid]):
                raise ValueError("Component bounds require three spatial axes")
        if "held_part" not in row:
            if geometry["resources"][rid].get("stationary_only") is not True:
                raise ValueError(f"Recorded custody is missing for {rid}")
        elif row["held_part"] is not None:
            part = row["held_part"]
            if part in held or part not in result["parts"]:
                raise ValueError("Recorded custody is conflicting or names an unknown part")
            held[part] = rid
    return boxes, matrices, held


def _checkpoint(snapshot: dict, geometry: dict, models: dict, *, index: int,
                moving: tuple[str, ContinuousMotion, float, float] | None = None,
                peer_holds: dict | None = None, _geometry_cache: dict | None = None) -> dict:
    if not snapshot.get("frame") or snapshot["frame"] != geometry.get("frame"):
        raise ValueError("Snapshot and geometry require the same explicit frame")
    if set(snapshot["resources"]) != set(geometry["resources"]):
        raise ValueError("Resource geometry must cover the complete recorded population")
    result = deepcopy(snapshot)
    # The cache is local to one _motion_checkpoint call with unchanged inputs.
    # Copies protect its resource box lists when carried parts are appended.
    local_geometry = {} if _geometry_cache is None else _geometry_cache
    if "resources" not in local_geometry:
        local_geometry["resources"] = _resource_boxes(result, geometry, models)
    resource_boxes, resource_matrices, held = local_geometry["resources"]
    boxes = {rid: list(items) for rid, items in resource_boxes.items()}
    matrices = dict(resource_matrices)
    part_boxes = {}
    if moving is not None:
        rid, motion, start, end = moving
        motion_matrices = motion.transforms(start, end)
        boxes[rid] = [transformed_box(motion_matrices[item["link"]], item["bounds"], item["id"])
                      for item in motion.configuration["components"]]
        matrices[rid] = motion_matrices[motion.configuration["reference_link"]]
    peer_geometry = local_geometry.setdefault("peers", {})
    for peer, evidence in (peer_holds or {}).items():
        if peer not in peer_geometry:
            if (evidence.get("status") != "supported"
                    or evidence.get("model_fingerprint") != recovery_validation_fingerprint(models[peer]["configuration"])
                    or evidence.get("start_joint_positions") != snapshot["resources"][peer]["joint_positions"]):
                raise ValueError("Conditional peer hold is stale or unsupported: " + peer)
            peer_geometry[peer] = _joint_geometry(result["resources"][peer], models[peer],
                                                  joint_intervals=evidence["joint_intervals"])
        peer_boxes, matrices[peer] = peer_geometry[peer]
        boxes[peer] = list(peer_boxes)
    for name, row in result["parts"].items():
        bounds = geometry["parts"][name]["local_bounds"]
        if name in held:
            rid = held[name]
            if row.get("contained_by") is not None:
                raise ValueError("A held part cannot also have a containment owner")
            grasp = _unit_pose(result["resources"][rid]["grasp_transform"], "grasp_transform")
            if rid not in matrices:
                raise ValueError("Carried part needs recorded resource link geometry")
            matrix = _matrix_product(matrices[rid], quaternion_transform(grasp[:3], grasp[3:]))
            part_boxes[name] = [transformed_box(matrix, bounds, name)]
            if moving is None and rid not in (peer_holds or {}):
                row["current_pose"] = representative_pose(matrix)
            boxes[rid].extend(part_boxes[name])
        else:
            pose = _unit_pose(row["current_pose"], "part current_pose")
            part_boxes[name] = [transformed_box(quaternion_transform(pose[:3], pose[3:]), bounds, name)]
    values, part_values = {}, {}
    for region, shape in geometry["regions"].items():
        if len(shape["bounds"]) != 3:
            raise ValueError("Region bounds require three spatial axes")
        for pair in shape["bounds"]:
            interval(*pair)
        values[region] = {rid: occupancy(items, shape["bounds"]) for rid, items in boxes.items()}
        part_values[region] = {name: occupancy(items, shape["bounds"]) for name, items in part_boxes.items()}
    result.update(time=index, time_exact=str(index), phase="between" if moving else "at",
                  occupancy_possibilities=values, part_occupancy_possibilities=part_values,
                  region_occupancy={key: {rid: v[0] if len(v) == 1 else None for rid, v in rows.items()}
                                    for key, rows in values.items()},
                  part_region_occupancy={key: {name: v[0] if len(v) == 1 else None for name, v in rows.items()}
                                         for key, rows in part_values.items()},
                  carried_parts={rid: [part for part, owner in held.items() if owner == rid]
                                 for rid in result["resources"]})
    return result


def _motion_checkpoint(snapshot: dict, geometry: dict, models: dict, *, index: int,
                       moving: tuple, peer_holds: dict | None = None) -> dict:
    """Refine enclosing motion bounds without adding semantic observations."""
    from cais_spade_llm.resources.continuous_observations import _refine

    rid, motion, _, _ = moving
    start, end = Fraction(0), motion.exact_times[-1]
    cache = {}
    geometry_cache = {}

    def observed(a, b):
        key = (a, b)
        if key not in cache:
            cache[key] = _checkpoint(snapshot, geometry, models, index=index,
                                    moving=(rid, motion, a, b), peer_holds=peer_holds,
                                    _geometry_cache=geometry_cache)
        return cache[key]

    row = deepcopy(observed(start, end))
    boundaries = {start, end}
    for field in ("occupancy_possibilities", "part_occupancy_possibilities"):
        for cell in _refine(start, end, lambda a, b: observed(a, b)[field], None):
            boundaries.update(Fraction(value) for value in cell["interval"])
    ordered = sorted(boundaries)
    cells = [{"interval": [str(a), str(b)], **{field: deepcopy(observed(a, b)[field])
              for field in ("occupancy_possibilities", "part_occupancy_possibilities")}}
             for a, b in zip(ordered, ordered[1:], strict=False)]
    row.update(continuous_interval=[str(start), str(end)], continuous_cells=cells)
    for field, summary in (("occupancy_possibilities", "region_occupancy"),
                           ("part_occupancy_possibilities", "part_region_occupancy")):
        for region, items in row[field].items():
            for identity in items:
                values = sorted({value for cell in cells for value in cell[field][region][identity]})
                items[identity] = values
                row[summary][region][identity] = values[0] if len(values) == 1 else None
    return row


def _part_entries(observations: list) -> None:
    preceding = None
    for row in observations:
        row["part_entry_possibilities"] = {}
        for region, parts in row["part_occupancy_possibilities"].items():
            entries = row["part_entry_possibilities"][region] = {}
            for part, possible in parts.items():
                if preceding is None:
                    entries[part] = [False]
                elif row["phase"] == "between":
                    entries[part] = ([False] if possible == [False] or
                                     possible == [True] and preceding[region][part] == [True]
                                     else [False, True])
                else:
                    entries[part] = sorted({current and not prior for current in possible
                                           for prior in preceding[region][part]})
        prior_cell = preceding
        for cell in row.get("continuous_cells", []):
            cell["part_entry_possibilities"] = {}
            for region, parts in cell["part_occupancy_possibilities"].items():
                entries = cell["part_entry_possibilities"][region] = {}
                for part, possible in parts.items():
                    entries[part] = ([False] if possible == [False] or
                        possible == [True] and prior_cell is not None and prior_cell[region][part] == [True]
                        else [False, True])
            prior_cell = cell["part_occupancy_possibilities"]
        if row["phase"] == "at":
            preceding = row["part_occupancy_possibilities"]


def _configuration_after(models: dict, rid: str, evidence: dict) -> None:
    if "configuration_after" not in evidence:
        return
    before, after = models[rid]["configuration"], evidence["configuration_after"]
    if ({key: value for key, value in before.items() if key not in {"joints", "fixed_joint_positions"}}
            != {key: value for key, value in after.items() if key not in {"joints", "fixed_joint_positions"}}
            or len(before["joints"]) != len(after["joints"])):
        raise ValueError("A tool-state projection cannot change the recorded geometry model")
    for first, last in zip(before["joints"], after["joints"], strict=True):
        permitted = {"frozen_position", "frozen_interval"} if first["name"] not in before["joint_names"] else set()
        if ({key: value for key, value in first.items() if key not in permitted}
                != {key: value for key, value in last.items() if key not in permitted}):
            raise ValueError("A tool-state projection can update only declared frozen joints")
    if set(before.get("fixed_joint_positions", {})) != set(after.get("fixed_joint_positions", {})):
        raise ValueError("A tool-state projection cannot change recorded fixed joint identities")
    for name, value in after.get("fixed_joint_positions", {}).items():
        joint = next((row for row in after["joints"] if row["name"] == name), None)
        if joint is not None:
            bounds = joint.get("frozen_interval")
            if joint.get("frozen_position") != value and not (
                    isinstance(bounds, list) and len(bounds) == 2 and bounds[0] <= value <= bounds[1]):
                raise ValueError("The fixed joint record differs from the projected joint configuration")
        if joint is None and value != before["fixed_joint_positions"][name]:
            raise ValueError("An undeclared fixed joint cannot be changed")
    models[rid]["configuration"] = deepcopy(after)


def _prepared_motion(state: dict, resource_id: str, models: dict, evidence: dict) -> ContinuousMotion:
    configuration = models[resource_id]["configuration"]
    if "configuration" in evidence and evidence["configuration"] != configuration:
        raise ValueError("Prepared motion changes the recorded resource model")
    motion = ContinuousMotion(evidence["joint_trajectory"], configuration)
    owner = state["resources"][resource_id]
    if motion.names != owner["joint_names"] or motion.points[0]["positions"] != owner["joint_positions"]:
        raise ValueError("Prepared motion does not start at the projected joint state")
    peers = set(state["resources"]) - {resource_id}
    if set(evidence.get("stationary_resources", [])) != peers:
        raise ValueError("Interval projection requires declared stationary peer resources")
    free_parts = set(state["parts"]) - {row.get("held_part") for row in state["resources"].values()}
    if set(evidence.get("stationary_parts", [])) != free_parts:
        raise ValueError("Interval projection requires declared stationary unheld parts")
    return motion


def _declared_completion(state: dict, observation: dict, row: dict,
                         support: dict, rid: str, geometry: dict) -> tuple[dict, dict]:
    from cais_spade_llm.agents.central_controller._product_effect_evidence import (
        _join_product_effects,
    )
    from cais_spade_llm.agents.central_controller.reviewed_primitive_program_safety import (
        _complete_target_ledger,
    )

    effects = row.get("declared_effects", {})
    before = row.get("start_state", {}).get("part", {}).get("processCompleted")
    end_part = row.get("end_state", {}).get("part", {})
    if "processCompleted" in effects and "processCompleted" not in end_part:
        raise ValueError("Declared process completion has no established resulting ledger")
    after = end_part.get("processCompleted", before)
    if after == before:
        return state, observation
    if "processCompleted" not in effects or not row.get("declaration_fingerprint"):
        raise ValueError("Projected process completion requires its declared primitive effect")
    part = support["part_name"]
    ledger = _complete_target_ledger(state["parts"][part])
    if (before != ledger or not isinstance(after, list) or len(after) <= len(ledger)
            or after[:len(ledger)] != ledger):
        raise ValueError("Projected process completion must append to the exact recorded ledger")
    _complete_target_ledger({**state["parts"][part], "processCompleted": after})
    added = deepcopy(after[len(ledger):])
    if any("target" in item and item["target"] != geometry["parts"][part].get("target") for item in added):
        raise ValueError("Projected process completion differs from the configured part target")
    update = {"task_id": support["task_fingerprint"], "resource_id": rid,
              "time": observation["time"], "kind": "predicted",
              "declaration": {"primitive": row["primitive"], "params": deepcopy(row["params"]),
                              "declaration_fingerprint": row["declaration_fingerprint"],
                              "effects": deepcopy(effects)},
              "product_effects": {part: {"processCompleted": added}},
              "source_kind": "resource_primitive_witness"}
    joined = _join_product_effects({"observations": [observation], "projected_snapshot": state},
                                   [update], state)
    return joined["projected_snapshot"], joined["observations"][0]


def _check_projected_custody(state: dict, row: dict, rid: str, phase: str) -> None:
    expected = row.get(phase, {}).get("resource", {})
    if "held_part" in expected and state["resources"][rid].get("held_part") != expected["held_part"]:
        raise ValueError("Projected geometry custody disagrees with the validated primitive transition")


def _completion_checkpoint(state: dict, row: dict, support: dict, rid: str,
                           geometry: dict, models: dict, index: int) -> tuple[dict, dict]:
    if not row.get("task_contract") or not row.get("declaration_fingerprint"):
        raise ValueError("Task completion requires its validated declaration witness")
    _check_projected_custody(state, row, rid, "start_state")
    _check_projected_custody(state, row, rid, "end_state")
    final = _checkpoint(state, geometry, models, index=index)
    return _declared_completion(state, final, row, support, rid, geometry)


def _custody(snapshot: dict, rid: str, effects: list, geometry: dict) -> None:
    for effect in effects:
        part, kind = effect["part"], effect["kind"]
        resource, row = snapshot["resources"][rid], snapshot["parts"][part]
        pose = _unit_pose(effect["pose"], "custody pose")
        actual = _unit_pose(row["current_pose"], "part current_pose")
        if any(abs(a - b) > 1e-9 for a, b in zip(pose, actual, strict=True)):
            raise ValueError("Custody evidence changes a part pose without motion evidence")
        if kind == "grasp":
            if resource["held_part"] is not None or any(r.get("held_part") == part for r in snapshot["resources"].values()):
                raise ValueError("Conflicting grasp custody")
            grasp = _unit_pose(effect["grasp_transform"], "grasp_transform")
            derived = _compose(tuple(resource["current_pose"]), grasp)
            if any(abs(a - b) > 1e-9 for a, b in zip(derived, pose, strict=True)):
                raise ValueError("Grasp transform disagrees with the recorded part pose")
            receiver = row.get("contained_by")
            if receiver is not None:
                snapshot["resources"][receiver]["contained_parts"].remove(part)
            resource.update(held_part=part, grasp_transform=_plain(grasp), gripper_state="closed")
            row["contained_by"] = None
        elif kind == "release":
            if resource["held_part"] != part or "contained_by" not in effect:
                raise ValueError("Release evidence does not match recorded custody")
            receiver = effect["contained_by"]
            if receiver is not None:
                snapshot["resources"][receiver]["contained_parts"].append(part)
            resource.update(held_part=None, grasp_transform=None, gripper_state="open")
            row.update(current_pose=_plain(pose), contained_by=receiver)
        else:
            raise ValueError("Unsupported declared custody effect")
        if part not in geometry["parts"]:
            raise ValueError("Custody effect has no configured part geometry")


def project_recovery_geometry(*, resource_id: str, snapshot: dict, geometry: dict,
                              resource_models: dict, primitive_support: dict,
                              step_evidence: dict | list | None = None) -> dict[str, Any]:
    """Project exact recorded primitive witnesses without invoking live interfaces.

    Numeric named joint poses establish endpoints only. Interval occupancy is
    supplied only by a validated owner trajectory and declared stationary peers.
    The result describes a model and never grants execution.
    """
    result: dict[str, Any] = {"status": "needs_context", "feasibility_status": "NEEDS_CONTEXT",
                              "observations": [], "unresolved": [], "interval_complete": True,
                              "geometry": deepcopy(geometry), "execution_authorized": False}
    state, resource_models = deepcopy(snapshot), deepcopy(resource_models)
    try:
        if primitive_support.get("semantics") != "primitive_transition":
            raise ValueError("A resource-validated primitive transition witness is required")
        primitives, trace = primitive_support["primitives"], primitive_support["trace"]
        if not primitives or len(primitives) != sum(row.get("primitive") is not None for row in trace):
            raise ValueError("Primitive transition witness is incomplete")
        observations = result["observations"]
        observations.append(_checkpoint(state, geometry, resource_models, index=0))
        state = {**state, "resources": deepcopy(observations[-1]["resources"]),
                 "parts": deepcopy(observations[-1]["parts"])}
        primitive_index = 0
        for row in trace:
            if row.get("primitive") is None:
                state, final = _completion_checkpoint(state, row, primitive_support, resource_id,
                                                       geometry, resource_models, len(observations))
                observations.append(final)
                continue
            index, command = primitive_index, primitives[primitive_index]
            primitive_index += 1
            if any(row.get(key) != command.get(key) for key in ("primitive", "params")):
                raise ValueError("Primitive witness trace does not match the derived commands")
            rows = step_evidence or {}
            evidence = (rows[index] if index < len(rows) else {}) if isinstance(rows, list) else rows.get(index, rows.get(str(index), {}))
            if (evidence.get("validated") is not True or not evidence.get("source")
                    or any(evidence.get(key) != command.get(key) for key in ("primitive", "params"))):
                raise ValueError(f"Validated geometry evidence is missing for primitive step {index}")
            owner = state["resources"][resource_id]
            _check_projected_custody(state, row, resource_id, "start_state")
            if "joint_trajectory" in evidence:
                motion = _prepared_motion(state, resource_id, resource_models, evidence)
                observations.append(_checkpoint(state, geometry, resource_models,
                                                index=len(observations),
                                                moving=(resource_id, motion, 0, motion.duration)))
                owner["joint_positions"] = deepcopy(motion.points[-1]["positions"])
            elif "named_pose" in evidence or "target_joint_positions" in evidence:
                target = (resource_models[resource_id]["named_joint_poses"][evidence["named_pose"]]
                          if "named_pose" in evidence else evidence["target_joint_positions"])
                if target["joint_names"] != owner["joint_names"]:
                    raise ValueError("Named joint pose changes the recorded joint identities")
                owner["joint_positions"] = deepcopy(target["positions"])
                result["interval_complete"] = False
            elif evidence.get("geometry_unchanged") is not True and not evidence.get("configuration_after"):
                raise ValueError(f"Primitive step {index} has no declared geometry effect")
            _configuration_after(resource_models, resource_id, evidence)
            if evidence.get("interval_complete") is False or "configuration_after" in evidence:
                result["interval_complete"] = False
            if resource_id in resource_models and "configuration" in resource_models[resource_id]:
                _, matrix = _joint_geometry(owner, resource_models[resource_id])
                owner["current_pose"] = representative_pose(matrix)
            projected = _checkpoint(state, geometry, resource_models, index=len(observations))
            state.update(resources=deepcopy(projected["resources"]), parts=deepcopy(projected["parts"]))
            if evidence.get("custody_effects"):
                observations.append(projected)
                _custody(state, resource_id, evidence["custody_effects"], geometry)
            _check_projected_custody(state, row, resource_id, "end_state")
            final = _checkpoint(state, geometry, resource_models, index=len(observations))
            state, final = _declared_completion(state, final, row, primitive_support, resource_id, geometry)
            observations.append(final)
        _part_entries(observations)
        result.update(status="projected", feasibility_status="FEASIBLE",
                      projected_snapshot=deepcopy(state), resource_models=deepcopy(resource_models),
                      provenance={"kind": "projected", "source": "resource_primitive_witness",
                                  "model_fingerprint": primitive_support.get("model_fingerprint"),
                                  "snapshot_fingerprint": primitive_support.get("snapshot_fingerprint")})
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        result["unresolved"].append(str(exc))
        result["requires_projection"] = bool(result["observations"])
        result["context_requests"] = [] if result["requires_projection"] else ["/physical_evidence"]
    return result


def outline_event_binding(task: dict, recovery_snapshot: dict,
                          recovery_des_model: dict, evidence: dict) -> dict:
    """Bind owner evidence to conditions and the exact isolated starting scene.

    LLM-authored names and rationale are identifiers, not physical conditions.
    They therefore do not prevent reuse of evidence for identical conditions.
    """
    scene_models = {rid: {key: value for key, value in model.items()
                         if key not in {"event_projection_contracts", "named_joint_poses"}}
                    for rid, model in (evidence.get("resource_models") or {}).items()}
    # Additional saved previews and named targets do not change the physical
    # scene. The selected contract and numeric endpoint are bound separately.
    # These views are hashed synchronously and never mutated or returned.
    scene = {field: evidence.get(field) for field in ("snapshot", "geometry", "jids")}
    scene["resource_models"] = scene_models
    scene["stationary_parts"] = evidence.get("stationary_parts")
    scene["stationary_part_evidence"] = evidence.get("stationary_part_evidence")
    for field in ("physics_evidence", "physics_meshes", "part_support_policy", "environment_models", "environment_valuation",
                  "products", "requirements", "product_name", "physical_preview_contract_version",
                  "endpoint_evidence_policy_version", "endpoint_robot_description", "endpoint_occupancy"):
        if field in evidence:
            scene[field] = evidence[field]
    scene["stationary_resources"] = {rid: {field: row[field] for field in
        ("model_static", "idle", "stationary_contract", "launch_id", "controller_identity",
         "controller_hold_model", "controller_hold_observation", "controller_hold_activity",
         "projected_controller_hold", "projected_endpoint", "validated_pickup_target_calculation") if field in row}
        for rid, row in (evidence.get("resources") or {}).items()}
    return {
        "version": 1,
        "resource_jid": task.get("resource_jid"),
        "conditions_fingerprint": recovery_validation_fingerprint({
            field: task[field] for field in (
                "resource_jid", "part_name", "expected_start_state", "expected_end_state",
                "params", "required_task_contract") if field in task}),
        "snapshot_fingerprint": recovery_validation_fingerprint(recovery_snapshot),
        "model_fingerprint": recovery_validation_fingerprint(recovery_des_model),
        "scene_fingerprint": recovery_validation_fingerprint(scene),
    }


