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


def retain_part_support_geometry(evidence: dict) -> None:
    """Retain rigid geometry for later calculations, without retaining observations.

    Args:
        evidence: Attempt-local physical evidence before old observations retire.
    """
    if 'part_support_geometry' in evidence or not evidence.get('physics_evidence'):
        return
    physics = evidence['physics_evidence']
    models, parts, resources = {}, {}, {}
    for part, row in evidence.get('snapshot', {}).get('parts', {}).items():
        name = row.get('model_name')
        model = physics.get('models', {}).get(name)
        geometry = evidence.get('geometry', {}).get('parts', {}).get(part)
        if model is not None and geometry is not None and len(model.get('links', {})) == 1:
            models[name] = deepcopy(model)
            parts[part] = {'model_name': name, 'geometry_fingerprint': recovery_validation_fingerprint(geometry)}
    for name, model in physics.get('models', {}).items():
        geometry = evidence.get('geometry', {}).get('resources', {}).get(name)
        pose = evidence.get('snapshot', {}).get('resources', {}).get(name, {}).get('current_pose')
        if (model.get('static') is True and geometry is not None and isinstance(pose, list)
                and len(pose) == 7 and pose == model.get('pose')):
            models[name] = deepcopy(model)
            resources[name] = {'current_pose': deepcopy(pose),
                'geometry_fingerprint': recovery_validation_fingerprint(geometry)}
    for model in models.values():
        for link in model.get('links', {}).values():
            for key in ('linear_velocity', 'angular_velocity', 'linear_speed', 'angular_speed', 'enabled'):
                link.pop(key, None)
    body = {'version': 1, 'physics': {key: deepcopy(physics[key]) for key in
        ('version', 'instance_id', 'simulation_time', 'physics_engine', 'physics_configuration',
         'gravity', 'contact_surface_layer') if key in physics},
        'parts': parts, 'resources': resources,
        'physics_source_fingerprint': recovery_validation_fingerprint(physics),
        'policy_fingerprint': recovery_validation_fingerprint(evidence.get('part_support_policy')),
        'meshes_fingerprint': recovery_validation_fingerprint(evidence.get('physics_meshes'))}
    body['physics'].update(models=models, contacts=[])
    body['fingerprint'] = recovery_validation_fingerprint(body)
    evidence['part_support_geometry'] = body


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


def project_outline_evidence(task: dict, primitive_support: dict, evidence: dict) -> dict:
    """Adapt a detached physical capture using its exact resource JID mapping."""
    try:
        jid = task["resource_jid"]
        matches = [rid for rid, value in evidence["jids"].items() if value == jid]
        if len(matches) != 1:
            raise ValueError("The candidate resource has no unique captured identity")
        snapshot = deepcopy(evidence.get("snapshot") or {
            "frame": evidence.get("frame"), "parts": evidence["parts"],
            "resources": {rid: row.get("observation_state", row)
                          for rid, row in evidence["resources"].items()},
        })
        if set(snapshot["resources"]) != set(evidence["jids"]):
            raise ValueError("Physical snapshot must retain every captured resource identity")
        result = project_recovery_geometry(
            resource_id=matches[0], snapshot=snapshot, geometry=evidence["geometry"],
            resource_models=evidence.get("resource_models", {}), primitive_support=primitive_support,
            step_evidence=evidence.get("step_evidence", {}))
        if result["status"] == "projected":
            successor = deepcopy(evidence)
            successor["snapshot"] = deepcopy(result["projected_snapshot"])
            successor["resource_models"] = deepcopy(result["resource_models"])
            successor["provenance"] = deepcopy(result["provenance"])
            successor["step_evidence"] = {}
            result["successor_evidence"] = successor
        return result
    except (KeyError, TypeError, ValueError) as exc:
        return {"status": "needs_context", "feasibility_status": "NEEDS_CONTEXT",
                "unresolved": [str(exc)], "execution_authorized": False,
                "context_requests": ["/physical_evidence"], "requires_projection": False}


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


def _event_effects(updates: dict, bindings: dict, declarations: dict, scope: str) -> dict:
    values = {}
    if not isinstance(updates, dict):
        raise ValueError("Owner event updates must be a mapping")
    for field, effect in updates.items():
        declaration = declarations.get(field)
        if (not isinstance(declaration, dict) or declaration.get("private") is True
                or declaration.get("scope", "resource") != scope):
            raise ValueError(f"Owner event effect {field} requires its public {scope} declaration")
        if not isinstance(effect, dict):
            raise ValueError(f"Unsupported owner event effect for {field}")
        if set(effect) == {"set"}:
            values[field] = deepcopy(effect["set"])
        elif set(effect) == {"set_from_param"} and effect["set_from_param"] in bindings:
            values[field] = deepcopy(bindings[effect["set_from_param"]])
        else:
            raise ValueError(f"Unresolved owner event effect for {field}")
    return values


def rejected_owner_event_effects(*, task: dict, recovery_snapshot: dict,
                                recovery_des_model: dict, physical_evidence: dict) -> dict | None:
    """Reject only contradictions shared by every declared successor alternative.

    This checks declared public effects, without ordering primitives or claiming
    motion clearance. Unknown effects and incomplete alternatives remain unresolved.
    """
    from cais_spade_llm.resources.recovery_feasibility import derive_primitive_union

    jids, snapshot = physical_evidence.get("jids"), physical_evidence.get("snapshot")
    if (not isinstance(jids, dict) or not isinstance(snapshot, dict)
            or not isinstance(snapshot.get("resources"), dict)
            or not isinstance(task.get("resource_jid"), str) or not task["resource_jid"]):
        return None
    owners = [rid for rid, jid in jids.items() if jid == task["resource_jid"]]
    if (len(owners) != 1 or owners[0] not in snapshot["resources"]
            or recovery_des_model.get("resource_jid", task.get("resource_jid")) != task.get("resource_jid")):
        return None
    support = derive_primitive_union(task=task, recovery_des_model=recovery_des_model,
                                    recovery_snapshot=recovery_snapshot)
    alternatives = support["matching_transitions"]
    if task.get("required_task_contract") is not None:
        alternatives = [row for row in alternatives if row["event_name"] == task["required_task_contract"]]
    if (support["model_complete"] is not True or support["missing_support"]
            or not alternatives or len(alternatives) > 4096):
        return None
    declarations = recovery_des_model["state_variables"]
    # Successor matching can discard an alternative at its first unequal
    # resource value. Unsupported expressions cannot establish that mismatch
    # for an exhaustive owner-effect contradiction.
    for event in recovery_des_model["events"]:
        for field, effect in event["updates"].items():
            declaration = declarations.get(field)
            if not isinstance(declaration, dict):
                return None
            if declaration.get("private") is True or declaration.get("scope", "resource") == "part":
                continue
            if not isinstance(effect, dict):
                return None
            if set(effect) == {"set"}:
                value = effect["set"]
                if (value is not None and type(value) not in (str, bool, int, float)
                        or type(value) is float and not math.isfinite(value)):
                    return None
            elif (set(effect) != {"set_from_param"}
                  or not isinstance(effect["set_from_param"], str) or not effect["set_from_param"]):
                return None
    conflicts, checked = [], []
    for alternative in alternatives:
        events = [event for event in recovery_des_model["events"]
                  if event["event_name"] == alternative["event_name"]]
        if len(events) != 1:
            return None
        event, bindings = events[0], alternative["parameter_bindings"]
        contradictions = []
        for field, requested in task["expected_end_state"].items():
            declaration = declarations.get(field)
            if not isinstance(declaration, dict) or declaration.get("private") is True:
                continue
            if declaration.get("scope", "resource") == "part":
                contract = event.get("composition_contract")
                transition = contract.get("part_transition") if isinstance(contract, dict) else None
                completed = transition.get("completed") if isinstance(transition, dict) else None
                # A receiving resource or camera may establish a different
                # product state after release. Only the declared carried-part
                # completion is a final owner value at this boundary.
                if (not isinstance(completed, dict) or set(completed) != {"state", "location_template"}
                        or completed["location_template"] != "{resource_jid}"
                        or field not in {"part_state", "part_location"}):
                    continue
                completed_value = completed["state"] if field == "part_state" else task["resource_jid"]
            effect = event["updates"].get(field)
            if not isinstance(effect, dict):
                continue
            if set(effect) == {"set"}:
                derived = effect["set"]
            elif (set(effect) == {"set_from_param"} and isinstance(effect["set_from_param"], str)
                  and effect["set_from_param"] in bindings):
                derived = bindings[effect["set_from_param"]]
            else:
                continue
            if (any(value is not None and type(value) not in (str, bool, int, float)
                    or type(value) is float and not math.isfinite(value) for value in (derived, requested))
                    or declaration.get("domain_complete") is True and not any(
                        type(derived) is type(value) and derived == value for value in declaration["domain"])):
                continue
            if (declaration.get("scope", "resource") == "part"
                    and (type(derived) is not type(completed_value) or derived != completed_value)):
                continue
            if type(derived) is not type(requested) or derived != requested:
                contradictions.append({"field": field, "requested": deepcopy(requested),
                                       "derived": deepcopy(derived)})
        if not contradictions:
            return None
        conflicts.extend(contradictions)
        checked.append({"event_name": alternative["event_name"],
            "declaration_fingerprint": recovery_validation_fingerprint(event),
            "parameter_bindings": deepcopy(bindings), "conflicts": contradictions})
    reason = "Every declared resource successor contradicts a proposed public end condition"
    return {"status": "rejected", "validation_status": "rejected", "owner_effect_rejected": True,
        "owner_effect_result": {"feasibility_status": "INFEASIBLE", "reason": reason,
            "conflicts": conflicts, "alternatives": checked, "alternatives_complete": True,
            "model_fingerprint": recovery_validation_fingerprint(recovery_des_model)},
        "event_record_binding": outline_event_binding(task, recovery_snapshot, recovery_des_model, physical_evidence),
        "task": deepcopy(task), "reason": reason, "observations": [], "interval_complete": False,
        "execution_authorized": False, "command_sent": False}


def _event_endpoint(contract: dict, bindings: dict, state: dict, rid: str, part: str | None,
                    intervals: list | None = None) -> dict:
    endpoint = contract.get("endpoint")
    if not isinstance(endpoint, dict):
        raise ValueError("Owner event projection requires an endpoint declaration")
    supported = {"joint_positions", "current_pose", "part_pose"}
    if set(endpoint) - supported:
        raise ValueError("Unsupported owner event endpoint condition")
    numerical = contract.get("endpoint_numerical_precision", {})
    if not isinstance(numerical, dict) or set(numerical) - set(endpoint) or set(numerical) - {"current_pose"}:
        raise ValueError("Unsupported owner numerical endpoint conditions")
    checks = {}
    for field, expression in endpoint.items():
        if not isinstance(expression, dict):
            raise ValueError("Owner event endpoint requires an explicit value or binding")
        if set(expression) == {"set"}:
            expected = expression["set"]
        elif set(expression) == {"set_from_param"} and expression["set_from_param"] in bindings:
            expected = bindings[expression["set_from_param"]]
        else:
            raise ValueError("Owner event endpoint binding is unavailable")
        actual = state["parts"][part]["current_pose"] if field == "part_pose" else state["resources"][rid][field]
        precision = numerical.get(field)
        if precision is not None:
            if (field != "current_pose" or precision != owner_cartesian_endpoint_precision()
                    or expression != {"set_from_param": "target_pose"}):
                raise ValueError("Unsupported owner numerical endpoint contract")
            motions = [row for row in intervals or [] if "joint_trajectory" in row]
            proof = motions[-1].get("native_endpoint_evidence", {}) if motions else {}
            if (not motions or proof.get("target_pose") != expected or proof.get("trajectory_fingerprint") !=
                    recovery_validation_fingerprint(motions[-1]["joint_trajectory"])):
                raise ValueError("Owner numerical endpoint lacks its exact native motion evidence")
            for kind in ("position", "orientation"):
                limit, residual = proof.get(f"cartesian_{kind}_precision"), proof.get(f"cartesian_{kind}_residual")
                if (limit != precision[f"{kind}_precision"] or type(residual) not in (int, float)
                        or not math.isfinite(residual) or not 0 <= residual <= limit):
                    raise ValueError("Native endpoint numerical precision is unavailable or incompatible")
            residual = _cartesian_endpoint_residual(actual, expected)
            if any(residual[f"{kind}_residual"] > precision[f"{kind}_precision"]
                   for kind in ("position", "orientation")):
                raise ValueError(f"Projected owner event does not establish {field} at its numerical precision")
            checks[field] = {"numerical_precision": deepcopy(precision), **residual,
                "actual": deepcopy(actual), "target": deepcopy(expected),
                "native_endpoint_evidence": deepcopy(proof)}
            continue
        tolerance = contract.get("endpoint_tolerance", {}).get(field, 0.)
        if type(tolerance) not in (int, float) or not math.isfinite(tolerance) or tolerance < 0:
            raise ValueError("Owner event endpoint tolerance must be declared and finite")
        if (not isinstance(expected, list) or len(expected) != len(actual)
                or any(type(value) not in (float, int) or not math.isfinite(value) for value in expected)
                or any(abs(value - target) > tolerance for value, target in zip(actual, expected, strict=True))):
            raise ValueError(f"Projected owner event does not establish {field}")
    return checks


def _owner_product_custody(*, task: dict, state: dict, evidence: dict,
                           resource_updates: dict, support: dict, rid: str) -> tuple[dict, dict, dict]:
    """Use established owner custody with the existing runtime product meaning."""
    from cais_spade_llm.resources.environment_models import product_state_from_custody

    part = task.get("part_name")
    models, valuations = evidence.get("environment_models", {}), deepcopy(evidence.get("environment_valuation", {}))
    if not part or rid not in models or rid not in valuations:
        return {}, valuations, {}
    values = valuations[rid]
    values.update({key: deepcopy(value) for key, value in resource_updates.items()
                   if key in models[rid].get("state_variables", {})})
    if state["resources"][rid].get("held_part") == part:
        if "held_part" not in models[rid].get("state_variables", {}) or values.get("held_part") != part:
            return {}, valuations, {}
        owner = rid
        prior_owner = evidence.get("snapshot", {}).get("parts", {}).get(part, {}).get("contained_by")
        if prior_owner in models and prior_owner in valuations:
            prior_fields = models[prior_owner].get("state_variables", {})
            occupied = [field for field, value in valuations[prior_owner].items()
                        if value == part and prior_fields.get(field, {}).get("reference") == "part_name"]
            if len(occupied) != 1 or "resource_state" in prior_fields:
                raise ValueError("Pickup needs the source owner's complete declared occupancy update")
            valuations[prior_owner][occupied[0]] = None
    else:
        owner = state["parts"][part].get("contained_by")
        if owner not in models or owner not in valuations or part not in support:
            return {}, valuations, {}
        receiver = evidence.get("resources", {}).get(owner, {})
        source_model = receiver.get("geometry_source", {}).get("model")
        supported_models = {name for name, model in evidence.get("physics_evidence", {}).get("models", {}).items()
            for link in model.get("links", {}).values() for collision in link.get("collisions", [])
            if collision.get("id") in support[part].get("support_collisions", [])}
        if (receiver.get("idle") is not True or receiver.get("model_static") is not True
                or receiver.get("stationary_contract", {}).get("kind") != "static_body_and_idle_containment"
                or supported_models != {source_model}):
            return {}, valuations, {}
        values = valuations[owner]
        reference_fields = {field for field, declaration in models[owner].get("state_variables", {}).items()
                            if declaration.get("reference") == "part_name"}
        occupied = [field for field in reference_fields if values.get(field) == part]
        if not occupied:
            pose = state["parts"][part]["current_pose"]
            zones = [zone for zone in models[owner].get("assignments", {}).get("zones", [])
                     if isinstance(zone.get("pose"), list) and len(zone["pose"]) >= 3
                     and all(math.isclose(pose[index], zone["pose"][index], rel_tol=0., abs_tol=1e-12)
                             for index in range(3))]
            if len(zones) != 1:
                return {}, valuations, {}
            field = f"zone_{zones[0]['zone']}_part"
            if field not in reference_fields or field not in values or values[field] is not None:
                return {}, valuations, {}
            values[field] = part
            occupied = [field]
        if len(occupied) != 1:
            raise ValueError("Projected product custody requires one declared receiving occupancy")
    product = product_state_from_custody(part, owner, values, evidence.get("product_name"))
    return {"part_state": product["state"], "part_location": product["location"]}, valuations, {
        "kind": "projected", "source": "cais_spade_llm.resources.environment_models.product_state_from_custody",
        "resource_id": owner, "resource_model_fingerprint": recovery_validation_fingerprint(models[owner]),
        "resource_values": deepcopy(values), "physical_custody": deepcopy(state["parts"][part].get("contained_by"))}


def _owner_product_completion(*, task: dict, state: dict, geometry: dict, evidence: dict,
                              contract: dict, rid: str, support: dict) -> dict | None:
    """Bind a declared product effect to its supported deposited target, privately."""
    from cais_spade_llm.agents.central_controller._product_effect_evidence import _declared_product_effects
    from cais_spade_llm.resources.environment_models import _check_product_requirements

    part = task.get("part_name")
    if part not in support or state["resources"][rid].get("held_part") == part:
        return None
    models = evidence.get("environment_models", {})
    if rid not in models or part not in geometry.get("parts", {}):
        return None
    target = contract.get("placement_target", {})
    origin = target.get("target_origin_pose", {})
    pose = state["parts"][part]["current_pose"]
    at_target = all(key in origin and math.isclose(pose[index], origin[key], rel_tol=0., abs_tol=1e-12)
                    for index, key in enumerate(("x", "y", "z")))
    deposited = support[part]
    if contract.get("version") in {6, 7} and deposited.get("kind") == "projected_fixture_handoff":
        # The owner preserves the seated pose; fixture attachment does not snap
        # the measured or calculated coordinates to the nominal target.
        policy = target.get("simulation_mating_contact") or {}
        at_target = (deposited.get("status") == "supported" and deposited.get("seated") is True
            and deposited.get("retained_fixture") is True and deposited.get("part_name") == part
            and deposited.get("current_pose") == pose and deposited.get("target_origin_pose") == origin
            and deposited.get("target_collision_object") == policy.get("target_collision_object")
            and deposited.get("policy_fingerprint") == recovery_validation_fingerprint(policy))
    if (target.get("status") != "calculated" or target.get("target_reference", {}).get("surface_role") != "assembly_slot"
            or target.get("target_reference", {}).get("target_point") != "inserted_part_origin"
            or not at_target):
        return None
    destination = target.get("destination_location")
    configured_target = geometry["parts"][part].get("target")
    if target.get("target") != configured_target or not configured_target:
        return None
    functions = {row["event_name"] for row in contract["owner_declarations"]}
    matches = []
    for event in models[rid].get("events", []):
        if event.get("function_name") not in functions or not event.get("product_effects", {}).get("processCompleted"):
            continue
        params = {}
        for name, binding in event.get("parameter_bindings", {}).items():
            if "equals" in binding:
                params[name] = deepcopy(binding["equals"])
            elif name == "part_name":
                params[name] = part
            elif name == "target":
                params[name] = configured_target
            else:
                break
        else:
            if params.get("resource_id") != rid or params.get("destination_location") != destination:
                continue
            declaration = {"event_id": event["event_id"], "event_name": event["event_name"], "parameters": params}
            effects = _declared_product_effects(models, state["parts"], geometry, rid, declaration)
            if not isinstance(evidence.get("requirements"), dict) or not evidence["requirements"]:
                raise ValueError("Declared completion requires the captured product requirements")
            _check_product_requirements({**declaration, "resource_id": rid}, state["parts"],
                                        evidence["product_name"], evidence["requirements"], event)
            location_effect = event.get("updates", {}).get("part_location")
            if location_effect == {"set": destination} or location_effect == {"set_from_param": "destination_location"}:
                location = destination
            else:
                raise ValueError("Declared completion has no supported product location effect")
            matches.append({"declaration": declaration, "product_effects": effects,
                "declaration_fingerprint": recovery_validation_fingerprint(event), "resource_id": rid,
                "part_location": location, "kind": "predicted", "source_kind": "resource_event_projection"})
    if len(matches) > 1:
        raise ValueError("More than one declared completion effect matches the deposited target")
    return matches[0] if matches else None


def _event_clock_checkpoint(checkpoint: dict, time: Fraction) -> None:
    """Bind a newly derived checkpoint to the native trajectory clock."""
    checkpoint.update(time=float(time), time_exact=str(time))


def _event_clock_motion(checkpoint: dict, start: Fraction) -> Fraction:
    """Offset a verified interval without rounding its recorded duration."""
    interval_start, interval_end = map(Fraction, checkpoint["continuous_interval"])
    if interval_start != 0 or interval_end <= 0:
        raise ValueError("Owner trajectory requires a positive exact duration from zero")
    _event_clock_checkpoint(checkpoint, start)
    checkpoint["continuous_interval"] = [str(start), str(start + interval_end)]
    for cell in checkpoint["continuous_cells"]:
        cell["interval"] = [str(start + Fraction(value)) for value in cell["interval"]]
    return start + interval_end


def _event_custody_source(interval_record: dict, declared_events: list, record: dict) -> dict:
    operations = {"grasp": "grasp_part", "release": "release_part"}
    ops = {operations[effect["kind"]] for effect in interval_record["custody_effects"]}
    if len(ops) != 1:
        raise ValueError("Physical custody requires one exact declared operation")
    op = next(iter(ops))
    matches = {}
    for event, _ in declared_events:
        steps = [step for step in event.get("composition_contract", {}).get("steps", [])
                 if step.get("op") == op]
        if len(steps) != 1:
            raise ValueError("Physical custody requires its unambiguous declared owner step")
        matches[recovery_validation_fingerprint(steps[0])] = steps[0]
    if len(matches) != 1:
        raise ValueError("Supporting owner declarations disagree on the custody step")
    return {"step_fingerprint": next(iter(matches)), "op": op,
            "source_fingerprint": recovery_validation_fingerprint(record)}


def _event_record_projection(*, task: dict, recovery_snapshot: dict,
                             recovery_des_model: dict, evidence: dict, record: dict, rid: str,
                             resource_evidence: dict | None = None) -> dict:
    if any(field in record for field in ("primitives", "primitive_support", "trace")):
        raise ValueError("Event geometry evidence cannot contain an inferred primitive program")
    if not isinstance(record.get("source"), str) or not record["source"]:
        raise ValueError("Owner event evidence has no source")
    models = deepcopy(evidence["resource_models"])
    contracts = models[rid].get("event_projection_contracts", {})
    contract = contracts.get(record.get("contract_id"))
    if (not isinstance(contract, dict) or contract.get("version") not in {1, 2, 3, 4, 5, 6, 7}
            or record.get("contract_fingerprint") != recovery_validation_fingerprint(contract)):
        raise ValueError("Owner event projection contract is unavailable or changed")
    required_version = evidence.get("physical_preview_contract_version")
    if required_version is not None and contract["version"] != required_version:
        raise ValueError("Owner event projection contract does not match this attempt's required version")
    nominal_owner_evidence = None
    if contract["version"] == 7 and contract.get("task_contract") is not None:
        from cais_spade_llm.resources.recovery_parameter_evidence import projected_resource_evidence

        if not isinstance(resource_evidence, dict):
            raise ValueError("Nominal owner program requires authoritative resource_evidence")
        nominal_owner_evidence = projected_resource_evidence(resource_jid=task["resource_jid"],
            resource_evidence=resource_evidence, physical_evidence=evidence,
            recovery_snapshot=recovery_snapshot)
        recorded_owner_evidence = contract.get("nominal_owner_evidence")
        if (not isinstance(recorded_owner_evidence, dict)
                or recovery_validation_fingerprint(recorded_owner_evidence)
                != recovery_validation_fingerprint(nominal_owner_evidence)):
            raise ValueError("Nominal owner program calculation context differs from authoritative resource_evidence")
    motion_capabilities = None
    if contract["version"] in {4, 5, 6, 7}:
        from cais_spade_llm.resources.resource_safety_preparation import validate_outline_motion_capabilities

        motion_capabilities = validate_outline_motion_capabilities(contract=contract, model=models[rid],
            resource_jid=task["resource_jid"])
    owner_declarations = contract.get("owner_declarations")
    if not isinstance(owner_declarations, list) or not owner_declarations:
        raise ValueError("Event projection requires its recorded supporting owner declarations")
    declared_events = []
    for declaration in owner_declarations:
        recorded = [row for row in recovery_des_model.get("events", [])
                    if row.get("event_name") == declaration.get("event_name")]
        if len(recorded) != 1 or recovery_validation_fingerprint(recorded[0]) != declaration.get("fingerprint"):
            raise ValueError("Event projection owner declaration changed after target preparation")
        declared_events.append((recorded[0], declaration.get("parameter_bindings") or {}))
    bindings = record.get("bindings")
    params = contract.get("params")
    if not isinstance(bindings, dict) or not isinstance(params, dict) or set(bindings) != set(params):
        raise ValueError("Owner event parameter bindings differ from their declaration")
    for name, value in bindings.items():
        declaration = params[name]
        if not isinstance(declaration, dict):
            raise ValueError("Owner event parameter declaration is unavailable")
        if "const" in declaration and (type(value) is not type(declaration["const"]) or value != declaration["const"]):
            raise ValueError("Owner event parameter differs from its declared value")
        if name in (task.get("params") or {}) and value != task["params"][name]:
            raise ValueError("Owner event changes the proposal parameter binding")
    declarations = recovery_des_model.get("state_variables") or {}
    private_updates = contract.get("private_resource_updates", {})
    if not isinstance(private_updates, dict):
        raise ValueError("Owner private projected effects require a mapping")
    for field, value in private_updates.items():
        if declarations.get(field, {}).get("private") is not True:
            raise ValueError("Owner private projection requires its declared private field")
        for event, event_bindings in declared_events:
            effect = event.get("updates", {}).get(field)
            if not isinstance(effect, dict):
                raise ValueError(f"Owner private effect {field} is not declared")
            if set(effect) == {"set"}:
                supported = effect["set"]
            elif set(effect) == {"set_from_param"} and effect["set_from_param"] in event_bindings:
                supported = event_bindings[effect["set_from_param"]]
            else:
                raise ValueError(f"Owner private effect {field} has unresolved semantics")
            if type(value) is not type(supported) or value != supported:
                raise ValueError(f"Owner private effect {field} differs from its declaration")
    resource_updates = _event_effects(contract.get("resource_updates", {}), bindings, declarations, "resource")
    part_updates = _event_effects(contract.get("part_updates", {}), bindings, declarations, "part")
    if contract.get("carried_part_transition") is True:
        for event, event_bindings in declared_events:
            completed = event.get("composition_contract", {}).get("part_transition", {}).get("completed", {})
            if (set(completed) != {"state", "location_template"}
                    or completed["location_template"] != "{resource_jid}"
                    or part_updates != {"part_state": completed["state"], "part_location": task["resource_jid"]}):
                raise ValueError("Carried-part projection differs from its owner completion declaration")
            for field, value in part_updates.items():
                declared = _event_effects({field: event.get("updates", {}).get(field)}, event_bindings, declarations, "part")
                if type(declared[field]) is not type(value) or declared[field] != value:
                    raise ValueError("Carried-part projection differs from its declared effect")
    for field, value in resource_updates.items():
        for event, event_bindings in declared_events:
            effect = event.get("updates", {}).get(field)
            if effect is None:
                keys = {"resource_state": ("resource_state", "current_state"),
                        "resource_location": ("resource_location", "current_location")}.get(field, (field,))
                known = [key for key in keys if key in recovery_snapshot]
                if not known:
                    raise ValueError(f"Preserved event condition {field} has no recorded value")
                supported = recovery_snapshot[known[0]]
            else:
                supported = _event_effects({field: effect}, event_bindings, declarations, "resource")[field]
            if type(value) is not type(supported) or value != supported:
                raise ValueError(f"Event effect {field} differs from the supporting owner declaration")
    expected = task["expected_end_state"]
    required_resource_fields = {field for field in expected
        if declarations.get(field, {}).get("scope", "resource") == "resource"}
    if required_resource_fields - set(resource_updates):
        raise ValueError("Owner event projection leaves resource end conditions unsupported: "
                         + ", ".join(sorted(required_resource_fields - set(resource_updates))))
    for field, value in {**resource_updates, **part_updates}.items():
        if field in expected and (type(value) is not type(expected[field]) or value != expected[field]):
            raise ValueError(f"Owner event effect conflicts with proposed {field}")
    if "processCompleted" in part_updates:
        raise ValueError("Process completion requires a separately declared completion ledger witness")
    state, geometry = deepcopy(evidence["snapshot"]), deepcopy(evidence["geometry"])
    if set(state["resources"]) != set(evidence["jids"]):
        raise ValueError("Event scene must retain every recorded resource identity")
    intervals = record.get("intervals")
    if not isinstance(intervals, list) or not intervals or len(intervals) > 4096:
        raise ValueError("Owner event requires bounded complete physical intervals")
    pickup_departure = contract.get("pickup_departure")
    grasp_indices = [index for index, row in enumerate(intervals)
                     if any(effect.get("kind") == "grasp" for effect in row.get("custody_effects", []))]
    if contract["version"] >= 2 and grasp_indices:
        if len(grasp_indices) != 1 or not isinstance(pickup_departure, dict):
            raise ValueError("A pickup event requires its declared post-grasp departure")
        for event, _ in declared_events:
            steps = event.get("composition_contract", {}).get("steps") or []
            matches = [step for step in steps if step.get("id") == pickup_departure.get("step_id")]
            if (len(matches) != 1 or matches[0].get("op") != pickup_departure.get("op")
                    or recovery_validation_fingerprint(matches[0]) != pickup_departure.get("step_fingerprint")):
                raise ValueError("Pickup departure differs from the captured owner declaration")
        departing = [index for index, row in enumerate(intervals) if "support_departure" in row]
        if (len(departing) != 1 or departing[0] <= grasp_indices[0]
                or "joint_trajectory" not in intervals[departing[0]] or departing[0] != len(intervals) - 1):
            raise ValueError("Pickup departure must follow acquired custody and establish the event endpoint")
    elif contract["version"] >= 2 and pickup_departure is not None:
        raise ValueError("Pickup departure has no corresponding acquired custody")
    placement_departure = contract.get("placement_departure")
    release_indices = [index for index, row in enumerate(intervals)
                       if any(effect.get("kind") == "release" for effect in row.get("custody_effects", []))]
    snap_indices = [index for index, row in enumerate(intervals) if "nominal_snap_evidence" in row]
    expected_snap = contract["version"] == 7 and contract.get("nominal_snap") is not None
    if expected_snap:
        if (len(release_indices) != 1 or snap_indices != [release_indices[0] + 1]
                or snap_indices[0] != len(intervals) - 2):
            raise ValueError("Nominal simulation correction must follow release and precede departure")
    elif snap_indices:
        raise ValueError("Simulation correction has no exact nominal owner declaration")
    if contract["version"] >= 3 and release_indices:
        if (len(release_indices) != 1 or not isinstance(placement_departure, dict)
                or release_indices[0] != len(intervals) - (3 if expected_snap else 2)
                or "joint_trajectory" not in intervals[-1]):
            raise ValueError("A placement event requires its declared motion after release")
        for event, _ in declared_events:
            steps = event.get("composition_contract", {}).get("steps") or []
            if (not steps or not any(step.get("op") == "release_part" for step in steps[:-1])
                    or steps[-1].get("id") != placement_departure.get("step_id")
                    or steps[-1].get("op") != placement_departure.get("op")
                    or recovery_validation_fingerprint(steps[-1]) != placement_departure.get("step_fingerprint")):
                raise ValueError("Placement departure differs from the captured owner declaration")
        target = contract.get("placement_target", {})
        context, source = target.get("departure_context", {}), target.get("departure_context_source", {})
        nominal_context = source.get("kind") == "nominal_task"
        if (not nominal_context and context.get("part_name") != task.get("part_name")
                or source != placement_departure.get("context_source")):
            raise ValueError("Placement departure context does not identify this part and source")
        if nominal_context:
            if (contract["version"] != 7 or not task.get("required_task_contract")
                    or not isinstance(nominal_owner_evidence, dict)
                    or recovery_validation_fingerprint(context) != recovery_validation_fingerprint(
                        nominal_owner_evidence.get("state", {}).get("_task_ctx"))
                    or source != {"kind": "nominal_task", "task_contract": task["required_task_contract"],
                        "context_fingerprint": recovery_validation_fingerprint(context)}):
                raise ValueError("Nominal departure differs from its authoritative task context")
        elif source.get("kind") == "projected":
            captured = evidence.get("resources", {}).get(rid, {}).get("validated_pickup_target_calculation", {})
            prior = models[rid].get("event_projection_contracts", {}).get(captured.get("contract_id"), {})
            calculation = captured.get("calculation", {})
            if (captured.get("source") != "resource-owned pickup preview"
                    or captured.get("physical_execution_verified") is not False
                    or captured.get("binding", {}).get("resource_jid") != task["resource_jid"]
                    or captured.get("part_name") != task.get("part_name")
                    or calculation.get("part_name") != task.get("part_name")
                    or captured.get("contract_fingerprint") != recovery_validation_fingerprint(prior)
                    or prior.get("pickup_target_calculation") != calculation
                    or context != {"part_name": task.get("part_name"), "travel_z": calculation.get("travel_z")}
                    or source != {"kind": "projected", "contract_id": captured.get("contract_id"),
                        "contract_fingerprint": captured.get("contract_fingerprint"), "binding": captured.get("binding"),
                        "calculation": calculation.get("source")}):
                raise ValueError("Placement departure differs from this owner's validated pickup evidence")
        elif source != {"kind": "recorded", "ref": "/resources/" + task["resource_jid"]
                       + "/resource_evidence/state/_task_ctx/travel_z"}:
            raise ValueError("Placement departure has no supported context source")
    elif contract["version"] >= 3 and placement_departure is not None:
        raise ValueError("Placement departure has no corresponding released custody")
    clock_version = evidence.get("owner_projection_clock_version")
    if clock_version is not None and (type(clock_version) is not int or clock_version != 1):
        raise ValueError("Unsupported owner projection clock version")
    exact_clock = clock_version == 1
    clock = Fraction(0)
    custody_source = None
    observations = [_checkpoint(state, geometry, models, index=0)]

    def append_checkpoint(checkpoint: dict) -> None:
        if exact_clock:
            _event_clock_checkpoint(checkpoint, clock)
            if custody_source is not None:
                fields = ("held_part", "gripper_state")
                if rid in evidence.get("environment_models", {}):
                    fields = tuple(field for field in fields
                        if field in evidence["environment_models"][rid].get("state_variables", {}))
                checkpoint["owner_safety_evidence"] = {
                    "physical_state_updates": {rid: {field: deepcopy(checkpoint["resources"][rid][field])
                        for field in fields}},
                    "physical_state_sources": {rid: deepcopy(custody_source)}}
        observations.append(checkpoint)
    state.update(resources=deepcopy(observations[-1]["resources"]), parts=deepcopy(observations[-1]["parts"]))
    interval_complete = True
    projected_support = {}
    stationary_support = deepcopy(evidence.get("stationary_part_evidence"))
    motion_index, native_source = 0, None
    routes = contract.get("cartesian_routes", {})
    if not isinstance(routes, dict) or (routes and contract["version"] not in {5, 6, 7}):
        raise ValueError("Calculated Cartesian routes require version-five evidence")
    checked_routes = set()
    last_mating = None
    nominal_interval_states = []
    for interval_record in intervals:
        if contract["version"] == 7 and contract.get("task_contract") is not None:
            nominal_interval_states.append(deepcopy(state["resources"][rid]))
        if not isinstance(interval_record, dict):
            raise ValueError("Owner event interval evidence must be a mapping")
        peer_holds = interval_record.get("stationary_resource_evidence", {})
        required_holds = {peer for peer, observed in evidence.get("resources", {}).items()
            if peer != rid and observed.get("model_static") is not True
            and ("controller_hold_model" in observed or observed.get("idle") is not True)}
        if set(peer_holds) != required_holds:
            raise ValueError("Conditional peer hold evidence is incomplete")
        for peer, hold in peer_holds.items():
            derived = commanded_hold_evidence(resource_id=peer, state=state["resources"][peer],
                model=models[peer], physical=evidence["resources"][peer])
            if hold != derived or derived.get("status") != "supported":
                raise ValueError("Saved peer hold differs from its recorded source evidence: " + peer)
        if "joint_trajectory" in interval_record:
            native = interval_record.get("native_preview_binding")
            if required_version in {2, 3, 4, 5, 6, 7} or native is not None:
                if (not isinstance(native, dict) or native.get("version") != 1
                        or native.get("source") != "PreviewRecoveryMotion"
                        or type(native.get("stage_index")) is not int or native["stage_index"] != motion_index
                        or native.get("resource_id") != rid
                        or native.get("proposal_fingerprint") != recovery_validation_fingerprint(task)
                        or native.get("stage_state_fingerprint") != recovery_validation_fingerprint(state)
                        or native.get("trajectory_fingerprint") != recovery_validation_fingerprint(interval_record["joint_trajectory"])
                        or native.get("command_sent") is not False or native.get("execution_authorized") is not False
                        or any(not isinstance(native.get(field), str) or not native[field] for field in (
                            "request_fingerprint", "response_fingerprint", "capture_fingerprint", "scene_fingerprint", "model_fingerprint"))):
                    raise ValueError("Owner motion lacks matching issued native preview evidence")
                source = {field: native[field] for field in ("resource_id", "capture_fingerprint", "proposal_fingerprint", "model_fingerprint")}
                if native_source is not None and source != native_source:
                    raise ValueError("Owner motion stages use different native capture or model evidence")
                native_source = source
            if motion_capabilities is not None:
                requests = motion_capabilities["motion_requests"]
                if (motion_index >= len(requests) or interval_record.get("motion_request") != requests[motion_index]
                        or native.get("motion_request_fingerprint") != recovery_validation_fingerprint(requests[motion_index])):
                    raise ValueError("Cached owner motion differs from its capability-checked targets")
                if str(motion_index) in routes:
                    from cais_spade_llm.resources.resource_safety_preparation import validate_outline_cartesian_route

                    if (grasp_indices and interval_record is intervals[grasp_indices[0] + 1]
                            or release_indices and interval_record is intervals[release_indices[0] + 1]):
                        raise ValueError("Calculated routes cannot replace declared custody departures")
                    validate_outline_cartesian_route(route=routes[str(motion_index)], state=state,
                        native=native, motion_request=requests[motion_index], static_capabilities=models[rid]["static_capabilities"])
                    checked_routes.add(str(motion_index))
            motion_index += 1
            if isinstance(stationary_support, dict):
                free_parts = set(state["parts"]) - {row.get("held_part") for row in state["resources"].values()}
                missing_support = [name for name in free_parts if stationary_support.get(name, {}).get("status") != "supported"
                    or stationary_support[name].get("current_pose") != state["parts"][name].get("current_pose")]
                if missing_support:
                    raise ValueError("Motion requires matching stationary part evidence: " + ", ".join(sorted(missing_support)))
            if "support_departure" in interval_record:
                from cais_spade_llm.resources.recovery_support_departure import (
                    validate_support_departure_samples, verify_support_departure,
                )

                validate_support_departure_samples(interval_record)
                verified = verify_support_departure(resource_id=rid, state=state,
                    evidence={**evidence, "resource_models": models},
                    joint_trajectory=interval_record["joint_trajectory"],
                    support_departure=interval_record["support_departure"])
                if (verified.get("status") != "supported"
                        or interval_record.get("support_departure_evidence") != verified):
                    raise ValueError("Pickup departure lacks matching continuous support evidence: "
                                     + str(verified.get("reason", "record changed")))
            expected_mating = (contract["version"] in {6, 7}
                and motion_index - 1 in contract.get("placement_arrival", {}).get("mating_motion_indices", [])
                and contract.get("placement_target", {}).get("simulation_mating_contact") is not None)
            if expected_mating != ("mating_contact" in interval_record):
                raise ValueError("Owner insertion interval lacks its exact recorded mating evidence")
            if expected_mating:
                from cais_spade_llm.resources.recovery_mating_contact import (
                    validate_mating_contact_samples, verify_mating_contact,
                )

                validate_mating_contact_samples(interval_record)
                verified = verify_mating_contact(resource_id=rid, state=state,
                    evidence={**evidence, "resource_models": models},
                    owner_evidence=contract["mating_owner_evidence"],
                    joint_trajectory=interval_record["joint_trajectory"],
                    mating_contact=interval_record["mating_contact"])
                if (interval_record.get("mating_start_state") != state or verified.get("status") != "supported"
                        or interval_record.get("mating_corridor_evidence") != verified):
                    raise ValueError("Insertion lacks matching continuous corridor evidence: "
                                     + str(verified.get("reason", "record changed")))
                last_mating = interval_record
            motion = _prepared_motion(state, rid, models, interval_record)
            motion_checkpoint = _motion_checkpoint(state, geometry, models, index=len(observations),
                moving=(rid, motion, 0, motion.duration),
                peer_holds=interval_record.get("stationary_resource_evidence"))
            append_checkpoint(motion_checkpoint)
            if exact_clock:
                clock = _event_clock_motion(motion_checkpoint, clock)
            state["resources"][rid]["joint_positions"] = deepcopy(motion.points[-1]["positions"])
        elif "nominal_snap_evidence" in interval_record:
            from cais_spade_llm.resources.robot.recovery_task_projection import project_nominal_snap

            if any(key in interval_record for key in ("configuration_after", "custody_effects", "joint_trajectory")):
                raise ValueError("Simulation correction cannot substitute robot motion or custody evidence")
            declared = contract["nominal_snap"]
            checked = project_nominal_snap(resource_id=rid, params=declared["params"], state=state,
                evidence={**evidence, "resource_models": models,
                          "stationary_part_evidence": stationary_support},
                owner_evidence=declared["owner_evidence"],
                endpoint_result=interval_record.get("scene_endpoint_result"))
            if (checked.get("status") != "supported" or interval_record["nominal_snap_evidence"] != checked
                    or interval_record.get("interval_complete") is not True):
                raise ValueError("Simulation correction lacks matching independently derived endpoint evidence")
            state.update(deepcopy(checked["projected_snapshot"]))
            part = checked["part_name"]
            projected_support[part] = deepcopy(checked["part_support_evidence"])
            if stationary_support is None:
                stationary_support = {}
            stationary_support[part] = deepcopy(checked["part_support_evidence"])
        elif interval_record.get("geometry_unchanged") is not True:
            raise ValueError("Owner event interval has no supported motion or unchanged-geometry declaration")
        fixture_supported = None
        if (contract["version"] in {6, 7}
                and any(effect.get("kind") == "release" for effect in interval_record.get("custody_effects", []))
                and contract.get("placement_target", {}).get("simulation_mating_contact", {}).get(
                    "retain_fixture_attachment") is True):
            from cais_spade_llm.resources.recovery_mating_contact import projected_fixture_handoff

            if last_mating is None:
                raise ValueError("Fixture release requires its validated insertion interval")
            fixture_supported = projected_fixture_handoff(resource_id=rid, part_name=task["part_name"],
                state=state, evidence={**evidence, "resource_models": models},
                owner_evidence=contract["mating_owner_evidence"], mating_contact=last_mating["mating_contact"],
                endpoint_evidence={"mating_start_state": last_mating["mating_start_state"],
                    "mating_native_record": last_mating,
                    "mating_corridor_evidence": last_mating["mating_corridor_evidence"]})
            if interval_record.get("fixture_handoff_evidence") != fixture_supported:
                raise ValueError("Fixture handoff differs from its recorded owner evidence")
        elif "fixture_handoff_evidence" in interval_record:
            raise ValueError("Fixture handoff lacks a corresponding owner declaration")
        if "configuration_after" in interval_record:
            interval_configuration = interval_record.get("configuration_interval")
            if interval_configuration is not None:
                bounded = deepcopy(models)
                _configuration_after(bounded, rid, {"configuration_after": interval_configuration})
                declared_bounds = models[rid].get("primitive_model", {}).get("configuration", {}).get("custody", {}).get("joint_bounds")
                if not isinstance(declared_bounds, dict):
                    raise ValueError("Gripper motion bounds are not recorded in its owner contract")
                for before, middle, after in zip(models[rid]["configuration"]["joints"],
                        interval_configuration["joints"], interval_record["configuration_after"]["joints"], strict=True):
                    if before == middle and before == after:
                        continue
                    limits = declared_bounds.get(before["name"])
                    if middle.get("frozen_interval") != limits or not isinstance(limits, list) or len(limits) != 2:
                        raise ValueError("Gripper interval differs from its declared complete joint bounds")
                    low, high = interval(*limits)
                    if not low <= before["frozen_position"] <= high or not low <= after["frozen_position"] <= high:
                        raise ValueError("Gripper endpoint lies outside its declared motion bounds")
                bounded_row = _checkpoint(state, geometry, bounded, index=len(observations), peer_holds=peer_holds)
                bounded_row["phase"] = "between"
                if exact_clock:
                    bounded_row["continuous_interval"] = [str(clock), str(clock)]
                append_checkpoint(bounded_row)
            else:
                # Endpoint configurations alone do not establish motion bounds.
                interval_complete = False
            _configuration_after(models, rid, interval_record)
        if interval_record.get("interval_complete") is False:
            interval_complete = False
        if "configuration" in models[rid]:
            _, matrix = _joint_geometry(state["resources"][rid], models[rid])
            state["resources"][rid]["current_pose"] = representative_pose(matrix)
        checkpoint = _checkpoint(state, geometry, models, index=len(observations))
        append_checkpoint(checkpoint)
        state.update(resources=deepcopy(checkpoint["resources"]), parts=deepcopy(checkpoint["parts"]))
        if interval_record.get("custody_effects"):
            _custody(state, rid, interval_record["custody_effects"], geometry)
            if exact_clock:
                custody_source = _event_custody_source(interval_record, declared_events, record)
            for effect in interval_record["custody_effects"]:
                if effect["kind"] != "release":
                    continue
                supported = (fixture_supported if fixture_supported is not None else
                    projected_part_support(part_name=effect["part"], state=state, evidence=evidence))
                expected_support = {key: value for key, value in supported.items() if key != "physics_evidence"}
                if interval_record.get("part_support_evidence") != expected_support:
                    raise ValueError("Release interval has no matching owner-derived deposited-part support evidence")
                if supported.get("status") != "supported":
                    raise ValueError("Released-part settling is unresolved: " + str(supported.get("reason", "missing support")))
                projected_support[effect["part"]] = deepcopy(expected_support)
                if stationary_support is None:
                    stationary_support = {}
                stationary_support[effect["part"]] = deepcopy(expected_support)
            checkpoint = _checkpoint(state, geometry, models, index=len(observations))
            append_checkpoint(checkpoint)
            state.update(resources=deepcopy(checkpoint["resources"]), parts=deepcopy(checkpoint["parts"]))
            if contract["version"] >= 3 and any(effect.get("kind") == "release" for effect in interval_record["custody_effects"]):
                from cais_spade_llm.resources.resource_safety_preparation import _outline_place_departure

                current = dict(zip(("x", "y", "z", "qx", "qy", "qz", "qw"), state["resources"][rid]["current_pose"], strict=True))
                derived = _outline_place_departure(task=task, declarations=owner_declarations,
                    model=recovery_des_model, target=contract.get("placement_target", {}), current_pose=current)
                if derived != placement_departure:
                    raise ValueError("Placement departure parameters differ from the supported release checkpoint")
    if motion_capabilities is not None and motion_index != len(motion_capabilities["motion_requests"]):
        raise ValueError("Cached owner event omits capability-checked motion targets")
    part = task.get("part_name")
    if contract.get("endpoint_numerical_precision") and (
            "current_pose" in task.get("expected_end_state", {})
            or "target_pose" in (task.get("params") or {})):
        raise ValueError("Owner numerical precision cannot replace an explicit proposal coordinate requirement")
    endpoint_evidence = _event_endpoint(contract, bindings, state, rid, part, intervals)
    if contract.get("carried_part_transition") is True and (
            evidence["snapshot"]["resources"][rid].get("held_part") != part
            or state["resources"][rid].get("held_part") != part
            or any(row.get("custody_effects") for row in intervals)):
        raise ValueError("Carried-part completion requires preserved physical custody throughout the event")
    for field in ("held_part", "gripper_state"):
        if field in resource_updates and state["resources"][rid].get(field) != resource_updates[field]:
            raise ValueError(f"Owner event geometry does not establish its {field} effect")
    if (part and declarations.get("part_location", {}).get("scope") == "part"
            and "part_location" in expected and state["parts"][part].get("contained_by") == expected["part_location"]
            and expected["part_location"] is not None):
        part_updates["part_location"] = deepcopy(expected["part_location"])
    if (part and declarations.get("part_location", {}).get("scope") == "part"
            and expected.get("part_location") == rid and state["resources"][rid].get("held_part") == part):
        part_updates["part_location"] = rid
    nominal_witness, nominal_effects = None, None
    if contract["version"] == 7 and contract.get("task_contract") is not None:
        from cais_spade_llm.resources.resource_safety_preparation import (
            _outline_nominal_program, _outline_nominal_task_effects,
        )
        from cais_spade_llm.resources.robot.recovery_task_projection import nominal_task_projection

        nominal_interval_states.append(deepcopy(state["resources"][rid]))
        owner_evidence = nominal_owner_evidence
        nominal = _outline_nominal_program(task=task, recovery_snapshot=recovery_snapshot,
            recovery_des_model=recovery_des_model, physical_evidence=evidence,
            resource_evidence=owner_evidence, intervals=intervals, interval_states=nominal_interval_states)
        if nominal["nominal_snap"] != contract.get("nominal_snap"):
            raise ValueError("Nominal simulation correction differs from its resolved program step")
        nominal_witness = nominal_task_projection(task=task, recovery_des_model=recovery_des_model,
            resource_evidence=owner_evidence, step_evidence=nominal["step_evidence"], intervals=intervals,
            resource_model=models[rid])
        if nominal_witness.get("status") != "supported":
            raise ValueError("Nominal program coverage is unresolved: " + str(nominal_witness.get("reason")))
        nominal_effects = _outline_nominal_task_effects(task=task, recovery_des_model=recovery_des_model,
            resource_evidence=owner_evidence, nominal=nominal)
        if nominal_effects.get("status") != "supported":
            raise ValueError("Nominal program effects are unresolved: " + str(nominal_effects.get("reason")))
        from cais_spade_llm.resources.robot.recovery_task_projection import check_nominal_task_effects

        nominal_effects["state_consistency"] = check_nominal_task_effects(
            nominal_effects=nominal_effects, resource_updates=resource_updates,
            physical_resource=state["resources"][rid])
        context = nominal_effects["state"].get("_task_ctx")
        if context is not None:
            if not isinstance(context, dict):
                raise ValueError("Nominal task context effect must preserve its declared dictionary")
            for field in list(state["resources"][rid]):
                if field.startswith("task_ctx."):
                    state["resources"][rid].pop(field)
            private_updates = {key: value for key, value in private_updates.items() if not key.startswith("task_ctx.")}
            private_updates.update({"task_ctx." + key: deepcopy(value) for key, value in context.items()})
    state["resources"][rid].update(deepcopy(resource_updates))
    state["resources"][rid].update(deepcopy(private_updates))
    custody_updates, valuations, custody_witness = _owner_product_custody(task=task, state=state,
        evidence=evidence, resource_updates={**resource_updates,
            **({"part_state": part_updates["part_state"]} if contract.get("carried_part_transition") is True else {})},
        support=projected_support, rid=rid)
    if exact_clock and rid in valuations:
        owner_fields = evidence.get("environment_models", {}).get(rid, {}).get("state_variables", {})
        valuations[rid].update({field: deepcopy(value) for field, value in resource_updates.items()
                               if field in owner_fields})
        if custody_source is not None:
            valuations[rid].update({field: deepcopy(state["resources"][rid][field])
                for field in ("held_part", "gripper_state") if field in owner_fields})
    for field, value in custody_updates.items():
        if declarations.get(field, {}).get("scope") != "part":
            continue
        if (field == "part_location" and value == rid and expected.get(field) == evidence["jids"][rid]
                and state["resources"][rid].get("held_part") == part):
            custody_witness["part_location_binding"] = {"resource_id": rid,
                "resource_jid": evidence["jids"][rid], "source": "physical_evidence/jids"}
            value = evidence["jids"][rid]
        if field in expected and (type(value) is not type(expected[field]) or value != expected[field]):
            raise ValueError(f"Established product custody conflicts with proposed {field}")
        part_updates[field] = deepcopy(value)
    if part:
        state["parts"][part].update(deepcopy(part_updates))
        for field, key in (("part_state", "state"), ("part_location", "location")):
            if field in part_updates:
                state["parts"][part][key] = deepcopy(part_updates[field])
    if checked_routes != set(routes):
        raise ValueError("Calculated route has no corresponding strict replay interval")
    append_checkpoint(_checkpoint(state, geometry, models, index=len(observations)))
    completion = _owner_product_completion(task=task, state=state, geometry=geometry,
        evidence=evidence, contract=contract, rid=rid, support=projected_support)
    if completion is not None:
        from cais_spade_llm.agents.central_controller._product_effect_evidence import _join_product_effects

        completion.update(task_id=record["binding"]["conditions_fingerprint"], time=observations[-1]["time"])
        joined = _join_product_effects({"observations": observations, "projected_snapshot": state}, [completion], state)
        state, observations = joined["projected_snapshot"], joined["observations"]
        effects = completion["product_effects"][part]
        if "state" in effects:
            part_updates["part_state"] = deepcopy(effects["state"])
            state["parts"][part]["part_state"] = deepcopy(effects["state"])
            observations[-1]["parts"][part]["part_state"] = deepcopy(effects["state"])
        part_updates["part_location"] = deepcopy(completion["part_location"])
        for field in ("part_location", "location"):
            state["parts"][part][field] = deepcopy(completion["part_location"])
            observations[-1]["parts"][part][field] = deepcopy(completion["part_location"])
        part_updates["processCompleted"] = deepcopy(state["parts"][part]["processCompleted"])
        for field, value in part_updates.items():
            if field in expected and (type(value) is not type(expected[field]) or value != expected[field]):
                raise ValueError(f"Declared product completion conflicts with proposed {field}")
    _part_entries(observations)
    owner_interval_contract = None
    if exact_clock:
        owner_interval_contract = {
            "kind": "resource-owned event preview", "projection_clock_version": 1,
            "source_contract_fingerprint": record["contract_fingerprint"],
            "source_contract": deepcopy(contract),
            "preview_fingerprint": recovery_validation_fingerprint(record),
            "owner_declarations": [deepcopy(event) for event, _ in declared_events],
            "declared_task": {"resource_id": rid, "parameters": deepcopy(bindings)},
            "duration_exact": str(clock), "zero_duration_intervals": "geometry bounds only"}
        before_values = evidence.get("environment_valuation", {})
        for index, observation in enumerate(observations):
            completed = (Fraction(observation["time_exact"]) == clock
                         and observation.get("phase") != "between")
            values = valuations if completed else before_values
            observation["state_snapshot"] = deepcopy(values)
            overlays = observation.get("owner_safety_evidence", {}).get("physical_state_updates", {})
            for owner, fields in values.items():
                physical = observation["resources"][owner]
                owner_fields = evidence.get("environment_models", {}).get(owner, {}).get("state_variables", {})
                declared_fields = {field.replace("{part_name}", name) if "{part_name}" in field else field
                    for field in owner_fields
                    for name in (geometry["parts"] if "{part_name}" in field else [None])}
                for field, value in fields.items():
                    if field not in declared_fields:
                        raise ValueError("Owner valuation contains an undeclared field: " + owner + "/" + field)
                    if field not in physical and field in {"current_pose", "joint_names", "joint_positions",
                            "base_pose", "grasp_transform", "held_part", "gripper_state", "contained_parts", "frame"}:
                        raise ValueError("Owner valuation cannot supply missing physical evidence: " + owner + "/" + field)
                    supported = overlays.get(owner, {}).get(field, value)
                    if field in physical and recovery_validation_fingerprint(physical[field]) != recovery_validation_fingerprint(supported):
                        previous = before_values.get(owner, {})
                        declared_endpoint = (completed and field in previous
                            and field not in {"held_part", "gripper_state"}
                            and recovery_validation_fingerprint(physical[field]) == recovery_validation_fingerprint(previous[field]))
                        if not declared_endpoint:
                            raise ValueError("Physical checkpoint contradicts recorded or derived owner field: " + owner + "/" + field)
                    physical[field] = deepcopy(supported)
        state = deepcopy({**state, "resources": observations[-1]["resources"],
                          "parts": observations[-1]["parts"]})
    # Keep record ordering while avoiding copies that the supported successor
    # immediately replaces. These local placeholders are never returned.
    successor = deepcopy({key: None if key in {"snapshot", "resource_models", "step_evidence"} else value
                          for key, value in evidence.items()})
    successor.update(snapshot=deepcopy(state), resource_models=deepcopy(models), step_evidence={})
    if contract["version"] >= 3 and grasp_indices:
        calculation = contract.get("pickup_target_calculation")
        if (not isinstance(calculation, dict) or calculation.get("status") != "calculated"
                or calculation.get("part_name") != part or not isinstance(calculation.get("source"), dict)):
            raise ValueError("Validated pickup must retain its part-specific owner target calculation")
        successor.setdefault("resources", {}).setdefault(rid, {})["validated_pickup_target_calculation"] = {
            "part_name": part, "calculation": deepcopy(calculation), "contract_id": record["contract_id"],
            "contract_fingerprint": record["contract_fingerprint"], "binding": deepcopy(record["binding"]),
            "source": "resource-owned pickup preview", "physical_execution_verified": False}
    elif release_indices:
        successor.setdefault("resources", {}).setdefault(rid, {}).pop("validated_pickup_target_calculation", None)
    if completion is not None:
        successor.setdefault("product_completion_records", []).append(deepcopy(completion))
    if successor.get("resources", {}).get(rid, {}).get("controller_hold_model"):
        successor["resources"][rid]["projected_controller_hold"] = {
            "kind": "projected", "joint_names": deepcopy(state["resources"][rid]["joint_names"]),
            "positions": deepcopy(state["resources"][rid]["joint_positions"]),
            "binding": deepcopy(record["binding"]), "source": record["source"],
            "command_sent": False, "physical_execution_verified": False}
    if "environment_valuation" in evidence:
        successor["environment_valuation"] = deepcopy(valuations)
    if part and part in successor.get("products", {}):
        successor["products"][part].update({key: deepcopy(state["parts"][part][key])
            for key in ("state", "location", "target", "processCompleted") if key in state["parts"][part]})
    for name, supported in projected_support.items():
        successor.setdefault("stationary_part_evidence", {})[name] = deepcopy(supported)
        if name not in successor.setdefault("stationary_parts", []):
            successor["stationary_parts"].append(name)
    successor["provenance"] = {"kind": "projected", "source": record["source"],
        "binding": deepcopy(record["binding"]), "contract_fingerprint": record["contract_fingerprint"]}
    result = {"status": "projected", "observations": observations, "geometry": geometry,
        "resource_models": models, "projected_snapshot": deepcopy(state), "successor_evidence": successor,
        "resource_updates": resource_updates, "part_updates": part_updates,
        "endpoint_evidence": endpoint_evidence,
        "projected_resource_snapshot": {**deepcopy(recovery_snapshot), **deepcopy(resource_updates), **deepcopy(private_updates),
            "current_pose": deepcopy(state["resources"][rid].get("current_pose")), "current_pose_source": "projected_state"},
        "deferred_valuation_fields": [field for field in expected if field not in resource_updates and field not in part_updates],
        "deferred_part_fields": [field for field in expected if field not in part_updates
                                 and declarations.get(field, {}).get("scope") == "part"],
        "interval_complete": interval_complete, "provenance": deepcopy(successor["provenance"]),
        "product_custody_evidence": deepcopy(custody_witness), "product_completion_evidence": deepcopy(completion),
        "resource_valuation_updates": {owner: {field: deepcopy(value) for field, value in values.items()
            if field not in evidence.get("environment_valuation", {}).get(owner, {})
            or evidence["environment_valuation"][owner][field] != value}
            for owner, values in valuations.items()},
        "model_assumptions": {name: deepcopy(row.get("model_assumptions", [])) for name, row in
                              successor.get("stationary_part_evidence", {}).items() if row.get("model_assumptions")},
        "controller_hold_model_assumptions": {peer: deepcopy(hold) for item in intervals
            for peer, hold in item.get("stationary_resource_evidence", {}).items()},
        "execution_authorized": False, "command_sent": False, "unresolved": []}
    if owner_interval_contract is not None:
        result["owner_interval_contract"] = owner_interval_contract
        result["projection_clock_version"] = 1
    if routes:
        result["composition_obligations"] = [{"stage_index": int(index),
            "route_fingerprint": recovery_validation_fingerprint(route),
            "artifact": deepcopy(route["artifact"]), "requires_composition_validation": True}
            for index, route in routes.items()]
    if nominal_witness is not None:
        result["nominal_task_effects"] = nominal_effects
        for field in list(result["projected_resource_snapshot"]):
            if field.startswith("task_ctx.") and field not in state["resources"][rid]:
                result["projected_resource_snapshot"].pop(field)
    if contract.get("task_contract") is not None:
        result["task_contract_witness"] = {**(nominal_witness or {}),
            "task_contract": contract["task_contract"],
            "declaration_fingerprint": contract.get("task_declaration_fingerprint"),
            "params": deepcopy(record.get("task_params", {})), "resource_updates": deepcopy(resource_updates),
            "part_updates": deepcopy(part_updates), "contract_fingerprint": record["contract_fingerprint"]}
    return result


def project_recovery_outline_event(*, task: dict, recovery_snapshot: dict,
                                   recovery_des_model: dict, evidence: dict,
                                   event_evidence: list | None = None,
                                   resource_evidence: dict | None = None) -> dict:
    """Validate an independent owner event preview without composing primitives.

    Every trajectory and attachment change starts at the accepted projected scene.
    All calculations are local; missing owner contracts never call live interfaces.
    """
    result = {"status": "needs_context", "observations": [], "interval_complete": False,
        "unresolved": [], "execution_authorized": False, "command_sent": False,
        "context_requests": ["/physical_evidence", f"/resources/{task.get('resource_jid')}/outline_event_evidence"],
        "requires_projection": True}
    try:
        matches = [rid for rid, jid in evidence["jids"].items() if jid == task["resource_jid"]]
        if len(matches) != 1:
            raise ValueError("Owner event has no unique captured resource identity")
        rejected = rejected_owner_event_effects(task=task, recovery_snapshot=recovery_snapshot,
            recovery_des_model=recovery_des_model, physical_evidence=evidence)
        if rejected is not None:
            return rejected
        binding = outline_event_binding(task, recovery_snapshot, recovery_des_model, evidence)
        records = event_evidence if event_evidence is not None else evidence.get("event_evidence", [])
        if not isinstance(records, list):
            raise ValueError("Recorded owner event evidence must be a list")
        matching = [row for row in records if isinstance(row, dict) and row.get("binding") == binding]
        if not matching:
            raise ValueError("No owner event preview matches the proposed conditions and projected starting scene")
        failures = []
        for record in matching:
            try:
                projected = _event_record_projection(task=task, recovery_snapshot=recovery_snapshot,
                    recovery_des_model=recovery_des_model, evidence=evidence, record=record, rid=matches[0],
                    resource_evidence=resource_evidence)
                if projected["interval_complete"]:
                    return projected
                failures.append(projected)
            except (KeyError, TypeError, ValueError, IndexError) as exc:
                result["unresolved"].append(str(exc))
        if failures:
            return failures[0]
        return result
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        result["unresolved"].append(str(exc))
        return result
