from __future__ import annotations

"""Recorded support contact and continuous, non-executing departure evidence."""

from copy import deepcopy
from fractions import Fraction
from itertools import product
import hashlib
import json
import math
from typing import Any

from cais_spade_llm.agents.shared_information.recovery_validation_protocol import recovery_validation_fingerprint
from cais_spade_llm.resources.continuous_motion import (
    ContinuousMotion, add, interval, kinematic_transforms, mul, neg, polynomial_bounds,
    quaternion_transform, representative_pose, rotation, transform,
)
from cais_spade_llm.resources.primitive_observations_3d import _compose, _unit_pose


_LIMIT = 4096
_ZERO = (0., 0.)


def validate_support_departure_samples(record: dict) -> None:
    """Check the saved native contact audit separately from the continuous proof.

    Args:
        record: One saved departure interval including its exact request bytes.

    Raises:
        ValueError: The complete native contact inventory is absent or conflicts.
    """
    text = record.get("support_departure_json")
    if (not isinstance(text, str) or hashlib.sha256(text.encode()).hexdigest()
            != record.get("support_departure_fingerprint")):
        raise ValueError("Native support departure request fingerprint differs")
    support = json.loads(text)
    if support != record.get("support_departure"):
        raise ValueError("Native support departure request differs from its continuous evidence")
    audit = record.get("native_support_departure_evidence")
    if (not isinstance(audit, dict) or audit.get("version") != 1
            or audit.get("source") != "PreviewRecoveryMotion" or audit.get("pairs_complete") is not True
            or audit.get("continuous_collision_verified") is not False or not audit.get("samples")):
        raise ValueError("Complete native support departure contact evidence is unavailable")
    points = record.get("joint_trajectory", {}).get("points")
    samples = audit["samples"]
    if (not isinstance(points, list) or len(points) < 2 or not isinstance(samples, list)
            or len(samples) != len(points)
            or any(not isinstance(sample, dict) or type(sample.get("state_index")) is not int
                   or sample["state_index"] != index
                   for index, sample in enumerate(samples))):
        raise ValueError("Native support departure audit does not cover every trajectory waypoint")
    expected = {(row["part_object_id"], row["support_object_id"]): row for row in support["pairs"]}
    first_separation, previous = {}, -1
    for sample in samples:
        index = sample.get("state_index")
        if type(index) is not int or index <= previous:
            raise ValueError("Native support departure sample order is unavailable")
        previous = index
        count = sample.get("checked_contacts")
        if type(count) is not int or not 0 <= count < 4096:
            raise ValueError("Native support departure contact inventory is incomplete")
        rows = sample.get("pairs", [])
        actual = {(row["part_object_id"], row["support_object_id"]): row for row in rows}
        if set(actual) != set(expected) or len(rows) != len(expected):
            raise ValueError("Native support departure sample omits a recorded pair")
        for pair, row in actual.items():
            bound = expected[pair]["initial_penetration_bound"]
            depth, distance, contact = row.get("maximum_depth"), row.get("minimum_signed_distance"), row.get("contact")
            if (not _number(depth) or not 0 <= depth <= bound or not _number(distance)
                    or distance < -bound or type(contact) is not bool):
                raise ValueError("Native support departure exceeds its recorded contact bound")
            separated = not contact and distance > 0
            if pair in first_separation and not separated:
                raise ValueError("Native support departure re-enters the recorded support")
            if separated:
                first_separation.setdefault(pair, index)
    rows = audit.get("pairs", [])
    actual = {(row["part_object_id"], row["support_object_id"]): row for row in rows}
    if set(actual) != set(expected) or len(rows) != len(expected) or set(first_separation) != set(expected):
        raise ValueError("Native support departure has no complete separated endpoint")
    for pair, row in actual.items():
        if (row.get("endpoint_separated") is not True
                or row.get("first_separation_index") != first_separation[pair]
                or row.get("initial_penetration_bound") != expected[pair]["initial_penetration_bound"]):
            raise ValueError("Native support departure endpoint differs from its recorded samples")


def _number(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def _vector(value: Any, name: str) -> list:
    if not isinstance(value, list) or len(value) != 3 or not all(_number(v) for v in value):
        raise ValueError(f"{name} requires three finite coordinates")
    return value


def _add(a, b):
    return b if a == _ZERO else a if b == _ZERO else add(a, b)


def _mul(a, b):
    return _ZERO if a == _ZERO or b == _ZERO else mul(a, b)


def _sum(values):
    result = _ZERO
    for value in values:
        result = _add(result, value)
    return result


def _matrix(a, b):
    return [[_sum(_mul(a[i][k], b[k][j]) for k in range(4)) for j in range(4)] for i in range(4)]


def _inverse(pose):
    from cais_spade_llm.recovery_framework.geometry import rotate

    p = list(map(float, _unit_pose(pose, "recorded pose")))
    q = [-v for v in p[3:6]] + p[6:]
    return [*rotate(q, [-v for v in p[:3]]), *q]


def _vertices(collision: dict, model: dict, evidence: dict) -> list[list[float]]:
    shape = collision["geometry"]
    if shape["type"] == "box":
        size = _vector(shape["size"], "box size")
        if min(size) <= 0:
            raise ValueError("Box sizes must be positive")
        vertices = [[s[i] * size[i] / 2 for i in range(3)] for s in product((-1., 1.), repeat=3)]
    elif shape["type"] == "mesh":
        mesh = evidence["physics_meshes"][shape["uri"]]
        scale = _vector(shape["scale"], "mesh scale")
        if (mesh.get("source_kind") != "recorded_collision_mesh" or not mesh.get("source_sha256")
                or mesh.get("uri") != shape["uri"] or min(scale) <= 0
                or not mesh.get("vertices") or mesh.get("vertex_count") != len(mesh["vertices"])):
            raise ValueError("Complete recorded collision mesh vertices and source hash are required")
        vertices = [[v[i] * scale[i] for i in range(3)]
                    for v in (_vector(row, "mesh vertex") for row in mesh["vertices"])]
    else:
        # An inscribed cylinder polygon cannot bound all carried geometry.
        raise ValueError("Support departure currently requires exact box or mesh collision geometry")
    local = _compose(_unit_pose(_inverse(model["pose"]), "inverse model pose"),
                     _unit_pose(collision["pose"], "collision pose"))
    return [list(row) for row in sorted({tuple(map(float, _compose(local,
        (*vertex, 0., 0., 0., 1.))[:3])) for vertex in vertices})]


def _object_id(collision_id: str, evidence: dict) -> str:
    candidates = []
    for binding in evidence.get("scene_object_bindings", {}).values():
        for row in binding.get("objects", []):
            if row.get("recorded_geometry", {}).get("collision_id") == collision_id:
                candidates.append(row.get("object_id"))
    explicit = evidence.get("support_collision_object_bindings", {}).get(collision_id)
    if isinstance(explicit, dict) and explicit.get("source"):
        candidates.append(explicit.get("object_id"))
    identities = {value for value in candidates if isinstance(value, str) and value}
    if len(identities) != 1:
        raise ValueError(f"Exact recorded collision-object identity is required for {collision_id}")
    return next(iter(identities))


def _plane(collision: dict, gravity: list) -> tuple[list, list]:
    from cais_spade_llm.recovery_framework.geometry import rotate

    if collision["geometry"].get("type") != "box":
        raise ValueError("Support departure requires a recorded static box face")
    size = _vector(collision["geometry"]["size"], "support box size")
    if min(size) <= 0:
        raise ValueError("Support box size must be positive")
    pose = list(map(float, _unit_pose(collision["pose"], "support pose")))
    length = math.sqrt(sum(v * v for v in gravity))
    if length == 0:
        raise ValueError("Support departure requires recorded nonzero gravity")
    up = [-v / length for v in gravity]
    for axis in range(3):
        for sign in (-1., 1.):
            normal = rotate(pose[3:], [sign if i == axis else 0. for i in range(3)])
            if all(math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12) for a, b in zip(normal, up, strict=True)):
                point = list(map(float, _compose(_unit_pose(pose, "support pose"),
                    (*[sign * size[i] / 2 if i == axis else 0. for i in range(3)], 0., 0., 0., 1.))[:3]))
                return point, normal
    raise ValueError("Support face requires an unsupported friction/contact model")


def _affine(matrix, normal, plane):
    return [ _sum(_mul(interval(normal[i]), matrix[i][j]) for i in range(3)) for j in range(3)] + [
        _sum(_mul(interval(normal[i]), _add(matrix[i][3], interval(-plane[i]))) for i in range(3))]


def _minimum(affine, vertices):
    values = [_add(_sum(_mul(affine[i], interval(v[i])) for i in range(3)), affine[3]) for v in vertices]
    return min(v[0] for v in values), min(v[1] for v in values)


def prepare_support_departure(*, resource_id: str, state: dict, evidence: dict) -> dict:
    """Bind a just-acquired part to its recorded support without allowing deeper contact.

    Args:
        resource_id: Exact owning resource identity.
        state: Projected snapshot immediately after grasp, before departure.
        evidence: Frozen physics, scene mappings and resource geometry.

    Returns:
        A private contact record, or explicit unavailable evidence. No motion is sent.
    """
    result = {"status": "NEEDS_CONTEXT", "version": 1, "command_sent": False, "execution_authorized": False}
    try:
        from cais_spade_llm.resources.recovery_geometry import (
            _cartesian_endpoint_residual, owner_cartesian_endpoint_precision, projected_part_support,
        )

        owner = state["resources"][resource_id]
        part_name = owner["held_part"]
        if not part_name or sum(row.get("held_part") == part_name for row in state["resources"].values()) != 1:
            raise ValueError("Support departure requires unambiguous acquired-part custody")
        part = state["parts"][part_name]
        configuration = evidence["resource_models"][resource_id]["configuration"]
        physics = evidence["physics_evidence"]
        support = evidence["stationary_part_evidence"][part_name]
        if support.get("status") != "supported":
            raise ValueError("Recorded part support is unavailable")
        projected_support = support.get("source", {}).get("kind") == "projected"
        if projected_support:
            support_state = deepcopy(state)
            support_state["parts"][part_name]["current_pose"] = deepcopy(support["current_pose"])
            derived = projected_part_support(part_name=part_name, state=support_state, evidence=evidence)
            if (derived.get("status") != "supported" or
                    {key: value for key, value in derived.items() if key != "physics_evidence"} != support):
                raise ValueError("Projected support no longer matches its established model calculation")
            physics = derived["physics_evidence"]
        model = physics["models"][part["model_name"]]
        precision = owner_cartesian_endpoint_precision()
        for prior in (model["pose"], support["current_pose"],
                      _compose(_unit_pose(owner["current_pose"], "tool pose"), _unit_pose(owner["grasp_transform"], "grasp transform"))):
            residual = _cartesian_endpoint_residual(part["current_pose"], list(map(float, prior)))
            if (residual["position_residual"] > precision["position_precision"]
                    or residual["orientation_residual"] > precision["orientation_precision"]):
                raise ValueError("Acquired part no longer matches its recorded support pose")
        if owner["joint_names"] != configuration["joint_names"]:
            raise ValueError("Owner joint identities differ from the recorded model")
        positions = {name: interval(value) for name, value in zip(owner["joint_names"], owner["joint_positions"], strict=True)}
        transforms = kinematic_transforms(configuration, positions)
        reference_pose = representative_pose(transforms[configuration["reference_link"]])
        residual = _cartesian_endpoint_residual(reference_pose, owner["current_pose"])
        if (residual["position_residual"] > precision["position_precision"]
                or residual["orientation_residual"] > precision["orientation_precision"]):
            raise ValueError("Recorded joints and reference pose do not establish the same departure start")
        grasp = list(map(float, _unit_pose(owner["grasp_transform"], "grasp transform")))
        attached = _matrix(transforms[configuration["reference_link"]], quaternion_transform(grasp[:3], grasp[3:]))
        own = {row["id"]: row for link in model["links"].values() for row in link["collisions"]}
        collisions = {row["id"]: (body, row) for body in physics["models"].values()
                      for link in body["links"].values() for row in link["collisions"]}
        contacts = {}
        for row in physics["contacts"]:
            first, second = row["collision1"], row["collision2"]
            if not (first in own or second in own):
                continue
            part_id, support_id = (first, second) if first in own else (second, first)
            if support_id in own or collisions[support_id][0].get("static") is not True:
                raise ValueError("Initial support contact includes an unsupported moving body")
            if not row.get("positions") or not (len(row["positions"]) == len(row.get("normals", [])) == len(row.get("depths", []))):
                raise ValueError("Complete initial contact positions, normals and depths are required")
            if any(not _number(v) or v < 0 for v in row["depths"]):
                raise ValueError("Recorded contact penetration is invalid")
            contacts.setdefault((part_id, support_id), []).append(row)
        if not contacts:
            raise ValueError("Recorded supporting collision contacts are missing")
        pairs = []
        for (part_id, support_id), records in sorted(contacts.items()):
            surface = collisions[support_id][1]
            point, normal = _plane(surface, _vector(physics["gravity"], "gravity"))
            for row in records:
                for value in row["normals"]:
                    value = _vector(value, "contact normal")
                    length = math.sqrt(sum(v * v for v in value))
                    if not length or not math.isclose(abs(sum(a * b for a, b in zip(value, normal, strict=True))) / length,
                                                     1., abs_tol=1e-12, rel_tol=1e-12):
                        raise ValueError("Recorded contact is not normal to the supporting face")
            vertices = _vertices(own[part_id], model, evidence)
            signed = _minimum(_affine(attached, normal, point), vertices)
            if signed[0] > 0:
                raise ValueError("Recorded part is already separated from the requested supporting face")
            pairs.append({"part_object_id": _object_id(part_id, evidence),
                "support_object_id": _object_id(support_id, evidence),
                "part_collision_id": part_id, "support_collision_id": support_id,
                "plane_point": point, "plane_normal": normal, "part_vertices": vertices,
                "initial_signed_distance": list(signed), "initial_penetration_bound": max(0., -signed[0]),
                "recorded_contact_depths": [v for row in records for v in row["depths"]],
                "source_contact_fingerprint": recovery_validation_fingerprint(records),
                "geometry_fingerprint": recovery_validation_fingerprint({"part": own[part_id], "support": surface,
                    "vertices": vertices}), "contact_tolerance_added": 0.})
        binding = {"state_fingerprint": recovery_validation_fingerprint(state),
            "model_fingerprint": recovery_validation_fingerprint({key: value
                for key, value in evidence["resource_models"][resource_id].items()
                if key != "event_projection_contracts"}),
            "physics_fingerprint": recovery_validation_fingerprint(physics),
            "mesh_fingerprint": recovery_validation_fingerprint(evidence.get("physics_meshes", {})),
            "scene_object_bindings_fingerprint": recovery_validation_fingerprint({"parts": evidence.get("scene_object_bindings"),
                "supports": evidence.get("support_collision_object_bindings")}),
            "peer_fingerprint": recovery_validation_fingerprint({rid: {"state": row,
                "model": evidence.get("resource_models", {}).get(rid),
                "observation": evidence.get("resources", {}).get(rid)}
                for rid, row in state["resources"].items() if rid != resource_id})}
        return {**result, "status": "prepared", "resource_id": resource_id, "part_name": part_name,
            "reference_link": configuration["reference_link"], "binding": binding, "pairs": pairs,
            "max_refinements": _LIMIT, "source": {"kind": (
                "projected support and projected attachment" if projected_support
                else "recorded support and projected attachment"),
                "instance_id": physics["instance_id"], "simulation_time": physics["simulation_time"],
                "basis_physics_fingerprint": recovery_validation_fingerprint(evidence["physics_evidence"]),
                "support_fingerprint": recovery_validation_fingerprint(support)},
            "physical_execution_verified": False}
    except (KeyError, ValueError, TypeError, IndexError) as exc:
        return {**result, "reason": str(exc)}


def _derivatives(coefficients, start, end):
    derivatives = [[i * v for i, v in enumerate(row)][1:] for row in coefficients]
    # A common nonnegative endpoint factor is removed before interval evaluation.
    # This retains coordinated zero starting/stopping speed instead of treating
    # independently bounded joint speeds as unrelated disturbances.
    while all(not row or row[0] == 0 for row in derivatives) and any(derivatives):
        derivatives = [row[1:] if row else [] for row in derivatives]
    while any(derivatives) and all(sum(row) == 0 for row in derivatives):
        divided = []
        for row in derivatives:
            quotient = []
            for value in row[:-1]:
                quotient.append(value + (quotient[-1] if quotient else 0))
            divided.append(quotient)
        derivatives = divided
    return [polynomial_bounds(row, start, end) if row else _ZERO for row in derivatives]


def _axis_skew(axis):
    norm = sum(Fraction(v) ** 2 for v in axis)
    low = high = math.sqrt(norm)
    while Fraction(low) ** 2 > norm:
        low = math.nextafter(low, -math.inf)
    while Fraction(high) ** 2 < norm:
        high = math.nextafter(high, math.inf)
    inverse = (math.nextafter(float(1 / Fraction(high)), -math.inf),
               math.nextafter(float(1 / Fraction(low)), math.inf))
    x, y, z = [_mul(interval(v), inverse) for v in axis]
    return [[_ZERO, neg(z), y, _ZERO], [z, _ZERO, neg(x), _ZERO],
            [neg(y), x, _ZERO, _ZERO], [_ZERO] * 4]


def _transforms_with_derivative(configuration, positions, derivatives, *, origins=None, root=None, axis_skews=None):
    zero = [[_ZERO] * 4 for _ in range(4)]
    if root is None:
        root = (quaternion_transform(configuration["root_xyz"], configuration["root_quaternion"])
                if "root_quaternion" in configuration else transform(configuration["root_xyz"], configuration["root_rpy"]))
    result = {configuration["root"]: (root, zero)}
    for row in configuration["joints"]:
        parent, parent_derivative = result[row["parent"]]
        origin = origins[row["child"]] if origins is not None else transform(row["xyz"], row["rpy"])
        frame, derivative = _matrix(parent, origin), _matrix(parent_derivative, origin)
        position = positions.get(row["name"], interval(row.get("frozen_position", 0.)))
        speed = derivatives.get(row["name"], _ZERO)
        change = [[interval(int(i == j)) for j in range(4)] for i in range(4)]
        change_derivative = [[_ZERO] * 4 for _ in range(4)]
        if row["type"] in {"continuous", "revolute"}:
            change = rotation(row["axis"], position)
            skew = axis_skews[row["child"]] if axis_skews is not None else _axis_skew(row["axis"])
            change_derivative = _matrix([[_mul(v, speed) for v in line] for line in skew], change)
        elif row["type"] == "prismatic":
            for i in range(3):
                change[i][3] = _mul(interval(row["axis"][i]), position)
                change_derivative[i][3] = _mul(interval(row["axis"][i]), speed)
        left, right = _matrix(derivative, change), _matrix(frame, change_derivative)
        result[row["child"]] = (_matrix(frame, change), [[_add(left[i][j], right[i][j]) for j in range(4)] for i in range(4)])
    return result


def verify_support_departure(*, resource_id: str, state: dict, evidence: dict,
                             joint_trajectory: dict, support_departure: dict) -> dict:
    """Prove no deeper support penetration and strict separation over native splines.

    The recorded motion is evaluated in an isolated model. Distance or positive
    derivative bounds apply until contact ends; later intervals remain clear.
    Unresolved geometry or an exhausted refinement limit never passes.
    """
    result = {"status": "NEEDS_CONTEXT", "command_sent": False, "execution_authorized": False,
              "physical_execution_verified": False, "continuous_support_departure_verified": False}
    try:
        recorded = prepare_support_departure(resource_id=resource_id, state=state, evidence=evidence)
        if recorded["status"] != "prepared":
            raise ValueError(recorded["reason"])
        if any(support_departure.get(key) != value for key, value in recorded.items()):
            raise ValueError("Support departure record differs from the unchanged starting evidence")
        configuration = evidence["resource_models"][resource_id]["configuration"]
        if any(configuration.get("joint_position_error", {}).values()):
            raise ValueError("Time-varying joint tracking error has no recorded departure derivative model")
        motion = ContinuousMotion(joint_trajectory, configuration)
        if motion.points[0]["positions"] != state["resources"][resource_id]["joint_positions"]:
            raise ValueError("Departure trajectory starts from a different joint configuration")
        reference_links = {configuration["reference_link"]}
        for joint in reversed(configuration["joints"]):
            if joint["child"] in reference_links:
                reference_links.add(joint["parent"])
                if "frozen_interval" in joint and joint["name"] not in motion.names:
                    raise ValueError("A reference-link joint interval has no recorded departure derivative model")
        grasp = list(map(float, _unit_pose(state["resources"][resource_id]["grasp_transform"], "grasp transform")))
        attachment = quaternion_transform(grasp[:3], grasp[3:])
        root = (quaternion_transform(configuration["root_xyz"], configuration["root_quaternion"])
                if "root_quaternion" in configuration else transform(configuration["root_xyz"], configuration["root_rpy"]))
        axis_skews = {row["child"]: _axis_skew(row["axis"]) for row in configuration["joints"]
                      if row["type"] in {"continuous", "revolute"}}
        proof, explored = [], 0
        for pair in recorded["pairs"]:
            clear = False
            cells = []
            for segment, (a, b) in enumerate(zip(motion.exact_times, motion.exact_times[1:], strict=False)):
                pending = [(Fraction(0), Fraction(1))]
                while pending:
                    low, high = pending.pop()
                    explored += 1
                    if explored > _LIMIT:
                        raise ValueError("Continuous support departure exceeded 4096 refinement cells")
                    positions = {name: polynomial_bounds(row, low, high)
                                 for name, row in zip(motion.names, motion.coefficients[segment], strict=True)}
                    velocities = dict(zip(motion.names, _derivatives(motion.coefficients[segment], low, high), strict=True))
                    matrices = _transforms_with_derivative(configuration, positions, velocities,
                        origins=motion.origins, root=root, axis_skews=axis_skews)
                    value, derivative = matrices[configuration["reference_link"]]
                    value, derivative = _matrix(value, attachment), _matrix(derivative, attachment)
                    distances = _minimum(_affine(value, pair["plane_normal"], pair["plane_point"]), pair["part_vertices"])
                    slopes = _minimum(_affine(derivative, pair["plane_normal"], [0., 0., 0.]), pair["part_vertices"])
                    if distances[0] > 0:
                        clear = True
                        kind = "separated"
                    elif (not clear and distances[0] >= -pair["initial_penetration_bound"]
                          and distances[1] <= 0):
                        # This also establishes that complete separation did not
                        # occur inside the cell before contact resumed. A negative
                        # vertex derivative alone does not exceed the depth bound.
                        kind = "within_initial_penetration_bound"
                    elif not clear and slopes[0] >= 0:
                        kind = "non_increasing_penetration"
                    else:
                        if distances[1] < 0 and clear:
                            raise ValueError("The carried part re-enters its supporting surface")
                        if not clear and distances[1] < -pair["initial_penetration_bound"]:
                            raise ValueError("The carried part increases its initial support penetration")
                        midpoint = (low + high) / 2
                        midpoint_positions = {name: polynomial_bounds(row, midpoint, midpoint)
                            for name, row in zip(motion.names, motion.coefficients[segment], strict=True)}
                        midpoint_value = _matrix(kinematic_transforms(configuration, midpoint_positions, motion.origins)[configuration["reference_link"]], attachment)
                        midpoint_distance = _minimum(_affine(midpoint_value, pair["plane_normal"], pair["plane_point"]), pair["part_vertices"])
                        if clear and midpoint_distance[1] < 0:
                            raise ValueError("The carried part re-enters its supporting surface")
                        if not clear and midpoint_distance[1] < -pair["initial_penetration_bound"]:
                            raise ValueError("The carried part increases its initial support penetration")
                        if midpoint == low or midpoint == high or high - low <= Fraction(1, 2**48):
                            raise ValueError("Continuous support departure remains unresolved")
                        pending.extend(((midpoint, high), (low, midpoint)))
                        continue
                    cells.append({"segment": segment, "start": float(a + low * (b - a)),
                        "end": float(a + high * (b - a)), "status": kind,
                        "signed_distance_bounds": list(distances), "scaled_normal_derivative_bounds": list(slopes)})
                    if not clear:
                        endpoint_positions = {name: polynomial_bounds(row, high, high)
                            for name, row in zip(motion.names, motion.coefficients[segment], strict=True)}
                        endpoint = _matrix(kinematic_transforms(configuration, endpoint_positions, motion.origins)[configuration["reference_link"]], attachment)
                        endpoint_distance = _minimum(_affine(endpoint, pair["plane_normal"], pair["plane_point"]), pair["part_vertices"])
                        cells[-1]["endpoint_signed_distance_bounds"] = list(endpoint_distance)
                        clear = endpoint_distance[0] > 0
            if not clear:
                raise ValueError("Departure does not establish strict separation from the support")
            proof.append({"part_object_id": pair["part_object_id"], "support_object_id": pair["support_object_id"], "cells": cells})
        return {**result, "status": "supported", "continuous_support_departure_verified": True,
            "support_departure_fingerprint": recovery_validation_fingerprint(recorded),
            "trajectory_fingerprint": recovery_validation_fingerprint(joint_trajectory),
            "binding": deepcopy(recorded["binding"]), "refinement_cells": explored, "pairs": proof,
            "model_assumptions": ["The acquired part remains rigidly attached to the recorded reference link",
                "The supplied joint trajectory follows its declared spline interpolation",
                "The recorded static supporting geometry and peer configuration remain unchanged"]}
    except (KeyError, ValueError, TypeError, IndexError) as exc:
        return {**result, "reason": str(exc)}
