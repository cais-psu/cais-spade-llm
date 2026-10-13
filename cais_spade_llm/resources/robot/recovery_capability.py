from __future__ import annotations

"""Robot-owned calculations for independent recovery primitive capability."""

import math
from copy import deepcopy
from itertools import product

from cais_spade_llm.resources.recovery_parameter_evidence import prepare_resource_primitive_parameters
from cais_spade_llm.resources.recovery_feasibility import derive_primitive_union
from cais_spade_llm.resources.robot.target_calculations import (
    controlled_link_height, pick_travel_height, placement_poses, vertical_pick_bias,
)


def _number(value) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def _pose_mapping(value) -> dict:
    if isinstance(value, dict):
        return deepcopy(value)
    if isinstance(value, list) and len(value) == 7 and all(_number(item) for item in value):
        return dict(zip(("x", "y", "z", "qx", "qy", "qz", "qw"), value))
    return {}


def _destination_geometry(evidence: dict, destination: str, part: str) -> dict | None:
    """Read existing destination declarations without loading current files."""
    inputs = evidence.get("inputs") or {}
    geometry = inputs.get("geometry") or {}
    scene = inputs.get("scene") or {}
    declared = evidence.get("destination_geometry")
    if isinstance(declared, dict) and declared.get("destination_location") == destination:
        return deepcopy(declared)
    if destination == "assembly_board-v1":
        from cais_spade_llm.product.profile import ProductProfile

        board = geometry.get("assembly_board") or {}
        center, slot = board.get("center") or {}, board.get("slots", {}).get(part)
        height = geometry.get("parts", {}).get("heights_m", {}).get(part)
        target_z = board.get("target_origin_z_m", {}).get(part)
        if (not isinstance(slot, list) or len(slot) != 2 or not all(_number(value) for value in slot)
                or not all(_number(center.get(axis)) for axis in ("x", "y", "z"))
                or not _number(height) or height <= 0 or not _number(target_z)):
            return None
        declared = ProductProfile.geometry_for_part_from_geometry(part, geometry)
        reference = declared.get("target_reference")
        if not isinstance(reference, dict) or reference.get("surface_role") != "assembly_slot":
            return None
        return {**declared, "destination_location": destination,
                "target": geometry.get("parts", {}).get("assembly_target_map", {}).get(part),
                "simulation_mating_contact": deepcopy(board.get("mating_contact_by_part", {}).get(part))}
    node = scene.get(destination)
    if not isinstance(node, dict):
        node = next((row for row in scene.get("machines", []) if row.get("resource_id") == destination), None)
    if not isinstance(node, dict):
        return None
    poses = [node[name] for name in ("pickup_pose", "workholding_pose", "completed_product_pose") if name in node]
    if len(poses) != 1 or not isinstance(poses[0], list) or len(poses[0]) < 3 or not all(_number(v) for v in poses[0][:3]):
        return None
    height = geometry.get("parts", {}).get("heights_m", {}).get(part)
    if not _number(height):
        return None
    return {"destination_location": destination, "part_height_m": height,
            "target_origin_pose": dict(zip(("x", "y", "z"), poses[0][:3])),
            **({"support_surface_height": node["surface_height"]} if "surface_height" in node else {}),
            "target_reference": {"target_point": "part_origin", "surface_role": "support"}}


def _support_place_pose(*, part: str, destination: str, target: dict, physical_evidence: dict) -> dict:
    """Calculate face contact from recorded rigid geometry, without predicting settling.

    Align the uniquely nearest box face to the recorded horizontal support.
    The movement to this calculated target still requires an owner preview;
    neither the captured pose nor its attachment is changed here.
    """
    from cais_spade_llm.agents.shared_information.recovery_validation_protocol import recovery_validation_fingerprint
    from cais_spade_llm.recovery_framework.geometry import compose, multiply, rotate
    from cais_spade_llm.resources.primitive_observations_3d import _unit_pose
    from cais_spade_llm.resources.recovery_geometry import projected_part_support

    def pose(value, name):
        return [float(item) for item in _unit_pose(value, name)]

    snapshot = physical_evidence["snapshot"]
    physics = physical_evidence["physics_evidence"]
    gravity = physics["gravity"]
    if (not isinstance(gravity, list) or len(gravity) != 3 or not all(_number(value) for value in gravity)
            or gravity[0] != 0 or gravity[1] != 0 or gravity[2] >= 0):
        raise ValueError("Supported placement requires recorded vertical gravity")
    part_state = snapshot["parts"][part]
    body = physics["models"][part_state["model_name"]]
    if type(body.get("joint_count")) is not int or body["joint_count"] != 0 or len(body["links"]) != 1:
        raise ValueError("Supported placement requires one recorded rigid part link")
    link = next(iter(body["links"].values()))
    if len(link["collisions"]) != 1:
        raise ValueError("Supported placement requires one exact recorded part collision")
    collision = link["collisions"][0]
    shape = collision["geometry"]
    if (shape.get("type") != "box" or not isinstance(shape.get("size"), list) or len(shape["size"]) != 3
            or any(not _number(value) or value <= 0 for value in shape["size"])):
        raise ValueError("Supported placement requires recorded rigid box geometry")
    original = pose(body["pose"], "recorded part model pose")
    inverse_q = [-original[3], -original[4], -original[5], original[6]]
    inverse = [*rotate(inverse_q, [-value for value in original[:3]]), *inverse_q]
    local = compose(inverse, pose(collision["pose"], "recorded part collision pose"))
    current = pose(part_state["current_pose"], "current part pose")
    world_collision = compose(current, local)
    axes = [rotate(world_collision[3:], [float(i == axis) for i in range(3)]) for axis in range(3)]
    ranked = sorted(range(3), key=lambda axis: abs(axes[axis][2]), reverse=True)
    if abs(abs(axes[ranked[0]][2]) - abs(axes[ranked[1]][2])) <= 1e-12:
        raise ValueError("The closest part support face is ambiguous")
    axis = ranked[0]
    direction = 1. if axes[axis][2] > 0 else -1.
    normal = [direction * value for value in axes[axis]]
    correction = [normal[1], -normal[0], 0., 1. + normal[2]]
    length = math.sqrt(sum(value * value for value in correction))
    correction = [value / length for value in correction]
    target_q = multiply(correction, current[3:])
    vertices = [compose(local, [*corner, 0., 0., 0., 1.])[:3]
                for corner in product(*[(-size / 2, size / 2) for size in shape["size"]])]
    rotated = [rotate(target_q, vertex) for vertex in vertices]
    origin = target["target_origin_pose"]
    height = target.get("support_surface_height")
    declared_height = "support_surface_height" in target
    if declared_height and not _number(height):
        raise ValueError("The destination has no declared support surface height")
    source = physical_evidence["resources"][destination]["geometry_source"]
    receiver = physics["models"][source["model"]]
    if receiver.get("static") is not True:
        raise ValueError("Supported placement requires recorded static destination geometry")
    supports = []
    for receiver_link in receiver["links"].values():
        for surface in receiver_link["collisions"]:
            geometry = surface.get("geometry", {})
            if geometry.get("type") != "box":
                if not declared_height:
                    raise ValueError("Deriving support surface height requires complete recorded box geometry")
                continue
            sizes = geometry.get("size")
            if (not isinstance(sizes, list) or len(sizes) != 3
                    or any(not _number(value) or value <= 0 for value in sizes)):
                raise ValueError("Recorded support geometry has invalid dimensions")
            surface_pose = pose(surface["pose"], "recorded support pose")
            surface_axes = [rotate(surface_pose[3:], [float(i == selected) for i in range(3)])
                            for selected in range(3)]
            vertical = [i for i, value in enumerate(surface_axes) if abs(abs(value[2]) - 1.) <= 1e-12]
            if len(vertical) != 1:
                continue
            vertical_axis = vertical[0]
            top = surface_pose[2] + sizes[vertical_axis] / 2
            if declared_height and abs(top - height) > 1e-12:
                continue
            inverse_surface = [-surface_pose[3], -surface_pose[4], -surface_pose[5], surface_pose[6]]
            footprint = [rotate(inverse_surface, [origin["x"] + point[0] - surface_pose[0],
                        origin["y"] + point[1] - surface_pose[1], top - surface_pose[2]]) for point in rotated]
            if all(abs(point[i]) < sizes[i] / 2 for point in footprint for i in range(3) if i != vertical_axis):
                supports.append((surface, top))
    if len(supports) != 1:
        raise ValueError("The placement target requires one recorded support face covering the part")
    surface, calculated_height = supports[0]
    height_evidence = {}
    if not declared_height:
        height = calculated_height
        height_evidence = {"support_surface_height": height, "support_surface_height_source": {
            "kind": "calculated", "calculation": "uniquely covering recorded horizontal box face",
            "collision_id": surface["id"], "geometry_source": deepcopy(source),
            "receiver_model_fingerprint": recovery_validation_fingerprint(receiver),
            "destination_fingerprint": recovery_validation_fingerprint(target)}}
    target_pose = [origin["x"], origin["y"], height - min(point[2] for point in rotated), *target_q]
    proposed = deepcopy(snapshot)
    proposed["parts"][part]["current_pose"] = target_pose
    support = projected_part_support(part_name=part, state=proposed, evidence=physical_evidence)
    if support.get("status") != "supported":
        raise ValueError("Calculated placement lacks supported static equilibrium: " + str(support.get("reason")))
    return {"part_pose": target_pose, "support_collision_id": surface["id"],
            "part_collision_id": collision["id"], "part_face_axis": axis, "part_face_direction": -direction,
            "source": {"kind": "calculated", "calculation": "recorded rigid box face and static support contact",
                       "part_model_fingerprint": recovery_validation_fingerprint(body),
                       "receiver_model_fingerprint": recovery_validation_fingerprint(receiver),
                       "destination_fingerprint": recovery_validation_fingerprint(target),
                       "starting_part_pose": deepcopy(part_state["current_pose"])},
            "support_evidence": {key: value for key, value in support.items() if key != "physics_evidence"},
            "command_sent": False, "settling_modeled": False, **height_evidence}


def calculate_recovery_place_target(*, task: dict, recovery_snapshot: dict, resource_evidence: dict,
                                   physical_evidence: dict | None = None) -> dict:
    """Calculate a declared placement target with explicit recorded inputs.

    The returned pose is a calculation, never an observation or a claim about
    the intervening motion. Frame/attachment and trajectory coverage are checked
    by the separate event projection owner.
    """
    jid = task.get("resource_jid", "")
    root = f"/resources/{jid}/resource_evidence"
    destination = (task.get("expected_end_state") or {}).get("resource_location")
    part = task.get("part_name")
    context = (resource_evidence.get("state") or {}).get("_task_ctx") or {}
    target = _destination_geometry(resource_evidence, destination, part)
    missing = []
    if target is None:
        missing.append({"ref": root + "/inputs", "reason": "The proposed destination has no recorded placement geometry"})
    if "part_name" in context and context["part_name"] != part:
        context = {}
    current = _pose_mapping(recovery_snapshot.get("current_pose"))
    recorded = _pose_mapping((resource_evidence.get("snapshot") or {}).get("current_pose"))
    if all(_number(current.get(axis)) and _number(recorded.get(axis)) and abs(current[axis] - recorded[axis]) <= 1e-6
           for axis in ("x", "y", "z")):
        for key in ("qx", "qy", "qz", "qw"):
            if key not in current and key in recorded:
                current[key] = recorded[key]
    orientation = {key: current[key] for key in ("qx", "qy", "qz", "qw") if key in current}
    if len(orientation) != 4 or not all(_number(value) for value in orientation.values()) or not .999 <= sum(v*v for v in orientation.values()) <= 1.001:
        missing.append({"ref": root + "/snapshot/current_pose", "reason": "Placement calculation requires a recorded tool orientation"})
    attachment = resource_evidence.get("attachment") or {}
    transform = attachment.get("grasp_transform")
    projected_attachment = (attachment.get("projected_attachment_complete") is True
                            and attachment.get("source", {}).get("kind") == "projected"
                            and bool(attachment.get("source", {}).get("contract_fingerprint")))
    supported_transform = ((attachment.get("observed_attachment_complete") is True or projected_attachment)
        and attachment.get("part_name") == part
        and isinstance(transform, list) and len(transform) == 7 and all(_number(value) for value in transform)
        and .999 <= sum(value * value for value in transform[3:]) <= 1.001)
    if not supported_transform and target and target.get("target_reference", {}).get("surface_role") == "assembly_slot":
        missing.append({"ref": root + "/attachment", "reason": "Assembly placement requires the recorded full held-part transform"})
    if not supported_transform:
        for key in ("pick_tcp_z", "tz", "tcp_offset_z"):
            if not _number(context.get(key)):
                missing.append({"ref": root + "/state/_task_ctx/" + key, "reason": "Placement calculation requires the recorded grasp transform"})
    record = {"status": "NEEDS_CONTEXT" if missing else "calculated", "missing_facts": missing,
              "context_requests": [row["ref"] for row in missing], "command_sent": False,
              "destination_location": destination, "part_name": part,
              "source": {"kind": "calculated", "calculation": "controlled_link_height, placement_poses",
                         "inputs": [root + "/inputs", root + "/state/_task_ctx", root + "/snapshot/current_pose"]}}
    if missing:
        return record
    pose = target["target_origin_pose"]
    if not isinstance(pose, dict) or not all(_number(pose.get(axis)) for axis in ("x", "y", "z")):
        return {**record, "status": "NEEDS_CONTEXT", "missing_facts": [{"ref": root + "/inputs", "reason": "Recorded placement target is invalid"}], "context_requests": [root + "/inputs"]}
    fields = {"target": target.get("target"), "target_reference": deepcopy(target["target_reference"]),
              "part_height": target["part_height_m"], "destination_location": destination,
              "target_origin_pose": deepcopy(pose)}
    if target["target_reference"].get("surface_role") == "assembly_slot":
        fields.update(slot_x=pose["x"], slot_y=pose["y"], board_top_z=target.get("slot_floor_z_m"),
                      place_part_origin_z=pose["z"])
    if supported_transform:
        from cais_spade_llm.recovery_framework.geometry import multiply, rotate

        mating = target.get("simulation_mating_contact")
        if target["target_reference"].get("surface_role") == "support" and physical_evidence:
            try:
                placement = _support_place_pose(part=part, destination=destination, target=target,
                                                 physical_evidence=physical_evidence)
            except (KeyError, ValueError, TypeError, IndexError) as exc:
                return {**record, "status": "NEEDS_CONTEXT", "context_requests": ["/physical_evidence"],
                        "missing_facts": [{"ref": "/physical_evidence", "reason": str(exc)}]}
            values = placement["part_pose"]
            pose = dict(zip(("x", "y", "z"), values[:3], strict=True))
            relative = transform[3:]
            tool_q = multiply(values[3:], [-relative[0], -relative[1], -relative[2], relative[3]])
            orientation = dict(zip(("qx", "qy", "qz", "qw"), tool_q, strict=True))
            fields["support_placement"] = placement
        elif target["target_reference"].get("surface_role") == "assembly_slot" and mating is not None:
            yaw = mating.get("target_yaw_rad") if isinstance(mating, dict) else None
            if resource_evidence.get("execution_mode") != "simulation" or not _number(yaw):
                return {**record, "status": "NEEDS_CONTEXT", "context_requests": [root + "/inputs/geometry/assembly_board/mating_contact_by_part/" + str(part)],
                    "missing_facts": [{"ref": root + "/inputs/geometry/assembly_board/mating_contact_by_part/" + str(part),
                                       "reason": "Declared simulation mating orientation is unavailable"}]}
            relative = transform[3:]
            target_q = [0., 0., math.sin(yaw / 2), math.cos(yaw / 2)]
            tool_q = multiply(target_q, [-relative[0], -relative[1], -relative[2], relative[3]])
            orientation = dict(zip(("qx", "qy", "qz", "qw"), tool_q, strict=True))
            fields["simulation_mating_contact"] = deepcopy(mating)
        elif target["target_reference"].get("surface_role") == "assembly_slot" and "place_tool_yaw_offset_rad" in target:
            yaw = target["place_tool_yaw_offset_rad"]
            if not _number(yaw):
                return {**record, "status": "NEEDS_CONTEXT", "context_requests": [root + "/inputs/geometry/assembly_board/place_tool_yaw_offset_rad_by_part"],
                    "missing_facts": [{"ref": root + "/inputs/geometry/assembly_board/place_tool_yaw_offset_rad_by_part", "reason": "Declared tool yaw offset is invalid"}]}
            tool_q = multiply([0., 0., math.sin(yaw / 2), math.cos(yaw / 2)], [orientation[key] for key in ("qx", "qy", "qz", "qw")])
            orientation = dict(zip(("qx", "qy", "qz", "qw"), tool_q, strict=True))
        offset = rotate([orientation[key] for key in ("qx", "qy", "qz", "qw")], transform[:3])
        return {**record, **fields, "insert_pose": {**{axis: pose[axis] - offset[index] for index, axis in enumerate(("x", "y", "z"))},
                                          **orientation}, "target_origin_pose": deepcopy(pose),
                "target_reference": deepcopy(target["target_reference"]), "grasp_transform": deepcopy(transform),
                "source": {"kind": "calculated", "calculation": "attachment transform at supported orientation",
                           "attachment_source": deepcopy(attachment.get("source")),
                           "inputs": [root + "/attachment", root + "/snapshot/current_pose", root + "/inputs"]}}
    place_z = controlled_link_height(pose["z"] + context["pick_tcp_z"] - context["tz"], context["tcp_offset_z"], 0.)
    return {**record, **fields, "insert_pose": placement_poses(pose["x"], pose["y"], place_z, orientation,
                simulation_assembly_slot=False, insertion_depth=0.)["insert_pose"],
            "target_origin_pose": deepcopy(pose), "target_reference": deepcopy(target["target_reference"])}


def calculate_recovery_pick_target(*, task: dict, recovery_snapshot: dict, resource_evidence: dict) -> dict:
    """Calculate a supported simulation pick from recorded observation and policy.

    All geometry, offsets and policy choices must be recorded. Physical CAD
    grasp qualification remains an owning-resource obligation outside this
    vertical simulation calculation.
    """
    root = f"/resources/{task.get('resource_jid', '')}/resource_evidence"
    part = task.get("part_name")
    observed = (resource_evidence.get("observed_parts") or {}).get(part) or {}
    pose = _pose_mapping(observed.get("observed_pose", observed.get("current_pose")))
    current = _pose_mapping(recovery_snapshot.get("current_pose"))
    captured_current = _pose_mapping((resource_evidence.get("snapshot") or {}).get("current_pose"))
    if all(_number(current.get(axis)) and _number(captured_current.get(axis))
           and abs(current[axis] - captured_current[axis]) <= 1e-6 for axis in ("x", "y", "z")):
        for axis in ("qx", "qy", "qz", "qw"):
            if axis not in current and axis in captured_current:
                current[axis] = captured_current[axis]
    motion = resource_evidence.get("motion_config") or resource_evidence.get("controller_config", {}).get("motion") or {}
    geometry = (resource_evidence.get("pick_geometry") or {}).get(part) or {}
    policy = (resource_evidence.get("pick_policy") or {}).get(part) or {}
    offset_record = resource_evidence.get("tcp_observation") or {}
    offset = None
    offset_ref = "/tcp_observation/tcp_offset_z"
    recorded_pose = resource_evidence.get("recorded_snapshot", resource_evidence.get("snapshot")) or {}
    paired = _pose_mapping(offset_record.get("current_pose", recorded_pose.get("current_pose")))
    matching_orientation = (all(_number(current.get(axis)) and _number(paired.get(axis))
                                for axis in ("qx", "qy", "qz", "qw"))
        and abs(sum(current[key] * paired[key] for key in ("qx", "qy", "qz", "qw"))) >= 1. - 1e-6
        and .999 <= sum(current[key] ** 2 for key in ("qx", "qy", "qz", "qw")) <= 1.001
        and .999 <= sum(paired[key] ** 2 for key in ("qx", "qy", "qz", "qw")) <= 1.001)
    transform = motion.get("tcp_transform")
    if transform is not None:
        offset_ref = "/motion_config/tcp_transform"
        valid = (isinstance(transform, dict) and transform.get("fixed") is True
                 and transform.get("parent_frame") == motion.get("controlled_link")
                 and transform.get("child_frame") == motion.get("tcp_link")
                 and bool(transform.get("source"))
                 and isinstance(transform.get("translation"), list) and len(transform["translation"]) == 3
                 and all(_number(value) for value in transform["translation"])
                 and isinstance(transform.get("rotation"), list) and len(transform["rotation"]) == 4
                 and all(_number(value) for value in transform["rotation"])
                 and .999 <= sum(value * value for value in transform["rotation"]) <= 1.001
                 and all(_number(current.get(key)) for key in ("qx", "qy", "qz", "qw"))
                 and .999 <= sum(current[key] ** 2 for key in ("qx", "qy", "qz", "qw")) <= 1.001)
        if valid:
            from cais_spade_llm.recovery_framework.geometry import rotate

            offset = rotate([current[key] for key in ("qx", "qy", "qz", "qw")], transform["translation"])[2]
    else:
        if matching_orientation:
            offset = offset_record.get("tcp_offset_z")
        if offset is None and motion.get("tcp_transform_source") and motion.get("controlled_link") and motion.get("tcp_link"):
            if matching_orientation:
                offset = motion.get("tcp_offset_z")
                offset_ref = "/motion_config/tcp_offset_z"
    missing = []

    def need(condition, ref, reason):
        if not condition:
            missing.append({"ref": root + ref, "reason": reason})

    need(resource_evidence.get("execution_mode") == "simulation", "/execution_mode", "This calculation requires recorded simulation execution_mode")
    need(observed.get("frame", resource_evidence.get("frame")) == "world" and all(_number(pose.get(axis)) for axis in ("x", "y", "z")),
         "/observed_parts/" + str(part), "Current world-frame part observation is required")
    need(all(_number(current.get(axis)) for axis in ("x", "y", "z", "qx", "qy", "qz", "qw")),
         "/snapshot/current_pose", "Current tool pose and orientation are required")
    need(_number(offset), offset_ref, "A recorded fixed controlled-link/TCP transform or matching observed height offset is required")
    for key in ("pick_tcp_z_bias_min_m", "pick_tcp_z_bias_max_m", "approach_height_m"):
        need(_number(motion.get(key)), "/motion_config/" + key, "Recorded vertical pick policy is required")
    for key in ("surface_clearance_override_m", "pick_z_adjustment_m"):
        need(_number(policy.get(key)), "/pick_policy/" + str(part) + "/" + key, "Explicit pick policy parameter is required")
    need(type(policy.get("use_global_min_pick_tcp_z")) is bool, "/pick_policy/" + str(part) + "/use_global_min_pick_tcp_z", "Explicit TCP minimum policy is required")
    if policy.get("use_global_min_pick_tcp_z") is True:
        need(_number(motion.get("min_pick_tcp_z_m")), "/motion_config/min_pick_tcp_z_m", "Recorded TCP minimum height is required")
    need(_number(geometry.get("part_height_m")) and geometry.get("part_height_m", 0) > 0,
         "/pick_geometry/" + str(part) + "/part_height_m", "Recorded supported vertical part height is required")
    need(_number(geometry.get("board_center", {}).get("z")), "/pick_geometry/" + str(part) + "/board_center/z", "Recorded source support height is required")
    record = {"status": "NEEDS_CONTEXT" if missing else "calculated", "missing_facts": missing,
              "context_requests": [row["ref"] for row in missing], "part_name": part, "command_sent": False,
              "source": {"kind": "calculated", "calculation": "vertical_pick_bias, controlled_link_height, pick_travel_height",
                         "inputs": [root + "/observed_parts", root + "/pick_geometry", root + "/pick_policy",
                                    root + "/motion_config", root + offset_ref, root + "/snapshot"]}}
    if missing:
        return record
    handling_access = geometry.get("handling_robot_access") or {}
    if handling_access:
        return {**record, "status": "NEEDS_CONTEXT", "context_requests": [root + "/pick_geometry/" + str(part) + "/handling_robot_access"],
                "missing_facts": [{"ref": root + "/pick_geometry/" + str(part) + "/handling_robot_access",
                    "reason": "Recorded handling_robot_access requires an owning-resource access calculation"}]}
    bias = vertical_pick_bias(geometry["part_height_m"], motion["pick_tcp_z_bias_min_m"], motion["pick_tcp_z_bias_max_m"])
    pick_tcp_z = pose["z"] + bias + max(0., policy["surface_clearance_override_m"])
    if policy["use_global_min_pick_tcp_z"]:
        pick_tcp_z = max(pick_tcp_z, motion["min_pick_tcp_z_m"])
    pick_z = controlled_link_height(pick_tcp_z, offset, policy["pick_z_adjustment_m"])
    travel_z = pick_travel_height(pose["z"], geometry["board_center"]["z"], pick_z, motion["approach_height_m"], current["z"])
    orientation = {key: current[key] for key in ("qx", "qy", "qz", "qw")}
    return {**record, "target_pose": {"x": pose["x"], "y": pose["y"], "z": pick_z, **orientation},
            "approach_pose": {"x": pose["x"], "y": pose["y"], "z": travel_z, **orientation},
            "observed_pose": pose, "pick_tcp_z": pick_tcp_z, "tcp_offset_z": offset, "travel_z": travel_z,
            "tx": pose["x"], "ty": pose["y"], "tz": pose["z"], "pick_z": pick_z,
            "part_height": geometry["part_height_m"], "access_retreat_pose": None}


def evaluate_robot_primitive_predicate(*, predicate: str, resource_evidence: dict,
                                       recovery_snapshot: dict, resource_jid: str) -> dict:
    """Read declared robot guard facts without executing its primitive program."""
    root = f"/resources/{resource_jid}/resource_evidence"
    context = (resource_evidence.get("state") or {}).get("_task_ctx") or {}
    if predicate in {"held_part_empty", "held_part_exists"}:
        if "held_part" not in recovery_snapshot:
            return {"value": None, "context_requests": [root + "/snapshot/held_part"]}
        held = recovery_snapshot["held_part"]
        return {"value": (held is None) if predicate == "held_part_empty" else held is not None}
    if predicate == "move_insert_required":
        mode = resource_evidence.get("execution_mode")
        if mode == "simulation":
            if "surface_role" not in context:
                return {"value": None, "context_requests": [root + "/state/_task_ctx/surface_role"]}
            if context["surface_role"] != "assembly_slot":
                return {"value": False}
            if "move_insert_mode" in context:
                return {"value": context["move_insert_mode"] == "simulation_direct"}
            return {"value": None, "context_requests": [root + "/state/_task_ctx/move_insert_mode"]}
        if mode == "physical":
            if "move_insert_profile" in context and "move_insert_mode" in context:
                return {"value": bool(context["move_insert_profile"]) and context["move_insert_mode"] == "force_limited"}
            return {"value": None, "context_requests": [root + "/state/_task_ctx/move_insert_profile", root + "/state/_task_ctx/move_insert_mode"]}
        return {"value": None, "context_requests": [root + "/execution_mode"]}
    return {"value": None, "context_requests": [f"/resources/{resource_jid}/recovery_des_model"]}


def prepare_robot_primitive_parameters(
    *, task: dict, recovery_des_model: dict, recovery_snapshot: dict,
    resource_evidence: dict, resource_jid: str, physical_evidence: dict | None = None,
) -> dict:
    """Resolve robot fields locally while preserving all source observations."""
    evidence = deepcopy(resource_evidence)
    root = f"/resources/{resource_jid}/resource_evidence"
    state = evidence.setdefault("state", {})
    sources = evidence.setdefault("sources", {})
    destination = (task.get("expected_end_state") or {}).get("resource_location")
    part = task.get("part_name")
    applicability = evidence.setdefault("parameter_applicability", {})
    if destination is not None and destination != "assembly_board-v1":
        applicability.setdefault("release_part", {})["assembly_slot"] = {
            "applicable": False, "reason": "release_part permits assembly_slot only for destination_location='assembly_board-v1'"}
    task_ctx = state.get("_task_ctx")
    if task_ctx is None and "_task_ctx" not in state:
        task_ctx = state["_task_ctx"] = {}
    missing = []
    support = derive_primitive_union(task=task, recovery_des_model=recovery_des_model, recovery_snapshot=recovery_snapshot)
    context_fields, step_outputs = set(), set()

    def references(value):
        if isinstance(value, dict):
            if (value.get("$state") == "_task_ctx" and isinstance(value.get("path"), list)
                    and value["path"] and isinstance(value["path"][0], str)):
                context_fields.add(value["path"][0])
            if isinstance(value.get("$step"), str):
                step_outputs.add(value["$step"])
            if isinstance(value.get("field"), str) and value["field"].startswith("task_ctx."):
                context_fields.add(value["field"].split(".", 1)[1])
            for item in value.values():
                references(item)
        elif isinstance(value, list):
            for item in value:
                references(item)

    for primitive in support["primitives"]:
        for contributor in primitive["contributors"]:
            references(contributor["step"])
    if isinstance(task_ctx, dict):
        if "part_name" in task_ctx and task_ctx["part_name"] != part:
            evidence["parameter_context_exclusions"] = [{
                "source": root + "/state/_task_ctx", "recorded": deepcopy(task_ctx),
                "part_name": part, "reason": "Recorded task context belongs to another part"}]
            task_ctx = state["_task_ctx"] = {}
            for name in step_outputs:
                if name in evidence.get("step_outputs", {}):
                    evidence.setdefault("excluded_step_outputs", {})[name] = evidence["step_outputs"].pop(name)
        model_name = (evidence.get("inputs") or {}).get("geometry", {}).get("parts", {}).get("model_map", {}).get(part)
        if "model_name" in context_fields and "model_name" not in task_ctx and isinstance(model_name, str) and model_name:
            task_ctx["model_name"] = model_name
            sources["state/_task_ctx/model_name"] = root + "/inputs/geometry/parts/model_map/" + str(part)
        elif "model_name" in context_fields and "model_name" not in task_ctx:
            missing.append({"ref": root + "/inputs/geometry/parts/model_map/" + str(part),
                            "reason": "Requested part has no recorded model identity"})
        needs_pick = ("pick_targets" in step_outputs or "access_retreat_pose" in context_fields
                      or "travel_z" in context_fields and "travel_z" not in task_ctx)
        if needs_pick:
            calculated = calculate_recovery_pick_target(task=task, recovery_snapshot=recovery_snapshot, resource_evidence=evidence)
            if calculated["status"] == "calculated":
                evidence.setdefault("step_outputs", {})["pick_targets"] = deepcopy(calculated)
                sources["step_outputs/pick_targets"] = deepcopy(calculated["source"])
                for name in ("access_retreat_pose", "travel_z", "tx", "ty", "tz", "pick_z", "pick_tcp_z", "tcp_offset_z", "part_height"):
                    if name not in task_ctx:
                        task_ctx[name] = deepcopy(calculated[name])
                        sources["state/_task_ctx/" + name] = deepcopy(calculated["source"])
            else:
                missing.extend(calculated["missing_facts"])
        gripper = evidence.get("controller_config", {}).get("gripper") or evidence.get("gripper_config") or {}
        geometry = (evidence.get("pick_geometry") or {}).get(part) or {}
        width = next((geometry[key] for key in ("grasp_width_m", "part_width_m", "part_diameter_m", "diameter_m", "width_m")
                      if _number(geometry.get(key)) and geometry[key] > 0), None)
        if ("gripper_close_position" in context_fields and "gripper_close_position" not in task_ctx and width is not None
                and all(_number(gripper.get(key)) for key in ("open", "close"))
                and 0 <= gripper["close"] < gripper["open"] <= .25):
            task_ctx["gripper_close_position"] = max(gripper["close"], min(gripper["open"], width))
            sources["state/_task_ctx/gripper_close_position"] = {
                "kind": "calculated", "calculation": "GazeboPickPlaceController._derive_gripper_close_position",
                "inputs": [root + "/pick_geometry/" + str(part), root + "/controller_config/gripper"]}
        if destination is not None and context_fields & {"surface_role", "move_insert_mode", "insert_pose",
                "pre_insert_pose", "slot_x", "slot_y", "board_top_z", "place_part_origin_z"}:
            target = _destination_geometry(evidence, destination, part)
            if target is not None and evidence.get("execution_mode") == "simulation":
                task_ctx["surface_role"] = target["target_reference"]["surface_role"]
                task_ctx["move_insert_mode"] = "simulation_direct" if task_ctx["surface_role"] == "assembly_slot" else ""
                sources["state/_task_ctx/surface_role"] = root + "/inputs"
                sources["state/_task_ctx/move_insert_mode"] = "GazeboPickPlaceController.compute_place_targets simulation policy"
            changed = task_ctx.get("destination_location") != destination
            if changed:
                for name in ("insert_pose", "pre_insert_pose", "slot_x", "slot_y", "board_top_z", "place_part_origin_z"):
                    task_ctx.pop(name, None)
            needs_pose = "insert_pose" in context_fields
            if "insert_pose" not in task_ctx and needs_pose:
                calculated = calculate_recovery_place_target(task=task, recovery_snapshot=recovery_snapshot,
                    resource_evidence=evidence, physical_evidence=physical_evidence)
                missing.extend(calculated["missing_facts"])
                if calculated["status"] == "calculated":
                    for name in ("insert_pose", "slot_x", "slot_y", "part_height", "board_top_z", "place_part_origin_z"):
                        if name in calculated:
                            task_ctx[name] = deepcopy(calculated[name])
                            sources["state/_task_ctx/" + name] = deepcopy(calculated["source"])
    result = prepare_resource_primitive_parameters(task=task, recovery_des_model=recovery_des_model,
        recovery_snapshot=recovery_snapshot, resource_evidence=evidence, resource_jid=resource_jid)
    calculations, calculation_missing = _place_step_outputs(task=task, support=support,
        recovery_snapshot=recovery_snapshot, recovery_des_model=recovery_des_model,
        resource_evidence=evidence, physical_evidence=physical_evidence,
        resource_jid=resource_jid, preparation=result, requested_outputs=step_outputs)
    if calculations:
        result = prepare_resource_primitive_parameters(task=task, recovery_des_model=recovery_des_model,
            recovery_snapshot=recovery_snapshot, resource_evidence=evidence, resource_jid=resource_jid,
            calculated_step_outputs=calculations)
    missing.extend(calculation_missing)
    result["missing_facts"].extend(missing)
    result["context_requests"] = list(dict.fromkeys([*result["context_requests"], *(row["ref"] for row in missing)]))
    result["conflicts"] = [*deepcopy(evidence.get("conflicts") or []), *result.get("conflicts", [])]
    return result


def _recovery_place_step_output(*, task: dict, recovery_snapshot: dict,
                                resource_evidence: dict, physical_evidence: dict,
                                params: dict) -> dict:
    """Calculate declared placement pose outputs without applying task effects."""
    from types import SimpleNamespace
    from cais_spade_llm.recovery_framework.workflow_execution import _destination_geometry as argument_geometry

    required = {"pick_ctx", "part_name", "z_adjustment_m", "destination_location", "assembly_board_v1_aruco"}
    if (resource_evidence.get("execution_mode") != "simulation"
            or not required <= params.keys() or params.keys() - required - {"product_geometry"}
            or params["part_name"] != task.get("part_name")
            or params["destination_location"] != task.get("expected_end_state", {}).get("resource_location")
            or not _number(params["z_adjustment_m"]) or params["z_adjustment_m"] != 0.
            or params["pick_ctx"] != resource_evidence.get("state", {}).get("_task_ctx")
            or params["assembly_board_v1_aruco"] is not None):
        raise ValueError("Placement arguments differ from the supported recorded calculation")
    inputs = resource_evidence.get("inputs", {})
    parts = inputs.get("geometry", {}).get("parts", {})
    part = task["part_name"]
    dimensions, model = parts.get("dimensions_m", {}).get(part), parts.get("model_map", {}).get(part)
    if (not isinstance(dimensions, (list, tuple)) or len(dimensions) != 3
            or any(not _number(value) or value <= 0 for value in dimensions)
            or not isinstance(model, str) or not model):
        raise ValueError("Placement geometry lacks recorded dimensions or model identity")
    # None and omission select the recorded destination calculation; an explicit
    # override must agree with that destination's captured argument geometry.
    if params.get("product_geometry") is not None:
        owners = [name for name, jid in physical_evidence.get("jids", {}).items() if jid == task["resource_jid"]]
        if len(owners) != 1:
            raise ValueError("Placement geometry has no exact recorded owner")
        recorded_geometry = argument_geometry(SimpleNamespace(inputs=inputs,
            product_name=physical_evidence.get("product_name"),
            geometry={part: {"dimensions_m": dimensions, "model_name": model}}),
            part, params["destination_location"], owners[0])
        if params["product_geometry"] != recorded_geometry:
            raise ValueError("product_geometry differs from its recorded owner geometry")
    target = calculate_recovery_place_target(task=task, recovery_snapshot=recovery_snapshot,
        resource_evidence=resource_evidence, physical_evidence=physical_evidence)
    if target.get("status") != "calculated":
        raise ValueError("Placement calculation has missing source facts: " + str(target.get("missing_facts")))
    depth = resource_evidence.get("controller_config", {}).get("parts_tuning", {}).get("insertion_depth_m")
    if not _number(depth) or depth < 0:
        raise ValueError("Placement requires its recorded insertion depth")
    pose = target["insert_pose"]
    poses = placement_poses(pose["x"], pose["y"], pose["z"],
        {key: pose[key] for key in ("qx", "qy", "qz", "qw")},
        simulation_assembly_slot=target["target_reference"]["surface_role"] == "assembly_slot",
        insertion_depth=depth)
    return {"output": poses, "target_calculation": target,
        "source": {"kind": "calculated", "calculation": "calculate_recovery_place_target, placement_poses",
                   "branch": "simulation", "source": deepcopy(resource_evidence.get("source")),
                   "target_source": deepcopy(target["source"])}}


def _place_step_outputs(*, task: dict, support: dict, recovery_snapshot: dict,
                       recovery_des_model: dict, resource_evidence: dict,
                       physical_evidence: dict | None, resource_jid: str,
                       preparation: dict, requested_outputs: set[str]) -> tuple[list[dict], list[dict]]:
    """Evaluate declared placement calculations separately for each contributor.

    ``store_as`` supplies the recorded dependency name. No step is dispatched,
    and no state effect or preceding movement is applied to the starting facts.
    """
    from cais_spade_llm.agents.shared_information.recovery_validation_protocol import recovery_validation_fingerprint
    from cais_spade_llm.resources.recovery_parameter_evidence import evaluate_resource_primitive_condition
    from cais_spade_llm.resources.resource_safety_preparation import _outline_nominal_place_targets

    root = f"/resources/{resource_jid}/resource_evidence"
    calculations, missing = [], []
    for primitive in support["primitives"]:
        if primitive["primitive"] != "compute_place_targets":
            continue
        for contributor in primitive["contributors"]:
            bindings, step = contributor["parameter_bindings"], contributor["step"]
            programs = [transition["composition_contract"] for transition in support["matching_transitions"]
                if transition["event_name"] == contributor["event_name"]
                and transition["parameter_bindings"] == bindings]
            owners = [name for name, jid in (physical_evidence or {}).get("jids", {}).items()
                      if jid == resource_jid]
            if len(owners) == 1 and resource_evidence.get("execution_mode") == "simulation":
                function = resource_evidence.get("inputs", {}).get("scene", {}).get(
                    "resource_programs", {}).get("resources", {}).get(owners[0], {}).get(
                        "functions", {}).get(contributor["event_name"], {})
                if function.get("status") == "implemented" and isinstance(function.get("program"), dict):
                    programs.append(function["program"])
            declarations = [row for program in programs for row in program.get("steps", [])
                if row.get("op") == "compute_place_targets" and row.get("params", {}) == step.get("params", {})
                and row.get("when", []) == step.get("when", [])
                and isinstance(row.get("store_as"), str) and row["store_as"] in requested_outputs]
            if not declarations:
                continue
            condition = evaluate_resource_primitive_condition(step=step, bindings=bindings,
                snapshot=recovery_snapshot, resource_evidence=resource_evidence, resource_jid=resource_jid)
            if condition["status"] != "applicable":
                continue
            witnesses = [row for row in preparation["parameter_witnesses"]
                if row["primitive"] == "compute_place_targets" and row["event_name"] == contributor["event_name"]
                and row["step_index"] == contributor["step_index"] and row["parameter_bindings"] == bindings]
            if len(witnesses) != 1:
                continue
            try:
                calculate = _outline_nominal_place_targets if task.get("required_task_contract") else _recovery_place_step_output
                calculated = calculate(task=task, recovery_snapshot=recovery_snapshot,
                    resource_evidence=resource_evidence, physical_evidence=physical_evidence or {},
                    params=witnesses[0]["params"])
            except (KeyError, ValueError, TypeError) as exc:
                missing.append({"ref": root + "/inputs", "primitive": "compute_place_targets",
                    "event_name": contributor["event_name"], "step_index": contributor["step_index"],
                    "reason": "Recorded placement output cannot be calculated: " + str(exc)})
                continue
            for declaration in declarations:
                calculations.append({"event_name": contributor["event_name"],
                    "parameter_bindings": deepcopy(bindings), "step_index": contributor["step_index"],
                    "store_as": declaration["store_as"], "output": deepcopy(calculated["output"]),
                    "params": deepcopy(witnesses[0]["params"]), "source": deepcopy(calculated["source"]),
                    "source_capture": deepcopy(resource_evidence.get("source")),
                    "declaration_fingerprint": recovery_validation_fingerprint(declaration),
                    "snapshot_fingerprint": recovery_validation_fingerprint(recovery_snapshot),
                    "model_fingerprint": recovery_validation_fingerprint(recovery_des_model),
                    "source_fingerprint": recovery_validation_fingerprint(resource_evidence),
                    "command_sent": False})
    return calculations, missing


def check_robot_primitive_capability(
    *, primitive: str, params: dict, task: dict, recovery_snapshot: dict,
    resource_evidence: dict, static_capabilities: dict, primitive_catalog: list, resource_jid: str,
    part_context: dict | None = None, physical_evidence: dict | None = None,
) -> dict | None:
    """Check recorded robot capability without controller or perception calls."""
    root = f"/resources/{resource_jid}/resource_evidence"
    evidence = {"primitive": primitive, "params": deepcopy(params), "command_sent": False,
                "scope": "Independent capability; no ordering or physical projection established",
                "source": deepcopy(resource_evidence.get("source") or {})}

    def result(status, reason, *, refs=(), **facts):
        return {"allowed": status == "FEASIBLE", "feasibility_status": status, "reason": reason,
                "context_requests": list(refs), "evidence": {**evidence, **deepcopy(facts)}}

    conflicts = resource_evidence.get("conflicts") or []
    if conflicts:
        return result("NEEDS_CONTEXT", "Recorded resource evidence contains conflicting facts",
                      refs=[row["ref"] for row in conflicts if isinstance(row, dict) and "ref" in row], conflicts=conflicts)
    if primitive == "get_current_pose":
        pose = _pose_mapping(recovery_snapshot.get("current_pose"))
        if all(_number(pose.get(axis)) for axis in ("x", "y", "z")):
            return result("FEASIBLE", "Recorded owner pose is available", current_pose=pose)
        return result("NEEDS_CONTEXT", "Recorded owner pose is unavailable", refs=[root + "/snapshot/current_pose"])
    if primitive == "move_to_named_pose":
        declarations = [row["named_poses"] for row in (recovery_snapshot, static_capabilities) if "named_poses" in row]
        if (not declarations or any(not isinstance(row, (dict, list)) or not row for row in declarations)
                or any(row != declarations[0] for row in declarations[1:])):
            return result("NEEDS_CONTEXT", "Recorded named_poses declarations are missing or conflict", refs=[root + "/snapshot/named_poses"])
        if recovery_snapshot.get("controller_ready") is False:
            return result("INFEASIBLE", "Recorded controller_ready is false", controller_ready=False)
        if recovery_snapshot.get("controller_ready") is not True:
            return result("NEEDS_CONTEXT", "Recorded controller readiness is unavailable", refs=[root + "/snapshot/controller_ready"])
        return result("FEASIBLE" if params.get("pose_name") in declarations[0] else "INFEASIBLE",
                      "Checked pose_name against recorded owner declarations", named_poses=declarations[0])
    if primitive in {"grasp_part", "open_gripper", "close_gripper"}:
        if recovery_snapshot.get("controller_ready") is False:
            return result("INFEASIBLE", "Recorded controller_ready is false", controller_ready=False)
        if recovery_snapshot.get("controller_ready") is not True:
            return result("NEEDS_CONTEXT", "Recorded controller readiness is unavailable", refs=[root + "/snapshot/controller_ready"])
        gripper = resource_evidence.get("controller_config", {}).get("gripper") or resource_evidence.get("gripper_config") or {}
        if not all(_number(gripper.get(key)) for key in ("open", "close")) or not 0 <= gripper["close"] < gripper["open"] <= .25:
            return result("NEEDS_CONTEXT", "Recorded gripper open/close capability is unavailable", refs=[root + "/controller_config/gripper"])
        if primitive != "grasp_part":
            return result("FEASIBLE", "Recorded gripper configuration supports this independent capability",
                          gripper=gripper, composition_obligations=["custody and contact at execution"])
        if resource_evidence.get("execution_mode") != "simulation":
            return result("NEEDS_CONTEXT", "Independent physical grasp qualification is unavailable", refs=[root + "/execution_mode"])
        part = params.get("part_name", task.get("part_name"))
        target_model = (resource_evidence.get("inputs") or {}).get("geometry", {}).get("parts", {}).get("model_map", {}).get(part)
        if not target_model:
            return result("NEEDS_CONTEXT", "Recorded part model identity is unavailable", refs=[root + "/inputs/geometry/parts/model_map"])
        if params.get("model_name") != target_model:
            return result("INFEASIBLE", "grasp_part model identity conflicts with the recorded part", part_name=part)
        if "held_part" not in recovery_snapshot:
            return result("NEEDS_CONTEXT", "Recorded held_part is missing", refs=[root + "/snapshot/held_part"])
        if recovery_snapshot["held_part"] not in (None, part):
            return result("INFEASIBLE", "Recorded gripper custody conflicts with grasp_part", held_part=recovery_snapshot["held_part"])
        context = part_context or {}
        if "part_holder_resource_jid" not in context:
            return result("NEEDS_CONTEXT", "Recorded target-part custody is unavailable", refs=[f"/parts/{part}/observations"])
        if context["part_holder_resource_jid"] not in (None, resource_jid):
            return result("INFEASIBLE", "Another resource holds the target part", holder=context["part_holder_resource_jid"])
        attach = resource_evidence.get("controller_config", {}).get("attach") or {}
        if not attach.get("robot_model_name") or not attach.get("primary_attach_link") or not resource_evidence.get("controller_config", {}).get("services", {}).get("attach"):
            return result("NEEDS_CONTEXT", "Recorded simulation attachment interface is unavailable", refs=[root + "/controller_config/attach"])
        geometry = (resource_evidence.get("pick_geometry") or {}).get(part) or {}
        width = next((geometry[key] for key in ("grasp_width_m", "part_width_m", "part_diameter_m", "diameter_m", "width_m")
                      if _number(geometry.get(key)) and geometry[key] > 0), None)
        if width is None:
            return result("NEEDS_CONTEXT", "Recorded grasp width is unavailable", refs=[root + "/pick_geometry/" + str(part)])
        if width > gripper["open"]:
            return result("INFEASIBLE", "Recorded part width exceeds gripper opening", width=width, gripper=gripper)
        position = params.get("position")
        if position is not None and (not _number(position) or not gripper["close"] <= position <= gripper["open"]):
            return result("INFEASIBLE", "grasp_part position exceeds recorded gripper limits", gripper=gripper)
        return result("FEASIBLE", "Recorded tool, part and attachment declarations support independent grasp capability",
                      width=width, gripper=gripper, attach=attach,
                      composition_obligations=["pick_target_grounded", "motion_landed_on_target"])
    if primitive in {"compute_pick_targets", "compute_place_targets"}:
        calculated_task = deepcopy(task)
        if "part_name" in params:
            calculated_task["part_name"] = params["part_name"]
        if calculated_task.get("part_name") != task.get("part_name"):
            return result("INFEASIBLE", "Target calculation identifies a different part")
        if primitive == "compute_pick_targets":
            calculated = calculate_recovery_pick_target(task=calculated_task,
                recovery_snapshot=recovery_snapshot, resource_evidence=resource_evidence)
        else:
            calculated = calculate_recovery_place_target(task=calculated_task,
                recovery_snapshot=recovery_snapshot, resource_evidence=resource_evidence,
                physical_evidence=physical_evidence)
        return result("FEASIBLE" if calculated["status"] == "calculated" else "NEEDS_CONTEXT",
                      "Recorded inputs support the pure owner calculation" if calculated["status"] == "calculated" else "Owner calculation lacks required recorded inputs",
                      refs=calculated["context_requests"], calculation=calculated)
    if primitive == "detect_parts":
        part = params.get("part_name")
        observed = (resource_evidence.get("observed_parts") or {}).get(part) or {}
        pose = _pose_mapping(observed.get("observed_pose", observed.get("current_pose")))
        if observed.get("source", {}).get("kind") == "projected":
            recorded = (resource_evidence.get("recorded_observed_parts") or {}).get(part) or {}
            recorded_pose = _pose_mapping(recorded.get("observed_pose", recorded.get("current_pose")))
            context = part_context or {}
            observation_source = context.get("observed_pose_source") or {}
            owner_source = resource_evidence.get("source") or {}
            if (not observation_source.get("run_id") or "revision" not in observation_source
                    or observation_source.get("frame") != "world"
                    or observation_source.get("run_id") != owner_source.get("run_id")
                    or observation_source.get("revision") != owner_source.get("revision")
                    or context.get("observed_pose_current") is False
                    or not all(_number(pose.get(axis)) for axis in ("x", "y", "z"))
                    or pose != recorded_pose or pose != context.get("observed_pose")):
                return result("NEEDS_CONTEXT", "A projected part pose is not a perception observation", refs=[root + "/observed_parts/" + str(part)])
            return result("FEASIBLE", "Retained recorded observation supports independent detection capability",
                          observed_pose=recorded_pose, observation_source=observation_source,
                          composition_obligations=["Confirm detection when the primitive executes"])
        if all(_number(pose.get(axis)) for axis in ("x", "y", "z")) and observed.get("frame", resource_evidence.get("frame")) == "world":
            return result("FEASIBLE", "Recorded part observation is available without invoking perception", observed_pose=pose)
        return result("NEEDS_CONTEXT", "Recorded part observation is unavailable", refs=[root + "/observed_parts/" + str(part)])
    if primitive == "release_part":
        part = params.get("part_name", task.get("part_name"))
        if "held_part" not in recovery_snapshot:
            return result("NEEDS_CONTEXT", "Recorded held_part is missing", refs=[root + "/state/_held_part"])
        held = recovery_snapshot["held_part"]
        if part is not None and held not in (None, part):
            return result("INFEASIBLE", "release_part identifies a different held part", held_part=held, part_name=part)
        if held is None:
            return result("INFEASIBLE", "Recorded held_part is null for release_part", held_part=None)
        if recovery_snapshot.get("controller_ready") is False:
            return result("INFEASIBLE", "Recorded controller_ready is false", controller_ready=False)
        if recovery_snapshot.get("controller_ready") is not True:
            return result("NEEDS_CONTEXT", "Recorded controller readiness is unavailable", refs=[root + "/snapshot/controller_ready"])
        attachment = resource_evidence.get("attachment")
        if not isinstance(attachment, dict):
            return result("NEEDS_CONTEXT", "Recorded part attachment is unavailable", refs=[root + "/attachment"])
        target_model = (resource_evidence.get("inputs") or {}).get("geometry", {}).get("parts", {}).get("model_map", {}).get(held)
        if not isinstance(target_model, str) or not target_model:
            return result("NEEDS_CONTEXT", "Recorded part model identity is unavailable", refs=[root + "/inputs/geometry/parts/model_map"])
        if params.get("model_name", target_model) != target_model or attachment.get("model_name") != target_model:
            return result("INFEASIBLE", "release_part model identity conflicts with recorded custody", attachment=attachment)
        projected_attachment = (attachment.get("projected_attachment_complete") is True
            and attachment.get("part_name") == held and attachment.get("source", {}).get("kind") == "projected"
            and bool(attachment.get("source", {}).get("contract_fingerprint")))
        if (attachment.get("physical_attachment_completed") is not True and attachment.get("attached") is not True
                and not projected_attachment
                and not (attachment.get("observed_attachment_complete") is True and attachment.get("part_name") == held
                         and isinstance(attachment.get("records"), list) and len(attachment["records"]) == 1)):
            return result("NEEDS_CONTEXT", "Recorded attachment completion is unavailable", refs=[root + "/attachment"])
        slot = params.get("assembly_slot")
        if slot is not None:
            if (not isinstance(slot, dict) or slot.get("destination_location") != "assembly_board-v1"
                    or (task.get("expected_end_state") or {}).get("resource_location") != "assembly_board-v1"
                    or any(not _number(slot.get(key)) for key in ("slot_x", "slot_y", "part_height", "board_top_z", "part_origin_z"))):
                return result("INFEASIBLE", "release_part assembly_slot is incompatible with its declared destination")
            if resource_evidence.get("execution_mode") != "simulation":
                return result("NEEDS_CONTEXT", "assembly_slot requires recorded simulation execution_mode", refs=[root + "/execution_mode"])
        return result("FEASIBLE", "Recorded held part and attachment support independent release capability",
                      held_part=held, attachment=attachment, composition_obligations=["release_target_grounded", "motion_landed_on_target"])
    if primitive not in {"move_cartesian", "move_pose", "move_relative"}:
        return None
    entry = next((row for row in primitive_catalog if row.get("name") == primitive), {})
    frame = entry.get("frame", params.get("frame"))
    configured_frame = resource_evidence.get("controller_config", {}).get("move_group", {}).get("frame_id")
    if frame is None:
        frame = configured_frame
    if frame != "world" or configured_frame not in (None, frame):
        return result("NEEDS_CONTEXT", "Recorded primitive planning frame is unavailable or conflicting",
                      refs=[root + "/controller_config/move_group/frame_id"])
    pose = {axis: params.get(axis) for axis in ("x", "y", "z")}
    if primitive == "move_relative":
        current = _pose_mapping(recovery_snapshot.get("current_pose"))
        current_frame = recovery_snapshot.get("current_pose_frame", resource_evidence.get("frame"))
        if current_frame != frame:
            return result("NEEDS_CONTEXT", "Recorded current_pose frame is unavailable", refs=[root + "/snapshot/current_pose_frame"])
        if not all(_number(current.get(axis)) and _number(params.get("d" + axis)) for axis in ("x", "y", "z")):
            return result("NEEDS_CONTEXT", "Relative capability requires recorded pose and explicit dx, dy, dz", refs=[root + "/snapshot/current_pose"])
        pose = {axis: current[axis] + params["d" + axis] for axis in ("x", "y", "z")}
    if not all(_number(value) for value in pose.values()):
        return result("NEEDS_CONTEXT", "Cartesian capability requires explicit finite x, y, z")
    reach = static_capabilities.get("gripper_reach")
    if not isinstance(reach, dict) or reach.get("frame") != frame:
        return result("NEEDS_CONTEXT", "Recorded gripper_reach is unavailable", refs=[f"/resources/{resource_jid}/static_capabilities/gripper_reach"])
    from cais_spade_llm.agents.resource_agent.robot_agent import RobotAgent

    error = RobotAgent._recovery_pose_gripper_reach_error(pose, reach)
    return result(("INFEASIBLE" if error.startswith("pose is outside gripper_reach:") else "NEEDS_CONTEXT") if error else "FEASIBLE",
                  error or "Bound pose is inside recorded gripper_reach", checked_pose=pose, pose_frame=frame, gripper_reach=reach)


def check_outline_motion_capabilities(*, motion_requests: list, static_capabilities: dict,
                                      frame: str | None, resource_jid: str) -> dict:
    """Check calculated Cartesian targets against the owner's recorded limits.

    This applies the existing Cartesian target checks, without deriving an
    ordered primitive program or treating target checks as trajectory proof.
    Named joint motion retains its separately checked named-pose declaration.
    """
    from cais_spade_llm.agents.resource_agent.robot_agent import RobotAgent
    from cais_spade_llm.agents.shared_information.recovery_validation_protocol import recovery_validation_fingerprint

    result = {"feasibility_status": "FEASIBLE", "resource_jid": resource_jid,
        "static_capabilities_fingerprint": recovery_validation_fingerprint(static_capabilities),
        "motion_requests_fingerprint": recovery_validation_fingerprint(motion_requests),
        "frame": frame, "scope": "Calculated Cartesian targets only; no motion interval approval",
        "checks": [], "command_sent": False, "execution_authorized": False}
    if not isinstance(motion_requests, list):
        result.update(feasibility_status="NEEDS_CONTEXT", reason="Owner motion requests are unavailable")
        return result
    for motion_index, motion in enumerate(motion_requests):
        if not isinstance(motion, dict):
            result.update(feasibility_status="NEEDS_CONTEXT", reason="Owner motion has no supported target declaration")
            return result
        if motion.get("mode") == "joint_target":
            continue
        if motion.get("mode") != "cartesian" or not isinstance(motion.get("waypoints"), list) or not motion["waypoints"]:
            result.update(feasibility_status="NEEDS_CONTEXT", reason="Owner motion has no supported target declaration")
            return result
        reach = static_capabilities.get("gripper_reach") if isinstance(static_capabilities, dict) else None
        if frame != "world" or not isinstance(reach, dict) or reach.get("frame") != frame:
            result.update(feasibility_status="NEEDS_CONTEXT", reason="Recorded Cartesian target and gripper_reach frames are missing or conflicting")
            return result
        for target_index, waypoint in enumerate(motion["waypoints"]):
            pose = waypoint.get("position") if isinstance(waypoint, dict) else None
            if not isinstance(pose, dict) or not all(_number(pose.get(axis)) for axis in ("x", "y", "z")):
                result.update(feasibility_status="NEEDS_CONTEXT", reason="Calculated Cartesian target is incomplete or nonfinite")
                return result
            error = RobotAgent._recovery_pose_gripper_reach_error(pose, reach)
            status = (("INFEASIBLE" if error.startswith("pose is outside gripper_reach:") else "NEEDS_CONTEXT")
                      if error else "FEASIBLE")
            check = {"motion_index": motion_index, "target_index": target_index, "feasibility_status": status,
                "checked_pose": deepcopy(pose), "frame": frame, "gripper_reach": deepcopy(reach),
                "reason": error or "Calculated target is inside recorded gripper_reach"}
            result["checks"].append(check)
            if error:
                result.update(feasibility_status=status, reason=error)
                return result
    return result
