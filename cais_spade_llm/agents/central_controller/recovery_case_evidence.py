from __future__ import annotations

"""Calculate case APs from complete resource-owned bounds and hand-off records.

These calculations do not acquire observations or authorize execution. Interval
records must come from the independently checked owner projection; a command
argument or a claimed predicate value is not evidence for these APs.
"""

import hashlib
import json
from copy import deepcopy
from fractions import Fraction
from math import atan2, inf, isfinite, nextafter, pi, sqrt
from typing import Any
from xml.etree import ElementTree

CASE_AP_MEANINGS = {
    "base_moving": "The owner base velocity bounds establish whether the base moves.",
    "arm_parked": "Every arm joint bound lies within the configured parked_arm_configuration tolerance.",
    "arm_moving": "The owner arm joint velocity bounds establish whether the arm moves.",
    "docked": "The complete base pose bounds satisfy the exact configured docking target and tolerances.",
    "carried_part_clearance": "The complete carried-part clearance bounds satisfy every declared obstacle-pair clearance.",
    "handoff_identity": "The complete owner hand-off records preserve the same exact part identity and exclusive custody.",
    "handoff_acknowledged": "Every completed owner hand-off has its matching acknowledgement or independently validated projected completion effect.",
    "robot_clear": "The complete owner hand-off clearance bounds satisfy the declared clearance for every relevant robot and carried part.",
}


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _number(value: Any) -> Fraction:
    if type(value) not in (int, float) or not isfinite(value):
        raise ValueError("Owner bounds require finite numeric values")
    return Fraction(str(value))


def _bounds(value: Any) -> tuple[Fraction, Fraction]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError("Owner evidence requires explicit lower and upper bounds")
    lo, hi = map(_number, value)
    if lo > hi:
        raise ValueError("Owner evidence bounds are reversed")
    return lo, hi


def _all(values: list[list[bool]]) -> list[bool]:
    if any(row == [False] for row in values):
        return [False]
    return [True] if all(row == [True] for row in values) else [False, True]


def _inside(value: Any, target: Any, tolerance: Any) -> list[bool]:
    lo, hi = _bounds(value)
    center, error = _number(target), _number(tolerance)
    if error < 0:
        raise ValueError("Owner configuration tolerance cannot be negative")
    if center - error <= lo <= hi <= center + error:
        return [True]
    if hi < center - error or lo > center + error:
        return [False]
    return [False, True]


def _record(resource: str, observation: dict) -> dict:
    envelope = observation.get("owner_safety_evidence", {})
    if not isinstance(envelope, dict):
        raise ValueError("Owner safety evidence is unavailable")
    checkpoint = {key: observation[key] for key in ("resources", "parts", "frame", "state_snapshot")
                  if key in observation}
    if envelope.get("observation_fingerprint") != _digest(checkpoint):
        raise ValueError("Owner AP evidence is missing or bound to another physical checkpoint")
    records = envelope.get("resource_evidence")
    if not isinstance(records, dict):
        raise ValueError("Raw owner safety records are unavailable")
    record = records.get(resource)
    if not isinstance(record, dict):
        raise ValueError("Raw owner safety evidence is unavailable for " + resource)
    source = record.get("source", {})
    if (not isinstance(source, dict) or source.get("kind") not in {"recorded", "projected"}
            or not isinstance(source.get("fingerprint"), str) or not source["fingerprint"]):
        raise ValueError("Owner AP evidence requires its captured or projected source")
    required = observation.get("continuous_interval") if observation.get("phase") == "between" else [
        observation["time_exact"], observation["time_exact"]]
    if ([Fraction(str(item)) for item in record.get("interval", [])]
            != [Fraction(str(item)) for item in required]):
        raise ValueError("Owner AP bounds do not cover the exact physical interval")
    if resource not in observation["resources"]:
        raise ValueError("Owner AP resource is absent from the physical checkpoint")
    return record


def _joint_velocity_bounds(motion, start: Fraction, end: Fraction) -> dict:
    from cais_spade_llm.resources.continuous_motion import polynomial_bounds

    result = {}
    for index, name in enumerate(motion.names):
        bounds = []
        for segment, (a, b) in enumerate(zip(motion.exact_times, motion.exact_times[1:])):
            if max(a, start) > min(b, end):
                continue
            coefficients = motion.coefficients[segment][index]
            derivative = [Fraction(i) * value / (b-a) for i, value in enumerate(coefficients)][1:]
            # A zero polynomial is exact. Outward arithmetic otherwise keeps
            # all spline-interior extrema, including at segment boundaries.
            bounds.append((0., 0.) if not any(derivative) else polynomial_bounds(
                derivative, (max(a, start)-a)/(b-a), (min(b, end)-a)/(b-a)))
        if not bounds:
            raise ValueError("Velocity interval is outside the recorded trajectory")
        result[name] = [min(row[0] for row in bounds), max(row[1] for row in bounds)]
    return result


def _pose_bounds(pose: list) -> dict:
    if not isinstance(pose, list) or len(pose) != 7:
        raise ValueError("Base pose requires captured xyz and quaternion")
    values = [float(_number(value)) for value in pose]
    x, y, z, w = values[3:]
    if abs(x*x+y*y+z*z+w*w-1) > 1e-9:
        raise ValueError("Captured base quaternion is not a unit quaternion")
    yaw = atan2(2*(w*z+x*y), 1-2*(y*y+z*z))
    return {"x": [values[0], values[0]], "y": [values[1], values[1]],
            "yaw": [nextafter(yaw, -inf), nextafter(yaw, inf)]}


def _base_physics_description(evidence: dict, model: dict, resource_id: str):
    planning = ElementTree.fromstring(evidence["robot_description"])
    if evidence["version"] == 1:
        return planning
    description = evidence.get("physics_robot_description")
    if (not isinstance(description, str)
            or hashlib.sha256(description.encode()).hexdigest() != evidence.get("physics_description_sha256")
            or not isinstance(evidence.get("physics_description_node"), str)
            or not evidence["physics_description_node"].startswith("/")):
        raise ValueError("Base hold requires its separately captured physical description and source")
    physics = ElementTree.fromstring(description)
    if physics.get("name") != resource_id:
        raise ValueError("Base hold physical description belongs to another resource")

    def subtree(tree):
        links = {row.get("name"): row for row in tree.findall("link")}
        joints = {row.get("name"): row for row in tree.findall("joint")}
        if (None in links or None in joints or len(links) != len(tree.findall("link"))
                or len(joints) != len(tree.findall("joint")) or model["root"] not in links):
            raise ValueError("Base hold descriptions require exact unique link and joint identities")
        selected_links, selected_joints = {model["root"]}, set()
        pending = dict(joints)
        while pending:
            added = False
            for name, joint in list(pending.items()):
                parent, child = joint.find("parent"), joint.find("child")
                if parent is None or child is None:
                    raise ValueError("Base hold descriptions contain an incomplete joint declaration")
                if parent.get("link") not in selected_links:
                    continue
                child_name = child.get("link")
                if child_name not in links or child_name in selected_links:
                    raise ValueError("Base hold descriptions contain an invalid owner joint tree")
                selected_links.add(child_name)
                selected_joints.add(name)
                del pending[name]
                added = True
            if not added:
                break
        return ({name: links[name] for name in selected_links},
                {name: joints[name] for name in selected_joints})

    def declaration(element):
        return (element.tag, element.attrib, (element.text or "").strip(),
                [declaration(child) for child in element])

    planned_links, planned_joints = subtree(planning)
    actual_links, actual_joints = subtree(physics)
    if (set(planned_links) != set(actual_links) or set(planned_joints) != set(actual_joints)
            or any(declaration(planned_links[name]) != declaration(actual_links[name]) for name in planned_links)
            or any(declaration(planned_joints[name]) != declaration(actual_joints[name]) for name in planned_joints)):
        raise ValueError("Base hold physical links and joints differ from the captured planning geometry")
    return physics


def recorded_command_gate(record: dict) -> dict:
    """Check saved output authority while retaining external goal uncertainty."""
    proof = record.get("command_gate_evidence")
    gate = record.get("command_gate")
    activity = record.get("owned_activity")
    if not all(isinstance(value, dict) for value in (proof, gate, activity)):
        raise ValueError("Recorded controller command gate and owned activity are unavailable")
    declaration = proof.get("declaration")
    if not isinstance(declaration, dict):
        raise ValueError("Recorded controller command gate declaration is unavailable")
    owned, gated = declaration.get("owned_status_topics"), declaration.get("status_topics")
    source = record.get("controller_source_sha256")
    publisher = proof.get("output_publisher_gid")
    if (proof.get("status") != "supported" or proof.get("physical_execution_verified") is not False
            or proof.get("all_nav2_goals_idle", False) is not None
            or type(gate.get("version")) is not int or gate["version"] != 1
            or gate.get("closed") is not True or gate != proof.get("command_gate")
            or activity != proof.get("owned_activity")
            or activity.get("has_active_goal") is not False or activity.get("has_pending_goal") is not False
            or activity.get("holding") is not True
            or not isinstance(source, str) or len(source) != 64 or any(value not in "0123456789abcdef" for value in source)
            or source != proof.get("source_sha256")
            or not record.get("instance_id") or record["instance_id"] != proof.get("controller_instance_id")
            or record.get("controller_node") != declaration.get("controller_node")
            or record.get("controller_node") != proof.get("controller_node")
            or record.get("controller_node") != proof.get("output_publisher_node")
            or not isinstance(record.get("controller_node"), str) or not record["controller_node"].startswith("/")
            or type(proof.get("output_publisher_count")) is not int or proof["output_publisher_count"] != 1
            or not isinstance(publisher, list) or not publisher
            or any(type(value) is not int or not 0 <= value <= 255 for value in publisher)
            or not isinstance(owned, list) or not owned or not isinstance(gated, list) or not gated
            or any(not isinstance(topic, str) or not topic.startswith("/") for topic in owned + gated)
            or len(set(owned + gated)) != len(owned + gated)
            or gate.get("owned_status_topics") != owned or gate.get("gated_status_topics") != gated
            or not isinstance(declaration.get("input_topic"), str) or not declaration["input_topic"].startswith("/")
            or not isinstance(declaration.get("output_topic"), str) or not declaration["output_topic"].startswith("/")
            or gate.get("input_topic") != declaration["input_topic"]
            or gate.get("output_topic") != declaration["output_topic"]
            or not declaration.get("source_package") or not declaration.get("source_executable")
            or not isinstance(proof.get("nav2_goal_observations"), dict)
            or set(proof["nav2_goal_observations"]) != set(gated)):
        raise ValueError("Recorded controller command gate has incomplete or conflicting authority evidence")
    output = record.get("last_output")
    if (not isinstance(output, list) or len(output) != 3
            or any(type(value) not in (int, float) or value != 0 for value in output)):
        raise ValueError("Recorded closed command gate has no exact zero output")
    return deepcopy(activity)


def recorded_base_hold(*, resource_id: str, physical_evidence: dict, model: dict) -> dict:
    """Recheck a captured controller/model hold under no new base goals."""
    evidence = physical_evidence.get("resources", {}).get(resource_id, {}).get("base_hold_evidence")
    if (not isinstance(evidence, dict) or type(evidence.get("version")) is not int
            or evidence["version"] not in {1, 2}):
        raise ValueError("Captured base hold model and controller output are unavailable")
    controller = evidence["controller_observation"]
    description = evidence["robot_description"]
    activity = (recorded_command_gate(controller) if evidence["version"] == 2
                or "command_gate" in controller or "command_gate_evidence" in controller else controller)
    if (not isinstance(description, str)
            or hashlib.sha256(description.encode()).hexdigest() != model.get("description_sha256")
            or controller.get("controller_source_sha256") != evidence.get("controller_source_sha256")
            or not isinstance(evidence.get("controller_source_sha256"), str)
            or len(evidence["controller_source_sha256"]) != 64
            or controller.get("controller_node") != "/"+resource_id+"_base_controller"
            or controller.get("source") != "/"+resource_id+"/recovery_state"
            or activity.get("has_active_goal") is not False or activity.get("has_pending_goal") is not False
            or controller.get("inactive_goal_output") != [0., 0., 0.]
            or controller.get("last_output") != [0., 0., 0.]):
        raise ValueError("Base hold lacks the exact inactive controller and zero output declaration")
    if any(type(value) not in (int, float) or value != 0 for field in
           ("last_output", "inactive_goal_output") for value in controller[field]):
        raise ValueError("Base hold output must have typed finite zero components")
    root = _base_physics_description(evidence, model, resource_id)
    declarations = root.findall("gazebo[@reference='"+model["root"]+"']")
    if (len(declarations) != 1 or declarations[0].findtext("kinematic") != "true"
            or declarations[0].findtext("gravity") != "false"):
        raise ValueError("Base hold requires the exact kinematic gravity-free root declaration")
    plugins = [plugin for gazebo in root.findall("gazebo") for plugin in gazebo.findall("plugin")
               if plugin.get("filename") == "libgazebo_ros_planar_move.so"]
    if (len(plugins) != 1 or plugins[0].findtext("robot_base_frame") != model["root"]
            or plugins[0].findtext("ros/namespace") != "/"+resource_id):
        raise ValueError("Base hold lacks its exact planar controller model")
    if activity is not controller:
        plugin = plugins[0]
        # This captured plugin's unremapped command input is cmd_vel. Other
        # command bindings need their own supported model, not a guessed alias.
        command_topic = plugin.findtext("command_topic", "cmd_vel")
        if (command_topic != "cmd_vel" or plugin.findall("ros/remapping")
                or controller["command_gate"]["output_topic"] != "/"+resource_id+"/cmd_vel"):
            raise ValueError("Base hold command gate does not bind the captured planar input")
    pose = [*model["root_xyz"], *model["root_quaternion"]]
    if (evidence.get("root_link") != model["root"] or evidence.get("root_pose") != pose
            or evidence.get("linear_velocity") != [0.,0.,0.] or evidence.get("angular_velocity") != [0.,0.,0.]):
        raise ValueError("Base hold does not match measured root pose and zero velocity")
    if any(type(value) not in (int, float) or value != 0 for field in
           ("linear_velocity", "angular_velocity") for value in evidence[field]):
        raise ValueError("Base hold requires typed measured zero velocity")
    window = _number(evidence["max_age"])
    if (window <= 0 or abs(_number(controller["simulation_time"])-_number(evidence["physics_source"]["simulation_time"])) > window
            or controller.get("instance_id") != evidence.get("launch_id")):
        raise ValueError("Base hold controller and physics capture identities differ")
    return {"base_pose_bounds": _pose_bounds(pose),
            "base_velocity_bounds": {axis:[0.,0.] for axis in ("x","y","yaw")},
            "source": {"kind":"projected", "fingerprint":_digest(evidence),
                "assumptions":["No new base goals or external base commands during this owner interval",
                               "The captured planar controller and kinematic root follow the recorded zero-command model"],
                "physical_execution_verified":False}}


def _scene_boxes(scene: dict) -> dict:
    from cais_spade_llm.resources.continuous_motion import (
        _matrix_product, quaternion_transform, transformed_box,
    )

    result = {}
    for row in scene["world"]["collision_objects"]:
        identifier = row["id"]
        if not identifier or identifier in result or row.get("planes") or row.get("subframe_names"):
            raise ValueError("Collision-object inventory contains unsupported or duplicate geometry")
        frame = row.get("header", {}).get("frame_id")
        if frame != "world":
            raise ValueError("Collision object needs the explicit world frame")
        def matrix(pose):
            return quaternion_transform([pose["position"][a] for a in ("x", "y", "z")],
                [pose["orientation"][a] for a in ("x", "y", "z", "w")])
        origin = matrix(row["pose"]) if "pose" in row else None
        boxes = []
        for shape, pose in zip(row.get("primitives", []), row.get("primitive_poses", []), strict=True):
            dimensions = shape["dimensions"]
            if shape["type"] == 1 and len(dimensions) == 3:
                half = [float(_number(value))/2 for value in dimensions]
            elif shape["type"] == 2 and len(dimensions) == 1:
                half = [float(_number(dimensions[0]))]*3
            elif shape["type"] == 3 and len(dimensions) == 2:
                half = [float(_number(dimensions[1]))]*2+[float(_number(dimensions[0]))/2]
            else:
                raise ValueError("Collision primitive lacks supported complete bounds")
            if min(half) <= 0:
                raise ValueError("Collision primitive dimensions must be positive")
            transform = matrix(pose)
            boxes.append(transformed_box(_matrix_product(origin, transform) if origin else transform,
                [[-value, value] for value in half], identifier))
        for mesh, pose in zip(row.get("meshes", []), row.get("mesh_poses", []), strict=True):
            vertices = mesh["vertices"]
            if not vertices or not mesh["triangles"]:
                raise ValueError("Collision mesh lacks its captured complete geometry")
            bounds = [[min(float(_number(v[a])) for v in vertices), max(float(_number(v[a])) for v in vertices)]
                      for a in ("x", "y", "z")]
            transform = matrix(pose)
            boxes.append(transformed_box(_matrix_product(origin, transform) if origin else transform, bounds, identifier))
        if not boxes:
            raise ValueError("Collision object has no complete supported shapes")
        result[identifier] = boxes
    return result


def _box_distance(a: dict, b: dict) -> list:
    # Enclosing AABBs can establish separation, never penetration. Their
    # overlap remains a zero lower bound rather than a collision assertion.
    low2 = Fraction()
    high2 = Fraction()
    for axis in range(3):
        alo, ahi = Fraction(a["minimum"][axis][0]), Fraction(a["maximum"][axis][1])
        blo, bhi = Fraction(b["minimum"][axis][0]), Fraction(b["maximum"][axis][1])
        low2 += max(Fraction(), blo-ahi, alo-bhi)**2
        high2 += max(abs(alo-bhi), abs(ahi-blo))**2
    return [max(0., nextafter(sqrt(float(low2)), -inf)), nextafter(sqrt(float(high2)), inf)]


def calculate_case_resource_evidence(*, resource_id: str, observation: dict,
        physical_evidence: dict, configuration: dict, joint_trajectory: dict | None = None,
        trajectory_start_time=0, docking_target: str | None = None,
        planning_scene: dict | None = None, scene_configuration: dict | None = None,
        peer_holds: dict | None = None) -> dict:
    """Calculate raw case AP inputs from the owner's exact physical sources.

    Missing fields stay explicit; producing arm bounds does not establish base
    hold, docking, custody, obstacle completeness, or future execution tracking.
    The caller attaches this record to its independently validated observation.
    """
    row = observation["resources"][resource_id]
    interval = (observation["continuous_interval"] if observation.get("phase") == "between"
                else [observation["time_exact"]]*2)
    record = {"configuration": deepcopy(configuration), "interval": deepcopy(interval),
              "missing_facts": [], "physical_execution_verified": False}
    model = physical_evidence["resource_models"][resource_id]["configuration"]
    source = {"observation": {key: observation[key] for key in ("resources", "parts", "frame")},
              "model": model, "configuration": configuration, "joint_trajectory": joint_trajectory,
              "trajectory_start_time": trajectory_start_time}
    predicted = joint_trajectory is not None or physical_evidence.get("provenance", {}).get("kind") == "projected"
    record["source"] = {"kind": "projected" if predicted else "recorded",
                        "fingerprint": _digest(source)}
    names = configuration["arm_joint_names"]
    if names != model["joint_names"]:
        raise ValueError("Owner arm identities differ from the captured kinematic model")
    motion = None
    if joint_trajectory is not None:
        from cais_spade_llm.resources.continuous_motion import ContinuousMotion

        motion = ContinuousMotion(joint_trajectory, model)
        start, end = (Fraction(str(value))-Fraction(str(trajectory_start_time)) for value in interval)
        if not 0 <= start <= end <= motion.exact_times[-1]:
            raise ValueError("Case AP interval is outside the exact owner trajectory")
        record["joint_position_bounds"] = {key: list(value) for key, value in motion.joints(start, end).items()}
        record["joint_velocity_bounds"] = _joint_velocity_bounds(motion, start, end)
        try:
            hold = recorded_base_hold(resource_id=resource_id, physical_evidence=physical_evidence, model=model)
        except (KeyError, ValueError, TypeError, ElementTree.ParseError) as exc:
            record["missing_facts"].append("/resources/"+resource_id+"/base_hold_evidence")
            record["base_hold_unresolved"] = str(exc)
        else:
            record.update({key: deepcopy(hold[key]) for key in ("base_pose_bounds", "base_velocity_bounds")})
            record["source"]["base_hold"] = hold["source"]
    else:
        captured = physical_evidence.get("provenance", {}).get("kind") != "projected"
        physics = physical_evidence.get("physics_evidence", {}) if captured else {}
        observed = physics.get("models", {}).get(resource_id, {})
        joints = observed.get("joints", {})
        if observation.get("phase") != "between" and row.get("joint_names") == names and all(
                name in joints and joints[name].get("position") == position
                for name, position in zip(names, row.get("joint_positions", []), strict=True)):
            record["joint_position_bounds"] = {name: [joints[name]["position"]]*2 for name in names}
            if all(type(joints[name].get("velocity")) in (int, float) for name in names):
                record["joint_velocity_bounds"] = {name: [float(_number(joints[name]["velocity"]))]*2 for name in names}
        root = observed.get("links", {}).get(model["root"], {})
        if observation.get("phase") != "between" and root.get("pose") == row.get("base_pose") and root.get("pose"):
            record["base_pose_bounds"] = _pose_bounds(root["pose"])
            if all(isinstance(root.get(field), list) and len(root[field]) == 3
                   for field in ("linear_velocity", "angular_velocity")):
                record["base_velocity_bounds"] = {axis: [float(_number(value))]*2 for axis, value in
                    zip(("x", "y", "yaw"), [*root["linear_velocity"][:2], root["angular_velocity"][2]], strict=True)}
            record["source"]["physics_fingerprint"] = _digest(physics)
        if predicted:
            from cais_spade_llm.resources.recovery_geometry import commanded_hold_evidence

            hold = commanded_hold_evidence(resource_id=resource_id, state=row,
                model=physical_evidence["resource_models"][resource_id],
                physical=physical_evidence.get("resources", {}).get(resource_id, {}))
            if hold.get("status") == "supported":
                record["joint_position_bounds"] = deepcopy(hold["joint_intervals"])
                record["source"]["commanded_hold"] = deepcopy(hold)
                # A position hold is not a zero-velocity guarantee. Retain the
                # captured model's full velocity limits, when declared.
                description = physical_evidence.get("resources", {}).get(resource_id, {}).get(
                    "base_hold_evidence", {}).get("robot_description")
                if isinstance(description, str) and hashlib.sha256(description.encode()).hexdigest() == model.get("description_sha256"):
                    root = ElementTree.fromstring(description)
                    bounds = {}
                    for name in names:
                        joints = root.findall("joint[@name='"+name+"']")
                        if len(joints) != 1 or joints[0].find("limit") is None:
                            break
                        value = joints[0].find("limit").get("velocity")
                        if value is None or not isfinite(float(value)) or float(value) < 0:
                            break
                        bounds[name] = [-float(value), float(value)]
                    if set(bounds) == set(names):
                        record["joint_velocity_bounds"] = bounds
                        record["source"]["velocity_model"] = {
                            "description_sha256": model["description_sha256"],
                            "assumption": "Joint velocities remain within the captured URDF velocity limits"}
            else:
                record["commanded_hold_unresolved"] = hold.get("reason")
            try:
                base_hold = recorded_base_hold(resource_id=resource_id, physical_evidence=physical_evidence, model=model)
                if row.get("base_pose") != physical_evidence["resources"][resource_id]["base_hold_evidence"]["root_pose"]:
                    raise ValueError("Projected base pose differs from the captured base hold")
            except (KeyError, ValueError, TypeError, ElementTree.ParseError) as exc:
                record["missing_facts"].append("/resources/"+resource_id+"/base_hold_evidence")
                record["base_hold_unresolved"] = str(exc)
            else:
                record.update({key: deepcopy(base_hold[key]) for key in ("base_pose_bounds", "base_velocity_bounds")})
                record["source"]["base_hold"] = base_hold["source"]
        for field in ("joint_position_bounds", "joint_velocity_bounds", "base_pose_bounds", "base_velocity_bounds"):
            if field not in record:
                record["missing_facts"].append("/resources/"+resource_id+"/"+field)
    if "held_part" in row:
        record["held_part"] = row["held_part"]
    else:
        record["missing_facts"].append("/resources/"+resource_id+"/held_part")
    if docking_target is not None and scene_configuration is not None:
        rows = [item for item in scene_configuration.get("machines", []) if item.get("resource_id") == docking_target]
        if docking_target in scene_configuration:
            rows.append(scene_configuration[docking_target])
        targets = [item["KMR_docking_pose"] for item in rows if "KMR_docking_pose" in item]
        if len(targets) == 1:
            record.update(docking_target=docking_target, docking_targets={docking_target: deepcopy(targets[0])})
            record["source"]["docking_configuration_fingerprint"] = _digest(rows)
        else:
            record["missing_facts"].append("/scene/"+docking_target+"/KMR_docking_pose")
    if planning_scene is not None:
        obstacles = _scene_boxes(planning_scene)
        record["source"]["planning_scene_fingerprint"] = _digest(planning_scene)
        held = row.get("held_part")
        if held is not None and motion is not None:
            if ("minimum_clearance_m" not in configuration
                    and configuration.get("task_execution", {}).get("avoid_collisions") is not True):
                record["missing_facts"].append("/scene/"+resource_id+"/minimum_clearance_m")
            from cais_spade_llm.resources.continuous_motion import (
                _matrix_product, quaternion_transform, transformed_box,
            )
            from cais_spade_llm.resources.recovery_geometry import _joint_geometry, _resource_boxes

            snapshot = physical_evidence["snapshot"]
            geometry, models = physical_evidence["geometry"], physical_evidence["resource_models"]
            if set(snapshot["resources"]) != set(observation["resources"]) or set(geometry["resources"]) != set(snapshot["resources"]):
                raise ValueError("Carried clearance omits part of the physical resource population")
            peer_boxes, matrices, holders = _resource_boxes(snapshot, geometry, models)
            complete = True
            for peer in snapshot["resources"]:
                if peer == resource_id:
                    continue
                physical = physical_evidence["resources"][peer]
                if physical.get("model_static") is True:
                    continue
                hold = (peer_holds or {}).get(peer)
                if not isinstance(hold, dict) or hold.get("status") != "supported":
                    complete = False
                    record["missing_facts"].append("/resources/"+peer+"/stationary_interval_evidence")
                    continue
                from cais_spade_llm.agents.shared_information.recovery_validation_protocol import recovery_validation_fingerprint

                if (hold.get("model_fingerprint") != recovery_validation_fingerprint(models[peer]["configuration"])
                        or hold.get("start_joint_positions") != snapshot["resources"][peer]["joint_positions"]):
                    raise ValueError("Carried clearance peer hold is bound to another model or state")
                peer_boxes[peer], matrices[peer] = _joint_geometry(snapshot["resources"][peer], models[peer],
                    joint_intervals=hold["joint_intervals"])
            if complete:
                attached = planning_scene.get("robot_state", {}).get("attached_collision_objects", [])
                binding = physical_evidence.get("scene_object_bindings", {}).get(held, {})
                identifiers = [item["object_id"] for item in binding.get("objects", [])]
                own = [item for item in attached if item.get("object", {}).get("id") in identifiers]
                if not identifiers or len(own) != len(identifiers):
                    record["missing_facts"].append("/scene_object_bindings/"+held)
                    complete = False
                else:
                    touch_links = set.intersection(*(set(item.get("touch_links", [])) for item in own))
                    for component, box in zip(model["components"], motion.boxes(start, end), strict=True):
                        if component["link"] not in touch_links:
                            obstacles.setdefault(box["id"], []).append(box)
                    record["source"]["attachment_touch_links"] = sorted(touch_links)
            if complete:
                for peer, boxes in peer_boxes.items():
                    if peer == resource_id:
                        continue
                    for box in boxes:
                        obstacles.setdefault(box["id"], []).append(box)
                for part, peer in holders.items():
                    if peer == resource_id:
                        continue
                    grasp = snapshot["resources"][peer]["grasp_transform"]
                    transform = _matrix_product(matrices[peer], quaternion_transform(grasp[:3], grasp[3:]))
                    obstacles.setdefault(part, []).append(transformed_box(transform, geometry["parts"][part]["local_bounds"], part))
                carried = motion.carried_part_box(start, end, row["grasp_transform"], geometry["parts"][held]["local_bounds"], held)
                pairs = []
                for identifier, boxes in obstacles.items():
                    if identifier in identifiers:
                        raise ValueError("Carried part appears simultaneously attached and in the world")
                    distances = [_box_distance(carried, box) for box in boxes]
                    pairs.append({"body": held, "obstacle": identifier,
                        "distance_bounds": [min(value[0] for value in distances), min(value[1] for value in distances)]})
                record["carried_part_clearance"] = {"pairs_complete": True, "pairs": pairs}
                record["source"]["peer_hold_fingerprint"] = _digest(peer_holds or {})
        record["collision_object_ids"] = list(obstacles)
    return record


def bind_case_projection_evidence(projection: dict, *, physical_evidence: dict,
                                  scene_configuration: dict) -> dict:
    """Retain KMR peer facts for each independently supported owner checkpoint.

    An unchanged peer uses its captured commanded-hold model. Missing models,
    velocities or obstacle inventories remain unresolved rather than acquiring
    new observations or treating its resource_location as a docking proof.
    """
    result = deepcopy(projection)
    if result.get("status") != "projected" or result.get("interval_complete") is not True:
        raise ValueError("Case evidence requires an independently supported physical projection")
    if result.get("owner_interval_contract", {}).get("kind") == "resource_owned_shared_transport":
        try:
            clearance = _transport_clearance(result, physical_evidence, scene_configuration)
        except (KeyError, TypeError, ValueError, ElementTree.ParseError) as exc:
            result.setdefault("owner_safety_evidence", {}).setdefault("missing_facts", []).append(
                "Transfer corridor robot clearance: "+str(exc))
        else:
            for point in result["observations"]:
                for record in point.get("owner_safety_evidence", {}).get("resource_evidence", {}).values():
                    if "handoffs" in record:
                        record["robot_clear"] = deepcopy(clearance)
    configuration = scene_configuration.get("KMR")
    if not isinstance(configuration, dict):
        return result
    common = result.setdefault("owner_safety_evidence", {})
    common.setdefault("resource_configurations", {})["KMR"] = deepcopy(configuration)
    initial = physical_evidence.get("snapshot", {}).get("resources", {}).get("KMR")
    for index, observation in enumerate(result.get("observations", [])):
        if "KMR" not in observation.get("resources", {}):
            continue
        envelope = observation.setdefault("owner_safety_evidence", {})
        records = envelope.setdefault("resource_evidence", {})
        if "KMR" in records:
            continue
        row = observation["resources"]["KMR"]
        fields = ("joint_names", "joint_positions", "base_pose", "held_part", "grasp_transform")
        if not isinstance(initial, dict) or any(_digest(row.get(key)) != _digest(initial.get(key)) for key in fields):
            envelope.setdefault("missing_facts", []).append("KMR peer configuration changed without its owner motion evidence")
            continue
        source = dict(physical_evidence)
        if index or observation.get("phase") == "between" or physical_evidence.get("provenance", {}).get("kind") == "projected":
            source["provenance"] = {**physical_evidence.get("provenance", {}), "kind": "projected"}
        location = observation.get("state_snapshot", {}).get("KMR", {}).get("resource_location")
        if location is None:
            location = row.get("resource_location")
        try:
            records["KMR"] = calculate_case_resource_evidence(resource_id="KMR", observation=observation,
                physical_evidence=source, configuration=configuration,
                scene_configuration=scene_configuration, docking_target=location)
        except (KeyError, TypeError, ValueError, ElementTree.ParseError) as exc:
            envelope.setdefault("missing_facts", []).append("KMR peer evidence: "+str(exc))
    return result


def _transport_clearance(projection: dict, physical: dict, scene: dict) -> dict:
    from cais_spade_llm.resources.continuous_motion import _matrix_product, quaternion_transform, transformed_box
    from cais_spade_llm.resources.recovery_geometry import _joint_geometry, _resource_boxes, commanded_hold_evidence
    from cais_spade_llm.agents.shared_information.recovery_validation_protocol import recovery_validation_fingerprint

    contract = projection["owner_interval_contract"]
    if contract["physical_start_fingerprint"] != recovery_validation_fingerprint(physical):
        raise ValueError("Transport corridor differs from its exact physical start")
    participants = contract["event"]["participants"]
    declarations = {rid: scene[rid] for rid in participants}
    if not any("robot clear of transfer corridor" in row.get("transfer_requirements", [])
               for row in declarations.values()):
        raise ValueError("Transport has no declared robot-clear-of-transfer-corridor requirement")
    snapshot, models, geometry = physical["snapshot"], physical["resource_models"], physical["geometry"]
    population = set(snapshot["resources"])
    if population != set(models) or population != set(physical["resources"]) or population != set(geometry["resources"]):
        raise ValueError("Transfer corridor requires the complete resource and geometry population")
    boxes, matrices, holders = _resource_boxes(snapshot, geometry, models)
    robots, holds = [], {}
    for rid in sorted(population):
        if physical["resources"][rid].get("model_static") is True:
            continue
        if not models[rid].get("configuration", {}).get("joint_names"):
            raise ValueError("A non-static resource lacks its complete manipulator model: "+rid)
        robots.append(rid)
        hold = commanded_hold_evidence(resource_id=rid, state=snapshot["resources"][rid],
            model=models[rid], physical=physical["resources"][rid])
        if hold.get("status") != "supported":
            raise ValueError("Transfer corridor lacks supported peer hold: "+rid)
        if "base_pose" in snapshot["resources"][rid]:
            base = recorded_base_hold(resource_id=rid, physical_evidence=physical, model=models[rid]["configuration"])
            if snapshot["resources"][rid]["base_pose"] != physical["resources"][rid]["base_hold_evidence"]["root_pose"]:
                raise ValueError("Transfer corridor base hold is bound to another pose")
            hold = {**hold, "base_hold":base}
        boxes[rid], matrices[rid] = _joint_geometry(snapshot["resources"][rid], models[rid],
            joint_intervals=hold["joint_intervals"])
        holds[rid] = hold
    if not robots or any(participant in robots for participant in participants):
        raise ValueError("Shared transport participants must have static source/destination geometry")
    bodies = {rid: boxes[rid] for rid in robots}
    for part, rid in holders.items():
        if rid not in robots:
            raise ValueError("Carried part lacks its complete manipulator model")
        grasp = snapshot["resources"][rid]["grasp_transform"]
        matrix = _matrix_product(matrices[rid], quaternion_transform(grasp[:3], grasp[3:]))
        bodies[part] = [transformed_box(matrix, geometry["parts"][part]["local_bounds"], part)]
    corridor = {rid: boxes[rid] for rid in participants}
    displacement = float(_number(contract["displacement_m"]))
    if displacement <= 0 or not contract["part_names"]:
        raise ValueError("Transfer corridor requires the exact positive displacement and resident parts")
    for part in contract["part_names"]:
        if part in holders or part in corridor:
            raise ValueError("Transported part has conflicting custody or identity")
        pose = snapshot["parts"][part]["current_pose"]
        matrix = quaternion_transform(pose[:3], pose[3:])
        matrix[0][3] = (nextafter(pose[0], -inf), nextafter(pose[0]+displacement, inf))
        corridor[part] = [transformed_box(matrix, geometry["parts"][part]["local_bounds"], part)]
    for point in projection["observations"]:
        if set(point["resources"]) != population:
            raise ValueError("Transport checkpoint omits part of the resource population")
        for rid in robots:
            for field in ("joint_names", "joint_positions", "base_pose", "held_part", "grasp_transform"):
                if _digest(point["resources"][rid].get(field)) != _digest(snapshot["resources"][rid].get(field)):
                    raise ValueError("Transport peer changes without a matching owner interval")
    pairs = []
    for body, body_boxes in bodies.items():
        for obstacle, obstacle_boxes in corridor.items():
            distances = [_box_distance(a,b) for a in body_boxes for b in obstacle_boxes]
            pairs.append({"body":body, "obstacle":obstacle,
                "distance_bounds":[min(row[0] for row in distances),min(row[1] for row in distances)]})
    return {"kind":"resource_owned_transfer_corridor", "source_fingerprint":_digest(contract),
        "physical_start_fingerprint":contract["physical_start_fingerprint"],
        "resource_population":sorted(population), "robot_resources":robots,
        "resource_model_fingerprints":{rid:_digest(models[rid]["configuration"]) for rid in robots},
        "declarations":deepcopy(declarations), "stationary_resource_evidence":holds,
        "body_boxes":bodies, "corridor_objects":corridor, "pairs_complete":True, "pairs":pairs,
        "physical_execution_verified":False}


def _transport_clearance_value(evidence: dict, observation: dict) -> list[bool]:
    envelope = observation["owner_safety_evidence"]
    contract = envelope.get("owner_interval_contract")
    if (not isinstance(contract, dict) or contract.get("kind") != "resource_owned_shared_transport"
            or evidence.get("source_fingerprint") != _digest(contract)
            or evidence.get("physical_start_fingerprint") != contract.get("physical_start_fingerprint")
            or evidence.get("pairs_complete") is not True
            or set(evidence.get("resource_population", [])) != set(observation["resources"])):
        raise ValueError("Robot clearance differs from the validated transfer corridor")
    robots = evidence["robot_resources"]
    expected_models = envelope.get("resource_kinematic_model_fingerprints")
    if (not isinstance(expected_models, dict) or not expected_models
            or set(robots) != set(expected_models)
            or evidence["resource_model_fingerprints"] != expected_models):
        raise ValueError("Transfer corridor omits part of the captured manipulator population")
    expected_bodies = set(robots)
    for rid in robots:
        held = observation["resources"][rid]["held_part"]
        if held is not None:
            expected_bodies.add(held)
    bodies, corridor = evidence["body_boxes"], evidence["corridor_objects"]
    if set(bodies) != expected_bodies or set(corridor) != set(contract["event"]["participants"]+contract["part_names"]):
        raise ValueError("Transfer corridor omits required body or source/destination geometry")
    actual = {(row["body"],row["obstacle"]):row["distance_bounds"] for row in evidence["pairs"]}
    if len(actual) != len(evidence["pairs"]) or set(actual) != {(a,b) for a in bodies for b in corridor}:
        raise ValueError("Transfer corridor pair inventory is incomplete")
    values = []
    for body, body_boxes in bodies.items():
        for obstacle, obstacle_boxes in corridor.items():
            distances = [_box_distance(a,b) for a in body_boxes for b in obstacle_boxes]
            bounds = [min(row[0] for row in distances),min(row[1] for row in distances)]
            if _digest(bounds) != _digest(actual[body,obstacle]):
                raise ValueError("Transfer corridor distance differs from its full component bounds")
            values.append([True] if bounds[0] > 0 else [False,True])
    return _all(values)


def transport_belt_stopped(resource: str, observation: dict, *, interval: list | None = None) -> list[bool] | None:
    """Calculate the declared transport profile's belt_stopped alternatives."""
    records = observation.get("owner_safety_evidence", {}).get("resource_evidence", {})
    if not isinstance(records, dict) or "transport_motion" not in records.get(resource, {}):
        return None
    record = _record(resource, observation)
    motion = record["transport_motion"]
    from cais_spade_llm.recovery_framework.workflow_gazebo import _motion_duration

    source = motion["source"]
    if source.get("kind") != "projected" or not source.get("fingerprint"):
        raise ValueError("Transport motion lacks its exact owner calculation")
    contract = observation["owner_safety_evidence"].get("owner_interval_contract")
    if (not isinstance(contract, dict) or contract.get("kind") != "resource_owned_shared_transport"
            or source["fingerprint"] != _digest(contract)
            or any(_digest(motion[field]) != _digest(contract.get(field)) for field in
                   ("displacement_m", "speed_mps", "acceleration_mps2", "duration_sec"))):
        raise ValueError("Transport motion differs from its validated owner interval")
    distance, speed, acceleration, duration = (float(_number(motion[field])) for field in
        ("displacement_m", "speed_mps", "acceleration_mps2", "duration_sec"))
    if min(distance, speed, acceleration, duration) <= 0 or _motion_duration(distance, speed, acceleration) != duration:
        raise ValueError("Transport duration differs from the declared motion profile")
    first, last = map(lambda value: Fraction(str(value)), motion["interval"])
    if last-first != Fraction(str(duration)):
        raise ValueError("Transport motion interval differs from its exact profile")
    observed_interval = (observation.get("continuous_interval")
        if observation.get("phase") == "between" else [observation["time_exact"]]*2)
    a, b = map(lambda value: Fraction(str(value)), interval or observed_interval)
    outer_a, outer_b = map(lambda value: Fraction(str(value)), observed_interval)
    if not outer_a <= a <= b <= outer_b:
        raise ValueError("Transport cell exceeds its owner observation interval")
    if not first <= a <= b <= last:
        raise ValueError("Transport observation lies outside its declared profile")
    if a == b and a in (first, last):
        return [True]
    return [False] if first < a <= b < last else [False, True]


def _motion(record: dict, field: str, names: list[str]) -> list[bool]:
    values = record.get(field)
    if not isinstance(values, dict) or set(values) != set(names) or not names:
        raise ValueError("Motion evidence omits a configured velocity bound")
    bounds = [_bounds(values[name]) for name in names]
    if all(lo == hi == 0 for lo, hi in bounds):
        return [False]
    if any(lo > 0 or hi < 0 for lo, hi in bounds):
        return [True]
    return [False, True]


def _clearance(record: dict, key: str, expected_bodies: set[str]) -> list[bool]:
    evidence = record.get(key)
    if not isinstance(evidence, dict) or evidence.get("pairs_complete") is not True:
        raise ValueError("Complete owner clearance bounds are unavailable")
    # Pair enumeration is checked against the captured scene inventory, not a
    # caller-supplied 'clear' value or only the pairs which happened to collide.
    obstacles = record.get("collision_object_ids")
    if (not isinstance(obstacles, list) or not obstacles or len(set(obstacles)) != len(obstacles)
            or not all(isinstance(item, str) and item for item in obstacles)):
        raise ValueError("Clearance requires the complete captured collision-object inventory")
    expected = {(body, obstacle) for body in expected_bodies for obstacle in obstacles
                if body != obstacle}
    rows = evidence.get("pairs")
    if not isinstance(rows, list):
        raise ValueError("Clearance pair bounds are unavailable")
    actual = [(row["body"], row["obstacle"]) for row in rows]
    if len(set(actual)) != len(actual) or set(actual) != expected:
        raise ValueError("Clearance bounds omit or duplicate a required collision pair")
    configuration = record["configuration"]
    if "minimum_clearance_m" in configuration:
        limit = _number(configuration["minimum_clearance_m"])
    elif configuration.get("task_execution", {}).get("avoid_collisions") is True:
        limit = Fraction(0)
    else:
        raise ValueError("Carried clearance lacks a declared gap or strict collision-avoidance policy")
    if limit < 0:
        raise ValueError("Declared minimum clearance cannot be negative")
    values = []
    for row in rows:
        lo, hi = _bounds(row["distance_bounds"])
        separated = lo >= limit if limit > 0 else lo > 0
        values.append([True] if separated else [False] if hi < limit else [False, True])
    return _all(values)


def _shared_handoff_event(event: dict, task: dict, observation: dict) -> dict:
    from cais_spade_llm.resources.environment_models import _apply_resource_effects

    envelope = observation["owner_safety_evidence"]
    contract = envelope.get("owner_interval_contract", {})
    source = contract.get("event")
    selected = task["declared_task"]
    owner = selected.get("resource_id")
    if (contract.get("kind") != "resource_owned_shared_transport" or not isinstance(source, dict)
            or source.get("collection_effects", {}).get("validation_source") != "cais_spade_llm.resources.nominal_conveyor"
            or owner not in source.get("participants", [])):
        raise ValueError("Conditional hand-off effects need supported owner semantics")
    models = {}
    for participant in source["participants"]:
        declarations = envelope.get("resource_evidence", {}).get(participant, {}).get("handoff_declarations")
        fingerprint = envelope.get("resource_event_fingerprints", {}).get(participant)
        if not isinstance(declarations, list) or not fingerprint or _digest(declarations) != fingerprint:
            raise ValueError("Shared hand-off omits an exact participant declaration")
        models[participant] = {"events": declarations}
    source_matches = [row for row in models[owner]["events"] if row.get("event_id") == selected.get("event_id")]
    if len(source_matches) != 1 or _digest(source_matches[0]) != _digest(source):
        raise ValueError("Shared hand-off source differs from its captured owner declaration")
    for key in ("event_id", "event_name", "parameter_bindings", "participants", "product_effects"):
        if key not in event or key not in source or _digest(event[key]) != _digest(source[key]):
            raise ValueError("Shared hand-off participant declarations disagree")
    before = {row["resource_id"]: deepcopy(row["values"])
              for row in envelope["state_evidence"]["initial"]}
    for update in envelope["state_evidence"]["updates"]:
        if Fraction(str(update["time"])) <= Fraction(str(task["start_time"])):
            before[update["resource_id"]].update(deepcopy(update["values"]))
    after = deepcopy(before)
    _apply_resource_effects(models, before, after, source, selected["parameters"])
    actual = deepcopy(before[task["resource_id"]])
    for update in envelope["state_evidence"]["updates"]:
        if update["task_id"] == task["task_id"]:
            actual.update(deepcopy(update["values"]))
    if _digest(actual) != _digest(after[task["resource_id"]]):
        raise ValueError("Shared hand-off participant effects differ from the exact task ledger")
    return source


def _handoffs(record: dict, resource: str, observation: dict) -> list[dict]:
    ledger = record.get("handoffs")
    if not isinstance(ledger, dict) or ledger.get("complete") is not True:
        raise ValueError("Complete owner hand-off records are unavailable")
    rows = ledger.get("records")
    if not isinstance(rows, list):
        raise ValueError("Owner hand-off inventory is unavailable")
    tasks = observation["owner_safety_evidence"].get("task_evidence", {}).get("events")
    if not isinstance(tasks, list):
        raise ValueError("Hand-off evidence requires the complete owner task ledger")
    task_ids = [row.get("task_id") for row in rows]
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("Hand-off evidence duplicates a task")
    # The owner declares which exact functions change custody. Unknown task
    # semantics cannot establish that an empty ledger covers the interval.
    declarations = record.get("handoff_declarations")
    if declarations is not None:
        expected = observation["owner_safety_evidence"].get("resource_event_fingerprints", {}).get(resource)
        if not isinstance(declarations, list) or not expected or _digest(declarations) != expected:
            raise ValueError("Hand-off declarations differ from their recorded owner model")
        relevant = set()
        for task in tasks:
            if task["resource_id"] != resource:
                continue
            declared_task = task.get("declared_task")
            if not isinstance(declared_task, dict):
                raise ValueError("Conditional hand-off needs its exact declared task binding")
            matches = [row for row in declarations if row.get("event_id") == declared_task.get("event_id")
                       and row.get("event_name") == declared_task.get("event_name") == task["function"]]
            if len(matches) != 1:
                raise ValueError("Conditional hand-off declaration is missing or ambiguous")
            event = matches[0]
            if resource not in event.get("participants", []) or declared_task.get("resource_id") not in event["participants"]:
                raise ValueError("Hand-off task owner differs from its exact shared participants")
            if event.get("collection_effects", {}).get("validation_source") != "cais_spade_llm.resources.nominal_conveyor":
                event = _shared_handoff_event(event, task, observation)
            declaration = event.get("parameter_bindings", {}).get("delivered_part")
            params = declared_task.get("parameters", {})
            if not isinstance(declaration, dict) or "delivered_part" not in params:
                raise ValueError("Conditional hand-off lacks its explicit delivered_part binding")
            part = params["delivered_part"]
            if "equals" in declaration and _digest(part) != _digest(declaration["equals"]):
                raise ValueError("Conditional hand-off parameter conflicts with its declaration")
            if part is None:
                if declaration.get("equals", "missing") is not None:
                    raise ValueError("A null delivered_part lacks an exact no-delivery declaration")
                continue
            if not isinstance(part, str) or part not in observation["parts"]:
                raise ValueError("Conditional hand-off delivered_part is not a known part")
            matching = [row for row in rows if row.get("task_id") == task["task_id"]]
            if len(matching) != 1 or matching[0].get("part_name") != part:
                raise ValueError("Conditional hand-off record changes its exact part binding")
            relevant.add(task["task_id"])
    else:
        functions = record.get("configuration", {}).get("handoff_functions")
        if not isinstance(functions, list) or not functions or any(not isinstance(v, str) for v in functions):
            raise ValueError("Complete declared hand-off function inventory is unavailable")
        relevant = {task["task_id"] for task in tasks
                    if task["resource_id"] == resource and task["function"] in functions}
    if set(task_ids) != relevant:
        raise ValueError("Owner hand-off ledger omits a declared hand-off task")
    for row in rows:
        if (row["part_name"] not in observation["parts"]
                or row["source_resource"] not in observation["resources"]
                or row["destination_resource"] not in observation["resources"]
                or row["source_resource"] == row["destination_resource"]):
            raise ValueError("Hand-off records require exact modeled identities")
        task = next(task for task in tasks if task["task_id"] == row["task_id"])
        time = Fraction(observation["time_exact"])
        expected_phase = "completed" if time >= Fraction(str(task["end_time"])) else "pending"
        if row.get("phase") != expected_phase:
            raise ValueError("Hand-off phase disagrees with its exact task-completion time")
    return rows


def case_ap_value(kind: str, resource: str, observation: dict) -> list[bool]:
    """Return every AP truth value supported by the complete owner bounds."""
    record = _record(resource, observation)
    configuration = record.get("configuration", {})
    declarations = observation["owner_safety_evidence"].get("resource_configurations")
    if not isinstance(configuration, dict) or not isinstance(declarations, dict):
        raise ValueError("Captured owner safety configuration is unavailable")
    declared = declarations.get(resource)
    if not isinstance(declared, dict) or _digest(configuration) != _digest(declared):
        raise ValueError("Raw owner safety bounds changed their captured configuration")
    names = configuration.get("arm_joint_names")
    if kind == "base_moving":
        return _motion(record, "base_velocity_bounds", ["x", "y", "yaw"])
    if kind in {"arm_moving", "arm_parked"}:
        if not isinstance(names, list) or not names or len(set(names)) != len(names):
            raise ValueError("Configured arm joint names are unavailable")
        if kind == "arm_moving":
            return _motion(record, "joint_velocity_bounds", names)
        targets = configuration["parked_arm_configuration"]
        values = record["joint_position_bounds"]
        if len(targets) != len(names) or set(values) != set(names):
            raise ValueError("Parked-arm evidence omits a configured joint")
        return _all([_inside(values[name], target, configuration["base_control"]["arm_parked_tolerance_rad"])
                     for name, target in zip(names, targets, strict=True)])
    if kind == "docked":
        target = record["docking_targets"][record["docking_target"]]
        if not isinstance(target, list) or len(target) != 6:
            raise ValueError("Docking requires the exact configured six-value pose")
        bounds = record["base_pose_bounds"]
        control = configuration["base_control"]
        values = [_inside(bounds[axis], target[index], control["position_tolerance_m"])
                  for index, axis in enumerate(("x", "y"))]
        lo, hi = _bounds(bounds["yaw"])
        if hi - lo >= Fraction(str(2 * pi)):
            values.append([False, True])
        else:
            shift = round((float((lo + hi) / 2) - target[5]) / (2 * pi))
            # A wrapped branch needs certified angular interval arithmetic;
            # do not round a multiple of pi into an affirmative dock proof.
            values.append([False, True] if shift else
                          _inside(bounds["yaw"], target[5], control["yaw_tolerance_rad"]))
        return _all(values)
    if kind in {"carried_part_clearance", "robot_clear"}:
        if record.get("collision_object_ids") != observation["owner_safety_evidence"].get("collision_object_ids"):
            raise ValueError("Clearance collision inventory differs from the captured scene")
    if kind == "carried_part_clearance":
        held = observation["resources"][resource]["held_part"]
        if "held_part" not in record or record["held_part"] != held:
            raise ValueError("Carried-part clearance custody disagrees with the physical checkpoint")
        if held is None:
            return [True]
        return _clearance(record, "carried_part_clearance", {held})
    rows = _handoffs(record, resource, observation)
    if kind == "handoff_identity":
        for row in rows:
            before, after = row["custody_before"], row["custody_after"]
            population = set(observation["resources"])
            if set(before) != population or set(after) != population:
                raise ValueError("Hand-off identity requires complete custody before and after")
            part = row["part_name"]
            if (any(not isinstance(parts, list) or len(set(parts)) != len(parts) for parts in (*before.values(), *after.values()))
                    or any(name not in observation["parts"] for parts in (*before.values(), *after.values()) for name in parts)):
                raise ValueError("Hand-off custody inventory is malformed")
            current = {}
            for rid, physical in observation["resources"].items():
                if "held_part" not in physical and "contained_parts" not in physical:
                    raise ValueError("Hand-off identity lacks current complete physical custody")
                inventory = list(physical.get("contained_parts", []))
                if physical.get("held_part") is not None:
                    inventory.append(physical["held_part"])
                current[rid] = inventory
            if _digest(current) != _digest(after):
                raise ValueError("Hand-off custody differs from the physical checkpoint")
            if any(sum(name in parts for parts in custody.values()) > 1
                   for custody in (before, after) for name in observation["parts"]):
                return [False]
            expected_before = [row["source_resource"]]
            expected_after = [row["destination_resource"]] if row["phase"] == "completed" else expected_before
            if ([rid for rid in before if part in before[rid]] != expected_before
                    or [rid for rid in after if part in after[rid]] != expected_after):
                return [False]
        return [True]
    if kind == "handoff_acknowledged":
        for row in rows:
            if row["phase"] != "completed":
                continue
            ack = row.get("acknowledgement")
            if not isinstance(ack, dict) or not ack.get("source_fingerprint"):
                raise ValueError("Completed hand-off lacks its recorded acknowledgement")
            if ack.get("kind") not in {"recorded", "projected"}:
                raise ValueError("A hand-off command parameter is not acknowledgement evidence")
            if any(ack.get(key) != row[key] for key in ("task_id", "part_name", "source_resource", "destination_resource")):
                return [False]
            if any(type(ack.get(field)) is not bool for field in ("source_departed", "destination_arrived")):
                raise ValueError("Hand-off acknowledgement omits departure or arrival evidence")
            if ack["source_departed"] is not True or ack["destination_arrived"] is not True:
                return [False]
            if ack["kind"] == "projected":
                updates = observation["owner_safety_evidence"].get("state_evidence", {}).get("updates", [])
                effects = [update for update in updates if update["task_id"] == row["task_id"]]
                if not effects or ack.get("owner_effect_fingerprint") != _digest(effects):
                    raise ValueError("Projected hand-off completion lacks its exact validated owner effect")
        return [True]
    if kind == "robot_clear":
        if not rows:
            return [True]
        if record.get("robot_clear", {}).get("kind") == "resource_owned_transfer_corridor":
            return _transport_clearance_value(record["robot_clear"], observation)
        robots = record.get("configuration", {}).get("clearance_resources")
        if not isinstance(robots, list) or not robots or not set(robots) <= set(observation["resources"]):
            raise ValueError("Declared hand-off clearance population is unavailable")
        bodies = set(robots)
        for robot in robots:
            held = observation["resources"][robot]["held_part"]
            if held is not None:
                bodies.add(held)
        return _clearance(record, "robot_clear", bodies)
    raise ValueError("Unsupported owner safety condition")
