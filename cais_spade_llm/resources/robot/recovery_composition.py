from __future__ import annotations

"""Robot-owned calculations, step checks and completion evidence for authored recovery programs."""

from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
import math
from typing import Any

from cais_spade_llm.agents.shared_information.recovery_validation_protocol import recovery_validation_fingerprint

_POSE_AXES = ("x", "y", "z", "qx", "qy", "qz", "qw")
# The motions the composition's MoveIt check flies as the ur5e controller does.
_MOTIONS = ("move_cartesian", "move_relative", "move_to_named_pose")
# Arm motions a robot catalog may name that the MoveIt check has no route for.
_UNMODELLED_MOTIONS = ("move_pose", "move_joints", "rotate_joint", "move_insert")


def _calculation_inputs(primitive: str, part: str | None, calculated: dict, *, task: dict,
                        grounding_context: dict, resource_evidence: dict) -> dict:
    """Return the value of each parameter the composition's owner calculation reads, as it reads it.

    ``compute_pick_targets`` applies what the robot applies to an observed part
    (``RobotAgent._inject_observed_pose_for_pick_targets``); ``compute_place_targets`` reads the recorded pick
    context and the place the event states. A parameter not listed here is one the calculation does not model.
    """
    if primitive == "compute_pick_targets":
        motion = resource_evidence.get("motion_config") or resource_evidence.get("controller_config", {}).get("motion") or {}
        return {"part_name": part, "target_pose": None, "product_geometry": None, "detected_parts": None,
                "target_pose_source": "", "prefer_live_detection": False,
                "approach_height_override_m": motion.get("recovery_observed_pick_approach_height_m"),
                "ignore_current_height_for_travel_z": True, "min_pick_tcp_z_override_m": None,
                "use_global_min_pick_tcp_z": False,
                "surface_clearance_override_m": motion.get("recovery_observed_pick_surface_clearance_m"),
                "apply_pick_z_adjustments": False}
    # Owner preparation may add placement outputs to its private _task_ctx before this preview; the recorded
    # context is the one the event started from.
    recorded = grounding_context.get("resources", {}).get(task["resource_jid"], {}).get(
        "resource_evidence", resource_evidence)
    pick_ctx = recorded.get("state", {}).get("_task_ctx")
    attachment = resource_evidence.get("attachment") or {}
    # Held through the event's own grasp, the part's pick context is the one that grasp recorded.
    if attachment.get("part_name") == part and (attachment.get("source") or {}).get("kind") == "projected":
        pick_ctx = (resource_evidence.get("state") or {}).get("_task_ctx")
    return {"part_name": part, "pick_ctx": pick_ctx, "product_geometry": None,
            "destination_location": calculated["destination_location"], "z_adjustment_m": 0.,
            "assembly_board_v1_aruco": None}


def _parameter_facts(primitive: str, params: dict, inputs: dict, task: dict) -> list[str]:
    """State each given parameter the composition's calculation does not read as given."""
    facts = []
    for field, value in params.items():
        if field in inputs and value == inputs[field]:
            continue
        if field not in inputs:
            facts.append(f"{primitive}: the composition's calculation does not model {field}")
        elif field == "part_name":
            facts.append(f"{primitive} part_name {value!r} differs from the event's part {inputs[field]!r}")
        elif field == "destination_location":
            stated = "part_location" if isinstance((task.get("expected_end_state") or {}).get("part_location"), str) \
                else "resource_location"
            facts.append(f"{primitive} destination_location {value!r} differs from the event's {stated} "
                         f"{inputs[field]!r}")
        elif field == "prefer_live_detection" and value is True:
            facts.append(f"{primitive}: prefer_live_detection reads a live detection the capture does not record")
        elif field == "pick_ctx":
            facts.append(f"{primitive}: pick_ctx differs from the pick context the composition's calculation reads, "
                         f"/resources/{task.get('resource_jid')}/resource_evidence/state/_task_ctx")
        else:
            facts.append(f"{primitive}: {field} {value!r} differs from {inputs[field]!r}, the input the "
                         "composition's calculation reads")
    return facts


def preview_composition_output(*, primitive: str, params: dict, recovery_snapshot: dict,
                               part_context: dict, grounding_context: dict, task: dict,
                               resource_evidence: dict, physical_evidence: dict) -> dict:
    """Calculate target outputs from recorded owner facts, without legacy defaults.

    A given calculation parameter holds only where it equals the input the composition's calculation reads;
    any other is refused with that fact, as is one the calculation does not model.

    Args:
        primitive: The step's primitive.
        params: The step's resolved parameters.
        recovery_snapshot: The acting resource's state when the step starts.
        part_context: The event's part as the composition prefix leaves it.
        grounding_context: The grounded state, per resource.
        task: The outline event.
        resource_evidence: The owner's recorded evidence, such as ``observed_parts``.
        physical_evidence: The capture's runtime models, valuation and geometry.

    Returns:
        The calculated output with its ``event_fact_key``, a ``NEEDS_CONTEXT`` status with its reason, or
        nothing for a primitive with no calculated output.
    """
    from cais_spade_llm.resources.robot.recovery_capability import (
        _pose_mapping, calculate_recovery_pick_target, calculate_recovery_place_target,
    )

    part = task.get("part_name") if primitive in {"compute_pick_targets", "compute_place_targets"} else \
        params.get("part_name", task.get("part_name"))
    if primitive in {"compute_pick_targets", "compute_place_targets"}:
        if primitive == "compute_pick_targets":
            calculated = calculate_recovery_pick_target(task=task, recovery_snapshot=recovery_snapshot,
                                                       resource_evidence=resource_evidence)
            key = "pick_targets"
        else:
            calculated = calculate_recovery_place_target(task=task, recovery_snapshot=recovery_snapshot,
                resource_evidence=resource_evidence, physical_evidence=physical_evidence)
            key = "place_targets"
        if calculated.get("status") != "calculated":
            return {"status": "NEEDS_CONTEXT", "reason": str(calculated.get("missing_facts"))}
        facts = _parameter_facts(primitive, params, _calculation_inputs(
            primitive, part, calculated, task=task, grounding_context=grounding_context,
            resource_evidence=resource_evidence), task)
        if facts:
            return {"status": "NEEDS_CONTEXT", "reason": "; ".join(facts)}
        output = deepcopy(calculated)
        if "insert_pose" in output:
            from cais_spade_llm.resources.robot.target_calculations import placement_poses

            pose = output["insert_pose"]
            depth = resource_evidence.get("controller_config", {}).get("parts_tuning", {}).get("insertion_depth_m")
            if type(depth) not in (int, float) or not math.isfinite(depth) or depth < 0:
                return {"status": "NEEDS_CONTEXT", "reason": "The recorded placement insertion depth is missing"}
            output.update(placement_poses(pose["x"], pose["y"], pose["z"],
                {key: pose[key] for key in ("qx", "qy", "qz", "qw")},
                simulation_assembly_slot=output.get("target_reference", {}).get("surface_role") == "assembly_slot",
                insertion_depth=depth))
            # A part going into its slot is released where it is seated, not at the pre-insert height above it.
            if output.get("target_reference", {}).get("surface_role") == "assembly_slot":
                output["target_pose"] = dict(output["insert_pose"])
        output["model_name"] = resource_evidence.get("inputs", {}).get("geometry", {}).get("parts", {}).get("model_map", {}).get(part)
        travel_z = (output.get("travel_z") if primitive == "compute_pick_targets"
                    else resource_evidence.get("state", {}).get("_task_ctx", {}).get("travel_z"))
        if output.get("access_retreat_pose"):
            output["retreat_pose"] = deepcopy(output["access_retreat_pose"])
        elif type(travel_z) in (int, float) and math.isfinite(travel_z):
            # A place can sit above the pick's travel height; its retreat still rises, at least to its own approach.
            output["retreat_pose"] = {**output["target_pose"],
                                      "z": max(travel_z, output.get("approach_pose", {}).get("z", travel_z))}
        return {"event_fact_key": f"{key}.{part}", "output": output, "status": "calculated"}
    if primitive == "get_current_pose":
        return {"event_fact_key": "current_pose", "output": deepcopy(recovery_snapshot.get("current_pose")), "status": "calculated"}
    if primitive == "detect_parts":
        # The record the capability check and compute_pick_targets read; part_context has no pose for a part
        # only the owner observed.
        observed = (resource_evidence.get("observed_parts") or {}).get(part) or {}
        pose = _pose_mapping(observed.get("observed_pose", observed.get("current_pose")))
        if not pose:
            return {"status": "NEEDS_CONTEXT", "reason": "Recorded part observation is missing"}
        return {"event_fact_key": f"part_pose.{part}", "output": {"part_name": part, "pose": deepcopy(pose), **pose}, "status": "calculated"}
    return {}


def _source_records(part: str, acting_jid: str, grounding_context: dict, physical_evidence: dict,
                    stated: dict, origin: Any = None) -> list[tuple]:
    """Return the stated values that record the part where it rested, with the value each takes once it has left.

    As the RA's pick finds them (``ra_feasibility._Check.source_variables``): a resource-scope value that is the
    part, or a per-part variable for the part that is not null or false; it becomes null, or false where its
    declared values have no null. Code writes no value the event leaves out, so only the values its
    ``neighbor_end_state`` (``stated``) names are returned. The RA refuses one left out unless it is private or a
    ``task_ctx.`` or ``_`` name, and the postcondition compares the stated ones with these. What the owner's model
    declares taking the part from that slot leaves (``received_values``, such as a machine left idle) is returned
    for the values the event states too.
    """
    from cais_spade_llm.resources.environment_models import received_values

    records = []
    for owner, model in physical_evidence.get("environment_models", {}).items():
        jid = physical_evidence.get("jids", {}).get(owner)
        if jid is None or jid == acting_jid:
            continue
        declarations = {name.replace("{part_name}", part): declaration
                        for name, declaration in model.get("state_variables", {}).items()}
        values = {**physical_evidence.get("environment_valuation", {}).get(owner, {}),
                  **{key: value for key, value in grounding_context.get("resources", {}).get(jid, {}).items()
                     if key in declarations}}
        for name, declaration in declarations.items():
            if (declaration or {}).get("scope") == "part" or name not in values or name not in (stated.get(jid) or {}):
                continue
            value = values[name]
            if value == part or (name.endswith("." + part) and value is not None and value is not False):
                domain = (declaration or {}).get("domain") or []
                nullable = not domain or any(item is None for item in domain)
                records.append((jid, name, None if nullable or not any(item is False for item in domain) else False))
                if isinstance(origin, str):
                    received, _ = received_values(physical_evidence.get("environment_models", {}), owner, origin, name,
                                                  part, values, taken=True)
                    records.extend((jid, field, found) for field, found in received.items()
                                   if field != name and field in (stated.get(jid) or {}))
    return records


@dataclass(frozen=True)
class RobotEventScene:
    """What a robot's step checks read besides the step and the robot's own evidence.

    Attributes:
        task: The accepted outline event.
        program: The authored program as written; a motion's check reads which transfer follows it.
        resource_id: The robot's resource id.
        grounding_context: The other resources as the accepted composition prefix leaves them.
        physical_evidence: The capture's runtime models, geometry and physics.
        recovery_des_model: The robot's declared model.
        motion_model: Loads the capture's MoveIt model; raises ``MissingMotionContext`` when it cannot.
    """

    task: dict
    program: list
    resource_id: str
    grounding_context: dict
    physical_evidence: dict
    recovery_des_model: dict
    motion_model: Callable[[], Any]


def _pose_list(value: Any) -> list[float] | None:
    """A pose as ``[x, y, z, qx, qy, qz, qw]`` when it is a complete one, recorded as a list or by named fields."""
    if isinstance(value, dict):
        value = [value.get(axis) for axis in _POSE_AXES]
    if (not isinstance(value, list) or len(value) != 7
            or any(type(item) not in (int, float) or not math.isfinite(item) for item in value)):
        return None
    return [float(item) for item in value]


def _inverse(pose: list[float]) -> list[float]:
    from cais_spade_llm.recovery_framework.geometry import rotate

    rotation = [-pose[3], -pose[4], -pose[5], pose[6]]
    return [*rotate(rotation, [-value for value in pose[:3]]), *rotation]


def _xyz(pose: list[float]) -> str:
    return "(" + ", ".join(f"{value:.4f}" for value in pose[:3]) + ")"


def _carried_joints(evidence: dict, scene: RobotEventScene) -> list[float]:
    """The arm joints the robot's program has reached: carried in its evidence, else the captured ones."""
    joints = (evidence.get("snapshot") or {}).get("joint_positions")
    if not isinstance(joints, list):
        joints = scene.physical_evidence["snapshot"]["resources"][scene.resource_id]["joint_positions"]
    return [float(value) for value in joints]


def _tool_pose(scene: RobotEventScene, joints: list[float]) -> list[float]:
    """The reference link pose (tool0) of the robot's recorded chain at the joints."""
    from scipy.spatial.transform import Rotation

    from cais_spade_llm.resources.kinematic_reach import chain_frames

    configuration = scene.physical_evidence["resource_models"][scene.resource_id]["configuration"]
    frame = chain_frames(configuration, dict(zip(configuration["joint_names"], joints, strict=True)))[
        configuration["reference_link"]]
    return [*map(float, frame[:3, 3]), *map(float, Rotation.from_matrix(frame[:3, :3]).as_quat())]


def _held_grasp(evidence: dict, scene: RobotEventScene, part: str) -> list[float] | None:
    """The grasp transform (tool0 frame) by which the robot holds a part: its evidence's, else the captured one."""
    attachment = evidence.get("attachment") or {}
    if attachment.get("part_name") == part and attachment.get("grasp_transform"):
        return [float(value) for value in attachment["grasp_transform"]]
    captured = scene.physical_evidence["snapshot"]["resources"][scene.resource_id]
    if captured.get("held_part") == part and captured.get("grasp_transform"):
        return [float(value) for value in captured["grasp_transform"]]
    return None


def check_robot_step(evidence: dict, targets: dict, scene: RobotEventScene, *, index: int, primitive: str,
                     params: dict, recovery_snapshot: dict, checked: dict) -> dict:
    """Check one step of a robot program physically, in the cell its earlier steps and the accepted prefix leave.

    A motion is checked by MoveIt on the capture's recorded model and scene (``composed_motion_check``); a grasp by
    the controller's payload check at the actual tool pose; a release by where the part comes to rest. The owner's
    capability result comes back with the facts the check found, which ``advance_robot_program`` carries, or as a
    refusal stating the fact.

    Args:
        evidence: The robot's evidence as its earlier steps leave it.
        targets: The event's calculated targets, per calculation and part.
        scene: The event's fixed inputs.
        index: The step's index in the program.
        primitive: The step's primitive.
        params: The step's resolved parameters.
        recovery_snapshot: The robot's declared state at the step's start.
        checked: The owner's capability result for the step.

    Returns:
        ``checked`` with what the check found added to its ``evidence``, or a refusal.
    """
    written = (scene.program[index] or {}).get("primitive") if index < len(scene.program) else None
    if written != primitive:
        raise RuntimeError(f"Step {index} of the program is {written!r}, not the checked {primitive!r}")
    if primitive in _MOTIONS:
        found = _motion_check(evidence, scene, index=index, primitive=primitive, params=params,
                              recovery_snapshot=recovery_snapshot)
    elif primitive in _UNMODELLED_MOTIONS:
        found = {"status": "NEEDS_CONTEXT", "reason": f"{primitive} step {index} ({scene.resource_id}): the "
                                                      f"composition's MoveIt check does not model {primitive}"}
    elif primitive == "grasp_part":
        found = _grasp_check(evidence, scene, index=index, params=params)
    elif primitive == "release_part":
        found = _release_check(evidence, targets, scene, index=index, params=params,
                               recovery_snapshot=recovery_snapshot)
    else:
        return checked
    if "status" in found:
        return {"allowed": False, "feasibility_status": found["status"], "reason": found["reason"],
                "evidence": {**(checked.get("evidence") or {}), **(found.get("evidence") or {})}}
    return {**checked, "evidence": {**(checked.get("evidence") or {}), **found}}


def _robot_views(evidence: dict, scene: RobotEventScene, recovery_snapshot: dict) -> dict:
    """Every robot's composition view: the acting robot at the step's start, the others as the prefix leaves them."""
    views = {}
    for rid, jid in scene.physical_evidence.get("jids", {}).items():
        if rid == scene.resource_id:
            views[rid] = {**{field: recovery_snapshot[field] for field in ("held_part", "gripper_state")
                             if field in recovery_snapshot}, "resource_evidence": evidence}
            continue
        current = (scene.grounding_context.get("resources") or {}).get(jid)
        if isinstance(current, dict):
            views[rid] = {**{field: current[field] for field in ("held_part", "gripper_state", "base_pose")
                             if field in current}, "resource_evidence": current.get("resource_evidence") or {}}
    return views


def _next_transfer(program: list, index: int) -> dict | None:
    """The transfer the program takes right after step ``index``, when no motion, nor a base move, comes between."""
    for step in program[index + 1:]:
        name = (step or {}).get("primitive")
        if name in ("grasp_part", "release_part"):
            return step
        if name in _MOTIONS or name in _UNMODELLED_MOTIONS or name == "move_base":
            return None
    return None


def _slot_origin(evidence: dict, part: str) -> dict:
    """Where a part's origin sits in its assembly slot, as the recorded board geometry declares it."""
    from cais_spade_llm.recovery_framework.composed_motion_check import MissingMotionContext
    from cais_spade_llm.resources.robot.recovery_capability import _destination_geometry

    declared = _destination_geometry(evidence, "assembly_board-v1", part) or {}
    origin = declared.get("target_origin_pose")
    if not isinstance(origin, dict):
        raise MissingMotionContext(f"inputs.geometry.assembly_board records no slot for {part}")
    return {axis: float(origin[axis]) for axis in ("x", "y", "z")}


def _release_contacts(model: Any, state: dict, scene: RobotEventScene, evidence: dict, move: dict, *,
                      held: str | None) -> tuple[list, dict | None]:
    """Contacts the motion right before a release makes by design: its arrival on the support, or its insertion.

    The descent before a release onto a support may end with the part on the support that holds it there
    (``projected_part_support`` at the descent's end): that one pair is allowed for this one motion. The motion
    before a release into an assembly slot is the insertion, checked within the part's recorded corridor.
    """
    from cais_spade_llm.recovery_framework import composed_motion_check as motion
    from cais_spade_llm.recovery_framework.geometry import compose

    release = _next_transfer(scene.program, move["step_index"])
    if held is None or release is None or release.get("primitive") != "release_part":
        return [], None
    end = motion.motion_end_pose(model, state, scene.resource_id, move)
    if end is None:
        return [], None
    if "assembly_slot" in (release.get("params") or {}):
        return [], motion.mating_contact_record(model, state, scene.resource_id, evidence, part_name=held,
                                                target_origin_pose=_slot_origin(evidence, held), target_tool_pose=end)
    grasp = motion.projected_snapshot(model, state)["resources"][scene.resource_id]["grasp_transform"]
    support = motion.part_rest_support(model, held, compose(end, list(grasp)))
    if support.get("status") != "supported":
        return [], None
    return [motion.place_arrival_contact(model, held, support["support_collisions"])], None


def _motion_check(evidence: dict, scene: RobotEventScene, *, index: int, primitive: str, params: dict,
                  recovery_snapshot: dict) -> dict:
    """MoveIt's Cartesian path of one motion on the capture's model, from the joints the program has reached."""
    from cais_spade_llm.recovery_framework import composed_motion_check as motion

    rid = scene.resource_id
    move = {"step_index": index, "primitive": primitive, "params": params}
    held = recovery_snapshot.get("held_part")
    try:
        model = scene.motion_model()
        state = motion.composition_scene_state(model, _robot_views(evidence, scene, recovery_snapshot),
                                               evidence.get("observed_parts") or {})
        departure = None
        # The part a grasp took still rests on its support until the robot's first motion after the grasp.
        if held is not None and (evidence.get("attachment") or {}).get("resting_on_support"):
            departure = motion.support_departure_record(model, state, rid)
        contacts, mating = _release_contacts(model, state, scene, evidence, move, held=held)
    except motion.MissingMotionContext as exc:
        return {"status": "NEEDS_CONTEXT", "reason": f"{primitive} step {index} ({rid}): {exc}"}
    result = motion.check_robot_moves(model, state, rid, [move], allowed_contacts=contacts,
                                      support_departure=departure, mating=mating)
    summary = {"requests": result["requests"], "support_departure": departure is not None,
               "allowed_contacts": [{key: row[key] for key in ("pairs", "statement")} for row in contacts],
               "mating": mating is not None, "refinement_stops": result.get("refinement_stops") or []}
    if result["status"] != "FEASIBLE":
        reason = result["reason"] if result["step_index"] is not None else f"{primitive} step {index} ({rid}): " \
            f"{result['reason']}"
        return {"status": result["status"], "reason": reason, "evidence": {"composed_motion": summary}}
    step = result["steps"][-1]
    return {"composed_motion": {**summary, "route": step["route"], "end_joints": result["end_joints"],
                                "end_pose": step["end_pose"]}}


def _grasp_refusal(part_name: str, tool: list[float], part_pose: list[float], *, bounds: list,
                   offset: float, tolerance: float) -> str | None:
    """Return why a grasp with the tool at ``tool`` does not reach its part, by the controller's payload check.

    ``grasp_point_evidence`` reads the part's recorded bounds at its pose and the TCP, ``tcp_offset_z`` along the
    tool from the tool pose, against the recorded position tolerance.
    """
    from cais_spade_llm.recovery_framework.geometry import compose, rotate
    from cais_spade_llm.recovery_framework.part_collision import grasp_point_evidence

    center = [(low + high) / 2 for low, high in bounds]
    box = {"pose": compose(part_pose, [*center, 0., 0., 0., 1.]), "size": [high - low for low, high in bounds]}
    along_tool = rotate(tool[3:], [0., 0., abs(offset)])
    tcp = [tool[index] + along_tool[index] for index in range(3)]
    grasp = grasp_point_evidence([box], tcp, tolerance)
    if grasp["payload_at_gripper"]:
        return None
    return (f"The grasp's TCP at ({tcp[0]:.4f}, {tcp[1]:.4f}, {tcp[2]:.4f}) is {grasp['tcp_to_payload_distance_m']:.4f} m "
            f"from {part_name}, whose recorded box is centred at ({box['pose'][0]:.4f}, {box['pose'][1]:.4f}, "
            f"{box['pose'][2]:.4f}) with size {[round(value, 4) for value in box['size']]}, beyond the recorded "
            f"position tolerance {tolerance} m")


def _grasp_check(evidence: dict, scene: RobotEventScene, *, index: int, params: dict) -> dict:
    """The controller's payload check at the actual tool pose, and the attachment the grasp makes there."""
    from cais_spade_llm.recovery_framework.geometry import compose

    rid, part = scene.resource_id, params.get("part_name", scene.task.get("part_name"))
    prefix = f"grasp_part step {index} ({rid})"
    observed = (evidence.get("observed_parts") or {}).get(part) or {}
    inputs = {"pose (observed_parts)": _pose_list(observed.get("observed_pose", observed.get("current_pose"))),
              "local_bounds": scene.physical_evidence.get("geometry", {}).get("parts", {}).get(part, {}).get("local_bounds"),
              "tcp_offset_z": scene.physical_evidence.get("resources", {}).get(rid, {}).get("tcp_offset_z"),
              "position tolerance": evidence.get("controller_config", {}).get("move_group", {}).get("position_tolerance_m")}
    missing = [name for name, value in inputs.items() if value is None or value == []]
    if missing:
        return {"status": "NEEDS_CONTEXT", "reason": f"{prefix}: the grasp check of {part} needs the recorded "
                                                     f"{', '.join(missing)}"}
    pose = inputs["pose (observed_parts)"]
    tool = _tool_pose(scene, _carried_joints(evidence, scene))
    refusal = _grasp_refusal(part, tool, pose, bounds=inputs["local_bounds"], offset=inputs["tcp_offset_z"],
                             tolerance=inputs["position tolerance"])
    if refusal:
        return {"status": "INFEASIBLE", "reason": f"{prefix}: {refusal}"}
    event = next((row for row in scene.recovery_des_model.get("events", []) if row.get("event_name") == "pick_grasp"), {})
    attachment = {"part_name": part, "model_name": (evidence.get("inputs") or {}).get("geometry", {}).get("parts", {})
                  .get("model_map", {}).get(part), "grasp_transform": compose(_inverse(tool), pose),
                  "projected_attachment_complete": True, "resting_on_support": True,
                  "source": {"kind": "projected", "contract_fingerprint": recovery_validation_fingerprint(event)}}
    return {"grasp": {"tool_pose": tool, "part_pose": pose, "attachment": attachment}}


def _place_areas(place: str, regions: dict) -> dict:
    """A place's recorded areas, named as the RA names them (``ra_feasibility._Check.areas``).

    A region whose surface belongs to a recorded body is that body's; otherwise ``<place>`` and ``<place>.<area>``
    are the place's own.
    """
    areas = {}
    for name, row in regions.items():
        body = str((row.get("source") or {}).get("surface_collision") or "").split("::")[0]
        if row.get("bounds") and (body == place or not body and (name == place or name.startswith(place + "."))):
            areas[name] = row["bounds"]
    return areas


def _holds(bounds: list, pose: list[float]) -> bool:
    """Whether an area's footprint holds a point; an area's height is not a physical bound of what rests in it."""
    return all(low <= pose[axis] <= high for axis, (low, high) in enumerate(bounds[:2]))


def _release_check(evidence: dict, targets: dict, scene: RobotEventScene, *, index: int, params: dict,
                   recovery_snapshot: dict) -> dict:
    """Where the released part comes to rest: in its assembly slot, or on the support under it."""
    from cais_spade_llm.recovery_framework.geometry import compose

    rid, part = scene.resource_id, recovery_snapshot.get("held_part")
    prefix = f"release_part step {index} ({rid})"
    grasp = _held_grasp(evidence, scene, part)
    if grasp is None:
        return {"status": "NEEDS_CONTEXT", "reason": f"{prefix}: no recorded grasp says how {rid} holds {part}"}
    released = compose(_tool_pose(scene, _carried_joints(evidence, scene)), grasp)
    slot = params.get("assembly_slot")
    if slot is not None:
        return _slot_release(targets, scene, prefix=prefix, part=part, slot=slot, released=released, evidence=evidence)
    return rest_on_support(scene.physical_evidence, part, released, prefix=prefix)


def rest_on_support(physical_evidence: dict, part: str, released: list[float], *, prefix: str) -> dict:
    """Where a part let go at ``released`` comes to rest: on the recorded support under it, in one place's areas.

    ``projected_part_support`` (exact collision shapes, the recorded contact layer) must find the part resting on
    recorded support; its support collisions must belong to one place, and the part's xy must lie in one of that
    place's recorded areas. A robot's and the KMR's release read the same.

    Args:
        physical_evidence: The capture's runtime models, geometry and physics.
        part: The released part.
        released: Its pose at the release, ``[x, y, z, qx, qy, qz, qw]``.
        prefix: The step's name for the refusal text.

    Returns:
        ``{"release": ...}`` with the place, area and support; or a refusal stating where the part is.
    """
    from cais_spade_llm.resources.recovery_geometry import projected_part_support

    model_name = physical_evidence["snapshot"]["parts"][part]["model_name"]
    support = projected_part_support(part_name=part, state={"parts": {part: {
        "model_name": model_name, "current_pose": released}}}, evidence=physical_evidence, contact_layer_tolerance=True)
    if support.get("status") != "supported":
        return {"status": "INFEASIBLE", "reason": f"{prefix}: {part} released at {_xyz(released)} rests on no "
                                                  f"recorded support: {support.get('reason')}"}
    places = sorted({collision.split("::")[0] for collision in support["support_collisions"]})
    if len(places) != 1:
        return {"status": "INFEASIBLE", "reason": f"{prefix}: {part} released at {_xyz(released)} rests on "
                                                  f"{support['support_collisions']}, which belong to {places}"}
    areas = _place_areas(places[0], physical_evidence.get("geometry", {}).get("regions", {}))
    holding = sorted(name for name, bounds in areas.items() if _holds(bounds, released))
    if areas and not holding:
        return {"status": "INFEASIBLE", "reason": f"{prefix}: {part} rests on {places[0]} at {_xyz(released)}, outside "
                                                  f"its recorded areas {sorted(areas)}"}
    return {"release": {"part_name": part, "pose": released, "place": places[0], "area": holding[0] if holding else None,
                        "support_collisions": support["support_collisions"]}}


def _slot_release(targets: dict, scene: RobotEventScene, *, prefix: str, part: str, slot: Any, released: list[float],
                  evidence: dict) -> dict:
    """A release into an assembly slot: the slot the program states against the part's recorded one.

    The controller sets the part at the stated slot (``_snap_part_to_slot``): at ``part_origin_z`` with no turn,
    unless its own place calculation set a mating context for the part, when it keeps the part's orientation if the
    released part lies in its corridor.
    """
    from cais_spade_llm.recovery_framework.part_collision import mating_pose_valid
    from cais_spade_llm.resources.robot.recovery_capability import _destination_geometry

    declared = _destination_geometry(evidence, "assembly_board-v1", part) or {}
    origin = declared.get("target_origin_pose")
    if not isinstance(origin, dict):
        return {"status": "NEEDS_CONTEXT", "reason": f"{prefix}: inputs.geometry.assembly_board records no slot for {part}"}
    calculated = {"slot_x": origin["x"], "slot_y": origin["y"], "part_height": declared.get("part_height_m"),
                  "board_top_z": declared.get("slot_floor_z_m"), "part_origin_z": origin["z"],
                  "destination_location": "assembly_board-v1"}
    differing = [f"{field} {slot.get(field)!r} differs from {value!r}" for field, value in calculated.items()
                 if slot.get(field) != value]
    if differing:
        return {"status": "INFEASIBLE", "reason": f"{prefix}: release_part assembly_slot differs from {part}'s "
                                                  f"calculated slot: {'; '.join(differing)}"}
    contact = ((targets.get("compute_place_targets") or {}).get(part) or {}).get("simulation_mating_contact")
    orientation = [0., 0., 0., 1.]
    if isinstance(contact, dict):
        corridor = {**contact, "model_name": declared.get("model_name"), "target_origin_pose": origin,
                    "start_part_z": origin["z"]}
        if not mating_pose_valid(corridor, released):
            return {"status": "INFEASIBLE", "reason": f"{prefix}: {part} released at {_xyz(released)} lies outside "
                                                      "its mating corridor, and the controller then refuses the "
                                                      "release: Released mating part is not seated in its "
                                                      "configured pose"}
        orientation = released[3:]
    return {"release": {"part_name": part, "pose": [slot["slot_x"], slot["slot_y"], slot["part_origin_z"], *orientation],
                        "place": "assembly_board-v1", "released_pose": released,
                        "assembly_slot": {**calculated, "target": declared.get("target")}}}


def advance_robot_program(evidence: dict, targets: dict, *, task: dict, primitive: str, params: dict,
                          recovery_snapshot: dict, checked: dict) -> str | None:
    """Carry an accepted step into the robot's evidence, so the program's later steps and events start from it.

    A motion leaves the arm at the joints MoveIt's path ended on; a grasp attaches the part by the transform found at
    the actual tool pose, with the pick context of the event's own pick calculation; a release leaves the part where
    it rests and the gripper empty. Nothing here writes a value of the event.

    Args:
        evidence: The robot's evidence for this event, updated in place.
        targets: The event's calculated targets, per calculation and part.
        task: The accepted outline event.
        primitive: The step's primitive.
        params: The step's resolved parameters.
        recovery_snapshot: The robot's declared state before the step.
        checked: The checked step, with what ``check_robot_step`` found.

    Returns:
        None: every refusal is the step check's.
    """
    found = checked.get("evidence") or {}
    attachment = evidence.get("attachment") or {}
    if primitive in _MOTIONS:
        evidence.setdefault("snapshot", {})["joint_positions"] = deepcopy(found["composed_motion"]["end_joints"])
        if attachment.get("resting_on_support"):
            evidence["attachment"] = {key: value for key, value in attachment.items() if key != "resting_on_support"}
    elif primitive == "grasp_part":
        evidence["attachment"] = deepcopy(found["grasp"]["attachment"])
        pick = (targets.get("compute_pick_targets") or {}).get(evidence["attachment"]["part_name"])
        if pick is not None:
            evidence.setdefault("state", {})["_task_ctx"] = deepcopy(pick)
    elif primitive == "release_part":
        release = found["release"]
        pose = dict(zip(_POSE_AXES, release["pose"], strict=True))
        evidence["attachment"] = {}
        evidence.setdefault("observed_parts", {})[release["part_name"]] = {
            "current_pose": pose, "frame": "world", "part_location": release["place"],
            "source": {"kind": "projected", "contract_fingerprint": recovery_validation_fingerprint(release)}}
    return None


def _area_holds(physical_evidence: dict, area: str, origin: dict) -> bool:
    """Whether a recorded area's footprint holds a part's origin; its height bounds nothing that rests in it."""
    bounds = ((physical_evidence.get("geometry") or {}).get("regions", {}).get(area) or {}).get("bounds")
    return (isinstance(bounds, list) and len(bounds) >= 2 and all(isinstance(origin.get(axis), (int, float))
            for axis in ("x", "y")) and _holds(bounds, [origin["x"], origin["y"]]))


def _receiving_slot(part_name: str, destination: str, origin: dict, physical_evidence: dict) -> tuple[str, str] | None:
    """Find the resource and slot that take custody of a part placed on a supported place.

    The slot is the one whose custody puts the part at the destination (``product_state_from_custody``): the
    declared zone of the destination whose recorded area (``<destination>.zone_<n>``, as the RA names the zone's
    area) holds the part, or a resource's slot for a place it owns, such as a machine's workholding or its staging
    tray.

    Returns:
        The owning resource and its slot variable, or None when no single slot receives the part.
    """
    from cais_spade_llm.resources.environment_models import product_state_from_custody

    models = physical_evidence.get("environment_models", {})
    declared = models.get(destination, {}).get("assignments", {}).get("zones", [])
    zones = [zone for zone in declared if _area_holds(physical_evidence, f"{destination}.zone_{zone.get('zone')}", origin)]
    if declared:
        return (destination, f"zone_{zones[0]['zone']}_part") if len(zones) == 1 else None
    product = physical_evidence.get("product_name")
    slots = [(owner, field) for owner, model in models.items()
             for field, declaration in model.get("state_variables", {}).items()
             if declaration.get("reference") == "part_name" and product_state_from_custody(
                 part_name, owner, {"part_name": None, field: part_name}, product)["location"] == destination]
    return slots[0] if len(slots) == 1 else None


def _named_pose(jid: str, pose_name: str, physical_evidence: dict) -> dict | None:
    """Calculate the tool pose of a recorded named posture through the robot's recorded kinematic chain.

    ``move_to_named_pose`` leaves ``current_pose`` unknown by declaration. The recorded joints of the posture give
    where it ends, which the robot's next calculation starts from.
    """
    from cais_spade_llm.resources.continuous_motion import interval, kinematic_transforms, representative_pose

    rid = next((name for name, value in physical_evidence.get("jids", {}).items() if value == jid), None)
    model = physical_evidence.get("resource_models", {}).get(rid, {})
    joints = (model.get("named_joint_poses") or {}).get(pose_name)
    config = model.get("configuration") or {}
    if not isinstance(joints, dict) or not joints.get("joint_names") or "reference_link" not in config:
        return None
    transforms = kinematic_transforms(config, {name: interval(value) for name, value in
                                               zip(joints["joint_names"], joints["positions"], strict=True)})
    return dict(zip(("x", "y", "z", "qx", "qy", "qz", "qw"), representative_pose(transforms[config["reference_link"]]), strict=True))


def _load_onto_belt(part_name: str, destination: str, origin: dict, acting_jid: str, *, grounding_context: dict,
                    physical_evidence: dict, projected_resources: dict, stated: dict) -> dict:
    """Put a placed part on the belt loading position that holds it, as the belt's own order records it.

    The loading position is the one whose recorded area (``<destination>.<loading_position>``) holds the part. A
    belt takes a part only as the runtime does: idle and available, and not reserved for another resource. The
    belt's model sets the part's location and the order with the place, and the RA's place check finds them; only
    the ones the event states (``stated``, its ``neighbor_end_state``) are written, and the postcondition compares
    them.
    """
    from cais_spade_llm.resources.nominal_conveyor import conveyor_load_parameters

    model = physical_evidence["environment_models"][destination]
    rows = [row for row in model["assignments"]["loading_positions"].values()
            if _area_holds(physical_evidence, f"{destination}.{row.get('loading_position')}", origin)]
    if len(rows) != 1:
        return {"failure": f"No single recorded loading position of {destination} holds {part_name} at "
                           f"({origin.get('x')}, {origin.get('y')})", "status": "NEEDS_CONTEXT"}
    jid = physical_evidence.get("jids", {}).get(destination)
    acting = next((name for name, value in physical_evidence.get("jids", {}).items() if value == acting_jid), None)
    current = {**grounding_context.get("resources", {}).get(jid, {}), **projected_resources.get(jid, {})}
    values = deepcopy(physical_evidence.get("environment_valuation", {}).get(destination, {}))
    values.update({key: value for key, value in current.items() if key in values})
    if (current.get("availability", "available") != "available" or current.get("resource_state") != "idle"
            or values.get("loading_reserved_by") not in (None, acting)):
        return {"failure": f"{destination} cannot receive a part in its recorded state", "status": "INFEASIBLE"}
    try:
        order = conveyor_load_parameters(model, values, part_name, rows[0]["loading_position"])
    except ValueError as exc:
        return {"failure": str(exc), "status": "INFEASIBLE"}
    changed = {field.removeprefix("next_"): value for field, value in order.items()}
    changed[f"part_location.{part_name}"] = rows[0]["loading_position"]
    values.update({field: value for field, value in changed.items() if field in (stated.get(jid) or {})})
    projected_resources[jid] = values
    return {"location": destination, "owner": destination, "values": values, "model": model}


def receive_placed_part(part_name: str, destination: str, origin: dict, *, grounding_context: dict,
                        physical_evidence: dict, projected_resources: dict, acting_jid: str,
                        stated: dict) -> dict:
    """Put a placed part into the slot that receives it, in ``projected_resources``.

    Code writes no value the event leaves out: the receiving resource takes only the values the event states
    for it. The RA refuses a place that leaves the receiving slot or a belt's order out, and the postcondition
    compares the stated values with these. What the receiving resource's model declares the part leaves there
    (``received_values``, such as a machine loaded) is taken for the values the event states too.

    Args:
        part_name: The placed part.
        destination: The place the event states for the part.
        origin: Where the place target sets the part down.
        grounding_context: The grounded state, per resource.
        physical_evidence: The capture's runtime models and valuation.
        projected_resources: The values the event leaves, per resource, updated here.
        acting_jid: The resource that places the part.
        stated: The event's ``neighbor_end_state``.

    Returns:
        The part's ``location``, the receiving ``owner``, its ``values`` and ``model``; or ``failure`` with its
        ``status`` when no single free slot takes the part.
    """
    from cais_spade_llm.resources.environment_models import product_state_from_custody, received_values

    if physical_evidence.get("environment_models", {}).get(destination, {}).get("assignments", {}).get("loading_positions"):
        return _load_onto_belt(part_name, destination, origin, acting_jid, grounding_context=grounding_context,
                               physical_evidence=physical_evidence, projected_resources=projected_resources,
                               stated=stated)
    receiving = _receiving_slot(part_name, destination, origin, physical_evidence)
    if receiving is None:
        return {"failure": f"No single declared slot of {destination} receives {part_name} at "
                           f"({origin.get('x')}, {origin.get('y')})", "status": "NEEDS_CONTEXT"}
    owner, field = receiving
    model = physical_evidence.get("environment_models", {}).get(owner, {})
    declarations = model.get("state_variables", {})
    jid = physical_evidence.get("jids", {}).get(owner)
    values = deepcopy(physical_evidence.get("environment_valuation", {}).get(owner, {}))
    values.update({key: value for key, value in grounding_context.get("resources", {}).get(jid, {}).items() if key in declarations})
    values.update(projected_resources.get(jid, {}))
    if declarations.get(field, {}).get("reference") != "part_name" or field not in values:
        return {"failure": f"{owner} records no occupancy of {field}", "status": "NEEDS_CONTEXT"}
    if values[field] not in (None, part_name):
        return {"failure": f"{owner}'s {field} holds {values[field]}", "status": "INFEASIBLE"}
    received, _ = received_values(physical_evidence.get("environment_models", {}), owner, destination, field,
                                  part_name, values, taken=False)
    values[field] = part_name
    named = set(stated.get(jid) or {})
    values.update({name: value for name, value in received.items() if name in named})
    if named & {field, *received}:
        projected_resources[jid] = values
    product = product_state_from_custody(part_name, owner, values, physical_evidence.get("product_name"))
    return {"location": product["location"], "owner": owner, "values": values, "model": model}


def project_composition_completion(*, task: dict, trace: list, recovery_des_model: dict,
                                   recovery_snapshot: dict, projected_snapshot: dict,
                                   part_context: dict, grounding_context: dict,
                                   primitive_catalog: list, resource_evidence: dict,
                                   physical_evidence: dict, locate_pose: Callable[[dict], str | None] | None = None) -> dict:
    """Project a checked robot program: custody, where its part rests and where the robot ends.

    Each step's physical facts were found at the step (``check_robot_step``): the MoveIt path of a motion and the
    tool pose it ends at, the grasp at the actual tool pose, and the support or assembly slot a release leaves the
    part in. Completion applies the transfers' declared effects to them. The robot's location is the last place it
    reached: a transfer's declared effect sets it, a motion that ends with the gripper in a recorded area
    (``locate_pose``) or at a named pose with a declared place sets it, and a motion ending in no recorded area leaves
    it. Composition, like the outline, writes no value the event leaves out (AGENTS.md, "No hidden rules in recovery
    events"): it refuses a program that ends where the event does not say.
    """
    from cais_spade_llm.resources.robot.robot_primitives import robot_primitive_sequence_validator

    resource, part = deepcopy(projected_snapshot), deepcopy(part_context)
    evidence = deepcopy(resource_evidence)
    findings = robot_primitive_sequence_validator(outline_event=task,
        primitive_steps=[{"primitive": row["primitive"], "params": row["params"]} for row in trace],
        trace_metadata={"step_results": trace}, start_snapshot=recovery_snapshot,
        projected_snapshot=resource, grounding_context=grounding_context, primitive_catalog=primitive_catalog)
    if findings:
        return {"feasibility_status": "INFEASIBLE", "reason": "; ".join(row["reason"] for row in findings), "findings": findings}
    completed, projected_resources = [], {}
    events = {event["event_name"]: event for event in recovery_des_model.get("events", [])}
    stated = task.get("expected_end_state") or {}

    def fail(reason: str, status: str = "NEEDS_CONTEXT") -> dict:
        return {"feasibility_status": status, "reason": reason}

    def apply_updates(event: dict, bindings: dict, fields: set[str]) -> bool:
        from cais_spade_llm.resources import recovery_transition_derivation as contracts

        declarations = recovery_des_model["state_variables"]
        state = {"resource": {}, "part": {}, "unknown": {"resource": [], "part": []}}
        updates = {key: value for key, value in event.get("updates", {}).items() if key in fields}
        if set(updates) != fields:
            return False
        derived, error = contracts._apply(updates, state, bindings, declarations)
        if error:
            return False
        new_resource, new_part = contracts._runtime_views(derived, resource, part)
        resource.update(new_resource)
        part.update(new_part)
        completed.append({"event_name": event["event_name"], "effects": updates,
                          "bindings": deepcopy(bindings), "contract_fingerprint": recovery_validation_fingerprint(event)})
        return True

    places = (recovery_des_model["state_variables"].get("resource_location") or {}).get("domain") or []
    transfer_places = {}

    def located_by(event: dict, index: int, binding: str, place: Any) -> str | None:
        """Record where a transfer's declared resource_location effect leaves the robot; why it cannot, or None.

        A place the location does not declare says nothing about the robot, and a part at no place leaves the robot
        at none.
        """
        from cais_spade_llm.resources import recovery_transition_derivation as contracts

        effect = (event.get("updates") or {}).get("resource_location")
        if effect is None or (place is not None and place not in places):
            return None
        if place is None:
            transfer_places[index] = (event.get("event_name"), None)
            return None
        state = {"resource": {}, "part": {}, "unknown": {"resource": [], "part": []}}
        derived, error = contracts._apply({"resource_location": effect}, state, {binding: place},
                                          recovery_des_model["state_variables"])
        if error:
            return f"{event.get('event_name')}'s declared resource_location effect does not take {place}: {error}"
        transfer_places[index] = (event.get("event_name"), derived["resource"].get("resource_location"))
        return None

    transfers = [row for row in trace if row["primitive"] in {"grasp_part", "release_part"}]
    # One transfer, or the event's part taken and then put down, which Algorithm 1 checks as a grasp and a release.
    if [row["primitive"] for row in transfers] not in ([], ["grasp_part"], ["release_part"], ["grasp_part", "release_part"]):
        return fail("Only one custody transfer, or a grasp followed by a release, is supported in one event")
    for transfer in transfers:
        part_name = transfer["resolved_params"].get("part_name", task.get("part_name"))
        if part_name != task.get("part_name"):
            return fail(f"{transfer['primitive']} step {transfer['step_index']} names {part_name}, and the event's "
                        f"part is {task.get('part_name')}", "INFEASIBLE")
        if transfer["primitive"] == "grasp_part":
            event = events.get("pick_grasp", {})
            # Since the aligned part model (October 23) the part's location is its only state field.
            if not apply_updates(event, {"part_name": part_name}, {"part_location"}):
                return fail("The owner's pick_grasp declares no part_location effect")
            location = part_context.get("part_location", part_context.get("location"))
            refusal = located_by(event, transfer["step_index"], "origin_resource_location", location)
            if refusal:
                return fail(refusal)
            for jid, field, empty in _source_records(part_name, task["resource_jid"], grounding_context, physical_evidence,
                                                     task.get("neighbor_end_state") or {}, origin=location):
                projected_resources.setdefault(jid, {})[field] = empty
            part["part_holder_resource_jid"] = task["resource_jid"]
            continue
        event = events.get("place_insert", {})
        release = ((transfer.get("capability_evidence") or {}).get("evidence") or {}).get("release")
        if not isinstance(release, dict):
            return fail(f"release_part step {transfer['step_index']} has no checked resting place")
        destination = release["place"]
        part["part_holder_resource_jid"] = None
        if "assembly_slot" in release:
            if not apply_updates(event, {"destination_location": destination}, {"part_location"}):
                return fail("The owner's place_insert declares no part_location effect")
            # release_part declares no record entry and a robot's recovery model no record, so a stated record
            # takes the entries the runtime model declares for this place and the postcondition compares them.
            # Composition adds no entry the event left out.
            if "processCompleted" in stated:
                from cais_spade_llm.resources.environment_models import place_completion

                rid = next((name for name, value in physical_evidence.get("jids", {}).items()
                            if value == task["resource_jid"]), None)
                entries = place_completion(physical_evidence.get("environment_models", {}).get(rid) or {},
                                           destination, release["assembly_slot"].get("target"))
                record = list(part.get("processCompleted") or [])
                part["processCompleted"] = record + [entry for entry in entries if entry not in record]
        else:
            placed = receive_placed_part(part_name, destination, dict(zip(_POSE_AXES, release["pose"], strict=True)),
                                         grounding_context=grounding_context, physical_evidence=physical_evidence,
                                         projected_resources=projected_resources, acting_jid=task["resource_jid"],
                                         stated=task.get("neighbor_end_state") or {})
            if "failure" in placed:
                return fail(placed["failure"], placed["status"])
            part.update(part_location=placed["location"])
            completed.append({"source": "product_state_from_custody", "owner": placed["owner"],
                "resource_values": placed["values"], "contract_fingerprint": recovery_validation_fingerprint(placed["model"])})
        refusal = located_by(event, transfer["step_index"], "destination_location", destination)
        if refusal:
            return fail(refusal)
        rest = dict(zip(_POSE_AXES, release["pose"], strict=True))
        part["observed_pose"] = deepcopy(rest)
        part["observed_pose_current"] = False
        part["pose_source"] = "validated_composition_projection"
        evidence.setdefault("observed_parts", {})[part_name] = {"current_pose": deepcopy(rest), "frame": "world",
            "part_location": part.get("part_location"),
            "source": {"kind": "projected", "contract_fingerprint": recovery_validation_fingerprint(event)}}
        evidence["attachment"] = {}
    start = recovery_snapshot.get("resource_location", recovery_snapshot.get("current_location"))
    try:
        location, source = _last_place(trace, start, transfer_places, recovery_des_model,
                                       stated.get("resource_location"), locate_pose)
    except LookupError as exc:
        return fail(str(exc))
    if "resource_location" in stated:
        # The statement takes the last place the robot reached, and the postcondition compares them.
        write_location(resource, part, location)
    elif location != start:
        # The RA cannot see the program, so a move the event does not state is refused here.
        return fail(f"The program ends at {location or 'no named place'} and the event states no resource_location",
                    "INFEASIBLE")
    if location != start or "resource_location" in stated:
        completed.append({"resource_location": location, **source})
    last_motion = next((row for row in reversed(trace) if row["primitive"] in (*_MOTIONS, *_UNMODELLED_MOTIONS)), None)
    if last_motion is not None and last_motion["primitive"] == "move_to_named_pose":
        # move_to_named_pose leaves current_pose unknown by declaration; the named posture's recorded joints give
        # where it ends, which the robot's next calculation starts from.
        pose = _named_pose(task["resource_jid"], last_motion["resolved_params"].get("pose_name"), physical_evidence)
        if pose is not None:
            resource["current_pose"] = pose
    for field, aliases in {"part_state": ("state", "current_state"),
                           "part_location": ("location", "current_location"),
                           "part_holder_resource_jid": ("current_holder_resource_jid", "holder_resource_jid")}.items():
        if field in part:
            for alias in aliases:
                part[alias] = deepcopy(part[field])
    carried = resource_evidence.get("snapshot") or {}
    evidence["snapshot"] = deepcopy(resource)
    # The arm's joints are physical evidence, never a declared field: the robot's next event starts from them.
    for field in ("joint_names", "joint_positions"):
        if field in carried:
            evidence["snapshot"][field] = deepcopy(carried[field])
    evidence.setdefault("state", {})["_held_part"] = resource.get("held_part")
    evidence["state"]["_position"] = deepcopy(resource.get("current_pose"))
    return {"feasibility_status": "FEASIBLE", "projected_snapshot": resource,
            "projected_part_context": part, "completed_contracts": completed,
            "resource_evidence": evidence, "projected_resources": projected_resources,
            "scope": "Declared primitive effects, the transfers' declared effects at their physically checked steps, "
                     "and the last place the robot reached"}


def write_location(resource: dict, part: dict, value: Any) -> None:
    """Write a resource's location as a declared effect writes it: its aliases, its mirrors and its occupancy record.

    Args:
        resource: The projected resource snapshot, updated in place.
        part: The projected part context the derivation reads alongside it.
        value: The place, or None for none.
    """
    from cais_spade_llm.resources import recovery_transition_derivation as contracts

    written = {"resource_location": deepcopy(value)}
    if isinstance(resource.get("occupancy"), dict):
        written["occupancy.location"] = deepcopy(value)
    state = {"resource": written, "part": {}, "unknown": {"resource": [], "part": []}}
    resource.update(contracts._runtime_views(state, resource, part)[0])


def _named_pose_place(recovery_des_model: dict, pose_name: Any) -> tuple[dict, Any] | None:
    """The place a named pose's declared one-step contract (``move_home``) sets, with that contract."""
    for event in recovery_des_model.get("events", []):
        steps = event.get("composition_contract", {}).get("steps") or []
        effect = event.get("updates", {}).get("resource_location") or {}
        if (len(steps) == 1 and steps[0].get("op") == "move_to_named_pose" and not steps[0].get("when")
                and steps[0].get("params", {}).get("pose_name") == pose_name and set(effect) == {"set"}):
            return event, effect["set"]
    return None


def _last_place(trace: list, start: Any, transfer_places: dict, recovery_des_model: dict, stated: Any,
                locate_pose: Callable[..., str | None] | None) -> tuple[Any, dict]:
    """The last place a robot's program reached, and what set it.

    A transfer sets the place its declared effect names; a named pose the place its declared contract sets; any
    other motion the place whose recorded area holds the gripper where the motion ended, if one does. A motion that
    ends in no recorded area leaves the place as it was.

    Returns:
        The place and its source.

    Raises:
        LookupError: A motion has no checked end pose to locate.
    """
    location, source = start, {"source": "start"}
    for row in trace:
        index, primitive = row["step_index"], row["primitive"]
        if index in transfer_places:
            event_name, location = transfer_places[index]
            source = {"source": f"{event_name} effect", "step_index": index}
        elif primitive == "move_to_named_pose":
            named = _named_pose_place(recovery_des_model, row["resolved_params"].get("pose_name"))
            if named is not None:
                location, source = named[1], {"source": f"{named[0]['event_name']} effect", "step_index": index}
        elif primitive in _MOTIONS and callable(locate_pose):
            end = ((row.get("capability_evidence") or {}).get("evidence") or {}).get("composed_motion", {}).get("end_pose")
            if not isinstance(end, list) or len(end) != 7:
                raise LookupError(f"{primitive} step {index} has no checked end pose")
            reached = locate_pose(dict(zip(_POSE_AXES, end, strict=True)), stated)
            if reached is not None:
                location, source = reached, {"source": "recorded area holding the gripper", "step_index": index}
    return location, source
