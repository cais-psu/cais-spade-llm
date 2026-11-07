from __future__ import annotations

"""KMR-owned step checks, step tracking and completion evidence for authored recovery programs."""

import math
from copy import deepcopy
from typing import Any

from cais_spade_llm.agents.shared_information.recovery_validation_protocol import (
    recovery_validation_fingerprint,
)
from cais_spade_llm.recovery_framework.geometry import compose
from cais_spade_llm.recovery_framework.kmr_recovery import (
    CARRYING_TCP_TOLERANCE_M,
    GRASP_INITIAL_TOLERANCE_M,
    GRASP_REACH_M,
    kmr_place_at,
    transport_pose,
)

_POSE_AXES = ("x", "y", "z", "qx", "qy", "qz", "qw")
# The KMR arm motions the composition's MoveIt check flies as the kmr_gazebo worker does.
_ARM_MOTIONS = ("move_cartesian", "move_relative", "move_to_named_pose")
# Joint motions the MoveIt check has no route for; the worker refuses them under cartesian_motion_only.
_JOINT_MOTIONS = ("move_joints", "rotate_joint")
_FLOOR_SOURCE = ("the floor map is the map script's (ros2/cais_lab_robotics/scripts/generate_recovery_framework_map.py), "
                 "not a recorded fact")


def _inverse(pose: list[float]) -> list[float]:
    from cais_spade_llm.recovery_framework.kmr_gazebo import inverse

    return inverse(pose)


def _projected_source(payload: object) -> dict:
    return {"kind": "projected", "contract_fingerprint": recovery_validation_fingerprint(payload)}


def _complete_pose(value: Any) -> list[float] | None:
    """A pose as ``[x, y, z, qx, qy, qz, qw]`` when it is a complete one, recorded as a list or by named fields."""
    if isinstance(value, dict):
        value = [value.get(axis) for axis in _POSE_AXES]
    if (not isinstance(value, list) or len(value) != 7
            or any(type(item) not in (int, float) or not math.isfinite(item) for item in value)):
        return None
    return [float(item) for item in value]


def _xyz(values: list[float]) -> str:
    return "(" + ", ".join(f"{float(value):.4f}" for value in values[:3]) + ")"


def _yaw(pose: list[float]) -> float:
    return math.atan2(2 * (pose[6] * pose[5] + pose[3] * pose[4]), 1 - 2 * (pose[4] ** 2 + pose[5] ** 2))


def _kmr_joints(evidence: dict, scene: Any) -> list[float]:
    """The arm joints the KMR's program has reached: carried in its evidence, else the captured ones."""
    joints = (evidence.get("snapshot") or {}).get("joint_positions")
    if not isinstance(joints, list):
        joints = scene.physical_evidence["snapshot"]["resources"][scene.resource_id]["joint_positions"]
    return [float(value) for value in joints]


def _tool_pose(evidence: dict, scene: Any, joints: list[float]) -> list[float]:
    """The KMR's TCP (its recorded chain's reference link) at the joints, on the base its evidence records."""
    from scipy.spatial.transform import Rotation

    from cais_spade_llm.resources.kinematic_reach import chain_frames

    configuration = scene.physical_evidence["resource_models"][scene.resource_id]["configuration"]
    base = _complete_pose((evidence.get("snapshot") or {}).get("base_pose"))
    if base is not None:
        configuration = {**configuration, "root_xyz": base[:3], "root_quaternion": base[3:]}
    frame = chain_frames(configuration, dict(zip(configuration["joint_names"], joints, strict=True)))[
        configuration["reference_link"]]
    return [*map(float, frame[:3, 3]), *map(float, Rotation.from_matrix(frame[:3, :3]).as_quat())]


def check_kmr_step(evidence: dict, scene: Any, floor: dict, *, index: int, primitive: str, params: dict,
                   recovery_snapshot: dict, checked: dict) -> dict:
    """Check one step of a KMR program physically, in the cell its earlier steps and the accepted prefix leave.

    An arm motion is checked by MoveIt on the capture's recorded model and scene with the worker's own route and
    limits (``composed_motion_check``, a check of one IK path); a grasp by the worker's grasp precondition at the
    actual TCP; a release by where the part comes to rest; a base move by the base controller's own preconditions and
    whether a drive route exists with the arm and the part it holds. The owner's capability result comes back with
    what the check found, which ``advance_kmr_program`` carries, or as a refusal stating the fact.

    Args:
        evidence: The KMR's evidence as its earlier steps leave it.
        scene: The event's fixed inputs (``RobotEventScene``).
        floor: The navigation floor map (``floor_extent_xy`` and ``floor_obstacles_xy``) from the map script.
        index: The step's index in the program.
        primitive: The step's primitive.
        params: The step's resolved parameters.
        recovery_snapshot: The KMR's declared state at the step's start.
        checked: The owner's capability result for the step.

    Returns:
        ``checked`` with what the check found added to its ``evidence``, or a refusal.
    """
    from cais_spade_llm.resources.robot.recovery_composition import _motion_check

    written = (scene.program[index] or {}).get("primitive") if index < len(scene.program) else None
    if written != primitive:
        raise RuntimeError(f"Step {index} of the program is {written!r}, not the checked {primitive!r}")
    if primitive in _ARM_MOTIONS:
        found = _motion_check(evidence, scene, index=index, primitive=primitive, params=params,
                              recovery_snapshot=recovery_snapshot)
    elif primitive in _JOINT_MOTIONS:
        found = {"status": "NEEDS_CONTEXT", "reason": f"{primitive} step {index} ({scene.resource_id}): the "
                                                      f"composition's MoveIt check does not model {primitive}"}
    elif primitive == "grasp_part":
        found = _grasp_check(evidence, scene, index=index, params=params)
    elif primitive == "release_part":
        found = _release_check(evidence, scene, index=index, recovery_snapshot=recovery_snapshot)
    elif primitive == "move_base":
        found = _drive_check(evidence, scene, floor, index=index, params=params, recovery_snapshot=recovery_snapshot)
    else:
        return checked
    if "status" in found:
        return {"allowed": False, "feasibility_status": found["status"], "reason": found["reason"],
                "evidence": {**(checked.get("evidence") or {}), **(found.get("evidence") or {})}}
    return {**checked, "evidence": {**(checked.get("evidence") or {}), **found}}


def _grasp_check(evidence: dict, scene: Any, *, index: int, params: dict) -> dict:
    """The worker's own grasp precondition at the actual TCP, and the attachment the grasp makes there."""
    part = params.get("part_name") or scene.task.get("part_name")
    prefix = f"grasp_part step {index} ({scene.resource_id})"
    observed = (evidence.get("observed_parts") or {}).get(part) or {}
    pose = _complete_pose(observed.get("observed_pose", observed.get("current_pose")))
    if pose is None:
        return {"status": "NEEDS_CONTEXT", "reason": f"{prefix}: the grasp check of {part} needs its recorded pose "
                                                     "(observed_parts)"}
    initial = params.get("initial")
    if initial is not None:
        if len(initial) < 3:
            return {"status": "INFEASIBLE", "reason": f"{prefix}: initial has {len(initial)} values, and the KMR "
                                                      "worker compares its x, y and z with the observed part"}
        moved = math.dist([float(value) for value in initial[:3]], pose[:3])
        if moved > GRASP_INITIAL_TOLERANCE_M:
            return {"status": "INFEASIBLE", "reason": f"{prefix}: Part moved before grasp: {part} is observed at "
                                                      f"{_xyz(pose)}, {moved:.4f} m from the given initial pose; the "
                                                      f"KMR worker allows {GRASP_INITIAL_TOLERANCE_M} m (its code "
                                                      "constant, not a recorded fact)"}
    tool = _tool_pose(evidence, scene, _kmr_joints(evidence, scene))
    distance = math.dist(tool[:3], pose[:3])
    if distance > GRASP_REACH_M:
        return {"status": "INFEASIBLE", "reason": f"{prefix}: KMR grasp requires the part at the gripper: the TCP at "
                                                  f"{_xyz(tool)} is {distance:.4f} m from {part}'s observed origin "
                                                  f"{_xyz(pose)}; the KMR worker allows {GRASP_REACH_M} m (its code "
                                                  "constant, not a recorded fact)"}
    grasp = {"tool_pose": tool, "part_pose": pose, "tcp_to_part_m": distance}
    attachment = {"attached": True, "part_name": part, "held_part": part, "grasp_transform": compose(_inverse(tool), pose),
                  "resting_on_support": True, "source": _projected_source(grasp)}
    return {"grasp": {**grasp, "attachment": attachment}}


def _release_check(evidence: dict, scene: Any, *, index: int, recovery_snapshot: dict) -> dict:
    """Where the released part comes to rest: on the recorded support under it, in one place's recorded areas."""
    from cais_spade_llm.resources.robot.recovery_composition import rest_on_support

    part = recovery_snapshot.get("held_part")
    prefix = f"release_part step {index} ({scene.resource_id})"
    attachment = evidence.get("attachment") or {}
    holds = attachment.get("attached") is True and part in (attachment.get("part_name"), attachment.get("held_part"))
    grasp = _complete_pose(attachment.get("grasp_transform")) if holds else None
    if grasp is None:
        return {"status": "NEEDS_CONTEXT", "reason": f"{prefix}: no recorded grasp says how the KMR holds {part}"}
    released = compose(_tool_pose(evidence, scene, _kmr_joints(evidence, scene)), grasp)
    return rest_on_support(scene.physical_evidence, part, released, prefix=prefix)


def _carrying_refusal(evidence: dict, scene: Any, prefix: str, joints: list[float], base: list[float]) -> str | None:
    """The worker's confirm_carrying before a loaded drive: the TCP at the transport pose of the base."""
    tool = _tool_pose(evidence, scene, joints)
    try:
        transport = transport_pose(evidence["inputs"]["scene"], base, tool)
    except ValueError as exc:
        return f"{prefix}: {exc}, which the worker's confirm_carrying needs before a loaded drive"
    distance = math.dist(tool[:3], transport[:3])
    if distance <= CARRYING_TCP_TOLERANCE_M:
        return None
    return (f"{prefix}: KMR carrying TCP has not reached the configured transport posture: the TCP at {_xyz(tool)} is "
            f"{distance:.4f} m from the transport pose {_xyz(transport)}; the worker's confirm_carrying allows "
            f"{CARRYING_TCP_TOLERANCE_M} m (its code constant, not a recorded fact)")


def _drive_check(evidence: dict, scene: Any, floor: dict, *, index: int, params: dict,
                 recovery_snapshot: dict) -> dict:
    """Whether the KMR can drive as the step states: the base controller's own preconditions, then a route.

    The base controller takes an empty KMR only with its gripper open; the worker takes a loaded one only after its
    confirm_carrying. The route is an existence check (Eq. 12), not the controller's own route, which it chooses at
    run time: the RA's drive search (``kmr_recovery.drive_route_refusal``) with the arm at the joints the program has
    reached and the part it holds, leg by leg through the stated waypoints; every waypoint and the target must also
    pass the controller's footprint test on the floor map, which comes from the map script, not from 𝒦.
    """
    from cais_spade_llm.recovery_framework.kmr_recovery import drive_route_refusal
    from cais_spade_llm.resources.kinematic_reach import BASE_PADDING_M, footprint_obstacles

    prefix = f"move_base step {index} ({scene.resource_id})"
    snapshot = evidence.get("snapshot") or {}
    base = _complete_pose(snapshot.get("base_pose"))
    if base is None:
        return {"status": "NEEDS_CONTEXT", "reason": f"{prefix}: the drive check needs the KMR's recorded base_pose"}
    joints, held = _kmr_joints(evidence, scene), recovery_snapshot.get("held_part")
    carried = None
    if held is None:
        if recovery_snapshot.get("gripper_state") != "open":
            return {"status": "INFEASIBLE", "reason": f"{prefix}: the base controller drives an empty KMR only with its "
                                                      "gripper open, and its gripper_state is "
                                                      f"{recovery_snapshot.get('gripper_state')!r}"}
    else:
        refusal = _carrying_refusal(evidence, scene, prefix, joints, base)
        if refusal:
            return {"status": "INFEASIBLE", "reason": refusal}
        attachment = evidence.get("attachment") or {}
        grasp = _complete_pose(attachment.get("grasp_transform"))
        bounds = scene.physical_evidence.get("geometry", {}).get("parts", {}).get(held, {}).get("local_bounds")
        if grasp is None or not bounds:
            return {"status": "NEEDS_CONTEXT", "reason": f"{prefix}: the drive check of the held {held} needs its "
                                                         "recorded grasp_transform and local_bounds"}
        carried = {"part_name": held, "grasp_transform": grasp, "local_bounds": bounds}
    if not floor.get("floor_extent_xy") or not isinstance(floor.get("floor_obstacles_xy"), dict):
        return {"status": "NEEDS_CONTEXT", "reason": f"{prefix}: the drive check needs the navigation floor map, which "
                                                     "this composition request does not carry"}
    kmr_scene = evidence["inputs"]["scene"]
    route = [[base[0], base[1], _yaw(base)], *([float(value) for value in pose[:3]] for pose in params.get("waypoints") or []),
             [float(value) for value in params["target_pose"][:3]]]
    footprint = kmr_scene["KMR"]["platform_overall_dimensions_m"]
    for pose in route[1:]:
        overlaps = footprint_obstacles(floor, footprint, pose)
        if overlaps:
            return {"status": "INFEASIBLE", "reason": f"{prefix}: the KMR's footprint with its {BASE_PADDING_M} m margin "
                                                      f"at ({pose[0]:.3f}, {pose[1]:.3f}, yaw {pose[2]:.3f}) overlaps "
                                                      f"{', '.join(overlaps)} on the floor map, and the base controller "
                                                      f"refuses such a goal; {_FLOOR_SOURCE}"}
    config = scene.physical_evidence["resource_models"][scene.resource_id]["configuration"]
    bodies = scene.physical_evidence.get("physics_evidence", {}).get("models", {})
    for start, end in zip(route[:-1], route[1:], strict=True):
        refusal = drive_route_refusal(kmr_scene, config, floor, bodies, start, end, joints=joints, held=carried)
        if refusal:
            return {"status": "INFEASIBLE", "reason": f"{prefix}: {refusal}; {_FLOOR_SOURCE}"}
    return {"drive": {"route": route, "arm_joints": joints, "held_part": held,
                      "statement": "a drive route exists with the arm at its composed joints"
                                   + (f" holding {held}" if held else "") + ", leg by leg through the stated poses, "
                                   "the footprint on free floor and the arm clear of the recorded bodies (an existence "
                                   "check: the base controller chooses its own route at run time); " + _FLOOR_SOURCE}}


def advance_kmr_program(evidence: dict, targets: dict, *, task: dict, primitive: str, params: dict,
                        recovery_snapshot: dict, checked: dict) -> str | None:
    """Carry an accepted step into the KMR's evidence, so the program's later steps and events start from it.

    A calculation keeps its targets for the program's references; a base move carries the arm with the base; an arm
    motion leaves the arm at the joints MoveIt's path ended on and the TCP there; a grasp attaches the part by the
    transform found at the actual TCP; a release leaves the part where it rests and the gripper empty. Nothing here
    writes a value of the event, and nothing here refuses: every refusal is the step check's.

    Args:
        evidence: The owner's evidence for this event, updated in place.
        targets: The owner's calculated pick and place targets for this event, updated in place.
        task: The accepted outline event.
        primitive: The step's primitive.
        params: The step's resolved parameters.
        recovery_snapshot: The resource state before the step.
        checked: The checked step, with what ``check_kmr_step`` found.

    Returns:
        None.
    """
    state = evidence.setdefault("snapshot", {})
    facts = checked.get("evidence") or {}
    attachment = evidence.get("attachment") or {}
    if primitive in {"compute_pick_targets", "compute_place_targets"}:
        targets[primitive] = deepcopy(facts["calculated_targets"])
    elif primitive == "move_base":
        before = recovery_snapshot["base_pose"]
        x, y, yaw = params["target_pose"][:3]
        after = [x, y, before[2], 0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)]
        if "current_pose" in state:
            state["current_pose"] = compose(after, compose(_inverse(before), state["current_pose"]))
        state["base_pose"] = after
    elif primitive in _ARM_MOTIONS:
        motion = facts["composed_motion"]
        state["joint_positions"] = deepcopy(motion["end_joints"])
        state["current_pose"] = deepcopy(motion["end_pose"])
        # The part a grasp took leaves its support with the arm's first motion after the grasp.
        if attachment.get("resting_on_support"):
            evidence["attachment"] = {key: value for key, value in attachment.items() if key != "resting_on_support"}
    elif primitive == "grasp_part":
        evidence["attachment"] = deepcopy(facts["grasp"]["attachment"])
    elif primitive == "release_part":
        release = facts["release"]
        evidence["attachment"] = {"attached": False, "source": _projected_source(release)}
        evidence.setdefault("observed_parts", {})[release["part_name"]] = {
            "current_pose": deepcopy(release["pose"]), "frame": "world", "part_location": release["place"],
            "source": _projected_source(release)}
    return None


def preview_kmr_output(targets: dict, *, task: dict, primitive: str, **_unused) -> dict:
    """Publish the owner's calculated targets as ``event_facts``, under the names its decompositions use."""
    if primitive not in targets:
        return {}
    key = "pick_targets" if primitive == "compute_pick_targets" else "place_targets"
    return {"event_fact_key": f"{key}.{task['part_name']}", "output": deepcopy(targets[primitive]), "status": "calculated"}


def project_kmr_completion(*, task: dict, trace: list, recovery_snapshot: dict, projected_snapshot: dict,
                           part_context: dict, grounding_context: dict, resource_evidence: dict,
                           physical_evidence: dict, **_unused) -> dict:
    """Project a checked KMR program: where it left the base, the arm and the event's part.

    Each step's physical facts were found at the step (``check_kmr_step``): the MoveIt path of an arm motion, the
    grasp at the actual TCP, the support a release leaves the part on and the drive. Code writes no value the event
    left out. The KMR is at the place whose declared dock its base ends on, or at none: a location the event states
    takes it, for the postcondition to compare, and a base move the event does not state is refused. The custody of
    the part follows the transfer the step checked, and a released part rests where its check found it.
    """
    from cais_spade_llm.resources.robot.recovery_composition import (
        _source_records,
        receive_placed_part,
        write_location,
    )

    def fail(reason: str, status: str = "NEEDS_CONTEXT") -> dict:
        return {"feasibility_status": status, "reason": reason}

    resource, part = deepcopy(projected_snapshot), deepcopy(part_context)
    evidence = deepcopy(resource_evidence)
    stated = task.get("expected_end_state") or {}
    jid, projected_resources = task["resource_jid"], {}
    transfers = [row for row in trace if row["primitive"] in {"grasp_part", "release_part"}]
    if [row["primitive"] for row in transfers] not in ([], ["grasp_part"], ["release_part"], ["grasp_part", "release_part"]):
        return fail("Only one custody transfer, or a grasp followed by a release, is supported in one event")
    for transfer in transfers:
        index = transfer["step_index"]
        if transfer["primitive"] == "grasp_part":
            part_name = transfer["resolved_params"].get("part_name") or task.get("part_name")
            if part_name != task.get("part_name"):
                return fail(f"grasp_part step {index} names {part_name}, and the event's part is {task.get('part_name')}",
                            "INFEASIBLE")
            # The place the part leaves, whose owner's model declares what taking it leaves there, as for a robot.
            origin = part_context.get("part_location", part_context.get("location"))
            part.update(part_location=jid, part_holder_resource_jid=jid)
            for owner, field, empty in _source_records(part_name, jid, grounding_context, physical_evidence,
                                                       task.get("neighbor_end_state") or {}, origin=origin):
                projected_resources.setdefault(owner, {})[field] = empty
            continue
        release = ((transfer.get("capability_evidence") or {}).get("evidence") or {}).get("release")
        if not isinstance(release, dict):
            return fail(f"release_part step {index} has no checked resting place")
        part_name = release["part_name"]
        if part_name != task.get("part_name"):
            return fail(f"release_part step {index} puts down {part_name}, and the event's part is "
                        f"{task.get('part_name')}", "INFEASIBLE")
        if not isinstance(stated.get("part_location"), str):
            return fail("A KMR release needs the place the event states for its part")
        rest = deepcopy(release["pose"])
        placed = receive_placed_part(part_name, release["place"], dict(zip(("x", "y", "z"), rest, strict=False)),
                                     grounding_context=grounding_context, physical_evidence=physical_evidence,
                                     projected_resources=projected_resources, acting_jid=jid,
                                     stated=task.get("neighbor_end_state") or {})
        if "failure" in placed:
            return fail(placed["failure"], placed["status"])
        part.update(part_location=placed["location"], part_holder_resource_jid=None,
                    observed_pose=dict(zip(_POSE_AXES, rest, strict=True)), observed_pose_current=False,
                    pose_source="validated_composition_projection")
        evidence.setdefault("observed_parts", {})[part_name] = {"current_pose": rest, "frame": "world",
                                                                "part_location": placed["location"],
                                                                "source": _projected_source(release)}
    base = evidence.get("snapshot", {}).get("base_pose")
    if base is not None and base != recovery_snapshot.get("base_pose"):
        reached = kmr_place_at(evidence["inputs"]["scene"], base)
        if "resource_location" in stated:
            # The statement takes the place whose dock the base ends on, and the postcondition compares them.
            write_location(resource, part, reached)
        elif reached != resource.get("resource_location"):
            return fail(f"The program ends at {reached or 'no named place'} and the event states no resource_location",
                        "INFEASIBLE")
        elif "pose" not in stated:
            return fail("The program moves the base and the event states neither resource_location nor pose", "INFEASIBLE")
    for field, aliases in {"part_location": ("location", "current_location"),
                           "part_holder_resource_jid": ("current_holder_resource_jid", "holder_resource_jid")}.items():
        if field in part:
            for alias in aliases:
                part[alias] = deepcopy(part[field])
    state = evidence.setdefault("snapshot", {})
    for field in ("base_pose", "held_part", "gripper_state", "resource_location"):
        if field in resource:
            state[field] = deepcopy(resource[field])
    # Only what the program changed: the KMR's whole evidence is megabytes, and every event's record would keep it.
    changed = {key: evidence[key] for key in ("snapshot", "attachment", "observed_parts") if key in evidence}
    return {"feasibility_status": "FEASIBLE", "projected_snapshot": resource, "projected_part_context": part,
            "resource_evidence": changed, "projected_resources": projected_resources,
            "scope": "The steps' physical checks, the transfers' custody at their checked steps, and the dock the base "
                     "ends on; a location is taken only where the event states it"}
