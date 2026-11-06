from __future__ import annotations

"""KMR-owned step tracking and completion evidence for authored recovery programs."""

import math
from copy import deepcopy

from cais_spade_llm.agents.shared_information.recovery_validation_protocol import (
    recovery_validation_fingerprint,
)
from cais_spade_llm.recovery_framework.geometry import compose
from cais_spade_llm.recovery_framework.kmr_recovery import _pose_list, kmr_place_at, transport_pose

_POSE_AXES = ("x", "y", "z", "qx", "qy", "qz", "qw")
_MOTIONS = {"move_base", "move_cartesian", "move_relative", "move_to_named_pose"}


def _inverse(pose: list[float]) -> list[float]:
    from cais_spade_llm.recovery_framework.kmr_gazebo import inverse

    return inverse(pose)


def _same_pose(pose: list[float], target: list[float]) -> bool:
    return all(math.isclose(have, want, rel_tol=0.0, abs_tol=1e-9) for have, want in zip(pose, target, strict=True))


def _projected_source(payload: object) -> dict:
    return {"kind": "projected", "contract_fingerprint": recovery_validation_fingerprint(payload)}


def _arm_left_out(evidence: dict) -> str | None:
    """Say how far the arm stands from the transport pose of its base; None when it is there.

    That is the arm the CCA reads at the end of an empty KMR event, and the one a base move carries.
    """
    try:
        state, scene = evidence["snapshot"], evidence["inputs"]["scene"]
        transport = transport_pose(scene, state.get("base_pose"), state.get("current_pose"))
        tolerance = scene["KMR"]["task_execution"]["position_tolerance_m"]
    except (KeyError, ValueError) as exc:
        return f"The KMR transport pose cannot be checked: {exc}"
    distance = math.dist(state["current_pose"][:3], transport[:3])
    if distance <= tolerance:
        return None
    return (f"The KMR arm stands {distance:.3f} m from the transport pose of its base, and a base move and the end of "
            "a program need it there")


def _motion_refusal(evidence: dict, targets: dict, primitive: str, waypoints: list | None) -> str | None:
    """Refuse a motion that misses a calculated retreat or route, or that moves the base with the arm out.

    Only the pending move has waypoints to bind, as a release's retreat stands where its approach stood and has none.
    Position alone says the motion reached it, because a named pose keeps the tool's orientation.
    """
    pose = evidence["snapshot"].get("current_pose")
    if "retreat" in targets:
        retreat = targets.pop("retreat")
        if primitive == "move_base" or not _same_pose(pose, retreat):
            return "Finish the custody transfer at the calculated event_facts retreat"
    if primitive == "move_base":
        if "pending_move" in targets:
            return f"A base move leaves the calculated {targets['pending_move'][0]} stale"
        return _arm_left_out(evidence)
    if "pending_move" not in targets:
        return None
    name, field, calculated = targets["pending_move"]
    if not _same_pose(pose[:3], calculated[name][:3]):
        return None
    del targets["pending_move"]
    route, wanted = waypoints or [], calculated.get(field) or []
    if len(route) == len(wanted) and all(
            len(have) == len(want) and _same_pose(have, want) for have, want in zip(route, wanted, strict=True)):
        return None
    return f"The move to the calculated {name} leaves out event_facts {field}"


def advance_kmr_program(evidence: dict, targets: dict, *, task: dict, primitive: str, params: dict,
                        recovery_snapshot: dict, checked: dict) -> str | None:
    """Carry an accepted step into the KMR's evidence, so the program's later steps start from it.

    The KMR's arm motions declare no effects, so its tool pose and joints follow what its owner check found for each
    step. A base move carries the arm with it, which has to be at its transport pose. A custody transfer has to happen
    at the owner's calculated target, and the motion right after it has to reach the calculated retreat, which the RA
    has checked the arm reaches. A move to a calculated pose with calculated waypoints has to bind them.

    Args:
        evidence: The owner's evidence for this event, updated in place.
        targets: The owner's calculated pick and place targets for this event, updated in place.
        task: The accepted outline event.
        primitive: The step's primitive.
        params: The step's resolved parameters.
        recovery_snapshot: The resource state before the step.
        checked: The owner's capability result for the step.

    Returns:
        Why the program cannot take this step, or None.
    """
    state = evidence.setdefault("snapshot", {})
    facts = checked.get("evidence") or {}
    if primitive == "compute_place_targets" and params.get("part_pose") is not None:
        return "A release composes at the place the event states, so compute_place_targets takes no part_pose"
    if primitive in {"compute_pick_targets", "compute_place_targets"}:
        targets[primitive] = deepcopy(facts["calculated_targets"])
    elif primitive == "move_base":
        before = recovery_snapshot["base_pose"]
        x, y, yaw = params["target_pose"][:3]
        after = [x, y, before[2], 0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)]
        if "current_pose" in state:
            state["current_pose"] = compose(after, compose(_inverse(before), state["current_pose"]))
        state["base_pose"] = after
    elif primitive == "move_cartesian":
        state["current_pose"] = deepcopy(params["target"])
    elif primitive == "move_relative":
        position = [value + delta for value, delta in zip(state["current_pose"][:3], params["offset"], strict=True)]
        state["current_pose"] = [*position, *state["current_pose"][3:]]
    elif primitive == "move_to_named_pose":
        state["current_pose"] = deepcopy(facts["target"])
    if primitive in {"move_cartesian", "move_relative"} and "joint_positions" in facts:
        # The arm ends where its path leaves it, which may be another branch than the end-pose witness.
        path = facts.get("path") or {}
        state["joint_positions"] = deepcopy(path.get("final_joint_positions") or facts["joint_positions"])
    if primitive in _MOTIONS:
        return _motion_refusal(evidence, targets, primitive, params.get("waypoints"))
    if primitive == "compute_place_targets":
        targets["pending_move"] = ("approach", "transfer_waypoints", targets[primitive])
    elif primitive == "compute_pick_targets":
        targets["pending_move"] = ("approach", "approach_waypoints", targets[primitive])
    if primitive not in {"grasp_part", "release_part"}:
        return None
    calculated = targets.get("compute_pick_targets" if primitive == "grasp_part" else "compute_place_targets")
    if calculated is None:
        return "The custody transfer requires a preceding owner target calculation"
    if "pending_move" in targets or not _same_pose(state["current_pose"], calculated["target"]):
        through = ", through the calculated approach" if "pending_move" in targets else ""
        return f"The custody transfer must reach the calculated target pose and orientation{through}"
    targets["retreat"] = deepcopy(calculated["retreat"])
    pose, field = ("lift", "lift_waypoints") if primitive == "grasp_part" else ("transport", "withdraw_waypoints")
    targets["pending_move"] = (pose, field, calculated)
    part = params.get("part_name", task.get("part_name"))
    source = _projected_source(calculated)
    if primitive == "grasp_part":
        row = evidence.get("observed_parts", {}).get(part, {})
        observed = _pose_list(row.get("observed_pose", row.get("current_pose")))
        evidence["attachment"] = {"attached": True, "part_name": part, "held_part": part, "source": source,
                                  "grasp_transform": compose(_inverse(state["current_pose"]), observed)}
    else:
        evidence["attachment"] = {"attached": False, "source": source}
    return None


def preview_kmr_output(targets: dict, *, task: dict, primitive: str, **_unused) -> dict:
    """Publish the owner's calculated targets as ``event_facts``, under the names its decompositions use."""
    if primitive not in targets:
        return {}
    key = "pick_targets" if primitive == "compute_pick_targets" else "place_targets"
    return {"event_fact_key": f"{key}.{task['part_name']}", "output": deepcopy(targets[primitive]), "status": "calculated"}


def project_kmr_completion(*, task: dict, trace: list, recovery_snapshot: dict, projected_snapshot: dict,
                           part_context: dict, grounding_context: dict, targets: dict, resource_evidence: dict,
                           physical_evidence: dict, **_unused) -> dict:
    """Project a checked KMR program: where it left the base, the arm and the event's part.

    Code writes no value the event left out. A location the event states takes the dock the program's base ended on,
    or none, for the postcondition to compare; a base move the event does not state is refused. The custody of a
    part follows the transfer the owner checked. A program ends with the arm at the transport pose of its base, read
    from the owner's evidence, because the symbolic current_pose of a KMR is its base at the failure.
    """
    from cais_spade_llm.resources.robot.recovery_composition import (
        _source_records,
        receive_placed_part,
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
    if "retreat" in targets:
        return fail("Owner completion requires a retreat after the custody transfer", "INFEASIBLE")
    if "pending_move" in targets:
        return fail(f"The program does not reach the calculated {targets['pending_move'][0]}", "INFEASIBLE")
    for transfer in transfers:
        part_name = transfer["resolved_params"].get("part_name", task.get("part_name"))
        if part_name != task.get("part_name"):
            return fail("Custody transfer identifies a different part", "INFEASIBLE")
        if transfer["primitive"] == "grasp_part":
            # The place the part leaves, whose owner's model declares what taking it leaves there, as for a robot.
            origin = part_context.get("part_location", part_context.get("location"))
            part.update(part_location=jid, part_holder_resource_jid=jid)
            for owner, field, empty in _source_records(part_name, jid, grounding_context, physical_evidence,
                                                       task.get("neighbor_end_state") or {}, origin=origin):
                projected_resources.setdefault(owner, {})[field] = empty
            continue
        destination = stated.get("part_location")
        if not isinstance(destination, str):
            return fail("A KMR release needs the place the event states for its part")
        pose = targets["compute_place_targets"]["destination"]
        placed = receive_placed_part(part_name, destination, dict(zip(("x", "y", "z"), pose, strict=False)),
                                     grounding_context=grounding_context, physical_evidence=physical_evidence,
                                     projected_resources=projected_resources, acting_jid=jid,
                                     stated=task.get("neighbor_end_state") or {})
        if "failure" in placed:
            return fail(placed["failure"], placed["status"])
        part.update(part_location=placed["location"], part_holder_resource_jid=None,
                    observed_pose=dict(zip(_POSE_AXES, pose, strict=True)), observed_pose_current=False,
                    pose_source="validated_composition_projection")
        evidence.setdefault("observed_parts", {})[part_name] = {"current_pose": deepcopy(pose), "frame": "world",
                                                                "part_location": placed["location"],
                                                                "source": _projected_source(pose)}
    base = evidence.get("snapshot", {}).get("base_pose")
    if base is not None and base != recovery_snapshot.get("base_pose"):
        reached = kmr_place_at(evidence["inputs"]["scene"], base)
        if "resource_location" in stated:
            for name in ("resource_location", "current_location"):
                resource[name] = reached
            if isinstance(resource.get("occupancy"), dict):
                resource["occupancy"]["location"] = reached
        elif reached != resource.get("resource_location"):
            return fail(f"The program ends at {reached or 'no named place'} and the event states no resource_location",
                        "INFEASIBLE")
        elif "pose" not in stated:
            return fail("The program moves the base and the event states neither resource_location nor pose", "INFEASIBLE")
    left_out = _arm_left_out(evidence)
    if left_out:
        return fail(left_out, "INFEASIBLE")
    for field, aliases in {"part_location": ("location", "current_location"),
                           "part_holder_resource_jid": ("current_holder_resource_jid", "holder_resource_jid")}.items():
        if field in part:
            for alias in aliases:
                part[alias] = deepcopy(part[field])
    state = evidence.setdefault("snapshot", {})
    for field in ("base_pose", "held_part", "resource_location"):
        if field in resource:
            state[field] = deepcopy(resource[field])
    # Only what the program changed: the KMR's whole evidence is megabytes, and every event's record would keep it.
    changed = {key: evidence[key] for key in ("snapshot", "attachment", "observed_parts") if key in evidence}
    return {"feasibility_status": "FEASIBLE", "projected_snapshot": resource, "projected_part_context": part,
            "resource_evidence": changed, "projected_resources": projected_resources,
            "scope": "Owner-checked steps and custody transfers; a location is taken only where the event states it"}
