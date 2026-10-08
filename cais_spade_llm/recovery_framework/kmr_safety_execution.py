from __future__ import annotations

"""KMR owner contracts for nonexecuting preparation and retained trajectories."""

import math
from copy import deepcopy
from uuid import uuid4

from cais_spade_llm.recovery_framework import fingerprint
from cais_spade_llm.resources.continuous_motion import ContinuousMotion
from cais_spade_llm.resources.resource_safety_preparation import trajectory_record


def _target(primitive: str, params: dict, initial: list) -> list:
    if primitive == "move_cartesian":
        if set(params) - {"target", "waypoints", "seed"} or params.get("seed") is not None:
            raise ValueError("KMR preparation requires explicit Cartesian parameters without an unmodeled seed")
        target = params["target"]
    elif primitive == "move_relative":
        if set(params) != {"offset"} or len(params["offset"]) != 3:
            raise ValueError("KMR relative preparation requires its exact world XYZ offset")
        target = [*[a + b for a, b in zip(initial[:3], params["offset"], strict=True)], *initial[3:]]
    else:
        raise ValueError("KMR physical preparation does not support " + primitive)
    if (not isinstance(target, list) or len(target) != 7
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in target)
            or not math.isclose(sum(v * v for v in target[3:]), 1., abs_tol=1e-5)):
        raise ValueError("KMR Cartesian target requires a finite pose and unit quaternion")
    return target


def kmr_motion_effects(*, primitive, params, evidence, start_time, end_time, configuration):
    """Interpret KMR joint interpolation using the registered owner kinematics."""
    raw = evidence["continuous_motion"]
    if (raw["configuration"] != configuration["continuous_motion"]
            or raw["joint_trajectory"] != evidence["joint_trajectory"]):
        raise ValueError("KMR continuous evidence changed its owner configuration")
    motion = ContinuousMotion(raw["joint_trajectory"], raw["configuration"])
    if motion.exact_times[-1] != end_time - start_time:
        raise ValueError("KMR continuous trajectory duration changed")
    target = _target(primitive, params, motion.reference_pose(0))
    endpoint = motion.reference_pose(motion.duration)
    if (math.dist(endpoint[:3], target[:3]) > .001
            or abs(sum(a * b for a, b in zip(endpoint[3:], target[3:], strict=True))) < math.cos(.001 / 2)):
        raise ValueError("KMR prepared motion does not reach its command")
    return {"trajectory": None, "base_trajectory": None, "part_trajectories": {},
            "transfers": [], "resource_updates": {}, "continuous_motion": deepcopy(raw)}


def worker_safety_request(*, request, session, probe, config, kmr, observed_state,
                          updated_state, plan_motion, execute_plan, base_pose, tcp_pose, execution_evidence):
    """Prepare and execute KMR trajectories retained by the same live worker.

    Args:
        request: Owner request identifying preparation or the stored command.
        session: Persistent worker storage, never supplied by the request.
        probe: Verified Gazebo launch and scene identity.
        config: Registered KMR task execution configuration.
        kmr: Registered KMR scene declaration.
        observed_state: Fresh native RobotState reader.
        updated_state: Pure projected RobotState helper.
        plan_motion: Native nonexecuting collision-aware planner.
        execute_plan: Native dispatcher for an already planned trajectory.
        base_pose: Fresh physical base pose.
        tcp_pose: Fresh forward-kinematic tool pose reader.

    Returns:
        Owner-produced preparation or actual execution evidence.
    """
    saved = session.setdefault("safety_preparations", {})
    incarnation = session.setdefault("safety_incarnation", uuid4().hex)
    state = observed_state()
    joints = dict(zip(state.joint_state.name, state.joint_state.position, strict=True))
    names = kmr["arm_joint_names"]
    if request["mode"] == "safety_execute":
        identity = request["preparation_id"]
        stored = saved.get(identity)
        if stored is None or stored["consumed"]:
            raise ValueError("KMR preparation is absent or already consumed")
        record = stored["record"]
        if (record["start"]["launch_id"] != probe["launch_id"]
                or record["configuration_fingerprint"] != fingerprint(config)
                or record["start"]["base_pose"] != base_pose):
            raise ValueError("KMR owner or base changed after preparation")
        if any(abs(joints[name] - value) > .000001
               for name, value in zip(names, record["start"]["joint_positions"], strict=True)):
            raise ValueError("KMR moved after physical preparation")
        if (request["primitive"] != record["primitive"] or request["params"] != record["params"]
                or request["joint_trajectory"] != record["joint_trajectory"]):
            raise ValueError("KMR dispatch differs from its stored native trajectory")
        stored["consumed"] = True
        execute_plan(stored["plan"])
        actual = observed_state()
        values = dict(zip(actual.joint_state.name, actual.joint_state.position, strict=True))
        return {"status": "completed", "resource_id": "KMR", "result": {
            "success": True, "preparation_id": identity, "command_sent": True,
            "joint_trajectory": deepcopy(record["joint_trajectory"]),
            "controller_goals": [row for row in execution_evidence() if row.get("goal_id")],
            "observations": {"joint_names": list(names),
                             "joint_positions": [values[name] for name in names],
                             "current_pose": tcp_pose(), "base_pose": deepcopy(base_pose),
                             "launch_id": probe["launch_id"], "worker_incarnation": incarnation}}}
    start = deepcopy(request["start"])
    if (start["launch_id"] != probe["launch_id"] or start["base_pose"] != base_pose
            or start["joint_names"] != names
            or any(abs(joints[name] - value) > .000001
                   for name, value in zip(names, start["joint_positions"], strict=True))):
        raise ValueError("KMR planning snapshot no longer matches its physical owner")
    if config.get("avoid_collisions") is False:
        raise ValueError("KMR physical preparation requires collision-aware planning")
    result = []
    for index, command in enumerate(request["program"]["primitive_steps"]):
        target = _target(command["primitive"], command["params"], start["current_pose"])
        plan = plan_motion(state, target=target, cartesian=True,
                           waypoints=command["params"].get("waypoints"))
        trajectory = trajectory_record(plan[0].joint_trajectory, names)
        continuous = {"joint_trajectory": deepcopy(trajectory),
                      "configuration": deepcopy(request["continuous_motion"])}
        motion = ContinuousMotion(trajectory, continuous["configuration"])
        if math.dist(motion.reference_pose(0)[:3], start["current_pose"][:3]) > .001:
            raise ValueError("KMR configured kinematics disagree with the observed tool")
        if math.dist(motion.reference_pose(motion.duration)[:3], target[:3]) > .001:
            raise ValueError("KMR continuous motion does not reach the requested endpoint")
        record = {"status": "prepared", "observation_status": "prepared", "reason": "",
                  "primitive": command["primitive"], "params": deepcopy(command["params"]),
                  "binding": {**deepcopy(request["binding"]), "step_index": index},
                  "command_sent": False, "start": deepcopy(start), "target_pose": target,
                  "resolved_helper_outputs": {}, "configuration_fingerprint": fingerprint(config),
                  "joint_trajectory": trajectory, "joint_trajectory_fingerprint": fingerprint(trajectory),
                  "continuous_motion": continuous, "planning_evidence": deepcopy(plan[1]),
                  "worker_incarnation": incarnation}
        record["preparation_id"] = fingerprint(record)
        saved[record["preparation_id"]] = {"record": deepcopy(record), "plan": plan, "consumed": False}
        result.append(record)
        state = updated_state(state, names, trajectory["points"][-1]["positions"])
        start["joint_positions"] = list(trajectory["points"][-1]["positions"])
        start["current_pose"] = list(target)
    return {"status": "completed", "resource_id": "KMR", "preparation": {
        "status": "prepared", "resource_id": "KMR", "resource_jid": request["resource_jid"],
        "checkpoint_id": request["binding"]["checkpoint_id"],
        "program": deepcopy(request["program"]), "program_fingerprint": fingerprint(request["program"]),
        "steps": result}}
