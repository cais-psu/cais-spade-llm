from __future__ import annotations

"""CCA safety validation of each recovery outline event: Eq. 13 of the recovery paper.

Before an outline event is applied, the CCA extends every specification monitor with the labels of the
states the event passes through and leaves, π(s_F s^R_{k-1} e^R_k) = π(s_F s^R_{k-1}) ℓ(x^R_k), and
refuses the event when a monitor can no longer reach an accepting state. The LLM's candidates are
checked in its order, RA feasibility first, and the first candidate that passes both is applied
(Eq. 9: R* = R_feasible ∩ R_safe).

Labels come from the kinds of condition the predefined specifications use, one reader per kind and
never one per rule: a part's completed processes from the projected ledger; a resource in a region and
a part entering a region from physical evidence. Resources an event has not moved keep their recorded
collision boxes from the failure. The acting robot is placed, by forward kinematics over its recorded
chain, at the joint configuration the RA's reach check found for each work point and at its end pose;
the outline uses that posture, and the executed trajectory is checked again at primitive composition
(user decision, October 26). A specification that uses a kind with no reader stops the check before it
starts.

An event is judged in steps: during it (the acting robot at its work points, a placed part entering
its region), after it (the end pose and the new ledger), and once more for every process still running
when it ends, as if that process then completed on its own, since the CCA cannot prevent an
uncontrollable completion (user decision, October 26). Monitors start from their states at the failure,
replayed from the capture's recorded transitions. Captures are taken with the CCA bypassed: a failure
state that already breaks a specification stops the check unless a declared assumption covers it, for
example a robot that would have waited clear of an area until an event moves it.
"""

import json
from copy import deepcopy
from itertools import product
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from cais_spade_llm.agents.central_controller.base_safety_checker import BaseSafetyChecker
from cais_spade_llm.agents.central_controller.offline_safety_grounding import _catalog, _rules
from cais_spade_llm.agents.central_controller.ppr_ap import physical_ap_kind
from cais_spade_llm.agents.central_controller.predefined_safety import (
    compile_predefined_safety,
    parse_predefined_safety,
)
from cais_spade_llm.agents.central_controller.reviewed_primitive_program_safety import _ap_binding
from cais_spade_llm.recovery_framework.nominal_physical_gate import (
    NominalPhysicalGate,
    _overlap,
    _record_key,
)
from cais_spade_llm.resources.environment_models import _process_effects, build_environment_models
from cais_spade_llm.resources.kinematic_reach import chain_frames, pose_matrix, with_root
from cais_spade_llm.resources.nominal_des import FINISH_SUFFIX

SAFE, UNSAFE, NEEDS_CONTEXT = "SAFE", "UNSAFE", "NEEDS_CONTEXT"
# The kinds of condition this check can read; a specification using any other kind cannot be checked.
READERS = frozenset({"resource_region", "part_region_entry", "process_target_completed",
                     "process_result_completed"})
_UNKNOWN = (False, True)
_NO_CELL = {"resource": None, "occupancy": {}, "part": None, "part_entry": {}, "completions": frozenset()}


def capture_inputs(observations: dict, safety_files: list[str], assumptions: list[dict] | None = None) -> dict:
    """Ground the selected specifications over a capture and start their monitors at the failure.

    Args:
        observations: Saved failure capture, as read for an outline draft.
        safety_files: The selected predefined specification files.
        assumptions: Declared facts about the failure state the capture could not show, each
            ``{"resource": rid, "clear_of": [region, ...], "reason": text}``: the resource counts as outside
            those regions until an event moves it.

    Returns:
        What :func:`check_event` reads: the grounded rules and their monitor states at the failure, the
        recorded geometry, the robots' kinematic models and the JIDs.

    Raises:
        ValueError: A specification uses a kind of condition with no reader, or the failure state already
            breaks a specification that no assumption covers.
    """
    from cais_spade_llm.recovery_framework import ROOT

    physical = observations["physical_evidence"]
    scene = observations["inputs"]["scene"]
    document = _document(safety_files, ROOT)
    models = build_environment_models(scene)
    rules, checker = _ground(document, observations, models)
    geometry = physical["geometry"]
    model_map = observations["inputs"]["geometry"].get("parts", {}).get("model_map", {})
    bodies = physical.get("physics_evidence", {}).get("models", {})
    robots = [row["resource_id"] for row in scene.get("robots", [])] + (["KMR"] if isinstance(scene.get("KMR"), dict) else [])
    docks = {row["resource_id"]: row["KMR_docking_pose"] for row in scene.get("machines", []) if "KMR_docking_pose" in row}
    docks.update({name: row["KMR_docking_pose"] for name, row in scene.items()
                  if isinstance(row, dict) and isinstance(row.get("KMR_docking_pose"), list)})
    capture = {
        "jids": dict(observations["jids"]), "rids": {jid: rid for rid, jid in observations["jids"].items()},
        "rules": rules, "checker": checker,
        "requirements": {spec["id"]: spec.get("requirement", "") for spec in document["catalog"]["specifications"]},
        "regions": {name: row["bounds"] for name, row in geometry["regions"].items()},
        "region_resources": sorted({binding["resource"] for rule in rules for kind, binding in rule["ap_kinds"].values()
                                    if kind == "resource_region"}),
        "used_regions": sorted({binding["region"] for rule in rules for kind, binding in rule["ap_kinds"].values()
                                if kind == "resource_region"}),
        "entry_regions": sorted({binding["region"] for rule in rules for kind, binding in rule["ap_kinds"].values()
                                 if kind == "part_region_entry"}),
        "recorded_boxes": {rid: [row["bounds"] for row in values.get("component_bounds", [])]
                           for rid, values in geometry["resources"].items()},
        "part_boxes": {part: [collision["bounds"] for link in bodies[model]["links"].values()
                              for collision in link["collisions"] if collision.get("bounds")]
                       for part, model in model_map.items() if model in bodies},
        "part_reach": {part: max(abs(value) for axis in row["local_bounds"] for value in axis)
                       for part, row in geometry.get("parts", {}).items() if row.get("local_bounds")},
        "robots": {rid: _robot(rid, physical) for rid in robots if rid in physical.get("resource_models", {})},
        "docks": {name: [pose[0], pose[1], pose[5]] for name, pose in docks.items() if len(pose) == 6},
        "arms": ({"parked": scene["KMR"].get("parked_arm_configuration"),
                  "carrying": (scene["KMR"].get("task_execution") or {}).get("carrying_arm_configuration")}
                 if isinstance(scene.get("KMR"), dict) else {}),
        "assumptions": deepcopy(assumptions or []),
        "in_motion": _in_motion(observations),
        "models": models,
    }
    capture["failure_states"] = _failure_states(capture, observations)
    return capture


def check_event(event: dict, session: dict, prepared: dict, capture: dict, ra_result: dict) -> dict:
    """Check one candidate event against every specification (Eq. 13).

    Args:
        event: One candidate event as the LLM states it, with JIDs.
        session: The projected planning state before the event.
        prepared: Recorded failure context.
        capture: The grounded specifications and recorded geometry from :func:`capture_inputs`.
        ra_result: The RA's feasibility result for this candidate; its reach configurations place the
            acting robot.

    Returns:
        ``safety_status`` (SAFE, UNSAFE or NEEDS_CONTEXT), the refusal ``reasons`` (facts only), the
        ``steps`` checked and, for :func:`commit`, the monitor states and robot poses after the event.
    """
    state = _state(session, capture)
    after = _projected(event, session, prepared)
    rid = capture["rids"].get(event.get("resource_jid"), event.get("resource_jid"))
    jid = capture["jids"].get(rid, event.get("resource_jid"))
    before_parts, after_parts = session.get("symbolic_parts", {}), after.get("symbolic_parts", {})
    robots = deepcopy(state["robots"])
    result = {"resource_jid": event.get("resource_jid"), "event_name": event.get("event_name"), "steps": []}
    work, end = [], None
    if rid in capture["robots"]:
        work, end = _acting_poses(rid, jid, event, session, after, capture, robots, ra_result)
        if end is None:
            end_state = event.get("expected_end_state") or {}
            stated = " and ".join(f"{name} {json.dumps(end_state[name])}" for name in ("resource_location", "pose")
                                  if name in end_state)
            moving = (rid not in robots and rid in capture["in_motion"] and end_state.get("resource_location")
                      == session.get("symbolic_resources", {}).get(jid, {}).get("resource_location"))
            reason = (f"{rid}'s position after this event is not known: the event states {stated}"
                      + (f", the location {rid} records while it was still moving at the failure," if moving else "")
                      + f" and no reach configuration, named posture or declared pose places {rid} there")
            return {**result, "safety_status": NEEDS_CONTEXT, "reasons": [reason]}
    monitors = {rule_id: set(states) for rule_id, states in state["monitors"].items()}
    entries = _placed_entries(before_parts, after_parts, capture)
    steps = [("during the event", pose, entries if index == len(work) - 1 else {}, before_parts)
             for index, pose in enumerate(work)]
    if not work and entries:
        steps.append(("during the event", end, entries, before_parts))
    steps.append(("after the event", end, {}, after_parts))
    done = {"during the event": _completions(session, prepared, capture),
            "after the event": _completions(after, prepared, capture)}
    for label, pose, placed, parts in steps:
        base = _base(capture, robots, rid, pose, parts, placed, before_parts, done[label])
        reached, violation = _advance(capture, monitors, base, universal=True)
        result["steps"].append({"step": label, "status": SAFE if violation is None else UNSAFE})
        if violation is not None:
            return {**result, "safety_status": UNSAFE, "reasons": [_violation_text(capture, label, violation)],
                    "violation": _violation_evidence(violation)}
        monitors = reached
    final = _base(capture, robots, rid, end, after_parts, {}, before_parts, done["after the event"])
    for label, key in _running_completions(after, prepared, capture):
        base = {**final, "completions": {**final["completions"], key: (True,)}}
        _, violation = _advance(capture, monitors, base, universal=True)
        result["steps"].append({"step": label, "status": SAFE if violation is None else UNSAFE})
        if violation is not None:
            return {**result, "safety_status": UNSAFE, "reasons": [_violation_text(capture, label, violation)],
                    "violation": _violation_evidence(violation)}
    if rid in capture["robots"] and end is not None and (work or end != _pose_of(rid, robots, capture)):
        robots[rid] = deepcopy(end)
    return {**result, "safety_status": SAFE, "reasons": [],
            "state_after": {"monitors": {rule_id: sorted(states) for rule_id, states in monitors.items()},
                            "robots": robots}}


def commit(session: dict, result: dict) -> None:
    """Keep the monitor states and robot poses an applied event leaves, for the next event's check."""
    session["cca_safety_state"] = deepcopy(result["state_after"])


def recorded(result: dict) -> dict:
    """Return a check result as kept for review, without the state kept for the next check."""
    return {key: deepcopy(value) for key, value in result.items() if key != "state_after"}


def refusal_findings(events: list[dict], results: list[dict]) -> list[dict]:
    """Return one outline validation finding per candidate the CCA refused, stating only facts."""
    findings = []
    for result in results:
        if result["safety_status"] == SAFE:
            continue
        event = events[result["candidate_index"]]
        reason = "; ".join(result["reasons"])
        if result["safety_status"] == NEEDS_CONTEXT:
            reason = "NEEDS_CONTEXT: " + reason
        findings.append({"phase": "outline", "constraint_code": "cca_safety", "validation_category": "CCA",
                         "validation_status": result["safety_status"], "resource_jid": event.get("resource_jid"),
                         "part_name": event.get("part_name"), "event_name": event.get("event_name"),
                         "outline_id": event.get("outline_id"), "task_id": event.get("outline_id"),
                         "candidate_index": result["candidate_index"], "reason": reason})
    return findings


def _document(safety_files: list[str], root: Path) -> dict:
    """Merge the selected predefined specification files into one document."""
    documents = []
    for name in safety_files:
        path = Path(name) if Path(name).is_absolute() else root / name
        parsed = parse_predefined_safety(path.read_text(encoding="utf-8"))
        if parsed is None:
            raise ValueError(f"The CCA safety check reads predefined specifications only: {name}")
        documents.append(parsed)
    if not documents:
        raise ValueError("The CCA safety check needs at least one selected specification")
    merged = {"mode": "predefined", "version": documents[0]["version"],
              "catalog": {"version": documents[0]["catalog"]["version"],
                          "specifications": [spec for document in documents for spec in document["catalog"]["specifications"]]},
              "requirement_scopes": [scope for document in documents for scope in document["requirement_scopes"]]}
    geometry = next((document["product_geometry"] for document in documents if document.get("product_geometry")), None)
    if geometry is not None:
        merged["product_geometry"] = geometry
    return parse_predefined_safety(json.dumps(merged))


def _ground(document: dict, observations: dict, models: dict) -> tuple[list[dict], BaseSafetyChecker]:
    """Ground the specifications over the captured scene, as the nominal physical gate does."""
    _, compiled = compile_predefined_safety(document)
    geometry = deepcopy(observations["physical_evidence"]["geometry"])
    targets = observations["inputs"]["geometry"].get("parts", {}).get("assembly_target_map", {})
    for part, target in targets.items():
        geometry["parts"].setdefault(part, {}).setdefault("target", target)
    rules, _ = _rules(_catalog(document["catalog"], structured_groundings=True), document["requirement_scopes"],
                      models, geometry)
    for rule in rules:
        bindings = {}
        for ap in rule["aps"]:
            kind = physical_ap_kind(ap)
            if kind not in READERS:
                raise ValueError(f"The CCA safety check cannot check {rule['id']}: it has no reader for "
                                 f"{kind or 'structured'} conditions")
            bindings[ap["label"]] = (kind, _ap_binding(rule, ap))
        rule["dfa_dot"] = compiled[rule["id"]]
        rule["ap_kinds"] = bindings
    return rules, BaseSafetyChecker({rule["rule_id"]: rule["dfa_dot"] for rule in rules}, rules)


def _robot(rid: str, physical: dict) -> dict:
    """Return a robot's recorded kinematic model and its pose at the failure."""
    recorded = physical["resources"][rid]
    base = recorded.get("base_pose")
    yaw = float(Rotation.from_quat(base[3:]).as_euler("xyz")[2]) if isinstance(base, list) and len(base) == 7 else None
    model = physical["resource_models"][rid]
    return {"configuration": model["configuration"], "named": model.get("named_joint_poses") or {},
            "failure_pose": {"joints": list(recorded["joint_positions"]),
                             "base": [base[0], base[1], yaw] if yaw is not None else None},
            "tcp_offset_m": abs(float(recorded.get("tcp_offset_z") or 0.0))}


def _failure_states(capture: dict, observations: dict) -> dict:
    """Replay the capture's recorded transitions, then observe the failure state once.

    Region occupancy before the failure was never recorded, so the replay leaves it unknown and keeps
    every monitor state some occupancy reaches; the region rules judge only the present, so the final
    observation of the recorded scene decides them.
    """
    checker, rules = capture["checker"], capture["rules"]
    states = {rule["rule_id"]: {checker.dfas[rule["rule_id"]]["initial"]} for rule in rules}
    unknown = {rid: {region: _UNKNOWN for region in capture["used_regions"]} for rid in capture["region_resources"]}
    transitions = observations.get("transitions") or []
    for record in transitions:
        entries = {}
        for part, after in record["product_after"].items():
            location = after.get("location")
            if location in capture["entry_regions"] and record["product_before"].get(part, {}).get("location") != location:
                entries.setdefault(part, {})[location] = (True,)
        states, _ = _advance(capture, states, {"occupancy": unknown, "part_entry": entries,
                                               "completions": _ledger(record["product_after"])}, universal=False)
    products = transitions[-1]["product_after"] if transitions else observations.get("initial_product_states", {})
    clear = {row["resource"]: set(row.get("clear_of") or []) for row in capture["assumptions"]}
    occupancy = {}
    for rid in capture["region_resources"]:
        boxes = list(capture["recorded_boxes"].get(rid, []))
        boxes += [box for part, row in products.items() if row.get("location") == rid
                  for box in capture["part_boxes"].get(part, [])]
        occupancy[rid] = {region: (False,) if region in clear.get(rid, set())
                          else (any(_overlap(box, capture["regions"][region]) for box in boxes),)
                          for region in capture["used_regions"]}
    states, _ = _advance(capture, states, {"occupancy": occupancy, "part_entry": {},
                                           "completions": _ledger(products)}, universal=False)
    broken = [rule for rule in rules if not states[rule["rule_id"]]]
    if broken:
        facts = []
        for rule in broken:
            inside = [f"{binding['resource']} in {binding['region']}" for kind, binding in rule["ap_kinds"].values()
                      if kind == "resource_region" and occupancy[binding["resource"]][binding["region"]] == (True,)]
            facts.append(f"{rule['id']} ({'; '.join(inside) or 'from the recorded history'})")
        raise ValueError("The failure state already breaks " + ", ".join(facts) + ". The capture was taken "
                         "with the CCA bypassed; declare an assumption for a resource that would have waited "
                         "clear, or recapture")
    return {rule_id: sorted(value) for rule_id, value in states.items()}


def _in_motion(observations: dict) -> list[str]:
    """Resources with an activity started and not finished at the failure: they are between places."""
    started, finished = {}, set()
    for record in observations.get("transitions") or []:
        task = record["acknowledgement"]
        if record.get("kind", "finish") == "start":
            started[task["task_id"]] = task["resource_id"]
        elif record.get("kind", "finish") == "finish":
            finished.add(task["task_id"])
    return sorted({rid for task_id, rid in started.items() if task_id not in finished})


def _state(session: dict, capture: dict) -> dict:
    """Return the monitor states and moved robots' poses after the accepted events."""
    return session.get("cca_safety_state") or {"monitors": deepcopy(capture["failure_states"]), "robots": {}}


def _projected(event: dict, session: dict, prepared: dict) -> dict:
    """Project the candidate on a copy of the planning state, exactly as applying it would."""
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes import (
        multi_turn,
        multi_turn_diagnostic,
    )

    task = multi_turn._commit_selected_candidate_task(
        task=multi_turn_diagnostic.event_record(event, session, prepared),
        sequence_index=multi_turn._next_recovery_sequence_index(session))
    after = multi_turn_diagnostic._branch(session)
    multi_turn_diagnostic.apply_tentative_effects(deepcopy(task), after, prepared)
    return after


def _acting_poses(rid: str, jid: str, event: dict, session: dict, after: dict, capture: dict, robots: dict,
                  ra_result: dict) -> tuple[list[dict], dict | None]:
    """Return the acting robot's work poses and end pose, or None for an end pose nothing determines.

    Work poses are the RA's reach configurations. The end pose is a named posture the event names, a
    mobile base's stated base pose or else the dock its location names (arm parked, or carrying while it
    holds a part), else the last work pose; an event that states no new position, or restates the one it
    has, leaves the robot where it was.
    """
    robot = capture["robots"][rid]
    previous = _pose_of(rid, robots, capture)
    work = [{"joints": list(witness["joint_positions"]),
             "base": list(witness["base_pose"]) if witness.get("base_pose") else previous["base"]}
            for witness in _witnesses(ra_result)]
    end_state = event.get("expected_end_state") or {}
    location = end_state.get("resource_location")
    before_location = session.get("symbolic_resources", {}).get(jid, {}).get("resource_location")
    if isinstance(location, str) and location in robot["named"]:
        return work, {"joints": list(robot["named"][location]["positions"]), "base": previous["base"]}
    if previous["base"] is not None:
        held = after.get("symbolic_resources", {}).get(jid, {}).get("held_part")
        arm = capture["arms"].get("carrying" if held else "parked")
        pose = end_state.get("pose")
        if isinstance(pose, dict) and all(isinstance(pose.get(axis), (int, float)) for axis in ("x", "y")) and arm:
            return work, {"joints": list(arm), "base": [pose["x"], pose["y"], float(pose.get("yaw") or 0.0)]}
        if isinstance(location, str) and location in capture["docks"] and arm:
            return work, {"joints": list(arm), "base": list(capture["docks"][location])}
    if work:
        return work, work[-1]
    unchanged = {row.get("condition") for row in ra_result.get("conditions", []) if row.get("derivation") == "unchanged"}
    pose_changed = "pose" in end_state and "pose" not in unchanged
    location_changed = "resource_location" in end_state and location != before_location
    if pose_changed or location_changed:
        return work, None
    # A robot still moving at the failure has not reached the location it records, so restating it places nothing.
    if "resource_location" in end_state and rid not in robots and rid in capture["in_motion"]:
        return work, None
    return work, previous


def _witnesses(ra_result: dict) -> list[dict]:
    """Return the RA's reach configurations for the event, in the order its checks found them."""
    rows = [row.get("witness") for row in ra_result.get("capability", [])]
    rows += [producer.get("witness") for condition in ra_result.get("conditions", [])
             for producer in condition.get("producers", []) if producer.get("status") == "FEASIBLE"]
    return [row for row in rows if isinstance(row, dict) and isinstance(row.get("joint_positions"), list)]


def _pose_of(rid: str, robots: dict, capture: dict) -> dict:
    return robots.get(rid) or capture["robots"][rid]["failure_pose"]


def _robot_boxes(robot: dict, pose: dict, held: str | None, capture: dict) -> list:
    """Return a robot's collision boxes at a pose, and a box around its TCP for the part it holds."""
    configuration = robot["configuration"]
    if pose.get("base") is not None:
        x, y, yaw = pose["base"]
        root = pose_matrix([x, y, configuration["root_xyz"][2], *Rotation.from_euler("z", yaw).as_quat()])
        configuration = with_root(configuration, root)
    frames = chain_frames(configuration, dict(zip(configuration["joint_names"], pose["joints"], strict=True)))
    boxes = []
    for component in configuration["components"]:
        frame = frames[component["link"]]
        corners = np.array([frame[:3, :3] @ np.array(corner) + frame[:3, 3] for corner in product(*component["bounds"])])
        boxes.append([[float(corners[:, axis].min()), float(corners[:, axis].max())] for axis in range(3)])
    if held in capture["part_reach"]:
        tool = frames[configuration["reference_link"]]
        center = tool[:3, 3] + tool[:3, 2] * robot["tcp_offset_m"]
        reach = capture["part_reach"][held]
        boxes.append([[float(center[axis] - reach), float(center[axis] + reach)] for axis in range(3)])
    return boxes


def _base(capture: dict, robots: dict, actor: str | None, actor_pose: dict | None, parts: dict, placed: dict,
          before_parts: dict, completions: dict) -> dict:
    """Valuation of one step: region occupancy, parts entering regions and completed processes."""
    clear = {row["resource"]: set(row.get("clear_of") or []) for row in capture["assumptions"]}
    held = {}
    for part, row in parts.items():
        holder = capture["rids"].get(row.get("part_location"), row.get("part_location"))
        if holder in capture["region_resources"]:
            held.setdefault(holder, part)
    occupancy, entries = {}, deepcopy(placed)
    for rid in capture["region_resources"]:
        acting = rid == actor and actor_pose is not None
        # A robot an event leaves where it stands keeps its recorded boxes and any declared assumption.
        moved = rid in robots or (acting and actor_pose != capture["robots"].get(rid, {}).get("failure_pose"))
        if rid in capture["robots"] and moved:
            pose = actor_pose if acting else robots[rid]
            boxes = _robot_boxes(capture["robots"][rid], pose, held.get(rid), capture)
        else:
            boxes = list(capture["recorded_boxes"].get(rid, []))
            if rid in held:
                boxes += capture["part_boxes"].get(held[rid], [])
        assumed = clear.get(rid, set()) if not moved else set()
        occupancy[rid] = {region: (False,) if region in assumed
                          else (any(_overlap(box, capture["regions"][region]) for box in boxes),)
                          for region in capture["used_regions"]}
        if rid == actor and rid in held and rid in capture["robots"] and moved:
            # A carried part that reaches into a region it was not in enters it.
            part = held[rid]
            reach = _robot_boxes(capture["robots"][rid], actor_pose, part, capture)[-1]
            start = before_parts.get(part, {}).get("part_location")
            start = capture["rids"].get(start, start)
            for region in capture["entry_regions"]:
                if start != region and _overlap(reach, capture["regions"][region]):
                    entries.setdefault(part, {})[region] = (True,)
    return {"occupancy": occupancy, "part_entry": entries, "completions": completions}


def _placed_entries(before_parts: dict, after_parts: dict, capture: dict) -> dict:
    """Parts the event places in a region they were not in."""
    entries = {}
    for part, row in after_parts.items():
        location = row.get("part_location")
        if location in capture["entry_regions"] and before_parts.get(part, {}).get("part_location") != location:
            entries.setdefault(part, {})[location] = (True,)
    return entries


def _running_completions(after: dict, prepared: dict, capture: dict) -> list[tuple[str, tuple]]:
    """Completions that may happen next on their own: each uncontrollable event enabled after the event.

    An event is enabled when its declared guards hold on its resource's projected state; a completion
    records the part's current process, as the event's declared product effect states it. A faulted
    resource completes nothing.
    """
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes.multi_turn_diagnostic import (
        faulted_resources,
    )

    faulted = {capture["rids"].get(jid, jid) for jid in faulted_resources(after, prepared)}
    parts = after.get("symbolic_parts", {})
    done, pending = _completions(after, prepared, capture), {}
    for jid, model in (after.get("recovery_des_models") or {}).items():
        rid = capture["rids"].get(jid, jid)
        row = after.get("symbolic_resources", {}).get(jid, {})
        if rid in faulted:
            continue
        for event in model.get("events", []):
            if event.get("controllable") is not False or event.get("start_event_id") is not None:
                continue
            params = {name: binding["equals"] for name, binding in (event.get("parameter_bindings") or {}).items()
                      if isinstance(binding, dict) and "equals" in binding}
            if not _guards_hold(event.get("guards") or {}, row, params):
                continue
            part = params.get("part_name")
            current = (parts.get(part) or {}).get("current_process")
            for effect in (event.get("product_effects") or {}).get("processCompleted") or []:
                record = {name: params.get(value["set_from_param"]) if isinstance(value, dict) and "set_from_param" in value
                          else value for name, value in effect.items()}
                if not isinstance(current, dict) or any(current.get(name) != value for name, value in record.items()):
                    continue
                key = _record_key(part, record)
                if key not in done:
                    value = record.get("result") or record.get("target") or ""
                    pending[key] = f"if {rid} then completes its running {record['process']} {value} of {part} on its own"
    return [(label, key) for key, label in pending.items()]


def _guards_hold(guards: dict, row: dict, params: dict) -> bool:
    """Whether a resource's declared guards hold on its projected state.

    A guard form this reader does not know is taken as holding, so an uncontrollable event it cannot
    rule out is still checked.
    """
    for field, condition in guards.items():
        if not isinstance(condition, dict):
            continue
        if "equals" in condition and row.get(field) != condition["equals"]:
            return False
        if "equals_from_param" in condition and row.get(field) != params.get(condition["equals_from_param"]):
            return False
    return True


def _completions(state: dict, prepared: dict, capture: dict) -> dict:
    """Completed processes in a projected state: each part's ledger, plus the declared results of the
    planned tasks the recovery claimed it completed, credited as the CCA nonblocking check credits them."""
    done = _ledger_rows(state.get("symbolic_parts", {}))
    nodes = {node.get("id"): node for node in prepared.get("saved_pending_nodes", [])}
    for identifier in state.get("diagnostic_replaced_task_claims", {}):
        node = nodes.get(identifier, {})
        params = node.get("params", {})
        part = params.get("part_name")
        for event in (capture["models"].get(params.get("resource_id")) or {}).get("events", []):
            if event["event_name"] not in {node.get("function_name"), f"{node.get('function_name')}{FINISH_SUFFIX}"}:
                continue
            if any(name in params and params[name] != binding["equals"]
                   for name, binding in event["parameter_bindings"].items()
                   if isinstance(binding, dict) and "equals" in binding):
                continue
            try:
                results = _process_effects(event, params)
            except KeyError:
                # The recorded task does not name the value its result takes.
                continue
            done.update({_record_key(part, record): (True,) for record in results})
    return done


def _ledger(products: dict) -> dict:
    return {_record_key(part, record): (True,) for part, row in products.items()
            for record in row.get("processCompleted") or []}


def _ledger_rows(parts: dict) -> dict:
    return {_record_key(part, record): (True,) for part, row in parts.items()
            for record in row.get("processCompleted") or [] if isinstance(record, dict) and record.get("process")}


def _advance(capture: dict, states: dict, base: dict, *, universal: bool) -> tuple[dict, dict | None]:
    """Advance every monitor by one valuation.

    With ``universal``, every state and every possible valuation must keep an accepting state reachable,
    else the first violation is returned. Without it, a valuation that would violate a rule is ruled out
    and only the reached states are kept (an empty set means the history already breaks the rule).
    """
    checker = capture["checker"]
    reached_all = {}
    for rule in capture["rules"]:
        rule_id = rule["rule_id"]
        values = NominalPhysicalGate._values(None, rule, _NO_CELL, base)
        labels = list(values)
        alphabets = {frozenset(label for label, bit in zip(labels, bits, strict=True) if bit)
                     for bits in product(*values.values())}
        reached = set()
        for state in states[rule_id]:
            for alphabet in alphabets:
                transition = checker.transition_evidence(rule_id, state, alphabet)
                if transition["status"] == "passed":
                    reached.add(transition["to"])
                elif universal:
                    return states, {"rule": rule, "alphabet": sorted(alphabet), "transition": transition}
        reached_all[rule_id] = reached
    return reached_all, None


def _violation_text(capture: dict, step: str, violation: dict) -> str:
    """State the broken specification and the facts of the step that breaks it."""
    rule, true = violation["rule"], set(violation["alphabet"])
    inside, facts = {}, []
    for label, (kind, binding) in sorted(rule["ap_kinds"].items()):
        if kind == "resource_region" and label in true:
            inside.setdefault(binding["resource"], []).append(binding["region"])
        elif kind == "part_region_entry" and label in true:
            facts.append(f"{binding['part']} enters {binding['region']}")
        elif kind in {"process_target_completed", "process_result_completed"}:
            value = binding["target" if kind == "process_target_completed" else "result"]
            facts.append(f"{binding['part']} {binding['process']} {value} "
                         f"{'completed' if label in true else 'not completed'}")
    occupied = " while ".join(f"{rid} is in {' and '.join(sorted(set(regions)))}" for rid, regions in inside.items())
    stated = "; ".join(([occupied] if occupied else []) + facts)
    requirement = capture["requirements"].get(rule["id"]) or rule["id"]
    return f"{rule['id']} ({requirement}) is broken {step}: {stated}"


def _violation_evidence(violation: dict) -> dict:
    rule = violation["rule"]
    return {"rule_id": rule["rule_id"], "specification": rule["id"], "true_conditions": violation["alphabet"],
            "bindings": {label: {"kind": kind, **binding} for label, (kind, binding) in rule["ap_kinds"].items()},
            "transition": deepcopy(violation["transition"])}
