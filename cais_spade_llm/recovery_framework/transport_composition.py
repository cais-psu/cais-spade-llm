from __future__ import annotations

"""Transport-owned step checks and completion evidence for authored recovery programs."""

import math
from copy import deepcopy

from cais_spade_llm.agents.shared_information.recovery_validation_protocol import (
    recovery_validation_fingerprint,
)
from cais_spade_llm.recovery_framework.kmr_recovery import _pose_list
from cais_spade_llm.recovery_framework.workflow_gazebo import _shared_displacement
from cais_spade_llm.resources.environment_models import feasibility
from cais_spade_llm.resources.nominal_conveyor import conveyor_advance_parameters, conveyor_parts

_ADVANCE_EVENTS = {"advance_conveyor", "advance_part"}


def _valuation(evidence: dict, snapshot: dict) -> dict:
    """Read the transport's recorded variables as the accepted composition prefix leaves them."""
    values = deepcopy(evidence.get("environment_valuation", {}).get(evidence["resource_id"], {}))
    values.update({key: deepcopy(value) for key, value in snapshot.items() if key in values})
    return values


def _advance_event(model: dict, rid: str, values: dict, part: str | None) -> dict | None:
    """Return the resource's own advance event that can start from ``values``, never a delivery off the belt."""
    for event in model["events"]:
        bindings = event["parameter_bindings"]
        if (event["event_name"] in _ADVANCE_EVENTS and bindings["resource_id"].get("equals") == rid
                and "equals" in bindings.get("delivered_part", {"equals": None})
                and all(values.get(field) == (part if "equals_from_param" in guard else guard.get("equals"))
                        for field, guard in event["guards"].items())):
            return event
    return None


def _moved_parts(rid: str, model: dict, values: dict, part: str | None) -> list[str]:
    """Return the parts a transport moves: every belt resident, or the part in its source zone."""
    if rid == "Conveyor":
        return conveyor_parts(model, values)
    return [part] if part else []


def check_transport_primitive(*, primitive: str, params: dict, task: dict, recovery_snapshot: dict,
                              resource_evidence: dict) -> tuple[str, str]:
    """Check one transport primitive at one state, under the guards of the resource's own advance event.

    ``advance_conveyor`` declares no part location, since a belt's parts move together and the runtime binds where
    each one ends, so a task cannot be matched to it by its declared updates. These checks read its guards instead.

    Args:
        primitive: The step's primitive.
        params: The step's resolved parameters.
        task: The accepted outline event.
        recovery_snapshot: The transport's state before the step.
        resource_evidence: The transport's recorded evidence.

    Returns:
        The feasibility status and its reason.
    """
    rid, part = resource_evidence["resource_id"], task.get("part_name")
    model = resource_evidence["environment_models"][rid]
    values = _valuation(resource_evidence, recovery_snapshot)
    event = _advance_event(model, rid, values, part)
    if event is None:
        return "INFEASIBLE", f"{rid} cannot advance {part} from its recorded state"
    try:
        moved = _moved_parts(rid, model, values, part)
    except ValueError as exc:
        return "INFEASIBLE", str(exc)
    if not moved:
        return "NEEDS_CONTEXT", f"No recorded {rid} part custody is available"
    for name in moved:
        status, reasons = feasibility(model, {"event_name": event["event_name"], "parameters": {}},
                                      resource_evidence.get("geometry", {}).get(name, {}))
        if status != "FEASIBLE":
            return status, "; ".join(reasons)
    if primitive == "move_relative":
        configured = resource_evidence["inputs"]["scene"][rid]["simulation_transport"].get("speed_mps")
        speed = params.get("speed_mps")
        speed = configured if speed is None else speed
        if type(configured) not in (int, float) or not math.isfinite(configured) or configured <= 0:
            return "NEEDS_CONTEXT", "Declared transport speed is unavailable"
        if type(speed) not in (int, float) or not math.isfinite(speed) or not 0 < speed <= configured:
            return "INFEASIBLE", "Transport speed exceeds the declared owner capability"
    return "FEASIBLE", "Declared transport primitive; where its parts arrive is calculated from the observed parts"


def _belt_targets(evidence: dict, values: dict, stated: dict, observed: dict) -> dict:
    """Calculate the one displacement the belt gives every resident, to the places the event states.

    The destinations are the event's. One the event leaves out is refused, since the belt moves that part too.
    """
    model = evidence["environment_models"]["Conveyor"]
    fields = {name: f"part_location.{name}" for name in observed["names"]}
    unstated = [name for name, field in fields.items() if field not in stated]
    if unstated:
        raise ValueError(f"The belt moves {', '.join(unstated)} too, and the event states no part_location for it")
    next_locations = {name: stated[field] for name, field in fields.items()}
    if None in next_locations.values():
        raise ValueError("A part leaving the belt for Buffer For Machined parts has no composed program")
    conveyor_advance_parameters(model, values, next_locations, None)
    displacement = _shared_displacement(evidence["inputs"]["scene"], observed["initial_x"], next_locations)
    return {"target_x": {name: x + displacement for name, x in observed["initial_x"].items()},
            "displacement": displacement,
            "end_state": {fields[name]: location for name, location in next_locations.items()}}


def _zone_targets(evidence: dict, values: dict, stated: dict, observed: dict, part: str) -> dict:
    """Calculate the part's move into its next zone; the event has to state both zones it changes."""
    rid = evidence["resource_id"]
    bindings = _advance_event(evidence["environment_models"][rid], rid, values, part)["parameter_bindings"]
    zone, downstream = bindings["zone"]["equals"], bindings["downstream_zone"]["equals"]
    end_state = {f"zone_{zone}_part": None, f"zone_{downstream}_part": part}
    unstated = [field for field in end_state if field not in stated]
    if unstated:
        raise ValueError(f"{rid} moves {part}, and the event states no {', '.join(unstated)}")
    target_x = float(evidence["inputs"]["scene"][rid]["slot_poses"][downstream - 1][0])
    return {"target_x": {part: target_x}, "displacement": target_x - observed["initial_x"][part], "end_state": end_state}


def advance_transport_program(evidence: dict, targets: dict, *, task: dict, primitive: str, recovery_snapshot: dict,
                              **_unused) -> str | None:
    """Keep what an accepted step observed or calculated, as ``WorkflowPrimitiveRunner`` keeps it for the next step.

    Args:
        evidence: The transport's evidence for this event.
        targets: What this event's steps observed and calculated, updated in place.
        task: The accepted outline event.
        primitive: The step's primitive.
        recovery_snapshot: The transport's state before the step.

    Returns:
        Why the step cannot run at this point of the program, or None.
    """
    rid, part = evidence["resource_id"], task.get("part_name")
    model = evidence["environment_models"][rid]
    values = _valuation(evidence, recovery_snapshot)
    if primitive == "detect_parts":
        names = _moved_parts(rid, model, values, part)
        rows = {name: evidence.get("observed_parts", {}).get(name, {}) for name in names}
        unknown = [name for name, row in rows.items() if not row]
        if unknown:
            return f"{rid} has no recorded position of {', '.join(unknown)}"
        targets[primitive] = {"names": names, "initial_x": {
            name: _pose_list(row.get("observed_pose", row.get("current_pose")))[0] for name, row in rows.items()}}
    elif primitive == "compute_transport_targets":
        if "detect_parts" not in targets:
            return "compute_transport_targets needs the observed parts"
        stated = task.get("expected_end_state") or {}
        try:
            targets[primitive] = (_belt_targets(evidence, values, stated, targets["detect_parts"]) if rid == "Conveyor"
                                  else _zone_targets(evidence, values, stated, targets["detect_parts"], part))
        except ValueError as exc:
            return str(exc)
    return None


def preview_transport_output(targets: dict, *, primitive: str, **_unused) -> dict:
    """Publish the calculated targets as ``event_facts``; ``move_relative``'s declared effect reads them there."""
    if primitive != "compute_transport_targets" or primitive not in targets:
        return {}
    return {"event_fact_key": "transport_targets", "output": deepcopy(targets[primitive]), "status": "calculated"}


def _moved_row(row: dict, x: float) -> dict:
    """Return a part's observation with its position along the transport axis at ``x``."""
    key = "observed_pose" if "observed_pose" in row else "current_pose"
    pose = deepcopy(row[key])
    if isinstance(pose, dict):
        pose["x"] = x
    else:
        pose[0] = x
    return {**row, key: pose, "source": {"kind": "projected", "contract_fingerprint": recovery_validation_fingerprint(pose)}}


def project_transport_completion(*, task: dict, trace: list, projected_snapshot: dict, part_context: dict,
                                 grounding_context: dict, targets: dict, resource_evidence: dict, **_unused) -> dict:
    """Project a checked transport program: where its move left each part.

    ``move_relative`` declares ``part_pose.x``, so each moved part's position follows the calculated target. The
    locations come from the calculation, which took them from the event, and the postcondition compares them. A
    program with no move leaves everything as it was, for the postcondition to refuse.
    """
    resource, part = deepcopy(projected_snapshot), deepcopy(part_context)
    moved = next((row["step_effects"]["part_pose.x"] for row in trace if "step_effects" in row), None)
    if moved is None:
        return {"feasibility_status": "FEASIBLE", "projected_snapshot": resource, "projected_part_context": part}
    resource.update(deepcopy(targets["compute_transport_targets"]["end_state"]))
    rows, projected_parts = deepcopy(resource_evidence.get("observed_parts", {})), {}
    for name, x in moved.items():
        rows[name] = _moved_row(rows[name], x)
        current = part if name == task.get("part_name") else grounding_context.get("parts", {}).get(name, {})
        updates = {"observed_pose": {**(current.get("observed_pose") or {}), "x": x}, "observed_pose_current": False,
                   "pose_source": "validated_composition_projection"}
        if name == task.get("part_name"):
            part.update(updates)
        else:
            projected_parts[name] = updates
    return {"feasibility_status": "FEASIBLE", "projected_snapshot": resource, "projected_part_context": part,
            "resource_evidence": {"observed_parts": rows}, "projected_parts": projected_parts,
            "scope": "Owner-calculated transport of the observed parts; a location is taken only where the event states it"}
