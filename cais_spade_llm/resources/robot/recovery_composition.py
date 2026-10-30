from __future__ import annotations

"""Robot-owned calculations and completion evidence for authored recovery programs."""

from collections.abc import Callable
from copy import deepcopy
import math

from cais_spade_llm.agents.shared_information.recovery_validation_protocol import recovery_validation_fingerprint


def preview_composition_output(*, primitive: str, params: dict, recovery_snapshot: dict,
                               part_context: dict, grounding_context: dict, task: dict,
                               resource_evidence: dict, physical_evidence: dict) -> dict:
    """Calculate target outputs from recorded owner facts, without legacy defaults."""
    from cais_spade_llm.resources.robot.recovery_capability import (
        calculate_recovery_pick_target, calculate_recovery_place_target,
    )

    part = params.get("part_name", task.get("part_name"))
    if primitive in {"compute_pick_targets", "compute_place_targets"}:
        if params.get("part_name") != task.get("part_name"):
            return {"status": "NEEDS_CONTEXT", "reason": "Target calculation requires the exact event part_name"}
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
        if primitive == "compute_place_targets" and params.get("destination_location") != calculated["destination_location"]:
            return {"status": "NEEDS_CONTEXT", "reason": "Bind destination_location explicitly to the intended modeled destination"}
        # Explicit geometric overrides cannot silently select another calculation.
        defaults = {"part_name": part}
        if primitive == "compute_pick_targets":
            defaults.update(target_pose=None, product_geometry=None, detected_parts=None,
                target_pose_source="", prefer_live_detection=False, approach_height_override_m=None,
                ignore_current_height_for_travel_z=False, min_pick_tcp_z_override_m=None,
                use_global_min_pick_tcp_z=resource_evidence.get("pick_policy", {}).get(part, {}).get("use_global_min_pick_tcp_z"),
                surface_clearance_override_m=resource_evidence.get("pick_policy", {}).get(part, {}).get("surface_clearance_override_m"),
                apply_pick_z_adjustments=True)
        else:
            # The author binds the recorded input. Owner preparation may add
            # placement outputs to its private _task_ctx before this preview.
            recorded = grounding_context.get("resources", {}).get(task["resource_jid"], {}).get(
                "resource_evidence", resource_evidence)
            defaults.update(pick_ctx=recorded.get("state", {}).get("_task_ctx"), product_geometry=None,
                destination_location=calculated["destination_location"], z_adjustment_m=0., assembly_board_v1_aruco=None)
        unsupported = [field for field, value in params.items()
                       if field not in defaults or value != defaults[field]]
        if unsupported:
            return {"status": "NEEDS_CONTEXT", "reason":
                "Explicit parameters differ from the supported owner calculation: "
                + ", ".join(unsupported) + ". Omit these parameters to use recorded owner inputs; "
                "copying observed geometry into them is an override."}
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
        output["model_name"] = resource_evidence.get("inputs", {}).get("geometry", {}).get("parts", {}).get("model_map", {}).get(part)
        travel_z = (output.get("travel_z") if primitive == "compute_pick_targets"
                    else resource_evidence.get("state", {}).get("_task_ctx", {}).get("travel_z"))
        if output.get("access_retreat_pose"):
            output["retreat_pose"] = deepcopy(output["access_retreat_pose"])
        elif type(travel_z) in (int, float) and math.isfinite(travel_z):
            output["retreat_pose"] = {**output["target_pose"], "z": travel_z}
        return {"event_fact_key": f"{key}.{part}", "output": output, "status": "calculated"}
    if primitive == "get_current_pose":
        return {"event_fact_key": "current_pose", "output": deepcopy(recovery_snapshot.get("current_pose")), "status": "calculated"}
    if primitive == "detect_parts":
        pose = part_context.get("observed_pose")
        if not isinstance(pose, dict):
            return {"status": "NEEDS_CONTEXT", "reason": "Recorded part observation is missing"}
        return {"event_fact_key": f"part_pose.{part}", "output": {"part_name": part, "pose": deepcopy(pose), **pose}, "status": "calculated"}
    return {}


def _source_records(part: str, acting_jid: str, grounding_context: dict, physical_evidence: dict) -> list[tuple]:
    """Return every value that records the part where it rested, with the value it takes once the part has left.

    As the RA's pick finds them (``ra_feasibility._Check.source_variables``): a resource-scope value that is the
    part, or a per-part variable for the part that is not null or false; it becomes null, or false where its
    declared values have no null.
    """
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
            if (declaration or {}).get("scope") == "part" or name not in values:
                continue
            value = values[name]
            if value == part or (name.endswith("." + part) and value is not None and value is not False):
                domain = (declaration or {}).get("domain") or []
                nullable = not domain or any(item is None for item in domain)
                records.append((jid, name, None if nullable or not any(item is False for item in domain) else False))
    return records


def hold_picked_part(resource_evidence: dict, *, task: dict, recovery_des_model: dict, recovery_snapshot: dict,
                     pick_target: dict | None) -> None:
    """Let a place calculation after the event's own grasp read the part as that grasp holds it.

    In one event that takes a part and puts it down, the place target needs the held part's grasp transform and
    pick context. They come from the event's own pick calculation, as the completion records them for the grasp;
    the completion then checks that the grasp reached its target and retreated.
    """
    part, target = task.get("part_name"), pick_target
    keys = ("x", "y", "z", "qx", "qy", "qz", "qw")
    if (not isinstance(target, dict) or recovery_snapshot.get("held_part") != part
            or (resource_evidence.get("attachment") or {}).get("part_name") == part
            or not all(key in target.get("target_pose", {}) and key in target.get("observed_pose", {}) for key in keys)):
        return
    event = next((row for row in recovery_des_model.get("events", []) if row.get("event_name") == "pick_grasp"), {})
    _hold(resource_evidence, part, target, target["target_pose"], event)


def _hold(evidence: dict, part_name: str, target: dict, pose: dict, event: dict) -> None:
    """Record the part as a grasp at ``pose`` holds it: its grasp transform and the pick context."""
    from cais_spade_llm.recovery_framework.geometry import multiply, rotate

    observed = target["observed_pose"]
    inverse = [-pose["qx"], -pose["qy"], -pose["qz"], pose["qw"]]
    transform = [*rotate(inverse, [observed[key] - pose[key] for key in ("x", "y", "z")]),
                 *multiply(inverse, [observed[key] for key in ("qx", "qy", "qz", "qw")])]
    evidence["attachment"] = {"part_name": part_name, "model_name": target["model_name"],
        "grasp_transform": transform, "projected_attachment_complete": True,
        "source": {"kind": "projected", "contract_fingerprint": recovery_validation_fingerprint(event)}}
    evidence.setdefault("state", {})["_task_ctx"] = deepcopy(target)


def _receiving_slot(part_name: str, destination: str, origin: dict, physical_evidence: dict) -> tuple[str, str] | None:
    """Find the resource and slot that take custody of a part placed on a supported place.

    The slot is the one whose custody puts the part at the destination (``product_state_from_custody``): the
    declared zone of the destination under the target, or a resource's slot for a place it owns, such as a
    machine's workholding or its staging tray.

    Returns:
        The owning resource and its slot variable, or None when no single slot receives the part.
    """
    from cais_spade_llm.resources.environment_models import product_state_from_custody

    models = physical_evidence.get("environment_models", {})
    zones = [zone for zone in models.get(destination, {}).get("assignments", {}).get("zones", [])
             if all(math.isclose(zone.get("pose", [float("inf")] * 3)[i], origin.get(axis, float("inf")),
                                 rel_tol=0., abs_tol=1e-9) for i, axis in enumerate(("x", "y")))]
    if zones:
        return (destination, f"zone_{zones[0]['zone']}_part") if len(zones) == 1 else None
    product = physical_evidence.get("product_name")
    slots = [(owner, field) for owner, model in models.items()
             for field, declaration in model.get("state_variables", {}).items()
             if declaration.get("reference") == "part_name" and product_state_from_custody(
                 part_name, owner, {"part_name": None, field: part_name}, product)["location"] == destination]
    return slots[0] if len(slots) == 1 else None


def project_composition_completion(*, task: dict, trace: list, recovery_des_model: dict,
                                   recovery_snapshot: dict, projected_snapshot: dict,
                                   part_context: dict, grounding_context: dict,
                                   primitive_catalog: list, resource_evidence: dict,
                                   physical_evidence: dict, locate_pose: Callable[[dict], str | None] | None = None) -> dict:
    """Retain proven primitive effects and require evidence for further completion.

    ``locate_pose`` names the place whose recorded area holds a pose, as the outline projection locates one.
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

    def set_field(field: str, value) -> None:
        resource[field] = deepcopy(value)
        alias = {"resource_state": "current_state", "resource_location": "current_location"}[field]
        if alias in resource:
            resource[alias] = deepcopy(value)
        if field == "resource_location" and isinstance(resource.get("occupancy"), dict):
            resource["occupancy"]["location"] = deepcopy(value)

    motion = {"move_to_named_pose", "move_cartesian", "move_pose", "move_relative", "move_joints", "rotate_joint", "move_insert"}
    last_motion = next((row for row in reversed(trace) if row["primitive"] in motion), None)
    transfers = [row for row in trace if row["primitive"] in {"grasp_part", "release_part"}]
    # One transfer, or the event's part taken and then put down, which Algorithm 1 checks as a grasp and a release.
    if [row["primitive"] for row in transfers] not in ([], ["grasp_part"], ["release_part"], ["grasp_part", "release_part"]):
        return fail("Only one custody transfer, or a grasp followed by a release, is supported in one event")
    retreat = None
    for number, transfer in enumerate(transfers):
        index, op = transfer["step_index"], transfer["primitive"]
        part_name = transfer["resolved_params"].get("part_name", task.get("part_name"))
        if part_name != task.get("part_name"):
            return fail("Custody transfer identifies a different part", "INFEASIBLE")
        compute = "compute_pick_targets" if op == "grasp_part" else "compute_place_targets"
        first = transfers[number - 1]["step_index"] + 1 if number else 0
        targets = [step for step in trace[first:index] if step["primitive"] == compute]
        if not targets:
            return fail("The custody transfer requires a preceding owner target calculation", "INFEASIBLE")
        target = targets[-1].get("preview_output", {})
        pose = transfer["start_snapshot"].get("current_pose", {})
        if not all(key in pose and key in target.get("target_pose", {})
                   and math.isclose(pose[key], target["target_pose"][key], rel_tol=0., abs_tol=1e-9)
                   for key in ("x", "y", "z", "qx", "qy", "qz", "qw")):
            return fail("The custody transfer must reach the calculated target pose and orientation", "INFEASIBLE")
        end = transfers[number + 1]["step_index"] if number + 1 < len(transfers) else len(trace)
        # The motion right after the transfer is its retreat; later motions may take the robot elsewhere.
        retreat = next((step for step in trace[index + 1:end] if step["primitive"] in motion), None)
        if retreat is None or retreat["primitive"] not in {"move_relative", "move_cartesian", "move_pose"}:
            return fail("Owner completion requires a retreat after the custody transfer", "INFEASIBLE")
        reached = retreat.get("projected_snapshot", {}).get("current_pose", {})
        retreat_pose = target.get("retreat_pose", {})
        if not retreat_pose:
            return fail("The owner completion retreat pose is missing from the recorded target inputs")
        if not all(key in reached and key in retreat_pose
                   and math.isclose(reached[key], retreat_pose[key], rel_tol=0., abs_tol=1e-9)
                   for key in ("x", "y", "z", "qx", "qy", "qz", "qw")):
            return fail("Finish the custody transfer at the calculated event_facts target retreat_pose", "INFEASIBLE")
        event = events.get("pick_grasp" if op == "grasp_part" else "place_insert", {})
        native_steps = event.get("composition_contract", {}).get("steps", [])
        if not any(step.get("op") == op for step in native_steps) or not any(
            step.get("op") == "move_relative" and step.get("params", {}).get("dx") == 0.
            and step.get("params", {}).get("dy") == 0.
            and step.get("params", {}).get("dz") == {"$sub": [
                {"$state": "_task_ctx", "path": ["travel_z"]}, {"$state": "_position", "path": ["z"]}]}
            for step in native_steps):
            return fail("The owner has no supported custody transfer and retreat completion contract")
        if op == "grasp_part":
            event = events.get("pick_grasp", {})
            # Since the aligned part model (October 23) the part's location is its only state field.
            if not apply_updates(event, {"part_name": part_name}, {"part_location"}):
                return fail("The owner has no supported pick completion effects")
            location = part_context.get("part_location", part_context.get("location"))
            if location is not None and not apply_updates(event, {"origin_resource_location": location},
                                                          {"resource_location"}):
                return fail("The owner has no supported pickup location effect")
            for jid, field, empty in _source_records(part_name, task["resource_jid"], grounding_context, physical_evidence):
                projected_resources.setdefault(jid, {})[field] = empty
            part["part_holder_resource_jid"] = task["resource_jid"]
            observed = target.get("observed_pose", {})
            if not all(key in observed for key in ("x", "y", "z", "qx", "qy", "qz", "qw")):
                return fail("The owner requires the complete part pose for projected attachment")
            _hold(evidence, part_name, target, pose, event)
        else:
            event = events.get("place_insert", {})
            destination = target.get("destination_location")
            if not apply_updates(event, {"destination_location": destination}, {"resource_state", "resource_location"}):
                return fail("The owner has no supported placement completion effects")
            part["part_holder_resource_jid"] = None
            origin = target.get("target_origin_pose", {})
            role = target.get("target_reference", {}).get("surface_role")
            if role == "support":
                from cais_spade_llm.resources.environment_models import product_state_from_custody

                receiving = _receiving_slot(part_name, destination, origin, physical_evidence)
                if receiving is None or not target.get("support_placement"):
                    return fail("Placement requires a unique declared receiving zone and supported target")
                owner, field = receiving
                model = physical_evidence.get("environment_models", {}).get(owner, {})
                declarations = model.get("state_variables", {})
                jid = physical_evidence.get("jids", {}).get(owner)
                values = deepcopy(physical_evidence.get("environment_valuation", {}).get(owner, {}))
                values.update({key: value for key, value in grounding_context.get("resources", {}).get(jid, {}).items() if key in declarations})
                values.update(projected_resources.get(jid, {}))
                if declarations.get(field, {}).get("reference") != "part_name" or field not in values:
                    return fail("The receiving occupancy contract is missing")
                if values[field] not in (None, part_name):
                    return fail("The receiving zone is occupied", "INFEASIBLE")
                values[field] = part_name
                product = product_state_from_custody(part_name, owner, values, physical_evidence.get("product_name"))
                part.update(part_location=product["location"])
                projected_resources[jid] = values
                completed.append({"source": "product_state_from_custody", "owner": owner,
                    "resource_values": values, "contract_fingerprint": recovery_validation_fingerprint(model)})
            elif role == "assembly_slot":
                slot = transfer["resolved_params"].get("assembly_slot")
                required = {key: target[key] for key in ("slot_x", "slot_y", "part_height", "board_top_z") if key in target}
                required.update(destination_location=destination, part_origin_z=target.get("place_part_origin_z"))
                if not isinstance(slot, dict) or any(slot.get(key) != value for key, value in required.items()):
                    return fail("Assembly completion requires release_part.assembly_slot bound to the calculated target", "INFEASIBLE")
                if not apply_updates(event, {"destination_location": destination}, {"part_location"}):
                    return fail("The owner has no supported assembly completion effects")
            else:
                return fail("The owner has no completion contract for this target surface")
            supported_pose = target.get("support_placement", {}).get("part_pose")
            if isinstance(supported_pose, list) and len(supported_pose) == 7:
                origin = dict(zip(("x", "y", "z", "qx", "qy", "qz", "qw"), supported_pose, strict=True))
            else:
                from cais_spade_llm.recovery_framework.geometry import multiply

                transform = evidence.get("attachment", {}).get("grasp_transform", [])
                if len(transform) == 7:
                    orientation = multiply([pose[key] for key in ("qx", "qy", "qz", "qw")], transform[3:])
                    origin = {**origin, **dict(zip(("qx", "qy", "qz", "qw"), orientation, strict=True))}
            part["observed_pose"] = deepcopy(origin)
            part["observed_pose_current"] = False
            part["pose_source"] = "validated_composition_projection"
            evidence.setdefault("observed_parts", {})[part_name] = {"current_pose": deepcopy(origin), "frame": "world",
                "source": {"kind": "projected", "contract_fingerprint": recovery_validation_fingerprint(event)}}
            evidence["attachment"] = {}
    if last_motion is not None and last_motion["primitive"] == "move_to_named_pose":
        # The declared move_home contract has one exact primitive and a fixed outcome; as the last motion, it
        # is where the robot ends, after any transfer.
        for event in recovery_des_model.get("events", []):
            steps = event.get("composition_contract", {}).get("steps") or []
            if (len(steps) != 1 or steps[0].get("op") != "move_to_named_pose" or steps[0].get("when")
                    or steps[0].get("params", {}).get("pose_name") != last_motion["resolved_params"].get("pose_name")):
                continue
            for field, effect in event.get("updates", {}).items():
                if set(effect) == {"set"} and field in {"resource_state", "resource_location"}:
                    set_field(field, effect["set"])
            completed.append({"event_name": event["event_name"], "step_index": last_motion["step_index"]})
    stated = task.get("expected_end_state") or {}
    final = resource.get("current_pose")
    moved = (last_motion is not None and last_motion["primitive"] != "move_to_named_pose"
             and (not transfers or last_motion is not retreat))
    start = recovery_snapshot.get("resource_location", recovery_snapshot.get("current_location"))
    if callable(locate_pose) and isinstance(final, dict):
        if "pose" in stated and "resource_location" not in stated:
            # As the outline projection locates a stated pose.
            set_field("resource_location", locate_pose(final))
        elif "resource_location" in stated and moved and (transfers or stated["resource_location"] != start):
            # A move ends at the stated place when one of its recorded areas holds the final pose; a location
            # the robot keeps holds, and a transfer's place stands unless the robot moves on after its retreat.
            set_field("resource_location", locate_pose(final, stated["resource_location"]))
    for field, aliases in {"part_state": ("state", "current_state"),
                           "part_location": ("location", "current_location"),
                           "part_holder_resource_jid": ("current_holder_resource_jid", "holder_resource_jid")}.items():
        if field in part:
            for alias in aliases:
                part[alias] = deepcopy(part[field])
    evidence["snapshot"] = deepcopy(resource)
    evidence.setdefault("state", {})["_held_part"] = resource.get("held_part")
    evidence["state"]["_position"] = deepcopy(resource.get("current_pose"))
    return {"feasibility_status": "FEASIBLE", "projected_snapshot": resource,
            "projected_part_context": part, "completed_contracts": completed,
            "resource_evidence": evidence, "projected_resources": projected_resources,
            "scope": "Declared primitive effects and owner-proven custody transfer, target, and retreat completion effects"}
