from __future__ import annotations

"""Owner endpoints and conservative predicate bounds, without motion replay."""

from copy import deepcopy
import math

from cais_spade_llm.agents.shared_information.recovery_validation_protocol import recovery_validation_fingerprint as fingerprint
from cais_spade_llm.resources.recovery_geometry import _checkpoint, _joint_geometry, _part_entries

POLICY_VERSION = 2
PROJECTION_KIND = "owner_endpoints_with_predicate_bounds"
OCCUPANCY_FIELDS = ("occupancy_possibilities", "part_occupancy_possibilities")


def endpoint_policy(physical: dict) -> bool:
    """Recognize new attempts without upgrading historical recorded evidence."""
    return physical.get("endpoint_evidence_policy_version") in (1, POLICY_VERSION)


def _local_checkpoint(physical: dict, rid: str, *, hold: dict | None = None) -> dict:
    snapshot, geometry = deepcopy(physical["snapshot"]), deepcopy(physical["geometry"])
    owner = snapshot["resources"][rid]
    snapshot["resources"] = {rid: owner}
    snapshot["parts"] = {name: row for name, row in snapshot["parts"].items() if name == owner.get("held_part")}
    geometry["resources"] = {rid: geometry["resources"][rid]}
    return _checkpoint(snapshot, geometry, physical["resource_models"], index=0,
                       peer_holds={rid: hold} if hold else None)


def _occupancy_binding(physical: dict) -> str:
    return fingerprint([physical["endpoint_evidence_policy_version"], physical["snapshot"], physical["geometry"],
                        physical["resource_models"], physical["resources"]])


def _bounded_checkpoint(physical: dict, *, index: float, holds: dict | None = None,
                        unknown: set[str] | None = None) -> dict:
    """Enclose the whole population, retaining explicit unknowns for missing facts."""
    unknown = unknown or set()
    point = deepcopy(physical["snapshot"])
    point.update(time=index, time_exact=str(index), phase="at", time_basis="event_order_not_elapsed_seconds")
    point["occupancy_possibilities"] = {region: {rid: [False, True] for rid in point["resources"]}
                                         for region in physical["geometry"]["regions"]}
    point["part_occupancy_possibilities"] = {region: {name: [False, True] for name in point["parts"]}
                                              for region in physical["geometry"]["regions"]}
    point["carried_parts"] = {rid: [row["held_part"]] if row.get("held_part") else []
                               for rid, row in point["resources"].items()}
    point["bound_gaps"] = []
    for rid in point["resources"]:
        if rid in unknown:
            point["bound_gaps"].append(rid + ": occupancy may be inside or outside each bound region")
            continue
        try:
            local = _local_checkpoint(physical, rid, hold=(holds or {}).get(rid))
        except (KeyError, ValueError, TypeError, IndexError) as exc:
            point["bound_gaps"].append(rid + ": " + str(exc))
            continue
        for field in OCCUPANCY_FIELDS:
            for region, values in local[field].items():
                point[field][region].update(values)
    # Deposited parts do not become robot occupancy. Their own predicates still
    # require a supported pose, and future persistence needs a separate hold.
    held = {row.get("held_part") for row in point["resources"].values()}
    for name, row in point["parts"].items():
        if name in held:
            continue
        support = physical.get("stationary_part_evidence", {}).get(name, {})
        if holds is not None and (support.get("status") != "supported" or not support.get("source")
                                  or support.get("current_pose") != row.get("current_pose")):
            continue
        try:
            local = _checkpoint({"frame": point["frame"], "resources": {}, "parts": {name: row}},
                {**physical["geometry"], "resources": {}}, {}, index=0)
            for region, values in local["part_occupancy_possibilities"].items():
                point["part_occupancy_possibilities"][region].update(values)
        except (KeyError, ValueError, TypeError, IndexError):
            point["bound_gaps"].append(name + ": supported part pose is unavailable")
    return point


def _kmr_endpoint(*, task: dict, snapshot: dict, model: dict, evidence: dict,
                  physical: dict, declared: dict, event: dict) -> tuple[dict, list]:
    from .kmr_outline_evidence import _calculation, _initial_state, _resolve_step
    from .kmr_recovery import check_kmr_primitive_capability, transport_pose
    from cais_spade_llm.resources.continuous_motion import representative_pose
    from cais_spade_llm.resources.environment_recovery import kmr_validation_evidence
    from .outline_evidence_policy import POSE_FIELDS

    endpoint = deepcopy(physical)
    if physical["endpoint_evidence_policy_version"] == POLICY_VERSION:
        jid = model["resource_jid"]
        row = endpoint["snapshot"]["resources"]["KMR"]
        if (jid != task["resource_jid"] or jid != physical["jids"]["KMR"]
                or row.get("resource_jid") not in (None, jid)):
            raise ValueError("Endpoint owner identity differs from its resource binding")
        # Physical captures may omit the JID carried by the owner's public
        # snapshot. The accepted endpoint must keep that validated identity.
        row["resource_jid"] = jid
    state = _initial_state(task, model, evidence, physical, snapshot, declared)
    outputs, records = {}, []
    for step in event["program"]["steps"]:
        resolved = _resolve_step(step, declared, state, outputs, evidence)
        if resolved["status"] == "skipped":
            records.append({"step": deepcopy(step), "status": "skipped"})
            continue
        op, params = step["op"], resolved["params"]
        row = endpoint["snapshot"]["resources"]["KMR"]
        configuration = endpoint["resource_models"]["KMR"]["configuration"]
        values = kmr_validation_evidence(evidence, row, endpoint)
        values["snapshot"] = deepcopy(row)
        values["continuous_motion_configuration"] = deepcopy(configuration)
        check = check_kmr_primitive_capability(primitive=op, params=params, task=task,
            recovery_snapshot=row, resource_evidence=values)
        if check["feasibility_status"] != "FEASIBLE":
            raise ValueError(check["reason"])
        output = None
        if op in {"compute_pick_targets", "compute_place_targets"}:
            output = _calculation(step, params, declared, endpoint, evidence)
            if not isinstance(output, dict):
                raise ValueError("Owner target calculation returned no supported output")
        elif op in {"grasp_part", "release_part"}:
            from cais_spade_llm.resources.resource_safety_preparation import _outline_custody_evidence
            from cais_spade_llm.resources.recovery_geometry import _configuration_after, _custody, projected_part_support

            part = declared["task"]["parameters"].get("part_name")
            if set(params) - ({"initial", "part_name"} if op == "grasp_part" else {"transform", "part_name"}):
                raise ValueError("KMR custody parameters have no recorded endpoint calculation")
            if part not in endpoint["snapshot"]["parts"] or params.get("part_name", part) != part:
                raise ValueError("KMR custody requires the exact declared part")
            part_row = endpoint["snapshot"]["parts"][part]
            if op == "grasp_part":
                if (params.get("initial") is not None
                        and math.dist(params["initial"][:3], part_row["current_pose"][:3]) > .01):
                    raise ValueError("KMR recorded part conflicts with declared grasp observation")
                if math.dist(row["current_pose"][:3], part_row["current_pose"][:3]) > .06:
                    raise ValueError("KMR grasp requires the part at the gripper")
            elif (params.get("transform") is not None
                    and fingerprint(params["transform"]) != fingerprint(row.get("grasp_transform"))):
                raise ValueError("KMR release transform conflicts with projected attachment")
            contracts = endpoint["resource_models"]["KMR"]["primitive_model"]["configuration"]["custody_by_part"]
            contract = contracts.get(part, contracts["default"])
            target_state = "closed" if op == "grasp_part" else "open"
            if not contract.get("source") or not contract["joint_positions"][target_state]:
                raise ValueError("Recorded custody endpoint configuration is unavailable")
            for name, value in contract["joint_positions"][target_state].items():
                low, high = contract["joint_bounds"][name]
                if not math.isfinite(value) or not low <= value <= high:
                    raise ValueError("Recorded custody endpoint exceeds its joint bounds")
            output = _outline_custody_evidence(primitive=op,
                params={"part_name": part, "model_name": part_row["model_name"]},
                snapshot=endpoint["snapshot"], resource_id="KMR",
                model=endpoint["resource_models"]["KMR"], contract=contract)
            output["contract_fingerprint"] = fingerprint(contract)
            if op == "release_part":
                support = projected_part_support(part_name=part, state=endpoint["snapshot"], evidence=endpoint)
                if support.get("status") != "supported":
                    raise ValueError("KMR release has no supported deposited-part rest: " + str(support.get("reason")))
                destination = declared["task"]["parameters"].get("destination_location")
                if destination not in endpoint["snapshot"]["resources"]:
                    raise ValueError("KMR release needs a declared destination owner")
                output["custody_effects"][0]["contained_by"] = destination
                endpoint.setdefault("stationary_part_evidence", {})[part] = support
                if part not in endpoint.setdefault("stationary_parts", []):
                    endpoint["stationary_parts"].append(part)
            else:
                endpoint.get("stationary_part_evidence", {}).pop(part, None)
                endpoint["stationary_parts"] = [name for name in endpoint.get("stationary_parts", []) if name != part]
            _configuration_after(endpoint["resource_models"], "KMR", output)
            _custody(endpoint["snapshot"], "KMR", output["custody_effects"], endpoint["geometry"])
            output.update(attached=op == "grasp_part", held_part=row["held_part"],
                          gripper_state=row["gripper_state"], grasp_transform=deepcopy(row.get("grasp_transform")),
                          physical_execution_verified=False, collision_or_path_validated=False)
            if op == "grasp_part":
                output["support_departure_status"] = "not_checked"
        elif op == "open_gripper":
            if row.get("gripper_state") != "open" or row.get("held_part") is not None:
                raise ValueError("Gripper change requires supported configuration and custody effects")
            from cais_spade_llm.resources.recovery_geometry import _configuration_after

            contracts = endpoint['resource_models']['KMR']['primitive_model']['configuration']['custody_by_part']
            contract = contracts.get(task.get('part_name'), contracts['default'])
            positions = contract['joint_positions']['open']
            after = deepcopy(configuration)
            if not positions or not contract.get('source'):
                raise ValueError('Recorded gripper endpoint configuration is unavailable')
            for name, value in positions.items():
                low, high = contract['joint_bounds'][name]
                if not math.isfinite(value) or not low <= value <= high:
                    raise ValueError('Recorded gripper endpoint exceeds its joint bounds')
            for joint in after['joints']:
                if joint['name'] in positions:
                    joint.pop('frozen_interval', None)
                    joint['frozen_position'] = positions[joint['name']]
                    if joint['name'] in after.get('fixed_joint_positions', {}):
                        after['fixed_joint_positions'][joint['name']] = joint['frozen_position']
            _configuration_after(endpoint['resource_models'], 'KMR', {'configuration_after': after})
            output = {'gripper_state': 'open', 'width': contract['gripper']['open'],
                      'configuration_after': deepcopy(after), 'source': contract['source']}
        elif op == "move_base":
            x, y, yaw = params["target_pose"]
            pose = [x, y, configuration["root_xyz"][2], 0., 0., math.sin(yaw / 2), math.cos(yaw / 2)]
            if any(abs(v) > 1e-9 for v in configuration["root_quaternion"][:2]):
                raise ValueError("Base endpoint requires the declared planar root configuration")
            row["base_pose"] = pose
            configuration.update(root_xyz=pose[:3], root_quaternion=pose[3:])
        elif op in {"move_joints", "rotate_joint"}:
            joints = deepcopy(params["joints"] if op == "move_joints" else row["joint_positions"])
            if op == "rotate_joint":
                joints[row["joint_names"].index(params["joint_name"])] += math.radians(params["delta_deg"])
            row["joint_positions"] = joints
        elif op in {"move_cartesian", "move_relative", "move_to_named_pose"}:
            if op == "move_to_named_pose":
                if params["pose_name"] == "home" and row.get("gripper_state") != "open":
                    raise ValueError("Home endpoint requires evidence of its gripper opening effect")
                target = transport_pose(evidence["inputs"]["scene"], row["base_pose"], row["current_pose"])
                check = check_kmr_primitive_capability(primitive="move_cartesian", params={"target": target},
                    task=task, recovery_snapshot=row, resource_evidence=values)
            final_joints = check.get("evidence", {}).get("joint_positions")
            if not final_joints:
                raise ValueError("Endpoint requires an owner-supported final joint configuration for " + step["id"])
            row["joint_positions"] = deepcopy(final_joints)
        else:
            raise ValueError("Endpoint effect calculation is unsupported for primitive " + op)
        _, matrix = _joint_geometry(row, endpoint["resource_models"]["KMR"])
        row["current_pose"] = representative_pose(matrix)
        for field in POSE_FIELDS - {"current_pose", "base_pose", "joint_positions", "grasp_transform"}:
            row.pop(field, None)
        row.update(current_pose_source="projected_state", observed_pose_current=False)
        local = _local_checkpoint(endpoint, "KMR")
        for part in local["parts"].values():
            for field in POSE_FIELDS - {"current_pose"}:
                part.pop(field, None)
            part.update(current_pose_source="projected_state", observed_pose_current=False)
        endpoint["snapshot"]["parts"].update(local["parts"])
        records.append({"step": deepcopy(step), "params": deepcopy(params), "status": "calculated",
                        "resource_snapshot": deepcopy(row), "capability": check,
                        **({"output": deepcopy(output)} if output is not None else {})})
        if step.get("store_as") and output is not None:
            outputs[step["store_as"]] = deepcopy(output)
    return endpoint, records


def _readiness(physical: dict, rid: str) -> dict:
    from cais_spade_llm.agents.central_controller.recovery_case_evidence import recorded_base_hold

    projected = physical.get("resources", {}).get(rid, {}).get("projected_endpoint")
    if projected is not None:
        if (projected.get("binding") != physical.get("provenance", {}).get("binding")
                or projected.get("state_fingerprint") != fingerprint(physical["snapshot"]["resources"][rid])
                or projected.get("configuration_fingerprint") != fingerprint(physical["resource_models"][rid]["configuration"])):
            raise ValueError("Projected starting configuration is stale")
        if physical["endpoint_evidence_policy_version"] == 1:
            return {"status": "supported", "kind": "conditional_successor", "source": deepcopy(projected)}
        return {"status": "missing", "kind": "conditional_successor", "source": deepcopy(projected),
                "reason": "Execution readiness is not established by a projected endpoint"}
    try:
        support = recorded_base_hold(resource_id=rid, physical_evidence=physical,
                                     model=physical["resource_models"][rid]["configuration"])
        return {"status": "supported", "kind": "captured_idle_base", "source": support}
    except (KeyError, ValueError, TypeError) as exc:
        return {"status": "missing", "reason": "Owner readiness: " + str(exc)}


def _arm_hold(physical: dict, rid: str) -> dict:
    from cais_spade_llm.resources.recovery_geometry import commanded_hold_evidence

    state = deepcopy(physical["snapshot"]["resources"][rid])
    row = deepcopy(physical["resources"][rid])
    # A base move does not command these arm joints, so retaining their endpoint
    # configuration still requires the arm controller's supported hold.
    state.pop("base_pose", None)
    row["controller_hold_activity"] = {"arm": row.get("controller_hold_observation", {})}
    return commanded_hold_evidence(resource_id=rid, state=state, model=physical["resource_models"][rid],
                                   physical=row, include_observation_roundoff=True)


def _robot_successors(*, task: dict, snapshot: dict, model: dict, evidence: dict,
                      physical: dict) -> tuple[list[dict], list[str]]:
    from cais_spade_llm.resources.environment_models import project_transition
    from cais_spade_llm.resources.recovery_feasibility import derive_primitive_union

    rid = evidence["resource_id"]
    support = derive_primitive_union(task=task, recovery_des_model=model, recovery_snapshot=snapshot)
    results, failures = {}, []
    for match in support["matching_transitions"]:
        name, bindings = match["event_name"], match["parameter_bindings"]
        if task.get("required_task_contract") not in (None, name):
            continue
        owner = next(row for row in model["events"] if row["event_name"] == name)
        for event in evidence["environment_models"][rid]["events"]:
            if event.get("function_name", event["event_name"]) != name:
                continue
            params = {}
            for field, declaration in event["parameter_bindings"].items():
                if "equals" in declaration:
                    value = declaration["equals"]
                    if field in bindings and fingerprint(bindings[field]) != fingerprint(value):
                        break
                    params[field] = deepcopy(value)
                elif field in bindings:
                    params[field] = deepcopy(bindings[field])
                elif field == "part_name" and task.get("part_name"):
                    params[field] = task["part_name"]
                else:
                    break
            else:
                selected = {"resource_id": rid, "event_id": event["event_id"],
                            "event_name": event["event_name"], "parameters": params}
                try:
                    after, products = project_transition(evidence["environment_models"],
                        physical["environment_valuation"], physical["products"], selected,
                        evidence["product_name"], evidence["requirements"],
                        defer_resource_guards=not task.get("required_task_contract"))
                    public_parts = {}
                    for field, effect in owner["updates"].items():
                        declaration = model["state_variables"][field]
                        if set(effect) == {"set"}:
                            value = deepcopy(effect["set"])
                        elif set(effect) == {"set_from_param"} and effect["set_from_param"] in bindings:
                            value = deepcopy(bindings[effect["set_from_param"]])
                        else:
                            raise ValueError("Robot successor has unresolved owner effects: " + field)
                        if declaration.get("scope", "resource") == "part":
                            transition = owner.get("composition_contract", {}).get("part_transition") or {}
                            if transition.get("completed", {}).get("location_template") == "{resource_jid}":
                                public_parts[field] = value
                            continue
                        if declaration.get("private"):
                            continue
                        if field in event["updates"] and fingerprint(after[rid].get(field)) != fingerprint(value):
                            raise ValueError("Recorded robot declarations disagree on " + field)
                        after[rid][field] = value
                    key = fingerprint([after, products, params, owner, event.get("program")])
                    source = {"declaration_fingerprint": fingerprint(event), "event_id": event["event_id"],
                        "composition_prerequisites": [{"resource_id": peer, "guards": deepcopy(local["guards"])}
                            for peer in event["participants"] for local in evidence["environment_models"][peer]["events"]
                            if local["event_id"] == event["event_id"]]}
                    if key in results:
                        results[key]["owner_declarations"].append(source)
                    else:
                        results[key] = {"task": selected, "environment_valuation": after,
                            "environment_products": products, "public_part_updates": public_parts,
                            "owner_declarations": [source], "kind": "projected",
                            "source_model_fingerprint": fingerprint(model),
                            "source_fingerprint": fingerprint(evidence)}
                except (KeyError, ValueError, TypeError) as exc:
                    failures.append(str(exc))
    return [results[key] for key in sorted(results)], failures


def project_bounded_successor(*, task: dict, snapshot: dict, model: dict,
                              physical_input: dict, requirements: dict) -> dict:
    """Calculate supported endpoints and predicate alternatives with no fallback.

    Successful action completion is a planning condition, never an observation
    of arrival or execution authorization. Unsupported operations retain gaps.
    """
    from cais_spade_llm.resources.environment_recovery import declared_environment_successors

    physical = physical_input.get("physical_evidence") or {}
    result = _bounded_result(physical)
    try:
        guard_failures = []
        evidence = physical_input.get("resource_evidence") or {}
        if physical["endpoint_evidence_policy_version"] == POLICY_VERSION and evidence.get("resource_type") == "robot":
            identities = [rid for rid, jid in physical["jids"].items() if jid == task["resource_jid"]]
            if len(identities) != 1:
                raise ValueError("Robot endpoint requires one recorded owner identity")
            evidence = {**evidence, "resource_id": identities[0], **{field: physical[field] for field in
                ("environment_models", "environment_valuation", "product_name", "requirements")},
                "environment_products": physical["products"]}
            physical_input = {**physical_input, "resource_evidence": evidence}
            candidates, failures = _robot_successors(task=task, snapshot=snapshot, model=model,
                evidence=evidence, physical=physical)
        else:
            candidates, failures = declared_environment_successors(task=task, recovery_snapshot=snapshot,
                recovery_des_model=model, physical_input=physical_input, guard_failures=guard_failures)
        candidates = {fingerprint(row): row for row in candidates}
        if (physical['endpoint_evidence_policy_version'] == POLICY_VERSION and not candidates
                and failures and len(guard_failures) == len(failures)):
            result.update(status='rejected', physical_capability_rejected=True,
                reason='Recorded owner conditions block every supported declaration',
                capability_result={'feasibility_status': 'INFEASIBLE', 'guard_failures': guard_failures},
                capability_source={'kind': 'recorded', 'model_fingerprint': fingerprint(model),
                    'snapshot_fingerprint': fingerprint(snapshot)})
            return result
        if not candidates or (len(candidates) != 1 and physical["endpoint_evidence_policy_version"] == 1):
            raise ValueError("Endpoint requires one supported owner declaration: " + "; ".join(failures))
        calculated = [_project_bounded_declaration(task=task, snapshot=snapshot, model=model,
            physical_input=physical_input, requirements=requirements, declared=deepcopy(candidates[key]))
            for key in sorted(candidates)]
        if len(calculated) == 1:
            return calculated[0]
        records = [{key: deepcopy(row[key]) for key in (
            "status", "unresolved", "owner_successor", "owner_program_fingerprint", "endpoint_evidence")
            if key in row} for row in calculated]
        result["owner_alternatives"] = records
        if all(row.get("owner_effect_rejected") is True for row in calculated):
            conflicts = {fingerprint(conflict): conflict for row in calculated
                         for conflict in row["owner_effect_result"]["conflicts"]}
            result.update(status="rejected", owner_effect_rejected=True,
                reason="Proposed end conditions conflict with every supported owner declaration",
                owner_effect_result={"feasibility_status": "INFEASIBLE", "source": "declared owner effects",
                                     "conflicts": [conflicts[key] for key in sorted(conflicts)]})
            return result
        if any(row["status"] != "projected" for row in calculated):
            raise ValueError("Supported owner declarations require additional endpoint evidence")
        # Compare every consequence, ignoring only the retained declaration and
        # program source identities. No outcome is selected before they agree.
        consequences = []
        for row in calculated:
            values = deepcopy(row)
            values.pop("owner_successor", None)
            values.pop("owner_program_fingerprint", None)
            endpoint = values.get("endpoint_evidence", {})
            endpoint.pop("owner_program_fingerprint", None)
            endpoint.pop("steps", None)
            values.get("successor_evidence", {}).get("provenance", {}).pop("contract_fingerprint", None)
            consequences.append(fingerprint(values))
        if len(set(consequences)) != 1:
            raise ValueError("Supported owner declarations establish different endpoints, effects, or predicate evidence")
        result = calculated[0]
        result["owner_alternatives"] = records
        result["owner_declaration_comparison"] = {
            "status": "equivalent", "consequences_fingerprint": consequences[0],
            "declaration_fingerprints": sorted(candidates), "execution_authorized": False}
    except (KeyError, ValueError, TypeError, IndexError, StopIteration) as exc:
        result["status"] = "needs_context"
        result["unresolved"].append(str(exc))
        result["context_requests"] = ["/physical_evidence", f"/resources/{task.get('resource_jid')}/resource_evidence"]
    return result


def _bounded_result(physical: dict) -> dict:
    return {"status": "needs_context", "projection_kind": PROJECTION_KIND,
        "endpoint_evidence_policy_version": physical["endpoint_evidence_policy_version"], "interval_complete": False,
        "observations": [], "unresolved": [], "requires_projection": False,
        "context_requests": [], "command_sent": False, "execution_authorized": False,
        "projection_reasons": ["Calculate owner endpoints; use supported bounds or explicit unknown occupancy",
                               "Full trajectory generation, sampling, and replay are disabled by this policy"]}


def _project_bounded_declaration(*, task: dict, snapshot: dict, model: dict,
                                physical_input: dict, requirements: dict, declared: dict) -> dict:
    from .kmr_outline_evidence import _program
    from .outline_evidence_policy import _movement_effects, POSE_FIELDS, PROCESS_KINDS, SPATIAL_KINDS
    from cais_spade_llm.resources.environment_recovery import (
        _project_machine_command, _project_machine_dwell,
    )
    from cais_spade_llm.resources.machine.machine_jobs import MACHINE_PRIMITIVES
    from cais_spade_llm.resources.recovery_geometry import commanded_hold_evidence

    physical = physical_input.get("physical_evidence") or {}
    evidence = physical_input.get("resource_evidence") or {}
    result = _bounded_result(physical)
    try:
        result["owner_successor"] = deepcopy(declared)
        rid = evidence["resource_id"]
        if task.get("required_task_contract") not in (None, declared["task"]["event_name"]):
            raise ValueError("Exact nominal task contract differs from the supported owner declaration")
        products = declared["environment_products"]
        part = task.get("part_name")
        part_updates = {field: products[part][key] for field, key in
            (("part_state", "state"), ("part_location", "location"), ("processCompleted", "processCompleted"))
            if part in products and key in products[part]}
        part_updates.update(declared.get("public_part_updates", {}))
        conflicts = []
        for field, value in task["expected_end_state"].items():
            supported = (part_updates if model.get("state_variables", {}).get(field, {}).get("scope") == "part"
                         else declared["environment_valuation"][rid])
            if field in supported and fingerprint(value) != fingerprint(supported[field]):
                conflicts.append({"field": field, "requested": deepcopy(value), "derived": deepcopy(supported[field])})
        if conflicts:
            result.update(status="rejected", owner_effect_rejected=True,
                reason="Proposed end conditions conflict with the supported owner declaration",
                owner_effect_result={"feasibility_status": "INFEASIBLE", "source": "declared owner effects",
                                     "field": conflicts[0]["field"], "proposed": deepcopy(conflicts[0]["requested"]),
                                     "derived": deepcopy(conflicts[0]["derived"]), "conflicts": conflicts})
            return result
        machine_part = (physical["endpoint_evidence_policy_version"] == POLICY_VERSION
                        and declared["task"]["event_name"] == "machine_part")
        if machine_part or declared["task"]["event_name"] in MACHINE_PRIMITIVES:
            projector = _project_machine_dwell if machine_part else _project_machine_command
            machine = projector(task=task, recovery_snapshot=snapshot, recovery_des_model=model,
                resource_evidence=evidence, physical=physical, declared=declared)
            return {**result, **machine, "projection_kind": PROJECTION_KIND}
        robot = evidence.get("resource_type") == "robot"
        if rid != "KMR" and not robot:
            raise ValueError("Owner endpoint calculation is unavailable for " + rid)
        event = (next(row for row in evidence["environment_models"][rid]["events"]
                      if row["event_id"] == declared["task"]["event_id"])
                 if robot else _program(declared, evidence, physical))
        if event.get("product_effects") != {"processCompleted": []}:
            raise ValueError("Owner completion effects require additional evidence")
        if robot:
            from cais_spade_llm.resources.robot.recovery_capability import project_robot_outline_endpoint

            endpoint, steps, contract = project_robot_outline_endpoint(task=task, recovery_snapshot=snapshot,
                recovery_des_model=model, resource_evidence=evidence, physical_evidence=physical)
        else:
            endpoint, steps = _kmr_endpoint(task=task, snapshot=snapshot, model=model, evidence=evidence,
                                           physical=physical, declared=declared, event=event)
        local = _local_checkpoint(endpoint, rid)
        result["endpoint_evidence"] = {"status": "supported", "condition": "successful_owner_action_completion",
            "binding": deepcopy(requirements["binding"]), "owner_program_fingerprint": fingerprint(event),
            "steps": steps, "resource_snapshot": deepcopy(endpoint["snapshot"]["resources"][rid]),
            "configuration": deepcopy(endpoint["resource_models"][rid]["configuration"]),
            "carried_parts": deepcopy(local["parts"]),
            "component_bounds": _joint_geometry(endpoint["snapshot"]["resources"][rid], endpoint["resource_models"][rid])[0],
            "occupancy_possibilities": deepcopy(local["occupancy_possibilities"])}
        effects = _movement_effects(task=task, snapshot=snapshot, model=model, physical_input=physical_input,
            requirements=requirements, declared=declared, event=event, calculated_endpoint=endpoint)
        if effects is None:
            raise ValueError("Owner successor does not establish the proposed state, custody, or completion effects")
        result.update({key: value for key, value in effects.items()
                       if key not in {"projection_kind", "projection_reasons", "observations"}})
        successor = result["successor_evidence"]
        readiness = _readiness(physical, rid)
        result["readiness_evidence"] = readiness
        base_endpoint_declared = any(row["status"] == "calculated" and row["step"]["op"] == "move_base"
                                     for row in steps)
        if robot or (physical["endpoint_evidence_policy_version"] == POLICY_VERSION
                and (base_endpoint_declared or readiness.get("kind") == "conditional_successor")):
            # Successful owner completion conditions the endpoint. A stopped
            # base is not needed to bound the mover's occupancy as unknown.
            readiness.update(scope="execution", blocks_planning=False, execution_authorized=False)
            result["projection_reasons"].append(
                "Base readiness is retained for execution; planning uses the declared base endpoint and predicate bounds")
        elif readiness["status"] != "supported":
            result["unresolved"].append(readiness["reason"])
        arm_hold = ({"status": "not_required", "reason": "Robot joints are calculated at the endpoint; intervening occupancy is unknown"}
                    if robot else _arm_hold(physical, rid))
        if not robot and arm_hold["status"] != "supported":
            result["unresolved"].append("Arm configuration hold: " + arm_hold["reason"])
        holds, unknown, hold_reasons = {}, {rid}, {}
        for peer, state in physical["snapshot"]["resources"].items():
            if peer == rid or physical["resources"][peer].get("model_static") is True:
                continue
            hold = commanded_hold_evidence(resource_id=peer, state=state, model=physical["resource_models"][peer],
                physical=physical["resources"][peer], robot_description=physical.get("endpoint_robot_description"),
                include_observation_roundoff=True)
            if hold["status"] == "supported":
                holds[peer] = hold
            else:
                unknown.add(peer)
                hold_reasons[peer] = hold["reason"]
        first = _bounded_checkpoint(physical, index=0)
        cached = physical.get("endpoint_occupancy")
        if cached is not None:
            if cached["binding"] != _occupancy_binding(physical):
                raise ValueError("Retained endpoint occupancy is stale")
            for field in OCCUPANCY_FIELDS:
                first[field] = deepcopy(cached[field])
        between = _bounded_checkpoint(physical, index=.5, holds=holds, unknown=unknown)
        custody_parts = {effect["part"] for step in steps for effect in step.get("output", {}).get("custody_effects", [])}
        for part in custody_parts:
            for values in between["part_occupancy_possibilities"].values():
                values[part] = [False, True]
        between["carried_parts"][rid] = sorted(set(between["carried_parts"][rid]) | custody_parts)
        between.update(phase="between", continuous_interval=["0", "1"])
        between["continuous_cells"] = [{"interval": ["0", "1"],
            **{field: deepcopy(between[field]) for field in OCCUPANCY_FIELDS}}]
        endpoint_holds = deepcopy(holds)
        if arm_hold["status"] == "supported" and all(step["op"] == "move_base" for step in event["program"]["steps"]):
            endpoint_holds[rid] = {**deepcopy(arm_hold),
                "model_fingerprint": fingerprint(endpoint["resource_models"][rid]["configuration"])}
        last = _bounded_checkpoint(endpoint, index=1, holds=endpoint_holds, unknown=unknown - {rid})
        observations = [first, between, last]
        _part_entries(observations)
        # Restore only recalculated geometry or an independently supported hold.
        # All old scene caches and unsupported observations stay historical.
        for peer in {rid, *holds}:
            source = endpoint if peer == rid else physical
            successor["snapshot"]["resources"][peer] = {
                **deepcopy(source["snapshot"]["resources"][peer]), **declared["environment_valuation"][peer]}
            successor["resource_models"][peer] = deepcopy(source["resource_models"][peer])
            if not robot or peer != rid:
                successor["resource_models"][peer].pop("event_projection_contracts", None)
            for field in POSE_FIELDS - {"configuration"}:
                successor["resource_models"][peer].pop(field, None)
            successor["geometry"]["resources"][peer] = deepcopy(source["geometry"]["resources"][peer])
            successor["invalidated_geometry"]["resources"].pop(peer, None)
            if peer != rid:
                successor["resources"][peer] = deepcopy(source["resources"][peer])
            for name, row in _local_checkpoint(source, peer)["parts"].items():
                successor["snapshot"]["parts"][name] = deepcopy(row)
                successor["invalidated_geometry"]["parts"].pop(name, None)
        for part in custody_parts:
            successor["snapshot"]["parts"][part] = {
                **deepcopy(endpoint["snapshot"]["parts"][part]), **deepcopy(products[part])}
            successor["invalidated_geometry"]["parts"].pop(part, None)
            support = endpoint.get("stationary_part_evidence", {}).get(part)
            if support is None:
                successor.get("stationary_part_evidence", {}).pop(part, None)
                successor["stationary_parts"] = [name for name in successor.get("stationary_parts", []) if name != part]
            else:
                successor.setdefault("stationary_part_evidence", {})[part] = deepcopy(support)
                if part not in successor.setdefault("stationary_parts", []):
                    successor["stationary_parts"].append(part)
        for peer, state in endpoint["snapshot"]["resources"].items():
            if "contained_parts" in state:
                successor["snapshot"]["resources"][peer]["contained_parts"] = deepcopy(state["contained_parts"])
        owner = successor["snapshot"]["resources"][rid]
        boxes, _ = _joint_geometry(owner, successor["resource_models"][rid])
        geometry_row = successor["geometry"]["resources"][rid]
        for field in POSE_FIELDS | {"footprint"}:
            geometry_row.pop(field, None)
        geometry_row["component_bounds"] = [{"id": box["id"],
            "bounds": [[low[0], high[1]] for low, high in zip(box["minimum"], box["maximum"], strict=True)]}
            for box in boxes]
        geometry_row["source"] = {"kind": "projected", "binding": deepcopy(successor["provenance"]["binding"])}
        successor["resources"][rid] = {"model_static": False, "observation_state": deepcopy(owner),
            **{field: deepcopy(physical["resources"][rid][field]) for field in
               ("stationary_contract", "controller_hold_model", "controller_hold_observation")
               if field in physical["resources"][rid]},
            "projected_controller_hold": {"kind": "projected", "binding": deepcopy(successor["provenance"]["binding"]),
                "joint_names": deepcopy(owner["joint_names"]), "positions": deepcopy(owner["joint_positions"])},
            "projected_endpoint": {"binding": deepcopy(successor["provenance"]["binding"]),
                "state_fingerprint": fingerprint(owner),
                "configuration_fingerprint": fingerprint(successor["resource_models"][rid]["configuration"]),
                "condition": "successful_owner_action_completion", "physical_execution_verified": False}}
        if robot and endpoint["resources"][rid].get("validated_pickup_target_calculation"):
            successor["resources"][rid]["validated_pickup_target_calculation"] = deepcopy(
                endpoint["resources"][rid]["validated_pickup_target_calculation"])
        successor["endpoint_occupancy"] = {"binding": _occupancy_binding(successor),
                                            **{field: deepcopy(last[field]) for field in OCCUPANCY_FIELDS}}
        result.update(observations=observations, projected_resource_snapshot=deepcopy(owner),
            projected_snapshot=deepcopy(successor["snapshot"]), geometry=deepcopy(successor["geometry"]),
            resource_models=deepcopy(successor["resource_models"]),
            interval_evidence={"kind": "predicate_bounds", "time_basis": "event_order_not_elapsed_seconds",
                "peer_holds": holds, "arm_hold": arm_hold, "unknown_resources": sorted(unknown), "missing_peer_bounds": hold_reasons,
                "physical_execution_verified": False}, requires_projection=False)
        if physical["endpoint_evidence_policy_version"] == POLICY_VERSION:
            from cais_spade_llm.agents.central_controller.outline_evidence_safety import _owner_state_declarations

            # Participant declarations are needed even when the selected APs
            # require no task/state ledgers. This does not claim those ledgers.
            result["owner_safety_evidence"] = {"resource_models": _owner_state_declarations(
                evidence["environment_models"], physical["geometry"]["parts"])}
        unsupported = sorted({str(row["kind"]) for row in requirements["aps"]
                              if row["kind"] not in PROCESS_KINDS | SPATIAL_KINDS})
        if unsupported:
            result["unresolved"].append("Supported owner bounds are unavailable for predicates: " + ", ".join(unsupported))
        result["unresolved"].extend(requirements["unresolved"])
        result["status"] = "needs_context" if result["unresolved"] else "projected"
    except (KeyError, ValueError, TypeError, IndexError, StopIteration) as exc:
        result["status"] = "needs_context"
        result["unresolved"].append(str(exc))
    if result["unresolved"]:
        result["context_requests"] = ["/physical_evidence", f"/resources/{task.get('resource_jid')}/resource_evidence"]
    return result
