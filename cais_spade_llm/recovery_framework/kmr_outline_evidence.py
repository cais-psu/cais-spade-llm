from __future__ import annotations

"""KMR's declared task previews over an isolated, recorded physical scene.

The task declaration supplies its program. Unordered RA primitive support does
not supply an order. Every replay resolves the declaration again and checks each
native result against its exact private starting scene.
"""

from copy import deepcopy
from fractions import Fraction
import hashlib
import math
import xml.etree.ElementTree as ET

from cais_spade_llm.agents.shared_information.recovery_validation_protocol import recovery_validation_fingerprint as fingerprint


def _missing(reason: str) -> dict:
    return {"status": "NEEDS_CONTEXT", "reason": reason, "unresolved": [reason],
            "command_sent": False, "execution_authorized": False,
            "interval_complete": False, "requires_projection": True}


def _declarations(task: dict, model: dict, evidence: dict, physical: dict,
                  failures: list[dict] | None = None) -> list[dict]:
    from cais_spade_llm.recovery_framework.kmr_recovery import coordinate_parameters
    from cais_spade_llm.resources.environment_models import project_transition
    from cais_spade_llm.resources.recovery_feasibility import derive_primitive_union

    candidates, seen = [], set()
    snapshot = physical.get("environment_valuation", evidence.get("environment_valuation", {})).get("KMR", {})
    for row in derive_primitive_union(task=task, recovery_des_model=model, recovery_snapshot=snapshot)["matching_transitions"]:
        contract = row["composition_contract"]
        identity = (contract["event_id"], fingerprint(row["parameter_bindings"]))
        if identity in seen:
            continue
        seen.add(identity)
        if task.get("required_task_contract") not in (None, row["event_name"]):
            continue
        declared = {"resource_id": "KMR", "event_id": contract["event_id"],
                    "event_name": row["event_name"], "parameters": deepcopy(row["parameter_bindings"])}
        try:
            if declared['event_name'] == 'move_to_location':
                declared['parameters'], _ = coordinate_parameters(declared['parameters'], evidence['inputs']['scene'])
            after, products = project_transition(evidence["environment_models"],
                physical.get("environment_valuation", evidence["environment_valuation"]),
                physical.get("products", evidence["environment_products"]), declared,
                evidence["product_name"], evidence["requirements"])
        except (KeyError, ValueError, TypeError) as exc:
            if failures is not None:
                failures.append({"reason": str(exc), "declared_task": declared,
                                 "stage": "owner declaration projection"})
            continue
        candidates.append({"task": declared, "environment_valuation": after,
                           "environment_products": products})
    return candidates


def _program(declared: dict, evidence: dict, physical: dict) -> dict:
    rows = [row for row in evidence["environment_models"]["KMR"]["events"]
            if row["event_id"] == declared["task"]["event_id"]]
    if len(rows) != 1 or not rows[0].get("program", {}).get("steps"):
        raise ValueError("KMR event needs one recorded owner program")
    event = deepcopy(rows[0])
    owner = physical['snapshot']['resources']['KMR']
    if 'held_part' not in owner:
        raise ValueError('KMR owner program selection requires recorded held_part')
    held = owner['held_part']
    variant = ''
    if event['event_name'] == 'move_to_resource' and held is None:
        variant = 'empty_return' if declared['task']['parameters']['target_resource'] == 'Storage' else 'empty'
    elif event['event_name'] == 'move_to_location' and held is not None:
        variant = 'carrying'
    if variant:
        program = event.get('program_variants', {}).get(variant)
        if not isinstance(program, dict) or not program.get('steps'):
            raise ValueError('KMR owner program variant was not captured: ' + variant)
        event['program'] = deepcopy(program)
    event['selected_program_variant'] = variant
    return event


def _base_pose(physical: dict) -> list:
    model = physical["resource_models"]["KMR"]["configuration"]
    pose = [*model["root_xyz"], *model["root_quaternion"]]
    recorded = physical["snapshot"]["resources"]["KMR"].get("base_pose")
    if recorded is not None and (len(recorded) != 7 or math.dist(recorded[:3], pose[:3]) > 1e-9
            or abs(sum(a*b for a, b in zip(recorded[3:], pose[3:], strict=True))) < 1-1e-9):
        raise ValueError("KMR base observation and recorded model root disagree")
    return pose


def _initial_state(task: dict, model: dict, evidence: dict, physical: dict, snapshot: dict,
                   declared: dict) -> dict:
    from cais_spade_llm.recovery_framework.kmr_recovery import prepare_kmr_primitive_parameters

    values = deepcopy(evidence)
    values["snapshot"] = {**deepcopy(snapshot), **deepcopy(physical["snapshot"]["resources"]["KMR"]),
                          "base_pose": _base_pose(physical)}
    values["observed_parts"] = {part: {"current_pose": deepcopy(row["current_pose"])}
                                for part, row in physical["snapshot"]["parts"].items() if row.get("current_pose") is not None}
    calculated = prepare_kmr_primitive_parameters(task=task, recovery_des_model=model,
        recovery_snapshot=values["snapshot"], resource_evidence=values, resource_jid=task["resource_jid"])
    parameter_names = _program(declared, evidence, physical).get("parameter_bindings", {})
    matches = []
    for row in calculated["parameter_calculations"]:
        parameters = deepcopy(row.get("resolved_parameter_bindings", row["parameter_bindings"]))
        # A proposal can bind the carried part without adding an argument to a
        # movement declaration. Keep that context in the calculation audit.
        if ("part_name" not in parameter_names and "part_name" not in declared["task"]["parameters"]
                and parameters.get("part_name") == task.get("part_name")):
            parameters.pop("part_name", None)
        if (row["event_name"] == declared["task"]["event_name"]
                and fingerprint(parameters) == fingerprint(declared["task"]["parameters"])):
            matches.append(row)
    conflicts = [row for row in calculated["conflicts"]
                 if row.get("event_name") == declared["task"]["event_name"]
                 and any(fingerprint(row.get("parameter_bindings")) == fingerprint(match['parameter_bindings'])
                         for match in matches)]
    if conflicts:
        raise ValueError("KMR calculated parameters conflict with recorded task facts: " + str(conflicts))
    if len(matches) != 1:
        raise ValueError("KMR task parameter calculation has no unique recorded contributor")
    return deepcopy(matches[0]["calculated_state"])


def _resolve_step(step: dict, declared: dict, state: dict, outputs: dict, evidence: dict) -> dict:
    from cais_spade_llm.resources.robot.recovery_task_projection import resolve_nominal_task_step

    resolved = resolve_nominal_task_step(step=step, bindings=declared["task"]["parameters"],
        state=state, resource_evidence=evidence, step_outputs=outputs)
    if resolved["status"] not in {"resolved", "skipped"}:
        raise ValueError("KMR declared step " + step["id"] + ": " + str(resolved))
    return resolved


def _calculation(step: dict, params: dict, declared: dict, physical: dict, evidence: dict) -> dict | None:
    from cais_spade_llm.recovery_framework.kmr_recovery import pick_targets, place_targets, recorded_location

    owner, scene = physical["snapshot"]["resources"]["KMR"], evidence["inputs"]["scene"]
    bindings = declared["task"]["parameters"]
    if step["op"] == "compute_pick_targets":
        part = bindings["part_name"]
        return pick_targets(scene, part, physical["snapshot"]["parts"][part]["current_pose"],
            owner["current_pose"], _base_pose(physical), initial=params.get("initial"),
            location=bindings['origin_resource_location'])
    if step["op"] == "compute_place_targets":
        machine, _ = recorded_location(scene, bindings["destination_location"])
        return place_targets(scene, machine, owner["grasp_transform"], owner["current_pose"],
                            _base_pose(physical), machine.get("workholding_pose"), part_name=bindings['part_name'])
    return None


def _motion_request(step: dict, params: dict, physical: dict, evidence: dict) -> dict | None:
    from cais_spade_llm.recovery_framework.kmr_recovery import transport_pose, transfer_waypoints

    owner = physical["snapshot"]["resources"]["KMR"]
    op, scene = step["op"], evidence["inputs"]["scene"]
    if op == "move_to_named_pose":
        if params["pose_name"] not in {"transport", "home"}:
            raise ValueError("KMR named pose has no declared owner calculation")
        if params["pose_name"] == "home" and owner["held_part"] is not None:
            raise ValueError("KMR home declaration requires an empty gripper")
        target = transport_pose(scene, _base_pose(physical), owner["current_pose"])
        points = ([*transfer_waypoints(scene, _base_pose(physical), owner["current_pose"], target), target]
                  if params["pose_name"] == "transport" else [target])
    elif op == "move_relative":
        delta = params["offset"]
        if len(delta) != 3 or any(type(v) not in (int, float) or not math.isfinite(v) for v in delta):
            raise ValueError("KMR relative movement requires three finite offsets")
        points = [[*[a+b for a, b in zip(owner["current_pose"][:3], delta, strict=True)], *owner["current_pose"][3:]]]
    elif op == "move_cartesian":
        points = [*deepcopy(params.get("waypoints") or []), deepcopy(params["target"])]
        # The current owner's move_to_pose retains this argument but does not
        # replace observed_state with it. The exact argument stays in the step
        # record; the private preview starts from the captured configuration.
        seed = params.get("seed")
        if seed is not None and (not isinstance(seed, list) or len(seed) != len(owner["joint_names"])
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in seed)):
            raise ValueError("KMR Cartesian seed requires the exact seven-joint format")
        if not params.get("waypoints"):
            from cais_spade_llm.resources.robot.cartesian_waypoints import fixed_orientation_waypoints
            from cais_spade_llm.recovery_framework.geometry import compose

            settings = scene["KMR"]["task_execution"]["cartesian_waypoints"]
            middle = settings.get("transfer_orientation_waypoint_xyzw_in_base_frame")
            if middle is not None:
                middle = compose(_base_pose(physical), [0., 0., 0., *middle])[3:]
            points = fixed_orientation_waypoints(owner["current_pose"], params["target"], [],
                rotation_point=transport_pose(scene, _base_pose(physical), owner["current_pose"])[:3],
                angular_step=settings.get("angular_step_rad", .05), rotation_midpoint=middle) or [deepcopy(params["target"])]
    elif op in {"move_joints", "rotate_joint"}:
        positions = deepcopy(params["joints"] if op == "move_joints" else owner["joint_positions"])
        if op == "rotate_joint":
            delta = params["delta_deg"]
            if type(delta) not in (int, float) or not math.isfinite(delta):
                raise ValueError("KMR relative joint movement requires finite delta_deg")
            positions[owner["joint_names"].index(params["joint_name"])] += math.radians(delta)
        if len(positions) != len(owner["joint_names"]):
            raise ValueError("KMR joint target must preserve every recorded joint")
        return {"mode": "joint_target", "joint_target": {"name": deepcopy(owner["joint_names"]), "position": positions}}
    elif op == "move_base":
        raise ValueError("KMR base movement needs recorded continuous base-controller and carried-part motion evidence")
    else:
        return None
    if any(not isinstance(pose, list) or len(pose) != 7
           or any(type(v) not in (int, float) or not math.isfinite(v) for v in pose)
           or not math.isclose(sum(v*v for v in pose[3:]), 1., abs_tol=1e-9) for pose in points):
        raise ValueError("KMR Cartesian preview requires exact finite xyz/xyzw targets")
    return {"mode": "cartesian", "waypoints": [{"position": dict(zip(("x", "y", "z"), p[:3], strict=True)),
              "orientation": dict(zip(("x", "y", "z", "w"), p[3:], strict=True))} for p in points]}


def _holds(physical: dict, *, robot_description: str | None = None) -> dict:
    from cais_spade_llm.resources.recovery_geometry import commanded_hold_evidence

    if robot_description is not None:
        from cais_spade_llm.agents.central_controller.recovery_case_evidence import recorded_base_hold

        # Scoped peer holds cannot hide the moving owner's own readiness check.
        recorded_base_hold(resource_id="KMR", physical_evidence=physical,
                           model=physical["resource_models"]["KMR"]["configuration"])
        if (physical["snapshot"]["resources"]["KMR"].get("base_pose")
                != physical["resources"]["KMR"]["base_hold_evidence"]["root_pose"]):
            raise ValueError("KMR base readiness differs from the projected starting pose")
    holds = {}
    for rid, state in physical["snapshot"]["resources"].items():
        if rid == "KMR" or physical["resources"][rid].get("model_static") is True:
            continue
        row = physical["resources"][rid]
        if "controller_hold_model" in row or row.get("idle") is not True:
            result = commanded_hold_evidence(resource_id=rid, state=state,
                model=physical["resource_models"][rid], physical=row, robot_description=robot_description)
            if result.get("status") != "supported":
                raise ValueError("KMR interval lacks supported peer hold: " + rid + ": " + result.get("reason", ""))
            holds[rid] = result
        elif row.get("stationary_contract", {}).get("requires_no_active_goals") is not True:
            raise ValueError("KMR interval lacks stationary peer contract: " + rid)
    free = set(physical["snapshot"]["parts"]) - {r.get("held_part") for r in physical["snapshot"]["resources"].values()}
    for part in free:
        support = physical.get("stationary_part_evidence", {}).get(part, {})
        if (support.get("status") != "supported" or support.get("current_pose") != physical["snapshot"]["parts"][part].get("current_pose")):
            raise ValueError("KMR interval lacks supported stationary part evidence: " + part)
    return holds


def _project_base_state(scene_capture: dict, physical: dict, projected: dict) -> None:
    """Set the private planning base joints from the bound physical root pose.

    The planning scene's last joint-state publication can predate the frozen
    physical checkpoint. Its mobile root must not override that checkpoint.
    Only the owner's declared world x/y/yaw chain is supported here.
    """
    configuration = physical["resource_models"]["KMR"]["configuration"]
    description = scene_capture["robot_description"]
    if (not isinstance(description, str)
            or hashlib.sha256(description.encode()).hexdigest() != configuration["description_sha256"]):
        raise ValueError("KMR base projection differs from its recorded planning description")
    pose = _base_pose(physical)
    if (any(type(value) not in (int, float) or not math.isfinite(value) for value in pose)
            or any(abs(value) > 1e-12 for value in pose[2:5])
            or not math.isclose(pose[5] ** 2 + pose[6] ** 2, 1., abs_tol=1e-12, rel_tol=0.)):
        raise ValueError("KMR base projection requires a finite planar world pose")
    try:
        joints = ET.fromstring(description).findall("joint")
    except ET.ParseError as exc:
        raise ValueError("KMR base projection has an invalid recorded planning description") from exc
    parents = {joint.find("child").get("link"): joint for joint in joints if joint.find("child") is not None}
    if len(parents) != len(joints):
        raise ValueError("KMR base projection requires an unambiguous parent chain")
    child, chain = configuration["root"], []
    for name, kind, axis in reversed((
        ("KMR_base_x_joint", "prismatic", [1., 0., 0.]),
        ("KMR_base_y_joint", "prismatic", [0., 1., 0.]),
        ("KMR_base_yaw_joint", "continuous", [0., 0., 1.]),
    )):
        joint = parents.get(child)
        if (joint is None or joint.get("name") != name or joint.get("type") != kind
                or joint.find("parent") is None or joint.find("axis") is None or joint.find("mimic") is not None
                or [float(value) for value in joint.find("axis").get("xyz", "").split()] != axis):
            raise ValueError("KMR base projection lacks its declared world x/y/yaw joints")
        origin = joint.find("origin")
        if origin is not None and any([float(v) for v in origin.get(field, "0 0 0").split()] != [0., 0., 0.]
                                      for field in ("xyz", "rpy")):
            raise ValueError("KMR base projection has an unsupported joint origin")
        chain.append(name)
        child = joint.find("parent").get("link")
    if child != physical["snapshot"]["frame"] or child != "world":
        raise ValueError("KMR base projection requires the recorded world frame")
    values = dict(zip(reversed(chain), (pose[0], pose[1], math.atan2(2 * pose[6] * pose[5],
                                                                  1 - 2 * pose[5] ** 2)), strict=True))
    for state in (projected["start_state"], projected["planning_scene"]["robot_state"]):
        names, positions = state["joint_state"]["name"], state["joint_state"]["position"]
        if len(names) != len(set(names)) or len(names) != len(positions) or not set(values).issubset(names):
            raise ValueError("KMR base projection lacks complete private-scene joint identities")
        for name, value in values.items():
            positions[names.index(name)] = value


def _scene_request(task: dict, physical: dict, evidence: dict, scene_capture: dict, motion: dict,
                   *, project_base: bool = False) -> dict:
    from cais_spade_llm.recovery_framework.recovery_motion_preview import (
        apply_declared_mounting_contacts, prepare_recorded_root_transforms, preview_scene_fingerprint,
        project_preview_scene,
    )

    configured = evidence["inputs"]["scene"]
    settings, kmr = configured["safety_preparation"]["outline_preview"], configured["KMR"]["task_execution"]
    projected = project_preview_scene(scene_capture, physical)
    if project_base:
        _project_base_state(scene_capture, physical, projected)
    if "planning_scene_mounting_contacts" in configured:
        mounted = apply_declared_mounting_contacts(projected["planning_scene"],
            robot_description=scene_capture["robot_description"], robots=configured["robots"],
            physical_evidence=physical, declarations=configured["planning_scene_mounting_contacts"])
        projected["planning_scene"] = mounted["planning_scene"]
    roots = prepare_recorded_root_transforms(scene_capture, physical)
    request = {"planning_scene": projected["planning_scene"], "start_state": projected["start_state"],
        "scene_fingerprint": preview_scene_fingerprint(projected["planning_scene"]),
        "model_fingerprint": scene_capture["model_fingerprint"], "resource_id": "KMR",
        "capture_fingerprint": fingerprint(physical), "proposal_fingerprint": fingerprint(task),
        "group_name": kmr["planning_group"], "link_name": kmr["tcp_link"],
        "frame_id": physical["snapshot"]["frame"],
        **{key: settings[key] for key in ("max_translation_step", "max_rotation_step", "max_joint_step", "timeout_seconds")},
        **{key: kmr[key] for key in ("velocity_scaling", "acceleration_scaling")}, **deepcopy(motion)}
    if roots["recorded_root_transforms"]:
        request["recorded_root_transforms"] = roots["recorded_root_transforms"]
    if physical.get("kmr_pending_support_departure"):
        from cais_spade_llm.recovery_framework.recovery_motion_preview import bind_support_departure
        from cais_spade_llm.resources.recovery_support_departure import prepare_support_departure

        support = prepare_support_departure(resource_id="KMR", state=physical["snapshot"], evidence=physical)
        if support.get("status") != "prepared":
            raise ValueError(support.get("reason") or "KMR supporting contact lacks departure evidence")
        request = bind_support_departure(request, support)
    return request


def _apply_motion(task: dict, physical: dict, request: dict, response: dict, time: float,
                  *, robot_description: str | None = None) -> tuple[dict, list, dict]:
    from cais_spade_llm.recovery_framework.recovery_motion_preview import preview_result_record, validate_preview_request
    from cais_spade_llm.resources.recovery_geometry import _checkpoint, _motion_checkpoint, _prepared_motion

    request = validate_preview_request(request)
    if fingerprint(response.get("request")) != fingerprint(request) or response.get("source") != "PreviewRecoveryMotion":
        raise ValueError("KMR native response differs from its issued private request")
    checked = preview_result_record(request, response)
    if checked.get("status") != "PREVIEW_COMPLETE" or checked.get("sampled_collision_rejected") is True:
        raise ValueError(checked.get("reason") or "KMR native path is incomplete")
    if fingerprint(checked.get("joint_trajectory")) != fingerprint(response.get("joint_trajectory")):
        raise ValueError("KMR native trajectory differs from its recorded response")
    trajectory = deepcopy(checked["joint_trajectory"])
    stamp = trajectory["points"][-1]["time_from_start"]
    trajectory["duration_ns"] = stamp["sec"]*1_000_000_000 + stamp["nanosec"]
    snapshot, models, geometry = physical["snapshot"], physical["resource_models"], physical["geometry"]
    holds = _holds(physical, robot_description=robot_description)
    free = sorted(set(snapshot["parts"]) - {r.get("held_part") for r in snapshot["resources"].values()})
    interval = {"joint_trajectory": trajectory, "stationary_resources": sorted(set(snapshot["resources"])-{"KMR"}),
        "stationary_parts": free, "stationary_resource_evidence": holds,
        "native_request_fingerprint": fingerprint(request), "native_response_fingerprint": fingerprint(response)}
    motion = _prepared_motion(snapshot, "KMR", models, interval)
    if request["mode"] == "joint_target":
        target = request["joint_target"]
        if target["name"] != motion.names or target["position"] != motion.points[-1]["positions"]:
            raise ValueError("KMR native endpoint differs from its declared joint target")
    else:
        from cais_spade_llm.resources.recovery_geometry import _cartesian_endpoint_residual, owner_cartesian_endpoint_precision

        last = request["waypoints"][-1]
        target = [*(last["position"][a] for a in ("x", "y", "z")), *(last["orientation"][a] for a in ("x", "y", "z", "w"))]
        residual = _cartesian_endpoint_residual(motion.reference_pose(motion.duration), target)
        precision = owner_cartesian_endpoint_precision()
        if (residual["position_residual"] > precision["position_precision"]
                or residual["orientation_residual"] > precision["orientation_precision"]):
            raise ValueError("KMR native endpoint does not establish its owner-calculated Cartesian target")
    if physical.get("kmr_pending_support_departure"):
        import json
        from cais_spade_llm.resources.recovery_support_departure import verify_support_departure

        support = json.loads(request["support_departure_json"])
        verified = verify_support_departure(resource_id="KMR", state=snapshot,
            evidence=physical, joint_trajectory=trajectory, support_departure=support)
        if verified.get("status") != "supported":
            raise ValueError(verified.get("reason") or "KMR continuous support departure remains unresolved")
        interval["support_departure_evidence"] = verified
    during = _motion_checkpoint(snapshot, geometry, models, index=time,
        moving=("KMR", motion, 0., motion.duration), peer_holds=holds)
    offset = Fraction(str(time))
    during["continuous_interval"] = [str(offset + Fraction(value)) for value in during["continuous_interval"]]
    for cell in during["continuous_cells"]:
        cell["interval"] = [str(offset + Fraction(value)) for value in cell["interval"]]
    successor = deepcopy(physical)
    successor["snapshot"]["resources"]["KMR"].update(joint_positions=deepcopy(motion.points[-1]["positions"]),
        current_pose=motion.reference_pose(motion.duration))
    end_time = offset + motion.exact_times[-1]
    after = _checkpoint(successor["snapshot"], geometry, models, index=float(end_time))
    after["time_exact"] = str(end_time)
    successor["snapshot"].update(resources=deepcopy(after["resources"]), parts=deepcopy(after["parts"]))
    successor.pop("kmr_pending_support_departure", None)
    interval["duration"] = motion.duration
    interval["end_time_exact"] = str(end_time)
    return successor, [during, after], interval


def _apply_custody(step: dict, params: dict, declared: dict, physical: dict, time: float,
                   *, robot_description: str | None = None) -> tuple[dict, list, dict] | None:
    from cais_spade_llm.resources.resource_safety_preparation import _outline_custody_evidence
    from cais_spade_llm.resources.recovery_geometry import (
        _checkpoint, _configuration_after, _custody, projected_part_support,
    )

    op = step["op"]
    if op == "move_to_named_pose" and params.get("pose_name") == "home":
        op = "open_gripper"
    if op not in {"grasp_part", "release_part", "open_gripper", "close_gripper"}:
        return None
    state = physical["snapshot"]
    owner = state["resources"]["KMR"]
    part = declared["task"]["parameters"].get("part_name")
    if op == "release_part":
        part = owner["held_part"]
        if ("transform" in params and params["transform"] is not None
                and fingerprint(params["transform"]) != fingerprint(owner.get("grasp_transform"))):
            raise ValueError("KMR release transform conflicts with projected attachment")
    model = physical["resource_models"]["KMR"]
    contracts = model["primitive_model"]["configuration"]["custody_by_part"]
    contract = contracts.get(part, contracts["default"])
    target_state = "closed" if op in {"grasp_part", "close_gripper"} else "open"
    interval_model = deepcopy(model)
    configuration = interval_model["configuration"]
    for joint in configuration["joints"]:
        if joint["name"] in contract["joint_positions"][target_state]:
            if joint["name"] not in contract["joint_bounds"]:
                raise ValueError("KMR gripper change lacks complete joint bounds")
            joint.pop("frozen_position", None)
            joint["frozen_interval"] = deepcopy(contract["joint_bounds"][joint["name"]])
    _holds(physical, robot_description=robot_description)
    interval_models = {**physical["resource_models"], "KMR": interval_model}
    during = _checkpoint(state, physical["geometry"], interval_models, index=time)
    during.update(phase="between", continuous_interval=[str(time), str(time)])
    successor = deepcopy(physical)
    if op in {"grasp_part", "release_part"}:
        if part not in state["parts"]:
            raise ValueError("KMR custody operation needs the bound recorded part")
        if op == "grasp_part":
            if params.get("part_name", part) != part:
                raise ValueError("KMR grasp changes the declared part identity")
            if params.get("initial") is not None and math.dist(params["initial"][:3], state["parts"][part]["current_pose"][:3]) > .01:
                raise ValueError("KMR recorded part conflicts with declared grasp observation")
            if math.dist(owner["current_pose"][:3], state["parts"][part]["current_pose"][:3]) > .06:
                raise ValueError("KMR grasp requires the part at the gripper")
        projected = _outline_custody_evidence(primitive=op,
            params={"part_name": part, "model_name": state["parts"][part]["model_name"]},
            snapshot=state, resource_id="KMR", model=model, contract=contract)
        if op == "release_part":
            support = projected_part_support(part_name=part, state=state, evidence=physical)
            if support.get("status") != "supported":
                raise ValueError("KMR release has no supported deposited-part rest: " + str(support.get("reason")))
            successor.setdefault("stationary_part_evidence", {})[part] = support
            if part not in successor.setdefault("stationary_parts", []):
                successor["stationary_parts"].append(part)
            destination = declared["task"]["parameters"].get("destination_location")
            if destination not in state["resources"]:
                raise ValueError("KMR release needs a declared destination owner")
            projected["custody_effects"][0]["contained_by"] = destination
        else:
            successor["kmr_pending_support_departure"] = True
        _configuration_after(successor["resource_models"], "KMR", projected)
        _custody(successor["snapshot"], "KMR", projected["custody_effects"], physical["geometry"])
        end = successor["snapshot"]["resources"]["KMR"]
        output = {"attached": op == "grasp_part", "held_part": end["held_part"], "gripper_state": end["gripper_state"]}
        if op == "grasp_part":
            output["grasp_transform"] = deepcopy(end["grasp_transform"])
    else:
        if owner["held_part"] is not None:
            raise ValueError("KMR gripper-only preview cannot change held-part attachment")
        configuration_after = deepcopy(model["configuration"])
        for joint in configuration_after["joints"]:
            if joint["name"] in contract["joint_positions"][target_state]:
                joint.pop("frozen_interval", None)
                joint["frozen_position"] = contract["joint_positions"][target_state][joint["name"]]
                if joint["name"] in configuration_after.get("fixed_joint_positions", {}):
                    configuration_after["fixed_joint_positions"][joint["name"]] = joint["frozen_position"]
        _configuration_after(successor["resource_models"], "KMR", {"configuration_after": configuration_after})
        successor["snapshot"]["resources"]["KMR"]["gripper_state"] = target_state
        output = {"gripper_state": target_state, "width": contract["gripper"][target_state]}
    after = _checkpoint(successor["snapshot"], physical["geometry"], successor["resource_models"], index=time)
    successor["snapshot"].update(resources=deepcopy(after["resources"]), parts=deepcopy(after["parts"]))
    return successor, [during, after], output


def _case_evidence(points: list[dict], *, physical: dict, evidence: dict,
                   declared: dict, trajectory: dict | None = None,
                   stage_start: float = 0., planning_scene: dict | None = None,
                   recorded_initial: bool = False, custody_source: dict | None = None,
                   robot_description: str | None = None) -> None:
    from cais_spade_llm.agents.central_controller.recovery_case_evidence import calculate_case_resource_evidence

    source = physical if recorded_initial or trajectory is not None else {
        **physical, "physics_evidence": {}, "provenance": {**physical.get("provenance", {}), "kind": "projected"}}
    configuration = evidence["inputs"]["scene"]
    bindings = declared["task"]["parameters"]
    targets = {bindings[field] for field in ("origin_resource_location", "destination_location", "target_resource")
               if field in bindings}
    docking_target = next(iter(targets)) if len(targets) == 1 else None
    holds = _holds(physical, robot_description=robot_description) if trajectory is not None else None
    for point in points:
        raw = calculate_case_resource_evidence(resource_id="KMR", observation=point,
            physical_evidence=source, configuration=configuration["KMR"],
            joint_trajectory=trajectory, trajectory_start_time=stage_start,
            docking_target=docking_target, planning_scene=planning_scene,
            scene_configuration=configuration, peer_holds=holds)
        owner = point.setdefault("owner_safety_evidence", {})
        owner.setdefault("resource_evidence", {})["KMR"] = raw
        if "collision_object_ids" in raw:
            owner["collision_object_ids"] = deepcopy(raw["collision_object_ids"])
        if custody_source is not None:
            fields = (("held_part", "gripper_state") if custody_source["op"] in {"grasp_part", "release_part"}
                      else ("gripper_state",))
            owner["physical_state_updates"] = {"KMR": {
                field: deepcopy(point["resources"]["KMR"][field]) for field in fields}}
            owner["physical_state_sources"] = {"KMR": deepcopy(custody_source)}


def _replay(record: dict, *, task: dict, recovery_snapshot: dict, recovery_des_model: dict,
            resource_evidence: dict, physical: dict, declared: dict) -> dict:
    from cais_spade_llm.resources.environment_recovery import _finish_environment_projection
    from cais_spade_llm.resources.recovery_geometry import _checkpoint, outline_event_binding

    event = _program(declared, resource_evidence, physical)
    binding = outline_event_binding(task, recovery_snapshot, recovery_des_model, physical)
    if (type(record.get("version")) is not int or record["version"] not in {2, 3}
            or fingerprint(record.get("binding")) != fingerprint(binding)
            or record.get("owner_declaration_fingerprint") != fingerprint(event)
            or record.get("resource_evidence_fingerprint") != fingerprint(resource_evidence)
            or fingerprint(record.get("declared_task")) != fingerprint(declared["task"])):
        raise ValueError("KMR private event evidence has changed source, task, model, or physical start")
    robot_description = record["planning_scene_capture"]["robot_description"] if record["version"] == 3 else None
    if record["version"] == 3 and (not isinstance(robot_description, str) or not robot_description):
        raise ValueError("KMR scoped hold evidence requires the captured robot description")
    scene, outputs = deepcopy(physical), {}
    state = _initial_state(task, recovery_des_model, resource_evidence, physical, recovery_snapshot, declared)
    observations = [_checkpoint(scene["snapshot"], scene["geometry"], scene["resource_models"], index=0)]
    _case_evidence(observations, physical=physical, evidence=resource_evidence, declared=declared,
                   recorded_initial=physical.get("provenance", {}).get("kind") != "projected")
    rows = record["steps"]
    if len(rows) != len(event["program"]["steps"]):
        raise ValueError("KMR physical evidence omits declared task steps")
    time, intervals, custody_source = 0., [], None
    for step, row in zip(event["program"]["steps"], rows, strict=True):
        resolved = _resolve_step(step, declared, state, outputs, resource_evidence)
        if row.get("step_fingerprint") != fingerprint(step) or fingerprint(row.get("resolved")) != fingerprint(resolved):
            raise ValueError("KMR physical evidence changes a declared step or its parameter sources")
        if resolved["status"] == "skipped":
            output = None
        else:
            params = resolved["params"]
            output = _calculation(step, params, declared, scene, resource_evidence)
            custody = _apply_custody(step, params, declared, scene, time, robot_description=robot_description)
            if custody is not None:
                _case_evidence(custody[1][:-1], physical=scene, evidence=resource_evidence,
                    declared=declared, custody_source=custody_source)
                custody_source = {"step_fingerprint": fingerprint(step), "op": step["op"],
                                  "source_fingerprint": fingerprint(record)}
                _case_evidence(custody[1][-1:], physical=custody[0], evidence=resource_evidence,
                    declared=declared, custody_source=custody_source)
            if custody is not None and step["op"] == "move_to_named_pose":
                scene, points, _ = custody
                observations.extend(points)
            motion = _motion_request(step, params, scene, resource_evidence)
            if motion is not None:
                request = _scene_request(task, scene, resource_evidence, record["planning_scene_capture"], motion,
                                         project_base=record["version"] == 3)
                if fingerprint(row.get("request")) != fingerprint(request):
                    raise ValueError("KMR physical preview has a different projected scene or motion target")
                before_motion = scene
                scene, points, interval = _apply_motion(task, scene, request, row["response"], time,
                                                      robot_description=robot_description)
                _case_evidence(points, physical=before_motion, evidence=resource_evidence,
                    declared=declared, trajectory=interval["joint_trajectory"], stage_start=time,
                    planning_scene=request["planning_scene"], custody_source=custody_source,
                    robot_description=robot_description)
                observations.extend(points)
                time = float(Fraction(interval["end_time_exact"]))
                intervals.append(interval)
                output = {"tcp_pose": deepcopy(scene["snapshot"]["resources"]["KMR"]["current_pose"])}
            elif custody is not None:
                scene, points, output = custody
                observations.extend(points)
                intervals.append({"kind": "declared custody interval", "step_fingerprint": fingerprint(step),
                    "end_snapshot_fingerprint": fingerprint(scene["snapshot"])})
            elif output is None:
                raise ValueError("KMR declared operation requires supported physical evidence: " + step["op"])
        if fingerprint(row.get("output")) != fingerprint(output):
            raise ValueError("KMR declared step output differs from the supported calculation")
        if step.get("store_as"):
            outputs[step["store_as"]] = deepcopy(output)
    if not intervals:
        raise ValueError("KMR owner preview has no validated physical interval")
    if scene.get("kmr_pending_support_departure"):
        raise ValueError("KMR pickup completion lacks its declared validated departure")
    if any(scene["snapshot"]["resources"]["KMR"].get(field) != declared["environment_valuation"]["KMR"].get(field)
           for field in ("held_part", "gripper_state") if field in declared["environment_valuation"]["KMR"]):
        raise ValueError("KMR declared custody or gripper effect lacks physical evidence")
    after = deepcopy(scene["snapshot"])
    for rid, values in declared["environment_valuation"].items():
        after["resources"][rid].update(deepcopy(values))
    previous = observations[-1]
    observations[-1] = _checkpoint(after, scene["geometry"], scene["resource_models"], index=time)
    observations[-1]["time_exact"] = previous["time_exact"]
    observations[-1]["owner_safety_evidence"] = deepcopy(previous["owner_safety_evidence"])
    for index, point in enumerate(observations):
        point["state_snapshot"] = deepcopy(declared["environment_valuation"] if index == len(observations)-1
                                          else physical.get("environment_valuation", resource_evidence["environment_valuation"]))
    contract = {"kind": "KMR declared owner program preview", "owner_declaration": deepcopy(event),
                "declared_task": declared["task"], "intervals": intervals,
                "resource_configurations": {"KMR": deepcopy(resource_evidence["inputs"]["scene"]["KMR"])},
                "preview_fingerprint": fingerprint(record)}
    return _finish_environment_projection(task=task, recovery_snapshot=recovery_snapshot,
        recovery_des_model=recovery_des_model, resource_evidence=resource_evidence, physical=physical,
        declared=declared, geometry=scene["geometry"], models=scene["resource_models"],
        observations=observations, successor_snapshot=after, contract=contract)


def project_kmr_event(*, task: dict, recovery_snapshot: dict, recovery_des_model: dict,
                      resource_evidence: dict, physical: dict, declared: dict) -> dict:
    """Replay complete recorded KMR owner evidence without any live interface."""
    errors = []
    for record in physical.get("event_evidence", []):
        if record.get("source") != "KMR declared owner program preview":
            continue
        try:
            return _replay(record, task=task, recovery_snapshot=recovery_snapshot,
                recovery_des_model=recovery_des_model, resource_evidence=resource_evidence,
                physical=physical, declared=declared)
        except (KeyError, ValueError, TypeError, IndexError, StopIteration) as exc:
            errors.append(str(exc))
    return _missing("; ".join(errors) or "KMR requires its declared owner program preview from the recorded scene")


async def prepare_kmr_event_evidence(owner, *, task: dict, recovery_snapshot: dict,
        recovery_des_model: dict, physical_evidence: dict, resource_evidence: dict,
        observations: dict, preview_call=None, allow_live_capture: bool = True, cancel_event=None) -> dict:
    """Prepare and retain native KMR previews without dispatch or scene mutation."""
    if physical_evidence.get("endpoint_evidence_policy_version") in (1, 2):
        raise ValueError("Endpoint policy disables KMR trajectory previews; use owner endpoints and bounds")
    from cais_spade_llm.recovery_framework.outline_gazebo_evidence import (
        _check_cancelled, _saved_planning_scene, _prepare_scene_object_bindings,
    )
    from cais_spade_llm.recovery_framework.recovery_motion_preview import (
        capture_preview_scene, preview_recovery_motion, preview_scene_fingerprint,
    )
    from cais_spade_llm.resources.recovery_geometry import outline_event_binding

    if resource_evidence.get("execution_mode") != "simulation":
        raise ValueError("KMR outline previews require simulation")
    if not allow_live_capture and preview_call is None:
        raise ValueError("Offline KMR preview requires the saved-model interface")
    scene_capture = _saved_planning_scene(observations)
    settings = resource_evidence["inputs"]["scene"]["safety_preparation"]["outline_preview"]
    if scene_capture is None:
        if not allow_live_capture:
            return _missing("KMR preview requires the saved planning_scene_capture")
        scene_capture = await capture_preview_scene(execution_mode="simulation",
            planning_scene_service=settings["planning_scene_service"],
            preview_parameter_service=settings["preview_parameter_service"], timeout_seconds=settings["timeout_seconds"])
    if preview_scene_fingerprint(scene_capture["planning_scene"]) != scene_capture["scene_fingerprint"]:
        raise ValueError("KMR saved planning scene fingerprint changed")
    physical = deepcopy(physical_evidence)
    bindings = _prepare_scene_object_bindings(observations=observations, scene_capture=scene_capture,
        task=task, resource_id=resource_evidence['resource_id'])
    if bindings['status'] != 'ready':
        return {**_missing(bindings['reason']), **bindings}
    physical["scene_object_bindings"] = bindings['scene_object_bindings']
    failures = []
    for declared in _declarations(task, recovery_des_model, resource_evidence, physical, failures):
        record = {"version": 3, "source": "KMR declared owner program preview", "command_sent": False,
            "execution_authorized": False, "planning_scene_capture": deepcopy(scene_capture),
            "binding": outline_event_binding(task, recovery_snapshot, recovery_des_model, physical),
            "resource_evidence_fingerprint": fingerprint(resource_evidence), "declared_task": deepcopy(declared["task"]), "steps": []}
        try:
            event = _program(declared, resource_evidence, physical)
            record["owner_declaration_fingerprint"] = fingerprint(event)
            state = _initial_state(task, recovery_des_model, resource_evidence, physical, recovery_snapshot, declared)
            scene, outputs, time = deepcopy(physical), {}, 0.
            for step in event["program"]["steps"]:
                _check_cancelled(cancel_event)
                resolved = _resolve_step(step, declared, state, outputs, resource_evidence)
                row = {"step_fingerprint": fingerprint(step), "resolved": deepcopy(resolved)}
                record["steps"].append(row)
                output = None
                if resolved["status"] == "resolved":
                    params = resolved["params"]
                    output = _calculation(step, params, declared, scene, resource_evidence)
                    custody = _apply_custody(step, params, declared, scene, time,
                                             robot_description=scene_capture["robot_description"])
                    if custody is not None and step["op"] == "move_to_named_pose":
                        scene, _, _ = custody
                    motion = _motion_request(step, params, scene, resource_evidence)
                    if motion is not None:
                        request = _scene_request(task, scene, resource_evidence, scene_capture, motion, project_base=True)
                        response = await (preview_call or preview_recovery_motion)(request, execution_mode="simulation",
                            action_name=settings["action_name"], timeout_seconds=settings["timeout_seconds"], cancel_event=cancel_event)
                        row.update(request=deepcopy(request), response=deepcopy(response))
                        _check_cancelled(cancel_event)
                        scene, _, interval = _apply_motion(task, scene, request, response, time,
                                                          robot_description=scene_capture["robot_description"])
                        time = float(Fraction(interval["end_time_exact"]))
                        output = {"tcp_pose": deepcopy(scene["snapshot"]["resources"]["KMR"]["current_pose"])}
                    elif custody is not None:
                        scene, _, output = custody
                    elif output is None:
                        raise ValueError("KMR declared operation requires supported physical evidence: " + step["op"])
                row["output"] = deepcopy(output)
                if step.get("store_as"):
                    outputs[step["store_as"]] = deepcopy(output)
            projection = _replay(record, task=task, recovery_snapshot=recovery_snapshot,
                recovery_des_model=recovery_des_model, resource_evidence=resource_evidence,
                physical=physical, declared=declared)
            if projection.get("status") != "projected":
                raise ValueError(projection.get("reason") or str(projection.get("unresolved")))
            physical.setdefault("event_evidence", []).append(deepcopy(record))
            return {"status": "prepared", "command_sent": False, "execution_authorized": False,
                "event_evidence": [record], "physical_evidence": physical,
                "resource_models": deepcopy(physical["resource_models"]), "geometry_evidence": projection,
                "failed_owner_previews": failures}
        except (KeyError, ValueError, TypeError, IndexError, StopIteration) as exc:
            failures.append({"reason": str(exc), "record": record})
    return {**_missing("; ".join(row["reason"] for row in failures) or "No KMR owner declaration supports the proposal"),
            "failed_owner_previews": failures}
