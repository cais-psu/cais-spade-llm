from __future__ import annotations

"""Check recorded nominal robot steps against independently validated evidence.

This module reads an existing task program. It never orders primitive support,
calls a controller, or grants permission to execute the program.
"""

from copy import deepcopy
import math
from typing import Any

from cais_spade_llm.agents.shared_information.recovery_validation_protocol import (
    recovery_validation_fingerprint,
)
from cais_spade_llm.resources.recovery_parameter_evidence import _resolve


def nominal_task_bindings(*, task: dict, recovery_des_model: dict) -> dict:
    """Resolve optional callable defaults only from the matching owner declaration."""
    from .robot_task_registry import robot_task_registry

    name = task.get("required_task_contract")
    matches = [row for row in recovery_des_model.get("events", []) if row.get("event_name") == name]
    if len(matches) != 1:
        return {"status": "NEEDS_CONTEXT", "reason": "One exact nominal declaration is required"}
    program = matches[0].get("composition_contract", {})
    bindings, defaults = deepcopy(task.get("params") or {}), {}
    registered = robot_task_registry().get(name)
    declarations = program.get("params", {})
    registered_params = ({row.name: row.param_schema() for row in registered.arguments}
                         if registered is not None else None)
    for key, schema in declarations.items():
        if key in bindings:
            continue
        if "default" in schema:
            bindings[key] = deepcopy(schema["default"])
            defaults[key] = {"source": "recorded parameter default", "value": deepcopy(schema["default"])}
        elif key not in program.get("required_params", []) and declarations == registered_params:
            argument = next(row for row in registered.arguments if row.name == key)
            if not argument.required:
                bindings[key] = deepcopy(argument.default)
                defaults[key] = {"source": "matching RobotTaskArgument callable default",
                                 "value": deepcopy(argument.default), "recorded_argument_was_present": False}
    return {"status": "supported", "bindings": bindings, "default_evidence": defaults}


def _expression(value: Any, bindings: dict, state: dict, outputs: dict) -> tuple[bool, Any, list]:
    if isinstance(value, dict) and set(value) == {"$first"} and isinstance(value["$first"], list):
        for item in value["$first"]:
            known, resolved, missing = _expression(item, bindings, state, outputs)
            if not known:
                return False, None, missing
            if resolved not in (None, ""):
                return True, resolved, []
        return True, None, []
    if isinstance(value, dict) and set(value) == {"$format", "values"}:
        known, resolved, missing = _expression(value["values"], bindings, state, outputs)
        if not known or not isinstance(resolved, dict) or not isinstance(value["$format"], str):
            return False, None, missing or ["Unsupported recorded format expression"]
        try:
            return True, value["$format"].format_map({key: "" if item is None else item
                                                    for key, item in resolved.items()}), []
        except (KeyError, ValueError, TypeError):
            return False, None, ["Unresolved recorded format expression"]
    if isinstance(value, dict) and not any(str(key).startswith("$") for key in value):
        rows = {key: _expression(item, bindings, state, outputs) for key, item in value.items()}
        return (all(row[0] for row in rows.values()), {key: row[1] for key, row in rows.items()},
                [ref for row in rows.values() for ref in row[2]])
    if isinstance(value, list):
        rows = [_expression(item, bindings, state, outputs) for item in value]
        return all(row[0] for row in rows), [row[1] for row in rows], [ref for row in rows for ref in row[2]]
    return _resolve(value, bindings, state, "/nominal_task", outputs)


def nominal_step_condition(*, step: dict, bindings: dict, state: dict,
                           resource_evidence: dict, step_outputs: dict | None = None) -> dict:
    """Resolve recorded guards without treating absent values as false."""
    outputs, facts, checks = step_outputs or {}, [], []
    when = step.get("when", [])
    if not isinstance(when, list):
        return {"status": "NEEDS_CONTEXT", "facts": [], "reason": "Recorded step guards are unavailable"}
    for guard in when:
        if not isinstance(guard, dict):
            checks.append(None)
            continue
        predicate = guard.get("predicate", "always")
        known, args, missing = _expression(guard.get("args", {}), bindings, state, outputs)
        value, source = None, {}
        condition = guard.get("condition", {})
        if condition:
            if (not isinstance(condition, dict) or set(condition) - {"field", "operator", "value"}
                    or not isinstance(condition.get("field"), str)):
                checks.append(None)
                continue
            path = condition["field"].split(".")
            if path[0] == "task_ctx":
                path[0] = "_task_ctx"
            else:
                path[0] = {"held_part": "_held_part", "resource_state": "_current_state",
                           "gripper_state": "_gripper_state", "resource_location": "_recovery_pose_ref"}.get(
                               path[0], "_" + path[0])
            known, actual, missing = _resolve({"$state": path[0], "path": path[1:]}, bindings,
                                               state, "/nominal_task", outputs)
            expected_known, expected, refs = _expression(condition.get("value"), bindings, state, outputs)
            missing += refs
            operator = condition.get("operator")
            if known and expected_known:
                equal = actual == expected
                value = equal if operator == "equals" else not equal if operator == "not_equals" else (
                    actual not in (None, "") if operator == "exists" else None)
            source = {"field": condition["field"], "available": known, "actual": actual,
                      "operator": operator, "expected": expected}
        elif known and isinstance(args, dict):
            if predicate in ("", "always"):
                value, source = True, {"declaration": "always"}
            elif predicate == "execution_mode":
                present = "execution_mode" in resource_evidence
                value = resource_evidence["execution_mode"] == args.get("mode") if present and "mode" in args else None
                source = {"field": "execution_mode", "available": present,
                          "actual": resource_evidence.get("execution_mode"), "expected": args.get("mode")}
            elif predicate == "task_ctx_key_truthy":
                context, key = state.get("_task_ctx"), args.get("key")
                present = isinstance(context, dict) and isinstance(key, str) and key in context
                value = bool(context[key]) if present else None
                source = {"field": "_task_ctx", "key": key, "available": present,
                          "actual": deepcopy(context[key]) if present else None}
            elif predicate in {"held_part_exists", "held_part_empty"}:
                present = "_held_part" in state
                value = bool(state["_held_part"]) if present else None
                if value is not None and predicate == "held_part_empty":
                    value = not value
                source = {"field": "_held_part", "available": present, "actual": state.get("_held_part")}
            elif predicate == "move_insert_required":
                from .recovery_capability import evaluate_robot_primitive_predicate

                recorded = {**resource_evidence, "state": state}
                checked = evaluate_robot_primitive_predicate(predicate=predicate, resource_evidence=recorded,
                    recovery_snapshot={}, resource_jid=str(resource_evidence.get("resource_jid", "")))
                value = checked.get("value")
                source = {"execution_mode": resource_evidence.get("execution_mode"),
                          "_task_ctx": deepcopy(state.get("_task_ctx")), **checked}
        checks.append(value)
        facts.append({"predicate": predicate, "guard_fingerprint": recovery_validation_fingerprint(guard),
                      "value": value, "source": source, "missing_facts": missing})
    status = "NEEDS_CONTEXT" if None in checks else "not_applicable" if False in checks else "applicable"
    if not when:
        facts.append({"declaration": "unconditional", "when_fingerprint": recovery_validation_fingerprint(when)})
    return {"status": status, "facts": facts}


def resolve_nominal_task_step(*, step: dict, bindings: dict, state: dict,
                              resource_evidence: dict, step_outputs: dict | None = None) -> dict:
    """Resolve a step's exact arguments after its recorded guard is established."""
    outputs = step_outputs or {}
    condition = nominal_step_condition(step=step, bindings=bindings, state=state,
                                      resource_evidence=resource_evidence, step_outputs=outputs)
    result = {"step_id": step.get("id"), "step_fingerprint": recovery_validation_fingerprint(step),
              "condition": condition}
    if condition["status"] != "applicable":
        return {**result, "status": "skipped" if condition["status"] == "not_applicable" else "NEEDS_CONTEXT"}
    expressions = step.get("params", {})
    unused, parameter_evidence = {}, {}
    if step.get("op") == "move_insert" and resource_evidence.get("execution_mode") == "simulation":
        physical_only = {"calibration_id", "profile_sha256", "hard_caps_sha256", "insertion_axis_world",
            "contact_speed_m_s", "contact_force_delta_n", "engagement_progress_m", "insertion_force_n",
            "spiral_radius_m", "spiral_pitch_m", "spiral_speed_m_s", "spiral_acceleration_m_s2",
            "max_axial_force_n", "max_lateral_force_n", "max_torque_nm", "force_depth_profile",
            "baseline_force_uncertainty_n", "baseline_torque_uncertainty_nm", "tilt_tolerance_rad",
            "seated_depth_tolerance_m", "settle_time_sec", "timeout_sec"}
        used = {"part_name", "expected_start_pose", "target_pose", "trial_id"}
        if not isinstance(expressions, dict) or set(expressions) - physical_only - used:
            return {**result, "status": "NEEDS_CONTEXT", "reason": "Unsupported simulation move_insert argument"}
        unused = {key: {"expression": deepcopy(value), "source": "GazeboPickPlaceController.move_insert simulation branch",
                        "reason": "Not used to validate or execute simulation_direct motion"}
                  for key, value in expressions.items() if key in physical_only}
        expressions = {key: value for key, value in expressions.items() if key in used}
        trial_expression = {"$format": "{trial_id}", "values": {
            "trial_id": {"$state": "_task_ctx", "path": ["move_insert_trial_id"]}}}
        if (expressions.get("trial_id") == trial_expression and isinstance(state.get("_task_ctx"), dict)
                and "move_insert_trial_id" not in state["_task_ctx"]):
            from .robot_task_model import _resolve_value

            expressions["trial_id"] = _resolve_value(trial_expression, args=bindings,
                                                     runtime_state=state, step_outputs=outputs)
            parameter_evidence["trial_id"] = {
                "source": "robot_task_model._resolve_value recorded $format expression",
                "expression": trial_expression, "source_field_present": False,
                "value": expressions["trial_id"], "role": "simulation move_insert trace identifier"}
    known, params, missing = _expression(expressions, bindings, state, outputs)
    if not known or not isinstance(params, dict):
        return {**result, "status": "NEEDS_CONTEXT", "missing_facts": missing,
                "reason": "Nominal step parameters lack recorded values"}
    return {**result, "status": "resolved", "params": params,
            **({"unused_parameters": unused} if unused else {}),
            **({"parameter_evidence": parameter_evidence} if parameter_evidence else {})}


def _position(value: Any) -> dict:
    fields = ("x", "y", "z", "qx", "qy", "qz", "qw")
    if isinstance(value, (list, tuple)) and len(value) == 7:
        value = dict(zip(fields, value, strict=True))
    if (not isinstance(value, dict) or any(key not in value or type(value[key]) not in (int, float)
            or not math.isfinite(value[key]) for key in fields)):
        raise ValueError("Nominal motion requires a complete recorded pose")
    return {key: value[key] for key in fields}


def _motion_end(interval: dict) -> dict:
    request = interval.get("motion_request") or {}
    points = request.get("waypoints")
    if request.get("mode") != "cartesian" or not isinstance(points, list) or not points:
        raise ValueError("Nominal Cartesian step lacks its exact validated motion request")
    if not interval.get("joint_trajectory") or not interval.get("native_preview_binding"):
        raise ValueError("Nominal motion lacks native trajectory binding")
    pose = points[-1]
    return _position({**pose.get("position", {}),
                      **{"q" + key: value for key, value in pose.get("orientation", {}).items()}})


def _step_effect(*, step: dict, params: dict, state: dict, evidence: dict,
                  intervals: list, resource_evidence: dict, resource_model: dict) -> dict:
    op = step.get("op")
    if op in {"move_cartesian", "move_relative", "move_insert"}:
        if len(intervals) != 1:
            raise ValueError("Nominal movement requires one corresponding validated interval")
        expected = _position(state.get("_position"))
        if op == "move_relative":
            for axis in ("x", "y", "z"):
                delta = params.get("d" + axis)
                if type(delta) not in (int, float) or not math.isfinite(delta):
                    raise ValueError("Nominal relative displacement is unavailable")
                expected[axis] += delta
        elif op == "move_insert":
            if resource_evidence.get("execution_mode") != "simulation":
                raise ValueError("Nominal move_insert physical branch has no recorded preview model")
            from .gazebo_pick_place_controller import _pose_delta, _pose_from_mapping

            motion = resource_evidence.get("controller_config", {}).get("motion", {})
            tolerance = motion.get("simulation_insert_start_position_tolerance_m")
            if tolerance is None:
                cartesian_tolerance = motion.get("cartesian_position_tolerance_m")
                if type(cartesian_tolerance) not in (int, float) or not math.isfinite(cartesian_tolerance):
                    raise ValueError("Nominal move_insert lacks its recorded start-position tolerance")
                tolerance = cartesian_tolerance + .001
            if type(tolerance) not in (int, float) or not math.isfinite(tolerance):
                raise ValueError("Nominal move_insert start-position tolerance is invalid")
            delta, rotation = _pose_delta(_pose_from_mapping(expected),
                                         _pose_from_mapping(params.get("expected_start_pose")))
            if delta > max(.003, tolerance) or rotation > 3.:
                raise ValueError("Nominal move_insert expected start differs from its supported configuration")
            expected = _position(params.get("target_pose"))
        else:
            expected = _position(params)
        if _motion_end(intervals[0]) != expected:
            raise ValueError("Nominal motion target differs from its recorded step parameters")
        return {"kind": "motion", "target_pose": expected,
                "interval_fingerprint": recovery_validation_fingerprint(intervals[0])}
    if op in {"grasp_part", "release_part"}:
        if len(intervals) != 1:
            raise ValueError("Nominal custody step requires one validated interval")
        kind = "grasp" if op == "grasp_part" else "release"
        effects = intervals[0].get("custody_effects", [])
        part = params.get("part_name")
        if not isinstance(part, str) or not part:
            raise ValueError("Nominal custody requires its explicit resolved part_name")
        if "model_name" in params:
            recorded_model = resource_evidence.get("inputs", {}).get("geometry", {}).get(
                "parts", {}).get("model_map", {}).get(part)
            if not isinstance(recorded_model, str) or not recorded_model or params["model_name"] != recorded_model:
                raise ValueError("Nominal custody model_name differs from its recorded part identity")
        if (len(effects) != 1 or effects[0].get("kind") != kind or effects[0].get("part") != part
                or intervals[0].get("interval_complete") is not True
                or "assembly_slot" in params or params.get("assume_released_if_open") is True):
            raise ValueError("Nominal custody effect differs from the recorded step")
        return {"kind": "custody", "effect": deepcopy(effects[0]),
                "interval_fingerprint": recovery_validation_fingerprint(intervals[0])}
    if op == "move_to_named_pose":
        named = resource_model.get("named_joint_poses", {}).get(params.get("pose_name"))
        if not isinstance(named, dict) or len(intervals) != 1:
            raise ValueError("Nominal named motion requires its exact recorded joint target")
        request = intervals[0].get("motion_request", {})
        target = request.get("joint_target", {})
        trajectory = intervals[0].get("joint_trajectory", {})
        points = trajectory.get("points", [])
        if (request.get("mode") != "joint_target" or target.get("name") != named.get("joint_names")
                or target.get("position") != named.get("positions")
                or trajectory.get("joint_names") != named.get("joint_names") or not points
                or points[-1].get("positions") != named.get("positions")
                or not intervals[0].get("native_preview_binding")):
            raise ValueError("Nominal named motion differs from its recorded joint target or native evidence")
        return {"kind": "named_joint_motion", "pose_name": params["pose_name"],
                "named_joint_pose_fingerprint": recovery_validation_fingerprint(named),
                "interval_fingerprint": recovery_validation_fingerprint(intervals[0])}
    if op == "snap_part_to_slot":
        if len(intervals) != 1 or not isinstance(intervals[0].get("nominal_snap_evidence"), dict):
            raise ValueError("Nominal snap_part_to_slot has no separately validated simulation correction")
        proof = intervals[0]["nominal_snap_evidence"]
        if proof.get("status") != "supported" or proof.get("params") != params:
            raise ValueError("Nominal snap_part_to_slot evidence differs from its parameters")
        if not proof.get("binding") or proof.get("physical_execution_verified") is not False:
            raise ValueError("Nominal snap_part_to_slot provenance is unavailable")
        return {"kind": "simulation_state_correction", "evidence_fingerprint": recovery_validation_fingerprint(proof)}
    if op in {"compute_pick_targets", "compute_place_targets"}:
        if intervals or evidence.get("calculation") != op or evidence.get("params") != params:
            raise ValueError("Nominal calculation lacks its matching owner calculation record")
        if (not isinstance(evidence.get("output"), dict) or not evidence.get("source")
                or not evidence.get("input_fingerprint")):
            raise ValueError("Nominal calculation lacks recorded inputs or output provenance")
        return deepcopy(evidence)
    raise ValueError(f"Nominal step {step.get('id')} uses unsupported owner operation {op}")


def nominal_task_projection(*, task: dict, recovery_des_model: dict, resource_evidence: dict,
                            step_evidence: list, intervals: list, resource_model: dict | None = None) -> dict:
    """Bind every recorded nominal step to evidence from owner scene replay.

    The caller supplies states reconstructed by its validated interval replay.
    This function checks the task program and argument/effect correspondence;
    it does not turn caller-supplied intervals into validated physical evidence.
    """
    result = {"status": "NEEDS_CONTEXT", "coverage_version": 1, "steps": []}
    try:
        name = task.get("required_task_contract")
        matches = [row for row in recovery_des_model.get("events", []) if row.get("event_name") == name]
        if len(matches) != 1:
            raise ValueError("Nominal program requires one exact recorded owner declaration")
        event = matches[0]
        program = event.get("composition_contract")
        if not isinstance(program, dict) or not isinstance(program.get("steps"), list):
            raise ValueError("Nominal program steps are unavailable")
        resolved = nominal_task_bindings(task=task, recovery_des_model=recovery_des_model)
        if resolved["status"] != "supported":
            raise ValueError(resolved["reason"])
        steps, bindings = program["steps"], resolved["bindings"]
        if len(step_evidence) != len(steps):
            raise ValueError("Every recorded nominal step requires coverage")
        covered = set()
        for index, (step, row) in enumerate(zip(steps, step_evidence, strict=True)):
            if row.get("step_id") != step.get("id") or not isinstance(row.get("state_before"), dict):
                raise ValueError("Nominal step order or validated starting facts differ from the declaration")
            item = resolve_nominal_task_step(step=step, bindings=bindings, state=row["state_before"],
                resource_evidence=resource_evidence, step_outputs=row.get("step_outputs"))
            item["index"] = index
            refs = row.get("interval_indices", [])
            if (not isinstance(refs, list) or any(type(ref) is not int or ref < 0 or ref >= len(intervals) for ref in refs)
                    or len(set(refs)) != len(refs)):
                raise ValueError("Nominal step interval references are invalid")
            item["interval_indices"] = deepcopy(refs)
            if item["status"] == "skipped":
                if refs or row.get("effect_evidence"):
                    raise ValueError("A skipped nominal step cannot have projected effects")
            elif item["status"] == "resolved":
                proof = _step_effect(step=step, params=item["params"], state=row["state_before"],
                    evidence=row.get("effect_evidence") or {}, intervals=[intervals[ref] for ref in refs],
                    resource_evidence=resource_evidence, resource_model=resource_model or {})
                item.update(status="supported", effect_evidence_fingerprint=recovery_validation_fingerprint(proof))
                covered.update(refs)
            else:
                result["steps"].append(item)
                raise ValueError(item.get("reason", "Nominal step guard remains unresolved"))
            result["steps"].append(item)
        if covered != set(range(len(intervals))):
            raise ValueError("Owner preview contains intervals without a corresponding nominal step")
        return {**result, "status": "supported", "task_contract": name,
                "declaration_fingerprint": recovery_validation_fingerprint(event),
                "params": {key: value for key, value in bindings.items()
                           if key in (task.get("params") or {}) or "default" in program.get("params", {}).get(key, {})},
                "argument_default_evidence": resolved["default_evidence"],
                "program_fingerprint": recovery_validation_fingerprint(program), "interval_count": len(intervals)}
    except (KeyError, ValueError, TypeError, IndexError) as exc:
        return {**result, "reason": str(exc)}


def apply_nominal_task_effects(*, task: dict, recovery_des_model: dict, resource_evidence: dict,
                               state: dict, step_outputs: dict,
                               absent_output_fields: dict | None = None) -> dict:
    """Apply declared private bookkeeping to verified outputs, preserving physical pose."""
    result = {"status": "NEEDS_CONTEXT", "effects": []}
    try:
        matches = [row for row in recovery_des_model["events"]
                   if row.get("event_name") == task.get("required_task_contract")]
        if len(matches) != 1:
            raise ValueError("Nominal effects require one exact owner task")
        bindings = nominal_task_bindings(task=task, recovery_des_model=recovery_des_model)
        if bindings["status"] != "supported":
            raise ValueError(bindings["reason"])
        projected = deepcopy(state)
        for index, effect in enumerate(matches[0]["composition_contract"].get("effects", [])):
            condition = nominal_step_condition(step=effect, bindings=bindings["bindings"], state=projected,
                resource_evidence=resource_evidence, step_outputs=step_outputs)
            row = {"index": index, "effect_fingerprint": recovery_validation_fingerprint(effect), "condition": condition}
            if condition["status"] == "not_applicable":
                result["effects"].append({**row, "status": "skipped"})
                continue
            if condition["status"] != "applicable":
                raise ValueError("Nominal effect condition is unresolved")
            target, action = effect.get("target"), effect.get("action")
            if target not in {"task_ctx", "position", "held_part", "current_state", "gripper_state", "recovery_pose_ref"}:
                raise ValueError("Unsupported nominal bookkeeping target")
            if action not in {"set", "merge", "clear"} or (action == "merge" and target != "task_ctx"):
                raise ValueError("Unsupported nominal bookkeeping action")
            skipped = []
            expression = effect.get("value")
            if action == "clear":
                value = {} if target == "task_ctx" else None
            elif target == "task_ctx" and effect.get("skip_empty_values") is True and isinstance(expression, dict):
                value = {}
                for key, child in expression.items():
                    known, item, missing = _expression(child, bindings["bindings"], projected, step_outputs)
                    if not known:
                        path = child.get("path") if isinstance(child, dict) else None
                        absent = (absent_output_fields or {}).get(child.get("$step"), []) if isinstance(child, dict) else []
                        is_absent_output = (isinstance(child, dict) and set(child) <= {"$step", "path"}
                            and isinstance(step_outputs.get(child.get("$step")), dict)
                            and isinstance(path, list) and len(path) == 1 and path[0] in absent)
                        if not is_absent_output:
                            raise ValueError("Nominal task_ctx effect lacks a required recorded fact")
                        skipped.append({"field": key, "reason": "Owner calculation establishes this exact output field is absent",
                                        "missing_refs": missing})
                    elif item not in (None, ""):
                        value[key] = item
                    else:
                        skipped.append({"field": key, "reason": "Recorded skip_empty_values", "value": item})
            else:
                known, value, missing = _expression(expression, bindings["bindings"], projected, step_outputs)
                if not known:
                    raise ValueError("Nominal effect requires unavailable facts: " + ", ".join(missing))
            if action == "merge":
                if not isinstance(projected.get("_task_ctx"), dict) or not isinstance(value, dict):
                    raise ValueError("Nominal task_ctx merge requires its recorded starting dictionary")
                projected["_task_ctx"].update(deepcopy(value))
            elif target != "position":
                projected["_" + target] = deepcopy(value)
            result["effects"].append({**row, "status": "supported", "target": target, "action": action,
                "value": deepcopy(value), "skipped_fields": skipped,
                "physical_pose_replaced": False})
        return {**result, "status": "supported", "state": projected}
    except (KeyError, ValueError, TypeError, IndexError) as exc:
        return {**result, "reason": str(exc)}


def check_nominal_task_effects(*, nominal_effects: dict, resource_updates: dict,
                              physical_resource: dict) -> dict:
    """Compare declared owner state effects with independently supported results."""
    if nominal_effects.get("status") != "supported":
        raise ValueError("Nominal owner state effects are unresolved")
    declared = {row["target"] for row in nominal_effects["effects"]
                if row.get("status") == "supported" and row.get("target") in {
                    "current_state", "held_part", "gripper_state"}}
    checked = {}
    for target, public in (("current_state", "resource_state"), ("held_part", "held_part"),
                           ("gripper_state", "gripper_state")):
        if target not in declared:
            continue
        private = "_" + target
        if private not in nominal_effects["state"]:
            raise ValueError(f"Nominal owner effect {private} has no resolved value")
        value = nominal_effects["state"][private]
        sources = {"resource_updates": resource_updates} if target == "current_state" else {
            "physical_resource": physical_resource,
            **({"resource_updates": resource_updates} if public in resource_updates else {})}
        for source, values in sources.items():
            if public not in values:
                raise ValueError(f"Nominal owner effect {private} lacks supported {source}.{public}")
            if type(value) is not type(values[public]) or value != values[public]:
                raise ValueError(f"Nominal owner effect {private} conflicts with supported {source}.{public}")
        checked[target] = {"private_field": private, "public_field": public,
                           "value": deepcopy(value), "sources": list(sources)}
    return {"status": "supported", "fields": checked}


def project_nominal_snap(*, resource_id: str, params: dict, state: dict, evidence: dict,
                         owner_evidence: dict, endpoint_result: dict | None = None) -> dict:
    """Predict the declared simulation correction after independently supported release.

    A changed part pose requires a separate, complete native scene endpoint check.
    No trajectory, measurement, settling interval, or execution permission is
    inferred from Gazebo's SetEntityState operation.
    """
    result = {"status": "NEEDS_CONTEXT", "kind": "nominal_snap_part_to_slot", "version": 1,
              "command_sent": False, "execution_authorized": False, "physical_execution_verified": False,
              "params": deepcopy(params)}
    try:
        from cais_spade_llm.product.profile import ProductProfile
        from cais_spade_llm.resources.recovery_geometry import projected_part_support
        from cais_spade_llm.resources.recovery_support_departure import _object_id, _plane

        required = {"model_name", "slot_x", "slot_y", "part_height", "board_top_z",
                    "part_origin_z", "destination_location"}
        if set(params) != required:
            raise ValueError("Nominal snap requires every exact declared argument without fallback")
        if owner_evidence.get("execution_mode") != "simulation" or state.get("frame") != "world":
            raise ValueError("Nominal snap requires recorded simulation execution_mode and world frame")
        owner = state["resources"][resource_id]
        if "held_part" not in owner or owner["held_part"] is not None:
            raise ValueError("Nominal snap requires independently confirmed projected release")
        names = [name for name, row in state["parts"].items() if row.get("model_name") == params["model_name"]]
        if len(names) != 1:
            raise ValueError("Nominal snap requires one exact recorded part/model identity")
        part_name = names[0]
        if any(row.get("held_part") == part_name for row in state["resources"].values()):
            raise ValueError("Nominal snap conflicts with retained resource custody")
        config, geometry = owner_evidence["controller_config"], owner_evidence["inputs"]["geometry"]
        if not owner_evidence.get("source"):
            raise ValueError("Nominal snap owner source is unavailable")
        services = config["services"]
        if any(not isinstance(services.get(name), str) or not services[name] for name in ("attach", "detach")):
            raise ValueError("Nominal snap fixture attachment interfaces are unavailable")
        board = geometry["assembly_board"]
        if (not isinstance(board.get("center"), dict) or any(
                type(board["center"].get(axis)) not in (int, float)
                or not math.isfinite(board["center"][axis]) for axis in ("x", "y"))):
            raise ValueError("Nominal snap requires the recorded destination coordinate frame origin")
        fixture_name = board["component_attachment_model"]
        if (params["destination_location"] != "assembly_board-v1" or fixture_name != board["model_name"]
                or geometry["parts"]["model_map"].get(part_name) != params["model_name"]):
            raise ValueError("Nominal snap destination/fixture/model identity differs from its declaration")
        if board.get("mating_contact_by_part", {}).get(part_name):
            raise ValueError("Nominal mating snap requires its separately validated seated fixture handoff")
        target = ProductProfile.geometry_for_part_from_geometry(part_name, geometry)
        expected = {"slot_x": target["target_origin_pose"]["x"],
                    "slot_y": target["target_origin_pose"]["y"],
                    "part_origin_z": target["target_origin_pose"]["z"],
                    "board_top_z": target["slot_floor_z_m"], "part_height": target["part_height_m"]}
        if any(type(params[key]) not in (int, float) or not math.isfinite(params[key])
               or params[key] != value for key, value in expected.items()):
            raise ValueError("Nominal snap parameters differ from recorded destination geometry")
        physics = evidence["physics_evidence"]
        fixture = physics["models"][fixture_name]
        if (fixture.get("static") is not True or "link" not in fixture["links"]
                or "link" not in physics["models"][params["model_name"]]["links"]):
            raise ValueError("Nominal snap requires exact recorded part and static fixture links")
        before_support = projected_part_support(part_name=part_name, state=state, evidence=evidence)
        if before_support.get("status") != "supported":
            raise ValueError("Nominal snap cannot establish support for the preceding released interval: "
                             + before_support.get("reason", "support unavailable"))
        before_pose = list(_position(state["parts"][part_name]["current_pose"]).values())
        after_pose = [params["slot_x"], params["slot_y"], params["part_origin_z"], 0., 0., 0., 1.]
        candidate = deepcopy(state)
        candidate["parts"][part_name]["current_pose"] = after_pose
        after_support = projected_part_support(part_name=part_name, state=candidate, evidence=evidence)
        if after_support.get("status") != "supported":
            raise ValueError("Nominal snap corrected pose lacks independent static support: "
                             + after_support.get("reason", "support unavailable"))
        support_pairs = []
        collisions = {row["id"]: row for model in physics["models"].values()
                      for link in model.get("links", {}).values() for row in link.get("collisions", [])}
        for pair in after_support["source"]["contact_margins"]:
            point, normal = _plane(collisions[pair["collision2"]], physics["gravity"])
            support_pairs.append({"part_object_id": _object_id(pair["collision1"], evidence),
                "support_object_id": _object_id(pair["collision2"], evidence),
                "plane_point": point, "plane_normal": normal})
        changed = sorted({_object_id(row["id"], evidence)
            for link in physics["models"][params["model_name"]]["links"].values()
            for row in link.get("collisions", [])})
        if not changed or not support_pairs:
            raise ValueError("Nominal snap lacks exact collision-object/support identities")
        binding = {"state_fingerprint": recovery_validation_fingerprint(state),
            "candidate_state_fingerprint": recovery_validation_fingerprint(candidate),
            "params_fingerprint": recovery_validation_fingerprint(params),
            "owner_fingerprint": recovery_validation_fingerprint(owner_evidence),
            "models_fingerprint": recovery_validation_fingerprint({rid: {
                key: value for key, value in model.items() if key != "event_projection_contracts"}
                for rid, model in evidence["resource_models"].items()}),
            "physics_fingerprint": recovery_validation_fingerprint(physics),
            "scene_object_bindings_fingerprint": recovery_validation_fingerprint({
                "parts": evidence.get("scene_object_bindings"), "supports": evidence.get("support_collision_object_bindings")})}
        record = {"version": 1, "status": "prepared", "resource_id": resource_id,
            "command_sent": False, "execution_authorized": False, "changed_object_ids": changed,
            "source": {"kind": "predicted simulation snap_part_to_slot", "part_name": part_name,
                       "before_pose": before_pose, "after_pose": after_pose, "binding": binding},
            "support_pairs": support_pairs}
        candidate_result = {**result, "part_name": part_name, "current_pose": after_pose, "binding": binding,
            "candidate_state": candidate, "scene_endpoint_record": record,
            "pre_support_evidence": {key: deepcopy(value) for key, value in before_support.items() if key != "physics_evidence"},
            "part_support_evidence": {key: deepcopy(value) for key, value in after_support.items() if key != "physics_evidence"},
            "fixture_effects": {"fixture_model": fixture_name, "fixture_link": "link", "part_link": "link",
                                "temporarily_attached": True, "retained_fixture": False},
            "twist_after": {"linear": [0., 0., 0.], "angular": [0., 0., 0.]}}
        if endpoint_result is None:
            return {**candidate_result, "reason": "Nominal snap requires its independently checked private endpoint scene"}
        from cais_spade_llm.recovery_framework.recovery_motion_preview import (
            apply_declared_mounting_contacts, prepare_recorded_root_transforms,
            preview_result_record, project_preview_scene,
        )
        import json

        request = endpoint_result["request"]
        sent = json.loads(request["scene_endpoint_json"])
        if any(sent.get(key) != value for key, value in record.items()):
            raise ValueError("Nominal snap endpoint input differs from its calculated correction")
        scene_capture = evidence["planning_scene_capture"]
        scene_configuration = evidence["planning_scene_configuration"]
        physical = {**evidence, "snapshot": candidate}
        expected_scene = project_preview_scene(scene_capture, physical)
        if "planning_scene_mounting_contacts" in scene_configuration:
            mounted = apply_declared_mounting_contacts(expected_scene["planning_scene"],
                robot_description=scene_capture["robot_description"], robots=scene_configuration["robots"],
                physical_evidence=evidence, declarations=scene_configuration["planning_scene_mounting_contacts"])
            expected_scene["planning_scene"] = mounted["planning_scene"]
        roots = prepare_recorded_root_transforms(scene_capture, physical)["recorded_root_transforms"]
        group = config["move_group"]
        if (request.get("planning_scene") != expected_scene["planning_scene"]
                or request.get("start_state") != expected_scene["start_state"]
                or request.get("model_fingerprint") != scene_capture["model_fingerprint"]
                or request.get("recorded_root_transforms", []) != roots
                or any(request.get(field) != group[source] for field, source in (
                    ("group_name", "group_name"), ("link_name", "ee_link"), ("frame_id", "frame_id")))):
            raise ValueError("Nominal snap endpoint scene differs from its complete independently projected candidate")
        checked = preview_result_record(request, endpoint_result)
        audit = checked.get("scene_endpoint_evidence", {})
        if (checked.get("status") == "SCENE_ENDPOINT_REJECTED" and audit.get("pairs_complete") is True
                and audit.get("collision_free") is False and checked.get("command_sent") is False
                and checked.get("path_complete") is False and audit.get("changed_object_ids") == changed):
            return {**candidate_result, "status": "rejected", "reason": checked.get("reason"),
                "scene_endpoint_rejected": True, "endpoint_evidence": deepcopy(audit),
                "endpoint_request_fingerprint": recovery_validation_fingerprint(request)}
        if (checked.get("status") != "SCENE_ENDPOINT_COMPLETE" or checked.get("path_complete") is not False
                or checked.get("command_sent") is not False
                or checked.get("joint_trajectory", {}).get("points")
                or checked.get("joint_trajectory", {}).get("joint_names")
                or audit.get("pairs_complete") is not True or audit.get("collision_free") is not True
                or audit.get("robot_collision_checked") is not True or audit.get("robot_collision_free") is not True
                or audit.get("changed_object_ids") != changed):
            raise ValueError("Nominal snap endpoint collision evidence is incomplete or rejected")
        return {**candidate_result, "status": "supported", "projected_snapshot": candidate,
            "endpoint_evidence_fingerprint": recovery_validation_fingerprint(audit),
            "endpoint_request_fingerprint": recovery_validation_fingerprint(request),
            "observation_status": "predicted; runtime snap and position observations remain required"}
    except (KeyError, ValueError, TypeError, IndexError) as exc:
        return {**result, "reason": str(exc)}
