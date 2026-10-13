from __future__ import annotations

"""Resource-owned preparation contracts, command records and execution evidence.

Providers are registered by trusted application code. Scene documents contain
only identifiers and data; neither a document nor a verdict can install code.
"""

import json
import math
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, replace
from fractions import Fraction
from threading import RLock
from typing import Any
from uuid import uuid4

from cais_spade_llm.recovery_framework import fingerprint


@dataclass(frozen=True)
class PrimitiveModel:
    """A pure owner model with a serializable, versioned evidence contract.

    ``evaluate`` accepts exact primitive/params/evidence and start/end times. It
    returns trajectories, transfers and declared resource updates, never APs.
    Configuration must contain every fact on which the evaluator depends.
    """

    id: str
    version: int
    configuration: dict
    evaluate: Callable[..., dict]

    def descriptor(self) -> dict:
        """Return the frozen contract identity without executable objects."""
        if (not isinstance(self.id, str) or not self.id or type(self.version) is not int
                or self.version < 1 or not isinstance(self.configuration, dict)
                or not callable(self.evaluate)):
            raise ValueError("Invalid primitive model contract")
        result = {"id": self.id, "version": self.version,
                  "configuration": deepcopy(self.configuration)}
        json.dumps(result, allow_nan=False)
        return result

    def effects(self, *, primitive: str, params: dict, evidence: dict,
                start_time: Any, end_time: Any) -> dict:
        """Evaluate copied inputs and reject verdicts or unknown effect fields."""
        self.descriptor()
        if "primitive_contract" in evidence and evidence["primitive_contract"] != self.id:
            raise ValueError("Primitive evidence does not match its owner contract")
        result = self.evaluate(primitive=primitive, params=deepcopy(params),
                               evidence=deepcopy(evidence), start_time=start_time,
                               end_time=end_time, configuration=deepcopy(self.configuration))
        fields = {"trajectory", "base_trajectory", "part_trajectories", "transfers", "resource_updates"}
        if (not isinstance(result, dict) or not fields <= set(result)
                or set(result) - fields - {"continuous_motion", "custody_effects"}
                or "continuous_motion" in result and self.version < 2
                or "custody_effects" in result and self.version < 3):
            raise ValueError("Owner model must supply the complete supported effect record")
        if "custody_effects" in result and (not isinstance(result["custody_effects"], list)
                or any(not isinstance(row, dict) for row in result["custody_effects"])):
            raise ValueError("Custody effects require a list of explicit observation records")
        json.dumps(result, allow_nan=False)
        return deepcopy(result)


def prepare_recovery_outline_preview(*, resource_id: str, primitive_support: dict,
                                     projected_start: dict, isolated_scene: dict,
                                     primitive_model: PrimitiveModel | dict) -> dict:
    """Project owner-declared geometry without contacting live planning clients.

    Named joint configurations supply endpoints only. Recorded native motion
    needs the exact owner model; unsupported custody or missing interval
    evidence remains unresolved. No live scene, controller or cache changes.
    """
    from cais_spade_llm.resources.recovery_geometry import project_recovery_geometry

    result = {"status": "needs_context", "feasibility_status": "NEEDS_CONTEXT",
              "command_sent": False, "dispatch_authorized": False,
              "evidence_kind": "projected", "unresolved": []}
    try:
        scene = deepcopy(isolated_scene)
        snapshot = deepcopy(scene["snapshot"])
        if snapshot["resources"].get(resource_id) != projected_start:
            raise ValueError("Isolated scene and projected resource start disagree")
        model = primitive_model if isinstance(primitive_model, PrimitiveModel) else None
        descriptor = model.descriptor() if model is not None else deepcopy(primitive_model)
        configuration = descriptor["configuration"]
        models = deepcopy(scene["resource_models"])
        own = {"named_joint_poses": deepcopy(configuration.get("named_joint_poses", {}))}
        if isinstance(configuration.get("continuous_motion"), dict):
            own["configuration"] = deepcopy(configuration["continuous_motion"])
        saved = models.get(resource_id, {})
        if "configuration" in own and saved.get("configuration") != own["configuration"]:
            _validate_projected_gripper_configuration(own["configuration"], saved.get("configuration"),
                                                       configuration.get("custody"))
            own["configuration"] = deepcopy(saved["configuration"])
            configuration["continuous_motion"] = deepcopy(own["configuration"])
            if model is not None:
                model = replace(model, configuration=deepcopy(configuration))
        if any(saved.get(key, {}) != value for key, value in own.items()):
            raise ValueError("Isolated scene differs from the owner-declared resource model")
        evidence = deepcopy(scene.get("step_evidence") or {})
        if isinstance(evidence, list):
            evidence = dict(enumerate(evidence))
        projected = deepcopy(snapshot)
        projected_models = deepcopy(models)
        projection = None
        primitive_trace = [row for row in primitive_support.get("trace", []) if row.get("primitive") is not None]
        for index, command in enumerate(primitive_support.get("primitives", [])):
            primitive, params = command["primitive"], command["params"]
            trace = primitive_trace[index]
            if primitive == "move_to_named_pose":
                name = params.get("pose_name")
                if name not in own["named_joint_poses"]:
                    raise ValueError("Owner-declared named joint pose is unavailable")
                evidence[index] = {"primitive": primitive, "params": deepcopy(params),
                    "source": "owner-declared named_joint_poses", "validated": True,
                    "named_pose": name,
                    "target_joint_positions": deepcopy(own["named_joint_poses"][name])}
            elif primitive in {"grasp_part", "release_part"}:
                evidence[index] = _outline_custody_evidence(primitive=primitive, params=params,
                    snapshot=projected, resource_id=resource_id,
                    model=projected_models[resource_id], contract=configuration.get("custody"))
            elif (trace.get("declaration_fingerprint")
                    and isinstance(trace.get("declared_effects"), dict)
                    and set(trace["declared_effects"]) <= {"current_state", "resource_state"}
                    and {"primitive": primitive, "params": params} in configuration.get("geometry_unchanged", [])):
                evidence[index] = {"primitive": primitive, "params": deepcopy(params),
                    "validated": True, "source": "owner-declared geometry_unchanged",
                    "declaration_fingerprint": trace["declaration_fingerprint"], "geometry_unchanged": True}
            else:
                row = evidence.get(index, evidence.get(str(index)))
                if not isinstance(row, dict):
                    raise ValueError("Isolated owner motion or custody evidence is unavailable")
                if "joint_trajectory" not in row:
                    raise ValueError("Owner-declared custody preview model is unavailable")
                duration = row["joint_trajectory"]["duration_ns"] / 1_000_000_000
                if model is None:
                    if descriptor.get("id") != "joint_trajectory_continuous" or descriptor.get("version") != 2:
                        raise ValueError("Recorded continuous motion model is unsupported")
                    from cais_spade_llm.resources.robot.gazebo_pick_place_controller import _continuous_motion_effects
                    model = PrimitiveModel(descriptor["id"], descriptor["version"],
                                           configuration, _continuous_motion_effects)
                effects = model.effects(primitive=primitive, params=params,
                    evidence={"continuous_motion": {"configuration": own["configuration"],
                              "joint_trajectory": deepcopy(row["joint_trajectory"])}},
                    start_time=0., end_time=duration)
                if not effects.get("continuous_motion"):
                    raise ValueError("Owner model did not establish continuous motion")
                evidence[index] = {**row, "primitive": primitive, "params": deepcopy(params),
                                   "validated": True, "source": "owner-declared continuous_motion"}
            prefix = {**deepcopy(primitive_support),
                      "primitives": primitive_support["primitives"][:index + 1],
                      "trace": primitive_trace[:index + 1]}
            projection = project_recovery_geometry(resource_id=resource_id,
                snapshot=snapshot, geometry=scene["geometry"], resource_models=models,
                primitive_support=prefix, step_evidence=evidence)
            if projection.get("status") != "projected":
                break
            projected = deepcopy(projection["projected_snapshot"])
            projected_models = deepcopy(projection["resource_models"])
        if projection is None:
            raise ValueError("A resource-validated primitive transition witness is required")
        return {**projection, "command_sent": False, "dispatch_authorized": False,
                "evidence_kind": "projected", "step_evidence": evidence}
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        result["unresolved"].append(str(exc))
        return result


def _validate_projected_gripper_configuration(before: dict, after: dict | None,
                                             contract: dict | None) -> None:
    """Allow only the exact declared open/closed frozen gripper joint states."""
    if not isinstance(after, dict) or not isinstance(contract, dict):
        raise ValueError("Isolated scene differs from the owner-declared resource model")
    for positions in contract.get("joint_positions", {}).values():
        expected = deepcopy(before)
        for name, value in positions.items():
            joint = next((row for row in expected["joints"] if row["name"] == name), None)
            if joint is None or name in expected["joint_names"] or "frozen_position" not in joint:
                break
            joint["frozen_position"] = value
            joint.pop("frozen_interval", None)
            if name in expected.get("fixed_joint_positions", {}):
                expected["fixed_joint_positions"][name] = value
        else:
            if after == expected:
                return
    raise ValueError("Isolated scene differs from the owner-declared resource model")


def _outline_custody_evidence(*, primitive: str, params: dict, snapshot: dict,
                              resource_id: str, model: dict, contract: dict | None) -> dict:
    """Project the successful native custody effect, without predicting settling."""
    from cais_spade_llm.recovery_framework.geometry import multiply, rotate
    from cais_spade_llm.resources.primitive_observations_3d import _unit_pose

    if (not isinstance(contract, dict) or contract.get("kind") != "gazebo_link_attacher"
            or contract.get("preserves_part_pose_on_attachment") is not True):
        raise ValueError("Owner-declared custody preview model is unavailable")
    allowed = {"model_name", "part_name", "position"} if primitive == "grasp_part" else {
        "model_name", "part_name", "assume_released_if_open"}
    if set(params) - allowed or params.get("assume_released_if_open") is True:
        raise ValueError("Release correction or fallback has no declared custody preview model")
    owner = snapshot["resources"][resource_id]
    part = params.get("part_name") or (owner.get("held_part") if primitive == "release_part" else None)
    if part not in snapshot["parts"]:
        raise ValueError("Custody preview requires an exact recorded part identity")
    row = snapshot["parts"][part]
    if not row.get("model_name") or (params.get("model_name") and params["model_name"] != row["model_name"]):
        raise ValueError("Custody preview model_name differs from its recorded part")
    if primitive == "grasp_part" and not params.get("model_name"):
        raise ValueError("Grasp preview requires the recorded controller model_name")
    state = "closed" if primitive == "grasp_part" else "open"
    if params.get("position") is not None and params["position"] != contract["gripper"][state]:
        raise ValueError("Gripper position override has no declared geometry model")
    tool = [float(value) for value in _unit_pose(owner["current_pose"], "projected tool pose")]
    pose = [float(value) for value in _unit_pose(row["current_pose"], "recorded part pose")]
    effect = {"kind": "grasp" if primitive == "grasp_part" else "release", "part": part, "pose": pose}
    if primitive == "grasp_part":
        inverse = [-tool[3], -tool[4], -tool[5], tool[6]]
        effect["grasp_transform"] = [*rotate(inverse, [pose[i] - tool[i] for i in range(3)]),
                                      *multiply(inverse, pose[3:])]
    else:
        effect["contained_by"] = None
    configuration = deepcopy(model["configuration"])
    positions = contract["joint_positions"][state]
    for name, value in positions.items():
        joint = next((item for item in configuration["joints"] if item["name"] == name), None)
        if joint is None or name in configuration["joint_names"] or "frozen_position" not in joint:
            raise ValueError("Configured gripper geometry cannot be projected")
        joint["frozen_position"] = value
        joint.pop("frozen_interval", None)
        if name in configuration.get("fixed_joint_positions", {}):
            configuration["fixed_joint_positions"][name] = value
    return {"primitive": primitive, "params": deepcopy(params), "validated": True,
            "source": "owner-declared Gazebo attachment endpoint", "custody_effects": [effect],
            "configuration_after": configuration, "interval_complete": False}


def _outline_pick_departure(*, task: dict, declarations: list, model: dict,
                            target: dict, current_pose: dict) -> dict:
    """Resolve the owner's declared post-grasp branch from calculated facts."""
    from cais_spade_llm.resources.recovery_parameter_evidence import (
        _resolve, evaluate_resource_primitive_condition,
    )
    from cais_spade_llm.agents.shared_information.recovery_validation_protocol import recovery_validation_fingerprint

    state = {"_task_ctx": deepcopy(target), "_position": deepcopy(current_pose)}
    resolved = []
    for declaration in declarations:
        event = next(row for row in model["events"] if row["event_name"] == declaration["event_name"])
        steps = event.get("composition_contract", {}).get("steps")
        if not isinstance(steps, list) or not steps or steps[0].get("op") != "grasp_part":
            raise ValueError("Pickup preview requires a declared grasp and departure contract")
        selected = []
        for step in steps[1:]:
            check = evaluate_resource_primitive_condition(step=step, bindings=declaration["parameter_bindings"],
                snapshot={}, resource_evidence={"state": state}, resource_jid=task["resource_jid"])
            if check["status"] == "not_applicable":
                continue
            if check["status"] != "applicable":
                raise ValueError("Pickup departure branch requires recorded condition evidence")
            known, params, _ = _resolve(step.get("params"), declaration["parameter_bindings"], state,
                                        "/resources/" + task["resource_jid"])
            if not known or not isinstance(params, dict):
                raise ValueError("Pickup departure parameters are unresolved")
            pose = deepcopy(current_pose)
            if step.get("op") == "move_relative":
                for axis in ("x", "y", "z"):
                    delta = params.get("d" + axis)
                    if type(delta) not in (float, int) or not math.isfinite(delta):
                        raise ValueError("Pickup departure requires finite declared displacements")
                    pose[axis] += delta
            elif step.get("op") == "move_cartesian":
                for axis in ("x", "y", "z", "qx", "qy", "qz", "qw"):
                    value = params.get(axis)
                    if type(value) not in (float, int) or not math.isfinite(value):
                        raise ValueError("Pickup departure requires its complete declared retreat pose")
                    pose[axis] = value
            else:
                raise ValueError("Pickup departure uses unsupported owner motion semantics")
            selected.append({"step_id": step["id"], "op": step["op"], "params": params,
                "pose": pose, "step_fingerprint": recovery_validation_fingerprint(step)})
        if len(selected) != 1:
            raise ValueError("Pickup preview requires one resolved declared departure branch")
        resolved.append(selected[0])
    if not resolved or any(row != resolved[0] for row in resolved):
        raise ValueError("Pickup declarations do not establish a consistent departure")
    return resolved[0]


def _outline_cartesian_request(poses: list[dict]) -> dict:
    return {"mode": "cartesian", "waypoints": [{
        "position": {key: pose[key] for key in ("x", "y", "z")},
        "orientation": dict(zip(("x", "y", "z", "w"),
                                (pose[key] for key in ("qx", "qy", "qz", "qw")), strict=True))} for pose in poses]}


def _outline_pick_arrival(*, task: dict, declarations: list, model: dict,
                          target: dict, snapshot: dict, resource_evidence: dict,
                          owner: dict) -> list[dict]:
    """Resolve the declared approach without inferring a named joint pose."""
    from cais_spade_llm.resources.recovery_parameter_evidence import _resolve
    from cais_spade_llm.resources.robot.recovery_task_projection import nominal_step_condition

    alternatives = []
    for declaration in declarations:
        event = next(row for row in model["events"] if row["event_name"] == declaration["event_name"])
        steps = event.get("composition_contract", {}).get("steps")
        if not isinstance(steps, list) or not steps:
            raise ValueError("Pickup approach requires its declared calculation and motion steps")
        outputs, poses = {}, []
        for step in steps:
            check = nominal_step_condition(step=step, bindings=declaration["parameter_bindings"],
                state=resource_evidence.get("state") or {}, resource_evidence=resource_evidence,
                step_outputs=outputs)
            if check["status"] == "not_applicable":
                continue
            if check["status"] != "applicable":
                raise ValueError("Pickup approach requires its declared branch evidence")
            if step.get("op") == "detect_parts":
                continue
            if step.get("op") == "compute_pick_targets" and isinstance(step.get("store_as"), str):
                if step.get("params", {}).get("part_name") != {"$arg": "part_name"}:
                    raise ValueError("Pickup calculation has unsupported part bindings")
                outputs[step["store_as"]] = deepcopy(target)
            elif step.get("op") == "open_gripper":
                if owner.get("gripper_state") != "open":
                    raise ValueError("Pickup approach requires a supported gripper opening projection")
            elif step.get("op") == "move_cartesian":
                fields = ("x", "y", "z", "qx", "qy", "qz", "qw")
                params = step.get("params") or {}
                if not set(fields) <= set(params) or set(params) - {*fields, "speed"}:
                    raise ValueError("Pickup approach has unsupported motion parameters")
                known, pose, _ = _resolve({key: params[key] for key in fields},
                    declaration["parameter_bindings"], resource_evidence.get("state") or {},
                    "/resources/" + task["resource_jid"], outputs)
                if not known or any(type(value) not in (float, int) or not math.isfinite(value)
                                    for value in pose.values()):
                    raise ValueError("Pickup approach waypoint lacks recorded calculation inputs")
                poses.append(pose)
            else:
                raise ValueError("Pickup approach uses unsupported owner operation semantics")
        if not poses or not outputs:
            raise ValueError("Pickup approach lacks its calculation or motion declaration")
        alternatives.append(poses)
    if not alternatives or any(row != alternatives[0] for row in alternatives):
        raise ValueError("Pickup approach declarations have conflicting waypoints")
    return alternatives[0]


def _outline_place_arrival(*, task: dict, model: dict, target: dict,
                           resource_evidence: dict, release: bool) -> dict:
    """Resolve recorded placement waypoints without ordering primitive support."""
    from cais_spade_llm.resources.robot.target_calculations import placement_poses
    from cais_spade_llm.agents.shared_information.recovery_validation_protocol import recovery_validation_fingerprint

    depth = resource_evidence.get("controller_config", {}).get("parts_tuning", {}).get("insertion_depth_m")
    if type(depth) not in (float, int) or not math.isfinite(depth) or depth < 0:
        raise ValueError("Assembly placement requires its recorded insertion depth")
    pose = target["insert_pose"]
    poses = placement_poses(pose["x"], pose["y"], pose["z"],
        {key: pose[key] for key in ("qx", "qy", "qz", "qw")},
        simulation_assembly_slot=True, insertion_depth=depth)
    if release and task.get("required_task_contract"):
        return {"motion_requests": [_outline_cartesian_request([poses["insert_pose"]])],
            "mating_motion_indices": [0], "poses": poses,
            "source": {"kind": "calculated", "calculation": "placement_poses",
                       "required_task_contract": task["required_task_contract"],
                       "insertion_depth_m": depth}}
    contributors = []
    for event in model.get("events", []):
        steps = event.get("composition_contract", {}).get("steps", [])
        calculators = [(index, row) for index, row in enumerate(steps)
                       if row.get("op") == "compute_place_targets" and not row.get("when")]
        for index, calculator in calculators:
            source = calculator.get("store_as")
            following = steps[index + 1:]
            if not source or len(following) != 2:
                continue
            if all(row.get("op") == "move_cartesian" and not row.get("when")
                   and all(row.get("params", {}).get(key) == {"$step": source, "path": [name, key]}
                           for key in ("x", "y", "z", "qx", "qy", "qz", "qw"))
                   for row, name in zip(following, ("approach_pose", "target_pose"), strict=True)):
                contributors.append({"event_name": event["event_name"],
                    "event_fingerprint": recovery_validation_fingerprint(event),
                    "step_fingerprints": [recovery_validation_fingerprint(row)
                                          for row in (calculator, *following)]})
    if not contributors:
        raise ValueError("Assembly approach lacks its recorded target-calculation and waypoint declaration")
    descent = [poses["pre_insert_pose"]]
    if release:
        descent.append(poses["insert_pose"])
    return {"motion_requests": [_outline_cartesian_request([poses["approach_pose"]]),
                                _outline_cartesian_request(descent)],
        "mating_motion_indices": [1], "poses": poses,
        "source": {"kind": "calculated", "calculation": "placement_poses",
                   "insertion_depth_m": depth, "owner_declarations": contributors}}


def _outline_place_departure(*, task: dict, declarations: list, model: dict,
                             target: dict, current_pose: dict) -> dict:
    """Resolve the declared final placement motion using the same part's context."""
    from cais_spade_llm.resources.recovery_parameter_evidence import _resolve, evaluate_resource_primitive_condition
    from cais_spade_llm.agents.shared_information.recovery_validation_protocol import recovery_validation_fingerprint

    context = target.get("departure_context")
    source = target.get("departure_context_source", {})
    nominal_context = source.get("kind") == "nominal_task"
    if (not isinstance(context, dict)
            or not nominal_context and context.get("part_name") != task.get("part_name")):
        raise ValueError("Placement departure requires context for the proposed part")
    if nominal_context and (not task.get("required_task_contract") or source != {
            "kind": "nominal_task", "task_contract": task["required_task_contract"],
            "context_fingerprint": recovery_validation_fingerprint(context)}):
        raise ValueError("Nominal departure differs from its recorded task context")
    state = {"_task_ctx": deepcopy(context), "_position": deepcopy(current_pose)}
    resolved = []
    for declaration in declarations:
        event = next(row for row in model["events"] if row["event_name"] == declaration["event_name"])
        steps = event.get("composition_contract", {}).get("steps")
        if (not isinstance(steps, list) or not steps
                or not any(step.get("op") == "release_part" for step in steps[:-1])):
            raise ValueError("Placement departure requires a declared release followed by motion")
        step = steps[-1]
        check = evaluate_resource_primitive_condition(step=step, bindings=declaration["parameter_bindings"],
            snapshot={}, resource_evidence={"state": state}, resource_jid=task["resource_jid"])
        if check["status"] != "applicable":
            raise ValueError("Placement departure branch requires recorded condition evidence")
        known, params, _ = _resolve(step.get("params"), declaration["parameter_bindings"], state,
                                    "/resources/" + task["resource_jid"])
        if not known or not isinstance(params, dict) or step.get("op") != "move_relative":
            raise ValueError("Placement departure requires resolved declared relative motion")
        pose = deepcopy(current_pose)
        for axis in ("x", "y", "z"):
            delta = params.get("d" + axis)
            if type(delta) not in (float, int) or not math.isfinite(delta):
                raise ValueError("Placement departure requires finite declared displacements")
            pose[axis] += delta
        resolved.append({"step_id": step["id"], "op": step["op"], "params": params, "pose": pose,
            "step_fingerprint": recovery_validation_fingerprint(step),
            "context_source": deepcopy(target["departure_context_source"])})
    if not resolved or any(row != resolved[0] for row in resolved):
        raise ValueError("Placement declarations do not establish a consistent departure")
    return resolved[0]


def _outline_nominal_place_targets(*, task: dict, recovery_snapshot: dict,
                                   resource_evidence: dict, physical_evidence: dict,
                                   params: dict) -> dict:
    """Calculate the simulation output used by the recorded placement program."""
    from cais_spade_llm.resources.robot.recovery_capability import calculate_recovery_place_target
    from cais_spade_llm.resources.robot.target_calculations import placement_poses
    from cais_spade_llm.recovery_framework.workflow_execution import _destination_geometry
    from types import SimpleNamespace

    if resource_evidence.get("execution_mode") != "simulation":
        raise ValueError("Nominal placement output requires its supported simulation calculation")
    if (set(params) != {"pick_ctx", "product_geometry", "part_name", "z_adjustment_m",
                       "destination_location", "assembly_board_v1_aruco"}
            or params["part_name"] != task.get("part_name")
            or params["destination_location"] != task.get("expected_end_state", {}).get("resource_location")
            or type(params["z_adjustment_m"]) not in (int, float) or params["z_adjustment_m"] != 0.
            or params["pick_ctx"] != resource_evidence.get("state", {}).get("_task_ctx")
            or params["assembly_board_v1_aruco"] is not None):
        raise ValueError("Nominal placement arguments differ from the supported recorded calculation")
    inputs = resource_evidence.get("inputs", {})
    parts = inputs.get("geometry", {}).get("parts", {})
    part = task["part_name"]
    dimensions, model_name = parts.get("dimensions_m", {}).get(part), parts.get("model_map", {}).get(part)
    if (not isinstance(dimensions, (list, tuple)) or len(dimensions) != 3
            or any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0 for value in dimensions)
            or not isinstance(model_name, str) or not model_name):
        raise ValueError("Nominal placement geometry lacks recorded dimensions or model identity")
    owners = [name for name, jid in physical_evidence.get("jids", {}).items() if jid == task["resource_jid"]]
    if len(owners) != 1:
        raise ValueError("Nominal placement geometry has no exact recorded owner")
    recorded_geometry = _destination_geometry(SimpleNamespace(inputs=inputs,
        product_name=physical_evidence.get("product_name"),
        geometry={part: {"dimensions_m": dimensions, "model_name": model_name}}),
        part, params["destination_location"], owners[0])
    if params["product_geometry"] != recorded_geometry:
        raise ValueError("Nominal product_geometry differs from its recorded owner geometry")
    target = calculate_recovery_place_target(task=task, recovery_snapshot=recovery_snapshot,
        resource_evidence=resource_evidence, physical_evidence=physical_evidence)
    if target.get("status") != "calculated":
        raise ValueError("Nominal placement calculation has missing source facts: " + str(target.get("missing_facts")))
    depth = resource_evidence.get("controller_config", {}).get("parts_tuning", {}).get("insertion_depth_m")
    if type(depth) not in (int, float) or not math.isfinite(depth) or depth < 0:
        raise ValueError("Nominal placement requires its recorded insertion depth")
    pose = target["insert_pose"]
    assembly = target["target_reference"]["surface_role"] == "assembly_slot"
    poses = placement_poses(pose["x"], pose["y"], pose["z"],
        {key: pose[key] for key in ("qx", "qy", "qz", "qw")},
        simulation_assembly_slot=assembly, insertion_depth=depth)
    output = {"success": True, "part_name": part, "destination_location": target["destination_location"],
        "slot_x": target["slot_x"], "slot_y": target["slot_y"], "board_top_z": target["board_top_z"],
        "place_z": pose["z"], "place_part_origin_z": target["place_part_origin_z"],
        "approach_pose": poses["approach_pose"], "target_pose": poses["target_pose"],
        "pre_insert_pose": poses["pre_insert_pose"], "insert_pose": poses["insert_pose"],
        "insertion_axis_world": {"x": 0., "y": 0., "z": -1.}, "part_height": target["part_height"],
        "target_reference": deepcopy(target["target_reference"]), "surface_role": target["target_reference"]["surface_role"],
        "move_insert_mode": "simulation_direct" if assembly else "",
        "target_origin_pose": deepcopy(target["target_origin_pose"]),
        "model_name": physical_evidence["snapshot"]["parts"][part]["model_name"]}
    return {"output": output, "target_calculation": target,
        "source": {"kind": "calculated", "calculation": "calculate_recovery_place_target, placement_poses",
                   "branch": "simulation", "source": deepcopy(resource_evidence.get("source"))},
        "absent_output_fields": ["move_insert_profile", "move_insert_profile_sha256", "move_insert_timeout_sec",
            "assembly_board_v1_pose", "assembly_board_v1_registration"]}


def _outline_nominal_program(*, task: dict, recovery_snapshot: dict, recovery_des_model: dict,
                             physical_evidence: dict, resource_evidence: dict,
                             intervals: list | None = None, interval_states: list | None = None) -> dict:
    """Resolve an existing nominal program from owner facts and interval replay."""
    from cais_spade_llm.agents.shared_information.recovery_validation_protocol import recovery_validation_fingerprint
    from cais_spade_llm.resources.robot.recovery_task_projection import nominal_task_bindings, resolve_nominal_task_step

    matches = [row for row in recovery_des_model.get("events", [])
               if row.get("event_name") == task.get("required_task_contract")]
    if len(matches) != 1 or not isinstance(matches[0].get("composition_contract", {}).get("steps"), list):
        raise ValueError("Nominal projection requires its exact recorded program")
    program = matches[0]["composition_contract"]
    rid = next(name for name, jid in physical_evidence["jids"].items() if jid == task["resource_jid"])
    owner = physical_evidence["snapshot"]["resources"][rid]
    state = deepcopy(resource_evidence.get("state") or {})
    fields = ("x", "y", "z", "qx", "qy", "qz", "qw")
    state["_position"] = dict(zip(fields, owner["current_pose"], strict=True))
    state["_held_part"] = owner["held_part"]
    state["_current_state"] = recovery_snapshot.get("resource_state", recovery_snapshot.get("current_state"))
    binding_record = nominal_task_bindings(task=task, recovery_des_model=recovery_des_model)
    if binding_record.get("status") != "supported":
        raise ValueError(binding_record.get("reason", "Nominal arguments are unresolved"))
    bindings = binding_record["bindings"]
    if not isinstance(state.get("_task_ctx"), dict) and owner.get("held_part") == task.get("part_name"):
        previous = physical_evidence.get("resources", {}).get(rid, {}).get("validated_pickup_target_calculation")
        if isinstance(previous, dict):
            recorded = physical_evidence["resource_models"][rid].get("event_projection_contracts", {}).get(previous.get("contract_id"))
            calculation = previous.get("calculation")
            if (not isinstance(recorded, dict) or not isinstance(calculation, dict)
                    or previous.get("source") != "resource-owned pickup preview"
                    or previous.get("binding", {}).get("resource_jid") != task["resource_jid"]
                    or previous.get("contract_fingerprint") != recovery_validation_fingerprint(recorded)
                    or recorded.get("pickup_target_calculation") != calculation
                    or calculation.get("part_name") != task.get("part_name")):
                raise ValueError("Nominal pickup context differs from its accepted owner calculation")
            state["_task_ctx"] = deepcopy(calculation)
            state["_task_ctx"]["model_name"] = physical_evidence["snapshot"]["parts"][task["part_name"]]["model_name"]
    motions, rows, outputs, calculations, absent, resolved_positions = [], [], {}, {}, {}, {}
    custody_index, nominal_snap, cursor = None, None, 0
    if intervals is not None and (not isinstance(interval_states, list) or len(interval_states) != len(intervals) + 1):
        raise ValueError("Nominal coverage requires every independently replayed interval checkpoint")
    for step in program["steps"]:
        if intervals is not None:
            current = interval_states[cursor]
            state["_position"] = dict(zip(fields, current["current_pose"], strict=True))
            state["_held_part"] = current["held_part"]
        before, saved_outputs = deepcopy(state), deepcopy(outputs)
        resolved = resolve_nominal_task_step(step=step, bindings=bindings, state=state,
            resource_evidence=resource_evidence, step_outputs=outputs)
        row = {"step_id": step["id"], "state_before": before, "step_outputs": saved_outputs,
               "interval_indices": []}
        if resolved["status"] == "skipped":
            if step.get("store_as"):
                outputs[step["store_as"]] = None
            rows.append(row)
            continue
        if resolved["status"] != "resolved":
            raise ValueError("Nominal step " + step["id"] + " is unresolved: "
                + str({key: resolved[key] for key in ("reason", "missing_facts", "condition") if key in resolved}))
        params, op = resolved["params"], step["op"]
        if op == "compute_place_targets":
            calculated = _outline_nominal_place_targets(task=task, recovery_snapshot=recovery_snapshot,
                resource_evidence={**resource_evidence, "state": deepcopy(state)},
                physical_evidence=physical_evidence, params=params)
            if not step.get("store_as"):
                raise ValueError("Nominal target calculation has no declared result reference")
            row["effect_evidence"] = {"calculation": op, "params": deepcopy(params),
                "output": deepcopy(calculated["output"]), "source": deepcopy(calculated["source"]),
                "input_fingerprint": recovery_validation_fingerprint({"task": task, "snapshot": recovery_snapshot,
                    "resource_evidence": resource_evidence, "target": calculated["target_calculation"]})}
            outputs[step["store_as"]] = deepcopy(calculated["output"])
            calculations[step["store_as"]] = deepcopy(row["effect_evidence"])
            absent[step["store_as"]] = deepcopy(calculated["absent_output_fields"])
        elif op in {"move_cartesian", "move_relative", "move_insert", "move_to_named_pose"}:
            if op == "move_to_named_pose":
                target = physical_evidence["resource_models"][rid]["named_joint_poses"][params["pose_name"]]
                motion = {"mode": "joint_target", "joint_target": {
                    "name": deepcopy(target["joint_names"]), "position": deepcopy(target["positions"])}}
            else:
                if op == "move_relative":
                    pose = deepcopy(state["_position"])
                    for axis in ("x", "y", "z"):
                        delta = params["d" + axis]
                        if type(delta) not in (int, float) or not math.isfinite(delta):
                            raise ValueError("Nominal relative motion has no finite declared displacement")
                        pose[axis] += delta
                else:
                    pose = {key: params[key] for key in fields} if op == "move_cartesian" else params["target_pose"]
                motion = _outline_cartesian_request([pose])
                if intervals is None:
                    state["_position"] = deepcopy(pose)
            motions.append(motion)
            row["interval_indices"] = [cursor]
            if op == "move_cartesian":
                resolved_positions[step["id"]] = (deepcopy(pose) if intervals is None else
                    dict(zip(fields, interval_states[cursor + 1]["current_pose"], strict=True)))
            cursor += 1
        elif op in {"grasp_part", "release_part"}:
            if custody_index is not None:
                raise ValueError("Nominal owner preview supports one declared custody change")
            custody_index = len(motions)
            row["interval_indices"] = [cursor]
            cursor += 1
            if intervals is None:
                state["_held_part"] = params["part_name"] if op == "grasp_part" else None
        elif op == "snap_part_to_slot":
            if nominal_snap is not None:
                raise ValueError("Nominal owner preview supports one declared simulation correction")
            nominal_snap = {"step_id": step["id"], "step_fingerprint": recovery_validation_fingerprint(step),
                "params": deepcopy(params), "owner_evidence": deepcopy(resource_evidence)}
            row["interval_indices"] = [cursor]
            cursor += 1
        else:
            raise ValueError("Nominal step " + step["id"] + " has no supported owner effect model for " + op)
        rows.append(row)
    if intervals is not None:
        if cursor != len(intervals):
            raise ValueError("Nominal owner preview contains intervals outside its exact program")
        state["_position"] = dict(zip(fields, interval_states[-1]["current_pose"], strict=True))
        state["_held_part"] = interval_states[-1]["held_part"]
    return {"motion_requests": motions, "custody_motion_index": custody_index, "nominal_snap": nominal_snap,
        "step_evidence": rows, "state": state, "step_outputs": outputs,
        "calculation_evidence": calculations, "absent_output_fields": absent,
        "resolved_cartesian_positions": resolved_positions}


def _outline_nominal_task_effects(*, task: dict, recovery_des_model: dict,
                                  resource_evidence: dict, nominal: dict) -> dict:
    """Carry the owner's existing bookkeeping from independently replayed endpoints."""
    from dataclasses import asdict
    import json
    from cais_spade_llm.resources.robot.recovery_task_projection import apply_nominal_task_effects
    from cais_spade_llm.resources.robot.robot_task_registry import robot_task_registry

    result = apply_nominal_task_effects(task=task, recovery_des_model=recovery_des_model,
        resource_evidence=resource_evidence, state=nominal["state"], step_outputs=nominal["step_outputs"],
        absent_output_fields=nominal["absent_output_fields"])
    if result["status"] != "supported" or not nominal["resolved_cartesian_positions"]:
        return result
    registered = robot_task_registry().get(task.get("required_task_contract"))
    declaration = next(row for row in recovery_des_model["events"]
                       if row["event_name"] == task["required_task_contract"])
    if (registered is None or resource_evidence.get("execution_mode") != "simulation"
            or declaration["composition_contract"]["steps"] != json.loads(json.dumps(asdict(registered.program)))["steps"]):
        return {**result, "status": "NEEDS_CONTEXT", "reason": "Nominal Cartesian bookkeeping lacks its exact recorded owner program"}
    from cais_spade_llm.resources.robot.robot_task_runtime import _apply_resolved_cartesian_state

    position = deepcopy(result["state"]["_position"])
    error = _apply_resolved_cartesian_state(task=registered, physical_overrides={}, computed_positions={},
        computed_reference={}, computed_at=0., resolved_positions=nominal["resolved_cartesian_positions"],
        runtime_state=result["state"], step_outputs=nominal["step_outputs"])
    result["state"]["_position"] = position
    if error:
        return {**result, "status": "NEEDS_CONTEXT", "reason": error}
    result["cartesian_bookkeeping"] = {"source": "robot_task_runtime._apply_resolved_cartesian_state",
        "resolved_positions": deepcopy(nominal["resolved_cartesian_positions"]),
        "physical_pose_replaced": False}
    return result


def prepare_outline_event_preview_request(*, task: dict, recovery_snapshot: dict,
                                         recovery_des_model: dict, physical_evidence: dict,
                                         resource_evidence: dict, static_capabilities: dict | None = None) -> dict:
    """Calculate an owner motion target without producing a primitive program.

    The target follows recorded destination geometry or a declared named joint
    configuration. Only declared resource effects are used; product conditions
    remain obligations unless the physical evidence independently supports them.
    """
    from cais_spade_llm.agents.shared_information.recovery_validation_protocol import recovery_validation_fingerprint
    from cais_spade_llm.resources.recovery_feasibility import derive_primitive_union
    from cais_spade_llm.resources.recovery_geometry import (
        outline_event_binding, owner_cartesian_endpoint_precision, rejected_owner_event_effects,
    )
    from cais_spade_llm.resources.recovery_parameter_evidence import projected_resource_evidence

    result = {"status": "NEEDS_CONTEXT", "command_sent": False, "execution_authorized": False,
              "missing_facts": [], "motion_requests": []}
    resource_evidence_fingerprint = recovery_validation_fingerprint(resource_evidence)
    try:
        rejected = rejected_owner_event_effects(task=task, recovery_snapshot=recovery_snapshot,
            recovery_des_model=recovery_des_model, physical_evidence=physical_evidence)
        if rejected is not None:
            return rejected
        scene = deepcopy(physical_evidence)
        ids = [rid for rid, jid in scene["jids"].items() if jid == task["resource_jid"]]
        if len(ids) != 1:
            raise ValueError("The event resource has no unique captured physical identity")
        rid = ids[0]
        owner = scene["snapshot"]["resources"][rid]
        model = scene["resource_models"][rid]
        if scene.get("physical_preview_contract_version") in {4, 5, 6, 7}:
            captured_capabilities = model.get("static_capabilities")
            if not isinstance(captured_capabilities, dict):
                raise ValueError("Owner static_capabilities were not recorded in this attempt's physical model")
            if static_capabilities is not None and captured_capabilities != static_capabilities:
                raise ValueError("Owner static_capabilities differ from this attempt's recorded declarations")
        resource_evidence = projected_resource_evidence(resource_jid=task["resource_jid"],
            resource_evidence=resource_evidence, physical_evidence=scene, recovery_snapshot=recovery_snapshot)
        support = derive_primitive_union(task=task, recovery_des_model=recovery_des_model, recovery_snapshot=recovery_snapshot)
        candidates = support.get("matching_transitions") or []
        if task.get("required_task_contract"):
            candidates = [row for row in candidates if row["event_name"] == task["required_task_contract"]]
        if not candidates:
            raise ValueError("Owner event projection has no supporting resource declaration")
        declarations = recovery_des_model["state_variables"]
        updates = {}
        declaration_records = []
        for match in candidates:
            event = next(row for row in recovery_des_model["events"] if row["event_name"] == match["event_name"])
            values = {}
            for field in support["covered_valuation_fields"]:
                effect = event.get("updates", {}).get(field)
                if effect is None:
                    keys = {"resource_state": ("resource_state", "current_state"),
                            "resource_location": ("resource_location", "current_location")}.get(field, (field,))
                    present = [key for key in keys if key in recovery_snapshot]
                    if not present:
                        raise ValueError(f"Preserved resource condition {field} is not recorded")
                    value = recovery_snapshot[present[0]]
                elif set(effect) == {"set"}:
                    value = effect["set"]
                elif set(effect) == {"set_from_param"} and effect["set_from_param"] in match["parameter_bindings"]:
                    value = match["parameter_bindings"][effect["set_from_param"]]
                else:
                    raise ValueError(f"Owner event effect {field} requires unsupported semantics")
                if field not in declarations or declarations[field].get("scope", "resource") != "resource":
                    raise ValueError("Event target requires a public resource declaration")
                if type(value) is not type(task["expected_end_state"][field]) or value != task["expected_end_state"][field]:
                    raise ValueError(f"Owner event cannot establish the proposed {field}")
                values[field] = {"set": deepcopy(value)}
            if updates and values != updates:
                raise ValueError("Matching owner event declarations have incompatible effects")
            updates = values
            declaration_records.append({"event_name": event["event_name"],
                "fingerprint": recovery_validation_fingerprint(event),
                "parameter_bindings": deepcopy(match["parameter_bindings"])})
        end = task["expected_end_state"]
        destination = end.get("resource_location", end.get("current_location"))
        part = task.get("part_name")
        endpoint, bindings, motions, custody = {}, {}, [], None
        departure, placement_departure, custody_motion_index = None, None, None
        placement_arrival = None
        private_updates, carried_updates = {}, {}
        if "held_part" in end and end["held_part"] is not None and owner.get("held_part") is None:
            if end["held_part"] != part:
                raise ValueError("The proposed pickup does not name the requested part")
            from cais_spade_llm.resources.robot.recovery_capability import calculate_recovery_pick_target

            target = calculate_recovery_pick_target(task=task, recovery_snapshot=recovery_snapshot,
                                                    resource_evidence=resource_evidence)
            if target.get("status") != "calculated":
                return {**result, "missing_facts": deepcopy(target.get("missing_facts") or []),
                        "context_requests": deepcopy(target.get("context_requests") or []),
                        "reason": "Recorded geometry and pick policy cannot establish the requested pickup target"}
            departure = _outline_pick_departure(task=task, declarations=declaration_records,
                model=recovery_des_model, target=target, current_pose=target["target_pose"])
            values = [departure["pose"][key] for key in ("x", "y", "z", "qx", "qy", "qz", "qw")]
            bindings["target_pose"] = values
            endpoint["current_pose"] = {"set_from_param": "target_pose"}
            motions.append({"mode": "cartesian", "waypoints": [{
                "position": {key: pose[key] for key in ("x", "y", "z")},
                "orientation": dict(zip(("x", "y", "z", "w"),
                    (pose[key] for key in ("qx", "qy", "qz", "qw")), strict=True))}
                for pose in (target["approach_pose"], target["target_pose"])]})
            custody = {"kind": "grasp", "part_name": part,
                       "model_name": scene["snapshot"]["parts"][part]["model_name"],
                       "target_calculation": deepcopy(target)}
            custody_motion_index = len(motions)
            motions.append(_outline_cartesian_request([departure["pose"]]))
        elif ("held_part" in end and end["held_part"] is None and owner.get("held_part") is not None):
            if owner["held_part"] != part:
                raise ValueError("The proposed release does not name the recorded held part")
            from cais_spade_llm.resources.robot.recovery_capability import calculate_recovery_place_target

            target = calculate_recovery_place_target(task=task, recovery_snapshot=recovery_snapshot,
                                                    resource_evidence=resource_evidence, physical_evidence=scene)
            if target.get("status") != "calculated":
                return {**result, "missing_facts": deepcopy(target.get("missing_facts") or []),
                        "context_requests": deepcopy(target.get("context_requests") or []),
                        "reason": "Recorded geometry cannot establish the requested deposit target"}
            pose = target["insert_pose"]
            context = resource_evidence.get("state", {}).get("_task_ctx")
            nominal_context = scene.get("physical_preview_contract_version") == 7 and task.get("required_task_contract")
            if nominal_context and (not isinstance(context, dict) or "travel_z" not in context):
                raise ValueError("Nominal placement departure requires its recorded task context")
            if not nominal_context:
                context = context or {}
            context_source = {"kind": "recorded", "ref": "/resources/" + task["resource_jid"]
                + "/resource_evidence/state/_task_ctx/travel_z"}
            if nominal_context:
                context_source = {"kind": "nominal_task", "task_contract": task["required_task_contract"],
                    "context_fingerprint": recovery_validation_fingerprint(context)}
            elif context.get("part_name") != part or "travel_z" not in context:
                previous = scene.get("resources", {}).get(rid, {}).get("validated_pickup_target_calculation") or {}
                if not isinstance(previous, dict) or not isinstance(previous.get("binding"), dict):
                    raise ValueError("Placement departure has invalid recorded pickup context")
                previous_contract = model.get("event_projection_contracts", {}).get(previous.get("contract_id"), {})
                if not isinstance(previous_contract, dict):
                    raise ValueError("Placement departure has invalid recorded pickup contract")
                previous_target = previous_contract.get("pickup_target_calculation") or {}
                if not isinstance(previous_target, dict):
                    raise ValueError("Placement departure has invalid recorded pickup calculation")
                if (previous.get("source") != "resource-owned pickup preview"
                        or previous_target.get("part_name") != part or previous.get("part_name") != part
                        or previous.get("binding", {}).get("resource_jid") != task["resource_jid"]
                        or previous.get("contract_fingerprint") != recovery_validation_fingerprint(previous_contract)
                        or previous.get("calculation") != previous_target
                        or previous.get("physical_execution_verified") is not False):
                    raise ValueError("Placement departure lacks travel_z from this part's recorded or validated pickup")
                context = {"part_name": part, "travel_z": previous_target.get("travel_z")}
                context_source = {"kind": "projected", "contract_fingerprint": previous["contract_fingerprint"],
                    "contract_id": previous["contract_id"], "binding": deepcopy(previous["binding"]),
                    "calculation": deepcopy(previous_target.get("source"))}
            target["departure_context"] = (deepcopy(context) if nominal_context else
                {"part_name": part, "travel_z": deepcopy(context.get("travel_z"))})
            target["departure_context_source"] = context_source
            placement_departure = _outline_place_departure(task=task, declarations=declaration_records,
                model=recovery_des_model, target=target, current_pose=pose)
            values = [placement_departure["pose"][key] for key in ("x", "y", "z", "qx", "qy", "qz", "qw")]
            bindings["target_pose"] = values
            endpoint["current_pose"] = {"set_from_param": "target_pose"}
            if (scene.get("physical_preview_contract_version") in {6, 7}
                    and target.get("target_reference", {}).get("surface_role") == "assembly_slot"):
                placement_arrival = _outline_place_arrival(task=task, model=recovery_des_model,
                    target=target, resource_evidence=resource_evidence, release=True)
                motions.extend(placement_arrival["motion_requests"])
            else:
                motions.append(_outline_cartesian_request([pose]))
            custody = {"kind": "release", "part_name": part,
                       "model_name": scene["snapshot"]["parts"][part]["model_name"],
                       "destination_location": destination, "target_calculation": deepcopy(target)}
            custody_motion_index = len(motions)
            motions.append(_outline_cartesian_request([placement_departure["pose"]]))
        elif (part is not None and owner.get("held_part") is None and end.get("held_part") is None
                and any(row["primitive"] == "compute_pick_targets" for row in support.get("primitives", []))):
            from cais_spade_llm.resources.robot.recovery_capability import calculate_recovery_pick_target

            target = calculate_recovery_pick_target(task=task, recovery_snapshot=recovery_snapshot,
                resource_evidence=resource_evidence)
            if target.get("status") != "calculated":
                return {**result, "missing_facts": deepcopy(target.get("missing_facts") or []),
                    "context_requests": deepcopy(target.get("context_requests") or []),
                    "reason": "Recorded geometry and pick policy cannot establish the requested approach target"}
            poses = _outline_pick_arrival(task=task, declarations=declaration_records, model=recovery_des_model,
                target=target, snapshot=recovery_snapshot, resource_evidence=resource_evidence, owner=owner)
            bindings["target_pose"] = [poses[-1][key] for key in ("x", "y", "z", "qx", "qy", "qz", "qw")]
            endpoint["current_pose"] = {"set_from_param": "target_pose"}
            motions.append(_outline_cartesian_request(poses))
        elif destination in model.get("named_joint_poses", {}):
            named = model["named_joint_poses"][destination]
            if named["joint_names"] != owner["joint_names"]:
                raise ValueError("Named joint configuration differs from the recorded resource")
            bindings["joint_positions"] = deepcopy(named["positions"])
            endpoint["joint_positions"] = {"set_from_param": "joint_positions"}
            motions.append({"mode": "joint_target", "joint_target": {
                "name": deepcopy(named["joint_names"]), "position": deepcopy(named["positions"])}})
        elif (part is not None and owner.get("held_part") == part and end.get("held_part", part) == part
                and destination is not None and any(row["primitive"] == "compute_place_targets"
                    for row in support.get("primitives", []))):
            from cais_spade_llm.resources.robot.recovery_capability import calculate_recovery_place_target
            from cais_spade_llm.resources.robot.target_calculations import placement_poses

            target = calculate_recovery_place_target(task=task, recovery_snapshot=recovery_snapshot,
                                                    resource_evidence=resource_evidence, physical_evidence=scene)
            if target.get("status") != "calculated":
                return {**result, "missing_facts": deepcopy(target.get("missing_facts") or []),
                        "context_requests": deepcopy(target.get("context_requests") or []),
                        "reason": "Recorded geometry cannot establish the requested loaded destination"}
            assembly_slot = target.get("target_reference", {}).get("surface_role") == "assembly_slot"
            depth = resource_evidence.get("controller_config", {}).get("parts_tuning", {}).get("insertion_depth_m")
            if assembly_slot and (type(depth) not in (float, int) or not math.isfinite(depth) or depth < 0):
                raise ValueError("Loaded placement requires its recorded insertion depth")
            insert = target["insert_pose"]
            poses = placement_poses(insert["x"], insert["y"], insert["z"],
                {key: insert[key] for key in ("qx", "qy", "qz", "qw")},
                simulation_assembly_slot=assembly_slot, insertion_depth=depth if assembly_slot else 0.)
            values = [poses["pre_insert_pose"][key] for key in ("x", "y", "z", "qx", "qy", "qz", "qw")]
            bindings["target_pose"] = values
            endpoint["current_pose"] = {"set_from_param": "target_pose"}
            if scene.get("physical_preview_contract_version") in {6, 7} and assembly_slot:
                placement_arrival = _outline_place_arrival(task=task, model=recovery_des_model,
                    target=target, resource_evidence=resource_evidence, release=False)
                motions.extend(placement_arrival["motion_requests"])
            else:
                motions.append(_outline_cartesian_request([poses["approach_pose"], poses["pre_insert_pose"]]))
            for declaration in declaration_records:
                event = next(row for row in recovery_des_model["events"] if row["event_name"] == declaration["event_name"])
                completed = event.get("composition_contract", {}).get("part_transition", {}).get("completed", {})
                if not completed:
                    continue
                if (set(completed) != {"state", "location_template"}
                        or completed["location_template"] != "{resource_jid}"):
                    raise ValueError("Loaded motion part effects require a declared carried-part completion")
                values = {"part_state": {"set": completed["state"]},
                          "part_location": {"set": task["resource_jid"]}}
                if carried_updates and values != carried_updates:
                    raise ValueError("Loaded motion declarations have conflicting carried-part effects")
                carried_updates = values
        elif all(row["event_name"] in model.get("geometry_unchanged_events", []) for row in declaration_records):
            pass
        else:
            raise ValueError("No declared owner motion policy supports these event end conditions")
        for declaration in declaration_records:
            event = next(row for row in recovery_des_model["events"] if row["event_name"] == declaration["event_name"])
            for field, effect in event.get("updates", {}).items():
                if not declarations.get(field, {}).get("private"):
                    continue
                if set(effect) == {"set"}:
                    value = effect["set"]
                elif set(effect) == {"set_from_param"} and effect["set_from_param"] in declaration["parameter_bindings"]:
                    value = declaration["parameter_bindings"][effect["set_from_param"]]
                else:
                    raise ValueError(f"Owner private condition {field} has unresolved effect semantics")
                if field in private_updates and private_updates[field] != value:
                    raise ValueError(f"Owner private condition {field} has conflicting effects")
                private_updates[field] = deepcopy(value)
        nominal = None
        if scene.get("physical_preview_contract_version") == 7 and task.get("required_task_contract"):
            nominal = _outline_nominal_program(task=task, recovery_snapshot=recovery_snapshot,
                recovery_des_model=recovery_des_model, physical_evidence=scene, resource_evidence=resource_evidence)
            motions = nominal["motion_requests"]
            if custody is not None:
                if nominal["custody_motion_index"] is None:
                    raise ValueError("Nominal owner program does not contain its projected custody operation")
                custody_motion_index = nominal["custody_motion_index"]
            elif nominal["custody_motion_index"] is not None:
                raise ValueError("Nominal owner program contains custody not established by its projection")
            if motions and motions[-1].get("mode") == "cartesian" and "target_pose" in bindings:
                last = motions[-1]["waypoints"][-1]
                bindings["target_pose"] = [*(last["position"][axis] for axis in ("x", "y", "z")),
                                          *(last["orientation"][axis] for axis in ("x", "y", "z", "w"))]
            if placement_arrival is not None:
                stop = custody_motion_index if custody is not None else len(motions)
                placement_arrival["motion_requests"] = deepcopy(motions[:stop])
                placement_arrival["mating_motion_indices"] = [stop - 1] if stop else []
        contract = {"version": scene.get("physical_preview_contract_version", 3),
                    "params": {name: {"const": deepcopy(value)} for name, value in bindings.items()},
                    "resource_updates": updates, "part_updates": carried_updates, "endpoint": endpoint,
                    "owner_declarations": declaration_records, "private_resource_updates": private_updates}
        if departure is not None:
            contract["pickup_departure"] = deepcopy(departure)
            contract["pickup_target_calculation"] = deepcopy(custody["target_calculation"])
        if placement_departure is not None:
            contract["placement_departure"] = deepcopy(placement_departure)
        if placement_arrival is not None:
            contract["placement_arrival"] = deepcopy(placement_arrival)
            contract["placement_target"] = deepcopy(target)
            if target.get("simulation_mating_contact") is not None:
                contract["mating_owner_evidence"] = {
                    key: deepcopy(resource_evidence[key]) for key in
                    ("execution_mode", "source", "controller_config") if key in resource_evidence}
                contract["mating_owner_evidence"]["inputs"] = {
                    "geometry": deepcopy(resource_evidence.get("inputs", {}).get("geometry", {}))}
                contract["mating_owner_evidence_source_fingerprint"] = resource_evidence_fingerprint
        if carried_updates:
            contract["carried_part_transition"] = True
        if custody is not None and custody["kind"] == "release":
            contract["placement_target"] = deepcopy(custody["target_calculation"])
        if contract["version"] in {4, 5, 6, 7}:
            from cais_spade_llm.resources.robot.recovery_capability import check_outline_motion_capabilities

            if not isinstance(resource_evidence.get("source"), dict):
                raise ValueError("Recorded owner capability source is unavailable")
            frame = resource_evidence.get("controller_config", {}).get("move_group", {}).get("frame_id")
            capability = check_outline_motion_capabilities(motion_requests=motions,
                static_capabilities=captured_capabilities, frame=frame, resource_jid=task["resource_jid"])
            contract["motion_capability_evidence"] = {"version": 1,
                "source": {"kind": "recorded", "ref": "/resources/" + task["resource_jid"] + "/static_capabilities",
                           "capture": deepcopy(resource_evidence.get("source"))},
                "static_capabilities": deepcopy(captured_capabilities), "motion_requests": deepcopy(motions),
                "result": capability}
            if capability["feasibility_status"] != "FEASIBLE":
                rejected = capability["feasibility_status"] == "INFEASIBLE"
                return {**result, "status": "rejected" if rejected else "NEEDS_CONTEXT",
                    "validation_status": "rejected" if rejected else "NEEDS_CONTEXT",
                    "physical_capability_rejected": rejected, "capability_result": capability,
                    "event_record_binding": outline_event_binding(task, recovery_snapshot, recovery_des_model, scene),
                    "resource_evidence_fingerprint": resource_evidence_fingerprint,
                    "static_capabilities_fingerprint": recovery_validation_fingerprint(captured_capabilities),
                    "task": deepcopy(task), "motion_requests": deepcopy(motions),
                    "reason": capability.get("reason"), "missing_facts": [] if rejected else [capability.get("reason")]}
        tolerance = model.get("event_endpoint_tolerance")
        if tolerance is not None:
            contract["endpoint_tolerance"] = deepcopy(tolerance)
        if (endpoint.get("current_pose") == {"set_from_param": "target_pose"}
                and "current_pose" not in task.get("expected_end_state", {})
                and "target_pose" not in (task.get("params") or {})):
            contract["endpoint_numerical_precision"] = {"current_pose": owner_cartesian_endpoint_precision()}
        if task.get("required_task_contract"):
            contract.update(task_contract=task["required_task_contract"],
                            task_declaration_fingerprint=declaration_records[0]["fingerprint"])
            if nominal is not None:
                contract["nominal_owner_evidence"] = deepcopy(resource_evidence)
                if nominal["nominal_snap"] is not None:
                    contract["nominal_snap"] = deepcopy(nominal["nominal_snap"])
        task_params = deepcopy(task.get("params") or {})
        if task.get("required_task_contract"):
            event = next(row for row in recovery_des_model["events"] if row["event_name"] == task["required_task_contract"])
            for name, schema in event.get("composition_contract", {}).get("params", {}).items():
                if name not in task_params and "default" in schema:
                    task_params[name] = deepcopy(schema["default"])
        contract_id = recovery_validation_fingerprint(contract)
        model.setdefault("event_projection_contracts", {})[contract_id] = deepcopy(contract)
        record = {"binding": outline_event_binding(task, recovery_snapshot, recovery_des_model, scene),
                  "source": "resource-owned event preview", "contract_id": contract_id,
                  "contract_fingerprint": recovery_validation_fingerprint(contract), "bindings": bindings,
                  "task_params": task_params, "intervals": []}
        return {**result, "status": "ready", "physical_evidence": scene, "event_record": record,
                "motion_requests": motions, "custody_operation": custody, "resource_id": rid,
                **({"nominal_snap": deepcopy(nominal["nominal_snap"])} if nominal is not None and nominal["nominal_snap"] else {}),
                "custody_motion_index": custody_motion_index if custody_motion_index is not None else len(motions),
                "task": deepcopy(task), "recovery_snapshot": deepcopy(recovery_snapshot),
                "recovery_des_model": deepcopy(recovery_des_model),
                "resource_evidence_fingerprint": resource_evidence_fingerprint}
    except (KeyError, ValueError, TypeError, IndexError) as exc:
        return {**result, "reason": str(exc), "missing_facts": [str(exc)]}


def validate_outline_motion_capabilities(*, contract: dict, model: dict, resource_jid: str) -> dict:
    """Recheck owner targets against immutable capability evidence."""
    from cais_spade_llm.resources.robot.recovery_capability import check_outline_motion_capabilities

    audit = contract.get("motion_capability_evidence")
    captured = model.get("static_capabilities")
    if (not isinstance(audit, dict) or audit.get("version") != 1 or not isinstance(captured, dict)
            or audit.get("static_capabilities") != captured
            or not isinstance(audit.get("motion_requests"), list)
            or audit.get("source", {}).get("kind") != "recorded"
            or audit.get("source", {}).get("ref") != "/resources/" + resource_jid + "/static_capabilities"
            or not isinstance(audit.get("source", {}).get("capture"), dict)):
        raise ValueError("Owner Cartesian target capability evidence is missing or changed")
    result = check_outline_motion_capabilities(motion_requests=audit["motion_requests"],
        static_capabilities=captured, frame=audit.get("result", {}).get("frame"), resource_jid=resource_jid)
    if result != audit.get("result") or result.get("feasibility_status") != "FEASIBLE":
        raise ValueError("Owner Cartesian target capability validation failed: " + str(result.get("reason", "record changed")))
    return deepcopy(audit)


def install_outline_cartesian_route(*, prepared_preview: dict, motion_index: int,
                                   search_request: dict, route_proof: dict, progress: dict) -> dict:
    """Install calculated targets while retaining the exact owner motion endpoints."""
    from cais_spade_llm.recovery_framework.recovery_motion_preview import read_cartesian_route
    from cais_spade_llm.agents.shared_information.recovery_validation_protocol import recovery_validation_fingerprint
    from cais_spade_llm.resources.robot.recovery_capability import check_outline_motion_capabilities

    if prepared_preview["physical_evidence"].get("physical_preview_contract_version") not in {5, 6, 7}:
        raise ValueError("Calculated routes require version-five or version-six physical evidence")
    if prepared_preview.get("custody_operation") and motion_index == prepared_preview["custody_motion_index"]:
        raise ValueError("Calculated routes cannot replace a declared custody departure")
    request = read_cartesian_route(route_proof)
    original = prepared_preview["motion_requests"][motion_index]
    candidate = route_proof.get("candidate")
    if (request != search_request or not isinstance(candidate, dict)
            or any(request.get(field) != value for field, value in original.items())
            or original.get("mode") != "cartesian"):
        raise ValueError("Calculated route differs from its original owner motion request")
    prepared = deepcopy(prepared_preview)
    rid, record = prepared["resource_id"], prepared["event_record"]
    contracts = prepared["physical_evidence"]["resource_models"][rid]["event_projection_contracts"]
    contract = deepcopy(contracts[record["contract_id"]])
    audit = contract["motion_capability_evidence"]
    if (candidate["route_search"]["static_capabilities_fingerprint"] != recovery_validation_fingerprint(audit["static_capabilities"])
            or candidate["route_search"]["gripper_reach"] != audit["static_capabilities"].get("gripper_reach")):
        raise ValueError("Calculated route changed the recorded owner capabilities")
    route = deepcopy(route_proof)
    route.update(stage_index=motion_index, original_motion_request=deepcopy(original),
        stage_state_fingerprint=recovery_validation_fingerprint(progress["physical_evidence"]["snapshot"]))
    prepared["motion_requests"][motion_index] = {"mode": "cartesian", "waypoints": deepcopy(candidate["derived_waypoints"])}
    audit["motion_requests"] = deepcopy(prepared["motion_requests"])
    audit["result"] = check_outline_motion_capabilities(motion_requests=prepared["motion_requests"],
        static_capabilities=audit["static_capabilities"], frame=audit["result"]["frame"], resource_jid=prepared["task"]["resource_jid"])
    if audit["result"]["feasibility_status"] != "FEASIBLE":
        raise ValueError("Calculated route has unsupported owner motion targets")
    replay = deepcopy(request)
    replay.pop("route_search_json", None)
    replay.update(prepared["motion_requests"][motion_index])
    route["replay_request_fingerprint"] = recovery_validation_fingerprint(replay)
    contract.setdefault("cartesian_routes", {})[str(motion_index)] = route
    identity = recovery_validation_fingerprint(contract)
    contracts[identity] = contract
    record.update(contract_id=identity, contract_fingerprint=identity)
    return prepared


def validate_outline_cartesian_route(*, route: dict, state: dict, native: dict,
                                    motion_request: dict, static_capabilities: dict) -> None:
    """Recheck a saved route against its actual stage and ordinary strict replay."""
    from cais_spade_llm.recovery_framework.recovery_motion_preview import read_cartesian_route
    from cais_spade_llm.agents.shared_information.recovery_validation_protocol import recovery_validation_fingerprint

    request = read_cartesian_route(route)
    candidate = route.get("candidate")
    if not isinstance(candidate, dict):
        raise ValueError("Saved route has no calculated candidate")
    expected = deepcopy(request)
    expected.pop("route_search_json", None)
    expected["waypoints"] = candidate["derived_waypoints"]
    if (route.get("stage_index") != native.get("stage_index")
            or route.get("stage_state_fingerprint") != recovery_validation_fingerprint(state)
            or route.get("replay_request_fingerprint") != recovery_validation_fingerprint(expected)
            or native.get("request_fingerprint") != route["replay_request_fingerprint"]
            or any(native.get(field) != request.get(field) for field in (
                "resource_id", "capture_fingerprint", "proposal_fingerprint", "scene_fingerprint", "model_fingerprint"))
            or route.get("original_motion_request") != {"mode": "cartesian", "waypoints": request["waypoints"]}
            or motion_request != {"mode": "cartesian", "waypoints": candidate["derived_waypoints"]}
            or candidate["route_search"]["static_capabilities_fingerprint"] != recovery_validation_fingerprint(static_capabilities)
            or candidate["route_search"]["gripper_reach"] != static_capabilities.get("gripper_reach")):
        raise ValueError("Saved calculated route has changed stage, endpoints, scene or owner limits")


def _outline_native_response(prepared: dict, index: int, response: dict) -> dict | None:
    """Check the native result against the exact request issued for this interval."""
    from cais_spade_llm.recovery_framework.recovery_motion_preview import preview_result_record, validate_preview_request
    from cais_spade_llm.agents.shared_information.recovery_validation_protocol import recovery_validation_fingerprint

    strict = prepared["physical_evidence"].get("physical_preview_contract_version") in {2, 3, 4, 5, 6, 7}
    requests = prepared.get("native_requests")
    if not strict and requests is None:
        return None
    if not isinstance(requests, list) or index >= len(requests):
        raise ValueError("Owner preview lacks the recorded issued native request")
    request = validate_preview_request(requests[index])
    if (response.get("request") != request or response.get("source") != "PreviewRecoveryMotion"
            or response.get("command_sent") is not False or response.get("execution_authorized") is not False
            or request["resource_id"] != prepared["resource_id"]
            or request["proposal_fingerprint"] != recovery_validation_fingerprint(prepared["task"])
            or any(request.get(field) != value for field, value in prepared["motion_requests"][index].items())):
        raise ValueError("Native preview differs from its exact issued owner request")
    checked = preview_result_record(request, response)
    if checked["joint_trajectory"] != response.get("joint_trajectory"):
        raise ValueError("Native preview trajectory differs from its captured result")
    if response.get("status") == "PREVIEW_COMPLETE" and response.get("sampled_collision_rejected") is True:
        raise ValueError("Native preview cannot complete with a rejected collision sample")
    if response.get("sampled_collision_rejected") is True and not checked.get("collision_rejection_evidence"):
        raise ValueError("Native collision rejection lacks its checked contact evidence")
    if response.get("cartesian_failure_evidence") is not None and (
            not response.get("cartesian_diagnostics_json")
            or checked.get("cartesian_failure_evidence") != response["cartesian_failure_evidence"]):
        raise ValueError("Native Cartesian failure differs from its captured diagnostics")
    return {"version": 1, "source": "PreviewRecoveryMotion", "stage_index": index,
        "request_fingerprint": recovery_validation_fingerprint(request),
        "motion_request_fingerprint": recovery_validation_fingerprint(prepared["motion_requests"][index]),
        "response_fingerprint": recovery_validation_fingerprint(response),
        **{field: request[field] for field in ("resource_id", "capture_fingerprint", "proposal_fingerprint",
                                               "scene_fingerprint", "model_fingerprint")},
        "command_sent": False, "execution_authorized": False}


def outline_event_preview_progress(*, prepared_preview: dict, motion_results: list) -> dict:
    """Replay verified private intervals and custody changes without committing the event."""
    from cais_spade_llm.resources.recovery_geometry import (
        _checkpoint, _joint_geometry, _prepared_motion, _configuration_after, _custody,
        projected_part_support, commanded_hold_evidence,
    )
    from cais_spade_llm.resources.continuous_motion import representative_pose
    from cais_spade_llm.agents.shared_information.recovery_validation_protocol import recovery_validation_fingerprint

    if prepared_preview.get("status") != "ready" or len(motion_results) > len(prepared_preview["motion_requests"]):
        raise ValueError("Owner event preparation or matching motion responses are unavailable")
    scene = deepcopy(prepared_preview["physical_evidence"])
    record = deepcopy(prepared_preview["event_record"])
    rid = prepared_preview["resource_id"]
    if scene.get("physical_preview_contract_version") in {4, 5, 6, 7}:
        contract = scene["resource_models"][rid]["event_projection_contracts"][record["contract_id"]]
        audit = validate_outline_motion_capabilities(contract=contract, model=scene["resource_models"][rid],
            resource_jid=prepared_preview["task"]["resource_jid"])
        if audit["motion_requests"] != prepared_preview["motion_requests"]:
            raise ValueError("Owner motion requests differ from their capability-checked targets")
    state, models = deepcopy(scene["snapshot"]), deepcopy(scene["resource_models"])
    checkpoint = _checkpoint(state, scene["geometry"], models, index=0)
    state.update(resources=deepcopy(checkpoint["resources"]), parts=deepcopy(checkpoint["parts"]))
    peers = set(state["resources"]) - {rid}
    peer_holds = {}
    for peer in peers:
        observed = scene.get("resources", {}).get(peer, {})
        if observed.get("model_static") is True:
            continue
        if "controller_hold_model" in observed or observed.get("idle") is not True:
            hold = commanded_hold_evidence(resource_id=peer, state=state["resources"][peer],
                model=models.get(peer, {}), physical=observed)
            if hold["status"] != "supported":
                raise ValueError(f"Stationary owner evidence is unavailable for {peer}: {hold['reason']}")
            peer_holds[peer] = hold
        elif not observed.get("stationary_contract"):
            raise ValueError(f"Stationary owner evidence is unavailable for {peer}")
    operation = prepared_preview.get("custody_operation")
    custody_index = prepared_preview.get("custody_motion_index", len(prepared_preview["motion_requests"]))
    pending_nominal_snap = None
    nominal_endpoint_rejection = None

    def endpoint_rejection_result():
        replay = {key: deepcopy(prepared_preview[key]) for key in (
            "status", "resource_id", "motion_requests", "event_record", "custody_operation",
            "custody_motion_index", "resource_evidence_fingerprint") if key in prepared_preview}
        return {"status": "rejected", "validation_status": "rejected", "physical_motion_rejected": True,
            "physical_endpoint_rejected": True, "endpoint_rejection": deepcopy(nominal_endpoint_rejection["endpoint_evidence"]),
            "endpoint_replay_preparation": replay, "rejected_stage_index": custody_index,
            "native_requests": deepcopy(prepared_preview.get("native_requests", [])),
            "motion_results": deepcopy(motion_results),
            "nominal_snap_result": deepcopy(prepared_preview["nominal_snap_result"]),
            "event_record_binding": deepcopy(prepared_preview["event_record"]["binding"]),
            "task": deepcopy(prepared_preview["task"]), "reason": nominal_endpoint_rejection.get("reason"),
            "command_sent": False, "execution_authorized": False}

    def apply_nominal_snap():
        nonlocal pending_nominal_snap, nominal_endpoint_rejection
        contract = models[rid]["event_projection_contracts"][record["contract_id"]]
        declared = contract.get("nominal_snap")
        if contract.get("version") != 7 or declared is None:
            return
        from cais_spade_llm.resources.robot.recovery_task_projection import project_nominal_snap

        checked = project_nominal_snap(resource_id=rid, params=declared["params"], state=state,
            evidence={**scene, "resource_models": models}, owner_evidence=declared["owner_evidence"],
            endpoint_result=prepared_preview.get("nominal_snap_result"))
        if checked.get("status") == "rejected" and checked.get("scene_endpoint_rejected") is True:
            nominal_endpoint_rejection = checked
            return
        if checked.get("status") != "supported":
            if checked.get("candidate_state") and checked.get("scene_endpoint_record"):
                pending_nominal_snap = checked
                return
            raise ValueError(checked.get("reason") or "Nominal simulation correction remains unresolved")
        record["intervals"].append({"source": "recorded simulation snap_part_to_slot",
            "nominal_snap_evidence": deepcopy(checked),
            "scene_endpoint_result": deepcopy(prepared_preview.get("nominal_snap_result")),
            "stationary_resource_evidence": deepcopy(peer_holds), "interval_complete": True})
        state.update(deepcopy(checked["projected_snapshot"]))
        part = checked["part_name"]
        scene.setdefault("stationary_part_evidence", {})[part] = deepcopy(checked["part_support_evidence"])
        checkpoint = _checkpoint(state, scene["geometry"], models, index=0)
        state.update(resources=deepcopy(checkpoint["resources"]), parts=deepcopy(checkpoint["parts"]))

    def apply_custody():
        custody = models[rid]["primitive_model"]["configuration"]["custody"]
        primitive = "grasp_part" if operation["kind"] == "grasp" else "release_part"
        endpoint = _outline_custody_evidence(primitive=primitive,
            params={"part_name": operation["part_name"], "model_name": operation["model_name"]},
            snapshot=state, resource_id=rid, model=models[rid], contract=custody)
        if operation.get("destination_location") in state["resources"]:
            endpoint["custody_effects"][0]["contained_by"] = operation["destination_location"]
        interval_record = {key: deepcopy(endpoint[key]) for key in
            ("custody_effects", "configuration_after", "interval_complete", "source")}
        interval_record["geometry_unchanged"] = True
        interval_record["stationary_resource_evidence"] = deepcopy(peer_holds)
        bounds = custody.get("joint_bounds")
        if isinstance(bounds, dict) and bounds:
            configuration_interval = deepcopy(models[rid]["configuration"])
            gripper_state = "closed" if operation["kind"] == "grasp" else "open"
            changed = [row for row in configuration_interval["joints"]
                       if row["name"] in custody["joint_positions"][gripper_state]]
            if all(row["name"] in bounds for row in changed):
                for joint in changed:
                    joint.pop("frozen_position", None)
                    joint["frozen_interval"] = deepcopy(bounds[joint["name"]])
                interval_record["configuration_interval"] = configuration_interval
                interval_record["interval_complete"] = True
        if operation["kind"] == "release":
            for motion_record in record["intervals"]:
                points = motion_record.get("joint_trajectory", {}).get("points", [])
                if not points:
                    continue
                velocities = points[-1].get("velocities", [])
                stopped = (len(velocities) == len(points[-1]["positions"])
                           and all(value == 0 for value in velocities))
                constant = (all(point["positions"] == points[0]["positions"] for point in points)
                            and not any(point.get("velocities") or point.get("accelerations") for point in points))
                if not stopped and not constant:
                    raise ValueError("Release support requires a recorded stopped owner motion endpoint")
            contract = models[rid]["event_projection_contracts"][record["contract_id"]]
            mating_intervals = [row for row in record["intervals"] if "mating_contact" in row]
            if (contract.get("version") in {6, 7} and mating_intervals
                    and contract.get("placement_target", {}).get("simulation_mating_contact", {}).get(
                        "retain_fixture_attachment") is True):
                from cais_spade_llm.resources.recovery_mating_contact import projected_fixture_handoff

                last_mating = mating_intervals[-1]
                supported = projected_fixture_handoff(resource_id=rid, part_name=operation["part_name"],
                    state=state, evidence={**scene, "resource_models": models},
                    owner_evidence=contract["mating_owner_evidence"], mating_contact=last_mating["mating_contact"],
                    endpoint_evidence={"mating_start_state": last_mating["mating_start_state"],
                        "mating_native_record": last_mating,
                        "mating_corridor_evidence": last_mating["mating_corridor_evidence"]})
                interval_record["fixture_handoff_evidence"] = deepcopy(supported)
            else:
                supported = projected_part_support(part_name=operation["part_name"], state=state, evidence=scene)
            interval_record["part_support_evidence"] = {
                key: deepcopy(value) for key, value in supported.items() if key != "physics_evidence"}
            if supported.get("status") != "supported":
                interval_record["interval_complete"] = False
            else:
                part = operation["part_name"]
                scene.setdefault("stationary_part_evidence", {})[part] = deepcopy(interval_record["part_support_evidence"])
                if part not in scene.setdefault("stationary_parts", []):
                    scene["stationary_parts"].append(part)
        record["intervals"].append(interval_record)
        _configuration_after(models, rid, interval_record)
        _custody(state, rid, interval_record["custody_effects"], scene["geometry"])
        checkpoint = _checkpoint(state, scene["geometry"], models, index=0)
        state.update(resources=deepcopy(checkpoint["resources"]), parts=deepcopy(checkpoint["parts"]))
        if operation["kind"] == "release":
            apply_nominal_snap()

    if operation and custody_index == 0:
        apply_custody()
    for index, response in enumerate(motion_results):
        if nominal_endpoint_rejection is not None:
            return endpoint_rejection_result()
        if pending_nominal_snap is not None:
            raise ValueError("Nominal motion cannot follow an unvalidated simulation correction")
        request = prepared_preview["motion_requests"][index]
        native_binding = _outline_native_response(prepared_preview, index, response)
        if (response.get("status") != "PREVIEW_COMPLETE" or response.get("path_complete") is not True
                or response.get("fraction") != 1.0 or response.get("command_sent") is not False):
            raise ValueError(response.get("reason") or "Isolated motion preview did not complete")
        if any(response.get("request", {}).get(field) != value for field, value in request.items()):
            raise ValueError("Native preview does not match the owner event motion target")
        trajectory = deepcopy(response["joint_trajectory"])
        stamp = trajectory["points"][-1]["time_from_start"]
        trajectory["duration_ns"] = stamp["sec"] * 1_000_000_000 + stamp["nanosec"]
        validate_joint_trajectory(trajectory, state["resources"][rid]["joint_names"])
        free_parts = set(state["parts"]) - {row.get("held_part") for row in state["resources"].values()}
        missing_parts = free_parts - set(scene.get("stationary_parts", []))
        records = scene.get("stationary_part_evidence")
        if isinstance(records, dict):
            missing_parts.update(name for name in free_parts if records.get(name, {}).get("status") != "supported"
                or records[name].get("current_pose") != state["parts"][name].get("current_pose"))
        if missing_parts:
            raise ValueError("Stationary evidence is unavailable for unheld parts: " + ", ".join(sorted(missing_parts)))
        interval_record = {"joint_trajectory": trajectory, "source": "isolated owner motion preview",
            "stationary_resources": sorted(peers), "stationary_parts": sorted(free_parts),
            "stationary_resource_evidence": peer_holds}
        if scene.get("physical_preview_contract_version") in {4, 5, 6, 7}:
            interval_record["motion_request"] = deepcopy(request)
        if native_binding is not None:
            interval_record["native_preview_binding"] = {**native_binding,
                "stage_state_fingerprint": recovery_validation_fingerprint(state),
                "trajectory_fingerprint": recovery_validation_fingerprint(trajectory)}
        if request.get("mode") == "cartesian":
            pose = request["waypoints"][-1]
            interval_record["native_endpoint_evidence"] = {
                **{field: deepcopy(response.get(field)) for field in (
                    "cartesian_position_precision", "cartesian_orientation_precision",
                    "cartesian_position_residual", "cartesian_orientation_residual",
                    "model_fingerprint", "derived_model_fingerprint", "scene_fingerprint")},
                "target_pose": [*(pose["position"][axis] for axis in ("x", "y", "z")),
                                *(pose["orientation"][axis] for axis in ("x", "y", "z", "w"))],
                "trajectory_fingerprint": recovery_validation_fingerprint(trajectory),
                "request_fingerprint": recovery_validation_fingerprint(response["request"]),
            }
        support_text = response.get("request", {}).get("support_departure_json")
        support_departure = json.loads(support_text) if support_text else None
        if support_departure is not None:
            from cais_spade_llm.resources.recovery_support_departure import verify_support_departure

            verified = verify_support_departure(resource_id=rid, state=state,
                evidence={**scene, "resource_models": models}, joint_trajectory=trajectory,
                support_departure=support_departure)
            if verified.get("status") != "supported":
                raise ValueError(verified.get("reason") or "Continuous support departure remains unresolved")
            interval_record["support_departure"] = deepcopy(support_departure)
            interval_record["support_departure_evidence"] = verified
            interval_record["native_support_departure_evidence"] = deepcopy(response.get("support_departure_evidence"))
            interval_record["support_departure_json"] = support_text
            interval_record["support_departure_fingerprint"] = response["request"].get("support_departure_fingerprint")
        mating_text = response.get("request", {}).get("mating_contact_json")
        if mating_text:
            from cais_spade_llm.resources.recovery_mating_contact import (
                validate_mating_contact_samples, verify_mating_contact,
            )

            contract = models[rid]["event_projection_contracts"][record["contract_id"]]
            mating = json.loads(mating_text)
            verified = verify_mating_contact(resource_id=rid, state=state,
                evidence={**scene, "resource_models": models}, owner_evidence=contract["mating_owner_evidence"],
                joint_trajectory=trajectory, mating_contact=mating)
            if verified.get("status") == "rejected":
                if native_binding is None or not isinstance(verified.get("counterexample"), dict):
                    raise ValueError("Continuous mating rejection lacks bound native and model evidence")
                replay = {key: deepcopy(prepared_preview[key]) for key in (
                    "status", "resource_id", "motion_requests", "event_record", "custody_operation",
                    "custody_motion_index", "resource_evidence_fingerprint") if key in prepared_preview}
                return {"status": "rejected", "validation_status": "rejected", "physical_motion_rejected": True,
                    "continuous_mating_rejected": True, "mating_rejection": deepcopy(verified),
                    "mating_replay_preparation": replay, "rejected_stage_index": index,
                    "native_requests": deepcopy(prepared_preview["native_requests"][:index + 1]),
                    "motion_results": deepcopy(motion_results[:index + 1]), "rejected_motion": deepcopy(response),
                    "event_record_binding": deepcopy(prepared_preview["event_record"]["binding"]),
                    "task": deepcopy(prepared_preview["task"]), "reason": verified["reason"],
                    "command_sent": False, "execution_authorized": False}
            if verified.get("status") != "supported":
                raise ValueError(verified.get("reason") or "Continuous mating corridor remains unresolved")
            interval_record.update(mating_contact=deepcopy(mating), mating_corridor_evidence=verified,
                mating_start_state=deepcopy(state), mating_contact_json=mating_text,
                mating_contact_fingerprint=response["request"].get("mating_contact_fingerprint"),
                native_mating_contact_evidence=deepcopy(response.get("mating_contact_evidence")))
            validate_mating_contact_samples(interval_record)
        motion = _prepared_motion(state, rid, models, interval_record)
        state["resources"][rid]["joint_positions"] = deepcopy(motion.points[-1]["positions"])
        _, matrix = _joint_geometry(state["resources"][rid], models[rid])
        state["resources"][rid]["current_pose"] = representative_pose(matrix)
        checkpoint = _checkpoint(state, scene["geometry"], models, index=0)
        state.update(resources=deepcopy(checkpoint["resources"]), parts=deepcopy(checkpoint["parts"]))
        record["intervals"].append(interval_record)
        if operation and index + 1 == custody_index:
            apply_custody()
    if nominal_endpoint_rejection is not None:
        return endpoint_rejection_result()
    if not record["intervals"]:
        record["intervals"] = [{"geometry_unchanged": True, "source": "declared owner event geometry",
                               "stationary_resource_evidence": deepcopy(peer_holds)}]
    scene.update(snapshot=state, resource_models=models)
    return {"physical_evidence": scene, "event_record": record,
            **({"pending_nominal_snap": pending_nominal_snap} if pending_nominal_snap is not None else {})}


def prepare_outline_event_departure(*, prepared_preview: dict, progress: dict) -> dict:
    """Resolve departure from the verified custody endpoint in the private scene."""
    from cais_spade_llm.agents.shared_information.recovery_validation_protocol import recovery_validation_fingerprint

    prepared = deepcopy(prepared_preview)
    operation = prepared.get("custody_operation")
    if not operation:
        return prepared
    rid = prepared["resource_id"]
    pose = progress["physical_evidence"]["snapshot"]["resources"][rid]["current_pose"]
    current = dict(zip(("x", "y", "z", "qx", "qy", "qz", "qw"), pose, strict=True))
    record = prepared["event_record"]
    contracts = prepared["physical_evidence"]["resource_models"][rid]["event_projection_contracts"]
    contract = deepcopy(contracts[record["contract_id"]])
    calculate = _outline_pick_departure if operation["kind"] == "grasp" else _outline_place_departure
    departure = calculate(task=prepared["task"], declarations=contract["owner_declarations"],
        model=prepared["recovery_des_model"], target=operation["target_calculation"], current_pose=current)
    prepared["motion_requests"][prepared["custody_motion_index"]] = _outline_cartesian_request([departure["pose"]])
    if contract.get("version") in {4, 5, 6, 7}:
        from cais_spade_llm.resources.robot.recovery_capability import check_outline_motion_capabilities

        audit = contract["motion_capability_evidence"]
        audit["motion_requests"] = deepcopy(prepared["motion_requests"])
        audit["result"] = check_outline_motion_capabilities(motion_requests=prepared["motion_requests"],
            static_capabilities=audit["static_capabilities"], frame=audit["result"]["frame"],
            resource_jid=prepared["task"]["resource_jid"])
        if audit["result"]["feasibility_status"] != "FEASIBLE":
            raise ValueError("Supported custody endpoint gives an unsupported departure target: " + str(audit["result"].get("reason")))
    values = [departure["pose"][key] for key in ("x", "y", "z", "qx", "qy", "qz", "qw")]
    contract["pickup_departure" if operation["kind"] == "grasp" else "placement_departure"] = departure
    contract["params"]["target_pose"] = {"const": values}
    record["bindings"]["target_pose"] = deepcopy(values)
    identity = recovery_validation_fingerprint(contract)
    contracts[identity] = contract
    record.update(contract_id=identity, contract_fingerprint=identity)
    return prepared


def finalize_outline_event_preview(*, prepared_preview: dict, motion_results: list,
                                   owner_projector: Callable[..., dict] | None = None,
                                   resource_evidence: dict | None = None) -> dict:
    """Bind all isolated intervals and locally recheck the complete event geometry."""
    from cais_spade_llm.resources.recovery_geometry import project_recovery_outline_event
    from cais_spade_llm.agents.shared_information.recovery_validation_protocol import recovery_validation_fingerprint

    result = {"status": "NEEDS_CONTEXT", "command_sent": False, "execution_authorized": False}
    rejected = next((row for row in motion_results if row.get("sampled_collision_rejected") is True), None)
    if rejected is not None:
        index = motion_results.index(rejected)
        try:
            progress = outline_event_preview_progress(prepared_preview=prepared_preview, motion_results=motion_results[:index])
            if progress.get("continuous_mating_rejected") is True:
                return progress
            _outline_native_response(prepared_preview, index, rejected)
        except (KeyError, ValueError, TypeError, IndexError) as exc:
            return {**result, "reason": str(exc)}
        request = rejected.get("request") or {}
        if (index >= len(prepared_preview["motion_requests"])
                or any(request.get(field) != value for field, value in prepared_preview["motion_requests"][index].items())
                or rejected.get("source") != "PreviewRecoveryMotion"
                or rejected.get("command_sent") is not False or rejected.get("execution_authorized") is not False
                or any(not request.get(field) or request[field] != rejected.get(field) for field in (
                    "capture_fingerprint", "proposal_fingerprint", "scene_fingerprint", "model_fingerprint"))
                or request.get("proposal_fingerprint") != recovery_validation_fingerprint(prepared_preview["task"])):
            return {**result, "reason": "Rejected motion lacks the exact owner proposal and native request binding"}
        return {**result, "status": "rejected", "validation_status": "rejected",
            "physical_motion_rejected": True, "rejected_motion": deepcopy(rejected),
            "event_record_binding": deepcopy(prepared_preview["event_record"]["binding"]),
            "task": deepcopy(prepared_preview["task"]),
            "reason": rejected.get("reason") or "The supplied owner motion intersects recorded geometry"}
    incomplete = next((row for row in motion_results
        if row.get("status") != "PREVIEW_COMPLETE"
        and isinstance(row.get("cartesian_failure_evidence"), dict)
        and row["cartesian_failure_evidence"].get("complete_inputs") is True
        and row["cartesian_failure_evidence"].get("failure_kind") in {"ik_search_failed", "joint_jump"}), None)
    if incomplete is not None:
        index = motion_results.index(incomplete)
        try:
            progress = outline_event_preview_progress(prepared_preview=prepared_preview, motion_results=motion_results[:index])
            if progress.get("continuous_mating_rejected") is True:
                return progress
            binding = _outline_native_response(prepared_preview, index, incomplete)
        except (KeyError, ValueError, TypeError, IndexError) as exc:
            return {**result, "reason": str(exc)}
        if (binding is None or not prepared_preview.get("resource_evidence_fingerprint")
                or incomplete.get("path_complete") is not False
                or type(incomplete.get("fraction")) not in (float, int)
                or not 0 <= incomplete["fraction"] < 1):
            return {**result, "reason": "Incomplete motion lacks the exact owner proposal and native request binding"}
        return {**result, "validation_status": "NEEDS_CONTEXT", "physical_motion_unresolved": True,
            "incomplete_motion": deepcopy(incomplete),
            "event_record_binding": deepcopy(prepared_preview["event_record"]["binding"]),
            "resource_evidence_fingerprint": prepared_preview["resource_evidence_fingerprint"],
            "task": deepcopy(prepared_preview["task"]),
            "reason": incomplete.get("reason") or "The recorded motion search did not establish a complete path"}
    try:
        progress = outline_event_preview_progress(prepared_preview=prepared_preview, motion_results=motion_results)
        if progress.get("continuous_mating_rejected") is True or progress.get("physical_endpoint_rejected") is True:
            return progress
        if progress.get("pending_nominal_snap") is not None:
            return {**result, "reason": progress["pending_nominal_snap"].get("reason"),
                    "pending_nominal_snap": deepcopy(progress["pending_nominal_snap"])}
        if len(motion_results) != len(prepared_preview["motion_requests"]):
            failed = next((row for row in motion_results if row.get("status") != "PREVIEW_COMPLETE"), {})
            raise ValueError(failed.get("reason") or "Complete owner event motion responses are unavailable")
        scene = prepared_preview["physical_evidence"]
        record = progress["event_record"]
        if scene.get("physical_preview_contract_version") in {6, 7} and owner_projector is not None:
            scene = deepcopy(scene)
            if record not in scene.setdefault("event_evidence", []):
                scene["event_evidence"].append(deepcopy(record))
            projection = owner_projector(task=prepared_preview["task"],
                recovery_snapshot=prepared_preview["recovery_snapshot"],
                recovery_des_model=prepared_preview["recovery_des_model"],
                physical_input={"physical_evidence": scene,
                    **({"resource_evidence": resource_evidence} if resource_evidence is not None else {})})
            result["physical_evidence"] = scene
        else:
            projection = project_recovery_outline_event(task=prepared_preview["task"],
                recovery_snapshot=prepared_preview["recovery_snapshot"],
                recovery_des_model=prepared_preview["recovery_des_model"], evidence=scene, event_evidence=[record],
                resource_evidence=resource_evidence)
        if projection.get("owner_effect_rejected") is True:
            return {**result, **projection}
        return {**result, "status": "prepared" if projection.get("status") == "projected" else "NEEDS_CONTEXT",
                "event_evidence": [record], "resource_models": deepcopy(scene["resource_models"]),
                "geometry_evidence": projection, "reason": "; ".join(projection.get("unresolved") or [])}
    except (KeyError, ValueError, TypeError, IndexError) as exc:
        return {**result, "reason": str(exc)}


def primitive_model_descriptors(primitive_models: dict | None) -> dict:
    """Validate exact resource bindings and return only serializable metadata."""
    if primitive_models is None:
        return {}
    if not isinstance(primitive_models, dict):
        raise ValueError("primitive_models must bind exact resource identities")
    result = {}
    for resource, model in primitive_models.items():
        if not isinstance(resource, str) or not resource or not isinstance(model, PrimitiveModel):
            raise ValueError("Invalid resource-owned primitive model")
        result[resource] = model.descriptor()
    return result


_RESOURCE_PROVIDERS: dict[str, Any] = {}
_LEGACY_MODELS: dict[str, Callable[[], PrimitiveModel]] = {}


def register_resource_provider(identifier: str, provider: Any) -> None:
    """Register a trusted owner factory and its immutable program validator."""
    if (not isinstance(identifier, str) or not identifier
            or type(getattr(provider, "version", None)) is not int or provider.version < 1
            or not callable(getattr(provider, "create_owner", None))
            or not callable(getattr(provider, "validate_programs", None))):
        raise ValueError("Resource provider needs a version, owner factory and program validator")
    if identifier in _RESOURCE_PROVIDERS and _RESOURCE_PROVIDERS[identifier] is not provider:
        raise ValueError("Resource provider already registered: " + identifier)
    _RESOURCE_PROVIDERS[identifier] = provider


def resource_provider(identifier: str) -> Any:
    """Resolve an exact trusted provider reference without dynamic imports."""
    if not isinstance(identifier, str) or identifier not in _RESOURCE_PROVIDERS:
        raise ValueError("Resource owner provider unavailable: " + str(identifier))
    return _RESOURCE_PROVIDERS[identifier]


def resource_model_declarations(scene: dict) -> dict:
    """Read additive scene models without replacing any existing resource."""
    entries = scene.get("resource_models", {})
    if not isinstance(entries, dict):
        raise ValueError("scene.resource_models must be a mapping")
    identities = [*(row["resource_id"] for row in scene["robots"]),
                *(row["resource_id"] for row in scene["machines"]),
                "KMR", "Conveyor", "Buffer For Machined parts", "3D Printing Station", "Storage", "Exit"]
    existing = set(identities)
    if len(existing) != len(identities):
        raise ValueError("Duplicate configured resource identities")
    for rid, entry in entries.items():
        if (not isinstance(rid, str) or not rid or rid in existing
                or not isinstance(entry, dict) or set(entry) != {"model", "owner_provider"}
                or not isinstance(entry["owner_provider"], str) or not entry["owner_provider"]):
            raise ValueError("Invalid or duplicate additional resource declaration: " + str(rid))
        model = entry["model"]
        if (not isinstance(model, dict) or model.get("resource_id") != rid
                or not isinstance(model.get("events"), list)
                or not isinstance(model.get("state_variables"), dict)
                or not isinstance(model.get("current_valuation"), dict)
                or not isinstance(model.get("assignments"), dict)
                or not isinstance(model.get("marked_state_conditions"), list)):
            raise ValueError("Incomplete environment model for " + rid)
        if not {field for field in model["state_variables"] if "{part_name}" not in field} <= set(model["current_valuation"]):
            raise ValueError("Initial resource valuation must cover its declared fields")
        json.dumps(entry, allow_nan=False)
    return deepcopy(entries)


def legacy_primitive_model(evidence: dict) -> PrimitiveModel | None:
    """Resolve explicit historical contracts; absent tags retain legacy models."""
    contract = evidence.get("primitive_contract")
    if contract is None:
        return None
    if not _LEGACY_MODELS:
        # Built-in registration is separate from resource or action dispatch.
        from cais_spade_llm.resources.robot.gazebo_pick_place_controller import (
            register_preparation_contracts,
        )

        register_preparation_contracts(_LEGACY_MODELS)
    if not isinstance(contract, str) or contract not in _LEGACY_MODELS:
        raise ValueError("Unknown primitive observation contract: " + str(contract))
    return _LEGACY_MODELS[contract]()


def pose_values(pose) -> list[float]:
    """Copy an observed pose without changing its frame or quaternion."""
    values = [getattr(pose.position, key) for key in ("x", "y", "z")]
    values += [getattr(pose.orientation, key) for key in ("x", "y", "z", "w")]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Nonfinite observed pose")
    if not math.isclose(sum(value * value for value in values[3:]), 1., abs_tol=1e-5):
        raise ValueError("Observed pose requires a unit quaternion")
    return values


def validate_joint_trajectory(trajectory: dict, joint_names: list[str], duration: float | None = None) -> None:
    """Check exact configured joints, values, point order and nanosecond timing."""
    if (not joint_names or any(not isinstance(name, str) or not name for name in joint_names)
            or len(set(joint_names)) != len(joint_names) or trajectory["joint_names"] != joint_names
            or len(trajectory["points"]) < 2):
        raise ValueError("Joint trajectory does not cover the configured joint identities")
    previous = -1
    for index, row in enumerate(trajectory["points"]):
        stamp = row["time_from_start"]
        if (type(stamp["sec"]) is not int or stamp["sec"] < 0
                or type(stamp["nanosec"]) is not int or not 0 <= stamp["nanosec"] < 1_000_000_000):
            raise ValueError("Invalid joint trajectory timing")
        nanos = stamp["sec"] * 1_000_000_000 + stamp["nanosec"]
        if (nanos <= previous or previous == -1 and nanos != 0
                or len(row["positions"]) != len(joint_names)):
            raise ValueError(
                "Joint trajectory point order or coverage is invalid: "
                f"point_index={index}, time_from_start_ns={nanos}, previous_time_ns={previous}, "
                f"positions={len(row['positions'])}, joint_names={len(joint_names)}"
            )
        for field in ("positions", "velocities", "accelerations", "effort"):
            values = row.get(field, [])
            if len(values) not in (0, len(joint_names)) or any(type(v) not in (int, float) or not math.isfinite(v) for v in values):
                raise ValueError("Joint trajectory values are invalid")
        previous = nanos
    if (previous <= 0 or type(trajectory["duration_ns"]) is not int or trajectory["duration_ns"] != previous
            or duration is not None and not math.isclose(previous / 1e9, duration, abs_tol=1e-9)):
        raise ValueError("Joint trajectory duration disagrees with its interval")


def trajectory_record(trajectory, joint_names: list[str] | None = None, *, validate: bool = True) -> dict:
    """Preserve native values, allowing raw diagnostics before strict validation.

    Args:
        trajectory: Native ROS joint trajectory to copy.
        joint_names: Exact configured joint ordering, when available.
        validate: Require a complete trajectory beginning at zero. False only
            captures diagnostics; callers must validate before preparation.

    Returns:
        The unchanged native points, derivatives, header and timing.
    """
    names = list(trajectory.joint_names)
    rows = []
    for point in trajectory.points:
        stamp = point.time_from_start
        rows.append({**{key: list(getattr(point, key, [])) for key in (
            "positions", "velocities", "accelerations", "effort")},
            "time_from_start": {"sec": stamp.sec, "nanosec": stamp.nanosec}})
    if not rows:
        raise ValueError("Prepared trajectory has no points")
    header, last = trajectory.header, rows[-1]["time_from_start"]
    result = {"header": {"frame_id": header.frame_id,
                         "stamp": {"sec": header.stamp.sec, "nanosec": header.stamp.nanosec}},
              "joint_names": names, "points": rows,
              "duration_ns": last["sec"] * 1_000_000_000 + last["nanosec"]}
    if validate:
        validate_joint_trajectory(result, names if joint_names is None else joint_names)
    return result


def cartesian_coverage_fingerprint(trajectory: list, start_time) -> str:
    """Bind continuous coverage to relative time without inventing a schedule."""
    return fingerprint([{**row, "time": str(Fraction(str(row["time"])) - Fraction(str(start_time)))}
                        for row in trajectory])


def observation_identity(observation: dict) -> dict:
    """Exclude receipt times, retaining all owner-declared physical facts."""
    if observation.get("status") == "NEEDS_CONTEXT":
        raise ValueError(observation.get("reason", "Resource observation unavailable"))
    fields = {"observed_monotonic", "simulation_time", "tool_stamp", "joint_stamps",
              "joint_received_monotonic"}
    result = {key: deepcopy(value) for key, value in observation.items() if key not in fields}
    if isinstance(result.get("gazebo_observation"), dict):
        result["gazebo_observation"] = {key: value for key, value in result["gazebo_observation"].items()
                                        if key not in {"simulation_stamp", "observed_monotonic"}}
    return result


class LiveCommandLedger:
    """Retain pending commands before dispatch and outcomes until clearance.

    Args:
        lock: The runtime's shared admission transaction lock.
    """

    def __init__(self, lock: Any | None = None) -> None:
        self.lock = lock if lock is not None else RLock()
        self.revision = 0
        self.execution_epoch = 0
        self.commands: dict[str, dict[str, Any]] = {}

    def prepare(self, *, resource_id: str, task_id: str | None,
                command: dict, owner_identity: dict) -> str:
        """Record the exact owner command before its controller can accept it."""
        if not resource_id or not isinstance(command, dict) or not owner_identity:
            raise ValueError("Live command requires its resource, command, and owner identity")
        with self.lock:
            identifier = uuid4().hex
            self.commands[identifier] = {
                "command_id": identifier, "resource_id": resource_id,
                "task_id": task_id, "command": deepcopy(command),
                "command_fingerprint": fingerprint(command),
                "owner_identity": deepcopy(owner_identity), "status": "pending",
                "controller_goal_id": None, "observations": None,
            }
            self.revision += 1
            return identifier

    def authorize(self, identifier: str, *, reservation_token: str,
                  expected_revision: int) -> None:
        """Attach an atomic CCA reservation to an unchanged pending command."""
        with self.lock:
            row = self.commands[identifier]
            if (self.revision != expected_revision or row["status"] != "pending"
                    or row.get("reservation_token") or not isinstance(reservation_token, str) or not reservation_token):
                raise ValueError("Physical command reservation is stale or missing")
            row["reservation_token"] = reservation_token
            self.revision += 1
            self.execution_epoch += 1

    def authorize_many(self, identifiers: list[str], *, reservation_token: str,
                       expected_revision: int) -> None:
        """Authorize a complete program without exposing partial reservations."""
        with self.lock:
            if (self.revision != expected_revision or not identifiers
                    or len(identifiers) != len(set(identifiers))
                    or not isinstance(reservation_token, str) or not reservation_token):
                raise ValueError("Physical program reservation is stale or missing")
            rows = [self.commands[identifier] for identifier in identifiers]
            if any(row["status"] != "pending" or row.get("reservation_token") for row in rows):
                raise ValueError("Physical program has already started")
            for row in rows:
                row["reservation_token"] = reservation_token
            self.revision += 1
            self.execution_epoch += 1

    def require_authorized(self, identifier: str, *, command: dict) -> str:
        """Check the exact pending command immediately before dispatch."""
        with self.lock:
            row = self.commands[identifier]
            if (row["status"] != "pending" or row["command"] != command
                    or not row.get("reservation_token")):
                raise ValueError("Live command has no matching physical reservation")
            return row["reservation_token"]

    def start(self, identifier: str, *, command: dict, controller_goal_id: Any) -> None:
        """Bind controller acceptance to the unchanged pending command."""
        with self.lock:
            row = self.commands[identifier]
            if (row["status"] != "pending" or row["command"] != command
                    or not row.get("reservation_token")):
                raise ValueError("Controller command differs from its pending physical evidence")
            row.update(status="active", controller_goal_id=deepcopy(controller_goal_id))
            self.revision += 1
            self.execution_epoch += 1

    def finish(self, identifier: str, *, success: bool, observations: dict) -> None:
        """Record an outcome without inferring physical clearance from completion."""
        if type(success) is not bool or not isinstance(observations, dict):
            raise ValueError("Command completion requires observed outcome evidence")
        with self.lock:
            row = self.commands[identifier]
            if row["status"] not in {"pending", "active"}:
                raise ValueError("Live command already has a terminal outcome")
            row.update(status="completed" if success else "failed",
                       observations=deepcopy(observations))
            self.revision += 1
            self.execution_epoch += 1

    def bind_goal(self, identifier: str, controller_goal_id: list[int]) -> None:
        """Attach actual controller acceptance to the command already being sent."""
        with self.lock:
            row = self.commands[identifier]
            if row["status"] != "active" or row["controller_goal_id"] is not None:
                raise ValueError("Controller goal cannot be rebound")
            row["controller_goal_id"] = list(controller_goal_id)
            self.revision += 1
            self.execution_epoch += 1

    def snapshot(self) -> dict:
        """Return detached command history, including pending and failed motions."""
        with self.lock:
            return {"revision": self.revision, "execution_epoch": self.execution_epoch,
                    "commands": deepcopy(self.commands)}


def _validate_controller_contract(record: dict, planned: dict) -> None:
    """Check native provenance and distinguish ownership from physical bounds."""
    if (not isinstance(record, dict) or record.get("operation") != "prepare"
            or type(record.get("accepted")) is not bool):
        raise ValueError("Native motion contract response is malformed")
    if not record["accepted"]:
        raise ValueError("Native motion contract rejected preparation: " + str(record.get("reason", "")))
    if record.get("reason") != "":
        raise ValueError("Native motion contract accepted preparation with a conflicting reason")
    for field in ("instance_id", "expected_instance_id", "binding_fingerprint",
                  "requested_binding_fingerprint"):
        if not isinstance(record.get(field), str) or not record[field]:
            raise ValueError("Native motion contract identity is unavailable: " + field)
    for field in ("command_revision", "expected_command_revision", "contract_revision",
                  "expected_contract_revision", "rejected_commands"):
        if type(record.get(field)) is not int or record[field] < 0:
            raise ValueError("Native motion contract revision or count is unavailable: " + field)
    if (record.get("instance_id") != record["expected_instance_id"]
            or record.get("command_revision") != record["expected_command_revision"]
            or record.get("contract_revision") != record["expected_contract_revision"]
            or record.get("binding_fingerprint") != record["requested_binding_fingerprint"]):
        raise ValueError("Native motion contract identity changed during preparation")
    names = planned["joint_trajectory"]["joint_names"]
    if (not isinstance(names, list) or not names
            or any(not isinstance(name, str) or not name for name in names)
            or len(set(names)) != len(names) or record.get("joint_names") != names):
        raise ValueError("Native motion contract names another set of joints")
    for field in ("observed_positions", "observed_velocities"):
        values = record.get(field)
        if (not isinstance(values, list) or len(values) != len(names)
                or any(type(value) not in (float, int) or not math.isfinite(value) for value in values)):
            raise ValueError("Native motion contract has incomplete joint observations")
    if record["observed_positions"] != planned["start"]["joint_positions"]:
        raise ValueError("Native controller moved from the prepared initial positions")
    if record.get("stationary") is not False or record.get("reservation_token") != "":
        raise ValueError("Read-only motion preparation unexpectedly acquired execution authority")
    if (record.get("contract_state") != "unlocked" or record.get("observed_stationary") is not True
            or type(record.get("stationary_samples")) is not int or record["stationary_samples"] < 2
            or any(value != 0 for value in record["observed_velocities"])):
        raise ValueError("Native motion contract has no observed stationary unlocked owner")
    for field in ("simulation_time", "checkpoint_simulation_time", "elapsed_wall_s"):
        if type(record.get(field)) not in (int, float) or not math.isfinite(record[field]):
            raise ValueError("Native motion contract timing is unavailable")
    if (not 0 <= record["simulation_time"] - record["checkpoint_simulation_time"] <= 2.0
            or not 0 <= record["elapsed_wall_s"] <= 2.0):
        raise ValueError("Native motion contract observation is stale")
    if type(record.get("physical_execution_verified")) is not bool:
        raise ValueError("Native physical execution status is unavailable")
    if record["physical_execution_verified"]:
        raise ValueError("Unsupported physical containment assertion from the position controller")
    if (not isinstance(record.get("physical_execution_reason"), str)
            or not record["physical_execution_reason"]):
        raise ValueError("Native physical execution unavailability reason is missing")
    # These fields describe past samples only. Even zero observed error or
    # regular updates cannot establish future containment or stopping bounds.
    period = record.get("maximum_observed_update_period")
    errors = record.get("maximum_observed_position_error")
    if (type(period) not in (int, float) or not math.isfinite(period) or period < 0
            or not isinstance(errors, list) or len(errors) != len(names)
            or any(type(value) not in (int, float) or not math.isfinite(value) or value < 0
                   for value in errors)):
        raise ValueError("Native motion contract sample diagnostics are unavailable")


def validate_prepared_start(observed: dict, expected: dict, *, continuous_motion: dict | None = None) -> None:
    """Require unchanged owner state and every supplied physical geometry field.

    Detached preparation requires exact observations. A CCA-owned modeled start
    may use the retained joint_position_error assumption and its enclosing FK
    geometry. Neither path certifies future physical tracking or stopping.
    """
    for field in ("joint_names", "frame", "launch_id", "attachment"):
        if observed.get(field) != expected.get(field) or field not in observed or field not in expected:
            raise ValueError("Prepared motion owner or attachment changed: " + field)
    positions = observed["joint_positions"]
    if expected.get("model_execution") is True:
        from cais_spade_llm.resources.continuous_motion import ContinuousMotion, occupancy, quaternion_transform

        if continuous_motion is None:
            raise ValueError("Modeled start requires its retained continuous motion configuration")
        motion = ContinuousMotion(continuous_motion["joint_trajectory"], continuous_motion["configuration"])
        errors = motion.joint_position_error
        names = expected["joint_names"]
        if (motion.names != names or set(errors) != set(names) or len(positions) != len(names)
                or len(expected["joint_positions"]) != len(names)
                or any(type(value) not in (int, float) or not math.isfinite(value)
                       or abs(value - target) > errors[name]
                       for name, value, target in zip(names, positions, expected["joint_positions"], strict=True))):
            raise ValueError("Observed joints violate the configured joint_position_error at the prepared start")
        for field in ("base_pose", "root_pose", "geometry_source", "custody_complete", "model_static"):
            if field in expected and observed.get(field) != expected[field]:
                raise ValueError("Prepared physical geometry or custody changed: " + field)
        old_state, new_state = deepcopy(expected.get("observation_state", {})), deepcopy(observed.get("observation_state", {}))
        for state in (old_state, new_state):
            state.pop("current_pose", None)
            state.pop("joint_positions", None)
        if old_state != new_state:
            raise ValueError("Prepared physical geometry or custody changed: observation_state")
        pose = observed["current_pose"]
        if (not isinstance(pose, list) or len(pose) != 7
                or any(type(value) not in (int, float) or not math.isfinite(value) for value in pose)):
            raise ValueError("Observed current_pose is incomplete at the modeled start")
        measured = quaternion_transform(pose[:3], pose[3:])
        reference = motion.transforms(Fraction(0), Fraction(0))[motion.configuration["reference_link"]]
        for i in range(3):
            for j in range(4):
                if measured[i][j][0] < reference[i][j][0] or measured[i][j][1] > reference[i][j][1]:
                    raise ValueError("Observed current_pose is outside the configured initial FK enclosure")
        components = observed.get("component_bounds")
        if not isinstance(components, list) or not components:
            raise ValueError("Observed component_bounds are unavailable at the modeled start")
        boxes = motion.boxes(Fraction(0), Fraction(0))
        fixed_links, known_links = {motion.configuration["root"]}, {motion.configuration["root"]}
        for joint in motion.configuration["joints"]:
            known_links.add(joint["child"])
            if joint["type"] == "fixed" and joint["parent"] in fixed_links:
                fixed_links.add(joint["child"])
        fixed_ids = {row["id"] for row in motion.configuration["components"] if row["link"] in fixed_links}
        fixed_boxes = [box for box in boxes if box["id"] in fixed_ids]
        moving_boxes = [box for box in boxes if box["id"] not in fixed_ids]
        captured_components = expected.get("component_bounds")
        if not isinstance(captured_components, list) or not captured_components:
            raise ValueError("Captured component_bounds are unavailable at the modeled start")
        component_maps, fixed_observations = [], []
        for rows in (captured_components, components):
            identifiers, fixed = {}, []
            for component in rows:
                bounds, identifier, link = component.get("bounds"), component.get("id"), component.get("link")
                if (not isinstance(identifier, str) or not identifier or identifier in identifiers
                        or link not in known_links):
                    raise ValueError("Observed component_bounds require unchanged literal native link identities: "
                                     + str(identifier))
                identifiers[identifier] = link
                if (not isinstance(bounds, list) or len(bounds) != 3
                        or any(not isinstance(pair, list) or len(pair) != 2
                               or any(type(value) not in (int, float) or not math.isfinite(value) for value in pair)
                               or pair[0] > pair[1] for pair in bounds)):
                    raise ValueError("Observed component_bounds are incomplete: " + str(identifier))
                if link in fixed_links:
                    fixed.append({"id": identifier, "minimum": [[pair[0], pair[0]] for pair in bounds],
                                  "maximum": [[pair[1], pair[1]] for pair in bounds]})
            component_maps.append(identifiers)
            fixed_observations.append(fixed)
        if component_maps[0] != component_maps[1]:
            raise ValueError("Observed component_bounds native link population changed")
        if bool(fixed_boxes) != bool(fixed_observations[0]):
            raise ValueError("Observed fixed component_bounds do not cover the configured root-fixed links")
        if fixed_boxes:
            stationary = {"kind": "idle_commanded_hold", "requires_no_running_tasks": True,
                          "requires_no_active_goals": True, "future_execution_tracking": "not_established"}
            if expected.get("stationary_contract") != stationary or observed.get("stationary_contract") != stationary:
                raise ValueError("Fixed component occupancy requires the declared commanded stationary model")
            regions = expected.get("model_execution_regions")
            if not isinstance(regions, dict) or not regions:
                raise ValueError("Fixed component occupancy requires the retained complete model_execution_regions")
            # Only the root-fixed contribution is compared: a moving arm in a
            # region must never conceal a different base occupancy in that region.
            for name, region in regions.items():
                bounds = region.get("bounds") if isinstance(region, dict) else None
                if (not isinstance(region, dict) or region.get("frame") != expected["frame"]
                        or not isinstance(bounds, list) or len(bounds) != 3
                        or any(not isinstance(pair, list) or len(pair) != 2
                               or any(type(value) not in (int, float) or not math.isfinite(value) for value in pair)
                               or pair[0] > pair[1] for pair in bounds)):
                    raise ValueError("Fixed component occupancy requires a complete configured region: " + str(name))
                modeled = occupancy(fixed_boxes, bounds)
                if len(modeled) != 1 or any(occupancy(group, bounds) != modeled for group in fixed_observations):
                    raise ValueError("Fixed component occupancy changed or is unknown in configured region: " + str(name))
        for component in components:
            if component["link"] in fixed_links:
                continue
            bounds = component["bounds"]
            if not any(all(box["minimum"][i][0] <= bounds[i][0] <= bounds[i][1] <= box["maximum"][i][1]
                           for i in range(3)) for box in moving_boxes):
                raise ValueError("Observed component_bounds are outside the configured initial enclosure: "
                                 + str(component.get("id")))
        return
    if (positions != expected["joint_positions"]
            or any(type(value) not in (int, float) or not math.isfinite(value) for value in positions)):
        raise ValueError("Resource moved after physical motion preparation")
    for field in ("current_pose", "base_pose", "root_pose", "component_bounds", "footprint",
                  "geometry_source", "custody_complete", "model_static", "observation_state"):
        if field in expected and observed.get(field) != expected[field]:
            raise ValueError("Prepared physical geometry or custody changed: " + field)


class PreparedRobotEvidence:
    """Bind CCA requests to stored owner preparations and exact command execution.

    Args:
        owner: The initialized ResourceAgent.
        ledger: The run's shared pending/active command ledger.
    """

    def __init__(self, owner, ledger) -> None:
        self.owner = owner
        self.ledger = ledger
        self.steps: dict[str, dict] = {}
        self.preparations: dict[str, dict] = {}
        self.execution_observer = None
        self.dispatch_command = None
        self.execution_coverage_records: dict[str, dict] = {}

    def prepare_execution_coverage(self, *, prepared: dict, checkpoint: dict) -> dict:
        """Bind modeled Gazebo execution without promoting samples to bounds.

        The native controller owns command authorization and observation. Its
        position interface does not constrain Gazebo's intervening physics step,
        so this evidence never establishes a live physical admission certificate.

        Args:
            prepared: Exact program retained by this resource owner.
            checkpoint: Complete observed scene used for native preparation.

        Returns:
            Model coverage and separate physical-bound diagnostics. A prepared
            result remains conditional on its exact interpolation and stationary
            contracts; it grants no command or reservation by itself.
        """
        result = {
            "status": "NEEDS_CONTEXT", "resource_id": self.owner.agent_name,
            "checkpoint_id": checkpoint.get("checkpoint_id"), "dispatch_authorized": False,
            "command_sent": False, "native_contract": None,
            "physical_execution_verified": False,
            "physical_execution_reason": None,
            "native_preparation_verified": False,
            "native_preparation_reason": None,
            "model_execution_verified": False,
            "model_execution_assumptions": None,
            "missing_physical_evidence": [
                "verified tracking bounds", "future stationary containment bounds",
                "failure stopping bounds",
            ],
            "stationary_resources": [],
        }
        identity = fingerprint(prepared)
        result["prepared_fingerprint"] = identity
        try:
            checkpoint_identity = fingerprint(checkpoint)
            result["checkpoint_fingerprint"] = checkpoint_identity
            if prepared.get("status") != "prepared" or prepared.get("resource_id") != self.owner.agent_name:
                raise ValueError("Live coverage requires the responsible owner's prepared program")
            steps = prepared.get("steps")
            if not isinstance(steps, list) or len(steps) != 1 or steps[0].get("primitive") != "move_cartesian":
                raise ValueError("Live motion coverage is limited to one prepared move_cartesian")
            planned = steps[0]
            result["preparation_id"] = planned["preparation_id"]
            with self.ledger.lock:
                if self.steps.get(planned.get("preparation_id")) != planned:
                    raise ValueError("Live coverage differs from the retained owner preparation")
                if any(row["status"] == "active"
                       or row["status"] == "pending" and row.get("reservation_token")
                       for row in self.ledger.commands.values()):
                    raise ValueError("Outstanding physical commands prevent stationary scene coverage")
                ledger_revision = self.ledger.revision
                execution_epoch = self.ledger.execution_epoch
                result.update(command_ledger_revision=ledger_revision, execution_epoch=execution_epoch)
            if (not isinstance(checkpoint.get("checkpoint_id"), str) or not checkpoint["checkpoint_id"]
                    or checkpoint.get("unresolved") != []
                    or not isinstance(checkpoint.get("observations"), dict)
                    or self.owner.agent_name not in checkpoint["observations"]):
                raise ValueError("Live coverage requires a complete physical checkpoint")
            model_execution = checkpoint.get("model_execution") is True
            for resource, row in checkpoint["observations"].items():
                physical = row["physical"]
                if (not model_execution and physical.get("idle") is not True
                        or physical.get("custody_complete") is not True
                        or not isinstance(physical.get("attachment"), dict)
                        or "model_name" not in physical["attachment"]
                        or physical["attachment"]["model_name"] is not None
                        or not isinstance(physical.get("observation_state"), dict)
                        or physical["observation_state"].get("held_part") is not None):
                    raise ValueError("Live coverage requires observed stationary empty custody: " + resource)
            result["stationary_resources"] = sorted(checkpoint["observations"])
            controller = self.owner._controller
            if controller.execution_mode != "simulation":
                raise ValueError("The native motion contract supports Gazebo only")
            result["reason"] = (
                "Native command ownership is available, but the Gazebo position controller has no "
                "verified tracking, future stationary containment, or failure stopping bounds"
            )
            if model_execution:
                from cais_spade_llm.resources.continuous_motion import ContinuousMotion

                continuous = planned["continuous_motion"]
                if (continuous["joint_trajectory"] != planned["joint_trajectory"]
                        or planned["configuration_fingerprint"] != fingerprint(controller.controller_config)):
                    raise ValueError("Modeled execution differs from the retained controller configuration")
                ContinuousMotion(continuous["joint_trajectory"], continuous["configuration"])
                observer = controller.recovery_safety_observer
                if continuous["configuration"] != observer.motion_configuration:
                    raise ValueError("Modeled execution changed its registered kinematic configuration")
                stationary = {}
                for resource, row in checkpoint["observations"].items():
                    declaration = row["physical"].get("stationary_contract")
                    if not isinstance(declaration, dict) or not declaration.get("kind"):
                        raise ValueError("Modeled stationary contract is missing: " + resource)
                    stationary[resource] = deepcopy(declaration)
                goals = checkpoint.get("controller_goals")
                service = observer.configuration["controller_node"].rstrip("/") + "/recovery_state"
                if not isinstance(goals, dict) or service not in goals:
                    raise ValueError("Modeled execution has no observed controller identity")
                controller_goals = {}
                for name, state in goals.items():
                    if (not isinstance(state, dict) or not isinstance(state.get("instance_id"), str)
                            or not state["instance_id"] or state.get("holding") is not True
                            or state.get("has_active_goal") is not False or state.get("has_pending_goal") is not False
                            or state.get("contract_state", "unlocked") != "unlocked"
                            or state.get("reservation_token", "") != ""):
                        raise ValueError("Modeled stationary controller is not in an observed commanded hold: " + name)
                    for field in ("command_revision", "contract_revision"):
                        if field in state and (type(state[field]) is not int or state[field] < 0):
                            raise ValueError("Modeled controller revision is unavailable: " + name)
                    controller_goals[name] = {key: deepcopy(state[key]) for key in (
                        "instance_id", "controller", "command_revision", "contract_revision", "holding",
                        "has_active_goal", "has_pending_goal", "navigation_enabled", "contract_state",
                        "reservation_token", "binding_fingerprint",
                    ) if key in state}
                result.update(
                    status="prepared", model_execution_verified=True,
                    model_execution_assumptions={"continuous_motion": deepcopy(continuous),
                                                 "stationary_contracts": stationary,
                                                 "controller_goals": controller_goals},
                    reason="Gazebo execution is conditional on the retained interpolation, joint_position_error "
                           "and stationary contracts; physical bounds are not certified",
                )
            from cais_spade_llm.resources.robot.gazebo_pick_place_controller import (
                _prepare_controller_contract,
            )

            try:
                record = _prepare_controller_contract(controller, planned, checkpoint)
                result["native_contract"] = deepcopy(record)
                _validate_controller_contract(record, planned)
                result["native_preparation_verified"] = True
                result["physical_execution_reason"] = record["physical_execution_reason"]
            except (ValueError, KeyError, TypeError, AttributeError, RuntimeError, ImportError) as exc:
                result["native_preparation_reason"] = str(exc)
                if not model_execution:
                    raise
            if model_execution and isinstance(result["native_contract"], dict):
                # A rejection for sampled drift is diagnostic. A changed owner
                # incarnation or command revision invalidates this model binding.
                record = result["native_contract"]
                for field in ("instance_id", "command_revision", "contract_revision"):
                    if record.get(field) != goals[service].get(field):
                        raise ValueError("Modeled controller identity changed during preparation: " + field)
            with self.ledger.lock:
                if (self.ledger.revision != ledger_revision or self.ledger.execution_epoch != execution_epoch
                        or self.steps.get(planned["preparation_id"]) != planned
                        or fingerprint(prepared) != identity or fingerprint(checkpoint) != checkpoint_identity):
                    raise ValueError("Owner preparation, checkpoint or command ledger changed during native preparation")
        except (ValueError, KeyError, TypeError, AttributeError, RuntimeError, ImportError) as exc:
            result.update(status="NEEDS_CONTEXT", model_execution_verified=False,
                          native_preparation_verified=False, reason=str(exc))
        with self.ledger.lock:
            self.execution_coverage_records[identity] = deepcopy(result)
        return result

    def register_program(self, prepared: dict) -> None:
        """Retain only this owner's original continuously modeled preparation."""
        if prepared.get("status") != "prepared" or prepared.get("resource_id") != self.owner.agent_name:
            raise ValueError("Cannot register unavailable physical preparation")
        with self.ledger.lock:
            for step in prepared["steps"]:
                identity = step.get("preparation_id")
                if (step.get("observation_status") != "prepared" or not identity
                        or identity != fingerprint({key: value for key, value in step.items()
                                                    if key != "preparation_id"})):
                    raise ValueError("Physical preparation identity is unavailable")
                previous = self.steps.get(identity)
                if previous is not None and previous != step:
                    raise ValueError("Owner preparation changed")
                self.steps[identity] = deepcopy(step)

    def prepare(self, *, resource_agent, request: dict, physical_snapshot: dict) -> dict:
        """Verify every schedule against original native trajectory evidence."""
        if resource_agent is not self.owner or request["resource_jid"] != str(self.owner.jid):
            raise ValueError("Preparation belongs to another resource owner")
        identity = fingerprint(request)
        with self.ledger.lock:
            if identity in self.preparations:
                result = deepcopy(self.preparations[identity]["result"])
                result["command_ledger_revision"] = self.ledger.revision
                return result
            programs = request["event_programs"]
            if not programs:
                raise ValueError("No complete physical schedule supplied")
            selected = None
            for program in programs:
                rows = []
                for result in program["step_results"]:
                    evidence = result["model_evidence"]
                    planned = self.steps.get(evidence.get("preparation_id"))
                    if planned is None:
                        raise ValueError("CCA schedule has no original owner preparation")
                    if (planned["primitive"] != result["primitive"]
                            or planned["params"] != result["resolved_params"]
                            or planned["joint_trajectory"] != evidence["joint_trajectory"]
                            or planned["continuous_motion"] != evidence["continuous_motion"]
                            or planned.get("resolved_motion") != evidence.get("resolved_motion")):
                        raise ValueError("CCA schedule changed the owner motion")
                    rows.append(deepcopy(planned))
                if selected is not None and selected != rows:
                    raise ValueError("Schedules change the prepared primitive program")
                selected = rows
            if len(selected) != len(request["primitive_steps"]):
                raise ValueError("Prepared commands do not cover the complete event")
            command_ids = [self.ledger.prepare(
                resource_id=self.owner.agent_name, task_id=request["task_id"],
                command={"primitive": row["primitive"], "params": row["params"],
                         "joint_trajectory": row["joint_trajectory"]},
                owner_identity={"resource_jid": str(self.owner.jid),
                                "launch_id": row["start"]["launch_id"],
                                "preparation_id": row["preparation_id"]}) for row in selected]
            result = {"status": "prepared", "resource_jid": str(self.owner.jid),
                      "program_hash": request["program_hash"], "execution_mode": "live",
                      "preparation_id": identity, "command_ids": command_ids,
                      "command_ledger_revision": self.ledger.revision}
            self.preparations[identity] = {"request": deepcopy(request), "steps": selected,
                                           "result": result, "executed": set()}
            return deepcopy(result)

    def authorize_preparation(self, *, preparation_id: str, reservation_token: str,
                              command_ids: list[str], expected_revision: int) -> bool:
        """Atomically bind a whole primitive program to the CCA's region grant."""
        with self.ledger.lock:
            prepared = self.preparations[preparation_id]
            if prepared["result"]["command_ids"] != command_ids:
                raise ValueError("Physical reservation names another prepared program")
            self.ledger.authorize_many(command_ids, reservation_token=reservation_token,
                                       expected_revision=expected_revision)
            return True

    def execute_step(self, *, resource_agent, task_id: str, step_index: int,
                     primitive: str, params: dict, grant: dict) -> dict:
        """Consume a granted command once and dispatch its original trajectory."""
        if resource_agent is not self.owner:
            raise ValueError("Prepared executor owner changed")
        with self.ledger.lock:
            prepared = self.preparations[grant["preparation_id"]]
            if task_id != prepared["request"]["task_id"] or step_index in prepared["executed"]:
                raise ValueError("Prepared command task changed or command was consumed")
            planned = deepcopy(prepared["steps"][step_index])
            if planned["primitive"] != primitive or planned["params"] != params:
                raise ValueError("Dispatch parameters differ from the physical preparation")
            command_id = prepared["result"]["command_ids"][step_index]
            command = {"primitive": primitive, "params": params,
                       "joint_trajectory": planned["joint_trajectory"]}
            self.ledger.require_authorized(command_id, command=command)
            prepared["executed"].add(step_index)
            self.ledger.start(command_id, command=command, controller_goal_id=None)
            self.dispatch_command = command_id
        try:
            result = self.owner._controller.execute_prepared_recovery_step(planned)
        except (ValueError, KeyError, RuntimeError, TimeoutError) as exc:
            result = {"success": False, "command_id": command_id, "error": str(exc)}
            self.ledger.finish(command_id, success=False, observations={"error": str(exc)})
            self._publish(task_id, step_index, result)
            raise
        finally:
            self.dispatch_command = None
        self.ledger.finish(command_id, success=result.get("success") is True,
                           observations=result.get("observations", {}))
        self._publish(task_id, step_index, result)
        return result

    def _publish(self, task_id: str, step_index: int, result: dict) -> None:
        if not callable(self.execution_observer):
            raise ValueError("Live execution has no registered monitor observer")
        self.execution_observer(task_id=task_id, step_index=step_index, result=deepcopy(result))

    def dispatch_guard(self, topic: str, trajectory) -> str:
        """Reject physical sends that bypass an admitted original trajectory."""
        with self.ledger.lock:
            identifier = self.dispatch_command
            row = self.ledger.commands.get(identifier)
            if (row is None or row["status"] != "active" or not row.get("reservation_token")
                    or row["command"]["joint_trajectory"] != trajectory_record(trajectory)):
                raise ValueError("Controller dispatch lacks its exact live physical grant")
            if topic != self.owner._controller.arm_trajectory_topic:
                raise ValueError("Prepared trajectory belongs to another controller channel")
            return identifier

    def dispatch_accepted(self, identifier: str, goal_id: list[int]) -> None:
        """Retain the action server's actual accepted goal identity."""
        self.ledger.bind_goal(identifier, goal_id)
