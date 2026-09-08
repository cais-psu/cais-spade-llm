from __future__ import annotations

"""Evaluate an immutable candidate under an explicit rigid vertical assembly model."""

import asyncio
import math
import time
from collections.abc import Callable, Mapping
from contextlib import AsyncExitStack
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from ...adapters.isolated_moveit import IsolatedMoveItSession
from ...adapters.robot_validation_context import matrix_pose, pose_matrix
from ...adapters.target_calculation import CalculationUnavailable, calculate_target
from .composition_context import _resolve_json_pointer
from .primitive_composition import (
    _CompositionInputs,
    _evidence_value,
    _validate_parameter,
    _validate_steps,
)
from .program_dependencies import selected_references
from .program_dependencies import assess_program_dependencies
from .primitive_composition import _result_schema
from .refinement_records import append_record, fingerprint, verify_evidence_tree

VALIDATION_SCOPE = "rigid_vertical_gear_assembly_direct_cartesian"


class BindingUnavailable(ValueError):
    """Report an unavailable selected input without finding a substitute."""


def resolve_selected_values(
    value: Any,
    *,
    read_evidence: Callable[[str, str], Any],
    results: Mapping[int, Mapping[str, Any]],
) -> Any:
    """Resolve selected evidence and earlier results without altering authored values."""
    if isinstance(value, dict):
        if set(value) == {"value_ref"}:
            ref = value["value_ref"]
            return deepcopy(read_evidence(ref["record_ref"], ref["field_path"]))
        if set(value) == {"result_ref"}:
            ref = value["result_ref"]
            if ref["step_index"] not in results:
                raise BindingUnavailable(
                    f"Selected result from step {ref['step_index']} is not available."
                )
            try:
                return deepcopy(
                    _resolve_json_pointer(results[ref["step_index"]], ref["field_path"])
                    if ref["field_path"]
                    else results[ref["step_index"]]
                )
            except ValueError as exc:
                raise BindingUnavailable(
                    "The calculation did not return the selected conditional output."
                ) from exc
        return {
            key: resolve_selected_values(item, read_evidence=read_evidence, results=results)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            resolve_selected_values(item, read_evidence=read_evidence, results=results)
            for item in value
        ]
    return deepcopy(value)


def _finding(
    step: int | None, check: str, status: str, message: str, **details: Any
) -> dict[str, Any]:
    return {"step_index": step, "check": check, "status": status, "message": message, **details}


def _complete(value: Any, schema: Mapping[str, Any], path: str) -> None:
    if not isinstance(schema, Mapping):
        return
    if isinstance(value, Mapping):
        required = set(schema.get("required", [])) | set(schema.get("x-grounding-fields", []))
        missing = required - value.keys()
        if missing:
            raise BindingUnavailable(
                f"{path} is missing required geometry fields: {sorted(missing)}."
            )
        for name, item in value.items():
            _complete(item, schema.get("properties", {}).get(name, {}), path + "/" + name)
    if isinstance(value, list):
        for index, item in enumerate(value):
            _complete(item, schema.get("items", {}), path + "/" + str(index))


def _geometry_sources(step: Mapping[str, Any], inputs: _CompositionInputs) -> list[str]:
    """Require helper geometry to be evidence-backed, while allowing control proposals."""
    params = step["params"]
    warnings = []
    if step["primitive_symbol"] not in {"compute_pick_targets", "compute_place_targets"}:
        return warnings
    for name in ("product_geometry", "target_pose"):
        if name not in params:
            continue
        refs = selected_references(params[name])

        def unmeasured_number(value: Any) -> bool:
            if isinstance(value, dict):
                if set(value) in ({"value_ref"}, {"result_ref"}):
                    return False
                return any(unmeasured_number(item) for item in value.values())
            if isinstance(value, list):
                return any(unmeasured_number(item) for item in value)
            return type(value) in (int, float)

        if unmeasured_number(params[name]):
            raise BindingUnavailable(
                f"/{name} contains a literal measurement without selected evidence."
            )
        if not refs:
            raise BindingUnavailable(
                f"/{name} has no selected geometry evidence; literal measurements are unverified."
            )
        for path, kind, ref in refs:
            if kind != "value_ref":
                continue
            source = _evidence_value(inputs, ref["record_ref"], "")
            uncertainty = source.get("uncertainty", {})
            height_estimate = (
                name == "product_geometry"
                and path == "/part_height_m"
                and ref["field_path"] == "/part_height_m"
                and source.get("record_type") == "AssemblyGeometryEvidence"
                and source.get("status") == "ambiguous"
                and isinstance(uncertainty, Mapping)
                and uncertainty.get("method") == "highest_ranked_qualified_pose_hypothesis"
                and uncertainty.get("hypothesis_index") == 0
                and uncertainty.get("complete_pose_established") is False
                and isinstance(uncertainty.get("source_pose"), dict)
                and isinstance(source.get("warning"), str)
            )
            if name == "target_pose" and source.get("record_type") == "RobotFrameLocationRecord":
                raise BindingUnavailable(
                    "/target_pose selects an observed candidate center, but the CAD-origin "
                    "reference and grasp offset required by this validation scope remain unresolved."
                )
            if (
                source.get("record_type")
                not in {
                    "AssemblyGeometryEvidence",
                    "AssemblySurfaceEvidence",
                    "AssemblyGoalEvidence",
                }
                or (source.get("status") != "accepted" and not height_estimate)
                or source.get("frame_id") != "world"
                or source.get("units") != "m"
            ):
                raise BindingUnavailable(
                    f"/{name} requires accepted geometry with its world frame and semantic reference point established."
                )
            if name == "target_pose" and source.get("reference_point") != "CAD_origin":
                raise BindingUnavailable(
                    "The selected observed location does not establish the helper's part-origin reference point."
                )
            if height_estimate:
                warnings.append(source["warning"])
    return warnings


def _goal_check(
    part_pose: Mapping[str, Any], goal: Mapping[str, Any], specification: Mapping[str, Any]
) -> tuple[bool, dict[str, float]]:
    target = pose_matrix(goal["target_origin_pose"])
    actual = pose_matrix(part_pose)
    position_error = float(np.linalg.norm(actual[:3, 3] - target[:3, 3]))
    axis = np.asarray(goal["part_axis_local"], dtype=float)
    if axis.shape != (3,) or not np.isfinite(axis).all() or np.linalg.norm(axis) < 1e-9:
        raise ValueError("The selected assembly axis is not established.")
    axis /= np.linalg.norm(axis)
    angle = float(
        math.acos(float(np.clip((actual[:3, :3] @ axis) @ (target[:3, :3] @ axis), -1.0, 1.0)))
    )
    metrics = {"position_error_m": position_error, "axis_error_rad": angle}
    passed = (
        position_error <= specification["position_tolerance_m"]
        and angle <= specification["axis_tolerance_rad"]
    )
    if specification.get("yaw_required", False):
        rotation_error = float(Rotation.from_matrix(target[:3, :3].T @ actual[:3, :3]).magnitude())
        metrics["orientation_error_rad"] = rotation_error
        passed = passed and rotation_error <= specification["orientation_tolerance_rad"]
    return passed, metrics


def _grasp_check(
    part: Mapping[str, Any],
    part_pose: Mapping[str, Any],
    ee_pose: Mapping[str, Any],
    robot: Mapping[str, Any],
) -> bool:
    """Check a necessary geometric grasp envelope, not force closure or attachment."""
    tcp = pose_matrix(ee_pose) @ np.asarray(robot["ee_from_tcp"])
    part_origin = pose_matrix(part_pose)
    bounds = part["bounds_m"]
    minimum, maximum = np.asarray(bounds["minimum"]), np.asarray(bounds["maximum"])
    offset = part_origin[:3, 3] - np.asarray([part["origin_pose"][key] for key in ("x", "y", "z")])
    minimum, maximum = minimum + offset, maximum + offset
    tolerance = float(robot["position_tolerance_m"])
    center = part_origin[:3, 3]
    xy_error = np.linalg.norm(tcp[:2, 3] - center[:2])
    width = float(max(maximum[:2] - minimum[:2]))
    return bool(
        xy_error <= tolerance
        and minimum[2] - tolerance <= tcp[2, 3] <= maximum[2] + tolerance
        and width <= float(robot["gripper"]["open_width_mm"]) / 1000
    )


async def validate_program(
    *,
    inputs: _CompositionInputs,
    steps: list[dict[str, Any]],
    robot: Mapping[str, Any] | None,
    evidence: Mapping[str, Mapping[str, str]],
    directory: Path,
    profile: Mapping[str, Any],
    cache: dict[str, dict[str, Any]],
    session_factory: Callable[..., Any] = IsolatedMoveItSession,
) -> dict[str, Any]:
    """Calculate and check the exact candidate; return findings, never repair actions."""
    await asyncio.to_thread(_validate_steps, steps, inputs)
    binding_report = await asyncio.to_thread(
        assess_program_dependencies,
        steps,
        inputs.catalog,
        inputs.composition_input["robot_state"],
        read_evidence=lambda ref, pointer: _evidence_value(inputs, ref, pointer),
        result_schema=lambda ref: _result_schema(steps, ref, inputs),
    )
    root = inputs.root
    findings: list[dict[str, Any]] = []
    records = {}
    expected_types = {
        "part": "AssemblyGeometryEvidence",
        "goal": "AssemblyGeometryEvidence",
        "scene": "AssemblySceneEvidence",
        "specification": "AssemblyValidationSpecification",
    }
    evidence_purposes = {
        "part": "observed part geometry is needed for grasp and carried-part checks",
        "goal": "mating geometry and the final part origin are needed to check the assembly outcome",
        "scene": "observed scene coverage is needed for collision checking",
        "specification": "acceptance criteria must come from approved documents or an explicit experiment specification",
    }
    for role, kind in expected_types.items():
        reference = evidence.get(role)
        if reference is None:
            findings.append(
                _finding(
                    None,
                    role,
                    "unknown",
                    f"Required {role} evidence is missing: {evidence_purposes[role]}.",
                    authority="PA",
                )
            )
            continue
        record = await asyncio.to_thread(verify_evidence_tree, root, reference)
        if record.get("record_type") != kind or record.get("status") != "accepted":
            findings.append(
                _finding(
                    None,
                    role,
                    "unknown",
                    f"The selected {role} record does not establish accepted {kind} evidence.",
                    authority="PA",
                )
            )
            continue
        records[role] = record
    if robot is None:
        findings.append(
            _finding(
                None,
                "robot_context",
                "unknown",
                "Measured joint state and EE/TCP context are unavailable.",
                authority="RA",
            )
        )
        return _report(steps, findings, [], [], None)
    if (
        robot["resource_jid"] != inputs.assignment.selected_resource_jid
        or robot["assignment_fingerprint"] != inputs.assignment.fingerprint
    ):
        raise ValueError("Robot context does not belong to the selected assignment.")
    captured = robot.get("captured_at_ns", 0)
    stamps = [robot.get("joint_state", {}).get("stamp_ns", 0), *robot.get("tf_stamps_ns", [])]
    now_ros = robot.get("measured_at_ros_ns", 0)
    if (
        len(stamps) != 3
        or captured <= 0
        or not 0 <= time.time_ns() - captured <= profile["state_max_age_sec"] * 1e9
        or any(
            stamp <= 0 or not 0 <= now_ros - stamp <= profile["state_max_age_sec"] * 1e9
            for stamp in stamps
        )
        or max(stamps) - min(stamps) > profile["max_capture_skew_sec"] * 1e9
    ):
        findings.append(
            _finding(
                None,
                "robot_freshness",
                "unknown",
                "The measured robot context is stale or has incomplete timestamps.",
                authority="RA",
            )
        )
        return _report(steps, findings, [], [], None)
    for role in ("part", "goal", "scene"):
        if role in records:
            stamp = records[role].get("observation_timestamp_ns", 0)
            if (
                type(stamp) is not int
                or stamp <= 0
                or not 0 <= now_ros - stamp <= profile["scene_max_age_sec"] * 1e9
            ):
                findings.append(
                    _finding(
                        None,
                        "scene_freshness",
                        "unknown",
                        f"The {role} evidence is stale or has no valid observation timestamp.",
                        authority="PA",
                    )
                )
                records.pop(role)
    if "specification" in records:
        spec = records["specification"]
        if (
            spec.get("family") != "vertical_gear_assembly"
            or spec.get("requires_threading")
            or spec.get("requires_force_control")
        ):
            findings.append(
                _finding(
                    None,
                    "coverage",
                    "unknown",
                    "The required assembly operation lies outside rigid vertical gear validation.",
                )
            )
            records.pop("specification")
        else:
            required = ["position_tolerance_m", "axis_tolerance_rad"] + (
                ["orientation_tolerance_rad"] if spec.get("yaw_required") else []
            )
            if any(
                type(spec.get(key)) not in (int, float)
                or not math.isfinite(spec[key])
                or spec[key] <= 0
                for key in required
            ):
                findings.append(
                    _finding(
                        None,
                        "tolerances",
                        "unknown",
                        "Assembly acceptance tolerances are missing or invalid.",
                        authority="PA",
                    )
                )
                records.pop("specification")
    if "scene" in records:
        scene = records["scene"]
        if (
            scene.get("frame_id") != robot["frame_id"]
            or scene.get("coverage") != "all_observed_candidates"
            or scene.get("unresolved_candidates")
            or not scene.get("objects")
        ):
            findings.append(
                _finding(
                    None,
                    "scene",
                    "unknown",
                    "The collision scene does not establish the required observed geometry coverage.",
                    authority="PA",
                )
            )
            records.pop("scene")
    for role in ("part", "goal"):
        if role in records and records[role].get("frame_id") != robot["frame_id"]:
            findings.append(
                _finding(
                    None,
                    "frames",
                    "failed",
                    f"The {role} geometry frame disagrees with the configured planning frame.",
                )
            )
            records.pop(role)
        elif role in records and records[role].get("units") != "m":
            findings.append(
                _finding(None, "units", "failed", f"The {role} geometry does not establish metres.")
            )
            records.pop(role)
    if {"part", "goal"} <= records.keys():
        part_record, goal = records["part"], records["goal"]
        if (
            goal.get("part_object_id") != part_record.get("object_id")
            or goal.get("part_name") != part_record.get("part_name")
            or not part_record.get("assembly_association_sha256")
            or goal.get("assembly_association_sha256") != part_record["assembly_association_sha256"]
            or goal.get("reference_point") != "final_CAD_origin"
        ):
            findings.append(
                _finding(
                    None,
                    "assembly_identity",
                    "failed",
                    "The final relationship and observed part do not share the accepted assembly binding.",
                )
            )
            records.pop("goal")
    if {"part", "scene"} <= records.keys():
        part_record = records["part"]
        instances = [
            item
            for item in records["scene"]["objects"]
            if item.get("object_id") == part_record.get("object_id")
        ]
        if (
            len(instances) != 1
            or instances[0].get("mesh") != part_record.get("mesh")
            or instances[0].get("pose") != part_record.get("origin_pose")
        ):
            findings.append(
                _finding(
                    None,
                    "scene_identity",
                    "failed",
                    "The collision scene does not contain the exact selected observed part geometry.",
                )
            )
            records.pop("scene")
    if "goal" in records and records["goal"].get("nominal_radial_clearance_m", -1) < 0:
        findings.append(
            _finding(
                None,
                "mating_geometry",
                "failed",
                "The selected nominal mating radii do not provide a clearance fit.",
            )
        )
    results: dict[int, Mapping[str, Any]] = {}
    calculations, checked_steps = [], []
    pose, joints = deepcopy(robot["ee_pose"]), deepcopy(robot["joint_state"])
    part = records.get("part")
    part_pose = deepcopy(part.get("origin_pose")) if part else None
    held = deepcopy(
        robot.get("held_part", inputs.composition_input["robot_state"].get("held_part"))
    )
    attached = None
    grasp_transform = None
    prefix_valid = True
    async with AsyncExitStack() as stack:
        session = None
        if "scene" in records:
            try:
                session = await stack.enter_async_context(
                    session_factory(root, robot, records["scene"], profile)
                )
            except (ImportError, OSError, RuntimeError, ValueError) as exc:
                findings.append(
                    _finding(
                        None, "motion", "unknown", f"Isolated motion validation unavailable: {exc}"
                    )
                )
        if held is not None:
            if part is None or part_pose is None or not _grasp_check(part, part_pose, pose, robot):
                findings.append(
                    _finding(
                        None,
                        "held_part",
                        "unknown",
                        "The initial held-part/tool relationship has not been established.",
                        authority="RA",
                    )
                )
                prefix_valid = False
            else:
                grasp_transform = np.linalg.inv(pose_matrix(pose)) @ pose_matrix(part_pose)
                attached = {
                    "object_id": part["object_id"],
                    "mesh": part["mesh"],
                    "pose": matrix_pose(grasp_transform),
                    "touch_links": robot.get("touch_links", [robot["ee_link"]]),
                }
                if session:
                    await session.change_custody(remove=part["object_id"])
        for index, step in enumerate(steps, start=1):
            symbol = step["primitive_symbol"]
            if not prefix_valid:
                checked_steps.append(
                    {
                        "step_index": index,
                        "status": "unknown",
                        "message": "The preceding program state was not validated.",
                    }
                )
                continue
            try:
                incompatible = [
                    item
                    for item in binding_report["issues"]
                    if item["step_index"] == index and item["status"] == "incompatible"
                ]
                if incompatible:
                    raise ValueError(" ".join(item["message"] for item in incompatible))
                if symbol == "move_cartesian":
                    unknown_frames = [
                        item
                        for item in binding_report["issues"]
                        if item["step_index"] == index
                        and item["status"] in {"unverified", "missing"}
                    ]
                    if unknown_frames:
                        raise BindingUnavailable(
                            " ".join(item["message"] for item in unknown_frames)
                        )
                params = await asyncio.to_thread(
                    resolve_selected_values,
                    step["params"],
                    read_evidence=lambda ref, pointer: _evidence_value(inputs, ref, pointer),
                    results=results,
                )
                entry = inputs.catalog[symbol]
                for field, condition in entry.get("conditions", {}).items():
                    if (
                        field != "held_part"
                        or not isinstance(condition, dict)
                        or len(condition) != 1
                        or not set(condition) <= {"equals", "not_equals"}
                    ):
                        raise BindingUnavailable(
                            "A declared condition has no evaluator in this validation scope."
                        )
                    expected = condition.get("equals", condition.get("not_equals"))
                    if (held == expected) != ("equals" in condition):
                        raise ValueError("The declared held_part condition is not satisfied.")
                supported_effects = {"held_part", "current_pose", "current_pose_ref"}
                if not set(entry.get("effects", {})) <= supported_effects:
                    raise BindingUnavailable(
                        "A declared effect has no evaluator in this validation scope."
                    )
                schemas = entry["parameter_schemas"]
                required = {
                    parameter["name"]
                    for parameter in entry["typed_parameters"]
                    if parameter["required"]
                }
                required.update(
                    key
                    for key, schema in schemas.items()
                    if isinstance(schema, Mapping) and schema.get("x-grounding-required")
                )
                if not required <= params.keys():
                    raise BindingUnavailable(
                        f"Required inputs remain unbound: {sorted(required - params.keys())}."
                    )
                for name, value in params.items():
                    await asyncio.to_thread(
                        _validate_parameter,
                        value,
                        schemas[name],
                        inputs,
                        [],
                        allow_references=False,
                    )
                    _complete(value, schemas[name], "/" + name)
                if symbol in {"compute_pick_targets", "compute_place_targets"}:
                    warnings = await asyncio.to_thread(_geometry_sources, step, inputs)
                    findings.extend(
                        _finding(index, "part_height_m", "warning", message)
                        for message in warnings
                    )
                    source_refs = [
                        {
                            "ref": ref["record_ref"],
                            "sha256": inputs.record_hashes[ref["record_ref"]],
                        }
                        for _, kind, ref in selected_references(step["params"])
                        if kind == "value_ref"
                    ]
                    for source in source_refs:
                        await asyncio.to_thread(verify_evidence_tree, root, source)
                    robot_inputs = {
                        key: robot[key]
                        for key in (
                            "configuration_sha256",
                            "model_parameters_sha256",
                            "ee_from_tcp",
                            "policy",
                            "frame_id",
                        )
                    }
                    key = fingerprint(
                        {
                            "primitive_symbol": symbol,
                            "params": params,
                            "robot": robot_inputs,
                            "source_refs": source_refs,
                            "preceding_pose": pose,
                        }
                    )
                    reused = key in cache
                    if not reused:
                        cache[key] = await asyncio.to_thread(
                            calculate_target, symbol, params, robot, pose
                        )
                    output = deepcopy(cache[key])
                    for name, value in output.items():
                        if name in entry["result_schemas"]:
                            await asyncio.to_thread(
                                _validate_parameter,
                                value,
                                entry["result_schemas"][name],
                                inputs,
                                [],
                                allow_references=False,
                            )
                    results[index] = output
                    reference = await asyncio.to_thread(
                        append_record,
                        root,
                        directory,
                        f"calculation_{index:04d}.json",
                        {
                            "record_type": "PrimitiveCalculationRecord",
                            "primitive_symbol": symbol,
                            "step_index": index,
                            "resolved_params": params,
                            "result": output,
                            "frame_id": robot["frame_id"],
                            "preceding_pose": pose,
                            "robot_inputs": robot_inputs,
                            "source_refs": source_refs,
                            "input_fingerprint": key,
                            "reused": reused,
                            "created_at_ns": time.time_ns(),
                        },
                    )
                    calculations.append(reference)
                    checked_steps.append(
                        {
                            "step_index": index,
                            "status": "passed",
                            "resolved_params": params,
                            "calculation_ref": reference,
                        }
                    )
                elif symbol == "move_cartesian":
                    target = {key: params[key] for key in ("x", "y", "z")}
                    orientation = {
                        key: params[key] for key in ("qx", "qy", "qz", "qw") if key in params
                    }
                    if orientation and len(orientation) != 4:
                        raise ValueError("An orientation requires all four quaternion fields.")
                    target.update(
                        orientation or {key: pose[key] for key in ("qx", "qy", "qz", "qw")}
                    )
                    pose_matrix(target)
                    if "speed" in params and params["speed"] <= 0:
                        raise ValueError("The proposed trajectory time scale must be positive.")
                    if session is None:
                        raise BindingUnavailable(
                            "A complete collision scene and isolated planner are required for the proposed movement."
                        )
                    result = await session.check_segment(
                        joints=joints, start_pose=pose, target_pose=target, attached=attached
                    )
                    checked_steps.append({"step_index": index, "resolved_params": params, **result})
                    if result["status"] != "passed":
                        findings.append(
                            _finding(index, "motion", result["status"], result["message"])
                        )
                        prefix_valid = False
                        continue
                    pose, joints = target, deepcopy(result["end_joint_state"])
                    if grasp_transform is not None:
                        part_pose = matrix_pose(pose_matrix(pose) @ grasp_transform)
                elif symbol == "grasp_part":
                    if held is not None:
                        raise ValueError("The declared held_part precondition is not satisfied.")
                    if part is None or part_pose is None:
                        raise BindingUnavailable(
                            "Observed part geometry is required to assess the proposed grasp."
                        )
                    if params.get("part_name") != part.get("part_name"):
                        raise ValueError(
                            "The grasped part differs from PA's bound physical instance."
                        )
                    if not _grasp_check(part, part_pose, pose, robot):
                        raise ValueError(
                            "The proposed controlled-link/TCP pose does not satisfy the geometric grasp envelope."
                        )
                    held = params["part_name"]
                    grasp_transform = np.linalg.inv(pose_matrix(pose)) @ pose_matrix(part_pose)
                    attached = {
                        "object_id": part["object_id"],
                        "mesh": part["mesh"],
                        "pose": matrix_pose(grasp_transform),
                        "touch_links": robot.get("touch_links", [robot["ee_link"]]),
                    }
                    if session:
                        await session.change_custody(remove=part["object_id"])
                    checked_steps.append(
                        {
                            "step_index": index,
                            "status": "passed",
                            "message": "Geometric grasp compatibility checked under the rigid-grasp assumption.",
                        }
                    )
                elif symbol == "release_part":
                    if held is None:
                        raise ValueError("The declared held_part precondition is not satisfied.")
                    if params.get("part_name") != part.get("part_name"):
                        raise ValueError(
                            "The release selects a different part from the carried instance."
                        )
                    if not {"goal", "specification"} <= records.keys() or part_pose is None:
                        raise BindingUnavailable(
                            "A resolved seating relationship and acceptance tolerances are needed to assess release/support."
                        )
                    seated, metrics = _goal_check(
                        part_pose, records["goal"], records["specification"]
                    )
                    if not seated:
                        raise ValueError(
                            "The proposed release does not establish the required seated part relationship: "
                            + str(metrics)
                        )
                    held, grasp_transform, attached = None, None, None
                    if session:
                        await session.change_custody(
                            add={
                                "object_id": part["object_id"],
                                "mesh": part["mesh"],
                                "pose": part_pose,
                            }
                        )
                    checked_steps.append(
                        {"step_index": index, "status": "passed", "metrics": metrics}
                    )
                else:
                    raise BindingUnavailable(
                        "This primitive has no evaluator in the declared vertical assembly validation scope."
                    )
            except (BindingUnavailable, CalculationUnavailable) as exc:
                findings.append(
                    _finding(
                        index,
                        "bindings" if isinstance(exc, BindingUnavailable) else "calculation",
                        "unknown",
                        str(exc),
                    )
                )
                checked_steps.append(
                    {"step_index": index, "status": "unknown", "message": str(exc)}
                )
                # A pure calculation cannot invalidate the known robot state;
                # its unresolved result still blocks consumers of that result.
                if symbol not in {"compute_pick_targets", "compute_place_targets"}:
                    prefix_valid = False
            except (ImportError, OSError, RuntimeError) as exc:
                findings.append(_finding(index, "runtime", "unknown", str(exc)))
                if symbol not in {"compute_pick_targets", "compute_place_targets"}:
                    prefix_valid = False
            except (KeyError, TypeError, ValueError) as exc:
                findings.append(_finding(index, "conditions_or_geometry", "failed", str(exc)))
                checked_steps.append({"step_index": index, "status": "failed", "message": str(exc)})
                if symbol not in {"compute_pick_targets", "compute_place_targets"}:
                    prefix_valid = False
    if prefix_valid and {"part", "goal", "specification", "scene"} <= records.keys():
        seated, metrics = _goal_check(part_pose, records["goal"], records["specification"])
        findings.append(
            _finding(
                None,
                "assembly_outcome",
                "passed" if seated and held is None else "failed",
                "Predicted final seating and destination support checked; physical assembly is unobserved.",
                held_part=held,
                **metrics,
            )
        )
    else:
        findings.append(
            _finding(
                None,
                "assembly_outcome",
                "unknown",
                "The complete predicted assembly outcome could not be established.",
            )
        )
    return _report(
        steps,
        findings,
        checked_steps,
        calculations,
        {"ee_pose": pose, "part_pose": part_pose, "held_part": held},
    )


def _report(
    steps: list[dict[str, Any]],
    findings: list[dict[str, Any]],
    checked_steps: list[dict[str, Any]],
    calculations: list[dict[str, str]],
    final_state: Any,
) -> dict[str, Any]:
    statuses = {item["status"] for item in [*findings, *checked_steps]}
    status = (
        "failed"
        if "failed" in statuses
        else "unknown"
        if "unknown" in statuses or len(checked_steps) != len(steps)
        else "passed"
    )
    return {
        "record_type": "PrimitiveValidationReport",
        "status": status,
        "scope": VALIDATION_SCOPE,
        "candidate_fingerprint": fingerprint(steps),
        "findings": findings,
        "checked_steps": checked_steps,
        "calculation_refs": calculations,
        "predicted_final_state": final_state,
        "unmodeled": [
            "force closure",
            "contact dynamics",
            "physical assembly success",
            "unobserved environment geometry",
        ],
        "motion_executed": False,
        "created_at_ns": time.time_ns(),
    }
