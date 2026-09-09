from __future__ import annotations

"""Exercise audited calculations and bounded refinement without external model calls or motion."""

import asyncio
from contextlib import asynccontextmanager
import json
import logging
import time
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from cais_spade_llm.spec2primitives.agents.ra import primitive_composition
from cais_spade_llm.spec2primitives.agents.ra.validation_scope import (
    GAZEBO_OBSERVED_SCOPE, GAZEBO_PICK_PLACE_SCOPE, VALIDATION_SCOPE, read_validation_scope, required_validation_roles,
)

from cais_spade_llm.spec2primitives.agents.pa.primitive_context import (
    ProductPrimitiveContextRuntime,
    _issued_records,
    _observation_catalog,
    _pa_evidence_projection,
)
from cais_spade_llm.spec2primitives.adapters.target_calculation import (
    CalculationUnavailable,
    calculate_target,
)
from cais_spade_llm.spec2primitives.agents.ra.primitive_composition import (
    _evidence_value,
    _load_inputs,
    _result_schema,
    _with_refinement,
    author_primitive_program_candidate,
    read_primitive_composition_diagnostic,
)
from cais_spade_llm.spec2primitives.agents.ra.program_dependencies import (
    assess_program_dependencies,
)
from cais_spade_llm.spec2primitives.agents.ra.program_validation import (
    BindingUnavailable,
    _geometry_sources,
    _grasp_check,
    resolve_selected_values,
    validate_program,
)
from cais_spade_llm.spec2primitives.agents.ra.refinement import (
    PrimitiveRefinementRuntime,
    _robot_changed,
    cancel_primitive_refinement,
    enrich_composition_diagnostic,
    load_refinement_profile,
)
from cais_spade_llm.spec2primitives.agents.ra.refinement_records import (
    append_record,
    pin,
    read_pin,
    verify_evidence_tree,
)
from cais_spade_llm.spec2primitives.tests.test_cad_pose_estimation import (
    _accepted_registration,
    _hypothesis,
    _prepare_correspondence,
)
from cais_spade_llm.spec2primitives.tests.test_ra_context_handoff import (
    _ProgramRuntime,
    _capture_geometry_context,
    _program_action,
)
from cais_spade_llm.spec2primitives.tools.assembly_geometry import (
    AssemblyGeometryProducer,
    planar_features,
)
from cais_spade_llm.spec2primitives.tools.observation_presentation import ObservationPresentation
from cais_spade_llm.spec2primitives.tools.rgb_d_cad_grounding import (
    pose_estimation,
    record_camera_to_robot_calibration,
)


def _legacy_profile() -> dict[str, Any]:
    return {**load_refinement_profile(), "validation_scope": GAZEBO_PICK_PLACE_SCOPE}


def _pose(x: float = 0.0, z: float = 0.3) -> dict[str, float]:
    return {"x": x, "y": 0.0, "z": z, "qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0}


def _robot(inputs: Any) -> dict[str, Any]:
    tool = np.eye(4)
    tool[2, 3] = -0.1
    return {
        "record_type": "RobotValidationContext",
        "resource_jid": inputs.assignment.selected_resource_jid,
        "assignment_fingerprint": inputs.assignment.fingerprint,
        "frame_id": "world",
        "ee_link": "configured_ee",
        "tcp_link": "configured_tcp",
        "group_name": "xarm6",
        "ee_pose": _pose(),
        "ee_from_tcp": tool.tolist(),
        "joint_state": {"names": ["joint1"], "positions": [0.0], "stamp_ns": 1000000000},
        "tf_stamps_ns": [1000000000, 1000000000],
        "configuration_sha256": "a" * 64,
        "model_parameters_sha256": "b" * 64,
        "model_parameters": {"robot_description": "private robot model"},
        "position_tolerance_m": 0.001,
        "policy": {
            "approach_height_m": 0.1,
            "pick_tcp_z_bias_min_m": 0.003,
            "pick_tcp_z_bias_max_m": 0.02,
            "min_pick_tcp_z_m": 0.0,
            "place_surface_gap_m": 0.0,
            "insertion_depth_m": 0.01,
            "pick_z_adjustments_m": {},
        },
        "gripper": {"open_width_mm": 80.0},
        "measured_at_ros_ns": 1000000000,
        "captured_at_ns": time.time_ns(),
    }


def _evidence(
    root: Path, *, cad_origin_offset: tuple[float, float, float] | None = None,
) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]]]:
    shift = np.asarray(cad_origin_offset or (0.0, 0.0, 0.0))
    np.savez(
        root / "mesh.npz", triangles_m=np.asarray([[[0, 0, 0], [0.02, 0, 0], [0, 0.02, 0.02]]]) + shift
    )
    common = {
        "record_type": "AssemblyGeometryEvidence",
        "status": "accepted",
        "frame_id": "world",
        "units": "m",
        "observation_timestamp_ns": 1000000000,
        "assembly_association_sha256": "c" * 64,
    }
    part = {
        **common,
        "part_name": "medium gear",
        "object_id": "observed_part",
        "reference_point": "CAD_origin",
        "origin_pose": _pose(z=0.05),
        "part_height_m": 0.02,
        "bounds_m": {"minimum": [-0.02, -0.02, 0.04], "maximum": [0.02, 0.02, 0.06]},
        "mesh": pin(root, root / "mesh.npz"),
    }
    pick = {
        **common,
        "reference_point": "CAD_origin",
        "target_pose": {"x": 0.0, "y": 0.0, "z": 0.05},
        "product_geometry": {"board_center": {"z": 0.04}, "part_height_m": 0.02},
    }
    goal = {
        **common,
        "reference_point": "final_CAD_origin",
        "part_name": "medium gear",
        "part_object_id": "observed_part",
        "part_axis_local": [0.0, 0.0, 1.0],
        "target_origin_pose": _pose(x=0.2, z=0.05),
        "nominal_radial_clearance_m": 0.001,
        "product_geometry": {
            "board_center": {"x": 0.2, "y": 0.0},
            "slot_xy": [0.0, 0.0],
            "slot_floor_z_m": 0.04,
            "part_height_m": 0.02,
            "target_reference": {
                "target_point": "inserted_part_origin",
                "surface_role": "assembly_slot",
            },
            "target_origin_pose": {"x": 0.2, "y": 0.0, "z": 0.05},
        },
    }
    if cad_origin_offset is not None:
        part["grasp_reference"] = {
            "reference_point": "selected_CAD_feature", "point_CAD_m": shift.tolist(),
            "point_world_m": [0.0, 0.0, 0.05], "plane_id": "plane_0001", "circle_id": "circle_0001",
        }
        pick["reference_point"] = "selected_CAD_feature"
        pick["grasp_reference"] = deepcopy(part["grasp_reference"])
        goal["product_geometry"]["target_reference"]["grasp_point"] = "selected_CAD_feature"
        goal["product_geometry"]["grasp_point_offset_world_m"] = shift.tolist()
        for axis, delta in zip(("x", "y", "z"), shift, strict=True):
            part["origin_pose"][axis] -= float(delta)
            goal["target_origin_pose"][axis] -= float(delta)
            goal["product_geometry"]["target_origin_pose"][axis] -= float(delta)
            if axis in ("x", "y"):
                goal["product_geometry"]["board_center"][axis] -= float(delta)
    scene = {
        "record_type": "AssemblySceneEvidence",
        "status": "accepted",
        "frame_id": "world",
        "coverage": "all_observed_candidates",
        "unresolved_candidates": [],
        "objects": [
            {"object_id": "observed_part", "pose": part["origin_pose"], "mesh": part["mesh"]}
        ],
        "observation_timestamp_ns": 1000000000,
    }
    spec = {
        "record_type": "AssemblyValidationSpecification",
        "status": "accepted",
        "family": "vertical_gear_assembly",
        "position_tolerance_m": 0.001,
        "axis_tolerance_rad": 0.01,
        "yaw_required": False,
    }
    references = {
        name: append_record(root, root / "products/test_evidence", name + ".json", value)
        for name, value in {
            "part": part,
            "pick": pick,
            "goal": goal,
            "scene": scene,
            "specification": spec,
        }.items()
    }
    return references, {key: references[key] for key in ("part", "goal", "scene", "specification")}


def _value(ref: dict[str, str], path: str) -> dict[str, Any]:
    return {"value_ref": {"record_ref": ref["ref"], "field_path": path}}


def _result(index: int, path: str) -> dict[str, Any]:
    return {"result_ref": {"step_index": index, "field_path": path}}


def _program(
    refs: dict[str, dict[str, str]], *, lift: bool = True, insert: bool = True, bound: bool = True
) -> list[dict[str, Any]]:
    steps: list[tuple[str, dict[str, Any]]] = []
    pick = {"part_name": "medium gear"}
    if bound:
        pick.update(
            target_pose=_value(refs["pick"], "/target_pose"),
            product_geometry=_value(refs["pick"], "/product_geometry"),
        )
    steps.append(("compute_pick_targets", pick))

    def move(index: int, name: str) -> None:
        steps.append(
            (
                "move_cartesian",
                {key: _result(index, "/" + name + "/" + key) for key in ("x", "y", "z")},
            )
        )

    move(1, "approach_pose")
    move(1, "target_pose")
    steps.append(("grasp_part", {"part_name": "medium gear"}))
    if lift:
        move(1, "approach_pose")
    place_index = len(steps) + 1
    place = {
        "part_name": "medium gear",
        "pick_ctx": {
            key: _result(1, "/" + key)
            for key in ("part_name", "part_height", "pick_tcp_z", "tcp_offset_z", "tz")
        },
    }
    if bound:
        place["product_geometry"] = _value(refs["goal"], "/product_geometry")
    steps.append(("compute_place_targets", place))
    move(place_index, "pre_insert_pose")
    if insert:
        move(place_index, "insert_pose")
    steps.append(("release_part", {"part_name": "medium gear"}))
    move(place_index, "pre_insert_pose")
    return [{"primitive_symbol": symbol, "params": params} for symbol, params in steps]


class _PlanningSession:
    def __init__(self, *args: Any) -> None:
        self.calls: list[dict[str, Any]] = []
        self.custody: list[dict[str, Any]] = []

    async def __aenter__(self) -> _PlanningSession:
        return self

    async def __aexit__(self, *args: Any) -> None:
        return None

    async def check_segment(self, **request: Any) -> dict[str, Any]:
        self.calls.append(deepcopy(request))
        if (
            request["attached"]
            and request["start_pose"]["z"] < 0.2
            and abs(request["target_pose"]["x"] - request["start_pose"]["x"]) > 0.1
        ):
            return {
                "status": "failed",
                "message": "The carried part intersects the modeled shaft during this segment.",
            }
        return {
            "status": "passed",
            "message": "Controlled fixture segment accepted.",
            "end_joint_state": request["joints"],
        }

    async def change_custody(self, **values: Any) -> None:
        self.custody.append(deepcopy(values))


def _setup(
    root: Path, *, cad_origin_offset: tuple[float, float, float] | None = None,
) -> tuple[Any, dict[str, Any], dict[str, Any], dict[str, Any]]:
    _capture_geometry_context(root)
    inputs = _load_inputs(root)
    refs, roles = _evidence(root, cad_origin_offset=cad_origin_offset)
    inputs = replace(
        inputs,
        record_hashes={
            **inputs.record_hashes,
            **{item["ref"]: item["sha256"] for item in refs.values()},
        },
    )
    return inputs, _robot(inputs), refs, roles


def _observed_setup(root: Path) -> tuple[Any, dict[str, Any], dict[str, Any], dict[str, Any]]:
    inputs, robot, old_refs, _ = _setup(root)
    part = read_pin(root, old_refs["part"])
    part.update(
        record_type="ObservedGeometryEvidence", reference_point="observed_bounds_center",
        reference_pose=part.pop("origin_pose"), size_m=[0.04, 0.04, 0.02],
        product_geometry={"part_height_m": 0.02, "board_center": {"z": 0.04}},
        uncertainty={"CAD_orientation": "not_established", "occluded_geometry": "unmeasured"},
    )
    part.pop("mesh")
    scene = read_pin(root, old_refs["scene"])
    scene.update(geometry_model="observed_bounds", objects=[{
        "object_id": part["object_id"], "size_m": part["size_m"], "pose": part["reference_pose"],
    }])
    refs = {
        name: append_record(root, root / "observed", name + ".json", value)
        for name, value in {
            "part": part, "scene": scene,
            "pick": {**part, "target_pose": {key: part["reference_pose"][key] for key in ("x", "y", "z")}},
            "goal": {**part, "product_geometry": {
                "placement_surface_point": {"x": 0.2, "y": 0.0, "z": 0.04},
            }},
        }.items()
    }
    robot["gripper"]["open_width_mm"] = 1.0
    robot["touch_links"] = ["configured_finger"]
    inputs = primitive_composition._with_scope(replace(inputs, record_hashes={
        **inputs.record_hashes, **{ref["ref"]: ref["sha256"] for ref in refs.values()},
    }), GAZEBO_OBSERVED_SCOPE)
    return inputs, robot, refs, {role: refs[role] for role in ("part", "scene")}


def _observed_program(refs: dict[str, Any], *, bound: bool = True) -> list[dict[str, Any]]:
    steps = _program(refs, bound=bound)
    if bound:
        steps[0]["params"]["target_pose"] = _value(refs["part"], "/reference_pose")
    for step in steps:
        if step["primitive_symbol"] == "move_cartesian":
            for ref in step["params"].values():
                path = ref["result_ref"]["field_path"]
                ref["result_ref"]["field_path"] = path.replace("pre_insert_pose", "approach_pose").replace("insert_pose", "target_pose")
    return steps


@pytest.mark.parametrize("quantity", ["part_height", "support_height", "destination_height"])
def test_observed_heights_drive_the_corresponding_controlled_link_poses(quantity: str) -> None:
    """Keep three measured heights distinct while compensating the measured tool offset."""
    robot = _robot(SimpleNamespace(assignment=SimpleNamespace(selected_resource_jid="fixture", fingerprint="fixture")))
    robot["ee_from_tcp"][0][3] = 0.03
    pick_params = {
        "part_name": "medium gear", "target_pose": {"x": 0.0, "y": 0.0, "z": 0.05},
        "product_geometry": {"part_height_m": 0.02, "board_center": {"z": 0.04}},
        "ignore_current_height_for_travel_z": True,
    }

    def calculate() -> tuple[dict[str, Any], dict[str, Any]]:
        pick = calculate_target("compute_pick_targets", pick_params, robot, robot["ee_pose"], validation_scope=GAZEBO_OBSERVED_SCOPE)
        place = calculate_target("compute_place_targets", {
            "part_name": "medium gear", "pick_ctx": pick,
            "product_geometry": {"placement_surface_point": {"x": 0.2, "y": 0.0, "z": destination_z}},
        }, robot, robot["ee_pose"], validation_scope=GAZEBO_OBSERVED_SCOPE)
        return pick, place

    destination_z = 0.04
    before_pick, before_place = calculate()
    if quantity == "part_height":
        pick_params["product_geometry"]["part_height_m"] = 0.04
    elif quantity == "support_height":
        pick_params["product_geometry"]["board_center"]["z"] = 0.5
    else:
        destination_z += 0.1
    pick, place = calculate()
    assert pick["target_pose"]["x"] == pytest.approx(-0.03)
    assert place["target_pose"]["x"] == pytest.approx(0.17)
    if quantity == "part_height":
        assert pick["target_pose"]["z"] > before_pick["target_pose"]["z"]
        assert place["target_pose"]["z"] > before_place["target_pose"]["z"]
    elif quantity == "support_height":
        assert pick["approach_pose"]["z"] > before_pick["approach_pose"]["z"]
        assert pick["target_pose"] == before_pick["target_pose"]
        assert place == before_place
    else:
        assert pick == before_pick
        assert place["target_pose"]["z"] - before_place["target_pose"]["z"] == pytest.approx(0.1)


@pytest.mark.parametrize("fault", [
    None, "missing_input", "scene", "identity", "stale", "custody", "legacy_scope",
    "literal_geometry", "literal_target_pose",
])
def test_observed_scope_validates_boxes_without_cad_orientation_or_jaw_fit(tmp_path: Path, fault: str | None) -> None:
    inputs, robot, refs, roles = _observed_setup(tmp_path)
    steps = _observed_program(refs)
    profile = load_refinement_profile()
    if fault == "missing_input":
        del steps[5]["params"]["product_geometry"]
    elif fault == "scene":
        roles.pop("scene")
    elif fault == "identity":
        steps[3]["params"]["part_name"] = "different gear"
    elif fault == "stale":
        robot["captured_at_ns"] -= 3_000_000_000
    elif fault == "custody":
        steps = [step for step in steps if step["primitive_symbol"] != "release_part"]
    elif fault == "legacy_scope":
        profile = _legacy_profile()
    elif fault in {"literal_geometry", "literal_target_pose"}:
        parameter = "product_geometry" if fault == "literal_geometry" else "target_pose"
        selected = steps[0]["params"][parameter]["value_ref"]
        steps[0]["params"][parameter] = primitive_composition._evidence_value(
            inputs, selected["record_ref"], selected["field_path"],
        )
    before = deepcopy(steps)
    session = _PlanningSession()
    scenes = []

    def planner(root: Path, context: Any, scene: Any, profile: Any) -> Any:
        scenes.append(scene)
        return session

    report = asyncio.run(validate_program(
        inputs=inputs, steps=steps, robot=robot, evidence=roles, directory=tmp_path / "validation",
        profile=profile, cache={}, session_factory=planner,
    ))
    assert steps == before
    if fault is not None:
        assert report["status"] != "passed", report
        if fault in {"literal_geometry", "literal_target_pose"}:
            assert any("literal measurement without selected evidence" in finding["message"]
                       for finding in report["findings"])
        return
    assert report["status"] == "passed", report["findings"]
    assert scenes[0]["allowed_contacts"] == [{"object_id": "observed_part", "links": ["configured_finger"]}]
    assert any(call["attached"] and call["attached"]["size_m"] == [0.04, 0.04, 0.02] for call in session.calls)
    assert report["predicted_final_state"]["held_part"] is None
    assert not any(item["check"] in {"goal", "specification"} for item in report["findings"])
    for ref in report["calculation_refs"]:
        calculation = read_pin(tmp_path, ref)
        assert calculation["validation_scope"] == GAZEBO_OBSERVED_SCOPE
        assert calculate_target(
            calculation["primitive_symbol"], calculation["resolved_params"], robot,
            calculation["preceding_pose"], validation_scope=GAZEBO_OBSERVED_SCOPE,
        ) == calculation["result"]


def test_robot_comparison_retains_thresholds_and_reports_all_differences() -> None:
    previous = _robot(
        SimpleNamespace(
            assignment=SimpleNamespace(selected_resource_jid="xarm6@localhost", fingerprint="a")
        )
    )
    previous["joint_state"]["names"].append("ur5e_wrist_3_joint")
    previous["joint_state"]["positions"].append(0.0)
    current = deepcopy(previous)
    current["captured_at_ns"] += 10**9
    current["measured_at_ros_ns"] += 10**9
    current["tf_stamps_ns"] = [2 * 10**9, 2 * 10**9]
    current["joint_state"] = {
        "names": ["ur5e_wrist_3_joint", "joint1"],
        "positions": [2.6168797795378396e-8, 0.001],
        "stamp_ns": 2 * 10**9,
    }
    current["ee_from_tcp"][0][3] = 1e-6
    profile = _legacy_profile()
    assert _robot_changed(previous, current, profile) == []

    for field in (
        "configuration_sha256",
        "model_parameters_sha256",
        "frame_id",
        "ee_link",
        "tcp_link",
        "held_part",
    ):
        current[field] = "changed"
    current["joint_state"] = {
        "names": ["ur5e_wrist_3_joint", "new_joint"],
        "positions": [0.002, 0.0],
        "stamp_ns": 2 * 10**9,
    }
    current["ee_from_tcp"][0][3] = 2e-6
    differences = {item["field"]: item for item in _robot_changed(previous, current, profile)}
    assert set(differences) == {
        "configuration_sha256",
        "model_parameters_sha256",
        "frame_id",
        "ee_link",
        "tcp_link",
        "held_part",
        "ee_from_tcp[0][3]",
        "joint_state.names",
        "joint_state.positions['ur5e_wrist_3_joint']",
    }
    assert differences["joint_state.names"]["missing"] == ["joint1"]
    assert differences["joint_state.names"]["added"] == ["new_joint"]
    assert differences["ee_from_tcp[0][3]"]["difference"] == 2e-6
    assert differences["ee_from_tcp[0][3]"]["tolerance"] == 1e-6
    joint = differences["joint_state.positions['ur5e_wrist_3_joint']"]
    assert (joint["previous"], joint["current"], joint["difference"], joint["tolerance"]) == (
        0.0,
        0.002,
        0.002,
        0.001,
    )
    assert differences["held_part"]["previous"] is None


@pytest.mark.parametrize("after_validation", [False, True])
def test_robot_comparison_preserves_rejected_capture_and_diagnostics(
    tmp_path: Path, after_validation: bool
) -> None:
    """Persist both captures without accepting a changed context or provisional pass."""
    inputs, robot, _, _ = _setup(tmp_path)
    original = {path: path.read_bytes() for path in tmp_path.rglob("*.json")}
    program = _ProgramRuntime([_program_action([("grasp_part", {"part_name": "medium gear"})])] * 2)
    events = []
    captured = []

    class Robot:
        async def capture_validation_context(self, *args: Any, **kwargs: Any) -> Any:
            context = deepcopy(robot)
            if captured:
                context["configuration_sha256"] = "c" * 64
                context["joint_state"]["positions"][0] = 0.003
                context["ee_from_tcp"][0][3] = 0.001
            captured.append(context)
            return context

    async def validator(**kwargs: Any) -> dict[str, Any]:
        return {"scope": read_validation_scope(kwargs["profile"]), "status": "passed" if after_validation else "unknown", "findings": []}

    async def progress(event: Any) -> None:
        events.append(deepcopy(event))

    runtime = PrimitiveRefinementRuntime(
        program_runtime=program,
        robot_runtime=Robot(),
        validator=validator,
        profile={**_legacy_profile(), "max_candidates": 1 if after_validation else 2},
    )
    result = asyncio.run(runtime.compose(tmp_path, progress=progress))
    assert len(captured) == 2
    comparison = next(event for event in events if "differences" in event)
    assert len(comparison["differences"]) == 3
    assert "configuration_sha256 differs" in comparison["message"]
    assert "threshold 1e-06" in comparison["message"]
    assert "joint_state.positions['joint1']" not in comparison["message"]
    previous_ref = comparison["previous_robot_context_ref"]
    current_ref = comparison["robot_context_ref"]
    assert read_pin(tmp_path, previous_ref)["joint_state"]["positions"] == [0.0]
    assert read_pin(tmp_path, current_ref)["joint_state"]["positions"] == [0.003]
    report = read_pin(tmp_path, result["validation_refs"][-1])
    assert report["robot_context_ref"] == previous_ref
    if after_validation:
        assert result["status"] == "budget_exhausted"
        assert report["status"] == "unknown"
        assert report["final_robot_context_ref"] == current_ref
        assert report["findings"][-1]["check"] == "final_freshness"
        assert report["findings"][-1]["message"] == comparison["message"]
    else:
        assert result["status"] == "stale"
        assert result["stop_reason"] == comparison["message"]
        assert current_ref["ref"].endswith("robot_context_0002.json")
    restored = enrich_composition_diagnostic(tmp_path, {}, inputs.context_refs)
    assert restored["status"] == result["status"]
    assert restored["message"] == result["stop_reason"]
    restored_comparison = next(
        event for event in restored["refinement"]["events"] if "differences" in event
    )
    assert {key: restored_comparison[key] for key in comparison} == comparison
    assert all(path.read_bytes() == content for path, content in original.items())
    assert len(result["candidate_refs"]) == (1 if after_validation else 2)
    assert all(read_pin(tmp_path, ref)["status"] == "proposed" for ref in result["candidate_refs"])
    for reference in (previous_ref, current_ref):
        path = tmp_path / reference["ref"]
        content = path.read_bytes()
        path.write_bytes(content + b" ")
        with pytest.raises(ValueError, match="Pinned refinement evidence changed"):
            enrich_composition_diagnostic(tmp_path, {}, inputs.context_refs)
        path.write_bytes(content)


def test_dependency_report_propagates_missing_geometry_without_repair(tmp_path: Path) -> None:
    inputs, robot, refs, _ = _setup(tmp_path)
    steps = _program(refs, bound=False)
    original = deepcopy(steps)
    report = assess_program_dependencies(
        steps,
        inputs.catalog,
        inputs.composition_input["robot_state"],
        read_evidence=lambda ref, path: _evidence_value(inputs, ref, path),
        result_schema=lambda ref: _result_schema(steps, ref, inputs),
    )
    assert any(
        item["step_index"] == 6 and 1 in item.get("blocked_by", []) for item in report["issues"]
    )
    assert any(
        item["authority"] == "PA" and item["parameter_path"] == "/product_geometry/part_height_m"
        for item in report["context_requests"]
    )
    height = next(item for item in report["context_requests"] if item["quantity"] == "/product_geometry/part_height_m")
    assert height["quantity_schema"]["type"] == "number"
    assert height["quantity_schema"]["x-binding-role"] == "vertical_part_height"
    assert steps == original


@pytest.mark.parametrize(
    "lift,insert,expected",
    [(True, True, "passed"), (False, True, "failed"), (True, False, "failed")],
)
def test_ordered_validation_checks_carried_part_and_final_seating(
    tmp_path: Path, lift: bool, insert: bool, expected: str
) -> None:
    inputs, robot, refs, roles = _setup(tmp_path)
    steps = _program(refs, lift=lift, insert=insert)
    original = deepcopy(steps)
    session = _PlanningSession()
    report = asyncio.run(
        validate_program(
            inputs=inputs,
            steps=steps,
            robot=robot,
            evidence=roles,
            directory=tmp_path / "calculations",
            profile={**_legacy_profile(), "validation_scope": VALIDATION_SCOPE},
            cache={},
            session_factory=lambda *args: session,
        )
    )
    assert report["status"] == expected, report["findings"]
    assert report["motion_executed"] is False
    assert steps == original
    assert any(call["attached"] for call in session.calls)
    assert report["calculation_refs"]
    if expected == "passed":
        assert report["predicted_final_state"]["held_part"] is None
        assert len(session.custody) == 2
        assert session.calls[-1]["attached"] is None
    else:
        assert not any("repair_steps" in finding for finding in report["findings"])


@pytest.mark.parametrize("cad_origin_offset", [(0.0, 0.0, 0.0), (0.213, 0.182, 0.070)])
def test_selected_grasp_reference_keeps_validation_invariant_to_cad_origin(
    tmp_path: Path, cad_origin_offset: tuple[float, float, float],
) -> None:
    """Moving the CAD coordinate origin must not move the selected physical grasp or placement."""
    inputs, robot, refs, roles = _setup(tmp_path, cad_origin_offset=cad_origin_offset)
    steps = _program(refs)
    original = deepcopy(steps)
    session = _PlanningSession()
    report = asyncio.run(validate_program(
        inputs=inputs, steps=steps, robot=robot, evidence=roles,
        directory=tmp_path / "calculations", profile=_legacy_profile(), cache={},
        session_factory=lambda *args: session,
    ))
    assert report["status"] == "passed", report["findings"]
    pick, place = [read_pin(tmp_path, ref)["result"] for ref in report["calculation_refs"]]
    assert pick["target_pose"] == pytest.approx({"x": 0.0, "y": 0.0, "z": 0.155})
    assert place["insert_pose"]["x"] == pytest.approx(0.2)
    assert place["insert_pose"]["y"] == pytest.approx(0.0)
    assert place["insert_pose"]["z"] == pytest.approx(0.155)
    assert place["place_part_origin_z"] == pytest.approx(0.05 - cad_origin_offset[2])
    assert place["grasp_tcp_to_part_origin_z"] == pytest.approx(0.005 + cad_origin_offset[2])
    assert any(call["attached"] for call in session.calls)
    assert steps == original and report["motion_executed"] is False


@pytest.mark.parametrize("reference_kind", ["literal", "field_ref", "object_ref", "geometry_ref"])
def test_selected_grasp_reference_routes_missing_offset_to_pa(
    tmp_path: Path, reference_kind: str,
) -> None:
    """The declared selected reference controls whether an offset measurement is required."""
    inputs, robot, refs, roles = _setup(tmp_path, cad_origin_offset=(0.213, 0.182, 0.070))
    steps = _program(refs)
    placement = next(step for step in steps if step["primitive_symbol"] == "compute_place_targets")
    geometry = {
        name: _value(refs["goal"], "/product_geometry/" + name)
        for name in read_pin(tmp_path, refs["goal"])["product_geometry"]
        if name != "grasp_point_offset_world_m"
    }
    if reference_kind in {"literal", "field_ref"}:
        geometry["target_reference"] = {
            "target_point": "inserted_part_origin", "surface_role": "assembly_slot",
            "grasp_point": "selected_CAD_feature" if reference_kind == "literal"
            else _value(refs["goal"], "/product_geometry/target_reference/grasp_point"),
        }
    if reference_kind == "geometry_ref":
        record = read_pin(tmp_path, refs["goal"])
        del record["product_geometry"]["grasp_point_offset_world_m"]
        reference = append_record(tmp_path, tmp_path / "partial", "goal.json", record)
        inputs = replace(inputs, record_hashes={**inputs.record_hashes, reference["ref"]: reference["sha256"]})
        placement["params"]["product_geometry"] = _value(reference, "/product_geometry")
    else:
        placement["params"]["product_geometry"] = geometry
    before = deepcopy(steps)
    dependencies = assess_program_dependencies(
        steps, inputs.catalog, inputs.composition_input["robot_state"],
        read_evidence=lambda ref, path: _evidence_value(inputs, ref, path),
        result_schema=lambda ref: _result_schema(steps, ref, inputs),
    )
    missing = [need for need in dependencies["context_requests"]
               if need["quantity"] == "/product_geometry/grasp_point_offset_world_m"]
    assert len(missing) == 1 and missing[0]["authority"] == "PA"
    report = asyncio.run(validate_program(
        inputs=inputs, steps=steps, robot={**robot, "captured_at_ns": time.time_ns()}, evidence=roles,
        directory=tmp_path / "calculations", profile=_legacy_profile(), cache={},
        session_factory=_PlanningSession,
    ))
    assert report["status"] != "passed"
    assert any("grasp_point_offset_world_m" in item["message"] for item in report["findings"])
    assert steps == before


def test_pa_selected_grasp_point_preserves_cad_pose_and_measures_placement_offset(tmp_path: Path) -> None:
    """A selected feature supplies the grasp offset without moving the collision mesh's CAD frame."""
    inputs, robot, refs, _ = _setup(tmp_path, cad_origin_offset=(0.213, 0.182, 0.070))
    part = read_pin(tmp_path, refs["part"])
    point = part.pop("grasp_reference")["point_CAD_m"]
    part["features"] = [{
        "plane_id": "plane_0001", "normal": [0.0, 0.0, -1.0], "point_m": point,
        "world_normal": [0.0, 0.0, -1.0], "world_point_m": [0.0, 0.0, 0.05],
        "circles": [{"circle_id": "circle_0001", "center_m": point, "radius_m": 0.01}],
    }]
    target = deepcopy(part)
    target["origin_pose"] = {**_pose(x=0.2, z=0.05)}
    target["features"] = [{
        "plane_id": "plane_0002", "normal": [0.0, 0.0, 1.0], "point_m": [0.0, 0.0, 0.0],
        "world_normal": [0.0, 0.0, 1.0], "world_point_m": [0.2, 0.0, 0.05],
        "circles": [{"circle_id": "circle_0002", "center_m": [0.0, 0.0, 0.0],
                     "world_center_m": [0.2, 0.0, 0.05], "radius_m": 0.009}],
    }]
    surface = {
        "record_type": "AssemblySurfaceEvidence", "status": "accepted", "frame_id": "world",
        "units": "m", "normal": [0.0, 0.0, 1.0], "offset_m": -0.04,
        "rms_distance_m": 0.0001, "observation_timestamp_ns": 1_000_000_000,
    }
    sources = [append_record(tmp_path, tmp_path / "selected_geometry", name + ".json", value)
               for name, value in (("part", part), ("target", target), ("surface", surface))]
    producer = AssemblyGeometryProducer(tmp_path, tmp_path / "measurements", {
        item["ref"]: item["sha256"] for item in sources
    })
    with pytest.raises(ValueError, match="outside the measured part bounds"):
        producer.pick_geometry(sources[0]["ref"], sources[2]["ref"])
    runtime = ProductPrimitiveContextRuntime(SimpleNamespace(), object())
    selected = runtime._geometry(producer, "select_grasp_point", {
        "part_ref": sources[0]["ref"], "plane_id": "plane_0001", "circle_id": "circle_0001",
    })
    assert selected["record"]["origin_pose"] == part["origin_pose"]
    assert selected["record"]["mesh"] == part["mesh"]
    assert selected["record"]["grasp_reference"]["point_world_m"] == pytest.approx([0.0, 0.0, 0.05])
    pick = producer.pick_geometry(selected["record_ref"], sources[2]["ref"])["record"]
    goal = producer.assembly_geometry(
        selected["record_ref"], "plane_0001", "circle_0001",
        sources[1]["ref"], "plane_0002", "circle_0002",
    )["record"]
    assert goal["product_geometry"]["grasp_point_offset_world_m"] == pytest.approx(point)
    pick_targets = calculate_target("compute_pick_targets", {
        "part_name": "medium gear", "target_pose": pick["target_pose"],
        "product_geometry": pick["product_geometry"],
    }, robot, robot["ee_pose"])
    grasp_pose = {**robot["ee_pose"], **pick_targets["target_pose"]}
    assert _grasp_check(selected["record"], part["origin_pose"], grasp_pose, robot)
    assert not _grasp_check(part, part["origin_pose"], grasp_pose, robot)
    params = {"part_name": "medium gear", "pick_ctx": pick_targets, "product_geometry": goal["product_geometry"]}
    placed = calculate_target("compute_place_targets", params, robot, grasp_pose)
    assert placed["insert_pose"]["x"] == pytest.approx(0.2)
    assert placed["insert_pose"]["y"] == pytest.approx(0.0)
    assert placed["place_part_origin_z"] == pytest.approx(-0.02)
    assert placed["place_tcp_z"] == pytest.approx(0.055)
    del params["product_geometry"]["grasp_point_offset_world_m"]
    with pytest.raises(CalculationUnavailable, match="grasp_point_offset_world_m"):
        calculate_target("compute_place_targets", params, robot, grasp_pose)


@pytest.mark.parametrize("missing", ["scene", "specification", "robot"])
def test_missing_validation_evidence_never_passes(tmp_path: Path, missing: str) -> None:
    inputs, robot, refs, roles = _setup(tmp_path)
    if missing == "robot":
        robot = None
    else:
        roles.pop(missing)
    report = asyncio.run(
        validate_program(
            inputs=inputs,
            steps=_program(refs),
            robot=robot,
            evidence=roles,
            directory=tmp_path / "calculations",
            profile={**_legacy_profile(), "validation_scope": VALIDATION_SCOPE},
            cache={},
            session_factory=_PlanningSession,
        )
    )
    assert report["status"] == "unknown"


@pytest.mark.parametrize("delayed_check", ["_validate_steps", "verify_evidence_tree"])
def test_robot_freshness_is_checked_at_validation_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, delayed_check: str
) -> None:
    """Evidence checking cannot expire a snapshot that was fresh when received."""
    from cais_spade_llm.spec2primitives.agents.ra import program_validation

    inputs, robot, refs, roles = _setup(tmp_path)
    clock = [10_000_000_000]
    robot["captured_at_ns"] = clock[0] - 1_750_000_000
    original_robot = deepcopy(robot)
    monkeypatch.setattr(program_validation, "time", SimpleNamespace(time_ns=lambda: clock[0]))
    check = getattr(program_validation, delayed_check)

    def delayed(*args: Any) -> Any:
        result = check(*args)
        clock[0] += 2_000_000_000
        return result

    monkeypatch.setattr(program_validation, delayed_check, delayed)
    report = asyncio.run(
        validate_program(
            inputs=inputs, steps=_program(refs), robot=robot, evidence=roles,
            directory=tmp_path / "calculations", profile=_legacy_profile(), cache={},
            session_factory=_PlanningSession,
        )
    )
    assert report["created_at_ns"] - robot["captured_at_ns"] > 2_000_000_000
    assert report["status"] == "passed", report["findings"]
    assert report["motion_executed"] is False
    assert robot == original_robot


@pytest.mark.parametrize(
    "stamp,now_ros,expected",
    [
        (11_299_000_000, 131_299_000_000, None),
        (11_299_000_000, 148_200_000_000, "136.901 ROS seconds old (limit 120 seconds"),
        (148_300_000_000, 148_200_000_000, "observation and current simulation clock are inconsistent"),
        (0, 148_200_000_000, "no valid observation timestamp"),
        (None, 148_200_000_000, "no valid observation timestamp"),
        (True, 148_200_000_000, "no valid observation timestamp"),
    ],
)
def test_observation_freshness_uses_sensor_time_and_explains_the_prerequisite(
    tmp_path: Path, stamp: Any, now_ros: int, expected: str | None,
) -> None:
    """A newly written geometry record cannot refresh an expired RGB-D observation."""
    inputs, robot, refs, roles = _observed_setup(tmp_path)
    for role, observation_stamp in (("part", stamp), ("scene", now_ros)):
        record = read_pin(tmp_path, roles[role])
        record["observation_timestamp_ns"] = observation_stamp
        record["created_at_ns"] = time.time_ns()
        roles[role] = append_record(tmp_path, tmp_path / "timestamps", f"{role}.json", record)
    robot.update(measured_at_ros_ns=now_ros, tf_stamps_ns=[now_ros, now_ros])
    robot["joint_state"]["stamp_ns"] = now_ros
    robot["captured_at_ns"] = time.time_ns()
    original_roles = deepcopy(roles)
    report = asyncio.run(validate_program(
        inputs=inputs, steps=_observed_program(refs), robot=robot, evidence=roles,
        directory=tmp_path / "validation", profile=load_refinement_profile(), cache={},
        session_factory=_PlanningSession,
    ))
    assert roles == original_roles
    findings = [finding for finding in report["findings"] if finding["check"] == "scene_freshness"]
    if expected is None:
        assert not findings
        assert report["status"] == "passed", report["findings"]
    else:
        assert report["status"] != "passed"
        assert len(findings) == 1
        assert expected in findings[0]["message"]
        assert findings[0]["authority"] == "PA"
        if "136.901" in expected:
            assert "Recomputing geometry from the same observation does not refresh" in findings[0]["message"]


@pytest.mark.parametrize("already_stale", [False, True])
def test_validation_progress_finishes_before_robot_capture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, already_stale: bool
) -> None:
    """Slow persisted/UI progress must precede capture without accepting old snapshots."""
    from cais_spade_llm.spec2primitives.agents.ra import program_validation
    from cais_spade_llm.spec2primitives.agents.ra import refinement

    _, robot, _, _ = _setup(tmp_path)
    clock = [10_000_000_000]
    monkeypatch.setattr(program_validation, "time", SimpleNamespace(time_ns=lambda: clock[0]))
    monkeypatch.setattr(refinement, "time", SimpleNamespace(time_ns=lambda: clock[0], monotonic=time.monotonic))
    program = _ProgramRuntime([_program_action([("grasp_part", {"part_name": "medium gear"})])])
    order = []
    capture_times = []
    original_append = refinement.append_record

    def delayed_persistence(*args: Any, **kwargs: Any) -> Any:
        if args[-1].get("record_type") == "RobotValidationContext":
            order.append("persist")
            clock[0] += 3_000_000_000
        return original_append(*args, **kwargs)

    monkeypatch.setattr(refinement, "append_record", delayed_persistence)

    class Robot:
        @asynccontextmanager
        async def validation_context(self, *args: Any, **kwargs: Any) -> Any:
            order.append("capture")
            context = deepcopy(robot)
            context["captured_at_ns"] = clock[0] - (2_500_000_000 if already_stale else 0)
            capture_times.append(context["captured_at_ns"])
            try:
                yield context
            finally:
                order.append("cleanup")
                clock[0] += 3_000_000_000

    async def progress(event: Any) -> None:
        if event["stage"] == "validating":
            order.append("progress")
            clock[0] += 3_000_000_000

    async def validator(**kwargs: Any) -> dict[str, Any]:
        order.append("validation")
        return await validate_program(**kwargs, session_factory=_PlanningSession)

    runtime = PrimitiveRefinementRuntime(
        program_runtime=program, robot_runtime=Robot(), validator=validator,
        profile={**_legacy_profile(), "max_candidates": 1},
    )
    result = asyncio.run(runtime.compose(tmp_path, progress=progress))
    report = read_pin(tmp_path, result["validation_refs"][0])
    assert order == ["progress", "capture", "persist", "validation", "cleanup"]
    assert any(item["check"] == "robot_freshness" for item in report["findings"]) is already_stale
    assert read_pin(tmp_path, report["robot_context_ref"])["captured_at_ns"] == capture_times[0]
    assert result["motion_executed"] is False


@pytest.mark.parametrize(
    "fault,expected",
    [
        ("wall_age", "2.500 seconds old at validation entry"),
        ("wall_future", "wall clocks are inconsistent"),
        ("missing_capture", "missing or invalid"),
        ("missing_joint", "missing or invalid"),
        ("missing_tf", "missing or invalid"),
        ("missing_ros_clock", "missing or invalid"),
        ("invalid_timestamp", "missing or invalid"),
        ("ros_future", "later than the recorded ROS capture clock"),
        ("ros_age", "samples were stale at capture"),
        ("skew", "capture skew"),
    ],
)
def test_robot_freshness_rejects_invalid_snapshots_with_specific_reasons(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str, expected: str
) -> None:
    """Reject invalid snapshots before planning and identify the failed time constraint."""
    from cais_spade_llm.spec2primitives.agents.ra import program_validation

    inputs, robot, refs, roles = _setup(tmp_path)
    started = 10_000_000_000
    robot["captured_at_ns"] = started
    profile = _legacy_profile()
    if fault == "wall_age":
        robot["captured_at_ns"] -= 2_500_000_000
    elif fault == "wall_future":
        robot["captured_at_ns"] += 1
    elif fault == "missing_capture":
        robot.pop("captured_at_ns")
    elif fault == "missing_joint":
        robot["joint_state"].pop("stamp_ns")
    elif fault == "missing_tf":
        robot["tf_stamps_ns"].pop()
    elif fault == "missing_ros_clock":
        robot.pop("measured_at_ros_ns")
    elif fault == "invalid_timestamp":
        robot["tf_stamps_ns"][0] = True
    elif fault == "ros_future":
        robot["joint_state"]["stamp_ns"] += 1
    elif fault == "ros_age":
        robot["measured_at_ros_ns"] += 3_000_000_000
    else:
        profile["max_capture_skew_sec"] = 0.01
        robot["joint_state"]["stamp_ns"] -= 100_000_000
    monkeypatch.setattr(program_validation, "time", SimpleNamespace(time_ns=lambda: started))
    session = _PlanningSession()
    report = asyncio.run(
        validate_program(
            inputs=inputs, steps=_program(refs), robot=robot, evidence=roles,
            directory=tmp_path / "calculations", profile=profile, cache={},
            session_factory=lambda *args: session,
        )
    )
    assert report["status"] == "unknown"
    finding = next(item for item in report["findings"] if item["check"] == "robot_freshness")
    assert expected in finding["message"]
    assert not report["calculation_refs"] and not session.calls and not session.custody


@pytest.mark.parametrize("scope", [None, VALIDATION_SCOPE, GAZEBO_PICK_PLACE_SCOPE])
def test_pick_place_scope_does_not_require_assembly_specification(tmp_path: Path, scope: str | None) -> None:
    """The same grounded program needs assembly evidence only in its historical scope."""
    inputs, robot, refs, roles = _setup(tmp_path)
    profile = _legacy_profile()
    profile.pop("validation_scope")
    if scope is not None:
        profile["validation_scope"] = scope
    steps = _program(refs)
    before = deepcopy(steps)
    report = asyncio.run(validate_program(
        inputs=inputs, steps=steps, robot=robot,
        evidence={role: roles[role] for role in ("part", "scene")},
        directory=tmp_path / "calculations", profile=profile, cache={}, session_factory=_PlanningSession,
    ))
    assert report["scope"] == (scope or VALIDATION_SCOPE)
    assert steps == before and report["motion_executed"] is False
    if scope == GAZEBO_PICK_PLACE_SCOPE:
        assert report["status"] == "passed", report["findings"]
        assert report["predicted_final_state"]["held_part"] is None
        assert not any(item["check"] in {"goal", "specification", "assembly_outcome"} for item in report["findings"])
    else:
        assert report["status"] == "unknown"
        assert {item["check"] for item in report["findings"]} >= {"goal", "specification"}


@pytest.mark.parametrize("fault", ["missing_input", "held_part", "ambiguous_part", "scene", "robot_age", "changed_mesh"])
def test_pick_place_scope_keeps_input_geometry_and_custody_checks(tmp_path: Path, fault: str) -> None:
    """Simulation scope never substitutes missing geometry or permits unknown custody."""
    inputs, robot, refs, roles = _setup(tmp_path)
    roles = {role: roles[role] for role in ("part", "scene")}
    steps = _program(refs)
    if fault == "missing_input":
        del steps[5]["params"]["product_geometry"]
    elif fault == "held_part":
        steps = [step for step in steps if step["primitive_symbol"] != "release_part"]
    elif fault == "ambiguous_part":
        part = read_pin(tmp_path, roles["part"])
        part["status"] = "ambiguous"
        roles["part"] = append_record(tmp_path, tmp_path / "ambiguous", "part.json", part)
    elif fault == "scene":
        roles.pop("scene")
    elif fault == "robot_age":
        robot["captured_at_ns"] -= 5_000_000_000
    else:
        (tmp_path / "mesh.npz").write_bytes(b"changed geometry")
    before = deepcopy(steps)

    async def check() -> dict[str, Any]:
        return await validate_program(
            inputs=inputs, steps=steps, robot=robot, evidence=roles,
            directory=tmp_path / "calculations", profile=_legacy_profile(),
            cache={}, session_factory=_PlanningSession,
        )

    if fault == "changed_mesh":
        with pytest.raises(ValueError, match="dependency changed"):
            asyncio.run(check())
    else:
        report = asyncio.run(check())
        assert report["scope"] == GAZEBO_PICK_PLACE_SCOPE
        assert report["status"] != "passed", report["findings"]
    assert steps == before


@pytest.mark.parametrize("scope", [None, "unknown", [], 1])
def test_unknown_validation_scope_is_rejected(scope: Any) -> None:
    assert read_validation_scope({}) == VALIDATION_SCOPE
    assert load_refinement_profile()["validation_scope"] == GAZEBO_OBSERVED_SCOPE
    with pytest.raises(ValueError, match="Unknown validation scope"):
        read_validation_scope({"validation_scope": scope})


def test_target_calculation_rejects_defaults_tokens_and_tool_mismatch(tmp_path: Path) -> None:
    inputs, robot, refs, roles = _setup(tmp_path)
    params = {
        "part_name": "medium gear",
        "target_pose": {"x": 0.0, "y": 0.0, "z": 0.05},
        "product_geometry": {"board_center": {"z": 0.04}, "part_height_m": 0.02},
    }
    output = calculate_target("compute_pick_targets", params, robot, _pose())
    assert "model_name" not in output
    assert output["target_pose"]["z"] == pytest.approx(0.155)
    missing = deepcopy(params)
    missing["product_geometry"].pop("part_height_m")
    with pytest.raises(CalculationUnavailable):
        calculate_target("compute_pick_targets", missing, robot, _pose())
    with pytest.raises(CalculationUnavailable, match="Live detection"):
        calculate_target(
            "compute_pick_targets", {**params, "prefer_live_detection": True}, robot, _pose()
        )
    shifted = deepcopy(robot)
    shifted["ee_from_tcp"][0][3] = 0.02
    shifted_output = calculate_target("compute_pick_targets", params, shifted, _pose())
    assert shifted_output["target_pose"]["x"] == pytest.approx(-0.02)
    assert shifted_output["tx"] == output["tx"]
    invalid_tool = deepcopy(robot)
    invalid_tool["ee_from_tcp"] = [[1.0]]
    with pytest.raises(CalculationUnavailable, match="full EE"):
        calculate_target("compute_pick_targets", params, invalid_tool, _pose())
    place = {
        "part_name": "medium gear",
        "pick_ctx": output,
        "product_geometry": read_pin(tmp_path, refs["goal"])["product_geometry"],
        "destination_location": "middle gear shaft destination reference",
    }
    with pytest.raises(CalculationUnavailable, match="token"):
        calculate_target("compute_place_targets", place, robot, _pose())


@pytest.mark.parametrize("tool_translation", [[0.0, 0.0, 0.172], [0.02, -0.03, 0.172]])
def test_target_calculation_compensates_measured_tool_offset(tool_translation: list[float]) -> None:
    """An offset or slightly tilted tool still places its TCP at the selected target."""
    from scipy.spatial.transform import Rotation
    from cais_spade_llm.spec2primitives.adapters.robot_validation_context import pose_matrix

    robot = _robot(SimpleNamespace(assignment=SimpleNamespace(selected_resource_jid="fixture", fingerprint="fixture")))
    robot["ee_from_tcp"] = np.eye(4).tolist()
    for axis, value in enumerate(tool_translation):
        robot["ee_from_tcp"][axis][3] = value
    rotation = Rotation.from_euler("xyz", [0.0004, np.pi - 0.0003, 0.0002])
    pose = {**_pose(), **dict(zip(("qx", "qy", "qz", "qw"), rotation.as_quat(), strict=True))}
    pick = calculate_target("compute_pick_targets", {
        "part_name": "medium gear", "target_pose": {"x": 0.1, "y": -0.2, "z": 0.05},
        "product_geometry": {"board_center": {"z": 0.04}, "part_height_m": 0.02},
    }, robot, pose)
    tool = np.asarray(robot["ee_from_tcp"])
    tcp = (pose_matrix({**pose, **pick["target_pose"]}) @ tool)[:3, 3]
    assert tcp == pytest.approx([0.1, -0.2, pick["pick_tcp_z"]], abs=1e-12)
    assert pick["tx"] == 0.1 and pick["ty"] == -0.2
    place = calculate_target("compute_place_targets", {
        "part_name": "medium gear", "pick_ctx": pick,
        "product_geometry": {
            "board_center": {"x": 0.2, "y": 0.3}, "slot_xy": [0.0, 0.0],
            "slot_floor_z_m": 0.04, "part_height_m": 0.02,
            "target_reference": {"target_point": "inserted_part_origin", "surface_role": "assembly_slot"},
            "target_origin_pose": {"x": 0.2, "y": 0.3, "z": 0.05},
        },
    }, robot, pose)
    for name in ("approach_pose", "target_pose", "pre_insert_pose", "insert_pose"):
        target = place[name]
        target_tcp = (pose_matrix(target) @ tool)[:3, 3]
        assert target_tcp[:2] == pytest.approx([0.2, 0.3], abs=1e-12)
        assert pose_matrix(target)[:3, :3] == pytest.approx(rotation.as_matrix(), abs=1e-12)
    assert (pose_matrix(place["insert_pose"]) @ tool)[2, 3] == pytest.approx(place["place_tcp_z"], abs=1e-12)
    assert place["slot_x"] == 0.2 and place["slot_y"] == 0.3


def test_extracted_calculations_match_runtime_helpers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from cais_spade_llm.resources.robot import gazebo_pick_place_controller as controller_module
    from cais_spade_llm.resources.robot.robot_primitives import ROBOT_EXTRACT_OUTPUT_MAP
    from cais_spade_llm.spec2primitives.agents.ra.composition_context import _without_model_name

    inputs, robot, refs, _ = _setup(tmp_path)
    controller = object.__new__(controller_module.GazeboPickPlaceController)
    controller.execution_mode = "simulation"
    controller.robot_name = "xarm6"
    controller.wait_for_services = lambda: True
    controller._get_ee_pose = lambda: SimpleNamespace(
        position=SimpleNamespace(x=0.0, y=0.0, z=0.3),
        orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
    )
    controller._get_ee_tcp_world_z_offset = lambda: -0.1
    controller._derive_gripper_close_position = lambda **kwargs: None
    controller._make_pose = lambda *args: args
    controller._log = lambda: logging.getLogger(__name__)
    for name, value in robot["policy"].items():
        setattr(controller, name, value)
    controller.controller_config = {}
    monkeypatch.setattr(
        controller_module,
        "resolve_place_geometry",
        lambda **kwargs: deepcopy(kwargs["product_geometry"]),
    )
    pick = {
        "part_name": "medium gear",
        "target_pose": read_pin(tmp_path, refs["pick"])["target_pose"],
        "product_geometry": read_pin(tmp_path, refs["pick"])["product_geometry"],
    }
    runtime_pick = controller.compute_pick_targets(**pick)
    expected, error = ROBOT_EXTRACT_OUTPUT_MAP["compute_pick_targets"](pick, runtime_pick)
    assert error is None
    actual = calculate_target("compute_pick_targets", pick, robot, _pose())
    assert actual == _without_model_name(expected)
    place = {
        "part_name": "medium gear",
        "pick_ctx": actual,
        "product_geometry": read_pin(tmp_path, refs["goal"])["product_geometry"],
    }
    runtime_place = controller.compute_place_targets(**place)
    expected, error = ROBOT_EXTRACT_OUTPUT_MAP["compute_place_targets"](place, runtime_place)
    assert error is None
    assert calculate_target("compute_place_targets", place, robot, _pose()) == _without_model_name(
        expected
    )


def test_selected_conditional_output_cannot_be_fabricated() -> None:
    with pytest.raises(ValueError, match="conditional"):
        resolve_selected_values(
            _result(1, "/insert_pose/z"),
            read_evidence=lambda *args: {},
            results={1: {"target_pose": {"z": 0.1}}},
        )


@pytest.mark.parametrize("world_axis", ["z", "translated_yaw", "x"])
def test_ambiguous_height_uses_ranked_fit_in_world_frame(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, world_axis: str
) -> None:
    """Retain a height estimate despite differing fits; project along world vertical."""
    transform = np.eye(4)
    if world_axis == "translated_yaw":
        transform[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
        transform[:3, 3] = [0.4, 0.1, 0.3]
    elif world_axis == "x":
        transform[:3, :3] = [[0, 0, 1], [0, 1, 0], [-1, 0, 0]]
    runtime, producer, correspondence_ref, cad_ref = _pa_geometry(
        tmp_path, monkeypatch, ambiguous=True, tilt=0.3, world_transform=transform
    )
    pose = runtime._geometry(producer, "estimate_pose", {"correspondence_ref": correspondence_ref})
    converted = runtime._geometry(producer, "convert_pose", {"pose_ref": pose["record_ref"]})
    result = producer.inspect_features(cad_ref, converted["record_ref"])
    record = result["record"]
    size = producer.read(cad_ref)["bounds_m"]["size"]
    assert record["part_height_m"] == pytest.approx(size[0 if world_axis == "x" else 2])
    assert record["status"] == record["pose_status"] == "ambiguous"
    assert record["uncertainty"]["hypothesis_index"] == 0
    assert record["uncertainty"]["source_pose"] == pin(tmp_path, tmp_path / pose["record_ref"])
    assert record["uncertainty"]["complete_pose_established"] is False
    heights = record["uncertainty"]["qualified_height_estimates_m"]
    assert len(heights) == 2
    assert heights[0] == record["part_height_m"]
    if world_axis == "x":
        assert heights == pytest.approx([size[0], size[0]])
    else:
        assert heights[1] > heights[0] + 0.001
    assert record["uncertainty"]["qualified_height_range_m"] == [min(heights), max(heights)]
    assert "highest-ranked" in record["warning"]
    assert not {"origin_pose", "features", "world_from_CAD"} & record.keys()
    assert "numerical_agreement_m" not in record["uncertainty"]
    assert "invariant_evidence" not in result


def test_planar_features_measure_circular_boundaries_without_assigning_roles() -> None:
    angles = np.linspace(0, 2 * np.pi, 33)[:-1]
    inner = np.column_stack((0.01 * np.cos(angles), 0.01 * np.sin(angles), np.zeros(32)))
    outer = inner * 2
    triangles = np.asarray(
        [
            triangle
            for index in range(32)
            for triangle in (
                [inner[index], outer[index], outer[(index + 1) % 32]],
                [inner[index], outer[(index + 1) % 32], inner[(index + 1) % 32]],
            )
        ]
    )
    features = planar_features(triangles)
    assert len(features) == 1
    assert sorted(circle["radius_m"] for circle in features[0]["circles"]) == pytest.approx(
        [0.01, 0.02], abs=1e-7
    )
    assert "assembly_slot" not in json.dumps(features)


def test_surface_height_measures_selected_point_without_accepting_part_pose(tmp_path: Path) -> None:
    """Derive height from a plane and location, never substitute a plane coefficient or full pose."""
    inputs, _, refs, roles = _setup(tmp_path)
    normal = np.asarray([0.01, -0.02, -1.0])
    normal /= np.linalg.norm(normal)
    surface = append_record(tmp_path, tmp_path / "products/test_evidence", "surface.json", {
        "record_type": "AssemblySurfaceEvidence", "status": "accepted",
        "frame_id": "world", "units": "m", "normal": normal.tolist(),
        "offset_m": 1.05, "rms_distance_m": 0.0001, "observation_timestamp_ns": 1_000_000_000,
    })
    location = append_record(tmp_path, tmp_path / "products/test_evidence", "location.json", {
        "record_type": "RobotFrameLocationRecord", "location": "available",
        "robot_frame_conversion": "accepted", "target_frame": "world",
        "translated_location_m": [0.4, 0.3, 1.08], "observation_timestamp_ns": 1_000_000_000,
    })
    producer = AssemblyGeometryProducer(tmp_path, tmp_path / "measurements", {
        item["ref"]: item["sha256"] for item in (surface, location)
    })
    result = producer.surface_height(surface["ref"], location["ref"])
    record = result["record"]
    height = record["product_geometry"]["board_center"]["z"]
    assert float(normal @ [0.4, 0.3, height]) + 1.05 == pytest.approx(0.0, abs=1e-12)
    assert height != pytest.approx(1.05, abs=1e-4)
    assert record["record_type"] == "AssemblySurfaceEvidence" and record["status"] == "accepted"
    assert not {"origin_pose", "world_from_CAD", "part_height_m", "features"} & record.keys()
    derived = pin(tmp_path, tmp_path / result["record_ref"])
    inputs = replace(inputs, record_hashes={**inputs.record_hashes, **producer.authorized})
    params = {"product_geometry": {"board_center": {"z": _value(surface, "/offset_m")}}}
    with pytest.raises(BindingUnavailable, match="plane coefficient"):
        _geometry_sources({"primitive_symbol": "compute_pick_targets", "params": params}, inputs)
    params["product_geometry"]["board_center"]["z"] = _value(derived, "/product_geometry/board_center/z")
    assert _geometry_sources({"primitive_symbol": "compute_pick_targets", "params": params}, inputs) == []
    report = asyncio.run(validate_program(
        inputs=inputs, steps=_program(refs), robot=_robot(inputs), evidence={**roles, "part": derived},
        directory=tmp_path / "calculations", profile=_legacy_profile(), cache={},
        session_factory=_PlanningSession,
    ))
    assert report["status"] != "passed"
    assert any(item["check"] == "part" and item["status"] == "unknown" for item in report["findings"])
    path = tmp_path / location["ref"]
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="changed"):
        producer.surface_height(surface["ref"], location["ref"])


def _pa_geometry(
    root: Path, monkeypatch: pytest.MonkeyPatch, *, ambiguous: bool = False,
    tilt: float = 0.0, world_transform: np.ndarray | None = None,
    diameters_m: tuple[float, ...] = (0.042,),
) -> tuple[ProductPrimitiveContextRuntime, AssemblyGeometryProducer, str, str]:
    correspondence_path = _prepare_correspondence(root, diameters_m)
    correspondence_ref = pin(root, correspondence_path)
    cad_ref = read_pin(root, correspondence_ref)["CAD"]["record"]
    monkeypatch.setattr(pose_estimation, "_register_candidate", _accepted_registration)
    if ambiguous:

        def register(triangles: Any, *args: Any, **kwargs: Any) -> list[Any]:
            first = _accepted_registration(triangles, *args, **kwargs)[0]
            rotation = np.diag([-1.0, -1.0, 1.0])
            rotation = rotation @ np.asarray(
                [[1, 0, 0], [0, np.cos(tilt), -np.sin(tilt)], [0, np.sin(tilt), np.cos(tilt)]]
            )
            second = first.transformation.copy()
            second[:3, :3] = rotation
            second[:3, 3] += (np.eye(3) - rotation) @ triangles.mean(axis=(0, 1))
            return [
                _hypothesis(kwargs, fitness=first.fitness, initialization=2, transformation=second),
                first,
            ]

        monkeypatch.setattr(pose_estimation, "_register_candidate", register)

    class Calibration:
        def materialize_camera_to_world_calibration(self, **kwargs: Any) -> Any:
            return record_camera_to_robot_calibration(
                interaction_root=root,
                calibration_id="fixture_calibration",
                source_frame=kwargs["source_frame"],
                target_frame=kwargs["target_frame"],
                target_from_camera_transform=np.eye(4) if world_transform is None else world_transform,
                valid_from_ns=0,
                valid_until_ns=None,
                provenance_source="controlled_fixture",
                provenance_sha256="a" * 64,
                calibration_number=kwargs["calibration_number"],
            )

    runtime = ProductPrimitiveContextRuntime(
        SimpleNamespace(_camera_to_world_calibration_runtime=Calibration()), object()
    )
    producer = AssemblyGeometryProducer(
        root,
        root / "composition/refinement_runs/run_0001/pa_0001/geometry",
        {item["ref"]: item["sha256"] for item in (correspondence_ref, cad_ref)},
    )
    return runtime, producer, correspondence_ref["ref"], cad_ref["ref"]


def test_pa_observed_bounds_measure_support_and_preserve_collision_coverage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Measure selected RGB-D candidates without registration or implicit feature choice."""
    transform = np.diag([1.0, -1.0, -1.0, 1.0])
    transform[2, 3] = 1.0
    runtime, producer, correspondence_ref, cad_ref = _pa_geometry(
        tmp_path, monkeypatch, world_transform=transform, diameters_m=(0.042, 0.03),
    )
    correspondence = producer.read(correspondence_ref)
    segmentation_ref = correspondence["segmentation"]["record"]
    producer.authorized[segmentation_ref["ref"]] = segmentation_ref["sha256"]
    segmentation = producer.read(segmentation_ref["ref"])
    camera = next(item for item in segmentation["cameras"] if len(item["candidates"]) == 2)
    calibration = runtime.runtime._camera_to_world_calibration_runtime.materialize_camera_to_world_calibration(
        source_frame=camera["frame"], target_frame="world", calibration_number=1,
    )
    calibration_ref = pin(tmp_path, calibration.record_path)
    producer.authorized[calibration_ref["ref"]] = calibration_ref["sha256"]
    result = producer.observed_geometry(segmentation_ref["ref"], camera["observation_handle"], calibration_ref["ref"])
    assert len(result["geometry_refs"]) == len(camera["candidates"]) == 2
    part = result["measurements"][0]["record"]
    assert part["part_height_m"] > 0
    assert part["reference_pose"]["z"] == pytest.approx(
        part["product_geometry"]["board_center"]["z"] + part["part_height_m"] / 2,
    )
    assert part["placement_surface_point"]["z"] == part["bounds_m"]["maximum"][2]
    assert part["uncertainty"]["CAD_orientation"] == "not_established"
    producer.target_feature = {
        "assembly_feature_association": [{"assembly_features": [{
            "name": "selected bore", "state_name": "current_state", "state_value_name": "selected part",
            "owner": {"name": "medium gear", "evidence_refs": [correspondence["CAD"]["context_ref"]]},
        }]}],
        "resolved_state_values": [{"state": "current_state", "name": "selected part",
            "value_ref": {"record_ref": segmentation_ref["ref"]},
            "resolved_value": part["candidate_reference"]}],
    }
    bound = producer.bind_observed_part(result["geometry_refs"][0], cad_ref, "selected bore")
    assert bound["record"]["part_name"] == "medium gear"
    assert bound["record"]["reference_pose"] == part["reference_pose"]
    refs = [bound["record_ref"], *result["geometry_refs"][1:]]
    scene = producer.scene_geometry(refs, [result["surface_ref"]], [segmentation_ref["ref"]])
    assert scene["record"]["status"] == "accepted"
    assert scene["record"]["declared_observations"] == [{
        "segmentation_ref": segmentation_ref["ref"], "observation_handle": camera["observation_handle"],
    }]
    assert all("size_m" in item for item in scene["record"]["objects"][:-1])
    assert "mesh" in scene["record"]["objects"][-1]
    incomplete = producer.scene_geometry(refs[:1], [result["surface_ref"]], [segmentation_ref["ref"]])
    assert incomplete["record"]["status"] == "incomplete"
    assert len(incomplete["record"]["unresolved_candidates"]) == 1
    with pytest.raises(ValueError, match="duplicate"):
        producer.scene_geometry([*refs, refs[0]], [result["surface_ref"]], [segmentation_ref["ref"]])
    calibration.record_path.write_bytes(calibration.record_path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="changed"):
        verify_evidence_tree(tmp_path, pin(tmp_path, tmp_path / bound["record_ref"]))


@pytest.mark.parametrize("premature_status", ["not_investigated", "blocked"])
def test_pa_continues_unfinished_input_investigation_with_remaining_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, premature_status: str,
) -> None:
    """PA retains tool choice after premature finish and reports actual prerequisites."""
    _, _, refs, _ = _observed_setup(tmp_path)
    needs = [{"step_index": 1, "quantity": "/product_geometry/part_height_m", "authority": "PA"},
             {"step_index": None, "quantity": "scene", "authority": "PA"}]
    calls = []

    class Product:
        async def ask_llm_structured(self, prompt: str, **kwargs: Any) -> Any:
            state = json.loads(prompt.split("\n\n", 1)[1])
            calls.append(state)
            if len(calls) <= 3:
                return {"action": {"kind": "investigate", "tool_name": "read_record", "arguments": json.dumps({
                    "record_ref": refs["part"]["ref"],
                    "field_path": ("/record_type", "/reference_pose", "/product_geometry")[len(calls) - 1],
                })}}
            if len(calls) == 5:
                assert state["exchanges"][-1]["result"]["status"] == "continue"
                assert state["operations_remaining"] == 3
                assert "3/6 evidence operations used; 3 remain" in state["exchanges"][-1]["result"]["message"]
                return {"action": {"kind": "investigate", "tool_name": "read_record", "arguments": json.dumps({
                    "record_ref": refs["part"]["ref"], "field_path": "/part_height_m",
                })}}
            investigated = len(calls) == 6
            if investigated:
                assert state["exchanges"][-1]["result"]["value"] == 0.02
            return {"action": {
                "kind": "finish", "evidence_refs": [refs["part"]["ref"]] if investigated else [],
                "validation_refs": dict.fromkeys(("part", "goal", "scene", "specification")),
                "outcomes": [{
                    **{key: need[key] for key in ("step_index", "quantity")},
                    "status": ("provided" if investigated else premature_status) if i == 0 else "blocked",
                    "record_ref": refs["part"]["ref"] if i == 0 and investigated else None,
                    "field_path": "/part_height_m" if i == 0 and investigated else None,
                    "reason": (
                        "A calibrated observation of the other obstacle is unavailable."
                        if investigated else "The operation budget was consumed before the derived record was issued."
                    ),
                } for i, need in enumerate(needs)],
            }}

    runtime = ProductPrimitiveContextRuntime(SimpleNamespace(), Product())
    monkeypatch.setattr(runtime, "_investigation", lambda *args: SimpleNamespace(
        handles={}, _canonical_by_pa_ref={}, prior_evidence=(), retrieved_results={},
    ))
    result = asyncio.run(runtime.investigate(
        interaction_root=tmp_path, directory=tmp_path / "composition/refinement_runs/run_0001/pa_0001",
        request={"validation_scope": GAZEBO_OBSERVED_SCOPE, "target_feature": {}, "needs": needs,
                 "evidence_refs": [refs["part"]]}, max_operations=6,
    ))
    assert len(calls) == 6 and result["operations_used"] == 4
    assert result["evidence_refs"] == [refs["part"]]
    assert len(result["unresolved"]) == 1
    assert "blocked" in result["unresolved"][0] and "other obstacle" in result["unresolved"][0]


def test_pa_can_confirm_a_missing_source_without_forced_measurement_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Budget correction cannot prescribe tools or require spending the allowance."""
    _, _, refs, _ = _observed_setup(tmp_path)
    calls = []
    need = {"step_index": None, "quantity": "scene", "authority": "PA"}
    reason = "A calibrated observation of the other obstacle is unavailable."

    class Product:
        async def ask_llm_structured(self, prompt: str, **kwargs: Any) -> Any:
            calls.append(json.loads(prompt.split("\n\n", 1)[1]))
            return {"action": {
                "kind": "finish", "evidence_refs": [],
                "validation_refs": dict.fromkeys(("part", "goal", "scene", "specification")),
                "outcomes": [{"step_index": None, "quantity": "scene", "status": "blocked",
                              "record_ref": None, "field_path": None, "reason": reason}],
            }}

    runtime = ProductPrimitiveContextRuntime(SimpleNamespace(), Product())
    monkeypatch.setattr(runtime, "_investigation", lambda *args: SimpleNamespace(
        handles={}, _canonical_by_pa_ref={}, prior_evidence=(), retrieved_results={},
    ))
    result = asyncio.run(runtime.investigate(
        interaction_root=tmp_path, directory=tmp_path / "composition/refinement_runs/run_0001/pa_0001",
        request={"validation_scope": GAZEBO_OBSERVED_SCOPE, "target_feature": {}, "needs": [need],
                 "evidence_refs": [refs["part"]]}, max_operations=6,
    ))
    assert len(calls) == 2 and result["operations_used"] == 0
    assert calls[1]["operations_remaining"] == 6
    assert result["unresolved"] == [f"step_index None, quantity scene: blocked: {reason}"]


@pytest.mark.parametrize("cancel", [False, True])
def test_owned_capture_keeps_context_alive_until_validation_finishes(
    monkeypatch: pytest.MonkeyPatch, cancel: bool,
) -> None:
    """Release the capture worker after validation or cancellation, never before delivery."""
    from cais_spade_llm.spec2primitives.adapters.robot_validation_context import MeasuredRobotContextRuntime

    order = []
    runtime = MeasuredRobotContextRuntime()

    def capture(*args: Any) -> Any:
        order.append("captured")
        args[-1]({"captured_at_ns": 123})
        order.append("cleanup")

    monkeypatch.setattr(runtime, "_capture", capture)

    async def scenario() -> None:
        entered = asyncio.Event()
        release = asyncio.Event()

        async def validate() -> None:
            async with runtime.validation_context(resource_jid="fixture", assignment_fingerprint="fixture", configuration={}, profile={}) as record:
                assert record["captured_at_ns"] == 123
                order.append("validation")
                entered.set()
                await release.wait()

        task = asyncio.create_task(validate())
        await asyncio.wait_for(entered.wait(), timeout=2)
        assert order == ["captured", "validation"]
        if cancel:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            release.set()
            await task
        assert order == ["captured", "validation", "cleanup"]

    asyncio.run(scenario())


def test_evidence_reuse_is_limited_to_one_synchronous_check(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Shared ancestor bytes are read once per check; a later changed source still fails."""
    from cais_spade_llm.spec2primitives.agents.ra.refinement_records import _verify_evidence_tree

    mesh = tmp_path / "mesh.bin"
    mesh.write_bytes(b"measured geometry")
    reference = pin(tmp_path, mesh)
    parents = [append_record(tmp_path, tmp_path, f"parent_{i}.json", {"source_refs": [reference]}) for i in range(2)]
    reads = []
    read_bytes = Path.read_bytes

    def counted(path: Path) -> bytes:
        if path == mesh:
            reads.append(path)
        return read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", counted)
    checked = {}
    for parent in parents:
        _verify_evidence_tree(tmp_path, parent, checked)
    assert len(reads) == 1
    mesh.write_bytes(b"changed geometry")
    with pytest.raises(ValueError, match="dependency changed"):
        verify_evidence_tree(tmp_path, parents[0])


def test_private_scene_allows_only_selected_gripper_contact(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep SRDF exclusions and collision checks for the arm and other objects."""
    import sys
    from cais_spade_llm.spec2primitives.adapters.isolated_moveit import IsolatedMoveItSession

    matrix = SimpleNamespace(entry_names=["arm", "base"], entry_values=[
        SimpleNamespace(enabled=[False, True]), SimpleNamespace(enabled=[True, False]),
    ])
    scene = SimpleNamespace(is_diff=False, robot_state=SimpleNamespace(is_diff=False),
                            world=SimpleNamespace(collision_objects=[]))
    components = type("Components", (), {"ALLOWED_COLLISION_MATRIX": 128, "__init__": lambda self, **kw: self.__dict__.update(kw)})
    monkeypatch.setitem(sys.modules, "moveit_msgs.msg", SimpleNamespace(
        CollisionObject=SimpleNamespace, PlanningScene=lambda: scene,
        PlanningSceneComponents=components, AllowedCollisionEntry=SimpleNamespace,
    ))
    service = SimpleNamespace(Request=SimpleNamespace)
    monkeypatch.setitem(sys.modules, "moveit_msgs.srv", SimpleNamespace(
        GetPlanningScene=service, ApplyPlanningScene=service,
    ))
    calls = []

    def client(service: Any, name: str) -> Any:
        def call(request: Any) -> Any:
            calls.append(name.rsplit("/", 1)[-1])
            response = (SimpleNamespace(scene=SimpleNamespace(allowed_collision_matrix=matrix))
                        if name.endswith("get_planning_scene") else SimpleNamespace(success=True))
            return SimpleNamespace(done=lambda: True, result=lambda: response)
        return SimpleNamespace(wait_for_service=lambda **kw: True, call_async=call)

    session = IsolatedMoveItSession(tmp_path, {"joint_state": {}, "frame_id": "world"}, {
        "objects": [], "allowed_contacts": [{"object_id": "part", "links": ["finger"]}],
    }, load_refinement_profile())
    session._node = SimpleNamespace(create_client=client)
    monkeypatch.setattr(session, "_state", lambda *args: SimpleNamespace(is_diff=False))
    session._apply_scene()
    assert calls == ["get_planning_scene", "apply_planning_scene"]
    index = {name: i for i, name in enumerate(matrix.entry_names)}
    for left, right, expected in [("arm", "base", True), ("part", "finger", True),
                                  ("part", "arm", False), ("part", "base", False)]:
        assert matrix.entry_values[index[left]].enabled[index[right]] is expected
        assert matrix.entry_values[index[right]].enabled[index[left]] is expected


def _pa_observations(root: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Any, ...]:
    runtime, producer, correspondence_ref, _ = _pa_geometry(root, monkeypatch)
    segmentation_pin = producer.read(correspondence_ref)["segmentation"]["record"]
    producer.authorized[segmentation_pin["ref"]] = segmentation_pin["sha256"]
    segmentation = producer.read(segmentation_pin["ref"])
    for number, camera in enumerate((segmentation["cameras"][2], segmentation["cameras"][0]), 1):
        calibration = runtime.runtime._camera_to_world_calibration_runtime.materialize_camera_to_world_calibration(
            source_frame=camera["frame"], target_frame="world", calibration_number=number,
        )
        reference = pin(root, calibration.record_path)
        producer.authorized[reference["ref"]] = reference["sha256"]
    presentation = ObservationPresentation(root, create=True)
    return runtime, producer, segmentation_pin["ref"], segmentation, presentation


@pytest.mark.parametrize("read_first", [False, True])
@pytest.mark.parametrize("scope", [VALIDATION_SCOPE, GAZEBO_PICK_PLACE_SCOPE])
def test_pa_finds_third_observation_without_camera_index_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, read_first: bool, scope: str,
) -> None:
    """Expose all views and let PA choose read/measurement order using exact handles."""
    runtime, producer, ref, segmentation, presentation = _pa_observations(tmp_path, monkeypatch)
    selected = presentation.handle(segmentation["cameras"][2]["observation_handle"])
    original = {path: path.read_bytes() for path in tmp_path.rglob("*.json")}
    calls, messages = [], []
    order = ["read_observation", "observed_surface"] if read_first else ["observed_surface", "read_observation"]

    class Product:
        async def ask_llm_structured(self, prompt: str, **kwargs: Any) -> Any:
            state = json.loads(prompt.split("\n\n", 1)[1])
            calls.append(state)
            assert state["validation_scope"] == scope
            if scope == GAZEBO_PICK_PLACE_SCOPE:
                assert "Precise seating and assembly tolerances are not acceptance requirements" in prompt
            assert len(state["observation_catalog"]) == 4
            assert "cam_" not in prompt and "/cameras/0" not in prompt
            observation = next(item for item in state["observation_catalog"] if item["observation_handle"] == selected)
            assert observation["support_plane"]["status"] == "detected"
            assert len(observation["compatible_calibration_refs"]) == 1
            calibration_ref = observation["compatible_calibration_refs"][0]
            assert "calibration_0001/" in calibration_ref
            turn = len(state["exchanges"])
            if turn < 2:
                name = order[turn]
                arguments = {"segmentation_ref": ref, "observation_handle": selected}
                arguments.update({"field_path": "/support_plane"} if name == "read_observation" else {"calibration_ref": calibration_ref})
                return {"action": {"kind": "investigate", "tool_name": name, "arguments": json.dumps(arguments)}}
            measurement = state["exchanges"][order.index("observed_surface")]["result"]
            assert measurement["record"]["status"] == "accepted"
            assert measurement["record"]["frame_id"] == "world"
            return {"action": {"kind": "finish", "evidence_refs": [measurement["record_ref"]],
                               "validation_refs": dict.fromkeys(("part", "goal", "scene", "specification")),
                               "unresolved": ["Accepted object orientation remains required."]}}

    runtime.product_agent = Product()
    monkeypatch.setattr(runtime, "_investigation", lambda *args: SimpleNamespace(
        handles={}, _canonical_by_pa_ref={}, prior_evidence=(), retrieved_results={}
    ))

    async def progress(event: dict[str, Any]) -> None:
        messages.append(event["message"])

    result = asyncio.run(runtime.investigate(
        interaction_root=tmp_path, directory=producer.directory.parent,
        request={"validation_scope": scope, "target_feature": {}, "needs": [{"step_index": 1, "quantity": "/product_geometry/board_center/z"}],
                 "evidence_refs": [{"ref": key, "sha256": sha} for key, sha in producer.authorized.items()]},
        max_operations=6, progress=progress,
    ))
    assert len(calls) == 3 and result["operations_used"] == 2
    assert result["validation_refs"] == {}
    assert any("measurement/geometry observed_surface accepted" in message for message in messages)
    assert "2/6 operations used" in messages[-1]
    assert all(path.read_bytes() == content for path, content in original.items())


def test_observation_lookup_checks_sources_calibrations_and_filters_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exact reads retain measurements, reject wrong sources and hide sensor identities."""
    runtime, producer, ref, segmentation, presentation = _pa_observations(tmp_path, monkeypatch)
    records = _issued_records(producer)
    catalog = _pa_evidence_projection(tmp_path, _observation_catalog(tmp_path, records), records)
    reversed_records = deepcopy(records)
    reversed_records[ref]["cameras"].reverse()
    reordered = _pa_evidence_projection(tmp_path, _observation_catalog(tmp_path, reversed_records), reversed_records)
    assert sorted(catalog, key=lambda item: item["observation_handle"]) == sorted(reordered, key=lambda item: item["observation_handle"])
    camera = segmentation["cameras"][2]
    selected = {"segmentation_ref": ref, "observation_handle": camera["observation_handle"]}
    good = next(item for item in catalog if item["observation_handle"] == presentation.handle(camera["observation_handle"]))["compatible_calibration_refs"][0]
    bad = next(key for key in records if "calibration_0002/" in key)
    result = runtime._geometry(producer, "observed_surface", {**selected, "calibration_ref": bad})
    assert result["status"] == "unavailable" and result["compatible_calibration_refs"] == [good]
    other = {**selected, "observation_handle": segmentation["cameras"][1]["observation_handle"]}
    result = runtime._geometry(producer, "observed_surface", {**other, "calibration_ref": bad})
    assert result["compatible_calibration_refs"] == [] and "surface_calibration" in result["reason"]
    assert "has not been established" in result["reason"]
    runtime.runtime._camera_to_world_calibration_runtime = None
    with pytest.raises(ValueError, match="Approved camera calibration is unavailable"):
        runtime._geometry(producer, "surface_calibration", other)
    with pytest.raises(ValueError, match="absent or duplicated"):
        runtime._geometry(producer, "read_observation", {**selected, "observation_handle": "unknown", "field_path": ""})
    with pytest.raises(ValueError, match="not issued"):
        runtime._geometry(producer, "read_observation", {**selected, "segmentation_ref": "execution/private.json", "field_path": ""})
    for pointer in ("/cameras/0", "/cameras/2/frame"):
        with pytest.raises(ValueError, match="camera positions are internal"):
            runtime._geometry(producer, "read_record", {"record_ref": ref, "field_path": pointer})
    for record_ref in (ref, good):
        value = runtime._geometry(producer, "read_record", {"record_ref": record_ref, "field_path": ""})
        assert "cam_" not in json.dumps(value) and "camera_id" not in json.dumps(value)
    read = runtime._geometry(producer, "read_observation", {**selected, "field_path": "/support_plane"})
    assert read["value"] == camera["support_plane"]
    with pytest.raises(ValueError, match="field_path does not exist"):
        runtime._geometry(producer, "read_observation", {**selected, "field_path": "/absent"})
    duplicate = deepcopy(records)
    duplicate[ref]["cameras"].append(deepcopy(camera))
    with pytest.raises(ValueError, match="duplicated"):
        _observation_catalog(tmp_path, duplicate)
    safe = _pa_evidence_projection(tmp_path, {"reason": f"Failed {camera['frame']} /cameras/2", "extracted_text": "Approved document text."}, records)
    assert "cam_" not in json.dumps(safe) and "/cameras/2" not in json.dumps(safe)
    assert safe["extracted_text"] == "Approved document text."
    cad_coordinates = {"coordinate_frame": "CAD_local", "description": "A CAD_local dimension."}
    assert _pa_evidence_projection(tmp_path, cad_coordinates, records) == cad_coordinates
    expired = record_camera_to_robot_calibration(
        interaction_root=tmp_path, calibration_id="expired_fixture", source_frame=camera["frame"],
        target_frame="world", target_from_camera_transform=np.eye(4), valid_from_ns=0,
        valid_until_ns=1, provenance_source="controlled_fixture", provenance_sha256="a" * 64,
        calibration_number=3,
    )
    expired_ref = pin(tmp_path, expired.record_path)
    producer.authorized[expired_ref["ref"]] = expired_ref["sha256"]
    rejected = runtime._geometry(producer, "observed_surface", {**selected, "calibration_ref": expired_ref["ref"]})
    assert rejected["status"] == "unavailable" and rejected["compatible_calibration_refs"] == [good]
    oversized = append_record(tmp_path, tmp_path / "products/test_evidence", "large.json", {
        "record_type": "DocumentEvidence", "extracted_text": "x" * 12001,
    })
    producer.authorized[oversized["ref"]] = oversized["sha256"]
    assert "error" in runtime._geometry(producer, "read_record", {"record_ref": oversized["ref"], "field_path": "/extracted_text"})
    opaque_pointer = presentation.handle("/cameras/0/candidates/0") + "/point_count"
    selected_value = runtime._geometry(producer, "read_record", {"record_ref": ref, "field_path": opaque_pointer})
    assert selected_value["value"] == segmentation["cameras"][0]["candidates"][0]["point_count"]
    path = tmp_path / ref
    content = path.read_bytes()
    path.write_bytes(content + b" ")
    with pytest.raises(ValueError, match="changed"):
        runtime._geometry(producer, "read_observation", {**selected, "field_path": ""})


def test_pa_geometry_calls_require_explicit_conversion_and_retain_partial_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Recover through PA-selected calls and retain geometry without filling task roles."""
    runtime, producer, correspondence_ref, cad_ref = _pa_geometry(tmp_path, monkeypatch)
    original = {path: path.read_bytes() for path in tmp_path.rglob("*.json")}
    calls = []

    class Product:
        async def ask_llm_structured(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
            state = json.loads(prompt.split("\n\n", 1)[1])
            calls.append(state)
            exchanges = state["exchanges"]
            turn = len(exchanges)
            if turn == 0:
                name, arguments = (
                    "inspect_features",
                    {
                        "cad_ref": cad_ref,
                        "pose_ref": correspondence_ref,
                    },
                )
            elif turn == 1:
                assert (
                    "received 'CADSizeCorrespondenceRecord'" in (exchanges[-1]["result"]["reason"])
                )
                name, arguments = "estimate_pose", {"correspondence_ref": correspondence_ref}
            elif turn == 2:
                name, arguments = (
                    "inspect_features",
                    {
                        "cad_ref": cad_ref,
                        "pose_ref": exchanges[-1]["result"]["record_ref"],
                    },
                )
            elif turn == 3:
                assert "received 'CADPoseEstimationRecord'" in exchanges[-1]["result"]["reason"]
                assert "convert_pose" in exchanges[-1]["result"]["reason"]
                name, arguments = (
                    "convert_pose",
                    {
                        "pose_ref": exchanges[1]["result"]["record_ref"],
                    },
                )
            elif turn == 4:
                assert exchanges[-1]["result"]["record"]["record_type"] == "RobotFramePoseRecord"
                name, arguments = (
                    "inspect_features",
                    {
                        "cad_ref": cad_ref,
                        "pose_ref": exchanges[-1]["result"]["record_ref"],
                    },
                )
            else:
                geometry = exchanges[-1]["result"]
                assert geometry["record"]["status"] == "accepted"
                assert geometry["record"]["frame_id"] == "world"
                return {
                    "action": {
                        "kind": "finish",
                        "evidence_refs": [geometry["record_ref"]],
                        "validation_refs": dict.fromkeys(
                            ("part", "goal", "scene", "specification")
                        ),
                        "unresolved": [
                            "Assembly association and scene evidence remain unresolved."
                        ],
                    }
                }
            return {
                "action": {
                    "kind": "investigate",
                    "tool_name": name,
                    "arguments": json.dumps(arguments),
                }
            }

    runtime.product_agent = Product()
    monkeypatch.setattr(
        runtime,
        "_investigation",
        lambda *args: SimpleNamespace(
            handles={}, _canonical_by_pa_ref={}, prior_evidence=(), retrieved_results={}
        ),
    )
    result = asyncio.run(
        asyncio.wait_for(
            runtime.investigate(
                interaction_root=tmp_path,
                directory=producer.directory.parent,
                request={
                    "evidence_refs": [
                        {"ref": ref, "sha256": sha} for ref, sha in producer.authorized.items()
                    ],
                    "target_feature": {},
                    "needs": [],
                },
                max_operations=5,
            ),
            timeout=10,
        )
    )
    assert len(calls) == 6 and result["operations_used"] == 5
    assert result["status"] == "provided" and result["validation_refs"] == {}
    assert len(result["evidence_refs"]) == 1 and result["unresolved"]
    assert all(path.read_bytes() == content for path, content in original.items())


@pytest.mark.parametrize("document_first", [False, True])
def test_pa_reuses_partial_records_and_keeps_investigation_choices(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, document_first: bool
) -> None:
    """Expose prior evidence while PA chooses independent work and an early finish."""
    runtime, producer, correspondence_ref, cad_ref = _pa_geometry(
        tmp_path, monkeypatch, ambiguous=True
    )
    pose = runtime._geometry(producer, "estimate_pose", {"correspondence_ref": correspondence_ref})
    converted = runtime._geometry(producer, "convert_pose", {"pose_ref": pose["record_ref"]})
    document = append_record(
        tmp_path,
        tmp_path / "documents",
        "quantity.json",
        {
            "record_type": "DocumentQueryRecord",
            "text": "0.2 mm",
            "model_name": "excluded_fixture_identifier",
            "recovery_examples": ["excluded_fixture_recipe"],
        },
    )
    producer.authorized[document["ref"]] = document["sha256"]
    request = {
        "target_feature": {},
        "evidence_refs": [{"ref": ref, "sha256": sha} for ref, sha in producer.authorized.items()],
        "needs": [
            {
                "step_index": 1,
                "quantity": "Observed medium gear pose",
                "authority": "PA",
                "reason": "Check the selected pick.",
            },
            {
                "step_index": 6,
                "quantity": "position_tolerance_m",
                "authority": "PA",
                "reason": "Check the selected placement.",
            },
            {
                "step_index": None,
                "quantity": "scene",
                "authority": "PA",
                "reason": "Check the selected motions.",
            },
        ],
    }
    unresolved = [
        "Step 1: Observed medium gear pose was investigated and remains ambiguous.",
        "Program validation scene was not investigated; complete registered geometry is unavailable.",
    ]
    actions = [
        ("inspect_features", {"cad_ref": cad_ref, "pose_ref": converted["record_ref"]}),
        (
            "document_quantity",
            {
                "record_ref": document["ref"],
                "field_path": "/text",
                "start": 0,
                "end": 6,
                "quantity": "position_tolerance_m",
            },
        ),
    ]
    if document_first:
        actions.reverse()
    calls = []

    class Product:
        async def ask_llm_structured(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
            state = json.loads(prompt.split("\n\n", 1)[1])
            calls.append(state)
            assert state["needs"] == request["needs"]
            assert "Review every request in needs" in prompt
            assert "A null step_index identifies a program-level validation need" in prompt
            assert "Finish early when no useful supported operation remains" in prompt
            assert "excluded_fixture_" not in prompt
            catalog = {
                item["record_ref"]: item
                for item in state["evidence_catalog"]
                if "record_ref" in item
            }
            prior = catalog[converted["record_ref"]]
            assert prior["record_type"] == "RobotFramePoseRecord"
            assert prior["pose"] == "ambiguous"
            assert "source_pose" in prior["fields"]
            assert "model_name" not in catalog[document["ref"]]["fields"]
            assert "recovery_examples" not in catalog[document["ref"]]["fields"]
            exchanges = state["exchanges"]
            for exchange in exchanges:
                evidence = exchange["result"]
                assert catalog[evidence["record_ref"]]["status"] == evidence["record"]["status"]
            if len(exchanges) < len(actions):
                name, arguments = actions[len(exchanges)]
                return {
                    "action": {
                        "kind": "investigate",
                        "tool_name": name,
                        "arguments": json.dumps(arguments),
                    }
                }
            assert state["operations_remaining"] == 4
            return {
                "action": {
                    "kind": "finish",
                    "evidence_refs": [
                        converted["record_ref"],
                        *(item["result"]["record_ref"] for item in exchanges),
                    ],
                    "validation_refs": dict.fromkeys(("part", "goal", "scene", "specification")),
                    "unresolved": unresolved,
                }
            }

    runtime.product_agent = Product()
    monkeypatch.setattr(
        runtime,
        "_investigation",
        lambda *args: SimpleNamespace(
            handles={}, _canonical_by_pa_ref={}, prior_evidence=(), retrieved_results={}
        ),
    )
    original = {path: path.read_bytes() for path in tmp_path.rglob("*.json")}
    directory = tmp_path / "composition/refinement_runs/run_0001/pa_0002"
    result = asyncio.run(
        runtime.investigate(
            interaction_root=tmp_path, directory=directory, request=request, max_operations=6
        )
    )
    assert len(calls) == 3 and result["operations_used"] == 2
    assert result["unresolved"] == unresolved and result["validation_refs"] == {}
    assert [item["response"]["action"]["tool_name"] for item in calls[-1]["exchanges"]] == [
        name for name, _ in actions
    ]
    records = [read_pin(tmp_path, ref) for ref in result["evidence_refs"]]
    assert any(record.get("status") == "ambiguous" for record in records)
    assert any(record.get("quantity") == "position_tolerance_m" for record in records)
    assert all(path.read_bytes() == content for path, content in original.items())
    (tmp_path / document["ref"]).write_bytes(original[tmp_path / document["ref"]] + b" ")
    with pytest.raises(ValueError, match="Pinned evidence dependency changed: documents/quantity.json"):
        asyncio.run(
            runtime.investigate(
                interaction_root=tmp_path,
                directory=tmp_path / "composition/refinement_runs/run_0001/pa_0003",
                request=request,
                max_operations=6,
            )
        )
    assert len(calls) == 3


@pytest.mark.parametrize("measurement_tool", ["pick_geometry", "surface_height"])
def test_pa_selected_pick_measurements_return_bindable_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, measurement_tool: str,
) -> None:
    """PA's selected geometry call supplies a support height that RA can bind explicitly."""
    inputs, robot, refs, _ = _setup(tmp_path)
    surface = append_record(
        tmp_path, tmp_path / "products/test_evidence", "surface.json",
        {
            "record_type": "AssemblySurfaceEvidence", "status": "accepted",
            "frame_id": "world", "units": "m", "reference_point": "observed_plane",
            "normal": [0.0, 0.0, 1.0], "offset_m": -0.04, "rms_distance_m": 0.001,
            "observation_timestamp_ns": 1_000_000_000,
        },
    )
    location = append_record(
        tmp_path, tmp_path / "products/test_evidence", "location.json",
        {
            "record_type": "RobotFrameLocationRecord", "location": "available",
            "robot_frame_conversion": "accepted", "target_frame": "world",
            "translated_location_m": [0.1, 0.0, 0.06], "observation_timestamp_ns": 1_000_000_000,
        },
    )
    original = {path: path.read_bytes() for path in tmp_path.rglob("*.json")}
    requested = [{"step_index": 1, "quantity": "/product_geometry/board_center/z", "authority": "PA"}]
    calls, messages = [], []

    class Product:
        async def ask_llm_structured(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
            state = json.loads(prompt.split("\n\n", 1)[1])
            calls.append(state)
            assert state["needs"] == requested
            assert "An absent derived record is not an unavailable source measurement" in prompt
            assert "RA alone binds returned evidence" in prompt
            tools = {item["name"]: item for item in state["tools"] if "name" in item}
            assert "support_plane" in tools["observed_surface"]["description"]
            assert "product_geometry.board_center.z" in tools["pick_geometry"]["description"]
            assert "product_geometry.board_center.z" in tools["surface_height"]["description"]
            assert "slot_floor_z_m" in tools["assembly_geometry"]["description"]
            assert "no default tolerances" in tools["validation_specification"]["description"]
            if not state["exchanges"]:
                arguments = {"surface_ref": surface["ref"]}
                arguments.update(
                    {"part_ref": refs["part"]["ref"]} if measurement_tool == "pick_geometry"
                    else {"location_ref": location["ref"]}
                )
                return {
                    "action": {
                        "kind": "investigate", "tool_name": measurement_tool,
                        "arguments": json.dumps(arguments),
                    }
                }
            measured = state["exchanges"][-1]["result"]
            assert measured["record"]["product_geometry"]["board_center"]["z"] == pytest.approx(0.04)
            return {
                "action": {
                    "kind": "finish", "evidence_refs": [measured["record_ref"]],
                    "validation_refs": dict.fromkeys(("part", "goal", "scene", "specification")),
                    "unresolved": ["Assembly acceptance tolerances remain unavailable."],
                }
            }

    runtime = ProductPrimitiveContextRuntime(SimpleNamespace(), Product())
    monkeypatch.setattr(
        runtime, "_investigation",
        lambda *args: SimpleNamespace(
            handles={}, _canonical_by_pa_ref={}, prior_evidence=(), retrieved_results={}
        ),
    )

    async def progress(event: Mapping[str, Any]) -> None:
        messages.append(event["message"])

    outcome = asyncio.run(runtime.investigate(
        interaction_root=tmp_path,
        directory=tmp_path / "composition/refinement_runs/run_0001/pa_0001",
        request={"target_feature": {}, "needs": requested, "evidence_refs": [refs["part"], surface, location]},
        max_operations=6, progress=progress,
    ))
    assert outcome["operations_used"] == 1 and len(calls) == 2
    assert any(f"measurement/geometry {measurement_tool} accepted" in message for message in messages)
    assert outcome["validation_refs"] == {}
    measurement = outcome["evidence_refs"][0]
    inputs = replace(inputs, record_hashes={**inputs.record_hashes, measurement["ref"]: measurement["sha256"]})
    steps = [{
        "primitive_symbol": "compute_pick_targets",
        "params": {
            "part_name": "medium gear",
            "target_pose": _value(measurement, "/target_pose"),
            "product_geometry": _value(measurement, "/product_geometry"),
        },
    }]
    if measurement_tool == "surface_height":
        steps[0]["params"].update(
            target_pose=_value(refs["pick"], "/target_pose"),
            product_geometry={
                "board_center": {"z": _value(measurement, "/product_geometry/board_center/z")},
                "part_height_m": _value(refs["pick"], "/product_geometry/part_height_m"),
            },
        )
    authored = deepcopy(steps)
    report = asyncio.run(validate_program(
        inputs=inputs, steps=steps, robot={**robot, "captured_at_ns": time.time_ns()},
        evidence={}, directory=tmp_path / "calculations", profile={**_legacy_profile(), "validation_scope": VALIDATION_SCOPE},
        cache={}, session_factory=_PlanningSession,
    ))
    assert report["checked_steps"][0]["status"] == "passed", report
    calculation = read_pin(tmp_path, report["calculation_refs"][0])
    assert calculation["resolved_params"]["product_geometry"]["board_center"]["z"] == pytest.approx(0.04)
    assert report["status"] == "unknown"
    assert any(item["check"] == "specification" and item["status"] == "unknown" for item in report["findings"])
    assert steps == authored
    assert all(path.read_bytes() == content for path, content in original.items())


def test_pa_geometry_preserves_ambiguity_and_rejects_different_cad(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Conversion preserves pose ambiguity and never changes the selected CAD identity."""
    runtime, producer, correspondence_ref, cad_ref = _pa_geometry(
        tmp_path, monkeypatch, ambiguous=True
    )
    pose = runtime._geometry(producer, "estimate_pose", {"correspondence_ref": correspondence_ref})
    assert pose["record"]["pose"] == "ambiguous"
    converted = runtime._geometry(producer, "convert_pose", {"pose_ref": pose["record_ref"]})
    assert converted["record"]["pose"] == "ambiguous"
    geometry = producer.inspect_features(cad_ref, converted["record_ref"])
    assert geometry["record"]["status"] == "ambiguous"
    assert "origin_pose" not in geometry["record"]
    assert "part_height_m" in geometry["record"]
    assert geometry["record"]["uncertainty"]["complete_pose_established"] is False
    with pytest.raises(ValueError, match="accepted measurements"):
        producer.select_grasp_point(geometry["record_ref"], "plane_0001", "circle_0001")
    mismatched = deepcopy(converted["record"])
    mismatched["CAD"]["record"] = append_record(
        tmp_path, tmp_path / "mismatched", "different_CAD.json", producer.read(cad_ref)
    )
    reference = append_record(tmp_path, tmp_path / "mismatched", "pose.json", mismatched)
    producer.authorized[reference["ref"]] = reference["sha256"]
    with pytest.raises(ValueError, match="different CAD record"):
        producer.inspect_features(cad_ref, reference["ref"])


def test_pa_geometry_conversion_requires_approved_calibration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Missing calibration cannot produce a RobotFramePoseRecord."""
    runtime, producer, correspondence_ref, _ = _pa_geometry(tmp_path, monkeypatch)
    pose = runtime._geometry(producer, "estimate_pose", {"correspondence_ref": correspondence_ref})
    runtime.runtime._camera_to_world_calibration_runtime = None
    with pytest.raises(ValueError, match="Approved camera calibration is unavailable"):
        runtime._geometry(producer, "convert_pose", {"pose_ref": pose["record_ref"]})
    assert not list(tmp_path.glob("products/grounding/rgb_d_cad_grounding/robot_pose_*"))


def test_height_estimate_only_supplies_the_selected_scalar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Use a pinned estimate for height without admitting partial geometry or support."""
    inputs, robot, refs, _ = _setup(tmp_path)
    runtime, producer, correspondence_ref, cad_ref = _pa_geometry(
        tmp_path, monkeypatch, ambiguous=True, tilt=0.3
    )
    pose = runtime._geometry(producer, "estimate_pose", {"correspondence_ref": correspondence_ref})
    converted = runtime._geometry(producer, "convert_pose", {"pose_ref": pose["record_ref"]})
    geometry = producer.inspect_features(cad_ref, converted["record_ref"])
    height_ref = pin(tmp_path, tmp_path / geometry["record_ref"])
    inputs = replace(inputs, record_hashes={**inputs.record_hashes, **producer.authorized})
    step = _program(refs)[0]
    step["params"]["product_geometry"] = {
        "board_center": _value(refs["pick"], "/product_geometry/board_center"),
        "part_height_m": _value(height_ref, "/part_height_m"),
    }
    before = deepcopy(step)
    for use_as_support in (False, True):
        if use_as_support:
            step["params"]["product_geometry"]["board_center"] = {
                "z": _value(height_ref, "/part_height_m")
            }
        report = asyncio.run(validate_program(
            inputs=inputs, steps=[step], robot={**robot, "captured_at_ns": time.time_ns()},
            evidence={"part": height_ref, "goal": height_ref},
            directory=tmp_path / f"check_height_{use_as_support}",
            profile={**_legacy_profile(), "validation_scope": VALIDATION_SCOPE}, cache={}, session_factory=_PlanningSession,
        ))
        assert report["status"] == "unknown"
        assert report["checked_steps"][0]["status"] == ("unknown" if use_as_support else "passed")
        assert {item["check"] for item in report["findings"] if item["status"] == "unknown"} >= {
            "part", "goal", "scene", "specification",
        }
        if not use_as_support:
            assert step == before
            assert any(item["check"] == "part_height_m" and item["status"] == "warning" for item in report["findings"])
            assert read_pin(tmp_path, report["calculation_refs"][0])["result"]["part_height"] == geometry["record"]["part_height_m"]
    path = tmp_path / geometry["record_ref"]
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="Pinned evidence dependency changed"):
        verify_evidence_tree(tmp_path, height_ref)


def test_height_estimate_requires_one_observed_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An identity conflict cannot become a height estimate by picking a convenient fit."""
    runtime, producer, correspondence_ref, cad_ref = _pa_geometry(tmp_path, monkeypatch, ambiguous=True)
    pose = runtime._geometry(producer, "estimate_pose", {"correspondence_ref": correspondence_ref})
    converted = runtime._geometry(producer, "convert_pose", {"pose_ref": pose["record_ref"]})
    source = deepcopy(pose["record"])
    source["qualified_pose_hypotheses"][1]["candidate_handle"] = "different_candidate"
    source_ref = append_record(tmp_path, tmp_path / "conflict", "source.json", source)
    frame = deepcopy(converted["record"])
    frame["source_pose"] = source_ref
    frame_ref = append_record(tmp_path, tmp_path / "conflict", "frame.json", frame)
    producer.authorized[frame_ref["ref"]] = frame_ref["sha256"]
    result = producer.inspect_features(cad_ref, frame_ref["ref"])
    assert result["record"]["status"] == "ambiguous"
    assert "part_height_m" not in result["record"]


@pytest.mark.parametrize("source", ["CAD_local", "unverified_literal"])
def test_rejected_measurement_bindings_reach_pa_without_replacement(tmp_path: Path, source: str) -> None:
    """A populated height can need evidence without authorizing a host binding change."""
    inputs, _, refs, _ = _setup(tmp_path)
    cad = append_record(tmp_path, tmp_path / "products/test_evidence", "cad.json", {
        "record_type": "CADMeshRecord", "coordinate_frame": "CAD_local", "height_m": 0.02,
    })
    inputs = replace(inputs, record_hashes={**inputs.record_hashes, cad["ref"]: cad["sha256"]})
    step = _program(refs)[0]
    step["params"]["product_geometry"] = {
        "board_center": _value(refs["pick"], "/product_geometry/board_center"),
        "part_height_m": _value(cad, "/height_m") if source == "CAD_local" else 0.02,
    }
    original = deepcopy(step)
    report = assess_program_dependencies(
        [step], inputs.catalog, inputs.composition_input["robot_state"],
        read_evidence=lambda ref, pointer: _evidence_value(inputs, ref, pointer),
        result_schema=lambda ref: _result_schema([step], ref, inputs),
    )
    status = "incompatible" if source == "CAD_local" else "unverified"
    issue = next(item for item in report["issues"] if item["status"] == status)
    assert len(report["context_requests"]) == 1
    need = report["context_requests"][0]
    assert (need["authority"], need["step_index"], need["quantity"]) == (
        "PA", 1, "/product_geometry/part_height_m",
    )
    assert need["reason"] == issue["message"]
    assert need["quantity_schema"]["x-binding-role"] == "vertical_part_height"
    assert step == original


@pytest.mark.parametrize("symbol", ["detect_parts", "move_relative", "move_to_named_pose"])
def test_unsupported_primitive_stops_before_robot_capture_or_pa_investigation(
    tmp_path: Path, symbol: str,
) -> None:
    """Do not spend measurement budget on a program with an unavailable operation."""
    inputs, _, _, _ = _setup(tmp_path)
    steps = [(symbol, {}), ("compute_pick_targets", {"part_name": "medium gear"})]
    model = _ProgramRuntime([_program_action(steps)])

    async def unexpected_validation(**kwargs: Any) -> Any:
        pytest.fail("Unsupported submissions must stop before validation or measurements.")

    result = asyncio.run(PrimitiveRefinementRuntime(
        program_runtime=model, robot_runtime=object(), product_runtime=object(),
        validator=unexpected_validation,
    ).compose(tmp_path))
    assert result["status"] == "invalid"
    assert result["pa_operations"] == result["pa_batches"] == 0
    assert result["candidate_refs"] == []
    assert len(model.calls) == len(result["decision_refs"]) == 1
    decision = read_pin(tmp_path, result["decision_refs"][0])
    assert symbol in decision["reason"] and "no validated execution path" in decision["reason"]
    assert decision["primitive_steps"] == [
        {"primitive_symbol": name, "params": params} for name, params in steps
    ]
    view = enrich_composition_diagnostic(tmp_path, {}, inputs.context_refs)
    assert view["status"] == "invalid" and view["attempt_count"] == 0
    assert not any(event["stage"] in {"robot_context", "evidence", "validating"}
                   for event in view["refinement"]["events"])


@pytest.mark.parametrize("scope", [VALIDATION_SCOPE, GAZEBO_PICK_PLACE_SCOPE])
def test_historical_unsupported_program_is_readable_but_fails_before_geometry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, scope: str,
) -> None:
    """Preserve old programs without granting them new validation authority."""
    inputs, _, _, _ = _setup(tmp_path)
    steps = [("detect_parts", {}), ("compute_pick_targets", {"part_name": "medium gear"})]
    with monkeypatch.context() as patch:
        patch.setattr(primitive_composition, "supported_primitive_symbols", lambda scope: frozenset(inputs.catalog))
        old = asyncio.run(author_primitive_program_candidate(
            _ProgramRuntime([_program_action(steps)]), tmp_path, validation_scope=scope,
        ))
    assert old.record["status"] == "proposed"
    original = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    view = read_primitive_composition_diagnostic(tmp_path)
    assert view["status"] == "proposed" and view["candidate"] == old.record
    report = asyncio.run(validate_program(
        inputs=inputs, steps=old.record["primitive_steps"], robot=None, evidence={},
        directory=tmp_path / "unused_calculations", profile={"validation_scope": scope}, cache={},
    ))
    assert report["status"] == "failed"
    assert [(item["step_index"], item["check"], item["authority"]) for item in report["findings"]] == [
        (1, "primitive_capability", "RA"),
    ]
    assert report["calculation_refs"] == report["checked_steps"] == []
    assert not (tmp_path / "unused_calculations").exists()
    assert all(path.read_bytes() == data for path, data in original.items())


def test_observed_center_reports_the_unresolved_cad_origin(tmp_path: Path) -> None:
    inputs, _, refs, _ = _setup(tmp_path)
    observed = append_record(tmp_path, tmp_path / "products/test_evidence", "location.json", {
        "record_type": "RobotFrameLocationRecord", "target_frame": "world",
        "translated_location_m": [0.0, 0.0, 0.05],
    })
    inputs = replace(inputs, record_hashes={**inputs.record_hashes, observed["ref"]: observed["sha256"]})
    step = _program(refs)[0]
    step["params"]["target_pose"] = {
        axis: _value(observed, f"/translated_location_m/{index}")
        for index, axis in enumerate(("x", "y", "z"))
    }
    with pytest.raises(BindingUnavailable, match="observed candidate center.*CAD-origin"):
        _geometry_sources(step, inputs)


def test_failed_calculation_preserves_independent_calculations(tmp_path: Path) -> None:
    inputs, robot, refs, _ = _setup(tmp_path)
    steps = [_program(refs, bound=False)[0], _program(refs)[0]]
    before = deepcopy(steps)
    report = asyncio.run(validate_program(
        inputs=inputs, steps=steps, robot=robot, evidence={},
        directory=tmp_path / "independent", profile=_legacy_profile(), cache={},
        session_factory=_PlanningSession,
    ))
    assert report["status"] == "unknown"
    assert [item["status"] for item in report["checked_steps"]] == ["unknown", "passed"]
    assert read_pin(tmp_path, report["calculation_refs"][0])["step_index"] == 2
    assert steps == before


@pytest.mark.parametrize("decision", ["revise", "request_robot", "unsupported"])
@pytest.mark.parametrize("batch", [False, True])
def test_missing_measurements_dispatch_pa_and_leave_ra_in_control(
    tmp_path: Path, decision: str, batch: bool
) -> None:
    """Dispatch missing measurements once per evidence state, then return control to RA."""
    inputs, robot, _, _ = _setup(tmp_path)
    first = _program_action([("compute_pick_targets", {"part_name": "medium gear"})])
    choices = {
        "revise": _program_action([("compute_place_targets", {"part_name": "medium gear"})]),
        "request_robot": {
            "kind": "request_context",
            "requests": [
                {
                    "step_index": 1,
                    "quantity": "Measured EE/TCP context",
                    "authority": "RA",
                    "reason": "Check the tool offset for the selected calculation.",
                }
            ],
        },
        "unsupported": {"kind": "unsupported", "reason": "Required capability is unavailable."},
    }
    reads = []
    if batch:
        ref = inputs.composition_input["target_feature"]["current_state"]["state_values"][0]["value_ref"]["record_ref"]
        reads.append({"kind": "read_records", "requests": [
            {"record_ref": ref, "field_path": f"/translated_location_m/{index}"}
            for index in range(3)
        ]})
    program = _ProgramRuntime([*reads, first, choices[decision]])
    calls = []

    class Robot:
        async def capture_validation_context(self, *args: Any, **kwargs: Any) -> Any:
            return {**deepcopy(robot), "captured_at_ns": time.time_ns()}

    class Product:
        async def investigate(self, **kwargs: Any) -> Any:
            calls.append(kwargs)
            return {
                "status": "unavailable", "operations_used": 1, "evidence_refs": [],
                "validation_refs": {}, "unresolved": ["The requested measurements remain unavailable."],
            }

    async def validator(**kwargs: Any) -> dict[str, Any]:
        return await validate_program(**kwargs, session_factory=_PlanningSession)

    runtime = PrimitiveRefinementRuntime(
        program_runtime=program,
        robot_runtime=Robot(),
        product_runtime=Product(),
        validator=validator,
    )
    result = asyncio.run(runtime.compose(tmp_path))
    assert result["status"] == ("unsupported" if decision == "unsupported" else "no_progress")
    assert result["pa_batches"] == result["pa_operations"] == (2 if decision == "revise" else 1)
    assert {need["quantity"] for need in calls[0]["request"]["needs"]} == {
        "/product_geometry/board_center/z", "/product_geometry/part_height_m", "/target_pose",
        "part", "scene",
    }
    assert all(need["step_index"] == (1 if need["quantity"].startswith("/") else None)
               for need in calls[0]["request"]["needs"])
    revision_prompt = program.calls[len(reads) + 1]["prompt"]
    feedback = json.JSONDecoder().raw_decode(
        revision_prompt.split("\n\nCOMPOSITION_INPUT\n")[1]
    )[0]["refinement_context"]
    assert any(item.get("parameter_path") == "/product_geometry/part_height_m" for item in feedback["findings"])
    assert "The requested measurements remain unavailable." in revision_prompt
    assert {item.get("check") for item in feedback["findings"]} >= {
        "part",
        "scene",
    }
    assert not {"goal", "specification"} & {item.get("check") for item in feedback["findings"]}
    assert all(read_pin(tmp_path, ref)["status"] != "passed" for ref in result["validation_refs"])
    if decision == "revise":
        revised = read_pin(tmp_path, result["candidate_refs"][1])
        assert revised["primitive_steps"] == [
            {"primitive_symbol": "compute_place_targets", "params": {"part_name": "medium gear"}}
        ]


def test_supplemental_requests_reuse_partial_evidence_and_skip_duplicate_fields(tmp_path: Path) -> None:
    """Keep explicit supplemental needs while reusing the preceding measurement batch."""
    _, robot, refs, _ = _setup(tmp_path)
    first = _program_action([("compute_pick_targets", {"part_name": "medium gear"})])
    repeated = {
        "step_index": 1, "quantity": "/product_geometry/part_height_m", "authority": "PA",
        "reason": "The selected calculation needs its height input.",
    }
    supplemental = {
        "step_index": 1, "quantity": "Documented assembly acceptance criteria", "authority": "PA",
        "reason": "The validator needs the supplied acceptance criteria.",
    }
    program = _ProgramRuntime([
        first,
        {"kind": "request_context", "requests": [repeated, supplemental, deepcopy(supplemental)]},
        first,
    ])
    calls = []

    class Robot:
        async def capture_validation_context(self, *args: Any, **kwargs: Any) -> Any:
            return {**deepcopy(robot), "captured_at_ns": time.time_ns()}

    class Product:
        async def investigate(self, **kwargs: Any) -> Any:
            calls.append(kwargs)
            if len(calls) == 1:
                assert len(program.calls) == 1
                return {
                    "status": "provided", "operations_used": 1,
                    "evidence_refs": [refs["part"]], "validation_refs": {},
                    "unresolved": ["Support height remains unavailable."],
                }
            assert len(calls) == 2
            assert kwargs["request"]["needs"] == [supplemental]
            assert refs["part"] in kwargs["request"]["evidence_refs"]
            return {
                "status": "unavailable", "operations_used": 1, "evidence_refs": [],
                "validation_refs": {}, "unresolved": ["Acceptance criteria remain unavailable."],
            }

    async def validator(**kwargs: Any) -> dict[str, Any]:
        return await validate_program(**kwargs, session_factory=_PlanningSession)

    result = asyncio.run(PrimitiveRefinementRuntime(
        program_runtime=program, robot_runtime=Robot(), product_runtime=Product(),
        validator=validator, profile={**_legacy_profile(), "max_pa_batches": 3},
    ).compose(tmp_path))
    assert result["status"] == "no_progress"
    assert result["pa_batches"] == result["pa_operations"] == 2
    assert "Support height remains unavailable." in program.calls[1]["prompt"]
    assert refs["part"]["ref"] in program.calls[1]["prompt"]
    assert "Acceptance criteria remain unavailable." in program.calls[2]["prompt"]
    assert all(
        read_pin(tmp_path, ref)["primitive_steps"] == [
            {"primitive_symbol": "compute_pick_targets", "params": {"part_name": "medium gear"}}
        ]
        for ref in result["candidate_refs"]
    )


def test_changed_selected_inputs_can_request_measurements_from_the_same_catalog(tmp_path: Path) -> None:
    """Changing a selected source invalidates request reuse without changing the evidence pool."""
    inputs, robot, _, _ = _setup(tmp_path)
    location = next(
        {"ref": ref, "sha256": sha}
        for ref, sha in inputs.record_hashes.items()
        if read_pin(tmp_path, {"ref": ref, "sha256": sha}).get("record_type") == "RobotFrameLocationRecord"
    )
    first = _program_action([("compute_pick_targets", {"part_name": "medium gear"})])
    revised = _program_action([("compute_pick_targets", {
        "part_name": "medium gear",
        "target_pose": {axis: _value(location, f"/translated_location_m/{i}") for i, axis in enumerate(("x", "y", "z"))},
    })])
    program = _ProgramRuntime([
        first, revised, {"kind": "unsupported", "reason": "Measurements remain unavailable."},
    ])
    calls = []

    class Robot:
        async def capture_validation_context(self, *args: Any, **kwargs: Any) -> Any:
            return {**deepcopy(robot), "captured_at_ns": time.time_ns()}

    class Product:
        async def investigate(self, **kwargs: Any) -> Any:
            calls.append(kwargs)
            return {
                "status": "unavailable", "operations_used": 1, "evidence_refs": [],
                "validation_refs": {}, "unresolved": ["The selected measurements remain unavailable."],
            }

    async def validator(**kwargs: Any) -> dict[str, Any]:
        return await validate_program(**kwargs, session_factory=_PlanningSession)

    result = asyncio.run(PrimitiveRefinementRuntime(
        program_runtime=program, robot_runtime=Robot(), product_runtime=Product(), validator=validator,
    ).compose(tmp_path))
    assert result["status"] == "unsupported"
    assert len(calls) == result["pa_batches"] == 2
    assert calls[0]["request"]["evidence_refs"] == calls[1]["request"]["evidence_refs"]
    needs = calls[1]["request"]["needs"]
    assert {need["quantity"] for need in needs} == {
        "/product_geometry/part_height_m", "/product_geometry/board_center/z",
        "part", "scene",
    }
    assert all(need["evidence_refs"] for need in needs if need["step_index"] is not None)


def test_pa_feedback_reaches_ra_without_becoming_geometry_and_remains_pinned(
    tmp_path: Path,
) -> None:
    """Propagate PA explanations while preserving their source and evidence boundary."""
    inputs, robot, _, _ = _setup(tmp_path)
    original = {path: path.read_bytes() for path in tmp_path.rglob("*.json")}
    requests = [
        {
            "step_index": 1,
            "quantity": "Observed medium gear geometry and documented seating tolerances",
            "authority": "PA",
            "reason": "The selected grasp and requested assembly need verified geometry.",
        }
    ]
    requests.extend([
        {"step_index": 1, "quantity": "Measured EE/TCP context", "authority": "RA", "reason": "Check the tool context."},
        {"step_index": 1, "quantity": "Observed collision scene", "authority": "PA", "reason": "Check collision coverage."},
        deepcopy(requests[0]),
    ])
    selected_requests = [requests[0], requests[2]]
    program = _ProgramRuntime(
        [
            _program_action([("grasp_part", {"part_name": "medium gear"})]),
            {"kind": "request_context", "requests": requests},
        ]
    )
    unresolved = [
        "The observed pose remains ambiguous.",
        "Approved seating tolerances are unavailable.",
    ]

    class Robot:
        async def capture_validation_context(self, *args: Any, **kwargs: Any) -> Any:
            return {**deepcopy(robot), "captured_at_ns": time.time_ns()}

    class Product:
        async def investigate(self, **kwargs: Any) -> Any:
            if len(program.calls) == 1:
                assert kwargs["request"]["needs"] == [{
                    "step_index": None, "quantity": "part", "authority": "PA",
                    "reason": "PA has not supplied the required part evidence.",
                }]
                return {
                    "status": "unavailable", "operations_used": 1,
                    "evidence_refs": [], "validation_refs": {},
                    "unresolved": ["Initial part evidence is unavailable."],
                }
            assert len(program.calls) == 2
            assert kwargs["request"]["needs"] == selected_requests
            return {
                "status": "unavailable",
                "operations_used": 1,
                "evidence_refs": [],
                "validation_refs": {},
                "unresolved": unresolved,
            }

    async def validator(**kwargs: Any) -> dict[str, Any]:
        return {
            "scope": read_validation_scope(kwargs["profile"]),
            "status": "unknown",
            "findings": [
                {
                    "step_index": None,
                    "check": "part",
                    "status": "unknown",
                    "authority": "PA",
                    "message": "PA has not supplied the required part evidence.",
                }
            ],
            "calculation_refs": [],
        }

    runtime = PrimitiveRefinementRuntime(
        program_runtime=program,
        robot_runtime=Robot(),
        product_runtime=Product(),
        validator=validator,
    )
    result = asyncio.run(runtime.compose(tmp_path))
    assert result["status"] == "no_progress" and result["pa_operations"] == 2
    assert len(program.calls) == 3
    assert all(reason not in program.calls[1]["prompt"] for reason in unresolved)
    assert all(reason in program.calls[2]["prompt"] for reason in unresolved)
    assert "Missing product measurements declared by your selected primitive inputs are dispatched" in program.calls[1]["prompt"]
    diagnostic = read_primitive_composition_diagnostic(tmp_path)
    response = diagnostic["refinement"]["pa_responses"][-1]
    assert response["unresolved"] == unresolved
    event = next(item for item in reversed(diagnostic["refinement"]["events"]) if "pa_response_ref" in item)
    response_ref = event["pa_response_ref"]
    assert read_pin(tmp_path, response_ref) == response
    context_ref = pin(tmp_path, tmp_path / "composition/refinement_runs/run_0001/context_0002.json")
    revised_inputs = _with_refinement(inputs, context_ref)
    assert response_ref["ref"] not in revised_inputs.record_hashes
    assert all(path.read_bytes() == content for path, content in original.items())
    response_path = tmp_path / response_ref["ref"]
    response_path.write_bytes(response_path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="Pinned refinement evidence changed"):
        _with_refinement(inputs, context_ref)
    with pytest.raises(ValueError, match="Pinned refinement evidence changed"):
        enrich_composition_diagnostic(tmp_path, {}, inputs.context_refs)


@pytest.mark.parametrize("scope", [VALIDATION_SCOPE, GAZEBO_PICK_PLACE_SCOPE, GAZEBO_OBSERVED_SCOPE])
@pytest.mark.parametrize("rejected_height", [False, True])
def test_refinement_preserves_first_pass_and_only_ra_supplies_revision(
    tmp_path: Path, scope: str, rejected_height: bool,
) -> None:
    """Return contract-requested evidence and validate only the RA-authored revision."""
    inputs, robot, refs, roles = (_observed_setup if scope == GAZEBO_OBSERVED_SCOPE else _setup)(tmp_path)
    original = {path: path.read_bytes() for path in tmp_path.rglob("*.json")}
    build_program = _observed_program if scope == GAZEBO_OBSERVED_SCOPE else _program
    first, revised = build_program(refs, bound=False), build_program(refs)
    if rejected_height:
        first[0]["params"]["product_geometry"] = {"part_height_m": 0.02}
    program = _ProgramRuntime(
        [
            _program_action([(s["primitive_symbol"], s["params"]) for s in first]),
            _program_action([(s["primitive_symbol"], s["params"]) for s in revised]),
        ]
    )

    class Robot:
        async def capture_validation_context(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
            return {**deepcopy(robot), "captured_at_ns": time.time_ns()}

    class Product:
        calls = []

        async def investigate(self, **kwargs: Any) -> dict[str, Any]:
            assert len(program.calls) == 1
            assert kwargs["request"]["validation_scope"] == scope
            needs = kwargs["request"]["needs"]
            assert all(need["authority"] == "PA" for need in needs)
            assert {(need["step_index"], need["quantity"]) for need in needs} >= {
                (1, "/product_geometry/part_height_m"),
                (1, "/product_geometry/board_center/z"),
            } | ({(6, "/product_geometry/placement_surface_point/z")} if scope == GAZEBO_OBSERVED_SCOPE else {
                (6, "/product_geometry/target_origin_pose/z"),
                (6, "/product_geometry/target_reference/target_point"),
            })
            assert not any(need.get("parameter_path", "").startswith("/pick_ctx") for need in needs)
            assert {need["quantity"] for need in needs if need["step_index"] is None} == set(
                required_validation_roles(scope)
            )
            height = next(need for need in needs if need["quantity"] == "/product_geometry/part_height_m")
            assert height["reason"] == (
                "Part height has no measurement or geometry reference." if rejected_height else
                "Required input is unbound."
            )
            self.calls.append(kwargs)
            return {
                "status": "provided",
                "operations_used": 1,
                "evidence_refs": list(refs.values()),
                "validation_refs": {role: roles[role] for role in required_validation_roles(scope)},
                "unresolved": [],
            }

    async def validator(**kwargs: Any) -> dict[str, Any]:
        return await validate_program(**kwargs, session_factory=_PlanningSession)

    product = Product()
    runtime = PrimitiveRefinementRuntime(
        program_runtime=program, robot_runtime=Robot(), product_runtime=product, validator=validator,
        profile={**_legacy_profile(), "validation_scope": scope},
    )
    events = []

    async def progress(event: Any) -> None:
        events.append(event["stage"])

    result = asyncio.run(runtime.compose(tmp_path, progress=progress))
    assert result["status"] == "validated_for_declared_scope", result
    assert len(result["candidate_refs"]) == 2
    assert len(result["decision_refs"]) == len(program.calls) == 2
    assert read_pin(tmp_path, result["candidate_refs"][0])["primitive_steps"] == first
    assert read_pin(tmp_path, result["candidate_refs"][1])["primitive_steps"] == revised
    assert events.index("proposal") < events.index("evidence")
    assert len(product.calls) == 1
    batch = json.loads(
        (tmp_path / "composition/refinement_runs/run_0001/pa_0001/request.json").read_text()
    )
    assert batch["needs"] == product.calls[0]["request"]["needs"]
    assert "refinement_context" not in program.calls[0]["prompt"]
    assert "previous_candidate" in program.calls[1]["prompt"]
    after = json.JSONDecoder().raw_decode(
        program.calls[1]["prompt"].split("\n\nCOMPOSITION_INPUT\n")[1]
    )[0]
    assert any(
        finding.get("parameter_path") == "/product_geometry/part_height_m"
        for finding in after["refinement_context"]["findings"]
    )
    assert {item["record_ref"] for item in after["grounded_context"]["typed_records"]} >= {
        ref["ref"] for ref in refs.values()
    }
    assert all(path.read_bytes() == content for path, content in original.items())
    diagnostic = read_primitive_composition_diagnostic(tmp_path)
    assert diagnostic["status"] == "validated_for_declared_scope", diagnostic["message"]
    assert diagnostic["candidate"]["primitive_steps"] == revised
    assert diagnostic["validation_scope"] == diagnostic["validation"]["scope"] == scope
    for candidate_ref in result["candidate_refs"]:
        candidate = read_pin(tmp_path, candidate_ref)
        request = read_pin(tmp_path, {"ref": candidate["request_ref"], "sha256": candidate["request_sha256"]})
        assert request["validation_scope"] == scope
        assert f"Validation scope: {scope}" in request["prompt"]
    if scope in {GAZEBO_PICK_PLACE_SCOPE, GAZEBO_OBSERVED_SCOPE}:
        assert diagnostic["message"].startswith("Validated for Gazebo pick-and-place")
    if scope == GAZEBO_OBSERVED_SCOPE:
        delivered = after["refinement_context"]["resolved_measurements"]
        assert any(item["record_ref"] == refs["pick"]["ref"] and
                   item["value"]["part_height_m"] == 0.02 for item in delivered)


@pytest.mark.parametrize(
    "max_operations,max_batches,operations_used",
    [(1, 2, 1), (12, 1, 1), (12, 2, 7)],
)
def test_ra_requests_respect_pa_operation_and_batch_limits(
    tmp_path: Path, max_operations: int, max_batches: int, operations_used: int
) -> None:
    """Keep explicit requests bounded and reject a producer that exceeds its allowance."""
    _, robot, _, _ = _setup(tmp_path)
    requests = [
        {
            "step_index": 1,
            "quantity": quantity,
            "authority": "PA",
            "reason": "Ground an input of compute_pick_targets.",
        }
        for quantity in ("part_height_m", "board_center.z")
    ]
    program = _ProgramRuntime(
        [
            _program_action([("compute_pick_targets", {"part_name": "medium gear"})]),
            *[{"kind": "request_context", "requests": [request]} for request in requests],
            {"kind": "unsupported", "reason": "Required evidence remains unavailable."},
        ]
    )
    calls = []

    class Robot:
        async def capture_validation_context(self, *args: Any, **kwargs: Any) -> Any:
            return {**deepcopy(robot), "captured_at_ns": time.time_ns()}

    class Product:
        async def investigate(self, **kwargs: Any) -> Any:
            calls.append(kwargs)
            return {
                "status": "unavailable",
                "operations_used": operations_used,
                "evidence_refs": [],
                "validation_refs": {},
                "unresolved": ["The requested measurement remains unavailable."],
            }

    async def validator(**kwargs: Any) -> dict[str, Any]:
        return await validate_program(**kwargs, session_factory=_PlanningSession)

    runtime = PrimitiveRefinementRuntime(
        program_runtime=program,
        robot_runtime=Robot(),
        product_runtime=Product(),
        validator=validator,
        profile={
            **_legacy_profile(),
            "max_pa_operations": max_operations,
            "max_pa_batches": max_batches,
        },
    )
    result = asyncio.run(runtime.compose(tmp_path))
    assert len(calls) == result["pa_batches"] == 1
    assert calls[0]["max_operations"] == min(6, max_operations)
    assert {need["quantity"] for need in calls[0]["request"]["needs"]} == {
        "/product_geometry/part_height_m", "/product_geometry/board_center/z", "/target_pose",
        "part", "scene",
    }
    if operations_used > calls[0]["max_operations"]:
        assert result["status"] == "failed"
        assert "PA exceeded its evidence budget" in result["stop_reason"]
    else:
        assert result["status"] == "unsupported"
        assert result["pa_operations"] == operations_used
        assert len(program.calls) == 4


def test_duplicate_call_joins_one_run_and_cancellation_persists(tmp_path: Path) -> None:
    inputs, robot, refs, roles = _setup(tmp_path)
    gate = asyncio.Event()
    started = asyncio.Event()
    calls = []

    class Program:
        async def author_composition_action(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
            calls.append(kwargs)
            started.set()
            await gate.wait()
            return {"action": _program_action([("grasp_part", {"part_name": "medium gear"})])}

    async def scenario() -> Any:
        runtime = PrimitiveRefinementRuntime(program_runtime=Program(), robot_runtime=object())
        first = asyncio.create_task(runtime.compose(tmp_path, deadline_sec=900))
        await started.wait()
        second = asyncio.create_task(runtime.compose(tmp_path))
        await asyncio.sleep(0.02)
        assert len(calls) == 1
        assert cancel_primitive_refinement(tmp_path)
        outcomes = await asyncio.gather(first, second)
        assert runtime.profile["deadline_sec"] == 300
        first_request = read_pin(tmp_path, outcomes[0]["request_ref"])
        assert first_request["profile"]["deadline_sec"] == 900
        started.clear()
        normal = asyncio.create_task(runtime.compose(tmp_path))
        await started.wait()
        assert cancel_primitive_refinement(tmp_path)
        normal_result = await normal
        assert read_pin(tmp_path, normal_result["request_ref"])["profile"]["deadline_sec"] == 300
        return outcomes

    results = asyncio.run(scenario())
    assert all(result["status"] == "cancelled" for result in results)
    assert len(list((tmp_path / "composition/refinement_runs").glob("run_*"))) == 2


@pytest.mark.parametrize("deadline", [0, -1, 3601, float("inf"), float("nan"), True])
def test_composition_rejects_invalid_deadline_before_start(tmp_path: Path, deadline: Any) -> None:
    runtime = PrimitiveRefinementRuntime(program_runtime=object(), robot_runtime=object())
    with pytest.raises(ValueError, match="deadline"):
        asyncio.run(runtime.compose(tmp_path, deadline_sec=deadline))
    assert not (tmp_path / "composition").exists()


def test_changed_mesh_invalidates_derived_evidence(tmp_path: Path) -> None:
    _, _, refs, _ = _setup(tmp_path)
    verify_evidence_tree(tmp_path, refs["part"])
    (tmp_path / "mesh.npz").write_bytes(b"changed geometry")
    with pytest.raises(ValueError, match="dependency changed"):
        verify_evidence_tree(tmp_path, refs["part"])


@pytest.mark.parametrize(
    "fault", ["robot_age", "scene_age", "frame", "identity", "coverage", "tolerance"]
)
def test_incomplete_or_incompatible_validation_never_passes(tmp_path: Path, fault: str) -> None:
    inputs, robot, refs, roles = _setup(tmp_path)
    if fault == "robot_age":
        robot["captured_at_ns"] -= 5 * 10**9
    else:
        role = {
            "scene_age": "scene",
            "frame": "part",
            "identity": "goal",
            "coverage": "scene",
            "tolerance": "specification",
        }[fault]
        record = read_pin(tmp_path, roles[role])
        if fault == "scene_age":
            record["observation_timestamp_ns"] = 0
        elif fault == "frame":
            record["frame_id"] = "camera"
        elif fault == "identity":
            record["part_object_id"] = "another_observed_instance"
        elif fault == "coverage":
            record["coverage"] = "incomplete"
        else:
            record.pop("position_tolerance_m")
        roles[role] = append_record(tmp_path, tmp_path / "fault", "record.json", record)
    report = asyncio.run(
        validate_program(
            inputs=inputs,
            steps=_program(refs),
            robot=robot,
            evidence=roles,
            directory=tmp_path / "calculations",
            profile={**_legacy_profile(), "validation_scope": VALIDATION_SCOPE},
            cache={},
            session_factory=_PlanningSession,
        )
    )
    assert report["status"] != "passed", report


def test_refinement_deadline_and_unchanged_failures_stop(tmp_path: Path) -> None:
    inputs, robot, refs, roles = _setup(tmp_path)
    program = _ProgramRuntime([_program_action([("grasp_part", {"part_name": "medium gear"})])] * 3)

    class Robot:
        async def capture_validation_context(self, *args: Any, **kwargs: Any) -> Any:
            return {**deepcopy(robot), "captured_at_ns": time.time_ns()}

    async def unknown(**kwargs: Any) -> dict[str, Any]:
        return {
            "scope": read_validation_scope(kwargs["profile"]),
            "status": "unknown",
            "findings": [
                {
                    "check": "unsupported",
                    "status": "unknown",
                    "message": "Unsupported fixture operation.",
                }
            ],
            "calculation_refs": [],
        }

    runtime = PrimitiveRefinementRuntime(
        program_runtime=program, robot_runtime=Robot(), validator=unknown
    )
    result = asyncio.run(runtime.compose(tmp_path))
    assert result["status"] == "no_progress", result
    assert len(result["candidate_refs"]) == 2

    class WaitingProgram:
        async def author_composition_action(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
            await asyncio.Event().wait()

    profile = {**_legacy_profile(), "deadline_sec": 0.1}
    runtime = PrimitiveRefinementRuntime(
        program_runtime=WaitingProgram(), robot_runtime=Robot(), profile=profile
    )
    result = asyncio.run(runtime.compose(tmp_path))
    assert result["status"] == "budget_exhausted"
    assert "deadline" in result["stop_reason"]
