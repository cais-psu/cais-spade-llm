from __future__ import annotations

"""Exercise audited calculations and bounded refinement without external model calls or motion."""

import asyncio
import json
import logging
import time
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from cais_spade_llm.spec2primitives.agents.pa.primitive_context import (
    ProductPrimitiveContextRuntime,
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
    read_primitive_composition_diagnostic,
)
from cais_spade_llm.spec2primitives.agents.ra.program_dependencies import (
    assess_program_dependencies,
)
from cais_spade_llm.spec2primitives.agents.ra.program_validation import (
    BindingUnavailable,
    _geometry_sources,
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
from cais_spade_llm.spec2primitives.tools.rgb_d_cad_grounding import (
    pose_estimation,
    record_camera_to_robot_calibration,
)


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


def _evidence(root: Path) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]]]:
    np.savez(
        root / "mesh.npz", triangles_m=np.asarray([[[0, 0, 0], [0.02, 0, 0], [0, 0.02, 0.02]]])
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


def _setup(root: Path) -> tuple[Any, dict[str, Any], dict[str, Any], dict[str, Any]]:
    _capture_geometry_context(root)
    inputs = _load_inputs(root)
    refs, roles = _evidence(root)
    inputs = replace(
        inputs,
        record_hashes={
            **inputs.record_hashes,
            **{item["ref"]: item["sha256"] for item in refs.values()},
        },
    )
    return inputs, _robot(inputs), refs, roles


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
    profile = load_refinement_profile()
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
        return {"status": "passed" if after_validation else "unknown", "findings": []}

    async def progress(event: Any) -> None:
        events.append(deepcopy(event))

    runtime = PrimitiveRefinementRuntime(
        program_runtime=program,
        robot_runtime=Robot(),
        validator=validator,
        profile={**load_refinement_profile(), "max_candidates": 1 if after_validation else 2},
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
            profile=load_refinement_profile(),
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
            profile=load_refinement_profile(),
            cache={},
            session_factory=_PlanningSession,
        )
    )
    assert report["status"] == "unknown"


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
    with pytest.raises(CalculationUnavailable, match="lateral"):
        calculate_target("compute_pick_targets", params, shifted, _pose())
    place = {
        "part_name": "medium gear",
        "pick_ctx": output,
        "product_geometry": read_pin(tmp_path, refs["goal"])["product_geometry"],
        "destination_location": "middle gear shaft destination reference",
    }
    with pytest.raises(CalculationUnavailable, match="token"):
        calculate_target("compute_place_targets", place, robot, _pose())


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


def _pa_geometry(
    root: Path, monkeypatch: pytest.MonkeyPatch, *, ambiguous: bool = False,
    tilt: float = 0.0, world_transform: np.ndarray | None = None,
) -> tuple[ProductPrimitiveContextRuntime, AssemblyGeometryProducer, str, str]:
    correspondence_path = _prepare_correspondence(root, (0.042,))
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
                "step_index": 6,
                "quantity": "Observed collision scene",
                "authority": "PA",
                "reason": "Check the selected motions.",
            },
        ],
    }
    unresolved = [
        "Step 1: Observed medium gear pose was investigated and remains ambiguous.",
        "Step 6: Observed collision scene was not investigated; complete registered geometry is unavailable.",
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
            profile=load_refinement_profile(), cache={}, session_factory=_PlanningSession,
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


def test_invalid_bindings_do_not_request_replacement_measurements(tmp_path: Path) -> None:
    inputs, _, refs, _ = _setup(tmp_path)
    cad = append_record(tmp_path, tmp_path / "products/test_evidence", "cad.json", {
        "record_type": "CADMeshRecord", "coordinate_frame": "CAD_local", "height_m": 0.02,
    })
    inputs = replace(inputs, record_hashes={**inputs.record_hashes, cad["ref"]: cad["sha256"]})
    step = _program(refs)[0]
    step["params"]["product_geometry"] = {
        "board_center": _value(refs["pick"], "/product_geometry/board_center"),
        "part_height_m": _value(cad, "/height_m"),
    }
    report = assess_program_dependencies(
        [step], inputs.catalog, inputs.composition_input["robot_state"],
        read_evidence=lambda ref, pointer: _evidence_value(inputs, ref, pointer),
        result_schema=lambda ref: _result_schema([step], ref, inputs),
    )
    assert any(item["status"] == "incompatible" for item in report["issues"])
    assert report["context_requests"] == []


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
        directory=tmp_path / "independent", profile=load_refinement_profile(), cache={},
        session_factory=_PlanningSession,
    ))
    assert report["status"] == "unknown"
    assert [item["status"] for item in report["checked_steps"]] == ["unknown", "passed"]
    assert read_pin(tmp_path, report["calculation_refs"][0])["step_index"] == 2
    assert steps == before


@pytest.mark.parametrize("decision", ["revise", "request_robot", "unsupported"])
def test_missing_measurements_dispatch_pa_and_leave_ra_in_control(
    tmp_path: Path, decision: str
) -> None:
    """Dispatch missing measurements once per evidence state, then return control to RA."""
    _, robot, _, _ = _setup(tmp_path)
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
    program = _ProgramRuntime([first, choices[decision]])
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
    }
    assert all(need["step_index"] == 1 for need in calls[0]["request"]["needs"])
    feedback = json.JSONDecoder().raw_decode(
        program.calls[1]["prompt"].split("\n\nCOMPOSITION_INPUT\n")[1]
    )[0]["refinement_context"]
    assert any(item.get("parameter_path") == "/product_geometry/part_height_m" for item in feedback["findings"])
    assert "The requested measurements remain unavailable." in program.calls[1]["prompt"]
    assert {item.get("check") for item in feedback["findings"]} >= {
        "part",
        "goal",
        "scene",
        "specification",
    }
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
        validator=validator, profile={**load_refinement_profile(), "max_pa_batches": 3},
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
    }
    assert all(need["evidence_refs"] for need in needs)


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
    assert result["status"] == "no_progress" and result["pa_operations"] == 1
    assert len(program.calls) == 3
    assert all(reason not in program.calls[1]["prompt"] for reason in unresolved)
    assert all(reason in program.calls[2]["prompt"] for reason in unresolved)
    assert "Missing product measurements declared by your selected primitive inputs are dispatched" in program.calls[1]["prompt"]
    diagnostic = read_primitive_composition_diagnostic(tmp_path)
    response = diagnostic["refinement"]["pa_responses"][0]
    assert response["unresolved"] == unresolved
    event = next(item for item in diagnostic["refinement"]["events"] if "pa_response_ref" in item)
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


def test_refinement_preserves_first_pass_and_only_ra_supplies_revision(tmp_path: Path) -> None:
    """Return contract-requested evidence and validate only the RA-authored revision."""
    inputs, robot, refs, roles = _setup(tmp_path)
    original = {path: path.read_bytes() for path in tmp_path.rglob("*.json")}
    first, revised = _program(refs, bound=False), _program(refs)
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
            needs = kwargs["request"]["needs"]
            assert all(need["authority"] == "PA" for need in needs)
            assert {(need["step_index"], need["quantity"]) for need in needs} >= {
                (1, "/product_geometry/part_height_m"),
                (1, "/product_geometry/board_center/z"),
                (6, "/product_geometry/target_origin_pose/z"),
                (6, "/product_geometry/target_reference/target_point"),
            }
            assert not any(need["parameter_path"].startswith("/pick_ctx") for need in needs)
            self.calls.append(kwargs)
            return {
                "status": "provided",
                "operations_used": 1,
                "evidence_refs": list(refs.values()),
                "validation_refs": roles,
                "unresolved": [],
            }

    async def validator(**kwargs: Any) -> dict[str, Any]:
        return await validate_program(**kwargs, session_factory=_PlanningSession)

    product = Product()
    runtime = PrimitiveRefinementRuntime(
        program_runtime=program, robot_runtime=Robot(), product_runtime=product, validator=validator
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
            **load_refinement_profile(),
            "max_pa_operations": max_operations,
            "max_pa_batches": max_batches,
        },
    )
    result = asyncio.run(runtime.compose(tmp_path))
    assert len(calls) == result["pa_batches"] == 1
    assert calls[0]["max_operations"] == min(6, max_operations)
    assert {need["quantity"] for need in calls[0]["request"]["needs"]} == {
        "/product_geometry/part_height_m", "/product_geometry/board_center/z", "/target_pose",
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
        first = asyncio.create_task(runtime.compose(tmp_path))
        await started.wait()
        second = asyncio.create_task(runtime.compose(tmp_path))
        await asyncio.sleep(0.02)
        assert len(calls) == 1
        assert cancel_primitive_refinement(tmp_path)
        return await asyncio.gather(first, second)

    results = asyncio.run(scenario())
    assert all(result["status"] == "cancelled" for result in results)
    assert len(list((tmp_path / "composition/refinement_runs").glob("run_*"))) == 1


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
            profile=load_refinement_profile(),
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

    profile = {**load_refinement_profile(), "deadline_sec": 0.1}
    runtime = PrimitiveRefinementRuntime(
        program_runtime=WaitingProgram(), robot_runtime=Robot(), profile=profile
    )
    result = asyncio.run(runtime.compose(tmp_path))
    assert result["status"] == "budget_exhausted"
    assert "deadline" in result["stop_reason"]
