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

from cais_spade_llm.spec2primitives.adapters.target_calculation import (
    CalculationUnavailable,
    calculate_target,
)
from cais_spade_llm.spec2primitives.agents.ra.primitive_composition import (
    _evidence_value,
    _load_inputs,
    _result_schema,
    read_primitive_composition_diagnostic,
)
from cais_spade_llm.spec2primitives.agents.ra.program_dependencies import (
    assess_program_dependencies,
)
from cais_spade_llm.spec2primitives.agents.ra.program_validation import (
    resolve_selected_values,
    validate_program,
)
from cais_spade_llm.spec2primitives.agents.ra.refinement import (
    PrimitiveRefinementRuntime,
    cancel_primitive_refinement,
    load_refinement_profile,
)
from cais_spade_llm.spec2primitives.agents.ra.refinement_records import (
    append_record,
    pin,
    read_pin,
    verify_evidence_tree,
)
from cais_spade_llm.spec2primitives.tests.test_ra_context_handoff import (
    _ProgramRuntime,
    _capture_geometry_context,
    _program_action,
)
from cais_spade_llm.spec2primitives.tools.assembly_geometry import (
    invariant_quantities,
    planar_features,
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
        item["authority"] == "PA" and item["parameter_path"] == "/product_geometry"
        for item in report["context_requests"]
    )
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


def test_symmetric_hypotheses_only_supply_invariant_quantities() -> None:
    values = invariant_quantities(
        [{"height": 0.02, "yaw": 0.0}, {"height": 0.02001, "yaw": 3.14}],
        {"height": 0.0001, "yaw": 0.01},
    )
    assert values == {"height": pytest.approx(0.020005)}
    assert invariant_quantities([], {"height": 0.1}) == {}


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


def test_refinement_preserves_first_pass_and_only_ra_supplies_revision(tmp_path: Path) -> None:
    inputs, robot, refs, roles = _setup(tmp_path)
    original = {path: path.read_bytes() for path in tmp_path.rglob("*.json")}
    first, revised = _program(refs, bound=False), _program(refs)
    program = _ProgramRuntime(
        [
            _program_action([(s["primitive_symbol"], s["params"]) for s in steps])
            for steps in (first, revised)
        ]
    )

    class Robot:
        async def capture_validation_context(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
            return {**deepcopy(robot), "captured_at_ns": time.time_ns()}

    class Product:
        calls = []

        async def investigate(self, **kwargs: Any) -> dict[str, Any]:
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
    assert read_pin(tmp_path, result["candidate_refs"][0])["primitive_steps"] == first
    assert read_pin(tmp_path, result["candidate_refs"][1])["primitive_steps"] == revised
    assert events.index("proposal") < events.index("evidence")
    assert all(need["authority"] == "PA" for need in product.calls[0]["request"]["needs"])
    assert "refinement_context" not in program.calls[0]["prompt"]
    assert "previous_candidate" in program.calls[1]["prompt"]
    assert all(path.read_bytes() == content for path, content in original.items())
    diagnostic = read_primitive_composition_diagnostic(tmp_path)
    assert diagnostic["status"] == "validated_for_declared_scope", diagnostic["message"]
    assert diagnostic["candidate"]["primitive_steps"] == revised


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
