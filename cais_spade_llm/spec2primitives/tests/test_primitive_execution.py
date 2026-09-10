from __future__ import annotations

"""Test persisted execution authority and command ordering without live motion."""

import asyncio
import threading
import time
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import sys
from typing import Any

import numpy as np
import pytest

from cais_spade_llm.spec2primitives.adapters.gazebo_execution import (
    GazeboExecutionSession,
    fixture_instances,
    load_execution_profile,
    match_instance,
    prepare_trajectory,
)
from cais_spade_llm.spec2primitives.adapters.robot_validation_context import (
    matrix_pose,
    pose_matrix,
)
from cais_spade_llm.spec2primitives.agents.ra.execution_state import (
    assert_execution_available,
    execution_busy,
    execution_custody,
    read_primitive_execution_diagnostic,
)
from cais_spade_llm.spec2primitives.agents.ra.program_execution import (
    PrimitiveExecutionRuntime,
    load_validated_program,
)
from cais_spade_llm.spec2primitives.agents.ra.program_validation import validate_program
from cais_spade_llm.spec2primitives.agents.ra.refinement import PrimitiveRefinementRuntime, load_refinement_profile
from cais_spade_llm.spec2primitives.agents.ra.validation_scope import (
    GAZEBO_OBSERVED_SCOPE, GAZEBO_PICK_PLACE_SCOPE, VALIDATION_SCOPE, read_validation_scope, required_validation_roles,
    supported_primitive_symbols,
)
from cais_spade_llm.spec2primitives.agents.ra.refinement_records import (
    append_record,
    fingerprint,
    pin,
    read_pin,
    verify_evidence_tree,
)
from cais_spade_llm.spec2primitives.agents.ra.primitive_composition import _evidence_value
from cais_spade_llm.spec2primitives.tests.test_primitive_refinement import (
    _setup,
    _program,
    _PlanningSession,
    _observed_setup,
    _observed_program,
    _fitting_setup,
    _binding_fixture,
    _MessageProgramRuntime,
)
from cais_spade_llm.spec2primitives.tests.test_ra_context_handoff import (
    _program_action,
)
from cais_spade_llm.spec2primitives.tools.assembly_geometry import AssemblyGeometryProducer
from cais_spade_llm.spec2primitives.tools.exact_ref_resolver import approved_cad_path

_SHARE = Path(__file__).resolve().parents[3] / "ros2/cais_lab_robotics"


@pytest.mark.parametrize("fault", [None, "ambiguous", "position", "changed_mesh"])
def test_observed_instance_matching_uses_bounds_and_requires_one_match(tmp_path: Path, fault: str | None) -> None:
    """Post-composition matching accepts symmetric orientation without guessing an instance."""
    vertices = np.asarray([[-0.02, -0.02, -0.01], [0.02, 0.02, 0.01], [0.02, -0.02, -0.01]])
    np.savez(tmp_path / "cad.npz", triangles_m=vertices[None, :, :])
    part = {"frame_id": "world", "reference_point": "observed_bounds_center", "object_id": "selected",
            "reference_pose": matrix_pose(np.eye(4)), "CAD_mesh": pin(tmp_path, tmp_path / "cad.npz"),
            "bounds_m": {"minimum": vertices.min(axis=0).tolist(), "maximum": vertices.max(axis=0).tolist()}}
    actual = np.diag([-1.0, -1.0, 1.0, 1.0])
    if fault == "position":
        actual[0, 3] = 0.05
    instances = [{"model_name": "first", "mesh_scale": [1.0] * 3, "model_from_CAD": np.eye(4).tolist()}]
    if fault == "ambiguous":
        instances.append({**instances[0], "model_name": "second"})
    live = {item["model_name"]: {"pose": matrix_pose(actual)} for item in instances}
    if fault == "changed_mesh":
        (tmp_path / "cad.npz").write_bytes(b"changed")

    def match() -> Any:
        return match_instance(instances, live, part, cad_scale=1.0, profile=load_execution_profile(),
                              interaction_root=tmp_path, validation_scope=GAZEBO_OBSERVED_SCOPE)

    if fault:
        with pytest.raises(ValueError, match="changed|exactly one"):
            match()
    else:
        result = match()
        assert result["model_name"] == "first"
        assert result["orientation_difference_rad"] is None
        assert result["bounds_difference_m"] == pytest.approx(0.0)


class _TimedPlanner(_PlanningSession):
    async def check_segment(self, **request: Any) -> dict[str, Any]:
        result = await super().check_segment(**request)
        if result["status"] == "passed":
            names = [name for name in request["joints"]["names"] if name != "gripper_joint"]
            q = [
                request["joints"]["positions"][request["joints"]["names"].index(name)]
                for name in names
            ]
            result["trajectory"] = {
                "joint_names": names,
                "positions": [q, q],
                "velocities": [[0.0] * len(q)] * 2,
                "accelerations": [[0.0] * len(q)] * 2,
                "time_from_start_ns": [0, 1000000000],
            }
        return result


async def _validator(**kwargs: Any) -> dict[str, Any]:
    return await validate_program(**kwargs, session_factory=_TimedPlanner)


def _validated(
    root: Path, *, lateral_offset: float = 0.0,
    cad_origin_offset: tuple[float, float, float] | None = None,
    validation_profile: dict[str, Any] | None = None,
    observation_timestamps: tuple[int, int] | None = None,
    now_ros: int = 1_000_000_000,
    fitting: bool = False,
    monkeypatch: pytest.MonkeyPatch | None = None,
) -> tuple[Any, Any, Any]:
    root.mkdir()
    validation_profile = validation_profile if validation_profile is not None else {**load_refinement_profile(), "validation_scope": GAZEBO_PICK_PLACE_SCOPE}
    scope = read_validation_scope(validation_profile)
    inputs, robot, refs, roles = (
        _fitting_setup(root) if fitting else
        _observed_setup(root) if scope == GAZEBO_OBSERVED_SCOPE else
        _setup(root, cad_origin_offset=cad_origin_offset)
    )
    build_program = _observed_program if scope == GAZEBO_OBSERVED_SCOPE and not fitting else _program
    roles = {role: roles[role] for role in required_validation_roles(scope, inputs.composition_input["target_feature"])}
    if observation_timestamps is not None:
        for role, stamp in zip(("part", "scene"), observation_timestamps, strict=True):
            record = read_pin(root, roles[role])
            record["observation_timestamp_ns"] = stamp
            roles[role] = refs[role] = append_record(root, root / "timestamps", f"{role}.json", record)
    robot.update(measured_at_ros_ns=now_ros, tf_stamps_ns=[now_ros, now_ros])
    robot["joint_state"]["stamp_ns"] = now_ros
    if scope == GAZEBO_PICK_PLACE_SCOPE:
        refs.pop("specification")
    robot["ee_from_tcp"][0][3] = lateral_offset
    configuration = {
        "gripper": {"joint": "gripper_joint", "open": 0.0, "close": 1.0},
        "services": {},
        "attach": {},
    }
    robot["joint_state"]["names"].append("gripper_joint")
    robot["joint_state"]["positions"].append(0.0)
    robot["configuration_sha256"] = fingerprint(configuration)
    robot["model_parameters"] = {
        "robot_description": '<robot name="fixture"><joint name="joint1" type="revolute"><limit lower="-3" upper="3" velocity="1" effort="1"/></joint></robot>',
        "robot_description_planning.joint_limits.joint1.has_acceleration_limits": True,
        "robot_description_planning.joint_limits.joint1.max_acceleration": 1.0,
    }
    robot["model_parameters_sha256"] = fingerprint(robot["model_parameters"])
    robot["policy"]["trajectory_time_scale"] = 1.0
    part = read_pin(root, roles["part"])
    part["cad_context_ref"] = "Gear_Medium.STL"
    if scope == GAZEBO_OBSERVED_SCOPE:
        vertices = np.asarray([[-0.02, -0.02, -0.01], [0.02, 0.02, 0.01], [0.02, -0.02, -0.01]])
        np.savez(root / "observed_cad.npz", triangles_m=vertices[None, :, :])
        part["CAD_mesh"] = pin(root, root / "observed_cad.npz")
    roles["part"] = refs["part"] = append_record(root, root / "evidence", "part_cad.json", part)
    if fitting:
        goal = read_pin(root, refs["goal"])
        goal["part_geometry_ref"] = refs["part"]["ref"]
        goal["source_refs"][0] = refs["part"]
        roles["goal"] = refs["goal"] = append_record(root, root / "evidence", "goal_cad.json", goal)
    model = _MessageProgramRuntime([_program_action([
        (step["primitive_symbol"], step["params"]) for step in build_program(refs, bound=False)
    ])])

    class Robot:
        async def capture_validation_context(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
            return {**deepcopy(robot), "captured_at_ns": time.time_ns()}

        async def capture_execution_context(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
            return {
                **await self.capture_validation_context(),
                "held_part": kwargs["custody"]["held_part"],
            }

        async def execution_configuration(self, *args: Any) -> dict[str, Any]:
            return {
                "configuration": configuration,
                "primitive_catalog": read_pin(root, inputs.context_refs["primitive_catalog"])[
                    "primitive_catalog"
                ],
            }

    with pytest.MonkeyPatch.context() as patch:
        _, _, _, product, _ = _binding_fixture(root, patch, scope, setup=(inputs, robot, refs, roles))
        runtime = PrimitiveRefinementRuntime(
            program_runtime=model, robot_runtime=Robot(), product_runtime=product,
            validator=_validator, profile=validation_profile,
        )
        result = asyncio.run(runtime.compose(root))
    assert result["status"] == "validated_for_declared_scope", result
    if monkeypatch is not None:
        # Execution must read the same explicit fixture target as composition.
        _binding_fixture(root, monkeypatch, scope, setup=(inputs, robot, refs, roles))
    return Robot(), part, model


class _Transport:
    calls: list[Any]

    def __init__(
        self, part: dict[str, Any], *, fail: str | None = None, pause: asyncio.Event | None = None
    ) -> None:
        self.part, self.fail, self.pause = part, fail, pause
        self.calls = []
        self.stop: threading.Event | None = None

    def __call__(self, configuration: Any, profile: Any, stop: threading.Event) -> _Transport:
        self.stop = stop
        return self

    async def __aenter__(self) -> _Transport:
        return self

    async def __aexit__(self, *args: Any) -> None:
        return None

    async def entity_states(self, names: list[str]) -> dict[str, Any]:
        instance = fixture_instances(
            _SHARE, "table_spec2primitives.world", approved_cad_path("Gear_Medium.STL")
        )[0]
        pose = pose_matrix(self.part.get("reference_pose", self.part.get("origin_pose"))) @ np.linalg.inv(
            np.asarray(instance["model_from_CAD"])
        )
        return {name: {"pose": matrix_pose(pose)} for name in names}

    async def feedback(self, *args: Any) -> dict[str, Any]:
        if self.fail == "feedback":
            raise RuntimeError("Measured prefix differs.")
        expected = dict(zip(args[1]["names"], args[1]["positions"], strict=True))
        gripper_calls = [
            call[1] for call in self.calls if isinstance(call, tuple) and call[0] == "gripper"
        ]
        assert expected["gripper_joint"] == (gripper_calls[-1] if gripper_calls else 0.0)
        return {"measured_at_ros_ns": 1000000000}

    async def move(self, trajectory: Any) -> dict[str, Any]:
        self.calls.append("move_cartesian")
        if self.pause is not None:
            self.pause.set()
            while not self.stop.is_set():
                await asyncio.sleep(0.001)
            raise RuntimeError("Trajectory cancellation acknowledged.")
        if self.fail == "move":
            raise RuntimeError("Trajectory rejected.")
        if self.fail == "timeout":
            raise TimeoutError("Trajectory has no terminal result.")
        return {"success": True}

    async def gripper_command(self, position: float) -> dict[str, Any]:
        self.calls.append(("gripper", position))
        return {"success": True, "position": position}

    async def attachment(self, binding: Any, attach: bool) -> dict[str, Any]:
        self.calls.append(("attach" if attach else "detach", binding["model_name"]))
        if self.fail == ("attach" if attach else "detach"):
            raise TimeoutError("Attachment has no acknowledgment.")
        if self.fail == "negative_ack":
            return {"success": False, "attached": False}
        return {"success": True, "attached": attach}


@pytest.mark.parametrize("now_ros", [102_400_000_000, 400_000_000_000])
def test_observed_geometry_reuse_survives_final_validation_and_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, now_ros: int,
) -> None:
    """Keep run_0009's sensor times through composition and simulated transport checks."""
    root = tmp_path / "interaction"
    stamps = (168_674_000_000, 168_522_000_000)
    robot, part, _ = _validated(
        root, validation_profile=load_refinement_profile(), observation_timestamps=stamps, now_ros=now_ros, monkeypatch=monkeypatch,
    )
    program = load_validated_program(root)
    assert program.report["status"] == "passed"
    assert program.report["final_robot_context_ref"]
    original = {path: path.read_bytes() for path in root.rglob("*.json")}

    class Transport(_Transport):
        async def feedback(self, *args: Any) -> dict[str, Any]:
            return {**await super().feedback(*args), "measured_at_ros_ns": now_ros}

    transport = Transport(part)
    executor = PrimitiveExecutionRuntime(
        robot_runtime=robot, validator=_validator, session_factory=transport, share=_SHARE,
    )
    result = asyncio.run(executor.run(root))
    assert result["status"] == "completed", result
    assert result["completed_steps"] == len(program.steps)
    assert transport.calls.count("move_cartesian") == 6
    assert all(path.read_bytes() == content for path, content in original.items())
    for role, stamp in zip(("part", "scene"), stamps, strict=True):
        assert read_pin(root, program.report["evidence_refs"][role])["observation_timestamp_ns"] == stamp


@pytest.mark.parametrize("scope", [GAZEBO_OBSERVED_SCOPE, GAZEBO_PICK_PLACE_SCOPE, VALIDATION_SCOPE])
@pytest.mark.parametrize("now_ros", [102_400_000_000, 400_000_000_000])
def test_execution_scene_age_reuse_is_limited_to_observed_scope(scope: str, now_ros: int) -> None:
    """Execution retains the observation-age and clock checks for other scopes."""
    evidence = {"part": {"observation_timestamp_ns": 168_674_000_000},
                "scene": {"observation_timestamp_ns": 168_522_000_000}}
    original = deepcopy(evidence)
    profile = {**load_refinement_profile(), "validation_scope": scope}
    if scope == GAZEBO_OBSERVED_SCOPE:
        PrimitiveExecutionRuntime._check_scene_age(evidence, now_ros, profile)
    else:
        with pytest.raises(ValueError, match="Observed scene evidence is stale"):
            PrimitiveExecutionRuntime._check_scene_age(evidence, now_ros, profile)
    assert evidence == original


@pytest.mark.parametrize("bore_radius_m", [.00501, .004987318])
def test_calculated_targets_with_no_clearance_cannot_authorize_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bore_radius_m: float,
) -> None:
    setup = _fitting_setup(tmp_path, bore_radius_m=bore_radius_m)
    _, robot, refs, product, _ = _binding_fixture(tmp_path, monkeypatch, setup=setup)
    steps = _program(refs, bound=False)
    steps[0]["params"]["prefer_live_detection"] = False
    model = _MessageProgramRuntime([_program_action([(step["primitive_symbol"], step["params"]) for step in steps])])
    class Robot:
        async def capture_validation_context(self, *args: Any, **kwargs: Any) -> Any:
            return {**deepcopy(robot), "captured_at_ns": time.time_ns()}
    async def validator(**kwargs: Any) -> Any:
        return await validate_program(**kwargs, session_factory=_PlanningSession)
    result = asyncio.run(PrimitiveRefinementRuntime(
        program_runtime=model, robot_runtime=Robot(), product_runtime=product, validator=validator,
    ).compose(tmp_path))
    assert result["status"] == "failed"
    report = read_pin(tmp_path, result["validation_refs"][-1])
    assert [read_pin(tmp_path, ref)["step_index"] for ref in report["calculation_refs"]] == [1, 6]
    with pytest.raises(ValueError, match="validated_for_declared_scope"):
        load_validated_program(tmp_path)
    with pytest.raises(ValueError, match="validated_for_declared_scope"):
        asyncio.run(PrimitiveExecutionRuntime(robot_runtime=SimpleNamespace()).run(tmp_path))
    assert not (tmp_path / "execution").exists()


@pytest.mark.parametrize("lateral_offset,cad_origin_offset", [
    (0.0, None), (0.02, None), (0.02, (0.213, 0.182, 0.070)),
])
@pytest.mark.parametrize("scope", [GAZEBO_PICK_PLACE_SCOPE, GAZEBO_OBSERVED_SCOPE])
def test_whole_program_reuses_saved_authority_and_actual_cad_mapping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, lateral_offset: float, cad_origin_offset: tuple[float, float, float] | None,
    scope: str,
) -> None:
    root = tmp_path / "interaction"
    robot, part, model = _validated(
        root, lateral_offset=lateral_offset, cad_origin_offset=cad_origin_offset,
        validation_profile={**load_refinement_profile(), "validation_scope": scope},
        monkeypatch=monkeypatch,
    )
    original = {path: path.read_bytes() for path in root.rglob("*.json")}
    transport = _Transport(part)

    class Capture:
        def capture(self, directory: Path, identifier: str, *, timeout_sec: float) -> Path:
            directory.mkdir()
            path = directory / "capture.json"
            path.write_text('{"status": "captured", "source": "controlled RGB-D fixture"}')
            return path

    executor = PrimitiveExecutionRuntime(
        robot_runtime=robot,
        validator=_validator,
        session_factory=transport,
        share=_SHARE,
        capture_runtime=Capture(),
    )
    result = asyncio.run(executor.run(root))
    assert result["status"] == "completed", result
    assert result["message"].startswith("Pick-and-place completed")
    program = load_validated_program(root)
    assert program.report["scope"] == scope
    supported = supported_primitive_symbols(program.report["scope"])
    assert {step["primitive_symbol"] for step in program.steps} == supported
    for call in model.calls:
        variants = call["response_format"]["schema"]["properties"]["action"]["anyOf"]
        proposal = next(item for item in variants if item["properties"]["kind"]["enum"] == ["propose"])
        assert set(proposal["properties"]["primitive_steps"]["items"]["properties"]["primitive_symbol"]["enum"]) == supported
    assert set(program.report["evidence_refs"]) == {"part", "scene"}
    assert result["completed_steps"] == 10
    assert transport.calls == [
        "move_cartesian",
        "move_cartesian",
        ("gripper", 1.0),
        ("attach", "gear_medium"),
        "move_cartesian",
        "move_cartesian",
        "move_cartesian",
        ("gripper", 0.0),
        ("detach", "gear_medium"),
        "move_cartesian",
    ]
    assert len(model.calls) == 1, "Execution must not call composition again."
    assert all(path.read_bytes() == data for path, data in original.items())
    assert result["assembly_success"] is None
    assert (root / result["observation_ref"]).is_file()
    view = read_primitive_execution_diagnostic(root)
    assert view["status"] == "completed", view
    assert execution_custody(tmp_path, "xarm6@localhost") == {
        "held_part": None,
        "gripper_state": "open",
    }
    with pytest.raises(ValueError, match="already has an execution attempt"):
        asyncio.run(executor.run(root))


@pytest.mark.parametrize(
    "failure,expected_status,expected_calls",
    [
        ("feedback", "blocked", 0),
        ("move", "failed", 1),
        ("timeout", "unknown", 1),
        ("attach", "unknown", 4),
        ("detach", "unknown", 9),
        ("negative_ack", "failed", 4),
    ],
)
@pytest.mark.parametrize("scope", [GAZEBO_PICK_PLACE_SCOPE, GAZEBO_OBSERVED_SCOPE])
def test_failures_stop_dispatch_and_keep_unknown_custody(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str, expected_status: str, expected_calls: int, scope: str,
) -> None:
    root = tmp_path / "interaction"
    robot, part, _ = _validated(root, validation_profile={**load_refinement_profile(), "validation_scope": scope}, monkeypatch=monkeypatch)
    transport = _Transport(part, fail=failure)
    executor = PrimitiveExecutionRuntime(
        robot_runtime=robot, validator=_validator, session_factory=transport, share=_SHARE
    )
    result = asyncio.run(executor.run(root))
    assert result["status"] == expected_status, result
    assert len(transport.calls) == expected_calls
    if expected_calls <= 1:
        assert result["gripper_state"] is None
    if failure in {"attach", "detach", "negative_ack"}:
        assert result["custody_known"] is False
    if expected_status == "unknown" or failure == "negative_ack":
        with pytest.raises(ValueError, match="uncertain"):
            execution_custody(tmp_path, "xarm6@localhost")
        with pytest.raises(ValueError, match="both robots"):
            assert_execution_available(tmp_path)


def test_duplicate_clicks_join_and_stop_waits_for_terminal_acknowledgment(tmp_path: Path) -> None:
    root = tmp_path / "interaction"
    robot, part, model = _validated(root)

    async def scenario() -> None:
        started = asyncio.Event()
        transport = _Transport(part, pause=started)
        executor = PrimitiveExecutionRuntime(
            robot_runtime=robot, validator=_validator, session_factory=transport, share=_SHARE
        )
        first = asyncio.create_task(executor.run(root))
        await asyncio.wait_for(started.wait(), 10)
        second = asyncio.create_task(executor.run(root))
        await asyncio.sleep(0)
        assert execution_busy()
        assert read_primitive_execution_diagnostic(root)["status"] == "running"
        assert executor.stop(root)
        assert executor.diagnostic(root)["status"] == "stopping"
        a, b = await asyncio.gather(first, second)
        assert a == b
        assert a["status"] == "stopped", a
        assert transport.calls == ["move_cartesian"]
        assert len(model.calls) == 1

    asyncio.run(scenario())
    assert not execution_busy()
    assert len(list((root / "execution").glob("run_*"))) == 1


@pytest.mark.parametrize("record", ["candidate", "validation", "evidence", "catalog", "displayed"])
def test_tampered_or_stale_authority_never_constructs_transport(
    tmp_path: Path, record: str
) -> None:
    root = tmp_path / "interaction"
    robot, part, _ = _validated(root)
    program = load_validated_program(root)
    transport = _Transport(part)
    executor = PrimitiveExecutionRuntime(
        robot_runtime=robot, validator=_validator, session_factory=transport, share=_SHARE
    )
    if record == "displayed":
        with pytest.raises(ValueError, match="displayed program changed"):
            asyncio.run(executor.run(root, candidate_ref="composition/previous/candidate.json"))
        assert transport.calls == []
        return
    reference = {
        "candidate": program.candidate_ref,
        "validation": program.validation_ref,
        "evidence": program.report["evidence_refs"]["part"],
        "catalog": program.inputs.context_refs["primitive_catalog"],
    }[record]
    path = root / reference["ref"]
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError):
        asyncio.run(executor.run(root))
    assert transport.calls == []


@pytest.mark.parametrize("pending", ["gripper", "attachment"])
def test_stop_awaits_outstanding_gripper_and_service_results(tmp_path: Path, pending: str) -> None:
    root = tmp_path / "interaction"
    robot, part, _ = _validated(root)

    async def scenario() -> None:
        started, acknowledged = asyncio.Event(), asyncio.Event()

        class Transport(_Transport):
            async def gripper_command(self, position: float) -> dict[str, Any]:
                result = await super().gripper_command(position)
                if pending == "gripper":
                    started.set()
                    await acknowledged.wait()
                return result

            async def attachment(self, binding: Any, attach: bool) -> dict[str, Any]:
                result = await super().attachment(binding, attach)
                if pending == "attachment":
                    started.set()
                    await acknowledged.wait()
                return result

        transport = Transport(part)
        executor = PrimitiveExecutionRuntime(
            robot_runtime=robot, validator=_validator, session_factory=transport, share=_SHARE
        )
        running = asyncio.create_task(executor.run(root))
        await asyncio.wait_for(started.wait(), 15)
        assert executor.stop(root)
        await asyncio.sleep(0.03)
        assert not running.done(), "Stop must await the outstanding command acknowledgment."
        acknowledged.set()
        result = await running
        assert result["status"] == "stopped"
        assert len(transport.calls) == (3 if pending == "gripper" else 4)
        assert result["custody_known"] is (pending == "attachment")

    asyncio.run(scenario())


def test_saved_assembly_scope_survives_pick_place_default(tmp_path: Path) -> None:
    """An older profile without a scope keeps assembly validation after the default changes."""
    root = tmp_path / "interaction"
    historical_profile = load_refinement_profile()
    historical_profile.pop("validation_scope")
    _validated(root, validation_profile=historical_profile)
    original = {path: path.read_bytes() for path in root.rglob("*.json")}
    program = load_validated_program(root)
    assert load_refinement_profile()["validation_scope"] == GAZEBO_OBSERVED_SCOPE
    assert program.report["scope"] == VALIDATION_SCOPE
    assert "validation_scope" not in program.profile
    assert set(program.report["evidence_refs"]) == {"part", "goal", "scene", "specification"}
    assert all(path.read_bytes() == data for path, data in original.items())


@pytest.mark.parametrize("scope", [VALIDATION_SCOPE, "unknown", None, "different_binding"])
def test_fresh_validation_scope_mismatch_blocks_before_transport(tmp_path: Path, scope: str | None) -> None:
    """A passing report for another scope never authorizes a Gazebo command."""
    root = tmp_path / "interaction"
    robot, part, _ = _validated(root)
    transport = _Transport(part)

    async def mismatched(**kwargs: Any) -> dict[str, Any]:
        result = await _validator(**kwargs)
        assert result["status"] == "passed"
        if scope == "different_binding":
            result["binding_ref"] = {"ref": "different_binding.json", "sha256": "a" * 64}
        else:
            result["scope"] = scope
        return result

    executor = PrimitiveExecutionRuntime(
        robot_runtime=robot, validator=mismatched, session_factory=transport, share=_SHARE,
    )
    result = asyncio.run(executor.run(root))
    assert result["status"] == "blocked", result
    assert result["command_dispatched"] is False
    assert transport.stop is None and transport.calls == []


def test_fresh_robot_mismatch_blocks_before_any_command(tmp_path: Path) -> None:
    root = tmp_path / "interaction"
    robot, part, _ = _validated(root)
    capture = robot.capture_execution_context

    async def changed(*args: Any, **kwargs: Any) -> dict[str, Any]:
        value = await capture(*args, **kwargs)
        value["joint_state"]["positions"][0] = 0.1
        return value

    robot.capture_execution_context = changed
    transport = _Transport(part)
    result = asyncio.run(
        PrimitiveExecutionRuntime(
            robot_runtime=robot, validator=_validator, session_factory=transport, share=_SHARE
        ).run(root)
    )
    assert result["status"] == "blocked", result
    assert transport.calls == []


def test_actual_stl_binding_accounts_for_visual_transform_and_ambiguity() -> None:
    profile = load_execution_profile()
    instances = fixture_instances(
        _SHARE, profile["world_file"], approved_cad_path("Gear_Medium.STL")
    )
    assert len(instances) == 1
    assert instances[0]["model_name"] == "gear_medium"
    live_pose = {"x": 0.4, "y": -0.3, "z": 1.1, "qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0}
    actual = pose_matrix(live_pose) @ np.asarray(instances[0]["model_from_CAD"])
    part = {
        "object_id": "observed_part",
        "frame_id": "world",
        "reference_point": "CAD_origin",
        "origin_pose": matrix_pose(actual),
    }
    binding = match_instance(
        instances, {"gear_medium": {"pose": live_pose}}, part, cad_scale=0.001, profile=profile
    )
    assert binding["position_difference_m"] < 1e-12
    with pytest.raises(ValueError, match="0 Gazebo instances"):
        match_instance(
            instances,
            {"gear_medium": {"pose": live_pose}},
            {**part, "origin_pose": live_pose},
            cad_scale=0.001,
            profile=profile,
        )
    with pytest.raises(ValueError, match="2 Gazebo instances"):
        match_instance(
            [*instances, {**instances[0], "model_name": "gear_medium_copy"}],
            {name: {"pose": live_pose} for name in ("gear_medium", "gear_medium_copy")},
            part,
            cad_scale=0.001,
            profile=profile,
        )
    offset_pose = {**live_pose, "x": live_pose["x"] + 0.009}
    assert match_instance(
        instances, {"gear_medium": {"pose": offset_pose}}, part, cad_scale=0.001, profile=profile
    )["position_difference_m"] == pytest.approx(0.009)
    offset_pose["x"] = live_pose["x"] + 0.011
    with pytest.raises(ValueError, match="0 Gazebo instances"):
        match_instance(
            instances,
            {"gear_medium": {"pose": offset_pose}},
            part,
            cad_scale=0.001,
            profile=profile,
        )
    with pytest.raises(ValueError, match="units"):
        match_instance(
            instances, {"gear_medium": {"pose": live_pose}}, part, cad_scale=1.0, profile=profile
        )


def test_execution_records_cannot_become_ra_or_pa_evidence(tmp_path: Path) -> None:
    inputs, _, _, _ = _setup(tmp_path)
    reference = append_record(
        tmp_path,
        tmp_path / "execution/run_1",
        "binding.json",
        {"record_type": "GazeboInstanceBinding", "model_name": "gear_medium"},
    )
    extended = replace(
        inputs, record_hashes={**inputs.record_hashes, reference["ref"]: reference["sha256"]}
    )
    with pytest.raises(ValueError, match="excluded"):
        _evidence_value(extended, reference["ref"], "")
    with pytest.raises(ValueError, match="execution records"):
        verify_evidence_tree(tmp_path, reference)
    producer = AssemblyGeometryProducer(
        tmp_path, tmp_path / "geometry", {reference["ref"]: reference["sha256"]}
    )
    with pytest.raises(ValueError, match="execution records"):
        producer.read(reference["ref"])
    nested = append_record(
        tmp_path,
        tmp_path / "evidence",
        "nested.json",
        {"record_type": "AssemblyGeometryEvidence", "source_refs": [reference]},
    )
    with pytest.raises(ValueError, match="execution records"):
        verify_evidence_tree(tmp_path, nested)


def test_timing_scales_without_replanning_and_rejects_unsafe_speed(tmp_path: Path) -> None:
    root = tmp_path / "interaction"
    _, _, _ = _validated(root)
    robot = load_validated_program(root).robot
    trajectory = {
        "joint_names": ["joint1"],
        "positions": [[0.0], [0.5]],
        "velocities": [[0.5], [0.5]],
        "accelerations": [[0.0], [0.0]],
        "time_from_start_ns": [0, 1000000000],
    }
    slow = prepare_trajectory(trajectory, robot, 2.0)
    assert slow["positions"] == trajectory["positions"]
    assert slow["time_from_start_ns"] == [0, 2000000000]
    assert slow["velocities"] == [[0.25], [0.25]]
    with pytest.raises(ValueError, match="velocity"):
        prepare_trajectory(trajectory, robot, 0.1)
    with pytest.raises(ValueError, match="time stamps"):
        prepare_trajectory({**trajectory, "time_from_start_ns": [0, 0]}, robot, 1.0)
    robot["model_parameters"].update(
        {
            "robot_description_planning.joint_limits.joint1.has_position_limits": True,
            "robot_description_planning.joint_limits.joint1.min_position": -0.2,
            "robot_description_planning.joint_limits.joint1.max_position": 0.2,
        }
    )
    with pytest.raises(ValueError, match="position"):
        prepare_trajectory(trajectory, robot, 2.0)


def test_gripper_target_outside_configured_travel_is_rejected_before_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(
        sys.modules,
        "trajectory_msgs.msg",
        SimpleNamespace(
            JointTrajectory=object,
            JointTrajectoryPoint=object,
        ),
    )
    session = GazeboExecutionSession(
        {"gripper": {"open": 0.0, "close": 1.0, "move_time_sec": 1.0}},
        load_execution_profile(),
        threading.Event(),
    )
    with pytest.raises(ValueError, match="Invalid configured gripper command"):
        session._gripper_command(1.1)


def test_transport_stop_cancels_the_goal_and_waits_for_its_terminal_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise the real dispatch/cancel code against an action-server double."""

    class Future:
        def __init__(self, value: Any, done: bool = True) -> None:
            self.value, self.finished = value, done

        def done(self) -> bool:
            return self.finished

        def result(self) -> Any:
            return self.value

    class Point:
        def __init__(self, **kwargs: Any) -> None:
            self.__dict__.update(kwargs)
            self.time_from_start = SimpleNamespace(sec=0, nanosec=0)

    def make_goal() -> Any:
        return SimpleNamespace(
            trajectory=SimpleNamespace(joint_trajectory=SimpleNamespace(joint_names=[], points=[]))
        )

    monkeypatch.setitem(
        sys.modules,
        "moveit_msgs.action",
        SimpleNamespace(ExecuteTrajectory=SimpleNamespace(Goal=make_goal)),
    )
    monkeypatch.setitem(
        sys.modules, "trajectory_msgs.msg", SimpleNamespace(JointTrajectoryPoint=Point)
    )
    stop = threading.Event()
    terminal = Future(
        SimpleNamespace(status=5, result=SimpleNamespace(error_code=SimpleNamespace(val=-7))),
        done=False,
    )
    calls = []

    class Goal:
        accepted = True

        def get_result_async(self) -> Future:
            return terminal

        def cancel_goal_async(self) -> Future:
            calls.append("cancel")
            return Future(SimpleNamespace(goals_canceling=[1]))

    def spin(**kwargs: Any) -> None:
        if not stop.is_set():
            stop.set()
        elif calls == ["cancel"]:
            calls.append("terminal acknowledgment")
            terminal.finished = True

    session = GazeboExecutionSession({}, load_execution_profile(), stop)
    session.executor = SimpleNamespace(spin_once=spin)
    session.arm = SimpleNamespace(send_goal_async=lambda goal: Future(Goal()))
    trajectory = {
        "joint_names": ["joint1"],
        "positions": [[0.0], [0.2]],
        "velocities": [[0.0], [0.0]],
        "accelerations": [[0.0], [0.0]],
        "time_from_start_ns": [0, 1000000000],
    }
    with pytest.raises(RuntimeError, match="stopped or failed"):
        session._move(trajectory)
    assert calls == ["cancel", "terminal acknowledgment"]
    assert session.goal is None


def test_interrupted_execution_does_not_resume_or_assume_empty_custody(tmp_path: Path) -> None:
    root = tmp_path / "interaction"
    directory = root / "execution/run_1"
    append_record(
        root,
        directory,
        "request.json",
        {
            "record_type": "PrimitiveExecutionRequest",
            "resource_jid": "xarm6@localhost",
            "created_at_ns": 1,
            "total_steps": 1,
            "candidate_ref": {"ref": "candidate.json", "sha256": "a" * 64},
        },
    )
    assert read_primitive_execution_diagnostic(root)["status"] == "interrupted"
    with pytest.raises(ValueError, match="interrupted"):
        assert_execution_available(tmp_path)
    with pytest.raises(ValueError, match="no final custody"):
        execution_custody(tmp_path, "xarm6@localhost")


@pytest.mark.parametrize("replacement", ["binding", "proposal", "report", "bound_steps"])
def test_execution_rejects_replaced_program_binding_proposal_or_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, replacement: str) -> None:
    from cais_spade_llm.spec2primitives.agents.ra.primitive_composition import _write_record
    root = tmp_path / "interaction"
    robot, part, _ = _validated(root, validation_profile=load_refinement_profile(), monkeypatch=monkeypatch)
    program = load_validated_program(root)
    result = read_pin(root, program.result_ref)
    directory = (root / program.result_ref["ref"]).parent
    if replacement == "binding":
        payload = read_pin(root, program.binding_ref)
        result["binding_refs"].append(append_record(root, directory, "different_binding.json", payload))
    elif replacement == "proposal":
        payload = read_pin(root, program.candidate_ref)
        result["candidate_refs"][-1] = append_record(root, directory, "different_proposal.json", payload)
    elif replacement == "report":
        payload = read_pin(root, program.validation_ref)
        payload["different_report"] = True
        result["validation_refs"][-1] = append_record(root, directory, "different_report.json", payload)
    else:
        payload = read_pin(root, program.binding_ref)
        payload["primitive_steps"][1]["params"]["x"] = 1.2
        result["binding_refs"].append(append_record(root, directory, "different_steps.json", payload))
    (root / program.result_ref["ref"]).unlink()
    _write_record(root / program.result_ref["ref"], {key: value for key, value in result.items() if key != "fingerprint"})
    transport = _Transport(part)
    executor = PrimitiveExecutionRuntime(robot_runtime=robot, validator=_validator, session_factory=transport, share=_SHARE)
    with pytest.raises(ValueError):
        asyncio.run(executor.run(root))
    assert transport.calls == [] and not (root / "execution").exists()


def test_validation_and_run_selection_require_the_same_binding(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "interaction"
    robot, part, _ = _validated(root, validation_profile=load_refinement_profile(), monkeypatch=monkeypatch)
    program = load_validated_program(root)
    steps = deepcopy(program.steps)
    steps[1]["params"]["z"] = 0.99
    with pytest.raises(ValueError, match="pinned primitive binding"):
        asyncio.run(_validator(inputs=program.inputs, steps=steps, robot=program.robot,
            evidence=program.report["evidence_refs"], directory=root / "should_not_validate", profile=program.profile, cache={}))
    executor = PrimitiveExecutionRuntime(robot_runtime=robot, validator=_validator, session_factory=_Transport(part), share=_SHARE)
    with pytest.raises(ValueError, match="binding"):
        asyncio.run(executor.run(root, binding_ref="composition/different_binding.json"))
    assert not (root / "execution").exists()


@pytest.mark.parametrize("above_seat, unsupported", [(False, None), (True, None), (False, "threading"), (False, "press fit"), (False, "snap fit"), (False, "noncircular")])
def test_fitting_execution_checks_fresh_pose_before_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, above_seat: bool, unsupported: str | None,
) -> None:
    import json
    from cais_spade_llm.spec2primitives.agents.ra import primitive_composition, refinement, program_execution

    root = tmp_path / "fitting_execution"
    robot, part, model = _validated(root, fitting=True,
        validation_profile={**load_refinement_profile(), "validation_scope": GAZEBO_OBSERVED_SCOPE}, now_ros=102_400_000_000)
    target = json.loads(model.calls[-1]["prompt"].split("COMPOSITION_INPUT\n", 1)[1])["target_feature"]
    original = primitive_composition._load_inputs

    def load(path: Path) -> Any:
        inputs = original(path)
        return replace(inputs, composition_input={**inputs.composition_input, "target_feature": target})

    for module in (primitive_composition, refinement, program_execution):
        monkeypatch.setattr(module, "_load_inputs", load)

    class Transport(_Transport):
        async def feedback(self, *args: Any) -> Any:
            result = await super().feedback(*args)
            pose = deepcopy(args[2])
            if above_seat and self.calls.count("move_cartesian") == 5:
                pose["z"] += .002
            return {**result, "ee_pose": pose, "measured_at_ros_ns": 102_400_000_000}

    transport = Transport(part)
    async def validator(**kwargs: Any) -> Any:
        if unsupported:
            evidence = deepcopy(kwargs["evidence"])
            if unsupported == "noncircular":
                goal = {**read_pin(root, evidence["goal"]), "status": "unsupported",
                        "reason": "The approved CAD has no supported pair of coaxial circular end faces.", "product_geometry": {}}
                evidence["goal"] = append_record(root, root / "checked_fixture", "goal.json", goal)
            else:
                evidence["specification"] = append_record(root, root / "checked_fixture", "specification.json", {
                    "record_type": "AssemblyValidationSpecification", "status": "accepted",
                    "family": "vertical_gear_assembly" if unsupported == "threading" else unsupported,
                    "requires_threading": unsupported == "threading",
                })
            kwargs["evidence"] = evidence
        return await _validator(**kwargs)
    runtime = PrimitiveExecutionRuntime(robot_runtime=robot, session_factory=transport, validator=validator, share=_SHARE)
    result = asyncio.run(runtime.run(root))
    if unsupported:
        assert result["status"] == "blocked" and not result["command_dispatched"], result
        assert transport.calls == [] and result["assembly_success"] is None
        return
    assert (result["status"] == "completed") is (not above_seat), result
    detached = [call for call in transport.calls if isinstance(call, tuple) and call[0] == "detach"]
    assert bool(detached) is (not above_seat)
    if above_seat:
        assert "shaft fitting" in result["message"] or "shaft fitting" in result.get("reason", ""), result
        assert transport.calls.count("move_cartesian") == 5
