from __future__ import annotations

"""Test persisted execution authority and command ordering without live motion."""

import asyncio
import threading
import time
from contextlib import asynccontextmanager
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
    GripperCommandError,
    gripper_action_name,
    fixture_instances,
    load_execution_profile,
    match_instance,
    prepare_trajectory,
)
from cais_spade_llm.spec2primitives.adapters.robot_validation_context import (
    MeasuredRobotContextRuntime,
    matrix_pose,
    pose_matrix,
)
from cais_spade_llm.spec2primitives.adapters import in_process_robot_agent
from cais_spade_llm.spec2primitives.adapters.in_process_robot_agent import InProcessRobotAgentCompositionRuntime
from cais_spade_llm.spec2primitives.agents.ra import RAContextHandoffError
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
    GAZEBO_LINK_ATTACHER_SCOPE, GAZEBO_OBSERVED_SCOPE, GAZEBO_PICK_PLACE_SCOPE, VALIDATION_SCOPE, is_observed_scope, read_validation_scope, required_validation_roles,
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
    _LiveRobotAgent,
    _LiveRobotAgentHost,
    _declared_geometry_catalog,
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
            # This fixture's selected arm has one joint; peer and gripper joints
            # remain in the full validation state without becoming arm commands.
            names = [name for name in request["joints"]["names"] if name == "joint1"]
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
    bore_radius_m: float = .00549,
    launch_parameters_path: str | None = None,
    initial_joint_positions: dict[str, float] | None = None,
    monkeypatch: pytest.MonkeyPatch | None = None,
) -> tuple[Any, Any, Any]:
    root.mkdir()
    validation_profile = validation_profile if validation_profile is not None else {**load_refinement_profile(), "validation_scope": GAZEBO_PICK_PLACE_SCOPE}
    scope = read_validation_scope(validation_profile)
    inputs, robot, refs, roles = (
        _fitting_setup(root, bore_radius_m=bore_radius_m) if fitting else
        _observed_setup(root) if is_observed_scope(scope) else
        _setup(root, cad_origin_offset=cad_origin_offset)
    )
    build_program = _observed_program if is_observed_scope(scope) and not fitting else _program
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
    for name, position in (initial_joint_positions or {}).items():
        if name in robot["joint_state"]["names"]:
            robot["joint_state"]["positions"][robot["joint_state"]["names"].index(name)] = position
        else:
            robot["joint_state"]["names"].append(name)
            robot["joint_state"]["positions"].append(position)
    robot["configuration_sha256"] = fingerprint(configuration)
    robot["model_parameters"] = {
        "robot_description": '<robot name="fixture"><joint name="joint1" type="revolute"><limit lower="-3" upper="3" velocity="1" effort="1"/></joint></robot>',
        "robot_description_planning.joint_limits.joint1.has_acceleration_limits": True,
        "robot_description_planning.joint_limits.joint1.max_acceleration": 1.0,
    }
    if launch_parameters_path is not None:
        model_parameters = robot["model_parameters"]
        model_parameters["use_sim_time"] = True
        model_parameters["robot_description"] = model_parameters["robot_description"].replace(
            "</robot>", '<gazebo><plugin name="gazebo_ros2_control" filename="libgazebo_ros2_control.so">'
            f'<parameters>{launch_parameters_path}</parameters></plugin></gazebo></robot>',
        )
    robot["model_parameters_sha256"] = fingerprint(robot["model_parameters"])
    robot["policy"]["trajectory_time_scale"] = 1.0
    part = read_pin(root, roles["part"])
    part["cad_context_ref"] = "Gear_Medium.STL"
    if is_observed_scope(scope):
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

        async def execution_configuration(self, *args: Any, start_if_needed: bool = False) -> dict[str, Any]:
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
        self, part: dict[str, Any], *, fail: str | None = None, pause: asyncio.Event | None = None,
        initial_gripper_position: float = 0.0,
    ) -> None:
        self.part, self.fail, self.pause = part, fail, pause
        self.initial_gripper_position = initial_gripper_position
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
        assert expected["gripper_joint"] == (gripper_calls[-1] if gripper_calls else self.initial_gripper_position)
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


@pytest.mark.parametrize("blocker", [None, "readiness", "hardware", "agent_stopped"])
def test_restored_program_starts_selected_agent_once_before_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, blocker: str | None,
) -> None:
    """Start the owned RA for a restored program, preserving retries, ordering and replay guards."""
    root = tmp_path / "interaction"
    robot, part, model = _validated(root, monkeypatch=monkeypatch)
    program = load_validated_program(root)
    original = {path: path.read_bytes() for path in root.rglob("*.json")}
    configuration = asyncio.run(robot.execution_configuration())["configuration"]
    selected = _LiveRobotAgent(primitive_catalog=_declared_geometry_catalog(), robot_state={"held_part": None})
    selected.context_only = True
    selected.controller_config = configuration
    host = _LiveRobotAgentHost([], system_running=False, gazebo_state="running", startup_agent=selected)
    hardware = {"overall": "stopped"}
    host.hardware_stack_status = lambda name: hardware
    adapter = InProcessRobotAgentCompositionRuntime(host, contexts_root=tmp_path)
    transport = _Transport(part)
    move = transport.move

    async def acknowledged_move(trajectory: Any) -> dict[str, Any]:
        result = await move(trajectory)
        if blocker == "agent_stopped":
            selected.alive = False
        return result

    @asynccontextmanager
    async def measured_context(self: Any, **kwargs: Any) -> Any:
        assert kwargs["resource_jid"] == program.inputs.assignment.selected_resource_jid
        assert kwargs["configuration"] == configuration
        yield await robot.capture_validation_context()

    monkeypatch.setattr(transport, "move", acknowledged_move)
    monkeypatch.setattr(MeasuredRobotContextRuntime, "validation_context", measured_context)
    monkeypatch.setattr(in_process_robot_agent, "_ROBOT_AGENT_STARTUP_POLL_SECONDS", 0.001)
    if blocker == "readiness":
        monkeypatch.setattr(in_process_robot_agent, "_ROBOT_AGENT_STARTUP_TIMEOUT_SECONDS", 0.02)
    executor = PrimitiveExecutionRuntime(
        robot_runtime=adapter, validator=_validator, session_factory=transport, share=_SHARE,
    )

    async def scenario() -> dict[str, Any]:
        nonlocal blocker
        loop = asyncio.get_running_loop()
        entered = asyncio.Event()
        release = threading.Event()
        probes = []

        def readiness(force: bool = False) -> tuple[bool, str]:
            probes.append(force)
            if len(probes) == 1:
                loop.call_soon_threadsafe(entered.set)
                assert release.wait(2), "The event loop did not release the readiness probe."
            if blocker == "hardware":
                hardware["overall"] = "running"
            return (False, "Waiting for core services: /compute_cartesian_path") if blocker == "readiness" else (True, "")

        monkeypatch.setattr(host, "simulation_start_ready", readiness)
        first = asyncio.create_task(executor.run(root))
        second = None
        try:
            await asyncio.wait_for(entered.wait(), 5)
            second = asyncio.create_task(executor.run(root))
            await asyncio.sleep(0)
            assert execution_busy()
            assert host.start_calls == 0
            assert transport.calls == []
            release.set()
            a, b = await asyncio.wait_for(asyncio.gather(first, second, return_exceptions=True), 10)
            if blocker in {"readiness", "hardware"}:
                assert isinstance(a, RAContextHandoffError), a
                assert isinstance(b, RAContextHandoffError), b
                assert str(a) == str(b)
                assert ("/compute_cartesian_path" if blocker == "readiness" else "Hardware stack") in str(a)
                assert host.start_calls == 0
                assert transport.calls == []
                assert not (root / "execution").exists()
                blocker = None
                hardware["overall"] = "stopped"
                a = await executor.run(root)
            else:
                assert a == b
            assert host.start_calls == 1
            assert host.full_system_start_calls == 0
            assert host.resource_agents == []
            assert host._spec2primitives_robot_agent is selected
            assert all(probes)
            with pytest.raises(ValueError, match="already has an execution attempt"):
                await executor.run(root)
            assert host.start_calls == 1
            return a
        finally:
            release.set()
            await asyncio.gather(first, *([second] if second is not None else []), return_exceptions=True)

    result = asyncio.run(scenario())
    if blocker == "agent_stopped":
        assert result["status"] == "failed", result
        assert "RobotAgent" in result["message"] and "is not running" in result["message"]
        assert result["command_dispatched"] is True
        assert transport.calls == ["move_cartesian"]
    else:
        assert result["status"] == "completed", result
        assert result["completed_steps"] == len(program.steps)
        assert transport.calls == [
            "move_cartesian", "move_cartesian", ("gripper", 1.0), ("attach", "gear_medium"),
            "move_cartesian", "move_cartesian", "move_cartesian", ("gripper", 0.0),
            ("detach", "gear_medium"), "move_cartesian",
        ]
    assert len(model.calls) == 1
    assert all(path.read_bytes() == data for path, data in original.items())
    assert len(list((root / "execution").glob("run_*"))) == 1


@pytest.mark.parametrize("validation_passes", [True, False])
def test_saved_program_revalidates_current_gripper_and_peer_joints_before_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, validation_passes: bool,
) -> None:
    """A changed start state reaches fresh validation, which still controls all dispatch."""
    root = tmp_path / "interaction"
    before = {
        "gripper_joint": 0.40933734448149295,
        "ur5e_wrist_1_joint": -1.6023756639112603,
        "ur5e_elbow_joint": 1.6267104080390862,
    }
    current = {
        "gripper_joint": 0.42478389357529167,
        "ur5e_wrist_1_joint": -1.5995845939332678,
        "ur5e_elbow_joint": 1.623759949144783,
    }
    robot, part, model = _validated(root, initial_joint_positions=before, monkeypatch=monkeypatch)
    program = load_validated_program(root)
    original = {path: path.read_bytes() for path in root.rglob("*.json")}
    capture = robot.capture_validation_context
    validation_inputs = []
    planning_inputs = []
    transport = _Transport(part, initial_gripper_position=current["gripper_joint"])

    async def current_capture(*args: Any, **kwargs: Any) -> dict[str, Any]:
        record = await capture(*args, **kwargs)
        joints = record["joint_state"]
        for name, position in current.items():
            joints["positions"][joints["names"].index(name)] = position
        return record

    class Planner(_TimedPlanner):
        async def check_segment(self, **request: Any) -> dict[str, Any]:
            assert transport.calls == [], "All segments must be validated before the first command."
            planning_inputs.append(deepcopy(request["joints"]))
            if not validation_passes:
                return {"status": "failed", "message": "Current state is in collision."}
            return await super().check_segment(**request)

    async def validator(**kwargs: Any) -> dict[str, Any]:
        validation_inputs.append(deepcopy(kwargs["robot"]["joint_state"]))
        assert kwargs["steps"] == program.steps
        assert kwargs["cache"] == {}
        return await validate_program(**kwargs, session_factory=Planner)

    monkeypatch.setattr(robot, "capture_validation_context", current_capture)
    executor = PrimitiveExecutionRuntime(
        robot_runtime=robot, validator=validator, session_factory=transport, share=_SHARE,
    )
    result = asyncio.run(executor.run(root))
    assert len(validation_inputs) == 1
    assert planning_inputs[0] == validation_inputs[0]
    measured = dict(zip(validation_inputs[0]["names"], validation_inputs[0]["positions"], strict=True))
    assert {name: measured[name] for name in current} == current
    view = read_primitive_execution_diagnostic(root)
    assert any("Revalidating the unchanged saved program" in event["message"] for event in view["events"])
    if validation_passes:
        assert result["status"] == "completed", result
        assert result["completed_steps"] == len(program.steps)
        assert transport.calls == [
            "move_cartesian", "move_cartesian", ("gripper", 1.0), ("attach", "gear_medium"),
            "move_cartesian", "move_cartesian", "move_cartesian", ("gripper", 0.0),
            ("detach", "gear_medium"), "move_cartesian",
        ]
        with pytest.raises(ValueError, match="already has an execution attempt"):
            asyncio.run(executor.run(root))
    else:
        assert result["status"] == "blocked", result
        assert "Fresh validation did not pass" in result["message"]
        assert result["command_dispatched"] is False
        assert transport.stop is None and transport.calls == []
    assert len(model.calls) == 1
    assert all(path.read_bytes() == data for path, data in original.items())


@pytest.mark.parametrize("model_changed", [False, True])
def test_execution_revalidates_saved_program_after_gazebo_launch_filename_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, model_changed: bool,
) -> None:
    """Retain saved authority and command ordering across a harmless simulation restart."""
    root = tmp_path / "interaction"
    robot, part, model = _validated(
        root, launch_parameters_path="/tmp/launch_params_before", monkeypatch=monkeypatch,
    )
    program = load_validated_program(root)
    original = {path: path.read_bytes() for path in root.rglob("*.json")}
    capture = robot.capture_validation_context
    fresh_validations = []

    async def current_capture(*args: Any, **kwargs: Any) -> dict[str, Any]:
        record = await capture(*args, **kwargs)
        parameters = record["model_parameters"]
        parameters["robot_description"] = parameters["robot_description"].replace(
            "/tmp/launch_params_before", "/tmp/launch_params_after",
        )
        if model_changed:
            parameters["robot_description"] = parameters["robot_description"].replace('upper="3"', 'upper="2"')
        record["model_parameters_sha256"] = fingerprint(parameters)
        return record

    async def validator(**kwargs: Any) -> dict[str, Any]:
        fresh_validations.append(kwargs["robot"]["model_parameters"]["robot_description"])
        return await _validator(**kwargs)

    monkeypatch.setattr(robot, "capture_validation_context", current_capture)
    transport = _Transport(part)
    executor = PrimitiveExecutionRuntime(
        robot_runtime=robot, validator=validator, session_factory=transport, share=_SHARE,
    )
    result = asyncio.run(executor.run(root))
    if model_changed:
        assert result["status"] == "blocked", result
        assert result["command_dispatched"] is False
        assert result["completed_steps"] == 0
        assert fresh_validations == transport.calls == []
        assert execution_custody(tmp_path, "xarm6@localhost") == {"held_part": None, "gripper_state": None}
        event_path = root / result["last_event_ref"]["ref"]
        event_bytes = event_path.read_bytes()
        try:
            event_path.write_bytes(b"{}")
            with pytest.raises(ValueError, match="Execution records cannot be verified"):
                execution_custody(tmp_path, "xarm6@localhost")
        finally:
            event_path.write_bytes(event_bytes)
        original.update({path: path.read_bytes() for path in (root / "execution").rglob("*.json")})
        model_changed = False
        result = asyncio.run(executor.run(root))
    assert result["status"] == "completed", result
    assert result["completed_steps"] == len(program.steps)
    assert len(fresh_validations) == 1
    assert "/tmp/launch_params_after" in fresh_validations[0]
    assert transport.calls == [
        "move_cartesian", "move_cartesian", ("gripper", 1.0), ("attach", "gear_medium"),
        "move_cartesian", "move_cartesian", "move_cartesian", ("gripper", 0.0),
        ("detach", "gear_medium"), "move_cartesian",
    ]
    with pytest.raises(ValueError, match="already has an execution attempt"):
        asyncio.run(executor.run(root))
    assert len(model.calls) == 1
    assert all(path.read_bytes() == data for path, data in original.items())


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
        # A terminal gripper acknowledgment before attachment preserves prior custody.
        assert result["custody_known"] is True

    asyncio.run(scenario())


def test_saved_assembly_scope_survives_pick_place_default(tmp_path: Path) -> None:
    """An older profile without a scope keeps assembly validation after the default changes."""
    root = tmp_path / "interaction"
    historical_profile = load_refinement_profile()
    historical_profile.pop("validation_scope")
    _validated(root, validation_profile=historical_profile)
    original = {path: path.read_bytes() for path in root.rglob("*.json")}
    program = load_validated_program(root)
    assert load_refinement_profile()["validation_scope"] == GAZEBO_LINK_ATTACHER_SCOPE
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


@pytest.mark.parametrize("mismatch", [
    "during_preparation", "configuration_sha256", "model_parameters_sha256",
    "ee_link", "ee_from_tcp", "held_part", "joint_state.names",
])
def test_fresh_robot_mismatch_blocks_before_any_command(tmp_path: Path, mismatch: str) -> None:
    """Require compatible authority and stable current state throughout fresh validation."""
    root = tmp_path / "interaction"
    robot, part, _ = _validated(root)
    capture = robot.capture_execution_context
    captures = []
    validations = []

    async def changed(*args: Any, **kwargs: Any) -> dict[str, Any]:
        value = await capture(*args, **kwargs)
        captures.append(value)
        if mismatch == "during_preparation" and len(captures) > 1:
            value["joint_state"]["positions"][0] = 0.1
        elif mismatch == "ee_from_tcp":
            value["ee_from_tcp"][0][3] += 0.01
        elif mismatch == "joint_state.names":
            value["joint_state"]["names"][0] = "different_joint"
        elif mismatch not in {"during_preparation", "ee_from_tcp", "joint_state.names"}:
            value[mismatch] = "changed"
        return value

    async def validator(**kwargs: Any) -> dict[str, Any]:
        validations.append(kwargs["robot"])
        return await _validator(**kwargs)

    robot.capture_execution_context = changed
    transport = _Transport(part)
    result = asyncio.run(
        PrimitiveExecutionRuntime(
            robot_runtime=robot, validator=validator, session_factory=transport, share=_SHARE
        ).run(root)
    )
    assert result["status"] == "blocked", result
    assert result["command_dispatched"] is False
    assert len(validations) == (1 if mismatch == "during_preparation" else 0)
    expected_message = {
        "during_preparation": "joint_state.positions['joint1']",
        "configuration_sha256": "Live RobotAgent configuration differs",
    }.get(mismatch, mismatch)
    assert expected_message in result["message"]
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


@pytest.mark.parametrize("clock_samples,stamp_ns,stop_during_wait,error", [
    pytest.param([10_000_000_000], 9_900_000_000, False, None, id="already_fresh"),
    pytest.param([10_000_000_000, 10_020_000_000, 10_100_000_000], 10_035_000_000, False, None, id="clock_lags_reply"),
    pytest.param([0, 10_100_000_000], 10_035_000_000, False, None, id="clock_startup"),
    pytest.param([10_000_000_000], 10_035_000_000, False, "clock did not catch up", id="clock_timeout"),
    pytest.param([10_000_000_000], 1_700_000_000_000_000_000, False, "clock did not catch up", id="wrong_clock_domain"),
    pytest.param([13_000_000_000], 10_000_000_000, False, "feedback is stale", id="old_feedback"),
    pytest.param([10_000_000_000, 13_000_000_000], 10_035_000_000, False, "feedback is stale", id="old_after_clock_update"),
    pytest.param([0], 0, False, "no valid timestamp", id="missing_timestamp"),
    pytest.param([10_000_000_000, 0], 10_035_000_000, False, "clock moved backwards", id="clock_reset"),
    pytest.param([10_000_000_000, 10_100_000_000], 10_035_000_000, True, "Execution stopped", id="stop_during_wait"),
])
def test_entity_feedback_waits_for_clock_without_repeating_requests(
    monkeypatch: pytest.MonkeyPatch, clock_samples: list[int], stamp_ns: int,
    stop_during_wait: bool, error: str | None,
) -> None:
    """Accept a current service reply only after bounded, monotonic clock synchronization."""
    from cais_spade_llm.spec2primitives.adapters import gazebo_execution

    model_list = SimpleNamespace(Request=SimpleNamespace)
    entity_state = SimpleNamespace(Request=SimpleNamespace)
    monkeypatch.setitem(sys.modules, "gazebo_msgs.srv", SimpleNamespace(
        GetModelList=model_list, GetEntityState=entity_state,
    ))
    clocks = list(clock_samples)
    now_ns, wall_time, spin_count = clocks.pop(0), 0.0, 0
    monkeypatch.setattr(gazebo_execution, "time", SimpleNamespace(monotonic=lambda: wall_time))
    pose = SimpleNamespace(
        position=SimpleNamespace(x=0.4, y=-0.3, z=1.1),
        orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
    )
    seconds, nanoseconds = divmod(stamp_ns, 1_000_000_000)
    reply = SimpleNamespace(
        header=SimpleNamespace(stamp=SimpleNamespace(sec=seconds, nanosec=nanoseconds)),
        state=SimpleNamespace(pose=pose),
    )
    calls = []

    def service(kind: Any, name: str, request: Any) -> Any:
        calls.append(name)
        if kind is model_list:
            return SimpleNamespace(model_names=["selected_instance"])
        assert kind is entity_state
        assert (request.name, request.reference_frame) == ("selected_instance", "world")
        return reply

    stop = threading.Event()
    session = GazeboExecutionSession(
        {"services": {"get_entity_state": "/configured_entity_state"}},
        {**load_execution_profile(), "state_max_age_sec": 2.0, "service_timeout_sec": 0.2},
        stop,
    )
    session._service = service
    session.node = SimpleNamespace(get_clock=lambda: SimpleNamespace(
        now=lambda: SimpleNamespace(nanoseconds=now_ns),
    ))

    def spin(*, timeout_sec: float) -> None:
        nonlocal now_ns, wall_time, spin_count
        assert timeout_sec > 0
        wall_time += timeout_sec
        spin_count += 1
        if clocks:
            now_ns = clocks.pop(0)
        if stop_during_wait:
            stop.set()

    session.executor = SimpleNamespace(spin_once=spin)
    if error is not None:
        with pytest.raises(RuntimeError, match=error):
            session._entity_states(["selected_instance"])
    else:
        states = session._entity_states(["selected_instance"])
        assert states == {"selected_instance": {
            "pose": {"x": 0.4, "y": -0.3, "z": 1.1, "qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0},
            "stamp_ns": stamp_ns,
        }}
        assert 0 <= now_ns - stamp_ns <= 2_000_000_000
        assert spin_count == len(clock_samples) - 1
    assert calls == [session.profile["model_list_service"], "/configured_entity_state"]
    assert wall_time <= session.profile["service_timeout_sec"] + 1e-9


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


@pytest.mark.parametrize('rate,fault', [
    (1.0, None), (0.43, None), (0.0, 'paused'), (1.0, 'rejected'),
    (1.0, 'aborted'), (1.0, 'stale'), (1.0, 'missing'), (1.0, 'nonfinite'),
    (1.0, 'clock_reset'), (1.0, 'stop'), (1.0, 'cancel_missing'),
    (1.0, 'late_acceptance'), (1.0, 'acceptance_missing'),
])
def test_gripper_action_completion_uses_simulation_feedback_and_owned_cancellation(
    monkeypatch, rate, fault,
):
    """Exercise slow clocks and terminal outcomes with one command and at most one cancel."""
    from cais_spade_llm.spec2primitives.adapters import gazebo_execution

    class Future:
        def __init__(self, result, ready):
            self.value, self.ready = result, ready

        def done(self):
            return self.ready()

        def result(self):
            return self.value

    def point(**kwargs):
        return SimpleNamespace(**kwargs, time_from_start=SimpleNamespace(sec=0, nanosec=0))

    monkeypatch.setitem(sys.modules, 'trajectory_msgs.msg', SimpleNamespace(JointTrajectoryPoint=point))
    monkeypatch.setitem(sys.modules, 'control_msgs.msg', SimpleNamespace(JointTolerance=SimpleNamespace))
    monkeypatch.setitem(sys.modules, 'control_msgs.action', SimpleNamespace(
        FollowJointTrajectory=SimpleNamespace(Goal=lambda: SimpleNamespace(
            trajectory=SimpleNamespace(), goal_time_tolerance=SimpleNamespace(sec=0, nanosec=0))),
    ))
    wall = 0.0
    cancelled = None
    sent = []
    cancelled_calls = []
    stop = threading.Event()
    config = {'gripper': {'joint': 'drive', 'open': 0.0, 'close': 0.85,
                          'move_time_sec': 0.4, 'feedback_timeout_pad_sec': 0.5,
                          'position_tolerance': 0.01, 'settle_sec': 0.08}}
    profile = {**load_execution_profile(), 'state_max_age_sec': 0.2,
               'service_timeout_sec': 0.2, 'stop_timeout_sec': 0.2,
               'trajectory_timeout_pad_sec': 1.0}
    session = GazeboExecutionSession(config, profile, stop)

    def now():
        return int((2 + rate * wall) * 1e9) if fault != 'clock_reset' or wall < 0.1 else 0

    response = SimpleNamespace(status=4, result=SimpleNamespace(error_code=0, error_string=''))

    def terminal_ready():
        if cancelled is not None:
            if fault == 'cancel_missing':
                return False
            response.status = 5
            return wall >= cancelled + 0.04
        if fault == 'aborted' and rate * wall >= 0.2:
            response.status, response.result.error_code, response.result.error_string = 6, -5, 'goal tolerance'
            return True
        return rate * wall >= 0.4

    terminal = Future(response, terminal_ready)

    class Goal:
        accepted = fault != 'rejected'

        def get_result_async(self):
            return terminal

        def cancel_goal_async(self):
            nonlocal cancelled
            cancelled = wall
            cancelled_calls.append(wall)
            return Future(SimpleNamespace(goals_canceling=[1]), lambda: fault != 'cancel_missing')

    def send(goal):
        sent.append(goal)
        return Future(Goal(), lambda: fault != 'acceptance_missing' and
                      (fault != 'late_acceptance' or wall >= 0.3))

    def spin(*, timeout_sec):
        nonlocal wall
        wall += timeout_sec
        if fault in {'stop', 'cancel_missing', 'late_acceptance'} and wall >= 0.1:
            stop.set()
        stamp = now() if fault != 'stale' else 1_000_000_000
        value = 0.411 + (0.85 - 0.411) * min(1, rate * wall / 0.4)
        session.joints = SimpleNamespace(
            name=[] if fault == 'missing' else ['drive'],
            position=[float('nan') if fault == 'nonfinite' else value],
            header=SimpleNamespace(stamp=SimpleNamespace(sec=stamp // 10**9, nanosec=stamp % 10**9)),
        )

    monkeypatch.setattr(gazebo_execution, 'time', SimpleNamespace(monotonic=lambda: wall))
    session.node = SimpleNamespace(get_clock=lambda: SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=now())),
                                   destroy_node=lambda: None)
    session.executor = SimpleNamespace(spin_once=spin, shutdown=lambda: None)
    session.gripper = SimpleNamespace(send_goal_async=send)
    if fault is None:
        result = session._gripper_command(0.85)
        assert result['success'] and result['termination_confirmed']
        assert result['position'] == pytest.approx(0.85)
        assert result['elapsed_simulation_sec'] >= 0.46
        if rate == 0.43:
            assert result['elapsed_wall_sec'] > 0.9
    else:
        with pytest.raises(GripperCommandError) as error:
            session._gripper_command(0.85)
        assert error.value.outcome_known == (fault not in {'cancel_missing', 'acceptance_missing'})
        assert not error.value.diagnostics['success']
    try:
        session._close()
    except TimeoutError:
        assert fault in {'cancel_missing', 'acceptance_missing'}
    assert len(sent) == 1
    assert len(cancelled_calls) <= 1
    assert wall < 4


@pytest.mark.parametrize('topic', ['/xarm/controller/joint_trajectory', '/ur5e/controller/joint_trajectory'])
def test_gripper_action_uses_configured_controller(topic):
    assert gripper_action_name(topic) == topic.removesuffix('/joint_trajectory') + '/follow_joint_trajectory'
    with pytest.raises(ValueError):
        gripper_action_name('/unrelated')


@pytest.mark.parametrize("failed_primitive,known", [("grasp_part", True), ("grasp_part", False), ("release_part", True)])
def test_gripper_failure_preserves_diagnostics_and_acknowledged_attachment_custody(tmp_path, failed_primitive, known):
    from cais_spade_llm.spec2primitives.agents.ra.refinement_records import verify_record

    root = tmp_path / "interaction"
    robot, part, _ = _validated(root)

    class Transport(_Transport):
        async def gripper_command(self, position):
            await super().gripper_command(position)
            if (failed_primitive == "grasp_part") == (position == 1.0):
                raise GripperCommandError("Controller aborted closure.", {
                    "target": position, "position": 0.4, "stamp_ns": 10**9, "accepted": True,
                    "terminal_status": 6 if known else None, "termination_confirmed": known,
                    "elapsed_wall_sec": 1.0, "elapsed_simulation_sec": 0.43,
                    "cancel_requested": not known, "cancel_acknowledged": False, "success": False,
                })
            return {"success": True, "position": position}

    transport = Transport(part)
    executor = PrimitiveExecutionRuntime(robot_runtime=robot, validator=_validator, session_factory=transport, share=_SHARE)
    result = asyncio.run(executor.run(root))
    assert result["status"] == ("failed" if known else "unknown")
    assert failed_primitive in result["message"] and "Controller aborted" in result["message"]
    assert result["custody_known"] is known and result["gripper_state"] is None
    if failed_primitive == "grasp_part":
        assert not any(isinstance(call, tuple) and call[0] == "attach" for call in transport.calls)
        assert result["held_part"] is None
    else:
        assert not any(isinstance(call, tuple) and call[0] == "detach" for call in transport.calls)
        assert result["held_part"] is not None
    records = [verify_record(root, ref) for ref in result["record_refs"]]
    failed = next(record for record in records if record.get("record_type") == "PrimitiveExecutionGripperResult"
                  and record.get("success") is False)
    assert failed["position"] == 0.4 and failed["elapsed_simulation_sec"] == 0.43
    assert verify_record(root, failed["request_ref"])["target"] == failed["target"]


def _interrupted_reset_fixture(contexts_root):
    """Journal both robots without constructing composition, ROS or hardware clients."""
    roots = [contexts_root / "interaction_a", contexts_root / "interaction_b"]
    for index, (root, jid) in enumerate(zip(roots, ["xarm6@localhost", "ur5e@localhost"], strict=True)):
        directory = root / "execution" / f"run_{index}"
        request = append_record(root, directory, "request.json", {
            "record_type": "PrimitiveExecutionRequest", "resource_jid": jid,
            "created_at_ns": index + 1, "total_steps": 1, "candidate_ref": {},
        })
        append_record(root, directory, "robot_ready.json", {
            "record_type": "RobotValidationContext", "joint_state": {"names": ["arm_a", "arm_b"]},
        })
        append_record(root, directory, "result.json", {
            "record_type": "PrimitiveExecutionResult", "request_ref": request,
            "status": "unknown" if index == 0 else "stopped", "message": "Recorded interruption.",
            "command_dispatched": True, "custody_known": index == 1,
            "held_part": "medium gear" if index == 1 else None, "gripper_state": None,
            "record_refs": [], "last_event_ref": None,
        })
    return roots


class _ResetHost:
    execution_mode, robot_env, system_running = "simulation", "gazebo", False

    def __init__(self, *, running=False):
        self.state = "running" if running else "stopped"
        self.calls = []
        self.hardware = "stopped"

    def ros2_all_statuses(self):
        return {"gazebo_dual_spec2primitives": self.state}

    def hardware_stack_status(self, robot):
        return {"overall": self.hardware}

    def ros2_start(self, name):
        self.calls.append("start")
        self.state = "running"

    def ros2_stop(self, name):
        self.calls.append("stop")
        self.state = "stopped"

    def simulation_start_ready(self, force=False):
        return True, ""


class _ResetProbe:
    def __init__(self, profile, stop):
        self.stop = stop

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def snapshot(self):
        return {"ros_domain_id": 7, "publishers": {"/clock": ["old"], "/joint_states": ["old_joint"]}, "services": ["/old"]}

    def wait_stopped(self, runtime):
        return {"process_status": runtime.state, "endpoints": {"services": [], "publishers": {"/clock": [], "/joint_states": []}}}

    def wait_ready(self, old, required):
        return {"ros_domain_id": 7, "clock_publisher_gid": "new", "first_clock_ns": 10**9, "clock_ns": 2 * 10**9,
                "required_joints": required, "joint_feedback": {
                    name: {"position": 0.1, "stamp_ns": 2 * 10**9, "publisher_gid": "new_joint"} for name in required}}


@pytest.mark.parametrize("fault", [None, "stopped", "start", "readiness", "interlock", "record", "history", "old_program"])
def test_verified_reset_covers_both_robots_and_never_rewrites_history(tmp_path, fault):
    from cais_spade_llm.spec2primitives.agents.ra.execution_state import assert_interaction_current

    root, peer = _interrupted_reset_fixture(tmp_path)
    original = {path: path.read_bytes() for path in tmp_path.rglob("*.json")}
    host = _ResetHost(running=fault != "stopped")

    class Probe(_ResetProbe):
        def wait_ready(self, *args):
            if fault == "readiness":
                raise TimeoutError("Replacement clock is stalled.")
            if fault == "interlock":
                host.hardware = "starting"
            return super().wait_ready(*args)

    if fault == "start":
        host.ros2_start = lambda name: "Cannot start Gazebo."
    executor = PrimitiveExecutionRuntime(robot_runtime=object(), dual_gazebo=host, reset_probe_factory=Probe)
    result = asyncio.run(executor.reset_simulation(root))
    assert all(path.read_bytes() == data for path, data in original.items())
    if fault in {"start", "readiness", "interlock"}:
        assert result["status"] == "reset_required"
        with pytest.raises(ValueError, match="reset"):
            assert_execution_available(tmp_path)
        return
    assert result["status"] == "reset_completed"
    assert host.calls == (["start"] if fault == "stopped" else ["stop", "start"])
    if fault == "record":
        path = next(root.glob("execution/reset_*/result.json"))
        path.write_text(path.read_text().replace('"new"', '"old"'))
    if fault == "history":
        path = next(peer.glob("execution/run_*/result.json"))
        path.write_text(path.read_text().replace("medium gear", "changed"))
    if fault in {"record", "history"}:
        with pytest.raises(ValueError):
            assert_execution_available(tmp_path)
        return
    assert_execution_available(tmp_path)
    for jid in ["xarm6@localhost", "ur5e@localhost"]:
        assert execution_custody(tmp_path, jid) == {"held_part": None, "gripper_state": None}
    for old_root in [root, peer]:
        with pytest.raises(ValueError, match="fresh interaction"):
            assert_interaction_current(old_root)
    if fault == "old_program":
        with pytest.raises(ValueError, match="fresh interaction"):
            asyncio.run(executor.run(root))
        composer = PrimitiveRefinementRuntime(program_runtime=object(), robot_runtime=object())
        with pytest.raises(ValueError, match="fresh interaction"):
            asyncio.run(composer.compose(root))
    fresh = tmp_path / "interaction_fresh"
    fresh.mkdir()
    assert_interaction_current(fresh)


@pytest.mark.parametrize(("running", "expected_calls"), [
    (False, ["start"]),
    (True, []),
])
def test_run_starts_clean_scene_and_supersedes_failed_manual_reset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, running: bool,
    expected_calls: list[str],
) -> None:
    """Recover from stopped Gazebo and make its clean-scene record authoritative."""
    async def immediate(function: Any, *args: Any) -> Any:
        return function(*args)

    monkeypatch.setattr(asyncio, "to_thread", immediate)
    interrupted, _ = _interrupted_reset_fixture(tmp_path)
    reset_directory = interrupted / "execution" / "reset_3_failed"
    reset_request = append_record(interrupted, reset_directory, "request.json", {
        "record_type": "PrimitiveExecutionResetRequest", "created_at_ns": 3,
    })
    append_record(interrupted, reset_directory, "result.json", {
        "record_type": "PrimitiveExecutionResetResult", "request_ref": reset_request,
        "status": "reset_required", "message": "Gazebo start was interrupted.",
        "stopped": {
            "process_status": "stopped",
            "endpoints": {
                "services": [],
                "publishers": {"/clock": [], "/joint_states": []},
            },
        },
        "last_event_ref": None,
    })

    host, progress = _ResetHost(running=running), []
    executor = PrimitiveExecutionRuntime(robot_runtime=object(), dual_gazebo=host)
    async def report(event: dict[str, Any]) -> None:
        progress.append(event)

    fresh = asyncio.run(executor._start_gazebo_if_stopped(tmp_path, report))
    assert fresh is True
    assert host.calls == expected_calls
    assert progress[-1]["status"] == "preparing"
    assert_execution_available(tmp_path, fresh_simulation=fresh)

    root = tmp_path / "interaction_program"
    directory = root / "execution" / "run_4_clean"
    request = append_record(root, directory, "request.json", {
        "record_type": "PrimitiveExecutionRequest", "resource_jid": "xarm6@localhost",
        "created_at_ns": 4, "total_steps": 1, "candidate_ref": {},
        "fresh_simulation": True,
    })
    append_record(root, directory, "result.json", {
        "record_type": "PrimitiveExecutionResult", "request_ref": request,
        "status": "completed", "message": "Commands completed.",
        "command_dispatched": True, "custody_known": True,
        "held_part": None, "gripper_state": "open", "record_refs": [],
        "last_event_ref": None,
    })

    assert read_primitive_execution_diagnostic(root)["status"] == "completed"
    assert_execution_available(tmp_path)
    assert execution_custody(tmp_path, "xarm6@localhost") == {
        "held_part": None, "gripper_state": "open",
    }


def test_reset_joins_duplicates_survives_disconnect_and_blocks_conflicting_work(tmp_path):
    root, _ = _interrupted_reset_fixture(tmp_path)
    entered, released = threading.Event(), threading.Event()

    class Probe(_ResetProbe):
        def wait_ready(self, *args):
            entered.set()
            assert released.wait(5)
            return super().wait_ready(*args)

    async def scenario():
        host = _ResetHost()
        executor = PrimitiveExecutionRuntime(robot_runtime=object(), dual_gazebo=host, reset_probe_factory=Probe)
        first = asyncio.create_task(executor.reset_simulation(root))
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            assert executor.diagnostic(root)["status"] == "resetting"
            second = asyncio.create_task(executor.reset_simulation(root))
            await asyncio.sleep(0)
            first.cancel()
            with pytest.raises(asyncio.CancelledError):
                await first
            with pytest.raises(ValueError, match="different"):
                await executor.run(root)
            composer = PrimitiveRefinementRuntime(program_runtime=object(), robot_runtime=object())
            with pytest.raises(ValueError, match="active"):
                await composer.compose(root)
            assert execution_busy()
            released.set()
            result = await second
            assert result["status"] == "reset_completed"
            assert host.calls == ["start"]
            assert len(list(root.glob("execution/reset_*"))) == 1
        finally:
            released.set()

    asyncio.run(scenario())


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


@pytest.mark.parametrize("parent, attach, success", [
    (None, True, True), (None, False, True),
    ({"model_name": "Gear_Plate", "link": "Gear_Plate"}, True, True),
    ({"model_name": "Gear_Plate", "link": "Gear_Plate"}, True, False),
])
def test_attachment_service_uses_the_selected_parent_and_requires_acknowledgment(
    monkeypatch: pytest.MonkeyPatch, parent: dict[str, str] | None, attach: bool, success: bool,
) -> None:
    """Exercise the ROS request adapter without a Gazebo service or robot command."""
    import sys
    from types import SimpleNamespace

    service = SimpleNamespace(Request=SimpleNamespace)
    monkeypatch.setitem(sys.modules, "linkattacher_msgs.srv", SimpleNamespace(AttachLink=service, DetachLink=service))
    session = GazeboExecutionSession(
        {"attach": {"robot_model_name": "configured_robot", "primary_attach_link": "configured_finger"}},
        load_execution_profile(), threading.Event(),
    )
    requests = []
    session.attach = session.detach = SimpleNamespace(call_async=lambda request: requests.append(request))
    monkeypatch.setattr(session, "_await", lambda *args: SimpleNamespace(success=success, message="Fixture acknowledgment."))
    binding = {"model_name": "gear_medium", "link": "link"}
    if success:
        result = session._attachment(binding, attach, parent)
        assert result["success"] is True and result["attached"] is attach
        assert result.get("parent") == parent
    else:
        with pytest.raises(RuntimeError, match="attachment command failed"):
            session._attachment(binding, attach, parent)
    request = requests[0]
    expected = parent or {"model_name": "configured_robot", "link": "configured_finger"}
    assert (request.model1_name, request.link1_name) == (expected["model_name"], expected["link"])
    assert (request.model2_name, request.link2_name) == ("gear_medium", "link")


@pytest.mark.parametrize("fault", [None, "wrong_target", "missing_board", "board_ack", "board_timeout"])
def test_link_attachment_execution_places_only_after_acknowledged_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str | None,
) -> None:
    """An interfering nominal fit can place, but neither a bad target nor failed board custody can pass."""
    from xml.etree import ElementTree as ET

    fixture_package = ET.parse(_SHARE / "package.xml").getroot().findtext("name")
    package_shares = {fixture_package: _SHARE}
    package_lookups: list[str] = []

    def get_package_share_directory(package: str) -> str:
        package_lookups.append(package)
        return str(package_shares[package])

    monkeypatch.setitem(
        sys.modules, "ament_index_python.packages",
        SimpleNamespace(get_package_share_directory=get_package_share_directory),
    )
    root = tmp_path / "link_attachment"
    robot, part, model = _validated(
        root, fitting=True, bore_radius_m=.004987318, validation_profile=load_refinement_profile(),
        now_ros=102_400_000_000, monkeypatch=monkeypatch,
    )
    program = load_validated_program(root)
    original = {path: path.read_bytes() for path in root.rglob("*.json")}
    parent = load_execution_profile()["placement_attachment"]
    world = ET.parse(_SHARE / "worlds" / load_execution_profile()["world_file"]).getroot().find("world")
    fixture = world.find(f"model[@name='{parent['model_name']}']/link[@name='{parent['link']}']")
    assert fixture is not None, "Execution configuration must select a link in this fixture."

    class Transport(_Transport):
        async def entity_states(self, names: list[str]) -> dict[str, Any]:
            if fault == "missing_board" and names == [parent["model_name"]]:
                return {}
            return await super().entity_states(names)

        async def feedback(self, *args: Any) -> dict[str, Any]:
            result = await super().feedback(*args)
            pose = deepcopy(args[2])
            if self.calls.count("move_cartesian") == 5:
                pose["z"] += .002 if fault == "wrong_target" else .0005
            return {**result, "ee_pose": pose, "measured_at_ros_ns": 102_400_000_000}

        async def attachment(self, binding: Any, attach: bool, *, parent: Any = None) -> dict[str, Any]:
            if parent is None:
                return await super().attachment(binding, attach)
            self.calls.append(("board_attachment", binding["model_name"], dict(parent)))
            if fault == "board_timeout":
                raise TimeoutError("Board attachment has no acknowledgment.")
            return {"success": fault != "board_ack", "attached": attach, "parent": dict(parent)}

    transport = Transport(part)
    result = asyncio.run(PrimitiveExecutionRuntime(
        robot_runtime=robot, validator=_validator, session_factory=transport,
    ).run(root))
    assert package_lookups == [fixture_package]
    assert result["status"] == {None: "completed", "wrong_target": "failed", "missing_board": "blocked",
                                "board_ack": "failed", "board_timeout": "unknown"}[fault], result
    assert len(model.calls) == 1
    assert all(path.read_bytes() == content for path, content in original.items())
    assert result["assembly_success"] is None
    board_commands = [call for call in transport.calls if isinstance(call, tuple) and call[0] == "board_attachment"]
    if fault in {"wrong_target", "missing_board"}:
        assert board_commands == []
        assert not any(isinstance(call, tuple) and call[0] == "detach" for call in transport.calls)
    else:
        assert board_commands == [("board_attachment", "gear_medium", parent)]
        detach = transport.calls.index(("detach", "gear_medium"))
        assert transport.calls[detach + 1] == board_commands[0]
        if fault:
            assert result["custody_known"] is False and result["completed_steps"] == 8
            assert transport.calls.count("move_cartesian") == 5
        else:
            assert result["custody_known"] is True and result["held_part"] is None
            assert result["completed_steps"] == len(program.steps)
            release_ref = next(ref for ref in result["record_refs"] if ref["ref"].endswith("step_0009_result.json"))
            release = read_pin(root, release_ref)["outputs"]
            assert release["placement_attachment"]["parent"] == parent
            assert release["placement_metrics"]["position_error_m"] == pytest.approx(.0005)


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
