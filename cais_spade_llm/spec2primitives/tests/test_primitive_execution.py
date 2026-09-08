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
from cais_spade_llm.spec2primitives.agents.ra.refinement import PrimitiveRefinementRuntime
from cais_spade_llm.spec2primitives.agents.ra.refinement_records import (
    append_record,
    fingerprint,
    read_pin,
    verify_evidence_tree,
)
from cais_spade_llm.spec2primitives.agents.ra.primitive_composition import _evidence_value
from cais_spade_llm.spec2primitives.tests.test_primitive_refinement import (
    _setup,
    _program,
    _PlanningSession,
)
from cais_spade_llm.spec2primitives.tests.test_ra_context_handoff import (
    _ProgramRuntime,
    _program_action,
)
from cais_spade_llm.spec2primitives.tools.assembly_geometry import AssemblyGeometryProducer
from cais_spade_llm.spec2primitives.tools.exact_ref_resolver import approved_cad_path

_SHARE = Path(__file__).resolve().parents[3] / "ros2/cais_lab_robotics"


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


def _validated(root: Path) -> tuple[Any, Any, Any]:
    root.mkdir()
    inputs, robot, refs, roles = _setup(root)
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
    part.update(cad_context_ref="Gear_Medium.STL", reference_point="CAD_origin")
    roles["part"] = refs["part"] = append_record(root, root / "evidence", "part_cad.json", part)
    steps = _program(refs)
    model = _ProgramRuntime(
        [
            _program_action(
                [(s["primitive_symbol"], s["params"]) for s in _program(refs, bound=False)]
            ),
            {
                "kind": "request_context",
                "requests": [
                    {
                        "step_index": 1,
                        "quantity": "Part, goal and scene evidence",
                        "authority": "PA",
                        "reason": "Bind selected geometry.",
                    }
                ],
            },
            _program_action([(s["primitive_symbol"], s["params"]) for s in steps]),
        ]
    )

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

    class Product:
        async def investigate(self, **kwargs: Any) -> dict[str, Any]:
            return {
                "status": "provided",
                "operations_used": 1,
                "evidence_refs": list(refs.values()),
                "validation_refs": roles,
                "unresolved": [],
            }

    runtime = PrimitiveRefinementRuntime(
        program_runtime=model,
        robot_runtime=Robot(),
        product_runtime=Product(),
        validator=_validator,
    )
    result = asyncio.run(runtime.compose(root))
    assert result["status"] == "validated_for_declared_scope", result
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
        pose = pose_matrix(self.part["origin_pose"]) @ np.linalg.inv(
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
        if self.fail == "attach":
            raise TimeoutError("Attachment has no acknowledgment.")
        return {"success": True, "attached": attach}


def test_whole_program_reuses_saved_authority_and_actual_cad_mapping(tmp_path: Path) -> None:
    root = tmp_path / "interaction"
    robot, part, model = _validated(root)
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
    assert len(model.calls) == 3, "Execution must not call composition again."
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
    ],
)
def test_failures_stop_dispatch_and_keep_unknown_custody(
    tmp_path: Path, failure: str, expected_status: str, expected_calls: int
) -> None:
    root = tmp_path / "interaction"
    robot, part, _ = _validated(root)
    transport = _Transport(part, fail=failure)
    executor = PrimitiveExecutionRuntime(
        robot_runtime=robot, validator=_validator, session_factory=transport, share=_SHARE
    )
    result = asyncio.run(executor.run(root))
    assert result["status"] == expected_status, result
    assert len(transport.calls) == expected_calls
    if expected_calls <= 1:
        assert result["gripper_state"] is None
    if expected_status == "unknown":
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
        assert len(model.calls) == 3

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
