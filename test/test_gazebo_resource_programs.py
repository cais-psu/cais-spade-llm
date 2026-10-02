"""Saved Gazebo function programs, catalog gates, and pinned revisions."""

from __future__ import annotations

import ast
import asyncio
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from cais_spade_llm.recovery_framework import SCENE_PATH
from cais_spade_llm.recovery_framework.kmr_primitives import KMRPrimitives
from cais_spade_llm.recovery_framework.kmr_tasks import KMR_MOVE_VARIANTS, KMR_TASKS
from cais_spade_llm.resources.environment_models import build_environment_models
from cais_spade_llm.resources.gazebo_programs import (
    primitive_available, resource_program_revision, saved_robot_definition,
    saved_workflow_program, validate_resource_programs,
)
from cais_spade_llm.ui.components.nominal_resource_des import save_resource_program_edit


@pytest.fixture
def scene():
    return json.loads(SCENE_PATH.read_text())


def test_saved_catalog_covers_exact_resources_and_unused_executors(scene):
    validate_resource_programs(scene)
    resources = scene["resource_programs"]["resources"]
    robot = {
        "detect_parts", "compute_pick_targets", "compute_place_targets",
        "move_to_named_pose", "move_cartesian", "move_relative",
        "grasp_part", "release_part", "open_gripper", "close_gripper",
        "attach_part", "detach_part", "move_joints", "rotate_joint",
    }
    assert len(resources) == 12
    assert set(resources["KMR"]["functions"]) == {
        "pick_approach", "pick_part", "move_to_resource", "place_approach",
        "place_release", "move_to_location",
    }
    assert set(resources["KMR"]["primitives"]) == robot | {"move_base"}
    for rid in ("ur5e-1", "ur5e-2", "ur5e-3", "ur5e-4"):
        assert set(resources[rid]["functions"]) == {
            "pick_approach", "pick_grasp", "place_approach", "place_insert", "move_home",
        }
        assert set(resources[rid]["primitives"]) == robot
        assert not resources[rid]["primitives"]["move_joints"]["recovery_selectable"]
        assert not resources[rid]["primitives"]["rotate_joint"]["recovery_selectable"]
    assert not resources["KMR"]["primitives"]["move_joints"]["recovery_selectable"]
    assert not resources["KMR"]["primitives"]["rotate_joint"]["recovery_selectable"]
    assert {name: set(resources[name]["primitives"]) for name in ("M1", "M2", "Conveyor", "Buffer For Machined parts")} == {
        "M1": {"dwell"}, "M2": {"dwell"},
        "Conveyor": {"move_relative"}, "Buffer For Machined parts": {"move_relative"},
    }
    assert resources["3D Printing Station"]["primitives"] == {}
    assert resources["3D Printing Station"]["functions"]["print_part"]["program"]["steps"] == []
    assert not resources["Storage"]["primitives"] and not resources["Exit"]["primitives"]
    assert not primitive_available(scene, "KMR", "dock", recovery=True)
    assert not primitive_available(scene, "ur5e-1", "snap_part_to_slot", recovery=True)
    assert saved_workflow_program(scene, "M1", "machine_part")["steps"][0]["op"] == "dwell"
    assert saved_workflow_program(scene, "Conveyor", "advance_conveyor")["steps"][0]["op"] == "move_relative"


def test_place_insert_seats_assembly_part_before_release(scene):
    for resource_id in ("ur5e-1", "ur5e-2", "ur5e-3", "ur5e-4"):
        program = saved_robot_definition(scene, resource_id, "place_insert").program
        assert [(step.id, step.op) for step in program.steps] == [
            ("seat_part", "move_cartesian"),
            ("release_part", "release_part"),
            ("lift", "move_relative"),
        ]
        seat = program.steps[0]
        assert seat.when[0].predicate == "move_insert_required"
        assert all(seat.params[axis] == {"$state": "_task_ctx", "path": ["insert_pose", axis]}
                   for axis in ("x", "y", "z", "qx", "qy", "qz", "qw"))
        release = program.steps[1]
        assert ("assembly_slot" in release.params) == (resource_id in {"ur5e-3", "ur5e-4"})
        assert "move_insert" not in scene["resource_programs"]["resources"][resource_id]["primitives"]


def test_ui_save_changes_the_same_robot_program_used_by_runtime(scene, tmp_path):
    path = tmp_path / "scene.json"
    path.write_text(json.dumps(scene))
    bridge = SimpleNamespace(
        load_config=lambda filename: json.loads(Path(filename).read_text()),
        save_config=lambda filename, value: Path(filename).write_text(json.dumps(value)),
    )
    first = resource_program_revision(scene)
    function = deepcopy(scene["resource_programs"]["resources"]["ur5e-1"]["functions"]["move_home"])
    function["program"]["steps"][0]["params"]["speed"] = 0.2
    second = save_resource_program_edit(
        bridge, path, "ur5e-1", first, event_name="move_home", value=function,
    )
    saved = json.loads(path.read_text())
    assert second != first and second == resource_program_revision(saved)
    assert saved_robot_definition(saved, "ur5e-1", "move_home").program.steps[0].params["speed"] == 0.2
    event = next(row for row in build_environment_models(saved)["ur5e-1"]["events"]
                 if row["event_name"] == "move_home")
    assert event["program"]["steps"][0]["params"]["speed"] == 0.2
    from cais_spade_llm.ui.components.resource_function_catalog import resource_function_rows

    release = next(row for row in resource_function_rows(build_environment_models(saved)["ur5e-1"])["functions"]
                   if row["event_name"] == "place_release")
    assert release["program_key"] == "place_insert"
    with pytest.raises(ValueError, match="revision changed"):
        save_resource_program_edit(bridge, path, "ur5e-1", first,
                                   event_name="move_home", value=function)
    invalid = deepcopy(function)
    invalid["program"]["steps"] = []
    with pytest.raises(ValueError, match="guarded step sequence"):
        save_resource_program_edit(bridge, path, "ur5e-1", second,
                                   event_name="move_home", value=invalid)
    assert resource_program_revision(json.loads(path.read_text())) == second


def test_recovery_can_call_unused_kmr_primitive_but_not_removed_command(scene):
    worker = SimpleNamespace(run=AsyncMock(return_value={"result": {"parts": {}}}))
    agent = SimpleNamespace(
        _kmr_execution_request={"inputs": {"scene": scene}},
        environment_runtime=None, delivery_runtime=None,
        worker=worker, record_primitive_evidence=Mock(),
    )
    result = asyncio.run(KMRPrimitives(agent).detect_parts())
    assert result["success"] is True
    assert worker.run.await_args.args[0]["primitive"] == "detect_parts"
    unavailable = asyncio.run(KMRPrimitives(agent).move_joints([0.0] * 7))
    assert unavailable["success"] is False
    assert not primitive_available(scene, "KMR", "dock", recovery=True)
    bad = deepcopy(scene)
    bad["resource_programs"]["resources"]["M1"]["primitives"]["spindle_on"] = {
        "status": "executable", "recovery_selectable": True,
    }
    with pytest.raises(ValueError, match="Incomplete primitive catalog"):
        validate_resource_programs(bad)


def test_recovery_dispatch_accepts_unused_ur5e_control_and_rejects_removed_command(scene):
    from cais_spade_llm.agents.resource_agent.robot_agent import RobotAgent

    bundle = scene["resource_programs"]["resources"]["ur5e-1"]
    assert all(step["op"] != "close_gripper"
               for function in bundle["functions"].values()
               for step in function["program"]["steps"])
    controller = SimpleNamespace(close_gripper=Mock(return_value=True),
                                 get_current_pose=Mock(return_value={"success": True}))
    agent = object.__new__(RobotAgent)
    agent.agent_name = "ur5e-1"
    agent.execution_mode = "simulation"
    agent.gazebo_program_scene = scene
    agent.environment_runtime = SimpleNamespace(scene_file="")
    agent._controller_prewarm_done = True
    agent._controller = controller

    assert asyncio.run(agent._execute_primitive("close_gripper", {}))["success"] is True
    assert asyncio.run(agent._execute_primitive("get_current_pose", {}))["success"] is False
    controller.close_gripper.assert_called_once_with()
    controller.get_current_pose.assert_not_called()


def test_kmr_saved_handoffs_bind_previous_observations_and_reject_missing_outputs(scene):
    functions = scene["resource_programs"]["resources"]["KMR"]["functions"]
    assert functions["pick_part"]["program"]["steps"][0]["params"]["target"] == {
        "$state": "previous", "path": ["pick_targets", "target"],
    }
    assert functions["place_approach"]["program"]["steps"][0]["params"]["transform"] == {
        "$state": "previous", "path": ["grasp_transform"],
    }
    assert functions["place_release"]["program"]["steps"][0]["params"]["target"] == {
        "$state": "previous", "path": ["place_targets", "target"],
    }
    stale = deepcopy(scene)
    stale["resource_programs"]["resources"]["KMR"]["functions"]["pick_part"]["program"]["steps"][0]["params"]["target"] = {
        "$step": "pick", "path": ["target"],
    }
    with pytest.raises(ValueError, match="unavailable step output"):
        validate_resource_programs(stale)


def test_every_kmr_base_controller_call_belongs_to_a_composed_move_base_step():
    from cais_spade_llm.recovery_framework.kmr_tasks import KMR_LOCATION_TASK

    assert all(any(step.op == "move_base" for step in definition.program.steps)
               for definition in (KMR_TASKS["pick_approach"], KMR_TASKS["move_to_resource"],
                                  *KMR_MOVE_VARIANTS.values(), KMR_LOCATION_TASK))
    path = Path(__file__).resolve().parents[1] / "cais_spade_llm/recovery_framework/kmr_gazebo.py"
    tree = ast.parse(path.read_text())
    parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name) and node.func.id == "action"
             and node.args and isinstance(node.args[0], ast.Name)
             and node.args[0].id == "MoveBaseKMR"]
    assert len(calls) == 1
    ancestors = []
    current = calls[0]
    while current in parents:
        current = parents[current]
        if isinstance(current, ast.FunctionDef):
            ancestors.append(current.name)
    assert "dispatch_base_move" in ancestors
    assert any(isinstance(node, ast.FunctionDef) and node.name == "move_base"
               for node in ast.walk(tree))


def test_kmr_dock_requires_the_configured_yaw():
    from math import cos, sin

    from cais_spade_llm.recovery_framework.kmr_gazebo import _at_dock

    target = [1.0, 2.0, 0.0]
    aligned = [1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    turned = [1.0, 2.0, 0.0, 0.0, 0.0, sin(0.2), cos(0.2)]
    assert _at_dock(aligned, target)
    assert not _at_dock(turned, target)



def test_compact_dispatch_rejects_scene_revision_changed_after_start(scene, tmp_path, monkeypatch):
    from cais_spade_llm.recovery_framework import ROOT, delivery, read_json

    path = tmp_path / "scene.json"
    path.write_text(json.dumps(scene))
    from cais_spade_llm.recovery_framework import PRODUCT_PATH
    product = next(iter(read_json(PRODUCT_PATH).values()))
    inputs = {
        "scene": scene,
        "product_order": read_json(delivery.ORDER_PATH),
        "geometry": read_json(ROOT / product["product_geometry_file"])["gazebo"],
    }
    from threading import Event

    monkeypatch.setattr(delivery, "_cancelled", Event())
    monkeypatch.setattr(delivery, "RUN_DIRECTORY", tmp_path / "runs")
    actor = SimpleNamespace(agent_name="KMR", jid="kmr@localhost", configure_nominal=Mock())
    runtime = delivery.DeliveryRuntime(
        SimpleNamespace(),
        {"inputs": inputs, "setup": {"scene_file": str(path)}, "source_fingerprints": {}},
        [actor],
    )
    changed = deepcopy(scene)
    changed["resource_programs"]["revision"] += 1
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="Stale Gazebo program revision"):
        runtime.prepare_dispatch({})
    assert runtime.context.revision == 0


def test_saved_transport_speed_is_used_and_cannot_exceed_scene_limit(scene):
    from cais_spade_llm.recovery_framework.workflow_gazebo import WorkflowPrimitiveRunner

    function = deepcopy(scene["resource_programs"]["resources"]["Conveyor"]["functions"]["advance_conveyor"])
    step = next(row for row in function["program"]["steps"] if row["op"] == "move_relative")
    step["params"]["speed_mps"] = 1.0
    edited = deepcopy(scene)
    edited["resource_programs"]["resources"]["Conveyor"]["functions"]["advance_conveyor"] = function
    validate_resource_programs(edited)
    assert saved_workflow_program(edited, "Conveyor", "advance_conveyor")["steps"][0]["params"] == {
        "speed_mps": 1.0,
    }
    part_state = SimpleNamespace(pose=SimpleNamespace(position=SimpleNamespace(x=0.0)))
    durations = []

    def wait_simulation(duration, callback):
        durations.append(duration)
        callback(duration)
        return 0.0, duration

    runner = WorkflowPrimitiveRunner(
        {"scene": edited, "geometry": {"part": {"model_name": "part"}},
         "task": {"resource_id": "Conveyor", "event_name": "advance_conveyor",
                  "parameters": {"part_name": "part"}}},
        entity=lambda _name: part_state,
        set_pose=lambda _name, _pose: None,
        robot_clearance=lambda _names, _states: {},
        current_clock=lambda: 0.0,
        wait_simulation=wait_simulation,
    )
    runner.state.update(displacement=1.0, speed=2.0, acceleration=2.0,
                        target_x={"part": 1.0}, initial_x={"part": 0.0},
                        part_states={"part": part_state}, clearance_robots=[], duration=1.0)
    runner.observe_belt_residents = Mock()
    runner.compute_shared_displacement = Mock()
    runner.verify_transport_clearance = Mock()
    runner.confirm_arrival = Mock()
    runner.move_relative(**step["params"])
    for operation in (runner.observe_belt_residents, runner.compute_shared_displacement,
                      runner.verify_transport_clearance, runner.confirm_arrival):
        operation.assert_called_once_with()
    assert runner.state["speed"] == 1.0
    assert durations[0] > 1.0
    assert part_state.pose.position.x == pytest.approx(1.0)
    step["params"]["speed_mps"] = 3.0
    with pytest.raises(ValueError, match="Unsafe workflow primitive parameters"):
        validate_resource_programs(edited)


def test_saved_composition_adds_supported_steps_and_executes_them_in_saved_order(scene, tmp_path):
    from cais_spade_llm.recovery_framework.kmr_tasks import execute_composition

    path = tmp_path / "scene.json"
    path.write_text(json.dumps(scene))
    bridge = SimpleNamespace(
        load_config=lambda filename: json.loads(Path(filename).read_text()),
        save_config=lambda filename, value: Path(filename).write_text(json.dumps(value)),
    )
    function = deepcopy(scene["resource_programs"]["resources"]["KMR"]["functions"]["move_to_location"])
    function["program"]["steps"].insert(0, {
        "id": "read_parts", "op": "detect_parts", "params": {}, "when": [],
    })
    save_resource_program_edit(bridge, path, "KMR", resource_program_revision(scene),
                               event_name="move_to_location", value=function)
    saved = json.loads(path.read_text())
    definition = saved_robot_definition(saved, "KMR", "move_to_location")
    calls, records = [], []
    execute_composition(
        "move_to_location", arguments={},
        state={"base_target_pose": [0., 1., 0.], "base_waypoints": []},
        primitives={
            "detect_parts": lambda: calls.append("detect_parts") or {"parts": {}},
            "move_base": lambda **params: calls.append(("move_base", params)) or {"base_pose": [0., 1., 0.]},
        },
        records=records, operations=[], now=lambda: 0., check=lambda: None,
        serialize=lambda value: value, definition=definition,
    )
    assert calls == ["detect_parts", ("move_base", {"target_pose": [0., 1., 0.], "waypoints": []})]
    assert [(row["step_id"], row["primitive"]) for row in records] == [
        (step.id, step.op) for step in definition.program.steps
    ]
    assert all(row["status"] == "completed" for row in records)


@pytest.mark.parametrize("step", [
    {"id": "extra", "op": "get_current_pose", "params": {}},
    {"id": "extra", "op": "release_part", "params": {}},
    {"id": "extra", "op": "move_cartesian", "params": {"xyz": [0., 0., 1.]}},
    {"id": "extra", "op": "detect_parts", "params": {}, "continue_on_failure": True},
])
def test_composition_edit_rejects_removed_commands_and_contract_changing_steps(scene, step):
    scene["resource_programs"]["resources"]["KMR"]["functions"]["move_to_location"]["program"]["steps"].insert(0, step)
    with pytest.raises(ValueError):
        validate_resource_programs(scene)


def test_kmr_recovery_uses_its_own_probe_and_measured_hold_after_an_unrelated_event(scene):
    from cais_spade_llm.recovery_framework.kmr_agent import KMRResourceAgent

    part = "KET8_Square_8mm"
    transform = [0., 0., -.025, 0., 0., 0., 1.]
    grasp = {"attached": True, "held_part": part, "grasp_transform": transform}
    moved = {"attached": True, "base_pose": [-7., 3., 0.], "resource_location": None}
    worker = SimpleNamespace(run=AsyncMock(side_effect=[
        {"result": grasp, "primitive_results": [{"primitive": "grasp_part", "status": "completed", "result": grasp}]},
        {"result": moved, "primitive_results": [{"primitive": "move_base", "status": "completed", "result": moved}]},
    ]))
    agent = KMRResourceAgent("kmr-recovery-context@localhost", "none", worker=worker)
    nominal = {"KMR": {"resource_state": "idle", "held_part": None, "resource_location": "Storage"}}
    runtime = SimpleNamespace(
        context=SimpleNamespace(
            inputs={"scene": scene}, geometry={part: {"dimensions_m": [.008, .008, .05]}},
            transitions=[{"acknowledgement": {"resource_id": "ur5e-4"}, "observations": {"held_part": "gear_small"}}],
            snapshot=lambda: deepcopy(nominal),
        ),
        stopped=False, scene_file="", prepared={}, kmr_probe={"launch_id": "active-gazebo"},
    )
    agent.environment_runtime = runtime

    async def run():
        assert (await agent.kmr_primitives.grasp_part(part_name=part))["success"]
        assert (await agent.kmr_primitives.move_base([-7., 3., 0.], transform=transform))["success"]

    asyncio.run(run())
    request = worker.run.await_args.args[0]
    assert request["probe"] == runtime.kmr_probe
    assert request["custody"]["grasp_transform"] == transform
    assert request["valuation"]["KMR"]["held_part"] == part
    assert request["geometry"][part]["dimensions_m"] == [.008, .008, .05]
    assert nominal["KMR"]["held_part"] is None
    assert nominal["KMR"]["resource_state"] == "idle"
    assert agent._primitive_state["resource_location"] is None


@pytest.mark.parametrize("ticks,armed,observation_fails", [
    ([0., 0., 1., 1., 2.5], True, False),
    ([0., 2.49, 2.51], True, False),
    ([0., 0.25, 5.], False, False),
    ([0., 0., 0.], True, False),
    ([0., 1., 2.5], True, True),
])
def test_machine_fault_uses_simulation_clock_and_preserves_WIP(scene, tmp_path, ticks, armed, observation_fails):
    from cais_spade_llm.recovery_framework.workflow_gazebo import WorkflowPrimitiveRunner

    path = tmp_path / "fault.json"
    path.write_text(json.dumps({"run_id": "run", "resource_id": "M1",
                                "status": "armed" if armed else "disarmed"}))
    machine = next(row for row in scene["machines"] if row["resource_id"] == "M1")
    position = SimpleNamespace(**dict(zip(("x", "y", "z"), machine["workholding_pose"][:3])))
    state = SimpleNamespace(pose=SimpleNamespace(position=position))
    visited = []
    def wait(duration, observe):
        for elapsed in ticks:
            visited.append(elapsed)
            observe(elapsed)
        if ticks[-1] < duration:
            raise InterruptedError("Operator stopped while /clock was paused")
        return 10., 10. + ticks[-1]
    def entity(_name):
        if observation_fails and visited and visited[-1] >= 2.5:
            raise ValueError("Workholding observation unavailable")
        return state

    runner = WorkflowPrimitiveRunner(
        {"scene": scene, "geometry": {"KET4_Square_4mm": {"model_name": "KET4_Square_4mm"}},
         "task": {"run_id": "run", "task_id": "machining", "resource_id": "M1",
                  "event_name": "machine_part", "parameters": {
                      "part_name": "KET4_Square_4mm", "process": "trim", "result": "square"}},
         "failure_injection": {"run_id": "run", "task_id": "machining", "resource_id": "M1",
                               "checkpoint": "during_processing_halfway", "control_path": str(path),
                               "evidence_path": str(tmp_path / "interruption.json")}},
        entity=entity, set_pose=Mock(),
        robot_clearance=lambda _names, _states: {machine["handling_robot"]: 1.},
        current_clock=lambda: 10., wait_simulation=wait,
    )
    runner._kmr_clearance = lambda: {"link": 1.}
    if max(ticks) < 2.5:
        with pytest.raises(InterruptedError, match="/clock"):
            runner.execute()
        assert "failure_injection" not in runner.state
        assert not runner.observations
        assert not (tmp_path / "interruption.json").exists()
    elif observation_fails:
        with pytest.raises(ValueError, match="Workholding"):
            runner.execute()
        evidence = json.loads((tmp_path / "interruption.json").read_text())
        assert evidence["source"] == "gazebo_processing_checkpoint"
        assert evidence["observation_status"] == "failed"
        assert not evidence["process_completed"]
        assert "observed_pose" not in evidence
        assert not runner.observations
    elif armed:
        with pytest.raises(RuntimeError, match="Machining breakdown"):
            runner.execute()
        failure = runner.state["failure_injection"]
        assert failure["simulation_elapsed_sec"] == ticks[-1]
        assert failure["processing_fraction"] >= .5
        assert json.loads((tmp_path / "interruption.json").read_text()) == failure
        assert failure["observed_pose"]["z"] == machine["workholding_pose"][2]
        assert not failure["process_completed"]
        assert runner.observations == {}
        assert runner.primitive_trace[-1]["status"] == "failed"
    else:
        runner.execute()
        assert "failure_injection" not in runner.state
        assert runner.observations["processing_time_sec"] == 5.
    assert visited == ticks
