"""Physical placement interruption and approved concurrent pickup gates."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from cais_spade_llm.recovery_framework.placement_motion import (
    placement_motion,
    wait_for_placement_motion,
)
from cais_spade_llm.recovery_framework.slippage_execution import SlippagePickupGate


def pose(z, x=.03, y=.15):
    return SimpleNamespace(position=SimpleNamespace(x=x, y=y, z=z),
                           orientation=SimpleNamespace(x=0., y=1., z=0., w=0.))


def controller():
    request = {"run_id": "run", "task_id": "placement", "resource_id": "ur5e-4",
               "part_name": "gear_small", "model_name": "gear_small",
               "checkpoint": "during_place_lowering", "placement_progress": .5}
    goal = SimpleNamespace(status=2, goal_id=SimpleNamespace(uuid=[1, 2, 3]))
    state = {"done": False}
    future = SimpleNamespace(done=lambda: state["done"])
    terminal = SimpleNamespace(status=5)
    cancel = SimpleNamespace(goals_canceling=[SimpleNamespace(goal_id=goal.goal_id)])

    def cancel_goal():
        state["done"] = True
        goal.status = terminal.status
        return cancel

    goal.cancel_goal_async = Mock(side_effect=cancel_goal)
    fresh = Mock(side_effect=[pose(1.5), None, pose(1.4), pose(1.24), pose(1.24)])
    owner = SimpleNamespace(
        _simulation_fault_request=request, _robot_task_step=("place_approach", "descend"),
        _simulation_task_id="placement", robot_name="ur5e-4", execution_mode="simulation",
        _attached_model="gear_small", _fresh_simulation_tf_tool_pose=fresh,
        _simulation_fault_armed=lambda: True, _rclpy=SimpleNamespace(ok=lambda: True),
        _motion_pending=lambda _: lambda: True, _simulation_goal=goal,
        arm_joint_names=["joint"], _get_arm_joint_positions=lambda **_: ([.1], []),
        _fresh_stable_joint_target=Mock(return_value=True),
        _last_joint_target_observation={"fresh_stable": True},
        _wait_future=lambda item, **_: terminal if item is future else item,
    )
    return owner, goal, future, terminal, cancel


def test_lowering_uses_observed_distance_and_confirmed_stop(monkeypatch):
    owner, goal, future, _, _ = controller()
    motion = placement_motion(owner, pose(1.0))
    monkeypatch.setattr("cais_spade_llm.recovery_framework.placement_motion.time.sleep", lambda _: None)
    assert wait_for_placement_motion(owner, goal, future, 5., motion) is None
    evidence = owner._simulation_fault_evidence
    assert evidence["progress"] == pytest.approx(.52)
    assert evidence["observed_pose"]["z"] == 1.24
    assert evidence["goal_active"] and evidence["goal_cancelled"] and evidence["motion_stopped"]
    assert evidence["terminal_status"] == 5
    assert owner._simulation_goal is None
    goal.cancel_goal_async.assert_called_once()
    owner._fresh_stable_joint_target.assert_called_once()


@pytest.mark.parametrize("change", ["wrong_goal", "completed_goal"])
def test_lowering_rejects_unconfirmed_cancel(monkeypatch, change):
    owner, goal, future, terminal, cancel = controller()
    motion = placement_motion(owner, pose(1.0))
    if change == "wrong_goal":
        cancel.goals_canceling = [SimpleNamespace(goal_id=SimpleNamespace(uuid=[9]))]
    else:
        terminal.status = 4
    monkeypatch.setattr("cais_spade_llm.recovery_framework.placement_motion.time.sleep", lambda _: None)
    with pytest.raises(RuntimeError, match="cancellation was not confirmed"):
        wait_for_placement_motion(owner, goal, future, 5., motion)
    assert not owner._simulation_fault_evidence["motion_stopped"]


@pytest.mark.parametrize("field,value", [
    ("resource_id", "ur5e-3"), ("task_id", "old"), ("model_name", "other_part"),
    ("placement_progress", 1.), ("placement_progress", float("nan")),
])
def test_fault_binding_cannot_interrupt_another_motion(field, value):
    owner, *_ = controller()
    owner._simulation_fault_request[field] = value
    with pytest.raises(ValueError):
        placement_motion(owner, pose(1.0))


def test_unrelated_primitive_and_upward_segment_do_not_arm_lowering():
    owner, *_ = controller()
    owner._robot_task_step = ("pick_approach", "descend")
    assert placement_motion(owner, pose(1.0)) is None
    owner._robot_task_step = ("place_approach", "descend")
    assert placement_motion(owner, pose(1.6)) is None


def test_missing_start_observation_blocks_fault_motion():
    owner, *_ = controller()
    owner._fresh_simulation_tf_tool_pose = lambda: None
    with pytest.raises(ValueError, match="fresh observed"):
        placement_motion(owner, pose(1.0))


def test_completed_or_disarmed_goal_does_not_create_fault():
    owner, goal, future, _, _ = controller()
    motion = placement_motion(owner, pose(1.0))
    owner._simulation_fault_armed = lambda: False
    owner._motion_pending = lambda _: lambda: False
    assert wait_for_placement_motion(owner, goal, future, 5., motion).status == 5
    assert not hasattr(owner, "_simulation_fault_evidence")
    goal.cancel_goal_async.assert_not_called()


def test_pickup_gate_releases_only_current_admitted_tasks_together():
    async def scenario():
        pending = {}
        context = SimpleNamespace(run_id="run", negotiations=[], pending_for=pending.get)
        runtime = SimpleNamespace(context=context, stopped=False)
        gate = SlippagePickupGate(runtime, {
            "resource_id": "ur5e-4", "part_name": "gear_small",
            "additional_condition": {"resource_id": "ur5e-3", "part_name": "KET4_Square_4mm"}})
        tasks = [{
            "run_id": "run", "task_id": rid, "resource_id": rid, "event_name": "pick_approach",
            "parameters": {"part_name": part},
        } for rid, part in gate.parts.items()]
        pending.update({row["task_id"]: row for row in tasks})
        first = asyncio.create_task(gate.wait(tasks[0]))
        await asyncio.sleep(0)
        assert not first.done() and not gate.released.is_set()
        await gate.wait(tasks[1])
        await first
        assert gate.released.is_set()
        assert [row["kind"] for row in context.negotiations] == [
            "slippage_pickup_ready", "slippage_pickup_ready", "slippage_pickups_released"]
        stale = {**tasks[0], "run_id": "old"}
        other_gate = SlippagePickupGate(runtime, {
            "resource_id": "ur5e-4", "part_name": "gear_small",
            "additional_condition": {"resource_id": "ur5e-3", "part_name": "KET4_Square_4mm"}})
        with pytest.raises(ValueError, match="current admitted"):
            await other_gate.wait(stale)
    asyncio.run(scenario())

def test_buffer_initialization_Stop_does_not_move_part(monkeypatch):
    from cais_spade_llm.recovery_framework.slippage_initialization import _prepare_part
    from cais_spade_llm.recovery_framework import failure_effects

    monkeypatch.setattr(failure_effects, "_robot", Mock(side_effect=AssertionError("Stopped initialization")))
    with pytest.raises(asyncio.CancelledError):
        _prepare_part(SimpleNamespace(stopped=True), {})


def test_buffer_initialization_is_once_per_run_and_ordinary_runs_are_unchanged():
    from cais_spade_llm.recovery_framework.slippage_initialization import prepare_slippage_initial_conditions

    async def scenario():
        ordinary = SimpleNamespace()
        await prepare_slippage_initial_conditions(ordinary)
        assert not hasattr(ordinary, "slippage_initialization_evidence")
        evidence = {"run_id": "run", "status": "completed"}
        runtime = SimpleNamespace(
            context=SimpleNamespace(run_id="run"), conveyor_fault=SimpleNamespace(
                configuration={"initial_conditions": [{"part_name": "KET4_Square_4mm"}]}),
            slippage_initialization_evidence=evidence)
        await prepare_slippage_initial_conditions(runtime)
        assert runtime.slippage_initialization_evidence is evidence

    asyncio.run(scenario())

def test_native_controller_uuid_evidence_is_json_serializable(monkeypatch):
    import json
    import numpy as np

    owner, goal, future, _, _ = controller()
    goal.goal_id.uuid = np.array(range(16), dtype=np.uint8)
    motion = placement_motion(owner, pose(1.0))
    monkeypatch.setattr("cais_spade_llm.recovery_framework.placement_motion.time.sleep", lambda _: None)
    assert wait_for_placement_motion(owner, goal, future, 5., motion) is None
    assert json.loads(json.dumps(owner._simulation_fault_evidence))["controller_goal_id"] == list(range(16))


def test_recorder_releases_pickups_after_buffer_confirmation_and_capture():
    from cais_spade_llm.recovery_framework.failure_videos import _release_slippage_capture

    async def scenario():
        pending = {}
        context = SimpleNamespace(run_id="run", pending_for=pending.get, negotiations=[])
        runtime = SimpleNamespace(context=context, stopped=False,
                                  slippage_recording_ready=asyncio.Event())
        bridge = SimpleNamespace(product_agents=[SimpleNamespace(environment_runtime=runtime)])
        configuration = {"resource_id": "ur5e-4", "part_name": "gear_small",
                         "additional_condition": {"resource_id": "ur5e-3", "part_name": "KET4_Square_4mm"}}
        gate = SlippagePickupGate(runtime, configuration)
        tasks = [{"run_id": "run", "task_id": rid, "resource_id": rid, "event_name": "pick_approach",
                  "parameters": {"part_name": part}} for rid, part in gate.parts.items()]
        pending.update({task["task_id"]: task for task in tasks})
        workers = [asyncio.create_task(gate.wait(task)) for task in tasks]
        await asyncio.sleep(0)
        assert gate.released.is_set() and not any(worker.done() for worker in workers)
        with pytest.raises(ValueError, match="physically confirmed buffer"):
            await _release_slippage_capture(bridge)
        assert not runtime.slippage_recording_ready.is_set()
        runtime.slippage_initialization_evidence = {"run_id": "run", "status": "completed"}
        await _release_slippage_capture(bridge)
        await asyncio.gather(*workers)
        assert runtime.slippage_recording_ready.is_set()
        assert len(pending) == 2

    asyncio.run(scenario())


@pytest.mark.parametrize("fallen", [False, True])
def test_upright_buffer_initialization_observes_support_and_rejects_a_fallen_peg(monkeypatch, fallen):
    from copy import deepcopy
    from cais_spade_llm.recovery_framework import failure_effects
    from cais_spade_llm.recovery_framework.slippage_initialization import BUFFER, _prepare_part

    target = {"x": -.32, "y": .5, "z": 1.017,
              "qx": 0., "qy": 0., "qz": 0., "qw": 1.}
    physical = target.copy()
    buffer = {"handling_robot": "ur5e-3", "part_orientation_rpy": [0., 1.57079632679, 0.],
              "zone_pitch": .12, "guide_clear_width_m": .026,
              "zones": [{"zone": 4, "pose": [-.32, .5, 1.017, 0., 0., 0.]}]}
    def set_pose(_model, **values):
        physical.update(values)
        if fallen:
            physical.update(qy=2**-.5, qw=2**-.5)
        return {"success": True}
    owner = SimpleNamespace(
        _attached_model=None, set_entity_pose=Mock(side_effect=set_pose),
        _sync_part_collision=Mock(return_value=True),
        _last_command_evidence={"payload_collision": {"collision_scene_acknowledged": True}})
    monkeypatch.setattr(failure_effects, "_robot", lambda *_: SimpleNamespace(_controller=owner))
    monkeypatch.setattr(failure_effects, "_observe_part", lambda *_: deepcopy(physical))
    monkeypatch.setattr("cais_spade_llm.recovery_framework.slippage_initialization.time.sleep", lambda _: None)
    runtime = SimpleNamespace(stopped=False, context=SimpleNamespace(
        inputs={"scene": {BUFFER: buffer}}, geometry={"KET4_Square_4mm": {"model_name": "KET4_Square_4mm"}}))
    entry = {"part_name": "KET4_Square_4mm", "zone": 4, "orientation_quat": {
        "qx": 0., "qy": 0., "qz": 0., "qw": 1.}}
    if fallen:
        with pytest.raises(ValueError, match="retain its configured orientation"):
            _prepare_part(runtime, entry)
        owner._sync_part_collision.assert_not_called()
    else:
        evidence = _prepare_part(runtime, entry)
        assert evidence["pickup_ready"] and evidence["attachment"] is None
        assert len(evidence["support_observations"]) == 8
        assert abs(evidence["support_floor_error_m"]) <= .003
        assert evidence["orientation_error_rad"] == 0.
        assert {k: evidence["requested_pose"][k] for k in entry["orientation_quat"]} == entry["orientation_quat"]
        assert buffer["part_orientation_rpy"] == [0., 1.57079632679, 0.]
        owner._sync_part_collision.assert_called_once_with("KET4_Square_4mm")


def test_upright_peg_collision_geometry_reaches_the_nominal_square_grasp():
    from cais_spade_llm.recovery_framework.slippage_initialization import _bounds, _pose
    from cais_spade_llm.recovery_framework.part_collision import grasp_point_evidence, observed_part_boxes

    observation = {"x": -.32, "y": .5, "z": 1.017,
                   "qx": 0., "qy": 0., "qz": 0., "qw": 1.}
    boxes = observed_part_boxes("KET4_Square_4mm", _pose(observation))
    bounds = _bounds(boxes)
    assert bounds["max"][2] - bounds["min"][2] == pytest.approx(.05, abs=1e-6)
    assert bounds["min"][2] == pytest.approx(1.017, abs=1e-6)
    assert grasp_point_evidence(boxes, [-.32, .5, 1.070], .01)["payload_at_gripper"]


def test_initial_world_is_confirmed_before_the_run_becomes_pickup_ready(monkeypatch):
    from cais_spade_llm.recovery_framework import slippage_initialization as initialization

    async def scenario():
        part = "KET4_Square_4mm"
        entry = {"part_name": part, "resource_id": initialization.BUFFER, "zone": 4}
        context = SimpleNamespace(
            run_id="fresh-run", inputs={"scene": {}}, geometry={},
            resources={initialization.BUFFER: SimpleNamespace(valuation={"zone_4_part": part})},
            part_tracker={part: {"location": initialization.BUFFER}}, negotiations=[])
        runtime = SimpleNamespace(context=context, stopped=False, queue_save=Mock(),
                                  conveyor_fault=SimpleNamespace(configuration={"initial_conditions": [entry]}))
        monkeypatch.setattr(initialization, "_prepare_part", lambda *_: {"part_name": part})
        def confirm_world(_runtime):
            assert _runtime.slippage_initialization_evidence["status"] == "preparing"
            assert _runtime.slippage_initialization_evidence["parts"] == [{"part_name": part}]
            return {"acknowledged": True, "collision_object_ids": ["Buffer For Machined parts/link/zone_4_belt"]}
        monkeypatch.setattr(initialization, "_sync_initial_world", confirm_world)
        await initialization.prepare_slippage_initial_conditions(runtime)
        assert runtime.slippage_initialization_evidence["status"] == "completed"
        assert runtime.slippage_initialization_evidence["world_collision_scene"]["acknowledged"]
        assert context.negotiations[-1]["run_id"] == context.run_id
        runtime.queue_save.assert_called_once()
    asyncio.run(scenario())


@pytest.mark.parametrize("native_success", [True, False])
def test_slippage_release_uses_native_detachment_without_seating_or_fixture_handoff(native_success):
    from types import MethodType
    from cais_spade_llm.recovery_framework.failure_effects import _detach_slipped_part
    from cais_spade_llm.resources.robot.gazebo_pick_place_controller import GazeboPickPlaceController

    mating = {"model_name": "gear_small", "retain_fixture_attachment": True}
    native = SimpleNamespace(success=native_success, message="detached" if native_success else "rejected")
    service = SimpleNamespace(wait_for_service=Mock(return_value=True),
                              call_async=Mock(return_value=native))
    owner = SimpleNamespace(
        _simulation_mating_context=mating, _attached_model="gear_small",
        _attached_link="ur5e_4_rg2_base_link", _link_attacher_enabled=True,
        execution_mode="simulation", robot_model_name="dual_robot",
        wait_for_services=Mock(return_value=True), release_detach_retry_count=0,
        release_detach_timeout_sec=2., detach_timeout_sec=2., detach_max_link_attempts=1,
        primary_attach_link="ur5e_4_rg2_base_link",
        attach_link_candidates=["ur5e_4_rg2_base_link"],
        _detach_client=service, _detach_srv=SimpleNamespace(Request=SimpleNamespace),
        _wait_future=lambda future, **_: future, _sync_part_collision=Mock(return_value=True),
        _last_command_evidence={}, _log=lambda: Mock(),
        _release_open_fallback_allowed=Mock(return_value=False),
        _get_state_client=Mock(), _attach_part_to_assembly_board=Mock())
    owner._detach_part = MethodType(GazeboPickPlaceController._detach_part, owner)
    owner.detach_part = MethodType(GazeboPickPlaceController.detach_part, owner)
    result = _detach_slipped_part(owner, "gear_small")
    assert result["success"] is native_success
    assert owner._simulation_mating_context is mating
    owner._get_state_client.call_async.assert_not_called()
    owner._attach_part_to_assembly_board.assert_not_called()
    service.call_async.assert_called_once()
    request = service.call_async.call_args.args[0]
    assert request.model2_name == "gear_small" and request.link1_name == "ur5e_4_rg2_base_link"
    if native_success:
        assert owner._attached_model is None
        owner._sync_part_collision.assert_called_once_with("gear_small")
    else:
        assert owner._attached_model == "gear_small"
        owner._sync_part_collision.assert_not_called()


def test_slippage_release_rejects_an_unrelated_fixture_context():
    from cais_spade_llm.recovery_framework.failure_effects import _detach_slipped_part

    owner = SimpleNamespace(_simulation_mating_context={"model_name": "gear_large"}, detach_part=Mock())
    with pytest.raises(ValueError, match="unrelated mating context"):
        _detach_slipped_part(owner, "gear_small")
    owner.detach_part.assert_not_called()


@pytest.mark.parametrize("quaternion, expected", [
    ({"qx": 0., "qy": 0., "qz": .7071067811865476, "qw": .7071067811865476}, 0.),
    ({"qx": .7071067811865476, "qy": 0., "qz": 0., "qw": .7071067811865476}, 1.5707963267948966),
])
def test_upright_gear_check_measures_tilt_and_allows_yaw(quaternion, expected):
    from cais_spade_llm.recovery_framework.failure_effects import _upright_angle
    assert _upright_angle(quaternion) == pytest.approx(expected)


@pytest.mark.parametrize("quaternion", [
    {"qx": 0., "qy": 0., "qz": 0., "qw": 0.},
    {"qx": float("nan"), "qy": 0., "qz": 0., "qw": 1.},
])
def test_upright_gear_check_rejects_unusable_observations(quaternion):
    from cais_spade_llm.recovery_framework.failure_effects import _upright_angle
    with pytest.raises(ValueError):
        _upright_angle(quaternion)


@pytest.mark.parametrize("duration, expected", [(1., 5.), (8., 8.)])
def test_observed_lowering_retains_geometry_and_slows_derivatives(duration, expected):
    messages = pytest.importorskip("trajectory_msgs.msg")
    from cais_spade_llm.recovery_framework.placement_motion import _time_observed_lowering
    from cais_spade_llm.resources.robot.gazebo_pick_place_controller import GazeboPickPlaceController
    trajectory = messages.JointTrajectory(joint_names=["joint"])
    point = messages.JointTrajectoryPoint(positions=[.3], velocities=[.2], accelerations=[.1])
    point.time_from_start.sec = int(duration)
    trajectory.points = [point]
    owner = SimpleNamespace(_validate_simulation_trajectory=Mock())
    owner._scale_trajectory_timing = lambda solution, scale: GazeboPickPlaceController._scale_trajectory_timing(owner, solution, scale)
    evidence = {}
    _time_observed_lowering(owner, trajectory, evidence)
    scale = expected / duration
    assert list(point.positions) == [.3]
    assert point.velocities == pytest.approx([.2 / scale])
    assert point.accelerations == pytest.approx([.1 / scale ** 2])
    assert point.time_from_start.sec + point.time_from_start.nanosec / 1e9 == pytest.approx(expected)
    assert evidence["trajectory_duration_sec"] == pytest.approx(expected)
