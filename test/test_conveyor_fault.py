from __future__ import annotations

"""Observed simulation fault custody, availability, lifecycle and reset regressions."""

import asyncio
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from xml.etree import ElementTree

import pytest

from cais_spade_llm.product.environment import EnvironmentProductContext
from cais_spade_llm.recovery_framework.conveyor_fault import CHECKPOINT, ConveyorFault
from cais_spade_llm.recovery_framework.fault_visual import marker_sdf
from cais_spade_llm.ui import recovery_setup as settings


def configured_setup():
    setup = settings.default_setup()
    setup["failure_scenario"] = {
        "scenario": "Conveyor breakdown",
        "resource_id": "Conveyor",
        "checkpoint": CHECKPOINT,
        "mode": "once",
    }
    return setup


def runtime_at_pickup(*, evidence="resource", origin="M1", completed=True):
    setup = configured_setup()
    inputs = settings.validate_setup(setup)
    context = EnvironmentProductContext(
        inputs["scene"], inputs["product_order"], inputs["geometry"]
    )
    part = next(part for part in context.selected_parts if part.startswith("KET"))
    robot = context.resources["ur5e-1"]
    robot.valuation.update(resource_state="picked", held_part=part)
    context.part_tracker[part].update(location="ur5e-1", state="in_gripper")
    context.transitions.append(
        {
            "acknowledgement": {
                "resource_id": "ur5e-1",
                "event_name": "pick_grasp",
                "task_id": "observed_pickup",
                "run_id": context.run_id,
                "evidence": evidence,
                "parameters": {"part_name": part, "origin_resource_location": origin},
            },
            "before": {
                "M1": {
                    "resource_state": "completed" if completed else "processing",
                    "part_name": part,
                }
            },
            "observations": {
                "controller_result": {"status": "completed", "gripper_state": "closed"}
            },
        }
    )
    context.resources["Conveyor"].executors["advance_conveyor"] = Mock()
    runtime = SimpleNamespace(context=context, stopped=False, outcome={}, queue_save=Mock())

    def stop(reason, *, preserve_recovery=False):
        runtime.stopped = True
        for task_id in tuple(context.pending_tasks):
            context.cancel_pending(task_id)

    runtime.stop = stop

    async def cancel():
        return None

    runtime.cancel_owned = cancel
    runtime.conveyor_fault = ConveyorFault(runtime, setup)
    return runtime, part


def test_setup_supports_only_simulation_conveyor_at_the_documented_checkpoint():
    setup = configured_setup()
    assert settings.startup_block_reason(setup) == ""
    setup["execution_mode"] = "physical"
    assert "execution not integrated" in settings.startup_block_reason(setup)
    with pytest.raises(ValueError, match="simulation"):
        ConveyorFault(SimpleNamespace(), setup)
    setup["execution_mode"] = "simulation"
    setup["failure_scenario"]["checkpoint"] = "before_execute"
    assert "execution not integrated" in settings.startup_block_reason(setup)


def _printed_pickup(runtime):
    context = runtime.context
    context.resources['ur5e-4'].valuation.update(resource_state='picked', held_part='gear_small')
    context.part_tracker['gear_small'].update(location='ur5e-4', state='in_gripper')
    context.transitions.append({
        'acknowledgement': {'resource_id': 'ur5e-4', 'event_name': 'pick_grasp',
            'task_id': 'printed_pickup', 'run_id': context.run_id, 'evidence': 'resource',
            'parameters': {'part_name': 'gear_small', 'origin_resource_location': '3D Printing Station'}},
        'before': {'3D Printing Station': {'resource_state': 'completed'}},
        'observations': {'controller_result': {'status': 'completed', 'gripper_state': 'closed'}},
    })


def _conveyor_condition(runtime, part):
    runtime.conveyor_fault.configuration.update(
        part_name=part, additional_condition={'resource_id': 'ur5e-4', 'part_name': 'gear_small'})


def test_conveyor_example_configures_the_M1_pickup_without_a_held_peer():
    from cais_spade_llm.recovery_framework.part_slippage_example import example_setup

    setup = example_setup(scenario='Conveyor breakdown', outline_mode='manual', primitive_mode='manual')
    assert setup['failure_scenario']['part_name'] == 'KET4_Square_4mm'
    assert 'additional_condition' not in setup['failure_scenario']
    assert settings.validate_setup(setup)
    setup['failure_scenario']['part_name'] = 'unobserved_part'
    with pytest.raises(ValueError, match='eligible for ur5e-1'):
        settings.validate_setup(setup)


@pytest.mark.parametrize('condition', [
    {'resource_id': 'Conveyor', 'part_name': 'gear_small'},
    {'resource_id': 'ur5e-4', 'part_name': 'KET4_Square_4mm'},
    {'resource_id': 'ur5e-4', 'part_name': 'unobserved_part'},
    {'resource_id': 'unconfigured_resource', 'part_name': 'gear_small'},
])
def test_conveyor_setup_rejects_invalid_additional_condition(condition):
    from cais_spade_llm.recovery_framework.part_slippage_example import example_setup

    setup = example_setup(scenario='Conveyor breakdown')
    setup['failure_scenario']['additional_condition'] = condition
    with pytest.raises(ValueError, match='Additional condition'):
        settings.validate_setup(setup)


@pytest.mark.parametrize('first', ['ur5e-1', 'ur5e-4'])
def test_conveyor_waits_for_both_pickups_in_either_order_and_retains_custody(monkeypatch, first):
    runtime, part = runtime_at_pickup()
    context = runtime.context
    _conveyor_condition(runtime, part)
    primary = context.transitions.pop()
    context.resources['ur5e-1'].valuation.update(resource_state='idle', held_part=None)
    context.part_tracker[part]['location'] = 'M1'

    def primary_pickup():
        context.resources['ur5e-1'].valuation.update(resource_state='picked', held_part=part)
        context.part_tracker[part]['location'] = 'ur5e-1'
        context.transitions.append(primary)

    (primary_pickup if first == 'ur5e-1' else lambda: _printed_pickup(runtime))()
    assert not asyncio.run(runtime.conveyor_fault.after_acknowledgement())
    assert runtime.conveyor_fault.holds_task({'resource_id': first})
    peer = 'ur5e-4' if first == 'ur5e-1' else 'ur5e-1'
    assert not runtime.conveyor_fault.holds_task({'resource_id': peer})
    assert not runtime.conveyor_fault.holds_task({'resource_id': 'ur5e-3'})
    (primary_pickup if peer == 'ur5e-1' else lambda: _printed_pickup(runtime))()
    visual = Mock(return_value={'status': 'completed'})
    monkeypatch.setattr('cais_spade_llm.recovery_framework.conveyor_fault.marker', visual)
    assert asyncio.run(runtime.conveyor_fault.after_acknowledgement())
    assert not asyncio.run(runtime.conveyor_fault.after_acknowledgement())
    assert context.resources['ur5e-1'].valuation['held_part'] == part
    assert context.resources['ur5e-4'].valuation['held_part'] == 'gear_small'
    assert context.part_tracker[part]['location'] == 'ur5e-1'
    assert context.part_tracker['gear_small']['location'] == 'ur5e-4'
    assert set(runtime.conveyor_fault.evidence['pickups']) == {'ur5e-1', 'ur5e-4'}
    assert runtime.conveyor_fault.evidence['injection_status'] == 'completed'
    assert visual.call_count == 1


@pytest.mark.parametrize('change', ['run_id', 'origin', 'part', 'observations', 'evidence', 'later_ack', 'pending', 'primary_part'])
def test_conveyor_rejects_unacknowledged_or_changed_peer_pickup(change):
    runtime, part = runtime_at_pickup()
    _conveyor_condition(runtime, part)
    _printed_pickup(runtime)
    peer = runtime.context.transitions[-1]
    acknowledgement = peer['acknowledgement']
    if change == 'run_id':
        acknowledgement['run_id'] = 'another_run'
    elif change == 'origin':
        acknowledgement['parameters']['origin_resource_location'] = 'Buffer For Machined parts'
    elif change == 'part':
        acknowledgement['parameters']['part_name'] = part
    elif change == 'observations':
        peer['observations'] = {}
    elif change == 'evidence':
        acknowledgement['evidence'] = 'simulated'
    elif change == 'later_ack':
        later = deepcopy(peer)
        later['acknowledgement']['event_name'] = 'place_approach'
        runtime.context.transitions.append(later)
    elif change == 'pending':
        runtime.context.pending_tasks['moving_peer'] = {'resource_id': 'ur5e-4'}
    else:
        runtime.conveyor_fault.configuration['part_name'] = 'gear_small'
    assert runtime.conveyor_fault.checkpoint() is None
    assert not asyncio.run(runtime.conveyor_fault.after_acknowledgement())
    assert not runtime.stopped


@pytest.mark.parametrize(
    "fields", [{"evidence": "simulated"}, {"origin": "M2"}, {"completed": False}]
)
def test_trigger_rejects_unobserved_or_wrong_checkpoint_without_mutating_custody(fields):
    runtime, part = runtime_at_pickup(**fields)
    before = runtime.context.snapshot()
    with pytest.raises(ValueError, match="Waiting"):
        asyncio.run(runtime.conveyor_fault.trigger())
    assert runtime.context.snapshot() == before
    assert not runtime.stopped
    assert runtime.context.resources["Conveyor"].executors


def test_disarm_prevents_automatic_injection_and_rearm_requires_active_run(monkeypatch):
    runtime, part = runtime_at_pickup()
    fault = runtime.conveyor_fault
    fault.arm(False)
    assert not asyncio.run(fault.after_acknowledgement())
    assert not runtime.stopped
    runtime.stopped = True
    with pytest.raises(ValueError, match="active"):
        fault.arm(True)


@pytest.mark.parametrize("visual_status", ["completed", "failed"])
def test_breakdown_is_once_and_retains_custody_even_when_visualization_fails(
    monkeypatch, visual_status
):
    runtime, part = runtime_at_pickup()
    fault = runtime.conveyor_fault
    visual = Mock(return_value={"status": visual_status})
    monkeypatch.setattr("cais_spade_llm.recovery_framework.conveyor_fault.marker", visual)
    before = deepcopy(runtime.context.part_tracker)
    # Without a configured peer the checkpoint holds no robot; it latches on the M1 pickup.
    assert not any(fault.holds_task({"resource_id": rid}) for rid in ("ur5e-1", "ur5e-3", "ur5e-4", "KMR"))
    assert asyncio.run(fault.after_acknowledgement())
    assert set(fault.evidence["pickups"]) == {"ur5e-1"}
    assert fault.evidence["pickups"]["ur5e-1"]["acknowledgement"]["task_id"] == "observed_pickup"
    assert runtime.context.fault_reports["Conveyor"]["available"] is False
    assert runtime.stopped and runtime.outcome["failed_resource"] == "Conveyor"
    assert runtime.context.part_tracker == before
    assert runtime.context.snapshot()["ur5e-1"]["held_part"] == part
    assert not runtime.context.resources["Conveyor"].executors
    assert runtime.context.inputs["scene"]["Conveyor"]["transport_enabled"] is False
    assert runtime.context.unavailable_resources == {"Conveyor"}
    task = next(
        event
        for event in runtime.context.models["ur5e-1"]["events"]
        if event["event_name"] == "place_release" and "Conveyor" in event["participants"]
    )
    assert not runtime.context.allows_task(
        {
            "resource_id": "ur5e-1",
            "event_id": task["event_id"],
            "parameters": {"destination_location": "Conveyor"},
        }
    )
    again = asyncio.run(fault.trigger())
    assert again["status"] == "triggered" and visual.call_count == 1
    with pytest.raises(ValueError, match="reset"):
        fault.arm(True)
    assert again["visual"]["status"] == visual_status


def test_marker_has_no_collision_and_uses_configured_conveyor_geometry():
    scene = settings.validate_setup(settings.default_setup())["scene"]
    model = ElementTree.fromstring(marker_sdf(scene)).find("model")
    assert model.findtext("static") == "true"
    assert not model.findall(".//collision")
    assert len(model.findall(".//visual")) == 5
    assert model.findtext("pose") == " ".join(str(v) for v in scene["Conveyor"]["world_pose"])
    assert "CAIS/ConveyorBreakdown" in marker_sdf(scene)


def test_ur5e_1_sign_is_raised_without_changing_the_outline_or_collision_geometry():
    from cais_spade_llm.recovery_framework.fault_visual import marker_geometry

    scene = settings.validate_setup(settings.default_setup())["scene"]
    scene["_failure_marker"] = {"scenario": "ur5e-1 breakdown", "resource_id": "ur5e-1"}
    geometry = marker_geometry(scene)
    model = ElementTree.fromstring(marker_sdf(scene)).find("model")
    label = model.find("./link/visual[@name='failure_label']")
    assert geometry["label"] == "ur5e-1 breakdown"
    assert geometry["label_height"] == 1.15
    assert label.findtext("pose") == "0 0 1.2 0 0 0"
    assert label.findtext("geometry/box/size") == "1.4 .025 .28"
    assert model.findtext("pose") == " ".join(str(v) for v in geometry["pose"])
    assert not model.findall(".//collision")
    for visual in model.findall("./link/visual"):
        if visual.attrib["name"] != "failure_label":
            assert float(visual.findtext("pose").split()[2]) == pytest.approx(.09)


@pytest.mark.parametrize("cleared", [True, False])
def test_explicit_world_reset_clears_latch_only_after_marker_confirmation(monkeypatch, cleared):
    from cais_spade_llm.ui.bridge import SystemBridge

    runtime, _ = runtime_at_pickup()
    runtime.conveyor_fault.status = "triggered"
    bridge = SimpleNamespace(
        _GAZEBO_PROCESS_NAMES=(),
        _any_running=lambda names: True,
        system_running=False,
        _starting=False,
        _conveyor_fault_owner=lambda: runtime,
        _recovery_framework_gazebo=lambda: True,
        ros2_stop=Mock(return_value=None),
        ros2_start=Mock(return_value=None),
        simulation_start_ready=Mock(return_value=(True, "")),
        _ros2_command_output=Mock(return_value=(True, "/reset_world")),
        _restore_gazebo_scene_in_place=Mock(return_value=([], [])),
        ros2_exec=Mock(return_value=(True, "reset")),
    )
    monkeypatch.setattr(
        "cais_spade_llm.recovery_framework.conveyor_fault.marker",
        Mock(return_value={"status": "completed" if cleared else "failed", "visible": False}),
    )
    ok, detail = SystemBridge.ros2_reset_gazebo_environment(bridge)
    assert ok is cleared
    bridge._restore_gazebo_scene_in_place.assert_not_called()
    bridge.ros2_start.assert_called_once_with("gazebo_dual")
    assert runtime.conveyor_fault.status == ("reset" if cleared else "triggered")


def test_public_start_cannot_discard_latched_breakdown_after_agent_teardown():
    from cais_spade_llm.ui.bridge import SystemBridge

    runtime, _ = runtime_at_pickup()
    runtime.conveyor_fault.status = "triggered"
    bridge = SystemBridge.__new__(SystemBridge)
    bridge.product_agents = []
    bridge._conveyor_fault_runtime = runtime
    bridge.system_running = False
    bridge._starting = False
    bridge.last_error = ""
    asyncio.run(bridge.start_system())
    assert "Reset Gazebo or Reset All" in bridge.last_error
    assert bridge.get_conveyor_fault()["status"] == "triggered"


def test_bridge_fault_controls_use_the_agent_runtime_loop(monkeypatch):
    from cais_spade_llm.ui.bridge import SystemBridge

    runtime, _ = runtime_at_pickup()
    bridge = SystemBridge.__new__(SystemBridge)
    bridge.product_agents = []
    bridge._conveyor_fault_runtime = runtime
    bridge.execution_mode = "simulation"

    async def dispatch(coroutine):
        return await coroutine

    bridge._run_on_agent_runtime = AsyncMock(side_effect=dispatch)
    monkeypatch.setattr(
        "cais_spade_llm.recovery_framework.conveyor_fault.marker",
        Mock(return_value={"status": "completed", "present": True}),
    )

    async def scenario():
        assert (await bridge.arm_conveyor_fault(False))["status"] == "disarmed"
        assert (await bridge.arm_conveyor_fault(True))["status"] == "armed"
        assert (await bridge.trigger_conveyor_fault())["status"] == "triggered"
        assert bridge._run_on_agent_runtime.await_count == 3

    asyncio.run(scenario())


def runtime_with_failure(scenario, *, robot="ur5e-3"):
    from cais_spade_llm.recovery_framework.failure_checkpoints import CHECKPOINTS

    runtime, part = runtime_at_pickup()
    context = runtime.context
    context.transitions.clear()
    rid = {"ur5e-1 breakdown": "ur5e-1",
           "Machining breakdown during part processing": "M1",
           "Part slippage": robot}[scenario]
    setup = configured_setup()
    setup["failure_scenario"] = {"scenario": scenario, "resource_id": rid,
                                 "checkpoint": CHECKPOINTS[scenario], "mode": "once"}
    runtime.resource_agents = []
    runtime.retained_paths = {"part": {"path": [{"event_name": "place_insert"}]}}
    if scenario == "Part slippage":
        part = "KET4_Square_4mm" if robot == "ur5e-3" else "gear_large"
        setup["failure_scenario"] = settings.slippage_example(context.models, robot, part)
        setup["failure_scenario"]["drop_pose"] = {
            "x": 0.0, "y": -0.2 if robot == "ur5e-3" else 0.2, "z": 1.04,
        }
        other = setup["failure_scenario"]["additional_condition"]
        for owner, held in ((robot, part), (other["resource_id"], other["part_name"])):
            context.resources[owner].valuation.update(resource_state="picked", held_part=held)
            context.part_tracker[held].update(state="in_gripper", location=owner)
            context.transitions.append({
                "acknowledgement": {"resource_id": owner, "event_name": "pick_grasp",
                                    "task_id": owner + "_pickup", "evidence": "resource",
                                    "run_id": context.run_id,
                                    "parameters": {"part_name": held}},
                "observations": {"controller_result": {"status": "completed"}},
            })
            controller = SimpleNamespace(
                get_current_pose=Mock(return_value={"success": True, "pose": {"x": 0, "y": 0, "z": 1.2}}),
                open_gripper=Mock(return_value=True),
                detach_part=Mock(return_value={"success": True}),
                set_entity_pose=Mock(return_value={"success": True}),
                _sync_part_collision=Mock(return_value=True),
                _last_command_evidence={"collision_scene_acknowledged": True},
            )
            runtime.resource_agents.append(SimpleNamespace(
                agent_name=owner, execution_mode="simulation", _controller=controller,
                _held_part=held, _gripper_state="closed",
            ))
    else:
        context.resources["M1"].valuation.update(
            resource_state="completed" if scenario == "ur5e-1 breakdown" else "loaded",
            part_name=part,
        )
        context.part_tracker[part].update(location="M1")
        context.resources["ur5e-1"].valuation.update(resource_state="idle", held_part=None)
        context.transitions.append({
            "acknowledgement": {"resource_id": "M1", "event_name": "machine_part",
                                "task_id": "machined", "evidence": "resource",
                                "run_id": context.run_id,
                                "parameters": {"part_name": part}},
            "observations": {"correct_part_present": True, "process_completed": True},
        })
        runtime.resource_agents.append(SimpleNamespace(
            agent_name="ur5e-1", execution_mode="simulation",
            _controller=SimpleNamespace(get_current_pose=Mock(return_value={
                "success": True, "pose": {"x": -4, "y": 2, "z": 1},
            })),
        ))
        if scenario == "ur5e-1 breakdown":
            robot_values = context.resources["ur5e-1"].valuation
            robot_values.update(resource_state="at_pick", resource_location="M1", held_part=None)
            context.resources["ur5e-1"].executors.update(move_home=Mock(), pick_grasp=Mock())
            runtime.resource_agents[0]._current_state = "at_pick"
            context.transitions.append({
                "acknowledgement": {"resource_id": "ur5e-1", "event_name": "pick_approach",
                                    "task_id": "approached", "evidence": "resource",
                                    "run_id": context.run_id,
                                    "parameters": {"part_name": part, "origin_resource_location": "M1"}},
                "before": {"M1": deepcopy(context.resources["M1"].valuation)},
                "observations": {"controller_result": {"status": "completed"}},
            })
    runtime.conveyor_fault = ConveyorFault(runtime, setup)
    return runtime, part, setup


@pytest.mark.parametrize("robot", ["ur5e-3", "ur5e-4"])
def test_slippage_both_directions_preserve_other_custody_and_obligations(monkeypatch, robot):
    runtime, part, setup = runtime_with_failure("Part slippage", robot=robot)
    fault = runtime.conveyor_fault
    other = setup["failure_scenario"]["additional_condition"]
    other_before = deepcopy(runtime.context.part_tracker[other["part_name"]])
    tracker_before, values_before = deepcopy(runtime.context.part_tracker), runtime.context.snapshot()
    observed = {"x": 0.01, "y": setup["failure_scenario"]["drop_pose"]["y"], "z": 1.035,
                "qx": 0., "qy": 0., "qz": 0., "qw": 1.}
    monkeypatch.setattr("cais_spade_llm.recovery_framework.failure_effects._observe_part",
                        Mock(return_value=observed))
    monkeypatch.setattr("cais_spade_llm.recovery_framework.conveyor_fault.marker",
                        Mock(return_value={"status": "completed"}))
    assert fault.holds_task({"resource_id": robot, "event_name": "place_approach"})
    asyncio.run(fault.trigger())
    assert fault.evidence["injection_status"] == "completed"
    # The robot reports its own lost part; the product's beliefs wait for grounding.
    agent = next(owner for owner in runtime.resource_agents if owner.agent_name == robot)
    assert (agent._held_part, agent._current_state) == (None, "failed")
    assert runtime.context.fault_reports[robot]["reported_state"] == "failed"
    assert runtime.context.part_tracker == tracker_before and runtime.context.snapshot() == values_before
    assert runtime.context.part_tracker[part]["location"] == robot
    assert runtime.context.part_tracker[other["part_name"]] == other_before
    assert runtime.context.resources[other["resource_id"]].valuation["held_part"] == other["part_name"]
    assert set(fault.evidence["pickups"]) == {"ur5e-3", "ur5e-4"}
    assert fault.evidence["continuations"] == runtime.retained_paths
    assert fault.evidence["requested_drop_pose"] != fault.evidence["observed_drop_pose"]
    assert runtime.stopped
    with pytest.raises(ValueError, match="No pending task"):
        runtime.context.acknowledge({"task_id": "late_ack"})


@pytest.mark.parametrize("robot", ["ur5e-3", "ur5e-4"])
def test_slippage_requires_both_observed_pickups_and_no_pending_robot_task(robot):
    runtime, part, setup = runtime_with_failure("Part slippage", robot=robot)
    fault = runtime.conveyor_fault
    assert fault.checkpoint()
    runtime.context.pending_tasks["racing"] = {"resource_id": robot}
    assert fault.checkpoint() is None
    runtime.context.pending_tasks.clear()
    runtime.context.transitions[-1]["acknowledgement"]["evidence"] = "simulated"
    assert fault.checkpoint() is None
    before = runtime.context.snapshot()
    with pytest.raises(ValueError, match="Waiting"):
        asyncio.run(fault.trigger())
    assert runtime.context.snapshot() == before


@pytest.mark.parametrize("stage", ["open", "detach", "set_pose", "observation", "collision"])
def test_partial_slippage_keeps_latch_and_reports_uncertainty(monkeypatch, stage):
    runtime, part, _ = runtime_with_failure("Part slippage")
    controller = runtime.resource_agents[0]._controller
    operations = {"open": controller.open_gripper, "detach": controller.detach_part,
                  "set_pose": controller.set_entity_pose, "collision": controller._sync_part_collision}
    if stage in operations:
        operations[stage].return_value = False if stage in {"open", "collision"} else {"success": False}
    observe = Mock(return_value={"x": 0., "y": -.2, "z": 1.04})
    if stage == "observation":
        observe.side_effect = ValueError("no observation")
    monkeypatch.setattr("cais_spade_llm.recovery_framework.failure_effects._observe_part", observe)
    monkeypatch.setattr("cais_spade_llm.recovery_framework.conveyor_fault.marker",
                        Mock(return_value={"status": "failed"}))
    fault = runtime.conveyor_fault
    asyncio.run(fault.trigger())
    assert fault.status == "triggered" and runtime.stopped
    assert fault.evidence["injection_status"] == "failed"
    assert fault.evidence["physical_state_reconciliation_required"]
    assert "ur5e-3" in runtime.context.unavailable_resources
    assert runtime.context.resources["ur5e-3"].valuation["held_part"] == part
    assert ("ur5e-3" in runtime.context.fault_reports) == (stage not in {"open", "detach"})
    if stage in {"open", "detach"}:
        controller.set_entity_pose.assert_not_called()


def _frozen_machine_checkpoint(monkeypatch, runtime):
    """Stand in for the Gazebo machine capture and record what it observed."""
    seen = []

    async def capture(owner, failure):
        assert owner.recovery_checkpoint_capturing and owner.stopped
        seen.append(deepcopy(owner.context.fault_reports.get("ur5e-1")))

    monkeypatch.setattr("cais_spade_llm.recovery_framework.recovery_observations.capture_machine_checkpoint", capture)
    return seen


def test_robot_breakdown_retains_completed_part_in_M1_and_observed_pose(monkeypatch):
    runtime, part, _ = runtime_with_failure("ur5e-1 breakdown")
    monkeypatch.setattr("cais_spade_llm.recovery_framework.conveyor_fault.marker",
                        Mock(return_value={"status": "completed"}))
    seen = _frozen_machine_checkpoint(monkeypatch, runtime)
    before = deepcopy(runtime.context.part_tracker)
    executors = dict(runtime.context.resources["ur5e-1"].executors)
    asyncio.run(runtime.conveyor_fault.trigger())
    # The gripper's own report exists when the world is frozen for capture.
    assert [row["source"] for row in seen] == ["gripper object detection"]
    assert runtime.conveyor_fault.evidence["injection_status"] == "completed"
    assert runtime.context.part_tracker == before
    assert runtime.context.part_tracker[part]["location"] == "M1"
    assert runtime.context.resources["M1"].valuation["resource_state"] == "completed"
    assert runtime.context.unavailable_resources == set()
    assert runtime.context.resources["ur5e-1"].valuation["resource_state"] == "at_pick"
    assert runtime.context.fault_reports["ur5e-1"]["reported_state"] == "failed"
    assert runtime.resource_agents[0]._current_state == "failed"
    assert runtime.context.resources["ur5e-1"].executors == executors
    assert runtime.conveyor_fault.evidence["resource_values_before"]["ur5e-1"]["resource_state"] == "at_pick"
    assert runtime.conveyor_fault.evidence["pick_approach"]["acknowledgement"]["task_id"] == "approached"
    assert runtime.conveyor_fault.evidence["machine_part"]["acknowledgement"]["task_id"] == "machined"
    executors["pick_grasp"].assert_not_called()
    executors["move_home"].assert_not_called()
    assert runtime.conveyor_fault.evidence["robot_pose"] == {"x": -4, "y": 2, "z": 1}


@pytest.mark.parametrize("defect", ["processing_only", "origin", "part", "run", "observations",
                                  "custody", "state", "pending", "machine_run", "machine_before"])
def test_robot_breakdown_requires_current_acknowledged_M1_pick_approach(defect):
    runtime, part, _ = runtime_with_failure("ur5e-1 breakdown")
    context = runtime.context
    approach = context.transitions[-1]
    if defect == "processing_only":
        context.transitions.pop()
        context.resources["ur5e-1"].valuation["resource_state"] = "idle"
    elif defect == "origin":
        approach["acknowledgement"]["parameters"]["origin_resource_location"] = "Storage"
    elif defect == "part":
        approach["acknowledgement"]["parameters"]["part_name"] = "gear_small"
    elif defect == "run":
        approach["acknowledgement"]["run_id"] = "previous_run"
    elif defect == "observations":
        approach["observations"] = {}
    elif defect == "custody":
        context.resources["ur5e-1"].valuation["held_part"] = part
    elif defect == "state":
        context.resources["ur5e-1"].valuation["resource_state"] = "picked"
    elif defect == "pending":
        context.pending_tasks["grasp"] = {"resource_id": "ur5e-1", "event_name": "pick_grasp"}
    elif defect == "machine_run":
        context.transitions[0]["acknowledgement"]["run_id"] = "previous_run"
    elif defect == "machine_before":
        approach["before"]["M1"]["resource_state"] = "loaded"
    assert runtime.conveyor_fault.checkpoint() is None
    assert not runtime.stopped


def test_robot_breakdown_latches_after_approach_before_next_dispatch(monkeypatch):
    runtime, _, _ = runtime_with_failure("ur5e-1 breakdown")
    monkeypatch.setattr("cais_spade_llm.recovery_framework.conveyor_fault.marker",
                        Mock(return_value={"status": "completed"}))
    _frozen_machine_checkpoint(monkeypatch, runtime)
    assert asyncio.run(runtime.conveyor_fault.after_acknowledgement()) is True
    assert runtime.stopped
    assert asyncio.run(runtime.conveyor_fault.after_acknowledgement()) is False
    runtime.context.resources["ur5e-1"].executors["pick_grasp"].assert_not_called()


def test_failed_robot_retains_recorded_home_motion_capability(monkeypatch):
    from cais_spade_llm.resources.robot.recovery_capability import check_robot_primitive_capability

    runtime, _, _ = runtime_with_failure("ur5e-1 breakdown")
    monkeypatch.setattr("cais_spade_llm.recovery_framework.conveyor_fault.marker",
                        Mock(return_value={"status": "completed"}))
    _frozen_machine_checkpoint(monkeypatch, runtime)
    asyncio.run(runtime.conveyor_fault.trigger())
    snapshot = {**runtime.context.resources["ur5e-1"].valuation,
                "resource_state": runtime.context.fault_reports["ur5e-1"]["reported_state"],
                "named_poses": ["home"], "controller_ready": True}
    result = check_robot_primitive_capability(primitive="move_to_named_pose", params={"pose_name": "home"},
        task={}, recovery_snapshot=snapshot, resource_evidence={"source": {"run_id": runtime.context.run_id}},
        static_capabilities={"named_poses": ["home"]}, primitive_catalog=[], resource_jid="ur5e-1@localhost")
    assert result["feasibility_status"] == "FEASIBLE"
    assert result["evidence"]["command_sent"] is False
    runtime.context.resources["ur5e-1"].executors["move_home"].assert_not_called()


@pytest.mark.parametrize("scenario", ["ur5e-1 breakdown", "Machining breakdown during part processing", "Part slippage"])
def test_new_marker_uses_resource_or_observation_and_has_no_collision(scenario):
    from cais_spade_llm.recovery_framework.fault_visual import marker_geometry

    runtime, _, _ = runtime_with_failure(scenario)
    fault = runtime.conveyor_fault
    if scenario == "Part slippage":
        fault.evidence["observed_drop_pose"] = {"x": 0.12, "y": -0.23, "z": 1.035}
    scene = fault.marker_scene()
    geometry = marker_geometry(scene)
    xml = ElementTree.fromstring(marker_sdf(scene))
    assert not xml.findall(".//collision")
    assert geometry["label"].startswith(scenario)
    assert geometry["model_name"] == xml.find("model").attrib["name"]
    if scenario == "Part slippage":
        assert geometry["pose"][:3] == [0.12, -0.23, 1.035]


def test_machine_worker_fault_rejects_stale_evidence_and_never_completes_process(monkeypatch):
    runtime, part, _ = runtime_with_failure("Machining breakdown during part processing")
    task = {"resource_id": "M1", "event_name": "machine_part", "task_id": "process",
            "run_id": runtime.context.run_id, "parameters": {"part_name": part}}
    runtime.context.pending_tasks["process"] = task
    evidence = {**{key: task[key] for key in ("run_id", "task_id", "resource_id")},
                "part_name": part, "checkpoint": "during_processing_halfway",
                "source": "gazebo_workholding_observation", "processing_time_sec": 5.,
                "simulation_elapsed_sec": 2.5, "process_completed": False,
                "observed_pose": {"x": -4., "y": 1., "z": 1.04}}
    monkeypatch.setattr("cais_spade_llm.recovery_framework.conveyor_fault.marker",
                        Mock(return_value={"status": "completed"}))
    with pytest.raises(ValueError, match="Stale"):
        asyncio.run(runtime.conveyor_fault.accept_machine_failure(task, {
            "failure_injection": {**evidence, "run_id": "old_run"},
        }))
    before = deepcopy(runtime.context.part_tracker)
    runtime.conveyor_fault.arm(False)
    assert asyncio.run(runtime.conveyor_fault.accept_machine_failure(task, {"failure_injection": evidence}))
    assert runtime.context.part_tracker == before
    assert runtime.context.resources["M1"].valuation["resource_state"] == "loaded"
    assert not runtime.context.pending_tasks
    assert runtime.conveyor_fault.evidence["simulation_elapsed_sec"] == 2.5
    assert not runtime.conveyor_fault.evidence["process_completed"]


@pytest.mark.parametrize("capture_error", [None, "checkpoint observation conflict"])
def test_machine_checkpoint_precedes_cancellation_and_retains_capture_error(monkeypatch, capture_error):
    runtime, part, _ = runtime_with_failure("Machining breakdown during part processing")
    fault = runtime.conveyor_fault
    fault.configuration["checkpoint"] = "during_processing"
    fault.checkpoint = Mock(return_value={"source": "gazebo_workholding_observation"})
    runtime.context.pending_tasks["job"] = {"task_id": "job", "part_name": part}
    sequence = []
    original_stop = runtime.stop

    def stop(*args, **kwargs):
        sequence.append("stop")
        original_stop(*args, **kwargs)

    async def capture(owner, evidence):
        sequence.append("capture")
        assert owner.stopped and owner.recovery_checkpoint_capturing
        assert "job" in owner.context.pending_tasks
        assert "stop" not in sequence
        owner.recovery_capture_frozen = True
        if capture_error:
            raise ValueError(capture_error)
        owner.recovery_checkpoint_observations = {"processing_observations": {"M2": {"job_state": "running"}}}

    async def cancel():
        sequence.append("cancel")
        assert runtime.recovery_capture_frozen

    def unfreeze(paused):
        assert paused is False
        sequence.append("unfreeze")

    runtime.stop, runtime.cancel_owned = stop, cancel
    runtime.recovery_observation_preparation = SimpleNamespace(reader=SimpleNamespace(set_world_paused=unfreeze))
    runtime.part_slippage_recovery = SimpleNamespace(begin=Mock())
    monkeypatch.setattr("cais_spade_llm.recovery_framework.recovery_observations.capture_machine_checkpoint", capture)
    monkeypatch.setattr("cais_spade_llm.recovery_framework.conveyor_fault.marker", Mock(return_value={"status": "completed"}))
    asyncio.run(fault.trigger())
    assert sequence == ["capture", "stop", "cancel", "unfreeze"]
    assert not runtime.recovery_capture_frozen and not runtime.recovery_checkpoint_capturing
    assert not runtime.context.pending_tasks
    if capture_error:
        assert fault.evidence["injection_status"] == "failed"
        assert fault.evidence["error"] == capture_error
    else:
        assert fault.evidence["injection_status"] == "completed"
        assert runtime.recovery_checkpoint_observations["processing_observations"]["M2"]["job_state"] == "running"
        assert fault.evidence["processing_shutdown"] == {}


@pytest.mark.parametrize("scenario", ["Conveyor breakdown", "ur5e-1 breakdown", "Part slippage"])
def test_old_run_acknowledgements_cannot_establish_a_checkpoint(scenario):
    runtime = runtime_at_pickup()[0] if scenario == "Conveyor breakdown" else runtime_with_failure(scenario)[0]
    runtime.context.transitions[-1]["acknowledgement"]["run_id"] = "previous_run"
    assert runtime.conveyor_fault.checkpoint() is None
    with pytest.raises(ValueError, match="Waiting"):
        asyncio.run(runtime.conveyor_fault.trigger())
    assert not runtime.stopped


def test_cancelling_slippage_caller_keeps_effect_recording_and_blocks_early_reset(monkeypatch):
    from cais_spade_llm.recovery_framework.conveyor_fault import reset_fault_scene

    async def exercise():
        runtime, _, _ = runtime_with_failure("Part slippage")
        entered, finish = asyncio.Event(), asyncio.Event()

        async def slip(_runtime, _configuration, evidence):
            evidence["detach"] = {"success": True}
            entered.set()
            await finish.wait()
            evidence["observed_drop_pose"] = {"x": 0., "y": -.2, "z": 1.04}

        monkeypatch.setattr("cais_spade_llm.recovery_framework.failure_effects.slip_part", slip)
        visual = Mock(return_value={"status": "completed"})
        monkeypatch.setattr("cais_spade_llm.recovery_framework.conveyor_fault.marker", visual)
        caller = asyncio.create_task(runtime.conveyor_fault.trigger())
        await entered.wait()
        caller.cancel()
        with pytest.raises(asyncio.CancelledError):
            await caller
        fault = runtime.conveyor_fault
        assert fault.status == "triggered" and runtime.stopped
        bridge = SimpleNamespace(ros2_stop=Mock())
        ok, reason = reset_fault_scene(bridge, runtime)
        assert not ok and "still recording" in reason
        bridge.ros2_stop.assert_not_called()
        assert (await fault.trigger())["status"] == "triggered"
        finish.set()
        await fault._injection_task
        assert fault.evidence["injection_status"] == "completed"
        assert fault.evidence["detach"]["success"]
        visual.assert_called_once()

    asyncio.run(exercise())


@pytest.mark.parametrize("observed", [True, False])
def test_stop_retains_worker_clock_interruption_before_its_reply(monkeypatch, observed):
    import json
    from pathlib import Path

    runtime, part, _ = runtime_with_failure("Machining breakdown during part processing")
    fault = runtime.conveyor_fault
    task = {"resource_id": "M1", "event_name": "machine_part", "task_id": "processing",
            "run_id": runtime.context.run_id, "parameters": {"part_name": part}}
    runtime.context.pending_tasks["processing"] = task
    request = fault.worker_request(task)
    evidence = {key: request[key] for key in ("run_id", "resource_id", "task_id", "checkpoint")}
    evidence.update(part_name=part, processing_time_sec=5., simulation_elapsed_sec=2.5,
                    process_completed=False)
    if observed:
        evidence.update(source="gazebo_workholding_observation", observation_status="completed",
                        observed_pose={"x": -4., "y": 1., "z": 1.04})
    else:
        evidence.update(source="gazebo_processing_checkpoint", observation_status="pending")
    Path(request["evidence_path"]).write_text(json.dumps(evidence))
    before = deepcopy(runtime.context.part_tracker)
    runtime.stop("Stop raced with worker reply")
    assert not runtime.context.pending_tasks
    monkeypatch.setattr("cais_spade_llm.recovery_framework.conveyor_fault.marker",
                        Mock(return_value={"status": "failed"}))
    asyncio.run(fault.retain_worker_interruption())
    assert fault.status == "triggered" and runtime.stopped
    assert fault.evidence["injection_status"] == ("completed" if observed else "failed")
    assert runtime.context.part_tracker == before
    assert runtime.context.resources["M1"].valuation["resource_state"] == "loaded"
    asyncio.run(fault.retain_worker_interruption())
    assert "M1" in runtime.context.unavailable_resources


@pytest.mark.parametrize("invalid", [
    {"simulation_elapsed_sec": 2.49}, {"processing_time_sec": 0.},
    {"simulation_elapsed_sec": float("nan")}, {"process_completed": True},
    {"part_name": "other_part"}, {"observed_pose": {}},
])
def test_machine_checkpoint_rejects_incomplete_or_contradictory_evidence(invalid):
    runtime, part, _ = runtime_with_failure("Machining breakdown during part processing")
    task = {"resource_id": "M1", "event_name": "machine_part", "task_id": "process",
            "run_id": runtime.context.run_id, "parameters": {"part_name": part}}
    runtime.context.pending_tasks["process"] = task
    evidence = {key: task[key] for key in ("run_id", "task_id", "resource_id")}
    evidence.update(part_name=part, checkpoint="during_processing_halfway",
                    source="gazebo_workholding_observation", processing_time_sec=5.,
                    simulation_elapsed_sec=2.5, process_completed=False,
                    observed_pose={"x": -4., "y": 1., "z": 1.04})
    with pytest.raises(ValueError, match="Invalid"):
        asyncio.run(runtime.conveyor_fault.accept_machine_failure(task, {
            "failure_injection": {**evidence, **invalid},
        }))
    assert not runtime.stopped
    assert runtime.conveyor_fault.status == "armed"


@pytest.mark.parametrize("scenario", ["Conveyor breakdown", "ur5e-1 breakdown",
                                      "Machining breakdown during part processing", "Part slippage"])
@pytest.mark.parametrize("stage", ["stop", "start", "ready", "marker"])
def test_reset_failures_preserve_each_scenario_latch(monkeypatch, scenario, stage):
    from cais_spade_llm.recovery_framework.conveyor_fault import reset_fault_scene

    runtime = runtime_at_pickup()[0] if scenario == "Conveyor breakdown" else runtime_with_failure(scenario)[0]
    runtime.conveyor_fault.status = "triggered"
    bridge = SimpleNamespace(
        ros2_stop=Mock(return_value="cannot stop" if stage == "stop" else None),
        ros2_start=Mock(return_value="cannot start" if stage == "start" else None),
        simulation_start_ready=Mock(return_value=(stage != "ready", "unavailable")),
    )
    monkeypatch.setattr("cais_spade_llm.recovery_framework.conveyor_fault.time.monotonic",
                        Mock(side_effect=[0, 121]))
    monkeypatch.setattr("cais_spade_llm.recovery_framework.conveyor_fault.marker",
                        Mock(return_value={"status": "failed"}))
    ok, reason = reset_fault_scene(bridge, runtime)
    assert not ok and reason
    assert runtime.conveyor_fault.status == "triggered"


@pytest.mark.parametrize("scenario", ["Conveyor breakdown", "ur5e-1 breakdown",
                                      "Machining breakdown during part processing", "Part slippage"])
def test_unreachable_checkpoint_records_outcome_without_fabricating_effects(scenario):
    runtime = runtime_at_pickup()[0] if scenario == "Conveyor breakdown" else runtime_with_failure(scenario)[0]
    runtime.context.transitions.clear()
    runtime.outcome = {"status": "blocked", "reason": "CCA holds remaining candidates"}
    before = deepcopy(runtime.context.part_tracker)
    runtime.conveyor_fault.not_reached()
    assert runtime.outcome["failure_evidence"] == {
        "injection_status": "not_reached", "reason": "CCA holds remaining candidates",
    }
    assert runtime.context.part_tracker == before
    assert not runtime.context.unavailable_resources


@pytest.mark.parametrize("scenario", ["ur5e-1 breakdown", "Part slippage"])
@pytest.mark.parametrize("pose", [{}, {"x": float("nan"), "y": 0., "z": 1.}])
def test_robot_observation_failure_keeps_fault_stopped(monkeypatch, scenario, pose):
    runtime, part, _ = runtime_with_failure(scenario)
    runtime.resource_agents[0]._controller.get_current_pose.return_value = {
        "success": True, "pose": pose,
    }
    monkeypatch.setattr("cais_spade_llm.recovery_framework.conveyor_fault.marker",
                        Mock(return_value={"status": "completed"}))
    before = deepcopy(runtime.context.part_tracker)
    asyncio.run(runtime.conveyor_fault.trigger())
    assert runtime.conveyor_fault.evidence["injection_status"] == "failed"
    assert runtime.conveyor_fault.evidence["physical_state_reconciliation_required"]
    assert runtime.stopped
    assert runtime.context.part_tracker == before
    if scenario == "Part slippage":
        runtime.resource_agents[0]._controller.detach_part.assert_not_called()


def test_machine_worker_control_tracks_disarm_rearm_and_rejects_other_tasks():
    from cais_spade_llm.recovery_framework.workflow_gazebo import WorkflowPrimitiveRunner

    runtime, part, _ = runtime_with_failure("Machining breakdown during part processing")
    fault = runtime.conveyor_fault
    task = {"resource_id": "M1", "event_name": "machine_part", "task_id": "processing",
            "run_id": runtime.context.run_id, "parameters": {"part_name": part}}
    runner = WorkflowPrimitiveRunner.__new__(WorkflowPrimitiveRunner)
    runner.task = task
    runner.request = {"failure_injection": fault.worker_request(task)}
    assert runner._machine_fault_armed()
    fault.arm(False)
    assert not runner._machine_fault_armed()
    fault.arm(True)
    assert runner._machine_fault_armed()
    runner.task = {**task, "task_id": "previous_task"}
    with pytest.raises(ValueError, match="does not match"):
        runner._machine_fault_armed()
