from __future__ import annotations

"""Non-dispatching planning, authoritative checkpoints and detached CCA checking."""

import asyncio
import hashlib
import json
import runpy
import threading
import time
from copy import deepcopy
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import Mock

import pytest
from test_predefined_safety import _SOURCE, _document
from test_primitive_program_safety import _gear_precedence_case, _grounded_check
from test_simulation_timing import cartesian_motion_controller  # noqa: F401

from cais_spade_llm.agents.central_controller.local_composition import Budget
from cais_spade_llm.agents.central_controller.online_safety_monitor import OnlineSafetyMonitor
from cais_spade_llm.agents.resource_agent.robot_agent import RobotAgent
from cais_spade_llm.recovery_framework import fingerprint
from cais_spade_llm.recovery_framework.gazebo_safety_preparation import (
    LiveSafetyPreparation,
    _runtime_record,
    capture_checkpoint,
    prepare_and_check,
)
from cais_spade_llm.resources.robot.gazebo_pick_place_controller import (
    GazeboPickPlaceController,
    capture_ur5e_state,
    cartesian_coverage_fingerprint,
    prepare_ur5e_motion,
)


@pytest.mark.parametrize("decision", [
    {"status": "held", "reason": "possible_physical_requirement_violation"},
    {"status": "inconclusive", "reason": "live_execution_tracking_unverified"},
    {"status": "allowed", "committed": False},
])
def test_live_motion_harness_never_dispatches_without_committed_cca_grant(decision):
    from unittest.mock import AsyncMock

    script = runpy.run_path(str(Path(__file__).parents[1] / "scripts/check_live_recovery_motion.py"))
    task = {"task_id": "GAZEBO_MOTION_SAFE/ur5e-4", "resource_id": "ur5e-4",
            "resource_jid": "owner@localhost", "function_name": "execute_recovery_macro",
            "primitive_steps": [{"primitive": "move_cartesian", "params": {"x": 1.}}], "params": {}}
    reference = {"recovery_id": "GAZEBO_MOTION_SAFE", "task_id": task["task_id"]}
    owner = SimpleNamespace(agent_name="ur5e-4", execute_recovery_composition_step=AsyncMock())
    runtime = SimpleNamespace(product_jid="product@localhost", resource_agents=[owner],
                              context=SimpleNamespace(run_id="run"))
    live = SimpleNamespace(
        register=AsyncMock(return_value={"status": "allowed", "task_refs": {task["task_id"]: reference}}),
        check=AsyncMock(return_value=decision), observe=Mock(),
    )
    report = {"status": "NEEDS_CONTEXT", "dispatch_authorized": False,
              "command_sent": False, "acceptance_complete": False}
    asyncio.run(script["check_and_execute"](live, runtime, task, report))
    owner.execute_recovery_composition_step.assert_not_called()
    live.observe.assert_not_called()
    assert not report["dispatch_authorized"] and not report["command_sent"]
    assert not report["acceptance_complete"]
    assert live.check.call_args.kwargs == {"sender": "owner@localhost", "commit": True}


def test_live_motion_harness_serializes_ros_integer_ids_without_stringifying_evidence():
    import numpy

    script = runpy.run_path(str(Path(__file__).parents[1] / "scripts/check_live_recovery_motion.py"))
    value = {"controller_goal_id": [numpy.uint8(7), numpy.uint8(255)],
             "sequence": numpy.uint64(2**64 - 1)}
    encoded = json.dumps(value, allow_nan=False, default=script["_json_value"])
    assert json.loads(encoded) == {"controller_goal_id": [7, 255], "sequence": 2**64 - 1}
    with pytest.raises(TypeError, match="not JSON serializable"):
        json.dumps({"unrecognized_evidence": object()}, default=script["_json_value"])


@pytest.mark.parametrize("model_execution", [1, "true", None])
def test_model_execution_requires_literal_boolean_before_accessing_readers(model_execution):
    from cais_spade_llm.recovery_framework.gazebo_safety_preparation import controller_goal_identity

    preparation = LiveSafetyPreparation(SimpleNamespace(), SimpleNamespace(), {})
    with pytest.raises(ValueError, match="model_execution"):
        preparation.initialize(model_execution=model_execution)
    with pytest.raises(ValueError, match="model_execution"):
        capture_checkpoint(SimpleNamespace(), SimpleNamespace(), model_execution=model_execution)
    with pytest.raises(ValueError, match="model_execution"):
        controller_goal_identity({}, model_execution=model_execution)


def test_live_motion_harness_preserves_actual_owner_completion_and_grant_identity():
    from unittest.mock import AsyncMock

    script = runpy.run_path(str(Path(__file__).parents[1] / "scripts/check_live_recovery_motion.py"))
    step = {"primitive": "move_cartesian", "params": {"x": 1.}}
    task = {"task_id": "GAZEBO_MOTION_SAFE/ur5e-4", "resource_id": "ur5e-4",
            "resource_jid": "owner@localhost", "function_name": "execute_recovery_macro",
            "primitive_steps": [step], "params": {}}
    reference = {"recovery_id": "GAZEBO_MOTION_SAFE", "task_id": task["task_id"]}
    grant = {"primitive_steps": [step], "run_id": "run", "recovery_composition_ref": reference}
    owner = SimpleNamespace(agent_name="ur5e-4", execute_recovery_composition_step=AsyncMock(
        return_value={"success": True, "command_sent": True, "observations": {"sequence": 17}}))
    runtime = SimpleNamespace(product_jid="product@localhost", resource_agents=[owner],
                              context=SimpleNamespace(run_id="run"))
    live = SimpleNamespace(
        register=AsyncMock(return_value={"status": "allowed", "task_refs": {task["task_id"]: reference}}),
        check=AsyncMock(return_value={"status": "allowed", "committed": True,
                                     "recovery_composition_grant": grant}),
        observe=Mock(return_value={"status": "observed", "success": True}),
    )
    report = {"status": "NEEDS_CONTEXT", "dispatch_authorized": False,
              "command_sent": False, "acceptance_complete": False}
    asyncio.run(script["check_and_execute"](live, runtime, task, report))
    assert report["acceptance_complete"] and report["execution"]["observations"] == {"sequence": 17}
    assert owner.execute_recovery_composition_step.call_args.kwargs["grant"] == grant
    completion = live.observe.call_args.args[0]
    assert completion["run_id"] == "run" and completion["recovery_composition_ref"] == reference
    assert completion["resource_jid"] == live.observe.call_args.kwargs["sender"] == "owner@localhost"
    owner.execute_recovery_composition_step.reset_mock()
    grant["run_id"] = "changed"
    with pytest.raises(ValueError, match="differs"):
        asyncio.run(script["check_and_execute"](live, runtime, task, report))
    owner.execute_recovery_composition_step.assert_not_called()


def test_live_motion_harness_incomplete_checkpoint_only_queries_cca_without_commit():
    from unittest.mock import AsyncMock

    script = runpy.run_path(str(Path(__file__).parents[1] / "scripts/check_live_recovery_motion.py"))
    task = {"task_id": "GAZEBO_MOTION_SAFE/ur5e-4", "resource_id": "ur5e-4",
            "resource_jid": "owner@localhost", "function_name": "execute_recovery_macro", "params": {}}
    owner = SimpleNamespace(agent_name="ur5e-4", execute_recovery_composition_step=AsyncMock())
    runtime = SimpleNamespace(product_jid="product@localhost", resource_agents=[owner])
    live = SimpleNamespace(
        register=AsyncMock(return_value={"status": "allowed", "task_refs": {task["task_id"]: {}}}),
        check=AsyncMock(return_value={"status": "allowed", "committed": True,
                                     "recovery_composition_grant": {}}), observe=Mock(),
    )
    report = {"status": "NEEDS_CONTEXT", "dispatch_authorized": False, "command_sent": False,
              "acceptance_complete": False, "checkpoint": {"unresolved": [{"resource_id": "ur5e-4"}]}}
    asyncio.run(script["check_and_execute"](live, runtime, task, report))
    assert live.check.call_args.kwargs["commit"] is False
    assert report["cca_check_read_only"] and not report["dispatch_authorized"]
    assert not report["command_sent"] and not report["acceptance_complete"]
    owner.execute_recovery_composition_step.assert_not_called()
    live.observe.assert_not_called()


@pytest.mark.parametrize("composition", [None, {"status": "inconclusive", "reason": "running_work_missing"}])
def test_live_motion_harness_retains_model_assumptions_and_exact_owner_records(composition):
    script = runpy.run_path(str(Path(__file__).parents[1] / "scripts/check_live_recovery_motion.py"))
    trajectory = {"joint_names": ["ur5e_4_elbow_joint"], "points": [{"positions": [1.25]}]}
    assumptions = {"continuous_motion": {"joint_trajectory": deepcopy(trajectory)},
                   "stationary_contracts": {"ur5e-3": {"kind": "declared"}}}
    coverage = {"model_execution_verified": True, "physical_execution_verified": False,
                "model_execution_assumptions": assumptions}
    preparation = {"request": {"task_id": "task"}, "steps": [{"joint_trajectory": trajectory}],
                   "result": {"command_ids": ["command"]}, "executed": {0}}
    provider = SimpleNamespace(steps={"prepared": {"joint_trajectory": trajectory}},
                               preparations={"program": preparation},
                               execution_coverage_records={"coverage": coverage})
    live = SimpleNamespace(lock=threading.RLock(),
                           commands=SimpleNamespace(snapshot=lambda: {"commands": {"command": {"status": "active"}}}),
                           monitor=SimpleNamespace(history=[{"observations": {"sequence": 19}}], states={"SAFE_shared_area_mutex": {"state"}}),
                           preparation=SimpleNamespace(last_execution_coverage=coverage), last_results={})
    report = {"cca_decision": {} if composition is None else {"common_composition": composition},
              "dispatch_authorized": False, "command_sent": False}
    owners = [SimpleNamespace(agent_name="ur5e-4", recovery_composition_evidence_provider=provider)]
    script["retain_execution_evidence"](live, owners, report)
    assert report["common_composition"] == composition
    assert report["model_execution_assumptions"] == assumptions
    assert report["execution_coverage"]["physical_execution_verified"] is False
    assert report["prepared_programs"]["ur5e-4"]["steps"]["prepared"]["joint_trajectory"] == trajectory
    assert report["prepared_programs"]["ur5e-4"]["preparations"]["program"]["executed"] == [0]
    assert report["physical_history"] == [{"observations": {"sequence": 19}}]
    trajectory["points"][0]["positions"][0] = 99.
    assumptions["stationary_contracts"].clear()
    assert report["prepared_programs"]["ur5e-4"]["steps"]["prepared"]["joint_trajectory"]["points"][0]["positions"] == [1.25]
    assert report["model_execution_assumptions"]["stationary_contracts"]
    assert not report["dispatch_authorized"] and not report["command_sent"]
    json.dumps(report, allow_nan=False)


@pytest.fixture
def controller(cartesian_motion_controller):  # noqa: F811 - imported pytest fixture
    from geometry_msgs.msg import Pose, Quaternion

    controller, _, response = cartesian_motion_controller
    names = ["ur5e_3_" + name for name in (
        "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint")]
    controller.arm_joint_names = names
    controller._attached_model = "KET4_Square_4mm"
    response.solution.joint_trajectory.joint_names = names
    for index, point in enumerate(response.solution.joint_trajectory.points):
        point.positions = [.1 * index] * 6
        point.velocities = [0.] * 6
        point.accelerations = [0.] * 6
    controller._make_orientation = lambda *q: Quaternion(**dict(zip(("x", "y", "z", "w"), map(float, q), strict=True)))

    def pose(x, y, z, orientation):
        result = Pose()
        result.position.x, result.position.y, result.position.z = float(x), float(y), float(z)
        result.orientation = orientation
        return result

    controller._make_pose = pose
    for name in ("move_cartesian", "_send_simulation_joint_trajectory", "open_gripper", "close_gripper",
                 "attach_part", "detach_part", "set_entity_pose", "_publish_arm_joint_trajectory_and_wait",
                 "_consume_prepared_cartesian", "snap_part_to_slot"):
        setattr(controller, name, Mock(side_effect=AssertionError("Preparation dispatched " + name)))
    controller._GetEntityState = SimpleNamespace(Request=lambda **kwargs: SimpleNamespace(**kwargs))
    controller._get_state_client = SimpleNamespace(call_async=None)
    controller._node = SimpleNamespace(get_clock=lambda: SimpleNamespace(
        now=lambda: SimpleNamespace(nanoseconds=1_000_000_000, to_msg=lambda: None)))
    # GetCartesianPath requires a real ROS Time in the request header.
    from builtin_interfaces.msg import Time
    controller._node.get_clock = lambda: SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=1_000_000_000, to_msg=Time))
    for name in ("prepare_recovery_safety_program", "recovery_safety_configuration",
                 "get_recovery_safety_primitive_model", "read_recovery_entity_state", "validate_recovery_safety_step"):
        setattr(controller, name, MethodType(getattr(GazeboPickPlaceController, name), controller))
    return controller


def _start(controller) -> dict:
    return {"frame": "world", "current_pose": [1.1, 0, 1, 0, 0, 0, 1],
            "joint_names": controller.arm_joint_names, "joint_positions": [0.] * 6,
            "attachment": {"model_name": "KET4_Square_4mm", "observed_attachment_complete": True},
            "launch_id": "synthetic-launch"}


def test_plan_only_retains_exact_parameters_joints_timing_and_custody(controller):
    params = {"x": 1.5, "y": 0., "z": 1., "speed": 1.}
    start = _start(controller)
    binding = {"checkpoint_id": "checkpoint", "source": {"outline_id": "new_event", "step_index": 3}}
    result = prepare_ur5e_motion(controller, primitive="move_cartesian", params=params, start=start, binding=binding)
    assert result["status"] == "prepared", result
    assert result["params"] == params and result["start"] == start and result["binding"] == binding
    assert result["joint_trajectory"]["joint_names"] == controller.arm_joint_names
    assert result["joint_trajectory"]["duration_ns"] == 1_000_000_000
    assert result["joint_trajectory"]["points"][-1]["positions"] == [.1] * 6
    assert result["observation_status"] == "NEEDS_CONTEXT"
    assert controller._attached_model == "KET4_Square_4mm"
    assert result["command_sent"] is False
    controller._consume_prepared_cartesian.assert_not_called()
    controller._send_simulation_joint_trajectory.assert_not_called()
    assert controller._cart_client.call_args is None  # Only call_async is used.
    request = controller._cart_client.call_async.call_args.args[0]
    assert request.start_state.joint_state.name == controller.arm_joint_names
    assert list(request.start_state.joint_state.position) == [0.] * 6
    assert request.avoid_collisions


def test_continuous_owner_contract_preserves_the_native_prepared_trajectory(controller):
    from test_continuous_motion import configuration

    config = configuration()
    config.update(joint_names=list(controller.arm_joint_names), root='root', root_xyz=[1.1,0,1], reference_link='body6')
    config['joints'] = [{'name':name, 'parent':'root' if i == 0 else f'body{i}',
                         'child':f'body{i+1}', 'type':'prismatic', 'axis':[1,0,0],
                         'xyz':[0,0,0], 'rpy':[0,0,0]} for i,name in enumerate(controller.arm_joint_names)]
    config['components'][0]['link'] = 'body6'
    controller.recovery_safety_observer = SimpleNamespace(motion_configuration=config)
    params = {'x':1.7,'y':0.,'z':1.,'qx':0.,'qy':0.,'qz':0.,'qw':1.}
    result = prepare_ur5e_motion(controller,primitive='move_cartesian',params=params,
                                start=_start(controller),binding={'source':{'outline_id':'supplied','step_index':0}})
    assert result['status'] == 'prepared', result
    assert result['observation_status'] == 'prepared', result
    assert result['continuous_motion']['joint_trajectory'] == result['joint_trajectory']
    assert result['params'] == params and result['command_sent'] is False
    model = controller.get_recovery_safety_primitive_model()
    effect = model.effects(primitive='move_cartesian',params=params,
        evidence={'continuous_motion':result['continuous_motion'],'joint_trajectory':result['joint_trajectory']},
        start_time=0,end_time=1)
    assert effect['continuous_motion'] == result['continuous_motion']
    controller._send_simulation_joint_trajectory.assert_not_called()
    controller.attach_part.assert_not_called()
    controller.detach_part.assert_not_called()


def test_live_controller_rejects_changed_interpolation_or_geometry(controller):
    controller.recovery_safety_observer = SimpleNamespace(
        motion_configuration={'description_sha256':'unchanged','interpolation':'splines'},
        configuration={'description_node':'/description','controller_node':'/configured_controller'},
        provider=SimpleNamespace(reader=SimpleNamespace(parameter=lambda *_:'changed')))
    with pytest.raises(ValueError,match='changed'):
        GazeboPickPlaceController.capture_recovery_safety_state(controller)


@pytest.mark.parametrize('contract,idle', [(None,True), ({},True), ('supported',False)])
def test_future_stationary_coverage_needs_the_owner_contract_and_idle_evidence(contract,idle):
    from cais_spade_llm.recovery_framework.gazebo_safety_preparation import (
        GazeboResourceObservation,
    )

    if contract == 'supported':
        contract = {'kind':'idle_commanded_hold','requires_no_running_tasks':True,
                    'requires_no_active_goals':True,'future_execution_tracking':'not_established'}
    observer = GazeboResourceObservation(None,SimpleNamespace(agent_name='configured_resource'),{'stationary_contract':contract})
    checkpoint = {'checkpoint_id':'checkpoint','observations':{'configured_resource':{'physical':{'idle':idle}}}}
    with pytest.raises(ValueError,match='coverage|idle'):
        observer.stationary_coverage([[0,1]],checkpoint)


@pytest.mark.parametrize('succeeds', [True, False])
def test_read_only_observation_retries_lost_reply_without_changing_capture(succeeds):
    from cais_spade_llm.recovery_framework.gazebo_safety_preparation import GazeboSafetyReader

    reader = object.__new__(GazeboSafetyReader)
    observation = SimpleNamespace(success=True, message='exact captured source', observed_monotonic=123.)
    futures = [Mock() for _ in range(3)]
    for index, future in enumerate(futures):
        future.result.return_value = observation
        future.add_done_callback.side_effect = (
            (lambda callback: callback(None)) if succeeds and index == 2 else (lambda _: None)
        )
    client = Mock(srv_name='/GETRECOVERYSTATE')
    client.wait_for_service.return_value = True
    client.call_async.side_effect = futures
    if succeeds:
        assert reader._call(client, object(), timeout=.001) is observation
        assert observation.observed_monotonic == 123.
    else:
        with pytest.raises(ValueError, match='after 3 attempts'):
            reader._call(client, object(), timeout=.001)
    assert client.call_async.call_count == 3
    assert client.remove_pending_request.call_count == (2 if succeeds else 3)


@pytest.mark.parametrize("paused", [True, False])
@pytest.mark.parametrize("confirmed", [True, False])
def test_world_pause_requires_native_confirmation_and_no_job_command(monkeypatch, paused, confirmed):
    import sys
    from cais_spade_llm.recovery_framework.gazebo_safety_preparation import GazeboSafetyReader

    empty = SimpleNamespace(Request=lambda: object())
    monkeypatch.setitem(sys.modules, "std_srvs.srv", SimpleNamespace(Empty=empty))
    reader = object.__new__(GazeboSafetyReader)
    client = object()
    reader._node = SimpleNamespace(create_client=Mock(return_value=client), destroy_client=Mock())
    reader._call = Mock()
    reader.snapshot = Mock(return_value={"world_paused": paused if confirmed else not paused,
                                       "instance_id": "owned-world", "simulation_time": 102.})
    if confirmed:
        assert reader.set_world_paused(paused)["world_paused"] is paused
    else:
        with pytest.raises(ValueError, match="did not confirm"):
            reader.set_world_paused(paused)
    reader._node.create_client.assert_called_once_with(empty, "/pause_physics" if paused else "/unpause_physics")
    reader._node.destroy_client.assert_called_once_with(client)
    reader.snapshot.assert_called_once_with(refresh=True)
    assert reader._call.call_count == 1


def test_idle_observation_requires_a_record_from_the_current_publisher():
    from cais_spade_llm.recovery_framework.gazebo_safety_preparation import GazeboSafetyReader

    reader = object.__new__(GazeboSafetyReader)
    reader.configuration = {'controller_status_topics':['/configured_action/_action/status']}
    reader._lock = threading.RLock()
    reader._node = SimpleNamespace(get_publishers_info_by_topic=lambda _: [SimpleNamespace(endpoint_gid=[1])])
    reader._goal_status = {}
    with pytest.raises(ValueError,match='unavailable'):
        reader.idle_goals()
    topic = reader.configuration['controller_status_topics'][0]
    reader._goal_status[topic] = {'publisher_gid':[2],'goals':[]}
    with pytest.raises(ValueError,match='unavailable'):
        reader.idle_goals()
    reader._goal_status[topic]['publisher_gid'] = [1]
    reader._goal_status[topic]['goals'] = [{'id':[0],'status':2}]
    with pytest.raises(ValueError,match='active'):
        reader.idle_goals()
    reader._goal_status[topic]['goals'] = []
    assert reader.idle_goals()[topic]['goals'] == []


@pytest.mark.parametrize('with_info', [False, True])
def test_action_status_callback_supports_humble_without_authenticating_missing_info(with_info):
    import numpy as np
    from cais_spade_llm.recovery_framework.gazebo_safety_preparation import GazeboSafetyReader

    reader = object.__new__(GazeboSafetyReader)
    topic = '/configured_action/_action/status'
    reader.configuration = {'controller_status_topics': [topic]}
    reader._lock = threading.RLock()
    reader._goal_status = {}
    reader._node = SimpleNamespace(get_publishers_info_by_topic=lambda _: [SimpleNamespace(endpoint_gid=[1])])
    message = SimpleNamespace(status_list=[SimpleNamespace(
        goal_info=SimpleNamespace(goal_id=SimpleNamespace(uuid=[np.uint8(7)])), status=4,
    )])
    if with_info:
        reader._record_goals(topic, message, SimpleNamespace(publisher_gid=[np.uint8(1)]))
        assert reader.idle_goals()[topic]['publisher_gid'] == [1]
    else:
        reader._record_goals(topic, message)
        assert reader._goal_status[topic]['publisher_gid'] == []
        with pytest.raises(ValueError, match='unavailable'):
            reader.idle_goals()
    assert json.loads(json.dumps(reader._goal_status[topic]))['goals'] == [{'id': [7], 'status': 4}]


@pytest.mark.parametrize("change", [None, "missing", "duplicate", "active", "pending", "not_holding",
                                     "stale", "incarnation", "navigation", "unavailable"])
def test_controller_owner_query_requires_complete_fresh_idle_evidence(change):
    from cais_spade_llm.recovery_framework.gazebo_safety_preparation import GazeboSafetyReader

    reader = object.__new__(GazeboSafetyReader)
    topic = "/configured_controller/follow_joint_trajectory/_action/status"
    reader.configuration = {"controller_status_topics": [topic], "controller_state_services": {
        "/configured_controller/recovery_state": {
            "covers": [topic], "required_values": {"navigation_enabled": False}},
    }}
    observation = {"version": 1, "instance_id": "controller-incarnation", "simulation_time": 12.,
                   "has_active_goal": False, "has_pending_goal": False, "holding": True,
                   "navigation_enabled": False}
    if change in {"missing", "duplicate"}:
        reader.configuration["controller_state_services"]["/configured_controller/recovery_state"]["covers"] = (
            [] if change == "missing" else [topic, topic])
    elif change == "active":
        observation["has_active_goal"] = True
    elif change == "pending":
        observation["has_pending_goal"] = True
    elif change == "not_holding":
        observation["holding"] = False
    elif change == "stale":
        observation["simulation_time"] = 1.
    elif change == "incarnation":
        observation.pop("instance_id")
    elif change == "navigation":
        observation["navigation_enabled"] = True
    reader._type = SimpleNamespace(Request=lambda: object())
    reader._node = SimpleNamespace(create_client=Mock(return_value=object()), destroy_client=Mock())
    reader._call = Mock(return_value=SimpleNamespace(success=change != "unavailable",
                                                     message=json.dumps(observation)))
    reader.snapshot = lambda: {"simulation_time": 12.1}
    if change is None:
        rows = reader.idle_goals()
        assert rows["/configured_controller/recovery_state"]["instance_id"] == "controller-incarnation"
        assert rows["/configured_controller/recovery_state"]["covers"] == [topic]
    else:
        with pytest.raises(ValueError):
            reader.idle_goals()
    assert reader._node.destroy_client.call_count == reader._call.call_count


def test_recovery_configuration_declares_every_native_controller_observation_owner():
    path = Path(__file__).parents[1] / "cais_spade_llm/initialization/recovery_framework_gazebo.json"
    config = json.loads(path.read_text())["safety_preparation"]
    services = config["controller_state_services"]
    covered = [topic for row in services.values() for topic in (
        row["covers"] + row.get("command_gate", {}).get("status_topics", []))]
    assert len(covered) == len(set(covered))
    assert set(covered) == set(config["controller_status_topics"])
    for owner in config["resources"].values():
        if owner.get("controller_node"):
            assert owner["controller_node"] + "/recovery_state" in services
    kmr = services["/KMR/recovery_state"]
    assert "required_values" not in kmr
    assert kmr["command_gate"]["status_topics"] == [
        "/KMR/navigate_to_pose/_action/status", "/KMR/follow_path/_action/status"]
    assert not set(kmr["covers"]) & set(kmr["command_gate"]["status_topics"])


def _command_gate_reader(monkeypatch, tmp_path):
    import sys
    import numpy as np
    from cais_spade_llm.recovery_framework.gazebo_safety_preparation import GazeboSafetyReader

    source = tmp_path / "lib/cais_lab_robotics/kmr_base_controller.py"
    source.parent.mkdir(parents=True)
    source.write_text("recorded controller source")
    monkeypatch.setitem(sys.modules, "ament_index_python.packages", SimpleNamespace(
        get_package_prefix=lambda package: str(tmp_path)))
    config = json.loads((Path(__file__).parents[1] /
        "cais_spade_llm/initialization/recovery_framework_gazebo.json").read_text())["safety_preparation"]
    declaration = config["controller_state_services"]["/KMR/recovery_state"]
    reader = object.__new__(GazeboSafetyReader)
    reader.configuration = {"controller_state_services": {"/KMR/recovery_state": declaration},
        "controller_status_topics": declaration["covers"] + declaration["command_gate"]["status_topics"]}
    reader._lock, reader._goal_status = threading.RLock(), {}
    gate = declaration["command_gate"]
    row = {"version": 1, "instance_id": "KMR-instance", "simulation_time": 12.,
        "controller_node": "/KMR_base_controller", "controller_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "has_active_goal": True, "has_pending_goal": False, "holding": False,
        "navigation_enabled": True, "nav2_follow_path_active": True,
        "last_output": [0., 0., 0.], "owned_activity": {
            "has_active_goal": False, "has_pending_goal": False, "holding": True},
        "command_gate": {"version": 1, "closed": True, "input_topic": gate["input_topic"],
            "output_topic": gate["output_topic"], "owned_status_topics": declaration["covers"],
            "gated_status_topics": gate["status_topics"]}}
    publisher = SimpleNamespace(endpoint_gid=[np.uint8(7)], node_namespace="/", node_name="KMR_base_controller")
    reader._node = SimpleNamespace(create_client=Mock(return_value=object()), destroy_client=Mock(),
        get_publishers_info_by_topic=Mock(return_value=[publisher]))
    reader._type = SimpleNamespace(Request=lambda: object())
    reader._call = lambda *_: SimpleNamespace(success=True, message=json.dumps(row))
    reader.snapshot = lambda: {"simulation_time": 12.1}
    return reader, row, publisher


@pytest.mark.parametrize("observed", [False, True])
def test_command_gate_keeps_external_nav2_activity_distinct(monkeypatch, tmp_path, observed):
    reader, row, _ = _command_gate_reader(monkeypatch, tmp_path)
    for topic in row["command_gate"]["gated_status_topics"]:
        if observed:
            reader._goal_status[topic] = {"publisher_gid": [7], "received_monotonic": 1.,
                                          "goals": [{"id": [5], "status": 2}]}
    result = reader.idle_goals()["/KMR/recovery_state"]
    proof = result["command_gate_evidence"]
    assert result["has_active_goal"] is True
    assert result["holding"] is False
    assert proof["owned_activity"]["has_active_goal"] is False
    assert proof["all_nav2_goals_idle"] is None
    assert {item["status"] for item in proof["nav2_goal_observations"].values()} == {
        "observed" if observed else "unknown"}
    if observed:
        assert all(item["goals"][0]["status"] == 2 for item in proof["nav2_goal_observations"].values())
    assert json.loads(json.dumps(result))["command_gate_evidence"]["output_publisher_gid"] == [7]
    direct = reader.controller_state("/KMR")
    assert direct["command_gate_evidence"] == proof
    assert direct["source"] == "/KMR/recovery_state"


@pytest.mark.parametrize("change", ["missing", "open", "active", "pending", "holding", "source", "node",
    "input", "output", "coverage", "nonzero", "unrelated_publisher", "multiple_publishers", "missing_publisher"])
def test_command_gate_rejects_incomplete_authority(monkeypatch, tmp_path, change):
    reader, row, publisher = _command_gate_reader(monkeypatch, tmp_path)
    if change == "missing":
        row.pop("command_gate")
    elif change == "open":
        row["command_gate"]["closed"] = False
    elif change in {"active", "pending"}:
        row["owned_activity"]["has_" + change + "_goal"] = True
    elif change == "holding":
        row["owned_activity"]["holding"] = False
    elif change == "source":
        row["controller_source_sha256"] = "other"
    elif change == "node":
        row["controller_node"] = "/other"
    elif change in {"input", "output"}:
        row["command_gate"][change + "_topic"] = "/other"
    elif change == "coverage":
        row["command_gate"]["owned_status_topics"] = []
    elif change == "nonzero":
        row["last_output"] = [1., 0., 0.]
    elif change == "unrelated_publisher":
        publisher.node_name = "other"
    elif change == "multiple_publishers":
        reader._node.get_publishers_info_by_topic.return_value = [publisher, publisher]
    elif change == "missing_publisher":
        reader._node.get_publishers_info_by_topic.return_value = []
    with pytest.raises(ValueError):
        reader.idle_goals()
    recorded = reader.controller_state("/KMR")
    assert recorded['command_gate_evidence']['status'] == 'NEEDS_CONTEXT'
    assert recorded['has_active_goal'] is row['has_active_goal']


def test_command_gate_identity_retains_authority_changes_not_receipt_time(monkeypatch, tmp_path):
    from cais_spade_llm.recovery_framework.gazebo_safety_preparation import controller_goal_identity

    reader, row, _ = _command_gate_reader(monkeypatch, tmp_path)
    topic = row["command_gate"]["gated_status_topics"][0]
    reader._goal_status[topic] = {"publisher_gid": [7], "received_monotonic": 1., "goals": []}
    original = reader.idle_goals()
    later = deepcopy(original)
    observation = later['/KMR/recovery_state']['command_gate_evidence']['nav2_goal_observations'][topic]
    observation['received_monotonic'] = 99.
    assert controller_goal_identity(original) == controller_goal_identity(later)
    assert original['/KMR/recovery_state']['command_gate_evidence']['nav2_goal_observations'][topic]['received_monotonic'] == 1.
    later['/KMR/recovery_state']['command_gate_evidence']['output_publisher_gid'] = [8]
    assert controller_goal_identity(original) != controller_goal_identity(later)


def test_controller_goal_identity_retains_command_and_launch_changes():
    from cais_spade_llm.recovery_framework.gazebo_safety_preparation import controller_goal_identity

    original = {"/controller/recovery_state": {"instance_id": "launch-1", "command_revision": 1,
                "contract_revision": 1, "holding": True, "observed_stationary": True,
                "has_pending_goal": False, "simulation_time": 2., "sequence": 3,
                "positions": [0.], "has_active_goal": False}}
    fresh = deepcopy(original)
    fresh["/controller/recovery_state"].update(simulation_time=3., sequence=4, positions=[.001])
    assert controller_goal_identity(fresh) == controller_goal_identity(original)
    sampled = deepcopy(fresh)
    sampled["/controller/recovery_state"]["observed_stationary"] = False
    assert controller_goal_identity(sampled) != controller_goal_identity(original)
    assert controller_goal_identity(sampled, model_execution=True) == controller_goal_identity(
        original, model_execution=True)
    for key, value in (("instance_id", "launch-2"), ("command_revision", 2),
                       ("contract_revision", 2), ("holding", False),
                       ("has_active_goal", True), ("has_pending_goal", True)):
        changed = deepcopy(fresh)
        changed["/controller/recovery_state"][key] = value
        assert controller_goal_identity(changed) != controller_goal_identity(original)
        assert controller_goal_identity(changed, model_execution=True) != controller_goal_identity(
            original, model_execution=True)


@pytest.mark.parametrize("active", [False, True])
def test_controller_state_captures_actual_hold_targets_without_claiming_idle(active):
    from cais_spade_llm.recovery_framework.gazebo_safety_preparation import GazeboSafetyReader

    reader = object.__new__(GazeboSafetyReader)
    reader.configuration = {}
    observation = {"version": 1, "instance_id": "controller", "simulation_time": 12.,
        "joint_names": ["joint"], "positions": [.001], "commanded_positions": [0.],
        "commanded_positions_source": "native position command interfaces", "holding": not active,
        "has_active_goal": active, "has_pending_goal": False, "observed_stationary": False}
    reader._type = SimpleNamespace(Request=lambda: object())
    reader._node = SimpleNamespace(create_client=Mock(return_value=object()), destroy_client=Mock())
    reader._call = Mock(return_value=SimpleNamespace(success=True, message=json.dumps(observation)))
    result = reader.controller_state("/controller")
    assert {key: result[key] for key in observation} == observation
    assert result["source"] == "/controller/recovery_state"
    assert result["observed_stationary"] is False
    reader._node.destroy_client.assert_called_once()


@pytest.mark.parametrize('recorded', [False, True])
def test_controller_capture_keeps_recorded_endpoint_separate_from_freshness(recorded):
    from cais_spade_llm.recovery_framework.gazebo_safety_preparation import GazeboSafetyReader

    reader = object.__new__(GazeboSafetyReader)
    reader.configuration = {}
    row = {'version': 1, 'instance_id': 'controller', 'simulation_time': 12.,
           'recorded_observation': True, 'command_unchanged': True, 'observation_wall_age_sec': 30.,
           'has_active_goal': True, 'has_pending_goal': False, 'holding': False}
    reader._type = SimpleNamespace(Request=lambda: object())
    reader._node = SimpleNamespace(create_client=Mock(return_value=object()), destroy_client=Mock())
    reader._call = Mock(return_value=SimpleNamespace(success=True, message=json.dumps(row)))
    result = reader.controller_state('/controller', recorded=recorded)
    expected = '/controller/recovery_recorded_state' if recorded else '/controller/recovery_state'
    assert reader._node.create_client.call_args.args[1] == expected
    assert result['source'] == expected and result['simulation_time'] == 12.
    assert result['has_active_goal'] is True and result['holding'] is False
    row['command_unchanged'] = False
    reader._call.return_value.message = json.dumps(row)
    with pytest.raises(ValueError, match='Incomplete recorded controller state'):
        reader.controller_state('/controller', recorded=True)


def test_recorded_controller_activity_preserves_busy_status():
    from cais_spade_llm.recovery_framework.gazebo_safety_preparation import GazeboSafetyReader

    reader = object.__new__(GazeboSafetyReader)
    source = '/controller/recovery_state'
    topic = '/controller/follow_joint_trajectory/_action/status'
    reader.configuration = {'controller_state_services': {source: {'covers': [topic]}},
                            'controller_status_topics': [topic]}
    row = {'version': 1, 'instance_id': 'controller', 'simulation_time': 12.,
           'recorded_observation': True, 'command_unchanged': True,
           'has_active_goal': True, 'has_pending_goal': False, 'holding': False}
    reader._type = SimpleNamespace(Request=lambda: object())
    reader._node = SimpleNamespace(create_client=Mock(return_value=object()), destroy_client=Mock())
    reader._call = Mock(return_value=SimpleNamespace(success=True, message=json.dumps(row)))
    reader.snapshot = Mock(return_value={'simulation_time': 12.})
    result = reader.controller_activity(recorded=True)
    assert result[source]['has_active_goal'] is True and result[source]['holding'] is False
    assert result[source]['source'] == '/controller/recovery_recorded_state'
    with pytest.raises(ValueError, match='not holding'):
        reader.idle_goals()


def test_configured_fixed_equipment_coverage_needs_no_gripper():
    from cais_spade_llm.recovery_framework.gazebo_safety_preparation import (
        GazeboResourceObservation,
    )

    contract = {'kind': 'static_body_and_idle_containment', 'requires_no_running_tasks': True,
                'requires_no_active_goals': True, 'future_execution_tracking': 'not_established'}
    observer = GazeboResourceObservation(None, SimpleNamespace(agent_name='M1'), {'stationary_contract': contract})
    physical = {'idle': True, 'model_static': True, 'observation_state': {'resource_state': 'idle'}}
    checkpoint = {'checkpoint_id': 'checkpoint', 'observations': {'M1': {'physical': physical}}}
    result = observer.stationary_coverage([[0, 2]], checkpoint)
    assert result['contract'] == contract and result['intervals'] == [[0, 2]]
    assert 'gripper_state' not in physical['observation_state']
    physical['model_static'] = False
    with pytest.raises(ValueError, match='observed static'):
        observer.stationary_coverage([[0, 2]], checkpoint)


def _held_observation_case():
    from cais_spade_llm.recovery_framework.gazebo_safety_preparation import GazeboResourceObservation

    tool = [1., 0., 1., 0., 0., 0., 1.]
    part = [1.1, 0., 1., 0., 0., 0., 1.]
    def body(pose, bounds):
        return {"pose": pose, "linear_speed": 0., "angular_speed": 0.,
                "collisions": [{"id": "body", "bounds": bounds}]}
    record = {"attachment_complete": True, "attachment_revision": 7, "instance_id": "physics",
        "observed_monotonic": time.monotonic(), "simulation_time": 5.,
        "attachments": [{"model1": "robot", "link1": "tool", "model2": "gear_model", "link2": "link"}],
        "models": {
            "robot": {"static": False, "pose": tool, "links": {"tool": body(tool, [[.9, 1.1], [-.1, .1], [.9, 1.1]])},
                      "joints": {"finger": {"position": .02}}},
            "gear_model": {"pose": part, "links": {"link": body(part, [[1.05, 1.15], [-.02, .02], [.98, 1.02]])}},
        }}
    context = SimpleNamespace(snapshot=lambda: {"ur5e-4": {"held_part": "gear_small"}},
        part_tracker={"gear_small": {"location": "ur5e-4"}},
        inputs={"geometry": {"parts": {"model_map": {"gear_small": "gear_model"}}}})
    reader = SimpleNamespace(snapshot=lambda **kwargs: deepcopy(record))
    owner = SimpleNamespace(agent_name="ur5e-4", jid="ur5e-4@localhost")
    provider = SimpleNamespace(runtime=SimpleNamespace(context=context), reader=reader, launch_id="launch")
    config = {"model": "robot", "reference_link": "tool", "stationary_linear_speed_max": .01,
              "stationary_angular_speed_max": .01, "stationary_contract": {"kind": "idle_commanded_hold"},
              "gripper": {"joint": "finger", "open": .1, "closed": .02, "tolerance": .001}}
    observer = GazeboResourceObservation(provider, owner, config)
    return observer, record


def test_attached_part_capture_retains_measured_transform_and_geometry():
    observer, record = _held_observation_case()
    before = deepcopy(record)
    captured = observer.capture()
    assert captured["observation_state"]["held_part"] == "gear_small"
    assert captured["observation_state"]["grasp_transform"] == pytest.approx([.1, 0., 0., 0., 0., 0., 1.])
    assert captured["attachment"]["part_name"] == "gear_small"
    assert captured["attachment"]["records"] == record["attachments"]
    assert captured["attachment"]["part_local_bounds"][0] == pytest.approx([-.05, .05])
    assert captured["link_states"] == record["models"]["robot"]["links"]
    assert record == before


def test_reported_fault_checks_custody_against_the_robots_own_report():
    observer, record = _held_observation_case()
    record["attachments"] = []
    record["models"]["robot"]["joints"]["finger"]["position"] = .1
    with pytest.raises(ValueError, match="observed empty attachment set"):
        observer.capture(record=record)
    # The robot reported its fault and its own empty gripper; the product's belief still lags.
    observer.owner._held_part = None
    observer.provider.runtime.context.fault_reports = {"ur5e-4": {"reported_state": "failed"}}
    captured = observer.capture(record=record)
    assert captured["observation_state"]["held_part"] is None
    assert captured["attachment"]["model_name"] is None
    assert captured["custody_after_fault_report"] == {"belief_held_part": "gear_small", "reported_held_part": None}
    observer.provider.runtime.context.snapshot = lambda: {"ur5e-4": {"held_part": None}}
    assert "custody_after_fault_report" not in observer.capture(record=record)


def test_owner_kinematics_and_collisions_keep_one_physics_snapshot():
    observer, record = _held_observation_case()
    observer.provider.reader.snapshot = Mock(side_effect=AssertionError("must retain acquired physics snapshot"))
    source = {key: record[key] for key in ("instance_id", "simulation_time", "observed_monotonic")}
    native = {"current_pose": deepcopy(record["models"]["robot"]["pose"]), "native_physics_source": source,
              "joint_names": ["finger"], "joint_positions": [.02]}
    captured = observer.capture(native=native, record=record)
    assert captured["native_physics_source"] == source
    assert captured["simulation_time"] == record["simulation_time"]
    assert captured["link_states"] == record["models"]["robot"]["links"]
    observer.provider.reader.snapshot.assert_not_called()
    changed = deepcopy(record)
    changed["simulation_time"] += .001
    with pytest.raises(ValueError, match="different physics snapshots"):
        observer.capture(native=native, record=changed)
    changed = deepcopy(record)
    changed["observed_monotonic"] -= 100
    with pytest.raises(ValueError, match="Stale resource observation"):
        observer.capture(native=native, record=changed)


def test_resource_observation_without_kinematics_retains_native_source():
    observer, record = _held_observation_case()
    record["models"]["robot"]["static"] = True
    record["attachments"] = []
    observer.provider.runtime.context.snapshot = lambda: {"ur5e-4": {"resource_state": "idle"}}
    observer.provider.runtime.context.part_tracker = {}
    observer.provider.reader.snapshot = Mock(side_effect=AssertionError("use the recorded checkpoint"))
    captured = observer.capture(record=record)
    assert captured["model_static"] is True
    assert captured["native_physics_source"] == {
        key: record[key] for key in ("instance_id", "simulation_time", "observed_monotonic")
    }
    assert captured["link_states"] == record["models"]["robot"]["links"]
    assert "joint_positions" not in captured
    observer.provider.reader.snapshot.assert_not_called()


@pytest.mark.parametrize("change", [None, "joint", "time", "instance", "not_paused", "not_capturing"])
def test_frozen_resource_observation_requires_exact_native_checkpoint(change):
    observer, record = _held_observation_case()
    record["world_paused"] = True
    record["observed_monotonic"] -= 100
    observer.provider.runtime.recovery_capture_frozen = True
    observer.provider.runtime.recovery_capture_native = deepcopy(record)
    if change == "joint":
        record["models"]["robot"]["joints"]["finger"]["position"] += .001
    elif change == "time":
        record["simulation_time"] += .001
    elif change == "instance":
        record["instance_id"] = "different"
    elif change == "not_paused":
        record["world_paused"] = False
    elif change == "not_capturing":
        observer.provider.runtime.recovery_capture_frozen = False
    if change is not None:
        with pytest.raises(ValueError, match="Stale resource observation"):
            observer.capture(record=record)
    else:
        captured = observer.capture(record=record)
        assert captured["frozen_native_fingerprint"]
        assert captured["observed_monotonic"] == record["observed_monotonic"]
        assert captured["simulation_time"] == record["simulation_time"]


@pytest.mark.parametrize("frozen", [True, False])
def test_physical_capture_retains_verified_frozen_export_after_calculation(frozen):
    from cais_spade_llm.recovery_framework.recovery_observations import capture_recovery_physical_evidence

    observer, record = _held_observation_case()
    record.update(world_paused=True, observed_monotonic=time.monotonic() - 100)
    record["attachments"] = []
    runtime = observer.provider.runtime
    runtime.recovery_capture_frozen = True
    runtime.recovery_capture_native = deepcopy(record)
    context = runtime.context
    state = {"held_part": None, "resource_state": "idle"}
    context.snapshot = lambda: {"ur5e-4": deepcopy(state)}
    context.part_tracker = {}
    context.revision, context.run_id = 3, "captured"
    context.resources = {"ur5e-4": SimpleNamespace(valuation=deepcopy(state))}
    physical = observer.capture(record=record)
    owner = SimpleNamespace(agent_name="ur5e-4", recovery_safety_observer=observer,
        capture_recovery_safety_state=Mock(return_value=physical))
    runtime.resource_agents = [owner]
    runtime.jids = {"ur5e-4": "ur5e-4@localhost"}
    runtime.recovery_capture_frozen = frozen
    result = capture_recovery_physical_evidence(runtime)
    failures = [row for row in result["unresolved"] if row.get("resource_id") == "ur5e-4"]
    if frozen:
        assert not failures
        assert result["snapshot"]["resources"]["ur5e-4"]["held_part"] is None
        owner.capture_recovery_safety_state.assert_called_once()
    else:
        assert failures == [{"resource_id": "ur5e-4", "reason": "Stale resource observation"}]
        assert "ur5e-4" not in result["snapshot"]["resources"]


def test_observed_configuration_retains_current_passive_joints_and_root():
    from cais_spade_llm.resources.continuous_geometry import observed_joint_configuration

    model = {"joint_names": ["arm"], "joints": [
        {"name": "arm", "type": "revolute"},
        {"name": "finger", "type": "prismatic", "frozen_position": .11},
        {"name": "mount", "type": "fixed"}],
        "fixed_joint_positions": {"finger": .11}, "root_xyz": [0., 0., 0.]}
    original = deepcopy(model)
    result = observed_joint_configuration(model, {"finger": {"position": .02}}, [1., 2., 3., 0., 0., 0., 1.])
    assert result["joints"][1]["frozen_position"] == .02
    assert result["fixed_joint_positions"] == {"finger": .02}
    assert result["root_xyz"] == [1., 2., 3.]
    assert model == original
    with pytest.raises(ValueError, match="missing or invalid"):
        observed_joint_configuration(model, {}, [1., 2., 3., 0., 0., 0., 1.])


def test_continuous_geometry_excludes_peer_joint_observations():
    from cais_spade_llm.resources.continuous_geometry import (
        load_continuous_geometry, observed_joint_configuration,
    )
    from cais_spade_llm.resources.recovery_geometry import _configuration_after

    description = """<robot name="combined">
      <link name="base"/><link name="arm"/><link name="finger"/>
      <link name="peer_base"/><link name="peer_arm"/>
      <joint name="arm_joint" type="revolute">
        <parent link="base"/><child link="arm"/><axis xyz="0 0 1"/>
      </joint>
      <joint name="finger_joint" type="prismatic">
        <parent link="arm"/><child link="finger"/><axis xyz="0 1 0"/>
      </joint>
      <joint name="peer_joint" type="revolute">
        <parent link="peer_base"/><child link="peer_arm"/><axis xyz="0 0 1"/>
      </joint>
    </robot>"""
    observed = {"arm_joint": {"position": 0.}, "finger_joint": {"position": .1},
                "peer_joint": {"position": 1.}}
    model = load_continuous_geometry(description, root="base", reference_link="arm",
        joint_names=["arm_joint"], observed_joints=observed,
        root_pose=[0., 0., 0., 0., 0., 0., 1.], interpolation="linear")
    assert model["fixed_joint_positions"] == {"finger_joint": .1}
    changed = deepcopy(observed)
    changed["peer_joint"]["position"] = 2.
    changed["finger_joint"]["position"] = .02
    after = observed_joint_configuration(model, changed, [0., 0., 0., 0., 0., 0., 1.])
    records = {"owner": {"configuration": model}}
    _configuration_after(records, "owner", {"configuration_after": after})
    assert records["owner"]["configuration"]["fixed_joint_positions"] == {"finger_joint": .02}
    invalid = deepcopy(after)
    invalid["fixed_joint_positions"]["peer_joint"] = 2.
    with pytest.raises(ValueError, match="fixed joint identities"):
        _configuration_after(records, "owner", {"configuration_after": invalid})


@pytest.mark.parametrize("change,reason", [
    ("empty", "empty attachment"), ("wrong_part", "configured held_part"),
    ("duplicate", "ambiguous"), ("missing_geometry", "geometry is missing"),
])
def test_attached_part_capture_rejects_missing_or_conflicting_evidence(change, reason):
    observer, record = _held_observation_case()
    if change == "empty":
        record["attachments"] = []
    elif change == "wrong_part":
        record["attachments"][0]["model2"] = "invented_part"
    elif change == "duplicate":
        record["attachments"] *= 2
    else:
        record["models"]["gear_model"]["links"]["link"]["collisions"] = []
    with pytest.raises(ValueError, match=reason):
        observer.capture()


@pytest.mark.parametrize("initialize_reader", [False, True])
def test_new_failure_capture_retains_offline_physical_facts_without_cca(monkeypatch, initialize_reader):
    from cais_spade_llm.recovery_framework.recovery_observations import capture_observations
    from cais_spade_llm.resources.resource_safety_preparation import PrimitiveModel
    from test_continuous_motion import configuration

    observer, record = _held_observation_case()
    owner, runtime = observer.owner, observer.provider.runtime
    context = runtime.context
    context.run_id, context.revision = "capture", 4
    context.resources = {"ur5e-4": SimpleNamespace(valuation={"held_part": "gear_small"})}
    context.part_tracker["gear_small"]["processCompleted"] = []
    context.initial_product_states = deepcopy(context.part_tracker)
    context.requirements = {"gear_small": {"assembly": "assembly_board-v1"}}
    context.acknowledgements = {"pickup": {"task_id": "pickup", "status": "completed"}}
    context.transitions = [{"acknowledgement": deepcopy(context.acknowledgements["pickup"]),
                           "product_before": deepcopy(context.part_tracker), "product_after": deepcopy(context.part_tracker)}]
    context.inputs["geometry"]["parts"]["assembly_target_map"] = {"gear_small": "Gear_Plate/Gear_Shaft_1"}
    context.inputs["scene"] = {"safety_preparation": {"regions": {"assembly_board-v1": {"bounds": [[0., 1.]] * 3}}}}
    runtime.resource_agents, runtime.jids = [owner], {"ur5e-4": owner.jid}
    runtime.setup = {"execution_mode": "simulation",
                     "failure_scenario": {"scenario": "Part slippage", "resource_id": "ur5e-4"}}
    owner.recovery_safety_observer = observer
    owner.capture_recovery_safety_state = lambda **kwargs: observer.capture(native={
        "current_pose": record["models"]["robot"]["pose"], "joint_names": ["slide"], "joint_positions": [1.]})
    owner.get_recovery_safety_primitive_model = lambda: PrimitiveModel("captured", 2, {
        "continuous_motion": configuration(), "named_joint_poses": {"home": {"joint_names": ["slide"], "positions": [0.]}}}, lambda **kwargs: {})
    def entity(name, **kwargs):
        return {"name": name, "frame": "world", "pose": record["models"][name]["pose"],
                "instance_id": "physics", "simulation_time": 5., "simulation_stamp": 5.,
                "observed_monotonic": record["observed_monotonic"]}
    observer.provider.reader.entity = entity
    owner.get_recovery_entity_reader = lambda: entity
    initialize = Mock()
    if initialize_reader:
        del owner.recovery_safety_observer
        context.inputs["scene"]["safety_preparation"]["provider"] = "gazebo_idle_continuous_v1"
        def install(provider, *, model_execution):
            initialize(model_execution=model_execution, cca=provider.cca)
            owner.recovery_safety_observer = observer
        monkeypatch.setattr(LiveSafetyPreparation, "initialize", install)
    result = capture_observations(runtime, {"injection_status": "completed"})
    physical = result["physical_evidence"]
    assert physical["unresolved"] == []
    assert physical["snapshot"]["frame"] == "world"
    assert physical["snapshot"]["parts"]["gear_small"]["contained_by"] is None
    assert physical["snapshot"]["resources"]["ur5e-4"]["joint_positions"] == [1.]
    assert physical["resource_models"]["ur5e-4"]["named_joint_poses"]["home"]["positions"] == [0.]
    assert physical["geometry"]["parts"]["gear_small"]["local_bounds"]
    assert physical["geometry"]["parts"]["gear_small"]["target"] == "Gear_Plate/Gear_Shaft_1"
    assert physical["snapshot"]["parts"]["gear_small"]["processCompleted_complete"] is True
    assert result["acknowledgements"] == context.acknowledgements
    assert result["requirements"] == context.requirements
    assert not physical["dispatch_authorized"] and not physical["command_sent"]
    if initialize_reader:
        initialize.assert_called_once_with(model_execution=False, cca=None)
        assert runtime.recovery_observation_preparation.runtime is runtime
    context.acknowledgements.clear()
    record["attachments"].clear()
    assert result["acknowledgements"] and physical["resources"]["ur5e-4"]["attachment"]["records"]


def test_offline_physical_capture_cannot_initialize_live_readers():
    from cais_spade_llm.recovery_framework.recovery_observations import capture_recovery_physical_evidence

    runtime = SimpleNamespace(part_slippage_recovery=SimpleNamespace(offline=True))
    with pytest.raises(ValueError, match="Offline recovery cannot"):
        capture_recovery_physical_evidence(runtime)


@pytest.mark.parametrize("change", ["missing_initial", "missing_transition", "altered_completion", "missing_ack"])
def test_completion_capture_never_derives_assembly_from_a_state_label(change):
    from cais_spade_llm.recovery_framework.recovery_observations import _completion_ledger_evidence

    initial = {"gear_small": {"processCompleted": []}}
    ledger = {"processCompleted": [], "state": "assembled"}
    ack = {"pickup": {"task_id": "pickup", "status": "completed"}}
    transitions = [{"acknowledgement": ack["pickup"], "product_before": deepcopy(initial), "product_after": deepcopy(initial)}]
    if change == "missing_initial":
        initial.clear()
    elif change == "missing_transition":
        transitions.clear()
    elif change == "altered_completion":
        ledger["processCompleted"] = [{"process": "assembly"}]
    else:
        ack.clear()
    result = _completion_ledger_evidence("gear_small", ledger, initial, ack, transitions, run_id="run", revision=1)
    assert result["complete"] is False
    assert result["reason"]


def test_outline_named_pose_preview_is_isolated_and_has_no_interval_proof():
    from test_continuous_motion import configuration
    from cais_spade_llm.resources.resource_safety_preparation import PrimitiveModel

    model = configuration()
    named = {"home": {"joint_names": ["slide"], "positions": [2.]}}
    controller = SimpleNamespace(get_recovery_safety_primitive_model=lambda: PrimitiveModel(
        "joint_trajectory_continuous", 2, {"continuous_motion": model, "named_joint_poses": named}, lambda **kwargs: {}))
    for name in ("capture_recovery_safety_state", "_prepare_cartesian_motion", "_execute_simulation_motion_plan",
                 "move_to_named_pose", "attach_part", "detach_part"):
        setattr(controller, name, Mock(side_effect=AssertionError("Preview contacted a live interface")))
    start = {"joint_names": ["slide"], "joint_positions": [0.], "held_part": None,
             "current_pose": [0., 0., 0., 0., 0., 0., 1.]}
    scene = {"snapshot": {"frame": "world", "resources": {"ur5e-4": start}, "parts": {}},
             "geometry": {"frame": "world", "resources": {"ur5e-4": {}}, "parts": {},
                          "regions": {"assembly_board-v1": {"bounds": [[-.1, .1]] * 3}}},
             "resource_models": {"ur5e-4": {"configuration": model, "named_joint_poses": named}}}
    command = {"primitive": "move_to_named_pose", "params": {"pose_name": "home"}}
    support = {"semantics": "primitive_transition", "primitives": [command], "trace": [command]}
    before = deepcopy(scene)
    result = GazeboPickPlaceController.prepare_recovery_outline_preview(controller, resource_id="ur5e-4",
        primitive_support=support, projected_start=start, isolated_scene=scene)
    assert result["status"] == "projected", result
    assert result["projected_snapshot"]["resources"]["ur5e-4"]["joint_positions"] == [2.]
    assert result["interval_complete"] is False
    assert not result["command_sent"] and not result["dispatch_authorized"]
    assert scene == before
    scene["resource_models"]["ur5e-4"]["configuration"] = {**model, "root_xyz": [100, 0, 0]}
    blocked = GazeboPickPlaceController.prepare_recovery_outline_preview(controller, resource_id="ur5e-4",
        primitive_support=support, projected_start=start, isolated_scene=scene)
    assert blocked["feasibility_status"] == "NEEDS_CONTEXT"


@pytest.mark.parametrize("offline", [False, True])
@pytest.mark.parametrize("change", [None, "unknown_model", "assembly_slot", "missing_contract"])
def test_outline_custody_preview_uses_declared_geometry_without_attachment_calls(change, offline):
    from test_continuous_motion import configuration
    from cais_spade_llm.resources.resource_safety_preparation import PrimitiveModel

    model = configuration()
    model["joints"].append({"name": "finger", "parent": "body", "child": "finger_link",
        "type": "prismatic", "axis": [0, 1, 0], "xyz": [0, 0, 0], "rpy": [0, 0, 0], "frozen_position": .1})
    model["components"].append({"id": "finger", "link": "finger_link", "bounds": [[-.01, .01]] * 3})
    contract = {"kind": "gazebo_link_attacher", "gripper": {"joint": "finger", "open": .1, "closed": .02},
                "joint_positions": {"open": {"finger": .1}, "closed": {"finger": .02}},
                "preserves_part_pose_on_attachment": True, "release_settling_modeled": False}
    own = {"continuous_motion": model, "named_joint_poses": {}, "custody": contract}
    if change == "missing_contract":
        own["custody"] = None
    controller = SimpleNamespace(get_recovery_safety_primitive_model=lambda: PrimitiveModel(
        "joint_trajectory_continuous", 2, own, lambda **kwargs: {}))
    for name in ("capture_recovery_safety_state", "attach_part", "detach_part", "open_gripper", "close_gripper"):
        setattr(controller, name, Mock(side_effect=AssertionError("Custody preview dispatched")))
    start = {"joint_names": ["slide"], "joint_positions": [0.], "held_part": None,
             "gripper_state": "open", "current_pose": [0., 0., 0., 0., 0., 0., 1.]}
    part = {"model_name": "gear_model", "current_pose": [.1, 0., 0., 0., 0., 0., 1.], "contained_by": None}
    scene = {"snapshot": {"frame": "world", "resources": {"ur5e-4": start}, "parts": {"gear_small": part}},
             "geometry": {"frame": "world", "resources": {"ur5e-4": {}},
                          "parts": {"gear_small": {"local_bounds": [[-.01, .01]] * 3}}, "regions": {}},
             "resource_models": {"ur5e-4": {"configuration": model, "named_joint_poses": {}}}}
    commands = [{"primitive": name, "params": {"model_name": "gear_model", "part_name": "gear_small"}}
                for name in ("grasp_part", "release_part")]
    if change == "unknown_model":
        commands[0]["params"]["model_name"] = "unknown"
    elif change == "assembly_slot":
        commands[1]["params"]["assembly_slot"] = {"destination_location": "assembly_board-v1"}
    support = {"semantics": "primitive_transition", "primitives": commands, "trace": deepcopy(commands)}
    support["trace"].insert(1, {"primitive": None, "task_contract": "pick_approach",
                                 "declared_effects": {"resource_state": {"set": "picked"}}})
    before = deepcopy(scene)
    if offline:
        from cais_spade_llm.resources.resource_safety_preparation import prepare_recovery_outline_preview

        descriptor = json.loads(json.dumps(controller.get_recovery_safety_primitive_model().descriptor()))
        result = prepare_recovery_outline_preview(resource_id="ur5e-4", primitive_support=support,
            projected_start=start, isolated_scene=scene, primitive_model=descriptor)
    else:
        result = GazeboPickPlaceController.prepare_recovery_outline_preview(controller, resource_id="ur5e-4",
            primitive_support=support, projected_start=start, isolated_scene=scene)
    assert not result["command_sent"] and not result["dispatch_authorized"]
    assert scene == before
    if change:
        assert result["feasibility_status"] == "NEEDS_CONTEXT", result
    else:
        assert result["status"] == "projected", result.get("unresolved")
        assert result["interval_complete"] is False
        assert result["projected_snapshot"]["resources"]["ur5e-4"]["held_part"] is None
        assert result["step_evidence"][0]["custody_effects"][0]["grasp_transform"] == pytest.approx(
            [.1, 0., 0., 0., 0., 0., 1.])
        assert result["step_evidence"][0]["configuration_after"]["joints"][-1]["frozen_position"] == .02
        assert result["resource_models"]["ur5e-4"]["configuration"]["joints"][-1]["frozen_position"] == .1
        if offline:
            first = prepare_recovery_outline_preview(resource_id="ur5e-4",
                primitive_support={**support, "primitives": commands[:1], "trace": support["trace"][:1]},
                projected_start=start, isolated_scene=scene, primitive_model=descriptor)
            following = {**deepcopy(scene), "snapshot": first["projected_snapshot"],
                         "resource_models": first["resource_models"]}
            second = prepare_recovery_outline_preview(resource_id="ur5e-4",
                primitive_support={**support, "primitives": commands[1:], "trace": [commands[1]]},
                projected_start=following["snapshot"]["resources"]["ur5e-4"],
                isolated_scene=following, primitive_model=descriptor)
            assert second["status"] == "projected", second.get("unresolved")
            assert second["projected_snapshot"]["resources"]["ur5e-4"]["held_part"] is None


def test_outline_custody_configuration_retains_declared_mimic_joint_values():
    from cais_spade_llm.resources.robot.gazebo_pick_place_controller import _outline_custody_configuration

    description = '<robot><joint name="finger"/><joint name="other"><mimic joint="finger" multiplier="-1" offset=".02"/></joint></robot>'
    result = _outline_custody_configuration(description, {"joint": "finger", "open": .1, "closed": .02},
        {"joint_names": ["slide"], "joints": [{"name": "finger"}, {"name": "other"}]}, enabled=True)
    assert result["joint_positions"]["open"]["other"] == pytest.approx(-.08)
    assert result["joint_positions"]["closed"]["other"] == pytest.approx(0.)
    assert result["joint_bounds"] == {}


def test_outline_custody_configuration_retains_full_urdf_bounds():
    from cais_spade_llm.resources.robot.gazebo_pick_place_controller import _outline_custody_configuration

    description = ('<robot><joint name="finger"><limit lower="0" upper=".12"/></joint>'
                   '<joint name="other"><mimic joint="finger" multiplier="-1" offset=".02"/></joint></robot>')
    result = _outline_custody_configuration(description, {"joint": "finger", "open": .1, "closed": .02},
        {"joint_names": ["slide"], "joints": [{"name": "finger"}, {"name": "other"}]}, enabled=True)
    assert result["joint_bounds"]["finger"] == [0., .12]
    assert result["joint_bounds"]["other"] == pytest.approx([-.1, .02])
    assert result["release_settling_modeled"] is False


def test_outline_state_only_effect_does_not_require_invented_motion():
    from cais_spade_llm.resources.resource_safety_preparation import prepare_recovery_outline_preview

    start = {"resource_state": "failed", "contained_parts": []}
    scene = {"snapshot": {"frame": "world", "resources": {"M1": start}, "parts": {}},
             "geometry": {"frame": "world", "resources": {"M1": {"stationary_only": True,
                 "component_bounds": [{"id": "machine", "bounds": [[0., 1.]] * 3}]}},
                 "parts": {}, "regions": {}}, "resource_models": {}}
    command = {"primitive": "reset_fault", "params": {}}
    support = {"semantics": "primitive_transition", "primitives": [command], "trace": [{**command,
        "declaration_fingerprint": "owner-declaration", "declared_effects": {"resource_state": {"set": "idle"}}}]}
    descriptor = {"configuration": {"geometry_unchanged": [deepcopy(command)]}}
    result = prepare_recovery_outline_preview(resource_id="M1", primitive_support=support,
        projected_start=start, isolated_scene=scene, primitive_model=descriptor)
    assert result["status"] == "projected", result.get("unresolved")
    assert result["step_evidence"][0]["geometry_unchanged"] is True
    unavailable = prepare_recovery_outline_preview(resource_id="M1", primitive_support=support,
        projected_start=start, isolated_scene=scene, primitive_model={"configuration": {}})
    assert unavailable["feasibility_status"] == "NEEDS_CONTEXT"
    support["trace"][0]["declared_effects"]["current_pose"] = {"set": [0., 0., 0.]}
    result = prepare_recovery_outline_preview(resource_id="M1", primitive_support=support,
        projected_start=start, isolated_scene=scene, primitive_model=descriptor)
    assert result["feasibility_status"] == "NEEDS_CONTEXT"


def _static_part_geometry_case():
    pose = [0., 0., 1., 0., 0., 0., 1.]
    models = {'assembly_board_v1': {'static': True, 'pose': pose, 'links': {'link': {'collisions': []}}}}
    for name, bounds in [('Gear_Plate', [[-.2, .1], [0., .2], [1., 1.1]]),
                         ('GMC_Laser_Plate_Virtual', [[-.25, .25], [-.25, .25], [.99, 1.01]])]:
        models[name] = {'static': True, 'pose': pose, 'links': {'link': {'collisions': [{'id': name, 'bounds': bounds}]}}}
    return ({'instance_id': 'physics', 'simulation_time': 1., 'models': models},
            {'models': ['GMC_Laser_Plate_Virtual', 'Gear_Plate'], 'requires_static': True},
            {'name': 'assembly_board_v1', 'pose': pose, 'instance_id': 'physics', 'simulation_time': 1.})


def test_empty_carrier_uses_explicit_observed_constituents_without_guessed_dimensions():
    from cais_spade_llm.recovery_framework.gazebo_safety_preparation import (
        _configured_static_part_geometry,
    )

    record, declaration, part = _static_part_geometry_case()
    result = _configured_static_part_geometry(record, declaration, part)
    assert result['footprint'][0][0] < -.25 and result['footprint'][0][1] > .25
    assert result['footprint'][2][0] < -.01 and result['footprint'][2][1] > .1
    assert set(result) == {'frame', 'footprint'}


@pytest.mark.parametrize('change', ['missing', 'empty', 'moving', 'carrier', 'pose', 'duplicate', 'undeclared', 'instance', 'stamp'])
def test_incomplete_constituent_geometry_is_unavailable(change):
    from cais_spade_llm.recovery_framework.gazebo_safety_preparation import (
        _configured_static_part_geometry,
    )

    record, declaration, part = _static_part_geometry_case()
    if change == 'missing':
        del record['models']['Gear_Plate']
    elif change == 'empty':
        record['models']['Gear_Plate']['links']['link']['collisions'] = []
    elif change == 'moving':
        record['models']['Gear_Plate']['static'] = False
    elif change == 'carrier':
        record['models']['assembly_board_v1']['static'] = False
    elif change == 'pose':
        part = deepcopy(part)
        part['pose'][0] += .1
    elif change == 'duplicate':
        declaration['models'].append('Gear_Plate')
    elif change == 'instance':
        part['instance_id'] = 'another-launch'
    elif change == 'stamp':
        part['simulation_time'] = 0.
    else:
        declaration['requires_static'] = False
    with pytest.raises((ValueError, KeyError)):
        _configured_static_part_geometry(record, declaration, part)


def test_static_part_geometry_retains_sources_and_detects_repositioned_fixture():
    from cais_spade_llm.recovery_framework.gazebo_safety_preparation import LiveSafetyPreparation

    record, declaration, part = _static_part_geometry_case()
    runtime = SimpleNamespace(context=SimpleNamespace(inputs={'geometry': {}}))
    provider = LiveSafetyPreparation(runtime, None, {'regions': {}, 'part_geometry': {'assembly_board-v1': declaration}})
    provider.reader = SimpleNamespace(snapshot=lambda **_: record, entity=lambda *_, **__: deepcopy(part), idle_goals=lambda: {})
    parts = {'assembly_board-v1': deepcopy(part)}
    unresolved = []
    geometry = provider.geometry({}, parts, unresolved)
    assert not unresolved
    source = parts['assembly_board-v1']['geometry_source']
    assert source['declaration'] == declaration and source['instance_id'] == 'physics'
    checkpoint = {'checkpoint_id': 'checkpoint', 'parts': parts, 'observations': {}, 'geometry': geometry}
    provider.revalidate(checkpoint)
    record['models']['Gear_Plate']['links']['link']['collisions'][0]['bounds'][0][0] -= .2
    with pytest.raises(ValueError, match='geometry changed'):
        provider.revalidate(checkpoint)


@pytest.mark.parametrize('target', ['Gear_Plate/Gear_Shaft_1', 'Gear_Plate/Gear_Shaft_2', None])
def test_observed_part_geometry_retains_exact_configured_assembly_target(target):
    from cais_spade_llm.agents.central_controller.reviewed_primitive_program_safety import _scope_geometry
    from cais_spade_llm.recovery_framework.gazebo_safety_preparation import _envelope

    record, _, part = _static_part_geometry_case()
    part['name'] = 'Gear_Plate'
    targets = {} if target is None else {'gear_small': target}
    runtime = SimpleNamespace(context=SimpleNamespace(inputs={
        'geometry': {'parts': {'assembly_target_map': targets}},
    }))
    provider = LiveSafetyPreparation(runtime, None, {'regions': {}})
    provider.reader = SimpleNamespace(snapshot=lambda: record)
    unresolved = []
    geometry = provider.geometry({}, {'gear_small': part}, unresolved)
    assert not unresolved
    shape = geometry['parts']['gear_small']
    assert shape['footprint'] == _envelope(record['models']['Gear_Plate']['links'], part['pose'])
    assert shape['frame'] == 'world'
    assert shape.get('target') == target
    binding = {'part': 'gear_small', 'target': 'Gear_Plate/Gear_Shaft_1'}
    if target == binding['target']:
        _scope_geometry(binding, geometry, {}, {})
    else:
        with pytest.raises(ValueError, match='Scoped assembly target differs'):
            _scope_geometry(binding, geometry, {}, {})


def _slippage_gazebo_script(tmp_path):
    root = Path(__file__).resolve().parents[1]
    script = runpy.run_path(str(root / 'scripts/check_part_slippage_gazebo.py'))
    archived = script["COMPANIONS"]
    current = tmp_path / "current_companions"
    current.mkdir(exist_ok=True)
    for case in script["CASES"]:
        companion = json.loads((archived / (case + ".json")).read_text())
        companion["source_fixture"]["path"] = str(
            (archived / companion["source_fixture"]["path"]).resolve())
        # This is a new preflight input, not rewritten historical acceptance evidence.
        source = root / companion["predefined_safety"]["path"]
        companion["predefined_safety"]["sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
        (current / (case + ".json")).write_text(json.dumps(companion))
    load = script["load_companion"]
    script["load_companion"] = lambda name, directory=current: load(name, directory)
    script["COMPANIONS"] = current
    return script


@pytest.mark.parametrize('case', ['mutex', 'precedence', 'safe'])
def test_gazebo_companions_preserve_events_without_importing_synthetic_motion(tmp_path, case):
    companion = _slippage_gazebo_script(tmp_path)['load_companion'](case)
    assert len(companion['outline_events']) == 7
    assert sum(len(event['primitive_steps']) for event in companion['outline_events']) == 19
    assert companion['diagnostic_cca_bypass'] is False
    assert companion['native_preparation']['programs'] == []
    assert companion['native_preparation']['status'] == 'NEEDS_CONTEXT'
    assert not companion['acceptance_complete']
    for event in companion['outline_events']:
        assert 'start_time' not in event and 'end_time' not in event
        assert all(set(step) == {'primitive', 'source'} for step in event['primitive_steps'])


@pytest.mark.parametrize('change', ['event_name', 'predecessors', 'source', 'bypass', 'hash'])
def test_gazebo_companion_rejects_changed_identity_or_authority(tmp_path, change):
    script = _slippage_gazebo_script(tmp_path)
    companion = script['load_companion']('mutex')
    companion['source_fixture']['path'] = str(script['COMPANIONS'] / companion['source_fixture']['path'])
    if change == 'event_name':
        companion['outline_events'][0]['event_name'] = 'changed'
    elif change == 'predecessors':
        companion['outline_events'][0]['predecessors'] = ['changed']
    elif change == 'source':
        companion['outline_events'][0]['primitive_steps'][0]['source']['primitive_index'] = 42
    elif change == 'bypass':
        companion['diagnostic_cca_bypass'] = True
    else:
        companion['source_fixture']['sha256'] = 'changed'
    (tmp_path / 'mutex.json').write_text(json.dumps(companion))
    with pytest.raises(ValueError):
        script['load_companion']('mutex', tmp_path)


@pytest.mark.parametrize('case', ['mutex', 'precedence', 'safe'])
def test_live_preflight_cannot_count_missing_evidence_as_a_safety_rejection(tmp_path, monkeypatch, case):
    from cais_spade_llm.recovery_framework import gazebo_safety_preparation

    script = _slippage_gazebo_script(tmp_path)
    companion = script['load_companion'](case)
    provider = SimpleNamespace(initialize=lambda: None, validate_idle=Mock(side_effect=ValueError('Missing goals')))
    cca = SimpleNamespace(recovery_safety_preparation_provider=provider)
    runtime = SimpleNamespace(context=SimpleNamespace(run_id='observed-run'), resource_agents=[
        SimpleNamespace(agent_name=rid, jid=jid) for rid, jid in
        [('ur5e-3', 'recovery-resource-3@localhost'), ('ur5e-4', 'recovery-resource-4@localhost')]
    ])
    monkeypatch.setattr(gazebo_safety_preparation, 'install_live_preparation', lambda *_: None)
    monkeypatch.setattr(gazebo_safety_preparation, 'capture_checkpoint', lambda *_: {
        'unresolved': [{'reason': 'Missing goals'}], 'observations': {},
    })
    result = asyncio.run(script['capture_case'](companion, runtime, cca))
    assert result['status'] == 'NEEDS_CONTEXT' and result['phase'] == 'live_preflight'
    assert result['cca_decision'] is None and result['safety_verdict'] is None
    assert result['ap_evidence'] == [] and not result['acceptance_complete']
    assert not result['dispatch_authorized'] and not result['recording_started']
    assert result['published_videos'] == [] and result['staging_performed'] is False


@pytest.mark.parametrize('field', ['current_pose','joint_positions','attachment','footprint'])
def test_live_reobservation_rejects_even_small_unmodeled_changes(field):
    from cais_spade_llm.recovery_framework.gazebo_safety_preparation import LiveSafetyPreparation

    initial = {'launch_id':'run', 'custody_complete':True, 'attachment':{'revision':0},
               'stationary_contract':{'kind':'idle_commanded_hold'},
               'current_pose':[0,0,0,0,0,0,1], 'footprint':[[-.1,.1]]*3,
               'observation_state':{'joint_positions':[0.], 'current_pose':[0,0,0,0,0,0,1]}}
    changed = deepcopy(initial)
    if field == 'current_pose':
        changed[field][3] = 1e-12
    elif field == 'joint_positions':
        changed['observation_state'][field][0] = 1e-12
    elif field == 'attachment':
        changed[field]['revision'] = 1
    else:
        changed[field][0][0] -= 1e-12
    provider = LiveSafetyPreparation(None,None,{})
    provider.reader = SimpleNamespace(snapshot=lambda **_: {}, idle_goals=lambda: {})
    provider.observers = {'resource':SimpleNamespace(owner=SimpleNamespace(capture_recovery_safety_state=lambda **_:changed))}
    checkpoint = {'checkpoint_id':'checkpoint','observations':{'resource':{'physical':initial}},'parts':{}}
    with pytest.raises(ValueError,match='changed|moved'):
        provider.revalidate(checkpoint)


def test_preparation_cannot_overwrite_execution_feedback_arriving_during_planning(controller):
    controller._last_command_evidence = {"previous": "execution"}
    response = controller._cart_client.call_async.return_value
    def plan(request):
        controller._last_command_evidence = {"new": "observed execution feedback"}
        return response
    controller._cart_client.call_async.side_effect = plan
    result = prepare_ur5e_motion(controller, primitive="move_cartesian",
        params={"x": 1.5, "y": 0., "z": 1.}, start=_start(controller), binding={})
    assert result["status"] == "prepared", result
    assert controller._last_command_evidence == {"new": "observed execution feedback"}
    assert controller._planning_wall_time_sec == 0.


@pytest.mark.parametrize("change", ["orientation", "attachment", "joint_names", "release_part", "malformed", "partial", "task_context"])
def test_unsupported_preparation_never_dispatches(controller, change):
    params, start, primitive = {"x": 1.5, "y": 0., "z": 1.}, _start(controller), "move_cartesian"
    if change == "orientation":
        params.update(qx=0., qy=0., qz=1., qw=0.)
    elif change == "attachment":
        start["attachment"]["model_name"] = "gear_small"
    elif change == "joint_names":
        start["joint_names"] = ["other"] * 6
    elif change == "release_part":
        primitive, params = "release_part", {"part_name": "KET4_Square_4mm"}
    elif change == "partial":
        params["qx"] = 0.
    elif change == "task_context":
        controller._robot_task_step = ("place_approach", "move_above_destination")
    else:
        params["target"] = [1, 2, 3, 0, 0, 0, 1]
    result = prepare_ur5e_motion(controller, primitive=primitive, params=params, start=start, binding={})
    assert result["status"] == "NEEDS_CONTEXT"
    controller._cart_client.call_async.assert_not_called()


def _case(controller, *, completed=True, completion_time=None):
    case = _gear_precedence_case(completion_time=completion_time)
    document = _document()
    case.update({key: document[key] for key in ("catalog", "requirement_scopes")})
    for field in ("snapshot", "geometry"):
        rows = case[field]["resources"]
        rows["ur5e-3"], rows["KMR"] = rows["KMR"], rows["ur5e-3"]
    case["snapshot"]["resources"].pop("assembly_board-v1")
    case["stationary"].pop("assembly_board-v1", None)
    state = case["snapshot"]["resources"]["ur5e-3"]
    state.update(resource_id="ur5e-3", resource_jid="ur5e-3@localhost", resource_type="ur5e")
    case["stationary"]["KMR"], case["stationary"]["ur5e-3"] = [[0, 1]], []
    if completed:
        case["snapshot"]["parts"]["gear_small"]["processCompleted"].append(
            {"process": "assembly", "target": "Gear_Plate/Gear_Shaft_1"})
    program = case["programs"][0]
    program["resource_id"] = "ur5e-3"
    trace = program["step_results"][0]
    source = {"outline_id": "UR5E_PREPARE_SEQ1", "des_event_id": "UR5E_PREPARE_EVENT1",
              "event_name": "recover_KET4_Square_4mm", "step_index": 0,
              "resource_id": "ur5e-3", "resource_jid": "ur5e-3@localhost"}
    target = trace["resolved_params"]["target"]
    params = dict(zip(("x", "y", "z", "qx", "qy", "qz", "qw"), target, strict=True))
    trace.update(resolved_params=params, source=source)
    program["primitive_steps"][0].update(params=deepcopy(params), source=deepcopy(source))
    raw = {"resource_id": "ur5e-3", "primitive_steps": deepcopy(program["primitive_steps"])}
    request = {"recovery_id": "UR5E_PREPARE", "programs": [raw]}
    event = {key: source[key] for key in ("outline_id", "des_event_id", "event_name")}
    event.update(resource_id="ur5e-3", primitive_step_indices=[0], predecessors=[])
    case["scene"]["safety_geometry"] = deepcopy(case["geometry"])
    choice = {"id": "start", "starts": {event["outline_id"]: 0},
              "programs": deepcopy(case["programs"]), "stationary": deepcopy(case["stationary"])}
    composition = {"grounding_inputs": case, "recovery_events": [event], "running_work": [],
                   "event_start_choices": [choice], "completion": {
                       "resources": {"ur5e-3": {"held_part": "KET4_Square_4mm", "current_pose": target}},
                       "parts": {"KET4_Square_4mm": {"contained_by": None}}}}
    return composition, request


def _coverage(program, planned):
    evidence = program["step_results"][0]["model_evidence"]
    evidence.update(primitive_contract="ur5e", joint_trajectory=deepcopy(planned["joint_trajectory"]),
                    preparation_id=planned["preparation_id"])
    evidence["cartesian_coverage"] = {
        "source": "synthetic test contract", "interpolation": "piecewise_linear_fixed_orientation",
        "covers_resource_envelope": True,
        "joint_trajectory_fingerprint": fingerprint(evidence["joint_trajectory"]),
        "trajectory_fingerprint": cartesian_coverage_fingerprint(evidence["trajectory"], program["step_results"][0]["start_time"])}


def _harness(controller, *, completed=True):
    case, request = _case(controller, completed=completed)
    grounding = case["grounding_inputs"]
    parts = grounding["snapshot"]["parts"]
    owners = []
    for rid, state in grounding["snapshot"]["resources"].items():
        physical = {"observation_state": deepcopy(state), "current_pose": state["current_pose"],
                    "attachment": {"model_name": state.get("held_part"), "observed_attachment_complete": True},
                    "launch_id": "synthetic-launch", "frame": "world"}
        if rid == "ur5e-3":
            physical.update(_start(controller))
        owner = SimpleNamespace(agent_name=rid, jid=rid + "@localhost", execution_mode="dry_run",
                                _controller=controller if rid == "ur5e-3" else None,
                                _robot_motion_lock=threading.Lock(), static_capabilities={})
        owner.get_recovery_physical_snapshot = lambda state=state: {"snapshot": deepcopy(state)}
        owner.capture_recovery_safety_state = lambda max_age=2., row=physical: {
            **deepcopy(row), "observed_monotonic": time.monotonic()}
        owner.prepare_recovery_safety_program = MethodType(RobotAgent.prepare_recovery_safety_program, owner)
        for name in ("recovery_safety_configuration", "get_recovery_safety_primitive_model",
                     "get_recovery_entity_reader", "validate_recovery_safety_step"):
            setattr(owner, name, MethodType(getattr(RobotAgent, name), owner))
        if rid == "ur5e-3":
            controller.capture_recovery_safety_state = owner.capture_recovery_safety_state
        owners.append(owner)
    context = SimpleNamespace(
        admission_lock=threading.RLock(), run_id="synthetic-run", revision=0,
        inputs={"scene": grounding["scene"], "geometry": {"parts": {name: {"model_name": name} for name in parts}}},
        part_tracker=deepcopy(parts), pending_tasks={"M1_delivery": {"status": "pending"}}, reservations={},
        acknowledgements={}, resources={rid: SimpleNamespace(revision=0, model={}) for rid in grounding["snapshot"]["resources"]},
        snapshot=lambda: deepcopy(grounding["snapshot"]["resources"]))
    runtime = SimpleNamespace(context=context, resource_agents=owners, admission=None,
                              jids={owner.agent_name: str(owner.jid) for owner in owners})
    for owner in owners:
        owner.environment_runtime = runtime

    def entity(query):
        values = parts[query.name]["current_pose"]
        pose = controller._make_pose(*values[:3], controller._make_orientation(*values[3:]))
        return SimpleNamespace(success=True, state=SimpleNamespace(pose=pose),
                               header=SimpleNamespace(stamp=SimpleNamespace(sec=1, nanosec=0)))

    controller._get_state_client.call_async = entity
    cca = SimpleNamespace(safety_monitor=OnlineSafetyMonitor({}, [], tools_catalog=[]),
                          recovery_safety_scopes={}, recovery_composition_admissions={}, allow_mock_recovery_execution=True)

    def evidence(*, checkpoint, prepared, request):
        inputs = deepcopy(case)
        _coverage(inputs["event_start_choices"][0]["programs"][0], prepared[0]["steps"][0])
        return {"synthetic": True, "checkpoint_id": checkpoint["checkpoint_id"],
                "prepared_fingerprint": fingerprint(prepared), "composition_inputs": inputs,
                "physical_rule_activation": "prospective"}

    cca.recovery_safety_preparation_provider = evidence
    bridge = SimpleNamespace(cca=cca, resource_agents=owners, selected_safety_file=str(_SOURCE))
    return bridge, runtime, request


@pytest.mark.parametrize("completed,expected", [(True, "allowed"), (False, "held")])
def test_real_preparation_adapter_checks_detached_composition_without_admission(controller, tmp_path, completed, expected):
    bridge, runtime, request = _harness(controller, completed=completed)
    before = _runtime_record(runtime, bridge.cca)
    result = prepare_and_check(bridge, request, output_root=tmp_path, max_age=30, budget=Budget(seconds=30))
    assert result["status"] == expected, result.get("unresolved") or result.get("analysis")
    assert result["dispatch_authorized"] is False
    assert result["analysis"]["scope"]["included_resources"] == sorted(runtime.jids)
    assert len(result["analysis"]["scope"]["included_specifications"]) == 67
    assert _runtime_record(runtime, bridge.cca) == before
    assert not bridge.cca.recovery_composition_admissions
    assert runtime.context.pending_tasks == {"M1_delivery": {"status": "pending"}}
    assert json.loads(Path(result["artifact_path"]).read_text())["fingerprint"] == result["fingerprint"]
    controller._send_simulation_joint_trajectory.assert_not_called()


def test_checkpoint_retains_prepared_sessions_and_physical_history_without_owner_objects():
    context = SimpleNamespace(
        run_id="run", revision=0, inputs={}, part_tracker={}, resources={},
        pending_tasks={}, reservations={}, acknowledgements=[], snapshot=lambda: {},
    )
    runtime = SimpleNamespace(context=context, admission=None, resource_agents=[])
    graph = {"node": "initial", "path": [], "ledger": [], "grants": {},
             "invalid_reason": "", "physical_history": "history", "time_exact": "0"}
    work = {"grant": {"task_id": "task"}, "proof": {"history_revision": 0},
            "reservation": {"token": "reservation"}, "owner": threading.Lock(),
            "executions": {}, "complete": False, "plan_events": {"start"}}
    prepared = {"request": {"recovery_id": "recovery"}, "product_jid": "product@localhost",
                "tasks": {"task": {"resource_jid": "resource@localhost"}}, "refs": {"task": {}},
                "registration": {"registration_only": True}, "grants": {"task": work},
                "complete": False, "completed_tasks": set(), "invalid_reason": ""}
    monitor = OnlineSafetyMonitor({}, [], tools_catalog=[])
    physical = monitor.physical_monitor
    physical.rules = [{"rule_id": "SAFE_shared_area_mutex"}]
    physical.states = {"SAFE_shared_area_mutex": {"state_b", "state_a"}}
    physical.history = [{"revision": 0, "execution_evidence": {"task_id": "prior"}}]
    physical.revision = 1
    cca = SimpleNamespace(safety_monitor=monitor, recovery_safety_scopes={},
                          recovery_composition_admissions={"scope": SimpleNamespace(
                              sessions={"graph": graph, "prepared": prepared})})

    before = _runtime_record(runtime, cca)
    json.dumps(before, allow_nan=False)
    assert before["physical_sessions"]["scope"]["graph"] == graph
    saved = before["physical_sessions"]["scope"]["prepared"]
    assert saved["completed_tasks"] == []
    assert saved["grants"]["task"] == {
        "grant": {"task_id": "task"}, "proof": {"history_revision": 0},
        "reservation": {"token": "reservation"}, "executions": {},
        "complete": False, "plan_events": ["start"],
    }
    assert before["monitors"][0]["physical_monitor"] == {
        "rules": physical.rules, "states": {"SAFE_shared_area_mutex": ["state_a", "state_b"]},
        "history": physical.history, "revision": 1, "invalid_reason": "",
    }
    work["executions"][0] = {"success": True, "observations": {"joint_positions": [0.]}}
    assert _runtime_record(runtime, cca) != before
    assert saved["grants"]["task"]["executions"] == {}
    work["executions"].clear()
    prepared["completed_tasks"].add("task")
    assert _runtime_record(runtime, cca) != before
    prepared["completed_tasks"].clear()
    physical.history.append({"revision": 1, "execution_evidence": {"task_id": "task"}})
    assert _runtime_record(runtime, cca) != before
    assert len(before["monitors"][0]["physical_monitor"]["history"]) == 1


@pytest.mark.parametrize("change", ["stale", "participant", "program", "completion", "coverage", "synthetic_live", "revision", "history"])
def test_missing_changed_or_fabricated_evidence_cannot_pass(controller, tmp_path, change):
    bridge, runtime, request = _harness(controller)
    provider = bridge.cca.recovery_safety_preparation_provider

    def altered(**kwargs):
        evidence = provider(**kwargs)
        inputs = evidence["composition_inputs"]
        if change == "program":
            inputs["event_start_choices"][0]["programs"][0]["primitive_steps"][0]["params"]["x"] = 999
        elif change == "completion":
            inputs["grounding_inputs"]["snapshot"]["parts"]["gear_small"]["processCompleted"].append({"process": "fabricated"})
        elif change == "coverage":
            inputs["event_start_choices"][0]["programs"][0]["step_results"][0]["model_evidence"].pop("cartesian_coverage")
        elif change == "revision":
            runtime.context.revision += 1
        elif change == "history":
            evidence.pop("physical_rule_activation")
        return evidence

    bridge.cca.recovery_safety_preparation_provider = altered
    if change == "synthetic_live":
        for owner in runtime.resource_agents:
            owner.execution_mode = "simulation"
    if change == "participant":
        runtime.resource_agents.pop()
    if change == "stale":
        owner = runtime.resource_agents[0]
        capture = owner.capture_recovery_safety_state
        owner.capture_recovery_safety_state = lambda **kwargs: {**capture(**kwargs), "observed_monotonic": 0}
    result = prepare_and_check(bridge, request, output_root=tmp_path, max_age=30, budget=Budget(seconds=30))
    assert result["status"] in {"NEEDS_CONTEXT", "inconclusive"}, result
    assert result["dispatch_authorized"] is False
    assert not bridge.cca.recovery_composition_admissions


def test_checkpoint_capture_does_not_invent_fixed_resource_grippers(controller):
    bridge, runtime, _ = _harness(controller)
    checkpoint = capture_checkpoint(runtime, bridge.cca)
    for rid, row in checkpoint["geometry"]["resources"].items():
        if row.get("stationary_only"):
            state = checkpoint["observations"][rid]["physical"]["observation_state"]
            assert not {"gripper_state", "held_part", "grasp_transform"} & state.keys()
    assert len(checkpoint["observations"]) == 12


def test_ur5e_joint_evidence_alone_never_becomes_straight_tcp_motion(controller):
    case, _ = _case(controller)
    program = case["grounding_inputs"]["programs"][0]
    prepared = prepare_ur5e_motion(controller, primitive="move_cartesian",
        params=program["primitive_steps"][0]["params"], start=_start(controller), binding={})
    _coverage(program, prepared)
    assert _grounded_check(case["grounding_inputs"], trace_complete=True)["status"] == "satisfied"
    program["step_results"][0]["model_evidence"].pop("cartesian_coverage")
    assert _grounded_check(case["grounding_inputs"], trace_complete=True)["status"] == "unavailable"


def test_explicit_later_start_preserves_the_prepared_motion_contract(controller):
    from cais_spade_llm.agents.central_controller.offline_recovery_composition import (
        analyze_grounded_recovery_composition,
    )

    case, _ = _case(controller)
    grounding = case["grounding_inputs"]
    grounding["horizon"] = [0, 2]
    for part in grounding["snapshot"]["parts"].values():
        if "stationary_until" in part:
            part["stationary_until"] = 2
    first = case["event_start_choices"][0]
    for rid in first["stationary"]:
        first["stationary"][rid] = [[1, 2]] if rid == "ur5e-3" else [[0, 2]]
    program = first["programs"][0]
    planned = prepare_ur5e_motion(controller, primitive="move_cartesian",
        params=program["primitive_steps"][0]["params"], start=_start(controller), binding={})
    _coverage(program, planned)
    later = deepcopy(first)
    later.update(id="later", starts={"UR5E_PREPARE_SEQ1": 1})
    later["stationary"]["ur5e-3"] = [[0, 1]]
    trace = later["programs"][0]["step_results"][0]
    trace.update(start_time=1, end_time=2)
    for point in trace["model_evidence"]["trajectory"]:
        point["time"] += 1
    case["event_start_choices"].append(later)
    result = analyze_grounded_recovery_composition(**case, budget=Budget(seconds=30))
    assert result["status"] == "allowed", result["reason"]
    assert any(row["time"] == 1 for row in result["decision_prefixes"])


@pytest.mark.parametrize("completion_time,expected", [(.25, "satisfied"), (.375, "violated"), (.5, "violated")])
def test_supported_ur5e_observations_preserve_strict_precedence(controller, completion_time, expected):
    case, _ = _case(controller, completed=False, completion_time=completion_time)
    program = case["grounding_inputs"]["programs"][0]
    planned = prepare_ur5e_motion(controller, primitive="move_cartesian",
        params=program["primitive_steps"][0]["params"], start=_start(controller), binding={})
    _coverage(program, planned)
    result = _grounded_check(case["grounding_inputs"], trace_complete=True)
    assert result["status"] == expected, result


def test_ur5e_mutex_keeps_separate_ap_labels_and_original_provenance(controller, tmp_path):
    bridge, runtime, request = _harness(controller)
    provider = bridge.cca.recovery_safety_preparation_provider
    # Change both the owner observation and the supplied initial state, so the
    # negative variant is an actual modeled conflict, not a contradictory snapshot.
    owner = next(row for row in runtime.resource_agents if row.agent_name == "ur5e-4")
    capture = owner.capture_recovery_safety_state
    def occupied(**kwargs):
        row = capture(**kwargs)
        row["current_pose"] = [1.1, 0, 1, 0, 0, 0, 1]
        row["observation_state"]["current_pose"] = row["current_pose"]
        return row
    owner.capture_recovery_safety_state = occupied
    def evidence(**kwargs):
        row = provider(**kwargs)
        row["composition_inputs"]["grounding_inputs"]["snapshot"]["resources"]["ur5e-4"]["current_pose"] = [1.1, 0, 1, 0, 0, 0, 1]
        return row
    bridge.cca.recovery_safety_preparation_provider = evidence
    result = prepare_and_check(bridge, request, output_root=tmp_path, max_age=30, budget=Budget(seconds=30))
    assert result["status"] == "held", result["unresolved"]
    checks = [check for edge in result["analysis"]["graph"]["edges"] for check in edge.get("rule_checks", [])
              if check["rule_id"].startswith("SAFE_shared_area_mutex:")]
    assert any(row["ap_values"] == {"ap001": True, "ap002": True} for row in checks)
    assert any(row["ap_values"] != {"ap001": True, "ap002": True} for row in checks)
    assert any(step["source"]["outline_id"] == "UR5E_PREPARE_SEQ1"
               for edge in result["analysis"]["counterexample"] for step in edge.get("active_steps", []))


@pytest.mark.parametrize("change", [None, "joint_stamp", "joint_receipt", "gazebo_stamp", "launch", "tool_pose"])
def test_controller_capture_reads_actual_stamps_and_rejects_inconsistent_sources(controller, change):
    from geometry_msgs.msg import Pose

    controller._Pose = Pose
    controller._joint_lock = threading.Lock()
    controller._joint_positions = dict.fromkeys(controller.arm_joint_names, 0.)
    controller._joint_received_times = dict.fromkeys(controller.arm_joint_names, time.monotonic())
    controller._joint_sim_stamps = dict.fromkeys(controller.arm_joint_names, 1.)
    controller.gripper_joint = "ur5e_3_gripper_joint"
    if change == "joint_stamp":
        controller._joint_sim_stamps[controller.arm_joint_names[0]] = -5.
    elif change == "joint_receipt":
        controller._joint_received_times[controller.arm_joint_names[0]] = 0.
    pose = controller._make_pose(1.1, 0., 1., controller._make_orientation(0., 0., 0., 1.))
    controller._tf_buffer = SimpleNamespace(lookup_transform=lambda *args: SimpleNamespace(
        header=SimpleNamespace(stamp=SimpleNamespace(sec=1, nanosec=0)),
        transform=SimpleNamespace(translation=pose.position, rotation=pose.orientation)))
    controller._rclpy = SimpleNamespace(time=SimpleNamespace(Time=lambda: None))
    controller._tf2_ros = SimpleNamespace(LookupException=LookupError, ConnectivityException=ConnectionError,
                                         ExtrapolationException=RuntimeError)
    controller._last_pose_observation = {"simulation_stamp": -4. if change == "gazebo_stamp" else 1.,
                                         "source": "gazebo_link_state"}
    observed = deepcopy(pose)
    if change == "tool_pose":
        observed.position.x += .5
    controller._observed_simulation_link_poses = Mock(return_value={controller.ee_link: observed})
    client = SimpleNamespace(wait_for_service=lambda **kwargs: True,
        call_async=lambda query: SimpleNamespace(values=[SimpleNamespace(
            string_value="recovery_framework" if change == "launch" else "observed-launch")]))
    controller._node.create_client = Mock(return_value=client)
    controller._node.destroy_client = Mock()
    controller._cb_group = None
    if change:
        with pytest.raises(ValueError):
            capture_ur5e_state(controller)
    else:
        captured = capture_ur5e_state(controller)
        assert captured["joint_names"] == controller.arm_joint_names
        assert captured["launch_id"] == "observed-launch"
        assert captured["current_pose"] == [1.1, 0., 1., 0., 0., 0., 1.]
        assert captured["attachment"]["observed_attachment_complete"] is False
        controller._observed_simulation_link_poses.assert_called_once()
    controller._send_simulation_joint_trajectory.assert_not_called()


def test_recovery_ui_requires_explicit_prepare_and_check_click(monkeypatch, tmp_path):
    from nicegui import context, core, ui
    from nicegui.client import Client
    from test_recovery_setup import _capture_buttons

    from cais_spade_llm.ui.components import recovery_safety_preparation as component

    buttons = _capture_buttons(monkeypatch)
    saved = tmp_path / "result.json"
    saved.write_text("{}")
    check = Mock(return_value={"status": "NEEDS_CONTEXT", "prepared_programs": [],
                               "unresolved": [{"reason": "No registered Gazebo owners"}], "artifact_path": str(saved)})
    monkeypatch.setattr(component, "prepare_and_check", check)
    client, bridge = Client(context.client.page), SimpleNamespace()
    async def scenario():
        monkeypatch.setattr(core, "loop", asyncio.get_running_loop())
        with client:
            component.render_safety_preparation(bridge)
            check.assert_not_called()
            editor = next(element for element in client.elements.values() if isinstance(element, ui.textarea))
            editor.value = json.dumps({"recovery_id": "inspected", "programs": []})
            await buttons["Prepare and check"]()
            check.assert_called_once_with(bridge, {"recovery_id": "inspected", "programs": []})
            assert any(getattr(element, "text", "") == "Safety check: NEEDS_CONTEXT" for element in client.elements.values())
    asyncio.run(scenario())


def test_archived_companion_cannot_claim_current_schema_acceptance():
    root = Path(__file__).resolve().parents[1]
    script = runpy.run_path(str(root / "scripts/check_part_slippage_gazebo.py"))
    with pytest.raises(ValueError, match="Predefined safety definitions changed"):
        script["load_companion"]("mutex")
