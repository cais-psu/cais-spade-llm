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
