"""Recording ownership, success gates, and complete H.264 video validation."""

from __future__ import annotations

import asyncio
import json
import subprocess
from pathlib import Path
from unittest.mock import Mock

import pytest

from cais_spade_llm.recovery_framework import failure_videos
from cais_spade_llm.recovery_framework import gazebo_recording as recording


def test_recording_requires_real_concurrent_pickups_and_lowering():
    from copy import deepcopy

    configuration = {"checkpoint": "during_place_lowering", "placement_progress": .5,
                     "resource_id": "ur5e-4", "part_name": "gear_small",
                     "additional_condition": {"resource_id": "ur5e-3", "part_name": "KET4_Square_4mm"}}
    motion = {
        "run_id": "run", "task_id": "placement", "resource_id": "ur5e-4", "part_name": "gear_small",
        "checkpoint": "during_place_lowering", "source": "gazebo_placement_motion",
        "function_name": "place_approach", "step_id": "descend",
        "started_pose": {"x": .03, "y": .15, "z": 1.5},
        "target_pose": {"x": .03, "y": .15, "z": 1.0},
        "observed_pose": {"x": .03, "y": .15, "z": 1.25},
        "progress": .5, "goal_active": True, "goal_cancelled": True, "motion_stopped": True,
    }
    evidence = {"placement_motion": motion, "pending_tasks": [{
        "task_id": "placement", "resource_id": "ur5e-4", "event_name": "place_approach",
        "parameters": {"part_name": "gear_small"}}]}
    sample = {"run_id": "run", "slippage_initialization": {"run_id": "run", "status": "completed"},
              "physical_motions": [
                  {"run_id": "run", "resource_id": rid, "part_name": part, "task_id": rid,
                   "function_name": "pick_approach", "terminal_status": 4,
                   "started_at_unix": start, "ended_at_unix": end}
                  for rid, part, start, end in [
                      ("ur5e-3", "KET4_Square_4mm", 10., 20.), ("ur5e-4", "gear_small", 12., 18.)]]}
    result = failure_videos._validate_placement_slippage(sample, configuration, evidence)
    assert result["pickup_overlap"][0]["start"] == 12.
    assert result["pickup_overlap"][0]["end"] == 18.
    sequential = deepcopy(sample)
    sequential["physical_motions"][1].update(started_at_unix=21., ended_at_unix=25.)
    with pytest.raises(ValueError, match="did not overlap"):
        failure_videos._validate_placement_slippage(sequential, configuration, evidence)
    for field in ("goal_active", "goal_cancelled", "motion_stopped"):
        wrong = deepcopy(evidence)
        wrong["placement_motion"][field] = False
        with pytest.raises(ValueError, match="confirmed interruption"):
            failure_videos._validate_placement_slippage(sample, configuration, wrong)
    wrong = deepcopy(evidence)
    wrong["placement_motion"]["progress"] = .9
    with pytest.raises(ValueError, match="observed downward"):
        failure_videos._validate_placement_slippage(sample, configuration, wrong)


@pytest.mark.parametrize("number", [1, 2, 3])
def test_close_failure_camera_faces_the_existing_sign_and_preserves_setup(number):
    import math
    from copy import deepcopy

    from cais_spade_llm.recovery_framework.fault_visual import marker_geometry

    scene = json.loads((failure_videos.ROOT / "cais_spade_llm/initialization/recovery_framework_gazebo.json").read_text())
    original = deepcopy(scene)
    profile = failure_videos._failure_camera_profile(number, scene)
    scene["_failure_marker"] = {"scenario": failure_videos.SCENARIOS[number - 1],
                                "resource_id": ("Conveyor", "ur5e-1", "M1")[number - 1]}
    marker = marker_geometry(scene)
    pose = marker["pose"]
    label = [pose[0], pose[1], pose[2] + marker["height"] + marker.get("label_height", .55)]
    ray = [profile["position"][axis] - label[axis] for axis in range(3)]
    face_normal = [-math.sin(pose[5]), math.cos(pose[5]), 0.]
    cosine = abs(sum(ray[axis] * face_normal[axis] for axis in range(3))) / math.sqrt(sum(v * v for v in ray))
    assert cosine >= math.cos(math.radians(65))
    assert {"M1", "ur5e-1"} <= set(profile["subjects"])
    assert ("KMR" in profile["subjects"]) == (number == 3)
    assert profile["position"][2] > profile["look_at"][2]
    qx, qy, qz, qw = profile["orientation_xyzw"]
    forward = [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy + qw * qz), 2 * (qx * qz - qw * qy)]
    expected = [profile["look_at"][axis] - profile["position"][axis] for axis in range(3)]
    length = math.sqrt(sum(v * v for v in expected))
    assert forward == pytest.approx([v / length for v in expected])
    scene.pop("_failure_marker")
    assert scene == original


@pytest.mark.parametrize("number", [1, 2, 3])
def test_close_camera_publishes_its_profile_evidence(number, tmp_path, monkeypatch):
    scene = json.loads((failure_videos.ROOT / "cais_spade_llm/initialization/recovery_framework_gazebo.json").read_text())
    publish = Mock(return_value=subprocess.CompletedProcess([], 0, stdout="", stderr=""))
    monkeypatch.setattr(failure_videos.subprocess, "run", publish)
    failure_videos._set_failure_camera(number, scene, tmp_path)
    evidence = json.loads((tmp_path / "camera.json").read_text())
    assert evidence["scenario"] == number
    assert publish.call_args.args[0][:4] == ["gz", "topic", "-p", "/gazebo/default/user_camera/joy_pose"]
    assert publish.call_args.kwargs["check"] is True
    assert evidence["requested_gui_message"] == publish.call_args.args[0][-1]


@pytest.mark.parametrize("number", [1, 2, 3])
def test_close_failure_cameras_keep_the_accepted_slippage_rotation(number, tmp_path, monkeypatch):
    import re

    publish = Mock(return_value=subprocess.CompletedProcess([], 0, stdout="", stderr=""))
    monkeypatch.setattr(failure_videos.subprocess, "run", publish)
    failure_videos._set_assembly_camera(tmp_path)
    message = json.loads((tmp_path / "camera.json").read_text())["requested_gui_message"]
    reference = [float(value) for value in re.findall(r"[xyzw]: ([^ }]+)", message.split("orientation")[1])]
    scene = json.loads((failure_videos.ROOT / "cais_spade_llm/initialization/recovery_framework_gazebo.json").read_text())
    profile = failure_videos._failure_camera_profile(number, scene)
    assert profile["orientation_xyzw"] == pytest.approx(reference)
    assert profile["angle_reference"] == "Part slippage"
    if number == 3:
        assert "Storage" in profile["subjects"]
        assert profile["look_at"][0] < failure_videos._failure_camera_profile(2, scene)["look_at"][0]


@pytest.mark.parametrize("number", [4, 5])
def test_close_camera_selection_preserves_accepted_assembly_view(number, tmp_path, monkeypatch):
    previous_camera = Mock()
    monkeypatch.setattr(failure_videos, "_set_assembly_camera", previous_camera)
    failure_videos._set_failure_camera(number, {}, tmp_path)
    previous_camera.assert_called_once_with(tmp_path)
    assert not (tmp_path / "camera.json").exists()


def test_caption_timing_uses_captured_frames_despite_wall_clock_drift():
    frames = [dict(frame=0, observed_at_unix=100.),
              dict(frame=150, observed_at_unix=115.),
              dict(frame=300, observed_at_unix=130.)]
    assert failure_videos._caption_elapsed(115.02, frames, 15.) == 10.
    with pytest.raises(ValueError, match='no nearby recorded observation'):
        failure_videos._caption_elapsed(140., frames, 15.)


def test_navigation_readiness_timeout_does_not_accept_unobserved_controllers(tmp_path, monkeypatch):
    probe = Mock(side_effect=[subprocess.TimeoutExpired('readiness', 30),
                              subprocess.CompletedProcess([], 0, stdout=json.dumps({'ready': True}))])
    monkeypatch.setattr(failure_videos.subprocess, 'run', probe)
    asyncio.run(failure_videos._wait_for_navigation(tmp_path))
    observations = json.loads((tmp_path / 'navigation_readiness.json').read_text())['observations']
    assert [row['ready'] for row in observations] == [False, True]
    assert observations[0]['error'] == 'Nav2 readiness observation timed out'
    assert probe.call_count == 2


def test_recording_start_failure_discards_only_owned_video(tmp_path, monkeypatch):
    attempt = recording.RecordingAttempt(tmp_path)
    video = attempt.directory / 'capture.partial.mp4'
    video.write_bytes(b'incomplete')
    unrelated = tmp_path / 'previous-success.mp4'
    unrelated.write_bytes(b'preserve')
    monkeypatch.setattr(attempt, '_spawn', lambda operation: Mock(poll=lambda: 1))
    with pytest.raises(RuntimeError, match='did not become ready'):
        attempt.start()
    assert not video.exists() and unrelated.read_bytes() == b'preserve'


@pytest.mark.parametrize('change', ['run_id', 'injection', 'marker', 'cca'])
def test_failure_video_rejects_unconfirmed_or_unrelated_failure(change):
    sample = {
        'run_id': 'observed-run', 'diagnostic_cca_bypass': False,
        'unavailable_resources': ['Conveyor'],
        'fault': {'status': 'triggered', 'run_id': 'observed-run',
                  'scenario': 'Conveyor breakdown', 'visual': {'status': 'completed'},
                  'evidence': {'run_id': 'observed-run', 'injection_status': 'completed',
                               'checkpoint': 'after_M1_pick_before_release', 'source': 'M1',
                               'part_name': 'KET4_Square_4mm',
                               'resource_values_before': {'ur5e-1': {'held_part': 'KET4_Square_4mm'}},
                               'part_tracker_before': {'KET4_Square_4mm': {
                                   'location': 'ur5e-1', 'processCompleted': [{'process': 'trim', 'result': 'square'}]}}}},
    }
    config = {'scenario': 'Conveyor breakdown', 'resource_id': 'Conveyor',
              'checkpoint': 'after_M1_pick_before_release'}
    assert failure_videos.validate_failure(sample, config)['validated']
    if change == 'run_id':
        sample['fault']['evidence']['run_id'] = 'previous-run'
    elif change == 'injection':
        sample['fault']['evidence']['injection_status'] = 'failed'
    elif change == 'marker':
        sample['fault']['visual']['status'] = 'failed'
    else:
        sample['diagnostic_cca_bypass'] = True
    with pytest.raises(ValueError):
        failure_videos.validate_failure(sample, config)


@pytest.mark.parametrize('fraction', [None, .4, .6, float('nan')])
def test_machining_video_requires_observed_halfway_interruption(fraction):
    config = {'scenario': 'Machining breakdown during part processing', 'resource_id': 'M1',
              'checkpoint': 'during_processing_halfway'}
    evidence = {'run_id': 'observed-run', 'injection_status': 'completed',
                'checkpoint': config['checkpoint'], 'process_completed': False,
                'source': 'gazebo_workholding_observation', 'processing_fraction': .5}
    sample = {'run_id': 'observed-run', 'diagnostic_cca_bypass': False,
              'unavailable_resources': ['M1'],
              'fault': {'status': 'triggered', 'run_id': 'observed-run',
                        'scenario': config['scenario'], 'visual': {'status': 'completed'},
                        'evidence': evidence}}
    assert failure_videos.validate_failure(sample, config)['validated']
    evidence['processing_fraction'] = fraction
    with pytest.raises(ValueError, match='halfway'):
        failure_videos.validate_failure(sample, config)


@pytest.mark.parametrize('direction', ['corrected', 'custom_reverse'])
@pytest.mark.parametrize('change', ['pickup', 'other_pickup', 'region', 'assumed_release',
                                    'retained_slipped_part', 'lost_other_part'])
def test_slippage_video_requires_both_pickups_and_observed_region(change, direction):
    slipping, other = ('ur5e-4', 'ur5e-3') if direction == 'corrected' else ('ur5e-3', 'ur5e-4')
    part, retained = ('gear_small', 'KET4_Square_4mm') if direction == 'corrected' else ('KET4_Square_4mm', 'gear_small')
    y = .2 if direction == 'corrected' else -.2
    config = {'scenario': 'Part slippage', 'resource_id': slipping, 'part_name': part,
              'checkpoint': 'after_both_pickups_before_place', 'drop_pose': {'x': 0., 'y': y, 'z': 1.04},
              'additional_condition': {'resource_id': other, 'part_name': retained}}
    evidence = {'run_id': 'observed-run', 'injection_status': 'completed', 'checkpoint': config['checkpoint'],
                'detach': {'success': True, 'release_mode': 'detached'},
                'observed_drop_pose': {'x': 0., 'y': y, 'z': 1.025},
                'resource_values_before': {slipping: {'held_part': part}, other: {'held_part': retained}}}
    sample = {'run_id': 'observed-run', 'diagnostic_cca_bypass': False,
              'unavailable_resources': [slipping],
              'values': {slipping: {'held_part': None}, other: {'held_part': retained}},
              'fault': {'status': 'triggered', 'run_id': 'observed-run', 'scenario': config['scenario'],
                        'visual': {'status': 'completed'}, 'evidence': evidence}}
    assert failure_videos.validate_failure(sample, config)['validated']
    if change == 'pickup':
        evidence['resource_values_before'][slipping]['held_part'] = None
    elif change == 'other_pickup':
        evidence['resource_values_before'][other]['held_part'] = None
    elif change == 'region':
        evidence['observed_drop_pose']['y'] = -y
    elif change == 'assumed_release':
        evidence['detach']['release_mode'] = 'assumed_released_if_open'
    elif change == 'retained_slipped_part':
        sample['values'][slipping]['held_part'] = part
    else:
        sample['values'][other]['held_part'] = None
    with pytest.raises(ValueError):
        failure_videos.validate_failure(sample, config)


def test_slippage_recording_loads_the_shared_ui_preset(monkeypatch):
    from cais_spade_llm.ui import recovery_setup

    monkeypatch.setattr(failure_videos, 'load_setup', recovery_setup.default_setup)
    setup = failure_videos.scenario_setup(4)
    models = recovery_setup.validate_setup(setup)['models']
    assert setup['failure_scenario'] == recovery_setup.slippage_preset(models)
    assert setup['failure_scenario']['resource_id'] == 'ur5e-4'
    assert setup['failure_scenario']['part_name'] == 'gear_small'
    assert setup['failure_scenario']['additional_condition']['part_name'] == 'KET4_Square_4mm'
    assert setup['selected_product_order_file'].endswith('assembly_board-v1-two-parts.json')
    assert setup['diagnostic_cca_bypass'] is False
    assert 'ur5e-4 gear slips during placement into ur5e-3' in failure_videos.TITLES[3]


@pytest.mark.parametrize('failed', [False, True])
def test_recording_restores_inputs_and_preserves_historical_video(tmp_path, monkeypatch, failed):
    monkeypatch.setattr(failure_videos, 'ROOT', tmp_path)
    setup = tmp_path / 'cais_spade_llm/initialization/recovery_framework_setup.json'
    monkeypatch.setattr(failure_videos, 'SETUP_PATH', setup)
    protected = [setup, tmp_path / 'cais_spade_llm/initialization/tools.json',
                 tmp_path / 'cais_spade_llm/safety/cca_safety_logic.json']
    for path in protected:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(('original ' + path.name).encode())
    originals = {path: path.read_bytes() for path in protected}
    generated = tmp_path / 'cais_spade_llm/safety/workspace_mutex_dfa.dot'
    output = tmp_path / 'videos'
    output.mkdir()
    historical = output / 'historical-slippage-20x.mp4'
    historical.write_bytes(b'preserved historical recording')

    async def record(number, directory, target, timeout):
        directory.mkdir(parents=True)
        for path in protected:
            path.write_bytes(b'temporary recording inputs')
        generated.write_text('temporary rule')
        if failed:
            raise ValueError('unconfirmed drop')
        return {'video_file': str(target), 'validated': True}

    monkeypatch.setattr(failure_videos, 'record_one', record)
    result = asyncio.run(failure_videos.record_videos([4], output, tmp_path / 'evidence', 10))
    assert bool(result['blocked']) is failed
    assert bool(result['videos']) is not failed
    assert {path: path.read_bytes() for path in protected} == originals
    assert not generated.exists()
    assert historical.read_bytes() == b'preserved historical recording'


@pytest.mark.parametrize('change', ['none', 'upright', 'fallen_gear', 'partial', 'missing_gear', 'moved', 'lost_custody', 'joints_changed', 'interpolation_collision', 'missed_grasp', 'wrong_start', 'missed_part', 'closed_gripper', 'settled_rotation', 'unstable_position', 'unstable_orientation', 'planning_rotation', 'scene_refresh_failed', 'scene_refresh_unacknowledged', 'scene_refresh_wrong_model'])
def test_slippage_pickup_check_requires_complete_read_only_collision_plan(tmp_path, monkeypatch, change):
    from copy import deepcopy
    from types import SimpleNamespace

    from geometry_msgs.msg import Point, Pose, PoseStamped, Quaternion
    from moveit_msgs.msg import AttachedCollisionObject, CollisionObject, PlanningScene
    from moveit_msgs.srv import GetCartesianPath, GetStateValidity
    from trajectory_msgs.msg import JointTrajectoryPoint

    from cais_spade_llm.recovery_framework import (
        failure_effects,
        part_collision,
        workflow_execution,
    )

    names = ['joint-' + str(index) for index in range(6)]
    scene = PlanningScene()
    scene.robot_state.joint_state.name = [*names, 'gripper', 'mimic_joint']
    scene.robot_state.joint_state.position = [0.] * 7 + [.577]
    scene.robot_state.attached_collision_objects = [AttachedCollisionObject(
        object=CollisionObject(id='square_model/collision', operation=CollisionObject.ADD))]
    scene.world.collision_objects = [] if change == 'missing_gear' else [CollisionObject(id='gear_model/collision')]
    response = GetCartesianPath.Response()
    response.error_code.val = 1
    response.fraction = .8 if change == 'partial' else 1.
    response.solution.joint_trajectory.joint_names = names
    response.solution.joint_trajectory.points = [
        JointTrajectoryPoint(positions=[.2 if change == 'wrong_start' else 0.] * 6,
                             velocities=[0.] * 6, accelerations=[0.] * 6),
        JointTrajectoryPoint(positions=[.1] * 6, velocities=[0.] * 6, accelerations=[0.] * 6)]
    response.solution.joint_trajectory.points[-1].time_from_start.sec = 1
    fk_client = Mock(call_async=Mock(return_value=SimpleNamespace(
        error_code=SimpleNamespace(val=1), pose_stamped=[PoseStamped(pose=Pose(
            position=Point(x=.1 if change == 'missed_grasp' else 0., y=.2, z=1.26),
            orientation=Quaternion(w=1.)))])))

    def plan(request):
        response.start_state = deepcopy(scene.robot_state)
        response.start_state.joint_state.position[:6] = response.solution.joint_trajectory.points[-1].positions
        response.start_state.joint_state.position[-2] = .8
        response.start_state.joint_state.position[-1] = .123
        response.start_state.attached_collision_objects = []
        if change == 'closed_gripper':
            response.start_state.joint_state.position[-2] = 0.
        return response

    controller = SimpleNamespace(
        _attached_model='square_model',
        _sync_part_collision=Mock(return_value=change != 'scene_refresh_failed'),
        _last_command_evidence={'payload_collision': {
            'model_name': 'square_model' if change == 'scene_refresh_wrong_model' else 'gear_model',
            'attached_link': None, 'collision_objects': [{'id': 'gear_model/collision'}],
            'collision_scene_acknowledged': change != 'scene_refresh_unacknowledged'}},
        _payload_scene_clients={'get': Mock(
            call_async=Mock(return_value=SimpleNamespace(scene=scene)))},
        _cart_client=Mock(call_async=Mock(side_effect=plan)), _GetCartesianPath=GetCartesianPath,
        _state_validity_client=Mock(call_async=Mock(return_value=GetStateValidity.Response(
            valid=change != 'interpolation_collision'))),
        _node=Mock(create_client=Mock(return_value=fk_client)), _cb_group=None,
        cartesian_position_tolerance_m=.005, cartesian_orientation_tolerance_rad=.02,
        _wait_future=lambda future, **kwargs: future,
        _get_arm_joint_positions=Mock(side_effect=[
            ([0.] * 6, []), ([.2] * 6 if change == 'joints_changed' else [0.] * 6, [])]),
        compute_pick_targets=Mock(return_value={
            'success': True, 'tx': 0., 'ty': .2, 'travel_z': 1.4, 'pick_z': 1.26,
            'pick_tcp_z': 1.07 if change == 'missed_part' else 1.03}),
        _get_ee_pose=lambda: Pose(position=Point(x=.5, y=.2, z=1.4), orientation=Quaternion(w=1.)),
        _make_pose=lambda x, y, z, q: Pose(position=Point(x=x, y=y, z=z), orientation=q),
        gripper_joint='gripper', gripper_open=.8, frame_id='world', group_name='ur3', ee_link='tool',
        arm_joint_names=names, controller_config={
            'cartesian_motion': {'linear_step_m': .01, 'max_joint_step_rad': .35}},
        detach_part=Mock(), open_gripper=Mock(), _send_simulation_joint_trajectory=Mock())
    actor = SimpleNamespace(valuation={'held_part': None if change == 'lost_custody' else 'KET4_Square_4mm'})
    runtime = SimpleNamespace(
        context=SimpleNamespace(run_id='read-only-check', resources={'ur5e-3': actor},
                                geometry={'gear_small': {'model_name': 'gear_model'},
                                          'KET4_Square_4mm': {'model_name': 'square_model'}}),
        resource_agents=[SimpleNamespace(agent_name='ur5e-3', execution_mode='simulation', _controller=controller)])
    bridge = SimpleNamespace(product_agents=[SimpleNamespace(environment_runtime=runtime)])
    configuration = {'resource_id': 'ur5e-4', 'part_name': 'gear_small',
                     'additional_condition': {'resource_id': 'ur5e-3', 'part_name': 'KET4_Square_4mm'}}
    if change in {'upright', 'fallen_gear'}:
        configuration['require_upright'] = True
    pose = {'x': 0., 'y': .2, 'z': 1.025, 'qx': 0., 'qy': 0., 'qz': 0., 'qw': 1.}
    observed = {**pose, 'y': .23} if change == 'moved' else dict(pose)
    if change == 'fallen_gear':
        observed.update(qx=2 ** -.5, qw=2 ** -.5)
    if change == 'settled_rotation':
        observed.update(qz=.19866933079506122, qw=.9800665778412416)
    observation_count = 0

    def observe(*args):
        nonlocal observation_count
        observation_count += 1
        sampled = dict(observed)
        if change == 'unstable_position' and observation_count > 1:
            sampled['x'] += .003
        if ((change == 'unstable_orientation' and observation_count > 1)
                or (change == 'planning_rotation' and observation_count > 9)):
            sampled.update(qz=.09983341664682815, qw=.9950041652780258)
        return sampled

    monkeypatch.setattr(failure_effects, '_observe_part', observe)
    monkeypatch.setattr(failure_videos.time, 'sleep', lambda seconds: None)
    monkeypatch.setattr(workflow_execution, '_pick_geometry', lambda *args: {'model_name': 'gear_model'})
    monkeypatch.setattr(part_collision, 'observed_part_boxes', lambda *args: [
        {'id': 'gear_model/collision', 'pose': [0., .2, 1.025, 0., 0., 0., 1.], 'size': [.02, .02, .02]}])

    if change in {'none', 'upright', 'settled_rotation'}:
        result = failure_videos._slippage_pickup_check(
            bridge, configuration, {'observed_drop_pose': pose}, tmp_path)
        assert result['validated'] and result['stable_support'] and result['planning_only']
        assert result['checked_states'] > 1
        for call in controller._state_validity_client.call_async.call_args_list:
            sampled = call.args[0].robot_state
            assert sampled.is_diff is False
            assert not sampled.attached_collision_objects
            mimic_index = list(sampled.joint_state.name).index("mimic_joint")
            assert sampled.joint_state.position[mimic_index] == .123
        assert len(result['support_observations']) == 9
        controller._sync_part_collision.assert_called_once_with('gear_model')
        assert result['settled_collision_scene']['payload_collision']['collision_scene_acknowledged']
        assert result['observed_pose_after_aftermath'] == observed
        assert result['endpoint_position_error_m'] == 0.
        request = controller._cart_client.call_async.call_args.args[0]
        assert request.avoid_collisions
        if change == 'upright':
            assert len(request.waypoints) == 4
            assert request.waypoints[-2].position.x == pytest.approx(-.15)
            assert request.waypoints[-2].position.z == request.waypoints[-1].position.z
            assert result['upright_angle_rad'] == 0.
        assert request.start_state.attached_collision_objects[0].object.operation == CollisionObject.REMOVE
        assert request.start_state.joint_state.position[-1] == controller.gripper_open
        assert 'mimic_joint' not in request.start_state.joint_state.name
        assert controller.compute_pick_targets.call_args.kwargs['use_global_min_pick_tcp_z'] is False
        assert result['grasp_geometry']['payload_at_gripper']
        assert request.waypoints[-1].position.y == pose['y']
        assert controller.compute_pick_targets.call_args.kwargs['target_pose']['z'] == pose['z']
    else:
        with pytest.raises(ValueError):
            failure_videos._slippage_pickup_check(
                bridge, configuration, {'observed_drop_pose': pose}, tmp_path)
        assert json.loads((tmp_path / 'slippage_pickup_check.json').read_text())['validated'] is False
    assert scene.robot_state.attached_collision_objects[0].object.operation == CollisionObject.ADD
    assert scene.robot_state.joint_state.position[-2] == 0.
    assert scene.robot_state.joint_state.position[-1] == .577
    assert controller._attached_model == 'square_model'
    controller.detach_part.assert_not_called()
    controller.open_gripper.assert_not_called()
    controller._send_simulation_joint_trajectory.assert_not_called()


def test_failure_video_cannot_publish_unvalidated_capture(tmp_path):
    attempt = recording.RecordingAttempt(tmp_path)
    source = attempt.directory / 'capture.partial.mp4'
    source.write_bytes(b'owned original')
    output = tmp_path / 'failure-20x.mp4'
    with pytest.raises(ValueError, match='Observed run validation'):
        failure_videos.export_20x(attempt, output, 'Conveyor breakdown',
                                 {'validated': False, 'run_id': 'blocked'}, [])
    assert source.read_bytes() == b'owned original'
    assert not output.exists()


def test_verified_20x_export_decodes_and_removes_only_its_original(tmp_path):
    script = '''
import json, sys
from pathlib import Path
import numpy as np
from cais_spade_llm.recovery_framework.gazebo_recording import RecordingAttempt, _H264Writer, _write_json
from cais_spade_llm.recovery_framework.failure_videos import export_20x
root=Path(sys.argv[1])
attempt=RecordingAttempt(root)
writer=_H264Writer(attempt.directory/'capture.partial.mp4', 15., (640,360))
for index in range(60):
    writer.write(np.full((360,640,3), 80+index, dtype=np.uint8))
writer.release()
_write_json(attempt.directory/'capture.json', {'status':'captured','frames':60,'fps':15.})
(attempt.directory/'frames.jsonl').write_text(json.dumps({'frame':59,'elapsed_sec':59/15})+'\\n')
previous=root/'previous.mp4'
previous.write_bytes(b'preserve')
output=root/'videos'/'Conveyor breakdown-20x.mp4'
result=export_20x(attempt,output,'Conveyor breakdown', {'validated':True,'run_id':'encoding-fixture'},[])
assert abs(result['duration_sec']-result['source_duration_sec']/20) <= 2/15
assert result['frames_decoded'] >= 2 and output.exists()
assert not (attempt.directory/'capture.partial.mp4').exists()
assert previous.read_bytes() == b'preserve'
assert list(output.parent.iterdir()) == [output]
'''
    subprocess.run(['/usr/bin/python3', '-c', script, str(tmp_path)],
                   cwd=Path(__file__).resolve().parents[1], check=True, timeout=30)


def test_combined_export_requires_every_trial_validation_before_reading_video(tmp_path):
    attempt = recording.RecordingAttempt(tmp_path)
    source = attempt.directory / 'capture.partial.mp4'
    source.write_bytes(b'preserve incomplete capture')
    trial = {'attempt': attempt, 'title': 'Incomplete trial', 'captions': [],
             'validation': {'validated': False, 'run_id': 'incomplete'}}
    with pytest.raises(ValueError, match='observed run validation'):
        failure_videos.export_combined_20x([trial], tmp_path / 'combined-20x.mp4', 'Part slippage')
    assert source.read_bytes() == b'preserve incomplete capture'
    assert not (tmp_path / 'combined-20x.mp4').exists()


def test_combined_20x_export_keeps_one_video_and_distinct_trial_evidence(tmp_path):
    script = '''
import json, sys
from pathlib import Path
import numpy as np
from cais_spade_llm.recovery_framework.gazebo_recording import RecordingAttempt, _H264Writer, _write_json
from cais_spade_llm.recovery_framework.failure_videos import export_combined_20x
root=Path(sys.argv[1])
trials=[]
for index, name in enumerate(('mutex', 'precedence', 'safe')):
    attempt=RecordingAttempt(root/name)
    writer=_H264Writer(attempt.directory/'capture.partial.mp4',15.,(640,360))
    for frame in range(60):
        writer.write(np.full((360,640,3),60+index*30+frame,dtype=np.uint8))
    writer.release()
    _write_json(attempt.directory/'capture.json',{'status':'captured','frames':60,'fps':15.})
    (attempt.directory/'frames.jsonl').write_text(json.dumps({'frame':59,'elapsed_sec':59/15})+'\\n')
    trials.append({'attempt':attempt,'title':name+' | synthetic encoding fixture',
                   'captions':[{'text':'Encoding test only','start':1.,'end':3.}],
                   'validation':{'validated':True,'run_id':'encoding-'+name}})
previous=root/'historical.mp4'
previous.write_bytes(b'preserve history')
output=root/'videos'/'part_slippage-20x.mp4'
result=export_combined_20x(trials,output,'Synthetic encoding test')
assert result['separately_staged_trials'] is True
assert [row['run_id'] for row in result['trials']] == ['encoding-mutex','encoding-precedence','encoding-safe']
assert abs(result['source_duration_sec']-12.) < .01
assert abs(result['duration_sec']-.6) <= 2/15
assert result['frames_decoded'] > 0
assert len(list(output.parent.glob('*.mp4'))) == 1
assert previous.read_bytes() == b'preserve history'
assert all(not (row['attempt'].directory/'capture.partial.mp4').exists() for row in trials)
assert Path(result['validation_file']).exists()
assert list(output.parent.iterdir()) == [output]
'''
    subprocess.run(['/usr/bin/python3', '-c', script, str(tmp_path)],
                   cwd=Path(__file__).resolve().parents[1], check=True, timeout=60)


def test_mutex_video_requires_cca_hold_and_subsequent_access():
    def sample(timestamp, first, second):
        return {'run_id': 'mutex-run', 'observed_at_unix': timestamp,
                'diagnostic_cca_bypass': False,
                'values': {'ur5e-3': {'resource_location': first},
                           'ur5e-4': {'resource_location': second}}}
    samples = [sample(10, 'home', 'assembly_board-v1'),
               sample(10.5, 'home', 'home'), sample(11, 'assembly_board-v1', 'home')]
    negotiations = [
        {'kind': 'CCA', 'timestamp': 10, 'task_ids': ['entry-3']},
        {'kind': 'candidate_held', 'task_id': 'entry-3',
         'decision': {'status': 'held', 'included_specifications': ['workspace_mutex'],
                      'counterexample': [{'action': 'entry'}],
                      'task_bindings': {'entry': {'resource_id': 'ur5e-3', 'event_name': 'place_approach',
                                                 'parameters': {'destination_location': 'assembly_board-v1'}}}}},
        {'kind': 'CCA', 'timestamp': 10.6, 'decision': {'decisions': {'allowed-entry-3': {'status': 'allowed'}}}},
        {'kind': 'task_sent', 'task_id': 'allowed-entry-3', 'timestamp': 10.7,
         'resource_id': 'ur5e-3', 'event_name': 'place_approach'},
    ]
    result = failure_videos.validate_mutex(samples, negotiations)
    assert result['waiting_robot'] == 'ur5e-3' and result['first_robot'] == 'ur5e-4'
    with pytest.raises(ValueError, match='No observed CCA mutex hold'):
        failure_videos.validate_mutex(samples, [])
    samples.append(sample(12, 'assembly_board-v1', 'assembly_board-v1'))
    with pytest.raises(ValueError, match='overlapping occupancy'):
        failure_videos.validate_mutex(samples, negotiations)


def test_mutex_recording_binds_entry_and_persistent_occupancy_to_exact_resources():
    from cais_spade_llm.agents.central_controller.online_safety_monitor import OnlineSafetyMonitor

    checker = OnlineSafetyMonitor({}, [failure_videos.mutex_rule()])
    assert checker._map_task_to_aps('ur5e-3@localhost', 'place_approach',
                                    {'destination_location': 'assembly_board-v1'}) == ['ap5']
    assert checker._map_task_to_aps('ur5e-3@localhost', 'place_approach',
                                    {'destination_location': 'Conveyor'}) == []
    assert checker._map_state_to_aps('ur5e-4@localhost', 'placed',
                                     {'resource_location': 'assembly_board-v1'}) == ['ap4']
    assert checker._map_state_to_aps('ur5e-4@localhost', 'home',
                                     {'resource_location': 'home'}) == []
    checker.resource_bindings = {'recovery-resource-3@localhost': 'ur5e-3',
                                 'recovery-resource-4@localhost': 'ur5e-4'}
    assert checker._map_task_to_aps('recovery-resource-3@localhost', 'place_approach',
                                    {'destination_location': 'assembly_board-v1'}) == ['ap5']
    assert checker._map_state_to_aps('recovery-resource-4@localhost', 'placed',
                                     {'resource_location': 'assembly_board-v1'}) == ['ap4']
    assert checker._map_task_to_aps('unregistered@localhost', 'place_approach',
                                    {'resource_id': 'ur5e-3', 'destination_location': 'assembly_board-v1'}) == []


def test_cancel_terminates_owned_recorder_and_preserves_logs(tmp_path):
    attempt = recording.RecordingAttempt(tmp_path)
    attempt.process = Mock(poll=lambda: None)
    for name in recording.VIDEO_NAMES:
        (attempt.directory / name).write_bytes(b'partial')
    evidence = attempt.directory / 'frames.jsonl'
    evidence.write_text('observed\n')
    attempt.cancel()
    attempt.process.terminate.assert_called_once()
    attempt.process.wait.assert_called_once_with(timeout=10)
    assert not any((attempt.directory / name).exists() for name in recording.VIDEO_NAMES)
    assert evidence.read_text() == 'observed\n'


def test_abandoned_cleanup_preserves_active_successful_and_unowned_files(tmp_path):
    active = recording.RecordingAttempt(tmp_path)
    failed = recording.RecordingAttempt(tmp_path)
    successful = recording.RecordingAttempt(tmp_path)
    for attempt in (active, failed, successful):
        (attempt.directory / 'capture.partial.mp4').write_bytes(b'video')
    owner = {**failed.owner, 'pid': 99999999}
    recording._write_json(failed.directory / 'owner.json', owner)
    recording._write_json(successful.directory / 'owner.json', {**successful.owner, 'pid': 99999999})
    (successful.directory / 'success.json').write_text('{}')
    unknown = tmp_path / 'attempt-unowned'
    unknown.mkdir()
    (unknown / 'assembly.mp4').write_bytes(b'unrelated')
    recording.cleanup_abandoned(tmp_path)
    assert not (failed.directory / 'capture.partial.mp4').exists()
    assert (active.directory / 'capture.partial.mp4').exists()
    assert (successful.directory / 'capture.partial.mp4').exists()
    assert (unknown / 'assembly.mp4').exists()


def test_failed_physical_validation_cannot_retain_video(tmp_path):
    attempt = recording.RecordingAttempt(tmp_path)
    (attempt.directory / 'capture.partial.mp4').write_bytes(b'video')
    with pytest.raises(ValueError, match='physical validation'):
        attempt.promote({'validated': False, 'run_id': 'failed'})
    assert not (attempt.directory / 'capture.partial.mp4').exists()


def test_video_failure_discards_recording_even_if_physical_validation_passed(tmp_path, monkeypatch):
    attempt = recording.RecordingAttempt(tmp_path)
    attempt.process = Mock(poll=lambda: 0, returncode=0)
    (attempt.directory / 'capture.partial.mp4').write_bytes(b'broken')
    monkeypatch.setattr(attempt, '_spawn', lambda operation: Mock(poll=lambda: 1, returncode=1))
    with pytest.raises(RuntimeError, match='Video validation failed'):
        attempt.promote({'validated': True, 'run_id': 'physically-complete'})
    assert not (attempt.directory / 'capture.partial.mp4').exists()
    assert not (attempt.directory / 'success.json').exists()


def test_success_publishes_both_videos_and_survives_cleanup(tmp_path, monkeypatch):
    attempt = recording.RecordingAttempt(tmp_path)
    attempt.process = Mock(poll=lambda: 0, returncode=0)
    for name in recording.VIDEO_NAMES[:2]:
        (attempt.directory / name).write_bytes(b'validated')
    recording._write_json(attempt.directory / 'video_validation.json', {'validated': True})
    monkeypatch.setattr(attempt, '_spawn', lambda operation: Mock(poll=lambda: 0, returncode=0))
    result = attempt.promote({'validated': True, 'run_id': 'eleven-complete'})
    attempt.cancel()
    assert result['preview_speed'] == 10
    assert (attempt.directory / 'assembly.mp4').exists()
    assert (attempt.directory / 'assembly-10x.mp4').exists()
    assert not (attempt.directory / 'capture.partial.mp4').exists()


@pytest.mark.parametrize('duration_case', ['legacy', 'encoder_finalize_delay', 'invalid_frame_duration'])
def test_system_encoder_decodes_entire_video_and_labeled_preview(tmp_path, duration_case):
    attempt = recording.RecordingAttempt(tmp_path)
    script = '''
import sys, json
from pathlib import Path
import cv2
import numpy as np
from cais_spade_llm.recovery_framework.gazebo_recording import _preview, _write_json, _H264Writer
root = Path(sys.argv[1])
writer = _H264Writer(root/'capture.partial.mp4', 15., (640, 360))
for i in range(30):
    writer.write(np.full((360, 640, 3), 100 + i, dtype=np.uint8))
writer.release()
case = sys.argv[2]
_write_json(root/'capture.json', {'status':'captured', 'frames':30,
                                'wall_duration_sec':2. if case == 'legacy' else 4.2})
if case != 'legacy':
    rows = [{'frame':i, 'elapsed_sec':i/15, 'observed_at_unix':100+i/15} for i in range(30)]
    if case == 'invalid_frame_duration':
        rows[-1]['elapsed_sec'] += 3.
    (root/'frames.jsonl').write_text(''.join(json.dumps(row) + chr(10) for row in rows))
try:
    _preview(root)
except ValueError as error:
    assert case == 'invalid_frame_duration' and 'duration disagrees' in str(error)
else:
    assert case != 'invalid_frame_duration'
'''
    subprocess.run(['/usr/bin/python3', '-c', script, str(attempt.directory), duration_case],
                   cwd=Path(__file__).resolve().parents[1], check=True, timeout=20)
    if duration_case == 'invalid_frame_duration':
        assert not (attempt.directory / 'video_validation.json').exists()
        return
    result = json.loads((attempt.directory / 'video_validation.json').read_text())
    assert result['preview_encoder'] == 'h264_nvenc'
    assert result['full_frames_decoded'] == 30
    assert result['preview_frames_decoded'] == 3
    assert result['duration_sec'] == pytest.approx(result['preview_duration_sec'] * 10)


def test_blank_rendering_cannot_start_production_recording(tmp_path):
    attempt = recording.RecordingAttempt(tmp_path)
    script = '''
import sys
from pathlib import Path
import numpy as np
from cais_spade_llm.recovery_framework import gazebo_recording as recording
class BlankWindow:
    def frame(self): return np.zeros((720, 1280, 3), dtype=np.uint8)
    def close(self): pass
recording.X11Capture = BlankWindow
try:
    recording._capture(Path(sys.argv[1]))
except RuntimeError as error:
    assert "blank" in str(error)
else:
    raise AssertionError("Blank Gazebo rendering was accepted")
'''
    subprocess.run(["/usr/bin/python3", "-c", script, str(attempt.directory)],
                   cwd=Path(__file__).resolve().parents[1], check=True, timeout=20)
    assert not (attempt.directory / "ready.json").exists()
    assert not (attempt.directory / "capture.partial.mp4").exists()
    assert json.loads((attempt.directory / "capture.json").read_text())["status"] == "failed"


def test_encoder_failure_is_reported_and_owned_process_is_reaped(tmp_path, monkeypatch):
    process = Mock()
    process.wait.return_value = 1
    monkeypatch.setattr(recording.subprocess, 'Popen', Mock(return_value=process))
    writer = recording._H264Writer(tmp_path / 'capture.partial.mp4', 15., (640, 360))
    with pytest.raises(RuntimeError, match='encoder failed'):
        writer.release()
    process.stdin.close.assert_called_once()
    process.wait.assert_called_once_with(timeout=15)
    assert writer.log.closed
    writer.release()


@pytest.mark.parametrize('delay_stage', ['startup', 'shutdown'])
def test_encoder_startup_delay_precedes_capture_clock_and_ready(tmp_path, delay_stage):
    attempt = recording.RecordingAttempt(tmp_path)
    script = r"""
import sys, time, json
from pathlib import Path
import numpy as np
from cais_spade_llm.recovery_framework import gazebo_recording as recording
root = Path(sys.argv[1])
class Window:
    count = 0
    def frame(self):
        self.count += 1
        if self.count == 6:
            (root/'finish').touch()
        return np.tile(np.arange(640, dtype=np.uint8), (360, 1))[..., None].repeat(3, axis=2)
    def close(self): pass
recording.X11Capture = Window
stage = sys.argv[2]
method = 'wait_ready' if stage == 'startup' else 'release'
original = getattr(recording._H264Writer, method)
def delayed(self):
    original(self)
    time.sleep(2.2)
    if stage == 'startup':
        assert not (root/'ready.json').exists()
setattr(recording._H264Writer, method, delayed)
recording._capture(root)
setattr(recording._H264Writer, method, original)
recording._preview(root)
metadata = json.loads((root/'capture.json').read_text())
assert metadata['max_capture_gap_sec'] < 2.
assert metadata['wall_duration_sec'] < 2.
assert metadata['ended_at_unix'] - metadata['started_at_unix'] < 2.
if stage == 'shutdown':
    assert metadata['encoder_finalize_time_sec'] >= 2.2
"""
    subprocess.run(['/usr/bin/python3', '-c', script, str(attempt.directory), delay_stage],
                   cwd=Path(__file__).resolve().parents[1], check=True, timeout=30)
    assert json.loads((attempt.directory / 'ready.json').read_text())['encoder'] == 'h264_nvenc'
    assert json.loads((attempt.directory / 'video_validation.json').read_text())['validated']


def _clock_observations(directory: Path) -> None:
    """Record a simulation that changes speed halfway through a four-second capture."""
    clocks = []
    frames = []
    for index in range(61):
        elapsed = index / 15
        simulation = min(elapsed, 2) * .2 + max(0, elapsed - 2) * .8
        clocks.append({'observed_at_unix': 100 + elapsed, 'simulation_time_sec': simulation})
        if index < 60:
            frames.append({'frame': index, 'observed_at_unix': 100 + elapsed,
                           'elapsed_sec': elapsed, 'repeated_frames': 0})
    for name, rows in (('clock_samples.jsonl', clocks), ('frames.jsonl', frames)):
        (directory / name).write_text(''.join(json.dumps(row) + '\n' for row in rows))


def test_clock_pacing_tracks_speed_changes_instead_of_average_speed(tmp_path):
    _clock_observations(tmp_path)
    mapping = recording._simulation_clock_frames(tmp_path, fps=15, frame_count=60)
    # Advancing 0.1333 simulation seconds initially takes ten input frames.
    assert mapping['source_frame_indices'][:4] == [0, 10, 20, 30]
    # After the clock speeds up, the same advance takes two or three frames.
    steps = [b - a for a, b in zip(mapping['source_frame_indices'][3:],
                                  mapping['source_frame_indices'][4:])]
    assert set(steps) <= {2, 3}
    assert mapping['video_duration_sec'] == pytest.approx(mapping['simulation_duration_sec'] / 2,
                                                         abs=1 / 15)
    assert mapping['max_frame_clock_error_sec'] <= .027


@pytest.mark.parametrize('failure', ['reset', 'wall_reset', 'missing_start', 'missing_end', 'stale', 'nan'])
def test_clock_pacing_rejects_unverifiable_clock(tmp_path, failure):
    _clock_observations(tmp_path)
    path = tmp_path / 'clock_samples.jsonl'
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if failure == 'reset':
        rows[30]['simulation_time_sec'] = 0
    elif failure == 'wall_reset':
        rows[30]['observed_at_unix'] = rows[29]['observed_at_unix']
    elif failure == 'missing_start':
        rows = rows[1:]
    elif failure == 'missing_end':
        rows = rows[:-2]
    elif failure == 'stale':
        rows = rows[:10] + rows[50:]
    else:
        rows[30]['simulation_time_sec'] = float('nan')
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    with pytest.raises(ValueError):
        recording._simulation_clock_frames(tmp_path, fps=15, frame_count=60)


def test_clock_pacing_selects_actual_captures_not_repeated_fill_frames(tmp_path):
    _clock_observations(tmp_path)
    path = tmp_path / 'frames.jsonl'
    frames = [json.loads(line) for line in path.read_text().splitlines()]
    frames.pop(10)
    path.write_text(''.join(json.dumps(row) + '\n' for row in frames))
    mapping = recording._simulation_clock_frames(tmp_path, fps=15, frame_count=60)
    assert 10 not in mapping['source_frame_indices']
    assert mapping['source_frame_indices'] == sorted(mapping['source_frame_indices'])


def test_clock_video_validation_is_required_before_any_promotion(tmp_path, monkeypatch):
    attempt = recording.RecordingAttempt(tmp_path, record_simulation_clock=True)
    attempt.process = Mock(poll=lambda: 0, returncode=0)
    for name in recording.VIDEO_NAMES:
        (attempt.directory / name).write_bytes(b'video')
    recording._write_json(attempt.directory / 'video_validation.json', {'validated': True})
    monkeypatch.setattr(attempt, '_spawn', lambda operation: Mock(poll=lambda: 0, returncode=0))
    with pytest.raises(RuntimeError, match='Simulation-clock video validation'):
        attempt.promote({'validated': True, 'run_id': 'eleven-complete'})
    assert not any((attempt.directory / name).exists() for name in recording.VIDEO_NAMES)
    assert not (attempt.directory / 'success.json').exists()


def test_clock_video_encodes_decodes_and_promotes_with_existing_copies(tmp_path):
    attempt = recording.RecordingAttempt(tmp_path, record_simulation_clock=True)
    _clock_observations(attempt.directory)
    script = '''
import sys
from pathlib import Path
import numpy as np
from cais_spade_llm.recovery_framework.gazebo_recording import _H264Writer, _write_json
root = Path(sys.argv[1])
writer = _H264Writer(root/'capture.partial.mp4', 15., (640, 360))
for i in range(60):
    writer.write(np.full((360, 640, 3), 100 + i, dtype=np.uint8))
writer.release()
_write_json(root/'capture.json', {'status':'captured', 'frames':60, 'wall_duration_sec':4.})
'''
    subprocess.run(['/usr/bin/python3', '-c', script, str(attempt.directory)],
                   cwd=Path(__file__).resolve().parents[1], check=True, timeout=20)
    attempt.process = Mock(poll=lambda: 0, returncode=0)
    success = attempt.promote({'validated': True, 'run_id': 'eleven-complete'})
    assert success['simulation_clock_factor'] == 2
    assert success['video']['simulation_clock']['validated']
    assert success['video']['simulation_clock']['frames_decoded'] == 15
    for name in ('assembly.mp4', 'assembly-10x.mp4', 'assembly-rtf2.mp4'):
        assert (attempt.directory / name).stat().st_size > 0
    attempt.cancel()
    assert (attempt.directory / 'assembly-rtf2.mp4').exists()
