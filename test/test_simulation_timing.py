"""Global simulation timing, physical limits, and responsive cancellation."""

from __future__ import annotations

import math
from types import SimpleNamespace
import threading
import time
from unittest.mock import Mock

import pytest

from cais_spade_llm.resources.robot.simulation_timing import MotionDeadline, wait_for_simulation
from cais_spade_llm.resources.robot.gazebo_pick_place_controller import GazeboPickPlaceController
from cais_spade_llm.resources.robot.robot_task_runtime import _normalize_pick_targets
from cais_spade_llm.recovery_framework.workflow_gazebo import (
    _motion_distance,
    _motion_duration,
)


def test_simulation_named_pose_uses_resource_motion_plan_controller():
    execute = Mock(return_value=True)
    controller = SimpleNamespace(
        execution_mode="simulation",
        arm_trajectory_topic="/ur5e_1_joint_trajectory_controller/joint_trajectory",
        arm_joint_names=["joint_a", "joint_b"],
        _execute_simulation_motion_plan=execute,
        _nearest_simulation_joint_targets=lambda positions: positions,
    )

    assert GazeboPickPlaceController._move_joints_via_moveit(
        controller,
        [0.2, -0.4],
    )
    execute.assert_called_once_with(
        joint_positions=[0.2, -0.4],
        label="move_home",
    )


def test_configured_workflow_home_retains_observed_collision_free_pose():
    controller = SimpleNamespace(
        wait_for_services=Mock(return_value=True),
        named_positions={"home": [0.2, -0.4]},
        arm_joint_names=["joint_a", "joint_b"],
        _fresh_stable_joint_target=Mock(return_value=False),
        execution_mode="simulation",
        controller_config={"retain_observed_clear_pose_as_home": True},
        _current_simulation_state_is_collision_free=Mock(return_value=True),
        _last_command_evidence=None,
        _move_joints_via_moveit=Mock(return_value=False),
    )

    result = GazeboPickPlaceController.move_to_named_pose(controller, "home")

    assert result["success"] is True
    assert result["command_sent"] is False
    assert result["observed_clear_pose_as_home"] is True
    controller._move_joints_via_moveit.assert_not_called()


@pytest.mark.parametrize("condition", ["fresh", "stale", "moving", "nan", "nonfinite_target", "active_goal", "hardware"])
def test_joint_endpoint_observation_requires_finite_fresh_stable_feedback(condition):
    now = time.monotonic()
    controller = GazeboPickPlaceController.__new__(GazeboPickPlaceController)
    controller.execution_mode = "hardware" if condition == "hardware" else "simulation"
    controller._simulation_goal = object() if condition == "active_goal" else None
    controller.arm_joint_names = ["joint"]
    controller._joint_lock = threading.Lock()
    controller._joint_positions = {"joint": math.nan if condition == "nan" else 0.201}
    controller._joint_received_times = {"joint": now - (2 if condition == "stale" else 0.1)}
    controller._joint_stable_since = {"joint": now if condition == "moving" else now - 0.5}
    controller._last_joint_target_observation = {"old": True}
    assert controller._fresh_stable_joint_target(
        {"joint": math.inf if condition == "nonfinite_target" else 0.2}, tolerance=0.02,
    ) is (condition == "fresh")
    if condition == "fresh":
        assert controller._last_joint_target_observation["observed_positions"] == [0.201]
    else:
        assert controller._last_joint_target_observation is None


def test_simulation_cartesian_fallback_remains_collision_checked():
    target = object()
    execute = Mock(return_value=True)
    controller = SimpleNamespace(
        execution_mode="simulation",
        _cartesian_move=Mock(return_value=False),
        _get_ee_pose=Mock(
            return_value=SimpleNamespace(position=SimpleNamespace(x=0.0, y=0.0))
        ),
        _make_pose=Mock(return_value=target),
        _execute_simulation_motion_plan=execute,
        _log=lambda: Mock(),
    )

    assert GazeboPickPlaceController._move_xy_direct(
        controller,
        1.0,
        0.5,
        1.2,
        object(),
        "pick approach",
    )
    execute.assert_called_once_with(
        label="pick approach",
        target_pose=target,
        time_scale=1.0,
    )


def test_successful_controller_cartesian_endpoint_uses_observed_tool_pose():
    target = SimpleNamespace(
        position=SimpleNamespace(x=0.4, y=0.3, z=1.2),
        orientation=SimpleNamespace(x=0.0, y=1.0, z=0.0, w=0.0),
    )
    controller = SimpleNamespace(
        _last_simulation_controller_succeeded=True,
        cartesian_position_tolerance_m=0.005,
        cartesian_orientation_tolerance_rad=0.02,
        _get_ee_pose=Mock(
            return_value=SimpleNamespace(
                position=SimpleNamespace(x=0.403, y=0.3, z=1.2),
                orientation=SimpleNamespace(x=0.0, y=-1.0, z=0.0, w=0.0),
            )
        ),
    )

    evidence = GazeboPickPlaceController._observed_cartesian_endpoint_evidence(
        controller,
        target,
    )

    assert evidence is not None
    assert evidence["within_tolerance"] is True
    assert evidence["position_error_m"] == pytest.approx(0.003)
    assert evidence["orientation_error_rad"] == pytest.approx(0.0)


def test_failed_controller_result_cannot_use_observed_cartesian_endpoint():
    controller = SimpleNamespace(
        _last_simulation_controller_succeeded=False,
        _get_ee_pose=Mock(),
    )

    evidence = GazeboPickPlaceController._observed_cartesian_endpoint_evidence(
        controller,
        object(),
    )

    assert evidence is None
    controller._get_ee_pose.assert_not_called()


def test_configured_pick_orientation_controls_tool_pose_and_tcp_offset():
    controller = GazeboPickPlaceController.__new__(GazeboPickPlaceController)
    controller.execution_mode = "simulation"
    controller.pick_tcp_z_bias_min_m = 0.003
    controller.pick_tcp_z_bias_max_m = 0.02
    controller.min_pick_tcp_z_m = 0.0
    controller.pick_z_adjustments_m = {}
    controller.pick_tool0_z_adjustment_m = 0.0
    controller.approach_height_m = 0.2
    controller.gripper_open = 0.11
    controller.gripper_close = 0.02
    controller.init = lambda: True
    controller._get_ee_pose = lambda: SimpleNamespace(
        position=SimpleNamespace(x=0.0, y=0.0, z=1.5),
        orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
    )
    controller._get_ee_tcp_world_z_offset = lambda: 0.0
    controller._make_pose = lambda x, y, z, orientation: SimpleNamespace(
        position=SimpleNamespace(x=x, y=y, z=z), orientation=orientation
    )
    controller._log = lambda: Mock()

    result = controller.compute_pick_targets(
        part_name="gear_small",
        product_geometry={
            "model_name": "gear_small",
            "part_height_m": 0.02,
            "handling_robot_access": {
                "grasp_orientation_xyzw": [0.0, 1.0, 0.0, 0.0],
                "tcp_offset_z_m": -0.218,
                "approach_height_m": 0.12,
            },
        },
        detected_parts=[
            {
                "part_name": "gear_small",
                "model_name": "gear_small",
                "x": 0.44,
                "y": -0.58,
                "z": 1.11,
            }
        ],
        use_global_min_pick_tcp_z=False,
    )

    assert result["success"] is True
    assert result["tcp_offset_z"] == pytest.approx(-0.218)
    assert result["target_pose"] == pytest.approx(
        {
            "x": 0.44,
            "y": -0.58,
            "z": 1.333,
            "qx": 0.0,
            "qy": 1.0,
            "qz": 0.0,
            "qw": 0.0,
        },
        abs=1e-9,
    )
    normalized = _normalize_pick_targets(result)
    assert normalized["approach_pose"]["qy"] == 1.0
    assert normalized["target_pose"]["qy"] == 1.0

    from cais_spade_llm.recovery_framework import SCENE_PATH, read_json

    choices = read_json(SCENE_PATH)['3D Printing Station']['handling_robot_access'][
        'grasp_orientations_xyzw']
    for observed in (choices[1], choices[2]):
        controller._get_ee_pose = lambda observed=observed: SimpleNamespace(
            position=SimpleNamespace(x=.5, y=-.5, z=1.5),
            orientation=SimpleNamespace(x=observed[0], y=observed[1],
                                        z=observed[2], w=observed[3]),
        )
        chosen = controller.compute_pick_targets(
            part_name='gear_small',
            product_geometry={'model_name': 'gear_small', 'part_height_m': .02,
                'handling_robot_access': {'grasp_orientation_xyzw': choices[0],
                    'grasp_orientations_xyzw': choices,
                    'tcp_offset_z_m': -.218, 'approach_height_m': .12}},
            detected_parts=[{'part_name': 'gear_small', 'model_name': 'gear_small',
                             'x': .44, 'y': -.58, 'z': 1.11}],
            use_global_min_pick_tcp_z=False,
        )
        assert chosen['success'] is True
        assert [chosen['target_pose'][key] for key in ('qx', 'qy', 'qz', 'qw')] == pytest.approx(observed)

    from cais_spade_llm.recovery_framework.geometry import multiply, rotate

    access = read_json(SCENE_PATH)['3D Printing Station']['handling_robot_access']
    observed = choices[1]
    for part in ('gear_small', 'gear_medium', 'gear_large'):
        controller._get_ee_pose = lambda observed=observed: SimpleNamespace(
            position=SimpleNamespace(x=.5, y=-.5, z=1.5),
            orientation=SimpleNamespace(x=observed[0], y=observed[1],
                                        z=observed[2], w=observed[3]),
        )
        chosen = controller.compute_pick_targets(
            part_name=part,
            product_geometry={'model_name': part, 'part_height_m': .02,
                              'handling_robot_access': access},
            detected_parts=[{'part_name': part, 'model_name': part,
                             'x': .44, 'y': -.58, 'z': 1.11}],
            use_global_min_pick_tcp_z=False,
        )
        assert chosen['success'] is True
        pick_q = [chosen['target_pose'][key] for key in ('qx', 'qy', 'qz', 'qw')]
        assert pick_q in access['grasp_orientations_xyzw_by_part'][part]
        if part == 'gear_medium':
            assert pick_q == choices[0]
        place_q = multiply([0., 0., math.sqrt(.5), math.sqrt(.5)], pick_q)
        assert abs(rotate(place_q, [1., 0., 0.])[0]) < 1e-7
        observed = place_q


@pytest.mark.parametrize("cached_offset", [-.218, None, "invalid", float("nan")])
def test_gear_placement_rotates_tool_ninety_degrees_without_moving_slot(cached_offset):
    controller = SimpleNamespace(
        wait_for_services=lambda: True,
        execution_mode="simulation",
        robot_name="ur5e",
        controller_config={},
        pick_tcp_z_bias_min_m=0.0,
        pick_tcp_z_bias_max_m=0.1,
        place_surface_gap_m=0.0,
        insertion_depth_m=0.0,
        _get_ee_tcp_world_z_offset=Mock(return_value=-0.218),
    )
    grasp_pose = {
        "x": 0.44,
        "y": -0.58,
        "z": 1.3325,
        "qx": 0.0,
        "qy": 1.0,
        "qz": 0.0,
        "qw": 0.0,
    }

    result = GazeboPickPlaceController.compute_place_targets(
        controller,
        pick_ctx={
            "part_name": "gear_small",
            "model_name": "gear_small",
            "tx": 0.44,
            "ty": -0.58,
            "tz": 1.1095,
            "pick_tcp_z": 1.1145,
            "tcp_offset_z": cached_offset,
            "part_height": 0.02,
            "held_part_handoff": {"world_tool0_pose_at_grasp": grasp_pose},
        },
        part_name="gear_small",
        destination_location="assembly_board-v1",
        product_geometry={
            "part_name": "gear_small",
            "model_name": "gear_small",
            "part_height_m": 0.02,
            "board_center": {"x": 0.0, "y": 0.0, "z": 1.0239916},
            "slot_xy": [0.0294496, 0.1448242],
            "slot_floor_z_m": 1.02,
            "target_origin_pose": {"x": 0.0294496, "y": 0.1448242, "z": 1.03},
            "target_reference": {
                "target_point": "inserted_part_origin",
                "surface_role": "assembly_slot",
            },
            "place_tool_yaw_offset_rad": math.pi / 2,
        },
    )

    assert result["success"] is True
    assert result["slot_x"] == pytest.approx(0.0294496)
    assert result["slot_y"] == pytest.approx(0.1448242)
    for pose_name in ("approach_pose", "pre_insert_pose", "insert_pose"):
        pose = result[pose_name]
        assert pose["x"] == pytest.approx(0.0294496)
        assert pose["y"] == pytest.approx(0.1448242)
        assert pose["qx"] == pytest.approx(-math.sqrt(0.5))
        assert pose["qy"] == pytest.approx(math.sqrt(0.5))
        assert pose["qz"] == pytest.approx(0.0)
        assert pose["qw"] == pytest.approx(0.0)

    if cached_offset == -.218:
        controller._get_ee_tcp_world_z_offset.assert_not_called()
    else:
        controller._get_ee_tcp_world_z_offset.assert_called_once_with()


@pytest.mark.parametrize(
    ("observations", "expected"),
    [
        (
            [
                (0.0294496, 0.1448242, 1.03),
                (0.0299496, 0.1448242, 1.03),
                (0.0294496, 0.1453242, 1.03),
            ],
            True,
        ),
        ([(0.0314496, 0.1448242, 1.03)], False),
    ],
)
def test_slot_snap_requires_repeated_observed_gear_position(observations, expected):
    wait_process_time = Mock()
    get_position = Mock(side_effect=observations)
    controller = SimpleNamespace(
        snap_to_slot_position_tolerance_m=0.001,
        snap_to_slot_observation_samples=len(observations),
        snap_to_slot_observation_interval_sec=0.1,
        _wait_process_time=wait_process_time,
        _get_entity_world_position=get_position,
        _xyz_distance=lambda a, b: math.dist(a, b),
        _log=lambda: Mock(),
    )

    result = GazeboPickPlaceController._verify_snapped_entity_position(
        controller,
        "gear_small",
        (0.0294496, 0.1448242, 1.03),
    )

    assert result is expected
    assert get_position.call_count == len(observations)
    assert wait_process_time.call_count == len(observations)


def test_slot_snap_retries_missing_gazebo_observation_without_skipping_samples():
    expected = (.0294496, .1448242, 1.03)
    controller = SimpleNamespace(
        execution_mode='simulation', _shutdown_requested=False,
        snap_to_slot_position_tolerance_m=.001,
        snap_to_slot_observation_samples=3,
        snap_to_slot_observation_interval_sec=.1,
        _wait_process_time=Mock(),
        _get_entity_world_position=Mock(side_effect=[None, expected, expected, expected]),
        _xyz_distance=lambda a, b: math.dist(a, b),
        _log=lambda: Mock(), _node=Mock(), _get_state_client=Mock(),
        _GetEntityState=object(), service_get_entity_state='/get_entity_state',
        _cb_group=object(),
    )
    controller._node.create_client.return_value = Mock()
    assert GazeboPickPlaceController._verify_snapped_entity_position(
        controller, 'gear_small', expected)
    assert controller._get_entity_world_position.call_count == 4
    assert controller._node.destroy_client.call_count == 1
    controller._get_entity_world_position = Mock(return_value=None)
    assert not GazeboPickPlaceController._verify_snapped_entity_position(
        controller, 'gear_small', expected)
    assert controller._get_entity_world_position.call_count == 3
    controller.execution_mode = 'physical'
    controller._get_entity_world_position = Mock(return_value=None)
    assert not GazeboPickPlaceController._verify_snapped_entity_position(
        controller, 'gear_small', expected)
    assert controller._get_entity_world_position.call_count == 1


def test_configured_machine_access_controls_horizontal_pick_and_retreat():
    controller = GazeboPickPlaceController.__new__(GazeboPickPlaceController)
    controller.execution_mode = "simulation"
    controller.pick_tcp_z_bias_min_m = 0.003
    controller.pick_tcp_z_bias_max_m = 0.02
    controller.min_pick_tcp_z_m = 0.0
    controller.pick_z_adjustments_m = {}
    controller.pick_tool0_z_adjustment_m = 0.0
    controller.approach_height_m = 0.2
    controller.gripper_open = 0.11
    controller.gripper_close = 0.02
    controller.init = lambda: True
    controller._get_ee_pose = lambda: SimpleNamespace(
        position=SimpleNamespace(x=0.0, y=0.0, z=1.5),
        orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
    )
    controller._get_ee_tcp_world_z_offset = lambda: 0.0
    controller._make_pose = lambda x, y, z, orientation: SimpleNamespace(
        position=SimpleNamespace(x=x, y=y, z=z), orientation=orientation
    )
    controller._log = lambda: Mock()

    result = controller.compute_pick_targets(
        part_name="KET4_Square_4mm",
        product_geometry={
            "model_name": "KET4_Square_4mm",
            "part_height_m": 0.05,
            "handling_robot_access": {
                "grasp_orientation_xyzw": [0.0, 1.0, 0.0, 0.0],
                "tcp_offset_z_m": -0.218,
                "approach_pose": [-6.08, 1.64, 1.085, 0.0, 1.0, 0.0, 0.0],
                "target_pose": [-6.08, 1.82, 1.085, 0.0, 1.0, 0.0, 0.0],
                "pose_format": "xyz_xyzw",
            },
        },
        detected_parts=[
            {
                "part_name": "KET4_Square_4mm",
                "model_name": "KET4_Square_4mm",
                "x": -6.08,
                "y": 1.82,
                "z": 1.06,
            }
        ],
        use_global_min_pick_tcp_z=False,
    )

    assert result["success"] is True
    assert result["target_pose"] == pytest.approx(
        {
            "x": -6.08,
            "y": 1.82,
            "z": 1.303,
            "qx": 0.0,
            "qy": 1.0,
            "qz": 0.0,
            "qw": 0.0,
        }
    )
    assert result["approach_pose"]["y"] == pytest.approx(1.64)
    assert result["access_retreat_pose"] == result["approach_pose"]
    assert result["handling_robot_access_executed"] is True


@pytest.mark.parametrize('rate', [1, 2, 5, 10])
def test_process_delays_follow_simulation_progress(rate):
    clock = {'now': 0., 'wall': 0.}
    def poll(interval):
        clock['wall'] += interval
        clock['now'] += rate*interval
    wait_for_simulation(1., now=lambda: clock['now'], cancelled=lambda: False, poll=poll)
    assert clock['now'] >= 1.
    assert clock['wall'] == pytest.approx(1./rate, abs=.011)


def test_paused_reset_and_cancelled_clocks_fail_without_motion(monkeypatch):
    from cais_spade_llm.resources.robot import simulation_timing

    ticks = iter(range(100))
    monkeypatch.setattr(simulation_timing.time, 'monotonic', lambda values=ticks: next(values))
    with pytest.raises(TimeoutError, match='paused'):
        wait_for_simulation(1., now=lambda: 2., cancelled=lambda: False, poll=lambda _: None, stall_timeout=.5)
    ticks = iter([5., 4.])
    with pytest.raises(RuntimeError, match='reset'):
        wait_for_simulation(1., now=lambda: next(ticks), cancelled=lambda: False, poll=lambda _: None)
    with pytest.raises(InterruptedError):
        wait_for_simulation(1., now=lambda: 2., cancelled=lambda: True, poll=lambda _: None)


def test_legacy_speed_settings_do_not_compound_global_clock(monkeypatch):
    from cais_spade_llm.resources.robot.gazebo_pick_place_controller import _gazebo_timing_scale_from_env

    monkeypatch.setenv('CAIS_GAZEBO_WAIT_SCALE', '.1')
    controller = SimpleNamespace(execution_mode='simulation', trajectory_time_scale=.45, gripper_move_time_sec=1.)
    GazeboPickPlaceController._apply_gazebo_fast_timing_profile(controller, .1)
    assert controller.trajectory_time_scale == 1.
    assert controller.gripper_move_time_sec == 1.
    assert _gazebo_timing_scale_from_env('simulation') == 1.


def test_explicit_trajectory_speed_is_bounded_by_model_limits():
    limits = {'joint': {'lower': -1., 'upper': 1., 'velocity': .5, 'acceleration': 1.}}
    controller = SimpleNamespace(execution_mode='simulation', _simulation_joint_limits=lambda _: limits)
    controller._validate_simulation_trajectory = lambda trajectory: GazeboPickPlaceController._validate_simulation_trajectory(controller, trajectory)
    def point(position, sec):
        return SimpleNamespace(positions=[position], velocities=[.5], accelerations=[1.],
                               time_from_start=SimpleNamespace(sec=sec, nanosec=0))
    trajectory = SimpleNamespace(joint_names=['joint'], points=[point(0., 0), point(.5, 1)])
    GazeboPickPlaceController._scale_trajectory_timing(controller, SimpleNamespace(joint_trajectory=trajectory), .1)
    assert trajectory.points[-1].time_from_start.sec >= 1
    assert abs(trajectory.points[-1].velocities[0]) <= .5
    assert abs(trajectory.points[-1].accelerations[0]) <= 1.
    trajectory.points[-1].positions = [2.]
    with pytest.raises(ValueError, match='joint limit'):
        controller._validate_simulation_trajectory(trajectory)


@pytest.mark.parametrize('rate', [1, 2, 5, 10])
def test_motion_deadline_separates_accelerated_progress_from_wall_watchdog(rate, monkeypatch):
    from cais_spade_llm.resources.robot import simulation_timing

    clock = {'wall': 0., 'sim': 2., 'cancelled': False}
    monkeypatch.setattr(simulation_timing.time, 'monotonic', lambda: clock['wall'])
    deadline = MotionDeadline(2., lambda: clock['sim'], lambda: clock['cancelled'], stall_timeout=1.)
    clock.update(wall=.5/rate, sim=2.5)
    assert deadline.pending()
    clock.update(wall=2./rate, sim=4.)
    assert not deadline.pending()
    clock['cancelled'] = True
    with pytest.raises(InterruptedError):
        deadline.pending()
    clock['cancelled'] = False
    clock['sim'] = 0.
    with pytest.raises(RuntimeError, match='reset'):
        deadline.pending()
    paused = MotionDeadline(2., lambda: clock['sim'], lambda: False, stall_timeout=1.)
    clock['wall'] += 1.1
    with pytest.raises(TimeoutError, match='watchdog'):
        paused.pending()


@pytest.mark.parametrize('lookup', ['joint', 'ur5e_1_joint'])
def test_simulation_joint_aliases_cannot_revive_stale_feedback(lookup):
    controller = SimpleNamespace(
        robot_name='ur5e_1', execution_mode='simulation', _joint_lock=threading.Lock(),
        _joint_positions={'ur5e_1_joint': .2}, _joint_received_times={'ur5e_1_joint': 0.},
    )
    assert GazeboPickPlaceController._get_joint_position(controller, lookup) is None
    controller._joint_received_times['ur5e_1_joint'] = time.monotonic()
    assert GazeboPickPlaceController._get_joint_position(controller, lookup) == .2


def test_simulation_stop_cancels_the_active_controller_goal():
    from unittest.mock import Mock

    future = SimpleNamespace(done=lambda: True)
    goal = SimpleNamespace(cancel_goal_async=Mock(return_value=future))
    controller = SimpleNamespace(execution_mode='simulation', _simulation_goal=goal)
    GazeboPickPlaceController._cancel_simulation_goal(controller)
    goal.cancel_goal_async.assert_called_once()
    assert controller._simulation_goal is None


@pytest.mark.parametrize("speed", [0.1, 0.25, 0.5, 1.0, 2.0])
def test_controlled_transport_profiles_reach_target_with_bounded_acceleration(speed):
    duration = _motion_duration(1.0, speed, 1.0)
    assert duration > 0.0
    assert _motion_distance(0.0, 1.0, speed, 1.0) == 0.0
    assert _motion_distance(duration, 1.0, speed, 1.0) == pytest.approx(1.0)
    assert _motion_distance(duration / 2.0, 1.0, speed, 1.0) <= 1.0


@pytest.mark.parametrize('feedback, succeeds', [([None, .04, 0.], True), ([None, None, None], False)])
def test_controller_acknowledgement_also_requires_fresh_observed_targets(feedback, succeeds, monkeypatch):
    messages = pytest.importorskip('trajectory_msgs.msg')
    from unittest.mock import Mock

    trajectory = messages.JointTrajectory(joint_names=['joint'])
    trajectory.points = [messages.JointTrajectoryPoint(positions=[0.])]
    result = SimpleNamespace(status=4, result=SimpleNamespace(error_code=0))
    goal = SimpleNamespace(accepted=True, status=4, get_result_async=lambda: result)
    client = SimpleNamespace(wait_for_server=lambda **kwargs: True, send_goal_async=lambda _: goal)
    samples = iter(feedback)
    polls = iter([True, True, True, False])
    measured = Mock(side_effect=lambda _: next(samples))
    controller = SimpleNamespace(
        _simulation_joint_clients={'/arm/follow_joint_trajectory': client},
        _wait_future=lambda future, *args: future, _get_joint_position=measured,
        _motion_pending=lambda _: lambda: next(polls), _simulation_goal=None,
        arm_joint_names=['joint'],
        _angular_joint_error=GazeboPickPlaceController._angular_joint_error,
        _note_motion_dispatch=Mock(),
        _node=SimpleNamespace(get_clock=lambda: SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=0))),
    )
    monkeypatch.setattr(time, 'sleep', lambda _: None)
    assert GazeboPickPlaceController._send_simulation_joint_trajectory(
        controller, '/arm/joint_trajectory', trajectory) is succeeds
    assert measured.call_count == 3
    assert controller._simulation_goal is None


def test_controller_endpoint_accepts_equivalent_revolute_joint_angle(monkeypatch):
    messages = pytest.importorskip('trajectory_msgs.msg')

    trajectory = messages.JointTrajectory(joint_names=['joint'])
    trajectory.points = [messages.JointTrajectoryPoint(positions=[0.])]
    result = SimpleNamespace(status=4, result=SimpleNamespace(error_code=0))
    goal = SimpleNamespace(accepted=True, status=4, get_result_async=lambda: result)
    client = SimpleNamespace(wait_for_server=lambda **kwargs: True, send_goal_async=lambda _: goal)
    controller = SimpleNamespace(
        _simulation_joint_clients={'/arm/follow_joint_trajectory': client},
        _wait_future=lambda future, *args: future,
        _get_joint_position=lambda _name: 2.0 * math.pi,
        _motion_pending=lambda _: lambda: True,
        _simulation_goal=None,
        arm_joint_names=['joint'],
        _angular_joint_error=GazeboPickPlaceController._angular_joint_error,
        _note_motion_dispatch=Mock(),
        _node=SimpleNamespace(get_clock=lambda: SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=0))),
    )
    monkeypatch.setattr(time, 'sleep', lambda _: None)

    assert GazeboPickPlaceController._send_simulation_joint_trajectory(
        controller, '/arm/joint_trajectory', trajectory,
    ) is True


@pytest.mark.parametrize("observed_offset, expected", [(0.003, True), (0.0168, False)])
@pytest.mark.parametrize("joint_endpoint_observed", [True, False])
def test_cartesian_controller_success_requires_observed_endpoint(
    observed_offset, expected, joint_endpoint_observed,
):
    controller = GazeboPickPlaceController.__new__(GazeboPickPlaceController)
    target = SimpleNamespace(
        position=SimpleNamespace(x=0.4, y=0.3, z=1.2),
        orientation=SimpleNamespace(x=0.0, y=1.0, z=0.0, w=0.0),
    )
    controller.execution_mode = "simulation"
    controller.arm_trajectory_topic = "/arm/joint_trajectory"
    controller._last_simulation_controller_succeeded = True
    controller.cartesian_position_tolerance_m = 0.005
    controller.cartesian_orientation_tolerance_rad = 0.02
    controller._get_ee_pose = lambda: SimpleNamespace(
        position=SimpleNamespace(x=0.4, y=0.3, z=1.2 + observed_offset),
        orientation=target.orientation,
    )
    controller._motion_pending = lambda seconds: lambda: False
    controller._planning_wall_time_sec = 0.0
    controller._trajectory_duration_sec = 0.0
    controller.trajectory_time_scale = 1.0
    controller._last_failure_message = ""
    trajectory = SimpleNamespace(points=[SimpleNamespace(
        time_from_start=SimpleNamespace(sec=1, nanosec=0))])
    controller._consume_prepared_cartesian = lambda target: (
        SimpleNamespace(joint_trajectory=trajectory), "reused test preparation")
    controller._ExecuteTrajectory = SimpleNamespace(Goal=SimpleNamespace)
    controller._scale_trajectory_timing = Mock()
    controller._send_simulation_joint_trajectory = Mock(return_value=joint_endpoint_observed)
    warnings = []
    logger = SimpleNamespace(warning=lambda message: warnings.append(message), error=Mock())
    controller._log = lambda: logger

    assert controller._cartesian_move(target) is expected
    assert controller._last_command_evidence["controller_endpoint_observed"][
        "within_tolerance"] is expected
    assert len(warnings) == int(expected and not joint_endpoint_observed)
    if warnings:
        assert "position_error=0.003000m" in warnings[0]



@pytest.mark.parametrize("target_kind,joint_observed,controller_succeeded,pose_observed,expected", [
    ("pose", False, True, True, True),
    ("pose", False, True, False, False),
    ("pose", True, True, True, True),
    ("pose", True, True, False, False),
    ("pose", False, False, True, False),
    ("pose", False, True, None, False),
    ("joints", False, True, True, False),
    ("joints", True, True, False, True),
])
def test_free_space_completion_validates_the_requested_target(
    target_kind, joint_observed, controller_succeeded, pose_observed, expected,
):
    geometry = pytest.importorskip("geometry_msgs.msg")
    pytest.importorskip("moveit_msgs.msg")
    controller = GazeboPickPlaceController.__new__(GazeboPickPlaceController)
    controller.execution_mode = "simulation"
    controller.arm_trajectory_topic = "/arm/joint_trajectory"
    controller.arm_joint_names = ["joint_a"]
    controller.group_name, controller.frame_id, controller.ee_link = "arm", "world", "tool0"
    controller._GetMotionPlan = SimpleNamespace(Request=lambda: SimpleNamespace(
        motion_plan_request=SimpleNamespace(start_state=SimpleNamespace())))
    trajectory = SimpleNamespace(joint_trajectory=SimpleNamespace(points=[SimpleNamespace(
        time_from_start=SimpleNamespace(sec=1, nanosec=0))]))
    response = SimpleNamespace(motion_plan_response=SimpleNamespace(
        error_code=SimpleNamespace(val=1), trajectory=trajectory))
    controller._motion_plan_client = SimpleNamespace(call_async=Mock(return_value=response))
    controller._wait_future = lambda future, **_kwargs: future
    controller._planning_wall_time_sec = controller._trajectory_duration_sec = 0.
    controller._scale_trajectory_timing = Mock()
    controller._send_simulation_joint_trajectory = Mock(return_value=joint_observed)
    controller._last_simulation_controller_succeeded = controller_succeeded
    controller._last_failure_message = "joint settling mismatch" if not joint_observed else ""
    controller._motion_pending = lambda _timeout: lambda: False
    endpoint = None if pose_observed is None else {"within_tolerance": pose_observed}
    controller._observed_cartesian_endpoint_evidence = Mock(return_value=endpoint)
    controller._log = lambda: SimpleNamespace(error=lambda message: None)
    pose = geometry.Pose()
    pose.orientation.w = 1.
    target = {"target_pose": pose} if target_kind == "pose" else {"joint_positions": [0.]}

    assert controller._execute_simulation_motion_plan(label="test", **target) is expected
    if target_kind == "joints" or not controller_succeeded:
        controller._observed_cartesian_endpoint_evidence.assert_not_called()
    if expected:
        assert controller._last_command_evidence["joint_endpoint_observed"] is joint_observed
        if target_kind == "pose":
            assert controller._last_command_evidence["controller_endpoint_observed"]["within_tolerance"]


def test_cartesian_endpoint_waits_for_pose_feedback_without_fixed_delay(monkeypatch):
    old_pose = {"within_tolerance": False, "position_error_m": 0.0168}
    final_pose = {"within_tolerance": True, "position_error_m": 0.0001}
    observe = Mock(side_effect=[old_pose, final_pose])
    controller = SimpleNamespace(
        _last_simulation_controller_succeeded=True,
        _observed_cartesian_endpoint_evidence=observe,
        _motion_pending=lambda seconds: lambda: True,
    )
    monkeypatch.setattr(time, "sleep", lambda seconds: None)
    evidence = GazeboPickPlaceController._wait_for_simulation_cartesian_endpoint(
        controller, object())
    assert evidence is final_pose
    assert observe.call_count == 2


@pytest.mark.parametrize("limit, expected", [(2 * math.pi, 4.687), (math.pi, 4.687 - 2 * math.pi)])
def test_home_target_avoids_a_full_rotation_within_joint_limits(limit, expected):
    nominal = 4.687 - 2 * math.pi
    controller = SimpleNamespace(
        arm_joint_names=["shoulder"],
        _simulation_joint_limits=lambda names: {"shoulder": {"lower": -limit, "upper": limit}},
        _get_arm_joint_positions=lambda **kwargs: ([3.753], []),
    )
    target = GazeboPickPlaceController._nearest_simulation_joint_targets(controller, [nominal])
    assert target == pytest.approx([expected])
    assert abs(math.atan2(math.sin(target[0] - nominal), math.cos(target[0] - nominal))) < 1e-8


def test_observed_payload_bounds_preserve_neighbours_and_configured_support_contact():
    from cais_spade_llm.recovery_framework.part_collision import observed_part_boxes

    pose = SimpleNamespace(position=SimpleNamespace(x=-2.6, y=.5, z=.99),
                           orientation=SimpleNamespace(x=0., y=0., z=0., w=1.))
    full = observed_part_boxes("RGOCG4-50_Round_4mm", pose)
    carried = observed_part_boxes("RGOCG4-50_Round_4mm", pose, support_allowance=.001)
    assert len(full) == len(carried) == 1
    assert full[0]["id"].startswith("RGOCG4-50_Round_4mm/")
    assert carried[0]["size"][:2] == full[0]["size"][:2]
    assert carried[0]["size"][2] == pytest.approx(full[0]["size"][2] - .001)
    assert carried[0]["pose"][2] == pytest.approx(full[0]["pose"][2] + .0005)
    with pytest.raises(ValueError, match="No collision geometry"):
        observed_part_boxes("unknown", pose)


@pytest.mark.parametrize("attached", [False, True])
def test_payload_scene_diff_updates_only_owned_part(monkeypatch, attached):
    import sys
    from cais_spade_llm.recovery_framework.part_collision import part_scene_update

    class CollisionObject(SimpleNamespace):
        ADD = 0
        REMOVE = 1
        def __init__(self, **kwargs):
            super().__init__(header=SimpleNamespace(frame_id=""), **kwargs)

    class PlanningScene(SimpleNamespace):
        def __init__(self, **kwargs):
            super().__init__(robot_state=SimpleNamespace(attached_collision_objects=[]),
                             world=SimpleNamespace(collision_objects=[]), **kwargs)

    monkeypatch.setitem(sys.modules, "geometry_msgs.msg", SimpleNamespace(Pose=lambda: SimpleNamespace(
        position=SimpleNamespace(), orientation=SimpleNamespace())))
    monkeypatch.setitem(sys.modules, "moveit_msgs.msg", SimpleNamespace(
        CollisionObject=CollisionObject, PlanningScene=PlanningScene, AttachedCollisionObject=SimpleNamespace))
    class SolidPrimitive(SimpleNamespace):
        BOX = 1
    monkeypatch.setitem(sys.modules, "shape_msgs.msg", SimpleNamespace(SolidPrimitive=SolidPrimitive))
    scene = part_scene_update("peg", [{"id": "peg/link/collision", "pose": [1., 2., 3., 0., 0., 0., 1.],
                                      "size": [.01, .01, .05]}],
                              attached_link="owner_tcp" if attached else None, touch_links=["owner_finger"],
                              world_ids={"peg", "neighbour/link/collision"},
                              attached_ids={"peg/link/collision", "other_robot_payload"})
    assert {obj.id for obj in scene.world.collision_objects} == ({"peg"} if attached else {"peg", "peg/link/collision"})
    if attached:
        assert all(obj.operation == CollisionObject.REMOVE for obj in scene.world.collision_objects)
        payload, = scene.robot_state.attached_collision_objects
        assert payload.link_name == "owner_tcp"
        assert payload.touch_links == ["owner_finger"]
        assert payload.object.operation == CollisionObject.ADD
        assert payload.object.header.frame_id == "owner_tcp"
        assert payload.object.primitive_poses[0].position.z == 3.
    else:
        assert all(obj.object.operation == CollisionObject.REMOVE for obj in scene.robot_state.attached_collision_objects)
        assert scene.world.collision_objects[-1].operation == CollisionObject.ADD
        assert scene.world.collision_objects[-1].header.frame_id == "world"


@pytest.mark.parametrize("attached", [False, True])
@pytest.mark.parametrize("acknowledged", [False, True])
@pytest.mark.parametrize("offset", [0., 2.])
@pytest.mark.parametrize("payload_timeouts", [0, 1])
def test_payload_observation_uses_the_physical_attachment_frame(
    monkeypatch, attached, acknowledged, offset, payload_timeouts,
):
    import sys
    from cais_spade_llm.recovery_framework import part_collision

    class Components(SimpleNamespace):
        ALLOWED_COLLISION_MATRIX = 1
        WORLD_OBJECT_NAMES = 2
        ROBOT_STATE_ATTACHED_OBJECTS = 4

    monkeypatch.setitem(sys.modules, "moveit_msgs.msg", SimpleNamespace(PlanningSceneComponents=Components))
    service = SimpleNamespace(Request=SimpleNamespace)
    monkeypatch.setitem(sys.modules, "moveit_msgs.srv", SimpleNamespace(
        GetPlanningScene=service, ApplyPlanningScene=service))
    pose = object()
    rows = [{"id": "peg/link/collision", "pose": [offset, 0., .243, 0., 0., 0., 1.],
             "size": [.008, .007, .049]}]
    observe = Mock(return_value=rows)
    scene_update = Mock(return_value=object())
    monkeypatch.setattr(part_collision, "observed_part_boxes", observe)
    monkeypatch.setattr(part_collision, "part_scene_update", scene_update)
    get_pose = Mock()
    pose_response = SimpleNamespace(success=True, state=SimpleNamespace(pose=pose))
    get_pose.call_async.side_effect = [None] * payload_timeouts + [pose_response]
    gazebo_node = Mock()
    gazebo_node.create_client.return_value = get_pose
    get_scene = Mock()
    get_scene.call_async.return_value = SimpleNamespace(scene=SimpleNamespace(
        allowed_collision_matrix=SimpleNamespace(entry_names=["owner_rg2_finger", "other_rg2_finger"]),
        world=SimpleNamespace(collision_objects=[]),
        robot_state=SimpleNamespace(attached_collision_objects=[]),
    ))
    apply_scene = Mock()
    apply_scene.call_async.return_value = SimpleNamespace(success=acknowledged)
    controller = SimpleNamespace(
        controller_config={"payload_collision": {"enabled": True, "robot_prefix": "owner_",
                                                 "support_contact_allowance_m": .001}},
        execution_mode="simulation", robot_model_name="robot", attach_link_candidates=["owner_tcp"],
        _attached_model="peg" if attached else None,
        _payload_scene_clients={"get": get_scene, "apply": apply_scene},
        _get_state_client=get_pose, _GetEntityState=service,
        _wait_future=lambda future, **kwargs: future, _log=Mock(),
        _node=gazebo_node, _cb_group=object(), service_get_entity_state="/get_entity_state",
        _tf_buffer=SimpleNamespace(lookup_transform=lambda *args: SimpleNamespace(
            transform=SimpleNamespace(translation=SimpleNamespace(x=0., y=0., z=.243)))),
        _rclpy=SimpleNamespace(time=SimpleNamespace(Time=lambda: None)),
        tcp_link="owner_tcp", cartesian_position_tolerance_m=.005,
    )
    assert GazeboPickPlaceController._sync_part_collision(
        controller, "peg", attached_link="owner_tcp" if attached else None,
    ) is (acknowledged and not (attached and offset))
    evidence = controller._last_command_evidence["payload_collision"]
    if attached and offset:
        assert evidence["physical_attachment_completed"] is True
        assert evidence["payload_at_gripper"] is False
        assert evidence["collision_scene_acknowledged"] is False
        assert "outside the gripper" in evidence["error"]
        assert evidence["collision_objects"] == rows
        apply_scene.call_async.assert_not_called()
        return
    assert get_pose.call_async.call_count == payload_timeouts + 1
    assert gazebo_node.destroy_client.call_count == payload_timeouts
    query = get_pose.call_async.call_args.args[0]
    assert query.name == "peg"
    assert query.reference_frame == ("robot::owner_tcp" if attached else "world")
    observe.assert_called_once_with("peg", pose, support_allowance=.001 if attached else 0.)
    assert scene_update.call_args.args == ("peg", rows)
    assert scene_update.call_args.kwargs["attached_link"] == ("owner_tcp" if attached else None)
    assert scene_update.call_args.kwargs["touch_links"] == ["owner_rg2_finger", "owner_tcp"]
    evidence = controller._last_command_evidence["payload_collision"]
    assert evidence["collision_frame"] == ("owner_tcp" if attached else "world")
    assert evidence["physical_attachment_completed"] is attached
    assert evidence["collision_scene_acknowledged"] is acknowledged


@pytest.mark.parametrize("point, inside", [([1., 2., 3.], True), ([1.01, 2., 3.], True),
                                          ([1.02, 2., 3.], False), ([4., -1., 0.], False)])
def test_observed_payload_must_reach_the_gripper(point, inside):
    from cais_spade_llm.recovery_framework.part_collision import grasp_point_evidence

    rows = [{"pose": [1., 2., 3., 0., 0., math.sqrt(.5), math.sqrt(.5)],
             "size": [.008, .020, .05]}]
    evidence = grasp_point_evidence(rows, point, .005)
    assert evidence["payload_at_gripper"] is inside
    assert evidence["tcp_to_payload_distance_m"] >= 0.


def test_scene_update_failure_preserves_actual_physical_attachment():
    controller = SimpleNamespace(
        _link_attacher_enabled=True, _attached_model=None, _attached_link=None,
        attach_link_candidates=["owner_tcp"], robot_model_name="robot",
        _attach_srv=SimpleNamespace(Request=SimpleNamespace), _attach_client=Mock(),
        _wait_future=Mock(return_value=SimpleNamespace(success=True)),
        clear_motion_preparation=Mock(), _sync_part_collision=Mock(return_value=False),
    )
    assert not GazeboPickPlaceController._attach_part(controller, "peg")
    assert controller._attached_model == "peg"
    assert controller._attached_link == "owner_tcp"
    controller._sync_part_collision.assert_called_once_with("peg", attached_link="owner_tcp")


def test_joint_feedback_ignores_other_robots_and_out_of_order_samples():
    clock = SimpleNamespace(nanoseconds=20_000_000_000)
    controller = SimpleNamespace(
        execution_mode="simulation", arm_joint_names=["owner_joint"], gripper_joint="owner_gripper",
        _joint_lock=threading.Lock(), _joint_positions={}, _joint_received_times={},
        _joint_sim_stamps={}, _joint_stable_since={}, _joint_limits_cache={"owner_joint": {}},
        _last_joint_sim_time=None, _node=SimpleNamespace(get_clock=lambda: SimpleNamespace(now=lambda: clock)),
    )
    def receive(name, position, stamp):
        GazeboPickPlaceController._on_joint_state(controller, SimpleNamespace(
            name=[name], position=[position], header=SimpleNamespace(stamp=SimpleNamespace(sec=stamp, nanosec=0))))
    receive("owner_joint", .2, 20)
    receive("other_robot_joint", .8, 17)
    assert controller._joint_positions == {"owner_joint": .2}
    assert controller._last_joint_sim_time == 20
    receive("owner_joint", .1, 19)
    assert controller._joint_positions == {"owner_joint": .2}
    assert controller._joint_sim_stamps == {"owner_joint": 20}
    receive("owner_gripper", .11, 19)
    assert controller._joint_positions == {"owner_joint": .2, "owner_gripper": .11}
    assert controller._joint_sim_stamps == {"owner_joint": 20, "owner_gripper": 19}
    assert controller._last_joint_sim_time == 20
    receive("owner_gripper", .04, 18)
    assert controller._joint_positions["owner_gripper"] == .11
    clock.nanoseconds = 1_000_000_000
    receive("owner_joint", .0, 1)
    assert controller._joint_positions == {"owner_joint": .0}
    assert controller._joint_limits_cache == {}


@pytest.mark.parametrize("nanoseconds, fresh", [(950_000_000, True), (500_000_000, False)])
def test_simulation_cartesian_feedback_rejects_stale_transform(nanoseconds, fresh):
    transform = SimpleNamespace(header=SimpleNamespace(stamp=SimpleNamespace(sec=10, nanosec=nanoseconds)),
        transform=SimpleNamespace(translation=SimpleNamespace(x=1., y=2., z=3.), rotation=object()))
    controller = SimpleNamespace(execution_mode="simulation", tf_lookup_timeout_sec=0.,
        frame_id="world", ee_link="tool0", _tf_buffer=SimpleNamespace(lookup_transform=lambda *args: transform),
        _rclpy=SimpleNamespace(time=SimpleNamespace(Time=lambda: 0)),
        _node=SimpleNamespace(get_clock=lambda: SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=11_000_000_000))),
        _Pose=lambda: SimpleNamespace(position=SimpleNamespace(), orientation=None),
        _tf2_ros=SimpleNamespace(LookupException=LookupError, ConnectivityException=ConnectionError,
                                ExtrapolationException=ValueError), _log=lambda: Mock(), _last_failure_message="")
    assert (GazeboPickPlaceController._get_ee_pose(controller) is not None) is fresh


def test_workflow_pose_uses_resource_owned_observation():
    observed = SimpleNamespace(position=SimpleNamespace(z=1.2))
    tcp = SimpleNamespace(position=SimpleNamespace(z=.982))
    controller = SimpleNamespace(execution_mode="simulation", controller_config={"payload_collision": {"enabled": True}},
        ee_link="tool0", tcp_link="tcp", _observed_simulation_link_poses=Mock(return_value={"tool0": observed, "tcp": tcp}),
        _tf_buffer=Mock())
    assert GazeboPickPlaceController._get_ee_pose(controller) is observed
    assert GazeboPickPlaceController._get_ee_tcp_world_z_offset(controller) == pytest.approx(-.218)
    controller._tf_buffer.lookup_transform.assert_not_called()


@pytest.mark.parametrize("condition", [
    "fresh", "stale", "missing", "timeout", "cancelled", "delayed", "cancelled_after_stale", "reply_retry", "deadline_retry",
])
@pytest.mark.parametrize("observation_timeout", [2.0, 5.0])
def test_observed_tool_poses_require_fresh_gazebo_link_measurement(
    monkeypatch, condition, observation_timeout,
):
    import itertools
    from cais_spade_llm.resources.robot import gazebo_pick_place_controller as controller_module

    ticks = itertools.count(step=.2)
    elapsed = [0.]
    monkeypatch.setattr(controller_module.time, "monotonic",
                        lambda: elapsed[0] if condition == "deadline_retry" else next(ticks))
    parent = SimpleNamespace(position=SimpleNamespace(x=1., y=2., z=3.),
                             orientation=SimpleNamespace(x=0., y=0., z=math.sqrt(.5), w=math.sqrt(.5)))
    response = SimpleNamespace(success=condition != "missing", state=SimpleNamespace(pose=parent),
                               header=SimpleNamespace(stamp=SimpleNamespace(sec=9, nanosec=0 if condition in {"stale", "delayed", "cancelled_after_stale"} else 950_000_000)))
    client = Mock()

    def fixed_transform(parent_link, child_link, _time):
        assert parent_link == "owner_wrist"
        return SimpleNamespace(transform=SimpleNamespace(
            translation=SimpleNamespace(x=.1 if child_link == "tcp" else 0., y=0., z=.2),
            rotation=SimpleNamespace(x=0., y=0., z=0., w=1.),
        ))

    controller = SimpleNamespace(
        controller_config={"payload_collision": {"observation_link": "owner_wrist"}},
        tf_lookup_timeout_sec=observation_timeout,
        robot_model_name="robot", frame_id="world", ee_link="tool0", tcp_link="tcp",
        _shutdown_requested=condition == "cancelled", _get_state_client=client, _log=lambda: Mock(),
        _GetEntityState=SimpleNamespace(Request=SimpleNamespace),
        _node=SimpleNamespace(
            get_clock=lambda: SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=10_000_000_000)),
            create_client=Mock(return_value=client), destroy_client=Mock()),
        service_get_entity_state='/get_entity_state', _cb_group=object(),
        _wait_future=Mock(return_value=None if condition == "timeout" else response),
        _tf_buffer=SimpleNamespace(lookup_transform=Mock(side_effect=fixed_transform)),
        _rclpy=SimpleNamespace(time=SimpleNamespace(Time=lambda: 0)),
        _Pose=lambda: SimpleNamespace(position=SimpleNamespace(), orientation=SimpleNamespace()),
        _tf2_ros=SimpleNamespace(LookupException=LookupError, ConnectivityException=ConnectionError,
                                ExtrapolationException=ValueError),
    )
    def receive_observation(*_args, **_kwargs):
        if condition == "cancelled_after_stale":
            controller._shutdown_requested = True
        elif condition == "delayed" and controller._wait_future.call_count > 1:
            response.header.stamp.nanosec = 950_000_000
        return response

    if condition in {"delayed", "cancelled_after_stale"}:
        controller._wait_future.side_effect = receive_observation
    elif condition == "reply_retry":
        controller._wait_future.side_effect = [None, response]
    elif condition == "deadline_retry":
        def delayed_reply(_future, **kwargs):
            if controller._wait_future.call_count == 1:
                elapsed[0] += kwargs['timeout_sec']
                return None
            return response
        controller._wait_future.side_effect = delayed_reply
    result = GazeboPickPlaceController._observed_simulation_link_poses(controller)
    if condition in {"fresh", "delayed", "reply_retry", "deadline_retry"}:
        assert result["tool0"].position.z == pytest.approx(3.2)
        assert result["tcp"].position.y == pytest.approx(2.1)
        assert result["tcp"].position.x == pytest.approx(1.)
        request = client.call_async.call_args.args[0]
        assert (request.name, request.reference_frame) == ("robot::owner_wrist", "world")
        assert controller._last_pose_observation["source"] == "gazebo_link_state"
        assert controller._tf_buffer.lookup_transform.call_count == 2
        assert client.call_async.call_count == (2 if condition in {"delayed", "reply_retry", "deadline_retry"} else 1)
        assert controller._wait_future.call_args_list[0].kwargs['timeout_sec'] > 1.
        assert controller._node.destroy_client.call_count == (1 if condition in {'reply_retry', 'deadline_retry'} else 0)
    else:
        assert result is None
        assert controller._last_failure_message
        controller._tf_buffer.lookup_transform.assert_not_called()
        if condition == "cancelled":
            client.call_async.assert_not_called()
        elif condition == "cancelled_after_stale":
            assert client.call_async.call_count == 1
        elif condition in {"stale", "timeout"}:
            assert client.call_async.call_count > 1
            if condition == "timeout":
                assert client.call_async.call_count == 3
                assert controller._node.destroy_client.call_count == 2
            assert max(call.kwargs["timeout_sec"] for call in controller._wait_future.call_args_list) <= observation_timeout


@pytest.mark.parametrize("feedback_available", [False, True])
def test_joint_target_timing_waits_for_observed_gripper_feedback(monkeypatch, feedback_available):
    from cais_spade_llm.resources.robot import gazebo_pick_place_controller as controller_module

    limits = {"gripper": {"lower": 0., "upper": .11, "velocity": .1, "acceleration": .5}}
    feedback = Mock(side_effect=[None, .11] if feedback_available else [None])
    controller = SimpleNamespace(
        execution_mode="simulation", _simulation_joint_limits=lambda _names: limits,
        _get_joint_position=feedback, _trajectory_duration_sec=0.,
        tf_lookup_timeout_sec=2. if feedback_available else 0.,
    )
    target = SimpleNamespace(positions=[.02], time_from_start=SimpleNamespace(sec=0, nanosec=100_000_000))
    trajectory = SimpleNamespace(joint_names=["gripper"], points=[target])
    monkeypatch.setattr(controller_module.time, "sleep", lambda _delay: None)
    if feedback_available:
        GazeboPickPlaceController._time_joint_target(controller, trajectory)
        duration = target.time_from_start.sec + target.time_from_start.nanosec / 1e9
        assert duration >= 1.875 * (.11 - .02) / limits["gripper"]["velocity"]
        assert feedback.call_count == 2
    else:
        with pytest.raises(ValueError, match="Missing observed joint position: gripper"):
            GazeboPickPlaceController._time_joint_target(controller, trajectory)
        assert controller._trajectory_duration_sec == 0.


@pytest.mark.parametrize("failure", [None, "collision", "missing_observation", "transient_observation", "stop"])
def test_timed_motion_checks_intermediate_states_before_dispatch(failure):
    pytest.importorskip("moveit_msgs.srv")
    checked = []

    def valid(request):
        position = request.robot_state.joint_state.position[0]
        checked.append(position)
        if position == .05 and (failure == "missing_observation" or (
            failure == "transient_observation" and checked.count(.05) == 1
        )):
            return None
        collision = failure == "collision" and position == .05
        return SimpleNamespace(valid=not collision, contacts=[SimpleNamespace(
            contact_body_1="finger", contact_body_2="peg")] if collision else [])

    controller = GazeboPickPlaceController.__new__(GazeboPickPlaceController)
    controller.execution_mode = "simulation"
    controller.controller_config = {"payload_collision": {"enabled": True}}
    controller.group_name = "arm"
    controller._shutdown_requested = failure == "stop"
    controller._planning_wall_time_sec = 0.
    controller.tf_lookup_timeout_sec = .02
    controller._state_validity_client = SimpleNamespace(call_async=valid)
    controller._wait_future = lambda response, **kwargs: response
    trajectory = SimpleNamespace(joint_names=["joint"], points=[
        SimpleNamespace(positions=[0.]), SimpleNamespace(positions=[.1]),
    ])
    assert controller._simulation_trajectory_is_collision_free(trajectory) is (failure in {None, "transient_observation"})
    if failure == "stop":
        assert not checked
    else:
        assert .05 in checked
    if failure in {"collision", "missing_observation"}:
        assert controller._last_command_evidence["command_sent"] is False
        assert .1 not in checked
    if failure is None:
        assert controller._last_path_validation["collision_free"] is True
        assert checked == [0., .05, .1]
    if failure == "transient_observation":
        assert checked == [0., .05, .05, .1]
        assert controller._last_path_validation["observation_attempts"] == 4


def test_cartesian_joint_branch_jump_is_not_dispatched():
    pytest.importorskip("moveit_msgs.srv")
    from moveit_msgs.srv import GetCartesianPath
    from geometry_msgs.msg import Pose
    controller = GazeboPickPlaceController.__new__(GazeboPickPlaceController)
    controller.execution_mode = "simulation"
    controller._consume_prepared_cartesian = lambda _target: (None, "cache miss")
    controller._GetCartesianPath = GetCartesianPath
    controller.frame_id, controller.group_name, controller.ee_link = "world", "arm", "tool0"
    controller._node = SimpleNamespace(get_clock=lambda: SimpleNamespace(
        now=lambda: SimpleNamespace(to_msg=lambda: pytest.importorskip("builtin_interfaces.msg").Time())))
    controller._planning_wall_time_sec = 0.
    requests = []

    def plan(request):
        requests.append(request)
        # The observed branch discontinuity truncates the path with jump checks enabled.
        return SimpleNamespace(fraction=.75 if request.jump_threshold > 0 else 1.)

    controller._cart_client = SimpleNamespace(call_async=plan)
    controller._wait_future = lambda response, **kwargs: response
    controller._log = lambda: Mock()
    controller._send_simulation_joint_trajectory = Mock()
    assert controller._cartesian_move(Pose()) is False
    controller._send_simulation_joint_trajectory.assert_not_called()
    assert requests[0].avoid_collisions


@pytest.mark.parametrize("rollback_ok", [True, False])
def test_failed_grasp_retains_attachment_evidence_after_rollback(rollback_ok):
    payload = {"physical_attachment_completed": True, "payload_at_gripper": False,
               "tcp_to_payload_distance_m": .0615, "collision_scene_acknowledged": False}
    controller = SimpleNamespace(wait_for_services=lambda: True,
                                 close_gripper=lambda **kwargs: True,
                                 _last_failure_message="")

    def attach(*args, **kwargs):
        controller._last_command_evidence = {"payload_collision": payload}
        return {"success": False, "message": "observed payload outside gripper",
                "payload_collision": payload}

    def rollback():
        controller._last_command_evidence = {"target": .11, "command_sent": True}
        return rollback_ok

    controller.attach_part, controller.open_gripper = attach, rollback
    result = GazeboPickPlaceController.grasp_part(controller, "peg")
    assert result["success"] is False
    assert result["payload_collision"] == payload
    assert controller._last_command_evidence["payload_collision"] == payload
    assert controller._last_command_evidence["rollback"]["success"] is rollback_ok
    assert result["rollback"]["command_evidence"]["target"] == .11


def test_cartesian_dispatch_rejects_a_colliding_timed_sample():
    pytest.importorskip('moveit_msgs.srv')
    controller = GazeboPickPlaceController.__new__(GazeboPickPlaceController)
    controller.execution_mode = 'simulation'
    controller.controller_config = {'payload_collision': {'enabled': True}}
    controller._shutdown_requested = False
    controller.group_name = 'arm'
    trajectory = SimpleNamespace(joint_names=['joint'], points=[
        SimpleNamespace(positions=[0.]), SimpleNamespace(positions=[.1]),
    ])
    controller._consume_prepared_cartesian = lambda _target: (
        SimpleNamespace(joint_trajectory=trajectory), 'prepared')
    controller._planning_wall_time_sec = 0.
    controller._ExecuteTrajectory = SimpleNamespace(Goal=SimpleNamespace)
    controller.trajectory_time_scale = 1.
    controller._scale_trajectory_timing = Mock()
    controller._wait_future = lambda response, **kwargs: response
    controller._state_validity_client = SimpleNamespace(call_async=lambda request:
        SimpleNamespace(valid=not math.isclose(request.robot_state.joint_state.position[0], .05), contacts=[]))
    controller._send_simulation_joint_trajectory = Mock()
    assert controller._cartesian_move(object()) is False
    controller._send_simulation_joint_trajectory.assert_not_called()
    assert controller._last_command_evidence['command_sent'] is False


@pytest.mark.parametrize("parts, expected", [
    ("all", "KET4_Square_4mm"),
    (["gear_small", "RGOCG4-50_Round_4mm"], "RGOCG4-50_Round_4mm"),
    (["gear_small"], None),
])
def test_launch_selects_storage_part_without_empty_ros_argument(tmp_path, monkeypatch, parts, expected):
    import json
    import shlex
    from cais_spade_llm.recovery_framework import simulation

    monkeypatch.setattr(simulation, "ROOT", tmp_path)
    (tmp_path / "order.json").write_text(json.dumps({"parts": parts}))
    (tmp_path / "scene.json").write_text(json.dumps({"Storage": {"slots": {
        "KET4_Square_4mm": {}, "RGOCG4-50_Round_4mm": {},
    }}}))
    setup = tmp_path / "setup.json"
    setup.write_text(json.dumps({"selected_product_order_file": "order.json", "scene_file": "scene.json"}))
    arguments = dict(arg.split(":=", 1) for arg in shlex.split(simulation.launch_arguments(setup)))
    assert arguments.get("kmr_initial_part") == expected
    assert all(arguments.values())


@pytest.mark.parametrize('viewer', [15, 30, 60])
@pytest.mark.parametrize('threads', [0, 2, 4])
def test_viewer_and_ode_settings_are_forwarded_only_at_launch(tmp_path, viewer, threads):
    import json
    import shlex
    from cais_spade_llm.recovery_framework.simulation import launch_arguments, simulation_settings

    setup = {'simulation': {'gazebo_gui_rate_hz': viewer, 'ode_island_threads': threads,
                            'speed': 1, 'ur_controller_rate_hz': 250}}
    settings = simulation_settings(setup)
    assert settings['gazebo_gui_rate_hz'] == viewer
    assert settings['ode_island_threads'] == threads
    assert simulation_settings({})['ode_island_threads'] == 0
    path = tmp_path / 'setup.json'
    path.write_text(json.dumps(setup))
    arguments = dict(arg.split(':=', 1) for arg in shlex.split(launch_arguments(path)))
    assert arguments['gazebo_gui_rate_hz'] == str(viewer)
    assert arguments['ode_island_threads'] == str(threads)
    assert arguments['simulation_speed'] == '1'
    assert arguments['ur_controller_rate_hz'] == '250'
    assert arguments['dynamic_shadows'] == arguments['enable_camera_streams'] == 'false'


@pytest.mark.parametrize('setting,value', [
    ('ode_island_threads', 1), ('ode_island_threads', 8), ('ode_island_threads', True),
    ('ode_island_threads', 2.0), ('gazebo_gui_rate_hz', 10), ('gazebo_gui_rate_hz', True),
])
def test_viewer_and_ode_settings_reject_unsupported_values(setting, value):
    from cais_spade_llm.recovery_framework.simulation import simulation_settings

    with pytest.raises(ValueError):
        simulation_settings({'simulation': {setting: value}})


@pytest.mark.parametrize('part', ['gear_small', 'gear_medium', 'gear_large'])
def test_recovery_gear_collision_mesh_keeps_open_bore_and_fixture_height(part):
    import json
    from cais_spade_llm.recovery_framework import ROOT
    from cais_spade_llm.recovery_framework.geometry import collision_boxes

    world = ROOT / 'ros2/cais_lab_robotics/worlds/table_recovery_framework.world'
    models = ROOT / 'ros2/cais_lab_robotics/models'
    rows = collision_boxes(world, models, {part: [0., 0., 0., 0., 0., 0., 1.]},
                           exact_models={part, 'Gear_Plate'})
    gear = next(row for row in rows if row['id'].startswith(part + '/'))
    assert 'mesh' in gear
    vertices = gear['mesh']['vertices']
    for face in gear['mesh']['triangles']:
        a, b, c = [vertices[i] for i in face]
        cross = lambda u, v: u[0] * v[1] - u[1] * v[0]
        area = cross([b[i] - a[i] for i in range(2)], [c[i] - a[i] for i in range(2)])
        if abs(area) < 1e-15:
            continue
        weights = [cross(b, c) / area, cross(c, a) / area, cross(a, b) / area]
        assert not all(value >= -1e-10 for value in weights), 'Collision triangles fill the bore'
    shaft = next(row for row in rows if row['id'] == 'Gear_Plate/Gear_Shaft_1/collision')
    assert shaft['cylinder'] == [.02, .005]
    plate = next(row for row in rows if row['id'] == 'Gear_Plate/Gear_Plate/collision')
    top = plate['pose'][2] + plate['size'][2] / 2
    board = json.loads((ROOT / 'cais_spade_llm/specification/products/geometry/assembly_board-v1-recovery-framework.json').read_text())['gazebo']['assembly_board']
    assert board['slot_floor_z_m_by_part'][part] == pytest.approx(top, abs=2e-8)
    assert board['target_origin_z_m'][part] == pytest.approx(top + .01, abs=2e-8)


@pytest.mark.parametrize('change, permitted', [
    ({}, True), ({'contact_body_2': 'Gear_Plate/Gear_Shaft_2/collision'}, False),
    ({'contact_body_1': 'ur5e_4_rg2_left_inner_finger'}, False),
    ({'depth': .0006}, False), ({'depth': float('nan')}, False),
])
def test_mating_contact_only_permits_own_part_and_shaft_with_bounded_depth(change, permitted):
    from cais_spade_llm.recovery_framework.part_collision import mating_contacts_allowed

    authorization = {'model_name': 'gear_small', 'target_collision_object': 'Gear_Plate/Gear_Shaft_1/collision',
                     'max_contact_depth_m': .0005}
    contact = SimpleNamespace(**{'contact_body_1': 'gear_small/link/collision',
                                 'contact_body_2': 'Gear_Plate/Gear_Shaft_1/collision', 'depth': .00001, **change})
    assert mating_contacts_allowed(authorization, [contact]) is permitted
    assert not mating_contacts_allowed(authorization, [])
    assert not mating_contacts_allowed(authorization, [contact, SimpleNamespace(
        contact_body_1='gear_small/link/collision', contact_body_2='gear_medium/link/collision', depth=0.)])


@pytest.mark.parametrize('pose, permitted', [
    ([0., 0., 1.04, 0., 0., 0., 1.], True),
    ([.001, 0., 1.04, 0., 0., 0., 1.], False),
    ([0., 0., 1.02, 0., 0., 0., 1.], False),
    ([0., 0., 1.04, .1, 0., 0., .995], False),
])
def test_mating_corridor_requires_aligned_upright_insertion(pose, permitted):
    from cais_spade_llm.recovery_framework.part_collision import mating_pose_valid

    authorization = {'target_origin_pose': {'x': 0., 'y': 0., 'z': 1.039},
                     'start_part_z': 1.09, 'axis_tolerance_m': .0005}
    assert mating_pose_valid(authorization, pose) is permitted


@pytest.mark.parametrize('x, code, permitted', [(0., 1, True), (.002, 1, False), (0., -1, False)])
def test_contacting_trajectory_sample_requires_valid_forward_kinematics(monkeypatch, x, code, permitted):
    import sys

    request_type = lambda **kwargs: SimpleNamespace(header=SimpleNamespace(), **kwargs)
    monkeypatch.setitem(sys.modules, 'moveit_msgs.srv', SimpleNamespace(
        GetPositionFK=SimpleNamespace(Request=request_type)))
    controller = GazeboPickPlaceController.__new__(GazeboPickPlaceController)
    controller._mating_fk_client = Mock()
    controller.ee_link = 'ur5e_4_tool0'
    controller.frame_id = 'world'
    controller._wait_future = Mock(return_value=SimpleNamespace(
        error_code=SimpleNamespace(val=code), pose_stamped=[SimpleNamespace(pose=SimpleNamespace(
            position=SimpleNamespace(x=x, y=0., z=1.04),
            orientation=SimpleNamespace(x=0., y=0., z=0., w=1.)))]))
    authorization = {'model_name': 'gear_small', 'target_collision_object': 'Gear_Plate/Gear_Shaft_1/collision',
                     'axis_tolerance_m': .0005, 'max_contact_depth_m': .0005,
                     'target_origin_pose': {'x': 0., 'y': 0., 'z': 1.039}, 'start_part_z': 1.09,
                     'tool_to_part_pose': [0., 0., 0., 0., 0., 0., 1.]}
    contact = SimpleNamespace(contact_body_1='gear_small/link/collision',
                              contact_body_2='Gear_Plate/Gear_Shaft_1/collision', depth=.00001)
    assert controller._validate_simulation_mating_contact(authorization, [contact], object()) is permitted


@pytest.mark.parametrize("part", ["gear_small", "gear_medium", "gear_large"])
def test_gear_pick_uses_configured_cad_width_with_mesh_collision(part):
    from cais_spade_llm.product.environment import EnvironmentProductContext
    from cais_spade_llm.ui.recovery_setup import load_setup, validate_setup
    from cais_spade_llm.recovery_framework.workflow_execution import _pick_geometry

    inputs = validate_setup(load_setup())
    context = EnvironmentProductContext(inputs["scene"], inputs["product_order"], inputs["geometry"])
    geometry = _pick_geometry(context, part, "3D Printing Station", "ur5e-4")
    expected = max(context.geometry[part]["dimensions_m"][:2])
    controller = GazeboPickPlaceController.__new__(GazeboPickPlaceController)
    controller.gripper_open = .11
    controller.gripper_close = 0.
    assert geometry["grasp_width_m"] == expected
    assert controller._derive_gripper_close_position(
        model_name=part, product_geometry=geometry) == pytest.approx(expected)


def test_simulation_mating_targets_compensate_measured_grasp_offset():
    from cais_spade_llm.recovery_framework.geometry import compose, multiply, rotate

    controller = GazeboPickPlaceController.__new__(GazeboPickPlaceController)
    controller._attached_model = 'gear_medium'
    controller.insertion_depth_m = .0025
    controller.frame_id = 'world'
    current = SimpleNamespace(position=SimpleNamespace(x=.4, y=-.5, z=1.4),
                              orientation=SimpleNamespace(x=0., y=1., z=0., w=0.))
    part = SimpleNamespace(position=SimpleNamespace(x=.40006, y=-.50005, z=1.177),
                           orientation=SimpleNamespace(x=0., y=0., z=.001, w=math.sqrt(1-.001**2)))
    controller._get_ee_pose = lambda: current
    controller._get_state_client = Mock()
    controller._GetEntityState = SimpleNamespace(Request=lambda **kw: SimpleNamespace(**kw))
    controller._wait_future = Mock(return_value=SimpleNamespace(success=True, state=SimpleNamespace(pose=part)))
    target = {'x': -.0005504, 'y': .1448242, 'z': 1.0389916}
    result = controller._simulation_mating_poses('gear_medium', target, math.pi/2)
    q = [part.orientation.x, part.orientation.y, part.orientation.z, part.orientation.w]
    offset = [*rotate([0., -1., 0., 0.], [.00006, -.00005, -.223]), *multiply([0., -1., 0., 0.], q)]
    insert = result['insert_pose']
    achieved = compose([insert[k] for k in ('x','y','z','qx','qy','qz','qw')], offset)
    assert achieved[:3] == pytest.approx(list(target.values()), abs=1e-12)
    assert achieved[3:] == pytest.approx([0., 0., math.sqrt(.5), math.sqrt(.5)], abs=1e-12)
    fresh = SimpleNamespace(success=True, state=SimpleNamespace(pose=part))
    controller._node = Mock()
    controller._node.create_client.return_value = Mock()
    controller._log = Mock(return_value=Mock())
    controller.service_get_entity_state = '/get_entity_state'
    controller._cb_group = object()
    controller._wait_future = Mock(side_effect=[None, fresh])
    retried = controller._simulation_mating_poses('gear_medium', target, math.pi/2)
    assert retried['insert_pose'] == result['insert_pose']
    controller._node.destroy_client.assert_called_once()
    assert controller._wait_future.call_count == 2
    controller._wait_future = Mock(return_value=None)
    with pytest.raises(RuntimeError, match='Cannot observe the held mating transform'):
        controller._simulation_mating_poses('gear_medium', target, math.pi/2)
    assert controller._wait_future.call_count == 3
    controller._attached_model = 'gear_small'
    with pytest.raises(ValueError, match='identified attached part'):
        controller._simulation_mating_poses('gear_medium', target, math.pi/2)


@pytest.mark.parametrize('part', ['gear_small', 'gear_medium', 'gear_large'])
def test_gear_mating_rejects_fingers_aligned_with_shaft_row(part):
    controller = GazeboPickPlaceController.__new__(GazeboPickPlaceController)
    controller._attached_model = part
    controller.insertion_depth_m = .0025
    controller.frame_id = 'world'
    current = SimpleNamespace(
        position=SimpleNamespace(x=.4, y=-.5, z=1.4),
        orientation=SimpleNamespace(x=-math.sqrt(.5), y=math.sqrt(.5), z=0., w=0.),
    )
    observed_part = SimpleNamespace(
        position=SimpleNamespace(x=.4, y=-.5, z=1.177),
        orientation=SimpleNamespace(x=0., y=0., z=0., w=1.),
    )
    controller._get_ee_pose = lambda: current
    controller._get_state_client = Mock()
    controller._GetEntityState = SimpleNamespace(Request=lambda **kw: SimpleNamespace(**kw))
    controller._wait_future = Mock(return_value=SimpleNamespace(
        success=True, state=SimpleNamespace(pose=observed_part)))
    target = {'x': -.0005504, 'y': .1448242, 'z': 1.0389916}
    with pytest.raises(ValueError, match='gripper fingers do not clear'):
        controller._simulation_mating_poses(part, target, math.pi / 2)

    current.orientation = SimpleNamespace(x=0., y=1., z=0., w=0.)
    result = controller._simulation_mating_poses(part, target, math.pi / 2)
    from cais_spade_llm.recovery_framework.geometry import compose
    relative = [0., 0., .223, 0., -1., 0., 0.]
    insert = result['insert_pose']
    achieved = compose([insert[key] for key in ('x', 'y', 'z', 'qx', 'qy', 'qz', 'qw')], relative)
    assert achieved[:3] == pytest.approx(list(target.values()), abs=1e-12)
    assert achieved[3:] == pytest.approx([0., 0., math.sqrt(.5), math.sqrt(.5)], abs=1e-12)


@pytest.mark.parametrize('seated', [True, False])
def test_mating_snap_preserves_observed_phase_and_cannot_correct_unseated_part(monkeypatch, seated):
    import sys

    def vector(): return SimpleNamespace(x=0., y=0., z=0.)
    def state(): return SimpleNamespace(name='', pose=SimpleNamespace(position=vector(), orientation=SimpleNamespace(x=0.,y=0.,z=0.,w=1.)), twist=SimpleNamespace(linear=vector(),angular=vector()))
    monkeypatch.setitem(sys.modules, 'gazebo_msgs.msg', SimpleNamespace(EntityState=state))
    controller = GazeboPickPlaceController.__new__(GazeboPickPlaceController)
    controller.execution_mode = 'simulation'
    controller.frame_id = 'world'
    yaw = math.radians(103.2)
    controller._simulation_mating_context = {'model_name':'gear_small', 'target_yaw_rad':yaw,
        'axis_tolerance_m':.0005, 'target_origin_pose':{'x':0.,'y':0.,'z':1.039}, 'retain_fixture_attachment': True}
    observed = state();observed.pose.position.z = 1.039 if seated else 1.05
    observed.pose.orientation.z = math.sin(yaw/2);observed.pose.orientation.w = math.cos(yaw/2)
    controller._set_state_client = Mock()
    controller._get_state_client = Mock()
    controller._GetEntityState = SimpleNamespace(Request=lambda **kw: SimpleNamespace(**kw))
    controller._wait_future = Mock(return_value=SimpleNamespace(success=True,state=observed))
    controller._detach_part = Mock(return_value=True)
    controller._simulation_release_detach_timeout_sec = lambda: 1.
    controller._set_entity_state_for_snap = Mock(return_value=True)
    controller._attach_part_to_assembly_board = Mock(return_value=True)
    controller._detach_part_from_assembly_board = Mock(return_value=True)
    controller._verify_snapped_entity_position = Mock(return_value=True)
    controller._sync_part_collision = Mock(return_value=True)
    controller._last_command_evidence = {}
    controller._log = Mock()
    assert controller._snap_part_to_slot('gear_small',0.,0.,.02,1.029,part_origin_z=1.039,destination_location='assembly_board-v1') is seated
    if seated:
        corrected = controller._set_entity_state_for_snap.call_args.args[1]
        assert corrected.pose.orientation.z == observed.pose.orientation.z
        assert controller._last_command_evidence['seated_mating_part']['orientation_preserved'] is True
        assert controller._last_command_evidence['seated_mating_part']['fixture_attachment_retained'] is True
        controller._detach_part_from_assembly_board.assert_not_called()
    else:
        controller._set_entity_state_for_snap.assert_not_called()



@pytest.mark.parametrize('mode,seated,attached', [
    ('simulation', True, True), ('simulation', False, True),
    ('simulation', True, False), ('physical', False, True),
])
def test_verified_mating_fixture_handoff_precedes_release_settling(mode, seated, attached):
    calls = []
    controller = GazeboPickPlaceController.__new__(GazeboPickPlaceController)
    controller.execution_mode = mode
    controller._link_attacher_enabled = True
    controller._attached_model, controller._attached_link = 'gear_medium', 'wrist'
    controller.frame_id, controller.robot_model_name = 'world', 'dual_robot'
    controller.detach_timeout_sec, controller.detach_max_link_attempts = 1., 1
    controller.primary_attach_link, controller.attach_link_candidates = 'wrist', ['wrist']
    controller._simulation_mating_context = {
        'model_name': 'gear_medium', 'target_yaw_rad': math.pi / 2,
        'axis_tolerance_m': .0005, 'target_origin_pose': {'x': 0., 'y': 0., 'z': 1.039},
        'retain_fixture_attachment': True,
    }
    observed = SimpleNamespace(pose=SimpleNamespace(
        position=SimpleNamespace(x=0., y=0., z=1.039 if seated else 1.05),
        orientation=SimpleNamespace(x=0., y=0., z=math.sqrt(.5), w=math.sqrt(.5)),
    ))
    controller._get_state_client = Mock()
    controller._get_state_client.call_async.side_effect = lambda req: (
        calls.append('observe') or SimpleNamespace(success=True, state=observed))
    controller._GetEntityState = SimpleNamespace(Request=lambda **kw: SimpleNamespace(**kw))
    controller._detach_client = Mock()
    controller._detach_client.call_async.side_effect = lambda req: (
        calls.append('detach') or SimpleNamespace(success=True))
    controller._detach_srv = SimpleNamespace(Request=SimpleNamespace)
    controller._wait_future = lambda future, **kw: future
    controller._attach_part_to_assembly_board = Mock(
        side_effect=lambda *a, **kw: calls.append('fixture') or attached)
    controller._sync_part_collision = Mock(side_effect=lambda *a: calls.append('scene') or True)
    controller._last_command_evidence = {}
    success = controller._detach_part('gear_medium')
    if mode == 'physical':
        assert success and calls == ['detach', 'scene']
    elif not seated:
        assert not success and calls == ['observe']
        assert controller._attached_model == 'gear_medium'
    elif not attached:
        assert not success and calls == ['observe', 'detach', 'fixture']
    else:
        assert success and calls == ['observe', 'detach', 'fixture', 'scene']
        assert controller._last_command_evidence['seated_mating_part']['orientation_preserved']

def test_mating_corridor_rejects_wrong_tooth_phase():
    from cais_spade_llm.recovery_framework.part_collision import mating_pose_valid

    context = {'axis_tolerance_m':.0005, 'target_origin_pose':{'x':0.,'y':0.,'z':1.039},
               'start_part_z':1.08, 'target_yaw_rad':math.radians(103.2)}
    assert not mating_pose_valid(context, [0.,0.,1.04,0.,0.,0.,1.])
    yaw = context['target_yaw_rad']
    assert mating_pose_valid(context, [0.,0.,1.04,0.,0.,math.sin(yaw/2),math.cos(yaw/2)])


def test_simulation_gear_release_accepts_bounded_yaw_drift_but_rejects_wrong_phase():
    from cais_spade_llm.recovery_framework import ROOT, read_json
    from cais_spade_llm.recovery_framework.part_collision import mating_pose_valid

    geometry_path = (ROOT / 'cais_spade_llm/specification/products/geometry'
                     / 'assembly_board-v1-recovery-framework.json')
    board = read_json(geometry_path)['gazebo']['assembly_board']
    context = {**board['mating_contact_by_part']['gear_small'],
               'target_origin_pose': {'x': .0294496, 'y': .1448242, 'z': 1.0389916},
               'start_part_z': 1.0389916}
    observed = [.029534030456404785, .14462789733256984, 1.0390201506013756,
                .0001058975551361457, .0024431321180033144,
                .7078902776770403, .7063181823098459]
    assert context['yaw_tolerance_rad'] == .005
    assert mating_pose_valid(context, observed)
    wrong_yaw = context['target_yaw_rad'] + .006
    assert not mating_pose_valid(context, [*observed[:3], 0., 0.,
                                           math.sin(wrong_yaw/2), math.cos(wrong_yaw/2)])
    assert not mating_pose_valid(context, [observed[0] + .001, *observed[1:]])


def test_simulated_gear_mounting_ignores_only_configured_gear_contacts():
    from cais_spade_llm.recovery_framework.part_collision import mating_contacts_allowed

    context = {'model_name':'gear_medium', 'target_collision_object':'Gear_Plate/Gear_Shaft_2/collision',
               'max_contact_depth_m':.0005, 'ignore_tooth_contact_with':['gear_small/link/collision']}
    contact = lambda other,depth: SimpleNamespace(contact_body_1='gear_medium/link/collision',contact_body_2=other,depth=depth)
    assert mating_contacts_allowed(context, [contact('gear_small/link/collision', .001)])
    for other in ['gear_large/link/collision','ur5e_4_rg2_left_finger','Gear_Plate/Gear_Shaft_1/collision']:
        assert not mating_contacts_allowed(context, [contact(other, .0001)])
    assert not mating_contacts_allowed(context, [contact('Gear_Plate/Gear_Shaft_2/collision', .001)])


def test_home_waits_for_fresh_feedback_and_rejects_missing_feedback():
    feedback = Mock(side_effect=[([.1], []), (None, ["shoulder"])])
    controller = SimpleNamespace(arm_joint_names=["shoulder"],
        _simulation_joint_limits=lambda names: {"shoulder":{"lower":-math.pi,"upper":math.pi}},
        _get_arm_joint_positions=feedback)
    assert GazeboPickPlaceController._nearest_simulation_joint_targets(controller,[0.]) == [0.]
    feedback.assert_called_once_with(timeout_sec=2.0)
    with pytest.raises(ValueError,match="Missing observed joint position: shoulder"):
        GazeboPickPlaceController._nearest_simulation_joint_targets(controller,[0.])


@pytest.mark.parametrize("name", ["gear_small", "gear_medium", "gear_large"])
def test_mesh_support_allowance_lifts_lower_face_without_lowering_upper_face(name):
    from cais_spade_llm.recovery_framework.geometry import compose
    from cais_spade_llm.recovery_framework.part_collision import observed_part_boxes

    pose = SimpleNamespace(position=SimpleNamespace(x=0., y=0., z=1.11),
                           orientation=SimpleNamespace(x=0., y=0., z=0., w=1.))
    def bounds(allowance):
        row, = observed_part_boxes(name, pose, support_allowance=allowance)
        zs = [compose(row["pose"], [*v,0.,0.,0.,1.])[2] for v in row["mesh"]["vertices"]]
        return min(zs), max(zs)
    full = bounds(0.)
    carried = bounds(.001)
    assert carried[0] == pytest.approx(full[0] + .001, abs=1e-9)
    assert carried[1] == pytest.approx(full[1], abs=1e-9)


def test_explicit_waypoints_preserve_downward_orientation_and_seed_continuity():
    from cais_spade_llm.resources.robot.cartesian_waypoints import resolve_waypoints
    from cais_spade_llm.recovery_framework.geometry import rotate

    limits = {'joint': {'lower': -2., 'upper': 2., 'velocity': 1., 'acceleration': 2.}}
    calls = []
    def solve(pose, seed):
        assert rotate(pose[3:], [0., 0., 1.])[2] == pytest.approx(-1.)
        assert seed == ([calls[-1][0][2]] if calls else [0.])
        calls.append((pose, seed))
        return [pose[2]]
    rows = resolve_waypoints(start_pose=[0., 0., 0., 1., 0., 0., 0.], start_joints=[0.],
        waypoints=[[0., 0., .2, 1., 0., 0., 0.], [.1, 0., .2, 0., 1., 0., 0.]],
        names=['joint'], limits=limits, solve_ik=solve)
    assert rows[0]['positions'] == [0.] and rows[-1]['positions'] == [.2]
    assert all(b['time_from_start'] > a['time_from_start'] for a, b in zip(rows, rows[1:]))
    assert all(abs(row['velocities'][0]) <= 1.000001 for row in rows)
    assert all(abs(row['accelerations'][0]) <= 2.000001 for row in rows)
    assert rows[0]['velocities'] == rows[-1]['velocities'] == [0.]


def test_explicit_waypoints_reject_ik_branch_changes_and_unavailable_poses():
    from cais_spade_llm.resources.robot.cartesian_waypoints import continuous_joints, resolve_waypoints

    limits = {'joint': {'lower': -2., 'upper': 2., 'velocity': 1., 'acceleration': 2.}}
    assert continuous_joints([-2 * math.pi + .1], [0.], ['joint'], limits, .35) == pytest.approx([.1])
    with pytest.raises(ValueError, match='joint branch'):
        continuous_joints([1.], [0.], ['joint'], limits, .35)
    solve = Mock(side_effect=ValueError('unreachable waypoint'))
    with pytest.raises(ValueError, match='unreachable waypoint'):
        resolve_waypoints(start_pose=[0., 0., 0., 1., 0., 0., 0.], start_joints=[0.],
            waypoints=[[0., 0., .2, 1., 0., 0., 0.]], names=['joint'], limits=limits, solve_ik=solve)
    assert solve.call_count == 1


def test_waypoint_resource_never_calls_a_motion_planner_when_recording_is_missing(cartesian_motion_controller):
    controller, target, response = cartesian_motion_controller
    response.fraction = .8
    assert not GazeboPickPlaceController._cartesian_move(controller, target)
    controller._cart_client.call_async.assert_called_once()
    controller._execute_simulation_motion_plan.assert_not_called()
    controller._resolve_cartesian_waypoints.assert_not_called()
    assert controller._last_command_evidence['command_sent'] is False


def test_KMR_transfer_waypoints_preserve_downward_tilt_without_posture_commands():
    from cais_spade_llm.recovery_framework.kmr_motion import downward_transfer_waypoints
    from cais_spade_llm.recovery_framework.kmr_tasks import KMR_TASKS
    from cais_spade_llm.recovery_framework.geometry import rotate

    settings = {'turn_radius_m': .5, 'turn_step_rad': .15, 'minimum_turn_angle_rad': .5}
    start, target = [-.6, 0., .75, 1., 0., 0., 0.], [.6, 0., 1.15, 0., 1., 0., 0.]
    waypoints = downward_transfer_waypoints(start, target, [0., 0., .7], settings)
    assert len(waypoints) > 2
    for pose in waypoints:
        assert math.hypot(*pose[:2]) == pytest.approx(.5)
        assert pose[2] == 1.15
        assert rotate(pose[3:], [0., 0., 1.])[2] == pytest.approx(-1.)
    assert not {'move_to_configuration', 'rotate_arm_base'} & {
        step.op for task in KMR_TASKS.values() for step in task.program.steps}
    with pytest.raises(ValueError, match='downward'):
        downward_transfer_waypoints([0., 0., 1., 0., 0., 0., 1.], target, [0., 0., 0.], settings)


def test_payload_mesh_evidence_retains_provenance_without_triangle_copies():
    from cais_spade_llm.recovery_framework.part_collision import collision_geometry_evidence

    rows = [{'id': 'gear/link/collision', 'pose': [0.] * 7, 'size': [.1, .1, .02],
             'support_contact_allowance_m': .001,
             'mesh': {'vertices': [[0., 0., 0.]], 'triangles': [[0, 0, 0]],
                      'source': '/mesh.stl', 'source_sha256': 'recorded CAD hash', 'scale': [1.] * 3}}]
    result = collision_geometry_evidence(rows)
    assert result[0]['mesh']['source_sha256'] == rows[0]['mesh']['source_sha256']
    assert result[0]['mesh']['triangle_count'] == 1
    assert 'vertices' not in result[0]['mesh']
    assert 'vertices' in rows[0]['mesh']
    assert result[0]['support_contact_allowance_m'] == .001


@pytest.mark.parametrize('field,value', [
    ('linear_step', float('nan')), ('angular_step', float('inf')),
    ('maximum_joint_step', float('nan')), ('start_joints', [float('nan')]),
    ('limits', {'joint': {'lower': -2., 'upper': 2., 'velocity': 0., 'acceleration': 2.}}),
])
def test_explicit_waypoints_reject_invalid_configuration_before_ik(field, value):
    from cais_spade_llm.resources.robot.cartesian_waypoints import resolve_waypoints

    solve = Mock(return_value=[.1])
    arguments = dict(start_pose=[0., 0., 0., 1., 0., 0., 0.], start_joints=[0.],
                     waypoints=[[0., 0., .1, 1., 0., 0., 0.]], names=['joint'],
                     limits={'joint': {'lower': -2., 'upper': 2., 'velocity': 1., 'acceleration': 2.}},
                     solve_ik=solve)
    arguments[field] = value
    with pytest.raises(ValueError):
        resolve_waypoints(**arguments)
    solve.assert_not_called()


def test_explicit_waypoint_timing_bounds_controller_interpolation_between_samples():
    import numpy as np
    from cais_spade_llm.resources.robot.cartesian_waypoints import resolve_waypoints

    limits = {'joint': {'lower': -2., 'upper': 2., 'velocity': .4, 'acceleration': .6}}
    rows = resolve_waypoints(start_pose=[0., 0., 0., 1., 0., 0., 0.], start_joints=[0.],
        waypoints=[[.4, 0., 0., 1., 0., 0., 0.]], names=['joint'], limits=limits,
        linear_step=.04, solve_ik=lambda pose, seed: [math.sin(4 * pose[0])])
    # Solve the six boundary equations independently of the runtime coefficient helper.
    for left, right in zip(rows, rows[1:]):
        dt = right['time_from_start'] - left['time_from_start']
        matrix = np.array([[1.,0,0,0,0,0], [0,1.,0,0,0,0], [0,0,2.,0,0,0],
                           [1.,1,1,1,1,1], [0,1.,2,3,4,5], [0,0,2.,6,12,20]])
        boundary = [left['positions'][0], left['velocities'][0] * dt,
                    left['accelerations'][0] * dt**2, right['positions'][0],
                    right['velocities'][0] * dt, right['accelerations'][0] * dt**2]
        curve = np.polynomial.Polynomial(np.linalg.solve(matrix, boundary))
        samples = np.linspace(0., 1., 1001)
        assert np.max(np.abs(curve.deriv()(samples))) / dt <= .400001
        assert np.max(np.abs(curve.deriv(2)(samples))) / dt**2 <= .600001


def test_collision_samples_include_quintic_excursion_between_equal_positions():
    from cais_spade_llm.resources.robot.cartesian_waypoints import trajectory_samples

    def point(velocity, seconds):
        return SimpleNamespace(positions=[0.], velocities=[velocity], accelerations=[0.],
                               time_from_start=SimpleNamespace(sec=seconds, nanosec=0))
    trajectory = SimpleNamespace(joint_names=['joint'], points=[point(1., 0), point(-1., 2)])
    samples = list(trajectory_samples(trajectory, maximum_joint_step=.02))
    assert samples[0] == pytest.approx([0.]) and samples[-1] == pytest.approx([0.])
    assert max(row[0] for row in samples) > .6
    assert max(abs(b[0] - a[0]) for a, b in zip(samples, samples[1:])) <= .020001


def test_waypoint_cancellation_never_dispatches_or_calls_a_planner(cartesian_motion_controller):
    controller, target, _ = cartesian_motion_controller
    controller._shutdown_requested = True
    assert not GazeboPickPlaceController._cartesian_move(controller, target)
    controller._cart_client.call_async.assert_not_called()
    controller._send_simulation_joint_trajectory.assert_not_called()
    controller._resolve_cartesian_waypoints.assert_not_called()
    assert controller._last_command_evidence['command_sent'] is False


@pytest.fixture
def saved_waypoints_recording(tmp_path, monkeypatch):
    import json
    from cais_spade_llm.resources.robot import saved_waypoints

    names = ['joint']
    limits = {'joint': dict(lower=-2., upper=2., velocity=1., acceleration=2.)}
    start, target = [0., 0., 1., 0., 1., 0., 0.], [.1, 0., 1., 0., 1., 0., 0.]
    points = [dict(positions=[q], velocities=[0.], accelerations=[0.], time_from_start=t)
              for q, t in ((0., 0.), (.2, 1.))]
    resource = dict(joint_names=names, limits=limits, frame_id='world', routes=[
        dict(id='pick_approach', start_pose=start, target_pose=target, points=points)])
    payload = dict(version=1, execution_mode='simulation', source_fingerprints={},
                   resources={name: resource for name in ('KMR','ur5e-1','ur5e-2','ur5e-3','ur5e-4')})
    path = tmp_path/'waypoints.json'
    path.write_text(json.dumps(payload))
    saved_waypoints._read_saved.cache_clear()
    monkeypatch.setattr(saved_waypoints, 'source_fingerprints', lambda: {})
    command = Mock(side_effect=lambda names, points: (names, points))
    monkeypatch.setattr(saved_waypoints, 'robot_trajectory', command)
    return dict(settings={'saved_waypoints_file': str(path)}, resource_id='ur5e-4',
                names=names, limits=limits, start_pose=start, start_joints=[0.],
                target_pose=target), payload, path, command


@pytest.mark.parametrize('resource_id', ['KMR','ur5e-1','ur5e-2','ur5e-3','ur5e-4'])
def test_all_robots_replay_saved_joint_positions_and_timing(saved_waypoints_recording, resource_id, monkeypatch):
    from cais_spade_llm.resources.robot import saved_waypoints, cartesian_waypoints

    args, payload, _, command = saved_waypoints_recording
    calculate = Mock(side_effect=AssertionError('Runtime IK/timing is forbidden'))
    monkeypatch.setattr(cartesian_waypoints, 'resolve_waypoints', calculate)
    monkeypatch.setattr(cartesian_waypoints, '_segment_timing', calculate)
    args['resource_id'] = resource_id
    motion, evidence = saved_waypoints.saved_motion(**args)
    assert motion[1] == payload['resources'][resource_id]['routes'][0]['points']
    assert evidence['runtime_ik'] is False
    assert evidence['runtime_time_parameterization'] is False
    assert evidence['saved_waypoints_id'] == 'pick_approach'
    calculate.assert_not_called()
    command.assert_called_once()


@pytest.mark.parametrize('change', ['hardware','start','target','joints','limits','frame','missing','stale','timing','incomplete'])
def test_saved_waypoints_reject_incompatible_execution(saved_waypoints_recording, change):
    import json
    from cais_spade_llm.resources.robot.saved_waypoints import saved_motion

    args, payload, path, command = saved_waypoints_recording
    if change == 'hardware':
        args['execution_mode'] = 'physical'
    elif change == 'start':
        args['start_joints'] = [.1]
    elif change == 'target':
        args['target_pose'] = [.2, 0., 1., 0., 1., 0., 0.]
    elif change == 'joints':
        args['names'] = ['another_robot_joint']
    elif change == 'limits':
        args['limits'] = {'joint': dict(lower=-2., upper=2., velocity=.5, acceleration=2.)}
    elif change == 'frame':
        args['frame_id'] = 'another_frame'
    elif change == 'missing':
        path.unlink()
    elif change == 'stale':
        payload['source_fingerprints'] = {'changed': 'source'}
        path.write_text(json.dumps(payload))
    elif change == 'timing':
        payload['resources']['ur5e-4']['routes'][0]['points'][-1]['time_from_start'] = .01
        path.write_text(json.dumps(payload))
    elif change == 'incomplete':
        payload['failures'] = [{'resource_id': 'KMR', 'reason': 'unreachable'}]
        path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        saved_motion(**args)
    command.assert_not_called()


def test_saved_waypoints_for_every_configured_robot_and_home_policy():
    import json
    from cais_spade_llm.recovery_framework import SCENE_PATH

    scene = json.loads(SCENE_PATH.read_text())
    for robot in scene['robots']:
        assert robot['cartesian_motion']['only'] is True
        assert robot['cartesian_motion']['resource_id'] == robot['resource_id']
        assert 'saved_waypoints_file' not in robot['cartesian_motion']
        assert robot['cartesian_motion']['avoid_collisions'] is False
        assert len(robot['initial_joint_positions']) == 6
    assert scene['KMR']['task_execution']['cartesian_motion_only'] is True
    assert 'saved_waypoints_file' not in scene['KMR']['task_execution']['cartesian_waypoints']
    assert scene['KMR']['task_execution']['avoid_collisions'] is False


def test_prepared_recording_covers_configured_parts_and_observed_home_joints():
    import hashlib
    import json
    from cais_spade_llm.recovery_framework import ROOT, SCENE_PATH
    from cais_spade_llm.recovery_framework.kmr_gazebo import storage_home
    from cais_spade_llm.recovery_framework.workflow_execution import _robot_configuration
    from cais_spade_llm.resources.robot.saved_waypoints import SAVED_WAYPOINTS_FILE, source_fingerprints

    scene = json.loads(SCENE_PATH.read_text())
    recording = json.loads((ROOT / SAVED_WAYPOINTS_FILE).read_text())
    assert recording['failures'] == []
    # This superseded recording remains offline evidence, not a production dependency.
    archived_scene = ROOT / 'cais_spade_llm/monitor/recovery_gazebo_runs/recorded_two_part_validation/saved_waypoints_scene.json'
    assert recording['source_fingerprints'][str(SCENE_PATH.relative_to(ROOT))] == hashlib.sha256(archived_scene.read_bytes()).hexdigest()
    assert set(recording['resources']) == {'KMR', 'ur5e-1', 'ur5e-2', 'ur5e-3', 'ur5e-4'}
    for robot in scene['robots']:
        rid = robot['resource_id']
        _, named, _ = _robot_configuration(scene, robot)
        routes = recording['resources'][rid]['routes']
        returns = [row for row in routes if row['id'].endswith('/move_home')]
        assert returns
        assert all(row['points'][-1]['positions'] == named['home'] for row in returns)
        parts = ({'ur5e-3': list(scene['Storage']['slots']),
                  'ur5e-4': scene['3D Printing Station']['initial_products']}.get(rid)
                 or next(row['nominal_parts'] for row in scene['machines'] if row['handling_robot'] == rid))
        assert set(parts) <= {row['id'].split('/')[0] for row in returns}
    assert any(row['id'] == 'RGOCG4-50_Round_4mm/move_home'
               for row in recording['resources']['ur5e-1']['routes'])
    kmr = recording['resources']['KMR']['routes']
    for part in scene['Storage']['slots']:
        home = storage_home(scene, {'parts': [part]})
        assert any('/move_home' in row['id'] and row['points'][-1]['positions'] == home['joints']
                   for row in kmr)


def test_prepared_KMR_placement_preserves_configured_part_orientation():
    import json
    from cais_spade_llm.recovery_framework import ROOT, SCENE_PATH
    from cais_spade_llm.recovery_framework.geometry import compose, quaternion
    from cais_spade_llm.recovery_framework.kmr_gazebo import inverse
    from cais_spade_llm.resources.robot.saved_waypoints import SAVED_WAYPOINTS_FILE, poses_match

    scene = json.loads(SCENE_PATH.read_text())
    recording = json.loads((ROOT / SAVED_WAYPOINTS_FILE).read_text())
    config = scene['KMR']['task_execution']
    for part, slot in scene['Storage']['slots'].items():
        grasp = [*slot[:2], slot[2] + config['grasp_height_m'], *config['pick_orientation_xyzw']]
        held = compose(inverse(grasp), [*slot[:3], *quaternion(slot[3:])])
        for machine in scene['machines']:
            dock = machine['KMR_docking_pose']
            fixture = machine['workholding_pose']
            target = compose([*fixture[:3], *quaternion(fixture[3:])], inverse(held))
            target[2] += .003
            target = compose(inverse([*dock[:3], *quaternion(dock[3:])]), target)
            matches = [row for row in recording['resources']['KMR']['routes']
                       if row['id'].startswith(f'{part}/{machine["resource_id"]}/place_descend/')]
            assert matches
            assert all(poses_match(row['target_pose'], target) for row in matches)


@pytest.mark.parametrize('outcome', ['success', 'collision', 'moved', 'stop'])
def test_saved_waypoint_dispatch_preserves_validation_and_stop(outcome, cartesian_motion_controller):
    controller, target, response = cartesian_motion_controller
    solution = response.solution
    controller._fresh_stable_joint_target.return_value = outcome != 'moved'
    def validate(trajectory):
        assert trajectory is solution.joint_trajectory
        controller._shutdown_requested = outcome == 'stop'
        return outcome != 'collision'
    controller._simulation_trajectory_is_collision_free = Mock(side_effect=validate)
    assert GazeboPickPlaceController._cartesian_move(controller, target) is (outcome == 'success')
    controller._resolve_cartesian_waypoints.assert_not_called()
    controller._cart_client.call_async.assert_called_once()
    controller._scale_trajectory_timing.assert_called_once_with(solution, 1.)
    if outcome == 'success':
        controller._send_simulation_joint_trajectory.assert_called_once()
        assert controller._last_command_evidence['dynamic_target'] is True
    else:
        controller._send_simulation_joint_trajectory.assert_not_called()
        assert controller._last_command_evidence['command_sent'] is False


@pytest.fixture
def cartesian_motion_controller():
    pytest.importorskip('moveit_msgs.msg')
    from builtin_interfaces.msg import Time
    from geometry_msgs.msg import Pose
    from moveit_msgs.srv import GetCartesianPath
    from cais_spade_llm.resources.robot.cartesian_waypoints import robot_trajectory

    rows = [dict(positions=[q], velocities=[0.], accelerations=[0.], time_from_start=t)
            for q, t in ((0., 0.), (.2, 1.))]
    solution = robot_trajectory(['joint'], rows)
    response = SimpleNamespace(solution=solution, fraction=1., error_code=SimpleNamespace(val=1))
    target = Pose()
    target.position.x, target.position.z, target.orientation.w = .2, 1., 1.
    controller = SimpleNamespace(
        execution_mode='simulation', controller_config={'cartesian_motion': {
            'only': True, 'max_joint_step_rad': .35, 'linear_step_m': .01}},
        _shutdown_requested=False, _planning_wall_time_sec=0., _trajectory_duration_sec=0.,
        _resolve_cartesian_waypoints=Mock(), _load_saved_cartesian_waypoints=Mock(),
        _cart_client=Mock(), _ExecuteTrajectory=SimpleNamespace(Goal=SimpleNamespace),
        trajectory_time_scale=1., _scale_trajectory_timing=Mock(), arm_trajectory_topic='/arm/joint_trajectory',
        _fresh_stable_joint_target=Mock(return_value=True),
        _send_simulation_joint_trajectory=Mock(return_value=True),
        _wait_for_simulation_cartesian_endpoint=Mock(return_value={'within_tolerance': True}),
        _get_arm_joint_positions=Mock(return_value=([0.], [])), arm_joint_names=['joint'],
        frame_id='world', group_name='arm', ee_link='tool0', _GetCartesianPath=GetCartesianPath,
        _node=SimpleNamespace(get_clock=lambda: SimpleNamespace(now=lambda: SimpleNamespace(to_msg=Time))),
        _wait_future=lambda future, **kwargs: future, _log=lambda: Mock(),
        _simulation_trajectory_is_collision_free=Mock(return_value=True),
        _execute_simulation_motion_plan=Mock(),
    )
    controller._cart_client.call_async.return_value = response
    return controller, target, response


@pytest.mark.parametrize('resource_id', ['ur5e-1', 'ur5e-2', 'ur5e-3', 'ur5e-4'])
def test_each_UR_uses_changed_target_without_recording_or_per_sample_ik(resource_id, cartesian_motion_controller):
    from copy import deepcopy
    from cais_spade_llm.recovery_framework import SCENE_PATH, read_json

    controller, target, _ = cartesian_motion_controller
    settings = next(robot['cartesian_motion'] for robot in read_json(SCENE_PATH)['robots']
                    if robot['resource_id'] == resource_id)
    controller.controller_config['cartesian_motion'] = settings
    controller._state_validity_client = Mock(side_effect=AssertionError('Collision checks disabled'))
    controller._simulation_trajectory_is_collision_free = lambda trajectory: GazeboPickPlaceController._simulation_trajectory_is_collision_free(controller, trajectory)
    original = deepcopy(target)
    assert GazeboPickPlaceController._cartesian_move(controller, original)
    target.position.x += .04
    assert GazeboPickPlaceController._cartesian_move(controller, target)
    queries = [call.args[0] for call in controller._cart_client.call_async.call_args_list]
    assert [query.waypoints[-1].position.x for query in queries] == pytest.approx([.2, .24])
    assert all(list(query.start_state.joint_state.position) == [0.] for query in queries)
    assert all(query.avoid_collisions is False for query in queries)
    controller._resolve_cartesian_waypoints.assert_not_called()
    controller._load_saved_cartesian_waypoints.assert_not_called()
    controller._execute_simulation_motion_plan.assert_not_called()
    assert controller._last_command_evidence['motion_path_validation'] == {
        'checked_states': 0, 'collision_checks_bypassed': True}


@pytest.mark.parametrize('failure', ['partial', 'nan_fraction', 'nan_target', 'wrong_joints', 'jump', 'limits', 'no_start'])
def test_dynamic_waypoint_failure_has_no_partial_dispatch(failure, cartesian_motion_controller):
    controller, target, response = cartesian_motion_controller
    if failure == 'partial':
        response.fraction = .5
    elif failure == 'nan_fraction':
        response.fraction = math.nan
    elif failure == 'nan_target':
        target.position.x = math.nan
    elif failure == 'wrong_joints':
        response.solution.joint_trajectory.joint_names = ['another_robot_joint']
    elif failure == 'jump':
        response.solution.joint_trajectory.points[-1].positions = [1.]
    elif failure == 'limits':
        controller._scale_trajectory_timing.side_effect = ValueError('joint limit')
    elif failure == 'no_start':
        controller._get_arm_joint_positions.return_value = None, ['joint']
    assert not GazeboPickPlaceController._cartesian_move(controller, target, allow_partial=True)
    controller._send_simulation_joint_trajectory.assert_not_called()
    controller._execute_simulation_motion_plan.assert_not_called()


@pytest.mark.parametrize('feedback', ['fresh', 'missing', 'moved', 'stop'])
def test_cartesian_dispatch_refreshes_feedback_after_preparation(feedback, cartesian_motion_controller):
    controller, target, _ = cartesian_motion_controller
    controller._fresh_stable_joint_target.return_value = False
    reads = 0

    def observe(**_kwargs):
        nonlocal reads
        reads += 1
        if reads == 2:
            controller._fresh_stable_joint_target.return_value = feedback == 'fresh'
            controller._shutdown_requested = feedback == 'stop'
            if feedback == 'missing':
                return None, ['joint']
        return [1. if reads == 2 and feedback == 'moved' else 0.], []

    controller._get_arm_joint_positions.side_effect = observe
    assert GazeboPickPlaceController._cartesian_move(controller, target) is (feedback == 'fresh')
    assert reads == 2
    assert controller._send_simulation_joint_trajectory.called is (feedback == 'fresh')
    controller._execute_simulation_motion_plan.assert_not_called()


@pytest.mark.parametrize('mode', ['simulation', 'physical'])
def test_service_wait_stop_cancels_simulation_without_changing_physical_wait(mode, monkeypatch):
    from cais_spade_llm.resources.robot import gazebo_pick_place_controller as controller_module

    future = Mock()
    future.done.return_value = False
    future.result.return_value = 'observed'
    controller = SimpleNamespace(
        execution_mode=mode, _shutdown_requested=False,
        _rclpy=SimpleNamespace(ok=lambda: True), _log=lambda: Mock(),
    )
    waits = []

    def poll(_seconds):
        waits.append(_seconds)
        controller._shutdown_requested = True
        if len(waits) == 2:
            future.done.return_value = True

    event = Mock(wait=poll)
    monkeypatch.setattr(controller_module.threading, 'Event', lambda: event)
    result = GazeboPickPlaceController._wait_future(controller, future, 5., 'observe Gazebo wrist pose')
    assert result == ('observed' if mode == 'physical' else None)
    assert len(waits) == (2 if mode == 'physical' else 1)
    assert future.cancel.called is (mode == 'simulation')


def test_cartesian_route_preserves_missing_observation_reason():
    controller = SimpleNamespace(
        execution_mode='simulation', controller_config={'cartesian_motion': {'only': True}},
        _get_ee_pose=lambda: None, _cartesian_move=Mock(),
        _get_arm_joint_positions=lambda **kwargs: (None, ['joint']),
        _last_failure_message='Gazebo link observation timed out: dual_robot::ur5e_2_wrist_3_link',
    )
    assert not GazeboPickPlaceController._move_xy_direct(controller, 0., 0., 1., object(), 'home')
    assert 'dual_robot::ur5e_2_wrist_3_link' in controller._last_failure_message
    controller._cartesian_move.assert_not_called()


@pytest.mark.parametrize('stable_joints,fresh_tf,accepted', [
    (True, True, True), (False, True, False), (True, False, False),
])
def test_cartesian_route_start_uses_fresh_joints_and_tf_after_gazebo_timeout(
    stable_joints, fresh_tf, accepted,
):
    orientation = SimpleNamespace(x=0., y=1., z=0., w=0.)
    observed = SimpleNamespace(
        position=SimpleNamespace(x=0., y=0., z=1.), orientation=orientation)
    controller = SimpleNamespace(
        execution_mode='simulation', robot_name='ur5e-1', ee_link='ur5e_1_tool0',
        controller_config={'cartesian_motion': {'only': True}},
        _get_ee_pose=lambda: None,
        _get_arm_joint_positions=lambda **kwargs: ([0.] * 6, []),
        arm_joint_names=[f'joint_{i}' for i in range(6)],
        _fresh_stable_joint_target=Mock(return_value=stable_joints),
        _fresh_simulation_tf_tool_pose=Mock(return_value=observed if fresh_tf else None),
        _cartesian_move=Mock(return_value=True),
        _make_pose=lambda x, y, z, q: (x, y, z, q),
        _last_failure_message='Gazebo link observation timed out: dual_robot::ur5e_1_wrist_3_link',
        _last_command_evidence={},
    )
    result = GazeboPickPlaceController._move_xy_direct(
        controller, .1, 0., 1., orientation, 'Cartesian home')
    assert result is accepted
    if accepted:
        assert controller._last_pose_observation['source'] == 'fresh_joint_tf'
        assert controller._last_command_evidence['observed_start_pose'] == [0., 0., 1., 0., 1., 0., 0.]
    else:
        controller._cartesian_move.assert_not_called()
    assert controller._fresh_simulation_tf_tool_pose.called is stable_joints


def test_direct_motion_failure_preserves_controller_rejection():
    controller = SimpleNamespace(
        wait_for_services=lambda: True, _make_pose=lambda *args: object(),
        _cartesian_move=Mock(return_value=False), trajectory_time_scale=1.,
        _last_failure_message='Invalid Cartesian trajectory: joint limit',
    )
    controller._unavailable_message = lambda default: GazeboPickPlaceController._unavailable_message(controller, default)
    result = GazeboPickPlaceController._move_pose_direct(controller, 0., 0., 1., orientation=object())
    assert result['success'] is False
    assert 'Invalid Cartesian trajectory: joint limit' in result['message']


@pytest.mark.parametrize('task,step,key', [
    ('pick_approach', 'move_above_part', 'pick_transit_waypoints'),
    ('place_approach', 'move_above_destination', 'transit_waypoints'),
])
def test_configured_transit_waypoints_keep_the_final_computed_target(task, step, key):
    points = [[.1, .4, 1.5], [.2, .4, 1.5]]
    orientation = SimpleNamespace(x=1., y=0., z=0., w=0.)
    controller = SimpleNamespace(
        execution_mode='simulation', controller_config={'cartesian_motion': {
            'only': True, key: [points] if key == 'transit_waypoints' else points}},
        _robot_task_step=(task, step), _cartesian_move=Mock(return_value=True),
        _make_pose=lambda x, y, z, q: (x, y, z, q),
        _get_ee_pose=lambda: SimpleNamespace(
            position=SimpleNamespace(x=0., y=.4, z=1.5), orientation=orientation),
    )
    assert GazeboPickPlaceController._move_xy_direct(controller, .3, .45, 1.48, orientation, 'move_cartesian')
    call = controller._cartesian_move.call_args
    assert call.args[0] == (.3, .45, 1.48, orientation)
    assert call.kwargs['waypoints'] == [(*point, orientation) for point in points]


@pytest.mark.parametrize('yaw', [.4, -1.7, math.pi])
def test_cartesian_translation_holds_orientation_and_turns_at_saved_clearance(yaw):
    from cais_spade_llm.resources.robot.cartesian_waypoints import fixed_orientation_waypoints

    start = [0., 0., 1., 1., 0., 0., 0.]
    target = [.4, .2, 1.1, math.cos(yaw / 2), math.sin(yaw / 2), 0., 0.]
    rotation = [.2, 0., 1.3]
    poses = fixed_orientation_waypoints(
        start, target, [[0., 0., 1.3], rotation, rotation, [.4, .2, 1.3]],
        rotation_point=rotation, angular_step=.05,
    )
    assert poses[-1] == target
    for a, b in zip([start, *poses], poses):
        distance = math.dist(a[:3], b[:3])
        dot = abs(sum(x * y for x, y in zip(a[3:], b[3:])))
        angle = 2 * math.acos(min(1., dot))
        assert distance > 1e-9 or angle > 1e-9
        if distance > 1e-9:
            assert angle < 1e-7
        else:
            assert a[:3] == b[:3] == rotation
            assert angle <= .05 + 1e-7



@pytest.mark.parametrize('feedback', ['settles', 'stale', 'stopped'])
def test_cartesian_home_waits_for_observed_stability_without_repeating_motion(feedback, monkeypatch):
    pytest.importorskip('moveit_msgs.srv')
    pose = SimpleNamespace(position=SimpleNamespace(x=.2, y=.1, z=1.5), orientation=object())
    controller = SimpleNamespace(
        _home_fk_client=Mock(), arm_joint_names=['joint'], ee_link='tool0', frame_id='world',
        _wait_future=lambda *args, **kwargs: SimpleNamespace(
            error_code=SimpleNamespace(val=1), pose_stamped=[SimpleNamespace(pose=pose)]),
        _move_xy_direct=Mock(return_value=True),
        _get_arm_joint_positions=Mock(return_value=([.5], [])),
        _fresh_stable_joint_target=Mock(side_effect=[False, feedback == 'settles']),
    )
    polls = iter([True, True, False] if feedback != 'stopped' else [False])
    controller._motion_pending=lambda duration: lambda: next(polls)
    monkeypatch.setattr('cais_spade_llm.resources.robot.gazebo_pick_place_controller.time.sleep', lambda _: None)
    assert GazeboPickPlaceController._move_configuration_cartesian(controller, [.5]) is (feedback == 'settles')
    controller._move_xy_direct.assert_called_once()
    assert controller._get_arm_joint_positions.call_count == (0 if feedback == 'stopped' else 2)


def test_return_rotates_at_saved_endpoint_without_repeating_the_clearance_route():
    from cais_spade_llm.resources.robot.cartesian_waypoints import fixed_orientation_waypoints

    start = [1., 0., 1.2, 0., 1., 0., 0.]
    target = [0., 0., 1.1, 1., 0., 0., 0.]
    clearance = [[1., 1., 1.2], [0., 1., 1.2]]
    poses = fixed_orientation_waypoints(
        start, target, clearance, rotation_point=target[:3], angular_step=.05,
    )
    translations = [b[:3] for a, b in zip([start, *poses], poses)
                    if math.dist(a[:3], b[:3]) > 1e-9]
    assert translations == [*clearance, target[:3]]
    assert poses[0][3:] == start[3:]
    assert poses[-1] == target

def test_cartesian_unchanged_endpoint_has_no_duplicate_motion():
    from cais_spade_llm.resources.robot.cartesian_waypoints import fixed_orientation_waypoints

    pose = [0., 0., 1., 1., 0., 0., 0.]
    assert fixed_orientation_waypoints(
        pose, pose, [pose[:3], pose[:3]], rotation_point=pose[:3], angular_step=.05,
    ) == []



def test_simulation_collision_bypass_does_not_disable_hardware_request_checks(cartesian_motion_controller):
    controller, target, response = cartesian_motion_controller
    controller.execution_mode = 'physical'
    controller.controller_config['cartesian_motion']['avoid_collisions'] = False
    controller._consume_prepared_cartesian = Mock(return_value=(None, 'not eligible'))
    response.fraction = .5
    assert not GazeboPickPlaceController._cartesian_move(controller, target)
    assert controller._cart_client.call_async.call_args.args[0].avoid_collisions is True
    controller._send_simulation_joint_trajectory.assert_not_called()


def test_equivalent_KMR_grasps_keep_the_observed_downward_orientation():
    from cais_spade_llm.recovery_framework.kmr_gazebo import _nearest_pick_orientation

    choices = [[1., 0., 0., 0.], [0., 1., 0., 0.]]
    retained = [0., 1., 0., 0.]
    assert _nearest_pick_orientation(retained, choices, .02) == retained
    assert _nearest_pick_orientation([1., 0., 0., 0.], choices, .02) == choices[0]
    with pytest.raises(ValueError, match='unit grasp orientations'):
        _nearest_pick_orientation(retained, [[0., 0., 0., 0.]], .02)


def test_ur1_ur2_direct_home_and_next_pick_skip_subtolerance_turn():
    observed = SimpleNamespace(x=math.cos(.0012), y=math.sin(.0012), z=0., w=0.)
    nominal = SimpleNamespace(x=1., y=0., z=0., w=0.)
    for step, label in ((None, 'Cartesian home'), (('pick_approach', 'move_above_part'), 'pick'),
                        (('pick_approach', 'descend'), 'pick')):
        controller = SimpleNamespace(
            execution_mode='simulation', robot_name='ur5e-1',
            controller_config={'cartesian_motion': {'only': True,
                'transit_waypoints': [[[2., 0., 1.], [2., 1., 1.]]]}},
            _robot_task_step=step, _cartesian_move=Mock(return_value=True),
            _make_pose=lambda x, y, z, q: (x, y, z, q),
            _get_ee_pose=lambda: SimpleNamespace(
                position=SimpleNamespace(x=0., y=0., z=1.), orientation=observed),
        )
        assert GazeboPickPlaceController._move_xy_direct(
            controller, 1., 0., 1., nominal, label)
        target = controller._cartesian_move.call_args.args[0]
        assert target == (1., 0., 1., observed)
        assert controller._cartesian_move.call_args.kwargs['waypoints'] == []


@pytest.mark.parametrize('robot,x', [('ur5e-1', -6.08), ('ur5e-2', -2.68)])
def test_machine_pick_retreat_lifts_before_withdrawing_to_saved_clearance(robot, x):
    from cais_spade_llm.recovery_framework import SCENE_PATH, read_json

    scene = read_json(SCENE_PATH)
    machine = next(item for item in scene['machines'] if item['handling_robot'] == robot)
    settings = next(item['cartesian_motion'] for item in scene['robots']
                    if item['resource_id'] == robot)
    controller = GazeboPickPlaceController.__new__(GazeboPickPlaceController)
    controller.execution_mode = 'simulation'
    controller.robot_name = robot
    controller.controller_config = {'cartesian_motion': settings}
    controller.pick_tcp_z_bias_min_m = .003
    controller.pick_tcp_z_bias_max_m = .02
    controller.min_pick_tcp_z_m = 0.
    controller.pick_z_adjustments_m = {}
    controller.pick_tool0_z_adjustment_m = 0.
    controller.approach_height_m = .2
    controller.gripper_open = .11
    controller.gripper_close = .02
    controller.init = lambda: True
    controller._get_ee_pose = lambda: SimpleNamespace(
        position=SimpleNamespace(x=x-.218, y=.699, z=1.278),
        orientation=SimpleNamespace(x=0., y=1., z=0., w=0.))
    controller._make_pose = lambda xx, yy, zz, q: (xx, yy, zz, q)
    controller._log = lambda: Mock()
    result = controller.compute_pick_targets(
        part_name='KET4_Square_4mm',
        product_geometry={'model_name': 'KET4_Square_4mm', 'part_height_m': .05,
                          'handling_robot_access': machine['handling_robot_access']},
        detected_parts=[{'part_name': 'KET4_Square_4mm', 'model_name': 'KET4_Square_4mm',
                         'x': x, 'y': 1.82, 'z': 1.06}],
    )
    assert result['success'] is True
    assert result['approach_pose']['z'] == pytest.approx(1.303)
    assert result['target_pose']['z'] == pytest.approx(1.303)
    assert result['access_retreat_pose']['z'] == pytest.approx(1.35)
    assert result['access_retreat_pose']['y'] == pytest.approx(1.64)

    observed = SimpleNamespace(x=0., y=-1., z=.0011, w=0.)
    nominal = SimpleNamespace(x=0., y=1., z=0., w=0.)
    controller._robot_task_step = ('pick_grasp', 'retreat_from_source')
    controller._get_ee_pose = lambda: SimpleNamespace(
        position=SimpleNamespace(x=x, y=1.82, z=1.303), orientation=observed)
    controller._cartesian_move = Mock(return_value=True)
    assert GazeboPickPlaceController._move_xy_direct(
        controller, x, 1.64, 1.35, nominal, 'retreat')
    call = controller._cartesian_move.call_args
    assert call.kwargs['waypoints'] == [(x, 1.82, 1.35, observed)]
    assert call.args[0] == (x, 1.64, 1.35, observed)


def test_machine_pick_retreat_keeps_required_large_turn_at_saved_clearance():
    observed = SimpleNamespace(x=0., y=1., z=0., w=0.)
    target = SimpleNamespace(x=-math.sin(.05), y=math.cos(.05), z=0., w=0.)
    controller = SimpleNamespace(
        execution_mode='simulation', robot_name='ur5e-2',
        controller_config={'cartesian_motion': {'only': True,
            'rotation_waypoint': [-3.15, .7, 1.35], 'angular_step_rad': .05}},
        _robot_task_step=('pick_grasp', 'retreat_from_source'),
        _cartesian_move=Mock(return_value=True),
        _make_pose=lambda x, y, z, q: (x, y, z, q),
        _get_ee_pose=lambda: SimpleNamespace(
            position=SimpleNamespace(x=-2.68, y=1.82, z=1.303), orientation=observed),
    )
    assert GazeboPickPlaceController._move_xy_direct(
        controller, -2.68, 1.64, 1.35, target, 'retreat')
    waypoints = controller._cartesian_move.call_args.kwargs['waypoints']
    assert waypoints[0][:3] == (-2.68, 1.82, 1.35)
    assert any(point[:3] == (-3.15, .7, 1.35) for point in waypoints)


def test_simulation_endpoint_uses_fresh_joint_tf_only_after_gazebo_observation_fails(
    cartesian_motion_controller,
):
    controller, target, _ = cartesian_motion_controller
    controller._last_simulation_controller_succeeded = True
    controller.cartesian_position_tolerance_m = .005
    controller.cartesian_orientation_tolerance_rad = .02
    controller._wait_for_simulation_cartesian_endpoint = Mock(return_value=None)
    controller._fresh_simulation_tf_tool_pose = Mock(return_value=target)
    controller._observed_cartesian_endpoint_evidence = lambda pose, observed=None: (
        GazeboPickPlaceController._observed_cartesian_endpoint_evidence(
            controller, pose, observed=observed))
    assert GazeboPickPlaceController._cartesian_move(controller, target)
    assert controller._last_command_evidence['controller_endpoint_observed'][
        'observation_source'] == 'fresh_joint_tf'
    controller._fresh_simulation_tf_tool_pose.return_value = None
    assert not GazeboPickPlaceController._cartesian_move(controller, target)


@pytest.mark.parametrize('mode,age,accepted', [
    ('simulation', .1, True), ('simulation', .5, False), ('physical', .1, False),
])
def test_fresh_tool_tf_endpoint_rejects_stale_feedback_and_hardware(mode, age, accepted):
    transform = SimpleNamespace(
        header=SimpleNamespace(stamp=SimpleNamespace(sec=100, nanosec=0)),
        transform=SimpleNamespace(
            translation=SimpleNamespace(x=.2, y=.3, z=1.4),
            rotation=SimpleNamespace(x=0., y=1., z=0., w=0.)),
    )
    controller = SimpleNamespace(
        execution_mode=mode, _shutdown_requested=False,
        _tf_buffer=SimpleNamespace(lookup_transform=Mock(return_value=transform)),
        _rclpy=SimpleNamespace(time=SimpleNamespace(Time=lambda: None)),
        _tf2_ros=SimpleNamespace(LookupException=RuntimeError,
                                 ConnectivityException=ValueError,
                                 ExtrapolationException=KeyError),
        frame_id='world', ee_link='tool0',
        _node=SimpleNamespace(get_clock=lambda: SimpleNamespace(
            now=lambda: SimpleNamespace(nanoseconds=round((100+age)*1e9)))),
        _Pose=lambda: SimpleNamespace(position=SimpleNamespace(),orientation=None),
    )
    pose = GazeboPickPlaceController._fresh_simulation_tf_tool_pose(controller)
    assert (pose is not None) is accepted
    if pose is not None:
        assert pose.position.z == 1.4 and pose.orientation.y == 1.
