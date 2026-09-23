"""Cancellable ROS worker for observed KMR Storage-to-M1 task execution."""

from __future__ import annotations

import json
import os
import hashlib
import logging
import math
import signal
import sys
import time
from collections import deque
from copy import deepcopy
from pathlib import Path
from uuid import uuid4

from cais_spade_llm.recovery_framework import ROOT, SCENE_PATH, fingerprint
from cais_spade_llm.recovery_framework.geometry import collision_boxes, compose, quaternion, rotate
from cais_spade_llm.recovery_framework.kmr_tasks import capability_decompositions, delivery_bindings, execute_composition
from cais_spade_llm.recovery_framework.kmr_motion import (
    bounded_joints, joint_limits, retime_trajectory, trajectory_cost,
)
from cais_spade_llm.resources.robot.simulation_timing import MotionDeadline

_log = logging.getLogger(__name__)
_SCENE_IDENTITY_PARAMETERS = (
    'launch_id', 'scene_fingerprint', 'scene_asset_fingerprint', 'performance_settings',
)


def _scene_identity(values: list[str]) -> dict[str, str]:
    valid_settings = False
    if len(values) == len(_SCENE_IDENTITY_PARAMETERS):
        from cais_spade_llm.recovery_framework.simulation import DEFAULT_SETTINGS, simulation_settings
        try:
            settings = json.loads(values[-1])
            valid_settings = (isinstance(settings, dict) and set(settings) == set(DEFAULT_SETTINGS)
                              and simulation_settings({'simulation': settings}) == settings)
        except (TypeError, ValueError):
            pass
    if len(values) != len(_SCENE_IDENTITY_PARAMETERS) or not all(values) or not valid_settings:
        raise ValueError(
            'KMR controller scene identity is unavailable: /KMR_base_controller must supply '
            + ', '.join(_SCENE_IDENTITY_PARAMETERS)
            + '. The running controller may be outdated. Run make bootstrap-gazebo, '
            'then explicitly Reset Gazebo before Start System.'
        )
    return dict(zip(_SCENE_IDENTITY_PARAMETERS, values))


def _expected_scene_fingerprint(scene: dict, order: dict) -> str:
    """Accept only the bound machine program chosen from the saved Gazebo scene."""
    saved_scene = json.loads(SCENE_PATH.read_text(encoding='utf-8'))
    if scene == saved_scene:
        return fingerprint(saved_scene)
    machine_resource = order.get('machine_resource')
    if not isinstance(machine_resource, str):
        return fingerprint(scene)
    try:
        saved_machine = next(
            machine for machine in saved_scene['machines']
            if machine['resource_id'] == machine_resource
        )
        selected_machine = next(
            machine for machine in scene['machines']
            if machine['resource_id'] == machine_resource
        )
        part = order['parts'][0]
        trim_results = [
            requirement['result']
            for step in order['processPlan'][part]
            for requirement in step['processesToComplete']
            if requirement['process'] == 'trim'
        ]
    except (KeyError, IndexError, StopIteration, TypeError) as exc:
        raise ValueError('Bound machine scene configuration is invalid') from exc
    if len(trim_results) != 1:
        raise ValueError('Bound machine scene requires one trim result')
    selected_program = saved_machine.get('program_options', {}).get(trim_results[0])
    if selected_machine['current_configuration'] != selected_program:
        raise ValueError('Bound machine scene program differs from the saved option')
    expected_scene = deepcopy(saved_scene)
    expected_machine = next(
        machine for machine in expected_scene['machines']
        if machine['resource_id'] == machine_resource
    )
    expected_machine['current_configuration'] = deepcopy(selected_program)
    if scene != expected_scene:
        raise ValueError('Running scene differs outside the bound machine program')
    return fingerprint(saved_scene)


def storage_home(scene: dict, order: dict) -> dict:
    """Use a saved downward pickup posture above the configured Storage home dock."""
    storage = scene['Storage']
    dock = storage['KMR_docking_pose']
    candidates = list(dict.fromkeys([*order['parts'], *storage['slots']]))
    for part in candidates:
        pickup = storage['KMR_pick_docking_poses'].get(part)
        if pickup is None or math.dist(pickup[:2], dock[:2]) > .001:
            continue
        if abs(math.atan2(math.sin(pickup[2] - dock[5]), math.cos(pickup[2] - dock[5]))) > .001:
            continue
        config = scene['KMR']['task_execution']
        target = [*storage['slots'][part][:3], *config['pick_orientation_xyzw']]
        target[2] += config['grasp_height_m'] + storage['KMR_pick_approach_clearance_m'][part]
        return {'part_name': part, 'joints': list(storage['KMR_pick_arm_configurations'][part]),
                'tcp_pose': target}
    raise ValueError('Storage has no configured downward home posture at its home dock')


def inverse(pose: list[float]) -> list[float]:
    """Invert an xyz/xyzw rigid transform."""
    q = [-pose[3], -pose[4], -pose[5], pose[6]]
    return [*rotate(q, [-v for v in pose[:3]]), *q]


def run(request: dict, session: dict | None = None) -> dict:
    """Run one explicit ROS operation; imports remain inside the worker."""
    import rclpy
    from action_msgs.msg import GoalStatus
    from action_msgs.srv import CancelGoal
    from builtin_interfaces.msg import Duration
    from control_msgs.action import FollowJointTrajectory
    from gazebo_msgs.srv import GetEntityState
    from geometry_msgs.msg import Pose
    from linkattacher_msgs.srv import AttachLink, DetachLink
    from moveit_msgs.msg import (
        AttachedCollisionObject, CollisionObject, Constraints, JointConstraint,
        OrientationConstraint, PlanningScene, PlanningSceneComponents, PositionConstraint, RobotState, RobotTrajectory,
    )
    from moveit_msgs.srv import (
        ApplyPlanningScene, GetCartesianPath, GetMotionPlan, GetPositionFK, GetPositionIK, GetStateValidity, GetPlanningScene,
    )
    from nav2_msgs.action import NavigateToPose
    from rcl_interfaces.srv import GetParameters
    from rclpy.action import ActionClient
    from rclpy.clock import Clock, ClockType
    from rclpy.parameter import Parameter
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import JointState
    from shape_msgs.msg import SolidPrimitive
    from std_msgs.msg import String
    from std_srvs.srv import Trigger
    from tf2_ros import Buffer, TransformListener
    from trajectory_msgs.msg import JointTrajectoryPoint
    from unique_identifier_msgs.msg import UUID
    from cais_lab_robotics.action import DockKMR

    persistent = session is not None
    session = session if session is not None else {}
    owner_pid = session.setdefault('owner_pid', os.getppid())
    if 'node' not in session:
        rclpy.init(args=[])
        session['node'] = rclpy.create_node(
            'KMR_delivery_worker', parameter_overrides=[Parameter('use_sim_time', value=True)])
        from rclpy.executors import SingleThreadedExecutor
        session['executor'] = SingleThreadedExecutor()
        session['executor'].add_node(session['node'])
    node = session['node']
    executor = session['executor']
    scene = request['inputs']['scene']
    kmr = scene['KMR']
    config = deepcopy(kmr['task_execution'])
    pending = request.get('pending', {})
    mode = request.get('mode')
    name = pending.get('event_name')
    parameters = pending.get('parameters', {})
    valuation = request.get('valuation', {})
    held_part = valuation.get('KMR', {}).get('held_part')
    part = parameters.get('part_name') or held_part or pending.get('part_name') or config['part_name']
    part_geometry = request.get('geometry', {}).get(part, {})
    if part_geometry.get('dimensions_m'):
        config['part_dimensions_m'] = list(part_geometry['dimensions_m'])
    target_resource = (
        parameters.get('target_resource')
        or parameters.get('destination_location')
        or next(
            (
                row['resource_id']
                for row in scene['machines']
                if part in row.get('nominal_parts', [])
            ),
            'M1',
        )
    )
    source_resource = parameters.get('source_resource') or valuation.get('KMR', {}).get(
        'resource_location', 'Storage'
    )
    limits = joint_limits(ROOT/'ros2/cais_lab_robotics/urdf/KMR_recovery.urdf.xacro',
                          kmr['arm_joint_names'], config['joint_acceleration_limits'])
    if not bounded_joints(kmr['arm_joint_names'], config['carrying_arm_configuration'], limits):
        raise ValueError('Configured KMR carrying posture exceeds its joint limits')
    component_parts = list(request['inputs']['geometry']['assembly_board']['slots'])
    operations = []
    primitive_results = []
    joint_values = {}
    joint_times = {}
    joint_stamps = {}
    joint_samples = session.setdefault('joint_samples', {})
    stopped = False
    active_goals = []
    pending_goals = []
    clients = session.setdefault('clients', {})
    action_clients = session.setdefault('action_clients', {})
    transport_timer = None
    started = time.monotonic()
    request_received_at_unix = time.time()
    first_motion_at_unix = None
    clock_origin = None
    planning_seconds = 0.0
    base_motion_status = session.setdefault('base_motion_status', {})
    base_motion_samples = session.setdefault('base_motion_samples', deque(maxlen=1024))
    if 'tf_buffer' not in session:
        session['tf_buffer'] = Buffer()
        session['tf_listener'] = TransformListener(session['tf_buffer'], node)
    tf_buffer = session['tf_buffer']

    def interrupt(_signum, _frame):
        nonlocal stopped
        stopped = True

    old_int = signal.signal(signal.SIGINT, interrupt)
    old_term = signal.signal(signal.SIGTERM, interrupt)

    def on_joints(message):
        stamp = message.header.stamp.sec + message.header.stamp.nanosec / 1e9
        if stamp <= 0:
            return
        for name, position in zip(message.name, message.position):
            joint_samples.setdefault(name, deque(maxlen=100)).append((stamp, time.monotonic(), position))

    if 'subscription' not in session:
        session['subscription'] = node.create_subscription(JointState, '/joint_states', on_joints, qos_profile_sensor_data)
        session['transport'] = node.create_publisher(String, '/KMR/transport_custody', 10)
    transport = session['transport']

    def on_base_status(message):
        try:
            value = json.loads(message.data)
        except (TypeError, ValueError):
            return
        if isinstance(value, dict):
            base_motion_status.clear()
            base_motion_status.update(value)
            base_motion_samples.append((time.monotonic(), deepcopy(value)))

    if 'status_subscription' not in session:
        session['status_subscription'] = node.create_subscription(String, '/KMR/base_motion_status', on_base_status, 10)

    def timings():
        now = time.monotonic()
        simulated = node.get_clock().now().nanoseconds/1e9
        elapsed = simulated-clock_origin[1] if clock_origin else 0.0
        sampled_wall = now-clock_origin[0] if clock_origin else 0.0
        timing = {'wall_time_sec': now-started, 'simulation_time_sec': elapsed,
                'planning_wall_time_sec': planning_seconds,
                'real_time_factor': elapsed/sampled_wall if sampled_wall > 0 and elapsed >= 0 else None,
                'clock_reset': session.get('clock_reset', False),
                'trajectory_duration_sec': sum(
                    row['trajectory']['points'][-1]['time_from_start'] for row in operations
                    if row.get('trajectory', {}).get('points'))}
        timing.update({
            'dispatch_requested_at_unix': request.get('dispatch_requested_at_unix'),
            'request_received_at_unix': request_received_at_unix,
            'first_motion_at_unix': first_motion_at_unix,
        })
        requested = request.get('startup_timing', {}).get('start_requested_at_unix')
        if requested is not None and first_motion_at_unix is not None:
            timing['start_to_first_movement_wall_time_sec'] = first_motion_at_unix - requested
        return timing

    def check_clock():
        nonlocal stopped
        if os.getppid() != owner_pid:
            stopped = True
            raise InterruptedError('KMR execution owner exited')
        current = node.get_clock().now().nanoseconds / 1e9
        previous = session.get('last_clock', 0.)
        if current < previous:
            joint_samples.clear()
            session['clock_reset'] = True
            session.pop('prepared_place_turn', None)
        session['last_clock'] = current
        if session.get('clock_reset'):
            raise RuntimeError('Simulation clock reset; start an explicit fresh scene before execution')

    def spin_until(predicate, timeout=30., label='observation', progress_timeout=None):
        nonlocal clock_origin
        if stopped:
            raise InterruptedError('Stop System cancelled KMR execution')
        deadline = time.monotonic() + timeout
        progress = (MotionDeadline(
            progress_timeout, now=lambda: node.get_clock().now().nanoseconds / 1e9,
            cancelled=lambda: stopped, wall_timeout=timeout,
        ) if progress_timeout is not None else None)
        check_clock()
        while not predicate():
            if stopped:
                raise InterruptedError('Stop System cancelled KMR execution')
            if time.monotonic() >= deadline:
                raise TimeoutError(f'KMR {label} timed out')
            if progress is not None and not progress.pending():
                raise TimeoutError(f'KMR {label} exceeded its simulation duration')
            executor.spin_once(timeout_sec=.05)
            check_clock()
            if clock_origin is None and node.get_clock().now().nanoseconds > 0:
                clock_origin = (time.monotonic(), node.get_clock().now().nanoseconds/1e9)

    def service(kind, name, payload, timeout=30., retry_read=False):
        client = clients.setdefault(name, node.create_client(kind, name)) if name not in clients else clients[name]
        spin_until(client.service_is_ready, timeout, f'{name} discovery')
        deadline = time.monotonic() + timeout
        attempts = 0
        while True:
            attempts += 1
            future = client.call_async(payload)
            try:
                spin_until(future.done, min(5., deadline-time.monotonic()) if retry_read else timeout,
                           f'{name} response')
                break
            except TimeoutError:
                client.remove_pending_request(future)
                # Service discovery can precede DDS endpoint matching for a
                # newly joined client. Only repeat explicitly read-only queries.
                if not retry_read or time.monotonic() >= deadline:
                    raise
                _log.warning('Retrying read-only ROS query: %s', name)
        response = future.result()
        if response is None:
            raise RuntimeError(f'{name} returned no response')
        if retry_read and attempts > 1:
            operations.append({'operation': name, 'success': True, 'read_attempts': attempts})
        return response

    def action(kind, name, goal, timeout=120., while_active=None):
        nonlocal first_motion_at_unix
        if name not in action_clients:
            action_clients[name] = ActionClient(node, kind, name)
        client = action_clients[name]
        spin_until(client.server_is_ready, 30.)
        goal_id = UUID(uuid=list(uuid4().bytes))
        last_feedback = {}
        def feedback(message):
            value = message.feedback
            if hasattr(value, 'remaining_distance_m'):
                last_feedback.update(waypoint_index=value.waypoint_index,
                                     current_pose=[value.current_pose.x, value.current_pose.y, value.current_pose.theta],
                                     remaining_distance_m=value.remaining_distance_m,
                                     remaining_yaw_rad=value.remaining_yaw_rad)
        action_started = time.monotonic()
        simulation_started = node.get_clock().now().nanoseconds/1e9
        if kind is DockKMR:
            base_motion_samples.clear()
        pending = client.send_goal_async(goal, goal_uuid=goal_id, feedback_callback=feedback)
        entry = (name, goal_id, pending)
        pending_goals.append(entry)
        spin_until(pending.done, 10.)
        handle = pending.result()
        pending_goals.remove(entry)
        if not handle.accepted:
            raise RuntimeError(f'{name} rejected its goal')
        active_goals.append(handle)
        accepted_at = time.monotonic()
        if first_motion_at_unix is None:
            first_motion_at_unix = time.time()
        future = handle.get_result_async()
        if while_active is not None:
            try:
                while_active()
            except (RuntimeError, ValueError, TypeError, KeyError, TimeoutError) as exc:
                operations.append({
                    'operation': 'background_motion_preparation',
                    'success': False,
                    'used': False,
                    'reason': str(exc),
                })
        trajectory = getattr(goal, 'trajectory', None)
        trajectory = getattr(trajectory, 'joint_trajectory', trajectory)
        points = getattr(trajectory, 'points', [])
        progress_timeout = timeout
        if points:
            duration = points[-1].time_from_start
            progress_timeout = duration.sec + duration.nanosec / 1e9 + 5.
        spin_until(future.done, timeout, progress_timeout=progress_timeout)
        response = future.result()
        active_goals.remove(handle)
        code = getattr(response.result, 'error_code', None)
        operations.append({'operation': name, 'success': response.status == GoalStatus.STATUS_SUCCEEDED
                           and (code is None or getattr(code, 'val', code) in (0, 1))
                           and getattr(response.result, 'success', True),
                           'goal_id': bytes(goal_id.uuid).hex(), 'status': response.status,
                           'error_code': getattr(code, 'val', code),
                           'message': getattr(response.result, 'message', getattr(response.result, 'error_string', '')),
                           'dispatch_wall_time_sec': accepted_at-action_started,
                           'controller_execution_wall_time_sec': time.monotonic()-accepted_at,
                           'wall_time_sec': time.monotonic()-action_started,
                           'simulation_time_sec': node.get_clock().now().nanoseconds/1e9-simulation_started})
        if kind is DockKMR:
            operations[-1].update(
                feedback=deepcopy(last_feedback), base_motion_status=deepcopy(base_motion_status),
                base_motion_samples=[{'received_after_action_start_sec': received - action_started,
                                      **deepcopy(sample)} for received, sample in base_motion_samples],
            )
        if response.status != GoalStatus.STATUS_SUCCEEDED:
            detail = operations[-1]['message'] or base_motion_status.get('abort_reason', '')
            raise RuntimeError(f'{name} ended with action status {response.status}: {detail}')
        if code is not None and (getattr(code, 'val', code) not in (0, 1)):
            raise RuntimeError(f'{name} returned error {code}')
        if hasattr(response.result, 'success') and not response.result.success:
            raise RuntimeError(f'{name}: {response.result.message}')
        return response.result

    def pose_message(values):
        pose = Pose()
        pose.position.x, pose.position.y, pose.position.z = (float(v) for v in values[:3])
        pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = (float(v) for v in values[3:])
        return pose

    def pose_values(pose):
        return [pose.position.x, pose.position.y, pose.position.z,
                pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]

    def entity(name):
        query = GetEntityState.Request(name=name, reference_frame='world')
        response = service(GetEntityState, config['services']['get_state'], query, retry_read=True)
        if not response.success:
            raise RuntimeError(f'Gazebo entity unavailable: {name}')
        return pose_values(response.state.pose)

    def tcp():
        spin_until(lambda: tf_buffer.can_transform('world', config['tcp_link'], rclpy.time.Time()), 10.)
        transform = tf_buffer.lookup_transform('world', config['tcp_link'], rclpy.time.Time())
        stamp = transform.header.stamp.sec + transform.header.stamp.nanosec / 1e9
        if abs(node.get_clock().now().nanoseconds/1e9 - stamp) > 1.:
            raise ValueError('KMR TCP feedback is stale')
        v, q = transform.transform.translation, transform.transform.rotation
        return [v.x, v.y, v.z, q.x, q.y, q.z, q.w]

    def fresh_state(timeout=45.):
        required = [*kmr['arm_joint_names'], kmr['gripper_joint'], 'KMR_base_x_joint', 'KMR_base_y_joint', 'KMR_base_yaw_joint']
        def select_feedback():
            now = node.get_clock().now().nanoseconds / 1e9
            for name, samples in joint_samples.items():
                for stamp, received, position in reversed(samples):
                    # Gazebo joint stamps can arrive ahead of the lower-rate /clock.
                    # Retain them until clock catches up; never accept future feedback.
                    if 0 <= now-stamp <= 1. and time.monotonic()-received < 1.:
                        joint_values[name] = position
                        joint_times[name] = received
                        joint_stamps[name] = stamp
                        break
            return all(name in joint_times and time.monotonic()-joint_times[name] < 1.
                       and 0 <= now-joint_stamps[name] <= 1. for name in required)
        spin_until(select_feedback, timeout, 'fresh joint feedback')
        state = JointState()
        state.header.stamp = node.get_clock().now().to_msg()
        state.name = list(joint_values)
        state.position = [joint_values[name] for name in state.name]
        return state

    def apply(planning_scene):
        query = ApplyPlanningScene.Request(scene=planning_scene)
        for attempt in range(3):
            try:
                response = service(ApplyPlanningScene, config['services']['apply_scene'], query, 10.)
                break
            except TimeoutError:
                if attempt == 2:
                    raise
                # These diffs replace/remove collision objects by fixed ID;
                # repeating the identical update cannot start robot motion.
                _log.warning('Retrying idempotent KMR collision-scene update')
        if not response.success:
            raise RuntimeError('MoveIt rejected the collision scene')
        if attempt:
            operations.append({'operation': config['services']['apply_scene'], 'success': True,
                               'scene_update_attempts': attempt+1})

    def box_object(name, pose, size):
        obj = CollisionObject()
        obj.header.frame_id = 'world'
        obj.id = name
        primitive = SolidPrimitive(type=SolidPrimitive.BOX, dimensions=[float(v) for v in size])
        obj.primitives = [primitive]
        obj.primitive_poses = [pose_message(pose)]
        obj.operation = CollisionObject.ADD
        return obj

    def install_scene():
        planning_scene = PlanningScene(is_diff=True)
        empty_return = (mode == 'environment_task' and name == 'move_to_resource'
                        and held_part is None)
        initialize = not session.get('collision_scene_installed') or mode != 'environment_task'
        observed_parts = component_parts if initialize else ([] if empty_return else [part])
        part_poses = {part_name: entity(part_name) for part_name in observed_parts}
        from cais_spade_llm.recovery_framework.part_collision import collision_object

        boxes = collision_boxes(ROOT/'ros2/cais_lab_robotics/worlds/table_recovery_framework.world',
                                ROOT/'ros2/cais_lab_robotics/models', part_poses,
                                exact_models={*component_parts, 'Gear_Plate'})
        existing = service(GetPlanningScene, '/get_planning_scene', GetPlanningScene.Request(
            components=PlanningSceneComponents(components=PlanningSceneComponents.WORLD_OBJECT_NAMES)),
            retry_read=True).scene.world.collision_objects
        existing_ids = {obj.id for obj in existing}
        excluded_prefix = None if empty_return else part + '/'
        released_objects = sorted(set(observed_parts).intersection(existing_ids))
        planning_scene.world.collision_objects = [
            CollisionObject(id=identifier, operation=CollisionObject.REMOVE)
            for identifier in released_objects
        ]
        # Once initialized, each resource updates only the parts it owns. In
        # particular, UR payloads must never become fixed obstacles during carry.
        planning_scene.world.collision_objects.extend(
            CollisionObject(id=row['id'], operation=CollisionObject.REMOVE)
            if excluded_prefix and row['id'].startswith(excluded_prefix)
            else collision_object(row)
            for row in boxes
            if not (excluded_prefix and row['id'].startswith(excluded_prefix))
            or row['id'] in existing_ids
        )
        apply(planning_scene)
        session['collision_scene_installed'] = True
        operations.append({
            'operation': 'apply_planning_scene', 'success': True, 'collision_objects': len(boxes),
            'initialized': initialize,
            'removed_released_part_objects': released_objects,
            'observed_part_poses': part_poses,
        })

    def execute(trajectory):
        if not trajectory.joint_trajectory.points:
            raise ValueError('MoveIt returned an empty trajectory')
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = trajectory.joint_trajectory
        action(
            FollowJointTrajectory,
            kmr['arm_controller']+'/follow_joint_trajectory',
            goal,
        )

    def observed_state():
        return RobotState(joint_state=fresh_state(), is_diff=True)

    def updated_state(state, names, positions):
        result = deepcopy(state)
        values = dict(zip(result.joint_state.name, result.joint_state.position))
        values.update(zip(names, positions))
        result.joint_state.name = list(values)
        result.joint_state.position = [float(value) for value in values.values()]
        result.joint_state.velocity = []
        result.joint_state.effort = []
        return result

    def state_after(state, plan):
        trajectory = plan[0].joint_trajectory
        return updated_state(state, trajectory.joint_names, trajectory.points[-1].positions)

    def configuration_pose(state, joints):
        query = GetPositionFK.Request(
            robot_state=updated_state(state, kmr['arm_joint_names'], joints),
            fk_link_names=[config['tcp_link']])
        query.header.frame_id = 'world'
        result = service(GetPositionFK, '/compute_fk', query, retry_read=True)
        if result.error_code.val != 1 or len(result.pose_stamped) != 1:
            raise ValueError('KMR configuration FK is unavailable')
        return pose_values(result.pose_stamped[0].pose)

    def plan_motion(state, target=None, joints=None, cartesian=False, hold_arm_base=False, waypoints=None):
        nonlocal planning_seconds
        planning_started = time.monotonic()
        if config.get('cartesian_motion_only'):
            from cais_spade_llm.resources.robot.cartesian_waypoints import resolve_waypoints, robot_trajectory

            if joints is not None:
                raise ValueError('KMR arm motion requires explicit downward Cartesian waypoints')
            values = dict(zip(state.joint_state.name, state.joint_state.position))
            initial = [values[name] for name in kmr['arm_joint_names']]
            start = configuration_pose(state, initial)
            targets = [*(waypoints or []), target]
            if any(rotate(pose[3:], [0., 0., 1.])[2] > -math.cos(config['orientation_tolerance_rad'])
                   for pose in [start, *targets]):
                raise ValueError('KMR Cartesian waypoints must keep the gripper downward')
            def solve(pose, seed):
                if stopped:
                    raise InterruptedError('Stop System cancelled KMR waypoint conversion')
                query = GetPositionIK.Request()
                query.ik_request.group_name = config['planning_group']
                query.ik_request.ik_link_name = config['tcp_link']
                query.ik_request.robot_state = updated_state(state, kmr['arm_joint_names'], seed)
                query.ik_request.pose_stamped.header.frame_id = 'world'
                query.ik_request.pose_stamped.pose = pose_message(pose)
                query.ik_request.avoid_collisions = True
                query.ik_request.timeout = Duration(nanosec=200_000_000)
                result = service(GetPositionIK, '/compute_ik', query, retry_read=True)
                if result.error_code.val != 1:
                    raise ValueError(f'KMR waypoint IK unavailable at {pose}')
                positions = dict(zip(result.solution.joint_state.name, result.solution.joint_state.position))
                return [positions[name] for name in kmr['arm_joint_names']]
            settings = config['cartesian_waypoints']
            rows = resolve_waypoints(start_pose=start, start_joints=initial, waypoints=targets,
                names=kmr['arm_joint_names'], limits=limits, solve_ik=solve,
                linear_step=settings['linear_step_m'], angular_step=settings['angular_step_rad'],
                maximum_joint_step=settings['max_joint_step_rad'])
            trajectory = robot_trajectory(kmr['arm_joint_names'], rows)
            checked = 0
            for left, right in zip(rows, rows[1:]):
                count = max(1, math.ceil(max(abs(b - a) for a, b in zip(left['positions'], right['positions'])) / .05))
                for index in range(count + 1):
                    positions = [a + (b - a) * index / count for a, b in zip(left['positions'], right['positions'])]
                    query = GetStateValidity.Request(group_name=config['planning_group'],
                        robot_state=updated_state(state, kmr['arm_joint_names'], positions))
                    valid = service(GetStateValidity, '/check_state_validity', query, retry_read=True)
                    if not valid.valid:
                        raise ValueError(f'KMR waypoint motion is obstructed: {[(c.contact_body_1, c.contact_body_2) for c in valid.contacts]}')
                    checked += 1
            planning_seconds += time.monotonic() - planning_started
            info = {'operation': 'cartesian_path', 'motion_method': 'Cartesian waypoints',
                    'success': True, 'target': deepcopy(target), 'waypoints': deepcopy(targets),
                    'avoid_collisions': True, 'fraction': 1., 'checked_states': checked}
        elif cartesian:
            query = GetCartesianPath.Request()
            query.header.frame_id = 'world'
            query.group_name = config['planning_group']
            query.link_name = config['tcp_link']
            query.start_state = deepcopy(state)
            query.waypoints = [pose_message(value) for value in [*(waypoints or []), target]]
            query.max_step = .005
            query.jump_threshold = 2.
            query.revolute_jump_threshold = .35
            query.avoid_collisions = True
            if hold_arm_base:
                values = dict(zip(state.joint_state.name, state.joint_state.position))
                query.path_constraints.joint_constraints = [JointConstraint(
                    joint_name=kmr['arm_joint_names'][0], position=values[kmr['arm_joint_names'][0]],
                    tolerance_above=.005, tolerance_below=.005, weight=1.)]
            result = service(GetCartesianPath, config['services']['cartesian_path'], query, retry_read=True)
            info = {'operation': 'cartesian_path', 'success': result.error_code.val == 1
                               and result.fraction >= .999, 'target': deepcopy(target),
                               'avoid_collisions': True, 'fraction': result.fraction,
                               'error_code': result.error_code.val,
                               'waypoint_count': len(waypoints or []) + 1, 'joint_target': deepcopy(joints)}
            planning_seconds += time.monotonic()-planning_started
            if result.error_code.val != 1 or result.fraction < .999:
                raise ValueError(f'Collision-aware Cartesian path incomplete: {result.fraction}')
            trajectory = result.solution
        else:
            query = GetMotionPlan.Request()
            motion = query.motion_plan_request
            motion.group_name = config['planning_group']
            motion.start_state = deepcopy(state)
            motion.allowed_planning_time = 2.
            motion.num_planning_attempts = 1
            motion.max_velocity_scaling_factor = 1.0
            motion.max_acceleration_scaling_factor = 1.0
            constraints = Constraints()
            if joints is not None:
                constraints.joint_constraints = [JointConstraint(joint_name=name, position=float(value), tolerance_above=.005, tolerance_below=.005, weight=1.) for name, value in zip(kmr['arm_joint_names'], joints)]
            else:
                pc = PositionConstraint()
                pc.header.frame_id = 'world'
                pc.link_name = config['tcp_link']
                pc.weight = 1.
                pc.constraint_region.primitives = [SolidPrimitive(type=SolidPrimitive.SPHERE, dimensions=[.002])]
                pc.constraint_region.primitive_poses = [pose_message([*target[:3], 0., 0., 0., 1.])]
                oc = OrientationConstraint()
                oc.header.frame_id = 'world'
                oc.link_name = config['tcp_link']
                oc.orientation = pose_message(target).orientation
                oc.absolute_x_axis_tolerance = .01
                oc.absolute_y_axis_tolerance = .01
                oc.absolute_z_axis_tolerance = .01
                oc.weight = 1.
                constraints.position_constraints = [pc]
                constraints.orientation_constraints = [oc]
            motion.goal_constraints = [constraints]
            if hold_arm_base:
                values = dict(zip(state.joint_state.name, state.joint_state.position))
                motion.path_constraints.joint_constraints = [JointConstraint(
                    joint_name=kmr['arm_joint_names'][0], position=values[kmr['arm_joint_names'][0]],
                    tolerance_above=.005, tolerance_below=.005, weight=1.)]
            response = service(
                GetMotionPlan,
                config['services']['motion_plan'],
                query,
                60.,
            ).motion_plan_response
            info = {'operation': 'motion_plan', 'success': response.error_code.val == 1,
                    'target': deepcopy(target), 'joint_target': deepcopy(joints), 'avoid_collisions': True,
                    'error_code': response.error_code.val}
            planning_seconds += time.monotonic()-planning_started
            if response.error_code.val != 1:
                raise ValueError(f'Collision-aware KMR motion planning failed: {response.error_code.val}')
            trajectory = response.trajectory
        retime_trajectory(trajectory.joint_trajectory, limits,
                          config['velocity_scaling'], config['acceleration_scaling'])
        info['planning_wall_time_sec'] = time.monotonic()-planning_started
        info['trajectory'] = {
            'joint_names': list(trajectory.joint_trajectory.joint_names),
            'points': [{'positions': list(point.positions), 'velocities': list(point.velocities),
                        'accelerations': list(point.accelerations),
                        'time_from_start': point.time_from_start.sec+point.time_from_start.nanosec/1e9}
                       for point in trajectory.joint_trajectory.points],
        }
        return trajectory, info

    def execute_plan(plan):
        trajectory, info = plan
        measured = dict(zip((state := fresh_state()).name, state.position))
        if any(abs(measured[name]-value) > .02 for name, value in zip(
            trajectory.joint_trajectory.joint_names, trajectory.joint_trajectory.points[0].positions
        )):
            raise ValueError('KMR moved after arm planning; refusing the stale trajectory')
        operations.append(deepcopy(info))
        execute(trajectory)
        endpoint = trajectory.joint_trajectory.points[-1].positions
        observed_endpoint = {}
        def settled():
            state = fresh_state(timeout=5.)
            observed_endpoint.update(zip(state.name, state.position))
            return all(abs(observed_endpoint[name]-value) <= .005 for name, value in zip(
                trajectory.joint_trajectory.joint_names, endpoint,
            ))
        # Controller completion can precede the next /clock sample. Wait for
        # non-future joint feedback instead of comparing the next path to an old sample.
        spin_until(settled, 5., 'joint feedback at trajectory endpoint')
        operations.append({'operation': 'observed_arm_endpoint', 'success': True,
                           'joints': {name: observed_endpoint[name] for name in trajectory.joint_trajectory.joint_names},
                           'stamps': {name: joint_stamps[name] for name in trajectory.joint_trajectory.joint_names}})
        target = info['target']
        if target is not None:
            spin_until(lambda: (math.dist(tcp()[:3], target[:3]) <= config['position_tolerance_m']
                                    and abs(sum(a*b for a, b in zip(tcp()[3:], target[3:])))
                                    >= math.cos(config['orientation_tolerance_rad'] / 2)),
                       5., 'TCP feedback at requested pose')

    def move(target=None, joints=None, cartesian=False):
        execute_plan(plan_motion(observed_state(), target=target, joints=joints, cartesian=cartesian))

    def move_cartesian(target):
        """Execute a complete collision-checked path with fixed TCP orientation."""
        current = tcp()
        if abs(sum(a * b for a, b in zip(current[3:], target[3:]))) < math.cos(
            config['orientation_tolerance_rad'] / 2
        ):
            raise ValueError('Cartesian segment must preserve the observed TCP orientation')
        move(target=target, cartesian=True)
        return {'success': True, 'tcp_pose': tcp(), 'avoid_collisions': True, 'fraction': 1.0}

    def move_to_configuration(joints, hold_arm_base=False):
        measured = dict(zip((state := fresh_state()).name, state.position))
        if any(abs(measured[name] - value) > .005
               for name, value in zip(kmr['arm_joint_names'], joints, strict=True)):
            execute_plan(plan_motion(observed_state(), joints=joints, hold_arm_base=hold_arm_base))
        return {'success': True, 'joints': list(joints), 'tcp_pose': tcp()}

    def move_home():
        if valuation.get('KMR', {}).get('held_part') is not None or (request.get('custody') or {}).get('attached') is True:
            raise ValueError('KMR downward home requires an empty gripper')
        dock_pose = scene['Storage']['KMR_docking_pose']
        base_pose = entity('KMR')
        dock_orientation = quaternion(dock_pose[3:])
        if (math.dist(base_pose[:2], dock_pose[:2]) > .035
                or abs(sum(a * b for a, b in zip(base_pose[3:], dock_orientation))) < math.cos(.035 / 2)):
            raise ValueError('KMR downward home requires the observed Storage dock')
        home = storage_home(scene, request['inputs']['product_order'])
        gripper(kmr['gripper_stroke_m'])
        move_to_pose(home['tcp_pose'])
        actual = tcp()
        state = fresh_state()
        joints = dict(zip(state.name, state.position))
        error = max(abs(joints[name] - value) for name, value in zip(kmr['arm_joint_names'], home['joints'], strict=True))
        down = rotate(actual[3:], [0., 0., 1.])
        if (down[2] > -math.cos(config['orientation_tolerance_rad'])
                or math.dist(actual[:3], home['tcp_pose'][:3]) > .012):
            raise ValueError('KMR downward home endpoint was not observed')
        observation = {'operation': 'observed_storage_home', 'success': True,
                       'home_pose_observed': True, 'downward_facing': True,
                       'tcp_pose': actual, 'target': home, 'max_joint_error_rad': error,
                       'gripper_open': abs(joints[kmr['gripper_joint']] - kmr['gripper_stroke_m']) <= .006}
        if not observation['gripper_open']:
            raise ValueError('KMR home gripper is not open')
        operations.append(observation)
        return observation

    def move_to_pose(target, seed=None, waypoints=None):
        """Follow explicit downward waypoints from the observed arm state."""
        current = tcp()
        if (not waypoints and math.dist(current[:3], target[:3]) <= config['position_tolerance_m']
                and abs(sum(a * b for a, b in zip(current[3:], target[3:])))
                >= math.cos(config['orientation_tolerance_rad'] / 2)):
            return {'success': True, 'tcp_pose': current, 'motion_required': False}
        execute_plan(plan_motion(observed_state(), target=target, cartesian=True, waypoints=waypoints))
        return {'success': True, 'tcp_pose': tcp(), 'motion_method': 'Cartesian waypoints'}

    def rotate_arm_base(joint_a1):
        """Turn only joint_a1 after validating the complete arm and held-part sweep."""
        state = observed_state()
        values = dict(zip(state.joint_state.name, state.joint_state.position))
        initial = [values[name] for name in kmr['arm_joint_names']]
        delta = joint_a1 - initial[0]
        if config.get('cartesian_motion_only'):
            raise ValueError('Use the computed downward transfer waypoints for KMR arm motion')
        bound = limits[kmr['arm_joint_names'][0]]
        duration = max(.05, 1.875 * abs(delta) / bound['velocity'],
                       math.sqrt(5.774 * abs(delta) / bound['acceleration']))
        count = max(1, math.ceil(1.875 * abs(delta) / .02))
        trajectory = RobotTrajectory()
        trajectory.joint_trajectory.joint_names = list(kmr['arm_joint_names'])
        for index in range(count + 1):
            positions = initial.copy()
            u = index / count
            positions[0] += delta * (10*u**3 - 15*u**4 + 6*u**5)
            query = GetStateValidity.Request(group_name=config['planning_group'])
            query.robot_state = updated_state(state, kmr['arm_joint_names'], positions)
            response = service(GetStateValidity, '/check_state_validity', query, retry_read=True)
            if not response.valid:
                contacts = [(item.contact_body_1, item.contact_body_2) for item in response.contacts]
                raise ValueError(f'joint_a1 turn intersects geometry: {contacts}')
            velocity = delta * (30*u**2 - 60*u**3 + 30*u**4) / duration
            acceleration = delta * (60*u - 180*u**2 + 120*u**3) / duration**2
            sec, nanosec = divmod(round(duration * u * 1e9), 1_000_000_000)
            trajectory.joint_trajectory.points.append(JointTrajectoryPoint(
                positions=positions, velocities=[velocity, *([0.] * 6)],
                accelerations=[acceleration, *([0.] * 6)],
                time_from_start=Duration(sec=sec, nanosec=nanosec)))
        retime_trajectory(trajectory.joint_trajectory, limits,
                          config['velocity_scaling'], config['acceleration_scaling'])
        info = {'operation': 'rotate_arm_base', 'success': True, 'target': None,
                'avoid_collisions': True, 'checked_states': count + 1,
                'joint_a1': joint_a1, 'trajectory': {
                    'joint_names': list(kmr['arm_joint_names']),
                    'points': [{'positions': list(point.positions),
                                'time_from_start': point.time_from_start.sec + point.time_from_start.nanosec / 1e9}
                               for point in trajectory.joint_trajectory.points]}}
        execute_plan((trajectory, info))
        return {'success': True, 'joint_a1': joint_a1, 'tcp_pose': tcp()}

    def ik_candidates(target, state, seed_configurations=None):
        nonlocal planning_seconds
        values = dict(zip(state.joint_state.name, state.joint_state.position))
        current = [values[name] for name in kmr['arm_joint_names']]
        seeds = [current, *(seed_configurations if seed_configurations is not None else
                            [config['carrying_arm_configuration'], *config['ik_seed_configurations']])]
        solutions = []
        for seed in seeds:
            query = GetPositionIK.Request()
            query.ik_request.group_name = config['planning_group']
            query.ik_request.ik_link_name = config['tcp_link']
            query.ik_request.robot_state = updated_state(state, kmr['arm_joint_names'], seed)
            query.ik_request.pose_stamped.header.frame_id = 'world'
            query.ik_request.pose_stamped.pose = pose_message(target)
            query.ik_request.avoid_collisions = True
            query.ik_request.timeout = Duration(nanosec=200_000_000)
            before = time.monotonic()
            response = service(GetPositionIK, '/compute_ik', query, retry_read=True)
            planning_seconds += time.monotonic()-before
            if response.error_code.val != 1:
                continue
            answer = dict(zip(response.solution.joint_state.name, response.solution.joint_state.position))
            candidate = [answer[name] for name in kmr['arm_joint_names']]
            if bounded_joints(kmr['arm_joint_names'], candidate, limits) and all(
                max(abs(a-b) for a, b in zip(candidate, previous)) > .02 for previous in solutions
            ):
                solutions.append(candidate)
        return sorted(solutions, key=lambda candidate: sum(abs(a-b) for a, b in zip(candidate, current)))

    def planned_attachment(state, grasp_pose, part_pose):
        result = updated_state(state, [kmr['gripper_joint']], [config['closed_gripper_width_m']])
        support_allowance = float(config['attachment_support_contact_allowance_m'])
        center = compose(part_pose, [
            0.,
            0.,
            config['part_dimensions_m'][2] / 2 + support_allowance,
            0.,
            0.,
            0.,
            1.,
        ])
        attached = AttachedCollisionObject()
        attached.link_name = config['tcp_link']
        attached.touch_links = [config['tcp_link'], 'rg2_base_link', 'rg2_left_inner_finger', 'rg2_right_inner_finger']
        attached.object = box_object(part, compose(inverse(grasp_pose), center), config['part_dimensions_m'])
        attached.object.header.frame_id = config['tcp_link']
        result.attached_collision_objects = [attached]
        return result

    def route_between(source: str, target: str) -> list[list[float]]:
        for row in kmr['predefined_route_waypoints']:
            if row['resources'] == [source, target]:
                route = deepcopy(row['poses'])
                if source == 'Storage':
                    pick_pose = dict(
                        scene['Storage'].get('KMR_pick_docking_poses') or {}
                    ).get(part)
                    if pick_pose and math.dist(pick_pose[:2], route[0][:2]) > .01:
                        route = (
                            [list(pick_pose), *route[1:]]
                            if target == 'M2'
                            else [list(pick_pose), *route]
                        )
                return route
            if row['resources'] == [target, source]:
                return list(reversed(deepcopy(row['poses'])))
        raise ValueError(f'No configured KMR route from {source} to {target}')

    def validate_transport(state, route=None):
        nonlocal planning_seconds
        before = time.monotonic()
        route = route if route is not None else route_between(source_resource, target_resource)
        checked = 0
        for start, end in zip(route, route[1:]):
            steps = max(1, math.ceil(math.dist(start[:2], end[:2])/config['transport_sample_step_m']),
                        math.ceil(abs(end[2] - start[2]) / .05))
            for index in range(steps+1):
                pose = [a+(b-a)*index/steps for a, b in zip(start, end)]
                query = GetStateValidity.Request(group_name=config['planning_group'])
                query.robot_state = updated_state(state, ['KMR_base_x_joint', 'KMR_base_y_joint', 'KMR_base_yaw_joint'], pose)
                response = service(GetStateValidity, '/check_state_validity', query, retry_read=True)
                if not response.valid:
                    collisions = [(c.contact_body_1, c.contact_body_2) for c in response.contacts]
                    raise ValueError(f'KMR arm/gripper/part transport sweep is obstructed at {pose}: {collisions}')
                checked += 1
        planning_seconds += time.monotonic()-before
        return {'operation': 'validated_transport_sweep', 'success': True,
                'checked_states': checked, 'sample_step_m': config['transport_sample_step_m'],
                'route': deepcopy(route), 'arm_configuration': [
                    dict(zip(state.joint_state.name, state.joint_state.position))[name]
                    for name in kmr['arm_joint_names']]}

    def authorize_empty_transport(route=None):
        state = observed_state()
        sweep = validate_transport(state, route=route)
        operations.append(sweep)
        custody_id = uuid4().hex
        transport.publish(String(data=json.dumps({
            **probe, 'part_name': None, 'attached': False, 'custody_id': custody_id,
            'transport_sweep_validated': True,
            'carrying_arm_configuration': sweep['arm_configuration'],
        })))
        spin_until(lambda: base_motion_status.get('transport_custody_ack') == {
            'custody_id': custody_id, 'arm_parked': True},
            2., 'empty KMR downward-arm acknowledgement')
        return deepcopy(base_motion_status['transport_custody_ack'])

    def gripper(width):
        fresh_state()
        samples = list(joint_samples.get(kmr['gripper_joint'], ()))
        stable = (
            len(samples) >= 2
            and not active_goals
            and not pending_goals
            and abs(samples[-1][2] - width) <= .006
            and abs(samples[-2][2] - width) <= .006
            and abs(samples[-1][2] - samples[-2][2]) <= .001
            and time.monotonic() - samples[-1][1] < 1.
            and time.monotonic() - samples[-2][1] < 1.
        )
        if stable:
            operations.append({
                'operation': 'observed_gripper', 'success': True,
                'command_sent': False, 'reason': 'fresh stable endpoint already observed',
                'target': width, 'position': samples[-1][2], 'stamp': samples[-1][0],
            })
            return
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = [kmr['gripper_joint']]
        goal.trajectory.points = [JointTrajectoryPoint(positions=[float(width)], time_from_start=Duration(sec=1))]
        action(FollowJointTrajectory, kmr['gripper_controller']+'/follow_joint_trajectory', goal)
        def reached():
            fresh_state()
            return abs(joint_values[kmr['gripper_joint']]-width) <= .006
        spin_until(reached, 5., 'gripper feedback at requested width')
        operations.append({'operation': 'observed_gripper', 'success': True,
                           'command_sent': True,
                           'target': width, 'position': joint_values[kmr['gripper_joint']],
                           'stamp': joint_stamps[kmr['gripper_joint']]})

    def confirm_carrying():
        configuration = (request.get('custody') or {}).get('carrying_arm_configuration')
        if not isinstance(configuration, list) or not bounded_joints(kmr['arm_joint_names'], configuration, limits):
            raise ValueError('No acknowledged carrying arm configuration')
        def reached():
            fresh_state()
            return all(abs(joint_values[name]-target) <= .02 for name, target in zip(
                kmr['arm_joint_names'], configuration))
        spin_until(reached, 5., 'arm feedback at carrying configuration')
        operations.append({'operation': 'observed_carrying', 'success': True,
                           'joints': {name: joint_values[name] for name in kmr['arm_joint_names']}})

    def attach(enabled):
        kind = AttachLink if enabled else DetachLink
        query = kind.Request(model1_name='KMR', link1_name=config['attach_link'], model2_name=part, link2_name='link')
        response = service(kind, config['services']['attach' if enabled else 'detach'], query)
        operations.append({'operation': 'attach' if enabled else 'detach',
                           'success': response.success, 'message': response.message})
        if not response.success:
            raise RuntimeError(response.message)

    def part_collision(attached):
        planning_scene = PlanningScene(is_diff=True)
        planning_scene.robot_state.is_diff = True
        attached_object = AttachedCollisionObject()
        attached_object.link_name = config['tcp_link']
        attached_object.touch_links = [config['tcp_link'], 'rg2_base_link', 'rg2_left_inner_finger', 'rg2_right_inner_finger']
        observed = entity(part)
        center = compose(observed, [0., 0., config['part_dimensions_m'][2]/2, 0., 0., 0., 1.])
        attached_object.object = box_object(part, center, config['part_dimensions_m'])
        if not attached:
            attached_object.object.operation = CollisionObject.REMOVE
            planning_scene.world.collision_objects = [box_object(part, center, config['part_dimensions_m'])]
        planning_scene.robot_state.attached_collision_objects = [attached_object]
        apply(planning_scene)

    def custody(transform):
        measured = entity(part)
        predicted = compose(tcp(), transform)
        if math.dist(measured[:3], predicted[:3]) > .008:
            raise ValueError('Gazebo part does not follow the held-part transform')
        if abs(sum(a*b for a, b in zip(measured[3:], predicted[3:]))) < .995:
            raise ValueError('Gazebo held part orientation changed')
        operations.append({'operation': 'observed_custody', 'success': True, 'part_pose': measured, 'tcp_pose': tcp()})
        return measured

    def serialize(value):
        if isinstance(value, dict):
            return {key: serialize(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return [serialize(item) for item in value]
        if hasattr(value, 'get_fields_and_field_types'):
            from rosidl_runtime_py.convert import message_to_ordereddict
            return message_to_ordereddict(value)
        return deepcopy(value)

    def observe_grasp(initial):
        before = entity(part)
        if math.dist(before[:3], initial[:3]) > .01:
            raise ValueError('Part moved before the simulated grasp attachment')
        return compose(inverse(tcp()), before)

    def observe_carrying(transform):
        custody(transform)
        state = fresh_state()
        values = dict(zip(state.name, state.position))
        configuration = [values[name] for name in kmr['arm_joint_names']]
        return dict(part_location='KMR', attached=True, arm_parked=True, grasp_transform=transform,
                    carrying_arm_configuration=configuration, carrying_tcp_pose=tcp())

    def observe_custody(previous):
        transform = previous.get('grasp_transform')
        if previous.get('attached') is not True or not isinstance(transform, list) or len(transform) != 7:
            raise ValueError('No acknowledged grasp transform for the held part')
        custody(transform)
        return transform

    def dock(target_resource, transform):
        nonlocal transport_timer
        if target_resource not in {'M1', 'M2'}:
            raise ValueError('Loaded KMR delivery target must be M1 or M2')
        custody(transform)
        # Relative Gazebo observations share one physics timestamp,
        # avoiding a moving-base mismatch between service and TF samples.
        custody_query = GetEntityState.Request(
            name=part, reference_frame='KMR::'+config['attach_link'],
        )
        relative = service(GetEntityState, config['services']['get_state'], custody_query, retry_read=True)
        if not relative.success:
            raise ValueError('Cannot observe KMR transport custody relative to the attachment link')
        expected_relative = pose_values(relative.state.pose)
        transport_configuration = (request.get('custody') or {}).get('carrying_arm_configuration')
        if not any(row.get('operation') == 'validated_transport_sweep' and row.get('success')
                   for row in operations):
            operations.append(validate_transport(observed_state()))
        transport_posture = {'carrying_arm_configuration': transport_configuration,
                             'transport_sweep_validated': True}
        custody_evidence = {'operation': 'observed_transport_custody', 'success': True,
                            'checked_samples': 1, 'reference_frame': custody_query.reference_frame,
                            'expected_relative_pose': expected_relative.copy()}
        operations.append(custody_evidence)
        custody_id = uuid4().hex
        custody_client = clients[config['services']['get_state']]
        custody_future = custody_client.call_async(custody_query)
        def publish_custody():
            nonlocal custody_future
            if not custody_future.done():
                return
            measured = custody_future.result()
            actual = pose_values(measured.state.pose) if measured and measured.success else None
            matches = bool(actual and math.dist(actual[:3], expected_relative[:3]) <= .008
                           and abs(sum(a*b for a, b in zip(actual[3:], expected_relative[3:]))) >= .995)
            custody_evidence.update(success=matches, last_relative_pose=actual,
                                    checked_samples=custody_evidence['checked_samples']+1)
            payload = {**probe, **transport_posture, 'part_name': part, 'attached': matches, 'custody_id': custody_id}
            if matches:
                custody_future = custody_client.call_async(custody_query)
            else:
                payload['error'] = 'held part moved relative to the attachment link'
                transport_timer.cancel()
            transport.publish(String(data=json.dumps(payload)))
        transport_timer = node.create_timer(
            .1, publish_custody, clock=Clock(clock_type=ClockType.STEADY_TIME),
        )
        transport.publish(String(data=json.dumps({
            **probe, **transport_posture, 'part_name': part, 'attached': True, 'custody_id': custody_id,
        })))
        # DDS can deliver the action goal before the custody topic. Wait for
        # this controller's guarded acknowledgement, not an arbitrary delay.
        spin_until(lambda: base_motion_status.get('transport_custody_ack') == {
            'custody_id': custody_id, 'arm_parked': True,
        }, 2.)
        custody_evidence['controller_acknowledgement'] = deepcopy(
            base_motion_status['transport_custody_ack'],
        )
        action(
            DockKMR,
            kmr['docking_action'],
            DockKMR.Goal(target_resource=target_resource),
            180.,
        )

    def observe_dock(transform):
        custody(transform)
        if math.dist(entity('KMR')[:2], machine['KMR_docking_pose'][:2]) > .035:
            raise ValueError(f'KMR docking position disagrees with {target_resource}')
        return dict(part_location='KMR', attached=True, arm_parked=True,
                    resource_location=target_resource, grasp_transform=transform,
                    carrying_arm_configuration=(request.get('custody') or {})['carrying_arm_configuration'])

    def transport_pose():
        offset = config['cartesian_waypoints']['transport_tcp_position_in_base_frame_m']
        target = compose(entity('KMR'), [*offset, 0., 0., 0., 1.])
        return [*target[:3], *config['pick_orientation_xyzw']]

    def transfer_waypoints(start, target):
        from cais_spade_llm.recovery_framework.kmr_motion import downward_transfer_waypoints

        mount = compose(entity('KMR'), [*kmr['arm_mount_xyz'], *quaternion(kmr['arm_mount_rpy'])])
        return downward_transfer_waypoints(start, target, mount, config['cartesian_waypoints'])

    def compute_place_targets(transform):
        if math.dist(entity('KMR')[:2], machine['KMR_docking_pose'][:2]) > .035:
            raise ValueError(f'KMR is not docked at {target_resource}')
        for peg in component_parts:
            if peg != part and math.dist(entity(peg)[:3], fixture[:3]) < .08:
                raise ValueError(f'{target_resource} workholding is occupied before placement')
        destination = [*fixture[:3], *quaternion(fixture[3:])]
        target = compose(destination, inverse(transform))
        target[2] += .003
        if rotate(target[3:], [0., 0., 1.])[2] > -.999:
            raise ValueError('KMR placement requires a downward-facing grasp')
        approach = target.copy()
        approach[2] += config['minimum_pick_lift_m']
        retreat = target.copy()
        retreat[2] += config['release_offsets_m'][0][2]
        transport = transport_pose()
        return {'target': target, 'approach': approach, 'retreat': retreat, 'destination': destination,
                'transfer_waypoints': transfer_waypoints(tcp(), approach), 'transport': transport,
                'withdraw_waypoints': transfer_waypoints(retreat, transport)}

    def observe_release(destination):
        measured = entity(part)
        if math.dist(measured[:3], destination[:3]) > .012:
            raise ValueError(f'Released part is not supported at {target_resource} workholding')
        if abs(sum(a*b for a, b in zip(measured[3:], destination[3:]))) < .995:
            raise ValueError(f'Released part orientation disagrees with {target_resource} workholding')
        if math.dist(tcp()[:3], measured[:3]) < .15:
            raise ValueError('KMR has not withdrawn from the released part')
        operations.append({'operation': 'observed_release', 'success': True,
                           'part_pose': measured, 'tcp_pose': tcp()})
        transport.publish(String(data=json.dumps({**probe, 'part_name': part, 'attached': False})))
        executor.spin_once(timeout_sec=.1)
        return dict(part_location=target_resource, attached=False, robot_clear=True)

    def compute_pick_targets(initial=None):
        observed = entity(part)
        if initial is not None and math.dist(observed[:3], initial[:3]) > .01:
            raise ValueError('Storage part moved before target computation')
        target = [*observed[:3], *config['pick_orientation_xyzw']]
        target[2] += config['grasp_height_m']
        approach = target.copy()
        approach[2] += scene['Storage']['KMR_pick_approach_clearance_m'][part]
        lift = target.copy()
        lift[2] += config['minimum_pick_lift_m']
        retreat = lift.copy()
        retreat[0] += config['approach_clearance_m']
        return {'target': target, 'approach': approach, 'lift': lift, 'retreat': retreat,
                'seed': list(scene['Storage']['KMR_pick_arm_configurations'][part]),
                'carrying': list(config['carrying_arm_configuration'])}

    def check_primitive():
        if stopped:
            raise InterruptedError('Stop System cancelled KMR execution')
        fresh_state(timeout=5.)

    try:
        response = service(GetParameters, '/KMR_base_controller/get_parameters', GetParameters.Request(names=list(_SCENE_IDENTITY_PARAMETERS)), 60., retry_read=True)
        probe = _scene_identity([value.string_value for value in response.values])
        if probe['scene_fingerprint'] != _expected_scene_fingerprint(
            scene, request['inputs']['product_order']
        ):
            raise ValueError('Running Gazebo scene is incompatible with the saved setup; reset Gazebo')
        asset_hashes = {path: hashlib.sha256((ROOT/'ros2/cais_lab_robotics'/path).read_bytes()).hexdigest()
                        for path in config['scene_assets']}
        if probe['scene_asset_fingerprint'] != fingerprint(asset_hashes):
            raise ValueError('Running Gazebo assets differ from the configured scene; run make bootstrap-gazebo and reset Gazebo')
        if request.get('probe') and any(probe[k] != request['probe'][k] for k in probe):
            raise ValueError('Gazebo was restarted during the delivery')
        prepared_turn = session.get('prepared_place_turn')
        if prepared_turn is not None and (
            prepared_turn.get('launch_id') != probe['launch_id']
            or prepared_turn.get('scene_fingerprint') != probe['scene_fingerprint']
        ):
            session.pop('prepared_place_turn', None)
        # The base controller's parameter service can precede the spawned arm
        # controllers. Startup may wait; task-time freshness requirements stay unchanged.
        fresh_state(timeout=120. if request['mode'] == 'probe' else 45.)
        initial = entity(part)
        base = entity('KMR')
        machine = next(
            row for row in scene['machines'] if row['resource_id'] == target_resource
        ) if target_resource in {'M1', 'M2'} else None
        if machine is None:
            machine = next(row for row in scene['machines'] if row['resource_id'] == 'M1')
        fixture = machine['workholding_pose']
        operations.append({
            'operation': 'observed_scene', 'success': True, 'part_pose': initial,
            'KMR_pose': base, f'{machine["resource_id"]}_pose': entity(machine['resource_id']),
        })
        if mode == 'probe' or (name == 'pick_part' and mode != 'primitive'):
            if math.dist(initial[:3], scene['Storage']['slots'][part][:3]) > .012:
                raise ValueError('Part is not in its initial Storage slot; explicitly reset Gazebo before repeating')
            storage_docks = [scene['Storage']['KMR_docking_pose'],
                             *scene['Storage']['KMR_pick_docking_poses'].values()]
            if not any(math.dist(base[:2], pose[:2]) <= .035 for pose in storage_docks):
                raise ValueError('KMR is not at a configured Storage dock')
            if abs(joint_values[kmr['gripper_joint']]-kmr['gripper_stroke_m']) > .006:
                raise ValueError('KMR gripper is not initially empty/open')
            if mode != 'environment_task':
                for peg in component_parts:
                    if math.dist(entity(peg)[:3], fixture[:3]) < .08:
                        raise ValueError(f'{machine["resource_id"]} workholding is occupied')
        if mode == 'environment_task' and name == 'pick_part':
            pick_pose = dict(
                scene['Storage'].get('KMR_pick_docking_poses') or {}
            ).get(part)
            if pick_pose is None:
                raise ValueError(f'No configured KMR Storage pickup dock for {part}')
            if math.dist(base[:2], pick_pose[:2]) > .01:
                move_to_pose(transport_pose(), waypoints=transfer_waypoints(tcp(), transport_pose()))
                authorize_empty_transport(route=[[base[0], base[1], scene['Storage']['KMR_docking_pose'][5]], list(pick_pose)])
                action(
                    DockKMR,
                    kmr['docking_action'],
                    DockKMR.Goal(target_resource=f'Storage/{part}'),
                    180.,
                )
                base = entity('KMR')
                if math.dist(base[:2], pick_pose[:2]) > .035:
                    raise ValueError(f'KMR did not reach the configured Storage pickup dock for {part}')
                operations.append({
                    'operation': 'observed_storage_pick_dock',
                    'success': True,
                    'part_name': part,
                    'KMR_pose': base,
                    'configured_pose': list(pick_pose),
                })
        if mode == 'probe':
            core_services = (
                (GetMotionPlan, config['services']['motion_plan']),
                (GetCartesianPath, config['services']['cartesian_path']),
                (GetPositionIK, '/compute_ik'),
                (GetStateValidity, '/check_state_validity'),
                (ApplyPlanningScene, config['services']['apply_scene']),
                (GetEntityState, config['services']['get_state']),
                (AttachLink, config['services']['attach']),
                (DetachLink, config['services']['detach']),
            )
            ready_services = []
            for kind, endpoint in core_services:
                if endpoint not in clients:
                    clients[endpoint] = node.create_client(kind, endpoint)
                spin_until(
                    clients[endpoint].service_is_ready,
                    60.,
                    f'{endpoint} discovery',
                )
                ready_services.append(endpoint)
            for kind, endpoint in (
                (FollowJointTrajectory, kmr['arm_controller']+'/follow_joint_trajectory'),
                (FollowJointTrajectory, kmr['gripper_controller']+'/follow_joint_trajectory'),
                (DockKMR, kmr['docking_action']),
                (NavigateToPose, '/KMR/navigate_to_pose'),
            ):
                if endpoint not in action_clients:
                    action_clients[endpoint] = ActionClient(node, kind, endpoint)
                client = action_clients[endpoint]
                spin_until(client.server_is_ready, 30.)
            operations.append({
                'operation': 'verified_core_services',
                'success': True,
                'services': ready_services,
                'actions': [
                    kmr['arm_controller']+'/follow_joint_trajectory',
                    kmr['gripper_controller']+'/follow_joint_trajectory',
                    kmr['docking_action'],
                    '/KMR/navigate_to_pose',
                ],
            })
        install_scene()
        if mode == 'probe':
            return {'status': 'completed', **probe, 'operations': operations, 'part_pose': initial,
                    'KMR_pose': base, 'timing': timings()}
        if mode == 'validate_pickup':
            targets = compute_pick_targets(initial)
            pose = scene['Storage']['KMR_pick_docking_poses'][part]
            state = updated_state(observed_state(),
                ['KMR_base_x_joint', 'KMR_base_y_joint', 'KMR_base_yaw_joint',
                 *kmr['arm_joint_names'], kmr['gripper_joint']],
                [*pose, *targets['seed'], kmr['gripper_stroke_m']])
            query = GetStateValidity.Request(group_name=config['planning_group'], robot_state=state)
            validity = service(GetStateValidity, '/check_state_validity', query, retry_read=True)
            if not validity.valid:
                raise ValueError(f'Configured pickup posture collides: {[(c.contact_body_1, c.contact_body_2) for c in validity.contacts]}')
            descend = plan_motion(state, target=targets['target'], cartesian=True)
            held = planned_attachment(state_after(state, descend), targets['target'], initial)
            lift = plan_motion(held, target=targets['lift'], cartesian=True)
            sweep = validate_transport(state_after(held, lift))
            return {'status': 'completed', 'part_name': part, 'neighbors_present': component_parts,
                    'targets': targets, 'descend': descend[1], 'lift': lift[1],
                    'transport': sweep, 'timing': timings()}
        if (
            mode == 'environment_task'
            and name == 'move_to_resource'
            and valuation.get('KMR', {}).get('held_part') is None
        ):
            route = route_between(source_resource, target_resource)
            authorize_empty_transport(route=route)
            operations.append({
                'operation': 'validated_empty_transport_route', 'success': True,
                'route': route,
                'arm_parked': True,
                'controller_acknowledgement': deepcopy(
                    base_motion_status['transport_custody_ack']
                ),
            })
            action(
                DockKMR,
                kmr['docking_action'],
                DockKMR.Goal(target_resource=target_resource),
                180.,
            )
            expected_pose = (
                scene['Storage']['KMR_docking_pose']
                if target_resource == 'Storage'
                else next(
                    row['KMR_docking_pose']
                    for row in scene['machines']
                    if row['resource_id'] == target_resource
                )
            )
            if math.dist(entity('KMR')[:2], expected_pose[:2]) > .035:
                raise ValueError(f'KMR empty return did not arrive at {target_resource}')
            home_observation = None
            if target_resource == 'Storage':
                home_record = {'resource_id': 'KMR', 'function_name': name,
                               'task_id': pending.get('task_id'), 'primitive': 'move_home',
                               'parameters': {}, 'status': 'running'}
                primitive_results.append(home_record)
                try:
                    home_observation = move_home()
                    home_record.update(status='completed', result=deepcopy(home_observation))
                except (RuntimeError, ValueError, TypeError, KeyError, TimeoutError, InterruptedError) as exc:
                    home_record.update(status='failed', error=str(exc))
                    raise
            observation = {
                **probe, 'part_name': None, 'operations': operations,
                'primitive_results': primitive_results, 'home_observation': home_observation,
                'controllers_succeeded': True, 'resource_location': target_resource,
                'attached': False, 'arm_parked': home_observation is None,
                'transport_arm_parked': True, 'timing': timings(),
            }
            return {
                'status': 'completed', 'resource_id': 'KMR', 'event_name': name,
                'task_id': pending.get('task_id'), 'observations': observation,
                'timing': observation['timing'],
            }
        observation = {**probe, 'part_name': part, 'operations': operations,
                       'primitive_results': primitive_results,
                       'composition': capability_decompositions(function_name=name)}
        if mode == 'environment_task':
            if name not in {'pick_part', 'move_to_resource', 'place_release'}:
                raise ValueError('Unsupported environmental KMR event')
            if name == 'pick_part' and parameters.get('origin_resource_location') != 'Storage':
                raise ValueError('KMR pickup must use Storage')
            if name == 'place_release' and parameters.get('destination_location') not in {'M1', 'M2'}:
                raise ValueError('KMR release must use M1 or M2')
        elif mode != 'primitive':
            expected = delivery_bindings(part)
            if (name not in expected or json.dumps(parameters, sort_keys=True)
                    != json.dumps(expected[name], sort_keys=True)):
                raise ValueError('Unsupported KMR delivery task or parameters')
        primitives = {
            'open_gripper': lambda: gripper(kmr['gripper_stroke_m']),
            'close_gripper': lambda: gripper(config['closed_gripper_width_m']),
            'compute_pick_targets': compute_pick_targets,
            'move_to_pose': move_to_pose, 'move_to_configuration': move_to_configuration,
            'move_home': move_home,
            'rotate_arm_base': rotate_arm_base,
            'observe_grasp': observe_grasp, 'attach_part': lambda: attach(True),
            'detach_part': lambda: attach(False), 'part_collision': part_collision,
            'custody': custody, 'confirm_carrying': confirm_carrying,
            'observe_carrying': observe_carrying, 'observe_custody': observe_custody,
            'validate_transport': lambda: operations.append(validate_transport(observed_state())),
            'dock': dock, 'observe_dock': observe_dock, 'compute_place_targets': compute_place_targets,
            'move_cartesian': move_cartesian,
            'observe_release': observe_release,
        }
        if mode == 'primitive':
            primitive = request.get('primitive')
            if primitive not in primitives:
                raise ValueError(f'Unknown KMR primitive: {primitive}')
            params = request.get('primitive_parameters', {})
            record = {'resource_id': 'KMR', 'primitive': primitive,
                      'parameters': deepcopy(params), 'status': 'running'}
            primitive_results.append(record)
            try:
                check_primitive()
                output = primitives[primitive](**params)
                record.update(status='completed', result=serialize(output))
            except (RuntimeError, ValueError, TypeError, KeyError, TimeoutError, InterruptedError) as exc:
                record.update(status='failed', error=str(exc))
                raise
            finally:
                record['observations'] = deepcopy(operations)
                record['timing'] = timings()
            return {'status': 'completed', 'resource_id': 'KMR', 'primitive': primitive,
                    'result': serialize(output), 'primitive_results': primitive_results,
                    'operations': operations, 'timing': timings()}
        observation.update(execute_composition(
            name, arguments=parameters,
            state={'initial': initial, 'previous': request.get('custody') or {},
                   'task_id': pending.get('task_id')},
            primitives=primitives, records=primitive_results, operations=operations,
            now=lambda: node.get_clock().now().nanoseconds/1e9,
            check=check_primitive, serialize=serialize, planning_time=lambda: planning_seconds,
        ))
        observation['controllers_succeeded'] = True
        observation['timing'] = timings()
        return {
            'status': 'completed', 'resource_id': 'KMR', 'event_name': name,
            'task_id': pending.get('task_id'), 'observations': observation,
            'timing': observation['timing'],
        }
    except (RuntimeError, ValueError, TypeError, KeyError, TimeoutError, InterruptedError) as exc:
        return {'status': 'failed', 'error': str(exc), 'operations': operations,
                'primitive_results': primitive_results, 'timing': timings(),
                'cancelled': stopped}
    finally:
        # Cancellation is a controller operation; no detach or inventory reset occurs.
        if active_goals or pending_goals:
            # MoveIt's ExecuteTrajectory callback can delay its own cancel
            # callback until execution returns. Stop the exclusively owned KMR
            # controllers first; they hold position without releasing the part.
            for controller in (kmr['arm_controller'], kmr['gripper_controller']):
                endpoint = controller+'/follow_joint_trajectory/_action/cancel_goal'
                client = node.create_client(CancelGoal, endpoint)
                response = None
                if client.wait_for_service(timeout_sec=1.):
                    for _ in range(2):
                        future = client.call_async(CancelGoal.Request())
                        deadline = time.monotonic() + 2.
                        while not future.done() and time.monotonic() < deadline:
                            executor.spin_once(timeout_sec=.05)
                        if future.done():
                            response = future.result()
                            break
                        client.remove_pending_request(future)
                operations.append({'operation': endpoint, 'success': response is not None
                                   and response.return_code == 0,
                                   'return_code': response.return_code if response else None,
                                   'goals_canceling': [bytes(goal.goal_id.uuid).hex()
                                                      for goal in response.goals_canceling] if response else []})
        for name, goal_id, pending in pending_goals:
            deadline = time.monotonic()+2.
            while not pending.done() and time.monotonic() < deadline:
                executor.spin_once(timeout_sec=.05)
            if pending.done() and pending.result().accepted:
                active_goals.append(pending.result())
            else:
                client = node.create_client(CancelGoal, name+'/_action/cancel_goal')
                if client.wait_for_service(timeout_sec=1.):
                    query = CancelGoal.Request()
                    query.goal_info.goal_id = goal_id
                    future = client.call_async(query)
                    deadline = time.monotonic() + 2.
                    while not future.done() and time.monotonic() < deadline:
                        executor.spin_once(timeout_sec=.05)
        for handle in active_goals:
            future = handle.cancel_goal_async()
            deadline = time.monotonic()+5.
            while not future.done() and time.monotonic() < deadline:
                executor.spin_once(timeout_sec=.05)
            response = future.result() if future.done() else None
            operations.append({'operation': 'cancel_goal',
                               'success': response is not None and bool(response.goals_canceling),
                               'return_code': response.return_code if response else None})
            result = handle.get_result_async()
            deadline = time.monotonic()+3.
            while not result.done() and time.monotonic() < deadline:
                executor.spin_once(timeout_sec=.05)
            terminal = result.result() if result.done() else None
            operations.append({'operation': 'observed_goal_after_stop',
                               'success': terminal is not None and terminal.status in (
                                   GoalStatus.STATUS_CANCELED, GoalStatus.STATUS_ABORTED),
                               'status': terminal.status if terminal else None})
        if active_goals:
            client = node.create_client(Trigger, '/KMR/cancel_base_motion')
            if client.wait_for_service(timeout_sec=1.):
                future = client.call_async(Trigger.Request())
                deadline = time.monotonic() + 3.
                while not future.done() and time.monotonic() < deadline:
                    executor.spin_once(timeout_sec=.05)
        if transport_timer is not None:
            transport_timer.cancel()
        signal.signal(signal.SIGINT, old_int)
        signal.signal(signal.SIGTERM, old_term)
        if not persistent:
            executor.remove_node(node)
            executor.shutdown()
            node.destroy_node()
            rclpy.shutdown()


def serve(directory: Path) -> None:
    """Keep this resource's ROS clients alive between serialized function calls."""
    import rclpy

    session = {}
    previous = None
    previous_file = None
    owner_pid = os.getppid()
    try:
        while True:
            if os.getppid() != owner_pid:
                break
            path = directory / 'request.json'
            try:
                stamp = path.stat()
            except FileNotFoundError:
                stamp = None
            identity = (stamp.st_ino, stamp.st_mtime_ns, stamp.st_size) if stamp else None
            if identity is not None and identity != previous_file:
                previous_file = identity
                envelope = json.loads(path.read_text())
                if envelope['id'] != previous:
                    previous = envelope['id']
                    result = run(envelope['request'], session)
                    temporary = directory / 'result.tmp'
                    temporary.write_text(json.dumps({'id': previous, 'result': result}, allow_nan=False))
                    temporary.replace(directory / 'result.json')
                    if result.get('cancelled'):
                        break
            if 'node' in session:
                session['executor'].spin_once(timeout_sec=.02)
                current = session['node'].get_clock().now().nanoseconds / 1e9
                if current < session.get('last_clock', 0.):
                    session['clock_reset'] = True
                    session['joint_samples'].clear()
                session['last_clock'] = current
            else:
                time.sleep(.02)
    except KeyboardInterrupt:
        pass
    finally:
        if 'node' in session:
            session['executor'].remove_node(session['node'])
            session['executor'].shutdown()
            session['node'].destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def main() -> None:
    """Exchange JSON with the owning ResourceAgent; never run from page rendering."""
    if len(sys.argv) == 3 and sys.argv[1] == '--serve':
        serve(Path(sys.argv[2]))
        return
    request_path, output_path = map(Path, sys.argv[1:])
    request = json.loads(request_path.read_text())
    result = run(request)
    output_path.write_text(json.dumps(result, indent=2), encoding='utf-8')
    if result['status'] != 'completed':
        _log.error('KMR delivery failed: %s', result['error'])


if __name__ == '__main__':
    main()
