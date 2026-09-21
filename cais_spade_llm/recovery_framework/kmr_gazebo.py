"""Cancellable ROS worker for observed KMR Storage-to-M1 task execution."""

from __future__ import annotations

import json
import hashlib
import logging
import math
import signal
import sys
import time
from collections import deque
from pathlib import Path
from uuid import uuid4

from cais_spade_llm.recovery_framework import ROOT, fingerprint
from cais_spade_llm.recovery_framework.geometry import collision_boxes, compose, quaternion, rotate

_log = logging.getLogger(__name__)


def inverse(pose: list[float]) -> list[float]:
    """Invert an xyz/xyzw rigid transform."""
    q = [-pose[3], -pose[4], -pose[5], pose[6]]
    return [*rotate(q, [-v for v in pose[:3]]), *q]


def run(request: dict) -> dict:
    """Run one explicit ROS operation; imports remain inside the worker."""
    import rclpy
    from action_msgs.msg import GoalStatus
    from action_msgs.srv import CancelGoal
    from builtin_interfaces.msg import Duration
    from control_msgs.action import FollowJointTrajectory
    from gazebo_msgs.srv import GetEntityState
    from geometry_msgs.msg import Pose, PoseStamped
    from linkattacher_msgs.srv import AttachLink, DetachLink
    from moveit_msgs.action import ExecuteTrajectory
    from moveit_msgs.msg import (
        AttachedCollisionObject, CollisionObject, Constraints, JointConstraint,
        OrientationConstraint, PlanningScene, PositionConstraint,
    )
    from moveit_msgs.srv import ApplyPlanningScene, GetCartesianPath, GetMotionPlan, GetStateValidity
    from rcl_interfaces.srv import GetParameters
    from rclpy.action import ActionClient
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

    rclpy.init(args=[])
    node = rclpy.create_node('KMR_delivery_worker', parameter_overrides=[Parameter('use_sim_time', value=True)])
    scene = request['inputs']['scene']
    kmr = scene['KMR']
    config = kmr['task_execution']
    part = config['part_name']
    component_parts = list(request['inputs']['geometry']['assembly_board']['slots'])
    operations = []
    joint_values = {}
    joint_times = {}
    joint_stamps = {}
    joint_samples = {}
    stopped = False
    active_goals = []
    pending_goals = []
    clients = {}
    transport_timer = None
    tf_buffer = Buffer()
    tf_listener = TransformListener(tf_buffer, node)

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

    subscription = node.create_subscription(JointState, '/joint_states', on_joints, qos_profile_sensor_data)
    transport = node.create_publisher(String, '/KMR/transport_custody', 10)

    def spin_until(predicate, timeout=30., label='observation'):
        if stopped:
            raise InterruptedError('Stop System cancelled KMR execution')
        deadline = time.monotonic() + timeout
        while not predicate():
            if stopped:
                raise InterruptedError('Stop System cancelled KMR execution')
            if time.monotonic() >= deadline:
                raise TimeoutError(f'KMR {label} timed out')
            rclpy.spin_once(node, timeout_sec=.05)

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

    def action(kind, name, goal, timeout=120.):
        client = ActionClient(node, kind, name)
        spin_until(client.server_is_ready, 30.)
        goal_id = UUID(uuid=list(uuid4().bytes))
        pending = client.send_goal_async(goal, goal_uuid=goal_id)
        entry = (name, goal_id, pending)
        pending_goals.append(entry)
        spin_until(pending.done, 10.)
        handle = pending.result()
        pending_goals.remove(entry)
        if not handle.accepted:
            raise RuntimeError(f'{name} rejected its goal')
        active_goals.append(handle)
        future = handle.get_result_async()
        spin_until(future.done, timeout)
        response = future.result()
        active_goals.remove(handle)
        code = getattr(response.result, 'error_code', None)
        operations.append({'operation': name, 'success': response.status == GoalStatus.STATUS_SUCCEEDED
                           and (code is None or getattr(code, 'val', code) in (0, 1))
                           and getattr(response.result, 'success', True),
                           'goal_id': bytes(goal_id.uuid).hex(), 'status': response.status,
                           'error_code': getattr(code, 'val', code),
                           'message': getattr(response.result, 'message', getattr(response.result, 'error_string', ''))})
        if response.status != GoalStatus.STATUS_SUCCEEDED:
            raise RuntimeError(f'{name} ended with action status {response.status}')
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

    def fresh_state():
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
        spin_until(select_feedback, 45., 'fresh joint feedback')
        state = JointState()
        state.header.stamp = node.get_clock().now().to_msg()
        state.name = list(joint_values)
        state.position = [joint_values[name] for name in state.name]
        return state

    def apply(planning_scene):
        response = service(ApplyPlanningScene, config['services']['apply_scene'], ApplyPlanningScene.Request(scene=planning_scene))
        if not response.success:
            raise RuntimeError('MoveIt rejected the collision scene')

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
        part_poses = {name: entity(name) for name in component_parts if name != part}
        boxes = collision_boxes(ROOT/'ros2/cais_lab_robotics/worlds/table_recovery_framework.world',
                                ROOT/'ros2/cais_lab_robotics/models', part_poses)
        planning_scene.world.collision_objects = [box_object(row['id'], row['pose'], row['size']) for row in boxes]
        apply(planning_scene)
        operations.append({'operation': 'apply_planning_scene', 'success': True, 'collision_objects': len(boxes)})

    def execute(trajectory):
        if not trajectory.joint_trajectory.points:
            raise ValueError('MoveIt returned an empty trajectory')
        action(ExecuteTrajectory, '/execute_trajectory', ExecuteTrajectory.Goal(trajectory=trajectory))

    def move(target=None, joints=None, cartesian=False):
        state = fresh_state()
        if target is not None:
            measured = tcp()
            if math.dist(measured[:3], target[:3]) < .002 and abs(sum(a*b for a,b in zip(measured[3:], target[3:]))) > .9999:
                query = GetStateValidity.Request(group_name=config['planning_group'])
                query.robot_state.joint_state = state
                query.robot_state.is_diff = True
                if not service(GetStateValidity, '/check_state_validity', query).valid:
                    raise ValueError('Current KMR target pose is in collision')
                operations.append({'operation': 'observed_target_pose', 'success': True, 'tcp_pose': measured})
                return
        if cartesian:
            query = GetCartesianPath.Request()
            query.header.frame_id = 'world'
            query.group_name = config['planning_group']
            query.link_name = config['tcp_link']
            query.start_state.joint_state = state
            query.start_state.is_diff = True
            query.waypoints = [pose_message(target)]
            query.max_step = .005
            query.jump_threshold = 2.
            query.avoid_collisions = True
            result = service(GetCartesianPath, config['services']['cartesian_path'], query)
            operations.append({'operation': 'cartesian_path', 'success': result.error_code.val == 1
                               and result.fraction >= .999, 'target': target,
                               'avoid_collisions': True, 'fraction': result.fraction,
                               'error_code': result.error_code.val})
            if result.error_code.val != 1 or result.fraction < .999:
                raise ValueError(f'Collision-aware Cartesian path incomplete: {result.fraction}')
            trajectory = result.solution
            # Cartesian service does not expose velocity scaling in Humble.
            for point in trajectory.joint_trajectory.points:
                seconds = (point.time_from_start.sec + point.time_from_start.nanosec/1e9)/config['velocity_scaling']
                point.time_from_start = Duration(sec=int(seconds), nanosec=int((seconds%1)*1e9))
                point.velocities = [v*config['velocity_scaling'] for v in point.velocities]
                point.accelerations = [v*config['velocity_scaling']**2 for v in point.accelerations]
        else:
            query = GetMotionPlan.Request()
            motion = query.motion_plan_request
            motion.group_name = config['planning_group']
            motion.start_state.joint_state = state
            motion.start_state.is_diff = True
            motion.allowed_planning_time = 10.
            motion.num_planning_attempts = 8
            motion.max_velocity_scaling_factor = config['velocity_scaling']
            motion.max_acceleration_scaling_factor = config['acceleration_scaling']
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
            response = service(GetMotionPlan, config['services']['motion_plan'], query, 30.).motion_plan_response
            operations.append({'operation': 'motion_plan', 'success': response.error_code.val == 1,
                               'target': target, 'joint_target': joints, 'avoid_collisions': True,
                               'error_code': response.error_code.val})
            if response.error_code.val != 1:
                raise ValueError(f'Collision-aware KMR motion planning failed: {response.error_code.val}')
            trajectory = response.trajectory
        operations[-1]['trajectory'] = {
            'joint_names': list(trajectory.joint_trajectory.joint_names),
            'points': [{'positions': list(point.positions), 'velocities': list(point.velocities),
                        'accelerations': list(point.accelerations),
                        'time_from_start': point.time_from_start.sec+point.time_from_start.nanosec/1e9}
                       for point in trajectory.joint_trajectory.points],
        }
        execute(trajectory)
        if target is not None:
            spin_until(lambda: math.dist(tcp()[:3], target[:3]) <= config['position_tolerance_m'],
                       5., 'TCP feedback at requested pose')

    def gripper(width):
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = [kmr['gripper_joint']]
        goal.trajectory.points = [JointTrajectoryPoint(positions=[float(width)], time_from_start=Duration(sec=1))]
        action(FollowJointTrajectory, kmr['gripper_controller']+'/follow_joint_trajectory', goal)
        def reached():
            fresh_state()
            return abs(joint_values[kmr['gripper_joint']]-width) <= .006
        spin_until(reached, 5., 'gripper feedback at requested width')
        operations.append({'operation': 'observed_gripper', 'success': True,
                           'target': width, 'position': joint_values[kmr['gripper_joint']],
                           'stamp': joint_stamps[kmr['gripper_joint']]})

    def park():
        move(joints=kmr['parked_arm_configuration'])
        def reached():
            fresh_state()
            return all(abs(joint_values[name]-target) <= .02
                       for name, target in zip(kmr['arm_joint_names'], kmr['parked_arm_configuration']))
        spin_until(reached, 5., 'arm feedback at parked configuration')
        operations.append({'operation': 'observed_parked', 'success': True,
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

    try:
        response = service(GetParameters, '/KMR_base_controller/get_parameters', GetParameters.Request(names=['launch_id', 'scene_fingerprint', 'scene_asset_fingerprint']), 60., retry_read=True)
        launch_id, scene_fingerprint, scene_asset_fingerprint = [value.string_value for value in response.values]
        if not launch_id or scene_fingerprint != fingerprint(scene):
            raise ValueError('Running Gazebo scene is incompatible with the saved setup; reset Gazebo')
        asset_hashes = {path: hashlib.sha256((ROOT/'ros2/cais_lab_robotics'/path).read_bytes()).hexdigest()
                        for path in config['scene_assets']}
        if scene_asset_fingerprint != fingerprint(asset_hashes):
            raise ValueError('Running Gazebo assets differ from the configured scene; run make bootstrap-gazebo and reset Gazebo')
        probe = {'launch_id': launch_id, 'scene_fingerprint': scene_fingerprint,
                 'scene_asset_fingerprint': scene_asset_fingerprint}
        if request.get('probe') and any(probe[k] != request['probe'][k] for k in probe):
            raise ValueError('Gazebo was restarted during the delivery')
        fresh_state()
        initial = entity(part)
        base = entity('KMR')
        machine = next(row for row in scene['machines'] if row['resource_id'] == 'M1')
        fixture = machine['workholding_pose']
        operations.append({'operation': 'observed_scene', 'success': True, 'part_pose': initial, 'KMR_pose': base, 'M1_pose': entity('M1')})
        mode = request['mode']
        name = request.get('pending', {}).get('event_name')
        if mode == 'probe' or name == 'pick_part':
            if math.dist(initial[:3], scene['Storage']['slots'][part][:3]) > .012:
                raise ValueError('Part is not in its initial Storage slot; explicitly reset Gazebo before repeating')
            if math.dist(base[:2], kmr['initial_pose'][:2]) > .035:
                raise ValueError('KMR is not at the configured Storage dock')
            if abs(joint_values[kmr['gripper_joint']]-kmr['gripper_stroke_m']) > .006:
                raise ValueError('KMR gripper is not initially empty/open')
            for peg in component_parts:
                if math.dist(entity(peg)[:3], fixture[:3]) < .08:
                    raise ValueError('M1 workholding is occupied')
        if mode == 'probe':
            for kind, endpoint in (
                (ExecuteTrajectory, '/execute_trajectory'),
                (FollowJointTrajectory, kmr['arm_controller']+'/follow_joint_trajectory'),
                (FollowJointTrajectory, kmr['gripper_controller']+'/follow_joint_trajectory'),
                (DockKMR, kmr['docking_action']),
            ):
                client = ActionClient(node, kind, endpoint)
                spin_until(client.server_is_ready, 30.)
        install_scene()
        if mode == 'probe':
            return {'status': 'completed', **probe, 'operations': operations, 'part_pose': initial, 'KMR_pose': base}
        observation = {**probe, 'part_name': part, 'operations': operations}
        if name == 'pick_part':
            gripper(kmr['gripper_stroke_m'])
            target = [initial[0], initial[1], initial[2]+config['grasp_height_m'], *config['pick_orientation_xyzw']]
            approach = target.copy()
            approach[0] += config['approach_clearance_m']
            # This commissioned branch admits the entire straight grasp path;
            # an arbitrary approach IK solution can reach a joint limit midway.
            move(joints=config['pick_approach_configuration'])
            move(target=approach, cartesian=True)
            move(target=target, cartesian=True)
            gripper(config['closed_gripper_width_m'])
            before = entity(part)
            if math.dist(before[:3], initial[:3]) > .01:
                raise ValueError('Part moved before the simulated grasp attachment')
            transform = compose(inverse(tcp()), before)
            attach(True)
            part_collision(True)
            lift = target.copy()
            lift[2] += .035
            move(target=lift, cartesian=True)
            custody(transform)
            retreat = lift.copy()
            retreat[0] += config['approach_clearance_m']
            move(target=retreat, cartesian=True)
            retreat[2] += config['withdrawal_lift_m']
            move(target=retreat, cartesian=True)
            park()
            custody(transform)
            observation.update(part_location='KMR', attached=True, arm_parked=True, grasp_transform=transform)
        elif name in ('move_to_resource', 'place_release'):
            previous = request.get('custody') or {}
            transform = previous.get('grasp_transform')
            if previous.get('attached') is not True or not isinstance(transform, list) or len(transform) != 7:
                raise ValueError('No acknowledged grasp transform for the held part')
            custody(transform)
            if name == 'move_to_resource':
                def publish_custody():
                    transport.publish(String(data=json.dumps({**probe, 'part_name': part, 'attached': True})))
                transport_timer = node.create_timer(.1, publish_custody)
                publish_custody()
                action(DockKMR, kmr['docking_action'], DockKMR.Goal(target_resource='M1'), 180.)
                custody(transform)
                if math.dist(entity('KMR')[:2], machine['KMR_docking_pose'][:2]) > .035:
                    raise ValueError('KMR docking position disagrees with M1')
                observation.update(part_location='KMR', attached=True, arm_parked=True, resource_location='M1', grasp_transform=transform)
            else:
                if math.dist(base[:2], machine['KMR_docking_pose'][:2]) > .035:
                    raise ValueError('KMR is not docked at M1')
                for peg in component_parts:
                    if peg != part and math.dist(entity(peg)[:3], fixture[:3]) < .08:
                        raise ValueError('M1 workholding is occupied before placement')
                destination = [*fixture[:3], *quaternion(fixture[3:])]
                # Preserve the measured part-to-TCP transform during ordinary placement.
                target = compose(destination, inverse(transform))
                target[2] += .003
                approach = target.copy()
                approach[0] -= config['approach_clearance_m']
                approach[2] += .035
                move(target=approach)
                above = target.copy()
                above[2] += .035
                move(target=above, cartesian=True)
                move(target=target, cartesian=True)
                custody(transform)
                gripper(kmr['gripper_stroke_m'])
                attach(False)
                # Keep the measured released part in the planning scene.
                part_collision(False)
                retreat = target.copy()
                retreat[0] -= config['approach_clearance_m']
                move(target=retreat, cartesian=True)
                park()
                measured = entity(part)
                if math.dist(measured[:3], destination[:3]) > .012:
                    raise ValueError('Released part is not supported at M1 workholding')
                if abs(sum(a*b for a, b in zip(measured[3:], destination[3:]))) < .995:
                    raise ValueError('Released part orientation disagrees with M1 workholding')
                if math.dist(tcp()[:3], measured[:3]) < .15:
                    raise ValueError('KMR has not withdrawn from the released part')
                operations.append({'operation': 'observed_release', 'success': True, 'part_pose': measured, 'tcp_pose': tcp()})
                observation.update(part_location='M1', attached=False, robot_clear=True)
                transport.publish(String(data=json.dumps({**probe, 'part_name': part, 'attached': False})))
                rclpy.spin_once(node, timeout_sec=.1)
        else:
            raise ValueError('Unsupported KMR delivery task')
        observation['controllers_succeeded'] = True
        return {'status': 'completed', 'observations': observation}
    except (RuntimeError, ValueError, TimeoutError, InterruptedError) as exc:
        return {'status': 'failed', 'error': str(exc), 'operations': operations}
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
                        rclpy.spin_until_future_complete(node, future, timeout_sec=2.)
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
                rclpy.spin_once(node, timeout_sec=.05)
            if pending.done() and pending.result().accepted:
                active_goals.append(pending.result())
            else:
                client = node.create_client(CancelGoal, name+'/_action/cancel_goal')
                if client.wait_for_service(timeout_sec=1.):
                    query = CancelGoal.Request()
                    query.goal_info.goal_id = goal_id
                    future = client.call_async(query)
                    rclpy.spin_until_future_complete(node, future, timeout_sec=2.)
        for handle in active_goals:
            future = handle.cancel_goal_async()
            deadline = time.monotonic()+5.
            while not future.done() and time.monotonic() < deadline:
                rclpy.spin_once(node, timeout_sec=.05)
            response = future.result() if future.done() else None
            operations.append({'operation': 'cancel_goal',
                               'success': response is not None and bool(response.goals_canceling),
                               'return_code': response.return_code if response else None})
            result = handle.get_result_async()
            deadline = time.monotonic()+3.
            while not result.done() and time.monotonic() < deadline:
                rclpy.spin_once(node, timeout_sec=.05)
            terminal = result.result() if result.done() else None
            operations.append({'operation': 'observed_goal_after_stop',
                               'success': terminal is not None and terminal.status in (
                                   GoalStatus.STATUS_CANCELED, GoalStatus.STATUS_ABORTED),
                               'status': terminal.status if terminal else None})
        if active_goals:
            client = node.create_client(Trigger, '/KMR/cancel_base_motion')
            if client.wait_for_service(timeout_sec=1.):
                future = client.call_async(Trigger.Request())
                rclpy.spin_until_future_complete(node, future, timeout_sec=3.)
        if transport_timer is not None:
            transport_timer.cancel()
        signal.signal(signal.SIGINT, old_int)
        signal.signal(signal.SIGTERM, old_term)
        node.destroy_node()
        rclpy.shutdown()


def main() -> None:
    """Exchange JSON with the owning ResourceAgent; never run from page rendering."""
    request_path, output_path = map(Path, sys.argv[1:])
    request = json.loads(request_path.read_text())
    result = run(request)
    output_path.write_text(json.dumps(result, indent=2), encoding='utf-8')
    if result['status'] != 'completed':
        _log.error('KMR delivery failed: %s', result['error'])


if __name__ == '__main__':
    main()
