#!/usr/bin/env python3.10
"""
Keyboard teleop for dual xArm6 + UR5e via MoveIt.

Each keypress immediately plans + executes a move.
Robot moves visibly in both RViz and Gazebo.

Usage:
    source /opt/ros/humble/setup.bash && source ~/ros2_ws/install/setup.bash
    python3.10 keyboard_teleop.py

Controls — Cartesian mode (default):
    Z + Arrow UP/DOWN  — move Z up / down
    X + Arrow LEFT/RIGHT — both robots: X+/X-
    Y + Arrow LEFT/RIGHT — both robots: Y-/Y+
    Arrow alone         — jog last-selected axis (Z uses UP/DOWN, X/Y use LEFT/RIGHT)
    +/-                — change step size (default 10mm)

Controls — Joint mode:
    1-6                — select joint (switches to joint mode)
    Arrow UP/DOWN      — jog selected joint + / -
    +/-                — change step size (default 1 deg)

Controls — Gripper mode:
    G                  — switch to gripper mode
    Arrow UP/DOWN      — open / close current robot gripper
    +/-                — change gripper step

Common:
    M                  — switch to Cartesian mode
    TAB                — switch robot (xArm6 / UR5e)
    H                  — move both arms to saved "home" position
    P                  — precision profile
    F                  — fast profile
    S                  — save current position (prompts for name)
    Q                  — quit
"""

import argparse
import copy
import json
import math
import os
import select as sel_mod
import sys
import termios
import threading
import time
import tty
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup

from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Pose
from moveit_msgs.action import ExecuteTrajectory
from moveit_msgs.srv import GetCartesianPath

import tf2_ros

try:
    from control_msgs.action import GripperCommand
except Exception:
    GripperCommand = None

try:
    from xarm_msgs.srv import GripperMove, SetFloat32, SetInt16
except Exception:
    GripperMove = None
    SetFloat32 = None
    SetInt16 = None

# ── Robot definitions ────────────────────────────────────────────────────────

ROBOTS = {
    'xarm6': {
        # Primary entries target dual-robot Gazebo naming (prefixed).
        # Candidates include real-hardware single-robot naming (unprefixed).
        'joint_names': [
            'xarm6_joint1', 'xarm6_joint2', 'xarm6_joint3',
            'xarm6_joint4', 'xarm6_joint5', 'xarm6_joint6',
        ],
        'joint_name_candidates': [
            ['xarm6_joint1', 'xarm6_joint2', 'xarm6_joint3',
             'xarm6_joint4', 'xarm6_joint5', 'xarm6_joint6'],
            ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6'],
        ],
        'group_name': 'xarm6_xarm6',
        'group_name_candidates': ['xarm6_xarm6', 'xarm6', 'xarm'],
        'ee_link': 'xarm6_link_eef',
        'ee_link_candidates': ['xarm6_link_eef', 'link_eef', 'xarm6_link_tcp', 'link_tcp'],
        'frame_id': 'world',
        'frame_id_candidates': ['world', 'xarm6_link_base', 'xarm6_base_link', 'link_base', 'base_link'],
        'arm_controller_topic': '/xarm6_xarm6_traj_controller/joint_trajectory',
        'arm_controller_topics': [
            '/xarm6_xarm6_traj_controller/joint_trajectory',
            '/xarm6_traj_controller/joint_trajectory',
            '/xarm_traj_controller/joint_trajectory',
        ],
        'gripper_joint': 'xarm6_drive_joint',
        'gripper_joint_candidates': ['xarm6_drive_joint', 'drive_joint'],
        'gripper_controller_topic': '/xarm6_xarm_gripper_traj_controller/joint_trajectory',
        'gripper_controller_topics': [
            '/xarm6_xarm_gripper_traj_controller/joint_trajectory',
            '/xarm_gripper_traj_controller/joint_trajectory',
        ],
        'gripper_open': 0.0,
        'gripper_close': 0.85,
        'gripper_step_default': 0.10,
        'gripper_step_min': 0.001,
        'gripper_step_max': 0.2,
    },
    'ur5e': {
        'joint_names': [
            'ur5e_shoulder_pan_joint', 'ur5e_shoulder_lift_joint',
            'ur5e_elbow_joint', 'ur5e_wrist_1_joint',
            'ur5e_wrist_2_joint', 'ur5e_wrist_3_joint',
        ],
        'joint_name_candidates': [
            ['ur5e_shoulder_pan_joint', 'ur5e_shoulder_lift_joint',
             'ur5e_elbow_joint', 'ur5e_wrist_1_joint',
             'ur5e_wrist_2_joint', 'ur5e_wrist_3_joint'],
            ['shoulder_pan_joint', 'shoulder_lift_joint',
             'elbow_joint', 'wrist_1_joint',
             'wrist_2_joint', 'wrist_3_joint'],
        ],
        'group_name': 'ur5e_ur_manipulator',
        'group_name_candidates': ['ur5e_ur_manipulator', 'ur_manipulator'],
        'ee_link': 'ur5e_tool0',
        'ee_link_candidates': ['ur5e_tool0', 'tool0'],
        'frame_id': 'world',
        'frame_id_candidates': ['world', 'ur5e_base_link', 'base_link'],
        'arm_controller_topic': '/ur5e_joint_trajectory_controller/joint_trajectory',
        'arm_controller_topics': [
            '/ur5e_joint_trajectory_controller/joint_trajectory',
            '/scaled_joint_trajectory_controller/joint_trajectory',
            '/joint_trajectory_controller/joint_trajectory',
        ],
        'gripper_joint': 'ur5e_rg2_finger_width',
        'gripper_joint_candidates': ['ur5e_rg2_finger_width', 'rg2_finger_width'],
        'gripper_controller_topic': '/ur5e_rg2_gripper_traj_controller/joint_trajectory',
        'gripper_controller_topics': [
            '/ur5e_rg2_gripper_traj_controller/joint_trajectory',
            '/rg2_gripper_traj_controller/joint_trajectory',
        ],
        'gripper_open': 0.11,
        'gripper_close': 0.02,
        'gripper_step_default': 0.02,
        'gripper_step_min': 0.0005,
        'gripper_step_max': 0.03,
    },
}

DEFAULT_CONFIG_PATHS = {
    'xarm6': Path(__file__).resolve().parents[3]
             / 'cais_spade_llm' / 'initialization' / 'resources' / 'robot_xarm6.json',
    'ur5e': Path(__file__).resolve().parents[3]
            / 'cais_spade_llm' / 'initialization' / 'resources' / 'robot_ur5e.json',
}
# Home safety tuning:
# - Always try a small upward (+Z) pre-lift before home for collision margin.
# - Keep pre-lift short and fast to reduce overall home latency.
HOME_PRELIFT_DELTA_M = 0.05
HOME_PRELIFT_MAX_DELTA_M = 0.10
HOME_PRELIFT_GAZEBO_MIN_Z_M = 1.12
HOME_PRELIFT_VELOCITY_SCALE = 1.8


class KeyboardTeleop(Node):
    def __init__(self, cartesian_max_step_mm=30.0, joint_duration_sec=0.25, gripper_duration_sec=0.20):
        super().__init__('keyboard_teleop')
        self.cb_group = ReentrantCallbackGroup()
        self.joint_positions = {}
        self.joint_state_map = {}
        self.cartesian_max_step_m = max(0.001, cartesian_max_step_mm / 1000.0)
        self.joint_duration_sec = max(0.05, float(joint_duration_sec))
        self.gripper_duration_sec = max(0.05, float(gripper_duration_sec))

        self.create_subscription(JointState, '/joint_states', self._joint_state_cb, 10)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.execute_client = ActionClient(
            self, ExecuteTrajectory, '/execute_trajectory', callback_group=self.cb_group)
        self.cartesian_client = self.create_client(
            GetCartesianPath, '/compute_cartesian_path', callback_group=self.cb_group)
        self.arm_publishers = {}
        self.gripper_publishers = {}
        self.active_joint_names = {}
        self.active_group_name = {}
        self.active_gripper_joint = {}
        self.active_ee_link = {}
        self.active_frame_id = {}
        self._service_client_cache = {}
        self._action_client_cache = {}
        self._xarm_gripper_configured = False
        self._xarm_gripper_speed = None
        for robot_name, cfg in ROBOTS.items():
            arm_topics = cfg.get('arm_controller_topics') or [cfg['arm_controller_topic']]
            self.arm_publishers[robot_name] = {
                topic: self.create_publisher(JointTrajectory, topic, 10)
                for topic in arm_topics
            }
            gripper_topics = cfg.get('gripper_controller_topics') or [cfg['gripper_controller_topic']]
            self.gripper_publishers[robot_name] = {
                topic: self.create_publisher(JointTrajectory, topic, 10)
                for topic in gripper_topics
            }

    @staticmethod
    def _joint_name_candidates(robot):
        cfg = ROBOTS[robot]
        return cfg.get('joint_name_candidates') or [cfg['joint_names']]

    @staticmethod
    def _group_name_candidates(robot):
        cfg = ROBOTS[robot]
        return cfg.get('group_name_candidates') or [cfg['group_name']]

    @staticmethod
    def _ee_link_candidates(robot):
        cfg = ROBOTS[robot]
        return cfg.get('ee_link_candidates') or [cfg['ee_link']]

    @staticmethod
    def _frame_id_candidates(robot):
        cfg = ROBOTS[robot]
        return cfg.get('frame_id_candidates') or [cfg['frame_id']]

    @staticmethod
    def _gripper_joint_candidates(robot):
        cfg = ROBOTS[robot]
        return cfg.get('gripper_joint_candidates') or [cfg['gripper_joint']]

    def _detect_gripper_joint(self, robot):
        """Best-effort detection of active gripper joint across sim/hardware naming."""
        for joint in self._gripper_joint_candidates(robot):
            if joint in self.joint_state_map:
                return joint

        if robot == 'xarm6':
            preferred_tokens = ('drive_joint', 'gripper_joint')
        else:
            preferred_tokens = ('finger_width', 'gripper_joint')

        for joint in self.joint_state_map.keys():
            if any(token in joint for token in preferred_tokens):
                return joint
        return None

    def _current_joint_names(self, robot):
        return list(self.active_joint_names.get(robot, self._joint_name_candidates(robot)[0]))

    def _current_group_name(self, robot):
        return str(self.active_group_name.get(robot, self._group_name_candidates(robot)[0]))

    def _current_ee_link(self, robot):
        return str(self.active_ee_link.get(robot, self._ee_link_candidates(robot)[0]))

    def _current_frame_id(self, robot):
        return str(self.active_frame_id.get(robot, self._frame_id_candidates(robot)[0]))

    def _current_gripper_joint(self, robot):
        if robot in self.active_gripper_joint:
            return self.active_gripper_joint[robot]
        detected = self._detect_gripper_joint(robot)
        if detected:
            self.active_gripper_joint[robot] = detected
            return detected
        return self._gripper_joint_candidates(robot)[0]

    @staticmethod
    def _pick_publisher(pubs):
        # Prefer a publisher with active subscribers.
        for topic, pub in pubs.items():
            if pub.get_subscription_count() > 0:
                return pub, topic
        first_topic = next(iter(pubs))
        return pubs[first_topic], first_topic

    def _candidate_service_names(self, suffix):
        names = [f'/xarm/{suffix}', f'/{suffix}']
        try:
            for name, _types in self.get_service_names_and_types():
                if name == f'/{suffix}' or name.endswith(f'/{suffix}'):
                    names.append(name)
        except Exception:
            pass
        deduped = []
        for name in names:
            if name not in deduped:
                deduped.append(name)
        return deduped

    def _get_service_client(self, srv_type, suffix, wait_timeout_sec=0.05):
        key = (str(suffix), getattr(srv_type, '__name__', str(srv_type)))
        clients = self._service_client_cache.setdefault(key, {})

        for service_name in self._candidate_service_names(suffix):
            if service_name in clients:
                continue
            try:
                clients[service_name] = self.create_client(
                    srv_type, service_name, callback_group=self.cb_group)
            except Exception:
                continue

        for service_name, client in clients.items():
            try:
                if client.service_is_ready():
                    return service_name, client
            except Exception:
                continue

        for service_name, client in clients.items():
            try:
                if client.wait_for_service(timeout_sec=wait_timeout_sec):
                    return service_name, client
            except Exception:
                continue
        return None, None

    def _call_service(self, client, request, timeout_sec=2.0):
        try:
            future = client.call_async(request)
        except Exception as exc:
            return None, str(exc)
        if not self._wait_future(future, timeout=timeout_sec):
            return None, 'timeout'
        response = future.result()
        if response is None:
            return None, 'service failed'
        return response, None

    def _candidate_action_names(self, suffix):
        names = [f'/{suffix}', suffix]
        try:
            for name, _types in self.get_action_names_and_types():
                if name == f'/{suffix}' or name == suffix or name.endswith(f'/{suffix}'):
                    names.append(name)
        except Exception:
            pass
        deduped = []
        for name in names:
            if name not in deduped:
                deduped.append(name)
        return deduped

    def _get_action_client(self, action_type, suffix, wait_timeout_sec=0.05):
        key = (str(suffix), getattr(action_type, '__name__', str(action_type)))
        clients = self._action_client_cache.setdefault(key, {})

        for action_name in self._candidate_action_names(suffix):
            if action_name in clients:
                continue
            try:
                clients[action_name] = ActionClient(
                    self, action_type, action_name, callback_group=self.cb_group)
            except Exception:
                continue

        for action_name, client in clients.items():
            try:
                if client.server_is_ready():
                    return action_name, client
            except Exception:
                continue

        for action_name, client in clients.items():
            try:
                if client.wait_for_server(timeout_sec=wait_timeout_sec):
                    return action_name, client
            except Exception:
                continue
        return None, None

    @staticmethod
    def _xarm_joint_to_pulse(joint_position):
        open_pos = ROBOTS['xarm6']['gripper_open']
        close_pos = ROBOTS['xarm6']['gripper_close']
        denom = (open_pos - close_pos)
        if abs(denom) < 1e-9:
            return 0
        open_ratio = (joint_position - close_pos) / denom
        open_ratio = min(max(open_ratio, 0.0), 1.0)
        return int(round(850.0 * open_ratio))

    def _ensure_xarm_gripper_ready(self):
        if SetInt16 is None or self._xarm_gripper_configured:
            return
        for suffix, value in (('set_gripper_enable', 1), ('set_gripper_mode', 0)):
            _service_name, client = self._get_service_client(SetInt16, suffix, wait_timeout_sec=0.05)
            if client is None:
                continue
            req = SetInt16.Request()
            req.data = int(value)
            self._call_service(client, req, timeout_sec=1.0)
        self._xarm_gripper_configured = True

    def _set_xarm_gripper_speed(self, velocity_scale):
        if SetFloat32 is None:
            return
        speed = int(self._clamp(round(2000.0 * self._normalize_velocity_scale(velocity_scale)), 1, 5000))
        if self._xarm_gripper_speed == speed:
            return
        _service_name, client = self._get_service_client(SetFloat32, 'set_gripper_speed', wait_timeout_sec=0.05)
        if client is None:
            return
        req = SetFloat32.Request()
        req.data = float(speed)
        response, error = self._call_service(client, req, timeout_sec=1.0)
        if error is not None:
            return
        if int(getattr(response, 'ret', 0)) == 0:
            self._xarm_gripper_speed = speed

    def _move_xarm_gripper_service(self, target_joint, velocity_scale):
        if GripperMove is None:
            return False, 'xarm gripper service unavailable (xarm_msgs not found)'

        self._ensure_xarm_gripper_ready()
        self._set_xarm_gripper_speed(velocity_scale)

        service_name, client = self._get_service_client(GripperMove, 'set_gripper_position', wait_timeout_sec=0.1)
        if client is None:
            return False, 'xarm gripper service not available'

        pulse = self._xarm_joint_to_pulse(target_joint)
        req = GripperMove.Request()
        req.pos = float(pulse)
        req.wait = False
        req.timeout = 2.0

        response, error = self._call_service(client, req, timeout_sec=2.0)
        if error is not None:
            return False, f'{service_name}: {error}'

        ret = int(getattr(response, 'ret', -1))
        msg = str(getattr(response, 'message', '')).strip()
        if ret != 0:
            details = f'ret={ret}'
            if msg:
                details += f' {msg}'
            return False, f'{service_name}: {details}'
        return True, f'{service_name}: pos={pulse}'

    def _move_xarm_gripper_action(self, target_joint):
        if GripperCommand is None:
            return False, 'gripper action type unavailable'
        action_name, client = self._get_action_client(
            GripperCommand, 'xarm_gripper/gripper_action', wait_timeout_sec=0.1)
        if client is None:
            return False, 'xarm gripper action server not available'

        goal = GripperCommand.Goal()
        goal.command.position = float(target_joint)
        goal.command.max_effort = 0.0

        try:
            send_future = client.send_goal_async(goal)
        except Exception as exc:
            return False, f'{action_name}: send failed ({exc})'
        if not self._wait_future(send_future, timeout=1.5):
            return False, f'{action_name}: send timeout'

        goal_handle = send_future.result()
        if goal_handle is None or not goal_handle.accepted:
            return False, f'{action_name}: goal rejected'

        # Fast UI response: treat accepted goals as success and only check early failures.
        result_future = goal_handle.get_result_async()
        if self._wait_future(result_future, timeout=0.4):
            wrapped = result_future.result()
            result = getattr(wrapped, 'result', None)
            stalled = bool(getattr(result, 'stalled', False))
            reached = bool(getattr(result, 'reached_goal', True))
            if stalled and not reached:
                return False, f'{action_name}: stalled before reaching goal'
        return True, f'{action_name}: goal accepted'

    def _joint_state_cb(self, msg):
        for jname, pos in zip(msg.name, msg.position):
            self.joint_state_map[jname] = pos
        for robot_name, cfg in ROBOTS.items():
            for joint_names in self._joint_name_candidates(robot_name):
                positions = {}
                for jname, pos in zip(msg.name, msg.position):
                    if jname in joint_names:
                        positions[jname] = pos
                if len(positions) == len(joint_names):
                    self.joint_positions[robot_name] = [positions[n] for n in joint_names]
                    self.active_joint_names[robot_name] = list(joint_names)
                    break
            for gj in self._gripper_joint_candidates(robot_name):
                if gj in self.joint_state_map:
                    self.active_gripper_joint[robot_name] = gj
                    break

    @staticmethod
    def _duration_msg(seconds):
        sec = int(seconds)
        nanosec = int((seconds - sec) * 1e9)
        return Duration(sec=sec, nanosec=nanosec)

    @staticmethod
    def _clamp(value, min_value, max_value):
        return min(max(value, min_value), max_value)

    @staticmethod
    def _normalize_velocity_scale(scale, minimum=0.1, maximum=3.0):
        try:
            value = float(scale)
        except Exception:
            value = 1.0
        return min(max(value, minimum), maximum)

    @staticmethod
    def _wait_for_subscriber(pub, timeout_sec=1.0):
        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            if pub.get_subscription_count() > 0:
                return True
            time.sleep(0.01)
        return pub.get_subscription_count() > 0

    def _publish_joint_trajectory(self, publisher, joint_names, positions, duration_sec, wait_timeout_sec=1.0):
        if not self._wait_for_subscriber(publisher, timeout_sec=wait_timeout_sec):
            return False, 'Controller not connected'

        traj = JointTrajectory()
        traj.joint_names = list(joint_names)
        point = JointTrajectoryPoint()
        point.positions = [float(p) for p in positions]
        point.time_from_start = self._duration_msg(duration_sec)
        traj.points = [point]
        publisher.publish(traj)
        return True, 'OK'

    def get_ee_pose(self, robot):
        frame_candidates = [self._current_frame_id(robot)] + [
            frame for frame in self._frame_id_candidates(robot)
            if frame != self._current_frame_id(robot)
        ]
        candidates = [self._current_ee_link(robot)] + [
            link for link in self._ee_link_candidates(robot)
            if link != self._current_ee_link(robot)
        ]
        for frame_id in frame_candidates:
            for ee_link in candidates:
                try:
                    t = self.tf_buffer.lookup_transform(frame_id, ee_link, rclpy.time.Time())
                    pose = Pose()
                    pose.position.x = t.transform.translation.x
                    pose.position.y = t.transform.translation.y
                    pose.position.z = t.transform.translation.z
                    pose.orientation = t.transform.rotation
                    self.active_frame_id[robot] = frame_id
                    self.active_ee_link[robot] = ee_link
                    return pose
                except Exception:
                    continue
        return None

    @staticmethod
    def _quat_to_rpy(x, y, z, w):
        """Convert quaternion to roll/pitch/yaw (radians)."""
        t0 = 2.0 * (w * x + y * z)
        t1 = 1.0 - 2.0 * (x * x + y * y)
        roll = math.atan2(t0, t1)

        t2 = 2.0 * (w * y - z * x)
        t2 = max(-1.0, min(1.0, t2))
        pitch = math.asin(t2)

        t3 = 2.0 * (w * z + x * y)
        t4 = 1.0 - 2.0 * (y * y + z * z)
        yaw = math.atan2(t3, t4)
        return roll, pitch, yaw

    def get_robot_state(self, robot):
        """Return current robot state for UI: pose + orientation + J1..J6."""
        state = {}

        joints = self.joint_positions.get(robot)
        if joints is not None:
            joints = [float(v) for v in joints]
            state['joints_rad'] = joints
            state['joints_deg'] = [math.degrees(v) for v in joints]

        ee = self.get_ee_pose(robot)
        if ee is not None:
            state['position'] = {
                'x': float(ee.position.x),
                'y': float(ee.position.y),
                'z': float(ee.position.z),
            }
            rx, ry, rz = self._quat_to_rpy(
                float(ee.orientation.x),
                float(ee.orientation.y),
                float(ee.orientation.z),
                float(ee.orientation.w),
            )
            state['orientation_rad'] = {'rx': rx, 'ry': ry, 'rz': rz}
            state['orientation_deg'] = {
                'rx': math.degrees(rx),
                'ry': math.degrees(ry),
                'rz': math.degrees(rz),
            }

        return state

    @staticmethod
    def _wait_future(future, timeout=30.0):
        deadline = time.time() + timeout
        while not future.done() and time.time() < deadline:
            time.sleep(0.01)
        return future.done()

    def _scale_trajectory_timing(self, solution, scale: float):
        if not solution or not getattr(solution, 'joint_trajectory', None):
            return
        scale = max(0.05, float(scale))
        if abs(scale - 1.0) < 1e-6:
            return

        for point in solution.joint_trajectory.points:
            t = point.time_from_start.sec + point.time_from_start.nanosec * 1e-9
            t_scaled = max(0.0, t * scale)
            point.time_from_start = self._duration_msg(t_scaled)
            if point.velocities:
                point.velocities = [v / scale for v in point.velocities]
            if point.accelerations:
                point.accelerations = [a / (scale * scale) for a in point.accelerations]

    def move_cartesian(self, robot, dx_mm=0, dy_mm=0, dz_mm=0, velocity_scale=1.0):
        """Move end-effector by a delta in mm. Returns (ok, message)."""
        ee = self.get_ee_pose(robot)
        if ee is None:
            return False, 'No TF data'

        target = copy.deepcopy(ee)
        target.position.x += dx_mm / 1000.0
        target.position.y += dy_mm / 1000.0
        target.position.z += dz_mm / 1000.0

        cfg = ROBOTS[robot]
        response = None
        group_candidates = [self._current_group_name(robot)] + [
            g for g in self._group_name_candidates(robot) if g != self._current_group_name(robot)
        ]
        group_errors = []

        for group_name in group_candidates:
            request = GetCartesianPath.Request()
            request.header.frame_id = self._current_frame_id(robot)
            request.header.stamp = self.get_clock().now().to_msg()
            request.group_name = group_name
            request.link_name = self._current_ee_link(robot)
            request.waypoints = [target]
            request.max_step = self.cartesian_max_step_m
            request.jump_threshold = 0.0
            request.avoid_collisions = False
            request.start_state.is_diff = True

            cart_future = self.cartesian_client.call_async(request)
            if not self._wait_future(cart_future, timeout=10.0):
                group_errors.append(f'{group_name}: timeout')
                continue
            candidate = cart_future.result()
            if candidate is None:
                group_errors.append(f'{group_name}: service failed')
                continue

            err_code = getattr(getattr(candidate, 'error_code', None), 'val', None)
            if err_code not in (None, 1):
                group_errors.append(f'{group_name}: error_code={err_code}')
                continue
            if candidate.fraction <= 0.0:
                group_errors.append(f'{group_name}: no valid cartesian path')
                continue

            response = candidate
            self.active_group_name[robot] = group_name
            break

        if response is None:
            if group_errors:
                return False, '; '.join(group_errors[-2:])
            return False, 'CartesianPath failed'
        if response.fraction < 0.9:
            return False, f'Path incomplete ({response.fraction:.0%})'

        velocity_scale = self._normalize_velocity_scale(velocity_scale)
        time_scale = 1.0 / velocity_scale
        self._scale_trajectory_timing(response.solution, time_scale)

        exec_goal = ExecuteTrajectory.Goal()
        exec_goal.trajectory = response.solution
        exec_future = self.execute_client.send_goal_async(exec_goal)
        if not self._wait_future(exec_future, timeout=10.0):
            return False, 'Execute send timeout'
        goal_handle = exec_future.result()
        if not goal_handle or not goal_handle.accepted:
            return False, 'Execute rejected'
        result_future = goal_handle.get_result_async()
        if not self._wait_future(result_future, timeout=30.0):
            return False, 'Execute timeout'
        result = result_future.result()
        code = result.result.error_code.val if result else None
        if code == 1:
            return True, 'OK'
        return False, f'Error code {code}'

    def move_joint(self, robot, joint_idx, delta_deg, velocity_scale=1.0):
        """Jog a single arm joint through trajectory controller."""
        if robot not in self.joint_positions:
            return False, 'No joint state'

        joint_names = self._current_joint_names(robot)
        if joint_idx < 0 or joint_idx >= len(joint_names):
            return False, 'Invalid joint index'
        target = list(self.joint_positions[robot])
        target[joint_idx] += math.radians(delta_deg)

        velocity_scale = self._normalize_velocity_scale(velocity_scale)
        duration = max(0.05, self.joint_duration_sec / velocity_scale)
        return self.move_arm_to_joints(robot, target, duration_sec=duration)

    def move_arm_to_joints(self, robot, target_joints, duration_sec=None):
        """Move arm to explicit joint targets through trajectory controller."""
        joint_names = self._current_joint_names(robot)
        if len(target_joints) != len(joint_names):
            return False, f'Expected {len(joint_names)} joints, got {len(target_joints)}'
        move_duration = self.joint_duration_sec if duration_sec is None else max(0.05, float(duration_sec))

        arm_pub, _topic = self._pick_publisher(self.arm_publishers[robot])
        ok, msg = self._publish_joint_trajectory(
            arm_pub,
            joint_names,
            target_joints,
            duration_sec=move_duration,
        )
        if not ok:
            return False, msg

        # Keep a local optimistic state so rapid repeated keypresses accumulate.
        self.joint_positions[robot] = list(target_joints)
        for name, pos in zip(joint_names, target_joints):
            self.joint_state_map[name] = pos
        return True, 'OK'

    def move_gripper(self, robot, direction, step_size, velocity_scale=1.0):
        """Jog gripper open/close by step size."""
        cfg = ROBOTS[robot]
        joint_name = self._current_gripper_joint(robot)
        current = self.joint_state_map.get(joint_name)

        open_pos = cfg['gripper_open']
        close_pos = cfg['gripper_close']
        if current is None:
            target = open_pos if direction == 'open' else close_pos
        else:
            open_sign = 1.0 if open_pos > close_pos else -1.0
            step = abs(step_size)
            signed_step = open_sign * step if direction == 'open' else -open_sign * step
            target = self._clamp(current + signed_step, min(open_pos, close_pos), max(open_pos, close_pos))

        if current is not None and abs(target - current) < 1e-6:
            return True, 'Gripper already at limit'

        velocity_scale = self._normalize_velocity_scale(velocity_scale)
        duration = max(0.05, self.gripper_duration_sec / velocity_scale)

        # On xArm hardware, match MoveIt behavior via gripper action first.
        if robot == 'xarm6':
            ok, msg = self._move_xarm_gripper_action(target)
            if ok:
                self.joint_state_map[joint_name] = target
                return True, f'{joint_name}={target:.3f}'
            ok, msg = self._move_xarm_gripper_service(target, velocity_scale=velocity_scale)
            if ok:
                self.joint_state_map[joint_name] = target
                return True, f'{joint_name}={target:.3f}'
            errors = [msg]
        else:
            errors = []

        pubs = self.gripper_publishers[robot]
        ordered = sorted(pubs.items(), key=lambda item: item[1].get_subscription_count(), reverse=True)
        for topic, gripper_pub in ordered:
            wait_timeout = 0.2 if gripper_pub.get_subscription_count() > 0 else 0.05
            ok, msg = self._publish_joint_trajectory(
                gripper_pub,
                [joint_name],
                [target],
                duration_sec=duration,
                wait_timeout_sec=wait_timeout,
            )
            if ok:
                self.joint_state_map[joint_name] = target
                return True, f'{joint_name}={target:.3f}'
            errors.append(f'{topic}: {msg}')

        if robot == 'ur5e':
            return False, 'UR5e gripper controller not connected (RG2 hardware integration may be unavailable)'
        if errors:
            return False, '; '.join(errors[-2:])
        return False, 'gripper controller not connected'

    def save_position(self, robot, name, env='gazebo'):
        if robot not in self.joint_positions:
            return None, None
        positions = [round(p, 6) for p in self.joint_positions[robot]]
        path = DEFAULT_CONFIG_PATHS[robot]
        if path.exists():
            with open(path) as f:
                data = json.load(f)
        else:
            data = {}
        env_key = str(env or 'gazebo').strip().lower()
        if env_key not in ('gazebo', 'real'):
            env_key = 'gazebo'
        # Write into the selected robot environment block.
        robot_block = data.setdefault(robot, {})
        env_block = robot_block.setdefault(env_key, {})
        if not isinstance(env_block, dict):
            env_block = {}
            robot_block[env_key] = env_block
        named = env_block.setdefault('named_positions', {})
        if not isinstance(named, dict):
            named = {}
            env_block['named_positions'] = named
        named[name] = positions
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
            f.write('\n')
        return positions, str(path)

    def infer_robot_environment(self, robot):
        """Infer environment from active naming conventions: gazebo vs real."""
        joint_names = self._current_joint_names(robot)
        candidates = self._joint_name_candidates(robot)
        if len(candidates) >= 2 and joint_names == list(candidates[1]):
            return 'real'
        if len(candidates) >= 1 and joint_names == list(candidates[0]):
            return 'gazebo'

        if robot == 'xarm6':
            if joint_names and all(name.startswith('joint') for name in joint_names):
                return 'real'
            return 'gazebo'
        if robot == 'ur5e':
            if joint_names and all(not name.startswith('ur5e_') for name in joint_names):
                return 'real'
            return 'gazebo'
        return 'gazebo'

    def load_named_position(self, robot, name, preferred_env=None, strict_env=False):
        """Load a named joint position from robot config."""
        path = DEFAULT_CONFIG_PATHS[robot]
        if not path.exists():
            return None, f'Config not found: {path}'
        try:
            with open(path) as f:
                data = json.load(f)
        except Exception as exc:
            return None, f'Failed reading config: {exc}'

        env_key = str(preferred_env or '').strip().lower()
        if env_key not in ('gazebo', 'real'):
            env_key = ''

        candidates = []
        robot_block = data.get(robot)
        preferred_block = None
        if isinstance(robot_block, dict):
            if env_key:
                preferred_block = robot_block.get(env_key, {})
                if isinstance(preferred_block.get('named_positions'), dict):
                    candidates.append(preferred_block['named_positions'])
                if strict_env and not candidates:
                    return None, f'Named position "{name}" not found in {robot}.{env_key}.named_positions'

        if strict_env and env_key:
            # Strict mode only uses the preferred environment block.
            for named in candidates:
                values = named.get(name)
                if not isinstance(values, list):
                    continue
                if len(values) != len(ROBOTS[robot]['joint_names']):
                    continue
                try:
                    return [float(v) for v in values], None
                except Exception:
                    continue
            return None, f'Named position "{name}" not found in {robot}.{env_key}.named_positions'

        if isinstance(data.get('named_positions'), dict):
            candidates.append(data['named_positions'])
        if isinstance(robot_block, dict):
            if isinstance(robot_block.get('named_positions'), dict):
                candidates.append(robot_block['named_positions'])
            # Check environment sub-blocks (gazebo / real).
            ordered_envs = ['gazebo', 'real']
            if env_key:
                ordered_envs = [env_key] + [e for e in ordered_envs if e != env_key]
            for env in ordered_envs:
                env_block = robot_block.get(env, {})
                if isinstance(env_block.get('named_positions'), dict):
                    named = env_block['named_positions']
                    if named not in candidates:
                        candidates.append(named)

        for named in candidates:
            values = named.get(name)
            if not isinstance(values, list):
                continue
            if len(values) != len(ROBOTS[robot]['joint_names']):
                continue
            try:
                return [float(v) for v in values], None
            except Exception:
                continue

        return None, f'Named position "{name}" not found in {path}'


# ── Terminal helpers ─────────────────────────────────────────────────────────

ESC_SEQ_TIMEOUT = 0.2


def read_raw_byte(fd, timeout):
    """Read one raw byte from terminal fd, or None on timeout."""
    rlist, _, _ = sel_mod.select([fd], [], [], timeout)
    if not rlist:
        return None
    try:
        return os.read(fd, 1)
    except BlockingIOError:
        return None


def read_arrow(fd):
    """Try to read an arrow key sequence after ESC was received.
    Returns one of 'UP'/'DOWN'/'LEFT'/'RIGHT', or None."""
    ch2 = read_raw_byte(fd, ESC_SEQ_TIMEOUT)
    if ch2 not in (b'[', b'O'):
        return None

    # Common sequences are ESC [ A/B and ESC O A/B. Some terminals may include
    # modifiers (e.g. ESC [ 1 ; 2 A), so consume until a final arrow letter.
    deadline = time.time() + ESC_SEQ_TIMEOUT
    while time.time() < deadline:
        ch = read_raw_byte(fd, max(0.0, deadline - time.time()))
        if ch is None:
            break
        if ch == b'A':
            return 'UP'
        if ch == b'B':
            return 'DOWN'
        if ch == b'C':
            return 'RIGHT'
        if ch == b'D':
            return 'LEFT'
    return None


def get_key(timeout=0.1):
    """Read a keypress. Terminal must already be in raw mode.
    Returns one of:
        ('axis_move', axis, direction)  — e.g. ('axis_move', 'Z', 'UP')
        ('axis_select', axis)           — axis key pressed alone
        ('arrow', direction)            — bare arrow key
        ('char', ch)                    — any other single character
        None                            — timeout
    """
    fd = sys.stdin.fileno()

    if not hasattr(get_key, '_pending'):
        get_key._pending = []
    pending = get_key._pending

    ch = pending.pop(0) if pending else read_raw_byte(fd, timeout)
    if ch is None:
        return None

    # Axis combo keys: x/y/z + arrow
    if ch in (b'x', b'X', b'y', b'Y', b'z', b'Z'):
        axis = ch.decode('ascii').upper()

        # Drain repeated axis bytes from holding the same key.
        next_ch = None
        while True:
            maybe = read_raw_byte(fd, 0.0)
            if maybe is None:
                break
            if maybe == ch:
                continue
            next_ch = maybe
            break

        # Also check with a timeout — user might press arrow slightly after.
        if next_ch is None:
            next_ch = read_raw_byte(fd, ESC_SEQ_TIMEOUT)
        if next_ch == b'\x1b':
            arrow = read_arrow(fd)
            if arrow:
                return ('axis_move', axis, arrow)
            return ('axis_select', axis)

        # Keep the next non-axis byte for the next polling cycle.
        if next_ch is not None:
            pending.append(next_ch)

        # No arrow followed — just an axis selection
        return ('axis_select', axis)

    # Bare escape sequence (arrow without axis prefix)
    if ch == b'\x1b':
        arrow = read_arrow(fd)
        if arrow:
            return ('arrow', arrow)
        return None  # bare ESC

    text = ch.decode('utf-8', errors='ignore')
    if not text:
        return None
    return ('char', text)


def drain_stdin():
    """Flush any queued keypresses from stdin."""
    fd = sys.stdin.fileno()
    if hasattr(get_key, '_pending'):
        get_key._pending.clear()
    while True:
        if read_raw_byte(fd, 0.0) is None:
            break


def log(msg):
    # In raw mode, \n doesn't do carriage return, so use \r\n
    sys.stdout.write(msg + '\r\n')
    sys.stdout.flush()


def show_ee(node, robot):
    ee = node.get_ee_pose(robot)
    if ee:
        log(f'  EE: X={ee.position.x:+.3f}  Y={ee.position.y:+.3f}  Z={ee.position.z:+.3f}')


def fmt_value(value):
    return f'{value:.2f}'.rstrip('0').rstrip('.')


def clamp(value, minimum, maximum):
    return min(max(value, minimum), maximum)


def wait_for_joint_positions(node, robots, timeout_sec=30.0):
    """Wait until full arm joint states are available for all robots."""
    deadline = time.time() + timeout_sec
    targets = list(robots)
    while rclpy.ok() and time.time() < deadline:
        if all(robot in node.joint_positions for robot in targets):
            return True
        time.sleep(0.05)
    return all(robot in node.joint_positions for robot in targets)


def wait_for_joint_name(node, joint_name, timeout_sec=15.0):
    """Wait until a specific joint appears in /joint_states."""
    deadline = time.time() + timeout_sec
    while rclpy.ok() and time.time() < deadline:
        if joint_name in node.joint_state_map:
            return True
        time.sleep(0.05)
    return joint_name in node.joint_state_map


def wait_for_any_joint_name(node, joint_names, timeout_sec=15.0):
    """Wait until any joint from joint_names appears in /joint_states."""
    names = [str(n) for n in joint_names]
    deadline = time.time() + timeout_sec
    while rclpy.ok() and time.time() < deadline:
        for name in names:
            if name in node.joint_state_map:
                return name
        time.sleep(0.05)
    for name in names:
        if name in node.joint_state_map:
            return name
    return None


def wait_for_gripper_joint(node, robot, timeout_sec=15.0):
    """Wait until the robot gripper command joint is discoverable."""
    deadline = time.time() + timeout_sec
    while rclpy.ok() and time.time() < deadline:
        joint_name = node._detect_gripper_joint(robot)
        if joint_name:
            return joint_name
        time.sleep(0.05)
    return node._detect_gripper_joint(robot)


def move_home_robot(node, robot, home_duration_sec):
    """Move one robot to named position 'home' with environment-aware safety."""
    env = node.infer_robot_environment(robot)
    strict_env = (env == 'real')
    target, err = node.load_named_position(robot, 'home', preferred_env=env, strict_env=strict_env)
    if target is None:
        return False, err

    # For both gazebo and hardware, do a short +Z pre-lift (world frame) for safer home travel.
    ee = node.get_ee_pose(robot)
    if ee is not None:
        active_frame = str(node._current_frame_id(robot)).strip().lower()
        cartesian_ready = (
            node.cartesian_client.wait_for_service(timeout_sec=0.15)
            and node.execute_client.wait_for_server(timeout_sec=0.15)
        )
        if cartesian_ready and active_frame == 'world':
            dz_m = HOME_PRELIFT_DELTA_M
            if env == 'gazebo' and ee.position.z < HOME_PRELIFT_GAZEBO_MIN_Z_M:
                dz_m = max(dz_m, HOME_PRELIFT_GAZEBO_MIN_Z_M - ee.position.z)
            dz_m = min(max(0.0, dz_m), HOME_PRELIFT_MAX_DELTA_M)
            if dz_m > 1e-3:
                ok, msg = node.move_cartesian(
                    robot,
                    dz_mm=dz_m * 1000.0,
                    velocity_scale=HOME_PRELIFT_VELOCITY_SCALE,
                )
                if not ok:
                    node.get_logger().warn(f'[{robot}] pre-lift skipped: {msg}')

    return node.move_arm_to_joints(robot, target, duration_sec=home_duration_sec)


def run_server(args):
    """Run a persistent stdio JSON command server for low-latency UI teleop."""
    service_timeout_sec = max(1.0, float(args.service_timeout_sec))
    tf_warmup_sec = max(0.0, float(args.tf_warmup_sec))
    home_duration_sec = max(0.2, float(args.home_duration_sec))

    rclpy.init()
    node = KeyboardTeleop(
        cartesian_max_step_mm=args.cart_max_step_mm,
        joint_duration_sec=args.joint_duration_sec,
        gripper_duration_sec=args.gripper_duration_sec,
    )
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    planning_ready = False
    gripper_ready = {'xarm6': False, 'ur5e': False}

    def emit(payload):
        sys.stdout.write(json.dumps(payload) + '\n')
        sys.stdout.flush()

    def ensure_planning(robot):
        nonlocal planning_ready
        if not wait_for_joint_positions(node, [robot], timeout_sec=service_timeout_sec):
            return False, f'no joint state for {robot}'
        if not planning_ready:
            if not node.cartesian_client.wait_for_service(timeout_sec=service_timeout_sec):
                return False, '/compute_cartesian_path not available'
            if not node.execute_client.wait_for_server(timeout_sec=service_timeout_sec):
                return False, '/execute_trajectory not available'
            if tf_warmup_sec > 0:
                time.sleep(tf_warmup_sec)
            planning_ready = True
        return True, 'OK'

    def ensure_gripper(robot):
        if gripper_ready[robot]:
            return True, 'OK'
        joint_name = wait_for_gripper_joint(node, robot, timeout_sec=service_timeout_sec)
        if not joint_name:
            if robot == 'xarm6':
                if GripperCommand is not None:
                    _action_name, action_client = node._get_action_client(
                        GripperCommand, 'xarm_gripper/gripper_action', wait_timeout_sec=0.2)
                    if action_client is not None:
                        gripper_ready[robot] = True
                        return True, 'OK'
                if GripperMove is not None:
                    _service_name, service_client = node._get_service_client(
                        GripperMove, 'set_gripper_position', wait_timeout_sec=0.2)
                    if service_client is not None:
                        gripper_ready[robot] = True
                        return True, 'OK'
                return False, 'no gripper joint state, action, or service'
            return False, 'no gripper joint state'
        node.active_gripper_joint[robot] = joint_name
        gripper_ready[robot] = True
        return True, 'OK'

    emit({'ok': True, 'msg': 'ready'})

    try:
        for raw in sys.stdin:
            line = raw.strip()
            if not line:
                continue

            try:
                cmd = json.loads(line)
            except Exception as exc:
                emit({'ok': False, 'msg': f'invalid json: {exc}'})
                continue

            op = str(cmd.get('op', '')).strip().lower()
            robot = str(cmd.get('robot', args.robot)).strip().lower()
            if robot not in ROBOTS:
                emit({'ok': False, 'msg': f'unknown robot: {robot}'})
                continue

            if op == 'shutdown':
                emit({'ok': True, 'msg': 'bye'})
                break

            if op == 'cartesian':
                axis = str(cmd.get('axis', '')).strip().lower()
                if axis not in ('x', 'y', 'z'):
                    emit({'ok': False, 'msg': f'unknown axis: {axis}'})
                    continue
                try:
                    step_mm = float(cmd.get('step_mm'))
                except Exception:
                    emit({'ok': False, 'msg': 'missing or invalid step_mm'})
                    continue
                velocity_scale = node._normalize_velocity_scale(cmd.get('velocity_scale', 1.0))

                ok, msg = ensure_planning(robot)
                if not ok:
                    emit({'ok': False, 'msg': msg})
                    continue

                kwargs = {'dx_mm': 0.0, 'dy_mm': 0.0, 'dz_mm': 0.0}
                kwargs[f'd{axis}_mm'] = step_mm
                ok, msg = node.move_cartesian(robot, velocity_scale=velocity_scale, **kwargs)
                emit({'ok': bool(ok), 'msg': msg})
                continue

            if op == 'gripper':
                action = str(cmd.get('action', '')).strip().lower()
                if action not in ('open', 'close'):
                    emit({'ok': False, 'msg': f'unknown gripper action: {action}'})
                    continue
                step = cmd.get('step')
                if step is None:
                    step = ROBOTS[robot]['gripper_step_default']
                try:
                    step = float(step)
                except Exception:
                    emit({'ok': False, 'msg': 'invalid gripper step'})
                    continue
                velocity_scale = node._normalize_velocity_scale(cmd.get('velocity_scale', 1.0))

                ok, msg = ensure_gripper(robot)
                if not ok:
                    emit({'ok': False, 'msg': msg})
                    continue

                ok, msg = node.move_gripper(robot, action, step, velocity_scale=velocity_scale)
                emit({'ok': bool(ok), 'msg': msg})
                continue

            if op == 'home':
                if not wait_for_joint_positions(node, [robot], timeout_sec=service_timeout_sec):
                    emit({'ok': False, 'msg': f'no joint state for {robot}'})
                    continue
                ok, msg = move_home_robot(node, robot, home_duration_sec=home_duration_sec)
                if not ok:
                    emit({'ok': False, 'msg': msg})
                    continue
                emit({'ok': bool(ok), 'msg': msg})
                continue

            if op == 'joint':
                try:
                    joint = int(cmd.get('joint'))
                except Exception:
                    emit({'ok': False, 'msg': 'missing or invalid joint'})
                    continue
                if joint < 1 or joint > 6:
                    emit({'ok': False, 'msg': f'invalid joint: {joint}'})
                    continue
                try:
                    delta_deg = float(cmd.get('delta_deg'))
                except Exception:
                    emit({'ok': False, 'msg': 'missing or invalid delta_deg'})
                    continue
                velocity_scale = node._normalize_velocity_scale(cmd.get('velocity_scale', 1.0))

                if not wait_for_joint_positions(node, [robot], timeout_sec=service_timeout_sec):
                    emit({'ok': False, 'msg': f'no joint state for {robot}'})
                    continue

                ok, msg = node.move_joint(
                    robot,
                    joint_idx=joint - 1,
                    delta_deg=delta_deg,
                    velocity_scale=velocity_scale,
                )
                emit({'ok': bool(ok), 'msg': msg})
                continue

            if op == 'state':
                if not wait_for_joint_positions(node, [robot], timeout_sec=min(2.0, service_timeout_sec)):
                    emit({'ok': False, 'msg': f'no joint state for {robot}'})
                    continue
                state = node.get_robot_state(robot)
                if not state:
                    emit({'ok': False, 'msg': f'no state available for {robot}'})
                    continue
                emit({'ok': True, 'msg': 'OK', 'state': state})
                continue

            if op == 'save_position':
                name = str(cmd.get('name', '')).strip()
                if not name:
                    emit({'ok': False, 'msg': 'missing position name'})
                    continue
                if len(name) > 64:
                    emit({'ok': False, 'msg': 'position name too long (max 64 chars)'})
                    continue
                if any((not c.isalnum()) and c not in ('_', '-') for c in name):
                    emit({'ok': False, 'msg': 'position name must use [A-Za-z0-9_-]'})
                    continue
                env = str(cmd.get('env', 'gazebo')).strip().lower()
                if env not in ('gazebo', 'real'):
                    env = 'gazebo'

                if not wait_for_joint_positions(node, [robot], timeout_sec=service_timeout_sec):
                    emit({'ok': False, 'msg': f'no joint state for {robot}'})
                    continue

                positions, path = node.save_position(robot, name, env=env)
                if positions is None or path is None:
                    emit({'ok': False, 'msg': 'failed to save position'})
                    continue

                emit({'ok': True, 'msg': f'saved "{name}" in {path} ({env})'})
                continue

            emit({'ok': False, 'msg': f'unknown op: {op}'})
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
    return 0


def run_once(args):
    """Execute a single teleop command and exit with shell-style status code."""
    service_timeout_sec = max(1.0, float(args.service_timeout_sec))
    tf_warmup_sec = max(0.0, float(args.tf_warmup_sec))
    home_duration_sec = max(0.2, float(args.home_duration_sec))

    rclpy.init()
    node = KeyboardTeleop(
        cartesian_max_step_mm=args.cart_max_step_mm,
        joint_duration_sec=args.joint_duration_sec,
        gripper_duration_sec=args.gripper_duration_sec,
    )
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    try:
        robot = args.robot

        if args.once == 'cartesian':
            if args.axis is None or args.once_step_mm is None:
                log('ERROR: --once cartesian requires --axis and --once-step-mm')
                return 2

            if not wait_for_joint_positions(node, [robot], timeout_sec=service_timeout_sec):
                log(f'ERROR: no joint state for {robot}')
                return 1
            if not node.cartesian_client.wait_for_service(timeout_sec=service_timeout_sec):
                log('ERROR: /compute_cartesian_path not available')
                return 1
            if not node.execute_client.wait_for_server(timeout_sec=service_timeout_sec):
                log('ERROR: /execute_trajectory not available')
                return 1
            time.sleep(tf_warmup_sec)

            kwargs = {'dx_mm': 0.0, 'dy_mm': 0.0, 'dz_mm': 0.0}
            kwargs[f'd{args.axis}_mm'] = float(args.once_step_mm)
            ok, msg = node.move_cartesian(robot, **kwargs)
            if not ok:
                log(f'ERROR: {msg}')
                return 1
            log('OK')
            return 0

        if args.once == 'gripper':
            if args.gripper_action is None:
                log('ERROR: --once gripper requires --gripper-action')
                return 2

            joint_name = wait_for_gripper_joint(node, robot, timeout_sec=service_timeout_sec)
            if not joint_name:
                log('ERROR: no gripper joint state')
                return 1
            node.active_gripper_joint[robot] = joint_name

            step = args.gripper_step
            if step is None:
                step = ROBOTS[robot]['gripper_step_default']
            ok, msg = node.move_gripper(robot, args.gripper_action, step)
            if not ok:
                log(f'ERROR: {msg}')
                return 1
            log('OK')
            return 0

        if args.once == 'home':
            robots = ('xarm6', 'ur5e') if args.home_both else (robot,)
            if not wait_for_joint_positions(node, robots, timeout_sec=service_timeout_sec):
                log('ERROR: missing joint states for home command')
                return 1
            if tf_warmup_sec > 0:
                time.sleep(min(tf_warmup_sec, 0.2))

            failures = []
            for rob in robots:
                ok, msg = move_home_robot(node, rob, home_duration_sec=home_duration_sec)
                if not ok:
                    failures.append(f'[{rob}] {msg}')
            if failures:
                for item in failures:
                    log(f'ERROR: {item}')
                return 1
            log('OK')
            return 0

        log(f'ERROR: unknown --once mode: {args.once}')
        return 2
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='MoveIt keyboard teleop')
    parser.add_argument('--step-deg', type=float, default=4.0, help='initial joint step in degrees')
    parser.add_argument('--step-mm', type=float, default=50.0, help='initial Cartesian step in mm')
    parser.add_argument('--step-scale', type=float, default=1.5, help='scale factor for +/- step adjustment')
    parser.add_argument('--min-step-mm', type=float, default=0.5, help='minimum Cartesian step in mm')
    parser.add_argument('--max-step-mm', type=float, default=100.0, help='maximum Cartesian step in mm')
    parser.add_argument('--min-step-deg', type=float, default=0.01, help='minimum joint step in degrees')
    parser.add_argument('--max-step-deg', type=float, default=30.0, help='maximum joint step in degrees')
    parser.add_argument('--cart-max-step-mm', type=float, default=90.0,
                        help='Cartesian interpolation step in mm (higher = faster planning)')
    parser.add_argument('--joint-duration-sec', type=float, default=0.08,
                        help='arm joint command duration (seconds)')
    parser.add_argument('--gripper-duration-sec', type=float, default=0.08,
                        help='gripper command duration (seconds)')
    parser.add_argument('--home-duration-sec', type=float, default=1.2,
                        help='home command duration for both arms (seconds)')
    parser.add_argument('--key-poll-ms', type=float, default=8.0,
                        help='keyboard polling interval in milliseconds')
    parser.add_argument('--robot', choices=['xarm6', 'ur5e'], default='xarm6')
    parser.add_argument('--server', action='store_true',
                        help='run persistent JSON stdio command server for UI teleop')
    parser.add_argument('--once', choices=['cartesian', 'gripper', 'home'],
                        help='run one command and exit (non-interactive)')
    parser.add_argument('--axis', choices=['x', 'y', 'z'],
                        help='axis for --once cartesian')
    parser.add_argument('--once-step-mm', type=float,
                        help='signed Cartesian step for --once cartesian')
    parser.add_argument('--gripper-action', choices=['open', 'close'],
                        help='action for --once gripper')
    parser.add_argument('--gripper-step', type=float,
                        help='step override for --once gripper')
    parser.add_argument('--home-both', action='store_true',
                        help='for --once home, move both robots instead of selected --robot')
    parser.add_argument('--service-timeout-sec', type=float, default=30.0,
                        help='timeout when waiting for ROS services/actions in --once mode')
    parser.add_argument('--tf-warmup-sec', type=float, default=1.0,
                        help='TF warmup time after service readiness in --once mode')
    args = parser.parse_args()

    if args.server:
        return run_server(args)

    if args.once:
        return run_once(args)

    step_scale = max(1.1, args.step_scale)
    home_duration_sec = max(0.2, float(args.home_duration_sec))
    min_step_mm = max(0.1, args.min_step_mm)
    max_step_mm = max(min_step_mm, args.max_step_mm)
    min_step_deg = max(0.01, args.min_step_deg)
    max_step_deg = max(min_step_deg, args.max_step_deg)
    key_poll_sec = max(0.005, args.key_poll_ms / 1000.0)

    orig_term = termios.tcgetattr(sys.stdin.fileno())

    rclpy.init()
    node = KeyboardTeleop(
        cartesian_max_step_mm=args.cart_max_step_mm,
        joint_duration_sec=args.joint_duration_sec,
        gripper_duration_sec=args.gripper_duration_sec,
    )

    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    robot = args.robot
    mode = 'cartesian'
    axis = 'Z'      # last-selected axis for bare arrow keys
    step_mm = clamp(args.step_mm, min_step_mm, max_step_mm)
    step_deg = clamp(args.step_deg, min_step_deg, max_step_deg)
    gripper_step = {r: ROBOTS[r]['gripper_step_default'] for r in ROBOTS}
    joint_idx = 0

    profiles = {
        'precision': {
            'step_mm': 0.5,
            'step_deg': 0.05,
            'gripper_step': {'xarm6': 0.005, 'ur5e': 0.001},
            'key_poll_ms': 10.0,
            'cart_max_step_mm': 5.0,
            'joint_duration_sec': 0.30,
            'gripper_duration_sec': 0.30,
        },
        'fast': {
            'step_mm': 50.0,
            'step_deg': 4.0,
            'gripper_step': {'xarm6': 0.10, 'ur5e': 0.02},
            'key_poll_ms': 8.0,
            'cart_max_step_mm': 90.0,
            'joint_duration_sec': 0.08,
            'gripper_duration_sec': 0.08,
        },
    }

    def apply_profile(name):
        nonlocal step_mm, step_deg, key_poll_sec
        profile = profiles[name]
        step_mm = clamp(profile['step_mm'], min_step_mm, max_step_mm)
        step_deg = clamp(profile['step_deg'], min_step_deg, max_step_deg)
        for rob in ROBOTS:
            cfg = ROBOTS[rob]
            gripper_step[rob] = clamp(
                profile['gripper_step'][rob], cfg['gripper_step_min'], cfg['gripper_step_max'])
        key_poll_sec = max(0.005, profile['key_poll_ms'] / 1000.0)
        node.cartesian_max_step_m = max(0.001, profile['cart_max_step_mm'] / 1000.0)
        node.joint_duration_sec = max(0.05, float(profile['joint_duration_sec']))
        node.gripper_duration_sec = max(0.05, float(profile['gripper_duration_sec']))
        log(f'--- {name.upper()} mode | step={fmt_value(step_mm)}mm/{fmt_value(step_deg)}deg '
            f'grip={fmt_value(gripper_step[robot])} '
            f'poll={fmt_value(key_poll_sec * 1000.0)}ms cart_step={fmt_value(node.cartesian_max_step_m * 1000.0)}mm '
            f'joint_dur={fmt_value(node.joint_duration_sec)}s gripper_dur={fmt_value(node.gripper_duration_sec)}s ---')

    def cartesian_sign_for_arrow(active_robot, ax, direction):
        # Requested mapping:
        # Z uses UP/DOWN.
        # X/Y use LEFT/RIGHT with reversed handedness for both robots.
        # Keep X/Y UP/DOWN as backward-compatible aliases.
        if ax == 'Z':
            if direction == 'UP':
                return +1
            if direction == 'DOWN':
                return -1
            return None
        if ax == 'X':
            if direction == 'UP':
                return +1
            if direction == 'DOWN':
                return -1
            if direction == 'LEFT':
                return +1
            if direction == 'RIGHT':
                return -1
            return None
        if ax == 'Y':
            if direction == 'UP':
                return +1
            if direction == 'DOWN':
                return -1
            # Both robots use reversed Y handedness for LEFT/RIGHT.
            if direction == 'RIGHT':
                return +1
            if direction == 'LEFT':
                return -1
            return None
        return None

    def show_cartesian_mapping(active_robot):
        _ = active_robot
        log('  X + Arrow LEFT/RIGHT = X+/X-    Y + Arrow LEFT/RIGHT = Y-/Y+')

    log('Waiting for /joint_states ...')
    while rclpy.ok() and not node.joint_positions:
        time.sleep(0.1)
    log('Got joint states.')

    log('Waiting for /compute_cartesian_path ...')
    if not node.cartesian_client.wait_for_service(timeout_sec=60.0):
        log('ERROR: /compute_cartesian_path not available')
        rclpy.shutdown()
        return
    if not node.execute_client.wait_for_server(timeout_sec=30.0):
        log('ERROR: /execute_trajectory not available')
        rclpy.shutdown()
        return

    time.sleep(2.0)  # let TF buffer fill

    log('')
    log('=== Ready! ===')
    log(f'[{robot.upper()}] Cartesian mode, step={fmt_value(step_mm)}mm')
    log('  Startup profile: FAST (default)')
    log(f'  Tuning: poll={fmt_value(args.key_poll_ms)}ms, cart_step={fmt_value(args.cart_max_step_mm)}mm, '
        f'joint_dur={fmt_value(args.joint_duration_sec)}s, gripper_dur={fmt_value(args.gripper_duration_sec)}s')
    log('')
    log('  Z + Arrow UP/DOWN = Z+/Z-    (hold Z, press arrow)')
    show_cartesian_mapping(robot)
    log('  Arrow alone = jog last axis (Z:UP/DOWN, X/Y:LEFT/RIGHT)')
    log('  1-6 = select joint (joint mode), then arrows to jog')
    log('  G = gripper mode, then Arrow UP(open)/DOWN(close)')
    log(
        f'  H = pre-lift +Z {fmt_value(HOME_PRELIFT_DELTA_M * 1000.0)}mm '
        f'(cap {fmt_value(HOME_PRELIFT_MAX_DELTA_M * 1000.0)}mm), then both arms home '
        f'({fmt_value(home_duration_sec)}s)'
    )
    log('  +/-=step  TAB=robot  M=cartesian  P=precision  F=fast  S=save  Q=quit')
    log('')
    show_ee(node, robot)

    def do_cartesian(ax, sign):
        kwargs = {f'd{ax.lower()}_mm': step_mm * sign}
        ok, msg = node.move_cartesian(robot, **kwargs)
        drain_stdin()
        if not ok:
            log(f'  FAILED: {msg}')

    def do_joint(sign):
        ok, msg = node.move_joint(robot, joint_idx, step_deg * sign)
        drain_stdin()
        if not ok:
            log(f'  FAILED: {msg}')

    def do_gripper(open_gripper):
        direction = 'open' if open_gripper else 'close'
        ok, msg = node.move_gripper(robot, direction, gripper_step[robot])
        drain_stdin()
        if not ok:
            log(f'  FAILED: {msg}')

    def do_home_both():
        failures = []
        for rob in ('xarm6', 'ur5e'):
            ok, msg = move_home_robot(node, rob, home_duration_sec=home_duration_sec)
            if not ok:
                failures.append(f'[{rob}] {msg}')
        drain_stdin()
        if failures:
            for item in failures:
                log(f'  HOME FAILED: {item}')
            log('  Hint: press S and save position name "home" for each robot')
        else:
            log(f'--- HOME sent to both arms ({fmt_value(home_duration_sec)}s) ---')

    # Keep terminal in raw mode to prevent ^[[A echo during moves
    tty.setraw(sys.stdin.fileno())

    try:
        while rclpy.ok():
            result = get_key(timeout=key_poll_sec)
            if result is None:
                continue

            kind = result[0]

            # ── Axis + Arrow combo (e.g. Z + UP) ──
            if kind == 'axis_move':
                _, ax, direction = result
                if mode != 'cartesian':
                    mode = 'cartesian'
                    log('--- Cartesian mode ---')
                axis = ax  # remember for bare arrows
                sign = cartesian_sign_for_arrow(robot, ax, direction)
                if sign is None:
                    log(f'  INFO: {ax} uses different arrows')
                    continue
                do_cartesian(ax, sign)

            # ── Axis key alone (just selects axis) ──
            elif kind == 'axis_select':
                _, ax = result
                if mode != 'cartesian':
                    mode = 'cartesian'
                    log('--- Cartesian mode ---')
                axis = ax
                log(f'Axis: {axis}')

            # ── Bare arrow (uses last-selected axis, or joint in joint mode) ──
            elif kind == 'arrow':
                _, direction = result
                if mode == 'cartesian':
                    sign = cartesian_sign_for_arrow(robot, axis, direction)
                    if sign is None:
                        continue
                    do_cartesian(axis, sign)
                elif mode == 'joint':
                    if direction not in ('UP', 'DOWN'):
                        continue
                    sign = +1 if direction == 'UP' else -1
                    do_joint(sign)
                else:
                    if direction not in ('UP', 'DOWN'):
                        continue
                    do_gripper(open_gripper=(direction == 'UP'))

            # ── Single character keys ──
            elif kind == 'char':
                _, ch = result

                if ch in ('q', 'Q', '\x03'):
                    break

                elif ch == '\t':
                    robot = 'ur5e' if robot == 'xarm6' else 'xarm6'
                    mode = 'cartesian'
                    axis = 'Z'
                    log(f'\n--- {robot.upper()} | Cartesian {fmt_value(step_mm)}mm ---')
                    show_cartesian_mapping(robot)
                    show_ee(node, robot)

                elif ch in ('m', 'M'):
                    mode = 'cartesian'
                    log(f'--- Cartesian mode axis={axis} {fmt_value(step_mm)}mm ---')
                    show_cartesian_mapping(robot)
                    show_ee(node, robot)

                elif ch in ('g', 'G'):
                    mode = 'gripper'
                    joint_name = node._current_gripper_joint(robot)
                    cur = node.joint_state_map.get(joint_name)
                    cur_s = f'{cur:.3f}' if cur is not None else 'n/a'
                    log(f'--- Gripper mode [{robot.upper()}] joint={joint_name} '
                        f'pos={cur_s} step={fmt_value(gripper_step[robot])} (UP=open, DOWN=close) ---')

                elif ch in ('h', 'H'):
                    do_home_both()

                elif ch in ('p', 'P'):
                    apply_profile('precision')

                elif ch in ('f', 'F'):
                    apply_profile('fast')

                elif ch in ('1', '2', '3', '4', '5', '6'):
                    mode = 'joint'
                    joint_idx = int(ch) - 1
                    jnames = ROBOTS[robot]['joint_names']
                    short = jnames[joint_idx].replace('xarm6_', '').replace('ur5e_', '')
                    if robot in node.joint_positions:
                        cur_deg = math.degrees(node.joint_positions[robot][joint_idx])
                        log(f'--- Joint [{joint_idx+1}] {short} = {cur_deg:+.1f}deg | step={fmt_value(step_deg)}deg ---')
                    else:
                        log(f'--- Joint [{joint_idx+1}] {short} | step={fmt_value(step_deg)}deg ---')

                elif ch in ('+', '='):
                    if mode == 'cartesian':
                        step_mm = clamp(step_mm * step_scale, min_step_mm, max_step_mm)
                        log(f'Step: {fmt_value(step_mm)}mm')
                    elif mode == 'joint':
                        step_deg = clamp(step_deg * step_scale, min_step_deg, max_step_deg)
                        log(f'Step: {fmt_value(step_deg)}deg')
                    else:
                        cfg = ROBOTS[robot]
                        gripper_step[robot] = clamp(
                            gripper_step[robot] * step_scale,
                            cfg['gripper_step_min'],
                            cfg['gripper_step_max'],
                        )
                        log(f'Gripper step [{robot.upper()}]: {fmt_value(gripper_step[robot])}')

                elif ch in ('-', '_'):
                    if mode == 'cartesian':
                        step_mm = clamp(step_mm / step_scale, min_step_mm, max_step_mm)
                        log(f'Step: {fmt_value(step_mm)}mm')
                    elif mode == 'joint':
                        step_deg = clamp(step_deg / step_scale, min_step_deg, max_step_deg)
                        log(f'Step: {fmt_value(step_deg)}deg')
                    else:
                        cfg = ROBOTS[robot]
                        gripper_step[robot] = clamp(
                            gripper_step[robot] / step_scale,
                            cfg['gripper_step_min'],
                            cfg['gripper_step_max'],
                        )
                        log(f'Gripper step [{robot.upper()}]: {fmt_value(gripper_step[robot])}')

                elif ch in ('s', 'S'):
                    # Restore cooked mode for text input
                    termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, orig_term)
                    sys.stdout.write('Position name: ')
                    sys.stdout.flush()
                    name = input().strip()
                    if name:
                        positions, path = node.save_position(robot, name)
                        if positions:
                            log(f'Saved "{name}" -> {path}')
                        else:
                            log('ERROR: No joint state')
                    # Back to raw mode
                    tty.setraw(sys.stdin.fileno())

    except KeyboardInterrupt:
        pass
    finally:
        termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, orig_term)
        log('\nDone.')
        node.destroy_node()
        rclpy.try_shutdown()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
