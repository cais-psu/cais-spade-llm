#!/usr/bin/env python3.10
"""
Keyboard teleop for dual xArm6 + UR5e.

Hardware Stack commands use the direct xArm6 driver and guarded UR5e RTDE
actions. Gazebo commands use MoveIt.

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

from __future__ import annotations

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
import tf2_ros
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Pose, PoseStamped
from moveit_msgs.action import ExecuteTrajectory
from moveit_msgs.srv import GetCartesianPath
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

try:
    from control_msgs.action import FollowJointTrajectory, GripperCommand
except Exception:
    FollowJointTrajectory = None
    GripperCommand = None

try:
    from xarm_msgs.msg import RobotMsg
    from xarm_msgs.srv import GripperMove, MoveVelocity, SetFloat32, SetInt16
except Exception:
    RobotMsg = None
    GripperMove = None
    MoveVelocity = None
    SetFloat32 = None
    SetInt16 = None

try:
    from controller_manager_msgs.srv import ListControllers
except ImportError:
    ListControllers = None

try:
    from xarm_msgs.srv import MoveCartesian
except ImportError:
    MoveCartesian = None

try:
    from cais_lab_robotics.action import (
        MoveUR5eCartesian,
        MoveUR5eJointJog,
        MoveUR5eRelativeCartesian,
    )
    from cais_lab_robotics.srv import SetUR5eCartesianJog
except ImportError:
    MoveUR5eCartesian = None
    MoveUR5eJointJog = None
    MoveUR5eRelativeCartesian = None
    SetUR5eCartesianJog = None

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
            '/xarm6/xarm6_traj_controller/joint_trajectory',
            '/xarm6_xarm6_traj_controller/joint_trajectory',
            '/xarm6_traj_controller/joint_trajectory',
            '/xarm_traj_controller/joint_trajectory',
        ],
        'gripper_joint': 'xarm6_drive_joint',
        'gripper_joint_candidates': ['xarm6_drive_joint', 'drive_joint'],
        'gripper_controller_topic': '/xarm6_xarm_gripper_traj_controller/joint_trajectory',
        'gripper_controller_topics': [
            '/xarm6/xarm_gripper_traj_controller/joint_trajectory',
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
UR5E_HARDWARE_CARTESIAN_ACTION = (
    '/cais_ur5e_rtde_cartesian_controller/move_cartesian'
)
XARM6_HARDWARE_CARTESIAN_SERVICE = '/xarm6/xarm/set_position'
XARM6_HARDWARE_CARTESIAN_VELOCITY_SERVICE = '/xarm6/xarm/vc_set_cartesian_velocity'
XARM6_HARDWARE_ROBOT_STATES_TOPIC = '/xarm6/xarm/robot_states'
XARM6_HARDWARE_SET_MODE_SERVICE = '/xarm6/xarm/set_mode'
XARM6_HARDWARE_SET_STATE_SERVICE = '/xarm6/xarm/set_state'
XARM6_HARDWARE_CONTROL_SERVICE_WAIT_SEC = 5.0
XARM6_HARDWARE_FEEDBACK_DISCOVERY_WAIT_SEC = 8.0
XARM6_TRAJECTORY_MODE_HARD_SAFETY_TIMEOUT_SEC = 90.0
XARM6_TRAJECTORY_MODE_NO_PROGRESS_TIMEOUT_SEC = 15.0
XARM6_TRAJECTORY_MODE_POLL_INTERVAL_SEC = 0.10
UR5E_HARDWARE_RELATIVE_CARTESIAN_ACTION = (
    '/cais_ur5e_rtde_cartesian_controller/move_relative_cartesian'
)
UR5E_HARDWARE_CARTESIAN_JOG_SERVICE = (
    '/cais_ur5e_rtde_cartesian_controller/set_cartesian_jog'
)
UR5E_HARDWARE_JOINT_JOG_ACTION = (
    '/cais_ur5e_rtde_trajectory_controller/move_joint_jog'
)
XARM6_HARDWARE_MAX_JOINT_SPEED_RAD_S = 1.391
UR5E_HARDWARE_MAX_JOINT_SPEED_RAD_S = 1.125
UR5E_HARDWARE_MAX_JOINT_ACCEL_RAD_S2 = 1.263


def _normalized_quaternion(quaternion):
    norm = math.sqrt(sum(float(value) * float(value) for value in quaternion))
    if not math.isfinite(norm) or norm <= 1e-12:
        raise ValueError('quaternion norm is zero or non-finite')
    return tuple(float(value) / norm for value in quaternion)


def _quaternion_multiply(left, right):
    lx, ly, lz, lw = left
    rx, ry, rz, rw = right
    return (
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
        lw * rw - lx * rx - ly * ry - lz * rz,
    )


def _quaternion_rotate(quaternion, vector):
    qx, qy, qz, qw = _normalized_quaternion(quaternion)
    rotated = _quaternion_multiply(
        _quaternion_multiply((qx, qy, qz, qw), (*vector, 0.0)),
        (-qx, -qy, -qz, qw),
    )
    return rotated[:3]


def _compose_transforms(left, right):
    left_translation, left_quaternion = left
    right_translation, right_quaternion = right
    rotated = _quaternion_rotate(left_quaternion, right_translation)
    return (
        tuple(left_translation[index] + rotated[index] for index in range(3)),
        _normalized_quaternion(
            _quaternion_multiply(left_quaternion, right_quaternion)
        ),
    )


def _inverse_transform(transform):
    translation, quaternion = transform
    qx, qy, qz, qw = _normalized_quaternion(quaternion)
    inverse_quaternion = (-qx, -qy, -qz, qw)
    inverse_translation = _quaternion_rotate(
        inverse_quaternion,
        tuple(-value for value in translation),
    )
    return inverse_translation, inverse_quaternion


def _quaternion_error_rad(left, right):
    normalized_left = _normalized_quaternion(left)
    normalized_right = _normalized_quaternion(right)
    dot = abs(sum(a * b for a, b in zip(normalized_left, normalized_right)))
    return 2.0 * math.acos(max(-1.0, min(1.0, dot)))


def _quaternion_from_rpy(roll, pitch, yaw):
    half_roll = float(roll) * 0.5
    half_pitch = float(pitch) * 0.5
    half_yaw = float(yaw) * 0.5
    cr, sr = math.cos(half_roll), math.sin(half_roll)
    cp, sp = math.cos(half_pitch), math.sin(half_pitch)
    cy, sy = math.cos(half_yaw), math.sin(half_yaw)
    return _normalized_quaternion(
        (
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy,
        )
    )


def _xarm_pose_transform(values):
    if len(values) != 6:
        raise ValueError('xArm6 Cartesian pose must contain six values')
    converted = [float(value) for value in values]
    if not all(math.isfinite(value) for value in converted):
        raise ValueError('xArm6 Cartesian pose contains non-finite values')
    return (
        tuple(value / 1000.0 for value in converted[:3]),
        _quaternion_from_rpy(*converted[3:6]),
    )


class KeyboardTeleop(Node):
    def __init__(
        self,
        cartesian_max_step_mm=30.0,
        joint_duration_sec=0.25,
        gripper_duration_sec=0.20,
        ur5e_hardware_trajectory_action=(
            '/cais_ur5e_rtde_trajectory_controller/follow_joint_trajectory'
        ),
        ur5e_hardware_result_timeout_sec=45.0,
        node_name='keyboard_teleop',
    ):
        super().__init__(str(node_name or 'keyboard_teleop'))
        self.cb_group = ReentrantCallbackGroup()
        self.joint_positions = {}
        self.joint_velocities = {}
        self.joint_state_map = {}
        self.joint_state_received_monotonic = {}
        self.joint_velocity_received_monotonic = {}
        self.cartesian_max_step_m = max(0.001, cartesian_max_step_mm / 1000.0)
        self.joint_duration_sec = max(0.05, float(joint_duration_sec))
        self.gripper_duration_sec = max(0.05, float(gripper_duration_sec))
        self.ur5e_hardware_trajectory_action = str(
            ur5e_hardware_trajectory_action
            or '/cais_ur5e_rtde_trajectory_controller/follow_joint_trajectory'
        ).strip()
        self.ur5e_hardware_result_timeout_sec = max(
            10.0,
            float(ur5e_hardware_result_timeout_sec),
        )
        xarm6_real = self._load_real_robot_config('xarm6')
        ur5e_real = self._load_real_robot_config('ur5e')
        xarm6_controller = xarm6_real.get('controller', {})
        ur5e_controller = ur5e_real.get('controller', {})
        self.xarm6_hardware_cartesian_service = str(
            xarm6_controller.get(
                'hardware_cartesian_service',
                XARM6_HARDWARE_CARTESIAN_SERVICE,
            )
        ).strip()
        self.xarm6_hardware_cartesian_velocity_service = str(
            xarm6_controller.get(
                'hardware_cartesian_velocity_service',
                XARM6_HARDWARE_CARTESIAN_VELOCITY_SERVICE,
            )
        ).strip()
        self.xarm6_hardware_robot_states_topic = str(
            xarm6_controller.get(
                'hardware_robot_states_topic',
                XARM6_HARDWARE_ROBOT_STATES_TOPIC,
            )
        ).strip()
        self.xarm6_hardware_joint_duration_scale = max(
            1.0,
            float(xarm6_controller.get('hardware_joint_duration_scale', 1.0)),
        )
        self.xarm6_hardware_cartesian_speed_mm_s = max(
            1.0,
            float(xarm6_controller.get('hardware_cartesian_speed_mm_s', 50.0)),
        )
        self.xarm6_hardware_cartesian_max_speed_mm_s = max(
            self.xarm6_hardware_cartesian_speed_mm_s,
            float(
                xarm6_controller.get(
                    'hardware_cartesian_max_speed_mm_s',
                    50.0,
                )
            ),
        )
        self.xarm6_hardware_cartesian_acceleration_mm_s2 = max(
            1.0,
            float(
                xarm6_controller.get(
                    'hardware_cartesian_acceleration_mm_s2',
                    100.0,
                )
            ),
        )
        self.xarm6_hardware_cartesian_position_tolerance_m = max(
            0.0001,
            float(
                xarm6_controller.get(
                    'hardware_cartesian_position_tolerance_m',
                    0.003,
                )
            ),
        )
        self.xarm6_hardware_cartesian_orientation_tolerance_rad = max(
            0.001,
            float(
                xarm6_controller.get(
                    'hardware_cartesian_orientation_tolerance_rad',
                    math.radians(3.0),
                )
            ),
        )
        self.xarm6_hardware_workspace_bounds = dict(
            xarm6_real.get('static_capabilities', {}).get('workspace_bounds', {})
        )
        self._xarm6_cartesian_motion_attempted = False
        self.ur5e_hardware_cartesian_action = str(
            ur5e_controller.get(
                'hardware_cartesian_action',
                UR5E_HARDWARE_CARTESIAN_ACTION,
            )
        ).strip()
        self.ur5e_hardware_relative_cartesian_action = str(
            ur5e_controller.get(
                'hardware_relative_cartesian_action',
                UR5E_HARDWARE_RELATIVE_CARTESIAN_ACTION,
            )
        ).strip()
        self.ur5e_hardware_cartesian_jog_service = str(
            ur5e_controller.get(
                'hardware_cartesian_jog_service',
                UR5E_HARDWARE_CARTESIAN_JOG_SERVICE,
            )
        ).strip()
        self.ur5e_hardware_cartesian_speed_m_s = max(
            0.001,
            float(ur5e_controller.get('hardware_cartesian_speed_m_s', 0.05)),
        )
        self.ur5e_hardware_cartesian_max_speed_m_s = max(
            self.ur5e_hardware_cartesian_speed_m_s,
            float(
                ur5e_controller.get(
                    'hardware_cartesian_max_speed_m_s',
                    0.10,
                )
            ),
        )
        self.ur5e_hardware_cartesian_acceleration_m_s2 = max(
            0.001,
            float(
                ur5e_controller.get(
                    'hardware_cartesian_acceleration_m_s2',
                    0.10,
                )
            ),
        )
        self.ur5e_hardware_joint_jog_action = UR5E_HARDWARE_JOINT_JOG_ACTION
        self.ur5e_hardware_max_joint_speed_rad_s = (
            UR5E_HARDWARE_MAX_JOINT_SPEED_RAD_S
        )
        self.ur5e_hardware_max_joint_acceleration_rad_s2 = (
            UR5E_HARDWARE_MAX_JOINT_ACCEL_RAD_S2
        )
        self.xarm6_hardware_max_joint_speed_rad_s = (
            XARM6_HARDWARE_MAX_JOINT_SPEED_RAD_S
        )
        self._last_ur5e_joint_jog_state_uncertain = False

        self.create_subscription(JointState, '/joint_states', self._joint_state_cb, 10)
        self._xarm6_robot_state_lock = threading.Lock()
        self._xarm6_robot_state = None
        self._xarm6_robot_state_received_monotonic = 0.0
        self._xarm6_robot_state_subscription = None
        if RobotMsg is not None:
            self._xarm6_robot_state_subscription = self.create_subscription(
                RobotMsg,
                self.xarm6_hardware_robot_states_topic,
                self._xarm6_robot_state_cb,
                10,
            )
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.execute_client = ActionClient(
            self, ExecuteTrajectory, '/execute_trajectory', callback_group=self.cb_group)
        self.ur5e_hardware_trajectory_client = (
            ActionClient(
                self,
                FollowJointTrajectory,
                self.ur5e_hardware_trajectory_action,
                callback_group=self.cb_group,
            )
            if FollowJointTrajectory is not None
            else None
        )
        self.ur5e_hardware_cartesian_client = (
            ActionClient(
                self,
                MoveUR5eCartesian,
                self.ur5e_hardware_cartesian_action,
                callback_group=self.cb_group,
            )
            if MoveUR5eCartesian is not None
            else None
        )
        self.ur5e_hardware_relative_cartesian_client = (
            ActionClient(
                self,
                MoveUR5eRelativeCartesian,
                self.ur5e_hardware_relative_cartesian_action,
                callback_group=self.cb_group,
            )
            if MoveUR5eRelativeCartesian is not None
            else None
        )
        self.ur5e_hardware_joint_jog_client = (
            ActionClient(
                self,
                MoveUR5eJointJog,
                self.ur5e_hardware_joint_jog_action,
                callback_group=self.cb_group,
            )
            if MoveUR5eJointJog is not None
            else None
        )
        self.ur5e_hardware_cartesian_jog_client = (
            self.create_client(
                SetUR5eCartesianJog,
                self.ur5e_hardware_cartesian_jog_service,
                callback_group=self.cb_group,
            )
            if SetUR5eCartesianJog is not None
            else None
        )
        self._ur5e_smooth_active = False
        self._ur5e_last_stop_motion_confirmed = True
        self._ur5e_smooth_state_lock = threading.Lock()
        self._ur5e_smooth_service_lock = threading.Lock()
        self._ur5e_smooth_world_velocity_m_s = [0.0] * 3
        self._ur5e_smooth_watchdog_sec = 0.50
        self._ur5e_smooth_heartbeat_monotonic = 0.0
        self._ur5e_smooth_pending_error = ''
        self._ur5e_smooth_refresh_stop = threading.Event()
        self._ur5e_smooth_refresh_thread = None
        self.xarm6_hardware_cartesian_client = (
            self.create_client(
                MoveCartesian,
                self.xarm6_hardware_cartesian_service,
                callback_group=self.cb_group,
            )
            if MoveCartesian is not None
            else None
        )
        self.xarm6_hardware_cartesian_velocity_client = (
            self.create_client(
                MoveVelocity,
                self.xarm6_hardware_cartesian_velocity_service,
                callback_group=self.cb_group,
            )
            if MoveVelocity is not None
            else None
        )
        self.xarm6_set_mode_client = (
            self.create_client(
                SetInt16,
                XARM6_HARDWARE_SET_MODE_SERVICE,
                callback_group=self.cb_group,
            )
            if SetInt16 is not None
            else None
        )
        self.xarm6_set_state_client = (
            self.create_client(
                SetInt16,
                XARM6_HARDWARE_SET_STATE_SERVICE,
                callback_group=self.cb_group,
            )
            if SetInt16 is not None
            else None
        )
        self.xarm6_controller_list_client = (
            self.create_client(
                ListControllers,
                '/xarm6/controller_manager/list_controllers',
                callback_group=self.cb_group,
            )
            if ListControllers is not None
            else None
        )
        self._xarm6_smooth_active = False
        self._xarm6_last_stop_motion_confirmed = True
        self._xarm6_smooth_state_lock = threading.Lock()
        self._xarm6_smooth_service_lock = threading.Lock()
        self._xarm6_smooth_speeds = [0.0] * 6
        self._xarm6_smooth_watchdog_sec = 0.50
        self._xarm6_smooth_heartbeat_monotonic = 0.0
        self._xarm6_smooth_pending_error = ''
        self._xarm6_smooth_refresh_stop = threading.Event()
        self._xarm6_smooth_refresh_thread = None
        self._xarm6_cartesian_session_mode = 'off'
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
    def _load_real_robot_config(robot):
        path = DEFAULT_CONFIG_PATHS[robot]
        try:
            with path.open(encoding='utf-8') as config_file:
                payload = json.load(config_file)
        except (OSError, json.JSONDecodeError):
            return {}
        robot_block = payload.get(robot, {})
        if not isinstance(robot_block, dict):
            return {}
        real_block = robot_block.get('real', {})
        return real_block if isinstance(real_block, dict) else {}

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
        names = [
            f'/xarm6/xarm/{suffix}',
            f'/xarm6/{suffix}',
            f'/xarm/{suffix}',
            f'/{suffix}',
        ]
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
        except RuntimeError as exc:
            return None, str(exc)
        if not self._wait_future(future, timeout=timeout_sec):
            return None, 'timeout'
        response = future.result()
        if response is None:
            return None, 'service failed'
        return response, None

    def _candidate_action_names(self, suffix):
        if str(suffix) == 'xarm6_traj_controller/follow_joint_trajectory':
            names = [
                '/xarm6/xarm6_traj_controller/follow_joint_trajectory',
                '/xarm6_traj_controller/follow_joint_trajectory',
                '/xarm_traj_controller/follow_joint_trajectory',
                'xarm6_traj_controller/follow_joint_trajectory',
            ]
        elif str(suffix) == 'xarm_gripper/gripper_action':
            names = [
                '/xarm6/xarm_gripper/gripper_action',
                '/xarm/xarm_gripper/gripper_action',
                '/xarm_gripper/gripper_action',
                'xarm_gripper/gripper_action',
            ]
        elif str(suffix) == 'ur5e_rg2_gripper_traj_controller/follow_joint_trajectory':
            names = [
                '/ur5e_rg2_gripper_traj_controller/follow_joint_trajectory',
                'ur5e_rg2_gripper_traj_controller/follow_joint_trajectory',
                '/rg2_gripper_traj_controller/follow_joint_trajectory',
            ]
        else:
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

    def _move_ur5e_rg2_gripper_action(self, joint_name, target_joint, duration_sec):
        if FollowJointTrajectory is None:
            return False, 'FollowJointTrajectory action type unavailable'
        action_name, client = self._get_action_client(
            FollowJointTrajectory,
            'ur5e_rg2_gripper_traj_controller/follow_joint_trajectory',
            wait_timeout_sec=0.1,
        )
        if client is None:
            return False, 'UR5e RG2 action server not available'

        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = [joint_name]
        point = JointTrajectoryPoint()
        point.positions = [float(target_joint)]
        point.time_from_start = self._duration_msg(duration_sec)
        goal.trajectory.points = [point]

        try:
            send_future = client.send_goal_async(goal)
        except Exception as exc:
            return False, f'{action_name}: send failed ({exc})'
        if not self._wait_future(send_future, timeout=1.5):
            return False, f'{action_name}: send timeout'

        goal_handle = send_future.result()
        if goal_handle is None or not goal_handle.accepted:
            return False, f'{action_name}: goal rejected'

        result_future = goal_handle.get_result_async()
        if self._wait_future(result_future, timeout=max(0.5, duration_sec + 0.8)):
            wrapped = result_future.result()
            result = getattr(wrapped, 'result', None)
            error_code = int(getattr(result, 'error_code', 0))
            if error_code not in (0,):
                error_string = str(getattr(result, 'error_string', '')).strip()
                detail = f' error_code={error_code}'
                if error_string:
                    detail += f' {error_string}'
                return False, f'{action_name}:{detail}'
        return True, f'{action_name}: goal accepted'

    def _joint_state_cb(self, msg):
        received_at = time.monotonic()
        message_velocities = {
            str(name): float(velocity)
            for name, velocity in zip(msg.name, msg.velocity)
            if math.isfinite(float(velocity))
        }
        for jname, pos in zip(msg.name, msg.position):
            self.joint_state_map[jname] = pos
        for robot_name, cfg in ROBOTS.items():
            for joint_names in self._joint_name_candidates(robot_name):
                positions = {}
                for jname, pos in zip(msg.name, msg.position):
                    if jname in joint_names:
                        positions[jname] = pos
                if len(positions) == len(joint_names):
                    previous_positions = self.joint_positions.get(robot_name)
                    previous_at = self.joint_state_received_monotonic.get(robot_name)
                    self.joint_positions[robot_name] = [positions[n] for n in joint_names]
                    self.active_joint_names[robot_name] = list(joint_names)
                    self.joint_state_received_monotonic[robot_name] = received_at
                    if all(name in message_velocities for name in joint_names):
                        velocities = [message_velocities[name] for name in joint_names]
                    elif previous_positions is not None and previous_at is not None:
                        elapsed = received_at - float(previous_at)
                        velocities = (
                            [
                                (float(current) - float(previous)) / elapsed
                                for current, previous in zip(
                                    self.joint_positions[robot_name],
                                    previous_positions,
                                )
                            ]
                            if elapsed > 1e-6
                            else []
                        )
                    else:
                        velocities = []
                    if len(velocities) == len(joint_names) and all(
                        math.isfinite(value) for value in velocities
                    ):
                        self.joint_velocities[robot_name] = velocities
                        self.joint_velocity_received_monotonic[robot_name] = received_at
                    break
            for gj in self._gripper_joint_candidates(robot_name):
                if gj in self.joint_state_map:
                    self.active_gripper_joint[robot_name] = gj
                    break

    def _xarm6_robot_state_cb(self, msg):
        """Retain authoritative xArm6 controller TCP pose and active offset."""
        with self._xarm6_robot_state_lock:
            self._xarm6_robot_state = msg
            self._xarm6_robot_state_received_monotonic = time.monotonic()

    def _xarm6_robot_state_snapshot(self):
        with self._xarm6_robot_state_lock:
            message = self._xarm6_robot_state
            received_at = self._xarm6_robot_state_received_monotonic
        if message is None:
            return None, 'xArm6 robot_states feedback has not been received'
        age_sec = time.monotonic() - received_at
        if age_sec > 2.0:
            return None, f'xArm6 robot_states feedback is stale ({age_sec:.2f}s)'
        try:
            pose = [float(value) for value in list(message.pose)]
            offset = [float(value) for value in list(message.offset)]
        except (AttributeError, TypeError, ValueError) as exc:
            return None, f'xArm6 robot_states feedback is invalid: {exc}'
        if len(pose) != 6 or len(offset) != 6:
            return None, 'xArm6 robot_states pose and offset must contain six values'
        if not all(math.isfinite(value) for value in pose + offset):
            return None, 'xArm6 robot_states pose or offset contains non-finite values'
        return {
            'pose': pose,
            'offset': offset,
            'state': int(getattr(message, 'state', -1)),
            'mode': int(getattr(message, 'mode', -1)),
            'age_sec': age_sec,
        }, 'OK'

    def _stationary_readiness(
        self,
        robot,
        *,
        velocity_limit_rad_s=0.01,
        hold_sec=0.25,
        timeout_sec=3.0,
    ):
        """Require fresh joint feedback below the configured velocity limit."""
        limit = max(0.0, float(velocity_limit_rad_s))
        required_hold = max(0.0, float(hold_sec))
        deadline = time.monotonic() + max(required_hold, float(timeout_sec))
        stationary_since = None
        diagnostics = {
            'velocity_limit_rad_s': limit,
            'stationary_hold_required_sec': required_hold,
            'stationary_hold_sec': 0.0,
            'max_joint_velocity_rad_s': None,
        }
        while time.monotonic() < deadline:
            now = time.monotonic()
            positions_at = self.joint_state_received_monotonic.get(robot)
            velocities_at = self.joint_velocity_received_monotonic.get(robot)
            velocities = list(self.joint_velocities.get(robot) or [])
            feedback_fresh = bool(
                positions_at is not None
                and velocities_at is not None
                and now - float(positions_at) <= 0.5
                and now - float(velocities_at) <= 0.5
                and len(velocities) == 6
            )
            if feedback_fresh:
                max_velocity = max(abs(float(value)) for value in velocities)
                diagnostics['max_joint_velocity_rad_s'] = max_velocity
                if max_velocity <= limit:
                    stationary_since = stationary_since or now
                    diagnostics['stationary_hold_sec'] = now - stationary_since
                    if diagnostics['stationary_hold_sec'] >= required_hold:
                        return True, f'{robot} stationary feedback ready', diagnostics
                else:
                    stationary_since = None
                    diagnostics['stationary_hold_sec'] = 0.0
            else:
                stationary_since = None
                diagnostics['stationary_hold_sec'] = 0.0
            time.sleep(0.02)
        return False, f'{robot} did not remain stationary within {timeout_sec:.2f}s', diagnostics

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
            received_at = self.joint_state_received_monotonic.get(robot)
            if received_at is not None:
                state['joint_state_age_sec'] = max(0.0, time.monotonic() - received_at)

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

    def _get_world_ee_pose(self, robot):
        for ee_link in self._ee_link_candidates(robot):
            try:
                transform = self.tf_buffer.lookup_transform(
                    'world',
                    ee_link,
                    rclpy.time.Time(),
                )
            except (
                tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException,
            ):
                continue
            pose = Pose()
            pose.position.x = transform.transform.translation.x
            pose.position.y = transform.transform.translation.y
            pose.position.z = transform.transform.translation.z
            pose.orientation = transform.transform.rotation
            self.active_frame_id[robot] = 'world'
            self.active_ee_link[robot] = ee_link
            return pose
        return None

    def _xarm6_cartesian_readiness(
        self,
        *,
        allow_smooth_mode=False,
        expected_mode=None,
    ):
        """Validate xArm6 controller TCP feedback against world -> link_eef TF."""
        snapshot, error = self._xarm6_robot_state_snapshot()
        if snapshot is None:
            return False, f'Cartesian frame validation failed: {error}', {}
        try:
            world_base_message = self.tf_buffer.lookup_transform(
                'world',
                'link_base',
                rclpy.time.Time(),
            )
            world_eef_message = self.tf_buffer.lookup_transform(
                'world',
                'link_eef',
                rclpy.time.Time(),
            )
            eef_tcp_message = self.tf_buffer.lookup_transform(
                'link_eef',
                'link_tcp',
                rclpy.time.Time(),
            )
            world_base = (
                (
                    float(world_base_message.transform.translation.x),
                    float(world_base_message.transform.translation.y),
                    float(world_base_message.transform.translation.z),
                ),
                _normalized_quaternion(
                    (
                        float(world_base_message.transform.rotation.x),
                        float(world_base_message.transform.rotation.y),
                        float(world_base_message.transform.rotation.z),
                        float(world_base_message.transform.rotation.w),
                    )
                ),
            )
            tf_world_eef = (
                (
                    float(world_eef_message.transform.translation.x),
                    float(world_eef_message.transform.translation.y),
                    float(world_eef_message.transform.translation.z),
                ),
                _normalized_quaternion(
                    (
                        float(world_eef_message.transform.rotation.x),
                        float(world_eef_message.transform.rotation.y),
                        float(world_eef_message.transform.rotation.z),
                        float(world_eef_message.transform.rotation.w),
                    )
                ),
            )
            tf_eef_tcp = (
                (
                    float(eef_tcp_message.transform.translation.x),
                    float(eef_tcp_message.transform.translation.y),
                    float(eef_tcp_message.transform.translation.z),
                ),
                _normalized_quaternion(
                    (
                        float(eef_tcp_message.transform.rotation.x),
                        float(eef_tcp_message.transform.rotation.y),
                        float(eef_tcp_message.transform.rotation.z),
                        float(eef_tcp_message.transform.rotation.w),
                    )
                ),
            )
            base_tcp = _xarm_pose_transform(snapshot['pose'])
            active_eef_tcp = _xarm_pose_transform(snapshot['offset'])
            base_eef = _compose_transforms(base_tcp, _inverse_transform(active_eef_tcp))
            reconstructed_world_eef = _compose_transforms(world_base, base_eef)
            position_error = math.sqrt(
                sum(
                    (
                        reconstructed_world_eef[0][index]
                        - tf_world_eef[0][index]
                    )
                    ** 2
                    for index in range(3)
                )
            )
            orientation_error = _quaternion_error_rad(
                reconstructed_world_eef[1],
                tf_world_eef[1],
            )
            offset_position_difference = math.sqrt(
                sum(
                    (active_eef_tcp[0][index] - tf_eef_tcp[0][index]) ** 2
                    for index in range(3)
                )
            )
            offset_orientation_difference = _quaternion_error_rad(
                active_eef_tcp[1],
                tf_eef_tcp[1],
            )
        except (
            ValueError,
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as exc:
            return False, f'Cartesian frame validation failed: {exc}', {}
        diagnostics = {
            'controller_tcp_pose': list(snapshot['pose']),
            'controller_tcp_offset': list(snapshot['offset']),
            'position_error_m': position_error,
            'orientation_error_rad': orientation_error,
            'tf_link_eef_link_tcp_position_difference_m': offset_position_difference,
            'tf_link_eef_link_tcp_orientation_difference_rad': (
                offset_orientation_difference
            ),
            'controller_mode': snapshot['mode'],
            'controller_state': snapshot['state'],
        }
        required_mode = (
            int(expected_mode)
            if expected_mode is not None
            else (5 if allow_smooth_mode else 1)
        )
        if int(snapshot['mode']) != required_mode:
            return (
                False,
                'Cartesian frame validation failed: xArm6 controller mode '
                f"is {snapshot['mode']}, expected Mode {required_mode}",
                diagnostics,
            )
        controller_state = int(snapshot['state'])
        if controller_state > 2 or controller_state < 0:
            return (
                False,
                'Cartesian frame validation failed: xArm6 controller state '
                f"is {snapshot['state']}, expected a driver-ready state from 0 to 2",
                diagnostics,
            )
        if position_error > max(0.005, self.xarm6_hardware_cartesian_position_tolerance_m):
            return (
                False,
                'Cartesian frame validation failed: controller TCP reconstructed '
                f'world -> link_eef differs from TF by {position_error:.6f} m',
                diagnostics,
            )
        if orientation_error > self.xarm6_hardware_cartesian_orientation_tolerance_rad:
            return (
                False,
                'Cartesian frame validation failed: controller TCP reconstructed '
                f'world -> link_eef differs from TF by {orientation_error:.6f} rad',
                diagnostics,
            )
        message = 'xArm6 Cartesian frame validation ready'
        if offset_position_difference > 0.005 or offset_orientation_difference > math.radians(3.0):
            message += (
                '; active RobotMsg.offset differs from TF link_eef -> link_tcp '
                f'(position={offset_position_difference:.6f} m, '
                f'orientation={offset_orientation_difference:.6f} rad)'
            )
        return True, message, diagnostics

    def _prepare_xarm6_trajectory_mode(  # noqa: C901, PLR0912, PLR0915 - explicit UFactory transition gates.
        self,
    ):
        """Observe the UFactory-owned transition into trajectory Mode 1."""
        started_at = time.monotonic()
        hard_deadline = (
            started_at + XARM6_TRAJECTORY_MODE_HARD_SAFETY_TIMEOUT_SEC
        )
        last_progress_at = started_at
        feedback_missing_since = started_at
        controller_error_since = None
        last_semantic_state = None
        last_controller_state = 'missing'
        last_controller_error = ''
        snapshot_error = 'xArm6 robot_states feedback has not been received'
        last_diagnostics = {}
        mode_requested = False
        state_requested = False
        failure_reason = ''

        while time.monotonic() < hard_deadline:
            now = time.monotonic()
            snapshot, snapshot_error = self._xarm6_robot_state_snapshot()
            if snapshot is not None:
                feedback_missing_since = None
                try:
                    controller_mode = int(snapshot['mode'])
                    controller_state = int(snapshot['state'])
                except (KeyError, TypeError, ValueError):
                    snapshot = None
                    snapshot_error = 'xArm6 robot_states mode or state is invalid'
                    if feedback_missing_since is None:
                        feedback_missing_since = now
                else:
                    last_diagnostics = {
                        'controller_mode': controller_mode,
                        'controller_state': controller_state,
                    }
            elif feedback_missing_since is None:
                feedback_missing_since = now

            remaining_sec = max(0.1, hard_deadline - now)
            observed_controller_state, controller_error = (
                self._xarm6_trajectory_controller_state(
                    timeout_sec=min(1.0, remaining_sec)
                )
            )
            if observed_controller_state is not None:
                last_controller_state = observed_controller_state
                last_controller_error = ''
                controller_error_since = None
            else:
                last_controller_error = str(controller_error or '').strip()
                if controller_error_since is None:
                    controller_error_since = now

            semantic_state = (
                last_diagnostics.get('controller_mode'),
                last_diagnostics.get('controller_state'),
                last_controller_state,
            )
            if semantic_state != last_semantic_state:
                last_semantic_state = semantic_state
                last_progress_at = now

            if snapshot is not None:
                controller_mode = int(snapshot['mode'])
                controller_state = int(snapshot['state'])
                if (
                    controller_mode == 1
                    and 0 <= controller_state <= 2
                    and observed_controller_state == 'active'
                ):
                    return (
                        True,
                        'xArm6 trajectory controller Mode 1 is ready',
                        last_diagnostics,
                    )

                mode_change_requested = controller_mode != 1 and not mode_requested
                if mode_change_requested:
                    ok, message = self._xarm6_set_int16('set_mode', 1)
                    if not ok:
                        return (
                            False,
                            'xArm6 trajectory Mode 1 preparation failed: '
                            f'{message}',
                            last_diagnostics,
                        )
                    mode_requested = True
                    last_progress_at = time.monotonic()
                if (
                    (mode_change_requested or not 0 <= controller_state <= 2)
                    and not state_requested
                ):
                    ok, message = self._xarm6_set_int16('set_state', 0)
                    if not ok:
                        return (
                            False,
                            'xArm6 trajectory state preparation failed: '
                            f'{message}',
                            last_diagnostics,
                        )
                    state_requested = True
                    last_progress_at = time.monotonic()

                if controller_mode != 1 or not 0 <= controller_state <= 2:
                    # Fresh Mode/State transition feedback means UFactory still owns
                    # an active handoff, including its temporary State 5 period.
                    last_progress_at = now

            if (
                feedback_missing_since is not None
                and now - feedback_missing_since
                >= XARM6_HARDWARE_FEEDBACK_DISCOVERY_WAIT_SEC
            ):
                failure_reason = snapshot_error
                break
            if (
                controller_error_since is not None
                and now - controller_error_since
                >= XARM6_HARDWARE_FEEDBACK_DISCOVERY_WAIT_SEC
            ):
                failure_reason = (
                    last_controller_error
                    or '/xarm6/controller_manager/list_controllers is unavailable'
                )
                break
            if (
                snapshot is not None
                and int(snapshot['mode']) == 1
                and 0 <= int(snapshot['state']) <= 2
                and now - last_progress_at
                >= XARM6_TRAJECTORY_MODE_NO_PROGRESS_TIMEOUT_SEC
            ):
                failure_reason = (
                    'trajectory controller activation stopped progressing; '
                    f'state={last_controller_state}'
                )
                break
            time.sleep(XARM6_TRAJECTORY_MODE_POLL_INTERVAL_SEC)

        if not failure_reason:
            failure_reason = (
                'hard safety timeout while waiting for Mode 1 and '
                'xarm6_traj_controller=active'
            )
        if not last_diagnostics:
            subscription = getattr(
                self,
                '_xarm6_robot_state_subscription',
                None,
            )
            publisher_count = None
            if subscription is not None:
                try:
                    publisher_count = int(subscription.get_publisher_count())
                except (AttributeError, RuntimeError, TypeError, ValueError):
                    publisher_count = None
            discovery_detail = (
                ''
                if publisher_count is None
                else f'; discovered_publishers={publisher_count}'
            )
            return (
                False,
                'xArm6 trajectory Mode 1 preparation failed: '
                f'{failure_reason}{discovery_detail}',
                {},
            )
        return (
            False,
            'xArm6 trajectory Mode 1 preparation did not converge: '
            f'{failure_reason}; mode={last_diagnostics.get("controller_mode")!r} '
            f'state={last_diagnostics.get("controller_state")!r} '
            f'trajectory_controller={last_controller_state}',
            last_diagnostics,
        )

    def _world_vector_in_robot_base(self, vector):
        world_base_message = self.tf_buffer.lookup_transform(
            'world',
            'link_base',
            rclpy.time.Time(),
        )
        world_base_rotation = _normalized_quaternion(
            (
                float(world_base_message.transform.rotation.x),
                float(world_base_message.transform.rotation.y),
                float(world_base_message.transform.rotation.z),
                float(world_base_message.transform.rotation.w),
            )
        )
        return _quaternion_rotate(
            _inverse_transform(((0.0, 0.0, 0.0), world_base_rotation))[1],
            tuple(float(value) for value in vector),
        )

    def _move_ur5e_hardware_cartesian(self, target, velocity_scale):
        client = getattr(self, 'ur5e_hardware_cartesian_client', None)
        if client is None or MoveUR5eCartesian is None:
            return False, 'UR5e RTDE Cartesian action type is unavailable'
        if not client.wait_for_server(timeout_sec=2.0):
            return False, f'{self.ur5e_hardware_cartesian_action} is not available'

        values = (
            float(target.position.x),
            float(target.position.y),
            float(target.position.z),
            float(target.orientation.x),
            float(target.orientation.y),
            float(target.orientation.z),
            float(target.orientation.w),
        )
        if not all(math.isfinite(value) for value in values):
            return False, 'UR5e Cartesian target contains non-finite values'

        scale = min(1.0, self._normalize_velocity_scale(velocity_scale))
        goal = MoveUR5eCartesian.Goal()
        goal.target_tool0_pose = PoseStamped()
        goal.target_tool0_pose.header.frame_id = 'world'
        goal.target_tool0_pose.header.stamp = self.get_clock().now().to_msg()
        goal.target_tool0_pose.pose.position.x = values[0]
        goal.target_tool0_pose.pose.position.y = values[1]
        goal.target_tool0_pose.pose.position.z = values[2]
        goal.target_tool0_pose.pose.orientation.x = values[3]
        goal.target_tool0_pose.pose.orientation.y = values[4]
        goal.target_tool0_pose.pose.orientation.z = values[5]
        goal.target_tool0_pose.pose.orientation.w = values[6]
        goal.speed_m_s = max(
            0.005,
            min(self.ur5e_hardware_cartesian_speed_m_s,
                self.ur5e_hardware_cartesian_speed_m_s * scale),
        )
        goal.acceleration_m_s2 = max(
            0.01,
            min(self.ur5e_hardware_cartesian_acceleration_m_s2,
                self.ur5e_hardware_cartesian_acceleration_m_s2 * scale),
        )

        try:
            send_future = client.send_goal_async(goal)
        except RuntimeError as exc:
            return False, f'{self.ur5e_hardware_cartesian_action}: send failed ({exc})'
        if not self._wait_future(send_future, timeout=3.0):
            return False, f'{self.ur5e_hardware_cartesian_action}: send timeout'
        try:
            goal_handle = send_future.result()
        except RuntimeError as exc:
            return False, f'{self.ur5e_hardware_cartesian_action}: send failed ({exc})'
        if goal_handle is None or not goal_handle.accepted:
            return False, f'{self.ur5e_hardware_cartesian_action}: goal rejected'

        result_future = goal_handle.get_result_async()
        if not self._wait_future(
            result_future,
            timeout=self.ur5e_hardware_result_timeout_sec,
        ):
            try:
                cancel_future = goal_handle.cancel_goal_async()
                self._wait_future(cancel_future, timeout=2.0)
            except (AttributeError, RuntimeError):
                pass
            return False, f'{self.ur5e_hardware_cartesian_action}: result timeout'
        try:
            wrapped = result_future.result()
        except RuntimeError as exc:
            return False, f'{self.ur5e_hardware_cartesian_action}: result failed ({exc})'
        result = getattr(wrapped, 'result', None)
        status = int(getattr(wrapped, 'status', -1))
        error_code = int(getattr(result, 'error_code', -1))
        error_string = str(getattr(result, 'error_string', '')).strip()
        if status != 4 or error_code != 0:
            detail = f'status={status} error_code={error_code}'
            if error_string:
                detail += f' {error_string}'
            return False, f'{self.ur5e_hardware_cartesian_action}: {detail}'
        position_error = float(getattr(result, 'final_position_error_m', math.nan))
        orientation_error = float(
            getattr(result, 'final_orientation_error_rad', math.nan)
        )
        return True, (
            f'{self.ur5e_hardware_cartesian_action}: succeeded '
            f'(position_error={position_error:.6f} m, '
            f'orientation_error={orientation_error:.6f} rad)'
        )

    def _move_ur5e_relative_cartesian(
        self,
        world_delta_m,
        velocity_scale,
        *,
        speed_mm_s=None,
    ):
        client = getattr(self, 'ur5e_hardware_relative_cartesian_client', None)
        if client is None or MoveUR5eRelativeCartesian is None:
            return False, 'UR5e RTDE relative Cartesian action type is unavailable'
        if not client.wait_for_server(timeout_sec=2.0):
            return False, f'{self.ur5e_hardware_relative_cartesian_action} is not available'
        values = tuple(float(value) for value in world_delta_m)
        if len(values) != 3 or not all(math.isfinite(value) for value in values):
            return False, 'UR5e world Cartesian Step contains non-finite values'
        scale = min(1.0, self._normalize_velocity_scale(velocity_scale))
        requested_speed_m_s = (
            self.ur5e_hardware_cartesian_speed_m_s * scale
            if speed_mm_s is None
            else float(speed_mm_s) / 1000.0
        )
        if not math.isfinite(requested_speed_m_s) or not (
            0.0 < requested_speed_m_s <= self.ur5e_hardware_cartesian_max_speed_m_s
        ):
            return False, (
                'UR5e Cartesian speed must be finite and within '
                f'(0, {self.ur5e_hardware_cartesian_max_speed_m_s * 1000.0:.1f}] mm/s'
            )
        goal = MoveUR5eRelativeCartesian.Goal()
        goal.world_translation_m.x = values[0]
        goal.world_translation_m.y = values[1]
        goal.world_translation_m.z = values[2]
        goal.speed_m_s = requested_speed_m_s
        goal.acceleration_m_s2 = self.ur5e_hardware_cartesian_acceleration_m_s2
        try:
            send_future = client.send_goal_async(goal)
        except RuntimeError as exc:
            return False, f'{self.ur5e_hardware_relative_cartesian_action}: send failed ({exc})'
        if not self._wait_future(send_future, timeout=3.0):
            return False, f'{self.ur5e_hardware_relative_cartesian_action}: send timeout'
        goal_handle = send_future.result()
        if goal_handle is None or not goal_handle.accepted:
            return False, f'{self.ur5e_hardware_relative_cartesian_action}: goal rejected'
        result_future = goal_handle.get_result_async()
        expected_duration_sec = math.sqrt(sum(value * value for value in values)) / (
            requested_speed_m_s
        )
        result_timeout_sec = max(
            self.ur5e_hardware_result_timeout_sec,
            expected_duration_sec + 15.0,
        )
        if not self._wait_future(result_future, timeout=result_timeout_sec):
            try:
                cancel_future = goal_handle.cancel_goal_async()
                self._wait_future(cancel_future, timeout=2.0)
            except (AttributeError, RuntimeError):
                pass
            return False, f'{self.ur5e_hardware_relative_cartesian_action}: result timeout'
        wrapped = result_future.result()
        result = getattr(wrapped, 'result', None)
        status = int(getattr(wrapped, 'status', -1))
        error_code = int(getattr(result, 'error_code', -1))
        error_string = str(getattr(result, 'error_string', '')).strip()
        if status != 4 or error_code != 0:
            detail = f'status={status} error_code={error_code}'
            if error_string:
                detail += f' {error_string}'
            return False, f'{self.ur5e_hardware_relative_cartesian_action}: {detail}'
        translation_error = float(
            getattr(result, 'final_translation_error_m', math.nan)
        )
        orientation_drift = float(
            getattr(result, 'final_orientation_drift_rad', math.nan)
        )
        return True, (
            f'{self.ur5e_hardware_relative_cartesian_action}: succeeded '
            f'(translation_error={translation_error:.6f} m, '
            f'orientation_drift={orientation_drift:.6f} rad)'
        )

    def _move_ur5e_joint_jog(self, joint, delta_deg, speed_deg_s):
        """Execute one exact-speed UR5e Interactive Teleop joint jog."""
        self._last_ur5e_joint_jog_state_uncertain = False
        client = getattr(self, 'ur5e_hardware_joint_jog_client', None)
        if client is None or MoveUR5eJointJog is None:
            return False, 'UR5e RTDE joint jog action type is unavailable'
        if not client.wait_for_server(timeout_sec=2.0):
            return False, f'{self.ur5e_hardware_joint_jog_action} is not available'
        speed_rad_s = math.radians(float(speed_deg_s))
        delta_rad = math.radians(float(delta_deg))
        if not math.isfinite(delta_rad) or abs(delta_rad) <= 1e-12:
            return False, 'UR5e joint jog delta must be non-zero and finite'
        if not math.isfinite(speed_rad_s) or not (
            0.0 < speed_rad_s <= self.ur5e_hardware_max_joint_speed_rad_s
        ):
            return False, (
                'UR5e joint jog speed must be finite and within '
                f'(0, {math.degrees(self.ur5e_hardware_max_joint_speed_rad_s):.1f}] deg/s'
            )
        goal = MoveUR5eJointJog.Goal()
        goal.joint = int(joint)
        goal.delta_rad = delta_rad
        goal.speed_rad_s = speed_rad_s
        goal.acceleration_rad_s2 = self.ur5e_hardware_max_joint_acceleration_rad_s2
        try:
            send_future = client.send_goal_async(goal)
        except RuntimeError as exc:
            return False, f'{self.ur5e_hardware_joint_jog_action}: send failed ({exc})'
        if not self._wait_future(send_future, timeout=3.0):
            return False, f'{self.ur5e_hardware_joint_jog_action}: send timeout'
        goal_handle = send_future.result()
        if goal_handle is None or not goal_handle.accepted:
            return False, f'{self.ur5e_hardware_joint_jog_action}: goal rejected'
        result_future = goal_handle.get_result_async()
        result_timeout_sec = max(
            self.ur5e_hardware_result_timeout_sec,
            abs(delta_rad) / speed_rad_s + 10.0,
        )
        if not self._wait_future(result_future, timeout=result_timeout_sec):
            try:
                cancel_future = goal_handle.cancel_goal_async()
                self._wait_future(cancel_future, timeout=2.0)
            except (AttributeError, RuntimeError):
                pass
            self._last_ur5e_joint_jog_state_uncertain = True
            return False, f'{self.ur5e_hardware_joint_jog_action}: result timeout'
        wrapped = result_future.result()
        result = getattr(wrapped, 'result', None)
        status = int(getattr(wrapped, 'status', -1))
        error_code = int(getattr(result, 'error_code', -1))
        error_string = str(getattr(result, 'error_string', '')).strip()
        self._last_ur5e_joint_jog_state_uncertain = bool(
            getattr(result, 'state_uncertain', False)
        )
        if status != 4 or error_code != 0:
            detail = f'status={status} error_code={error_code}'
            if error_string:
                detail += f' {error_string}'
            return False, f'{self.ur5e_hardware_joint_jog_action}: {detail}'
        final_error = float(getattr(result, 'final_joint_error_rad', math.nan))
        return True, (
            f'{self.ur5e_hardware_joint_jog_action}: succeeded '
            f'(joint_error={final_error:.6f} rad)'
        )

    def _move_xarm6_relative_cartesian(
        self,
        world_delta_m,
        velocity_scale,
        *,
        speed_mm_s=None,
        restore_trajectory_control=True,
    ):
        self._xarm6_cartesian_motion_attempted = False
        client = getattr(self, 'xarm6_hardware_cartesian_client', None)
        if client is None or MoveCartesian is None:
            return False, 'xArm6 MoveCartesian service type is unavailable'
        if not client.wait_for_service(timeout_sec=2.0):
            return False, f'{self.xarm6_hardware_cartesian_service} is not available'
        session_start = None
        if restore_trajectory_control:
            ready, message, _diagnostics = self._xarm6_cartesian_readiness()
        else:
            if self._xarm6_cartesian_session_mode != 'step':
                return False, 'xArm6 Cartesian Step session is not prepared'
            session_start, message = self._xarm6_robot_state_snapshot()
            ready = bool(
                session_start is not None
                and int(session_start['mode']) == 0
                and 0 <= int(session_start['state']) <= 2
            )
            if session_start is not None and not ready:
                message = (
                    'xArm6 Cartesian Step session lost Mode 0 readiness; '
                    f"mode={session_start['mode']} state={session_start['state']}"
                )
        if not ready:
            return False, message
        current_world_eef = self._get_world_ee_pose('xarm6')
        if current_world_eef is None:
            return False, 'Cartesian frame validation failed: world -> link_eef is unavailable'
        world_target = copy.deepcopy(current_world_eef)
        world_target.position.x += float(world_delta_m[0])
        world_target.position.y += float(world_delta_m[1])
        world_target.position.z += float(world_delta_m[2])
        target_ready, target_message = self._xarm6_target_within_workspace(world_target)
        if not target_ready:
            return False, target_message
        try:
            base_delta_m = self._world_vector_in_robot_base(world_delta_m)
        except (
            ValueError,
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as exc:
            return False, f'xArm6 world -> link_base Step conversion failed: {exc}'
        start, start_error = (
            (session_start, '')
            if session_start is not None
            else self._xarm6_robot_state_snapshot()
        )
        if start is None:
            return False, start_error
        start_transform = _xarm_pose_transform(start['pose'])
        target_translation = tuple(
            start_transform[0][index] + base_delta_m[index]
            for index in range(3)
        )
        scale = min(1.0, self._normalize_velocity_scale(velocity_scale))
        requested_speed_mm_s = (
            self.xarm6_hardware_cartesian_speed_mm_s * scale
            if speed_mm_s is None
            else float(speed_mm_s)
        )
        speed_limit_mm_s = float(
            getattr(
                self,
                'xarm6_hardware_cartesian_max_speed_mm_s',
                self.xarm6_hardware_cartesian_speed_mm_s,
            )
        )
        if not math.isfinite(requested_speed_mm_s) or not (
            0.0 < requested_speed_mm_s <= speed_limit_mm_s
        ):
            return False, (
                'xArm6 Cartesian speed must be finite and within '
                f'(0, {speed_limit_mm_s:.3f}] mm/s'
            )
        request = MoveCartesian.Request()
        request.pose = [
            base_delta_m[0] * 1000.0,
            base_delta_m[1] * 1000.0,
            base_delta_m[2] * 1000.0,
            0.0,
            0.0,
            0.0,
        ]
        request.speed = requested_speed_mm_s
        request.acc = self.xarm6_hardware_cartesian_acceleration_mm_s2
        request.mvtime = 0.0
        request.wait = True
        expected_duration_sec = (
            math.sqrt(sum(float(value) ** 2 for value in base_delta_m))
            * 1000.0
            / requested_speed_mm_s
        )
        request.timeout = max(8.0, expected_duration_sec + 5.0)
        request.radius = -1.0
        request.is_tool_coord = False
        request.relative = True
        request.motion_type = 0
        if restore_trajectory_control:
            handoff_ok, handoff_message = self._xarm6_prepare_firmware_cartesian_mode()
            if not handoff_ok:
                return False, handoff_message
        self._xarm6_cartesian_motion_attempted = True
        response, error = self._call_service(
            client,
            request,
            timeout_sec=request.timeout + 2.0,
        )
        restore_ok, restore_message = (
            self._xarm6_restore_trajectory_control()
            if restore_trajectory_control
            else (True, 'xArm6 Cartesian Step session remains in Mode 0')
        )
        if error is not None:
            message = f'{self.xarm6_hardware_cartesian_service}: {error}'
            if not restore_ok:
                message += f'; trajectory control restore failed: {restore_message}'
            return False, message
        return_code = int(getattr(response, 'ret', -1))
        if return_code != 0:
            detail = str(getattr(response, 'message', '')).strip()
            message = (
                f'{self.xarm6_hardware_cartesian_service}: ret={return_code} {detail}'
            ).strip()
            if not restore_ok:
                message += f'; trajectory control restore failed: {restore_message}'
            return False, message
        if not restore_ok:
            return False, (
                f'{self.xarm6_hardware_cartesian_service}: command succeeded but '
                f'trajectory control restore failed: {restore_message}'
            )
        deadline = time.monotonic() + 2.0
        position_error = math.inf
        orientation_drift = math.inf
        while time.monotonic() < deadline:
            actual, _error = self._xarm6_robot_state_snapshot()
            if actual is not None:
                actual_transform = _xarm_pose_transform(actual['pose'])
                position_error = math.sqrt(
                    sum(
                        (actual_transform[0][index] - target_translation[index]) ** 2
                        for index in range(3)
                    )
                )
                orientation_drift = _quaternion_error_rad(
                    actual_transform[1],
                    start_transform[1],
                )
                if orientation_drift > math.radians(1.0):
                    return False, (
                        'xArm6 translation-only Cartesian Step changed orientation by '
                        f'{orientation_drift:.6f} rad'
                    )
                if position_error <= self.xarm6_hardware_cartesian_position_tolerance_m:
                    return True, (
                        f'{self.xarm6_hardware_cartesian_service}: relative Step succeeded '
                        f'(position_error={position_error:.6f} m, '
                        f'orientation_drift={orientation_drift:.6f} rad)'
                    )
            time.sleep(0.05)
        return False, (
            f'{self.xarm6_hardware_cartesian_service}: relative Step did not converge; '
            f'position_error={position_error:.6f} m '
            f'orientation_drift={orientation_drift:.6f} rad'
        )

    def _xarm6_target_within_workspace(self, target):
        bounds = getattr(self, 'xarm6_hardware_workspace_bounds', {})
        required = {
            'x_min_m', 'x_max_m', 'y_min_m',
            'y_max_m', 'z_min_m', 'z_max_m',
        }
        if not isinstance(bounds, dict) or not required.issubset(bounds):
            return False, 'xArm6 hardware workspace bounds are unavailable'
        coordinates = {
            'x': float(target.position.x),
            'y': float(target.position.y),
            'z': float(target.position.z),
        }
        for axis, value in coordinates.items():
            lower = float(bounds[f'{axis}_min_m'])
            upper = float(bounds[f'{axis}_max_m'])
            if not math.isfinite(value) or value < lower or value > upper:
                return False, (
                    f'xArm6 Cartesian target {axis}={value:.6f} m is outside '
                    f'[{lower:.6f}, {upper:.6f}] m'
                )
        return True, 'OK'

    def _move_xarm6_hardware_cartesian(self, target, velocity_scale):
        client = getattr(self, 'xarm6_hardware_cartesian_client', None)
        if client is None or MoveCartesian is None:
            return False, 'xArm6 MoveCartesian service type is unavailable'
        if not client.wait_for_service(timeout_sec=2.0):
            return False, f'{self.xarm6_hardware_cartesian_service} is not available'
        target_ready, target_message = self._xarm6_target_within_workspace(target)
        if not target_ready:
            return False, target_message

        try:
            world_target = (
                (
                    float(target.position.x),
                    float(target.position.y),
                    float(target.position.z),
                ),
                _normalized_quaternion(
                    (
                        float(target.orientation.x),
                        float(target.orientation.y),
                        float(target.orientation.z),
                        float(target.orientation.w),
                    )
                ),
            )
            world_base_message = self.tf_buffer.lookup_transform(
                'world',
                'link_base',
                rclpy.time.Time(),
            )
            world_base = (
                (
                    float(world_base_message.transform.translation.x),
                    float(world_base_message.transform.translation.y),
                    float(world_base_message.transform.translation.z),
                ),
                _normalized_quaternion(
                    (
                        float(world_base_message.transform.rotation.x),
                        float(world_base_message.transform.rotation.y),
                        float(world_base_message.transform.rotation.z),
                        float(world_base_message.transform.rotation.w),
                    )
                ),
            )
            base_target = _compose_transforms(_inverse_transform(world_base), world_target)
            roll, pitch, yaw = self._quat_to_rpy(*base_target[1])
        except (
            ValueError,
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as exc:
            return False, f'xArm6 world -> link_base Cartesian conversion failed: {exc}'

        scale = min(1.0, self._normalize_velocity_scale(velocity_scale))
        request = MoveCartesian.Request()
        request.pose = [
            base_target[0][0] * 1000.0,
            base_target[0][1] * 1000.0,
            base_target[0][2] * 1000.0,
            roll,
            pitch,
            yaw,
        ]
        request.speed = max(
            5.0,
            min(self.xarm6_hardware_cartesian_speed_mm_s,
                self.xarm6_hardware_cartesian_speed_mm_s * scale),
        )
        request.acc = max(
            10.0,
            min(self.xarm6_hardware_cartesian_acceleration_mm_s2,
                self.xarm6_hardware_cartesian_acceleration_mm_s2 * scale),
        )
        request.mvtime = 0.0
        request.wait = True
        request.timeout = 8.0
        request.radius = -1.0
        request.is_tool_coord = False
        request.relative = False
        request.motion_type = 0
        handoff_ok, handoff_message = self._xarm6_prepare_firmware_cartesian_mode()
        if not handoff_ok:
            return False, handoff_message
        response, error = self._call_service(client, request, timeout_sec=10.0)
        restore_ok, restore_message = self._xarm6_restore_trajectory_control()
        if error is not None:
            message = f'{self.xarm6_hardware_cartesian_service}: {error}'
            if not restore_ok:
                message += f'; trajectory control restore failed: {restore_message}'
            return False, message
        return_code = int(getattr(response, 'ret', -1))
        if return_code != 0:
            message = str(getattr(response, 'message', '')).strip()
            message = (
                f'{self.xarm6_hardware_cartesian_service}: '
                f'ret={return_code} {message}'
            ).strip()
            if not restore_ok:
                message += f'; trajectory control restore failed: {restore_message}'
            return False, message
        if not restore_ok:
            return False, (
                f'{self.xarm6_hardware_cartesian_service}: command succeeded but '
                f'trajectory control restore failed: {restore_message}'
            )

        deadline = time.monotonic() + 2.0
        position_error = math.inf
        orientation_error = math.inf
        while time.monotonic() < deadline:
            actual = self._get_world_ee_pose('xarm6')
            if actual is not None:
                position_error = math.sqrt(
                    (float(actual.position.x) - world_target[0][0]) ** 2
                    + (float(actual.position.y) - world_target[0][1]) ** 2
                    + (float(actual.position.z) - world_target[0][2]) ** 2
                )
                orientation_error = _quaternion_error_rad(
                    (
                        float(actual.orientation.x),
                        float(actual.orientation.y),
                        float(actual.orientation.z),
                        float(actual.orientation.w),
                    ),
                    world_target[1],
                )
                if (
                    position_error
                    <= self.xarm6_hardware_cartesian_position_tolerance_m
                    and orientation_error
                    <= self.xarm6_hardware_cartesian_orientation_tolerance_rad
                ):
                    return True, (
                        f'{self.xarm6_hardware_cartesian_service}: succeeded '
                        f'(position_error={position_error:.6f} m, '
                        f'orientation_error={orientation_error:.6f} rad)'
                    )
            time.sleep(0.05)
        return False, (
            f'{self.xarm6_hardware_cartesian_service}: terminal pose did not converge; '
            f'position_error={position_error:.6f} m '
            f'orientation_error={orientation_error:.6f} rad'
        )

    def _xarm6_trajectory_controller_state(self, *, timeout_sec=1.0):
        """Read one exact xarm6_traj_controller lifecycle state."""
        client = self.xarm6_controller_list_client
        if client is None or ListControllers is None:
            return None, 'xArm6 controller list service type is unavailable'
        try:
            service_ready = bool(
                client.wait_for_service(
                    timeout_sec=min(0.5, max(0.1, float(timeout_sec)))
                )
            )
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            return None, f'xArm6 controller list readiness failed ({exc})'
        if not service_ready:
            return None, '/xarm6/controller_manager/list_controllers is unavailable'
        response, error = self._call_service(
            client,
            ListControllers.Request(),
            timeout_sec=max(0.1, float(timeout_sec)),
        )
        if error is not None:
            return None, str(error)
        if response is None:
            return None, 'xArm6 controller list returned no response'
        controller = next(
            (
                row
                for row in list(getattr(response, 'controller', []))
                if str(getattr(row, 'name', '')).strip()
                == 'xarm6_traj_controller'
            ),
            None,
        )
        if controller is None:
            return 'missing', ''
        return str(getattr(controller, 'state', '')).strip().lower(), ''

    def _xarm6_wait_for_trajectory_controller_state(
        self,
        expected_state,
        *,
        timeout_sec=5.0,
    ):
        expected = str(expected_state).strip().lower()
        deadline = time.monotonic() + max(0.1, float(timeout_sec))
        last_state = 'missing'
        last_error = ''
        while time.monotonic() < deadline:
            remaining = max(0.1, deadline - time.monotonic())
            state, error = self._xarm6_trajectory_controller_state(
                timeout_sec=min(1.0, remaining)
            )
            if error:
                last_error = str(error)
            elif state is not None:
                last_state = state
                if last_state == expected:
                    return True, (
                        f'xArm6 trajectory controller is {expected}'
                    )
            time.sleep(0.05)
        detail = f'state={last_state}'
        if last_error:
            detail += f' last_error={last_error}'
        return False, (
            f'xArm6 trajectory controller did not become {expected}; {detail}'
        )

    def _xarm6_set_int16(self, suffix, value):
        if SetInt16 is None:
            return False, 'xArm6 SetInt16 service type is unavailable'
        service_name = {
            'set_mode': XARM6_HARDWARE_SET_MODE_SERVICE,
            'set_state': XARM6_HARDWARE_SET_STATE_SERVICE,
        }.get(str(suffix))
        client = {
            'set_mode': getattr(self, 'xarm6_set_mode_client', None),
            'set_state': getattr(self, 'xarm6_set_state_client', None),
        }.get(str(suffix))
        if service_name is not None and client is not None:
            try:
                service_ready = client.wait_for_service(
                    timeout_sec=XARM6_HARDWARE_CONTROL_SERVICE_WAIT_SEC
                )
            except (RuntimeError, TypeError, ValueError) as exc:
                return False, f'{service_name} readiness failed: {exc}'
            if not service_ready:
                return False, (
                    f'{service_name} is unavailable after '
                    f'{XARM6_HARDWARE_CONTROL_SERVICE_WAIT_SEC:.1f}s'
                )
        elif service_name is not None:
            return False, f'{service_name} client is unavailable'
        else:
            service_name, client = self._get_service_client(
                SetInt16,
                suffix,
                wait_timeout_sec=0.3,
            )
        if client is None:
            return False, f'xArm6 {suffix} service is unavailable'
        request = SetInt16.Request()
        request.data = int(value)
        response, error = self._call_service(client, request, timeout_sec=3.0)
        if error is not None:
            return False, f'{service_name}: {error}'
        ret = int(getattr(response, 'ret', -1))
        if ret != 0:
            return False, f'{service_name}: ret={ret} {getattr(response, "message", "")}'.strip()
        return True, 'OK'

    def _xarm6_wait_for_mode(self, expected_mode, *, timeout_sec=3.0):
        deadline = time.monotonic() + max(0.0, float(timeout_sec))
        last_mode = None
        last_state = None
        while time.monotonic() < deadline:
            snapshot, snapshot_error = self._xarm6_robot_state_snapshot()
            if snapshot is None:
                last_mode = snapshot_error
                time.sleep(0.05)
                continue
            last_mode = int(snapshot['mode'])
            last_state = int(snapshot['state'])
            if last_mode == int(expected_mode) and 0 <= last_state <= 2:
                return True, 'OK'
            time.sleep(0.05)
        return False, (
            f'xArm6 mode handoff did not converge to Mode {expected_mode}; '
            f'mode={last_mode!r} state={last_state!r}'
        )

    def _xarm6_confirm_cartesian_mode(self, expected_mode):
        ok, message = self._xarm6_wait_for_mode(expected_mode)
        if not ok:
            return False, message
        ok, message = self._xarm6_wait_for_trajectory_controller_state('inactive')
        if not ok:
            return False, message
        return True, f'xArm6 firmware Cartesian Mode {expected_mode} ready'

    def _prepare_xarm6_cartesian_session(self, mode):
        requested = str(mode or '').strip().lower()
        expected_mode = {'step': 0, 'smooth': 5}.get(requested)
        if expected_mode is None:
            return False, f'unknown xArm6 Cartesian mode: {mode}', False
        if self._xarm6_smooth_active:
            stop_ok, stop_message = self._stop_xarm6_cartesian_jog(
                restore_trajectory_control=False,
            )
            if not stop_ok:
                return False, stop_message, bool(
                    not self._xarm6_last_stop_motion_confirmed
                )
        snapshot, _snapshot_error = self._xarm6_robot_state_snapshot()
        already_ready = bool(
            snapshot is not None
            and int(snapshot['mode']) == expected_mode
            and 0 <= int(snapshot['state']) <= 2
        )
        if already_ready:
            ok, message = self._xarm6_confirm_cartesian_mode(expected_mode)
        else:
            ok = True
            message = 'OK'
            for suffix, value in (('set_mode', expected_mode), ('set_state', 0)):
                ok, message = self._xarm6_set_int16(suffix, value)
                if not ok:
                    break
            if ok:
                ok, message = self._xarm6_confirm_cartesian_mode(expected_mode)
        if not ok:
            restore_ok, restore_message = self._xarm6_restore_trajectory_control()
            self._xarm6_cartesian_session_mode = 'off'
            detail = f'xArm6 Cartesian {requested} preparation failed: {message}'
            if not restore_ok:
                detail += f'; trajectory control restore failed: {restore_message}'
            return False, detail, False
        if requested == 'smooth':
            ok, message = self._xarm6_set_tcp_maxacc()
            if not ok:
                restore_ok, restore_message = self._xarm6_restore_trajectory_control()
                self._xarm6_cartesian_session_mode = 'off'
                detail = f'xArm6 Cartesian smooth preparation failed: {message}'
                if not restore_ok:
                    detail += f'; trajectory control restore failed: {restore_message}'
                return False, detail, False
        ready, message, _diagnostics = self._xarm6_cartesian_readiness(
            expected_mode=expected_mode,
        )
        if not ready:
            self._xarm6_restore_trajectory_control()
            self._xarm6_cartesian_session_mode = 'off'
            return False, message, False
        self._xarm6_cartesian_session_mode = requested
        return True, f'xArm6 Cartesian {requested} Mode {expected_mode} ready', False

    def _close_xarm6_cartesian_session(self):
        stop_ok = True
        stop_message = 'xArm6 Cartesian motion already stopped'
        if self._xarm6_smooth_active:
            stop_ok, stop_message = self._stop_xarm6_cartesian_jog(
                restore_trajectory_control=False,
            )
        stop_confirmed = bool(self._xarm6_last_stop_motion_confirmed)
        restore_ok, restore_message = self._xarm6_restore_trajectory_control()
        self._xarm6_cartesian_session_mode = 'off'
        if not stop_ok:
            return False, stop_message, not stop_confirmed
        if not restore_ok:
            return False, (
                'xArm6 Cartesian motion stopped; trajectory Mode 1 is not ready: '
                f'{restore_message}'
            ), False
        return True, 'xArm6 Cartesian session closed; Mode 1 restored', False

    def xarm6_cartesian_session(self, mode):
        """Prepare, inspect, or close the explicit xArm6 Cartesian session."""
        requested = str(mode or '').strip().lower()
        if requested in {'step', 'smooth'}:
            return self._prepare_xarm6_cartesian_session(requested)
        if requested == 'off':
            return self._close_xarm6_cartesian_session()
        if requested == 'status':
            current = str(self._xarm6_cartesian_session_mode or 'off')
            expected_mode = {'step': 0, 'smooth': 5}.get(current)
            if expected_mode is None:
                return True, 'xArm6 Cartesian session is Off', False
            ok, message = self._xarm6_confirm_cartesian_mode(expected_mode)
            return ok, message, False
        return False, f'unknown xArm6 Cartesian session mode: {mode}', False

    def _xarm6_restore_trajectory_control(self):
        errors = []
        for suffix, value in (('set_mode', 1), ('set_state', 0)):
            ok, message = self._xarm6_set_int16(suffix, value)
            if not ok:
                errors.append(message)
        if errors:
            return False, '; '.join(errors)
        ok, message = self._xarm6_wait_for_mode(1)
        if not ok:
            return False, message
        ok, message = self._xarm6_wait_for_trajectory_controller_state('active')
        if not ok:
            return False, message
        return True, 'xArm6 trajectory controller Mode 1 restored'

    def _xarm6_prepare_firmware_cartesian_mode(self):
        for suffix, value in (('set_mode', 0), ('set_state', 0)):
            ok, message = self._xarm6_set_int16(suffix, value)
            if not ok:
                restore_ok, restore_message = self._xarm6_restore_trajectory_control()
                if not restore_ok:
                    self._xarm6_cartesian_motion_attempted = True
                detail = f'xArm6 Cartesian handoff failed: {message}'
                if not restore_ok:
                    detail += f'; trajectory control restore failed: {restore_message}'
                return False, detail
        ok, message = self._xarm6_wait_for_mode(0)
        if not ok:
            restore_ok, restore_message = self._xarm6_restore_trajectory_control()
            if not restore_ok:
                self._xarm6_cartesian_motion_attempted = True
            detail = f'xArm6 Cartesian handoff failed: {message}'
            if not restore_ok:
                detail += f'; trajectory control restore failed: {restore_message}'
            return False, detail
        ok, message = self._xarm6_wait_for_trajectory_controller_state('inactive')
        if not ok:
            restore_ok, restore_message = self._xarm6_restore_trajectory_control()
            detail = f'xArm6 Cartesian handoff failed: {message}'
            if not restore_ok:
                detail += f'; trajectory control restore failed: {restore_message}'
            return False, detail
        return True, 'xArm6 firmware Cartesian Mode 0 ready'

    def _xarm6_set_tcp_maxacc(self):
        if SetFloat32 is None:
            return False, 'xArm6 SetFloat32 service type is unavailable'
        service_name, client = self._get_service_client(
            SetFloat32,
            'set_tcp_maxacc',
            wait_timeout_sec=0.3,
        )
        if client is None:
            return False, 'xArm6 set_tcp_maxacc service is unavailable'
        request = SetFloat32.Request()
        request.data = float(self.xarm6_hardware_cartesian_acceleration_mm_s2)
        response, error = self._call_service(client, request, timeout_sec=3.0)
        if error is not None:
            return False, f'{service_name}: {error}'
        ret = int(getattr(response, 'ret', -1))
        if ret != 0:
            return False, f'{service_name}: ret={ret} {getattr(response, "message", "")}'.strip()
        return True, 'OK'

    def _send_xarm6_cartesian_velocity(self, speeds, *, duration, timeout_sec):
        request = MoveVelocity.Request()
        request.speeds = [float(value) for value in speeds]
        request.is_tool_coord = False
        request.duration = float(duration)
        with self._xarm6_smooth_service_lock:
            response, error = self._call_service(
                self.xarm6_hardware_cartesian_velocity_client,
                request,
                timeout_sec=timeout_sec,
            )
        if error is not None or int(getattr(response, 'ret', -1)) != 0:
            detail = error or f'ret={getattr(response, "ret", -1)}'
            return False, (
                f'{self.xarm6_hardware_cartesian_velocity_service}: {detail}'
            )
        return True, 'OK'

    def _xarm6_refresh_cartesian_jog_once(self):
        with self._xarm6_smooth_state_lock:
            if not self._xarm6_smooth_active:
                return False
            speeds = list(self._xarm6_smooth_speeds)
            watchdog_sec = float(self._xarm6_smooth_watchdog_sec)
            heartbeat_age_sec = (
                time.monotonic() - self._xarm6_smooth_heartbeat_monotonic
            )
        if heartbeat_age_sec > watchdog_sec:
            failure = (
                'xArm6 Cartesian Smooth Hold UI heartbeat expired after '
                f'{heartbeat_age_sec:.3f}s'
            )
        else:
            ok, failure = self._send_xarm6_cartesian_velocity(
                speeds,
                duration=0.30,
                timeout_sec=0.40,
            )
            if ok:
                return True
            failure = f'xArm6 Cartesian Smooth Hold refresh failed: {failure}'
        with self._xarm6_smooth_state_lock:
            if not self._xarm6_smooth_active:
                return False
            self._xarm6_smooth_pending_error = failure
        if hasattr(self, '_xarm6_cartesian_session_mode'):
            self._stop_xarm6_cartesian_jog(
                clear_pending_error=False,
                restore_trajectory_control=(
                    self._xarm6_cartesian_session_mode == 'off'
                ),
            )
        else:
            self._stop_xarm6_cartesian_jog(clear_pending_error=False)
        return False

    def _xarm6_cartesian_jog_refresh_loop(self):
        while not self._xarm6_smooth_refresh_stop.wait(0.10):
            if not self._xarm6_refresh_cartesian_jog_once():
                return

    def _start_xarm6_cartesian_jog_refresh(self):
        previous = self._xarm6_smooth_refresh_thread
        if previous is not None and previous.is_alive():
            self._xarm6_smooth_refresh_stop.set()
            previous.join(timeout=0.60)
            if previous.is_alive():
                return False, 'xArm6 Cartesian Smooth Hold refresh thread did not stop'
        self._xarm6_smooth_refresh_stop.clear()
        refresh_thread = threading.Thread(
            target=self._xarm6_cartesian_jog_refresh_loop,
            name='xarm6_cartesian_smooth_refresh',
            daemon=True,
        )
        self._xarm6_smooth_refresh_thread = refresh_thread
        refresh_thread.start()
        return True, 'OK'

    def _stop_xarm6_cartesian_jog(
        self,
        *,
        clear_pending_error=True,
        restore_trajectory_control=True,
    ):
        errors = []
        with self._xarm6_smooth_state_lock:
            motion_was_active = bool(self._xarm6_smooth_active)
            pending_error = str(self._xarm6_smooth_pending_error or '')
            self._xarm6_smooth_active = False
            self._xarm6_smooth_refresh_stop.set()
            if clear_pending_error:
                self._xarm6_smooth_pending_error = ''
        if pending_error:
            errors.append(pending_error)
        previous_stop_confirmed = bool(
            getattr(self, '_xarm6_last_stop_motion_confirmed', True)
        )
        self._xarm6_last_stop_motion_confirmed = bool(
            not motion_was_active and previous_stop_confirmed
        )
        client = self.xarm6_hardware_cartesian_velocity_client
        if client is not None and MoveVelocity is not None:
            ok, velocity_detail = self._send_xarm6_cartesian_velocity(
                [0.0] * 6,
                duration=0.30,
                timeout_sec=2.0,
            )
            if not ok:
                errors.append(velocity_detail)
            else:
                self._xarm6_last_stop_motion_confirmed = True
        elif motion_was_active:
            errors.append(
                f'{self.xarm6_hardware_cartesian_velocity_service}: '
                'stop service is unavailable'
            )
        if restore_trajectory_control:
            ok, message = self._xarm6_restore_trajectory_control()
            if not ok:
                errors.append(f'trajectory control restore failed: {message}')
        if errors:
            return False, '; '.join(errors)
        return True, (
            'xArm6 Cartesian Smooth Hold stopped'
            if restore_trajectory_control
            else 'xArm6 Cartesian Smooth Hold motion stopped; Mode 5 remains ready'
        )

    def _set_xarm6_cartesian_jog(  # noqa: C901, PLR0912 - guarded Smooth Hold lifecycle.
        self,
        world_velocity_m_s,
        watchdog_sec,
    ):
        client = self.xarm6_hardware_cartesian_velocity_client
        if client is None or MoveVelocity is None:
            return False, 'xArm6 Cartesian velocity service type is unavailable'
        if not client.wait_for_service(timeout_sec=2.0):
            return False, f'{self.xarm6_hardware_cartesian_velocity_service} is unavailable'
        with self._xarm6_smooth_state_lock:
            pending_error = str(self._xarm6_smooth_pending_error or '')
            if pending_error:
                self._xarm6_smooth_pending_error = ''
            smooth_active = bool(self._xarm6_smooth_active)
        if pending_error:
            return False, pending_error
        session_prepared = (
            getattr(self, '_xarm6_cartesian_session_mode', 'off') == 'smooth'
        )
        if smooth_active:
            snapshot, snapshot_error = self._xarm6_robot_state_snapshot()
            if snapshot is None:
                return False, f'Cartesian frame validation failed: {snapshot_error}'
            if int(snapshot['mode']) != 5:
                return False, (
                    'Cartesian frame validation failed: xArm6 controller mode '
                    f"is {snapshot['mode']}, expected Mode 5"
                )
            controller_state = int(snapshot['state'])
            if controller_state > 2 or controller_state < 0:
                return False, (
                    'Cartesian frame validation failed: xArm6 controller state '
                    f"is {snapshot['state']}, expected a driver-ready state from 0 to 2"
                )
        elif session_prepared:
            snapshot, snapshot_error = self._xarm6_robot_state_snapshot()
            if snapshot is None:
                return False, f'Cartesian frame validation failed: {snapshot_error}'
            if int(snapshot['mode']) != 5 or not 0 <= int(snapshot['state']) <= 2:
                return False, (
                    'xArm6 Cartesian Smooth Hold session lost Mode 5 readiness; '
                    f"mode={snapshot['mode']} state={snapshot['state']}"
                )
        else:
            ready, message, _diagnostics = self._xarm6_cartesian_readiness(
                expected_mode=1,
            )
            if not ready:
                return False, message
        try:
            base_velocity_m_s = self._world_vector_in_robot_base(world_velocity_m_s)
        except (
            ValueError,
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as exc:
            return False, f'xArm6 world -> link_base velocity conversion failed: {exc}'
        if not smooth_active and not session_prepared:
            ok, error = self._xarm6_set_int16('set_mode', 5)
            if not ok:
                restore_ok, restore_message = self._xarm6_restore_trajectory_control()
                if not restore_ok:
                    error += f'; trajectory control restore failed: {restore_message}'
                return False, error
            ok, error = self._xarm6_set_int16('set_state', 0)
            if not ok:
                self._stop_xarm6_cartesian_jog()
                return False, error
            ok, error = self._xarm6_wait_for_mode(5)
            if not ok:
                self._stop_xarm6_cartesian_jog()
                return False, error
            ok, error = self._xarm6_wait_for_trajectory_controller_state('inactive')
            if not ok:
                self._stop_xarm6_cartesian_jog()
                return False, error
            ok, error = self._xarm6_set_tcp_maxacc()
            if not ok:
                self._stop_xarm6_cartesian_jog()
                return False, error
        speeds = [
            base_velocity_m_s[0] * 1000.0,
            base_velocity_m_s[1] * 1000.0,
            base_velocity_m_s[2] * 1000.0,
            0.0,
            0.0,
            0.0,
        ]
        watchdog_sec = min(0.50, max(0.10, float(watchdog_sec)))
        if smooth_active:
            with self._xarm6_smooth_state_lock:
                if not self._xarm6_smooth_active:
                    return False, 'xArm6 Cartesian Smooth Hold is no longer active'
                self._xarm6_smooth_speeds = speeds
                self._xarm6_smooth_watchdog_sec = watchdog_sec
                self._xarm6_smooth_heartbeat_monotonic = time.monotonic()
            return True, 'xArm6 Cartesian Smooth Hold active'
        ok, detail = self._send_xarm6_cartesian_velocity(
            speeds,
            duration=0.30,
            timeout_sec=2.0,
        )
        if not ok:
            stop_ok, stop_message = self._stop_xarm6_cartesian_jog(
                restore_trajectory_control=not session_prepared,
            )
            if not stop_ok:
                detail += f'; restore failed: {stop_message}'
            return False, detail
        with self._xarm6_smooth_state_lock:
            self._xarm6_smooth_speeds = speeds
            self._xarm6_smooth_watchdog_sec = watchdog_sec
            self._xarm6_smooth_heartbeat_monotonic = time.monotonic()
            self._xarm6_smooth_pending_error = ''
            self._xarm6_smooth_active = True
        refresh_ok, refresh_error = self._start_xarm6_cartesian_jog_refresh()
        if not refresh_ok:
            with self._xarm6_smooth_state_lock:
                self._xarm6_smooth_pending_error = refresh_error
            self._stop_xarm6_cartesian_jog(
                restore_trajectory_control=not session_prepared,
            )
            return False, refresh_error
        return True, 'xArm6 Cartesian Smooth Hold active'

    def _send_ur5e_cartesian_jog(
        self,
        world_velocity_m_s,
        watchdog_sec,
        *,
        stop,
        timeout_sec,
    ):
        client = self.ur5e_hardware_cartesian_jog_client
        if client is None or SetUR5eCartesianJog is None:
            return False, 'UR5e Cartesian jog service type is unavailable'
        request = SetUR5eCartesianJog.Request()
        request.stop = bool(stop)
        if not stop:
            request.world_linear_velocity_m_s.x = float(world_velocity_m_s[0])
            request.world_linear_velocity_m_s.y = float(world_velocity_m_s[1])
            request.world_linear_velocity_m_s.z = float(world_velocity_m_s[2])
            request.acceleration_m_s2 = self.ur5e_hardware_cartesian_acceleration_m_s2
            request.watchdog_sec = min(0.50, max(0.10, float(watchdog_sec)))
        with self._ur5e_smooth_service_lock:
            response, error = self._call_service(
                client,
                request,
                timeout_sec=timeout_sec,
            )
        if error is not None:
            return False, f'{self.ur5e_hardware_cartesian_jog_service}: {error}'
        return bool(response.accepted), str(response.message)

    def _ur5e_refresh_cartesian_jog_once(self):
        with self._ur5e_smooth_state_lock:
            if not self._ur5e_smooth_active:
                return False
            world_velocity_m_s = list(self._ur5e_smooth_world_velocity_m_s)
            watchdog_sec = float(self._ur5e_smooth_watchdog_sec)
            heartbeat_age_sec = (
                time.monotonic() - self._ur5e_smooth_heartbeat_monotonic
            )
        if heartbeat_age_sec > watchdog_sec:
            failure = (
                'UR5e Cartesian Smooth Hold UI heartbeat expired after '
                f'{heartbeat_age_sec:.3f}s'
            )
        else:
            ok, failure = self._send_ur5e_cartesian_jog(
                world_velocity_m_s,
                watchdog_sec,
                stop=False,
                timeout_sec=0.30,
            )
            if ok:
                return True
            failure = f'UR5e Cartesian Smooth Hold refresh failed: {failure}'
        with self._ur5e_smooth_state_lock:
            if not self._ur5e_smooth_active:
                return False
            self._ur5e_smooth_pending_error = failure
        self._stop_ur5e_cartesian_jog(clear_pending_error=False)
        return False

    def _ur5e_cartesian_jog_refresh_loop(self):
        while not self._ur5e_smooth_refresh_stop.wait(0.05):
            if not self._ur5e_refresh_cartesian_jog_once():
                return

    def _start_ur5e_cartesian_jog_refresh(self):
        previous = self._ur5e_smooth_refresh_thread
        if previous is not None and previous.is_alive():
            self._ur5e_smooth_refresh_stop.set()
            previous.join(timeout=0.60)
            if previous.is_alive():
                return False, 'UR5e Cartesian Smooth Hold refresh thread did not stop'
        self._ur5e_smooth_refresh_stop.clear()
        refresh_thread = threading.Thread(
            target=self._ur5e_cartesian_jog_refresh_loop,
            name='ur5e_cartesian_smooth_refresh',
            daemon=True,
        )
        self._ur5e_smooth_refresh_thread = refresh_thread
        refresh_thread.start()
        return True, 'OK'

    def _stop_ur5e_cartesian_jog(self, *, clear_pending_error=True):
        errors = []
        with self._ur5e_smooth_state_lock:
            motion_was_active = bool(self._ur5e_smooth_active)
            pending_error = str(self._ur5e_smooth_pending_error or '')
            self._ur5e_smooth_active = False
            self._ur5e_smooth_refresh_stop.set()
            if clear_pending_error:
                self._ur5e_smooth_pending_error = ''
        if pending_error:
            errors.append(pending_error)
        previous_stop_confirmed = bool(
            getattr(self, '_ur5e_last_stop_motion_confirmed', True)
        )
        self._ur5e_last_stop_motion_confirmed = bool(
            not motion_was_active and previous_stop_confirmed
        )
        ok, stop_message = self._send_ur5e_cartesian_jog(
            (0.0, 0.0, 0.0),
            0.50,
            stop=True,
            timeout_sec=3.0,
        )
        if ok:
            self._ur5e_last_stop_motion_confirmed = True
        else:
            errors.append(stop_message)
        if errors:
            return False, '; '.join(errors)
        return True, stop_message or 'UR5e Cartesian Smooth Hold stopped'

    def _set_ur5e_cartesian_jog(self, world_velocity_m_s, watchdog_sec):
        client = self.ur5e_hardware_cartesian_jog_client
        if client is None or SetUR5eCartesianJog is None:
            return False, 'UR5e Cartesian jog service type is unavailable'
        with self._ur5e_smooth_state_lock:
            pending_error = str(self._ur5e_smooth_pending_error or '')
            if pending_error:
                self._ur5e_smooth_pending_error = ''
            smooth_active = bool(self._ur5e_smooth_active)
        if pending_error:
            return False, pending_error
        watchdog_sec = min(0.50, max(0.10, float(watchdog_sec)))
        if smooth_active:
            with self._ur5e_smooth_state_lock:
                if not self._ur5e_smooth_active:
                    return False, 'UR5e Cartesian Smooth Hold is no longer active'
                self._ur5e_smooth_world_velocity_m_s = list(world_velocity_m_s)
                self._ur5e_smooth_watchdog_sec = watchdog_sec
                self._ur5e_smooth_heartbeat_monotonic = time.monotonic()
            return True, 'UR5e Cartesian Smooth Hold active'
        if not client.service_is_ready():
            return False, f'{self.ur5e_hardware_cartesian_jog_service} is unavailable'
        ok, message = self._send_ur5e_cartesian_jog(
            world_velocity_m_s,
            watchdog_sec,
            stop=False,
            timeout_sec=3.0,
        )
        if not ok:
            return False, message
        with self._ur5e_smooth_state_lock:
            self._ur5e_smooth_world_velocity_m_s = list(world_velocity_m_s)
            self._ur5e_smooth_watchdog_sec = watchdog_sec
            self._ur5e_smooth_heartbeat_monotonic = time.monotonic()
            self._ur5e_smooth_pending_error = ''
            self._ur5e_smooth_active = True
        refresh_ok, refresh_error = self._start_ur5e_cartesian_jog_refresh()
        if not refresh_ok:
            with self._ur5e_smooth_state_lock:
                self._ur5e_smooth_pending_error = refresh_error
            self._stop_ur5e_cartesian_jog(clear_pending_error=False)
            return False, refresh_error
        return True, message or 'UR5e Cartesian Smooth Hold active'

    def set_cartesian_jog(
        self,
        robot,
        axis,
        speed_mm_s,
        *,
        stop=False,
        watchdog_sec=0.50,
    ):
        """Start, refresh, or stop translation-only hardware Cartesian velocity jog."""
        if robot not in {'xarm6', 'ur5e'}:
            return False, f'unknown robot: {robot}'
        if stop:
            if robot == 'xarm6':
                return self._stop_xarm6_cartesian_jog(
                    restore_trajectory_control=(
                        self._xarm6_cartesian_session_mode != 'smooth'
                    ),
                )
            return self._stop_ur5e_cartesian_jog()
        if axis not in {'x', 'y', 'z'}:
            return False, f'unknown axis: {axis}'
        speed_m_s = float(speed_mm_s) / 1000.0
        if not math.isfinite(speed_m_s) or abs(speed_m_s) <= 1e-9:
            return False, 'Cartesian Smooth Hold speed must be non-zero and finite'
        if robot == 'xarm6':
            xarm6_speed_limit_mm_s = float(
                getattr(
                    self,
                    'xarm6_hardware_cartesian_max_speed_mm_s',
                    self.xarm6_hardware_cartesian_speed_mm_s,
                )
            )
            requested_speed_mm_s = abs(speed_m_s) * 1000.0
            if not 0.0 < requested_speed_mm_s <= xarm6_speed_limit_mm_s:
                return False, (
                    'xArm6 Cartesian Smooth Hold speed must be finite and within '
                    f'(0, {xarm6_speed_limit_mm_s:.3f}] mm/s'
                )
        world_velocity_m_s = [0.0, 0.0, 0.0]
        world_velocity_m_s[{'x': 0, 'y': 1, 'z': 2}[axis]] = speed_m_s
        if robot == 'xarm6':
            return self._set_xarm6_cartesian_jog(
                world_velocity_m_s,
                watchdog_sec,
            )
        return self._set_ur5e_cartesian_jog(
            world_velocity_m_s,
            watchdog_sec,
        )

    def move_cartesian(
        self,
        robot,
        dx_mm=0,
        dy_mm=0,
        dz_mm=0,
        velocity_scale=1.0,
        *,
        speed_mm_s=None,
        xarm6_cartesian_session=False,
    ):
        """Move end-effector by a delta in mm. Returns (ok, message)."""
        environment = self.infer_robot_environment(robot)
        world_delta_m = (
            float(dx_mm) / 1000.0,
            float(dy_mm) / 1000.0,
            float(dz_mm) / 1000.0,
        )
        if environment == 'real':
            if robot == 'ur5e':
                if speed_mm_s is None:
                    return self._move_ur5e_relative_cartesian(
                        world_delta_m,
                        velocity_scale,
                    )
                return self._move_ur5e_relative_cartesian(
                    world_delta_m,
                    velocity_scale,
                    speed_mm_s=speed_mm_s,
                )
            if robot == 'xarm6':
                if speed_mm_s is None and not xarm6_cartesian_session:
                    return self._move_xarm6_relative_cartesian(
                        world_delta_m,
                        velocity_scale,
                    )
                return self._move_xarm6_relative_cartesian(
                    world_delta_m,
                    velocity_scale,
                    speed_mm_s=speed_mm_s,
                    restore_trajectory_control=not xarm6_cartesian_session,
                )
            return False, f'No direct Cartesian implementation for {robot}'
        ee = (
            self._get_world_ee_pose(robot)
            if environment == 'real'
            else self.get_ee_pose(robot)
        )
        if ee is None:
            return False, (
                f'No world -> {self._current_ee_link(robot)} TF data'
                if environment == 'real'
                else 'No TF data'
            )

        target = copy.deepcopy(ee)
        target.position.x += dx_mm / 1000.0
        target.position.y += dy_mm / 1000.0
        target.position.z += dz_mm / 1000.0

        return self._move_gazebo_cartesian(robot, target, velocity_scale)

    def _move_gazebo_cartesian(self, robot, target, velocity_scale):
        """Plan and execute one Gazebo Cartesian target through MoveIt."""
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

    def move_joint(
        self,
        robot,
        joint_idx,
        delta_deg,
        velocity_scale=1.0,
        *,
        speed_deg_s=None,
    ):
        """Jog a single arm joint through trajectory controller."""
        if robot not in self.joint_positions:
            return False, 'No joint state'

        joint_names = self._current_joint_names(robot)
        if joint_idx < 0 or joint_idx >= len(joint_names):
            return False, 'Invalid joint index'
        target = list(self.joint_positions[robot])
        target[joint_idx] += math.radians(delta_deg)

        environment = self.infer_robot_environment(robot)
        if speed_deg_s is not None:
            requested_speed_deg_s = float(speed_deg_s)
            speed_limit_rad_s = (
                self.xarm6_hardware_max_joint_speed_rad_s
                if robot == 'xarm6'
                else self.ur5e_hardware_max_joint_speed_rad_s
            )
            if not math.isfinite(requested_speed_deg_s) or not (
                0.0 < requested_speed_deg_s <= math.degrees(speed_limit_rad_s)
            ):
                return False, (
                    f'{robot} joint jog speed must be finite and within '
                    f'(0, {math.degrees(speed_limit_rad_s):.1f}] deg/s'
                )
            if environment == 'real' and robot == 'ur5e':
                return self._move_ur5e_joint_jog(
                    joint_idx + 1,
                    delta_deg,
                    requested_speed_deg_s,
                )
            duration = max(0.05, abs(float(delta_deg)) / requested_speed_deg_s)
            if environment == 'real' and robot == 'xarm6':
                return self._move_xarm6_arm_action(
                    joint_names,
                    target,
                    duration_sec=duration,
                )
            return self.move_arm_to_joints(robot, target, duration_sec=duration)

        velocity_scale = self._normalize_velocity_scale(velocity_scale)
        duration = max(0.05, self.joint_duration_sec / velocity_scale)
        return self.move_arm_to_joints(robot, target, duration_sec=duration)

    def move_arm_to_joints(self, robot, target_joints, duration_sec=None):
        """Move arm to explicit joint targets through trajectory controller."""
        joint_names = self._current_joint_names(robot)
        if len(target_joints) != len(joint_names):
            return False, f'Expected {len(joint_names)} joints, got {len(target_joints)}'
        move_duration = self.joint_duration_sec if duration_sec is None else max(0.05, float(duration_sec))

        if self.infer_robot_environment(robot) == 'real':
            if robot == 'xarm6':
                return self._move_xarm6_arm_action(
                    joint_names,
                    target_joints,
                    duration_sec=(
                        move_duration * self.xarm6_hardware_joint_duration_scale
                    ),
                )
            if robot == 'ur5e':
                return self._move_ur5e_arm_action(
                    joint_names,
                    target_joints,
                    duration_sec=move_duration,
                )

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

    def _move_xarm6_arm_action(self, joint_names, target_joints, duration_sec):
        """Send a real xArm6 joint target through its trajectory action."""
        if FollowJointTrajectory is None:
            return False, 'xArm6 FollowJointTrajectory action type is unavailable'
        action_name, client = self._get_action_client(
            FollowJointTrajectory,
            'xarm6_traj_controller/follow_joint_trajectory',
            wait_timeout_sec=2.0,
        )
        if client is None:
            return False, (
                'xArm6 trajectory action is unavailable; tried '
                f'{self._candidate_action_names("xarm6_traj_controller/follow_joint_trajectory")}'
            )

        current_positions = self.joint_positions.get('xarm6')
        if current_positions is None or len(current_positions) != len(joint_names):
            return False, 'xArm6 current joint state is unavailable'

        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = list(joint_names)
        start_point = JointTrajectoryPoint()
        start_point.positions = [float(position) for position in current_positions]
        start_point.time_from_start = self._duration_msg(0.0)
        target_point = JointTrajectoryPoint()
        target_point.positions = [float(position) for position in target_joints]
        target_point.time_from_start = self._duration_msg(duration_sec)
        goal.trajectory.points = [start_point, target_point]
        try:
            send_future = client.send_goal_async(goal)
        except RuntimeError as exc:
            return False, f'{action_name}: send failed ({exc})'
        if not self._wait_future(send_future, timeout=3.0):
            return False, f'{action_name}: send timeout'
        try:
            goal_handle = send_future.result()
        except RuntimeError as exc:
            return False, f'{action_name}: send failed ({exc})'
        if goal_handle is None or not goal_handle.accepted:
            return False, f'{action_name}: goal rejected'

        result_future = goal_handle.get_result_async()
        result_timeout = max(10.0, float(duration_sec) + 20.0)
        if not self._wait_future(result_future, timeout=result_timeout):
            return False, f'{action_name}: result timeout'
        try:
            wrapped = result_future.result()
        except RuntimeError as exc:
            return False, f'{action_name}: result failed ({exc})'
        result = getattr(wrapped, 'result', None)
        status = int(getattr(wrapped, 'status', -1))
        error_code = int(getattr(result, 'error_code', -1))
        error_string = str(getattr(result, 'error_string', '')).strip()
        if status != 4 or error_code != 0:
            detail = f'status={status} error_code={error_code}'
            if error_string:
                detail += f' {error_string}'
            return False, f'{action_name}: {detail}'

        self.joint_positions['xarm6'] = [float(position) for position in target_joints]
        for name, position in zip(joint_names, target_joints, strict=True):
            self.joint_state_map[name] = float(position)
        return True, f'{action_name}: succeeded'

    def _move_ur5e_arm_action(self, joint_names, target_joints, duration_sec):
        """Send a real UR5e joint target through the guarded RTDE action."""
        client = self.ur5e_hardware_trajectory_client
        if client is None or FollowJointTrajectory is None:
            return False, 'UR5e RTDE FollowJointTrajectory action type is unavailable'
        if not client.wait_for_server(timeout_sec=2.0):
            return False, f'{self.ur5e_hardware_trajectory_action} is not available'

        current_positions = self.joint_positions.get('ur5e')
        if current_positions is None or len(current_positions) != len(joint_names):
            return False, 'UR5e current joint state is unavailable'

        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = list(joint_names)
        start_point = JointTrajectoryPoint()
        start_point.positions = [float(position) for position in current_positions]
        start_point.time_from_start = self._duration_msg(0.0)
        target_point = JointTrajectoryPoint()
        target_point.positions = [float(position) for position in target_joints]
        target_point.time_from_start = self._duration_msg(duration_sec)
        goal.trajectory.points = [start_point, target_point]
        try:
            send_future = client.send_goal_async(goal)
        except RuntimeError as exc:
            return False, f'{self.ur5e_hardware_trajectory_action}: send failed ({exc})'
        if not self._wait_future(send_future, timeout=3.0):
            return False, f'{self.ur5e_hardware_trajectory_action}: send timeout'
        try:
            goal_handle = send_future.result()
        except RuntimeError as exc:
            return False, f'{self.ur5e_hardware_trajectory_action}: send failed ({exc})'
        if goal_handle is None or not goal_handle.accepted:
            return False, f'{self.ur5e_hardware_trajectory_action}: goal rejected'

        result_future = goal_handle.get_result_async()
        result_timeout = max(
            10.0,
            float(duration_sec) + 20.0,
            float(getattr(self, 'ur5e_hardware_result_timeout_sec', 45.0)),
        )
        if not self._wait_future(result_future, timeout=result_timeout):
            return False, f'{self.ur5e_hardware_trajectory_action}: result timeout'
        try:
            wrapped = result_future.result()
        except RuntimeError as exc:
            return False, f'{self.ur5e_hardware_trajectory_action}: result failed ({exc})'
        result = getattr(wrapped, 'result', None)
        error_code = int(getattr(result, 'error_code', -1))
        error_string = str(getattr(result, 'error_string', '')).strip()
        if error_code != 0:
            detail = f'error_code={error_code}'
            if error_string:
                detail += f' {error_string}'
            return False, f'{self.ur5e_hardware_trajectory_action}: {detail}'

        self.joint_positions['ur5e'] = [float(position) for position in target_joints]
        for name, position in zip(joint_names, target_joints, strict=True):
            self.joint_state_map[name] = float(position)
        return True, f'{self.ur5e_hardware_trajectory_action}: succeeded'

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

        # On xArm hardware, the service path is repeatable for rapid teleop commands.
        if robot == 'xarm6':
            ok, msg = self._move_xarm_gripper_service(target, velocity_scale=velocity_scale)
            if ok:
                self.joint_state_map[joint_name] = target
                return True, f'{joint_name}={target:.3f}'
            service_error = msg
            ok, msg = self._move_xarm_gripper_action(target)
            if ok:
                self.joint_state_map[joint_name] = target
                return True, f'{joint_name}={target:.3f}'
            errors = [service_error, msg]
        elif robot == 'ur5e':
            ok, msg = self._move_ur5e_rg2_gripper_action(joint_name, target, duration)
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
        ur5e_hardware_trajectory_action=args.ur5e_hardware_trajectory_action,
        ur5e_hardware_result_timeout_sec=args.ur5e_hardware_result_timeout_sec,
    )
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    planning_ready = False
    gripper_ready = {'xarm6': False, 'ur5e': False}
    active_request_id = None

    def emit(payload):
        body = dict(payload)
        if active_request_id:
            body['request_id'] = active_request_id
        sys.stdout.write(json.dumps(body) + '\n')
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
            active_request_id = None
            line = raw.strip()
            if not line:
                continue

            try:
                cmd = json.loads(line)
            except Exception as exc:
                emit({'ok': False, 'msg': f'invalid json: {exc}'})
                continue

            active_request_id = str(cmd.get('request_id') or '').strip() or None

            op = str(cmd.get('op', '')).strip().lower()
            robot = str(cmd.get('robot', args.robot)).strip().lower()
            if robot not in ROBOTS:
                emit({'ok': False, 'msg': f'unknown robot: {robot}'})
                continue

            if op == 'shutdown':
                if node._ur5e_smooth_active:
                    node._stop_ur5e_cartesian_jog()
                if node._xarm6_cartesian_session_mode != 'off':
                    node._close_xarm6_cartesian_session()
                elif node._xarm6_smooth_active:
                    node._stop_xarm6_cartesian_jog()
                emit({'ok': True, 'msg': 'bye'})
                break

            if op == 'stationary_readiness':
                if not wait_for_joint_positions(
                    node,
                    [robot],
                    timeout_sec=service_timeout_sec,
                ):
                    emit({'ok': False, 'msg': f'no joint state for {robot}'})
                    continue
                if node.infer_robot_environment(robot) != 'real':
                    emit({
                        'ok': False,
                        'msg': 'Stationary hardware readiness requires hardware',
                    })
                    continue
                ok, msg, diagnostics = node._stationary_readiness(
                    robot,
                    velocity_limit_rad_s=float(
                        cmd.get('velocity_limit_rad_s', 0.01)
                    ),
                    hold_sec=float(cmd.get('hold_sec', 0.25)),
                    timeout_sec=float(cmd.get('timeout_sec', 3.0)),
                )
                emit({
                    'ok': bool(ok),
                    'msg': msg,
                    'stationary_ready': bool(ok),
                    'diagnostics': diagnostics,
                })
                continue

            if op == 'cartesian_readiness':
                if not wait_for_joint_positions(
                    node,
                    [robot],
                    timeout_sec=service_timeout_sec,
                ):
                    emit({'ok': False, 'msg': f'no joint state for {robot}'})
                    continue
                if node.infer_robot_environment(robot) != 'real':
                    emit({'ok': False, 'msg': 'Cartesian hardware readiness requires hardware'})
                    continue
                if robot == 'xarm6':
                    session_mode = str(node._xarm6_cartesian_session_mode or 'off')
                    expected_mode = {'step': 0, 'smooth': 5}.get(session_mode, 1)
                    ok, msg, diagnostics = node._xarm6_cartesian_readiness(
                        expected_mode=expected_mode,
                    )
                    emit({
                        'ok': bool(ok),
                        'msg': msg,
                        'cartesian_jog_ready': bool(ok),
                        'cartesian_function_ready': bool(ok),
                        'cartesian_mode': session_mode,
                        'cartesian_mode_ready': bool(ok and session_mode != 'off'),
                        'diagnostics': diagnostics,
                    })
                    continue
                action_ready = bool(
                    node.ur5e_hardware_relative_cartesian_client is not None
                    and node.ur5e_hardware_relative_cartesian_client.wait_for_server(
                        timeout_sec=2.0
                    )
                )
                service_ready = bool(
                    node.ur5e_hardware_cartesian_jog_client is not None
                    and node.ur5e_hardware_cartesian_jog_client.wait_for_service(
                        timeout_sec=2.0
                    )
                )
                ok = action_ready and service_ready
                emit({
                    'ok': ok,
                    'msg': (
                        'UR5e Cartesian direct interfaces ready'
                        if ok
                        else 'UR5e relative Cartesian action or jog service is unavailable'
                    ),
                    'cartesian_jog_ready': ok,
                    'cartesian_function_ready': ok,
                })
                continue

            if op == 'cartesian_mode':
                mode = str(cmd.get('mode', '')).strip().lower()
                if robot != 'xarm6':
                    accepted = mode in {'off', 'step', 'smooth', 'status'}
                    emit({
                        'ok': accepted,
                        'msg': (
                            f'{robot} Cartesian {mode or "mode"} ready'
                            if accepted
                            else f'unknown Cartesian mode: {mode}'
                        ),
                        'cartesian_mode': mode if mode in {'off', 'step', 'smooth'} else 'off',
                        'cartesian_mode_ready': bool(accepted and mode in {'step', 'smooth'}),
                        'state_uncertain': False,
                    })
                    continue
                ok, msg, state_uncertain = node.xarm6_cartesian_session(mode)
                current_mode = str(node._xarm6_cartesian_session_mode or 'off')
                emit({
                    'ok': bool(ok),
                    'msg': msg,
                    'cartesian_mode': current_mode,
                    'cartesian_mode_ready': bool(ok and current_mode != 'off'),
                    'state_uncertain': bool(state_uncertain),
                })
                continue

            if op == 'prepare_xarm6_trajectory_mode':
                if robot != 'xarm6':
                    emit({
                        'ok': False,
                        'msg': 'prepare_xarm6_trajectory_mode requires xarm6',
                    })
                    continue
                ok, msg, diagnostics = node._prepare_xarm6_trajectory_mode()
                emit({
                    'ok': bool(ok),
                    'msg': msg,
                    'diagnostics': diagnostics,
                })
                continue

            if op == 'cartesian_smooth':
                command = str(cmd.get('command', '')).strip().lower()
                stop = command == 'stop'
                if command not in {'start', 'update', 'stop'}:
                    emit({'ok': False, 'msg': f'unknown Cartesian Smooth Hold command: {command}'})
                    continue
                if node.infer_robot_environment(robot) != 'real':
                    emit({'ok': False, 'msg': 'Cartesian Smooth Hold is hardware-only'})
                    continue
                axis = str(cmd.get('axis', '')).strip().lower()
                try:
                    speed_mm_s = float(cmd.get('speed_mm_s', 0.0))
                    watchdog_sec = float(cmd.get('watchdog_sec', 0.50))
                except (TypeError, ValueError):
                    emit({'ok': False, 'msg': 'invalid Cartesian Smooth Hold speed or watchdog'})
                    continue
                ok, msg = node.set_cartesian_jog(
                    robot,
                    axis,
                    speed_mm_s,
                    stop=stop,
                    watchdog_sec=watchdog_sec,
                )
                response = {'ok': bool(ok), 'msg': msg}
                if robot == 'xarm6' and stop:
                    response['state_uncertain'] = bool(
                        not node._xarm6_last_stop_motion_confirmed
                    )
                elif robot == 'ur5e' and stop:
                    response['state_uncertain'] = bool(
                        not node._ur5e_last_stop_motion_confirmed
                    )
                emit(response)
                continue

            if op == 'cartesian':
                axis = str(cmd.get('axis', '')).strip().lower()
                if axis not in ('x', 'y', 'z'):
                    emit({'ok': False, 'msg': f'unknown axis: {axis}'})
                    continue
                try:
                    step_mm = float(cmd.get('step_mm'))
                    speed_mm_s = (
                        None
                        if cmd.get('speed_mm_s') is None
                        else float(cmd.get('speed_mm_s'))
                    )
                except Exception:
                    emit({'ok': False, 'msg': 'missing or invalid Step or Cartesian speed'})
                    continue
                velocity_scale = node._normalize_velocity_scale(cmd.get('velocity_scale', 1.0))

                if node.infer_robot_environment(robot) == 'real':
                    ok = wait_for_joint_positions(
                        node,
                        [robot],
                        timeout_sec=service_timeout_sec,
                    )
                    msg = 'OK' if ok else f'no joint state for {robot}'
                else:
                    ok, msg = ensure_planning(robot)
                if not ok:
                    emit({'ok': False, 'msg': msg})
                    continue

                kwargs = {'dx_mm': 0.0, 'dy_mm': 0.0, 'dz_mm': 0.0}
                kwargs[f'd{axis}_mm'] = step_mm
                ok, msg = node.move_cartesian(
                    robot,
                    velocity_scale=velocity_scale,
                    speed_mm_s=speed_mm_s,
                    xarm6_cartesian_session=bool(
                        robot == 'xarm6'
                        and node._xarm6_cartesian_session_mode == 'step'
                    ),
                    **kwargs,
                )
                response = {'ok': bool(ok), 'msg': msg}
                if robot == 'xarm6':
                    response['state_uncertain'] = bool(
                        not ok and node._xarm6_cartesian_motion_attempted
                    )
                emit(response)
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

            if op == 'move_joints':
                positions = cmd.get('positions')
                if not isinstance(positions, list) or len(positions) != 6:
                    emit({'ok': False, 'msg': 'positions must be a list of 6 floats'})
                    continue
                try:
                    targets = [float(p) for p in positions]
                except (TypeError, ValueError):
                    emit({'ok': False, 'msg': 'positions must be numeric'})
                    continue
                if not wait_for_joint_positions(node, [robot], timeout_sec=service_timeout_sec):
                    emit({'ok': False, 'msg': f'no joint state for {robot}'})
                    continue
                ok, msg = node.move_arm_to_joints(robot, targets, duration_sec=home_duration_sec)
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
                    speed_deg_s = (
                        None
                        if cmd.get('speed_deg_s') is None
                        else float(cmd.get('speed_deg_s'))
                    )
                except Exception:
                    emit({'ok': False, 'msg': 'missing or invalid joint delta or speed'})
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
                    speed_deg_s=speed_deg_s,
                )
                response = {'ok': bool(ok), 'msg': msg}
                if robot == 'ur5e' and speed_deg_s is not None:
                    response['state_uncertain'] = bool(
                        node._last_ur5e_joint_jog_state_uncertain
                    )
                emit(response)
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
        if node._ur5e_smooth_active:
            node._stop_ur5e_cartesian_jog()
        if node._xarm6_cartesian_session_mode != 'off':
            node._close_xarm6_cartesian_session()
        elif node._xarm6_smooth_active:
            node._stop_xarm6_cartesian_jog()
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
        ur5e_hardware_trajectory_action=args.ur5e_hardware_trajectory_action,
        ur5e_hardware_result_timeout_sec=args.ur5e_hardware_result_timeout_sec,
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
    parser.add_argument(
        '--ur5e-hardware-trajectory-action',
        default='/cais_ur5e_rtde_trajectory_controller/follow_joint_trajectory',
        help='guarded FollowJointTrajectory action for real UR5e arm commands',
    )
    parser.add_argument(
        '--ur5e-hardware-result-timeout-sec',
        type=float,
        default=45.0,
        help='maximum wait for a guarded real UR5e trajectory result',
    )
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
        ur5e_hardware_trajectory_action=args.ur5e_hardware_trajectory_action,
        ur5e_hardware_result_timeout_sec=args.ur5e_hardware_result_timeout_sec,
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
