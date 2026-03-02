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

# ── Robot definitions ────────────────────────────────────────────────────────

ROBOTS = {
    'xarm6': {
        'joint_names': [
            'xarm6_joint1', 'xarm6_joint2', 'xarm6_joint3',
            'xarm6_joint4', 'xarm6_joint5', 'xarm6_joint6',
        ],
        'group_name': 'xarm6_xarm6',
        'ee_link': 'xarm6_link_eef',
        'arm_controller_topic': '/xarm6_xarm6_traj_controller/joint_trajectory',
        'gripper_joint': 'xarm6_drive_joint',
        'gripper_controller_topic': '/xarm6_xarm_gripper_traj_controller/joint_trajectory',
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
        'group_name': 'ur5e_ur_manipulator',
        'ee_link': 'ur5e_tool0',
        'arm_controller_topic': '/ur5e_joint_trajectory_controller/joint_trajectory',
        'gripper_joint': 'ur5e_rg2_finger_width',
        'gripper_controller_topic': '/ur5e_rg2_gripper_traj_controller/joint_trajectory',
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
HOME_PRELIFT_Z_M = 1.40


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
        for robot_name, cfg in ROBOTS.items():
            self.arm_publishers[robot_name] = self.create_publisher(
                JointTrajectory, cfg['arm_controller_topic'], 10)
            self.gripper_publishers[robot_name] = self.create_publisher(
                JointTrajectory, cfg['gripper_controller_topic'], 10)

    def _joint_state_cb(self, msg):
        for jname, pos in zip(msg.name, msg.position):
            self.joint_state_map[jname] = pos
        for robot_name, cfg in ROBOTS.items():
            positions = {}
            for jname, pos in zip(msg.name, msg.position):
                if jname in cfg['joint_names']:
                    positions[jname] = pos
            if len(positions) == len(cfg['joint_names']):
                self.joint_positions[robot_name] = [
                    positions[n] for n in cfg['joint_names']]

    @staticmethod
    def _duration_msg(seconds):
        sec = int(seconds)
        nanosec = int((seconds - sec) * 1e9)
        return Duration(sec=sec, nanosec=nanosec)

    @staticmethod
    def _clamp(value, min_value, max_value):
        return min(max(value, min_value), max_value)

    @staticmethod
    def _wait_for_subscriber(pub, timeout_sec=1.0):
        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            if pub.get_subscription_count() > 0:
                return True
            time.sleep(0.01)
        return pub.get_subscription_count() > 0

    def _publish_joint_trajectory(self, publisher, joint_names, positions, duration_sec):
        if not self._wait_for_subscriber(publisher, timeout_sec=1.0):
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
        ee_link = ROBOTS[robot]['ee_link']
        try:
            t = self.tf_buffer.lookup_transform('world', ee_link, rclpy.time.Time())
            pose = Pose()
            pose.position.x = t.transform.translation.x
            pose.position.y = t.transform.translation.y
            pose.position.z = t.transform.translation.z
            pose.orientation = t.transform.rotation
            return pose
        except Exception:
            return None

    @staticmethod
    def _wait_future(future, timeout=30.0):
        deadline = time.time() + timeout
        while not future.done() and time.time() < deadline:
            time.sleep(0.01)
        return future.done()

    def move_cartesian(self, robot, dx_mm=0, dy_mm=0, dz_mm=0):
        """Move end-effector by a delta in mm. Returns (ok, message)."""
        ee = self.get_ee_pose(robot)
        if ee is None:
            return False, 'No TF data'

        target = copy.deepcopy(ee)
        target.position.x += dx_mm / 1000.0
        target.position.y += dy_mm / 1000.0
        target.position.z += dz_mm / 1000.0

        cfg = ROBOTS[robot]
        request = GetCartesianPath.Request()
        request.header.frame_id = 'world'
        request.header.stamp = self.get_clock().now().to_msg()
        request.group_name = cfg['group_name']
        request.link_name = cfg['ee_link']
        request.waypoints = [target]
        request.max_step = self.cartesian_max_step_m
        request.jump_threshold = 0.0
        request.avoid_collisions = False
        request.start_state.is_diff = True

        cart_future = self.cartesian_client.call_async(request)
        if not self._wait_future(cart_future, timeout=10.0):
            return False, 'CartesianPath timeout'
        response = cart_future.result()
        if response is None:
            return False, 'CartesianPath failed'
        if response.fraction < 0.9:
            return False, f'Path incomplete ({response.fraction:.0%})'

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

    def move_joint(self, robot, joint_idx, delta_deg):
        """Jog a single arm joint through trajectory controller."""
        if robot not in self.joint_positions:
            return False, 'No joint state'

        cfg = ROBOTS[robot]
        if joint_idx < 0 or joint_idx >= len(cfg['joint_names']):
            return False, 'Invalid joint index'
        target = list(self.joint_positions[robot])
        target[joint_idx] += math.radians(delta_deg)

        return self.move_arm_to_joints(robot, target, duration_sec=self.joint_duration_sec)

    def move_arm_to_joints(self, robot, target_joints, duration_sec=None):
        """Move arm to explicit joint targets through trajectory controller."""
        cfg = ROBOTS[robot]
        if len(target_joints) != len(cfg['joint_names']):
            return False, f'Expected {len(cfg["joint_names"])} joints, got {len(target_joints)}'
        move_duration = self.joint_duration_sec if duration_sec is None else max(0.05, float(duration_sec))

        ok, msg = self._publish_joint_trajectory(
            self.arm_publishers[robot],
            cfg['joint_names'],
            target_joints,
            duration_sec=move_duration,
        )
        if not ok:
            return False, msg

        # Keep a local optimistic state so rapid repeated keypresses accumulate.
        self.joint_positions[robot] = list(target_joints)
        for name, pos in zip(cfg['joint_names'], target_joints):
            self.joint_state_map[name] = pos
        return True, 'OK'

    def move_gripper(self, robot, direction, step_size):
        """Jog gripper open/close by step size."""
        cfg = ROBOTS[robot]
        joint_name = cfg['gripper_joint']
        current = self.joint_state_map.get(joint_name)
        if current is None:
            return False, f'No joint state for {joint_name}'

        open_pos = cfg['gripper_open']
        close_pos = cfg['gripper_close']
        open_sign = 1.0 if open_pos > close_pos else -1.0
        step = abs(step_size)
        signed_step = open_sign * step if direction == 'open' else -open_sign * step
        target = self._clamp(current + signed_step, min(open_pos, close_pos), max(open_pos, close_pos))

        if abs(target - current) < 1e-6:
            return True, 'Gripper already at limit'

        ok, msg = self._publish_joint_trajectory(
            self.gripper_publishers[robot],
            [joint_name],
            [target],
            duration_sec=self.gripper_duration_sec,
        )
        if not ok:
            return False, msg

        self.joint_state_map[joint_name] = target
        return True, f'{joint_name}={target:.3f}'

    def save_position(self, robot, name):
        if robot not in self.joint_positions:
            return None, None
        positions = [round(p, 6) for p in self.joint_positions[robot]]
        path = DEFAULT_CONFIG_PATHS[robot]
        if path.exists():
            with open(path) as f:
                data = json.load(f)
        else:
            data = {}
        # Write into the robot's gazebo environment block.
        robot_block = data.setdefault(robot, {})
        gazebo_block = robot_block.setdefault('gazebo', {})
        named = gazebo_block.setdefault('named_positions', {})
        named[name] = positions
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
            f.write('\n')
        return positions, str(path)

    def load_named_position(self, robot, name):
        """Load a named joint position from robot config."""
        path = DEFAULT_CONFIG_PATHS[robot]
        if not path.exists():
            return None, f'Config not found: {path}'
        try:
            with open(path) as f:
                data = json.load(f)
        except Exception as exc:
            return None, f'Failed reading config: {exc}'

        candidates = []
        if isinstance(data.get('named_positions'), dict):
            candidates.append(data['named_positions'])
        robot_block = data.get(robot)
        if isinstance(robot_block, dict):
            if isinstance(robot_block.get('named_positions'), dict):
                candidates.append(robot_block['named_positions'])
            # Check environment sub-blocks (gazebo / real).
            for env in ('gazebo', 'real'):
                env_block = robot_block.get(env, {})
                if isinstance(env_block.get('named_positions'), dict):
                    candidates.append(env_block['named_positions'])

        for named in candidates:
            values = named.get(name)
            if not isinstance(values, list):
                continue
            if len(values) != len(ROBOTS[robot]['joint_names']):
                return None, f'Named position "{name}" has {len(values)} joints (expected {len(ROBOTS[robot]["joint_names"])})'
            try:
                return [float(v) for v in values], None
            except Exception:
                return None, f'Named position "{name}" contains non-numeric values'

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
    parser.add_argument('--home-duration-sec', type=float, default=2.5,
                        help='home command duration for both arms (seconds)')
    parser.add_argument('--key-poll-ms', type=float, default=8.0,
                        help='keyboard polling interval in milliseconds')
    parser.add_argument('--robot', choices=['xarm6', 'ur5e'], default='xarm6')
    args = parser.parse_args()

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
    log(f'  H = pre-lift to Z>={fmt_value(HOME_PRELIFT_Z_M)}m, then both arms home ({fmt_value(home_duration_sec)}s)')
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
            ee = node.get_ee_pose(rob)
            if ee is None:
                failures.append(f'[{rob}] No TF for pre-lift')
                continue
            if ee.position.z < HOME_PRELIFT_Z_M - 1e-3:
                dz_mm = (HOME_PRELIFT_Z_M - ee.position.z) * 1000.0
                ok, msg = node.move_cartesian(rob, dz_mm=dz_mm)
                if not ok:
                    failures.append(f'[{rob}] pre-lift failed: {msg}')
                    continue

            target, err = node.load_named_position(rob, 'home')
            if target is None:
                failures.append(f'[{rob}] {err}')
                continue
            ok, msg = node.move_arm_to_joints(rob, target, duration_sec=home_duration_sec)
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
                    cfg = ROBOTS[robot]
                    cur = node.joint_state_map.get(cfg['gripper_joint'])
                    cur_s = f'{cur:.3f}' if cur is not None else 'n/a'
                    log(f'--- Gripper mode [{robot.upper()}] joint={cfg["gripper_joint"]} '
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


if __name__ == '__main__':
    main()
