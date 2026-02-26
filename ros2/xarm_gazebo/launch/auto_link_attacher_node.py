#!/usr/bin/env python3
"""Automatic IFRA LinkAttacher bridge for xArm6 + UR5e grippers."""

import math
import os
import sys
import importlib
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState
from tf2_ros import Buffer, TransformException, TransformListener


PART_NAMES = [
    'gear_small',
    'rect_pin_small',
    'circ_pin_small',
    'gear_medium',
    'rect_pin_medium',
    'circ_pin_medium',
    'gear_large',
    'rect_pin_large',
    'circ_pin_large',
]

DEFAULT_PART_POSES = {
    'gear_small': (-0.4, 0.1, 1.15),
    'rect_pin_small': (-0.4, 0.0, 1.15),
    'circ_pin_small': (-0.4, -0.1, 1.15),
    'gear_medium': (0.4, 0.5, 1.15),
    'rect_pin_medium': (0.4, 0.4, 1.15),
    'circ_pin_medium': (0.4, 0.3, 1.15),
    'gear_large': (0.4, -0.3, 1.15),
    'rect_pin_large': (0.4, -0.4, 1.15),
    'circ_pin_large': (0.4, -0.5, 1.15),
}


def _import_linkattacher_srvs():
    """Import IFRA service types, even if workspace setup wasn't sourced."""
    def _load_srvs():
        srv_module = importlib.import_module('linkattacher_msgs.srv')
        return srv_module.AttachLink, srv_module.DetachLink

    try:
        return _load_srvs()
    except ModuleNotFoundError:
        py_ver = f'python{sys.version_info.major}.{sys.version_info.minor}'
        candidates = [
            os.path.expanduser(f'~/ros2_ws/install/linkattacher_msgs/local/lib/{py_ver}/dist-packages'),
            os.path.expanduser(f'~/ros2_ws/install/ros2_linkattacher/local/lib/{py_ver}/dist-packages'),
            f'/opt/ros/humble/lib/{py_ver}/dist-packages',
        ]
        for path in candidates:
            if os.path.isdir(path) and path not in sys.path:
                sys.path.append(path)
        return _load_srvs()


class AutoLinkAttacher(Node):
    def __init__(self) -> None:
        super().__init__('auto_link_attacher')
        attach_srv, detach_srv = _import_linkattacher_srvs()
        self.attach_srv = attach_srv
        self.detach_srv = detach_srv

        self.declare_parameter('world_frame', 'world')
        self.declare_parameter('model_name', 'dual_robot')
        self.declare_parameter('attach_distance_threshold', 0.22)
        self.declare_parameter('finger_distance_threshold', 0.25)
        self.declare_parameter('require_finger_consensus', True)
        self.declare_parameter('allow_tcp_fallback', True)
        self.declare_parameter('xarm_close_threshold', 0.75)
        self.declare_parameter('xarm_open_threshold', 0.08)
        self.declare_parameter('ur5e_close_threshold', 0.070)
        self.declare_parameter('ur5e_open_threshold', 0.100)
        self.declare_parameter('open_confirm_cycles', 6)

        self.world_frame = self.get_parameter('world_frame').value
        self.robot_model_name = self.get_parameter('model_name').value
        self.attach_distance_threshold = float(self.get_parameter('attach_distance_threshold').value)
        self.finger_distance_threshold = float(self.get_parameter('finger_distance_threshold').value)
        self.require_finger_consensus = bool(self.get_parameter('require_finger_consensus').value)
        self.allow_tcp_fallback = bool(self.get_parameter('allow_tcp_fallback').value)
        self.xarm_close_threshold = float(self.get_parameter('xarm_close_threshold').value)
        self.xarm_open_threshold = float(self.get_parameter('xarm_open_threshold').value)
        self.ur5e_close_threshold = float(self.get_parameter('ur5e_close_threshold').value)
        self.ur5e_open_threshold = float(self.get_parameter('ur5e_open_threshold').value)
        self.open_confirm_cycles = int(self.get_parameter('open_confirm_cycles').value)

        # Preferred link order for ATTACH/DETACH. We try prettier grasp-centric links first,
        # and fall back to robust wrist links if Gazebo doesn't expose a candidate link.
        self.attach_link_candidates: Dict[str, List[str]] = {
            # link_tcp/link_eef are not exposed as Gazebo physics links in this setup.
            # Use real gripper collision links first, then fall back to link6.
            'xarm': [
                'xarm6_right_inner_knuckle',
                'xarm6_left_inner_knuckle',
                'xarm6_right_finger',
                'xarm6_left_finger',
                'xarm6_link6',
            ],
            'ur5e': ['ur5e_rg2_gripper_tcp', 'ur5e_tool0', 'ur5e_wrist_3_link'],
        }
        self.attach_link_index = {'xarm': 0, 'ur5e': 0}
        # Link used for proximity checks (actual grasp TCP vicinity).
        self.pose_links = {
            'xarm': 'xarm6_link_tcp',
            'ur5e': 'ur5e_rg2_gripper_tcp',
        }
        # Finger links used to verify both jaws are actually around the same part.
        self.finger_links = {
            'xarm': ('xarm6_left_finger', 'xarm6_right_finger'),
            'ur5e': ('ur5e_rg2_left_finger_tip', 'ur5e_rg2_right_finger_tip'),
        }
        self.gripper_joints = {
            'xarm': 'xarm6_drive_joint',
            'ur5e': 'ur5e_rg2_finger_width',
        }

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.attach_client = self.create_client(self.attach_srv, '/ATTACHLINK')
        self.detach_client = self.create_client(self.detach_srv, '/DETACHLINK')
        self.create_subscription(
            JointState,
            '/joint_states',
            self._joint_state_cb,
            qos_profile_sensor_data,
        )

        self.joint_positions: Dict[str, float] = {}
        self.part_positions = self._load_part_positions()

        self.xarm_closed: Optional[bool] = None
        self.ur5e_closed: Optional[bool] = None
        self.attached_by_robot: Dict[str, Optional[str]] = {
            'xarm': None,
            'ur5e': None,
        }
        self.attached_owner_by_model: Dict[str, str] = {}
        self.attached_link_by_robot: Dict[str, Optional[str]] = {
            'xarm': None,
            'ur5e': None,
        }
        self.last_attach_skip_reason = {'xarm': '', 'ur5e': ''}
        self.open_counts = {'xarm': 0, 'ur5e': 0}

        self.pending_future = None
        self.pending_action: Optional[str] = None
        self.pending_robot: Optional[str] = None
        self.pending_model: Optional[str] = None
        self.pending_link: Optional[str] = None

        self.ready_logged = False
        self.create_timer(0.10, self._timer_cb)

    def _load_part_positions(self) -> Dict[str, Tuple[float, float, float]]:
        positions = dict(DEFAULT_PART_POSES)
        try:
            world_path = Path(get_package_share_directory('xarm_gazebo')) / 'worlds' / 'table.world'
            root = ET.parse(str(world_path)).getroot()
            world_elem = root.find('world')
            if world_elem is None:
                return positions
            for model in world_elem.findall('model'):
                name = model.get('name', '')
                if name not in PART_NAMES:
                    continue
                pose_elem = model.find('pose')
                if pose_elem is None or not pose_elem.text:
                    continue
                xyzrpy = pose_elem.text.strip().split()
                if len(xyzrpy) < 3:
                    continue
                positions[name] = (float(xyzrpy[0]), float(xyzrpy[1]), float(xyzrpy[2]))
        except Exception as exc:
            self.get_logger().warn(f'Failed parsing world part poses, using defaults: {exc}')
        return positions

    def _joint_state_cb(self, msg: JointState) -> None:
        for name, pos in zip(msg.name, msg.position):
            self.joint_positions[name] = pos

    def _lookup_robot_pose(self, robot_key: str) -> Optional[Tuple[float, float, float]]:
        link_name = self.pose_links[robot_key]
        return self._lookup_link_pose(link_name)

    def _lookup_link_pose(self, link_name: str) -> Optional[Tuple[float, float, float]]:
        try:
            tf_msg = self.tf_buffer.lookup_transform(
                self.world_frame,
                link_name,
                rclpy.time.Time(),
            )
        except TransformException:
            return None
        t = tf_msg.transform.translation
        return (t.x, t.y, t.z)

    def _find_nearest_part(self, point: Tuple[float, float, float]) -> Tuple[Optional[str], float]:
        nearest_name = None
        nearest_dist = float('inf')
        for model_name, model_pos in self.part_positions.items():
            dx = point[0] - model_pos[0]
            dy = point[1] - model_pos[1]
            dz = point[2] - model_pos[2]
            dist = math.sqrt(dx * dx + dy * dy + dz * dz)
            if dist < nearest_dist:
                nearest_name = model_name
                nearest_dist = dist
        return nearest_name, nearest_dist

    def _current_attach_link(self, robot_key: str) -> str:
        idx = self.attach_link_index[robot_key]
        return self.attach_link_candidates[robot_key][idx]

    def _advance_attach_link(self, robot_key: str) -> bool:
        next_idx = self.attach_link_index[robot_key] + 1
        if next_idx >= len(self.attach_link_candidates[robot_key]):
            return False
        self.attach_link_index[robot_key] = next_idx
        return True

    def _resolve_attach_candidate(self, robot_key: str) -> Tuple[Optional[str], float, str]:
        tcp_pose = self._lookup_robot_pose(robot_key)
        if tcp_pose is None:
            return None, float('inf'), 'tcp transform unavailable'
        tcp_model, tcp_dist = self._find_nearest_part(tcp_pose)
        if tcp_model is None or tcp_dist > self.attach_distance_threshold:
            return (
                None,
                float('inf'),
                f'tcp too far from parts ({tcp_dist:.3f} m > {self.attach_distance_threshold:.3f} m)',
            )

        left_link, right_link = self.finger_links[robot_key]
        left_pose = self._lookup_link_pose(left_link)
        right_pose = self._lookup_link_pose(right_link)
        if left_pose is None or right_pose is None:
            if self.allow_tcp_fallback:
                return tcp_model, tcp_dist, 'tcp fallback (finger transforms unavailable)'
            return None, float('inf'), 'finger transforms unavailable'

        left_model, left_dist = self._find_nearest_part(left_pose)
        right_model, right_dist = self._find_nearest_part(right_pose)
        left_ok = left_model == tcp_model and left_dist <= self.finger_distance_threshold
        right_ok = right_model == tcp_model and right_dist <= self.finger_distance_threshold

        if left_ok and right_ok:
            return tcp_model, max(tcp_dist, left_dist, right_dist), 'finger consensus'

        if self.require_finger_consensus:
            if self.allow_tcp_fallback and (left_ok or right_ok or tcp_dist <= 0.08):
                fallback_side = 'left' if left_ok else ('right' if right_ok else 'tcp-near')
                return tcp_model, tcp_dist, f'tcp fallback ({fallback_side})'
            return None, float('inf'), (
                f'finger mismatch for {tcp_model} '
                f'(left: {left_model} {left_dist:.3f} m, right: {right_model} {right_dist:.3f} m)'
            )

        if left_ok or right_ok:
            dists = [tcp_dist]
            if left_ok:
                dists.append(left_dist)
            if right_ok:
                dists.append(right_dist)
            return tcp_model, max(dists), 'single-finger consensus'

        if self.allow_tcp_fallback:
            return tcp_model, tcp_dist, 'tcp fallback (consensus disabled)'

        return None, float('inf'), 'no finger close to tcp target'

    def _timer_cb(self) -> None:
        if not self.ready_logged:
            if self.attach_client.service_is_ready() and self.detach_client.service_is_ready():
                self.ready_logged = True
                self.get_logger().info('Auto LinkAttacher ready')
            else:
                return

        if self.pending_future is not None:
            if not self.pending_future.done():
                return
            self._handle_pending_result()
            return

        for robot_key, model_name in self.attached_by_robot.items():
            if model_name is None:
                continue
            ee_pose = self._lookup_robot_pose(robot_key)
            if ee_pose is not None:
                self.part_positions[model_name] = ee_pose

        self._update_xarm_state()
        self._update_ur5e_state()

    def _update_xarm_state(self) -> None:
        pos = self.joint_positions.get(self.gripper_joints['xarm'])
        if pos is None:
            return

        # xArm gripper convention: open ~= 0.0, close ~= 0.85
        closed_now = pos >= self.xarm_close_threshold
        open_now = pos <= self.xarm_open_threshold
        self.open_counts['xarm'] = self.open_counts['xarm'] + 1 if open_now else 0

        if self.xarm_closed is None:
            self.xarm_closed = closed_now
            return

        # Attach only on explicit open -> close transition.
        if self.attached_by_robot['xarm'] is None and (not self.xarm_closed) and closed_now:
            self._try_attach('xarm')
        # Detach only after open is sustained for a few cycles (debounce).
        elif self.attached_by_robot['xarm'] is not None and self.open_counts['xarm'] >= self.open_confirm_cycles:
            self._try_detach('xarm')

        self.xarm_closed = closed_now

    def _update_ur5e_state(self) -> None:
        pos = self.joint_positions.get(self.gripper_joints['ur5e'])
        if pos is None:
            return

        closed_now = pos <= self.ur5e_close_threshold
        open_now = pos >= self.ur5e_open_threshold
        self.open_counts['ur5e'] = self.open_counts['ur5e'] + 1 if open_now else 0

        if self.ur5e_closed is None:
            self.ur5e_closed = closed_now
            return

        # Attach only on explicit open -> close transition.
        if self.attached_by_robot['ur5e'] is None and (not self.ur5e_closed) and closed_now:
            self._try_attach('ur5e')
        # Detach only after open is sustained for a few cycles (debounce).
        elif self.attached_by_robot['ur5e'] is not None and self.open_counts['ur5e'] >= self.open_confirm_cycles:
            self._try_detach('ur5e')

        self.ur5e_closed = closed_now

    def _try_attach(self, robot_key: str) -> None:
        if self.attached_by_robot[robot_key] is not None:
            return
        model_name, dist, reason = self._resolve_attach_candidate(robot_key)
        if model_name is None:
            if self.last_attach_skip_reason[robot_key] != reason:
                self.last_attach_skip_reason[robot_key] = reason
                self.get_logger().info(f'Attach skipped ({robot_key}): {reason}')
            return
        self.last_attach_skip_reason[robot_key] = ''
        owner = self.attached_owner_by_model.get(model_name)
        if owner is not None and owner != robot_key:
            conflict_reason = f'{model_name} is already attached to {owner}'
            if self.last_attach_skip_reason[robot_key] != conflict_reason:
                self.last_attach_skip_reason[robot_key] = conflict_reason
                self.get_logger().info(f'Attach skipped ({robot_key}): {conflict_reason}')
            return
        self._send_attach_request(robot_key, model_name, dist, reason)

    def _send_attach_request(self, robot_key: str, model_name: str, dist: float, reason: str) -> None:
        req = self.attach_srv.Request()
        req.model1_name = self.robot_model_name
        req.link1_name = self._current_attach_link(robot_key)
        req.model2_name = model_name
        req.link2_name = 'link'

        self.pending_future = self.attach_client.call_async(req)
        self.pending_action = 'attach'
        self.pending_robot = robot_key
        self.pending_model = model_name
        self.pending_link = req.link1_name
        dist_text = 'n/a' if not math.isfinite(dist) else f'{dist:.3f} m'
        self.get_logger().info(
            f'Attach request: {robot_key} -> {model_name} via {req.link1_name} (distance={dist_text}, rule={reason})'
        )

    def _try_detach(self, robot_key: str) -> None:
        model_name = self.attached_by_robot.get(robot_key)
        if model_name is None:
            return

        req = self.detach_srv.Request()
        req.model1_name = self.robot_model_name
        req.link1_name = self.attached_link_by_robot.get(robot_key) or self._current_attach_link(robot_key)
        req.model2_name = model_name
        req.link2_name = 'link'

        self.pending_future = self.detach_client.call_async(req)
        self.pending_action = 'detach'
        self.pending_robot = robot_key
        self.pending_model = model_name
        self.pending_link = req.link1_name
        self.get_logger().info(f'Detach request: {robot_key} -> {model_name} via {req.link1_name}')

    def _handle_pending_result(self) -> None:
        action = self.pending_action
        robot_key = self.pending_robot
        model_name = self.pending_model
        link_name = self.pending_link
        future = self.pending_future

        self.pending_action = None
        self.pending_robot = None
        self.pending_model = None
        self.pending_link = None
        self.pending_future = None

        try:
            resp = future.result()
        except Exception as exc:
            self.get_logger().warn(f'{action} call failed: {exc}')
            return

        if not resp.success:
            msg = getattr(resp, 'message', '')
            if action == 'attach' and 'Failed to find link with name:' in msg:
                old_link = link_name or self._current_attach_link(robot_key)
                if self._advance_attach_link(robot_key):
                    new_link = self._current_attach_link(robot_key)
                    self.get_logger().warn(
                        f'Attach link unavailable for {robot_key}: {old_link}; retrying with {new_link}'
                    )
                    self._send_attach_request(robot_key, model_name, float('nan'), 'link-fallback')
                    return
            self.get_logger().warn(f'{action} failed: {msg}')
            return

        if action == 'attach':
            self.attached_by_robot[robot_key] = model_name
            self.attached_owner_by_model[model_name] = robot_key
            self.attached_link_by_robot[robot_key] = link_name or self._current_attach_link(robot_key)
            self.get_logger().info(
                f'Attached {model_name} to {robot_key} via {self.attached_link_by_robot[robot_key]}'
            )
        elif action == 'detach':
            self.get_logger().info(f'Detached {model_name} from {robot_key}')
            self.attached_by_robot[robot_key] = None
            self.attached_link_by_robot[robot_key] = None
            self.attached_owner_by_model.pop(model_name, None)


def main() -> None:
    rclpy.init()
    node = AutoLinkAttacher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
