#!/usr/bin/env python3.10
"""
Record current robot joint positions and save with a name.

Works with any running ROS2 setup that publishes /joint_states.
Use after jogging the robot to a desired pose (via RViz drag, keyboard teleop,
or MoveIt Servo).

Usage:
    # Print current positions (no save):
    python3 record_pose.py --robot ur5e

    # Save a named position to stdout:
    python3 record_pose.py --robot ur5e --name "above_prusa_mk3"

    # Save to the robot's JSON config file:
    python3 record_pose.py --robot ur5e --name "above_prusa_mk3" --save

    # Save to a custom file:
    python3 record_pose.py --robot ur5e --name "home" --output positions.json

    # Batch mode — record multiple positions interactively:
    python3 record_pose.py --robot xarm6 --interactive
"""

import argparse
import json
import math
import sys
from pathlib import Path

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

# ── Robot definitions ────────────────────────────────────────────────────────

ROBOTS = {
    'ur5e': {
        'joint_names': [
            'ur5e_shoulder_pan_joint',
            'ur5e_shoulder_lift_joint',
            'ur5e_elbow_joint',
            'ur5e_wrist_1_joint',
            'ur5e_wrist_2_joint',
            'ur5e_wrist_3_joint',
        ],
    },
    'xarm6': {
        'joint_names': [
            'xarm6_joint1',
            'xarm6_joint2',
            'xarm6_joint3',
            'xarm6_joint4',
            'xarm6_joint5',
            'xarm6_joint6',
        ],
    },
}

DEFAULT_CONFIG_PATHS = {
    'xarm6': Path(__file__).resolve().parents[3]
             / 'cais_spade_llm' / 'initialization' / 'resources' / 'robot_xarm6.json',
    'ur5e': Path(__file__).resolve().parents[3]
            / 'cais_spade_llm' / 'initialization' / 'resources' / 'robot_ur5e.json',
}


class PoseRecorder(Node):
    def __init__(self, robot_name):
        super().__init__('pose_recorder')
        self.joint_names = ROBOTS[robot_name]['joint_names']
        self.latest = None
        self.create_subscription(JointState, '/joint_states', self._cb, 10)

    def _cb(self, msg):
        positions = {}
        for name, pos in zip(msg.name, msg.position):
            if name in self.joint_names:
                positions[name] = pos
        if len(positions) == len(self.joint_names):
            self.latest = [positions[n] for n in self.joint_names]


def wait_for_state(node, timeout=5.0):
    """Spin until we receive at least one joint state message."""
    import time
    start = time.time()
    while rclpy.ok() and node.latest is None:
        rclpy.spin_once(node, timeout_sec=0.1)
        if time.time() - start > timeout:
            return False
    return node.latest is not None


def print_positions(robot_name, joint_names, positions):
    """Pretty-print joint positions."""
    print(f'\n  {robot_name} joint positions:')
    for name, pos in zip(joint_names, positions):
        short = name.replace('xarm6_', '').replace('ur5e_', '')
        print(f'    {short:24s}  {math.degrees(pos):+8.2f} deg  ({pos:+.6f} rad)')


def save_to_json(filepath, name, positions, robot_name=None):
    """Merge a named position into a JSON file under the gazebo environment block."""
    filepath = Path(filepath)
    if filepath.exists():
        with open(filepath) as f:
            data = json.load(f)
    else:
        data = {}

    # Write into the robot's gazebo environment block when possible.
    if robot_name and robot_name in data:
        gazebo_block = data[robot_name].setdefault('gazebo', {})
        named = gazebo_block.setdefault('named_positions', {})
    else:
        named = data.setdefault('named_positions', {})
    named[name] = positions

    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
        f.write('\n')
    print(f'  -> Saved "{name}" to {filepath}')


def run_interactive(node, robot_name, save_path):
    """Interactive mode: repeatedly record positions until user quits."""
    joint_names = ROBOTS[robot_name]['joint_names']
    print(f'\n  Interactive recording for {robot_name}')
    print('  Jog the robot to a pose, then type a name here.')
    print('  Type "list" to show saved positions, "quit" to exit.\n')

    saved = {}
    while True:
        try:
            name = input('  Position name (or "quit"): ').strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not name or name == 'quit':
            break

        if name == 'list':
            if saved:
                print(json.dumps(saved, indent=4))
            else:
                print('  (no positions saved yet)')
            continue

        # Re-read current state
        node.latest = None
        if not wait_for_state(node, timeout=2.0):
            print('  ERROR: Lost connection to /joint_states')
            continue

        positions = [round(p, 6) for p in node.latest]
        print_positions(robot_name, joint_names, positions)
        saved[name] = positions

        if save_path:
            save_to_json(save_path, name, positions, robot_name=robot_name)
        else:
            print(f'  (use --save or --output to persist)')

    return saved


def main():
    parser = argparse.ArgumentParser(
        description='Record robot joint positions from /joint_states')
    parser.add_argument('--robot', required=True, choices=['ur5e', 'xarm6'],
                        help='Which robot to record')
    parser.add_argument('--name', help='Name for this position (e.g. "home")')
    parser.add_argument('--save', action='store_true',
                        help='Save to the robot default JSON config file')
    parser.add_argument('--output', help='Save to a custom JSON file')
    parser.add_argument('--interactive', action='store_true',
                        help='Interactive mode: record multiple positions')
    args = parser.parse_args()

    rclpy.init()
    node = PoseRecorder(args.robot)

    print(f'Waiting for /joint_states ({args.robot})...')
    if not wait_for_state(node):
        print('ERROR: No joint state received within 5 seconds.')
        print('Is the robot simulation running?')
        node.destroy_node()
        rclpy.shutdown()
        sys.exit(1)

    positions = [round(p, 6) for p in node.latest]
    joint_names = ROBOTS[args.robot]['joint_names']

    # Determine save path
    save_path = None
    if args.output:
        save_path = args.output
    elif args.save:
        save_path = str(DEFAULT_CONFIG_PATHS[args.robot])

    if args.interactive:
        run_interactive(node, args.robot, save_path)
    else:
        print_positions(args.robot, joint_names, positions)

        if args.name:
            print(f'\n  Named position "{args.name}":')
            print(f'  {json.dumps({args.name: positions}, indent=4)}')

            if save_path:
                save_to_json(save_path, args.name, positions, robot_name=args.robot)
            else:
                print('\n  (use --save or --output to persist to file)')

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
