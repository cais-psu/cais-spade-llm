"""Prepare saved Gazebo waypoints explicitly, outside production execution.

Use read-only MoveIt services or offline kinematics from captured robot parameters.
This module has no action clients, trajectory publishers, scene writes, or
production startup hooks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import time
import xml.etree.ElementTree as ET
from copy import deepcopy
from pathlib import Path

from cais_spade_llm.recovery_framework import ROOT, read_json
from cais_spade_llm.recovery_framework.geometry import compose, quaternion
from cais_spade_llm.recovery_framework.kmr_gazebo import inverse
from cais_spade_llm.resources.robot.cartesian_waypoints import resolve_waypoints, _segment_timing, continuous_joints, sample_segment
from cais_spade_llm.resources.robot.saved_waypoints import (
    SAVED_WAYPOINTS_FILE, source_fingerprints, validate_points,
)

logger = logging.getLogger(__name__)


def _values(pose) -> list[float]:
    p, q = pose.position, pose.orientation
    return [p.x, p.y, p.z, q.x, q.y, q.z, q.w]


def _pose(values):
    from geometry_msgs.msg import Pose

    pose = Pose()
    pose.position.x, pose.position.y, pose.position.z = map(float, values[:3])
    pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = map(float, values[3:])
    return pose


class WaypointPreparation:
    """Read kinematics and persist resource-owned waypoints before a run."""

    def __init__(self, node, inputs: dict, namespace: str = '', model_parameters: dict | None = None) -> None:
        """Bind read-only ROS clients to the selected saved configuration."""
        from rcl_interfaces.srv import GetParameters

        self.node, self.inputs, self.clients = node, inputs, {}
        self.namespace = namespace.rstrip('/')
        self.resources: dict = {}
        self.failures: list = []
        self.model_parameters = model_parameters
        self.chains: dict = {}
        self.motions: dict = {}
        self.description = (model_parameters['robot_description'] if model_parameters else
                            self.call(GetParameters, '/move_group/get_parameters',
                                      GetParameters.Request(names=['robot_description'])).values[0].string_value)
        self.urdf = ET.fromstring(self.description)

    def chain(self, resource: dict):
        """Read a KDL chain from the captured robot model for offline preparation."""
        import PyKDL as kdl

        key = resource['ee_link']
        if key in self.chains:
            return self.chains[key]
        parents = {joint.find('child').get('link'): joint for joint in self.urdf.findall('joint')}
        link, rows = key, []
        while link != resource['ros_frame']:
            joint = parents[link]
            rows.append((link, joint))
            link = joint.find('parent').get('link')
        chain = kdl.Chain()
        names = []
        for link, joint in reversed(rows):
            origin = joint.find('origin')
            xyz = [float(v) for v in origin.get('xyz', '0 0 0').split()] if origin is not None else [0.,0.,0.]
            rpy = [float(v) for v in origin.get('rpy', '0 0 0').split()] if origin is not None else [0.,0.,0.]
            frame = kdl.Frame(kdl.Rotation.RPY(*rpy), kdl.Vector(*xyz))
            if joint.get('type') == 'fixed':
                axis = kdl.Joint(joint.get('name'), kdl.Joint.Fixed)
            else:
                vector = kdl.Vector(*[float(v) for v in joint.find('axis').get('xyz').split()])
                kind = kdl.Joint.TransAxis if joint.get('type') == 'prismatic' else kdl.Joint.RotAxis
                axis = kdl.Joint(joint.get('name'), frame.p, frame.M * vector, kind)
                names.append(joint.get('name'))
            chain.addSegment(kdl.Segment(link, axis, frame))
        if names != resource['joint_names']:
            raise ValueError(f'Captured model joint chain does not match {resource["joint_names"]}')
        self.chains[key] = chain
        return chain

    def call(self, kind, endpoint: str, request):
        """Call an allowed read-only service with a bounded wall timeout."""
        import rclpy

        if endpoint not in {'/compute_fk', '/compute_ik', '/move_group/get_parameters'}:
            raise ValueError('Waypoint preparation only permits read-only kinematics services')
        if endpoint not in self.clients:
            self.clients[endpoint] = self.node.create_client(kind, self.namespace + endpoint)
        client = self.clients[endpoint]
        if not client.wait_for_service(timeout_sec=5.):
            raise RuntimeError(f'{endpoint} unavailable')
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self.node, future, timeout_sec=5.)
        if not future.done():
            future.cancel()
            raise TimeoutError(endpoint)
        return future.result()

    def state(self, names: list[str], joints: list[float]):
        """Describe the proposed joint positions without changing the robot."""
        from moveit_msgs.msg import RobotState

        state = RobotState(is_diff=True)
        state.joint_state.name, state.joint_state.position = names, list(map(float, joints))
        return state

    def fk(self, resource: dict, joints: list[float]) -> list[float]:
        """Read the controlled-link pose for an explicit joint configuration."""
        from moveit_msgs.srv import GetPositionFK

        if self.model_parameters is not None:
            import PyKDL as kdl
            values, frame = kdl.JntArray(len(joints)), kdl.Frame()
            for index, value in enumerate(joints):
                values[index] = value
            if kdl.ChainFkSolverPos_recursive(self.chain(resource)).JntToCart(values, frame) < 0:
                raise ValueError('Offline waypoint FK failed')
            return [frame.p.x(), frame.p.y(), frame.p.z(), *frame.M.GetQuaternion()]

        request = GetPositionFK.Request(robot_state=self.state(resource['joint_names'], joints),
                                        fk_link_names=[resource['ee_link']])
        request.header.frame_id = resource['ros_frame']
        result = self.call(GetPositionFK, '/compute_fk', request)
        if result.error_code.val != 1 or len(result.pose_stamped) != 1:
            raise ValueError(f'Waypoint FK failed: {result.error_code.val}')
        return _values(result.pose_stamped[0].pose)

    def solve(self, resource: dict, target: list[float], seed: list[float], preferred: list[float] | None = None) -> list[float]:
        """Resolve one saved Cartesian sample; production never calls this method."""
        from builtin_interfaces.msg import Duration
        from moveit_msgs.srv import GetPositionIK

        if self.model_parameters is not None:
            import numpy as np
            import PyKDL as kdl
            from scipy.optimize import least_squares

            chain = self.chain(resource)
            fk = kdl.ChainFkSolverPos_recursive(chain)
            values, frame = kdl.JntArray(len(seed)), kdl.Frame()
            desired = kdl.Frame(kdl.Rotation.Quaternion(*target[3:]), kdl.Vector(*target[:3]))
            def residual(joints):
                for index, value in enumerate(joints):
                    values[index] = value
                fk.JntToCart(values, frame)
                error = kdl.diff(frame, desired)
                reference = seed if preferred is None else preferred
                weight = 1e-4
                return np.array([error[i] for i in range(6)] + [weight * (a-b) for a,b in zip(joints, reference)])
            low = np.array([resource['limits'][name]['lower'] for name in resource['joint_names']])
            high = np.array([resource['limits'][name]['upper'] for name in resource['joint_names']])
            result = least_squares(residual, np.clip(seed, low + 1e-8, high - 1e-8),
                                   bounds=(low, high), ftol=1e-11, xtol=1e-11, gtol=1e-11, max_nfev=150)
            if np.linalg.norm(residual(result.x)[:3]) > 1e-5 or np.linalg.norm(residual(result.x)[3:6]) > 1e-5:
                raise ValueError(f'Offline waypoint IK failed at {target}')
            return continuous_joints(result.x.tolist(), seed, resource['joint_names'], resource['limits'], .35)

        query = GetPositionIK.Request()
        request = query.ik_request
        request.group_name, request.ik_link_name = resource['group_name'], resource['ee_link']
        request.pose_stamped.header.frame_id = resource['ros_frame']
        request.pose_stamped.pose = _pose(target)
        request.robot_state = self.state(resource['joint_names'], seed)
        # The preparation has no authority to change current payloads or fixtures.
        # Every stored motion undergoes current-scene collision checks at dispatch.
        request.avoid_collisions = False
        request.timeout = Duration(nanosec=200_000_000)
        result = self.call(GetPositionIK, '/compute_ik', query)
        if result.error_code.val != 1:
            raise ValueError(f'Waypoint IK failed ({result.error_code.val}) at {target}')
        values = dict(zip(result.solution.joint_state.name, result.solution.joint_state.position))
        try:
            return continuous_joints([values[name] for name in resource['joint_names']], seed,
                                     resource['joint_names'], resource['limits'], .35)
        except ValueError as exc:
            raise ValueError(f'{exc}; pose={target}; seed={seed}') from exc

    def motion(self, resource_id: str, identifier: str, joints: list[float],
               targets: list[list[float]], *, end_joints: list[float] | None = None) -> list[float]:
        """Save one complete motion, including joint positions and fixed timing."""
        resource = self.resources[resource_id]
        start = self.fk(resource, joints)
        key = (resource_id, tuple(joints), tuple(tuple(p) for p in targets), tuple(end_joints or []))
        if key in self.motions:
            recorded = deepcopy(self.motions[key])
            recorded['id'] = identifier
            resource['routes'].append(recorded)
            return list(recorded['points'][-1]['positions'])
        segment_starts = [start, *targets[:-1]]
        count = sum(len(sample_segment(left, right, linear_step=.005, angular_step=.025)) - 1
                    for left, right in zip(segment_starts, targets))
        index = 0
        def solve(pose, seed):
            nonlocal index
            index += 1
            preferred = ([a + (b-a) * index/count for a,b in zip(joints, end_joints)]
                         if end_joints is not None else None)
            return self.solve(resource, pose, seed, preferred)
        rows = resolve_waypoints(
            start_pose=start, start_joints=joints, waypoints=targets,
            names=resource['joint_names'], limits=resource['limits'],
            solve_ik=solve,
            linear_step=.005, angular_step=.025, maximum_joint_step=.35,
        )
        if end_joints is not None:
            if max(abs(a - b) for a, b in zip(rows[-1]['positions'], end_joints)) > .1:
                raise ValueError('Saved return does not reach the configured home joint branch')
            positions = [row['positions'] for row in rows]
            positions[-1] = list(end_joints)
            rows = _segment_timing(positions, resource['joint_names'], resource['limits'])
        validate_points(rows, resource['joint_names'], resource['limits'])
        resource['routes'].append({'id': identifier, 'start_pose': start,
                                   'target_pose': targets[-1], 'waypoints': targets, 'points': rows})
        self.motions[key] = deepcopy(resource['routes'][-1])
        logger.info('%s: saved %s (%d points)', resource_id, identifier, len(rows))
        return list(rows[-1]['positions'])

    def reverse_motion(self, resource_id: str, identifier: str) -> list[float]:
        """Save the exact reverse of the preceding motion, including fixed timing."""
        resource = self.resources[resource_id]
        original = resource['routes'][-1]
        duration = original['points'][-1]['time_from_start']
        points = [{**deepcopy(row), 'velocities': [-v for v in row['velocities']],
                   'time_from_start': duration-row['time_from_start']}
                  for row in reversed(original['points'])]
        validate_points(points, resource['joint_names'], resource['limits'])
        resource['routes'].append(dict(id=identifier, start_pose=original['target_pose'],
            target_pose=original['start_pose'], waypoints=[original['start_pose']], points=points))
        return list(points[-1]['positions'])

    def ur_limits(self, names: list[str]) -> dict:
        """Read the same active limits that the UR execution controller checks."""
        from rcl_interfaces.srv import GetParameters

        parameters = [f'robot_description_planning.joint_limits.{name}.{field}'
                      for name in names for field in ('max_velocity', 'max_acceleration')]
        if self.model_parameters is not None:
            from types import SimpleNamespace
            values = [SimpleNamespace(type=3 if name in self.model_parameters else 0,
                                      double_value=self.model_parameters.get(name, 0.)) for name in parameters]
        else:
            values = self.call(GetParameters, '/move_group/get_parameters',
                               GetParameters.Request(names=parameters)).values
        result = {}
        for index, name in enumerate(names):
            element = self.urdf.find(f"joint[@name='{name}']/limit")
            velocity, acceleration = values[2 * index:2 * index + 2]
            if element is None or acceleration.type == 0 or acceleration.double_value <= 0:
                raise ValueError(f'Missing running joint limits: {name}')
            result[name] = dict(lower=float(element.get('lower')), upper=float(element.get('upper')),
                                velocity=min(float(element.get('velocity')), velocity.double_value)
                                if velocity.type else float(element.get('velocity')),
                                acceleration=acceleration.double_value)
        return result

    def ur(self, robot: dict, context) -> None:
        """Prepare each configured UR handling cycle and its home return."""
        from cais_spade_llm.recovery_framework.workflow_execution import (
            _robot_configuration, _pick_geometry, _destination_geometry,
        )
        from cais_spade_llm.resources.robot.gazebo_pick_place_controller import GazeboPickPlaceController
        from cais_spade_llm.resources.robot.robot_task_runtime import _normalize_pick_targets
        from cais_spade_llm.resources.robot.target_calculations import placement_poses

        scene = self.inputs['scene']
        rid = robot['resource_id']
        cfg, named, _ = _robot_configuration(scene, robot)
        resource = self.resources[rid] = dict(joint_names=cfg['arm_joint_names'],
            limits=self.ur_limits(cfg['arm_joint_names']), frame_id='world', ros_frame='world',
            group_name=cfg['move_group']['group_name'], ee_link=cfg['move_group']['ee_link'], routes=[])
        home = named['home']
        home_pose = self.fk(resource, home)
        controller = GazeboPickPlaceController(robot_name='ur5e', node_name='saved_waypoints',
            controller_config=cfg, named_positions=named, arm_joint_names=cfg['arm_joint_names'])
        controller.init = lambda: True
        controller.wait_for_services = lambda: True
        from geometry_msgs.msg import Pose, Quaternion
        controller._Pose, controller._Quaternion = Pose, Quaternion
        controller._get_ee_pose = lambda: _pose(home_pose)
        controller._get_ee_tcp_world_z_offset = lambda: -.218
        controller._log = lambda: logger
        machines = {row['handling_robot']: row for row in scene['machines']}
        if rid in machines:
            machine = machines[rid]
            parts, origin, destination = machine['nominal_parts'], machine['resource_id'], 'Conveyor'
            parts = list(parts)
            for order_path in sorted((ROOT / 'cais_spade_llm/specification/products/orders').glob('assembly_board-v1-*.json')):
                order = read_json(order_path)
                if order.get('machine_resource') == origin:
                    parts.extend(part for part in order['parts'] if part not in parts)
        elif rid == 'ur5e-3':
            parts, origin, destination = list(scene['Storage']['slots']), 'Buffer For Machined parts', context.product_name
        else:
            parts, origin, destination = scene['3D Printing Station']['initial_products'], '3D Printing Station', context.product_name
        for part in parts:
            first = len(resource['routes'])
            try:
                if rid in machines:
                    source = machines[rid]['workholding_pose']
                elif rid == 'ur5e-3':
                    # Transport preserves the Conveyor release pose while changing X.
                    machine = next(m for m in scene['machines'] if part in m['nominal_parts'])
                    source = [scene[origin]['pickup_pose'][0], *machine['conveyor_loading_pose'][1:]]
                else:
                    source = scene[origin]['output_poses'][part]
                source_pose = [*source[:3], *quaternion(source[3:])]
                pick = controller.compute_pick_targets(part_name=part,
                    product_geometry=_pick_geometry(context, part, origin, rid),
                    detected_parts=[dict(part_name=part, model_name=part,
                        **dict(zip(('x','y','z','qx','qy','qz','qw'), source_pose)))])
                if not pick.get('success'):
                    raise ValueError(pick.get('message'))
                pick = _normalize_pick_targets(pick)
                def pose_values(row):
                    return [row[key] for key in ('x', 'y', 'z')] + [row.get(key, home_pose[i + 3])
                            for i, key in enumerate(('qx', 'qy', 'qz', 'qw'))]
                approach, grasp = pose_values(pick['approach_pose']), pose_values(pick['target_pose'])
                pick_route = [[*xyz, *approach[3:]] for xyz in
                              robot['cartesian_motion'].get('pick_transit_waypoints', [])]
                joints = self.motion(rid, f'{part}/pick_approach/move_above_part', home, [*pick_route, approach])
                joints = self.motion(rid, f'{part}/pick_approach/descend', joints, [grasp])
                lift = pose_values(pick['access_retreat_pose']) if pick.get('access_retreat_pose') else [*grasp[:2], pick['travel_z'], *grasp[3:]]
                joints = self.motion(rid, f'{part}/pick_grasp/lift', joints, [lift])
                pick['held_part_handoff'] = {'world_tool0_pose_at_grasp': dict(zip(('x','y','z','qx','qy','qz','qw'), grasp))}
                geometry = _destination_geometry(context, part, destination, rid)
                contact = geometry.get('simulation_mating_contact')
                if contact:
                    relative = compose(inverse(grasp), source_pose)
                    target = geometry['target_origin_pose']
                    tool_pose = compose([*[target[a] for a in ('x','y','z')], *quaternion([0.,0.,contact['target_yaw_rad']])], inverse(relative))
                    places = placement_poses(*tool_pose[:3], dict(zip(('qx','qy','qz','qw'),tool_pose[3:])),
                        simulation_assembly_slot=True, insertion_depth=controller.insertion_depth_m)
                else:
                    places = controller.compute_place_targets(pick_ctx=pick, product_geometry=geometry,
                        part_name=part, destination_location=destination)
                    if not places.get('success'):
                        raise ValueError(places.get('message'))
                above, down, seated = [pose_values(places[key]) for key in ('approach_pose','target_pose','insert_pose')]
                route = []
                if robot['cartesian_motion']['transit_waypoints']:
                    xyz = list(robot['cartesian_motion']['transit_waypoints'][0])
                    if math.dist(lift[:3], xyz[-1]) < math.dist(lift[:3], xyz[0]):
                        xyz.reverse()
                    route = [[*point, *above[3:]] for point in xyz]
                joints = self.motion(rid, f'{part}/place_approach/move_above_destination', joints, [*route, above])
                joints = self.motion(rid, f'{part}/place_approach/descend', joints, [down])
                if math.dist(down, seated) > 1e-7:
                    joints = self.motion(rid, f'{part}/place_insert/insert', joints, [seated])
                after = [*seated[:2], pick['travel_z'], *seated[3:]]
                joints = self.motion(rid, f'{part}/place_insert/lift', joints, [after])
                self.motion(rid, f'{part}/move_home', joints, [home_pose], end_joints=home)
            except (ValueError, RuntimeError, TimeoutError) as exc:
                del resource['routes'][first:]
                self.failures.append({'resource_id': rid, 'part_name': part, 'reason': str(exc)})
                logger.error('%s %s: %s', rid, part, exc)
        controller._planning_executor.shutdown(wait=False)

    def kmr(self) -> None:
        """Prepare KMR pickup, placement, Storage return and pickup-dock moves."""
        from cais_spade_llm.recovery_framework.kmr_motion import joint_limits, downward_transfer_waypoints

        scene = self.inputs['scene']
        kmr, storage = scene['KMR'], scene['Storage']
        cfg = kmr['task_execution']
        settings = cfg['cartesian_waypoints']
        resource = self.resources['KMR'] = dict(joint_names=kmr['arm_joint_names'],
            limits=joint_limits(ROOT/'ros2/cais_lab_robotics/urdf/KMR_recovery.urdf.xacro',
                                kmr['arm_joint_names'], cfg['joint_acceleration_limits']),
            frame_id='KMR', ros_frame='KMR_base_link', group_name=cfg['planning_group'],
            ee_link=cfg['tcp_link'], routes=[])
        dock = [*storage['KMR_docking_pose'][:3], *quaternion(storage['KMR_docking_pose'][3:])]
        home = storage['KMR_pick_arm_configurations']['KET4_Square_4mm']
        home_pose = self.fk(resource, home)
        homes = {part: storage['KMR_pick_arm_configurations'][part]
                 for part, pose in storage['KMR_pick_docking_poses'].items()
                 if math.dist(pose[:2], dock[:2]) < .001
                 and abs(math.atan2(math.sin(pose[2] - storage['KMR_docking_pose'][5]),
                                    math.cos(pose[2] - storage['KMR_docking_pose'][5]))) < .001}
        downward = compose(inverse(dock), [*dock[:3], *cfg['pick_orientation_xyzw']])[3:]
        transport = [*settings['transport_tcp_position_in_base_frame_m'], *downward]
        mount = [*kmr['arm_mount_xyz'], *quaternion(kmr['arm_mount_rpy'])]
        def transfer(joints, target):
            start = self.fk(resource, joints)
            waypoints = downward_transfer_waypoints(start, target, mount, settings)
            if waypoints and abs(sum(a*b for a,b in zip(start[3:], target[3:]))) < .01:
                middle = settings['transfer_orientation_waypoint_xyzw_in_base_frame']
                half = len(waypoints)//2
                first = sample_segment(start, [*start[:3], *middle], linear_step=1e6,
                                       angular_step=1e6, minimum_samples=half)
                second = sample_segment([*target[:3], *middle], target, linear_step=1e6,
                                        angular_step=1e6, minimum_samples=len(waypoints)-half-1)
                for waypoint, orientation in zip(waypoints, [*first, *second[1:]], strict=True):
                    waypoint[3:] = orientation[3:]
            return [*waypoints, target]
        parked = self.motion('KMR', 'Storage/transport', home, transfer(home, transport))
        self.motion('KMR', 'Storage/move_home', parked, transfer(parked, home_pose), end_joints=home)
        for part, other_home in homes.items():
            if other_home == home:
                continue
            self.motion('KMR', f'Storage/{part}/transport', other_home,
                        transfer(other_home, transport), end_joints=parked)
            self.motion('KMR', f'Storage/{part}/move_home', parked,
                        transfer(parked, self.fk(resource, other_home)), end_joints=other_home)
        for part, slot in storage['slots'].items():
            for machine in scene['machines']:
                first = len(resource['routes'])
                try:
                    pick_dock = storage['KMR_pick_docking_poses'][part]
                    base = [pick_dock[0], pick_dock[1], 0., *quaternion([0.,0.,pick_dock[2]])]
                    target = compose(inverse(base), [*slot[:2], slot[2] + cfg['grasp_height_m'], *cfg['pick_orientation_xyzw']])
                    approach = [*target[:2], target[2] + storage['KMR_pick_approach_clearance_m'][part], *target[3:]]
                    key = f'{part}/{machine["resource_id"]}'
                    starts = [*homes.values(), parked] if math.dist(pick_dock[:2], dock[:2]) < .001 else [parked]
                    for number, start in enumerate(starts):
                        pickup_joints = storage['KMR_pick_arm_configurations'][part]
                        joints = self.motion('KMR', f'{key}/pick_approach/{number}', start, [approach], end_joints=pickup_joints)
                        joints = self.motion('KMR', f'{key}/pick_descend/{number}', joints, [target])
                        lift = [*target[:2], target[2] + cfg['minimum_pick_lift_m'], *target[3:]]
                        if math.dist(lift, approach) <= 1e-8:
                            joints = self.reverse_motion('KMR', f'{key}/pick_lift/{number}')
                        else:
                            joints = self.motion('KMR', f'{key}/pick_lift/{number}', joints, [lift])
                        machine_dock = machine['KMR_docking_pose']
                        base = [*machine_dock[:3], *quaternion(machine_dock[3:])]
                        fixture = machine['workholding_pose']
                        grasp_world = [*slot[:2], slot[2] + cfg['grasp_height_m'], *cfg['pick_orientation_xyzw']]
                        grasp_transform = compose(inverse(grasp_world), [*slot[:3], *quaternion(slot[3:])])
                        place_world = compose([*fixture[:3], *quaternion(fixture[3:])], inverse(grasp_transform))
                        place_world[2] += .003
                        place = compose(inverse(base), place_world)
                        above = [*place[:2], place[2] + cfg['minimum_pick_lift_m'], *place[3:]]
                        joints = self.motion('KMR', f'{key}/place_approach/{number}', joints, transfer(joints, above))
                        joints = self.motion('KMR', f'{key}/place_descend/{number}', joints, [place])
                        retreat = [*place[:2], place[2] + cfg['release_offsets_m'][0][2], *place[3:]]
                        joints = self.motion('KMR', f'{key}/withdraw/{number}', joints, [retreat])
                        joints = self.motion('KMR', f'{key}/transport/{number}', joints, transfer(joints, transport))
                        self.motion('KMR', f'{key}/move_home/{number}', joints, transfer(joints, home_pose), end_joints=home)
                        for home_part, other_home in homes.items():
                            if other_home != home:
                                self.motion('KMR', f'{key}/move_home/{number}/{home_part}', joints,
                                            transfer(joints, self.fk(resource, other_home)), end_joints=other_home)
                except (ValueError, RuntimeError, TimeoutError) as exc:
                    del resource['routes'][first:]
                    self.failures.append({'resource_id':'KMR','part_name':part,
                                          'destination':machine['resource_id'],'reason':str(exc)})
                    logger.error('KMR %s %s: %s', part, machine['resource_id'], exc)


def main() -> None:
    """Prepare a saved file using read-only services or captured robot parameters."""
    import rclpy
    from cais_spade_llm.ui.recovery_setup import validate_setup
    from cais_spade_llm.product.environment import EnvironmentProductContext

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/SAVED_WAYPOINTS_FILE)
    parser.add_argument('--resource', action='append', choices=['KMR', 'ur5e-1', 'ur5e-2', 'ur5e-3', 'ur5e-4'])
    parser.add_argument('--namespace', default='')
    parser.add_argument('--model-parameters', type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    setup = read_json(ROOT/'cais_spade_llm/initialization/recovery_framework_setup.json')
    inputs = validate_setup(setup)
    context = EnvironmentProductContext(inputs['scene'], inputs['product_order'], inputs['geometry'])
    rclpy.init()
    node = rclpy.create_node('prepare_saved_waypoints_read_only')
    started = time.time()
    try:
        model_parameters = None
        if args.model_parameters:
            import yaml
            model_parameters = yaml.safe_load(args.model_parameters.read_text())['/**']['ros__parameters']
        preparation = WaypointPreparation(node, inputs, args.namespace, model_parameters)
        for robot in inputs['scene']['robots']:
            if not args.resource or robot['resource_id'] in args.resource:
                preparation.ur(robot, context)
        try:
            if not args.resource or 'KMR' in args.resource:
                preparation.kmr()
        except (ValueError, RuntimeError, TimeoutError) as exc:
            preparation.failures.append({'resource_id':'KMR','reason':str(exc)})
        result = dict(version=1, execution_mode='simulation', source_fingerprints=source_fingerprints(),
                      robot_description_sha256=hashlib.sha256(preparation.description.encode()).hexdigest(),
                      prepared_at_unix=time.time(), preparation_wall_time_sec=time.time()-started,
                      validation='kinematics and configured limits; current-scene collision validation required at execution',
                      resources=preparation.resources, failures=preparation.failures)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix('.tmp')
        temporary.write_text(json.dumps(result, separators=(',', ':'))+'\n', encoding='utf-8')
        temporary.replace(args.output)
        logger.info('Saved %d motions for %d robots; %d incomplete cycles',
                    sum(len(r['routes']) for r in preparation.resources.values()),
                    len(preparation.resources), len(preparation.failures))
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
