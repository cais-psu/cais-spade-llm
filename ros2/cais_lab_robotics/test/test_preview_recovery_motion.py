from __future__ import annotations

"""Isolated synthetic-model tests; no Gazebo or controller processes are started.

Run with PREVIEW_RECOVERY_MOTION_TEST_SERVER pointing at the built executable.
The test owns a distinct ROS domain, server, configuration, and full scene.
"""

import asyncio
import copy
import json
import math
import os
import secrets
from pathlib import Path
import signal
import sys
import subprocess
import tempfile
import time
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from cais_spade_llm.recovery_framework.recovery_motion_preview import _scene_message_fingerprint


def _model_parameters():
    """Return the same declared model for action and saved-file checks."""
    urdf = '''<robot name="preview_test">
      <link name="world"/>
      <link name="tool"><collision><geometry><sphere radius="0.02"/></geometry></collision></link>
      <link name="peer_tool"><collision><geometry><sphere radius="0.02"/></geometry></collision></link>
      <link name="peer_finger"><collision><geometry><sphere radius="0.01"/></geometry></collision></link>
      <joint name="slider" type="prismatic"><parent link="world"/><child link="tool"/><axis xyz="1 0 0"/><limit lower="-2" upper="2" velocity="1" effort="10"/></joint>
      <joint name="peer" type="prismatic"><parent link="world"/><child link="peer_tool"/><axis xyz="1 0 0"/><limit lower="-2" upper="2" velocity="1" effort="10"/></joint>
      <joint name="peer_mimic" type="prismatic"><parent link="world"/><child link="peer_finger"/><origin xyz="0 1 0"/><axis xyz="1 0 0"/><limit lower="-2" upper="2" velocity="1" effort="10"/><mimic joint="peer" multiplier="1" offset="0"/></joint>
    </robot>'''
    srdf = '<robot name="preview_test"><group name="arm"><joint name="slider"/></group></robot>'
    params = {
        "robot_description": urdf, "robot_description_semantic": srdf,
        "robot_description_planning": {"joint_limits": {name: {
            "has_velocity_limits": True, "max_velocity": 1.0,
            "has_acceleration_limits": True, "max_acceleration": 1.0,
        } for name in ("slider", "peer")}},
    }
    return params


def _support_departure_goal(goal, base=(0., 0., 0., 0., 0., 0., 1.)):
    """Bind a measured box support contact to the synthetic private scene."""
    from geometry_msgs.msg import Pose
    from moveit_msgs.msg import AttachedCollisionObject, CollisionObject
    from shape_msgs.msg import SolidPrimitive
    from rosidl_runtime_py.convert import message_to_ordereddict
    from cais_spade_llm.recovery_framework.geometry import compose, rotate
    from cais_spade_llm.recovery_framework.recovery_motion_preview import bind_support_departure

    part = CollisionObject(id="part/link/collision")
    part.header.frame_id = "tool"
    part.primitives = [SolidPrimitive(type=SolidPrimitive.BOX, dimensions=[.0625] * 3)]
    pose = Pose()
    pose.position.x, pose.position.y, pose.orientation.w = .0625, .25, 1.
    part.primitive_poses = [pose]
    goal.planning_scene.robot_state.attached_collision_objects = [
        AttachedCollisionObject(link_name="tool", touch_links=["tool"], object=part)]
    support = CollisionObject(id="support/link/collision")
    support.header.frame_id = "world"
    support.primitives = [SolidPrimitive(type=SolidPrimitive.BOX, dimensions=[.0625, .125, .125])]
    actual = compose(base, [.015625, .25, 0., 0., 0., 0., 1.])
    pose = Pose()
    pose.position.x, pose.position.y, pose.position.z = actual[:3]
    pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = actual[3:]
    support.primitive_poses = [pose]
    goal.planning_scene.world.collision_objects = [support]
    goal.start_state = copy.deepcopy(goal.planning_scene.robot_state)
    goal.scene_fingerprint = _scene_message_fingerprint(goal.planning_scene)
    evidence = {"version": 1, "status": "prepared", "resource_id": goal.resource_id,
        "command_sent": False, "execution_authorized": False, "pairs": [{
            "part_object_id": part.id, "support_object_id": support.id,
            "part_collision_id": "part::link::collision", "support_collision_id": "support::link::collision",
            "source_contact_fingerprint": "recorded-contact", "geometry_fingerprint": "recorded-boxes",
            "plane_point": compose(base, [.046875, .25, 0., 0., 0., 0., 1.])[:3],
            "plane_normal": rotate(base[3:], [1., 0., 0.]),
            "initial_penetration_bound": .015625 + 1e-12}]}
    bound = bind_support_departure(message_to_ordereddict(goal), evidence)
    goal.support_departure_json = bound["support_departure_json"]
    goal.support_departure_fingerprint = bound["support_departure_fingerprint"]
    return goal


def _assert_rejected_sample(test, goal, response, phase):
    """Check the real native failure through the same recorded client boundary."""
    from rosidl_runtime_py.convert import message_to_ordereddict
    from cais_spade_llm.recovery_framework.recovery_motion_preview import preview_result_record

    record = preview_result_record(message_to_ordereddict(goal), message_to_ordereddict(response))
    evidence = record["collision_rejection_evidence"]
    test.assertEqual(evidence["phase"], phase)
    test.assertTrue(evidence["contacts_complete"])
    test.assertEqual(set(evidence["joint_state"]["name"]), set(goal.start_state.joint_state.name))
    test.assertFalse(record["execution_authorized"])
    if goal.support_departure_json:
        audit = json.loads(response.support_departure_evidence_json)
        test.assertEqual(audit["rejected_sample"], evidence)
    return evidence


class PreviewRecoveryMotionTests(unittest.TestCase):
    """Exercise the real action boundary using one moving and one stationary link."""

    @classmethod
    def setUpClass(cls):
        executable = os.environ.get("PREVIEW_RECOVERY_MOTION_TEST_SERVER")
        if not executable:
            raise unittest.SkipTest("PREVIEW_RECOVERY_MOTION_TEST_SERVER is not set")
        # Keep RTPS ports below Linux's ephemeral port range while remaining
        # separate from the simulator's domain.
        os.environ["ROS_DOMAIN_ID"] = str(60 + secrets.randbelow(40))
        os.environ["ROS_LOCALHOST_ONLY"] = "0"
        import rclpy
        import yaml
        from cais_lab_robotics.action import PreviewRecoveryMotion
        from rclpy.action import ActionClient

        cls.temp = tempfile.TemporaryDirectory(prefix="recovery-preview-test-")
        cls.addClassCleanup(cls.temp.cleanup)
        os.environ["ROS_LOG_DIR"] = str(Path(cls.temp.name) / "logs")
        params = _model_parameters()
        configuration = Path(cls.temp.name) / "parameters.yaml"
        configuration.write_text(yaml.safe_dump({"preview_recovery_motion": {"ros__parameters": params}}))
        cls.params = configuration
        cls.log = (Path(cls.temp.name) / "server.log").open("w+")
        cls.addClassCleanup(cls.log.close)
        cls.process = subprocess.Popen([executable, "--ros-args", "--params-file", str(configuration)],
                                       stdout=cls.log, stderr=subprocess.STDOUT)
        cls.addClassCleanup(cls.stop_server)
        rclpy.init()
        cls.addClassCleanup(rclpy.shutdown)
        cls.node = rclpy.create_node("preview_isolation_test")
        cls.addClassCleanup(cls.node.destroy_node)
        cls.client = ActionClient(cls.node, PreviewRecoveryMotion, "/preview_recovery_motion")
        cls.addClassCleanup(cls.client.destroy)
        if not cls.client.wait_for_server(timeout_sec=15.0):
            cls.log.seek(0)
            detail = cls.log.read()
            raise RuntimeError("Preview server did not start: " + detail)
        cls.model_fingerprint = ""
        # A fresh client avoids retaining a failed startup service exchange.
        # Read the server's actual value; no local fingerprint substitutes for it.
        for attempt in range(3):
            try:
                response = subprocess.run(
                    ["ros2", "param", "get", "--no-daemon", "--spin-time", "5",
                     "/preview_recovery_motion", "model_fingerprint"],
                    capture_output=True, text=True, timeout=15.0, check=True,
                )
            except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
                if attempt == 2:
                    raise
                if cls.process.poll() is not None:
                    cls.log.seek(0)
                    raise RuntimeError("Preview server exited during discovery: " + cls.log.read())
                continue
            prefix = "String value is: "
            assert response.stdout.strip().startswith(prefix), response.stdout
            cls.model_fingerprint = response.stdout.strip()[len(prefix):]
            break
        assert len(cls.model_fingerprint) == 64

    @classmethod
    def stop_server(cls):
        if cls.process.poll() is None:
            cls.process.send_signal(signal.SIGINT)
            try:
                cls.process.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                cls.process.kill()
                cls.process.wait(timeout=5.0)

    @classmethod
    def wait(cls, future, timeout=15.0):
        import rclpy

        deadline = time.monotonic() + timeout
        while not future.done():
            if time.monotonic() > deadline:
                cls.log.seek(0)
                raise TimeoutError("Isolated preview test timed out: " + cls.log.read())
            rclpy.spin_once(cls.node, timeout_sec=0.01)
        return future.result()

    def goal(self):
        from cais_lab_robotics.action import PreviewRecoveryMotion
        from rclpy.serialization import serialize_message

        goal = PreviewRecoveryMotion.Goal()
        goal.planning_scene.robot_model_name = "preview_test"
        goal.planning_scene.robot_state.joint_state.name = ["slider", "peer", "peer_mimic"]
        goal.planning_scene.robot_state.joint_state.position = [0.0, 1.5, 1.49]
        goal.start_state = copy.deepcopy(goal.planning_scene.robot_state)
        goal.group_name, goal.link_name, goal.frame_id = "arm", "tool", "world"
        goal.mode = "joint_target"
        goal.joint_target.name, goal.joint_target.position = ["slider"], [0.6]
        goal.max_translation_step = goal.max_rotation_step = goal.max_joint_step = 0.01
        goal.velocity_scaling = goal.acceleration_scaling = 0.1
        goal.timeout_seconds = 10.0
        goal.resource_id, goal.capture_fingerprint, goal.proposal_fingerprint = "resource", "capture", "proposal"
        goal.scene_fingerprint = _scene_message_fingerprint(goal.planning_scene)
        goal.model_fingerprint = self.model_fingerprint
        return goal

    def run_goal(self, goal):
        handle = self.wait(self.client.send_goal_async(goal))
        self.assertTrue(handle.accepted)
        return self.wait(handle.get_result_async()).result

    def test_complete_joint_path_retains_peer_and_input_scene(self):
        from rclpy.serialization import serialize_message

        goal = self.goal()
        original = serialize_message(goal.planning_scene)
        response = self.run_goal(goal)
        self.assertEqual(response.status, "PREVIEW_COMPLETE", response.reason)
        self.assertTrue(response.path_complete)
        self.assertFalse(response.continuous_collision_verified)
        observed = dict(zip(response.end_state.joint_state.name, response.end_state.joint_state.position, strict=True))
        self.assertEqual(observed["peer"], 1.5)
        self.assertEqual(observed["peer_mimic"], 1.49)
        self.assertAlmostEqual(response.trajectory.joint_trajectory.points[-1].positions[0], 0.6)
        self.assertEqual(serialize_message(goal.planning_scene), original)

    def test_collision_and_missing_start_are_unresolved(self):
        from geometry_msgs.msg import Pose
        from moveit_msgs.msg import CollisionObject
        from rclpy.serialization import serialize_message
        from shape_msgs.msg import SolidPrimitive

        goal = self.goal()
        obstacle = CollisionObject(id="obstacle")
        obstacle.header.frame_id = "world"
        obstacle.primitives = [SolidPrimitive(type=SolidPrimitive.BOX, dimensions=[0.1, 0.1, 0.1])]
        pose = Pose()
        pose.position.x, pose.orientation.w = 0.35, 1.0
        obstacle.primitive_poses = [pose]
        goal.planning_scene.world.collision_objects = [obstacle]
        goal.scene_fingerprint = _scene_message_fingerprint(goal.planning_scene)
        response = self.run_goal(goal)
        self.assertEqual(response.status, "NEEDS_CONTEXT")
        self.assertIn("collision", response.reason)
        goal = self.goal()
        goal.planning_scene.robot_state.joint_state.name = ["slider"]
        goal.planning_scene.robot_state.joint_state.position = [0.0]
        goal.start_state = copy.deepcopy(goal.planning_scene.robot_state)
        goal.scene_fingerprint = _scene_message_fingerprint(goal.planning_scene)
        response = self.run_goal(goal)
        self.assertIn("Missing joint variable: peer", response.reason)
        self.assertFalse(response.path_complete)

    def test_model_and_scene_fingerprints_reject_stale_evidence(self):
        for field in ("model_fingerprint", "scene_fingerprint"):
            goal = self.goal()
            setattr(goal, field, "changed")
            response = self.run_goal(goal)
            self.assertEqual(response.status, "NEEDS_CONTEXT")
            self.assertIn("fingerprint", response.reason)

    def test_cancel_settles_before_next_request(self):
        goal = self.goal()
        goal.max_joint_step = 0.00001
        handle = self.wait(self.client.send_goal_async(goal))
        self.assertTrue(handle.accepted)
        self.wait(handle.cancel_goal_async())
        response = self.wait(handle.get_result_async()).result
        self.assertEqual(response.status, "CANCELED")
        self.assertFalse(response.path_complete)
        response = self.run_goal(self.goal())
        self.assertEqual(response.status, "PREVIEW_COMPLETE", response.reason)

    def test_lazy_client_round_trip(self):
        from rosidl_runtime_py.convert import message_to_ordereddict
        from cais_spade_llm.recovery_framework.recovery_motion_preview import preview_recovery_motion

        request = message_to_ordereddict(self.goal())
        response = asyncio.run(preview_recovery_motion(request, execution_mode="simulation", timeout_seconds=15.0))
        self.assertEqual(response["status"], "PREVIEW_COMPLETE", response["reason"])
        self.assertFalse(response["execution_authorized"])
        self.assertFalse(response["command_sent"])
        self.assertFalse(response["coverage"]["cca_validated"])
        self.assertEqual(response["request"], request)

    def test_captured_attachment_is_preserved_without_live_mutation(self):
        from geometry_msgs.msg import Pose
        from moveit_msgs.msg import AttachedCollisionObject, CollisionObject
        from shape_msgs.msg import SolidPrimitive

        goal = self.goal()
        obj = CollisionObject(id="captured-payload")
        obj.header.frame_id = "tool"
        obj.primitives = [SolidPrimitive(type=SolidPrimitive.BOX, dimensions=[0.01, 0.01, 0.01])]
        pose = Pose()
        pose.position.x, pose.orientation.w = 0.03, 1.0
        obj.primitive_poses = [pose]
        attached = AttachedCollisionObject(link_name="tool", touch_links=["tool"], object=obj)
        goal.planning_scene.robot_state.attached_collision_objects = [attached]
        goal.start_state = copy.deepcopy(goal.planning_scene.robot_state)
        goal.scene_fingerprint = _scene_message_fingerprint(goal.planning_scene)
        original = copy.deepcopy(goal)
        response = self.run_goal(goal)
        self.assertEqual(response.status, "PREVIEW_COMPLETE", response.reason)
        self.assertEqual(response.end_state.attached_collision_objects[0].object.id, "captured-payload")
        self.assertEqual(goal, original)

    def test_missing_cartesian_solver_cannot_produce_complete_path(self):
        from geometry_msgs.msg import Pose
        from sensor_msgs.msg import JointState

        goal = self.goal()
        goal.mode = "cartesian"
        goal.joint_target = JointState()
        pose = Pose()
        pose.position.x, pose.orientation.w = 0.5, 1.0
        goal.waypoints = [pose]
        response = self.run_goal(goal)
        self.assertEqual(response.status, "NEEDS_CONTEXT")
        self.assertFalse(response.path_complete)

        diagnostic = json.loads(response.cartesian_diagnostics_json)
        self.assertEqual(diagnostic["failure_kind"], "solver_unavailable")
        self.assertFalse(diagnostic["complete_inputs"])
        self.assertEqual(diagnostic["ik_queries"], [])

    def test_support_departure_preserves_scene_and_separates_sampled_evidence(self):
        goal = _support_departure_goal(self.goal())
        original = copy.deepcopy(goal)
        response = self.run_goal(goal)
        self.assertEqual(response.status, "PREVIEW_COMPLETE", response.reason)
        self.assertFalse(response.continuous_collision_verified)
        self.assertFalse(response.sampled_collision_rejected)
        evidence = json.loads(response.support_departure_evidence_json)
        self.assertTrue(evidence["pairs_complete"])
        self.assertTrue(evidence["pairs"][0]["endpoint_separated"])
        self.assertGreater(evidence["pairs"][0]["first_separation_index"], 0)
        self.assertFalse(evidence["continuous_collision_verified"])
        self.assertEqual(response.support_departure_fingerprint, goal.support_departure_fingerprint)
        self.assertEqual(goal, original)

    def test_support_departure_missing_record_deeper_contact_and_no_separation_fail(self):
        from cais_spade_llm.recovery_framework.recovery_motion_preview import bind_support_departure
        from rosidl_runtime_py.convert import message_to_ordereddict

        for defect in ("missing", "deeper", "no_separation", "wrong_plane"):
            with self.subTest(defect=defect):
                goal = _support_departure_goal(self.goal())
                if defect == "missing":
                    goal.support_departure_json = goal.support_departure_fingerprint = ""
                elif defect == "no_separation":
                    goal.joint_target.position = [0.]
                else:
                    record = json.loads(goal.support_departure_json)
                    if defect == "deeper":
                        record["pairs"][0]["initial_penetration_bound"] = .001
                    else:
                        record["pairs"][0]["plane_point"][0] += .01
                    request = bind_support_departure(message_to_ordereddict(goal), record)
                    goal.support_departure_json = request["support_departure_json"]
                    goal.support_departure_fingerprint = request["support_departure_fingerprint"]
                response = self.run_goal(goal)
                self.assertEqual(response.status, "NEEDS_CONTEXT", response.reason)
                self.assertFalse(response.path_complete)
                if defect in ("missing", "deeper"):
                    self.assertTrue(response.sampled_collision_rejected)
                    _assert_rejected_sample(self, goal, response, "initial")

    def test_support_departure_does_not_exempt_unrelated_contact(self):
        from geometry_msgs.msg import Pose
        from moveit_msgs.msg import CollisionObject
        from shape_msgs.msg import SolidPrimitive
        from rosidl_runtime_py.convert import message_to_ordereddict
        from cais_spade_llm.recovery_framework.recovery_motion_preview import bind_support_departure

        goal = _support_departure_goal(self.goal())
        blocker = CollisionObject(id="unrelated_obstacle")
        blocker.header.frame_id = "world"
        blocker.primitives = [SolidPrimitive(type=SolidPrimitive.BOX, dimensions=[.02] * 3)]
        pose = Pose()
        pose.position.x, pose.orientation.w = .3, 1.
        blocker.primitive_poses = [pose]
        goal.planning_scene.world.collision_objects.append(blocker)
        goal.scene_fingerprint = _scene_message_fingerprint(goal.planning_scene)
        bound = bind_support_departure(message_to_ordereddict(goal), json.loads(goal.support_departure_json))
        goal.support_departure_json, goal.support_departure_fingerprint = bound["support_departure_json"], bound["support_departure_fingerprint"]
        response = self.run_goal(goal)
        self.assertEqual(response.status, "NEEDS_CONTEXT")
        self.assertTrue(response.sampled_collision_rejected)
        self.assertIn("unrelated_obstacle", response.reason)
        evidence = _assert_rejected_sample(self, goal, response, "joint_target")
        self.assertGreater(evidence["state_index"], 0)
        self.assertTrue(any("unrelated_obstacle" in (row["body_1"], row["body_2"])
                            for row in evidence["contacts"]))

    def test_server_has_no_execution_or_scene_service_clients(self):
        clients = self.node.get_client_names_and_types_by_node("preview_recovery_motion", "/")
        self.assertEqual(clients, [])


class PreviewRecoveryMotionFileTests(unittest.TestCase):
    """Exercise the same native validator through exact serialized Goal files."""

    model_parameters = staticmethod(_model_parameters)

    @classmethod
    def setUpClass(cls):
        cls.executable = os.environ.get("PREVIEW_RECOVERY_MOTION_TEST_SERVER")
        if not cls.executable:
            raise unittest.SkipTest("PREVIEW_RECOVERY_MOTION_TEST_SERVER is not set")
        import yaml
        from cais_lab_robotics.action import PreviewRecoveryMotion
        from rclpy.serialization import deserialize_message, serialize_message

        cls.temp = tempfile.TemporaryDirectory(prefix="recovery-preview-file-test-")
        cls.addClassCleanup(cls.temp.cleanup)
        cls.directory = Path(cls.temp.name)
        cls.params = cls.directory / "parameters.yaml"
        cls.params.write_text(yaml.safe_dump({"preview_recovery_motion": {
            "ros__parameters": cls.model_parameters(),
        }}))
        cls.environment = {**os.environ, "ROS_DOMAIN_ID": str(60 + secrets.randbelow(40)),
                           "ROS_LOCALHOST_ONLY": "1", "ROS_LOG_DIR": str(cls.directory / "logs")}
        goal = PreviewRecoveryMotion.Goal()
        goal.timeout_seconds = 10.0
        goal.resource_id, goal.capture_fingerprint, goal.proposal_fingerprint = "resource", "capture", "proposal"
        goal.model_fingerprint = "read-native-model-fingerprint"
        source = cls.directory / "fingerprint-goal.cdr"
        destination = cls.directory / "fingerprint-result.cdr"
        source.write_bytes(serialize_message(goal))
        completed = subprocess.run([cls.executable, "--input-goal", str(source),
            "--output-result", str(destination), "--ros-args", "--params-file", str(cls.params)],
            env=cls.environment, capture_output=True, text=True, timeout=30.0, check=True)
        if not destination.is_file():
            raise RuntimeError("Native file preview omitted the result: " + completed.stdout + completed.stderr)
        response = deserialize_message(destination.read_bytes(), PreviewRecoveryMotion.Result)
        if response.status != "NEEDS_CONTEXT" or "fingerprint" not in response.reason:
            raise AssertionError("Native model discovery did not reject the mismatched model: " + response.reason)
        cls.model_fingerprint = response.model_fingerprint
        assert len(cls.model_fingerprint) == 64

    def goal(self):
        return PreviewRecoveryMotionTests.goal(self)

    def run_goal(self, goal):
        from cais_lab_robotics.action import PreviewRecoveryMotion
        from rclpy.serialization import deserialize_message, serialize_message

        with tempfile.TemporaryDirectory(dir=self.directory, prefix="goal-") as directory:
            source = Path(directory) / "goal.cdr"
            destination = Path(directory) / "result.cdr"
            original_goal = copy.deepcopy(goal)
            original = serialize_message(goal)
            source.write_bytes(original)
            completed = subprocess.run([self.executable, "--input-goal", str(source),
                "--output-result", str(destination), "--ros-args", "--params-file", str(self.params)],
                env=self.environment, capture_output=True, text=True, timeout=30.0, check=True)
            self.assertTrue(destination.is_file(), completed.stdout + completed.stderr)
            response = deserialize_message(destination.read_bytes(), PreviewRecoveryMotion.Result)
            self.assertEqual(source.read_bytes(), original)
            self.assertEqual(goal, original_goal)
            self.assertEqual(response.resource_id, goal.resource_id)
            self.assertEqual(response.capture_fingerprint, goal.capture_fingerprint)
            self.assertEqual(response.proposal_fingerprint, goal.proposal_fingerprint)
            self.assertFalse(response.continuous_collision_verified)
            return response

    def test_file_complete_path_retains_peer_mimic_and_input_scene(self):
        PreviewRecoveryMotionTests.test_complete_joint_path_retains_peer_and_input_scene(self)

    def test_file_joint_target_retains_exact_binary64_endpoint(self):
        start, target = -1.1032364978932825 / 4, 5.142049823384553 / 4
        self.assertNotEqual(start + (target - start), target)
        goal = self.goal()
        goal.planning_scene.robot_state.joint_state.position[0] = start
        goal.start_state = copy.deepcopy(goal.planning_scene.robot_state)
        goal.joint_target.position = [target]
        goal.scene_fingerprint = _scene_message_fingerprint(goal.planning_scene)
        response = self.run_goal(goal)
        self.assertEqual(response.status, "PREVIEW_COMPLETE", response.reason)
        trajectory = response.trajectory.joint_trajectory
        self.assertEqual(trajectory.joint_names, ["slider"])
        positions = [point.positions[0] for point in trajectory.points]
        self.assertEqual(positions[0], start)
        self.assertEqual(positions[-1], target)
        self.assertGreater(len(positions), 2)
        self.assertTrue(all(start <= value <= target for value in positions))
        self.assertTrue(all(0 < after - before <= goal.max_joint_step
                            for before, after in zip(positions, positions[1:])))
        end = dict(zip(response.end_state.joint_state.name,
                       response.end_state.joint_state.position, strict=True))
        self.assertEqual(end["slider"], target)
        self.assertEqual(end["peer"], 1.5)
        self.assertEqual(end["peer_mimic"], 1.49)

    def test_file_exact_joint_target_keeps_interior_and_endpoint_collision_checks(self):
        from geometry_msgs.msg import Pose
        from moveit_msgs.msg import CollisionObject
        from shape_msgs.msg import SolidPrimitive

        start, target = -1.1032364978932825 / 4, 5.142049823384553 / 4
        for position in ((start + target) / 2, target):
            with self.subTest(position=position):
                goal = self.goal()
                goal.planning_scene.robot_state.joint_state.position[0] = start
                goal.start_state = copy.deepcopy(goal.planning_scene.robot_state)
                goal.joint_target.position = [target]
                obstacle = CollisionObject(id="recorded_obstacle")
                obstacle.header.frame_id = "world"
                obstacle.primitives = [SolidPrimitive(type=SolidPrimitive.BOX,
                                                       dimensions=[0.01, 0.1, 0.1])]
                pose = Pose()
                pose.position.x, pose.orientation.w = position, 1.0
                obstacle.primitive_poses = [pose]
                goal.planning_scene.world.collision_objects = [obstacle]
                goal.scene_fingerprint = _scene_message_fingerprint(goal.planning_scene)
                response = self.run_goal(goal)
                self.assertEqual(response.status, "NEEDS_CONTEXT", response.reason)
                self.assertFalse(response.path_complete)
                evidence = _assert_rejected_sample(self, goal, response, "joint_target")
                self.assertGreater(evidence["state_index"], 0)

    def test_file_collision_and_missing_start_remain_unresolved(self):
        PreviewRecoveryMotionTests.test_collision_and_missing_start_are_unresolved(self)

    def test_file_collision_reports_exact_pairs_at_start_and_during_motion(self):
        from geometry_msgs.msg import Pose
        from moveit_msgs.msg import CollisionObject
        from shape_msgs.msg import SolidPrimitive

        for position, phase in ((0.0, "Starting configuration"), (0.35, "Joint interpolation")):
            with self.subTest(position=position):
                goal = self.goal()
                obstacle = CollisionObject(id="recorded_obstacle")
                obstacle.header.frame_id = "world"
                obstacle.primitives = [SolidPrimitive(type=SolidPrimitive.BOX, dimensions=[0.1, 0.1, 0.1])]
                pose = Pose()
                pose.position.x, pose.orientation.w = position, 1.0
                obstacle.primitive_poses = [pose]
                goal.planning_scene.world.collision_objects = [obstacle]
                goal.scene_fingerprint = _scene_message_fingerprint(goal.planning_scene)
                response = self.run_goal(goal)
                self.assertEqual(response.status, "NEEDS_CONTEXT")
                self.assertFalse(response.path_complete)
                self.assertIn(phase, response.reason)
                self.assertIn("recorded_obstacle", response.reason)
                self.assertIn("tool", response.reason)
                _assert_rejected_sample(self, goal, response, "initial" if position == 0 else "joint_target")

    def test_file_stale_model_and_scene_fingerprints_are_rejected(self):
        PreviewRecoveryMotionTests.test_model_and_scene_fingerprints_reject_stale_evidence(self)

    def test_file_attachment_is_preserved_without_live_mutation(self):
        PreviewRecoveryMotionTests.test_captured_attachment_is_preserved_without_live_mutation(self)

    def test_file_support_departure_preserves_scene(self):
        PreviewRecoveryMotionTests.test_support_departure_preserves_scene_and_separates_sampled_evidence(self)

    def test_file_support_departure_invalid_and_incomplete_contact_fails(self):
        PreviewRecoveryMotionTests.test_support_departure_missing_record_deeper_contact_and_no_separation_fail(self)

    def test_file_support_departure_keeps_unrelated_collisions(self):
        PreviewRecoveryMotionTests.test_support_departure_does_not_exempt_unrelated_contact(self)

    def test_file_truncated_contact_inventory_cannot_approve_a_path(self):
        from geometry_msgs.msg import Pose
        from moveit_msgs.msg import CollisionObject
        from shape_msgs.msg import SolidPrimitive

        goal = self.goal()
        for index in range(4097):
            obj = CollisionObject(id=f"contact_{index}")
            obj.header.frame_id = "world"
            obj.primitives = [SolidPrimitive(type=SolidPrimitive.BOX, dimensions=[.01] * 3)]
            pose = Pose()
            pose.orientation.w = 1.
            obj.primitive_poses = [pose]
            goal.planning_scene.world.collision_objects.append(obj)
        goal.scene_fingerprint = _scene_message_fingerprint(goal.planning_scene)
        response = self.run_goal(goal)
        self.assertEqual(response.status, "NEEDS_CONTEXT")
        self.assertFalse(response.path_complete)
        self.assertFalse(response.sampled_collision_rejected)
        self.assertEqual(response.collision_rejection_evidence_json, "")
        self.assertIn("truncated", response.reason)

    def test_file_missing_cartesian_solver_cannot_produce_complete_path(self):
        PreviewRecoveryMotionTests.test_missing_cartesian_solver_cannot_produce_complete_path(self)

    def test_file_incomplete_scene_and_conflicting_start_are_rejected(self):
        goal = self.goal()
        goal.planning_scene.is_diff = True
        goal.scene_fingerprint = _scene_message_fingerprint(goal.planning_scene)
        response = self.run_goal(goal)
        self.assertEqual(response.status, "NEEDS_CONTEXT")
        self.assertIn("complete PlanningScene", response.reason)
        self.assertFalse(response.path_complete)
        goal = self.goal()
        goal.start_state.joint_state.position[0] = 0.1
        response = self.run_goal(goal)
        self.assertEqual(response.status, "NEEDS_CONTEXT")
        self.assertIn("Start state must exactly match", response.reason)
        self.assertFalse(response.path_complete)

    def test_file_invalid_cdr_does_not_create_success_evidence(self):
        with tempfile.TemporaryDirectory(dir=self.directory, prefix="invalid-") as directory:
            source = Path(directory) / "goal.cdr"
            destination = Path(directory) / "result.cdr"
            source.write_bytes(b"not a serialized PreviewRecoveryMotion.Goal")
            completed = subprocess.run([self.executable, "--input-goal", str(source),
                "--output-result", str(destination), "--ros-args", "--params-file", str(self.params)],
                env=self.environment, capture_output=True, text=True, timeout=30.0, check=False)
            self.assertNotEqual(completed.returncode, 0)
            self.assertFalse(destination.exists())


    def endpoint_goal(self, *, part_z=.5, obstacle=False):
        from geometry_msgs.msg import Pose
        from moveit_msgs.msg import CollisionObject
        from shape_msgs.msg import SolidPrimitive
        from rosidl_runtime_py.convert import message_to_ordereddict
        from cais_spade_llm.recovery_framework.recovery_motion_preview import bind_scene_endpoint

        goal = self.goal()
        objects = []
        rows = [("part/link/collision", [0., .25, part_z], [.0625] * 3),
                ("table/link/collision", [0., .25, .4375], [.5, .5, .0625])]
        if obstacle:
            rows.append(("obstacle/link/collision", [0., .25, .5], [.03125] * 3))
        for identity, position, dimensions in rows:
            obj = CollisionObject(id=identity)
            obj.header.frame_id = "world"
            obj.primitives = [SolidPrimitive(type=SolidPrimitive.BOX, dimensions=dimensions)]
            pose = Pose()
            pose.position.x, pose.position.y, pose.position.z = position
            pose.orientation.w = 1.
            obj.primitive_poses = [pose]
            objects.append(obj)
        goal.planning_scene.world.collision_objects = objects
        goal.scene_fingerprint = _scene_message_fingerprint(goal.planning_scene)
        record = {"version": 1, "status": "prepared", "resource_id": goal.resource_id,
            "command_sent": False, "execution_authorized": False,
            "source": {"step_id": "snap_part_to_slot", "step_fingerprint": "recorded-step"},
            "changed_object_ids": ["part/link/collision"], "support_pairs": [{
                "part_object_id": "part/link/collision", "support_object_id": "table/link/collision",
                "plane_point": [0., .25, .46875], "plane_normal": [0., 0., 1.]}]}
        bound = bind_scene_endpoint(message_to_ordereddict(goal), record)
        goal.mode, goal.waypoints = "scene_endpoint", []
        goal.joint_target.name, goal.joint_target.position = [], []
        goal.scene_endpoint_json, goal.scene_endpoint_fingerprint = bound["scene_endpoint_json"], bound["scene_endpoint_fingerprint"]
        return goal

    def test_scene_endpoint_checks_free_world_geometry_without_motion(self):
        from rosidl_runtime_py.convert import message_to_ordereddict
        from cais_spade_llm.recovery_framework.recovery_motion_preview import preview_result_record

        goal = self.endpoint_goal()
        before = message_to_ordereddict(goal)
        response = self.run_goal(goal)
        self.assertEqual(response.status, "SCENE_ENDPOINT_COMPLETE", response.reason)
        record = preview_result_record(message_to_ordereddict(goal), message_to_ordereddict(response))
        audit = record["scene_endpoint_evidence"]
        self.assertEqual(audit["support_pairs"][0]["minimum_signed_distance"], 0.)
        self.assertEqual(len(audit["checked_pairs"]), 1)
        self.assertTrue(audit["pairs_complete"])
        self.assertFalse(record["path_complete"])
        self.assertFalse(record["trajectory"]["joint_trajectory"]["points"])
        self.assertEqual(message_to_ordereddict(goal), before)

    def test_scene_endpoint_rejects_positive_depth_and_unrelated_world_contacts(self):
        from rosidl_runtime_py.convert import message_to_ordereddict
        from cais_spade_llm.recovery_framework.recovery_motion_preview import preview_result_record

        for kwargs in ({"part_z": .499}, {"obstacle": True}):
            with self.subTest(kwargs=kwargs):
                goal = self.endpoint_goal(**kwargs)
                response = self.run_goal(goal)
                self.assertEqual(response.status, "SCENE_ENDPOINT_REJECTED", response.reason)
                audit = preview_result_record(message_to_ordereddict(goal), message_to_ordereddict(response))["scene_endpoint_evidence"]
                self.assertFalse(audit["collision_free"])
                self.assertTrue(audit["pairs_complete"])
                self.assertFalse(response.sampled_collision_rejected)

    def test_scene_endpoint_cannot_hide_world_contact_using_scene_acm(self):
        from moveit_msgs.msg import AllowedCollisionEntry
        from rosidl_runtime_py.convert import message_to_ordereddict
        from cais_spade_llm.recovery_framework.recovery_motion_preview import bind_scene_endpoint

        goal = self.endpoint_goal(obstacle=True)
        goal.planning_scene.allowed_collision_matrix.entry_names = ["part/link/collision", "obstacle/link/collision"]
        goal.planning_scene.allowed_collision_matrix.entry_values = [AllowedCollisionEntry(enabled=[True, True])] * 2
        goal.scene_fingerprint = _scene_message_fingerprint(goal.planning_scene)
        bound = bind_scene_endpoint(message_to_ordereddict(goal), json.loads(goal.scene_endpoint_json))
        goal.scene_endpoint_json, goal.scene_endpoint_fingerprint = bound["scene_endpoint_json"], bound["scene_endpoint_fingerprint"]
        response = self.run_goal(goal)
        self.assertEqual(response.status, "SCENE_ENDPOINT_REJECTED", response.reason)

    def test_scene_endpoint_retains_ordinary_robot_collision_checks(self):
        from rosidl_runtime_py.convert import message_to_ordereddict
        from cais_spade_llm.recovery_framework.recovery_motion_preview import bind_scene_endpoint

        goal = self.endpoint_goal()
        part = goal.planning_scene.world.collision_objects[0]
        part.primitive_poses[0].position.y = part.primitive_poses[0].position.z = 0.
        goal.scene_fingerprint = _scene_message_fingerprint(goal.planning_scene)
        evidence = json.loads(goal.scene_endpoint_json)
        evidence["support_pairs"] = []
        bound = bind_scene_endpoint(message_to_ordereddict(goal), evidence)
        goal.scene_endpoint_json, goal.scene_endpoint_fingerprint = bound["scene_endpoint_json"], bound["scene_endpoint_fingerprint"]
        response = self.run_goal(goal)
        self.assertEqual(response.status, "SCENE_ENDPOINT_REJECTED", response.reason)
        audit = json.loads(response.scene_endpoint_evidence_json)
        self.assertFalse(audit["robot_collision_free"])
        self.assertTrue(audit["robot_contacts"])

    def test_scene_endpoint_rejects_unbound_or_invented_support_plane(self):
        import hashlib

        for defect in ("fingerprint", "plane", "motion"):
            with self.subTest(defect=defect):
                goal = self.endpoint_goal()
                if defect == "fingerprint":
                    goal.scene_endpoint_fingerprint = "different"
                elif defect == "plane":
                    evidence = json.loads(goal.scene_endpoint_json)
                    evidence["support_pairs"][0]["plane_point"][2] += .001
                    goal.scene_endpoint_json = json.dumps(evidence)
                    goal.scene_endpoint_fingerprint = hashlib.sha256(goal.scene_endpoint_json.encode()).hexdigest()
                else:
                    goal.joint_target.name, goal.joint_target.position = ["slider"], [.25]
                response = self.run_goal(goal)
                self.assertEqual(response.status, "NEEDS_CONTEXT", response.reason)
                self.assertFalse(response.scene_endpoint_evidence_json)


class PreviewRecoveryMotionCartesianFileTests(unittest.TestCase):
    """Check exact waypoint FK after native KDL and bounded joint refinement."""

    @staticmethod
    def model_parameters():
        parameters = _model_parameters()
        urdf = parameters["robot_description"].replace(
            '<joint name="slider" type="prismatic"><parent link="world"/><child link="tool"/><axis xyz="1 0 0"/><limit lower="-2" upper="2" velocity="1" effort="10"/></joint>', '')
        links = ["base", "x", "y", "z", "roll", "pitch", "tool"]
        chain = '<link name="base"/><joint name="mount" type="fixed"><parent link="world"/><child link="base"/><origin xyz="0.4 -0.3 0.1" rpy="0 0 0.4"/></joint>'
        axes = ["1 0 0", "0 1 0", "0 0 1"] * 2
        for index, (parent, child) in enumerate(zip(links, links[1:], strict=False)):
            if child != "tool":
                chain += f'<link name="{child}"/>'
            kind = "prismatic" if index < 3 else "revolute"
            chain += (f'<joint name="joint{index}" type="{kind}"><parent link="{parent}"/>'
                      f'<child link="{child}"/><axis xyz="{axes[index]}"/>'
                      '<limit lower="-2" upper="2" velocity="1" effort="10"/></joint>')
        chain += ('<link name="grasp_mount"/><joint name="grasp_fixed" type="fixed">'
                  '<parent link="tool"/><child link="grasp_mount"/><origin xyz="0.01 0 0"/></joint>')
        parameters["robot_description"] = urdf.replace('</robot>', chain + '</robot>')
        parameters["robot_description_semantic"] = (
            '<robot name="preview_test"><group name="arm"><chain base_link="base" tip_link="tool"/></group></robot>')
        parameters["robot_description_kinematics"] = {"arm": {
            "kinematics_solver": "kdl_kinematics_plugin/KDLKinematicsPlugin",
            "kinematics_solver_timeout": .05,
            "kinematics_solver_search_resolution": .005,
        }}
        parameters["robot_description_planning"]["joint_limits"] = {name: {
            "has_velocity_limits": True, "max_velocity": 1.0,
            "has_acceleration_limits": True, "max_acceleration": 1.0,
        } for name in [*(f"joint{index}" for index in range(6)), "peer"]}
        return parameters

    @classmethod
    def setUpClass(cls):
        PreviewRecoveryMotionFileTests.setUpClass.__func__(cls)

    def run_goal(self, goal):
        return PreviewRecoveryMotionFileTests.run_goal(self, goal)

    @staticmethod
    def forward(values):
        from cais_spade_llm.recovery_framework.geometry import multiply, quaternion

        base = quaternion([0., 0., .4])
        rotation = base
        for index, angle in enumerate(values[3:]):
            angles = [0., 0., 0.]
            angles[index] = angle
            rotation = multiply(rotation, quaternion(angles))
        return [.4 + math.cos(.4) * values[0] - math.sin(.4) * values[1],
                -.3 + math.sin(.4) * values[0] + math.cos(.4) * values[1], .1 + values[2], *rotation]

    def goal(self, targets):
        from cais_lab_robotics.action import PreviewRecoveryMotion
        from geometry_msgs.msg import Pose

        goal = PreviewRecoveryMotion.Goal()
        goal.planning_scene.robot_model_name = "preview_test"
        goal.planning_scene.robot_state.joint_state.name = [*(f"joint{index}" for index in range(6)), "peer", "peer_mimic"]
        goal.planning_scene.robot_state.joint_state.position = [0.] * 6 + [1.5, 1.49]
        goal.start_state = copy.deepcopy(goal.planning_scene.robot_state)
        goal.group_name, goal.link_name, goal.frame_id = "arm", "tool", "world"
        goal.mode = "cartesian"
        for values in targets:
            position = self.forward(values)
            pose = Pose()
            pose.position.x, pose.position.y, pose.position.z = position[:3]
            pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = position[3:]
            goal.waypoints.append(pose)
        goal.max_translation_step = goal.max_rotation_step = goal.max_joint_step = .05
        goal.velocity_scaling = goal.acceleration_scaling = .1
        goal.timeout_seconds = 10.
        goal.resource_id, goal.capture_fingerprint, goal.proposal_fingerprint = "resource", "capture", "proposal"
        goal.scene_fingerprint = _scene_message_fingerprint(goal.planning_scene)
        goal.model_fingerprint = self.model_fingerprint
        return goal

    def mating_goal(self, *, depth=.0003):
        from geometry_msgs.msg import Pose
        from moveit_msgs.msg import AttachedCollisionObject, CollisionObject
        from shape_msgs.msg import SolidPrimitive
        from rosidl_runtime_py.convert import message_to_ordereddict
        from cais_spade_llm.recovery_framework.geometry import compose
        from cais_spade_llm.recovery_framework.recovery_motion_preview import bind_mating_contact

        goal = self.goal([[0., 0., -.05, 0., 0., 0.]])
        grasp = [0., 0., -.06, 0., 0., 0., 1.]
        start_tool = self.forward([0.] * 6)
        start_part = compose(start_tool, grasp)
        end_tool = self.forward([0., 0., -.05, 0., 0., 0.])
        end_part = compose(end_tool, grasp)
        part = CollisionObject(id="part/link/collision")
        part.header.frame_id = "tool"
        part.primitives = [SolidPrimitive(type=SolidPrimitive.BOX, dimensions=[.02] * 3)]
        part_pose = Pose()
        part_pose.position.z, part_pose.orientation.w = -.06, 1.
        part.primitive_poses = [part_pose]
        goal.planning_scene.robot_state.attached_collision_objects = [
            AttachedCollisionObject(link_name="tool", touch_links=["tool"], object=part)]
        shaft = CollisionObject(id="shaft/link/collision")
        shaft.header.frame_id = "world"
        shaft.primitives = [SolidPrimitive(type=SolidPrimitive.BOX, dimensions=[.04, .04, .04])]
        shaft_pose = Pose()
        shaft_pose.position.x, shaft_pose.position.y = end_part[:2]
        shaft_pose.position.z, shaft_pose.orientation.w = end_part[2] - .03 + depth, 1.
        shaft.primitive_poses = [shaft_pose]
        goal.planning_scene.world.collision_objects = [shaft]
        goal.start_state = copy.deepcopy(goal.planning_scene.robot_state)
        goal.scene_fingerprint = _scene_message_fingerprint(goal.planning_scene)
        record = {"version": 1, "status": "prepared", "resource_id": goal.resource_id,
            "part_name": "part", "model_name": "part", "part_object_id": part.id,
            "reference_link": "tool", "frame_id": "world", "command_sent": False, "execution_authorized": False,
            "start_tool_pose": start_tool, "start_part_pose": start_part, "tool_to_part_pose": grasp,
            "target_tool_pose": end_tool, "start_part_z": start_part[2],
            "target_origin_pose": dict(zip("xyz", end_part[:3], strict=True)),
            "binding": {name: name for name in ("state_fingerprint", "model_fingerprint", "owner_evidence_fingerprint",
                "physics_fingerprint", "scene_object_bindings_fingerprint", "peer_fingerprint")},
            "simulation_mating_contact": {"target_collision_object": shaft.id, "axis_tolerance_m": .0005,
                "max_contact_depth_m": .0005, "target_yaw_rad": .4, "yaw_tolerance_rad": .005,
                "ignore_tooth_contact_with": [], "retain_fixture_attachment": True}}
        prepared = bind_mating_contact(message_to_ordereddict(goal), record)
        goal.mating_contact_json = prepared["mating_contact_json"]
        goal.mating_contact_fingerprint = prepared["mating_contact_fingerprint"]
        return goal

    def test_captured_mating_contact_is_limited_to_aligned_insertion(self):
        from rosidl_runtime_py.convert import message_to_ordereddict
        from cais_spade_llm.recovery_framework.recovery_motion_preview import preview_result_record

        goal = self.mating_goal()
        original = copy.deepcopy(goal)
        ordinary = copy.deepcopy(goal)
        ordinary.mating_contact_json = ordinary.mating_contact_fingerprint = ""
        rejected = self.run_goal(ordinary)
        self.assertEqual(rejected.status, "NEEDS_CONTEXT")
        self.assertTrue(rejected.sampled_collision_rejected)
        response = self.run_goal(goal)
        self.assertEqual(response.status, "PREVIEW_COMPLETE", response.reason)
        captured = preview_result_record(message_to_ordereddict(goal), message_to_ordereddict(response))
        audit = captured["mating_contact_evidence"]
        self.assertEqual(len(audit["samples"]), len(response.trajectory.joint_trajectory.points))
        self.assertTrue(any(row["contacts"] for row in audit["samples"]))
        self.assertFalse(audit["continuous_collision_verified"])
        self.assertFalse(captured["execution_authorized"])
        self.assertEqual(goal, original)
        self.assertEqual(response.end_state.attached_collision_objects[0].object.id, "part/link/collision")

    def test_mating_accepts_only_model_fixed_attachment_links(self):
        from rosidl_runtime_py.convert import message_to_ordereddict
        from cais_spade_llm.recovery_framework.recovery_motion_preview import bind_mating_contact, preview_result_record

        for link, offset, accepted in (("grasp_mount", -.01, True), ("pitch", 0., False)):
            with self.subTest(link=link):
                goal = self.mating_goal()
                attachment = goal.planning_scene.robot_state.attached_collision_objects[0]
                attachment.link_name = attachment.object.header.frame_id = link
                attachment.touch_links = ["tool", link]
                attachment.object.primitive_poses[0].position.x = offset
                goal.start_state = copy.deepcopy(goal.planning_scene.robot_state)
                goal.scene_fingerprint = _scene_message_fingerprint(goal.planning_scene)
                bound = bind_mating_contact(message_to_ordereddict(goal), json.loads(goal.mating_contact_json))
                goal.mating_contact_json = bound["mating_contact_json"]
                goal.mating_contact_fingerprint = bound["mating_contact_fingerprint"]
                response = self.run_goal(goal)
                if accepted:
                    self.assertEqual(response.status, "PREVIEW_COMPLETE", response.reason)
                    preview_result_record(message_to_ordereddict(goal), message_to_ordereddict(response))
                    self.assertEqual(response.end_state.attached_collision_objects[0].link_name, link)
                else:
                    self.assertEqual(response.status, "NEEDS_CONTEXT")
                    self.assertIn("not fixed to the controlled link", response.reason)
                    self.assertFalse(response.trajectory.joint_trajectory.points)

    def test_captured_mating_depth_and_unrelated_contacts_still_reject(self):
        from geometry_msgs.msg import Pose
        from moveit_msgs.msg import CollisionObject
        from shape_msgs.msg import SolidPrimitive
        from rosidl_runtime_py.convert import message_to_ordereddict
        from cais_spade_llm.recovery_framework.recovery_motion_preview import bind_mating_contact

        goal = self.mating_goal(depth=.0007)
        response = self.run_goal(goal)
        self.assertEqual(response.status, "NEEDS_CONTEXT")
        self.assertEqual(_assert_rejected_sample(self, goal, response, "cartesian")["kind"], "mating_depth")
        goal = self.mating_goal()
        obstacle = CollisionObject(id="unrelated/link/collision")
        obstacle.header.frame_id = "world"
        obstacle.primitives = [SolidPrimitive(type=SolidPrimitive.BOX, dimensions=[.01] * 3)]
        pose = Pose()
        pose.position.x, pose.position.y, pose.position.z, pose.orientation.w = .4, -.3, .015, 1.
        obstacle.primitive_poses = [pose]
        goal.planning_scene.world.collision_objects.append(obstacle)
        goal.scene_fingerprint = _scene_message_fingerprint(goal.planning_scene)
        prepared = bind_mating_contact(message_to_ordereddict(goal), json.loads(goal.mating_contact_json))
        goal.mating_contact_json = prepared["mating_contact_json"]
        goal.mating_contact_fingerprint = prepared["mating_contact_fingerprint"]
        response = self.run_goal(goal)
        self.assertEqual(response.status, "NEEDS_CONTEXT")
        self.assertEqual(_assert_rejected_sample(self, goal, response, "cartesian")["kind"], "unrelated_collision")

    def test_mating_rejects_transit_stale_and_missing_geometry(self):
        import hashlib

        for defect in ("transit", "orientation", "stale", "fingerprint", "geometry", "part", "route"):
            with self.subTest(defect=defect):
                goal = self.mating_goal()
                record = json.loads(goal.mating_contact_json)
                if defect == "transit":
                    goal.waypoints[0].position.x += .01
                elif defect == "orientation":
                    goal.waypoints[0].orientation.z = math.sin(.3)
                    goal.waypoints[0].orientation.w = math.cos(.3)
                elif defect == "stale":
                    record["preview_binding"]["capture_fingerprint"] = "changed"
                elif defect == "fingerprint":
                    record["binding"]["peer_fingerprint"] = ""
                elif defect == "geometry":
                    record["simulation_mating_contact"]["target_collision_object"] = "missing"
                elif defect == "part":
                    record["part_object_id"] = "other/link/collision"
                else:
                    goal.route_search_json = json.dumps(self.route_settings())
                goal.mating_contact_json = json.dumps(record)
                goal.mating_contact_fingerprint = hashlib.sha256(goal.mating_contact_json.encode()).hexdigest()
                result = self.run_goal(goal)
                self.assertEqual(result.status, "NEEDS_CONTEXT", result.reason)
                self.assertFalse(result.path_complete)
                self.assertFalse(result.trajectory.joint_trajectory.points)

    @staticmethod
    def route_settings():
        """Bound a candidate calculation using this fixture's declared workspace."""
        return {"version": 1, "algorithm": "Cartesian RRT extension with greedy goal connections",
                "waypoint_index": 0, "maximum_edges": 128, "maximum_seconds": 5., "random_seed": 42,
                "static_capabilities_fingerprint": "recorded-fixture-capabilities",
                "gripper_reach": {"frame": "world", "origin_pose": {"x": .4, "y": -.3, "z": .1},
                                  "max_xy_radius_m": .8, "z_min_m": -.6, "z_max_m": .8, "tolerance_m": 0.}}

    def test_calculated_route_needs_separate_strict_replay_and_preserves_endpoints(self):
        from geometry_msgs.msg import Pose
        from moveit_msgs.msg import CollisionObject
        from shape_msgs.msg import SolidPrimitive

        goal = self.goal([[.3, 0., 0., 0., 0., 0.], [.3, 0., -.1, 0., 0., 0.]])
        obstacle = CollisionObject(id="recorded-obstacle")
        obstacle.header.frame_id = "world"
        obstacle.primitives = [SolidPrimitive(type=SolidPrimitive.BOX, dimensions=[.08] * 3)]
        pose = Pose()
        pose.position.x, pose.position.y, pose.position.z = self.forward([.15, 0., 0., 0., 0., 0.])[:3]
        pose.orientation.w = 1.
        obstacle.primitive_poses = [pose]
        goal.planning_scene.world.collision_objects = [obstacle]
        goal.scene_fingerprint = _scene_message_fingerprint(goal.planning_scene)
        original = copy.deepcopy(goal)
        goal.route_search_json = json.dumps(self.route_settings())
        response = self.run_goal(goal)
        self.assertEqual(response.status, "ROUTE_CANDIDATE", response.reason)
        self.assertFalse(response.path_complete)
        self.assertFalse(response.trajectory.joint_trajectory.points)
        self.assertFalse(response.continuous_collision_verified)
        audit = json.loads(response.route_candidate_json)
        self.assertTrue(audit["candidate_only"])
        self.assertTrue(audit["solved"])
        self.assertLess(audit["edges"][0]["fraction"], 1.)
        self.assertLessEqual(len(audit["edges"]), 128)
        self.assertEqual(audit["route_search"], self.route_settings())
        self.assertEqual(audit["request_binding"]["scene_fingerprint"], goal.scene_fingerprint)
        self.assertEqual(audit["original_waypoints"], [
            {"position": {"x": p.position.x, "y": p.position.y, "z": p.position.z},
             "orientation": {"x": p.orientation.x, "y": p.orientation.y,
                             "z": p.orientation.z, "w": p.orientation.w}}
            for p in original.waypoints])
        for index, expected in zip(audit["mandatory_waypoint_indices"], audit["original_waypoints"], strict=True):
            self.assertEqual(audit["derived_waypoints"][index], expected)
        for node in audit["waypoints"]:
            state = node["state"]["joint_state"]
            observed = dict(zip(state["name"], state["position"], strict=True))
            self.assertEqual(observed["peer"], 1.5)
            self.assertEqual(observed["peer_mimic"], 1.49)
        replay = copy.deepcopy(original)
        replay.waypoints = []
        for row in audit["derived_waypoints"]:
            waypoint = Pose()
            waypoint.position.x, waypoint.position.y, waypoint.position.z = [row["position"][axis] for axis in "xyz"]
            waypoint.orientation = copy.deepcopy(original.waypoints[0].orientation)
            self.assertEqual(row["orientation"], audit["original_waypoints"][0]["orientation"])
            replay.waypoints.append(waypoint)
        completed = self.run_goal(replay)
        self.assertEqual(completed.status, "PREVIEW_COMPLETE", completed.reason)
        self.assertTrue(completed.path_complete)
        self.assertFalse(completed.route_candidate_json)
        self.assertFalse(completed.continuous_collision_verified)
        self.assertEqual(goal.planning_scene, original.planning_scene)
        self.assertEqual(goal.start_state, original.start_state)

    def test_route_calculation_rejects_invalid_policy_and_unsupported_changes(self):
        invalid = [
            ("maximum_edges", 129), ("maximum_edges", 0), ("maximum_seconds", 121.),
            ("version", 2), ("waypoint_index", 1), ("random_seed", -1),
            ("static_capabilities_fingerprint", ""),
        ]
        for field, value in invalid:
            with self.subTest(field=field, value=value):
                goal = self.goal([[.2, 0., 0., 0., 0., 0.]])
                settings = self.route_settings()
                settings[field] = value
                goal.route_search_json = json.dumps(settings)
                result = self.run_goal(goal)
                self.assertEqual(result.status, "NEEDS_CONTEXT")
                self.assertFalse(result.path_complete)
                self.assertFalse(result.trajectory.joint_trajectory.points)
        goal = self.goal([[.2, 0., 0., 0., .1, 0.]])
        goal.route_search_json = json.dumps(self.route_settings())
        result = self.run_goal(goal)
        self.assertEqual(result.status, "NEEDS_CONTEXT")
        self.assertIn("fixed starting orientation", result.reason)

    def test_route_calculation_keeps_reach_limits_and_reports_exhausted_budget(self):
        goal = self.goal([[1., 0., 0., 0., 0., 0.]])
        goal.route_search_json = json.dumps(self.route_settings())
        result = self.run_goal(goal)
        self.assertEqual(result.status, "NEEDS_CONTEXT")
        self.assertIn("endpoint capabilities", result.reason)
        goal = self.goal([[.2, 0., 0., 0., 0., 0.]])
        settings = self.route_settings()
        settings["maximum_seconds"] = 1e-12
        goal.route_search_json = json.dumps(settings)
        result = self.run_goal(goal)
        self.assertEqual(result.status, "NEEDS_CONTEXT")
        audit = json.loads(result.route_candidate_json)
        self.assertFalse(audit["solved"])
        self.assertTrue(audit["stopped_by_limit"])
        self.assertEqual(audit["derived_waypoints"], [])
        self.assertFalse(result.trajectory.joint_trajectory.points)

    def test_world_frame_waypoints_use_refined_joint_evidence_and_retain_peers(self):
        targets = [[.2, .1, .15, .08, .09, -.06], [.3, .08, .22, .15, -.11, .2]]
        goal = self.goal(targets)
        response = self.run_goal(goal)
        self.assertEqual(response.status, "PREVIEW_COMPLETE", response.reason)
        self.assertIn("Cartesian waypoint FK residuals <= 1e-13", response.reason)
        self.assertEqual(response.cartesian_position_precision, 1e-13)
        self.assertEqual(response.cartesian_orientation_precision, 1e-13)
        self.assertLessEqual(response.cartesian_position_residual, response.cartesian_position_precision)
        self.assertLessEqual(response.cartesian_orientation_residual, response.cartesian_orientation_precision)
        self.assertNotIn("added refinement states: 0", response.reason)
        trajectory = response.trajectory.joint_trajectory
        self.assertEqual(trajectory.joint_names, [f"joint{index}" for index in range(6)])
        for desired in targets:
            pose = self.forward(desired)
            errors = [max(abs(a - b) for a, b in zip(self.forward(point.positions), pose, strict=True))
                      for point in trajectory.points]
            self.assertLess(min(errors), 2e-13)
        for first, second in zip(trajectory.points, trajectory.points[1:], strict=False):
            self.assertLessEqual(max(abs(a - b) for a, b in zip(first.positions, second.positions, strict=True)), goal.max_joint_step)
        observed = dict(zip(response.end_state.joint_state.name, response.end_state.joint_state.position, strict=True))
        self.assertEqual(observed["peer"], 1.5)
        self.assertEqual(observed["peer_mimic"], 1.49)
        # The supplied initial state is checked before the sampled-state counter.
        self.assertGreaterEqual(response.checked_states + 1, len(trajectory.points))
        diagnostic = json.loads(response.cartesian_diagnostics_json)
        self.assertTrue(diagnostic["diagnostic_only"])
        self.assertTrue(diagnostic["complete_inputs"])
        self.assertNotIn("failure_kind", diagnostic)
        self.assertEqual(diagnostic["group_default_ik_timeout"], .05)
        self.assertEqual(len(diagnostic["waypoints"]), len(targets))
        for query in diagnostic["ik_queries"]:
            self.assertTrue(query["success"])
            self.assertEqual(query["error_code"], 1)
            self.assertEqual(query["timeout_seconds"], .05)
            self.assertEqual(query["consistency_limits"], [.05] * 6)
            self.assertEqual(query["joint_names"], [f"joint{index}" for index in range(6)])
            self.assertFalse(query["return_approximate_solution"])

    def test_cartesian_support_departure_rejects_reentry_and_deeper_contact(self):
        for targets, phrase in (([[.2, 0., 0., 0., 0., 0.], [0.] * 6], "reenters"),
                                ([[-.03, 0., 0., 0., 0., 0.]], "penetration")):
            with self.subTest(phrase=phrase):
                goal = _support_departure_goal(self.goal(targets), self.forward([0.] * 6))
                response = self.run_goal(goal)
                self.assertEqual(response.status, "NEEDS_CONTEXT", response.reason)
                self.assertTrue(response.sampled_collision_rejected, response.reason)
                self.assertIn(phrase, response.reason)
                evidence = _assert_rejected_sample(self, goal, response, "trajectory" if phrase == "reenters" else "cartesian")
                self.assertTrue(evidence["support_pairs"])

    def test_cartesian_support_departure_has_sampled_separation(self):
        goal = _support_departure_goal(self.goal([[.2, 0., 0., 0., 0., 0.]]), self.forward([0.] * 6))
        response = self.run_goal(goal)
        self.assertEqual(response.status, "PREVIEW_COMPLETE", response.reason)
        self.assertTrue(json.loads(response.support_departure_evidence_json)["pairs"][0]["endpoint_separated"])
        self.assertFalse(response.continuous_collision_verified)

    def test_cartesian_collision_cannot_be_accepted_by_refinement(self):
        from geometry_msgs.msg import Pose
        from moveit_msgs.msg import CollisionObject
        from shape_msgs.msg import SolidPrimitive

        target = [.3, .08, .22, .15, -.11, .2]
        goal = self.goal([target])
        obstacle = CollisionObject(id="cartesian_target_obstacle")
        obstacle.header.frame_id = "world"
        obstacle.primitives = [SolidPrimitive(type=SolidPrimitive.BOX, dimensions=[.02] * 3)]
        pose = Pose()
        pose.position.x, pose.position.y, pose.position.z = self.forward(target)[:3]
        pose.orientation.w = 1.
        obstacle.primitive_poses = [pose]
        goal.planning_scene.world.collision_objects = [obstacle]
        goal.scene_fingerprint = _scene_message_fingerprint(goal.planning_scene)
        response = self.run_goal(goal)
        self.assertEqual(response.status, "NEEDS_CONTEXT")
        self.assertFalse(response.path_complete)
        self.assertIn("cartesian_target_obstacle", response.reason)

    def test_unreachable_cartesian_target_and_deadline_remain_unresolved(self):
        for changed in ("unreachable", "deadline"):
            with self.subTest(changed=changed):
                goal = self.goal([[.3, .08, .22, .15, -.11, .2]])
                if changed == "unreachable":
                    goal.waypoints[-1].position.z = 8.
                else:
                    goal.timeout_seconds = 1e-9
                response = self.run_goal(goal)
                self.assertEqual(response.status, "NEEDS_CONTEXT")
                self.assertFalse(response.path_complete)
                if changed == "unreachable":
                    from rosidl_runtime_py.convert import message_to_ordereddict
                    from cais_spade_llm.recovery_framework.recovery_motion_preview import preview_result_record

                    record = preview_result_record(message_to_ordereddict(goal), message_to_ordereddict(response))
                    diagnostic = record["cartesian_failure_evidence"]
                    self.assertTrue(diagnostic["complete_inputs"])
                    self.assertEqual(diagnostic["failure_kind"], "ik_search_failed")
                    self.assertFalse(diagnostic["ik_queries"][-1]["success"])
                    self.assertEqual(diagnostic["ik_queries"][-1]["error_code"], -6)
                    self.assertNotIn("solution", diagnostic["ik_queries"][-1])
                    self.assertTrue(diagnostic["waypoints"][-1]["last_accepted_state"]["joint_state"]["name"])
                    self.assertFalse(record["coverage"]["path_complete"])
                    self.assertFalse(record["execution_authorized"])

    def test_cartesian_truncated_contacts_are_missing_evidence_not_ik_failure(self):
        from geometry_msgs.msg import Pose
        from moveit_msgs.msg import CollisionObject
        from shape_msgs.msg import SolidPrimitive

        goal = self.goal([[.2, 0., 0., 0., 0., 0.]])
        for index in range(4097):
            obstacle = CollisionObject(id=f"contact_{index}")
            obstacle.header.frame_id = "world"
            obstacle.primitives = [SolidPrimitive(type=SolidPrimitive.BOX, dimensions=[.01] * 3)]
            pose = Pose()
            pose.position.x, pose.position.y, pose.position.z = self.forward([.2, 0., 0., 0., 0., 0.])[:3]
            pose.orientation.w = 1.
            obstacle.primitive_poses = [pose]
            goal.planning_scene.world.collision_objects.append(obstacle)
        goal.scene_fingerprint = _scene_message_fingerprint(goal.planning_scene)
        response = self.run_goal(goal)
        self.assertEqual(response.status, "NEEDS_CONTEXT")
        self.assertFalse(response.path_complete)
        self.assertFalse(response.sampled_collision_rejected)
        diagnostic = json.loads(response.cartesian_diagnostics_json)
        self.assertEqual(diagnostic["failure_kind"], "collision_evidence_incomplete")
        self.assertFalse(diagnostic["complete_inputs"])
        self.assertTrue(diagnostic["validity_rejections"])

    def test_recorded_fixed_mount_drives_actual_fk_without_changing_source_model(self):
        from geometry_msgs.msg import TransformStamped
        from rosidl_runtime_py.convert import message_to_ordereddict
        from cais_spade_llm.recovery_framework.geometry import compose, quaternion, rotate
        from cais_spade_llm.recovery_framework.recovery_motion_preview import recorded_root_model_fingerprint

        values = [.3, .08, .22, .15, -.11, .2]
        goal = self.goal([values])
        mounted = [.40000003, -.299998, .1000002, *quaternion([.000002, -.000003, .400004])]
        rotation = quaternion([0., 0., -.4])
        inverse = [*rotate(rotation, [-.4, .3, -.1]), *rotation]
        expected = compose(mounted, compose(inverse, self.forward(values)))
        pose = goal.waypoints[0]
        pose.position.x, pose.position.y, pose.position.z = expected[:3]
        pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = expected[3:]
        root = TransformStamped(child_frame_id="base")
        root.header.frame_id = "world"
        root.transform.translation.x, root.transform.translation.y, root.transform.translation.z = mounted[:3]
        root.transform.rotation.x, root.transform.rotation.y, root.transform.rotation.z, root.transform.rotation.w = mounted[3:]
        goal.recorded_root_transforms = [root]
        response = self.run_goal(goal)
        self.assertEqual(response.status, "PREVIEW_COMPLETE", response.reason)
        self.assertEqual(response.model_fingerprint, self.model_fingerprint)
        self.assertEqual(response.derived_model_fingerprint,
                         recorded_root_model_fingerprint(self.model_fingerprint, [message_to_ordereddict(root)]))
        obtained = response.trajectory.joint_trajectory.points[-1].positions
        actual = compose(mounted, compose(inverse, self.forward(obtained)))
        self.assertLess(max(abs(a - b) for a, b in zip(actual, expected, strict=True)), 2e-13)
        source_only = self.run_goal(self.goal([values]))
        self.assertEqual(source_only.status, "PREVIEW_COMPLETE", source_only.reason)
        self.assertEqual(source_only.derived_model_fingerprint, "")

    def test_recorded_moving_root_must_match_supplied_joint_configuration(self):
        from geometry_msgs.msg import TransformStamped

        for changed in (False, True):
            with self.subTest(changed=changed):
                goal = self.goal([[.2, .1, .15, .08, .09, -.06]])
                peer = TransformStamped(child_frame_id="peer_tool")
                peer.header.frame_id = "world"
                peer.transform.translation.x = 1.5001 if changed else 1.5
                peer.transform.rotation.w = 1.
                goal.recorded_root_transforms = [peer]
                response = self.run_goal(goal)
                self.assertEqual(response.status, "NEEDS_CONTEXT" if changed else "PREVIEW_COMPLETE", response.reason)
                if changed:
                    self.assertIn("moving root conflicts", response.reason)

    def test_recorded_root_frames_and_duplicates_are_rejected(self):
        from geometry_msgs.msg import TransformStamped

        for defect in ("frame", "duplicate", "unknown"):
            with self.subTest(defect=defect):
                goal = self.goal([[.2, .1, .15, .08, .09, -.06]])
                root = TransformStamped(child_frame_id="missing" if defect == "unknown" else "base")
                root.header.frame_id = "missing_frame" if defect == "frame" else "world"
                root.transform.rotation.w = 1.
                goal.recorded_root_transforms = [root, copy.deepcopy(root)] if defect == "duplicate" else [root]
                response = self.run_goal(goal)
                self.assertEqual(response.status, "NEEDS_CONTEXT")
                self.assertFalse(response.path_complete)


class PreviewRecoveryMotionBaseFileTests(unittest.TestCase):
    """Check base ancestors against the complete saved scene without a node."""

    @staticmethod
    def model_parameters():
        parameters = _model_parameters()
        urdf = parameters['robot_description']
        urdf = urdf.replace('<parent link="world"/><child link="tool"/>',
                            '<parent link="KMR_base_link"/><child link="tool"/><origin xyz="0 0 0.7"/>')
        chain = '''<link name="KMR_base_x_link"/><link name="KMR_base_y_link"/>
          <link name="KMR_base_link"><collision><origin xyz="0 0 0.1"/>
            <geometry><box size="0.2 0.2 0.2"/></geometry></collision></link>
          <joint name="KMR_base_x_joint" type="prismatic"><parent link="world"/><child link="KMR_base_x_link"/>
            <axis xyz="1 0 0"/><limit lower="-2" upper="2" velocity="1" effort="10"/></joint>
          <joint name="KMR_base_y_joint" type="prismatic"><parent link="KMR_base_x_link"/><child link="KMR_base_y_link"/>
            <axis xyz="0 1 0"/><limit lower="-2" upper="2" velocity="1" effort="10"/></joint>
          <joint name="KMR_base_yaw_joint" type="continuous"><parent link="KMR_base_y_link"/><child link="KMR_base_link"/>
            <axis xyz="0 0 1"/><limit velocity="1" effort="10"/></joint>'''
        parameters['robot_description'] = urdf.replace('</robot>', chain + '</robot>')
        return parameters

    @classmethod
    def setUpClass(cls):
        PreviewRecoveryMotionFileTests.setUpClass.__func__(cls)

    def run_goal(self, goal):
        return PreviewRecoveryMotionFileTests.run_goal(self, goal)

    def goal(self):
        from trajectory_msgs.msg import JointTrajectoryPoint
        from builtin_interfaces.msg import Duration

        goal = PreviewRecoveryMotionTests.goal(self)
        goal.mode, goal.base_link = 'base_trajectory', 'KMR_base_link'
        goal.joint_target.name, goal.joint_target.position = [], []
        names = ['KMR_base_x_joint', 'KMR_base_y_joint', 'KMR_base_yaw_joint']
        goal.planning_scene.robot_state.joint_state.name.extend(names)
        goal.planning_scene.robot_state.joint_state.position.extend([0.] * 3)
        goal.start_state = copy.deepcopy(goal.planning_scene.robot_state)
        goal.scene_fingerprint = _scene_message_fingerprint(goal.planning_scene)
        goal.base_trajectory.joint_names = names
        goal.base_trajectory.points = [
            JointTrajectoryPoint(positions=[0., 0., 0.], velocities=[.05, 0., 0.], time_from_start=Duration()),
            JointTrajectoryPoint(positions=[.6, 0., 0.], velocities=[0., 0., 0.], time_from_start=Duration(sec=3))]
        return goal

    def test_file_base_motion_preserves_timing_arm_peers_and_complete_scene(self):
        from rosidl_runtime_py.convert import message_to_ordereddict
        from cais_spade_llm.recovery_framework.recovery_motion_preview import preview_result_record

        goal = self.goal()
        response = self.run_goal(goal)
        self.assertEqual(response.status, 'PREVIEW_COMPLETE', response.reason)
        self.assertEqual(response.trajectory.joint_trajectory, goal.base_trajectory)
        end = dict(zip(response.end_state.joint_state.name, response.end_state.joint_state.position))
        self.assertEqual([end[name] for name in goal.base_trajectory.joint_names], [.6, 0., 0.])
        self.assertEqual([end[name] for name in ('slider', 'peer', 'peer_mimic')], [0., 1.5, 1.49])
        record = preview_result_record(message_to_ordereddict(goal), message_to_ordereddict(response))
        self.assertFalse(record['execution_authorized'])
        self.assertFalse(record['coverage']['continuous_collision_verified'])

    def test_file_base_motion_checks_interior_and_carried_part_contacts(self):
        from geometry_msgs.msg import Pose
        from moveit_msgs.msg import CollisionObject, AttachedCollisionObject
        from shape_msgs.msg import SolidPrimitive

        for carried in (False, True):
            with self.subTest(carried=carried):
                goal = self.goal()
                obstacle = CollisionObject(id='obstacle')
                obstacle.header.frame_id = 'world'
                obstacle.primitives = [SolidPrimitive(type=SolidPrimitive.SPHERE, dimensions=[.03])]
                pose = Pose()
                pose.position.x, pose.position.z, pose.orientation.w = (.95 if carried else .3), .7, 1.
                obstacle.primitive_poses = [pose]
                goal.planning_scene.world.collision_objects = [obstacle]
                if carried:
                    part = CollisionObject(id='part/link/collision')
                    part.header.frame_id = 'tool'
                    part.primitives = [SolidPrimitive(type=SolidPrimitive.SPHERE, dimensions=[.03])]
                    local = Pose()
                    local.position.x, local.orientation.w = .4, 1.
                    part.primitive_poses = [local]
                    goal.planning_scene.robot_state.attached_collision_objects = [
                        AttachedCollisionObject(link_name='tool', touch_links=['tool'], object=part)]
                goal.start_state = copy.deepcopy(goal.planning_scene.robot_state)
                goal.scene_fingerprint = _scene_message_fingerprint(goal.planning_scene)
                response = self.run_goal(goal)
                self.assertEqual(response.status, 'NEEDS_CONTEXT')
                self.assertFalse(response.path_complete)
                evidence = _assert_rejected_sample(self, goal, response, 'base_trajectory')
                if carried:
                    self.assertTrue(any('part/link/collision' in (row['body_1'], row['body_2']) for row in evidence['contacts']))

    def test_file_base_motion_rejects_missing_joints_peers_stale_start_and_bad_timing(self):
        for defect in ('peer', 'missing', 'duplicate', 'start', 'velocity', 'time', 'bounds', 'nonfinite', 'mixed_mode'):
            with self.subTest(defect=defect):
                goal = self.goal()
                if defect == 'peer':
                    goal.base_trajectory.joint_names[0] = 'peer'
                elif defect == 'missing':
                    goal.base_trajectory.joint_names.pop()
                    for point in goal.base_trajectory.points:
                        point.positions = list(point.positions)[:-1]
                        point.velocities = list(point.velocities)[:-1]
                elif defect == 'duplicate':
                    goal.base_trajectory.joint_names[0] = goal.base_trajectory.joint_names[1]
                elif defect == 'start':
                    goal.base_trajectory.points[0].positions[0] = .1
                elif defect == 'velocity':
                    goal.base_trajectory.points[0].velocities = []
                elif defect == 'time':
                    goal.base_trajectory.points[-1].time_from_start.sec = 0
                elif defect == 'bounds':
                    goal.base_trajectory.points[-1].positions[0] = 3.
                elif defect == 'nonfinite':
                    goal.base_trajectory.points[-1].positions[0] = float('inf')
                else:
                    goal.mode = 'joint_target'
                    goal.joint_target.name, goal.joint_target.position = ['slider'], [.1]
                response = self.run_goal(goal)
                self.assertEqual(response.status, 'NEEDS_CONTEXT')
                self.assertFalse(response.path_complete)


if __name__ == "__main__":
    unittest.main()
