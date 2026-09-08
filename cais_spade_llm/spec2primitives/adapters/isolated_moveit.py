from __future__ import annotations

"""Validate Cartesian segments in a private MoveIt scene with execution disabled."""

import asyncio
import math
import threading
import time
import subprocess
import tempfile
import uuid
import xml.etree.ElementTree as ET
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from ..agents.ra.refinement_records import owned_path

_ALLOWED_CAPABILITIES = frozenset(
    {
        "move_group/MoveGroupCartesianPathService",
        "move_group/MoveGroupKinematicsService",
        "move_group/MoveGroupStateValidationService",
        "move_group/MoveGroupGetPlanningSceneService",
        "move_group/ApplyPlanningSceneService",
        "move_group/MoveGroupPlanService",
        "move_group/MoveGroupQueryPlannersService",
    }
)


class IsolatedMoveItSession:
    """Own one disposable namespace, process and scene for a frozen candidate."""

    def __init__(
        self,
        root: Path,
        robot: Mapping[str, Any],
        scene: Mapping[str, Any],
        profile: Mapping[str, Any],
    ) -> None:
        """Retain immutable inputs; no ROS connection is made in the constructor."""
        self.root, self.robot, self.scene, self.profile = root, robot, scene, profile
        self.namespace = "/spec2primitives_validation_" + uuid.uuid4().hex
        self._process: subprocess.Popen[bytes] | None = None
        self._temporary: tempfile.TemporaryDirectory[str] | None = None
        self._node: Any = None
        self._context: Any = None
        self._executor: Any = None
        self._cancelled = threading.Event()
        self._clients: dict[str, Any] = {}
        self._log: Any = None
        self.last_worker_log = ""
        self._starting = True

    async def __aenter__(self) -> IsolatedMoveItSession:
        """Start the owned non-executing worker and populate its private scene."""
        try:
            await self._work(self._start)
        except BaseException:
            # Cancellation must also tear down an owned process; re-raise intact.
            await asyncio.to_thread(self._close)
            raise
        return self

    async def _work(self, function: Any, *args: Any, **kwargs: Any) -> Any:
        task = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            self._cancelled.set()
            try:
                await task
            except (OSError, RuntimeError, ValueError):
                pass
            raise

    async def __aexit__(self, *args: Any) -> None:
        """Release the worker and read-only ROS clients on every exit."""
        await asyncio.to_thread(self._close)

    def _start(self) -> None:
        import rclpy
        from ament_index_python.packages import get_package_prefix, get_package_share_directory
        from rclpy.node import Node
        from rclpy.context import Context
        from rclpy.executors import SingleThreadedExecutor

        share = Path(get_package_share_directory("moveit_ros_move_group"))
        capabilities = {
            entry.attrib["name"]
            for entry in ET.parse(share / "default_capabilities_plugin_description.xml").findall(
                "class"
            )
        }
        if not _ALLOWED_CAPABILITIES <= capabilities:
            raise RuntimeError("Installed MoveIt lacks the required validation-only capabilities.")
        parameters = dict(self.robot["model_parameters"])
        parameters.update(
            {
                "allow_trajectory_execution": False,
                "disable_capabilities": " ".join(sorted(capabilities - _ALLOWED_CAPABILITIES)),
                "publish_robot_description": False,
                "publish_robot_description_semantic": False,
                "planning_scene_monitor.publish_planning_scene": False,
            }
        )
        self._temporary = tempfile.TemporaryDirectory(prefix="spec2primitives_moveit_")
        directory = Path(self._temporary.name)
        parameter_file = directory / "parameters.yaml"
        parameter_file.write_text(
            yaml.safe_dump({"/**": {"ros__parameters": parameters}}), encoding="utf-8"
        )
        executable = (
            Path(get_package_prefix("moveit_ros_move_group"))
            / "lib/moveit_ros_move_group/move_group"
        )
        self._log = (directory / "worker.log").open("wb")
        # No shared scene, joint-state or collision-object topic can mutate the
        # frozen worker. TF remains a read-only runtime source; fixed transforms
        # and every planning start state are explicitly supplied below.
        remaps = [
            "__ns:=" + self.namespace,
            "__node:=move_group",
            "/joint_states:=" + self.namespace + "/frozen_joint_states",
            "/planning_scene:=" + self.namespace + "/frozen_planning_scene",
            "/collision_object:=" + self.namespace + "/unused_collision_object",
            "/attached_collision_object:=" + self.namespace + "/unused_attached_object",
        ]
        command = [str(executable), "--ros-args", "--params-file", str(parameter_file)]
        for remap in remaps:
            command.extend(["-r", remap])
        self._process = subprocess.Popen(
            command, stdin=subprocess.DEVNULL, stdout=self._log, stderr=subprocess.STDOUT
        )
        self._context = Context()
        rclpy.init(context=self._context)
        self._executor = SingleThreadedExecutor(context=self._context)
        self._node = Node(
            "spec2primitives_validation_client_" + uuid.uuid4().hex[:8], context=self._context
        )
        self._executor.add_node(self._node)
        self._apply_scene()
        self._starting = False

    def _call(self, service_type: Any, suffix: str, request: Any) -> Any:
        if self._cancelled.is_set():
            raise RuntimeError("Private MoveIt validation was cancelled.")
        if suffix not in {
            "apply_planning_scene",
            "compute_cartesian_path",
            "check_state_validity",
            "compute_fk",
        }:
            raise ValueError("Only private plan-only services are available.")
        client = self._clients.get(suffix)
        if client is None:
            client = self._node.create_client(service_type, self.namespace + "/" + suffix)
            self._clients[suffix] = client
        timeout_field = "worker_startup_timeout_sec" if self._starting else "service_timeout_sec"
        deadline = time.monotonic() + float(self.profile[timeout_field])
        while not client.wait_for_service(timeout_sec=0.1):
            if self._cancelled.is_set() or time.monotonic() >= deadline:
                raise RuntimeError(
                    "Private MoveIt validation service is unavailable or cancelled: " + suffix
                )
        future = client.call_async(request)
        deadline = time.monotonic() + float(self.profile["planning_timeout_sec"])
        while not future.done() and not self._cancelled.is_set() and time.monotonic() < deadline:
            self._executor.spin_once(timeout_sec=0.1)
        if not future.done() or future.result() is None:
            future.cancel()
            raise RuntimeError("Private MoveIt validation timed out: " + suffix)
        return future.result()

    def _state(self, joints: Mapping[str, Any], attached: Mapping[str, Any] | None = None) -> Any:
        from moveit_msgs.msg import AttachedCollisionObject, RobotState

        state = RobotState()
        state.is_diff = False
        state.joint_state.name = list(joints["names"])
        state.joint_state.position = list(joints["positions"])
        if attached is not None:
            item = AttachedCollisionObject()
            item.link_name = self.robot["ee_link"]
            item.touch_links = list(attached.get("touch_links", []))
            item.object = self._object(attached, self.robot["ee_link"])
            state.attached_collision_objects.append(item)
        return state

    def _object(self, value: Mapping[str, Any], frame: str) -> Any:
        import hashlib
        from geometry_msgs.msg import Point, Pose
        from moveit_msgs.msg import CollisionObject
        from shape_msgs.msg import Mesh, MeshTriangle, Plane, SolidPrimitive

        item = CollisionObject()
        item.id, item.header.frame_id = value["object_id"], frame
        item.operation = CollisionObject.ADD
        raw_pose = value["pose"]
        pose = Pose()
        pose.position.x, pose.position.y, pose.position.z = (
            float(raw_pose[key]) for key in ("x", "y", "z")
        )
        pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = (
            float(raw_pose[key]) for key in ("qx", "qy", "qz", "qw")
        )
        if "plane" in value:
            coefficients = [float(coefficient) for coefficient in value["plane"]]
            if len(coefficients) != 4 or not all(
                math.isfinite(coefficient) for coefficient in coefficients
            ):
                raise ValueError("Observed collision plane is invalid.")
            item.planes, item.plane_poses = [Plane(coef=coefficients)], [pose]
        elif "mesh" in value:
            mesh_ref = value["mesh"]
            path = owned_path(self.root, mesh_ref["ref"])
            if hashlib.sha256(path.read_bytes()).hexdigest() != mesh_ref["sha256"]:
                raise ValueError("Collision mesh changed after observation grounding.")
            with np.load(path, allow_pickle=False) as arrays:
                triangles = np.asarray(arrays["triangles_m"], dtype=float)
            if (
                triangles.ndim != 3
                or triangles.shape[1:] != (3, 3)
                or len(triangles) > 200000
                or not np.isfinite(triangles).all()
            ):
                raise ValueError("Collision mesh is invalid or exceeds the bounded scene budget.")
            vertices, inverse = np.unique(triangles.reshape(-1, 3), axis=0, return_inverse=True)
            mesh = Mesh()
            mesh.vertices = [Point(x=float(x), y=float(y), z=float(z)) for x, y, z in vertices]
            mesh.triangles = [
                MeshTriangle(vertex_indices=list(map(int, indices)))
                for indices in inverse.reshape(-1, 3)
            ]
            item.meshes, item.mesh_poses = [mesh], [pose]
        else:
            shape = SolidPrimitive()
            shape.type = SolidPrimitive.BOX
            shape.dimensions = [float(value) for value in value["size_m"]]
            if len(shape.dimensions) != 3 or any(
                not math.isfinite(size) or size <= 0 for size in shape.dimensions
            ):
                raise ValueError("Collision box dimensions are invalid.")
            item.primitives, item.primitive_poses = [shape], [pose]
        return item

    def _apply_scene(
        self, *, remove: str | None = None, add: Mapping[str, Any] | None = None
    ) -> None:
        from moveit_msgs.msg import CollisionObject, PlanningScene
        from moveit_msgs.srv import ApplyPlanningScene

        scene = PlanningScene()
        # Retain the model's SRDF self-collision matrix in this new private scene.
        scene.is_diff = True
        scene.robot_state.is_diff = True
        if remove is None and add is None:
            scene.robot_state = self._state(self.robot["joint_state"])
            scene.world.collision_objects = [
                self._object(item, self.robot["frame_id"]) for item in self.scene["objects"]
            ]
        if remove is not None:
            item = CollisionObject(id=remove, operation=CollisionObject.REMOVE)
            scene.world.collision_objects.append(item)
        if add is not None:
            scene.world.collision_objects.append(self._object(add, self.robot["frame_id"]))
        response = self._call(
            ApplyPlanningScene, "apply_planning_scene", ApplyPlanningScene.Request(scene=scene)
        )
        if not response.success:
            raise RuntimeError("Private collision scene was not accepted.")

    async def change_custody(
        self, *, remove: str | None = None, add: Mapping[str, Any] | None = None
    ) -> None:
        """Change only hypothetical object placement in the worker's private scene."""
        await self._work(self._apply_scene, remove=remove, add=add)

    async def check_segment(
        self,
        *,
        joints: Mapping[str, Any],
        start_pose: Mapping[str, Any],
        target_pose: Mapping[str, Any],
        attached: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        """Require one complete collision-checked Cartesian segment from its prefix state."""
        return await self._work(self._check_segment, joints, start_pose, target_pose, attached)

    def _check_segment(
        self,
        joints: Mapping[str, Any],
        start_pose: Mapping[str, Any],
        target_pose: Mapping[str, Any],
        attached: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        from geometry_msgs.msg import Pose
        from moveit_msgs.srv import GetCartesianPath, GetPositionFK, GetStateValidity

        state = self._state(joints, attached)
        if not self._within_joint_limits(
            dict(zip(joints["names"], joints["positions"], strict=True))
        ):
            return {
                "status": "failed",
                "message": "The prefix joint state violates recorded robot joint limits.",
            }
        validity = GetStateValidity.Request(robot_state=state, group_name=self.robot["group_name"])
        if not self._call(GetStateValidity, "check_state_validity", validity).valid:
            return {
                "status": "failed",
                "message": "The proposed prefix state violates collision or state validity constraints.",
            }
        fk = GetPositionFK.Request(fk_link_names=[self.robot["ee_link"]], robot_state=state)
        fk.header.frame_id = self.robot["frame_id"]
        fk_result = self._call(GetPositionFK, "compute_fk", fk)
        if fk_result.error_code.val != 1 or not fk_result.pose_stamped:
            return {
                "status": "unknown",
                "message": "Forward kinematics could not establish the prefix pose.",
            }
        if not self._pose_matches(fk_result.pose_stamped[0].pose, start_pose):
            return {
                "status": "unknown",
                "message": "Prefix joint state and controlled-link pose disagree.",
            }
        request = GetCartesianPath.Request()
        request.header.frame_id, request.group_name, request.link_name = (
            self.robot["frame_id"],
            self.robot["group_name"],
            self.robot["ee_link"],
        )
        request.start_state = state
        request.avoid_collisions = True
        request.max_step = float(self.profile["cartesian_max_step_m"])
        request.jump_threshold = float(self.profile["cartesian_jump_threshold"])
        pose = Pose()
        pose.position.x, pose.position.y, pose.position.z = (
            float(target_pose[key]) for key in ("x", "y", "z")
        )
        pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = (
            float(target_pose[key]) for key in ("qx", "qy", "qz", "qw")
        )
        request.waypoints = [pose]
        response = self._call(GetCartesianPath, "compute_cartesian_path", request)
        if response.error_code.val != 1 or not math.isclose(response.fraction, 1.0, abs_tol=1e-9):
            return {
                "status": "failed",
                "message": "A complete collision-checked Cartesian segment was not found; this does not prove global infeasibility.",
                "fraction": response.fraction,
                "error_code": response.error_code.val,
            }
        trajectory = response.solution.joint_trajectory
        if not trajectory.points or len(trajectory.points) > 20000:
            return {
                "status": "unknown",
                "message": "The Cartesian result is empty or exceeds the trajectory budget.",
            }
        positions = dict(zip(joints["names"], joints["positions"], strict=True))
        for point in trajectory.points:
            point_values = dict(zip(trajectory.joint_names, point.positions, strict=True))
            if not set(point_values) <= positions.keys() or not self._within_joint_limits(
                point_values
            ):
                return {
                    "status": "failed",
                    "message": "The Cartesian trajectory violates recorded joint limits or uses unmeasured joints.",
                }
        positions.update(zip(trajectory.joint_names, trajectory.points[-1].positions, strict=True))
        end = {"names": list(positions), "positions": list(positions.values())}
        fk.robot_state = self._state(end, attached)
        endpoint = self._call(GetPositionFK, "compute_fk", fk)
        if (
            endpoint.error_code.val != 1
            or not endpoint.pose_stamped
            or not self._pose_matches(endpoint.pose_stamped[0].pose, target_pose)
        ):
            return {
                "status": "unknown",
                "message": "The returned trajectory does not establish the authored endpoint pose.",
            }
        return {
            "status": "passed",
            "message": "Complete direct Cartesian segment checked in the isolated scene.",
            "fraction": response.fraction,
            "end_joint_state": end,
            "trajectory": {
                "joint_names": list(trajectory.joint_names),
                "positions": [list(point.positions) for point in trajectory.points],
            },
        }

    def _pose_matches(self, actual: Any, expected: Mapping[str, float]) -> bool:
        position, rotation = actual.position, actual.orientation
        distance = np.linalg.norm(
            np.asarray([position.x, position.y, position.z])
            - [expected[key] for key in ("x", "y", "z")]
        )
        error = (
            Rotation.from_quat([rotation.x, rotation.y, rotation.z, rotation.w]).inv()
            * Rotation.from_quat([expected[key] for key in ("qx", "qy", "qz", "qw")])
        ).magnitude()
        return bool(
            distance <= float(self.robot["position_tolerance_m"])
            and error <= float(self.profile["fk_orientation_tolerance_rad"])
        )

    def _within_joint_limits(self, values: Mapping[str, float]) -> bool:
        description = ET.fromstring(self.robot["model_parameters"]["robot_description"])
        joints = {joint.attrib["name"]: joint for joint in description.findall("joint")}
        for name, value in values.items():
            if name not in joints or not math.isfinite(value):
                return False
            joint = joints[name]
            limit = joint.find("limit")
            if joint.attrib["type"] in {"revolute", "prismatic"}:
                if limit is None or not float(limit.attrib["lower"]) <= value <= float(
                    limit.attrib["upper"]
                ):
                    return False
            prefix = "robot_description_planning.joint_limits." + name + "."
            parameters = self.robot["model_parameters"]
            if parameters.get(prefix + "has_position_limits") is True:
                if (
                    not parameters[prefix + "min_position"]
                    <= value
                    <= parameters[prefix + "max_position"]
                ):
                    return False
        return True

    def _close(self) -> None:
        if self._node is not None:
            self._node.destroy_node()
            self._node = None
        if self._process is not None:
            if self._process.poll() is None:
                self._process.terminate()
            try:
                self._process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=3)
            self._process = None
        if self._log is not None:
            self._log.close()
            self._log = None
        if self._temporary is not None:
            log_path = Path(self._temporary.name) / "worker.log"
            if log_path.exists():
                self.last_worker_log = log_path.read_bytes()[-12000:].decode(
                    "utf-8", errors="replace"
                )
            self._temporary.cleanup()
            self._temporary = None
        if self._executor is not None:
            self._executor.shutdown()
            self._executor = None
        if self._context is not None:
            self._context.try_shutdown()
            self._context = None
