from __future__ import annotations

"""Execute pinned simulation commands without recovery or composition tools."""

import asyncio
import hashlib
import json
import math
import threading
import time
import uuid
import xml.etree.ElementTree as ET
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from .robot_validation_context import matrix_pose, pose_matrix
from ..agents.ra.validation_scope import GAZEBO_OBSERVED_SCOPE, VALIDATION_SCOPE


def load_execution_profile() -> dict[str, Any]:
    """Load simulation transport settings; no product coordinates are configured."""
    profile = json.loads(
        (Path(__file__).resolve().parents[1] / "config/gazebo_execution.json").read_bytes()
    )
    for key in (
        "instance_position_tolerance_m",
        "instance_orientation_tolerance_rad",
        "service_timeout_sec",
        "trajectory_timeout_pad_sec",
        "stop_timeout_sec",
        "capture_timeout_sec",
    ):
        if (
            type(profile.get(key)) not in (int, float)
            or not math.isfinite(profile[key])
            or not 0 < profile[key] <= 60
        ):
            raise ValueError(f"Invalid Gazebo execution setting: {key}.")
    if Path(profile["world_file"]).name != profile["world_file"]:
        raise ValueError("Gazebo execution requires a package-local world file.")
    return profile


def _sdf_pose(element: ET.Element) -> np.ndarray:
    pose = element.find("pose")
    if pose is None:
        return np.eye(4)
    if pose.attrib:
        raise ValueError("Relative SDF pose conventions require an explicit execution adapter.")
    values = [float(value) for value in (pose.text or "").split()]
    if len(values) != 6 or not all(math.isfinite(value) for value in values):
        raise ValueError("Invalid SDF pose.")
    result = np.eye(4)
    result[:3, :3] = Rotation.from_euler("xyz", values[3:]).as_matrix()
    result[:3, 3] = values[:3]
    return result


def fixture_instances(share: Path, world_file: str, cad_path: Path) -> list[dict[str, Any]]:
    """Connect exact CAD bytes to declared instances after composition is sealed.

    World spawn poses are deliberately unused. Live Gazebo state supplies instance
    placement; SDF link/visual transforms supply only the CAD-to-model transform.
    """
    world_path = share / "worlds" / world_file
    world = ET.parse(world_path).getroot().find("world")
    if world is None:
        raise ValueError("The configured Gazebo fixture has no world.")
    cad_hash = hashlib.sha256(cad_path.read_bytes()).hexdigest()
    entries: list[tuple[ET.Element, str, Path]] = []
    for model in world.findall("model"):
        entries.append((model, model.attrib["name"], world_path))
    for include in world.findall("include"):
        uri = include.findtext("uri", "")
        if not uri.startswith("model://"):
            continue
        relative = Path(uri[len("model://") :])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Gazebo model reference escapes its model directory.")
        source = share / "models" / relative / "model.sdf"
        if not source.is_file():
            continue  # Gazebo's unrelated sun/table assets need no CAD binding.
        model = ET.parse(source).getroot().find("model")
        if model is not None:
            entries.append((model, include.findtext("name") or model.attrib["name"], source))
    result = []
    for model, model_name, source in entries:
        for link in model.findall("link"):
            for visual in link.findall("visual"):
                mesh = visual.find("geometry/mesh")
                if mesh is None:
                    continue
                uri = mesh.findtext("uri", "")
                if not uri.startswith("model://"):
                    continue
                relative = Path(uri[len("model://") :])
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError("Gazebo mesh reference escapes its model directory.")
                mesh_path = share / "models" / relative
                if not mesh_path.is_file() and relative.parts[0] == "cad_models":
                    mesh_path = share / relative
                if (
                    not mesh_path.is_file()
                    or hashlib.sha256(mesh_path.read_bytes()).hexdigest() != cad_hash
                ):
                    continue
                scale = [float(value) for value in mesh.findtext("scale", "1 1 1").split()]
                if len(scale) != 3 or not all(
                    math.isfinite(value) and value > 0 for value in scale
                ):
                    raise ValueError("Invalid Gazebo CAD scale.")
                result.append(
                    {
                        "model_name": model_name,
                        "link": link.attrib["name"],
                        "mesh_sha256": cad_hash,
                        "mesh_scale": scale,
                        "model_from_CAD": (_sdf_pose(link) @ _sdf_pose(visual)).tolist(),
                        "sources": [
                            {
                                "path": str(path.resolve()),
                                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                            }
                            for path in (world_path, source, mesh_path, cad_path)
                        ],
                    }
                )
    return result


def match_instance(
    instances: list[dict[str, Any]],
    live: Mapping[str, Mapping[str, Any]],
    part: Mapping[str, Any],
    *,
    cad_scale: float,
    profile: Mapping[str, Any],
    interaction_root: Path | None = None,
    validation_scope: str = VALIDATION_SCOPE,
) -> dict[str, Any]:
    """Require one live instance matching the accepted CAD and scoped observed geometry."""
    observed_bounds = validation_scope == GAZEBO_OBSERVED_SCOPE
    reference = "observed_bounds_center" if observed_bounds else "CAD_origin"
    if part.get("frame_id") != "world" or part.get("reference_point") != reference:
        raise ValueError(f"Instance binding requires the accepted world-frame {reference} reference.")
    observed = pose_matrix(part["reference_pose"] if observed_bounds else part["origin_pose"])
    vertices = None
    if observed_bounds:
        from ..agents.ra.refinement_records import owned_path

        if interaction_root is None:
            raise ValueError("Observed instance matching requires its approved CAD evidence root.")
        mesh = part["CAD_mesh"]
        path = owned_path(interaction_root, mesh["ref"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != mesh["sha256"]:
            raise ValueError("The selected CAD mesh changed before instance matching.")
        with np.load(path, allow_pickle=False) as arrays:
            vertices = np.asarray(arrays["triangles_m"], dtype=float).reshape(-1, 3)
        if not len(vertices) or not np.isfinite(vertices).all():
            raise ValueError("The selected CAD mesh does not establish finite instance bounds.")
    matches = []
    for instance in instances:
        if instance["model_name"] not in live:
            continue
        if not np.allclose(instance["mesh_scale"], [cad_scale] * 3, rtol=0, atol=1e-12):
            raise ValueError("Gazebo mesh units differ from the accepted CAD units.")
        actual = pose_matrix(live[instance["model_name"]]["pose"]) @ np.asarray(
            instance["model_from_CAD"]
        )
        if observed_bounds:
            points = vertices @ actual[:3, :3].T + actual[:3, 3]
            minimum, maximum = points.min(axis=0), points.max(axis=0)
            center = (minimum + maximum) / 2
            distance = float(np.linalg.norm(center - observed[:3, 3]))
            bounds_error = float(np.max(np.abs(np.asarray([minimum, maximum]) - np.asarray([
                part["bounds_m"]["minimum"], part["bounds_m"]["maximum"]]))))
            compatible = max(distance, bounds_error) <= profile["instance_position_tolerance_m"]
            metrics = {"bounds_difference_m": bounds_error, "orientation_difference_rad": None}
        else:
            distance = float(np.linalg.norm(actual[:3, 3] - observed[:3, 3]))
            angle = float(Rotation.from_matrix(actual[:3, :3].T @ observed[:3, :3]).magnitude())
            compatible = distance <= profile["instance_position_tolerance_m"] and angle <= profile["instance_orientation_tolerance_rad"]
            metrics = {"orientation_difference_rad": angle}
        if compatible:
            matches.append(
                {
                    **deepcopy(instance),
                    "object_id": part["object_id"],
                    "origin_pose": matrix_pose(actual),
                    "live_state": deepcopy(live[instance["model_name"]]),
                    "position_difference_m": distance,
                    **metrics,
                }
            )
    if len(matches) != 1:
        raise ValueError(
            f"The accepted CAD and observed pose match {len(matches)} Gazebo instances; exactly one is required."
        )
    return matches[0]


def prepare_trajectory(
    trajectory: Mapping[str, Any], robot: Mapping[str, Any], speed: float
) -> dict[str, Any]:
    """Scale only timing and reject missing timing or violated joint limits."""
    if type(speed) not in (int, float) or not math.isfinite(speed) or speed <= 0:
        raise ValueError("Trajectory time scale must be finite and positive.")
    value = deepcopy(dict(trajectory))
    names, positions = value["joint_names"], value["positions"]
    times = value["time_from_start_ns"]
    if (
        not names
        or len(names) != len(set(names))
        or len(positions) < 2
        or len(positions) != len(times)
    ):
        raise ValueError("The checked trajectory is incomplete.")
    if any(type(stamp) is not int or stamp < 0 for stamp in times) or any(
        b <= a for a, b in zip(times, times[1:])
    ):
        raise ValueError("The checked trajectory has invalid time stamps.")
    times = [int(stamp * speed) for stamp in times]
    if times[-1] <= 0 or any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError("Trajectory scaling invalidated its timing.")
    value["time_from_start_ns"] = times
    for field, divisor in (("velocities", speed), ("accelerations", speed * speed)):
        rows = value[field]
        if len(rows) != len(positions) or any(len(row) != len(names) for row in rows):
            raise ValueError(f"The checked trajectory lacks complete {field}.")
        value[field] = [[float(number) / divisor for number in row] for row in rows]
    arrays = [
        np.asarray(value[field], dtype=float)
        for field in ("positions", "velocities", "accelerations")
    ]
    if any(
        array.shape != (len(times), len(names)) or not np.isfinite(array).all() for array in arrays
    ):
        raise ValueError("The checked trajectory contains invalid numerical values.")
    description = ET.fromstring(robot["model_parameters"]["robot_description"])
    joints = {joint.attrib["name"]: joint for joint in description.findall("joint")}
    parameters = robot["model_parameters"]
    for column, name in enumerate(names):
        if name not in robot["joint_state"]["names"] or name not in joints:
            raise ValueError("The trajectory uses a joint outside the measured robot.")
        joint = joints[name]
        limit = joint.find("limit")
        if limit is None:
            raise ValueError(f"Joint limits are unavailable for {name}.")
        prefix = "robot_description_planning.joint_limits." + name + "."
        velocity = float(limit.attrib["velocity"])
        if parameters.get(prefix + "has_velocity_limits") is True:
            velocity = min(velocity, float(parameters[prefix + "max_velocity"]))
        acceleration = parameters.get(prefix + "max_acceleration")
        if (
            parameters.get(prefix + "has_acceleration_limits") is not True
            or type(acceleration) not in (int, float)
            or not math.isfinite(acceleration)
            or acceleration <= 0
        ):
            raise ValueError(f"Acceleration limits are unavailable for {name}.")
        if not math.isfinite(velocity) or velocity <= 0:
            raise ValueError(f"Velocity limits are invalid for {name}.")
        q, dq, ddq = (array[:, column] for array in arrays)
        if joint.attrib["type"] in {"revolute", "prismatic"}:
            lower, upper = float(limit.attrib["lower"]), float(limit.attrib["upper"])
            if parameters.get(prefix + "has_position_limits") is True:
                lower = max(lower, float(parameters[prefix + "min_position"]))
                upper = min(upper, float(parameters[prefix + "max_position"]))
            if not math.isfinite(lower) or not math.isfinite(upper) or lower > upper:
                raise ValueError(f"Position limits are invalid for {name}.")
            if q.min() < lower or q.max() > upper:
                raise ValueError(f"Trajectory position exceeds limits for {name}.")
        dt = np.diff(times) / 1e9
        if max(np.max(np.abs(dq)), np.max(np.abs(np.diff(q) / dt))) > velocity + 1e-6:
            raise ValueError(f"Trajectory velocity exceeds limits for {name}.")
        if max(np.max(np.abs(ddq)), np.max(np.abs(np.diff(dq) / dt))) > acceleration + 1e-6:
            raise ValueError(f"Trajectory acceleration exceeds limits for {name}.")
    return value


class GazeboExecutionSession:
    """Own ROS clients for one simulation run; no shared agent is made executable."""

    def __init__(
        self, configuration: Mapping[str, Any], profile: Mapping[str, Any], stop: threading.Event
    ) -> None:
        """Retain selected-robot settings without constructing ROS clients."""
        self.configuration, self.profile, self.stop = (
            deepcopy(dict(configuration)),
            dict(profile),
            stop,
        )
        self.node = self.context = self.executor = self.goal = self.pending_goal = None
        self.joints: Any = None
        self._clients: dict[str, Any] = {}

    async def work(self, function: Any, *args: Any) -> Any:
        """Await worker completion even if the owning UI disconnects or cancels."""
        task = asyncio.create_task(asyncio.to_thread(function, *args))
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            self.stop.set()
            await task
            raise

    async def __aenter__(self) -> GazeboExecutionSession:
        """Create only simulation clients after runtime authorization."""
        try:
            await self.work(self._start)
        except BaseException:
            await self.work(self._close)
            raise
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Release this session's ROS context after any outstanding command ends."""
        await self.work(self._close)

    def _start(self) -> None:
        import rclpy
        import tf2_ros
        from rclpy.context import Context
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.node import Node
        from rclpy.parameter import Parameter
        from rclpy.action import ActionClient
        from moveit_msgs.action import ExecuteTrajectory
        from sensor_msgs.msg import JointState
        from trajectory_msgs.msg import JointTrajectory
        from linkattacher_msgs.srv import AttachLink, DetachLink

        if self.stop.is_set():
            raise RuntimeError("Execution stopped before ROS startup.")
        self.context = Context()
        rclpy.init(context=self.context)
        self.node = Node(
            "spec2primitives_execution_" + uuid.uuid4().hex[:8],
            context=self.context,
            parameter_overrides=[Parameter("use_sim_time", value=True)],
        )
        self.executor = SingleThreadedExecutor(context=self.context)
        self.executor.add_node(self.node)
        self.buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.buffer, self.node)
        self.node.create_subscription(
            JointState, self.profile["joint_states_topic"], self._joint, 50
        )
        self.gripper = self.node.create_publisher(
            JointTrajectory, self.configuration["gripper"]["topic"], 10
        )
        services = self.configuration["services"]
        self.arm = ActionClient(self.node, ExecuteTrajectory, services["execute_trajectory"])
        self.attach = self.node.create_client(AttachLink, services["attach"])
        self.detach = self.node.create_client(DetachLink, services["detach"])
        for client in (self.attach, self.detach):
            if not client.wait_for_service(timeout_sec=self.profile["service_timeout_sec"]):
                raise RuntimeError("Gazebo attachment services are unavailable.")
        if not self.arm.wait_for_server(timeout_sec=self.profile["service_timeout_sec"]):
            raise RuntimeError("Gazebo trajectory execution is unavailable.")

    def _joint(self, value: Any) -> None:
        self.joints = value

    def _await(self, future: Any, timeout: float) -> Any:
        deadline = time.monotonic() + timeout
        while not future.done() and time.monotonic() < deadline:
            self.executor.spin_once(timeout_sec=0.05)
        if not future.done():
            # A service timeout is an unknown outcome, not a negative response.
            raise TimeoutError("Gazebo command acknowledgment timed out.")
        return future.result()

    def _service(self, kind: Any, name: str, request: Any) -> Any:
        client = self._clients.get(name)
        if client is None:
            client = self.node.create_client(kind, name)
            self._clients[name] = client
        if not client.wait_for_service(timeout_sec=self.profile["service_timeout_sec"]):
            raise RuntimeError(f"Required Gazebo service is unavailable: {name}.")
        result = self._await(client.call_async(request), self.profile["service_timeout_sec"])
        if result is None or result.success is not True:
            raise RuntimeError("Gazebo could not establish the requested instance state.")
        return result

    async def entity_states(self, names: list[str]) -> dict[str, Any]:
        """Read current instance poses for the already accepted CAD candidates."""
        return await self.work(self._entity_states, names)

    def _entity_states(self, names: list[str]) -> dict[str, Any]:
        from gazebo_msgs.srv import GetEntityState, GetModelList

        listed = self._service(
            GetModelList, self.profile["model_list_service"], GetModelList.Request()
        )
        result = {}
        for name in names:
            if name not in listed.model_names:
                continue
            response = self._service(
                GetEntityState,
                self.configuration["services"].get("get_entity_state", "/get_entity_state"),
                GetEntityState.Request(name=name, reference_frame="world"),
            )
            pose = response.state.pose
            stamp = response.header.stamp.sec * 1000000000 + response.header.stamp.nanosec
            if (
                not 0
                <= self.node.get_clock().now().nanoseconds - stamp
                <= self.profile["state_max_age_sec"] * 1e9
            ):
                raise RuntimeError("Gazebo instance feedback is stale.")
            result[name] = {"pose": self._pose(pose), "stamp_ns": stamp}
        return result

    @staticmethod
    def _pose(pose: Any) -> dict[str, float]:
        return {
            "x": pose.position.x,
            "y": pose.position.y,
            "z": pose.position.z,
            "qx": pose.orientation.x,
            "qy": pose.orientation.y,
            "qz": pose.orientation.z,
            "qw": pose.orientation.w,
        }

    async def feedback(
        self,
        robot: Mapping[str, Any],
        expected_joints: Mapping[str, Any],
        expected_pose: Mapping[str, Any],
        tolerance: float,
    ) -> dict[str, Any]:
        """Require fresh joint and tool feedback at the expected motion prefix."""
        return await self.work(self._feedback, robot, expected_joints, expected_pose, tolerance)

    def _feedback(
        self,
        robot: Mapping[str, Any],
        expected: Mapping[str, Any],
        pose: Mapping[str, Any],
        tolerance: float,
    ) -> dict[str, Any]:
        from rclpy.time import Time
        from tf2_ros import TransformException

        deadline = time.monotonic() + self.profile["service_timeout_sec"]
        while time.monotonic() < deadline:
            self.executor.spin_once(timeout_sec=0.05)
            if self.joints is None:
                continue
            stamp = self.joints.header.stamp.sec * 1000000000 + self.joints.header.stamp.nanosec
            now = self.node.get_clock().now().nanoseconds
            if not 0 <= now - stamp <= self.profile["state_max_age_sec"] * 1e9:
                continue
            actual = dict(zip(self.joints.name, self.joints.position, strict=True))
            if any(
                name not in actual
                or not math.isfinite(actual[name])
                or abs(actual[name] - value)
                > (
                    self.configuration["gripper"]["position_tolerance"]
                    if name == self.configuration["gripper"]["joint"]
                    else tolerance
                )
                for name, value in zip(expected["names"], expected["positions"], strict=True)
            ):
                continue
            try:
                transform = self.buffer.lookup_transform(
                    robot["frame_id"], robot["ee_link"], Time()
                )
            except TransformException:
                continue
            tf_stamp = transform.header.stamp.sec * 1000000000 + transform.header.stamp.nanosec
            if not 0 <= now - tf_stamp <= self.profile["state_max_age_sec"] * 1e9:
                continue
            t, q = transform.transform.translation, transform.transform.rotation
            measured = {"x": t.x, "y": t.y, "z": t.z, "qx": q.x, "qy": q.y, "qz": q.z, "qw": q.w}
            difference = np.linalg.inv(pose_matrix(pose)) @ pose_matrix(measured)
            if (
                np.linalg.norm(difference[:3, 3]) <= robot["position_tolerance_m"]
                and Rotation.from_matrix(difference[:3, :3]).magnitude()
                <= self.profile["fk_orientation_tolerance_rad"]
            ):
                return {"joint_positions": actual, "ee_pose": measured, "measured_at_ros_ns": now}
        raise RuntimeError("Fresh robot feedback does not match the validated program prefix.")

    async def move(self, trajectory: Mapping[str, Any]) -> dict[str, Any]:
        """Execute precisely the checked timed trajectory and retain cancellation."""
        return await self.work(self._move, trajectory)

    def _move(self, trajectory: Mapping[str, Any]) -> dict[str, Any]:
        from moveit_msgs.action import ExecuteTrajectory
        from trajectory_msgs.msg import JointTrajectoryPoint

        goal = ExecuteTrajectory.Goal()
        goal.trajectory.joint_trajectory.joint_names = list(trajectory["joint_names"])
        for index, stamp in enumerate(trajectory["time_from_start_ns"]):
            point = JointTrajectoryPoint(
                positions=trajectory["positions"][index],
                velocities=trajectory["velocities"][index],
                accelerations=trajectory["accelerations"][index],
            )
            point.time_from_start.sec, point.time_from_start.nanosec = divmod(stamp, 1000000000)
            goal.trajectory.joint_trajectory.points.append(point)
        if self.stop.is_set():
            raise RuntimeError("Execution stopped before trajectory dispatch.")
        self.pending_goal = self.arm.send_goal_async(goal)
        self.goal = self._await(self.pending_goal, self.profile["service_timeout_sec"])
        self.pending_goal = None
        if self.goal is None or not self.goal.accepted:
            raise RuntimeError("Gazebo rejected the trajectory goal.")
        future = self.goal.get_result_async()
        deadline = (
            time.monotonic()
            + trajectory["time_from_start_ns"][-1] / 1e9
            + self.profile["trajectory_timeout_pad_sec"]
        )
        cancel_sent = False
        while not future.done():
            if (self.stop.is_set() or time.monotonic() >= deadline) and not cancel_sent:
                self._await(self.goal.cancel_goal_async(), self.profile["stop_timeout_sec"])
                cancel_sent = True
                deadline = time.monotonic() + self.profile["stop_timeout_sec"]
            if cancel_sent and time.monotonic() >= deadline:
                raise TimeoutError(
                    "Trajectory cancellation has no terminal acknowledgment; motion state is unknown."
                )
            self.executor.spin_once(timeout_sec=0.05)
        response = future.result()
        self.goal = None
        if cancel_sent or response.status != 4 or response.result.error_code.val != 1:
            raise RuntimeError(
                "Trajectory stopped or failed; no subsequent primitive was dispatched."
            )
        return {"success": True, "error_code": response.result.error_code.val}

    async def gripper_command(self, position: float) -> dict[str, Any]:
        """Command the configured gripper and require its measured target position."""
        return await self.work(self._gripper_command, position)

    def _gripper_command(self, position: float) -> dict[str, Any]:
        from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

        if self.stop.is_set():
            raise RuntimeError("Execution stopped before gripper dispatch.")
        config = self.configuration["gripper"]
        duration = float(config["move_time_sec"])
        endpoints = (float(config["open"]), float(config["close"]))
        if (
            type(position) not in (int, float)
            or not all(math.isfinite(value) for value in (*endpoints, position, duration))
            or duration <= 0
            or not min(endpoints) <= position <= max(endpoints)
        ):
            raise ValueError("Invalid configured gripper command.")
        deadline = time.monotonic() + self.profile["service_timeout_sec"]
        while self.gripper.get_subscription_count() == 0 and time.monotonic() < deadline:
            self.executor.spin_once(timeout_sec=0.05)
        if self.gripper.get_subscription_count() == 0:
            raise RuntimeError("Gazebo gripper controller is unavailable.")
        point = JointTrajectoryPoint(positions=[float(position)])
        point.time_from_start.sec, point.time_from_start.nanosec = divmod(
            int(duration * 1e9), 1000000000
        )
        sent = self.node.get_clock().now().nanoseconds
        self.gripper.publish(JointTrajectory(joint_names=[config["joint"]], points=[point]))
        deadline = time.monotonic() + duration + float(config["feedback_timeout_pad_sec"])
        while time.monotonic() < deadline:
            self.executor.spin_once(timeout_sec=0.05)
            if self.joints is None or config["joint"] not in self.joints.name:
                continue
            stamp = self.joints.header.stamp.sec * 1000000000 + self.joints.header.stamp.nanosec
            actual = self.joints.position[self.joints.name.index(config["joint"])]
            if (
                stamp >= sent
                and math.isfinite(actual)
                and abs(actual - position) <= float(config["position_tolerance"])
            ):
                return {"success": True, "position": actual, "stamp_ns": stamp}
        raise TimeoutError("Gripper feedback did not acknowledge the commanded position.")

    async def attachment(self, binding: Mapping[str, Any], attach: bool) -> dict[str, Any]:
        """Issue exactly one attach/detach request and require its acknowledgment."""
        return await self.work(self._attachment, binding, attach)

    def _attachment(self, binding: Mapping[str, Any], attach: bool) -> dict[str, Any]:
        from linkattacher_msgs.srv import AttachLink, DetachLink

        if self.stop.is_set():
            raise RuntimeError("Execution stopped before attachment dispatch.")
        config = self.configuration["attach"]
        request = (AttachLink if attach else DetachLink).Request()
        request.model1_name, request.link1_name = (
            config["robot_model_name"],
            config["primary_attach_link"],
        )
        request.model2_name, request.link2_name = binding["model_name"], binding["link"]
        response = self._await(
            (self.attach if attach else self.detach).call_async(request),
            self.profile["service_timeout_sec"],
        )
        if response is None or response.success is not True:
            raise RuntimeError("Gazebo attachment command failed.")
        return {"success": True, "attached": attach, "message": response.message}

    def _close(self) -> None:
        try:
            # A late goal acknowledgment still belongs to this run. Obtain its
            # handle before teardown so cancellation is not silently abandoned.
            if self.pending_goal is not None:
                self.stop.set()
                self.goal = self._await(self.pending_goal, self.profile["stop_timeout_sec"])
                self.pending_goal = None
            if self.goal is not None and self.goal.accepted:
                self.stop.set()
                self._await(self.goal.cancel_goal_async(), self.profile["stop_timeout_sec"])
                self._await(self.goal.get_result_async(), self.profile["stop_timeout_sec"])
        finally:
            self.goal = None
            if self.executor is not None:
                self.executor.shutdown()
            if self.node is not None:
                self.node.destroy_node()
            if self.context is not None:
                self.context.try_shutdown()
