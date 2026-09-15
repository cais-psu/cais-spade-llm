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
from collections import deque
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from .robot_validation_context import matrix_pose, pose_matrix
from ..agents.ra.validation_scope import is_observed_scope, VALIDATION_SCOPE


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
    placement = profile.get("placement_attachment", {})
    if set(placement) != {"model_name", "link"} or not all(
        isinstance(value, str) and value.strip() for value in placement.values()
    ):
        raise ValueError("Gazebo placement requires a configured board model and link.")
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
    observed_bounds = is_observed_scope(validation_scope)
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
        peak_velocity = max(np.max(np.abs(dq)), np.max(np.abs(np.diff(q) / dt)))
        if peak_velocity > velocity + 1e-6:
            raise ValueError(
                f"Trajectory velocity exceeds limits for {name}. "
                f"Maximum={peak_velocity:.9g}, limit={velocity:.9g}."
            )
        peak_acceleration = max(np.max(np.abs(ddq)), np.max(np.abs(np.diff(dq) / dt)))
        if peak_acceleration > acceleration + 1e-6:
            raise ValueError(
                f"Trajectory acceleration exceeds limits for {name}. "
                f"Maximum={peak_acceleration:.9g}, limit={acceleration:.9g}."
            )
    return value


class GripperCommandError(RuntimeError):
    """Retain command evidence, including a late terminal acknowledgment at teardown."""

    def __init__(self, message: str, diagnostics: dict[str, Any]) -> None:
        """Attach the session-owned diagnostic consumed after transport cleanup."""
        super().__init__(message)
        self.diagnostics = diagnostics

    @property
    def outcome_known(self) -> bool:
        """Return whether rejection or a controller terminal result was confirmed."""
        return self.diagnostics.get("termination_confirmed") is True


def gripper_action_name(topic: str) -> str:
    """Resolve the standard action on the explicitly configured trajectory controller."""
    suffix = "/joint_trajectory"
    if not isinstance(topic, str) or not topic.endswith(suffix) or not topic[:-len(suffix)]:
        raise ValueError("Gripper topic must identify a controller's /joint_trajectory endpoint.")
    return topic[:-len(suffix)] + "/follow_joint_trajectory"


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
        self.goal_result = self.cancel_request = None
        self.gripper_diagnostics: dict[str, Any] | None = None
        self.joints: Any = None
        self._clients: dict[str, Any] = {}
        self._feedback_condition = threading.Condition()
        self._joint_samples: deque[Any] = deque(maxlen=4096)
        self._joint_samples_dropped = 0
        self._spin_stop = threading.Event()
        self._spin_thread: threading.Thread | None = None
        self._spin_error: RuntimeError | None = None

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
        try:
            await self.work(self._close)
        except (TimeoutError, RuntimeError) as exc:
            if len(args) > 1 and isinstance(args[1], GripperCommandError):
                args[1].diagnostics["cleanup_error"] = str(exc)
            else:
                raise

    def _start(self) -> None:
        import rclpy
        from rclpy.context import Context
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.node import Node
        from rclpy.parameter import Parameter
        from rclpy.action import ActionClient
        from moveit_msgs.action import ExecuteTrajectory
        from control_msgs.action import FollowJointTrajectory
        from sensor_msgs.msg import JointState
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
        self.node.create_subscription(
            JointState, self.profile["joint_states_topic"], self._joint, 50
        )
        self.gripper = ActionClient(
            self.node, FollowJointTrajectory,
            gripper_action_name(self.configuration["gripper"]["topic"]),
        )
        services = self.configuration["services"]
        self.arm = ActionClient(self.node, ExecuteTrajectory, services["execute_trajectory"])
        self.attach = self.node.create_client(AttachLink, services["attach"])
        self.detach = self.node.create_client(DetachLink, services["detach"])
        self._spin_thread = threading.Thread(target=self._spin, name=self.node.get_name(), daemon=True)
        self._spin_thread.start()
        for client in (self.attach, self.detach):
            if not client.wait_for_service(timeout_sec=self.profile["service_timeout_sec"]):
                raise RuntimeError("Gazebo attachment services are unavailable.")
        if not self.arm.wait_for_server(timeout_sec=self.profile["service_timeout_sec"]):
            raise RuntimeError("Gazebo trajectory execution is unavailable.")
        if not self.gripper.wait_for_server(timeout_sec=self.profile["service_timeout_sec"]):
            raise RuntimeError("Gazebo gripper action is unavailable: " +
                               gripper_action_name(self.configuration["gripper"]["topic"]))

    def _joint(self, value: Any) -> None:
        with self._feedback_condition:
            self.joints = value
            if len(self._joint_samples) == self._joint_samples.maxlen:
                self._joint_samples_dropped += 1
            self._joint_samples.append(value)
            self._feedback_condition.notify_all()

    def _spin(self) -> None:
        from rclpy.executors import ExternalShutdownException, ShutdownException

        previous_clock = 0
        try:
            while not self._spin_stop.is_set():
                self.executor.spin_once(timeout_sec=0.05)
                now = self.node.get_clock().now().nanoseconds
                if now < previous_clock:
                    self._spin_error = RuntimeError("Gazebo clock moved backwards during execution.")
                previous_clock = now
                with self._feedback_condition:
                    self._feedback_condition.notify_all()
        except (ExternalShutdownException, ShutdownException, RuntimeError) as exc:
            if not self._spin_stop.is_set():
                self._spin_error = RuntimeError(f"Gazebo feedback executor stopped: {exc}")
        finally:
            with self._feedback_condition:
                self._feedback_condition.notify_all()

    def _wait_for_update(self, *, timeout_sec: float) -> None:
        # Only the session thread spins this executor. Command waits must not
        # compete with it or stop /clock and joint delivery between primitives.
        with self._feedback_condition:
            self._feedback_condition.wait(timeout=timeout_sec)
        if self._spin_error is not None:
            raise self._spin_error

    @staticmethod
    def _joint_stamp(value: Any) -> int:
        return value.header.stamp.sec * 1_000_000_000 + value.header.stamp.nanosec

    def _await(self, future: Any, timeout: float) -> Any:
        deadline = time.monotonic() + timeout
        while not future.done() and time.monotonic() < deadline:
            self._wait_for_update(timeout_sec=0.05)
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
        """Read instance poses after synchronizing their stamps with the local ROS clock."""
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
            if stamp <= 0:
                raise RuntimeError(f"Gazebo instance feedback has no valid timestamp for {name!r}.")
            # Gazebo stamps service replies at world time, which may be ahead of
            # this node's latest /clock update. Wait for that update without
            # repeating the request or relaxing the feedback age limit.
            now = self.node.get_clock().now().nanoseconds
            deadline = time.monotonic() + self.profile["service_timeout_sec"]
            while now < stamp and not self.stop.is_set():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                previous = now
                self._wait_for_update(timeout_sec=min(0.05, remaining))
                now = self.node.get_clock().now().nanoseconds
                if now < previous:
                    raise RuntimeError("Gazebo simulation clock moved backwards while checking instance feedback.")
            if self.stop.is_set():
                raise RuntimeError("Execution stopped while waiting for Gazebo instance feedback.")
            if now < stamp:
                raise RuntimeError(
                    f"Gazebo simulation clock did not catch up to feedback for {name!r} "
                    f"within {self.profile['service_timeout_sec']:g} s "
                    f"(clock={now / 1e9:.6f} s, feedback={stamp / 1e9:.6f} s)."
                )
            age = (now - stamp) / 1e9
            if age > self.profile["state_max_age_sec"]:
                raise RuntimeError(
                    f"Gazebo instance feedback is stale for {name!r} "
                    f"(age={age:.6f} s, limit={self.profile['state_max_age_sec']:g} s)."
                )
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
        sent = previous_clock = self.node.get_clock().now().nanoseconds
        if sent <= 0:
            raise RuntimeError("Gazebo clock is unavailable before trajectory dispatch.")
        self.gripper_diagnostics = None
        self.cancel_request = self.goal_result = None
        self.pending_goal = self.arm.send_goal_async(goal)
        self.goal = self._await(self.pending_goal, self.profile["service_timeout_sec"])
        self.pending_goal = None
        if self.goal is None or not self.goal.accepted:
            raise RuntimeError("Gazebo rejected the trajectory goal.")
        future = self.goal_result = self.goal.get_result_async()
        # Controller durations follow /clock even when physics runs slowly.
        # A stopped clock and cancellation still need bounded wall-time waits.
        deadline_ns = sent + trajectory["time_from_start_ns"][-1] + int(
            self.profile["trajectory_timeout_pad_sec"] * 1e9
        )
        last_clock_advance = time.monotonic()
        cancel_reason = None
        cancel_deadline = None
        while not future.done():
            now = self.node.get_clock().now().nanoseconds
            wall = time.monotonic()
            if now < previous_clock:
                raise RuntimeError("Gazebo clock moved backwards during trajectory execution.")
            if now > previous_clock:
                last_clock_advance = wall
            previous_clock = now
            if cancel_reason is None:
                if self.stop.is_set():
                    cancel_reason = "Execution stopped."
                elif now >= deadline_ns:
                    cancel_reason = "Trajectory completion timed out in Gazebo simulation time."
                elif wall - last_clock_advance >= self.profile["service_timeout_sec"]:
                    cancel_reason = "Gazebo clock stopped advancing during trajectory execution."
                if cancel_reason is not None:
                    self._request_cancel()
                    cancel_deadline = time.monotonic() + self.profile["stop_timeout_sec"]
            if cancel_deadline is not None and not future.done() and time.monotonic() >= cancel_deadline:
                raise TimeoutError(
                    f"{cancel_reason} Trajectory cancellation has no terminal acknowledgment; "
                    "motion state is unknown."
                )
            if not future.done():
                self._wait_for_update(timeout_sec=0.05)
        response = future.result()
        self.goal = None
        # A timeout cancellation can race with an already successful controller
        # result. The terminal acknowledgment establishes the motion outcome.
        if self.stop.is_set() or response.status != 4 or response.result.error_code.val != 1:
            raise RuntimeError(
                "Trajectory stopped or failed; "
                f"controller status {response.status}, MoveIt error_code {response.result.error_code.val}. "
                f"{cancel_reason or ''} No subsequent primitive was dispatched."
            )
        return {"success": True, "error_code": response.result.error_code.val}

    async def gripper_command(self, position: float) -> dict[str, Any]:
        """Command the configured gripper and require its measured target position."""
        return await self.work(self._gripper_command, position)

    def _gripper_command(self, position: float) -> dict[str, Any]:
        if self.stop.is_set():
            raise RuntimeError("Execution stopped before gripper dispatch.")
        config = self.configuration["gripper"]
        duration = float(config["move_time_sec"])
        padding = float(config.get("feedback_timeout_pad_sec", 0.5))
        settle = float(config.get("settle_sec", 0.0))
        tolerance = float(config.get("position_tolerance", 0.01))
        endpoints = (float(config["open"]), float(config["close"]))
        if (
            type(position) not in (int, float)
            or not all(math.isfinite(value) for value in (*endpoints, position, duration, padding, settle, tolerance))
            or duration <= 0 or padding < 0 or settle < 0 or tolerance <= 0
            or not min(endpoints) <= position <= max(endpoints)
        ):
            raise ValueError("Invalid configured gripper command.")
        from control_msgs.action import FollowJointTrajectory
        from control_msgs.msg import JointTolerance
        from trajectory_msgs.msg import JointTrajectoryPoint

        goal = FollowJointTrajectory.Goal()
        point = JointTrajectoryPoint(positions=[float(position)])
        point.time_from_start.sec, point.time_from_start.nanosec = divmod(
            int(duration * 1e9), 1000000000
        )
        goal.trajectory.joint_names, goal.trajectory.points = [config["joint"]], [point]
        goal.goal_tolerance = [JointTolerance(name=config["joint"], position=tolerance)]
        goal.goal_time_tolerance.sec, goal.goal_time_tolerance.nanosec = divmod(int(padding * 1e9), 1000000000)
        sent = previous_clock = self.node.get_clock().now().nanoseconds
        started = time.monotonic()
        report = self.gripper_diagnostics = {
            "joint": config["joint"], "target": position, "accepted": None,
            "termination_confirmed": False, "terminal_status": None, "error_code": None,
            "cancel_requested": False, "cancel_acknowledged": False,
            "position": None, "stamp_ns": None, "sent_ros_ns": sent,
            "elapsed_simulation_sec": 0.0, "elapsed_wall_sec": 0.0, "success": False,
            "clock_ros_ns": sent, "feedback_age_sec": None,
        }
        with self._feedback_condition:
            self._joint_samples.clear()
            dropped = self._joint_samples_dropped
        if self.joints is not None and config["joint"] in self.joints.name:
            index = self.joints.name.index(config["joint"])
            if index < len(self.joints.position) and math.isfinite(self.joints.position[index]):
                report["position"] = float(self.joints.position[index])
                report["stamp_ns"] = self.joints.header.stamp.sec * 10**9 + self.joints.header.stamp.nanosec
        self.cancel_request = self.goal_result = None
        stable_since = None
        last_feedback_stamp = 0
        feedback_reason = "No fresh gripper joint feedback."
        try:
            if sent <= 0:
                report["termination_confirmed"] = True
                raise RuntimeError("Gazebo clock is unavailable before gripper dispatch.")
            self.pending_goal = self.gripper.send_goal_async(goal)
            self.goal = self._await(self.pending_goal, self.profile["service_timeout_sec"])
            self.pending_goal = None
            report["accepted"] = bool(self.goal is not None and self.goal.accepted)
            if not report["accepted"]:
                report["termination_confirmed"] = True
                self.goal = None
                raise RuntimeError("Gazebo rejected the gripper goal.")
            self.goal_result = self.goal.get_result_async()
            deadline_ns = sent + int((duration + padding + self.profile["trajectory_timeout_pad_sec"]) * 1e9)
            last_clock_advance = time.monotonic()
            while True:
                self._wait_for_update(timeout_sec=0.02)
                now = self.node.get_clock().now().nanoseconds
                wall = time.monotonic()
                report["clock_ros_ns"] = now
                report["elapsed_wall_sec"] = wall - started
                report["elapsed_simulation_sec"] = (now - sent) / 1e9
                if now < previous_clock:
                    raise RuntimeError("Gazebo clock moved backwards during gripper execution.")
                if now > previous_clock:
                    last_clock_advance = wall
                previous_clock = now
                if self.goal_result.done():
                    response = self._gripper_terminal()
                    if response.status != 4 or response.result.error_code != 0:
                        raise RuntimeError(
                            f"Gripper controller terminated with status {response.status}, "
                            f"error {response.result.error_code}: {response.result.error_string}"
                        )
                if self.stop.is_set():
                    raise RuntimeError("Gripper execution stopped; no subsequent primitive was dispatched.")
                if now >= deadline_ns:
                    raise TimeoutError("Gripper completion timed out in Gazebo simulation time. " + feedback_reason)
                if wall - last_clock_advance >= self.profile["service_timeout_sec"]:
                    raise TimeoutError("Gazebo clock stopped advancing during gripper execution. " + feedback_reason)
                with self._feedback_condition:
                    if self._joint_samples_dropped != dropped:
                        stable_since = None
                        dropped = self._joint_samples_dropped
                    pending = self._joint_samples[0] if self._joint_samples else None
                    if pending is not None and self._joint_stamp(pending) <= now:
                        sample = self._joint_samples.popleft()
                    else:
                        sample = None
                if sample is None:
                    if pending is not None:
                        report["stamp_ns"] = self._joint_stamp(pending)
                        report["feedback_age_sec"] = (now - report["stamp_ns"]) / 1e9
                        feedback_reason = "Waiting for Gazebo clock to reach gripper joint feedback."
                    if last_feedback_stamp and (now - last_feedback_stamp) / 1e9 > self.profile["state_max_age_sec"]:
                        stable_since = None
                        feedback_reason = "Gripper joint feedback is stale."
                    continue
                if config["joint"] not in sample.name:
                    stable_since = None
                    feedback_reason = "No measured gripper joint in feedback."
                    continue
                if sample.name.count(config["joint"]) != 1:
                    raise RuntimeError("Gripper joint feedback is ambiguous.")
                index = sample.name.index(config["joint"])
                if index >= len(sample.position):
                    raise RuntimeError("Gripper joint feedback has no measured position.")
                actual = float(sample.position[index])
                stamp = self._joint_stamp(sample)
                if stamp < last_feedback_stamp:
                    raise RuntimeError("Gripper feedback timestamp moved backwards.")
                if (stamp - last_feedback_stamp) / 1e9 > self.profile["state_max_age_sec"]:
                    stable_since = None
                distinct = stamp > last_feedback_stamp
                last_feedback_stamp = stamp
                report.update(position=actual if math.isfinite(actual) else None, stamp_ns=stamp,
                              feedback_age_sec=(now - stamp) / 1e9)
                if not math.isfinite(actual):
                    raise RuntimeError("Gripper joint feedback is non-finite.")
                fresh = stamp >= sent and stamp > 0 and 0 <= now - stamp <= self.profile["state_max_age_sec"] * 1e9
                if not fresh or abs(actual - position) > tolerance:
                    stable_since = None
                    feedback_reason = ("Gripper joint feedback is stale or its clock has not synchronized."
                                       if not fresh else f"Target {position:g}; measured {actual:g}.")
                    continue
                if not distinct:
                    continue
                # Advance settling only with measured simulation timestamps, never
                # by repeatedly accepting one cached sample while Gazebo is paused.
                stable_since = stamp if stable_since is None else stable_since
                if (report["termination_confirmed"] and stamp > stable_since
                        and (stamp - stable_since) / 1e9 >= settle):
                    report["success"] = True
                    return deepcopy(report)
        except (OSError, RuntimeError, ValueError, TypeError) as exc:
            report["message"] = str(exc)
            try:
                self._finish_outstanding_goal()
            except (OSError, RuntimeError, ValueError, TypeError) as cleanup:
                report["cleanup_error"] = str(cleanup)
            report["elapsed_wall_sec"] = time.monotonic() - started
            report["clock_ros_ns"] = self.node.get_clock().now().nanoseconds
            report["elapsed_simulation_sec"] = (report["clock_ros_ns"] - sent) / 1e9
            if report["stamp_ns"] is not None:
                report["feedback_age_sec"] = (report["clock_ros_ns"] - report["stamp_ns"]) / 1e9
            raise GripperCommandError(str(exc), report) from exc

    def _gripper_terminal(self) -> Any:
        response = self.goal_result.result()
        if response is None or response.status not in {4, 5, 6}:
            raise RuntimeError("Gripper controller returned no valid terminal result.")
        self.gripper_diagnostics.update(
            termination_confirmed=True, terminal_status=response.status,
            error_code=response.result.error_code, error_string=response.result.error_string,
        )
        self.goal = None
        return response

    def _request_cancel(self) -> None:
        if self.cancel_request is None:
            self.cancel_request = self.goal.cancel_goal_async()
            if self.gripper_diagnostics is not None:
                self.gripper_diagnostics["cancel_requested"] = True
        response = self._await(self.cancel_request, self.profile["stop_timeout_sec"])
        if self.gripper_diagnostics is not None:
            self.gripper_diagnostics["cancel_acknowledged"] = bool(response.goals_canceling)

    def _finish_outstanding_goal(self) -> None:
        if self.pending_goal is not None:
            self.goal = self._await(self.pending_goal, self.profile["stop_timeout_sec"])
            self.pending_goal = None
            if self.gripper_diagnostics is not None:
                self.gripper_diagnostics["accepted"] = bool(self.goal is not None and self.goal.accepted)
                if not self.gripper_diagnostics["accepted"]:
                    self.gripper_diagnostics["termination_confirmed"] = True
        if self.goal is None or not self.goal.accepted:
            return
        if self.goal_result is None:
            self.goal_result = self.goal.get_result_async()
        if not self.goal_result.done():
            self._request_cancel()
            self._await(self.goal_result, self.profile["stop_timeout_sec"])
        if self.gripper_diagnostics is not None:
            self._gripper_terminal()
        else:
            self.goal = None

    async def attachment(
        self, binding: Mapping[str, Any], attach: bool, *, parent: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        """Attach/detach the bound part to the gripper or an explicit execution fixture.

        Args:
            binding: Instance selected only after the program has been validated.
            attach: Whether to attach or detach that instance.
            parent: Configured placement fixture; None selects the robot gripper.

        Returns:
            The acknowledged command, including the placement parent when supplied.
        """
        return await self.work(self._attachment, binding, attach, parent)

    def _attachment(
        self, binding: Mapping[str, Any], attach: bool, parent: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        from linkattacher_msgs.srv import AttachLink, DetachLink

        if self.stop.is_set():
            raise RuntimeError("Execution stopped before attachment dispatch.")
        config = self.configuration["attach"]
        request = (AttachLink if attach else DetachLink).Request()
        request.model1_name, request.link1_name = (
            (parent["model_name"], parent["link"]) if parent is not None else
            (config["robot_model_name"], config["primary_attach_link"])
        )
        request.model2_name, request.link2_name = binding["model_name"], binding["link"]
        response = self._await(
            (self.attach if attach else self.detach).call_async(request),
            self.profile["service_timeout_sec"],
        )
        if response is None or response.success is not True:
            raise RuntimeError("Gazebo attachment command failed.")
        return {"success": True, "attached": attach, "message": response.message,
                **({"parent": dict(parent)} if parent is not None else {})}

    def _close(self) -> None:
        try:
            # A late goal acknowledgment still belongs to this run. Obtain its
            # handle before teardown so cancellation is not silently abandoned.
            if self.pending_goal is not None or (self.goal is not None and self.goal.accepted):
                self._finish_outstanding_goal()
        finally:
            self.goal = None
            self._spin_stop.set()
            if self.executor is not None:
                if self.executor.shutdown(timeout_sec=self.profile["stop_timeout_sec"]) is False:
                    raise RuntimeError("Gazebo feedback executor shutdown timed out.")
            if self._spin_thread is not None:
                self._spin_thread.join(timeout=self.profile["stop_timeout_sec"])
                if self._spin_thread.is_alive():
                    raise RuntimeError("Gazebo feedback executor did not stop; transport cleanup is incomplete.")
            if self.node is not None:
                self.node.destroy_node()
            if self.context is not None:
                self.context.try_shutdown()
