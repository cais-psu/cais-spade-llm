from __future__ import annotations

"""Narrow runtime adapter for the Spec2Primitives no-hardware dual Gazebo UI."""

import math
import threading
import time
import uuid
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any, Protocol

DUAL_GAZEBO_NAME = "gazebo_dual_spec2primitives"
_HARDWARE_STACK_NAMES = ("xarm6", "ur5e", "dual robots")
_HARDWARE_BLOCKED_REASON = "Blocked: hardware stack is running. Stop hardware first."


class DualGazeboRuntime(Protocol):
    """Runtime operations used by the isolated Spec2Primitives Gazebo adapter."""

    execution_mode: str
    robot_env: str
    system_running: bool

    def ros2_all_statuses(self) -> dict[str, str]:
        """Return tracked ROS2 process statuses."""
        ...

    def hardware_stack_status(self, robot: str) -> dict[str, object]:
        """Return one hardware stack's current status."""
        ...

    def ros2_start(self, name: str) -> str | None:
        """Start one tracked ROS2 process and return an error when blocked."""
        ...

    def ros2_stop(self, name: str) -> None:
        """Stop one tracked ROS2 process."""
        ...

    def simulation_start_ready(self, force: bool = False) -> tuple[bool, str]:
        """Probe the application's configured core simulation services."""
        ...


@dataclass(frozen=True)
class DualGazeboStatus:
    """Current dual Gazebo state and any hardware interlock message."""

    state: str
    blocked_reason: str | None = None


def read_dual_gazebo_status(runtime: DualGazeboRuntime) -> DualGazeboStatus:
    """Read fresh dual Gazebo and hardware-stack state from the runtime."""
    statuses = runtime.ros2_all_statuses()
    state = str(statuses.get(DUAL_GAZEBO_NAME) or "stopped")
    if state == "running":
        return DualGazeboStatus(state=state)

    for robot in _HARDWARE_STACK_NAMES:
        hardware_status = runtime.hardware_stack_status(robot)
        if hardware_status.get("overall") == "running":
            return DualGazeboStatus(
                state=state,
                blocked_reason=_HARDWARE_BLOCKED_REASON,
            )
    return DualGazeboStatus(state=state)


def start_dual_gazebo(runtime: DualGazeboRuntime) -> str | None:
    """Start `gazebo_dual_spec2primitives` after a fresh state and hardware check."""
    status = read_dual_gazebo_status(runtime)
    if status.blocked_reason:
        return status.blocked_reason
    if status.state == "running":
        return "Dual Robots (xArm6 + UR5e) is already running."
    if status.state != "stopped":
        return f"Dual Robots (xArm6 + UR5e) cannot start while status is {status.state}."
    return runtime.ros2_start(DUAL_GAZEBO_NAME)


def stop_dual_gazebo(runtime: DualGazeboRuntime) -> None:
    """Stop only the tracked `gazebo_dual_spec2primitives` process."""
    runtime.ros2_stop(DUAL_GAZEBO_NAME)


def assert_reset_interlocks(runtime: DualGazeboRuntime) -> None:
    """Require exclusive simulation authority before and after lifecycle changes."""
    if (runtime.execution_mode != "simulation" or runtime.robot_env != "gazebo"
            or runtime.system_running):
        raise RuntimeError("Reset requires exclusive Spec2Primitives simulation mode.")
    for robot in _HARDWARE_STACK_NAMES:
        if runtime.hardware_stack_status(robot).get("overall") not in {"stopped", "idle"}:
            raise RuntimeError("Reset blocked: hardware stack is active or its state is unavailable.")


class GazeboResetProbe:
    """Observe a process replacement in the configured ROS domain without commanding robots."""

    def __init__(self, profile: dict[str, Any], stop: threading.Event) -> None:
        """Retain bounded discovery settings and the reset cancellation signal."""
        self.profile, self.stop = profile, stop
        self.context = self.node = self.executor = None
        self.clock_ns = 0
        self.clock_gid = None
        self.joints: dict[str, tuple[Any, str, float]] = {}
        self.description: str | None = None

    def __enter__(self) -> GazeboResetProbe:
        """Subscribe with an independent context that survives the old simulator stopping."""
        import rclpy
        from rclpy.context import Context
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.node import Node

        self.context = Context()
        rclpy.init(context=self.context)
        try:
            self.node = Node("spec2primitives_reset_" + uuid.uuid4().hex[:8], context=self.context)
            self.executor = SingleThreadedExecutor(context=self.context)
            self.executor.add_node(self.node)
        except (ImportError, RuntimeError, ValueError):
            self.__exit__()
            raise
        return self

    def _subscribe(self) -> None:
        from rclpy.qos import QoSProfile, DurabilityPolicy, qos_profile_sensor_data
        from rosgraph_msgs.msg import Clock
        from sensor_msgs.msg import JointState
        from std_msgs.msg import String

        # Humble's executor does not forward MessageInfo. Subscribe only after
        # old endpoints are gone, so old queued samples cannot establish a baseline.
        self.node.create_subscription(Clock, "/clock", self._clock, qos_profile_sensor_data)
        for topic in self.profile["joint_states_topics"]:
            def receive(message: Any, topic: str = topic) -> None:
                publishers = self.node.get_publishers_info_by_topic(topic)
                if len(publishers) == 1:
                    self.joints[topic] = (message, bytes(publishers[0].endpoint_gid).hex(), time.monotonic())
            self.node.create_subscription(JointState, topic, receive, qos_profile_sensor_data)
        self.node.create_subscription(
            String, "/robot_description", self._description,
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL),
        )

    def __exit__(self, *args: Any) -> None:
        """Release only this read-only probe's ROS resources."""
        if self.executor is not None:
            self.executor.shutdown()
        if self.node is not None:
            self.node.destroy_node()
        if self.context is not None and self.context.ok():
            self.context.shutdown()

    def _clock(self, message: Any) -> None:
        publishers = self.node.get_publishers_info_by_topic("/clock")
        if len(publishers) == 1:
            self.clock_ns = message.clock.sec * 10**9 + message.clock.nanosec
            self.clock_gid = bytes(publishers[0].endpoint_gid).hex()

    def _description(self, message: Any) -> None:
        self.description = message.data

    def _endpoints(self) -> dict[str, Any]:
        topics = ["/clock", *self.profile["joint_states_topics"]]
        return {
            "ros_domain_id": self.context.get_domain_id(),
            "publishers": {topic: sorted(bytes(info.endpoint_gid).hex()
                                          for info in self.node.get_publishers_info_by_topic(topic))
                           for topic in topics},
            "services": sorted(name for name, types in self.node.get_service_names_and_types()
                               if any(kind.startswith(("gazebo_msgs/", "moveit_msgs/",
                                      "controller_manager_msgs/", "control_msgs/", "linkattacher_msgs/"))
                                      for kind in types)),
        }

    def _spin(self) -> None:
        if self.stop.is_set():
            raise RuntimeError("Gazebo reset interrupted; a verified reset is still required.")
        self.executor.spin_once(timeout_sec=0.05)

    def snapshot(self) -> dict[str, Any]:
        """Record old discovery identities before any lifecycle operation."""
        deadline = time.monotonic() + self.profile["service_timeout_sec"]
        while time.monotonic() < deadline:
            self._spin()
        return self._endpoints()

    def wait_stopped(self, runtime: DualGazeboRuntime) -> dict[str, Any]:
        """Require a tracked stop and a sustained absence of old ROS command endpoints."""
        deadline = time.monotonic() + self.profile["stop_timeout_sec"]
        quiet_since = None
        while time.monotonic() < deadline:
            self._spin()
            assert_reset_interlocks(runtime)
            endpoints = self._endpoints()
            stopped = read_dual_gazebo_status(runtime).state == "stopped"
            if stopped and not endpoints["services"] and not any(endpoints["publishers"].values()):
                quiet_since = time.monotonic() if quiet_since is None else quiet_since
                if time.monotonic() - quiet_since >= 1.0:
                    self.joints.clear()
                    self.description, self.clock_gid, self.clock_ns = None, None, 0
                    self._subscribe()
                    return {"process_status": "stopped", "endpoints": endpoints}
            else:
                quiet_since = None
        raise TimeoutError("Gazebo stopped status and disappearance of old ROS endpoints were not confirmed.")

    def wait_ready(self, old: dict[str, Any], required_joints: list[str]) -> dict[str, Any]:
        """Require a new clock publisher, advancing time and fresh feedback for both robots."""
        deadline = time.monotonic() + self.profile["readiness_timeout_sec"]
        first_clock = previous = 0
        replacement = None
        reason = "Waiting for a replacement Gazebo clock publisher."
        while time.monotonic() < deadline:
            self._spin()
            endpoints = self._endpoints()
            publishers = endpoints["publishers"]["/clock"]
            if len(publishers) != 1 or publishers[0] in old["publishers"]["/clock"]:
                continue
            if self.clock_gid != publishers[0] or self.clock_ns <= 0:
                continue
            if replacement is None:
                replacement, first_clock = publishers[0], self.clock_ns
            if replacement != publishers[0] or self.clock_ns < previous:
                raise RuntimeError("Gazebo clock reset or changed publisher during reset verification.")
            previous = self.clock_ns
            reason = "Waiting for advancing simulation time and fresh feedback for both robots."
            if self.clock_ns - first_clock < 250_000_000 or self.description is None:
                continue
            model = ET.fromstring(self.description)
            controlled = {joint.attrib["name"] for joint in model.findall("./ros2_control/joint")}
            if not controlled or not set(required_joints).issubset(controlled):
                raise RuntimeError("Replacement robot description does not contain the recorded robot joints.")
            measured = {}
            for topic, (message, gid, received) in self.joints.items():
                stamp = message.header.stamp.sec * 10**9 + message.header.stamp.nanosec
                if (gid in old["publishers"].get(topic, [])
                        or gid not in endpoints["publishers"].get(topic, [])
                        or not first_clock <= stamp <= self.clock_ns
                        or (self.clock_ns - stamp) / 1e9 > self.profile["state_max_age_sec"]
                        or time.monotonic() - received > self.profile["state_max_age_sec"]):
                    continue
                if len(message.name) != len(message.position) or len(set(message.name)) != len(message.name):
                    raise RuntimeError("Replacement joint feedback is malformed.")
                for name, position in zip(message.name, message.position, strict=True):
                    if not math.isfinite(position):
                        raise RuntimeError("Replacement joint feedback is not finite.")
                    measured[name] = {"position": position, "stamp_ns": stamp, "publisher_gid": gid}
            if controlled.issubset(measured):
                return {"clock_publisher_gid": replacement, "first_clock_ns": first_clock,
                        "clock_ns": self.clock_ns, "required_joints": sorted(controlled),
                        "ros_domain_id": self.context.get_domain_id(),
                        "joint_feedback": {name: measured[name] for name in sorted(controlled)}}
        raise TimeoutError(reason)
