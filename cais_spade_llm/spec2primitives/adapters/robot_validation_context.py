from __future__ import annotations

"""Capture measured robot geometry through read-only ROS interfaces."""

import asyncio
import math
import time
import threading
import xml.etree.ElementTree as ET
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import asynccontextmanager
from copy import deepcopy
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from ..agents.ra.refinement_records import fingerprint

_MOTION_POLICY_FIELDS = (
    "approach_height_m",
    "pick_tcp_z_bias_max_m",
    "pick_tcp_z_bias_min_m",
    "min_pick_tcp_z_m",
    "place_surface_gap_m",
    "trajectory_time_scale",
)


def pose_matrix(pose: Mapping[str, Any]) -> np.ndarray:
    """Validate a finite pose and return its rigid transform."""
    values = [pose[name] for name in ("x", "y", "z", "qx", "qy", "qz", "qw")]
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
        for value in values
    ):
        raise ValueError("A measured pose requires finite coordinates and quaternion.")
    quaternion = np.asarray(values[3:], dtype=float)
    if abs(float(np.linalg.norm(quaternion)) - 1.0) > 1e-5:
        raise ValueError("Pose quaternion is not unit length.")
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_quat(quaternion).as_matrix()
    matrix[:3, 3] = values[:3]
    return matrix


def matrix_pose(matrix: Any) -> dict[str, float]:
    """Convert an already validated rigid transform to the primitive pose fields."""
    matrix = np.asarray(matrix, dtype=float)
    if (
        matrix.shape != (4, 4)
        or not np.isfinite(matrix).all()
        or not np.allclose(matrix[3], [0, 0, 0, 1])
    ):
        raise ValueError("Invalid rigid transform.")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6) or not math.isclose(
        float(np.linalg.det(rotation)), 1, abs_tol=1e-6
    ):
        raise ValueError("Transform rotation is invalid.")
    return dict(
        zip(
            ("x", "y", "z", "qx", "qy", "qz", "qw"),
            [*matrix[:3, 3], *Rotation.from_matrix(rotation).as_quat()],
            strict=True,
        )
    )


def calculation_policy(configuration: Mapping[str, Any]) -> dict[str, Any]:
    """Capture effective non-geometric controller settings without recovery recipes."""
    motion = configuration["motion"]
    policy = {name: float(motion[name]) for name in _MOTION_POLICY_FIELDS}
    policy["insertion_depth_m"] = float(configuration["parts_tuning"]["insertion_depth_m"])
    if not all(math.isfinite(value) for value in policy.values()):
        raise ValueError("Controller calculation policy is non-finite.")
    # These are the unchanged runtime's configuration-key conventions, not a
    # change to the recognized part symbol or a product geometry inference.
    policy["pick_z_adjustments_m"] = {
        str(key or "").strip().upper(): float(value)
        for key, value in configuration["parts_tuning"].get("pick_z_adjustments_m", {}).items()
        if str(key or "").strip()
    }
    if not all(math.isfinite(value) for value in policy["pick_z_adjustments_m"].values()):
        raise ValueError("Controller adjustment policy is non-finite.")
    return policy


class MeasuredRobotContextRuntime:
    """Read joint state, full TF and robot-model parameters without action clients."""

    async def capture(
        self,
        *,
        resource_jid: str,
        assignment_fingerprint: str,
        configuration: Mapping[str, Any],
        profile: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Capture one selected robot's synchronized, timestamped calculation context."""
        return await asyncio.to_thread(
            self._capture,
            resource_jid,
            assignment_fingerprint,
            deepcopy(dict(configuration)),
            deepcopy(dict(profile)),
        )

    @asynccontextmanager
    async def validation_context(
        self, *, resource_jid: str, assignment_fingerprint: str,
        configuration: Mapping[str, Any], profile: Mapping[str, Any],
    ) -> AsyncIterator[dict[str, Any]]:
        """Deliver a measured snapshot before tearing down its owned ROS context."""
        loop = asyncio.get_running_loop()
        ready = loop.create_future()
        release = threading.Event()

        def deliver(record: dict[str, Any]) -> None:
            def publish() -> None:
                if not ready.done():
                    ready.set_result(record)
            loop.call_soon_threadsafe(publish)
            release.wait()

        async def capture() -> None:
            try:
                await asyncio.to_thread(self._capture, resource_jid, assignment_fingerprint,
                                        deepcopy(dict(configuration)), deepcopy(dict(profile)), deliver)
            except (ImportError, OSError, RuntimeError, KeyError, TypeError, ValueError) as exc:
                if not ready.done():
                    ready.set_exception(exc)
                else:
                    raise

        worker = asyncio.create_task(capture())
        try:
            yield await ready
        finally:
            release.set()
            await asyncio.shield(worker)

    def _capture(
        self,
        resource_jid: str,
        assignment_fingerprint: str,
        configuration: dict[str, Any],
        profile: dict[str, Any],
        publish: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """Capture fresh feedback against the simulation clock and retain timeout details."""
        import rclpy
        from rcl_interfaces.srv import GetParameters, ListParameters
        from rclpy.node import Node
        from rclpy.context import Context
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.parameter import parameter_value_to_python
        from rclpy.parameter import Parameter
        from sensor_msgs.msg import JointState
        from tf2_ros import Buffer, TransformException, TransformListener

        move_group = configuration["move_group"]
        resource = profile["resources"][resource_jid]
        policy = calculation_policy(configuration)
        context = Context()
        rclpy.init(context=context)
        executor = SingleThreadedExecutor(context=context)
        node = Node(
            "spec2primitives_measured_robot_context",
            context=context,
            parameter_overrides=[Parameter("use_sim_time", value=True)],
        )
        executor.add_node(node)
        buffer = Buffer()
        listener = TransformListener(buffer, node)
        received: list[Any] = []
        subscription = node.create_subscription(
            JointState, resource["joint_states_topic"], lambda value: received.append(value), 10
        )
        # A new ROS context must discover its peers before it can supply data.
        # Keep startup separate from the age of the measured feedback.
        deadline = time.monotonic() + float(profile["worker_startup_timeout_sec"])

        def call(client: Any, request: Any) -> Any:
            remaining = max(0.0, deadline - time.monotonic())
            if not client.wait_for_service(timeout_sec=remaining):
                raise RuntimeError(f"Robot model parameter service is unavailable: {client.srv_name}.")
            timeout = min(float(profile["service_timeout_sec"]), max(0.0, deadline - time.monotonic()))
            response_deadline = time.monotonic() + timeout
            retry_at = response_deadline - timeout / 2
            futures = [client.call_async(request)] if timeout > 0 else []
            try:
                while futures:
                    for future in futures:
                        if future.done() and future.result() is not None:
                            return future.result()
                    remaining = max(0.0, response_deadline - time.monotonic())
                    if remaining <= 0 or len(futures) == 2 and all(future.done() for future in futures):
                        break
                    if len(futures) == 1 and (time.monotonic() >= retry_at or futures[0].done()):
                        # These reads have no side effects. Keep a slow first
                        # reply eligible so retrying cannot shorten its budget.
                        futures.append(client.call_async(request))
                        continue
                    wait_until = retry_at if len(futures) == 1 else response_deadline
                    executor.spin_once(timeout_sec=min(0.1, max(0.0, wait_until - time.monotonic())))
            finally:
                for future in futures:
                    client.remove_pending_request(future)
                    if not future.done():
                        future.cancel()
            selection = (
                f"{len(request.names)} parameters" if hasattr(request, "names")
                else "prefixes=" + ", ".join(request.prefixes)
            )
            raise RuntimeError(
                f"Robot model parameter read timed out: {client.srv_name} "
                f"({selection}; {len(futures)} attempts within {timeout:g} seconds)."
            )

        try:
            frame, ee_link, tcp_link = (
                move_group[key] for key in ("frame_id", "ee_link", "tcp_link")
            )
            # Parameter discovery may take longer than the feedback age limit.
            # Read the model first while subscriptions collect current samples.
            parameter_node = resource["move_group_node"].rstrip("/")
            listing = node.create_client(ListParameters, parameter_node + "/list_parameters")
            prefixes = [
                "robot_description",
                "robot_description_semantic",
                "robot_description_kinematics",
                "robot_description_planning",
                "planning_pipelines",
                "default_planning_pipeline",
                "ompl",
                "use_sim_time",
            ]
            names = call(listing, ListParameters.Request(prefixes=prefixes, depth=0)).result.names
            getter = node.create_client(GetParameters, parameter_node + "/get_parameters")
            values = call(getter, GetParameters.Request(names=names)).values
            model_parameters = {
                name: parameter_value_to_python(value)
                for name, value in zip(names, values, strict=True)
            }
            # The active pipeline need not use the ompl parameter namespace.
            # Copy its declared settings from the same selected MoveIt node.
            pipelines = model_parameters.get("planning_pipelines", []) or []
            if not isinstance(pipelines, list) or any(not isinstance(name, str) for name in pipelines):
                raise ValueError("The selected robot's planning_pipelines parameter is invalid.")
            pipelines = list(pipelines)
            default = model_parameters.get("default_planning_pipeline")
            if isinstance(default, str) and default and default not in pipelines:
                pipelines.append(default)
            if pipelines:
                names = call(listing, ListParameters.Request(prefixes=pipelines, depth=0)).result.names
                names = [name for name in names if name not in model_parameters]
                if names:
                    values = call(getter, GetParameters.Request(names=names)).values
                    model_parameters.update({
                        name: parameter_value_to_python(value)
                        for name, value in zip(names, values, strict=True)
                    })
            if not all(
                isinstance(model_parameters.get(name), str) and model_parameters[name]
                for name in ("robot_description", "robot_description_semantic")
            ):
                raise RuntimeError("The selected robot's URDF/SRDF parameters are unavailable.")
            touch_links = gripper_touch_links(
                model_parameters["robot_description"], configuration["gripper"]["joint"]
            )
            joint = None
            transforms = None
            last_issue = f"No JointState messages received on {resource['joint_states_topic']}."
            while time.monotonic() < deadline:
                executor.spin_once(timeout_sec=0.05)
                transforms = None
                if not received:
                    continue
                now_ns = node.get_clock().now().nanoseconds
                if now_ns <= 0:
                    last_issue = "The simulation clock has not supplied a positive time."
                    continue
                # JointState can arrive ahead of the lower-rate /clock update.
                # Select a buffered sample without accepting future timestamps
                # or relaxing the existing age and capture-skew checks below.
                joint = next((value for value in reversed(received)
                              if value.header.stamp.sec * 10**9 + value.header.stamp.nanosec <= now_ns), None)
                if joint is None:
                    latest_stamp = received[-1].header.stamp
                    last_issue = (
                        f"No joint state at or before ROS time {now_ns} on {resource['joint_states_topic']}; "
                        f"latest stamp is {latest_stamp.sec * 10**9 + latest_stamp.nanosec}."
                    )
                    continue
                try:
                    transforms = [
                        buffer.lookup_transform(frame, link, rclpy.time.Time())
                        for link in (ee_link, tcp_link)
                    ]
                except TransformException as exc:
                    last_issue = f"TF lookup failed for {frame} -> {ee_link} and {frame} -> {tcp_link}: {exc}"
                    continue
                stamps = [joint.header.stamp.sec * 10**9 + joint.header.stamp.nanosec]
                stamps.extend(
                    t.header.stamp.sec * 10**9 + t.header.stamp.nanosec for t in transforms
                )
                # The planning-frame-to-EE/TCP transforms are dynamic; unlike a
                # static tool edge, a zero stamp cannot establish their freshness.
                if any(
                    stamp <= 0
                    or not 0 <= now_ns - stamp <= float(profile["state_max_age_sec"]) * 1e9
                    for stamp in stamps
                ):
                    last_issue = (
                        f"Feedback timestamps do not satisfy state_max_age_sec={profile['state_max_age_sec']}: "
                        f"ROS time={now_ns}, {resource['joint_states_topic']} stamp={stamps[0]}, "
                        f"{frame} -> {ee_link} stamp={stamps[1]}, {frame} -> {tcp_link} stamp={stamps[2]}."
                    )
                    transforms = None
                    continue
                if max(stamps) - min(stamps) > float(profile["max_capture_skew_sec"]) * 1e9:
                    last_issue = (
                        f"Feedback timestamp skew is {(max(stamps) - min(stamps)) / 1e9:.6g} seconds; "
                        f"max_capture_skew_sec={profile['max_capture_skew_sec']}."
                    )
                    transforms = None
                    continue
                break
            if transforms is None or joint is None:
                raise RuntimeError("Fresh joint state and EE/TCP transforms are unavailable. " + last_issue)
            if (
                len(joint.name) != len(joint.position)
                or not joint.name
                or len(set(joint.name)) != len(joint.name)
                or not all(math.isfinite(value) for value in joint.position)
            ):
                raise ValueError("Measured joint state is invalid.")
            poses = []
            for transform in transforms:
                translation, rotation = (
                    transform.transform.translation,
                    transform.transform.rotation,
                )
                pose = {
                    "x": translation.x,
                    "y": translation.y,
                    "z": translation.z,
                    "qx": rotation.x,
                    "qy": rotation.y,
                    "qz": rotation.z,
                    "qw": rotation.w,
                }
                pose_matrix(pose)
                poses.append(pose)
            record = {
                "record_type": "RobotValidationContext",
                "resource_jid": resource_jid,
                "assignment_fingerprint": assignment_fingerprint,
                "frame_id": frame,
                "ee_link": ee_link,
                "tcp_link": tcp_link,
                "group_name": move_group["group_name"],
                "ee_pose": poses[0],
                "tcp_pose": poses[1],
                "position_tolerance_m": float(move_group["position_tolerance_m"]),
                "touch_links": touch_links,
                "ee_from_tcp": (
                    np.linalg.inv(pose_matrix(poses[0])) @ pose_matrix(poses[1])
                ).tolist(),
                "joint_state": {
                    "names": list(joint.name),
                    "positions": list(joint.position),
                    "stamp_ns": stamps[0],
                },
                "tf_stamps_ns": stamps[1:],
                "measured_at_ros_ns": now_ns,
                "captured_at_ns": time.time_ns(),
                "configuration_sha256": fingerprint(configuration),
                "policy": policy,
                "gripper": {
                    key: deepcopy(configuration["gripper"][key])
                    for key in ("joint", "open", "close", "open_width_mm")
                },
                "model_parameters": model_parameters,
                "model_parameters_sha256": fingerprint(model_parameters),
            }
            if publish is not None:
                publish(record)
            return record
        finally:
            node.destroy_subscription(subscription)
            listener.unregister()
            node.destroy_node()
            executor.shutdown()
            context.try_shutdown()


@asynccontextmanager
async def validation_capture(
    runtime: Any, assignment: Any, *, profile: Mapping[str, Any], custody: Mapping[str, Any] | None = None,
) -> AsyncIterator[Mapping[str, Any]]:
    """Keep the owned live capture open across validation; retain standalone adapters."""
    context = getattr(runtime, "validation_context", None)
    if context is not None:
        async with context(assignment, profile=profile, _execution_custody=custody) as record:
            yield record
    elif custody is not None:
        yield await runtime.capture_execution_context(assignment, profile=profile, custody=custody)
    else:
        yield await runtime.capture_validation_context(assignment, profile=profile)


def composition_robot_context(record: Mapping[str, Any]) -> dict[str, Any]:
    """Project measured feedback without exposing large internal robot model records."""
    return {key: deepcopy(value) for key, value in record.items() if key != "model_parameters"}


def gripper_touch_links(description: str, gripper_joint: str) -> list[str]:
    """Read the configured gripper's descendant links from its measured model configuration."""
    robot = ET.fromstring(description)
    joints = robot.findall("joint")
    selected = next((joint for joint in joints if joint.attrib.get("name") == gripper_joint), None)
    if selected is None or selected.find("parent") is None:
        raise ValueError("Configured gripper joint is absent from the robot model.")
    links = {selected.find("parent").attrib["link"]}
    changed = True
    while changed:
        changed = False
        for joint in joints:
            parent, child = joint.find("parent"), joint.find("child")
            if (
                parent is not None
                and child is not None
                and parent.attrib["link"] in links
                and child.attrib["link"] not in links
            ):
                links.add(child.attrib["link"])
                changed = True
    return sorted(links)
