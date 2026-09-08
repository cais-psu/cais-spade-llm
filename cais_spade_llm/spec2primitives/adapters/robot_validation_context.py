from __future__ import annotations

"""Capture measured robot geometry through read-only ROS interfaces."""

import asyncio
import math
import time
import xml.etree.ElementTree as ET
from collections.abc import Mapping
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

    def _capture(
        self,
        resource_jid: str,
        assignment_fingerprint: str,
        configuration: dict[str, Any],
        profile: dict[str, Any],
    ) -> dict[str, Any]:
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
        deadline = time.monotonic() + float(profile["service_timeout_sec"])

        def call(client: Any, request: Any) -> Any:
            remaining = max(0.0, deadline - time.monotonic())
            if not client.wait_for_service(timeout_sec=remaining):
                raise RuntimeError("Robot model parameter service is unavailable.")
            future = client.call_async(request)
            executor.spin_until_future_complete(
                future, timeout_sec=max(0.0, deadline - time.monotonic())
            )
            if not future.done() or future.result() is None:
                future.cancel()
                raise RuntimeError("Robot model parameter read timed out.")
            return future.result()

        try:
            frame, ee_link, tcp_link = (
                move_group[key] for key in ("frame_id", "ee_link", "tcp_link")
            )
            transforms = None
            while time.monotonic() < deadline:
                executor.spin_once(timeout_sec=0.05)
                if not received:
                    continue
                try:
                    transforms = [
                        buffer.lookup_transform(frame, link, rclpy.time.Time())
                        for link in (ee_link, tcp_link)
                    ]
                except TransformException:
                    continue
                now_ns = node.get_clock().now().nanoseconds
                joint = received[-1]
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
                    transforms = None
                    continue
                if max(stamps) - min(stamps) > float(profile["max_capture_skew_sec"]) * 1e9:
                    transforms = None
                    continue
                break
            if transforms is None or not received:
                raise RuntimeError("Fresh joint state and EE/TCP transforms are unavailable.")
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
            if not all(
                isinstance(model_parameters.get(name), str) and model_parameters[name]
                for name in ("robot_description", "robot_description_semantic")
            ):
                raise RuntimeError("The selected robot's URDF/SRDF parameters are unavailable.")
            now_ns = node.get_clock().now().nanoseconds
            if any(
                not 0 <= now_ns - stamp <= float(profile["state_max_age_sec"]) * 1e9
                for stamp in stamps
            ):
                raise RuntimeError("Measured robot feedback became stale during parameter capture.")
            return {
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
                "touch_links": gripper_touch_links(
                    model_parameters["robot_description"], configuration["gripper"]["joint"]
                ),
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
        finally:
            node.destroy_subscription(subscription)
            listener.unregister()
            node.destroy_node()
            executor.shutdown()
            context.try_shutdown()


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
