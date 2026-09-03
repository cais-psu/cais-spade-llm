"""Run collision-aware MoveIt Cartesian plans without executing robot motion."""

from __future__ import annotations

import asyncio
import math
import re
import time
from collections.abc import Mapping, Sequence
from typing import Any

_MAX_STEP_M = 0.01
_JUMP_THRESHOLD = 0.0
_MIN_COMPLETE_FRACTION = 0.999
_PHASE_ROLES = {
    "pick": ("pick_approach", "grasp", "pick_retreat"),
    "place": ("transfer", "place_approach", "placement", "place_retreat"),
}


class MoveItPlanOnlyRuntime:
    """Validate a complete Cartesian pick-and-place path through MoveIt."""

    def __init__(
        self,
        *,
        service_timeout_sec: float = 10.0,
        result_timeout_sec: float = 20.0,
        tf_timeout_sec: float = 5.0,
    ) -> None:
        """Configure bounded, strict, no-motion Cartesian validation."""
        self._service_timeout_sec = service_timeout_sec
        self._result_timeout_sec = result_timeout_sec
        self._tf_timeout_sec = tf_timeout_sec

    async def validate(self, request: Mapping[str, object]) -> Mapping[str, object]:
        """Plan both chained Cartesian phases off the caller's event loop."""
        return await asyncio.to_thread(self._validate_sync, dict(request))

    def _validate_sync(self, request: Mapping[str, object]) -> Mapping[str, object]:
        """Query live TF and plan two complete collision-aware Cartesian paths."""
        validated = _validated_request(request)
        try:
            import rclpy
            import tf2_ros
            from moveit_msgs.srv import GetCartesianPath
            from rclpy.node import Node
        except ImportError as exc:
            return _unavailable_response(f"ROS2 MoveIt Cartesian interfaces are unavailable: {exc}")

        initialized_here = False
        if not rclpy.ok():
            rclpy.init()
            initialized_here = True
        node_name = re.sub(
            r"[^A-Za-z0-9_]",
            "_",
            f"spec2primitives_{validated['resource_symbol']}_cartesian_plan_only",
        )
        node = Node(node_name)
        client = node.create_client(
            GetCartesianPath,
            str(validated["cartesian_path_service"]),
        )
        tf_buffer = tf2_ros.Buffer()
        tf_listener = tf2_ros.TransformListener(tf_buffer, node)
        try:
            if not client.wait_for_service(timeout_sec=self._service_timeout_sec):
                return _unavailable_response(
                    f"{validated['cartesian_path_service']} is unavailable for "
                    "Cartesian plan-only validation."
                )
            ee_transform = _lookup_live_transform(
                rclpy,
                node,
                tf_buffer,
                target_frame=str(validated["target_frame"]),
                source_frame=str(validated["end_effector_link"]),
                timeout_sec=self._tf_timeout_sec,
                transform_exception=tf2_ros.TransformException,
            )
            tcp_transform = _lookup_live_transform(
                rclpy,
                node,
                tf_buffer,
                target_frame=str(validated["target_frame"]),
                source_frame=str(validated["tcp_link"]),
                timeout_sec=self._tf_timeout_sec,
                transform_exception=tf2_ros.TransformException,
            )
            if ee_transform is None or tcp_transform is None:
                return _unavailable_response(
                    "Live end-effector and TCP transforms are unavailable."
                )
            live_start_pose, ee_to_tcp = _live_pose_and_ee_to_tcp(
                ee_transform,
                tcp_transform,
                target_frame=str(validated["target_frame"]),
                end_effector_link=str(validated["end_effector_link"]),
                tcp_link=str(validated["tcp_link"]),
            )
            waypoints = _cartesian_waypoints(
                validated,
                live_start_pose=live_start_pose,
                ee_to_tcp=ee_to_tcp,
            )
            pick_result, terminal_state = _plan_cartesian_phase(
                rclpy,
                node,
                client,
                GetCartesianPath,
                validated,
                phase="pick",
                waypoints=waypoints,
                start_state=None,
                timeout_sec=self._result_timeout_sec,
            )
            if pick_result["status"] != "accepted" or terminal_state is None:
                place_result = _skipped_phase(
                    "The place phase was not planned because the pick phase did not "
                    "produce an accepted terminal robot state."
                )
                return _combined_response(
                    live_start_pose,
                    ee_to_tcp,
                    waypoints,
                    pick_result,
                    place_result,
                )
            place_result, _ = _plan_cartesian_phase(
                rclpy,
                node,
                client,
                GetCartesianPath,
                validated,
                phase="place",
                waypoints=waypoints,
                start_state=terminal_state,
                timeout_sec=self._result_timeout_sec,
            )
            return _combined_response(
                live_start_pose,
                ee_to_tcp,
                waypoints,
                pick_result,
                place_result,
            )
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            return _unavailable_response(
                f"Cartesian plan-only validation is unavailable: {type(exc).__name__}: {exc}"
            )
        finally:
            del tf_listener
            node.destroy_node()
            if initialized_here and rclpy.ok():
                rclpy.shutdown()


def _validated_request(  # noqa: C901
    value: Mapping[str, object],
) -> Mapping[str, object]:
    expected = {
        "process_symbol",
        "process_iri",
        "feature_iri",
        "resource_symbol",
        "resource_iri",
        "resource_jid",
        "execution_mode",
        "motion_mode",
        "moveit_group",
        "end_effector_link",
        "tcp_link",
        "target_frame",
        "cartesian_path_service",
        "validation_scope",
        "checked_constraints",
        "unvalidated_constraints",
        "cartesian_parameters",
        "current_state",
        "desired_state",
        "grounded_targets",
        "mode",
        "motion_executed",
        "request_fingerprint",
    }
    if set(value) != expected:
        raise ValueError("Cartesian plan-only allocation request fields are invalid.")
    for field in (
        "process_symbol",
        "process_iri",
        "feature_iri",
        "resource_symbol",
        "resource_iri",
        "resource_jid",
        "moveit_group",
        "end_effector_link",
        "tcp_link",
        "target_frame",
        "cartesian_path_service",
        "request_fingerprint",
    ):
        if not isinstance(value[field], str) or not str(value[field]).strip():
            raise ValueError(f"Cartesian plan-only allocation {field} is invalid.")
    if value["execution_mode"] != "simulation":
        raise ValueError("Live Cartesian allocation validation is simulation-only.")
    if value["motion_mode"] != "cartesian_pick_place":
        raise ValueError("Cartesian plan-only allocation motion_mode is invalid.")
    if value["mode"] != "plan_only" or value["motion_executed"] is not False:
        raise ValueError("Cartesian plan-only allocation may not execute motion.")
    if value["validation_scope"] != "cartesian_pick_place":
        raise ValueError("Cartesian plan-only validation scope is invalid.")
    if value["checked_constraints"] != [
        "live_tf",
        "collision_aware_cartesian_pick_path",
        "collision_aware_cartesian_transfer_place_path",
        "complete_path_fraction",
    ]:
        raise ValueError("Cartesian plan-only checked constraints are invalid.")
    expected_unvalidated = [
        "grasp_contact",
        "gripper_actuation",
        "attached_part_collision_geometry",
        f"{value['process_symbol']}_tolerance",
        "force_control",
        "final_constrained_insertion_stroke",
    ]
    if value["unvalidated_constraints"] != expected_unvalidated:
        raise ValueError("Cartesian plan-only unvalidated constraints are invalid.")
    parameters = value["cartesian_parameters"]
    if not isinstance(parameters, Mapping) or set(parameters) != {
        "max_step_m",
        "jump_threshold",
        "avoid_collisions",
        "minimum_fraction",
    }:
        raise ValueError("Cartesian planning parameters are invalid.")
    if (
        not math.isclose(float(parameters["max_step_m"]), _MAX_STEP_M)
        or not math.isclose(float(parameters["jump_threshold"]), _JUMP_THRESHOLD)
        or parameters["avoid_collisions"] is not True
        or not math.isclose(
            float(parameters["minimum_fraction"]),
            _MIN_COMPLETE_FRACTION,
        )
    ):
        raise ValueError("Cartesian planning parameters are not strict.")
    for state_name in ("current_state", "desired_state"):
        state = value[state_name]
        if not isinstance(state, Mapping) or set(state) != {
            "state_iri",
            "evidence_handle",
            "translation_m",
            "location_record_ref",
            "location_record_sha256",
        }:
            raise ValueError(f"Cartesian plan-only {state_name} is invalid.")
        if not isinstance(state["state_iri"], str) or not state["state_iri"]:
            raise ValueError(f"Cartesian plan-only {state_name} state_iri is invalid.")
        if not isinstance(state["evidence_handle"], str) or not state["evidence_handle"]:
            raise ValueError(f"Cartesian plan-only {state_name} evidence_handle is invalid.")
        _finite_vector3(state["translation_m"], f"{state_name}.translation_m")
    targets = value["grounded_targets"]
    if not isinstance(targets, Mapping) or set(targets) != {
        "pick_object_center_m",
        "pick_support_point_m",
        "pick_surface_normal",
        "place_support_point_m",
        "place_surface_normal",
        "place_object_center_m",
        "part_dimensions_m",
        "support_dimensions_m",
        "part_height_m",
        "motion_offsets",
    }:
        raise ValueError("Grounded Cartesian targets are invalid.")
    for key in (
        "pick_object_center_m",
        "pick_support_point_m",
        "pick_surface_normal",
        "place_support_point_m",
        "place_surface_normal",
        "place_object_center_m",
        "part_dimensions_m",
        "support_dimensions_m",
    ):
        _finite_vector3(targets[key], f"grounded_targets.{key}")
    part_height = _finite_number(targets["part_height_m"], "part_height_m")
    if part_height <= 0.0:
        raise ValueError("Grounded part_height_m must be positive.")
    offsets = targets["motion_offsets"]
    if not isinstance(offsets, Mapping) or set(offsets) != {
        "pick_approach_height_m",
        "pick_surface_clearance_m",
        "pick_tcp_z_bias_min_m",
        "pick_tcp_z_bias_max_m",
        "transfer_clearance_m",
        "place_approach_height_m",
    }:
        raise ValueError("Grounded Cartesian motion offsets are invalid.")
    offset_values = {
        key: _finite_number(raw, f"motion_offsets.{key}") for key, raw in offsets.items()
    }
    if any(number < 0.0 for number in offset_values.values()) or (
        offset_values["pick_tcp_z_bias_min_m"] > offset_values["pick_tcp_z_bias_max_m"]
    ):
        raise ValueError("Grounded Cartesian motion offsets are inconsistent.")
    _unit_vector(targets["pick_surface_normal"], "pick_surface_normal")
    _unit_vector(targets["place_surface_normal"], "place_surface_normal")
    return value


def _lookup_live_transform(
    rclpy: Any,
    node: Any,
    tf_buffer: Any,
    *,
    target_frame: str,
    source_frame: str,
    timeout_sec: float,
    transform_exception: type[Exception] = RuntimeError,
) -> object | None:
    deadline = time.monotonic() + timeout_sec
    while rclpy.ok() and time.monotonic() < deadline:
        try:
            return tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                rclpy.time.Time(),
            )
        except transform_exception:
            rclpy.spin_once(
                node,
                timeout_sec=min(0.1, max(0.0, deadline - time.monotonic())),
            )
    return None


def _live_pose_and_ee_to_tcp(
    ee_transform: object,
    tcp_transform: object,
    *,
    target_frame: str,
    end_effector_link: str,
    tcp_link: str,
) -> tuple[dict[str, object], dict[str, object]]:
    ee_position, ee_quaternion = _transform_components(ee_transform, "end-effector")
    tcp_position, tcp_quaternion = _transform_components(tcp_transform, "TCP")
    world_delta = tuple(tcp_position[index] - ee_position[index] for index in range(3))
    ee_rotation = _quaternion_rotation_matrix(ee_quaternion)
    relative_translation = tuple(
        sum(ee_rotation[row][column] * world_delta[row] for row in range(3)) for column in range(3)
    )
    relative_rotation = _quaternion_multiply(
        _quaternion_conjugate(ee_quaternion),
        tcp_quaternion,
    )
    live_start_pose = {
        "frame_id": target_frame,
        "link_name": end_effector_link,
        "position_m": list(ee_position),
        "orientation_xyzw": list(ee_quaternion),
    }
    ee_to_tcp = {
        "parent_link": end_effector_link,
        "child_link": tcp_link,
        "translation_m": list(relative_translation),
        "rotation_xyzw": list(relative_rotation),
    }
    return live_start_pose, ee_to_tcp


def _cartesian_waypoints(
    request: Mapping[str, object],
    *,
    live_start_pose: Mapping[str, object],
    ee_to_tcp: Mapping[str, object],
) -> list[dict[str, object]]:
    targets = request["grounded_targets"]
    assert isinstance(targets, Mapping)
    offsets = targets["motion_offsets"]
    assert isinstance(offsets, Mapping)
    pick_center = _finite_vector3(targets["pick_object_center_m"], "pick center")
    place_center = _finite_vector3(targets["place_object_center_m"], "place center")
    pick_normal = _unit_vector(targets["pick_surface_normal"], "pick normal")
    place_normal = _unit_vector(targets["place_surface_normal"], "place normal")
    part_height = _finite_number(targets["part_height_m"], "part height")
    pick_bias = max(
        _finite_number(offsets["pick_tcp_z_bias_min_m"], "pick bias minimum"),
        min(
            _finite_number(offsets["pick_tcp_z_bias_max_m"], "pick bias maximum"),
            part_height * 0.25,
        ),
    )
    pick_bias += _finite_number(
        offsets["pick_surface_clearance_m"],
        "pick surface clearance",
    )
    grasp_tcp = _vector_add(pick_center, _vector_scale(pick_normal, pick_bias))
    grasp_tcp_to_part = tuple(grasp_tcp[index] - pick_center[index] for index in range(3))
    place_tcp = _vector_add(place_center, grasp_tcp_to_part)

    ee_quaternion = _normalized_quaternion(
        live_start_pose["orientation_xyzw"],
        "live end-effector orientation",
    )
    ee_to_tcp_translation = _finite_vector3(
        ee_to_tcp["translation_m"],
        "EE-to-TCP translation",
    )
    world_ee_to_tcp = _rotate_vector(ee_quaternion, ee_to_tcp_translation)
    grasp_ee = _vector_subtract(grasp_tcp, world_ee_to_tcp)
    place_ee = _vector_subtract(place_tcp, world_ee_to_tcp)
    pick_approach = _vector_add(
        grasp_ee,
        _vector_scale(
            pick_normal,
            _finite_number(offsets["pick_approach_height_m"], "pick approach"),
        ),
    )
    transfer = _vector_add(
        place_ee,
        _vector_scale(
            place_normal,
            _finite_number(offsets["transfer_clearance_m"], "transfer clearance"),
        ),
    )
    place_approach = _vector_add(
        place_ee,
        _vector_scale(
            place_normal,
            _finite_number(offsets["place_approach_height_m"], "place approach"),
        ),
    )
    positions = (
        ("pick", "pick_approach", pick_approach),
        ("pick", "grasp", grasp_ee),
        ("pick", "pick_retreat", pick_approach),
        ("place", "transfer", transfer),
        ("place", "place_approach", place_approach),
        ("place", "placement", place_ee),
        ("place", "place_retreat", place_approach),
    )
    return [
        {
            "phase": phase,
            "role": role,
            "pose": {
                "position_m": list(position),
                "orientation_xyzw": list(ee_quaternion),
            },
        }
        for phase, role, position in positions
    ]


def _plan_cartesian_phase(  # noqa: PLR0913
    rclpy: Any,
    node: Any,
    client: Any,
    service_type: Any,
    request: Mapping[str, object],
    *,
    phase: str,
    waypoints: Sequence[Mapping[str, object]],
    start_state: object | None,
    timeout_sec: float,
) -> tuple[dict[str, object], object | None]:
    phase_waypoints = [item for item in waypoints if item.get("phase") == phase]
    roles = [str(item["role"]) for item in phase_waypoints]
    if tuple(roles) != _PHASE_ROLES[phase]:
        raise ValueError(f"{phase} Cartesian waypoint roles are invalid.")
    service_request = service_type.Request()
    service_request.header.frame_id = str(request["target_frame"])
    service_request.header.stamp = node.get_clock().now().to_msg()
    service_request.group_name = str(request["moveit_group"])
    service_request.link_name = str(request["end_effector_link"])
    service_request.waypoints = [_geometry_pose(item["pose"]) for item in phase_waypoints]
    service_request.max_step = _MAX_STEP_M
    service_request.jump_threshold = _JUMP_THRESHOLD
    service_request.avoid_collisions = True
    if start_state is None:
        service_request.start_state.is_diff = True
    else:
        service_request.start_state = start_state

    future = client.call_async(service_request)
    if not _wait_future(rclpy, node, future, timeout_sec=timeout_sec):
        return (
            _phase_result(
                phase,
                "needs_context",
                roles,
                fraction=None,
                error_code=None,
                terminal_state_available=False,
                message=f"{phase} Cartesian planning service timed out.",
            ),
            None,
        )
    response = future.result()
    if response is None:
        return (
            _phase_result(
                phase,
                "needs_context",
                roles,
                fraction=None,
                error_code=None,
                terminal_state_available=False,
                message=f"{phase} Cartesian planning returned no response.",
            ),
            None,
        )
    fraction = getattr(response, "fraction", None)
    error_code = getattr(getattr(response, "error_code", None), "val", None)
    if isinstance(fraction, bool) or not isinstance(fraction, (int, float)):
        return (
            _phase_result(
                phase,
                "needs_context",
                roles,
                fraction=None,
                error_code=error_code if isinstance(error_code, int) else None,
                terminal_state_available=False,
                message=f"{phase} Cartesian planning returned no valid path fraction.",
            ),
            None,
        )
    fraction_value = float(fraction)
    if not math.isfinite(fraction_value):
        raise ValueError(f"{phase} Cartesian path fraction is non-finite.")
    if not isinstance(error_code, int) or isinstance(error_code, bool):
        return (
            _phase_result(
                phase,
                "needs_context",
                roles,
                fraction=fraction_value,
                error_code=None,
                terminal_state_available=False,
                message=f"{phase} Cartesian planning returned no MoveIt error code.",
            ),
            None,
        )
    terminal_state = _trajectory_terminal_state(getattr(response, "solution", None))
    accepted = error_code == 1 and fraction_value >= _MIN_COMPLETE_FRACTION
    if not accepted:
        return (
            _phase_result(
                phase,
                "rejected",
                roles,
                fraction=fraction_value,
                error_code=error_code,
                terminal_state_available=terminal_state is not None,
                message=(
                    f"{phase} Cartesian path was rejected: fraction "
                    f"{fraction_value:.6f}, MoveIt error code {error_code}."
                ),
            ),
            None,
        )
    if terminal_state is None:
        return (
            _phase_result(
                phase,
                "needs_context",
                roles,
                fraction=fraction_value,
                error_code=error_code,
                terminal_state_available=False,
                message=f"{phase} Cartesian path has no terminal robot state.",
            ),
            None,
        )
    return (
        _phase_result(
            phase,
            "accepted",
            roles,
            fraction=fraction_value,
            error_code=error_code,
            terminal_state_available=True,
            message=f"{phase} has a complete collision-aware Cartesian path.",
        ),
        terminal_state,
    )


def _geometry_pose(value: object) -> Any:
    from geometry_msgs.msg import Pose

    if not isinstance(value, Mapping) or set(value) != {
        "position_m",
        "orientation_xyzw",
    }:
        raise ValueError("Cartesian waypoint pose is invalid.")
    position = _finite_vector3(value["position_m"], "waypoint position")
    orientation = _normalized_quaternion(
        value["orientation_xyzw"],
        "waypoint orientation",
    )
    pose = Pose()
    pose.position.x, pose.position.y, pose.position.z = position
    (
        pose.orientation.x,
        pose.orientation.y,
        pose.orientation.z,
        pose.orientation.w,
    ) = orientation
    return pose


def _trajectory_terminal_state(trajectory: object) -> object | None:
    joint_trajectory = getattr(trajectory, "joint_trajectory", None)
    joint_names = getattr(joint_trajectory, "joint_names", None)
    points = getattr(joint_trajectory, "points", None)
    if (
        not isinstance(joint_names, Sequence)
        or isinstance(joint_names, (str, bytes))
        or not joint_names
        or not isinstance(points, Sequence)
        or not points
    ):
        return None
    positions = getattr(points[-1], "positions", None)
    if not isinstance(positions, Sequence) or len(positions) != len(joint_names):
        return None
    from moveit_msgs.msg import RobotState

    state = RobotState()
    state.joint_state.name = [str(name) for name in joint_names]
    state.joint_state.position = [float(position) for position in positions]
    state.is_diff = True
    return state


def _wait_future(
    rclpy: Any,
    node: Any,
    future: Any,
    *,
    timeout_sec: float,
) -> bool:
    deadline = time.monotonic() + timeout_sec
    while rclpy.ok() and time.monotonic() < deadline and not future.done():
        rclpy.spin_once(node, timeout_sec=0.1)
    return future.done()


def _phase_result(
    phase: str,
    status: str,
    waypoint_roles: Sequence[str],
    *,
    fraction: float | None,
    error_code: int | None,
    terminal_state_available: bool,
    message: str,
) -> dict[str, object]:
    return {
        "phase": phase,
        "status": status,
        "waypoint_roles": list(waypoint_roles),
        "fraction": fraction,
        "moveit_error_code": error_code,
        "terminal_state_available": terminal_state_available,
        "message": message,
    }


def _skipped_phase(message: str) -> dict[str, object]:
    return _phase_result(
        "place",
        "needs_context",
        _PHASE_ROLES["place"],
        fraction=None,
        error_code=None,
        terminal_state_available=False,
        message=message,
    )


def _combined_response(
    live_start_pose: Mapping[str, object],
    ee_to_tcp: Mapping[str, object],
    waypoints: Sequence[Mapping[str, object]],
    pick: Mapping[str, object],
    place: Mapping[str, object],
) -> Mapping[str, object]:
    phase_statuses = {pick["status"], place["status"]}
    status = (
        "rejected"
        if "rejected" in phase_statuses
        else "needs_context"
        if "needs_context" in phase_statuses
        else "accepted"
    )
    feedback = None
    if status != "accepted":
        feedback = " ".join(
            str(item["message"]) for item in (pick, place) if item["status"] != "accepted"
        )
    return {
        "status": status,
        "live_start_pose": dict(live_start_pose),
        "ee_to_tcp_transform": dict(ee_to_tcp),
        "waypoints": [dict(item) for item in waypoints],
        "phases": {"pick": dict(pick), "place": dict(place)},
        "feedback": feedback,
        "motion_executed": False,
    }


def _unavailable_response(message: str) -> Mapping[str, object]:
    return {
        "status": "needs_context",
        "live_start_pose": None,
        "ee_to_tcp_transform": None,
        "waypoints": [],
        "phases": {
            "pick": _phase_result(
                "pick",
                "needs_context",
                _PHASE_ROLES["pick"],
                fraction=None,
                error_code=None,
                terminal_state_available=False,
                message=message,
            ),
            "place": _skipped_phase(message),
        },
        "feedback": message,
        "motion_executed": False,
    }


def _transform_components(
    transform: object,
    label: str,
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    body = getattr(transform, "transform", None)
    translation = getattr(body, "translation", None)
    rotation = getattr(body, "rotation", None)
    position = _finite_vector3(
        [
            getattr(translation, "x", None),
            getattr(translation, "y", None),
            getattr(translation, "z", None),
        ],
        f"live {label} translation",
    )
    quaternion = _normalized_quaternion(
        [
            getattr(rotation, "x", None),
            getattr(rotation, "y", None),
            getattr(rotation, "z", None),
            getattr(rotation, "w", None),
        ],
        f"live {label} orientation",
    )
    return position, quaternion


def _finite_number(value: object, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} is invalid.")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} is invalid.") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} is invalid.")
    return result


def _finite_vector3(value: object, label: str) -> tuple[float, float, float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 3:
        raise ValueError(f"{label} is invalid.")
    return tuple(_finite_number(item, label) for item in value)  # type: ignore[return-value]


def _unit_vector(value: object, label: str) -> tuple[float, float, float]:
    vector = _finite_vector3(value, label)
    norm = math.sqrt(sum(component * component for component in vector))
    if norm <= 1e-12:
        raise ValueError(f"{label} is invalid.")
    return tuple(component / norm for component in vector)  # type: ignore[return-value]


def _normalized_quaternion(
    value: object,
    label: str,
) -> tuple[float, float, float, float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 4:
        raise ValueError(f"{label} is invalid.")
    quaternion = tuple(_finite_number(item, label) for item in value)
    norm = math.sqrt(sum(component * component for component in quaternion))
    if norm <= 1e-12:
        raise ValueError(f"{label} is invalid.")
    return tuple(component / norm for component in quaternion)  # type: ignore[return-value]


def _quaternion_conjugate(
    value: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    return (-value[0], -value[1], -value[2], value[3])


def _quaternion_multiply(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    lx, ly, lz, lw = left
    rx, ry, rz, rw = right
    return _normalized_quaternion(
        (
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
            lw * rw - lx * rx - ly * ry - lz * rz,
        ),
        "quaternion product",
    )


def _quaternion_rotation_matrix(
    value: tuple[float, float, float, float],
) -> tuple[tuple[float, float, float], ...]:
    x, y, z, w = value
    return (
        (
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - z * w),
            2.0 * (x * z + y * w),
        ),
        (
            2.0 * (x * y + z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - x * w),
        ),
        (
            2.0 * (x * z - y * w),
            2.0 * (y * z + x * w),
            1.0 - 2.0 * (x * x + y * y),
        ),
    )


def _rotate_vector(
    quaternion: tuple[float, float, float, float],
    vector: tuple[float, float, float],
) -> tuple[float, float, float]:
    rotation = _quaternion_rotation_matrix(quaternion)
    return tuple(
        sum(rotation[row][column] * vector[column] for column in range(3)) for row in range(3)
    )  # type: ignore[return-value]


def _vector_add(
    left: tuple[float, float, float],
    right: tuple[float, float, float],
) -> tuple[float, float, float]:
    return tuple(left[index] + right[index] for index in range(3))  # type: ignore[return-value]


def _vector_subtract(
    left: tuple[float, float, float],
    right: tuple[float, float, float],
) -> tuple[float, float, float]:
    return tuple(left[index] - right[index] for index in range(3))  # type: ignore[return-value]


def _vector_scale(
    value: tuple[float, float, float],
    scale: float,
) -> tuple[float, float, float]:
    return tuple(component * scale for component in value)  # type: ignore[return-value]


__all__ = ["MoveItPlanOnlyRuntime"]
