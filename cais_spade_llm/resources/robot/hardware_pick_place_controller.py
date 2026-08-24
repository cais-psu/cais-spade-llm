"""Hardware/digital_twin pick-place controller helpers for taught functions."""

from __future__ import annotations

import json
import math
import os
import threading
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .gazebo_pick_place_controller import (
    UR5E_JOINT_NAMES,
    UR5E_JOINT_STATES_TOPIC,
    UR5E_TRAJECTORY_TOPIC,
    XARM6_JOINT_STATES_TOPIC,
    GazeboPickPlaceController,
    derive_move_insert_timeout_sec,
)

TAUGHT_FUNCTIONS_ROOT = Path(__file__).resolve().parent / "taught_functions"
UR5E_RTDE_STATUS_PATH = Path("/tmp/cais_ur5e_rtde_trajectory_status.json")
XARM6_HARDWARE_JOINT_NAMES = [f"joint{index}" for index in range(1, 7)]


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return bool(default)


def _clamp(value: float, lo: float, hi: float) -> float:
    lower = min(float(lo), float(hi))
    upper = max(float(lo), float(hi))
    return max(lower, min(upper, float(value)))


def _normalized_quaternion(
    quaternion: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    norm = math.sqrt(sum(value * value for value in quaternion))
    if not math.isfinite(norm) or norm <= 1e-12:
        raise ValueError("quaternion norm is zero or non-finite")
    return tuple(value / norm for value in quaternion)


def _quaternion_multiply(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    lx, ly, lz, lw = left
    rx, ry, rz, rw = right
    return (
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
        lw * rw - lx * rx - ly * ry - lz * rz,
    )


def _quaternion_rotate(
    quaternion: tuple[float, float, float, float],
    vector: tuple[float, float, float],
) -> tuple[float, float, float]:
    qx, qy, qz, qw = _normalized_quaternion(quaternion)
    rotated = _quaternion_multiply(
        _quaternion_multiply((qx, qy, qz, qw), (*vector, 0.0)),
        (-qx, -qy, -qz, qw),
    )
    return rotated[:3]


def _compose_transforms(
    left: tuple[
        tuple[float, float, float],
        tuple[float, float, float, float],
    ],
    right: tuple[
        tuple[float, float, float],
        tuple[float, float, float, float],
    ],
) -> tuple[
    tuple[float, float, float],
    tuple[float, float, float, float],
]:
    left_translation, left_quaternion = left
    right_translation, right_quaternion = right
    rotated = _quaternion_rotate(left_quaternion, right_translation)
    return (
        tuple(left_translation[index] + rotated[index] for index in range(3)),
        _normalized_quaternion(
            _quaternion_multiply(left_quaternion, right_quaternion)
        ),
    )


def _inverse_transform(
    transform: tuple[
        tuple[float, float, float],
        tuple[float, float, float, float],
    ],
) -> tuple[
    tuple[float, float, float],
    tuple[float, float, float, float],
]:
    translation, quaternion = transform
    qx, qy, qz, qw = _normalized_quaternion(quaternion)
    inverse_quaternion = (-qx, -qy, -qz, qw)
    inverse_translation = _quaternion_rotate(
        inverse_quaternion,
        tuple(-value for value in translation),
    )
    return inverse_translation, inverse_quaternion


def _quaternion_to_rpy(
    quaternion: tuple[float, float, float, float],
) -> tuple[float, float, float]:
    x, y, z, w = _normalized_quaternion(quaternion)
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return roll, pitch, yaw


def _quaternion_from_rpy(
    roll: float,
    pitch: float,
    yaw: float,
) -> tuple[float, float, float, float]:
    half_roll = float(roll) * 0.5
    half_pitch = float(pitch) * 0.5
    half_yaw = float(yaw) * 0.5
    cr, sr = math.cos(half_roll), math.sin(half_roll)
    cp, sp = math.cos(half_pitch), math.sin(half_pitch)
    cy, sy = math.cos(half_yaw), math.sin(half_yaw)
    return _normalized_quaternion(
        (
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy,
        )
    )


def _xarm6_pose_transform(
    values: list[float] | tuple[float, ...],
) -> tuple[
    tuple[float, float, float],
    tuple[float, float, float, float],
]:
    converted = [float(value) for value in values]
    if len(converted) != 6 or not all(math.isfinite(value) for value in converted):
        raise ValueError("xArm6 Cartesian pose must contain six finite values")
    return (
        tuple(value / 1000.0 for value in converted[:3]),
        _quaternion_from_rpy(*converted[3:6]),
    )


def _quaternion_error_rad(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    normalized_left = _normalized_quaternion(left)
    normalized_right = _normalized_quaternion(right)
    dot = abs(sum(a * b for a, b in zip(normalized_left, normalized_right, strict=True)))
    return 2.0 * math.acos(max(-1.0, min(1.0, dot)))


@dataclass(frozen=True)
class UR5eRG2GripperControllerSettings:
    hostname: str = "192.168.1.172"
    open_position: float = 0.11
    close_position: float = 0.02
    open_width_mm: float = 70.0
    close_width_mm: float = 10.0
    open_force: float = 10.0
    close_force: float = 40.0
    open_settle_sec: float = 1.2
    close_settle_sec: float = 2.0
    rtde_method: str = "function"
    disable_remote_control_check: bool = False

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any] | None,
        *,
        hostname: str | None = None,
    ) -> UR5eRG2GripperControllerSettings:
        gripper = dict(config or {})
        rtde = dict(gripper.get("rtde") or {})
        resolved_hostname = (
            str(hostname or "").strip() or str(rtde.get("hostname") or "").strip() or cls.hostname
        )
        return cls(
            hostname=resolved_hostname,
            open_position=_as_float(gripper.get("open"), cls.open_position),
            close_position=_as_float(gripper.get("close"), cls.close_position),
            open_width_mm=_as_float(rtde.get("open_width_mm"), cls.open_width_mm),
            close_width_mm=_as_float(rtde.get("close_width_mm"), cls.close_width_mm),
            open_force=_as_float(rtde.get("open_force"), cls.open_force),
            close_force=_as_float(rtde.get("close_force"), cls.close_force),
            open_settle_sec=_as_float(rtde.get("open_settle_sec"), cls.open_settle_sec),
            close_settle_sec=_as_float(rtde.get("close_settle_sec"), cls.close_settle_sec),
            rtde_method=str(rtde.get("method") or gripper.get("rtde_method") or cls.rtde_method),
            disable_remote_control_check=_as_bool(
                rtde.get("disable_remote_control_check"),
                cls.disable_remote_control_check,
            ),
        )

    def position_from_width_mm(self, width_mm: float) -> float:
        """Map one physical RG2 width to its configured trajectory position."""
        minimum_width = min(float(self.close_width_mm), float(self.open_width_mm))
        maximum_width = max(float(self.close_width_mm), float(self.open_width_mm))
        requested_width = float(width_mm)
        if not math.isfinite(requested_width):
            raise ValueError("RG2 grasp width must be finite")
        if not minimum_width <= requested_width <= maximum_width:
            raise ValueError(
                f"RG2 grasp width {requested_width:.3f} mm is outside "
                f"[{minimum_width:.3f}, {maximum_width:.3f}] mm"
            )
        width_span = float(self.open_width_mm) - float(self.close_width_mm)
        if abs(width_span) <= 1e-9:
            raise ValueError("RG2 configured width range is zero")
        ratio = (requested_width - float(self.close_width_mm)) / width_span
        return float(self.close_position) + ratio * (
            float(self.open_position) - float(self.close_position)
        )


class UR5eRG2GripperController:
    """UR5e OnRobot RG2 control over UR RTDE custom script execution."""

    def __init__(
        self,
        *,
        hostname: str = "192.168.1.172",
        open_position: float = 0.11,
        close_position: float = 0.02,
        open_width_mm: float = 70.0,
        close_width_mm: float = 10.0,
        open_force: float = 10.0,
        close_force: float = 40.0,
        open_settle_sec: float = 1.2,
        close_settle_sec: float = 2.0,
        rtde_method: str = "function",
        disable_remote_control_check: bool = False,
        rtde_factory: Callable[..., Any] | None = None,
    ) -> None:
        resolved_rtde_method = str(rtde_method or "function").strip().lower()
        if resolved_rtde_method not in {"function", "script"}:
            resolved_rtde_method = "function"
        self.settings = UR5eRG2GripperControllerSettings(
            hostname=str(hostname or "").strip() or "192.168.1.172",
            open_position=float(open_position),
            close_position=float(close_position),
            open_width_mm=float(open_width_mm),
            close_width_mm=float(close_width_mm),
            open_force=float(open_force),
            close_force=float(close_force),
            open_settle_sec=float(open_settle_sec),
            close_settle_sec=float(close_settle_sec),
            rtde_method=resolved_rtde_method,
            disable_remote_control_check=bool(disable_remote_control_check),
        )
        factory = rtde_factory or self._default_rtde_factory()
        self.rtde = factory(
            hostname=self.settings.hostname,
            **self._rtde_factory_kwargs(self.settings),
        )

    @classmethod
    def from_settings(
        cls,
        settings: UR5eRG2GripperControllerSettings,
        *,
        rtde_factory: Callable[..., Any] | None = None,
    ) -> UR5eRG2GripperController:
        return cls(
            hostname=settings.hostname,
            open_position=settings.open_position,
            close_position=settings.close_position,
            open_width_mm=settings.open_width_mm,
            close_width_mm=settings.close_width_mm,
            open_force=settings.open_force,
            close_force=settings.close_force,
            open_settle_sec=settings.open_settle_sec,
            close_settle_sec=settings.close_settle_sec,
            rtde_method=settings.rtde_method,
            disable_remote_control_check=settings.disable_remote_control_check,
            rtde_factory=rtde_factory,
        )

    @staticmethod
    def _default_rtde_factory() -> Callable[..., Any]:
        from rtde_control import RTDEControlInterface

        return RTDEControlInterface

    @staticmethod
    def _rtde_factory_kwargs(settings: UR5eRG2GripperControllerSettings) -> dict[str, Any]:
        if not settings.disable_remote_control_check:
            return {}
        try:
            from rtde_control import RTDEControlInterface
        except ModuleNotFoundError:
            return {"flags": 0}

        return {
            "flags": (
                RTDEControlInterface.FLAGS_DEFAULT
                | RTDEControlInterface.FLAG_DISABLE_REMOTE_CONTROL_CHECK
            )
        }

    @staticmethod
    def script_body(width_mm: float, force: float) -> str:
        return f"""
      local rg = rpc_factory("xmlrpc","http://localhost:41414")
      local ret = rg.rg_grip(0, {float(width_mm)}, {float(force)})
      textmsg("rg_grip returned: ", ret)
    """

    @staticmethod
    def program_body(width_mm: float, force: float, *, name: str = "rg2_cmd") -> str:
        body = "\n".join(
            f"  {line.strip()}"
            for line in UR5eRG2GripperController.script_body(width_mm, force).strip().splitlines()
        )
        return f"def {name}():\n{body}\nend\n"

    @staticmethod
    def inline_program_body(width_mm: float, force: float) -> str:
        body = "\n".join(
            f"  {line.strip()}"
            for line in UR5eRG2GripperController.script_body(width_mm, force).strip().splitlines()
        )
        return f"def program():\n{body}\nend\nrun program\n"

    @staticmethod
    def secondary_program_body(width_mm: float, force: float, *, name: str = "rg2_cmd") -> str:
        body = "\n".join(
            f"  {line.strip()}"
            for line in UR5eRG2GripperController.script_body(width_mm, force).strip().splitlines()
        )
        return f"sec {name}():\n{body}\nend\n"

    def width_mm_from_position(self, position: float) -> float:
        settings = self.settings
        open_position = float(settings.open_position)
        close_position = float(settings.close_position)
        if abs(open_position - close_position) < 1e-9:
            return float(settings.open_width_mm)
        ratio = (float(position) - close_position) / (open_position - close_position)
        width = settings.close_width_mm + ratio * (settings.open_width_mm - settings.close_width_mm)
        return _clamp(width, settings.close_width_mm, settings.open_width_mm)

    def force_for_position(self, position: float) -> float:
        settings = self.settings
        midpoint = (float(settings.open_position) + float(settings.close_position)) * 0.5
        if settings.open_position >= settings.close_position:
            return settings.open_force if float(position) >= midpoint else settings.close_force
        return settings.open_force if float(position) <= midpoint else settings.close_force

    def settle_sec_for_position(self, position: float) -> float:
        settings = self.settings
        midpoint = (float(settings.open_position) + float(settings.close_position)) * 0.5
        if settings.open_position >= settings.close_position:
            return (
                settings.open_settle_sec
                if float(position) >= midpoint
                else settings.close_settle_sec
            )
        return (
            settings.open_settle_sec if float(position) <= midpoint else settings.close_settle_sec
        )

    def command_width(
        self,
        width_mm: float,
        force: float,
        *,
        settle_sec: float = 0.0,
        blocking: bool = True,
    ) -> None:
        if self.settings.rtde_method == "script":
            result = self.rtde.sendCustomScript(self.inline_program_body(width_mm, force))
        else:
            body = self.script_body(width_mm, force)
            result = self.rtde.sendCustomScriptFunction("rg2_cmd", body)
        if result is False:
            raise RuntimeError(f"RTDE RG2 {self.settings.rtde_method} command timed out")
        if blocking and settle_sec > 0.0:
            time.sleep(float(settle_sec))

    def command_position(self, position: float, *, blocking: bool = True) -> float:
        width_mm = self.width_mm_from_position(position)
        self.command_width(
            width_mm,
            self.force_for_position(position),
            settle_sec=self.settle_sec_for_position(position),
            blocking=blocking,
        )
        return width_mm

    def open_gripper(self, *, blocking: bool = True) -> None:
        settings = self.settings
        self.command_width(
            settings.open_width_mm,
            settings.open_force,
            settle_sec=settings.open_settle_sec,
            blocking=blocking,
        )

    def close_gripper(self, *, blocking: bool = True) -> None:
        settings = self.settings
        self.command_width(
            settings.close_width_mm,
            settings.close_force,
            settle_sec=settings.close_settle_sec,
            blocking=blocking,
        )

    def disconnect(self) -> None:
        disconnect = getattr(self.rtde, "disconnect", None)
        if callable(disconnect):
            disconnect()


class HardwarePickPlaceController(GazeboPickPlaceController):
    """Hardware/digital_twin controller with taught function replay."""

    taught_functions_root = TAUGHT_FUNCTIONS_ROOT

    def init(self) -> bool:
        """Initialize physical ROS feedback and TF without MoveIt or Gazebo clients."""
        if self._initialized:
            return True
        if not self._config_valid:
            self._log().error(self._last_failure_message)
            return False
        try:
            import rclpy
            import tf2_ros
            from builtin_interfaces.msg import Duration
            from geometry_msgs.msg import Pose
            from rclpy.action import ActionClient
            from rclpy.callback_groups import ReentrantCallbackGroup
            from rclpy.executors import MultiThreadedExecutor
            from sensor_msgs.msg import JointState
            from std_srvs.srv import Trigger
            from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
        except ImportError as exc:
            self._last_failure_message = (
                f"physical ROS2 imports failed (environment not sourced?): {exc}"
            )
            return False
        try:
            if not rclpy.ok():
                rclpy.init()
            self._rclpy = rclpy
            self._ActionClient = ActionClient
            self._ReentrantCallbackGroup = ReentrantCallbackGroup
            self._MultiThreadedExecutor = MultiThreadedExecutor
            self._Trigger = Trigger
            self._ExecuteTrajectory = None
            self._GetCartesianPath = None
            self._SetEntityState = None
            self._GetEntityState = None
            self._Pose = Pose
            self._JointTrajectory = JointTrajectory
            self._JointTrajectoryPoint = JointTrajectoryPoint
            self._Duration = Duration
            self._JointState = JointState
            self._tf2_ros = tf2_ros
            self._node = rclpy.create_node(self.node_name)
            self._cb_group = ReentrantCallbackGroup()
            self._tf_buffer = tf2_ros.Buffer()
            self._tf_listener = tf2_ros.TransformListener(
                self._tf_buffer,
                self._node,
            )
            self._detect_all_client_legacy = self._node.create_client(
                Trigger,
                self.service_detect_all,
                callback_group=self._cb_group,
            )
            self._node.create_subscription(
                JointState,
                self.joint_states_topic,
                self._on_joint_state,
                50,
            )
            self._cart_client = None
            self._exec_client = None
            self._set_state_client = None
            self._get_state_client = None
            self._gripper_pub = None
            self._arm_pub = None
            self._attach_client = None
            self._detach_client = None
            self._link_attacher_enabled = False
            # Physical controllers receive continuous joint-state and TF traffic while
            # waiting for action goal/result responses.  A second worker prevents that
            # feedback traffic from starving a completed action response.
            self._executor = MultiThreadedExecutor(num_threads=2)
            self._executor.add_node(self._node)
            self._shutdown_requested = False
            self._spin_thread = threading.Thread(
                target=self._spin_executor,
                daemon=True,
            )
            self._spin_thread.start()
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            self._last_failure_message = f"physical ROS2 initialization failed: {exc}"
            return False
        self._initialized = True
        self._last_failure_message = ""
        self._log().info(
            f"[{self.robot_name}] Controller initialized (execution_mode={self.execution_mode})"
        )
        return True

    def _cartesian_move(
        self,
        target: Any,
        label: str = "",
        avoid_collisions: bool = True,
        min_fraction: float = 0.9,
        allow_partial: bool = False,
        time_scale: float | None = None,
    ) -> bool:
        """Fail closed unless the exact physical robot provides direct Cartesian control."""
        _ = (target, avoid_collisions, min_fraction, allow_partial, time_scale)
        self._last_failure_message = (
            f"[{label}] direct physical Cartesian control is not implemented for "
            f"{self.robot_name}"
        )
        self._log().error(self._last_failure_message)
        return False

    def _move_xy_direct(
        self,
        target_x: float,
        target_y: float,
        z: float,
        orientation: Any,
        label_prefix: str,
        *,
        time_scale: float | None = None,
    ) -> bool:
        """Issue one physical Cartesian target without axis-by-axis retries."""
        return self._cartesian_move(
            self._make_pose(target_x, target_y, z, orientation),
            label_prefix,
            time_scale=time_scale,
        )

    def attach_part(
        self,
        model_name: str,
        link: str | None = None,
        part_name: str = "",
    ) -> dict[str, Any]:
        """Record physical mechanical custody without calling a Gazebo service."""
        _ = link
        target = str(model_name or part_name or "").strip()
        if not target:
            return {"success": False, "message": "physical grasp target is empty"}
        self._attached_model = str(model_name or target)
        return {"success": True, "message": f"physical grasp custody recorded for {target}"}

    def detach_part(
        self,
        model_name: str = "",
        link: str | None = None,
        assume_released_if_open: bool = False,
    ) -> dict[str, Any]:
        """Clear physical custody after the hardware gripper opens."""
        _ = (link, assume_released_if_open)
        target = str(model_name or self._attached_model or "held part").strip()
        self._attached_model = None
        self._attached_link = None
        return {"success": True, "message": f"physical release custody recorded for {target}"}

    @staticmethod
    def _safe_name(name: object) -> str:
        value = str(name or "").strip()
        safe = "".join(c if (c.isalnum() or c in "-_") else "_" for c in value)
        return safe or "default"

    @classmethod
    def function_file_path(
        cls,
        robot: str,
        function_name: str,
        name: str,
        *,
        storage_source: str = "hardware",
    ) -> Path:
        return (
            cls.taught_functions_root
            / str(robot or "").strip().lower()
            / str(function_name or "").strip()
            / f"{cls._safe_name(name)}__{str(storage_source or 'hardware').strip()}.json"
        )

    @classmethod
    def load_function(
        cls,
        robot: str,
        function_name: str,
        name: str,
        *,
        storage_source: str = "hardware",
    ) -> dict[str, Any]:
        path = cls.function_file_path(
            robot,
            function_name,
            name,
            storage_source=storage_source,
        )
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, dict):
            raise ValueError(f"taught function file is not a JSON object: {path}")
        return payload

    @staticmethod
    def _step_positions(step: dict[str, Any]) -> list[float]:
        waypoint = dict(step.get("waypoint") or {})
        positions = waypoint.get("joint_positions") or waypoint.get("positions") or []
        return [float(value) for value in positions]

    def replay_step(self, step: dict[str, Any], *, duration_sec: float = 2.0) -> dict[str, Any]:
        primitive = str(step.get("primitive") or "").strip()
        if primitive == "delay":
            params = dict(step.get("params") or {})
            return self.delay(duration_sec=params.get("duration_sec", 0.0))
        if primitive in {"move_cartesian", "move_relative", "move_to_named_pose"}:
            positions = self._step_positions(step)
            if not positions:
                return {"success": False, "message": f"{primitive} step has no waypoint."}
            ok = self.move_joints(positions, duration_sec=duration_sec)
            return {
                "success": bool(ok),
                "message": f"{primitive} waypoint replay {'succeeded' if ok else 'failed'}.",
            }
        if primitive == "grasp_part":
            params = dict(step.get("params") or {})
            ok = self.close_gripper(position=params.get("position"))
            return {"success": bool(ok), "message": "grasp_part"}
        if primitive == "release_part":
            ok = self.open_gripper()
            return {"success": bool(ok), "message": "release_part"}
        if primitive == "open_gripper":
            ok = self.open_gripper()
            return {"success": bool(ok), "message": "open_gripper"}
        if primitive == "close_gripper":
            ok = self.close_gripper()
            return {"success": bool(ok), "message": "close_gripper"}
        return {"success": False, "message": f"unsupported primitive: {primitive}"}

    def replay_function_payload(
        self,
        payload: dict[str, Any],
        *,
        duration_sec: float = 2.0,
    ) -> dict[str, Any]:
        steps = list(payload.get("steps") or [])
        if not steps:
            return {"success": False, "message": "function has no steps."}
        results: list[dict[str, Any]] = []
        for step in steps:
            result = self.replay_step(dict(step), duration_sec=duration_sec)
            results.append(result)
            if not result.get("success"):
                return {
                    "success": False,
                    "message": str(result.get("message") or "step replay failed."),
                    "results": results,
                }
        return {"success": True, "message": "function replay succeeded.", "results": results}

    def replay_function(
        self,
        function_name: str,
        name: str,
        *,
        storage_source: str = "hardware",
        duration_sec: float = 2.0,
    ) -> dict[str, Any]:
        payload = self.load_function(
            self.robot_name,
            function_name,
            name,
            storage_source=storage_source,
        )
        return self.replay_function_payload(payload, duration_sec=duration_sec)

    def taught_function_step(
        self,
        function_name: str,
        name: str,
        step_name: str,
        *,
        storage_source: str = "hardware",
    ) -> tuple[dict[str, Any] | None, Path, str]:
        path = self.function_file_path(
            self.robot_name,
            function_name,
            name,
            storage_source=storage_source,
        )
        try:
            payload = self.load_function(
                self.robot_name,
                function_name,
                name,
                storage_source=storage_source,
            )
        except FileNotFoundError:
            return None, path, f"taught function file not found: {path}"
        except Exception as exc:
            return None, path, f"could not load taught function file {path}: {exc}"
        target_step = str(step_name or "").strip()
        for step in list(payload.get("steps") or []):
            item = dict(step or {})
            if str(item.get("step_name") or "").strip() == target_step:
                return item, path, ""
        return (
            None,
            path,
            f"taught function step not found: {function_name}.{target_step} in {path}",
        )

    def replay_taught_function_step(
        self,
        function_name: str,
        name: str,
        step_name: str,
        *,
        storage_source: str = "hardware",
        duration_sec: float = 2.0,
    ) -> dict[str, Any]:
        step, path, err = self.taught_function_step(
            function_name,
            name,
            step_name,
            storage_source=storage_source,
        )
        if err or step is None:
            return {"success": False, "message": err, "file": str(path)}
        result = self.replay_step(step, duration_sec=duration_sec)
        result.setdefault("file", str(path))
        result.setdefault("step_name", str(step_name or ""))
        return result


class UR5eHardwareController(HardwarePickPlaceController):
    """Config-driven UR5e hardware/digital_twin controller."""

    def __init__(
        self,
        trajectory_topic: str = UR5E_TRAJECTORY_TOPIC,
        joint_states_topic: str = UR5E_JOINT_STATES_TOPIC,
        *,
        controller_config: dict[str, Any] | None = None,
        named_positions: dict[str, Any] | None = None,
        execution_mode: str = "physical",
    ) -> None:
        super().__init__(
            robot_name="ur5e",
            node_name=f"ur5e_controller_{os.getpid()}",
            controller_config=controller_config or {},
            named_positions=named_positions,
            execution_mode=execution_mode,
            arm_joint_names=UR5E_JOINT_NAMES,
            arm_trajectory_topic=trajectory_topic,
            joint_states_topic=joint_states_topic,
        )
        gripper_config = dict(self.controller_config.get("gripper") or {})
        self._ur5e_hardware_trajectory_action = str(
            self.controller_config.get("hardware_trajectory_action") or ""
        ).strip()
        self._ur5e_hardware_cartesian_action = str(
            self.controller_config.get("hardware_cartesian_action") or ""
        ).strip()
        self._ur5e_hardware_insert_action = str(
            self.controller_config.get("hardware_insert_action") or ""
        ).strip()
        self._ur5e_hardware_insertion_demonstration_action = str(
            self.controller_config.get("hardware_insertion_demonstration_action")
            or ""
        ).strip()
        self._ur5e_action_send_timeout_sec = max(
            3.0,
            _as_float(
                self.controller_config.get("hardware_action_send_timeout_sec"),
                10.0,
            ),
        )
        self._ur5e_cartesian_speed_m_s = max(
            0.001,
            _as_float(
                self.controller_config.get("hardware_cartesian_speed_m_s"),
                0.05,
            ),
        )
        self._ur5e_cartesian_acceleration_m_s2 = max(
            0.001,
            _as_float(
                self.controller_config.get("hardware_cartesian_acceleration_m_s2"),
                0.10,
            ),
        )
        self._ur5e_hardware_trajectory_client: Any | None = None
        self._ur5e_hardware_cartesian_client: Any | None = None
        self._ur5e_hardware_insert_client: Any | None = None
        self._ur5e_hardware_insertion_demonstration_client: Any | None = None
        self._move_insert_goal_condition = threading.Condition()
        self._move_insert_dispatch_active = False
        self._active_move_insert_send_future: Any | None = None
        self._active_move_insert_goal_handle: Any | None = None
        self._active_move_insert_result_future: Any | None = None
        self._insertion_demonstration_condition = threading.Condition()
        self._active_insertion_demonstration_send_future: Any | None = None
        self._active_insertion_demonstration_goal_handle: Any | None = None
        self._active_insertion_demonstration_result_future: Any | None = None
        self._active_insertion_demonstration_status: dict[str, Any] = {}
        self._rg2_action_name = str(gripper_config.get("action") or "").strip()
        self._rg2_action_client: Any | None = None
        self._FollowJointTrajectory: Any | None = None
        self._MoveUR5eCartesian: Any | None = None
        self._MoveUR5eInsert: Any | None = None
        self._RecordUR5eInsertionDemonstration: Any | None = None
        self._PoseStamped: Any | None = None
        self._Vector3: Any | None = None

    def _ur5e_rg2_settings(self) -> UR5eRG2GripperControllerSettings:
        settings = getattr(self, "_cached_ur5e_rg2_settings", None)
        if isinstance(settings, UR5eRG2GripperControllerSettings):
            return settings
        gripper_config = dict(
            getattr(self, "controller_config", {}).get("gripper") or {}
        )
        settings = UR5eRG2GripperControllerSettings.from_config(gripper_config)
        self._cached_ur5e_rg2_settings = settings
        return settings

    def _derive_gripper_close_position(
        self,
        *,
        model_name: str = "",
        product_geometry: dict[str, Any] | None = None,
    ) -> float | None:
        """Map explicit physical part width to RG2 position without Gazebo geometry."""
        _ = model_name
        geometry = product_geometry if isinstance(product_geometry, dict) else {}
        grasp_width_m = None
        for key in (
            "grasp_width_m",
            "part_width_m",
            "part_diameter_m",
            "diameter_m",
            "width_m",
        ):
            if key not in geometry:
                continue
            candidate = _as_float(geometry.get(key), 0.0)
            if candidate > 0.0 and math.isfinite(candidate):
                grasp_width_m = candidate
                break
        if grasp_width_m is None:
            self._last_failure_message = (
                "physical grasp geometry has no explicit width; Gazebo model geometry is not "
                "accepted"
            )
            return None
        try:
            return self._ur5e_rg2_settings().position_from_width_mm(
                grasp_width_m * 1000.0
            )
        except ValueError as exc:
            self._last_failure_message = str(exc)
            return None

    def _physical_stl_pick_readiness(
        self,
        product_geometry: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Validate the actual-STL MG hub against the configured stock RG2 fingertips."""
        geometry = dict(product_geometry or {})
        source_stl = str(geometry.get("source_stl") or "").strip()
        if not source_stl:
            return {
                "success": False,
                "message": "physical MG grasp requires the actual source_stl geometry",
            }
        if not Path(source_stl).is_file():
            return {
                "success": False,
                "message": f"physical MG actual source_stl is unavailable: {source_stl}",
            }
        if geometry.get("hub_up") is not True:
            return {
                "success": False,
                "message": "physical MG actual STL grasp requires hub_up=true",
            }
        source_stl_sha256 = str(geometry.get("source_stl_sha256") or "").strip()
        try:
            hash_is_valid = (
                len(source_stl_sha256) == 64 and int(source_stl_sha256, 16) >= 0
            )
        except ValueError:
            hash_is_valid = False
        if not hash_is_valid:
            return {
                "success": False,
                "message": "physical MG actual source_stl_sha256 is missing or invalid",
            }

        numeric_fields: dict[str, float] = {}
        for field_name in (
            "part_height_m",
            "hub_diameter_m",
            "hub_height_m",
            "tooth_diameter_m",
            "tooth_height_m",
            "grasp_width_m",
            "tooth_clearance_m",
            "minimum_hub_overlap_m",
        ):
            try:
                value = float(geometry[field_name])
            except (KeyError, TypeError, ValueError, OverflowError):
                return {
                    "success": False,
                    "message": f"physical MG actual STL geometry is missing {field_name}",
                }
            if not math.isfinite(value) or value <= 0.0:
                return {
                    "success": False,
                    "message": (
                        f"physical MG actual STL geometry has invalid {field_name}={value!r}"
                    ),
                }
            numeric_fields[field_name] = value

        gripper_position = self._derive_gripper_close_position(
            model_name=str(geometry.get("model_name") or ""),
            product_geometry=geometry,
        )
        if gripper_position is None:
            return {
                "success": False,
                "message": self._last_failure_message or "physical MG RG2 width is invalid",
            }

        gripper_config = dict(
            getattr(self, "controller_config", {}).get("gripper") or {}
        )
        stock_fingertip = dict(gripper_config.get("stock_fingertip") or {})
        try:
            open_gripper_position = float(stock_fingertip["open_gripper_position"])
            mg_gripper_close_position = float(
                stock_fingertip["mg_gripper_close_position"]
            )
            open_inner_pad_lower_z = float(
                stock_fingertip["open_inner_pad_lower_z_from_tcp_m"]
            )
            open_inner_pad_upper_z = float(
                stock_fingertip["open_inner_pad_upper_z_from_tcp_m"]
            )
            closed_inner_pad_lower_z = float(
                stock_fingertip["inner_pad_lower_z_from_tcp_m"]
            )
            closed_inner_pad_upper_z = float(
                stock_fingertip["inner_pad_upper_z_from_tcp_m"]
            )
        except (KeyError, TypeError, ValueError, OverflowError):
            return {
                "success": False,
                "message": (
                    "physical MG grasp requires the configured stock RG2 open and closed "
                    "fingertip bands"
                ),
            }
        fingertip_values = (
            open_gripper_position,
            mg_gripper_close_position,
            open_inner_pad_lower_z,
            open_inner_pad_upper_z,
            closed_inner_pad_lower_z,
            closed_inner_pad_upper_z,
        )
        if not all(math.isfinite(value) for value in fingertip_values):
            return {
                "success": False,
                "message": "stock RG2 open or closed fingertip band contains non-finite values",
            }
        if (
            open_inner_pad_lower_z >= open_inner_pad_upper_z
            or closed_inner_pad_lower_z >= closed_inner_pad_upper_z
        ):
            return {
                "success": False,
                "message": "stock RG2 open or closed fingertip band is inverted",
            }
        position_tolerance_m = 5e-6
        if (
            abs(open_gripper_position - 0.11) > position_tolerance_m
            or abs(float(self.gripper_open) - open_gripper_position)
            > position_tolerance_m
        ):
            return {
                "success": False,
                "message": (
                    "physical MG stock RG2 open fingertip band requires gripper "
                    "position 0.11"
                ),
            }
        if (
            abs(mg_gripper_close_position - 0.047) > position_tolerance_m
            or abs(gripper_position - mg_gripper_close_position) > position_tolerance_m
        ):
            return {
                "success": False,
                "message": (
                    "physical MG stock RG2 closed fingertip band requires calculated "
                    "gripper position approximately 0.047"
                ),
            }

        lower_closing_z_displacement_m = (
            closed_inner_pad_lower_z - open_inner_pad_lower_z
        )
        upper_closing_z_displacement_m = (
            closed_inner_pad_upper_z - open_inner_pad_upper_z
        )
        predicted_closing_z_displacement_m = (
            lower_closing_z_displacement_m + upper_closing_z_displacement_m
        ) / 2.0
        closing_displacement_tolerance_m = 5e-5
        if (
            abs(lower_closing_z_displacement_m - upper_closing_z_displacement_m)
            > closing_displacement_tolerance_m
            or abs(predicted_closing_z_displacement_m - (-0.02616))
            > closing_displacement_tolerance_m
        ):
            return {
                "success": False,
                "message": (
                    "physical MG stock RG2 closing displacement must be approximately "
                    "-0.02616 m"
                ),
            }

        tooth_height_m = numeric_fields["tooth_height_m"]
        part_height_m = numeric_fields["part_height_m"]
        tooth_clearance_m = numeric_fields["tooth_clearance_m"]
        minimum_hub_overlap_m = numeric_fields["minimum_hub_overlap_m"]
        pick_tcp_z_offset_from_table_m = (
            tooth_height_m + tooth_clearance_m - closed_inner_pad_lower_z
        )
        mg_pick_z_adjustment_m = float(self.pick_z_adjustments_m.get("MG", 0.0))
        if (
            not math.isfinite(mg_pick_z_adjustment_m)
            or mg_pick_z_adjustment_m < 0.0
            or mg_pick_z_adjustment_m > 0.002
        ):
            return {
                "success": False,
                "message": (
                    "physical MG pick_z_adjustments_m.MG must remain between 0.000 "
                    "and 0.002 m"
                ),
            }
        adjusted_tcp_z_offset_from_table_m = (
            pick_tcp_z_offset_from_table_m + mg_pick_z_adjustment_m
        )
        lowest_closing_endpoint_z_from_tcp_m = min(
            open_inner_pad_lower_z,
            closed_inner_pad_lower_z,
        )
        pad_lower_m = (
            adjusted_tcp_z_offset_from_table_m
            + lowest_closing_endpoint_z_from_tcp_m
        )
        closed_pad_lower_m = (
            adjusted_tcp_z_offset_from_table_m + closed_inner_pad_lower_z
        )
        closed_pad_upper_m = (
            adjusted_tcp_z_offset_from_table_m + closed_inner_pad_upper_z
        )
        hub_overlap_m = max(
            0.0,
            min(closed_pad_upper_m, part_height_m)
            - max(closed_pad_lower_m, tooth_height_m),
        )
        measured_tooth_clearance_m = pad_lower_m - tooth_height_m
        if measured_tooth_clearance_m + 1e-9 < tooth_clearance_m:
            return {
                "success": False,
                "message": (
                    "stock RG2 fingertip tooth clearance is insufficient: "
                    f"{measured_tooth_clearance_m * 1000.0:.2f} mm"
                ),
            }
        if hub_overlap_m + 1e-9 < minimum_hub_overlap_m:
            return {
                "success": False,
                "message": (
                    "stock RG2 fingertip hub overlap is insufficient: "
                    f"{hub_overlap_m * 1000.0:.2f} mm"
                ),
            }
        return {
            "success": True,
            "source_stl": source_stl,
            "source_stl_sha256": source_stl_sha256,
            "hub_up": True,
            **numeric_fields,
            "gripper_close_position": gripper_position,
            "pick_tcp_z_offset_from_table_m": pick_tcp_z_offset_from_table_m,
            "pick_z_adjustment_m": mg_pick_z_adjustment_m,
            "finger_tooth_clearance_m": measured_tooth_clearance_m,
            "finger_hub_overlap_m": hub_overlap_m,
            "open_gripper_position": open_gripper_position,
            "mg_gripper_close_position": mg_gripper_close_position,
            "open_inner_pad_lower_z_from_tcp_m": open_inner_pad_lower_z,
            "open_inner_pad_upper_z_from_tcp_m": open_inner_pad_upper_z,
            "closed_inner_pad_lower_z_from_tcp_m": closed_inner_pad_lower_z,
            "closed_inner_pad_upper_z_from_tcp_m": closed_inner_pad_upper_z,
            "predicted_closing_z_displacement_m": predicted_closing_z_displacement_m,
            "lowest_closing_endpoint_z_from_tcp_m": (
                lowest_closing_endpoint_z_from_tcp_m
            ),
            "inner_pad_lower_z_from_tcp_m": closed_inner_pad_lower_z,
            "inner_pad_upper_z_from_tcp_m": closed_inner_pad_upper_z,
        }

    def init(self) -> bool:
        """Initialize the serialized physical arm and RG2 action clients."""
        if not super().init():
            return False
        if all(
            client is not None
            for client in (
                self._ur5e_hardware_trajectory_client,
                self._ur5e_hardware_cartesian_client,
                self._rg2_action_client,
            )
        ):
            return True
        if not self._ur5e_hardware_trajectory_action:
            self._last_failure_message = "UR5e hardware trajectory action is not configured"
            return False
        if not self._rg2_action_name:
            self._last_failure_message = "RG2 gripper action is not configured"
            return False
        if not self._ur5e_hardware_cartesian_action:
            self._last_failure_message = "UR5e hardware Cartesian action is not configured"
            return False
        try:
            from cais_lab_robotics.action import MoveUR5eCartesian
            from control_msgs.action import FollowJointTrajectory
            from geometry_msgs.msg import PoseStamped

            self._FollowJointTrajectory = FollowJointTrajectory
            self._MoveUR5eCartesian = MoveUR5eCartesian
            self._PoseStamped = PoseStamped
            self._ur5e_hardware_trajectory_client = self._ActionClient(
                self._node,
                FollowJointTrajectory,
                self._ur5e_hardware_trajectory_action,
                callback_group=self._cb_group,
            )
            self._ur5e_hardware_cartesian_client = self._ActionClient(
                self._node,
                MoveUR5eCartesian,
                self._ur5e_hardware_cartesian_action,
                callback_group=self._cb_group,
            )
            self._rg2_action_client = self._ActionClient(
                self._node,
                FollowJointTrajectory,
                self._rg2_action_name,
                callback_group=self._cb_group,
            )
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            self._last_failure_message = f"physical UR5e action client unavailable: {exc}"
            arm_client = self._ur5e_hardware_trajectory_client
            cartesian_client = self._ur5e_hardware_cartesian_client
            self._ur5e_hardware_trajectory_client = None
            self._ur5e_hardware_cartesian_client = None
            self._rg2_action_client = None
            for client in (arm_client, cartesian_client):
                destroy = getattr(client, "destroy", None)
                if callable(destroy):
                    with suppress(AttributeError, RuntimeError):
                        destroy()
            return False
        return True

    def wait_for_services(self, timeout_sec: float = 60.0) -> bool:
        """Wait for direct RTDE arm/RG2 actions and current joint feedback."""
        if not self.init():
            return False
        if self._services_ready:
            return True
        timeout = max(0.0, float(timeout_sec))
        arm_client = self._ur5e_hardware_trajectory_client
        cartesian_client = self._ur5e_hardware_cartesian_client
        gripper_client = self._rg2_action_client
        try:
            arm_ready = bool(
                arm_client is not None
                and arm_client.wait_for_server(timeout_sec=min(8.0, timeout))
            )
            gripper_ready = bool(
                gripper_client is not None
                and gripper_client.wait_for_server(timeout_sec=min(8.0, timeout))
            )
            cartesian_ready = bool(
                cartesian_client is not None
                and cartesian_client.wait_for_server(timeout_sec=min(8.0, timeout))
            )
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            self._last_failure_message = f"physical UR5e action readiness failed: {exc}"
            return False
        if not arm_ready:
            self._last_failure_message = (
                f"{self._ur5e_hardware_trajectory_action} is unavailable"
            )
            return False
        if not gripper_ready:
            self._last_failure_message = f"{self._rg2_action_name} is unavailable"
            return False
        if not cartesian_ready:
            self._last_failure_message = (
                f"{self._ur5e_hardware_cartesian_action} is unavailable"
            )
            return False
        positions, missing = self._get_arm_joint_positions(
            timeout_sec=min(2.0, timeout)
        )
        if positions is None:
            missing_text = ", ".join(missing) if missing else "unknown"
            self._last_failure_message = (
                f"UR5e current joint state is unavailable; missing={missing_text}"
            )
            return False
        self._services_ready = True
        self._last_failure_message = ""
        return True

    def reset_ur5e_hardware_trajectory_client(
        self,
        *,
        timeout_sec: float = 8.0,
    ) -> tuple[bool, str]:
        """Recreate cached arm actions after one serialized RTDE replacement."""
        if (
            self._node is None
            or self._FollowJointTrajectory is None
            or self._MoveUR5eCartesian is None
        ):
            message = "UR5e hardware joint and Cartesian controllers are not initialized"
            self._last_failure_message = message
            return False, message
        if not self._ur5e_hardware_trajectory_action:
            message = "UR5e hardware trajectory action is not configured"
            self._last_failure_message = message
            return False, message
        if not self._ur5e_hardware_cartesian_action:
            message = "UR5e hardware Cartesian action is not configured"
            self._last_failure_message = message
            return False, message
        previous_client = self._ur5e_hardware_trajectory_client
        previous_cartesian_client = self._ur5e_hardware_cartesian_client
        self._ur5e_hardware_trajectory_client = None
        self._ur5e_hardware_cartesian_client = None
        for stale_client in (
            previous_client,
            previous_cartesian_client,
        ):
            destroy = getattr(stale_client, "destroy", None)
            if callable(destroy):
                with suppress(AttributeError, RuntimeError):
                    destroy()

        try:
            client = self._ActionClient(
                self._node,
                self._FollowJointTrajectory,
                self._ur5e_hardware_trajectory_action,
                callback_group=self._cb_group,
            )
            self._ur5e_hardware_trajectory_client = client
            cartesian_client = self._ActionClient(
                self._node,
                self._MoveUR5eCartesian,
                self._ur5e_hardware_cartesian_action,
                callback_group=self._cb_group,
            )
            self._ur5e_hardware_cartesian_client = cartesian_client
            deadline = time.monotonic() + max(0.0, float(timeout_sec))
            ready = bool(
                client.wait_for_server(
                    timeout_sec=max(0.0, deadline - time.monotonic())
                )
            )
            cartesian_ready = bool(
                cartesian_client.wait_for_server(
                    timeout_sec=max(0.0, deadline - time.monotonic())
                )
            )
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            message = (
                "failed to recreate UR5e hardware trajectory action client: "
                f"{type(exc).__name__}: {exc}"
            )
            self._last_failure_message = message
            return False, message
        if not ready:
            message = (
                f"{self._ur5e_hardware_trajectory_action} was not discovered by the "
                "recreated Robot Functions action client"
            )
            self._last_failure_message = message
            return False, message
        if not cartesian_ready:
            message = (
                f"{self._ur5e_hardware_cartesian_action} was not discovered by the "
                "recreated Robot Functions action client"
            )
            self._last_failure_message = message
            return False, message
        self._last_failure_message = ""
        return True, "UR5e Robot Functions joint and Cartesian action clients recreated"

    @staticmethod
    def _wait_ur5e_action_future_without_cancel(
        future: Any,
        timeout_sec: float,
    ) -> Any | None:
        """Wait for one ROS action future without discarding late acceptance."""
        deadline = time.monotonic() + max(0.0, float(timeout_sec))
        while time.monotonic() < deadline and not future.done():
            time.sleep(0.01)
        return future.result() if future.done() else None

    def _wait_ur5e_action_terminal_settlement(
        self,
        goal_handle: Any,
        result_future: Any,
        *,
        timeout_sec: float,
    ) -> Any:
        """Cancel once after timeout, then retain the caller until terminal status."""
        try:
            wrapped = self._wait_ur5e_action_future_without_cancel(
                result_future,
                timeout_sec,
            )
        except (AttributeError, RuntimeError, TypeError, ValueError):
            while True:
                time.sleep(1.0)
        if wrapped is not None:
            return wrapped
        try:
            cancel_future = goal_handle.cancel_goal_async()
            self._wait_ur5e_action_future_without_cancel(cancel_future, 3.0)
        except (AttributeError, RuntimeError, TypeError, ValueError):
            pass
        while not result_future.done():
            time.sleep(0.01)
        try:
            return result_future.result()
        except (AttributeError, RuntimeError, TypeError, ValueError):
            while True:
                time.sleep(1.0)

    @staticmethod
    def _retain_accepted_action_without_terminal_observer() -> None:
        """Retain the runtime motion lock when accepted-goal settlement is unknowable."""
        while True:
            time.sleep(1.0)

    def _cancel_ur5e_hardware_trajectory_goal(
        self,
        goal_handle: Any,
        *,
        label: str,
    ) -> str:
        """Request cancellation and wait briefly for the RTDE goal to become terminal."""
        cancel_accepted = False
        try:
            cancel_future = goal_handle.cancel_goal_async()
            cancel_response = self._wait_future(
                cancel_future,
                timeout_sec=3.0,
                label=f"cancel:{label}",
                timeout_log_level="warning",
            )
            cancel_accepted = bool(
                cancel_response is not None
                and list(getattr(cancel_response, "goals_canceling", []) or [])
            )
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            return f"cancel request failed ({exc})"

        try:
            terminal_future = goal_handle.get_result_async()
            terminal = self._wait_future(
                terminal_future,
                timeout_sec=5.0,
                label=f"cancel-result:{label}",
                timeout_log_level="warning",
            )
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            terminal = None
            terminal_error = str(exc)
        else:
            terminal_error = ""
        if terminal is not None:
            try:
                terminal_status = int(getattr(terminal, "status", -1))
            except (TypeError, ValueError):
                terminal_status = -1
            if terminal_status in {4, 5, 6}:
                return f"goal reached terminal status {terminal_status} after cancel request"
        if terminal_error:
            cancel_status = "accepted" if cancel_accepted else "unconfirmed"
            return f"cancel {cancel_status}; result failed ({terminal_error})"
        return (
            "cancel accepted but terminal result was not confirmed"
            if cancel_accepted
            else "cancel request was not confirmed"
        )

    def _command_ur5e_hardware_trajectory_action(  # noqa: C901 - explicit action gates.
        self,
        positions: list[float],
        *,
        duration_sec: float,
        label: str,
    ) -> bool:
        """Send one guarded two-point arm trajectory to the physical RTDE action."""
        if not self.wait_for_services():
            return False
        client = self._ur5e_hardware_trajectory_client
        action_type = self._FollowJointTrajectory
        if client is None or action_type is None:
            self._last_failure_message = "UR5e hardware trajectory action client is unavailable"
            return False
        if len(positions) != len(self.arm_joint_names):
            self._last_failure_message = (
                f"{label} expected {len(self.arm_joint_names)} joints, got {len(positions)}"
            )
            return False
        try:
            target_positions = [float(position) for position in positions]
        except (TypeError, ValueError) as exc:
            self._last_failure_message = f"{label} contains invalid joint values: {exc}"
            return False
        if not all(math.isfinite(position) for position in target_positions):
            self._last_failure_message = f"{label} contains non-finite joint values"
            return False

        current_positions, missing = self._get_arm_joint_positions(timeout_sec=1.0)
        if current_positions is None:
            missing_text = ", ".join(missing) if missing else "unknown"
            self._last_failure_message = (
                f"UR5e current joint state is unavailable; missing={missing_text}"
            )
            return False
        try:
            current_joint_positions = [float(position) for position in current_positions]
        except (TypeError, ValueError):
            current_joint_positions = []
        if len(current_joint_positions) != len(self.arm_joint_names) or not all(
            math.isfinite(position) for position in current_joint_positions
        ):
            self._last_failure_message = "UR5e current joint state contains invalid values"
            return False
        try:
            action_ready = bool(client.wait_for_server(timeout_sec=2.0))
        except (RuntimeError, TypeError, ValueError) as exc:
            self._last_failure_message = (
                f"{self._ur5e_hardware_trajectory_action}: wait failed ({exc})"
            )
            return False
        if not action_ready:
            self._last_failure_message = (
                f"{self._ur5e_hardware_trajectory_action} is unavailable"
            )
            return False

        goal = action_type.Goal()
        goal.trajectory.joint_names = list(self.arm_joint_names)
        start_point = self._JointTrajectoryPoint()
        start_point.positions = current_joint_positions
        start_point.time_from_start = self._Duration(sec=0, nanosec=0)
        target_point = self._JointTrajectoryPoint()
        target_point.positions = target_positions
        duration = max(0.1, float(duration_sec))
        sec = int(duration)
        nsec = int((duration - sec) * 1_000_000_000)
        target_point.time_from_start = self._Duration(sec=sec, nanosec=nsec)
        goal.trajectory.points = [start_point, target_point]

        send_timeout_sec = float(
            getattr(self, "_ur5e_action_send_timeout_sec", 10.0)
        )
        try:
            send_future = client.send_goal_async(goal)
            goal_handle = self._wait_future(
                send_future,
                timeout_sec=send_timeout_sec,
                label=f"send:{label}",
            )
        except (RuntimeError, TypeError, ValueError) as exc:
            self._last_failure_message = (
                f"{self._ur5e_hardware_trajectory_action}: send failed ({exc})"
            )
            return False
        if goal_handle is None:
            self._last_failure_message = (
                f"{self._ur5e_hardware_trajectory_action}: send acknowledgement "
                f"timeout after {send_timeout_sec:.1f}s; goal acceptance is unknown "
                "and physical motion may still be executing"
            )
            return False
        if not bool(getattr(goal_handle, "accepted", False)):
            self._last_failure_message = (
                f"{self._ur5e_hardware_trajectory_action}: goal rejected"
            )
            return False

        try:
            result_future = goal_handle.get_result_async()
            # The RTDE server may lengthen the nominal duration to enforce its
            # velocity, acceleration, and jerk limits before starting motion.
            wrapped = self._wait_future(
                result_future,
                timeout_sec=max(45.0, duration + 45.0),
                label=f"result:{label}",
            )
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            cancel_detail = self._cancel_ur5e_hardware_trajectory_goal(
                goal_handle,
                label=label,
            )
            self._last_failure_message = (
                f"{self._ur5e_hardware_trajectory_action}: result failed ({exc}); "
                f"{cancel_detail}"
            )
            return False
        if wrapped is None:
            cancel_detail = self._cancel_ur5e_hardware_trajectory_goal(
                goal_handle,
                label=label,
            )
            self._last_failure_message = (
                f"{self._ur5e_hardware_trajectory_action}: result timeout; {cancel_detail}"
            )
            return False
        result = getattr(wrapped, "result", None)
        try:
            goal_status = int(getattr(wrapped, "status", -1))
            error_code = int(getattr(result, "error_code", -1))
        except (TypeError, ValueError):
            goal_status = -1
            error_code = -1
        if goal_status != 4 or error_code != 0:
            error_string = str(getattr(result, "error_string", "") or "").strip()
            detail = f"goal_status={goal_status} error_code={error_code}"
            if error_string:
                detail = f"{detail} {error_string}"
            self._last_failure_message = (
                f"{self._ur5e_hardware_trajectory_action}: {detail}"
            )
            return False
        self._last_failure_message = ""
        return True

    def move_joints(self, positions: list[float], duration_sec: float = 2.0) -> bool:
        """Replay physical UR5e joint waypoints through the guarded RTDE action."""
        return self._command_ur5e_hardware_trajectory_action(
            positions,
            duration_sec=duration_sec,
            label="move_joints",
        )

    def _cartesian_move(
        self,
        target: Any,
        label: str = "",
        avoid_collisions: bool = True,
        min_fraction: float = 0.9,
        allow_partial: bool = False,
        time_scale: float | None = None,
    ) -> bool:
        """Send one exact world -> tool0 target to the single RTDE motion owner."""
        _ = (avoid_collisions, min_fraction, allow_partial)
        if not self.wait_for_services():
            return False
        client = self._ur5e_hardware_cartesian_client
        action_type = self._MoveUR5eCartesian
        pose_stamped_type = self._PoseStamped
        if client is None or action_type is None or pose_stamped_type is None:
            self._last_failure_message = "UR5e direct Cartesian action client is unavailable"
            return False
        try:
            values = (
                float(target.position.x),
                float(target.position.y),
                float(target.position.z),
                float(target.orientation.x),
                float(target.orientation.y),
                float(target.orientation.z),
                float(target.orientation.w),
            )
        except (AttributeError, TypeError, ValueError) as exc:
            self._last_failure_message = f"[{label}] invalid Cartesian target: {exc}"
            return False
        if not all(math.isfinite(value) for value in values):
            self._last_failure_message = f"[{label}] Cartesian target contains non-finite values"
            return False

        scale = max(0.05, _as_float(time_scale, self.trajectory_time_scale))
        speed_m_s = min(
            self._ur5e_cartesian_speed_m_s,
            self._ur5e_cartesian_speed_m_s / scale,
        )
        acceleration_m_s2 = min(
            self._ur5e_cartesian_acceleration_m_s2,
            self._ur5e_cartesian_acceleration_m_s2 / (scale * scale),
        )
        goal = action_type.Goal()
        stamped = pose_stamped_type()
        stamped.header.frame_id = "world"
        stamped.header.stamp = self._node.get_clock().now().to_msg()
        stamped.pose.position.x = values[0]
        stamped.pose.position.y = values[1]
        stamped.pose.position.z = values[2]
        stamped.pose.orientation.x = values[3]
        stamped.pose.orientation.y = values[4]
        stamped.pose.orientation.z = values[5]
        stamped.pose.orientation.w = values[6]
        goal.target_tool0_pose = stamped
        goal.speed_m_s = float(speed_m_s)
        goal.acceleration_m_s2 = float(acceleration_m_s2)

        send_timeout_sec = float(
            getattr(self, "_ur5e_action_send_timeout_sec", 10.0)
        )
        try:
            if not client.wait_for_server(timeout_sec=2.0):
                self._last_failure_message = (
                    f"{self._ur5e_hardware_cartesian_action} is unavailable"
                )
                return False
            send_future = client.send_goal_async(goal)
            goal_handle = self._wait_ur5e_action_future_without_cancel(
                send_future,
                send_timeout_sec,
            )
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            self._last_failure_message = (
                f"{self._ur5e_hardware_cartesian_action}: send failed ({exc})"
            )
            return False
        if goal_handle is None:
            while not send_future.done():
                time.sleep(0.01)
            try:
                goal_handle = send_future.result()
            except (AttributeError, RuntimeError, TypeError, ValueError):
                while True:
                    time.sleep(1.0)
        if not bool(getattr(goal_handle, "accepted", False)):
            self._last_failure_message = (
                f"{self._ur5e_hardware_cartesian_action}: goal rejected"
            )
            return False
        try:
            result_future = goal_handle.get_result_async()
            wrapped = self._wait_ur5e_action_terminal_settlement(
                goal_handle,
                result_future,
                timeout_sec=75.0,
            )
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            self._last_failure_message = (
                f"{self._ur5e_hardware_cartesian_action}: result failed ({exc})"
            )
            self._retain_accepted_action_without_terminal_observer()
            return False
        result = getattr(wrapped, "result", None)
        try:
            goal_status = int(getattr(wrapped, "status", -1))
            error_code = int(getattr(result, "error_code", -1))
        except (TypeError, ValueError):
            goal_status = -1
            error_code = -1
        if goal_status != 4 or error_code != 0:
            error_string = str(getattr(result, "error_string", "") or "").strip()
            detail = f"goal_status={goal_status} error_code={error_code}"
            if error_string:
                detail = f"{detail} {error_string}"
            self._last_failure_message = (
                f"{self._ur5e_hardware_cartesian_action}: {detail}"
            )
            return False
        self._last_failure_message = ""
        return True

    def _live_insert_max_timeout_sec(self) -> tuple[float, str]:
        """Read the insertion timeout hard cap from fresh RTDE server status."""
        try:
            status = json.loads(UR5E_RTDE_STATUS_PATH.read_text(encoding="utf-8"))
            updated_at = float(status["updated_at"])
            hard_cap = float(status["insert_max_timeout_sec"])
        except (FileNotFoundError, KeyError, OSError, TypeError, ValueError) as exc:
            return 0.0, f"live UR5e RTDE insertion hard cap is unavailable: {exc}"
        age_sec = time.time() - updated_at
        if not math.isfinite(age_sec) or age_sec < -1.0 or age_sec > 3.0:
            return 0.0, f"UR5e RTDE insertion hard-cap status is stale (age={age_sec:.2f}s)"
        if status.get("insert_action_ready") is not True:
            return 0.0, "UR5e RTDE status does not report insert_action_ready"
        if not math.isfinite(hard_cap) or hard_cap <= 0.0:
            return 0.0, "UR5e RTDE insert_max_timeout_sec is not finite and positive"
        return hard_cap, ""

    def _ensure_move_insert_client_ready(
        self,
        *,
        timeout_sec: float = 2.0,
    ) -> tuple[bool, str]:
        """Create and discover the optional move_insert action client on demand."""
        if not self._ur5e_hardware_insert_action:
            return False, "UR5e hardware move_insert action is not configured"
        if self._MoveUR5eInsert is None or self._Vector3 is None:
            try:
                from cais_lab_robotics.action import MoveUR5eInsert
                from geometry_msgs.msg import Vector3
            except ImportError as exc:
                return False, f"UR5e move_insert action type is unavailable: {exc}"
            self._MoveUR5eInsert = MoveUR5eInsert
            self._Vector3 = Vector3
        if self._ur5e_hardware_insert_client is None:
            try:
                self._ur5e_hardware_insert_client = self._ActionClient(
                    self._node,
                    self._MoveUR5eInsert,
                    self._ur5e_hardware_insert_action,
                    callback_group=self._cb_group,
                )
            except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
                return False, f"could not create UR5e move_insert action client: {exc}"
        try:
            ready = bool(
                self._ur5e_hardware_insert_client.wait_for_server(
                    timeout_sec=max(0.0, float(timeout_sec))
                )
            )
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            return False, f"UR5e move_insert action readiness failed: {exc}"
        if not ready:
            return False, f"{self._ur5e_hardware_insert_action} is unavailable"
        return True, ""

    def _ensure_insertion_demonstration_client_ready(
        self,
        *,
        timeout_sec: float = 2.0,
    ) -> tuple[bool, str]:
        """Create the passive insertion-demonstration action client on demand."""
        action_name = self._ur5e_hardware_insertion_demonstration_action
        if not action_name:
            return False, "UR5e insertion demonstration action is not configured"
        if (
            self._RecordUR5eInsertionDemonstration is None
            or self._PoseStamped is None
        ):
            try:
                from cais_lab_robotics.action import RecordUR5eInsertionDemonstration
                from geometry_msgs.msg import PoseStamped
            except ImportError as exc:
                return False, (
                    "UR5e insertion demonstration action type is unavailable: "
                    f"{exc}"
                )
            self._RecordUR5eInsertionDemonstration = (
                RecordUR5eInsertionDemonstration
            )
            self._PoseStamped = PoseStamped
        if self._ur5e_hardware_insertion_demonstration_client is None:
            try:
                self._ur5e_hardware_insertion_demonstration_client = self._ActionClient(
                    self._node,
                    self._RecordUR5eInsertionDemonstration,
                    action_name,
                    callback_group=self._cb_group,
                )
            except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
                return False, (
                    "could not create UR5e insertion demonstration action client: "
                    f"{exc}"
                )
        try:
            ready = bool(
                self._ur5e_hardware_insertion_demonstration_client.wait_for_server(
                    timeout_sec=max(0.0, float(timeout_sec))
                )
            )
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            return False, f"UR5e insertion demonstration readiness failed: {exc}"
        if not ready:
            return False, f"{action_name} is unavailable"
        return True, ""

    @staticmethod
    def _insertion_demonstration_pose(pose_message: Any) -> dict[str, float]:
        pose = getattr(pose_message, "pose", pose_message)
        return {
            "x": float(pose.position.x),
            "y": float(pose.position.y),
            "z": float(pose.position.z),
            "qx": float(pose.orientation.x),
            "qy": float(pose.orientation.y),
            "qz": float(pose.orientation.z),
            "qw": float(pose.orientation.w),
        }

    @staticmethod
    def _insertion_demonstration_result_payload(wrapped: Any) -> dict[str, Any]:
        result = getattr(wrapped, "result", None)
        if result is None:
            return {
                "success": False,
                "active": False,
                "message": "insertion demonstration result is unavailable",
                "state_uncertain": True,
            }
        try:
            force_bias = [float(value) for value in result.force_bias]
            payload = {
                "success": int(result.error_code) == 0,
                "active": False,
                "goal_status": int(getattr(wrapped, "status", -1)),
                "error_code": int(result.error_code),
                "message": str(result.error_string or ""),
                "state_uncertain": bool(result.state_uncertain),
                "motion_settled": bool(result.motion_settled),
                "recording_id": str(result.recording_id or ""),
                "trace_path": str(result.trace_path or ""),
                "trace_sha256": str(result.trace_sha256 or ""),
                "sample_count": int(result.sample_count),
                "started_at": float(result.started_at),
                "finished_at": float(result.finished_at),
                "baseline_valid": bool(result.baseline_valid),
                "force_bias": force_bias,
                "baseline_force_span_n": float(result.baseline_force_span_n),
                "baseline_torque_span_nm": float(result.baseline_torque_span_nm),
            }
        except (AttributeError, TypeError, ValueError, OverflowError) as exc:
            return {
                "success": False,
                "active": False,
                "message": f"insertion demonstration result is invalid: {exc}",
                "state_uncertain": True,
            }
        return payload

    def start_insertion_demonstration(
        self,
        *,
        recording_id: str,
        part_name: str,
        destination_location: str,
        context_sha256: str,
        expected_start_tool0_pose: dict[str, float],
        max_duration_sec: float = 300.0,
    ) -> dict[str, Any]:
        """Start passive feedback recording without commanding robot motion."""
        ready, message = self._ensure_insertion_demonstration_client_ready()
        if not ready:
            return {"success": False, "active": False, "message": message}
        condition = self._insertion_demonstration_condition
        with condition:
            if (
                self._active_insertion_demonstration_send_future is not None
                or self._active_insertion_demonstration_goal_handle is not None
            ):
                return {
                    "success": False,
                    "active": True,
                    "message": "another insertion demonstration is already active",
                }
            action_type = self._RecordUR5eInsertionDemonstration
            pose_stamped_type = self._PoseStamped
            client = self._ur5e_hardware_insertion_demonstration_client
            if any(value is None for value in (action_type, pose_stamped_type, client)):
                return {
                    "success": False,
                    "active": False,
                    "message": "UR5e insertion demonstration action client is unavailable",
                }
            goal = action_type.Goal()
            goal.recording_id = str(recording_id)
            goal.part_name = str(part_name)
            goal.destination_location = str(destination_location)
            goal.context_sha256 = str(context_sha256)
            goal.max_duration_sec = float(max_duration_sec)
            stamped = pose_stamped_type()
            stamped.header.frame_id = "world"
            stamped.header.stamp = self._node.get_clock().now().to_msg()
            stamped.pose.position.x = float(expected_start_tool0_pose["x"])
            stamped.pose.position.y = float(expected_start_tool0_pose["y"])
            stamped.pose.position.z = float(expected_start_tool0_pose["z"])
            stamped.pose.orientation.x = float(expected_start_tool0_pose["qx"])
            stamped.pose.orientation.y = float(expected_start_tool0_pose["qy"])
            stamped.pose.orientation.z = float(expected_start_tool0_pose["qz"])
            stamped.pose.orientation.w = float(expected_start_tool0_pose["qw"])
            goal.expected_start_tool0_pose = stamped

            def feedback_callback(feedback_message: Any) -> None:
                feedback = getattr(feedback_message, "feedback", feedback_message)
                try:
                    status = {
                        "success": True,
                        "active": True,
                        "recording_id": str(recording_id),
                        "phase": str(feedback.phase or ""),
                        "sample_count": int(feedback.sample_count),
                        "elapsed_sec": float(feedback.elapsed_sec),
                        "actual_tool0_pose": self._insertion_demonstration_pose(
                            feedback.actual_tool0_pose
                        ),
                        "actual_tcp_force": [
                            float(value) for value in feedback.actual_tcp_force
                        ],
                        "actual_tcp_speed": [
                            float(value) for value in feedback.actual_tcp_speed
                        ],
                        "baseline_valid": bool(feedback.baseline_valid),
                        "force_bias": [float(value) for value in feedback.force_bias],
                        "updated_at": time.time(),
                    }
                except (AttributeError, TypeError, ValueError, OverflowError):
                    return
                with condition:
                    self._active_insertion_demonstration_status = status
                    condition.notify_all()

            send_future = client.send_goal_async(
                goal,
                feedback_callback=feedback_callback,
            )
            self._active_insertion_demonstration_send_future = send_future
            self._active_insertion_demonstration_status = {
                "success": True,
                "active": True,
                "recording_id": str(recording_id),
                "phase": "starting",
                "sample_count": 0,
                "baseline_valid": False,
                "updated_at": time.time(),
            }
        try:
            goal_handle = self._wait_ur5e_action_future_without_cancel(
                send_future,
                self._ur5e_action_send_timeout_sec,
            )
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            return {
                "success": False,
                "active": True,
                "recording_id": recording_id,
                "message": f"insertion demonstration acceptance is unknown: {exc}",
            }
        if goal_handle is None:
            return {
                "success": False,
                "active": True,
                "recording_id": recording_id,
                "message": "insertion demonstration acceptance is still pending",
            }
        if not bool(getattr(goal_handle, "accepted", False)):
            with condition:
                self._active_insertion_demonstration_send_future = None
                self._active_insertion_demonstration_status = {}
            return {
                "success": False,
                "active": False,
                "recording_id": recording_id,
                "message": "insertion demonstration goal was rejected",
            }
        try:
            result_future = goal_handle.get_result_async()
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            self._retain_accepted_action_without_terminal_observer()
            raise RuntimeError("unreachable") from exc
        with condition:
            self._active_insertion_demonstration_send_future = None
            self._active_insertion_demonstration_goal_handle = goal_handle
            self._active_insertion_demonstration_result_future = result_future
            condition.notify_all()
            return dict(self._active_insertion_demonstration_status)

    def insertion_demonstration_status(self) -> dict[str, Any]:
        """Return the latest passive recording feedback or terminal result."""
        condition = self._insertion_demonstration_condition
        with condition:
            result_future = self._active_insertion_demonstration_result_future
            status = dict(self._active_insertion_demonstration_status)
        if result_future is not None and result_future.done():
            try:
                terminal = self._insertion_demonstration_result_payload(
                    result_future.result()
                )
            except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
                terminal = {
                    "success": False,
                    "active": False,
                    "state_uncertain": True,
                    "message": f"insertion demonstration result failed: {exc}",
                }
            with condition:
                self._active_insertion_demonstration_status = dict(terminal)
            return terminal
        return status or {
            "success": True,
            "active": False,
            "phase": "",
            "message": "no insertion demonstration is active",
        }

    def stop_insertion_demonstration(self, *, timeout_sec: float = 10.0) -> dict[str, Any]:
        """Stop passive recording and wait for its trace to become terminal."""
        condition = self._insertion_demonstration_condition
        with condition:
            send_future = self._active_insertion_demonstration_send_future
            goal_handle = self._active_insertion_demonstration_goal_handle
            result_future = self._active_insertion_demonstration_result_future
        if goal_handle is None and send_future is not None:
            goal_handle = self._wait_ur5e_action_future_without_cancel(
                send_future,
                self._ur5e_action_send_timeout_sec,
            )
            if goal_handle is not None and bool(getattr(goal_handle, "accepted", False)):
                result_future = goal_handle.get_result_async()
                with condition:
                    self._active_insertion_demonstration_send_future = None
                    self._active_insertion_demonstration_goal_handle = goal_handle
                    self._active_insertion_demonstration_result_future = result_future
        if goal_handle is None or result_future is None:
            return {
                "success": False,
                "active": send_future is not None,
                "message": "no accepted insertion demonstration is available to stop",
            }
        try:
            cancel_future = goal_handle.cancel_goal_async()
            self._wait_ur5e_action_future_without_cancel(cancel_future, 3.0)
            wrapped = self._wait_ur5e_action_future_without_cancel(
                result_future,
                timeout_sec,
            )
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            return {
                "success": False,
                "active": True,
                "message": f"insertion demonstration stop is unresolved: {exc}",
                "state_uncertain": True,
            }
        if wrapped is None:
            return {
                "success": False,
                "active": True,
                "message": "insertion demonstration is still stopping",
                "state_uncertain": True,
            }
        payload = self._insertion_demonstration_result_payload(wrapped)
        with condition:
            self._active_insertion_demonstration_send_future = None
            self._active_insertion_demonstration_goal_handle = None
            self._active_insertion_demonstration_result_future = None
            self._active_insertion_demonstration_status = dict(payload)
            condition.notify_all()
        return payload

    def _claim_move_insert_dispatch(self) -> bool:
        condition = self._move_insert_goal_condition
        with condition:
            if self._move_insert_dispatch_active:
                return False
            self._move_insert_dispatch_active = True
            self._active_move_insert_send_future = None
            self._active_move_insert_goal_handle = None
            self._active_move_insert_result_future = None
            return True

    def _set_pending_move_insert_send_future(self, send_future: Any) -> None:
        condition = self._move_insert_goal_condition
        with condition:
            self._active_move_insert_send_future = send_future
            condition.notify_all()

    def _set_active_move_insert_goal(
        self,
        goal_handle: Any,
        result_future: Any,
    ) -> None:
        condition = self._move_insert_goal_condition
        with condition:
            self._active_move_insert_send_future = None
            self._active_move_insert_goal_handle = goal_handle
            self._active_move_insert_result_future = result_future
            condition.notify_all()

    def _clear_move_insert_goal(self, goal_handle: Any | None) -> None:
        condition = self._move_insert_goal_condition
        with condition:
            if (
                goal_handle is None
                or self._active_move_insert_goal_handle is goal_handle
            ):
                self._active_move_insert_send_future = None
                self._active_move_insert_goal_handle = None
                self._active_move_insert_result_future = None
                self._move_insert_dispatch_active = False
                condition.notify_all()

    @staticmethod
    def _wait_move_insert_terminal_result(
        result_future: Any,
        timeout_sec: float,
    ) -> Any | None:
        """Wait for an insertion result without canceling its shared ROS future."""
        deadline = time.monotonic() + max(0.0, float(timeout_sec))
        while time.monotonic() < deadline and not result_future.done():
            time.sleep(0.01)
        return result_future.result() if result_future.done() else None

    def _retain_move_insert_until_terminal_settlement(
        self,
        goal_handle: Any,
        result_future: Any,
        settlement: dict[str, Any],
    ) -> dict[str, Any]:
        """Keep insertion ownership only while its terminal result is unknown."""
        if bool(settlement.get("terminal", False)):
            return settlement
        while True:
            with self._move_insert_goal_condition:
                dispatch_active = self._move_insert_dispatch_active
            if not dispatch_active:
                return settlement
            if result_future.done():
                try:
                    wrapped = result_future.result()
                    result = getattr(wrapped, "result", None)
                    terminal_status = int(getattr(wrapped, "status", -1))
                    error_code = int(getattr(result, "error_code", -1))
                    result_uncertain = bool(
                        getattr(result, "state_uncertain", True)
                    )
                    motion_settled = bool(
                        getattr(result, "motion_settled", not result_uncertain)
                    )
                    result_message = str(
                        getattr(result, "error_string", "") or ""
                    ).strip()
                    result_trial_id = str(
                        getattr(result, "trial_id", "") or ""
                    )
                    force_mode_stop_acknowledged = bool(
                        getattr(result, "force_mode_stop_acknowledged", False)
                    )
                    servo_stop_acknowledged = bool(
                        getattr(result, "servo_stop_acknowledged", False)
                    )
                    stop_l_command_completed = bool(
                        getattr(result, "stop_l_command_completed", False)
                    )
                    stationary_confirmed = bool(
                        getattr(result, "stationary_confirmed", False)
                    )
                    relief_load_cleared = bool(
                        getattr(result, "relief_load_cleared", False)
                    )
                    relief_backoff_m = float(
                        getattr(result, "relief_backoff_m", math.nan)
                    )
                    relief_planned_backoff_m = float(
                        getattr(result, "relief_planned_backoff_m", math.nan)
                    )
                    total_relief_backoff_m = float(
                        getattr(result, "total_relief_backoff_m", math.nan)
                    )
                    relief_resume_phase = str(
                        getattr(result, "relief_resume_phase", "") or ""
                    )
                    relief_force_mode_stop_acknowledged = bool(
                        getattr(
                            result,
                            "relief_force_mode_stop_acknowledged",
                            False,
                        )
                    )
                    relief_stop_l_command_completed = bool(
                        getattr(result, "relief_stop_l_command_completed", False)
                    )
                    relief_stationary_confirmed = bool(
                        getattr(result, "relief_stationary_confirmed", False)
                    )
                    relief_force_mode_restart_acknowledged = bool(
                        getattr(
                            result,
                            "relief_force_mode_restart_acknowledged",
                            False,
                        )
                    )
                    server_trace_id = str(
                        getattr(result, "server_trace_id", "") or ""
                    )
                    server_trace_path = str(
                        getattr(result, "server_trace_path", "") or ""
                    )
                    server_trace_sha256 = str(
                        getattr(result, "server_trace_sha256", "") or ""
                    )
                    server_trace_status = str(
                        getattr(result, "server_trace_status", "") or ""
                    )
                    server_trace_complete = bool(
                        getattr(result, "server_trace_complete", False)
                    )
                    server_trace_sample_count = int(
                        getattr(result, "server_trace_sample_count", 0) or 0
                    )
                except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
                    self._clear_move_insert_goal(goal_handle)
                    return {
                        "settled": False,
                        "terminal": True,
                        "state_uncertain": True,
                        "message": (
                            "move_insert produced an immutable terminal result that "
                            f"could not be read ({exc}); action ownership was released "
                            "for Hardware Stack recovery"
                        ),
                    }
                terminal = terminal_status in {4, 5, 6}
                self._clear_move_insert_goal(goal_handle)
                uncertain = bool(
                    not terminal or result_uncertain or not motion_settled
                )
                if terminal and motion_settled and not result_uncertain:
                    message = "move_insert terminal settlement was confirmed"
                else:
                    message = result_message or (
                        f"move_insert terminal status {terminal_status} did not prove "
                        "stationary settlement"
                    )
                    message += (
                        "; action ownership was released for Hardware Stack recovery"
                    )
                return {
                    "settled": bool(terminal and motion_settled),
                    "terminal": True,
                    "state_uncertain": uncertain,
                    "message": message,
                    "goal_status": terminal_status,
                    "error_code": error_code,
                    "trial_id": result_trial_id,
                    "force_mode_stop_acknowledged": force_mode_stop_acknowledged,
                    "servo_stop_acknowledged": servo_stop_acknowledged,
                    "stop_l_command_completed": stop_l_command_completed,
                    "stationary_confirmed": stationary_confirmed,
                    "relief_load_cleared": relief_load_cleared,
                    "relief_backoff_m": relief_backoff_m,
                    "relief_planned_backoff_m": relief_planned_backoff_m,
                    "total_relief_backoff_m": total_relief_backoff_m,
                    "relief_resume_phase": relief_resume_phase,
                    "relief_force_mode_stop_acknowledged": (
                        relief_force_mode_stop_acknowledged
                    ),
                    "relief_stop_l_command_completed": (
                        relief_stop_l_command_completed
                    ),
                    "relief_stationary_confirmed": relief_stationary_confirmed,
                    "relief_force_mode_restart_acknowledged": (
                        relief_force_mode_restart_acknowledged
                    ),
                    "server_trace_id": server_trace_id,
                    "server_trace_path": server_trace_path,
                    "server_trace_sha256": server_trace_sha256,
                    "server_trace_status": server_trace_status,
                    "server_trace_complete": server_trace_complete,
                    "server_trace_sample_count": server_trace_sample_count,
                }
            time.sleep(0.05)

    def cancel_move_insert(  # noqa: C901, PLR0915 - explicit cancellation lifecycle.
        self,
        timeout_sec: float = 8.0,
    ) -> dict[str, Any]:
        """Cancel the accepted move_insert goal and wait for its terminal result."""
        try:
            timeout = max(0.1, float(timeout_sec))
        except (TypeError, ValueError, OverflowError):
            timeout = 8.0
        deadline = time.monotonic() + timeout
        condition = self._move_insert_goal_condition
        goal_handle = None
        result_future = None
        dispatch_active = False
        while time.monotonic() < deadline:
            with condition:
                goal_handle = self._active_move_insert_goal_handle
                result_future = self._active_move_insert_result_future
                send_future = getattr(self, "_active_move_insert_send_future", None)
                dispatch_active = self._move_insert_dispatch_active
            if goal_handle is not None or not dispatch_active:
                break
            if send_future is not None and send_future.done():
                try:
                    accepted_handle = send_future.result()
                except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
                    return {
                        "success": False,
                        "canceled": False,
                        "settled": False,
                        "state_uncertain": True,
                        "message": f"move_insert goal acceptance is unknown ({exc})",
                    }
                if accepted_handle is None:
                    return {
                        "success": False,
                        "canceled": False,
                        "settled": False,
                        "state_uncertain": True,
                        "message": "move_insert goal acceptance is still pending",
                    }
                if not bool(getattr(accepted_handle, "accepted", False)):
                    self._clear_move_insert_goal(None)
                    return {
                        "success": True,
                        "canceled": False,
                        "settled": True,
                        "state_uncertain": False,
                        "message": "move_insert goal was rejected before motion",
                    }
                try:
                    accepted_result_future = accepted_handle.get_result_async()
                except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
                    self._set_active_move_insert_goal(accepted_handle, None)
                    return {
                        "success": False,
                        "canceled": False,
                        "settled": False,
                        "state_uncertain": True,
                        "message": f"move_insert terminal result cannot be observed ({exc})",
                    }
                self._set_active_move_insert_goal(
                    accepted_handle,
                    accepted_result_future,
                )
                goal_handle = accepted_handle
                result_future = accepted_result_future
                break
            with condition:
                condition.wait(
                    timeout=max(0.0, min(0.05, deadline - time.monotonic()))
                )
        if goal_handle is None:
            return {
                "success": False,
                "canceled": False,
                "settled": False,
                "state_uncertain": bool(dispatch_active),
                "message": (
                    "move_insert goal acceptance is still pending"
                    if dispatch_active
                    else "no accepted move_insert goal is active"
                ),
            }
        if result_future is None:
            try:
                result_future = goal_handle.get_result_async()
            except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
                return {
                    "success": False,
                    "canceled": False,
                    "settled": False,
                    "state_uncertain": True,
                    "message": f"move_insert terminal result cannot be observed ({exc})",
                }
            self._set_active_move_insert_goal(goal_handle, result_future)

        cancel_accepted = False
        cancel_detail = ""
        try:
            cancel_response = self._wait_future(
                goal_handle.cancel_goal_async(),
                timeout_sec=min(3.0, max(0.1, deadline - time.monotonic())),
                label="cancel:move_insert",
                timeout_log_level="warning",
            )
            cancel_accepted = bool(
                cancel_response is not None
                and list(getattr(cancel_response, "goals_canceling", []) or [])
            )
            if not cancel_accepted:
                cancel_detail = "cancel request was not confirmed"
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            cancel_detail = f"cancel request failed ({exc})"

        remaining_sec = max(0.0, deadline - time.monotonic())
        try:
            wrapped = self._wait_move_insert_terminal_result(
                result_future,
                remaining_sec,
            )
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            self._clear_move_insert_goal(goal_handle)
            return {
                "success": False,
                "canceled": False,
                "settled": False,
                "terminal": True,
                "state_uncertain": True,
                "message": (
                    f"terminal move_insert result is unavailable ({exc}); action "
                    "ownership was released for Hardware Stack recovery"
                ),
            }
        if wrapped is None:
            return {
                "success": False,
                "canceled": False,
                "settled": False,
                "state_uncertain": True,
                "message": (
                    f"{cancel_detail}; terminal move_insert settlement was not confirmed"
                    if cancel_detail
                    else "cancel accepted but terminal move_insert settlement was not confirmed"
                ),
            }

        try:
            terminal_status = int(getattr(wrapped, "status", -1))
            result = getattr(wrapped, "result", None)
            error_code = int(getattr(result, "error_code", -1))
            result_uncertain = bool(getattr(result, "state_uncertain", False))
            motion_settled = bool(
                getattr(result, "motion_settled", not result_uncertain)
            )
            force_mode_stop_acknowledged = bool(
                getattr(result, "force_mode_stop_acknowledged", False)
            )
            servo_stop_acknowledged = bool(
                getattr(result, "servo_stop_acknowledged", False)
            )
            stop_l_command_completed = bool(
                getattr(result, "stop_l_command_completed", False)
            )
            stationary_confirmed = bool(
                getattr(result, "stationary_confirmed", False)
            )
            relief_load_cleared = bool(
                getattr(result, "relief_load_cleared", False)
            )
            relief_backoff_m = float(
                getattr(result, "relief_backoff_m", math.nan)
            )
            relief_planned_backoff_m = float(
                getattr(result, "relief_planned_backoff_m", math.nan)
            )
            total_relief_backoff_m = float(
                getattr(result, "total_relief_backoff_m", math.nan)
            )
            relief_resume_phase = str(
                getattr(result, "relief_resume_phase", "") or ""
            )
            relief_force_mode_stop_acknowledged = bool(
                getattr(result, "relief_force_mode_stop_acknowledged", False)
            )
            relief_stop_l_command_completed = bool(
                getattr(result, "relief_stop_l_command_completed", False)
            )
            relief_stationary_confirmed = bool(
                getattr(result, "relief_stationary_confirmed", False)
            )
            relief_force_mode_restart_acknowledged = bool(
                getattr(result, "relief_force_mode_restart_acknowledged", False)
            )
            server_trace_id = str(
                getattr(result, "server_trace_id", "") or ""
            )
            server_trace_path = str(
                getattr(result, "server_trace_path", "") or ""
            )
            server_trace_sha256 = str(
                getattr(result, "server_trace_sha256", "") or ""
            )
            server_trace_status = str(
                getattr(result, "server_trace_status", "") or ""
            )
            server_trace_complete = bool(
                getattr(result, "server_trace_complete", False)
            )
            server_trace_sample_count = int(
                getattr(result, "server_trace_sample_count", 0) or 0
            )
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            self._clear_move_insert_goal(goal_handle)
            return {
                "success": False,
                "canceled": False,
                "settled": False,
                "terminal": True,
                "state_uncertain": True,
                "message": (
                    f"terminal move_insert result is unavailable ({exc}); action "
                    "ownership was released for Hardware Stack recovery"
                ),
            }

        terminal = terminal_status in {4, 5, 6}
        settled = bool(terminal and motion_settled)
        if result_future.done():
            # A completed ROS result future is immutable. Keeping controller ownership
            # cannot make an unsettled terminal result become settled and would prevent
            # the normal Hardware Stack recovery path from acquiring its lock.
            self._clear_move_insert_goal(goal_handle)
        canceled = bool(terminal_status == 5 and error_code == -3)
        result_message = str(getattr(result, "error_string", "") or "").strip()
        result_trial_id = str(getattr(result, "trial_id", "") or "")
        state_uncertain = bool(
            not terminal or result_uncertain or not motion_settled
        )
        if settled and not result_uncertain:
            message = (
                "move_insert canceled and stationary settlement confirmed"
                if canceled
                else "move_insert reached a settled terminal result before cancellation"
            )
        else:
            message = result_message or (
                f"move_insert terminal status {terminal_status} did not prove settlement"
            )
            if terminal:
                message += (
                    "; action ownership was released for Hardware Stack recovery"
                )
        return {
            "success": settled,
            "canceled": canceled,
            "settled": settled,
            "terminal": terminal,
            "state_uncertain": state_uncertain,
            "motion_settled": motion_settled,
            "message": message,
            "cancel_accepted": cancel_accepted,
            "goal_status": terminal_status,
            "error_code": error_code,
            "trial_id": result_trial_id,
            "force_mode_stop_acknowledged": force_mode_stop_acknowledged,
            "servo_stop_acknowledged": servo_stop_acknowledged,
            "stop_l_command_completed": stop_l_command_completed,
            "stationary_confirmed": stationary_confirmed,
            "relief_load_cleared": relief_load_cleared,
            "relief_backoff_m": relief_backoff_m,
            "relief_planned_backoff_m": relief_planned_backoff_m,
            "total_relief_backoff_m": total_relief_backoff_m,
            "relief_resume_phase": relief_resume_phase,
            "relief_force_mode_stop_acknowledged": (
                relief_force_mode_stop_acknowledged
            ),
            "relief_stop_l_command_completed": relief_stop_l_command_completed,
            "relief_stationary_confirmed": relief_stationary_confirmed,
            "relief_force_mode_restart_acknowledged": (
                relief_force_mode_restart_acknowledged
            ),
            "server_trace_id": server_trace_id,
            "server_trace_path": server_trace_path,
            "server_trace_sha256": server_trace_sha256,
            "server_trace_status": server_trace_status,
            "server_trace_complete": server_trace_complete,
            "server_trace_sample_count": server_trace_sample_count,
        }

    def move_insert(  # noqa: C901, PLR0912, PLR0913, PLR0915 - fixed action contract.
        self,
        part_name: str,
        calibration_id: str,
        profile_sha256: str,
        hard_caps_sha256: str,
        expected_start_pose: dict[str, Any],
        target_pose: dict[str, Any],
        insertion_axis_world: dict[str, Any],
        contact_speed_m_s: float,
        contact_force_delta_n: float,
        engagement_progress_m: float,
        insertion_force_n: float,
        spiral_radius_m: float,
        spiral_pitch_m: float,
        spiral_speed_m_s: float,
        spiral_acceleration_m_s2: float,
        max_axial_force_n: float,
        max_lateral_force_n: float,
        max_torque_nm: float,
        force_depth_profile: dict[str, Any],
        baseline_force_uncertainty_n: float,
        baseline_torque_uncertainty_nm: float,
        tilt_tolerance_rad: float,
        seated_depth_tolerance_m: float,
        settle_time_sec: float,
        timeout_sec: float,
        trial_id: str = "",
    ) -> dict[str, Any]:
        """Execute one force-limited physical insertion through the RTDE action owner."""
        requested_part = part_name if isinstance(part_name, str) else ""
        if requested_part not in {"SG", "MG", "LG", "SCP", "MCP", "LCP"}:
            return {
                "success": False,
                "message": f"move_insert does not support exact part identifier {requested_part!r}",
                "state_uncertain": False,
            }
        selected_calibration_id = calibration_id if isinstance(calibration_id, str) else ""
        selected_hash = profile_sha256 if isinstance(profile_sha256, str) else ""
        selected_hard_caps_hash = (
            hard_caps_sha256 if isinstance(hard_caps_sha256, str) else ""
        )
        selected_trial_id = trial_id if isinstance(trial_id, str) else ""
        try:
            valid_hash = len(selected_hash) == 64 and int(selected_hash, 16) >= 0
            valid_hard_caps_hash = (
                len(selected_hard_caps_hash) == 64
                and int(selected_hard_caps_hash, 16) >= 0
            )
        except ValueError:
            valid_hash = False
            valid_hard_caps_hash = False
        if (
            not selected_calibration_id
            or not valid_hash
            or not valid_hard_caps_hash
            or not isinstance(trial_id, str)
            or selected_trial_id != selected_trial_id.strip()
        ):
            return {
                "success": False,
                "message": (
                    "move_insert calibration_id, profile_sha256, hard_caps_sha256, "
                    "or trial_id is invalid"
                ),
                "state_uncertain": False,
            }

        def normalized_pose(raw_pose: Any, label: str) -> tuple[dict[str, float], str]:
            if not isinstance(raw_pose, dict):
                return {}, f"{label} must be an object"
            try:
                pose = {
                    field: float(raw_pose[field])
                    for field in ("x", "y", "z", "qx", "qy", "qz", "qw")
                }
                if not all(math.isfinite(value) for value in pose.values()):
                    raise ValueError("contains a non-finite value")
                quaternion = _normalized_quaternion(
                    tuple(pose[field] for field in ("qx", "qy", "qz", "qw"))
                )
            except (KeyError, TypeError, ValueError, OverflowError) as exc:
                return {}, f"{label} is invalid: {exc}"
            pose.update(
                dict(zip(("qx", "qy", "qz", "qw"), quaternion, strict=True))
            )
            return pose, ""

        start, start_error = normalized_pose(
            expected_start_pose,
            "move_insert expected_start_pose",
        )
        target, target_error = normalized_pose(target_pose, "move_insert target_pose")
        if start_error or target_error:
            return {
                "success": False,
                "message": start_error or target_error,
                "state_uncertain": False,
            }
        profile = {
            "contact_speed_m_s": contact_speed_m_s,
            "contact_force_delta_n": contact_force_delta_n,
            "engagement_progress_m": engagement_progress_m,
            "insertion_force_n": insertion_force_n,
            "spiral_radius_m": spiral_radius_m,
            "spiral_pitch_m": spiral_pitch_m,
            "spiral_speed_m_s": spiral_speed_m_s,
            "spiral_acceleration_m_s2": spiral_acceleration_m_s2,
            "max_axial_force_n": max_axial_force_n,
            "max_lateral_force_n": max_lateral_force_n,
            "max_torque_nm": max_torque_nm,
            "tilt_tolerance_rad": tilt_tolerance_rad,
            "seated_depth_tolerance_m": seated_depth_tolerance_m,
            "settle_time_sec": settle_time_sec,
        }
        if not isinstance(force_depth_profile, dict):
            return {
                "success": False,
                "message": "move_insert force_depth_profile must be an object",
                "state_uncertain": False,
            }
        force_depth_fields = (
            "depth_fraction",
            "axial_upper_n",
            "lateral_upper_n",
            "torque_upper_nm",
        )
        try:
            force_depth_values = {
                field_name: [
                    float(value)
                    for value in force_depth_profile[field_name]
                ]
                for field_name in force_depth_fields
            }
        except (KeyError, TypeError, ValueError, OverflowError):
            return {
                "success": False,
                "message": "move_insert force_depth_profile is invalid",
                "state_uncertain": False,
            }
        if (
            any(len(values) != 16 for values in force_depth_values.values())
            or not all(
                math.isfinite(value)
                for values in force_depth_values.values()
                for value in values
            )
        ):
            return {
                "success": False,
                "message": (
                    "move_insert force_depth_profile requires 16 finite synchronized points"
                ),
                "state_uncertain": False,
            }
        try:
            baseline_force_uncertainty_n = float(
                baseline_force_uncertainty_n
            )
            baseline_torque_uncertainty_nm = float(
                baseline_torque_uncertainty_nm
            )
        except (TypeError, ValueError, OverflowError):
            baseline_force_uncertainty_n = math.nan
            baseline_torque_uncertainty_nm = math.nan
        if (
            not math.isfinite(baseline_force_uncertainty_n)
            or baseline_force_uncertainty_n <= 0.0
            or not math.isfinite(baseline_torque_uncertainty_nm)
            or baseline_torque_uncertainty_nm <= 0.0
        ):
            return {
                "success": False,
                "message": "move_insert baseline uncertainty evidence is invalid",
                "state_uncertain": False,
            }
        hard_cap, hard_cap_error = self._live_insert_max_timeout_sec()
        if hard_cap_error:
            return {
                "success": False,
                "message": hard_cap_error,
                "state_uncertain": False,
            }
        derived_timeout, timeout_error = derive_move_insert_timeout_sec(
            start,
            target,
            insertion_axis_world,
            profile,
            part_name=requested_part,
            insert_max_timeout_sec=hard_cap,
        )
        try:
            supplied_timeout = float(timeout_sec)
        except (TypeError, ValueError, OverflowError):
            supplied_timeout = math.nan
        if timeout_error or not math.isfinite(supplied_timeout) or not math.isclose(
            supplied_timeout,
            derived_timeout,
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            return {
                "success": False,
                "message": timeout_error or (
                    "move_insert timeout_sec does not match the timeout derived from "
                    "the frozen poses and profile"
                ),
                "state_uncertain": False,
            }
        if derived_timeout > hard_cap:
            return {
                "success": False,
                "message": (
                    f"derived move_insert timeout {derived_timeout:.3f}s exceeds live "
                    f"RTDE insert_max_timeout_sec {hard_cap:.3f}s"
                ),
                "state_uncertain": False,
            }
        if not self.wait_for_services():
            return {
                "success": False,
                "message": self._last_failure_message or "services not ready",
                "state_uncertain": False,
            }
        insert_ready, insert_error = self._ensure_move_insert_client_ready()
        if not insert_ready:
            return {
                "success": False,
                "message": insert_error,
                "state_uncertain": False,
            }
        client = self._ur5e_hardware_insert_client
        action_type = self._MoveUR5eInsert
        pose_stamped_type = self._PoseStamped
        vector_type = self._Vector3
        if any(value is None for value in (client, action_type, pose_stamped_type, vector_type)):
            return {
                "success": False,
                "message": "UR5e move_insert action client is unavailable",
                "state_uncertain": False,
            }

        def pose_stamped(pose: dict[str, float]) -> Any:
            stamped = pose_stamped_type()
            stamped.header.frame_id = "world"
            stamped.header.stamp = self._node.get_clock().now().to_msg()
            stamped.pose.position.x = pose["x"]
            stamped.pose.position.y = pose["y"]
            stamped.pose.position.z = pose["z"]
            stamped.pose.orientation.x = pose["qx"]
            stamped.pose.orientation.y = pose["qy"]
            stamped.pose.orientation.z = pose["qz"]
            stamped.pose.orientation.w = pose["qw"]
            return stamped

        goal = action_type.Goal()
        goal.trial_id = selected_trial_id
        goal.part_name = requested_part
        goal.calibration_id = selected_calibration_id
        goal.profile_sha256 = selected_hash
        goal.hard_caps_sha256 = selected_hard_caps_hash
        goal.expected_start_tool0_pose = pose_stamped(start)
        goal.target_tool0_pose = pose_stamped(target)
        axis = vector_type()
        try:
            axis.x = float(insertion_axis_world["x"])
            axis.y = float(insertion_axis_world["y"])
            axis.z = float(insertion_axis_world["z"])
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            return {
                "success": False,
                "message": f"move_insert insertion_axis_world is invalid: {exc}",
                "state_uncertain": False,
            }
        goal.insertion_axis_world = axis
        for field_name, value in profile.items():
            setattr(goal, field_name, float(value))
        goal.force_depth_fraction = force_depth_values["depth_fraction"]
        goal.force_depth_axial_upper_n = force_depth_values["axial_upper_n"]
        goal.force_depth_lateral_upper_n = force_depth_values["lateral_upper_n"]
        goal.force_depth_torque_upper_nm = force_depth_values["torque_upper_nm"]
        goal.baseline_force_uncertainty_n = float(
            baseline_force_uncertainty_n
        )
        goal.baseline_torque_uncertainty_nm = float(
            baseline_torque_uncertainty_nm
        )
        goal.timeout_sec = supplied_timeout

        feedback_trace: list[dict[str, Any]] = []
        feedback_trace_lock = threading.Lock()

        def capture_feedback(feedback_message: Any) -> None:
            feedback = getattr(feedback_message, "feedback", feedback_message)
            captured_at = time.time()

            def finite_number(field_name: str) -> float | None:
                try:
                    value = float(getattr(feedback, field_name))
                except (AttributeError, TypeError, ValueError, OverflowError):
                    return None
                return value if math.isfinite(value) else None

            def finite_six(field_name: str) -> list[float] | None:
                try:
                    values = [float(value) for value in getattr(feedback, field_name)]
                except (AttributeError, TypeError, ValueError, OverflowError):
                    return None
                if len(values) != 6 or not all(math.isfinite(value) for value in values):
                    return None
                return values

            def integer(field_name: str) -> int | None:
                try:
                    return int(getattr(feedback, field_name))
                except (AttributeError, TypeError, ValueError, OverflowError):
                    return None

            def pose_stamped_value(field_name: str) -> dict[str, float] | None:
                try:
                    pose = getattr(feedback, field_name).pose
                    return {
                        "x": float(pose.position.x),
                        "y": float(pose.position.y),
                        "z": float(pose.position.z),
                        "qx": float(pose.orientation.x),
                        "qy": float(pose.orientation.y),
                        "qz": float(pose.orientation.z),
                        "qw": float(pose.orientation.w),
                    }
                except (AttributeError, TypeError, ValueError, OverflowError):
                    return None

            trace_item: dict[str, Any] = {
                "timestamp": captured_at,
                "trial_id": str(
                    getattr(feedback, "trial_id", selected_trial_id) or ""
                ),
                "phase": str(getattr(feedback, "phase", "") or ""),
                "insertion_depth_m": finite_number("insertion_depth_m"),
                "depth_error_m": finite_number("depth_error_m"),
                "lateral_offset_m": finite_number("lateral_offset_m"),
                "search_radius_m": finite_number("search_radius_m"),
                "axial_force_n": finite_number("axial_force_n"),
                "lateral_force_n": finite_number("lateral_force_n"),
                "torque_nm": finite_number("torque_nm"),
                "filtered_axial_force_n": finite_number(
                    "filtered_axial_force_n"
                ),
                "filtered_lateral_force_n": finite_number(
                    "filtered_lateral_force_n"
                ),
                "filtered_torque_nm": finite_number("filtered_torque_nm"),
                "tool_flange_torque_nm": finite_number(
                    "tool_flange_torque_nm"
                ),
                "filtered_tool_flange_torque_nm": finite_number(
                    "filtered_tool_flange_torque_nm"
                ),
                "current_force_depth_fraction": finite_number(
                    "current_force_depth_fraction"
                ),
                "force_depth_axial_upper_n": finite_number(
                    "force_depth_axial_upper_n"
                ),
                "force_depth_lateral_upper_n": finite_number(
                    "force_depth_lateral_upper_n"
                ),
                "force_depth_torque_upper_nm": finite_number(
                    "force_depth_torque_upper_nm"
                ),
                "axial_profile_exceeded": bool(
                    getattr(feedback, "axial_profile_exceeded", False)
                ),
                "axial_progress_stalled": bool(
                    getattr(feedback, "axial_progress_stalled", False)
                ),
                "contact_detected": bool(
                    getattr(feedback, "contact_detected", False)
                ),
                "engagement_detected": bool(
                    getattr(feedback, "engagement_detected", False)
                ),
                "seated_detected": bool(
                    getattr(feedback, "seated_detected", False)
                ),
                "soft_overload_detected": bool(
                    getattr(feedback, "soft_overload_detected", False)
                ),
                "soft_overload_reason": str(
                    getattr(feedback, "soft_overload_reason", "") or ""
                ),
                "soft_overload_duration_sec": finite_number(
                    "soft_overload_duration_sec"
                ),
                "relief_cycle_count": integer("relief_cycle_count"),
                "relief_elapsed_sec": finite_number("relief_elapsed_sec"),
                "relief_retreat_m": finite_number("relief_retreat_m"),
                "relief_load_cleared": bool(
                    getattr(feedback, "relief_load_cleared", False)
                ),
                "relief_backoff_m": finite_number("relief_backoff_m"),
                "relief_planned_backoff_m": finite_number(
                    "relief_planned_backoff_m"
                ),
                "total_relief_backoff_m": finite_number(
                    "total_relief_backoff_m"
                ),
                "relief_resume_phase": str(
                    getattr(feedback, "relief_resume_phase", "") or ""
                ),
                "commanded_axial_force_n": finite_number(
                    "commanded_axial_force_n"
                ),
                "commanded_lateral_force_x_n": finite_number(
                    "commanded_lateral_force_x_n"
                ),
                "commanded_lateral_force_y_n": finite_number(
                    "commanded_lateral_force_y_n"
                ),
                "hard_limit_detected": bool(
                    getattr(feedback, "hard_limit_detected", False)
                ),
                "hard_limit_reason": str(
                    getattr(feedback, "hard_limit_reason", "") or ""
                ),
                "limit_trigger": str(
                    getattr(feedback, "limit_trigger", "") or ""
                ),
                "limit_trigger_value": finite_number("limit_trigger_value"),
                "limit_trigger_threshold": finite_number(
                    "limit_trigger_threshold"
                ),
                "limit_trigger_actual_tcp_force": finite_six(
                    "limit_trigger_actual_tcp_force"
                ),
                "limit_trigger_tared_tcp_force": finite_six(
                    "limit_trigger_tared_tcp_force"
                ),
                "tactile_center_valid": bool(
                    getattr(feedback, "tactile_center_valid", False)
                ),
                "tactile_center_tool0_pose": pose_stamped_value(
                    "tactile_center_tool0_pose"
                ),
                "tactile_center_depth_m": finite_number(
                    "tactile_center_depth_m"
                ),
                "tactile_center_confidence": finite_number(
                    "tactile_center_confidence"
                ),
                "tactile_center_evidence_sha256": str(
                    getattr(feedback, "tactile_center_evidence_sha256", "") or ""
                ),
                "scheduled_search_radius_m": finite_number(
                    "scheduled_search_radius_m"
                ),
                "explored_search_radius_m": finite_number(
                    "explored_search_radius_m"
                ),
                "explored_search_angle_rad": finite_number(
                    "explored_search_angle_rad"
                ),
                "disengagement_cycle_count": integer(
                    "disengagement_cycle_count"
                ),
                "last_disengagement_reason": str(
                    getattr(feedback, "last_disengagement_reason", "") or ""
                ),
                "disengagement_withdrawal_m": finite_number(
                    "disengagement_withdrawal_m"
                ),
                "disengagement_contact_cleared": bool(
                    getattr(feedback, "disengagement_contact_cleared", False)
                ),
                "disengagement_force_mode_stop_acknowledged": bool(
                    getattr(
                        feedback,
                        "disengagement_force_mode_stop_acknowledged",
                        False,
                    )
                ),
                "recenter_position_error_m": finite_number(
                    "recenter_position_error_m"
                ),
                "recenter_command_acknowledged": bool(
                    getattr(feedback, "recenter_command_acknowledged", False)
                ),
                "disengagement_stationary_confirmed": bool(
                    getattr(feedback, "disengagement_stationary_confirmed", False)
                ),
                "retare_baseline_consistent": bool(
                    getattr(feedback, "retare_baseline_consistent", False)
                ),
                "force_bias_valid": bool(
                    getattr(feedback, "force_bias_valid", False)
                ),
                "force_bias": finite_six("force_bias"),
                "tared_tcp_force": finite_six("tared_tcp_force"),
                "actual_tcp_force": finite_six("actual_tcp_force"),
                "actual_tcp_speed": finite_six("actual_tcp_speed"),
            }
            stamped = getattr(feedback, "actual_tool0_pose", None)
            pose_message = getattr(stamped, "pose", None)
            try:
                pose = {
                    "x": float(pose_message.position.x),
                    "y": float(pose_message.position.y),
                    "z": float(pose_message.position.z),
                    "qx": float(pose_message.orientation.x),
                    "qy": float(pose_message.orientation.y),
                    "qz": float(pose_message.orientation.z),
                    "qw": float(pose_message.orientation.w),
                }
            except (AttributeError, TypeError, ValueError, OverflowError):
                pose = {}
            if pose and all(math.isfinite(value) for value in pose.values()):
                trace_item["actual_tool0_pose"] = pose
            with feedback_trace_lock:
                if feedback_trace:
                    previous = feedback_trace[-1]
                    elapsed = captured_at - float(previous["timestamp"])
                    previous_depth = previous.get("insertion_depth_m")
                    current_depth = trace_item.get("insertion_depth_m")
                    if (
                        elapsed > 1e-9
                        and isinstance(previous_depth, (int, float))
                        and isinstance(current_depth, (int, float))
                    ):
                        trace_item["axial_speed_m_s"] = (
                            current_depth
                            - float(previous_depth)
                        ) / elapsed
                feedback_trace.append(trace_item)

        def captured_feedback_trace() -> list[dict[str, Any]]:
            with feedback_trace_lock:
                return [dict(item) for item in feedback_trace]

        send_timeout_sec = float(getattr(self, "_ur5e_action_send_timeout_sec", 10.0))
        if not self._claim_move_insert_dispatch():
            return {
                "success": False,
                "message": "another move_insert goal is already active",
                "state_uncertain": False,
                "feedback_trace": [],
            }
        send_attempted = False
        try:
            if not client.wait_for_server(timeout_sec=2.0):
                self._clear_move_insert_goal(None)
                return {
                    "success": False,
                    "message": f"{self._ur5e_hardware_insert_action} is unavailable",
                    "state_uncertain": False,
                    "feedback_trace": [],
                }
            send_attempted = True
            send_future = client.send_goal_async(
                goal,
                feedback_callback=capture_feedback,
            )
            self._set_pending_move_insert_send_future(send_future)
            goal_handle = self._wait_move_insert_terminal_result(
                send_future,
                send_timeout_sec,
            )
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            if not send_attempted:
                self._clear_move_insert_goal(None)
            else:
                with self._move_insert_goal_condition:
                    while self._move_insert_dispatch_active:
                        self._move_insert_goal_condition.wait(timeout=1.0)
            return {
                "success": False,
                "message": (
                    f"{self._ur5e_hardware_insert_action}: send failed ({exc}); "
                    + (
                        "goal acceptance is unknown and physical motion may still be executing"
                        if send_attempted
                        else "goal was not dispatched"
                    )
                ),
                "state_uncertain": send_attempted,
                "feedback_trace": captured_feedback_trace(),
            }
        if goal_handle is None:
            acknowledgement_error = (
                f"{self._ur5e_hardware_insert_action}: send acknowledgement timeout "
                f"after {send_timeout_sec:.1f}s"
            )
            while not send_future.done():
                time.sleep(0.01)
            try:
                goal_handle = send_future.result()
            except (AttributeError, RuntimeError, TypeError, ValueError):
                while True:
                    time.sleep(1.0)
            if goal_handle is None:
                while True:
                    time.sleep(1.0)
            if not bool(getattr(goal_handle, "accepted", False)):
                self._clear_move_insert_goal(None)
                return {
                    "success": False,
                    "message": f"{acknowledgement_error}; delayed goal was rejected",
                    "state_uncertain": False,
                    "feedback_trace": captured_feedback_trace(),
                }
            try:
                result_future = goal_handle.get_result_async()
            except (AttributeError, RuntimeError, TypeError, ValueError):
                self._set_active_move_insert_goal(goal_handle, None)
                while True:
                    time.sleep(1.0)
            self._set_active_move_insert_goal(goal_handle, result_future)
            cancel_result = self.cancel_move_insert(timeout_sec=8.0)
            cancel_result = self._retain_move_insert_until_terminal_settlement(
                goal_handle,
                result_future,
                cancel_result,
            )
            return {
                "success": False,
                "message": f"{acknowledgement_error}; {cancel_result['message']}",
                "state_uncertain": bool(
                    cancel_result.get("state_uncertain", True)
                ),
                "motion_settled": bool(cancel_result.get("settled", False)),
                "trial_id": str(
                    cancel_result.get("trial_id") or selected_trial_id
                ),
                "force_mode_stop_acknowledged": bool(
                    cancel_result.get("force_mode_stop_acknowledged", False)
                ),
                "servo_stop_acknowledged": bool(
                    cancel_result.get("servo_stop_acknowledged", False)
                ),
                "stop_l_command_completed": bool(
                    cancel_result.get("stop_l_command_completed", False)
                ),
                "stationary_confirmed": bool(
                    cancel_result.get("stationary_confirmed", False)
                ),
                "relief_load_cleared": bool(
                    cancel_result.get("relief_load_cleared", False)
                ),
                "relief_backoff_m": cancel_result.get("relief_backoff_m"),
                "relief_planned_backoff_m": cancel_result.get(
                    "relief_planned_backoff_m"
                ),
                "total_relief_backoff_m": cancel_result.get(
                    "total_relief_backoff_m"
                ),
                "relief_resume_phase": str(
                    cancel_result.get("relief_resume_phase") or ""
                ),
                "relief_force_mode_stop_acknowledged": bool(
                    cancel_result.get(
                        "relief_force_mode_stop_acknowledged",
                        False,
                    )
                ),
                "relief_stop_l_command_completed": bool(
                    cancel_result.get("relief_stop_l_command_completed", False)
                ),
                "relief_stationary_confirmed": bool(
                    cancel_result.get("relief_stationary_confirmed", False)
                ),
                "relief_force_mode_restart_acknowledged": bool(
                    cancel_result.get(
                        "relief_force_mode_restart_acknowledged",
                        False,
                    )
                ),
                "server_trace_id": str(
                    cancel_result.get("server_trace_id") or ""
                ),
                "server_trace_path": str(
                    cancel_result.get("server_trace_path") or ""
                ),
                "server_trace_sha256": str(
                    cancel_result.get("server_trace_sha256") or ""
                ),
                "server_trace_status": str(
                    cancel_result.get("server_trace_status") or ""
                ),
                "server_trace_complete": bool(
                    cancel_result.get("server_trace_complete", False)
                ),
                "server_trace_sample_count": int(
                    cancel_result.get("server_trace_sample_count", 0) or 0
                ),
                "feedback_trace": captured_feedback_trace(),
            }
        if not bool(getattr(goal_handle, "accepted", False)):
            self._clear_move_insert_goal(None)
            return {
                "success": False,
                "message": f"{self._ur5e_hardware_insert_action}: goal rejected",
                "state_uncertain": False,
                "feedback_trace": captured_feedback_trace(),
            }
        self._set_active_move_insert_goal(goal_handle, None)
        result_future = None
        try:
            result_future = goal_handle.get_result_async()
            self._set_active_move_insert_goal(goal_handle, result_future)
            wrapped = self._wait_move_insert_terminal_result(
                result_future,
                supplied_timeout + 10.0,
            )
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            cancel_result = self.cancel_move_insert(timeout_sec=8.0)
            if result_future is None:
                while True:
                    time.sleep(1.0)
            cancel_result = self._retain_move_insert_until_terminal_settlement(
                goal_handle,
                result_future,
                cancel_result,
            )
            return {
                "success": False,
                "message": (
                    f"{self._ur5e_hardware_insert_action}: result failed ({exc}); "
                    f"{cancel_result['message']}"
                ),
                "state_uncertain": True,
                "motion_settled": bool(cancel_result.get("settled", False)),
                "trial_id": str(
                    cancel_result.get("trial_id") or selected_trial_id
                ),
                "force_mode_stop_acknowledged": bool(
                    cancel_result.get("force_mode_stop_acknowledged", False)
                ),
                "servo_stop_acknowledged": bool(
                    cancel_result.get("servo_stop_acknowledged", False)
                ),
                "stop_l_command_completed": bool(
                    cancel_result.get("stop_l_command_completed", False)
                ),
                "stationary_confirmed": bool(
                    cancel_result.get("stationary_confirmed", False)
                ),
                "relief_load_cleared": bool(
                    cancel_result.get("relief_load_cleared", False)
                ),
                "relief_backoff_m": cancel_result.get("relief_backoff_m"),
                "relief_planned_backoff_m": cancel_result.get(
                    "relief_planned_backoff_m"
                ),
                "total_relief_backoff_m": cancel_result.get(
                    "total_relief_backoff_m"
                ),
                "relief_resume_phase": str(
                    cancel_result.get("relief_resume_phase") or ""
                ),
                "relief_force_mode_stop_acknowledged": bool(
                    cancel_result.get(
                        "relief_force_mode_stop_acknowledged",
                        False,
                    )
                ),
                "relief_stop_l_command_completed": bool(
                    cancel_result.get("relief_stop_l_command_completed", False)
                ),
                "relief_stationary_confirmed": bool(
                    cancel_result.get("relief_stationary_confirmed", False)
                ),
                "relief_force_mode_restart_acknowledged": bool(
                    cancel_result.get(
                        "relief_force_mode_restart_acknowledged",
                        False,
                    )
                ),
                "server_trace_id": str(
                    cancel_result.get("server_trace_id") or ""
                ),
                "server_trace_path": str(
                    cancel_result.get("server_trace_path") or ""
                ),
                "server_trace_sha256": str(
                    cancel_result.get("server_trace_sha256") or ""
                ),
                "server_trace_status": str(
                    cancel_result.get("server_trace_status") or ""
                ),
                "server_trace_complete": bool(
                    cancel_result.get("server_trace_complete", False)
                ),
                "server_trace_sample_count": int(
                    cancel_result.get("server_trace_sample_count", 0) or 0
                ),
                "feedback_trace": captured_feedback_trace(),
            }
        if wrapped is None:
            cancel_result = self.cancel_move_insert(timeout_sec=8.0)
            cancel_result = self._retain_move_insert_until_terminal_settlement(
                goal_handle,
                result_future,
                cancel_result,
            )
            return {
                "success": False,
                "message": (
                    f"{self._ur5e_hardware_insert_action}: result timeout; "
                    f"{cancel_result['message']}"
                ),
                "state_uncertain": True,
                "motion_settled": bool(cancel_result.get("settled", False)),
                "trial_id": str(
                    cancel_result.get("trial_id") or selected_trial_id
                ),
                "force_mode_stop_acknowledged": bool(
                    cancel_result.get("force_mode_stop_acknowledged", False)
                ),
                "servo_stop_acknowledged": bool(
                    cancel_result.get("servo_stop_acknowledged", False)
                ),
                "stop_l_command_completed": bool(
                    cancel_result.get("stop_l_command_completed", False)
                ),
                "stationary_confirmed": bool(
                    cancel_result.get("stationary_confirmed", False)
                ),
                "relief_load_cleared": bool(
                    cancel_result.get("relief_load_cleared", False)
                ),
                "relief_backoff_m": cancel_result.get("relief_backoff_m"),
                "relief_planned_backoff_m": cancel_result.get(
                    "relief_planned_backoff_m"
                ),
                "total_relief_backoff_m": cancel_result.get(
                    "total_relief_backoff_m"
                ),
                "relief_resume_phase": str(
                    cancel_result.get("relief_resume_phase") or ""
                ),
                "relief_force_mode_stop_acknowledged": bool(
                    cancel_result.get(
                        "relief_force_mode_stop_acknowledged",
                        False,
                    )
                ),
                "relief_stop_l_command_completed": bool(
                    cancel_result.get("relief_stop_l_command_completed", False)
                ),
                "relief_stationary_confirmed": bool(
                    cancel_result.get("relief_stationary_confirmed", False)
                ),
                "relief_force_mode_restart_acknowledged": bool(
                    cancel_result.get(
                        "relief_force_mode_restart_acknowledged",
                        False,
                    )
                ),
                "server_trace_id": str(
                    cancel_result.get("server_trace_id") or ""
                ),
                "server_trace_path": str(
                    cancel_result.get("server_trace_path") or ""
                ),
                "server_trace_sha256": str(
                    cancel_result.get("server_trace_sha256") or ""
                ),
                "server_trace_status": str(
                    cancel_result.get("server_trace_status") or ""
                ),
                "server_trace_complete": bool(
                    cancel_result.get("server_trace_complete", False)
                ),
                "server_trace_sample_count": int(
                    cancel_result.get("server_trace_sample_count", 0) or 0
                ),
                "feedback_trace": captured_feedback_trace(),
            }
        result = getattr(wrapped, "result", None)
        try:
            goal_status = int(getattr(wrapped, "status", -1))
            error_code = int(getattr(result, "error_code", -1))
        except (TypeError, ValueError):
            goal_status = -1
            error_code = -1
        state_uncertain = bool(getattr(result, "state_uncertain", False))
        motion_settled = bool(
            getattr(result, "motion_settled", not state_uncertain)
        )
        terminal = goal_status in {4, 5, 6}
        if terminal:
            self._clear_move_insert_goal(goal_handle)
        else:
            settlement = self._retain_move_insert_until_terminal_settlement(
                goal_handle,
                result_future,
                {
                    "settled": False,
                    "terminal": False,
                    "state_uncertain": True,
                    "message": "move_insert terminal settlement is unconfirmed",
                },
            )
            state_uncertain = bool(settlement.get("state_uncertain", True))
            motion_settled = bool(settlement.get("settled", False))
        state_uncertain = bool(
            state_uncertain or not terminal or not motion_settled
        )
        final_tool0_pose_valid = bool(
            getattr(result, "final_tool0_pose_valid", False)
        )
        result_trial_id = str(
            getattr(result, "trial_id", selected_trial_id) or ""
        )
        result_hard_caps_sha256 = str(
            getattr(result, "hard_caps_sha256", "") or ""
        )
        if result_hard_caps_sha256 != selected_hard_caps_hash:
            state_uncertain = True
        response = {
            "success": bool(
                terminal
                and goal_status == 4
                and error_code == 0
                and motion_settled
                and not state_uncertain
            ),
            "message": str(getattr(result, "error_string", "") or "").strip(),
            "terminal": terminal,
            "goal_status": goal_status,
            "state_uncertain": state_uncertain,
            "motion_settled": motion_settled,
            "trial_id": result_trial_id,
            "hard_caps_sha256": result_hard_caps_sha256,
            "final_phase": str(getattr(result, "final_phase", "") or ""),
            "final_tool0_pose_valid": final_tool0_pose_valid,
            "error_code": error_code,
            "final_insertion_depth_m": float(
                getattr(result, "final_insertion_depth_m", math.nan)
            ),
            "final_depth_error_m": float(
                getattr(result, "final_depth_error_m", math.nan)
            ),
            "final_lateral_offset_m": float(
                getattr(result, "final_lateral_offset_m", math.nan)
            ),
            "final_tilt_error_rad": float(
                getattr(result, "final_tilt_error_rad", math.nan)
            ),
            "final_search_radius_m": float(
                getattr(result, "final_search_radius_m", math.nan)
            ),
            "peak_axial_force_n": float(
                getattr(result, "peak_axial_force_n", math.nan)
            ),
            "peak_lateral_force_n": float(
                getattr(result, "peak_lateral_force_n", math.nan)
            ),
            "peak_torque_nm": float(getattr(result, "peak_torque_nm", math.nan)),
            "peak_filtered_axial_force_n": float(
                getattr(result, "peak_filtered_axial_force_n", math.nan)
            ),
            "peak_filtered_lateral_force_n": float(
                getattr(result, "peak_filtered_lateral_force_n", math.nan)
            ),
            "peak_filtered_torque_nm": float(
                getattr(result, "peak_filtered_torque_nm", math.nan)
            ),
            "peak_tool_flange_torque_nm": float(
                getattr(result, "peak_tool_flange_torque_nm", math.nan)
            ),
            "contact_detected": bool(getattr(result, "contact_detected", False)),
            "engagement_detected": bool(
                getattr(result, "engagement_detected", False)
            ),
            "seated_detected": bool(getattr(result, "seated_detected", False)),
            "soft_overload_detected": bool(
                getattr(result, "soft_overload_detected", False)
            ),
            "soft_overload_recovered": bool(
                getattr(result, "soft_overload_recovered", False)
            ),
            "relief_exhausted": bool(
                getattr(result, "relief_exhausted", False)
            ),
            "relief_cycle_count": int(
                getattr(result, "relief_cycle_count", 0) or 0
            ),
            "last_soft_overload_reason": str(
                getattr(result, "last_soft_overload_reason", "") or ""
            ),
            "relief_load_cleared": bool(
                getattr(result, "relief_load_cleared", False)
            ),
            "relief_backoff_m": float(
                getattr(result, "relief_backoff_m", math.nan)
            ),
            "relief_planned_backoff_m": float(
                getattr(result, "relief_planned_backoff_m", math.nan)
            ),
            "total_relief_backoff_m": float(
                getattr(result, "total_relief_backoff_m", math.nan)
            ),
            "relief_resume_phase": str(
                getattr(result, "relief_resume_phase", "") or ""
            ),
            "relief_force_mode_stop_acknowledged": bool(
                getattr(result, "relief_force_mode_stop_acknowledged", False)
            ),
            "relief_stop_l_command_completed": bool(
                getattr(result, "relief_stop_l_command_completed", False)
            ),
            "relief_stationary_confirmed": bool(
                getattr(result, "relief_stationary_confirmed", False)
            ),
            "relief_force_mode_restart_acknowledged": bool(
                getattr(
                    result,
                    "relief_force_mode_restart_acknowledged",
                    False,
                )
            ),
            "hard_limit_detected": bool(
                getattr(result, "hard_limit_detected", False)
            ),
            "hard_limit_reason": str(
                getattr(result, "hard_limit_reason", "") or ""
            ),
            "limit_trigger": str(getattr(result, "limit_trigger", "") or ""),
            "limit_trigger_value": float(
                getattr(result, "limit_trigger_value", math.nan)
            ),
            "limit_trigger_threshold": float(
                getattr(result, "limit_trigger_threshold", math.nan)
            ),
            "force_bias_valid": bool(getattr(result, "force_bias_valid", False)),
            "force_mode_stop_acknowledged": bool(
                getattr(result, "force_mode_stop_acknowledged", False)
            ),
            "servo_stop_acknowledged": bool(
                getattr(result, "servo_stop_acknowledged", False)
            ),
            "stop_l_command_completed": bool(
                getattr(result, "stop_l_command_completed", False)
            ),
            "stationary_confirmed": bool(
                getattr(result, "stationary_confirmed", False)
            ),
            "server_trace_id": str(
                getattr(result, "server_trace_id", "") or ""
            ),
            "server_trace_path": str(
                getattr(result, "server_trace_path", "") or ""
            ),
            "server_trace_sha256": str(
                getattr(result, "server_trace_sha256", "") or ""
            ),
            "server_trace_status": str(
                getattr(result, "server_trace_status", "") or ""
            ),
            "server_trace_complete": bool(
                getattr(result, "server_trace_complete", False)
            ),
            "server_trace_sample_count": int(
                getattr(result, "server_trace_sample_count", 0) or 0
            ),
            "tactile_center_valid": bool(
                getattr(result, "tactile_center_valid", False)
            ),
            "tactile_center_depth_m": float(
                getattr(result, "tactile_center_depth_m", math.nan)
            ),
            "tactile_center_confidence": float(
                getattr(result, "tactile_center_confidence", math.nan)
            ),
            "tactile_center_evidence_sha256": str(
                getattr(result, "tactile_center_evidence_sha256", "") or ""
            ),
            "scheduled_search_radius_m": float(
                getattr(result, "scheduled_search_radius_m", math.nan)
            ),
            "explored_search_radius_m": float(
                getattr(result, "explored_search_radius_m", math.nan)
            ),
            "explored_search_angle_rad": float(
                getattr(result, "explored_search_angle_rad", math.nan)
            ),
            "disengagement_cycle_count": int(
                getattr(result, "disengagement_cycle_count", 0) or 0
            ),
            "last_disengagement_reason": str(
                getattr(result, "last_disengagement_reason", "") or ""
            ),
            "disengagement_withdrawal_m": float(
                getattr(result, "disengagement_withdrawal_m", math.nan)
            ),
            "disengagement_contact_cleared": bool(
                getattr(result, "disengagement_contact_cleared", False)
            ),
            "disengagement_force_mode_stop_acknowledged": bool(
                getattr(
                    result,
                    "disengagement_force_mode_stop_acknowledged",
                    False,
                )
            ),
            "recenter_position_error_m": float(
                getattr(result, "recenter_position_error_m", math.nan)
            ),
            "recenter_command_acknowledged": bool(
                getattr(result, "recenter_command_acknowledged", False)
            ),
            "disengagement_stationary_confirmed": bool(
                getattr(result, "disengagement_stationary_confirmed", False)
            ),
            "retare_baseline_consistent": bool(
                getattr(result, "retare_baseline_consistent", False)
            ),
            "profile_sha256": selected_hash,
            "feedback_trace": captured_feedback_trace(),
        }
        try:
            result_force_bias = [
                float(value) for value in getattr(result, "force_bias", [])
            ]
        except (TypeError, ValueError, OverflowError):
            result_force_bias = []
        response["force_bias"] = (
            result_force_bias
            if len(result_force_bias) == 6
            and all(math.isfinite(value) for value in result_force_bias)
            else None
        )
        if response["force_bias"] is None:
            response["force_bias_valid"] = False
        try:
            tactile_pose = result.tactile_center_tool0_pose.pose
            response["tactile_center_tool0_pose"] = {
                "x": float(tactile_pose.position.x),
                "y": float(tactile_pose.position.y),
                "z": float(tactile_pose.position.z),
                "qx": float(tactile_pose.orientation.x),
                "qy": float(tactile_pose.orientation.y),
                "qz": float(tactile_pose.orientation.z),
                "qw": float(tactile_pose.orientation.w),
            }
        except (AttributeError, TypeError, ValueError, OverflowError):
            response["tactile_center_tool0_pose"] = None
            response["tactile_center_valid"] = False
        for field_name in (
            "limit_trigger_actual_tcp_force",
            "limit_trigger_tared_tcp_force",
        ):
            try:
                values = [float(value) for value in getattr(result, field_name, [])]
            except (TypeError, ValueError, OverflowError):
                values = []
            response[field_name] = (
                values
                if len(values) == 6
                and all(math.isfinite(value) for value in values)
                else None
            )
        if result_trial_id != selected_trial_id:
            response.update(
                {
                    "success": False,
                    "state_uncertain": True,
                    "message": (
                        "move_insert terminal trial_id does not match the dispatched "
                        f"trial_id {selected_trial_id!r}"
                    ),
                }
            )
        if result_hard_caps_sha256 != selected_hard_caps_hash:
            response.update(
                {
                    "success": False,
                    "state_uncertain": True,
                    "message": (
                        "move_insert terminal hard_caps_sha256 does not match the "
                        "dispatched exact-part hard caps"
                    ),
                }
            )
        if not response["message"]:
            response["message"] = (
                "move_insert completed"
                if response["success"]
                else f"goal_status={goal_status} error_code={error_code}"
            )
        if response["success"] and not (
            response["engagement_detected"] and response["seated_detected"]
        ):
            response.update(
                {
                    "success": False,
                    "state_uncertain": True,
                    "message": (
                        "move_insert action succeeded without confirmed engagement "
                        "and seating evidence"
                    ),
                }
            )
        if final_tool0_pose_valid:
            final_stamped = getattr(result, "final_tool0_pose", None)
            final_frame = str(
                getattr(getattr(final_stamped, "header", None), "frame_id", "") or ""
            ).strip()
            final_pose_message = getattr(final_stamped, "pose", None)
            final_pose, final_pose_error = normalized_pose(
                {
                    "x": getattr(getattr(final_pose_message, "position", None), "x", None),
                    "y": getattr(getattr(final_pose_message, "position", None), "y", None),
                    "z": getattr(getattr(final_pose_message, "position", None), "z", None),
                    "qx": getattr(getattr(final_pose_message, "orientation", None), "x", None),
                    "qy": getattr(getattr(final_pose_message, "orientation", None), "y", None),
                    "qz": getattr(getattr(final_pose_message, "orientation", None), "z", None),
                    "qw": getattr(getattr(final_pose_message, "orientation", None), "w", None),
                },
                "move_insert final_tool0_pose",
            )
            if final_frame == "world" and not final_pose_error:
                response["absolute_position"] = final_pose
                response["final_tool0_pose"] = final_pose
        if response["success"] and "absolute_position" not in response:
            response.update(
                {
                    "success": False,
                    "state_uncertain": True,
                    "message": (
                        "move_insert action succeeded without a valid complete world -> "
                        "tool0 final pose"
                    ),
                }
            )
        self._last_failure_message = "" if response["success"] else str(response["message"])
        return response

    def move_to_named_pose(
        self,
        pose_name: str,
        speed: float | None = None,
    ) -> dict[str, Any]:
        """Move to one configured physical UR5e named position through RTDE."""
        positions = self.named_positions.get(str(pose_name))
        if not isinstance(positions, (list, tuple)) or not positions:
            available = sorted(self.named_positions.keys()) if self.named_positions else []
            return {
                "success": False,
                "message": f"unknown pose '{pose_name}'; available={available}",
            }
        duration_sec = self._scaled_joint_duration(self.named_pose_duration_sec, speed)
        if self._command_ur5e_hardware_trajectory_action(
            list(positions),
            duration_sec=duration_sec,
            label=f"move_to_named_pose:{pose_name}",
        ):
            return {"success": True, "message": f"moved to named pose '{pose_name}'"}
        detail = str(self._last_failure_message or "").strip()
        message = f"failed to move to named pose '{pose_name}'"
        if detail:
            message = f"{message}: {detail}"
        return {"success": False, "message": message}

    def move_home(self, speed: float | None = None) -> dict[str, Any]:
        """Move to the configured physical UR5e home position through RTDE."""
        if not self.wait_for_services():
            detail = str(self._last_failure_message or "").strip()
            message = "services not ready"
            if detail:
                message = f"{message}: {detail}"
            return {"success": False, "message": message}
        home = self.named_positions.get("home")
        if not isinstance(home, (list, tuple)) or not home:
            return {"success": False, "message": "no home pose available"}
        target_positions = [float(position) for position in home]
        duration_sec = self._scaled_joint_duration(self.move_home_duration_sec, speed)
        if self._command_ur5e_hardware_trajectory_action(
            target_positions,
            duration_sec=duration_sec,
            label="move_home",
        ):
            self._last_start_pose = None
            return {"success": True, "message": "moved to named home pose"}
        detail = str(self._last_failure_message or "").strip()
        message = "failed to move to named home pose"
        if detail:
            message = f"{message}: {detail}"
        return {"success": False, "message": message}

    def _command_rg2_gripper_action(self, position: float, label: str) -> bool:
        try:
            target_position = float(position)
        except (TypeError, ValueError, OverflowError):
            self._last_failure_message = "RG2 gripper position is invalid"
            return False
        lower_position = min(float(self.gripper_close), float(self.gripper_open))
        upper_position = max(float(self.gripper_close), float(self.gripper_open))
        if not math.isfinite(target_position):
            self._last_failure_message = "RG2 gripper position must be finite"
            return False
        if not lower_position <= target_position <= upper_position:
            self._last_failure_message = (
                f"RG2 gripper position {target_position:.6f} is outside "
                f"[{lower_position:.6f}, {upper_position:.6f}]"
            )
            return False
        if not self.wait_for_services():
            return False
        client = self._rg2_action_client
        action_type = self._FollowJointTrajectory
        if client is None or action_type is None:
            self._last_failure_message = "RG2 gripper action client is unavailable"
            return False
        try:
            action_ready = bool(client.wait_for_server(timeout_sec=2.0))
        except (RuntimeError, TypeError, ValueError) as exc:
            self._last_failure_message = f"{self._rg2_action_name}: wait failed ({exc})"
            return False
        if not action_ready:
            self._last_failure_message = f"{self._rg2_action_name} is unavailable"
            return False

        goal = action_type.Goal()
        goal.trajectory.joint_names = [self.gripper_joint]
        point = self._JointTrajectoryPoint()
        point.positions = [target_position]
        duration = max(0.05, float(self.gripper_move_time_sec))
        sec = int(duration)
        nsec = int((duration - sec) * 1_000_000_000)
        point.time_from_start = self._Duration(sec=sec, nanosec=nsec)
        goal.trajectory.points = [point]

        try:
            send_future = client.send_goal_async(goal)
        except (RuntimeError, TypeError, ValueError) as exc:
            self._last_failure_message = f"{self._rg2_action_name}: send failed ({exc})"
            return False
        send_timeout_sec = float(
            getattr(self, "_ur5e_action_send_timeout_sec", 10.0)
        )
        try:
            goal_handle = self._wait_ur5e_action_future_without_cancel(
                send_future,
                send_timeout_sec,
            )
        except (RuntimeError, TypeError, ValueError) as exc:
            self._last_failure_message = f"{self._rg2_action_name}: send failed ({exc})"
            return False
        if goal_handle is None:
            while not send_future.done():
                time.sleep(0.01)
            try:
                goal_handle = send_future.result()
            except (AttributeError, RuntimeError, TypeError, ValueError):
                while True:
                    time.sleep(1.0)
        if not bool(getattr(goal_handle, "accepted", False)):
            self._last_failure_message = f"{self._rg2_action_name}: goal rejected"
            return False

        try:
            result_future = goal_handle.get_result_async()
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            self._last_failure_message = f"{self._rg2_action_name}: result failed ({exc})"
            self._retain_accepted_action_without_terminal_observer()
            return False
        result_timeout = max(
            8.0,
            duration + float(self.gripper_feedback_timeout_pad_sec) + 4.0,
        )
        try:
            wrapped = self._wait_ur5e_action_terminal_settlement(
                goal_handle,
                result_future,
                timeout_sec=result_timeout,
            )
        except (RuntimeError, TypeError, ValueError) as exc:
            self._last_failure_message = f"{self._rg2_action_name}: result failed ({exc})"
            self._retain_accepted_action_without_terminal_observer()
            return False
        result = getattr(wrapped, "result", None)
        try:
            goal_status = int(getattr(wrapped, "status", -1))
            error_code = int(getattr(result, "error_code", -1))
        except (TypeError, ValueError):
            goal_status = -1
            error_code = -1
        if goal_status != 4 or error_code != 0:
            error_string = str(getattr(result, "error_string", "") or "").strip()
            detail = f"goal_status={goal_status} error_code={error_code}"
            if error_string:
                detail = f"{detail} {error_string}"
            self._last_failure_message = f"{self._rg2_action_name}: {detail}"
            return False
        self._last_failure_message = ""
        return True

    def open_gripper(self) -> bool:
        return self._command_rg2_gripper_action(self.gripper_open, "open_gripper")

    def close_gripper(self, position: float | None = None) -> bool:
        target = self.gripper_close if position is None else float(position)
        return self._command_rg2_gripper_action(target, "close_gripper")

    def shutdown(self) -> None:
        """Release physical action clients before destroying their ROS node."""
        arm_client = getattr(self, "_ur5e_hardware_trajectory_client", None)
        cartesian_client = getattr(self, "_ur5e_hardware_cartesian_client", None)
        insert_client = getattr(self, "_ur5e_hardware_insert_client", None)
        demonstration_client = getattr(
            self,
            "_ur5e_hardware_insertion_demonstration_client",
            None,
        )
        gripper_client = getattr(self, "_rg2_action_client", None)
        self._ur5e_hardware_trajectory_client = None
        self._ur5e_hardware_cartesian_client = None
        self._ur5e_hardware_insert_client = None
        self._ur5e_hardware_insertion_demonstration_client = None
        self._rg2_action_client = None
        self._FollowJointTrajectory = None
        self._MoveUR5eCartesian = None
        self._MoveUR5eInsert = None
        self._RecordUR5eInsertionDemonstration = None
        self._PoseStamped = None
        self._Vector3 = None
        for client in (
            arm_client,
            cartesian_client,
            insert_client,
            demonstration_client,
            gripper_client,
        ):
            if client is None:
                continue
            destroy = getattr(client, "destroy", None)
            if callable(destroy):
                with suppress(AttributeError, RuntimeError):
                    destroy()
        super().shutdown()


class XArm6HardwareController(HardwarePickPlaceController):
    """Config-driven xArm6 hardware/digital_twin controller."""

    def __init__(
        self,
        *,
        trajectory_topic: str | None = None,
        joint_states_topic: str = XARM6_JOINT_STATES_TOPIC,
        controller_config: dict[str, Any] | None = None,
        named_positions: dict[str, Any] | None = None,
        execution_mode: str = "physical",
    ) -> None:
        super().__init__(
            robot_name="xarm6",
            node_name=f"xarm6_controller_{os.getpid()}",
            controller_config=controller_config or {},
            named_positions=named_positions,
            execution_mode=execution_mode,
            arm_joint_names=XARM6_HARDWARE_JOINT_NAMES,
            arm_trajectory_topic=trajectory_topic,
            joint_states_topic=joint_states_topic,
        )
        self._xarm6_hardware_trajectory_actions = [
            str(value).strip()
            for value in list(
                self.controller_config.get("hardware_trajectory_actions") or []
            )
            if str(value).strip()
        ]
        self._xarm6_hardware_cartesian_service = str(
            self.controller_config.get("hardware_cartesian_service") or ""
        ).strip()
        self._xarm6_hardware_robot_states_topic = str(
            self.controller_config.get("hardware_robot_states_topic")
            or "/xarm6/xarm/robot_states"
        ).strip()
        self._xarm6_set_mode_service = str(
            self.controller_config.get("hardware_set_mode_service")
            or "/xarm6/xarm/set_mode"
        ).strip()
        self._xarm6_set_state_service = str(
            self.controller_config.get("hardware_set_state_service")
            or "/xarm6/xarm/set_state"
        ).strip()
        self._xarm6_controller_list_service = (
            "/xarm6/controller_manager/list_controllers"
        )
        self._xarm6_joint_duration_scale = max(
            1.0,
            _as_float(
                self.controller_config.get("hardware_joint_duration_scale"),
                1.0,
            ),
        )
        self._xarm6_cartesian_speed_mm_s = max(
            1.0,
            _as_float(
                self.controller_config.get("hardware_cartesian_speed_mm_s"),
                50.0,
            ),
        )
        self._xarm6_cartesian_acceleration_mm_s2 = max(
            1.0,
            _as_float(
                self.controller_config.get("hardware_cartesian_acceleration_mm_s2"),
                100.0,
            ),
        )
        self._xarm6_cartesian_position_tolerance_m = max(
            0.0001,
            _as_float(
                self.controller_config.get(
                    "hardware_cartesian_position_tolerance_m"
                ),
                0.003,
            ),
        )
        self._xarm6_cartesian_orientation_tolerance_rad = max(
            0.001,
            _as_float(
                self.controller_config.get(
                    "hardware_cartesian_orientation_tolerance_rad"
                ),
                math.radians(3.0),
            ),
        )
        gripper_config = dict(self.controller_config.get("gripper") or {})
        self._xarm6_gripper_actions = [
            str(value).strip()
            for value in list(gripper_config.get("action_candidates") or [])
            if str(value).strip()
        ]
        self._xarm6_hardware_trajectory_clients: list[tuple[str, Any]] = []
        self._xarm6_gripper_clients: list[tuple[str, Any]] = []
        self._xarm6_hardware_trajectory_action = ""
        self._xarm6_gripper_action = ""
        self._xarm6_hardware_cartesian_client: Any | None = None
        self._xarm6_set_mode_client: Any | None = None
        self._xarm6_set_state_client: Any | None = None
        self._xarm6_controller_list_client: Any | None = None
        self._FollowJointTrajectory: Any | None = None
        self._GripperCommand: Any | None = None
        self._MoveCartesian: Any | None = None
        self._SetInt16: Any | None = None
        self._ListControllers: Any | None = None
        self._xarm6_robot_state: Any | None = None
        self._xarm6_robot_state_received_monotonic = 0.0
        self._xarm6_robot_state_lock = threading.Lock()
        self._xarm6_robot_state_subscription: Any | None = None
        self._xarm6_clients_ready = False
        self._xarm6_open_width_mm = _as_float(
            gripper_config.get("open_width_mm"),
            85.0,
        )
        self._xarm6_close_width_mm = _as_float(
            gripper_config.get("close_width_mm"),
            0.0,
        )

    def _derive_gripper_close_position(
        self,
        *,
        model_name: str = "",
        product_geometry: dict[str, Any] | None = None,
    ) -> float | None:
        """Map an explicit physical part width to the configured xArm6 gripper joint."""
        _ = model_name
        geometry = product_geometry if isinstance(product_geometry, dict) else {}
        grasp_width_m = None
        for key in (
            "grasp_width_m",
            "part_width_m",
            "part_diameter_m",
            "diameter_m",
            "width_m",
        ):
            if key not in geometry:
                continue
            candidate = _as_float(geometry.get(key), 0.0)
            if candidate > 0.0 and math.isfinite(candidate):
                grasp_width_m = candidate
                break
        if grasp_width_m is None:
            self._last_failure_message = (
                "physical grasp geometry has no explicit width; Gazebo model geometry is not "
                "accepted"
            )
            return None
        requested_width_mm = grasp_width_m * 1000.0
        minimum_width = min(self._xarm6_close_width_mm, self._xarm6_open_width_mm)
        maximum_width = max(self._xarm6_close_width_mm, self._xarm6_open_width_mm)
        if not minimum_width <= requested_width_mm <= maximum_width:
            self._last_failure_message = (
                f"xArm6 grasp width {requested_width_mm:.3f} mm is outside "
                f"[{minimum_width:.3f}, {maximum_width:.3f}] mm"
            )
            return None
        width_span = self._xarm6_open_width_mm - self._xarm6_close_width_mm
        if abs(width_span) <= 1e-9:
            self._last_failure_message = "xArm6 configured gripper width range is zero"
            return None
        ratio = (requested_width_mm - self._xarm6_close_width_mm) / width_span
        return float(self.gripper_close) + ratio * (
            float(self.gripper_open) - float(self.gripper_close)
        )

    def init(self) -> bool:
        """Initialize physical xArm6 arm and gripper action clients."""
        if not super().init():
            return False
        if (
            self._xarm6_hardware_trajectory_clients
            and self._xarm6_gripper_clients
            and self._xarm6_hardware_cartesian_client is not None
            and self._xarm6_set_mode_client is not None
            and self._xarm6_set_state_client is not None
            and self._xarm6_controller_list_client is not None
        ):
            return True
        if not self._xarm6_hardware_trajectory_actions:
            self._last_failure_message = (
                "xArm6 hardware trajectory action candidates are not configured"
            )
            return False
        if not self._xarm6_gripper_actions:
            self._last_failure_message = "xArm6 gripper action candidates are not configured"
            return False
        if not self._xarm6_hardware_cartesian_service:
            self._last_failure_message = "xArm6 hardware Cartesian service is not configured"
            return False
        try:
            from control_msgs.action import FollowJointTrajectory, GripperCommand
            from controller_manager_msgs.srv import ListControllers
            from xarm_msgs.msg import RobotMsg
            from xarm_msgs.srv import MoveCartesian, SetInt16

            self._FollowJointTrajectory = FollowJointTrajectory
            self._GripperCommand = GripperCommand
            self._MoveCartesian = MoveCartesian
            self._SetInt16 = SetInt16
            self._ListControllers = ListControllers
            self._xarm6_hardware_trajectory_clients = [
                (
                    action_name,
                    self._ActionClient(
                        self._node,
                        FollowJointTrajectory,
                        action_name,
                        callback_group=self._cb_group,
                    ),
                )
                for action_name in self._xarm6_hardware_trajectory_actions
            ]
            self._xarm6_gripper_clients = [
                (
                    action_name,
                    self._ActionClient(
                        self._node,
                        GripperCommand,
                        action_name,
                        callback_group=self._cb_group,
                    ),
                )
                for action_name in self._xarm6_gripper_actions
            ]
            self._xarm6_hardware_cartesian_client = self._node.create_client(
                MoveCartesian,
                self._xarm6_hardware_cartesian_service,
                callback_group=self._cb_group,
            )
            self._xarm6_set_mode_client = self._node.create_client(
                SetInt16,
                self._xarm6_set_mode_service,
                callback_group=self._cb_group,
            )
            self._xarm6_set_state_client = self._node.create_client(
                SetInt16,
                self._xarm6_set_state_service,
                callback_group=self._cb_group,
            )
            self._xarm6_controller_list_client = self._node.create_client(
                ListControllers,
                self._xarm6_controller_list_service,
                callback_group=self._cb_group,
            )
            self._xarm6_robot_state_subscription = self._node.create_subscription(
                RobotMsg,
                self._xarm6_hardware_robot_states_topic,
                self._xarm6_robot_state_cb,
                10,
                callback_group=self._cb_group,
            )
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            self._last_failure_message = f"physical xArm6 action client unavailable: {exc}"
            self._destroy_xarm6_action_clients()
            return False
        return True

    @staticmethod
    def _ready_action_client(
        candidates: list[tuple[str, Any]],
        *,
        timeout_sec: float,
    ) -> tuple[str, Any] | None:
        deadline = time.monotonic() + max(0.0, float(timeout_sec))
        while True:
            for action_name, client in candidates:
                try:
                    if client.server_is_ready() or client.wait_for_server(
                        timeout_sec=min(0.1, max(0.0, deadline - time.monotonic()))
                    ):
                        return action_name, client
                except (AttributeError, RuntimeError, TypeError, ValueError):
                    continue
            if time.monotonic() >= deadline:
                break
        return None

    def is_usable(self) -> bool:
        """Return whether direct feedback and both physical action clients are ready."""
        return bool(super().is_usable() and self._xarm6_clients_ready)

    def wait_for_services(self, timeout_sec: float = 60.0) -> bool:
        """Wait for direct xArm6 arm/gripper actions and current joint feedback."""
        if not self.init():
            return False
        if self._services_ready and self._xarm6_clients_ready:
            positions, _missing = self._get_arm_joint_positions(
                timeout_sec=min(2.0, max(0.0, float(timeout_sec)))
            )
            return positions is not None
        arm = self._ready_action_client(
            self._xarm6_hardware_trajectory_clients,
            timeout_sec=min(8.0, timeout_sec),
        )
        if arm is None:
            self._last_failure_message = (
                "xArm6 hardware trajectory action is unavailable; tried "
                f"{self._xarm6_hardware_trajectory_actions}"
            )
            return False
        gripper = self._ready_action_client(
            self._xarm6_gripper_clients,
            timeout_sec=min(8.0, timeout_sec),
        )
        if gripper is None:
            self._last_failure_message = (
                "xArm6 gripper action is unavailable; tried "
                f"{self._xarm6_gripper_actions}"
            )
            return False
        cartesian_client = self._xarm6_hardware_cartesian_client
        try:
            cartesian_ready = bool(
                cartesian_client is not None
                and cartesian_client.wait_for_service(
                    timeout_sec=min(8.0, max(0.0, float(timeout_sec)))
                )
            )
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            self._last_failure_message = f"xArm6 Cartesian service readiness failed: {exc}"
            return False
        if not cartesian_ready:
            self._last_failure_message = (
                f"{self._xarm6_hardware_cartesian_service} is unavailable"
            )
            return False
        for service_name, service_client in (
            (self._xarm6_set_mode_service, self._xarm6_set_mode_client),
            (self._xarm6_set_state_service, self._xarm6_set_state_client),
            (
                self._xarm6_controller_list_service,
                self._xarm6_controller_list_client,
            ),
        ):
            try:
                service_ready = bool(
                    service_client is not None
                    and service_client.wait_for_service(
                        timeout_sec=min(3.0, max(0.0, float(timeout_sec)))
                    )
                )
            except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
                self._last_failure_message = (
                    f"xArm6 Mode 1 service readiness failed for {service_name}: {exc}"
                )
                return False
            if not service_ready:
                self._last_failure_message = f"{service_name} is unavailable"
                return False
        self._xarm6_hardware_trajectory_action, self._xarm6_hardware_trajectory_client = arm
        self._xarm6_gripper_action, self._xarm6_gripper_client = gripper
        positions, missing = self._get_arm_joint_positions(
            timeout_sec=min(2.0, max(0.0, float(timeout_sec)))
        )
        if positions is None:
            missing_text = ", ".join(missing) if missing else "unknown"
            self._last_failure_message = self._last_failure_message or (
                f"xArm6 current joint state is unavailable; missing={missing_text}"
            )
            return False
        self._xarm6_clients_ready = True
        self._services_ready = True
        self._last_failure_message = ""
        return True

    def _xarm6_robot_state_cb(self, message: Any) -> None:
        with self._xarm6_robot_state_lock:
            self._xarm6_robot_state = message
            self._xarm6_robot_state_received_monotonic = time.monotonic()

    def _prepare_xarm6_mode_one(  # noqa: C901 - explicit UFactory service gates.
        self,
        *,
        timeout_sec: float = 3.0,
    ) -> bool:
        """Restore and verify UFactory Mode 1 without commanding robot motion."""
        with self._xarm6_robot_state_lock:
            message = self._xarm6_robot_state
            received_at = self._xarm6_robot_state_received_monotonic
        if message is not None and time.monotonic() - received_at <= 2.0:
            try:
                if int(message.mode) == 1 and 0 <= int(message.state) <= 2:
                    controller_ready, controller_message = (
                        self._wait_for_xarm6_trajectory_controller_state(
                            "active",
                            timeout_sec=timeout_sec,
                        )
                    )
                    if controller_ready:
                        self._last_failure_message = ""
                        return True
                    self._last_failure_message = controller_message
                    return False
            except (AttributeError, TypeError, ValueError):
                pass

        service_type = self._SetInt16
        if service_type is None:
            self._last_failure_message = "xArm6 SetInt16 service type is unavailable"
            return False
        for service_name, client, value in (
            (self._xarm6_set_mode_service, self._xarm6_set_mode_client, 1),
            (self._xarm6_set_state_service, self._xarm6_set_state_client, 0),
        ):
            if client is None:
                self._last_failure_message = f"{service_name} client is unavailable"
                return False
            request = service_type.Request()
            request.data = value
            try:
                response = self._wait_future(
                    client.call_async(request),
                    timeout_sec=3.0,
                    label=f"service:{service_name}",
                )
            except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
                self._last_failure_message = f"{service_name} call failed ({exc})"
                return False
            if response is None:
                self._last_failure_message = f"{service_name} response timed out"
                return False
            try:
                return_code = int(getattr(response, "ret", -1))
            except (TypeError, ValueError):
                return_code = -1
            if return_code != 0:
                detail = str(getattr(response, "message", "") or "").strip()
                self._last_failure_message = (
                    f"{service_name}: ret={return_code} {detail}"
                ).strip()
                return False

        deadline = time.monotonic() + max(0.0, float(timeout_sec))
        last_mode = None
        last_state = None
        while time.monotonic() < deadline:
            with self._xarm6_robot_state_lock:
                message = self._xarm6_robot_state
                received_at = self._xarm6_robot_state_received_monotonic
            if message is not None and time.monotonic() - received_at <= 2.0:
                try:
                    last_mode = int(message.mode)
                    last_state = int(message.state)
                except (AttributeError, TypeError, ValueError):
                    pass
                else:
                    if last_mode == 1 and 0 <= last_state <= 2:
                        controller_ready, controller_message = (
                            self._wait_for_xarm6_trajectory_controller_state(
                                "active",
                                timeout_sec=timeout_sec,
                            )
                        )
                        if controller_ready:
                            self._last_failure_message = ""
                            return True
                        self._last_failure_message = controller_message
                        return False
            time.sleep(0.05)
        self._last_failure_message = (
            "xArm6 Mode 1 preparation did not converge; "
            f"mode={last_mode!r} state={last_state!r}"
        )
        return False

    def _set_xarm6_control_value(
        self,
        service_name: str,
        client: Any,
        value: int,
    ) -> tuple[bool, str]:
        service_type = self._SetInt16
        if service_type is None or client is None:
            return False, f"{service_name} client is unavailable"
        request = service_type.Request()
        request.data = int(value)
        try:
            response = self._wait_future(
                client.call_async(request),
                timeout_sec=3.0,
                label=f"service:{service_name}",
            )
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            return False, f"{service_name} call failed ({exc})"
        if response is None:
            return False, f"{service_name} response timed out"
        try:
            return_code = int(getattr(response, "ret", -1))
        except (TypeError, ValueError):
            return_code = -1
        if return_code != 0:
            detail = str(getattr(response, "message", "") or "").strip()
            return False, f"{service_name}: ret={return_code} {detail}".strip()
        return True, "OK"

    def _wait_for_xarm6_trajectory_controller_state(
        self,
        expected_state: str,
        *,
        timeout_sec: float = 5.0,
    ) -> tuple[bool, str]:
        service_type = self._ListControllers
        client = self._xarm6_controller_list_client
        if service_type is None or client is None:
            return False, f"{self._xarm6_controller_list_service} client is unavailable"
        try:
            service_ready = bool(client.wait_for_service(timeout_sec=2.0))
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            return False, f"xArm6 controller list readiness failed ({exc})"
        if not service_ready:
            return False, f"{self._xarm6_controller_list_service} is unavailable"

        expected = str(expected_state).strip().lower()
        deadline = time.monotonic() + max(0.1, float(timeout_sec))
        last_state = "missing"
        last_error = ""
        while time.monotonic() < deadline:
            try:
                response = self._wait_future(
                    client.call_async(service_type.Request()),
                    timeout_sec=min(1.0, max(0.1, deadline - time.monotonic())),
                    label=f"service:{self._xarm6_controller_list_service}",
                )
            except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
                last_error = str(exc)
            else:
                controller = next(
                    (
                        row
                        for row in list(getattr(response, "controller", []))
                        if str(getattr(row, "name", "")).strip()
                        == "xarm6_traj_controller"
                    ),
                    None,
                )
                last_state = (
                    str(getattr(controller, "state", "")).strip().lower()
                    if controller is not None
                    else "missing"
                )
                if last_state == expected:
                    return True, f"xArm6 trajectory controller is {expected}"
            time.sleep(0.05)
        detail = f"state={last_state}"
        if last_error:
            detail += f" last_error={last_error}"
        return False, (
            f"xArm6 trajectory controller did not become {expected}; {detail}"
        )

    def _wait_for_xarm6_mode(
        self,
        expected_mode: int,
        *,
        timeout_sec: float = 3.0,
    ) -> tuple[bool, str]:
        deadline = time.monotonic() + max(0.0, float(timeout_sec))
        last_mode = None
        last_state = None
        while time.monotonic() < deadline:
            with self._xarm6_robot_state_lock:
                message = self._xarm6_robot_state
                received_at = self._xarm6_robot_state_received_monotonic
            if message is not None and time.monotonic() - received_at <= 2.0:
                try:
                    last_mode = int(message.mode)
                    last_state = int(message.state)
                except (AttributeError, TypeError, ValueError):
                    pass
                else:
                    if last_mode == int(expected_mode) and 0 <= last_state <= 2:
                        return True, "OK"
            time.sleep(0.05)
        return False, (
            f"xArm6 mode handoff did not converge to Mode {expected_mode}; "
            f"mode={last_mode!r} state={last_state!r}"
        )

    def _restore_xarm6_trajectory_control(self) -> tuple[bool, str]:
        errors: list[str] = []
        for service_name, client, value in (
            (self._xarm6_set_mode_service, self._xarm6_set_mode_client, 1),
            (self._xarm6_set_state_service, self._xarm6_set_state_client, 0),
        ):
            ok, message = self._set_xarm6_control_value(service_name, client, value)
            if not ok:
                errors.append(message)
        if errors:
            return False, "; ".join(errors)
        ok, message = self._wait_for_xarm6_mode(1)
        if not ok:
            return False, message
        ok, message = self._wait_for_xarm6_trajectory_controller_state("active")
        if not ok:
            return False, message
        return True, "xArm6 trajectory controller Mode 1 restored"

    def _prepare_xarm6_firmware_cartesian_mode(self) -> tuple[bool, str]:
        for service_name, client, value in (
            (self._xarm6_set_mode_service, self._xarm6_set_mode_client, 0),
            (self._xarm6_set_state_service, self._xarm6_set_state_client, 0),
        ):
            ok, message = self._set_xarm6_control_value(service_name, client, value)
            if not ok:
                restore_ok, restore_message = self._restore_xarm6_trajectory_control()
                detail = f"xArm6 Cartesian handoff failed: {message}"
                if not restore_ok:
                    detail += f"; trajectory control restore failed: {restore_message}"
                return False, detail
        ok, message = self._wait_for_xarm6_mode(0)
        if not ok:
            restore_ok, restore_message = self._restore_xarm6_trajectory_control()
            detail = f"xArm6 Cartesian handoff failed: {message}"
            if not restore_ok:
                detail += f"; trajectory control restore failed: {restore_message}"
            return False, detail
        ok, message = self._wait_for_xarm6_trajectory_controller_state("inactive")
        if not ok:
            restore_ok, restore_message = self._restore_xarm6_trajectory_control()
            detail = f"xArm6 Cartesian handoff failed: {message}"
            if not restore_ok:
                detail += f"; trajectory control restore failed: {restore_message}"
            return False, detail
        return True, "xArm6 firmware Cartesian Mode 0 ready"

    def _get_arm_joint_positions(
        self,
        *,
        timeout_sec: float = 0.0,
    ) -> tuple[list[float] | None, list[str]]:
        """Read physical arm positions directly from UFactory robot_states feedback."""
        deadline = time.monotonic() + max(0.0, float(timeout_sec))
        missing = list(self.arm_joint_names)
        while True:
            with self._xarm6_robot_state_lock:
                message = self._xarm6_robot_state
                received_at = self._xarm6_robot_state_received_monotonic

            if message is None:
                detail = "xArm6 robot_states joint feedback has not been received"
            else:
                age_sec = time.monotonic() - received_at
                if age_sec > 2.0:
                    detail = (
                        "xArm6 robot_states joint feedback is stale "
                        f"({age_sec:.2f}s)"
                    )
                else:
                    try:
                        angles = list(message.angle)
                    except (AttributeError, TypeError, ValueError) as exc:
                        detail = (
                            "xArm6 robot_states joint feedback is invalid: "
                            f"angle is unavailable ({exc})"
                        )
                    else:
                        if len(angles) < len(self.arm_joint_names):
                            detail = (
                                "xArm6 robot_states joint feedback is invalid: "
                                f"angle has {len(angles)} values; expected at least "
                                f"{len(self.arm_joint_names)}"
                            )
                        else:
                            try:
                                positions = [
                                    float(value)
                                    for value in angles[: len(self.arm_joint_names)]
                                ]
                            except (TypeError, ValueError, OverflowError) as exc:
                                detail = (
                                    "xArm6 robot_states joint feedback is invalid: "
                                    f"angle[0:6] contains invalid values ({exc})"
                                )
                            else:
                                if all(math.isfinite(value) for value in positions):
                                    self._last_failure_message = ""
                                    return positions, []
                                detail = (
                                    "xArm6 robot_states joint feedback is invalid: "
                                    "angle[0:6] contains non-finite values"
                                )

            if time.monotonic() >= deadline:
                self._last_failure_message = detail
                return None, missing
            time.sleep(0.02)

    def _xarm6_controller_pose_and_offset(
        self,
    ) -> tuple[
        tuple[
            tuple[float, float, float],
            tuple[float, float, float, float],
        ],
        tuple[
            tuple[float, float, float],
            tuple[float, float, float, float],
        ],
    ] | None:
        with self._xarm6_robot_state_lock:
            message = self._xarm6_robot_state
            received_at = self._xarm6_robot_state_received_monotonic
        if message is None:
            self._last_failure_message = "xArm6 robot_states feedback has not been received"
            return None
        age_sec = time.monotonic() - received_at
        if age_sec > 2.0:
            self._last_failure_message = (
                f"xArm6 robot_states feedback is stale ({age_sec:.2f}s)"
            )
            return None
        try:
            return (
                _xarm6_pose_transform(list(message.pose)),
                _xarm6_pose_transform(list(message.offset)),
            )
        except (AttributeError, TypeError, ValueError) as exc:
            self._last_failure_message = f"xArm6 robot_states feedback is invalid: {exc}"
            return None

    def _cartesian_move(
        self,
        target: Any,
        label: str = "",
        avoid_collisions: bool = True,
        min_fraction: float = 0.9,
        allow_partial: bool = False,
        time_scale: float | None = None,
    ) -> bool:
        """Send one exact world -> link_eef target to /xarm6/xarm/set_position."""
        _ = (avoid_collisions, min_fraction, allow_partial)
        if not self.wait_for_services():
            return False
        if not self._prepare_xarm6_mode_one():
            return False
        client = self._xarm6_hardware_cartesian_client
        service_type = self._MoveCartesian
        if client is None or service_type is None:
            self._last_failure_message = "xArm6 direct Cartesian service client is unavailable"
            return False
        try:
            world_target = (
                (
                    float(target.position.x),
                    float(target.position.y),
                    float(target.position.z),
                ),
                _normalized_quaternion(
                    (
                        float(target.orientation.x),
                        float(target.orientation.y),
                        float(target.orientation.z),
                        float(target.orientation.w),
                    )
                ),
            )
            world_base_message = self._tf_buffer.lookup_transform(
                self.frame_id,
                "link_base",
                self._rclpy.time.Time(),
            )
            world_base = (
                (
                    float(world_base_message.transform.translation.x),
                    float(world_base_message.transform.translation.y),
                    float(world_base_message.transform.translation.z),
                ),
                _normalized_quaternion(
                    (
                        float(world_base_message.transform.rotation.x),
                        float(world_base_message.transform.rotation.y),
                        float(world_base_message.transform.rotation.z),
                        float(world_base_message.transform.rotation.w),
                    )
                ),
            )
            controller_state = self._xarm6_controller_pose_and_offset()
            if controller_state is None:
                return False
            _actual_base_tcp, active_eef_tcp = controller_state
            base_eef_target = _compose_transforms(
                _inverse_transform(world_base),
                world_target,
            )
            base_tcp_target = _compose_transforms(base_eef_target, active_eef_tcp)
            roll, pitch, yaw = _quaternion_to_rpy(base_tcp_target[1])
        except (
            AttributeError,
            RuntimeError,
            TypeError,
            ValueError,
            self._tf2_ros.LookupException,
            self._tf2_ros.ConnectivityException,
            self._tf2_ros.ExtrapolationException,
        ) as exc:
            self._last_failure_message = (
                f"[{label}] xArm6 world -> link_base Cartesian conversion failed: {exc}"
            )
            return False
        values = (*base_tcp_target[0], roll, pitch, yaw)
        if not all(math.isfinite(value) for value in values):
            self._last_failure_message = f"[{label}] xArm6 Cartesian target is non-finite"
            return False

        scale = max(0.05, _as_float(time_scale, self.trajectory_time_scale))
        request = service_type.Request()
        request.pose = [
            base_tcp_target[0][0] * 1000.0,
            base_tcp_target[0][1] * 1000.0,
            base_tcp_target[0][2] * 1000.0,
            roll,
            pitch,
            yaw,
        ]
        request.speed = min(
            self._xarm6_cartesian_speed_mm_s,
            self._xarm6_cartesian_speed_mm_s / scale,
        )
        request.acc = min(
            self._xarm6_cartesian_acceleration_mm_s2,
            self._xarm6_cartesian_acceleration_mm_s2 / (scale * scale),
        )
        request.mvtime = 0.0
        request.wait = True
        request.timeout = 60.0
        request.radius = -1.0
        request.is_tool_coord = False
        request.relative = False
        request.motion_type = 0
        handoff_ok, handoff_message = self._prepare_xarm6_firmware_cartesian_mode()
        if not handoff_ok:
            self._last_failure_message = handoff_message
            return False
        try:
            response = self._wait_future(
                client.call_async(request),
                timeout_sec=65.0,
                label=f"service:{label}",
            )
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            self._last_failure_message = (
                f"{self._xarm6_hardware_cartesian_service}: call failed ({exc})"
            )
            restore_ok, restore_message = self._restore_xarm6_trajectory_control()
            if not restore_ok:
                self._last_failure_message += (
                    f"; trajectory control restore failed: {restore_message}"
                )
            return False
        restore_ok, restore_message = self._restore_xarm6_trajectory_control()
        if response is None:
            self._last_failure_message = (
                f"{self._xarm6_hardware_cartesian_service}: response timed out"
            )
            if not restore_ok:
                self._last_failure_message += (
                    f"; trajectory control restore failed: {restore_message}"
                )
            return False
        try:
            return_code = int(getattr(response, "ret", -1))
        except (TypeError, ValueError):
            return_code = -1
        if return_code != 0:
            message = str(getattr(response, "message", "") or "").strip()
            self._last_failure_message = (
                f"{self._xarm6_hardware_cartesian_service}: ret={return_code} {message}"
            ).strip()
            if not restore_ok:
                self._last_failure_message += (
                    f"; trajectory control restore failed: {restore_message}"
                )
            return False
        if not restore_ok:
            self._last_failure_message = (
                f"{self._xarm6_hardware_cartesian_service}: command succeeded but "
                f"trajectory control restore failed: {restore_message}"
            )
            return False

        deadline = time.monotonic() + 5.0
        final_position_error_m = math.inf
        final_orientation_error_rad = math.inf
        while time.monotonic() < deadline:
            actual = self._get_ee_pose()
            if actual is not None:
                final_position_error_m = math.sqrt(
                    (float(actual.position.x) - world_target[0][0]) ** 2
                    + (float(actual.position.y) - world_target[0][1]) ** 2
                    + (float(actual.position.z) - world_target[0][2]) ** 2
                )
                try:
                    final_orientation_error_rad = _quaternion_error_rad(
                        (
                            float(actual.orientation.x),
                            float(actual.orientation.y),
                            float(actual.orientation.z),
                            float(actual.orientation.w),
                        ),
                        world_target[1],
                    )
                except ValueError:
                    final_orientation_error_rad = math.inf
                if (
                    final_position_error_m
                    <= self._xarm6_cartesian_position_tolerance_m
                    and final_orientation_error_rad
                    <= self._xarm6_cartesian_orientation_tolerance_rad
                ):
                    self._last_failure_message = ""
                    return True
            time.sleep(0.05)
        self._last_failure_message = (
            f"{self._xarm6_hardware_cartesian_service}: terminal pose did not converge; "
            f"position_error={final_position_error_m:.6f} m "
            f"orientation_error={final_orientation_error_rad:.6f} rad"
        )
        return False

    def _cancel_xarm6_goal(self, goal_handle: Any, *, label: str) -> str:
        try:
            cancel_future = goal_handle.cancel_goal_async()
            response = self._wait_future(
                cancel_future,
                timeout_sec=3.0,
                label=f"cancel:{label}",
                timeout_log_level="warning",
            )
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            return f"cancel request failed ({exc})"
        accepted = bool(response and list(getattr(response, "goals_canceling", []) or []))
        return "cancel accepted" if accepted else "cancel request was not confirmed"

    def _command_xarm6_hardware_trajectory_action(  # noqa: C901 - explicit action gates.
        self,
        positions: list[float],
        *,
        duration_sec: float,
        label: str,
    ) -> bool:
        """Send one physical xArm6 trajectory and verify terminal result and feedback."""
        if not self.wait_for_services():
            return False
        if not self._prepare_xarm6_mode_one():
            return False
        if len(positions) != len(self.arm_joint_names):
            self._last_failure_message = (
                f"{label} expected {len(self.arm_joint_names)} joints, got {len(positions)}"
            )
            return False
        try:
            targets = [float(position) for position in positions]
        except (TypeError, ValueError) as exc:
            self._last_failure_message = f"{label} contains invalid joint values: {exc}"
            return False
        if not all(math.isfinite(position) for position in targets):
            self._last_failure_message = f"{label} contains non-finite joint values"
            return False
        current_positions, missing = self._get_arm_joint_positions(timeout_sec=1.0)
        if current_positions is None:
            self._last_failure_message = self._last_failure_message or (
                "xArm6 current joint state is unavailable; "
                f"missing={', '.join(missing) if missing else 'unknown'}"
            )
            return False

        goal = self._FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = list(self.arm_joint_names)
        start_point = self._JointTrajectoryPoint()
        start_point.positions = list(current_positions)
        start_point.time_from_start = self._Duration(sec=0, nanosec=0)
        target_point = self._JointTrajectoryPoint()
        target_point.positions = targets
        duration = max(
            0.1,
            float(duration_sec) * self._xarm6_joint_duration_scale,
        )
        seconds = int(duration)
        target_point.time_from_start = self._Duration(
            sec=seconds,
            nanosec=int((duration - seconds) * 1_000_000_000),
        )
        goal.trajectory.points = [start_point, target_point]

        try:
            send_future = self._xarm6_hardware_trajectory_client.send_goal_async(goal)
            goal_handle = self._wait_future(
                send_future,
                timeout_sec=3.0,
                label=f"send:{label}",
            )
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            self._last_failure_message = (
                f"{self._xarm6_hardware_trajectory_action}: send failed ({exc})"
            )
            return False
        if goal_handle is None:
            self._last_failure_message = (
                f"{self._xarm6_hardware_trajectory_action}: send timeout"
            )
            return False
        if not bool(getattr(goal_handle, "accepted", False)):
            self._last_failure_message = (
                f"{self._xarm6_hardware_trajectory_action}: goal rejected"
            )
            return False
        try:
            wrapped = self._wait_future(
                goal_handle.get_result_async(),
                timeout_sec=max(15.0, duration + 10.0),
                label=f"result:{label}",
            )
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            cancel_detail = self._cancel_xarm6_goal(goal_handle, label=label)
            self._last_failure_message = (
                f"{self._xarm6_hardware_trajectory_action}: result failed ({exc}); "
                f"{cancel_detail}"
            )
            return False
        if wrapped is None:
            cancel_detail = self._cancel_xarm6_goal(goal_handle, label=label)
            self._last_failure_message = (
                f"{self._xarm6_hardware_trajectory_action}: result timeout; "
                f"{cancel_detail}"
            )
            return False
        result = getattr(wrapped, "result", None)
        try:
            status = int(getattr(wrapped, "status", -1))
            error_code = int(getattr(result, "error_code", -1))
        except (TypeError, ValueError):
            status = -1
            error_code = -1
        if status != 4 or error_code != 0:
            error_string = str(getattr(result, "error_string", "") or "").strip()
            detail = f"goal_status={status} error_code={error_code}"
            if error_string:
                detail = f"{detail} {error_string}"
            self._last_failure_message = (
                f"{self._xarm6_hardware_trajectory_action}: {detail}"
            )
            return False
        if not self._wait_for_arm_joint_targets(
            targets,
            timeout_sec=max(2.0, duration + 2.0),
            tolerance_rad=0.08,
        ):
            self._last_failure_message = (
                f"{self._xarm6_hardware_trajectory_action}: terminal result succeeded "
                "but xArm6 joints did not converge"
            )
            return False
        if not self._prepare_xarm6_mode_one():
            detail = str(self._last_failure_message or "").strip()
            self._last_failure_message = (
                f"{self._xarm6_hardware_trajectory_action}: terminal result succeeded "
                f"but Mode 1 recovery failed: {detail or 'unknown error'}"
            )
            return False
        self._last_failure_message = ""
        return True

    def move_joints(self, positions: list[float], duration_sec: float = 2.0) -> bool:
        """Replay physical xArm6 joint waypoints through one action goal."""
        return self._command_xarm6_hardware_trajectory_action(
            positions,
            duration_sec=duration_sec,
            label="move_joints",
        )

    def move_to_named_pose(
        self,
        pose_name: str,
        speed: float | None = None,
    ) -> dict[str, Any]:
        """Move to one configured physical xArm6 named position."""
        positions = self.named_positions.get(str(pose_name))
        if not isinstance(positions, (list, tuple)) or not positions:
            available = sorted(self.named_positions.keys()) if self.named_positions else []
            return {
                "success": False,
                "message": f"unknown pose '{pose_name}'; available={available}",
            }
        duration_sec = self._scaled_joint_duration(self.named_pose_duration_sec, speed)
        if self._command_xarm6_hardware_trajectory_action(
            list(positions),
            duration_sec=duration_sec,
            label=f"move_to_named_pose:{pose_name}",
        ):
            return {"success": True, "message": f"moved to named pose '{pose_name}'"}
        return {
            "success": False,
            "message": self._with_last_failure(
                f"failed to move to named pose '{pose_name}'"
            ),
        }

    def move_home(self, speed: float | None = None) -> dict[str, Any]:
        """Move to the configured physical xArm6 home position."""
        result = self.move_to_named_pose("home", speed=speed)
        if result.get("success"):
            self._last_start_pose = None
        return result

    def _command_xarm6_gripper_action(self, position: float, label: str) -> bool:
        if not self.wait_for_services():
            return False
        try:
            target = float(position)
        except (TypeError, ValueError, OverflowError):
            self._last_failure_message = "xArm6 gripper position is invalid"
            return False
        lower = min(float(self.gripper_open), float(self.gripper_close))
        upper = max(float(self.gripper_open), float(self.gripper_close))
        if not math.isfinite(target) or not lower <= target <= upper:
            self._last_failure_message = (
                f"xArm6 gripper position {target!r} is outside "
                f"[{lower:.6f}, {upper:.6f}]"
            )
            return False
        goal = self._GripperCommand.Goal()
        goal.command.position = target
        goal.command.max_effort = 0.0
        try:
            send_future = self._xarm6_gripper_client.send_goal_async(goal)
            goal_handle = self._wait_future(
                send_future,
                timeout_sec=3.0,
                label=f"send:{label}",
            )
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            self._last_failure_message = (
                f"{self._xarm6_gripper_action}: send failed ({exc})"
            )
            return False
        if goal_handle is None:
            self._last_failure_message = f"{self._xarm6_gripper_action}: send timeout"
            return False
        if not bool(getattr(goal_handle, "accepted", False)):
            self._last_failure_message = f"{self._xarm6_gripper_action}: goal rejected"
            return False
        try:
            wrapped = self._wait_future(
                goal_handle.get_result_async(),
                timeout_sec=max(8.0, self.gripper_move_time_sec + 5.0),
                label=f"result:{label}",
            )
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            cancel_detail = self._cancel_xarm6_goal(goal_handle, label=label)
            self._last_failure_message = (
                f"{self._xarm6_gripper_action}: result failed ({exc}); {cancel_detail}"
            )
            return False
        if wrapped is None:
            cancel_detail = self._cancel_xarm6_goal(goal_handle, label=label)
            self._last_failure_message = (
                f"{self._xarm6_gripper_action}: result timeout; {cancel_detail}"
            )
            return False
        result = getattr(wrapped, "result", None)
        try:
            status = int(getattr(wrapped, "status", -1))
        except (TypeError, ValueError):
            status = -1
        if status != 4 or bool(getattr(result, "stalled", False)) or not bool(
            getattr(result, "reached_goal", True)
        ):
            self._last_failure_message = (
                f"{self._xarm6_gripper_action}: goal_status={status} "
                f"stalled={bool(getattr(result, 'stalled', False))} "
                f"reached_goal={bool(getattr(result, 'reached_goal', False))}"
            )
            return False
        self._last_failure_message = ""
        return True

    def open_gripper(self) -> bool:
        """Open the physical xArm6 gripper through its action server."""
        return self._command_xarm6_gripper_action(self.gripper_open, "open_gripper")

    def close_gripper(self, position: float | None = None) -> bool:
        """Close the physical xArm6 gripper through its action server."""
        target = self.gripper_close if position is None else float(position)
        return self._command_xarm6_gripper_action(target, "close_gripper")

    def _destroy_xarm6_action_clients(self) -> None:
        clients = [
            client
            for _name, client in (
                self._xarm6_hardware_trajectory_clients
                + self._xarm6_gripper_clients
            )
        ]
        self._xarm6_hardware_trajectory_clients = []
        self._xarm6_gripper_clients = []
        service_clients = (
            self._xarm6_hardware_cartesian_client,
            self._xarm6_set_mode_client,
            self._xarm6_set_state_client,
            self._xarm6_controller_list_client,
        )
        self._xarm6_hardware_cartesian_client = None
        self._xarm6_set_mode_client = None
        self._xarm6_set_state_client = None
        self._xarm6_controller_list_client = None
        self._xarm6_hardware_trajectory_client = None
        self._xarm6_gripper_client = None
        self._xarm6_clients_ready = False
        for client in clients:
            destroy = getattr(client, "destroy", None)
            if callable(destroy):
                with suppress(AttributeError, RuntimeError):
                    destroy()
        for service_client in service_clients:
            destroy = getattr(service_client, "destroy", None)
            if callable(destroy):
                with suppress(AttributeError, RuntimeError):
                    destroy()

    def shutdown(self) -> None:
        """Release physical xArm6 action clients before destroying their ROS node."""
        self._destroy_xarm6_action_clients()
        self._FollowJointTrajectory = None
        self._GripperCommand = None
        self._MoveCartesian = None
        self._SetInt16 = None
        self._ListControllers = None
        super().shutdown()
