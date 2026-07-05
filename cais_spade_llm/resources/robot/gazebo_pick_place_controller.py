"""
Config-driven Gazebo pick/place controller.

This module intentionally avoids importing ROS2 packages at module import time.
All ROS2 imports happen lazily inside `init()` so non-ROS workflows can still
import the package.
"""

from __future__ import annotations

import importlib
import json
import logging
import math
import os
import sys
import threading
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from cais_spade_llm.product.profile import ProductProfile

destination_token_from_place_inputs = ProductProfile.destination_token_from_place_inputs
has_place_geometry_fields = ProductProfile.has_place_geometry_fields
resolve_place_geometry = ProductProfile.resolve_place_geometry

logger = logging.getLogger(__name__)


def _import_linkattacher_srvs():
    """Import IFRA service types, even if workspace setup wasn't sourced."""

    def _load_srvs():
        srv_module = importlib.import_module("linkattacher_msgs.srv")
        return srv_module.AttachLink, srv_module.DetachLink

    try:
        return _load_srvs()
    except ModuleNotFoundError:
        py_ver = f"python{sys.version_info.major}.{sys.version_info.minor}"
        candidates = [
            os.path.expanduser(
                f"~/ros2_ws/install/linkattacher_msgs/local/lib/{py_ver}/dist-packages"
            ),
            os.path.expanduser(
                f"~/ros2_ws/install/ros2_linkattacher/local/lib/{py_ver}/dist-packages"
            ),
            f"/opt/ros/humble/lib/{py_ver}/dist-packages",
        ]
        for path in candidates:
            if os.path.isdir(path) and path not in sys.path:
                sys.path.append(path)
        try:
            return _load_srvs()
        except Exception:
            return None, None


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _gazebo_world_file_candidates() -> list[Path]:
    candidates: list[Path] = []
    env_path = str(os.environ.get("CAIS_GAZEBO_WORLD_FILE") or "").strip()
    if env_path:
        candidates.append(Path(env_path).expanduser())
    module_path = Path(__file__).resolve()
    for parent in module_path.parents:
        candidates.append(parent / "ros2/cais_lab_gazebo/worlds/table.world")
    candidates.append(Path.home() / "ros2_ws/src/xarm_ros2/cais_lab_gazebo/worlds/table.world")
    return candidates


def _model_footprint_width_from_gazebo_world(model_name: str) -> float | None:
    target_model = str(model_name or "").strip()
    if not target_model:
        return None
    for world_path in _gazebo_world_file_candidates():
        if not world_path.is_file():
            continue
        try:
            root = ET.parse(world_path).getroot()
        except Exception:
            continue
        for model in root.iter("model"):
            if str(model.attrib.get("name") or "").strip() != target_model:
                continue
            for geometry in model.iter("geometry"):
                cylinder = geometry.find("cylinder")
                if cylinder is not None:
                    radius = _as_float(
                        cylinder.findtext("radius"),
                        0.0,
                    )
                    if radius > 0.0:
                        return radius * 2.0
                box = geometry.find("box")
                if box is not None:
                    tokens = str(box.findtext("size") or "").split()
                    if len(tokens) >= 2:
                        try:
                            return max(float(tokens[0]), float(tokens[1]))
                        except (TypeError, ValueError):
                            continue
    return None


def _gazebo_timing_scale_from_env(execution_mode: str) -> float:
    if str(execution_mode or "").strip().lower() != "simulation":
        return 1.0
    if str(os.environ.get("ROBOT_ENV", "gazebo") or "").strip().lower() != "gazebo":
        return 1.0
    scale = _as_float(os.environ.get("CAIS_GAZEBO_WAIT_SCALE"), 1.0)
    if scale <= 0.0:
        return 1.0
    return float(scale)


class GazeboPickPlaceController:
    """
    Generic Gazebo pick/place controller for a single robot.

    Public phase methods map to framework tool names:
      - pick_approach
      - pick_grasp
      - place_approach
      - place_insert
      - move_home
    """

    def __init__(
        self,
        *,
        robot_name: str,
        node_name: str,
        controller_config: dict[str, Any],
        named_positions: dict[str, Any] | None = None,
        execution_mode: str = "simulation",
        arm_joint_names: list[str] | None = None,
        arm_trajectory_topic: str | None = None,
        joint_states_topic: str = "/joint_states",
    ) -> None:
        self.robot_name = robot_name
        self.node_name = node_name
        self.execution_mode = str(execution_mode or "simulation").strip().lower()
        self.controller_config = controller_config or {}
        self.named_positions = named_positions or {}
        self.arm_joint_names = list(arm_joint_names or [])
        self.arm_trajectory_topic = arm_trajectory_topic
        self.joint_states_topic = joint_states_topic
        self._last_failure_message = ""
        self._config_errors: list[str] = []

        move_group = self.controller_config.get("move_group", {})
        gripper = self.controller_config.get("gripper", {})
        services = self.controller_config.get("services", {})
        attach_cfg = self.controller_config.get("attach", {})
        motion = self.controller_config.get("motion", {})
        parts_tuning = self.controller_config.get("parts_tuning", {})

        def need_str(section: dict[str, Any], key: str, path: str) -> str:
            raw = section.get(key)
            if raw is None or str(raw).strip() == "":
                self._config_errors.append(path)
                return ""
            return str(raw)

        def need_float(section: dict[str, Any], key: str, path: str) -> float:
            raw = section.get(key)
            if raw is None:
                self._config_errors.append(path)
                return 0.0
            try:
                return float(raw)
            except Exception:
                self._config_errors.append(path)
                return 0.0

        def need_int(section: dict[str, Any], key: str, path: str) -> int:
            raw = section.get(key)
            if raw is None:
                self._config_errors.append(path)
                return 0
            try:
                return int(raw)
            except Exception:
                self._config_errors.append(path)
                return 0

        def opt_float(section: dict[str, Any], key: str, default: float) -> float:
            raw = section.get(key)
            if raw is None:
                return float(default)
            try:
                return float(raw)
            except Exception:
                return float(default)

        def opt_int(section: dict[str, Any], key: str, default: int) -> int:
            raw = section.get(key)
            if raw is None:
                return int(default)
            try:
                return int(raw)
            except Exception:
                return int(default)

        self.group_name = need_str(move_group, "group_name", "controller.move_group.group_name")
        self.ee_link = need_str(move_group, "ee_link", "controller.move_group.ee_link")
        self.tcp_link = need_str(move_group, "tcp_link", "controller.move_group.tcp_link")
        self.frame_id = need_str(move_group, "frame_id", "controller.move_group.frame_id")

        self.gripper_joint = need_str(gripper, "joint", "controller.gripper.joint")
        self.gripper_topic = need_str(gripper, "topic", "controller.gripper.topic")
        self.gripper_open = need_float(gripper, "open", "controller.gripper.open")
        self.gripper_close = need_float(gripper, "close", "controller.gripper.close")
        self.gripper_move_time_sec = need_float(
            gripper, "move_time_sec", "controller.gripper.move_time_sec"
        )
        self.gripper_settle_sec = need_float(gripper, "settle_sec", "controller.gripper.settle_sec")
        self.gripper_feedback_timeout_pad_sec = need_float(
            gripper,
            "feedback_timeout_pad_sec",
            "controller.gripper.feedback_timeout_pad_sec",
        )
        self.gripper_position_tol = need_float(
            gripper, "position_tolerance", "controller.gripper.position_tolerance"
        )

        self.service_detect_all = need_str(services, "detect_all", "controller.services.detect_all")
        self.service_cartesian_path = need_str(
            services, "cartesian_path", "controller.services.cartesian_path"
        )
        self.service_execute_traj = need_str(
            services, "execute_trajectory", "controller.services.execute_trajectory"
        )
        self.service_attach = need_str(services, "attach", "controller.services.attach")
        self.service_detach = need_str(services, "detach", "controller.services.detach")
        self.service_set_entity_state = need_str(
            services, "set_entity_state", "controller.services.set_entity_state"
        )
        self.service_get_entity_state = str(
            services.get("get_entity_state") or "/get_entity_state"
        ).strip()

        self.robot_model_name = need_str(
            attach_cfg, "robot_model_name", "controller.attach.robot_model_name"
        )
        raw_candidates = attach_cfg.get("attach_link_candidates")
        if isinstance(raw_candidates, list) and raw_candidates:
            self.attach_link_candidates = [str(v) for v in raw_candidates if str(v).strip()]
        else:
            self.attach_link_candidates = []
            self._config_errors.append("controller.attach.attach_link_candidates")
        raw_release_candidates = attach_cfg.get("release_detach_link_candidates")
        if isinstance(raw_release_candidates, list):
            self.release_detach_link_candidates = [
                str(v) for v in raw_release_candidates if str(v).strip()
            ]
        else:
            self.release_detach_link_candidates = []
        self.primary_attach_link = need_str(
            attach_cfg, "primary_attach_link", "controller.attach.primary_attach_link"
        )
        self.detach_timeout_sec = need_float(
            attach_cfg, "detach_timeout_sec", "controller.attach.detach_timeout_sec"
        )
        self.detach_max_link_attempts = need_int(
            attach_cfg,
            "detach_max_link_attempts",
            "controller.attach.detach_max_link_attempts",
        )
        self.release_detach_timeout_sec = max(
            self.detach_timeout_sec,
            opt_float(attach_cfg, "release_detach_timeout_sec", 5.0),
        )

        self.approach_height_m = need_float(
            motion, "approach_height_m", "controller.motion.approach_height_m"
        )
        self.pick_tcp_z_bias_max_m = need_float(
            motion, "pick_tcp_z_bias_max_m", "controller.motion.pick_tcp_z_bias_max_m"
        )
        self.pick_tcp_z_bias_min_m = need_float(
            motion, "pick_tcp_z_bias_min_m", "controller.motion.pick_tcp_z_bias_min_m"
        )
        self.min_pick_tcp_z_m = need_float(
            motion, "min_pick_tcp_z_m", "controller.motion.min_pick_tcp_z_m"
        )
        self.place_surface_gap_m = need_float(
            motion, "place_surface_gap_m", "controller.motion.place_surface_gap_m"
        )
        self.release_preopen_settle_sec = need_float(
            motion,
            "release_preopen_settle_sec",
            "controller.motion.release_preopen_settle_sec",
        )
        self.release_postopen_settle_sec = need_float(
            motion,
            "release_postopen_settle_sec",
            "controller.motion.release_postopen_settle_sec",
        )
        self.release_postdetach_settle_sec = need_float(
            motion,
            "release_postdetach_settle_sec",
            "controller.motion.release_postdetach_settle_sec",
        )
        self.release_descend_time_scale = max(
            1.0,
            opt_float(
                motion,
                "release_descend_time_scale",
                1.35,
            ),
        )
        self.release_detach_retry_count = max(
            0,
            opt_int(motion, "release_detach_retry_count", 2),
        )
        self.release_detach_retry_delay_sec = max(
            0.0,
            opt_float(motion, "release_detach_retry_delay_sec", 0.35),
        )
        self.release_detach_verify_distance_m = max(
            0.005,
            opt_float(motion, "release_detach_verify_distance_m", 0.04),
        )
        self.release_detach_verify_timeout_sec = max(
            0.0,
            opt_float(motion, "release_detach_verify_timeout_sec", 0.75),
        )
        self.release_detach_verify_poll_sec = max(
            0.05,
            opt_float(motion, "release_detach_verify_poll_sec", 0.1),
        )
        self.release_best_effort_detach_timeout_sec = max(
            0.05,
            opt_float(
                motion,
                "release_best_effort_detach_timeout_sec",
                min(
                    self.release_detach_timeout_sec,
                    max(2.0, self.detach_timeout_sec),
                ),
            ),
        )
        self.snap_to_slot_timeout_sec = max(
            0.1,
            opt_float(motion, "snap_to_slot_timeout_sec", 5.0),
        )
        self.snap_to_slot_retry_count = max(
            0,
            opt_int(motion, "snap_to_slot_retry_count", 1),
        )
        self.snap_to_slot_retry_delay_sec = max(
            0.0,
            opt_float(motion, "snap_to_slot_retry_delay_sec", 0.25),
        )
        self.release_retry_lift_m = max(
            0.0,
            opt_float(motion, "release_retry_lift_m", 0.005),
        )
        self.trajectory_time_scale = need_float(
            motion, "trajectory_time_scale", "controller.motion.trajectory_time_scale"
        )
        self.named_pose_duration_sec = max(
            0.1,
            opt_float(motion, "named_pose_duration_sec", 4.0),
        )
        self.move_home_duration_sec = max(
            0.1,
            opt_float(motion, "move_home_duration_sec", self.named_pose_duration_sec),
        )
        self.xy_axis_step_m = max(
            0.01,
            opt_float(motion, "xy_axis_step_m", 1.0),
        )

        self.insertion_depth_m = need_float(
            parts_tuning, "insertion_depth_m", "controller.parts_tuning.insertion_depth_m"
        )
        raw_pick_z_adjustments = parts_tuning.get("pick_z_adjustments_m", {})
        self.pick_z_adjustments_m: dict[str, float] = {}
        if isinstance(raw_pick_z_adjustments, dict):
            for raw_key, raw_value in raw_pick_z_adjustments.items():
                key = str(raw_key or "").strip().upper()
                if not key:
                    continue
                try:
                    self.pick_z_adjustments_m[key] = float(raw_value)
                except (TypeError, ValueError):
                    self._config_errors.append(
                        f"controller.parts_tuning.pick_z_adjustments_m.{key}"
                    )
        if (
            self.pick_tcp_z_bias_min_m > 0.0
            and self.pick_tcp_z_bias_max_m > 0.0
            and self.pick_tcp_z_bias_min_m > self.pick_tcp_z_bias_max_m
        ):
            self._config_errors.append(
                "controller.motion.pick_tcp_z_bias_min_m<=pick_tcp_z_bias_max_m"
            )

        self._apply_gazebo_fast_timing_profile()

        self._config_valid = not self._config_errors
        if not self._config_valid:
            self._last_failure_message = "invalid controller_config; missing/invalid: " + ", ".join(
                sorted(set(self._config_errors))
            )

        self._initialized = False
        self._services_ready = False
        self._spin_thread: threading.Thread | None = None
        self._shutdown_requested = False
        self._executor = None

        self._rclpy = None
        self._node = None
        self._cb_group = None
        self._tf_buffer = None
        self._tf_listener = None
        self._cart_client = None
        self._exec_client = None
        self._detect_all_client_legacy = None
        self._attach_client = None
        self._detach_client = None
        self._set_state_client = None
        self._get_state_client = None
        self._gripper_pub = None
        self._arm_pub = None

        self._attach_srv = None
        self._detach_srv = None
        self._link_attacher_enabled = False
        self._attached_model: str | None = None
        self._attached_link: str | None = None

        self._joint_lock = threading.Lock()
        self._joint_positions: dict[str, float] = {}

        # Remembered start pose for move_home (set externally or by UI bridge).
        self._last_start_pose = None

    def _apply_gazebo_fast_timing_profile(self, scale: float | None = None) -> None:
        resolved_scale = (
            _gazebo_timing_scale_from_env(self.execution_mode)
            if scale is None
            else _as_float(scale, 1.0)
        )
        self._gazebo_wait_scale = resolved_scale
        if resolved_scale <= 0.0 or abs(resolved_scale - 1.0) < 1e-6:
            return

        def scaled_attr(
            attr_name: str,
            *,
            minimum: float = 0.0,
            scale_override: float | None = None,
        ) -> None:
            current = _as_float(getattr(self, attr_name, 0.0), 0.0)
            scale_value = resolved_scale if scale_override is None else scale_override
            setattr(self, attr_name, max(float(minimum), current * scale_value))

        motion_scale = resolved_scale
        if 0.0 < resolved_scale < 1.0:
            # Keep Gazebo service waits at the configured scale, but push arm motion
            # harder. This speeds up motion without shortening attach/detach calls.
            motion_scale = resolved_scale * (0.25 / 0.35)

        scaled_attr("gripper_move_time_sec", minimum=0.15)
        scaled_attr("gripper_settle_sec")
        scaled_attr("release_preopen_settle_sec")
        scaled_attr("release_postopen_settle_sec")
        scaled_attr("release_postdetach_settle_sec")
        scaled_attr("release_detach_retry_delay_sec")
        scaled_attr("release_detach_verify_timeout_sec", minimum=0.05)
        scaled_attr("release_detach_verify_poll_sec", minimum=0.01)
        scaled_attr("snap_to_slot_retry_delay_sec")
        scaled_attr("trajectory_time_scale", minimum=0.20, scale_override=motion_scale)
        scaled_attr("named_pose_duration_sec", minimum=0.25, scale_override=motion_scale)
        scaled_attr("move_home_duration_sec", minimum=0.25, scale_override=motion_scale)

    def _scaled_wall_wait_sec(self, seconds: float, *, minimum: float = 0.0) -> float:
        scale = _as_float(getattr(self, "_gazebo_wait_scale", 1.0), 1.0)
        if scale <= 0.0:
            scale = 1.0
        return max(float(minimum), float(seconds) * scale)

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def _spin_executor(self) -> None:
        """Run executor spin loop and suppress teardown-time RCLError noise."""
        try:
            if self._executor is not None:
                self._executor.spin()
        except Exception as exc:
            msg = str(exc)
            is_context_invalid = (
                "context is not valid" in msg.lower() or exc.__class__.__name__ == "RCLError"
            )
            if self._shutdown_requested and is_context_invalid:
                self._log().debug(f"Executor stopped during shutdown: {msg}")
                return
            self._log().warning(
                f"Executor spin terminated unexpectedly for {self.node_name}: {msg}"
            )

    def init(self) -> bool:
        if self._initialized:
            return True

        if not self._config_valid:
            logger.error("[%s] %s", self.robot_name, self._last_failure_message)
            return False

        if self.execution_mode not in {"simulation", "physical"}:
            logger.warning(
                "[%s] Unknown execution_mode '%s'; treating as 'simulation'",
                self.robot_name,
                self.execution_mode,
            )

        try:
            import rclpy
            import tf2_ros
            from builtin_interfaces.msg import Duration
            from gazebo_msgs.srv import GetEntityState, SetEntityState
            from geometry_msgs.msg import Pose
            from moveit_msgs.action import ExecuteTrajectory
            from moveit_msgs.srv import GetCartesianPath
            from rclpy.action import ActionClient
            from rclpy.callback_groups import ReentrantCallbackGroup
            from rclpy.executors import MultiThreadedExecutor
            from sensor_msgs.msg import JointState
            from std_srvs.srv import Trigger
            from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
        except Exception:
            logger.exception(
                "[%s] ROS2 imports failed. Is the ROS2 environment sourced?",
                self.robot_name,
            )
            self._last_failure_message = "ros2 imports failed (environment not sourced?)"
            return False

        if not rclpy.ok():
            rclpy.init()

        self._rclpy = rclpy
        self._ActionClient = ActionClient
        self._ReentrantCallbackGroup = ReentrantCallbackGroup
        self._MultiThreadedExecutor = MultiThreadedExecutor
        self._Trigger = Trigger
        self._ExecuteTrajectory = ExecuteTrajectory
        self._GetCartesianPath = GetCartesianPath
        self._SetEntityState = SetEntityState
        self._GetEntityState = GetEntityState
        self._Pose = Pose
        self._JointTrajectory = JointTrajectory
        self._JointTrajectoryPoint = JointTrajectoryPoint
        self._Duration = Duration
        self._JointState = JointState
        self._tf2_ros = tf2_ros

        self._node = rclpy.create_node(self.node_name)
        self._cb_group = ReentrantCallbackGroup()

        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self._node)

        self._cart_client = self._node.create_client(
            GetCartesianPath, self.service_cartesian_path, callback_group=self._cb_group
        )
        self._exec_client = ActionClient(
            self._node,
            ExecuteTrajectory,
            self.service_execute_traj,
            callback_group=self._cb_group,
        )
        if self.execution_mode != "physical":
            self._detect_all_client_legacy = self._node.create_client(
                Trigger, self.service_detect_all, callback_group=self._cb_group
            )
        self._set_state_client = self._node.create_client(
            SetEntityState, self.service_set_entity_state, callback_group=self._cb_group
        )
        self._get_state_client = self._node.create_client(
            GetEntityState, self.service_get_entity_state, callback_group=self._cb_group
        )

        if self.gripper_topic and self.gripper_joint:
            self._gripper_pub = self._node.create_publisher(JointTrajectory, self.gripper_topic, 10)

        if self.arm_trajectory_topic:
            self._arm_pub = self._node.create_publisher(
                JointTrajectory, self.arm_trajectory_topic, 10
            )

        self._node.create_subscription(
            JointState, self.joint_states_topic, self._on_joint_state, 50
        )

        self._attach_srv, self._detach_srv = _import_linkattacher_srvs()
        if self._attach_srv and self._detach_srv:
            self._attach_client = self._node.create_client(
                self._attach_srv, self.service_attach, callback_group=self._cb_group
            )
            self._detach_client = self._node.create_client(
                self._detach_srv, self.service_detach, callback_group=self._cb_group
            )
            self._link_attacher_enabled = True
        else:
            self._link_attacher_enabled = False
            self._log().warn("linkattacher_msgs.srv not importable; attach/detach disabled")

        self._executor = MultiThreadedExecutor(num_threads=1)
        self._executor.add_node(self._node)
        self._shutdown_requested = False
        self._spin_thread = threading.Thread(target=self._spin_executor, daemon=True)
        self._spin_thread.start()

        self._initialized = True
        self._log().info(
            f"[{self.robot_name}] Controller initialized (execution_mode={self.execution_mode})"
        )
        self._last_failure_message = ""
        return True

    def shutdown(self) -> None:
        if not self._initialized:
            return

        self._services_ready = False
        self._shutdown_requested = True

        try:
            if self._executor and self._node:
                try:
                    self._executor.remove_node(self._node)
                except Exception:
                    pass
                try:
                    self._executor.shutdown(timeout_sec=1.0)
                except TypeError:
                    self._executor.shutdown()
        except Exception:
            pass

        if self._spin_thread:
            self._spin_thread.join(timeout=2.0)

        try:
            if self._node:
                self._node.destroy_node()
        except Exception:
            pass

        self._spin_thread = None
        self._executor = None
        self._node = None
        self._initialized = False

    def is_usable(self) -> bool:
        """Whether this controller instance is healthy enough for reuse."""
        spin_alive = bool(self._spin_thread and self._spin_thread.is_alive())
        return bool(self._initialized and self._services_ready and spin_alive)

    def wait_for_services(self, timeout_sec: float = 60.0) -> bool:
        if not self.init():
            return False
        if self._services_ready:
            return True

        self._log().info("Waiting for services/actions...")
        deadline = time.monotonic() + timeout_sec

        if self.execution_mode != "physical":
            if not self._wait_service(
                self._detect_all_client_legacy, self.service_detect_all, deadline
            ):
                return False
        if not self._wait_service(self._cart_client, self.service_cartesian_path, deadline):
            return False
        if not self._wait_action_server(self._exec_client, self.service_execute_traj, deadline):
            return False

        if self._link_attacher_enabled:
            if not self._wait_service(self._attach_client, self.service_attach, deadline):
                return False
            if not self._wait_service(self._detach_client, self.service_detach, deadline):
                return False

        while time.monotonic() < deadline:
            try:
                if self._tf_buffer.can_transform(
                    self.frame_id, self.ee_link, self._rclpy.time.Time()
                ):
                    break
            except Exception:
                pass
            time.sleep(0.1)
        else:
            self._log().error(f"TF not ready for {self.frame_id} -> {self.ee_link}")
            self._last_failure_message = f"tf not ready for {self.frame_id} -> {self.ee_link}"
            return False

        # Best effort gripper feedback readiness. Keep the warmup short so
        # Gazebo startup is gated by core services, not late joint-state echo.
        feedback_deadline = min(deadline, time.monotonic() + 1.0)
        while time.monotonic() < feedback_deadline:
            if self._get_joint_position(self.gripper_joint) is not None:
                break
            time.sleep(0.05)

        self._services_ready = True
        self._last_failure_message = ""
        self._log().info("All services ready.")
        return True

    # ------------------------------------------------------------------ #
    # Public primitives (bridge-visible low-level API)
    # ------------------------------------------------------------------ #
    def move_cartesian(
        self,
        x: float,
        y: float,
        z: float,
        speed: float | None = None,
    ) -> dict[str, Any]:
        """
        ---
        description: Move end-effector to an absolute Cartesian position.
        params:
          x: {type: number, description: "Target X coordinate in meters (base frame)"}
          y: {type: number, description: "Target Y coordinate in meters (base frame)"}
          z: {type: number, description: "Target Z coordinate in meters (base frame)"}
          speed: {type: number, description: "Trajectory time scale (>1 slower, <1 faster). Optional."}
        preconditions: {}
        effects:
          current_pose:
            pose_absolute_from_params: [x, y, z]
          current_pose_ref:
            set_unknown: true
        ---
        """
        if not self.wait_for_services():
            return {"success": False, "message": self._unavailable_message("services not ready")}
        ee = self._get_ee_pose()
        if ee is None:
            return {"success": False, "message": "cannot read current ee pose"}
        target_x = float(x)
        target_y = float(y)
        target_z = float(z)
        same_xy = math.isclose(float(ee.position.x), target_x, abs_tol=1e-6) and math.isclose(
            float(ee.position.y), target_y, abs_tol=1e-6
        )
        if same_xy:
            return self._move_pose_direct(
                target_x,
                target_y,
                target_z,
                orientation=ee.orientation,
                label="move_cartesian",
                speed=speed,
            )
        return self._move_xy_at_z(
            target_x,
            target_y,
            target_z,
            orientation=ee.orientation,
            label="move_cartesian",
            speed=speed,
        )

    def move_relative(
        self,
        dx: float,
        dy: float,
        dz: float,
        speed: float | None = None,
    ) -> dict[str, Any]:
        """
        ---
        description: Move end-effector relative to its current position.
        params:
          dx: {type: number, description: "Delta X in meters"}
          dy: {type: number, description: "Delta Y in meters"}
          dz: {type: number, description: "Delta Z in meters"}
          speed: {type: number, description: "Trajectory time scale (>1 slower, <1 faster). Optional."}
        preconditions:
          current_pose:
            exists: true
        effects:
          current_pose:
            pose_relative_from_params: [dx, dy, dz]
          current_pose_ref:
            set_unknown: true
        ---
        """
        if not self.wait_for_services():
            return {"success": False, "message": self._unavailable_message("services not ready")}
        ee = self._get_ee_pose()
        if ee is None:
            return {"success": False, "message": "cannot read current ee pose"}
        target_x = ee.position.x + float(dx)
        target_y = ee.position.y + float(dy)
        target_z = ee.position.z + float(dz)
        time_scale = _as_float(speed, self.trajectory_time_scale)
        ok = self._cartesian_move(
            self._make_pose(target_x, target_y, target_z, ee.orientation),
            f"move_relative(dx={dx}, dy={dy}, dz={dz})",
            time_scale=time_scale,
        )
        direct_failure = str(getattr(self, "_last_failure_message", "") or "").strip().lower()
        if (
            not ok
            and "timed out" not in direct_failure
            and math.isclose(float(dx), 0.0, abs_tol=1e-9)
            and math.isclose(float(dy), 0.0, abs_tol=1e-9)
            and not math.isclose(float(dz), 0.0, abs_tol=1e-9)
        ):
            self._log().warn(
                f"move_relative vertical fallback: retrying no-collision move for dz={float(dz):.4f}"
            )
            ok = self._cartesian_move(
                self._make_pose(target_x, target_y, target_z, ee.orientation),
                f"move_relative(dx={dx}, dy={dy}, dz={dz}) (no-collision)",
                avoid_collisions=False,
                min_fraction=0.70,
                allow_partial=True,
                time_scale=time_scale,
            )
        if not ok:
            return {"success": False, "message": f"failed relative move ({dx}, {dy}, {dz})"}
        return {"success": True, "message": f"moved relative ({dx}, {dy}, {dz})"}

    def move_pose(
        self,
        x: float,
        y: float,
        z: float,
        qx: float,
        qy: float,
        qz: float,
        qw: float,
        speed: float | None = None,
    ) -> dict[str, Any]:
        """
        ---
        description: Move end-effector to an absolute pose with explicit quaternion orientation.
        params:
          x: {type: number, description: "Target X coordinate in meters (base frame)"}
          y: {type: number, description: "Target Y coordinate in meters (base frame)"}
          z: {type: number, description: "Target Z coordinate in meters (base frame)"}
          qx: {type: number, description: "Target quaternion X component"}
          qy: {type: number, description: "Target quaternion Y component"}
          qz: {type: number, description: "Target quaternion Z component"}
          qw: {type: number, description: "Target quaternion W component"}
          speed: {type: number, description: "Trajectory time scale (>1 slower, <1 faster). Optional."}
        preconditions: {}
        effects:
          current_pose:
            pose_absolute_from_params: [x, y, z]
          current_pose_ref:
            set_unknown: true
        ---
        """
        if not self.wait_for_services():
            return {"success": False, "message": self._unavailable_message("services not ready")}
        target_orientation = self._make_orientation(qx, qy, qz, qw)
        ee = self._get_ee_pose()
        if ee is None:
            return {"success": False, "message": "cannot read current ee pose"}
        same_xy = math.isclose(float(ee.position.x), float(x), abs_tol=1e-6) and math.isclose(
            float(ee.position.y), float(y), abs_tol=1e-6
        )
        if same_xy:
            return self._move_pose_direct(
                float(x),
                float(y),
                float(z),
                orientation=target_orientation,
                label="move_pose",
                speed=speed,
            )
        return self._move_xy_at_z(
            float(x),
            float(y),
            float(z),
            orientation=target_orientation,
            label="move_pose",
            speed=speed,
        )

    def move_to_named_pose(self, pose_name: str, speed: float | None = None) -> dict[str, Any]:
        """
        ---
        description: Move to a named joint configuration (e.g. 'home').
        params:
          pose_name: {type: string, description: "Name of the joint configuration from robot manifest"}
          speed: {type: number, description: "Trajectory time scale (>1 slower, <1 faster). Optional."}
        preconditions:
          held_part:
            equals: null
        effects:
          current_state:
            set: idle
          current_pose_ref:
            set_from_param: pose_name
          current_pose:
            set_unknown: true
          occupancy.location:
            set_from_param: pose_name
        ---
        """
        if not self.wait_for_services():
            return {"success": False, "message": self._unavailable_message("services not ready")}
        positions = self.named_positions.get(str(pose_name))
        if not isinstance(positions, (list, tuple)) or not positions:
            available = sorted(self.named_positions.keys()) if self.named_positions else []
            return {
                "success": False,
                "message": f"unknown pose '{pose_name}'; available={available}",
            }
        joint_values = [float(v) for v in positions]
        duration_sec = self._scaled_joint_duration(self.named_pose_duration_sec, speed)
        # Try trajectory publisher first, then MoveIt fallback.
        if self._arm_pub and self._publish_arm_joint_trajectory_and_wait(
            joint_values,
            duration_sec=duration_sec,
        ):
            return {"success": True, "message": f"moved to named pose '{pose_name}'"}
        if self._exec_client and self._move_joints_via_moveit(
            joint_values, duration_sec=duration_sec
        ):
            return {"success": True, "message": f"moved to named pose '{pose_name}' via MoveIt"}
        return {"success": False, "message": f"failed to move to named pose '{pose_name}'"}

    def get_current_pose(self) -> dict[str, Any]:
        """
        ---
        description: Return the current end-effector pose in the base frame.
        params: {}
        preconditions: {}
        effects:
          current_pose_ref:
            set_unknown: true
        ---
        """
        if not self.wait_for_services():
            return {"success": False, "message": self._unavailable_message("services not ready")}
        ee = self._get_ee_pose()
        if ee is None:
            return {"success": False, "message": "cannot read current ee pose"}
        return {
            "success": True,
            "message": "current pose",
            "pose": {
                "x": ee.position.x,
                "y": ee.position.y,
                "z": ee.position.z,
                "qx": ee.orientation.x,
                "qy": ee.orientation.y,
                "qz": ee.orientation.z,
                "qw": ee.orientation.w,
            },
        }

    def attach_part(
        self,
        model_name: str,
        link: str | None = None,
        part_name: str = "",
    ) -> dict[str, Any]:
        """
        ---
        description: Attach a part model to the robot gripper (Gazebo link attacher).
        params:
          model_name: {type: string, description: "Gazebo model name of the part to attach"}
          link: {type: string, description: "Optional specific attach link. Uses default candidates if omitted."}
          part_name: {type: string, description: "Optional canonical part identifier for semantic state projection."}
        preconditions:
          held_part:
            equals: null
          gripper_state:
            equals: closed
        effects:
          held_part:
            set_from_param_any_of: ["part_name", "model_name"]
        ---
        """
        if not self.wait_for_services():
            return {"success": False, "message": self._unavailable_message("services not ready")}
        ok = self._attach_part(str(model_name))
        if not ok:
            return {"success": False, "message": f"failed to attach {model_name}"}
        return {"success": True, "message": f"attached {model_name}"}

    def detach_part(
        self,
        model_name: str = "",
        link: str | None = None,
        assume_released_if_open: bool = False,
    ) -> dict[str, Any]:
        """
        ---
        description: Detach a part model from the robot gripper (Gazebo link detacher).
        params:
          model_name: {type: string, description: "Gazebo model name to detach. Uses currently attached model if empty."}
          link: {type: string, description: "Optional specific detach link. Tries all candidates if omitted."}
        preconditions:
          held_part:
            not_equals: null
          gripper_state:
            equals: open
        effects:
          held_part:
            set: null
        ---
        """
        if not self.wait_for_services():
            return {"success": False, "message": self._unavailable_message("services not ready")}
        target_model = str(model_name or "")
        attempts = max(1, 1 + self.release_detach_retry_count)
        ok = False
        for attempt_idx in range(attempts):
            ok = self._detach_part(
                target_model,
                timeout_sec=self.release_detach_timeout_sec,
                log_failure=(attempt_idx == attempts - 1),
            )
            if ok:
                break

            if attempt_idx + 1 < attempts:
                self._log().warn(
                    f"Detach retry {attempt_idx + 1}/{attempts - 1} for "
                    f"{target_model or 'held part'}"
                )
                time.sleep(self.release_detach_retry_delay_sec)
        if not ok:
            if (
                self._release_open_fallback_allowed(assume_released_if_open)
                and self._gripper_is_open_enough()
            ):
                if not target_model and self._attached_model:
                    target_model = str(self._attached_model)
                verified_release = self._verify_detach_timeout_release(target_model)
                if verified_release is False:
                    return {
                        "success": False,
                        "message": (
                            f"detach timed out and release verification kept "
                            f"{target_model or 'held part'} near the gripper"
                        ),
                        "release_mode": "verification_failed_after_detach_timeout",
                    }
                self._attached_model = None
                self._attached_link = None
                if verified_release is True:
                    self._log().warn(
                        f"Confirmed {target_model or 'held part'} was released after detach timeout because the model is separated from the gripper"
                    )
                    return {
                        "success": True,
                        "message": (
                            f"verified detached {target_model or 'held part'} after gripper opened"
                        ),
                        "release_mode": "verified_open_after_detach_timeout",
                    }
                self._log().warn(
                    f"Assuming {target_model or 'held part'} was released because the gripper is already open and detach verification is unavailable"
                )
                return {
                    "success": True,
                    "message": (
                        f"assumed detached {target_model or 'held part'} after gripper "
                        "opened (verification unavailable)"
                    ),
                    "release_mode": "assumed_open_after_detach_timeout",
                }
            return {"success": False, "message": f"failed to detach {model_name or 'held part'}"}
        return {"success": True, "message": f"detached {model_name or 'held part'}"}

    def grasp_part(
        self,
        model_name: str,
        part_name: str = "",
        position: float | None = None,
    ) -> dict[str, Any]:
        """
        ---
        description: Close the gripper and attach the target part as one high-level grasp primitive.
        params:
          model_name: {type: string, description: "Gazebo model name of the part to attach"}
          part_name: {type: string, description: "Optional canonical part identifier for held-part tracking."}
          position: {type: number, description: "Optional gripper closing position override."}
        preconditions:
          held_part:
            equals: null
        effects:
          current_state:
            set: picked
          gripper_state:
            set: closed
          held_part:
            set_from_param_any_of: ["part_name", "model_name"]
        ---
        """
        if not self.wait_for_services():
            return {
                "success": False,
                "message": self._unavailable_message("services not ready"),
            }

        target_model = str(model_name or "").strip()
        target_part = str(part_name or "").strip()

        if not self.close_gripper(position=position):
            return {
                "success": False,
                "message": (
                    self._last_failure_message
                    or f"failed to close gripper to grasp {target_part or target_model or 'part'}"
                ),
            }

        attached = self.attach_part(target_model, part_name=target_part)
        if attached.get("success"):
            return {
                "success": True,
                "message": (
                    f"grasped {target_part or target_model or 'part'}"
                    if (target_part or target_model)
                    else "grasped part"
                ),
            }

        rollback_ok = self.open_gripper()
        rollback_message = (
            "reopened gripper after failed attach"
            if rollback_ok
            else (self._last_failure_message or "failed to reopen gripper after failed attach")
        )
        return {
            "success": False,
            "message": (
                f"{str(attached.get('message') or 'failed to attach part')}; "
                f"rollback: {rollback_message}"
            ),
        }

    def release_part(
        self,
        model_name: str = "",
        part_name: str = "",
        assume_released_if_open: bool = False,
    ) -> dict[str, Any]:
        """
        ---
        description: Open the gripper and detach the currently held part as one high-level release primitive.
        params:
          model_name: {type: string, description: "Optional controller model name to detach."}
          part_name: {type: string, description: "Optional canonical bridge part name for release trace validation."}
          assume_released_if_open: {type: boolean, description: "Treat an already-open gripper as an idempotent release when true."}
        preconditions:
          held_part:
            not_equals: null
        effects:
          current_state:
            set: idle
          gripper_state:
            set: open
          held_part:
            set: null
        ---
        """
        if not self.wait_for_services():
            return {
                "success": False,
                "message": self._unavailable_message("services not ready"),
            }

        target_model = str(model_name or "").strip()
        target_part = str(part_name or "").strip()
        if (
            not assume_released_if_open
            and str(getattr(self, "execution_mode", "") or "").strip().lower() == "simulation"
        ):
            assume_released_if_open = True

        time.sleep(self.release_preopen_settle_sec)
        if not self.open_gripper():
            return {
                "success": False,
                "message": (
                    self._last_failure_message
                    or f"failed to open gripper to release {target_part or target_model or 'part'}"
                ),
            }
        time.sleep(self.release_postopen_settle_sec)

        used_simulation_release_fallback = False
        if self._simulation_release_fallback_enabled(assume_released_if_open):
            used_simulation_release_fallback = True
            detached = self._release_part_simulation_best_effort_detach(
                target_model,
                target_part,
            )
        else:
            detached = self.detach_part(
                target_model,
                assume_released_if_open=assume_released_if_open,
            )
        if detached.get("success"):
            time.sleep(self.release_postdetach_settle_sec)
            release_mode = str(detached.get("release_mode") or "").strip()
            if release_mode == "verification_unavailable_after_detach_timeout":
                release_message = str(detached.get("message") or "")
            elif target_part or target_model:
                release_message = f"released {target_part or target_model or 'part'}"
            else:
                release_message = "released part"
            result = {
                "success": True,
                "message": release_message,
            }
            if release_mode:
                result["release_mode"] = release_mode
            return result

        if (
            self._release_open_fallback_allowed(assume_released_if_open)
            and not used_simulation_release_fallback
        ):
            fallback_model = target_model
            if not fallback_model and self._attached_model:
                fallback_model = str(self._attached_model)
            verified_release = self._verify_detach_timeout_release(fallback_model)
            if verified_release is not False:
                self._attached_model = None
                self._attached_link = None
                time.sleep(self.release_postdetach_settle_sec)
                result = {
                    "success": True,
                    "message": (
                        f"released {target_part or fallback_model or 'part'}"
                        if (target_part or fallback_model)
                        else "released part"
                    ),
                    "release_mode": (
                        "verified_open_after_detach_timeout"
                        if verified_release is True
                        else "assumed_open_after_detach_timeout"
                    ),
                }
                if verified_release is True:
                    self._log().warn(
                        f"Confirmed {fallback_model or 'held part'} was released after detach failure because the model is separated from the gripper"
                    )
                else:
                    self._log().warn(
                        f"Assuming {fallback_model or 'held part'} was released because the gripper open command succeeded and detach verification is unavailable"
                    )
                return result
            detached = {
                "success": False,
                "message": (
                    f"detach failed and release verification kept "
                    f"{fallback_model or 'held part'} near the gripper"
                ),
            }

        if used_simulation_release_fallback:
            return {
                "success": False,
                "message": str(detached.get("message") or "failed to detach part"),
                "release_mode": detached.get(
                    "release_mode",
                    "verification_unavailable_after_detach_timeout",
                ),
            }

        rollback_ok = self.close_gripper()
        rollback_message = (
            "reclosed gripper after failed detach"
            if rollback_ok
            else (self._last_failure_message or "failed to reclose gripper after failed detach")
        )
        return {
            "success": False,
            "message": (
                f"{str(detached.get('message') or 'failed to detach part')}; "
                f"rollback: {rollback_message}"
            ),
        }

    def _release_open_fallback_allowed(self, assume_released_if_open: bool) -> bool:
        if not assume_released_if_open:
            return False
        mode = str(getattr(self, "execution_mode", "") or "").strip().lower()
        return mode != "physical"

    def _simulation_release_fallback_enabled(self, assume_released_if_open: bool) -> bool:
        if not self._release_open_fallback_allowed(assume_released_if_open):
            return False
        mode = str(getattr(self, "execution_mode", "") or "").strip().lower()
        return mode == "simulation"

    def _simulation_release_detach_timeout_sec(self) -> float:
        configured = _as_float(
            getattr(self, "release_best_effort_detach_timeout_sec", None),
            0.75,
        )
        release_timeout = _as_float(
            getattr(self, "release_detach_timeout_sec", None),
            configured,
        )
        if configured <= 0.0:
            configured = 0.75
        if release_timeout > 0.0:
            configured = min(configured, release_timeout)
        return max(0.05, configured)

    def _release_part_simulation_best_effort_detach(
        self,
        target_model: str,
        target_part: str,
    ) -> dict[str, Any]:
        fallback_model = str(target_model or "").strip()
        if not fallback_model and self._attached_model:
            fallback_model = str(self._attached_model)
        display_name = target_part or fallback_model or "held part"

        ok = self._detach_part(
            fallback_model,
            timeout_sec=self._simulation_release_detach_timeout_sec(),
            attached_link_only=False,
            log_failure=False,
            timeout_log_level="warn",
            break_on_timeout=False,
            prefer_attached_link=False,
            extra_link_candidates=getattr(self, "release_detach_link_candidates", []),
        )
        if ok:
            return {
                "success": True,
                "message": f"detached {display_name}",
            }

        verified_release = self._verify_detach_timeout_release(
            fallback_model,
            timeout_log_level="debug",
        )
        if verified_release is False:
            return {
                "success": False,
                "message": (
                    f"detach failed and release verification kept "
                    f"{fallback_model or 'held part'} near the gripper"
                ),
                "release_mode": "verification_failed_after_detach_timeout",
            }

        if verified_release is True:
            self._attached_model = None
            self._attached_link = None
            self._log().warn(
                f"Confirmed {fallback_model or 'held part'} was released after detach timeout because the model is separated from the gripper"
            )
            return {
                "success": True,
                "message": f"verified detached {display_name} after gripper opened",
                "release_mode": "verified_open_after_detach_timeout",
            }

        return {
            "success": True,
            "message": (
                f"released {display_name} after gripper opened (detach verification unavailable)"
            ),
            "release_mode": "verification_unavailable_after_detach_timeout",
        }

    # ------------------------------------------------------------------ #
    # Legacy low-level API (kept for backward compat)
    # ------------------------------------------------------------------ #
    def move_joints(self, positions: list[float], duration_sec: float = 2.0) -> bool:
        if not self.wait_for_services():
            return False
        if not self._arm_pub:
            self._log().error("Arm trajectory publisher not configured")
            return False
        if len(positions) != len(self.arm_joint_names):
            self._log().error(
                f"move_joints expected {len(self.arm_joint_names)} joints, got {len(positions)}"
            )
            return False

        traj = self._JointTrajectory()
        traj.joint_names = self._get_arm_joint_command_names()
        point = self._JointTrajectoryPoint()
        point.positions = [float(v) for v in positions]
        duration = max(0.1, float(duration_sec))
        sec = int(duration)
        nsec = int((duration - sec) * 1_000_000_000)
        point.time_from_start = self._Duration(sec=sec, nanosec=nsec)
        traj.points = [point]
        self._arm_pub.publish(traj)
        return True

    def _publish_arm_joint_trajectory_and_wait(
        self,
        positions: list[float],
        *,
        duration_sec: float,
        tolerance_rad: float = 0.08,
    ) -> bool:
        if not self.move_joints(positions, duration_sec=duration_sec):
            return False
        timeout_sec = max(2.0, float(duration_sec) + 2.0)
        if self._wait_for_arm_joint_targets(
            positions,
            timeout_sec=timeout_sec,
            tolerance_rad=tolerance_rad,
            log_miss=False,
        ):
            return True
        self._log().warn(
            f"Arm joint trajectory command did not converge within {timeout_sec:.2f}s; falling back"
        )
        return False

    def open_gripper(self) -> bool:
        """
        ---
        description: Open the robot gripper.
        params: {}
        preconditions: {}
        effects:
          gripper_state:
            set: open
        ---
        """
        if not self.wait_for_services():
            return False
        return self._gripper_command(self.gripper_open, "OPEN")

    def close_gripper(self, position: float | None = None) -> bool:
        """
        ---
        description: Close the robot gripper.
        params:
          position: {type: number, description: "Optional custom gripper position override. If omitted, uses the default close position."}
        preconditions: {}
        effects:
          gripper_state:
            set: closed
        ---
        """
        if not self.wait_for_services():
            return False
        target = float(position) if position is not None else self.gripper_close
        return self._gripper_command(target, "CLOSE")

    def delay(self, duration_sec: float) -> dict[str, Any]:
        """
        ---
        description: Wait intentionally between robot task steps.
        params:
          duration_sec: {type: number, description: "Intentional wait duration in seconds."}
        preconditions: {}
        effects: {}
        synthesis_hidden: true
        ---
        """
        try:
            duration = float(duration_sec)
        except (TypeError, ValueError):
            return {
                "success": False,
                "message": f"delay duration_sec must be numeric: {duration_sec!r}",
            }
        if not math.isfinite(duration) or duration < 0.0:
            return {
                "success": False,
                "message": "delay duration_sec must be finite and non-negative",
            }
        wait_sec = self._scaled_wall_wait_sec(duration)
        time.sleep(wait_sec)
        return {
            "success": True,
            "message": f"delay {duration:.3f}s",
            "duration_sec": duration,
            "wait_sec": wait_sec,
        }

    def _derive_gripper_close_position(
        self,
        *,
        model_name: str = "",
        product_geometry: dict[str, Any] | None = None,
    ) -> float | None:
        if not (
            float(self.gripper_open) > float(self.gripper_close)
            and 0.0 <= float(self.gripper_close) <= float(self.gripper_open) <= 0.25
        ):
            return None

        geometry = product_geometry if isinstance(product_geometry, dict) else {}
        width = None
        for key in (
            "grasp_width_m",
            "part_width_m",
            "part_diameter_m",
            "diameter_m",
            "width_m",
        ):
            if key in geometry:
                candidate = _as_float(geometry.get(key), 0.0)
                if candidate > 0.0:
                    width = candidate
                    break
        if width is None:
            width = _model_footprint_width_from_gazebo_world(str(model_name or ""))
        if width is None or width <= 0.0:
            return None

        target = width
        target = max(float(self.gripper_close), min(float(self.gripper_open), target))
        return target

    # ------------------------------------------------------------------ #
    # Geometry helpers (agent calls these to compute targets, then moves)
    # ------------------------------------------------------------------ #
    def compute_pick_targets(
        self,
        part_name: str = "",
        product_geometry: dict[str, Any] | None = None,
        target_pose: dict[str, Any] | None = None,
        target_pose_source: str = "",
        prefer_live_detection: bool = False,
        approach_height_override_m: float | None = None,
        ignore_current_height_for_travel_z: bool = False,
        min_pick_tcp_z_override_m: float | None = None,
        use_global_min_pick_tcp_z: bool = True,
        surface_clearance_override_m: float | None = None,
        apply_pick_z_adjustments: bool = True,
    ) -> dict[str, Any]:
        """
        ---
        description: Compute pick target positions from perception + geometry without moving.
        params:
          part_name: {type: string, description: "Name of the detected part to pick"}
          product_geometry: {type: object, description: "Optional geometry override dict"}
          target_pose: {type: object, description: "Optional known target pose with x/y/z; skips perception when provided"}
          target_pose_source: {type: string, description: "Optional source label for target_pose, e.g. observed_pose"}
          prefer_live_detection: {type: boolean, description: "When true, try perception first and use target_pose only as fallback"}
          approach_height_override_m: {type: number, description: "Optional vertical approach distance"}
          ignore_current_height_for_travel_z: {type: boolean}
          min_pick_tcp_z_override_m: {type: number}
          use_global_min_pick_tcp_z: {type: boolean, description: "When false, do not clamp pick TCP Z to controller.motion.min_pick_tcp_z_m"}
          surface_clearance_override_m: {type: number, description: "Optional target-surface clearance added to the raw pick TCP Z"}
          apply_pick_z_adjustments: {type: boolean, description: "When false, skip per-part pick Z adjustments"}
        preconditions: {}
        effects: {}
        ---
        Compute pick target positions from perception + geometry without moving.

        Returns a dict with keys: part_name, model_name, tx, ty, tz, pick_z,
        travel_z, part_height, tcp_offset_z, pick_tcp_z, start_x, start_y,
        start_z, or {"success": False, "message": ...} on failure.
        """
        if not self.wait_for_services():
            return {"success": False, "message": self._unavailable_message("services not ready")}

        target = None
        parts = None
        detection_attempted = False
        normalized_target_pose = None
        if isinstance(target_pose, dict) and {"x", "y", "z"} <= set(target_pose.keys()):
            try:
                normalized_target_pose = {
                    "x": float(target_pose["x"]),
                    "y": float(target_pose["y"]),
                    "z": float(target_pose["z"]),
                }
            except (TypeError, ValueError):
                normalized_target_pose = None

        if bool(prefer_live_detection) and str(part_name or "").strip():
            detection_attempted = True
            parts = self.detect_parts()
            if parts:
                target = next((p for p in parts if p.get("part_name") == part_name), None)

        if target is not None:
            tx = _as_float(target.get("x"), 0.0)
            ty = _as_float(target.get("y"), 0.0)
            tz = _as_float(target.get("z"), 0.0)
            target_part_name = str(target.get("part_name") or part_name or "")
            target_pose_source_used = "live_detection"
        elif normalized_target_pose is None:
            if parts is None and not detection_attempted:
                parts = self.detect_parts()
            if not parts:
                return {"success": False, "message": "no parts detected"}

            if part_name:
                target = next((p for p in parts if p.get("part_name") == part_name), None)
                if target is None:
                    detected_names = sorted(
                        {
                            str(p.get("part_name"))
                            for p in parts
                            if str(p.get("part_name") or "").strip()
                        }
                    )
                    return {
                        "success": False,
                        "message": f"requested part '{part_name}' not detected; detected={detected_names}",
                    }
            if target is None:
                target = parts[0]
            tx = _as_float(target.get("x"), 0.0)
            ty = _as_float(target.get("y"), 0.0)
            tz = _as_float(target.get("z"), 0.0)
            target_part_name = str(target.get("part_name") or part_name or "")
            target_pose_source_used = "live_detection"
        else:
            tx = float(normalized_target_pose["x"])
            ty = float(normalized_target_pose["y"])
            tz = float(normalized_target_pose["z"])
            target_part_name = str(part_name or target_pose.get("part_name") or "")
            target_pose_source_used = str(target_pose_source or "")

        geo = product_geometry or {}
        board_center = geo.get("board_center", {}) if isinstance(geo, dict) else {}
        board_center_z = _as_float(board_center.get("z"), 1.02)

        target_height = _as_float(geo.get("part_height_m"), 0.08)
        pose_model_name = target_pose.get("model_name") if isinstance(target_pose, dict) else ""
        target_model = str(
            geo.get("model_name") or (target or {}).get("model_name") or pose_model_name or ""
        )

        ee = self._get_ee_pose()
        if ee is None:
            return {"success": False, "message": "cannot read current ee pose"}
        self._last_start_pose = self._make_pose(
            ee.position.x,
            ee.position.y,
            ee.position.z,
            ee.orientation,
        )

        ee_tcp_offset_z = self._get_ee_tcp_world_z_offset()
        pick_bias = max(
            self.pick_tcp_z_bias_min_m,
            min(self.pick_tcp_z_bias_max_m, target_height * 0.25),
        )
        surface_clearance_m = max(0.0, _as_float(surface_clearance_override_m, 0.0))
        pick_tcp_z_raw = tz + pick_bias + surface_clearance_m
        if min_pick_tcp_z_override_m is not None:
            effective_min_tcp_z = float(min_pick_tcp_z_override_m)
            pick_tcp_z = max(pick_tcp_z_raw, effective_min_tcp_z)
        elif bool(use_global_min_pick_tcp_z):
            effective_min_tcp_z = self.min_pick_tcp_z_m
            pick_tcp_z = max(pick_tcp_z_raw, effective_min_tcp_z)
        else:
            effective_min_tcp_z = None
            pick_tcp_z = pick_tcp_z_raw
        pick_z_adjustment_m = (
            self.pick_z_adjustments_m.get(target_part_name.upper(), 0.0)
            if bool(apply_pick_z_adjustments)
            else 0.0
        )
        gripper_close_position = self._derive_gripper_close_position(
            model_name=target_model,
            product_geometry=geo,
        )
        pick_z = pick_tcp_z - ee_tcp_offset_z
        pick_z += pick_z_adjustment_m

        approach_height = _as_float(approach_height_override_m, self.approach_height_m)
        travel_candidates = [
            tz + approach_height,
            board_center_z + approach_height,
            pick_z + 0.05,
        ]
        if not bool(ignore_current_height_for_travel_z):
            travel_candidates.append(ee.position.z)
        travel_z = max(travel_candidates)

        self._log().info(
            "[ComputePickTargets] "
            f"part={target_part_name} "
            f"current=({ee.position.x:.3f}, {ee.position.y:.3f}, {ee.position.z:.3f}) "
            f"target=({tx:.3f}, {ty:.3f}, {tz:.3f}) "
            f"source={target_pose_source_used or 'perception'} "
            f"surface_clearance={surface_clearance_m:.3f} "
            f"pick_tcp_z={pick_tcp_z:.3f} "
            f"travel_z={travel_z:.3f} pick_z={pick_z:.3f} tcp_offset_z={ee_tcp_offset_z:.3f}"
        )

        return {
            "success": True,
            "part_name": target_part_name,
            "model_name": target_model,
            "tx": tx,
            "ty": ty,
            "tz": tz,
            "pick_z": pick_z,
            "travel_z": travel_z,
            "part_height": target_height,
            "tcp_offset_z": ee_tcp_offset_z,
            "pick_tcp_z": pick_tcp_z,
            "pick_tcp_z_raw": pick_tcp_z_raw,
            "surface_clearance_m": surface_clearance_m,
            "pick_z_adjustment_m": pick_z_adjustment_m,
            "gripper_close_position": gripper_close_position,
            "apply_pick_z_adjustments": bool(apply_pick_z_adjustments),
            "effective_min_pick_tcp_z": effective_min_tcp_z,
            "use_global_min_pick_tcp_z": bool(use_global_min_pick_tcp_z),
            "target_pose_source": target_pose_source_used,
            "prefer_live_detection": bool(prefer_live_detection),
            "start_x": ee.position.x,
            "start_y": ee.position.y,
            "start_z": ee.position.z,
        }

    def compute_place_targets(
        self,
        pick_ctx: dict[str, Any] | None = None,
        product_geometry: dict[str, Any] | None = None,
        part_name: str = "",
        z_adjustment_m: float = 0.0,
        destination_location: str = "",
    ) -> dict[str, Any]:
        """
        ---
        description: Compute placement target positions from pick context + geometry without moving.
        params:
          part_name: {type: string, description: "Name of the held part to place"}
          pick_ctx: {type: object, description: "Optional output context from previous pick"}
          product_geometry: {type: object, description: "Optional geometry override dict"}
          z_adjustment_m: {type: number, description: "Extra Z vertical adjustment"}
          destination_location: {type: string, description: "Optional symbolic destination token, such as assembly_board-v1, resolved internally to placement geometry when available"}
        preconditions: {}
        effects: {}
        ---
        Compute placement target positions from pick context + geometry without moving.

        Returns a dict with keys: slot_x, slot_y, board_top_z, place_z,
        part_height, or {"success": False, "message": ...} on failure.
        """
        if not self.wait_for_services():
            return {"success": False, "message": self._unavailable_message("services not ready")}

        pick_ctx = dict(pick_ctx or {})
        target_part_name = str(part_name or pick_ctx.get("part_name") or "")
        geo = resolve_place_geometry(
            part_name=target_part_name,
            destination_location=destination_location,
            product_geometry=product_geometry,
            execution_mode=self.execution_mode,
        )
        symbolic_destination = destination_token_from_place_inputs(
            destination_location=destination_location,
            product_geometry=product_geometry,
        )
        if symbolic_destination and not has_place_geometry_fields(geo):
            return {
                "success": False,
                "message": (
                    f"failed to resolve placement geometry for destination "
                    f"'{symbolic_destination}' and part '{target_part_name or '?'}'"
                ),
            }
        board_center = geo.get("board_center", {}) if isinstance(geo, dict) else {}
        slot_xy = geo.get("slot_xy")
        if isinstance(slot_xy, (list, tuple)) and len(slot_xy) >= 2:
            bx = _as_float(board_center.get("x"), 0.0) + _as_float(slot_xy[0], 0.0)
            by = _as_float(board_center.get("y"), 0.0) + _as_float(slot_xy[1], 0.0)
        else:
            bx = _as_float(board_center.get("x"), pick_ctx.get("tx", 0.0))
            by = _as_float(board_center.get("y"), pick_ctx.get("ty", 0.0))

        board_top_z = _as_float(
            geo.get("slot_floor_z_m"),
            _as_float(board_center.get("z"), 1.025),
        )
        target_height = _as_float(geo.get("part_height_m"), pick_ctx.get("part_height", 0.08))
        target_reference = dict(geo.get("target_reference") or {})
        target_origin_pose = dict(geo.get("target_origin_pose") or {})

        if pick_ctx:
            grasp_tcp_to_part_origin_z = _as_float(pick_ctx.get("pick_tcp_z"), 0.0) - _as_float(
                pick_ctx.get("tz"), 0.0
            )
            tcp_offset_z = _as_float(
                pick_ctx.get("tcp_offset_z"), self._get_ee_tcp_world_z_offset()
            )
        else:
            # Recovery insert macros may only know the target geometry, not the earlier pick context.
            grasp_tcp_to_part_origin_z = max(
                self.pick_tcp_z_bias_min_m,
                min(self.pick_tcp_z_bias_max_m, target_height * 0.25),
            )
            tcp_offset_z = self._get_ee_tcp_world_z_offset()

        target_point = str(target_reference.get("target_point") or "").strip()
        reference_z = target_origin_pose.get("z")
        place_part_origin_z_source = "slot_geometry"
        origin_pose = (
            pick_ctx.get("origin_pose") if isinstance(pick_ctx.get("origin_pose"), dict) else {}
        )
        pick_origin_location = str(pick_ctx.get("origin_resource_location") or "").strip()
        requested_destination = str(destination_location or "").strip()
        use_measured_origin_pose = (
            target_point == "part_origin"
            and bool(origin_pose)
            and (not requested_destination or requested_destination == pick_origin_location)
            and {"x", "y", "z"} <= set(origin_pose.keys())
        )
        if use_measured_origin_pose:
            bx = _as_float(origin_pose.get("x"), bx)
            by = _as_float(origin_pose.get("y"), by)
            reference_z = origin_pose.get("z")
            target_origin_pose = {
                "x": bx,
                "y": by,
                "z": _as_float(reference_z, board_top_z + (target_height * 0.5)),
                "source": "pick_ctx.origin_pose",
            }
            place_part_origin_z_source = "pick_ctx.origin_pose"
        if target_point == "part_origin":
            place_part_origin_z = _as_float(reference_z, board_top_z + (target_height * 0.5))
            if place_part_origin_z_source != "pick_ctx.origin_pose":
                reference_z_value = _as_float(reference_z, math.nan)
                place_part_origin_z_source = (
                    "target_origin_pose"
                    if math.isfinite(reference_z_value)
                    else "support_geometry_fallback"
                )
        else:
            place_gap = self.place_surface_gap_m - self.insertion_depth_m
            place_part_origin_z = board_top_z + (target_height * 0.5) + place_gap
            if target_point != "part_origin":
                place_part_origin_z = max(
                    place_part_origin_z,
                    board_top_z + (target_height * 0.5),
                )
        place_tcp_z = place_part_origin_z + grasp_tcp_to_part_origin_z
        place_z = place_tcp_z - tcp_offset_z + _as_float(z_adjustment_m, 0.0)

        return {
            "success": True,
            "part_name": target_part_name,
            "slot_x": bx,
            "slot_y": by,
            "board_top_z": board_top_z,
            "place_z": place_z,
            "place_tcp_z": place_tcp_z,
            "place_part_origin_z": place_part_origin_z,
            "place_part_origin_z_source": place_part_origin_z_source,
            "approach_pose": {"x": bx, "y": by, "z": place_z + 0.05},
            "target_pose": {"x": bx, "y": by, "z": place_z},
            "part_height": target_height,
            "tcp_offset_z": tcp_offset_z,
            "grasp_tcp_to_part_origin_z": grasp_tcp_to_part_origin_z,
            "target_reference": target_reference,
            "target_origin_pose": target_origin_pose,
            "model_name": str(geo.get("model_name") or pick_ctx.get("model_name") or ""),
        }

    def get_tcp_offset_z(self) -> float:
        """Return the world-frame Z offset between EE link and TCP link."""
        if not self.wait_for_services():
            return -0.17
        return self._get_ee_tcp_world_z_offset()

    def snap_part_to_slot(
        self,
        model_name: str,
        slot_x: float,
        slot_y: float,
        part_height: float,
        board_top_z: float,
        part_origin_z: float | None = None,
        destination_location: str = "",
    ) -> bool:
        """Teleport a Gazebo model to its exact slot pose (post-placement correction)."""
        if not self.wait_for_services():
            return False
        return self._snap_part_to_slot(
            model_name,
            slot_x,
            slot_y,
            part_height,
            board_top_z,
            part_origin_z=part_origin_z,
            destination_location=destination_location,
        )

    def detect_parts(self, part_name: str | None = None) -> list[dict[str, Any]]:
        """
        ---
        description: Detect parts via perception service. Optionally filter by part name.
        params:
          part_name: {type: string, description: "Filter results to this part name. Returns all parts if omitted."}
        preconditions: {}
        effects: {}
        ---
        """
        if not self.wait_for_services():
            return []
        if self.execution_mode == "physical":
            self._log().warning(
                "Physical mode detect_parts is unavailable in ROS2 controller path; "
                "use direct physical perception integration."
            )
            return []

        future = self._detect_all_client_legacy.call_async(self._Trigger.Request())
        result = self._wait_future(future, timeout_sec=10.0, label="detect_all_legacy")
        if not result or not result.success:
            msg = result.message if result else "timeout"
            self._log().error(f"/detect_all failed: {msg}")
            return []
        try:
            parsed = json.loads(result.message)
        except Exception:
            self._log().exception("Failed to parse /detect_all payload")
            return []
        parts = parsed if isinstance(parsed, list) else []
        if part_name:
            parts = [p for p in parts if p.get("part_name") == part_name]
        return parts

    # ------------------------------------------------------------------ #
    # Standalone utility methods (called directly by UI bridge / tests)
    # ------------------------------------------------------------------ #
    def return_to_remembered_start_pose(self) -> dict[str, Any]:
        if not self.wait_for_services():
            return {
                "success": False,
                "message": self._unavailable_message("services not ready"),
            }

        if self._last_start_pose is None:
            return {"success": False, "message": "no remembered start pose available"}

        if self._cartesian_move(self._last_start_pose, "Return to remembered start pose"):
            self._last_start_pose = None
            return {"success": True, "message": "returned to remembered start pose"}
        return {"success": False, "message": "failed to return to remembered start pose"}

    def move_home(self, speed: float | None = None) -> dict[str, Any]:
        if not self.wait_for_services():
            return {
                "success": False,
                "message": self._unavailable_message("services not ready"),
            }

        home = self.named_positions.get("home")
        if not isinstance(home, (list, tuple)) or not home:
            return {"success": False, "message": "no home pose available"}

        target_positions = [float(v) for v in home]
        actual_positions, missing = self._get_arm_joint_positions(timeout_sec=0.2)
        if actual_positions is not None:
            if len(actual_positions) == len(target_positions):
                already_home = all(
                    self._angular_joint_error(actual, target) <= 0.08
                    for actual, target in zip(actual_positions, target_positions)
                )
                if already_home:
                    self._last_start_pose = None
                    return {"success": True, "message": "already at named home pose"}
        elif missing:
            self._log().warning(
                f"move_home could not confirm current joint state before homing; missing={missing}"
            )

        # Move to the explicit named joint-space home pose.
        duration_sec = self._scaled_joint_duration(self.move_home_duration_sec, speed)
        if self._arm_pub and self._publish_arm_joint_trajectory_and_wait(
            target_positions,
            duration_sec=duration_sec,
        ):
            self._last_start_pose = None
            return {"success": True, "message": "moved to named home pose"}
        if self._arm_pub:
            actual_positions, missing = self._get_arm_joint_positions(timeout_sec=0.5)
            if actual_positions is not None and len(actual_positions) == len(target_positions):
                already_home_after_attempt = all(
                    self._angular_joint_error(actual, target) <= 0.08
                    for actual, target in zip(actual_positions, target_positions)
                )
                if already_home_after_attempt:
                    self._last_start_pose = None
                    return {"success": True, "message": "already at named home pose"}
            elif missing:
                self._log().warning(
                    f"move_home could not confirm current joint state after homing attempt; missing={missing}"
                )
            if not self._exec_client:
                return {
                    "success": False,
                    "message": self._with_last_failure("failed to move to named home pose"),
                }

        # Fallback to MoveIt execute_trajectory action (for robots like
        # xarm6 that have no direct arm trajectory publisher).
        if self._exec_client and self._JointTrajectory:
            if self._move_joints_via_moveit(target_positions, duration_sec=duration_sec):
                self._last_start_pose = None
                return {"success": True, "message": "moved to named home pose"}
            actual_positions, missing = self._get_arm_joint_positions(timeout_sec=0.5)
            if actual_positions is not None and len(actual_positions) == len(target_positions):
                already_home_after_attempt = all(
                    self._angular_joint_error(actual, target) <= 0.08
                    for actual, target in zip(actual_positions, target_positions)
                )
                if already_home_after_attempt:
                    self._last_start_pose = None
                    self._log().warning(
                        "move_home execute_trajectory reported failure but current joints are already at home; treating as success"
                    )
                    return {"success": True, "message": "already at named home pose"}

        return {
            "success": False,
            "message": self._with_last_failure("failed to move to named home pose"),
        }

    def _format_moveit_error(self, code: int | None) -> str:
        if code is None:
            return "unknown error (timeout or action server failure)"
        mapping = {
            1: "SUCCESS",
            -1: "PLANNING_FAILED (No valid trajectory found. Target may be unreachable or in self-collision)",
            -2: "INVALID_MOTION_PLAN",
            -3: "MOTION_PLAN_INVALIDATED_BY_ENVIRONMENT_CHANGE",
            -4: "CONTROL_FAILED",
            -5: "UNABLE_TO_AQUIRE_SENSOR_DATA",
            -6: "TIMED_OUT",
            -7: "PREEMPTED",
            -10: "START_STATE_IN_COLLISION",
            -11: "START_STATE_VIOLATES_PATH_CONSTRAINTS",
            -12: "GOAL_IN_COLLISION (Target pose intersects with an obstacle)",
            -13: "GOAL_VIOLATES_PATH_CONSTRAINTS",
            -14: "GOAL_CONSTRAINTS_VIOLATED",
            -15: "INVALID_GROUP_NAME",
            -16: "INVALID_GOAL_CONSTRAINTS",
            -17: "INVALID_ROBOT_STATE (Robot state is outside joint limits)",
            -18: "INVALID_LINK_NAME",
            -19: "INVALID_OBJECT_NAME",
            -21: "FRAME_TRANSFORM_FAILURE",
            -22: "COLLISION_CHECKING_UNAVAILABLE",
            -23: "ROBOT_STATE_STALE",
            -24: "SENSOR_INFO_STALE",
            -31: "NO_IK_SOLUTION (Inverse Kinematics failed. Target pose is impossible to reach)",
        }
        return mapping.get(code, f"error_code={code}")

    def _move_joints_via_moveit(
        self,
        positions: list[float],
        duration_sec: float = 4.0,
    ) -> bool:
        """Move to joint positions using the MoveIt execute_trajectory action."""
        if len(positions) != len(self.arm_joint_names):
            self._log().error(
                "_move_joints_via_moveit expected "
                f"{len(self.arm_joint_names)} joints, got {len(positions)}"
            )
            return False
        try:
            from moveit_msgs.msg import RobotTrajectory
        except ImportError:
            self._log().error("moveit_msgs not available for joint move")
            return False

        traj = self._JointTrajectory()
        traj.joint_names = self._get_arm_joint_command_names()
        point = self._JointTrajectoryPoint()
        point.positions = [float(v) for v in positions]
        duration = max(0.1, float(duration_sec))
        sec = int(duration)
        nsec = int((duration - sec) * 1_000_000_000)
        point.time_from_start = self._Duration(sec=sec, nanosec=nsec)
        traj.points = [point]

        robot_traj = RobotTrajectory()
        robot_traj.joint_trajectory = traj

        exec_goal = self._ExecuteTrajectory.Goal()
        exec_goal.trajectory = robot_traj

        send_future = self._exec_client.send_goal_async(exec_goal)
        goal_handle = self._wait_future(send_future, timeout_sec=10.0, label="send:move_home")
        if not goal_handle or not goal_handle.accepted:
            self._log().error("move_home trajectory goal rejected")
            return False

        result_future = goal_handle.get_result_async()
        result = self._wait_future(result_future, timeout_sec=30.0, label="result:move_home")
        code = result.result.error_code.val if result else None
        if code != 1:
            err_msg = self._format_moveit_error(code)
            self._log().error(f"move_home execute_trajectory failed: {err_msg}")
        return code == 1

    def _scaled_joint_duration(self, base_duration_sec: float, speed: float | None) -> float:
        scale = _as_float(speed, self.trajectory_time_scale)
        if scale <= 0.0:
            scale = self.trajectory_time_scale
        return max(0.5, float(base_duration_sec) * float(scale))

    def detach_model(self, model_name: str, *, quiet: bool = False) -> dict[str, Any]:
        if not self.wait_for_services():
            return {
                "success": False,
                "message": self._unavailable_message("services not ready"),
            }
        target_model = str(model_name or "").strip()
        if not target_model:
            return {"success": False, "message": "model name is required"}
        ok = self._detach_model_from_any_link(target_model)
        if not ok:
            ok = self._detach_part(target_model, log_failure=not quiet)
        return {
            "success": bool(ok),
            "message": "detached" if ok else f"failed to detach {target_model}",
        }

    def set_entity_pose(
        self,
        model_name: str,
        *,
        x: float,
        y: float,
        z: float,
        qx: float = 0.0,
        qy: float = 0.0,
        qz: float = 0.0,
        qw: float = 1.0,
        reference_frame: str = "world",
    ) -> dict[str, Any]:
        if not self.wait_for_services():
            return {
                "success": False,
                "message": self._unavailable_message("services not ready"),
            }
        target_model = str(model_name or "").strip()
        if not target_model:
            return {"success": False, "message": "model name is required"}
        if not self._set_state_client.wait_for_service(timeout_sec=2.0):
            return {"success": False, "message": "set_entity_state service unavailable"}

        from gazebo_msgs.msg import EntityState

        state = EntityState()
        state.name = target_model
        state.pose.position.x = float(x)
        state.pose.position.y = float(y)
        state.pose.position.z = float(z)
        state.pose.orientation.x = float(qx)
        state.pose.orientation.y = float(qy)
        state.pose.orientation.z = float(qz)
        state.pose.orientation.w = float(qw)
        state.reference_frame = str(reference_frame or "world")

        if self._link_attacher_enabled:
            for board_link in (f"anchor_{target_model}", "link"):
                try:
                    self._detach_part_from_assembly_board(target_model, board_link)
                except Exception as exc:
                    self._log().debug(
                        f"set_entity_pose board detach ignored for {target_model}:{board_link}: {exc}"
                    )

        req = self._SetEntityState.Request()
        req.state = state
        future = self._set_state_client.call_async(req)
        response = self._wait_future(
            future, timeout_sec=5.0, label=f"set_entity_pose:{target_model}"
        )
        if response and response.success:
            return {"success": True, "message": f"entity pose reset for {target_model}"}
        detail = getattr(response, "status_message", "") if response is not None else ""
        detail = str(detail or "").strip() or f"failed to set pose for {target_model}"
        return {"success": False, "message": detail}

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _log(self):
        return self._node.get_logger() if self._node else logger

    def _unavailable_message(self, default: str) -> str:
        if self._last_failure_message:
            return f"{default}: {self._last_failure_message}"
        return default

    def _wait_service(self, client, name: str, deadline: float) -> bool:
        while time.monotonic() < deadline:
            if client and client.wait_for_service(timeout_sec=0.5):
                return True
        self._log().error(f"Timed out waiting for service: {name}")
        self._last_failure_message = f"timed out waiting for service: {name}"
        return False

    def _wait_action_server(self, action_client, name: str, deadline: float) -> bool:
        while time.monotonic() < deadline:
            if action_client and action_client.wait_for_server(timeout_sec=0.5):
                return True
        self._log().error(f"Timed out waiting for action server: {name}")
        self._last_failure_message = f"timed out waiting for action server: {name}"
        return False

    def _get_ee_pose(self):
        try:
            t = self._tf_buffer.lookup_transform(
                self.frame_id, self.ee_link, self._rclpy.time.Time()
            )
            pose = self._Pose()
            pose.position.x = t.transform.translation.x
            pose.position.y = t.transform.translation.y
            pose.position.z = t.transform.translation.z
            pose.orientation = t.transform.rotation
            return pose
        except Exception as e:
            self._log().error(f"TF lookup failed: {e}")
            return None

    def _get_ee_tcp_world_z_offset(self) -> float:
        try:
            ee_tf = self._tf_buffer.lookup_transform(
                self.frame_id, self.ee_link, self._rclpy.time.Time()
            )
            tcp_tf = self._tf_buffer.lookup_transform(
                self.frame_id, self.tcp_link, self._rclpy.time.Time()
            )
            return tcp_tf.transform.translation.z - ee_tf.transform.translation.z
        except Exception:
            self._log().warn("Could not get world EE-to-TCP offset, using default -0.17m")
            return -0.17

    def _on_joint_state(self, msg):
        with self._joint_lock:
            for name, pos in zip(msg.name, msg.position):
                self._joint_positions[name] = pos

    def _get_joint_position(self, joint_name: str) -> float | None:
        with self._joint_lock:
            exact = self._joint_positions.get(joint_name)
            if exact is not None:
                return exact

            prefix = f"{self.robot_name}_"
            prefixed_name = (
                joint_name if str(joint_name).startswith(prefix) else f"{prefix}{joint_name}"
            )
            prefixed = self._joint_positions.get(prefixed_name)
            if prefixed is not None:
                return prefixed

            suffix_matches = [
                value
                for name, value in self._joint_positions.items()
                if str(name).endswith(str(joint_name))
            ]
            if len(suffix_matches) == 1:
                return suffix_matches[0]
            return None

    def _get_arm_joint_positions(
        self,
        *,
        timeout_sec: float = 0.0,
    ) -> tuple[list[float] | None, list[str]]:
        deadline = time.monotonic() + max(0.0, float(timeout_sec))
        missing: list[str] = []
        while True:
            values: list[float] = []
            missing = []
            for joint_name in self.arm_joint_names:
                value = self._get_joint_position(joint_name)
                if value is None:
                    missing.append(joint_name)
                else:
                    values.append(float(value))
            if not missing:
                return values, []
            if time.monotonic() >= deadline:
                return None, missing
            time.sleep(0.02)

    def _get_arm_joint_command_names(self) -> list[str]:
        with self._joint_lock:
            available_names = set(str(name) for name in self._joint_positions.keys())

        if not available_names:
            return list(self.arm_joint_names)

        resolved: list[str] = []
        prefix = f"{self.robot_name}_"
        for joint_name in self.arm_joint_names:
            exact_name = str(joint_name)
            prefixed_name = exact_name if exact_name.startswith(prefix) else f"{prefix}{exact_name}"
            if exact_name in available_names:
                resolved.append(exact_name)
            elif prefixed_name in available_names:
                resolved.append(prefixed_name)
            else:
                resolved.append(exact_name)
        return resolved

    @staticmethod
    def _angular_joint_error(actual: float, target: float) -> float:
        return abs(math.atan2(math.sin(actual - target), math.cos(actual - target)))

    def _wait_for_arm_joint_targets(
        self,
        targets: list[float],
        timeout_sec: float,
        *,
        tolerance_rad: float = 0.08,
        log_miss: bool = True,
    ) -> bool:
        if len(targets) != len(self.arm_joint_names):
            return False

        deadline = time.monotonic() + max(0.0, float(timeout_sec))
        saw_feedback = False
        last_values: list[float] | None = None
        missing: list[str] = []

        while time.monotonic() < deadline:
            values, missing = self._get_arm_joint_positions(timeout_sec=0.0)
            if values is not None:
                saw_feedback = True
                last_values = list(values)
                if all(
                    self._angular_joint_error(values[index], targets[index]) <= tolerance_rad
                    for index in range(len(targets))
                ):
                    return True
            time.sleep(0.02)

        if not log_miss:
            return False

        if not saw_feedback:
            self._log().warn(f"Timed out waiting for arm joint feedback; missing={missing}")
            return False

        max_error = max(
            self._angular_joint_error(last_values[index], targets[index])
            for index in range(len(targets))
        )
        self._log().warn(
            "Timed out waiting for arm joint target; "
            f"max_error={max_error:.4f}rad tolerance={tolerance_rad:.4f}rad"
        )
        return False

    def _wait_for_gripper_target(
        self,
        target: float,
        timeout_sec: float,
        *,
        log_miss: bool = True,
    ) -> bool:
        deadline = time.monotonic() + timeout_sec
        saw_feedback = False
        last_pos = None
        while self._rclpy.ok() and time.monotonic() < deadline:
            pos = self._get_joint_position(self.gripper_joint)
            if pos is not None:
                saw_feedback = True
                last_pos = pos
                if abs(pos - target) <= self.gripper_position_tol:
                    return True
            time.sleep(0.02)

        if not log_miss:
            return False
        if not saw_feedback:
            self._last_failure_message = (
                f"no joint-state feedback for '{self.gripper_joint}' while waiting gripper move"
            )
            self._log().warn(
                f"No joint-state feedback for '{self.gripper_joint}' while waiting gripper move"
            )
        else:
            self._last_failure_message = (
                f"gripper target not reached: target={target:.3f} current={float(last_pos):.3f}"
            )
            self._log().warn(
                f"Gripper target not reached: target={target:.3f} current={float(last_pos):.3f}"
            )
        return False

    def _gripper_is_open_enough(self) -> bool:
        pos = self._get_joint_position(self.gripper_joint)
        if pos is None:
            return False
        midpoint = (float(self.gripper_open) + float(self.gripper_close)) * 0.5
        tol = max(float(self.gripper_position_tol) * 2.0, 0.01)
        if self.gripper_open >= self.gripper_close:
            return float(pos) >= (midpoint - tol)
        return float(pos) <= (midpoint + tol)

    def _get_link_world_position(self, link_name: str) -> tuple[float, float, float] | None:
        link_name = str(link_name or "").strip()
        if not link_name:
            return None
        try:
            transform = self._tf_buffer.lookup_transform(
                self.frame_id,
                link_name,
                self._rclpy.time.Time(),
            )
        except Exception:
            return None
        translation = transform.transform.translation
        return (
            float(translation.x),
            float(translation.y),
            float(translation.z),
        )

    def _get_entity_world_position(
        self,
        model_name: str,
        timeout_log_level: str = "error",
    ) -> tuple[float, float, float] | None:
        target_model = str(model_name or "").strip()
        mode = str(getattr(self, "execution_mode", "") or "").strip().lower()
        if not target_model or mode == "physical":
            return None
        if not self._get_state_client or not getattr(self, "_GetEntityState", None):
            return None
        if not self._get_state_client.wait_for_service(timeout_sec=0.2):
            return None

        req = self._GetEntityState.Request()
        req.name = target_model
        req.reference_frame = str(self.frame_id or "world")
        future = self._get_state_client.call_async(req)
        response = self._wait_future(
            future,
            timeout_sec=1.0,
            label=f"get_entity_state:{target_model}",
            timeout_log_level=timeout_log_level,
        )
        if not response or not getattr(response, "success", False):
            return None
        pose = response.state.pose
        return (
            float(pose.position.x),
            float(pose.position.y),
            float(pose.position.z),
        )

    @staticmethod
    def _xyz_distance(
        a: tuple[float, float, float],
        b: tuple[float, float, float],
    ) -> float:
        return math.sqrt(
            (float(a[0]) - float(b[0])) ** 2
            + (float(a[1]) - float(b[1])) ** 2
            + (float(a[2]) - float(b[2])) ** 2
        )

    def _verify_detach_timeout_release(
        self,
        target_model: str,
        timeout_log_level: str = "error",
    ) -> bool | None:
        target_model = str(target_model or "").strip()
        if not target_model:
            return None

        link_candidates: list[str] = []
        if self._attached_link:
            link_candidates.append(str(self._attached_link))
        if self.primary_attach_link and self.primary_attach_link not in link_candidates:
            link_candidates.append(self.primary_attach_link)
        for link_name in self.attach_link_candidates:
            if link_name not in link_candidates:
                link_candidates.append(link_name)
        if not link_candidates:
            return None

        deadline = time.monotonic() + max(0.0, self.release_detach_verify_timeout_sec)
        best_min_distance: float | None = None
        best_link_name = ""
        while True:
            model_position = self._get_entity_world_position(
                target_model,
                timeout_log_level=timeout_log_level,
            )
            if model_position is None:
                return None

            min_distance: float | None = None
            min_link_name = ""
            for link_name in link_candidates:
                link_position = self._get_link_world_position(link_name)
                if link_position is None:
                    continue
                distance = self._xyz_distance(model_position, link_position)
                if min_distance is None or distance < min_distance:
                    min_distance = distance
                    min_link_name = link_name

            if min_distance is None:
                return None

            if best_min_distance is None or min_distance > best_min_distance:
                best_min_distance = min_distance
                best_link_name = min_link_name

            if min_distance > self.release_detach_verify_distance_m:
                self._log().info(
                    f"Verified detach fallback for {target_model}: closest link "
                    f"{min_link_name or '<unknown>'} is {min_distance:.3f}m away"
                )
                return True

            if time.monotonic() >= deadline:
                break
            time.sleep(self.release_detach_verify_poll_sec)

        self._log().warn(
            f"Detach fallback verification failed for {target_model}: closest link "
            f"{best_link_name or '<unknown>'} remained within "
            f"{float(best_min_distance or 0.0):.3f}m "
            f"(threshold={self.release_detach_verify_distance_m:.3f}m)"
        )
        return False

    def _gripper_command(
        self,
        position: float,
        label: str,
        move_time_s: float | None = None,
        wait_s: float | None = None,
        require_target: bool = False,
        log_target_miss: bool | None = None,
    ) -> bool:
        if not self._gripper_pub:
            self._last_failure_message = "gripper publisher is not configured"
            self._log().error("Gripper publisher is not configured")
            return False

        move_time_s = self.gripper_move_time_sec if move_time_s is None else float(move_time_s)
        wait_s = self.gripper_settle_sec if wait_s is None else float(wait_s)
        if log_target_miss is None:
            log_target_miss = bool(require_target)

        self._log().info(f"Gripper: {label} (position={position:.3f})")
        traj = self._JointTrajectory()
        traj.joint_names = [self.gripper_joint]
        point = self._JointTrajectoryPoint()
        point.positions = [position]
        move_time_s = max(0.05, move_time_s)
        sec = int(move_time_s)
        nsec = int((move_time_s - sec) * 1_000_000_000)
        point.time_from_start = self._Duration(sec=sec, nanosec=nsec)
        traj.points = [point]

        self._gripper_pub.publish(traj)
        time.sleep(self._scaled_wall_wait_sec(0.05))
        self._gripper_pub.publish(traj)

        feedback_timeout = max(move_time_s + self.gripper_feedback_timeout_pad_sec, 1.0)
        reached = self._wait_for_gripper_target(
            position,
            feedback_timeout,
            log_miss=bool(log_target_miss),
        )
        if require_target and not reached:
            if log_target_miss:
                self._log().error(
                    f"Gripper command did not reach required target: target={position:.3f}"
                )
            if not self._last_failure_message:
                self._last_failure_message = (
                    f"gripper command did not reach required target {position:.3f}"
                )
            return False
        time.sleep(max(0.0, wait_s))
        self._last_failure_message = ""
        return True

    def _wait_future(
        self,
        future,
        timeout_sec: float,
        label: str,
        timeout_log_level: str = "error",
    ):
        deadline = time.monotonic() + timeout_sec
        while self._rclpy.ok() and not future.done() and time.monotonic() < deadline:
            time.sleep(0.01)
        if not future.done():
            try:
                future.cancel()
            except Exception:
                pass
            message = f"[{label}] timed out"
            logger = self._log()
            level = str(timeout_log_level or "error").strip().lower()
            if level == "debug":
                log_method = (
                    getattr(logger, "debug", None)
                    or getattr(logger, "warn", None)
                    or getattr(logger, "warning", None)
                )
                if log_method:
                    log_method(message)
                else:
                    logger.error(message)
            elif level in {"warn", "warning"}:
                log_method = getattr(logger, "warn", None) or getattr(
                    logger,
                    "warning",
                    None,
                )
                if log_method:
                    log_method(message)
                else:
                    logger.error(message)
            elif level == "info" and hasattr(logger, "info"):
                logger.info(message)
            else:
                logger.error(message)
            return None
        return future.result()

    def _scale_trajectory_timing(self, solution, scale: float):
        if solution is None:
            return
        # `scale` multiplies trajectory duration:
        #   >1.0 => slower, <1.0 => faster, ==1.0 => unchanged.
        if scale <= 0.0 or abs(scale - 1.0) < 1e-6:
            return
        joint_traj = solution.joint_trajectory
        if not joint_traj.points:
            return
        for point in joint_traj.points:
            total_ns = int(point.time_from_start.sec) * 1_000_000_000 + int(
                point.time_from_start.nanosec
            )
            scaled_ns = max(1, int(total_ns * scale))
            point.time_from_start.sec = scaled_ns // 1_000_000_000
            point.time_from_start.nanosec = scaled_ns % 1_000_000_000
            if point.velocities:
                point.velocities = [v / scale for v in point.velocities]
            if point.accelerations:
                point.accelerations = [a / (scale * scale) for a in point.accelerations]

    def _attach_part(self, model_name: str) -> bool:
        if not self._link_attacher_enabled:
            return True
        if not model_name:
            self._log().error("Cannot attach: empty model name")
            return False

        if self._attached_model and self._attached_model != model_name:
            self._detach_part(self._attached_model)

        for link_name in self.attach_link_candidates:
            req = self._attach_srv.Request()
            req.model1_name = self.robot_model_name
            req.link1_name = link_name
            req.model2_name = model_name
            req.link2_name = "link"

            future = self._attach_client.call_async(req)
            response = self._wait_future(future, timeout_sec=5.0, label=f"attach:{link_name}")
            if response is None:
                continue
            if response and response.success:
                self._attached_model = model_name
                self._attached_link = link_name
                return True

            msg = response.message if response else "no response"
            if "already attached to another link" in str(msg).lower():
                self._detach_model_from_any_link(model_name)
                continue

        self._log().error(f"Failed to attach {model_name}")
        return False

    def _detach_model_from_any_link(self, target_model: str) -> bool:
        if not self._link_attacher_enabled:
            return True
        if not target_model:
            return False
        if not self._detach_client.wait_for_service(timeout_sec=0.5):
            return False

        detached_any = False
        for link_name in self.attach_link_candidates:
            req = self._detach_srv.Request()
            req.model1_name = self.robot_model_name
            req.link1_name = link_name
            req.model2_name = target_model
            req.link2_name = "link"

            future = self._detach_client.call_async(req)
            response = self._wait_future(
                future,
                timeout_sec=self.detach_timeout_sec,
                label=f"detach-recover:{target_model}:{link_name}",
            )
            if response and response.success:
                detached_any = True

        if detached_any and self._attached_model == target_model:
            self._attached_model = None
            self._attached_link = None
        return detached_any

    def _detach_part(
        self,
        model_name: str = "",
        timeout_sec: float | None = None,
        attached_link_only: bool = False,
        log_failure: bool = True,
        timeout_log_level: str = "error",
        break_on_timeout: bool = True,
        prefer_attached_link: bool = True,
        extra_link_candidates: list[str] | tuple[str, ...] | None = None,
    ) -> bool:
        if not self._link_attacher_enabled:
            return True

        target_model = model_name or self._attached_model
        if not target_model:
            return True
        if not self._detach_client.wait_for_service(timeout_sec=0.5):
            return False
        detach_timeout = _as_float(timeout_sec, self.detach_timeout_sec)
        if detach_timeout <= 0.0:
            detach_timeout = self.detach_timeout_sec

        links_to_try: list[str] = []
        if prefer_attached_link and self._attached_link:
            links_to_try.append(self._attached_link)
        if not (attached_link_only and links_to_try):
            if self.primary_attach_link and self.primary_attach_link not in links_to_try:
                links_to_try.append(self.primary_attach_link)
            for link in self.attach_link_candidates:
                if link not in links_to_try:
                    links_to_try.append(link)
            for link in extra_link_candidates or []:
                link_name = str(link or "").strip()
                if link_name and link_name not in links_to_try:
                    links_to_try.append(link_name)
        if (
            not prefer_attached_link
            and self._attached_link
            and self._attached_link not in links_to_try
        ):
            links_to_try.append(self._attached_link)
        max_link_attempts = (
            len(links_to_try)
            if not break_on_timeout and not attached_link_only
            else max(1, self.detach_max_link_attempts)
        )
        links_to_try = links_to_try[: max(1, max_link_attempts)]

        for link_name in links_to_try:
            req = self._detach_srv.Request()
            req.model1_name = self.robot_model_name
            req.link1_name = link_name
            req.model2_name = target_model
            req.link2_name = "link"

            future = self._detach_client.call_async(req)
            response = self._wait_future(
                future,
                timeout_sec=detach_timeout,
                label=f"detach:{link_name}",
                timeout_log_level=timeout_log_level,
            )
            if response and response.success:
                self._attached_model = None
                self._attached_link = None
                return True

            if response is None:
                if log_failure:
                    suffix = "skipping remaining links" if break_on_timeout else "trying next link"
                    self._log().error(f"Detach service timed out on {link_name}, {suffix}")
                if break_on_timeout:
                    break
                continue

        if log_failure:
            self._log().error(f"Failed to detach {target_model}")
        return False

    def _snap_part_to_slot(
        self,
        model_name: str,
        slot_x: float,
        slot_y: float,
        part_height: float,
        board_top_z: float,
        part_origin_z: float | None = None,
        destination_location: str = "",
    ) -> bool:
        if not self._set_state_client.wait_for_service(timeout_sec=2.0):
            return False

        from gazebo_msgs.msg import EntityState

        state = EntityState()
        state.name = model_name
        state.pose.position.x = slot_x
        state.pose.position.y = slot_y
        state.pose.position.z = _as_float(
            part_origin_z,
            board_top_z + (part_height * 0.5),
        )
        state.pose.orientation.w = 1.0
        state.twist.linear.x = 0.0
        state.twist.linear.y = 0.0
        state.twist.linear.z = 0.0
        state.twist.angular.x = 0.0
        state.twist.angular.y = 0.0
        state.twist.angular.z = 0.0
        state.reference_frame = "world"

        self._detach_part(
            model_name,
            timeout_sec=self._simulation_release_detach_timeout_sec(),
            attached_link_only=False,
            log_failure=False,
            timeout_log_level="debug",
            break_on_timeout=False,
            prefer_attached_link=False,
            extra_link_candidates=getattr(self, "release_detach_link_candidates", []),
        )

        if not self._set_entity_state_for_snap(model_name, state):
            return False
        if not self._attach_part_to_assembly_board(
            model_name,
            destination_location=destination_location,
        ):
            return False
        if not self._set_entity_state_for_snap(model_name, state):
            return False

        self._log().info(
            f"snap_to_slot stabilized {model_name} at "
            f"({slot_x:.3f}, {slot_y:.3f}, {state.pose.position.z:.3f})"
        )
        return True

    def _set_entity_state_for_snap(self, model_name: str, state) -> bool:
        req = self._SetEntityState.Request()
        req.state = state

        attempts = max(1, 1 + int(getattr(self, "snap_to_slot_retry_count", 0) or 0))
        timeout_sec = _as_float(getattr(self, "snap_to_slot_timeout_sec", None), 5.0)
        retry_delay_sec = _as_float(getattr(self, "snap_to_slot_retry_delay_sec", None), 0.25)
        saw_success = False
        for attempt_idx in range(attempts):
            future = self._set_state_client.call_async(req)
            response = self._wait_future(
                future,
                timeout_sec=timeout_sec,
                label="snap_to_slot",
            )
            if response and response.success:
                saw_success = True
                if attempt_idx + 1 < attempts:
                    time.sleep(retry_delay_sec)
                continue
            if attempt_idx + 1 < attempts:
                self._log().warn(
                    f"snap_to_slot retry {attempt_idx + 1}/{attempts - 1} for {model_name}"
                )
                time.sleep(retry_delay_sec)
        if saw_success:
            return True
        return False

    def _attach_part_to_assembly_board(
        self,
        model_name: str,
        *,
        destination_location: str = "",
    ) -> bool:
        if not self._link_attacher_enabled:
            return True
        target_model = str(model_name or "").strip()
        if not target_model:
            return False
        destination = str(destination_location or "").strip()
        if destination and destination != "assembly_board-v1":
            return True
        if not self._attach_client.wait_for_service(timeout_sec=0.5):
            return False

        board_links = [f"anchor_{target_model}", "link"]
        for board_link in board_links:
            self._detach_part_from_assembly_board(target_model, board_link)

        last_message = ""
        for attempt_idx in range(2):
            for board_link in board_links:
                req = self._attach_srv.Request()
                req.model1_name = "assembly_board_v1"
                req.link1_name = board_link
                req.model2_name = target_model
                req.link2_name = "link"

                future = self._attach_client.call_async(req)
                response = self._wait_future(
                    future,
                    timeout_sec=5.0,
                    label=f"attach:assembly_board_v1:{board_link}:{target_model}",
                    timeout_log_level="warn",
                )
                if response and response.success:
                    if self._attached_model == target_model:
                        self._attached_model = None
                        self._attached_link = None
                    return True

                last_message = str(response.message if response else "no response")
                msg = last_message.lower()
                if "failed to find link" in msg and board_link != "link":
                    continue
                if "already attached" in msg and board_link == f"anchor_{target_model}":
                    self._detach_part(
                        target_model,
                        timeout_sec=self._simulation_release_detach_timeout_sec(),
                        attached_link_only=False,
                        log_failure=False,
                        timeout_log_level="debug",
                        break_on_timeout=False,
                        prefer_attached_link=False,
                        extra_link_candidates=getattr(self, "release_detach_link_candidates", []),
                    )
                    self._detach_part_from_assembly_board(target_model, board_link)
                    break
            if attempt_idx == 0:
                time.sleep(self._scaled_wall_wait_sec(0.05))
        self._log().error(f"Failed to attach {target_model} to assembly_board_v1: {last_message}")
        return False

    def _detach_part_from_assembly_board(self, model_name: str, board_link: str) -> bool:
        if not self._link_attacher_enabled:
            return True
        target_model = str(model_name or "").strip()
        link_name = str(board_link or "").strip()
        if not target_model or not link_name:
            return False
        if not self._detach_client.wait_for_service(timeout_sec=0.5):
            return False

        req = self._detach_srv.Request()
        req.model1_name = "assembly_board_v1"
        req.link1_name = link_name
        req.model2_name = target_model
        req.link2_name = "link"
        future = self._detach_client.call_async(req)
        response = self._wait_future(
            future,
            timeout_sec=0.5,
            label=f"detach:assembly_board_v1:{link_name}:{target_model}",
            timeout_log_level="debug",
        )
        return bool(response and response.success)

    def _cartesian_move(
        self,
        target,
        label: str = "",
        avoid_collisions: bool = True,
        min_fraction: float = 0.9,
        allow_partial: bool = False,
        time_scale: float | None = None,
    ) -> bool:
        request = self._GetCartesianPath.Request()
        request.header.frame_id = self.frame_id
        request.header.stamp = self._node.get_clock().now().to_msg()
        request.group_name = self.group_name
        request.link_name = self.ee_link
        request.waypoints = [target]
        request.max_step = 0.01
        request.jump_threshold = 0.0
        request.avoid_collisions = avoid_collisions
        request.start_state.is_diff = True

        future = self._cart_client.call_async(request)
        response = self._wait_future(future, timeout_sec=10.0, label=f"plan:{label}")
        if response is None:
            self._last_failure_message = f"[{label}] planning response timed out"
            self._log().error(f"[{label}] planning response timed out")
            return False
        if response.fraction < min_fraction:
            self._last_failure_message = (
                f"[{label}] planning fraction too low: {response.fraction:.3f} < {min_fraction:.3f}"
            )
            self._log().error(
                f"[{label}] planning fraction too low: {response.fraction:.3f} < {min_fraction:.3f}"
            )
            return False
        if response.fraction < 0.999 and not allow_partial:
            self._last_failure_message = f"[{label}] planning fraction incomplete: {response.fraction:.3f} (partial not allowed)"
            self._log().error(
                f"[{label}] planning fraction incomplete: {response.fraction:.3f} (partial not allowed)"
            )
            return False

        exec_goal = self._ExecuteTrajectory.Goal()
        if time_scale is None:
            scale = self.trajectory_time_scale
        else:
            scale = _as_float(time_scale, self.trajectory_time_scale)
        self._scale_trajectory_timing(response.solution, scale)
        exec_goal.trajectory = response.solution

        send_future = self._exec_client.send_goal_async(exec_goal)
        goal_handle = self._wait_future(send_future, timeout_sec=10.0, label=f"send:{label}")
        if not goal_handle or not goal_handle.accepted:
            self._last_failure_message = f"[{label}] trajectory goal rejected by execute action"
            self._log().error(f"[{label}] trajectory goal rejected by execute action")
            return False

        result_future = goal_handle.get_result_async()
        result = self._wait_future(result_future, timeout_sec=30.0, label=f"result:{label}")
        code = result.result.error_code.val if result else None
        if code != 1:
            err_msg = self._format_moveit_error(code)
            self._last_failure_message = f"[{label}] execute_trajectory failed: {err_msg}"
            self._log().error(self._last_failure_message)
            return False
        self._last_failure_message = ""
        return True

    def _move_xy_at_z(
        self,
        x: float,
        y: float,
        z: float,
        *,
        orientation=None,
        label: str = "move_xy_at_z",
        speed: float | None = None,
    ) -> dict[str, Any]:
        if not self.wait_for_services():
            return {"success": False, "message": self._unavailable_message("services not ready")}
        if orientation is None:
            ee = self._get_ee_pose()
            if ee is None:
                return {"success": False, "message": "cannot read current ee pose"}
            orientation = ee.orientation
        ok = self._move_xy_direct(
            float(x),
            float(y),
            float(z),
            orientation,
            label,
            time_scale=_as_float(speed, self.trajectory_time_scale),
        )
        if not ok:
            return {"success": False, "message": f"failed to move above target ({x}, {y}, {z})"}
        return {"success": True, "message": f"moved above target ({x:.4f}, {y:.4f}, {z:.4f})"}

    def _move_pose_direct(
        self,
        x: float,
        y: float,
        z: float,
        *,
        orientation=None,
        label: str = "move_pose_direct",
        speed: float | None = None,
        avoid_collisions: bool = True,
        min_fraction: float = 0.9,
        allow_partial: bool = False,
    ) -> dict[str, Any]:
        if not self.wait_for_services():
            return {"success": False, "message": self._unavailable_message("services not ready")}
        if orientation is None:
            ee = self._get_ee_pose()
            if ee is None:
                return {"success": False, "message": "cannot read current ee pose"}
            orientation = ee.orientation
        ok = self._cartesian_move(
            self._make_pose(float(x), float(y), float(z), orientation),
            label,
            avoid_collisions=avoid_collisions,
            min_fraction=min_fraction,
            allow_partial=allow_partial,
            time_scale=_as_float(speed, self.trajectory_time_scale),
        )
        if not ok:
            return {"success": False, "message": f"failed to move directly to ({x}, {y}, {z})"}
        return {"success": True, "message": f"moved directly to ({x:.4f}, {y:.4f}, {z:.4f})"}

    def _release_part_sequence(
        self,
        *,
        model_name: str,
        slot_x: float,
        slot_y: float,
        part_height: float,
        board_top_z: float,
        place_z: float,
        travel_z: float,
        orientation=None,
    ) -> dict[str, Any]:
        if not self.wait_for_services():
            return {"success": False, "message": self._unavailable_message("services not ready")}

        if orientation is None:
            ee = self._get_ee_pose()
            if ee is None:
                return {"success": False, "message": "cannot read current ee pose"}
            orientation = ee.orientation

        released = self.release_part(str(model_name))
        if not released.get("success"):
            return {
                "success": False,
                "message": str(released.get("message") or "failed to release part"),
            }

        if model_name:
            self._snap_part_to_slot(
                str(model_name),
                float(slot_x),
                float(slot_y),
                float(part_height),
                float(board_top_z),
            )

        lift_ok = self._cartesian_move(
            self._make_pose(float(slot_x), float(slot_y), float(travel_z), orientation),
            "Lift after place",
        )
        if not lift_ok:
            lift_ok = self._cartesian_move(
                self._make_pose(float(slot_x), float(slot_y), float(travel_z), orientation),
                "Lift after place (no-collision)",
                avoid_collisions=False,
                min_fraction=0.70,
                allow_partial=True,
            )

        if lift_ok:
            return {"success": True, "message": "released part and lifted clear"}
        return {"success": False, "message": "failed to lift clear after release"}

    def _move_xy_direct(
        self,
        target_x: float,
        target_y: float,
        z: float,
        orientation,
        label_prefix: str,
        *,
        time_scale: float | None = None,
    ) -> bool:
        if self._cartesian_move(
            self._make_pose(target_x, target_y, z, orientation),
            label_prefix,
            time_scale=time_scale,
        ):
            return True

        current = self._get_ee_pose()
        if current is None:
            self._log().error(f"[{label_prefix}] cannot read current EE pose for staged fallback")
            return False
        if math.isclose(float(current.position.x), float(target_x), abs_tol=1e-6) and math.isclose(
            float(current.position.y), float(target_y), abs_tol=1e-6
        ):
            self._log().warn(
                f"[{label_prefix}] direct Cartesian move failed with no XY delta; "
                "retrying direct no-collision fallback"
            )
            return self._cartesian_move(
                self._make_pose(target_x, target_y, z, orientation),
                f"{label_prefix} (no-collision)",
                avoid_collisions=False,
                min_fraction=0.70,
                allow_partial=True,
                time_scale=time_scale,
            )

        self._log().warn(
            f"[{label_prefix}] direct Cartesian move failed; retrying staged XY fallback"
        )
        for axis_order in (("x", "y"), ("y", "x")):
            if self._move_xy_axis_order(
                current=current,
                target_x=target_x,
                target_y=target_y,
                z=z,
                orientation=orientation,
                label_prefix=label_prefix,
                axis_order=axis_order,
                time_scale=time_scale,
            ):
                if axis_order == ("y", "x"):
                    self._log().info(
                        f"[{label_prefix}] staged XY fallback succeeded with axis order Y->X"
                    )
                return True
        return False

    def _move_xy_axis_order(
        self,
        *,
        current,
        target_x: float,
        target_y: float,
        z: float,
        orientation,
        label_prefix: str,
        axis_order: tuple[str, str],
        time_scale: float | None = None,
    ) -> bool:
        start_x = float(current.position.x)
        start_y = float(current.position.y)
        order_suffix = "" if axis_order == ("x", "y") else ", alt-order"
        first_axis = axis_order[0]
        first_target_x = target_x if first_axis == "x" else start_x
        first_target_y = target_y if first_axis == "y" else start_y
        leg_targets = [
            (first_axis, first_target_x, first_target_y),
            (axis_order[1], target_x, target_y),
        ]

        current_x = start_x
        current_y = start_y
        for axis_name, leg_x, leg_y in leg_targets:
            label_base = f"{label_prefix} (leg {axis_name.upper()}{order_suffix}"
            step_targets = self._split_xy_leg_targets(
                start_x=current_x,
                start_y=current_y,
                target_x=leg_x,
                target_y=leg_y,
            )
            for step_index, (step_x, step_y) in enumerate(step_targets, start=1):
                step_suffix = (
                    f", step {step_index}/{len(step_targets)}" if len(step_targets) > 1 else ""
                )
                if not self._cartesian_move(
                    self._make_pose(step_x, step_y, z, orientation),
                    f"{label_base}{step_suffix})",
                    min_fraction=0.85,
                    allow_partial=False,
                    time_scale=time_scale,
                ):
                    if not self._cartesian_move(
                        self._make_pose(step_x, step_y, z, orientation),
                        f"{label_base}{step_suffix}, no-collision)",
                        avoid_collisions=False,
                        min_fraction=0.70,
                        allow_partial=True,
                        time_scale=time_scale,
                    ):
                        return False
            current_x = leg_x
            current_y = leg_y
        return True

    def _split_xy_leg_targets(
        self,
        *,
        start_x: float,
        start_y: float,
        target_x: float,
        target_y: float,
    ) -> list[tuple[float, float]]:
        delta_x = float(target_x) - float(start_x)
        delta_y = float(target_y) - float(start_y)
        span = max(abs(delta_x), abs(delta_y))
        if span <= 1e-9:
            return []
        segments = max(1, int(math.ceil(span / self.xy_axis_step_m)))
        return [
            (
                float(start_x) + (delta_x * idx / segments),
                float(start_y) + (delta_y * idx / segments),
            )
            for idx in range(1, segments + 1)
        ]

    def _make_pose(self, x: float, y: float, z: float, orientation):
        pose = self._Pose()
        pose.position.x = float(x)
        pose.position.y = float(y)
        pose.position.z = float(z)
        pose.orientation = orientation
        return pose

    def _make_orientation(self, qx: float, qy: float, qz: float, qw: float):
        orientation = self._Pose().orientation
        orientation.x = float(qx)
        orientation.y = float(qy)
        orientation.z = float(qz)
        orientation.w = float(qw)
        return orientation


UR5E_JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]
UR5E_TRAJECTORY_TOPIC = "/ur5e_joint_trajectory_controller/joint_trajectory"
UR5E_JOINT_STATES_TOPIC = "/joint_states"

XARM6_JOINT_NAMES = [
    "xarm6_joint1",
    "xarm6_joint2",
    "xarm6_joint3",
    "xarm6_joint4",
    "xarm6_joint5",
    "xarm6_joint6",
]
XARM6_JOINT_STATES_TOPIC = "/joint_states"


class UR5eGazeboController(GazeboPickPlaceController):
    """Config-driven UR5e Gazebo controller."""

    def __init__(
        self,
        trajectory_topic: str = UR5E_TRAJECTORY_TOPIC,
        joint_states_topic: str = UR5E_JOINT_STATES_TOPIC,
        *,
        controller_config: dict[str, Any] | None = None,
        named_positions: dict[str, Any] | None = None,
        execution_mode: str = "simulation",
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


class XArm6GazeboController(GazeboPickPlaceController):
    """Config-driven xArm6 Gazebo controller."""

    def __init__(
        self,
        *,
        trajectory_topic: str | None = None,
        joint_states_topic: str = XARM6_JOINT_STATES_TOPIC,
        controller_config: dict[str, Any] | None = None,
        named_positions: dict[str, Any] | None = None,
        execution_mode: str = "simulation",
    ) -> None:
        super().__init__(
            robot_name="xarm6",
            node_name=f"xarm6_controller_{os.getpid()}",
            controller_config=controller_config or {},
            named_positions=named_positions,
            execution_mode=execution_mode,
            arm_joint_names=XARM6_JOINT_NAMES,
            arm_trajectory_topic=trajectory_topic,
            joint_states_topic=joint_states_topic,
        )
