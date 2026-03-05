"""
Config-driven ROS2 pick/place controller.

This module intentionally avoids importing ROS2 packages at module import time.
All ROS2 imports happen lazily inside `init()` so non-ROS workflows can still
import the package.
"""

from __future__ import annotations

import importlib
import json
import logging
import os
import sys
import threading
import time
from typing import Any

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


class Ros2PickPlaceController:
    """
    Generic pick/place controller for a single robot.

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
        self.gripper_settle_sec = need_float(
            gripper, "settle_sec", "controller.gripper.settle_sec"
        )
        self.gripper_feedback_timeout_pad_sec = need_float(
            gripper,
            "feedback_timeout_pad_sec",
            "controller.gripper.feedback_timeout_pad_sec",
        )
        self.gripper_position_tol = need_float(
            gripper, "position_tolerance", "controller.gripper.position_tolerance"
        )

        self.service_detect_all = need_str(
            services, "detect_all", "controller.services.detect_all"
        )
        self.service_cartesian_path = need_str(
            services, "cartesian_path", "controller.services.cartesian_path"
        )
        self.service_execute_traj = need_str(
            services, "execute_trajectory", "controller.services.execute_trajectory"
        )
        self.service_attach = need_str(
            services, "attach", "controller.services.attach"
        )
        self.service_detach = need_str(
            services, "detach", "controller.services.detach"
        )
        self.service_set_entity_state = need_str(
            services, "set_entity_state", "controller.services.set_entity_state"
        )

        self.robot_model_name = need_str(
            attach_cfg, "robot_model_name", "controller.attach.robot_model_name"
        )
        raw_candidates = attach_cfg.get("attach_link_candidates")
        if isinstance(raw_candidates, list) and raw_candidates:
            self.attach_link_candidates = [str(v) for v in raw_candidates if str(v).strip()]
        else:
            self.attach_link_candidates = []
            self._config_errors.append("controller.attach.attach_link_candidates")
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
        self.release_retry_lift_m = max(
            0.0,
            opt_float(motion, "release_retry_lift_m", 0.005),
        )
        self.trajectory_time_scale = need_float(
            motion, "trajectory_time_scale", "controller.motion.trajectory_time_scale"
        )

        self.insertion_depth_m = need_float(
            parts_tuning, "insertion_depth_m", "controller.parts_tuning.insertion_depth_m"
        )

        if (
            self.pick_tcp_z_bias_min_m > 0.0
            and self.pick_tcp_z_bias_max_m > 0.0
            and self.pick_tcp_z_bias_min_m > self.pick_tcp_z_bias_max_m
        ):
            self._config_errors.append(
                "controller.motion.pick_tcp_z_bias_min_m<=pick_tcp_z_bias_max_m"
            )

        self._config_valid = not self._config_errors
        if not self._config_valid:
            self._last_failure_message = (
                "invalid controller_config; missing/invalid: "
                + ", ".join(sorted(set(self._config_errors)))
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
        self._gripper_pub = None
        self._arm_pub = None

        self._attach_srv = None
        self._detach_srv = None
        self._link_attacher_enabled = False
        self._attached_model: str | None = None
        self._attached_link: str | None = None

        self._joint_lock = threading.Lock()
        self._joint_positions: dict[str, float] = {}

        # Runtime pose context across phased calls.
        self._active_ctx: dict[str, Any] = {}
        self._last_start_pose = None

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
                "context is not valid" in msg.lower()
                or exc.__class__.__name__ == "RCLError"
            )
            if self._shutdown_requested and is_context_invalid:
                self._log().debug("Executor stopped during shutdown: %s", msg)
                return
            self._log().warning(
                "Executor spin terminated unexpectedly for %s: %s",
                self.node_name,
                msg,
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
            from rclpy.action import ActionClient
            from rclpy.callback_groups import ReentrantCallbackGroup
            from rclpy.executors import MultiThreadedExecutor

            from std_srvs.srv import Trigger
            from moveit_msgs.action import ExecuteTrajectory
            from moveit_msgs.srv import GetCartesianPath
            from gazebo_msgs.srv import SetEntityState
            from geometry_msgs.msg import Pose
            from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
            from builtin_interfaces.msg import Duration
            from sensor_msgs.msg import JointState
            import tf2_ros
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

        if self.gripper_topic and self.gripper_joint:
            self._gripper_pub = self._node.create_publisher(
                JointTrajectory, self.gripper_topic, 10
            )

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
        if not self._wait_service(
            self._cart_client, self.service_cartesian_path, deadline
        ):
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
            self._last_failure_message = (
                f"tf not ready for {self.frame_id} -> {self.ee_link}"
            )
            return False

        # Best effort gripper feedback readiness.
        feedback_deadline = min(deadline, time.monotonic() + 5.0)
        while time.monotonic() < feedback_deadline:
            if self._get_joint_position(self.gripper_joint) is not None:
                break
            time.sleep(0.05)

        self._services_ready = True
        self._last_failure_message = ""
        self._log().info("All services ready.")
        return True

    # ------------------------------------------------------------------ #
    # Legacy low-level API
    # ------------------------------------------------------------------ #
    def move_joints(self, positions: list[float], duration_sec: int = 2) -> bool:
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
        traj.joint_names = list(self.arm_joint_names)
        point = self._JointTrajectoryPoint()
        point.positions = [float(v) for v in positions]
        point.time_from_start = self._Duration(sec=max(1, int(duration_sec)))
        traj.points = [point]
        self._arm_pub.publish(traj)
        return True

    def open_gripper(self) -> bool:
        if not self.wait_for_services():
            return False
        return self._gripper_command(self.gripper_open, "OPEN")

    def close_gripper(self) -> bool:
        if not self.wait_for_services():
            return False
        return self._gripper_command(self.gripper_close, "CLOSE")

    # ------------------------------------------------------------------ #
    # Framework tool phases
    # ------------------------------------------------------------------ #
    def detect_parts(self) -> list[dict[str, Any]]:
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
        return parsed if isinstance(parsed, list) else []

    def pick_approach(
        self,
        *,
        origin_resource_location: str = "",
        part_name: str = "",
        product_geometry: dict[str, Any] | None = None,
        speed: float | None = None,
    ) -> dict[str, Any]:
        del origin_resource_location, speed  # location is resolved by perception in ROS2 path.
        if not self.wait_for_services():
            return {
                "success": False,
                "message": self._unavailable_message("services not ready"),
            }

        parts = self.detect_parts()
        if not parts:
            return {"success": False, "message": "no parts detected"}

        target = None
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
                    "message": (
                        f"requested part '{part_name}' not detected; "
                        f"detected={detected_names}"
                    ),
                }
        if target is None:
            target = parts[0]

        tx = _as_float(target.get("x"), 0.0)
        ty = _as_float(target.get("y"), 0.0)
        tz = _as_float(target.get("z"), 0.0)
        target_part_name = str(target.get("part_name") or part_name or "")

        geo = product_geometry or {}
        board_center = geo.get("board_center", {}) if isinstance(geo, dict) else {}
        board_center_z = _as_float(board_center.get("z"), 1.02)

        target_height = _as_float(geo.get("part_height_m"), 0.08)
        target_model = str(geo.get("model_name") or target.get("model_name") or "")

        ee = self._get_ee_pose()
        if ee is None:
            return {"success": False, "message": "cannot read current ee pose"}
        start_pose = self._make_pose(
            ee.position.x, ee.position.y, ee.position.z, ee.orientation
        )
        ori = ee.orientation

        ee_tcp_offset_z = self._get_ee_tcp_world_z_offset()
        pick_bias = max(
            self.pick_tcp_z_bias_min_m,
            min(self.pick_tcp_z_bias_max_m, target_height * 0.25),
        )
        pick_tcp_z_raw = tz + pick_bias
        pick_tcp_z = max(pick_tcp_z_raw, self.min_pick_tcp_z_m)
        pick_z = pick_tcp_z - ee_tcp_offset_z

        travel_z = max(
            ee.position.z,
            tz + self.approach_height_m,
            board_center_z + self.approach_height_m,
            pick_z + 0.05,
        )

        if not self._gripper_command(self.gripper_open, "OPEN"):
            return {"success": False, "message": "failed to open gripper"}

        if not self._move_xy_direct(
            tx, ty, travel_z, ori, "Move above part"
        ):
            return {"success": False, "message": "failed to move above part"}

        if not self._cartesian_move(
            self._make_pose(tx, ty, pick_z, ori),
            f"Descend to pick (EE z={pick_z:.3f}, TCP z={pick_tcp_z:.3f})",
        ):
            return {"success": False, "message": "failed to descend to pick"}

        self._last_start_pose = start_pose
        self._active_ctx = {
            "part_name": target_part_name,
            "target_model": target_model,
            "target_height": target_height,
            "tx": tx,
            "ty": ty,
            "tz": tz,
            "orientation": ori,
            "pick_tcp_z": pick_tcp_z,
            "pick_z": pick_z,
            "travel_z": travel_z,
            "ee_tcp_offset_z": ee_tcp_offset_z,
            "destination_location": None,
            "slot_x": None,
            "slot_y": None,
            "board_top_z": None,
            "place_z": None,
            "grasped": False,
        }
        return {
            "success": True,
            "message": f"Approached {target_part_name} for grasp",
            "part_name": target_part_name,
        }

    def pick_grasp(
        self,
        *,
        part_name: str = "",
        origin_resource_location: str = "",
        product_geometry: dict[str, Any] | None = None,
        gripper: str | None = None,
    ) -> dict[str, Any]:
        del origin_resource_location, product_geometry, gripper
        if not self.wait_for_services():
            return {
                "success": False,
                "message": self._unavailable_message("services not ready"),
            }
        ctx = self._active_ctx
        if not ctx:
            return {"success": False, "message": "pick_approach must run first"}
        if part_name and ctx.get("part_name") and part_name != ctx.get("part_name"):
            self._log().warn(
                f"pick_grasp requested {part_name} but active part is {ctx.get('part_name')}"
            )

        if not self._gripper_command(self.gripper_close, "CLOSE — grasping"):
            return {"success": False, "message": "failed to close gripper"}

        model_name = str(ctx.get("target_model") or "")
        if not self._attach_part(model_name):
            return {"success": False, "message": f"failed to attach {model_name}"}

        ctx["grasped"] = True
        return {"success": True, "message": f"Grasped {ctx.get('part_name', '')}"}

    def place_approach(
        self,
        *,
        destination_location: str = "",
        part_name: str = "",
        product_geometry: dict[str, Any] | None = None,
        speed: float | None = None,
    ) -> dict[str, Any]:
        del speed
        if not self.wait_for_services():
            return {
                "success": False,
                "message": self._unavailable_message("services not ready"),
            }
        ctx = self._active_ctx
        if not ctx or not ctx.get("grasped"):
            return {"success": False, "message": "pick_grasp must complete first"}
        if part_name and ctx.get("part_name") and part_name != ctx.get("part_name"):
            self._log().warn(
                f"place_approach requested {part_name} but active part is {ctx.get('part_name')}"
            )

        geo = product_geometry or {}
        board_center = geo.get("board_center", {}) if isinstance(geo, dict) else {}
        slot_xy = geo.get("slot_xy")
        if isinstance(slot_xy, (list, tuple)) and len(slot_xy) >= 2:
            bx = _as_float(board_center.get("x"), 0.0) + _as_float(slot_xy[0], 0.0)
            by = _as_float(board_center.get("y"), 0.0) + _as_float(slot_xy[1], 0.0)
        else:
            bx = _as_float(board_center.get("x"), ctx.get("tx", 0.0))
            by = _as_float(board_center.get("y"), ctx.get("ty", 0.0))

        board_top_z = _as_float(
            geo.get("slot_floor_z_m"),
            _as_float(board_center.get("z"), 1.025),
        )
        target_height = _as_float(geo.get("part_height_m"), ctx.get("target_height", 0.08))
        if geo.get("model_name"):
            ctx["target_model"] = str(geo.get("model_name"))

        tx = _as_float(ctx.get("tx"), 0.0)
        ty = _as_float(ctx.get("ty"), 0.0)
        travel_z = _as_float(ctx.get("travel_z"), 1.2)
        ori = ctx.get("orientation")

        if not self._cartesian_move(
            self._make_pose(tx, ty, travel_z, ori), "Lift with part"
        ):
            return {"success": False, "message": "failed to lift with part"}

        if not self._move_xy_direct(
            bx, by, travel_z, ori, "Move above destination"
        ):
            return {"success": False, "message": "failed to move above destination"}

        grasp_tcp_to_part_origin_z = _as_float(ctx.get("pick_tcp_z"), 0.0) - _as_float(
            ctx.get("tz"), 0.0
        )
        place_gap = self.place_surface_gap_m - self.insertion_depth_m
        place_part_origin_z = board_top_z + (target_height * 0.5) + place_gap
        place_tcp_z = place_part_origin_z + grasp_tcp_to_part_origin_z
        place_z = place_tcp_z - _as_float(ctx.get("ee_tcp_offset_z"), -0.17)

        if not self._cartesian_move(
            self._make_pose(bx, by, place_z, ori),
            f"Descend to place (EE z={place_z:.3f}, TCP z={place_tcp_z:.3f})",
            time_scale=self.release_descend_time_scale,
        ):
            return {"success": False, "message": "failed to descend to place"}

        ctx["destination_location"] = destination_location
        ctx["slot_x"] = bx
        ctx["slot_y"] = by
        ctx["board_top_z"] = board_top_z
        ctx["target_height"] = target_height
        ctx["place_z"] = place_z
        return {
            "success": True,
            "message": f"Positioned over {destination_location or 'target slot'}",
        }

    def place_insert(
        self,
        *,
        destination_location: str = "",
        part_name: str = "",
        product_geometry: dict[str, Any] | None = None,
        orientation: str | None = None,
    ) -> dict[str, Any]:
        del product_geometry, orientation
        if not self.wait_for_services():
            return {
                "success": False,
                "message": self._unavailable_message("services not ready"),
            }
        ctx = self._active_ctx
        if not ctx or ctx.get("place_z") is None:
            return {"success": False, "message": "place_approach must complete first"}
        if part_name and ctx.get("part_name") and part_name != ctx.get("part_name"):
            self._log().warn(
                f"place_insert requested {part_name} but active part is {ctx.get('part_name')}"
            )

        model_name = str(ctx.get("target_model") or "")
        slot_x = _as_float(ctx.get("slot_x"), 0.0)
        slot_y = _as_float(ctx.get("slot_y"), 0.0)
        board_top_z = _as_float(ctx.get("board_top_z"), 1.025)
        target_height = _as_float(ctx.get("target_height"), 0.08)
        place_z = _as_float(ctx.get("place_z"), board_top_z + target_height)
        travel_z = _as_float(ctx.get("travel_z"), 1.2)
        ori = ctx.get("orientation")

        time.sleep(self.release_preopen_settle_sec)

        open_ok = self._gripper_command(
            self.gripper_open,
            "OPEN — releasing",
            move_time_s=self.gripper_move_time_sec,
            wait_s=self.gripper_settle_sec,
            require_target=True,
        )
        if not open_ok:
            self._log().warn("Release open target not reached; retrying gripper open once")
            open_ok = self._gripper_command(
                self.gripper_open,
                "OPEN — releasing (retry)",
                move_time_s=max(0.8, self.gripper_move_time_sec * 1.5),
                wait_s=max(0.10, self.gripper_settle_sec),
                require_target=True,
            )
        if not open_ok:
            return {
                "success": False,
                "message": "failed to open gripper to release part",
            }
        time.sleep(self.release_postopen_settle_sec)

        detached = False
        attempts = max(1, 1 + self.release_detach_retry_count)
        for attempt_idx in range(attempts):
            if attempt_idx > 0:
                if attempt_idx == 1 and self.release_retry_lift_m > 0.0:
                    lift_z = place_z + self.release_retry_lift_m
                    lift_ok = self._cartesian_move(
                        self._make_pose(slot_x, slot_y, lift_z, ori),
                        f"Release micro-lift +{self.release_retry_lift_m * 1000.0:.1f}mm",
                        avoid_collisions=False,
                        min_fraction=0.70,
                        allow_partial=True,
                        time_scale=self.release_descend_time_scale,
                    )
                    if not lift_ok:
                        self._log().warn("Release micro-lift retry move failed")
                time.sleep(self.release_detach_retry_delay_sec)

            detached = self._detach_part(
                model_name,
                timeout_sec=self.release_detach_timeout_sec,
                attached_link_only=(attempt_idx == 0),
            )
            if detached:
                break
            self._log().warn(
                f"Detach attempt {attempt_idx + 1}/{attempts} failed for {model_name or 'held part'}"
            )
        if not detached:
            self._log().warn("Detach still failed after retries")
        time.sleep(self.release_postdetach_settle_sec)

        if detached and model_name:
            self._snap_part_to_slot(model_name, slot_x, slot_y, target_height, board_top_z)

        lift_ok = self._cartesian_move(
            self._make_pose(slot_x, slot_y, travel_z, ori),
            "Lift after place",
        )
        if not lift_ok:
            lift_ok = self._cartesian_move(
                self._make_pose(slot_x, slot_y, travel_z, ori),
                "Lift after place (no-collision)",
                avoid_collisions=False,
                min_fraction=0.70,
                allow_partial=True,
            )

        if not detached:
            self._detach_part(model_name, timeout_sec=self.release_detach_timeout_sec)

        placed_part = str(ctx.get("part_name") or part_name or "")
        self._active_ctx = {}
        status = "completed" if (detached and lift_ok) else "partial"
        return {
            "success": bool(detached and lift_ok),
            "status": status,
            "message": f"Placed {placed_part} at {destination_location or 'target'}",
        }

    def move_home(self) -> dict[str, Any]:
        if not self.wait_for_services():
            return {
                "success": False,
                "message": self._unavailable_message("services not ready"),
            }

        # 1) Prefer returning to remembered Cartesian start pose.
        if self._last_start_pose is not None:
            if self._cartesian_move(self._last_start_pose, "Return home"):
                return {"success": True, "message": "returned to remembered start pose"}

        # 2) Fallback to named joint-space home if publisher is configured.
        home = self.named_positions.get("home")
        if isinstance(home, (list, tuple)) and home and self._arm_pub:
            if self.move_joints([float(v) for v in home], duration_sec=4):
                return {"success": True, "message": "sent joint-space home command"}

        return {"success": False, "message": "no home pose available"}

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
        if not self._set_state_client.wait_for_service(timeout_sec=2.0):
            return {"success": False, "message": "set_entity_state service unavailable"}

        from gazebo_msgs.msg import EntityState

        state = EntityState()
        state.name = str(model_name)
        state.pose.position.x = float(x)
        state.pose.position.y = float(y)
        state.pose.position.z = float(z)
        state.pose.orientation.x = float(qx)
        state.pose.orientation.y = float(qy)
        state.pose.orientation.z = float(qz)
        state.pose.orientation.w = float(qw)
        state.reference_frame = str(reference_frame or "world")

        req = self._SetEntityState.Request()
        req.state = state
        future = self._set_state_client.call_async(req)
        response = self._wait_future(future, timeout_sec=5.0, label=f"set_entity_pose:{model_name}")
        if response and response.success:
            return {"success": True, "message": f"entity pose reset for {model_name}"}
        detail = getattr(response, "status_message", "") if response is not None else ""
        detail = str(detail or "").strip() or f"failed to set pose for {model_name}"
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
            self._log().warn(
                "Could not get world EE-to-TCP offset, using default -0.17m"
            )
            return -0.17

    def _on_joint_state(self, msg):
        with self._joint_lock:
            for name, pos in zip(msg.name, msg.position):
                self._joint_positions[name] = pos

    def _get_joint_position(self, joint_name: str) -> float | None:
        with self._joint_lock:
            return self._joint_positions.get(joint_name)

    def _wait_for_gripper_target(self, target: float, timeout_sec: float) -> bool:
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

        if not saw_feedback:
            self._log().warn(
                f"No joint-state feedback for '{self.gripper_joint}' while waiting gripper move"
            )
        else:
            self._log().warn(
                f"Gripper target not reached: target={target:.3f} current={float(last_pos):.3f}"
            )
        return False

    def _gripper_command(
        self,
        position: float,
        label: str,
        move_time_s: float | None = None,
        wait_s: float | None = None,
        require_target: bool = False,
    ) -> bool:
        if not self._gripper_pub:
            self._log().error("Gripper publisher is not configured")
            return False

        move_time_s = self.gripper_move_time_sec if move_time_s is None else float(move_time_s)
        wait_s = self.gripper_settle_sec if wait_s is None else float(wait_s)

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
        time.sleep(0.05)
        self._gripper_pub.publish(traj)

        feedback_timeout = max(move_time_s + self.gripper_feedback_timeout_pad_sec, 1.0)
        reached = self._wait_for_gripper_target(position, feedback_timeout)
        if require_target and not reached:
            self._log().error(
                f"Gripper command did not reach required target: target={position:.3f}"
            )
            return False
        time.sleep(max(0.0, wait_s))
        return True

    def _wait_future(self, future, timeout_sec: float, label: str):
        deadline = time.monotonic() + timeout_sec
        while self._rclpy.ok() and not future.done() and time.monotonic() < deadline:
            time.sleep(0.01)
        if not future.done():
            try:
                future.cancel()
            except Exception:
                pass
            self._log().error(f"[{label}] timed out")
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
            total_ns = (
                int(point.time_from_start.sec) * 1_000_000_000
                + int(point.time_from_start.nanosec)
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
            response = self._wait_future(
                future, timeout_sec=5.0, label=f"attach:{link_name}"
            )
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
        if self._attached_link:
            links_to_try.append(self._attached_link)
        if not (attached_link_only and links_to_try):
            if self.primary_attach_link and self.primary_attach_link not in links_to_try:
                links_to_try.append(self.primary_attach_link)
            for link in self.attach_link_candidates:
                if link not in links_to_try:
                    links_to_try.append(link)
        links_to_try = links_to_try[: max(1, self.detach_max_link_attempts)]

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
            )
            if response and response.success:
                self._attached_model = None
                self._attached_link = None
                return True

        if log_failure:
            self._log().error(f"Failed to detach {target_model}")
        return False

    def _snap_part_to_slot(
        self, model_name: str, slot_x: float, slot_y: float, part_height: float, board_top_z: float
    ) -> bool:
        if not self._set_state_client.wait_for_service(timeout_sec=2.0):
            return False

        from gazebo_msgs.msg import EntityState

        state = EntityState()
        state.name = model_name
        state.pose.position.x = slot_x
        state.pose.position.y = slot_y
        state.pose.position.z = board_top_z + (part_height * 0.5)
        state.pose.orientation.w = 1.0
        state.reference_frame = "world"

        req = self._SetEntityState.Request()
        req.state = state

        future = self._set_state_client.call_async(req)
        response = self._wait_future(future, timeout_sec=5.0, label="snap_to_slot")
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
            self._log().error(f"[{label}] planning response timed out")
            return False
        if response.fraction < min_fraction:
            self._log().error(
                f"[{label}] planning fraction too low: {response.fraction:.3f} < {min_fraction:.3f}"
            )
            return False
        if response.fraction < 0.999 and not allow_partial:
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
            self._log().error(f"[{label}] trajectory goal rejected by execute action")
            return False

        result_future = goal_handle.get_result_async()
        result = self._wait_future(result_future, timeout_sec=30.0, label=f"result:{label}")
        code = result.result.error_code.val if result else None
        if code != 1:
            self._log().error(f"[{label}] execute_trajectory failed with error_code={code}")
        return code == 1

    def _move_xy_direct(
        self, target_x: float, target_y: float, z: float, orientation, label_prefix: str
    ) -> bool:
        if self._cartesian_move(
            self._make_pose(target_x, target_y, z, orientation), label_prefix
        ):
            return True

        current = self._get_ee_pose()
        if current is None:
            return False

        if not self._cartesian_move(
            self._make_pose(target_x, current.position.y, z, orientation),
            f"{label_prefix} (leg X)",
            min_fraction=0.85,
            allow_partial=False,
        ):
            if not self._cartesian_move(
                self._make_pose(target_x, current.position.y, z, orientation),
                f"{label_prefix} (leg X, no-collision)",
                avoid_collisions=False,
                min_fraction=0.70,
                allow_partial=True,
            ):
                return False

        if not self._cartesian_move(
            self._make_pose(target_x, target_y, z, orientation),
            f"{label_prefix} (leg Y)",
            min_fraction=0.85,
            allow_partial=False,
        ):
            if not self._cartesian_move(
                self._make_pose(target_x, target_y, z, orientation),
                f"{label_prefix} (leg Y, no-collision)",
                avoid_collisions=False,
                min_fraction=0.70,
                allow_partial=True,
            ):
                return False

        return True

    def _make_pose(self, x: float, y: float, z: float, orientation):
        pose = self._Pose()
        pose.position.x = float(x)
        pose.position.y = float(y)
        pose.position.z = float(z)
        pose.orientation = orientation
        return pose
