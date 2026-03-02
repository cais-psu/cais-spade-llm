"""
Pick and place: UR5e picks a camera-detected part and places it on the assembly board.

Uses Cartesian path planning for straight-line moves + gripper control.

Sequence:
  1. Open gripper
  2. Move X/Y above the target part (keeping current Z)
  3. Descend to pick height
  4. Close gripper
  5. Lift back up
  6. Move X/Y above assembly board center
  7. Descend to place height
  8. Open gripper
  9. Lift back up

Prerequisites:
    ros2 launch xarm_gazebo dual_moveit_gazebo.launch.py
    (perception node starts automatically with run_perception:=true)

Usage:
    python3.10 test/test_ur5e_camera_move.py --part LG
    python3.10 test/test_ur5e_camera_move.py --part LRP
"""
import argparse
import importlib
import json
import os
import sys
import time
import threading
import math

import rclpy
from rclpy.node import Node
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

# UR5e MoveIt config
GROUP_NAME = "ur5e_ur_manipulator"
EE_LINK = "ur5e_tool0"
TCP_LINK = "ur5e_rg2_gripper_tcp"  # gripper TCP frame (used by link attacher)
FRAME_ID = "world"

# Gripper config (from keyboard_teleop.py)
GRIPPER_JOINT = "ur5e_rg2_finger_width"
GRIPPER_TOPIC = "/ur5e_rg2_gripper_traj_controller/joint_trajectory"
GRIPPER_OPEN = 0.11
GRIPPER_CLOSE = 0.02

# Heights (meters, above target surface).
# The link attacher triggers when ur5e RG2 TCP is close enough to a part.
# We query TF at runtime to find the world-frame EE-to-TCP offset, then
# compute a safe pick height for the EE.
APPROACH_HEIGHT = 0.20   # safe travel height above part/board
PICK_TCP_Z_BIAS_MAX = 0.02   # upper bound bias above detected part origin
PICK_TCP_Z_BIAS_MIN = 0.003  # lower bound bias for very short parts
MIN_PICK_TCP_Z = 1.07    # safety floor to avoid table/printer-bed contact
PLACE_SURFACE_GAP = -0.01  # deeper insertion into slot (~10mm push to clear 6mm socket rim)
RELEASE_PREOPEN_SETTLE_SEC = 0.15
RELEASE_POSTOPEN_SETTLE_SEC = 0.60
RELEASE_POSTDETACH_SETTLE_SEC = 0.35

# Assembly board center (from table.world)
ASSEMBLY_BOARD = {"x": 0.0, "y": 0.0, "z": 1.02}
ASSEMBLY_BOARD_THICKNESS = 0.01
# Board top surface Z: board center 1.02 + half-thickness 0.005 = 1.025
# Parts are teleported ON TOP of the solid board (no recessed slots).
ASSEMBLY_BOARD_SOCKET_FLOOR_Z = 1.025
ASSEMBLY_SLOTS = {
    # Board-local slot centers (x-spacing 10cm, y-spacing 8cm).
    "SG": (-0.10, 0.08),
    "MG": (0.0, 0.08),
    "LG": (0.10, 0.08),
    "SRP": (-0.10, 0.0),
    "MRP": (0.0, 0.0),
    "LRP": (0.10, 0.0),
    "SCP": (-0.10, -0.08),
    "MCP": (0.0, -0.08),
    "LCP": (0.10, -0.08),
}

# Link attacher config
ROBOT_MODEL_NAME = "dual_robot"
PART_MODEL_MAP = {
    "SG": "gear_small",
    "MG": "gear_medium",
    "LG": "gear_large",
    "SRP": "rect_pin_small",
    "MRP": "rect_pin_medium",
    "LRP": "rect_pin_large",
    "SCP": "circ_pin_small",
    "MCP": "circ_pin_medium",
    "LCP": "circ_pin_large",
}
PART_HEIGHTS = {
    "SG": 0.01,
    "MG": 0.015,
    "LG": 0.02,
    "SRP": 0.06,
    "MRP": 0.08,
    "LRP": 0.10,
    "SCP": 0.06,
    "MCP": 0.08,
    "LCP": 0.10,
}
# Extra insertion depth below nominal place pose (meters), shared for all parts.
INSERTION_DEPTH_M = 0.0025
ATTACH_LINK_CANDIDATES = [
    # Prefer wrist/tool links for stable placement.
    "ur5e_wrist_3_link",
    "ur5e_tool0",
    "ur5e_rg2_gripper_tcp",
]
PRIMARY_ATTACH_LINK = "ur5e_wrist_3_link"
DETACH_TIMEOUT_SEC = 2.0
DETACH_MAX_LINK_ATTEMPTS = 3

# Motion/gripper pacing.
TRAJECTORY_TIME_SCALE = 1.0
GRIPPER_MOVE_TIME_SEC = 0.4
GRIPPER_SETTLE_SEC = 0.08
GRIPPER_FEEDBACK_TIMEOUT_PAD_SEC = 1.5
GRIPPER_POSITION_TOL = 0.01


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
            os.path.expanduser(f"~/ros2_ws/install/linkattacher_msgs/local/lib/{py_ver}/dist-packages"),
            os.path.expanduser(f"~/ros2_ws/install/ros2_linkattacher/local/lib/{py_ver}/dist-packages"),
            f"/opt/ros/humble/lib/{py_ver}/dist-packages",
        ]
        for path in candidates:
            if os.path.isdir(path) and path not in sys.path:
                sys.path.append(path)
        try:
            return _load_srvs()
        except Exception:
            return None, None


class UR5ePickPlace(Node):

    def __init__(self, target_part: str, place_surface_gap: float, return_home: bool):
        super().__init__("ur5e_pick_place")
        self._target_part = target_part
        self._place_surface_gap = place_surface_gap
        self._return_home = return_home

        # Use reentrant callback group so service calls don't deadlock
        self._cb_group = ReentrantCallbackGroup()

        # TF for reading current EE pose
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Cartesian path planning service
        self._cart_client = self.create_client(
            GetCartesianPath, "/compute_cartesian_path", callback_group=self._cb_group
        )

        # Trajectory execution action
        self._exec_client = ActionClient(
            self, ExecuteTrajectory, "/execute_trajectory", callback_group=self._cb_group
        )

        # Perception service client
        self._detect_client = self.create_client(
            Trigger, "/detect_all", callback_group=self._cb_group
        )

        # Optional IFRA link attacher service clients
        self._attach_srv, self._detach_srv = _import_linkattacher_srvs()
        if self._attach_srv and self._detach_srv:
            self._attach_client = self.create_client(
                self._attach_srv, "/ATTACHLINK", callback_group=self._cb_group
            )
            self._detach_client = self.create_client(
                self._detach_srv, "/DETACHLINK", callback_group=self._cb_group
            )
            self._link_attacher_enabled = True
        else:
            self._attach_client = None
            self._detach_client = None
            self._link_attacher_enabled = False
            self.get_logger().warn(
                "linkattacher_msgs.srv not importable; ATTACHLINK/DETACHLINK disabled"
            )
        self._attached_model = None
        self._attached_link = None

        # Gazebo set_entity_state client (snap part into slot after release)
        self._set_state_client = self.create_client(
            SetEntityState, "/set_entity_state", callback_group=self._cb_group
        )

        # Gripper publisher
        self._gripper_pub = self.create_publisher(JointTrajectory, GRIPPER_TOPIC, 10)
        self._joint_positions = {}
        self._joint_lock = threading.Lock()
        self._joint_state_sub = self.create_subscription(
            JointState, "/joint_states", self._on_joint_state, 50
        )

    def wait_for_services(self):
        """Block until all required services/actions are available."""
        self.get_logger().info("Waiting for services...")
        self._detect_client.wait_for_service()
        self._cart_client.wait_for_service()
        self._exec_client.wait_for_server()
        if self._link_attacher_enabled:
            self._attach_client.wait_for_service()
            self._detach_client.wait_for_service()
        # Wait for TF buffer to fill
        self.get_logger().info("Waiting for TF...")
        while not self.tf_buffer.can_transform(FRAME_ID, EE_LINK, rclpy.time.Time()):
            time.sleep(0.5)
        self.get_logger().info("Waiting for gripper joint state...")
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and self._get_joint_position(GRIPPER_JOINT) is None:
            time.sleep(0.05)
        if self._get_joint_position(GRIPPER_JOINT) is None:
            self.get_logger().warn(
                f"Joint state for '{GRIPPER_JOINT}' not received yet; using timed waits as fallback"
            )
        self.get_logger().info("All services ready.")

    def _get_ee_pose(self) -> Pose | None:
        """Get current EE pose from TF."""
        try:
            t = self.tf_buffer.lookup_transform(FRAME_ID, EE_LINK, rclpy.time.Time())
            pose = Pose()
            pose.position.x = t.transform.translation.x
            pose.position.y = t.transform.translation.y
            pose.position.z = t.transform.translation.z
            pose.orientation = t.transform.rotation
            return pose
        except Exception as e:
            self.get_logger().error(f"TF lookup failed: {e}")
            return None

    def _get_ee_tcp_world_z_offset(self) -> float:
        """Get world-frame Z offset (tcp_z - ee_z)."""
        try:
            ee_tf = self.tf_buffer.lookup_transform(FRAME_ID, EE_LINK, rclpy.time.Time())
            tcp_tf = self.tf_buffer.lookup_transform(FRAME_ID, TCP_LINK, rclpy.time.Time())
            return tcp_tf.transform.translation.z - ee_tf.transform.translation.z
        except Exception:
            self.get_logger().warn("Could not get world EE-to-TCP offset, using default -0.17m")
            return -0.17  # conservative default

    def _on_joint_state(self, msg: JointState):
        """Cache latest joint positions for gripper completion checks."""
        with self._joint_lock:
            for name, pos in zip(msg.name, msg.position):
                self._joint_positions[name] = pos

    def _get_joint_position(self, joint_name: str) -> float | None:
        with self._joint_lock:
            return self._joint_positions.get(joint_name)

    def _wait_for_gripper_target(self, target: float, timeout_sec: float) -> bool:
        """Wait until gripper joint reaches target position within tolerance."""
        deadline = time.monotonic() + timeout_sec
        saw_feedback = False
        last_pos = None
        while rclpy.ok() and time.monotonic() < deadline:
            pos = self._get_joint_position(GRIPPER_JOINT)
            if pos is not None:
                saw_feedback = True
                last_pos = pos
                if abs(pos - target) <= GRIPPER_POSITION_TOL:
                    return True
            time.sleep(0.02)

        if not saw_feedback:
            self.get_logger().warn(
                f"No joint-state feedback for '{GRIPPER_JOINT}' while waiting gripper move"
            )
        else:
            self.get_logger().warn(
                f"Gripper target not reached: target={target:.3f}, current={last_pos:.3f}"
            )
        return False

    def _gripper_command(
        self,
        position: float,
        label: str,
        move_time_s: float = GRIPPER_MOVE_TIME_SEC,
        wait_s: float = GRIPPER_SETTLE_SEC,
    ):
        """Send gripper to a position (UR5e RG2: open~0.11, closed~0.02)."""
        self.get_logger().info(f"Gripper: {label} (position={position:.2f})")
        traj = JointTrajectory()
        traj.joint_names = [GRIPPER_JOINT]
        point = JointTrajectoryPoint()
        point.positions = [position]
        move_time_s = max(0.05, float(move_time_s))
        sec = int(move_time_s)
        nsec = int((move_time_s - sec) * 1_000_000_000)
        point.time_from_start = Duration(sec=sec, nanosec=nsec)
        traj.points = [point]
        self._gripper_pub.publish(traj)
        # Publish twice to reduce one-shot command drops.
        time.sleep(0.05)
        self._gripper_pub.publish(traj)
        feedback_timeout = max(move_time_s + GRIPPER_FEEDBACK_TIMEOUT_PAD_SEC, 1.0)
        self._wait_for_gripper_target(position, feedback_timeout)
        time.sleep(max(0.0, float(wait_s)))

    def _wait_future(self, future, timeout_sec: float, label: str):
        """Wait for async result without nested spinning (executor thread handles callbacks)."""
        deadline = time.monotonic() + timeout_sec
        while rclpy.ok() and not future.done() and time.monotonic() < deadline:
            time.sleep(0.01)
        if not future.done():
            try:
                future.cancel()
            except Exception:
                pass
            self.get_logger().error(f"[{label}] timed out")
            return None
        return future.result()

    def _scale_trajectory_timing(self, solution, scale: float):
        """Slow down trajectory execution by stretching time and scaling derivatives."""
        if scale <= 1.0 or solution is None:
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
        """Attach target part to UR5e gripper via IFRA link attacher."""
        if not self._link_attacher_enabled:
            return True
        if not model_name:
            self.get_logger().error("Cannot attach: empty model name")
            return False

        # If we still track a prior part, detach it first.
        if self._attached_model and self._attached_model != model_name:
            self.get_logger().warn(
                f"Detaching stale attached model '{self._attached_model}' before new attach"
            )
            self._detach_part(self._attached_model)

        for link_name in ATTACH_LINK_CANDIDATES:
            req = self._attach_srv.Request()
            req.model1_name = ROBOT_MODEL_NAME
            req.link1_name = link_name
            req.model2_name = model_name
            req.link2_name = "link"

            future = self._attach_client.call_async(req)
            response = self._wait_future(future, timeout_sec=5.0, label=f"attach:{link_name}")
            if response is None:
                self.get_logger().warn(f"Attach timeout for link {link_name}")
                continue
            if response and response.success:
                self._attached_model = model_name
                self._attached_link = link_name
                self.get_logger().info(
                    f"Attached {model_name} using {link_name}: {response.message}"
                )
                return True

            msg = response.message if response else "no response"
            if "already attached to another link" in msg.lower():
                self.get_logger().warn(
                    f"Attach reports stale existing attachment for {model_name}; "
                    "attempting detach recovery before retry"
                )
                self._detach_model_from_any_link(model_name)
                continue
            self.get_logger().warn(
                f"Attach attempt failed ({link_name} -> {model_name}): {msg}"
            )

        self.get_logger().error(f"Failed to attach {model_name} with all link candidates")
        return False

    def _detach_model_from_any_link(self, target_model: str) -> bool:
        """Best-effort detach of a model from any candidate gripper link."""
        if not self._link_attacher_enabled:
            return True
        if not target_model:
            return False
        if not self._detach_client.wait_for_service(timeout_sec=0.5):
            self.get_logger().warn("/DETACHLINK service unavailable during stale detach recovery")
            return False

        detached_any = False
        for link_name in ATTACH_LINK_CANDIDATES:
            req = self._detach_srv.Request()
            req.model1_name = ROBOT_MODEL_NAME
            req.link1_name = link_name
            req.model2_name = target_model
            req.link2_name = "link"

            future = self._detach_client.call_async(req)
            response = self._wait_future(
                future,
                timeout_sec=DETACH_TIMEOUT_SEC,
                label=f"detach-recover:{target_model}:{link_name}",
            )
            if response and response.success:
                detached_any = True
                self.get_logger().info(
                    f"Recovered stale attachment: detached {target_model} via {link_name}"
                )

        if detached_any and self._attached_model == target_model:
            self._attached_model = None
            self._attached_link = None
        return detached_any

    def _detach_part(self, model_name: str = "") -> bool:
        """Detach currently attached part (or provided model) from UR5e gripper."""
        if not self._link_attacher_enabled:
            return True

        target_model = model_name or self._attached_model
        if not target_model:
            self.get_logger().info("No attached model tracked; skipping detach")
            return True

        if not self._detach_client.wait_for_service(timeout_sec=0.5):
            self.get_logger().error("/DETACHLINK service unavailable")
            return False

        links_to_try = []
        if self._attached_link:
            links_to_try.append(self._attached_link)
        if PRIMARY_ATTACH_LINK not in links_to_try:
            links_to_try.append(PRIMARY_ATTACH_LINK)
        for link in ATTACH_LINK_CANDIDATES:
            if link not in links_to_try:
                links_to_try.append(link)
        links_to_try = links_to_try[:DETACH_MAX_LINK_ATTEMPTS]

        for link_name in links_to_try:
            req = self._detach_srv.Request()
            req.model1_name = ROBOT_MODEL_NAME
            req.link1_name = link_name
            req.model2_name = target_model
            req.link2_name = "link"

            future = self._detach_client.call_async(req)
            response = self._wait_future(
                future,
                timeout_sec=DETACH_TIMEOUT_SEC,
                label=f"detach:{link_name}",
            )
            if response is None:
                self.get_logger().warn(f"Detach timeout for link {link_name}")
                continue
            if response and response.success:
                self.get_logger().info(
                    f"Detached {target_model} using {link_name}: {response.message}"
                )
                self._attached_model = None
                self._attached_link = None
                return True

            msg = response.message if response else "no response"
            self.get_logger().warn(
                f"Detach attempt failed ({link_name} -> {target_model}): {msg}"
            )

        self.get_logger().error(f"Failed to detach {target_model}")
        return False

    def _snap_part_to_slot(
        self, model_name: str, slot_x: float, slot_y: float, part_height: float
    ) -> bool:
        """Teleport a released part into its exact slot position via /set_entity_state.

        Gazebo ODE cannot reliably resolve peg-in-hole insertion, so we snap the
        part to the socket floor after releasing it from the gripper.
        """
        if not self._set_state_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn("/set_entity_state unavailable; skipping snap-to-slot")
            return False

        from gazebo_msgs.msg import EntityState
        state = EntityState()
        state.name = model_name
        state.pose.position.x = slot_x
        state.pose.position.y = slot_y
        state.pose.position.z = ASSEMBLY_BOARD_SOCKET_FLOOR_Z + (part_height * 0.5)
        state.pose.orientation.w = 1.0
        state.reference_frame = "world"

        req = SetEntityState.Request()
        req.state = state

        future = self._set_state_client.call_async(req)
        response = self._wait_future(future, timeout_sec=5.0, label="snap_to_slot")
        if response and response.success:
            self.get_logger().info(
                f"Snapped {model_name} into slot at ({slot_x:.3f}, {slot_y:.3f}, "
                f"{state.pose.position.z:.3f})"
            )
            return True
        msg = getattr(response, "status_message", "unknown") if response else "timeout"
        self.get_logger().warn(f"Snap-to-slot failed for {model_name}: {msg}")
        return False

    def _cartesian_move(
        self,
        target: Pose,
        label: str = "",
        avoid_collisions: bool = True,
        min_fraction: float = 0.9,
        allow_partial: bool = False,
    ) -> bool:
        """Plan and execute a Cartesian straight-line move to target pose."""
        request = GetCartesianPath.Request()
        request.header.frame_id = FRAME_ID
        request.header.stamp = self.get_clock().now().to_msg()
        request.group_name = GROUP_NAME
        request.link_name = EE_LINK
        request.waypoints = [target]
        request.max_step = 0.01  # 1cm interpolation steps
        request.jump_threshold = 0.0
        request.avoid_collisions = avoid_collisions
        request.start_state.is_diff = True

        future = self._cart_client.call_async(request)
        response = self._wait_future(future, timeout_sec=10.0, label=f"plan:{label}")
        if response is None:
            self.get_logger().error(f"[{label}] Cartesian path planning timed out")
            return False

        if response is None or response.fraction < min_fraction:
            frac = response.fraction if response else 0
            self.get_logger().error(f"[{label}] Cartesian path incomplete ({frac:.0%})")
            return False

        if response.fraction < 0.999 and allow_partial:
            self.get_logger().warn(
                f"[{label}] Executing partial Cartesian path ({response.fraction:.0%})"
            )

        self.get_logger().info(f"[{label}] Path planned ({response.fraction:.0%}), executing...")

        # Execute the trajectory
        exec_goal = ExecuteTrajectory.Goal()
        self._scale_trajectory_timing(response.solution, TRAJECTORY_TIME_SCALE)
        exec_goal.trajectory = response.solution

        send_future = self._exec_client.send_goal_async(exec_goal)
        goal_handle = self._wait_future(send_future, timeout_sec=10.0, label=f"send:{label}")
        if not goal_handle or not goal_handle.accepted:
            self.get_logger().error(f"[{label}] Trajectory execution rejected")
            return False

        result_future = goal_handle.get_result_async()
        result = self._wait_future(result_future, timeout_sec=30.0, label=f"result:{label}")
        code = result.result.error_code.val if result else None
        if code != 1:
            self.get_logger().error(f"[{label}] Execution failed with error code: {code}")
            return False

        self.get_logger().info(f"[{label}] Done.")
        return True

    def _lift_to_z(
        self,
        x: float,
        y: float,
        target_z: float,
        orientation,
        label_prefix: str,
        step_m: float = 0.02,
    ) -> bool:
        """Raise EE in small Cartesian Z steps to improve robustness near slots/contacts."""
        z_finish_tol = 0.001
        # If we are within about half a step, consider lift complete.
        near_target_tol = max(0.006, step_m * 0.5)
        min_progress = 0.0002
        max_stall_retries = 3

        current = self._get_ee_pose()
        if current is None:
            self.get_logger().error(f"[{label_prefix}] Cannot read current EE pose")
            return False

        if current.position.z >= target_z - z_finish_tol:
            return True

        step_idx = 0
        stall_count = 0
        while current.position.z < target_z - z_finish_tol:
            step_idx += 1
            next_z = min(current.position.z + step_m, target_z)
            target_pose = self._make_pose(x, y, next_z, orientation)
            before_z = current.position.z

            # Accept partial progress in challenging near-contact regions.
            ok = self._cartesian_move(
                target_pose,
                f"{label_prefix} step {step_idx}",
                avoid_collisions=True,
                min_fraction=0.60,
                allow_partial=True,
            )
            if not ok:
                self.get_logger().warn(
                    f"[{label_prefix}] Step {step_idx} failed with collision checks, retrying no-collision"
                )
                ok = self._cartesian_move(
                    target_pose,
                    f"{label_prefix} step {step_idx} (no-collision)",
                    avoid_collisions=False,
                    min_fraction=0.60,
                    allow_partial=True,
                )
            if not ok:
                return False

            current = self._get_ee_pose()
            if current is None:
                self.get_logger().error(f"[{label_prefix}] Lost TF after step {step_idx}")
                return False

            remaining = target_z - current.position.z
            progressed = current.position.z - before_z
            if progressed <= min_progress:
                if remaining <= near_target_tol:
                    self.get_logger().warn(
                        f"[{label_prefix}] Near target within tolerance "
                        f"(remaining {remaining:.4f}m), accepting"
                    )
                    return True
                stall_count += 1
                self.get_logger().warn(
                    f"[{label_prefix}] Low upward progress at step {step_idx} "
                    f"(dz={progressed:.4f}m, remaining={remaining:.4f}m, retry {stall_count}/{max_stall_retries})"
                )
                if stall_count >= max_stall_retries:
                    self.get_logger().error(
                        f"[{label_prefix}] No upward progress after {stall_count} retries"
                    )
                    return False
            else:
                stall_count = 0

        return True

    def _move_xy_direct(self, target_x: float, target_y: float, z: float, orientation, label_prefix: str) -> bool:
        """
        Prefer a single direct Cartesian XY move.
        If that fails, use at most two short fallback legs (X then Y).
        """
        if self._cartesian_move(
            self._make_pose(target_x, target_y, z, orientation),
            label_prefix,
        ):
            return True

        self.get_logger().warn(f"{label_prefix} failed; retrying with 2-leg fallback")
        current = self._get_ee_pose()
        if current is None:
            self.get_logger().error(f"[{label_prefix}] Cannot read current EE pose for fallback")
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

    def _make_pose(self, x: float, y: float, z: float, orientation) -> Pose:
        """Create a Pose with given position and orientation."""
        pose = Pose()
        pose.position.x = x
        pose.position.y = y
        pose.position.z = z
        pose.orientation = orientation
        return pose

    def run(self):
        """Full pick-and-place workflow."""
        # --- Detect parts ---
        self.get_logger().info("Calling /detect_all...")
        future = self._detect_client.call_async(Trigger.Request())
        result = self._wait_future(future, timeout_sec=10.0, label="detect_all")
        if not result or not result.success:
            msg = result.message if result else "timeout"
            self.get_logger().error(f"/detect_all failed: {msg}")
            return

        parts = json.loads(result.message)
        if not parts:
            self.get_logger().error("No parts detected!")
            return

        self.get_logger().info(f"Detected {len(parts)} parts:")
        for p in parts:
            self.get_logger().info(
                f"  {p['part_name']}: x={p['x']:.3f} y={p['y']:.3f} z={p['z']:.3f}"
            )

        # --- Select target ---
        if self._target_part:
            target = next(
                (p for p in parts if p["part_name"] == self._target_part), None
            )
            if target is None:
                self.get_logger().error(
                    f"Part '{self._target_part}' not found. "
                    f"Available: {[p['part_name'] for p in parts]}"
                )
                return
        else:
            target = parts[0]

        tx, ty, tz = target["x"], target["y"], target["z"]
        target_part_name = target.get("part_name", "")
        target_model = target.get("model_name") or PART_MODEL_MAP.get(target_part_name, "")
        target_height = PART_HEIGHTS.get(target_part_name, 0.08)
        board_top_z = ASSEMBLY_BOARD["z"] + ASSEMBLY_BOARD_THICKNESS * 0.5
        slot_xy_local = ASSEMBLY_SLOTS.get(target_part_name, (0.0, 0.0))
        slot_x = ASSEMBLY_BOARD["x"] + slot_xy_local[0]
        slot_y = ASSEMBLY_BOARD["y"] + slot_xy_local[1]

        # Get current EE pose (for orientation and travel height)
        ee = self._get_ee_pose()
        if ee is None:
            self.get_logger().error("Cannot read current EE pose")
            return
        start_pose = self._make_pose(
            ee.position.x,
            ee.position.y,
            ee.position.z,
            ee.orientation,
        )

        ori = ee.orientation  # preserve tool-down orientation throughout

        # Compute pick height in world Z.
        # /detect_all returns Gazebo model origin (typically center), so bias up slightly.
        ee_tcp_offset_z = self._get_ee_tcp_world_z_offset()  # tcp_z - ee_z
        pick_bias = max(PICK_TCP_Z_BIAS_MIN, min(PICK_TCP_Z_BIAS_MAX, target_height * 0.25))
        pick_tcp_z_raw = tz + pick_bias
        pick_tcp_z = max(pick_tcp_z_raw, MIN_PICK_TCP_Z)
        pick_z = pick_tcp_z - ee_tcp_offset_z  # ee_z that puts TCP at pick_tcp_z

        # Keep travel moves high enough above both pick and place regions.
        travel_z = max(
            ee.position.z,
            tz + APPROACH_HEIGHT,
            ASSEMBLY_BOARD["z"] + APPROACH_HEIGHT,
            pick_z + 0.05,
        )

        self.get_logger().info(
            f"Current EE: x={ee.position.x:.3f} y={ee.position.y:.3f} z={travel_z:.3f}"
        )
        self.get_logger().info(
            f"World EE->TCP z offset: {ee_tcp_offset_z:.3f}m "
            f"(pick bias={pick_bias:.3f}, pick TCP z raw={pick_tcp_z_raw:.3f}, "
            f"clamped={pick_tcp_z:.3f}, pick EE z={pick_z:.3f})"
        )
        self.get_logger().info(
            f"Target: {target['part_name']} ({target_model}) at ({tx:.3f}, {ty:.3f}, {tz:.3f}), "
            f"height={target_height:.3f}m"
        )
        self.get_logger().info(
            f"Assembly slot: ({slot_x:.3f}, {slot_y:.3f}) for {target_part_name}"
        )
        self.get_logger().info("=" * 50)

        # === PICK SEQUENCE ===

        # Step 1: Open gripper
        self._gripper_command(GRIPPER_OPEN, "OPEN")

        # Step 2: Move X/Y above part (keep travel height)
        if not self._cartesian_move(
            self._make_pose(tx, ty, travel_z, ori),
            "Move above part"
        ):
            return

        # Step 3: Descend — TCP goes to part height
        if not self._cartesian_move(
            self._make_pose(tx, ty, pick_z, ori),
            f"Descend to pick (EE z={pick_z:.3f}, TCP target z={pick_tcp_z:.3f})"
        ):
            return

        # Step 4: Close gripper (grasp)
        self._gripper_command(GRIPPER_CLOSE, "CLOSE — grasping")
        if not self._attach_part(target_model):
            self.get_logger().error("Attach failed after grasp; aborting to avoid dropping part")
            return

        # Step 5: Lift back to travel height
        if not self._cartesian_move(
            self._make_pose(tx, ty, travel_z, ori),
            "Lift with part"
        ):
            return

        # === PLACE SEQUENCE ===

        bx, by = slot_x, slot_y

        # Step 6: Move X/Y above assembly board (single-step preferred).
        if not self._move_xy_direct(
            bx, by, travel_z, ori, "Move above assembly board"
        ):
            return

        # Step 7: Descend to place height
        grasp_tcp_to_part_origin_z = pick_tcp_z - tz
        insertion_depth = INSERTION_DEPTH_M
        place_gap = self._place_surface_gap - insertion_depth
        place_part_origin_z = board_top_z + (target_height * 0.5) + place_gap
        place_tcp_z = place_part_origin_z + grasp_tcp_to_part_origin_z
        place_z = place_tcp_z - ee_tcp_offset_z
        self.get_logger().info(
            f"Place gap for {target_part_name}: base={self._place_surface_gap:.4f} "
            f"insertion_depth={insertion_depth:.4f} "
            f"effective={place_gap:.4f}"
        )
        if not self._cartesian_move(
            self._make_pose(bx, by, place_z, ori),
            f"Descend to place (EE z={place_z:.3f}, TCP z={place_tcp_z:.3f}, board_top={board_top_z:.3f})"
        ):
            return

        # Step 8: Release at place pose, then always retreat up safely.
        # Detach first so opening gripper does not sweep an attached part sideways.
        time.sleep(RELEASE_PREOPEN_SETTLE_SEC)
        detached = self._detach_part(target_model)
        if not detached:
            self.get_logger().warn("Detach failed before opening; opening and retrying detach")

        # Open gripper after detach (or to help second detach attempt).
        self._gripper_command(
            GRIPPER_OPEN,
            "OPEN — releasing",
            move_time_s=GRIPPER_MOVE_TIME_SEC,
            wait_s=GRIPPER_SETTLE_SEC,
        )
        time.sleep(RELEASE_POSTOPEN_SETTLE_SEC)

        # Retry detach once after opening in case first detach failed.
        if not detached:
            time.sleep(0.15)
            detached = self._detach_part(target_model)
            if not detached:
                self.get_logger().warn(
                    "Detach still failed after opening gripper; continuing retreat/home anyway"
                )
        time.sleep(RELEASE_POSTDETACH_SETTLE_SEC)

        # Snap part into exact slot position (Gazebo ODE can't do peg-in-hole)
        if detached:
            self._snap_part_to_slot(target_model, slot_x, slot_y, target_height)

        # Step 9: Lift once back to travel height.
        lift_ok = self._cartesian_move(
            self._make_pose(bx, by, travel_z, ori),
            "Lift after place"
        )
        if not lift_ok:
            self.get_logger().warn("Lift-after-place failed with collisions; trying no-collision retreat")
            lift_ok = self._cartesian_move(
                self._make_pose(bx, by, travel_z, ori),
                "Lift after place (no-collision)",
                avoid_collisions=False,
                min_fraction=0.70,
                allow_partial=True,
            )
        if not lift_ok:
            self.get_logger().error("Retreat lift failed; aborting before return-home")
            return

        # One more detach attempt after clearing the slot.
        if not detached:
            detached = self._detach_part(target_model)
            if not detached:
                self.get_logger().warn("Part may still be attached after retreat")

        # Step 10: Return to starting pose ("home") for repeatable next cycle.
        if self._return_home:
            self.get_logger().info("Returning to start pose...")
            if not self._cartesian_move(start_pose, "Return home"):
                self.get_logger().warn("Return home failed; pick-place still completed")

        self.get_logger().info("=" * 50)
        if detached:
            self.get_logger().info(
                f"SUCCESS — picked {target['part_name']} and placed on assembly board!"
            )
        else:
            self.get_logger().warn(
                f"PARTIAL — placed {target['part_name']} pose reached, but detach did not confirm"
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--part",
        default="",
        help="Part abbreviation to pick (e.g. LG, LRP, LCP). "
        "Default: first detected part.",
    )
    parser.add_argument(
        "--place-gap",
        type=float,
        default=PLACE_SURFACE_GAP,
        help=(
            "Final bottom-surface clearance above assembly board in meters. "
            "Negative lowers placement; positive raises it."
        ),
    )
    parser.add_argument(
        "--no-return-home",
        action="store_true",
        help="Skip final return to initial pose after placement.",
    )
    args = parser.parse_args()

    rclpy.init()
    node = UR5ePickPlace(
        target_part=args.part,
        place_surface_gap=args.place_gap,
        return_home=not args.no_return_home,
    )

    executor = MultiThreadedExecutor()
    executor.add_node(node)

    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    node.wait_for_services()
    node.run()

    rclpy.shutdown()
    spin_thread.join(timeout=2.0)


if __name__ == "__main__":
    main()
