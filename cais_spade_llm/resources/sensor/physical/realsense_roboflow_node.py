"""ROS2 RealSense and Roboflow gear-pose service node."""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv

from .realsense_pose_estimator import (
    CalibrationError,
    ColorIntrinsics,
    DepthQualityError,
    RigidTransform,
    deproject_pixel,
    gear_center_world_point,
    load_hand_eye_calibration,
    pose_motion,
    robust_surface_depth,
)
from .roboflow_detector import (
    UNSUPPORTED_PARTS,
    DuplicateDetectionError,
    RoboflowConfigurationError,
    RoboflowGearDetector,
    RoboflowResponseError,
    RoboflowSettings,
)

logger = logging.getLogger(__name__)

DEFAULT_SNAPSHOT_PATH = Path("/tmp/cais_physical_perception.json")


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Replace a JSON snapshot without exposing partial content to Gazebo."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def _transform_from_message(transform: Any) -> RigidTransform:
    translation = transform.transform.translation
    rotation = transform.transform.rotation
    return RigidTransform(
        translation=(float(translation.x), float(translation.y), float(translation.z)),
        quaternion=(float(rotation.x), float(rotation.y), float(rotation.z), float(rotation.w)),
    )


class RealSenseRoboflowNode:
    """Own synchronized frames, validated detections, services, and the twin snapshot."""

    def __init__(self) -> None:
        try:
            import message_filters
            import rclpy
            import tf2_ros
            from cv_bridge import CvBridge
            from geometry_msgs.msg import TransformStamped
            from rclpy.callback_groups import ReentrantCallbackGroup
            from rclpy.node import Node
            from sensor_msgs.msg import CameraInfo, Image
            from std_srvs.srv import Trigger
        except ImportError as exc:
            raise RuntimeError(
                "ROS2 perception imports are unavailable; source ROS2 and install RealSense dependencies"
            ) from exc

        self._rclpy = rclpy
        self._tf2_ros = tf2_ros
        self._Trigger = Trigger
        self._TransformStamped = TransformStamped
        self.node: Any = Node("realsense_roboflow_perception")
        self._callback_group = ReentrantCallbackGroup()
        self._bridge = CvBridge()
        self._lock = threading.RLock()
        self._latest_frame: tuple[Any, np.ndarray, np.ndarray, ColorIntrinsics] | None = None
        self._last_rows: list[dict[str, Any]] = []
        self._last_error = "waiting for synchronized RealSense frames"
        self._roboflow_model_validated = False
        self._inference_lock = threading.Lock()

        self._declare_parameters()
        self.world_frame = str(self.node.get_parameter("world_frame").value)
        self.tool_frame = str(self.node.get_parameter("tool_frame").value)
        self.camera_link_frame = str(self.node.get_parameter("camera_link_frame").value)
        self.camera_optical_frame = str(self.node.get_parameter("camera_optical_frame").value)
        self.minimum_confidence = float(self.node.get_parameter("minimum_confidence").value)
        self.background_rate_hz = float(self.node.get_parameter("background_rate_hz").value)
        self.snapshot_path = Path(str(self.node.get_parameter("snapshot_path").value)).expanduser()
        calibration_path = str(self.node.get_parameter("hand_eye_config").value)

        self._calibration = load_hand_eye_calibration(calibration_path)
        self._detector = RoboflowGearDetector(
            RoboflowSettings.from_environment(),
            minimum_confidence=self.minimum_confidence,
        )

        self._tf_buffer = tf2_ros.Buffer(cache_time=rclpy.duration.Duration(seconds=30.0))
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self.node)
        self._static_broadcaster = tf2_ros.StaticTransformBroadcaster(self.node)
        self._publish_tool_to_camera_link()

        color_sub = message_filters.Subscriber(
            self.node,
            Image,
            str(self.node.get_parameter("color_topic").value),
        )
        depth_sub = message_filters.Subscriber(
            self.node,
            Image,
            str(self.node.get_parameter("aligned_depth_topic").value),
        )
        info_sub = message_filters.Subscriber(
            self.node,
            CameraInfo,
            str(self.node.get_parameter("camera_info_topic").value),
        )
        self._synchronizer = message_filters.ApproximateTimeSynchronizer(
            [color_sub, depth_sub, info_sub],
            queue_size=10,
            slop=0.08,
        )
        self._synchronizer.registerCallback(self._on_synchronized_frame)

        self.node.create_service(
            Trigger,
            "/detect_all",
            self._detect_all_service,
            callback_group=self._callback_group,
        )
        self.node.create_service(
            Trigger,
            "/detect_part",
            self._detect_part_service,
            callback_group=self._callback_group,
        )
        period = 1.0 / max(0.01, self.background_rate_hz)
        self.node.create_timer(period, self._background_detection, callback_group=self._callback_group)
        self._write_snapshot([])

    def _declare_parameters(self) -> None:
        self.node.declare_parameter("color_topic", "/camera/camera/color/image_raw")
        self.node.declare_parameter(
            "aligned_depth_topic",
            "/camera/camera/aligned_depth_to_color/image_raw",
        )
        self.node.declare_parameter("camera_info_topic", "/camera/camera/color/camera_info")
        self.node.declare_parameter("world_frame", "world")
        self.node.declare_parameter("tool_frame", "tool0")
        self.node.declare_parameter("camera_link_frame", "camera_link")
        self.node.declare_parameter("camera_optical_frame", "camera_color_optical_frame")
        self.node.declare_parameter("minimum_confidence", 0.70)
        self.node.declare_parameter("minimum_depth_samples", 25)
        self.node.declare_parameter("maximum_depth_mad_m", 0.003)
        self.node.declare_parameter("maximum_frame_age_sec", 1.0)
        self.node.declare_parameter("background_rate_hz", 0.2)
        self.node.declare_parameter("snapshot_path", str(DEFAULT_SNAPSHOT_PATH))
        self.node.declare_parameter(
            "hand_eye_config",
            os.path.expanduser(
                os.environ.get(
                    "REALSENSE_HAND_EYE_CONFIG",
                    "~/.config/cais-spade-llm/ur5e_realsense_hand_eye.yaml",
                )
            ),
        )
        self.node.declare_parameter("target_part", "")

    def _publish_tool_to_camera_link(self) -> None:
        transform = self._calibration.get("tool0_to_camera_link")
        if not isinstance(transform, dict):
            raise CalibrationError("calibration is missing tool0_to_camera_link")
        translation = transform.get("translation", {})
        quaternion = transform.get("quaternion", {})
        message = self._TransformStamped()
        message.header.stamp = self.node.get_clock().now().to_msg()
        message.header.frame_id = self.tool_frame
        message.child_frame_id = self.camera_link_frame
        message.transform.translation.x = float(translation["x"])
        message.transform.translation.y = float(translation["y"])
        message.transform.translation.z = float(translation["z"])
        message.transform.rotation.x = float(quaternion["x"])
        message.transform.rotation.y = float(quaternion["y"])
        message.transform.rotation.z = float(quaternion["z"])
        message.transform.rotation.w = float(quaternion["w"])
        self._static_broadcaster.sendTransform(message)

    def _on_synchronized_frame(self, color_msg: Any, depth_msg: Any, info_msg: Any) -> None:
        color = self._bridge.imgmsg_to_cv2(color_msg, desired_encoding="bgr8")
        depth = self._bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
        depth_array = np.asarray(depth)
        if depth_array.dtype == np.uint16:
            depth_m = depth_array.astype(np.float32) * 0.001
        else:
            depth_m = depth_array.astype(np.float32)
        intrinsics = ColorIntrinsics(
            fx=float(info_msg.k[0]),
            fy=float(info_msg.k[4]),
            cx=float(info_msg.k[2]),
            cy=float(info_msg.k[5]),
        )
        with self._lock:
            self._latest_frame = (color_msg.header.stamp, color.copy(), depth_m.copy(), intrinsics)

    def _lookup_transform(self, target: str, source: str, stamp: Any | None = None) -> RigidTransform:
        lookup_time = self._rclpy.time.Time() if stamp is None else self._rclpy.time.Time.from_msg(stamp)
        try:
            transform = self._tf_buffer.lookup_transform(
                target,
                source,
                lookup_time,
                timeout=self._rclpy.duration.Duration(seconds=1.0),
            )
        except self._tf2_ros.TransformException as exc:
            raise CalibrationError(
                f"TF unavailable for {target} <- {source}: {exc}"
            ) from exc
        return _transform_from_message(transform)

    def _frame_copy(self) -> tuple[Any, np.ndarray, np.ndarray, ColorIntrinsics]:
        with self._lock:
            frame = self._latest_frame
            if frame is None:
                raise RuntimeError("no synchronized RealSense color/depth frame")
            stamp, color, depth_m, intrinsics = frame
            age = self.node.get_clock().now().nanoseconds / 1e9 - (
                float(stamp.sec) + float(stamp.nanosec) / 1e9
            )
            maximum_age = float(self.node.get_parameter("maximum_frame_age_sec").value)
            if age > maximum_age:
                raise RuntimeError(f"RealSense frame is stale ({age:.2f} s)")
            return stamp, color.copy(), depth_m.copy(), intrinsics

    def _run_detection(self) -> list[dict[str, Any]]:
        if not self._inference_lock.acquire(blocking=False):
            raise RuntimeError("Roboflow inference is already running")
        try:
            stamp, color, depth_m, intrinsics = self._frame_copy()
            stationary_start = self._lookup_transform(self.world_frame, self.tool_frame)
            time.sleep(0.15)
            tool_before = self._lookup_transform(self.world_frame, self.tool_frame)
            pre_translation_m, pre_rotation_deg = pose_motion(stationary_start, tool_before)
            if pre_translation_m > 0.001 or pre_rotation_deg > 0.5:
                raise RuntimeError("UR5e is moving; inference was not started")
            detections = self._detector.detect(color)
            self._roboflow_model_validated = True
            camera_to_world = self._lookup_transform(
                self.world_frame,
                self.camera_optical_frame,
                stamp,
            )
            tool_after = self._lookup_transform(self.world_frame, self.tool_frame)
            translation_m, rotation_deg = pose_motion(tool_before, tool_after)
            if translation_m > 0.001 or rotation_deg > 0.5:
                raise RuntimeError(
                    "UR5e moved during inference "
                    f"({translation_m * 1000.0:.2f} mm, {rotation_deg:.2f} deg)"
                )

            captured_at = float(stamp.sec) + float(stamp.nanosec) / 1e9
            rows: list[dict[str, Any]] = []
            for detection in detections:
                estimate = robust_surface_depth(
                    depth_m,
                    detection,
                    minimum_samples=int(
                        self.node.get_parameter("minimum_depth_samples").value
                    ),
                    maximum_mad_m=float(
                        self.node.get_parameter("maximum_depth_mad_m").value
                    ),
                )
                camera_point = deproject_pixel(
                    detection.center_x,
                    detection.center_y,
                    estimate.depth_m,
                    intrinsics,
                )
                world_point = gear_center_world_point(camera_point, camera_to_world)
                rows.append(
                    {
                        "part_name": detection.part_name,
                        "model_name": detection.model_name,
                        "x": float(world_point[0]),
                        "y": float(world_point[1]),
                        "z": float(world_point[2]),
                        "qx": 0.0,
                        "qy": 0.0,
                        "qz": 0.0,
                        "qw": 1.0,
                        "frame_id": self.world_frame,
                        "confidence": detection.confidence,
                        "captured_at": captured_at,
                        "source": "realsense_roboflow",
                        "model_id": self._detector.settings.model_id,
                        "depth_sample_count": estimate.sample_count,
                        "depth_mad_m": estimate.mad_m,
                    }
                )
            self._last_rows = rows
            self._last_error = ""
            self._write_snapshot(rows)
            return rows
        finally:
            self._inference_lock.release()

    def _write_snapshot(self, rows: list[dict[str, Any]]) -> None:
        with self._lock:
            frame = self._latest_frame
        frame_time = None
        if frame is not None:
            stamp = frame[0]
            frame_time = float(stamp.sec) + float(stamp.nanosec) / 1e9
        validation = self._calibration.get("validation", {})
        _atomic_write_json(
            self.snapshot_path,
            {
                "updated_at": time.time(),
                "frame_captured_at": frame_time,
                "detections": rows,
                "last_error": self._last_error,
                "realsense_connected": frame is not None,
                "roboflow_model_configured": True,
                "roboflow_ready": self._roboflow_model_validated,
                "model_id": self._detector.settings.model_id,
                "calibration": {
                    "identity": str(self._calibration.get("calibration_id", "unknown")),
                    "median_reprojection_error_px": validation.get(
                        "median_reprojection_error_px"
                    ),
                    "fixed_board_translation_rms_m": validation.get(
                        "fixed_board_translation_rms_m"
                    ),
                    "fixed_board_rotation_rms_deg": validation.get(
                        "fixed_board_rotation_rms_deg"
                    ),
                },
                "unsupported": UNSUPPORTED_PARTS,
            },
        )

    def _service_result(self, response: Any, *, target_part: str = "") -> Any:
        target = str(target_part or "").strip().upper()
        if target in UNSUPPORTED_PARTS:
            response.success = False
            response.message = UNSUPPORTED_PARTS[target]
            return response
        try:
            rows = self._run_detection()
            if target:
                rows = [row for row in rows if row.get("part_name") == target]
                if not rows:
                    response.success = False
                    response.message = f"{target} was not detected"
                    return response
                response.message = json.dumps({"detected": True, **rows[0]})
            else:
                response.message = json.dumps(rows)
            response.success = True
        except (
            CalibrationError,
            DepthQualityError,
            DuplicateDetectionError,
            RoboflowConfigurationError,
            RoboflowResponseError,
            RuntimeError,
            ValueError,
            KeyError,
            OSError,
        ) as exc:
            self._last_error = str(exc)
            self._write_snapshot(self._last_rows)
            response.success = False
            response.message = str(exc)
        return response

    def _detect_all_service(self, _request: Any, response: Any) -> Any:
        return self._service_result(response)

    def _detect_part_service(self, _request: Any, response: Any) -> Any:
        target = str(self.node.get_parameter("target_part").value)
        return self._service_result(response, target_part=target)

    def _background_detection(self) -> None:
        if self._latest_frame is None or self._inference_lock.locked():
            return
        try:
            self._run_detection()
        except (
            CalibrationError,
            DepthQualityError,
            DuplicateDetectionError,
            RoboflowConfigurationError,
            RoboflowResponseError,
            RuntimeError,
            ValueError,
            KeyError,
            OSError,
        ) as exc:
            self._last_error = str(exc)
            self._write_snapshot(self._last_rows)
            self.node.get_logger().warning(f"Background gear detection rejected: {exc}")


def main() -> None:
    """Run the physical perception node in a multithreaded ROS2 executor."""
    load_dotenv()
    logging.basicConfig(level=logging.INFO)
    try:
        import rclpy
        from rclpy.executors import MultiThreadedExecutor
    except ImportError as exc:
        raise SystemExit("ROS2 Python is unavailable; source /opt/ros/humble/setup.bash") from exc

    rclpy.init()
    wrapper: RealSenseRoboflowNode | None = None
    try:
        wrapper = RealSenseRoboflowNode()
        executor = MultiThreadedExecutor(num_threads=4)
        executor.add_node(wrapper.node)
        executor.spin()
    finally:
        if wrapper is not None:
            wrapper.node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
