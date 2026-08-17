"""Write throttled RealSense color/depth previews without running inference."""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
from contextlib import suppress
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .assembly_board_v1_aruco import (
    DEFAULT_MARKER_LENGTH_M,
    ArucoLocalizationError,
    AssemblyBoardV1ArucoLocalizer,
    CalibrationProvenance,
    CameraCalibration,
    camera_calibration_from_info,
    load_calibration_provenance,
    matrix_from_transform_message,
    ros_stamp_to_epoch_seconds,
)


def _atomic_json_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_jpeg_write(path: Path, image: np.ndarray, quality: int = 82) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.jpg")
    ok = cv2.imwrite(str(temporary), image, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise RuntimeError(f"failed to encode preview frame: {path}")
    os.replace(temporary, path)


def depth_preview(depth: np.ndarray) -> np.ndarray:
    """Convert a RealSense depth image into an operator-friendly color map."""
    values = np.asarray(depth, dtype=np.float32)
    valid = values[np.isfinite(values) & (values > 0)]
    if valid.size == 0:
        return np.zeros((*values.shape[:2], 3), dtype=np.uint8)
    low, high = np.percentile(valid, [2.0, 98.0])
    if high <= low:
        high = low + 1.0
    scaled = np.clip((values - low) / (high - low), 0.0, 1.0)
    scaled[~np.isfinite(values) | (values <= 0)] = 0.0
    return cv2.applyColorMap((scaled * 255.0).astype(np.uint8), cv2.COLORMAP_TURBO)


def charuco_overlay(image: np.ndarray) -> tuple[np.ndarray, int]:
    """Draw visible DICT_4X4_50 markers without invoking Roboflow."""
    if not hasattr(cv2, "aruco"):
        return image, 0
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    corners, identifiers, _rejected = cv2.aruco.detectMarkers(image, dictionary)
    count = 0 if identifiers is None else len(identifiers)
    if count:
        cv2.aruco.drawDetectedMarkers(image, corners, identifiers)
        cv2.putText(
            image,
            f"ChArUco markers: {count}",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
    return image, count


class RealSensePreviewNode:
    """Subscribe to one camera and publish local preview artifacts for NiceGUI."""

    def __init__(  # noqa: PLR0913 - camera topics, frames, and calibration stay explicit.
        self,
        *,
        camera_role: str,
        color_topic: str,
        depth_topic: str,
        output_root: Path,
        camera_info_topic: str = "/camera/camera/color/camera_info",
        world_frame: str = "world",
        parent_frame: str = "tool0",
        camera_optical_frame: str = "camera_color_optical_frame",
        hand_eye_config: str | Path | None = None,
        marker_length_m: float = DEFAULT_MARKER_LENGTH_M,
        maximum_rate_hz: float = 5.0,
        expected_rate_hz: float = 6.0,
    ) -> None:
        import rclpy
        import tf2_ros
        from cv_bridge import CvBridge
        from rclpy.node import Node
        from sensor_msgs.msg import CameraInfo, Image

        self.rclpy = rclpy
        self.node = Node(f"{camera_role}_realsense_preview")
        self.camera_role = camera_role
        self.output_dir = output_root / camera_role
        self.world_frame = world_frame
        self.parent_frame = parent_frame
        self.camera_optical_frame = camera_optical_frame
        self.bridge = CvBridge()
        self.minimum_period = 1.0 / max(0.1, float(maximum_rate_hz))
        self.expected_rate_hz = max(0.1, float(expected_rate_hz))
        self._lock = threading.Lock()
        self._last_color_write = 0.0
        self._last_depth_write = 0.0
        self._color_count = 0
        self._depth_count = 0
        self._started_at = time.time()
        self._last_frame_at = 0.0
        self._last_error = "waiting for camera frames"
        self._charuco_marker_count = 0
        self._camera_calibration: CameraCalibration | None = None
        self._aruco_calibration: CalibrationProvenance | None = None
        self._aruco_calibration_mtime_ns = -1
        self._aruco_valid = False
        self._aruco_last_error = "waiting for CameraInfo and camera frames"
        self._tf2_ros = tf2_ros
        self._tf_buffer = tf2_ros.Buffer(cache_time=rclpy.duration.Duration(seconds=30.0))
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self.node)
        default_hand_eye = Path(
            f"~/.config/cais-spade-llm/{camera_role}_realsense_hand_eye.yaml"
        ).expanduser()
        self._hand_eye_config = Path(hand_eye_config or default_hand_eye).expanduser()
        self._aruco_localizer = (
            AssemblyBoardV1ArucoLocalizer(
                camera_role=camera_role,
                world_frame=world_frame,
                parent_frame=parent_frame,
                marker_length_m=marker_length_m,
                output_path=self.output_dir / "assembly_board-v1_aruco.json",
            )
            if camera_role in ("ur5e", "xarm6")
            else None
        )
        self.node.create_subscription(Image, color_topic, self._on_color, 10)
        self.node.create_subscription(Image, depth_topic, self._on_depth, 10)
        self.node.create_subscription(CameraInfo, camera_info_topic, self._on_camera_info, 10)
        self.node.create_timer(1.0, self._write_status)
        if self._aruco_localizer is not None:
            self._aruco_localizer.invalidate(self._aruco_last_error)
        self._write_status()

    def _on_camera_info(self, message: Any) -> None:
        try:
            camera = camera_calibration_from_info(message)
            if camera.frame_id != self.camera_optical_frame:
                raise ArucoLocalizationError(
                    "RealSense CameraInfo frame does not match camera_optical_frame: "
                    f"{camera.frame_id} != {self.camera_optical_frame}"
                )
        except ArucoLocalizationError as exc:
            self._camera_calibration = None
            self._invalidate_aruco(str(exc))
            return
        self._camera_calibration = camera

    def _load_aruco_calibration(self) -> CalibrationProvenance:
        try:
            mtime_ns = self._hand_eye_config.stat().st_mtime_ns
        except OSError:
            mtime_ns = -1
        if self._aruco_calibration is None or mtime_ns != self._aruco_calibration_mtime_ns:
            self._aruco_calibration = None
            self._aruco_calibration_mtime_ns = mtime_ns
            calibration = load_calibration_provenance(
                self._hand_eye_config,
                camera_role=self.camera_role,
                parent_frame=self.parent_frame,
            )
            self._aruco_calibration = calibration
        return self._aruco_calibration

    def _world_to_parent(self, stamp: Any) -> np.ndarray:
        if self.world_frame == self.parent_frame:
            return np.eye(4, dtype=np.float64)
        try:
            transform = self._tf_buffer.lookup_transform(
                self.world_frame,
                self.parent_frame,
                self.rclpy.time.Time.from_msg(stamp),
                timeout=self.rclpy.duration.Duration(seconds=0.25),
            )
        except self._tf2_ros.TransformException as exc:
            raise ArucoLocalizationError(
                f"timestamped TF unavailable for {self.world_frame} <- {self.parent_frame}: {exc}"
            ) from exc
        return matrix_from_transform_message(transform)

    def _invalidate_aruco(
        self,
        error: str,
        *,
        frame_captured_at: float | None = None,
    ) -> None:
        self._aruco_valid = False
        self._aruco_last_error = str(error)
        if self._aruco_localizer is not None:
            self._aruco_localizer.invalidate(
                error,
                frame_captured_at=frame_captured_at,
                calibration=self._aruco_calibration,
                camera=self._camera_calibration,
            )

    def _localize_assembly_board_v1_aruco(
        self,
        image: np.ndarray,
        message: Any,
    ) -> np.ndarray | None:
        if self._aruco_localizer is None:
            return None
        try:
            captured_at = ros_stamp_to_epoch_seconds(message.header.stamp)
            camera = self._camera_calibration
            if camera is None:
                raise ArucoLocalizationError("waiting for RealSense CameraInfo")
            calibration = self._load_aruco_calibration()
            world_to_parent = self._world_to_parent(message.header.stamp)
            payload, corners = self._aruco_localizer.observe(
                image=image,
                frame_captured_at=captured_at,
                camera=camera,
                calibration=calibration,
                world_to_parent=world_to_parent,
            )
        except (ArucoLocalizationError, cv2.error, ValueError) as exc:
            stamp = getattr(getattr(message, "header", None), "stamp", None)
            raw_captured_at = None
            if stamp is not None:
                with suppress(AttributeError, TypeError, ValueError):
                    raw_captured_at = float(stamp.sec) + float(stamp.nanosec) / 1e9
            self._invalidate_aruco(str(exc), frame_captured_at=raw_captured_at)
            return None
        self._aruco_valid = bool(payload["valid"])
        self._aruco_last_error = str(payload["last_error"])
        return corners

    def _on_color(self, message: Any) -> None:
        now = time.time()
        with self._lock:
            self._color_count += 1
            self._last_frame_at = now
            if now - self._last_color_write < self.minimum_period:
                return
            self._last_color_write = now
        try:
            image = self.bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
            aruco_corners = self._localize_assembly_board_v1_aruco(image, message)
            image, self._charuco_marker_count = charuco_overlay(image)
            if aruco_corners is not None:
                cv2.polylines(
                    image,
                    [np.rint(aruco_corners).astype(np.int32)],
                    True,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    image,
                    "assembly_board-v1 ArUco ID 70",
                    (12, 56),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.62,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
            _atomic_jpeg_write(self.output_dir / "color.jpg", image)
            self._last_error = ""
        except (RuntimeError, ValueError, cv2.error) as exc:
            self._last_error = f"color preview failed: {exc}"

    def _on_depth(self, message: Any) -> None:
        now = time.time()
        with self._lock:
            self._depth_count += 1
            self._last_frame_at = now
            if now - self._last_depth_write < self.minimum_period:
                return
            self._last_depth_write = now
        try:
            depth = self.bridge.imgmsg_to_cv2(message, desired_encoding="passthrough")
            _atomic_jpeg_write(self.output_dir / "depth.jpg", depth_preview(depth))
            self._last_error = ""
        except (RuntimeError, ValueError, cv2.error) as exc:
            self._last_error = f"depth preview failed: {exc}"

    def _write_status(self) -> None:
        now = time.time()
        elapsed = max(0.001, now - self._started_at)
        _atomic_json_write(
            self.output_dir / "status.json",
            {
                "camera_role": self.camera_role,
                "updated_at": now,
                "frame_captured_at": self._last_frame_at or None,
                "color_frame_count": self._color_count,
                "depth_frame_count": self._depth_count,
                "color_average_hz": self._color_count / elapsed,
                "depth_average_hz": self._depth_count / elapsed,
                "color_dropped_estimate": max(
                    0,
                    int(elapsed * self.expected_rate_hz) - self._color_count,
                ),
                "depth_dropped_estimate": max(
                    0,
                    int(elapsed * self.expected_rate_hz) - self._depth_count,
                ),
                "charuco_marker_count": self._charuco_marker_count,
                "charuco_visible": self._charuco_marker_count >= 4,
                "assembly_board-v1_aruco_valid": self._aruco_valid,
                "assembly_board-v1_aruco_error": self._aruco_last_error,
                "last_error": self._last_error,
            },
        )

    def destroy(self) -> None:
        """Write final status and release the ROS node."""
        self._write_status()
        self.node.destroy_node()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera-role", choices=("ur5e", "xarm6", "stationary"), required=True)
    parser.add_argument("--color-topic", required=True)
    parser.add_argument("--depth-topic", required=True)
    parser.add_argument("--camera-info-topic", required=True)
    parser.add_argument("--world-frame", default="world")
    parser.add_argument("--parent-frame", required=True)
    parser.add_argument("--camera-optical-frame", required=True)
    parser.add_argument("--hand-eye-config", type=Path)
    parser.add_argument("--marker-length-m", type=float, default=DEFAULT_MARKER_LENGTH_M)
    parser.add_argument("--output-root", type=Path, default=Path("/tmp/cais_perception_previews"))
    parser.add_argument("--maximum-rate-hz", type=float, default=5.0)
    parser.add_argument("--expected-rate-hz", type=float, default=6.0)
    return parser


def main() -> None:
    """Run the preview node until interrupted."""
    import rclpy
    from rclpy._rclpy_pybind11 import RCLError
    from rclpy.executors import MultiThreadedExecutor

    args, ros_args = _build_parser().parse_known_args()
    rclpy.init(args=ros_args)
    preview: RealSensePreviewNode | None = None
    executor: MultiThreadedExecutor | None = None
    try:
        preview = RealSensePreviewNode(
            camera_role=args.camera_role,
            color_topic=args.color_topic,
            depth_topic=args.depth_topic,
            output_root=args.output_root,
            camera_info_topic=args.camera_info_topic,
            world_frame=args.world_frame,
            parent_frame=args.parent_frame,
            camera_optical_frame=args.camera_optical_frame,
            hand_eye_config=args.hand_eye_config,
            marker_length_m=args.marker_length_m,
            maximum_rate_hz=args.maximum_rate_hz,
            expected_rate_hz=args.expected_rate_hz,
        )
        executor = MultiThreadedExecutor(num_threads=2)
        executor.add_node(preview.node)
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        if executor is not None:
            try:
                executor.shutdown()
            except RCLError:
                pass
        if preview is not None:
            preview.destroy()
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except RCLError:
            pass


if __name__ == "__main__":
    main()
