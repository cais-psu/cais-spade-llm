"""Write throttled RealSense color/depth previews without running inference."""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np


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

    def __init__(
        self,
        *,
        camera_role: str,
        color_topic: str,
        depth_topic: str,
        output_root: Path,
        maximum_rate_hz: float = 5.0,
        expected_rate_hz: float = 6.0,
    ) -> None:
        import rclpy
        from cv_bridge import CvBridge
        from rclpy.node import Node
        from sensor_msgs.msg import Image

        self.rclpy = rclpy
        self.node = Node(f"{camera_role}_realsense_preview")
        self.camera_role = camera_role
        self.output_dir = output_root / camera_role
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
        self.node.create_subscription(Image, color_topic, self._on_color, 10)
        self.node.create_subscription(Image, depth_topic, self._on_depth, 10)
        self.node.create_timer(1.0, self._write_status)
        self._write_status()

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
            image, self._charuco_marker_count = charuco_overlay(image)
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
    parser.add_argument("--output-root", type=Path, default=Path("/tmp/cais_perception_previews"))
    parser.add_argument("--maximum-rate-hz", type=float, default=5.0)
    parser.add_argument("--expected-rate-hz", type=float, default=6.0)
    return parser


def main() -> None:
    """Run the preview node until interrupted."""
    import rclpy
    from rclpy._rclpy_pybind11 import RCLError

    args, ros_args = _build_parser().parse_known_args()
    rclpy.init(args=ros_args)
    preview = RealSensePreviewNode(
        camera_role=args.camera_role,
        color_topic=args.color_topic,
        depth_topic=args.depth_topic,
        output_root=args.output_root,
        maximum_rate_hz=args.maximum_rate_hz,
        expected_rate_hz=args.expected_rate_hz,
    )
    try:
        rclpy.spin(preview.node)
    except KeyboardInterrupt:
        pass
    finally:
        preview.destroy()
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except RCLError:
            pass


if __name__ == "__main__":
    main()
