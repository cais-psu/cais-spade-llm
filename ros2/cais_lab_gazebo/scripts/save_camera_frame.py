#!/usr/bin/env python3
"""
Quick diagnostic: save one frame from each camera to disk as PNG.
Also prints HSV stats to help tune color detection ranges.

Usage:
    source /opt/ros/humble/setup.bash && source ~/ros2_ws/install/setup.bash
    python3 ros2/cais_lab_gazebo/scripts/save_camera_frame.py
"""

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image

CAMERAS = ["cam_mk3", "cam_mk4_1", "cam_mk4_2", "cam_assembly"]


class FrameSaver(Node):
    def __init__(self):
        super().__init__("frame_saver")
        self.bridge = CvBridge()
        self.saved = set()
        self.depth_saved = set()

        for cam in CAMERAS:
            self.create_subscription(
                Image, f"/{cam}/image_raw",
                lambda msg, c=cam: self._on_rgb(c, msg), 10
            )
            self.create_subscription(
                Image, f"/{cam}/depth/image_raw",
                lambda msg, c=cam: self._on_depth(c, msg), 10
            )

        self.get_logger().info("Waiting for camera frames...")

    def _on_rgb(self, cam_name, msg):
        if cam_name in self.saved:
            return
        self.saved.add(cam_name)

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"{cam_name} RGB convert failed: {e}")
            self.get_logger().info(f"  encoding={msg.encoding} size={msg.width}x{msg.height}")
            return

        path = f"/tmp/{cam_name}_rgb.png"
        cv2.imwrite(path, frame)
        self.get_logger().info(f"Saved {path} ({frame.shape})")

        # Analyze colors
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        self.get_logger().info(
            f"  {cam_name} HSV stats: "
            f"H=[{h.min()}-{h.max()}] S=[{s.min()}-{s.max()}] V=[{v.min()}-{v.max()}]"
        )

        # Check for specific colors
        for color, lower, upper in [
            ("green",  np.array([35, 80, 80]),  np.array([85, 255, 255])),
            ("yellow", np.array([20, 80, 80]),  np.array([35, 255, 255])),
            ("orange", np.array([5, 80, 80]),   np.array([20, 255, 255])),
        ]:
            mask = cv2.inRange(hsv, lower, upper)
            count = cv2.countNonZero(mask)
            self.get_logger().info(f"  {cam_name} {color} pixels: {count}")

        self._check_done()

    def _on_depth(self, cam_name, msg):
        if cam_name in self.depth_saved:
            return
        self.depth_saved.add(cam_name)

        try:
            depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="32FC1")
        except Exception as e:
            self.get_logger().error(f"{cam_name} depth convert failed: {e}")
            self.get_logger().info(f"  encoding={msg.encoding} size={msg.width}x{msg.height}")
            return

        valid = depth[np.isfinite(depth) & (depth > 0)]
        self.get_logger().info(
            f"  {cam_name} depth: min={valid.min():.3f}m max={valid.max():.3f}m "
            f"mean={valid.mean():.3f}m" if len(valid) > 0 else
            f"  {cam_name} depth: NO VALID PIXELS"
        )

        # Save depth as normalized grayscale for visual inspection
        depth_vis = np.where(np.isfinite(depth), depth, 0)
        depth_vis = (depth_vis / depth_vis.max() * 255).astype(np.uint8) if depth_vis.max() > 0 else depth_vis.astype(np.uint8)
        path = f"/tmp/{cam_name}_depth.png"
        cv2.imwrite(path, depth_vis)
        self.get_logger().info(f"Saved {path}")

        self._check_done()

    def _check_done(self):
        if len(self.saved) == len(CAMERAS) and len(self.depth_saved) == len(CAMERAS):
            self.get_logger().info("All frames saved to /tmp/cam_*.png — done!")
            self.get_logger().info("View with: eog /tmp/cam_mk3_rgb.png  (or any image viewer)")
            raise SystemExit(0)


def main():
    rclpy.init()
    node = FrameSaver()
    try:
        rclpy.spin(node)
    except SystemExit:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
