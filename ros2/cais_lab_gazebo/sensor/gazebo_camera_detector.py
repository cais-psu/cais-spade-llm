#!/usr/bin/env python3
"""
Part detection node for Gazebo simulation.

Uses Gazebo ground truth (model state service) for reliable part positions.
This is the standard research approach — develop pick-and-place with known
positions, then swap in real perception (YOLO, GPD, etc.) for hardware.

ROS2 Services:
  /detect_part   - Find a specific part by abbreviation (SG, MCP, etc.)
  /detect_all    - List all currently detected parts with positions

Usage:
  python3.10 ros2/cais_lab_gazebo/sensor/gazebo_camera_detector.py

  # Test:
  ros2 param set /perception_node target_part SG
  ros2 service call /detect_part std_srvs/srv/Trigger
  ros2 service call /detect_all std_srvs/srv/Trigger
"""

import json
import time

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from std_srvs.srv import Trigger
from gazebo_msgs.srv import GetEntityState

# ---------------------------------------------------------------------------
# Part abbreviation -> Gazebo model name mapping
# ---------------------------------------------------------------------------

PART_MAP = {
    "SG":  "gear_small",
    "MG":  "gear_medium",
    "LG":  "gear_large",
    "SRP": "rect_pin_small",
    "MRP": "rect_pin_medium",
    "LRP": "rect_pin_large",
    "SCP": "circ_pin_small",
    "MCP": "circ_pin_medium",
    "LCP": "circ_pin_large",
}


class PerceptionNode(Node):

    def __init__(self):
        super().__init__("perception_node")

        # Declare target_part parameter
        self.declare_parameter("target_part", "")

        # Separate callback groups so service callbacks can call Gazebo
        # without deadlocking (service callback and client response are
        # handled by different threads in MultiThreadedExecutor)
        self._gazebo_cb_group = MutuallyExclusiveCallbackGroup()
        self._service_cb_group = MutuallyExclusiveCallbackGroup()

        # Gazebo model state client (own callback group)
        self._get_state_client = self.create_client(
            GetEntityState, "/get_entity_state",
            callback_group=self._gazebo_cb_group,
        )

        # Services (own callback group)
        self.create_service(
            Trigger, "/detect_part", self._detect_part_callback,
            callback_group=self._service_cb_group,
        )
        self.create_service(
            Trigger, "/detect_all", self._detect_all_callback,
            callback_group=self._service_cb_group,
        )

        self.get_logger().info(
            f"PerceptionNode started (Gazebo ground truth mode) — "
            f"{len(PART_MAP)} parts registered"
        )

        # Wait for Gazebo service (retry — Gazebo may still be loading)
        self.get_logger().info("Waiting for /get_entity_state service...")
        for attempt in range(6):
            if self._get_state_client.wait_for_service(timeout_sec=5.0):
                self.get_logger().info("Connected to /get_entity_state service")
                break
            self.get_logger().info(
                f"Attempt {attempt + 1}/6 — /get_entity_state not yet available, retrying..."
            )
        else:
            self.get_logger().warn(
                "/get_entity_state not available after 30s — is Gazebo running?"
            )

    def _get_part_pose(self, model_name: str):
        """Query Gazebo for a model's world pose. Returns (x, y, z) or None."""
        if not self._get_state_client.service_is_ready():
            self.get_logger().warn("Gazebo service not ready")
            return None

        request = GetEntityState.Request()
        request.name = model_name

        future = self._get_state_client.call_async(request)
        deadline = time.monotonic() + 5.0
        # Avoid nested spin calls inside a service callback. The executor is
        # already spinning in another thread, so just wait for completion.
        while rclpy.ok() and not future.done() and time.monotonic() < deadline:
            time.sleep(0.01)

        if not future.done():
            self.get_logger().warn(f"Timed out waiting for state of '{model_name}'")
            return None

        if future.result() is None:
            self.get_logger().warn(f"Failed to get state for '{model_name}'")
            return None

        result = future.result()
        if not result.success:
            self.get_logger().warn(
                f"Model '{model_name}' not found in Gazebo"
            )
            return None

        pos = result.state.pose.position
        return (pos.x, pos.y, pos.z)

    def _detect_part_callback(self, request, response):
        """
        /detect_part service.
        Set target_part parameter first:
            ros2 param set /perception_node target_part SG
        """
        target = self.get_parameter("target_part").get_parameter_value().string_value

        if target not in PART_MAP:
            response.success = False
            response.message = json.dumps({
                "part_name": target,
                "detected": False,
                "error": f"Unknown part '{target}'. Valid: {list(PART_MAP.keys())}",
            })
            return response

        model_name = PART_MAP[target]
        pose = self._get_part_pose(model_name)

        if pose:
            x, y, z = pose
            response.success = True
            response.message = json.dumps({
                "part_name": target,
                "model_name": model_name,
                "x": round(x, 4),
                "y": round(y, 4),
                "z": round(z, 4),
                "detected": True,
            })
            self.get_logger().info(
                f"Detected {target} ({model_name}) at "
                f"({x:.4f}, {y:.4f}, {z:.4f})"
            )
        else:
            response.success = False
            response.message = json.dumps({
                "part_name": target,
                "detected": False,
            })

        return response

    def _detect_all_callback(self, request, response):
        """
        /detect_all service.
        Returns JSON list of all parts with their current positions.
        """
        detections = []

        for part_id, model_name in PART_MAP.items():
            pose = self._get_part_pose(model_name)
            if pose:
                x, y, z = pose
                detections.append({
                    "part_name": part_id,
                    "model_name": model_name,
                    "x": round(x, 4),
                    "y": round(y, 4),
                    "z": round(z, 4),
                })

        response.success = True
        response.message = json.dumps(detections)
        self.get_logger().info(
            f"detect_all: found {len(detections)}/{len(PART_MAP)} parts"
        )
        return response


def main(args=None):
    rclpy.init(args=args)
    node = PerceptionNode()
    # MultiThreadedExecutor allows the Gazebo client callback to complete
    # while a service callback is waiting for it (avoids deadlock)
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
