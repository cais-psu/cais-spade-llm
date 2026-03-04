#!/usr/bin/env python3
"""
Part detection node for Gazebo simulation.

API:
  /detect_part  - std_srvs/srv/Trigger + target_part parameter
  /detect_all   - std_srvs/srv/Trigger + JSON payload in message
"""

from __future__ import annotations

import json
import time

import rclpy
from gazebo_msgs.srv import GetEntityState
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_srvs.srv import Trigger


PART_MAP = {
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


class PerceptionNode(Node):
    def __init__(self):
        super().__init__("perception_node")

        # Retained for legacy Trigger compatibility.
        self.declare_parameter("target_part", "")

        self._gazebo_cb_group = MutuallyExclusiveCallbackGroup()
        self._service_cb_group = MutuallyExclusiveCallbackGroup()

        self._get_state_client = self.create_client(
            GetEntityState,
            "/get_entity_state",
            callback_group=self._gazebo_cb_group,
        )

        self.create_service(
            Trigger,
            "/detect_part",
            self._detect_part_legacy_callback,
            callback_group=self._service_cb_group,
        )
        self.create_service(
            Trigger,
            "/detect_all",
            self._detect_all_legacy_callback,
            callback_group=self._service_cb_group,
        )

        self.get_logger().info(
            "PerceptionNode started (Gazebo ground truth mode, api=trigger_json)"
        )

        self.get_logger().info("Waiting for /get_entity_state service...")
        for attempt in range(6):
            if self._get_state_client.wait_for_service(timeout_sec=5.0):
                self.get_logger().info("Connected to /get_entity_state service")
                break
            self.get_logger().info(
                f"Attempt {attempt + 1}/6: /get_entity_state not available yet, retrying"
            )
        else:
            self.get_logger().warn(
                "/get_entity_state not available after 30s; is Gazebo running?"
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
            self.get_logger().warn(f"Model '{model_name}' not found in Gazebo")
            return None

        pos = result.state.pose.position
        return (pos.x, pos.y, pos.z)

    @staticmethod
    def _detection_dict(part_id: str, model_name: str, pose: tuple[float, float, float]) -> dict:
        x, y, z = pose
        return {
            "part_name": part_id,
            "model_name": model_name,
            "x": round(float(x), 4),
            "y": round(float(y), 4),
            "z": round(float(z), 4),
        }

    def _all_detections(self) -> list[dict]:
        detections: list[dict] = []
        for part_id, model_name in PART_MAP.items():
            pose = self._get_part_pose(model_name)
            if pose is not None:
                detections.append(self._detection_dict(part_id, model_name, pose))
        return detections

    # ------------------------------------------------------------------ #
    # Trigger API callbacks
    # ------------------------------------------------------------------ #
    def _detect_part_legacy_callback(self, request, response):  # noqa: ARG002
        target = self.get_parameter("target_part").get_parameter_value().string_value
        target = str(target or "").strip().upper()

        if target not in PART_MAP:
            response.success = False
            response.message = json.dumps(
                {
                    "part_name": target,
                    "detected": False,
                    "error": f"Unknown part '{target}'. Valid: {list(PART_MAP.keys())}",
                }
            )
            return response

        model_name = PART_MAP[target]
        pose = self._get_part_pose(model_name)

        if pose is not None:
            det = self._detection_dict(target, model_name, pose)
            response.success = True
            response.message = json.dumps(
                {
                    "part_name": det["part_name"],
                    "model_name": det["model_name"],
                    "x": det["x"],
                    "y": det["y"],
                    "z": det["z"],
                    "detected": True,
                }
            )
        else:
            response.success = False
            response.message = json.dumps({"part_name": target, "detected": False})

        return response

    def _detect_all_legacy_callback(self, request, response):  # noqa: ARG002
        detections = self._all_detections()
        response.success = True
        response.message = json.dumps(detections)
        return response


def main(args=None):
    rclpy.init(args=args)
    node = PerceptionNode()
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
