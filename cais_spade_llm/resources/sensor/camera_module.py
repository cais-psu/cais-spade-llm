from __future__ import annotations

import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


class CameraModule:
    """
    Sensor module for observing part positions in the workspace.

    Backends:
      - none: no observation (dry-run placeholder path).
      - mock: deterministic map from mock_observations.
      - gazebo_gt: ROS2 /detect_part and /detect_all from Gazebo perception node.
      - yolo: ROS2 /detect_part and /detect_all from the RealSense node.
    """

    def __init__(
        self,
        mock_observations: dict[str, dict[str, Any]] | None = None,
        use_ros2: bool = False,
        backend: str | None = None,
        ros2_timeout_sec: float = 5.0,
        ros2_call_retries: int = 2,
    ) -> None:
        self._backend = self._resolve_backend(
            backend=backend,
            use_ros2=use_ros2,
            mock_observations=mock_observations,
        )
        self._mock_observations: dict[str, dict[str, Any] | None] = mock_observations or {}
        self._use_ros2 = self._backend in {"gazebo_gt", "yolo"}
        self._ros2_timeout = ros2_timeout_sec
        self._ros2_call_retries = max(1, int(ros2_call_retries))
        default_node_name = (
            "/realsense_roboflow_perception"
            if self._backend == "yolo"
            else "/perception_node"
        )
        self._perception_node_name = (
            str(os.environ.get("PERCEPTION_NODE_NAME", default_node_name)).strip()
            or default_node_name
        )

        self._ros2_node = None
        self._Trigger = None

        self._legacy_detect_part_client = None
        self._legacy_detect_all_client = None
        self._ros2_init_attempted = False

        logger.info("CameraModule initialized backend=%s", self._backend)

    @staticmethod
    def _resolve_backend(
        *,
        backend: str | None,
        use_ros2: bool,
        mock_observations: dict[str, dict[str, Any]] | None,
    ) -> str:
        aliases = {
            "ros2": "gazebo_gt",
            "gazebo": "gazebo_gt",
            "gazebo_gt": "gazebo_gt",
            "none": "none",
            "mock": "mock",
            "yolo": "yolo",
        }
        explicit = aliases.get(str(backend or "").strip().lower())
        if explicit:
            return explicit

        env_value = aliases.get(str(os.environ.get("PERCEPTION_BACKEND", "")).strip().lower())
        if env_value:
            return env_value

        if use_ros2:
            return "gazebo_gt"
        if mock_observations:
            return "mock"
        return "none"

    def _init_ros2(self) -> None:
        """Initialize ROS2 clients for Trigger-based legacy API."""
        if self._ros2_init_attempted:
            return
        self._ros2_init_attempted = True
        try:
            import rclpy
            from std_srvs.srv import Trigger

            self._Trigger = Trigger

            if not rclpy.ok():
                rclpy.init()

            self._ros2_node = rclpy.create_node("camera_module_client")

            # Keep legacy clients for compatibility with older perception nodes.
            self._legacy_detect_part_client = self._ros2_node.create_client(Trigger, "/detect_part")
            self._legacy_detect_all_client = self._ros2_node.create_client(Trigger, "/detect_all")

            logger.info(
                "CameraModule ROS2 clients ready backend=%s node=%s",
                self._backend,
                self._perception_node_name,
            )
        except ImportError:
            logger.warning("CameraModule: rclpy unavailable; disabling ROS2 camera path")
            self._use_ros2 = False
        except Exception as exc:
            logger.warning("CameraModule: ROS2 init failed (%s); disabling ROS2 camera path", exc)
            self._use_ros2 = False

    def _ensure_ros2(self) -> bool:
        """Lazy-initialize ROS2 clients only when observation is actually requested."""
        if not self._use_ros2:
            return False
        if self._ros2_node is None:
            self._init_ros2()
        return bool(
            self._ros2_node and self._legacy_detect_part_client and self._legacy_detect_all_client
        )

    def observe(self, part_name: str) -> dict[str, Any] | None:
        if self._backend == "mock":
            return self._mock_observations.get(part_name)
        if self._backend == "none":
            return None
        if self._ensure_ros2():
            return self._observe_ros2_trigger(part_name)
        return None

    def _observe_ros2_trigger(self, part_name: str) -> dict[str, Any] | None:
        if not (self._legacy_detect_part_client and self._Trigger):
            return None

        import rclpy

        try:
            from rcl_interfaces.msg import Parameter as ParameterMsg
            from rcl_interfaces.msg import ParameterType, ParameterValue
            from rcl_interfaces.srv import SetParameters

            # Legacy protocol: set target_part parameter then call Trigger.
            param_client = self._ros2_node.create_client(
                SetParameters, f"{self._perception_node_name}/set_parameters"
            )
            if param_client.wait_for_service(timeout_sec=self._ros2_timeout):
                param_req = SetParameters.Request()
                param_msg = ParameterMsg()
                param_msg.name = "target_part"
                param_msg.value = ParameterValue()
                param_msg.value.type = ParameterType.PARAMETER_STRING
                param_msg.value.string_value = str(part_name or "").strip().upper()
                param_req.parameters = [param_msg]
                param_future = param_client.call_async(param_req)
                rclpy.spin_until_future_complete(
                    self._ros2_node, param_future, timeout_sec=self._ros2_timeout
                )

            if not self._legacy_detect_part_client.wait_for_service(timeout_sec=self._ros2_timeout):
                return None

            result = None
            for _ in range(self._ros2_call_retries):
                request = self._Trigger.Request()
                future = self._legacy_detect_part_client.call_async(request)
                rclpy.spin_until_future_complete(
                    self._ros2_node, future, timeout_sec=self._ros2_timeout
                )
                if future.done() and future.result() is not None:
                    result = future.result()
                    break

            if result is None:
                return None

            if not result.success:
                logger.warning("CameraModule: /detect_part rejected: %s", result.message)
                return None
            data = json.loads(result.message)
            if data.get("detected"):
                return dict(data)
            return None
        except Exception:
            logger.exception("CameraModule: /detect_part Trigger call failed")
            return None

    def observe_all(self) -> dict[str, dict[str, Any]]:
        if self._backend == "mock":
            return {k: v for k, v in self._mock_observations.items() if v is not None}
        if self._backend == "none":
            return {}
        if self._ensure_ros2():
            return self._observe_all_ros2_trigger()
        return {}

    def _observe_all_ros2_trigger(self) -> dict[str, dict[str, Any]]:
        if not (self._legacy_detect_all_client and self._Trigger):
            return {}

        import rclpy

        try:
            if not self._legacy_detect_all_client.wait_for_service(timeout_sec=self._ros2_timeout):
                return {}

            request = self._Trigger.Request()
            future = self._legacy_detect_all_client.call_async(request)
            rclpy.spin_until_future_complete(
                self._ros2_node, future, timeout_sec=self._ros2_timeout
            )
            result = future.result()
            if result is None:
                return {}
            if not result.success:
                logger.warning("CameraModule: /detect_all rejected: %s", result.message)
                return {}

            detections = json.loads(result.message)
            parsed: dict[str, dict[str, Any]] = {}
            for d in detections:
                parsed[str(d["part_name"])] = dict(d)
            return parsed
        except Exception:
            logger.exception("CameraModule: /detect_all Trigger call failed")
            return {}

    def destroy(self) -> None:
        if self._ros2_node:
            self._ros2_node.destroy_node()
            self._ros2_node = None


if __name__ == "__main__":
    import subprocess
    import sys

    if sys.version_info[:2] != (3, 10):
        print(
            f"Current Python is {sys.version_info.major}.{sys.version_info.minor}, "
            "re-launching with python3.10 (required for rclpy)..."
        )
        result = subprocess.run(["python3.10", __file__])
        sys.exit(result.returncode)

    logging.basicConfig(level=logging.INFO)
    selected_backend = str(os.environ.get("PERCEPTION_BACKEND", "gazebo_gt")).strip() or "gazebo_gt"
    cam = CameraModule(backend=selected_backend)
    if selected_backend == "gazebo_gt" and not cam._use_ros2:
        print("ERROR: ROS2 mode not available. Is perception node running?")
        sys.exit(1)

    print("\n--- observe('SG') ---")
    print(f"  SG: {cam.observe('SG')}")

    print("\n--- observe_all() ---")
    for name, pos in cam.observe_all().items():
        print(f"  {name}: x={pos['x']:.1f} y={pos['y']:.1f} z={pos['z']:.1f}")

    cam.destroy()
