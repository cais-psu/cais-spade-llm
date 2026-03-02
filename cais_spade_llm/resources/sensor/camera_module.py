from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class CameraModule:
    """
    Sensor module for observing part positions in the workspace.

    ProductAgent uses this to verify placement after every place_insert ACK.
    The robot places the part and reports mechanical outcome; CameraModule
    provides the ground truth about where the part actually ended up.

    Observation outcomes:
      - Returns {"x": float, "y": float, "z": float} if part is detected.
      - Returns None if part cannot be located (untracked → human intervention).

    Modes:
      - mock: Pass mock_observations at construction time (default, for tests).
      - ros2: Connect to the gazebo_camera_detector's /detect_part service.
              Set use_ros2=True. Falls back to mock if ROS2 unavailable.

    Example (mock mode):
        CameraModule(mock_observations={
            "SG":  {"x": 750, "y": -200, "z": 50},
            "MCP": {"x": 400, "y": -100, "z": 50},
        })

    Example (ROS2 mode):
        CameraModule(use_ros2=True)

    Workspace boundaries (from robot configs, units: mm):
        xarm6: x=[-150, 650], y=[-550, 100], z=[0, 600]
        ur5e:  x=[100,  900], y=[-450, 200], z=[0, 600]
    """

    def __init__(
        self,
        mock_observations: Optional[Dict[str, Dict[str, float]]] = None,
        use_ros2: bool = False,
        ros2_timeout_sec: float = 5.0,
        ros2_call_retries: int = 2,
    ) -> None:
        self._mock_observations: Dict[str, Optional[Dict[str, float]]] = mock_observations or {}
        self._use_ros2 = use_ros2
        self._ros2_timeout = ros2_timeout_sec
        self._ros2_call_retries = max(1, int(ros2_call_retries))
        self._ros2_node = None
        self._detect_client = None
        self._detect_all_client = None

        if self._use_ros2:
            self._init_ros2()

    def _init_ros2(self) -> None:
        """Initialize ROS2 service clients for gazebo_camera_detector."""
        try:
            import rclpy
            from std_srvs.srv import Trigger

            if not rclpy.ok():
                rclpy.init()

            self._ros2_node = rclpy.create_node("camera_module_client")
            self._detect_client = self._ros2_node.create_client(
                Trigger, "/detect_part"
            )
            self._detect_all_client = self._ros2_node.create_client(
                Trigger, "/detect_all"
            )
            logger.info("CameraModule: ROS2 service clients created")
        except ImportError:
            logger.warning(
                "CameraModule: rclpy not available, falling back to mock mode"
            )
            self._use_ros2 = False
        except Exception as e:
            logger.warning(
                f"CameraModule: ROS2 init failed ({e}), falling back to mock mode"
            )
            self._use_ros2 = False

    def observe(self, part_name: str) -> Optional[Dict[str, Any]]:
        """
        Query camera for the current position of a part.

        Args:
            part_name: Name of the part to locate.

        Returns:
            Position dict {"x": float, "y": float, "z": float} if detected,
            None if the part cannot be found.
        """
        if self._mock_observations:
            return self._mock_observations.get(part_name)

        if self._use_ros2 and self._ros2_node and self._detect_client:
            return self._observe_ros2(part_name)

        return None

    def _observe_ros2(self, part_name: str) -> Optional[Dict[str, Any]]:
        """Call /detect_part ROS2 service to get part position."""
        import rclpy
        from std_srvs.srv import Trigger

        try:
            # Set target_part on /perception_node via parameter service.
            # Avoid setting it on this client node, which does not declare it.
            from rcl_interfaces.srv import SetParameters
            from rcl_interfaces.msg import Parameter as ParameterMsg, ParameterValue, ParameterType

            param_client = self._ros2_node.create_client(
                SetParameters, "/perception_node/set_parameters"
            )
            if param_client.wait_for_service(timeout_sec=self._ros2_timeout):
                param_req = SetParameters.Request()
                param_msg = ParameterMsg()
                param_msg.name = "target_part"
                param_msg.value = ParameterValue()
                param_msg.value.type = ParameterType.PARAMETER_STRING
                param_msg.value.string_value = part_name
                param_req.parameters = [param_msg]
                param_future = param_client.call_async(param_req)
                rclpy.spin_until_future_complete(
                    self._ros2_node, param_future, timeout_sec=self._ros2_timeout
                )

            # Call /detect_part
            if not self._detect_client.wait_for_service(timeout_sec=self._ros2_timeout):
                logger.warning("CameraModule: /detect_part service not available")
                return None

            result = None
            for attempt in range(1, self._ros2_call_retries + 1):
                request = Trigger.Request()
                future = self._detect_client.call_async(request)
                rclpy.spin_until_future_complete(
                    self._ros2_node, future, timeout_sec=self._ros2_timeout
                )

                if future.done() and future.result() is not None:
                    result = future.result()
                    break

                logger.warning(
                    "CameraModule: /detect_part call timed out (attempt %d/%d)",
                    attempt,
                    self._ros2_call_retries,
                )

            if result is None:
                logger.warning("CameraModule: /detect_part call failed after retries")
                return None

            data = json.loads(result.message)

            if data.get("detected"):
                # Convert from metres to mm (matching existing interface)
                return {
                    "x": data["x"] * 1000,
                    "y": data["y"] * 1000,
                    "z": data["z"] * 1000,
                }
            return None

        except Exception as e:
            logger.warning(f"CameraModule: ROS2 observe failed: {e}")
            return None

    def observe_all(self) -> Dict[str, Dict[str, float]]:
        """
        Query camera for all currently detected parts.

        Returns:
            Dict mapping part_name → {"x", "y", "z"} for all detected parts.
        """
        if self._mock_observations:
            return {k: v for k, v in self._mock_observations.items() if v is not None}

        if self._use_ros2 and self._ros2_node and self._detect_all_client:
            return self._observe_all_ros2()

        return {}

    def _observe_all_ros2(self) -> Dict[str, Dict[str, float]]:
        """Call /detect_all ROS2 service."""
        import rclpy
        from std_srvs.srv import Trigger

        try:
            if not self._detect_all_client.wait_for_service(timeout_sec=self._ros2_timeout):
                return {}

            request = Trigger.Request()
            future = self._detect_all_client.call_async(request)
            rclpy.spin_until_future_complete(
                self._ros2_node, future, timeout_sec=self._ros2_timeout
            )

            if future.result() is None:
                return {}

            detections = json.loads(future.result().message)
            result = {}
            for d in detections:
                result[d["part_name"]] = {
                    "x": d["x"] * 1000,
                    "y": d["y"] * 1000,
                    "z": d["z"] * 1000,
                }
            return result

        except Exception as e:
            logger.warning(f"CameraModule: ROS2 observe_all failed: {e}")
            return {}

    def destroy(self) -> None:
        """Clean up ROS2 resources."""
        if self._ros2_node:
            self._ros2_node.destroy_node()
            self._ros2_node = None


if __name__ == "__main__":
    import sys
    import subprocess

    # ROS2 Humble requires Python 3.10 — re-exec if running under wrong version
    if sys.version_info[:2] != (3, 10):
        print(f"Current Python is {sys.version_info.major}.{sys.version_info.minor}, "
              f"re-launching with python3.10 (required for rclpy)...")
        result = subprocess.run(["python3.10", __file__])
        sys.exit(result.returncode)

    logging.basicConfig(level=logging.INFO)

    cam = CameraModule(use_ros2=True)
    if not cam._use_ros2:
        print("ERROR: ROS2 mode not available. Is gazebo_camera_detector running?")
        sys.exit(1)

    print("\n--- observe('SG') ---")
    result = cam.observe("SG")
    print(f"  SG: {result}")

    print("\n--- observe('MCP') ---")
    result = cam.observe("MCP")
    print(f"  MCP: {result}")

    print("\n--- observe_all() ---")
    all_parts = cam.observe_all()
    for name, pos in all_parts.items():
        print(f"  {name}: x={pos['x']:.1f} y={pos['y']:.1f} z={pos['z']:.1f}")

    cam.destroy()
