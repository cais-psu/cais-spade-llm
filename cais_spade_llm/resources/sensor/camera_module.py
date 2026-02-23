from __future__ import annotations

from typing import Any, Dict, Optional


class CameraModule:
    """
    Sensor module for observing part positions in the workspace.

    ProductAgent uses this to verify placement after every assemble_part ACK.
    The robot places the part and reports mechanical outcome; CameraModule
    provides the ground truth about where the part actually ended up.

    Observation outcomes:
      - Returns {"x": float, "y": float, "z": float} if part is detected.
      - Returns None if part cannot be located (untracked → human intervention).

    In production, replace observe() with real perception logic.
    For simulation, pass mock_observations at construction time.

    Example (SG slipped into UR5e territory, MCP correctly placed):
        CameraModule(mock_observations={
            "SG":  {"x": 750, "y": -200, "z": 50},   # ur5e-only zone (x > 650)
            "MCP": {"x": 400, "y": -100, "z": 50},   # assembly board, both reachable
        })

    Workspace boundaries (from robot configs, units: mm):
        xarm6: x=[-150, 650], y=[-550, 100], z=[0, 600]
        ur5e:  x=[100,  900], y=[-450, 200], z=[0, 600]
    """

    def __init__(
        self,
        mock_observations: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> None:
        # part_name → {"x": float, "y": float, "z": float}
        # None entry or missing key → part is undetectable (human intervention required)
        self._mock_observations: Dict[str, Optional[Dict[str, float]]] = mock_observations or {}

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

        # TODO: integrate real camera perception.
        # Example integration point:
        #   frame = self._capture_frame()
        #   detection = self._run_detector(frame, part_name)
        #   if detection:
        #       return {"x": detection.x, "y": detection.y, "z": detection.z}
        #   return None
        return None
