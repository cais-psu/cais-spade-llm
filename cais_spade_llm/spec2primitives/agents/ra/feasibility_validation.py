from __future__ import annotations

"""Expose the owned no-motion planning boundary for ProductAgent arm checks."""

from collections.abc import Mapping
from typing import Protocol


class RobotAgentFeasibilityRuntime(Protocol):
    """Validate position plans through the exact requested RobotAgent."""

    async def validate_plan_only_allocation(
        self,
        request: Mapping[str, object],
    ) -> Mapping[str, object]:
        """Return per-location planning results without executing motion."""
        ...

    async def read_resource_base_pose(
        self, *, base_frame: str, target_frame: str
    ) -> Mapping[str, object]:
        """Read the configured robot base in the grounded location frame."""
        ...
