"""Printing resource agent with bridge-capable job control primitives.

This module keeps the agent thin and defers general behavior to
ResourceAgent. Printer-specific bridge primitives (pause/resume/cancel)
are defined here with YAML frontmatter for catalog introspection.
"""

from __future__ import annotations

import logging
from typing import Any

from cais_spade_llm.agents.resource_agent.resource_agent import ResourceAgent
from cais_spade_llm.resources.machine.printer_profile import PRINTER_PROFILE

logger = logging.getLogger(__name__)


# --- Printing / Resource Agent ---
class PrintingAgent(ResourceAgent):
    """Printing agent with bridge-capable job control primitives."""

    _RESOURCE_PROFILE = PRINTER_PROFILE
    _BRIDGE_PRIMITIVES: list[str] = ["pause_job", "resume_job", "cancel_job"]

    # Expose minimal runtime state for replanning context.
    def _snapshot_state(self) -> dict[str, Any]:
        return {
            "resource_type": "printer",
            "agent_name": self.agent_name,
            "current_state": getattr(self, "_current_state", "idle"),
            "current_location": getattr(self, "_current_location", None),
            "active_job": getattr(self, "_active_job", None),
            "job_state": getattr(self, "_job_state", getattr(self, "_current_state", "idle")),
            "material_state": getattr(self, "_material_state", None),
            "bed_state": getattr(self, "_bed_state", None),
        }

    # ------------------------------------------------------------------
    # Bridge primitives — YAML frontmatter MUST come first in docstring
    # for FunctionAnalyzer._extract_yaml_frontmatter() to parse it.
    # ------------------------------------------------------------------

    async def pause_job(self, *, job_id: str = "", **kwargs: Any) -> dict[str, Any]:
        """
        ---
        preconditions:
          current_state:
            equals: "printing"
        effects:
          current_state:
            set: "paused"
          job_state:
            set: "paused"
        ---
        Pause the active print job without cancelling it.
        """
        logger.info("[Printer] pause_job: job_id=%s", job_id or self._active_job)
        self._current_state = "paused"
        if hasattr(self, "_job_state"):
            self._job_state = "paused"
        return {"success": True, "state": "paused"}

    async def resume_job(self, *, job_id: str = "", **kwargs: Any) -> dict[str, Any]:
        """
        ---
        preconditions:
          current_state:
            equals: "paused"
        effects:
          current_state:
            set: "printing"
          job_state:
            set: "printing"
        ---
        Resume a previously paused print job.
        """
        logger.info("[Printer] resume_job: job_id=%s", job_id or self._active_job)
        self._current_state = "printing"
        if hasattr(self, "_job_state"):
            self._job_state = "printing"
        return {"success": True, "state": "printing"}

    async def cancel_job(self, *, job_id: str = "", **kwargs: Any) -> dict[str, Any]:
        """
        ---
        preconditions:
          current_state:
            not_equals: "idle"
        effects:
          current_state:
            set: "idle"
          active_job:
            set: null
          job_state:
            set: "idle"
        ---
        Cancel the active print job and return the printer to idle.
        """
        logger.info("[Printer] cancel_job: job_id=%s", job_id or self._active_job)
        self._current_state = "idle"
        self._active_job = None
        if hasattr(self, "_job_state"):
            self._job_state = "idle"
        return {"success": True, "state": "idle"}

    def bridge_feasibility_oracle(
        self,
        *,
        event_instance: Any | None = None,
        schema: Any | None = None,
        projection: Any | None = None,
        part_context: dict[str, Any] | None = None,
        bridge_snapshot: dict[str, Any] | None = None,
        **_compat_kwargs: Any,
    ) -> dict[str, Any]:
        """Printer-specific bridge feasibility checks."""
        del event_instance, schema, projection, part_context
        snapshot = bridge_snapshot or {}
        material = str(snapshot.get("material_state") or "").strip()
        bed = str(snapshot.get("bed_state") or "").strip()

        if material and material in ("empty", "out"):
            return {
                "allowed": False,
                "reason": f"material_state is '{material}' — cannot execute bridge",
            }
        if bed and bed in ("error", "fault"):
            return {
                "allowed": False,
                "reason": f"bed_state is '{bed}' — printer fault detected",
            }
        return {"allowed": True, "reason": "printer feasibility OK"}
