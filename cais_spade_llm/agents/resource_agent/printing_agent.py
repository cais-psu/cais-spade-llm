"""Printing resource agent with recovery-capable job control primitives.

This module keeps the agent thin and defers general behavior to
ResourceAgent. Printer-specific recovery primitives (pause/resume/cancel)
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
    """Printing agent with recovery-capable job control primitives."""

    _RESOURCE_PROFILE = PRINTER_PROFILE
    _RECOVERY_PRIMITIVES: list[str] = ["pause_job", "resume_job", "cancel_job"]

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

    def recovery_des_model(
        self,
        *,
        snapshot: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return the printer's private task-level recovery DES model."""
        from cais_spade_llm.resources.resource_primitives import (
            build_recovery_des_model,
        )

        live_snapshot = dict(snapshot or self.get_recovery_snapshot())
        current_state = live_snapshot.get("current_state")
        raw_descriptor = {
            "state_variables": {
                "resource_state": {
                    "scope": "resource",
                    "domain": [current_state, "idle", "printing", "paused"],
                },
            },
            "current_valuation": {
                "resource_state": current_state,
            },
            "events": [
                {
                    "event_name": "pause_job",
                    "controllable": True,
                    "observable": True,
                    "guards": {"resource_state": {"equals": "printing"}},
                    "updates": {"resource_state": {"set": "paused"}},
                },
                {
                    "event_name": "resume_job",
                    "controllable": True,
                    "observable": True,
                    "guards": {"resource_state": {"equals": "paused"}},
                    "updates": {"resource_state": {"set": "printing"}},
                },
                {
                    "event_name": "cancel_job",
                    "controllable": True,
                    "observable": True,
                    "guards": {"resource_state": {"not_equals": "idle"}},
                    "updates": {"resource_state": {"set": "idle"}},
                },
            ],
            "marked_state_conditions": list(
                self.static_capabilities.get("recovery_marked_state_conditions")
                or []
            ),
        }
        return build_recovery_des_model(
            self,
            snapshot=live_snapshot,
            descriptor=raw_descriptor,
        )

    # ------------------------------------------------------------------
    # Recovery primitives — YAML frontmatter MUST come first in docstring
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

    def check_recovery_physical_feasibility(
        self,
        *,
        part_context: dict[str, Any] | None = None,
        recovery_snapshot: dict[str, Any] | None = None,
        grounded_action: dict[str, Any] | None = None,
        **_compat_kwargs: Any,
    ) -> dict[str, Any]:
        """Run printer-specific recovery physical feasibility checks."""
        del part_context, grounded_action
        snapshot = recovery_snapshot or {}
        material = str(snapshot.get("material_state") or "").strip()
        bed = str(snapshot.get("bed_state") or "").strip()

        if material and material in ("empty", "out"):
            return {
                "allowed": False,
                "reason": f"material_state is '{material}' - cannot execute recovery",
            }
        if bed and bed in ("error", "fault"):
            return {
                "allowed": False,
                "reason": f"bed_state is '{bed}' — printer fault detected",
            }
        return {"allowed": True, "reason": "printer feasibility OK"}
