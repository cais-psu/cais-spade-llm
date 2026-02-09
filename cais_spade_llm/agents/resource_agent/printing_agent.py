"""Printing resource agent stub.

This module intentionally keeps the agent thin and defers behavior to
ResourceAgent. Add printer-specific tools here as the system evolves.
"""

from agents.resource_agent.resource_agent import ResourceAgent

# --- Printing / Resource Agent ---
class PrintingAgent(ResourceAgent):
    """Placeholder printing agent; inherits all task-handling from ResourceAgent."""
    # No printer-specific tools yet, but expose minimal runtime state for replanning context.
    def _snapshot_state(self):
        return {
            "resource_type": "printer",
            "agent_name": self.agent_name,
            "current_state": getattr(self, "_current_state", "idle"),
            "active_job": getattr(self, "_active_job", None),
        }
