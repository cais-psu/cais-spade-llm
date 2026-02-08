"""Printing resource agent stub.

This module intentionally keeps the agent thin and defers behavior to
ResourceAgent. Add printer-specific tools here as the system evolves.
"""

import json

from agents.resource_agent.resource_agent import ResourceAgent

# --- Printing / Resource Agent ---
class PrintingAgent(ResourceAgent):
    """Placeholder printing agent; inherits all task-handling from ResourceAgent."""
    # No extra behavior yet; tool methods can be added as needed.
    pass
