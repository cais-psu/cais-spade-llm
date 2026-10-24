"""Ordered Gazebo resource functions shared by execution and capability views."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

_PROGRAMS: dict[str, dict[str, Any]] = {
    "machine_part": {
        "function_name": "trim_part",
        "entry_state": "loaded",
        "success_state": "completed",
        "status": "implemented",
        "steps": [{"id": "process", "op": "dwell"}],
    },
    # Transport observes the parts it moves and computes their targets before it moves, as a robot does.
    "advance_conveyor": {
        "function_name": "move_parts_downstream",
        "entry_state": "idle",
        "success_state": "idle",
        "status": "implemented",
        "steps": [
            {"id": "observe", "op": "detect_parts"},
            {"id": "targets", "op": "compute_transport_targets"},
            {"id": "transport", "op": "move_relative"},
        ],
    },
    "advance_part": {
        "function_name": "move_to_next_zone",
        "entry_state": "idle",
        "success_state": "idle",
        "status": "implemented",
        "steps": [
            {"id": "observe", "op": "detect_parts"},
            {"id": "targets", "op": "compute_transport_targets"},
            {"id": "transport", "op": "move_relative"},
        ],
    },
    # Printing starts the print; the output appears when its completion is observed.
    "print_part": {
        "function_name": "print_part",
        "entry_state": "idle",
        "success_state": "printing",
        "status": "planned",
        "availability_note": "The current Gazebo scene starts with all supported printer outputs present; no printing executor is bound.",
        "steps": [],
    },
}


def workflow_task_program(event_name: str) -> dict[str, Any]:
    """Return the declared program for one exact capability event.

    Args:
        event_name: Existing capability event name.

    Returns:
        An independent program dictionary, or an empty dictionary if absent.
    """
    return deepcopy(_PROGRAMS.get(event_name, {}))
