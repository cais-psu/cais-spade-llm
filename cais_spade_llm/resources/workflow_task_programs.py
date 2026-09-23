"""Ordered Gazebo resource functions shared by execution and capability views."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

_PROGRAMS: dict[str, dict[str, Any]] = {
    "machine_part": {
        "entry_state": "loaded",
        "success_state": "completed",
        "status": "implemented",
        "steps": [
            {"id": "workholding", "op": "observe_workholding"},
            {"id": "clearance", "op": "verify_process_clearance"},
            {"id": "process", "op": "run_machining_clock"},
            {"id": "observation", "op": "confirm_process_observation"},
        ],
    },
    "advance_conveyor": {
        "entry_state": "belt_stopped",
        "success_state": "belt_stopped",
        "status": "implemented",
        "steps": [
            {"id": "residents", "op": "observe_belt_residents"},
            {"id": "displacement", "op": "compute_shared_displacement"},
            {"id": "clearance", "op": "verify_transport_clearance"},
            {"id": "transport", "op": "move_belt_residents"},
            {"id": "arrival", "op": "confirm_arrival"},
        ],
    },
    "advance_part": {
        "entry_state": "occupied source zone",
        "success_state": "occupied downstream zone",
        "status": "implemented",
        "steps": [
            {"id": "part", "op": "observe_zone_part"},
            {"id": "motion", "op": "compute_downstream_motion"},
            {"id": "clearance", "op": "verify_transport_clearance"},
            {"id": "transport", "op": "move_buffer_part"},
            {"id": "arrival", "op": "confirm_arrival"},
        ],
    },
    "print_part": {
        "entry_state": "output absent",
        "success_state": "output present",
        "status": "planned",
        "availability_note": "The current Gazebo scene starts with all supported printer outputs present; no printing executor is bound.",
        "steps": [
            {"id": "request", "op": "validate_print_request"},
            {"id": "job", "op": "run_print_cycle"},
            {"id": "output", "op": "confirm_printed_output"},
        ],
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
