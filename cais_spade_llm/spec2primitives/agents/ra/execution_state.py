from __future__ import annotations

"""Read execution progress and custody without exposing simulator data to composition."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .refinement_records import pin, verify_record

# Application tasks live beyond an individual NiceGUI connection. Transport and
# simulator bindings never enter this read-only context boundary.
ACTIVE: dict[Path, Any] = {}


def execution_busy() -> bool:
    """Return whether this application owns an unfinished Gazebo execution."""
    return any(not session.task.done() for session in tuple(ACTIVE.values()))


def assert_execution_available(contexts_root: Path) -> None:
    """Block either robot when an earlier command may still be active."""
    for path in contexts_root.glob("*/execution/run_*/request.json"):
        root = path.parents[2]
        request_ref = pin(root, path)
        verify_record(root, request_ref)
        result_path = path.parent / "result.json"
        if not result_path.exists():
            raise ValueError(
                "An earlier Gazebo execution was interrupted; its command outcome must be resolved before another run."
            )
        result = verify_record(root, pin(root, result_path))
        if result.get("request_ref") != request_ref:
            raise ValueError("An execution result does not match its request.")
        if result.get("custody_known") is not True:
            raise ValueError(
                "An earlier Gazebo command has an unknown outcome; further execution is blocked for both robots."
            )


def read_primitive_execution_diagnostic(root: Path) -> dict[str, Any]:
    """Read immutable records; never resume a run or construct ROS clients."""
    root = root.resolve()
    directories = sorted((root / "execution").glob("run_*"))
    if not directories:
        return {
            "status": "idle",
            "message": "Validate a program before running it in Gazebo.",
            "events": [],
        }
    directory = directories[-1]
    try:
        request_ref = pin(root, directory / "request.json")
        request = verify_record(root, request_ref)
        if request.get("record_type") != "PrimitiveExecutionRequest":
            raise ValueError("Invalid execution request.")
        events = []
        previous = None
        for path in sorted(directory.glob("event_*.json")):
            reference = pin(root, path)
            event = verify_record(root, reference)
            if event.get("previous_event_ref") != previous:
                raise ValueError("Execution event lineage changed.")
            events.append(event)
            previous = reference
        result_path = directory / "result.json"
        result = verify_record(root, pin(root, result_path)) if result_path.exists() else None
        if result is not None:
            if result.get("request_ref") != request_ref:
                raise ValueError("Execution result does not match its request.")
            if result.get("last_event_ref") != previous:
                raise ValueError("Execution result does not match its events.")
            for reference in result.get("record_refs", []):
                verify_record(root, reference)
            status, message = result["status"], result["message"]
        elif root in ACTIVE and not ACTIVE[root].task.done():
            status = events[-1]["status"] if events else "preparing"
            message = events[-1]["message"] if events else "Preparing Gazebo execution."
            if ACTIVE[root].stop.is_set():
                status, message = (
                    "stopping",
                    "Stopping execution; awaiting command acknowledgments.",
                )
        else:
            status, message = (
                "interrupted",
                "Execution was interrupted. It will not resume automatically.",
            )
        return {
            "status": status,
            "message": message,
            "request": request,
            "result": result,
            "events": events,
            "step_index": events[-1].get("step_index") if events else None,
            "total_steps": request["total_steps"],
            "candidate_ref": request["candidate_ref"],
        }
    except (OSError, KeyError, TypeError, ValueError) as exc:
        return {
            "status": "blocked",
            "message": f"Execution records cannot be verified: {exc}",
            "events": [],
        }


def execution_custody(contexts_root: Path, resource_jid: str) -> Mapping[str, Any] | None:
    """Return only verified semantic custody from the latest resource execution.

    Unknown acknowledgments and unfinished commands must not appear as an empty
    gripper in a later composition snapshot. Simulator identifiers stay private.
    """
    matches = []
    for path in contexts_root.glob("*/execution/run_*/request.json"):
        root = path.parents[2]
        request = verify_record(root, pin(root, path))
        if request.get("resource_jid") == resource_jid:
            matches.append((request["created_at_ns"], root, path.parent))
    if not matches:
        return None
    _, root, directory = max(matches, key=lambda item: item[0])
    result_path = directory / "result.json"
    if not result_path.exists():
        raise ValueError(
            "A previous execution has no final custody acknowledgment; further robot work is blocked."
        )
    view = read_primitive_execution_diagnostic(root)
    if view["status"] == "blocked":
        raise ValueError(view["message"])
    result = verify_record(root, pin(root, result_path))
    if result.get("custody_known") is not True:
        raise ValueError(
            "Execution custody is uncertain; inspect the recorded execution before further robot work."
        )
    return {"held_part": result["held_part"], "gripper_state": result["gripper_state"]}
