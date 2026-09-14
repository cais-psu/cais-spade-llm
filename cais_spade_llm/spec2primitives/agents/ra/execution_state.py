from __future__ import annotations

"""Read execution progress and custody without exposing simulator data to composition."""

from collections.abc import Mapping
import math
from pathlib import Path
from typing import Any

from .refinement_records import owned_path, pin, verify_record

# Application tasks live beyond an individual NiceGUI connection. Transport and
# simulator bindings never enter this read-only context boundary.
ACTIVE: dict[Path, Any] = {}


def execution_busy() -> bool:
    """Return whether this application owns an unfinished Gazebo execution."""
    return any(not session.task.done() for session in tuple(ACTIVE.values()))


def reset_history(contexts_root: Path) -> list[dict[str, Any]]:
    """Pin every historical execution/reset record before changing the shared scene."""
    history = []
    for path in sorted(contexts_root.glob("*/execution/*/request.json")):
        if not path.parent.name.startswith(("run_", "reset_")):
            continue
        root = path.parents[2]
        references = [pin(root, record) for record in sorted(path.parent.rglob("*.json"))]
        for reference in references:
            verify_record(root, reference)
        history.append({"interaction": root.name, "directory": path.parent.relative_to(root).as_posix(),
                        "records": references})
    return history


def verified_reset(contexts_root: Path) -> dict[str, Any] | None:
    """Read the latest reset, rejecting incomplete results or modified covered history."""
    resets = []
    for path in contexts_root.glob("*/execution/reset_*/request.json"):
        root = path.parents[2]
        request_ref = pin(root, path)
        request = verify_record(root, request_ref)
        if request.get("record_type") != "PrimitiveExecutionResetRequest":
            raise ValueError("Invalid Gazebo reset request.")
        resets.append((request["created_at_ns"], root, path.parent, request_ref, request))
    if not resets:
        return None
    _, root, directory, request_ref, request = max(resets, key=lambda item: item[0])
    result_path = directory / "result.json"
    if not result_path.exists():
        raise ValueError("Gazebo reset was interrupted; a verified reset is still required.")
    result = verify_record(root, pin(root, result_path))
    if result.get("request_ref") != request_ref or result.get("record_type") != "PrimitiveExecutionResetResult":
        raise ValueError("Gazebo reset result does not match its request.")
    previous = None
    for path in sorted(directory.glob("event_*.json")):
        reference = pin(root, path)
        event = verify_record(root, reference)
        if event.get("previous_event_ref") != previous:
            raise ValueError("Gazebo reset event lineage changed.")
        previous = reference
    if result.get("last_event_ref") != previous:
        raise ValueError("Gazebo reset result does not match its events.")
    if result.get("status") != "reset_completed":
        raise ValueError("Gazebo reset is not verified: " + str(result.get("message", "unknown outcome")))
    baseline = result["baseline"]
    if (result.get("stopped", {}).get("process_status") != "stopped"
            or result["stopped"]["endpoints"]["services"]
            or any(result["stopped"]["endpoints"]["publishers"].values())
            or not baseline["clock_publisher_gid"]
            or baseline["ros_domain_id"] != result["old_endpoints"]["ros_domain_id"]
            or baseline["clock_publisher_gid"] in result["old_endpoints"]["publishers"]["/clock"]
            or not 0 < baseline["first_clock_ns"] < baseline["clock_ns"]
            or not baseline["required_joints"]
            or not set(request["required_joints"]).issubset(baseline["required_joints"])):
        raise ValueError("Gazebo reset baseline is invalid.")
    for name in baseline["required_joints"]:
        joint = baseline["joint_feedback"][name]
        if (not math.isfinite(joint["position"]) or not joint["publisher_gid"]
                or not baseline["first_clock_ns"] <= joint["stamp_ns"] <= baseline["clock_ns"]):
            raise ValueError("Gazebo reset robot feedback is invalid.")
    covered = set()
    for historical in request["history"]:
        owner = owned_path(contexts_root, historical["interaction"])
        if owner.parent != contexts_root:
            raise ValueError("Reset history must reference an exact interaction.")
        folder = owned_path(owner, historical["directory"])
        if not folder.name.startswith(("run_", "reset_")) or folder.parent != owner / "execution":
            raise ValueError("Reset history must reference an execution directory.")
        if {path.relative_to(owner).as_posix() for path in folder.rglob("*.json")} != {
            reference["ref"] for reference in historical["records"]
        }:
            raise ValueError("Covered execution history changed after reset.")
        for reference in historical["records"]:
            if not owned_path(owner, reference["ref"]).is_relative_to(folder):
                raise ValueError("Reset history record belongs to another directory.")
            verify_record(owner, reference)
        covered.add(folder)
    # Every older reset must belong to the latest reset's immutable coverage.
    if any(folder != directory and folder not in covered for _, _, folder, _, _ in resets):
        raise ValueError("Gazebo reset does not cover earlier reset attempts.")
    return {"request": request, "result": result, "covered": covered}


def interrupted_reset_cleared_scene(contexts_root: Path) -> bool:
    """Return whether the latest failed reset proved the old ROS scene disappeared."""
    resets = []
    for path in contexts_root.glob("*/execution/reset_*/request.json"):
        root = path.parents[2]
        request_ref = pin(root, path)
        request = verify_record(root, request_ref)
        if request.get("record_type") != "PrimitiveExecutionResetRequest":
            raise ValueError("Invalid Gazebo reset request.")
        resets.append((int(request["created_at_ns"]), root, path.parent, request_ref))
    if not resets:
        return False
    _, root, directory, request_ref = max(resets, key=lambda item: item[0])
    result_path = directory / "result.json"
    if not result_path.exists():
        return False
    result = verify_record(root, pin(root, result_path))
    if (
        result.get("record_type") != "PrimitiveExecutionResetResult"
        or result.get("request_ref") != request_ref
    ):
        raise ValueError("Gazebo reset result does not match its request.")
    previous = None
    for path in sorted(directory.glob("event_*.json")):
        reference = pin(root, path)
        event = verify_record(root, reference)
        if event.get("previous_event_ref") != previous:
            raise ValueError("Gazebo reset event lineage changed.")
        previous = reference
    if result.get("last_event_ref") != previous:
        raise ValueError("Gazebo reset result does not match its events.")
    stopped = result.get("stopped") or {}
    endpoints = stopped.get("endpoints") or {}
    publishers = endpoints.get("publishers") or {}
    return (
        result.get("status") == "reset_required"
        and stopped.get("process_status") == "stopped"
        and endpoints.get("services") == []
        and "/clock" in publishers
        and len(publishers) >= 2
        and not any(publishers.values())
    )


def _fresh_simulation_cutoff(contexts_root: Path) -> int:
    """Return the newest execution-owned clean-scene start timestamp."""
    cutoff = 0
    for path in contexts_root.glob("*/execution/run_*/request.json"):
        root = path.parents[2]
        request = verify_record(root, pin(root, path))
        if request.get("record_type") != "PrimitiveExecutionRequest":
            raise ValueError("Invalid execution request.")
        if request.get("fresh_simulation") is True:
            cutoff = max(cutoff, int(request["created_at_ns"]))
    return cutoff


def _applicable_verified_reset(contexts_root: Path) -> dict[str, Any] | None:
    """Ignore an older failed manual reset after execution owns a clean restart."""
    cutoff = _fresh_simulation_cutoff(contexts_root)
    latest_reset = 0
    for path in contexts_root.glob("*/execution/reset_*/request.json"):
        root = path.parents[2]
        request = verify_record(root, pin(root, path))
        if request.get("record_type") != "PrimitiveExecutionResetRequest":
            raise ValueError("Invalid Gazebo reset request.")
        latest_reset = max(latest_reset, int(request["created_at_ns"]))
    if cutoff > latest_reset:
        return None
    return verified_reset(contexts_root)


def assert_interaction_current(root: Path) -> None:
    """Keep pre-reset observations, context and programs unavailable for new robot work."""
    reset = _applicable_verified_reset(root.parent)
    if reset is not None and root.name in reset["request"]["invalidated_interactions"]:
        raise ValueError("Gazebo was reset. Start a fresh interaction, observe the scene and compose again.")


def assert_execution_available(
    contexts_root: Path, *, fresh_simulation: bool = False,
) -> None:
    """Block either robot when an earlier command may still be active."""
    if fresh_simulation:
        return
    reset = _applicable_verified_reset(contexts_root)
    cutoff = _fresh_simulation_cutoff(contexts_root)
    for path in contexts_root.glob("*/execution/run_*/request.json"):
        root = path.parents[2]
        request_ref = pin(root, path)
        request = verify_record(root, request_ref)
        if int(request["created_at_ns"]) < cutoff:
            continue
        if reset is not None and path.parent in reset["covered"]:
            continue
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


def read_primitive_execution_diagnostic(root: Path, *, include_active: bool = True) -> dict[str, Any]:
    """Read immutable records; never resume a run or construct ROS clients."""
    root = root.resolve()
    session = ACTIVE.get(root)
    if include_active and session is not None and not session.task.done() and getattr(session, "diagnostic", None):
        view = dict(session.diagnostic)
        if session.stop.is_set():
            view.update(status="stopping", message="Stopping; awaiting outstanding acknowledgments.")
        return view
    try:
        reset = _applicable_verified_reset(root.parent)
    except (OSError, KeyError, TypeError, ValueError) as exc:
        return {"status": "reset_required", "message": str(exc), "events": [], "reset_required": True}
    if reset is not None and root.name in reset["request"]["invalidated_interactions"]:
        return {"status": "reset_completed", "message": reset["result"]["message"],
                "result": reset["result"], "events": [], "interaction_invalidated": True}
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

    Args:
        contexts_root: The owned collection of interaction records.
        resource_jid: The exact resource whose latest custody is required.

    Returns:
        Verified held-part and gripper state, or None when no execution exists.

    Raises:
        ValueError: Records are invalid, commands are unfinished, or custody is unknown.
    """
    reset = _applicable_verified_reset(contexts_root)
    cutoff = _fresh_simulation_cutoff(contexts_root)
    matches = []
    for path in contexts_root.glob("*/execution/run_*/request.json"):
        root = path.parents[2]
        request = verify_record(root, pin(root, path))
        if int(request["created_at_ns"]) < cutoff:
            continue
        if reset is not None and path.parent in reset["covered"]:
            continue
        if request.get("resource_jid") == resource_jid:
            matches.append((request["created_at_ns"], root, path.parent))
    if not matches:
        return {"held_part": None, "gripper_state": None} if reset is not None else None
    _, root, directory = max(matches, key=lambda item: item[0])
    result_path = directory / "result.json"
    if not result_path.exists():
        raise ValueError(
            "A previous execution has no final custody acknowledgment; further robot work is blocked."
        )
    view = read_primitive_execution_diagnostic(root, include_active=False)
    # A verified zero-command preparation failure must permit an explicit retry.
    # Corrupt records share its blocked display status and must still fail closed.
    if view["status"] == "blocked" and (
        not isinstance(view.get("result"), Mapping)
        or view["result"].get("command_dispatched") is not False
    ):
        raise ValueError(view["message"])
    result = verify_record(root, pin(root, result_path))
    if result.get("custody_known") is not True:
        raise ValueError(
            "Execution custody is uncertain; inspect the recorded execution before further robot work."
        )
    return {"held_part": result["held_part"], "gripper_state": result["gripper_state"]}
