from __future__ import annotations

"""Fresh Gazebo preparation for a saved failure capture."""

import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

from cais_spade_llm.recovery_framework.capture_files import write_capture
from cais_spade_llm.recovery_framework.diagnostics import write_record
from cais_spade_llm.recovery_framework.recovery_observations import (
    fingerprint,
)


async def _prepare_saved_capture(runtime, fresh: dict, template: dict, path: Path) -> dict:
    """Persist raw evidence and preparation errors before publishing ready inputs."""
    write_capture(path, fresh)
    try:
        return await _fresh_preparation(runtime, fresh, template)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        write_record(path.parent / "preparation_error.json", {
            "error_type": type(exc).__name__, "reason": str(exc),
            "observation_path": str(path), "context_revision": fresh.get("context_revision"),
            "captured_at": fresh.get("captured_at"), "recorded_at": datetime.now(timezone.utc).isoformat()})
        raise


async def _fresh_preparation(runtime, fresh: dict, template: dict) -> dict:
    """Rebuild prepared facts through the existing non-LLM preparation path."""
    from cais_spade_llm.recovery_framework.recovery_observations import captured_resource_snapshots

    planner = runtime.agent.process_planner
    tracker = deepcopy(fresh["part_tracker"])
    states = {name: row.get("state") for name, row in tracker.items()}
    locations = {name: row.get("location") for name, row in tracker.items()}
    resources = {}
    for rid, values in fresh["resource_states"].items():
        jid = fresh["jids"][rid]
        owner_state = fresh.get("resource_evidence", {}).get(jid, {}).get("state", {})
        resources[jid] = deepcopy(values)
        if "resource_state" in values:
            resources[jid]["current_state"] = deepcopy(values["resource_state"])
        elif "_current_state" in owner_state:
            resources[jid]["current_state"] = deepcopy(owner_state["_current_state"])
        if "resource_location" in values:
            resources[jid]["current_location"] = deepcopy(values["resource_location"])
        if "unavailable_resources" in fresh:
            resources[jid]["availability"] = "unavailable" if rid in fresh["unavailable_resources"] else values.get("availability", "available")
    missing = [fresh["jids"][rid] for rid, values in fresh["resource_states"].items()
               if "resource_state" in values and values["resource_state"] is None]
    if missing:
        raise ValueError("NEEDS_CONTEXT: recorded resource_state or owner _current_state is missing for " + ", ".join(missing))
    recorded_snapshots = captured_resource_snapshots(fresh)
    robot_ids = {row["resource_id"] for row in fresh.get("inputs", {}).get("scene", {}).get("robots", [])}
    for owner in getattr(runtime, "resource_agents", []):
        physical = fresh.get("physical_evidence", {}).get("snapshot", {}).get("resources", {}).get(owner.agent_name, {})
        jid = str(owner.jid)
        if jid not in recorded_snapshots:
            raise ValueError("NEEDS_CONTEXT: recorded owner snapshot is unavailable for " + owner.agent_name)
        snapshot = deepcopy(recorded_snapshots[jid])
        if owner.agent_name in robot_ids or getattr(owner, "static_capabilities", {}).get("resource_type") == "robot":
            pose = physical.get("current_pose")
            if isinstance(pose, list) and len(pose) == 7:
                pose = dict(zip(("x", "y", "z", "qx", "qy", "qz", "qw"), pose, strict=True))
            if isinstance(pose, dict):
                snapshot["current_pose"] = pose
        else:
            for key in ("current_pose", "base_pose", "joint_names", "joint_positions"):
                if key in physical and key in snapshot and fingerprint(snapshot[key]) != fingerprint(physical[key]):
                    raise ValueError("NEEDS_CONTEXT: conflicting captured " + owner.agent_name + "." + key +
                        ": " + json.dumps({"snapshot": snapshot[key], "physical": physical[key],
                            "source": {field: fresh.get("physical_evidence", {}).get("resources", {}).get(owner.agent_name, {}).get(field)
                                       for field in ("source", "frame", "simulation_time", "observed_monotonic", "launch_id")}},
                            ensure_ascii=False))
        if "held_part" in physical and snapshot.get("held_part") != physical["held_part"]:
            raise ValueError("NEEDS_CONTEXT: captured custody conflicts with owner state for " + owner.agent_name)
        if "availability" in resources.get(jid, {}):
            snapshot["availability"] = deepcopy(resources[jid]["availability"])
        recorded_snapshots[jid] = snapshot
    failure_context = deepcopy(template["failure_context_raw"])
    if "failure" in fresh:
        from cais_spade_llm.agents.intelligent_product.replanner.failure_context import build_failure_event

        failure = fresh["failure"]
        part = failure["part_name"]
        rid = next(rid for rid, jid in fresh["jids"].items() if jid == template["ra_jid"])
        failure_context = build_failure_event(
            failed_task_id=failure_context.get("failed_task_id", ""),
            failed_resource_jid=template["ra_jid"],
            failed_function_name=failure_context.get("failed_function_name", ""),
            final_status="failed:" + fresh["scenario"], part_name=part,
            observations=deepcopy(failure),
            affected_entities=[{"entity_type": "part", "entity_id": part,
                                "state": tracker[part]["state"], "location": tracker[part]["location"]}],
            state_before=failure.get("resource_values_before", {}).get(rid),
            state_after={**deepcopy(fresh["resource_states"][rid]),
                         "part_state": tracker[part]["state"], "part_location": tracker[part]["location"]})
    stuck = planner._build_resource_search_state(resource_jid=template["ra_jid"], resource_states=resources,
        default_resource_state="", part_states=states, part_locations=locations)
    prepared = await planner.prepare_recovery_request(
        stuck_state=stuck, P_id=deepcopy(template["P_id"]), ra_jid=template["ra_jid"],
        goal_state=template["goal_state"], tools_catalog=deepcopy(template["tools_catalog"]),
        part_tracker=tracker, obligation_targets=deepcopy(template["obligation_targets"]),
        recovery_feedback="", resource_states=resources, default_resource_state="",
        part_states=states, part_locations=locations, recovery_safety_context={},
        failure_context=failure_context,
        recorded_resource_snapshots=recorded_snapshots)
    context = getattr(runtime, "context", None)
    if context is not None and fresh.get("context_revision") is not None and context.revision != fresh["context_revision"]:
        raise ValueError("NEEDS_CONTEXT: runtime observations changed during preparation")
    for jid, evidence in fresh.get("resource_evidence", {}).items():
        if jid in prepared.get("recovery_resources", {}):
            prepared["recovery_resources"][jid]["resource_evidence"] = deepcopy(evidence)
    return prepared


