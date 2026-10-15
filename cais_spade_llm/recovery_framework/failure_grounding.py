from __future__ import annotations

"""Detect and ground a captured failure from raw facts, without recorded labels.

Detection is an assumed capability: execution monitoring reports task outcomes,
device fault reports and measured state. These rules read only those raw facts.
Labels written by failure injection, the scenario name and capture-time derived
views are excluded, so the same rules apply to any disruption.
"""

import argparse
import json
import logging
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any

from cais_spade_llm.recovery_framework import SCENE_PATH
from cais_spade_llm.recovery_framework.capture_files import RAW_SCHEMA_VERSION, pending_nodes
from cais_spade_llm.recovery_framework.diagnostics import write_record

logger = logging.getLogger(__name__)

_FAILURE_GOAL_FIELDS = ("task_id", "resource_id", "goal_cancelled", "motion_stopped", "terminal_status",
                        "progress", "target_pose")
_JOB_FIELDS = ("task_id", "job_state", "process_completed", "processing_fraction", "part_name")
_PART_FIELDS = ("state", "location")


@lru_cache(maxsize=1)
def load_grounding_settings() -> dict:
    """Read the configured execution-monitoring thresholds.

    Returns:
        The ``execution_monitoring`` settings from the scene configuration.
    """
    return json.loads(SCENE_PATH.read_text(encoding="utf-8"))["execution_monitoring"]


def raw_failure_view(observations: dict, settings: dict | None = None) -> dict:
    """Copy only the raw facts the grounding rules may read.

    Args:
        observations: A saved failure capture.
        settings: Execution-monitoring settings; defaults to the configuration.

    Returns:
        Agent self-reports, measurements, device fault reports and configured
        geometry, each with its source path in ``sources``.
    """
    settings = settings or load_grounding_settings()
    failure = observations.get("failure") or {}
    physical = observations.get("physical_evidence") or {}
    geometry = physical.get("geometry") or {}
    goal = ({key: deepcopy(failure[key]) for key in _FAILURE_GOAL_FIELDS if key in failure}
            if "controller_goal_id" in failure else {})
    jobs = {rid: [{key: deepcopy(job.get(key)) for key in _JOB_FIELDS} for job in (record.get("jobs") or {}).values()]
            for rid, record in (observations.get("processing_observations") or {}).items()}
    unavailable = list(observations.get("unavailable_resources") or [])
    if observations.get("schema_version") == RAW_SCHEMA_VERSION:
        reports = {rid: row for rid, row in (observations.get("fault_reports") or {}).items() if isinstance(row, dict)}
        reported = {rid: row["reported_state"] for rid, row in reports.items()
                    if row.get("reported_state") in settings["fault_report_values"]}
        unavailable += [rid for rid, row in reports.items() if row.get("available") is False and rid not in unavailable]
        nodes = pending_nodes(failure, observations)
        sources = {"reported_fault_states": "/fault_reports/*/reported_state",
                   "pending_nodes": "/failure/{pending_tasks,continuations} without /acknowledgements"}
    else:
        # Schema 1 has no separate device report; a resource's own state value
        # counts only when it is a configured fault report.
        reported = {rid: values["resource_state"] for rid, values in (observations.get("resource_states") or {}).items()
                    if isinstance(values, dict) and values.get("resource_state") in settings["fault_report_values"]}
        nodes = observations.get("pending_nodes") or []
        sources = {"reported_fault_states": "/resource_states/*/resource_state", "pending_nodes": "/pending_nodes"}
    resources = physical.get("resources") or {}
    return {
        "transitions": [{key: deepcopy(row.get(key)) for key in ("acknowledgement", "after", "product_after")}
                        for row in observations.get("transitions") or []],
        "resource_values_before": deepcopy(failure.get("resource_values_before") or {}),
        "part_tracker_before": {name: {key: deepcopy(row.get(key)) for key in _PART_FIELDS}
                                for name, row in (failure.get("part_tracker_before") or {}).items()},
        "pending_nodes": [{key: deepcopy(node.get(key)) for key in ("id", "function_name", "params", "sequence_index")}
                          for node in nodes],
        "controller_goal": goal,
        "machine_jobs": jobs,
        "unavailable_resources": unavailable,
        "reported_fault_states": reported,
        "controller_faults": {rid: str((row.get("controller_hold_observation") or {}).get("fault_reason") or "")
                              for rid, row in resources.items()},
        "attachments": {rid: (row.get("attachment") or {}).get("model_name")
                        for rid, row in resources.items() if isinstance(row.get("attachment"), dict)},
        "fixture_attachments": {row.get("model2"): row.get("model1")
                                for row in (physical.get("physics_evidence") or {}).get("attachments") or []
                                if isinstance(row, dict) and row.get("model2")},
        "parts": {name: {"model_name": row.get("name"), "pose": deepcopy(row.get("pose"))}
                  for name, row in (physical.get("parts") or {}).items()},
        "part_bounds": {name: deepcopy(row.get("local_bounds")) for name, row in (geometry.get("parts") or {}).items()},
        "regions": {name: deepcopy(row.get("bounds")) for name, row in (geometry.get("regions") or {}).items()},
        "jids": deepcopy(observations.get("jids") or {}),
        "sources": {
            "transitions": "/transitions/*/{acknowledgement,after,product_after}",
            "resource_values_before": "/failure/resource_values_before",
            "part_tracker_before": "/failure/part_tracker_before",
            "pending_nodes": sources["pending_nodes"],
            "controller_goal": "/failure (controller goal report)",
            "machine_jobs": "/processing_observations/*/jobs",
            "unavailable_resources": "/unavailable_resources",
            "reported_fault_states": sources["reported_fault_states"],
            "controller_faults": "/physical_evidence/resources/*/controller_hold_observation/fault_reason",
            "attachments": "/physical_evidence/resources/*/attachment/model_name",
            "fixture_attachments": "/physical_evidence/physics_evidence/attachments",
            "parts": "/physical_evidence/parts/*/pose",
            "part_bounds": "/physical_evidence/geometry/parts/*/local_bounds",
            "regions": "/physical_evidence/geometry/regions/*/bounds",
        },
    }


def expected_state(raw: dict) -> dict:
    """Read each agent's own report at the failure and check it against the transition log.

    Args:
        raw: Output of :func:`raw_failure_view`.

    Returns:
        Expected resource values, expected part ledger and any disagreement with
        the latest recorded transition.
    """
    resources = deepcopy(raw["resource_values_before"])
    parts = deepcopy(raw["part_tracker_before"])
    conflicts = []
    if raw["transitions"]:
        latest = max(raw["transitions"], key=lambda row: (row.get("acknowledgement") or {}).get("revision") or 0)
        task = (latest.get("acknowledgement") or {}).get("task_id")
        for rid, values in (latest.get("after") or {}).items():
            for field, value in (values or {}).items():
                if rid in resources and field in resources[rid] and resources[rid][field] != value:
                    conflicts.append({"resource_id": rid, "field": field, "reported": resources[rid][field],
                                      "transition_log": value, "task_id": task})
    return {"resources": resources, "parts": parts, "expected_state_conflicts": conflicts}


def _part_radius(bounds: list | None) -> float:
    if not bounds:
        return 0.0
    return max(max(abs(low), abs(high)) for low, high in bounds)


def _overlaps(pose: list, radius: float, bounds: list, margin: float) -> bool:
    return all(low - margin - radius <= value <= high + margin + radius
               for value, (low, high) in zip(pose[:3], bounds, strict=True))


def _regions_for(location: str, regions: dict) -> dict:
    return {name: bounds for name, bounds in regions.items() if name == location or name.startswith(location + ".")}


def _held_by(attachments: dict, model_name: str | None) -> str | None:
    return next((rid for rid, model in attachments.items() if model_name and model == model_name), None)


def detect_signals(raw: dict, expected: dict, settings: dict) -> list[dict]:
    """Apply the generic execution-monitoring rules.

    Args:
        raw: Output of :func:`raw_failure_view`.
        expected: Output of :func:`expected_state`.
        settings: Execution-monitoring settings.

    Returns:
        One record per detected signal with its rule, resource, task, part and evidence path.
    """
    signals = []
    goal = raw["controller_goal"]
    if goal and (goal.get("goal_cancelled") is True or goal.get("terminal_status") in settings["goal_status_not_completed"]):
        signals.append({"rule": "A1", "signal": "task_outcome", "resource_id": goal.get("resource_id"),
                        "task_id": goal.get("task_id"), "evidence": raw["sources"]["controller_goal"],
                        "values": {key: goal.get(key) for key in ("goal_cancelled", "terminal_status", "progress")}})
    for rid, jobs in raw["machine_jobs"].items():
        for job in jobs:
            if job["job_state"] in settings["job_states_not_completed"] and job["process_completed"] is False:
                signals.append({"rule": "A2", "signal": "task_outcome", "resource_id": rid, "task_id": job["task_id"],
                                "part_name": job["part_name"], "evidence": raw["sources"]["machine_jobs"],
                                "values": {key: job[key] for key in ("job_state", "processing_fraction")}})
    for rid, reason in raw["controller_faults"].items():
        if reason:
            signals.append({"rule": "B1", "signal": "device_fault_report", "resource_id": rid,
                            "evidence": raw["sources"]["controller_faults"], "values": {"fault_reason": reason}})
    for rid in raw["unavailable_resources"]:
        signals.append({"rule": "B2", "signal": "device_fault_report", "resource_id": rid,
                        "evidence": raw["sources"]["unavailable_resources"], "values": {"available": False}})
    for rid, value in raw["reported_fault_states"].items():
        signals.append({"rule": "B2", "signal": "device_fault_report", "resource_id": rid,
                        "evidence": raw["sources"]["reported_fault_states"], "values": {"resource_state": value}})
    margin = float(settings["region_margin_m"])
    for name, part in raw["parts"].items():
        holder = next((rid for rid, values in expected["resources"].items() if values.get("held_part") == name), None)
        measured = _held_by(raw["attachments"], part["model_name"])
        if holder is not None or measured is not None:
            if holder != measured:
                signals.append({"rule": "C1", "signal": "state_discrepancy", "resource_id": holder or measured,
                                "part_name": name, "evidence": raw["sources"]["attachments"],
                                "values": {"expected_holder": holder, "measured_holder": measured}})
            continue
        # A measured fixture attachment places the part; regions only test loose parts.
        if raw["fixture_attachments"].get(part["model_name"]):
            continue
        location = (expected["parts"].get(name) or {}).get("location")
        regions = _regions_for(location, raw["regions"]) if isinstance(location, str) else {}
        if not regions or not isinstance(part["pose"], list):
            continue
        radius = _part_radius(raw["part_bounds"].get(name))
        if not any(_overlaps(part["pose"], radius, bounds, margin) for bounds in regions.values()):
            signals.append({"rule": "C2", "signal": "state_discrepancy", "part_name": name,
                            "evidence": raw["sources"]["parts"], "values": {"expected_location": location,
                                                                            "measured_position": part["pose"][:3]}})
    return signals


def _names_resource(node: dict, rid: str, settings: dict) -> bool:
    params = node.get("params") or {}
    return params.get("resource_id") == rid or any(params.get(key) == rid for key in settings["location_parameters"])


def ground_failure(observations: dict, *, settings: dict | None = None) -> dict:
    """Produce the grounded fault event and symbolic state from raw facts.

    Args:
        observations: A saved failure capture.
        settings: Execution-monitoring settings; defaults to the configuration.

    Returns:
        ``fault_event`` (failure_anchor key names plus detection detail), grounded
        ``parts`` and ``resources``, and ``expected_state_conflicts``.
    """
    settings = settings or load_grounding_settings()
    raw = raw_failure_view(observations, settings)
    expected = expected_state(raw)
    signals = detect_signals(raw, expected, settings)
    primary = [row for row in signals if row["signal"] != "state_discrepancy"]
    faulted = list(dict.fromkeys(row["resource_id"] for row in primary or signals if row.get("resource_id")))
    nodes = sorted(raw["pending_nodes"], key=lambda node: node.get("sequence_index") or 0)
    in_progress = next((row for row in primary if row.get("task_id") and row["resource_id"] in faulted), None)
    blocked = [node for node in nodes if any(_names_resource(node, rid, settings) for rid in faulted)]
    task = next((node for node in nodes if in_progress and node["id"] == in_progress["task_id"]), None)
    task = task or (blocked[0] if blocked else None)
    task_id = in_progress["task_id"] if in_progress else (task or {}).get("id", "")

    measured_parts = {name: _held_by(raw["attachments"], part["model_name"]) for name, part in raw["parts"].items()}
    discrepant = {row["part_name"] for row in signals if row["signal"] == "state_discrepancy"}
    parts = deepcopy(expected["parts"])
    updates = []
    for name in discrepant:
        holder = measured_parts.get(name)
        row = parts.setdefault(name, {})
        if holder:
            row.update(state="in_gripper", location=holder)
        else:
            row.update(state="misplaced", location=None, observed_pose=deepcopy(raw["parts"][name]["pose"]))
        updates.append({"part_name": name, **{key: deepcopy(row[key]) for key in ("state", "location", "observed_pose")
                                              if key in row}})
    resources = deepcopy(expected["resources"])
    for rid, values in resources.items():
        if rid in raw["attachments"] and "held_part" in values:
            model = raw["attachments"][rid]
            measured = next((name for name, part in raw["parts"].items() if model and part["model_name"] == model), None)
            values["des_consistent"] = values.get("held_part") == measured
            values["held_part_measured"] = measured
            if rid in faulted:
                # After a fault the measured custody replaces the interrupted DES bookkeeping.
                if values["held_part"] != measured:
                    updates.append({"resource_id": rid, "field": "held_part", "value": measured})
                values["held_part"] = measured
        if rid in raw["reported_fault_states"]:
            values["resource_state"] = raw["reported_fault_states"][rid]
            updates.append({"resource_id": rid, "field": "resource_state", "value": values["resource_state"]})
        values["faults"] = [row["rule"] for row in signals if row.get("resource_id") == rid]
    affected = list(dict.fromkeys(
        [((task or {}).get("params") or {}).get("part_name")] + sorted(discrepant) +
        [name for name, row in expected["parts"].items() if row.get("location") in faulted]))
    jid = raw["jids"].get(faulted[0]) if faulted else None
    fault_event = {
        "detected": bool(signals),
        "signals": signals,
        "faulted_resources": faulted,
        "failed_task_id": task_id,
        "failed_function_name": (task or {}).get("function_name", ""),
        "failed_resource_jid": jid or "",
        "blocked_tasks": [node["id"] for node in blocked],
        "resource_state_before": deepcopy(expected["resources"].get(faulted[0], {})) if faulted else {},
        "affected_parts": [{"part_name": name, "state": (parts.get(name) or {}).get("state"),
                            "location": (parts.get(name) or {}).get("location")} for name in affected if name],
    }
    return {"fault_event": fault_event, "parts": parts, "resources": resources, "belief_updates": updates,
            "expected_state_conflicts": expected["expected_state_conflicts"]}


def fault_identity(grounded: dict) -> tuple:
    """Return what a grounding identifies, without measured values that vary between samples."""
    event = grounded["fault_event"]
    return (tuple(event["faulted_resources"]), event["failed_task_id"],
            tuple((row["part_name"], row["state"]) for row in event["affected_parts"]))


def update_beliefs(context, observations: dict, grounded: dict) -> list[dict]:
    """Apply grounded fault state to the product's beliefs, live and recorded.

    Execution monitoring, not failure injection, changes the beliefs: a reported
    fault state, measured custody of a faulted resource, and the measured state
    of a discrepant part.

    Args:
        context: Live environment context owning the valuation and part tracker.
        observations: In-memory capture whose belief copies are kept in step.
        grounded: Output of :func:`ground_failure`.

    Returns:
        The applied belief updates; empty when the beliefs already agree.
    """
    updates = grounded["belief_updates"]
    if not updates:
        return []
    with context.admission_lock:
        for row in updates:
            if "resource_id" in row:
                actor = context.resources[row["resource_id"]]
                actor.valuation[row["field"]] = deepcopy(row["value"])
                actor.revision += 1
                recorded = observations.get("resource_states", {}).get(row["resource_id"])
                if isinstance(recorded, dict):
                    recorded[row["field"]] = deepcopy(row["value"])
            else:
                values = {key: deepcopy(value) for key, value in row.items() if key != "part_name"}
                context.part_tracker[row["part_name"]].update(values)
                recorded = observations.get("part_tracker", {}).get(row["part_name"])
                if isinstance(recorded, dict):
                    recorded.update(deepcopy(values))
        context.revision += 1
    return deepcopy(updates)


def has_raw_failure_facts(observations: dict) -> bool:
    """Return whether a capture records the agents' own state at the failure."""
    failure = observations.get("failure") or {}
    return isinstance(failure.get("resource_values_before"), dict) and isinstance(failure.get("part_tracker_before"), dict)


def prompt_fault_event(grounded: dict) -> dict:
    """Present the grounded fault with detection facts and no scenario or checkpoint label.

    Args:
        grounded: Output of :func:`ground_failure`.

    Returns:
        Fault fields in the public ``fault_event`` vocabulary.
    """
    event = grounded["fault_event"]
    faulted = event["faulted_resources"][0] if event["faulted_resources"] else None
    signals = [{key: deepcopy(value) for key, value in {"signal": row["signal"], "resource_id": row.get("resource_id"),
                "task_id": row.get("task_id"), "part_name": row.get("part_name"), **row.get("values", {})}.items()
                if value is not None} for row in event["signals"]]
    return {"focused_resource_jid": event["failed_resource_jid"], "blocked_at_task_id": event["failed_task_id"],
            "blocked_at_function": event["failed_function_name"],
            "resource_state": (grounded["resources"].get(faulted) or {}).get("resource_state"),
            "affected_part_names": [row["part_name"] for row in event["affected_parts"]],
            "detected_signals": signals}


def apply_grounding(prepared: dict, observations: dict) -> dict | None:
    """Replace stored failure summaries in a prepared request with code grounding.

    Captures without recorded agent state at the failure keep their stored
    summaries. A capture with that state must contain a detectable fault.

    Args:
        prepared: Prepared recovery request, updated in place.
        observations: The saved failure capture.

    Returns:
        The grounding, or ``None`` for captures without raw failure facts.

    Raises:
        ValueError: The raw facts show no detectable fault.
    """
    if not has_raw_failure_facts(observations):
        return None
    grounded = ground_failure(observations)
    event = grounded["fault_event"]
    if not event["detected"]:
        raise ValueError("NEEDS_CONTEXT: no failure is detectable from the recorded execution-monitoring facts")
    prepared["code_grounding"] = deepcopy(grounded)
    prepared.setdefault("llm_input", {})["fault_event"] = prompt_fault_event(grounded)
    anchor = prepared.setdefault("failure_anchor", {})
    anchor.pop("failure_type", None)
    anchor.pop("checkpoint", None)
    anchor.update(failed_task_id=event["failed_task_id"], failed_function_name=event["failed_function_name"],
                  failed_resource_jid=event["failed_resource_jid"],
                  resource_state_before=deepcopy(event["resource_state_before"]),
                  affected_parts=deepcopy(event["affected_parts"]))
    for row in prepared["llm_input"].get("part_facts") or []:
        part = grounded["parts"].get(row.get("part_name"))
        if part:
            row.update(current_state=part.get("state"), current_location=part.get("location"))
            if part.get("observed_pose"):
                row["observed_pose"] = deepcopy(part["observed_pose"])
    jids = observations.get("jids") or {}
    for row in (prepared["llm_input"].get("observed_runtime_state") or {}).get("resources") or []:
        rid = next((rid for rid, jid in jids.items() if jid == row.get("resource_jid")), None)
        values = grounded["resources"].get(rid) if rid in event["faulted_resources"] else None
        if values:
            for key in [key for key in ("current_state", "resource_state") if key in row] or ["current_state"]:
                row[key] = values.get("resource_state")
            if "held_part" in row and "held_part" in values:
                row["held_part"] = values["held_part"]
    return grounded


def compare_with_recorded_labels(observations: dict, grounded: dict) -> dict:
    """Compare grounded output with the labels stored by the capture.

    This is the only reader of recorded labels; it exists to measure agreement.

    Args:
        observations: The saved failure capture.
        grounded: Output of :func:`ground_failure`.

    Returns:
        Per-item ``agree``, ``differ`` or ``filled`` (stored value empty) results.
    """
    stored = ((observations.get("prepared_recovery_request") or {}).get("llm_input") or {}).get("fault_event") or {}
    event = grounded["fault_event"]
    jids = observations.get("jids") or {}

    def verdict(recorded: Any, value: Any) -> dict:
        status = "filled" if recorded in (None, "", []) and value not in (None, "", []) else (
            "agree" if recorded == value else "differ")
        return {"stored": recorded, "grounded": value, "status": status}

    result = {
        "faulted_resource": verdict(stored.get("focused_resource_jid"), event["failed_resource_jid"]),
        "interrupted_task": verdict(stored.get("blocked_at_task_id"), event["failed_task_id"]),
        "interrupted_function": verdict(stored.get("blocked_at_function"), event["failed_function_name"]),
        "affected_parts": verdict(sorted(stored.get("affected_part_names") or []),
                                  sorted(row["part_name"] for row in event["affected_parts"])),
        "parts": {},
        "resources": {},
    }
    tracker = observations.get("part_tracker") or {}
    for row in event["affected_parts"]:
        recorded = tracker.get(row["part_name"]) or {}
        result["parts"][row["part_name"]] = verdict({key: recorded.get(key) for key in _PART_FIELDS},
                                                    {key: row[key] for key in _PART_FIELDS})
    states = observations.get("resource_states") or {}
    for rid in event["faulted_resources"]:
        recorded = (states.get(rid) or {}).get("resource_state")
        result["resources"][jids.get(rid, rid)] = verdict(recorded, (grounded["resources"].get(rid) or {}).get("resource_state"))
    return result


def main() -> None:
    """Ground official captures and write an agreement report under the monitor tree."""
    from cais_spade_llm.recovery_framework.part_slippage_example import default_observation_paths
    from cais_spade_llm.recovery_framework.recovery_observations import RECORDS, read_observations

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", help="Saved observations.json files")
    parser.add_argument("--official", action="store_true", help="Use the retained official capture per failure")
    parser.add_argument("--output", type=Path, default=RECORDS.parent / "failure_grounding_report.json")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    paths = list(default_observation_paths().values()) if args.official else args.paths
    report = {"settings": load_grounding_settings(), "captures": []}
    for path in paths:
        observations = read_observations(path)
        grounded = ground_failure(observations)
        report["captures"].append({"source_path": str(path), "run_id": observations.get("run_id"),
                                   "grounded": grounded, "agreement": compare_with_recorded_labels(observations, grounded)})
        event = grounded["fault_event"]
        logger.info("%s: signals=%s resource=%s task=%s function=%s parts=%s", Path(path).parent.name,
                    [row["rule"] for row in event["signals"]], event["faulted_resources"], event["failed_task_id"],
                    event["failed_function_name"], [row["part_name"] for row in event["affected_parts"]])
    write_record(args.output, report)
    logger.info("Report written to %s", args.output)


if __name__ == "__main__":
    main()
