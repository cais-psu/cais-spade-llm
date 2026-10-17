from __future__ import annotations

"""Store a failure capture as raw facts plus the views derived from them.

``observations.json`` (schema_version 2) keeps only what the run observed or was
told: experiment metadata, the resolved run configuration with source-file
hashes, agent and device reports, measurements, and agent declarations.
``beliefs.json`` keeps what the framework derives from those facts: beliefs,
plant-model copies, reconciled resource evidence, pending nodes and the prepared
recovery request. It is bound to the raw file by sha256, and readers merge both
files into the in-memory capture the framework already uses.
"""

import argparse
import hashlib
import json
import logging
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

from cais_spade_llm.recovery_framework import ROOT
from cais_spade_llm.recovery_framework.diagnostics import read_object, write_record

logger = logging.getLogger(__name__)

RAW_SCHEMA_VERSION = 2
BELIEFS_NAME = "beliefs.json"

METADATA_KEYS = frozenset({
    "schema_version", "run_id", "captured_at", "context_revision", "program_revision", "evidence_source",
    "scenario", "setup", "jids", "nominal_permitted_resources", "recovery_permitted_resources",
    "converted_from", "source_path", "source_sha256",
})
CONFIGURATION_KEYS = frozenset({"inputs", "requirements", "initial_product_states", "config_references"})
REPORT_KEYS = frozenset({
    "failure", "transitions", "acknowledgements", "unavailable_resources", "fault_reports",
    "processing_observations", "processing_shutdown", "failure_processing_observations", "monitor_history",
    "additional_observations", "checkpoint_motion_cancellation", "capture_checkpoint",
})
MEASUREMENT_KEYS = frozenset({"physical_evidence", "planning_scene_capture", "robot_poses"})
RAW_KEYS = METADATA_KEYS | CONFIGURATION_KEYS | REPORT_KEYS | MEASUREMENT_KEYS
DERIVED_KEYS = frozenset({
    "prepared_recovery_request", "resource_evidence", "pending_nodes", "part_tracker", "resource_states",
    "offline_ready", "missing", "safety_status", "llm_model", "llm_reasoning_effort", "product_jid",
})
# Views inside raw sections that are computed from beliefs or configuration.
DERIVED_FIELDS = {
    "failure": ("resource_values_after", "part_tracker_after"),
    "physical_evidence": ("snapshot", "environment_models", "environment_valuation", "products",
                          "stationary_parts", "stationary_part_evidence", "evidence_fingerprint"),
}
# Agent-owned fields recorded inside derived records; raw files keep a copy.
RAW_EXTRACTS = ("agent_reports", "agent_declarations")
_DECLARED_FIELDS = ("resource_type", "static_capabilities", "execution_primitive_catalog")
_CONFIGURATION_FILES = ("scene_file", "product_geometry_file", "selected_product_order_file",
                        "recovery_experiment_settings_file")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _encode(value: dict) -> bytes:
    # Matches write_record so raw bytes and their recorded hash agree.
    return json.dumps(value, indent=2, ensure_ascii=False).encode("utf-8")


def _publish(path: Path, data: bytes) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(data)
    temporary.replace(path)


def _beliefs(data: bytes, derived: dict) -> bytes:
    return _encode({"preparation_version": 1, "observations_sha256": _sha256(data),
                    "derived_at": datetime.now(timezone.utc).isoformat(), **derived})


def config_references(setup: dict, *, root: Path = ROOT) -> list[dict]:
    """Hash the configuration files a run was started from.

    Args:
        setup: Runtime setup naming the scene, geometry, order, settings and safety files.
        root: Repository root for relative paths.

    Returns:
        One ``{"path", "sha256"}`` row per readable file, in setup order.
    """
    paths = [setup.get(key) for key in _CONFIGURATION_FILES] + list(setup.get("selected_safety_files") or [])
    references = []
    for value in paths:
        if not isinstance(value, str) or not value or value == "__NONE__":
            continue
        path = Path(value) if Path(value).is_absolute() else root / value
        if path.is_file():
            references.append({"path": value, "sha256": _sha256(path.read_bytes())})
    return references


def agent_reports(resource_evidence: dict) -> dict:
    """Copy each resource agent's own recorded state from its evidence record."""
    return {jid: deepcopy(row["state"]) for jid, row in (resource_evidence or {}).items()
            if isinstance(row, dict) and isinstance(row.get("state"), dict)}


def agent_declarations(prepared: dict) -> dict:
    """Copy what each resource agent declared about itself during preparation.

    Robot DES models are bound to the robot's own snapshot. Other resources'
    models are built from plant models and beliefs, so they stay derived.
    """
    declarations = {}
    for jid, entry in ((prepared or {}).get("recovery_resources") or {}).items():
        if not isinstance(entry, dict):
            continue
        row = {field: deepcopy(entry[field]) for field in _DECLARED_FIELDS if field in entry}
        if entry.get("resource_type") == "robot" and isinstance(entry.get("recovery_des_model"), dict):
            row["recovery_des_model"] = deepcopy(entry["recovery_des_model"])
        declarations[jid] = row
    return declarations


def completed_tasks(observations: dict) -> dict:
    """Return this run's tasks acknowledged as completed by their resource."""
    records = observations.get("acknowledgements", {})
    records = records.values() if isinstance(records, dict) else records if isinstance(records, list) else []
    return {record["task_id"]: record for record in records if isinstance(record, dict)
            and isinstance(record.get("task_id"), str) and record["task_id"]
            and record.get("run_id") == observations.get("run_id")
            and record.get("status") == "completed" and record.get("evidence") == "resource"}


def pending_nodes(failure: dict, observations: dict, *, run_id: str | None = None,
                  jids: dict | None = None) -> list[dict]:
    """Build the planner's pending work from its own report at the failure.

    Args:
        failure: Failure record with ``pending_tasks`` and retained ``continuations``.
        observations: Capture providing acknowledgements, ``run_id`` and ``jids``.
        run_id: Run naming tasks without an identifier; defaults to the capture's run.
        jids: Resource identities; default to the capture's.

    Returns:
        Unacknowledged, deduplicated task nodes in report order.
    """
    run_id = observations["run_id"] if run_id is None else run_id
    jids = observations["jids"] if jids is None else jids
    tasks = list(failure.get("pending_tasks", []))
    continuations = failure.get("continuations", {})
    # Resource cleanup paths can have been cached before the acknowledged
    # pickup. The retained part work must precede those cleanup goals.
    keys = sorted(continuations, key=lambda key: key.startswith("resource:"))
    tasks += [task for key in keys for task in continuations[key].get("path", [])]
    nodes, seen = [], set()
    completed = completed_tasks(observations)
    for index, task in enumerate(tasks):
        if task.get("task_id") in completed:
            continue
        key = (task["resource_id"], task["event_id"], str(task["parameters"]))
        if key in seen:
            continue
        seen.add(key)
        nodes.append({
            "id": task.get("task_id", f"{run_id}_pending_{index}"),
            "type": "task", "function_name": task["event_name"],
            "resource_jid": jids[task["resource_id"]],
            "params": deepcopy(task["parameters"]), "status": "pending",
            "predecessors": [], "successors": [], "sequence_index": index,
        })
    return nodes


def split_capture(record: dict) -> tuple[dict, dict]:
    """Separate raw facts from derived views without copying large values.

    Args:
        record: In-memory schema-2 capture.

    Returns:
        The raw record and the derived ``views`` with nested ``derived_fields``.
    """
    raw, views, fields = {}, {}, {}
    for key, value in record.items():
        if key in RAW_EXTRACTS:
            continue
        (raw if key in RAW_KEYS else views)[key] = value
    unknown = sorted(key for key in views if key not in DERIVED_KEYS)
    if unknown:
        logger.debug("Capture keys stored as derived views: %s", ", ".join(unknown))
    for section, names in DERIVED_FIELDS.items():
        if isinstance(raw.get(section), dict):
            raw[section] = dict(raw[section])
            moved = {name: raw[section].pop(name) for name in names if name in raw[section]}
            if moved:
                fields[section] = moved
    raw["schema_version"] = RAW_SCHEMA_VERSION
    if record.get("resource_evidence"):
        raw["agent_reports"] = agent_reports(record["resource_evidence"])
    if record.get("prepared_recovery_request"):
        raw["agent_declarations"] = agent_declarations(record["prepared_recovery_request"])
    return raw, {"views": views, "derived_fields": fields}


def merge_capture(raw: dict, beliefs: dict) -> dict:
    """Rebuild the in-memory capture from raw facts and their derived views."""
    record = {key: value for key, value in raw.items() if key not in RAW_EXTRACTS}
    record.update(beliefs.get("views") or {})
    for section, fields in (beliefs.get("derived_fields") or {}).items():
        record[section] = {**(record.get(section) or {}), **fields}
    return record


def write_capture(path: Path, record: dict) -> None:
    """Publish a capture; schema-2 records are split into raw and derived files.

    Args:
        path: Destination ``observations.json``.
        record: In-memory capture. Older schemas are written unchanged.
    """
    path = Path(path)
    if record.get("schema_version") != RAW_SCHEMA_VERSION:
        write_record(path, record)
        return
    raw, derived = split_capture(record)
    data = _encode(raw)
    _publish(path.with_name(BELIEFS_NAME), _beliefs(data, derived))
    _publish(path, data)


def write_beliefs(path: Path, record: dict) -> None:
    """Rewrite the derived views of a saved schema-2 capture without touching its raw file.

    Args:
        path: The capture's ``observations.json``.
        record: The capture as read by :func:`read_capture`, with changed derived views.

    Raises:
        ValueError: The record's raw facts differ from the saved raw file.
    """
    path = Path(path)
    raw, derived = split_capture(record)
    data = path.read_bytes()
    if _encode(raw) != data:
        raise ValueError(f"{path} holds different raw facts; only derived views may be rewritten")
    _publish(path.with_name(BELIEFS_NAME), _beliefs(data, derived))


def read_capture(path: Path) -> dict:
    """Read a capture file, merging its derived views when they match the raw file.

    Args:
        path: An ``observations.json`` file.

    Returns:
        The in-memory capture. Stale or missing views are left out, so
        preparation reports what is missing instead of using them.
    """
    path = Path(path)
    data = path.read_bytes()
    record = json.loads(data)
    if not isinstance(record, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    if record.get("schema_version") != RAW_SCHEMA_VERSION:
        return record
    beliefs_path = path.with_name(BELIEFS_NAME)
    if not beliefs_path.is_file():
        return merge_capture(record, {})
    beliefs = read_object(beliefs_path)
    if beliefs.get("observations_sha256") != _sha256(data):
        logger.warning("%s does not match %s; its derived views are ignored", beliefs_path, path.name)
        return merge_capture(record, {})
    return merge_capture(record, beliefs)


def legacy_fault_reports(observations: dict, fault_report_values: list[str]) -> dict:
    """Restate a schema-1 capture's fault states as device fault reports.

    Schema-1 injection wrote these values into the resource valuation, so the
    report source names that path.
    """
    reports = {}
    for rid, values in (observations.get("resource_states") or {}).items():
        if isinstance(values, dict) and values.get("resource_state") in fault_report_values:
            reports[rid] = {"reported_state": values["resource_state"],
                            "source": f"/resource_states/{rid}/resource_state (schema 1)"}
    return reports


def convert_legacy_capture(observations: dict, fault_report_values: list[str]) -> dict:
    """Restate a schema-1 capture in schema 2 without removing any fact.

    Args:
        observations: A schema-1 capture as returned by ``read_observations``.
        fault_report_values: Configured device fault-report values.

    Returns:
        A schema-2 in-memory capture. Labels the old injection wrote into agent
        state stay where they were recorded; only re-capture removes them.
    """
    record = {key: value for key, value in observations.items() if key not in {"source_path", "source_sha256"}}
    record.update(schema_version=RAW_SCHEMA_VERSION,
                  fault_reports=legacy_fault_reports(observations, fault_report_values),
                  converted_from={"schema_version": observations.get("schema_version"),
                                  "source_path": observations.get("source_path"),
                                  "source_sha256": observations.get("source_sha256")})
    return record


_STATE_FIELDS = frozenset({"resource_state", "current_state", "state"})
_OWN_REPORTS = ("/fault_reports", "/agent_reports", "/agent_declarations", "/physical_evidence/resources/")


def label_audit(raw: dict, fault_report_values: list[str]) -> list[dict]:
    """List state labels in a raw file that no agent reported about itself.

    A resource may report its own fault state: in its device fault report, its
    recorded agent state, or its own physical observation. ``misplaced``
    anywhere, or a fault state elsewhere, is a label written for the agent.

    Args:
        raw: A raw schema-2 record.
        fault_report_values: Configured device fault-report values.

    Returns:
        One ``{"path", "value"}`` row per label.
    """
    findings = []

    def visit(value, path: str, field: str) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                visit(item, f"{path}/{key}", key)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                visit(item, f"{path}/{index}", field)
        elif value == "misplaced" or (
                field in _STATE_FIELDS and value in fault_report_values and not path.startswith(_OWN_REPORTS)):
            findings.append({"path": path, "value": value})

    visit(raw, "", "")
    return findings


def _without_sources(grounding: dict) -> dict:
    # Evidence paths name where a fact is stored, which differs between schemas.
    result = deepcopy(grounding)
    for row in result["fault_event"]["signals"]:
        row.pop("evidence", None)
    return result


def main() -> None:
    """Convert retained schema-1 captures into schema-2 copies and check them."""
    from cais_spade_llm.recovery_framework.failure_grounding import (
        ground_failure,
        load_grounding_settings,
    )
    from cais_spade_llm.recovery_framework.part_slippage_example import default_observation_paths
    from cais_spade_llm.recovery_framework.recovery_observations import (
        observation_fingerprint,
        read_observations,
    )

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("paths", nargs="*", help="Saved schema-1 observations.json files")
    parser.add_argument("--official", action="store_true", help="Convert the retained official captures")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "cais_spade_llm/monitor/environment_runs/capture_v2_preview")
    args = parser.parse_args()
    paths = [Path(value) for value in args.paths]
    if args.official:
        paths += [Path(value) for value in default_observation_paths().values()]
    settings = load_grounding_settings()
    report = {"created_at": datetime.now(timezone.utc).isoformat(), "captures": []}
    for path in paths:
        original = read_observations(path)
        record = convert_legacy_capture(original, settings["fault_report_values"])
        destination = args.output_dir / path.parent.name / "observations.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        write_capture(destination, record)
        raw = json.loads(destination.read_bytes())
        merged = read_capture(destination)
        raw_grounding = ground_failure({key: value for key, value in raw.items() if key not in RAW_EXTRACTS})
        report["captures"].append({
            "source": str(path), "scenario": original.get("scenario"), "output": str(destination),
            "lossless": observation_fingerprint(merged) == observation_fingerprint(record),
            "grounding_equal": _without_sources(raw_grounding) == _without_sources(ground_failure(original)),
            "grounding_evidence_paths": sorted({row["evidence"] for row in raw_grounding["fault_event"]["signals"]}),
            "raw_bytes": destination.stat().st_size,
            "beliefs_bytes": destination.with_name(BELIEFS_NAME).stat().st_size,
            "original_bytes": path.stat().st_size,
            "raw_keys": sorted(raw), "unclassified_views": sorted(
                key for key in record if key not in RAW_KEYS | DERIVED_KEYS | set(RAW_EXTRACTS)),
            "labels_in_raw": label_audit(raw, settings["fault_report_values"]),
        })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_record(args.output_dir / "conversion_report.json", report)
    for row in report["captures"]:
        logger.info("%s: lossless=%s grounding_equal=%s raw=%d KB labels=%d", row["scenario"], row["lossless"],
                    row["grounding_equal"], row["raw_bytes"] // 1024, len(row["labels_in_raw"]))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
