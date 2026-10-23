from __future__ import annotations

"""Persist failure facts independently of prompt generation and page clients."""

import hashlib
import asyncio
import json
import math
import shutil
import time
import xml.etree.ElementTree as ET
from copy import deepcopy
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from uuid import uuid4

from cais_spade_llm.recovery_framework import ROOT, SCENE_PATH
from cais_spade_llm.recovery_framework.capture_files import (
    BELIEFS_NAME, RAW_SCHEMA_VERSION, config_references, read_capture,
)
from cais_spade_llm.recovery_framework.diagnostics import read_object, write_record

RECORDS = ROOT / "cais_spade_llm/monitor/environment_runs/recovery"


def fingerprint(value: object) -> str:
    """Hash the stored values without changing any project symbols."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def observation_fingerprint(record: dict) -> str:
    """Bind proposals to captured content, independently of its file location."""
    return fingerprint({key: value for key, value in record.items() if key not in {"source_path", "source_sha256"}})


def recorded_processing_fields(record: dict, *, resource_id: str, run_id: str, inputs: dict,
                               values: dict | None = None) -> dict:
    """Validate a saved owner job record before using its public state.

    Args:
        record: The machine's saved owner job record.
        resource_id: The machine.
        run_id: The run the record belongs to.
        inputs: Recorded run inputs with the scene.
        values: The machine's current values, which name the part it holds.

    Returns:
        The machine state the record shows (see ``machine_public_state``).
    """
    from cais_spade_llm.resources.machine.machine_jobs import MachineJobs, machine_public_state

    machines = [row for row in inputs.get("scene", {}).get("machines", [])
                if row.get("resource_id") == resource_id]
    if len(machines) != 1 or not isinstance(record, dict):
        raise ValueError("NEEDS_CONTEXT: recorded machine declaration or job record for " + resource_id)
    MachineJobs(machines[0], run_id, record)
    if type(record.get("revision")) is not int or record["revision"] < 0:
        raise ValueError("NEEDS_CONTEXT: machine job revision for " + resource_id)
    if "active_job" not in record or not isinstance(record.get("jobs"), dict):
        raise ValueError("NEEDS_CONTEXT: machine job identities for " + resource_id)
    active = record["active_job"]
    if active is not None and (not isinstance(active, str) or active not in record["jobs"]):
        raise ValueError("Conflicting active_job identity for " + resource_id)
    for identity, job in record["jobs"].items():
        if not isinstance(job, dict) or job.get("job_id") != identity:
            raise ValueError("Conflicting saved job identity for " + resource_id)
    return machine_public_state(record, values or {})


def reconcile_resource_snapshot(evidence: dict, physical_evidence: dict, *, resource_id: str,
                                run_id: str, revision: int, inputs: dict) -> dict:
    """Combine frozen owner fields with measurements while retaining both sources.

    GETRECOVERYSTATE observes geometry and custody. Its machine job fields are
    copied from the symbolic environment and are not machine-worker observations.
    Earlier primitive endpoints remain history, not simultaneous pose evidence.
    """
    result = deepcopy(evidence)
    owner = deepcopy(result.get("owner_snapshot", result.get("snapshot", {})))
    physical = physical_evidence.get("resources", {}).get(resource_id, {})
    measured = physical_evidence.get("snapshot", {}).get("resources", {}).get(resource_id, {})
    conflicts = result.setdefault("conflicts", [])
    sources = result.setdefault("snapshot_sources", {})
    observed = physical.get("observation_state", {})
    provenance = physical_evidence.get("provenance", {})
    origin = {key: deepcopy(physical.get(key)) for key in (
        "source", "frame", "launch_id", "simulation_time", "observed_monotonic", "native_physics_source")}

    def conflict(field: str, first, second, reason: str) -> None:
        conflicts.append({"field": field, "reason": reason,
            "recorded": deepcopy(first), "observed": deepcopy(second),
            "owner_source": deepcopy(result.get("source", {})), "physical_source": deepcopy(origin)})

    if provenance and (provenance.get("run_id") != run_id or provenance.get("context_revision") != revision):
        conflict("provenance", {"run_id": run_id, "context_revision": revision}, provenance,
                 "Physical observations belong to another run or revision")
    for key, value in measured.items():
        if key in observed and fingerprint(observed[key]) != fingerprint(value):
            conflict(key, value, observed[key], "Physical snapshot disagrees with its source observation")
    for key in ("current_pose", "base_pose", "joint_names", "joint_positions"):
        if key in measured and key in physical and fingerprint(measured[key]) != fingerprint(physical[key]):
            conflict(key, measured[key], physical[key], "Measurements disagree within one observation")
    if measured and physical.get("source") and physical.get("frame") != physical_evidence.get("snapshot", {}).get("frame"):
        conflict("frame", physical_evidence.get("snapshot", {}).get("frame"), physical.get("frame"),
                 "Physical observation frames disagree")
    if "held_part" in owner and "held_part" in measured and owner["held_part"] != measured["held_part"]:
        conflict("held_part", owner["held_part"], measured["held_part"], "Recorded custody conflicts with measured custody")

    snapshot = {**{key: value for key, value in owner.items()
                  if not key.startswith("_") and key != "last_execution_evidence"}, **deepcopy(measured)}
    for field in measured:
        sources[field] = {"ref": f"physical_evidence/snapshot/resources/{resource_id}/{field}", **deepcopy(origin)}
    # base_pose is supplied by the KMR owner alongside its joints and TCP pose.
    for field in ("base_pose", "joint_names", "joint_positions"):
        if field in physical:
            snapshot[field] = deepcopy(physical[field])
            sources[field] = {"ref": f"physical_evidence/resources/{resource_id}/{field}", **deepcopy(origin)}
    processing = result.get("processing_observations")
    if processing is None and any(row.get("resource_id") == resource_id for row in inputs.get("scene", {}).get("machines", [])):
        # The environment's initial nulls are not observations from the worker.
        for field in ("active_job", "job_state", "selected_program"):
            snapshot.pop(field, None)
            sources.pop(field, None)
        result.setdefault("missing", []).append({"ref": "processing_observations/" + resource_id,
            "reason": "Machine-worker observations are unavailable; copied symbolic job fields are not measured facts"})
    if processing is not None:
        try:
            fields = recorded_processing_fields(processing, resource_id=resource_id, run_id=run_id, inputs=inputs,
                                                values=snapshot)
        except (ValueError, KeyError, TypeError) as exc:
            conflict("processing_observations", processing, None, str(exc))
            fields = {}
        for field, value in fields.items():
            for path, row in (("owner_snapshot", owner), ("state", result.get("state", {})),
                              ("environment_valuation", result.get("environment_valuation", {}).get(resource_id, {}))):
                if field in row and fingerprint(row[field]) != fingerprint(value):
                    conflict(field, row[field], value, path + " conflicts with machine-worker job record")
            if field in measured and physical.get("source") != "registered owner / GETRECOVERYSTATE":
                if fingerprint(measured[field]) != fingerprint(value):
                    conflict(field, measured[field], value, "Unidentified physical job-state source conflicts with owner record")
            snapshot[field] = deepcopy(value)
            sources[field] = {"ref": "processing_observations/" + resource_id,
                "run_id": processing["run_id"], "revision": processing["revision"],
                "simulation_clock_sec": processing.get("simulation_clock_sec"),
                "model_fingerprint": processing["model_fingerprint"], "kind": "machine-worker record"}
    result.update(snapshot=snapshot, owner_snapshot=owner,
                  physical_snapshot=deepcopy(measured), snapshot_reconciliation_version=1)
    if not conflicts and physical:
        record_sources = result.setdefault("sources", {})
        for field in ("attachment", "controller_identity", "frame", "custody_complete", "continuous_motion_configuration"):
            if field in physical and field not in result:
                result[field] = deepcopy(physical[field])
                record_sources[field] = f"physical_evidence/resources/{resource_id}/{field}"
        for field in ("tcp_pose", "tcp_link", "controlled_link", "tcp_offset_z", "tcp_transform_source"):
            if field in physical and field not in result.get("motion_config", {}):
                result.setdefault("motion_config", {})[field] = deepcopy(physical[field])
                record_sources["motion_config/" + field] = f"physical_evidence/resources/{resource_id}/{field}"
        configuration = physical_evidence.get("resource_models", {}).get(resource_id, {}).get("configuration")
        if (isinstance(configuration, dict) and physical.get("controlled_link") and physical.get("tcp_link")
                and "tcp_transform" not in result.get("motion_config", {})):
            source_path = f"physical_evidence/resource_models/{resource_id}/configuration"
            try:
                result.setdefault("motion_config", {})["tcp_transform"] = _captured_tcp_transform(
                    physical, configuration, source_path=source_path)
                record_sources["motion_config/tcp_transform"] = source_path
            except (ValueError, KeyError, TypeError) as exc:
                result.setdefault("missing", []).append({"ref": source_path, "reason": str(exc)})
    return result


def captured_resource_snapshots(observations: dict) -> dict:
    """Read reconciled frozen snapshots without querying an owner or controller."""
    snapshots = {}
    for jid, evidence in observations.get("resource_evidence", {}).items():
        if evidence.get("conflicts"):
            raise ValueError("Conflicting recorded resource evidence: " + jid + ": " +
                             json.dumps(evidence["conflicts"], ensure_ascii=False))
        snapshot = evidence.get("snapshot")
        if isinstance(snapshot, dict):
            snapshots[jid] = deepcopy(snapshot)
    return snapshots


def derive_recorded_observations(record: dict, *, source_path: str, source_sha256: str) -> dict:
    """Repair snapshot assembly from saved sources, retaining an immutable original.

    This does not supply missing measurements or certify the compatibility of
    failure-time processing with later physical observations.
    """
    result = deepcopy(record)
    changes = []
    for rid, jid in result.get("jids", {}).items():
        evidence = result.get("resource_evidence", {}).get(jid)
        if not isinstance(evidence, dict):
            continue
        if evidence.get("snapshot_reconciliation_version") == 1:
            continue
        if evidence.get("source", {}).get("kind") != "recorded_owner_state":
            raise ValueError("NEEDS_CONTEXT: unsupported owner snapshot source for " + rid)
        owner = deepcopy(evidence.get("state", {})) if evidence.get("environment_models") else {}
        evidence["owner_snapshot"] = owner
        processing = result.get("processing_observations", {}).get(rid)
        if processing is not None:
            failure = result.get("failure", {})
            for path, existing in (("failure/processing_observations", failure.get("checkpoint_processing_observations", failure.get("processing_observations", {})).get(rid)),
                                   ("resource_evidence/processing_observations", evidence.get("processing_observations"))):
                if existing is None or fingerprint(existing) != fingerprint(processing):
                    raise ValueError("Conflicting recorded processing_observations: " + rid + ": " + path)
        corrected = reconcile_resource_snapshot(evidence, result.get("physical_evidence", {}),
            resource_id=rid, run_id=result["run_id"], revision=result["context_revision"], inputs=result["inputs"])
        if corrected["conflicts"]:
            raise ValueError("Conflicting recorded resource evidence: " + rid + ": " + json.dumps(corrected["conflicts"]))
        for field, value in corrected["snapshot"].items():
            previous = evidence.get("snapshot", {})
            if field not in previous or fingerprint(previous[field]) != fingerprint(value):
                changes.append({"path": f"resource_evidence/{jid}/snapshot/{field}",
                    "previous_present": field in previous, "previous": deepcopy(previous.get(field)),
                    "value": deepcopy(value), "source": deepcopy(corrected["snapshot_sources"].get(field,
                        {"ref": f"resource_evidence/{jid}/state/{field}"})),
                    "reason": "Rebuilt from recorded owner facts and measured physical fields"})
        result["resource_evidence"][jid] = corrected
        entry = result.get("prepared_recovery_request", {}).get("recovery_resources", {}).get(jid)
        if isinstance(entry, dict):
            entry["recovery_snapshot"] = {**deepcopy(entry.get("recovery_snapshot", {})), **deepcopy(corrected["snapshot"])}
            entry["resource_evidence"] = deepcopy(corrected)
    prepared = result.get("prepared_recovery_request", {})
    for field in ("multi_turn_session_state", "source_observation_fingerprint", "attempt_provenance"):
        prepared.pop(field, None)
    result.pop("source_path", None)
    result.pop("source_sha256", None)
    result["observation_repair"] = {"version": 1, "source_path": source_path, "source_sha256": source_sha256,
        "derived_at": datetime.now(timezone.utc).isoformat(), "changes": changes,
        "physical_observations_unchanged": True, "validation_results_reused": False}
    result["missing"] = offline_preparation_missing(result)
    result["offline_ready"] = not result["missing"]
    return result


def checkpoint_fact_fingerprint(observations: dict) -> str:
    """Bind a frozen checkpoint to its exact recorded owner and physical facts."""
    physical = observations.get("physical_evidence", {})
    facts = {"run_id": observations.get("run_id"), "context_revision": observations.get("context_revision"),
        "inputs": observations.get("inputs"), "jids": observations.get("jids"),
        "part_tracker": observations.get("part_tracker"), "resource_states": observations.get("resource_states"),
        "failure": {key: observations.get("failure", {}).get(key)
                    for key in ("run_id", "resource_id", "checkpoint", "part_name", "pending_tasks", "continuations", "requirements")},
        "processing_observations": observations.get("processing_observations"),
        "resource_evidence": observations.get("resource_evidence"),
        "physical": {key: physical.get(key) for key in ("snapshot", "resources", "parts", "resource_models", "geometry")},
        "planning_scene_capture": observations.get("planning_scene_capture")}
    if "checkpoint_motion_cancellation" in observations:
        facts["checkpoint_motion_cancellation"] = observations["checkpoint_motion_cancellation"]
    return fingerprint(facts)


def _checkpoint_root_motion(declaration: dict, description: str | None) -> dict:
    """Keep mobile roots exact and use declared stationary bounds for fixed roots."""
    result = {"root_link": declaration["root_link"], "fixed_to_world": False,
              "linear_speed_max": 0., "angular_speed_max": 0.}
    if description is None:
        return result
    try:
        robot = ET.fromstring(description)
    except ET.ParseError as exc:
        raise ValueError("Checkpoint root has an invalid physics robot description") from exc
    joints = robot.findall("joint")
    link_rows = robot.findall("link")
    links = {row.get("name") for row in link_rows}
    if (robot.tag != "robot" or len(links) != len(link_rows) or None in links
            or declaration["root_link"] not in links
            or any(row.find("parent") is None or row.find("child") is None
                   or row.find("parent").get("link") not in links
                   or row.find("child").get("link") not in links for row in joints)):
        raise ValueError("Checkpoint root needs a complete physics robot description")
    parents = {row.find("child").get("link"): row for row in joints}
    if len(parents) != len(joints) or None in parents:
        raise ValueError("Checkpoint root has ambiguous parent joints")
    result.update(description_sha256=hashlib.sha256(description.encode()).hexdigest(),
                  description_node=declaration.get("physics_description_node", declaration.get("description_node")))
    link, visited = declaration["root_link"], set()
    while link in parents:
        if link in visited:
            raise ValueError("Checkpoint root has a cyclic parent chain")
        visited.add(link)
        joint = parents[link]
        if joint.get("type") != "fixed":
            return result
        link = joint.find("parent").get("link")
    if link != "world":
        return result
    for field in ("linear_speed_max", "angular_speed_max"):
        value = declaration.get("stationary_" + field)
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError("Fixed checkpoint root lacks declared stationary_" + field)
        result[field] = value
    result["fixed_to_world"] = True
    return result


async def _cancel_checkpoint_motion(runtime, reader, directory: Path) -> dict:
    """Cancel owned handling motion without stopping or pausing machine jobs."""
    if (runtime.setup.get("execution_mode") != "simulation" or not runtime.stopped
            or not getattr(runtime, "recovery_checkpoint_capturing", False)):
        raise ValueError("Checkpoint motion cancellation requires blocked nominal dispatch")
    configuration = runtime.context.inputs["scene"]["safety_preparation"]
    declarations = configuration["resources"]
    owners = {owner.agent_name: owner for owner in runtime.resource_agents}
    moving = {rid: row for rid, row in declarations.items() if row.get("controller_node")}
    machines = {row["resource_id"] for row in runtime.context.inputs["scene"]["machines"]}
    if machines.intersection(moving):
        raise ValueError("Checkpoint handling cancellation cannot include a machine job owner")
    record = {"run_id": runtime.context.run_id, "context_revision": runtime.context.revision,
        "status": "cancelling", "resources": {}, "nominal_dispatch_blocked": True,
        "pending_tasks": deepcopy(runtime.context.pending_tasks), "machine_commands": []}
    path = directory / "checkpoint_motion_cancellation.json"
    write_record(path, record)
    try:
        descriptions = record["robot_descriptions"] = {}
        for rid in moving:
            owner = owners.get(rid)
            row = record["resources"][rid] = {"status": "cancelling"}
            declaration = moving[rid]
            node = declaration.get("physics_description_node", declaration.get("description_node"))
            if node and node not in descriptions:
                descriptions[node] = await asyncio.to_thread(reader.parameter, node, "robot_description")
            row["root_motion"] = _checkpoint_root_motion(declaration, descriptions.get(node))
            controller = getattr(owner, "_controller", None)
            cancel = getattr(controller, "_cancel_simulation_goal", None)
            worker = getattr(owner, "workflow_worker", None)
            if not callable(cancel) and worker is None:
                if any(task.get("resource_id") == rid for task in runtime.context.pending_tasks.values()):
                    raise ValueError("Pending checkpoint motion has no cancellation owner: " + rid)
                row["status"] = "no_owned_motion"
                continue
            if callable(cancel):
                await asyncio.to_thread(cancel)
            if worker is not None:
                await worker.cancel()
                row["worker_result"] = deepcopy(worker.last_result)
            row["status"] = "cancellation_requested"
            write_record(path, record)
        deadline = time.monotonic() + configuration["outline_preview"]["timeout_seconds"]
        while True:
            if getattr(runtime.part_slippage_recovery, "_stopped", False):
                raise asyncio.CancelledError()
            try:
                activity = await asyncio.to_thread(reader.idle_goals)
                native = await asyncio.to_thread(reader.snapshot, refresh=True)
                if native.get("world_paused") is not False:
                    raise ValueError("Handling stop must be observed before freezing simulation time")
                for rid, declaration in moving.items():
                    root = native["models"][declaration["model"]]["links"][declaration["root_link"]]
                    for field in ("linear_velocity", "angular_velocity"):
                        values = root.get(field)
                        if (not isinstance(values, list) or len(values) != 3
                                or any(type(value) not in (int, float) or not math.isfinite(value) for value in values)
                                or math.hypot(*values) > record["resources"][rid]["root_motion"][
                                    field.replace("velocity", "speed_max")]):
                            raise ValueError("Handling root has not stopped: " + rid + "/" + field)
            except (ValueError, TimeoutError) as exc:
                record["last_observation_error"] = str(exc)
                write_record(path, record)
                if time.monotonic() >= deadline:
                    raise
                await asyncio.sleep(.1)
                continue
            record.update(status="observed_stopped", controller_observations=activity,
                          native_observation=native)
            record.pop("last_observation_error", None)
            write_record(path, record)
            return record
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, TimeoutError) as exc:
        record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        write_record(path, record)
        raise
    except asyncio.CancelledError:
        record.update(status="cancelled")
        write_record(path, record)
        raise


async def capture_machine_checkpoint(runtime, failure: dict) -> dict:
    """Save stopped handling motion and current jobs before machine shutdown."""
    from cais_spade_llm.recovery_framework.gazebo_safety_preparation import GazeboSafetyReader, LiveSafetyPreparation
    from cais_spade_llm.recovery_framework.recovery_motion_preview import capture_preview_scene

    if runtime.setup.get("execution_mode") != "simulation":
        raise ValueError("Frozen checkpoint capture is restricted to simulation")
    configuration = runtime.context.inputs["scene"]["safety_preparation"]
    provider = getattr(runtime, "recovery_observation_preparation", None)
    if provider is None:
        provider = LiveSafetyPreparation(runtime, None, configuration)
        runtime.recovery_observation_preparation = provider
    if provider.reader is None:
        provider.reader = GazeboSafetyReader(configuration)
    reader = provider.reader
    directory = runtime.part_slippage_recovery.path
    identifier, revision = uuid4().hex, runtime.context.revision
    before = await asyncio.to_thread(reader.snapshot, refresh=True)
    if before.get("world_paused") is not False:
        raise ValueError("Capture requires the owned running simulation")
    try:
        write_record(directory / "checkpoint_before_motion_cancellation.json", {
            "failure": deepcopy(failure), "native_observation": before,
            "pending_tasks": deepcopy(runtime.context.pending_tasks),
            "run_id": runtime.context.run_id, "context_revision": revision})
        motion_cancellation = await _cancel_checkpoint_motion(runtime, reader, directory)
        if runtime.context.revision != revision or runtime.part_slippage_recovery._stopped:
            raise ValueError("Checkpoint was replaced or stopped during motion cancellation")
        runtime.recovery_capture_frozen = True
        frozen = await asyncio.to_thread(reader.set_world_paused, True)
        runtime.recovery_capture_native = deepcopy(frozen)
        write_record(directory / "checkpoint_native_before.json", frozen)
        await asyncio.to_thread(provider.initialize, model_execution=False)
        jobs, replies = {}, {}
        owners = {owner.agent_name: owner for owner in runtime.resource_agents}
        for declaration in runtime.context.inputs["scene"]["machines"]:
            rid = declaration["resource_id"]
            reply = await owners[rid].workflow_worker.observe_machine_checkpoint(
                run_id=runtime.context.run_id, resource_id=rid, checkpoint_id=identifier)
            replies[rid] = reply
            write_record(directory / "checkpoint_machine_observations.json", replies)
            source = reply.get("native_source", {})
            if (reply.get("status") != "observed" or reply.get("checkpoint_id") != identifier
                    or reply.get("command_sent") is not False
                    or any(source.get(key) != frozen.get(key) for key in ("instance_id", "simulation_time", "world_paused"))):
                raise ValueError("Machine observation is not bound to the frozen checkpoint: " + rid)
            record = reply["processing_observations"]
            recorded_processing_fields(record, resource_id=rid, run_id=runtime.context.run_id,
                                       inputs=runtime.context.inputs)
            jobs[rid] = deepcopy(record)
        failure["checkpoint_processing_observations"] = jobs
        captured = await asyncio.to_thread(capture_observations, runtime, failure)
        captured["failure_processing_observations"] = deepcopy(failure.get("processing_observations", {}))
        captured["checkpoint_motion_cancellation"] = motion_cancellation
        write_record(directory / "checkpoint_observations_raw.json", captured)
        settings = configuration["outline_preview"]
        scene = await capture_preview_scene(execution_mode="simulation",
            planning_scene_service=settings["planning_scene_service"],
            preview_parameter_service=settings["preview_parameter_service"], timeout_seconds=settings["timeout_seconds"])
        captured["planning_scene_capture"] = {**scene, "status": "captured", "compatibility": "not_checked",
            "physical_evidence_fingerprint": captured["physical_evidence"].get("evidence_fingerprint")}
        after = await asyncio.to_thread(reader.snapshot, refresh=True)
        write_record(directory / "checkpoint_native_after.json", after)
        same_native = fingerprint({k: v for k, v in frozen.items() if k != "observed_monotonic"}) == fingerprint(
            {k: v for k, v in after.items() if k != "observed_monotonic"})
        checkpoint = {"version": 1, "checkpoint_id": identifier, "run_id": runtime.context.run_id,
            "context_revision": revision, "simulation_time": frozen["simulation_time"],
            "instance_id": frozen["instance_id"], "world_paused": after.get("world_paused"),
            "native_observations_unchanged": same_native,
            "fault_simulation_time": failure.get("actual_interruption_simulation_sec"),
            "machine_observations": replies, "facts_fingerprint": checkpoint_fact_fingerprint(captured)}
        captured["capture_checkpoint"] = checkpoint
        write_record(directory / "checkpoint_observations_raw.json", captured)
        if runtime.context.revision != revision or runtime.part_slippage_recovery._stopped:
            raise ValueError("Checkpoint was replaced or stopped during capture")
        compatibility = processing_capture_compatibility(captured)
        if compatibility["status"] != "supported":
            raise ValueError("NEEDS_CONTEXT: " + "; ".join(row["reason"] for row in compatibility["checks"] if row["status"] != "supported"))
        for rid, record in jobs.items():
            if rid == failure.get("resource_id"):
                continue
            # A robot fault can follow its machine's completed job for the affected part;
            # every other machine must still be processing when the world is frozen.
            if record.get("active_job") is None and any(
                    job.get("part_name") == failure.get("part_name") and job.get("process_completed") is True
                    for job in record.get("jobs", {}).values()):
                continue
            active = record.get("jobs", {}).get(record.get("active_job"), {})
            if active.get("job_state") != "running" or active.get("process_completed") is not False:
                raise ValueError("Processing overlap ended before frozen capture: " + rid)
        runtime.recovery_checkpoint_observations = captured
        return captured
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, TimeoutError) as exc:
        write_record(directory / "checkpoint_error.json", {"checkpoint_id": identifier,
            "error": f"{type(exc).__name__}: {exc}", "execution_authorized": False})
        raise


def processing_capture_compatibility(observations: dict) -> dict:
    """Keep failure-time job facts distinct from a later physical scene."""
    records = observations.get("processing_observations", {})
    if not records:
        return {"status": "not_required", "checks": []}
    physical = observations.get("physical_evidence", {})
    checkpoint = observations.get("capture_checkpoint")
    if checkpoint is not None:
        checks = []
        def check(condition, reason, **fields):
            checks.append({**fields, "status": "supported" if condition else "NEEDS_CONTEXT", "reason": reason})
        check(checkpoint.get("version") == 1 and bool(checkpoint.get("checkpoint_id"))
              and checkpoint.get("world_paused") is True and checkpoint.get("native_observations_unchanged") is True,
              "Frozen native checkpoint must remain unchanged during collection")
        check(checkpoint.get("run_id") == observations.get("run_id")
              and checkpoint.get("context_revision") == observations.get("context_revision")
              and checkpoint.get("facts_fingerprint") == checkpoint_fact_fingerprint(observations),
              "Recorded checkpoint identities, revisions and fact fingerprints must match")
        for rid, record in records.items():
            reply = checkpoint.get("machine_observations", {}).get(rid, {})
            source = reply.get("native_source", {})
            check(reply.get("status") == "observed" and reply.get("command_sent") is False
                  and reply.get("checkpoint_id") == checkpoint.get("checkpoint_id")
                  and reply.get("processing_observations") == record
                  and record.get("run_id") == observations.get("run_id") and record.get("resource_id") == rid
                  and record.get("simulation_clock_sec") == checkpoint.get("simulation_time")
                  and source.get("instance_id") == checkpoint.get("instance_id")
                  and source.get("simulation_time") == checkpoint.get("simulation_time") and source.get("world_paused") is True,
                  "Machine owner record must acknowledge the frozen checkpoint", resource_id=rid)
        check(set(physical.get("resources", {})) == set(observations.get("jids", {}))
              and all(row.get("simulation_time") == checkpoint.get("simulation_time")
                      and row.get("native_physics_source", {}).get("instance_id") == checkpoint.get("instance_id")
                      for row in physical.get("resources", {}).values()),
              "Every physical resource must belong to the frozen native observation")
        return {"status": "supported" if all(row["status"] == "supported" for row in checks) else "NEEDS_CONTEXT",
                "checks": checks, "integrity_errors": [deepcopy(checks[1])] if checks[1]["status"] != "supported" else [],
                "processing_shutdown_used": False, "execution_authorized": False}
    clocks = {rid: row.get("simulation_time") for rid, row in physical.get("resources", {}).items()}
    checks = []
    for rid, record in records.items():
        clock = record.get("simulation_clock_sec")
        same = (type(clock) in (int, float) and math.isfinite(clock) and bool(clocks)
                and all(type(value) in (int, float) and math.isfinite(value) and value == clock for value in clocks.values()))
        checks.append({"resource_id": rid, "processing_revision": record.get("revision"),
            "processing_simulation_time": clock, "physical_simulation_times": deepcopy(clocks),
            "status": "NEEDS_CONTEXT",
            "reason": ("Equal simulation times lack a verified checkpoint" if same else
                       "Failure-time processing and later physical observations lack recorded checkpoint compatibility")})
    return {"status": "supported" if all(row["status"] == "supported" for row in checks) else "NEEDS_CONTEXT",
            "checks": checks, "processing_shutdown_used": False, "execution_authorized": False}


def projection_capture_compatibility(*, physical: dict, task: dict,
                                     resource_evidence: dict, projection: dict | None = None) -> dict:
    """Check the processing facts used by this event and its selected APs.

    The capture-wide diagnostic is retained, but an unrelated job cannot veto
    a proposal. Time bounds are calculations, not observations or a substitute
    for a missing command history or a physical interval.
    """
    from cais_spade_llm.agents.central_controller.ppr_ap import parse_ap_record, physical_ap_kind
    from cais_spade_llm.resources.machine.machine_jobs import MachineJobs

    recorded = physical.get("processing_capture_compatibility") or {}
    if recorded.get("integrity_errors"):
        return {"status": "NEEDS_CONTEXT", "checks": deepcopy(recorded["integrity_errors"]),
                "capture_diagnostic": deepcopy(recorded), "processing_shutdown_used": False,
                "execution_authorized": False}
    if not recorded or recorded.get("status") in {"supported", "not_required"}:
        return {"status": recorded.get("status", "not_required"), "checks": [],
                "processing_shutdown_used": False, "execution_authorized": False}
    records = physical.get("processing_observations", {})
    rid = resource_evidence.get("resource_id")
    if not rid:
        matches = [name for name, jid in physical.get("jids", {}).items() if jid == task.get("resource_jid")]
        rid = matches[0] if len(matches) == 1 else None
    required = {rid: ["proposed resource uses its recorded job state"]} if rid in records else {}
    ap_fields: dict[str, set[str]] = {}
    specifications = physical.get("outline_safety_specifications")
    unresolved = []
    if specifications is None:
        unresolved.append({"ref": "/physical_evidence/outline_safety_specifications",
                           "reason": "Selected AP dependencies were not recorded"})
    for specification in specifications or []:
        for rule in specification["document"]["catalog"]["specifications"]:
            for ap in rule["aps"]:
                definition = parse_ap_record(ap)
                kind = physical_ap_kind(ap)
                product = definition["product"]
                if kind in {"process_result_completed", "process_target_completed"}:
                    for owner, record in records.items():
                        if any(product in {"any", "*", job.get("part_name")} or product.startswith("$")
                               for job in record.get("jobs", {}).values()):
                            required.setdefault(owner, []).append(rule["id"] + ": " + ap["label"])
                elif kind is None:
                    owners = (records if definition["resource"] in {"any", "*"}
                              or definition["resource"].startswith("$") else [definition["resource"]])
                    for owner in owners:
                        if owner in records:
                            required.setdefault(owner, []).append(rule["id"] + ": " + ap["label"])
                            field = (definition["state"]["symbol"].split("=", 1)[0]
                                     if definition["kind"] == "ap_state" and "=" in definition["state"]["symbol"]
                                     else "resource_state" if definition["kind"] == "ap_state" else "task_history")
                            ap_fields.setdefault(owner, set()).add(field)
    observations = (projection or {}).get("observations") or []
    duration = None
    if observations:
        from fractions import Fraction

        duration = Fraction(str(observations[-1]["time"])) - Fraction(str(observations[0]["time"]))
        if duration < 0:
            raise ValueError("Projected observation time moved backwards")
        if duration:
            for owner, record in records.items():
                if record.get("active_job") is not None and owner not in physical.get("unavailable_resources", []):
                    required.setdefault(owner, []).append("processing over the proposed interval")
    checks = []
    clocks = {name: row.get("simulation_time") for name, row in physical.get("resources", {}).items()}
    for owner, reasons in sorted(required.items()):
        record = records[owner]
        fields = {"process_completed"} | ap_fields.get(owner, set())
        if owner == rid:
            fields.update({"active_job", "job_state"})
            requested = set(task.get("expected_start_state", {})) | set(task.get("expected_end_state", {}))
            changed = set((projection or {}).get("resource_updates", {}))
            if "selected_program" in requested | changed:
                fields.add("selected_program")
        if duration:
            fields.add("processing_active_sec")
        check = {"resource_id": owner, "required_fields": sorted(fields), "inclusion_reasons": reasons,
                 "sources": [f"/physical_evidence/processing_observations/{owner}"],
                 "processing_revision": record.get("revision"),
                 "processing_simulation_time": record.get("simulation_clock_sec"),
                 "physical_simulation_times": deepcopy(clocks), "status": "NEEDS_CONTEXT"}
        # The upper bound proves only that this recorded job cannot yet have
        # completed. It cannot establish unchanged control/configuration state.
        machine = next((row for row in resource_evidence.get("inputs", {}).get("scene", {}).get("machines", [])
                        if row.get("resource_id") == owner), None)
        calculations = []
        if machine is not None:
            run_id = physical.get("provenance", {}).get("run_id", resource_evidence.get("source", {}).get("run_id"))
            MachineJobs(machine, record["run_id"] if run_id is None else run_id, record)
            for identity, job in record.get("jobs", {}).items():
                start = job.get("simulation_clock_sec")
                active, total = job.get("processing_active_sec"), job.get("processing_time_sec")
                if (job.get("job_state") == "running" and all(type(value) in (int, float)
                        and math.isfinite(value) for value in (start, active, total)) and clocks
                        and all(type(value) in (int, float) and math.isfinite(value) and value >= start
                                for value in clocks.values())):
                    end = max(clocks.values())
                    from fractions import Fraction

                    upper = Fraction(str(active)) + Fraction(str(end)) - Fraction(str(start))
                    calculations.append({"job_id": identity, "kind": "calculated_bound",
                        "covered_interval": [start, end],
                        "processing_active_sec": [active, min(total, math.nextafter(float(upper), math.inf))],
                        "completion_possible": upper >= Fraction(str(total)),
                        "model_fingerprint": record["model_fingerprint"],
                        "dependencies": [f"/processing_observations/{owner}/jobs/{identity}",
                                         "/physical_evidence/resources"],
                        "scope": "Accumulated active time at most elapsed simulation time; control state is unresolved"})
        check["calculations"] = calculations
        missing_fields = sorted(fields - ({"process_completed"} if calculations
            and all(row["completion_possible"] is False for row in calculations) else set()))
        if not missing_fields:
            check.update(status="supported", reason="Declared processing duration excludes completion within this interval")
        else:
            check.update(reason="Recorded evidence does not establish " + ", ".join(missing_fields)
                         + " with the required physical observations",
                         missing_fields=missing_fields,
                         context_requests=[f"/physical_evidence/processing_observations/{owner}"])
        checks.append(check)
    checks.extend({**row, "status": "NEEDS_CONTEXT"} for row in unresolved)
    return {"status": "NEEDS_CONTEXT" if any(row["status"] != "supported" for row in checks) else "supported",
            "checks": checks, "required_resources": sorted(required),
            "capture_diagnostic": deepcopy(recorded), "processing_shutdown_used": False,
            "execution_authorized": False}


def recorded_capability_evidence(observations: dict) -> dict:
    """Extract measured arrivals and their recorded checks, never a recovery route.

    Historical records retain their own scene and payload scope. They are not
    parameter witnesses for a new proposal or evidence of hardware execution.
    """
    result = {}
    for index, transition in enumerate(observations.get("transitions", [])):
        acknowledgement = transition.get("acknowledgement") or {}
        execution = transition.get("observations") or {}
        observed = execution.get("observations") or {}
        rid, task = acknowledgement.get("resource_id"), acknowledgement.get("task_id")
        location = observed.get("resource_location")
        if not isinstance(rid, str) or not isinstance(location, str):
            continue
        ref = f"/transitions/{index}"
        problems = []
        for field, expected in (("resource_id", rid), ("task_id", task)):
            if expected is None or execution.get(field) != expected:
                problems.append(field)
        if acknowledgement.get("run_id") != observations.get("run_id"):
            problems.append("run_id")
        revision = acknowledgement.get("revision")
        if (type(revision) is not int or type(observations.get("context_revision")) is not int
                or revision > observations["context_revision"]):
            problems.append("revision")
        after = transition.get("after", {}).get(rid, {})
        if after.get("resource_location") != location:
            problems.append("resource_location")
        measurements, checks = [], []
        for step_index, step in enumerate(observed.get("primitive_results", [])):
            outcome = step.get("result") or {}
            if outcome.get("resource_location") != location:
                continue
            if step.get("resource_id") != rid or step.get("task_id") != task:
                problems.append("measurement identity")
                continue
            poses = {key: deepcopy(outcome[key]) for key in ("base_pose", "current_pose")
                     if isinstance(outcome.get(key), list) and len(outcome[key]) in (3, 7)
                     and all(type(value) in (int, float) and math.isfinite(value) for value in outcome[key])}
            if poses and step.get("status") == "completed":
                measurements.append({"ref": ref + f"/observations/observations/primitive_results/{step_index}/result", **poses})
            for check_index, check in enumerate(step.get("observations", [])):
                if check.get("operation") not in {
                    "validated_transport_sweep", "validated_empty_transport_route", "observed_transport_custody"
                }:
                    continue
                checks.append({key: deepcopy(check[key]) for key in (
                    "operation", "success", "checked_states", "checked_samples", "sample_step_m",
                    "collision_checks_bypassed", "allowed_relative_error_m", "relative_error_m"
                ) if key in check} | {"ref": ref + f"/observations/observations/primitive_results/{step_index}/observations/{check_index}"})
        completed = acknowledgement.get("status") == execution.get("status") == "completed"
        record = {"event_name": acknowledgement.get("event_name"), "resource_location": location,
                  "recorded_execution": bool(completed and measurements and not problems),
                  "geometric_validation": [row["operation"] for row in checks
                      if not problems and row.get("success") is True and row.get("collision_checks_bypassed") is not True],
                  "ref": ref, "run_id": acknowledgement.get("run_id"), "revision": revision,
                  "source": observations.get("evidence_source"), "current_proposal_validated": False,
                  "measurements": measurements, "checks": checks,
                  "source_fingerprint": fingerprint(transition),
                  "scope": {key: deepcopy(observed[key]) for key in (
                      "part_name", "scene_fingerprint", "scene_asset_fingerprint", "launch_id",
                      "attached", "grasp_transform", "carrying_arm_configuration"
                  ) if key in observed}}
        if "held_part" in after:
            record["held_part"] = deepcopy(after["held_part"])
        if problems:
            record["conflicts"] = sorted(set(problems))
        if not measurements:
            record["missing"] = ["measured arrival"]
        result.setdefault(rid, []).append(record)
    return result


@lru_cache(maxsize=1)
def _observed_pose_agreement() -> tuple[float, float]:
    """Read the configured bounds within which two recorded part poses agree."""
    configuration = json.loads(SCENE_PATH.read_text(encoding="utf-8"))
    agreement = configuration.get("safety_preparation", {}).get("observed_pose_agreement") or {}
    return float(agreement.get("position_m", 0.0)), float(agreement.get("orientation_rad", 0.0))


def _same_recorded_pose(first: dict, second: dict) -> bool:
    """Compare two recorded reads of one part within the configured agreement bounds.

    Separate reads of a resting part differ by simulator noise; any larger
    difference remains a conflict. Without configured bounds, poses must match.
    """
    if fingerprint(first) == fingerprint(second):
        return True
    position_m, orientation_rad = _observed_pose_agreement()
    try:
        distance = math.dist([first[axis] for axis in "xyz"], [second[axis] for axis in "xyz"])
        dot = abs(sum(first[key] * second[key] for key in ("qx", "qy", "qz", "qw")))
    except (KeyError, TypeError):
        return False
    return distance <= position_m and 2 * math.acos(min(1.0, dot)) <= orientation_rad


def restore_recorded_public_facts(prepared: dict, observations: dict) -> dict:
    """Expose saved measurements and requirements in a new prepared attempt.

    Source observations remain unchanged. Physical measurements are distinct
    from symbolic locations and from later accepted projections.
    """
    sources = {}
    for rid, records in recorded_capability_evidence(observations).items():
        jid = observations.get("jids", {}).get(rid)
        if jid in prepared.get("recovery_resources", {}):
            prepared["recovery_resources"][jid]["capability_evidence"] = deepcopy(records)
    physical = observations.get("physical_evidence", {})
    for name, record in physical.get("parts", {}).items():
        pose = record.get("pose")
        if pose is None:
            continue
        model_name = physical.get("snapshot", {}).get("parts", {}).get(name, {}).get("model_name", name)
        if (record.get("name") != model_name or record.get("frame") != physical.get("snapshot", {}).get("frame")
                or not isinstance(pose, list) or len(pose) != 7
                or any(type(value) not in (int, float) or not math.isfinite(value) for value in pose)):
            raise ValueError("NEEDS_CONTEXT: invalid recorded part observation for " + name)
        measured = dict(zip(("x", "y", "z", "qx", "qy", "qz", "qw"), pose, strict=True))
        rows = [prepared.setdefault("part_tracker", {}).setdefault(name, {}),
                prepared.setdefault("grounding_context", {}).setdefault("parts", {}).setdefault(name, {})]
        rows.extend(row for row in prepared.get("llm_input", {}).get("part_facts", [])
                    if row.get("part_name") == name)
        for row in rows:
            existing = row.get("observed_pose")
            if existing is not None and not _same_recorded_pose(existing, measured):
                raise ValueError("NEEDS_CONTEXT: conflicting recorded observed_pose for " + name)
        source = {key: deepcopy(record[key]) for key in
                  ("source", "frame", "instance_id", "simulation_time", "observed_monotonic") if key in record}
        source.update(ref=f"/physical_evidence/parts/{name}/pose", run_id=observations.get("run_id"),
                      revision=observations.get("context_revision"), captured_at=observations.get("captured_at"))
        for row in rows:
            row["observed_pose"] = deepcopy(measured)
            row["observed_pose_source"] = deepcopy(source)
            row["observed_pose_frame"] = record["frame"]
        sources[f"/parts/{name}/observed_pose"] = source
    # Product requirements are goals, not a saved nominal route or a recovery
    # instruction. Retain only the existing public requirement vocabulary.
    allowed = {"processesToComplete", "process", "result", "target", "state", "location"}

    def public_requirement(value):
        if isinstance(value, dict):
            return {key: public_requirement(item) for key, item in value.items() if key in allowed}
        if isinstance(value, list):
            return [public_requirement(item) for item in value]
        return deepcopy(value)

    if observations.get("requirements"):
        prepared.setdefault("llm_input", {})["recorded_product_requirements"] = {
            name: public_requirement(rows) for name, rows in observations["requirements"].items()}
        sources["/requirements"] = {"ref": "/requirements", "run_id": observations.get("run_id"),
                                    "revision": observations.get("context_revision")}
    prepared["recorded_public_fact_sources"] = deepcopy(sources)
    return sources


def restore_recorded_resource_geometry(physical: dict, inputs: dict) -> dict:
    """Recover omitted collision observations from the same saved native record.

    These are endpoint observations. Missing joint models, controller holds,
    tool transforms and motion coverage remain missing.
    """
    from cais_spade_llm.recovery_framework.gazebo_safety_preparation import _links, _observed_attachment

    result = deepcopy(physical)
    native = physical.get("physics_evidence") or {}
    configurations = inputs.get("scene", {}).get("safety_preparation", {}).get("resources", {})
    supplements = result.setdefault("recorded_geometry_supplements", {})
    if not native:
        return result
    for rid in sorted(set(physical.get("environment_models", {})) - set(physical.get("snapshot", {}).get("resources", {}))):
        sources = ["/physical_evidence/physics_evidence", f"/inputs/scene/safety_preparation/resources/{rid}"]
        supplement = {"sources": sources, "native_fingerprint": fingerprint(native),
                      "instance_id": native.get("instance_id"), "simulation_time": native.get("simulation_time"),
                      "coverage": "endpoint", "motion_interval_supported": False}
        try:
            if rid in physical.get("invalidated_geometry", {}).get("resources", {}):
                raise ValueError("Captured geometry was invalidated by an accepted event; supported projection is required")
            if native.get("attachment_complete") is not True or not native.get("instance_id"):
                raise ValueError("Complete native attachment identity is required")
            config = configurations[rid]
            links = _links(native, config)
            bounds = [{**deepcopy(collision), "link": name} for name, row in links.items()
                      for collision in row["collisions"]]
            if not bounds or any(not row.get("bounds") for row in bounds):
                raise ValueError("Complete native collision bounds are unavailable")
            state = deepcopy(physical["environment_valuation"][rid])
            previous = physical.get("resources", {}).get(rid)
            if previous:
                supplement["previous_observation_fingerprint"] = fingerprint(previous)
                identity = previous.get("native_physics_source", previous.get("attachment", {}))
                if (identity.get("instance_id") == native["instance_id"]
                        and previous.get("simulation_time") == native["simulation_time"]
                        and "component_bounds" in previous and previous["component_bounds"] != bounds):
                    raise ValueError("Conflicting collision measurements for the same native observation")
            model = native["models"][config["model"]]
            pose = links.get(config.get("reference_link"), {}).get("pose")
            if state.get("held_part") is not None and pose is None:
                raise ValueError("Held part requires the recorded tool transform")
            attachment = _observed_attachment(native, config, links, state, pose,
                physical.get("products", {}), inputs.get("geometry", {}).get("parts", {}))
            if pose is not None:
                state["current_pose"] = deepcopy(pose)
            measured = {"source": "saved GETRECOVERYSTATE", "frame": "world", "model_static": model["static"],
                "simulation_time": native["simulation_time"], "observed_monotonic": native["observed_monotonic"],
                "attachment": attachment, "custody_complete": True, "observation_state": deepcopy(state),
                "component_bounds": bounds, "link_states": deepcopy(links),
                "native_physics_source": {key: native[key] for key in ("instance_id", "simulation_time", "observed_monotonic")}}
            result.setdefault("resources", {})[rid] = measured
            result["snapshot"]["resources"][rid] = state
            geometry = {"frame": "world", "component_bounds": deepcopy(bounds), "source": deepcopy(supplement)}
            if "held_part" not in state:
                geometry["stationary_only"] = model["static"] is True
            result["geometry"]["resources"][rid] = geometry
            supplement.update(status="recorded", missing=["resource-owned joint model", "controller hold interval"])
        except (KeyError, ValueError, TypeError) as exc:
            supplement.update(status="NEEDS_CONTEXT", reason=str(exc))
        supplements[rid] = supplement
    return result


def observation_summary(record: dict) -> dict:
    """Keep page polling small; complete context remains in the original file."""
    summary = {key: deepcopy(record.get(key)) for key in (
        "run_id", "captured_at", "evidence_source", "scenario", "part_tracker", "resource_states",
        "offline_ready", "missing", "safety_status", "robot_poses",
    )}
    summary["failure"] = {key: deepcopy(record.get("failure", {}).get(key)) for key in (
        "checkpoint", "injection_status", "pending_tasks", "continuations", "resource_id", "part_name",
    )}
    summary["missing"] = offline_preparation_missing(record)
    summary["offline_ready"] = not summary["missing"]
    return summary


def offline_preparation_missing(record: dict) -> list[str]:
    """Check proposal inputs; physical compatibility remains a validation result."""
    missing = []
    from cais_spade_llm.ui.recovery_setup import FAILURE_SCENARIOS

    if record.get("scenario") not in FAILURE_SCENARIOS:
        missing.append("scenario must identify an existing failure scenario")
    if (record.get("failure") or {}).get("injection_status") != "completed":
        missing.append("failure.injection_status must record completed reconciliation")
    prepared = record.get("prepared_recovery_request")
    if not isinstance(prepared, dict) or not prepared:
        return [*missing, "prepared_recovery_request"]
    for name in ("llm_input", "recovery_session", "recovery_debug"):
        if not isinstance(prepared.get(name), dict):
            missing.append(f"prepared_recovery_request.{name}")
    resources = prepared.get("recovery_resources")
    if not isinstance(resources, dict) or not resources:
        missing.append("prepared_recovery_request.recovery_resources")
    else:
        from cais_spade_llm.recovery_framework.offline_recovery import SavedRecoveryResource

        for jid, entry in resources.items():
            if not isinstance(entry, dict):
                missing.append(f"recovery_resources[{jid}] must contain recorded resource metadata")
                continue
            evidence = record.get("resource_evidence", {}).get(jid, {})
            if evidence.get("conflicts"):
                missing.append(f"resource_evidence[{jid}].conflicts: " + json.dumps(evidence["conflicts"], ensure_ascii=False))
            if isinstance(entry.get("recovery_snapshot"), dict) and entry["recovery_snapshot"]:
                try:
                    owner = SavedRecoveryResource(jid, entry, record, _copy_observations=False)
                    owner.get_recovery_snapshot()
                except (ValueError, KeyError, TypeError) as exc:
                    missing.append(str(exc))
        if prepared.get("ra_jid") not in resources:
            missing.append("prepared_recovery_request.ra_jid must identify a recorded resource")
    if prepared.get("loaded_safety_rules"):
        missing.append("saved preparation contains active safety rules; this example requires safety checks deferred")
    return missing


def capture_observations(runtime, failure: dict, *, include_physical: bool = True) -> dict:
    """Copy reconciled runtime facts before any potentially failing preparation.

    Args:
        runtime: Runtime owning the selected simulation failure.
        failure: Reconciled failure observations and retained task evidence.
        include_physical: Whether to collect the current physical observations.

    Returns:
        Saved resource, product, failure, and safety observations.
    """
    context = runtime.context
    configuration = runtime.setup["failure_scenario"]
    robot_poses = {}
    for key, rid in (("robot_pose", configuration.get("resource_id")),
                     ("other_robot_pose", (configuration.get("additional_condition") or {}).get("resource_id"))):
        if rid and isinstance(failure.get(key), dict):
            robot_poses[rid] = deepcopy(failure[key])
    result = {
        "schema_version": RAW_SCHEMA_VERSION,
        "run_id": context.run_id,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "evidence_source": "Gazebo",
        "scenario": runtime.setup["failure_scenario"]["scenario"],
        "setup": deepcopy(runtime.setup),
        "failure": deepcopy(failure),
        "part_tracker": deepcopy(context.part_tracker),
        "initial_product_states": deepcopy(context.initial_product_states),
        "resource_states": {rid: deepcopy(actor.valuation) for rid, actor in context.resources.items()},
        "robot_poses": robot_poses,
        "inputs": deepcopy(context.inputs),
        "jids": deepcopy(runtime.jids),
        "context_revision": context.revision,
        "program_revision": getattr(runtime, "program_revision", ""),
        "requirements": deepcopy(getattr(context, "requirements", {})),
        "unavailable_resources": sorted(getattr(context, "unavailable_resources", set())),
        "fault_reports": deepcopy(getattr(context, "fault_reports", {})),
        "config_references": config_references(runtime.setup),
        "nominal_permitted_resources": deepcopy(getattr(context, "permitted_resources", runtime.setup.get("permitted_resources", []))),
        "recovery_permitted_resources": deepcopy(runtime.setup.get("recovery_permitted_resources", runtime.setup.get("permitted_resources", []))),
        "acknowledgements": deepcopy(getattr(context, "acknowledgements", [])),
        "transitions": deepcopy(getattr(context, "transitions", [])),
        "offline_ready": False,
        "missing": ["prepared_recovery_request"],
        "safety_status": "skipped",
    }
    result["physical_evidence"] = (capture_recovery_physical_evidence(runtime) if include_physical else
                                   {"status": "pending", "reason": "Physical capture awaits nominal_idle"})
    result["resource_evidence"] = capture_resource_evidence(runtime, result["physical_evidence"])
    if "processing_observations" in failure and (configuration["scenario"] != "ur5e-1 breakdown"
                                                 or "checkpoint_processing_observations" in failure):
        result["processing_observations"] = deepcopy(failure.get("checkpoint_processing_observations", failure["processing_observations"]))
        result["processing_shutdown"] = deepcopy(failure.get("processing_shutdown", {}))
        for rid, record in result["processing_observations"].items():
            jid = result["jids"].get(rid)
            if jid not in result["resource_evidence"]:
                continue
            owner = result["resource_evidence"][jid]
            previous = owner.get("processing_observations")
            if previous is not None and fingerprint(previous) != fingerprint(record):
                owner.setdefault("conflicts", []).append({"field": "processing_observations",
                    "reason": "Captured owner and failure-time processing records differ",
                    "owner_record": deepcopy(previous), "failure_record": deepcopy(record)})
                continue
            owner["processing_observations"] = deepcopy(record)
            try:
                fields = recorded_processing_fields(record, resource_id=rid, run_id=context.run_id, inputs=context.inputs,
                                                    values=owner["environment_valuation"][rid])
            except (ValueError, KeyError, TypeError) as exc:
                owner.setdefault("conflicts", []).append({"field": "processing_observations", "reason": str(exc)})
                continue
            for key in ("snapshot", "state"):
                owner.setdefault(key, {}).update(deepcopy(fields))
            owner["environment_valuation"][rid].update(deepcopy(fields))
            result["resource_states"][rid].update(deepcopy(fields))
            owner.setdefault("sources", {})["processing_observations"] = "failure/processing_observations/" + rid
    admission = getattr(runtime, "admission", None)
    monitor = getattr(admission, "monitor", None)
    physical = getattr(monitor, "physical_monitor", None)
    result["monitor_history"] = {
        "active": monitor is not None,
        "current_states": deepcopy(getattr(monitor, "current_states", {})),
        "history_error": deepcopy(getattr(monitor, "history_error", None)),
        "acknowledgement_history": deepcopy(getattr(admission, "acknowledgement_history", [])),
        "physical_monitor": {
            "rules": deepcopy(getattr(physical, "rules", None)),
            "states": {key: sorted(values) for key, values in getattr(physical, "states", {}).items()},
            "revision": getattr(physical, "revision", None),
            "history": deepcopy(getattr(physical, "history", [])),
            "invalid_reason": getattr(physical, "invalid_reason", ""),
        },
    }
    return result


def _captured_tcp_transform(physical: dict, configuration: dict, *, source_path: str) -> dict:
    """Calculate an exact fixed-link transform from captured owner declarations."""
    from cais_spade_llm.recovery_framework.geometry import compose, quaternion, rotate

    parent, child = physical.get("controlled_link"), physical.get("tcp_link")
    if not parent or not child or not configuration.get("description_sha256"):
        raise ValueError("Captured controlled link, TCP link and robot description fingerprint are required")
    graph = {}
    for joint in configuration.get("joints", []):
        if joint.get("type") != "fixed":
            continue
        xyz, rpy = joint.get("xyz"), joint.get("rpy")
        if any(not isinstance(value, list) or len(value) != 3
               or any(type(item) not in (int, float) or not math.isfinite(item) for item in value)
               for value in (xyz, rpy)):
            raise ValueError("Captured fixed joint origin is incomplete")
        pose = [*xyz, *quaternion(rpy)]
        inverse = [-pose[3], -pose[4], -pose[5], pose[6]]
        reverse = [*rotate(inverse, [-value for value in pose[:3]]), *inverse]
        graph.setdefault(joint["parent"], []).append((joint["child"], pose, joint))
        graph.setdefault(joint["child"], []).append((joint["parent"], reverse, joint))
    queue, seen = [(parent, [0., 0., 0., 0., 0., 0., 1.], [])], {parent}
    for frame, pose, path in queue:
        if frame == child:
            return {"translation": pose[:3], "rotation": pose[3:], "parent_frame": parent,
                    "child_frame": child, "fixed": True,
                    "source": {"kind": "captured fixed joint declarations", "ref": source_path,
                               "description_sha256": configuration["description_sha256"],
                               "configuration_fingerprint": fingerprint(configuration),
                               "joints": deepcopy(path)}}
        for target, transform, joint in graph.get(frame, []):
            if target not in seen:
                seen.add(target)
                queue.append((target, compose(pose, transform), [*path, joint]))
    raise ValueError("Captured controlled-link to TCP chain is not fully declared fixed")


def capture_resource_evidence(runtime, physical_evidence: dict) -> dict:
    """Copy owner configuration and state without dispatch or placeholder values.

    These records preserve calculation inputs independently of prompt selection.
    Observation records supply attachment evidence; cached owner state remains
    explicitly identified and cannot establish a future motion outcome.
    """
    context = runtime.context
    captured_at = datetime.now(timezone.utc).isoformat()
    result = {}
    for owner in getattr(runtime, "resource_agents", []):
        rid, jid = owner.agent_name, str(owner.jid)
        physical = physical_evidence.get("resources", {}).get(rid, {})
        state = {key: deepcopy(getattr(owner, key)) for key in ("_task_ctx", "_held_part", "_position", "_current_state", "_primitive_state", "_primitive_evidence")
                 if hasattr(owner, key)}
        robot_ids = {row["resource_id"] for row in context.inputs.get("scene", {}).get("robots", [])}
        owner_evidence = {}
        if rid in robot_ids:
            from cais_spade_llm.resources.robot.robot_profile import _robot_snapshot_builder

            pose = physical_evidence.get("snapshot", {}).get("resources", {}).get(rid, {}).get("current_pose")
            if isinstance(pose, list) and len(pose) == 7:
                pose = dict(zip(("x", "y", "z", "qx", "qy", "qz", "qw"), pose, strict=True))
            owner_evidence["snapshot"] = _robot_snapshot_builder(owner, recorded_pose=pose if isinstance(pose, dict) else {})
        getter = getattr(owner, "get_recovery_resource_evidence", None)
        if rid not in robot_ids and callable(getter):
            try:
                owner_evidence = getter()
                if not isinstance(owner_evidence, dict):
                    raise ValueError(f"{rid} recovery evidence must be an object")
            except (ValueError, KeyError, TypeError) as exc:
                owner_evidence = {"conflicts": [{"field": "owner_snapshot", "reason": str(exc)}]}
        if (rid not in robot_ids and "snapshot" not in owner_evidence and not owner_evidence.get("conflicts")
                and callable(getattr(owner, "get_recovery_snapshot", None))):
            try:
                snapshot = owner.get_recovery_snapshot()
                if not isinstance(snapshot, dict):
                    raise ValueError(f"{rid} recovery snapshot must be an object")
                owner_evidence["snapshot"] = deepcopy(snapshot)
            except (ValueError, KeyError, TypeError) as exc:
                owner_evidence = {"conflicts": [{"field": "owner_snapshot", "reason": str(exc)}]}
        record = {
            **deepcopy(owner_evidence),
            "state": {**deepcopy(owner_evidence.get("state", {})), **state},
            "snapshot": deepcopy(owner_evidence.get("snapshot", {})),
            "owner_snapshot": deepcopy(owner_evidence.get("snapshot", {})),
            "inputs": deepcopy(context.inputs),
            "observed_parts": deepcopy(physical_evidence.get("snapshot", {}).get("parts", {})),
            "source": {"run_id": context.run_id, "captured_at": captured_at,
                       "revision": context.revision, "kind": "recorded_owner_state"},
            "sources": {key: f"resource_agents/{jid}/{key}" for key in state},
            "conflicts": deepcopy(owner_evidence.get("conflicts", [])),
            "unavailable_resources": sorted(getattr(context, "unavailable_resources", set())),
        }
        if "execution_mode" in runtime.setup:
            record["execution_mode"] = runtime.setup["execution_mode"]
            record["sources"]["execution_mode"] = "setup/execution_mode"
        capabilities = getattr(owner, "static_capabilities", {})
        if isinstance(capabilities, dict) and "resource_type" in capabilities:
            record["resource_type"] = capabilities["resource_type"]
            record["sources"]["resource_type"] = f"resource_agents/{jid}/static_capabilities/resource_type"
        controller = getattr(owner, "_controller", None)
        configuration = getattr(controller, "controller_config", None)
        if isinstance(configuration, dict):
            record["controller_config"] = deepcopy(configuration)
            record["controller_config_fingerprint"] = fingerprint(configuration)
            record["sources"]["controller_config"] = f"resource_agents/{jid}/_controller/controller_config"
            record["motion_config"] = deepcopy(configuration.get("motion", {}))
            for field in ("pick_z_adjustments_m",):
                if field in configuration.get("parts_tuning", {}):
                    record["motion_config"][field] = deepcopy(configuration["parts_tuning"][field])
        pick_policy = getattr(controller, "recovery_pick_policy", None)
        if callable(pick_policy) and record.get("execution_mode") == "simulation":
            from cais_spade_llm.product.profile import ProductProfile

            record["pick_geometry"], record["pick_policy"] = {}, {}
            for part in record["observed_parts"]:
                geometry = ProductProfile.geometry_for_part_from_geometry(part, context.inputs.get("geometry"))
                grasp_geometry = getattr(controller, "recovery_grasp_geometry", None)
                if callable(grasp_geometry):
                    model_name = context.inputs.get("geometry", {}).get("parts", {}).get("model_map", {}).get(part)
                    if isinstance(model_name, str) and model_name:
                        calculated = grasp_geometry(model_name=model_name, product_geometry=geometry)
                        geometry = deepcopy(calculated["geometry"])
                        record.setdefault("pick_geometry_sources", {})[part] = deepcopy(calculated["source"])
                        identity = physical_evidence.get("parts", {}).get(part, {}).get("collision_scene_identity")
                        if isinstance(identity, dict):
                            record["pick_geometry_sources"][part]["recorded_model_identity"] = deepcopy(identity)
                if geometry:
                    record["pick_geometry"][part] = deepcopy(geometry)
                    record["sources"]["pick_geometry/" + part] = "inputs/geometry via ProductProfile.geometry_for_part_from_geometry"
                record["pick_policy"][part] = deepcopy(pick_policy(part))
                record["sources"]["pick_policy/" + part] = f"resource_agents/{jid}/_controller/recovery_pick_policy"
        command = getattr(controller, "_last_command_evidence", None)
        if isinstance(command, dict) and isinstance(command.get("payload_collision"), dict):
            record["payload_collision"] = deepcopy(command["payload_collision"])
            record["sources"]["payload_collision"] = f"resource_agents/{jid}/_controller/_last_command_evidence/payload_collision"
        record = reconcile_resource_snapshot(record, physical_evidence, resource_id=rid,
            run_id=context.run_id, revision=context.revision, inputs=context.inputs)
        try:
            json.dumps(record, allow_nan=False)
        except (TypeError, ValueError) as exc:
            record = {"source": record["source"], "conflicts": [f"Owner evidence cannot be serialized: {exc}"]}
        result[jid] = record
    return result


def _stationary_part_observation(name: str, part: dict, record: dict, evidence: dict,
                                 configuration: dict) -> dict:
    """Record supported stationary geometry without promoting sampled rest.

    A static model has fixed geometry. Dynamic parts require the recorded
    owner's static containment contract and an idle physics observation. A
    loose part has no interval coverage merely because its measured speed is
    zero; neither branch establishes a physical execution guarantee.
    """
    model = record["models"][part["name"]]
    result = {"status": "NEEDS_CONTEXT", "part_name": name, "model_name": part["name"],
              "current_pose": deepcopy(part["pose"]), "model_static": model.get("static"),
              "source": {"service": "GETRECOVERYSTATE", "instance_id": record["instance_id"],
                         "simulation_time": record["simulation_time"]},
              "future_execution_tracking": "not_established"}
    if type(model.get("static")) is not bool:
        return {**result, "reason": "Part static/dynamic physics state is unavailable"}
    held = [rid for rid, row in evidence["snapshot"]["resources"].items() if row.get("held_part") == name]
    if held:
        return {**result, "reason": "Carried geometry follows the recorded resource configuration", "holders": held}
    links = model.get("links")
    if not isinstance(links, dict) or not links:
        return {**result, "reason": "Part link motion observations are unavailable"}
    speeds = {link: {key: row.get(key) for key in ("linear_speed", "angular_speed")} for link, row in links.items()}
    result["link_speeds"] = deepcopy(speeds)
    if any(type(value) not in (int, float) or not math.isfinite(value) or value < 0
           for values in speeds.values() for value in values.values()):
        return {**result, "reason": "Part link motion observations are incomplete"}
    if model.get("static") is True:
        if any(value != 0 for values in speeds.values() for value in values.values()):
            return {**result, "reason": "Observed static part has conflicting motion"}
        return {**result, "status": "supported", "reason": "Observed static Gazebo model"}
    attachments = record.get("attachments")
    if record.get("attachment_complete") is not True or not isinstance(attachments, list):
        return {**result, "reason": "Complete recorded attachment graph is unavailable"}
    for row in attachments:
        if (not isinstance(row, dict) or any(not isinstance(row.get(field), str) or not row[field]
                for field in ("model1", "link1", "model2", "link2"))):
            return {**result, "reason": "Recorded attachment graph contains an incomplete endpoint"}
        for side in (1, 2):
            body = record["models"].get(row["model" + str(side)])
            endpoint_links = body.get("links") if isinstance(body, dict) else None
            if not isinstance(endpoint_links, dict) or not isinstance(endpoint_links.get(row["link" + str(side)]), dict):
                return {**result, "reason": "Recorded attachment graph contains an unresolved model or link"}
    from cais_spade_llm.resources.recovery_geometry import commanded_hold_evidence

    owner_evidence = {}
    for rid, row in evidence.get("resources", {}).items():
        if row.get("model_static") is True or row.get("idle") is True:
            owner_evidence[rid] = {"status": "supported", "model_static": row.get("model_static"),
                                   "observed_idle": row.get("idle")}
        else:
            owner_evidence[rid] = commanded_hold_evidence(resource_id=rid,
                state=evidence["snapshot"]["resources"].get(rid, {}),
                model=evidence.get("resource_models", {}).get(rid, {}), physical=row)
    owners_idle = bool(owner_evidence) and all(row["status"] == "supported" for row in owner_evidence.values())
    result["owner_evidence"] = owner_evidence
    if any(part["name"] in (row["model1"], row["model2"]) for row in attachments):
        from cais_spade_llm.resources.recovery_geometry import fixed_attached_part_observation

        if (part.get("instance_id") != record["instance_id"]
                or part.get("simulation_time") != record["simulation_time"]):
            return {**result, "reason": "Part pose and fixed attachment belong to different observations"}
        supported = fixed_attached_part_observation(model_name=part["name"], current_pose=part["pose"],
            physics=record, owners_idle=owners_idle)
        return {**result, **supported}
    if record.get("physics_engine") == "ode" and any(row.get("enabled") is False for row in links.values()):
        from cais_spade_llm.resources.recovery_geometry import disabled_part_observation

        supported = disabled_part_observation(model_name=part["name"], physics=record,
            owners_idle=owners_idle)
        return {**result, **supported, "current_pose": deepcopy(part["pose"])}
    receiver = evidence["snapshot"]["parts"][name].get("contained_by")
    owner = evidence.get("resources", {}).get(receiver, {})
    declaration = configuration.get("resources", {}).get(receiver, {})
    contract = declaration.get("stationary_contract")
    expected = {"kind": "static_body_and_idle_containment", "requires_no_running_tasks": True,
                "requires_no_active_goals": True, "future_execution_tracking": "not_established"}
    if (contract != expected or owner.get("stationary_contract") != contract or owner.get("model_static") is not True
            or owner.get("idle") is not True or name not in owner.get("observation_state", {}).get("contained_parts", [])):
        from cais_spade_llm.resources.recovery_geometry import gravity_supported_part

        support = gravity_supported_part(model_name=part["name"], physics=record,
            policy=configuration.get("part_support") or {},
            owners_idle=owners_idle)
        return {**result, **support, "current_pose": deepcopy(part["pose"])}
    limits = {key: declaration.get("stationary_" + key + "_max") for key in ("linear_speed", "angular_speed")}
    if any(type(value) not in (int, float) or not math.isfinite(value) or value < 0 for value in limits.values()):
        return {**result, "reason": "Owner stationary part thresholds are unavailable"}
    if any(value > limits[key] for values in speeds.values() for key, value in values.items()):
        return {**result, "reason": "Contained part is moving outside the owner's declared idle thresholds"}
    return {**result, "status": "supported", "contained_by": receiver, "contract": deepcopy(contract),
            "configuration_fingerprint": fingerprint(declaration), "limits": limits,
            "reason": "Observed idle part under its owner's declared static containment model"}


def _freeze_collision_mesh(uri: str) -> dict:
    """Save the exact observed mesh's vertices and faces for detached calculations."""
    from cais_spade_llm.recovery_framework.geometry import _mesh_shape

    if not isinstance(uri, str) or not uri:
        raise ValueError("Observed collision mesh URI is unavailable")
    if uri.startswith("model://"):
        path = ROOT / "ros2/cais_lab_robotics/models" / uri.removeprefix("model://")
        if not path.is_file():
            path = ROOT / "ros2/cais_lab_robotics" / uri.removeprefix("model://")
    elif uri.startswith("file://"):
        path = Path(uri.removeprefix("file://"))
    else:
        path = Path(uri)
    path = path.resolve()
    if not path.is_relative_to((ROOT / "ros2").resolve()) or not path.is_file():
        raise ValueError("Observed collision mesh must resolve to a recorded ROS2 workspace asset")
    stamp = path.stat()
    shape = _mesh_shape(path, stamp.st_mtime_ns, stamp.st_size, (1., 1., 1.))
    return {"uri": uri, "source": str(path), "source_sha256": shape["source_sha256"],
        "vertices": [[vertex[index] + shape["center"][index] for index in range(3)] for vertex in shape["vertices"]],
        "vertex_count": len(shape["vertices"]), "triangles": deepcopy(shape["triangles"]),
        "triangle_count": len(shape["triangles"]), "source_kind": "recorded_collision_mesh"}


def capture_recovery_physical_evidence(runtime, *, max_age: float = 2.) -> dict:
    """Save owner observations without preparing, admitting, or executing work.

    Each missing resource remains visible. This capture does not require a CCA
    monitor, so failure facts survive diagnostic runs and preparation errors.
    The caller must persist this evidence before starting prompt generation.
    """
    from cais_spade_llm.recovery_framework.gazebo_safety_preparation import (
        LiveSafetyPreparation, _check_custody, _part_local_bounds,
        _configured_static_part_geometry,
    )
    from cais_spade_llm.recovery_framework import fingerprint as native_fingerprint
    from cais_spade_llm.resources.resource_safety_preparation import observation_identity
    from cais_spade_llm.recovery_framework.part_collision import observed_collision_identities

    if type(max_age) not in (int, float) or not math.isfinite(max_age) or max_age <= 0:
        raise ValueError("Physical capture maximum age must be finite and positive")
    if getattr(getattr(runtime, "part_slippage_recovery", None), "offline", False):
        raise ValueError("Offline recovery cannot capture live physical observations")
    context = runtime.context
    revision = context.revision
    valuations = {rid: deepcopy(actor.valuation) for rid, actor in context.resources.items()}
    tracker = deepcopy(context.part_tracker)
    initial = deepcopy(getattr(context, "initial_product_states", {}))
    acknowledgements = deepcopy(getattr(context, "acknowledgements", {}))
    transitions = deepcopy(getattr(context, "transitions", []))
    started = time.monotonic()
    captured_at = datetime.now(timezone.utc).isoformat()
    result = {"version": 1, "snapshot": {"frame": "world", "resources": {}, "parts": {}},
              "resources": {}, "parts": {}, "resource_models": {}, "step_evidence": [],
              "stationary_parts": [], "stationary_part_evidence": {},
              "physics_meshes": {},
              "environment_models": {rid: deepcopy(actor.model) for rid, actor in context.resources.items()
                                     if isinstance(getattr(actor, "model", None), dict)},
              "environment_valuation": deepcopy(valuations), "products": deepcopy(tracker),
              "requirements": deepcopy(getattr(context, "requirements", {})),
              "product_name": context.inputs.get("scene", {}).get("Exit", {}).get("completed_product"),
              "part_support_policy": deepcopy(context.inputs.get("scene", {}).get(
                  "safety_preparation", {}).get("part_support", {})),
              "geometry": {"frame": "world", "resources": {}, "parts": {},
                           "regions": deepcopy(context.inputs.get("scene", {}).get(
                               "safety_preparation", {}).get("regions", {}))},
              "jids": deepcopy(runtime.jids), "unresolved": [],
              "provenance": {"source": "registered owner observations", "run_id": context.run_id,
                             "captured_at": captured_at, "context_revision": revision,
                             "captured_monotonic": started},
              "dispatch_authorized": False, "command_sent": False}
    owners = {owner.agent_name: owner for owner in getattr(runtime, "resource_agents", [])}
    configuration = context.inputs.get("scene", {}).get("safety_preparation")
    if (owners and isinstance(configuration, dict)
            and configuration.get("provider") == "gazebo_idle_continuous_v1"
            and any(getattr(owner, "recovery_safety_observer", None) is None for owner in owners.values())):
        try:
            if runtime.setup.get("execution_mode") != "simulation":
                raise ValueError("Gazebo observation preparation requires the selected simulation run")
            provider = getattr(runtime, "recovery_observation_preparation", None)
            if provider is None:
                if getattr(runtime, "live_safety_runtime", None) is not None:
                    raise ValueError("An active CCA owner must retain its registered observation provider")
                provider = LiveSafetyPreparation(runtime, None, configuration)
                runtime.recovery_observation_preparation = provider
            if not isinstance(provider, LiveSafetyPreparation) or provider.runtime is not runtime:
                raise ValueError("Observation preparation belongs to another runtime")
            provider.initialize(model_execution=False)
        except (ImportError, OSError, ValueError, KeyError, TypeError, AttributeError, RuntimeError) as exc:
            result["unresolved"].append({"reason": "Read-only observation preparation unavailable: " + str(exc)})
    hold_model = None
    hold_activity = None
    provider = getattr(runtime, "recovery_observation_preparation", None)
    if isinstance(provider, LiveSafetyPreparation) and provider.reader is not None:
        try:
            parameter = "trajectory_execution.allowed_start_tolerance"
            value = provider.reader.double_parameter("/move_group", parameter)
            if value < 0:
                raise ValueError("Recorded MoveIt start tolerance must be nonnegative")
            hold_model = {"value": value, "node": "/move_group", "parameter": parameter,
                "kind": "conditional model assumption", "captured_at": datetime.now(timezone.utc).isoformat(),
                "physical_execution_verified": False, "future_execution_tracking": "not_established"}
            hold_activity = provider.reader.controller_activity(recorded=True) if getattr(
                runtime, "recovery_capture_frozen", False) else provider.reader.idle_goals()
        except (ImportError, OSError, ValueError, KeyError, TypeError, AttributeError, RuntimeError) as exc:
            hold_model = {"status": "NEEDS_CONTEXT", "reason": str(exc)}
    started = time.monotonic()
    result["provenance"].update(captured_monotonic=started,
                                captured_at=datetime.now(timezone.utc).isoformat())
    frozen = getattr(runtime, "recovery_capture_native", None)
    frozen_fingerprint = native_fingerprint({key: value for key, value in frozen.items()
        if key != "observed_monotonic"}) if (getattr(runtime, "recovery_capture_frozen", False)
        and isinstance(frozen, dict) and frozen.get("world_paused") is True) else None
    for rid in valuations:
        owner = owners.get(rid)
        try:
            capture = getattr(owner, "capture_recovery_safety_state", None)
            if not callable(capture):
                raise ValueError("Resource-owned physical observation unavailable")
            physical = capture(max_age=max_age)
            frozen_observation = (frozen_fingerprint is not None
                and physical.get("frozen_native_fingerprint") == frozen_fingerprint)
            observed = physical.get("observed_monotonic")
            if (not frozen_observation and type(observed) in (int, float) and math.isfinite(observed)
                    and time.monotonic() - observed > max_age):
                physical = capture(max_age=max_age)
            observation_identity(physical)
            json.dumps(physical, allow_nan=False)
            result["resources"][rid] = deepcopy(physical)
            if not frozen_observation and not 0 <= time.monotonic() - physical["observed_monotonic"] <= max_age:
                raise ValueError("Stale resource observation")
            owner_observer = getattr(owner, "recovery_safety_observer", None)
            controller_node = getattr(owner_observer, "configuration", {}).get("controller_node")
            if controller_node:
                try:
                    reader = owner_observer.provider.reader
                    controller_state = (reader.controller_state(controller_node, recorded=True)
                        if getattr(runtime, "recovery_capture_frozen", False) else reader.controller_state(controller_node))
                    if abs(controller_state["simulation_time"] - physical["simulation_time"]) > max_age:
                        raise ValueError("Controller and physics observations exceed the capture window")
                    result["resources"][rid]["controller_hold_observation"] = controller_state
                    result["resources"][rid]["controller_hold_model"] = deepcopy(hold_model)
                    result["resources"][rid]["controller_hold_activity"] = deepcopy(hold_activity)
                except (ImportError, OSError, ValueError, KeyError, TypeError, AttributeError, RuntimeError) as exc:
                    result["resources"][rid]["controller_hold_observation"] = {
                        "status": "NEEDS_CONTEXT", "reason": str(exc)}
            state = physical.get("observation_state")
            if not isinstance(state, dict):
                raise ValueError("Observed resource state unavailable")
            _check_custody(physical, valuations[rid], {})
            result["snapshot"]["resources"][rid] = deepcopy(state)
            result["geometry"]["resources"][rid] = {
                key: deepcopy(physical[key]) for key in ("component_bounds", "footprint") if key in physical}
            result["geometry"]["resources"][rid]["frame"] = physical.get("frame")
            if "held_part" not in state:
                result["geometry"]["resources"][rid]["stationary_only"] = physical.get("model_static") is True
            read_model = getattr(owner, "get_recovery_safety_primitive_model", None)
            model = read_model() if callable(read_model) else None
            if model is not None:
                descriptor = model.descriptor()
                configuration = descriptor["configuration"]
                observed_configuration = physical.get("continuous_motion_configuration")
                if observed_configuration is not None:
                    from cais_spade_llm.resources.recovery_geometry import _configuration_after

                    declared = deepcopy(configuration.get("continuous_motion"))
                    if not isinstance(declared, dict) or not isinstance(observed_configuration, dict):
                        raise ValueError("Observed continuous geometry lacks the owner model")
                    # The owner binds these transforms to the same physics
                    # snapshot. Other geometry and joint declarations stay fixed.
                    for field in ("root_xyz", "root_quaternion", "root_rpy"):
                        if field in observed_configuration:
                            declared[field] = deepcopy(observed_configuration[field])
                        else:
                            declared.pop(field, None)
                    checked = {rid: {"configuration": declared}}
                    _configuration_after(checked, rid, {"configuration_after": observed_configuration})
                    configuration["continuous_motion"] = deepcopy(observed_configuration)
                result["resource_models"][rid] = {
                    "primitive_model": descriptor,
                    "named_joint_poses": deepcopy(configuration.get("named_joint_poses", {})),
                }
                if isinstance(configuration.get("continuous_motion"), dict):
                    result["resource_models"][rid]["configuration"] = deepcopy(configuration["continuous_motion"])
        except (ImportError, OSError, ValueError, KeyError, TypeError, AttributeError, RuntimeError) as exc:
            result["unresolved"].append({"resource_id": rid, "reason": str(exc)})
    observers = [getattr(owner, "recovery_safety_observer", None) for owner in owners.values()]
    observer = next((item for item in observers if item is not None), None)
    record, physics_error = None, None
    try:
        if observer is None:
            raise ValueError("Part collision geometry observation unavailable")
        record = observer.provider.reader.snapshot(max_age=max_age)
        if not 0 <= time.monotonic() - record["observed_monotonic"] <= max_age:
            raise ValueError("Stale part physics observation")
        result["physics_evidence"] = deepcopy(record)
    except (OSError, ValueError, KeyError, TypeError, AttributeError, RuntimeError) as exc:
        physics_error = str(exc)
    acquired_monotonic = time.monotonic()
    result["provenance"]["acquired_monotonic"] = acquired_monotonic
    geometry = context.inputs.get("geometry", {})
    part_geometry = geometry.get("parts", {})
    for name, ledger in tracker.items():
        model = part_geometry.get("model_map", {}).get(name) or part_geometry.get(name, {}).get("model_name")
        if name == context.inputs.get("scene", {}).get("Exit", {}).get("completed_product"):
            model = geometry.get("assembly_board", {}).get("model_name", model)
        try:
            held_owners = [rid for rid, row in result["snapshot"]["resources"].items() if row.get("held_part") == name]
            if len(held_owners) > 1:
                raise ValueError("Conflicting recorded holders for part")
            held = bool(held_owners)
            held_physical = result["resources"].get(held_owners[0], {}) if held else {}
            attachment = held_physical.get("attachment", {})
            if not model:
                raise ValueError("Configured part observation unavailable")
            if held:
                if (attachment.get("part_name") != name or attachment.get("model_name") != model
                        or attachment.get("observed_attachment_complete") is not True
                        or not isinstance(attachment.get("part_pose"), list)
                        or not isinstance(attachment.get("part_local_bounds"), list)):
                    raise ValueError("Held part requires its owning resource's complete attachment observation")
                part = {"name": model, "frame": held_physical["frame"], "pose": deepcopy(attachment["part_pose"]),
                    "instance_id": attachment["instance_id"], "simulation_time": held_physical["simulation_time"],
                    "simulation_stamp": held_physical["simulation_time"],
                    "observed_monotonic": held_physical["observed_monotonic"],
                    "source": "owning resource attachment observation", "resource_id": held_owners[0]}
            else:
                if record is None:
                    raise ValueError(physics_error or "Part collision geometry observation unavailable")
                part = {"name": model, "frame": "world", "pose": deepcopy(record["models"][model]["pose"]),
                    "instance_id": record["instance_id"], "simulation_time": record["simulation_time"],
                    "simulation_stamp": record["simulation_time"], "observed_monotonic": record["observed_monotonic"],
                    "source": "GETRECOVERYSTATE"}
            result["parts"][name] = part
            result["snapshot"]["parts"][name] = {
                **deepcopy(ledger), "current_pose": deepcopy(part["pose"]), "model_name": model,
                "contained_by": ledger.get("location") if not held and ledger.get("location") in valuations else None}
            completion = _completion_ledger_evidence(name, ledger, initial, acknowledgements, transitions,
                                                     run_id=context.run_id, revision=revision)
            result["snapshot"]["parts"][name].update(processCompleted_complete=completion["complete"],
                                                    processCompleted_evidence=completion)
            # Free parts share one acquired physics record. Mesh decoding and
            # other local calculations cannot refresh it halfway through.
            if record is None:
                raise ValueError(physics_error or "Part collision geometry observation unavailable")
            if (record["instance_id"] != part.get("instance_id")
                    or not held and record["simulation_time"] != part.get("simulation_time")):
                raise ValueError("Part pose and collision geometry belong to different observations")
            stationary = _stationary_part_observation(name, part, record, result,
                context.inputs.get("scene", {}).get("safety_preparation") or {})
            result["stationary_part_evidence"][name] = stationary
            if stationary["status"] == "supported":
                result["stationary_parts"].append(name)
            links = record["models"][model]["links"]
            declaration = context.inputs.get("scene", {}).get("safety_preparation", {}).get("part_geometry", {}).get(name)
            if declaration is not None:
                _configured_static_part_geometry(record, declaration, part)
                links = {source + "/" + key: deepcopy(value) for source in declaration["models"]
                         for key, value in record["models"][source]["links"].items()}
            for link in links.values():
                for collision in link.get("collisions", []):
                    shape = collision.get("geometry") or {}
                    if shape.get("type") == "mesh" and shape.get("uri") not in result["physics_meshes"]:
                        result["physics_meshes"][shape["uri"]] = _freeze_collision_mesh(shape["uri"])
            result["geometry"]["parts"][name] = {"frame": "part",
                "local_bounds": deepcopy(attachment["part_local_bounds"]) if held else _part_local_bounds(links, part["pose"]),
                "source": {"model": model, "instance_id": part["instance_id"],
                           "simulation_time": part["simulation_time"]}}
            if held:
                result["geometry"]["parts"][name]["source"].update(
                    resource_id=held_owners[0], source="owning resource attachment observation")
            if declaration is not None:
                result["geometry"]["parts"][name]["source"].update(
                    declaration=deepcopy(declaration), declaration_fingerprint=fingerprint(declaration),
                    constituent_models_fingerprint=fingerprint({key: record["models"][key] for key in declaration["models"]}))
            try:
                if declaration is None:
                    identity = observed_collision_identities(model, part)
                else:
                    constituents = [observed_collision_identities(source, {
                        "name": source, "frame": "world", "pose": record["models"][source]["pose"],
                        "source": "GETRECOVERYSTATE", "instance_id": record["instance_id"],
                        "simulation_time": record["simulation_time"],
                        "observed_monotonic": record["observed_monotonic"]}) for source in declaration["models"]]
                    identity = {"status": "recorded", "model_name": model,
                        "collision_objects": [deepcopy(row) for item in constituents for row in item["collision_objects"]],
                        "source": "Configured static constituent models and exact SDF collision identities",
                        "source_files": {path: digest for item in constituents for path, digest in item["source_files"].items()},
                        "observation": deepcopy(part), "declaration": deepcopy(declaration),
                        "declaration_fingerprint": fingerprint(declaration), "constituents": constituents,
                        "collision_scene_acknowledged": False}
                result["parts"][name]["collision_scene_identity"] = identity
            except (OSError, ValueError, KeyError, TypeError) as exc:
                result["parts"][name]["collision_scene_identity"] = {
                    "status": "NEEDS_CONTEXT", "model_name": model, "reason": str(exc),
                    "source": "SDF declarations used by collision_boxes and part_scene_update"}
            targets = part_geometry.get("assembly_target_map", {})
            if name in targets:
                if not isinstance(targets[name], str) or not targets[name]:
                    raise ValueError("Configured assembly target is unavailable: " + name)
                result["geometry"]["parts"][name]["target"] = targets[name]
        except (OSError, ValueError, KeyError, TypeError, AttributeError, RuntimeError) as exc:
            result["unresolved"].append({"part": name, "reason": str(exc)})
    current = {rid: actor.valuation for rid, actor in context.resources.items()}
    if revision != context.revision or valuations != current or tracker != context.part_tracker:
        result["unresolved"].append({"reason": "Runtime observations changed during physical capture"})
        result["snapshot"] = {"frame": "world", "resources": {}, "parts": {}}
        result["stationary_parts"] = []
    launches = {row.get("launch_id") for row in result["resources"].values()}
    frozen_resources = (frozen_fingerprint is not None and bool(result["resources"])
        and all(row.get("frozen_native_fingerprint") == frozen_fingerprint
                for row in result["resources"].values()))
    if (not frozen_resources and acquired_monotonic - started > max_age) or len(launches) > 1:
        result["unresolved"].append({"reason": "Physical capture window is stale or launch identities disagree"})
    result["evidence_fingerprint"] = fingerprint(result)
    return result


def _completion_ledger_evidence(name: str, ledger: dict, initial: dict, acknowledgements: dict,
                                transitions: list, *, run_id: str, revision: int) -> dict:
    """Bind completion facts to an uninterrupted recorded acknowledgement ledger."""
    record = {"source_kind": "recorded_runtime_acknowledgements", "complete": False,
              "run_id": run_id, "context_revision": revision,
              "initial_product_states_fingerprint": fingerprint(initial),
              "acknowledgements_fingerprint": fingerprint(acknowledgements),
              "transitions_fingerprint": fingerprint(transitions)}
    record["checkpoint"] = fingerprint(record)
    current = initial.get(name, {}).get("processCompleted")
    if not isinstance(current, list) or not isinstance(acknowledgements, dict):
        return {**record, "reason": "Initial completion ledger or acknowledged task index unavailable"}
    seen = set()
    for transition in transitions:
        acknowledgement = transition.get("acknowledgement") or {}
        task_id = acknowledgement.get("task_id")
        before = transition.get("product_before", {}).get(name, {}).get("processCompleted")
        after = transition.get("product_after", {}).get(name, {}).get("processCompleted")
        if (not task_id or task_id in seen or acknowledgements.get(task_id) != acknowledgement
                or acknowledgement.get("status") != "completed" or before != current
                or not isinstance(after, list)):
            return {**record, "reason": "Completion ledger does not match the recorded acknowledgement chain"}
        seen.add(task_id)
        current = after
    if set(acknowledgements) != seen or current != ledger.get("processCompleted"):
        return {**record, "reason": "Completion ledger has missing or inconsistent acknowledged transitions"}
    return {**record, "complete": True}


def read_observations(path: str | Path, *, root: Path = RECORDS) -> dict:
    """Read a contained capture, including facts retained by older runs."""
    source = Path(path).resolve()
    if not source.is_relative_to(root.resolve()) or source.name not in {"observations.json", "environment_run.json"}:
        raise ValueError("Select a saved observation record from this Recovery page")
    record = read_capture(source) if source.name == "observations.json" else read_object(source)
    if source.name == "environment_run.json":
        fault = record.get("conveyor_fault") or {}
        record = {
            "schema_version": 1, "run_id": record.get("run_id"),
            "captured_at": None, "evidence_source": "Gazebo",
            "scenario": fault.get("scenario"), "failure": fault.get("evidence", {}),
            "part_tracker": record.get("final_product_states", {}),
            "resource_states": record.get("final_valuation", {}),
            "inputs": record.get("inputs", {}), "safety_status": record.get("formal_safety_status"),
            "offline_ready": False, "missing": ["prepared_recovery_request"],
        }
        # Older runs sometimes completed preparation before their UI disconnected.
        for candidate in sorted(source.parent.glob("generation_*/inputs.json"), reverse=True):
            prepared = read_object(candidate).get("prepared_recovery_request")
            if isinstance(prepared, dict) and prepared.get("recovery_resources"):
                record.update(prepared_recovery_request=prepared, offline_ready=True, missing=[])
                break
    record["source_path"] = str(source)
    record["source_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    return record


def copy_generation_records(state: dict, source_directory: Path, directory: Path) -> dict:
    """Retain original stage files in a separate review or Gazebo-test attempt."""
    parent = source_directory.resolve()
    source = Path(state["artifact_directory"]).resolve()
    if not source.is_relative_to(parent):
        raise ValueError("Saved composition refers to another run")
    paths = {}
    for key, stage in (("outline", "recovery_outline"), ("primitive", "recovery_primitves")):
        captures = Path(state.get(f"{key}_artifact_directory") or source / stage).resolve()
        if not captures.is_relative_to(parent):
            raise ValueError("Saved prompt captures refer to another run")
        if captures.is_dir():
            shutil.copytree(captures, directory / stage)
        files = ("inputs.json", "input_preview.json") if key == "outline" else ("primitive_input_preview.json",)
        for name in files:
            if (captures.parent / name).is_file():
                shutil.copy2(captures.parent / name, directory / name)
        paths[f"{key}_artifact_directory"] = str(directory / stage)
    for name in ("proposal.json", "outline_request.json", "outline.json"):
        if (source / name).is_file():
            shutil.copy2(source / name, directory / name)
    if (source / "proposal.json").is_file():
        shutil.copy2(source / "proposal.json", directory / "source_proposal.json")
    write_record(directory / "source_run.json", state)
    return paths


@lru_cache(maxsize=512)
def _observation_list_summary(path: Path, root: Path, modified_ns: int, size: int,
                              preparation_modified_ns: int = 0) -> tuple[dict, str]:
    # Cache only display metadata; loading and validation always reread the capture.
    # Schema-2 derived views live beside the raw file, so their time is part of the key.
    record = read_observations(path, root=root)
    item = {key: record.get(key) for key in (
        "source_path", "run_id", "captured_at", "scenario", "offline_ready", "missing",
    )}
    item["missing"] = offline_preparation_missing(record)
    item["offline_ready"] = not item["missing"]
    return item, observation_fingerprint(record)


def list_observations(*, root: Path = RECORDS) -> list[dict]:
    """List each captured input once, retaining links to its offline attempts."""
    captures: dict[str, list[tuple[dict, int]]] = {}
    for directory in sorted(root.glob("*"), reverse=True):
        path = directory / "observations.json"
        if not path.is_file():
            continue
        try:
            stat = path.stat()
            beliefs = directory / BELIEFS_NAME
            summary, capture_fingerprint = _observation_list_summary(
                path, root, stat.st_mtime_ns, stat.st_size,
                beliefs.stat().st_mtime_ns if beliefs.is_file() else 0)
        except (OSError, ValueError, TypeError):
            continue
        item = deepcopy(summary)
        item["saved_run_id"] = directory.name
        run_path = directory / "run.json"
        result_time = 0
        try:
            saved_state = read_object(run_path) if run_path.is_file() else {}
            item["result_available"] = (saved_state.get("status") == "offline_completed" or
                saved_state.get("status") == "outline_review" and saved_state.get("outline_status") == "completed"
                and saved_state.get("validation_contract_version") == 2)
            if item["result_available"]:
                result_time = run_path.stat().st_mtime_ns
        except (OSError, ValueError):
            item["result_available"] = False
        captures.setdefault(capture_fingerprint, []).append((item, result_time))
    records = []
    for copies in captures.values():
        # Offline attempts preserve the input content but add source file metadata.
        # Prefer the original run's file so preparing another attempt cannot move it.
        item = dict(min((row for row, _ in copies), key=lambda row: (
            row["saved_run_id"] != row["run_id"], row["source_path"],
        )))
        item["related_observation_paths"] = sorted(row["source_path"] for row, _ in copies)
        completed = [(row, stamp) for row, stamp in copies if row["result_available"]]
        item["result_available"] = bool(completed)
        item["result_observation_path"] = max(completed, key=lambda entry: entry[1])[0]["source_path"] if completed else ""
        records.append(item)
    def capture_time(item: dict) -> float:
        try:
            stamp = datetime.fromisoformat(item.get("captured_at") or "")
            return stamp.replace(tzinfo=stamp.tzinfo or timezone.utc).timestamp()
        except (TypeError, ValueError):
            return float("-inf")

    return sorted(records, key=capture_time, reverse=True)

def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _native_archive_file(directory: Path, entry: dict) -> Path:
    name = entry.get('path')
    if not isinstance(name, str) or not name or Path(name).is_absolute():
        raise ValueError('Native preview archive requires contained relative file paths')
    candidate = directory / name
    if '..' in Path(name).parts or any(parent.is_symlink() for parent in (candidate, *candidate.parents)
                                       if parent != directory.parent):
        raise ValueError('Native preview archive cannot contain symlinks or path escapes')
    path = candidate.resolve()
    if not path.is_relative_to(directory.resolve()) or not path.is_file():
        raise ValueError('Native preview archive file is unavailable: ' + name)
    if _file_sha256(path) != entry.get('sha256'):
        raise ValueError('Native preview archive file changed: ' + name)
    return path


def read_native_preview_model(observation_path: str | Path, observations: dict) -> dict:
    """Verify a capture's private model companion without live interfaces.

    Args:
        observation_path: Original immutable observation file.
        observations: Exact loaded observation values, before supplementation.

    Returns:
        Verified local paths and bindings for file-based owner calculations.
    """
    source = Path(observation_path).resolve()
    capture_fingerprint = observation_fingerprint(observations)
    records_parent, visited = source.parent.parent, set()
    while not (source.parent / 'native_preview_model' / 'manifest.json').is_file():
        if source in visited or len(visited) >= 8:
            raise ValueError('Saved native model source reference is cyclic or too deep')
        visited.add(source)
        saved = read_capture(source)
        if observation_fingerprint(saved) != capture_fingerprint:
            raise ValueError('Saved native model copy differs from the selected observations')
        reference = saved.get('source_path')
        if not isinstance(reference, str) or not reference or 'source_sha256' not in saved:
            raise ValueError('NEEDS_CONTEXT: saved native preview model archive is unavailable')
        candidate = Path(reference)
        if (candidate.is_symlink() or not candidate.is_absolute() or candidate.name != 'observations.json'
                or candidate.resolve().parent.parent != records_parent or not candidate.is_file()):
            raise ValueError('Saved native model source must remain inside the same Recovery records')
        candidate = candidate.resolve()
        if _file_sha256(candidate) != saved['source_sha256']:
            raise ValueError('Saved native model source hash changed')
        source = candidate
    return _read_native_preview_model(source, observations, source.parent / 'native_preview_model')


def _read_native_preview_model(source: Path, observations: dict, directory: Path) -> dict:
    manifest_path = directory / 'manifest.json'
    if directory.is_symlink() or manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError('NEEDS_CONTEXT: saved native preview model archive is unavailable')
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    scene = observations.get('planning_scene_capture') or {}
    source_sha = _file_sha256(source)
    binding = {'source_sha256': source_sha, 'observation_fingerprint': observation_fingerprint(observations),
               'run_id': observations.get('run_id'), 'scene_fingerprint': scene.get('scene_fingerprint'),
               'model_fingerprint': scene.get('model_fingerprint')}
    if (type(manifest.get('version')) is not int or manifest['version'] != 1
            or fingerprint(manifest.get('binding')) != fingerprint(binding)
            or not all(binding.values()) or manifest.get('execution_authorized') is not False):
        raise ValueError('Native preview model archive has stale capture or model bindings')
    actual = read_capture(source)
    if observation_fingerprint(actual) != binding['observation_fingerprint']:
        raise ValueError('Native preview model requires unchanged original observations')
    files = manifest.get('files')
    if not isinstance(files, dict) or not {'parameters', 'verification'} <= set(files):
        raise ValueError('Native preview model archive lacks parameters or verification')
    paths = {key: _native_archive_file(directory, entry) for key, entry in files.items()}
    verification = json.loads(paths['verification'].read_bytes())
    if (verification.get('parameters_sha256') != files['parameters']['sha256']
            or verification.get('model_fingerprint') != binding['model_fingerprint']
            or verification.get('fingerprint_verified') is not True
            or verification.get('native_exit_code') != 0
            or verification.get('command_sent') is not False
            or verification.get('execution_authorized') is not False):
        raise ValueError('Native preview model fingerprint was not independently verified')
    if 'typed_parameters' in files:
        import yaml

        typed = json.loads(paths['typed_parameters'].read_bytes())
        if fingerprint(typed) != fingerprint(scene.get('native_model_parameters')):
            raise ValueError('Archived typed parameters differ from captured parameter values')
        unset = sorted(name for name, row in typed.items() if type(row.get('type')) is int and row['type'] == 0)
        if unset and (verification.get('unset_parameter_names') != unset
                      or verification.get('unset_parameter_handling') != 'retained as PARAMETER_NOT_SET; no YAML override'):
            raise ValueError('Native preview verification does not preserve captured unset parameters')
        parameters = yaml.safe_load(paths['parameters'].read_text())
        expected = {'/**': {'ros__parameters': _captured_native_parameters(scene)}}
        if fingerprint(parameters) != fingerprint(expected):
            raise ValueError('Archived native YAML changes captured parameter types or values')
    declarations = None
    if 'supplementary_model_declarations' in files:
        if 'supplementary_source' not in files:
            raise ValueError('Supplementary model declarations lack their exact source file')
        supplemental = json.loads(paths['supplementary_model_declarations'].read_bytes())
        original = json.loads(paths['supplementary_source'].read_bytes())
        if (supplemental.get('sha256') != files['supplementary_source']['sha256']
                or 'planning_scene_mounting_contacts' not in original
                or fingerprint(supplemental.get('declarations')) != fingerprint(original['planning_scene_mounting_contacts'])):
            raise ValueError('Supplementary model declarations conflict with their recorded source')
        declarations = supplemental['declarations']
        recorded = observations.get('inputs', {}).get('scene', {})
        if 'planning_scene_mounting_contacts' in recorded and fingerprint(recorded['planning_scene_mounting_contacts']) != fingerprint(declarations):
            raise ValueError('Supplementary model declarations conflict with captured values')
    if manifest_path.read_bytes() != manifest_bytes or _file_sha256(source) != source_sha:
        raise ValueError('Native preview model or observations changed during reading')
    return {'source_path': str(source), 'manifest_path': str(manifest_path),
            'manifest_sha256': hashlib.sha256(manifest_bytes).hexdigest(), 'binding': binding,
            'parameters_path': str(paths['parameters']), 'parameters_sha256': files['parameters']['sha256'],
            'files': {str(paths[key]): files[key]['sha256'] for key in paths},
            'capture_verification': verification, 'supplementary_model_declarations': deepcopy(declarations)}


def native_preview_observations(observations: dict, archive: dict) -> dict:
    """Supply only verified absent mounting declarations in a private copy."""
    verified = read_native_preview_model(archive['source_path'], observations)
    if fingerprint(verified) != fingerprint(archive):
        raise ValueError('Native preview archive changed after preparation')
    result = deepcopy(observations)
    declarations = verified['supplementary_model_declarations']
    if declarations is not None:
        scene = result.setdefault('inputs', {}).setdefault('scene', {})
        if 'planning_scene_mounting_contacts' not in scene:
            scene['planning_scene_mounting_contacts'] = deepcopy(declarations)
    return result


def import_native_preview_model(observation_path: str | Path, *, parameters_path: str | Path,
                                verification_path: str | Path, provenance_path: str | Path,
                                supplementary_path: str | Path | None = None,
                                supplementary_source_path: str | Path | None = None) -> dict:
    """Publish an explicitly selected, source-bound archive beside a saved capture.

    Original capture and provenance bytes remain unchanged. Existing companions
    are accepted only when they contain the same source files and bindings.
    """
    import tempfile

    source = Path(observation_path).resolve()
    observations = read_capture(source)
    scene = observations.get('planning_scene_capture') or {}
    parameters, verification_file, provenance_file = map(Path, (parameters_path, verification_path, provenance_path))
    verification, provenance = json.loads(verification_file.read_bytes()), json.loads(provenance_file.read_bytes())
    sha = _file_sha256(source)
    parameter_sha = _file_sha256(parameters)
    if (provenance.get('source_sha256') != sha or provenance.get('parameters_sha256') != parameter_sha
            or provenance.get('expected_model_fingerprint') != scene.get('model_fingerprint')
            or provenance.get('exact_native_model_match') is not True
            or provenance.get('source_verification_sha256') != _file_sha256(verification_file)):
        raise ValueError('Recorded native preview provenance does not bind this capture and parameters')
    entries = {'parameters': parameters, 'verification': verification_file, 'provenance': provenance_file}
    for index, startup in enumerate(verification.get('startup_files', [])):
        original = Path(startup['archive'])
        if _file_sha256(original) != startup['sha256']:
            raise ValueError('Recorded native startup archive changed')
        entries['startup_' + str(index)] = original
    if supplementary_path is not None:
        if supplementary_source_path is None:
            raise ValueError('Supplementary declarations require their original source')
        entries['supplementary_model_declarations'] = Path(supplementary_path)
        entries['supplementary_source'] = Path(supplementary_source_path)
    for name in ('typed_parameters.json', 'fingerprint-goal.cdr', 'fingerprint-result.cdr', 'fingerprint-native.log'):
        candidate = verification_file.parent / name
        if candidate.is_file():
            entries[name.rsplit('.', 1)[0]] = candidate
    directory = source.parent / 'native_preview_model'
    temporary = Path(tempfile.mkdtemp(prefix='.native_preview_model_', dir=source.parent))
    try:
        files = {}
        for key, original in entries.items():
            suffix = '.yaml' if key == 'parameters' or key.startswith('startup_') else original.suffix
            name = key + suffix
            data = original.read_bytes()
            (temporary / name).write_bytes(data)
            if _file_sha256(original) != hashlib.sha256(data).hexdigest():
                raise ValueError('Recorded native preview input changed during import')
            files[key] = {'path': name, 'sha256': hashlib.sha256(data).hexdigest()}
        manifest = {'version': 1, 'binding': {'source_sha256': sha,
            'observation_fingerprint': observation_fingerprint(observations), 'run_id': observations.get('run_id'),
            'scene_fingerprint': scene.get('scene_fingerprint'), 'model_fingerprint': scene.get('model_fingerprint')},
            'files': files, 'command_sent': False, 'execution_authorized': False}
        write_record(temporary / 'manifest.json', manifest)
        _read_native_preview_model(source, observations, temporary)
        if directory.exists():
            if fingerprint(json.loads((directory / 'manifest.json').read_bytes())) != fingerprint(manifest):
                raise ValueError('A different native preview companion already exists for this capture')
        else:
            # Validate before publishing by checking every bound source and the
            # immutable capture; the reader independently repeats these checks.
            if _file_sha256(source) != sha:
                raise ValueError('Saved observations changed during archive import')
            temporary.rename(directory)
        return read_native_preview_model(source, observations)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)

def _captured_native_parameters(scene: dict) -> dict:
    """Read exact set ROS parameter values without inventing unset values.

    PARAMETER_NOT_SET remains recorded in typed_parameters.json and supplies no
    YAML override. An archive is published only after the native startup model
    fingerprint independently matches the capture; omission is not validation.
    """
    typed = scene.get('native_model_parameters')
    if not isinstance(typed, dict) or not {'robot_description', 'robot_description_semantic'} <= set(typed):
        raise ValueError('Captured typed native model parameters are unavailable')
    fields = {1: ('bool_value', bool), 2: ('integer_value', int), 3: ('double_value', float),
              4: ('string_value', str), 6: ('bool_array_value', bool), 7: ('integer_array_value', int),
              8: ('double_array_value', float), 9: ('string_array_value', str)}
    result = {}
    for name, row in typed.items():
        if name not in {'robot_description', 'robot_description_semantic'} and not any(
                name == prefix or name.startswith(prefix + '.')
                for prefix in ('robot_description_kinematics', 'robot_description_planning')):
            raise ValueError('Unsupported captured native parameter: ' + name)
        kind = row.get('type')
        if type(kind) is int and kind == 0:
            if name in {'robot_description', 'robot_description_semantic'}:
                raise ValueError('Required captured native description is unset: ' + name)
            defaults = {'bool_value': False, 'integer_value': 0, 'double_value': 0., 'string_value': '',
                        'byte_array_value': [], 'bool_array_value': [], 'integer_array_value': [],
                        'double_array_value': [], 'string_array_value': []}
            if any(field not in defaults or fingerprint(value) != fingerprint(defaults[field])
                   for field, value in row.items() if field != 'type'):
                raise ValueError('Captured unset parameter contains conflicting value fields: ' + name)
            continue
        if type(kind) is not int or kind not in fields:
            raise ValueError(f'Native parameter {name} type {kind!r} cannot be preserved in a ROS YAML archive')
        field, expected = fields[kind]
        value = row.get(field)
        values = value if kind >= 6 else [value]
        if not isinstance(values, list) or not values or any(type(item) is not expected for item in values):
            raise ValueError('Captured native parameter type or array is unresolved: ' + name)
        if expected is float and any(not math.isfinite(item) for item in values):
            raise ValueError('Captured native parameter is non-finite: ' + name)
        result[name] = deepcopy(value)
    if any(result[name] != scene.get(name) for name in ('robot_description', 'robot_description_semantic')):
        raise ValueError('Captured native description parameters differ from the saved scene')
    return result


async def save_native_preview_model_from_capture(observation_path: str | Path, *, cancel_event=None) -> dict:
    """Archive and independently verify newly captured typed parameters locally.

    Call after the final observation save. Native file mode computes only the
    model fingerprint: its intentionally incomplete Goal cannot request motion.
    """
    import tempfile
    import yaml
    from cais_spade_llm.recovery_framework.recovery_motion_preview import verify_native_model_parameters

    source = Path(observation_path).resolve()
    observations = read_capture(source)
    existing = source.parent / 'native_preview_model'
    if existing.exists():
        return read_native_preview_model(source, observations)
    scene = observations.get('planning_scene_capture') or {}
    parameters = _captured_native_parameters(scene)
    directory = Path(tempfile.mkdtemp(prefix='.captured_native_parameters_', dir=source.parent))
    try:
        parameter_path = directory / 'parameters.yaml'
        parameter_path.write_text(yaml.safe_dump({'/**': {'ros__parameters': parameters}}, sort_keys=True))
        write_record(directory / 'typed_parameters.json', scene['native_model_parameters'])
        verification = await verify_native_model_parameters(parameter_path,
            expected_model_fingerprint=scene['model_fingerprint'], artifact_directory=directory,
            cancel_event=cancel_event)
        verification['unset_parameter_names'] = sorted(name for name, row in scene['native_model_parameters'].items()
            if type(row.get('type')) is int and row['type'] == 0)
        verification['unset_parameter_handling'] = 'retained as PARAMETER_NOT_SET; no YAML override'
        write_record(directory / 'verification.json', verification)
        write_record(directory / 'provenance.json', {'kind': 'captured_typed_native_parameters',
            'source_sha256': _file_sha256(source), 'parameters_sha256': _file_sha256(parameter_path),
            'expected_model_fingerprint': scene['model_fingerprint'], 'exact_native_model_match': True,
            'source_verification_sha256': _file_sha256(directory / 'verification.json'),
            'command_sent': False, 'execution_authorized': False})
        if observation_fingerprint(read_capture(source)) != observation_fingerprint(observations):
            raise ValueError('Saved observations changed during native model verification')
        return import_native_preview_model(source, parameters_path=parameter_path,
            verification_path=directory / 'verification.json', provenance_path=directory / 'provenance.json')
    finally:
        # Preserve verification files as independent diagnostics, including an
        # incomplete calculation; only the published companion is used offline.
        retained = source.parent / ('native_model_capture_' + directory.name.removeprefix('.captured_native_parameters_'))
        directory.rename(retained)
