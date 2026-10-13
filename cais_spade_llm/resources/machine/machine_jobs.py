from __future__ import annotations

"""Resource-owned job control for the declared simulation machining process."""

import hashlib
import json
import math
import random
from copy import deepcopy
from fractions import Fraction
from typing import Any

MACHINE_PRIMITIVES = ("select_program", "start_job", "pause_job", "resume_job", "stop_job")
JOB_CONTROL_VERSION = 1


def job_public_state(record: dict) -> dict:
    """Read public fields from the owner record without manufacturing completion."""
    active = record["active_job"]
    jobs = list(record["jobs"].values())
    job = record["jobs"][active] if active else (jobs[-1] if jobs else {})
    return {"active_job": active, "job_state": job.get("job_state"),
            "selected_program": record["selected_program"]}


def public_processing_observations(record: dict) -> dict:
    """Expose factual job progress without command history or private programs."""
    fields = {"job_id", "task_id", "part_name", "process", "result", "program", "job_state",
              "simulation_started_sec", "simulation_clock_sec", "simulation_elapsed_sec",
              "processing_time_sec", "processing_active_sec", "simulation_paused_sec",
              "processing_fraction", "process_completed", "simulation_completed_sec"}
    return {**job_public_state(record), "run_id": record["run_id"], "resource_id": record["resource_id"],
            "revision": record["revision"], "model_fingerprint": record["model_fingerprint"],
            "jobs": {key: {field: deepcopy(value) for field, value in row.items() if field in fields}
                     for key, row in record["jobs"].items()},
            "progress_basis": "active simulated processing time divided by configured duration; not measured material removal"}


def project_job_command(machine: dict, record: dict, primitive: str, params: dict,
                        observations: dict, *, command_id: str) -> tuple[dict, dict]:
    """Check one command against a copied owner snapshot without live interfaces."""
    if not record:
        raise ValueError("NEEDS_CONTEXT: processing_observations")
    owner = MachineJobs(machine, record["run_id"], record)
    now = record.get("simulation_clock_sec")
    if now is None:
        raise ValueError("NEEDS_CONTEXT: observed simulation_clock_sec")
    reply = owner.command(primitive, params, command_id=command_id, now=now, observations=observations)
    return owner.snapshot(), reply


def missing_processing_projection(before: dict, projection: dict) -> list[str]:
    """Keep active peer jobs from silently freezing during a proposed interval."""
    records = before.get('processing_observations', {})
    if not records or projection.get('status') != 'projected':
        return []
    observations = projection.get('observations', [])
    if not observations:
        return ['processing interval duration']
    duration = Fraction(str(observations[-1]['time'])) - Fraction(str(observations[0]['time']))
    if duration == 0:
        return []
    after = projection.get('successor_evidence', {}).get('processing_observations', {})
    missing = []
    for rid, record in records.items():
        if rid in before.get('unavailable_resources', []) or record.get('active_job') is None:
            continue
        identity = record['active_job']
        job = record['jobs'][identity]
        if job['job_state'] == 'running' and after.get(rid, {}).get('jobs', {}).get(identity) == job:
            missing.append(rid + '/jobs/' + identity)
        elif job['job_state'] == 'running' and identity not in after.get(rid, {}).get('jobs', {}):
            missing.append(rid + '/jobs/' + identity)
    return missing


def processing_failure_time(records: dict, resources: list[str], *, now: float,
                            seed: int, observation_interval_sec: float) -> dict | None:
    """Choose one reproducible nonterminal point from observed processing overlap."""
    if type(seed) is not int:
        raise ValueError("NEEDS_CONTEXT: random failure seed")
    if set(records) != set(resources):
        return None
    jobs = []
    for rid in resources:
        record = records[rid]
        if record.get("resource_id") != rid:
            raise ValueError("Processing observation resource identity mismatch")
        if record.get("active_job") is None:
            if any(row["job_state"] == "completed" for row in record["jobs"].values()):
                raise ValueError("Processing overlap was not observed before completion")
            return None
        job = record["jobs"][record["active_job"]]
        if job["job_state"] != "running" or job["process_completed"]:
            raise ValueError("Processing overlap lacks two running jobs")
        jobs.append(job)
    if len({row["run_id"] for row in records.values()}) != 1:
        raise ValueError("Processing observations belong to different runs")
    end = min(job["simulation_clock_sec"] + job["processing_time_sec"] - job["processing_active_sec"] for job in jobs)
    low, high = now + observation_interval_sec, end - observation_interval_sec
    if not math.isfinite(low) or not math.isfinite(high) or high <= low:
        raise ValueError("Insufficient nonterminal processing overlap; trial was not repeated")
    return {"seed": seed, "observed_at_simulation_sec": now,
            "overlap_end_simulation_sec": end,
            "interruption_simulation_sec": random.Random(seed).uniform(low, high),
            "source_job_revisions": {rid: records[rid]["revision"] for rid in resources}}


def fingerprint(value: Any) -> str:
    """Return the identity of recorded JSON configuration or command data."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def machine_programs(machine: dict) -> dict:
    """Return declared program options without inventing a program identifier."""
    programs = deepcopy(machine.get("program_options", {}))
    selected = machine.get("selected_program")
    current = machine.get("current_configuration")
    if not isinstance(selected, str) or selected not in programs:
        raise ValueError("NEEDS_CONTEXT: declared selected_program and program_options are required")
    if current != programs[selected]:
        raise ValueError("Current configuration differs from the declared selected_program")
    return programs


def machine_job_functions() -> dict:
    """Declare job commands separately from acknowledged machining completion."""
    functions = {name: {"function_name": name, "status": "implemented",
                        "program": {"steps": [{"id": name, "op": name}]}}
                 for name in MACHINE_PRIMITIVES}
    functions["start_job"]["program"]["entry_state"] = "loaded"
    functions["machine_part"] = {
        "function_name": "trim_part", "status": "implemented",
        "program": {"entry_state": "loaded", "success_state": "completed",
                    "steps": [{"id": "process", "op": "start_job"}]},
    }
    return functions


class MachineJobs:
    """Retain simulated jobs independently of workholding and worker requests.

    Args:
        machine: Captured owner declaration, including both program options.
        run_id: Identity of the run owning this job record.
        record: Optional immutable saved record to copy for replay or projection.

    The class performs no I/O. Callers supply observations and simulation time;
    a prediction must never be labelled as a worker acknowledgement.
    """

    def __init__(self, machine: dict, run_id: str, record: dict | None = None) -> None:
        if machine.get("simulation_process", {}).get("job_control", {}).get("version") != JOB_CONTROL_VERSION:
            raise ValueError("NEEDS_CONTEXT: machine job-control declaration is unavailable")
        self.machine = deepcopy(machine)
        self.programs = machine_programs(machine)
        self.run_id = run_id
        self.model_fingerprint = fingerprint(machine)
        self.record = deepcopy(record) if record is not None else {
            "version": JOB_CONTROL_VERSION, "run_id": run_id,
            "resource_id": machine["resource_id"], "model_fingerprint": self.model_fingerprint,
            "selected_program": machine["selected_program"],
            "current_configuration": deepcopy(machine["current_configuration"]),
            "active_job": None, "jobs": {}, "commands": {}, "revision": 0,
        }
        for field, value in (("version", JOB_CONTROL_VERSION), ("run_id", run_id),
                             ("resource_id", machine["resource_id"]),
                             ("model_fingerprint", self.model_fingerprint)):
            if self.record.get(field) != value:
                raise ValueError("Stale machine job record: " + field)

    def _time(self, now: float) -> float:
        if type(now) not in (int, float) or not math.isfinite(now) or now < 0:
            raise ValueError("NEEDS_CONTEXT: finite simulation time is required")
        previous = self.record.get("simulation_clock_sec")
        if previous is not None and now < previous:
            raise ValueError("Simulation clock moved backwards")
        return float(now)

    def _configuration(self, program: str) -> dict:
        if program not in self.programs:
            raise ValueError("Requested program is not declared")
        target = self.programs[program]
        installed = self.record["current_configuration"]
        for key in ("tool", "workholding"):
            if not installed.get(key) or not target.get(key):
                raise ValueError("NEEDS_CONTEXT: installed " + key + " is unavailable")
            if target[key] != installed[key]:
                raise ValueError("Program requires a different " + key)
        declaration = target.get("program", {})
        if declaration.get("validated") is not True:
            raise ValueError("NEEDS_CONTEXT: program has no validated effects")
        if any(installed.get(key) != value for key, value in declaration.get("required_configuration", {}).items()):
            raise ValueError("Installed configuration does not satisfy the selected program")
        return deepcopy(target)

    def _ready(self, part: str, observations: dict) -> None:
        for field in ("part_name", "correct_part_present", "robot_access_clear"):
            if field not in observations or observations[field] is None:
                raise ValueError("NEEDS_CONTEXT: " + field)
        if observations["part_name"] != part or observations["correct_part_present"] is not True:
            raise ValueError("Observed workholding does not contain the job's part")
        if observations["robot_access_clear"] is not True:
            raise ValueError("Machine access is occupied")

    def command(self, primitive: str, params: dict, *, command_id: str,
                now: float, observations: dict) -> dict:
        """Apply one confirmed owner command, preserving idempotent replies.

        Args:
            primitive: Exact declared job-control primitive.
            params: Bound job or program arguments.
            command_id: Unique request identity; reuse requires identical arguments.
            now: Observed simulation time when the worker applies the command.
            observations: Current workholding and access observations.

        Returns:
            A command acknowledgement, never a machining completion record.
        """
        if primitive not in MACHINE_PRIMITIVES or not command_id:
            raise ValueError("Unknown machine command or missing command identity")
        signature = fingerprint([primitive, params])
        previous = self.record["commands"].get(command_id)
        if previous is not None:
            if previous["signature"] != signature:
                raise ValueError("Command identity was reused with conflicting parameters")
            return deepcopy(previous["reply"])
        now = self._time(now)
        if (primitive in {"pause_job", "stop_job"}
                and self.record["active_job"] == params.get("job_id")
                and self.record["active_job"] is not None):
            # An observed completion wins a simultaneous control request.
            self.observe(now, observations)
        before = deepcopy(self.record)
        try:
            job_id = params.get("job_id")
            active = self.record["active_job"]
            if primitive == "select_program":
                if active is not None:
                    raise ValueError("Cannot select_program while a job is processing")
                target = self._configuration(params.get("program"))
                self.record.update(selected_program=params["program"], current_configuration=target)
            elif primitive == "start_job":
                if not isinstance(job_id, str) or not job_id or job_id in self.record["jobs"]:
                    raise ValueError("start_job requires a new job identity")
                if active is not None:
                    raise ValueError("A machining job is already active")
                part = params.get("part_name")
                if not isinstance(part, str) or not part or not params.get("task_id"):
                    raise ValueError("NEEDS_CONTEXT: part and originating task identities are required")
                if any(row["part_name"] == part and row["job_state"] in {"paused", "stopped", "interrupted"}
                       for row in self.record["jobs"].values()):
                    raise ValueError("Restarting this unfinished part has no declared support")
                self._ready(part, observations)
                configuration = self._configuration(self.record["selected_program"])
                effect = {"process": params.get("process"), "result": params.get("result")}
                if effect not in configuration["program"].get("effects", []):
                    raise ValueError("Selected program cannot establish the requested result")
                duration = self.machine["simulation_process"]["processing_time_sec"]
                if type(duration) not in (int, float) or not math.isfinite(duration) or duration <= 0:
                    raise ValueError("NEEDS_CONTEXT: configured processing_time_sec")
                self.record["jobs"][job_id] = {
                    "job_id": job_id, "task_id": params["task_id"], "part_name": part,
                    **effect, "program": self.record["selected_program"],
                    "configuration": configuration, "configuration_fingerprint": fingerprint(configuration),
                    "job_state": "running", "simulation_started_sec": now,
                    "simulation_clock_sec": now, "simulation_elapsed_sec": 0.0,
                    "processing_time_sec": float(duration), "processing_active_sec": 0.0,
                    "simulation_paused_sec": 0.0, "processing_fraction": 0.0,
                    "process_completed": False, "observations": deepcopy(observations),
                }
                self.record["active_job"] = job_id
            else:
                if job_id not in self.record["jobs"]:
                    raise ValueError("NEEDS_CONTEXT: retained job " + str(job_id))
                job = self.record["jobs"][job_id]
                if primitive == "resume_job":
                    if job["job_state"] != "paused" or active is not None:
                        raise ValueError("resume_job requires a paused job and no active processing")
                    self._ready(job["part_name"], observations)
                    if self.record["current_configuration"] != job["configuration"]:
                        raise ValueError("Retained job requires its original configuration")
                    if self.record["selected_program"] != job["program"]:
                        raise ValueError("Retained job requires its original program")
                    if self.machine["simulation_process"]["job_control"].get("resume_after_reload") is not True:
                        raise ValueError("NEEDS_CONTEXT: resume_after_reload is not declared")
                    job["simulation_paused_sec"] += now - job["simulation_clock_sec"]
                    job["job_state"] = "running"
                    self.record["active_job"] = job_id
                elif primitive == "pause_job":
                    if job["job_state"] != "running" or active != job_id:
                        raise ValueError("pause_job requires the active running job")
                    self._account(job, now, observations)
                    job["job_state"] = "paused"
                    self.record["active_job"] = None
                else:
                    if job["job_state"] not in {"running", "paused"}:
                        raise ValueError("stop_job requires an unfinished running or paused job")
                    if job["job_state"] == "running":
                        self._account(job, now, observations)
                    else:
                        job["simulation_paused_sec"] += now - job["simulation_clock_sec"]
                    job["job_state"] = "stopped"
                    if active == job_id:
                        self.record["active_job"] = None
                job.update(simulation_clock_sec=now,
                           simulation_elapsed_sec=now - job["simulation_started_sec"],
                           observations=deepcopy(observations))
            self.record["revision"] += 1
            self.record["simulation_clock_sec"] = now
            reply = {"status": "completed", "primitive": primitive, "command_id": command_id,
                     "run_id": self.run_id, "resource_id": self.machine["resource_id"],
                     "job_id": job_id, "selected_program": self.record["selected_program"],
                     "revision": self.record["revision"], "simulation_clock_sec": now,
                     "job_state": self.record["jobs"][job_id]["job_state"] if job_id else None,
                     "simulation_processing_quiescent": self.record["active_job"] is None,
                     "process_completed": False}
            self.record["commands"][command_id] = {"signature": signature, "reply": reply}
            return deepcopy(reply)
        except (KeyError, TypeError, ValueError):
            self.record = before
            raise

    def _account(self, job: dict, now: float, observations: dict) -> None:
        self._ready(job["part_name"], observations)
        if self.record["current_configuration"] != job["configuration"]:
            raise ValueError("Processing configuration changed during an active job")
        job["processing_active_sec"] = min(job["processing_time_sec"],
            job["processing_active_sec"] + now - job["simulation_clock_sec"])
        job["processing_fraction"] = job["processing_active_sec"] / job["processing_time_sec"]

    def observe(self, now: float, observations: dict) -> dict | None:
        """Record one verified active interval and establish completion once."""
        now = self._time(now)
        active = self.record["active_job"]
        if active is None:
            return None
        job = self.record["jobs"][active]
        self._account(job, now, observations)
        job.update(simulation_clock_sec=now, simulation_elapsed_sec=now - job["simulation_started_sec"],
                   observations=deepcopy(observations))
        self.record.update(simulation_clock_sec=now, revision=self.record["revision"] + 1)
        if job["processing_active_sec"] >= job["processing_time_sec"]:
            job.update(job_state="completed", process_completed=True, simulation_completed_sec=now)
            self.record["active_job"] = None
            return deepcopy(job)
        return None

    def interrupt(self, reason: str) -> None:
        """Retain unfinished work after shutdown without acknowledging a pause."""
        active = self.record["active_job"]
        if active is not None:
            self.record["jobs"][active].update(job_state="interrupted", interruption=reason)
            self.record.update(active_job=None, revision=self.record["revision"] + 1)

    def snapshot(self) -> dict:
        """Return a detached record for capture, validation, or worker persistence."""
        return deepcopy(self.record)
