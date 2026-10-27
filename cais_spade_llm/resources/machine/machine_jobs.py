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


def program_refusal(installed: dict, target: dict) -> str | None:
    """Return why a machine with an installed configuration refuses to select a program, or None.

    ``MachineJobs`` raises with this reason, and the primitive contracts offer only the programs it
    accepts, so both judge a program by the same rule.

    Args:
        installed: The machine's installed configuration (tool, workholding, program).
        target: The configuration of the program to select, as ``program_options`` declares it.

    Returns:
        The command's refusal reason, or None when the program can be selected.
    """
    for key in ("tool", "workholding"):
        if not installed.get(key) or not target.get(key):
            return "NEEDS_CONTEXT: installed " + key + " is unavailable"
        if target[key] != installed[key]:
            return "Program requires a different " + key
    declaration = target.get("program", {})
    if declaration.get("validated") is not True:
        return "NEEDS_CONTEXT: program has no validated effects"
    if any(installed.get(key) != value for key, value in declaration.get("required_configuration", {}).items()):
        return "Installed configuration does not satisfy the selected program"
    return None


def machine_primitive_contracts(machine: dict) -> dict[str, list[dict]]:
    """Declare what each job command needs and changes, as ``MachineJobs.command`` enforces it.

    Machine primitives are dispatched by name, so their contracts are data beside the commands. They
    name the machine's ``resource_state`` and ``selected_program`` and the part's ``current_process``,
    which carries the job and the machine that holds it; none names where a part is. The command's
    program limits are declared too: a program is selected only when the installed tool and workholding
    accept it, a job starts only with a result its selected program's validated effects establish, and
    a paused job resumes only with the program that makes its result selected again. So ``start_job``
    and ``resume_job`` have one alternative per program and result. Completion is not a command: it is
    the observed ``complete_job`` outcome, uncontrollable as in the machine automaton, once the running
    job's active processing time has passed. ``$evidence: job.*`` reads the machine's job for the part.

    Args:
        machine: The machine's scene entry, with ``resource_id``, ``program_options``,
            ``current_configuration`` and ``simulation_process.job_control``.

    Returns:
        Each primitive's alternative contracts, plus ``complete_job``: ``preconditions`` and ``effects``
        on the machine, ``product_guards`` and ``product_effects`` on the part, ``parameters`` with the
        declared values a command accepts (its programs, processes and results), and for completion
        ``controllable`` false with the ``after_sec`` it waits.
    """
    rid = machine["resource_id"]
    part = {"part_name": {"equals_from_param": "part_name"}}
    processes = machine.get("process_capabilities") or {}
    options = machine.get("program_options") or {}
    # Selecting a program keeps the installed tool and workholding, which every program must match, so the
    # programs a machine can select follow from the configuration installed at the capture.
    selectable = sorted(name for name, option in options.items()
                        if program_refusal(machine.get("current_configuration") or {}, option) is None)
    # Once selected, a program becomes the installed configuration, and a job starts only when that
    # configuration satisfies the program itself. Each such program establishes the processes and results
    # of its validated effects, within the machine's declared process capabilities.
    makes = [(program, effect["process"], effect["result"]) for program in selectable
             if program_refusal(options[program], options[program]) is None
             for effect in options[program]["program"].get("effects", [])
             if effect.get("result") in (processes.get(effect.get("process")) or {}).get("supported_results", [])]

    def own_job(**fields) -> dict:
        # A machine commands only a job in its own record, as the command refuses any other job: the
        # part's unfinished job names this machine.
        return {"current_process": {"matches": {**fields, "machine": rid}}}

    def control(before: dict, status: str, job: dict | None = None) -> dict:
        return {"preconditions": {"resource_state": before, **part},
                "product_guards": job or own_job(),
                "effects": {"resource_state": {"set": status}},
                "product_effects": {"current_process": {"set_status": status}}}

    def resumed(before: dict, **job) -> list[dict]:
        # The command resumes a job only with the job's own program selected again. The part's job record
        # names its process and result, not its program, so the program is the one that makes them: exact
        # while one selectable program makes each result, as every declared machine has it.
        variants = []
        for program, process, result in makes:
            variant = control(before, "running", own_job(**job, process=process, result=result))
            variant["preconditions"]["selected_program"] = {"equals": program}
            variants.append(variant)
        return variants

    resume = []
    # The command refuses every resume unless the machine declares resume_after_reload.
    if machine.get("simulation_process", {}).get("job_control", {}).get("resume_after_reload") is True:
        # A paused job continues where it was paused, also after its part is unloaded and loaded back in;
        # once reloaded, only the part's own record still says its job is paused.
        resume = [*resumed({"equals": "paused"}), *resumed({"equals": "loaded"}, status="paused")]
    return deepcopy({
        "select_program": [{"preconditions": {"resource_state": {"not_equals": "running"}},
                            "parameters": {"program": {"enum": selectable}},
                            "effects": {"selected_program": {"set_from_param": "program"}}}],
        "start_job": [{
            "preconditions": {"resource_state": {"equals": "loaded"}, **part,
                              "selected_program": {"equals": program}},
            "parameters": {"process": {"enum": [process]}, "result": {"enum": [result]}},
            # A paused job resumes and a stopped one cannot restart, so the part has no unfinished job here.
            "product_guards": {"current_process": {"not_matches": {"machine": rid}}},
            "effects": {"resource_state": {"set": "running"}},
            "product_effects": {"current_process": {"set": {
                "process": {"set_from_param": "process"}, "result": {"set_from_param": "result"},
                "status": "running", "machine": rid}}},
        } for program, process, result in makes],
        "pause_job": [control({"equals": "running"}, "paused")],
        "resume_job": resume,
        "stop_job": [control({"in": ["running", "paused"]}, "stopped")],
        # Pausing or stopping halts the active time, so a job completes once, and a job already running
        # at the failure completes too.
        "complete_job": [{
            "controllable": False,
            "after_sec": {"subtract": [{"$evidence": "job.processing_time_sec"},
                                       {"$evidence": "job.processing_active_sec"}]},
            "preconditions": {"resource_state": {"equals": "running"}, **part},
            "product_guards": own_job(),
            "effects": {"resource_state": {"set": "completed"}},
            "product_effects": {"current_process": {"set": None},
                                "processCompleted": [{"process": "trim", "result": {"set_from_param": "result"}}]},
        }],
    })


def job_public_state(record: dict) -> dict:
    """Read public fields from the owner record without manufacturing completion."""
    active = record["active_job"]
    jobs = list(record["jobs"].values())
    job = record["jobs"][active] if active else (jobs[-1] if jobs else {})
    return {"active_job": active, "job_state": job.get("job_state"),
            "selected_program": record["selected_program"]}


def machine_public_state(record: dict, values: dict) -> dict:
    """Read the machine's own state from its job record, beside its selected program.

    A job runs, pauses, stops or completes the machine's loaded part, so those are states of the
    machine itself; an interrupted job is the observed fault state. Without a job for the held part
    the machine keeps its modeled state.

    Args:
        record: The machine's owner job record.
        values: The machine's current values, which name the part it holds.

    Returns:
        ``selected_program`` and, when a job of the held part decides it, ``resource_state`` (and
        ``current_state`` when ``values`` carries that alias).
    """
    part_name, active = values.get("part_name"), record["active_job"]
    job = record["jobs"][active] if active else next(
        (row for row in reversed(list(record["jobs"].values())) if part_name and row.get("part_name") == part_name), {})
    state = {"selected_program": record["selected_program"]}
    if job.get("job_state") in {"running", "paused", "stopped", "interrupted", "completed"}:
        state["resource_state"] = job["job_state"]
        if "current_state" in values:
            state["current_state"] = job["job_state"]
    return state


def part_job_states(record: dict) -> dict:
    """Return each part's latest job state; a part keeps it when it leaves the machine."""
    return {row["part_name"]: row.get("job_state") for row in record["jobs"].values() if row.get("part_name")}


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
                        observations: dict, *, command_id: str,
                        allow_relative_time: bool = False) -> tuple[dict, dict]:
    """Check one command against a copied owner snapshot without live interfaces."""
    if not record:
        raise ValueError("NEEDS_CONTEXT: processing_observations")
    calculated = deepcopy(record)
    now = record.get("simulation_clock_sec")
    relative = now is None and primitive in {"start_job", "resume_job"} and allow_relative_time
    if relative:
        if (record.get("active_job") is not None or any(job.get("job_state") not in
                {"paused", "stopped", "interrupted", "completed"} for job in record["jobs"].values())):
            raise ValueError("NEEDS_CONTEXT: relative processing time requires inactive retained jobs")
        job = calculated["jobs"].get(params.get("job_id"), {})
        # Only active processing duration is used. The unobserved pause before
        # this proposal is not zero and is never reported as measured time.
        now = job.get("simulation_clock_sec", job.get("calculation_clock_sec", 0.0))
        if primitive == "resume_job":
            for name in ("started", "clock", "elapsed", "paused"):
                field = f"simulation_{name}_sec"
                job.setdefault(field, job.get(f"calculation_{name}_sec", now if name in {"started", "clock"} else 0.0))
        calculated["simulation_clock_sec"] = now
    if now is None and primitive != "select_program":
        raise ValueError("NEEDS_CONTEXT: observed simulation_clock_sec")
    owner = MachineJobs(machine, record["run_id"], calculated)
    reply = owner.command(primitive, params, command_id=command_id, now=now, observations=observations)
    projected = owner.snapshot()
    if relative:
        _retain_calculation_time(projected, params["job_id"], origin=now,
            source=fingerprint(record), command_id=command_id)
        reply.pop("simulation_clock_sec", None)
        reply.update(calculation_clock_sec=now, time_basis="relative_owner_calculation", execution_authorized=False)
        projected["commands"][command_id]["reply"] = deepcopy(reply)
    return projected, reply


def _retain_calculation_time(record: dict, job_id: str, *, origin: float,
                             source: str, command_id: str) -> None:
    job = record["jobs"][job_id]
    for name in ("started", "clock", "completed"):
        field = f"simulation_{name}_sec"
        if field in job:
            job[f"calculation_{name}_sec"] = job.pop(field)
    # Total elapsed and paused time cannot be extended across an unknown gap.
    for name in ("elapsed", "paused"):
        job.pop(f"simulation_{name}_sec", None)
    record["calculation_clock_sec"] = record.pop("simulation_clock_sec")
    record["calculation_time"] = {"kind": "relative_owner_calculation", "job_id": job_id,
        "origin_sec": origin, "source_record_fingerprint": source, "command_id": command_id,
        "absolute_time_known": False, "execution_authorized": False}


def project_job_completion(machine: dict, record: dict, duration: float,
                           observations: dict) -> dict:
    """Calculate completion over an owner duration without inventing an observed clock."""
    source = deepcopy(record)
    relative = record.get("calculation_time") if "simulation_clock_sec" not in record else None
    if relative:
        active = record.get("active_job")
        if active != relative.get("job_id") or active not in record["jobs"]:
            raise ValueError("NEEDS_CONTEXT: matching relative job calculation")
        source["simulation_clock_sec"] = record["calculation_clock_sec"]
        job = source["jobs"][active]
        for name in ("started", "clock"):
            job[f"simulation_{name}_sec"] = job[f"calculation_{name}_sec"]
        job.update(simulation_elapsed_sec=0.0, simulation_paused_sec=0.0)
    owner = MachineJobs(machine, record["run_id"], source)
    completion = owner.observe(source["simulation_clock_sec"] + duration, observations)
    if completion is None:
        raise ValueError("NEEDS_CONTEXT: no owner-supported machining completion")
    projected = owner.snapshot()
    if relative:
        _retain_calculation_time(projected, relative["job_id"], origin=relative["origin_sec"],
            source=relative["source_record_fingerprint"], command_id=relative["command_id"])
    return projected


def interrupted_pickup_conditions(machine: dict, record: dict, valuation: dict,
                                  part: str, *, run_id: str) -> dict:
    """Establish inactive processing for pickup of an interrupted loaded part.

    The simulation owner clears active_job when interrupting its worker. This
    establishes no processing progression, not geometric access, successful
    machining, a pause acknowledgement, or permission to restart the machine.

    Args:
        machine: Recorded versioned machine declaration.
        record: Processing record from the same run and machine configuration.
        valuation: Current declared machine state and part custody.
        part: Exact recorded part identifier.
        run_id: Run identity from the processing observation source.

    Returns:
        Supported guards and their source identities for detached calculation.
    """
    if (not record or not run_id or not isinstance(record.get("jobs"), dict)
            or any(not isinstance(job, dict) for job in record["jobs"].values())
            or type(record.get("revision")) is not int):
        raise KeyError("Recorded machine job identity for interrupted pickup")
    try:
        owner = MachineJobs(machine, run_id, record)
    except ValueError as exc:
        raise KeyError("Matching recorded machine declaration for interrupted pickup: " + str(exc)) from exc
    if any(field not in valuation for field in ("resource_state", "part_name", "job_state", "active_job")):
        raise KeyError("Recorded machine custody and processing state for interrupted pickup")
    public = job_public_state(owner.record)
    if any(valuation[field] != public[field] for field in ("active_job", "job_state")):
        raise KeyError("Consistent machine owner record for interrupted pickup")
    if (valuation["resource_state"] != "loaded" or valuation["part_name"] != part
            or public["job_state"] != "interrupted" or public["active_job"] is not None):
        raise ValueError("Machine owner does not establish an interrupted loaded part with inactive processing")
    jobs = [job for job in owner.record["jobs"].values() if job.get("part_name") == part
            and job.get("job_state") == "interrupted" and job.get("process_completed") is False]
    if len(jobs) != 1 or any(job.get("job_state") == "running" for job in owner.record["jobs"].values()):
        raise KeyError("Unambiguous interrupted job and inactive processing for the recorded part")
    job = jobs[0]
    if (not isinstance(job.get("job_id"), str) or owner.record["jobs"].get(job["job_id"]) != job
            or job.get("configuration_fingerprint") != fingerprint(job.get("configuration"))
            or job.get("configuration") != owner.record["current_configuration"]):
        raise KeyError("Recorded interrupted job identity and configuration")
    return {"resource_id": machine["resource_id"], "run_id": run_id,
        "model_fingerprint": owner.model_fingerprint, "record_fingerprint": fingerprint(record),
        "revision": record["revision"], "job_id": job["job_id"],
        "guards": {"resource_state": {"equals": "loaded"}, "job_state": {"equals": "interrupted"},
                   "active_job": {"equals": None}},
        "simulation_processing_quiescent": True, "process_completed": False,
        "physical_access_validated": False, "execution_authorized": False}


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
    """Declare the job commands; machining completion is observed, so it has no function."""
    functions = {name: {"function_name": name, "status": "implemented",
                        "program": {"steps": [{"id": name, "op": name}]}}
                 for name in MACHINE_PRIMITIVES}
    # Each command moves the one machine automaton; a command allowed from several states names "any".
    for name, (entry, success) in {"start_job": ("loaded", "running"), "pause_job": ("running", "paused"),
                                   "resume_job": ("any", "running"), "stop_job": ("any", "stopped")}.items():
        functions[name]["program"].update(entry_state=entry, success_state=success)
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
        refusal = program_refusal(self.record["current_configuration"], target)
        if refusal:
            raise ValueError(refusal)
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
                now: float | None, observations: dict) -> dict:
        """Apply one confirmed owner command, preserving idempotent replies.

        Args:
            primitive: Exact declared job-control primitive.
            params: Bound job or program arguments.
            command_id: Unique request identity; reuse requires identical arguments.
            now: Observed simulation time. None permits only program selection
                with established inactive jobs and establishes no elapsed time.
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
        if now is None:
            if (primitive != "select_program" or self.record["active_job"] is not None
                    or any(not isinstance(job, dict) or job.get("job_state") not in {
                        "paused", "stopped", "interrupted", "completed"}
                        for job in self.record["jobs"].values())):
                raise ValueError("NEEDS_CONTEXT: program selection without simulation time requires inactive retained jobs")
        else:
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
            if now is None:
                self.record.pop("simulation_clock_sec", None)
            else:
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
