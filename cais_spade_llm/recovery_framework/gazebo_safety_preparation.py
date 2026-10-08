from __future__ import annotations

"""Capture, prepare and inspect Gazebo safety evidence without admitting work.

This adapter only reads registered owners. It neither starts a runtime nor calls
the admission coordinator. A configured CCA evidence provider may supply complete
observation contracts; UI input contains candidate programs, never safety facts.
"""

import json
import math
import threading
import time
from copy import deepcopy
from fractions import Fraction
from functools import partial
from pathlib import Path
from uuid import uuid4

from cais_spade_llm.agents.central_controller.local_composition import Budget
from cais_spade_llm.agents.central_controller.offline_recovery_composition import (
    analyze_grounded_recovery_composition,
)
from cais_spade_llm.agents.central_controller.predefined_safety import (
    compile_predefined_safety,
    parse_predefined_safety,
    predefined_safety_metadata,
)
from cais_spade_llm.agents.central_controller.recovery_composition_admission import _synthetic
from cais_spade_llm.recovery_framework import ROOT, fingerprint
from cais_spade_llm.recovery_framework.environment_composition import detached_checker
from cais_spade_llm.resources.environment_models import build_environment_models
from cais_spade_llm.resources.resource_safety_preparation import (
    observation_identity,
    primitive_model_descriptors,
)

REPORT_ROOT = ROOT / "cais_spade_llm/monitor/recovery_safety_preparation"


def _monitor_rows(cca) -> list[tuple]:
    rows = [(None, cca.safety_monitor)]
    for key, scope in sorted(cca.recovery_safety_scopes.items()):
        if scope.get("status") != "ready":
            raise ValueError("CCA recovery monitor is not ready")
        rows.append((key, scope.get("monitor")))
    if any(monitor is None or monitor.history_error for _, monitor in rows):
        raise ValueError("CCA monitor history unavailable")
    return rows


def _owner_model(owner):
    hook = getattr(owner, "get_recovery_safety_primitive_model", None)
    return hook() if callable(hook) else None


def _owner_configuration(owner) -> dict:
    model = _owner_model(owner)
    return {"configuration": deepcopy(owner.recovery_safety_configuration()),
            "primitive_model": None if model is None else model.descriptor()}


def _runtime_record(runtime, cca) -> dict:
    context, admission = runtime.context, runtime.admission
    monitors = []
    for scope_id, monitor in _monitor_rows(cca):
        physical = getattr(monitor, "physical_monitor", None)
        monitors.append({"scope_id": scope_id, "rules": deepcopy(monitor.safety_rules),
                         "dfas": deepcopy(monitor.dfas), "current_states": deepcopy(monitor.current_states),
                         "running_aps": sorted(monitor.running_aps),
                         "resource_states": deepcopy(monitor.resource_states),
                         "tools_catalog": deepcopy(monitor.tools_catalog),
                         "resource_bindings": deepcopy(getattr(monitor, "resource_bindings", {})),
                         "physical_monitor": None if physical is None else {
                             "rules": deepcopy(physical.rules), "revision": physical.revision,
                             "history": deepcopy(physical.history), "invalid_reason": physical.invalid_reason,
                             "states": {key: sorted(value) for key, value in physical.states.items()}}})
    if admission is not None and (admission.invalid_reason or admission.ack_cursor != len(context.acknowledgements)):
        raise ValueError("CCA acknowledgement history has not caught up; capture will not synchronize it")
    physical_sessions = {}
    for key, coordinator in getattr(cca, "recovery_composition_admissions", {}).items():
        sessions = {}
        for name, session in coordinator.sessions.items():
            if "registration" in session:
                record = {field: deepcopy(session[field]) for field in (
                    "request", "product_jid", "tasks", "refs", "registration", "complete", "invalid_reason")}
                record["completed_tasks"] = sorted(session["completed_tasks"])
                record["grants"] = {
                    task_id: {**{field: deepcopy(work[field]) for field in (
                        "grant", "proof", "reservation", "executions", "complete")},
                        "plan_events": sorted(work.get("plan_events", set()))}
                    for task_id, work in session["grants"].items()}
            else:
                record = {field: deepcopy(session[field]) for field in (
                    "node", "path", "ledger", "grants", "invalid_reason", "physical_history", "time_exact")}
            sessions[name] = record
        physical_sessions[str(key)] = sessions
    return {"run_id": context.run_id, "revision": context.revision,
            "initial_product_states": deepcopy(getattr(context, 'initial_product_states', None)),
            "inputs_fingerprint": fingerprint(context.inputs),
            "resources": context.snapshot(), "part_tracker": deepcopy(context.part_tracker),
            "resource_revisions": {key: {"revision": row.revision, "model": fingerprint(row.model)}
                                   for key, row in context.resources.items()},
            "pending_tasks": deepcopy(context.pending_tasks), "reservations": deepcopy(context.reservations),
            "acknowledgements": deepcopy(context.acknowledgements), "monitors": monitors,
            "predefined_safety_fingerprint": getattr(cca, "predefined_safety_fingerprint", None),
            "admission": None if admission is None else {
                "ack_cursor": admission.ack_cursor, "epoch": admission.epoch,
                "goals": deepcopy(admission.goals), "contexts": deepcopy(admission.contexts),
                "running": [deepcopy(action.task) for action in admission.grants.values()]},
            "owners": {owner.agent_name: {"jid": str(owner.jid), "incarnation": id(owner),
                       "physical": deepcopy(owner.get_recovery_physical_snapshot()),
                       **_owner_configuration(owner)}
                       for owner in runtime.resource_agents},
            "physical_sessions": physical_sessions}


def _entity(reader, name: str, *, max_age: float = 2.) -> dict:
    row = reader(name, max_age=max_age)
    json.dumps(row, allow_nan=False)
    if (row.get("name") != name or row.get("frame") != "world"
            or not 0 <= time.monotonic() - row["observed_monotonic"] <= max_age
            or not math.isfinite(row["simulation_time"]) or not math.isfinite(row["simulation_stamp"])
            or not 0 <= row["simulation_time"] - row["simulation_stamp"] <= max_age):
        raise ValueError("Stale or mismatched entity observation: " + name)
    return deepcopy(row)


def _entity_reader(owners: dict):
    for owner in owners.values():
        hook = getattr(owner, "get_recovery_entity_reader", None)
        reader = hook() if callable(hook) else None
        if callable(reader):
            return reader
    return None


def _check_custody(physical: dict, symbolic: dict, owner_state: dict) -> None:
    state = physical.get("observation_state", {})
    for key in ("held_part", "gripper_state"):
        for prior in (symbolic, owner_state):
            if key in state and key in prior and state[key] != prior[key]:
                raise ValueError("Observed custody contradicts its registered owner: " + key)


def capture_checkpoint(runtime, cca, *, max_age: float = 2., model_execution: bool = False) -> dict:  # noqa: C901 - one complete resource/part capture window
    """Capture all configured owners and reject a changed or stale capture window.

    Missing observations remain individually inspectable. Model execution retains
    measured link motion and uses an explicit commanded-hold assumption; it does
    not claim that the physical scene is perfectly stationary. Detached callers
    retain the existing idle requirement by default.
    """
    if type(model_execution) is not bool:
        raise ValueError("Checkpoint model_execution must be Boolean")
    if type(max_age) not in (int, float) or not math.isfinite(max_age) or max_age <= 0:
        raise ValueError("Checkpoint maximum age must be finite and positive")
    started = time.monotonic()
    with runtime.context.admission_lock:
        before = _runtime_record(runtime, cca)
        json.dumps(before, allow_nan=False)
        scene = deepcopy(runtime.context.inputs["scene"])
        geometry = deepcopy(scene.get("safety_geometry"))
    population = set(build_environment_models(scene))
    owners = {owner.agent_name: owner for owner in runtime.resource_agents}
    if len(owners) != len(runtime.resource_agents) or set(owners) != population:
        raise ValueError("Registered owners must cover every configured resource exactly once")
    observations, unresolved = {}, []
    reader = _entity_reader(owners)
    configurations = {row["resource_id"]: row for key in ("robots", "machines") for row in scene[key]}
    configurations.update({key: value for key, value in scene.items() if key in population})
    launches = set()
    for rid in sorted(population):
        owner = owners[rid]
        row = {"resource_id": rid, "resource_jid": str(owner.jid),
               "owner_snapshot": deepcopy(owner.get_recovery_physical_snapshot()),
               "primitive_model": deepcopy(before["owners"][rid]["primitive_model"])}
        observations[rid] = row
        try:
            capture = getattr(owner, "capture_recovery_safety_state", None)
            if callable(capture):
                physical = capture(max_age=max_age)
                observation_identity(physical)
                json.dumps(physical, allow_nan=False)
                row["physical"] = deepcopy(physical)
                if physical.get('idle') is False and not model_execution:
                    unresolved.append({'resource_id':rid,'reason':'Observed link motion prevents an idle checkpoint',
                                       'moving_links':deepcopy(physical['moving_links'])})
                _check_custody(physical, before["resources"][rid], row["owner_snapshot"].get("snapshot", {}))
                observed = physical["observed_monotonic"]
                if not 0 <= time.monotonic() - observed <= max_age:
                    raise ValueError("Stale resource observation")
                if not physical.get("launch_id"):
                    raise ValueError("Missing observed launch identity")
                launches.add(physical["launch_id"])
                fixed = (geometry or {}).get("resources", {}).get(rid, {}).get("stationary_only") is True
                if (not fixed and physical.get("attachment", {}).get("observed_attachment_complete") is not True
                        and physical.get("custody_complete") is not True):
                    unresolved.append({"resource_id": rid, "reason": "Fresh attachment/custody evidence unavailable"})
            else:
                model = configurations.get(rid, {}).get("gazebo_model")
                if reader is not None and model:
                    row["gazebo_entity"] = _entity(reader, model, max_age=max_age)
                unresolved.append({"resource_id": rid, "reason": "Complete resource observation contract unavailable"})
        except (ValueError, KeyError, TypeError, AttributeError, RuntimeError) as exc:
            unresolved.append({"resource_id": rid, "reason": str(exc)})
    parts = {}
    product_geometry = runtime.context.inputs["geometry"].get("parts", {})
    for name in before["part_tracker"]:
        model = product_geometry.get('model_map', {}).get(name) or product_geometry.get(name, {}).get("model_name")
        if name == scene.get('Exit',{}).get('completed_product'):
            model = runtime.context.inputs['geometry'].get('assembly_board',{}).get('model_name',model)
        try:
            if reader is None or not model:
                raise ValueError("Configured Gazebo part observation unavailable")
            parts[name] = _entity(reader, model, max_age=max_age)
        except (ValueError, KeyError, TypeError, AttributeError, RuntimeError) as exc:
            unresolved.append({"part": name, "reason": str(exc)})
    provider = getattr(cca,'recovery_safety_preparation_provider',None)
    controller_goals = None
    if isinstance(provider,LiveSafetyPreparation):
        try:
            geometry = provider.geometry(observations,parts,unresolved)
        except (KeyError,ValueError,TypeError) as exc:
            unresolved.append({'reason':'Live configured geometry unavailable: '+str(exc)})
        try:
            controller_goals = provider.reader.idle_goals()
        except ValueError as exc:
            unresolved.append({'reason':str(exc)})
    if geometry is None:
        unresolved.append({"reason": "Configured safety_geometry and region bounds unavailable"})
    with runtime.context.admission_lock:
        if before != _runtime_record(runtime, cca):
            raise ValueError("Runtime revisions or monitor history changed during checkpoint capture")
    finished = time.monotonic()
    if finished - started > max_age or len(launches) != 1:
        unresolved.append({"reason": "Checkpoint window is stale or launch identities disagree"})
    checkpoint = {"version": 1, "captured_monotonic": finished, "capture_started_monotonic": started,
                  "launch_id": next(iter(launches)) if len(launches) == 1 else None,
                  "runtime": before, "scene": scene, "geometry": geometry,
                  "observations": observations, "parts": parts, "unresolved": unresolved}
    if model_execution:
        if not isinstance(provider, LiveSafetyPreparation) or not controller_goals:
            unresolved.append({"reason": "Modeled execution requires complete owner controller observations"})
        checkpoint["model_execution"] = True
    if isinstance(provider, LiveSafetyPreparation):
        checkpoint['controller_goals'] = controller_goals
    checkpoint["checkpoint_id"] = fingerprint(checkpoint)
    return checkpoint


def _validate_evidence(checkpoint: dict, prepared: list[dict], evidence: dict, owners: dict) -> dict:
    if evidence.get("checkpoint_id") != checkpoint["checkpoint_id"]:
        raise ValueError("Safety evidence belongs to another checkpoint")
    if evidence.get("prepared_fingerprint") != fingerprint(prepared):
        raise ValueError("Safety evidence does not bind the exact prepared programs")
    if checkpoint["unresolved"]:
        raise ValueError("Checkpoint evidence is incomplete")
    inputs = deepcopy(evidence["composition_inputs"])
    if "primitive_models" in inputs or "primitive_models" in inputs.get("grounding_inputs", {}):
        raise ValueError("Primitive model implementations come from registered owners")
    grounding = inputs["grounding_inputs"]
    if grounding["scene"] != checkpoint["scene"] or grounding["geometry"] != checkpoint["geometry"]:
        raise ValueError("Scene or configured safety geometry changed")
    population = set(build_environment_models(checkpoint["scene"]))
    if set(grounding["snapshot"]["resources"]) != population:
        raise ValueError("Safety snapshot omits configured participants")
    for rid, row in checkpoint["observations"].items():
        physical, state = row["physical"], grounding["snapshot"]["resources"][rid]
        if state != physical.get("observation_state"):
            raise ValueError("Grounded resource state lacks its exact owner observation: " + rid)
    for part, row in grounding["snapshot"]["parts"].items():
        actual = checkpoint["parts"].get(part)
        ledger = checkpoint["runtime"]["part_tracker"].get(part)
        complete = ledger.get('processCompleted_complete') if ledger is not None else None
        ledger_source = ledger.get('processCompleted_evidence') if ledger is not None else None
        if evidence.get('ledger_evidence') is not None:
            initialized = checkpoint['runtime'].get('initial_product_states')
            supplied = evidence['ledger_evidence']
            if (initialized is None or initialized != checkpoint['runtime']['part_tracker']
                    or checkpoint['runtime']['acknowledgements'] or checkpoint['runtime']['revision'] != 0
                    or supplied != {'source_kind':'validated_initialization','checkpoint':checkpoint['checkpoint_id'],
                                    'complete':True,'run_id':checkpoint['runtime']['run_id'],
                                    'initial_product_states_fingerprint':fingerprint(initialized)}):
                raise ValueError('Completion ledger is not justified by the captured initialization history')
            complete,ledger_source = True,supplied
        if (actual is None or row["current_pose"] != actual["pose"] or ledger is None
                or row.get("processCompleted") != ledger.get("processCompleted")
                or row.get("processCompleted_complete") is not complete
                or row.get("processCompleted_evidence") != ledger_source
                or "contained_by" in ledger and row.get("contained_by") != ledger["contained_by"]):
            raise ValueError("Part pose or completion ledger lacks matching observed evidence: " + part)
    running = {row["task_id"] for row in (checkpoint["runtime"].get("admission") or {}).get("running", [])}
    if running != {row["task_id"] for row in inputs["running_work"]}:
        raise ValueError("Running work does not match CCA grants")
    _validate_programs(prepared, inputs, owners)
    return inputs


def _validate_programs(prepared: list[dict], inputs: dict, owners: dict) -> None:
    by_resource = {row["resource_id"]: row for row in prepared}
    if len(by_resource) != len(prepared) or any(row["status"] != "prepared" for row in prepared):
        raise ValueError("Incomplete or ambiguous resource preparation")
    for choice in inputs["event_start_choices"]:
        checked = set()
        for program in choice["programs"]:
            rid = program["resource_id"]
            if rid not in by_resource:
                if rid not in {row["resource_id"] for row in inputs["running_work"]}:
                    raise ValueError("Program has no registered preparation")
                continue
            saved = by_resource[rid]
            if program["primitive_steps"] != saved["program"]["primitive_steps"]:
                raise ValueError("Primitive parameters or provenance changed after preparation")
            if len(program["step_results"]) != len(saved["steps"]):
                raise ValueError("Prepared primitive coverage incomplete")
            for index, (step, planned) in enumerate(zip(program["step_results"], saved["steps"], strict=True)):
                model = step["model_evidence"]
                if (step.get("source") != saved["program"]["primitive_steps"][index]["source"]
                        or model.get("preparation_id") != planned["preparation_id"]
                        or planned["preparation_id"] != fingerprint({key: value for key, value in planned.items() if key != "preparation_id"})):
                    raise ValueError("Observation evidence does not match its prepared step")
                owners[rid].validate_recovery_safety_step(deepcopy(planned), deepcopy(step))
            checked.add(rid)
        if checked != set(by_resource):
            raise ValueError("Schedule omits a prepared resource")


def _detach_monitors(runtime, cca, evidence: dict, checkpoint: dict) -> dict | None:
    with runtime.context.admission_lock:
        if _runtime_record(runtime, cca) != checkpoint["runtime"]:
            raise ValueError("Stale runtime or CCA snapshot after preparation")
        native = deepcopy(evidence.get("task_monitor_context"))
        rows = _monitor_rows(cca)
        if not any(monitor.safety_rules for _, monitor in rows):
            if native is not None:
                raise ValueError("Unexpected provider-supplied native monitors")
            return None
        if native is None:
            raise ValueError("Existing task monitor continuation unavailable")
        native.update(resources=deepcopy(checkpoint["runtime"]["resources"]),
                      products=deepcopy(checkpoint["runtime"]["part_tracker"]),
                      contexts=deepcopy((checkpoint["runtime"]["admission"] or {}).get("contexts", {})),
                      jids=deepcopy(runtime.jids))
        native.pop("monitor", None)
        native.pop("current_states", None)
        native["monitors"] = []
        for scope, actual_monitor in rows:
            monitor = detached_checker(actual_monitor)
            event_ids = []
            for identity, binding in native["tasks"].items():
                task = binding["task"]
                task_scope = task["parameters"].get("recovery_safety_scope_id")
                applies = task_scope == scope if task_scope else scope is None
                if not task_scope and scope is not None:
                    args = (native["jids"][task["resource_id"]], task.get("function_name", task["event_name"]), task["parameters"])
                    applies = bool(monitor._map_task_to_aps(*args) or monitor._predict_state_aps(*args))
                if applies:
                    event_ids.append(identity)
            native["monitors"].append({"scope_id": scope, "monitor": monitor,
                                       "current_states": deepcopy(monitor.current_states), "event_ids": event_ids})
        return native


def _reobserve(owners: dict, checkpoint: dict, max_age: float) -> None:
    for rid, row in checkpoint["observations"].items():
        fresh = owners[rid].capture_recovery_safety_state(max_age=max_age)
        if not 0 <= time.monotonic() - fresh["observed_monotonic"] <= max_age:
            raise ValueError("Stale re-observation: " + rid)
        old = row["physical"]
        if observation_identity(fresh) != observation_identity(old):
            raise ValueError("Physical checkpoint changed during planning: " + rid)
    reader = _entity_reader(owners)
    if reader is None:
        raise ValueError("Gazebo part observation owner unavailable")
    for part, old in checkpoint["parts"].items():
        if _entity(reader, old["name"], max_age=max_age)["pose"] != old["pose"]:
            raise ValueError("Gazebo part moved during preparation: " + part)
    if time.monotonic() - checkpoint["captured_monotonic"] > max_age:
        raise ValueError("Checkpoint expired during preparation")


def _selected_document(bridge) -> tuple[dict, dict]:
    selected = Path(bridge.selected_safety_file)
    if not selected.is_absolute():
        selected = ROOT / selected if len(selected.parts) > 1 else ROOT / "cais_spade_llm/specification/safety" / selected
    text = selected.read_text(encoding="utf-8")
    document = parse_predefined_safety(text)
    if document is None:
        raise ValueError("Select a predefined safety document for Prepare and check")
    compile_predefined_safety(document)
    metadata = predefined_safety_metadata(document, text)
    active = getattr(bridge.cca, "predefined_safety", None)
    if active is not None and (active != document or getattr(bridge.cca, "predefined_geometry_sha256", None) != metadata["predefined_geometry_sha256"]):
        raise ValueError("Selected definitions differ from CCA's existing physical specifications")
    return document, metadata


def _validate_request(request: dict) -> None:
    json.dumps(request, allow_nan=False)
    if not isinstance(request, dict) or set(request) != {"recovery_id", "programs"}:
        raise ValueError("Provide recovery_id and programs only; safety evidence comes from registered owners")
    if not isinstance(request["recovery_id"], str) or not request["recovery_id"] or not isinstance(request["programs"], list) or not request["programs"]:
        raise ValueError("Recovery identity and primitive programs are required")
    resources = set()
    for program in request["programs"]:
        if set(program) != {"resource_id", "primitive_steps"} or program["resource_id"] in resources:
            raise ValueError("Provide one exact primitive program per resource")
        resources.add(program["resource_id"])
        if not isinstance(program["primitive_steps"], list) or not program["primitive_steps"]:
            raise ValueError("Primitive steps are required")
        for step in program["primitive_steps"]:
            if not isinstance(step.get("params"), dict) or not step.get("primitive"):
                raise ValueError("Exact primitive and params are required")
            source = step.get("source", {})
            if any(not isinstance(source.get(key), str) or not source[key]
                   for key in ("outline_id", "des_event_id", "event_name")) or type(source.get("step_index")) is not int or source["step_index"] < 0:
                raise ValueError("Each primitive requires its original event and step references")


def _prepare_programs(owners: dict, request: dict, checkpoint: dict, report: dict) -> None:
    for program in request["programs"]:
        rid = program["resource_id"]
        hook = getattr(owners[rid], "prepare_recovery_safety_program", None)
        if not callable(hook):
            raise ValueError("Non-dispatching preparation unavailable for " + rid)
        prepared = hook(deepcopy(program), deepcopy(checkpoint))
        json.dumps(prepared, allow_nan=False)
        if (prepared.get("program") != program or prepared.get("program_fingerprint") != fingerprint(program)
                or prepared.get("checkpoint_id") != checkpoint["checkpoint_id"]):
            raise ValueError("Owner preparation changed the requested program")
        report["prepared_programs"].append(prepared)
        if prepared.get("status") != "prepared":
            report["unresolved"].append({"resource_id": rid, "reason": prepared.get("reason", "Incomplete preparation")})


def _resource_support(owners: dict, checkpoint: dict, prepared: list) -> list:
    result = []
    by_resource = {row["resource_id"]: row for row in prepared}
    for rid in sorted(owners):
        saved = by_resource.get(rid, {})
        result.append({"resource_id": rid,
            "observation_status": "NEEDS_CONTEXT" if any(row.get("resource_id") == rid for row in checkpoint["unresolved"]) else "captured",
            "primitive_model": deepcopy(checkpoint["runtime"]["owners"][rid]["primitive_model"]),
            "preparation_status": saved.get("status", "not_requested"), "reason": saved.get("reason", "")})
        support_error = getattr(getattr(owners[rid], 'recovery_safety_observer', None), 'support_error', None)
        if support_error:
            result[-1].update(preparation_support='NEEDS_CONTEXT', reason=support_error)
    return result


def prepare_and_check(bridge, request: dict, *, output_root: Path = REPORT_ROOT,
                      max_age: float = 2., budget: Budget | None = None) -> dict:
    """Save a read-only preparation attempt and detached CCA analysis.

    Args:
        bridge: Existing SystemBridge, used read-only to locate registered owners.
        request: Recovery identity and exact per-resource primitive programs.
        output_root: Directory for a new, immutable attempt; no prior files replaced.
        max_age: Maximum capture age in monotonic seconds.
        budget: Composition budget; the normal 20,000 states / 2 seconds is retained.

    Returns:
        Persisted evidence with dispatch_authorized=False, even if analysis allows
        a continuation. Synthetic providers are accepted only on mock resources.
    """
    report = {"version": 1, "attempt_id": uuid4().hex, "status": "NEEDS_CONTEXT",
              "dispatch_authorized": False, "request": None, "prepared_programs": [],
              "unresolved": [], "analysis": None}
    try:
        _validate_request(request)
        report["request"] = deepcopy(request)
        document, metadata = _selected_document(bridge)
        report["predefined_safety"] = metadata
        cca = bridge.cca
        launch_reader = getattr(bridge, "_simulation_launch_key", lambda: ())
        launch_identity = deepcopy(launch_reader())
        report["simulation_launch_identity"] = launch_identity
        runtimes = {id(owner.environment_runtime): owner.environment_runtime
                    for owner in bridge.resource_agents if getattr(owner, "environment_runtime", None) is not None}
        if cca is None or len(runtimes) != 1:
            raise ValueError("One active registered EnvironmentRuntime and CCA are required")
        runtime = next(iter(runtimes.values()))
        install_live_preparation(runtime,cca)
        provider = getattr(cca,'recovery_safety_preparation_provider',None)
        initialize = getattr(provider,'initialize',None)
        if callable(initialize):
            initialize()
        candidate = getattr(provider,'configuration',{}).get('candidates',{}).get(request['recovery_id'])
        if candidate is not None:
            report['supplied_candidate'] = deepcopy(candidate)
        checkpoint = capture_checkpoint(runtime, cca, max_age=max_age)
        report["checkpoint"] = checkpoint
        report["unresolved"].extend(checkpoint["unresolved"])
        owners = {row.agent_name: row for row in runtime.resource_agents}
        report["resource_support"] = _resource_support(owners, checkpoint, [])
        validate_idle = getattr(provider, 'validate_idle', None)
        if callable(validate_idle):
            validate_idle(checkpoint)
        if checkpoint['unresolved']:
            raise ValueError('Complete fresh checkpoint evidence is required before planning')
        _prepare_programs(owners, request, checkpoint, report)
        report["resource_support"] = _resource_support(owners, checkpoint, report["prepared_programs"])
        provider = getattr(cca, "recovery_safety_preparation_provider", None)
        if not callable(provider):
            raise ValueError("CCA-owned complete motion, stationary coverage and observation evidence unavailable")
        evidence = provider(checkpoint=deepcopy(checkpoint), prepared=deepcopy(report["prepared_programs"]),
                            request=deepcopy(request))
        json.dumps(evidence, allow_nan=False)
        report["supplied_evidence"] = deepcopy(evidence)
        if (evidence.get("synthetic") is not False or _synthetic(evidence)) and not (
                getattr(cca, "allow_mock_recovery_execution", False)
                and all(owner.execution_mode == "dry_run" for owner in owners.values())):
            raise ValueError("Synthetic safety evidence cannot be used with Gazebo resources")
        inputs = _validate_evidence(checkpoint, report["prepared_programs"], evidence, owners)
        for key in ("catalog", "requirement_scopes"):
            if inputs["grounding_inputs"].get(key, document[key]) != document[key]:
                raise ValueError("Evidence attempts to replace predefined specifications")
            inputs["grounding_inputs"][key] = deepcopy(document[key])
        inputs["task_monitor_context"] = _detach_monitors(runtime, cca, evidence, checkpoint)
        if any(checkpoint["runtime"]["physical_sessions"].values()) and evidence.get("physical_checkpoint") is None:
            raise ValueError("Existing physical history requires replayable observed evidence")
        if evidence.get("physical_checkpoint") is None and evidence.get("physical_rule_activation") != "prospective":
            raise ValueError("Physical history or an explicit prospective checkpoint is required")
        inputs["physical_checkpoint"] = deepcopy(evidence.get("physical_checkpoint"))
        revalidate = getattr(provider,'revalidate',None)
        if callable(revalidate):
            report['revalidated_checkpoint'] = revalidate(checkpoint,max_age=max_age)
        else:
            _reobserve(owners, checkpoint, max_age)
        primitive_models = {}
        for rid, owner in owners.items():
            model = _owner_model(owner)
            if model is not None:
                primitive_models[rid] = model
        report["primitive_models"] = primitive_model_descriptors(primitive_models)
        report["analysis"] = analyze_grounded_recovery_composition(
            **inputs, budget=budget or Budget(), primitive_models=primitive_models)
        with runtime.context.admission_lock:
            if _runtime_record(runtime, cca) != checkpoint["runtime"]:
                raise ValueError("Runtime or monitor history changed while checking")
        _, current_metadata = _selected_document(bridge)
        if current_metadata != metadata:
            raise ValueError("Selected safety definitions changed while checking")
        if launch_reader() != launch_identity:
            raise ValueError("Gazebo launch changed during preparation")
        report["status"] = report["analysis"]["status"]
    except (ValueError, KeyError, TypeError, AttributeError, OSError, RuntimeError) as exc:
        report["status"] = "NEEDS_CONTEXT"
        report["unresolved"].append({"reason": str(exc)})
    output_root.mkdir(parents=True, exist_ok=True)
    directory = output_root / report["attempt_id"]
    directory.mkdir()
    report["artifact_path"] = str(directory / "result.json")
    report["fingerprint"] = fingerprint(report)
    Path(report["artifact_path"]).write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def controller_goal_identity(observations: dict | None, *, model_execution: bool = False) -> dict | None:
    """Compare controller incarnations and command revisions, retaining raw stamps separately."""
    if type(model_execution) is not bool:
        raise ValueError("Controller model_execution must be Boolean")
    if observations is None:
        return None
    changing = {"sequence", "simulation_time", "observed_monotonic", "positions", "received_monotonic",
                "velocities", "stationary_samples", "maximum_observed_update_period",
                "maximum_observed_position_error"}
    if model_execution:
        changing.add("observed_stationary")
    return {name: {key: deepcopy(value) for key, value in row.items() if key not in changing}
            for name, row in observations.items()}


class GazeboSafetyReader:
    """Read one physics-thread snapshot through the registered observation service."""

    def __init__(self, configuration: dict):
        import rclpy
        from rclpy.executors import SingleThreadedExecutor
        from std_srvs.srv import Trigger

        self.configuration = deepcopy(configuration)
        self._context = rclpy.context.Context()
        rclpy.init(context=self._context)
        self._node = rclpy.create_node("recovery_safety_reader", context=self._context)
        self._executor = SingleThreadedExecutor(context=self._context)
        self._executor.add_node(self._node)
        self._thread = threading.Thread(target=self._executor.spin, daemon=True)
        self._thread.start()
        self._client = self._node.create_client(Trigger, configuration["observation_service"])
        self._type = Trigger
        self._cached = None
        self._lock = threading.RLock()
        self._snapshot_lock = threading.RLock()
        from action_msgs.msg import GoalStatusArray
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

        self._goal_status = {}
        self._subscriptions = []
        qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        for topic in configuration["controller_status_topics"]:
            self._subscriptions.append(
                self._node.create_subscription(
                    GoalStatusArray,
                    topic,
                    partial(self._record_goals, topic),
                    qos,
                )
            )

    def _record_goals(self, topic, message, info=None):
        """Retain status without claiming publisher identity when ROS omits it."""
        with self._lock:
            self._goal_status[topic] = {
                "received_monotonic": time.monotonic(),
                "publisher_gid": [int(value) for value in info.publisher_gid] if info is not None else [],
                "goals": [
                    {"id": [int(value) for value in row.goal_info.goal_id.uuid], "status": row.status}
                    for row in message.status_list
                ],
            }

    def idle_goals(self):
        """Require a connected authoritative status publisher and no active goal."""
        if self.configuration.get("controller_state_services"):
            return self._idle_service_goals()
        with self._lock:
            for topic in self.configuration["controller_status_topics"]:
                publishers = self._node.get_publishers_info_by_topic(topic)
                if (
                    len(publishers) != 1
                    or topic not in self._goal_status
                    or list(publishers[0].endpoint_gid) != self._goal_status[topic]["publisher_gid"]
                ):
                    raise ValueError("Complete controller goal observations unavailable: " + topic)
                if any(row["status"] in (1, 2, 3) for row in self._goal_status[topic]["goals"]):
                    raise ValueError(
                        "An active controller goal prevents idle preparation: " + topic
                    )
            return deepcopy(self._goal_status)

    def _idle_service_goals(self):
        sources = self.configuration["controller_state_services"]
        covered = [topic for row in sources.values() for topic in row["covers"]]
        if (len(covered) != len(set(covered))
                or set(covered) != set(self.configuration["controller_status_topics"])):
            raise ValueError("Controller observation owners do not cover all configured endpoints")
        result = {}
        for service, configuration in sources.items():
            client = self._node.create_client(self._type, service)
            try:
                reply = self._call(client, self._type.Request())
                if not reply.success:
                    raise ValueError("Controller state unavailable: " + service + ": " + reply.message)
                row = json.loads(reply.message)
            finally:
                self._node.destroy_client(client)
            stamp = row.get("simulation_time")
            now = self.snapshot()["simulation_time"]
            if (row.get("version") != 1 or not row.get("instance_id")
                    or type(stamp) not in (int, float) or not math.isfinite(stamp)
                    or abs(now - stamp) > 2):
                raise ValueError("Stale or incomplete controller state: " + service)
            if (row.get("has_active_goal") is not False or row.get("has_pending_goal") is not False
                    or row.get("holding") is not True):
                raise ValueError("Controller is not holding without active goals: " + service)
            if any(row.get(key) != value for key, value in configuration.get("required_values", {}).items()):
                raise ValueError("Controller observation contract is unavailable: " + service)
            result[service] = {**row, "observed_monotonic": time.monotonic(),
                               "covers": deepcopy(configuration["covers"])}
        return result

    def _call(self, client, request, timeout=2.0):
        if not client.wait_for_service(timeout_sec=timeout):
            raise ValueError("Gazebo read-only service unavailable: " + client.srv_name)
        future = client.call_async(request)
        completed = threading.Event()
        future.add_done_callback(lambda _: completed.set())
        if not completed.wait(timeout):
            future.cancel()
            raise ValueError("Gazebo read-only service timed out: " + client.srv_name)
        result = future.result()
        if result is None:
            raise ValueError("Gazebo read-only service returned no observation")
        return result

    def parameter(self, node: str, name: str):
        """Read a declared string parameter without changing controller state."""
        from rcl_interfaces.srv import GetParameters

        client = self._node.create_client(GetParameters, node.rstrip("/") + "/get_parameters")
        try:
            result = self._call(client, GetParameters.Request(names=[name]))
            if len(result.values) != 1 or result.values[0].type != 4:
                raise ValueError("Required string parameter unavailable: " + node + "/" + name)
            return result.values[0].string_value
        finally:
            self._node.destroy_client(client)

    def snapshot(self, *, max_age=2.0, refresh=False):
        """Return a finite stamped physics snapshot, retaining one capture window."""
        with self._snapshot_lock:
            if (
                refresh
                or self._cached is None
                or time.monotonic() - self._cached["observed_monotonic"] > max_age
            ):
                result = self._call(self._client, self._type.Request())
                if not result.success:
                    raise ValueError(result.message)
                row = json.loads(result.message)
                json.dumps(row, allow_nan=False)
                if (
                    row.get("version") != 1
                    or row.get("attachment_complete") is not True
                    or not row.get("instance_id")
                ):
                    raise ValueError("Incomplete Gazebo attachment snapshot")
                row["observed_monotonic"] = time.monotonic()
                self._cached = row
            return deepcopy(self._cached)

    def entity(self, name: str, *, max_age=2.0):
        """Return one model pose from the complete stamped physics observation."""
        row = self.snapshot(max_age=max_age)
        model = row["models"][name]
        return {
            "name": name,
            "frame": "world",
            "pose": deepcopy(model["pose"]),
            "simulation_time": row["simulation_time"],
            "simulation_stamp": row["simulation_time"],
            "observed_monotonic": row["observed_monotonic"],
            "source": "GETRECOVERYSTATE",
            "instance_id": row["instance_id"],
        }

    def close(self):
        """Release this read-only ROS context without touching the simulation."""
        self._executor.shutdown(timeout_sec=2.0)
        self._node.destroy_node()
        self._context.shutdown()
        self._thread.join(timeout=2.0)


def _links(record, configuration):
    links = record["models"][configuration["model"]]["links"]
    prefix = configuration.get("link_prefix", "")
    result = {name: row for name, row in links.items() if name.startswith(prefix)}
    if not result:
        raise ValueError("Configured resource links are missing")
    return result


def _envelope(links, pose):
    boxes = [row["bounds"] for link in links.values() for row in link["collisions"]]
    if not boxes or any(row is None for row in boxes):
        raise ValueError("Configured resource collision geometry is missing")
    return [
        [
            math.nextafter(min(row[i][0] for row in boxes) - pose[i], -math.inf),
            math.nextafter(max(row[i][1] for row in boxes) - pose[i], math.inf),
        ]
        for i in range(3)
    ]


def _configured_static_part_geometry(record: dict, declaration: dict, part: dict) -> dict:
    """Enclose configured static fixtures relative to their observed carrier.

    An empty carrier link has no collision envelope of its own. Constituent
    models supply geometry only while they and the carrier are observed static
    in the same physics snapshot.
    """
    if set(declaration) != {"models", "requires_static"} or declaration["requires_static"] is not True:
        raise ValueError("Configured constituent geometry requires an explicit static contract")
    names = declaration["models"]
    if (not isinstance(names, list) or not names or any(not isinstance(name, str) or not name for name in names)
            or len(set(names)) != len(names) or part["name"] in names):
        raise ValueError("Configured constituent model identities must be distinct")
    carrier = record["models"][part["name"]]
    if (carrier["static"] is not True or carrier["pose"] != part["pose"]
            or part.get("instance_id") != record["instance_id"]
            or part.get("simulation_time") != record["simulation_time"]):
        raise ValueError("Constituent geometry requires the same observed static carrier")
    links = {}
    for name in names:
        model = record["models"][name]
        if model["static"] is not True:
            raise ValueError("Configured constituent model is not static: " + name)
        # One empty constituent must not be hidden by another usable envelope.
        _envelope(model["links"], part["pose"])
        links.update({name + "/" + key: value for key, value in model["links"].items()})
    return {"frame": "world", "footprint": _envelope(links, part["pose"])}


class GazeboResourceObservation:
    """Resource-owned idle observation support, selected entirely by configuration."""

    def __init__(self, provider, owner, configuration):
        self.provider, self.owner = provider, owner
        self.configuration = deepcopy(configuration)
        self.motion_configuration = None
        self.support_error = None

    @property
    def reader(self):
        """Expose the registered entity reader without a robot-controller dependency."""
        return self.provider.reader.entity

    def capture(self, *, max_age=2.0, native=None):
        """Capture physical facts; record observed motion without claiming idle coverage."""
        record = self.provider.reader.snapshot(max_age=max_age)
        config, rid = self.configuration, self.owner.agent_name
        links = _links(record, config)
        moving = {
            name: {key: row[key] for key in ("linear_speed", "angular_speed")}
            for name, row in links.items()
            if row["linear_speed"] > config["stationary_linear_speed_max"]
            or row["angular_speed"] > config["stationary_angular_speed_max"]
        }
        state = deepcopy(self.provider.runtime.context.snapshot()[rid])
        model = record["models"][config["model"]]
        reference = config.get("reference_link")
        if native is not None:
            # Fixed URDF links can be lumped out of Gazebo's physics tree. The
            # controller owns the stamped tool transform; body bounds still
            # come from the physics owner, not from the transform endpoint.
            pose = native["current_pose"]
            if reference in links and math.dist(pose[:3], links[reference]["pose"][:3]) > 0.001:
                raise ValueError("Controller and physics-thread tool observations disagree")
        else:
            pose = links[reference]["pose"] if reference else model["pose"]
        attached = [
            row
            for row in record["attachments"]
            if any(
                row[f"model{side}"] == config["model"] and row[f"link{side}"] in links
                for side in (1, 2)
            )
        ]
        # This slice does not silently reinterpret assembly containment as grasp.
        if attached:
            raise ValueError(
                "Idle motion preparation requires observed empty resource custody: " + rid
            )
        if state.get("held_part") is not None:
            raise ValueError("Symbolic custody disagrees with the observed empty attachment set")
        state["current_pose"] = deepcopy(pose)
        state["contained_parts"] = sorted(
            part
            for part, row in self.provider.runtime.context.part_tracker.items()
            if row.get("location") == rid
        )
        if "held_part" in state:
            grip = config["gripper"]
            width = model["joints"][grip["joint"]]["position"]
            matches = [
                name for name in ("open", "closed") if abs(width - grip[name]) <= grip["tolerance"]
            ]
            if len(matches) != 1:
                raise ValueError("Gripper observation is not a supported configured state")
            state["gripper_state"] = matches[0]
            state["grasp_transform"] = None
        if native is not None:
            state["joint_positions"] = deepcopy(native["joint_positions"])
            state["joint_names"] = deepcopy(native["joint_names"])
        physical = {
            **deepcopy(native or {}),
            "source": "registered owner / GETRECOVERYSTATE",
            "model_static": model["static"],
            "frame": "world",
            "current_pose": deepcopy(pose),
            "observation_state": state,
            "launch_id": self.provider.launch_id,
            "observed_monotonic": record["observed_monotonic"],
            "simulation_time": record["simulation_time"],
            "custody_complete": True,
            "attachment": {
                "model_name": None,
                "observed_attachment_complete": True,
                "instance_id": record["instance_id"],
                "revision": record["attachment_revision"],
            },
            "stationary_contract": deepcopy(config["stationary_contract"]),
            "idle": not moving,
            "moving_links": moving,
            "geometry_source": {
                "service": "GETRECOVERYSTATE",
                "model": config["model"],
                "links": sorted(links),
                "configuration": fingerprint(config),
            },
            "component_bounds": [
                {**deepcopy(collision), "link": name}
                for name, link in links.items() for collision in link["collisions"]
            ],
            "footprint": _envelope(links, pose),
        }
        return physical

    def stationary_coverage(self, intervals: list, checkpoint: dict) -> dict:
        """Bind the owner's declared hold contract to explicit intervals in this check."""
        contract = self.configuration.get('stationary_contract')
        kinds = ('idle_commanded_hold', 'static_body_and_idle_containment')
        supported = [{'kind': kind, 'requires_no_running_tasks': True,
                      'requires_no_active_goals': True, 'future_execution_tracking': 'not_established'}
                     for kind in kinds]
        if contract not in supported:
            raise ValueError('Supported owner stationary coverage is unavailable: '+self.owner.agent_name)
        physical = checkpoint['observations'][self.owner.agent_name]['physical']
        if physical.get('idle') is not True:
            raise ValueError('Owner has not observed an idle checkpoint: '+self.owner.agent_name)
        if contract['kind'] == 'static_body_and_idle_containment' and physical.get('model_static') is not True:
            raise ValueError('Static coverage requires observed static equipment: '+self.owner.agent_name)
        return {'contract':deepcopy(contract),'resource_id':self.owner.agent_name,
                'checkpoint_id':checkpoint['checkpoint_id'], 'intervals':deepcopy(intervals),
                'configuration_fingerprint':fingerprint(self.configuration)}


class LiveSafetyPreparation:
    """CCA-owned assembler for an idle supplied candidate; never an admission proof."""

    def __init__(self, runtime, cca, configuration):
        self.runtime, self.cca, self.configuration = runtime, cca, deepcopy(configuration)
        self.reader = None
        self.launch_id = None
        self.observers = {}
        self.last_prepared = None
        self.last_execution_coverage = None
        self.model_execution = False

    def initialize(self, *, model_execution: bool = False):
        """Initialize read-only support before capturing the atomic runtime record."""
        if type(model_execution) is not bool:
            raise ValueError("Preparation model_execution must be Boolean")
        if self.configuration.get("provider") != "gazebo_idle_continuous_v1":
            raise ValueError("Unregistered live preparation provider")
        mode_changed = self.model_execution != model_execution
        self.model_execution = model_execution
        population = set(build_environment_models(self.runtime.context.inputs["scene"]))
        owners = {owner.agent_name: owner for owner in self.runtime.resource_agents}
        if set(self.configuration["resources"]) != population or set(owners) != population:
            raise ValueError("Live preparation configuration must cover every configured resource")
        if self.reader is None:
            self.reader = GazeboSafetyReader(self.configuration)
        self.launch_id = self.reader.parameter(
            self.configuration["launch_parameter_node"], "launch_id"
        )
        if not self.launch_id or self.launch_id == "recovery_framework":
            raise ValueError("Observed simulation launch identity is unavailable")
        record = self.reader.snapshot(refresh=True)
        for rid, owner in owners.items():
            if rid not in self.observers:
                self.observers[rid] = GazeboResourceObservation(
                    self, owner, self.configuration["resources"][rid]
                )
                owner.recovery_safety_observer = self.observers[rid]
            hook = getattr(owner, "configure_recovery_safety_observer", None)
            if callable(hook) and (mode_changed or self.observers[rid].motion_configuration is None):
                try:
                    hook(self.observers[rid], record)
                    self.observers[rid].support_error = None
                except (KeyError, ValueError, RuntimeError, TypeError) as exc:
                    self.observers[rid].motion_configuration = None
                    self.observers[rid].support_error = str(exc)

    def require_execution_coverage(
        self, checkpoint: dict, resource_id: str, *, prepared: dict | None = None,
    ) -> None:
        """Require owner evidence for the exact modeled Gazebo execution.

        Model-based execution checks a registered owner's interpolation and
        explicit stationary assumptions. It does not promote them to physical
        containment guarantees. Detached idle preparation remains read-only.

        Args:
            checkpoint: Complete observed scene captured before planning.
            resource_id: Exact resource responsible for the requested motion.
            prepared: Its retained native program, when available.

        Raises:
            ValueError: The owner cannot bind the complete modeled execution.
        """
        self.last_execution_coverage = None
        if resource_id not in self.observers or checkpoint.get("unresolved"):
            raise ValueError("live_execution_tracking_unverified: physical owner observation unavailable")
        if prepared is not None:
            from cais_spade_llm.resources.resource_safety_preparation import (
                PreparedRobotEvidence,
            )

            owner = self.observers[resource_id].owner
            provider = getattr(owner, "recovery_composition_evidence_provider", None)
            if isinstance(provider, PreparedRobotEvidence) and provider.owner is owner:
                self.last_execution_coverage = provider.prepare_execution_coverage(
                    prepared=prepared, checkpoint=checkpoint,
                )
                if (checkpoint.get("model_execution") is True
                        and self.last_execution_coverage.get("status") == "prepared"
                        and self.last_execution_coverage.get("model_execution_verified") is True
                        and self.last_execution_coverage.get("physical_execution_verified") is False
                        and self.last_execution_coverage.get("prepared_fingerprint") == fingerprint(prepared)
                        and self.last_execution_coverage.get("checkpoint_fingerprint") == fingerprint(checkpoint)):
                    return
                raise ValueError(
                    "live_execution_tracking_unverified: " + self.last_execution_coverage["reason"]
                )
        raise ValueError(
            "live_execution_tracking_unverified: gazebo_idle_continuous_v1 does not certify "
            "controller tracking, future stationary containment, or failure stopping bounds"
        )

    def geometry(self, observations, parts, unresolved):
        """Freeze configured regions and observed configured collision envelopes."""
        record = self.reader.snapshot()
        result = {
            "dimension": 3,
            "frame": "world",
            "regions": deepcopy(self.configuration["regions"]),
            "resources": {},
            "parts": {},
        }
        for rid, row in observations.items():
            if "physical" not in row:
                continue
            physical = row["physical"]
            shape = {
                "frame": "world",
                "footprint": physical["footprint"],
                "component_bounds": deepcopy(physical["component_bounds"]),
            }
            if "held_part" not in physical["observation_state"]:
                shape["stationary_only"] = True
            result["resources"][rid] = shape
        for name, row in parts.items():
            model = record["models"][row["name"]]
            try:
                declaration = self.configuration.get("part_geometry", {}).get(name)
                if declaration is None:
                    result["parts"][name] = {
                        "frame": "world",
                        "footprint": _envelope(model["links"], row["pose"]),
                    }
                else:
                    result["parts"][name] = _configured_static_part_geometry(record, declaration, row)
                    row["geometry_source"] = {
                        "service": "GETRECOVERYSTATE",
                        "instance_id": record["instance_id"],
                        "simulation_time": record["simulation_time"],
                        "declaration": deepcopy(declaration),
                        "declaration_fingerprint": fingerprint(declaration),
                        "models_fingerprint": fingerprint({key: record["models"][key] for key in declaration["models"]}),
                    }
                targets = self.runtime.context.inputs["geometry"].get("parts", {}).get("assembly_target_map", {})
                if name in targets:
                    target = targets[name]
                    if not isinstance(target, str) or not target:
                        raise ValueError("Configured assembly target is unavailable: " + name)
                    result["parts"][name]["target"] = target
            except (ValueError, KeyError, TypeError) as exc:
                unresolved.append({"part": name, "reason": str(exc)})
        return result

    def validate_idle(self, checkpoint):
        """Require an observed idle checkpoint and the known initialization ledger."""
        goals = self.reader.idle_goals()
        if controller_goal_identity(checkpoint.get("controller_goals")) != controller_goal_identity(goals):
            raise ValueError("Controller goals changed or were not captured at the checkpoint")
        runtime = checkpoint["runtime"]
        admission = runtime.get("admission") or {}
        if admission.get("running") or runtime["reservations"]:
            raise ValueError(
                "Prepare and check requires an idle checkpoint without running work or reservations"
            )
        if any(monitor["running_aps"] for monitor in runtime["monitors"]):
            raise ValueError("Active task monitor work is outside idle preparation")
        if any(runtime["physical_sessions"].values()):
            raise ValueError(
                "Existing physical history requires a compatible continuous checkpoint"
            )
        if (
            runtime["acknowledgements"]
            or self.runtime.context.part_tracker != self.runtime.context.initial_product_states
        ):
            raise ValueError(
                "Live ledger replay beyond the known initialization checkpoint is unavailable"
            )
        if runtime["revision"] != 0:
            raise ValueError("The initialized preparation checkpoint has changed")

    def revalidate(self, checkpoint, *, max_age=2.0):
        """Recapture after planning without changing the preparation's original start."""
        record = self.reader.snapshot(refresh=True)
        self.reader.idle_goals()
        observations = {}
        for rid, observer in self.observers.items():
            fresh = observer.owner.capture_recovery_safety_state(max_age=max_age)
            prior = checkpoint["observations"][rid]["physical"]
            for field in ("launch_id", "attachment", "custody_complete", "stationary_contract"):
                if fresh[field] != prior[field]:
                    raise ValueError("Physical identity or custody changed during planning: " + rid)
            # No observation-error envelope is currently provided. A tolerance
            # here would accept an initial state absent from the prepared proof.
            if fresh["current_pose"] != prior["current_pose"]:
                raise ValueError("Resource moved during preparation: " + rid)
            old, new = deepcopy(prior["observation_state"]), deepcopy(fresh["observation_state"])
            old.pop("current_pose", None)
            new.pop("current_pose", None)
            old_joints, new_joints = old.pop("joint_positions", []), new.pop("joint_positions", [])
            if (
                old != new
                or old_joints != new_joints
                or fresh["footprint"] != prior["footprint"]
                or fresh.get("component_bounds") != prior.get("component_bounds")
            ):
                raise ValueError("Resource checkpoint changed during preparation: " + rid)
            observations[rid] = fresh
        parts = {}
        for name, old in checkpoint["parts"].items():
            fresh = self.reader.entity(old["name"], max_age=max_age)
            if fresh["pose"] != old["pose"]:
                raise ValueError("Part moved during preparation: " + name)
            declaration = self.configuration.get("part_geometry", {}).get(name)
            if declaration is not None:
                current = _configured_static_part_geometry(record, declaration, fresh)
                source = old.get("geometry_source", {})
                if (current != checkpoint["geometry"]["parts"][name]
                        or source.get("declaration_fingerprint") != fingerprint(declaration)
                        or source.get("models_fingerprint") != fingerprint({
                            key: record["models"][key] for key in declaration["models"]})):
                    raise ValueError("Configured constituent geometry changed during preparation: " + name)
            parts[name] = fresh
        result = {
            "observations": observations,
            "parts": parts,
            "original_checkpoint_id": checkpoint["checkpoint_id"],
        }
        result["checkpoint_id"] = fingerprint(result)
        return result

    def __call__(self, *, checkpoint, prepared, request):
        """Assemble one finite, fully bound schedule from exact prepared programs."""
        self.validate_idle(checkpoint)
        if checkpoint["unresolved"] or any(row["status"] != "prepared" for row in prepared):
            raise ValueError("Live checkpoint or primitive preparation is incomplete")
        configured = self.configuration["candidates"].get(request["recovery_id"])
        if configured is None:
            raise ValueError("Supplied candidate has no registered finite start schedule")
        schedules = configured["start_offsets"]
        if set(schedules) != {row["resource_id"] for row in prepared}:
            raise ValueError("Candidate starts do not cover its supplied programs")
        programs, events, ends = [], [], []
        for program in prepared:
            rid = program["resource_id"]
            cursor = Fraction(str(schedules[rid]))
            if cursor < 0:
                raise ValueError("Candidate start offsets must be nonnegative")
            results = []
            previous = None
            for index, step in enumerate(program["steps"]):
                command = program["program"]["primitive_steps"][index]
                if step.get("observation_status") != "prepared":
                    raise ValueError("Continuous prepared motion is unavailable: " + rid)
                end = cursor + Fraction(step["joint_trajectory"]["duration_ns"], 1_000_000_000)
                source = command["source"]
                if previous != source["outline_id"]:
                    events.append(
                        {
                            "outline_id": source["outline_id"],
                            "des_event_id": source["des_event_id"],
                            "event_name": source["event_name"],
                            "resource_id": rid,
                            "predecessors": [] if previous is None else [previous],
                            "primitive_step_indices": [],
                        }
                    )
                    previous = source["outline_id"]
                events[-1]["primitive_step_indices"].append(index)
                evidence = {
                    "frame": "world",
                    "preparation_id": step["preparation_id"],
                    "joint_trajectory": deepcopy(step["joint_trajectory"]),
                    "continuous_motion": deepcopy(step["continuous_motion"]),
                }
                results.append(
                    {
                        "primitive": command["primitive"],
                        "resolved_params": deepcopy(command["params"]),
                        "start_time": float(cursor),
                        "end_time": float(end),
                        "success": True,
                        "source": deepcopy(source),
                        "model_evidence": evidence,
                    }
                )
                cursor = end
            ends.append(cursor)
            programs.append(
                {
                    "resource_id": rid,
                    "primitive_steps": deepcopy(program["program"]["primitive_steps"]),
                    "step_results": results,
                }
            )
        horizon = [0.0, float(max(ends))]
        stationary = {rid: [deepcopy(horizon)] for rid in checkpoint["observations"]}
        for program in programs:
            a, b = program["step_results"][0]["start_time"], program["step_results"][-1]["end_time"]
            stationary[program["resource_id"]] = ([[0.0, a]] if a else []) + (
                [[b, horizon[1]]] if b < horizon[1] else []
            )
        coverage = {rid:self.observers[rid].stationary_coverage(intervals,checkpoint)
                    for rid,intervals in stationary.items()}
        resources = {
            rid: deepcopy(row["physical"]["observation_state"])
            for rid, row in checkpoint["observations"].items()
        }
        parts = {}
        ledger_evidence = {
            "source_kind": "validated_initialization",
            "checkpoint": checkpoint["checkpoint_id"],
            "complete": True,
            "run_id": checkpoint["runtime"]["run_id"],
            "initial_product_states_fingerprint": fingerprint(
                self.runtime.context.initial_product_states
            ),
        }
        for name, ledger in checkpoint["runtime"]["part_tracker"].items():
            parts[name] = {
                **deepcopy(ledger),
                "current_pose": deepcopy(checkpoint["parts"][name]["pose"]),
                "contained_by": ledger.get("location")
                if ledger.get("location") in resources
                else None,
                "stationary_until": horizon[1],
                "processCompleted_complete": True,
                "processCompleted_evidence": deepcopy(ledger_evidence),
            }
        snapshot = {"resources": resources, "parts": parts}
        # These records preserve outstanding work; the candidate does not complete it.
        snapshot["nominal_tasks"] = deepcopy(checkpoint["runtime"]["pending_tasks"])
        inputs = {
            "grounding_inputs": {
                "scene": deepcopy(checkpoint["scene"]),
                "programs": programs,
                "snapshot": snapshot,
                "geometry": deepcopy(checkpoint["geometry"]),
                "horizon": horizon,
                "stationary": stationary,
            },
            "recovery_events": events,
            "running_work": [],
            "event_start_choices": [
                {
                    "id": request["recovery_id"],
                    "starts": {
                        event["outline_id"]: next(
                            step["start_time"]
                            for program in programs
                            for step in program["step_results"]
                            if step["source"]["outline_id"] == event["outline_id"]
                        )
                        for event in events
                    },
                    "programs": deepcopy(programs),
                    "stationary": deepcopy(stationary),
                }
            ],
            "completion": {
                "resources": {
                    rid: {"held_part": None} for rid in resources if "held_part" in resources[rid]
                },
                "parts": {
                    name: {
                        "processCompleted": deepcopy(row["processCompleted"]),
                        "contained_by": row["contained_by"],
                    }
                    for name, row in parts.items()
                },
            },
        }
        self.last_prepared = fingerprint(prepared)
        return {
            "synthetic": False,
            "checkpoint_id": checkpoint["checkpoint_id"],
            "prepared_fingerprint": fingerprint(prepared),
            "composition_inputs": inputs,
            "physical_rule_activation": "prospective",
            "ledger_evidence": ledger_evidence,
            "stationary_contracts": coverage,
        }


def install_live_preparation(runtime, cca):
    """Register only the trusted configured provider, without activating a proof."""
    configuration = runtime.context.inputs["scene"].get("safety_preparation")
    if configuration is None:
        return
    previous = getattr(cca, "recovery_safety_preparation_provider", None)
    if previous is None:
        cca.recovery_safety_preparation_provider = LiveSafetyPreparation(
            runtime, cca, configuration
        )
    elif isinstance(previous, LiveSafetyPreparation) and previous.runtime is not runtime:
        if previous.reader is not None:
            previous.reader.close()
        cca.recovery_safety_preparation_provider = LiveSafetyPreparation(
            runtime, cca, configuration
        )


def build_supplied_candidate(bridge, candidate_id: str) -> dict:
    """Resolve a configured mock motion recipe against actual observed retreat poses."""
    runtimes = {
        id(owner.environment_runtime): owner.environment_runtime
        for owner in bridge.resource_agents
        if getattr(owner, "environment_runtime", None) is not None
    }
    if len(runtimes) != 1 or bridge.cca is None:
        raise ValueError("One registered runtime and CCA are required")
    runtime = next(iter(runtimes.values()))
    install_live_preparation(runtime, bridge.cca)
    provider = bridge.cca.recovery_safety_preparation_provider
    if not isinstance(provider, LiveSafetyPreparation):
        raise ValueError("Configured live preparation is unavailable")
    provider.initialize()
    checkpoint = capture_checkpoint(runtime, bridge.cca)
    recipe = provider.configuration["candidates"][candidate_id]
    programs = []
    for rid, targets in recipe["poses"].items():
        if "physical" not in checkpoint["observations"][rid]:
            raise ValueError("Candidate targets need an observed resource pose: " + rid)
        start = checkpoint["observations"][rid]["physical"]["current_pose"]
        identity = candidate_id + "/" + rid
        steps = []
        for index, target in enumerate((targets["approach"], targets["entry"], start[:3])):
            steps.append(
                {
                    "primitive": "move_cartesian",
                    "params": dict(zip(("x", "y", "z"), target, strict=True)),
                    "source": {
                        "outline_id": identity,
                        "des_event_id": identity,
                        "event_name": identity,
                        "step_index": index,
                    },
                }
            )
        programs.append({"resource_id": rid, "primitive_steps": steps})
    return {"recovery_id": candidate_id, "programs": programs}
