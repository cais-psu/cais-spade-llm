"""Capture, prepare and inspect Gazebo safety evidence without admitting work.

This adapter only reads registered owners. It neither starts a runtime nor calls
the admission coordinator. A configured CCA evidence provider may supply complete
observation contracts; UI input contains candidate programs, never safety facts.
"""

from __future__ import annotations

import json
import math
import time
from copy import deepcopy
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
        monitors.append({"scope_id": scope_id, "rules": deepcopy(monitor.safety_rules),
                         "dfas": deepcopy(monitor.dfas), "current_states": deepcopy(monitor.current_states),
                         "running_aps": sorted(monitor.running_aps),
                         "resource_states": deepcopy(monitor.resource_states),
                         "tools_catalog": deepcopy(monitor.tools_catalog),
                         "resource_bindings": deepcopy(getattr(monitor, "resource_bindings", {}))})
    if admission is not None and (admission.invalid_reason or admission.ack_cursor != len(context.acknowledgements)):
        raise ValueError("CCA acknowledgement history has not caught up; capture will not synchronize it")
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
            "physical_sessions": {str(key): {name: {field: deepcopy(session[field]) for field in (
                "node", "path", "ledger", "grants", "invalid_reason", "physical_history", "time_exact")}
                                             for name, session in value.sessions.items()}
                                  for key, value in getattr(cca, "recovery_composition_admissions", {}).items()}}


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


def capture_checkpoint(runtime, cca, *, max_age: float = 2.) -> dict:  # noqa: C901 - one complete resource/part capture window
    """Capture all configured owners and reject a changed or stale capture window.

    Missing observations remain individually inspectable. No stationary interval
    or completion-ledger completeness is inferred from a single observation.
    """
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
                if physical.get('idle') is False:
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
    from cais_spade_llm.recovery_framework.live_safety_preparation import LiveSafetyPreparation
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
        from cais_spade_llm.recovery_framework.live_safety_preparation import install_live_preparation
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
