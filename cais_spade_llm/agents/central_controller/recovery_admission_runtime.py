"""Connect CCA-owned recovery admission to configured evidence owners.

Agent messages identify requests and report observations. They never install an
evidence provider, compiled monitor, current DFA state, or mock execution mode.
"""

from __future__ import annotations

import asyncio
from copy import deepcopy
from threading import RLock

from cais_spade_llm.agents.central_controller.recovery_composition_admission import (
    RecoveryCompositionAdmission,
)
from cais_spade_llm.agents.shared_information.recovery_validation_protocol import (
    recovery_validation_fingerprint,
)
from cais_spade_llm.recovery_framework.environment_composition import (
    detached_checker,
    state_labels,
)


def _monitors(agent) -> list[dict]:
    if agent.safety_monitor is None:
        raise ValueError("native_monitor_unavailable")
    rows = [{"scope_id": None, "monitor": agent.safety_monitor}]
    for scope_id, scope in sorted(agent.recovery_safety_scopes.items()):
        if scope.get("status") != "ready" or scope.get("monitor") is None:
            raise ValueError("recovery_scope_monitor_unavailable")
        rows.append({"scope_id": scope_id, "monitor": scope["monitor"]})
    return rows


def _context(agent, product_jid: str, recovery_id: str) -> dict:
    provider = getattr(agent, "recovery_composition_context_provider", None)
    if provider is None:
        raise ValueError("live_recovery_evidence_unavailable")
    supplied = provider(product_jid, recovery_id)
    if not isinstance(supplied, dict):
        raise ValueError("owner_recovery_context_unavailable")
    context = deepcopy(supplied)
    from cais_spade_llm.agents.central_controller.predefined_safety_runtime import (
        bind_predefined_context,
    )

    bind_predefined_context(agent, context)
    if (getattr(agent, "predefined_safety_required", False) and context.get("physical_checkpoint") is None
            and any(session["product_jid"] != product_jid or identity != recovery_id
                    for coordinator in getattr(agent, "recovery_composition_admissions", {}).values()
                    for identity, session in coordinator.sessions.items())):
        raise ValueError("existing_physical_history_requires_checkpoint")
    # This audit belongs to the nominal admission authority, never the provider.
    context["nominal_acknowledgements"] = []
    context["nominal_run_id"] = None
    context["owner_incarnations"] = [(str(resource.jid), id(resource)) for resource in agent.resource_agents]
    native = context.get("task_monitor_context")
    if not isinstance(native, dict):
        raise ValueError("resource_owned_task_effects_unavailable")
    runtime = agent._environment_runtime_for_sender(product_jid)
    revision = {"evidence_owner": context.get("revision"),
                "predefined_safety_fingerprint": context.get("predefined_safety_fingerprint")}
    if runtime is not None:
        admission = agent._environment_admission(runtime)
        admission.synchronize()
        if admission.invalid_reason or runtime.stopped:
            raise ValueError("nominal_runtime_history_unavailable")
        if len(admission.acknowledgement_history) != admission.ack_cursor:
            raise ValueError("nominal_acknowledgement_history_unavailable")
        native.update(resources=runtime.context.snapshot(),
                      products=deepcopy(runtime.context.part_tracker),
                      contexts=deepcopy(admission.contexts), jids=dict(runtime.jids))
        context["nominal_acknowledgements"] = deepcopy(admission.acknowledgement_history)
        context["nominal_run_id"] = runtime.context.run_id
        context["nominal_running_tasks"] = [deepcopy(action.task) for action in admission.grants.values()]
        revision["runtime"] = {
            "run_id": runtime.context.run_id,
            "revision": runtime.context.revision,
            "resources": runtime.context.revisions(),
            "ack_cursor": admission.ack_cursor, "epoch": admission.epoch,
            "reservations": deepcopy(runtime.context.reservations),
            "goals": deepcopy(admission.goals),
            "running": [deepcopy(action.task) for action in admission.grants.values()],
        }
    monitors = _monitors(agent)
    native.pop("monitor", None)
    native.pop("current_states", None)
    native["monitors"] = []
    for row in monitors:
        event_ids = []
        for identity, binding in native["tasks"].items():
            task = binding["task"]
            params = task["parameters"]
            scope_id = params.get("recovery_safety_scope_id")
            if scope_id:
                applies = scope_id == row["scope_id"]
            elif row["scope_id"] is None:
                applies = True
            else:
                args = (native["jids"][task["resource_id"]],
                        task.get("function_name", task["event_name"]), params)
                applies = bool(row["monitor"]._map_task_to_aps(*args)
                               or row["monitor"]._predict_state_aps(*args))
            if applies:
                event_ids.append(identity)
        native["monitors"].append({
            "scope_id": row["scope_id"], "monitor": detached_checker(row["monitor"]),
            "current_states": deepcopy(row["monitor"].current_states), "event_ids": event_ids,
        })
    revision["monitors"] = [
        {"scope_id": row["scope_id"],
         "rules": row["monitor"].safety_rules, "dfas": row["monitor"].dfas,
         "resource_bindings": getattr(row["monitor"], "resource_bindings", {}),
         "tools_catalog": row["monitor"].tools_catalog,
         "current_states": row["monitor"].current_states,
         "running_aps": sorted(row["monitor"].running_aps),
         "resource_states": row["monitor"].resource_states,
         "history_error": row["monitor"].history_error}
        for row in monitors
    ]
    context["revision"] = recovery_validation_fingerprint(revision)
    return context


async def _prepare(agent, request: dict) -> dict:
    resources = [resource for resource in agent.resource_agents
                 if str(resource.jid).split("/", 1)[0] == request["resource_jid"]]
    if len(resources) != 1:
        return {"status": "NEEDS_CONTEXT", "reason": "resource_owner_unavailable"}
    hook = getattr(resources[0], "prepare_recovery_composition_evidence", None)
    if hook is None:
        return {"status": "NEEDS_CONTEXT", "reason": "resource_preparation_unavailable"}
    return await hook(request)


def _primitive_models(agent) -> dict:
    """Resolve pure implementations only from the registered resource owners."""
    models = {}
    for owner in agent.resource_agents:
        hook = getattr(owner, "get_recovery_safety_primitive_model", None)
        model = hook() if callable(hook) else None
        if model is not None:
            if owner.agent_name in models:
                raise ValueError("duplicate_resource_primitive_model")
            models[owner.agent_name] = model
    return models


def _nominal_start_actions(agent, product_jid: str, context: dict, record: dict):
    """Prepare exact nominal grants before mutating either monitor history."""
    nominal_actions = []
    admission = None
    runtime = agent._environment_runtime_for_sender(product_jid)
    if runtime is not None and record["kind"] == "start":
        from cais_spade_llm.agents.central_controller.local_composition import Action
        from cais_spade_llm.recovery_framework.environment_composition import task_key

        admission = agent._environment_admission(runtime)
        for identity in record.get("event_ids", []):
            task = context["tasks"][identity]["task"]
            if task.get("function_name") == "execute_recovery_macro":
                continue
            pending = runtime.context.pending_for(task["task_id"])
            if pending is None or any(pending.get(key) != task.get(key) for key in
                                      ("task_id", "resource_id", "event_id", "event_name", "parameters")):
                raise ValueError("nominal_task_identity_changed")
            claims = frozenset(runtime.context._task_reservations(pending, pending["parameters"].get("part_name") or pending.get("part_name")))
            labels = frozenset(agent.safety_monitor._map_task_to_aps(
                runtime.jids[pending["resource_id"]], pending["event_name"], pending["parameters"]))
            nominal_actions.append(Action(task_key(pending), deepcopy(pending), claims, labels))
    return admission, nominal_actions


def _commit(agent, product_jid: str, recovery_id: str, state: dict, record: dict) -> None:
    """Commit a matched acknowledgement or start, never a predicted endpoint."""
    context = _context(agent, product_jid, recovery_id)["task_monitor_context"]
    monitors = {row["scope_id"]: row["monitor"] for row in _monitors(agent)}
    histories = state.get("monitors")
    if histories is None:
        histories = [{"scope_id": None, "state": state}]
    if {row["scope_id"] for row in histories} != set(monitors):
        raise ValueError("native_monitor_scope_changed")
    admission, nominal_actions = _nominal_start_actions(agent, product_jid, context, record)
    for row in histories:
        monitor = monitors[row["scope_id"]]
        history = row["state"]
        if row["scope_id"] in record.get("reused_monitor_scopes", []):
            if monitor.current_states != history["states"]:
                raise ValueError("reused_native_monitor_state_changed")
        else:
            monitor.current_states = deepcopy(history["states"])
        labels = set()
        eligible = next(item["event_ids"] for item in context["monitors"]
                        if item["scope_id"] == row["scope_id"])
        for identity in history["running"]:
            if identity not in eligible:
                continue
            task = context["tasks"][identity]["task"]
            labels.update(monitor._map_task_to_aps(
                context["jids"][task["resource_id"]], task.get("function_name", task["event_name"]),
                {**task["parameters"], "task_id": task["task_id"]}))
        monitor.running_aps = labels
        for resource_id, values in history["resources"].items():
            jid = context["jids"][resource_id]
            monitor.resource_state_aps[jid] = set(state_labels(
                monitor, {resource_id: values}, history["products"], context["jids"],
                history["contexts"]))
            monitor.resource_states[jid] = {
                "current_state": values.get("resource_state", ""),
                "params": {**history["contexts"].get(resource_id, {}), **deepcopy(values)},
            }
    for action in nominal_actions:
        admission.grants[action.task["task_id"]] = action
        admission.epoch += 1
    plan_monitor = getattr(agent, "plan_fsa_monitor", None)
    if plan_monitor is not None:
        for transition in record.get("acknowledged_transitions", [record]):
            identities = (transition.get("event_ids", []) if transition["kind"] == "start"
                          else [transition["event_id"]] if transition["kind"] == "task_completion"
                          else [])
            for identity in identities:
                task = context["tasks"][identity]["task"]
                started = transition["kind"] == "start"
                plan_monitor.process_event(
                    event_type="start" if started else "done", task_id=task["task_id"],
                    function_name=task.get("function_name", task["event_name"]),
                    resource_jid=context["jids"][task["resource_id"]],
                    status="running" if started else "completed")


def recovery_admission(agent, product_jid: str) -> RecoveryCompositionAdmission:
    """Return the CCA coordinator sharing this run's admission transaction lock."""
    for resource in agent.resource_agents:
        resource.recovery_composition_start_guard = lambda: interacting_recovery_admission(agent)
    coordinators = getattr(agent, "recovery_composition_admissions", None)
    if coordinators is None:
        coordinators = agent.recovery_composition_admissions = {}
    if product_jid not in coordinators:
        runtime = agent._environment_runtime_for_sender(product_jid)
        lock = runtime.context.admission_lock if runtime is not None else RLock()
        coordinator = RecoveryCompositionAdmission(
            context_provider=lambda product, recovery: _context(agent, product, recovery),
            resource_evidence_provider=lambda request: _prepare(agent, request),
            primitive_models_provider=lambda: _primitive_models(agent),
            lock=lock,
            allow_mock_execution=getattr(agent, "allow_mock_recovery_execution", False),
            allow_nominal_tasks=getattr(agent, "predefined_safety_required", False),
        )
        coordinator.native_history_commit = lambda state, record: _commit(
            agent, product_jid, record["recovery_id"], state, record)
        def enabled(task):
            if task["function_name"] != "execute_recovery_macro":
                pending = runtime.context.pending_for(task["task_id"]) if runtime is not None else None
                return bool(pending is not None and not runtime.stopped
                            and pending["resource_id"] == task["resource_id"]
                            and pending["event_name"] == task["function_name"]
                            and pending["parameters"] == task["params"])
            return bool(agent.plan_fsa_monitor is not None and task["task_id"] in
                        agent.plan_fsa_monitor._next_task_ids_from_state(agent.plan_fsa_monitor.current_state))

        coordinator.start_validator = enabled
        coordinators[product_jid] = coordinator
    return coordinators[product_jid]


def interacting_recovery_admission(agent) -> bool:
    """Hold unmodeled starts while any complete-resource recovery proof is active."""
    return bool(getattr(agent, "recovery_composition_registering", False)) or any(coordinator.holds() for coordinator in
               getattr(agent, "recovery_composition_admissions", {}).values())


async def register_recovery_composition(agent, request: dict, product_jid: str) -> dict:
    """Serialize complete-resource registrations across ProductAgents."""
    lock = getattr(agent, "recovery_composition_registration_lock", None)
    if lock is None:
        lock = agent.recovery_composition_registration_lock = asyncio.Lock()
    async with lock:
        if any(product != product_jid and coordinator.holds() for product, coordinator in
               getattr(agent, "recovery_composition_admissions", {}).items()):
            return {"status": "held", "reason": "another_product_recovery_proof_is_active"}
        agent.recovery_composition_registering = True
        try:
            from cais_spade_llm.agents.central_controller.predefined_safety_runtime import (
                predefined_required,
                predefined_scope,
            )

            if predefined_required(agent):
                predefined_scope(agent, request.get("recovery_safety_scope_id"))
            return await recovery_admission(agent, product_jid).register(request, product_jid=product_jid)
        except (ValueError, KeyError, TypeError, OSError) as exc:
            return {"status": "inconclusive", "reason": str(exc)}
        finally:
            agent.recovery_composition_registering = False


def observe_recovery_event(agent, coordinator, record: dict, sender: str) -> dict:
    """Let the configured owner validate feedback before the coordinator consumes it."""
    provider = getattr(agent, "recovery_composition_context_provider", None)
    validate = getattr(provider, "observe", None)
    with coordinator.lock:
        authenticated = False
        for session in coordinator.sessions.values():
            if session["complete"]:
                continue
            task_id = record.get("task_id")
            task = session["tasks"].get(task_id)
            if task is not None:
                if (task["function_name"] != "execute_recovery_macro"
                        and sender == task["resource_jid"]
                        and record.get("resource_jid") == sender
                        and record.get("run_id") == session.get("nominal_run_id")):
                    record = deepcopy(record)
                    record["params"] = {**(record.get("params") or {}),
                                        "recovery_composition_ref": session["refs"][task_id]}
                reference = (record.get("params") or {}).get("recovery_composition_ref")
                if reference != session["refs"][task_id]:
                    continue
                authenticated = (
                    sender == task["resource_jid"]
                    and record.get("resource_jid") == sender
                    and record.get("status") != "recovery_acknowledgement"
                ) or (sender == session["product_jid"]
                      and record.get("status") == "recovery_acknowledgement")
            else:
                jids = session["native_identity"]["jids"]
                authenticated = any(
                    work["task_id"] == task_id
                    and sender == jids.get(work["resource_id"])
                    and record.get("resource_jid") == sender
                    and (session.get("nominal_run_id") is None
                         or record.get("run_id") == session["nominal_run_id"])
                    for work in session["inputs"]["running_work"])
            if authenticated:
                break
        if not authenticated:
            return {"status": "ignored", "reason": "unrecognized_recovery_observation_sender"}
        if validate is not None:
            validate(deepcopy(record), sender=sender)
        return coordinator.observe(record, sender=sender)
