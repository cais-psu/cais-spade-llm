from __future__ import annotations

"""Fixed safety definitions remain authoritative at the actual CCA message gates."""

import asyncio
import json
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from test_predefined_safety import _SOURCE, _document
from test_recovery_admission_coordinator import _PRODUCT, _SCOPE, _Harness
from test_recovery_composition_admission import _cca, _packet

from cais_spade_llm.agents.central_controller.local_composition import Budget
from cais_spade_llm.agents.central_controller.predefined_safety_runtime import (
    bind_predefined_context,
    initialize_predefined_safety,
)
from cais_spade_llm.agents.central_controller.recovery_admission_runtime import (
    recovery_admission,
    register_recovery_composition,
)


def _fixed_cca(path=_SOURCE, **kwargs):
    cca = _cca(safety_file=path, **kwargs)
    cca.ask_llm = AsyncMock(side_effect=AssertionError("Fixed specifications cannot call the LLM"))
    assert initialize_predefined_safety(cca)
    assert not cca.predefined_safety_error, cca.predefined_safety_error
    return cca


def _fixed_harness():
    harness = _Harness()
    grounding = harness.case["grounding_inputs"]
    document = _document()
    grounding.update({key: deepcopy(document[key]) for key in ("catalog", "requirement_scopes")})
    grounding["snapshot"]["parts"]["gear_small"] = {
        "current_pose": [20, 20, 20, 0, 0, 0, 1], "contained_by": None, "stationary_until": grounding["horizon"][1],
        "processCompleted": [{"process": "assembly", "target": "Gear_Plate/Gear_Shaft_1"}],
        "processCompleted_complete": True,
        "processCompleted_evidence": {"complete": True, "source_kind": "synthetic", "checkpoint": "assembled"},
    }
    grounding["geometry"]["parts"]["gear_small"] = {
        "frame": "world", "footprint": [[-.01, .01]] * 3, "target": "Gear_Plate/Gear_Shaft_1",
    }
    harness.context["current_snapshot"] = deepcopy(grounding["snapshot"])
    return harness


def test_predefined_cca_loads_both_rules_without_native_physical_ap_fallback():
    async def scenario():
        cca = _fixed_cca()
        assert len(cca.safety_rules) == 2
        assert cca.safety_logic.predefined_metadata["mode"] == "predefined"
        assert cca.safety_logic.safety_text_sha256 == cca.safety_logic.predefined_metadata["predefined_source_sha256"]
        assert cca.safety_monitor.safety_rules == []
        assert cca.safety_monitor.dfas == {}
        assert cca.safety_rules[0]["aps"][0]["label"] == cca.safety_rules[1]["aps"][0]["label"] == "ap001"
        cca.ask_llm.assert_not_called()
    asyncio.run(scenario())


@pytest.mark.parametrize("change", ["formula", "scope", "source"])
def test_fixed_source_and_owner_context_cannot_be_replaced(tmp_path, change):
    async def scenario():
        source = tmp_path / "given.txt"
        source.write_text(_SOURCE.read_text())
        cca = _fixed_cca(source)
        context = {"composition_inputs": {"grounding_inputs": {}}}
        bind_predefined_context(cca, context)
        if change == "source":
            source.write_text(source.read_text() + " ")
        elif change == "formula":
            context["composition_inputs"]["grounding_inputs"]["catalog"]["specifications"][0]["formula"] = "G ap001"
        else:
            context["composition_inputs"]["grounding_inputs"]["requirement_scopes"].pop()
        with pytest.raises(ValueError):
            bind_predefined_context(cca, context)
        cca.ask_llm.assert_not_called()
    asyncio.run(scenario())


def test_recovery_generation_associates_fixed_rules_without_generation(monkeypatch):
    async def scenario():
        cca = _fixed_cca()
        generation = AsyncMock(side_effect=AssertionError("No replacement safety generation"))
        module = "cais_spade_llm.agents.central_controller.central_controller_agent"
        monkeypatch.setattr("cais_spade_llm.agents.central_controller.recovery_safety_generation.generate_recovery_safety_bundle", generation)
        sent = AsyncMock()
        monkeypatch.setattr(module + ".send_agent_message", sent)
        inbox = cca._RecoverySafetyGeneration()
        cca.add_behaviour(inbox)
        inbox.receive = AsyncMock(return_value=_packet(_PRODUCT, "recovery_safety_generate", {
            "product_jid": _PRODUCT, "recovery_safety_scope_id": _SCOPE, "request_id": "fixed",
        }))
        await inbox.run()
        result = json.loads(sent.call_args.args[1].body)
        assert result["ok"] and result["requires_grounded_composition"]
        assert result["rule_ids"] == [rule["id"] for rule in cca.safety_rules]
        generation.assert_not_called()
        cca.ask_llm.assert_not_called()
    asyncio.run(scenario())


def test_predefined_nominal_start_without_physical_owner_evidence_blocks():
    async def scenario():
        cca = _fixed_cca()
        inbox = cca._Monitor()
        cca.add_behaviour(inbox)
        inbox._send_decision = AsyncMock()
        inbox.receive = AsyncMock(return_value=_packet("ur5e-3@localhost", "resource_event", {
            "task_id": "nominal_place", "resource_jid": "ur5e-3@localhost", "status": "safety_check",
            "function_name": "place_approach", "params": {},
        }))
        await inbox.run()
        assert inbox._send_decision.call_args.args[2] == "block"
        assert inbox._send_decision.call_args.kwargs["recovery_composition"]["reason"] == "predefined_physical_evidence_required"
    asyncio.run(scenario())


@pytest.mark.parametrize("change", [None, "missing_gear_completion", "missing_participant"])
def test_fixed_rules_reach_joint_recovery_registration_and_exact_start(change):
    async def scenario():
        harness = _fixed_harness()
        if change == "missing_gear_completion":
            harness.case["grounding_inputs"]["snapshot"]["parts"]["gear_small"]["processCompleted"] = []
            harness.context["current_snapshot"] = deepcopy(harness.case["grounding_inputs"]["snapshot"])
        elif change == "missing_participant":
            harness.case["grounding_inputs"]["geometry"]["resources"].pop("M1")
        actor = SimpleNamespace(jid=harness.jids["KMR"],
                                prepare_recovery_composition_evidence=AsyncMock(side_effect=harness.prepare))
        cca = _fixed_cca(resource_agents=[actor], recovery_composition_context_provider=harness.provide_context,
                         allow_mock_recovery_execution=True)
        cca.plan_fsa_monitor = SimpleNamespace(current_state="checkpoint", process_event=lambda **_: None,
            _next_task_ids_from_state=lambda _: [task["task_id"] for task in harness.request["tasks"]])
        coordinator = recovery_admission(cca, _PRODUCT)
        coordinator.budget_factory = lambda: Budget(seconds=30)
        harness.coordinator = coordinator
        result = await register_recovery_composition(cca, harness.request, _PRODUCT)
        if change is not None:
            assert result["status"] != "allowed", result
            assert not coordinator.sessions
            return
        assert result["status"] == "allowed", result
        harness.registration = result
        checks = coordinator.sessions[harness.request["recovery_id"]]["analysis"]
        assert len(checks["scope"]["included_specifications"]) == 67
        inbox = cca._Monitor()
        cca.add_behaviour(inbox)
        inbox._send_decision = AsyncMock()
        event = harness.event(0)
        inbox.receive = AsyncMock(return_value=_packet(event["resource_jid"], "resource_event", event))
        await inbox.run()
        assert inbox._send_decision.call_args.args[2] == "allow", inbox._send_decision.call_args
        assert len(harness.session["grants"]) == 1
        cca.ask_llm.assert_not_called()
    asyncio.run(scenario())


def _fixed_nominal_case():
    """Use synthetic pending tasks with real CCA and nominal history consumers."""
    from threading import RLock

    from cais_spade_llm.agents.central_controller.local_composition import Action
    from cais_spade_llm.recovery_framework.environment_admission import EnvironmentAdmission

    harness = _fixed_harness()
    native = harness.context["task_monitor_context"]
    pending = {}
    for requested in harness.request["tasks"]:
        binding = native["tasks"][requested["outline_id"]]
        task = binding["task"]
        requested["primitive_steps"] = requested["params"]["primitive_steps"]
        requested["function_name"] = task["event_name"]
        requested["params"] = {"resource_id": task["resource_id"]}
        task.update(function_name=task["event_name"], event_id=requested["outline_id"],
                    parameters=deepcopy(requested["params"]))
        pending[task["task_id"]] = deepcopy(task)
    values = deepcopy(native["resources"])
    context = SimpleNamespace(
        admission_lock=RLock(), run_id="synthetic_predefined_nominal_run", revision=0,
        reservations={}, part_tracker=deepcopy(native["products"]), transitions=[],
        unavailable_resources=set(), requirements={},
        geometry=deepcopy(harness.case["grounding_inputs"]["geometry"]),
        permitted_resources=list(harness.jids), negotiations=[],
        snapshot=lambda: deepcopy(values), revisions=lambda: {key: 0 for key in values},
        pending_for=lambda identity: deepcopy(pending.get(identity)),
        pending_tasks=pending, relevant_revisions_match=lambda task: True,
        _task_reservations=lambda task, part: ["resource:" + task["resource_id"]],
        _task_participants=lambda task: [task["resource_id"]],
    )
    for task in pending.values():
        task["run_id"] = context.run_id

    class Owner:
        def __call__(self, product_jid, recovery_id):
            return harness.provide_context(product_jid, recovery_id)

        def nominal_request(self, product_jid, task):
            assert product_jid == _PRODUCT
            assert context.pending_for(task["task_id"]) == task
            return deepcopy(harness.request)

    runtime = SimpleNamespace(context=context, jids=harness.jids, stopped=False,
                              product_jid=_PRODUCT, admission=None)
    actor = SimpleNamespace(jid=harness.jids["KMR"], environment_runtime=runtime,
                            prepare_recovery_composition_evidence=AsyncMock(side_effect=harness.prepare))
    cca = _fixed_cca(resource_agents=[actor], recovery_composition_context_provider=Owner(),
                     allow_mock_recovery_execution=True)
    admission = EnvironmentAdmission(runtime, cca.safety_monitor)
    for work in harness.case["running_work"]:
        task = deepcopy(native["tasks"][work["task_id"]]["task"])
        task.update(event_id=work["task_id"], run_id=context.run_id)
        pending[task["task_id"]] = task
        admission.grants[task["task_id"]] = Action(task["task_id"], task, frozenset(), frozenset())
    coordinator = recovery_admission(cca, _PRODUCT)
    coordinator.budget_factory = lambda: Budget(seconds=30)
    harness.coordinator = coordinator
    inbox = cca._Monitor()
    cca.add_behaviour(inbox)
    inbox._send_decision = AsyncMock()
    return SimpleNamespace(harness=harness, cca=cca, actor=actor, inbox=inbox,
                           admission=admission, context=context, pending=pending, values=values)


async def _request_fixed_nominal(case, index=0, *, change=None):
    requested = case.harness.request["tasks"][index]
    task = case.pending[requested["task_id"]]
    event = {"task_id": task["task_id"], "run_id": case.context.run_id,
             "resource_jid": requested["resource_jid"], "status": "safety_check",
             "function_name": task["event_name"], "params": deepcopy(task["parameters"])}
    if change == "function":
        event["function_name"] = case.harness.request["tasks"][1]["function_name"]
    elif change == "params":
        event["params"]["resource_id"] = "ur5e-3"
    elif change == "task":
        event["task_id"] = "not_prepared"
    case.inbox.receive = AsyncMock(return_value=_packet(event["resource_jid"], "resource_event", event))
    await case.inbox.run()
    return case.inbox._send_decision.call_args


def _advance_fixed_nominal(case, time):
    from fractions import Fraction

    harness = case.harness
    for _ in range(1000):
        edges = harness.session["edges"].get(harness.session["node"], [])
        forced = [edge for edge in edges if not edge["controllable"]]
        edge = forced[0] if forced else edges[0] if len(edges) == 1 and edges[0]["kind"] == "wait" else None
        if edge is None or Fraction(edge["time_exact"]) > Fraction(str(time)):
            return
        if edge["kind"] == "task_completion":
            task = case.pending.pop(edge["task_id"])
            state = edge["task_monitor_state"]["monitors"][0]["state"]
            case.values.update(deepcopy(state["resources"]))
            case.context.part_tracker = deepcopy(state["products"])
            case.context.transitions.append({"acknowledgement": {**task, "status": "completed"},
                                             "after": deepcopy(state["resources"]),
                                             "product_after": deepcopy(state["products"])})
            case.context.revision += 1
            case.admission.synchronize()
        harness.feed(edge)
    raise AssertionError("Synthetic nominal graph walk did not terminate")


def test_predefined_nominal_complete_owner_evidence_grants_and_replays_acknowledgement():
    async def scenario():
        case = _fixed_nominal_case()
        first = case.harness.request["tasks"][0]["task_id"]
        decision = await _request_fixed_nominal(case)
        assert decision.args[2] == "allow", decision
        assert decision.kwargs["recovery_composition"]["committed"] is True
        grant = decision.kwargs["recovery_composition_grant"]
        assert grant["nominal_task"] == case.pending[first]
        assert grant["run_id"] == case.context.run_id
        assert case.admission.grants[first].task == case.pending[first]
        assert first in case.harness.session["grants"]
        analysis = case.harness.session["analysis"]
        assert analysis["status"] == "allowed"
        assert len(analysis["scope"]["included_specifications"]) == 67
        assert len(case.harness.preparations) == 5
        for preparation in case.harness.preparations:
            prepared_task = case.pending[preparation["task_id"]]
            assert preparation["function_name"] == prepared_task["event_name"]
            assert preparation["params"] == prepared_task["parameters"]

        epoch = case.admission.epoch
        path = deepcopy(case.harness.session["path"])
        duplicate = await _request_fixed_nominal(case)
        assert duplicate.args[2] == "allow", duplicate
        assert duplicate.kwargs["recovery_composition"]["reason"] == "already_admitted"
        assert case.admission.epoch == epoch
        assert case.harness.session["path"] == path

        _advance_fixed_nominal(case, 5)
        assert case.admission.ack_cursor == 1
        assert case.admission.acknowledgement_history[0]["task_id"] == first
        assert case.admission.acknowledgement_history[0]["admitted"] is True
        assert case.harness.session["nominal_acknowledgements"] == case.admission.acknowledgement_history
        assert first in case.harness.session["completed_tasks"]
        assert first not in case.admission.grants
        assert case.harness.context["current_snapshot"]["resources"]["KMR"]["held_part"] is None
        assert (await _request_fixed_nominal(case, 1)).args[2] == "allow"
        assert case.admission.ack_cursor == 1
        case.cca.ask_llm.assert_not_called()
    asyncio.run(scenario())


@pytest.mark.parametrize("change", ["task", "function", "params", "missing_evidence"])
def test_predefined_nominal_exact_dispatch_or_missing_evidence_blocks(change):
    async def scenario():
        case = _fixed_nominal_case()
        if change == "missing_evidence":
            case.harness.case["grounding_inputs"]["geometry"]["resources"].pop("M1")
        before = deepcopy(case.admission.grants)
        decision = await _request_fixed_nominal(case, change=change)
        assert decision.args[2] == "block", decision
        assert case.admission.grants == before
        assert not case.harness.coordinator.sessions
        assert not case.admission.acknowledgement_history
        case.cca.ask_llm.assert_not_called()
    asyncio.run(scenario())


@pytest.mark.parametrize("change", ["task", "function", "params", "primitive_steps"])
def test_predefined_nominal_owner_preparation_must_match_pending_task(change):
    async def scenario():
        case = _fixed_nominal_case()
        requested = case.harness.request["tasks"][0]
        if change == "task":
            requested["task_id"] = "not_prepared"
        elif change == "function":
            requested["function_name"] = case.harness.request["tasks"][1]["function_name"]
        elif change == "params":
            requested["params"]["resource_id"] = "ur5e-3"
        else:
            requested["primitive_steps"][0]["params"]["velocity"] = 999
        task = next(row for identity, row in case.pending.items()
                    if identity == case.harness.case["recovery_events"][0]["outline_id"] + "_TASK")
        event = {"task_id": task["task_id"], "run_id": case.context.run_id,
                 "resource_jid": case.harness.jids["KMR"], "status": "safety_check",
                 "function_name": task["event_name"], "params": deepcopy(task["parameters"])}
        case.inbox.receive = AsyncMock(return_value=_packet(event["resource_jid"], "resource_event", event))
        before = deepcopy(case.admission.grants)
        await case.inbox.run()
        decision = case.inbox._send_decision.call_args
        assert decision.args[2] == "block", decision
        assert case.admission.grants == before
        assert not case.harness.coordinator.sessions
        case.cca.ask_llm.assert_not_called()
    asyncio.run(scenario())


def test_other_product_cannot_restart_physical_monitor_prospectively():
    from cais_spade_llm.agents.central_controller.recovery_admission_runtime import _context

    async def scenario():
        harness = _fixed_harness()
        cca = _fixed_cca(recovery_composition_context_provider=harness.provide_context)
        cca.recovery_composition_admissions = {"previous@localhost": SimpleNamespace(sessions={
            "previous": {"product_jid": "previous@localhost", "complete": True},
        })}
        with pytest.raises(ValueError, match="existing_physical_history_requires_checkpoint"):
            _context(cca, _PRODUCT, harness.request["recovery_id"])
    asyncio.run(scenario())


@pytest.mark.parametrize("change", [None, "task", "run", "resource", "sender"])
def test_nominal_physical_grant_inbox_binds_exact_pending_task(change):
    """A CCA reply can only authorize its current resource-owned nominal task."""
    from logging import getLogger
    from cais_spade_llm.agents.resource_agent.resource_agent import ResourceAgent

    async def scenario():
        task = {"task_id": "nominal-prepared", "resource_id": "KMR",
                "event_name": "move_base", "parameters": {"destination": "Storage"}}
        runtime = SimpleNamespace(
            context=SimpleNamespace(run_id="run-current", pending_for=lambda _: deepcopy(task)),
            jids={"KMR": "kmr@localhost"},
        )
        actor = SimpleNamespace(cca_jid="cca@localhost", jid="kmr@localhost",
            agent_name="KMR", environment_runtime=runtime, logger=getLogger(__name__),
            _pending_recovery_composition_refs={}, _pending_recovery_composition_request_ids={},
            _recovery_composition_grants={}, _safety_decisions={})
        grant = {"nominal_task": deepcopy(task), "run_id": "run-current",
                 "preparation_id": "owner-prepared", "recovery_composition_ref": {"task_id": task["task_id"]}}
        sender = "cca@localhost"
        if change == "task":
            grant["nominal_task"]["parameters"]["destination"] = "M1"
        elif change == "run":
            grant["run_id"] = "run-previous"
        elif change == "resource":
            grant["nominal_task"]["resource_id"] = "ur5e-1"
        elif change == "sender":
            sender = "untrusted@localhost"
        packet = _packet(sender, "safety_decision", {
            "task_id": task["task_id"], "decision": "allow", "recovery_composition_grant": grant})
        inbox = SimpleNamespace(agent=actor, receive=AsyncMock(return_value=packet))
        await ResourceAgent._SafetyDecisionInbox.run(inbox)
        if change is None:
            assert actor._safety_decisions == {task["task_id"]: "allow"}
            assert actor._recovery_composition_grants[task["task_id"]] == grant
        else:
            assert actor._safety_decisions == {}
            assert actor._recovery_composition_grants == {}

    asyncio.run(scenario())
