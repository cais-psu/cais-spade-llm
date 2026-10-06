"""Real CCA message routing for generated recovery admission."""

from __future__ import annotations

import asyncio
import json
import logging
from copy import deepcopy
from fractions import Fraction
from threading import Lock, RLock
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from spade.message import Message

from cais_spade_llm.agents.central_controller.central_controller_agent import (
    CentralControllerAgent,
)
from cais_spade_llm.agents.central_controller.online_safety_monitor import OnlineSafetyMonitor


def _packet(sender, kind, data):
    packet = Message(to="cca@localhost", sender=sender)
    packet.set_metadata("type", kind)
    packet.body = json.dumps(data)
    return packet


def _cca(**kwargs):
    cca = CentralControllerAgent("cca@localhost", "none", name="cca", **kwargs)
    cca.safety_monitor = OnlineSafetyMonitor({}, [])
    return cca


def test_robot_registered_execution_requires_both_owner_and_controller_support():
    from cais_spade_llm.agents.resource_agent.robot_agent import RobotAgent

    async def scenario():
        robot = object.__new__(RobotAgent)
        robot._controller = SimpleNamespace()
        robot._robot_motion_lock = Lock()
        execute = Mock(return_value={"success": True})
        robot.recovery_composition_evidence_provider = SimpleNamespace(execute_step=execute)
        assert not robot.supports_recovery_composition_execution()
        result = await robot.execute_recovery_macro(
            "registered", [], recovery_composition_ref={"recovery_id": "bound"})
        assert result["status"] == "failed:recovery_composition_grant"
        execute.assert_not_called()
        robot._controller.execute_prepared_recovery_step = Mock()
        assert robot.supports_recovery_composition_execution()
        args = {"primitive": "move_cartesian", "params": {"x": 0.}, "task_id": "exact-task",
                "step_index": 2, "grant": {"preparation_id": "exact-preparation"}, "owner": None}
        robot._robot_motion_lock.acquire()
        try:
            with pytest.raises(ValueError, match="active motion"):
                await robot.execute_recovery_composition_step(**args)
        finally:
            robot._robot_motion_lock.release()
        execute.assert_not_called()
        assert await robot.execute_recovery_composition_step(**args) == {"success": True}
        assert execute.call_args.kwargs == {"resource_agent": robot, **{key: value for key, value in args.items()
                                                                      if key != "owner"}}
        assert not robot._robot_motion_lock.locked()
        execute.side_effect = ValueError("observed deviation")
        with pytest.raises(ValueError, match="observed deviation"):
            await robot.execute_recovery_composition_step(**args)
        assert not robot._robot_motion_lock.locked()

    asyncio.run(scenario())


def test_unregistered_generated_task_blocks_before_nominal_event_matching():
    async def run():
        runtime = SimpleNamespace(product_jid="product@localhost",
                                  jids={"KMR": "recovery-resource-8@localhost"})
        cca = _cca(resource_agents=[SimpleNamespace(environment_runtime=runtime)])
        behaviour = cca._Monitor()
        cca.add_behaviour(behaviour)
        behaviour._send_decision = AsyncMock()
        params = {"product_jid": "product@localhost", "start_safety_mode": "cca_check",
                  "recovery_safety_scope_id": "KMR_storage_interruption_scope"}
        packet = _packet("recovery-resource-8@localhost", "resource_event", {
            "task_id": "KMR_STORAGE_INTERRUPTION_SEQ3", "status": "safety_check",
            "resource_jid": "recovery-resource-8@localhost",
            "function_name": "execute_recovery_macro", "params": params,
        })
        # A missing evidence reference must block without touching nominal pending_for.
        runtime.context = SimpleNamespace(admission_lock=RLock())
        behaviour.receive = AsyncMock(return_value=packet)
        await behaviour.run()
        assert behaviour._send_decision.call_args.args[2] == "block"
        assert "recovery_composition_ref" in behaviour._send_decision.call_args.kwargs[
            "recovery_composition"]["reason"]
    asyncio.run(run())


def test_plan_registration_without_live_evidence_returns_block(monkeypatch, tmp_path):
    async def run():
        cca = _cca()
        cca.safety_logic = None
        behaviour = cca._PlanValidation()
        cca.add_behaviour(behaviour)
        replies = []

        async def send(_, message, **kwargs):
            replies.append(json.loads(message.body))

        monkeypatch.setattr(
            "cais_spade_llm.agents.central_controller.central_controller_agent.send_agent_message", send)
        packet = _packet("product@localhost", "plan_safety_check", {
            "request_id": "KMR_storage_interruption_registration",
            "product_jid": "product@localhost", "fsa": {"A": {"x0": "checkpoint", "Tr": []}},
            "recovery_composition_request": {
                "recovery_id": "KMR_storage_interruption", "tasks": [],
                "recovery_safety_scope_id": "KMR_storage_interruption_scope",
            },
        })
        behaviour.receive = AsyncMock(return_value=packet)
        await behaviour.run()
        assert replies[-1]["ok"] is False
        assert replies[-1]["recovery_composition"]["reason"] == "live_recovery_evidence_unavailable"
        assert cca.plan_fsa_monitor.fsa == json.loads(packet.body)["fsa"]
    monkeypatch.chdir(tmp_path)
    asyncio.run(run())


def test_registering_same_scope_does_not_reset_native_history():
    async def run():
        cca = _cca()
        result = {"ok": True, "recovery_safety_scope_id": "KMR_storage_interruption_scope",
                  "rules": [], "rule_dfas": {}}
        assert cca._register_recovery_safety_scope_result(result)
        original = cca.recovery_safety_scopes[result["recovery_safety_scope_id"]]["monitor"]
        original.history_error = {"reason": "observed_deviation"}
        original.running_aps.add("ap001")
        assert cca._register_recovery_safety_scope_result(deepcopy(result))
        assert cca.recovery_safety_scopes[result["recovery_safety_scope_id"]]["monitor"] is original
        assert original.history_error == {"reason": "observed_deviation"}
        assert original.running_aps == {"ap001"}
        changed = {**result, "rules": [{"id": "changed"}]}
        assert not cca._register_recovery_safety_scope_result(changed)
        assert original.history_error == {"reason": "observed_deviation"}
    asyncio.run(run())


def test_complete_resource_registration_serializes_different_products():
    from cais_spade_llm.agents.central_controller.recovery_admission_runtime import (
        register_recovery_composition,
    )

    async def scenario():
        coordinators = {}
        for product in ("product1@localhost", "product2@localhost"):
            coordinator = SimpleNamespace(active=False)
            coordinator.holds = lambda owner=coordinator: owner.active

            async def register(request, *, product_jid, owner=coordinator):
                await asyncio.sleep(0)
                owner.active = True
                return {"status": "allowed"}

            coordinator.register = AsyncMock(side_effect=register)
            coordinators[product] = coordinator
        cca = SimpleNamespace(resource_agents=[], recovery_composition_admissions=coordinators)
        results = await asyncio.gather(*(
            register_recovery_composition(cca, {}, product) for product in coordinators))
        assert [row["status"] for row in results] == ["allowed", "held"]
        assert results[1]["reason"] == "another_product_recovery_proof_is_active"
        assert sum(coordinator.register.await_count for coordinator in coordinators.values()) == 1
        assert not cca.recovery_composition_registering

    asyncio.run(scenario())


def test_existing_plan_start_gate_is_additional_to_composition():
    from test_recovery_admission_coordinator import _harness

    async def scenario():
        harness = _harness()
        await harness.register()
        harness.coordinator.start_validator = lambda task: False
        before = deepcopy(harness.session["path"])
        result = await harness.check(0)
        assert result["status"] == "held"
        assert result["reason"] == "plan_task_not_enabled"
        assert harness.session["path"] == before
        assert not harness.session["grants"]
        harness.coordinator.start_validator = lambda task: True
        assert (await harness.check(0))["status"] == "allowed"

    asyncio.run(scenario())


def test_pa_ra_cca_mock_execution_restores_storage_and_holds_conflicting_start(monkeypatch, tmp_path):  # noqa: C901, PLR0915
    from test_recovery_admission_coordinator import _PRODUCT, _SCOPE, _harness

    from cais_spade_llm.agents.central_controller.local_composition import Budget
    from cais_spade_llm.agents.central_controller.recovery_admission_runtime import (
        recovery_admission,
    )
    from cais_spade_llm.agents.intelligent_product.product_agent import ProductAgent
    from cais_spade_llm.agents.intelligent_product.product_recovery_controller import (
        ProductRecoveryController,
    )
    from cais_spade_llm.agents.resource_agent.resource_agent import ResourceAgent
    from cais_spade_llm.recovery_framework.kmr_agent import KMRResourceAgent

    async def scenario():  # noqa: C901, PLR0915
        harness = _harness()
        program = harness.case["event_start_choices"][1]["programs"][0]
        calls, packets, completed = [], [], set()
        clock = {"time": 0}

        async def worker_run(request):
            index = len(calls)
            row = program["step_results"][index]
            assert request["primitive"] == row["primitive"]
            for key, value in row["resolved_params"].items():
                assert request["primitive_parameters"][key] == value
            calls.append(deepcopy(request))
            clock["time"] = row["end_time"]
            output = deepcopy(row["model_evidence"]["outputs"])
            return {"status": "completed", "result": output, "primitive_results": [
                {"primitive": row["primitive"], "status": "completed", "result": output}]}

        actor = KMRResourceAgent(harness.jids["KMR"], "none", worker=SimpleNamespace(run=worker_run),
                                 cca_jid="cca@localhost")
        actor._primitive_state = deepcopy(harness.context["current_snapshot"]["resources"]["KMR"])
        actor._primitive_state.pop("resource_type", None)
        actor.workflow_custody = {"grasp_transform": deepcopy(actor._primitive_state["grasp_transform"]),
                                 "attached": True}
        actor._kmr_execution_request = {}
        actor.recovery_composition_evidence_provider = SimpleNamespace(
            prepare=lambda **kwargs: harness.prepare(kwargs["request"]))
        nodes = [{"id": task["task_id"], "type": "task", "status": "pending",
                  "resource_id": task["resource_id"], "resource_jid": task["resource_jid"],
                  "function_name": task["function_name"], "recovery_outline_id": task["outline_id"],
                  "event_name": event["event_name"], "params": deepcopy(task["params"])}
                 for task, event in zip(harness.request["tasks"], harness.case["recovery_events"], strict=True)]
        sequence = {"recovery_sequence_id": harness.request["recovery_id"], "validation_policy": "validated",
                    "recovery_safety_scope_id": _SCOPE, "recovery_task_ids": [node["id"] for node in nodes],
                    "pending_nominal_task_ids": deepcopy(harness.request["pending_nominal_task_ids"])}
        transitions = []
        for index, task in enumerate(nodes):
            for offset, suffix in enumerate(("start", "done")):
                transitions.append({"from": f"checkpoint{2 * index + offset}",
                                    "to": f"checkpoint{2 * index + offset + 1}",
                                    "event": task["id"] + "." + suffix, "task_id": task["id"]})
        running_id = harness.case["running_work"][0]["task_id"]
        transitions.extend({"from": f"checkpoint{index}", "to": f"checkpoint{index}",
                            "event": running_id + ".done", "task_id": running_id} for index in range(9))
        fsa = {"A": {"x0": "checkpoint0", "X": [f"checkpoint{index}" for index in range(9)],
                     "Xm": ["checkpoint8"], "Tr": transitions}}
        product = SimpleNamespace(
            jid=_PRODUCT, cca_jid="cca@localhost", logger=logging.getLogger(__name__),
            runtime_recovery={"validation_policy": "validated", "active_recovery_sequence": sequence},
            _runtime_recovery_context={}, _active_recovery_sequence=lambda: sequence,
            process_planner=SimpleNamespace(nodes=nodes, global_fsa=fsa),
            _enrich_observed_pose_recovery_params=deepcopy,
            _runtime_recovery_session_validation_policy=lambda: "validated",
            _build_runtime_plan_context=lambda: {},
            _should_ignore_stale_recovery_macro_ack=lambda **kwargs: False,
            _tracked_part_name_for_task=lambda node: None,
            _handle_recovery_macro_ack=AsyncMock(return_value=True),
            task_states={}, execution_timeline=[],
        )
        controller = ProductRecoveryController(product)
        product._dispatch_params_for_task_node = controller._dispatch_params_for_task_node
        product._build_recovery_composition_request = controller._build_recovery_composition_request
        payload = ProductAgent._build_plan_validation_payload(product, request_id="register-KMR")
        harness.request = deepcopy(payload["recovery_composition_request"])
        for task in harness.request["tasks"]:
            native_task = harness.context["task_monitor_context"]["tasks"][task["outline_id"]]["task"]
            native_task.update(parameters=deepcopy(task["params"]), function_name=task["function_name"])

        class Owner:
            notifications = 0

            def __call__(self, product_jid, recovery_id):
                return harness.provide_context(product_jid, recovery_id)

            def observe(self, record, *, sender):
                self.notifications += 1
                if record.get("status") == "recovery_acknowledgement":
                    assert sender == _PRODUCT
                    assert record["observations"]["status"] == "completed"
                    completed.add(record["task_id"])
                self.advance(clock["time"])

            def advance(self, until):
                session = harness.session
                node = session["node"]
                for _ in range(1000):
                    edges = session["edges"].get(node, [])
                    forced = [edge for edge in edges if not edge["controllable"]]
                    edge = forced[0] if forced else edges[0] if len(edges) == 1 and edges[0]["kind"] == "wait" else None
                    if edge is None or Fraction(edge["time_exact"]) > Fraction(str(until)):
                        break
                    if (edge["kind"] == "task_completion"
                            and edge["task_id"] in sequence["recovery_task_ids"]
                            and edge["task_id"] not in completed):
                        break
                    node = edge["target"]
                    if edge["kind"] in {"observation", "task_completion"}:
                        observation = {"record_id": edge["edge_id"], "kind": edge["kind"],
                                       "time_exact": edge["time_exact"]}
                        if edge["kind"] == "observation":
                            observation["observation"] = deepcopy(edge["observation"])
                        else:
                            observation.update(task_id=edge["task_id"], status="completed")
                        harness.context["observations"].append(observation)
                target = session["nodes"][node]
                harness.context["time_exact"] = str(until)
                harness.context["current_snapshot"] = {key: deepcopy(target["state"][key])
                                                        for key in ("resources", "parts")}
                native = target["task_monitor_state"]["monitors"][0]["state"]
                for key in ("resources", "products", "contexts"):
                    harness.context["task_monitor_context"][key] = deepcopy(native[key])

        owner = Owner()
        cca = _cca(resource_agents=[actor], recovery_composition_context_provider=owner,
                   allow_mock_recovery_execution=True)
        cca._register_recovery_safety_scope_result({"ok": True, "recovery_safety_scope_id": _SCOPE,
                                                   "rules": [], "rule_dfas": {}})
        coordinator = recovery_admission(cca, _PRODUCT)
        coordinator.budget_factory = lambda: Budget(seconds=30)
        harness.coordinator = coordinator
        plan_inbox, monitor_inbox = cca._PlanValidation(), cca._Monitor()
        cca.add_behaviour(plan_inbox)
        cca.add_behaviour(monitor_inbox)

        async def send(inbox, packet, **kwargs):
            packet.sender = str(inbox.agent.jid)
            data = json.loads(packet.body)
            kind = packet.get_metadata("type")
            packets.append((kind, deepcopy(data)))
            if kind == "resource_event":
                monitor_inbox.receive = AsyncMock(return_value=packet)
                await monitor_inbox.run()
            elif kind == "safety_decision":
                decision_inbox = SimpleNamespace(agent=actor, receive=AsyncMock(return_value=packet))
                await ResourceAgent._SafetyDecisionInbox.run(decision_inbox)
            elif kind == "plan_safety_result":
                assert data["ok"], data
                harness.registration = data["recovery_composition"]
                sequence["recovery_composition"] = deepcopy(harness.registration)
            elif kind == "ack":
                ack_inbox = SimpleNamespace(agent=product, receive=AsyncMock(return_value=packet))
                await ProductAgent._AckInbox.run(ack_inbox)

        for module in ("central_controller.central_controller_agent", "resource_agent.resource_agent",
                       "intelligent_product.product_agent"):
            monkeypatch.setattr("cais_spade_llm.agents." + module + ".send_agent_message", send)
        registration = _packet(_PRODUCT, "plan_safety_check", payload)
        plan_inbox.receive = AsyncMock(return_value=registration)
        await plan_inbox.run()
        assert harness.registration["status"] == "allowed"
        forged = harness.event(0)
        forged["status"] = "failed"
        monitor_inbox.receive = AsyncMock(return_value=_packet("other@localhost", "resource_event", forged))
        await monitor_inbox.run()
        assert owner.notifications == 0
        assert not harness.session["invalid_reason"]

        async def dispatch(index):
            task = nodes[index]
            message = _packet(_PRODUCT, "task", {"task_id": task["id"], "instruction": {
                "function_name": task["function_name"], "params": controller._dispatch_params_for_task_node(task)}})
            inbox = actor._TaskInbox()
            actor.add_behaviour(inbox)
            inbox.receive = AsyncMock(return_value=message)
            await asyncio.wait_for(inbox.run(), 10)
            await asyncio.sleep(0)
            assert not harness.session["invalid_reason"], harness.session["invalid_reason"]

        await dispatch(0)
        assert len(calls) == 5
        await dispatch(1)
        assert len(calls) == 10
        await dispatch(2)
        assert len(calls) == 10  # The unsafe t=10 start never reaches a primitive executor.
        assert packets[-1][1].get("status") == "blocked"
        clock["time"] = 12
        owner.advance(12)
        running = harness.case["running_work"][0]
        notification = _packet(harness.jids[running["resource_id"]], "resource_event", {
            "task_id": running["task_id"], "resource_jid": harness.jids[running["resource_id"]],
            "function_name": running["event_name"], "params": {}, "status": "completed"})
        monitor_inbox.receive = AsyncMock(return_value=notification)
        await monitor_inbox.run()
        await dispatch(2)
        assert len(calls) == 14, [data for kind, data in packets if kind == "safety_decision"][-1]
        await dispatch(3)
        assert len(calls) == 22
        clock["time"] = 26
        owner.advance(26)
        final_observation = harness.event(3)
        final_observation["status"] = "recovery_primitive_observed"
        notification = _packet(harness.jids["KMR"], "resource_event", final_observation)
        monitor_inbox.receive = AsyncMock(return_value=notification)
        await monitor_inbox.run()
        assert harness.session["complete"]
        assert cca.plan_fsa_monitor.current_state == "checkpoint8"
        assert set(sequence["recovery_task_ids"]) <= cca.plan_fsa_monitor.completed_task_ids
        assert actor._primitive_state["held_part"] == "KET8_Square_8mm"
        snapshot = harness.context["current_snapshot"]
        assert snapshot["parts"]["KET4_Square_4mm"]["contained_by"] == "assembly_board-v1"
        assert snapshot["parts"]["KET4_Square_4mm"]["processCompleted"] == [{"process": "trim", "result": "square"}]
        assert snapshot["parts"]["KET8_Square_8mm"]["processCompleted"] == []
        assert harness.request["pending_nominal_task_ids"] == sequence["pending_nominal_task_ids"]
        assert all(harness.case["grounding_inputs"]["snapshot"]["task_statuses"][task_id] == "pending"
                   for task_id in sequence["pending_nominal_task_ids"])

    monkeypatch.chdir(tmp_path)
    asyncio.run(scenario())
