from __future__ import annotations

"""Real CCA message routing for generated recovery admission."""

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
        # This harness is explicitly synthetic; actual KMR dispatch requires its
        # live prepared owner. Macro grant/parameter checks remain exercised.
        actor.execute_recovery_composition_step = ResourceAgent.execute_recovery_composition_step.__get__(actor)
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

    from cais_spade_llm.agents.shared_information.llm_agent import LlmAgent

    monkeypatch.setattr(LlmAgent, "_TOOLS_CATALOG", [])
    monkeypatch.chdir(tmp_path)
    asyncio.run(scenario())


def test_physical_admission_routes_configured_live_owner_without_graph_fallback():
    from cais_spade_llm.agents.central_controller.recovery_admission_runtime import (
        observe_recovery_event,
        physical_admission,
    )

    owner = SimpleNamespace(observe=Mock(return_value={"status": "observed"}))
    cca = SimpleNamespace(live_safety_runtime=owner, predefined_safety_required=True)
    assert physical_admission(cca, "product@localhost") is owner
    record = {"task_id": "exact-task", "status": "completed"}
    assert observe_recovery_event(cca, owner, record, "resource@localhost") == {"status": "observed"}
    owner.observe.assert_called_once_with(record, sender="resource@localhost")


def test_physical_admission_missing_live_owner_cannot_authorize_selected_safety():
    from cais_spade_llm.agents.central_controller.recovery_admission_runtime import (
        physical_admission,
    )

    async def scenario():
        cca = SimpleNamespace(predefined_safety_required=True)
        gate = physical_admission(cca, "product@localhost")
        result = await gate.check({"task_id": "work"}, sender="resource@localhost")
        assert result["status"] == "inconclusive"
        assert result["reason"] == "live_physical_safety_owner_unavailable"
        assert result["committed"] is False
        assert gate.holds()

    asyncio.run(scenario())


def test_outline_physical_safety_deferral_allows_planning_without_permission(monkeypatch):
    async def scenario():
        cca = _cca()
        cca.predefined_safety_required = True
        cca._wait_for_safety_monitor_ready = AsyncMock(return_value=True)
        before = deepcopy(cca.safety_monitor.current_states)
        sent = AsyncMock()
        validator = Mock(side_effect=AssertionError("An outline cannot supply physical AP evidence"))
        module = "cais_spade_llm.agents.central_controller.central_controller_agent"
        monkeypatch.setattr(module + ".send_agent_message", sent)
        monkeypatch.setattr(module + ".validate_outline_macro_recovery_safety", validator)
        inbox = cca._RecoveryOutlineSafetyValidation()
        cca.add_behaviour(inbox)
        inbox.receive = AsyncMock(return_value=_packet("product@localhost",
            "recovery_outline_safety_validate", {
                "product_jid": "product@localhost", "candidates": [
                    {"candidate_index": 0, "event_id": "generated-recovery", "safety_input": {}}
                ],
            }))
        await inbox.run()
        response = json.loads(sent.call_args.args[1].body)
        result = response["results"][0]
        assert result["is_safe"] is False
        assert result["validation_status"] == "deferred_physical"
        assert result["planning_only"] is True
        assert result["execution_authorized"] is False
        assert result["safety_dfa_states_before"] == result["safety_dfa_states_after"] == before
        assert response["admissible_recovery_event_ids"] == []
        assert response["planning_expandable_event_ids"] == ["generated-recovery"]
        validator.assert_not_called()
        assert cca.safety_monitor.current_states == before

    asyncio.run(scenario())


@pytest.mark.parametrize("changed", ["planning_only", "execution_authorized", "is_safe", "findings"])
def test_outline_deferral_cannot_be_confused_with_safety_pass_or_unavailable(changed):
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes.multi_turn_outline_generation import (
        _planning_only_physical_deferral,
    )

    result = {"validation_status": "deferred_physical", "planning_only": True,
              "execution_authorized": False, "is_safe": False, "findings": []}
    assert _planning_only_physical_deferral(result)
    result[changed] = [{"constraint_code": "unavailable"}] if changed == "findings" else not result[changed]
    assert not _planning_only_physical_deferral(result)


@pytest.mark.parametrize("stale_execution", [False, True])
def test_live_admission_rechecks_full_scene_hold_after_concurrent_preparations(stale_execution):
    from cais_spade_llm.agents.central_controller.recovery_composition_admission import (
        LiveSafetyRuntime,
    )
    from cais_spade_llm.agents.central_controller.region_admission import RegionReservationLedger

    async def scenario():
        owners = [SimpleNamespace(agent_name=rid, jid=rid + "@localhost") for rid in ("r1", "r2")]
        runtime = SimpleNamespace(
            context=SimpleNamespace(admission_lock=RLock(), run_id="run"), resource_agents=owners)
        plan = SimpleNamespace(current_state="ready",
                               _next_task_ids_from_state=lambda _: ["task1", "task2"])
        from cais_spade_llm.agents.central_controller.online_safety_monitor import (
            OnlineSafetyMonitor,
        )

        cca = SimpleNamespace(safety_monitor=OnlineSafetyMonitor({}, []),
                              plan_fsa_monitor=plan, _environment_admission=lambda _: SimpleNamespace())
        ledger = RegionReservationLedger(lock=runtime.context.admission_lock)
        preparation = SimpleNamespace(require_execution_coverage=Mock())
        live = LiveSafetyRuntime(runtime=runtime, cca=cca, preparation=preparation, region_reservations=ledger)
        live.monitor = SimpleNamespace(invalid_reason="", check=lambda *args, **kwargs: {"status": "allowed"})
        live._identity = lambda: live.commands.execution_epoch
        checkpoint = {"observations": {
            rid: {"physical": {"observation_state": {"state": "idle"}}} for rid in ("r1", "r2")}}
        observation = {"revision": 0, "time_exact": "0", "region_occupancy": {},
                       "stationary_resources": ["r1", "r2"]}
        live._capture = lambda: (deepcopy(checkpoint), deepcopy(observation))
        proof = {"frozen": {"horizon": [0, 1], "geometry": {"regions": {}}},
                 "models": {"r1": {}, "r2": {}}, "rules": [],
                 "observations": [{"time_exact": "0", "region_occupancy": {}}],
                 "owner_program": {"step_results": [{"primitive": "wait", "resolved_params": {}}]}}
        def ground(task, *args):
            result = deepcopy(proof)
            result["common_composition"] = {"status": "allowed", "choices": [{
                "kind": "start", "event_ids": [task["outline_id"]], "status": "allowed"}]}
            return result

        live._ground = ground
        both_prepared = asyncio.Event()
        arrived = 0

        async def prepare(request):
            nonlocal arrived
            arrived += 1
            if arrived == 2:
                if stale_execution:
                    live.commands.execution_epoch += 1
                both_prepared.set()
            await both_prepared.wait()
            return {"status": "prepared", "preparation_id": request["task_id"],
                    "command_ids": [], "command_ledger_revision": live.commands.revision}

        tasks, refs, events = {}, {}, []
        for index, owner in enumerate(owners, 1):
            task_id = "task" + str(index)
            task = {"task_id": task_id, "outline_id": task_id, "resource_id": owner.agent_name,
                    "resource_jid": owner.jid, "function_name": "execute_recovery_macro",
                    "params": {}, "primitive_steps": [{"primitive": "wait", "params": {}}]}
            reference = {"recovery_id": "recovery", "task_id": task_id, "problem_id": "proof"}
            tasks[task_id], refs[task_id] = task, reference
            events.append({"task_id": task_id, "resource_jid": owner.jid,
                           "function_name": task["function_name"],
                           "params": {"recovery_composition_ref": reference}})
            owner.prepare_recovery_safety_program = Mock(return_value={"status": "prepared"})
            owner.prepare_recovery_composition_evidence = prepare
            owner.recovery_composition_evidence_provider = SimpleNamespace(
                authorize_preparation=Mock(return_value=True))
        live.sessions["recovery"] = {"tasks": tasks, "refs": refs, "invalid_reason": "",
                                      "grants": {}, "completed_tasks": set(),
                                      "product_jid": "product@localhost", "complete": False}
        results = await asyncio.gather(*(live.check(event, sender=event["resource_jid"]) for event in events))
        if stale_execution:
            assert all(row["status"] == "inconclusive" for row in results), results
            assert all(row["reason"] == "stale_live_admission_snapshot" for row in results)
            assert not live.sessions["recovery"]["grants"]
            assert not ledger.claims
            return
        assert sorted(row["status"] for row in results) == ["allowed", "held"], results
        assert len(live.sessions["recovery"]["grants"]) == 1
        assert sum(owner.recovery_composition_evidence_provider.authorize_preparation.call_count
                   for owner in owners) == 1
        index = next(index for index, row in enumerate(results) if row["status"] == "allowed")
        snapshot = ledger.snapshot()
        duplicate = await live.check(events[index], sender=events[index]["resource_jid"])
        assert duplicate["recovery_composition_grant"] == results[index]["recovery_composition_grant"]
        assert ledger.snapshot() == snapshot
        assert arrived == 2

    asyncio.run(scenario())


@pytest.mark.parametrize("change", ["missing_ref", "wrong_ref", "old_run", "missing_owner_evidence"])
def test_live_completion_requires_exact_identity_and_retained_owner_evidence(change):
    from cais_spade_llm.agents.central_controller.recovery_composition_admission import (
        LiveSafetyRuntime,
    )

    live = object.__new__(LiveSafetyRuntime)
    live.lock = RLock()
    live.preparation = object()
    live.epoch = 0
    live.runtime = SimpleNamespace(context=SimpleNamespace(run_id="run"))
    live.cca = SimpleNamespace(plan_fsa_monitor=None)
    live.monitor = SimpleNamespace(commit=Mock(), revision=0)
    live.regions = SimpleNamespace(finish=Mock())
    reference = {"recovery_id": "recovery", "task_id": "task"}
    task = {"task_id": "task", "resource_jid": "r1@localhost",
            "function_name": "execute_recovery_macro", "primitive_steps": [{}]}
    work = {"grant": {"recovery_composition_ref": reference, "region_reservation_token": "token"},
            "complete": False, "executions": {0: {"success": True}}, "proof": {}}
    session = {"tasks": {"task": task}, "grants": {"task": work},
               "completed_tasks": set(), "product_jid": "product@localhost", "complete": False}
    live.sessions = {"recovery": session}
    record = {"task_id": "task", "status": "completed", "run_id": "run",
              "resource_jid": "r1@localhost", "function_name": "execute_recovery_macro",
              "params": {"recovery_composition_ref": deepcopy(reference)}}
    if change == "missing_ref":
        record["params"] = {}
    elif change == "wrong_ref":
        record["params"]["recovery_composition_ref"]["recovery_id"] = "previous"
    elif change == "old_run":
        record["run_id"] = "previous"
    else:
        work["executions"] = {}
    result = live.observe(record, sender="r1@localhost")
    assert result["status"] in {"ignored", "inconclusive"}
    live.monitor.commit.assert_not_called()
    live.regions.finish.assert_not_called()
    assert not work["complete"]


def test_physical_admission_installs_existing_live_runtime_before_first_nominal_request(monkeypatch):
    from cais_spade_llm.agents.central_controller.recovery_admission_runtime import (
        physical_admission,
    )

    runtime, owner = object(), object()
    cca = SimpleNamespace(predefined_safety_required=True,
                          _environment_runtime_for_sender=lambda product: runtime)
    install = Mock(return_value=owner)
    monkeypatch.setattr(
        "cais_spade_llm.agents.central_controller.recovery_admission_runtime.install_live_runtime", install)
    assert physical_admission(cca, "product@localhost") is owner
    install.assert_called_once_with(runtime, cca)


def test_live_physical_history_retains_latched_condition_and_rejects_stale_commit():
    from cais_spade_llm.agents.central_controller.online_safety_monitor import LivePhysicalMonitor
    from cais_spade_llm.agents.central_controller.ppr_ap import ap_record, make_ap_definition
    from cais_spade_llm.agents.central_controller.reviewed_primitive_program_safety import (
        _compile_formula,
    )

    formula = "G (ap001 -> G !ap002)"
    rule = {"rule_id": "response", "formula": formula,
            "dfa_dot": _compile_formula(formula, {"ap001", "ap002"}),
            "aps": [
                ap_record(label, make_ap_definition("ap_state", "any", "any", "r1", symbol), "")
                for label, symbol in (("ap001", "started"), ("ap002", "completed"))
            ]}
    prepared = {"rules": [rule], "observations": [{"time_exact": "0", "phase": "at"}],
                "valuations": [{"response": {"ap001": True, "ap002": False}}]}
    monitor = LivePhysicalMonitor()
    proof = monitor.check(prepared)
    assert proof["status"] == "allowed"
    assert not monitor.states and not monitor.history
    monitor.commit(proof, success=True, execution_evidence={"owner_step": "first"})
    retained = deepcopy(monitor.states)
    prepared["valuations"][0]["response"] = {"ap001": False, "ap002": False}
    pending = monitor.check(prepared)
    assert pending["end_states"]["response"] == sorted(retained["response"])
    monitor.commit(pending, success=True, execution_evidence={"owner_step": "second"})
    with pytest.raises(ValueError, match="history or proof changed"):
        monitor.commit(proof, success=True, execution_evidence={"owner_step": "duplicate"})
    prepared["valuations"][0]["response"]["ap002"] = True
    forbidden = monitor.check(prepared)
    assert forbidden["status"] == "held"
    assert forbidden["reason"] == "possible_physical_requirement_violation"
    assert monitor.states == retained
    fresh = LivePhysicalMonitor()
    assert fresh.check(prepared)["status"] == "allowed"


def test_live_pending_response_requires_continuation_evidence_before_admission():
    from cais_spade_llm.agents.central_controller.online_safety_monitor import LivePhysicalMonitor
    from cais_spade_llm.agents.central_controller.reviewed_primitive_program_safety import (
        _compile_formula,
    )

    formula = "G (ap001 -> F ap002)"
    rule = {"rule_id": "response", "formula": formula, "aps": [],
            "dfa_dot": _compile_formula(formula, {"ap001", "ap002"})}
    prepared = {"rules": [rule], "observations": [{"time_exact": "0", "phase": "at"}],
                "valuations": [{"response": {"ap001": True, "ap002": False}}]}
    monitor = LivePhysicalMonitor()
    proof = monitor.check(prepared)
    assert proof["status"] == "inconclusive"
    assert proof["reason"] == "temporal_continuation_proof_unavailable"
    assert proof["prefix_safe"] is True
    assert proof["pending_rule_ids"] == ["response"]
    assert proof["end_accepting"] is False
    assert not monitor.history
    with pytest.raises(ValueError):
        monitor.commit(proof, success=True, execution_evidence={"owner_step": "not-admitted"})


def _prepared_admission_case():
    """Exercise the admission transaction with declared software-fixture motion."""
    from cais_spade_llm.agents.central_controller.recovery_composition_admission import (
        RecoveryCompositionAdmission,
    )

    stages = []
    task = {"task_id": "nominal", "resource_id": "r1", "event_name": "move_home",
            "parameters": {}, "run_id": "run"}
    context = SimpleNamespace(admission_lock=RLock(), run_id="run", revision=0,
        snapshot=lambda: {}, part_tracker={}, pending_tasks={"nominal": task})
    owner = SimpleNamespace(agent_name="r1", jid="r1@localhost")
    runtime = SimpleNamespace(context=context, resource_agents=[owner],
                              product_jid="product@localhost", jids={"r1": owner.jid})
    native = SimpleNamespace(check=AsyncMock(return_value={"status": "allowed", "snapshot_revision": 0}),
        _revision=lambda: 0, commit_prepared=Mock(return_value={"status": "allowed"}))
    plan = SimpleNamespace(current_state="ready", _next_task_ids_from_state=lambda _: ["recovery"],
                           process_event=Mock())
    cca = SimpleNamespace(safety_monitor=OnlineSafetyMonitor({}, []), predefined_safety_fingerprint="selected",
                          _environment_admission=lambda _: native, plan_fsa_monitor=plan)
    preparation = SimpleNamespace(require_execution_coverage=lambda *args, **kwargs: stages.append("coverage"))
    coordinator = RecoveryCompositionAdmission(runtime=runtime, cca=cca, preparation=preparation)
    checkpoint = {"observations": {"r1": {"physical": {"observation_state": {"state": "idle"}}}}}
    revision = 0

    def capture():
        nonlocal revision
        revision += 1
        stages.append("capture")
        observation = {"revision": revision, "time_exact": str(revision),
                       "region_occupancy": {}, "stationary_resources": ["r1"]}
        coordinator.regions.observe(observation)
        return deepcopy(checkpoint), observation

    coordinator._capture = capture
    steps = [{"primitive": "move_cartesian", "params": {"x": 1}}]
    proof = {"frozen": {"horizon": [0, 1], "geometry": {"regions": {}}}, "models": {"r1": {}},
             "rules": [], "observations": [{"time_exact": "0", "phase": "at", "region_occupancy": {}}],
             "valuations": [{}], "owner_program": {"step_results": [
                 {"primitive": "move_cartesian", "resolved_params": {"x": 1}}]}}

    def ground(task, *args):
        stages.append("physical")
        result = deepcopy(proof)
        result["common_composition"] = {"status": "allowed", "choices": [{
            "kind": "start", "event_ids": [task["outline_id"]], "status": "allowed",
            "time_exact": "0", "edge_id": "fixture_start"}]}
        return result

    coordinator._ground = ground
    owner.prepare_recovery_safety_program = lambda *args: stages.append("prepare") or {"status": "prepared"}
    owner.prepare_nominal_safety_program = lambda task: {"primitive_steps": deepcopy(steps)}

    async def evidence(request):
        command = coordinator.commands.prepare(resource_id="r1", task_id=request["task_id"],
            command=steps[0], owner_identity={"preparation_id": request["task_id"]})
        return {"status": "prepared", "preparation_id": request["task_id"], "command_ids": [command],
                "command_ledger_revision": coordinator.commands.revision}

    def authorize(**kwargs):
        stages.append("authorize")
        coordinator.commands.authorize_many(kwargs["command_ids"], reservation_token=kwargs["reservation_token"],
                                            expected_revision=kwargs["expected_revision"])
        return True

    owner.prepare_recovery_composition_evidence = evidence
    owner.recovery_composition_evidence_provider = SimpleNamespace(authorize_preparation=Mock(side_effect=authorize))
    return coordinator, owner, native, task, steps, stages


def test_shared_coordinator_nominal_then_recovery_retains_history_and_empty_region_authorization(monkeypatch):
    from cais_spade_llm.agents.central_controller.recovery_composition_admission import (
        LiveSafetyRuntime,
        RecoveryCompositionAdmission,
    )
    monkeypatch.setattr("cais_spade_llm.agents.central_controller.predefined_safety_runtime.predefined_scope",
                        lambda *args: None)

    async def scenario():
        coordinator, owner, native, task, steps, stages = _prepared_admission_case()
        assert LiveSafetyRuntime is RecoveryCompositionAdmission
        assert coordinator.monitor is coordinator.cca.safety_monitor.physical_monitor
        assert coordinator.commands is coordinator.runtime.physical_commands
        preview = await coordinator.check_nominal(task, coordinator.runtime.product_jid, commit=False)
        assert preview["status"] == "allowed" and preview["committed"] is False
        assert "recovery_composition_grant" not in preview and not coordinator.regions.claims
        assert all(not row.get("reservation_token") for row in coordinator.commands.commands.values())
        native.commit_prepared.assert_not_called()
        stages.clear()
        result = await coordinator.check_nominal(task, coordinator.runtime.product_jid, commit=True)
        assert result["status"] == "allowed", result
        grant = result["recovery_composition_grant"]
        assert grant["nominal_task"] == task and grant["run_id"] == "run"
        assert result["region_reservation"]["claim"]["regions"] == []
        native.commit_prepared.assert_called_once_with(task, native.check.return_value)
        assert stages == ["capture", "prepare", "coverage", "physical", "capture", "authorize"]
        repeat_preview = await coordinator.check_nominal(task, coordinator.runtime.product_jid, commit=False)
        assert "recovery_composition_grant" not in repeat_preview
        for command_id, command in coordinator.commands.commands.items():
            if command.get("reservation_token"):
                coordinator.commands.start(command_id, command=steps[0], controller_goal_id=[1] * 16)
                coordinator.commands.finish(command_id, success=True, observations={"stopped": True})
        coordinator._record_execution(task_id=task["task_id"], step_index=0,
                                      result={"success": True, "observations": {"stopped": True}})
        record = {"task_id": task["task_id"], "resource_jid": owner.jid, "function_name": "move_home",
                  "status": "completed", "run_id": "run", "recovery_composition_ref": grant["recovery_composition_ref"]}
        assert coordinator.observe(record, sender=owner.jid)["success"] is True
        assert coordinator.observe(record, sender=owner.jid)["status"] == "ignored"
        assert coordinator.monitor.revision == 1 and len(coordinator.monitor.history) == 1
        registration = await coordinator.register({"recovery_id": "recovery", "recovery_safety_scope_id": "scope",
            "tasks": [{"task_id": "recovery", "outline_id": "recovery", "resource_id": "r1",
                       "resource_jid": owner.jid, "function_name": "execute_recovery_macro",
                       "params": {"primitive_steps": steps}}]}, product_jid=coordinator.runtime.product_jid)
        assert registration["status"] == "allowed"
        second = await coordinator.check({"task_id": "recovery", "resource_jid": owner.jid,
            "function_name": "execute_recovery_macro", "params": {"primitive_steps": steps,
                "recovery_composition_ref": registration["task_refs"]["recovery"]}}, sender=owner.jid)
        assert second["status"] == "allowed", second
        assert second["physical_proof"]["history_revision"] == 1
        assert len(coordinator.sessions) == 2 and coordinator.monitor.revision == 1
        assert native.commit_prepared.call_count == 1

    asyncio.run(scenario())


@pytest.mark.parametrize("failure", ["coverage", "authorization", "native_commit", "native_exception"])
def test_prepared_admission_failure_never_publishes_grant_and_retains_committed_claim(monkeypatch, failure):
    monkeypatch.setattr("cais_spade_llm.agents.central_controller.predefined_safety_runtime.predefined_scope",
                        lambda *args: None)

    async def scenario():
        coordinator, owner, native, task, _, stages = _prepared_admission_case()
        if failure == "coverage":
            coordinator.preparation.require_execution_coverage = Mock(side_effect=ValueError("NEEDS_CONTEXT"))
        elif failure == "authorization":
            owner.recovery_composition_evidence_provider.authorize_preparation = Mock(return_value=None)
        elif failure == "native_commit":
            native.commit_prepared.return_value = {"status": "inconclusive", "reason": "stale_snapshot"}
        else:
            native.commit_prepared.side_effect = RuntimeError("stale_snapshot")
        result = await coordinator.check_nominal(task, coordinator.runtime.product_jid, commit=True)
        assert result["status"] == "inconclusive", result
        assert not coordinator.sessions["nominal_nominal"]["grants"]
        assert coordinator.monitor.revision == 0
        if failure == "coverage":
            assert "prepare" in stages and "physical" not in stages
            assert not coordinator.regions.claims
            native.commit_prepared.assert_not_called()
        else:
            assert len(coordinator.regions.claims) == 1 and coordinator.holds()
            assert next(iter(coordinator.regions.claims.values()))["status"] == "failed"

    asyncio.run(scenario())


@pytest.mark.parametrize("when", ["before", "during_preparation"])
def test_prepared_recovery_stopped_runtime_never_authorizes_motion(when):
    async def scenario():
        coordinator, owner, _, _, steps, _ = _prepared_admission_case()
        registration = await coordinator.register({"recovery_id": "recovery", "recovery_safety_scope_id": "scope",
            "tasks": [{"task_id": "recovery", "outline_id": "recovery", "resource_id": "r1",
                       "resource_jid": owner.jid, "function_name": "execute_recovery_macro",
                       "params": {"primitive_steps": steps}}]}, product_jid=coordinator.runtime.product_jid)
        if when == "before":
            coordinator.runtime.stopped = True
        else:
            original = owner.prepare_recovery_composition_evidence

            async def prepare(request):
                result = await original(request)
                coordinator.runtime.stopped = True
                return result

            owner.prepare_recovery_composition_evidence = prepare
        result = await coordinator.check({"task_id": "recovery", "resource_jid": owner.jid,
            "function_name": "execute_recovery_macro", "params": {"primitive_steps": steps,
                "recovery_composition_ref": registration["task_refs"]["recovery"]}}, sender=owner.jid)
        assert result["status"] == "inconclusive"
        assert not coordinator.regions.claims and not coordinator.sessions["recovery"]["grants"]
        owner.recovery_composition_evidence_provider.authorize_preparation.assert_not_called()

    asyncio.run(scenario())


def test_prepared_registration_cannot_reuse_task_identity_from_completed_session():
    async def scenario():
        coordinator, owner, _, _, steps, _ = _prepared_admission_case()
        request = {"recovery_id": "first", "recovery_safety_scope_id": "scope",
            "tasks": [{"task_id": "recovery", "outline_id": "recovery", "resource_id": "r1",
                       "resource_jid": owner.jid, "function_name": "execute_recovery_macro",
                       "params": {"primitive_steps": steps}}]}
        assert (await coordinator.register(request, product_jid=coordinator.runtime.product_jid))["status"] == "allowed"
        coordinator.sessions["first"]["complete"] = True
        request["recovery_id"] = "second"
        assert (await coordinator.register(request, product_jid=coordinator.runtime.product_jid))["status"] == "inconclusive"
        assert set(coordinator.sessions) == {"first"}

    asyncio.run(scenario())


@pytest.mark.parametrize("authorization", [True, None, False, "raises"])
def test_graph_commands_without_mutex_regions_still_require_atomic_authorization(authorization):
    from cais_spade_llm.agents.central_controller.recovery_composition_admission import (
        RecoveryCompositionAdmission,
    )

    authorizer = Mock(return_value=authorization)
    if authorization == "raises":
        authorizer.side_effect = RuntimeError("owner rejected changed command")
    coordinator = RecoveryCompositionAdmission(preparation_authorizer=authorizer)
    session = {"analysis": {"region_mutexes": [], "problem_id": "proof",
                            "scope": {"included_resources": ["r1"]}},
               "execution_mode": "live", "completed_tasks": set(), "grants": {},
               "product_jid": "product@localhost", "refs": {"task": {"task_id": "task"}},
               "invalid_reason": ""}
    task = {"task_id": "task", "resource_id": "r1", "outline_id": "event"}
    context = {"time_exact": "0", "current_snapshot": {"resources": {"r1": {}}},
               "region_observation": {"revision": 0, "time_exact": "0",
                                      "region_occupancy": {}, "stationary_resources": ["r1"]}}
    prepared = {"preparation_id": "exact", "command_ids": ["command"]}
    preview = coordinator._reserve_regions(session, task, prepared, {}, context, commit=False)
    assert preview["regions"] == [] and not coordinator.region_reservations.claims
    authorizer.assert_not_called()
    if authorization is True:
        claim = coordinator._reserve_regions(session, task, prepared, {}, context, commit=True)
        assert claim["regions"] == []
        authorizer.assert_called_once_with(prepared, claim)
    else:
        with pytest.raises((ValueError, RuntimeError)):
            coordinator._reserve_regions(session, task, prepared, {}, context, commit=True)
        assert session["invalid_reason"]
        assert next(iter(coordinator.region_reservations.claims.values()))["status"] == "failed"


@pytest.mark.parametrize("retained", ["commands", "nominal_grants", "reservations"])
def test_new_coordinator_cannot_migrate_runtime_execution_authority(retained):
    from cais_spade_llm.agents.central_controller.recovery_composition_admission import (
        RecoveryCompositionAdmission,
    )

    coordinator, _, _, _, steps, _ = _prepared_admission_case()
    if retained == "commands":
        coordinator.commands.prepare(resource_id="r1", task_id="previous", command=steps[0],
                                     owner_identity={"preparation_id": "previous"})
    elif retained == "nominal_grants":
        coordinator.runtime.admission = SimpleNamespace(grants={"previous": object()})
    else:
        coordinator.regions.claims["previous"] = {"status": "active"}
    with pytest.raises(ValueError, match="fresh runtime"):
        RecoveryCompositionAdmission(runtime=coordinator.runtime, cca=coordinator.cca,
                                     preparation=coordinator.preparation,
                                     region_reservations=coordinator.regions)
    assert coordinator.runtime.physical_commands is coordinator.commands


def _prepared_graph_case(monkeypatch, *, conflict=False, formula=None):
    from test_continuous_motion import composition_case

    from cais_spade_llm.agents.central_controller.recovery_composition_admission import (
        RecoveryCompositionAdmission,
    )

    case, models = composition_case(conflict=conflict)
    inputs = case["grounding_inputs"]
    if formula is not None:
        inputs["catalog"]["specifications"][0]["formula"] = formula
    program = inputs["programs"][0]
    source = program["primitive_steps"][0]["source"]
    task = {"task_id": "supplied_task", "outline_id": source["outline_id"],
            "resource_id": program["resource_id"], "function_name": "execute_recovery_macro",
            "primitive_steps": deepcopy(program["primitive_steps"])}
    jids = {rid: row.get("resource_jid", rid + "@localhost")
            for rid, row in inputs["snapshot"]["resources"].items()}
    task["resource_jid"] = jids[task["resource_id"]]
    task["primitive_steps"][0]["source"]["resource_jid"] = jids[task["resource_id"]]
    owners = [SimpleNamespace(agent_name=rid, get_recovery_safety_primitive_model=lambda value=model: value)
              for rid, model in models.items()]
    runtime = SimpleNamespace(context=SimpleNamespace(admission_lock=RLock(), run_id="run"),
                              resource_agents=owners, jids=jids)
    cca = SimpleNamespace(safety_monitor=OnlineSafetyMonitor({}, []))
    coordinator = RecoveryCompositionAdmission(runtime=runtime, cca=cca, preparation=SimpleNamespace())
    checkpoint = {"checkpoint_id": "captured", "scene": deepcopy(inputs["scene"]),
        "geometry": deepcopy(inputs["geometry"]),
        "observations": {rid: {"physical": {"idle": True, "observation_state": deepcopy(state)}}
                         for rid, state in inputs["snapshot"]["resources"].items()},
        "parts": {name: {"pose": deepcopy(part["current_pose"])}
                  for name, part in inputs["snapshot"]["parts"].items()},
        "runtime": {"part_tracker": deepcopy(inputs["snapshot"]["parts"]), "acknowledgements": [],
                    "monitors": [], "admission": {"running": []}}}
    row = program["step_results"][0]
    evidence = row["model_evidence"]["continuous_motion"]
    prepared = {"resource_id": task["resource_id"], "steps": [{
        "binding": {"resource_id": task["resource_id"], "resource_jid": task["resource_jid"]},
        "primitive": row["primitive"], "params": deepcopy(row["resolved_params"]),
        "preparation_id": "owner_prepared", "joint_trajectory": deepcopy(evidence["joint_trajectory"]),
        "continuous_motion": deepcopy(evidence)}]}

    def selected(_cca, context):
        context["composition_inputs"]["grounding_inputs"].update(
            {key: deepcopy(inputs[key]) for key in ("catalog", "requirement_scopes")})

    monkeypatch.setattr("cais_spade_llm.agents.central_controller.predefined_safety_runtime.bind_predefined_context",
                        selected)
    return coordinator, task, prepared, checkpoint


@pytest.mark.parametrize("conflict,expected", [(False, "allowed"), (True, "held")])
def test_prepared_motion_uses_complete_common_graph(monkeypatch, conflict, expected):
    coordinator, task, prepared, checkpoint = _prepared_graph_case(monkeypatch, conflict=conflict)
    before = deepcopy(checkpoint["runtime"]["part_tracker"])
    proof = coordinator._ground(task, prepared, checkpoint)
    graph = proof["common_composition"]
    assert graph["status"] == expected, graph["reason"]
    assert len(graph["bindings"]) == 67
    assert checkpoint["runtime"]["part_tracker"] == before
    for name, part in proof["projected_snapshot"]["parts"].items():
        assert part.get("processCompleted") == before[name].get("processCompleted")
    assert not coordinator.monitor.history and not coordinator.monitor.states
    if expected == "allowed":
        assert any(choice["status"] == "allowed" and choice["kind"] == "start"
                   and choice["event_ids"] == [task["outline_id"]] for choice in graph["choices"])
        assert graph["completion_witness"] is not None


@pytest.mark.parametrize("missing", ["running", "remaining", "history", "native", "source", "primitive", "event"])
def test_prepared_common_graph_requires_complete_existing_context(monkeypatch, missing):
    coordinator, task, prepared, checkpoint = _prepared_graph_case(monkeypatch)
    reason = {"running": "ongoing_nominal_programs", "remaining": "remaining_recovery_programs",
              "history": "compatible_continuous_checkpoint", "native": "native_common_continuation",
              "source": "source.des_event_id", "primitive": "every registered primitive",
              "event": "registered task parameters"}[missing]
    if missing == "running":
        checkpoint["runtime"]["admission"]["running"] = [{"task_id": "already_running"}]
    elif missing == "remaining":
        coordinator.sessions["recovery"] = {"tasks": {task["task_id"]: task, "future": {}},
                                            "completed_tasks": set()}
    elif missing == "history":
        coordinator.monitor.history.append({"observed": "earlier"})
    elif missing == "native":
        checkpoint["runtime"]["monitors"] = [{"rules": [{"rule_id": "existing"}]}]
    elif missing == "primitive":
        task["primitive_steps"].append(deepcopy(task["primitive_steps"][0]))
    elif missing == "event":
        task["params"] = {"des_event_id": "different_registered_event"}
    else:
        task["primitive_steps"][0]["source"].pop("des_event_id")
    with pytest.raises(ValueError, match=reason):
        coordinator._ground(task, prepared, checkpoint)


def test_prepared_model_stationarity_is_declared_and_requires_controller_hold(monkeypatch):
    coordinator, task, prepared, checkpoint = _prepared_graph_case(monkeypatch)
    checkpoint["model_execution"] = True
    checkpoint["controller_goals"] = {"controller": {"holding": True, "has_active_goal": False,
                                                     "has_pending_goal": False}}
    for row in checkpoint["observations"].values():
        row["physical"].update(idle=False, stationary_contract={"kind": "idle_commanded_hold",
            "requires_no_running_tasks": True, "requires_no_active_goals": True,
            "future_execution_tracking": "not_established"})
    proof = coordinator._ground(task, prepared, checkpoint)
    assert proof["common_composition"]["status"] == "allowed"
    assert all(row["physical_execution_verified"] is False for row in proof["model_execution_assumptions"])
    checkpoint["controller_goals"]["controller"]["has_pending_goal"] = True
    with pytest.raises(ValueError, match="current idle controller"):
        coordinator._ground(task, prepared, checkpoint)


@pytest.mark.parametrize("formula", ["F ap001", "(!ap001 U ap001)"])
def test_prepared_common_graph_does_not_grant_an_unfulfilled_temporal_obligation(monkeypatch, formula):
    coordinator, task, prepared, checkpoint = _prepared_graph_case(monkeypatch, formula=formula)
    result = coordinator._ground(task, prepared, checkpoint)["common_composition"]
    assert result["status"] == "inconclusive", result
    assert result["pending_rule_ids"]
    assert result["completion_witness"] is None
    assert not any(choice["status"] == "allowed" for choice in result["choices"])


@pytest.mark.parametrize("graph", [None, {"status": "held", "reason": "no_common_continuation"},
    {"status": "allowed", "choices": []},
    {"status": "allowed", "choices": [{"kind": "wait", "event_ids": [], "status": "allowed"}]},
    {"status": "allowed", "choices": [{"kind": "start", "event_ids": ["different"], "status": "allowed"}]}])
def test_prepared_admission_requires_exact_winning_immediate_graph_start(monkeypatch, graph):
    monkeypatch.setattr("cais_spade_llm.agents.central_controller.predefined_safety_runtime.predefined_scope",
                        lambda *args: None)

    async def scenario():
        coordinator, owner, native, task, _, _ = _prepared_admission_case()
        original = coordinator._ground

        def ground(*args):
            proof = original(*args)
            proof["common_composition"] = deepcopy(graph)
            return proof

        coordinator._ground = ground
        result = await coordinator.check_nominal(task, coordinator.runtime.product_jid, commit=True)
        assert result["status"] in ("inconclusive", "held"), result
        assert not coordinator.sessions["nominal_nominal"]["grants"]
        assert not coordinator.regions.claims
        assert all(not row.get("reservation_token") for row in coordinator.commands.commands.values())
        native.commit_prepared.assert_not_called()
        owner.recovery_composition_evidence_provider.authorize_preparation.assert_not_called()

    asyncio.run(scenario())


def _prepared_AP_equivalence_proof(result, change, fresh):
    if change not in ("AP_change", "intermediate_AP", "unknown_AP", "missing_AP", "boundary", "rechecked_graph"):
        return result
    result["rules"] = [{"rule_id": "selected_rule", "aps": [{"label": "ap001"}, {"label": "ap002"}]}]
    result["valuations"] = [{"selected_rule": {"ap001": False, "ap002": False}} for _ in range(3)]
    result["observations"] = [{"phase": phase, "time_exact": stamp}
        for phase, stamp in (("at", "0"), ("between", "1/2"), ("at", "1"))]
    if change == "unknown_AP":
        result["valuations"][0]["selected_rule"]["ap002"] = [False, True]
    if change == "missing_AP":
        result["valuations"][0]["selected_rule"].pop("ap002")
    if fresh:
        if change == "AP_change":
            result["valuations"][0]["selected_rule"]["ap002"] = True
        elif change == "intermediate_AP":
            result["valuations"][1]["selected_rule"]["ap002"] = True
        elif change == "boundary":
            result["observations"][1]["rule_cells"] = {"selected_rule": []}
        elif change == "rechecked_graph":
            result["common_composition"]["status"] = "held"
    return result


@pytest.mark.parametrize("change,allowed", [
    (None, True), ("instance", False), ("command", False), ("contract", False),
    ("launch", False), ("configuration", False), ("custody", False),
    ("joint", False), ("collision", False), ("stationary_pose", True),
    ("stationary_collision", True), ("part", True), ("part_source", False),
     ("geometry_source", False), ("primitive_model", False),
    ("AP_change", False), ("intermediate_AP", False), ("unknown_AP", False),
    ("missing_AP", False), ("boundary", False), ("rechecked_graph", False),
])
def test_prepared_admission_binds_fresh_start_to_model_and_controller(monkeypatch, change, allowed):
    from test_resource_safety_preparation import _modeled_start_case

    monkeypatch.setattr("cais_spade_llm.agents.central_controller.predefined_safety_runtime.predefined_scope",
                        lambda *args: None)

    async def scenario():
        coordinator, owner, native, task, _, _ = _prepared_admission_case()
        start, observed, motion = _modeled_start_case()
        stationary = {"current_pose": [2., 0., 0., 0., 0., 0., 1.],
            "observation_state": {"state": "idle", "current_pose": [2., 0., 0., 0., 0., 0., 1.]},
            "component_bounds": [{"id": "fixed", "bounds": [[1.95, 2.05], [-.05, .05], [-.05, .05]]}],
            "geometry_source": {"configuration": "fixed"}, "idle": True, "moving_links": {}}
        checkpoint = {"model_execution": True, "launch_id": "launch", "runtime": {"configuration": "fixed"},
            "scene": {"configured": "scene"},
            "controller_goals": {"/arm/recovery_state": {"instance_id": "controller", "command_revision": 3,
                "contract_revision": 1, "holding": True, "has_active_goal": False, "has_pending_goal": False,
                "positions": [0.]}},
            "observations": {"r1": {"physical": deepcopy(start), "primitive_model": {"version": 2}},
                             "r2": {"physical": stationary}},
            "geometry": {"regions": {}, "resources": {
                "r1": {"frame": "world", "component_bounds": deepcopy(start["component_bounds"])},
                "r2": {"frame": "world", "component_bounds": deepcopy(stationary["component_bounds"])}},
                "parts": {"gear_small": {"footprint": [[-.01, .01]] * 3}}},
            "parts": {"gear_small": {"name": "gear_small", "pose": [-1., 0., 0., 0., 0., 0., 1.],
                "geometry_source": {"instance_id": "physics", "declaration_fingerprint": "fixed", "simulation_time": 1.}}}}
        fresh = deepcopy(checkpoint)
        fresh["observations"]["r1"]["physical"] = deepcopy(observed)
        fresh["geometry"]["resources"]["r1"]["component_bounds"] = deepcopy(observed["component_bounds"])
        fresh["parts"]["gear_small"]["geometry_source"]["simulation_time"] = 2.
        current = fresh["observations"]["r1"]["physical"]
        if change in ("instance", "command", "contract"):
            field = {"instance": "instance_id", "command": "command_revision", "contract": "contract_revision"}[change]
            fresh["controller_goals"]["/arm/recovery_state"][field] = "restarted" if change == "instance" else 9
            # Restart/command rejection must hold even without any position change.
            fresh["observations"]["r1"]["physical"] = deepcopy(start)
            fresh["geometry"]["resources"]["r1"] = deepcopy(checkpoint["geometry"]["resources"]["r1"])
        elif change == "launch":
            fresh["launch_id"] = "restarted"
        elif change == "configuration":
            fresh["runtime"]["configuration"] = "changed"
        elif change == "custody":
            current["observation_state"]["held_part"] = "gear_small"
        elif change == "joint":
            current["joint_positions"] = [.011]
        elif change == "collision":
            current["component_bounds"][0]["bounds"][0][1] = .07
        elif change == "stationary_pose":
            fresh["observations"]["r2"]["physical"]["current_pose"][0] += .000001
        elif change == "stationary_collision":
            fresh["geometry"]["resources"]["r2"]["component_bounds"][0]["bounds"][0][1] += .000001
        elif change == "part":
            fresh["parts"]["gear_small"]["pose"][0] += .000001
        elif change == "part_source":
            fresh["parts"]["gear_small"]["geometry_source"]["declaration_fingerprint"] = "changed"
        elif change == "geometry_source":
            current["geometry_source"]["configuration"] = "changed"
        elif change == "primitive_model":
            fresh["observations"]["r1"]["primitive_model"]["version"] = 3
        captures = iter((checkpoint, fresh))
        original_capture = coordinator._capture

        def capture():
            _, observation = original_capture()
            return deepcopy(next(captures)), observation

        coordinator._capture = capture
        original_ground = coordinator._ground
        ground_calls = iter((False, True))
        coordinator._ground = lambda *args: _prepared_AP_equivalence_proof(
            original_ground(*args), change, next(ground_calls))
        coordinator.monitor.check = Mock(return_value={"status": "allowed", "history_revision": 0})

        owner.prepare_recovery_safety_program = lambda *args: {"status": "prepared", "steps": [{
            "start": deepcopy(start), "continuous_motion": deepcopy(motion)}]}
        result = await coordinator.check_nominal(task, coordinator.runtime.product_jid, commit=True)
        assert result["status"] == ("allowed" if allowed else "inconclusive"), result
        assert native.commit_prepared.call_count == int(allowed)
        assert owner.recovery_composition_evidence_provider.authorize_preparation.call_count == int(allowed)
        assert bool(coordinator.sessions["nominal_nominal"]["grants"]) is allowed
        assert bool(coordinator.regions.claims) is allowed
        assert result.get("recovery_composition_grant", {}).get("physical_execution_verified", False) is False

    asyncio.run(scenario())


def test_prepared_common_graph_reobserves_all_APs_without_a_geometric_epsilon(monkeypatch):
    coordinator, task, prepared, checkpoint = _prepared_graph_case(monkeypatch)
    before = coordinator._ground(task, prepared, checkpoint)
    fresh = deepcopy(checkpoint)
    resource = fresh["observations"]["ur5e-3"]["physical"]["observation_state"]
    resource["current_pose"][0] += .000001
    unchanged = coordinator._ground(task, prepared, fresh)
    assert len(unchanged["rules"]) == 67
    assert before["rules"] == unchanged["rules"]
    assert before["valuations"] == unchanged["valuations"]
    assert unchanged["common_composition"]["status"] == "allowed"
    resource["current_pose"] = [1., 0., 0., 0., 0., 0., 1.]
    changed = coordinator._ground(task, prepared, fresh)
    assert changed["rules"] == before["rules"]
    assert changed["valuations"][0] != before["valuations"][0]
    assert changed["common_composition"]["status"] == "held"


@pytest.mark.parametrize("association", ["omitted_source", "wrong_source", "wrong_binding", "wrong_resource"])
def test_prepared_source_optional_resource_jid_keeps_authoritative_owner_binding(monkeypatch, association):
    coordinator, task, prepared, checkpoint = _prepared_graph_case(monkeypatch)
    if association == "omitted_source":
        task["primitive_steps"][0]["source"].pop("resource_jid")
        source = deepcopy(task["primitive_steps"][0]["source"])
        proof = coordinator._ground(task, prepared, checkpoint)
        assert proof["common_composition"]["status"] == "allowed", proof["common_composition"]
        assert proof["owner_program"]["primitive_steps"][0]["source"] == source
        assert proof["owner_program"]["step_results"][0]["source"] == source
        return
    if association == "wrong_source":
        task["primitive_steps"][0]["source"]["resource_jid"] = "another@localhost"
    elif association == "wrong_binding":
        prepared["steps"][0]["binding"]["resource_jid"] = "another@localhost"
    else:
        prepared["resource_id"] = "ur5e-3"
    with pytest.raises(ValueError, match="resource owner"):
        coordinator._ground(task, prepared, checkpoint)
