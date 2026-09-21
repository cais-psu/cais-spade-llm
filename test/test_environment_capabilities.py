"""Runtime environmental discovery, resource ownership, and v2 acknowledgement evidence."""

from __future__ import annotations

import asyncio
import json
import time
from contextlib import asynccontextmanager
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from spade.behaviour import CyclicBehaviour
from spade.template import Template

from cais_spade_llm.agents.intelligent_product.product_agent import ProductAgent
from cais_spade_llm.agents.resource_agent.resource_agent import ResourceAgent
from cais_spade_llm.agents.shared_information.environment_capabilities import (
    CapabilityReplyInbox,
    CapabilityRequestInbox,
    explore,
    message,
)
from cais_spade_llm.agents.shared_information.local_dispatch import send_agent_message
from cais_spade_llm.product.environment import EnvironmentProductContext, verify_environment_run
from cais_spade_llm.product.order import validate_product_order
from cais_spade_llm.recovery_framework import PRODUCT_PATH, ROOT, SCENE_PATH, read_json
from cais_spade_llm.recovery_framework.environment_runtime import EnvironmentRuntime
from cais_spade_llm.resources.environment_models import (
    build_environment_models,
    candidates,
    matches_requirement,
    process_json,
    project_transition,
)

SQUARE = "KET4_Square_4mm"
BUFFER = "Buffer For Machined parts"


@pytest.fixture
def inputs():
    meta = next(iter(read_json(PRODUCT_PATH).values()))
    result = {
        "scene": read_json(SCENE_PATH),
        "product_order": read_json(ROOT / meta["product_order_file"]),
        "geometry": read_json(ROOT / meta["product_geometry_file"])["gazebo"],
    }
    result["product_order"]["parts"] = [SQUARE]
    result["geometry"]["assembly_board"]["dimensions_m"] = [0.5, 0.5, 0.05]
    for machine, shape in zip(result["scene"]["machines"], ("square", "circle"), strict=True):
        machine["current_configuration"] = {
            "tool": "test_tool",
            "workholding": "test_workholding",
            "program": {
                "validated": True,
                "effects": [{"process": "trim", "result": shape}],
                "required_configuration": {"tool": "test_tool", "workholding": "test_workholding"},
            },
            "parameters": {},
        }
    return result


@asynccontextmanager
async def network(inputs, permitted=None):
    """Drive real SPADE inbox dispatch locally without XMPP, LLM calls, or robots."""
    resources = [
        ResourceAgent(f"test-resource-{index}@localhost", "none", name=rid, cca_jid="cca@localhost")
        for index, rid in enumerate(build_environment_models(inputs["scene"]), 1)
    ]
    product = ProductAgent(
        "assembly_board-v1@localhost",
        "none",
        name="assembly_board-v1",
        resource_agents=resources,
        resource_jids=[str(r.jid) for r in resources],
        cca_jid="cca@localhost",
        product_order_file=str(
            ROOT
            / "cais_spade_llm/specification/products/orders/assembly_board-v1-recovery-framework.json"
        ),
    )
    prepared = {
        "inputs": inputs,
        "setup": {"permitted_resources": permitted or [r.agent_name for r in resources]},
    }
    runtime = EnvironmentRuntime(product, prepared, resources)
    product.environment_runtime = runtime
    agents = {str(a.jid): a for a in [product, *resources]}
    container = SimpleNamespace(
        has_agent=lambda jid: jid in agents, get_agent=lambda jid: agents[jid], agents=agents
    )
    inboxes = []
    for agent in resources:
        inbox = CapabilityRequestInbox()
        agent.add_behaviour(inbox, Template(metadata={"type": "capability_request"}))
        inboxes.append(inbox)
    inbox = CapabilityReplyInbox()
    product.add_behaviour(inbox, Template(metadata={"type": "capability_reply"}))
    inboxes.append(inbox)
    for agent in agents.values():
        agent.container = container

    async def consume(inbox):
        while True:
            await inbox.run()

    workers = [asyncio.create_task(consume(inbox)) for inbox in inboxes]
    driver = SimpleNamespace(
        agent=product, send=AsyncMock(side_effect=AssertionError("Unexpected XMPP send"))
    )
    try:
        yield runtime, driver
    finally:
        for worker in workers:
            worker.cancel()
        await asyncio.gather(*workers, return_exceptions=True)


def test_resource_definitions_do_not_enumerate_product_catalogue(inputs):
    context = EnvironmentProductContext(**inputs)
    for model in context.models.values():
        encoded = json.dumps(
            {"state_variables": model["state_variables"], "events": model["events"]}
        )
        assert "KET4_Square_4mm" not in encoded
        assert "RGOCG4-50_Round_4mm" not in encoded
        assert "from_assignment" not in encoded
        assert "nominal_parts" not in model["assignments"]
        assert "supported_products" not in model["assignments"]
    assert context.models[BUFFER]["state_variables"]["zone_1_part"]["reference"] == "part_name"
    for name in ("machine_part", "advance_conveyor", "advance_part", "place_insert"):
        exported = process_json(context.models, name)
        assert exported["events"] and all(
            "guards" in event and "updates" in event for event in exported["events"]
        )
    assert process_json(context.models, "place_insert")["events"][0]["program"]["steps"]
    scene = deepcopy(inputs["scene"])
    for machine in scene["machines"]:
        machine.pop("nominal_parts")
    scene["3D Printing Station"].pop("supported_products")
    without_assignments = build_environment_models(scene, list(context.part_tracker))
    assert without_assignments == context.models


def test_matching_order_requires_explicit_results_and_preserves_historical_records(inputs):
    from cais_spade_llm.resources.environment_models import matches_requirement

    order = deepcopy(inputs["product_order"])
    order.pop("processPlan")
    with pytest.raises(ValueError, match="processPlan must declare"):
        validate_product_order(order, inputs["geometry"], require_process_requirements=True)
    assert not matches_requirement(
        {"processCompleted": ["machine_part"]}, {"process": "trim", "result": "square"}
    )
    with pytest.raises(ValueError, match="v1 remains historical"):
        verify_environment_run({"schema_version": 1})


def test_initial_buffer_occupancy_is_a_reference_and_never_implies_processing(inputs):
    inputs["scene"][BUFFER]["zones"][0]["initial_part"] = SQUARE
    with pytest.raises(ValueError, match="duplicate part custody"):
        EnvironmentProductContext(**inputs)
    inputs["scene"]["Storage"]["slots"].pop(SQUARE)
    context = EnvironmentProductContext(**inputs)
    assert context.contact_resource(SQUARE) == BUFFER
    assert context.models[BUFFER]["current_valuation"]["zone_1_part"] == SQUARE
    assert context.part_tracker[SQUARE]["processCompleted"] == []
    assert context.outstanding() == (SQUARE, {"processesToComplete": [{"process": "trim", "result": "square"}]})
    inputs["scene"][BUFFER]["zones"][0]["initial_part"] = "unknown"
    with pytest.raises(ValueError, match="Invalid resource value"):
        EnvironmentProductContext(**inputs)


@pytest.mark.parametrize("schema_version", [2, 3])
def test_distributed_square_trim_then_assembly_and_exit(inputs, schema_version):
    if schema_version == 2:
        inputs["product_order"]["requirements"] = {
            part: [
                {"state": "assembled", "target": inputs["geometry"]["parts"]["assembly_target_map"][part]}
                if requirement["process"] == "assembly" else requirement
                for step in steps for requirement in step["processesToComplete"]
            ]
            for part, steps in inputs["product_order"].pop("processPlan").items()
        }
    async def scenario():
        async with network(inputs) as (runtime, driver):
            context = runtime.context
            before = context.snapshot()
            result = await explore(runtime, driver, timeout=15)
            assert result["status"] == "planned", result
            assert context.snapshot() == before and context.revision == 0
            assert [task["event_name"] for task in result["tasks"]] == [
                "pick_part",
                "move_to_resource",
                "place_release",
                "machine_part",
            ]
            assert result["tasks"][-1]["resource_id"] == "M1"
            for task in result["tasks"]:
                pending = context.prepare(task, simulated=True)
                ack = {**pending, "status": "completed"}
                assert context.acknowledge(ack)
                assert not context.acknowledge(ack)
            assert context.part_tracker[SQUARE]["processCompleted"] == [
                {"process": "trim", "result": "square"}
            ]
            result = await explore(runtime, driver, timeout=15)
            assert result["status"] == "planned", result
            actors = [task["resource_id"] for task in result["tasks"]]
            assert (
                "ur5e-1" in actors
                and "Conveyor" in actors
                and BUFFER in actors
                and "ur5e-3" in actors
            )
            completed = [{"process": "trim", "result": "square"}]
            for task in result["tasks"]:
                if task["event_name"] == "place_insert":
                    invalid = deepcopy(task)
                    invalid["parameters"]["target"] = "Gear_Plate/Gear_Shaft_1"
                    with pytest.raises(ValueError, match="product geometry"):
                        context.prepare(invalid, simulated=True)
                pending = context.prepare(task, simulated=True)
                assert context.part_tracker[SQUARE]["processCompleted"] == completed
                if task["event_name"] == "place_insert":
                    before = deepcopy(context.part_tracker)
                    with pytest.raises(ValueError, match="Acknowledgement"):
                        context.acknowledge({**pending, "status": "failed"})
                    assert context.part_tracker == before
                context.acknowledge({**pending, "status": "completed"})
                if schema_version == 3 and task["event_name"] == "place_insert":
                    completed.append({"process": "assembly", "target": task["parameters"]["target"]})
                assert context.part_tracker[SQUARE]["processCompleted"] == completed
            assert context.part_tracker[SQUARE]["state"] == "assembled"
            assert (
                context.part_tracker[SQUARE]["target"]
                == inputs["geometry"]["parts"]["assembly_target_map"][SQUARE]
            )
            result = await explore(runtime, driver)
            assert result["status"] == "planned", result
            for task in result["tasks"]:
                context.acknowledge(
                    {**context.prepare(task, simulated=True), "status": "completed"}
                )
            assert context.outstanding() is None
            assert context.report()["schema_version"] == schema_version
            verify_environment_run(context.report())

    asyncio.run(scenario())


def test_process_plan_steps_are_conjunctions_and_later_effects_are_blocked(inputs):
    steps = inputs["product_order"]["processPlan"][SQUARE]
    steps[0]["processesToComplete"].append({"process": "trim", "result": "circle"})
    context = EnvironmentProductContext(**inputs)
    desired = context.outstanding()[1]
    options = list(candidates(context.models["M1"], context.snapshot(), SQUARE, desired, []))
    assert {task["parameters"]["result"] for task in options
            if task["event_name"] == "machine_part"} == {"square", "circle"}
    context.part_tracker[SQUARE]["processCompleted"] = [{"process": "trim", "result": "square"}]
    assert not matches_requirement(context.part_tracker[SQUARE], desired)
    assert context.outstanding() == (SQUARE, desired)
    assembly = context.requirements[SQUARE][1]
    task = next(task for task in candidates(
        context.models["ur5e-3"], context.snapshot(), SQUARE, assembly, []
    ) if task["event_name"] == "place_insert")
    with pytest.raises(ValueError, match="current processPlan step"):
        project_transition(context.models, context.snapshot(), context.part_tracker, task,
                           context.product_name, context.requirements)
    context.part_tracker[SQUARE]["processCompleted"].append({"process": "trim", "result": "circle"})
    assert context.outstanding() == (SQUARE, assembly)
    assert assembly["processesToComplete"] == [
        {"process": "assembly", "target": inputs["geometry"]["parts"]["assembly_target_map"][SQUARE]}
    ]
    assert inputs["product_order"]["processPlan"][SQUARE][1] == {
        "processesToComplete": [{"process": "assembly"}]
    }


def test_process_matching_uses_declared_effects_and_preserves_exact_results(inputs):
    context = EnvironmentProductContext(**inputs)
    machine = deepcopy(context.models["M1"])
    desired = {"process": "trim", "result": "square"}
    task = next(task for task in candidates(machine, context.snapshot(), SQUARE, desired, [])
                if task["event_name"] == "machine_part")
    assert task["parameters"]["result"] == "square"
    event = next(event for event in machine["events"] if event["event_name"] == "machine_part")
    event["product_effects"]["processCompleted"][0]["result"] = "circle"
    assert not any(task["event_name"] == "machine_part"
                   for task in candidates(machine, context.snapshot(), SQUARE, desired, []))
    assert not matches_requirement({"processCompleted": [desired]}, {"process": "trim", "result": "Square"})
    release = process_json(context.models, "place_release")
    assert all(event["product_effects"]["processCompleted"] == [] for event in release["events"])
    report = context.report()
    report["schema_version"] = 2
    with pytest.raises(ValueError, match="process semantics"):
        verify_environment_run(report)


def test_alternative_machine_and_unknown_setup(inputs):
    inputs["scene"]["machines"][1]["current_configuration"] = deepcopy(
        inputs["scene"]["machines"][0]["current_configuration"]
    )

    async def scenario():
        permitted = [rid for rid in build_environment_models(inputs["scene"]) if rid != "M1"]
        async with network(inputs, permitted) as (runtime, driver):
            result = await explore(runtime, driver)
            assert result["status"] == "planned", result
            assert result["tasks"][-1]["resource_id"] == "M2"
        for machine in inputs["scene"]["machines"]:
            machine["current_configuration"] = {}
        async with network(inputs) as (runtime, driver):
            result = await explore(runtime, driver)
            assert result["status"] == "needs_context", result
            assert result["unresolved"]
            assert runtime.context.revision == 0

    asyncio.run(scenario())


def test_added_registered_part_needs_no_resource_definition_change(inputs):
    new_part = "new_workpiece"
    geometry = inputs["geometry"]
    geometry["assembly_board"]["slots"][new_part] = deepcopy(
        geometry["assembly_board"]["slots"][SQUARE]
    )
    for field in (
        "dimensions_m",
        "model_map",
        "initial_source_resource_map",
        "assembly_target_map",
    ):
        geometry["parts"][field][new_part] = deepcopy(geometry["parts"][field][SQUARE])
    geometry["parts"]["assembly_target_map"][new_part] = "GMC_Laser_Plate_Virtual/new_workpiece"
    inputs["scene"]["Storage"]["slots"][new_part] = deepcopy(
        inputs["scene"]["Storage"]["slots"][SQUARE]
    )
    inputs["product_order"]["parts"] = [new_part]
    inputs["product_order"]["processPlan"][new_part] = deepcopy(
        inputs["product_order"]["processPlan"][SQUARE]
    )

    async def scenario():
        async with network(inputs) as (runtime, driver):
            result = await explore(runtime, driver)
            assert result["status"] == "planned", result
            assert result["tasks"][-1]["parameters"]["part_name"] == new_part

    asyncio.run(scenario())


def test_robot_transport_requires_declared_connection(inputs):
    async def scenario():
        permitted = [rid for rid in build_environment_models(inputs["scene"]) if rid != "KMR"]
        async with network(inputs, permitted) as (runtime, driver):
            assert (await explore(runtime, driver))["status"] == "blocked"
        inputs["scene"]["robots"][0]["handling_connections"] = [
            {"origin_resource_location": "Storage", "destination_location": "M1"}
        ]
        async with network(inputs, permitted) as (runtime, driver):
            result = await explore(runtime, driver)
            assert result["status"] == "planned", result
            assert {task["resource_id"] for task in result["tasks"]} == {"ur5e-1", "M1"}

    asyncio.run(scenario())


def test_geometry_and_missing_connection_block_assembly(inputs):
    async def scenario():
        async with network(inputs) as (runtime, driver):
            context = runtime.context
            result = await explore(runtime, driver)
            for task in result["tasks"]:
                context.acknowledge(
                    {**context.prepare(task, simulated=True), "status": "completed"}
                )
            context.permitted_resources.remove("Conveyor")
            assert (await explore(runtime, driver))["status"] == "blocked"
            context.permitted_resources.append("Conveyor")
            context.geometry[SQUARE]["dimensions_m"] = [0.04, 0.04, 0.05]
            result = await explore(runtime, driver)
            assert result["status"] == "blocked", result
            assert any(
                "cross section" in row["reason"] for row in context.environment_model["rejections"]
            )
            assert context.part_tracker[SQUARE]["state"] == "completed"

    asyncio.run(scenario())


def test_stale_ack_unknown_identity_and_controller_evidence_never_commit(inputs):
    async def scenario():
        async with network(inputs) as (runtime, driver):
            context = runtime.context
            result = await explore(runtime, driver)
            task = result["tasks"][0]
            invalid = deepcopy(task)
            invalid["parameters"]["part_name"] = "unknown"
            before = context.snapshot()
            with pytest.raises(ValueError):
                context.prepare(invalid, simulated=True)
            assert context.snapshot() == before
            with pytest.raises(ValueError, match="execution adapter"):
                context.prepare(task)
            pending = context.prepare(task, simulated=True)
            context.resources["M1"].model["current_configuration"]["tool"] = "changed"
            with pytest.raises(ValueError, match="current configuration"):
                context.acknowledge({**pending, "status": "completed"})
            assert context.snapshot() == before and context.revision == 0

    asyncio.run(scenario())


def test_timeout_closes_late_reply_window(inputs):
    async def scenario():
        async with network(inputs) as (runtime, driver):
            custodian = driver.agent.container.get_agent(runtime.jids["Storage"])
            custodian.dispatch = lambda msg: None
            before = runtime.context.snapshot()
            result = await explore(runtime, driver, timeout=0.02)
            assert result["status"] == "timeout" and result["tasks"] == []
            assert runtime.context.environment_model["closed"]
            assert runtime.context.snapshot() == before

    asyncio.run(scenario())


def test_request_sender_and_revisions_are_checked(inputs):
    async def scenario():
        async with network(inputs) as (runtime, driver):
            request = runtime.context.request(time.time() + 5)
            resource = driver.agent.container.get_agent(runtime.jids["Storage"])
            inbox = next(b for b in resource.behaviours if isinstance(b, CapabilityRequestInbox))
            with pytest.raises(ValueError, match="sender"):
                await inbox.handle(runtime, "unrelated@localhost", request)
            runtime.context.resources["M1"].model["current_configuration"]["tool"] = "changed"
            assert not runtime.context.current_request(request)
            assert runtime.context.revision == 0

    asyncio.run(scenario())


def test_resource_execution_requires_cca_and_validated_ack(inputs, monkeypatch):
    from cais_spade_llm.recovery_framework import environment_runtime as execution

    async def scenario():
        async with network(inputs) as (runtime, driver):
            result = await explore(runtime, driver)
            task = result["tasks"][0]
            context = runtime.context
            resource = driver.agent.container.get_agent(runtime.jids[task["resource_id"]])
            actor = resource.environment_context
            controller = AsyncMock(return_value={"observed": "custody transfer"})
            start = AsyncMock(return_value=True)
            actor.bind_executor(
                task["event_name"],
                controller,
                lambda task, evidence: evidence == {"observed": "custody transfer"},
                validate_start=start,
            )
            pending = context.prepare(task)
            with pytest.raises(ValueError, match="not validated"):
                context.acknowledge({**pending, "status": "completed"})
            packets = []

            async def capture(behaviour, msg, **kwargs):
                packets.append((msg.metadata["type"], json.loads(msg.body)))

            monkeypatch.setattr(execution, "send_agent_message", capture)
            resource._wait_for_safety_decision = AsyncMock(return_value="block")
            incoming = message(str(resource.jid), "task", pending)
            incoming.sender = runtime.product_jid
            behaviour = SimpleNamespace(agent=resource)
            before = context.snapshot()
            await execution.execute_environment_task(behaviour, incoming, pending)
            controller.assert_not_awaited()
            assert packets[-1][1]["status"] == "blocked"
            resource._wait_for_safety_decision = AsyncMock(return_value="allow")
            await execution.execute_environment_task(behaviour, incoming, pending)
            assert context.snapshot() == before
            acknowledgement = packets[-1][1]["acknowledgement"]
            assert context.acknowledge(acknowledgement)
            await execution.execute_environment_task(behaviour, incoming, pending)
            assert not context.acknowledge(packets[-1][1]["acknowledgement"])
            assert controller.await_count == 1
            assert [data["status"] for kind, data in packets if kind == "resource_event"] == [
                "safety_check",
                "safety_check",
                "running",
                "completed",
            ]

    asyncio.run(scenario())


def test_runtime_export_and_ui_use_the_evaluated_model(inputs, tmp_path, monkeypatch):
    from cais_spade_llm.recovery_framework import environment_runtime
    from cais_spade_llm.ui.components.nominal_resource_des import (
        environment_capability_mermaid,
        nominal_inventory_rows,
        nominal_state_rows,
    )

    monkeypatch.setattr(environment_runtime, "RUN_DIRECTORY", tmp_path)

    async def scenario():
        async with network(inputs) as (runtime, driver):
            await explore(runtime, driver)
            runtime.save()
            snapshot = runtime.snapshot()
            assert snapshot["environment_model"]["selected_path"]
            diagram = environment_capability_mermaid(snapshot["environment_model"])
            assert diagram.count("-->") == len(snapshot["environment_model"]["selected_path"])
            assert "machine_part" in diagram and "KMR" in diagram
            assert all(
                edge["prerequisites"] and "predicted_effects" in edge
                for edge in snapshot["environment_model"]["edges"]
            )
            assert all(
                not row["initial"].startswith("KET")
                for row in nominal_state_rows(snapshot["models"][BUFFER])
            )
            assert not nominal_inventory_rows(snapshot["models"][BUFFER])
            for name in ("machine_part", "advance_conveyor", "advance_part", "place_insert"):
                exported = read_json(runtime.path / "processes" / f"{name}.json")
                assert exported == process_json(runtime.context.models, name)
            release = process_json(runtime.context.models, "place_release")
            assert any("Conveyor" in event["collection_guards"] for event in release["events"])

    asyncio.run(scenario())


def test_startup_factories_install_environment_protocol_and_existing_fsa(
    inputs, tmp_path, monkeypatch
):
    from cais_spade_llm import agent_creator
    from cais_spade_llm.recovery_framework import delivery, environment_runtime
    from cais_spade_llm.ui import recovery_setup

    monkeypatch.setattr(delivery, "prepared_start", lambda: None)
    monkeypatch.setattr(environment_runtime, "_prepared", None)
    monkeypatch.setattr(recovery_setup, "validate_setup", lambda setup: deepcopy(inputs))
    setup = {"permitted_resources": list(build_environment_models(inputs["scene"]))}
    environment_runtime.prepare_environment_start(setup)
    order_path = tmp_path / "order.json"
    order_path.write_text(json.dumps(inputs["product_order"]))
    cca_path = str(ROOT / "cais_spade_llm/initialization/cca.json")

    async def scenario():
        resources = agent_creator.create_resource_agents([], cca_path)
        assert [resource.agent_name for resource in resources] == setup["permitted_resources"]
        assert len(resources) == 12
        (product,) = agent_creator.create_product_agents(
            [str(PRODUCT_PATH)], resources, cca_path, product_order_file=str(order_path)
        )
        runtime = product.environment_runtime
        assert runtime.context.explorations == []
        assert runtime.context.revision == 0
        assert product.resource_jids == [str(resource.jid) for resource in resources]
        await product.setup()
        for resource in resources:
            await resource.setup()
            assert any(isinstance(inbox, CapabilityRequestInbox) for inbox in resource.behaviours)
        assert any(isinstance(inbox, CapabilityReplyInbox) for inbox in product.behaviours)
        assert not any(isinstance(inbox, product._Kickoff) for inbox in product.behaviours)
        context = runtime.context
        task = next(
            offer["task"]
            for offer in context.resources["KMR"].alternatives(
                context, context.snapshot(), context.part_tracker, SQUARE, context.outstanding()[1]
            )
            if offer.get("status") == "FEASIBLE"
        )
        runtime.set_plan(task)
        payload = product._build_plan_validation_payload(request_id="checked-plan")
        assert payload["request_id"] == "checked-plan"
        assert payload["plan"]["nodes"][0]["params"] == task["parameters"]
        assert payload["fsa"]
        assert context.revision == 0 and context.pending is None

    asyncio.run(scenario())
    environment_runtime.prepare_environment_start(None)


def test_changed_conditions_trigger_discovery_and_executable_alternative(
    inputs, tmp_path, monkeypatch
):
    from cais_spade_llm.recovery_framework import environment_runtime

    monkeypatch.setattr(environment_runtime, "RUN_DIRECTORY", tmp_path)
    inputs["scene"]["machines"][1]["current_configuration"] = deepcopy(
        inputs["scene"]["machines"][0]["current_configuration"]
    )

    async def scenario():
        async with network(inputs) as (runtime, driver):
            loop = environment_runtime.EnvironmentProductLoop()
            loop.set_agent(driver.agent)
            await loop.run()
            assert runtime.outcome["status"] == "execution_unavailable"
            assert not runtime.outcome["executable"]
            first_request = runtime.context.environment_model["request_id"]
            await loop.run()
            assert runtime.context.environment_model["request_id"] == first_request
            runtime.context.resources["M1"].model["current_configuration"]["tool"] = "incompatible"
            await loop.run()
            assert runtime.context.environment_model["request_id"] != first_request
            assert runtime.outcome["tasks"][-1]["resource_id"] == "M2"
            for rid in ("KMR", "M2"):
                actor = runtime.context.resources[rid]
                for name in actor.model["local_event_alphabet"]:
                    actor.bind_executor(
                        name,
                        AsyncMock(),
                        lambda task, evidence: False,
                        validate_start=AsyncMock(return_value=False),
                    )
            # Both machines can establish the result, but only M2 has an adapter.
            runtime.context.resources["M1"].model["current_configuration"]["tool"] = "test_tool"
            result = await explore(runtime, driver)
            assert result["executable"] and result["tasks"][-1]["resource_id"] == "M2"
            assert runtime.context.revision == 0

    asyncio.run(scenario())


def test_shared_handoff_disagreement_cannot_commit(inputs):
    context = EnvironmentProductContext(**inputs)
    task = next(
        offer["task"]
        for offer in context.resources["KMR"].alternatives(
            context, context.snapshot(), context.part_tracker, SQUARE, context.outstanding()[1]
        )
        if offer["task"]["event_name"] == "pick_part"
    )
    participant = next(
        event
        for event in context.models["Storage"]["events"]
        if event["event_id"] == task["event_id"]
    )
    participant["parameter_bindings"]["part_name"] = {"equals": "another_part"}
    before = context.snapshot()
    with pytest.raises(ValueError, match="participants disagree"):
        context.prepare(task, simulated=True)
    assert context.snapshot() == before and context.pending is None


def test_task_ack_and_cca_exchange_uses_real_agent_inboxes(inputs, tmp_path, monkeypatch):
    from cais_spade_llm.recovery_framework import environment_runtime

    monkeypatch.setattr(environment_runtime, "RUN_DIRECTORY", tmp_path)
    checks = []

    class SafetyInbox(CyclicBehaviour):
        async def run(self):
            packet = await self.receive(timeout=1)
            if packet is None:
                return
            body = json.loads(packet.body)
            if packet.metadata["type"] == "plan_safety_check":
                checks.append(body)
                await send_agent_message(
                    self,
                    message(
                        str(packet.sender),
                        "plan_safety_result",
                        {"ok": True, "request_id": body["request_id"]},
                    ),
                )
            elif body["status"] == "safety_check":
                await send_agent_message(
                    self,
                    message(
                        str(packet.sender),
                        "safety_decision",
                        {"decision": "allow", "task_id": body["task_id"]},
                    ),
                )

    async def consume(inbox):
        while True:
            await inbox.run()

    async def scenario():
        async with network(inputs) as (runtime, driver):
            product = driver.agent
            cca = ResourceAgent("cca@localhost", "none", name="cca")
            cca.container = product.container
            cca.container.agents[str(cca.jid)] = cca
            safety = SafetyInbox()
            cca.add_behaviour(
                safety,
                Template(metadata={"type": "plan_safety_check"})
                | Template(metadata={"type": "resource_event"}),
            )
            inboxes = [safety]
            for rid in ("KMR", "M1"):
                agent = product.container.get_agent(runtime.jids[rid])
                for name in agent.environment_context.model["local_event_alphabet"]:

                    async def execute(task):
                        return {"task_id": task["task_id"]}

                    agent.environment_context.bind_executor(
                        name,
                        execute,
                        lambda task, evidence: evidence == {"task_id": task["task_id"]},
                        validate_start=AsyncMock(return_value=True),
                    )
                for inbox, kind in (
                    (agent._TaskInbox(), "task"),
                    (agent._SafetyDecisionInbox(), "safety_decision"),
                ):
                    agent.add_behaviour(inbox, Template(metadata={"type": kind}))
                    inboxes.append(inbox)
            loop = environment_runtime.EnvironmentProductLoop()
            product.add_behaviour(
                loop,
                Template(metadata={"type": "plan_safety_result"})
                | Template(metadata={"type": "ack"}),
            )
            workers = [asyncio.create_task(consume(inbox)) for inbox in inboxes]
            try:
                await asyncio.wait_for(loop.run(), 45)
                assert runtime.context.revision == 4
                assert product.part_tracker[SQUARE]["processCompleted"] == [
                    {"process": "trim", "result": "square"}
                ]
                assert len(checks) == 4 and all(check["request_id"] for check in checks)
                assert runtime.context.pending is None
                assert runtime.outcome["status"] == "execution_unavailable"
                assert all(
                    row["acknowledgement"]["evidence"] == "resource"
                    for row in runtime.context.transitions
                )
            finally:
                for worker in workers:
                    worker.cancel()
                await asyncio.gather(*workers, return_exceptions=True)

    asyncio.run(scenario())



def test_environment_display_revision_tracks_discovery_without_full_model_copy(inputs, monkeypatch):
    async def check():
        async with network(inputs) as (runtime, driver):
            initial = runtime.snapshot_revision()
            # This display query must never serialize all capability definitions.
            monkeypatch.setattr(runtime.context, "revisions", lambda: (_ for _ in ()).throw(AssertionError("full models read")))
            assert runtime.snapshot_revision() == initial
            runtime.context.environment_model["edges"].append({"task": "candidate"})
            discovered = runtime.snapshot_revision()
            assert discovered != initial
            runtime.outcome = {"status": "needs_context"}
            assert runtime.snapshot_revision() != discovered
            configured = runtime.snapshot_revision()
            runtime.context.resources["M1"].model["current_configuration"]["tool"] = "changed"
            assert runtime.snapshot_revision() != configured

    asyncio.run(check())
