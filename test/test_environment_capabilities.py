"""Runtime environmental discovery, resource ownership, and v2 acknowledgement evidence."""

from __future__ import annotations

import asyncio
import json
import time
from contextlib import asynccontextmanager
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from spade.behaviour import CyclicBehaviour
from spade.template import Template

from cais_spade_llm.agents.intelligent_product.product_agent import ProductAgent
from cais_spade_llm.agents.resource_agent.resource_agent import ResourceAgent
from cais_spade_llm.agents.shared_information.environment_capabilities import (
    CapabilityReplyInbox,
    CapabilityRequestInbox,
    explore,
    match_intake,
    message,
)
from cais_spade_llm.agents.shared_information.local_dispatch import send_agent_message
from cais_spade_llm.product.environment import EnvironmentProductContext, verify_environment_run
from cais_spade_llm.product.order import validate_product_order
from cais_spade_llm.recovery_framework import PRODUCT_PATH, ROOT, SCENE_PATH, fingerprint, read_json
from cais_spade_llm.recovery_framework.environment_runtime import EnvironmentRuntime
from cais_spade_llm.resources.environment_models import (
    build_environment_models,
    candidates,
    matches_requirement,
    process_json,
    project_transition,
    validate_environment_composition,
)

SQUARE = "KET4_Square_4mm"
BUFFER = "Buffer For Machined parts"


@pytest.fixture(autouse=True)
def isolated_environment_reports(tmp_path, monkeypatch):
    from cais_spade_llm.recovery_framework import environment_runtime
    from cais_spade_llm.agents.resource_agent.robot_agent import RobotAgent

    monkeypatch.setattr(environment_runtime, 'RUN_DIRECTORY', tmp_path / 'environment_runs')
    # These tests exercise agent protocols, never ROS controller construction.
    monkeypatch.setattr(RobotAgent, '_build_controller', lambda self: None)


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
async def network(inputs, permitted=None, *, diagnostic_cca_bypass=False):
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
        "setup": {
            "permitted_resources": permitted or [r.agent_name for r in resources],
            "execution_mode": "simulation",
        },
        "diagnostic_cca_bypass": diagnostic_cca_bypass,
        "launch_identity": ("test-simulation", 42),
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

    closed = asyncio.Event()

    async def consume(inbox):
        while not closed.is_set():
            await inbox.run()

    workers = [asyncio.create_task(consume(inbox)) for inbox in inboxes]
    driver = SimpleNamespace(
        agent=product, send=AsyncMock(side_effect=AssertionError("Unexpected XMPP send"))
    )
    try:
        yield runtime, driver
    finally:
        closed.set()
        for worker in workers:
            worker.cancel()
        await asyncio.gather(*workers, return_exceptions=True)
        runtime._close_calculations()
        await runtime.flush_reports()


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


def test_composition_event_ids_and_participants_are_consistent(inputs):
    models = build_environment_models(inputs["scene"])
    validate_environment_composition(models)
    homes = {
        rid: next(event for event in models[rid]["events"] if event["event_name"] == "move_home")
        for rid in ("ur5e-1", "ur5e-2", "ur5e-3", "ur5e-4")
    }
    assert len({event["event_id"] for event in homes.values()}) == 4
    assert all(event["participants"] == [rid] for rid, event in homes.items())
    releases = [
        event for event in models["KMR"]["events"]
        if event["event_name"] == "place_release"
    ]
    assert len({event["event_id"] for event in releases}) == len(releases)
    assert {tuple(event["participants"]) for event in releases} == {
        ("M1", "KMR"), ("M2", "KMR")
    }

    pick = next(event for event in models["KMR"]["events"] if event["event_name"] == "pick_part")
    event_id = pick["event_id"]
    missing = deepcopy(models)
    missing["Storage"]["events"] = [
        event for event in missing["Storage"]["events"] if event["event_id"] != event_id
    ]
    with pytest.raises(ValueError, match="participants disagree"):
        validate_environment_composition(missing)

    duplicate = deepcopy(models)
    peer = next(event for event in duplicate["Storage"]["events"] if event["event_id"] == event_id)
    duplicate["Storage"]["events"].append(deepcopy(peer))
    with pytest.raises(ValueError, match="participants disagree"):
        validate_environment_composition(duplicate)

    mismatched = deepcopy(models)
    peer = next(event for event in mismatched["Storage"]["events"] if event["event_id"] == event_id)
    peer["parameter_bindings"]["handoff_acknowledged"] = {"equals": False}
    with pytest.raises(ValueError, match="variants disagree"):
        validate_environment_composition(mismatched)

    undeclared = deepcopy(models)
    peer = next(event for event in undeclared["Storage"]["events"] if event["event_id"] == event_id)
    peer["updates"]["resource_state"] = {"set": "idle"}
    with pytest.raises(ValueError, match="Undeclared composition field"):
        validate_environment_composition(undeclared)


def test_same_named_home_events_do_not_synchronize_unrelated_robots(inputs):
    context = EnvironmentProductContext(**inputs)
    home = next(
        event for event in context.models["ur5e-1"]["events"]
        if event["event_name"] == "move_home"
    )
    parameters = {
        name: binding["equals"] for name, binding in home["parameter_bindings"].items()
    }
    task = {
        "resource_id": "ur5e-1", "event_id": home["event_id"],
        "event_name": "move_home", "parameters": parameters,
    }
    before = context.snapshot()
    after, _ = project_transition(
        context.models, before, context.part_tracker, task,
        context.product_name, context.requirements,
    )
    assert after["ur5e-1"]["resource_location"] == "home"
    assert all(
        after[rid] == before[rid] for rid in before if rid != "ur5e-1"
    )
    other = next(
        event for event in context.models["ur5e-2"]["events"]
        if event["event_name"] == "move_home"
    )
    with pytest.raises(ValueError, match="Unknown resource event"):
        project_transition(
            context.models, before, context.part_tracker,
            {**task, "event_id": other["event_id"]},
            context.product_name, context.requirements,
        )


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
def test_distributed_square_trim_then_stationary_assembly(inputs, schema_version):
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
            for actor in context.resources.values():
                for name in actor.model['local_event_alphabet']:
                    actor.bind_executor(name, AsyncMock(return_value={}), lambda _task, _evidence: True,
                                        validate_start=lambda _task, _state, _geometry: True)
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
            assert result == {"status": "completed", "tasks": []}
            assert context.outstanding() is None
            assert context.part_tracker[context.product_name]["location"] == context.product_name
            assert context.report()["schema_version"] == schema_version
            verify_environment_run(context.report())

    asyncio.run(scenario())


def test_one_part_round_order_selects_M1_circle_program_per_run(inputs):
    from cais_spade_llm.ui import recovery_setup

    order_path = (
        ROOT / "cais_spade_llm/specification/products/orders/assembly_board-v1-round-4mm-m1.json"
    )
    full_order = deepcopy(inputs["product_order"])
    inputs["product_order"] = read_json(order_path)
    original = deepcopy(inputs["scene"])
    context = EnvironmentProductContext(**inputs)
    assert context.machine_resource == "M1"
    assert context.models["M1"]["current_configuration"]["program"]["effects"] == [
        {"process": "trim", "result": "circle"}
    ]
    assert inputs["scene"] == original
    assert original["machines"][0]["current_configuration"]["program"]["effects"] == [
        {"process": "trim", "result": "square"}
    ]
    full_context = EnvironmentProductContext(
        original, full_order, inputs["geometry"]
    )
    assert full_context.models["M1"]["current_configuration"]["program"]["effects"] == [
        {"process": "trim", "result": "square"}
    ]

    setup = recovery_setup.default_setup()
    setup["selected_product_order_file"] = str(order_path.relative_to(ROOT))
    checked = recovery_setup.validate_setup(setup)
    assert checked["scene"]["machines"][0]["current_configuration"]["program"]["effects"] == [
        {"process": "trim", "result": "circle"}
    ]
    assert read_json(ROOT / setup["scene_file"])["machines"][0][
        "current_configuration"
    ]["program"]["effects"] == [{"process": "trim", "result": "square"}]

    unavailable = deepcopy(inputs)
    unavailable["scene"]["machines"][0].pop("program_options")
    with pytest.raises(ValueError, match="no program"):
        EnvironmentProductContext(**unavailable)


def test_one_part_round_gazebo_accepts_only_saved_M1_circle_option(inputs):
    from cais_spade_llm.recovery_framework.kmr_gazebo import _expected_scene_fingerprint

    scene = read_json(SCENE_PATH)
    order = read_json(
        ROOT / "cais_spade_llm/specification/products/orders/assembly_board-v1-round-4mm-m1.json"
    )
    selected = EnvironmentProductContext(scene, order, inputs["geometry"]).inputs["scene"]
    assert _expected_scene_fingerprint(selected, order) == fingerprint(scene)
    assert _expected_scene_fingerprint(scene, inputs["product_order"]) == fingerprint(scene)

    altered = deepcopy(selected)
    altered["KMR"]["initial_pose"][0] += 0.1
    with pytest.raises(ValueError, match="outside the bound machine program"):
        _expected_scene_fingerprint(altered, order)


def test_one_part_round_route_negotiates_empty_KMR_return_and_assembles(inputs):
    from cais_spade_llm.recovery_framework.environment_runtime import (
        EnvironmentProductLoop,
        _kmr_at_storage,
        _robots_at_home,
    )

    part = "RGOCG4-50_Round_4mm"
    inputs["product_order"] = read_json(
        ROOT / "cais_spade_llm/specification/products/orders/assembly_board-v1-round-4mm-m1.json"
    )

    async def scenario():
        async with network(inputs) as (runtime, driver):
            context = runtime.context
            loop = EnvironmentProductLoop()
            home_goals = [goal for _, _, _, goal in loop._negotiation_goals(runtime) if goal]
            assert {goal["resource_id"] for goal in home_goals} == {
                row["resource_id"] for row in inputs["scene"]["robots"]
            }
            for goal in home_goals:
                result = await explore(runtime, driver, part=part,
                                       desired=context.requirements[part][-1], resource_goal=goal)
                assert result["tasks"][0]["event_name"] == "move_home"
                pending = context.prepare(result["tasks"][0], simulated=True)
                assert not any(key.startswith("part:") for key in pending["reservations"])
                context.acknowledge({**pending, "status": "completed"})
            assert _robots_at_home(context)
            first = await explore(runtime, driver, timeout=20)
            assert first["status"] == "planned"
            assert [(task["resource_id"], task["event_name"]) for task in first["tasks"]] == [
                ("KMR", "pick_part"),
                ("KMR", "move_to_resource"),
                ("KMR", "place_release"),
                ("M1", "machine_part"),
            ]
            assert first["tasks"][1]["parameters"]["target_resource"] == "M1"
            assert first["tasks"][2]["parameters"]["destination_location"] == "M1"

            for task in first["tasks"][:3]:
                pending = context.prepare(task, simulated=True)
                context.acknowledge({**pending, "status": "completed"})
            assert context.snapshot()["KMR"]["resource_location"] == "M1"
            assert not _kmr_at_storage(context)

            returned = await explore(runtime, driver, part=part,
                                     desired=context.requirements[part][-1], resource_goal={
                "resource_id": "KMR", "values": {
                    "resource_location": "Storage", "resource_state": "idle", "held_part": None,
                },
            })
            assert returned["status"] == "planned"
            return_task = returned["tasks"][0]
            assert return_task["resource_id"] == "KMR"
            assert return_task["event_name"] == "move_to_resource"
            assert return_task["parameters"]["source_resource"] == "M1"
            assert return_task["parameters"]["target_resource"] == "Storage"
            pending = context.prepare(return_task, simulated=True)
            before = context.snapshot()
            with pytest.raises(ValueError, match="Acknowledgement"):
                context.acknowledge({**pending, "status": "failed"})
            assert context.snapshot() == before
            assert not _kmr_at_storage(context)
            context.acknowledge({**pending, "status": "completed"})
            assert _kmr_at_storage(context)

            pending = context.prepare(first["tasks"][3], simulated=True)
            context.acknowledge({**pending, "status": "completed"})
            assert context.part_tracker[part]["processCompleted"] == [
                {"process": "trim", "result": "circle"}
            ]
            second = await explore(runtime, driver, timeout=20)
            assert second["status"] == "planned"
            actors = [task["resource_id"] for task in second["tasks"]]
            assert list(dict.fromkeys(actors)) == [
                "ur5e-1", "Conveyor", BUFFER, "ur5e-3"
            ]
            assert any(
                task["parameters"].get("destination_location") == "Conveyor"
                for task in second["tasks"] if task["resource_id"] == "ur5e-1"
            )
            assert second["tasks"][-1]["event_name"] == "place_insert"
            assert second["tasks"][-1]["parameters"]["destination_location"] == "assembly_board-v1"
            for index, task in enumerate(second["tasks"]):
                pending = context.prepare(task, simulated=True)
                context.acknowledge({**pending, "status": "completed"})
                if task["event_name"] not in {"place_release", "place_insert"}:
                    continue
                assert not _robots_at_home(context)
                goal = next(goal for key, _, _, goal in loop._negotiation_goals(runtime)
                            if key == f"resource:{task['resource_id']}")
                returned = await explore(runtime, driver, part=part,
                                         desired=context.requirements[part][-1], resource_goal=goal)
                assert len(returned["tasks"]) == 1
                home = context.prepare(returned["tasks"][0], simulated=True)
                assert home["event_name"] == "move_home"
                with pytest.raises(ValueError, match="Acknowledgement"):
                    context.acknowledge({**home, "status": "failed"})
                assert not _robots_at_home(context)
                if index + 1 < len(second["tasks"]):
                    # An empty return must not reserve the released part on the conveyor.
                    downstream = context.prepare(second["tasks"][index + 1], simulated=True)
                    context.cancel_pending(downstream["task_id"])
                context.acknowledge({**home, "status": "completed"})
                assert context.snapshot()[task["resource_id"]]["resource_location"] == "home"
            assert context.part_tracker[part]["state"] == "assembled"
            assert _robots_at_home(context)
            assert loop._negotiation_goals(runtime) == []
            assert context.outstanding() is None
            assert _kmr_at_storage(context)
            assert (await explore(runtime, driver))["status"] == "completed"
            runtime.outcome = {"status": "completed", "tasks": []}
            runtime.stop("Completed run owner exited")
            runtime.save()
            assert runtime.stopped
            assert read_json(runtime.path / "run.json")["outcome"]["status"] == "completed"

    asyncio.run(scenario())


def test_one_part_round_machine_binding_rejects_M2_tasks(inputs):
    inputs["product_order"] = read_json(
        ROOT / "cais_spade_llm/specification/products/orders/assembly_board-v1-round-4mm-m1.json"
    )
    context = EnvironmentProductContext(**inputs)
    event = next(
        event for event in context.models["KMR"]["events"]
        if event["event_name"] == "move_to_resource"
        and event["parameter_bindings"]["source_resource"]["equals"] == "Storage"
        and event["parameter_bindings"]["target_resource"]["equals"] == "M2"
    )
    task = {
        "resource_id": "KMR",
        "event_id": event["event_id"],
        "event_name": "move_to_resource",
        "parameters": {
            key: binding["equals"]
            for key, binding in event["parameter_bindings"].items()
        },
        "part_name": "RGOCG4-50_Round_4mm",
    }
    assert not context.allows_task(task)
    with pytest.raises(ValueError, match="bound machine lane"):
        context.prepare(task, simulated=True)


def test_shared_handoffs_check_all_guards_and_update_participants_atomically(inputs):
    context = EnvironmentProductContext(**inputs)
    models = context.models
    valuation = context.snapshot()
    products = context.part_tracker

    def task_for(resource_id, event_name, **parameters):
        event = next(
            event for event in models[resource_id]["events"]
            if event["event_name"] == event_name
            and all(
                "equals" not in event["parameter_bindings"].get(name, {})
                or event["parameter_bindings"][name]["equals"] == value
                for name, value in parameters.items()
            )
        )
        return {
            "resource_id": resource_id,
            "event_id": event["event_id"],
            "event_name": event_name,
            "parameters": {
                name: binding["equals"] if "equals" in binding else parameters[name]
                for name, binding in event["parameter_bindings"].items()
            },
        }

    def project(task, before, before_products, participants):
        after, predicted = project_transition(
            models, before, before_products, task, context.product_name,
            context.requirements,
        )
        assert all(after[rid] == before[rid] for rid in models if rid not in participants)
        return after, predicted

    def blocked_by_peer(task, before, before_products, peer_id, field, expected):
        altered = deepcopy(models)
        peer = next(event for event in altered[peer_id]["events"]
                    if event["event_id"] == task["event_id"])
        peer["guards"][field] = {"equals": expected}
        with pytest.raises(ValueError, match="Guard blocked"):
            project_transition(
                altered, before, before_products, task, context.product_name,
                context.requirements,
            )

    pick = task_for("KMR", "pick_part", origin_resource_location="Storage", part_name=SQUARE)
    blocked_by_peer(pick, valuation, products, "Storage", "inventory.{part_name}", False)
    valuation, products = project(pick, valuation, products, {"Storage", "KMR"})
    assert valuation["Storage"][f"inventory.{SQUARE}"] is False
    assert valuation["KMR"]["held_part"] == SQUARE

    route = task_for("KMR", "move_to_resource", source_resource="Storage", target_resource="M1")
    valuation, products = project(route, valuation, products, {"KMR"})
    release = task_for("KMR", "place_release", destination_location="M1", part_name=SQUARE)
    blocked_by_peer(release, valuation, products, "M1", "resource_state", "completed")
    valuation, products = project(release, valuation, products, {"KMR", "M1"})
    assert valuation["KMR"]["held_part"] is None
    assert valuation["M1"]["part_name"] == SQUARE

    machine = task_for("M1", "machine_part", part_name=SQUARE, process="trim", result="square")
    valuation, products = project(machine, valuation, products, {"M1"})
    assert {"process": "trim", "result": "square"} in products[SQUARE]["processCompleted"]
    valuation["ur5e-1"].update(resource_state="at_pick")
    valuation["ur5e-1"]["task_ctx.origin_resource_location"] = "M1"
    valuation["ur5e-1"]["task_ctx.part_name"] = SQUARE
    robot_pick = task_for("ur5e-1", "pick_grasp", origin_resource_location="M1", part_name=SQUARE)
    blocked_by_peer(robot_pick, valuation, products, "M1", "resource_state", "idle")
    valuation, products = project(robot_pick, valuation, products, {"ur5e-1", "M1"})
    assert valuation["ur5e-1"]["held_part"] == SQUARE
    assert valuation["M1"]["part_name"] is None

    # A coherent downstream pre-state isolates the Conveyor-Buffer handoff.
    valuation["ur5e-1"].update(resource_state="idle", held_part=None)
    valuation["Conveyor"][f"part_location.{SQUARE}"] = "output_nest"
    valuation["Conveyor"][f"part_order.{SQUARE}"] = 0
    delivered = task_for(
        "Conveyor", "advance_conveyor", delivered_part=SQUARE, next_locations={}
    )
    assert "delivered_part" in delivered["parameters"]
    assert "part_name" not in delivered["parameters"]
    buffer_event = next(event for event in models[BUFFER]["events"]
                        if event["event_id"] == delivered["event_id"])
    assert buffer_event["updates"]["zone_1_part"] == {"set_from_param": "delivered_part"}
    blocked_by_peer(delivered, valuation, products, BUFFER, "zone_1_part", SQUARE)
    after, _ = project(delivered, valuation, products, {"Conveyor", BUFFER})
    assert after[BUFFER]["zone_1_part"] == SQUARE
    assert after["Conveyor"][f"part_location.{SQUARE}"] is None
    assert after["Conveyor"][f"part_order.{SQUARE}"] is None


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


def test_stale_ack_checks_only_task_participants_and_never_commits(inputs):
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
            executable = context.prepare(task)
            assert executable["evidence"] == "resource"
            context.cancel_pending(executable["task_id"])
            pending = context.prepare(task, simulated=True)
            context.resources["M1"].model["current_configuration"]["tool"] = "changed"
            assert context.acknowledge({**pending, "status": "completed"})
            next_task = result["tasks"][1]
            pending = context.prepare(next_task, simulated=True)
            before = context.snapshot()
            context.resources["KMR"].model["current_configuration"] = {"changed": True}
            with pytest.raises(ValueError, match="current configuration"):
                context.acknowledge({**pending, "status": "completed"})
            assert context.snapshot() == before and context.revision == 1

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
            runtime.context.request_dependencies(request, ["M1"])
            runtime.context.resources["M1"].model["current_configuration"]["tool"] = "changed"
            assert not runtime.context.current_request(request)
            assert runtime.context.revision == 0

    asyncio.run(scenario())


@pytest.mark.parametrize("bypass", [False, True])
def test_resource_execution_requires_cca_and_validated_ack(inputs, monkeypatch, bypass):
    from cais_spade_llm.recovery_framework import environment_runtime as execution

    async def scenario():
        async with network(inputs, diagnostic_cca_bypass=bypass) as (runtime, driver):
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
            if bypass:
                start.return_value = False
            incoming = message(str(resource.jid), "task", pending)
            incoming.sender = runtime.product_jid
            behaviour = SimpleNamespace(agent=resource)
            before = context.snapshot()
            await execution.execute_environment_task(behaviour, incoming, pending)
            controller.assert_not_awaited()
            assert packets[-1][1]["status"] == "blocked"
            start.return_value = True
            resource._wait_for_safety_decision = AsyncMock(return_value="allow")
            await execution.execute_environment_task(behaviour, incoming, pending)
            assert context.snapshot() == before
            acknowledgement = packets[-1][1]["acknowledgement"]
            assert context.acknowledge(acknowledgement)
            await execution.execute_environment_task(behaviour, incoming, pending)
            assert not context.acknowledge(packets[-1][1]["acknowledgement"])
            assert controller.await_count == 1
            assert [data["status"] for kind, data in packets if kind == "resource_event"] == ([] if bypass else [
                "safety_check",
                "safety_check",
                "running",
                "completed",
            ])
            if bypass:
                resource._wait_for_safety_decision.assert_not_awaited()
                assert runtime._report_snapshot()[1]["diagnostic_cca_bypass"] is True

    asyncio.run(scenario())


def test_disjoint_tasks_commit_in_completion_order_and_conflicts_stay_reserved(inputs):
    gear = "gear_small"
    inputs["product_order"]["parts"] = [gear, SQUARE]

    async def scenario():
        async with network(inputs) as (runtime, driver):
            context = runtime.context
            gear_result, square_result = await asyncio.gather(
                explore(runtime, driver, part=gear, desired=context.requirements[gear][1]),
                explore(runtime, driver, part=SQUARE, desired=context.requirements[SQUARE][0]),
            )
            gear_task, square_task = gear_result["tasks"][0], square_result["tasks"][0]
            assert {gear_task["resource_id"], square_task["resource_id"]} == {"ur5e-4", "KMR"}
            assert len({row["request"]["request_id"] for row in context.explorations}) == 2
            assert not runtime.request_queues
            gear_pending = context.prepare(gear_task, simulated=True)
            square_pending = context.prepare(square_task, simulated=True)
            with pytest.raises(ValueError, match="active reservations"):
                context.prepare(square_task, simulated=True)
            square_ack = {**square_pending, "status": "completed"}
            gear_ack = {**gear_pending, "status": "completed"}
            assert context.acknowledge(square_ack)
            assert context.acknowledge(gear_ack)
            assert not context.acknowledge(gear_ack)
            assert context.pending is None
            assert [row["acknowledgement"]["task_id"] for row in context.transitions] == [
                square_pending["task_id"], gear_pending["task_id"],
            ]
    asyncio.run(scenario())


def test_kmr_empty_return_and_machining_dispatch_concurrently(inputs):
    from cais_spade_llm.recovery_framework.environment_runtime import EnvironmentProductLoop

    async def scenario():
        async with network(inputs) as (runtime, driver):
            context = runtime.context
            first = await explore(runtime, driver)
            for task in first["tasks"][:3]:
                pending = context.prepare(task, simulated=True)
                assert context.acknowledge({**pending, "status": "completed"})
            goals = EnvironmentProductLoop()._negotiation_goals(runtime)
            assert {SQUARE, "resource:KMR"} <= {key for key, *_ in goals}
            goals = [goal for goal in goals if goal[0] in {SQUARE, "resource:KMR"}]
            results = await asyncio.gather(*(
                explore(runtime, driver, part=part, desired=desired, resource_goal=goal)
                for key, part, desired, goal in goals
            ))
            tasks = [result["tasks"][0] for result in results]
            return_task = next(task for task in tasks if task["resource_id"] == "KMR")
            machine_task = next(task for task in tasks if task["resource_id"] == "M1")
            assert return_task["parameters"]["source_resource"] == "M1"
            assert return_task["parameters"]["target_resource"] == "Storage"
            assert machine_task["event_name"] == "machine_part"
            returned = context.prepare(return_task, simulated=True)
            machining = context.prepare(machine_task, simulated=True)
            assert len(context.pending_tasks) == 2
            assert not context._task_reservations(return_task, SQUARE) & context._task_reservations(machine_task, SQUARE)
            assert context.acknowledge({**machining, "status": "completed"})
            assert context.snapshot()["KMR"]["resource_location"] == "M1"
            # The next handoff can be negotiated before the empty return acknowledges.
            next_step = await explore(runtime, driver)
            assert next_step["tasks"][0]["resource_id"] == "ur5e-1"
            assert context.acknowledge({**returned, "status": "completed"})
            assert context.snapshot()["KMR"]["resource_location"] == "Storage"
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
            assert snapshot["initial_product_states"] == runtime.context.initial_product_states
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


def test_prepared_environment_controllers_transfer_once_and_match_start(monkeypatch):
    from cais_spade_llm.recovery_framework import environment_runtime, startup

    controller = Mock()
    prepared = {
        'setup': {'permitted_resources': ['ur5e-1']},
        'inputs': {'scene': 'snapshot'},
        'launch_identity': ('gazebo_dual', 42),
        'source_fingerprints': {'scene': 'abc'},
        'prepared_controller_token': 'prepared-once',
    }
    binding = environment_runtime._preparation_binding(
        prepared['setup'], prepared['inputs'], prepared['launch_identity'],
        prepared['source_fingerprints'],
    )
    environment_runtime._prepared_controllers['prepared-once'] = {
        'binding': binding,
        'controllers': {'ur5e-1': controller},
    }
    monkeypatch.setattr(
        startup, 'configuration_fingerprints',
        lambda setup: deepcopy(prepared['source_fingerprints']),
    )
    assert environment_runtime.claim_prepared_environment_controllers(prepared) == {
        'ur5e-1': controller,
    }
    assert environment_runtime.claim_prepared_environment_controllers(prepared) == {}
    controller.shutdown.assert_not_called()


def test_stop_during_environment_controller_preparation_rejects_late_owner(monkeypatch):
    from cais_spade_llm.recovery_framework import environment_runtime, workflow_execution
    from cais_spade_llm.resources.robot import gazebo_pick_place_controller

    controller = Mock()
    controller.wait_for_services.return_value = True
    monkeypatch.setattr(
        workflow_execution,
        '_robot_configuration',
        lambda scene, robot: ({
            'node_name': 'ur5e_1_workflow_controller',
            'arm_joint_names': ['joint'],
            'arm_trajectory_topic': '/arm/joint_trajectory',
        }, {'home': [0.]}, {}),
    )
    monkeypatch.setattr(
        gazebo_pick_place_controller,
        'GazeboPickPlaceController',
        lambda **kwargs: controller,
    )
    context = SimpleNamespace(
        inputs={'scene': {'robots': [{'resource_id': 'ur5e-1'}]}},
        models={'ur5e-1': {}},
    )
    started_generation = environment_runtime._discard_prepared_controllers()
    environment_runtime._discard_prepared_controllers()
    with pytest.raises(RuntimeError, match='cancelled before ownership transfer'):
        environment_runtime._prepare_environment_controllers(
            context,
            {'permitted_resources': ['ur5e-1']},
            ('gazebo_dual', 42),
            {'scene': 'abc'},
            started_generation,
        )
    controller.shutdown.assert_called_once()


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


@pytest.mark.parametrize("all_capabilities", [False, True, "eight", "eleven", "bypass"])
def test_task_ack_and_cca_exchange_uses_real_agent_inboxes(inputs, tmp_path, monkeypatch, all_capabilities):
    from cais_spade_llm.recovery_framework import environment_runtime

    monkeypatch.setattr(environment_runtime, "RUN_DIRECTORY", tmp_path)
    checks = []
    storage_discoveries = []
    active_storage_discoveries = set()
    multi_part = all_capabilities in ("eight", "eleven")
    active_source_discoveries = {}
    if all_capabilities == "bypass":
        inputs["product_order"] = read_json(
            ROOT / "cais_spade_llm/specification/products/orders/assembly_board-v1-two-parts.json"
        )
    if multi_part:
        inputs["product_order"] = read_json(
            ROOT / "cais_spade_llm/specification/products/orders" / (
                "assembly_board-v1-eight-pegs.json" if all_capabilities == "eight"
                else "assembly_board-v1-recovery-framework.json"
            )
        )
        negotiate = environment_runtime.EnvironmentProductLoop._negotiate_goal

        async def observe_discovery(loop, runtime, key, part, desired, resource_goal):
            intake = resource_goal is None and runtime.context.part_tracker[part]["location"] == "Storage"
            source = runtime.context.part_tracker[part]["location"]
            source_intake = (resource_goal is None and source == "3D Printing Station"
                             and part not in runtime.admitted_parts)
            if source_intake:
                active_source_discoveries.setdefault(source, set()).add(part)
                assert len(active_source_discoveries[source]) == 1
            if intake:
                active_storage_discoveries.add(part)
                storage_discoveries.append(part)
                assert len(active_storage_discoveries) == 1
            try:
                return await negotiate(loop, runtime, key, part, desired, resource_goal)
            finally:
                if intake:
                    active_storage_discoveries.remove(part)
                if source_intake:
                    active_source_discoveries[source].remove(part)

        monkeypatch.setattr(environment_runtime.EnvironmentProductLoop, "_negotiate_goal", observe_discovery)

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

    closed = asyncio.Event()

    async def consume(inbox):
        while not closed.is_set():
            await inbox.run()

    async def scenario():
        async with network(inputs, diagnostic_cca_bypass=all_capabilities == "bypass") as (runtime, driver):
            product = driver.agent
            cca = ResourceAgent("cca@localhost", "none", name="cca")
            cca.container = product.container
            if all_capabilities != "bypass":
                cca.container.agents[str(cca.jid)] = cca
            safety = SafetyInbox()
            cca.add_behaviour(
                safety,
                Template(metadata={"type": "plan_safety_check"})
                | Template(metadata={"type": "resource_event"}),
            )
            inboxes = [safety]
            handling_return_started = asyncio.Event()
            kmr_pick_started, gear_pick_started = asyncio.Event(), asyncio.Event()
            execution_starts = []
            for rid in (runtime.context.models if all_capabilities else ("KMR", "M1")):
                agent = product.container.get_agent(runtime.jids[rid])
                for name in agent.environment_context.model["local_event_alphabet"]:

                    async def execute(task):
                        context = runtime.context
                        rid, name = task["resource_id"], task["event_name"]
                        execution_starts.append((rid, name))
                        if all_capabilities == "bypass":
                            if rid == "KMR" and name == "pick_part":
                                kmr_pick_started.set()
                                await asyncio.wait_for(gear_pick_started.wait(), 10)
                            elif rid == "ur5e-4" and name == "pick_approach":
                                gear_pick_started.set()
                                await asyncio.wait_for(kmr_pick_started.wait(), 10)
                        if rid == "ur5e-1" and name == "move_home" and context.resources[rid].valuation["resource_state"] == "placed":
                            if not multi_part:
                                assert context.part_tracker[SQUARE]["state"] != "assembled"
                            handling_return_started.set()
                        if rid == "Conveyor" and not multi_part:
                            await asyncio.wait_for(handling_return_started.wait(), 10)
                        if rid == "ur5e-3" and name == "move_home" and context.resources[rid].valuation["resource_state"] == "placed":
                            if not multi_part:
                                assert context.part_tracker[task["part_name"]]["state"] == "assembled"
                            assert runtime.outcome["status"] != "completed"
                        if multi_part:
                            await asyncio.sleep(.1 if name == "machine_part" else .02)
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
                await asyncio.wait_for(loop.run(), 45 * len(runtime.context.selected_parts))
                if not all_capabilities:
                    assert runtime.context.revision == 5
                    assert product.part_tracker[SQUARE]["processCompleted"] == [
                        {"process": "trim", "result": "square"}
                    ]
                else:
                    assert handling_return_started.is_set()
                    assert product.part_tracker[SQUARE]["state"] == "assembled"
                    assert environment_runtime._robots_at_home(runtime.context)
                    assert runtime.outcome["status"] == "completed", runtime.outcome
                    if multi_part:
                        assert all(product.part_tracker[part]["state"] == "assembled"
                                   for part in runtime.context.selected_parts)
                        assignments = runtime.intake_assignments
                        assert set(assignments.values()) == {"M1", "M2"}
                        intake = [row for row in runtime.context.negotiations
                                  if row["kind"] == "dispatch" and row["event_name"] == "pick_part"]
                        assert {row["offered_machine"] for row in intake[:2]} == {"M1", "M2"}
                        assert len(intake) == 8
                        assert set(storage_discoveries) == set(inputs["scene"]["Storage"]["slots"])
                        assert not any(active_source_discoveries.values())
                        if all_capabilities == "eleven":
                            assert len(runtime.context.selected_parts) == 11
                            assert sum(rid == "ur5e-4" and event == "place_insert"
                                       for rid, event in execution_starts) == 3
                            assert not any(event == "print_part" for _, event in execution_starts)
                        assert not active_storage_discoveries
                        starts = {row["task_id"]: row for row in runtime.context.negotiations
                                  if row["kind"] == "execution_started"}
                        ends = {row["task_id"]: row["timestamp"] for row in runtime.context.negotiations
                                if row["kind"] == "execution_completed"}
                        assert any(
                            left["part_name"] != right["part_name"]
                            and left["resource_id"] != right["resource_id"]
                            and left["event_name"] != "move_home" and right["event_name"] != "move_home"
                            and max(left["timestamp"], right["timestamp"]) < min(ends[a], ends[b])
                            for a, left in starts.items() for b, right in starts.items() if a != b
                        )
                    release = execution_starts.index(("ur5e-1", "place_release"))
                    home = execution_starts.index(("ur5e-1", "move_home"), release)
                    assembly = execution_starts.index(("ur5e-3", "place_insert"))
                    assert release < home < assembly
                assert runtime.context.snapshot()["KMR"] == {
                    "resource_state": "idle", "held_part": None, "resource_location": "Storage"
                }
                events = [row["acknowledgement"]["event_name"]
                          for row in runtime.context.transitions]
                if not all_capabilities:
                    assert events[:3] == ["pick_part", "move_to_resource", "place_release"]
                    assert sorted(events[3:]) == ["machine_part", "move_to_resource"]
                    assert 4 <= len(checks) <= 5
                if all_capabilities == "bypass":
                    assert not checks
                    assert runtime.context.selected_parts == [SQUARE, "gear_small"]
                    assert all(product.part_tracker[part]["state"] == "assembled"
                               for part in runtime.context.selected_parts)
                    starts = {row["task_id"]: row["timestamp"] for row in runtime.context.negotiations
                              if row["kind"] == "execution_started" and
                              (row["resource_id"], row["event_name"]) in
                              {("KMR", "pick_part"), ("ur5e-4", "pick_approach")}}
                    ends = {row["task_id"]: row["timestamp"] for row in runtime.context.negotiations
                            if row["kind"] == "execution_completed" and row["task_id"] in starts}
                    assert len(starts) == len(ends) == 2
                    assert max(starts.values()) < min(ends.values())
                    assert any(row["kind"] == "CCA_bypassed" for row in runtime.context.negotiations)
                    assert not any(row["kind"] in {"CCA", "resource_safety_requested"}
                                   for row in runtime.context.negotiations)
                    assert runtime._report_snapshot()[1]["diagnostic_cca_bypass"] is True
                else:
                    assert checks and all(check["request_id"] for check in checks)
                    approved = {node["id"] for check in checks for node in check["plan"]["nodes"]}
                    assert {row["acknowledgement"]["task_id"]
                            for row in runtime.context.transitions} <= approved
                assert runtime.context.pending is None
                if not all_capabilities:
                    assert runtime.outcome["status"] == "execution_unavailable"
                assert all(
                    row["acknowledgement"]["evidence"] == "resource"
                    for row in runtime.context.transitions
                )
                for transition in runtime.context.transitions:
                    task_id = transition['acknowledgement']['task_id']
                    stages = ('task_sent', 'task_received', 'execution_started', 'execution_completed',
                              'ack_sent', 'ack_received', 'acknowledgement')
                    rows = {row['kind']: row['timestamp'] for row in runtime.context.negotiations
                            if row.get('task_id') == task_id and row['kind'] in stages
                            and row.get('status', 'completed') == 'completed'}
                    timestamps = [rows[stage] for stage in stages]
                    assert timestamps == sorted(timestamps)
                assert runtime.message_loop_timing
            finally:
                closed.set()
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


def test_ur5e1_unavailable_after_discovery_rejects_offer_and_renegotiates(inputs):
    async def scenario():
        async with network(inputs) as (runtime, driver):
            context = runtime.context
            for actor in context.resources.values():
                for name in actor.model['local_event_alphabet']:
                    actor.bind_executor(name, AsyncMock(return_value={}), lambda _task, _evidence: True,
                                        validate_start=lambda _task, _state, _geometry: True)
            first = await explore(runtime, driver)
            for task in first['tasks']:
                pending = context.prepare(task, simulated=True)
                context.acknowledge({**pending, 'status': 'completed'})
            discovered = await explore(runtime, driver)
            task = discovered['tasks'][0]
            assert task['resource_id'] == 'ur5e-1'
            context.resources['ur5e-1'].executors.clear()
            with pytest.raises(ValueError, match='Stale capability offer'):
                context.prepare(task)
            renewed = await explore(runtime, driver, timeout=10, candidate=task,
                                    part=SQUARE, desired=context.requirements[SQUARE][-1])
            assert not renewed.get('executable')
            assert renewed.get('execution_unavailable') or renewed['status'] != 'planned'
            assert not context.pending_tasks
            assert len({row['request']['request_id'] for row in context.explorations}) >= 3
    asyncio.run(scenario())


def test_revision_cache_detects_nested_configuration_and_availability_changes(inputs, monkeypatch):
    from cais_spade_llm.product import environment

    context = EnvironmentProductContext(**inputs)
    original = environment.fingerprint
    hashed = Mock(side_effect=original)
    monkeypatch.setattr(environment, 'fingerprint', hashed)
    before = context.revisions()
    assert hashed.call_count == len(context.resources)
    assert context.revisions() == before
    assert hashed.call_count == len(context.resources)
    context.resources['M1'].model['current_configuration']['parameters']['speed'] = 7
    changed = context.revisions()
    assert changed['M1'] != before['M1']
    assert hashed.call_count == len(context.resources) + 1
    actor = context.resources['ur5e-1']
    event = actor.model['local_event_alphabet'][0]
    actor.bind_executor(event, AsyncMock(), lambda _t, _e: True,
                        validate_start=lambda _t, _s, _g: True)
    available = context.revisions()
    actor.executors.clear()
    assert context.revisions()['ur5e-1'] != available['ur5e-1']


def test_discovery_wakes_when_acknowledgement_invalidates_in_flight_requests(inputs, monkeypatch):
    from cais_spade_llm.agents.shared_information import environment_capabilities

    async def scenario():
        async with network(inputs) as (runtime, driver):
            sent = asyncio.Event()

            async def hold_request(_behaviour, _message):
                sent.set()

            monkeypatch.setattr(environment_capabilities, 'send_agent_message', hold_request)
            pending = asyncio.create_task(explore(runtime, driver, timeout=60))
            await sent.wait()
            runtime.context.resources["Storage"].revision += 1
            result = await asyncio.wait_for(pending, 2)
            assert result['status'] == 'stale'
            assert runtime.request_queues == {}
    asyncio.run(scenario())


@pytest.mark.parametrize("change", [
    {}, {"fresh_stable": False}, {"observed_positions": [0.4, 0.0]},
    {"observed_positions": [float("nan"), 0.0]}, {"missing_joints": ["joint_2"]},
    {"held_part": SQUARE}, {"observed_positions": None},
])
def test_home_ack_requires_observed_empty_robot_at_configured_joints(change):
    from cais_spade_llm.recovery_framework.workflow_execution import _validate_robot_completion

    task = {"resource_id": "ur5e-3", "event_name": "move_home", "task_id": "home-3"}
    evidence = {**task, "controller_result": {"status": "completed"}}
    assert not _validate_robot_completion(task, evidence)
    evidence["home_observation"] = {
        "pose_name": "home", "joint_names": ["joint_1", "joint_2"],
        "target_positions": [0.0, 0.0], "observed_positions": [0.001, -0.001],
        "fresh_stable": True, "held_part": None, "missing_joints": [], **change,
    }
    assert _validate_robot_completion(task, evidence) is (not change)


@pytest.mark.parametrize("change", ["valid", "old", "other_robot", "other_target", "other_joints", "held"])
def test_home_ack_preserves_this_functions_observed_endpoint_after_feedback_expires(change, monkeypatch):
    import threading
    from cais_spade_llm.recovery_framework import workflow_execution
    from cais_spade_llm.resources.robot.gazebo_pick_place_controller import GazeboPickPlaceController

    controller = GazeboPickPlaceController.__new__(GazeboPickPlaceController)
    controller.execution_mode = "simulation"
    controller._simulation_goal = None
    controller.arm_joint_names = ["joint_1", "joint_2"]
    controller.named_positions = {"home": [0.2, -0.4]}
    controller.wait_for_services = Mock(return_value=True)
    controller._joint_lock = threading.Lock()
    controller._joint_positions = {"joint_1": 0.201, "joint_2": -0.401}
    controller._joint_received_times = {name: time.monotonic() - 0.1 for name in controller.arm_joint_names}
    controller._joint_stable_since = dict(controller._joint_received_times)
    started = time.time()
    raw = controller.move_to_named_pose("home")
    assert raw["success"] and raw["command_sent"] is False
    record = {
        "resource_id": "ur5e-3", "function_name": "move_home", "primitive": "move_to_named_pose",
        "parameters": {"pose_name": "home"}, "status": "completed",
        "started_at_unix": started, "completed_at_unix": time.time(),
        "command_evidence": deepcopy(controller._last_command_evidence),
    }
    snapshot = record["command_evidence"]["joint_observation"]
    assert snapshot["observed_positions"] == [0.201, -0.401]
    if change == "old":
        snapshot["observed_at_unix"] = started - 1
    elif change == "other_robot":
        record["resource_id"] = "ur5e-4"
    elif change == "other_target":
        snapshot["target_positions"][0] += 0.1
    elif change == "other_joints":
        snapshot["joint_names"][0] = "other_joint"
    controller._joint_positions.clear()
    controller._last_joint_target_observation = None
    controller._get_arm_joint_positions = Mock(return_value=(None, controller.arm_joint_names))
    controller._fresh_stable_joint_target = Mock(return_value=False)
    monkeypatch.setattr(workflow_execution, "time", SimpleNamespace(
        monotonic=Mock(side_effect=[0., 3.]), time=time.time,
    ))
    agent = SimpleNamespace(agent_name="ur5e-3", _controller=controller,
                            named_positions=controller.named_positions, _held_part=SQUARE if change == "held" else None)
    result = {"status": "completed", "observations": {"simulation_execution": {
        "resource_id": "ur5e-3", "function_name": "move_home", "primitive_results": [record],
    }}}
    observed = workflow_execution._observe_robot_home(agent, result)
    task = {"resource_id": "ur5e-3", "event_name": "move_home", "task_id": "home-3"}
    assert workflow_execution._validate_robot_completion(
        task, {**task, "controller_result": result, "home_observation": observed},
    ) is (change == "valid")
    if change in {"valid", "held"}:
        controller._get_arm_joint_positions.assert_not_called()
    else:
        controller._get_arm_joint_positions.assert_called_once()


@pytest.mark.parametrize("failure", [None, "services unavailable", "cancelled"])
def test_environment_prepares_selected_kmr_worker_once_before_dispatch(inputs, monkeypatch, failure):
    from cais_spade_llm.recovery_framework import workflow_execution
    from cais_spade_llm.recovery_framework.kmr_agent import KMRResourceAgent

    inputs["product_order"] = read_json(
        ROOT / "cais_spade_llm/specification/products/orders/assembly_board-v1-round-4mm-m1.json")
    inputs["scene"] = read_json(SCENE_PATH)
    context = EnvironmentProductContext(**inputs)
    probe = {"status": "completed", "launch_id": "launch", "scene_fingerprint": "scene"}
    worker = SimpleNamespace(run=AsyncMock(return_value=probe))
    if failure == "cancelled":
        worker.run.side_effect = asyncio.CancelledError()
    elif failure:
        worker.run.side_effect = workflow_execution.GazeboExecutionError({"status": "failed", "error": failure})
    monkeypatch.setattr(workflow_execution, "GazeboWorker", Mock())

    async def scenario():
        resources = [KMRResourceAgent("kmr@localhost", "none", worker=worker)]
        resources.extend(SimpleNamespace(agent_name=rid) for rid in context.resources if rid != "KMR")
        runtime = SimpleNamespace(context=context)
        workflow_execution.bind_environment_executors(runtime, resources)
        if failure:
            expected = asyncio.CancelledError if failure == "cancelled" else ValueError
            with pytest.raises(expected, match=None if failure == "cancelled" else "KMR readiness failed: services unavailable"):
                await runtime.prepare_execution()
            assert runtime.kmr_probe is None
            return
        await runtime.prepare_execution()
        await runtime.prepare_execution()
        worker.run.assert_awaited_once()
        request = worker.run.call_args.args[0]
        assert request["mode"] == "probe"
        assert request["pending"]["parameters"] == {"part_name": "RGOCG4-50_Round_4mm", "target_resource": "M1"}
        assert runtime.kmr_probe == probe
        worker.run.reset_mock()
        await workflow_execution._kmr_executor(runtime, resources[0])({"task_id": "pick", "event_name": "pick_part"})
        assert worker.run.call_args.args[0]["probe"] == probe

    asyncio.run(scenario())


def test_home_requirement_does_not_interrupt_pick_context(inputs):
    from cais_spade_llm.recovery_framework.environment_runtime import EnvironmentProductLoop

    async def scenario():
        async with network(inputs) as (runtime, driver):
            actor = runtime.context.resources["ur5e-3"]
            for state, held in [("at_pick", None), ("picked", SQUARE), ("positioned", SQUARE)]:
                actor.valuation.update(resource_state=state, held_part=held)
                assert "resource:ur5e-3" not in {
                    key for key, *_ in EnvironmentProductLoop()._negotiation_goals(runtime)
                }
    asyncio.run(scenario())


def test_home_offer_rejects_changed_availability_and_ignores_part_size(inputs):
    from cais_spade_llm.recovery_framework.environment_runtime import EnvironmentProductLoop

    async def scenario():
        async with network(inputs) as (runtime, driver):
            context = runtime.context
            actor = context.resources["ur5e-3"]
            actor.bind_executor("move_home", AsyncMock(return_value={}), lambda _t, _e: True,
                                validate_start=AsyncMock(return_value=True))
            context.geometry[SQUARE]["dimensions_m"] = [100.0, 100.0, 100.0]
            goal = next(goal for key, _, _, goal in EnvironmentProductLoop()._negotiation_goals(runtime)
                        if key == "resource:ur5e-3")
            result = await explore(runtime, driver, part=SQUARE,
                                   desired=context.requirements[SQUARE][-1], resource_goal=goal)
            assert result["executable"]
            task = result["tasks"][0]
            assert task["event_name"] == "move_home"
            actor.executors.clear()
            with pytest.raises(ValueError, match="Stale capability offer"):
                context.prepare(task)
            result = await explore(runtime, driver, part=SQUARE,
                                   desired=context.requirements[SQUARE][-1], resource_goal=goal)
            assert not result["executable"]
            assert not context.pending_tasks
            assert actor.valuation["resource_location"] is None
    asyncio.run(scenario())


@pytest.mark.parametrize("home", [None, {}, {"home_pose_observed": True},
                                  {"home_pose_observed": True, "downward_facing": False, "gripper_open": True}])
def test_empty_KMR_return_requires_observed_downward_home(home):
    from cais_spade_llm.recovery_framework.workflow_execution import _validate_kmr_completion

    task = {"event_name": "move_to_resource", "task_id": "return-home",
            "parameters": {"target_resource": "Storage"}}
    evidence = {"status": "completed", "task_id": "return-home", "observations": {
        "controllers_succeeded": True, "attached": False, "part_name": None,
        "home_observation": home,
    }}
    assert not _validate_kmr_completion(task, evidence)
    evidence["observations"]["home_observation"] = {
        "home_pose_observed": True, "downward_facing": True, "gripper_open": True,
    }
    assert _validate_kmr_completion(task, evidence)


def test_discovery_survives_unrelated_acknowledgements_and_rejects_dependency_changes(inputs):
    context = EnvironmentProductContext(**inputs)
    request = context.request(time.time() + 60)
    context.revision += 1
    context.resources["ur5e-4"].revision += 1
    assert context.current_request(request)
    context.request_dependencies(request, ["M1"])
    context.resources["M1"].revision += 1
    assert not context.current_request(request)


def test_eight_peg_order_preserves_machine_programs_and_other_orders(inputs):
    order_path = ROOT / "cais_spade_llm/specification/products/orders"
    inputs["product_order"] = read_json(order_path / "assembly_board-v1-eight-pegs.json")
    context = EnvironmentProductContext(**inputs)
    assert set(context.selected_parts) == set(inputs["scene"]["Storage"]["slots"])
    assert len(context.selected_parts) == 8
    assert context.machine_resource is None
    assert context.models["M1"]["current_configuration"]["program"]["effects"] == [{"process": "trim", "result": "square"}]
    assert context.models["M2"]["current_configuration"]["program"]["effects"] == [{"process": "trim", "result": "circle"}]
    assert read_json(order_path / "assembly_board-v1-recovery-framework.json")["parts"] == "all"
    assert read_json(order_path / "assembly_board-v1-round-4mm-m1.json")["machine_resource"] == "M1"


def test_intake_uses_local_machine_offers_and_observes_availability(inputs):
    inputs["product_order"] = read_json(
        ROOT / "cais_spade_llm/specification/products/orders/assembly_board-v1-eight-pegs.json"
    )

    async def scenario():
        async with network(inputs) as (runtime, driver):
            context = runtime.context
            for rid in context.machine_ids:
                context.resources[rid].bind_executor(
                    "machine_part", AsyncMock(), lambda task, evidence: True,
                    validate_start=AsyncMock(return_value=True),
                )
            result = await match_intake(runtime, driver, context.selected_parts)
            assert result["status"] == "matched"
            assert len(result["offers"]) == 8
            assert all(offer["available"] and offer["executable"] for offer in result["offers"])
            assert {offer["resource_id"] for offer in result["offers"]} == {"M1", "M2"}
            replies = [row for row in context.negotiations
                       if row["kind"] == "capability_reply" and row.get("scope") == "intake"]
            assert {row["resource_id"] for row in replies} == {"M1", "M2"}
            assert sum(len(row["offers"]) for row in replies) == 8
            assert not any(model.get("selected_path") for model in context.exploration_models.values())
            context.resources["M1"].executors.clear()
            context.resources["M2"].valuation["part_name"] = "RGOCG4-50_Round_4mm"
            updated = await match_intake(runtime, driver, context.selected_parts)
            assert updated["request_id"] != result["request_id"]
            assert all(not offer["executable"] for offer in updated["offers"]
                       if offer["resource_id"] == "M1")
            assert all(not offer["available"] for offer in updated["offers"]
                       if offer["resource_id"] == "M2")
            assert not runtime.request_queues

    asyncio.run(scenario())


@pytest.mark.parametrize('lost_replies', [1, 3])
def test_intake_timeout_retries_without_resource_revision_change(inputs, monkeypatch, lost_replies):
    from cais_spade_llm.recovery_framework import environment_runtime

    async def scenario():
        async with network(inputs, diagnostic_cca_bypass=True) as (runtime, driver):
            context = runtime.context
            for actor in context.resources.values():
                for name in actor.model['local_event_alphabet']:
                    actor.bind_executor(name, AsyncMock(), lambda *_: True,
                                        validate_start=AsyncMock(return_value=True))
            loop = environment_runtime.EnvironmentProductLoop()
            driver.agent.add_behaviour(loop)
            loop._negotiation_goals = lambda _: [(SQUARE, SQUARE, context.requirements[SQUARE][0], None)]
            runtime.prepare_execution = AsyncMock()
            runtime.queue_save = Mock()
            calls, sent = [], []
            revisions = context.revisions()

            async def matching(runtime, behaviour, parts):
                calls.append(context.revisions())
                if len(calls) <= lost_replies:
                    return {'status': 'timeout', 'offers': [],
                            'dependencies': {rid: revisions[rid] for rid in context.machine_ids}}
                return await match_intake(runtime, behaviour, parts)

            async def send(_behaviour, packet):
                assert packet.metadata['type'] == 'task'
                sent.append(json.loads(packet.body))
                runtime.stopped = True

            monkeypatch.setattr(environment_runtime, 'match_intake', matching)
            monkeypatch.setattr(environment_runtime, 'send_agent_message', send)
            await asyncio.wait_for(loop.work(runtime), 15)
            assert all(current == revisions for current in calls)
            assert len(calls) == (2 if lost_replies == 1 else 3)
            if lost_replies == 1:
                assert [(task['resource_id'], task['event_name']) for task in sent] == [('KMR', 'pick_part')]
            else:
                assert not sent and runtime.outcome['status'] == 'blocked'
                assert runtime.outcome['details']['intake']['status'] == 'timeout'
                assert not context.pending_tasks and not context.reservations

    asyncio.run(scenario())


def test_intake_timeout_is_reported_and_late_reply_window_closed(inputs, monkeypatch):
    from cais_spade_llm.agents.shared_information import environment_capabilities

    async def scenario():
        async with network(inputs) as (runtime, driver):
            request = runtime.context.request

            def short_request(_deadline, **kwargs):
                return request(time.time() + .02, **kwargs)

            monkeypatch.setattr(runtime.context, 'request', short_request)
            monkeypatch.setattr(environment_capabilities, 'send_agent_message', AsyncMock())
            result = await match_intake(runtime, driver, [SQUARE])
            assert result['status'] == 'timeout' and not result['offers']
            model = runtime.context.exploration_models[result['request_id']]
            assert model['closed'] and model['status'] == 'timeout'
            assert not runtime.request_queues
            assert runtime.context.negotiations[-1]['result'] == result
            assert not runtime.context.current_request({'run_id': runtime.context.run_id,
                                                        'request_id': result['request_id']})

    asyncio.run(scenario())



@pytest.mark.parametrize("home_completed", [False, True])
def test_retained_assembly_path_reuses_home_ack_from_another_part(inputs, home_completed):
    from cais_spade_llm.recovery_framework.environment_runtime import EnvironmentProductLoop

    part = "RGOCG4-50_Round_4mm"
    inputs["product_order"]["parts"] = [SQUARE, part]

    async def scenario():
        async with network(inputs) as (runtime, driver):
            context = runtime.context
            context.resources["Storage"].valuation[f"inventory.{part}"] = False
            context.resources[BUFFER].valuation["zone_4_part"] = part
            context.part_tracker[part].update(
                location=BUFFER, state="ready",
                processCompleted=[{"process": "trim", "result": "circle"}],
            )
            context.resources["Storage"].valuation[f"inventory.{SQUARE}"] = False
            context.part_tracker[SQUARE].update(
                location=context.product_name, state="assembled",
                processCompleted=[effect for step in context.requirements[SQUARE]
                                  for effect in step["processesToComplete"]],
            )
            robot = context.resources["ur5e-3"]
            robot.valuation.update(
                resource_state="placed", resource_location=context.product_name,
                part_state="assembled", part_location=context.product_name,
            )
            for name in robot.model["local_event_alphabet"]:
                robot.bind_executor(name, AsyncMock(), lambda _task, _evidence: True,
                                    validate_start=AsyncMock(return_value=True))
            loop = EnvironmentProductLoop()
            loop.set_agent(driver.agent)
            runtime.retained_paths = {}
            desired = context.requirements[part][-1]
            result = await loop._negotiate_goal(runtime, part, part, desired, None)
            assert result["status"] == "planned", result
            assert result["tasks"][0]["event_name"] == "move_home"
            home = deepcopy(result["tasks"][0])
            home["part_name"] = SQUARE
            home.pop("offer_product_state", None)
            pending = context.prepare(home, simulated=True)
            if home_completed:
                context.acknowledge({**pending, "status": "completed"})
            else:
                with pytest.raises(ValueError, match="Acknowledgement"):
                    context.acknowledge({**pending, "status": "failed"})
                context.cancel_pending(pending["task_id"])
            transition_count = len(context.transitions)
            result = await loop._negotiate_goal(runtime, part, part, desired, None)
            assert result["status"] == "planned", result
            assert result["tasks"][0]["event_name"] == (
                "pick_approach" if home_completed else "move_home"
            )
            assert len(context.transitions) == transition_count
            assert context.part_tracker[part]["location"] == BUFFER
            assert bool([row for row in context.negotiations
                         if row["kind"] == "retained_step_satisfied"]) == home_completed
            if home_completed:
                prepared = context.prepare(result["tasks"][0], simulated=True)
                assert prepared["parameters"]["part_name"] == part
                assert prepared["parameters"]["origin_resource_location"] == BUFFER

    asyncio.run(scenario())


def test_home_ack_accepts_equivalent_observed_revolute_angles():
    from cais_spade_llm.recovery_framework.workflow_execution import _validate_robot_completion
    import math

    task = {"resource_id": "ur5e-1", "event_name": "move_home", "task_id": "home-equivalent"}
    evidence = {**task, "controller_result": {"status": "completed"}, "home_observation": {
        "pose_name": "home", "joint_names": ["shoulder"],
        "target_positions": [-1.596], "observed_positions": [-1.596 + 2 * math.pi],
        "fresh_stable": True, "held_part": None, "missing_joints": [],
    }}
    assert _validate_robot_completion(task, evidence)
    evidence["home_observation"]["observed_positions"][0] += 0.1
    assert not _validate_robot_completion(task, evidence)


def test_printer_admission_revalidates_available_handler_and_keeps_pick_context(inputs):
    from cais_spade_llm.recovery_framework.environment_runtime import EnvironmentProductLoop

    inputs["product_order"]["parts"] = ["gear_small", "gear_medium", "gear_large"]

    async def scenario():
        async with network(inputs) as (runtime, driver):
            context = runtime.context
            loop = EnvironmentProductLoop()
            for actor in context.resources.values():
                for name in actor.model["local_event_alphabet"]:
                    actor.bind_executor(name, AsyncMock(), lambda _task, _evidence: True,
                                        validate_start=AsyncMock(return_value=True))
            goals = loop._negotiation_goals(runtime)
            assert loop._source_waiting(context, goals, set()) == {
                "3D Printing Station": inputs["product_order"]["parts"]}
            assert loop._intake_ready(context, "gear_small")
            discovered = await explore(runtime, driver, part="gear_small",
                                       desired=context.requirements["gear_small"][-1])
            task = discovered["tasks"][0]
            assert task["resource_id"] == "ur5e-4" and task["event_name"] == "pick_approach"
            actor = context.resources["ur5e-4"]
            executors = dict(actor.executors)
            actor.executors.clear()
            assert not loop._intake_ready(context, "gear_small")
            with pytest.raises(ValueError, match="Stale capability offer"):
                context.prepare(task)
            renewed = await explore(runtime, driver, part="gear_small",
                                    desired=context.requirements["gear_small"][-1], candidate=task)
            assert not renewed.get("executable")
            actor.executors.update(executors)
            discovered = await explore(runtime, driver, part="gear_small",
                                       desired=context.requirements["gear_small"][-1])
            pending = context.prepare(discovered["tasks"][0], simulated=True)
            assert not loop._intake_ready(context, "gear_medium")
            context.acknowledge({**pending, "status": "completed"})
            assert not loop._intake_ready(context, "gear_medium")
            assert loop._source_waiting(context, loop._negotiation_goals(runtime), {"gear_small"}) == {
                "3D Printing Station": ["gear_medium", "gear_large"]}
    asyncio.run(scenario())


def test_pa_commits_running_ack_while_another_CCA_approval_is_pending(inputs, monkeypatch):
    from cais_spade_llm.recovery_framework import environment_runtime

    gear = 'gear_small'
    inputs['product_order']['parts'] = [SQUARE, gear]
    async def scenario():
        async with network(inputs) as (runtime, driver):
            context = runtime.context
            first = await explore(runtime, driver, part=SQUARE, desired=context.requirements[SQUARE][0])
            active = context.prepare(first['tasks'][0], simulated=True)
            for name in context.resources['ur5e-4'].model['local_event_alphabet']:
                context.resources['ur5e-4'].bind_executor(
                    name, AsyncMock(return_value={}), lambda *_: True,
                    validate_start=AsyncMock(return_value=True))
            loop = environment_runtime.EnvironmentProductLoop()
            driver.agent.add_behaviour(loop)
            loop._negotiation_goals = lambda _: [(gear, gear, context.requirements[gear][1], None)]
            queue = asyncio.Queue()
            loop.receive = lambda **_: asyncio.wait_for(queue.get(), .5)
            runtime.prepare_execution = AsyncMock()
            runtime.save = Mock()
            approvals = []
            async def decide(request_id):
                packet = message(runtime.product_jid, 'ack', {'task_id': active['task_id'],
                    'status': 'completed', 'acknowledgement': {**active, 'status': 'completed'}})
                packet.sender = runtime.jids['KMR']
                await queue.put(packet)
                deadline = time.monotonic() + 5
                while active['task_id'] not in context.acknowledgements:
                    assert time.monotonic() < deadline, 'PA deferred an unrelated acknowledgement while waiting for CCA'
                    await asyncio.sleep(.01)
                packet = message(runtime.product_jid, 'plan_safety_result', {'ok': True, 'request_id': request_id})
                packet.sender = driver.agent.cca_jid
                await queue.put(packet)
            async def send(_behaviour, packet):
                if packet.metadata['type'] == 'plan_safety_check':
                    approvals.append(asyncio.create_task(decide(json.loads(packet.body)['request_id'])))
                elif packet.metadata['type'] == 'task':
                    assert active['task_id'] in context.acknowledgements
                    runtime.stopped = True
            monkeypatch.setattr(environment_runtime, 'send_agent_message', send)
            async def receive(**_kwargs):
                try:
                    return await asyncio.wait_for(queue.get(), .05)
                except asyncio.TimeoutError:
                    return None
            loop.receive = receive
            try:
                await asyncio.wait_for(loop.work(runtime), 15)
                await asyncio.gather(*approvals)
                assert approvals and active['task_id'] in context.acknowledgements
                assert runtime.stopped
            finally:
                for approval in approvals:
                    approval.cancel()
                await asyncio.gather(*approvals, return_exceptions=True)
    asyncio.run(scenario())


@pytest.mark.parametrize('ending', ['timeout', 'rejected', 'stop', 'cancelled'])
def test_pending_CCA_approval_releases_reservations_without_dispatch(inputs, monkeypatch, ending):
    from cais_spade_llm.recovery_framework import environment_runtime

    inputs['product_order']['parts'] = ['gear_small']

    async def scenario():
        async with network(inputs) as (runtime, driver):
            for name in runtime.context.resources['ur5e-4'].model['local_event_alphabet']:
                runtime.context.resources['ur5e-4'].bind_executor(
                    name, AsyncMock(return_value={}), lambda *_: True,
                    validate_start=AsyncMock(return_value=True))
            loop = environment_runtime.EnvironmentProductLoop()
            driver.agent.add_behaviour(loop)
            runtime.prepare_execution = AsyncMock()
            runtime.cancel_owned = AsyncMock()
            runtime.save = Mock()
            queue = asyncio.Queue()
            requested = []
            offset = 0.
            monkeypatch.setattr(environment_runtime, 'time', SimpleNamespace(
                time=time.time, monotonic=lambda: time.monotonic() + offset))

            async def send(_behaviour, packet):
                nonlocal offset
                assert packet.metadata['type'] != 'task', 'Unapproved task dispatched'
                if packet.metadata['type'] != 'plan_safety_check':
                    return
                request = json.loads(packet.body)['request_id']
                requested.append(request)
                assert runtime.context.pending_tasks and runtime.context.reservations
                if ending == 'timeout':
                    offset = 61.
                elif ending == 'stop':
                    runtime.stop()
                elif ending == 'cancelled':
                    asyncio.current_task().cancel()
                else:
                    reply = message(runtime.product_jid, 'plan_safety_result',
                                    {'ok': False, 'request_id': request})
                    reply.sender = driver.agent.cca_jid
                    await queue.put(reply)

            async def receive(**_kwargs):
                try:
                    return await asyncio.wait_for(queue.get(), .01)
                except asyncio.TimeoutError:
                    return None

            loop.receive = receive
            monkeypatch.setattr(environment_runtime, 'send_agent_message', send)
            if ending == 'timeout':
                with pytest.raises(TimeoutError, match='plan_safety_result'):
                    await asyncio.wait_for(loop.work(runtime), 10)
            elif ending == 'cancelled':
                with pytest.raises(asyncio.CancelledError):
                    await asyncio.wait_for(loop.work(runtime), 10)
            else:
                await asyncio.wait_for(loop.work(runtime), 10)
            assert requested and not runtime.context.pending_tasks and not runtime.context.reservations
            assert loop._awaiting_plan_request is None and loop._plan_decision is None
            # Old and uncorrelated decisions cannot authorize new work after cleanup.
            runtime.stopped = False
            for request_id in [requested[0], None]:
                reply = message(runtime.product_jid, 'plan_safety_result', {'ok': True, 'request_id': request_id})
                reply.sender = driver.agent.cca_jid
                await queue.put(reply)
                assert await loop._collect_acknowledgement(runtime)
                assert loop._plan_decision is None

    asyncio.run(scenario())


@pytest.mark.parametrize('slow_stage', ['materialization', 'persistence'])
def test_report_write_does_not_delay_ack_and_Stop_report_supersedes_queued_work(inputs, monkeypatch, slow_stage):
    import threading
    from cais_spade_llm.recovery_framework import environment_runtime

    async def scenario():
        async with network(inputs) as (runtime, driver):
            first = await explore(runtime, driver, part=SQUARE, desired=runtime.context.requirements[SQUARE][0])
            active = runtime.context.prepare(first['tasks'][0], simulated=True)
            entered, release = threading.Event(), threading.Event()
            original = runtime.reports.save
            calls = []

            def write(report, processes):
                if slow_stage == 'persistence' and not calls:
                    entered.set()
                    assert release.wait(5), 'Test did not release its slow report writer'
                calls.append(report['outcome']['status'])
                return original(report, processes)

            monkeypatch.setattr(runtime.reports, 'save', write)
            materialize = runtime._materialize_report
            def copy_report(report):
                if not calls:
                    entered.set()
                    assert release.wait(5), 'Test did not release its slow report construction'
                return materialize(report)
            if slow_stage == 'materialization':
                monkeypatch.setattr(runtime, '_materialize_report', copy_report)
            loop = environment_runtime.EnvironmentProductLoop()
            driver.agent.add_behaviour(loop)
            packet = message(runtime.product_jid, 'ack', {'task_id': active['task_id'],
                'status': 'completed', 'acknowledgement': {**active, 'status': 'completed'}})
            packet.sender = runtime.jids['KMR']
            loop._deferred_acks = [packet]
            runtime.retained_paths = {}
            runtime.queue_save()
            try:
                assert await asyncio.to_thread(entered.wait, 2)
                started = time.monotonic()
                assert await asyncio.wait_for(loop._collect_acknowledgement(runtime), .5)
                assert time.monotonic() - started < .5
                assert active['task_id'] in runtime.context.acknowledgements
                assert not release.is_set()
                started = time.monotonic()
                runtime.stop()
                assert time.monotonic() - started < .5
                runtime.queue_save()
            finally:
                release.set()
                await runtime.flush_reports()
            report = json.loads((runtime.path / 'run.json').read_text())
            assert report['outcome']['status'] == 'stopped'
            assert len(report['transitions']) == 1 and not report['pending_tasks']
            assert calls == ['prepared', 'stopped']
            verify_environment_run(report)
            older = runtime._report_snapshot()
            runtime.outcome = {'status': 'completed'}
            runtime.save()
            runtime._persist_report(older)
            assert json.loads((runtime.path / 'run.json').read_text())['outcome']['status'] == 'completed'

    asyncio.run(scenario())


def test_slow_capability_inbox_does_not_delay_unrelated_ack_or_Stop(inputs, monkeypatch):
    import threading
    from cais_spade_llm.agents.shared_information import environment_capabilities
    from cais_spade_llm.recovery_framework.environment_runtime import EnvironmentProductLoop

    inputs['product_order']['parts'] = [SQUARE, 'gear_small']
    async def scenario():
        async with network(inputs) as (runtime, driver):
            context = runtime.context
            first = await explore(runtime, driver, part=SQUARE, desired=context.requirements[SQUARE][0])
            active = context.prepare(first['tasks'][0], simulated=True)
            entered, release = threading.Event(), threading.Event()
            calculate = environment_capabilities._calculate_reply
            def slow(*args, **kwargs):
                entered.set()
                assert release.wait(5), 'Test did not release its slow capability calculation'
                return calculate(*args, **kwargs)
            monkeypatch.setattr(environment_capabilities, '_calculate_reply', slow)
            discovery = asyncio.create_task(explore(
                runtime, driver, part='gear_small', desired=context.requirements['gear_small'][1]))
            loop = EnvironmentProductLoop()
            driver.agent.add_behaviour(loop, Template(metadata={'type': 'ack'}))
            loop._deferred_acks = []
            runtime.retained_paths = {}
            try:
                assert await asyncio.to_thread(entered.wait, 2)
                packet = message(runtime.product_jid, 'ack', {'task_id': active['task_id'],
                    'status': 'completed', 'acknowledgement': {**active, 'status': 'completed'}})
                resource = next(a for a in runtime.resource_agents if a.agent_name == 'KMR')
                started = time.monotonic()
                await send_agent_message(SimpleNamespace(agent=resource), packet)
                assert await asyncio.wait_for(loop._collect_acknowledgement(runtime), .5)
                assert time.monotonic() - started < .5
                assert active['task_id'] in context.acknowledgements
                assert not discovery.done() and not release.is_set()
                started = time.monotonic()
                runtime.stop()
                assert time.monotonic() - started < .5
                assert not context.pending_tasks and not context.reservations
            finally:
                release.set()
                await asyncio.wait_for(discovery, 2)
            assert not runtime.request_queues
            assert not any(row.get('kind') == 'dispatch' for row in context.negotiations)

    asyncio.run(scenario())


@pytest.mark.parametrize('change', ['model', 'valuation', 'geometry', 'requirements', 'permitted', 'reservation', 'closed'])
def test_capability_worker_rechecks_changes_before_accepting_result(inputs, monkeypatch, change):
    import threading
    from cais_spade_llm.agents.shared_information import environment_capabilities

    async def scenario():
        async with network(inputs) as (runtime, driver):
            context = runtime.context
            request = context.request(time.time() + 5)
            request.update(scope='intake', intake_parts=[SQUARE])
            context.exploration_models[request['request_id']]['intake_parts'] = [SQUARE]
            actor = context.resources['M1']
            resource = next(a for a in runtime.resource_agents if a.agent_name == 'M1')
            inbox = next(b for b in resource.behaviours if isinstance(b, CapabilityRequestInbox))
            entered, release = threading.Event(), threading.Event()
            original = environment_capabilities._calculate_reply
            captured = []
            replies = []
            def slow(snapshot, *args, **kwargs):
                captured.append(snapshot)
                entered.set()
                assert release.wait(5)
                return original(snapshot, *args, **kwargs)
            async def send(_inbox, packet):
                replies.append(json.loads(packet.body))
            monkeypatch.setattr(environment_capabilities, '_calculate_reply', slow)
            monkeypatch.setattr(environment_capabilities, 'send_agent_message', send)
            pending = asyncio.create_task(inbox.handle(runtime, runtime.product_jid, request))
            try:
                assert await asyncio.to_thread(entered.wait, 2)
                if change == 'model':
                    actor.model['configuration_revision'] = 'changed'
                elif change == 'valuation':
                    actor.valuation['part_name'] = SQUARE
                elif change == 'geometry':
                    context.geometry[SQUARE]['mass_kg'] = 9000
                elif change == 'requirements':
                    context.requirements[SQUARE][0]['changed'] = True
                elif change == 'permitted':
                    context.permitted_resources.remove('M1')
                elif change == 'reservation':
                    context.reservations['access:M1'] = 'other-task'
                else:
                    context.exploration_models[request['request_id']]['closed'] = True
                assert captured[0].resources['M1'].valuation['part_name'] is None
                assert 'access:M1' not in captured[0].reservations
            finally:
                release.set()
                await asyncio.wait_for(pending, 2)
            if change == 'closed':
                assert not replies
            elif change == 'reservation':
                assert replies[0]['intake_offers']
                assert not any(offer['available'] for offer in replies[0]['intake_offers'])
            else:
                assert not replies[0].get('intake_offers')
                assert replies[0]['rejections']

    asyncio.run(scenario())


def test_capability_worker_limit_and_Stop_discard_queued_work(inputs):
    import threading

    async def scenario():
        async with network(inputs) as (runtime, driver):
            entered, release = threading.Event(), threading.Event()
            lock = threading.Lock()
            threads = set()
            loop_thread = threading.get_ident()
            def slow(*, stopped):
                with lock:
                    threads.add(threading.get_ident())
                    if len(threads) == 2:
                        entered.set()
                assert release.wait(5)
                return None if stopped() else 'result'
            tasks = [asyncio.create_task(runtime.calculate_capability(
                slow, request={'request_id': 'test', 'branch_id': str(index)}, resource_id='M1'))
                for index in range(6)]
            try:
                assert await asyncio.to_thread(entered.wait, 2)
                assert len(threads) == 2 and loop_thread not in threads
                runtime.stop()
            finally:
                release.set()
                results = await asyncio.gather(*tasks, return_exceptions=True)
            assert len(threads) == 2
            assert all(result is None or isinstance(result, asyncio.CancelledError) for result in results)

    asyncio.run(scenario())


@pytest.mark.parametrize("mode,identity,bypass", [
    ("physical", ("launch", 42), True),
    ("simulation", None, True),
    ("simulation", ("launch", 42), "true"),
])
def test_CCA_bypass_rejects_hardware_missing_launch_and_nonboolean_values(mode, identity, bypass):
    from cais_spade_llm.recovery_framework.environment_runtime import prepare_environment_start

    setup = {"execution_mode": mode}
    with pytest.raises(ValueError, match="explicitly prepared simulation"):
        prepare_environment_start(setup, prewarm_controllers=True,
                                  launch_identity=identity, diagnostic_cca_bypass=bypass)
    with pytest.raises(ValueError, match="explicitly prepared simulation"):
        EnvironmentRuntime(None, {"setup": setup, "launch_identity": identity,
                                  "diagnostic_cca_bypass": bypass}, [])


@pytest.mark.parametrize("blocker", ["Stop", "stale", "completion"])
def test_CCA_bypass_preserves_Stop_revisions_and_completion_evidence(inputs, monkeypatch, blocker):
    from cais_spade_llm.recovery_framework import environment_runtime as execution

    async def scenario():
        async with network(inputs, diagnostic_cca_bypass=True) as (runtime, driver):
            result = await explore(runtime, driver)
            task = result["tasks"][0]
            context = runtime.context
            resource = driver.agent.container.get_agent(runtime.jids[task["resource_id"]])
            actor = resource.environment_context
            controller = AsyncMock(return_value={"observed": False})
            actor.bind_executor(task["event_name"], controller, lambda _task, _evidence: False,
                                validate_start=AsyncMock(return_value=True))
            pending = context.prepare(task)
            before = context.snapshot()
            packets = []

            async def capture(_behaviour, packet, **_kwargs):
                assert packet.metadata["type"] == "ack"
                packets.append(json.loads(packet.body))

            monkeypatch.setattr(execution, "send_agent_message", capture)
            if blocker == "Stop":
                runtime.stop()
            elif blocker == "stale":
                actor.revision += 1
            incoming = message(str(resource.jid), "task", pending)
            incoming.sender = runtime.product_jid
            await execution.execute_environment_task(SimpleNamespace(agent=resource), incoming, pending)
            assert packets[-1]["status"] == "blocked"
            assert context.snapshot() == before and not context.transitions
            assert controller.await_count == (1 if blocker == "completion" else 0)
            if blocker == "Stop":
                assert not context.pending_tasks and not context.reservations

    asyncio.run(scenario())
