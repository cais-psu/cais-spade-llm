"""PA/RA planning, acknowledged custody, and configured operator views."""

from __future__ import annotations

import asyncio
import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from cais_spade_llm.product.nominal import NominalProductContext
from cais_spade_llm.product.nominal_planner import search_nominal_part
from cais_spade_llm.recovery_framework import PRODUCT_PATH, ROOT, SCENE_PATH, read_json

SQUARE = "KET4_Square_4mm"
CIRCULAR = "RGOCG4-50_Round_4mm"
BUFFER = "Buffer For Machined parts"


@pytest.fixture
def inputs():
    meta = next(iter(read_json(PRODUCT_PATH).values()))
    return {
        "scene": read_json(SCENE_PATH),
        "product_order": read_json(ROOT / meta["product_order_file"]),
        "geometry": read_json(ROOT / meta["product_geometry_file"])["gazebo"],
    }


@pytest.fixture
def context(inputs):
    return NominalProductContext(**inputs)


def task_for(context, resource, name, **params):
    params = {"resource_id": resource, **params}
    matches = []
    for event in context.models[resource]["events"]:
        if event["event_name"] != name or set(event["parameter_bindings"]) != set(params):
            continue
        if all(
            "equals" not in rule or rule["equals"] == params[field]
            for field, rule in event["parameter_bindings"].items()
        ):
            matches.append(event)
    if len(matches) != 1:
        raise ValueError("No exact nominal task binding")
    return {
        "resource_id": resource,
        "event_id": matches[0]["event_id"],
        "event_name": name,
        "parameters": params,
    }


def acknowledge(context, task):
    pending = context.prepare(task)
    ack = {**pending, "status": "completed", "evidence": "simulated"}
    context.acknowledge(ack)
    return ack


@pytest.fixture
def traces(context, monkeypatch):
    # Reuse established valid setup traces, executing each task through PA/RA handlers.
    import test_nominal_resource_des as traces

    def through_agents(models, state, resource, name, **params):
        assert state == context.snapshot()
        acknowledge(context, task_for(context, resource, name, **params))
        return context.snapshot()

    monkeypatch.setattr(traces, "event", through_agents)
    return traces


def test_full_order_state_history_and_replay(context):
    while context.revision < 1_000:
        plan = context.plan()
        if plan["status"] == "completed":
            break
        assert plan["status"] == "planned"
        assert plan["tasks"]
        for task in plan["tasks"]:
            acknowledge(context, task)
    assert context.plan()["status"] == "completed"
    assert context.snapshot()["Exit"]["part_name"] == "assembly_board-v1"
    assert len(context.models) == 12
    assert context.requirements[SQUARE] == {
        "ur5e-3": {f"assembled.{SQUARE}": True, "resource_state": "idle"}
    }
    assert not context.initial_product_states[SQUARE]["processCompleted"]
    assert context.initial_product_states["gear_small"]["processCompleted"] == ["print_part"]
    assert not any(record["task"]["event_name"] == "print_part" for record in context.transitions)
    for part, state in context.part_tracker.items():
        if part == "assembly_board-v1":
            assert state["location"] == "Exit" and state["state"] != "assembled"
        else:
            assert state["location"] == "assembly_board-v1"
            assert state["state"] == "assembled"
            assert "place_insert" in state["processCompleted"]
    machine_tasks = [
        record["task"]
        for record in context.transitions
        if record["task"]["event_name"] == "machine_part"
    ]
    assert all(
        task["parameters"]["part_name"].startswith("KET")
        for task in machine_tasks
        if task["resource_id"] == "M1"
    )
    assert all(
        task["parameters"]["part_name"].startswith("RGOCG")
        for task in machine_tasks
        if task["resource_id"] == "M2"
    )
    square_history = context.history[SQUARE]
    machining = next(
        index for index, row in enumerate(square_history) if row["event_name"] == "machine_part"
    )
    assert all(
        "machine_part" in row["state"]["processCompleted"] for row in square_history[machining:]
    )
    assert all(
        row["state"]["state"] != "assembled"
        for row in square_history
        if row["event_name"] == "place_release"
    )
    replay = NominalProductContext(**context.inputs)
    for record in context.transitions:
        task = {
            key: value for key, value in record["task"].items()
            if key not in {"task_id", "revision"}
        }
        assert replay.prepare(task) == record["task"]
        assert replay.acknowledge(record["acknowledgement"])
        assert replay.transitions[-1] == record
    assert replay.snapshot() == context.snapshot()
    assert replay.part_tracker == context.part_tracker
    assert replay.history == context.history


def test_subset_and_missing_printed_product(inputs):
    inputs["product_order"]["parts"] = ["gear_small"]
    inputs["scene"]["3D Printing Station"]["initial_products"].remove("gear_small")
    context = NominalProductContext(**inputs)
    assert context.part_tracker["gear_small"]["processCompleted"] == []
    for task in context.plan()["tasks"]:
        acknowledge(context, task)
    assert context.plan()["status"] == "completed"
    assert context.history["gear_small"][0]["event_name"] == "print_part"
    assert context.snapshot()["Exit"]["part_name"] is None
    assert context.part_tracker["gear_medium"]["state"] == "ready"


def test_bounded_inventory_rejects_quantity_and_unknown_parts(inputs):
    inputs["product_order"]["quantity"] = 2
    with pytest.raises(ValueError, match="quantity"):
        NominalProductContext(**inputs)
    inputs["product_order"].update(quantity=1, parts=["unknown"])
    with pytest.raises(ValueError, match="unknown"):
        NominalProductContext(**inputs)
    inputs["product_order"]["parts"] = [SQUARE, SQUARE]
    with pytest.raises(ValueError, match="repeat"):
        NominalProductContext(**inputs)


def test_capability_search_is_read_only_and_distinguishes_limits(context):
    before = context.snapshot()
    result = context.plan(max_search_states=1)
    assert result["status"] == "budget_exhausted"
    assert context.snapshot() == before and context.revision == 0
    assert all(not history for history in context.history.values())
    assert set(context.capability_requests[0]["resources"]) == set(context.models)
    unreachable = search_nominal_part(
        context.resources, before, "unconfigured", max_search_states=500
    )
    assert unreachable["status"] == "blocked"


def test_blocked_component_does_not_block_other_selected_goals(inputs):
    inputs["product_order"]["parts"] = ["gear_small", "gear_medium"]
    context = NominalProductContext(**inputs)
    original = context.resources["ur5e-4"].validate_nominal_event

    def unavailable(models, valuation, task):
        if task["parameters"].get("part_name") == "gear_small":
            raise ValueError("gear_small unavailable")
        return original(models, valuation, task)

    context.resources["ur5e-4"].validate_nominal_event = unavailable
    result = context.plan()
    assert result["part_name"] == "gear_medium"
    assert result["requests"][0]["status"] == "blocked"


def test_no_ack_no_commit_and_duplicates_are_idempotent(context):
    plan = context.plan()
    task = plan["tasks"][0]
    before, products = context.snapshot(), deepcopy(context.part_tracker)
    pending = context.prepare(task)
    assert context.snapshot() == before and context.part_tracker == products
    ack = {**pending, "status": "completed", "evidence": "simulated"}
    assert context.acknowledge(ack)
    committed = context.snapshot()
    assert not context.acknowledge(ack)
    assert context.revision == 1 and len(context.transitions) == 1
    assert context.snapshot() == committed
    with pytest.raises(ValueError, match="Conflicting"):
        context.acknowledge({**ack, "resource_id": "M1"})


def test_acknowledgement_preserves_exact_json_types(context):
    pending = context.prepare(context.plan()["tasks"][0])
    before = context.snapshot()
    with pytest.raises(ValueError, match="acknowledgement"):
        context.acknowledge(
            {**pending, "revision": 0.0, "status": "completed", "evidence": "simulated"}
        )
    with pytest.raises(ValueError, match="task_id"):
        context.acknowledge(
            {**pending, "task_id": [], "status": "completed", "evidence": "simulated"}
        )
    assert context.snapshot() == before and context.revision == 0


@pytest.mark.parametrize(
    "change",
    [
        {"task_id": "unknown"},
        {"resource_id": "M1"},
        {"revision": 100},
        {"status": "running"},
        {"status": "failed"},
        {"evidence": "live"},
        {"parameters": {}},
    ],
)
def test_bad_acknowledgements_are_atomic(context, change):
    pending = context.prepare(context.plan()["tasks"][0])
    before = context.snapshot()
    with pytest.raises(ValueError, match="acknowledgement"):
        context.acknowledge({**pending, "status": "completed", "evidence": "simulated", **change})
    assert context.revision == 0 and context.snapshot() == before
    assert all(not history for history in context.history.values())


def test_stale_pending_task_rechecks_revision_and_participant_guards(context):
    task = context.plan()["tasks"][0]
    pending = context.prepare(task)
    context.revision = 1
    with pytest.raises(ValueError, match="revision"):
        context.acknowledge({**pending, "status": "completed", "evidence": "simulated"})
    assert not context.transitions


@pytest.mark.parametrize("machine_order", [("M1", "M2"), ("M2", "M1")])
def test_multi_part_belt_and_shared_product_updates(context, traces, machine_order):
    parts = {"M1": SQUARE, "M2": CIRCULAR}
    state = context.snapshot()
    for machine in machine_order:
        state = traces.load(context.models, state, machine, parts[machine])
    state = traces.advance(
        context.models, state, {CIRCULAR: "output_nest", SQUARE: "after loading_position_1"}
    )
    movement = context.transitions[-1]
    assert set(movement["affected_parts"]) == {SQUARE, CIRCULAR}
    assert state["Conveyor"][f"part_order.{CIRCULAR}"] == 0
    before = deepcopy(state)
    with pytest.raises(ValueError):
        traces.advance(
            context.models, state, {SQUARE: "output_nest", CIRCULAR: "loading_position_1"}
        )
    assert context.snapshot() == before
    state = traces.advance(context.models, state, {SQUARE: "after loading_position_1"}, CIRCULAR)
    assert context.part_tracker[CIRCULAR]["location"] == BUFFER
    assert state[BUFFER]["zone_1_part"] == CIRCULAR
    assert state["Conveyor"][f"part_location.{CIRCULAR}"] is None


def test_loading_area_blocking_and_staging(context, traces):
    state = traces.load(context.models, context.snapshot(), "M1", SQUARE)
    state = traces.advance(context.models, state, {SQUARE: "loading_position_2"})
    state = traces.completed_machine_part(context.models, state, "M2", CIRCULAR)
    state = traces.pick(context.models, state, "ur5e-2", CIRCULAR, "M2")
    approach = task_for(
        context, "ur5e-2", "place_approach", part_name=CIRCULAR, destination_location="Conveyor"
    )
    with pytest.raises(ValueError, match="guard"):
        context.resources["ur5e-2"].validate_nominal_event(context.models, state, approach)
    assert context.snapshot() == state
    state = traces.release(context.models, state, "ur5e-2", CIRCULAR, "M2 staging tray")
    assert state["M2"]["staging_part"] == CIRCULAR
    assert context.part_tracker[CIRCULAR]["location"] == "M2 staging tray"


def test_full_buffer_backpressure_and_release(context, traces):
    state = context.snapshot()
    parts = context.models["M1"]["assignments"]["nominal_parts"]
    for index, part in enumerate(parts):
        state = traces.load(context.models, state, "M1", part)
        state = traces.advance(context.models, state, {part: "output_nest"})
        state = traces.advance(context.models, state, {}, part)
        for zone in range(1, 4 - index):
            state = traces.buffer_advance(context.models, state, part, zone)
    before = deepcopy(state)
    # All four zones are occupied; a new loading approach must fail without updates.
    state = traces.completed_machine_part(context.models, state, "M2", CIRCULAR)
    state = traces.pick(context.models, state, "ur5e-2", CIRCULAR, "M2")
    with pytest.raises(ValueError, match="guard"):
        acknowledge(
            context,
            task_for(
                context,
                "ur5e-2",
                "place_approach",
                part_name=CIRCULAR,
                destination_location="Conveyor",
            ),
        )
    assert context.snapshot() == state
    state = traces.release(context.models, state, "ur5e-2", CIRCULAR, "M2 staging tray")
    state = traces.assemble(context.models, state, "ur5e-3", parts[0], BUFFER)
    for zone in (3, 2, 1):
        state = traces.buffer_advance(
            context.models, state, state[BUFFER][f"zone_{zone}_part"], zone
        )
    assert state[BUFFER]["zone_1_part"] is None
    assert all(before[BUFFER].values())


def test_missing_unknown_and_incompatible_bindings(context):
    task = context.plan()["tasks"][0]
    for parameters in (
        {},
        {**task["parameters"], "part_name": "unknown"},
        {**task["parameters"], "part_name": SQUARE},
    ):
        before = context.snapshot()
        with pytest.raises(ValueError):
            context.prepare({**task, "parameters": parameters})
        assert context.snapshot() == before
    wrong_event = {**task, "event_id": -1}
    with pytest.raises(ValueError, match="capability"):
        context.prepare(wrong_event)
    for malformed in (None, {}, {**task, "event_id": False}, {**task, "resource_id": []}):
        with pytest.raises(ValueError):
            context.prepare(malformed)


def test_shared_participant_disagreement_is_atomic(context):
    task = context.plan()["tasks"][1]
    acknowledge(context, context.plan()["tasks"][0])
    event = next(
        row
        for row in context.models["3D Printing Station"]["events"]
        if row["event_id"] == task["event_id"]
    )
    event["product_effects"]["processCompleted"] = ["place_insert"]
    before = context.snapshot()
    with pytest.raises(ValueError, match="matching participant"):
        context.prepare(task)
    assert context.snapshot() == before


def test_actual_agents_delegate_without_starting_behaviours(monkeypatch, inputs):
    monkeypatch.setenv("OPENAI_API_KEY", "offline-test-not-a-credential")
    from cais_spade_llm.agents.intelligent_product.process_planner import ProcessPlanner
    from cais_spade_llm.agents.intelligent_product.product_agent import ProductAgent
    from cais_spade_llm.agents.resource_agent.resource_agent import ResourceAgent

    pa = SimpleNamespace()
    context = ProductAgent.configure_nominal(pa, **inputs)
    plan = ProductAgent.plan_nominal(pa)
    assert ProcessPlanner.plan_nominal(SimpleNamespace(), context) == plan
    task = plan["tasks"][0]
    ra = SimpleNamespace(agent_name=task["resource_id"])
    ResourceAgent.configure_nominal(ra, context.resources[ra.agent_name])
    assert ResourceAgent.nominal_des_model(ra)["resource_id"] == ra.agent_name
    expected = ResourceAgent.validate_nominal_event(ra, context.models, context.snapshot(), task)
    pending = context.prepare(task)
    assert ProductAgent.acknowledge_nominal(
        pa, {**pending, "status": "completed", "evidence": "simulated"}
    )
    assert context.snapshot() == expected and pa.part_tracker == context.part_tracker
    with pytest.raises(ValueError, match="exactly"):
        ResourceAgent.configure_nominal(
            SimpleNamespace(agent_name="ur5e"), context.resources["ur5e-1"]
        )


@pytest.mark.parametrize("page_name", ["products", "resources"])
def test_complete_pages_read_models_and_saved_runs_without_dispatch(
    page_name, inputs, tmp_path, monkeypatch
):
    from nicegui import core, ui

    from cais_spade_llm.ui.pages import products, resources

    selected_geometry = tmp_path / "selected_geometry.json"
    selected_geometry.write_text(json.dumps({"gazebo": inputs["geometry"]}))
    alternate_manifest = tmp_path / "alternate_product.json"
    alternate_meta = {
        "product_geometry_file": str(selected_geometry),
        "product_order_file": str(
            ROOT
            / "cais_spade_llm/specification/products/orders/assembly_board-v1-recovery-framework.json"
        ),
    }
    alternate_manifest.write_text(json.dumps({"assembly_board-v1": alternate_meta}))
    downloads = []
    monkeypatch.setattr(ui, "download", lambda data, name: downloads.append((data, name)))
    polls = []
    writes = []
    monkeypatch.setattr(ui, "timer", lambda interval, callback: polls.append((interval, callback)))

    class ReadOnlyBridge:
        execution_mode = "simulation"

        def list_product_files(self):
            return [str(PRODUCT_PATH), str(alternate_manifest)]

        def load_config(self, path):
            return read_json(path)

        def get_robot_states(self):
            return {}

        def get_part_tracker(self):
            return {}

        def save_config(self, *args):
            writes.append(args)
            raise AssertionError("page display attempted a write")

    page = ui.column()

    async def check_page():
        monkeypatch.setattr(core, "loop", asyncio.get_running_loop())
        with page:
            (products if page_name == "products" else resources).render(ReadOnlyBridge())
        await asyncio.sleep(0)
        elements = list(page.descendants())
        texts = [getattr(item, "text", "") for item in elements]
        assert "Offline run" not in texts
        assert not any(
            isinstance(item, (ui.select, ui.input, ui.number))
            and item._props.get("label") in {"Saved nominal run", "Saved run.json", "Event index"}
            for item in elements
        )
        assert (
            "Product requirements" if page_name == "products" else "Live Robot Status"
        ) in texts
        assert len(polls) == 1
        refreshed = polls[0][1]()
        if asyncio.iscoroutine(refreshed):
            await refreshed
        if page_name == "products":
            model = next(
                item for item in elements
                if isinstance(item, ui.table)
                and any(column["field"] == "requirement" for column in item.columns)
            )
            from cais_spade_llm.product.environment import EnvironmentProductContext

            context = EnvironmentProductContext(**inputs)
            assert [row["part_name"] for row in model.rows] == [
                *context.selected_parts, context.product_name,
            ]
            for row in model.rows:
                expected = context.product_order["processPlan"].get(
                    row["part_name"], {"location": "Exit", "state": "completed"}
                )
                assert json.loads(row["requirement"]) == expected
                assert row["location"] == context.part_tracker[row["part_name"]]["location"]
            geometry_label = next(
                item for item in elements if getattr(item, "text", "") == "Product Geometry"
            )
            geometry_card = geometry_label.parent_slot.parent
            expansions = [
                item._props["label"]
                for item in geometry_card.descendants()
                if isinstance(item, ui.expansion)
            ]
            assert "assembly_board-v1-recovery-framework.json" in expansions
            assert "assembly_board-v1.json" not in expansions
            assert not any(label.startswith("prusa-") for label in expansions)
            metadata = next(
                item
                for item in geometry_card.descendants()
                if isinstance(item, ui.table)
                and any(column["field"] == "cad_filename" for column in item.columns)
            )
            assert {row["part"]: row["cad_filename"] for row in metadata.rows} == (
                inputs["geometry"]["parts"]["cad_filename_map"]
            )
            template_button = next(
                item
                for item in elements
                if isinstance(item, ui.button) and item.text == "Download Template"
            )
            for listener in template_button._event_listeners.values():
                if listener.type == "click":
                    listener.handler(None)
            assert len(downloads) == 1
            assert json.loads(downloads[0][0]) == {"gazebo": inputs["geometry"]}
            product_selector = next(
                item for item in elements if isinstance(item, ui.select) and item.label == "Product"
            )
            product_selector.set_value(str(alternate_manifest))
            await asyncio.sleep(0)
            expansions = [
                item._props["label"]
                for item in geometry_card.descendants()
                if isinstance(item, ui.expansion)
            ]
            assert "selected_geometry.json" in expansions
            assert "assembly_board-v1-recovery-framework.json" not in expansions
            product_selector.set_value(str(PRODUCT_PATH))
            await asyncio.sleep(0)
        assert len(polls) == 1
        assert not writes
        page.delete()
        await asyncio.sleep(0)

    asyncio.run(check_page())
