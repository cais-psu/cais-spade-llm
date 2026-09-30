"""Nominal resource DES traces, shared custody, and read-only Resources display."""

from __future__ import annotations

import asyncio
import json
from copy import deepcopy
from itertools import product
from pathlib import Path
from unittest.mock import Mock

import pytest

from cais_spade_llm.resources.nominal_conveyor import conveyor_parts
from cais_spade_llm.resources.nominal_des import (
    build_nominal_resource_des_models,
    initial_nominal_valuation,
    project_nominal_event,
)
from cais_spade_llm.resources.robot.robot_task_registry import robot_task_registry
from cais_spade_llm.ui.components.nominal_resource_des import (
    nominal_capability_graph,
    nominal_capability_mermaid,
    nominal_capability_rows,
    nominal_composition_event_rows,
    nominal_des_mermaid,
    nominal_event_rows,
    nominal_function_graph,
    nominal_function_mermaid,
    nominal_inventory_rows,
    nominal_product_process_event_rows,
    nominal_product_process_plan_diagram,
    nominal_resource_default_fields,
    nominal_resource_capability_diagram,
    nominal_resource_product_diagram,
    nominal_resource_state_diagram,
    nominal_resource_diagram,
    nominal_state_rows,
    render_nominal_resource_des,
)

ROOT = Path(__file__).resolve().parents[1]
SCENE_PATH = ROOT / "cais_spade_llm/initialization/recovery_framework_gazebo.json"
SQUARE = "KET4_Square_4mm"
CIRCULAR = "RGOCG4-50_Round_4mm"
BUFFER = "Buffer For Machined parts"


@pytest.fixture(scope="module")
def scene():
    return json.loads(SCENE_PATH.read_text())


@pytest.fixture(scope="module")
def models(scene):
    return build_nominal_resource_des_models(scene)


def event(models, state, resource, name, **params):
    return project_nominal_event(models, state, resource, name, params)


def move_mobile(models, state, source, target):
    return event(
        models,
        state,
        "KMR",
        "move_to_resource",
        source_resource=source,
        target_resource=target,
        arrival_acknowledged=True,
        arm_parked=True,
    )


def home(models, state, robot):
    return event(models, state, robot, "move_home", home_available=True)


def pick(models, state, robot, part, source, **inputs):
    args = {"part_name": part, "origin_resource_location": source, **inputs}
    state = event(models, state, robot, "pick_approach", **args)
    return event(
        models,
        state,
        robot,
        "pick_grasp",
        **args,
        handoff_acknowledged=True,
        source_clear=True,
        robot_clear=True,
    )


def release(models, state, robot, part, destination, **inputs):
    state = event(
        models, state, robot, "place_approach", part_name=part, destination_location=destination
    )
    return event(
        models,
        state,
        robot,
        "place_release",
        part_name=part,
        destination_location=destination,
        handoff_acknowledged=True,
        robot_clear=True,
        **inputs,
    )


def completed_machine_part(models, state, machine, part):
    state = event(models, state, "KMR", "pick_approach", part_name=part,
                  origin_resource_location="Storage")
    state = event(
        models,
        state,
        "KMR",
        "pick_part",
        part_name=part,
        origin_resource_location="Storage",
        handoff_acknowledged=True,
    )
    state = move_mobile(models, state, "Storage", machine)
    state = event(models, state, "KMR", "place_approach", part_name=part,
                  destination_location=machine)
    state = event(
        models,
        state,
        "KMR",
        "place_release",
        part_name=part,
        destination_location=machine,
        handoff_acknowledged=True,
        robot_clear=True,
    )
    state = move_mobile(models, state, machine, "Storage")
    return event(
        models,
        state,
        machine,
        "machine_part",
        part_name=part,
        machining_acknowledged=True,
        robot_clear=True,
    )


def load(models, state, machine, part):
    robot = models[machine]["assignments"]["handling_robot"]
    state = completed_machine_part(models, state, machine, part)
    if state[robot]["resource_state"] == "placed":
        state = home(models, state, robot)
    state = pick(models, state, robot, part, machine)
    position = models["Conveyor"]["assignments"]["loading_positions"][robot]["loading_position"]
    return release(
        models, state, robot, part, "Conveyor", reserved_by=robot, loading_position=position
    )


def advance(models, state, next_locations, delivered_part=None, **overrides):
    params = dict(
        next_locations=next_locations,
        delivered_part=delivered_part,
        robot_clear=True,
        drives_synchronized=True,
        downstream_reserved=True,
    )
    if delivered_part is not None:
        params.update(handoff_acknowledged=True, source_clear=True, destination_acknowledged=True)
    params.update(overrides)
    return event(models, state, "Conveyor", "advance_conveyor", **params)


def buffer_advance(models, state, part, zone):
    return event(
        models,
        state,
        BUFFER,
        "advance_part",
        part_name=part,
        zone=zone,
        downstream_zone=zone + 1,
        downstream_reserved=True,
        robot_clear=True,
        drives_synchronized=True,
        source_clear=True,
        destination_acknowledged=True,
    )


def assemble(models, state, robot, part, source):
    if state[robot]["resource_state"] == "placed":
        state = home(models, state, robot)
    inputs = {"part_identity_observed": True}
    if source == BUFFER:
        inputs.update(zone_stopped=True, stop_raised=True, assembly_destination_available=True)
    state = pick(models, state, robot, part, source, **inputs)
    state = event(
        models,
        state,
        robot,
        "place_approach",
        part_name=part,
        destination_location="assembly_board-v1",
    )
    return event(
        models,
        state,
        robot,
        "place_insert",
        part_name=part,
        destination_location="assembly_board-v1",
        handoff_acknowledged=True,
        assembly_acknowledged=True,
    )


def test_catalog_preserves_assignments_and_only_nominal_tasks(models, scene):
    assert set(models) == {
        "M1",
        "M2",
        "ur5e-1",
        "ur5e-2",
        "ur5e-3",
        "ur5e-4",
        "KMR",
        "Storage",
        "Conveyor",
        BUFFER,
        "3D Printing Station",
        "Exit",
    }
    assert models["M1"]["assignments"]["nominal_parts"] == list(scene["Storage"]["slots"])[:4]
    assert models["M2"]["assignments"]["nominal_parts"] == list(scene["Storage"]["slots"])[4:]
    names = {row["event_name"] for model in models.values() for row in model["events"]}
    assert names == {
        "move_to_resource",
        "move_to_location",
        "pick_part",
        "place_release",
        "machine_part",
        "pick_approach",
        "pick_grasp",
        "place_approach",
        "place_insert",
        "move_home",
        "advance_conveyor",
        "advance_part",
        "print_part",
    }
    assert "DockKMR" not in names and "load_machine" not in names
    assert models["KMR"]["assignments"]["docking_action"] == "/KMR/dock"
    assert set(robot_task_registry()) == {
        "pick_approach",
        "pick_grasp",
        "place_approach",
        "place_insert",
        "move_home",
    }


def test_finite_domains_declared_updates_and_shared_bindings(models):
    for model in models.values():
        fields = model["state_variables"]
        assert set(model["current_valuation"]) == set(fields)
        assert model["marked_state_conditions"]
        for field, declaration in fields.items():
            assert isinstance(declaration["domain"], list)
            assert model["current_valuation"][field] in declaration["domain"]
        for row in model["events"]:
            for field in set(row["guards"]) | set(row["updates"]):
                if "{" not in field:
                    assert field in fields
                    continue
                parameter = field.split("{", 1)[1].removesuffix("}")
                binding = row["parameter_bindings"][parameter]
                reference = binding["from_assignment"]
                for part in models[reference["resource_id"]]["assignments"][reference["field"]]:
                    assert field.replace("{" + parameter + "}", part) in fields
            assert "recovery_visible_steps" not in row
            assert row["controllable"] and row["observable"]
            assert row["capability_transition"]["source"]
            assert row["capability_transition"]["target"]
            for peer in row["participants"]:
                matching = [e for e in models[peer]["events"] if e["event_id"] == row["event_id"]]
                assert len(matching) == 1
                assert matching[0]["parameter_bindings"] == row["parameter_bindings"]


def test_kmr_pick_approach_binds_the_part_for_pick_part(models):
    state = initial_nominal_valuation(models)
    state = event(models, state, "KMR", "pick_approach", part_name=SQUARE,
                  origin_resource_location="Storage")
    assert state["KMR"]["approached_part"] == SQUARE
    before = deepcopy(state)
    with pytest.raises(ValueError, match="guard blocked: KMR.approached_part"):
        event(models, state, "KMR", "pick_part", part_name=CIRCULAR,
              origin_resource_location="Storage", handoff_acknowledged=True)
    assert state == before
    state = event(models, state, "KMR", "pick_part", part_name=SQUARE,
                  origin_resource_location="Storage", handoff_acknowledged=True)
    assert state["KMR"]["approached_part"] is None
    assert state["KMR"]["held_part"] == SQUARE
    assert state["Storage"][f"inventory.{SQUARE}"] is False
    assert state["Storage"][f"inventory.{CIRCULAR}"] is True


def test_machine_assignment_and_mobile_routes_are_guarded(models):
    state = initial_nominal_valuation(models)
    state = event(models, state, "KMR", "pick_approach", part_name=SQUARE,
                  origin_resource_location="Storage")
    state = event(
        models,
        state,
        "KMR",
        "pick_part",
        part_name=SQUARE,
        origin_resource_location="Storage",
        handoff_acknowledged=True,
    )
    state = move_mobile(models, state, "Storage", "M2")
    with pytest.raises(ValueError, match="binding"):
        event(
            models,
            state,
            "KMR",
            "place_release",
            part_name=SQUARE,
            destination_location="M2",
            handoff_acknowledged=True,
            robot_clear=True,
        )
    with pytest.raises(ValueError, match="binding"):
        move_mobile(models, state, "M2", "M1")
    assert state["KMR"]["held_part"] == SQUARE
    assert state["Storage"][f"inventory.{SQUARE}"] is False


@pytest.mark.parametrize(
    "robot,machine,part,other",
    [("ur5e-1", "M1", SQUARE, CIRCULAR), ("ur5e-2", "M2", CIRCULAR, SQUARE)],
)
def test_robot_roles_restrict_parts_locations_and_task_bindings(
    models, robot, machine, part, other
):
    model = models[robot]
    assert model["state_variables"]["held_part"]["domain"] == [
        None,
        *models[machine]["assignments"]["nominal_parts"],
    ]
    assert model["state_variables"]["task_ctx.origin_resource_location"]["domain"] == [
        None,
        machine,
        f"{machine} staging tray",
    ]
    assert model["state_variables"]["task_ctx.destination_location"]["domain"] == [
        None,
        "Conveyor",
        f"{machine} staging tray",
    ]
    state = completed_machine_part(models, initial_nominal_valuation(models), machine, part)
    before = deepcopy(state)
    with pytest.raises(ValueError, match="binding"):
        pick(models, state, robot, other, machine)
    assert state == before
    state = pick(models, state, robot, part, machine)
    assert state[robot]["held_part"] == part


def test_assembly_robot_domains_follow_their_configured_roles(models):
    assert models["ur5e-3"]["state_variables"]["held_part"]["domain"] == [
        None,
        *models["Conveyor"]["assignments"]["nominal_parts"],
        "assembly_board-v1",
    ]
    assert models["ur5e-4"]["state_variables"]["held_part"]["domain"] == [
        None,
        *models["3D Printing Station"]["assignments"]["supported_products"],
    ]


def test_inventory_growth_does_not_duplicate_capability_events(models, scene):
    from cais_spade_llm.resources.environment_models import build_environment_models

    changed = deepcopy(scene)
    part = "KET20_Square_20mm"
    changed["machines"][0]["nominal_parts"].append(part)
    changed["Storage"]["slots"][part] = deepcopy(changed["Storage"]["slots"][SQUARE])
    expanded = build_nominal_resource_des_models(changed)
    for rid in models:
        assert len(expanded[rid]["events"]) == len(models[rid]["events"])
    capabilities = build_environment_models(scene)
    expanded_capabilities = build_environment_models(changed)
    for rid in capabilities:
        assert nominal_capability_rows(expanded_capabilities[rid]) == nominal_capability_rows(capabilities[rid])
        assert nominal_capability_mermaid(expanded_capabilities[rid]) == nominal_capability_mermaid(capabilities[rid])
    assert part in expanded["ur5e-1"]["state_variables"]["held_part"]["domain"]
    assert part not in expanded["ur5e-2"]["state_variables"]["held_part"]["domain"]
    state = load(expanded, initial_nominal_valuation(expanded), "M1", part)
    assert conveyor_parts(expanded["Conveyor"], state["Conveyor"]) == [part]


@pytest.mark.parametrize(
    "changes,missing",
    [
        ({"part_name": "unknown"}, None),
        ({"part_name": None}, None),
        ({"part_name": CIRCULAR.lower()}, None),
        ({"handoff_acknowledged": 1}, None),
        ({"resource_id": "ur5e-1"}, None),
        ({}, "part_name"),
        ({}, "handoff_acknowledged"),
    ],
)
def test_missing_unknown_or_incompatible_parameters_do_not_change_custody(models, changes, missing):
    state = initial_nominal_valuation(models)
    before = deepcopy(state)
    parameters = dict(
        part_name=SQUARE, origin_resource_location="Storage", handoff_acknowledged=True
    )
    parameters.update(changes)
    if missing:
        parameters.pop(missing)
    with pytest.raises(ValueError, match="binding"):
        event(models, state, "KMR", "pick_part", **parameters)
    assert state == before


def test_shared_parameter_definitions_must_match_before_handoff(models):
    changed = deepcopy(models)
    peer = next(row for row in changed["Storage"]["events"] if row["event_name"] == "pick_part")
    peer["parameter_bindings"]["part_name"] = {"equals": CIRCULAR}
    state = initial_nominal_valuation(changed)
    state = event(changed, state, "KMR", "pick_approach", part_name=SQUARE,
                  origin_resource_location="Storage")
    before = deepcopy(state)
    with pytest.raises(ValueError, match="matching participant"):
        event(
            changed,
            state,
            "KMR",
            "pick_part",
            part_name=SQUARE,
            origin_resource_location="Storage",
            handoff_acknowledged=True,
        )
    assert state == before


@pytest.mark.parametrize("machines", [("M1", "M2"), ("M2", "M1")])
def test_two_loading_orders_use_belt_position_not_arrival_order(models, machines):
    state = initial_nominal_valuation(models)
    for machine in machines:
        state = load(models, state, machine, SQUARE if machine == "M1" else CIRCULAR)
    assert conveyor_parts(models["Conveyor"], state["Conveyor"]) == [CIRCULAR, SQUARE]
    assert state["ur5e-1"]["held_part"] is None
    assert state["ur5e-2"]["held_part"] is None
    assert state["ur5e-1"]["part_state"] != "assembled"
    assert state["ur5e-2"]["part_state"] != "assembled"


def test_movement_after_one_load_allows_later_downstream_load(models):
    state = load(models, initial_nominal_valuation(models), "M1", SQUARE)
    state = advance(models, state, {SQUARE: "after loading_position_1"})
    state = load(models, state, "M2", CIRCULAR)
    state = advance(
        models, state, {CIRCULAR: "after loading_position_2", SQUARE: "loading_position_2"}
    )
    assert conveyor_parts(models["Conveyor"], state["Conveyor"]) == [CIRCULAR, SQUARE]


def test_m1_part_at_second_loading_area_blocks_m2_and_allows_staging(models):
    state = load(models, initial_nominal_valuation(models), "M1", SQUARE)
    state = advance(models, state, {SQUARE: "loading_position_2"})
    state = completed_machine_part(models, state, "M2", CIRCULAR)
    state = pick(models, state, "ur5e-2", CIRCULAR, "M2")
    before = deepcopy(state)
    with pytest.raises(ValueError, match="guard blocked"):
        release(
            models,
            state,
            "ur5e-2",
            CIRCULAR,
            "Conveyor",
            reserved_by="ur5e-2",
            loading_position="loading_position_2",
        )
    assert state == before
    state = release(models, state, "ur5e-2", CIRCULAR, "M2 staging tray")
    assert state["M2"]["staging_part"] == CIRCULAR
    assert state["ur5e-2"]["held_part"] is None


@pytest.mark.parametrize(
    "next_locations",
    [
        {CIRCULAR: "loading_position_2", SQUARE: "after loading_position_1"},
        {CIRCULAR: "after loading_position_2"},
        {CIRCULAR: "after loading_position_1", SQUARE: "output_nest"},
        {CIRCULAR: "output_nest", SQUARE: "output_nest"},
        {CIRCULAR: "output_nest", SQUARE: "loading_position_2"},
    ],
)
def test_invalid_coupled_movement_is_atomic(models, next_locations):
    state = load(models, initial_nominal_valuation(models), "M1", SQUARE)
    state = load(models, state, "M2", CIRCULAR)
    before = deepcopy(state)
    with pytest.raises(ValueError):
        advance(models, state, next_locations)
    assert state == before


@pytest.mark.parametrize(
    "overrides",
    [
        {"handoff_acknowledged": False},
        {"source_clear": False},
        {"destination_acknowledged": False},
        {"downstream_reserved": False},
        {"robot_clear": False},
    ],
)
def test_unacknowledged_buffer_handoff_retains_conveyor_custody(models, overrides):
    state = load(models, initial_nominal_valuation(models), "M1", SQUARE)
    state = advance(models, state, {SQUARE: "output_nest"})
    before = deepcopy(state)
    with pytest.raises(ValueError):
        advance(models, state, {}, SQUARE, **overrides)
    assert state == before
    assert state[BUFFER]["zone_1_part"] is None


def test_only_leading_part_at_output_can_transfer(models):
    state = load(models, initial_nominal_valuation(models), "M1", SQUARE)
    state = load(models, state, "M2", CIRCULAR)
    with pytest.raises(ValueError):
        advance(models, state, {SQUARE: "after loading_position_1"}, CIRCULAR)
    state = advance(models, state, {CIRCULAR: "output_nest", SQUARE: "after loading_position_1"})
    with pytest.raises(ValueError):
        advance(models, state, {CIRCULAR: "output_nest"}, SQUARE)
    state = advance(models, state, {SQUARE: "after loading_position_1"}, CIRCULAR)
    assert state[BUFFER]["zone_1_part"] == CIRCULAR
    assert state["Conveyor"][f"part_location.{CIRCULAR}"] is None
    assert conveyor_parts(models["Conveyor"], state["Conveyor"]) == [SQUARE]


def test_full_buffer_applies_backpressure_and_pickup_permits_progress(models):
    state = initial_nominal_valuation(models)
    parts = models["M1"]["assignments"]["nominal_parts"]
    for index, part in enumerate(parts):
        state = load(models, state, "M1", part)
        state = advance(models, state, {part: "output_nest"})
        if index == len(parts) - 1:
            state = load(models, state, "M2", CIRCULAR)
            state = advance(models, state, {CIRCULAR: "after loading_position_2"}, part)
        else:
            state = advance(models, state, {}, part)
        for zone in range(1, 4 - index):
            state = buffer_advance(models, state, part, zone)
    assert all(state[BUFFER].values())
    before = deepcopy(state)
    with pytest.raises(ValueError, match="guard blocked"):
        advance(models, state, {CIRCULAR: "output_nest"})
    assert state == before
    second_circular = models["M2"]["assignments"]["nominal_parts"][1]
    state = completed_machine_part(models, state, "M2", second_circular)
    state = home(models, state, "ur5e-2")
    state = pick(models, state, "ur5e-2", second_circular, "M2")
    with pytest.raises(ValueError, match="guard blocked"):
        release(
            models,
            state,
            "ur5e-2",
            second_circular,
            "Conveyor",
            reserved_by="ur5e-2",
            loading_position="loading_position_2",
        )
    state = release(models, state, "ur5e-2", second_circular, "M2 staging tray")
    assert state["M2"]["staging_part"] == second_circular
    state = assemble(models, state, "ur5e-3", parts[0], BUFFER)
    for zone in (3, 2, 1):
        state = buffer_advance(models, state, state[BUFFER][f"zone_{zone}_part"], zone)
    state = advance(models, state, {CIRCULAR: "output_nest"})
    assert state[BUFFER]["zone_1_part"] is None


@pytest.mark.parametrize(
    "field,value", [("belt_stopped", False), ("loading_reserved_by", "ur5e-2")]
)
def test_loading_and_movement_do_not_overlap(models, field, value):
    state = load(models, initial_nominal_valuation(models), "M1", SQUARE)
    state["Conveyor"][field] = value
    with pytest.raises(ValueError, match="guard blocked"):
        advance(models, state, {SQUARE: "after loading_position_1"})


def test_placement_requires_its_own_reservation_and_clears_it(models):
    state = completed_machine_part(models, initial_nominal_valuation(models), "M1", SQUARE)
    state = pick(models, state, "ur5e-1", SQUARE, "M1")
    state["Conveyor"]["loading_reserved_by"] = "ur5e-2"
    with pytest.raises(ValueError, match="guard blocked"):
        release(
            models,
            state,
            "ur5e-1",
            SQUARE,
            "Conveyor",
            reserved_by="ur5e-1",
            loading_position="loading_position_1",
        )
    state["Conveyor"]["loading_reserved_by"] = "ur5e-1"
    state = release(
        models,
        state,
        "ur5e-1",
        SQUARE,
        "Conveyor",
        reserved_by="ur5e-1",
        loading_position="loading_position_1",
    )
    assert state["Conveyor"]["loading_reserved_by"] is None


@pytest.mark.parametrize(
    "missing",
    ["zone_stopped", "stop_raised", "part_identity_observed", "assembly_destination_available"],
)
def test_buffer_pickup_requires_configured_nominal_conditions(models, missing):
    state = load(models, initial_nominal_valuation(models), "M1", SQUARE)
    state = advance(models, state, {SQUARE: "output_nest"})
    state = advance(models, state, {}, SQUARE)
    for zone in (1, 2, 3):
        state = buffer_advance(models, state, SQUARE, zone)
    ready = dict(
        zone_stopped=True,
        stop_raised=True,
        part_identity_observed=True,
        assembly_destination_available=True,
    )
    ready[missing] = False
    with pytest.raises(ValueError, match="binding"):
        pick(models, state, "ur5e-3", SQUARE, BUFFER, **ready)
    assert state[BUFFER]["zone_4_part"] == SQUARE
    assert state["ur5e-3"]["held_part"] is None


def test_conflicting_initial_custody_is_rejected(scene):
    changed = deepcopy(scene)
    changed[BUFFER]["zones"][0]["initial_part"] = SQUARE
    with pytest.raises(ValueError, match="Duplicate nominal part custody"):
        build_nominal_resource_des_models(changed)


def test_nominal_trace_assembles_all_parts_and_delivers_product_to_exit(models):
    state = initial_nominal_valuation(models)
    for machine in ("M1", "M2"):
        for part in models[machine]["assignments"]["nominal_parts"]:
            state = load(models, state, machine, part)
            state = advance(models, state, {part: "output_nest"})
            state = advance(models, state, {}, part)
            for zone in (1, 2, 3):
                state = buffer_advance(models, state, part, zone)
            state = assemble(models, state, "ur5e-3", part, BUFFER)
    for part in models["3D Printing Station"]["assignments"]["supported_products"]:
        state = assemble(models, state, "ur5e-4", part, "3D Printing Station")
    state = home(models, state, "ur5e-3")
    state = pick(
        models, state, "ur5e-3", "assembly_board-v1", "assembly_board-v1", product_completed=True
    )
    state = release(models, state, "ur5e-3", "assembly_board-v1", "Exit")
    assert state["Exit"] == {
        "resource_state": "occupied",
        "part_name": "assembly_board-v1",
        "product_location": "Exit",
    }
    assert not any(state["Storage"].values())
    assert not any(state["3D Printing Station"].values())
    assert not any(state[BUFFER].values())
    assert conveyor_parts(models["Conveyor"], state["Conveyor"]) == []


def test_premature_product_completion_and_duplicate_printing_are_rejected(models):
    state = initial_nominal_valuation(models)
    with pytest.raises(ValueError, match="guard blocked"):
        pick(
            models,
            state,
            "ur5e-3",
            "assembly_board-v1",
            "assembly_board-v1",
            product_completed=True,
        )
    state = pick(
        models, state, "ur5e-4", "gear_small", "3D Printing Station", part_identity_observed=True
    )
    with pytest.raises(ValueError, match="Duplicate nominal part custody"):
        event(
            models,
            state,
            "3D Printing Station",
            "print_part",
            part_name="gear_small",
            printing_acknowledged=True,
        )


def test_printing_missing_initial_output_creates_only_the_declared_output(scene):
    changed = deepcopy(scene)
    changed["3D Printing Station"]["initial_products"].remove("gear_small")
    models = build_nominal_resource_des_models(changed)
    state = event(
        models,
        initial_nominal_valuation(models),
        "3D Printing Station",
        "print_part",
        part_name="gear_small",
        printing_acknowledged=True,
    )
    assert state["3D Printing Station"]["output.gear_small"] is True


@pytest.mark.parametrize("schema_version", [1, 2, 3])
def test_ui_rows_and_diagrams_use_the_same_des_definitions(models, scene, schema_version):
    from cais_spade_llm.resources.environment_models import build_environment_models

    if schema_version != 1:
        models = build_environment_models(scene, schema_version=schema_version)
    for rid, model in models.items():
        rows = nominal_state_rows(model)
        assert [row["field"] for row in rows] == list(model["state_variables"])
        field = (
            "resource_state" if "resource_state" in model["state_variables"] else rows[0]["field"]
        )
        diagram = nominal_des_mermaid(model, field)
        assert "flowchart LR" in diagram
        capability_diagram = nominal_capability_mermaid(model)
        assert "flowchart TB" in capability_diagram
        for row in nominal_capability_rows(model):
            assert row["event"] in capability_diagram
            if row["signature"] not in capability_diagram.replace("<br/>", " "):
                assert f"event_id={row['id']}" in capability_diagram
                assert f"formal event={row['event']}" in capability_diagram
            definition = next(event for event in model["events"] if event["event_id"] == row["id"])
            for field in definition["guards"]:
                assert f"{rid}.{field} = " in row["source"]
            for field in definition["updates"]:
                assert f"{rid}.{field} = " in row["target"]
            assert all(line.startswith(rid + ".") for line in row["source"].splitlines())
            assert all(line.startswith(rid + ".") for line in row["target"].splitlines())
        for event_name in model["local_event_alphabet"]:
            event_rows = nominal_event_rows(models, rid, event_name)
            assert event_rows and {row["event"] for row in event_rows} == {event_name}
            for row in event_rows:
                assert set(json.loads(row["guards"])) <= set(models)
                assert set(json.loads(row["updates"])) <= set(models)
    release_rows = nominal_event_rows(models, "M1", "place_release")
    assert any("KMR" in row["updates"] and "M1" in row["updates"] for row in release_rows)


def test_capability_graphs_distinguish_robot_paths_without_per_part_expansion(models):
    assert nominal_des_mermaid(models["ur5e-1"], "resource_state") == nominal_des_mermaid(
        models["ur5e-2"], "resource_state"
    )
    for robot, machine, position, other in (
        ("ur5e-1", "M1", "loading_position_1", "M2"),
        ("ur5e-2", "M2", "loading_position_2", "M1"),
    ):
        graph = nominal_capability_mermaid(models[robot])
        assert f"{machine} staging tray" in graph
        assert position in graph
        assert "Conveyor" in graph
        assert other not in graph
        assert "part_name" in graph
        assert SQUARE not in graph and CIRCULAR not in graph
        inventory = [row["part_name"] for row in nominal_inventory_rows(models[robot])]
        assert inventory == models[machine]["assignments"]["nominal_parts"]


def test_function_graph_uses_saved_ur5e_in_and_out_states(scene):
    from cais_spade_llm.resources.environment_models import build_environment_models

    models = build_environment_models(scene)
    for robot in ("ur5e-1", "ur5e-2", "ur5e-3", "ur5e-4"):
        graph = nominal_function_graph(models[robot], models)
        states = {node["id"]: node["state"] for node in graph["nodes"]}
        assert set(states.values()) == {"idle", "at_pick", "picked", "positioned", "placed"}
        transitions = {
            (states[edge["source"]], edge["function_name"], states[edge["target"]])
            for edge in graph["edges"]
        }
        assert transitions == {
            ("idle", "pick_approach", "at_pick"),
            ("at_pick", "pick_grasp", "picked"),
            ("picked", "place_approach", "positioned"),
            ("positioned", "place_insert", "placed"),
            ("idle", "move_home", "idle"),
            ("at_pick", "move_home", "idle"),
            ("placed", "move_home", "idle"),
        }
        assert len(graph["edges"]) == 7
        assert "event_id=" not in nominal_function_mermaid(models[robot], models)
        home = [edge for edge in graph["edges"] if edge["function_name"] == "move_home"]
        assert all(edge["saved_in_state"] == "any" for edge in home)
        assert all(edge["event_ids"] == [home[0]["event_ids"][0]] for edge in home)

    ordinary = models["ur5e-1"]
    place = next(edge for edge in nominal_function_graph(ordinary, models)["edges"]
                 if edge["function_name"] == "place_insert")
    assert {
        event["event_name"] for event in ordinary["events"]
        if event["event_id"] in place["event_ids"]
    } == {"place_release"}


def test_function_graph_handles_variants_planned_and_unowned_resources(scene):
    from cais_spade_llm.resources.environment_models import build_environment_models

    models = build_environment_models(scene)
    kmr = nominal_function_graph(models["KMR"], models)
    states = {node["id"]: node["state"] for node in kmr["nodes"]}
    movement = [edge for edge in kmr["edges"]
                if edge["function_name"] == "move_to_resource"]
    assert {(states[edge["source"]], states[edge["target"]]) for edge in movement} == {
        ("idle", "idle"), ("carrying", "carrying"),
    }
    assert next(edge for edge in movement if states[edge["source"]] == "idle")[
        "program_variants"
    ] == ["empty", "empty_return"]
    assert len({event_id for edge in movement for event_id in edge["event_ids"]}) == 4
    location = [edge for edge in kmr["edges"]
                if edge["function_name"] == "move_to_location"]
    assert {(states[edge["source"]], states[edge["target"]]) for edge in location} == {
        ("idle", "idle"), ("carrying", "carrying"),
    }

    printer = nominal_function_graph(models["3D Printing Station"], models)
    assert len(printer["edges"]) == 1
    assert printer["edges"][0]["program_status"] == "planned"
    assert "print_part (planned)" in nominal_function_mermaid(
        models["3D Printing Station"], models
    )
    for resource in ("M1", "M2", "Conveyor", BUFFER):
        graph = nominal_function_graph(models[resource], models)
        assert len(graph["edges"]) == 1
        assert graph["edges"][0]["event_ids"]
    for resource in ("Storage", "Exit"):
        assert nominal_function_graph(models[resource], models) == {
            "nodes": [], "edges": [],
        }


def test_capability_graph_excludes_events_that_only_consult_resource_guards(models):
    names = {row["event"] for row in nominal_capability_rows(models[BUFFER])}
    assert "advance_conveyor" in names
    assert "pick_grasp" in names
    assert "place_approach" not in names
    assert "place_release" not in names


def test_resource_capability_diagrams_use_existing_event_endpoints(scene):
    from cais_spade_llm.resources.environment_models import build_environment_models

    models = build_environment_models(scene)
    for resource_id, model in models.items():
        diagram = nominal_resource_capability_diagram(model)
        assert diagram["nodes"] and diagram["mermaid"].startswith("flowchart LR"), resource_id
        represented = {
            event_id
            for edge in diagram["edges"]
            for event_id in edge["event_ids"]
        }
        assert represented == {row["id"] for row in nominal_capability_rows(model)}, resource_id
        endpoints = {
            json.dumps(event["capability_transition"][end], sort_keys=True)
            for event in model["events"]
            if event["event_id"] in represented
            for end in ("source", "target")
        }
        assert {json.dumps(node["value"], sort_keys=True) for node in diagram["nodes"]} == endpoints
        assert {row["id"] for row in diagram["state_rows"]} == {
            node["id"] for node in diagram["nodes"]
        }
        assert SQUARE not in diagram["mermaid"]


def test_robot_product_graph_keeps_move_home_guard_in_event_details(scene):
    from cais_spade_llm.resources.environment_models import build_environment_models

    models = build_environment_models(scene)
    model = models["ur5e-1"]
    diagram = nominal_resource_capability_diagram(model)
    values = {node["id"]: node["value"] for node in diagram["nodes"]}
    assert model["state_variables"]["resource_state"]["domain"] == [
        "idle", "at_pick", "picked", "positioned", "placed"
    ]
    home = next(edge for edge in diagram["edges"] if edge["event_name"] == "move_home")
    assert values[home["source"]] == {"resource_id": "ur5e-1", "resource_state": "any"}
    assert values[home["target"]] == {
        "resource_id": "ur5e-1", "resource_location": "home", "resource_state": "idle"
    }
    original = next(event for event in model["events"] if event["event_id"] in home["event_ids"])
    assert original["guards"]["held_part"] == {"equals": None}
    assert original["parameter_bindings"]["home_available"] == {"equals": True}
    assert not any(
        "start -->" in line or "(((" in line for line in diagram["mermaid"].splitlines()
    )

    model["current_valuation"]["resource_state"] = "picked"
    model["marked_state_conditions"] = [{"resource_state": {"equals": "placed"}}]
    assert nominal_resource_capability_diagram(model) == diagram


def test_product_state_diagrams_use_owned_declarations_without_inventory_or_mutation(scene):
    from cais_spade_llm.resources.environment_models import build_environment_models

    models = build_environment_models(scene)
    original = deepcopy(models)
    diagrams = {rid: nominal_resource_product_diagram(model, models)
                for rid, model in models.items()}
    assert models == original
    for rid, graph in diagrams.items():
        owned = {event["event_id"] for event in models[rid]["events"]
                 if event["parameter_bindings"]["resource_id"]["equals"] == rid}
        represented = {event_id for edge in graph["edges"] for event_id in edge["event_ids"]}
        assert represented <= owned
        assert {event["event_id"] for event in graph["events"]} == represented
        assert all(edge["resource_id"] == rid for edge in graph["edges"])
        assert SQUARE not in graph["mermaid"] and CIRCULAR not in graph["mermaid"]
        assert "start -->" not in graph["mermaid"] and "(((" not in graph["mermaid"]
        for node in graph["nodes"]:
            assert "resource_state" not in node["product_state"]
            assert "resource_location" not in node["product_state"]
        if rid in ("Storage", "Exit"):
            assert not graph["nodes"] and not graph["edges"]
        else:
            assert graph["nodes"] and graph["edges"]

    for model in models.values():
        model["current_valuation"] = {}
        model["events"].reverse()
    assert {rid: nominal_resource_product_diagram(model, models)
            for rid, model in models.items()} == diagrams


def test_product_state_machine_results_are_declared_additions_not_current_program_claims(scene):
    from cais_spade_llm.resources.environment_models import build_environment_models

    models = build_environment_models(scene)
    for rid in ("M1", "M2"):
        model = models[rid]
        diagram = nominal_resource_product_diagram(model, models)
        states = {node["id"]: node for node in diagram["nodes"]}
        assert {edge["event_name"] for edge in diagram["edges"]} == {"machine_part"}
        assert {edge["parameters"]["result"] for edge in diagram["edges"]} == {"square", "circle"}
        assert len(model["current_configuration"]["program"]["effects"]) == 1
        for edge in diagram["edges"]:
            before, after = states[edge["source"]], states[edge["target"]]
            assert before["product_state"] == {"part_location": rid}
            assert before["conditions"] == {f"{rid}.resource_state": "loaded"}
            assert after["product_state"]["processCompleted"] == [
                {"process": "trim", "result": edge["parameters"]["result"]}
            ]
            assert "contains" in after["label"]
        event = diagram["events"][0]
        assert event["product_effects"]["processCompleted"] == [
            {"process": "trim", "result": {"set_from_param": "result"}}
        ]
        assert event["product_guards"]["ordered_requirements"]

        model["process_capabilities"]["trim"]["supported_results"] = ["square"]
        limited = nominal_resource_product_diagram(model, models)
        assert len(limited["edges"]) == 1
        assert limited["edges"][0]["parameters"] == {"result": "square"}


def test_product_state_kmr_connects_pickup_travel_and_release_with_custody(scene):
    from cais_spade_llm.resources.environment_models import build_environment_models

    models = build_environment_models(scene)
    graph = nominal_resource_product_diagram(models["KMR"], models)
    nodes = {node["id"]: node for node in graph["nodes"]}
    paths = [(node["id"], []) for node in graph["nodes"]]
    for name in ("pick_part", "move_to_resource", "place_approach", "place_release"):
        paths = [(edge["target"], [*path, edge]) for source, path in paths
                 for edge in graph["edges"]
                 if edge["source"] == source and edge["event_name"] == name]
    assert {nodes[end]["product_state"]["part_location"] for end, _ in paths} == {"M1", "M2"}
    for end, path in paths:
        carried = nodes[path[0]["target"]]
        assert carried["product_state"] == {"part_location": "KMR"}
        assert carried["conditions"]["resource_location"] == "Storage"
        arrived = nodes[path[1]["target"]]
        assert arrived["product_state"] == carried["product_state"]
        assert arrived["conditions"]["resource_location"] == nodes[end]["product_state"]["part_location"]
    events = {event["event_id"]: event for event in graph["events"]}
    for edge in graph["edges"]:
        if edge["event_name"] not in ("move_to_resource", "move_to_location"):
            continue
        for variant in edge["variants"]:
            assert variant["source_conditions"]["KMR.held_part"] == {"reference": "part_name"}
            assert variant["target_conditions"]["KMR.held_part"] == {"reference": "part_name"}
        for event_id in edge["event_ids"]:
            assert events[event_id]["guards"]["KMR"].get("held_part") != {"equals": None}


def test_product_state_robot_destinations_and_assembly_parameters_remain_distinct(scene):
    from cais_spade_llm.resources.environment_models import build_environment_models

    models = build_environment_models(scene)
    graph = nominal_resource_product_diagram(models["ur5e-1"], models)
    in_transit = [node for node in graph["nodes"]
                  if node["product_state"].get("part_state") == "in_transit"]
    assert len(in_transit) == 2
    assert in_transit[0]["product_state"] == in_transit[1]["product_state"]
    assert {node["conditions"]["task_ctx.destination_location"] for node in in_transit} == {
        "Conveyor", "M1 staging tray",
    }
    assert "move_home" not in {edge["event_name"] for edge in graph["edges"]}
    assert all("processCompleted" not in node["product_state"] for node in graph["nodes"])
    for node in in_transit:
        release = next(edge for edge in graph["edges"]
                       if edge["source"] == node["id"] and edge["event_name"] == "place_release")
        target = next(item for item in graph["nodes"] if item["id"] == release["target"])
        assert target["product_state"]["part_location"] == node["conditions"]["task_ctx.destination_location"]

    graph = nominal_resource_product_diagram(models["ur5e-3"], models)
    assembled = next(node for node in graph["nodes"]
                     if node["product_state"].get("part_state") == "assembled")
    assert assembled["product_state"]["processCompleted"] == [
        {"process": "assembly", "target": {"set_from_param": "target"}}
    ]
    assert "GMC_Laser_Plate_Virtual/KET4_Square_4mm" not in graph["mermaid"]


def test_product_state_transport_and_printer_keep_declared_part_effects(scene):
    from cais_spade_llm.resources.environment_models import build_environment_models

    models = build_environment_models(scene)
    buffer = nominal_resource_product_diagram(models[BUFFER], models)
    zones = {node["id"]: node["product_state"]["zone"] for node in buffer["nodes"]}
    assert {(zones[edge["source"]], zones[edge["target"]]) for edge in buffer["edges"]} == {
        (1, 2), (2, 3), (3, 4),
    }
    assert {edge["event_name"] for edge in buffer["edges"]} == {"advance_part"}
    conveyor = nominal_resource_product_diagram(models["Conveyor"], models)
    assert len(conveyor["edges"]) == 2
    assert {edge["event_name"] for edge in conveyor["edges"]} == {"advance_conveyor"}
    assert any(edge["source"] == edge["target"] for edge in conveyor["edges"])
    assert any(event["collection_effects"]["Conveyor"] for event in conveyor["events"])
    assert all("processCompleted" not in node["product_state"]
               for graph in (buffer, conveyor) for node in graph["nodes"])
    printer = nominal_resource_product_diagram(models["3D Printing Station"], models)
    assert len(printer["edges"]) == 1
    states = [node["product_state"] for node in printer["nodes"]]
    assert any(state.get("output.{part_name}") is False for state in states)
    assert any(state.get("processCompleted") == [{"process": "print_part"}] for state in states)
    assert "print_part (planned)" in printer["mermaid"]


def test_resource_state_diagram_follows_the_five_robot_states_and_guards(scene):
    from cais_spade_llm.resources.environment_models import build_environment_models

    models = build_environment_models(scene)
    for robot in ("ur5e-1", "ur5e-2", "ur5e-3", "ur5e-4"):
        model = models[robot]
        diagram = nominal_resource_state_diagram(model, "resource_state")
        values = {node["id"]: node["value"] for node in diagram["nodes"]}
        assert list(values.values()) == ["idle", "at_pick", "picked", "positioned", "placed"]
        assert values[diagram["initial_id"]] == "idle"
        arrows = {
            (values[edge["source"]], edge["event_name"], values[edge["target"]])
            for edge in diagram["edges"]
        }
        assert {("idle", "pick_approach", "at_pick"),
                ("at_pick", "pick_grasp", "picked"),
                ("picked", "place_approach", "positioned"),
                ("placed", "move_home", "idle")} <= arrows
        assert {source for source, name, _ in arrows if name == "move_home"} == {
            "idle", "at_pick", "placed"
        }
        approach = next(
            edge for edge in diagram["edges"]
            if values[edge["source"]] == "idle"
            and edge["event_name"] == "pick_approach"
        )
        assert set(approach["event_ids"]) == {
            event["event_id"] for event in model["events"]
            if event["event_name"] == "pick_approach"
        }


def test_resource_state_diagrams_use_exact_fields_events_and_symbolic_effects(scene):
    from cais_spade_llm.resources.environment_models import build_environment_models

    models = build_environment_models(scene)
    for resource_id, model in models.items():
        for field in model["state_variables"]:
            diagram = nominal_resource_state_diagram(model, field)
            represented = {
                event_id for edge in diagram["edges"] for event_id in edge["event_ids"]
            }
            expected = {
                event["event_id"] for event in model["events"]
                if field in event["updates"] or field in event.get("collection_effects", {})
            }
            assert represented == expected, (resource_id, field)
            assert diagram["mermaid"].startswith("flowchart LR")
            assert len(diagram["state_rows"]) == len(diagram["nodes"])

    kmr = nominal_resource_state_diagram(models["KMR"], "resource_location")
    assert {edge["event_name"] for edge in kmr["edges"]} == {
        "move_to_resource", "move_to_location",
    }
    buffer = nominal_resource_state_diagram(models[BUFFER], "zone_1_part")
    assert {json.dumps(node["value"], sort_keys=True) for node in buffer["nodes"]} == {
        "null", '{"reference": "delivered_part"}', '{"reference": "part_name"}'
    }
    zone_4 = nominal_resource_state_diagram(models[BUFFER], "zone_4_part")
    assert {edge["event_name"] for edge in zone_4["edges"]} == {
        "advance_part", "pick_grasp"
    }
    conveyor = nominal_resource_state_diagram(models["Conveyor"], "part_location.{part_name}")
    movement = [edge for edge in conveyor["edges"] if edge["event_name"] == "advance_conveyor"]
    values = {node["id"]: node["value"] for node in conveyor["nodes"]}
    assert all(isinstance(values[edge["target"]], dict) for edge in movement)
    assert all(values[edge["target"]] == {
        "collection_effect": "acknowledged shared belt occupancy"
    } for edge in movement)


def test_all_resource_default_diagrams_retain_exact_event_variants(scene):
    from cais_spade_llm.resources.environment_models import build_environment_models

    models = build_environment_models(scene)
    expected = {
        **{robot: ["resource_state"] for robot in
           ("ur5e-1", "ur5e-2", "ur5e-3", "ur5e-4")},
        "M1": ["resource_state", "staging_part"],
        "M2": ["resource_state", "staging_part"],
        "KMR": ["resource_state", "resource_location"],
        "Conveyor": [
            "part_location.{part_name}", "part_order.{part_name}",
            "loading_reserved_by",
        ],
        BUFFER: ["zone_1_part", "zone_2_part", "zone_3_part", "zone_4_part"],
        "Storage": ["inventory.{part_name}"],
        "3D Printing Station": ["output.{part_name}"],
        "Exit": ["resource_state", "product_location"],
    }
    assert {rid: nominal_resource_default_fields(model) for rid, model in models.items()} == expected
    for resource_id, fields in expected.items():
        details = {row["event_id"]: row for row in
                   nominal_composition_event_rows(models, resource_id)}
        assert set(details) == {event["event_id"] for event in models[resource_id]["events"]}
        for field in fields:
            diagram = nominal_resource_state_diagram(models[resource_id], field)
            represented = {event_id for edge in diagram["edges"]
                           for event_id in edge["event_ids"]}
            assert represented <= details.keys(), (resource_id, field)
            for event_id in represented:
                original = next(e for e in models[resource_id]["events"]
                                if e["event_id"] == event_id)
                row = details[event_id]
                assert row["event_name"] == original["event_name"]
                assert row["parameter_bindings"] == original["parameter_bindings"]
                assert row["participants"] == original["participants"]
                assert row["actor"] == original["parameter_bindings"]["resource_id"]["equals"]
                assert row["guards"][resource_id] == original["guards"]
                assert row["updates"][resource_id] == original["updates"]
                assert row["product_effects"] == original["product_effects"]
        if resource_id == "Conveyor":
            assert any(row["event_name"] == "place_approach" for row in details.values())
            assert any(len(edge["event_ids"]) > 1 for field in fields
                       for edge in nominal_resource_state_diagram(
                           models[resource_id], field
                       )["edges"])


def test_product_process_plan_stages_keep_order_initial_print_and_exact_effects(scene):
    from cais_spade_llm.resources.environment_models import build_environment_models

    models = build_environment_models(scene)
    trim = {"process": "trim", "result": "square"}
    printing = {"process": "print_part"}
    assembly = {"process": "assembly", "target": "GMC_Laser_Plate_Virtual/KET4_Square_4mm"}
    requirements = {SQUARE: [
        {"processesToComplete": [trim, printing]},
        {"processesToComplete": [assembly]},
    ]}
    initial = {SQUARE: {"processCompleted": [printing]}}
    diagram = nominal_product_process_plan_diagram(requirements, initial, SQUARE)
    values = {node["id"]: node["completed"] for node in diagram["nodes"]}
    assert values[diagram["initial_id"]] == [printing]
    assert len(diagram["nodes"]) == 5 and len(diagram["edges"]) == 5
    assert {tuple(edge["requirement"].items()) for edge in diagram["edges"]} == {
        tuple(effect.items()) for effect in (trim, printing, assembly)
    }
    assert all(set(map(str, values[edge["source"]])) == set(map(str, [trim, printing]))
               for edge in diagram["edges"] if edge["requirement"] == assembly)
    assert any(edge["requirement"] == trim and values[edge["source"]] == [printing]
               for edge in diagram["edges"])
    assert 'result=square' in diagram["mermaid"] and 'target=GMC_Laser_Plate_Virtual/KET4_Square_4mm' in diagram["mermaid"]
    assert f'    start --> {diagram["initial_id"]}' in diagram["mermaid"]
    assert '(((' in diagram["mermaid"]

    variants = nominal_product_process_event_rows(models, requirements, SQUARE)
    by_requirement = {name: {row["event_name"] for row in variants
                             if row["requirement"]["process"] == name}
                      for name in ("trim", "print_part", "assembly")}
    assert "machine_part" in by_requirement["trim"]
    assert "print_part" in by_requirement["print_part"]
    assert "place_insert" in by_requirement["assembly"]
    assert all(row["event_id"] in {
        event["event_id"] for event in models[row["actor"]]["events"]
    } for row in variants)


def test_resource_state_diagram_uses_configured_initial_and_complete_marking(scene):
    from cais_spade_llm.resources.environment_models import build_environment_models

    models = build_environment_models(scene)
    model = models["ur5e-1"]
    initial = deepcopy(model["current_valuation"])
    diagram = nominal_resource_state_diagram(model, "resource_state", initial_valuation=initial)
    assert "start --> s0" in diagram["mermaid"]
    assert "(((" not in diagram["mermaid"]
    model["current_valuation"]["resource_state"] = "picked"
    assert nominal_resource_state_diagram(
        model, "resource_state", initial_valuation=initial
    ) == diagram

    storage = models["Storage"]
    storage["marked_state_conditions"] = [
        {"inventory.{part_name}": {"equals": True}}
    ]
    marked = nominal_resource_state_diagram(storage, "inventory.{part_name}")
    true_id = next(node["id"] for node in marked["nodes"] if node["value"] is True)
    assert f'{true_id}((("{true_id}<br/>true")))' in marked["mermaid"]
    assert marked["initial_id"] is None


def test_resource_diagrams_show_kmr_routes_buffer_handoffs_and_shared_belt(scene):
    from cais_spade_llm.resources.environment_models import build_environment_models

    models = build_environment_models(scene)
    kmr = nominal_resource_capability_diagram(models["KMR"])
    values = {node["id"]: node["value"] for node in kmr["nodes"]}
    routes = {
        (values[edge["source"]]["resource_location"], values[edge["target"]]["resource_location"])
        for edge in kmr["edges"] if edge["event_name"] == "move_to_resource"
    }
    assert routes == {("Storage", "M1"), ("M1", "Storage"),
                      ("Storage", "M2"), ("M2", "Storage")}

    buffer = nominal_resource_capability_diagram(models[BUFFER])
    states = {node["id"]: node["value"] for node in buffer["nodes"]}
    transitions = {
        (json.dumps(states[edge["source"]], sort_keys=True), edge["event_name"],
         json.dumps(states[edge["target"]], sort_keys=True))
        for edge in buffer["edges"]
    }
    conveyor_state = {"part_location": "Conveyor"}

    def zone(number):
        return {"part_location": BUFFER, "zone": number}

    gripper = {"part_location": "ur5e-3", "part_state": "in_gripper"}
    for source, event_name, target in [
        (conveyor_state, "advance_conveyor", zone(1)),
        *((zone(number), "advance_part", zone(number + 1)) for number in (1, 2, 3)),
        (zone(4), "pick_grasp", gripper),
    ]:
        assert (json.dumps(source, sort_keys=True), event_name,
                json.dumps(target, sort_keys=True)) in transitions

    conveyor = nominal_resource_capability_diagram(models["Conveyor"])
    values = {node["id"]: node["value"] for node in conveyor["nodes"]}
    movement = [edge for edge in conveyor["edges"] if edge["event_name"] == "advance_conveyor"]
    assert {(json.dumps(values[edge["source"]], sort_keys=True),
             json.dumps(values[edge["target"]], sort_keys=True)) for edge in movement} == {
        (json.dumps(conveyor_state, sort_keys=True), json.dumps(conveyor_state, sort_keys=True)),
        (json.dumps(conveyor_state, sort_keys=True), json.dumps(zone(1), sort_keys=True)),
    }
    movement_ids = {
        event["event_id"] for event in models["Conveyor"]["events"]
        if event["event_name"] == "advance_conveyor"
    }
    assert {event_id for edge in movement for event_id in edge["event_ids"]} == movement_ids
    assert "next_locations" not in conveyor["mermaid"]
    detailed = nominal_capability_graph(models["Conveyor"], models)
    assert all(edge["collection_effects"]["Conveyor"] for edge in detailed["edges"]
               if edge["event"]["event_name"] == "advance_conveyor")


def test_repeated_endpoint_arrows_keep_original_event_ids(scene):
    from cais_spade_llm.resources.environment_models import build_environment_models

    model = build_environment_models(scene)["Storage"]
    second = deepcopy(model["events"][0])
    second["event_id"] += 1000
    second["parameter_bindings"]["handoff_acknowledged"] = {"equals": False}
    model["events"].append(second)
    diagram = nominal_resource_capability_diagram(model)
    assert len(diagram["edges"]) == 1
    assert diagram["edges"][0]["event_ids"] == sorted(
        event["event_id"] for event in model["events"] if event["event_name"] == "pick_part"
    )
    detailed = nominal_capability_graph(model, {"Storage": model})
    for original in (event for event in model["events"] if event["event_name"] == "pick_part"):
        variant = next(
            edge for edge in detailed["edges"] if edge["event_id"] == original["event_id"]
        )
        assert variant["resource_id"] == "KMR"
        assert variant["event"] == original
        assert variant["guards"]["Storage"] == original["guards"]
        assert variant["updates"]["Storage"] == original["updates"]


def test_loading_capabilities_connect_to_the_shared_conveyor_movement(scene):
    from cais_spade_llm.resources.environment_models import build_environment_models

    models = build_environment_models(scene)
    rows = nominal_capability_rows(models["Conveyor"])
    loading = [row for row in rows if row["event"] == "place_release"]
    assert {row["resource_id"] for row in loading} == {"ur5e-1", "ur5e-2"}
    for row in loading:
        position = "loading_position_1" if row["resource_id"] == "ur5e-1" else "loading_position_2"
        assert f"Conveyor.part_location.{{part_name}} = {position}" in row["target"]
    assert {row["event"] for row in rows} == {"place_release", "advance_conveyor"}
    for row in rows:
        assert "Conveyor.belt_stopped" in row["source"]
        assert "Conveyor.part_order.{part_name}" in row["target"]
    assert _graph_paths(nominal_capability_graph(models["Conveyor"], models),
                        ["place_release", "advance_conveyor"])


def _graph_paths(graph, names):
    paths = [(node["id"], []) for node in graph["nodes"]]
    for name in names:
        paths = [(edge["target"], [*path, edge]) for source, path in paths
                 for edge in graph["edges"]
                 if edge["source"] == source and edge["event"]["event_name"] == name]
    return [path for _, path in paths]


def test_environment_graphs_project_local_states_and_retain_shared_event_details(scene):
    from cais_spade_llm.resources.environment_models import build_environment_models

    models = build_environment_models(scene)
    original = deepcopy(models)
    for rid, model in models.items():
        graph = nominal_capability_graph(model, models)
        assert graph["nodes"] and graph["edges"], rid
        for node in graph["nodes"]:
            assert all(field.startswith(rid + ".") for field in node["state"])
            assert "processCompleted" not in node["state"]
        represented = {edge["event_id"] for edge in graph["edges"]}
        assert represented == {row["id"] for row in nominal_capability_rows(model)}
        assert represented == {
            event["event_id"] for event in model["events"]
            if event["updates"] or event.get("collection_effects")
            or event["parameter_bindings"]["resource_id"] == {"equals": rid}
        }
        for edge in graph["edges"]:
            actual = next(event for event in models[edge["resource_id"]]["events"]
                          if event["event_id"] == edge["event_id"])
            assert actual["parameter_bindings"] == edge["event"]["parameter_bindings"]
            for participant in actual["participants"]:
                local = next(event for event in models[participant]["events"]
                             if event["event_id"] == edge["event_id"])
                assert edge["guards"][participant] == local["guards"]
                assert edge["updates"][participant] == local["updates"]
                for field in ("collection_guards", "collection_effects"):
                    if field in local:
                        assert edge[field][participant] == local[field]
            source = graph["nodes"][edge["source"]]["state"]
            target = graph["nodes"][edge["target"]]["state"]
            for field, guard in edge["guards"][rid].items():
                field = f"{rid}.{field}"
                if field not in source or isinstance(source[field], dict):
                    continue
                if "not_equals" in guard:
                    assert source[field] != guard["not_equals"], (rid, edge["event_id"], field)
                elif "equals" in guard:
                    assert source[field] == guard["equals"], (rid, edge["event_id"], field)
            for field, update in edge["updates"][rid].items():
                if "set" in update:
                    assert target[f"{rid}.{field}"] == update["set"]
            assert edge["event"]["product_effects"] == actual["product_effects"]
        encoded = nominal_capability_mermaid(model, models=models)
        assert SQUARE not in encoded and CIRCULAR not in encoded
    assert models == original


@pytest.mark.parametrize("machine,robot", [("M1", "ur5e-1"), ("M2", "ur5e-2")])
def test_machining_graph_shows_local_loading_machining_pickup_and_staging(scene, machine, robot):
    from cais_spade_llm.resources.environment_models import build_environment_models

    models = build_environment_models(scene)
    graph = nominal_capability_graph(models[machine], models)
    paths = _graph_paths(graph, ["place_release", "machine_part", "pick_grasp", "place_release", "pick_grasp"])
    paths = [path for path in paths
             if path[0]["resource_id"] == "KMR"
             and path[2]["event"]["parameter_bindings"]["origin_resource_location"] == {"equals": machine}
             and path[3]["resource_id"] == robot
             and path[4]["event"]["parameter_bindings"]["origin_resource_location"] == {"equals": f"{machine} staging tray"}]
    assert paths
    for path in paths:
        assert graph["nodes"][path[0]["target"]]["state"][f"{machine}.resource_state"] == "loaded"
        assert graph["nodes"][path[1]["target"]]["state"][f"{machine}.resource_state"] == "completed"
        assert graph["nodes"][path[2]["target"]]["state"][f"{machine}.resource_state"] == "idle"
        assert path[1]["event"]["product_effects"]["processCompleted"] == [
            {"process": "trim", "result": {"set_from_param": "result"}}
        ]
        staged = graph["nodes"][path[3]["target"]]["state"]
        assert staged[f"{machine}.staging_part"] == {"reference": "part_name"}
        assert graph["nodes"][path[4]["target"]]["state"][f"{machine}.staging_part"] is None
    assert {edge["event"]["event_name"] for edge in graph["edges"]} == {
        "place_release", "machine_part", "pick_grasp"
    }


def test_kmr_graph_preserves_custody_while_moving_and_does_not_invent_routes(scene):
    from cais_spade_llm.resources.environment_models import build_environment_models

    models = build_environment_models(scene)
    graph = nominal_capability_graph(models["KMR"], models)
    paths = _graph_paths(graph, ["pick_approach", "pick_part", "move_to_resource",
                                "place_approach", "place_release"])
    assert {graph["nodes"][path[-1]["target"]]["state"]["KMR.resource_location"] for path in paths} == {"M1", "M2"}
    for path in paths:
        for edge in path[1:4]:
            assert graph["nodes"][edge["target"]]["state"]["KMR.held_part"] == {"reference": "part_name"}
        assert graph["nodes"][path[-1]["target"]]["state"]["KMR.held_part"] is None
    for edge in graph["edges"]:
        if edge["event"]["event_name"] == "move_to_resource":
            source = graph["nodes"][edge["source"]]["state"]
            target = graph["nodes"][edge["target"]]["state"]
            assert source.get("KMR.held_part") == target.get("KMR.held_part")
    models["KMR"]["events"] = [event for event in models["KMR"]["events"]
                              if not any(event["parameter_bindings"].get(field) == {"equals": "M2"}
                                         for field in ("source_resource", "target_resource", "destination_location"))]
    graph = nominal_capability_graph(models["KMR"], models)
    assert not any(node["state"].get("KMR.resource_location") == "M2" for node in graph["nodes"])


def test_graph_identity_ignores_label_field_order_and_keeps_conflicting_guards(scene):
    from cais_spade_llm.resources.environment_models import build_environment_models

    models = build_environment_models(scene)
    original = nominal_capability_mermaid(models["M1"], models=models)
    for model in models.values():
        model["events"].reverse()
        for event in model["events"]:
            for section in ("guards", "updates"):
                event[section] = dict(reversed(list(event[section].items())))
            for endpoint in ("source", "target"):
                event["capability_transition"][endpoint] = dict(reversed(
                    list(event["capability_transition"][endpoint].items())
                ))
    assert nominal_capability_mermaid(models["M1"], models=models) == original
    for event in models["M1"]["events"]:
        if event["event_name"] == "pick_grasp" and "resource_state" in event["guards"]:
            event["guards"]["resource_state"] = {"equals": "loaded"}
    graph = nominal_capability_graph(models["M1"], models)
    assert not any(path[1]["event"]["parameter_bindings"]["origin_resource_location"] == {"equals": "M1"}
                   for path in _graph_paths(graph, ["machine_part", "pick_grasp"]))


def test_neighbors_change_event_details_without_changing_local_graphs(scene):
    from cais_spade_llm.resources.environment_models import build_environment_models

    models = build_environment_models(scene)
    for rid, model in models.items():
        original = nominal_capability_graph(model, models)
        changed = deepcopy(models)
        for neighbor, descriptor in changed.items():
            if neighbor == rid:
                continue
            for event in descriptor["events"]:
                event["guards"] = {"neighbor_internal_state": {"equals": "changed"}}
                event["updates"] = {"neighbor_internal_state": {"set": "changed"}}
                event["capability_transition"] = {
                    "source": {"neighbor_internal_state": "before"},
                    "target": {"neighbor_internal_state": "after"},
                }
        current = nominal_capability_graph(model, changed)
        assert current["nodes"] == original["nodes"]
        assert [(edge["source"], edge["target"], edge["event_id"], edge["resource_id"])
                for edge in current["edges"]] == [
                    (edge["source"], edge["target"], edge["event_id"], edge["resource_id"])
                    for edge in original["edges"]
                ]
        assert current["edges"] != original["edges"]
        diagram = nominal_capability_mermaid(model, models=models)
        assert nominal_capability_mermaid(model, models=changed) == diagram
        assert nominal_capability_mermaid(model) == diagram
        assert nominal_capability_rows(model, changed) == nominal_capability_rows(model, models)


def test_current_occupancy_does_not_seed_or_restrict_capability_graphs(scene):
    from cais_spade_llm.resources.environment_models import build_environment_models

    for model in build_environment_models(scene).values():
        original = nominal_capability_graph(model)
        changed = deepcopy(model)
        changed["current_valuation"] = {}
        assert nominal_capability_graph(changed) == original


def test_shared_collection_effects_remain_visible_without_scalar_updates(scene):
    from cais_spade_llm.resources.environment_models import build_environment_models

    model = build_environment_models(scene)["Conveyor"]
    movement = next(event for event in model["events"] if event["event_name"] == "advance_conveyor")
    assert movement["updates"] == {}
    movement["parameter_bindings"]["resource_id"] = {"equals": "ur5e-1"}
    model["events"] = [movement]
    graph = nominal_capability_graph(model)
    assert {edge["event_id"] for edge in graph["edges"]} == {movement["event_id"]}
    assert {row["id"] for row in nominal_capability_rows(model)} == {movement["event_id"]}
    assert all(edge["resource_id"] == "ur5e-1" for edge in graph["edges"])
    for edge in graph["edges"]:
        assert edge["collection_effects"]["Conveyor"] == movement["collection_effects"]
        assert all(field.startswith("Conveyor.") for field in graph["nodes"][edge["target"]]["state"])
    movement.pop("collection_effects")
    assert nominal_capability_graph(model) == {"nodes": [], "edges": []}
    assert nominal_capability_rows(model) == []


def test_local_graph_highlights_exact_shared_event_id(scene):
    from cais_spade_llm.resources.environment_models import build_environment_models

    model = build_environment_models(scene)["Conveyor"]
    graph = nominal_capability_graph(model)
    event_id = next(edge["event_id"] for edge in graph["edges"]
                    if edge["event"]["event_name"] == "place_release")
    diagram = nominal_capability_mermaid(model, event_id)
    assert [line for line in diagram.splitlines() if "linkStyle" in line] == [
        f"    linkStyle {index} stroke:#d97706,stroke-width:4px"
        for index, edge in enumerate(graph["edges"]) if edge["event_id"] == event_id
    ]


def test_shared_movement_buffer_backpressure_and_handoff_are_visible(scene):
    from cais_spade_llm.resources.environment_models import build_environment_models

    models = build_environment_models(scene)
    conveyor = nominal_capability_graph(models["Conveyor"], models)
    assert _graph_paths(conveyor, ["place_release", "advance_conveyor", "advance_conveyor"])
    for edge in conveyor["edges"]:
        assert edge["guards"][BUFFER]["zone_1_part"] == {"equals": None}
        if edge["event"]["event_name"] != "advance_conveyor":
            continue
        assert "Conveyor" in edge["collection_effects"]
        target = conveyor["nodes"][edge["target"]]["state"]
        for field in ("part_location.{part_name}", "part_order.{part_name}"):
            assert target[f"Conveyor.{field}"] == {
                "collection_effect": edge["collection_effects"]["Conveyor"][field]
            }
        if edge["updates"][BUFFER]:
            assert edge["guards"]["Conveyor"]["part_location.{delivered_part}"] == {"equals": "output_nest"}
            assert edge["updates"][BUFFER]["zone_1_part"] == {"set_from_param": "delivered_part"}
    buffer = nominal_capability_graph(models[BUFFER], models)
    paths = _graph_paths(buffer, ["advance_conveyor", "advance_part", "advance_part", "advance_part",
                                "pick_grasp"])
    paths = [path for path in paths
             if [edge["event"]["parameter_bindings"]["zone"]["equals"]
                 for edge in path[1:4]] == [1, 2, 3]]
    assert paths
    for path in paths:
        assert path[0]["guards"][BUFFER]["zone_1_part"] == {"equals": None}
        assert buffer["nodes"][path[0]["target"]]["state"][f"{BUFFER}.zone_1_part"] == {"reference": "delivered_part"}
        assert path[-1]["updates"][BUFFER]["zone_4_part"] == {"set": None}
        for index, edge in enumerate(path[1:4], 2):
            assert edge["guards"][BUFFER][f"zone_{index}_part"] == {"equals": None}
    for rid, names in {
        "ur5e-1": ["pick_approach", "pick_grasp", "place_approach", "place_release", "move_home"],
        "ur5e-2": ["pick_approach", "pick_grasp", "place_approach", "place_release", "move_home"],
        "ur5e-3": ["pick_approach", "pick_grasp", "place_approach", "place_insert", "move_home"],
        "ur5e-4": ["pick_approach", "pick_grasp", "place_approach", "place_insert", "move_home"],
        "Storage": ["pick_part"],
        "3D Printing Station": ["print_part", "pick_grasp"],
        "Exit": ["pick_grasp", "place_release"],
    }.items():
        assert _graph_paths(nominal_capability_graph(models[rid], models), names), rid


def test_resources_panel_renders_all_resources_with_only_read_calls(scene):
    from nicegui import ui

    class ReadOnlyBridge:
        def __init__(self):
            self.reads = []

        def load_config(self, path):
            self.reads.append(("load_config", path))
            return deepcopy(scene)

    bridge = ReadOnlyBridge()
    with ui.column() as panel:
        refresh = render_nominal_resource_des(bridge)
        asyncio.run(refresh())
    elements = list(panel.descendants())
    texts = [getattr(element, "text", "") for element in elements]
    assert "Live observations" not in texts
    assert "Resource capability graph" in texts
    assert "Product state graph" in texts
    assert "Product processPlan stages" in texts
    assert "No active processPlan." in texts
    assert any("all resident parts together" in text for text in texts)
    assert "Configured / assumed initial values — not live observations." in texts
    for rid in build_nominal_resource_des_models(scene):
        assert rid in texts
    assert {row[0] for row in bridge.reads} == {"load_config"}
    panel.delete()


def test_complete_resources_page_keeps_one_read_only_live_status_poll(scene, models, monkeypatch):
    from nicegui import core, ui

    from cais_spade_llm.ui.pages.resources import render

    polls = []
    monkeypatch.setattr(ui, "timer", lambda interval, callback: polls.append((interval, callback)) or Mock())

    class ReadOnlyBridge:
        def __init__(self):
            self.reads = []
            self.states = {"ur5e-1": {"current_state": "picked", "held_part": SQUARE}}

        def load_config(self, path):
            self.reads.append("load_config")
            return deepcopy(scene)

        def get_robot_states(self):
            self.reads.append("get_robot_states")
            return deepcopy(self.states)

        def get_runtime_recoveries(self):
            self.reads.append("get_runtime_recoveries")
            return []

    bridge = ReadOnlyBridge()
    with ui.column() as page:
        render(bridge)
        asyncio.run(polls[0][1]())
    texts = [getattr(item, "text", "") for item in page.descendants()]
    assert texts.count("Resources") == 1
    assert "Live Resource Status" in texts
    assert "Resource Agent Chat" not in texts
    assert "Live observations" not in texts
    assert "Save" not in texts
    assert not any(isinstance(item, ui.textarea) for item in page.descendants())
    assert len(polls) == 1 and polls[0][0] == 2.0
    assert bridge.reads == ["load_config", "get_runtime_recoveries", "get_robot_states", "load_config"]

    selector = next(
        item
        for item in page.descendants()
        if isinstance(item, ui.select) and item.label == "Resource"
    )

    async def switch_resources():
        monkeypatch.setattr(core, "loop", asyncio.get_running_loop())
        from cais_spade_llm.resources.environment_models import build_environment_models

        runtime_models = build_environment_models(scene)
        for resource in models:
            selector.set_value(resource)
            await asyncio.sleep(0)
            graphs = [item.content for item in page.descendants() if isinstance(item, ui.mermaid)]
            expected = [nominal_resource_diagram(
                runtime_models[resource], runtime_models
            )["mermaid"]]
            product_graph = nominal_resource_product_diagram(runtime_models[resource], runtime_models)
            if product_graph["edges"]:
                expected.append(product_graph["mermaid"])
            else:
                assert any(getattr(item, "text", "") == "No owned product-state transitions."
                           for item in page.descendants())
            assert graphs == expected, resource
            for title in ("DES details", "Shared event details", "Current environmental exploration"):
                expansion = next(item for item in page.descendants()
                                 if isinstance(item, ui.expansion) and item.text == title)
                expansion.set_value(True)
            if resource == BUFFER:
                buffer_details = next(item for item in page.descendants()
                                      if isinstance(item, ui.expansion)
                                      and item.text == "Buffer graph conditions")
                buffer_details.set_value(True)
                await asyncio.sleep(0)
                texts = [getattr(item, "text", "") for item in buffer_details.descendants()]
                assert "The pickup removes the final part." in texts
                assert "At least one other zone remains occupied after pickup." in texts
                variants = [variant for item in buffer_details.descendants()
                            if isinstance(item, ui.code) for variant in json.loads(item.content)]
                assert len(variants) == 28
                assert all(set(variant["source_conditions"]) == set(runtime_models[BUFFER]["state_variables"])
                           for variant in variants)
            await asyncio.sleep(0)
            assert len([item for item in page.descendants() if isinstance(item, ui.mermaid)]) == len(expected)
            if resource == "ur5e-1":
                place = next(item for item in page.descendants()
                             if isinstance(item, ui.expansion) and item.text == "place_insert")
                codes = [json.loads(item.content) for item in place.descendants()
                         if isinstance(item, ui.code)]
                assert any(isinstance(value, dict)
                           and {"in_state", "out_state", "event_variants"} <= set(value)
                           for value in codes)
                texts = [getattr(item, "text", "") for item in place.descendants()]
                assert "Formal event: place_release" in texts
                steps = scene["resource_programs"]["resources"][resource]["functions"][
                    "place_insert"
                ]["program"]["steps"]
                assert all(step["op"] in texts for step in steps)
            assert any(
                getattr(item, "text", "") == "Marked state conditions: " + json.dumps(
                    runtime_models[resource]["marked_state_conditions"], ensure_ascii=False
                )
                for item in page.descendants()
            )
        await asyncio.sleep(0)

    asyncio.run(switch_resources())
    assert bridge.reads == ["load_config", "get_runtime_recoveries", "get_robot_states", "load_config"]
    bridge.states = {}
    asyncio.run(polls[0][1]())
    assert any("No resources available" in getattr(item, "text", "") for item in page.descendants())
    bridge.states = {"ur5e-2": {"current_state": "positioned", "held_part": CIRCULAR}}
    asyncio.run(polls[0][1]())
    assert bridge.reads == [
        "load_config",
        "get_runtime_recoveries",
        "get_robot_states",
        "load_config",
        "get_runtime_recoveries",
        "get_robot_states",
        "get_runtime_recoveries",
        "get_robot_states",
    ]
    assert any(getattr(item, "text", "") == CIRCULAR for item in page.descendants())
    page.delete()


def test_missing_scene_displays_an_error_without_live_or_dispatch_calls():
    from nicegui import ui

    class MissingBridge:
        def load_config(self, path):
            raise FileNotFoundError(path)

    with ui.column() as panel:
        refresh = render_nominal_resource_des(MissingBridge())
        asyncio.run(refresh())
    assert any(
        "Nominal resource DES unavailable" in getattr(item, "text", "")
        for item in panel.descendants()
    )
    panel.delete()


def test_live_status_cards_use_each_resource_domain_and_valuation(scene):
    from nicegui import ui

    from cais_spade_llm.resources.environment_models import build_environment_models
    from cais_spade_llm.ui.components.robot_status_card import render_robot_status_card

    for name, model in build_environment_models(scene).items():
        state = deepcopy(model["current_valuation"])
        state["current_state"] = "idle"
        domain = model["state_variables"].get("resource_state", {}).get("domain", [])
        if domain:
            state["resource_state"] = domain[-1]
        with ui.column() as panel:
            render_robot_status_card(name, state, model=model, evidence="configured initial assumptions")
        elements = list(panel.descendants())
        texts = [getattr(item, "text", "") for item in elements]
        badges = [item.text for item in elements if isinstance(item, ui.badge)]
        phases = [item.text for item in elements if isinstance(item, ui.label) and "rounded" in item._classes]
        highlighted = [item.text for item in elements if isinstance(item, ui.label) and "bg-blue-500" in item._classes]
        assert phases == domain, name
        assert badges == ([domain[-1]] if domain else []), name
        assert highlighted == ([domain[-1]] if domain else []), name
        assert "configured initial assumptions" in texts
        assert not {"execution_mode", "controller_ready", "gripper_state", "position"} & set(texts)
        rows = [row for item in elements if isinstance(item, ui.table) for row in item.rows]
        if name == "Storage":
            assert {"field": "inventory.KET4_Square_4mm", "value": "true"} in rows
            assert any(row["value"] == "false" for row in rows)
        if name == "Conveyor":
            assert "belt_stopped" in texts and "true" in texts
            assert {"part_name": SQUARE, "part_location": "null", "part_order": "null"} in rows
        if name == BUFFER:
            assert {"field": "zone_1_part", "value": "null"} in rows
        if name == "3D Printing Station":
            assert {"field": "output.gear_small", "value": "true"} in rows
        panel.delete()


def test_live_status_never_fills_missing_values_from_configured_state(models):
    from nicegui import ui

    from cais_spade_llm.ui.components.robot_status_card import render_robot_status_card
    from cais_spade_llm.ui.resource_status import SNAPSHOT_EVIDENCE

    with ui.column() as panel:
        render_robot_status_card("KMR", {"resource_state": "carrying"}, model=models["KMR"])
    texts = [getattr(item, "text", "") for item in panel.descendants()]
    assert "resource_location" in texts and "Unavailable" in texts
    assert "Storage" not in texts
    assert "null" not in texts
    assert SNAPSHOT_EVIDENCE in texts
    panel.delete()

    with ui.column() as panel:
        render_robot_status_card("legacy", {
            "current_state": "recovery_required", "held_part": None,
            "controller_ready": False, "position": {"x": 0},
        })
    texts = [getattr(item, "text", "") for item in panel.descendants()]
    assert {"recovery_required", "null", "false", '{"x": 0}'} <= set(texts)
    assert not {"idle", "at_pick", "picked", "positioned", "placed", "gripper_state"} & set(texts)
    panel.delete()

    with ui.column() as panel:
        render_robot_status_card("legacy", {})
    texts = [getattr(item, "text", "") for item in panel.descendants()]
    assert "Unavailable" in texts and "idle" not in texts
    panel.delete()


def test_live_status_reader_prefers_runtime_values_and_clears_old_resources(scene):
    from cais_spade_llm.resources.environment_models import build_environment_models
    from cais_spade_llm.ui.bridge import SystemBridge
    from cais_spade_llm.ui.resource_status import ResourceStatusReader, SNAPSHOT_EVIDENCE

    bridge = Mock(spec=SystemBridge)
    models = build_environment_models(scene)
    models["M1"].update(state_evidence="acknowledged controller completion")
    models["M1"]["current_valuation"].update(resource_state="loaded", part_name=SQUARE)
    bridge.get_robot_states.return_value = {"M1": {"current_state": "idle", "part_name": None}}
    bridge.get_environment_capabilities_revision.return_value = ("run", 1)
    bridge.get_environment_capabilities.return_value = {"models": models, "outcome": {"status": "planned"}}
    reader = ResourceStatusReader(bridge)
    display = reader.read()
    assert list(display["resources"]) == ["M1"]
    assert display["resources"]["M1"]["state"]["part_name"] == SQUARE
    assert display["resources"]["M1"]["evidence"] == "acknowledged controller completion"
    reader.read()
    assert bridge.get_environment_capabilities.call_count == 1
    assert bridge.get_robot_states.call_count == 2
    bridge.load_config.assert_not_called()

    bridge.get_environment_capabilities_revision.return_value = ("run", 2)
    models["M1"]["current_valuation"]["resource_state"] = "completed"
    assert reader.read()["resources"]["M1"]["state"]["resource_state"] == "completed"
    assert bridge.get_environment_capabilities.call_count == 2

    bridge.get_robot_states.return_value = {}
    bridge.get_environment_capabilities_revision.return_value = None
    bridge.get_environment_capabilities.return_value = {}
    assert reader.read() == {"resources": {}, "outcome": {}, "error": ""}

    bridge.load_config.return_value = scene
    bridge.get_robot_states.return_value = {"KMR": {"current_state": "carrying"}}
    display = reader.read()
    assert list(display["resources"]) == ["KMR"]
    assert display["resources"]["KMR"]["state"] == {"current_state": "carrying"}
    assert display["resources"]["KMR"]["evidence"] == SNAPSHOT_EVIDENCE
    bridge.get_robot_states.return_value["KMR"]["held_part"] = SQUARE
    assert reader.read()["resources"]["KMR"]["state"]["held_part"] == SQUARE
    bridge.load_config.assert_called_once()
    failure = 'KMR stopping footprint intersects an obstacle or map is unavailable'
    bridge.get_robot_states.return_value["KMR"].update(
        state_evidence='Last Gazebo acknowledgement: pick_part; revision 1.',
        execution_outcome={'status': 'failed:gazebo', 'details': {'content': failure}},
    )
    display = reader.read()
    assert display['outcome']['reason'] == failure
    assert display['resources']['KMR']['evidence'].startswith('Last Gazebo acknowledgement:')
    from nicegui import ui
    from cais_spade_llm.ui.components.robot_status_card import render_environment_outcome
    with ui.column() as panel:
        render_environment_outcome(display['outcome'])
    assert failure in [getattr(item, 'text', '') for item in panel.descendants()]
    panel.delete()
    bridge.get_robot_states.return_value = {}
    assert reader.read()['outcome'] == {}
    bridge.start_system.assert_not_called()
    bridge.ros2_start.assert_not_called()


def test_live_status_configuration_cache_invalidates_and_reports_missing_scene(scene, tmp_path, monkeypatch):
    from cais_spade_llm.ui import recovery_setup
    from cais_spade_llm.ui.resource_status import ResourceStatusReader

    path = tmp_path / "scene.json"
    path.write_text(json.dumps(scene))
    monkeypatch.setattr(recovery_setup, "load_setup", lambda: {"scene_file": str(path)})

    class ReadOnlyBridge:
        get_robot_states = Mock(return_value={"Storage": {"current_state": "idle"}})
        load_config = Mock(side_effect=lambda filename: json.loads(Path(filename).read_text()))

    bridge = ReadOnlyBridge()
    reader = ResourceStatusReader(bridge)
    assert reader.read()["resources"]["Storage"]["model"] is not None
    reader.read()
    assert bridge.load_config.call_count == 1
    path.write_text(path.read_text() + "\n")
    reader.read()
    assert bridge.load_config.call_count == 2
    path.unlink()
    display = reader.read()
    assert "Resource descriptors unavailable" in display["error"]
    assert display["resources"]["Storage"]["model"] is None
    assert display["resources"]["Storage"]["state"] == {"current_state": "idle"}



def test_resource_refresh_uses_revisions_and_preserves_expanded_controls(scene, monkeypatch):
    from nicegui import context, core, ui
    from nicegui.client import Client
    from cais_spade_llm.resources.environment_models import build_environment_models
    from cais_spade_llm.ui.components import nominal_resource_des as component

    models = build_environment_models(scene)
    requirements = {
        SQUARE: [{"processesToComplete": [{"process": "trim", "result": "square"}]}],
        "gear_small": [
            {"processesToComplete": [{"process": "print_part"}]},
            {"processesToComplete": [{"process": "assembly", "target": "Gear_Plate/Gear_Shaft_1"}]},
        ],
    }
    initial_products = {
        SQUARE: {"processCompleted": []},
        "gear_small": {"processCompleted": [{"process": "print_part"}]},
    }
    state = {
        "revision": 1,
        "snapshot": {
            "models": models,
            "environment_model": {},
            "outcome": {"status": "prepared"},
            "run_id": "configured-run",
            "processPlan": {},
            "requirements": requirements,
            "initial_product_states": initial_products,
            "product_states": deepcopy(initial_products),
        },
    }
    bridge = Mock()
    bridge.load_config.return_value = scene
    bridge.get_environment_capabilities_revision.side_effect = lambda: state["revision"]
    bridge.get_environment_capabilities.side_effect = lambda: deepcopy(state["snapshot"])
    export = Mock(wraps=component.process_json)
    monkeypatch.setattr(component, "process_json", export)
    client = Client(context.client.page)

    async def check():
        monkeypatch.setattr(core, "loop", asyncio.get_running_loop())
        with client:
            refresh = component.render_nominal_resource_des(bridge)
            await refresh()
            export.assert_not_called()
            resource = next(e for e in client.elements.values()
                            if isinstance(e, ui.select) and e.label == "Resource")
            part = next(e for e in client.elements.values()
                        if isinstance(e, ui.select) and e.label == "Part")
            assert resource.value == "Conveyor" and part.value == SQUARE
            assert len([e for e in client.elements.values() if isinstance(e, ui.mermaid)]) == 2
            initial_conditions = next(e for e in client.elements.values()
                                      if isinstance(e, ui.expansion)
                                      and e.text == "Configured initial values and marked state conditions")
            initial_conditions.set_value(True)
            await asyncio.sleep(0)
            initial_data = json.loads(next(e.content for e in initial_conditions.descendants()
                                           if isinstance(e, ui.code)))
            assert initial_data["marked_state_conditions"] == models["Conveyor"]["marked_state_conditions"]
            assert "part_location.KET4_Square_4mm" in initial_data["current_valuation"]

            des = next(e for e in client.elements.values()
                       if isinstance(e, ui.expansion) and e.text == "DES details")
            des.set_value(True)
            event = next(e for e in client.elements.values()
                         if isinstance(e, ui.select) and e.label == "Nominal event")
            event.set_value(event.options[-1])
            process = next(e for e in client.elements.values()
                           if isinstance(e, ui.expansion) and e.text == "Complete process JSON")
            process.set_value(True)
            export.assert_called()
            details = next(e for e in client.elements.values()
                           if isinstance(e, ui.expansion)
                           and e.text == "Shared event details")
            assert not any(isinstance(item, ui.mermaid) for item in details.descendants())
            details.set_value(True)
            await asyncio.sleep(0)
            assert not any(isinstance(item, ui.mermaid) for item in details.descendants())
            detail_codes = [json.loads(e.content) for e in details.descendants() if isinstance(e, ui.code)]
            assert any(isinstance(value, list)
                       and value == nominal_composition_event_rows(models, "Conveyor")
                       for value in detail_codes)

            part.set_value("gear_small")
            await asyncio.sleep(0)
            product_table = next(e for e in client.elements.values() if isinstance(e, ui.table)
                                 and any(column["field"] == "completed" for column in e.columns))
            assert any("print_part" in row["completed"] for row in product_table.rows)
            assert len([e for e in client.elements.values() if isinstance(e, ui.mermaid)]) == 2
            variants = next(e for e in client.elements.values() if isinstance(e, ui.expansion)
                            and e.text == "Declared event variants for these process steps")
            assert not any(isinstance(e, ui.code) for e in variants.descendants())
            variants.set_value(True)
            await asyncio.sleep(0)
            assert any(isinstance(e, ui.code) for e in variants.descendants())

            local_content = nominal_resource_diagram(models["Conveyor"], models)["mermaid"]
            compact = [e for e in client.elements.values() if isinstance(e, ui.mermaid)
                       and e.content == local_content and e not in details.descendants()]
            assert len(compact) == 1
            compact_ids = [e.id for e in compact]
            product_id = product_table.id
            elements = set(client.elements)
            for _ in range(3):
                await refresh()
            assert set(client.elements) == elements
            assert bridge.get_environment_capabilities.call_count == 1

            state["revision"] = 2
            state["snapshot"]["outcome"] = {"status": "needs_context", "reason": "Tool evidence missing"}
            state["snapshot"]["product_states"]["gear_small"]["processCompleted"] = [
                {"process": "print_part"},
                {"process": "assembly", "target": "Gear_Plate/Gear_Shaft_1"}
            ]
            await refresh()
            assert bridge.get_environment_capabilities.call_count == 2
            assert resource.value == "Conveyor" and part.value == "gear_small"
            assert des.value and process.value and details.value and variants.value
            assert event.value == event.options[-1]
            assert set(client.elements) == elements
            assert all(eid in client.elements for eid in compact_ids + [product_id])
            assert any("Tool evidence missing" in getattr(e, "text", "")
                       for e in client.elements.values())
            assert any("Current processCompleted" in getattr(e, "text", "")
                       and "assembly" in e.text for e in client.elements.values())

            state["revision"] = 3
            peer = next(e for e in models["ur5e-1"]["events"]
                        if e["event_name"] == "place_release" and "Conveyor" in e["participants"])
            original_guard = deepcopy(peer["guards"]["resource_state"])
            peer["guards"]["resource_state"] = {"equals": "idle"}
            await refresh()
            assert all(eid in client.elements for eid in compact_ids + [product_id])
            assert any("disagrees with its in state" in getattr(e, "text", "")
                       for e in client.elements.values())
            peer["guards"]["resource_state"] = original_guard

            state["revision"] = 4
            models["Conveyor"]["events"] = [
                e for e in models["Conveyor"]["events"] if e["event_name"] != "advance_conveyor"
            ]
            await refresh()
            assert product_id in client.elements
            local_content = nominal_resource_diagram(models["Conveyor"], models)["mermaid"]
            assert nominal_function_graph(models["Conveyor"], models) == {
                "nodes": [], "edges": [],
            }
            assert not any(eid in client.elements for eid in compact_ids)
            replacement_graph = next(e for e in client.elements.values()
                                     if isinstance(e, ui.mermaid))
            assert replacement_graph.content == local_content
            assert "place_release" in replacement_graph.content
            assert "advance_conveyor" not in replacement_graph.content

            resource.set_value("ur5e-1")
            await asyncio.sleep(0)
            robot_graph = next(e for e in client.elements.values() if isinstance(e, ui.mermaid))
            configured_graph = robot_graph.content
            assert "pick_approach" in configured_graph and "place_release" in configured_graph
            assert "event_id=" not in configured_graph
            state["revision"] = 5
            models["ur5e-1"]["current_valuation"]["resource_state"] = "picked"
            await refresh()
            assert robot_graph.content == configured_graph
            resource.set_value("Conveyor")
            await asyncio.sleep(0)
            assert part.value == "gear_small" or any(
                isinstance(e, ui.select) and e.label == "Part" and e.value == "gear_small"
                for e in client.elements.values()
            )
            assert any(isinstance(e, ui.expansion)
                       and e.text == "Configured initial values and marked state conditions"
                       and e.value for e in client.elements.values())
            assert any(isinstance(e, ui.expansion)
                       and e.text == "Shared event details"
                       and e.value for e in client.elements.values())
            assert any(isinstance(e, ui.expansion)
                       and e.text == "DES details" and e.value
                       for e in client.elements.values())
            assert any(isinstance(e, ui.expansion)
                       and e.text == "Declared event variants for these process steps"
                       and e.value for e in client.elements.values())

    asyncio.run(check())
    client.delete()


def test_product_state_ui_refreshes_capabilities_and_open_configuration_details(scene, monkeypatch):
    from nicegui import context, core, ui
    from nicegui.client import Client
    from cais_spade_llm.resources.environment_models import build_environment_models

    models = build_environment_models(scene)
    state = {"revision": 0, "snapshot": {"models": models}}
    bridge = Mock()
    bridge.load_config.return_value = scene
    bridge.get_environment_capabilities_revision.side_effect = lambda: state["revision"]
    bridge.get_environment_capabilities.side_effect = lambda: deepcopy(state["snapshot"])
    client = Client(context.client.page)

    async def check():
        monkeypatch.setattr(core, "loop", asyncio.get_running_loop())
        with client:
            refresh = render_nominal_resource_des(bridge)
            await refresh()
            resource = next(e for e in client.elements.values()
                            if isinstance(e, ui.select) and e.label == "Resource")
            resource.set_value("M1")
            await asyncio.sleep(0)
            product_graph = next(e for e in client.elements.values()
                                 if isinstance(e, ui.mermaid) and "Part at M1" in e.content)
            assert "result = square" in product_graph.content and "result = circle" in product_graph.content
            assert any(getattr(e, "text", "") == "No active processPlan."
                       for e in client.elements.values())
            details = next(e for e in client.elements.values()
                           if isinstance(e, ui.expansion) and e.text == "Product state and event details")
            details.set_value(True)
            await asyncio.sleep(0)

            def detail_data():
                return json.loads(next(e.content for e in details.descendants()
                                       if isinstance(e, ui.code)))

            assert detail_data()["current_configuration"]["program"]["effects"] == [
                {"process": "trim", "result": "square"}
            ]
            state["revision"] += 1
            models["M1"]["current_configuration"]["program"]["effects"] = [
                {"process": "trim", "result": "circle"}
            ]
            await refresh()
            assert product_graph.id in client.elements
            assert detail_data()["current_configuration"]["program"]["effects"] == [
                {"process": "trim", "result": "circle"}
            ]

            state["revision"] += 1
            models["M1"]["process_capabilities"]["trim"]["supported_results"] = ["circle"]
            await refresh()
            assert details.value and resource.value == "M1"
            assert product_graph.id not in client.elements
            replacement = next(e for e in client.elements.values()
                               if isinstance(e, ui.mermaid) and "Part at M1" in e.content)
            assert "result = circle" in replacement.content
            assert "result = square" not in replacement.content
            data = detail_data()
            assert len(data["transitions"]) == 1
            assert data["transitions"][0]["parameters"] == {"result": "circle"}
            assert data["states"] == nominal_resource_product_diagram(models["M1"], models)["nodes"]

            state["revision"] += 1
            models["M1"]["current_valuation"]["resource_state"] = "loaded"
            await refresh()
            assert replacement.id in client.elements
            bridge.save_config.assert_not_called()

    try:
        asyncio.run(check())
    finally:
        client.delete()


def test_gazebo_capability_functions_cover_owned_events_and_passive_resources(scene):
    from cais_spade_llm.resources.environment_models import build_environment_models
    from cais_spade_llm.ui.components.resource_function_catalog import resource_function_rows

    models = build_environment_models(scene)
    assert len(models) == 12
    for resource_id, model in models.items():
        view = resource_function_rows(model)
        owned = [
            event for event in model["events"]
            if event["parameter_bindings"]["resource_id"]["equals"] == resource_id
        ]
        assert sum(len(row["variants"]) for row in view["functions"]) == len(owned)
        for event in owned:
            if event["event_name"] == "print_part":
                assert event["program"]["steps"] == []
                assert event["program_status"] == "planned"
            else:
                assert event["program"]["steps"], (resource_id, event["event_name"])
            assert event["function_name"]
    assert not resource_function_rows(models["Storage"])["functions"]
    assert not resource_function_rows(models["Exit"])["functions"]
    assert resource_function_rows(models["Storage"])["participating"]
    assert resource_function_rows(models["Exit"])["participating"]

    robot_release = next(
        event for event in models["ur5e-1"]["events"]
        if event["event_name"] == "place_release"
    )
    kmr_release = next(
        event for event in models["KMR"]["events"]
        if event["event_name"] == "place_release"
    )
    assert robot_release["function_name"] == "place_insert"
    assert kmr_release["function_name"] == "place_release"
    assert robot_release["program"]["steps"] != kmr_release["program"]["steps"]
    expected_workflow_steps = {
        "M1": ("dwell",),
        "M2": ("dwell",),
        "Conveyor": ("move_relative",),
        BUFFER: ("move_relative",),
        "3D Printing Station": (),
    }
    for resource_id, expected_steps in expected_workflow_steps.items():
        functions = resource_function_rows(models[resource_id])["functions"]
        assert len(functions) == 1
        assert tuple(step["op"] for step in functions[0]["program"]["steps"]) == expected_steps
    assert len(resource_function_rows(models["Conveyor"])["functions"][0]["variants"]) == 2
    assert len(resource_function_rows(models[BUFFER])["functions"][0]["variants"]) == 3
    printer_availability = resource_function_rows(models["3D Printing Station"])[
        "functions"
    ][0]["availability"]
    assert printer_availability.startswith("Planned only — no Gazebo executor")
    assert "all supported printer outputs present" in printer_availability
    live_machine = deepcopy(models["M1"])
    live_machine["executable_tasks"] = ["machine_part"]
    assert resource_function_rows(live_machine)["functions"][0]["availability"] == "Gazebo executable"


def test_resource_catalog_renders_steps_and_current_generated_program(scene):
    from nicegui import context, ui
    from nicegui.client import Client
    from cais_spade_llm.resources.environment_models import build_environment_models
    from cais_spade_llm.ui.components.resource_function_catalog import (
        recovery_program_rows,
        render_generated_recovery_programs,
        render_resource_function_rows,
        resource_function_rows,
    )

    models = build_environment_models(scene)
    recovery = {
        "product_jid": "product@localhost",
        "product_name": "assembly_board-v1",
        "status": "llm_recovery",
        "recovery_approval_state": "primitive_pending",
        "recovery_debug": {
            "final_output": {
                "accepted_primitive_program": [
                    {
                        "resource_jid": "resource@localhost",
                        "event_name": "machine_part",
                        "primitive_steps": [{"primitive": "observe_workholding", "params": {}}],
                    }
                ]
            }
        },
    }
    agents = [{"jid": "resource@localhost", "name": "M1"}]
    rows = recovery_program_rows([recovery], agents)
    assert rows[0]["resource_name"] == "M1"
    assert rows[0]["approval_state"] == "primitive_pending"
    assert rows[0]["source"] == "final_output"
    pending = deepcopy(recovery)
    pending["recovery_debug"] = {
        "multi_turn_session": {
            "accepted_primitive_program": recovery["recovery_debug"]["final_output"]["accepted_primitive_program"]
        }
    }
    assert recovery_program_rows([pending], agents)[0]["source"] == "multi_turn_session"

    bridge = Mock()
    bridge.get_runtime_recoveries.return_value = [recovery]
    bridge.get_agent_statuses.return_value = agents
    client = Client(context.client.page)

    async def check():
        with client:
            render_resource_function_rows(resource_function_rows(models["M1"]))
            render_resource_function_rows(resource_function_rows(models["ur5e-1"]))
            refresh = render_generated_recovery_programs(bridge)
            await refresh()
        labels = [getattr(element, "text", "") for element in client.elements.values()]
        assert any("observe_workholding" in label for label in labels)
        assert any("Formal event: place_release" in label for label in labels)
        assert any("Formal event: machine_part" in label for label in labels)
        headings = [element._props.get("label") for element in client.elements.values()]
        assert "place_insert" in headings and "trim_part" in headings
        assert any("assembly_board-v1 · M1 · machine_part" in label for label in labels)
        assert any("Approval: primitive_pending" in label for label in labels)

    asyncio.run(check())
    client.delete()


def test_one_resource_diagram_hides_identities_and_preserves_guarded_events(scene):
    from cais_spade_llm.resources.environment_models import build_environment_models

    models = build_environment_models(scene)
    original = deepcopy(models)
    identities = [*scene["Storage"]["slots"], *scene["3D Printing Station"]["initial_products"]]
    for resource_id, model in models.items():
        diagram = nominal_resource_diagram(model, models)
        assert diagram["nodes"], resource_id
        assert not any(name in diagram["mermaid"] for name in identities)
        assert "staging_part" not in diagram["mermaid"]
        represented = {event_id for edge in diagram["edges"] for event_id in edge["event_ids"]}
        assert represented == {
            event["event_id"] for event in model["events"]
            if event["updates"] or event.get("collection_effects")
            or event["parameter_bindings"]["resource_id"] == {"equals": resource_id}
        }, resource_id
        assert "[guarded]" not in diagram["mermaid"]
        lines = diagram["mermaid"].splitlines()
        arrows = [line for line in lines if "-->" in line]
        assert [line for line in lines if "linkStyle" in line] == [
            f"    linkStyle {index} stroke-dasharray:5 5"
            for index, line in enumerate(arrows) if "(planned)" in line
        ]
    assert models == original
    for resource_id in ("M1", "M2"):
        diagram = nominal_resource_diagram(models[resource_id], models)
        states = {node["id"]: node["label"] for node in diagram["nodes"]}
        assert set(states.values()) == {"idle", "loaded", "completed"}
        assert states[diagram["initial_id"]] == "idle"
        assert {
            ("idle", "place_release", "loaded"),
            ("loaded", "machine_part", "completed"),
            ("completed", "pick_grasp", "idle"),
        } <= {(states[e["source"]], e["event_name"], states[e["target"]])
              for e in diagram["edges"]}
        staging_ids = {e["event_id"] for e in models[resource_id]["events"]
                       if "staging_part" in e["updates"]}
        for event_id in staging_ids:
            represented = [e for e in diagram["edges"] if event_id in e["event_ids"]]
            assert represented and all(e["source"] == e["target"] for e in represented)

    conveyor = nominal_resource_diagram(models["Conveyor"], models)
    assert all(field not in conveyor["mermaid"]
               for field in ("part_location", "part_order", "loading_reserved_by"))
    assert all("belt_stopped" in node["label"] for node in conveyor["nodes"])
    printer = nominal_resource_diagram(models["3D Printing Station"], models)
    states = {node["id"]: node["label"] for node in printer["nodes"]}
    assert states[printer["initial_id"]] == "completed"
    assert {(states[e["source"]], e["event_name"], states[e["target"]])
            for e in printer["edges"]} == {
        ("idle", "print_part", "completed"),
        ("completed", "pick_grasp", "completed"),
        ("completed", "pick_grasp", "idle"),
    }
    assert "print_part (planned)" in printer["mermaid"]


def test_buffer_graph_preserves_empty_downstream_and_pickup_conditions(scene):
    from cais_spade_llm.resources.environment_models import build_environment_models

    models = build_environment_models(scene)
    diagram = nominal_resource_diagram(models[BUFFER], models)
    states = {node["id"]: node["label"] for node in diagram["nodes"]}
    assert len(states) == 2 and set(states.values()) == {"empty", "occupied"}
    assert states[diagram["initial_id"]] == "empty"
    assert len(diagram["edges"]) == 5
    assert {(states[edge["source"]], edge["resource_id"], edge["event_name"], states[edge["target"]])
            for edge in diagram["edges"]} == {
        ("empty", "Conveyor", "advance_conveyor", "occupied"),
        ("occupied", "Conveyor", "advance_conveyor", "occupied"),
        ("occupied", BUFFER, "advance_part", "occupied"),
        ("occupied", "ur5e-3", "pick_grasp", "occupied"),
        ("occupied", "ur5e-3", "pick_grasp", "empty"),
    }
    assert sum(len(edge["variants"]) for edge in diagram["edges"]) == 28
    for edge in diagram["edges"]:
        assert set(edge["event_ids"]) == {variant["event_id"] for variant in edge["variants"]}
        for variant in edge["variants"]:
            event_id = variant["event_id"]
            before, after = variant["source_conditions"], variant["target_conditions"]
            event = next(e for e in models[BUFFER]["events"] if e["event_id"] == event_id)
            assert edge["resource_id"] == event["parameter_bindings"]["resource_id"]["equals"]
            assert edge["program_status"] == event["program_status"]
            assert states[edge["source"]] == (
                "empty" if all(rule == {"equals": None} for rule in before.values()) else "occupied"
            )
            assert states[edge["target"]] == (
                "empty" if all(rule == {"equals": None} for rule in after.values()) else "occupied"
            )
            for field, guard in event["guards"].items():
                assert before[field] == (
                    {"equals": None} if guard == {"equals": None} else {"not_equals": None}
                )
            for field, update in event["updates"].items():
                assert after[field] == (
                    {"equals": None} if update == {"set": None} else {"not_equals": None}
                )
            assert all(after[field] == before[field] for field in before
                       if field not in event["updates"])


def test_buffer_graph_initial_condition_uses_configured_zone_occupancy(scene):
    from cais_spade_llm.resources.environment_models import build_environment_models

    models = build_environment_models(scene)
    model = models[BUFFER]
    fields = list(model["state_variables"])
    parts = list(scene["Storage"]["slots"])[:len(fields)]
    model["current_valuation"] = dict(zip(fields, parts, strict=True))
    for occupied in product((False, True), repeat=len(fields)):
        initial = {field: part if present else None
                   for field, part, present in zip(fields, parts, occupied, strict=True)}
        diagram = nominal_resource_diagram(model, models, initial_valuation=initial)
        nodes = {node["label"]: node for node in diagram["nodes"]}
        expected = "occupied" if any(occupied) else "empty"
        assert nodes[expected]["id"] == diagram["initial_id"]
        assert nodes["empty"]["condition_match"] == "all"
        assert nodes["empty"]["conditions"] == {field: {"equals": None} for field in fields}
        assert nodes["occupied"]["condition_match"] == "any"
        assert nodes["occupied"]["conditions"] == {field: {"not_equals": None} for field in fields}
        assert not any(part in diagram["mermaid"] for part in parts)


def test_buffer_graph_follows_declared_transitions(scene):
    from cais_spade_llm.resources.environment_models import build_environment_models

    models = build_environment_models(scene)
    model = models[BUFFER]
    model["events"] = [event for event in model["events"] if event["event_name"] != "advance_part"]
    diagram = nominal_resource_diagram(model, models)
    assert len(diagram["nodes"]) == 2
    assert len(diagram["edges"]) == 4
    assert all(edge["event_name"] != "advance_part" for edge in diagram["edges"])
