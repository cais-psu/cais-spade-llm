"""Recovery-framework NIST catalog and small-gear slippage contracts."""

from __future__ import annotations

import asyncio
import copy
import json
import math
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, call

import pytest

from cais_spade_llm.product import profile
from cais_spade_llm.product.order import load_product_order_file, validate_product_order
from cais_spade_llm.product.profile import ProductProfile
from cais_spade_llm.ui.pages.products import (
    _known_nist_component_rows,
    _preferred_product_file,
    _validate_geometry_payload,
)

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = (
    ROOT / "cais_spade_llm/initialization/products/assembly_board-v1-recovery-framework.json"
)
GEOMETRY_PATH = (
    ROOT
    / "cais_spade_llm/specification/products/geometry/assembly_board-v1-recovery-framework.json"
)
ORDER_PATH = (
    ROOT / "cais_spade_llm/specification/products/orders/assembly_board-v1-recovery-framework.json"
)
LEGACY_GEOMETRY_PATH = ROOT / "test/fixtures/product_geometry/assembly_board-v1-historical.json"
CAD_DIR = ROOT / "ros2/cais_lab_robotics/cad_models"

COMPONENTS = [
    "gear_small",
    "gear_medium",
    "gear_large",
    "KET4_Square_4mm",
    "KET8_Square_8mm",
    "KET12_Square_12mm",
    "KET16_Square_16mm",
    "RGOCG4-50_Round_4mm",
    "RGOCG8-50_8mm",
    "RGOCG12-50_12mm",
    "RGOCG16-50_16mm",
]
FIXTURES = {
    "GMC_Laser_Plate_Virtual",
    "Gear_Plate",
    "Gear_Shaft_1",
    "Gear_Shaft_2",
    "Gear_Shaft_3",
}
CAD_FILENAMES = {
    "gear_small": "Gear_Small.STL",
    "gear_medium": "Gear_Medium.STL",
    "gear_large": "Gear_Large.STL",
    "KET4_Square_4mm": "KET4_Square_4mm.STL",
    "KET8_Square_8mm": "KET8_Square_8mm.STL",
    "KET12_Square_12mm": "KET12_Square_12mm.STL",
    "KET16_Square_16mm": "KET16_Square_16mm.STL",
    "RGOCG4-50_Round_4mm": "RGOCG4-50_Round_4mm.STL",
    "RGOCG8-50_8mm": "RGOCG8-50_8mm.STL",
    "RGOCG12-50_12mm": "RGOCG12-50_12mm.STL",
    "RGOCG16-50_16mm": "RGOCG16-50_16mm.STL",
}
TARGETS = {
    "gear_small": "Gear_Plate/Gear_Shaft_1",
    "gear_medium": "Gear_Plate/Gear_Shaft_2",
    "gear_large": "Gear_Plate/Gear_Shaft_3",
    "KET4_Square_4mm": "GMC_Laser_Plate_Virtual/KET4_Square_4mm",
    "KET8_Square_8mm": "GMC_Laser_Plate_Virtual/KET8_Square_8mm",
    "KET12_Square_12mm": "GMC_Laser_Plate_Virtual/KET12_Square_12mm",
    "KET16_Square_16mm": "GMC_Laser_Plate_Virtual/KET16_Square_16mm",
    "RGOCG4-50_Round_4mm": "GMC_Laser_Plate_Virtual/RGOCG4-50_Round_4mm",
    "RGOCG8-50_8mm": "GMC_Laser_Plate_Virtual/RGOCG8-50_8mm",
    "RGOCG12-50_12mm": "GMC_Laser_Plate_Virtual/RGOCG12-50_12mm",
    "RGOCG16-50_16mm": "GMC_Laser_Plate_Virtual/RGOCG16-50_16mm",
}


def _geometry_document() -> dict:
    return json.loads(GEOMETRY_PATH.read_text(encoding="utf-8"))


def _gazebo_geometry() -> dict:
    return _geometry_document()["gazebo"]


def test_recovery_manifest_keeps_one_internal_product_agent_symbol() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert list(manifest) == ["assembly_board-v1"]
    product = manifest["assembly_board-v1"]
    assert product["jid"] == "assembly_board-v1@localhost"
    assert product["product_geometry_file"] == str(GEOMETRY_PATH.relative_to(ROOT))
    assert product["product_order_file"] == str(ORDER_PATH.relative_to(ROOT))
    assert _preferred_product_file(["another_product.json", str(MANIFEST_PATH)]) == str(
        MANIFEST_PATH
    )


def test_catalog_contains_exactly_eleven_selectable_components() -> None:
    geometry_document = _geometry_document()
    geometry = geometry_document["gazebo"]
    slots = geometry["assembly_board"]["slots"]
    parts = geometry["parts"]

    assert _validate_geometry_payload(geometry_document) == ""
    assert list(slots) == COMPONENTS
    for map_name in (
        "model_map",
        "heights_m",
        "cad_filename_map",
        "initial_source_resource_map",
        "assembly_target_map",
    ):
        assert list(parts[map_name]) == COMPONENTS
    assert parts["model_map"] == {name: name for name in COMPONENTS}
    assert parts["cad_filename_map"] == CAD_FILENAMES
    assert not (set(slots) & FIXTURES)
    assert all((CAD_DIR / filename).is_file() for filename in CAD_FILENAMES.values())


def test_catalog_binds_exact_sources_and_targets() -> None:
    parts = _gazebo_geometry()["parts"]
    sources = parts["initial_source_resource_map"]
    assert {sources[name] for name in COMPONENTS[:3]} == {"3D Printing Station"}
    assert {sources[name] for name in COMPONENTS[3:]} == {"Storage"}
    assert parts["assembly_target_map"] == TARGETS

    rows = _known_nist_component_rows(_gazebo_geometry())
    assert [row["identifier"] for row in rows] == COMPONENTS
    assert rows[0] == {
        "identifier": "gear_small",
        "cad_filename": "Gear_Small.STL",
        "gazebo_model": "gear_small",
        "initial_source_resource": "3D Printing Station",
        "assembly_target": "Gear_Plate/Gear_Shaft_1",
    }


@pytest.mark.parametrize("steps", [
    [],
    [{"processesToComplete": []}],
    [{"processesToComplete": [{"process": "assembly"}], "locationsToComplete": []}],
    [{"processesToComplete": [{"process": "trim"}]},
     {"processesToComplete": [{"process": "assembly"}]}],
    [{"processesToComplete": [{"process": "assembly", "target": "Gear_Plate/Gear_Shaft_1"}]}],
    [{"processesToComplete": [{"process": "assembly"}, {"process": "assembly"}]}],
])
def test_process_plan_rejects_incomplete_or_mixed_contracts(steps) -> None:
    order = load_product_order_file(ORDER_PATH)
    order["parts"] = ["gear_small"]
    order["processPlan"]["gear_small"] = steps
    with pytest.raises(ValueError):
        validate_product_order(order, _gazebo_geometry(), require_process_requirements=True)


def test_process_plan_requires_geometry_and_preserves_legacy_orders() -> None:
    geometry = _gazebo_geometry()
    order = load_product_order_file(ORDER_PATH)
    order["parts"] = ["gear_small"]
    legacy = copy.deepcopy(order)
    legacy.pop("processPlan")
    legacy["requirements"] = {"gear_small": [
        {"process": "print_part"}, {"state": "assembled", "target": TARGETS["gear_small"]}
    ]}
    assert validate_product_order(legacy, geometry).payload == legacy
    order["requirements"] = legacy["requirements"]
    with pytest.raises(ValueError, match="not both"):
        validate_product_order(order, geometry)
    order.pop("requirements")
    del geometry["parts"]["assembly_target_map"]["gear_small"]
    with pytest.raises(ValueError, match="exact feature"):
        validate_product_order(order, geometry)


def test_all_and_subset_orders_round_trip_with_exact_identifiers(tmp_path: Path) -> None:
    geometry = _gazebo_geometry()
    all_order = load_product_order_file(ORDER_PATH)
    validated_all = validate_product_order(all_order, geometry)
    assert validated_all.payload["product"] == "assembly_board-v1"
    assert validated_all.payload["parts"] == "all"
    assert validated_all.selected_parts == COMPONENTS
    assert "requirements" not in all_order
    assert all_order["processPlan"]["KET4_Square_4mm"] == [
        {"processesToComplete": [{"process": "trim", "result": "square"}]},
        {"processesToComplete": [{"process": "assembly"}]},
    ]
    assert "locationsToComplete" not in json.dumps(all_order)

    for selected in (
        ["gear_small"],
        ["KET16_Square_16mm", "RGOCG4-50_Round_4mm", "gear_large"],
    ):
        payload = {**all_order, "parts": selected}
        saved = tmp_path / f"subset-{len(selected)}.json"
        saved.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        reloaded = load_product_order_file(saved)
        validated = validate_product_order(reloaded, geometry)
        assert validated.payload["parts"] == selected
        assert validated.selected_parts == selected
        for identifier in selected:
            binding = ProductProfile.geometry_for_part_from_geometry(identifier, geometry)
            assert binding["part_name"] == identifier
            assert binding["model_name"] == identifier
            assert binding["slot_xy"] == geometry["assembly_board"]["slots"][identifier]


def test_unknown_fixture_and_missing_geometry_are_rejected() -> None:
    geometry = _gazebo_geometry()
    base_order = load_product_order_file(ORDER_PATH)
    for unknown in ("unknown_component", "GMC_Laser_Plate_Virtual", "Gear_Shaft_1"):
        with pytest.raises(ValueError, match="unknown product order part"):
            validate_product_order({**base_order, "parts": [unknown]}, geometry)
    with pytest.raises(ValueError, match="has no assembly_board slots"):
        validate_product_order(base_order, {})


def test_recovery_assembly_targets_match_gravity_supported_fixture_surfaces() -> None:
    geometry = _gazebo_geometry()

    gear = ProductProfile.geometry_for_part_from_geometry("gear_small", geometry)
    machined = ProductProfile.geometry_for_part_from_geometry(
        "KET4_Square_4mm",
        geometry,
    )

    assert gear["slot_floor_z_m"] == pytest.approx(1.0289916)
    assert gear["target_origin_pose"]["z"] == pytest.approx(1.0389916)
    assert gear["place_tool_yaw_offset_rad"] == pytest.approx(math.pi / 2)
    assert machined["slot_floor_z_m"] == pytest.approx(1.0239916)
    assert machined["target_origin_pose"]["z"] == pytest.approx(1.0239916)


def test_incomplete_known_nist_maps_are_rejected() -> None:
    document = _geometry_document()
    for map_name in (
        "cad_filename_map",
        "initial_source_resource_map",
        "assembly_target_map",
    ):
        incomplete = copy.deepcopy(document)
        incomplete["gazebo"]["parts"][map_name].pop("gear_small")
        assert "must contain exactly the assembly_board slots" in _validate_geometry_payload(
            incomplete
        )


def test_historical_recovery_fixture_retains_its_original_component_symbols() -> None:
    legacy_geometry = json.loads(LEGACY_GEOMETRY_PATH.read_text(encoding="utf-8"))
    runtime_context = json.loads(
        (ROOT / "test/fixtures/part_slippage/runtime_context.json").read_text(encoding="utf-8")
    )
    assert ROOT / runtime_context["product_geometry"] == GEOMETRY_PATH
    assert set(runtime_context["part_tracker"]) == {"KET4_Square_4mm", "gear_large"}
    assert legacy_geometry["gazebo"]["parts"]["model_map"] == {
        "SG": "gear_small",
        "MG": "gear_medium",
        "LG": "gear_large",
        "SRP": "rect_pin_small",
        "MRP": "rect_pin_medium",
        "LRP": "rect_pin_large",
        "SCP": "circ_pin_small",
        "MCP": "circ_pin_medium",
        "LCP": "circ_pin_large",
    }


def test_active_product_files_contain_only_the_configured_nist_setup() -> None:
    for active_path in (MANIFEST_PATH, GEOMETRY_PATH):
        assert sorted(active_path.parent.glob("*.json")) == [active_path]
    assert {path.name for path in ORDER_PATH.parent.glob("*.json")} == {
        ORDER_PATH.name, "assembly_board-v1-kmr-storage-m1.json",
        "assembly_board-v1-eight-pegs.json", "assembly_board-v1-round-4mm-m1.json",
        "assembly_board-v1-two-parts.json"
    }


def test_product_geometry_resolves_the_exact_identifier_from_its_manifest() -> None:
    profile._load_product_meta.cache_clear()
    profile._load_geometry_doc_for_destination.cache_clear()
    assert profile._product_geometry_path_for_token("assembly_board-v1") == GEOMETRY_PATH
    assert profile._load_product_meta("assembly_board-v1-recovery-framework") == {}
    for part_name in COMPONENTS:
        resolved = ProductProfile.resolve_place_geometry(
            part_name=part_name,
            destination_location="assembly_board-v1",
            execution_mode="simulation",
        )
        assert resolved["model_name"] == part_name
        assert resolved["slot_xy"] == _gazebo_geometry()["assembly_board"]["slots"][part_name]
    assert (
        ProductProfile.resolve_place_geometry(
            part_name="MG", destination_location="assembly_board-v1", execution_mode="simulation"
        )
        == {}
    )


def test_product_manifest_lookup_rejects_ambiguous_identifiers(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(profile, "_PACKAGE_ROOT", tmp_path)
    manifests = tmp_path / "initialization/products"
    manifests.mkdir(parents=True)
    meta = {"product_geometry_file": str(GEOMETRY_PATH)}
    first = manifests / "different_filename.json"
    first.write_text(json.dumps({"assembly_board-v1": meta}))
    profile._load_product_meta.cache_clear()
    try:
        assert profile._load_product_meta("assembly_board-v1") == meta
        assert profile._load_product_meta("different_filename") == {}
        (manifests / "duplicate.json").write_text(first.read_text())
        profile._load_product_meta.cache_clear()
        assert profile._load_product_meta("assembly_board-v1") == {}
    finally:
        profile._load_product_meta.cache_clear()


def test_nist_catalog_matches_configured_scene_inventory_and_meshes() -> None:
    scene = json.loads(
        (ROOT / "cais_spade_llm/initialization/recovery_framework_gazebo.json").read_text()
    )
    world = ET.parse(ROOT / "ros2/cais_lab_robotics/worlds/table_recovery_framework.world")
    pegs = list(scene["Storage"]["slots"])
    gears = scene["3D Printing Station"]["supported_products"]
    assert set(COMPONENTS) == set(pegs + gears)
    for part_name in pegs:
        model = world.find(f"./world/model[@name='{part_name}']")
        assert model is not None
        assert model.findtext("link/visual/geometry/mesh/uri") == (
            f"model://cad_models/{CAD_FILENAMES[part_name]}"
        )
    for part_name in gears:
        model = world.find(f"./world/model[@name='{part_name}']")
        assert model is not None
        collision = model.find("link/collision")
        visual = model.find("link/visual")
        assert collision.findtext("geometry/mesh/uri") == visual.findtext("geometry/mesh/uri")
        assert collision.findtext("pose") == visual.findtext("pose")


def test_stationary_registration_does_not_reuse_removed_geometry() -> None:
    from cais_spade_llm.ui.perception_manager import PerceptionManager

    result = PerceptionManager.stationary_inspection_configuration(
        SimpleNamespace(project_root=ROOT)
    )
    assert Path(result["geometry_path"]) == GEOMETRY_PATH
    assert result["configured"] is False


def test_two_part_order_preserves_full_order_requirements_and_default_selection():
    order = load_product_order_file(ORDER_PATH.with_name('assembly_board-v1-two-parts.json'))
    checked = validate_product_order(order, _gazebo_geometry(), require_process_requirements=True)
    assert checked.selected_parts == ['KET4_Square_4mm', 'gear_small']
    full = load_product_order_file(ORDER_PATH)
    assert order['processPlan'] == {part: full['processPlan'][part] for part in checked.selected_parts}
    assert 'machine_resource' not in order
    from cais_spade_llm.ui.recovery_setup import default_setup

    assert default_setup()['selected_product_order_file'] == str(ORDER_PATH.relative_to(ROOT))



def _record_small_gear_slippage_pickups(runtime) -> None:
    context = runtime.context
    context.part_tracker["KET4_Square_4mm"]["processCompleted"] = [
        {"process": "trim", "result": "square"},
    ]
    runtime.retained_paths = {
        part: {"path": [
            {"resource_id": owner, "event_name": event,
             "parameters": {"part_name": part, "destination_location": "assembly_board-v1"}}
            for event in ("place_approach", "place_insert")
        ]}
        for owner, part in (("ur5e-4", "gear_small"), ("ur5e-3", "KET4_Square_4mm"))
    }
    for owner, part in (("ur5e-4", "gear_small"), ("ur5e-3", "KET4_Square_4mm")):
        context.resources[owner].valuation.update(resource_state="picked", held_part=part)
        context.part_tracker[part].update(state="in_gripper", location=owner)
        agent = next(agent for agent in runtime.resource_agents if agent.agent_name == owner)
        agent._held_part = part
        agent._gripper_state = "closed"
        context.transitions.append({
            "acknowledgement": {
                "resource_id": owner,
                "event_name": "pick_grasp",
                "task_id": owner + "_pickup",
                "run_id": context.run_id,
                "evidence": "resource",
                "parameters": {"part_name": part},
            },
            "observations": {"controller_result": {"status": "completed", "gripper_state": "closed"}},
        })


@pytest.fixture
def small_gear_slippage_runtime(tmp_path: Path, monkeypatch):
    """Build real two-part runtimes with every ROS effect and report isolated."""
    from cais_spade_llm.recovery_framework import environment_runtime, workflow_execution
    from cais_spade_llm.ui import recovery_setup as settings

    monkeypatch.setattr(environment_runtime, "RUN_DIRECTORY", tmp_path / "environment_runs")
    monkeypatch.setattr(
        workflow_execution, "GazeboWorker",
        lambda **kwargs: SimpleNamespace(cancel=AsyncMock(), last_result=None),
    )
    monkeypatch.setattr(
        "cais_spade_llm.recovery_framework.failure_effects._observe_part",
        Mock(return_value={"x": 0.01, "y": 0.2, "z": 1.035,
                           "qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0}),
    )
    monkeypatch.setattr(
        "cais_spade_llm.recovery_framework.conveyor_fault.marker",
        Mock(return_value={"status": "completed"}),
    )
    setup = settings.default_setup()
    setup["failure_scenario"] = None
    setup["selected_product_order_file"] = str(
        ORDER_PATH.with_name("assembly_board-v1-two-parts.json").relative_to(ROOT)
    )
    inputs = settings.validate_setup(setup)
    failure = settings.slippage_example(inputs["models"], "ur5e-4", "gear_small")
    failure.update(
        checkpoint="after_both_pickups_before_place",
        drop_pose={"x": 0.0, "y": 0.2, "z": 1.04},
        additional_condition={"resource_id": "ur5e-3", "part_name": "KET4_Square_4mm"},
    )
    setup["failure_scenario"] = failure
    inputs = settings.validate_setup(setup)

    def create(*, picked: bool = True, checkpoint: str = "after_both_pickups_before_place",
               slipping_resource: str = "ur5e-4"):
        run_setup = copy.deepcopy(setup)
        run_setup["failure_scenario"]["checkpoint"] = checkpoint
        if slipping_resource == "ur5e-3":
            alternate = settings.slippage_example(inputs["models"], "ur5e-3", "KET4_Square_4mm")
            alternate.update(checkpoint=checkpoint, drop_pose={"x": 0., "y": -.2, "z": 1.04},
                             additional_condition={"resource_id": "ur5e-4", "part_name": "gear_small"})
            run_setup["failure_scenario"] = alternate
        resources = []
        for rid in inputs["models"]:
            controller = None
            if rid in {"ur5e-3", "ur5e-4"}:
                controller = SimpleNamespace(
                    get_current_pose=Mock(return_value={
                        "success": True, "pose": {"x": 0.0, "y": 0.0, "z": 1.2},
                    }),
                    open_gripper=Mock(return_value=True),
                    detach_part=Mock(return_value={"success": True, "release_mode": "detached"}),
                    set_entity_pose=Mock(return_value={"success": True}),
                    _sync_part_collision=Mock(return_value=True),
                    _last_command_evidence={"collision_scene_acknowledged": True},
                    _cancel_simulation_goal=Mock(),
                )
            resources.append(SimpleNamespace(
                agent_name=rid, jid=rid + "@localhost", execution_mode="simulation",
                _controller=controller, _held_part=None, _gripper_state="open",
            ))
        runtime = environment_runtime.EnvironmentRuntime(
            SimpleNamespace(jid="assembly_board-v1@localhost"),
            {"setup": run_setup, "inputs": {key: inputs[key] for key in ("scene", "product_order", "geometry")}},
            resources,
        )
        runtime.queue_save = Mock()
        runtime.retained_paths = {}
        for owner in ("ur5e-3", "ur5e-4"):
            runtime.context.resources[owner].executors["place_insert"] = Mock()
        if picked:
            _record_small_gear_slippage_pickups(runtime)
        return runtime

    return create


@pytest.mark.parametrize("first", ["ur5e-3", "ur5e-4"])
def test_small_gear_slippage_holds_each_pickup_until_both_are_observed(
    small_gear_slippage_runtime, first,
) -> None:
    """Either pickup order must hold the first robot before placement can start."""
    runtime = small_gear_slippage_runtime()
    context, fault = runtime.context, runtime.conveyor_fault
    pickups = {record["acknowledgement"]["resource_id"]: record for record in context.transitions}
    context.transitions.clear()
    assert fault.checkpoint() is None
    assert not fault.holds_task({"resource_id": first, "event_name": "place_approach"})
    context.transitions.append(pickups[first])
    second = "ur5e-4" if first == "ur5e-3" else "ur5e-3"
    assert fault.holds_task({"resource_id": first, "event_name": "place_approach"})
    assert not fault.holds_task({"resource_id": second, "event_name": "place_approach"})
    assert not asyncio.run(fault.after_acknowledgement())
    assert not runtime.stopped
    context.transitions.append(pickups[second])
    checkpoint = fault.checkpoint()
    assert checkpoint["part_name"] == "gear_small"
    assert checkpoint["custodian"] == "ur5e-4"
    assert checkpoint["other_custodian"] == "ur5e-3"
    assert checkpoint["other_part_name"] == "KET4_Square_4mm"
    assert set(checkpoint["pickups"]) == {"ur5e-3", "ur5e-4"}
    assert fault.holds_task({"resource_id": second, "event_name": "place_insert"})
    assert not fault.holds_task({"resource_id": "ur5e-1", "event_name": "place_approach"})


@pytest.mark.parametrize("owner", ["ur5e-3", "ur5e-4"])
@pytest.mark.parametrize("invalid", [
    "missing_pickup", "missing_observation", "stale_run", "simulated_evidence",
    "wrong_part", "later_release", "wrong_location", "empty_gripper", "idle_robot", "pending_task",
])
def test_small_gear_slippage_rejects_invalid_pickup_evidence_without_effects(
    small_gear_slippage_runtime, owner, invalid,
) -> None:
    """Both exact custodians need current observations and no outstanding robot task."""
    runtime = small_gear_slippage_runtime()
    context, fault = runtime.context, runtime.conveyor_fault
    record = next(row for row in context.transitions if row["acknowledgement"]["resource_id"] == owner)
    task = record["acknowledgement"]
    part = task["parameters"]["part_name"]
    if invalid == "missing_pickup":
        context.transitions.remove(record)
    elif invalid == "missing_observation":
        record["observations"] = {}
    elif invalid == "stale_run":
        task["run_id"] = "previous_run"
    elif invalid == "simulated_evidence":
        task["evidence"] = "simulated"
    elif invalid == "wrong_part":
        task["parameters"]["part_name"] = "gear_large"
    elif invalid == "later_release":
        later = copy.deepcopy(record)
        later["acknowledgement"].update(event_name="place_release", task_id=owner + "_release")
        context.transitions.append(later)
    elif invalid == "wrong_location":
        context.part_tracker[part]["location"] = "assembly_board-v1"
    elif invalid == "empty_gripper":
        context.resources[owner].valuation["held_part"] = None
    elif invalid == "idle_robot":
        context.resources[owner].valuation["resource_state"] = "idle"
    else:
        context.pending_tasks["racing_place"] = {"resource_id": owner}
    before = context.snapshot()
    parts_before = copy.deepcopy(context.part_tracker)
    assert fault.checkpoint() is None
    assert not asyncio.run(fault.after_acknowledgement())
    with pytest.raises(ValueError, match="Waiting"):
        asyncio.run(fault.trigger())
    assert context.snapshot() == before
    assert context.part_tracker == parts_before
    assert fault.status == "armed" and not runtime.stopped
    assert not context.unavailable_resources
    for agent in runtime.resource_agents:
        if agent._controller is not None:
            for method in ("open_gripper", "detach_part", "set_entity_pose", "_sync_part_collision"):
                getattr(agent._controller, method).assert_not_called()


def test_small_gear_slippage_detaches_only_gear_and_retains_interrupted_obligations(
    small_gear_slippage_runtime,
) -> None:
    """A confirmed gear drop preserves the held peg and both unfinished assemblies."""
    runtime = small_gear_slippage_runtime()
    context, fault = runtime.context, runtime.conveyor_fault
    before, parts_before = context.snapshot(), copy.deepcopy(context.part_tracker)
    requirements = copy.deepcopy(context.requirements)
    continuations = copy.deepcopy(runtime.retained_paths)
    pending = {"task_id": "interrupted_return", "run_id": context.run_id,
               "resource_id": "KMR", "event_name": "move_to_resource",
               "parameters": {"source_resource": "M1", "target_resource": "Storage"},
               "reservations": ["resource:KMR"]}
    context.pending_tasks[pending["task_id"]] = copy.deepcopy(pending)
    context.reservations["resource:KMR"] = pending["task_id"]
    assert fault.checkpoint() is not None
    assert asyncio.run(fault.after_acknowledgement())
    gear = next(agent for agent in runtime.resource_agents if agent.agent_name == "ur5e-4")
    peg = next(agent for agent in runtime.resource_agents if agent.agent_name == "ur5e-3")
    gear._controller.open_gripper.assert_called_once_with()
    gear._controller.detach_part.assert_called_once_with("gear_small", assume_released_if_open=False)
    gear._controller.set_entity_pose.assert_called_once_with(
        "gear_small", **fault.configuration["drop_pose"], **fault.configuration["orientation_quat"],
    )
    gear._controller._sync_part_collision.assert_called_once_with("gear_small")
    for method in ("open_gripper", "detach_part", "set_entity_pose", "_sync_part_collision"):
        getattr(peg._controller, method).assert_not_called()
    assert gear._held_part is None and gear._gripper_state == "open"
    assert peg._held_part == "KET4_Square_4mm" and peg._gripper_state == "closed"
    assert context.part_tracker["gear_small"] == {
        **parts_before["gear_small"], "state": "misplaced", "location": None,
        "observed_pose": fault.evidence["observed_drop_pose"],
    }
    assert {part: state for part, state in context.part_tracker.items() if part != "gear_small"} == {
        part: state for part, state in parts_before.items() if part != "gear_small"
    }
    assert context.resources["ur5e-4"].valuation["held_part"] is None
    assert context.resources["ur5e-3"].valuation == before["ur5e-3"]
    assert context.unavailable_resources == {"ur5e-4"}
    assert not context.resources["ur5e-4"].executors
    assert context.resources["ur5e-3"].executors
    assert context.requirements == requirements
    assert runtime.retained_paths == continuations
    assert fault.evidence["requirements"] == requirements
    assert fault.evidence["continuations"] == continuations
    assert fault.evidence["pending_tasks"] == [pending]
    assert runtime.outcome["cancelled_tasks"] == [pending]
    assert not context.pending_tasks and not context.reservations
    assert runtime.stopped and context.environment_model["closed"]
    assert runtime.outcome["failed_resource"] == "ur5e-4"
    assert fault.evidence["injection_status"] == "completed"
    assert fault.evidence["requested_drop_pose"] != fault.evidence["observed_drop_pose"]
    assert fault.evidence["collision_scene"]["collision_scene_acknowledged"] is True
    assert runtime.agent.part_tracker == context.part_tracker
    with pytest.raises(ValueError, match="No pending task"):
        context.acknowledge({"task_id": pending["task_id"]})
    assert not asyncio.run(fault.after_acknowledgement())
    asyncio.run(fault.trigger())
    assert gear._controller.detach_part.call_count == 1


@pytest.mark.parametrize("stage", ["detach", "assumed_detach", "set_pose", "observation", "collision"])
def test_small_gear_slippage_partial_effects_keep_peg_custody_and_failure_latch(
    small_gear_slippage_runtime, monkeypatch, stage,
) -> None:
    """Partial gear effects need reconciliation while peg custody stays authoritative."""
    runtime = small_gear_slippage_runtime()
    context, fault = runtime.context, runtime.conveyor_fault
    peg_before = copy.deepcopy(context.part_tracker["KET4_Square_4mm"])
    gear = next(agent for agent in runtime.resource_agents if agent.agent_name == "ur5e-4")
    if stage == "detach":
        gear._controller.detach_part.return_value = {"success": False}
    elif stage == "assumed_detach":
        gear._controller.detach_part.return_value = {"success": True, "release_mode": "assumed_open"}
    elif stage == "set_pose":
        gear._controller.set_entity_pose.return_value = {"success": False}
    elif stage == "observation":
        monkeypatch.setattr(
            "cais_spade_llm.recovery_framework.failure_effects._observe_part",
            Mock(side_effect=ValueError("no settled gear observation")),
        )
    else:
        gear._controller._sync_part_collision.return_value = False
    asyncio.run(fault.trigger())
    assert runtime.stopped and fault.status == "triggered"
    assert fault.evidence["injection_status"] == "failed"
    assert fault.evidence["physical_state_reconciliation_required"] is True
    assert context.unavailable_resources == {"ur5e-4"}
    assert context.part_tracker["KET4_Square_4mm"] == peg_before
    assert context.resources["ur5e-3"].valuation["held_part"] == "KET4_Square_4mm"
    peg = next(agent for agent in runtime.resource_agents if agent.agent_name == "ur5e-3")
    for method in ("open_gripper", "detach_part", "set_entity_pose", "_sync_part_collision"):
        getattr(peg._controller, method).assert_not_called()
    if stage in {"detach", "assumed_detach"}:
        assert context.resources["ur5e-4"].valuation["held_part"] == "gear_small"
        assert context.part_tracker["gear_small"]["location"] == "ur5e-4"
        assert gear._held_part == "gear_small"
        assert fault.evidence["custody_uncertain"] is True
        gear._controller.set_entity_pose.assert_not_called()
    else:
        assert context.resources["ur5e-4"].valuation["held_part"] is None
        assert context.part_tracker["gear_small"]["state"] == "misplaced"
        assert gear._held_part is None
        assert "observed_pose" not in context.part_tracker["gear_small"]
    with pytest.raises(ValueError, match="reset"):
        fault.arm(True)


@pytest.mark.parametrize("clear_succeeds", [True, False])
def test_small_gear_slippage_reset_requires_confirmation_and_rerun_uses_fresh_pickups(
    small_gear_slippage_runtime, monkeypatch, clear_succeeds,
) -> None:
    """Only confirmed scene reset permits a fresh run with newly observed pickups."""
    from cais_spade_llm.recovery_framework.conveyor_fault import reset_fault_scene

    runtime = small_gear_slippage_runtime()
    fault = runtime.conveyor_fault
    asyncio.run(fault.trigger())
    retained_evidence = copy.deepcopy(fault.evidence)
    bridge = SimpleNamespace(
        ros2_stop=Mock(return_value=None), ros2_start=Mock(return_value=None),
        simulation_start_ready=Mock(return_value=(True, "")),
    )
    visual = Mock(return_value={"status": "completed" if clear_succeeds else "failed"})
    monkeypatch.setattr("cais_spade_llm.recovery_framework.conveyor_fault.marker", visual)
    ok, reason = reset_fault_scene(bridge, runtime)
    assert ok is clear_succeeds
    assert bridge.ros2_stop.call_args_list == [call("recovery_rviz"), call("gazebo_dual")]
    bridge.ros2_start.assert_called_once_with("gazebo_dual")
    visual.assert_called_once_with(fault.marker_scene(), "clear")
    assert fault.status == ("reset" if clear_succeeds else "triggered")
    assert runtime.stopped and fault.evidence == retained_evidence
    with pytest.raises(ValueError, match="reset"):
        fault.arm(True)
    if not clear_succeeds:
        assert "absence could not be confirmed" in reason
        visual.return_value = {"status": "completed"}
        assert reset_fault_scene(bridge, runtime)[0]
        assert fault.status == "reset"
    rerun = small_gear_slippage_runtime(picked=False)
    assert not rerun.stopped and rerun.conveyor_fault.status == "armed"
    assert rerun.context.run_id != runtime.context.run_id
    assert not rerun.context.transitions and not rerun.conveyor_fault.evidence
    assert rerun.context.part_tracker == rerun.context.initial_product_states
    assert not rerun.retained_paths
    assert not rerun.context.unavailable_resources
    assert rerun.context.resources["ur5e-4"].valuation["held_part"] is None
    assert rerun.context.resources["ur5e-3"].valuation["held_part"] is None
    assert not asyncio.run(rerun.conveyor_fault.after_acknowledgement())
    _record_small_gear_slippage_pickups(rerun)
    old_pickups = copy.deepcopy(list(retained_evidence["pickups"].values()))
    rerun.context.transitions.extend(old_pickups)
    assert rerun.conveyor_fault.checkpoint() is None
    assert not asyncio.run(rerun.conveyor_fault.after_acknowledgement())
    del rerun.context.transitions[-len(old_pickups):]
    assert asyncio.run(rerun.conveyor_fault.after_acknowledgement())
    assert rerun.conveyor_fault.evidence["run_id"] == rerun.context.run_id
    assert all(record["acknowledgement"]["run_id"] == rerun.context.run_id
               for record in rerun.conveyor_fault.evidence["pickups"].values())
    assert rerun.context.resources["ur5e-3"].valuation["held_part"] == "KET4_Square_4mm"
    assert rerun.context.part_tracker["gear_small"]["state"] == "misplaced"
    assert fault.evidence == retained_evidence



def _small_gear_place_task(runtime, *, owner="ur5e-4", part="gear_small", event="place_approach") -> dict:
    task = {"run_id": runtime.context.run_id, "task_id": owner + "_placement",
            "resource_id": owner, "event_name": event,
            "parameters": {"part_name": part, "destination_location": "assembly_board-v1"},
            "reservations": ["resource:" + owner, "workspace:assembly_board-v1"]}
    runtime.context.pending_tasks[task["task_id"]] = copy.deepcopy(task)
    runtime.context.reservations.update({key: task["task_id"] for key in task["reservations"]})
    return task


def _small_gear_placement_failure(task, *, progress=.5) -> dict:
    observed = {"x": .1, "y": .2, "z": 1.5 - progress * .5}
    stamp = time.time()
    return {**{key: task[key] for key in ("run_id", "task_id", "resource_id")},
            "part_name": task["parameters"]["part_name"], "checkpoint": "during_place_lowering",
            "source": "gazebo_placement_motion", "function_name": "place_approach", "step_id": "descend",
            "started_pose": {"x": .1, "y": .2, "z": 1.5},
            "target_pose": {"x": .1, "y": .2, "z": 1.}, "observed_pose": observed,
            "progress": progress, "goal_active": True, "goal_cancelled": True, "motion_stopped": True,
            "controller_goal_id": list(range(16)), "observed_at_unix": stamp,
            "stopped_at_unix": stamp + .15, "stopped_pose": copy.deepcopy(observed),
            "stopped_joint_observation": {"positions": [.1] * 6, "stable": True},
            "motion_path_validation": {"validated": True}}


def _assert_no_slippage_effects(runtime) -> None:
    for agent in runtime.resource_agents:
        if agent._controller is not None:
            for method in ("open_gripper", "detach_part", "set_entity_pose", "_sync_part_collision"):
                getattr(agent._controller, method).assert_not_called()


def test_part_slippage_default_and_supported_legacy_checkpoints(small_gear_slippage_runtime):
    from cais_spade_llm.recovery_framework.conveyor_fault import ConveyorFault
    from cais_spade_llm.recovery_framework.failure_checkpoints import (
        CHECKPOINTS,
        SUPPORTED_CHECKPOINTS,
    )

    assert CHECKPOINTS["Part slippage"] == "during_place_lowering"
    assert set(SUPPORTED_CHECKPOINTS["Part slippage"]) == {
        "during_place_lowering", "after_both_pickups_before_place",
    }
    assert all(default in SUPPORTED_CHECKPOINTS[scenario] for scenario, default in CHECKPOINTS.items())
    runtime = small_gear_slippage_runtime(checkpoint=CHECKPOINTS["Part slippage"])
    invalid = {"execution_mode": "simulation", "failure_scenario": copy.deepcopy(runtime.conveyor_fault.configuration)}
    invalid["failure_scenario"]["checkpoint"] = "during_unrelated_motion"
    with pytest.raises(ValueError, match="configured observed failure checkpoints"):
        ConveyorFault(runtime, invalid)


@pytest.mark.parametrize("first", ["ur5e-3", "ur5e-4"])
def test_lowering_slippage_holds_retained_pickup_and_only_allows_selected_approach(
    small_gear_slippage_runtime, first,
):
    runtime = small_gear_slippage_runtime(checkpoint="during_place_lowering")
    context, fault = runtime.context, runtime.conveyor_fault
    pickups = {row["acknowledgement"]["resource_id"]: row for row in context.transitions}
    context.transitions.clear()
    gear = {"resource_id": "ur5e-4", "event_name": "place_approach",
            "parameters": {"part_name": "gear_small"}}
    peg = {"resource_id": "ur5e-3", "event_name": "place_approach",
           "parameters": {"part_name": "KET4_Square_4mm"}}
    assert not fault.holds_task(gear) and not fault.holds_task(peg)
    context.transitions.append(pickups[first])
    assert fault.holds_task(gear) is (first == "ur5e-4")
    assert fault.holds_task(peg) is (first == "ur5e-3")
    context.transitions.append(pickups["ur5e-4" if first == "ur5e-3" else "ur5e-3"])
    assert not fault.holds_task(gear)
    assert fault.holds_task({**gear, "event_name": "place_insert"})
    assert fault.holds_task({**gear, "parameters": {"part_name": "gear_large"}})
    assert fault.holds_task(peg)
    assert not fault.holds_task({"resource_id": "ur5e-1", "event_name": "place_approach"})
    assert fault.checkpoint() is None and not fault.snapshot()["ready"]
    assert not asyncio.run(fault.after_acknowledgement())
    with pytest.raises(ValueError, match="Waiting"):
        asyncio.run(fault.trigger())
    task = _small_gear_place_task(runtime)
    request = fault.robot_request(task)
    assert request == {"run_id": context.run_id, "task_id": task["task_id"], "resource_id": "ur5e-4",
                       "part_name": "gear_small", "checkpoint": "during_place_lowering",
                       "placement_progress": .5}
    task["parameters"]["part_name"] = "gear_large"
    assert fault._robot_task["parameters"]["part_name"] == "gear_small"
    assert fault.checkpoint() is None and not runtime.stopped
    _assert_no_slippage_effects(runtime)


@pytest.mark.parametrize("invalid", [
    "missing_pickup", "missing_observation", "old_pickup", "retained_released", "wrong_custody",
    "later_ack", "retained_pending", "other_gear_pending", "not_pending", "wrong_event", "wrong_part",
    "wrong_resource", "old_task", "disarmed", "stopped",
])
def test_lowering_robot_request_requires_current_both_pickups_and_bound_pending_task(
    small_gear_slippage_runtime, invalid,
):
    runtime = small_gear_slippage_runtime(checkpoint="during_place_lowering")
    context, fault = runtime.context, runtime.conveyor_fault
    task = _small_gear_place_task(runtime)
    pickup = next(row for row in context.transitions if row["acknowledgement"]["resource_id"] == "ur5e-3")
    if invalid == "missing_pickup":
        context.transitions.remove(pickup)
    elif invalid == "missing_observation":
        pickup["observations"] = {}
    elif invalid == "old_pickup":
        pickup["acknowledgement"]["run_id"] = "old_run"
    elif invalid == "retained_released":
        context.resources["ur5e-3"].valuation["held_part"] = None
    elif invalid == "wrong_custody":
        context.part_tracker["KET4_Square_4mm"]["location"] = "assembly_board-v1"
    elif invalid == "later_ack":
        later = copy.deepcopy(pickup)
        later["acknowledgement"].update(event_name="place_approach", task_id="later_placement")
        context.transitions.append(later)
    elif invalid in {"retained_pending", "other_gear_pending"}:
        context.pending_tasks["other_placement"] = {"resource_id": "ur5e-3" if invalid == "retained_pending" else "ur5e-4"}
    elif invalid == "not_pending":
        context.cancel_pending(task["task_id"])
    elif invalid in {"wrong_event", "wrong_resource", "old_task", "wrong_part"}:
        if invalid == "wrong_event":
            task["event_name"] = "place_insert"
        elif invalid == "wrong_resource":
            task["resource_id"] = "ur5e-3"
        elif invalid == "old_task":
            task["run_id"] = "old_run"
        else:
            task["parameters"]["part_name"] = "gear_large"
        context.pending_tasks[task["task_id"]] = copy.deepcopy(task)
    elif invalid == "disarmed":
        fault.arm(False)
    else:
        runtime.stop("Stopped before placement")
    before = context.snapshot()
    assert fault.robot_request(task) == {}
    assert fault._robot_task is None and fault.checkpoint() is None
    assert context.snapshot() == before
    _assert_no_slippage_effects(runtime)


@pytest.mark.parametrize("key,value", [
    ("run_id", "old_run"), ("task_id", "other_task"), ("resource_id", "ur5e-3"),
    ("part_name", "gear_large"), ("checkpoint", "after_both_pickups_before_place"),
    ("source", "gazebo_processing_checkpoint"), ("function_name", "pick_grasp"), ("step_id", "travel"),
])
def test_lowering_slippage_rejects_mismatched_controller_evidence(small_gear_slippage_runtime, key, value):
    runtime = small_gear_slippage_runtime(checkpoint="during_place_lowering")
    task = _small_gear_place_task(runtime)
    fault = runtime.conveyor_fault
    assert fault.robot_request(task)
    evidence = _small_gear_placement_failure(task)
    evidence[key] = value
    before = runtime.context.snapshot()
    with pytest.raises(ValueError, match="Stale or unrelated"):
        asyncio.run(fault.accept_robot_failure(task, {"failure_injection": evidence}))
    assert runtime.context.pending_for(task["task_id"]) == task
    assert runtime.context.snapshot() == before and not runtime.stopped
    assert fault.status == "armed" and fault.checkpoint() is None
    _assert_no_slippage_effects(runtime)


@pytest.mark.parametrize("invalid", [
    {"goal_active": False}, {"goal_cancelled": False}, {"motion_stopped": False}, {"motion_stopped": 1},
    {"progress": .49}, {"progress": 1.}, {"progress": float("nan")}, {"progress": True},
    {"controller_goal_id": []}, {"controller_goal_id": [256] * 16}, {"controller_goal_id": "goal"},
    {"started_pose": {}}, {"target_pose": {"x": .1, "y": .2, "z": 1.6}},
    {"observed_pose": {"x": .1, "y": .2, "z": 1.3}},
    {"observed_pose": {"x": .1, "y": .2, "z": 1.}},
    {"observed_pose": {"x": float("nan"), "y": .2, "z": 1.25}},
    {"observed_at_unix": float("inf")}, {"observed_at_unix": 0},
])
def test_lowering_slippage_rejects_unconfirmed_or_unobserved_descent(small_gear_slippage_runtime, invalid):
    runtime = small_gear_slippage_runtime(checkpoint="during_place_lowering")
    task = _small_gear_place_task(runtime)
    fault = runtime.conveyor_fault
    assert fault.robot_request(task)
    evidence = {**_small_gear_placement_failure(task), **invalid}
    with pytest.raises(ValueError, match="Invalid|downward progress"):
        asyncio.run(fault.accept_robot_failure(task, {"failure_injection": evidence}))
    assert runtime.context.pending_for(task["task_id"]) == task
    assert fault.status == "armed" and fault.checkpoint() is None and not runtime.stopped
    assert not runtime.context.unavailable_resources
    _assert_no_slippage_effects(runtime)


@pytest.mark.parametrize("invalid", ["retained_custody", "gear_custody", "later_ack", "cancelled_without_binding"])
def test_lowering_slippage_rechecks_custody_before_consuming_failure(small_gear_slippage_runtime, invalid):
    runtime = small_gear_slippage_runtime(checkpoint="during_place_lowering")
    context, fault = runtime.context, runtime.conveyor_fault
    task = _small_gear_place_task(runtime)
    evidence = _small_gear_placement_failure(task)
    if invalid != "cancelled_without_binding":
        assert fault.robot_request(task)
    if invalid in {"retained_custody", "gear_custody"}:
        context.resources["ur5e-3" if invalid == "retained_custody" else "ur5e-4"].valuation["held_part"] = None
    elif invalid == "later_ack":
        later = copy.deepcopy(context.transitions[0])
        later["acknowledgement"].update(event_name="place_approach", task_id="late_placement")
        context.transitions.append(later)
    else:
        runtime.stop("No controller binding was created")
    with pytest.raises(ValueError, match="Stale placement task|pickup custody"):
        asyncio.run(fault.accept_robot_failure(task, {"failure_injection": evidence}))
    assert fault.status == "armed" and not fault.evidence
    _assert_no_slippage_effects(runtime)


@pytest.mark.parametrize("progress", [.5, .75, .999])
@pytest.mark.parametrize("disarmed_after_capture", [False, True])
def test_lowering_slippage_retains_pending_placement_peg_and_obligations(
    small_gear_slippage_runtime, progress, disarmed_after_capture,
):
    runtime = small_gear_slippage_runtime(checkpoint="during_place_lowering")
    context, fault = runtime.context, runtime.conveyor_fault
    task = _small_gear_place_task(runtime)
    assert fault.robot_request(task)
    captured = _small_gear_placement_failure(task, progress=progress)
    peg = next(agent for agent in runtime.resource_agents if agent.agent_name == "ur5e-3")
    gear = next(agent for agent in runtime.resource_agents if agent.agent_name == "ur5e-4")
    peg_before = copy.deepcopy(context.resources["ur5e-3"].valuation)
    peg_part_before = copy.deepcopy(context.part_tracker["KET4_Square_4mm"])
    requirements, continuations = copy.deepcopy(context.requirements), copy.deepcopy(runtime.retained_paths)
    if disarmed_after_capture:
        fault.arm(False)
    assert asyncio.run(fault.accept_robot_failure(task, {"failure_injection": captured}))
    assert fault.status == "triggered" and runtime.stopped
    assert fault.evidence["placement_motion"] == captured
    assert fault.evidence["pending_tasks"] == [task] == runtime.outcome["cancelled_tasks"]
    assert not context.pending_tasks and not context.reservations
    assert context.resources["ur5e-3"].valuation == peg_before
    assert context.part_tracker["KET4_Square_4mm"] == peg_part_before
    assert peg._held_part == "KET4_Square_4mm" and peg._gripper_state == "closed"
    assert context.part_tracker["gear_small"]["state"] == "misplaced"
    assert context.resources["ur5e-4"].valuation["held_part"] is None
    assert context.requirements == fault.evidence["requirements"] == requirements
    assert runtime.retained_paths == fault.evidence["continuations"] == continuations
    assert all(row["acknowledgement"]["event_name"] == "pick_grasp" for row in context.transitions)
    assert task["task_id"] not in context.acknowledgements
    assert set(fault.evidence["pickups"]) == {"ur5e-3", "ur5e-4"}
    gear._controller.detach_part.assert_called_once_with("gear_small", assume_released_if_open=False)
    for method in ("open_gripper", "detach_part", "set_entity_pose", "_sync_part_collision"):
        getattr(peg._controller, method).assert_not_called()
    assert asyncio.run(fault.accept_robot_failure(task, {"failure_injection": captured}))
    assert not asyncio.run(fault.after_acknowledgement())
    gear._controller.detach_part.assert_called_once()
    captured["observed_pose"]["z"] = 99.
    assert fault.evidence["placement_motion"]["observed_pose"]["z"] < 1.5


@pytest.mark.parametrize("partial", [False, True])
def test_stop_retains_controller_lowering_evidence_after_request_cleanup(
    small_gear_slippage_runtime, partial,
):
    async def exercise():
        runtime = small_gear_slippage_runtime(checkpoint="during_place_lowering")
        task = _small_gear_place_task(runtime)
        fault = runtime.conveyor_fault
        request = fault.robot_request(task)
        controller = next(agent._controller for agent in runtime.resource_agents if agent.agent_name == "ur5e-4")
        captured = _small_gear_placement_failure(task)
        if partial:
            captured.update(goal_cancelled=False, motion_stopped=False)
        controller._simulation_fault_evidence = copy.deepcopy(captured)
        controller._simulation_fault_request = {}
        controller._simulation_task_id = None
        assert request and fault._robot_task == task
        runtime.stop("Stop cancelled the execution coroutine")
        await fault.retain_worker_interruption()
        if partial:
            assert fault.status == "armed" and fault.checkpoint() is None
            assert fault.evidence["injection_status"] == "interrupted"
            assert fault.evidence["physical_state_reconciliation_required"]
            assert fault.evidence["placement_motion"] == captured
            assert fault.evidence["pending_tasks"] == [task]
            assert fault.evidence["continuations"] == runtime.retained_paths
            retained = copy.deepcopy(fault.evidence)
            fault.not_reached()
            assert fault.evidence == retained
            _assert_no_slippage_effects(runtime)
            controller._simulation_fault_evidence.update(goal_cancelled=True, motion_stopped=True)
            await fault.retain_worker_interruption()
        assert fault.status == "triggered" and fault.evidence["injection_status"] == "completed"
        assert fault.evidence["pending_tasks"] == runtime.outcome["cancelled_tasks"] == [task]
        assert fault.evidence["placement_motion"]["goal_cancelled"] is True
        assert runtime.context.resources["ur5e-3"].valuation["held_part"] == "KET4_Square_4mm"
        controller.detach_part.assert_called_once()
        await fault.retain_worker_interruption()
        controller.detach_part.assert_called_once()

    asyncio.run(exercise())


def test_stop_ignores_stale_controller_fault_and_other_resource_capture(small_gear_slippage_runtime):
    runtime = small_gear_slippage_runtime(checkpoint="during_place_lowering")
    task = _small_gear_place_task(runtime)
    fault = runtime.conveyor_fault
    assert fault.robot_request(task)
    gear = next(agent._controller for agent in runtime.resource_agents if agent.agent_name == "ur5e-4")
    peg = next(agent._controller for agent in runtime.resource_agents if agent.agent_name == "ur5e-3")
    gear._simulation_fault_evidence = {**_small_gear_placement_failure(task), "run_id": "old_run"}
    peg._simulation_fault_evidence = _small_gear_placement_failure(task)
    runtime.stop("Stop without current selected-controller evidence")
    asyncio.run(fault.retain_worker_interruption())
    assert fault.status == "armed" and not fault.evidence
    _assert_no_slippage_effects(runtime)


@pytest.mark.parametrize("clear_succeeds", [False, True])
def test_lowering_fault_reset_and_rerun_require_a_new_bound_interruption(
    small_gear_slippage_runtime, monkeypatch, clear_succeeds,
):
    from cais_spade_llm.recovery_framework.conveyor_fault import reset_fault_scene

    runtime = small_gear_slippage_runtime(checkpoint="during_place_lowering")
    task = _small_gear_place_task(runtime)
    fault = runtime.conveyor_fault
    assert fault.robot_request(task)
    captured = _small_gear_placement_failure(task)
    assert asyncio.run(fault.accept_robot_failure(task, {"failure_injection": captured}))
    retained = copy.deepcopy(fault.evidence)
    bridge = SimpleNamespace(ros2_stop=Mock(return_value=None), ros2_start=Mock(return_value=None),
                             simulation_start_ready=Mock(return_value=(True, "ready")))
    visual = Mock(return_value={"status": "completed" if clear_succeeds else "failed"})
    monkeypatch.setattr("cais_spade_llm.recovery_framework.conveyor_fault.marker", visual)
    ok, _ = reset_fault_scene(bridge, runtime)
    assert ok is clear_succeeds and fault.evidence == retained
    if not clear_succeeds:
        assert fault.status == "triggered" and fault._robot_task == task
        visual.return_value = {"status": "completed"}
        assert reset_fault_scene(bridge, runtime)[0]
    assert fault.status == "reset" and fault._robot_task is None and fault._robot_evidence is None
    assert fault.robot_request(task) == {}
    with pytest.raises(ValueError, match="reset"):
        fault.arm(True)
    rerun = small_gear_slippage_runtime(picked=False, checkpoint="during_place_lowering")
    assert rerun.context.run_id != runtime.context.run_id
    fresh_task = _small_gear_place_task(rerun)
    assert rerun.conveyor_fault.robot_request(fresh_task) == {}
    _record_small_gear_slippage_pickups(rerun)
    assert not asyncio.run(rerun.conveyor_fault.after_acknowledgement())
    assert rerun.conveyor_fault.robot_request(fresh_task)
    with pytest.raises(ValueError, match="Stale or unrelated"):
        asyncio.run(rerun.conveyor_fault.accept_robot_failure(fresh_task, {"failure_injection": captured}))
    assert asyncio.run(rerun.conveyor_fault.accept_robot_failure(
        fresh_task, {"failure_injection": _small_gear_placement_failure(fresh_task)}))
    assert rerun.conveyor_fault.evidence["placement_motion"]["run_id"] == rerun.context.run_id
    assert fault.evidence == retained


@pytest.mark.parametrize("slipping_resource", ["ur5e-3", "ur5e-4"])
def test_explicit_legacy_slippage_keeps_both_directions_and_pickup_trigger(
    small_gear_slippage_runtime, slipping_resource,
):
    runtime = small_gear_slippage_runtime(slipping_resource=slipping_resource)
    fault = runtime.conveyor_fault
    assert fault.configuration["checkpoint"] == "after_both_pickups_before_place"
    assert fault.checkpoint() is not None
    assert fault.holds_task({"resource_id": slipping_resource, "event_name": "place_approach"})
    assert fault.robot_request({"resource_id": slipping_resource, "event_name": "place_approach"}) == {}
    assert asyncio.run(fault.after_acknowledgement())
    other = fault.configuration["additional_condition"]
    assert runtime.context.resources[other["resource_id"]].valuation["held_part"] == other["part_name"]
    assert "placement_motion" not in fault.evidence



def test_lowering_disarm_and_nominal_results_preserve_the_ordinary_task(small_gear_slippage_runtime):
    runtime = small_gear_slippage_runtime(checkpoint="during_place_lowering")
    task = _small_gear_place_task(runtime)
    fault = runtime.conveyor_fault
    before = runtime.context.snapshot()
    fault.arm(False)
    assert fault.robot_request(task) == {}
    assert not asyncio.run(fault.accept_robot_failure(task, {"status": "completed"}))
    assert not asyncio.run(fault.after_acknowledgement())
    assert runtime.context.pending_for(task["task_id"]) == task
    assert runtime.context.snapshot() == before and not runtime.stopped
    fault.arm(True)
    assert fault.robot_request(task)
    assert not asyncio.run(fault.accept_robot_failure(task, {"failure_injection": None}))
    assert fault.checkpoint() is None and runtime.context.pending_for(task["task_id"]) == task
    _assert_no_slippage_effects(runtime)


def test_lowering_request_and_evidence_obey_a_configured_later_progress(small_gear_slippage_runtime):
    runtime = small_gear_slippage_runtime(checkpoint="during_place_lowering")
    task = _small_gear_place_task(runtime)
    fault = runtime.conveyor_fault
    fault.configuration["placement_progress"] = .75
    assert fault.robot_request(task)["placement_progress"] == .75
    with pytest.raises(ValueError, match="Invalid"):
        asyncio.run(fault.accept_robot_failure(task, {"failure_injection": _small_gear_placement_failure(task)}))
    assert not runtime.stopped and runtime.context.pending_for(task["task_id"]) == task
    assert asyncio.run(fault.accept_robot_failure(
        task, {"failure_injection": _small_gear_placement_failure(task, progress=.75)}))


@pytest.mark.parametrize("progress", [.49, 1., float("nan")])
def test_lowering_request_rejects_invalid_progress_before_controller_dispatch(
    small_gear_slippage_runtime, progress,
):
    runtime = small_gear_slippage_runtime(checkpoint="during_place_lowering")
    task = _small_gear_place_task(runtime)
    runtime.conveyor_fault.configuration["placement_progress"] = progress
    with pytest.raises(ValueError, match="Placement interruption progress"):
        runtime.conveyor_fault.robot_request(task)
    assert runtime.conveyor_fault._robot_task is None and not runtime.stopped
    assert runtime.context.pending_for(task["task_id"]) == task
    _assert_no_slippage_effects(runtime)


def test_lowering_caller_cancellation_keeps_fault_effects_and_blocks_early_reset(
    small_gear_slippage_runtime, monkeypatch,
):
    from cais_spade_llm.recovery_framework.conveyor_fault import reset_fault_scene

    async def exercise():
        runtime = small_gear_slippage_runtime(checkpoint="during_place_lowering")
        task = _small_gear_place_task(runtime)
        fault = runtime.conveyor_fault
        assert fault.robot_request(task)
        captured = _small_gear_placement_failure(task)
        entered, finish = asyncio.Event(), asyncio.Event()

        async def slip(_runtime, _configuration, evidence):
            evidence["detach"] = {"success": True}
            entered.set()
            await finish.wait()
            evidence["observed_drop_pose"] = {"x": 0., "y": .2, "z": 1.04}

        monkeypatch.setattr("cais_spade_llm.recovery_framework.failure_effects.slip_part", slip)
        caller = asyncio.create_task(fault.accept_robot_failure(task, {"failure_injection": captured}))
        await entered.wait()
        caller.cancel()
        with pytest.raises(asyncio.CancelledError):
            await caller
        assert fault.status == "triggered" and runtime.stopped
        assert fault.evidence["pending_tasks"] == [task]
        assert fault.evidence["placement_motion"] == captured
        bridge = SimpleNamespace(ros2_stop=Mock())
        ok, reason = reset_fault_scene(bridge, runtime)
        assert not ok and "still recording" in reason
        bridge.ros2_stop.assert_not_called()
        assert (await fault.trigger())["status"] == "triggered"
        finish.set()
        await fault._injection_task
        assert fault.evidence["injection_status"] == "completed"

    asyncio.run(exercise())
