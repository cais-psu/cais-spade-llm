"""Focused contracts for the recovery-framework known NIST Products catalog."""

from __future__ import annotations

import copy
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

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
LEGACY_GEOMETRY_PATH = ROOT / "test/fixtures/case3_recovery/assembly_board-v1.json"
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
        (LEGACY_GEOMETRY_PATH.parent / "runtime_context.json").read_text(encoding="utf-8")
    )
    assert ROOT / runtime_context["product_geometry"] == LEGACY_GEOMETRY_PATH
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
