"""Tests for the dedicated Spec2Primitives NIST Gazebo scene and launch route."""

from __future__ import annotations

import hashlib
import math
import xml.etree.ElementTree as ET
from pathlib import Path

from cais_spade_llm.ui.ros2_processes import (
    build_ros2_launch_cmds,
    ros2_launch_required_paths,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
WORLD_PATH = (
    REPOSITORY_ROOT
    / "ros2"
    / "cais_lab_robotics"
    / "worlds"
    / "table_spec2primitives.world"
)
CAD_ROOT = REPOSITORY_ROOT / "ros2" / "cais_lab_robotics" / "cad_models"

KET_MODELS = {
    "KET4_Square_4mm": ("-0.46 0.06 1.075 0 0 0", "0.004 0.004 0.05"),
    "KET8_Square_8mm": ("-0.34 0.06 1.075 0 0 0", "0.008 0.007 0.05"),
    "KET12_Square_12mm": ("-0.46 -0.06 1.075 0 0 0", "0.012 0.008 0.05"),
    "KET16_Square_16mm": ("-0.34 -0.06 1.075 0 0 0", "0.016 0.010 0.05"),
}
RGOCG_MODELS = {
    "RGOCG4-50_Round_4mm": ("0.34 0.36 1.075 0 0 0", "0.002"),
    "RGOCG8-50_8mm": ("0.46 0.36 1.075 0 0 0", "0.004"),
    "RGOCG12-50_12mm": ("0.34 0.24 1.075 0 0 0", "0.006"),
    "RGOCG16-50_16mm": ("0.46 0.24 1.075 0 0 0", "0.008"),
}
GEAR_POSES = {
    "gear_small": "0.30 -0.30 1.10 0 0 0",
    "gear_medium": "0.40 -0.30 1.10 0 0 0",
    "gear_large": "0.50 -0.30 1.10 0 0 0",
}
GEAR_SHAFT_POSES = {
    "Gear_Shaft_1": "0 0.0500000 0.0050000 0 0 0",
    "Gear_Shaft_2": "0 0.0200000 0.0050000 0 0 0",
    "Gear_Shaft_3": "0 -0.0300000 0.0050000 0 0 0",
}
GMC_GEAR_MOUNT_HOLE_CENTERS_MM = {
    "Gear_Shaft_1": (221.4496, 47.1758),
    "Gear_Shaft_2": (191.4496, 47.1758),
    "Gear_Shaft_3": (141.4496, 47.1758),
}


def _world() -> ET.Element:
    world = ET.parse(WORLD_PATH).getroot().find("world")
    assert world is not None
    return world


def _models_by_name(world: ET.Element) -> dict[str, ET.Element]:
    return {model.attrib["name"]: model for model in world.findall("model")}


def _includes_by_name(world: ET.Element) -> dict[str, ET.Element]:
    return {
        name.text: include
        for include in world.findall("include")
        if (name := include.find("name")) is not None and name.text
    }


def _required_text(element: ET.Element, path: str) -> str:
    value = element.findtext(path)
    assert value is not None
    return value.strip()


def test_plate_uses_exact_mesh_pose_scale_and_collision() -> None:
    plate = _models_by_name(_world())["GMC_Laser_Plate_Virtual"]

    assert _required_text(plate, "static") == "true"
    assert _required_text(plate, "pose") == (
        "0 0 1.0439916 3.141592653589793 0 0"
    )
    for element_name in ("collision", "visual"):
        element = plate.find(f"link/{element_name}")
        assert element is not None
        assert _required_text(element, "pose") == "-0.192 -0.192 0 0 0 0"
        assert (
            _required_text(element, "geometry/mesh/uri")
            == "model://cad_models/GMC_Laser_Plate_Virtual.STL"
        )
        assert _required_text(element, "geometry/mesh/scale") == "0.001 0.001 0.001"


def test_installed_gear_fixture_uses_exact_cad_and_collisions() -> None:
    world = _world()
    gear_plate_models = [
        model for model in world.findall("model") if model.attrib["name"] == "Gear_Plate"
    ]

    assert len(gear_plate_models) == 1
    gear_plate = gear_plate_models[0]
    assert _required_text(gear_plate, "static") == "true"
    assert _required_text(gear_plate, "pose") == (
        "-0.0205504 0.1448242 1.0439916 0 0 -1.57079632679"
    )

    links = {link.attrib["name"]: link for link in gear_plate.findall("link")}
    assert links.keys() == {"Gear_Plate", *GEAR_SHAFT_POSES}

    plate_link = links["Gear_Plate"]
    plate_mesh_pose = "-0.2131605 0.2018883 0.0847159 3.141592653589793 0 0"
    for element_name in ("collision", "visual"):
        element = plate_link.find(element_name)
        assert element is not None
        assert _required_text(element, "pose") == plate_mesh_pose
        assert (
            _required_text(element, "geometry/mesh/uri")
            == "model://cad_models/Gear_Plate.STL"
        )
        assert _required_text(element, "geometry/mesh/scale") == (
            "0.001 0.001 0.001"
        )
    assert _required_text(plate_link, "visual/material/script/name") == "Gazebo/Grey"

    shaft_mesh_pose = "-0.2131605 0.2318883 0.0797159 3.141592653589793 0 0"
    for name, pose in GEAR_SHAFT_POSES.items():
        shaft = links[name]
        assert _required_text(shaft, "pose") == pose
        assert _required_text(shaft, "collision/pose") == (
            "0 0 0.0100000 0 0 0"
        )
        assert _required_text(shaft, "collision/geometry/cylinder/radius") == "0.005"
        assert _required_text(shaft, "collision/geometry/cylinder/length") == "0.020"
        assert _required_text(shaft, "visual/pose") == shaft_mesh_pose
        assert (
            _required_text(shaft, "visual/geometry/mesh/uri")
            == "model://cad_models/Gear_Shaft.STL"
        )
        assert _required_text(shaft, "visual/geometry/mesh/scale") == (
            "0.001 0.001 0.001"
        )
        assert _required_text(shaft, "visual/material/script/name") == "Gazebo/Grey"


def test_gear_fixture_surfaces_and_cad_centers_align() -> None:
    models = _models_by_name(_world())
    gmc_pose = [float(value) for value in _required_text(
        models["GMC_Laser_Plate_Virtual"], "pose"
    ).split()]
    gear_plate = models["Gear_Plate"]
    gear_plate_pose = [
        float(value) for value in _required_text(gear_plate, "pose").split()
    ]
    gmc_upper_face = gmc_pose[2]
    gmc_lower_face = gmc_pose[2] - 0.0089916

    assert math.isclose(gmc_pose[3], math.pi, abs_tol=1e-15)
    assert math.isclose(gmc_lower_face, 1.035, abs_tol=1e-9)
    assert math.isclose(gear_plate_pose[2], gmc_upper_face, abs_tol=1e-9)

    links = {link.attrib["name"]: link for link in gear_plate.findall("link")}
    yaw = gear_plate_pose[5]
    for name, hole_center_mm in GMC_GEAR_MOUNT_HOLE_CENTERS_MM.items():
        relative_pose = [
            float(value) for value in _required_text(links[name], "pose").split()
        ]
        world_pose = (
            gear_plate_pose[0]
            + math.cos(yaw) * relative_pose[0]
            - math.sin(yaw) * relative_pose[1],
            gear_plate_pose[1]
            + math.sin(yaw) * relative_pose[0]
            + math.cos(yaw) * relative_pose[1],
            gear_plate_pose[2] + relative_pose[2],
        )
        expected_pose = (
            hole_center_mm[0] / 1000 - 0.192,
            -(hole_center_mm[1] / 1000 - 0.192),
            1.0489916,
        )
        assert all(
            math.isclose(actual, expected, abs_tol=1e-9)
            for actual, expected in zip(world_pose, expected_pose, strict=True)
        )


def test_pin_models_use_exact_stls_collisions_poses_and_neutral_material() -> None:
    models = _models_by_name(_world())

    for name, (pose, box_size) in KET_MODELS.items():
        model = models[name]
        assert _required_text(model, "pose") == pose
        assert _required_text(model, "link/collision/geometry/box/size") == box_size
        assert (
            _required_text(model, "link/visual/geometry/mesh/uri")
            == f"model://cad_models/{name}.STL"
        )
        assert _required_text(model, "link/visual/geometry/mesh/scale") == (
            "0.001 0.001 0.001"
        )
        assert _required_text(model, "link/visual/material/script/name") == "Gazebo/Grey"

    for name, (pose, radius) in RGOCG_MODELS.items():
        model = models[name]
        assert _required_text(model, "pose") == pose
        assert _required_text(model, "link/collision/geometry/cylinder/radius") == radius
        assert _required_text(model, "link/collision/geometry/cylinder/length") == "0.05"
        assert (
            _required_text(model, "link/visual/geometry/mesh/uri")
            == f"model://cad_models/{name}.STL"
        )
        assert _required_text(model, "link/visual/geometry/mesh/scale") == (
            "0.001 0.001 0.001"
        )
        assert _required_text(model, "link/visual/material/script/name") == "Gazebo/Grey"


def test_printers_gears_and_camera_configuration() -> None:
    world = _world()
    models = _models_by_name(world)
    includes = _includes_by_name(world)

    printer_poses = {
        "prusa_mk3": "-0.4 0 1.04 0 0 0",
        "prusa_mk4_1": "0.4 0.3 1.04 0 0 0",
        "prusa_mk4_2": "0.4 -0.3 1.04 0 0 0",
    }
    for name, pose in printer_poses.items():
        assert _required_text(models[name], "pose") == pose
    for name, pose in GEAR_POSES.items():
        assert _required_text(includes[name], "uri") == f"model://{name}"
        assert _required_text(includes[name], "pose") == pose

    cameras = {"cam_mk3", "cam_mk4_1", "cam_mk4_2", "cam_assembly"}
    camera_far_m = {
        "cam_mk3": "5.0",
        "cam_mk4_1": "5.0",
        "cam_mk4_2": "5.0",
        "cam_assembly": "2.0",
    }
    camera_min_depth_m = {
        "cam_mk3": "0.05",
        "cam_mk4_1": "0.05",
        "cam_mk4_2": "0.05",
    }
    assert cameras <= models.keys()
    for name in cameras:
        sensor = models[name].find("link/sensor")
        assert sensor is not None
        assert sensor.attrib["type"] == "depth"
        assert _required_text(sensor, "camera/image/width") == "640"
        assert _required_text(sensor, "camera/image/height") == "480"
        assert _required_text(sensor, "camera/image/format") == "R8G8B8"
        assert _required_text(sensor, "camera/clip/far") == camera_far_m[name]
        plugin = sensor.find("plugin")
        assert plugin is not None
        if name in camera_min_depth_m:
            assert _required_text(plugin, "min_depth") == camera_min_depth_m[name]
        else:
            assert plugin.find("min_depth") is None
    assembly_camera = models["cam_assembly"]
    assert _required_text(assembly_camera, "pose") == "0 0 1.55 0 1.5708 0"
    sensor = assembly_camera.find("link/sensor")
    assert sensor is not None
    horizontal_fov = float(_required_text(sensor, "camera/horizontal_fov"))
    distance_to_plate_top = 1.55 - 1.0439916
    vertical_half_angle = math.atan(math.tan(horizontal_fov / 2) * 480 / 640)
    vertical_coverage = 2 * distance_to_plate_top * math.tan(vertical_half_angle)
    assert vertical_coverage > 0.384


def test_gear_models_use_nist_stl_visuals_and_configured_collisions() -> None:
    expected = {
        "gear_small": ("Gear_Small.STL", "0.01093880465", "0.02"),
        "gear_medium": ("Gear_Medium.STL", "0.02099796295", "0.02"),
        "gear_large": ("Gear_Large.STL", "0.0309972", "0.02"),
    }

    for model_name, (filename, radius, length) in expected.items():
        model_path = (
            REPOSITORY_ROOT
            / "ros2"
            / "cais_lab_robotics"
            / "models"
            / model_name
            / "model.sdf"
        )
        model = ET.parse(model_path).getroot().find("model")
        assert model is not None
        assert _required_text(model, "link/collision/geometry/cylinder/radius") == radius
        assert _required_text(model, "link/collision/geometry/cylinder/length") == length
        assert (
            _required_text(model, "link/visual/geometry/mesh/uri")
            == f"model://{model_name}/meshes/{filename}"
        )
        assert _required_text(model, "link/visual/geometry/mesh/scale") == (
            "0.001 0.001 0.001"
        )


def test_dedicated_world_has_no_mock_pin_models() -> None:
    source = WORLD_PATH.read_text(encoding="utf-8")

    assert "rect_pin_" not in source
    assert "circ_pin_" not in source
    assert "assembly_board_v1" not in source


def test_all_nist_mesh_references_resolve_without_copies() -> None:
    world = _world()
    expected_stems = {
        "GMC_Laser_Plate_Virtual",
        "Gear_Plate",
        "Gear_Shaft",
        *KET_MODELS,
        *RGOCG_MODELS,
    }
    mesh_uris = {
        uri.text.strip()
        for uri in world.findall(".//mesh/uri")
        if uri.text and uri.text.strip().startswith("model://cad_models/")
    }

    assert mesh_uris == {
        f"model://cad_models/{stem}.STL" for stem in expected_stems
    }
    for stem in expected_stems:
        assert (CAD_ROOT / f"{stem}.STL").is_file()


def test_spec2primitives_process_command_and_world_prerequisite_are_exact() -> None:
    commands = build_ros2_launch_cmds(
        project_root=REPOSITORY_ROOT,
        venv_python=Path("/tmp/spec2primitives-venv"),
        ur5e_rg2_gripper_script=Path("/tmp/spec2primitives-gripper.py"),
        ur5e_rtde_trajectory_script=Path(
            "/tmp/spec2primitives/config/runtime/trajectory.py"
        ),
        ur5e_rtde_trajectory_status=Path("/tmp/spec2primitives-status.json"),
    )

    assert commands["gazebo_dual_spec2primitives"] == (
        "ros2 launch cais_lab_robotics dual_moveit_gazebo.launch.py "
        "world_file:=table_spec2primitives.world run_perception:=false "
        "include_assembly_parts:=true include_loose_parts:=true"
    )
    assert commands["gazebo_dual"] == (
        "ros2 launch cais_lab_robotics dual_moveit_gazebo.launch.py "
        "run_perception:=false include_assembly_parts:=true include_loose_parts:=true"
    )

    required_paths = ros2_launch_required_paths(
        "gazebo_dual_spec2primitives",
        venv_python=Path("/tmp/spec2primitives-venv"),
        ur5e_rg2_gripper_script=Path("/tmp/spec2primitives-gripper.py"),
        ur5e_rtde_trajectory_script=Path(
            "/tmp/spec2primitives/config/runtime/trajectory.py"
        ),
    )
    assert any(
        path.name == "table_spec2primitives.world" and "Spec2Primitives NIST world" in error
        for path, error in required_paths
    )


def test_world_file_argument_defaults_and_pass_through_are_declared() -> None:
    xarm_launch = (
        REPOSITORY_ROOT
        / "ros2/cais_lab_robotics/launch/xarm6_ur5e_gazebo.launch.py"
    ).read_text(encoding="utf-8")
    dual_launch = (
        REPOSITORY_ROOT
        / "ros2/cais_lab_robotics/launch/dual_moveit_gazebo.launch.py"
    ).read_text(encoding="utf-8")

    for source in (xarm_launch, dual_launch):
        assert "'world_file'" in source
        assert "default_value='table.world'" in source
    assert "'world_file': world_file" in dual_launch
    assert "LaunchConfiguration('world_file').perform(context)" in xarm_launch


def test_spec2primitives_navigation_and_route_pass_the_existing_runtime() -> None:
    app_source = (
        REPOSITORY_ROOT / "cais_spade_llm" / "ui" / "app.py"
    ).read_text(encoding="utf-8")

    assert '("Spec2Primitives", "/spec2primitives", "account_tree")' in app_source
    assert '@ui.page("/spec2primitives")' in app_source
    assert "spec2primitives_ui.render(bridge)" in app_source


def test_reference_pdf_hash_is_recorded_content() -> None:
    reference_pdf = (
        REPOSITORY_ROOT
        / "cais_spade_llm/spec2primitives/references/products/"
        "NIST_assembly_instructions.pdf"
    )

    digest = hashlib.sha256(reference_pdf.read_bytes()).hexdigest()
    assert digest == "a0aa044e88aee3f1d1011eb8e681a626c4b459706bded36b684a21f0fd189e03"


def test_spec2primitives_directory_boundaries_are_present() -> None:
    spec2primitives_root = REPOSITORY_ROOT / "cais_spade_llm" / "spec2primitives"
    expected_directories = (
        "agents/pa",
        "agents/ra",
        "tools/document_evidence",
        "tools/rgb_d_cad_grounding",
        "references/products",
        "references/resources/primitive_catalogs",
        "contexts",
        "evaluations/ground_truth",
    )

    for relative_path in expected_directories:
        assert (spec2primitives_root / relative_path).is_dir()

    assert not (spec2primitives_root / "artifacts").exists()
    assert not (spec2primitives_root / "exact_ref_resolver.py").exists()
    assert not (spec2primitives_root / "observation_context.py").exists()


def test_gear_fixture_cad_hashes_and_manifest_are_exact() -> None:
    expected_hashes = {
        "Gear_Plate.STL": (
            "5a087e7e8a0803d4a74a1bd273a346d5551747113ac8e66ce8ed4d12e7472555"
        ),
        "Gear_Shaft.STL": (
            "0f6c7b27502a308f49dfdbaf1c145bd7446ff1291ba44af9e8ba55729fe3f9ee"
        ),
    }
    reference_manifest = (
        REPOSITORY_ROOT
        / "cais_spade_llm/spec2primitives/references/products/README.md"
    ).read_text(encoding="utf-8")

    for filename, expected_hash in expected_hashes.items():
        digest = hashlib.sha256((CAD_ROOT / filename).read_bytes()).hexdigest()
        assert digest == expected_hash
        assert filename in reference_manifest
        assert expected_hash in reference_manifest

    assert "three M6 bolts" in reference_manifest
    assert "three `Gear_Shaft` parts" in reference_manifest


def test_mandatory_no_answer_leak_rule_is_in_every_scope_document() -> None:
    spec2primitives_root = REPOSITORY_ROOT / "cais_spade_llm" / "spec2primitives"
    required_documents = (
        "AGENTS.md",
        "ICRA_SCOPE.md",
        "IMPLEMENTATION_PLAN.md",
        "README.md",
    )
    required_terms = (
        "## MUST: Do not leak the answer",
        "approved candidate CAD files",
        "/gazebo/model_states",
        "/get_entity_state",
        "current detector responses",
        "evaluator labels",
        "after the prediction is finalized",
        "must not import, invoke, or share runtime objects",
        "invalid and must not be reported",
    )

    for filename in required_documents:
        contents = (spec2primitives_root / filename).read_text(encoding="utf-8")
        normalized_contents = " ".join(contents.split())
        for required_term in required_terms:
            assert required_term in normalized_contents, (
                f"{filename} is missing {required_term!r}"
            )
