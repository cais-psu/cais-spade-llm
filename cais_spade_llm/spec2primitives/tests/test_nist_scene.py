"""Tests for the Spec2Primitives and recovery framework NIST Gazebo scenes."""

from __future__ import annotations

import ast
import hashlib
import math
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import pytest

from cais_spade_llm.ui.ros2_processes import (
    build_ros2_launch_cmds,
    ros2_launch_prereq_error,
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
RECOVERY_WORLD_PATH = WORLD_PATH.with_name("table_recovery_framework.world")
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
RECOVERY_KET_POSES = {
    "KET4_Square_4mm": "-8.95 1.88 1.165 0 0 1.57079632679",
    "KET8_Square_8mm": "-8.95 2.16 1.165 0 0 1.57079632679",
    "KET12_Square_12mm": "-8.95 2.44 1.165 0 0 1.57079632679",
    "KET16_Square_16mm": "-8.95 2.72 1.165 0 0 1.57079632679",
}
RECOVERY_RGOCG_POSES = {
    "RGOCG4-50_Round_4mm": "-8.65 1.88 1.165 0 0 1.57079632679",
    "RGOCG8-50_8mm": "-8.65 2.16 1.165 0 0 1.57079632679",
    "RGOCG12-50_12mm": "-8.65 2.44 1.165 0 0 1.57079632679",
    "RGOCG16-50_16mm": "-8.65 2.72 1.165 0 0 1.57079632679",
}
RECOVERY_GEAR_POSES = {
    "gear_small": "0.44 -0.58 1.11 0 0 0",
    "gear_medium": "0.44 -0.50 1.11 0 0 0",
    "gear_large": "0.44 -0.42 1.11 0 0 0",
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


@pytest.fixture(
    params=(WORLD_PATH, RECOVERY_WORLD_PATH),
    ids=lambda path: path.name,
)
def world_path(request: pytest.FixtureRequest) -> Path:
    """Run the NIST scene checks against each independently selectable world."""
    return request.param


def _world(world_path: Path) -> ET.Element:
    world = ET.parse(world_path).getroot().find("world")
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


def test_plate_uses_exact_mesh_pose_scale_and_collision(world_path: Path) -> None:
    plate = _models_by_name(_world(world_path))["GMC_Laser_Plate_Virtual"]

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


def test_installed_gear_fixture_uses_exact_cad_and_collisions(world_path: Path) -> None:
    world = _world(world_path)
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


def test_gear_fixture_surfaces_and_cad_centers_align(world_path: Path) -> None:
    models = _models_by_name(_world(world_path))
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


def test_pin_models_use_exact_stls_collisions_poses_and_neutral_material(world_path: Path) -> None:
    models = _models_by_name(_world(world_path))

    for name, (pose, box_size) in KET_MODELS.items():
        model = models[name]
        expected_pose = RECOVERY_KET_POSES[name] if world_path == RECOVERY_WORLD_PATH else pose
        assert _required_text(model, "pose") == expected_pose
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
        expected_pose = RECOVERY_RGOCG_POSES[name] if world_path == RECOVERY_WORLD_PATH else pose
        assert _required_text(model, "pose") == expected_pose
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


def test_printers_gears_and_camera_configuration(world_path: Path) -> None:
    world = _world(world_path)
    models = _models_by_name(world)
    includes = _includes_by_name(world)

    printer_poses = {"prusa_mk4_2": "0.50 -0.50 1.04 0 0 -1.57079632679"}
    if world_path == WORLD_PATH:
        printer_poses.update({
            "prusa_mk4_2": "0.4 -0.3 1.04 0 0 0",
            "prusa_mk3": "-0.4 0 1.04 0 0 0",
            "prusa_mk4_1": "0.4 0.3 1.04 0 0 0",
        })
    if world_path == WORLD_PATH:
        for name, pose in printer_poses.items():
            assert _required_text(models[name], "pose") == pose
        for name, pose in GEAR_POSES.items():
            assert _required_text(includes[name], "uri") == f"model://{name}"
            assert _required_text(includes[name], "pose") == pose
    else:
        assert _required_text(includes["prusa_mk4_2"], "uri") == "model://prusa_mk4_2"
        assert _required_text(includes["prusa_mk4_2"], "pose") == printer_poses["prusa_mk4_2"]
        for name, pose in RECOVERY_GEAR_POSES.items():
            assert _required_text(includes[name], "uri") == f"model://{name}"
            assert _required_text(includes[name], "pose") == pose
        assert {"prusa_mk3", "prusa_mk4_1"}.isdisjoint(models)
        assert _required_text(models["Exit"], "pose") == "0.50 0.58 1.04 0 0 0"

    cameras = {"cam_mk3", "cam_mk4_1", "cam_mk4_2", "cam_assembly"}
    if world_path == RECOVERY_WORLD_PATH:
        cameras = {"cam_storage", "cam_mk4_2", "cam_assembly"}
    camera_far_m = {
        "cam_mk3": "5.0",
        "cam_mk4_1": "5.0",
        "cam_mk4_2": "5.0",
        "cam_storage": "5.0",
        "cam_assembly": "2.0",
    }
    camera_min_depth_m = {
        "cam_mk3": "0.05",
        "cam_mk4_1": "0.05",
        "cam_mk4_2": "0.05",
        "cam_storage": "0.05",
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
        if model_name != "gear_medium":
            assert _required_text(model, "link/collision/geometry/cylinder/radius") == radius
            assert _required_text(model, "link/collision/geometry/cylinder/length") == length
        else:
            assert len(model.findall("link/collision")) == 64
            assert model.find("link/collision/geometry/cylinder") is None
        assert (
            _required_text(model, "link/visual/geometry/mesh/uri")
            == f"model://{model_name}/meshes/{filename}"
        )
        assert _required_text(model, "link/visual/geometry/mesh/scale") == (
            "0.001 0.001 0.001"
        )


def test_dedicated_world_has_no_mock_pin_models(world_path: Path) -> None:
    source = world_path.read_text(encoding="utf-8")

    assert "rect_pin_" not in source
    assert "circ_pin_" not in source
    assert "assembly_board_v1" not in source


def test_all_nist_mesh_references_resolve_without_copies(world_path: Path) -> None:
    world = _world(world_path)
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


def test_nist_world_preserves_state_and_attachment_plugins(world_path: Path) -> None:
    """Keep evaluator feedback and acknowledged attachment in both NIST scenes."""
    world = _world(world_path)
    plugins = {plugin.attrib["name"]: plugin for plugin in world.findall("plugin")}

    state = plugins["gazebo_ros_state"]
    assert state.attrib["filename"] == "libgazebo_ros_state.so"
    assert _required_text(state, "ros/namespace") == "/"
    assert _required_text(state, "update_rate") == "50"
    assert plugins["gazebo_link_attacher"].attrib["filename"] == "libgazebo_link_attacher.so"


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
        "world_file:=table_recovery_framework.world run_perception:=false "
        "include_assembly_parts:=true include_loose_parts:=true"
    )

    for name, world_file, description in (
        ("gazebo_dual", "table_recovery_framework.world", "recovery framework NIST world"),
        ("gazebo_dual_spec2primitives", "table_spec2primitives.world", "Spec2Primitives NIST world"),
    ):
        required_paths = ros2_launch_required_paths(
            name,
            venv_python=Path("/tmp/spec2primitives-venv"),
            ur5e_rg2_gripper_script=Path("/tmp/spec2primitives-gripper.py"),
            ur5e_rtde_trajectory_script=Path(
                "/tmp/spec2primitives/config/runtime/trajectory.py"
            ),
        )
        worlds = [(path, error) for path, error in required_paths if path.suffix == ".world"]
        assert len(worlds) == 1
        path, error = worlds[0]
        assert path.parts[-3:] == ("cais_lab_robotics", "worlds", world_file)
        assert description in error
        assert "make bootstrap-gazebo" in error


@pytest.mark.parametrize("missing_world", [
    "table_recovery_framework.world", "table_spec2primitives.world",
])
def test_missing_nist_world_blocks_only_its_launch(
    monkeypatch: pytest.MonkeyPatch, missing_world: str,
) -> None:
    """A missing selected world fails preflight even when other assets exist."""
    monkeypatch.setattr(Path, "is_file", lambda path: True)
    monkeypatch.setattr(Path, "exists", lambda path: path.name != missing_world)
    for name, world_file in (
        ("gazebo_dual", "table_recovery_framework.world"),
        ("gazebo_dual_spec2primitives", "table_spec2primitives.world"),
    ):
        error = ros2_launch_prereq_error(
            name,
            gazebo_workspace_launch_files={name: "dual_moveit_gazebo.launch.py"},
            venv_python=Path("/tmp/spec2primitives-venv"),
            ur5e_rg2_gripper_script=Path("/tmp/spec2primitives-gripper.py"),
            ur5e_rtde_trajectory_script=Path(
                "/tmp/spec2primitives/config/runtime/trajectory.py"
            ),
        )
        if world_file == missing_world:
            assert error is not None
            assert missing_world in error
            assert "make bootstrap-gazebo" in error
        else:
            assert error is None


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
        assert "default_value='table_recovery_framework.world'" in source
    assert "'world_file': world_file" in dual_launch
    assert "LaunchConfiguration('world_file').perform(context)" in xarm_launch
    assert "' -p world_file:=', str(gazebo_world)" in xarm_launch


@pytest.mark.parametrize("include_assembly_parts,include_loose_parts,include_printers", [
    (True, True, True), (True, False, True), (True, False, False), (False, True, True),
])
def test_nist_world_filtering_preserves_passive_scene_options(
    world_path: Path, include_assembly_parts: bool, include_loose_parts: bool,
    include_printers: bool,
) -> None:
    """Remove NIST loose parts and fixtures when the selected launch disables them."""
    launch = REPOSITORY_ROOT / "ros2/cais_lab_robotics/launch/xarm6_ur5e_gazebo.launch.py"
    module = ast.parse(launch.read_text())
    constants = {
        "ASSEMBLY_PART_MODELS", "LOOSE_PART_MODELS",
        "PRUSA_PRINTERS_AND_ASSEMBLY_BOARD_MODELS", "GAZEBO_TABLE_SURFACE_Z_M",
    }
    nodes = [node for node in module.body if (
        isinstance(node, ast.FunctionDef) and node.name == "_filtered_world"
    ) or (
        isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id in constants for target in node.targets)
    )]
    scope = {"ET": ET, "tempfile": tempfile}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(launch), "exec"), scope)
    output = Path(scope["_filtered_world"](
        world_path, include_assembly_parts=include_assembly_parts,
        include_loose_parts=include_loose_parts,
        include_prusa_printers_and_assembly_board=include_printers,
    ))
    try:
        world = _world(output)
    finally:
        output.unlink()
    names = set(_models_by_name(world)) | set(_includes_by_name(world))
    original = _world(world_path)
    expected = set(_models_by_name(original)) | set(_includes_by_name(original))
    if not include_assembly_parts:
        expected -= scope["ASSEMBLY_PART_MODELS"]
    else:
        if not include_loose_parts:
            expected -= {*KET_MODELS, *RGOCG_MODELS, *GEAR_POSES}
        if not include_printers:
            expected -= {
                "GMC_Laser_Plate_Virtual", "Gear_Plate",
                "prusa_mk3", "prusa_mk4_1", "prusa_mk4_2",
            }
    assert names == expected
    assert {include.findtext("uri") for include in world.findall("include")} >= {
        "model://ground_plane", "model://sun",
    }


def test_auto_link_attacher_reads_nist_poses_without_mock_fallbacks(
    world_path: Path, tmp_path: Path,
) -> None:
    """Read the selected world's initial parts and respect an empty scene."""
    launch = REPOSITORY_ROOT / "ros2/cais_lab_robotics/launch/auto_link_attacher_node.py"
    module = ast.parse(launch.read_text())
    names = next(node for node in module.body if isinstance(node, ast.Assign)
                 and any(isinstance(target, ast.Name) and target.id == "PART_NAMES"
                         for target in node.targets))
    cls = next(node for node in module.body if isinstance(node, ast.ClassDef)
               and node.name == "AutoLinkAttacher")
    function = next(node for node in cls.body if isinstance(node, ast.FunctionDef)
                    and node.name == "_load_part_positions")
    scope = {"ET": ET, "Path": Path}
    exec(compile(ast.Module(body=[names, function], type_ignores=[]), str(launch), "exec"), scope)
    warnings = []
    parameter = SimpleNamespace(value=str(world_path))
    node = SimpleNamespace(
        get_parameter=lambda name: parameter,
        get_logger=lambda: SimpleNamespace(warn=warnings.append),
    )
    positions = scope[function.name](node)
    if world_path == RECOVERY_WORLD_PATH:
        expected_poses = {
            **RECOVERY_KET_POSES,
            **RECOVERY_RGOCG_POSES,
            **RECOVERY_GEAR_POSES,
        }
    else:
        expected_poses = {
            **{name: values[0] for name, values in KET_MODELS.items()},
            **{name: values[0] for name, values in RGOCG_MODELS.items()},
            **GEAR_POSES,
        }
    assert set(positions) == set(expected_poses)
    for name, pose in expected_poses.items():
        assert positions[name] == tuple(float(value) for value in pose.split()[:3])
    assert warnings == []

    empty_world = tmp_path / "empty.world"
    empty_world.write_text('<sdf version="1.4"><world name="default"/></sdf>')
    parameter.value = str(empty_world)
    assert scope[function.name](node) == {}
    parameter.value = str(tmp_path / "missing.world")
    assert scope[function.name](node) == {}
    assert len(warnings) == 1


def test_recovery_world_is_used_by_product_and_robot_geometry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resolve NIST geometry from the surviving default world without ROS calls."""
    from cais_spade_llm.product import profile
    from cais_spade_llm.resources.robot import gazebo_pick_place_controller

    monkeypatch.delenv("CAIS_GAZEBO_WORLD_PATH", raising=False)
    monkeypatch.delenv("CAIS_GAZEBO_WORLD_FILE", raising=False)
    assert profile._gazebo_world_path() == WORLD_PATH.with_name("table_recovery_framework.world")
    pose = profile._load_gazebo_model_spawn_pose(
        model_name="KET4_Square_4mm", world_path=str(profile._gazebo_world_path()),
    )
    assert pose == {"x": -8.95, "y": 1.88, "z": 1.165}
    width = gazebo_pick_place_controller._model_footprint_width_from_gazebo_world("KET8_Square_8mm")
    assert width == 0.008


def test_spec2primitives_navigation_and_route_pass_the_composed_runtime() -> None:
    app_source = (
        REPOSITORY_ROOT / "cais_spade_llm" / "ui" / "app.py"
    ).read_text(encoding="utf-8")

    assert '("Spec2Primitives", "/spec2primitives", "account_tree")' in app_source
    assert '@ui.page("/spec2primitives")' in app_source
    assert (
        "from cais_spade_llm.spec2primitives import spec2primitives_ui"
        in app_source
    )
    assert "create_spec2primitives_ui_runtime" in app_source
    assert "spec2primitives_runtime = create_spec2primitives_ui_runtime(bridge)" in app_source
    assert "spec2primitives_ui.render(spec2primitives_runtime)" in app_source
    assert "await bridge.shutdown_spec2primitives_robot_agent()" in app_source


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


def test_gear_medium_compound_collision_preserves_the_cad_opening_and_contact_settings() -> None:
    import numpy as np
    from scipy.spatial import ConvexHull

    directory = REPOSITORY_ROOT / "ros2/cais_lab_robotics/models/gear_medium"
    model = ET.parse(directory / "model.sdf").getroot().find("model")
    assert model is not None
    vertices = np.array([[float(v) for v in line.split()[1:]]
                         for line in (directory / "meshes/Gear_Medium_collision_segment.stl").read_text().splitlines()
                         if line.strip().startswith("vertex ")])
    triangles = vertices.reshape(-1, 3, 3)
    unique = np.unique(vertices, axis=0)
    assert len(unique) == 8
    assert ConvexHull(unique).volume > 0
    assert np.isclose(np.ptp(vertices[:, 2]), .02, atol=1e-8)
    inner = np.linalg.norm(vertices[:, :2], axis=1).min()
    assert .00498 < inner < .005  # The through wall, not the 5.3 mm entry chamfer.
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    assert np.all(np.sum(normals * (triangles.mean(axis=1) - unique.mean(axis=0)), axis=1) > 0)
    for index, collision in enumerate(model.findall("link/collision")):
        pose = [float(v) for v in _required_text(collision, "pose").split()]
        assert np.allclose(pose[:5], 0)
        assert math.isclose(pose[5], 2 * math.pi * index / 64, abs_tol=1e-14)
        assert _required_text(collision, "geometry/mesh/uri") == "model://gear_medium/meshes/Gear_Medium_collision_segment.stl"
        assert _required_text(collision, "surface/friction/ode/mu") == "0.5"
        assert _required_text(collision, "surface/contact/ode/kp") == "20000.0"
        assert _required_text(collision, "surface/contact/ode/kd") == "8.0"
        assert _required_text(collision, "surface/contact/ode/max_vel") == "0.05"
        assert _required_text(collision, "surface/contact/ode/min_depth") == "0.0005"
    assert _required_text(model, "link/inertial/mass") == "0.10"
    assert _required_text(model, "link/visual/pose") == "-0.21316049955 0.18188829805 0.0697159157 3.141592653589793 0 0"
