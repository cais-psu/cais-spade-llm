"""Contract tests for the `cais_lab_robotics` ROS2 package boundary."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from xml.etree import ElementTree as ET

import yaml

from cais_spade_llm.ui.ros2_processes import build_ros2_launch_cmds

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT / "ros2" / "cais_lab_robotics"


def _realsense_launch_module():
    path = PACKAGE_ROOT / "launch" / "realsense_camera.launch.py"
    spec = importlib.util.spec_from_file_location("cais_realsense_camera_launch_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cais_lab_robotics_is_an_ament_cmake_package() -> None:
    package_root = ET.parse(PACKAGE_ROOT / "package.xml").getroot()
    assert package_root.findtext("name") == "cais_lab_robotics"
    assert package_root.findtext("export/build_type") == "ament_cmake"
    assert "xarm_gazebo" in {
        dependency.text for dependency in package_root.findall("exec_depend")
    }

    cmake = (PACKAGE_ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    for asset_directory in (
        "cad_models",
        "config",
        "launch",
        "models",
        "rviz",
        "scripts",
        "sensor",
        "worlds",
    ):
        assert asset_directory in cmake


def test_cais_lab_robotics_generates_direct_ur5e_cartesian_action() -> None:
    action_path = PACKAGE_ROOT / "action" / "MoveUR5eCartesian.action"
    action = action_path.read_text(encoding="utf-8")
    cmake = (PACKAGE_ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    package = (PACKAGE_ROOT / "package.xml").read_text(encoding="utf-8")

    assert "geometry_msgs/PoseStamped target_tool0_pose" in action
    assert "float64 final_position_error_m" in action
    assert "float64 final_orientation_error_rad" in action
    assert '"action/MoveUR5eCartesian.action"' in cmake
    assert "rosidl_generate_interfaces" in cmake
    assert "rosidl_default_generators" in package


def test_cais_lab_robotics_generates_translation_only_ur5e_jog_interfaces() -> None:
    relative_action = (
        PACKAGE_ROOT / "action" / "MoveUR5eRelativeCartesian.action"
    ).read_text(encoding="utf-8")
    jog_service = (
        PACKAGE_ROOT / "srv" / "SetUR5eCartesianJog.srv"
    ).read_text(encoding="utf-8")
    joint_jog_action = (
        PACKAGE_ROOT / "action" / "MoveUR5eJointJog.action"
    ).read_text(encoding="utf-8")
    cmake = (PACKAGE_ROOT / "CMakeLists.txt").read_text(encoding="utf-8")

    assert "geometry_msgs/Vector3 world_translation_m" in relative_action
    assert "float64 final_orientation_drift_rad" in relative_action
    assert "geometry_msgs/Vector3 world_linear_velocity_m_s" in jog_service
    assert "bool stop" in jog_service
    assert "int32 joint" in joint_jog_action
    assert "float64 speed_rad_s" in joint_jog_action
    assert "bool state_uncertain" in joint_jog_action
    assert '"action/MoveUR5eRelativeCartesian.action"' in cmake
    assert '"action/MoveUR5eJointJog.action"' in cmake
    assert '"srv/SetUR5eCartesianJog.srv"' in cmake


def test_project_launch_assets_resolve_from_cais_lab_robotics() -> None:
    launch_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((PACKAGE_ROOT / "launch").glob("*.py"))
    )
    assert "get_package_share_directory('xarm_gazebo')" not in launch_text
    assert 'get_package_share_directory("xarm_gazebo")' not in launch_text
    assert "FindPackageShare('xarm_gazebo')" not in launch_text
    assert 'FindPackageShare("xarm_gazebo")' not in launch_text
    assert "xarm_gazebo_grasp_fix" in launch_text
    assert "GAZEBO_MODEL_PATH" in launch_text


def test_ui_launch_commands_use_cais_lab_robotics() -> None:
    commands = build_ros2_launch_cmds(
        project_root=ROOT,
        venv_python=ROOT / ".venv" / "bin" / "python",
        ur5e_rg2_gripper_script=PACKAGE_ROOT / "scripts" / "ur5e_rg2_rtde_gripper.py",
        ur5e_rtde_trajectory_script=(
            PACKAGE_ROOT / "scripts" / "ur5e_rtde_trajectory_server.py"
        ),
        ur5e_rtde_trajectory_status=Path("/tmp/cais_ur5e_rtde_trajectory_status.json"),
    )
    ros2_launch_commands = [command for command in commands.values() if "ros2 launch" in command]
    assert ros2_launch_commands
    assert all("ros2 launch cais_lab_robotics " in command for command in ros2_launch_commands)
    assert all("ros2 launch xarm_gazebo " not in command for command in ros2_launch_commands)
    assert "include_prusa_printers_and_assembly_board:=false" in commands[
        "gazebo_dual_passive"
    ]
    assert "include_prusa_printers_and_assembly_board:=false" not in commands[
        "gazebo_dual"
    ]


def test_realsense_serials_remain_strings_in_generated_parameter_yaml() -> None:
    launch = _realsense_launch_module()

    assert launch._serial_no_for_driver("048522073304") == "_048522073304"
    assert launch._serial_no_for_driver("103422070738") == "_103422070738"
    assert launch._serial_no_for_driver("") == ""
    assert launch._serial_no_for_driver("_048522073304") == "_048522073304"
    parsed = yaml.safe_load(
        f"serial_no: {launch._serial_no_for_driver('048522073304')}"
    )
    assert parsed["serial_no"] == "_048522073304"
    assert isinstance(parsed["serial_no"], str)


def test_bootstrap_registers_package_without_overlaying_xarm_gazebo() -> None:
    bootstrap = (ROOT / "scripts" / "bootstrap_gazebo_workspace.sh").read_text(
        encoding="utf-8"
    )
    assert 'CAIS_PACKAGE_LINK="${ROS2_WS}/src/cais_lab_robotics"' in bootstrap
    assert 'ln -s "${CAIS_PACKAGE_SOURCE}" "${CAIS_PACKAGE_LINK}"' in bootstrap
    assert 'ros2 launch cais_lab_robotics dual_moveit_gazebo.launch.py' in bootstrap
    assert 'STALE_FAST_WORLD="${ROS2_WS}/install/cais_lab_robotics' in bootstrap
    assert 'copy_glob_if_not_same "${REPO_ROOT}/ros2/cais_lab_robotics' not in bootstrap


def test_standard_table_world_is_the_only_dual_world_profile() -> None:
    worlds = PACKAGE_ROOT / "worlds"
    table_world = worlds / "table.world"
    removed_world = worlds / ("table_" + "fast.world")

    assert table_world.is_file()
    assert not removed_world.exists()
    assert "<real_time_update_rate>1000</real_time_update_rate>" in table_world.read_text(
        encoding="utf-8"
    )

    runtime_paths = [
        ROOT / "cais_spade_llm" / "ui" / "bridge.py",
        ROOT / "cais_spade_llm" / "ui" / "pages" / "dashboard.py",
        ROOT / "cais_spade_llm" / "ui" / "ros2_processes.py",
        PACKAGE_ROOT / "launch" / "dual_moveit_gazebo.launch.py",
        PACKAGE_ROOT / "launch" / "xarm6_ur5e_gazebo.launch.py",
    ]
    runtime_text = "\n".join(path.read_text(encoding="utf-8") for path in runtime_paths)
    removed_tokens = (
        "table_" + "fast.world",
        "fast_" + "sim",
        "fast_" + "forward_simulation",
        "Fast Forward " + "Simulation",
    )
    assert all(token not in runtime_text for token in removed_tokens)


def test_beginner_docs_describe_the_installed_cais_package() -> None:
    doc_paths = [
        ROOT / "README.md",
        ROOT / "ros2" / "ROS2_SETUP_README.md",
        ROOT / "ros2" / "docs" / "ros2_setup_from_scratch.md",
    ]
    docs = [path.read_text(encoding="utf-8") for path in doc_paths]

    for text in docs:
        assert "make bootstrap-gazebo" in text
        assert "source ~/ros2_ws/install/setup.bash" in text
        assert "ros2 pkg prefix cais_lab_robotics" in text
        assert "ros2 pkg prefix xarm_gazebo" in text

    combined = "\n".join(docs)
    assert "copy into xarm_ros2" not in combined
    assert "copies this repo's `ros2/cais_lab_robotics/worlds/*.world`" not in combined
    assert "copies config and RViz assets into the installed `xarm_gazebo`" not in combined
