"""Contract tests for the `cais_lab_robotics` ROS2 package boundary."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from xml.etree import ElementTree as ET

import pytest
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


def test_ur5e_state_publishers_use_exact_cartesian_world_base_translation() -> None:
    runtime = yaml.safe_load(
        (
            PACKAGE_ROOT
            / "config"
            / "hardware_runtime"
            / "xarm6_ur5e_hardware_runtime.yaml"
        ).read_text(encoding="utf-8")
    )
    world_base = runtime["ur5e"]["rtde"]["cartesian_world_base"]

    assert world_base == {
        "x_m": 0.0,
        "y_m": 0.5,
        "z_m": 1.021,
        "roll_rad": 0.0,
        "pitch_rad": 0.0,
        "yaw_rad": 0.0,
    }
    launch_sources = {
        launch_name: (PACKAGE_ROOT / "launch" / launch_name).read_text(
            encoding="utf-8"
        )
        for launch_name in (
            "ur5e_rg2_hardware_moveit.launch.py",
            "dual_robots_hardware_moveit.launch.py",
        )
    }
    for source in launch_sources.values():
        assert "def _required_float(" in source
        assert '("ur5e", "rtde", "cartesian_world_base", field_name)' in source
        assert 'for field_name in ("x_m", "y_m", "z_m")' in source
        for field_name in ("roll_rad", "pitch_rad", "yaw_rad"):
            assert (
                f'("ur5e", "rtde", "cartesian_world_base", "{field_name}")'
                in source
            )
        assert "UR5E_WORLD_BASE_YAW_RAD - math.pi" in source
        assert "UR5E_BASE_RPY" in source
    assert 'origin.set("xyz", UR5E_BASE_XYZ)' in launch_sources[
        "ur5e_rg2_hardware_moveit.launch.py"
    ]
    assert '"xyz": UR5E_BASE_XYZ' in launch_sources[
        "dual_robots_hardware_moveit.launch.py"
    ]
    assert 'origin.set("rpy", UR5E_BASE_RPY)' in launch_sources[
        "ur5e_rg2_hardware_moveit.launch.py"
    ]
    assert '"rpy": UR5E_BASE_RPY' in launch_sources[
        "dual_robots_hardware_moveit.launch.py"
    ]
    assert "3.142" not in launch_sources["ur5e_rg2_hardware_moveit.launch.py"]
    assert (
        '{"xyz": f"0.0 {ROBOT_BASE_Y} 1.021", "rpy": "0 0 3.142"}'
        not in launch_sources["dual_robots_hardware_moveit.launch.py"]
    )


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


def test_cais_lab_robotics_generates_guarded_ur5e_insert_action() -> None:
    action = (PACKAGE_ROOT / "action" / "MoveUR5eInsert.action").read_text(
        encoding="utf-8"
    )
    cmake = (PACKAGE_ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    runtime = yaml.safe_load(
        (
            PACKAGE_ROOT
            / "config"
            / "hardware_runtime"
            / "xarm6_ur5e_hardware_runtime.yaml"
        ).read_text(encoding="utf-8")
    )

    assert action.count("\n---\n") == 2
    assert '"action/MoveUR5eInsert.action"' in cmake
    assert runtime["ur5e"]["hardware_insert_action"] == (
        "/cais_ur5e_rtde_cartesian_controller/move_insert"
    )
    insert_caps = {
        key: value
        for key, value in runtime["ur5e"]["rtde"].items()
        if key.startswith("insert_max_") or key.startswith("insert_start_")
    }
    assert insert_caps
    relief_profile = {
        "insert_max_tool_flange_torque_nm": 3.0,
        "insert_soft_filter_window_sec": 0.05,
        "insert_soft_overload_hold_sec": 0.10,
        "insert_relief_unload_dwell_sec": 0.10,
        "insert_relief_clear_dwell_sec": 0.10,
        "insert_relief_clear_hysteresis_ratio": 0.80,
        "insert_relief_timeout_sec": 1.0,
        "insert_relief_axial_force_ratio": 0.50,
        "insert_relief_reverse_force_ratio": 0.25,
        "insert_relief_resume_ramp_sec": 0.10,
        "insert_relief_search_force_ratio": 0.50,
        "insert_relief_search_speed_ratio": 0.50,
        "insert_relief_backoff_step_m": 0.0001,
        "insert_max_relief_retreat_m": 0.0003,
        "insert_relief_stationary_speed_m_s": 0.0005,
        "insert_relief_stationary_angular_speed_rad_s": 0.01,
    }
    for field, expected in relief_profile.items():
        assert runtime["ur5e"]["rtde"][field] == pytest.approx(expected)
    assert runtime["ur5e"]["rtde"]["insert_max_relief_cycles"] == 3
    assert all(
        isinstance(value, (int, float)) and not isinstance(value, bool)
        for value in insert_caps.values()
    )
    assert insert_caps["insert_max_spiral_radius_m"] == pytest.approx(0.002)
    assert insert_caps["insert_max_insertion_force_n"] == pytest.approx(15.0)
    assert runtime["ur5e"]["rtde"]["MG"] == {
        "insert_max_insertion_force_n": 76.0,
        "insert_max_axial_force_n": 104.0,
        "insert_max_lateral_force_n": 55.0,
        "insert_max_torque_nm": 2.9,
        "insert_max_tool_flange_torque_nm": 7.8,
            "insert_max_relief_retreat_m": 0.0006,
            "insert_max_contact_search_radius_m": 0.01,
            "insert_max_disengagement_cycles": 6,
            "insert_search_peck_retreat_m": 0.003,
            "insert_search_peck_interval_sec": 0.75,
        }


def test_ur5e_insert_action_field_order_is_stable() -> None:
    action_lines = [
        line.strip()
        for line in (PACKAGE_ROOT / "action" / "MoveUR5eInsert.action")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    action_sections: list[list[str]] = [[]]
    for line in action_lines:
        if line == "---":
            action_sections.append([])
        else:
            action_sections[-1].append(line)
    assert action_sections == [[
        "string part_name",
        "string calibration_id",
        "string profile_sha256",
        "string hard_caps_sha256",
        "string trial_id",
        "geometry_msgs/PoseStamped expected_start_tool0_pose",
        "geometry_msgs/PoseStamped target_tool0_pose",
        "geometry_msgs/Vector3 insertion_axis_world",
        "float64 contact_speed_m_s",
        "float64 contact_force_delta_n",
        "float64 engagement_progress_m",
        "float64 insertion_force_n",
        "float64 spiral_radius_m",
        "float64 spiral_pitch_m",
        "float64 spiral_speed_m_s",
        "float64 spiral_acceleration_m_s2",
        "float64 max_axial_force_n",
        "float64 max_lateral_force_n",
        "float64 max_torque_nm",
        "float64[] force_depth_fraction",
        "float64[] force_depth_axial_upper_n",
        "float64[] force_depth_lateral_upper_n",
        "float64[] force_depth_torque_upper_nm",
        "float64 baseline_force_uncertainty_n",
        "float64 baseline_torque_uncertainty_nm",
        "float64 tilt_tolerance_rad",
        "float64 seated_depth_tolerance_m",
        "float64 settle_time_sec",
        "float64 timeout_sec",
    ], [
        "int32 error_code",
        "string error_string",
        "string trial_id",
        "string hard_caps_sha256",
        "bool state_uncertain",
        "bool motion_settled",
        "bool final_tool0_pose_valid",
        "geometry_msgs/PoseStamped final_tool0_pose",
        "float64 final_insertion_depth_m",
        "float64 final_depth_error_m",
        "float64 final_lateral_offset_m",
        "float64 final_tilt_error_rad",
        "float64 final_search_radius_m",
        "float64 peak_axial_force_n",
        "float64 peak_lateral_force_n",
        "float64 peak_torque_nm",
        "float64 peak_filtered_axial_force_n",
        "float64 peak_filtered_lateral_force_n",
        "float64 peak_filtered_torque_nm",
        "float64 peak_tool_flange_torque_nm",
        "bool contact_detected",
        "bool engagement_detected",
        "bool seated_detected",
        "bool force_bias_valid",
        "float64[6] force_bias",
        "string final_phase",
        "bool soft_overload_detected",
        "bool soft_overload_recovered",
        "bool relief_exhausted",
        "int32 relief_cycle_count",
        "string last_soft_overload_reason",
        "bool relief_load_cleared",
        "float64 relief_backoff_m",
        "float64 relief_planned_backoff_m",
        "float64 total_relief_backoff_m",
        "string relief_resume_phase",
        "bool relief_force_mode_stop_acknowledged",
        "bool relief_stop_l_command_completed",
        "bool relief_stationary_confirmed",
        "bool relief_force_mode_restart_acknowledged",
        "bool hard_limit_detected",
        "string hard_limit_reason",
        "string limit_trigger",
        "float64 limit_trigger_value",
        "float64 limit_trigger_threshold",
        "float64[6] limit_trigger_actual_tcp_force",
        "float64[6] limit_trigger_tared_tcp_force",
        "bool force_mode_stop_acknowledged",
        "bool servo_stop_acknowledged",
        "bool stop_l_command_completed",
        "bool stationary_confirmed",
        "string server_trace_id",
        "string server_trace_path",
        "string server_trace_sha256",
        "string server_trace_status",
        "bool server_trace_complete",
        "int32 server_trace_sample_count",
        "bool tactile_center_valid",
        "geometry_msgs/PoseStamped tactile_center_tool0_pose",
        "float64 tactile_center_depth_m",
        "float64 tactile_center_confidence",
        "string tactile_center_evidence_sha256",
        "float64 scheduled_search_radius_m",
        "float64 explored_search_radius_m",
        "float64 explored_search_angle_rad",
        "int32 disengagement_cycle_count",
        "string last_disengagement_reason",
        "float64 disengagement_withdrawal_m",
        "bool disengagement_contact_cleared",
        "bool disengagement_force_mode_stop_acknowledged",
        "float64 recenter_position_error_m",
        "bool recenter_command_acknowledged",
        "bool disengagement_stationary_confirmed",
        "bool retare_baseline_consistent",
    ], [
        "string phase",
        "string trial_id",
        "geometry_msgs/PoseStamped actual_tool0_pose",
        "float64 insertion_depth_m",
        "float64 depth_error_m",
        "float64 lateral_offset_m",
        "float64 search_radius_m",
        "float64 axial_force_n",
        "float64 raw_axial_force_n",
        "float64 lateral_force_n",
        "float64 torque_nm",
        "float64 filtered_axial_force_n",
        "float64 filtered_lateral_force_n",
        "float64 filtered_torque_nm",
        "float64 tool_flange_torque_nm",
        "float64 filtered_tool_flange_torque_nm",
        "float64 current_force_depth_fraction",
        "float64 force_depth_axial_upper_n",
        "float64 force_depth_lateral_upper_n",
        "float64 force_depth_torque_upper_nm",
        "bool axial_profile_exceeded",
        "bool axial_progress_stalled",
        "bool contact_detected",
        "bool engagement_detected",
        "bool seated_detected",
        "bool force_bias_valid",
        "float64[6] force_bias",
        "float64[6] actual_tcp_force",
        "float64[6] tared_tcp_force",
        "float64[6] actual_tcp_speed",
        "bool soft_overload_detected",
        "string soft_overload_reason",
        "float64 soft_overload_duration_sec",
        "int32 relief_cycle_count",
        "float64 relief_elapsed_sec",
        "float64 relief_retreat_m",
        "bool relief_load_cleared",
        "float64 relief_backoff_m",
        "float64 relief_planned_backoff_m",
        "float64 total_relief_backoff_m",
        "string relief_resume_phase",
        "float64 commanded_axial_force_n",
        "float64 commanded_lateral_force_x_n",
        "float64 commanded_lateral_force_y_n",
        "bool hard_limit_detected",
        "string hard_limit_reason",
        "string limit_trigger",
        "float64 limit_trigger_value",
        "float64 limit_trigger_threshold",
        "float64[6] limit_trigger_actual_tcp_force",
        "float64[6] limit_trigger_tared_tcp_force",
        "bool tactile_center_valid",
        "geometry_msgs/PoseStamped tactile_center_tool0_pose",
        "float64 tactile_center_depth_m",
        "float64 tactile_center_confidence",
        "string tactile_center_evidence_sha256",
        "float64 scheduled_search_radius_m",
        "float64 explored_search_radius_m",
        "float64 explored_search_angle_rad",
        "int32 disengagement_cycle_count",
        "string last_disengagement_reason",
        "float64 disengagement_withdrawal_m",
        "bool disengagement_contact_cleared",
        "bool disengagement_force_mode_stop_acknowledged",
        "float64 recenter_position_error_m",
        "bool recenter_command_acknowledged",
        "bool disengagement_stationary_confirmed",
        "bool retare_baseline_consistent",
    ]]


def test_cais_lab_robotics_generates_passive_insertion_demonstration_action() -> None:
    action = (
        PACKAGE_ROOT / "action" / "RecordUR5eInsertionDemonstration.action"
    ).read_text(encoding="utf-8")
    cmake = (PACKAGE_ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    runtime = yaml.safe_load(
        (
            PACKAGE_ROOT
            / "config"
            / "hardware_runtime"
            / "xarm6_ur5e_hardware_runtime.yaml"
        ).read_text(encoding="utf-8")
    )

    assert action.count("\n---\n") == 2
    assert "string recording_id" in action
    assert "geometry_msgs/PoseStamped expected_start_tool0_pose" in action
    assert "float64[6] force_bias" in action
    assert "float64[6] actual_tcp_force" in action
    assert "float64[6] actual_tcp_speed" in action
    assert '"action/RecordUR5eInsertionDemonstration.action"' in cmake
    assert runtime["ur5e"]["hardware_insertion_demonstration_action"] == (
        "/cais_ur5e_rtde_cartesian_controller/record_insertion_demonstration"
    )
    assert runtime["ur5e"]["rtde"]["insert_demonstration_max_duration_sec"] == 300.0


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
