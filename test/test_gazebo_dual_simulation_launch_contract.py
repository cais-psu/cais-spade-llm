from __future__ import annotations

from pathlib import Path

from cais_spade_llm.ui import ros2_processes

ROOT = Path(__file__).resolve().parents[1]


def _launch_commands() -> dict[str, str]:
    return ros2_processes.build_ros2_launch_cmds(
        project_root=ROOT,
        venv_python=ROOT / ".venv" / "bin" / "python",
        ur5e_rg2_gripper_script=(
            ROOT / "ros2" / "cais_lab_robotics" / "scripts" / "ur5e_rg2_rtde_gripper.py"
        ),
        ur5e_rtde_trajectory_script=(
            ROOT / "ros2" / "cais_lab_robotics" / "scripts" / "ur5e_rtde_trajectory_server.py"
        ),
        ur5e_rtde_trajectory_status=ROOT / "tmp" / "ur5e_rtde_status.json",
    )


def test_gazebo_dual_normal_launch_is_no_hardware_dual_moveit_rviz() -> None:
    commands = _launch_commands()
    rendered = ros2_processes.render_ros2_launch_cmd(
        commands,
        hardware_ips={},
        hw_ip_defaults={"xarm6": "192.168.1.1", "ur5e": "192.168.1.2"},
        name="gazebo_dual",
        fast_forward_simulation=False,
    )

    assert "dual_moveit_gazebo.launch.py" in rendered
    assert "launch_rviz:=false" not in rendered
    assert "fast_sim:=true" not in rendered
    assert "hardware_moveit.launch.py" not in rendered
    assert "dual_robots_hardware_moveit.launch.py" not in rendered
    assert "ur5e_rtde_trajectory_server.py" not in rendered
    assert "ur5e_rg2_rtde_gripper.py" not in rendered


def test_gazebo_dual_fast_forward_launch_stays_headless() -> None:
    commands = _launch_commands()
    rendered = ros2_processes.render_ros2_launch_cmd(
        commands,
        hardware_ips={},
        hw_ip_defaults={"xarm6": "192.168.1.1", "ur5e": "192.168.1.2"},
        name="gazebo_dual",
        fast_forward_simulation=True,
    )

    assert "dual_moveit_gazebo.launch.py" in rendered
    assert "fast_sim:=true" in rendered
    assert "launch_rviz:=false" in rendered


def test_gazebo_dual_prerequisites_include_dual_rviz_and_controller_assets() -> None:
    required_paths = ros2_processes.ros2_launch_required_paths(
        "gazebo_dual",
        venv_python=ROOT / ".venv" / "bin" / "python",
        ur5e_rg2_gripper_script=(
            ROOT / "ros2" / "cais_lab_robotics" / "scripts" / "ur5e_rg2_rtde_gripper.py"
        ),
        ur5e_rtde_trajectory_script=(
            ROOT / "ros2" / "cais_lab_robotics" / "scripts" / "ur5e_rtde_trajectory_server.py"
        ),
    )
    required_path_text = {str(path) for path, _message in required_paths}

    assert any(path.endswith("/rviz/dual_moveit.rviz") for path in required_path_text)
    assert any(
        path.endswith("/config/gazebo_ros2_control/xarm6_ur5e_gazebo_ros2_control_controllers.yaml")
        for path in required_path_text
    )
    assert any(
        path.endswith(
            "/config/gazebo_initial_joint_positions/ur5e_gazebo_initial_joint_positions.yaml"
        )
        for path in required_path_text
    )


def test_gazebo_dual_rviz_and_launch_expose_dual_robots_group() -> None:
    rviz_text = (ROOT / "ros2" / "cais_lab_robotics" / "rviz" / "dual_moveit.rviz").read_text()
    launch_text = (
        ROOT / "ros2" / "cais_lab_robotics" / "launch" / "dual_moveit_gazebo.launch.py"
    ).read_text()

    assert "Planning Group: dual_robots" in rviz_text
    assert "'name': 'dual_robots'" in launch_text
    assert "f'{xarm_prefix}xarm6'" in launch_text
    assert "f'{ur5e_prefix}ur_manipulator'" in launch_text
    assert "ompl['move_group']['dual_robots']" in launch_text
