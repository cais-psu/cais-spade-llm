"""Pure ROS2 launch command, domain, and workspace path helpers."""

from __future__ import annotations

import os
import shlex
from pathlib import Path
from typing import Any

_CAIS_ROS_LOG_DIR = Path(__file__).resolve().parents[1] / "log" / "ros"

# The non-headless OpenCV wheel points Qt at its bundled plugins when imported.
# ROS GUI clients use the system Qt installation and abort if they inherit those paths.
ROS2_ENV = (
    "unset VIRTUAL_ENV PYTHONHOME QT_QPA_PLATFORM_PLUGIN_PATH QT_QPA_FONTDIR; "
    "export PATH=/usr/bin:/usr/local/bin:$PATH; "
    f"mkdir -p {shlex.quote(str(_CAIS_ROS_LOG_DIR))}; "
    f"export ROS_LOG_DIR={shlex.quote(str(_CAIS_ROS_LOG_DIR))}; "
    "source /opt/ros/humble/setup.bash && "
    "source $HOME/ros2_ws/install/setup.bash && "
)


def teleop_script_path(project_root: Path) -> Path:
    return Path(project_root) / "ros2" / "cais_lab_robotics" / "scripts" / "keyboard_teleop.py"


def dual_drag_markers_script_path(project_root: Path) -> Path:
    return Path(project_root) / "ros2" / "cais_lab_robotics" / "scripts" / "dual_drag_markers.py"


def digital_twin_sync_script_path(project_root: Path) -> Path:
    return Path(project_root) / "ros2" / "cais_lab_robotics" / "scripts" / "digital_twin_sync.py"


def physical_part_twin_sync_script_path(project_root: Path) -> Path:
    return (
        Path(project_root)
        / "ros2"
        / "cais_lab_robotics"
        / "scripts"
        / "physical_part_twin_sync.py"
    )


def hardware_arms_config_path(project_root: Path) -> Path:
    """Return the checked-in ROS2 hardware runtime config path."""
    return (
        Path(project_root)
        / "ros2"
        / "cais_lab_robotics"
        / "config"
        / "hardware_runtime"
        / "xarm6_ur5e_hardware_runtime.yaml"
    )


def load_hardware_arms_config(project_root: Path) -> dict[str, Any]:
    """Load `xarm6_ur5e_hardware_runtime.yaml` for UI-side process defaults."""
    path = hardware_arms_config_path(project_root)
    try:
        import yaml
    except ImportError:
        return {}
    try:
        with path.open(encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
    except (OSError, TypeError, yaml.YAMLError):
        return {}
    return dict(loaded) if isinstance(loaded, dict) else {}


def hardware_arms_value(
    config: dict[str, Any],
    keys: tuple[str, ...],
    default: Any,
) -> Any:
    """Return a nested value from `xarm6_ur5e_hardware_runtime.yaml` data."""
    current: Any = config
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def hardware_arms_str(
    config: dict[str, Any],
    keys: tuple[str, ...],
    default: str,
) -> str:
    """Return a string value from `xarm6_ur5e_hardware_runtime.yaml` data."""
    value = hardware_arms_value(config, keys, default)
    text = str(value or "").strip()
    return text or str(default)


def hardware_arms_float(
    config: dict[str, Any],
    keys: tuple[str, ...],
    default: float,
) -> float:
    """Return a float value from `xarm6_ur5e_hardware_runtime.yaml` data."""
    value = hardware_arms_value(config, keys, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def hardware_arms_int(
    config: dict[str, Any],
    keys: tuple[str, ...],
    default: int,
) -> int:
    """Return an int value from `xarm6_ur5e_hardware_runtime.yaml` data."""
    value = hardware_arms_value(config, keys, default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def build_ros2_launch_cmds(
    *,
    project_root: Path,
    venv_python: Path,
    ur5e_rg2_gripper_script: Path,
    ur5e_rtde_trajectory_script: Path,
    ur5e_rtde_trajectory_status: Path,
) -> dict[str, str]:
    teleop_script = str(teleop_script_path(project_root))
    hardware_config = load_hardware_arms_config(project_root)
    hardware_config_path = hardware_arms_config_path(project_root)
    perception_script = (
        Path(project_root) / "ros2" / "cais_lab_robotics" / "sensor" / "gazebo_camera_detector.py"
    )
    physical_perception_config = (
        Path(project_root)
        / "ros2"
        / "cais_lab_robotics"
        / "config"
        / "perception"
        / "realsense_roboflow.yaml"
    )
    ur5e_gripper_backend = hardware_arms_str(
        hardware_config,
        ("ur5e", "gripper", "backend"),
        "xmlrpc",
    )
    return {
        "gazebo_dual": (
            "ros2 launch cais_lab_robotics dual_moveit_gazebo.launch.py "
            "run_perception:=false include_assembly_parts:=true include_loose_parts:=true"
        ),
        "gazebo_dual_spec2primitives": (
            "ros2 launch cais_lab_robotics dual_moveit_gazebo.launch.py "
            "world_file:=table_spec2primitives.world run_perception:=false "
            "include_assembly_parts:=true include_loose_parts:=true"
        ),
        "gazebo_dual_gazebo_only": (
            "ros2 launch cais_lab_robotics dual_moveit_gazebo.launch.py "
            "launch_moveit:=false launch_rviz:=false "
            "run_perception:=false include_assembly_parts:=true include_loose_parts:=false"
        ),
        "gazebo_dual_moveit_only": (
            "ros2 launch cais_lab_robotics dual_moveit_gazebo.launch.py "
            "launch_gazebo:=false launch_moveit:=true launch_rviz:=true "
            "run_perception:=false include_assembly_parts:=true include_loose_parts:=false"
        ),
        "gazebo_dual_passive": (
            "ros2 launch cais_lab_robotics xarm6_ur5e_gazebo.launch.py "
            "passive:=true run_perception:=false include_assembly_parts:=true "
            "include_loose_parts:=false "
            "include_prusa_printers_and_assembly_board:=false"
        ),
        "gazebo_xarm6": "ros2 launch cais_lab_robotics xarm6_moveit_single_gazebo.launch.py",
        "gazebo_ur5e": "ros2 launch cais_lab_robotics ur5e_rg2_moveit_gazebo.launch.py",
        "gazebo_xarm6_passive": (
            "ros2 launch cais_lab_robotics xarm6_single_gazebo.launch.py passive:=true"
        ),
        "gazebo_ur5e_passive": (
            "ros2 launch cais_lab_robotics ur5e_rg2_gazebo.launch.py "
            "passive:=true run_perception:=false"
        ),
        "hardware_xarm6_driver": (
            "ros2 launch cais_lab_robotics xarm6_hardware_driver.launch.py robot_ip:={xarm6_ip}"
        ),
        "hardware_xarm6_moveit": (
            "ros2 launch cais_lab_robotics xarm6_hardware_moveit.launch.py "
            "robot_ip:={xarm6_ip} show_rviz:=false launch_rviz:=false"
        ),
        "hardware_ur5e_rtde_trajectory_server": (
            f"{venv_python} {ur5e_rtde_trajectory_script} "
            f"--robot-ip {{ur5e_ip}} --status-file {ur5e_rtde_trajectory_status} "
            f"--config {hardware_config_path}"
        ),
        "hardware_ur5e_rg2_gripper": (
            f"{venv_python} {ur5e_rg2_gripper_script} --robot-ip {{ur5e_ip}} "
            f"--config {hardware_config_path} --backend {ur5e_gripper_backend}"
        ),
        "hardware_ur5e_moveit": (
            "ros2 launch cais_lab_robotics ur5e_rg2_hardware_moveit.launch.py "
            "launch_rviz:=false"
        ),
        "hardware_dual_robots_moveit": (
            "ros2 launch cais_lab_robotics dual_robots_hardware_moveit.launch.py "
            "launch_rviz:=false"
        ),
        "hardware_robot_state_publisher": (
            "ros2 launch cais_lab_robotics "
            "dual_robots_hardware_state_publisher.launch.py"
        ),
        "realsense_camera": "ros2 launch cais_lab_robotics realsense_camera.launch.py",
        "physical_perception": (
            f"{venv_python} -m "
            "cais_spade_llm.resources.sensor.physical.realsense_roboflow_node "
            f"--ros-args --params-file {physical_perception_config}"
        ),
        "physical_part_twin_sync": (
            "ros2 run cais_lab_robotics physical_part_twin_sync.py"
        ),
        "perception": f"python3.10 {perception_script}",
        "teleop_xarm6": f"python3.10 {teleop_script} --robot xarm6",
        "teleop_ur5e": f"python3.10 {teleop_script} --robot ur5e",
    }


def render_ros2_launch_cmd(
    commands: dict[str, str],
    hardware_ips: dict[str, str],
    hw_ip_defaults: dict[str, str],
    name: str,
) -> str:
    cmd = commands[name]
    return cmd.format(
        xarm6_ip=hardware_ips.get("xarm6", hw_ip_defaults["xarm6"]),
        ur5e_ip=hardware_ips.get("ur5e", hw_ip_defaults["ur5e"]),
    )


def env_int(name: str, default: int) -> int:
    raw = str(os.environ.get(name, "")).strip()
    if not raw:
        return int(default)
    try:
        return int(raw)
    except (TypeError, ValueError):
        return int(default)


def default_ros_domain_id() -> int:
    return env_int("ROS_DOMAIN_ID", 0)


def digital_twin_domain_ids(
    *,
    gazebo_default: int,
    hardware_default: int,
    hardware_xarm6_default: int,
    hardware_ur5e_default: int,
) -> dict[str, int]:
    return {
        "gazebo": env_int("CAIS_DIGITAL_TWIN_GAZEBO_DOMAIN_ID", gazebo_default),
        "hardware": env_int("CAIS_DIGITAL_TWIN_HARDWARE_DOMAIN_ID", hardware_default),
        "hardware_xarm6": env_int(
            "CAIS_DIGITAL_TWIN_HARDWARE_XARM6_DOMAIN_ID",
            hardware_xarm6_default,
        ),
        "hardware_ur5e": env_int(
            "CAIS_DIGITAL_TWIN_HARDWARE_UR5E_DOMAIN_ID",
            hardware_ur5e_default,
        ),
    }


def ros2_domain_export(ros_domain_id: int | None) -> str:
    if ros_domain_id is None:
        return ""
    try:
        domain = int(ros_domain_id)
    except (TypeError, ValueError):
        return ""
    return f"export ROS_DOMAIN_ID={domain}; "


def ros2_setup_path() -> Path:
    return Path("/opt/ros/humble/setup.bash")


def ros2_workspace_root() -> Path:
    return Path.home() / "ros2_ws"


def ros2_workspace_setup_path() -> Path:
    return ros2_workspace_root() / "install" / "setup.bash"


def ros2_workspace_launch_dir() -> Path:
    return ros2_workspace_root() / "src" / "cais_lab_robotics" / "launch"


def ros2_workspace_install_pkg_path(pkg_name: str) -> Path:
    return ros2_workspace_root() / "install" / str(pkg_name).strip()


def ros2_workspace_install_share_pkg_path(pkg_name: str) -> Path:
    pkg_name = str(pkg_name).strip()
    return ros2_workspace_install_pkg_path(pkg_name) / "share" / pkg_name


def ros2_system_share_pkg_path(pkg_name: str) -> Path:
    return Path("/opt/ros/humble/share") / str(pkg_name).strip()


def ros2_launch_required_paths(
    name: str,
    *,
    venv_python: Path,
    ur5e_rg2_gripper_script: Path,
    ur5e_rtde_trajectory_script: Path,
) -> list[tuple[Path, str]]:
    launch_key = str(name or "").strip().lower()
    cais_lab_robotics_share = ros2_workspace_install_share_pkg_path("cais_lab_robotics")
    repo_hardware_config = (
        Path(ur5e_rtde_trajectory_script).resolve().parents[1]
        / "config"
        / "hardware_runtime"
        / "xarm6_ur5e_hardware_runtime.yaml"
    )
    hardware_config_asset = (
        cais_lab_robotics_share
        / "config"
        / "hardware_runtime"
        / "xarm6_ur5e_hardware_runtime.yaml",
        "ROS2 workspace is missing the xArm6 + UR5e hardware runtime config. "
        "Re-run `make bootstrap-gazebo`.",
    )
    repo_hardware_config_asset = (
        repo_hardware_config,
        f"xArm6 + UR5e hardware runtime config is missing at {repo_hardware_config}.",
    )
    workspace_cais = (
        (
            ros2_workspace_install_pkg_path("cais_lab_robotics"),
            "ROS2 workspace is missing package 'cais_lab_robotics'. "
            "Re-run `make bootstrap-gazebo`.",
        ),
    )
    workspace_xarm = (
        (
            ros2_workspace_install_pkg_path("xarm_gazebo"),
            "ROS2 workspace is missing package 'xarm_gazebo'. Re-run `make bootstrap-gazebo`.",
        ),
        (
            ros2_workspace_install_pkg_path("xarm_moveit_config"),
            "ROS2 workspace is missing package 'xarm_moveit_config'. "
            "Re-run `make bootstrap-gazebo`.",
        ),
    )
    xarm_description_ws = (
        (
            ros2_workspace_install_pkg_path("xarm_description"),
            "ROS2 workspace is missing package 'xarm_description'. "
            "Re-run `make bootstrap-gazebo`.",
        ),
    )
    moveit_core = (
        (
            ros2_system_share_pkg_path("moveit_ros_move_group"),
            "MoveIt is not installed. Install `ros-humble-moveit`.",
        ),
    )
    ur_stack = (
        (
            ros2_system_share_pkg_path("ur_description"),
            "UR description is not installed. Install `ros-humble-ur-description`.",
        ),
        (
            ros2_system_share_pkg_path("ur_moveit_config"),
            "UR MoveIt config is not installed. Install `ros-humble-ur-moveit-config`.",
        ),
    )
    ur_description = (
        (
            ros2_system_share_pkg_path("ur_description"),
            "UR description is not installed. Install `ros-humble-ur-description`.",
        ),
    )
    onrobot_ws = (
        (
            ros2_workspace_install_pkg_path("onrobot_description"),
            "ROS2 workspace is missing package 'onrobot_description'. "
            "Re-run `make bootstrap-gazebo`.",
        ),
    )
    link_attacher_ws = (
        (
            ros2_workspace_install_pkg_path("linkattacher_msgs"),
            "ROS2 workspace is missing package 'linkattacher_msgs'. "
            "Re-run `make bootstrap-gazebo` to install IFRA LinkAttacher.",
        ),
        (
            ros2_workspace_install_pkg_path("ros2_linkattacher"),
            "ROS2 workspace is missing package 'ros2_linkattacher'. "
            "Re-run `make bootstrap-gazebo` to install IFRA LinkAttacher.",
        ),
    )
    dual_assets = (
        (
            cais_lab_robotics_share
            / "config"
            / "gazebo_ros2_control"
            / "xarm6_ur5e_gazebo_ros2_control_controllers.yaml",
            "ROS2 workspace is missing the dual-robot cais_lab_robotics controller config. "
            "Re-run `make bootstrap-gazebo`.",
        ),
        (
            cais_lab_robotics_share
            / "config"
            / "gazebo_initial_joint_positions"
            / "ur5e_gazebo_initial_joint_positions.yaml",
            "ROS2 workspace is missing the UR5e initial positions config. "
            "Re-run `make bootstrap-gazebo`.",
        ),
        (
            cais_lab_robotics_share / "rviz" / "dual_moveit.rviz",
            "ROS2 workspace is missing the dual-robot RViz config. Re-run `make bootstrap-gazebo`.",
        ),
    )
    spec2primitives_world_asset = (
        (
            cais_lab_robotics_share / "worlds" / "table_spec2primitives.world",
            "ROS2 workspace is missing the Spec2Primitives NIST world. "
            "Re-run `make bootstrap-gazebo`.",
        ),
    )
    dual_launch_assets = {
        "gazebo_dual": dual_assets,
        "gazebo_dual_spec2primitives": (*dual_assets, *spec2primitives_world_asset),
    }
    dual_passive_assets = (
        (
            cais_lab_robotics_share
            / "config"
            / "gazebo_ros2_control"
            / "xarm6_ur5e_gazebo_ros2_control_controllers.yaml",
            "ROS2 workspace is missing the dual-robot cais_lab_robotics controller config. "
            "Re-run `make bootstrap-gazebo`.",
        ),
        (
            cais_lab_robotics_share
            / "config"
            / "gazebo_initial_joint_positions"
            / "ur5e_gazebo_initial_joint_positions.yaml",
            "ROS2 workspace is missing the UR5e initial positions config. "
            "Re-run `make bootstrap-gazebo`.",
        ),
    )
    xarm_hardware_driver_assets = (
        (
            cais_lab_robotics_share / "launch" / "xarm6_hardware_driver.launch.py",
            "ROS2 workspace is missing the xArm6 hardware driver launch file. "
            "Re-run `make bootstrap-gazebo`.",
        ),
    )
    xarm_hardware_moveit_assets = (
        hardware_config_asset,
        (
            cais_lab_robotics_share / "launch" / "xarm6_hardware_moveit.launch.py",
            "ROS2 workspace is missing the xArm6 hardware MoveIt launch file. "
            "Re-run `make bootstrap-gazebo`.",
        ),
    )
    dual_hardware_moveit_assets = (
        hardware_config_asset,
        (
            cais_lab_robotics_share / "launch" / "dual_robots_hardware_moveit.launch.py",
            "ROS2 workspace is missing the dual robots hardware MoveIt launch file. "
            "Re-run `make bootstrap-gazebo`.",
        ),
    )
    hardware_state_publisher_assets = (
        hardware_config_asset,
        (
            cais_lab_robotics_share
            / "launch"
            / "dual_robots_hardware_state_publisher.launch.py",
            "ROS2 workspace is missing the dual robots hardware state publisher launch file. "
            "Re-run `make bootstrap-gazebo`.",
        ),
        (
            cais_lab_robotics_share / "launch" / "dual_robots_hardware_moveit.launch.py",
            "ROS2 workspace is missing the combined hardware model launch file. "
            "Re-run `make bootstrap-gazebo`.",
        ),
    )
    ur_assets = (
        (
            cais_lab_robotics_share
            / "config"
            / "gazebo_ros2_control"
            / "ur5e_rg2_gazebo_ros2_control_controllers.yaml",
            "ROS2 workspace is missing the UR5e RG2 controller config. "
            "Re-run `make bootstrap-gazebo`.",
        ),
    )
    ur_hardware_rg2_assets = (
        hardware_config_asset,
        (
            cais_lab_robotics_share / "launch" / "ur5e_rg2_hardware_moveit.launch.py",
            "ROS2 workspace is missing the UR5e RG2 hardware MoveIt launch file. "
            "Re-run `make bootstrap-gazebo`.",
        ),
    )
    ur_rg2_bridge_assets = (
        (
            venv_python,
            f"Python venv is missing at {venv_python}. "
            "Create the project venv before starting the UR5e RG2 bridge.",
        ),
        (
            ur5e_rg2_gripper_script,
            f"UR5e RG2 bridge script is missing at {ur5e_rg2_gripper_script}.",
        ),
        repo_hardware_config_asset,
    )
    ur_rtde_trajectory_assets = (
        (
            venv_python,
            f"Python venv is missing at {venv_python}. "
            "Create the project venv before starting the UR5e RTDE trajectory server.",
        ),
        (
            ur5e_rtde_trajectory_script,
            f"UR5e RTDE trajectory server script is missing at {ur5e_rtde_trajectory_script}.",
        ),
        repo_hardware_config_asset,
    )

    selected_dual_assets = dual_launch_assets.get(launch_key)
    if selected_dual_assets is not None:
        return [
            *workspace_cais,
            *workspace_xarm,
            *moveit_core,
            *ur_stack,
            *onrobot_ws,
            *link_attacher_ws,
            *selected_dual_assets,
        ]
    if launch_key == "gazebo_dual_passive":
        return [
            *workspace_cais,
            *workspace_xarm,
            *ur_stack,
            *onrobot_ws,
            *dual_passive_assets,
        ]
    if launch_key == "gazebo_xarm6":
        return [*workspace_cais, *workspace_xarm, *moveit_core, *link_attacher_ws]
    if launch_key == "gazebo_ur5e":
        return [
            *workspace_cais,
            *workspace_xarm,
            *moveit_core,
            *ur_stack,
            *onrobot_ws,
            *link_attacher_ws,
            *ur_assets,
        ]
    if launch_key == "gazebo_xarm6_passive":
        return [*workspace_cais, *workspace_xarm]
    if launch_key == "gazebo_ur5e_passive":
        return [*workspace_cais, *workspace_xarm, *ur_stack, *onrobot_ws, *ur_assets]
    if launch_key == "hardware_xarm6_driver":
        return [*workspace_cais, *workspace_xarm, *xarm_hardware_driver_assets]
    if launch_key == "hardware_xarm6_moveit":
        return [
            *workspace_cais,
            *workspace_xarm,
            *moveit_core,
            *xarm_hardware_moveit_assets,
        ]
    if launch_key == "hardware_dual_robots_moveit":
        return [
            *workspace_cais,
            *workspace_xarm,
            *moveit_core,
            *ur_stack,
            *onrobot_ws,
            *dual_hardware_moveit_assets,
        ]
    if launch_key == "hardware_robot_state_publisher":
        return [
            *workspace_cais,
            *xarm_description_ws,
            *ur_description,
            *onrobot_ws,
            *hardware_state_publisher_assets,
        ]
    if launch_key == "hardware_ur5e_moveit":
        return [
            *workspace_cais,
            *moveit_core,
            *ur_stack,
            *onrobot_ws,
            *ur_hardware_rg2_assets,
        ]
    if launch_key == "hardware_ur5e_rg2_gripper":
        return [*ur_rg2_bridge_assets]
    if launch_key == "hardware_ur5e_rtde_trajectory_server":
        return [*ur_rtde_trajectory_assets]
    if launch_key == "realsense_camera":
        return [
            *workspace_cais,
            (
                ros2_system_share_pkg_path("realsense2_camera"),
                "RealSense ROS is not installed. Install "
                "`sudo apt install ros-humble-realsense2-camera "
                "ros-humble-realsense2-description`.",
            ),
            (
                ros2_system_share_pkg_path("realsense2_description"),
                "RealSense description is not installed. Install "
                "`sudo apt install ros-humble-realsense2-camera "
                "ros-humble-realsense2-description`.",
            ),
        ]
    if launch_key == "physical_perception":
        return [
            *workspace_cais,
            (
                venv_python,
                f"Python venv is missing at {venv_python}. Run `poetry install`.",
            ),
            (
                cais_lab_robotics_share
                / "config"
                / "perception"
                / "realsense_roboflow.yaml",
                "ROS2 workspace is missing the RealSense Roboflow config. "
                "Re-run `make bootstrap-gazebo`.",
            ),
        ]
    if launch_key == "physical_part_twin_sync":
        return [
            *workspace_cais,
            *link_attacher_ws,
            (
                cais_lab_robotics_share / "models" / "gear_small" / "model.sdf",
                "ROS2 workspace is missing the Gazebo gear models. "
                "Re-run `make bootstrap-gazebo`.",
            ),
        ]
    return []


def ros2_launch_prereq_error(
    name: str,
    *,
    gazebo_workspace_launch_files: dict[str, str],
    venv_python: Path,
    ur5e_rg2_gripper_script: Path,
    ur5e_rtde_trajectory_script: Path,
) -> str | None:
    ros_setup = ros2_setup_path()
    if not ros_setup.is_file():
        return (
            f"ROS 2 Humble is not installed: missing {ros_setup}. "
            "Install the README simulation dependencies first."
        )

    ws_setup = ros2_workspace_setup_path()
    if not ws_setup.is_file():
        return (
            f"ROS2 workspace is not built yet: missing {ws_setup}. "
            "Run `make bootstrap-gazebo` after installing the README simulation packages."
        )

    launch_file = gazebo_workspace_launch_files.get(str(name or "").strip().lower())
    if launch_file:
        launch_path = ros2_workspace_launch_dir() / launch_file
        if not launch_path.is_file():
            return (
                f"ROS2 workspace is missing {launch_file} at {launch_path}. "
                "Re-run `make bootstrap-gazebo` to install cais_lab_robotics and rebuild."
            )
    for required_path, remedy in ros2_launch_required_paths(
        name,
        venv_python=venv_python,
        ur5e_rg2_gripper_script=ur5e_rg2_gripper_script,
        ur5e_rtde_trajectory_script=ur5e_rtde_trajectory_script,
    ):
        if not required_path.exists():
            return f"Missing ROS dependency at {required_path}. {remedy}"
    return None
