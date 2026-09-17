# ROS2 Setup From Scratch

This guide prepares the three CAIS-SPADE simulation modes on Ubuntu 22.04 with
ROS2 Humble:

1. `ur5e_rg2_moveit_gazebo.launch.py`
2. `xarm6_moveit_single_gazebo.launch.py`
3. `dual_moveit_gazebo.launch.py`

The root `README.md` is the primary full-application installation guide. This
document covers the ROS2 workspace specifically.

## 1. Install the Base Environment

Install Ubuntu 22.04 and ROS2 Humble. For WSL2, follow
`ros2/docs/wsl_ubuntu22_ros2_humble_setup.md` first.

Do not run raw ROS2, Gazebo, MoveIt, or RViz commands inside the Poetry virtual
environment.

```bash
sudo apt update
sudo apt install -y \
  ros-humble-desktop \
  python3-colcon-common-extensions \
  python3-rosdep \
  ros-humble-moveit \
  ros-humble-gazebo-ros-pkgs \
  ros-humble-gazebo-ros2-control \
  ros-humble-ur-description \
  ros-humble-ur-moveit-config \
  ros-humble-controller-manager \
  ros-humble-joint-state-broadcaster \
  ros-humble-joint-trajectory-controller \
  ros-humble-realsense2-camera \
  ros-humble-realsense2-description \
  ros-humble-xacro \
  ros-humble-robot-state-publisher
```

Initialize rosdep once:

```bash
if [ ! -e /etc/ros/rosdep/sources.list.d/20-default.list ]; then
  sudo rosdep init
fi
rosdep update
```

The real UR5e runtime uses the repository's RTDE trajectory server. It does not
require `ur_robot_driver`.

## 2. Clone CAIS-SPADE-LLM

```bash
mkdir -p ~/projects
cd ~/projects
git clone <CAIS-SPADE-LLM-REPOSITORY-URL> cais-spade-llm
cd cais-spade-llm
```

If the repository is already present, run the remaining commands from its root.

## 3. Build `~/ros2_ws`

```bash
make bootstrap-gazebo
```

Bootstrap is the complete workspace setup. It:

- creates or reuses `~/ros2_ws`;
- clones `xarm_ros2`, `OnRobot_ROS2_Description`, and `IFRA_LinkAttacher`;
- registers the tracked `ros2/cais_lab_robotics` package through
  `~/ros2_ws/src/cais_lab_robotics`;
- applies the maintained IFRA link-attacher source patch; and
- builds all required packages with `colcon`.

Do not manually copy project files into `xarm_gazebo`, and do not perform a
second initial build after bootstrap.

## 4. Understand the Workspace

```text
Repository source:
  ~/projects/cais-spade-llm/ros2/cais_lab_robotics/

ROS2 source registration:
  ~/ros2_ws/src/cais_lab_robotics -> repository source

Generated colcon state:
  ~/ros2_ws/build/
  ~/ros2_ws/log/
  ~/ros2_ws/install/
```

Edit the repository source. Never edit generated files under `build` or
`install`. Gazebo, MoveIt, RViz, hardware control, and the digital twin need the
installed workspace; Python-only planning and `dry_run` do not.

## 5. Source and Verify Packages

```bash
deactivate 2>/dev/null || true
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash

ros2 pkg prefix cais_lab_robotics
ros2 pkg prefix xarm_gazebo
ros2 pkg prefix xarm_description
ros2 pkg prefix ur_description
ros2 pkg prefix ur_moveit_config
ros2 pkg prefix onrobot_description
ros2 pkg prefix linkattacher_msgs
ros2 pkg prefix ros2_linkattacher
```

The CAIS, xArm, OnRobot, and IFRA packages should resolve from
`~/ros2_ws/install`. The UR packages should resolve from `/opt/ros/humble`.

## 6. Run One of the Three Simulation Modes

In a terminal where both ROS2 setup files are sourced:

UR5e with RG2:

```bash
ros2 launch cais_lab_robotics ur5e_rg2_moveit_gazebo.launch.py
```

xArm6 with xArm gripper:

```bash
ros2 launch cais_lab_robotics xarm6_moveit_single_gazebo.launch.py
```

Dual robots:

```bash
ros2 launch cais_lab_robotics dual_moveit_gazebo.launch.py
```

The dual simulation defaults to
`ros2/cais_lab_robotics/worlds/table_recovery_framework.world` with NIST components.
Use `world_file:=table_spec2primitives.world` for the separate Spec2Primitives scene.

## 7. Verify the Running Simulation

```bash
ros2 control list_controllers
```

Expected controllers:

- UR5e mode: `ur5e_joint_trajectory_controller` and
  `ur5e_rg2_gripper_traj_controller`.
- xArm6 mode: `xarm6_xarm6_traj_controller` and
  `xarm6_xarm_gripper_traj_controller`.
- Dual mode: all four controllers.

Check the corresponding planning groups in the RViz MotionPlanning panel.

## 8. Rebuild After Source Changes

```bash
cd ~/projects/cais-spade-llm
make bootstrap-gazebo

source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
```

For hardware commands and daily operation, continue with
`ros2/docs/ros2_three_mode_control_guide.md`. For IFRA details, see
`ros2/docs/ifra_linkattacher_setup_from_scratch.md`.

## Common Issues

| Problem | Resolution |
|---|---|
| `Package 'cais_lab_robotics' not found` | Run bootstrap and source `~/ros2_ws/install/setup.bash`. |
| A launch/config/RViz change is ignored | Rebuild with bootstrap and use a newly sourced terminal. |
| `No module named 'lxml'` or `rclpy` import failure | Deactivate the Poetry virtual environment for raw ROS2 commands. |
| Gazebo spawn timeout | Stop stale Gazebo processes and relaunch. |
| `/ATTACHLINK` or `/DETACHLINK` is missing | Verify the IFRA packages and rerun bootstrap. |
