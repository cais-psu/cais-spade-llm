# ROS2 Setup From Scratch (Three Control Modes)

This guide reproduces all three simulation/control modes on ROS2 Humble:

1. `ur5e_rg2_moveit_gazebo.launch.py` (UR5e + RG2 + one table)
2. `xarm6_moveit_single_gazebo.launch.py` (xArm6 + xArm gripper + one table)
3. `dual_moveit_gazebo.launch.py` (UR5e + RG2 + xArm6 + xArm gripper)

For operation details after setup, see:
`ros2/docs/ros2_three_mode_control_guide.md`

For IFRA LinkAttacher world-plugin setup (including patched multi-attach source), see:
`ros2/docs/ifra_linkattacher_setup_from_scratch.md`

## 0. Base Environment

Use Ubuntu 22.04 + ROS2 Humble.

- If you still need OS/ROS installation on WSL, follow:
  `ros2/docs/wsl_ubuntu22_ros2_humble_setup.md`
- Continue here after ROS is installed.

Important:
- Do not run ROS2/Gazebo commands inside Poetry venv.
- In ROS terminals always source:
  `source /opt/ros/humble/setup.bash`

## 1. Install Required ROS Packages

```bash
sudo apt update && sudo apt install -y \
  ros-humble-gazebo-ros-pkgs \
  ros-humble-gazebo-ros2-control \
  ros-humble-ur-robot-driver \
  ros-humble-ur-description \
  ros-humble-ur-moveit-config \
  ros-humble-controller-manager \
  ros-humble-joint-state-broadcaster \
  ros-humble-joint-trajectory-controller \
  ros-humble-xacro \
  ros-humble-robot-state-publisher
```

## 2. Create Workspace And Clone Upstream Repos

```bash
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src

git clone -b humble https://github.com/xArm-Developer/xarm_ros2.git --recursive
git clone https://github.com/tonydle/OnRobot_ROS2_Description.git
```

## 3. Copy Custom Files Into `xarm_ros2/xarm_gazebo`

From `~/projects/cais-spade-llm`:

```bash
# Worlds
cp ros2/cais_lab_robotics/worlds/table.world \
  ~/ros2_ws/src/xarm_ros2/xarm_gazebo/worlds/table.world
cp ros2/cais_lab_robotics/worlds/single_table.world \
  ~/ros2_ws/src/xarm_ros2/xarm_gazebo/worlds/single_table.world

# Gazebo launches
cp ros2/cais_lab_robotics/launch/xarm6_ur5e_gazebo.launch.py \
  ~/ros2_ws/src/xarm_ros2/xarm_gazebo/launch/xarm6_ur5e_gazebo.launch.py
cp ros2/cais_lab_robotics/launch/ur5e_rg2_gazebo.launch.py \
  ~/ros2_ws/src/xarm_ros2/xarm_gazebo/launch/ur5e_rg2_gazebo.launch.py
cp ros2/cais_lab_robotics/launch/xarm6_single_gazebo.launch.py \
  ~/ros2_ws/src/xarm_ros2/xarm_gazebo/launch/xarm6_single_gazebo.launch.py

# MoveIt + Gazebo launches
cp ros2/cais_lab_robotics/launch/dual_moveit_gazebo.launch.py \
  ~/ros2_ws/src/xarm_ros2/xarm_gazebo/launch/dual_moveit_gazebo.launch.py
cp ros2/cais_lab_robotics/launch/ur5e_rg2_moveit_gazebo.launch.py \
  ~/ros2_ws/src/xarm_ros2/xarm_gazebo/launch/ur5e_rg2_moveit_gazebo.launch.py
cp ros2/cais_lab_robotics/launch/xarm6_moveit_single_gazebo.launch.py \
  ~/ros2_ws/src/xarm_ros2/xarm_gazebo/launch/xarm6_moveit_single_gazebo.launch.py

# ros2_control configs
mkdir -p ~/ros2_ws/src/xarm_ros2/xarm_gazebo/config/gazebo_ros2_control
cp ros2/cais_lab_robotics/config/gazebo_ros2_control/xarm6_ur5e_gazebo_ros2_control_controllers.yaml \
  ~/ros2_ws/src/xarm_ros2/xarm_gazebo/config/gazebo_ros2_control/xarm6_ur5e_gazebo_ros2_control_controllers.yaml
cp ros2/cais_lab_robotics/config/gazebo_ros2_control/ur5e_rg2_gazebo_ros2_control_controllers.yaml \
  ~/ros2_ws/src/xarm_ros2/xarm_gazebo/config/gazebo_ros2_control/ur5e_rg2_gazebo_ros2_control_controllers.yaml
mkdir -p ~/ros2_ws/src/xarm_ros2/xarm_gazebo/config/gazebo_initial_joint_positions
cp ros2/cais_lab_robotics/config/gazebo_initial_joint_positions/ur5e_gazebo_initial_joint_positions.yaml \
  ~/ros2_ws/src/xarm_ros2/xarm_gazebo/config/gazebo_initial_joint_positions/ur5e_gazebo_initial_joint_positions.yaml
mkdir -p ~/ros2_ws/src/xarm_ros2/xarm_gazebo/config/hardware_runtime
cp ros2/cais_lab_robotics/config/hardware_runtime/xarm6_ur5e_hardware_runtime.yaml \
  ~/ros2_ws/src/xarm_ros2/xarm_gazebo/config/hardware_runtime/xarm6_ur5e_hardware_runtime.yaml

# RViz profile for dual mode
mkdir -p ~/ros2_ws/src/xarm_ros2/xarm_gazebo/rviz
cp ros2/cais_lab_robotics/rviz/dual_moveit.rviz \
  ~/ros2_ws/src/xarm_ros2/xarm_gazebo/rviz/dual_moveit.rviz
```

## 4. Build

```bash
cd ~/ros2_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select xarm_gazebo
source install/setup.bash
```

## 5. Run One Of The Three Modes

```bash
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
```

Mode 1 (UR5e + RG2, one table):

```bash
ros2 launch xarm_gazebo ur5e_rg2_moveit_gazebo.launch.py
```
This single command opens both Gazebo and MoveIt (RViz).

Mode 2 (xArm6 + gripper, one table):

```bash
ros2 launch xarm_gazebo xarm6_moveit_single_gazebo.launch.py
```
This single command opens both Gazebo and MoveIt (RViz).

Mode 3 (dual robots + both grippers):

```bash
ros2 launch xarm_gazebo dual_moveit_gazebo.launch.py
```
This single command opens both Gazebo and MoveIt (RViz).

Hardware commands for each mode are documented in:
`ros2/docs/ros2_three_mode_control_guide.md`
(sections "Step C: Hardware" for all three modes).

## 6. Verify

```bash
ros2 control list_controllers
```

Expected per mode:
- Mode 1: `ur5e_joint_trajectory_controller`, `ur5e_rg2_gripper_traj_controller`
- Mode 2: `xarm6_xarm6_traj_controller`, `xarm6_xarm_gripper_traj_controller`
- Mode 3: all four controllers above

Check MoveIt planning groups in RViz:
- Mode 1: `ur5e_ur_manipulator`, `ur5e_rg2_gripper`
- Mode 2: `xarm6_xarm6`, `xarm6_xarm_gripper`
- Mode 3: all four groups above

## 7. Smoke Tests

UR5e RG2 close/open:

```bash
ros2 topic pub --once /ur5e_rg2_gripper_traj_controller/joint_trajectory \
  trajectory_msgs/msg/JointTrajectory \
  "{joint_names: ['ur5e_rg2_finger_width'], points: [{positions: [0.08], time_from_start: {sec: 2}}]}"
```

xArm gripper close/open:

```bash
ros2 topic pub --once /xarm6_xarm_gripper_traj_controller/joint_trajectory \
  trajectory_msgs/msg/JointTrajectory \
  "{joint_names: ['xarm6_drive_joint'], points: [{positions: [0.85], time_from_start: {sec: 2}}]}"
```

## 8. Robot Placement

- UR5e single mode: `(0.0, 0.0, 1.021)`, yaw `3.142` (180 degrees)
- xArm6 single mode: `(0.0, 0.0, 1.021)`, yaw `3.142` (180 degrees)
- Dual mode:
  xArm6 `(0.0, -0.7, 1.021)`, yaw `3.142`
  UR5e `(0.0, 0.7, 1.021)`, yaw `3.142`

## 9. Common Issues

- `No module named 'lxml'`: Poetry venv is active; run `deactivate`.
- Gazebo spawn timeout on WSL: wait longer or restart stale Gazebo processes.
- Missing RG2 controls in RViz: confirm `ur5e_rg2_moveit_gazebo.launch.py` and `ur5e_rg2_gazebo_ros2_control_controllers.yaml` were copied and rebuilt.
- RG2 instability: confirm updated `ur5e_rg2_gazebo.launch.py` is installed and rebuilt.
