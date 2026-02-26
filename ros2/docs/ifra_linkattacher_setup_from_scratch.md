# IFRA LinkAttacher Setup From Scratch (Dual Robot)

This guide reproduces the dual-robot grasp setup on a fresh desktop and includes the
patched IFRA LinkAttacher plugin required for simultaneous attachments.

## 1. Prerequisites

- Ubuntu 22.04
- ROS2 Humble
- Gazebo Classic (`gazebo_ros`)
- Workspace path: `~/ros2_ws`

Install required system packages:

```bash
sudo apt update && sudo apt install -y \
  ros-humble-gazebo-ros-pkgs \
  ros-humble-gazebo-ros2-control \
  ros-humble-controller-manager \
  ros-humble-joint-state-broadcaster \
  ros-humble-joint-trajectory-controller \
  ros-humble-ur-description \
  ros-humble-ur-moveit-config \
  ros-humble-xacro \
  ros-humble-robot-state-publisher
```

## 2. Create Workspace And Clone Upstream Repos

```bash
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src

git clone -b humble https://github.com/xArm-Developer/xarm_ros2.git --recursive
git clone https://github.com/tonydle/OnRobot_ROS2_Description.git
git clone https://github.com/IFRA-Cranfield/IFRA_LinkAttacher.git
```

## 3. Copy Versioned Custom Files From This Repo

From your project root (`~/projects/cais-spade-llm`):

```bash
# xarm_gazebo custom launch/world/rviz
cp ros2/xarm_gazebo/worlds/table.world \
  ~/ros2_ws/src/xarm_ros2/xarm_gazebo/worlds/table.world
cp ros2/xarm_gazebo/launch/xarm6_ur5e_gazebo.launch.py \
  ~/ros2_ws/src/xarm_ros2/xarm_gazebo/launch/xarm6_ur5e_gazebo.launch.py
cp ros2/xarm_gazebo/launch/dual_moveit_gazebo.launch.py \
  ~/ros2_ws/src/xarm_ros2/xarm_gazebo/launch/dual_moveit_gazebo.launch.py
cp ros2/xarm_gazebo/launch/auto_link_attacher_node.py \
  ~/ros2_ws/src/xarm_ros2/xarm_gazebo/launch/auto_link_attacher_node.py
cp ros2/xarm_gazebo/rviz/dual_moveit.rviz \
  ~/ros2_ws/src/xarm_ros2/xarm_gazebo/rviz/dual_moveit.rviz

# Patched IFRA plugin source (multi-attach + preserve grasp pose)
cp ros2/third_party/IFRA_LinkAttacher/ros2_LinkAttacher/src/gazebo_link_attacher.cpp \
  ~/ros2_ws/src/IFRA_LinkAttacher/ros2_LinkAttacher/src/gazebo_link_attacher.cpp
```

## 4. Build (Plugin First, Then Gazebo Package)

```bash
cd ~/ros2_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select linkattacher_msgs ros2_linkattacher xarm_gazebo
source ~/ros2_ws/install/setup.bash
```

## 5. Launch

```bash
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
ros2 launch xarm_gazebo dual_moveit_gazebo.launch.py
```

## 6. Confirm World Plugin Line Exists

`table.world` must include the IFRA world plugin entry:

```xml
<plugin name="gazebo_ros_link_attacher" filename="libgazebo_link_attacher.so"/>
```

In this repository it is already present in:
`ros2/xarm_gazebo/worlds/table.world`

## 7. Verify LinkAttacher Is Active

In another terminal:

```bash
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
ros2 service list | rg 'ATTACHLINK|DETACHLINK'
```

Expected:

- `/ATTACHLINK`
- `/DETACHLINK`

## 8. Common Failure And Fix

If you see:

- `Both links have already been attached, aborting new attachment.`

then the patched IFRA source was not copied/built. Repeat steps 3 and 4.

If you see:

- `package 'ros2_linkattacher' not found`

then `IFRA_LinkAttacher` repo is missing from `~/ros2_ws/src` or was not built.

## 9. Files You Should Keep Under Git (`ros2/`)

These are the editable, versioned files in this repo for your setup:

- `ros2/xarm_gazebo/launch/auto_link_attacher_node.py`
- `ros2/xarm_gazebo/launch/dual_moveit_gazebo.launch.py`
- `ros2/xarm_gazebo/launch/xarm6_ur5e_gazebo.launch.py`
- `ros2/xarm_gazebo/rviz/dual_moveit.rviz`
- `ros2/xarm_gazebo/worlds/table.world`
- `ros2/third_party/IFRA_LinkAttacher/ros2_LinkAttacher/src/gazebo_link_attacher.cpp`
