# ROS2 Three-Mode Control Guide

This guide defines the three launch modes:

1. UR5e control mode
2. xArm6 control mode
3. Dual robot control mode

All commands assume:

```bash
deactivate 2>/dev/null || true
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
```

## 0. Clean Before Switching Modes

```bash
killall -9 gzserver gzclient robot_state_publisher spawner spawn_entity.py rviz2 move_group ros2 2>/dev/null
pkill -9 -f gazebo 2>/dev/null
pkill -9 -f rviz 2>/dev/null
```

## Quickstart (Copy-Paste)

### UR5e Mode

```bash
# MoveIt only (no Gazebo)
ros2 launch ur_robot_driver ur5e.launch.py robot_ip:=127.0.0.1 use_fake_hardware:=true launch_rviz:=false initial_joint_controller:=joint_trajectory_controller activate_joint_controller:=true
# In another terminal:
ros2 launch ur_moveit_config ur_moveit.launch.py ur_type:=ur5e launch_rviz:=true use_sim_time:=false

# Gazebo + MoveIt + RViz (one command opens both)
ros2 launch xarm_gazebo ur5e_rg2_moveit_gazebo.launch.py

# Hardware
ros2 launch ur_robot_driver ur5e.launch.py robot_ip:=192.168.1.172 launch_rviz:=false
# In another terminal:
ros2 launch ur_moveit_config ur_moveit.launch.py ur_type:=ur5e launch_rviz:=true use_sim_time:=false
```

### xArm6 Mode

```bash
# MoveIt only (no Gazebo)
ros2 launch xarm_moveit_config xarm6_moveit_fake.launch.py add_gripper:=true

# Gazebo + MoveIt + RViz (one command opens both)
ros2 launch xarm_gazebo xarm6_moveit_single_gazebo.launch.py

# Hardware
ros2 launch xarm_api xarm6_driver.launch.py robot_ip:=192.168.1.240
# In another terminal:
ros2 launch xarm_moveit_config xarm6_moveit_realmove.launch.py robot_ip:=192.168.1.240 add_gripper:=true
```

### Dual Mode

```bash
# MoveIt only (no Gazebo): run two separate MoveIt stacks
ros2 launch ur_robot_driver ur5e.launch.py robot_ip:=127.0.0.1 use_fake_hardware:=true launch_rviz:=false initial_joint_controller:=joint_trajectory_controller activate_joint_controller:=true
ros2 launch ur_moveit_config ur_moveit.launch.py ur_type:=ur5e launch_rviz:=true use_sim_time:=false
ros2 launch xarm_moveit_config xarm6_moveit_fake.launch.py add_gripper:=true

# Gazebo + MoveIt + RViz (one command opens both)
ros2 launch xarm_gazebo dual_moveit_gazebo.launch.py

# Hardware (current approach: two stacks)
ros2 launch ur_robot_driver ur5e.launch.py robot_ip:=192.168.1.172 launch_rviz:=false
ros2 launch ur_moveit_config ur_moveit.launch.py ur_type:=ur5e launch_rviz:=true use_sim_time:=false
ros2 launch xarm_api xarm6_driver.launch.py robot_ip:=192.168.1.240
ros2 launch xarm_moveit_config xarm6_moveit_realmove.launch.py robot_ip:=192.168.1.240 add_gripper:=true
```



## 1. UR5e Control Mode (UR5e + RG2 + one table)

### Step A: MoveIt Only (No Gazebo)

This starts MoveIt planning without Gazebo physics.

Terminal 1 (fake hardware):

```bash
source /opt/ros/humble/setup.bash
ros2 launch ur_robot_driver ur5e.launch.py \
  robot_ip:=127.0.0.1 \
  use_fake_hardware:=true \
  launch_rviz:=false \
  initial_joint_controller:=joint_trajectory_controller \
  activate_joint_controller:=true
```

Terminal 2 (MoveIt):

```bash
source /opt/ros/humble/setup.bash
ros2 launch ur_moveit_config ur_moveit.launch.py \
  ur_type:=ur5e \
  launch_rviz:=true \
  use_sim_time:=false
```

### Step B: Gazebo (Simulation Validation)

Gazebo only:

```bash
ros2 launch xarm_gazebo ur5e_rg2_gazebo.launch.py
```

Gazebo + MoveIt + RViz:

```bash
ros2 launch xarm_gazebo ur5e_rg2_moveit_gazebo.launch.py
```
This single command launches both Gazebo and MoveIt (RViz).

MoveIt groups:
- `ur5e_ur_manipulator`
- `ur5e_rg2_gripper`

Controllers:
- `ur5e_joint_trajectory_controller`
- `ur5e_rg2_gripper_traj_controller`

### Step C: Hardware (Real UR5e)

Terminal 1 (UR driver):

```bash
source /opt/ros/humble/setup.bash
ros2 launch ur_robot_driver ur5e.launch.py \
  robot_ip:=192.168.1.172 \
  launch_rviz:=false
```

Terminal 2 (MoveIt):

```bash
source /opt/ros/humble/setup.bash
ros2 launch ur_moveit_config ur_moveit.launch.py \
  ur_type:=ur5e \
  launch_rviz:=true \
  use_sim_time:=false
```

Notes:
- UR5e base in custom Gazebo mode starts at yaw `3.142` (180 degrees).
- RG2 simulation uses command joint `ur5e_rg2_finger_width`.
- Real RG2 hardware control is not integrated in this repo yet; UR5e arm hardware control is supported.

## 2. xArm6 Control Mode (xArm6 + xArm gripper + one table)

### Step A: MoveIt Only (No Gazebo)

This starts MoveIt planning without Gazebo physics.

```bash
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
ros2 launch xarm_moveit_config xarm6_moveit_fake.launch.py add_gripper:=true
```

### Step B: Gazebo (Simulation Validation)

Gazebo only:

```bash
ros2 launch xarm_gazebo xarm6_single_gazebo.launch.py
```

Gazebo + MoveIt + RViz:

```bash
ros2 launch xarm_gazebo xarm6_moveit_single_gazebo.launch.py
```
This single command launches both Gazebo and MoveIt (RViz).

MoveIt groups:
- `xarm6_xarm6`
- `xarm6_xarm_gripper`

Controllers:
- `xarm6_xarm6_traj_controller`
- `xarm6_xarm_gripper_traj_controller`

### Step C: Hardware (Real xArm6)

Terminal 1 (xArm driver):

```bash
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
ros2 launch xarm_api xarm6_driver.launch.py robot_ip:=192.168.1.240
```

Terminal 2 (MoveIt real move):

```bash
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
ros2 launch xarm_moveit_config xarm6_moveit_realmove.launch.py \
  robot_ip:=192.168.1.240 add_gripper:=true
```

Note:
- xArm6 base in custom Gazebo mode starts at yaw `3.142` (180 degrees).

## 3. Dual Robot Control Mode (UR5e + xArm6 + both grippers)

### Step A: MoveIt Only (No Gazebo)

There is no single merged UR5e+xArm6 MoveIt-only launch in this repo.
For MoveIt-only planning, run separately:
- UR5e: Section 1 Step A
- xArm6: Section 2 Step A

### Step B: Gazebo (Simulation Validation)

Gazebo only:

```bash
ros2 launch xarm_gazebo xarm6_ur5e_gazebo.launch.py
```

Gazebo + MoveIt + RViz:

```bash
ros2 launch xarm_gazebo dual_moveit_gazebo.launch.py
```
This single command launches both Gazebo and MoveIt (RViz).

MoveIt groups:
- `xarm6_xarm6`
- `xarm6_xarm_gripper`
- `ur5e_ur_manipulator`
- `ur5e_rg2_gripper`

Controllers:
- `xarm6_xarm6_traj_controller`
- `xarm6_xarm_gripper_traj_controller`
- `ur5e_joint_trajectory_controller`
- `ur5e_rg2_gripper_traj_controller`

### Step C: Hardware (Real Dual Setup)

Current hardware-supported approach is two driver stacks and two MoveIt stacks:

Terminal 1 (UR5e driver):

```bash
source /opt/ros/humble/setup.bash
ros2 launch ur_robot_driver ur5e.launch.py robot_ip:=192.168.1.172 launch_rviz:=false
```

Terminal 2 (UR5e MoveIt):

```bash
source /opt/ros/humble/setup.bash
ros2 launch ur_moveit_config ur_moveit.launch.py ur_type:=ur5e launch_rviz:=true use_sim_time:=false
```

Terminal 3 (xArm6 driver):

```bash
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
ros2 launch xarm_api xarm6_driver.launch.py robot_ip:=192.168.1.240
```

Terminal 4 (xArm6 MoveIt):

```bash
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
ros2 launch xarm_moveit_config xarm6_moveit_realmove.launch.py robot_ip:=192.168.1.240 add_gripper:=true
```

Note:
- Dual hardware with one combined `move_group` (both robots in one MoveIt instance) is not yet implemented in this repo.

## 4. Hardware Support Matrix

| Mode | MoveIt | Gazebo | Hardware |
|---|---|---|---|
| UR5e mode | Yes | Yes | UR5e arm yes, RG2 real-hardware integration pending |
| xArm6 mode | Yes | Yes | Yes (arm + xArm gripper) |
| Dual mode | Yes (single MoveIt in Gazebo) | Yes | Yes using two separate hardware stacks; single merged dual-hardware MoveIt pending |

## 5. Quick Verification

```bash
ros2 control list_controllers
```

If MoveIt is running, confirm the expected planning groups are present in RViz.

## 6. Typical Gripper Tests

UR5e RG2:

```bash
ros2 topic pub --once /ur5e_rg2_gripper_traj_controller/joint_trajectory \
  trajectory_msgs/msg/JointTrajectory \
  "{joint_names: ['ur5e_rg2_finger_width'], points: [{positions: [0.08], time_from_start: {sec: 2}}]}"
```

xArm gripper:

```bash
ros2 topic pub --once /xarm6_xarm_gripper_traj_controller/joint_trajectory \
  trajectory_msgs/msg/JointTrajectory \
  "{joint_names: ['xarm6_drive_joint'], points: [{positions: [0.85], time_from_start: {sec: 2}}]}"
```
