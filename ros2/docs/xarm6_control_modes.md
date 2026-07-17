# xArm6 Control Modes

Three ways to control the xArm6, from development through production.

---

## 0. Clean Every Time (Before Switching Modes)

```bash
killall -9 gzserver gzclient robot_state_publisher spawner spawn_entity.py \
  rviz2 move_group static_transform_publisher 2>/dev/null
pkill -9 -f gazebo 2>/dev/null
pkill -9 -f rviz 2>/dev/null
pkill -9 -f ros2 2>/dev/null
```

Verify nothing is left running:

```bash
ros2 node list   # should return empty
```

---

## Mode 1: MoveIt + Gazebo (Simulation)

**When to use:** Development, testing trajectories, validating pick-and-place logic.

### Launch (single command)

```bash
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
ros2 launch cais_lab_robotics xarm6_moveit_single_gazebo.launch.py
```

This starts:
- Gazebo with full environment (tables, printers, parts, assembly board, camera)
- Both robots (xArm6 active, UR5e visual-only)
- MoveIt + RViz connected to xArm6 (30s delay for Gazebo to load)

### Control the arm

**In MoveIt RViz:**
1. Uncheck `RobotModel` in the left panel (removes duplicate ghost)
2. Planning Group → `xarm6_xarm6` for arm, `xarm6_xarm_gripper` for gripper
3. Drag the interactive marker → Plan → Execute

**From command line (separate terminal):**

```bash
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash

# Move arm (6 joint values in radians, time in seconds)
ros2 topic pub --once /xarm6_xarm6_traj_controller/joint_trajectory \
  trajectory_msgs/msg/JointTrajectory \
  "{joint_names: ['xarm6_joint1','xarm6_joint2','xarm6_joint3','xarm6_joint4','xarm6_joint5','xarm6_joint6'], \
    points: [{positions: [0.5, -0.3, 0.2, 0.0, 0.3, 0.0], time_from_start: {sec: 3}}]}"

# Home position (all zeros)
ros2 topic pub --once /xarm6_xarm6_traj_controller/joint_trajectory \
  trajectory_msgs/msg/JointTrajectory \
  "{joint_names: ['xarm6_joint1','xarm6_joint2','xarm6_joint3','xarm6_joint4','xarm6_joint5','xarm6_joint6'], \
    points: [{positions: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], time_from_start: {sec: 3}}]}"

# Close gripper (0.85 = closed, 0.0 = open)
ros2 topic pub --once /xarm6_xarm_gripper_traj_controller/joint_trajectory \
  trajectory_msgs/msg/JointTrajectory \
  "{joint_names: ['xarm6_drive_joint'], \
    points: [{positions: [0.85], time_from_start: {sec: 1}}]}"

# Open gripper
ros2 topic pub --once /xarm6_xarm_gripper_traj_controller/joint_trajectory \
  trajectory_msgs/msg/JointTrajectory \
  "{joint_names: ['xarm6_drive_joint'], \
    points: [{positions: [0.0], time_from_start: {sec: 1}}]}"
```

---

## Mode 2: MoveIt + Real Hardware

**When to use:** Running on the physical xArm6 after validating in simulation.

> ⚠️ **SAFETY:**
> - Clear the workspace around the robot
> - Stay within reach of the **E-STOP** button
> - Set **Velocity Scaling to 0.05** (5%) for first moves
> - Ensure robot is in **Remote Mode** on the teach pendant

### Launch (two terminals)

**Terminal 1 — Hardware driver:**

```bash
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
ros2 launch xarm_api xarm6_driver.launch.py robot_ip:=192.168.1.240
```

**Terminal 2 — MoveIt:**

```bash
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
ros2 launch xarm_moveit_config xarm6_moveit_realmove.launch.py \
  robot_ip:=192.168.1.240 add_gripper:=true
```

### Control the arm

Same MoveIt RViz interface as simulation. Drag → Plan → Execute.

> **Important:** The real-move launch uses **unprefixed** joint names (`joint1` instead of `xarm6_joint1`). If using command-line topic commands, check controller names first:
>
> ```bash
> ros2 action list   # see available controllers
> ```

---

## Mode 3: Digital Twin (TODO)

**Goal:** Run Gazebo and real hardware simultaneously so the simulation mirrors the physical robot in real-time.

**Status:** Not yet implemented. Requires a bridge node that forwards real joint states into a passive Gazebo model.

**Architecture when implemented:**

```
MoveIt ──commands──► Real xArm6 ──joint_states──► Gazebo (passive mirror)
```

**Required work:**
- [ ] Create a ROS 2 bridge node that subscribes to real `/joint_states` and calls Gazebo's `set_model_configuration` service
- [ ] Create a launch file that starts Gazebo in passive mode alongside the real hardware driver
- [ ] Handle namespace separation to avoid controller conflicts
- [ ] Test synchronization latency

---

## Quick Reference

| | Mode 1: Simulation | Mode 2: Real Hardware |
|---|---|---|
| **Command** | `ros2 launch cais_lab_robotics xarm6_moveit_single_gazebo.launch.py` | T1: `ros2 launch xarm_api xarm6_driver.launch.py robot_ip:=192.168.1.240` <br> T2: `ros2 launch xarm_moveit_config xarm6_moveit_realmove.launch.py robot_ip:=192.168.1.240 add_gripper:=true` |
| **Terminals** | 1 | 2 |
| **MoveIt GUI** | ✅ Same | ✅ Same |
| **Python code** | ✅ Same | ✅ Same |
| **Joint prefix** | `xarm6_` | None (unprefixed) |
| **Arm controller** | `xarm6_xarm6_traj_controller` | Check with `ros2 action list` |
| **Gripper controller** | `xarm6_xarm_gripper_traj_controller` | Check with `ros2 action list` |
| **Safety** | None needed | E-STOP ready, 5% velocity |

---

## Workflow: Sim → Real

```
1. Develop in Mode 1 (Gazebo)
   ├── Test trajectories
   ├── Verify no collisions
   └── Iterate until perfect

2. Clean up (Section 0)

3. Switch to Mode 2 (Real Hardware)
   ├── Same MoveIt, same Python code
   ├── Set velocity to 5%
   └── Execute validated trajectories
```

**Zero code changes between modes.** Only the launch file changes.
