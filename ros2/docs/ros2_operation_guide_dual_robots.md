# ROS2 Dual-Robot Operation Guide: xArm6 + UR5e

How to launch xArm6 and UR5e together in the same Gazebo Classic simulation.

For the full 3-mode workflow (UR5e-only, xArm6-only, dual), follow:
`ros2/docs/ros2_three_mode_control_guide.md`

For a clean environment bootstrap, follow:
`ros2/docs/ros2_setup_from_scratch.md`

---

## Overview

| | xArm6 | UR5e |
|---|---|---|
| TF prefix | `xarm6_` | `ur5e_` |
| Position | Right (0.0, -0.7, 1.021) | Left (0.0, 0.7, 1.021) |
| Yaw | 180° (π) | 180° (π) |
| Hardware mode | `gazebo_ros2_control/GazeboSystem` | `gazebo_ros2_control/GazeboSystem` |
| Arm controller | `xarm6_xarm6_traj_controller` | `ur5e_joint_trajectory_controller` |
| Gripper controller | `xarm6_xarm_gripper_traj_controller` | `ur5e_rg2_gripper_traj_controller` |

Both robots share a **single combined URDF** with one `gazebo_ros2_control` plugin and one
`controller_manager`. Both have full physics in Gazebo and are controllable via MoveIt, including their respective grippers.

---

## CRITICAL: Two-Terminal Rule

> **Never run ROS2/Gazebo commands with the Poetry venv activated.**

```bash
deactivate   # if venv is currently active
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
```

---

## 0. Clean Everything First (Crucial for Gazebo)

Gazebo Classic prevents multiple identical nodes from occupying the same namespace and ports. If you previously launched Gazebo and it crashed or you stopped it with `Ctrl+C`, ghost processes will often remain in the background blocking new attempts (resulting in an `exit code 255`). Always aggressively clean your environment:

```bash
killall -9 gzserver gzclient robot_state_publisher spawner spawn_entity.py ros2
pkill -9 -f gazebo
```

---

## 1. Build the Workspace (first time or after changes)

```bash
cd ~/ros2_ws
colcon build --packages-select xarm_gazebo
source install/setup.bash
```

---

## 2. Launch Both Robots

```bash
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
ros2 launch xarm_gazebo xarm6_ur5e_gazebo.launch.py
```

**What happens:**
1. Gazebo Classic starts with `table.world` (two wooden tables side-by-side, assembly board, printers, parts).
2. After **30 seconds**: Both robots spawn as a single combined model. Robot positions are encoded in the URDF fixed joints (not spawn arguments).
3. After spawn completes: All controllers load (`joint_state_broadcaster`, xArm6 arm/gripper, UR5e arm).

*(Note: The Top, Bottom-Left, and Bottom-Right areas are left explicitly empty to allow space for the `prusa-mk3`, `prusa-mk4-1`, and `prusa-mk4-2` printer models).*

> **WSL note:** Gazebo is slow to initialize in WSL (typically 30–60 seconds before
> the `/spawn_entity` service becomes available). The 30-second delay is intentional.
> Do not kill the launch if nothing appears immediately — wait at least 2 minutes.

> **GUI note:** Closing the Gazebo window does NOT kill the simulation.
> `gui_required:=false` means gzserver keeps running even without gzclient.
> Use `pkill -f gazebo` to fully stop it.

---

## 3. Verify Both Robots Are Running

### Check spawned entities
```bash
ros2 service call /gazebo/get_world_properties gazebo_msgs/srv/GetWorldProperties
```
`dual_robot` should appear in the model list (both robots are one combined model).

### Check active controllers
```bash
ros2 control list_controllers
```
Expected output:
```
joint_state_broadcaster[joint_state_broadcaster/JointStateBroadcaster] active
xarm6_xarm6_traj_controller[joint_trajectory_controller/JointTrajectoryController] active
xarm6_xarm_gripper_traj_controller[joint_trajectory_controller/JointTrajectoryController] active
ur5e_joint_trajectory_controller[joint_trajectory_controller/JointTrajectoryController] active
ur5e_rg2_gripper_traj_controller[joint_trajectory_controller/JointTrajectoryController] active
```

### Check TF frames
```bash
ros2 run tf2_ros tf2_monitor
```
Should show frames like `xarm6_link_base`, `xarm6_link_eef`, `ur5e_base_link`, `ur5e_tool0`.

---

## 4. Move the Robots

### Move xArm6

```bash
ros2 topic pub --once /xarm6_xarm6_traj_controller/joint_trajectory \
  trajectory_msgs/msg/JointTrajectory \
  "{joint_names: ['xarm6_joint1','xarm6_joint2','xarm6_joint3','xarm6_joint4','xarm6_joint5','xarm6_joint6'], \
    points: [{positions: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], time_from_start: {sec: 2}}]}"
```

### Move UR5e

```bash
ros2 topic pub --once /ur5e_joint_trajectory_controller/joint_trajectory \
  trajectory_msgs/msg/JointTrajectory \
  "{joint_names: ['ur5e_shoulder_pan_joint','ur5e_shoulder_lift_joint','ur5e_elbow_joint','ur5e_wrist_1_joint','ur5e_wrist_2_joint','ur5e_wrist_3_joint'], \
    points: [{positions: [0.0, -1.57, 1.57, -1.57, -1.57, 0.0], time_from_start: {sec: 3}}]}"
```

### Read joint positions
```bash
# All joints (both robots) from shared broadcaster
ros2 topic echo /joint_states --once
```

### TCP positions
```bash
ros2 run tf2_ros tf2_echo xarm6_link_base xarm6_link_eef
ros2 run tf2_ros tf2_echo ur5e_base_link ur5e_tool0
```

---

## 4b. Control BOTH Robots with MoveIt

```bash
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
ros2 launch xarm_gazebo dual_moveit_gazebo.launch.py
```

**What happens (timeline):**
1. **0s** — Gazebo Classic starts with combined URDF (both robots with physics)
2. **30s** — Both robots spawn + all controllers load
3. **40s** — Single MoveIt `move_group` + RViz launches (both robots in one instance)

**One RViz window** will open with both robots visible. Use the **Planning Group**
dropdown in the MotionPlanning panel to switch between robots:
- `xarm6_xarm6` — xArm6 arm (6-DOF)
- `xarm6_xarm_gripper` — xArm6 gripper
- `ur5e_ur_manipulator` — UR5e arm (6-DOF)
- `ur5e_rg2_gripper` — UR5e RG2 gripper

Both robots move in **both** Gazebo and RViz when plans are executed.

---

## 5. Key Topics in Dual Mode

| Topic | Notes |
|---|---|
| `/joint_states` | All joint positions (both robots, shared broadcaster) |
| `/xarm6_xarm6_traj_controller/joint_trajectory` | Command xArm6 arm joints |
| `/xarm6_xarm_gripper_traj_controller/joint_trajectory` | Command xArm6 gripper |
| `/ur5e_joint_trajectory_controller/joint_trajectory` | Command UR5e arm joints |
| `/ur5e_rg2_gripper_traj_controller/joint_trajectory` | Command UR5e RG2 gripper |
| `/controller_manager/*` | Shared controller lifecycle (root namespace) |
| `/tf` | All TF frames (both robots) |

> **Why `xarm6_xarm6_traj_controller`?** The controller name includes the robot prefix
> (`xarm6_`) prepended to the base name (`xarm6_traj_controller`). This ensures
> no name collision when multiple robots share the same `controller_manager`.

---

## 6. Architecture Notes

### Combined URDF approach
Both robots are merged into a single URDF with one `gazebo_ros2_control` plugin. This
avoids the Gazebo Classic limitation where multiple `gazebo_ros2_control` plugin instances
conflict (the installed `gazebo_ros2_control` v0.4.10 doesn't properly support separate
namespaces). Robot positions are encoded as fixed joints from a shared `world` link:
- `xarm6_world_joint`: world → xarm6_link_base at (0, -0.7, 1.021) with π yaw
- `ur5e_world_joint`: world → ur5e_base_link at (0, 0.7, 1.021) with π yaw

### Why Gazebo Classic (not Ignition)?
The xArm6 packages (`xarm_ros2`) use `gazebo_ros2_control` which targets Gazebo Classic.
The standalone UR5e simulation uses Gazebo Ignition via `ur_simulation_gz`. Since xArm6
requires Classic, the dual-robot setup uses Classic for both.

### Why strip `ground_plane` and `world` from each robot URDF?
The standard UR5e and xArm6 URDFs include `world` links. Since the combined URDF defines
its own `world` link, each robot's individual `world` link is removed during URDF merging.
Similarly, `ground_plane` from UR5e conflicts with Gazebo's built-in ground plane.

### Controller namespace
`gazebo_ros2_control` starts `controller_manager` at the root namespace `/`.
All controllers (xArm6 and UR5e) are spawned with `--controller-manager /controller_manager`.

### Combined MoveIt approach
Both robots are managed by a **single `move_group`** node at the root namespace.
This avoids ROS2 action namespace prefixing issues that occur when running separate
`move_group` nodes in different namespaces. The combined SRDF merges both robots'
planning groups, and the MoveIt Simple Controller Manager maps each group to its
corresponding ROS2 controller action server at the root namespace.

---

## 7. Launch File Locations

```
# Gazebo base launch (combined URDF + controllers)
~/projects/cais-spade-llm/ros2/cais_lab_gazebo/launch/xarm6_ur5e_gazebo.launch.py

# Combined controller config
~/projects/cais-spade-llm/ros2/cais_lab_gazebo/config/xarm6_ur5e_controllers.yaml

# MoveIt launch (Gazebo + MoveIt for both robots)
~/projects/cais-spade-llm/ros2/cais_lab_gazebo/launch/dual_moveit_gazebo.launch.py

# Runtime copy used by ros2 launch after build
~/ros2_ws/src/xarm_ros2/xarm_gazebo/launch/dual_moveit_gazebo.launch.py
```

After any changes, rebuild:
```bash
cd ~/ros2_ws
colcon build --packages-select xarm_gazebo
source install/setup.bash
```

---

## 8. Troubleshooting

| Problem | Fix |
|---|---|
| `No module named 'lxml'` | Poetry venv active — run `deactivate` first |
| Gazebo doesn't open | Wait 60+ seconds; normal for WSL |
| Robots not visible | Wait 35+ seconds after launch; spawn is delayed 30s |
| `spawn_entity: service not available` | Gazebo not ready yet — wait longer, or increase `TimerAction(period=...)` |
| Controllers not loading | Check `ros2 control list_controllers`; spawn must complete first |
| `xarm6_xarm6_traj_controller` not found | Rebuild workspace: `colcon build --packages-select xarm_gazebo` |
| `ur5e_joint_trajectory_controller` not found | Rebuild workspace; verify `xarm6_ur5e_controllers.yaml` is installed |
| Closing Gazebo window kills everything | Rebuild with the current launch file — it uses `gui_required:=false` |
| UR5e doesn't move in Gazebo | Verify `ur5e_joint_trajectory_controller` is `active` in controller list |
