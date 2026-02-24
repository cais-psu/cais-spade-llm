# ROS2 Dual-Robot Operation Guide: xArm6 + UR5e

How to launch xArm6 and UR5e together in the same Gazebo Classic simulation.

---

## Overview

| | xArm6 | UR5e |
|---|---|---|
| Namespace | `xarm6` | `ur5e` |
| TF prefix | `xarm6_` | `ur5e_` |
| Position | Right (0.0, -0.7, 1.021) | Left (0.0, 0.7, 1.021) |
| Yaw | 180° (π) | 0° (0.0) |
| Controllers | joint_state_broadcaster + trajectory controllers | Visual only (no active control) |
| Hardware mode | `gazebo_ros2_control/GazeboSystem` | `use_fake_hardware:=true` |

> **UR5e configuration:** In the dual-robot launch, the UR5e is loaded with
> `use_fake_hardware:=true` in Gazebo Classic. This allows it to physically 
> exist in the Classic simulator alongside the xArm6 without crashing the 
> `gazebo_ros2_control` driver.

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
1. Gazebo Classic starts with `table.world` (which loads two individual wooden tables touching side-by-side to create one large physical surface matching the real-world lab setup). 
2. A generic visual box model representing the **`assembly_board-v1`** spawns directly in the `[0, 0]` center bridging the two tables.
3. After **30 seconds**: xArm6 spawns on the **Right Outer Edge** side of the tables at `(0.0, -0.7, 1.021)` with 180° yaw.
4. After **32 seconds**: UR5e spawns on the **Left Outer Edge** side of the tables at `(0.0, 0.7, 1.021)` with 0° yaw.
5. After xArm6 spawn completes: joint controllers load for xArm6.

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
Both `xarm6` and `ur5e` should appear in the model list.

### Check active controllers (xArm6 only)
```bash
ros2 control list_controllers
```
Expected output:
```
joint_state_broadcaster[joint_state_broadcaster/JointStateBroadcaster] active
xarm6_xarm6_traj_controller[joint_trajectory_controller/JointTrajectoryController] active
xarm6_xarm_gripper_traj_controller[joint_trajectory_controller/JointTrajectoryController] active
```

### Check TF frames
```bash
ros2 run tf2_ros tf2_monitor
```
Should show frames like `xarm6_link_base`, `xarm6_link_eef`, `ur5e_base_link`, `ur5e_tool0`.

---

## 4. Move the xArm6

In dual mode, xArm6 joint names all have the `xarm6_` prefix.

### Command line
```bash
ros2 topic pub --once /xarm6_xarm6_traj_controller/joint_trajectory \
  trajectory_msgs/msg/JointTrajectory \
  "{joint_names: ['xarm6_joint1','xarm6_joint2','xarm6_joint3','xarm6_joint4','xarm6_joint5','xarm6_joint6'], \
    points: [{positions: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], time_from_start: {sec: 2}}]}"
```

### Read joint positions
```bash
# xArm6 joints (prefix: xarm6_)
ros2 topic echo /xarm6/joint_states --once

# UR5e joints (prefix: ur5e_, read-only — no active control)
ros2 topic echo /ur5e/joint_states --once
```

### TCP position (xArm6)
```bash
ros2 run tf2_ros tf2_echo xarm6_link_base xarm6_link_eef
```

### TCP position (UR5e)
```bash
ros2 run tf2_ros tf2_echo ur5e_base_link ur5e_tool0
```

---

## 4b. Control BOTH Robots with MoveIt (New!)

The new `dual_moveit_gazebo.launch.py` starts Gazebo + MoveIt for **both** robots:

```bash
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
ros2 launch xarm_gazebo dual_moveit_gazebo.launch.py
```

**What happens (timeline):**
1. **0s** — Gazebo Classic starts with `table.world` + both robots
2. **30s** — xArm6 MoveIt + RViz launches
3. **35s** — UR5e `ros2_control_node` starts (mock hardware, `/ur5e` namespace)
4. **40–45s** — UR5e controllers spawn (`joint_state_broadcaster`, `joint_trajectory_controller`)
5. **50s** — UR5e MoveIt `move_group` + RViz launches

**Two RViz windows** will open:
- **RViz 1** — xArm6 MoveIt (planning group: `xarm6_xarm6`)
- **RViz 2** — UR5e MoveIt (planning group: `ur_manipulator`)

### Control from command line

```bash
# xArm6: same as before
ros2 topic pub --once /xarm6_xarm6_traj_controller/joint_trajectory \
  trajectory_msgs/msg/JointTrajectory \
  "{joint_names: ['xarm6_joint1','xarm6_joint2','xarm6_joint3','xarm6_joint4','xarm6_joint5','xarm6_joint6'], \
    points: [{positions: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], time_from_start: {sec: 2}}]}"

# UR5e: note the /ur5e namespace
ros2 topic pub --once /ur5e/joint_trajectory_controller/joint_trajectory \
  trajectory_msgs/msg/JointTrajectory \
  "{joint_names: ['ur5e_shoulder_pan_joint','ur5e_shoulder_lift_joint','ur5e_elbow_joint','ur5e_wrist_1_joint','ur5e_wrist_2_joint','ur5e_wrist_3_joint'], \
    points: [{positions: [0.0, -1.57, 1.57, -1.57, -1.57, 0.0], time_from_start: {sec: 3}}]}"
```

> **Architecture note:** The UR5e runs a separate `ros2_control_node` with `mock_components/GenericSystem` in the `/ur5e` namespace. This is independent of the xArm6's `gazebo_ros2_control`. The UR5e moves in RViz but NOT in Gazebo (Gazebo shows it as a static visual model). The xArm6 moves in both Gazebo and RViz.

---

## 5. Key Topics in Dual Mode

| Topic | Notes |
|---|---|
| `/xarm6/joint_states` | xArm6 joint positions |
| `/ur5e/joint_states` | UR5e joint positions (mock hardware controller) |
| `/xarm6_xarm6_traj_controller/joint_trajectory` | Command xArm6 joints |
| `/xarm6_xarm_gripper_traj_controller/joint_trajectory` | Command xArm6 gripper |
| `/ur5e/joint_trajectory_controller/joint_trajectory` | Command UR5e joints |
| `/controller_manager/*` | xArm6 controller lifecycle (root namespace) |
| `/ur5e/controller_manager/*` | UR5e controller lifecycle (ur5e namespace) |
| `/tf` | All TF frames (both robots) |

> **Why `xarm6_xarm6_traj_controller`?** The controller name includes the robot prefix
> (`xarm6_`) prepended to the base name (`xarm6_traj_controller`). This ensures
> no name collision when multiple robots share the same `controller_manager`.

---

## 6. Architecture Notes

### Why Gazebo Classic (not Ignition)?
The xArm6 packages (`xarm_ros2`) use `gazebo_ros2_control` which targets Gazebo Classic.
The standalone UR5e simulation uses Gazebo Ignition via `ur_simulation_gz`. Since xArm6
requires Classic, the dual-robot setup uses Classic for both.

### Why `use_fake_hardware` for UR5e?
Gazebo Classic supports only **one** `gazebo_ros2_control` plugin per simulation.
xArm6 uses that slot. Loading UR5e with `sim_gazebo:=true` would attempt to load a
second `gazebo_ros2_control` instance, which causes the spawn service to hang indefinitely.
Using `use_fake_hardware:=true` loads the UR5e as a visual-only model with a mock hardware
interface that does not conflict.

### Why strip `ground_plane` and `world` from UR5e?
The standard UR5e URDF (from `ur_description`) includes `ground_plane` and `world` links. 
Gazebo already contains a `ground_plane` from the `table.world` environment, which causes duplicate collisions. Furthermore, the `world` link rigidly anchors the UR5e to coordinates `0, 0, 0` inside Gazebo, completely overriding our requested `-x 1.5 -y -0.5 -z 1.021` spawn arguments (which buries the robot invisibly inside the table). The launch file dynamically scrubs these links from the XML tree before spawning.

### Controller namespace
`gazebo_ros2_control` always starts `controller_manager` at the root namespace `/`,
regardless of the robot's ROS namespace. Controllers must be spawned with
`--controller-manager /controller_manager`.

---

## 7. Launch File Location

```
~/ros2_ws/src/xarm_ros2/xarm_gazebo/launch/xarm6_ur5e_gazebo.launch.py
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
| Only xArm6 visible, no UR5e | Wait 35+ seconds after launch; UR5e spawns 2s after xArm6 |
| UR5e spawn hangs indefinitely | `ground_plane` conflict — rebuild workspace to pick up the latest launch file |
| `spawn_entity: service not available` | Gazebo not ready yet — wait longer, or increase `TimerAction(period=...)` |
| Controllers not loading | Check `ros2 control list_controllers`; xArm6 spawn must complete first |
| `xarm6_xarm6_traj_controller` not found | Rebuild workspace: `colcon build --packages-select xarm_gazebo` |
| Closing Gazebo window kills everything | Rebuild with the current launch file — it uses `gui_required:=false` |
| Can't control UR5e joints | Expected — UR5e runs in fake hardware mode (visual only) in dual setup |
