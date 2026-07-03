# UR5e Control Modes

Four ways to control the UR5e, from lightweight development through production.

> **Key difference from xArm6:** The UR5e uses the `ur_robot_driver` package (not xArm SDK).
> Simulation uses **Gazebo Ignition** (`gz`), not Gazebo Classic.

---

## 0. Clean Every Time (Before Switching Modes)

```bash
killall -9 gzserver gzclient robot_state_publisher spawner spawn_entity.py \
  rviz2 move_group static_transform_publisher ros2 2>/dev/null
pkill -9 -f gazebo 2>/dev/null
pkill -9 -f rviz 2>/dev/null
pkill -9 -f ros2 2>/dev/null
pkill -9 -f "gz sim" 2>/dev/null
```

Verify nothing is left running:

```bash
ros2 node list   # should return empty
```

> ⚠️ **Two-Terminal Rule:** Never run ROS2/Gazebo commands with the Poetry venv activated.
> If your venv is active (`which python3` points to `.venv/bin/python3`), run `deactivate` first.

---

## Mode 1: Fake Hardware (No Physics, Fastest Startup)

**When to use:** Quick development, testing Python control scripts, validating joint trajectories without waiting for Gazebo to load. The robot model is loaded by `ros2_control` with a mock hardware interface — commands are instantly mirrored to state.

### Launch (two terminals)

**Terminal 1 — Fake hardware driver + controllers:**

```bash
source /opt/ros/humble/setup.bash
ros2 launch ur_robot_driver ur5e.launch.py \
  robot_ip:=yyy.yyy.yyy.yyy \
  use_fake_hardware:=true \
  launch_rviz:=false \
  initial_joint_controller:=joint_trajectory_controller \
  activate_joint_controller:=true
```

> `robot_ip` is required by the launch file but **ignored** in fake-hardware mode — any value works.
> We use `joint_trajectory_controller` instead of `scaled_joint_trajectory_controller` because the scaled variant depends on real-hardware speed scaling that doesn't exist in fake mode.

**Terminal 2 — MoveIt (optional, for drag-to-plan):**

```bash
source /opt/ros/humble/setup.bash
ros2 launch ur_moveit_config ur_moveit.launch.py \
  ur_type:=ur5e \
  launch_rviz:=true \
  use_sim_time:=false
```

> `use_sim_time:=false` because fake hardware runs on wall-clock time, not simulation time.

### Control the arm

**In MoveIt RViz:**
1. In the MoveIt panel, Planning Group → `ur_manipulator`
2. Drag the interactive marker on the end-effector → **Plan** → **Execute**

**From command line (separate terminal):**

```bash
source /opt/ros/humble/setup.bash

# Move arm (6 joint values in radians, time in seconds)
ros2 topic pub --once /joint_trajectory_controller/joint_trajectory \
  trajectory_msgs/msg/JointTrajectory \
  "{joint_names: ['shoulder_pan_joint','shoulder_lift_joint','elbow_joint','wrist_1_joint','wrist_2_joint','wrist_3_joint'], \
    points: [{positions: [0.0, -1.57, 1.57, -1.57, -1.57, 0.0], time_from_start: {sec: 3}}]}"

# All-zeros pose (arm straight up)
ros2 topic pub --once /joint_trajectory_controller/joint_trajectory \
  trajectory_msgs/msg/JointTrajectory \
  "{joint_names: ['shoulder_pan_joint','shoulder_lift_joint','elbow_joint','wrist_1_joint','wrist_2_joint','wrist_3_joint'], \
    points: [{positions: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], time_from_start: {sec: 3}}]}"
```

**From Python:**

```python
from cais_spade_llm.resources.robot.gazebo_pick_place_controller import UR5eGazeboController
import time

ctrl = UR5eGazeboController()
ctrl.init()
time.sleep(1)

ctrl.move_joints([0.0, -1.57, 1.57, -1.57, -1.57, 0.0], duration_sec=3)
time.sleep(4)
ctrl.shutdown()
```

> ⚠️ The Python controller publishes to `/scaled_joint_trajectory_controller/joint_trajectory` by default.
> In fake-hardware mode with `joint_trajectory_controller`, you need to either:
> - Modify `TRAJECTORY_TOPIC` in `ur5e_controller.py` to `/joint_trajectory_controller/joint_trajectory`, **or**
> - Launch with `initial_joint_controller:=scaled_joint_trajectory_controller` (works but may log warnings).

### Verify it's working

```bash
# Check controllers are active
ros2 control list_controllers

# Should show:
#   joint_state_broadcaster    [active]
#   joint_trajectory_controller [active]   ← this is the one we command

# Read current joint positions
ros2 topic echo /joint_states --once
```

---

## Mode 2: MoveIt + Gazebo Ignition (Full Simulation)

**When to use:** Testing with physics, gravity, collisions. Visual validation of trajectories before going to real hardware.

### Launch (two terminals)

**Terminal 1 — Gazebo Ignition + controllers:**

```bash
source /opt/ros/humble/setup.bash
ros2 launch ur_simulation_gz ur_sim_control.launch.py \
  ur_type:=ur5e \
  launch_rviz:=false
```

- Opens **Gazebo Ignition** and loads the robot with `joint_trajectory_controller`
- Wait until the Gazebo window appears and the robot is visible

**Terminal 2 — MoveIt (optional, for drag-to-plan):**

```bash
source /opt/ros/humble/setup.bash
ros2 launch ur_moveit_config ur_moveit.launch.py \
  ur_type:=ur5e \
  launch_rviz:=true \
  use_sim_time:=true
```

> `use_sim_time:=true` is required — Gazebo publishes its own `/clock`.

### Control the arm

**In MoveIt RViz:** Same as Mode 1 — drag → Plan → Execute.

**From command line:**

```bash
source /opt/ros/humble/setup.bash

# Move arm
ros2 topic pub --once /joint_trajectory_controller/joint_trajectory \
  trajectory_msgs/msg/JointTrajectory \
  "{joint_names: ['shoulder_pan_joint','shoulder_lift_joint','elbow_joint','wrist_1_joint','wrist_2_joint','wrist_3_joint'], \
    points: [{positions: [0.0, -1.57, 1.57, -1.57, -1.57, 0.0], time_from_start: {sec: 2}}]}"
```

**From Python:** Same `UR5eGazeboController` code, but note the controller topic is `/joint_trajectory_controller/joint_trajectory` (not `scaled_joint_trajectory_controller`).

**Joint sliders GUI (quick test):**

```bash
ros2 run rqt_joint_trajectory_controller rqt_joint_trajectory_controller
```

---

## Mode 3: MoveIt + Real Hardware

**When to use:** Running on the physical UR5e after validating in simulation.

> ⚠️ **SAFETY:**
> - Clear the workspace around the robot
> - Stay within reach of the **E-STOP** button
> - Set **Velocity Scaling to 5%** for first moves
> - The teach pendant must have the **External Control** URCap program loaded and running

### Launch (two terminals)

**Terminal 1 — Hardware driver:**

```bash
source /opt/ros/humble/setup.bash
ros2 launch ur_robot_driver ur5e.launch.py \
  robot_ip:=192.168.1.172 \
  launch_rviz:=false
```

> On the teach pendant: Press **Play** on the `External Control` program so the robot accepts commands.

**Terminal 2 — MoveIt:**

```bash
source /opt/ros/humble/setup.bash
ros2 launch ur_moveit_config ur_moveit.launch.py \
  ur_type:=ur5e \
  launch_rviz:=true \
  use_sim_time:=false
```

### Control the arm

Same MoveIt RViz interface as simulation. Drag → Plan → Execute.

**From command line:**

```bash
source /opt/ros/humble/setup.bash

# The real driver defaults to scaled_joint_trajectory_controller
ros2 topic pub --once /scaled_joint_trajectory_controller/joint_trajectory \
  trajectory_msgs/msg/JointTrajectory \
  "{joint_names: ['shoulder_pan_joint','shoulder_lift_joint','elbow_joint','wrist_1_joint','wrist_2_joint','wrist_3_joint'], \
    points: [{positions: [0.0, -1.57, 1.57, -1.57, -1.57, 0.0], time_from_start: {sec: 5}}]}"
```

> **Note:** Use `scaled_joint_trajectory_controller` for real hardware — it respects the speed slider on the teach pendant.

---

## Mode 4: Digital Twin (TODO)

**Goal:** Run Gazebo and real hardware simultaneously so the simulation mirrors the physical robot in real-time.

**Status:** Not yet implemented. Requires a bridge node that forwards real joint states into a passive Gazebo model.

**Architecture when implemented:**

```
MoveIt ──commands──► Real UR5e ──joint_states──► Gazebo Ignition (passive mirror)
```

**Required work:**
- [ ] Create a ROS 2 bridge node that subscribes to real `/joint_states` and publishes to Gazebo Ignition model
- [ ] Create a launch file that starts Gazebo in passive mode alongside the real hardware driver
- [ ] Handle controller conflicts (only one controller can be active at a time)
- [ ] Test synchronization latency

---

## Quick Reference

| | Mode 1: Fake Hardware | Mode 2: Gazebo Ignition | Mode 3: Real Hardware |
|---|---|---|---|
| **Physics** | None (instant mirror) | Full simulation | Real world |
| **Startup time** | ~5 seconds | ~30 seconds | ~10 seconds |
| **Terminal 1** | `ur_robot_driver ur5e.launch.py robot_ip:=yyy use_fake_hardware:=true initial_joint_controller:=joint_trajectory_controller` | `ur_simulation_gz ur_sim_control.launch.py ur_type:=ur5e` | `ur_robot_driver ur5e.launch.py robot_ip:=192.168.1.172` |
| **Terminal 2** | `ur_moveit_config ur_moveit.launch.py ur_type:=ur5e use_sim_time:=false` | `ur_moveit_config ur_moveit.launch.py ur_type:=ur5e use_sim_time:=true` | `ur_moveit_config ur_moveit.launch.py ur_type:=ur5e use_sim_time:=false` |
| **Controller** | `joint_trajectory_controller` | `joint_trajectory_controller` | `scaled_joint_trajectory_controller` |
| **Traj topic** | `/joint_trajectory_controller/joint_trajectory` | `/joint_trajectory_controller/joint_trajectory` | `/scaled_joint_trajectory_controller/joint_trajectory` |
| **use_sim_time** | `false` | `true` | `false` |
| **MoveIt GUI** | ✅ Same | ✅ Same | ✅ Same |
| **Python code** | ✅ Same (adjust topic) | ✅ Same (adjust topic) | ✅ Same |
| **Joint prefix** | None (unprefixed) | None (unprefixed) | None (unprefixed) |
| **Safety** | None needed | None needed | E-STOP ready, 5% velocity |

---

## UR5e Joint Reference

| Index | Joint Name | Range | Axis |
|---|---|---|---|
| 0 | `shoulder_pan_joint` | ±360° | Z (base rotation) |
| 1 | `shoulder_lift_joint` | ±360° | Y |
| 2 | `elbow_joint` | ±360° | Y |
| 3 | `wrist_1_joint` | ±360° | Y |
| 4 | `wrist_2_joint` | ±360° | Z |
| 5 | `wrist_3_joint` | ±360° | Y |

All values are in **radians**. π/2 ≈ 1.5708 = 90°.

---

## Common Poses

```python
POSES = {
    # All joints at zero — arm pointing straight up
    "zero":  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],

    # Standard "home" — arm folded, out of the way
    "home":  [0.0, -1.5708, 1.5708, -1.5708, -1.5708, 0.0],

    # Arm stretched forward horizontally
    "forward": [0.0, -1.5708, 0.0, -1.5708, 0.0, 0.0],
}
```

---

## Key ROS2 Topics

| Topic | Type | Purpose |
|---|---|---|
| `/joint_states` | `sensor_msgs/JointState` | Current joint positions (read) |
| `/joint_trajectory_controller/joint_trajectory` | `trajectory_msgs/JointTrajectory` | Send joint commands — fake hw & Gazebo |
| `/scaled_joint_trajectory_controller/joint_trajectory` | `trajectory_msgs/JointTrajectory` | Send joint commands — real hardware |
| `/tf` | `tf2_msgs/TFMessage` | All frame transforms including TCP |

---

## Workflow: Fake → Sim → Real

```
1. Develop in Mode 1 (Fake Hardware)
   ├── Fastest iteration
   ├── Test joint trajectories
   └── Validate Python control code

2. Clean up (Section 0)

3. Validate in Mode 2 (Gazebo Ignition)
   ├── Test with physics and gravity
   ├── Verify no collisions
   └── Iterate until perfect

4. Clean up (Section 0)

5. Switch to Mode 3 (Real Hardware)
   ├── Same MoveIt, same Python code
   ├── Set velocity to 5%
   ├── Controller changes to scaled_joint_trajectory_controller
   └── Execute validated trajectories
```

**Minimal code changes between modes.** Only the launch file and trajectory controller topic change.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `No module named 'lxml'` | Poetry venv is active. Run `deactivate` first. |
| No RViz window | Check `echo $DISPLAY` — must show `:0` for WSLg |
| Robot doesn't move (fake hw) | Verify `joint_trajectory_controller` is active: `ros2 control list_controllers` |
| Robot doesn't move (real hw) | On the teach pendant, press **Play** on the External Control program |
| `Could not find controller 'scaled_joint_trajectory_controller'` | In fake hw mode, use `joint_trajectory_controller` instead |
| Gazebo crashes with `Ogre::UnimplementedException` | WSL OpenGL issue — run `export LIBGL_ALWAYS_SOFTWARE=1` first |
| MoveIt says "no active controller" | MoveIt defaults to `scaled_joint_trajectory_controller`. In sim/fake mode, pass `use_sim_time:=true/false` so MoveIt switches to `joint_trajectory_controller` |
| `source` keeps being needed | Add `source /opt/ros/humble/setup.bash` to your `~/.bashrc` |
