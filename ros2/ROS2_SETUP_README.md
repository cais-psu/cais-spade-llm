# CAIS-SPADE ROS 2 Simulation Environment

This folder contains everything needed to reproduce the full dual-robot Gazebo Classic
simulation used by the CAIS-SPADE multi-agent system.

---

## Quick Links

| Document | What it covers |
|---|---|
| **This file** | Overview, folder structure, and full setup from scratch |
| [`docs/wsl_ubuntu22_ros2_humble_setup.md`](docs/wsl_ubuntu22_ros2_humble_setup.md) | Installing Ubuntu 22.04 on WSL, ROS 2 Humble, Python 3.12, and Poetry |
| [`docs/ros2_operation_guide_dual_robots.md`](docs/ros2_operation_guide_dual_robots.md) | Running the dual-robot simulation (xArm6 + UR5e + RG2), verifying it, and controlling robots |
| [`docs/ros2_operation_guide_xarm6.md`](docs/ros2_operation_guide_xarm6.md) | xArm6 standalone simulation, joint reference, MoveIt2, Python control |
| [`docs/ros2_operation_guide_ur5e.md`](docs/ros2_operation_guide_ur5e.md) | UR5e standalone simulation (Gazebo Ignition), joint reference |

---

## Folder Structure

```
ros2/
├── README.md                              ← You are here
├── docs/
│   ├── wsl_ubuntu22_ros2_humble_setup.md  ← Step 1: OS and ROS 2 install
│   ├── ros2_operation_guide_dual_robots.md ← Step 3: Daily operation guide
│   ├── ros2_operation_guide_xarm6.md      ← Reference: xArm6 standalone
│   └── ros2_operation_guide_ur5e.md       ← Reference: UR5e standalone
└── cais_lab_gazebo/
    ├── launch/
    │   └── xarm6_ur5e_gazebo.launch.py    ← Custom launch file (copy into xarm_ros2)
    └── worlds/
        └── table.world                    ← Custom Gazebo world (copy into xarm_ros2)
```

---

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Ubuntu | 22.04 LTS | Native or WSL2. See [`docs/wsl_ubuntu22_ros2_humble_setup.md`](docs/wsl_ubuntu22_ros2_humble_setup.md) |
| ROS 2 | Humble Hawksbill | `ros-humble-desktop` |
| Gazebo Classic | 11.x | Installed via `ros-humble-gazebo-ros-pkgs` |
| Python | 3.10 (system) | **Do NOT run ROS 2 commands from a virtualenv** |

---

## Full Setup from Scratch (5 Steps)

> **Already have ROS 2 Humble installed?** Skip to Step 2.
>
> **Brand new machine?** Start with [`docs/wsl_ubuntu22_ros2_humble_setup.md`](docs/wsl_ubuntu22_ros2_humble_setup.md) first, then come back here at Step 2.

### Step 1: Install ROS 2 Dependencies

```bash
sudo apt update && sudo apt install -y \
  ros-humble-gazebo-ros-pkgs \
  ros-humble-gazebo-ros2-control \
  ros-humble-ur-description \
  ros-humble-controller-manager \
  ros-humble-joint-state-broadcaster \
  ros-humble-joint-trajectory-controller \
  ros-humble-xacro \
  ros-humble-robot-state-publisher
```

### Step 2: Clone Repositories into ROS 2 Workspace

```bash
mkdir -p ~/ros2_ws/src && cd ~/ros2_ws/src

# xArm ROS2 SDK (includes xarm_gazebo, xarm_description, etc.)
git clone -b humble https://github.com/xArm-Developer/xarm_ros2.git --recursive

# OnRobot RG2 gripper description (STL meshes + URDF)
git clone https://github.com/tonydle/OnRobot_ROS2_Description.git
```

### Step 3: Copy Custom Simulation Files

From the root of this repository (`cais-spade-llm/`):

```bash
# Copy the custom Gazebo world
# (tables, printers, parts, assembly board markers, overhead camera, gravity)
cp ros2/cais_lab_gazebo/worlds/table.world \
   ~/ros2_ws/src/xarm_ros2/xarm_gazebo/worlds/table.world

# Copy the dual-robot launch file
# (xArm6 + UR5e with RG2 gripper, spawn timing, controllers)
cp ros2/cais_lab_gazebo/launch/xarm6_ur5e_gazebo.launch.py \
   ~/ros2_ws/src/xarm_ros2/xarm_gazebo/launch/xarm6_ur5e_gazebo.launch.py
```

### Step 4: Build Everything

```bash
cd ~/ros2_ws
source /opt/ros/humble/setup.bash
colcon build
```

> ⏱ **First build takes ~15–30 minutes.** Subsequent incremental builds are fast.

### Step 5: Source and Launch

```bash
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
ros2 launch xarm_gazebo xarm6_ur5e_gazebo.launch.py
```

> **Tip:** Add both `source` lines to `~/.bashrc` so they auto-load:
> ```bash
> echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
> echo "source ~/ros2_ws/install/setup.bash" >> ~/.bashrc
> ```

---

## What You Should See

After launching, Gazebo Classic opens with:

1. **Two wooden tables** placed side-by-side
2. **Assembly board** (white) in the center with 9 color-coded target markers
3. **Three printer beds** (flat grey boxes) — Prusa MK3, MK4-1, MK4-2
4. **Nine grabbable parts** — 3 gears (green), 3 rect pins (yellow), 3 circ pins (orange)
5. **Overhead camera** streaming to `/overhead_camera/image_raw`
6. After ~30s: **xArm6** spawns on the right with built-in gripper
7. After ~32s: **UR5e** spawns on the left with OnRobot RG2 gripper

For detailed operation instructions, see [`docs/ros2_operation_guide_dual_robots.md`](docs/ros2_operation_guide_dual_robots.md).

---

## Environment Layout

```
            +X direction
              ↑
              |
     +--------+--------+
     |  MK4-1 |  MK4-2 |     ← Right side printers (X = +0.6)
     | (0.6,  | (0.6,  |
     |  0.4)  | -0.4)  |
     +--------+--------+
     |     Assembly     |     ← Center (0, 0)
     |      Board       |
     +--------+--------+
     |   Prusa MK3      |     ← Left side printer (X = -0.6)
     |  (-0.6, 0)       |
     +--------+--------+
              |
     ← -X direction

     xArm6: (0.0, -0.7, 1.021) facing -X (180° yaw)   [RIGHT, active control]
     UR5e:  (0.0,  0.7, 1.021) facing +X (0° yaw)      [LEFT, visual + RG2]
```

### Parts Distribution

| Printer | Gear (Green) | Rect Pin (Yellow) | Circ Pin (Orange) |
|---|---|---|---|
| MK3 (Left) | Small | Small | Small |
| MK4-1 (Right-Top) | Medium | Medium | Medium |
| MK4-2 (Right-Bot) | Large | Large | Large |

### Assembly Board Target Markers (3×3 Grid)

| Row | Y offset | Shape | Color | Sizes (left→right) |
|---|---|---|---|---|
| Top | +0.1 | Circles | Green | r=0.02, r=0.03, r=0.04 |
| Middle | 0.0 | Squares | Yellow | 0.02², 0.025², 0.03² |
| Bottom | -0.1 | Circles | Orange | r=0.01, r=0.013, r=0.015 |

---

## Robot Configuration

| | xArm6 | UR5e |
|---|---|---|
| **Namespace** | `xarm6` | `ur5e` |
| **TF prefix** | `xarm6_` | `ur5e_` |
| **Position** | (0.0, -0.7, 1.021) | (0.0, 0.7, 1.021) |
| **Yaw** | 180° (π) | 0° (0.0) |
| **Gripper** | Built-in xArm gripper | OnRobot RG2 (visual only) |
| **Control mode** | `gazebo_ros2_control` (active) | `use_fake_hardware` (static) |
| **Controllers** | joint_state_broadcaster + trajectory | None (frozen in place) |

---

## Viewing the Overhead Camera

In a **separate terminal** (no virtualenv active):

```bash
source /opt/ros/humble/setup.bash
ros2 run rqt_image_view rqt_image_view
```

Select `/overhead_camera/image_raw` from the dropdown.

---

## Custom Files Reference

Only **two files** in `cais_lab_gazebo/` need to be copied to reproduce the simulation.
Everything else comes from public upstream repositories:

| File | Purpose |
|---|---|
| `cais_lab_gazebo/launch/xarm6_ur5e_gazebo.launch.py` | Spawns both robots in the same world, injects RG2 gripper onto UR5e, starts xArm6 controllers |
| `cais_lab_gazebo/worlds/table.world` | Gazebo SDF world: tables, printers, parts with physics, assembly board with markers, overhead camera, gravity enabled |

### Upstream Dependencies (no modifications needed)

| Repository | Package | Purpose |
|---|---|---|
| [xArm-Developer/xarm_ros2](https://github.com/xArm-Developer/xarm_ros2) | `xarm_gazebo`, `xarm_description`, `xarm_controller`, etc. | xArm6 robot model and Gazebo integration |
| [Universal Robots](https://packages.ros.org) | `ur_description` (apt) | UR5e robot model |
| [tonydle/OnRobot_ROS2_Description](https://github.com/tonydle/OnRobot_ROS2_Description) | `onrobot_description` | RG2 gripper STL meshes and URDF |

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `ModuleNotFoundError: rclpy._rclpy_pybind11` | You are inside a Python virtualenv. Run `deactivate` first. |
| Gazebo window is blank/black for 60+ seconds | Normal on WSL2. Wait for software rendering to initialize. |
| Parts fly off the table | Verify `<gravity>0 0 -9.81</gravity>` in `table.world` (not `0 0 0`). |
| UR5e falls off the table | Verify `<gazebo><static>true</static></gazebo>` injection in launch file. |
| `spawn_entity.py` hangs forever | Kill all previous Gazebo processes with `killall -9 gzserver gzclient`. |
| Camera topic not visible in rqt | Ensure `libgazebo_ros_camera.so` plugin exists in `table.world`. |
| `No module named 'lxml'` | Poetry venv is active. Run `deactivate` before any ROS 2 command. |
| Only xArm6 visible, no UR5e | Wait 35+ seconds. UR5e spawns 2s after xArm6. |
| Can't control UR5e joints | Expected — UR5e is visual-only in dual mode (`use_fake_hardware`). |

---

## Next Steps

- **Operating the robots:** See [`docs/ros2_operation_guide_dual_robots.md`](docs/ros2_operation_guide_dual_robots.md)
- **xArm6 details:** See [`docs/ros2_operation_guide_xarm6.md`](docs/ros2_operation_guide_xarm6.md)
- **UR5e details:** See [`docs/ros2_operation_guide_ur5e.md`](docs/ros2_operation_guide_ur5e.md)
- **ProtoTwin integration:** See [`../docs/prototwin_setup_guide.md`](../docs/prototwin_setup_guide.md)
