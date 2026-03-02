# Keyboard Teleop & Position Recording Guide

How to jog the xArm6 and UR5e robots using keyboard controls and save named positions.

Replaces the tedious RViz drag-and-plan workflow with direct keyboard control through MoveIt.

---

## Prerequisites

- Dual Gazebo + MoveIt working (see `ros2/docs/ros2_operation_guide_dual_robots.md`)
- Must use `python3.10` (not `python3`) — ROS2 Humble C extensions require Python 3.10

---

## Step 1: Launch the Simulation

**Terminal 1:**

```bash
source /opt/ros/humble/setup.bash && source ~/ros2_ws/install/setup.bash
ros2 launch xarm_gazebo dual_moveit_gazebo.launch.py
```

Wait ~40 seconds for Gazebo + MoveIt + RViz to fully load. You should see both robots in RViz.

---

## Step 2: (Optional) Start Perception Node

The perception node provides ground truth part positions from Gazebo via `/detect_part` and `/detect_all` services.

**Terminal 2:**

```bash
source /opt/ros/humble/setup.bash && source ~/ros2_ws/install/setup.bash
python3.10 ~/projects/cais-spade-llm/ros2/cais_lab_gazebo/sensor/gazebo_camera_detector.py
```

You should see:
```
PerceptionNode started (Gazebo ground truth mode) — 9 parts registered
Connected to /get_entity_state service
```

Test it:
```bash
ros2 param set /perception_node target_part SG
ros2 service call /detect_part std_srvs/srv/Trigger
ros2 service call /detect_all std_srvs/srv/Trigger
```

To test the CameraModule integration independently:
```bash
python3.10 ~/projects/cais-spade-llm/cais_spade_llm/resources/sensor/camera_module.py
```

---

## Step 3: Keyboard Teleop

**Terminal 3:**

```bash
source /opt/ros/humble/setup.bash && source ~/ros2_ws/install/setup.bash && cd ~/projects/cais-spade-llm/ros2/cais_lab_gazebo/scripts && python3.10 keyboard_teleop.py
```

The script waits for `/joint_states`, `/move_action`, and `/compute_cartesian_path` to become available, then prints `=== Ready! ===`.

### CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `--robot` | `xarm6` | Start with `xarm6` or `ur5e` |
| `--step-mm` | `10` | Initial Cartesian step size in mm |
| `--step-deg` | `1.0` | Initial joint step size in degrees |
| `--step-scale` | `1.5` | `+/-` multiplier for step adjustments |
| `--cart-max-step-mm` | `30` | Cartesian interpolation step (`higher = faster`) |
| `--joint-plan-time` | `0.8` | Joint planning time budget in seconds |
| `--joint-plan-attempts` | `1` | Joint planning attempts per keypress |
| `--key-poll-ms` | `20` | Keyboard polling interval in milliseconds |

Example (faster + precise): `python3.10 keyboard_teleop.py --robot ur5e --step-mm 5 --step-deg 0.3`

### Cartesian Mode (default)

Each keypress immediately plans and executes a move through MoveIt. The robot moves visibly in both RViz and Gazebo.

| Keys | Action |
|------|--------|
| `Z` + Arrow UP | Move Z+ (up) |
| `Z` + Arrow DOWN | Move Z- (down) |
| `X` + Arrow UP | Move X+ (forward) |
| `X` + Arrow DOWN | Move X- (backward) |
| `Y` + Arrow UP | Move Y+ (left) |
| `Y` + Arrow DOWN | Move Y- (right) |
| Arrow UP/DOWN alone | Jog last-selected axis (default: Z) |
| `+` / `-` | Increase / decrease step size |

**How axis+arrow works:** Press the axis letter (e.g. `Z`), then press the arrow key. The script detects the combo and executes immediately. If you press just the axis letter without an arrow, it selects that axis for future bare arrow presses.

### Joint Mode

| Keys | Action |
|------|--------|
| `1` - `6` | Select joint (auto-switches to joint mode) |
| Arrow UP / DOWN | Jog selected joint + / - |
| `+` / `-` | Increase / decrease step size |

Joint names for each robot:

| Joint # | xArm6 | UR5e |
|---------|-------|------|
| 1 | joint1 | shoulder_pan_joint |
| 2 | joint2 | shoulder_lift_joint |
| 3 | joint3 | elbow_joint |
| 4 | joint4 | wrist_1_joint |
| 5 | joint5 | wrist_2_joint |
| 6 | joint6 | wrist_3_joint |

### Common Controls

| Key | Action |
|-----|--------|
| `TAB` | Switch between xArm6 and UR5e |
| `M` | Switch back to Cartesian mode |
| `P` | Apply precision profile (small increments, smoother planning) |
| `F` | Apply fast profile (larger increments, faster planning) |
| `S` | Save current position (prompts for a name) |
| `Q` | Quit |

---

## Step 4: Saving Positions

Press `S` during keyboard teleop to save the current joint positions:

1. The terminal prompts: `Position name: `
2. Type a name (e.g. `home`, `above_prusa_mk3`) and press Enter
3. The position is saved immediately to the robot's JSON config file

### Save Locations

| Robot | File |
|-------|------|
| xArm6 | `cais_spade_llm/initialization/resources/robot_xarm6.json` |
| UR5e | `cais_spade_llm/initialization/resources/robot_ur5e.json` |

### Saved Format

Positions are stored under the `named_positions` key as arrays of joint angles in radians:

```json
{
  "named_positions": {
    "home": [0.0, -1.5708, 1.5708, -1.5708, -1.5708, 0.0],
    "above_prusa_mk3": [-0.523, -1.2, 1.8, -2.1, -1.57, 0.0]
  }
}
```

### Current Saved Positions

**xArm6** (`robot_xarm6.json`):
- `home` — rest position
- `test`, `test2` — test positions

**UR5e** (`robot_ur5e.json`): check file for current positions.

### Positions to Record

Record these named positions for **both** xArm6 and UR5e:

- [ ] `home` — safe rest position
- [ ] `above_prusa_mk3` — approach height above Prusa MK3 printer bed
- [ ] `above_prusa_mk4_1` — approach height above Prusa MK4-1
- [ ] `above_prusa_mk4_2` — approach height above Prusa MK4-2
- [ ] `above_assembly_board` — approach height above assembly board center
- [ ] `pick_height_prusa_mk3` — actual pick height at MK3
- [ ] `place_assembly_board` — actual place height at assembly board

---

## Alternative: Record Pose CLI

If you prefer to jog the robot using RViz drag-and-drop or another method, use `record_pose.py` to snapshot the current joint positions separately.

```bash
cd ~/projects/cais-spade-llm/ros2/cais_lab_gazebo/scripts
```

### Print Current Positions

```bash
python3.10 record_pose.py --robot ur5e
```

### Save a Named Position to Robot JSON

```bash
python3.10 record_pose.py --robot ur5e --name "above_prusa_mk3" --save
```

### Save to a Custom File

```bash
python3.10 record_pose.py --robot ur5e --name "home" --output my_positions.json
```

### Interactive Batch Mode

Record multiple positions in one session:

```bash
python3.10 record_pose.py --robot xarm6 --interactive --save
```

In interactive mode:
- Type a position name and press Enter to record
- Type `list` to see all saved positions
- Type `quit` to exit

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError: No module named 'rclpy'` | Use `python3.10` (or a Python 3.10 venv) instead of a newer interpreter. ROS2 Humble C extensions are built for Python 3.10. |
| `rclpy._rclpy_pybind11` error | Same issue — wrong Python version. Use `python3.10`. |
| Arrow keys show `^[[A` garbage | Normal during execution — queued keypresses are flushed after each move completes. |
| `ERROR: /move_action not available` | MoveIt hasn't started yet. Wait longer after launch (~40s). |
| `No TF data` | TF buffer needs time to fill. Wait a few seconds after `=== Ready! ===`. |
| `Path incomplete (X%)` | Target position may be unreachable. Try a smaller step size (`-` key). |
| `Goal rejected` | MoveIt can't plan to the target. The robot may be at a joint limit. Try a different direction. |
| Save doesn't work | Check that `cais_spade_llm/initialization/resources/robot_xarm6.json` exists and is writable. |
| Perception node: `/get_entity_state not available` | The `libgazebo_ros_state.so` plugin must be in `table.world`. Restart the simulation after adding it. |
| Perception node: `Failed to get state` | Deadlock issue — make sure `gazebo_camera_detector.py` uses `MultiThreadedExecutor` with separate callback groups. |

---

## Script & Node Locations

```
ros2/cais_lab_gazebo/scripts/keyboard_teleop.py       — keyboard jogging via MoveIt
ros2/cais_lab_gazebo/scripts/record_pose.py            — standalone position snapshot CLI
ros2/cais_lab_gazebo/scripts/save_camera_frame.py      — diagnostic: save camera frames to /tmp
ros2/cais_lab_gazebo/sensor/gazebo_camera_detector.py   — Gazebo ground truth perception (ROS2 node)
ros2/cais_lab_gazebo/launch/perception.launch.py       — launch file for perception node
ros2/cais_lab_gazebo/launch/dual_moveit_gazebo.launch.py — main simulation launch
cais_spade_llm/resources/sensor/camera_module.py   — SPADE-side camera client (mock or ROS2)
```
