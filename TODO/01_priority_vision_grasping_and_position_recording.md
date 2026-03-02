# TODO: Vision-Based Grasping & Position Recording — PRIORITY 1

## Goal

Enable efficient robot position teaching (replacing tedious RViz drag-and-plan) and integrate camera-based object detection for dynamic grasping in the dual xArm6 + UR5e setup.

## Prerequisites

- Dual Gazebo + MoveIt working: `ros2 launch xarm_gazebo dual_moveit_gazebo.launch.py`
- Overhead camera already publishing to `/overhead_camera/image_raw` (640x480, 15fps) via `table.world`

---

## Part 1: Position Recording via Keyboard Teleop + Joint Snapshot

### Status

- [x] `keyboard_teleop.py` — MoveIt-based keyboard jogging with Cartesian + Joint modes (created)
- [x] `record_pose.py` — joint state snapshot + save to JSON (created)
- [ ] Record all key named positions for UR5e and xArm6
- [ ] Store in `initialization/resources/robot_ur5e.json` and `robot_xarm6.json`

Scripts location: `ros2/cais_lab_gazebo/scripts/`

### Workflow (Step-by-Step)

**Terminal 1 — Launch the dual robot simulation:**

```bash
source /opt/ros/humble/setup.bash && source ~/ros2_ws/install/setup.bash
ros2 launch xarm_gazebo dual_moveit_gazebo.launch.py
```

Wait ~40s for Gazebo + MoveIt to fully load.

**Terminal 2 — Keyboard teleop (jog to desired poses):**

```bash
source /opt/ros/humble/setup.bash && source ~/ros2_ws/install/setup.bash
cd ~/projects/cais-spade-llm/ros2/cais_lab_gazebo/scripts
python3 keyboard_teleop.py
```

All motion goes through MoveIt plan+execute, so moves are **visible in both RViz and Gazebo** with collision checking.

Controls:
| Key | Action |
|-----|--------|
| `TAB` | Switch between xArm6 and UR5e |
| `M` | Toggle between **Cartesian** and **Joint** mode |
| **Cartesian mode** | |
| `UP`/`DOWN` | Move end-effector in X +/- |
| `LEFT`/`RIGHT` | Move end-effector in Y +/- |
| `;`/`.` | Move end-effector in Z up/down |
| **Joint mode** | |
| `1`-`6` | Select joint (auto-switches to joint mode) |
| `UP`/`DOWN` | Jog selected joint +/- |
| **Common** | |
| `+`/`-` | Increase/decrease step size (default: 10mm Cartesian, 2 deg joint) |
| `S` | Save current pose (prompts for a name, shows save path) |
| `Q` | Quit (auto-saves all positions on exit) |

Options: `--step-mm 10` (Cartesian step), `--step-deg 2` (joint step), `--robot ur5e`, `--mode joint`

Save path: `cais_spade_llm/initialization/resources/robot_xarm6.json` (or `robot_ur5e.json`).

**Terminal 2 (alternative) — Record pose separately:**

If you prefer using RViz drag-and-drop or `rqt_joint_trajectory_controller` for jogging, use `record_pose.py` in a separate terminal to snapshot:

```bash
# Print current joint positions
python3 record_pose.py --robot ur5e

# Save a named position to the robot's JSON config
python3 record_pose.py --robot ur5e --name "above_prusa_mk3" --save

# Interactive mode: record multiple positions
python3 record_pose.py --robot xarm6 --interactive --save

# Save to a custom file
python3 record_pose.py --robot ur5e --name "home" --output my_positions.json
```

### Positions to Record

Record for **both robots** (UR5e and xArm6):

- [ ] `home` — safe rest position
- [ ] `above_prusa_mk3` — approach height above Prusa MK3 printer bed
- [ ] `above_prusa_mk4_1` — approach height above Prusa MK4-1
- [ ] `above_prusa_mk4_2` — approach height above Prusa MK4-2
- [ ] `above_assembly_board` — approach height above assembly board center
- [ ] `pick_height_prusa_mk3` — actual pick height at MK3
- [ ] `place_assembly_board` — actual place height at assembly board

Output format (saved in robot JSON under `named_positions`):
```json
"named_positions": {
  "home": [0.0, -1.5708, 1.5708, -1.5708, -1.5708, 0.0],
  "above_prusa_mk3": [-0.523, -1.2, 1.8, -2.1, -1.57, 0.0],
  "above_assembly_board": [0.1, -1.4, 1.6, -1.7, -1.57, 0.0]
}
```

### Future: MoveIt Servo (Upgrade Path)

The keyboard teleop above sends incremental joint trajectory commands directly to the controllers. This works for position teaching but does **not** have collision checking.

For collision-checked real-time jogging, upgrade to MoveIt Servo later:
- `ros-humble-moveit-servo` is already installed
- `xarm_moveit_servo` package in `~/ros2_ws/` has keyboard input + servo configs
- Requires creating a servo launch that integrates with the dual combined URDF
- References:
  - [MoveIt Servo Tutorial (Humble)](https://moveit.picknik.ai/humble/doc/examples/realtime_servo/realtime_servo_tutorial.html)
  - [xArm Servo Config](~/ros2_ws/install/xarm_moveit_servo/share/xarm_moveit_servo/config/xarm_moveit_servo_config.yaml)
  - [Gamepad Teleoperation](https://moveit.picknik.ai/main/doc/how_to_guides/controller_teleoperation/controller_teleoperation.html)

---

## Part 2: Vision-Based Dynamic Grasping Pipeline

### Why

In real life, you'd use YOLO to detect objects and compute grasp poses. In ROS2/Gazebo, the same pipeline works — simulated cameras publish standard ROS2 image topics that any detector can consume.

### How Other Researchers Do It

The standard perception-to-grasp pipeline in ROS2:

```
Gazebo Camera Plugin ──► /camera/image_raw (sensor_msgs/Image)
                    ──► /camera/depth/image_raw (optional)
                    ──► /camera/points (optional)
        │
        ▼
YOLO Detector Node ──► Bounding boxes + class labels
        │
        ▼
3D Pose Estimation ──► Object position in robot base frame
        │
        ▼
MoveIt Grasp Plan  ──► Collision-free trajectory to grasp pose
        │
        ▼
Execute + Gripper  ──► Pick up the object
```

**Common packages used by researchers:**
- [Ultralytics YOLOv8](https://docs.ultralytics.com/guides/ros-quickstart/) — `pip install ultralytics`, wrap in a ROS2 node (most common, simplest)
- [yolov8_ros](https://github.com/mgonzs13/yolov8_ros) — community ROS2 wrapper for YOLOv8
- [darknet_ros](https://github.com/leggedrobotics/darknet_ros) — older YOLO ROS package (battle-tested)
- [easy_perception_deployment](https://github.com/ros-industrial/easy_manipulation_deployment) — ROS-Industrial modular perception stack

**Grasp pose estimation approaches:**
- **Simple (our case)**: Known flat table + known object shapes → top-down grasp at detection centroid. Our `grasp_strategies` in robot JSONs (`"top"`, `"above-top"`, `"center"`) already define this.
- **GPD**: [Grasp Pose Detection](https://moveit.picknik.ai/humble/doc/examples/moveit_deep_grasps/moveit_deep_grasps_tutorial.html) — 6-DOF grasp from point clouds
- **Dex-Net**: Deep learning grasp quality estimation
- **GraspNet**: End-to-end grasp detection network

### Phase 2.1: Add Depth Camera to Gazebo World

- [ ] Modify `ros2/cais_lab_gazebo/worlds/table.world` — add depth camera plugin to existing overhead camera
  - Publishes `/overhead_camera/depth/image_raw` (for 3D projection)
  - Publishes `/overhead_camera/points` (point cloud, optional)
  - Keep existing RGB at `/overhead_camera/image_raw`

### Phase 2.2: Create YOLO Perception ROS2 Node

- [ ] Install: `pip install ultralytics`
- [ ] Create `ros2/cais_lab_gazebo/sensor/gazebo_camera_detector.py`
  - Subscribes to `/overhead_camera/image_raw`
  - Runs YOLOv8 inference (pretrained or fine-tuned on Gazebo screenshots)
  - Publishes detections as `vision_msgs/Detection2DArray`
  - Provides a ROS2 service `/detect_part`:
    - Request: `part_name` (string)
    - Response: `{"x": float, "y": float, "z": float, "detected": bool}`
  - For simulation, can start with OpenCV color-based detection (parts are green/yellow/orange) before training YOLO

### Phase 2.3: 2D → 3D Projection

- [ ] Implement pixel-to-world coordinate transformation
  - **With depth camera**: Use depth value at detection centroid + `image_geometry` package for projection
  - **Overhead camera shortcut**: Known camera height (2.0m) + known table height (1.021m) → simple pinhole model
  - Camera calibration matrices already defined in robot JSON configs (`cameraA_offsets`, `cameraB_offsets`)

### Phase 2.4: Integrate with CameraModule

- [ ] Modify `cais_spade_llm/resources/sensor/camera_module.py`
  - Add `ROS2CameraModule` subclass that calls `/detect_part` service
  - Falls back to mock if ROS2 is not available
  - Returns same `{"x": float, "y": float, "z": float}` interface

### Key Files

| File | Action |
|------|--------|
| `ros2/cais_lab_gazebo/worlds/table.world` | Add depth camera plugin |
| `ros2/cais_lab_gazebo/sensor/gazebo_camera_detector.py` | New — YOLO detection node |
| `ros2/cais_lab_gazebo/launch/servo_keyboard.launch.py` | New — MoveIt Servo keyboard launch |
| `ros2/cais_lab_gazebo/scripts/record_pose.py` | New — joint state snapshot CLI |
| `cais_spade_llm/resources/sensor/camera_module.py` | Replace mock with ROS2 service client |
| `cais_spade_llm/initialization/resources/robot_ur5e.json` | Add `named_positions` |
| `cais_spade_llm/initialization/resources/robot_xarm6.json` | Add `named_positions` |

---

## Implementation Order

| Priority | Step | What | Depends on |
|----------|------|------|------------|
| **1a** | Servo setup | MoveIt Servo keyboard launch for dual robots | MoveIt Servo package |
| **1b** | Record tool | `record_pose.py` CLI | ROS2 running |
| **1c** | Record positions | All named positions for both robots | Steps 1a + 1b |
| **1d** | Update configs | `named_positions` in robot JSONs | Step 1c |
| **2a** | Depth camera | Add depth plugin to `table.world` | — |
| **2b** | Perception node | YOLO detection ROS2 node | Step 2a, ultralytics |
| **2c** | 3D projection | Pixel → world coordinates | Step 2b |
| **2d** | Camera integration | Update `CameraModule` to use detections | Step 2c |

**Do Part 1 (position recording) first** — it unblocks the `MoveItBackend` TODO and doesn't require any new packages beyond what's already installed.

---

## Related TODOs

- `connect_ros2_moveit_to_agent.md` — Phase 3 (position teaching) is addressed by Part 1 here
