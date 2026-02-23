# ROS2 UR5e Operation Guide

How to run the UR5e simulation, read positions, plan motions, and control from Python.

---

## 1. Launch the Simulation

### Terminal 1 — Gazebo + RViz
```bash
source /opt/ros/jazzy/setup.bash
ros2 launch ur_simulation_gz ur_sim_control.launch.py ur_type:=ur5e
```
- Opens **Gazebo** (physics simulation) and **RViz** (visualization)
- Wait until both windows appear and the robot is visible
- The UR5e starts in its default pose

### Terminal 2 — MoveIt2 (optional, for drag-to-plan)
```bash
source /opt/ros/jazzy/setup.bash
ros2 launch ur_moveit_config ur_moveit.launch.py ur_type:=ur5e launch_rviz:=true use_sim_time:=true
```
- Adds motion planning to RViz (drag end-effector → Plan → Execute)
- Only needed if you want interactive Cartesian planning
- Skip this if you only need Python control

---

## 2. Read Current Robot Position

### Joint angles (6 values in radians)
```bash
source /opt/ros/jazzy/setup.bash
ros2 topic echo /joint_states --once
```
Output:
```
name: [shoulder_pan_joint, shoulder_lift_joint, elbow_joint, wrist_1_joint, wrist_2_joint, wrist_3_joint]
position: [0.0, -1.57, 1.57, -1.57, -1.57, 0.0]
```

### TCP position (XYZ of end-effector)
```bash
ros2 run tf2_ros tf2_echo base_link tool0
```
Output:
```
Translation: [0.400, -0.200, 0.172]
Rotation: [...]
```

---

## 3. Move the Robot

### Option A: Python script (direct joint control)
```python
from cais_spade_llm.resources.robot.ur5e_controller import UR5eController
import math, time

ctrl = UR5eController()
ctrl.init()
time.sleep(1)  # wait for publisher to register

# All 6 joints in radians: [shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3]
ctrl.move_joints([0.0, -1.57, 1.57, -1.57, -1.57, 0.0], duration_sec=3)
time.sleep(4)
ctrl.shutdown()
```

Run with:
```bash
source /opt/ros/jazzy/setup.bash
cd ~/projects/cais-spade-llm
python3 your_script.py
```

### Option B: Command line (quick test)
```bash
ros2 topic pub --once /scaled_joint_trajectory_controller/joint_trajectory \
  trajectory_msgs/msg/JointTrajectory \
  "{joint_names: ['shoulder_pan_joint','shoulder_lift_joint','elbow_joint','wrist_1_joint','wrist_2_joint','wrist_3_joint'], \
    points: [{positions: [0.0, -1.57, 1.57, -1.57, -1.57, 0.0], time_from_start: {sec: 2}}]}"
```

### Option C: Joint sliders GUI
```bash
ros2 run rqt_joint_trajectory_controller rqt_joint_trajectory_controller
```
Opens a window with 6 sliders — drag to move joints in real-time.

### Option D: MoveIt2 drag-and-plan (requires Terminal 2)
1. In RViz, drag the **orange ball** (interactive marker) on the end-effector
2. In the MoveIt2 panel → click **Plan** → see the trajectory
3. Click **Execute** → robot moves in Gazebo

---

## 4. Key ROS2 Topics

| Topic | Type | Purpose |
|---|---|---|
| `/joint_states` | `sensor_msgs/JointState` | Current joint positions (read) |
| `/scaled_joint_trajectory_controller/joint_trajectory` | `trajectory_msgs/JointTrajectory` | Send joint commands (write) |
| `/tf` | `tf2_msgs/TFMessage` | All frame transforms including TCP |

---

## 5. UR5e Joint Reference

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

## 6. Common Poses

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

## 7. Troubleshooting

| Problem | Fix |
|---|---|
| No Gazebo/RViz window | Check `echo $DISPLAY` — must show `:0` for WSLg |
| RViz window too big | `export QT_SCALE_FACTOR=0.7` before launching |
| Robot doesn't move | Ensure Gazebo launched first, controllers loaded (`ros2 topic list` shows `/scaled_joint_trajectory_controller/`) |
| RViz and Gazebo out of sync | Kill everything, restart Gazebo first, then MoveIt2 with `use_sim_time:=true` |
| `source` keeps being needed | Add `source /opt/ros/jazzy/setup.bash` to your `~/.bashrc` |
