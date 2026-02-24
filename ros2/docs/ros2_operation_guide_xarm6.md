# ROS2 xArm6 Operation Guide

How to run the xArm6 simulation, read positions, plan motions, and control from Python.

> **Simulator:** xArm6 uses **Gazebo Classic** (the `gazebo` command) via `xarm_gazebo`.
> This is different from UR5e, which uses Gazebo Ignition.
> Do not mix the two; each robot has its own simulation environment.

---

## 0. Clean Everything First

If Gazebo exited improperly, the ports might still be blocked. Clean the processes first:

```bash
killall -9 gzserver gzclient robot_state_publisher spawner spawn_entity.py ros2
pkill -9 -f gazebo
```

---

## CRITICAL: Two-Terminal Rule

> **Never run ROS2/Gazebo commands with the Poetry venv activated.**

If your Poetry venv is active (`which python3` points to `.venv/bin/python3`),
ROS2 tools like `spawn_entity.py` will fail with `No module named 'lxml'`.

**Before running any ROS2 command, deactivate the venv:**
```bash
deactivate   # if venv is currently active
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
```

---

## 1. Launch the Simulation

### Terminal 1 — Gazebo Classic (simulation)
```bash
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
ros2 launch xarm_gazebo xarm6_beside_table_gazebo.launch.py add_gripper:=true
```
- Opens **Gazebo Classic** and loads the robot controllers
- `add_gripper:=true` attaches the xArm gripper; omit if using a custom end-effector
- Wait until Gazebo appears and the robot is visible (can take 30–60 seconds in WSL)
- The xArm6 starts in its default pose

### Terminal 2 — MoveIt2 (optional, for drag-to-plan)
```bash
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
ros2 launch xarm_moveit_config xarm6_moveit_gazebo.launch.py add_gripper:=true
```
- Adds motion planning to RViz (drag end-effector → Plan → Execute)
- Only needed if you want interactive Cartesian planning
- Skip this if you only need Python control

---

## 2. Read Current Robot Position

### Joint angles (6 values in radians)
```bash
source /opt/ros/humble/setup.bash
ros2 topic echo /joint_states --once
```
Output:
```
name: [joint1, joint2, joint3, joint4, joint5, joint6]
position: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
```
> Note: output is alphabetically ordered — reorder to [joint1..joint6] manually.

### TCP position (XYZ of end-effector)
```bash
ros2 run tf2_ros tf2_echo link_base link_eef
```
Output:
```
Translation: [x, y, z]
Rotation: [...]
```

---

## 3. Move the Robot

### Option A: Python script (direct joint control)
```python
from cais_spade_llm.resources.robot.xarm6_controller import XArm6Controller
import time

ctrl = XArm6Controller()
ctrl.init()
time.sleep(1)  # wait for publisher to register

# All 6 joints in radians: [joint1, joint2, joint3, joint4, joint5, joint6]
ctrl.move_joints([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], duration_sec=3)
time.sleep(4)
ctrl.shutdown()
```

> Note: `XArm6Controller` is currently a **placeholder** (mock). It does not yet
> communicate with Gazebo or real hardware. See `cais_spade_llm/resources/robot/xarm6_controller.py`.

Run with (do NOT have Poetry venv active):
```bash
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
cd ~/projects/cais-spade-llm
python3 your_script.py
```

### Option B: Command line (quick test)
```bash
ros2 topic pub --once /xarm6_traj_controller/joint_trajectory \
  trajectory_msgs/msg/JointTrajectory \
  "{joint_names: ['joint1','joint2','joint3','joint4','joint5','joint6'], \
    points: [{positions: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], time_from_start: {sec: 2}}]}"
```

### Option C: Joint sliders GUI
```bash
ros2 run rqt_joint_trajectory_controller rqt_joint_trajectory_controller
```
Opens a window with 6 sliders — drag to move joints in real-time.

### Option D: MoveIt2 drag-and-plan (requires Terminal 2)
1. In RViz, drag the **interactive marker** on the end-effector
2. In the MoveIt2 panel → click **Plan** → see the trajectory
3. Click **Execute** → robot moves in Gazebo

### Option E: xArm SDK (real hardware only)
```python
from xarm.wrapper import XArmAPI

arm = XArmAPI('192.168.1.100')  # replace with your xArm IP
arm.motion_enable(enable=True)
arm.set_mode(0)   # position control mode
arm.set_state(0)  # sport state

# Move to joint angles (degrees for SDK, not radians)
arm.set_servo_angle(angle=[0, 0, 0, 0, 0, 0], speed=30, wait=True)
arm.disconnect()
```
> Note: the xArm SDK uses **degrees**, not radians. Convert: `degrees = radians * (180 / math.pi)`

---

## 4. Key ROS2 Topics

| Topic | Type | Purpose |
|---|---|---|
| `/joint_states` | `sensor_msgs/JointState` | Current joint positions (read) |
| `/xarm6_traj_controller/joint_trajectory` | `trajectory_msgs/JointTrajectory` | Send joint commands (write) |
| `/tf` | `tf2_msgs/TFMessage` | All frame transforms including TCP |
| `/xarm6_traj_controller/state` | `control_msgs/JointTrajectoryControllerState` | Controller state |
| `/controller_manager/*` | — | Controller lifecycle (at root `/`, not `/xarm6/`) |

> **Note on controller namespace:** `gazebo_ros2_control` starts the `controller_manager`
> at `/controller_manager` (root namespace), not `/xarm6/controller_manager`.
> When spawning controllers manually, always use `--controller-manager /controller_manager`.

---

## 5. xArm6 Joint Reference

| Index | Joint Name | Range | Axis |
|---|---|---|---|
| 0 | `joint1` | ±360° | Z (base rotation) |
| 1 | `joint2` | ±118° | Y |
| 2 | `joint3` | ±225° | Y |
| 3 | `joint4` | ±360° | X |
| 4 | `joint5` | ±97°  | Y |
| 5 | `joint6` | ±360° | Z |

All ROS2 values are in **radians**. π/2 ≈ 1.5708 = 90°.
xArm SDK uses **degrees**.

---

## 6. Common Poses

```python
POSES = {
    # All joints at zero — arm pointing straight up
    "zero": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],

    # Standard home — arm folded back, out of the way
    "home": [0.0, -0.349, 0.0, 0.349, 0.0, 0.349],

    # Arm stretched forward horizontally
    "forward": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
}
```

---

## 7. Real Hardware Setup

```bash
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
ros2 launch xarm_api xarm6_driver.launch.py robot_ip:=192.168.1.100
```
- Same topics as simulation — your Python code and SPADE agents work unchanged
- Set `use_sim_time:=false` (or omit it) for real hardware
- Enable robot on teach pendant first (mode 0, state 0)
- Test with slow speeds before running full trajectories

---

## 8. UR5e vs xArm6 Quick Comparison

| | UR5e | xArm6 |
|---|---|---|
| Simulator | Gazebo Ignition (`gz`) | Gazebo Classic (`gazebo`) |
| Launch package | `ur_simulation_gz` | `xarm_gazebo` |
| Controller topic | `/scaled_joint_trajectory_controller/joint_trajectory` | `/xarm6_traj_controller/joint_trajectory` |
| TCP frame | `tool0` | `link_eef` |
| Base frame | `base_link` | `link_base` |
| SDK | `ur_robot_driver` (ROS2 only) | `xarm-python-sdk` (direct IP) |
| SDK units | radians | degrees |

---

## 9. Troubleshooting

| Problem | Fix |
|---|---|
| `No module named 'lxml'` | Poetry venv is active. Run `deactivate` first, then re-run the ROS2 command. |
| No Gazebo/RViz window | Check `echo $DISPLAY` — must show `:0` for WSLg |
| Gazebo takes very long to start | Normal in WSL — wait 30–60 seconds before assuming failure |
| Robot doesn't move | Ensure Gazebo launched first, check `/xarm6_traj_controller/` appears in `ros2 topic list` |
| Controller spawner: `controller_manager service not available` | Use `--controller-manager /controller_manager` (root namespace) |
| RViz and Gazebo out of sync | Kill everything, restart Gazebo first, then MoveIt2 with `use_sim_time:=true` |
| xArm SDK connection refused | Check robot IP, ensure robot is powered on and in remote mode |
| `source` keeps being needed | Add `source /opt/ros/humble/setup.bash` and `source ~/ros2_ws/install/setup.bash` to `~/.bashrc` |
