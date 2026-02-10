# ROS2 + Gazebo + UR5e Setup Guide
# From scratch on Windows WSL2 to running robot simulation

---

## PREREQUISITES: Windows Setup

### 1. Check Windows Version
- Windows 11 (any version) OR
- Windows 10 Build 19044+

### 2. Install / Update WSL2
Open **PowerShell as Administrator** on Windows:
```powershell
# Install WSL2 with Ubuntu 24.04
wsl --install -d Ubuntu-24.04

# If already installed, update WSL
wsl --update

# Verify WSL and WSLg versions
wsl --version
```

Expected output should show:
- WSL version: 2.x.x
- WSLg version: 1.x.x

### 3. Verify WSLg is Working
Inside WSL terminal:
```bash
# Should output ":0"
echo $DISPLAY

# Should output "wayland-0"
echo $WAYLAND_DISPLAY
```

### 4. Test GUI works (optional)
```bash
sudo apt install x11-apps -y
xclock  # A clock window should appear on Windows desktop
```

---

## PART 1: Install ROS2 Jazzy (for Ubuntu 24.04)

```bash
# 1. Set up locale
sudo apt update && sudo apt install locales -y
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8

# 2. Enable Universe repository
sudo apt install software-properties-common -y
sudo add-apt-repository universe -y

# 3. Add ROS2 GPG key
sudo apt update && sudo apt install curl -y
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
  -o /usr/share/keyrings/ros-archive-keyring.gpg

# 4. Add ROS2 repository
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
  http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | \
  sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

# 5. Install ROS2 Jazzy Desktop (takes 5-10 min)
sudo apt update
sudo apt upgrade -y
sudo apt install ros-jazzy-desktop -y

# 6. Install dev tools
sudo apt install ros-dev-tools -y

# 7. Add to shell (run once)
echo "source /opt/ros/jazzy/setup.bash" >> ~/.bashrc
source ~/.bashrc

# 8. Verify
ros2 pkg list | head -5
```

---

## PART 2: Install Gazebo Harmonic

```bash
# 1. Add Gazebo repository
sudo wget https://packages.osrfoundation.org/gazebo.gpg \
  -O /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg

echo "deb [arch=$(dpkg --print-architecture) \
  signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] \
  http://packages.osrfoundation.org/gazebo/ubuntu-stable \
  $(lsb_release -cs) main" | \
  sudo tee /etc/apt/sources.list.d/gazebo-stable.list > /dev/null

# 2. Install Gazebo Harmonic
sudo apt update
sudo apt install gz-harmonic -y

# 3. Install ROS2-Gazebo bridge packages
sudo apt install ros-jazzy-ros-gz -y
sudo apt install ros-jazzy-ros-gz-sim ros-jazzy-ros-gz-bridge ros-jazzy-ros-gz-image -y
sudo apt install ros-jazzy-ros-gz-sim-demos -y

# 4. Verify
gz sim --version
```

---

## PART 3: Install UR Robot Packages

```bash
# Install UR description and driver packages
sudo apt install ros-jazzy-ur-description -y
sudo apt install ros-jazzy-ur-robot-driver -y
sudo apt install ros-jazzy-ur -y
sudo apt install ros-jazzy-ur-moveit-config -y

# Install UR Gazebo simulation package
sudo apt install ros-jazzy-ur-simulation-gz -y

# Verify
source /opt/ros/jazzy/setup.bash
ros2 pkg list | grep "^ur"
```

Expected output:
```
ur_client_library
ur_controllers
ur_dashboard_msgs
ur_description
ur_moveit_config
ur_msgs
ur_robot_driver
```

---

## PART 4: Launch UR5e in Gazebo + RViz

### Terminal 1 - Launch simulation:
```bash
source /opt/ros/jazzy/setup.bash
ros2 launch ur_simulation_gz ur_sim_control.launch.py ur_type:=ur5e
```

This opens:
- **Gazebo** window: 3D physics simulation with UR5e robot
- **RViz** window: visualization dashboard

### Optional flags:
```bash
# Without RViz
ros2 launch ur_simulation_gz ur_sim_control.launch.py ur_type:=ur5e launch_rviz:=false

# Without Gazebo GUI (headless)
ros2 launch ur_simulation_gz ur_sim_control.launch.py ur_type:=ur5e gazebo_gui:=false
```

---

## PART 5: Move the Robot

### Option A: Quick test via terminal (Terminal 2)
```bash
source /opt/ros/jazzy/setup.bash
ros2 topic pub /scaled_joint_trajectory_controller/joint_trajectory \
  trajectory_msgs/msg/JointTrajectory \
  "{
    joint_names: [shoulder_pan_joint, shoulder_lift_joint, elbow_joint, wrist_1_joint, wrist_2_joint, wrist_3_joint],
    points: [{
      positions: [0.0, -1.57, 1.57, -1.57, -1.57, 0.0],
      time_from_start: {sec: 3}
    }]
  }" --once
```

### Option B: Python script (move through multiple poses)

Save as `move_ur5e.py`:
```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
import time

POSES = {
    "home":     [0.0,   -1.57,  0.0,   -1.57,  0.0,   0.0],
    "up":       [0.0,   -1.57,  0.5,   -1.57,  0.0,   0.0],
    "left":     [1.57,  -1.57,  1.57,  -1.57, -1.57,  0.0],
    "right":    [-1.57, -1.57,  1.57,  -1.57,  1.57,  0.0],
    "reach_fwd":[0.0,   -1.0,   1.5,   -2.0,  -1.57,  0.0],
}

JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

class UR5eMover(Node):
    def __init__(self):
        super().__init__("ur5e_mover")
        self.publisher = self.create_publisher(
            JointTrajectory,
            "/scaled_joint_trajectory_controller/joint_trajectory",
            10,
        )
        time.sleep(1.0)
        self.get_logger().info("UR5e mover ready!")

    def move_to(self, pose_name: str, duration_sec: int = 3):
        positions = POSES[pose_name]
        self.get_logger().info(f"Moving to '{pose_name}'...")
        msg = JointTrajectory()
        msg.joint_names = JOINT_NAMES
        point = JointTrajectoryPoint()
        point.positions = positions
        point.time_from_start = Duration(sec=duration_sec)
        msg.points = [point]
        self.publisher.publish(msg)
        time.sleep(duration_sec + 0.5)
        self.get_logger().info(f"Reached '{pose_name}'")

def main():
    rclpy.init()
    mover = UR5eMover()
    sequence = [
        ("home",      3),
        ("up",        3),
        ("reach_fwd", 3),
        ("left",      3),
        ("right",     3),
        ("home",      3),
    ]
    for pose_name, duration in sequence:
        mover.move_to(pose_name, duration)
    mover.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
```

Run it (Terminal 2, while Gazebo is running):
```bash
source /opt/ros/jazzy/setup.bash
python3 move_ur5e.py
```

---

## QUICK REFERENCE

| What | Command |
|------|---------|
| Start simulation | `ros2 launch ur_simulation_gz ur_sim_control.launch.py ur_type:=ur5e` |
| Check controllers | `ros2 topic list \| grep trajectory` |
| Check robot state | `ros2 topic echo /joint_states` |
| Run move script | `python3 move_ur5e.py` |
| Gazebo only | `gz sim empty.sdf` |
| View robot model | `ros2 launch ur_description view_ur.launch.py ur_type:=ur5e` |

---

## SYSTEM SUMMARY

| Component | Version |
|-----------|---------|
| OS | Ubuntu 24.04 LTS (Noble) |
| WSL | 2.x with WSLg |
| ROS2 | Jazzy |
| Gazebo | Harmonic |
| Robot | UR5e (Universal Robots) |
| MoveIt2 | Included with ros-jazzy-ur-moveit-config |
