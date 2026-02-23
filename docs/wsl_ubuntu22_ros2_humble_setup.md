# WSL Ubuntu 22.04 + ROS2 Humble Setup Guide

Install Ubuntu 22.04 alongside existing Ubuntu 24.04, then set up ROS2 Humble with UR5e and xArm6.

---

## 0. Backup Current Work (Windows PowerShell)

```powershell
# Optional but recommended — export your current Ubuntu 24 as a backup
wsl --export Ubuntu-24.04 C:\backup\ubuntu24.tar
```

---

## 1. Install Ubuntu 22.04 on WSL (Windows PowerShell)

```powershell
wsl --install -d Ubuntu-22.04
```

- This installs alongside your existing Ubuntu 24.04
- Both distros are available simultaneously
- Switch between them with: `wsl -d Ubuntu-22.04` or `wsl -d Ubuntu-24.04`

Set 22.04 as your default if desired:
```powershell
wsl --set-default Ubuntu-22.04
```

---

## 2. First Boot — Initial Setup (inside Ubuntu 22.04)

Open Ubuntu 22.04 from the Start menu or:
```powershell
wsl -d Ubuntu-22.04
```

Update the system:
```bash
sudo apt update && sudo apt upgrade -y
```

---

## 3. Install ROS2 Humble

```bash
# Set locale
sudo apt install -y locales
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8

# Add ROS2 apt repo
sudo apt install -y software-properties-common curl
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
  -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
  http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" \
  | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

# Install ROS2 Humble desktop
sudo apt update
sudo apt install -y ros-humble-desktop

# Install colcon and rosdep
sudo apt install -y python3-colcon-common-extensions python3-rosdep
sudo rosdep init
rosdep update
```

Add to `~/.bashrc` so it sources automatically:
```bash
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

---

## 4. Install UR5e Packages

```bash
sudo apt install -y \
  ros-humble-ur \
  ros-humble-ur-robot-driver \
  ros-humble-ur-simulation-gz \
  ros-humble-ur-moveit-config
```

Test:
```bash
source /opt/ros/humble/setup.bash
ros2 launch ur_simulation_gz ur_sim_control.launch.py ur_type:=ur5e launch_rviz:=false
```

---

## 5. Install xArm6 Packages

```bash
# Install Gazebo Classic dependencies
sudo apt install -y ros-humble-gazebo-ros-pkgs

# Create workspace
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src
git clone https://github.com/xArm-Developer/xarm_ros2.git --recursive

# Install dependencies
cd ~/ros2_ws
rosdep install --from-paths src --ignore-src -r -y

# Build
colcon build
source install/setup.bash
```

Add to `~/.bashrc`:
```bash
echo "source ~/ros2_ws/install/setup.bash" >> ~/.bashrc
```

Test:
```bash
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
ros2 launch xarm_gazebo xarm6_beside_table_gazebo.launch.py add_gripper:=true
```

---

## 6. Copy Project from Ubuntu 24.04

From Windows PowerShell, copy your project between WSL distros:
```powershell
# Copy project files from Ubuntu 24 to Ubuntu 22
wsl -d Ubuntu-24.04 tar -czf /tmp/cais-spade-llm.tar.gz -C /home/jongh/projects cais-spade-llm
wsl -d Ubuntu-22.04 bash -c "mkdir -p ~/projects && tar -xzf /tmp/cais-spade-llm.tar.gz -C ~/projects"
```

Or from inside Ubuntu 22.04 (WSL distros share `/mnt/wsl`):
```bash
mkdir -p ~/projects
cp -r /mnt/wsl/instances/Ubuntu-24.04/home/jongh/projects/cais-spade-llm ~/projects/
```

---

## 7. Set Up Python Environment in Ubuntu 22.04

```bash
cd ~/projects/cais-spade-llm
sudo apt install -y python3-pip python3-venv
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

---

## 8. WSLg Display (for Gazebo/RViz)

WSLg is included in WSL2 — no extra setup needed. Verify:
```bash
echo $DISPLAY   # should show :0 or similar
```

If Gazebo windows are too large:
```bash
export QT_SCALE_FACTOR=0.7
```

---

## 9. Switching Between Distros

```powershell
# Use Ubuntu 22 (ROS2 Humble — UR5e + xArm6)
wsl -d Ubuntu-22.04

# Use Ubuntu 24 (ROS2 Jazzy — UR5e only, old setup)
wsl -d Ubuntu-24.04
```

---

## 10. Full ~/.bashrc for Ubuntu 22.04

```bash
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
export QT_SCALE_FACTOR=0.7   # optional: fix oversized windows
```
