# WSL Ubuntu 22.04 + ROS2 Humble Setup Guide

Install Ubuntu 22.04 alongside existing Ubuntu 24.04, then set up ROS2 Humble with UR5e and xArm6.

After completing this OS/ROS bootstrap, continue with:
`ros2/docs/ros2_setup_from_scratch.md`

Then use:
`ros2/docs/ros2_three_mode_control_guide.md`
to run the three control modes.

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

## 7. Set Up Python Environment (Poetry + Python 3.10)

The project uses **Poetry** for dependency management and should run on **Python 3.10**
to stay compatible with ROS2 Humble.

### 7a. Install Python 3.10

```bash
sudo apt install -y python3.10 python3.10-venv python3.10-dev
```

### 7b. Install required system packages

```bash
# 'python' binary needed by Poetry
sudo apt install -y python-is-python3

# lxml needed by ROS2 tools (spawn_entity.py, etc.)
sudo apt install -y python3-lxml
```

> **Why `python-is-python3`?** Poetry looks for a `python` binary (not `python3`).
> Without this package, `poetry install` fails with "No such file or directory: 'python'".

> **Why `python3-lxml`?** ROS2's `spawn_entity.py` and other tools use
> `#!/usr/bin/env python3`. When a Poetry venv is active, `python3` resolves to
> the venv's Python — which does not have `lxml`. This causes Gazebo spawning to fail.
> Installing `python3-lxml` system-wide fixes this for all ROS2 tools.

### 7c. Install Poetry

```bash
curl -sSL https://install.python-poetry.org | python3.10
```

Add to `~/.bashrc`:
```bash
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

### 7d. Configure Poetry and install dependencies

```bash
cd ~/projects/cais-spade-llm

# Store venv inside the project (creates .venv/)
poetry config virtualenvs.in-project true

# Use Python 3.10 for this project
poetry env use /usr/bin/python3.10

# Install all dependencies
poetry install
```

### 7e. Verify

```bash
# Should print 3.10.x
.venv/bin/python --version

# Should run the main entry point
source .venv/bin/activate
python cais_spade_llm/spade_main.py
```

---

## 8. CRITICAL: Two-Terminal Rule

> **Never run ROS2/Gazebo commands with the Poetry venv activated.**

The Poetry venv does not contain ROS2 tools or `lxml`. When the venv is active,
`python3` resolves to the venv Python and ROS2 internal tools (e.g. `spawn_entity.py`) fail.

| Terminal | Purpose | Environment |
|---|---|---|
| **Terminal A** (Python/SPADE) | Run `spade_main.py`, tests, Poetry commands | Poetry venv **activated** (`source .venv/bin/activate`) |
| **Terminal B** (ROS2/Gazebo) | `ros2 launch`, `ros2 topic`, Gazebo | Poetry venv **NOT activated** — only `source /opt/ros/humble/setup.bash` |

**How to check:** Run `which python3` — if it points to `.venv/bin/python3`, the venv is active.
Deactivate with `deactivate` before running any ROS2 commands.

---

## 9. WSLg Display (for Gazebo/RViz)

WSLg is included in WSL2 — no extra setup needed. Verify:
```bash
echo $DISPLAY   # should show :0 or similar
```

Gazebo can be slow to start in WSL (30–60 seconds is normal). Wait for the window to appear before trying to spawn robots.

> **Gazebo Troubleshooting:** If Gazebo exits immediately with `[gzclient] process has died`,
> this is usually because the window was closed manually. Use `gui_required:=false` in launch
> args so closing the Gazebo GUI doesn't kill the entire simulation.

---

## 10. Switching Between Distros

```powershell
# Use Ubuntu 22 (ROS2 Humble — UR5e + xArm6)
wsl -d Ubuntu-22.04

# Use Ubuntu 24 (ROS2 Jazzy — UR5e only, old setup)
wsl -d Ubuntu-24.04
```

---

## 11. Full ~/.bashrc for Ubuntu 22.04

```bash
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
export PATH="$HOME/.local/bin:$PATH"
```

> Do **not** add `source .venv/bin/activate` to `.bashrc` — this would break ROS2 tools
> in every new terminal. Activate the Poetry venv manually only when needed.
