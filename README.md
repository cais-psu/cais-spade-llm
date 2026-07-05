# CAIS-SPADE-LLM

CAIS-SPADE-LLM is a multi-agent manufacturing automation system. It combines:

- SPADE agents for product, robot, user, and central-controller coordination
- LLM-based process planning and recovery planning
- LTL/FSA safety validation and runtime monitoring
- xArm6 + UR5e dual-robot execution
- ROS2 Humble, Gazebo Classic, MoveIt2, RViz, RTDE, and RG2 support
- a NiceGUI operator UI for setup, control, plans, safety, and status

This README is the beginner install guide for a new Ubuntu 22.04 / WSL2 PC.
It is written for the full digital twin environment first, then notes what can
run without ROS2 or without physical robots.

For code orientation after installation, read [docs/codebase_map.md](docs/codebase_map.md).

## What Works At Each Level

| Level | What you can run | What you need |
| --- | --- | --- |
| `dry_run` | UI, agents, planning, safety files, sample product flow | Python 3.10, Poetry, OpenAI key |
| Gazebo + RViz | xArm6 + UR5e simulation with MoveIt/RViz | ROS2 Humble, Gazebo Classic, `~/ros2_ws`, bootstrap |
| digital twin | hardware MoveIt/RViz with Gazebo as visual mirror | ROS2 stack, `~/ros2_ws`, xArm6 network, UR5e RTDE network, RG2 bridge |
| physical product flow | real robots plus production perception | lab networking, calibration, physical perception implementation |

The physical perception files are still stubs:

- [cais_spade_llm/resources/sensor/physical/detect_all_service.py](cais_spade_llm/resources/sensor/physical/detect_all_service.py)
- [cais_spade_llm/resources/sensor/physical/detect_part_service.py](cais_spade_llm/resources/sensor/physical/detect_part_service.py)

## New PC Install: Digital Twin Environment

Use Ubuntu 22.04, either native or WSL2. ROS2 Humble is built for Ubuntu 22.04.
On WSL2, WSLg is enough for Gazebo and RViz windows on modern Windows installs.

### 1. Install base system tools

Open Ubuntu 22.04 and run:

```bash
sudo apt update
sudo apt install -y \
  git \
  curl \
  python3-pip \
  python3.10 \
  python3.10-venv \
  python3.10-dev \
  python-is-python3 \
  graphviz
```

Install Poetry:

```bash
curl -sSL https://install.python-poetry.org | python3.10
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

### 2. Clone this repo

```bash
mkdir -p ~/projects
cd ~/projects
git clone <repo-url> cais-spade-llm
cd cais-spade-llm
```

### 3. Create `.env`

```bash
cp .env.example .env
```

Edit `.env` and set:

```bash
OPENAI_API_KEY=...
```

The optional values in `.env.example` can stay commented for a first install.

### 4. Install Python dependencies

```bash
poetry config virtualenvs.in-project true
poetry env use /usr/bin/python3.10
poetry install
poetry run pip install -r requirements-ui.txt
```

`requirements-ui.txt` is required for the NiceGUI web UI and the normal
`poetry run python -m cais_spade_llm` entry point.

The UR5e RTDE Python dependency is installed through Poetry from `pyproject.toml`.

### 5. Verify the Python-only path

```bash
poetry check
poetry run python -m compileall -q cais_spade_llm ros2
poetry run python -m cais_spade_llm.ui_main --help
poetry run python -m cais_spade_llm
```

Open:

```text
http://localhost:8080
```

For a first UI run, use `dry_run` before starting ROS2 or hardware.

### 6. Install ROS2 Humble, Gazebo, MoveIt, and controllers

Install locale and ROS apt repository support:

```bash
sudo apt install -y locales software-properties-common curl
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
```

Add the ROS apt repository if this PC does not already have it:

```bash
if [ ! -e /etc/apt/sources.list.d/ros2.sources ]; then
  sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
    -o /usr/share/keyrings/ros-archive-keyring.gpg
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
    http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" \
    | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
fi
```

If apt reports a duplicate `Signed-By` conflict and both files exist, keep
`ros2.sources` and disable the manual `ros2.list` file:

```bash
if [ -e /etc/apt/sources.list.d/ros2.sources ] && [ -e /etc/apt/sources.list.d/ros2.list ]; then
  sudo mv /etc/apt/sources.list.d/ros2.list /etc/apt/sources.list.d/ros2.list.disabled
fi
```

Install ROS2 and this project's ROS dependencies:

```bash
sudo apt update
sudo apt install -y \
  ros-humble-desktop \
  python3-colcon-common-extensions \
  python3-rosdep \
  ros-humble-moveit \
  ros-humble-gazebo-ros-pkgs \
  ros-humble-gazebo-ros2-control \
  ros-humble-ur-description \
  ros-humble-ur-moveit-config \
  ros-humble-controller-manager \
  ros-humble-joint-state-broadcaster \
  ros-humble-joint-trajectory-controller \
  ros-humble-xacro \
  ros-humble-robot-state-publisher
```

Initialize rosdep once:

```bash
if [ ! -e /etc/ros/rosdep/sources.list.d/20-default.list ]; then
  sudo rosdep init
fi
rosdep update
```

### 7. Build the ROS2 workspace

From this repo:

```bash
cd ~/projects/cais-spade-llm
make bootstrap-gazebo
```

This creates and builds `~/ros2_ws`. The first build can take a while.

### 8. Source ROS2 in ROS terminals

For any terminal where you run raw `ros2`, `gazebo`, or RViz commands:

```bash
deactivate 2>/dev/null || true
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
```

Do not run raw ROS2/Gazebo commands from an activated Poetry venv. The UI is
started with Poetry, but raw ROS2 terminals should use the system ROS2
environment.

Optional convenience lines for `~/.bashrc`:

```bash
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
echo "source ~/ros2_ws/install/setup.bash" >> ~/.bashrc
```

Do not add `source .venv/bin/activate` to `~/.bashrc`.

## What `make bootstrap-gazebo` Does

`make bootstrap-gazebo` runs [scripts/bootstrap_gazebo_workspace.sh](scripts/bootstrap_gazebo_workspace.sh).

It does this:

- creates or reuses `~/ros2_ws`
- clones `xarm_ros2` into `~/ros2_ws/src/xarm_ros2`
- clones `OnRobot_ROS2_Description` for the RG2 meshes/URDF
- clones `IFRA_LinkAttacher` for Gazebo attach/detach services
- copies this repo's `ros2/cais_lab_gazebo/worlds/*.world` into the xArm Gazebo package
- copies this repo's `ros2/cais_lab_gazebo/launch/*.py` into the xArm Gazebo package
- copies this repo's `ros2/cais_lab_gazebo/config/*.yaml` into the xArm Gazebo package
- copies this repo's `ros2/cais_lab_gazebo/rviz/*.rviz` into the xArm Gazebo package
- copies the patched IFRA `gazebo_link_attacher.cpp`
- runs `colcon build --packages-skip d435i_xarm_setup`
- copies config and RViz assets into the installed `xarm_gazebo` share directory

The IFRA LinkAttacher packages provide:

- `linkattacher_msgs`
- `ros2_linkattacher`
- `/ATTACHLINK`
- `/DETACHLINK`

Verify after bootstrap:

```bash
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
ros2 pkg prefix xarm_gazebo
ros2 pkg prefix linkattacher_msgs
ros2 pkg prefix ros2_linkattacher
```

## Tracked Repo Files vs ROS2 Workspace Files

This repo is the source of truth for custom ROS2 files:

```text
ros2/cais_lab_gazebo/launch/
ros2/cais_lab_gazebo/config/
ros2/cais_lab_gazebo/rviz/
ros2/cais_lab_gazebo/worlds/
ros2/cais_lab_gazebo/scripts/
ros2/third_party/IFRA_LinkAttacher/
```

The runtime ROS2 workspace is outside this repo:

```text
~/ros2_ws/src/xarm_ros2/xarm_gazebo/
~/ros2_ws/src/OnRobot_ROS2_Description/
~/ros2_ws/src/IFRA_LinkAttacher/
~/ros2_ws/install/
```

Git in this project does not track files under `~/ros2_ws`. If you manually edit
`~/ros2_ws/src/...` or `~/ros2_ws/install/...`, those edits are local to that PC
and will not appear in this repository.

The safe workflow is:

1. Edit the source file in this repo under `ros2/cais_lab_gazebo/...`.
2. Run `make bootstrap-gazebo`.
3. Test through `~/ros2_ws`.
4. Commit only the repo file.

If a ROS2 launch change seems ignored, rerun `make bootstrap-gazebo` before
debugging the launch itself.

## Run The Gazebo + RViz Simulation

Use a ROS terminal:

```bash
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
ros2 launch xarm_gazebo dual_moveit_gazebo.launch.py
```

Expected result:

- Gazebo Classic opens with the dual-robot table world.
- RViz opens with MoveIt.
- xArm6 and UR5e are visible.
- Planning groups are available in the MotionPlanning panel.

Useful checks:

```bash
ros2 topic echo /joint_states --once
ros2 control list_controllers
ros2 service list | grep -E '/ATTACHLINK|/DETACHLINK'
```

WSL2 note: Gazebo and RViz can take 30-60 seconds to show useful output.

## Run The UI

Use a project terminal:

```bash
cd ~/projects/cais-spade-llm
poetry run python -m cais_spade_llm
```

Open:

```text
http://localhost:8080
```

The important UI page for ROS2 and digital twin work is:

```text
/control
```

The Control page can launch Gazebo, hardware processes, teleop, and digital twin
processes after the ROS2 workspace has been bootstrapped.

## Digital Twin Hardware Stack

The current dual-robot digital twin uses these pieces:

| Piece | Runtime path |
| --- | --- |
| xArm6 hardware | xArm hardware driver and xArm MoveIt realmove launch |
| UR5e arm | [ros2/cais_lab_gazebo/scripts/ur5e_rtde_trajectory_server.py](ros2/cais_lab_gazebo/scripts/ur5e_rtde_trajectory_server.py) |
| UR5e RG2 | [ros2/cais_lab_gazebo/scripts/ur5e_rg2_rtde_gripper.py](ros2/cais_lab_gazebo/scripts/ur5e_rg2_rtde_gripper.py) |
| combined hardware MoveIt/RViz | `dual_robots_hardware_moveit.launch.py` |
| Gazebo mirror | passive Gazebo launch files copied into `~/ros2_ws` |
| sync/replay helper | [ros2/cais_lab_gazebo/scripts/digital_twin_sync.py](ros2/cais_lab_gazebo/scripts/digital_twin_sync.py) |
| paired RViz markers | [ros2/cais_lab_gazebo/scripts/dual_drag_markers.py](ros2/cais_lab_gazebo/scripts/dual_drag_markers.py) |

The UR5e arm path is RTDE-based. Do not expect the old UR dashboard/external
control path to be the main runtime path for this project.

The UI starts the UR5e RTDE trajectory server with:

```text
/cais_ur5e_rtde_trajectory_controller/follow_joint_trajectory
```

The UI starts the UR5e RG2 bridge with:

```text
/ur5e_rg2_gripper_traj_controller/follow_joint_trajectory
```

The combined hardware MoveIt/RViz launch file is:

```text
ros2/cais_lab_gazebo/launch/dual_robots_hardware_moveit.launch.py
```

After bootstrap, ROS2 launches it from:

```text
~/ros2_ws/src/xarm_ros2/xarm_gazebo/launch/dual_robots_hardware_moveit.launch.py
```

## Digital Twin Run Checklist

Before starting hardware digital twin:

1. Build the Python venv with `poetry install`.
2. Build the ROS2 workspace with `make bootstrap-gazebo`.
3. Put the xArm6 and UR5e on the same network as the PC.
4. Confirm the robot IPs in the Control page.
5. Start the UI with `poetry run python -m cais_spade_llm`.
6. Use the Control page to start the relevant Gazebo, hardware, and digital twin processes.
7. Confirm RViz and Gazebo are open.
8. Confirm `/joint_states` is publishing.

Hardware IP defaults in the app are:

| Robot | Default IP |
| --- | --- |
| xArm6 | `192.168.1.240` |
| UR5e | `192.168.1.172` |

If your lab uses different addresses, change them in the Control page and apply
the hardware IPs before launching hardware processes.

## Terminal Rules

Use two terminal styles:

| Terminal | Use it for | Environment |
| --- | --- | --- |
| Project terminal | Poetry, UI, Python checks | `poetry run ...` or `.venv` |
| ROS terminal | `ros2 launch`, `ros2 topic`, RViz/Gazebo checks | no Poetry venv, source ROS2 setup files |

If ROS2 commands fail with Python import errors, check:

```bash
which python3
```

If it points into `.venv/bin/python3`, run:

```bash
deactivate
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
```

## Default Sample Product

The repo includes a default sample product:

- [cais_spade_llm/initialization/products/assembly_board-v1.json](cais_spade_llm/initialization/products/assembly_board-v1.json)
- [cais_spade_llm/specification/products/requirements/assembly_board-v1.txt](cais_spade_llm/specification/products/requirements/assembly_board-v1.txt)
- [cais_spade_llm/specification/products/geometry/assembly_board-v1.json](cais_spade_llm/specification/products/geometry/assembly_board-v1.json)
- [cais_spade_llm/specification/safety/safety_requirements.txt](cais_spade_llm/specification/safety/safety_requirements.txt)

That sample is enough to start the UI and exercise planning/safety flows after
Python dependencies are installed.

## Useful Commands

```bash
make install
make run
make headless
make bootstrap-gazebo
```

Equivalent direct commands:

```bash
poetry run python -m cais_spade_llm
poetry run python -m cais_spade_llm.ui_main
poetry run python -m cais_spade_llm.ui_main --headless
```

## Troubleshooting

| Problem | Fix |
| --- | --- |
| `ROS2 workspace is not built yet` | Run `make bootstrap-gazebo` from the repo root. |
| ROS2 launch cannot find custom launch/config/RViz files | Run `make bootstrap-gazebo`; the files in `~/ros2_ws` are copied from this repo. |
| Gazebo/RViz does not appear quickly on WSL2 | Wait 30-60 seconds. First launch is slow. |
| ROS2 Python import error mentions `.venv` | Deactivate the Poetry venv before raw ROS2 commands. |
| `/ATTACHLINK` or `/DETACHLINK` missing | Rerun `make bootstrap-gazebo`, then restart Gazebo. |
| `/joint_states` missing | Wait for launch startup, then check controllers and hardware process status. |
| UR5e RTDE process cannot connect | Check UR5e IP, network route, and robot-side RTDE availability. |
| RG2 bridge cannot connect | Check UR5e IP and the RG2 bridge status in the Control page. |

## More ROS2 Notes

Older detailed ROS2 notes remain under:

- [ros2/ROS2_SETUP_README.md](ros2/ROS2_SETUP_README.md)
- [ros2/docs/wsl_ubuntu22_ros2_humble_setup.md](ros2/docs/wsl_ubuntu22_ros2_humble_setup.md)
- [ros2/docs/ros2_operation_guide_dual_robots.md](ros2/docs/ros2_operation_guide_dual_robots.md)

Use this README first for the current beginner install path. Some older ROS2
notes are lower-level references and may describe narrower simulation modes.

## Sudo Commands Summary

These are the commands above that require `sudo` on a new Ubuntu 22.04 / WSL2 PC.

```bash
sudo apt update
sudo apt install -y \
  git \
  curl \
  python3-pip \
  python3.10 \
  python3.10-venv \
  python3.10-dev \
  python-is-python3 \
  graphviz

sudo apt install -y locales software-properties-common curl
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8

if [ ! -e /etc/apt/sources.list.d/ros2.sources ]; then
  sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
    -o /usr/share/keyrings/ros-archive-keyring.gpg
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
    http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" \
    | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
fi

if [ -e /etc/apt/sources.list.d/ros2.sources ] && [ -e /etc/apt/sources.list.d/ros2.list ]; then
  sudo mv /etc/apt/sources.list.d/ros2.list /etc/apt/sources.list.d/ros2.list.disabled
fi

sudo apt update
sudo apt install -y \
  ros-humble-desktop \
  python3-colcon-common-extensions \
  python3-rosdep \
  ros-humble-moveit \
  ros-humble-gazebo-ros-pkgs \
  ros-humble-gazebo-ros2-control \
  ros-humble-ur-description \
  ros-humble-ur-moveit-config \
  ros-humble-controller-manager \
  ros-humble-joint-state-broadcaster \
  ros-humble-joint-trajectory-controller \
  ros-humble-xacro \
  ros-humble-robot-state-publisher

if [ ! -e /etc/ros/rosdep/sources.list.d/20-default.list ]; then
  sudo rosdep init
fi
```

## What Is Still Machine-Specific

The install is reproducible, but a real hardware digital twin still depends on:

1. xArm6 and UR5e being reachable from the PC network
2. correct robot IPs in the Control page
3. robot-side safety and calibration setup
4. RG2 availability on the UR5e setup
5. physical perception implementation for production use
