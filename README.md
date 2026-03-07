# CAIS-SPADE-LLM

CAIS-SPADE-LLM is a multi-agent manufacturing system with:

- LLM-based product planning
- LTLf/DFA-based safety validation and monitoring
- dual-robot execution with UR5e and xArm6
- a NiceGUI operator console for products, plans, safety, and control

This README is the deployment contract for a new engineer or an LLM. It is written to answer one question clearly:

`What can I run immediately after cloning, and what still requires extra system setup?`

## Reality Check

This repository is deployable on a new PC, but not all modes are equally turnkey.

| Mode | After host tools install | Extra setup required | Notes |
| --- | --- | --- | --- |
| `dry_run` | Yes | No | Best first-run path |
| `simulation` | No | ROS 2 Humble + Gazebo workspace | A bootstrap script is included |
| `physical` | No | Robot drivers, networking, calibration, perception implementation | Physical perception is not fully implemented yet |

Important limits:

- Core dependencies install from Poetry, but the UI still uses a checked-in [`requirements-ui.txt`](requirements-ui.txt) because the current `spade` and `nicegui` dependency constraints do not solve cleanly together in Poetry. If you want the web UI or the default `python -m cais_spade_llm` entry point, you need this file. If you only run `--headless`, you can skip it.
- On a brand-new Ubuntu/WSL machine, the `dry_run` path still needs a few base OS packages first: `git`, `curl`, `python3-pip`, and `python3.10-venv`.
- Gazebo simulation still depends on an external ROS 2 workspace under `~/ros2_ws`.
- Physical perception is not turnkey yet because these files are still stubs:
  - [`cais_spade_llm/resources/sensor/physical/detect_all_service.py`](cais_spade_llm/resources/sensor/physical/detect_all_service.py)
  - [`cais_spade_llm/resources/sensor/physical/detect_part_service.py`](cais_spade_llm/resources/sensor/physical/detect_part_service.py)

## Guidance for LLM-Assisted Setup

If an engineer or an LLM turns this README into machine setup instructions, use this format:

- Give the normal install and run steps first.
- At the very end, include one final section titled `Sudo Commands Summary`.
- That final section should repeat every command that needs `sudo` for the requested setup mode on that machine.
- Do not imply that a fresh Ubuntu/WSL machine is `clone only` if host packages are still missing.
- If Gazebo/MoveIt still will not launch, end with a short `What Is Still Missing` section that names the exact missing file, package, or workspace step and the next command the user should run.
- State whether [`requirements-ui.txt`](requirements-ui.txt) is required:
  - required for the NiceGUI web UI and the default entry point
  - optional only for headless-only usage

## Fastest Successful Bring-Up

Use this path on any new PC to confirm the repo works before touching Gazebo or hardware.

### 1. Install host tools

For a fresh Ubuntu 22.04 / WSL2 machine, this is the shortest reproducible path:

```bash
sudo apt update
sudo apt install -y git curl python3-pip python3.10-venv

# Optional, but useful for saved DOT/graph rendering
sudo apt install -y graphviz

curl -sSL https://install.python-poetry.org | python3
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

Required for the `dry_run` path:

- Python `3.10`
- Poetry
- Git

Optional but useful:

- Graphviz
- MONA, if you want real DFA graphs in the Safety page

### 2. Clone and configure

```bash
git clone <repo-url>
cd cais-spade-llm
cp .env.example .env
```

Edit `.env` and set:

```bash
OPENAI_API_KEY=...
```

### 3. Install Python dependencies

```bash
poetry config virtualenvs.in-project true
poetry install
poetry run pip install -r requirements-ui.txt
```

`requirements-ui.txt` installs `nicegui`, which is needed for the web UI. If you plan to run only:

```bash
poetry run python -m cais_spade_llm.ui_main --headless
```

you can skip the `poetry run pip install -r requirements-ui.txt` step.

### 4. Start the UI

```bash
poetry run python -m cais_spade_llm
```

If you launch the app from VS Code, use the checked-in debug configuration in [`.vscode/launch.json`](.vscode/launch.json). It now runs Python through [`scripts/ros_python.sh`](scripts/ros_python.sh), which sources `/opt/ros/humble/setup.bash` and `~/ros2_ws/install/setup.bash` first so simulation controllers can import `rclpy`.

The UI runs on `http://localhost:8080`.

### 5. First-run recommendation

Start in:

- execution mode: `dry_run`
- robot environment: `gazebo`

This avoids ROS 2/hardware setup while validating that:

- Python environment is correct
- LLM access works
- the UI works
- the shipped sample product and safety files load

## UI Pages

Current routes:

| Page | Path |
| --- | --- |
| Dashboard | `/` |
| Control | `/control` |
| Plans | `/plans` |
| Safety | `/safety` |
| Products | `/products` |
| Resources | `/resources` |

## What Ships as the Default Sample

The repo now includes a coherent default sample for the tracked product:

- Product manifest: [`cais_spade_llm/initialization/products/assembly_board-v1.json`](cais_spade_llm/initialization/products/assembly_board-v1.json)
- Product requirements: [`cais_spade_llm/specification/products/requirements/assembly_board-v1.txt`](cais_spade_llm/specification/products/requirements/assembly_board-v1.txt)
- Product geometry: [`cais_spade_llm/specification/products/geometry/assembly_board-v1.json`](cais_spade_llm/specification/products/geometry/assembly_board-v1.json)
- Default safety file: [`cais_spade_llm/specification/safety/safety_requirements.txt`](cais_spade_llm/specification/safety/safety_requirements.txt)

That means a fresh clone no longer depends on your local untracked safety/product text files just to start the sample flow.

## Commands for a New Engineer

Common commands are also wrapped in the included [`Makefile`](Makefile):

```bash
make install
make run
make headless
```

## Gazebo Simulation Setup

`simulation` mode requires ROS 2 Humble and Gazebo outside the Poetry environment.

### System packages required for simulation

If `/opt/ros/humble/setup.bash` is not present on the machine yet, install ROS 2 Humble first.

On Ubuntu 22.04 / WSL2:

```bash
sudo apt install -y locales
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8

sudo apt install -y software-properties-common curl

# Only add the ROS apt repo manually if the machine does not already have
# /etc/apt/sources.list.d/ros2.sources from ros-apt-source.
if [ ! -e /etc/apt/sources.list.d/ros2.sources ]; then
  sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
    -o /usr/share/keyrings/ros-archive-keyring.gpg
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
    http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" \
    | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
fi

# If apt reports a conflicting Signed-By error, disable the duplicate manual file
# and keep the existing ros2.sources entry.
if [ -e /etc/apt/sources.list.d/ros2.sources ] && [ -e /etc/apt/sources.list.d/ros2.list ]; then
  sudo mv /etc/apt/sources.list.d/ros2.list /etc/apt/sources.list.d/ros2.list.disabled
fi

sudo apt update
sudo apt install -y ros-humble-desktop python3-colcon-common-extensions python3-rosdep
sudo rosdep init
rosdep update
```

For this repository's dual-robot Gazebo setup, also install:

```bash
sudo apt install -y \
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

### One-time Gazebo workspace bootstrap

This repo now includes:

- [`scripts/bootstrap_gazebo_workspace.sh`](scripts/bootstrap_gazebo_workspace.sh)

It will:

- create or reuse `~/ros2_ws`
- clone `xarm_ros2`
- clone `OnRobot_ROS2_Description`
- clone `IFRA_LinkAttacher`
- copy this repo's custom world, launch, config, and RViz files into the ROS 2 workspace
- copy this repo's patched IFRA `gazebo_link_attacher.cpp` into the workspace before build
- copy the custom config and RViz assets into the installed `xarm_gazebo` package share that the launch files read at runtime
- run `colcon build --packages-skip d435i_xarm_setup`

The IFRA LinkAttacher build is required for this repository's Gazebo grasp/attach flow:

- workspace package `linkattacher_msgs`
- workspace package `ros2_linkattacher`
- Gazebo services `/ATTACHLINK` and `/DETACHLINK`

If those are missing, startup may succeed but simulated grasp attach/detach will be disabled.

Why `d435i_xarm_setup` is skipped:

- it is an optional xArm camera / hand-eye example package
- it depends on `object_recognition_msgs`
- it is not required for this repository's dual-robot Gazebo + MoveIt launch path

If you later want that optional package too:

```bash
sudo apt install -y ros-humble-object-recognition-msgs
```

Run:

```bash
make bootstrap-gazebo
```

or:

```bash
bash scripts/bootstrap_gazebo_workspace.sh
```

If this step has not been completed yet, the UI `Dual Robots` launch will not work because it requires:

```bash
~/ros2_ws/install/setup.bash
```

If the UI reports:

```text
ROS2 workspace is not built yet: missing /home/<user>/ros2_ws/install/setup.bash.
```

the next step is:

```bash
cd /path/to/cais-spade-llm
make bootstrap-gazebo
```

If `make bootstrap-gazebo` fails, finish the missing ROS 2 / Gazebo apt packages from `System packages required for simulation`, then rerun it.

After a successful bootstrap, verify the IFRA attacher services once:

```bash
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
ros2 service list | grep -E '/ATTACHLINK|/DETACHLINK'
```

If they are missing, restart the Gazebo launch once so it reloads the new plugin build.

### Start Gazebo

To match the UI Control page's `Dual Robots` button, launch:

```bash
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
ros2 launch xarm_gazebo dual_moveit_gazebo.launch.py
```

If you want the lower-level world bring-up without the combined MoveIt launch, use:

```bash
ros2 launch xarm_gazebo xarm6_ur5e_gazebo.launch.py
```

Then, in a separate terminal:

```bash
cd /path/to/cais-spade-llm
poetry run python -m cais_spade_llm
```

You can also manage Gazebo and hardware launch from the UI Control page once the ROS 2 workspace is prepared.

Full ROS 2 notes remain in:

- [`ros2/ROS2_SETUP_README.md`](ros2/ROS2_SETUP_README.md)

## Physical Hardware Setup

`physical` mode is not yet fully one-command deployable from this repository alone.

What is already here:

- robot-agent wiring
- hardware launch integration in the UI
- configurable robot IPs
- execution-mode switching

What still requires machine-specific integration:

- physical robot drivers and MoveIt stacks
- hardware network configuration
- calibration
- real perception implementation for:
  - [`cais_spade_llm/resources/sensor/physical/detect_all_service.py`](cais_spade_llm/resources/sensor/physical/detect_all_service.py)
  - [`cais_spade_llm/resources/sensor/physical/detect_part_service.py`](cais_spade_llm/resources/sensor/physical/detect_part_service.py)

So the physical path is deployable as a framework, but not yet turnkey as a fully finished hardware product.

## Safety DFA Rendering

Safety preview works without MONA, but real DFA graphs require the external `mona` executable.

Without MONA:

- safety parsing still works
- DOT files may still be saved
- PNG DFA graphs will be unavailable

## Environment Variables

The main environment file is:

- [`.env.example`](.env.example)

Most users only need:

```bash
OPENAI_API_KEY=...
```

Optional overrides already supported by the code include:

- `EXECUTION_MODE`
- `ROBOT_ENV`
- `PERCEPTION_BACKEND`
- `PERCEPTION_NODE_NAME`
- `CAIS_XMPP_HOST`
- `CAIS_XMPP_DB_IN_MEMORY`
- `ENABLE_ROBOT_AGENT_PREWARM`

## Entry Points

Primary entry point:

```bash
poetry run python -m cais_spade_llm
```

Alternative entry points:

```bash
poetry run python -m cais_spade_llm.ui_main
poetry run python -m cais_spade_llm.ui_main --headless
```

## Sudo Commands Summary

This section is intentionally redundant. It lists only the commands that require `sudo` on a new Ubuntu 22.04 / WSL2 desktop.

If you only want the `dry_run` path:

```bash
sudo apt update
sudo apt install -y git curl python3-pip python3.10-venv
sudo apt install -y graphviz
```

If you also need ROS 2 Humble and Gazebo simulation:

```bash
sudo apt install -y locales
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8

sudo apt install -y software-properties-common curl

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
sudo apt install -y ros-humble-desktop python3-colcon-common-extensions python3-rosdep
sudo rosdep init

sudo apt install -y \
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

## If You Want True "Clone And Deploy Anywhere"

This README gets the repo much closer, but true one-command deployment on a brand-new PC would still need follow-up work:

1. a scripted system dependency installer for Ubuntu
2. a scripted ROS 2 dependency installer, not just workspace bootstrap
3. a finished physical perception pipeline
4. a clean dependency strategy that removes the current `spade`/`nicegui` installer split
5. optionally a container/devcontainer for the non-ROS `dry_run` path
6. optionally a dedicated installer or compose-style orchestration for ROS 2 + UI

That is the difference between:

- `well-documented and reproducible`
- and `fully turnkey on any machine`

This repo is now set up for the first one, and partially prepared for the second.
