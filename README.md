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
| digital twin | hardware MoveIt/RViz with Gazebo as visual mirror and physical gear poses | ROS2 stack, `~/ros2_ws`, robot networks, RealSense, Roboflow, hand-eye calibration |
| physical product flow | real robots plus validated SG/MG world poses | lab networking, RealSense, Roboflow key, accepted hand-eye calibration |

Physical gear perception supports exact `small_gear → SG` and
`medium_gear → MG` mappings. `gear_large` is fully available in Gazebo, but
`LG` perception remains blocked until the Roboflow model contains the exact
`large_gear` class.

## Minimum First Run Without ROS2

If you only want to open the UI and confirm the Python path, complete steps 1-5
below and stop before installing ROS2. The `dry_run` path does not need ROS2,
Gazebo, RViz, xArm6, UR5e, RG2, or robot networking.

After step 4, this is enough for the first local UI check:

```bash
make run
```

Open `http://localhost:8080`, then use `dry_run` before starting ROS2 or
hardware.

## New PC Install: Digital Twin Environment

Use Ubuntu 22.04, either native or WSL2. ROS2 Humble is built for Ubuntu 22.04.
On WSL2, WSLg is enough for Gazebo and RViz windows on modern Windows installs.

### 1. Install base system tools

Open Ubuntu 22.04 and run:

```bash
sudo apt update
sudo apt install -y \
  build-essential \
  cmake \
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
poetry --version
```

Use Poetry 2.x for this repo. Older Poetry 1.x installs can fail on the
dependency group metadata in `pyproject.toml`.

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

The optional values in `.env.example` are safe defaults for a first `dry_run`
install.

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

Equivalent Make target:

```bash
make install
```

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
No robot network connection is required for this Python-only check.

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
  ros-humble-realsense2-camera \
  ros-humble-realsense2-description \
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

`~/ros2_ws` is generated ROS2 build/runtime infrastructure, not a second copy of
this application. Gazebo, MoveIt, RViz, hardware control, and the digital twin
need its installed package index. Python-only planning and `dry_run` operation do
not require it.

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
- registers this repo's `ros2/cais_lab_robotics` package at
  `~/ros2_ws/src/cais_lab_robotics` with a source symlink
- copies the patched IFRA `gazebo_link_attacher.cpp`
- runs `colcon build --packages-skip d435i_xarm_setup`
- installs the `cais_lab_robotics` launch, world, model, config, RViz, sensor, and
  script assets under `~/ros2_ws/install/cais_lab_robotics`

`xarm_gazebo` remains an upstream dependency. It is not the container for CAIS
project files.

The IFRA LinkAttacher packages provide:

- `linkattacher_msgs`
- `ros2_linkattacher`
- `/ATTACHLINK`
- `/DETACHLINK`

Verify after bootstrap:

```bash
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
ros2 pkg prefix cais_lab_robotics
ros2 pkg prefix xarm_gazebo
ros2 pkg prefix xarm_description
ros2 pkg prefix onrobot_description
ros2 pkg prefix linkattacher_msgs
ros2 pkg prefix ros2_linkattacher
```

## Tracked Repo Files vs ROS2 Workspace Files

This repo is the source of truth for custom ROS2 files:

```text
ros2/cais_lab_robotics/launch/
ros2/cais_lab_robotics/config/
ros2/cais_lab_robotics/models/
ros2/cais_lab_robotics/cad_models/
ros2/cais_lab_robotics/rviz/
ros2/cais_lab_robotics/worlds/
ros2/cais_lab_robotics/scripts/
ros2/cais_lab_robotics/sensor/
ros2/third_party/IFRA_LinkAttacher/
```

The runtime ROS2 workspace is outside this repo:

```text
~/ros2_ws/
├── src/
│   ├── cais_lab_robotics -> this repo's ros2/cais_lab_robotics/
│   ├── xarm_ros2/
│   ├── OnRobot_ROS2_Description/
│   └── IFRA_LinkAttacher/
├── build/                     # generated colcon build state
├── log/                       # generated colcon logs
└── install/                   # generated ROS2 package index and installed assets
```

Do not edit `~/ros2_ws/build` or `~/ros2_ws/install`; both are regenerated by
`colcon`. The `~/ros2_ws/src/cais_lab_robotics` entry points back to this
repository, so edit the clearer repo path `ros2/cais_lab_robotics/...`. Vendor
source changes under the other `~/ros2_ws/src` directories are local to that PC
and are not tracked by this repository.

The safe workflow is:

1. Edit the source file in this repo under `ros2/cais_lab_robotics/...`.
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
ros2 launch cais_lab_robotics dual_moveit_gazebo.launch.py
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

For `dry_run` or Simulation, use a project terminal:

```bash
cd ~/projects/cais-spade-llm
make run
```

For physical `ur5e` Function Execution or the `ur5e only` passive Digital Twin,
start the UI on the UR5e hardware domain before ROS initializes:

```bash
cd ~/projects/cais-spade-llm
make run-physical-ur5e
```

For VS Code debugging, both the tracked `CAIS UI: Physical ur5e` configuration and
`Python: Current File (F5 Default)` set `ROS_DOMAIN_ID=42`. Opening `ui_main.py` and
pressing F5 therefore uses the UR5e hardware domain before ROS initializes.

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
| UR5e arm | [ros2/cais_lab_robotics/scripts/ur5e_rtde_trajectory_server.py](ros2/cais_lab_robotics/scripts/ur5e_rtde_trajectory_server.py) |
| UR5e RG2 | [ros2/cais_lab_robotics/scripts/ur5e_rg2_rtde_gripper.py](ros2/cais_lab_robotics/scripts/ur5e_rg2_rtde_gripper.py) |
| combined hardware MoveIt/RViz | `dual_robots_hardware_moveit.launch.py` |
| Gazebo mirror | passive Gazebo launch files installed from `cais_lab_robotics` |
| sync/replay helper | [ros2/cais_lab_robotics/scripts/digital_twin_sync.py](ros2/cais_lab_robotics/scripts/digital_twin_sync.py) |
| paired RViz markers | [ros2/cais_lab_robotics/scripts/dual_drag_markers.py](ros2/cais_lab_robotics/scripts/dual_drag_markers.py) |

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
ros2/cais_lab_robotics/launch/dual_robots_hardware_moveit.launch.py
```

After bootstrap, ROS2 launches it from:

```text
~/ros2_ws/src/cais_lab_robotics/launch/dual_robots_hardware_moveit.launch.py
```

## Digital Twin Run Checklist

Before starting hardware digital twin:

1. Build the Python venv with `poetry install`.
2. Build the ROS2 workspace with `make bootstrap-gazebo`.
3. Put the xArm6 and UR5e on the same network as the PC.
4. Confirm the robot IPs in the Control page.
5. For `ur5e only`, start the UI with `make run-physical-ur5e`.
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
make run-physical-ur5e
make headless
make bootstrap-gazebo
make check
```

Equivalent direct commands:

```bash
poetry install
poetry run pip install -r requirements-ui.txt
poetry run python -m cais_spade_llm
poetry run python -m cais_spade_llm.ui_main
poetry run python -m cais_spade_llm.ui_main --headless
bash scripts/bootstrap_gazebo_workspace.sh
poetry check
poetry run python -m compileall -q cais_spade_llm ros2
poetry run python -m cais_spade_llm.ui_main --help
```

## Case 3 Recovery Dry-Run Prerequisites

The Case 3 recovery debugger and its focused pytest suite are in
`test/test_case3_recovery_dryrun.py`. Run them from the repository root after
installing the Poetry environment.

The test depends on the immutable verified bundle at this exact path:

```text
cais_spade_llm/user_verified_plan/bundles/case3_llm_recovery/
```

The directory must contain `bundle_manifest.json` and every artifact referenced
by the manifest, including the tools catalogue, requirements, plan, safety
logic, and validation artifacts. The tracked runtime context at
`test/fixtures/case3_recovery/runtime_context.json` intentionally uses the fixed
`case3_llm_recovery` bundle name. Do not substitute or rename another bundle.

Most generated verified bundles remain machine-local and Git-ignored. The
`.gitignore` file makes a narrow exception for `case3_llm_recovery` because this
bundle is a prerequisite for the tracked test. After generating and verifying
the bundle on the school laptop, commit the entire directory:

```bash
git add \
  cais_spade_llm/user_verified_plan/bundles/case3_llm_recovery \
  .gitignore README.md
git status --short
```

Do not commit `.env`, API keys, credentials, logs, or other generated bundles.

The module creates its shared OpenAI client during import, so
`OPENAI_API_KEY` must be non-empty even while pytest uses mocked LLM response
fixtures. CI can use a non-secret placeholder for this mocked suite:

```bash
OPENAI_API_KEY=ci-placeholder poetry run pytest -q test/test_case3_recovery_dryrun.py
```

Direct script execution is different: it performs live LLM requests and needs
a real key in the ignored `.env` file or process environment:

```bash
poetry run python test/test_case3_recovery_dryrun.py --mode outline
poetry run python test/test_case3_recovery_dryrun.py --mode primitive
poetry run python test/test_case3_recovery_dryrun.py --mode safety
poetry run python test/test_case3_recovery_dryrun.py --mode full
```

If `bundle_manifest.json` is missing after a clone, copy the complete
`case3_llm_recovery` directory from the machine that generated it or retrieve it
as a CI artifact. Creating only an empty manifest is not sufficient.

## Troubleshooting

| Problem | Fix |
| --- | --- |
| `ROS2 workspace is not built yet` | Run `make bootstrap-gazebo` from the repo root. |
| ROS2 cannot find `cais_lab_robotics` or its launch/config/RViz files | Run `make bootstrap-gazebo`, source `~/ros2_ws/install/setup.bash`, then check `ros2 pkg prefix cais_lab_robotics`. |
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
  build-essential \
  cmake \
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
  ros-humble-realsense2-camera \
  ros-humble-realsense2-description \
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
5. a rigid RealSense wrist mount and an accepted ChArUco hand-eye calibration

## Wrist RealSense and Roboflow Setup

Rotate any Roboflow key that has appeared in chat or logs. Put the replacement
only in the ignored `.env` file, never in source:

```text
ROBOFLOW_API_KEY=...
ROBOFLOW_API_URL=https://detect.roboflow.com
ROBOFLOW_MODEL_ID=hrc-assembly-gph6m/5
```

The runtime sends the synchronized RealSense color image directly to the
published `hrc-assembly-gph6m/5` model. No Roboflow Workflow is required.

Provision Linux once. This installs the ROS RealSense packages and adds the
operator to the `video` group; it does not store a sudo password or create a
passwordless sudo rule:

```bash
make setup-perception-host
```

Under WSL, first install usbipd-win. In Administrator PowerShell, bind each
RealSense once with `usbipd list` and `usbipd bind --busid <BUSID>`. After that,
the normal-user **Perception** page can attach a previously bound camera to WSL.
Attachment must be repeated after unplugging a camera or restarting WSL.

Open **Perception** at `/perception`. Assign connected serial numbers to the
exact `ur5e`, `xarm6`, and `stationary` roles; assignments are stored only in
`~/.config/cais-spade-llm/perception_cameras.yaml`. The page starts/stops each
camera, provides embedded color/depth views, opens optional `rqt_image_view`,
captures reviewed ChArUco samples, solves/activates/rolls back calibration, and
runs role-specific Test Detection without robot motion.

For `ur5e` and `xarm6`, save 25 varied reviewed poses with **Save Pose +
Capture**. **Preview Automatic Calibration** first plans every reviewed pose
without motion. Only after that preview succeeds can the operator explicitly
confirm **Run Automatic Calibration**, the only calibration control that moves
a robot. Replay rechecks each plan before execution and provides Pause, Resume,
Skip, and Abort. The stationary camera uses a measured fixed ChArUco `world`
pose. Its live view remains available until that pose is configured, but
world-pose comparison stays blocked.

Calibration activates only when median reprojection error is at most 1 px,
fixed-board translation RMS is at most 5 mm, and rotation RMS is at most 1
degree. The UR5e **Calibrate Table Plane** action collects 10 `/detect_all`
results and rejects table-plane MAD above 2 mm or SG/MG median disagreement
above 5 mm. Restart that perception instance and passive Gazebo after activating
a calibration.

UR5e retains canonical `/detect_all` and `/detect_part`. The diagnostic services
are `/perception/ur5e/detect_all`, `/perception/xarm6/detect_all`, and
`/perception/stationary/detect_all`, with corresponding `/detect_part` services.
xArm6 and stationary never replace or average the executable UR5e world pose.

| Mode | Pose authority | Gazebo gear behavior |
| --- | --- | --- |
| Gazebo only | `gazebo_gt` `/detect_all` | predefined world models remain authoritative |
| Hardware only | RealSense + Roboflow `/detect_all` in world-frame metres | no Gazebo gear synchronization process |
| digital twin | the same physical `/detect_all` payload | the atomic snapshot crosses ROS domains; Gazebo spawns/updates/freezes/attaches the matching model |
