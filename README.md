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

| Mode | Clone + Poetry only | Extra setup required | Notes |
| --- | --- | --- | --- |
| `dry_run` | Yes | No | Best first-run path |
| `simulation` | No | ROS 2 Humble + Gazebo workspace | A bootstrap script is included |
| `physical` | No | Robot drivers, networking, calibration, perception implementation | Physical perception is not fully implemented yet |

Important limits:

- Core dependencies install from Poetry, but the UI still uses a checked-in [`requirements-ui.txt`](requirements-ui.txt) because the current `spade` and `nicegui` dependency constraints do not solve cleanly together in Poetry.
- Gazebo simulation still depends on an external ROS 2 workspace under `~/ros2_ws`.
- Physical perception is not turnkey yet because these files are still stubs:
  - [`cais_spade_llm/resources/sensor/physical/detect_all_service.py`](cais_spade_llm/resources/sensor/physical/detect_all_service.py)
  - [`cais_spade_llm/resources/sensor/physical/detect_part_service.py`](cais_spade_llm/resources/sensor/physical/detect_part_service.py)

## Fastest Successful Bring-Up

Use this path on any new PC to confirm the repo works before touching Gazebo or hardware.

### 1. Install host tools

Required:

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
poetry install
poetry run pip install -r requirements-ui.txt
```

### 4. Start the UI

```bash
poetry run python -m cais_spade_llm
```

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

### One-time Gazebo workspace bootstrap

This repo now includes:

- [`scripts/bootstrap_gazebo_workspace.sh`](scripts/bootstrap_gazebo_workspace.sh)

It will:

- create or reuse `~/ros2_ws`
- clone `xarm_ros2`
- clone `OnRobot_ROS2_Description`
- copy this repo's custom world and launch file into the ROS 2 workspace
- run `colcon build`

Run:

```bash
make bootstrap-gazebo
```

or:

```bash
bash scripts/bootstrap_gazebo_workspace.sh
```

### Start Gazebo

```bash
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
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
