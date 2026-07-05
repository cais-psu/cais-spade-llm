# CLAUDE.md

## Project overview

CAIS-SPADE-LLM is a multi-agent manufacturing automation system. It uses SPADE
agents for coordination, LLM planning, safety validation, ROS2/Gazebo/MoveIt for
robot execution, UR5e + xArm6 dual robots, and a NiceGUI operator UI.

## Tech stack

- Python 3.10 (`>=3.10,<3.11`, required by ROS2 Humble)
- Poetry
- SPADE 4.1.2
- LangChain + OpenAI API
- NiceGUI
- ROS2 Humble, MoveIt2, Gazebo Classic 11
- LTL/FSA safety validation

## Project structure

```text
cais_spade_llm/          # Main Python package
  agents/                # SPADE agents
  resources/             # Robot controllers, sensors, grippers, primitives
  ui/                    # NiceGUI pages, components, and SystemBridge
ros2/                    # ROS2 launch files, scripts, configs, RViz assets
initialization/          # JSON configs for products, robots, and agents
specification/           # Product requirements, safety rules, geometry
monitor/                 # Runtime outputs: logs, state, plans, history
```

## Run commands

```bash
poetry install
poetry run python -m cais_spade_llm
poetry run python -m cais_spade_llm.ui_main --headless
```

For ROS2/Gazebo work, bootstrap the workspace before checking installed launch
behavior:

```bash
make bootstrap-gazebo
```

## Verification

```bash
poetry check
poetry run python -m compileall -q cais_spade_llm ros2
poetry run python -m cais_spade_llm.ui_main --help
```

Create focused or temporary tests in `test/` when a feature needs them. Run the
tests you create or restore for the touched behavior. For ROS2 launch, script, or
RViz changes, run `make bootstrap-gazebo` before checking the installed
workspace.

## Coding rules

Read `AGENTS.md` before changing code. It is the source for project coding rules:
keep changes scoped, preserve project terms exactly as written, avoid generic
canonical rewrites, keep `SystemBridge` as the UI-to-runtime surface unless the
user asks otherwise, and write clean human-readable code.

@AGENTS.md
