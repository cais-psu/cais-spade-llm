# Codebase Map

Use this file to find the right code path quickly. It is a coding/navigation map,
not an install guide.

## What Runs When The App Starts

`python -m cais_spade_llm` or `make run` follows this path:

1. [`cais_spade_llm/__main__.py`](../cais_spade_llm/__main__.py) hands off to
   [`cais_spade_llm/ui_main.py`](../cais_spade_llm/ui_main.py).
2. [`cais_spade_llm/ui_main.py`](../cais_spade_llm/ui_main.py) parses mode flags,
   prepares startup/shutdown handling, and starts the UI or headless agents.
3. [`cais_spade_llm/ui/app.py`](../cais_spade_llm/ui/app.py) builds the NiceGUI
   shell and registers five routes: `/`, `/control`, `/products`, `/resources`,
   and `/safety`.
4. [`cais_spade_llm/ui/bridge.py`](../cais_spade_llm/ui/bridge.py) exposes
   `SystemBridge`, the public surface that UI pages call for runtime work.
5. [`cais_spade_llm/agent_creator.py`](../cais_spade_llm/agent_creator.py) wires
   ProductAgent, CCA, RobotAgent, and UserAgent when the system starts.
6. Robot execution flows through `resources/robot/`: task methods, primitives,
   then simulation or hardware controllers.

## Modes

| Mode | What it runs |
| --- | --- |
| `dry_run` | Pure Python agents, planning, and safety validation. No ROS2 required. |
| `simulation` | Python system plus Gazebo/MoveIt launch files under `ros2/cais_lab_robotics/`. |
| `physical` | Python system plus hardware controllers, RTDE/xArm paths, and machine-specific setup. |

## Where To Look

| Task | Start here |
| --- | --- |
| UI startup, CLI flags, shutdown behavior | `cais_spade_llm/ui_main.py` |
| Page routing and layout shell | `cais_spade_llm/ui/app.py` |
| UI button behavior and runtime state | `cais_spade_llm/ui/bridge.py` |
| ROS2 launch commands, domains, workspace prerequisites | `cais_spade_llm/ui/ros2_processes.py` |
| Control page behavior | `cais_spade_llm/ui/pages/control.py` |
| Product files and order handling | `cais_spade_llm/product/` and `cais_spade_llm/initialization/products/` |
| Product planning and recovery | `cais_spade_llm/agents/intelligent_product/` |
| Safety validation and monitoring | `cais_spade_llm/agents/central_controller/` and `cais_spade_llm/specification/safety/` |
| Robot task execution | `cais_spade_llm/resources/robot/` |
| ROS2 launch/config/RViz/world assets | `ros2/cais_lab_robotics/` |
| ROS2 helper scripts | `ros2/cais_lab_robotics/scripts/` |
| Generated runtime state and logs | `cais_spade_llm/monitor/`, `cais_spade_llm/log/`, `cais_spade_llm/user_verified_*/` |
| Focused tests while developing | `test/` |

## ROS2 Workspace Rule

The repo source files are under `ros2/cais_lab_robotics/`. The running ROS2
workspace is under `~/ros2_ws`, outside this git repository.

After changing repo ROS2 launch/config/RViz/world files, run:

```bash
make bootstrap-gazebo
```

Then test the installed workspace behavior.

## Quick Answers

- Robot refused a step: inspect CCA safety validation and the relevant safety
  rules first.
- UI button did not do what was expected: search the page under `ui/pages/`, then
  follow the call into `SystemBridge`.
- Product setup changed: inspect product JSON under `cais_spade_llm/initialization/`
  and product specs under `cais_spade_llm/specification/`.
- ROS2 launch failed: check `cais_spade_llm/ui/ros2_processes.py`, then the
  matching file under `ros2/cais_lab_robotics/`.
- A change touches cleanup/extraction only: use [`refactoring.md`](refactoring.md).
