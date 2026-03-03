# CLAUDE.md — Project Instructions for Claude Code

## Project Overview

CAIS-SPADE-LLM is a multi-agent manufacturing automation system combining LLM-driven planning (SPADE + OpenAI function calling), safety-aware execution (LTL/FSA validation), dual-robot coordination (UR5e + xArm6), and a NiceGUI web UI operator console.

## Tech Stack

- **Python 3.10** (strict: `>=3.10,<3.11` — required by ROS2 Humble)
- **Package manager:** Poetry
- **Multi-agent framework:** SPADE 4.1.2 (XMPP-based)
- **LLM:** LangChain + OpenAI API
- **UI:** NiceGUI 2.0+ (port 8080)
- **Robotics:** ROS2 Humble, MoveIt2, Gazebo Classic 11
- **Safety:** LTL formulas → DFA automata (ltlf2dfa)

## Project Structure

```
cais_spade_llm/          # Main Python package
  agents/                # SPADE agents (ProductAgent, RobotAgent, CCA, UserAgent)
  resources/             # Robot controllers, camera, gripper modules
  ui/                    # NiceGUI web UI (pages, components, bridge)
ros2/                    # ROS2 Gazebo packages and launch files
test/                    # Test scripts
initialization/          # JSON configs (products, robots, CCA)
specification/           # Assembly specs, safety rules, geometry
monitor/                 # Runtime outputs (logs, state, plans)
```

## Running the Project

```bash
# Activate venv
source .venv/bin/activate

# Run in simulation mode (no ROS2 needed)
python -m cais_spade_llm --mode simulate

# Run with ROS2 + Gazebo
python -m cais_spade_llm --mode ros2
```

## Running Tests

```bash
# Smoke tests (no ROS2 needed)
python3 test/test_robot_wiring_smoke.py --robot both

# ROS2 integration tests (requires Gazebo running)
python3 test/test_ur5e_camera_move.py
python3 test/test_xarm6_camera_move.py
```

## Code Conventions

- **Type hints** everywhere (use `from __future__ import annotations`)
- **Google-style docstrings**
- **snake_case** for functions/variables, **PascalCase** for classes
- **Private methods** prefixed with underscore
- **Logging** via `logging.getLogger(...)` per module — no print statements
- **Async/await** patterns throughout (SPADE behaviours are async)
- **Lazy imports** for ROS2 modules (allows non-ROS testing)
- **Configuration-driven** — robot capabilities, products, and safety rules live in JSON manifests under `initialization/` and `specification/`

## Important Patterns

- Robot controllers are instantiated dynamically from JSON manifests (`initialization/resources/robot_*.json`)
- SPADE agents communicate over XMPP; the `SystemBridge` syncs agent state to the NiceGUI UI
- Safety validation runs both offline (pre-execution LTL→DFA) and online (runtime FSA monitoring)
- Three execution modes: `simulate` (pure Python), `ros2` (Gazebo), `real` (hardware)

## Things to Avoid

- Do NOT add `.env` or credentials to commits
- Do NOT import ROS2 modules at the top level — use lazy imports so non-ROS environments work
- Do NOT hardcode robot positions — they come from JSON manifests
- Do NOT use `print()` — use the `logging` module
