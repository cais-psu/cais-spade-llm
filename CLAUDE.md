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

# Run in dry run mode (no ROS2 needed)
python -m cais_spade_llm --mode dry_run

# Run with ROS2 + Gazebo
python -m cais_spade_llm --mode simulation
```

## Running Tests

```bash
# Full pytest suite (no ROS2 needed)
poetry run python -m pytest test/

# Digital twin / Teach / Replay in Twin / ROS2 launch gate (per AGENTS.md)
poetry run python -m pytest test/test_ur5e_rg2_rtde_gripper.py
```

## Conventions & standards

Code conventions, repository boundaries, and the enforced clean-code standards
(ruff rules, `make lint` / `make lint-fix`, pre-commit, and the docstring &
comment policy) live in **AGENTS.md** and the files under `docs/`. AGENTS.md is
imported below so this file and AGENTS.md never drift — read it before changing
code.

@AGENTS.md

## Important Patterns

- Robot controllers are instantiated dynamically from JSON manifests (`initialization/resources/robot_*.json`)
- SPADE agents communicate over XMPP; the `SystemBridge` syncs agent state to the NiceGUI UI
- Safety validation runs both offline (pre-execution LTL→DFA) and online (runtime FSA monitoring)
- Three execution modes: `dry_run` (pure Python), `simulation` (Gazebo), `physical` (hardware)
