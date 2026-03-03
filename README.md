# CAIS-SPADE-LLM

Multi-agent autonomous manufacturing system that uses LLM-driven planning, safety-aware execution, and dual-robot coordination for product assembly tasks.

## Architecture

The system is built on four agent types communicating via XMPP (SPADE framework):

- **ProductAgent** — Receives assembly requirements, generates a task DAG via LLM, dispatches tasks to robots, tracks part state and execution timeline.
- **RobotAgent** — Executes pick-and-place phases (approach, grasp, place, insert, home) on UR5e or xArm6 robots via ROS2 controllers or simulation.
- **CentralControllerAgent (CCA)** — Validates plans offline using LTL/FSA safety rules, monitors execution online, triggers replanning on failures.
- **UserAgent** — Operator interface agent for seeding tasks.

```
ProductAgent  ──task──>  RobotAgent (ur5e / xarm6)
     │                        │
     │ plan_safety_check       │ resource_event
     v                        v
CentralControllerAgent (safety validation + online monitoring)
```

## Quick Start

### Prerequisites

- Python 3.10
- Poetry
- OpenAI API key in `.env` file (`OPENAI_API_KEY=...`)
- (Optional) ROS2 Humble + Gazebo for robot simulation — see `ros2/` for setup details

### Install

```bash
git clone <repo-url>
cd cais-spade-llm
poetry install
poetry run pip install "nicegui>=2.0" "aiofiles>=23.0"
```

### Run

```bash
# Operator console (web UI) — default
python3 -m cais_spade_llm

# Opens at http://localhost:8080
```

```bash
# Headless mode (no web UI, Ctrl+C to stop)
python3 -m cais_spade_llm --headless
```

## Operator Console

The web UI at `http://localhost:8080` provides a full operator console:

| Page | Path | Description |
|------|------|-------------|
| Dashboard | `/` | Start/stop system, select product & execution mode, agent status grid, quick stats |
| Plan | `/plan` | Live task DAG visualization (Mermaid), task states table, execution timeline |
| Robots | `/robots` | Robot phase pipeline, gripper/position state, JSON config editor |
| Safety | `/safety` | Safety rules table, DFA states, blocked tasks, editable safety requirements |
| Logs | `/logs` | Live-streaming logs per robot with regex filtering |
| Products | `/products` | Product spec viewer, part tracker, geometry table |

### Dashboard Controls

- **Product Specification** — Select which product assembly to run (from `initialization/products/`)
- **Execution Mode** — `simulate` (no ROS2), `ros2` (Gazebo simulation), or `real` (physical hardware)
- **Robot Environment** — `gazebo` or `real` (selects which config block to read from robot manifests)

## Execution Modes

| Mode | Description | ROS2 Required |
|------|-------------|---------------|
| `simulate` | Async simulation, no robot controllers | No |
| `ros2` | Gazebo simulation with MoveIt motion planning | Yes |
| `real` | Physical robot hardware | Yes |

## Project Structure

```
cais_spade_llm/
├── ui_main.py                 # Unified entry point (UI + headless)
├── __main__.py                # python3 -m cais_spade_llm support
├── spade_main.py              # Legacy entry point (kept as reference)
├── agent_creator.py           # Agent factory from JSON manifests
├── function_analyzer.py       # Introspects agent methods -> tools.json
├── prompts.py                 # LLM prompt templates
├── utils.py                   # File I/O helpers
│
├── agents/
│   ├── intelligent_product/
│   │   ├── product_agent.py       # Manufacturing orchestration agent
│   │   ├── process_planner.py     # NL -> requirements -> task DAG -> FSA
│   │   └── replanner/             # Online replanning logic
│   ├── resource_agent/
│   │   ├── robot_agent.py         # UR5e / xArm6 control agent
│   │   ├── resource_agent.py      # Base resource agent
│   │   └── printing_agent.py      # 3D printer agent (stub)
│   ├── central_controller/
│   │   ├── central_controller_agent.py  # Safety coordinator
│   │   ├── offline_safety_validator.py  # LTL-based offline validation
│   │   ├── online_safety_monitor.py     # Runtime DFA monitoring
│   │   ├── online_fsa_monitor.py        # Plan FSA monitoring
│   │   └── safety_logic.py             # Safety rule parsing
│   └── shared_information/
│       ├── llm_agent.py           # Base LLM-aware agent
│       └── user.py                # Operator interface agent
│
├── resources/
│   ├── robot/
│   │   ├── ros2_pick_place_controller.py  # Generic ROS2 pick/place controller
│   │   ├── ur5e_controller.py             # UR5e wrapper
│   │   └── xarm6_controller.py            # xArm6 wrapper
│   └── sensor/
│       └── camera_module.py       # Part detection (ROS2 or mock)
│
├── ui/                            # Operator console (NiceGUI)
│   ├── app.py                     # App factory, layout, page routing
│   ├── bridge.py                  # SystemBridge: SPADE <-> UI state bridge
│   ├── polling.py                 # Async log tailer
│   ├── pages/                     # One module per UI page
│   └── components/                # Reusable UI components
│
├── initialization/                # JSON configs loaded at startup
│   ├── products/                  # Product specifications
│   ├── resources/                 # Robot manifests (ur5e, xarm6)
│   ├── cca.json                   # Central controller config
│   └── tools.json                 # Auto-generated tool catalogue
│
├── specification/
│   ├── products/
│   │   ├── requirements/          # Natural language assembly tasks
│   │   ├── geometry/              # Part placement coordinates
│   │   └── cad/                   # CAD models
│   └── safety/
│       └── safety_requirements.txt  # Safety constraints
│
├── monitor/                       # Runtime outputs (auto-archived per run)
│   ├── plan/                      # Task DAG snapshots
│   ├── state/                     # Product/resource state snapshots
│   ├── history/                   # Event logs (JSONL)
│   └── debug/                     # LLM conversation traces
│
└── log/                           # Agent action logs
    ├── ur5e_actions.log
    └── xarm6_actions.log
```

## ROS2 / Gazebo

All ROS2 launch files, controller configs, world files, and sensor nodes live in `ros2/`. See:

- `ros2/ROS2_SETUP_README.md` — ROS2 installation and workspace setup
- `ros2/docs/` — Detailed guides for robot control, teleop, and operation

## Configuration

### Robot Manifests (`initialization/resources/robot_*.json`)

Each robot manifest defines capabilities, controller config, and named positions per environment (gazebo/real). Key fields:

- `execution_mode` — `simulate`, `ros2`, or `real`
- `functions` — Allowed pick/place phase functions
- `controller` — Move group, gripper, service endpoints, motion parameters
- `named_positions` — Home joint angles

### Product Specifications (`initialization/products/`)

Define assembly tasks, target resources, inbox messages, and links to requirement/geometry/safety files.

### Safety Rules (`specification/safety/safety_requirements.txt`)

Natural language safety constraints converted to LTL formulas and DFA automata for offline plan validation and online execution monitoring.

## Tests

```bash
# Robot wiring smoke test (no ROS2 needed)
python3 test/test_robot_wiring_smoke.py --robot both

# With controller home movement (requires ROS2 + Gazebo)
python3 test/test_robot_wiring_smoke.py --robot ur5e --run-controller-home

# Camera movement tests (requires ROS2 + Gazebo)
python3 test/test_ur5e_camera_move.py
python3 test/test_xarm6_camera_move.py
```
