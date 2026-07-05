# Codebase Map

One page to re-orient yourself. What runs, in what order, and where to look.

## What happens when you run it

`python -m cais_spade_llm` (or `make run`) does this, in order:

1. [`cais_spade_llm/__main__.py`](../cais_spade_llm/__main__.py) — thin launcher, hands off to:
2. [`cais_spade_llm/ui_main.py`](../cais_spade_llm/ui_main.py) — the real entry point.
   Parses flags (`--mode dry_run|simulation|physical`, `--headless`), installs
   startup/shutdown cleanup, then starts the web UI (or agents only with `--headless`).
3. [`cais_spade_llm/ui/app.py`](../cais_spade_llm/ui/app.py) — builds the NiceGUI app
   at http://localhost:8080 and registers exactly **5 pages**:
   `/` dashboard · `/control` · `/products` · `/resources` · `/safety`.
4. [`cais_spade_llm/ui/bridge.py`](../cais_spade_llm/ui/bridge.py) — **`SystemBridge`, the
   hub of the whole system.** Every UI button lands here. It owns: config/product-order
   files, agent startup, ROS2 launch/process control, digital twin, hardware status,
   function recording, safety intent preview, and runtime status. (Its digital-twin
   helpers live in [`ui/digital_twin.py`](../cais_spade_llm/ui/digital_twin.py); an
   embedded XMPP server is started as a subprocess via `ui/xmpp_server_runner.py`.)
5. When you press start, `SystemBridge.start_system()` creates the **SPADE agents**
   (wired up by [`agent_creator.py`](../cais_spade_llm/agent_creator.py)), which talk
   to each other over XMPP:
   - **ProductAgent** (`agents/intelligent_product/`) — plans the product's assembly
     with the LLM (`process_planner.py`), and re-plans on failure
     (`product_recovery_controller.py`, `replanner/llm_bridge/`).
   - **CCA** (`agents/central_controller/`) — validates every plan against safety
     rules before execution: LTL formulas → DFA automata (`safety_logic.py`,
     `plan_safety_validator.py`), plus runtime FSA monitoring.
   - **RobotAgent** (`agents/resource_agent/robot_agent.py`) — executes approved steps
     on a robot by calling the controllers in `resources/robot/`.
6. **Robot execution** (`resources/robot/`) — `robot_tasks.py` → `robot_primitives.py`
   → a controller (`gazebo_pick_place_controller.py` for simulation, RTDE/xArm
   hardware controllers for `physical`). Which robot has which capability comes from
   JSON manifests in `initialization/resources/robot_*.json` — never hardcoded.

## The 3 modes

| Mode | What actually runs |
|---|---|
| `dry_run` | Pure Python: agents + planning + safety validation. No ROS2 anywhere. Best for development. |
| `simulation` | Same, plus Gazebo via the launch files in `ros2/cais_lab_gazebo/` (bootstrap once with `make bootstrap-gazebo`). |
| `physical` | Same, plus real UR5e (RTDE) / xArm6 drivers. Physical perception services are still stubs. |

## Directory guide — "what do I open when…"

| Path | What it is |
|---|---|
| `cais_spade_llm/ui_main.py` | Entry point: flags, startup, shutdown |
| `cais_spade_llm/ui/` | Web UI: `app.py` (pages), `bridge.py` (SystemBridge hub), `pages/`, `components/` |
| `cais_spade_llm/agents/` | The three agent roles: `intelligent_product/` (LLM planning), `central_controller/` (safety), `resource_agent/` (execution), `shared_information/` (LLM/user helpers) |
| `cais_spade_llm/resources/` | Robot/machine/sensor drivers and primitives — the "hands" |
| `cais_spade_llm/prompts.py` | The LLM prompt texts |
| `cais_spade_llm/agent_creator.py` | Builds and wires the agents at startup |
| `cais_spade_llm/initialization/` | **JSON manifests: products, robot capabilities, CCA config.** Change behavior here, not in code |
| `cais_spade_llm/specification/` | Assembly specs, safety rules, geometry |
| `cais_spade_llm/safety/`, `monitor/`, `log/`, `bundles/`, `user_verified_*/` | Runtime outputs (DFAs, state, plans, logs) — generated, mostly gitignored |
| `ros2/cais_lab_gazebo/` | Gazebo/MoveIt launch files, digital-twin + teleop scripts, RViz configs. **Never auto-format this tree — tests assert its exact source text** |
| `test/` | 5 pytest files; `test_ur5e_rg2_rtde_gripper.py` is the gate for digital-twin/ROS2-launch changes |
| `initialization/`, `specification/` (repo root) | Root-level copies of configs used by some flows |
| `docs/` | This map + `architecture.md` (boundaries) + `code_review.md` + `refactoring.md` |
| `TODO/`, `writing/` | Research notes and paper drafts — not code |

## Quick answers

- **"Why did the robot refuse to do a step?"** → CCA safety validation:
  `agents/central_controller/safety_logic.py` and the rules in `specification/safety/`.
- **"Where do UI buttons actually do things?"** → `ui/bridge.py` (`SystemBridge`) —
  search for the button's label in `ui/pages/`, follow the call into the bridge.
- **"How do I add/modify a product?"** → JSON in `cais_spade_llm/initialization/products/`.
- **"Robot moves wrong in Gazebo?"** → `resources/robot/gazebo_pick_place_controller.py`
  and the launch files in `ros2/cais_lab_gazebo/launch/`.
- **"What ran last time?"** → `monitor/` (state, plans, history) and `log/`.
- **Code health:** `make lint-report` shows remaining debt; `make lint-fix` cleans
  what's automatable. Conventions live in `AGENTS.md`.
