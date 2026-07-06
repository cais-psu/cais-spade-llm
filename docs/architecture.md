# Architecture Boundaries

This file explains where runtime responsibilities live. It is for coding and
debugging after the project is installed. Installation steps belong in
[`README.md`](../README.md); coding rules belong in [`AGENTS.md`](../AGENTS.md).

## Entry Points

- `cais_spade_llm/__main__.py` routes package execution into the UI entry path.
- `cais_spade_llm/ui_main.py` is the application entry point for the NiceGUI UI
  and the headless SPADE agent path. It owns command-line flags, startup
  preparation, shutdown handling, and process cleanup.
- `cais_spade_llm/ui/app.py` registers the NiceGUI shell and the five UI routes:
  `/`, `/control`, `/products`, `/resources`, and `/safety`.
- `Makefile` keeps common local commands: `install`, `run`, `headless`, and
  `bootstrap-gazebo`.

## UI To Runtime

- `cais_spade_llm/ui/bridge.py` owns `SystemBridge`, the public UI-to-runtime
  surface. UI pages should call existing `SystemBridge` methods unless the user
  explicitly asks for a public interface change.
- `SystemBridge` coordinates config files, product order files, safety intent
  preview state, bundle state, agent startup, ROS2 launch/process control,
  `digital twin`, hardware status, function recording, and UI-facing runtime
  status.
- Helper code may live behind `SystemBridge`; keep the public UI surface stable
  and preserve existing actions, resources, states, predicates, and user-facing
  terms.
- `cais_spade_llm/ui/ros2_processes.py` contains ROS2 command rendering, domain
  ID helpers, workspace paths, and launch prerequisite checks used by the bridge.

## Agents And Execution

- `cais_spade_llm/agent_creator.py` builds and wires the SPADE agents.
- `cais_spade_llm/agents/intelligent_product/` owns product planning and recovery.
- `cais_spade_llm/agents/central_controller/` owns plan safety validation and
  runtime safety monitoring.
- `cais_spade_llm/agents/resource_agent/` owns robot-agent execution dispatch.
- `cais_spade_llm/resources/robot/` owns robot tasks, primitives, and controller
  adapters. Robot positions and capabilities should come from JSON manifests,
  not hardcoded constants.

## ROS2 And `digital twin`

- `ros2/cais_lab_robotics/launch/`, `config/`, `rviz/`, `worlds/`, and `scripts/`
  are the repo source files for this project's ROS2 integration.
- `make bootstrap-gazebo` copies those repo files into `~/ros2_ws`, builds the
  workspace, and installs the config/RViz assets used at runtime. Edits made only
  in `~/ros2_ws` are outside this git repository.
- Hardware MoveIt/RViz is the operator surface for `digital twin`; Gazebo is the
  passive mirror unless the user asks for a different architecture.
- UR5e hardware motion uses the RTDE trajectory server and the RG2 bridge in
  `ros2/cais_lab_robotics/scripts/`; xArm6 uses the xArm hardware driver and MoveIt
  path.
- After launch, script, config, world, or RViz edits, run `make bootstrap-gazebo`
  before checking installed workspace behavior.

## Safety Validation And Recovery

- Safety logic is split across `cais_spade_llm/agents/central_controller/`,
  `cais_spade_llm/agents/intelligent_product/`, `cais_spade_llm/specification/`,
  and UI bridge preview/status code.
- Keep validation-stage checks separate from downstream runtime dispatch gating.
- When changing recovery behavior, preserve existing action, resource, state,
  predicate, and bridge-event names exactly.

## Verification Boundary

- Python-only changes usually need `poetry check`,
  `poetry run python -m compileall -q cais_spade_llm ros2`, and a targeted import
  or CLI smoke check.
- Entrypoint changes need `poetry run python -m cais_spade_llm.ui_main --help`.
- ROS2 launch/script/RViz changes need `make bootstrap-gazebo` before installed
  workspace checks.
- Use `test/` for focused or temporary tests when a feature needs them.
