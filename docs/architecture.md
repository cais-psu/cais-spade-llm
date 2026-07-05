# Architecture Boundaries

This file is a human map of the current repository boundaries. It does not rename existing terms or introduce replacement terms for code symbols.

For a plain-language tour of the runtime flow and every directory, see [codebase_map.md](codebase_map.md).

## Entry points

- `cais_spade_llm/ui_main.py` is the unified entry point for the NiceGUI UI and the headless SPADE agent path. It owns startup cleanup, shutdown cleanup, and command-line flags.
- `cais_spade_llm/__main__.py` routes package execution into the application entry path.
- `Makefile` keeps common local commands: `install`, `run`, `headless`, and `bootstrap-gazebo`.

## UI

- `cais_spade_llm/ui/app.py` creates the NiceGUI application.
- `cais_spade_llm/ui/pages/` contains page rendering code.
- `cais_spade_llm/ui/components/` contains reusable NiceGUI components.
- `cais_spade_llm/ui/pages/control.py` is the main operator surface for ROS2 launch control, `digital twin`, function recording, `Teach`, `Preview in Gazebo`, `Replay in Twin`, and teleop.

## `bridge.py`

- `cais_spade_llm/ui/bridge.py` currently owns the `SystemBridge` public surface used by the UI.
- `SystemBridge` currently coordinates config files, product order files, safety intent preview state, bundle state, agent startup, ROS2 launch/process control, `digital twin`, hardware status, function recording, and UI-facing runtime status.
- Cleanup should keep `SystemBridge` callable by existing UI code first. Move helper code behind `SystemBridge` in small slices and keep behavior unchanged unless the user asks for behavior changes.
- Do not move code out of `bridge.py` by inventing replacement names for existing actions, resources, states, predicates, or user-facing terms.

## ROS2

- `ros2/cais_lab_gazebo/launch/` contains launch files for Gazebo, MoveIt, hardware MoveIt/RViz, and `digital twin` support.
- `ros2/cais_lab_gazebo/scripts/` contains runtime helpers such as `digital_twin_sync.py`, `dual_drag_markers.py`, `keyboard_teleop.py`, and `ur5e_rtde_trajectory_server.py`.
- `ros2/cais_lab_gazebo/rviz/` contains RViz configurations.
- After launch/script/RViz edits, run `make bootstrap-gazebo` before installed workspace checks because the UI uses the installed ROS2 workspace.

## `digital twin`

- `Monitor` is the hardware-led operator flow.
- `Teach` is sim/recovery authoring.
- `Preview in Gazebo` publishes to Gazebo only.
- `Replay in Twin` commits the saved sim waypoint through hardware MoveIt and then resumes hardware-to-Gazebo sync.
- Hardware MoveIt/RViz is the operator surface; Gazebo is the passive mirror unless the user explicitly asks to change this architecture.

## Safety validation and recovery

- Safety logic is split across `cais_spade_llm/agents/central_controller/`, `cais_spade_llm/agents/intelligent_product/`, `cais_spade_llm/specification/safety/`, and UI bridge preview/status code.
- Keep validation-stage checks separate from runtime dispatch gating.
- When changing recovery behavior, preserve existing action, resource, state, predicate, and bridge-event names exactly.
- Tests in `test/` describe many current contracts. Prefer adding focused tests around the exact behavior being changed rather than broad snapshot rewrites.
