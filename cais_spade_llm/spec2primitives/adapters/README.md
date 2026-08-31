# Adapters

`dual_gazebo.py` is the scene-only adapter for the no-hardware
`gazebo_dual_spec2primitives` process. It accepts a structural runtime protocol and
does not import `SystemBridge` or issue ROS2 shell commands. The registered
process forces `run_perception:=false` and launches `table_spec2primitives.world`.

`ui_runtime.py` composes that dual-Gazebo authority, the narrow ProductAgent
context runtime, validated package-local model configuration, the optional
OpenAI document vision boundary, the request-scoped live observation capture
boundary, and the Spec2Primitives `contexts/` root. The main PA loop, Phase 4.1
document evidence, and Phase 4.2 geometry providers are connected through the
production grounding runtime. Standalone diagnostics remain independent. The
UI does not expose geometry-processing controls; PA may invoke request-scoped
live observation through the controlled `retrieve` boundary, while the UI
polls only compact processing status.

ProductAgent and future RobotAgent adapters belong under `../agents/pa/` and
`../agents/ra/`, respectively. ProductAgent and RobotAgent remain shared,
read-only runtime authorities. This directory retains only non-agent runtime
composition, including the Gazebo and UI adapters.
