# Adapters

`dual_gazebo.py` is the scene-only adapter for the no-hardware
`gazebo_dual_spec2primitives` process. It accepts a structural runtime protocol and
does not import `SystemBridge` or issue ROS2 shell commands. The registered
process forces `run_perception:=false` and launches `table_spec2primitives.world`.

`ui_runtime.py` composes that dual-Gazebo authority, the narrow ProductAgent
context runtime, and the Spec2Primitives `contexts/` root for the Phase 2.1 UI
connection.

ProductAgent and future RobotAgent adapters belong under `../agents/pa/` and
`../agents/ra/`, respectively. ProductAgent and RobotAgent remain shared,
read-only runtime authorities. This directory retains only non-agent runtime
composition, including the Gazebo and UI adapters.
