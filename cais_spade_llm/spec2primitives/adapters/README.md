# Adapters

`dual_gazebo.py` is the scene-only adapter for the no-hardware
`gazebo_dual_spec2primitives` process. It accepts a structural runtime protocol and
does not import `SystemBridge` or issue ROS2 shell commands. The registered
process forces `run_perception:=false` and launches `table_spec2primitives.world`.

Future ProductAgent and RobotAgent adapters belong under `../agents/pa/` and
`../agents/ra/`, respectively. ProductAgent and RobotAgent remain shared,
read-only runtime authorities. This directory retains only non-agent adapters,
including the existing scene-only Gazebo adapter.
