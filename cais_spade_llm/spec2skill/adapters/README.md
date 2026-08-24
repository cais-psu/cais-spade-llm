# Adapters

`dual_gazebo.py` is the scene-only adapter for the no-hardware
`gazebo_dual_spec2skill` process. It accepts a structural runtime protocol and
does not import `SystemBridge` or issue ROS2 shell commands. The registered
process forces `run_perception:=false` and launches `table_spec2skill.world`.

Future narrow adapters may connect to the existing ProductAgent, ResourceAgent,
CCA, and RobotAgent public interfaces. Those agents remain shared, read-only
runtime authorities. Future agent adapter work requires a separately authorized
task and must not modify those authorities.
