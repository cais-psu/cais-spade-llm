# CAIS Lab Robotics Config Ownership

This directory keeps ROS2 runtime configuration separate from robot/resource
metadata. Keep each value in the file that owns that concern.

## hardware_runtime/xarm6_ur5e_hardware_runtime.yaml

Authors hardware and digital twin runtime values for `xarm6`, `ur5e`, and
`dual_robots`.

- hardware IP defaults
- hardware MoveIt defaults
- hardware joint limits used by MoveIt launch files
- UR5e RTDE limits and moveJ defaults
- hardware action names and controller names
- hardware gripper runtime settings
- paired marker defaults
- digital twin ROS domain and status path defaults
- hardware to Gazebo mirror topics

## gazebo_ros2_control/*_gazebo_ros2_control_controllers.yaml

Authors Gazebo `ros2_control` controller-manager values.

- controller-manager parameters
- Gazebo controller names
- simulated joints
- command interfaces
- state interfaces
- controller constraints and tolerances

These files do not author hardware IPs, hardware RTDE caps, resource
capabilities, or named positions.

## gazebo_initial_joint_positions/*_gazebo_initial_joint_positions.yaml

Authors Gazebo startup joint positions only.

These files do not author hardware home poses, task-level named positions, or
MoveIt speed defaults.

## robot_xarm6.json and robot_ur5e.json

The robot JSON files under `cais_spade_llm/initialization/resources/` author
SPADE/resource semantics.

- resource metadata
- capabilities
- reachability and workspace bounds
- staging areas
- task motion tuning
- attach behavior
- named positions

Do not put Gazebo `ros2_control` internals or UR5e RTDE runtime caps in the
robot JSON files.
