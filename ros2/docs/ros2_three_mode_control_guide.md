# ROS2 Control Guide

This repo uses three operator surfaces:

- Gazebo simulation
- Hardware MoveIt
- Digital twin monitor/replay

## xArm6

xArm6 hardware keeps its existing xArm ROS2 stack and action paths.

## UR5e

UR5e arm hardware uses the CAIS RTDE trajectory server:

```text
/cais_ur5e_rtde_trajectory_controller/follow_joint_trajectory
```

UR5e hardware MoveIt launches always use controller name:

```text
cais_ur5e_rtde_trajectory_controller
```

UR5e RG2 remains separate:

```text
/ur5e_rg2_gripper_traj_controller/follow_joint_trajectory
```

## Digital Twin

Monitor mode keeps hardware as the source of truth and Gazebo as the passive
mirror. Replay commits saved motion through the hardware execution path, then
hardware-to-Gazebo mirror resumes.

## Status Files

```text
/tmp/cais_ur5e_rtde_trajectory_status.json
```

Use the status file to inspect RTDE connection state, trajectory retiming,
blocked reason, and final joint error.
