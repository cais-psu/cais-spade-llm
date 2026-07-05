# UR5e Control Modes

This repo uses one UR5e arm hardware mode: CAIS RTDE trajectory execution.

## Hardware

The UR5e arm command owner is:

```text
hardware_ur5e_rtde_trajectory_server
```

The public action is:

```text
/cais_ur5e_rtde_trajectory_controller/follow_joint_trajectory
```

MoveIt plans normally through RViz, then executes through the RTDE trajectory
controller configured in the UR5e hardware MoveIt launch files.

## Gazebo

Gazebo is used for simulation and digital-twin visualization. In digital twin
mode, Gazebo mirrors UR5e `/joint_states` from RTDE feedback and does not own
UR5e hardware motion.

## RG2

UR5e RG2 is separate from the UR5e arm server and remains on:

```text
/ur5e_rg2_gripper_traj_controller/follow_joint_trajectory
```

## Status

UR5e arm status is written to:

```text
/tmp/cais_ur5e_rtde_trajectory_status.json
```

Use this file to diagnose trajectory validation, RTDE connection, final joint
error, and blocked reasons.
