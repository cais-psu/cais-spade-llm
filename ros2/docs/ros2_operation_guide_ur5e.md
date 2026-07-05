# UR5e RTDE Operation Guide

UR5e arm hardware execution in this repo uses the CAIS RTDE trajectory server.
xArm6 and UR5e RG2 keep their existing paths.

## UR5e Hardware Stack

Start the UR5e hardware stack from the UI or launch the same pieces manually:

- `hardware_ur5e_rtde_trajectory_server`
- `hardware_ur5e_rg2_gripper`
- `hardware_ur5e_moveit`

The UR5e arm action is:

```text
/cais_ur5e_rtde_trajectory_controller/follow_joint_trajectory
```

MoveIt uses controller name:

```text
cais_ur5e_rtde_trajectory_controller
```

The RG2 action stays:

```text
/ur5e_rg2_gripper_traj_controller/follow_joint_trajectory
```

## Digital Twin

Gazebo is a passive mirror of hardware state.

UR5e mirror timing is tuned for RTDE feedback:

- `UR5E_MIRROR_POINT_TIME_SEC = 0.12`
- `UR5E_MIRROR_MIN_PUBLISH_PERIOD_SEC = 0.05`
- `UR5E_MIRROR_MIN_JOINT_DELTA_RAD = 0.0010`

## Runtime Status

Use the RTDE status file for arm execution diagnostics:

```bash
cat /tmp/cais_ur5e_rtde_trajectory_status.json
```

Expected healthy fields:

- `state: ready`
- `rtde_connected: true`
- `joint_states_fresh: true`
- `blocked_reason: ""`

## Manual Checks

```bash
ros2 action list | grep cais_ur5e_rtde
ros2 topic echo /joint_states --once
```

If MoveIt fails, check the RTDE status JSON first. It reports start-pose mismatch,
trajectory timing, max segment velocity, final joint error, and blocked reason.
