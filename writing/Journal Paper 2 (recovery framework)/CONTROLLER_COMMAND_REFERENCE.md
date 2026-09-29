# Controller command reference for recovery primitives

This table identifies current Gazebo commands and comparable published device commands. The device references are vocabulary for the paper; they do not claim those devices were driven by this recovery framework.

| Resource | Current Gazebo command | Published controller vocabulary | Current integration |
| --- | --- | --- | --- |
| `KMR` base | `move_base(target_pose, waypoints)` through `/KMR/move_base`; legacy `/KMR/dock` adapter | [Nav2 `ComputePathToPose`](https://docs.nav2.org/rolling/configuration_and_development/configuration_guide/core_servers/bt_plugins/actions/ComputePathToPose/) plans to a goal; [Nav2 `FollowPath`](https://docs.nav2.org/rolling/getting_started/nav2_behavior_trees/nav2_specific_nodes/nav2_specific_nodes/) runs the controller path | The Gazebo controller validates the supplied base path, stopping footprint, arm posture, and final dock observations. The legacy named action uses the same checks. |
| `ur5e-1` through `ur5e-4`, KMR arm | `move_cartesian`, `move_relative`, `move_to_named_pose`; `move_joints` and `rotate_joint` are executable only under a compatible arm policy | [Universal Robots MoveJ and MoveL](https://www.universal-robots.com/manuals/EN/HTML/SW5_19/Content/prod-usr-man/complianceUR5e/SW_sections/first_program/move.htm): MoveJ coordinates joint motion; MoveL follows a linear TCP path | Gazebo calls MoveIt and ROS controllers with configured waypoints. It does not dispatch URScript `movej` or `movel`. Both general joint commands are unavailable for recovery while Cartesian-only motion is configured. |
| `M1`, `M2` Bantam Desktop CNC | `dwell` | [Bantam G-code reference](https://support.bantamtools.com/hc/en-us/articles/115001656393-G-code-Reference): `G0` traverse, `G1` feed, `G4` dwell, `M3` clockwise spindle start, `M5` spindle stop | Current Gazebo `dwell` checks workholding and waits simulation time. There is no spindle or cutting executor. Bantam states that its spindle does not support `M4` counterclockwise operation. |
| `Conveyor`, `Buffer For Machined parts` | `move_relative` | Beckhoff documents [`MC_MoveRelative`](https://infosys.beckhoff.com/content/1033/tcplclib_tc2_mc2/70096267.html), [`MC_MoveVelocity`](https://infosys.beckhoff.com/content/1033/tcplclib_tc2_mc2/70102411.html), and [`MC_Halt`](https://infosys.beckhoff.com/content/1033/tcplclib_tc2_mc2/70107019.html) for an axis | These are comparisons for a future driven transport axis. Current Gazebo moves observed part entities after occupancy and clearance checks; no PLC axis or belt motor is connected. |
| `3D Printing Station` (`prusa_mk4_2`) | No executable printing primitive | [Prusa Buddy G-code reference](https://help.prusa3d.com/article/buddy-firmware-specific-g-code-commands_633112) lists `G0`/`G1` movement, `G28` homing, `M104` extruder temperature, and `M109` temperature and wait | `print_part` has no Gazebo steps. The three gear components are initially on the bed. Heating, extrusion, and new output generation require separate executors and observation. |

There is no lathe in this saved layout. A future lathing resource would need an identified controller, checked axis and spindle commands, workholding checks, and observed output before any primitive could be executable.

The base controller checks each sampled robot state with MoveIt's `GetStateValidity`. It supplies measured arm/gripper joints and the proposed base pose as `RobotState(is_diff=True)`. The [MoveIt state validation service](https://github.com/moveit/moveit2/blob/humble/moveit_ros/move_group/src/default_capabilities/state_validation_service_capability.cpp) starts from the current planning-scene state, and [RobotState conversion](https://github.com/moveit/moveit2/blob/humble/moveit_core/robot_state/src/conversions.cpp) preserves its attached objects for a diff. This includes the currently attached payload in the sampled clearance checks.

KMR `move_cartesian` resolves the declared Cartesian samples through MoveIt's `/compute_ik`, seeding each from the preceding joint state and checking the absolute joint change before timing. The [Humble Cartesian path service implementation](https://github.com/moveit/moveit2/blob/humble/moveit_ros/move_group/src/default_capabilities/cartesian_path_service_capability.cpp) uses the relative `jump_threshold` and time-parameterizes the path before returning it; its implementation does not use the request's `revolute_jump_threshold`. Checking only the returned time-sampled joint differences can therefore miss an earlier IK branch jump. KMR also checks interpolated `/compute_fk` poses and `/check_state_validity` results before dispatch to its joint trajectory controller. These checks are internal execution behavior of the retained Cartesian primitive.

The [Humble MPPI CostCritic implementation](https://github.com/ros-navigation/navigation2/blob/humble/nav2_mppi_controller/src/critics/cost_critic.cpp) suppresses preferential obstacle repulsion within `near_goal_distance`, while retaining collision and inscribed-footprint costs. This scene uses a 1 m distance to permit valid approaches inside the inflation band. Footprint collision checking, the 1,000,000 collision cost, and the independent base/arm/stopping checks remain active. This parameter does not change the final docking tolerance.

The Humble [PathFollowCritic](https://github.com/ros-navigation/navigation2/blob/humble/nav2_mppi_controller/src/critics/path_follow_critic.cpp) stops scoring path progress inside `threshold_to_consider`; [GoalCritic](https://github.com/ros-navigation/navigation2/blob/humble/nav2_mppi_controller/src/critics/goal_critic.cpp) starts scoring distance to the goal inside its threshold. Both are set to 0.20 m here, matching the checked docking handoff. This keeps path progress active through a curved approach instead of switching to direct goal attraction at the former 1.4 m distance. The change does not alter footprint, payload, stopping, or final-pose validation.

The current `move_base` executor selects a validated straight path for clear, nearly equal-heading route points, including long sideways aisle travel. It retains Nav2 for obstructed segments. Intermediate path endpoints use the 6 cm goal check; the last segment uses the 20 cm handoff followed by the existing precise docking controller. A measured connector, full robot sweep, and fresh stopping checks guard each transition. These are execution choices inside `move_base`; they add no function or primitive. KMR `move_to_named_pose("home")` now uses the configured transport position above the measured base and retains the observed downward orientation. An already parked arm needs no new trajectory.

A stalled intermediate Nav2 approach can hand control back within 20 cm after one simulation second with less than 5 mm translation. The controller first confirms Nav2 termination and braking, revalidates the measured connection, and uses the existing 6 cm / 0.05 radian intermediate tolerance. The 5 mm / 0.035 radian final dock remains unchanged. This is a guarded execution fallback, not a separate primitive.

For the 2026-09-29 compact layout, ur5e-4 enables `cartesian_motion.pick_rotation_at_current_pose` for `pick_approach` → `move_above_part`. The existing `move_cartesian` executor holds observed tool XYZ during the orientation change, then translates at the required grasp orientation. Placement keeps its saved rotation point. This adds no public controller command and does not enable joint primitives under the Cartesian-only policy. The [layout record](MACHINING_STATION_LAYOUT.md) lists the translated docks, robot poses and shortened Conveyor.

The additional 0.15 m shift toward assembly sets M1/M2 docks to (-4.70, 2.20) and (-2.70, 2.20), with yaw π/2. `move_to_resource` still resolves these saved targets into `move_base`; the translation changes no function or primitive interface.

### Assembly release acknowledgement (2026-09-29)

The temporary assembly-board detach inside the existing `release_part` slot
adjustment uses the configured `detach_timeout_sec` (3 seconds for these UR
controllers), replacing a fixed 0.5-second wait. A missing acknowledgement
still fails execution and prevents committing the function's out state. The
response is retained in command evidence. This changes neither the primitive
catalog nor the saved composition, Cartesian path, or assembly target.

### Recorded compact-layout acceptance (2026-09-29)

Run `82cb3858f1a94394a8f846a0b6831cf3` passed the normal 11-part workflow with
195 matching transitions and zero KMR or UR equipment contacts across 622,956
contact messages. All three ur5e-4 pickup rotations passed the additional
collision sweep validation; all 22 KMR base actions retained their movement
and stopping-clearance checks. The final placement error was at most 0.040 mm.
Saved compositions, catalogs, contracts and revision 4 are unchanged.
The [recording and evidence](../../cais_spade_llm/monitor/recovery_gazebo_runs/attempt-f56611c75e094e28b449ddcbb85aa8bd/README.md) include 1×, 10× and
simulation-clock-paced RTF 2 videos, plus the timing comparison against 22m30s.

### Raised KMR startup pose (2026-09-29)

`KMR.parked_arm_configuration` now stores the validated final seven joint
positions from the first `park` trajectory in run
`82cb3858f1a94394a8f846a0b6831cf3`. Its tool position in the base frame is
`[-0.25, 0.50, 1.16]` m, with a downward gripper. Fresh Gazebo launches and
resets initialize both the arm model and controller from these saved values.
The selected part still determines its Storage dock and pickup targets.

Before production, the existing startup probe checks fresh measured joints,
tool position, downward orientation and MoveIt collision clearance. This is
an internal readiness check, not a selectable primitive. The saved
`pick_approach` composition still begins with `move_to_named_pose("transport")`;
when the measured arm is already positioned, that step sends no trajectory.
Functions, catalogs, state contracts and program revision 4 are unchanged.
The [fresh recorded acceptance](../../cais_spade_llm/monitor/recovery_gazebo_runs/attempt-f2090c8976d84acd90cf35353c5acc34/README.md)
passed all 11 parts in run `2a18a5d1ee544380b6ed6be402e43839`: 195 matching function/model
transitions, zero equipment contacts, and all five robots visible. The first
transport step took 0.0129 s and dispatched no trajectory (previously 17.598 s).
The [1×](../../cais_spade_llm/monitor/recovery_gazebo_runs/attempt-f2090c8976d84acd90cf35353c5acc34/assembly.mp4) and
[20×](../../cais_spade_llm/monitor/recovery_gazebo_runs/attempt-f2090c8976d84acd90cf35353c5acc34/assembly-20x.mp4) videos decode completely.
Full duration was 21m13s versus 21m06s previously; the measured startup saving
did not produce an overall speedup in this run. See the
[startup evidence](../../cais_spade_llm/monitor/recovery_gazebo_runs/attempt-f2090c8976d84acd90cf35353c5acc34/startup_pose_audit.json) and
[timing comparison](../../cais_spade_llm/monitor/recovery_gazebo_runs/attempt-f2090c8976d84acd90cf35353c5acc34/speed_comparison.json).
