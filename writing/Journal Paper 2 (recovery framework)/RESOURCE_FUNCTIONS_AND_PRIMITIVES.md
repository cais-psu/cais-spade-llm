# Gazebo resource functions and primitives

**Document role: saved function and primitive reference.** Use the
[implementation plan](IMPLEMENTATION_PLAN.md) for future work and experiment
gates, and [implementation history](IMPLEMENTATION_HISTORY.md) for the dated
project narrative. Commissioning results below retain their original limits;
later accepted runs supersede earlier incomplete-run status.

The saved source for this layout is `cais_spade_llm/initialization/recovery_framework_gazebo.json`, in `resource_programs`. It stores each resource's complete primitive catalog separately from its ordered function steps. The Resources UI reads and edits this record. Gazebo and recovery pin its validated revision. Changed revisions, undeclared commands, and commands without a valid executor are rejected. Controller observations, attachment state, clearance, and final position are checked during execution even when a function's saved composition is edited.

The editor accepts parameter changes and additional supported robot motion or perception steps. Required steps retain their relative order, executor, guards, and safety metadata; the required final step remains last. Added steps require unique IDs, supported arguments, and valid bindings to earlier outputs. The executor consumes the resulting saved order. This permits added waypoints without allowing an edit to remove a required handoff or final observation.

A function owns a modeled in state and out state. A primitive takes bound arguments and reports a direct physical or measured effect. Primitive contracts deliberately do not assign the complete function state or encode an entire recipe. The LLM may propose a recovery sequence; the validator checks its steps and bindings without adding or reordering steps. **Primitive contracts alone are insufficient to synthesize the complete recovery composition symbolically.**

## Implemented catalogs

| Resource | Complete implemented primitive catalog | Function events and names |
| --- | --- | --- |
| `ur5e-1` through `ur5e-4` | `detect_parts`, `compute_pick_targets`, `compute_place_targets`, `move_to_named_pose`, `move_cartesian`, `move_relative`, `grasp_part`, `release_part`, `open_gripper`, `close_gripper`, `attach_part`, `detach_part`, `move_joints`, `rotate_joint` | `pick_approach`, `pick_grasp`, `place_approach`, `place_release` → `place_insert`, `move_home` |
| `KMR` | The same 14 arm and gripper commands, plus `move_base` | `pick_approach`, `pick_part`, `move_to_resource`, `place_approach`, `place_release`, `move_to_location` |
| `M1`, `M2` | `dwell` | `machine_part` → `trim_part` |
| `Conveyor` | `move_relative` | `advance_conveyor` → `move_parts_downstream` |
| `Buffer For Machined parts` | `move_relative` | `advance_part` → `move_to_next_zone` |
| `3D Printing Station` | None | `print_part` is planned, with no executable steps |
| `Storage`, `Exit` | None | Participate in shared handoffs |

The eight commands from main are retained as the robot baseline. The four gripper and attachment controls remain separately callable for recovery, although `grasp_part` and `release_part` invoke them internally. `move_joints(joints)` takes absolute radians for all configured arm joints: six for a UR5e, seven for KMR. `rotate_joint(joint_name, delta_deg)` changes one measured joint and holds the others at their measured start positions. Both are unavailable to recovery under this scene's Cartesian-only arm policy. The KMR machine-placement half turn remains in the configured Cartesian transfer waypoints. KMR Cartesian sampling is 5 mm. A stationary replay of the recorded M1 trajectory found a 0.14 mm finger/spindle intersection between the former 10 mm samples; replanning the same waypoints at 5 mm passed 1,710 continuous joint-interpolation checks. The configured Cartesian positions and transfer turn are preserved. The executor now consumes each declared XYZ/quaternion waypoint, including the turn along the clearance arc. It resolves IK from each preceding sample with an absolute joint-step bound before timing, then checks interpolated FK against the declared path, downward orientation, and the collision scene. The saved half-turn midpoint remains active across the configured docking-yaw tolerance. Empty withdrawal returns to the observed pre-placement orientation, preventing repeated transfers from accumulating a wrist turn. This changes the earlier executor behavior that discarded waypoint orientations and rotated at the starting point.

## Function transitions and saved KMR steps

| Function | In state and participating condition | Out state and participating effect | Gazebo primitives in order |
| --- | --- | --- | --- |
| `KMR.pick_approach` | `idle`, empty gripper, Storage part still in inventory | `at_pick`, same Storage inventory; observed part-specific pickup position | `move_to_named_pose` → `move_base` → `open_gripper` → `compute_pick_targets` → `move_cartesian` |
| `KMR.pick_part` | `at_pick`, Storage inventory still owns the part | `carrying`, KMR holds and is attached to the part; Storage inventory clears | `move_cartesian` → `grasp_part` → `move_cartesian` |
| `KMR.move_to_resource` | `carrying` with held part, or `idle` with no held part; configured source location | Same held-part state; observed location changes to configured destination | Loaded or empty: `move_base`; empty return to Storage: `move_base` → `move_to_named_pose` |
| `KMR.place_approach` | `carrying` at `M1` or `M2`, machine workholding empty | `positioned`, same held part and empty workholding | `compute_place_targets` → `move_cartesian` |
| `KMR.place_release` | `positioned`, bound machine workholding empty | `idle`, no held part; machine workholding becomes loaded | `move_cartesian` → `release_part` → `move_cartesian` → `move_cartesian` |
| `KMR.move_to_location(x, y, yaw)` | `idle` or `carrying`, with its current held part | Same function state and held part; measured base pose changes. `resource_location` is the matched configured dock, or `None` | `move_base` |

KMR `home` is the configured parked transport position above its measured base, using `transport_tcp_position_in_base_frame_m` and the observed downward gripper orientation. It does not depend on the first part in the order. The empty-return composition remains `move_base` → `move_to_named_pose("home")`; if the arm is already parked, the latter verifies the endpoint without sending another arm trajectory. The next `pick_approach` resolves the actual next part and its configured pickup pose.

`pick_part` commits the Storage inventory handoff; `place_release` commits the M1/M2 loading handoff. The two approaches establish observed positions without transferring inventory. `move_to_location` has new idle and carrying model variants appended after the existing event IDs; `observed_resource_location` is bound from the controller result. Existing event IDs are unchanged.

UR functions retain their configured state sequence: `pick_approach` `idle → at_pick`, `pick_grasp` `at_pick → picked`, `place_approach` `picked → positioned`, `place_insert` `positioned → placed`, and `move_home` `any → idle`. Their saved Gazebo programs use the configured Cartesian waypoints. For an assembly slot, `place_insert` uses `move_cartesian` from the pre-insert pose to the configured `insert_pose` before `release_part`; the release guard requires that final seated position. The physical insertion path remains separate. In Gazebo, only the Buffer and printer handling robots receive an assembly-slot binding in `place_insert`; `release_part` applies that correction only when the bound destination is `assembly_board-v1`. Machine handling robots and standalone releases use ordinary release without a slot correction. `M1` and `M2` use `dwell` for `loaded → completed`. Conveyor `move_relative` updates all belt residents by the same displacement. Buffer `move_relative` changes the configured zone occupancy.

## Base movement and command ownership

`move_to_resource(target_resource)` resolves its named target and route from the saved scene. `move_to_location(x, y, yaw)` binds a world-frame `[x, y, yaw]` target in metres and radians. Both call one `move_base(target_pose, waypoints)` primitive. `/KMR/move_base` is the coordinate action; `/KMR/dock` is a compatibility action through the same validated path follower. Between configured route points with headings within 0.05 radians, the controller first tests a straight path at the requested heading, including sideways travel. There is no 1 m length limit. Both the padded base footprint and the complete measured arm/gripper/payload sweep must pass before that path can be selected. A blocked straight path falls back to Nav2 planning; the selected Nav2 path receives the same validation. Each intermediate endpoint uses the existing 6 cm position and 0.05 radian yaw checks. The next connection is rebuilt from the measured base position and validated before travel. Only the last segment hands off within the configured 20 cm slow-approach distance for precise final alignment. The final configured limit remains 5 mm in position and 0.035 radians in yaw. Direct public Nav2 path execution retains its 6 cm goal check. All paths retain observed braking, fresh stopping-footprint checks, and arm/payload clearance checks. The worker checks the parked arm, grasp transform, attachment, and full robot transport sweep on the saved route. The original 0.5 rad/s base limit and 8 mm attachment tolerance apply during motion and after arrival. A Gazebo model plugin adds the tangential link velocity required for the parked kinematic arm to rotate rigidly with the base; it does not expose a new primitive or controller action. The existing `iiwa_link_7` attachment remains in use. Base footprint checks do not inherit the arm's `avoid_collisions` setting. Invalid maps, stale observations, blocked targets, and failed path or clearance checks stop travel.

If Nav2 makes less than 5 mm of translational progress for one simulation second within 20 cm of an intermediate endpoint, `move_base` cancels that Nav2 goal and waits for its terminal response and observed braking. It then validates the short connection from the measured pose and aligns only to the existing 6 cm / 0.05 radian intermediate tolerance. Final docking still uses 5 mm / 0.035 radians. A Nav2 abort, cancellation timeout, stale observation, blocked connection, or failed robot sweep rejects the handoff. This handles the measured 11 cm stall at Storage during the 8 mm square peg delivery without adding a function or primitive or changing a route point.

The previously saved Storage-to-M1 and Storage-to-M2 base route points and eight part-specific pickup poses passed the static padded-base-footprint map audit. No base or arm waypoint was changed by that audit. The base controller additionally checks measured arm and gripper positions and the current planning-scene attachment along the actual selected Nav2 path, reconnected segments, precision alignment, and stopping sweeps. Missing collision services, cancellation, stale observations, and collision results reject movement. A stopping-check timeout immediately publishes zero velocity and retries from fresh observations; the timed-out result never authorizes motion. Repeated timeouts lasting the configured arm-state timeout abort the action. Path preparation retains its separate timeout. Collision-service requests are bounded to four outstanding states per sweep. These checks run independently of the arm's `avoid_collisions` setting. Observation age for stopping distance uses simulation time, so pausing Gazebo does not invent additional physical travel; command and observation freshness remain mandatory.

Fixed occupancy data and its blocked-cell index are cached on each map update. This changes lookup cost, not the occupied/unknown-cell policy or footprint sampling. Independent MoveIt state-validity queries use at most four outstanding requests; seeded IK remains sequential. Timeouts, missing responses, cancellation, and invalid states still prevent dispatch. Sampling density, physics timestep, joint limits, configured arm waypoints, and the saved function programs are unchanged by these optimizations.

`dwell` performs workholding and clearance checks, simulation-time waiting, and completion observation inside one command. Conveyor and Buffer `move_relative` perform resident or zone observation, displacement calculation, clearance, movement, and arrival verification inside one command. `attach_part` and `detach_part` own Gazebo attachment and planning-scene updates. The configured 1 mm support allowance applies only to the bottom face of the attached payload collision box; release restores its full dimensions. Robot and equipment collision checks remain active. Motion results and runtime snapshots provide measured poses. Internal observations remain evidence, not selectable primitives or primitive trace steps.

## Removed standalone commands

- Robot helpers: `localize_assembly_board_v1`, `get_current_pose`, `delay`, `snap_part_to_slot`; hardware-only `move_insert` is absent from this Gazebo catalog.
- KMR helpers: `custody`, `part_collision`, `compute_pick_dock_pose`, all former `observe_*`, `confirm_*`, and `validate_transport` primitive steps.
- Machine helpers: `observe_workholding`, `verify_process_clearance`, `confirm_process_observation`.
- Transport helpers: `observe_belt_residents`, `compute_shared_displacement`, `observe_zone_part`, `compute_downstream_motion`, `verify_transport_clearance`, `confirm_arrival`.
- Printer placeholders: `validate_print_request`, `run_print_cycle`, `confirm_printed_output`.
- Duplicate movements: `move_pose`, KMR `move_to_pose`, `move_to_configuration`, `rotate_arm_base`, KMR primitive `move_home`, and `dock`. Their retained equivalents are `move_cartesian`, `move_joints`, `rotate_joint`, `move_to_named_pose`, and `move_base`.

Historical traces keep their original names. New primitive traces record only the saved commands above, with controller operations attributed to their owning step. Action timeouts and cancellation retain measured feedback, controller status, and the accepted handle for stop cleanup. A failed command can leave measured partial progress; a function's out state commits only after its completion evidence is accepted.

## Simulation limits and acceptance

`dwell` waits in simulation; it does not spin or cut. Conveyor and Buffer `move_relative` change observed Gazebo part positions; no belt motor is driven. Three gear components start preprinted on the printer bed; `print_part` is not executable. Future printer, spindle, and motor commands in [the controller reference](CONTROLLER_COMMAND_REFERENCE.md) become recovery-selectable only after validated Gazebo executors exist.

A historical 2026-09-25 run used the previous `dock`-based programs and completed one Storage-to-M1 KMR delivery, plus separate staged M2, Conveyor, and Buffer checks. That trace is preserved as prior-revision evidence. It does not validate this revised program or the complete 11-part workflow. Acceptance for this revision requires the normal run of four square pegs through M1, four round pegs through M2, and three initially preprinted gears through assembly, with all 11 parts placed and no KMR equipment contact. The 2026-09-28 run `e0e168d9bd9346249d5ff88a4481edcc` completed 153 function transitions with saved-step/trace agreement and assembled nine parts. It failed acceptance: KMR contacted M2 during the 12 mm round peg transfer, then timed out near the first route point for the 16 mm square peg delivery. Recorded Cartesian trajectory knots revealed a detour hidden by IK branch changes and subsequent time parameterization. These results are diagnostic evidence; the complete workflow remains unverified until a fresh run passes.

The subsequent normal run `6eedba1e63b64a6981b0072d9fcf94c5` assembled nine parts with zero KMR equipment contact messages. Its 150 completed function transitions matched the saved compositions, including 15 attributed base actions. It still failed acceptance: the empty return from M2 stalled approximately 0.57 m from a Nav2 segment endpoint and reached the action timeout. Recorded feedback showed fresh observations and successful clearance checks while Nav2 requested nearly zero translation. The next configuration increases `CostCritic.near_goal_distance` from 0.5 m to 1 m; this changes the soft obstacle-distance preference near a valid goal and preserves collision rejection. Full acceptance remains pending.

A bounded fresh-scene check (`base_return_95ff59d4475c4acd90b66fb10c2438f3`) subsequently completed four `move_to_location` calls, including departure from the previously stalled pose and return to Storage, with zero equipment contacts across 29,909 native contact messages. The arbitrary targets reported `resource_location=None`; the final dock reported `Storage`, with `idle` and no held part preserved. The same check executed `detect_parts`, which KMR nominal functions do not use, and `move_to_named_pose` through recovery dispatch before any delivery. This is additional command evidence, not a substitute for normal 11-part acceptance.

The next normal run, `2a65b7990cd746d1b1cbb3592ba3ffb0`, assembled six parts and recorded zero KMR equipment contacts across 449,278 native contact messages. All 84 completed transitions matched the saved programs and formal state projections. It timed out on an empty M1 return, 1.21 m from the goal, with fresh clearance observations and almost zero requested translation. The 1 m obstacle-preference adjustment alone was therefore insufficient. The following configuration keeps `PathFollowCritic` active until 0.20 m and starts `GoalCritic` at that same distance, matching the precision docking handoff; fresh route and complete-workflow checks are required.


The 2026-09-28 recorded normal run `5096f2ba7cd2409bacd171b575cec86e` passed complete Gazebo acceptance under saved program revision 4. All 11 parts reached valid configured final poses: four square pegs through M1, four round pegs through M2, and three initially preprinted gears through assembly. Its 195 completed transitions matched the saved primitive steps and formal model effects; 22 KMR base actions were attributed to `move_base`. The native Gazebo contact monitor counted 803,005 messages and zero KMR equipment contacts. The [acceptance audit](evidence/2026-09-28-gazebo-11-parts/acceptance_audit.json), [trace audit](evidence/2026-09-28-gazebo-11-parts/trace_audit.json), and [Gazebo video](evidence/2026-09-28-gazebo-11-parts/gazebo_11_parts_success.mp4) record this run. For nearby M1 route points, `move_base` can select a separately footprint-validated straight path when Nav2 returns a longer detour; the same swept robot and stopping checks still guard execution. The UR named-home target is cached after a measured FK lookup to avoid repeated `/compute_fk` timeouts during the complete workflow.

The subsequent recorded normal run `2dbf14ffe977425096a3d354b625df90` passed all 11 placements and 195 matching transitions after the base-execution and parked-home changes described above. It recorded zero KMR equipment contacts across 700,777 native messages; all five robots were visible in inspected GUI and decoded video frames. All eight empty returns kept the arm parked without sending another arm trajectory. The saved program revision remains 4, and the scene, functions, primitive catalogs, state contracts and configured arm waypoints are unchanged by these performance changes. The [recording and acceptance evidence](../../cais_spade_llm/monitor/recovery_gazebo_runs/attempt-10fae2a02d124a1c987439ce84686757/README.md) include full and 10× videos. The full recording fell from 24m45.93s to 22m30.47s (9.1%); the [settings comparison](../../cais_spade_llm/monitor/recovery_gazebo_runs/performance-20260928/README.md) documents the benchmark scope and the final intermediate-stall correction.

## Compact layout and pickup rotation, 2026-09-29

The [layout](MACHINING_STATION_LAYOUT.md) places M1 at (-4.05, 1.90), M2 at (-2.05, 1.90), Storage at (-6.45, 2.30), and the 4.20 m Conveyor at (-2.85, 0.50). Related robot bases, fixtures, access poses, inventory and waypoints are translated together. Assembly resources retain their positions. The saved function compositions, primitive catalogs, state transitions and revision 4 are unchanged; new scene/map fingerprints are required.

ur5e-4 `pick_approach` retains its existing `move_cartesian` step. `pick_rotation_at_current_pose=true` makes `move_above_part` turn at fresh measured XYZ before translation. Placement continues to use its existing rotation waypoint, and `home_preserve_orientation` remains true. Commissioning must validate the complete new pickup trajectory, including the open gripper and printer, before accepting a recorded normal 11-part run.

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
