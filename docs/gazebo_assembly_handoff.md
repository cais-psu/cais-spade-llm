# Gazebo assembly implementation handoff

## Retained 11-part assembly recording

Run `185d59e911854ed28e725a2f7d802b2d` completed all 11 ordered `processPlan` requirements. Independent Gazebo observations confirmed all 11 final placements (maximum position error 0.136 mm), all four UR robots empty at home, and KMR empty at Storage. No pending tasks or reservations remained.

The [continuous recording](../cais_spade_llm/monitor/recovery_gazebo_runs/ode_island_threads_comparison/20260924T203551Z/run_01_threads_0/attempt-8c0009d681c0425381998799df3e99c4/assembly.mp4), [10× preview](../cais_spade_llm/monitor/recovery_gazebo_runs/ode_island_threads_comparison/20260924T203551Z/run_01_threads_0/attempt-8c0009d681c0425381998799df3e99c4/assembly-10x.mp4), [final scene](../cais_spade_llm/monitor/recovery_gazebo_runs/ode_island_threads_comparison/20260924T203551Z/run_01_threads_0/attempt-8c0009d681c0425381998799df3e99c4/final_scene.png), and [run report](../cais_spade_llm/monitor/recovery_gazebo_runs/ode_island_threads_comparison/20260924T203551Z/run_01_threads_0/attempt-8c0009d681c0425381998799df3e99c4/run.json) remain in the selected attempt. Both videos fully decoded: 12,208 frames over 813.87 seconds and 1,221 preview frames over 81.4 seconds. Assembly took 795.51 wall seconds at a measured 0.679 real-time factor.

KMR lifts from the shelf to the saved 1.16 m carrying point before base travel, retains its downward gripper orientation after release, and returns to Storage without a forced grasp reset. Equivalent peg half-turn and gear quarter-turn grasps are saved in simulation configuration and chosen from the observed grip. `ur5e-1` and `ur5e-2` use direct pickup and post-Conveyor home routes; `ur5e-4` returns to saved home XYZ while staying downward. Home confirmation checks fresh stable joints, planned endpoint, required XYZ and orientation, and an empty gripper. Dynamic pick/place targets, complete Cartesian paths, joint limits, no joint-space fallback, contact physics, current speeds, and hardware behavior remain in place.

The [ODE comparison results](../cais_spade_llm/monitor/recovery_gazebo_runs/ode_island_threads_comparison/20260924T203551Z/results.md) record six launches in the order 0 → 2 → 4 → 4 → 2 → 0. Both 2-thread runs and both 4-thread runs failed the KMR held-part transform check at `environment_7`; the final 0-thread run stopped at `environment_74` after a Gazebo wrist-link observation timeout. Only the retained 0-thread run validated, so the speed comparison is inconclusive and the saved setting remains `ode_island_threads=0`.

The retained run used the Gazebo-only collision bypass and saved `diagnostic_cca_bypass: true`. Gear seating is **simulated fixture attachment with tooth physics disabled**. `RecordingAttempt` owned capture and publication; automation used SystemBridge startup/Stop and never dispatched robot primitives directly. Raw failed attempts and earlier run folders were removed on 2026-09-25; their condensed comparison results remain in the linked summary.

Comparison preflight: 28 focused recording, thread-setting, and simulation-timing checks passed. The retained full video and 10× preview also passed independent complete decoding. No hardware execution was performed; production and Gazebo are stopped.

## Historical handoff before this request

Updated: 2026-09-23. The directions and measurements below describe that earlier handoff. Their raw run folders were removed on 2026-09-25; the retained current evidence is linked above.

## Current behavior

- KMR and ur5e-1, ur5e-2, ur5e-3, ur5e-4 follow the two-arm pattern inspected on `main` (`d960aee`). Named configurations, program steps and resource-owned clearance waypoints stay saved. `compute_pick_targets` and `compute_place_targets` use current observations and geometry.
- Each Cartesian move sends the fresh target and configured clearance waypoints in one `GetCartesianPath` request. MoveIt converts that supplied path to timed joint commands. There is no Python IK request per sample and no dependency on matching a pre-recorded motion endpoint. Joint-space planner fallback and partial dispatch remain disabled for these robots.
- The previous full-motion recording was too restrictive. Its 371-motion preparation record is historical; its raw folder was removed on 2026-09-25. Startup and production no longer load it. Its approximately 21-second validation was also removed from startup.
- The user requested collision checking off for this Gazebo run. `avoid_collisions: false` is saved in each UR robot's `cartesian_motion` and KMR's `task_execution`. Arm collision queries, KMR arm/gripper/payload transport sweeps and configured-route footprint checks are bypassed. Reports and command evidence label this with `collision_checks_bypassed`; skipped sweeps are not claimed as validated.
- Gazebo contact physics remains active. Separate interactive Nav2 planning retains its existing checks. Hardware collision checking is unchanged; the simulation bypass cannot authorize hardware motion.
- Finite targets, bounded joint positions/velocities/accelerations, continuous joint solutions, fresh starts, observed endpoints, custody, reservations, completion acknowledgements and Stop remain active.
- `Bypass CCA (simulation only)` remains enabled. It skips PA plan approval and resource CCA permission. Reports retain `diagnostic_cca_bypass`; physical mode rejects it.
- At that time, the operator selected `assembly_board-v1-round-4mm-m1.json`: `RGOCG4-50_Round_4mm` to M1. `assembly_board-v1-two-parts.json` was available for a later `KET4_Square_4mm` and `gear_small` recording. UR controller rate was 250 Hz, simulation speed 1, cameras/shadows off and Gazebo GUI 15 FPS. `ode_island_threads` accepted 0, 2 or 4; the later comparison is summarized above. No joint limits were increased.
- `SystemBridge`, public messages and hardware profiles remain unchanged.

## September 23 agent and Gazebo performance update

One ProductAgent still selects and dispatches work; the existing ResourceAgents own execution. Capability evaluation is deterministic guard, process, geometry and custody evaluation, not an LLM request. At most two background workers calculate detached snapshots. The resource inbox rechecks Stop, correlation, revisions, requirements and geometry before accepting a result. Intake availability is refreshed against current reservations; `prepare()` still rejects conflicting reservations. No controller runs in a calculation worker.

Report capture reuses unchanged models and completed history. Full report construction and persistence run through one background writer, with complete final reports and no older write replacing newer evidence. CCA bypass dispatches immediately after admission without another inbox-wait cycle. Task send, receipt, execution, acknowledgement send/receipt/commit, capability queue/calculation time, report capture/construction/write time and message-loop delay are recorded separately. Message-loop samples retain the most recent 600 entries plus the maximum and sample count for the whole work interval.

The baseline operator run `6cf3e23ab3574d7998f8d1291c91e291` completed KMR pickup, release at M1 and empty return to Storage. M1 acknowledged `trim` with `result: circle`. Its machining function overlapped KMR's return function for **36.08 wall seconds**. The later assembly negotiation was blocked; this is not completed assembly or joint-observed motion-overlap acceptance.

| Baseline KMR function | Worker wall seconds | Simulation seconds | Measured simulation speed |
| --- | ---: | ---: | ---: |
| `pick_part` | 12.887 | 2.300 | 0.179× |
| `move_to_resource` to M1 | 39.383 | 6.100 | 0.155× |
| `place_release` | 111.592 | 18.300 | 0.164× |
| `move_to_resource` to Storage | 42.051 | 7.400 | 0.176× |

The first pickup waited **50.33 wall seconds** for negotiation. Report snapshots in that run took **0.316–2.511 seconds** (median 0.918); persistence took **1.112–4.858 seconds**. An offline reconstruction of the full 2.2 MB report preserves every existing report field and measures a **2.29 ms median capture**, with **2.52 ms background construction** after history reuse. These offline timings do not prove a live negotiation or Gazebo speedup.

At the time of this September 23 note, the viewer rate was 15 FPS and ODE island threads were 0. The 1 ms physics step, UR 250 Hz, KMR 225 Hz, limits, contacts, all five robots, inventory and requested speed 1× were held fixed. The then-pending full-scene ODE comparison was subsequently attempted and is summarized above; no nonzero setting qualified for a speed recommendation.

The `recorded_two_part_validation` raw run and check folders cited in this historical note were removed on 2026-09-25. The offline startup fixture now prevents construction of ROS controllers. Both selected-part live acceptance and eleven-part live acceptance were outstanding for this revision.

Verification: the combined capability/motion/delivery/configuration/setup/product run passed **654 cases**, with three new test-fixture assumptions corrected. The final configuration rerun passed **25 cases** (including all six absent/explicit ODE solver cases); the final reporting rerun passed **2 cases**. No focused failures remain after those reruns. Real-inbox tests observed at most **0.320 seconds** of message-loop delay across the completed stub-executor workloads; slow calculation/construction tests handled acknowledgement and Stop within 0.5 seconds. `poetry check`, compilation, CLI help and `git diff --check` passed. `make bootstrap-gazebo` finished **16 packages**, and installed launch/controller/world sources and the rebuilt viewer plugin were checked. See `performance_checks.json` for exact logs.

## Run from localhost

1. Restart the UI and use a fresh operator-controlled Gazebo launch to load the rebuilt assets, new settings and fresh inventory. Source/launch identity checks remain enforced; the old running scene is not a validation of these changes.
2. Open `/recovery-framework?tab=setup`. Keep Simulation, the operator's selected `assembly_board-v1-round-4mm-m1.json` and `Bypass CCA (simulation only)` selected. Select `assembly_board-v1-two-parts.json` only when returning to the two-part recording.
3. Use Start Simulation and wait for readiness. Start your recording before Start System.
4. Record both placements and every robot home return, followed by ten seconds of the board close-up. KMR must finish empty at Storage, open and downward-facing; unselected inventory remains present.

To restore production collision checks later, set `avoid_collisions` to true in all five saved resource configurations and restart Gazebo/UI. Re-enable CCA separately with its Setup checkbox.

## Verification and limits

The earlier round-peg run `04eb734ba41546e58641201a753d38db` stopped before KMR pickup. `ur5e-3.move_home` confirmed a fresh stable home without dispatching motion, but a second joint-feedback read lost freshness and rejected completion. That global Stop cancelled the ongoing Storage → KMR → M1 negotiation before ProductAgent sent `pick_part`. The current intake match succeeded; this was a separate failure from the earlier unanswered intake.

Home completion now retains the actual measured joint snapshot from the successful primitive. It accepts that snapshot only for the same robot, home target and current primitive's time interval; stale, mismatched, nonfinite or held-part evidence remains invalid. KMR is probed once before production, using the selected Storage part and machine, and the same worker and launch identity continue into task execution. KMR startup, readiness, task and error logs now reach the operator console while remaining in the worker log.

Motion/delivery checks passed **285 tests**; focused home/KMR checks passed **19 tests**. A separate broader startup selection was interrupted because it instantiated ROS controllers; it is not counted as a completed suite. The owned test process and probe worker were cleaned up. The live KMR probe completed for `RGOCG4-50_Round_4mm` → M1, verifying fresh joints, open gripper, Storage inventory/dock, core services/actions and scene/asset identity. It sent no motion commands. That raw probe folder was removed on 2026-09-25. These checks were preparation evidence. The later `6cf3e23ab3574d7998f8d1291c91e291` operator run completed pickup, transfer and release, as recorded in this historical section.

The earlier run `882091833a354d698aadd6423d0733f8` stopped at `ur5e-4.place_insert`. The preceding descent reached `pre_insert_pose` within 0.00000675 m. The final 0.0025 m move was rejected before dispatch; release and lift were never commanded. The old direct-motion wrapper discarded the controller rejection detail, so the exact failed guard cannot be recovered from that run. A read-only request from the stopped robot returned the complete insertion path, passed joint-limit validation, and reached the dispatch boundary with observed joint feedback; the probe intercepted the command without moving the robot.

KMR received no pickup task. Its discovery became stale during the UR home acknowledgements. A renewed intake request has no recorded replies. The timeout branch cached an empty result until machine revisions changed, which could leave KMR idle permanently. Intake now renews timed-out requests up to three attempts and records the result and waiting duration. Tests reproduce a lost reply, verify renewed KMR pickup without any revision change, and verify bounded failure after repeated timeouts.

Cartesian dispatch now waits for fresh joint feedback after preparation before checking the original start. It still rejects changed or missing observations and Stop. Direct-motion failures retain the actual controller reason. This addresses a transient-feedback failure path; it does not retrospectively prove that this was the hidden guard in the operator's run. Motion/delivery regression checks passed **278 tests**. The failed run and read-only probe raw files were removed on 2026-09-25. Restart the UI and reset Gazebo to fresh inventory before the next operator-run test.

The selected two-part inbox/concurrency, intake, stale-offer and pending-CCA checks passed **12 tests**. An additional broader selection was interrupted after seven passes to limit CPU load; its log was retained at the time but the raw folder was removed on 2026-09-25; it is not counted as a completed suite.

The `recorded_two_part_validation` raw folder was removed on 2026-09-25. The dynamic-motion and delivery rerun passed **272 tests**, including changed targets for every UR robot and KMR, complete-path rejection, joint/start checks, Stop, and hardware isolation. The preceding motion/delivery/Gazebo run passed 435 cases with four ROS-array assertion fixture failures; those four are corrected and covered by the final rerun. All Gazebo configuration/base-controller cases passed in that preceding run, including both checked and bypassed empty-return authorization.

`make bootstrap-gazebo` completed **16 packages**. `poetry check`, compilation and `git diff --check` passed. Additional concurrency results were formerly stored in `checks/dynamic_motion_concurrency.log`; that raw check folder was removed on 2026-09-25. Earlier saved-playback test/preparation results are historical and do not describe the current executor.

Live placements, release observations, homes, motion overlap, makespan and speed remain unverified for this revision. The user owns the live run and capture. A run with collision checks and CCA bypassed does not establish collision clearance or CCA coordination. Eleven-part live acceptance remains outstanding.

Gear seating must be labeled **simulated fixture attachment with tooth physics disabled**.

## Prior delay evidence

The operator's earlier run `06fbbf6a2b58461a8672e1e1ab81ae22` already bypassed CCA. ur5e-4 pickup approach took 29.02 wall seconds, including 13.40 seconds of per-sample IK preparation. Gazebo advanced at 0.19–0.23 times real time. KMR and ur5e-4 function intervals overlapped, but the board approach failed and the run stopped. The new single-request Cartesian execution and collision bypass have not yet been timed live.

Acknowledgements can be processed while another CCA approval is pending. Timeout, rejection, Stop and late replies clean up unapproved work. Ordered background reporting avoids blocking unrelated agents; snapshot and persistence delays were measured separately. The prior raw run folders were removed on 2026-09-25.

The final concurrency selection passed **17 tests** (real agent inboxes, delayed CCA, pending approval cleanup, CCA bypass and report/Stop handling). The additional base/sweep bypass selection passed **23 tests**.
