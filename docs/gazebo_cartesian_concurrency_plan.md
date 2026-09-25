# Gazebo Cartesian waypoints and concurrent resource agents

## Current authorized execution

The user approved agent-operated recording of all 11 parts from `assembly_board-v1-recovery-framework.json`. This supersedes the operator-only startup and single-part selection instructions retained below as history. Required tool rotations are permitted at saved clearance waypoints; XYZ travel keeps orientation fixed. Pick/place targets and their Cartesian trajectories remain calculated at runtime. Run `185d59e911854ed28e725a2f7d802b2d` completed all 11 parts with empty home returns and KMR at Storage. Its continuous recording and 10× preview passed full decoding; measured real-time factor was 0.679 over 795.51 assembly wall seconds. See the [handoff](gazebo_assembly_handoff.md) and [comparison results](../cais_spade_llm/monitor/recovery_gazebo_runs/ode_island_threads_comparison/20260924T203551Z/results.md) for the retained video and validation. Older attempt directories were removed on 2026-09-25; their condensed comparison outcomes remain.

## Previous planning record

Updated: 2026-09-23. The sections below preserve the earlier planning record. Raw run folders cited there were removed on 2026-09-25; current file-backed evidence is linked above.
Implementation status and restart instructions: [handoff](gazebo_assembly_handoff.md).

## Latest operator instruction

The user will run Gazebo from localhost and record it themselves. Do not start production or recording automatically. The operator currently selected **`RGOCG4-50_Round_4mm` to M1**, in `assembly_board-v1-round-4mm-m1.json`; preserve that selection. The agreed later recording pair remains **`KET4_Square_4mm` and `gear_small`**, in `assembly_board-v1-two-parts.json`.

The latest correction applies to **KMR and all four UR robots**: follow `main`'s two-arm behavior. Keep named configurations, program steps and resource-owned clearance waypoints saved. `compute_pick_targets` and `compute_place_targets` remain dynamic. Cartesian movements consume their fresh outputs in one `GetCartesianPath` request, as in `main` (`d960aee`), with no Python loop making an IK service call per sample. The controller still converts dynamic Cartesian targets to timed joint commands. The superseded full-motion recording is no longer a startup or execution dependency.

The user subsequently requested **no collision checking for the Gazebo production run**. Saved scene settings set `avoid_collisions: false` for the four UR controllers and KMR. Arm collision queries, KMR arm/gripper/payload transport sweeps and KMR configured-route footprint checks are bypassed with explicit evidence. Gazebo contact physics remains active; separate interactive Nav2 planning retains its existing checks. Hardware mode cannot use this bypass. Finite inputs, joint limits, continuous joints, observed starts/endpoints, custody and Stop remain enforced.

The user explicitly requested a temporary **simulation-only CCA bypass** to remove approval waits. The Setup checkbox `Bypass CCA (simulation only)` is enabled in the saved setup; Run shows its saved value. It skips PA plan approval and resource CCA permission while preserving resource checks, reservations, completion acknowledgement, controller limits and Stop. Physical mode rejects this setting. Reports record `diagnostic_cca_bypass` and `collision_checks_bypassed`; this run cannot establish CCA coordination or collision clearance. Turn the checkbox off to restore CCA for later acceptance.

Saved UR controller rate is reduced from 1000 Hz to 250 Hz after the user reported PC slowdown, heat and noise. This reduces controller work; it is not a measured simulation speedup. Revised routes, final placements, home returns and motion overlap still need live observation.

The earlier round-peg run `04eb734ba41546e58641201a753d38db` stopped before ProductAgent dispatched KMR pickup: `ur5e-3` confirmed its configured home, but a second feedback read rejected the completed command and stopped the ongoing negotiation. Home completion now uses the measured snapshot from that same successful primitive, bound to its robot, target and execution time. KMR now prepares its persistent worker before dispatch and exposes startup, readiness, task and failure logs. The live readiness probe passed for the selected round peg and M1 without commanding motion. Motion/delivery checks passed **285 tests**, and focused home/KMR checks passed **19 tests**. Those checks required a UI restart; the later operator run below completed pickup-to-M1 execution. The broader startup selection was interrupted and is not included in the passing count.

The earlier operator run `882091833a354d698aadd6423d0733f8` did not complete either part. KMR was never dispatched after stale discovery and an unanswered intake request. A timed-out intake now retries, bounded to three attempts, without requiring a resource revision change. `ur5e-4` reached `pre_insert_pose` but rejected the final 2.5 mm insertion before dispatch; the former wrapper hid the controller's exact rejection. Fresh-feedback waiting and error propagation are corrected. A read-only insertion request and controller checks pass from the stopped pose, with execution intercepted. This is preparation evidence only; successful release, home returns and productive motion overlap remain unverified. The handoff records the evidence and limits.

## September 23 performance implementation

Keep one ProductAgent and the existing ResourceAgents. Deterministic capability calculations now use at most two background workers on detached inputs; the owning inbox checks current revisions, requirements, geometry, reservations and Stop before accepting results. ProductAgent retains selection and dispatch, ResourceAgents retain execution, and CCA retains authority when its diagnostic bypass is off. The bypass dispatches approved work in the same scheduling cycle.

Acknowledgement and dispatch handling no longer construct a full report. Unchanged models and completed history are reused; one background writer preserves complete final reports. Timing evidence separates task send/receipt, execution, acknowledgement, calculation queue/time, report capture/construction/write time and message-loop delay. Tests prevent ROS controller construction and exercise acknowledgement/Stop within 0.5 seconds during deliberately slow calculation or report work, stale snapshots, reservations, cancellation and delayed/late CCA replies.

The saved next-launch viewer rate is **15 FPS**. `ode_island_threads` accepts **0, 2 or 4**, default **0**, and is forwarded through both launch files to ODE's `island_threads`. Preserve the 1 ms step, current UR 250 Hz and KMR 225 Hz, limits, contact physics, inventory and all five robots. Requested speed stays 1×; cameras and shadows stay off. Rebuild installed assets, then use a fresh operator-controlled UI/Gazebo launch. No live scene was restarted by this implementation.

Run `6cf3e23ab3574d7998f8d1291c91e291` established round-peg pickup, release at M1 and empty KMR Storage return. M1 machining and KMR return functions overlapped **36.08 wall seconds**; later assembly negotiation remained blocked. Pickup negotiation waited **50.33 seconds**, while KMR functions measured **0.155–0.179×** simulation speed. These are separate bottlenecks. Recorded report capture ranged **0.316–2.511 seconds**; an offline reconstruction of the same complete report now captures in **2.29 ms median**, preserving the report fields. This is offline evidence, not a measured live speedup.

The later 15 FPS comparison attempted 0 → 2 → 4 → 4 → 2 → 0 with fresh scenes and identical inventory. The first 0-thread run completed 11 parts at 0.679 real-time factor; both 2-thread runs and both 4-thread runs failed at `environment_7` on the KMR held-part transform check, and the final 0-thread run failed at `environment_74` after a Gazebo wrist-link observation timeout. The speed comparison is inconclusive; `ode_island_threads` remains 0. The separate 30 FPS/0 comparison was not run. See the [results](../cais_spade_llm/monitor/recovery_gazebo_runs/ode_island_threads_comparison/20260924T203551Z/results.md); raw failed-run directories were removed on 2026-09-25.

Verification: the combined capability/motion/delivery/configuration/setup/product run passed **654 cases**, with three new test-fixture assumptions corrected. The final configuration rerun passed **25 cases** (including all six absent/explicit ODE solver cases); the final reporting rerun passed **2 cases**. No focused failures remain after those reruns. Real-inbox tests observed at most **0.320 seconds** of message-loop delay across the completed stub-executor workloads; slow calculation/construction tests handled acknowledgement and Stop within 0.5 seconds. `poetry check`, compilation, CLI help and `git diff --check` passed. `make bootstrap-gazebo` finished **16 packages**, and installed launch/controller/world sources and the rebuilt viewer plugin were checked. See `performance_checks.json` for exact logs.

## Intended behavior

Assemble eight Storage pegs and three completed printer gears in visible, GPU-rendered Gazebo. Different resources should perform compatible work concurrently. Admit parts incrementally; do not launch eight complete searches together just because the order has eight pegs.

The user wants resources to behave like independent agents in Repast Simphony: each resource advances when its own work becomes ready. Preserve SPADE, PA–RA communication and CCA. This request does not authorize replacing them with a different framework.

Repast's scheduler executes scheduled actions against a simulation clock. Independent agent progress does not itself require one operating-system thread per agent. Here, the acceptance criterion is observed concurrent resource work, with independent completion processing. [Repast scheduling API](https://repast.github.io/docs/api/repast_simphony/repast/simphony/engine/schedule/Schedule.html)

## Arm motion: latest correction

- Use explicit Cartesian waypoints owned by each robot's primitives.
- Use `main`'s Cartesian service to follow the supplied waypoints and dynamic final target. Do not silently fall back to a joint-space route planner.
- Named configurations and clearance waypoints are saved; perception, pick/place targets and their resulting Cartesian moves remain dynamic. Whole-motion playback must not freeze those target calculations or require matching a pre-recorded endpoint.
- An unreachable, discontinuous or incomplete segment blocks before dispatch. Collision obstruction is not tested while the requested simulation bypass is enabled.
- Compose functions from robot-owned `compute_pick_targets`, `compute_place_targets`, motion, gripper, observation and custody primitives.
- Preserve resource-specific controllers and the existing task/recovery contracts. This motion correction does not remove adaptive PA–RA requirement/capability matching.

### KMR

- Keep the gripper downward during startup, pickup, loaded transport, placement, withdrawal, empty return and final Storage home. Required yaw changes may preserve the downward tool axis.
- Remove unnecessary folding and configuration changes from nominal task compositions.
- Use approach, Cartesian descent, grasp, short vertical lift, transfer waypoints, descent, release and vertical withdrawal.
- Turn around the arm mount through explicit clearance waypoints when needed. Avoid an unrelated whole-arm sweep.
- KMR follows saved base routes. Arm/gripper/carried-part sweep checks are bypassed for the requested run and reported as bypassed, never as validated.
- KMR returns **empty to Storage after its own handling cycle**, with an open gripper and an observed downward home. It must not wait for final assembly.
- All eight pegs remain arranged on the lower accessible shelf with the upper shelf/supports removed. Validate each pickup with neighbors present.

### UR robots

- ur5e-1 and ur5e-2 need explicit transfer waypoints around the base region that their direct machine-to-Conveyor segment cannot traverse.
- Validate ur5e-3's Buffer-to-board route and ur5e-4's printer-to-shaft routes, including home returns.
- Each robot returns home after its own handling cycle while downstream work continues.
- A Cartesian TCP path still requires joint rotation. Acceptance concerns the prescribed TCP route, tool orientation and continuous joint behavior; it does not mean frozen arm joints.

## Resource concurrency and CCA

- Use the existing PA–RA capability requests/replies and resource start validation. CCA approvals remain the normal path, with the explicitly requested temporary simulation bypass described above.
- PA requests unmet product/resource effects; RAs offer feasible transitions, prerequisites, effects, revisions and execution availability.
- Reuse valid discovered paths. Reject stale offers and renegotiate relevant changes.
- Keep requests independently correlated and keep accepting acknowledgements while another request or CCA approval is pending.
- Dispatch compatible ready work after each acknowledgement. Do not wait for unrelated active functions to finish.
- Incremental Storage intake and incremental gear intake may proceed alongside already admitted work.
- Prioritize active handling cycles, immediate home returns and downstream capacity release. Rank intake by offered machine workload and waiting age.
- CCA remains responsible for shared-workspace coordination when enabled. Do not add a competing application mutex for robot workspaces.
- Shared assembly-fixture registration belongs to ordered startup, once before production.
- A function holds its reservations until its complete event is acknowledged. Internal primitive completion does not release capacity or enable downstream handling.
- Blocking report serialization/file work must not stall unrelated agent communication. Measure before attributing all delays to SPADE or CCA.

## Routes and order

For the next two-part run select `assembly_board-v1-two-parts.json`. `assembly_board-v1-recovery-framework.json` remains the full eleven-part order; preserve the eight-peg and one-part orders.

| Parts | Required nominal route |
| --- | --- |
| `KET4_Square_4mm`, `KET8_Square_8mm`, `KET12_Square_12mm`, `KET16_Square_16mm` | Storage → KMR → M1 → ur5e-1 → Conveyor → Buffer For Machined parts → ur5e-3 → assembly_board-v1 |
| `RGOCG4-50_Round_4mm`, `RGOCG8-50_8mm`, `RGOCG12-50_12mm`, `RGOCG16-50_16mm` | Storage → KMR → M2 → ur5e-2 → Conveyor → Buffer For Machined parts → ur5e-3 → assembly_board-v1 |
| `gear_small`, `gear_medium`, `gear_large` | 3D Printing Station → ur5e-4 → configured board shafts |

These routes must result from capability matching with current configurations and connections. Preserve the separate one-part round order's explicit M1 constraint and its copied circle program; keep saved full-order M1 square and M2 circle defaults.

## Shared function boundaries

Re-read the other chat's implementation before editing shared adapters/catalogs. Keep these internal sequences:

| Function | Ordered primitives |
| --- | --- |
| `machine_part(part_name, process="trim", result)` | observe_workholding → verify_process_clearance → run_machining_clock → confirm_process_observation |
| `advance_conveyor(next_locations, delivered_part)` | observe_belt_residents → compute_shared_displacement → verify_transport_clearance → move_belt_residents → confirm_arrival |
| `advance_part(part_name, zone, downstream_zone)` | observe_zone_part → compute_downstream_motion → verify_transport_clearance → move_buffer_part → confirm_arrival |

Preserve configured machining time, shared belt displacement, observed buffer arrival, partial-motion evidence and failure handling. Machine completion acknowledges a trim result; it does not reshape CAD. Preserve `place_release` executing `place_insert`, Storage/Exit event participation and unavailable, display-only `print_part`.

## Gear simulation choice

The user explicitly accepts ignoring gear-to-gear tooth-alignment physics and automatically attaching seated gears to the assembly fixture. Retain pickup, transfer, insertion, release and pose observations. Label the result **simulated fixture attachment with tooth physics disabled**. Do not claim validated physical gear meshing. Keep other robot/fixture collision checks.

## Recording and acceptance

- Fresh inventory; preserve prior successful evidence and diagnostic logs.
- Visible NVIDIA-rendered Gazebo; capture its window at 15 fps, preserving aspect ratio and no wider than 1920 pixels, without audio.
- Record continuously before first production motion until all robot home returns, followed by ten seconds of the completed board close-up.
- Preserve wall-clock playback timing; record capture timestamps and missed frames.
- Retain a playable continuous H.264 MP4 and a clearly labeled 10× preview only after successful physical and video validation.
- Delete videos from failed, stopped, timed-out or incompletely validated attempts; preserve diagnostic logs.
- For the selected pair require both correct placements, observed release, clear transport/staging areas and all five robots empty at observed homes. KMR must be at Storage, open and downward-facing. Unselected inventory remains present. Eleven-part acceptance additionally requires all eleven correct targets occupied and remains outstanding.
- Report makespan, throughput, observed productive motion overlap, function overlap, planning/approval/waiting delays, simulation speed and recording performance separately.
- Keep `SystemBridge` unchanged.

This paragraph originally described the September 23 planning state. The later 11-part success and inconclusive ODE comparison are linked above; `recorded_two_part_validation` raw files were removed on 2026-09-25. Collision clearance and CCA coordination remain unproven by the simulation bypass.
