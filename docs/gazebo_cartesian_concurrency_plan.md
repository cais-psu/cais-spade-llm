# Gazebo Cartesian waypoints and concurrent resource agents

Updated: 2026-09-23. These are the user's accepted requirements and latest corrections.
Implementation status and restart instructions: [handoff](gazebo_assembly_handoff.md).

## Intended behavior

Assemble eight Storage pegs and three completed printer gears in visible, GPU-rendered Gazebo. Different resources should perform compatible work concurrently. Admit parts incrementally; do not launch eight complete searches together just because the order has eight pegs.

The user wants resources to behave like independent agents in Repast Simphony: each resource advances when its own work becomes ready. Preserve SPADE, PA–RA communication and CCA. This request does not authorize replacing them with a different framework.

Repast's scheduler executes scheduled actions against a simulation clock. Independent agent progress does not itself require one operating-system thread per agent. Here, the acceptance criterion is observed concurrent resource work, with independent completion processing. [Repast scheduling API](https://repast.github.io/docs/api/repast_simphony/repast/simphony/engine/schedule/Schedule.html)

## Arm motion: latest correction

- Use explicit Cartesian waypoints owned by each robot's primitives.
- Do not use a Cartesian motion planner or joint-space route planner to choose the nominal arm route. Do not silently fall back to a joint-space plan.
- IK converts the supplied XYZ/orientation samples into joint commands. Seed each sample from the preceding solution, enforce joint continuity and configured limits, and validate collision clearance before execution.
- An unreachable or obstructed segment blocks that function with evidence. It must not dispatch a partial path or report completion.
- Compose functions from robot-owned `compute_pick_targets`, `compute_place_targets`, motion, gripper, observation and custody primitives.
- Preserve resource-specific controllers and the existing task/recovery contracts. This motion correction does not remove adaptive PA–RA requirement/capability matching.

### KMR

- Keep the gripper downward during startup, pickup, loaded transport, placement, withdrawal, empty return and final Storage home. Required yaw changes may preserve the downward tool axis.
- Remove unnecessary folding and configuration changes from nominal task compositions.
- Use approach, Cartesian descent, grasp, short vertical lift, transfer waypoints, descent, release and vertical withdrawal.
- Turn around the arm mount through explicit clearance waypoints when needed. Avoid an unrelated whole-arm sweep.
- Validate the complete arm, gripper and carried part along base routes. Base navigation remains a separate resource-owned operation.
- KMR returns **empty to Storage after its own handling cycle**, with an open gripper and an observed downward home. It must not wait for final assembly.
- All eight pegs remain arranged on the lower accessible shelf with the upper shelf/supports removed. Validate each pickup with neighbors present.

### UR robots

- ur5e-1 and ur5e-2 need explicit transfer waypoints around the base region that their direct machine-to-Conveyor segment cannot traverse.
- Validate ur5e-3's Buffer-to-board route and ur5e-4's printer-to-shaft routes, including home returns.
- Each robot returns home after its own handling cycle while downstream work continues.
- A Cartesian TCP path still requires joint rotation. Acceptance concerns the prescribed TCP route, tool orientation and continuous joint behavior; it does not mean frozen arm joints.

## Resource concurrency and CCA

- Use the existing PA–RA capability requests/replies, resource start validation and CCA approvals.
- PA requests unmet product/resource effects; RAs offer feasible transitions, prerequisites, effects, revisions and execution availability.
- Reuse valid discovered paths. Reject stale offers and renegotiate relevant changes.
- Keep requests independently correlated and keep accepting acknowledgements while another request or CCA approval is pending.
- Dispatch compatible ready work after each acknowledgement. Do not wait for unrelated active functions to finish.
- Incremental Storage intake and incremental gear intake may proceed alongside already admitted work.
- Prioritize active handling cycles, immediate home returns and downstream capacity release. Rank intake by offered machine workload and waiting age.
- CCA remains responsible for shared-workspace coordination. Do not add a competing application mutex for robot workspaces.
- Shared assembly-fixture registration belongs to ordered startup, once before production.
- A function holds its reservations until its complete event is acknowledged. Internal primitive completion does not release capacity or enable downstream handling.
- Blocking report serialization/file work must not stall unrelated agent communication. Measure before attributing all delays to SPADE or CCA.

## Routes and order

Select `assembly_board-v1-recovery-framework.json`; preserve the eight-peg and one-part orders.

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
- Require all eleven correct targets occupied, previous parts still seated, transport/staging areas clear and all five robots empty at home. KMR must be at Storage, open and downward-facing.
- Report makespan, throughput, observed productive motion overlap, function overlap, planning/approval/waiting delays, simulation speed and recording performance separately.
- Keep `SystemBridge` unchanged.
