# KMR raised startup: recorded 11-part acceptance

Run `2a18a5d1ee544380b6ed6be402e43839` assembled all **11 components** at their configured targets:
four square pegs through M1, four round pegs through M2, and three preprinted gears.

## Videos

- [1× full recording](assembly.mp4): **21m13s**.
- [20× recording](assembly-20x.mp4): **1m04s**.
- [10× recording](assembly-10x.mp4): **2m07s**.

The accelerated copies use fixed playback speed. They include the complete recorded
workflow; they do not change Gazebo's simulation rate. All copies decode completely.

## Raised startup

KMR starts at the existing transport TCP `[-0.25, 0.50, 1.16]` m in its base frame,
with the gripper down. The seven joint targets are taken from the final trajectory
point of the first `park` step in run `82cb3858f1a94394a8f846a0b6831cf3`.
Fresh measured joints, TCP, and mandatory MoveIt clearance passed before production.
The first saved `pick_approach → move_to_named_pose("transport")` step sent **zero
arm trajectories** and reported `motion_required=false`.

Functions, primitives, state contracts, waypoints, Storage position, and pickup docks
are unchanged. Both controller initialization and new Gazebo launches use the same
saved `parked_arm_configuration`. This run began with a fresh installed launch.

## Validation

- 195 acknowledged transitions; no saved-step or model discrepancies.
- Program revision 4; protected files and complete saved programs unchanged.
- Zero KMR equipment contacts in 645,198 contact messages.
- Zero UR equipment or cross-UR contacts.
- 2,019 base output samples include mandatory map and transport clearance.
- 39 clearance waits held zero output before continuing.
- All three ur5e-4 pickup rotation checks passed.
- KMR and all four UR5e arms visible in startup, motion, final, and sampled video frames.

Settings: speed 1, ODE threads 0, unchanged timestep and robot motion limits,
diagnostic CCA bypass enabled, UR `avoid_collisions=false`, KMR collision checks active.
This is Gazebo simulation evidence.

## Timing comparison

Reference: accepted run `82cb3858f1a94394a8f846a0b6831cf3` (21m06s).
Its run data and timing summary were preserved before this test. The old video directory
was already absent before this run; surviving successful recordings were preserved.

| Measurement | Previous | Raised startup |
| --- | ---: | ---: |
| Full recording | 21m06s | 21m13s |
| First transport step | 17.598s | 0.013s |
| First transport arm trajectories | 1 | 0 |
| KMR base wall time | 9m18s | 9m19s |
| KMR base simulation time | 4m19s | 4m21s |
| KMR base simulation/wall rate | 0.465 | 0.467 |

The startup trajectory was removed. Total wall time also depends on workstation load
and Gazebo's measured simulation rate; total differences cannot be attributed entirely
to the startup change.

## Evidence

- [Startup joints, TCP, clearance, and first step](startup_pose_audit.json).
- [Acceptance and final placements](acceptance_audit.json), [complete run](run.json).
- [Primitive trace audit](trace_audit.json), [model transition audit](model_audit.json).
- [Clearance and contacts](clearance_audit.json), [render evidence](visual_evidence.json).
- [Timing comparison](speed_comparison.json), [source fingerprints](verification_sources.json).
- [Full and 10× validation](video_validation.json), [20× validation](assembly-20x-validation.json).
- [Saved scene](scene.json), [previous run data](previous_run_baseline/previous_run.json).

626 focused regression checks passed, alongside Python compile, CLI help, Poetry checks,
and `make bootstrap-gazebo`. Two pre-existing fixture-dependent tests were excluded:
`retired_lg_slippage_has_no_active_bindings` (unrelated active lg_slippage file) and
`prepared_recording_covers_configured_parts_and_observed_home_joints` (missing historical fixture).
