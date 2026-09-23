# Whole manufacturing simulation performance validation

Validated on 2026-09-22 with the complete manufacturing scene, visible Gazebo, all four UR5e arms/grippers, KMR, Nav2, fixtures, and dynamic parts. The 0.001-second physics timestep, solver, collision geometry, resource locations, and joint limits are unchanged. These measurements describe this computer; the requested clock rate is a ceiling, not a promised throughput.

## Selected defaults

1× requested speed; UR controllers 1,000 Hz; KMR controller manager 225 Hz; Gazebo viewer capped at 30 FPS; RViz off; unused depth cameras off; dynamic shadows off. UR status publication is 50 Hz and visualization TF is 30 Hz. Functional control, collision monitoring, navigation, and ground-truth observation services remain available.

Run → Simulation speed offers 1×, 2×, 5×, and 10×. Stop System, then Stop Simulation before saving changed settings. The next explicit Start uses a fresh scene. The display refreshes requested/observed speed and applied settings every two seconds. Open RViz and Close RViz control the viewer and its interactive-marker services independently.

## Full-scene benchmarks

Each controller trial commands all four arms and all four grippers simultaneously, checks the combined motion at 21 collision states, verifies measured endpoints and successful results, cancels all eight controllers, and returns them to their initial positions. Warm trials repeat the same two-second simulated motion four times after startup.

| UR rate | Median warm workload, wall s | Idle real-time factor | Maximum arm tracking error, rad |
| ---: | ---: | ---: | ---: |
| 1000 Hz | 5.63 | 0.370× | 0.000141 |
| 500 Hz | 5.68 | 0.369× | 0.000282 |
| 250 Hz | 5.89 | 0.370× | 0.000563 |

| Requested speed at 1,000 Hz | Median warm workload, wall s | Idle measured rate |
| ---: | ---: | ---: |
| 1× | 5.63 | 0.370× |
| 2× | 6.05 | 0.360× |
| 5× | 6.32 | 0.369× |
| 10× | 6.22 | 0.350× |

1×/1,000 Hz was fastest in this workload. Lowering controller rates or requesting 2×–10× did not increase throughput. Baseline process samples showed gzserver at roughly 155% CPU, gzclient at 75%, and MoveIt at 27% (100% represents one CPU core). Physics and embedded controller work share gzserver; these CPU figures do not separate their costs. The viewer cap was confirmed loaded; no independent physics speedup is attributed to that cap. DDS/controller discovery and model loading are included only in the startup measurements below.

## Existing assembly executor

All four resource-owned shared controllers passed simultaneous wrist motion, close/open gripper, simulation-time delay, and the existing move_home function through its declared move_to_named_pose primitive. Fresh measured targets were checked after controller acknowledgements. Assembly decomposition metadata was unchanged. The full concurrent sweep was collision checked; no new manufacturing bindings were installed.

| Resource | Primitive sequence + move_home, wall s | move_home, wall s | move_home trajectory, simulation s |
| --- | ---: | ---: | ---: |
| ur5e-1 | 13.54 | 2.48 | 0.50 |
| ur5e-2 | 13.86 | 2.48 | 0.50 |
| ur5e-3 | 13.84 | 2.50 | 0.50 |
| ur5e-4 | 13.81 | 2.44 | 0.50 |

These per-resource sequences ran concurrently. There is no recorded historical per-resource assembly baseline, so an original-to-current UR speedup is not claimed. The controller-rate table compares equivalent current workloads. /detect_all returned all eight exact NIST part identifiers with cameras disabled; service calls took about 0.1 seconds. RViz and its marker process opened and stopped independently while the same Gazebo and MoveIt scene remained alive.

## Three fresh-scene delivery repetitions

Each repetition completed pick_part → move_to_resource → place_release with three authenticated acknowledgements and 32 recorded primitive results (16/5/11). All three used identical final source fingerprints and distinct launch IDs. Each ended with M1 loaded with KET4_Square_4mm, Storage.inventory.KET4_Square_4mm=false, and KMR idle, empty, at M1. The arm stayed in its validated low carrying posture during the direct 1.390 m route; all placement plans selected the lowest-joint turn.

| Run ID | Preparation, wall s | Pickup, wall s | Base, wall s | Place, wall s | Total including startup, wall s |
| --- | ---: | ---: | ---: | ---: | ---: |
| 82f5dafb36474fc5b908f49315c00744 | 31.12 | 26.78 | 11.63 | 21.94 | 102.28 |
| f587ba1404dc497f99dbf337d2305e5d | 32.83 | 24.93 | 12.00 | 21.35 | 102.66 |
| 05211b14930c4c0c92d263c3687d6d3b | 37.87 | 24.83 | 10.04 | 20.06 | 104.64 |

Median worker execution is 58.29 seconds for the supported delivery, including 10.81 seconds of planning. The prior compact validation recorded median base/place worker times of 17.84/27.63 seconds; current medians are 11.63/21.35 seconds (1.53×/1.29× faster). Pickup worker median is 24.93 seconds. Pickup arm trajectory duration remains 4.22 simulation seconds, versus the original 40-second trajectory baseline (9.49×). The original 221-second pickup figure and previous 39.12-second figure measured dispatch intervals, so they are not used as exact speedup ratios against worker-only timing.

Duplicate Start retained the same agents in all three repetitions. Stop ended the agents and owned workers while retaining the scene. The first repetition additionally rejected a repeat delivery without a fresh scene. The harness stopped its owned scene after each repetition. Timings separate simulation progress, wall time, planning, trajectory duration, and measured real-time factor in the latest report; individual primitive parameters/results/observations are retained.

## Reports and scope

Only latest/run.json is replaced for each reporting directory; logical run/task IDs remain unique and stale owners cannot overwrite a newer run. Archive this run explicitly makes an immutable copy of the displayed run. With writers stopped, the prior latest reports were copied and parsed before removing 64 superseded generated JSON files and empty directories. Primitive/function definitions, catalogs, configuration, orders, safety/collision artifacts, test fixtures, scripts, logs, and the earlier compact validation summary were retained.

This validates controllers, existing assembly function execution, and the supported Storage → M1 delivery. It does not establish execution of a complete manufacturing order: the unfinished machine, conveyor, buffer, and assembly handoff bindings and automatic recovery remain outside this change.

## Checks

- Three fresh-scene acknowledged deliveries: passed.
- All four arms and grippers, simultaneous controller load, cancellation, observations, shared assembly executor, and viewer independence: passed.
- [Final affected suites](performance_tests.log): 461 passed. [Environmental reporting/acknowledgement checks](performance_environment_tests.log): five passed. Coverage includes clock acceleration, paused/reset clocks, stale feedback, braking, collision stops, partial primitive evidence, settings/fingerprints, and report retention.
- poetry check, Python compilation, CLI help, and focused unused/undefined-name lint: passed; Poetry retains its existing metadata deprecation warnings.
- [make bootstrap-gazebo](performance_bootstrap.log): 16 packages built successfully.
- A broader Case 3 suite reached eight failures because its retired initialization/failure_scenarios/lg_slippage.json fixture is absent. A combined environmental discovery run stopped progressing on test_added_registered_part_needs_no_resource_definition_change; its affected reporting, acknowledgement, expiry, and UI revision tests passed independently. Shared agent authorities were left unchanged.

[Latest complete delivery evidence](latest/run.json) · [Earlier compact validation](20260922T050909_b8572a70/validation.md)
