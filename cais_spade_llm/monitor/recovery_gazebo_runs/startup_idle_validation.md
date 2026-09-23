# Gazebo startup and idle-time validation — 2026-09-22

The explicit Start path now retains the probed KMR worker and transfers it once to the KMR resource adapter. Full manufacturing startup prepares one controller for each selected `ur5e-1` through `ur5e-4` resource and transfers those controllers through the existing `prewarmed_controller` interface. Worker and controller objects remain outside serialized preparation snapshots. Both transfers are bound to the saved source fingerprints and launch identity, reject stale or changed preparation, and clean up unclaimed ownership after failure, Stop, or cancellation.

MoveIt no longer starts behind a fixed four-second timer. It starts from the UR controller-spawner completion event while KMR and Nav2 finish loading. The final bridge readiness check remains authoritative. KMR requests share one session-long ROS executor. ROS clients, TF state, subscriptions, and controller connections remain alive across the three delivery functions.

KMR and UR5e background preparation use one non-blocking scene-wide planning permit. Prepared paths never dispatch motion or mutate the planning scene, and every consumed path is checked against current joints, attachments, targets, and every collision state. Unfinished, rejected, or failed preparation falls back to the synchronous path. Fresh stable endpoint observations can complete an unchanged gripper or named-position primitive without sending a redundant command; the primitive and its observation evidence remain in the declared composition.

## Background-planning decision

Two complete full-scene trials with KMR placement planning overlapped with base travel reduced placement planning after arrival from a synchronous median of 7.61 seconds to 0.26 seconds. Start-to-placement-motion fell from 6.33 seconds in the final synchronous run to 0.57–0.78 seconds in the overlap trials. The extra MoveIt load slowed Gazebo on this 8 GB host: median three-action worker time was 71.30 seconds with overlap versus 59.42 seconds in the three synchronous comparison runs. Because total workload time regressed, `background_preparation_enabled` is saved as `false`. The validated preparation path remains available for a machine where both handoff latency and total time improve.

| Configuration | Runs | Median three-action worker time | Placement planning after docking |
| --- | ---: | ---: | ---: |
| Synchronous selected configuration | 3 | 59.42 s | 7.61 s median |
| Background placement preparation | 2 | 71.30 s | 0.26 s median |

The final rebuilt synchronous run started the full visible scene in 33.69 seconds and completed `pick_part`, `move_to_resource`, and `place_release` in 26.41, 10.48, and 22.90 seconds. Its controller-to-controller request gaps were 0.29 and 0.34 seconds. All three acknowledgements passed, M1 was `loaded` with `KET4_Square_4mm`, Storage inventory was `false`, and KMR was `idle`, empty, and at M1.

Three earlier fresh visible-scene runs of the same selected synchronous motion configuration also completed with the required final states. Two additional repetitions after the final rebuild could not complete because this host exhausted available memory under repeated visible full-scene launches: one KMR parameter request missed its startup deadline and one worker and the observation process were killed with `SIGKILL`. These failures were not counted as successful repetitions. Worker failure text now retains the process return code and signal when no ROS evidence can be written.

## Timing and contracts

Evidence records the original Start click, agent readiness, first accepted controller motion, function dispatch/acknowledgement, each primitive, planning wall time, trajectory duration, simulation time, wall time, and measured real-time factor. Overlapping planning is reported on the operation that performs it and is not added again to total elapsed time. The KMR `pick_part` → `move_to_resource` → `place_release` function order and all primitive decompositions are unchanged. CCA checks, final bridge readiness, fresh-scene rules, controller limits, safety guards, custody, collision validation, and acknowledgement protocols remain in force.

## Verification

- `test_recovery_delivery.py`: 76 passed.
- `test_environment_capabilities.py`: 23 passed in the full run; the final ownership and Stop-during-preparation checks passed in a focused rerun. A later repeat stalled in this sandbox after three tests and was interrupted.
- `test_recovery_framework_gazebo.py`: 121 passed, 24 skipped because ROS-only fixtures were unavailable to that test process.
- `test_dual_robot_rviz_startup.py` and `test_simulation_timing.py`: 77 passed, 2 skipped.
- `test_project_pages.py` completed all 17 test cases, then its NiceGUI process did not exit during teardown in this sandbox and was interrupted without a pytest summary.
- `poetry check`, Python compilation, CLI help, and `git diff --check`: passed. Poetry emitted its existing metadata deprecation warnings.
- `make bootstrap-gazebo`: 16 packages built successfully on the retry after the linker was killed once by transient memory pressure.
- One post-build fresh visible-scene delivery completed all three actions and final-state assertions. Existing four-arm/gripper concurrent workload evidence remains in `performance_validation.md`; another live four-arm series was not started while repeated full-scene launches were triggering host memory kills.

No `SystemBridge` public API or shared ProductAgent, ResourceAgent, CCA, or RobotAgent authority changed. The saved product order remains `assembly_board-v1-kmr-storage-m1.json`, so the next explicit Start selects Storage → M1.
