# KMR base speed validation — 2026-09-22

The selected Storage → M1 profile retains the 20 Hz base control loop, 1.5 m/s speed limit, 1.5 m/s² acceleration limit, direct 1.390 m route, carrying posture, geometry, and physics. Arm velocity and acceleration scaling remain 1.0. The KMR controller manager remains at 225 Hz.

The command watchdog is now 0.10 s instead of 0.25 s; the odometry watchdog is 0.40 s instead of 0.50 s. These shorter wall-time watchdogs reduce the reserved braking delay from 0.85 s to 0.60 s at the existing requested simulation rate. The docking position gain is configurable and selected at 4.0/s, subject to the existing braking and approach speed caps. Unsafe measured stopping envelopes, stale feedback, custody loss, and arm disagreement still stop movement.

## Full-scene measurements

All rows below completed pickup, travel, and placement in a fresh scene with visible Gazebo, four UR5e controllers, Nav2, cameras off, RViz off, and the existing physics settings. Base timing covers the acknowledged `/KMR/dock` action; startup and arm motion are excluded. Speed is a sampled peak from the controller's existing status stream.

| Profile | Base wall s | Base simulation s | Measured rate during base move | Sampled peak speed, m/s |
| --- | ---: | ---: | ---: | ---: |
| Original 20 Hz / original watchdogs | 8.48 | 3.9 | 0.460× | 0.930 |
| 50 Hz / shorter watchdogs / gain 4.0 | 9.79 | 2.9 | 0.296× | 1.089 |
| Selected 20 Hz, repetition 1 | 5.86 | 2.4 | 0.410× | 1.140 |
| Selected 20 Hz, repetition 2 | 8.81 | 2.8 | 0.318× | 1.154 |
| Selected 20 Hz, repetition 3 | 7.77 | 3.2 | 0.412× | 1.080 |

Selected median: **2.8 simulation seconds instead of 3.9 (28% shorter)** and **7.77 wall seconds instead of 8.48 (8% shorter)**. Wall time varied from 5.86 to 8.81 seconds as Gazebo throughput varied. These are comparisons with one clean baseline run, not a guaranteed throughput improvement. Increasing control frequency to 50 Hz did not improve elapsed time on this machine. More aggressive watchdog candidates failed delayed-feedback checks and were excluded. The simulation clock was not accelerated.

## Delivery and handoff checks

The selected profile completed three fresh-scene deliveries through the persistent Gazebo worker and `NominalProductContext.acknowledge_gazebo`, retaining `pick_part` → `move_to_resource` → `place_release` and 16/5/11 primitive results. Every repetition ended with M1 `loaded` with `KET4_Square_4mm`, Storage inventory `false`, and KMR `idle`, empty, at M1.

| Repetition | Logical run ID | Startup wall s | Complete worker sequence wall s |
| --- | --- | ---: | ---: |
| 1 | `19bb5ae3bfbc49908e83ac8f6b48d75b` | 28.72 | 59.42 |
| 2 | `5694b5d137cb42c88eaf1db569b8e20b` | 23.35 | 59.67 |
| 3 | `0d9e7a25b2da4a37a8ce7aedc949eb98` | 40.04 | 50.23 |

An initial faster trial exposed custody-topic/action ordering: the docking goal could arrive before the controller received custody. The worker now waits for an acknowledgement with a unique custody ID and the guarded `arm_parked` value. Stale IDs and unconfirmed posture cannot dispatch docking. No primitive was removed or reordered. Docking evidence now includes copied velocity/status samples and the custody acknowledgement; runtime authorities and `SystemBridge` are unchanged.

## Verification and scope

- Affected ROS-enabled suites: 216 tests passed in the full run; the remaining collision fixture was moved 5 cm closer to preserve an unsafe stopping envelope under the shorter watchdog, and its focused rerun passed. Total affected coverage: 217 passing cases, including delayed-feedback braking, custody, cancellation, failure evidence, and composition checks.
- `poetry check` passed with existing metadata deprecation warnings; Python compilation passed.
- `make bootstrap-gazebo` passed: 16 packages built. Installed controller content matches the source.
- Three selected-profile fresh-scene deliveries passed. This was physical Gazebo execution with context acknowledgements, not a new SPADE/CCA end-to-end manufacturing-order validation. No hardware validation was performed.

Temporary profiling evidence is under `/tmp/cais_kmr_tuning_qjj0jcm1`; no generated JSON reports were added to the repository. The previous completed UI report (`d09ec9dee818425ba300d06c98417634`) was preserved. Owned benchmark scenes were stopped. The saved configuration applies at the next explicit fresh Gazebo start.
