# KMR Storage → M1 validation

Three fresh Gazebo scenes completed `pick_part` → `move_to_resource` → `place_release`, with three authenticated acknowledgements per run. Each scene used the same final motion/configuration fingerprints and a distinct launch ID. Gazebo and RViz remained enabled; unused camera streams were disabled. Physics settings and simulation clock speed were unchanged.

Every final valuation was:

- `M1.resource_state = loaded`, `M1.part_name = KET4_Square_4mm`
- `Storage.inventory.KET4_Square_4mm = false`
- `KMR.resource_state = idle`, `KMR.held_part = null`, `KMR.resource_location = M1`

Each run also verified duplicate Start retained the same agent instances, Stop ended the agents while retaining the scene, and repeating delivery without an explicit fresh scene was rejected. The validation harness then stopped its owned scene. The existing operator UI/debugger was left running and needs a restart to load changed UI modules.

## Measured timings

| Run / complete evidence | Pickup arm trajectories (s) | Pickup dispatch-to-next-dispatch wall (s) | Base worker wall (s) | Place worker wall (s) | Mean measured real-time factor | Pre-motion startup retries |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `20260922T050322_148ebc65` (historical generated report removed after latest-only migration) | 4.21 | 39.12 | 17.84 | 32.97 | 0.248 | 0 |
| `20260922T050613_efb1ff3a` (historical generated report removed after latest-only migration) | 4.22 | 43.96 | 17.89 | 25.54 | 0.268 | 0 |
| `20260922T050909_b8572a70` (historical generated report removed after latest-only migration) | 4.22 | 35.89 | 17.19 | 27.63 | 0.255 | 1 |

The `recorded baseline` (historical generated report removed after latest-only migration) contains 40.00 seconds of pickup arm trajectories; its product-action dispatch interval was 221.011 seconds wall time. The final median is **4.22 seconds of arm trajectories (9.49× faster)** and **39.12 seconds pickup wall time (5.65× faster)**. The roughly 10× target was approached for trajectory duration; measured pickup wall-time improvement is the factor reported above. Gazebo remained below real time, averaging approximately 0.26×; the simulation clock was not accelerated. Cold scene startup is excluded from both pickup comparisons and is recorded separately in each `control_verification.json`.

The arm held the configured low carrying posture during the direct 1.390 m constant-heading base route. The base speed cap reserves the configured 0.5-second odometry window, 0.25-second command window, and two 0.05-second control periods before the measured stopping guard. The prior configured detour was 3.31 m. Lowest-joint-turn selections were true, true, true. Plans stayed inside bounded joint position, velocity, and acceleration limits, with complete Cartesian-path checks and arm/gripper/attached-part collision checks along transport.

Wall time, simulation time, planning time, trajectory duration, measured real-time factor, copied trajectory targets, custody observations, docking feedback, and stopping-check context are in the individual reports and `machine-readable summary` (historical generated report removed after latest-only migration). Superseded generated reports were subsequently removed at the operator’s request; these compact measurements, scripts, logs, and collision-validation artifacts are retained.

## Verification

- [Affected suites](affected_tests.log): 324 passed with the final code, covering delivery, recovery setup, Gazebo, resource status, agents, and pages, including delayed-feedback braking and the original abort reason.
- [Final delivery suite](final_delivery_tests.log): 58 passed after that final worker change, including identical idempotent scene-update retries, bounded failure, endpoint feedback synchronization, collision rejection, cancellation, acknowledgement handling, and no repeat of motion/effect requests.
- `poetry check`: passed with existing metadata deprecation warnings.
- `poetry run python -m compileall -q cais_spade_llm ros2`: passed.
- `poetry run python -m cais_spade_llm.ui_main --help`: passed.
- [`make bootstrap-gazebo`](bootstrap_gazebo.log): 16 packages built successfully.
- Camera option tests verify default removal of the three unused sensors, preserved geometry/physics, explicit re-enabling, and forwarding through the launch wrapper.
- Hard-stop tests cover real stopping-envelope collisions, stale feedback, custody loss, posture mismatch, and original abort-message propagation.

These results validate the requested simulation delivery. Grasp custody uses the existing acknowledged Gazebo fixed-joint attachment; physical finger-contact dynamics were not part of this task.
