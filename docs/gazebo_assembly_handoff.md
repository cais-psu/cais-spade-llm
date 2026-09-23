# Gazebo assembly implementation handoff

Updated: 2026-09-23. Work stopped at the user's request to write Markdown before leaving for school.

## Read this first

**The revised implementation is unfinished. Do not report an eleven-part success or start an acceptance recording without completing validation.**

- Latest requirements: [Cartesian waypoints and resource concurrency](gazebo_cartesian_concurrency_plan.md).
- Code changes are present in the working tree and are uncommitted. Preserve them and unrelated edits; start with `git status --short` and root `AGENTS.md`.
- No `SystemBridge` changes. No Spec2Primitives edits. Do not modify physical controller profiles.
- No successful eleven-part recording exists.
- The prior successful eight-peg evidence remains intact under `cais_spade_llm/monitor/recovery_gazebo_runs/adaptive_eight_validation/`.
- Production was not running when the user requested this handoff. The latest Gazebo instance was a fresh preflight scene, with no production execution.

## What the evidence actually showed

### ur5e-1 sweep

The failed eleven-part trace reports a direct Cartesian fraction of approximately **0.34**, followed by the controller's joint-space free-space fallback. That explains the sweep. KMR also explicitly used `move_to_configuration`, `rotate_arm_base` and a parked configuration, so its extra posture changes were not caused only by the planner.

A read-only preflight found complete ur5e-1 Cartesian routes with two waypoints near `(-5.55, 1.55, 1.35)` and `(-5.55, 0.70, 1.35)`, followed by the Conveyor approach. That preflight used the previous Cartesian service and a representative virtual payload. **It does not validate the new direct-IK waypoint executor, all peg sizes or live motion.**

### Parallelism

For failed run `5f00bffecfb84829bc0a4a779da3e7cc`:

| Measurement | Result |
| --- | --- |
| Completed-event observation window | 300.40 wall seconds |
| Acknowledged events | 26 |
| Assembled before failure | `gear_small`, `gear_medium` |
| Simulation / wall time | 0.211 |
| Productive function overlap | 180.55 wall seconds |
| Sampled productive robot/base motion overlap | 41.98 wall seconds |
| Prepared dispatch → execution median / maximum | 2.63 / 8.43 seconds |
| Execution completion → PA acknowledgement median / maximum | 0.15 / 1.55 seconds |
| Negotiation median / maximum | 0.40 / 16.04 seconds |

Motion samples showed KMR overlapping ur5e-4 and ur5e-1 on different parts. This was a failed run, and incomplete functions were excluded from the completed-function intervals. Motion duration is sampled, not exact continuous timing.

The resources are independent SPADE agents but share a Python process/event loop. Confirmed code issues:

1. The PA awaited a CCA plan reply inside the main work loop and deferred other acknowledgements during that wait.
2. Gear completion evidence repeatedly embedded thousands of mesh vertices/triangles. Copies and synchronous report writes can block the shared loop.
3. Gazebo was slow independently of negotiation. Do not attribute all waiting time to agent scheduling.

CCA should continue coordinating shared workspaces; do not replace it with another robot-workspace mutex.

## Changes currently in the tree

### Direct Cartesian waypoints — NEW, live validation pending

- `cais_spade_llm/resources/robot/cartesian_waypoints.py` is a new shared conversion helper: explicit XYZ/quaternion interpolation, IK seeded from the previous sample, joint-branch rejection, bounded segment timing and ROS trajectory construction. It does not search for a route.
- `gazebo_pick_place_controller.py` uses this helper for Recovery Framework resources configured with `cartesian_motion.only`. It bypasses `GetCartesianPath` and joint-space planning for that mode. Legacy profiles retain their existing paths.
- Configured UR motion includes waypoint spacing and joint-step limits. ur5e-1/2 have candidate transit waypoints. Their configured home wrist angle was changed by π to match the downward pickup yaw and reduce rotation. **Validate startup and every home return before acceptance.**
- `kmr_gazebo.py` uses the same explicit-waypoint conversion in the configured mode. Downward endpoints and collision checks are required; joint-configuration commands are rejected in this mode.
- `kmr_motion.py` constructs configured downward clearance arcs about the arm mount.
- `kmr_tasks.py` replaces nominal clearance/turn/park joint commands with `move_to_pose` calls using computed transfer/withdrawal waypoints. Descent/lift remain `move_cartesian`.
- `kmr_primitives.py` accepts optional waypoint arrays on `move_to_pose`.
- KMR empty return holds its observed, sweep-validated posture and goes to the Cartesian Storage home. Its redundant joint solution may differ from the saved seed; final TCP/downward/open observations remain required.
- `ros2/cais_lab_robotics/scripts/kmr_base_controller.py` accepts this empty posture only when `use_sim_time=True`, the waypoint simulation configuration is enabled, launch/scene identity match, a sweep has been validated and observed joints match. Existing loaded-custody checks remain.

Automatic approval review initially rejected a broader base-controller change because of possible shared/physical impact. Read-only inspection established the Recovery Gazebo launch boundary; the narrower change with explicit simulation/configuration gates was accepted. There is no outstanding user approval request.

### Concurrent PA loop — NEW, regression verification incomplete

- `environment_runtime.py` keeps a correlated pending CCA approval instead of waiting inside `receive_from`.
- Its ordinary inbox continues processing resource acknowledgements and discoveries while approval is pending.
- Approved tasks still require revision checks, RA start validation and resource CCA permission before execution.
- Added timestamps for plan approval requests/results, RA receipt and resource safety request/decision. Use these to separate approval latency from execution and simulation time.
- Existing incremental Storage intake and gear admission remain.
- Synchronous report persistence is still present. Mesh compaction reduces payload cost; further reporting changes require measurement and ordering/Stop tests.

### Collision evidence and earlier eleven-part work

- `geometry.py` retains mesh source/hash/scale/center provenance.
- `part_collision.py` records compact collision evidence with mesh counts and observed transforms, while collision checking uses full meshes.
- Gear support clearance was corrected to follow model +Z rather than an upside-down mesh's local +Z. A read-only live check of the previously failing large-gear approach passed before the waypoint rewrite.
- Gear world collision meshes preserve bores. Gear-to-gear physics contact is disabled for the accepted simplified attachment mode; correctly seated gears retain their fixture attachment.
- Gear target floor is `1.0289916 m`; target origin is `1.0389916 m`.
- `_pick_geometry` supplies configured CAD grasp widths.
- The full eleven-part order is selected; demos remain.
- `gazebo_recording.py` implements owned capture, success-only promotion, failure deletion, complete decode verification and labeled 10× preview. Existing recorder tests previously passed.

## Latest verification results

These are actual results; do not treat the failed tests as passes.

| Check | Result |
| --- | --- |
| `test_simulation_timing.py` + `test_recovery_delivery.py` | **205 passed, 18 skipped, 2 failed** |
| Focused PA/CCA test selection | **1 passed, 1 failed, 54 deselected** |
| `poetry check` | Passed, with existing Poetry metadata deprecation warnings |
| `poetry run python -m compileall -q cais_spade_llm ros2` | Passed for runtime edits at that checkpoint |
| `git diff --check` | Passed at the checkpoint before the final test additions |
| Live revised eleven-part production | **Not run** |

Current failures:

1. `test_scene_refresh_removes_old_payload_boxes_and_preserves_observed_parts[False/True]`: the test's `collision_boxes` stub does not accept the new `exact_models` keyword. Update the fixture to exercise and verify the exact-model argument; retain its original scene-refresh assertions.
2. `test_pa_commits_running_ack_while_another_CCA_approval_is_pending`: its setup calls `bind_executor` without required `validate_start`. Fix the test setup, then run the behavioral assertion. It has **not yet proved** the nonblocking-CCA behavior.

Logs:

- `/tmp/cais-waypoint-focused-tests.log`
- `/tmp/cais-concurrent-ack-tests.log`
- `/tmp/cais-waypoint-bootstrap.log`

`make bootstrap-gazebo` was started after the runtime/controller edits. Its final result and owned-process shutdown are recorded in the closing status below.

## Resume in this order

1. Read current diffs and the other chat's shared primitive/catalog changes. Preserve `SystemBridge`, model event names, CCA authority and partial-failure evidence.
2. Fix the two test-fixture issues above and rerun the focused suites. Add a regression for the Gazebo-only empty-posture gate: hardware mode, stale launch/scene, unvalidated sweep, wrong joints and closed gripper must not use the new empty-return path.
3. Review the new waypoint converter before execution: finite configuration validation, endpoint timing, interpolated acceleration limits, IK continuity, cancellation, complete collision checking and no partial dispatch. Ensure no planner service is called in configured nominal arm execution.
4. Validate KMR arcs and the proposed transport TCP `[-0.25, 0.50, 1.10]` in the base frame with full geometry. These are **candidate values, not accepted live evidence**. Validate all eight picks, both machine docks, placement, withdrawal, empty routes and downward Storage home. Confirm primitive catalog signatures reflect the new optional waypoints.
5. Validate all UR pickup/transfer/place/home routes with held and neighboring parts. ur5e-3 currently has no additional configured transit route; the previous eight-peg log contained fallback uses there. Resolve those explicitly before production. Verify exact configured UR home joints are reachable through the waypoint route.
6. Run the real-inbox eight/eleven-part tests, stale-offer/availability tests and delayed-CCA acknowledgement test. Test Stop during approval and motion. Check pending-task cleanup on approval timeout/cancellation, including the new approval state.
7. Measure event-loop/reporting delay and approval phases. Dispatch all compatible ready work; preserve per-resource order and CCA decisions. Do not replace incremental admission with eight simultaneous full searches.
8. Run focused tests, `poetry check`, compile checks and `make bootstrap-gazebo` again after any ROS edits. Current runtime changes have not passed every acceptance gate.
9. Copy the final sources/configuration to the live harness; reset to fresh inventory. Validate visible GPU rendering before recording. The current harness copy predates the newest waypoint/concurrency edits.
10. Run all eleven parts. Require measured productive overlap, all correct insertions/releases, cleared areas and all five homes. Retain the video and generate the 10× preview only after complete validation.

## Evidence and harness locations

- Prior successful eight-peg run: `cais_spade_llm/monitor/recovery_gazebo_runs/adaptive_eight_validation/`.
- Live harness: `/tmp/cais-eleven-live-20260923/`.
- Failed attempts: `launch-failed-1/`, `startup-failed-2/`, `production-failed-3/`, `production-failed-4/`, `production-failed-5/`, `startup-failed-6/`, `production-failed-7/` under that harness.
- Latest failed-run measurements: `production-failed-7/timing_summary.json` and `production-failed-7/motion_overlap.json`.
- Read-only old route probe: `cartesian_routes.log`. It is historical service-based evidence, not proof of the new executor.
- Runner: `live_eleven_run.py`; helper scripts validate trace, homes, physical final state, timing, motion overlap and video.
- Capture size was lowered to 1280×502 to reduce recording overhead; this fits the accepted aspect-ratio/maximum-width requirements. It has not yet been benchmarked in a successful full run.
- Final camera height was changed to 2.25 m to fit the board close-up.
- ROS domain: 47. DDS profile: `/tmp/cais-eleven-live-20260923/dds_udp.xml`.
- Intended final evidence directory: `cais_spade_llm/monitor/recovery_gazebo_runs/recorded_eleven_validation/`; no successful result has been promoted there.

Temporary directories may be lost after reboot. Preserve required diagnostic evidence before resetting or cleaning them.

## Launch cautions

- Do not start production with stale harness source/configuration or stale inventory.
- The existing, unchanged launch surface uses `pkill -9 -f gazebo`. Run launch commands separately from tests/builds; do not put the literal `gazebo` in the parent runner command line or chain unrelated archive/test commands into it. Earlier attempts killed their own parent shell and a matching test process.
- Source `/opt/ros/humble/setup.bash` and `/home/jongh/ros2_ws/install/setup.bash`; set `ROS_DOMAIN_ID=47` and the DDS profile above.
- The harness needs both its root and its `cais_spade_llm` directory in `PYTHONPATH`.
- Do not copy or print credentials. The harness uses local dummy API credentials for nominal matching.
- Stop only launch/recorder/worker processes whose ownership is verified. Preserve previous successful evidence; failed videos must remain deleted.
- Do not launch new production merely to answer a status question. The user's latest instruction was to write this handoff and stop implementation work.

## Closing status

- `make bootstrap-gazebo`: **passed, 16 packages, 1 minute 24 seconds**.
- The owned preflight launch (PID 774888, verified launch command and process group) was stopped cleanly. No new production run was started.
- Failed-attempt video check found no remaining attempt MP4 files.
- Final `git diff --check` passed. Source changes and both handoff documents remain uncommitted.
- Also review legacy planner-service readiness requirements: nominal configured execution now bypasses those planner services, but some startup/probe gates still name them.
- Review final KMR validation scripts for assumptions about exact seed joint values. The requested Cartesian home must be verified by observed pose, downward orientation, open gripper and clearance; do not silently accept an unobserved home.
