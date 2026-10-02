# Simulation failure injections

All four scenarios use one once-per-run latch. Select a scenario in Setup,
choose Simulation, save, and use Start Simulation followed by Start System in
Run. Keep CCA enabled. The Run card shows the selected scenario, resource,
checkpoint readiness, Arm, Disarm, Trigger, injection evidence, and marker
result. An automatic trigger runs once when the observed checkpoint is reached.

| Scenario | Observed checkpoint and stopped state |
| --- | --- |
| Conveyor breakdown | Acknowledged ur5e-1 pickup from completed M1, before Conveyor release. The robot retains the part and Conveyor transport becomes unavailable. |
| ur5e-1 breakdown | Acknowledged M1 processing completion before pickup. M1 retains the completed part; ur5e-1 execution is disabled and its pose is observed. |
| Machining breakdown during part processing | The selected machine, default M1, interrupts at the first simulation-clock observation at or beyond half its configured dwell. The record retains WIP, observed pose, elapsed simulation time and unfinished process requirements. No successful machining effect is committed. |
| Part slippage | Both ur5e-3 and ur5e-4 must acknowledge pickup of different selected NIST parts. Subsequent tasks for each held part wait for the peer. Either robot can be selected to slip. |

For Part slippage choose the slipping robot, its eligible part, the other held
part, and the interrupted `place_insert` task. Supply explicit finite world
drop coordinates and a unit quaternion. The target must lie on the other
robot's side according to the configured robot bases. No default drop position
is silently supplied. The selected robot's own controller opens its gripper,
confirms detachment, applies the target, observes the settled part, and
acknowledges collision-scene synchronization. The requested target is stored
separately from the observed pose. The peer's custody and both task
continuations are retained. Unreachable checkpoints are recorded as
`not_reached`; custody and CCA permissions are never fabricated.

A machining fault is evaluated inside the worker's simulation-clock dwell,
with a control file bound to its run, task and resource. Pausing `/clock` does
not advance processing. Changing simulation speed does not replace this with
wall-clock timing. Arm and Disarm update that worker control; disarming after
an interruption has physically occurred cannot undo it.

Each scenario adds a non-colliding red outline and labeled sign at its affected
resource or observed dropped-part position. Gazebo service evidence confirms
marker entity presence; it does not claim that a human checked the rendered
text. Marker failures leave execution blocked. Partial detach, pose,
observation, or collision failures retain their completed steps and explicitly
require physical-state reconciliation.

Before an injection the runtime snapshots pending tasks, retained
continuations, requirements, custody, observations and resource revisions.
Execution admission closes and CCA grants are invalidated. Stop System and
agent teardown retain the fault latch. Reset Gazebo or Reset All recreates the
four-arm scene; the latch clears only after scene readiness and marker absence
are confirmed. Reset is refused while physical-effect reporting is in flight.
Reset Plan alone cannot clear a fault. Start System after a successful reset
creates fresh resource state and a new run.

The implementation uses the existing SystemBridge fault-control signatures and
conveyor-named compatibility fields. `bridge.py` is not extended. These scenarios
end with stopped-state evidence; automatic recovery and nominal resumption are
not part of this feature.

## Setup and verification

Run from the WSL checkout with ROS sourced:

```bash
cd ~/projects/cais-spade-llm
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
make bootstrap-gazebo
make run
```

Current diagnostic fixtures are under `test/fixtures/part_slippage/`, including
both slipping-robot directions. Their synthetic contexts and mocked responses
exercise diagnostics and recovery validators without claiming live validation.
The retired `lg_slippage` scenario, its preprogrammed recipe and its scenario
bundle have been removed; historical experiment records and unrelated product
geometry are retained.

## Validation on 2026-10-02

Validation used the WSL checkout on `recovery-framework-journal`.
`make bootstrap-gazebo` built all 16 workspace packages before the installed
Gazebo checks. `poetry check`, Python compilation and the UI CLI help check
passed; Poetry reported existing metadata deprecation warnings.

The normal SystemBridge startup path was exercised with CCA enabled
(`diagnostic_cca_bypass=False`) for all five scenario/resource combinations:

| Scenario/resource | Live outcome |
| --- | --- |
| Conveyor breakdown | Checkpoint `not_reached`; CCA held remaining candidates. |
| ur5e-1 breakdown | Checkpoint `not_reached`; CCA held remaining candidates. |
| Machining breakdown / M1 | Checkpoint `not_reached`; CCA held remaining candidates. |
| Part slippage / ur5e-3 | Startup failed while waiting for the ur5e-1 `/compute_fk` service; no checkpoint was established. |
| Part slippage / ur5e-4 | Checkpoint `not_reached`; CCA held remaining candidates. |

The four started runs recorded zero acknowledged task transitions. Their CCA
decisions included `no_joint_completion` and `time_limit`. No injected
custody, process completion, placement, or successful physical fault is
claimed from these attempts.

A separate installed Gazebo check confirmed marker presence for Conveyor,
ur5e-1, M1, M2, and both Part slippage resource selections. Removal was confirmed
for Conveyor, ur5e-1, M1, and M2. Part slippage removal checks encountered
intermittent `/get_entity_state` timeouts and reported failure. Scene recreation,
readiness and marker absence verification passed in an isolated reset check;
that check did not fabricate or exercise a physical failure checkpoint.

Full physical checkpoint-to-reset acceptance remains unverified until nominal
execution can reach these checkpoints with CCA enabled and controller services
remain available. The regression suites separately exercise fault effects,
both custody directions, clock timing, cancellation, partial effects, failed
markers and reset behavior using controlled observations.

The broad regression run completed with **1,058 passed, 26 skipped and 10
failed**. Five failures were corrected in recovery test fixtures: the historical
geometry/context split, current reset coordinates, bridge fault-state stubs, and
deferred report reads. Re-running the complete delivery, product and dual-robot
startup suites then passed **215 tests**.

Five failures remain in `spec2primitives/tests/test_ra_context_handoff.py`:
the expected synthesis-primitive catalog, two context-only catalog descriptions,
selected-context Gazebo readiness, and conditional geometry binding. Those
implementation paths were not changed. The only change in that file removes
the obsolete expectation that a default xarm6 loads `lg_slippage`; no
Spec2Primitives runtime integration or refactoring was performed.

The broad run passed the fault, setup, environment-execution, Gazebo-program,
diagnostic, recovery dry-run and feasibility suites, including both slippage
directions and the new cancellation/teardown cases. An active-code search found
no remaining `lg_slippage` or retired recipe imports. `bridge.py` retained
its original working-tree contents throughout this change.
