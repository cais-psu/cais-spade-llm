# Eight-peg production implementation

## Scheduling

Intake uses one PA capability-request/reply window for local machine offers. It ranks an available executable offer by machine in-flight workload and waiting age, then discovers the next selected peg's route. It does not launch eight initial end-to-end searches. Already admitted parts and required robot home returns negotiate independently, so different resources can execute concurrently. Each offered next transition is revalidated before CCA approval and dispatch.

Resource-state, capability and execution availability changes invalidate dependent offers. Unaffected paths remain reusable. Controller ownership, access and material capacity remain part of the existing event/reservation/CCA contract. No extra fixture mutual-exclusion lock was added. The existing stationary fixture observation runs once in ordered startup.

## Motion fixes found in live trials

- Successful controller completion now also requires the observed Cartesian endpoint before a following primitive consumes pose feedback. Both direct Cartesian and free-space fallback paths use this pose postcondition, including after a joint settling mismatch; joint-target home moves retain strict joint checks. This resolved a 16.8 mm stale-pose insertion-start discrepancy without widening the insertion tolerance. The settling-mismatch warning uses the ROS logger signature and is covered by a strict-signature test.
- UR home planning chooses the nearest equivalent revolute joint targets within the running joint limits. Collision planning and observed empty-home acknowledgement remain required. A failed full-turn request became a valid 61 ms read-only plan.
- Collision geometry follows resource custody. KMR initializes the shared scene, then updates its own peg. UR resources transfer their observed CAD collision bounds into and out of the owning gripper; transport resources update all moved residents at observed arrival. Other resources' payloads are untouched. Empty KMR returns retain part obstacles. No additional mutual-exclusion lock is introduced.
- Two live collision defects were reproduced without robot motion: a stale M1 duplicate reduced a second-square descent to 21.4%, and a carried round peg represented as a fixed obstacle reduced ur5e-2's descent to 80%. Correct ownership produced complete collision-checked paths in both cases. Failed scene acknowledgement retains physical custody/movement evidence and blocks the event acknowledgement.

- UR feedback now filters messages to the owning robot, rejects out-of-order samples per joint, and identifies actual clock resets separately. Arm and gripper observations cannot suppress one another; joint-target timing waits for fresh feedback before calculating motion limits. Simulation subscriptions consume recent sensor samples, and the controller executor runs callbacks directly on its existing single spin thread. Recovery Framework Cartesian poses and TCP offsets use a fresh Gazebo observation of the owned wrist and its fixed tool transforms. This avoids delayed world TF and MoveIt service contention during concurrent planning. Delayed or unanswered read-only observations are retried within a configured five-second observation deadline, with the 0.25 simulation-second freshness limit unchanged. Recovery Framework controllers consume recent dynamic TF samples instead of a long reliable backlog. Persistent staleness, missing links, failed queries and cancellation still fail. A read-only live probe of the actual helper matched all eight tool/TCP reference poses within 0.03 mm; warmed queries took 2–3 ms. Tolerances and motion limits are unchanged.


- UR payload geometry is expressed in the physically observed attachment frame. Grasp completion now also requires the observed payload bounds to reach the gripper TCP within the existing Cartesian tolerance. A failed custody observation retains the physical attachment result and measured distance and withholds event acknowledgement. An isolated square8 grasp passed; the preceding full run had a physical peg displacement whose cause is still under investigation. Continuous part observations are enabled for the next validation run.

- Placement target calculation uses the recorded finite pickup tool offset directly and only queries a fallback when it is missing or invalid. This removes an unnecessary live query without changing the computed target.

- Storage pegs sharing a dock now acknowledge the requested endpoint using its observed pose and unchanged tolerance. The first matching endpoint label cannot reject the neighboring row's valid dock.
- Cached paths omit resource effects already present in acknowledged state. An assembly robot home completed for a preceding peg therefore does not run again or prevent the next buffered peg from advancing. Failed home acknowledgements do not satisfy that cached step; the next necessary transition still goes through PA–RA validation and CCA approval.

- KMR velocity commands use a one-sample best-effort stream with the existing command expiry. The Gazebo receiver also consumes only the newest velocity, avoiding delayed reliable replay during braking. The next two live runs passed all KMR docking approaches reached before unrelated UR failures.

- KMR waits for observed braking completion before changing direction at route waypoints and acknowledging docking. This prevents residual lateral velocity from cutting the M2 approach corner; the map, stopping-footprint checks, and configured speed/acceleration limits are unchanged.

The exact eight Storage identities are in `assembly_board-v1-eight-pegs.json`. Saved full-order and one-part orders remain available. M1 remains configured for square trimming and M2 for circle trimming; machining durations and motion limits are unchanged. The M2 docking pose and visible marker are (-3.25, 2.20, pi/2).

## Shared functions

Machine and transport functions consume `resources/workflow_task_programs.py`. Internal step outcomes retain task/event correlation and partial-motion observations. Only complete observed functions acknowledge capability events. Failed functions block production; downstream occupancy is not acknowledged. The Resources catalog, robot place_release/place_insert mapping and unavailable planned printer function are shared with the other implementation.

- A later round8 approach exposed a Cartesian IK branch discontinuity: with jump checking disabled, MoveIt returned fraction 1.0, but collision checking its timed samples found a collision. Repeating the same query with jump checking enabled truncated it to 0.755 before execution. UR Cartesian requests now use jump checking; timed UR paths and interpolated samples are collision checked before dispatch. This shares existing MoveIt resource motion validation and does not add mutual exclusion outside CCA. Failed grasp evidence survives gripper rollback. Timed-path state observations retry unanswered read-only queries within the existing observation deadline; an invalid state or exhausted deadline still prevents dispatch. The local round8 M2 approach and grasp passed with the peg within 0.01 mm of its workholding pose throughout the approach. The focused motion/recovery suite passes 244 tests; the eight-peg real-inbox regression passes with one active Storage route discovery.

## Validation status

Live eight-peg validation completed successfully with 164 acknowledged events. Independent Gazebo observations confirmed all eight correct slots and all five empty, open robots at home, including downward-facing KMR at Storage. See [validation.md](validation.md) for measured timing, overlap, limits, and preserved evidence.
