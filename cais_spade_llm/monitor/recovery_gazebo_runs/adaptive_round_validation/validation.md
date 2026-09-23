# Live round production validation

Run: `1b875cb1dbec4316abd98d45e98ef4d8`. Final status: **completed**.

`Storage → KMR → M1 → ur5e-1 → Conveyor → Buffer For Machined parts → ur5e-3 → assembly_board-v1`

The observed final target is `GMC_Laser_Plate_Virtual/RGOCG4-50_Round_4mm`.
The ur5e-3 trace includes completed insertion and release primitives.
KMR is empty and idle at Storage.

## Motion

- Eight configured pickup positions passed complete collision-checked Cartesian descent/lift and loaded transport sweeps with neighboring pegs present. These are live planning checks; the production run physically picks the selected round peg.
- KMR retains the 7 cm vertical lift during mobile-base travel. M1 placement uses a checked joint_a1 turn, downward Cartesian placement, and 7 cm withdrawal.
- Mobile-base heading remains constant for this route.
- Independent planning checked 2415 combined UR1/KMR configurations for pickup entry and withdrawal with KMR at M1 and along its empty return. The production trace itself completed the KMR return before UR1 pickup.
- The link attachment plugin executes joint mutations on the Gazebo update thread. Twenty attach/detach cycles passed with clock progress and stable part position.

## Adaptive execution

The saved report records PA requests, RA transitions and rejection reasons, selections, CCA approvals, and acknowledgements.
Negotiation entries: `{"CCA": 17, "acknowledgement": 18, "capability_reply": 207, "capability_request": 19, "selection": 19}`.
M1 machining and the empty KMR return are independently negotiated and can execute concurrently.
The saved full order and M1 square default remain selected in the repository; the copied test setup selects the one-part M1 circle order.

## Timing

Observed production from first KMR worker request through assembly acknowledgement: 334.73 wall seconds.

Configured simulation speed: 1. Last measured full-scene clock rate after worker cleanup: 0.648 simulation seconds per wall second.
GPU renderer: D3D12 (NVIDIA GeForce GTX 1650 Ti), Mesa 23.2.1.
Recorded motion intervals ran mostly at about 0.2–0.3 simulation seconds per wall second.
Controller preparation: 3.70 s.
Agent startup and preparation until readiness: 30.04 s (Gazebo launch excluded).

Overlapping task durations should not be summed as elapsed production time. Planning columns are the workers' recorded MoveIt planning times; other validation and messaging overhead remains in wall time.

| Resource | Event | Wall s | Simulation s | Recorded planning s |
| --- | --- | ---: | ---: | ---: |
| KMR | pick_part | 9.67 | 2.20 | 0.07 |
| KMR | move_to_resource | 21.46 | 6.00 | 0.51 |
| KMR | place_release | 39.20 | 10.40 | 1.44 |
| M1 | machine_part | 19.14 | 5.00 | 0.00 |
| KMR | move_to_resource | 23.44 | 6.00 | 0.66 |
| ur5e-1 | pick_approach | 13.40 | 3.60 | 0.17 |
| ur5e-1 | pick_grasp | 6.62 | 2.00 | 0.11 |
| ur5e-1 | place_approach | 16.44 | 4.00 | 0.65 |
| ur5e-1 | place_release | 10.85 | 3.00 | 0.05 |
| Conveyor | advance_conveyor | 27.53 | 4.70 | 0.00 |
| Conveyor | advance_conveyor | 4.21 | 0.70 | 0.00 |
| Buffer For Machined parts | advance_part | 3.45 | 0.70 | 0.00 |
| Buffer For Machined parts | advance_part | 3.08 | 0.70 | 0.00 |
| Buffer For Machined parts | advance_part | 3.96 | 0.70 | 0.00 |
| ur5e-3 | pick_approach | 18.09 | 3.90 | 0.57 |
| ur5e-3 | pick_grasp | 8.32 | 1.90 | 0.12 |
| ur5e-3 | place_approach | 8.82 | 2.30 | 0.22 |
| ur5e-3 | place_insert | 13.73 | 3.30 | 0.17 |

## Verification

- 347 focused tests passed; 24 ROS-dependent tests skipped in the non-ROS test interpreter.
- `poetry check`, Python compile checks, and UI CLI help passed.
- `make bootstrap-gazebo`: 16 packages built, including the patched link attachment plugin.
- Live visible Gazebo production and attachment stress validation passed.

The final snapshot observed completion before the asynchronous report writer updated the outcome field. `run.json` combines that matching run/revision snapshot with the unchanged acknowledged transition trace and records this provenance. `runtime_saved_report.json` preserves the original file. Cleanup now preserves completed outcomes; the focused regression covers this fix.

Files: `run.json`, `live_final_snapshot.json`, `runtime_saved_report.json`, `pickup_validation.json`, `attachment_stress.json`, `overlap_validation.json`, `final_observation.json`, `simulation_performance.json`.

## Home completion follow-up

The UR arms now negotiate home after their own release task, and KMR returns to an observed downward-facing Storage home. See [home verification](home_validation.md) for the live continuation, endpoint observations, timing and regression evidence.
