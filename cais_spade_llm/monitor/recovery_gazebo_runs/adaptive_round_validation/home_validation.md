# Robot home completion verification

Each robot returns when its own handling cycle has finished. Product progress continues independently.

- KMR: after placement and clearance at M1, negotiate the empty return to Storage, then use its owned `move_home` primitive to reach the saved downward pickup posture with the gripper open before acknowledging the return.
- ur5e-1 and ur5e-2: negotiate `move_home` after release to the conveyor or staging tray.
- ur5e-3 and ur5e-4: negotiate `move_home` after insertion, release, and withdrawal.
- Product completion requires all five robots empty and idle at their home locations. Fresh UR joint feedback and observed KMR downward home are required for the corresponding acknowledgements.

## Live validation

The existing Gazebo production scene was preserved. The round peg remained assembled in its exact slot. This was a homing continuation of the previous acknowledged production run, not a new production replay. `home_continuation_source.json` identifies the starting state and original run.

All four UR home requirements used real PA–RA capability requests/replies, CCA approval, RobotAgent execution and validated acknowledgement. Machine and assembly workspace reservations prevent conflicting home motions.

| Robot | Observed result | Task wall seconds |
| --- | --- | ---: |
| ur5e-2 | Already home; observed without movement | 0.021 |
| ur5e-3 | Returned home | 6.495 |
| ur5e-4 | Already home; observed without movement | 0.016 |
| ur5e-1 | Returned home | 9.325 |

Protocol records: `{"CCA": 3, "acknowledgement": 4, "capability_reply": 11, "capability_request": 7, "offer_rejected": 3, "selection": 7}`.

KMR's downward home was then exercised separately through its owned callable primitive in the same scene. MoveIt collision planning and controller endpoint observation succeeded with the neighboring pegs present. Its normal empty-return executor now calls that same primitive before acknowledgement.

An independent ROS joint-state, entity-state and TF probe confirmed all five endpoints, KMR at Storage and the round peg still in its slot. Maximum joint error: 0.004301 rad. KMR tool Z axis world-Z component: -0.999996218 (downward is -1).

Gazebo remains open. Owned validation controllers/workers were cleaned up.

## Regression checks

- Adaptive capability, nominal model and Gazebo regressions: 230 passed, 24 skipped in the non-ROS test interpreter.
- Updated home and scheduling checks: 16 passed, including the actual scheduler and agent inboxes starting ur5e-1's home after release while assembly remains pending. This test also holds conveyor work until that return starts, verifying that homing does not reserve the released part.
- Delivery and KMR primitive regressions: 81 passed.
- `poetry check` and `poetry run python -m compileall -q cais_spade_llm ros2` passed. Poetry reports existing metadata deprecation warnings.

Test counts overlap. The full physical production route was validated in the earlier run; this follow-up verifies the new home motions live and immediate-return scheduling in the protocol tests. No ROS2 launch/script/RViz sources were changed in this follow-up.
