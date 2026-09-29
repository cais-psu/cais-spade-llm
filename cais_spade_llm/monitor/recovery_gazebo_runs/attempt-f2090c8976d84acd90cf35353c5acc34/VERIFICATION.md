# Verification of raised KMR startup

## Code and installed assets

- 25 new focused checks passed: all eight first-part selections, actual URDF FK,
  spawn observation without a trajectory, stale/invalid joint and TCP feedback,
  collision rejection, and the saved first transport step.
- Broader focused suites: **626 passed, 2 deselected**.
  Files: `test_recovery_framework_gazebo.py`, `test_simulation_timing.py`,
  `test_recovery_delivery.py`, `test_gazebo_resource_programs.py`.
- The deselected tests depend on pre-existing workspace conditions:
  `retired_lg_slippage_has_no_active_bindings` sees the unrelated active
  `lg_slippage.json`; `prepared_recording_covers_configured_parts_and_observed_home_joints`
  lacks its historical saved-waypoints fixture. Neither was changed or removed.
- `poetry check`, `poetry run python -m compileall -q cais_spade_llm ros2`,
  `poetry run python -m cais_spade_llm.ui_main --help`, `git diff --check`,
  and `make bootstrap-gazebo` passed. Poetry reported existing metadata deprecations.
- All configured installed scene assets matched the source before fresh launch.
- Scene changes are limited to `KMR.parked_arm_configuration` and its descriptive
  `parked_pose_role`. See [scope audit](startup_change_scope.json).

## Gazebo acceptance

The [acceptance audit](acceptance_audit.json) verifies 11 configured placements,
195 acknowledged transitions, exact saved primitive steps, formal model agreement,
unchanged execution sources during the run, and zero KMR/UR equipment contacts.
The [startup audit](startup_pose_audit.json) records fresh joint/TCP feedback,
mandatory startup collision validation, and no first-step arm trajectory.
The [clearance audit](clearance_audit.json) covers all 22 KMR base actions.

The native contact monitor counted 645,198 messages with zero equipment contacts.
Existing observation retries recovered using fresh Gazebo feedback; the complete
[execution log](acceptance_run.log) is retained. All three ur5e-4 rotation checks passed.
All five robots were observed in Gazebo links and visible in sampled GUI/video frames.

## Video validation

- Full: 19,089 decoded frames, 1908 × 998, 15 fps, 1272.6 seconds.
- 20×: 955 decoded frames, 1908 × 998, 15 fps, 63.667 seconds.
- NVIDIA H.264 encoding; all streams fully decoded without errors.
- 20× duration matches full duration divided by 20 within one frame.
- The full video checksum remained unchanged during accelerated export.

Settings remain speed 1, ODE threads 0, unchanged timestep and robot limits,
diagnostic CCA bypass enabled, UR `avoid_collisions=false`, KMR collision checks active.
