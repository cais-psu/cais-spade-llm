# Supplied Gazebo motion candidates

These are mock candidate recipes, not generated recovery or observed execution.
Use **Load supplied candidate** followed by **Prepare and check**. The request
keeps exact native `move_cartesian` parameters and event/step references. Retreat
coordinates come from the actual checkpoint; no gripper operation is included.

`GAZEBO_MOTION_SAFE` prepares `ur5e-4`. `GAZEBO_MOTION_CONFLICT` also prepares
`ur5e-3` using the declared simultaneous start offsets. Actual overlap must be
established from the prepared trajectories; the recipe name does not establish
a verdict. Neither recipe establishes the desired part-slippage initial state.

Live observations and prepared trajectories are saved separately under
`cais_spade_llm/monitor/recovery_safety_preparation/`. No fixture coordinates,
expected result, or supplied `allowed` flag grants execution permission.
