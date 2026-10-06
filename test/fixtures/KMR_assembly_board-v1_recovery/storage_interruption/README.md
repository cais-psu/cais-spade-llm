# Storage interruption

This manually authored synthetic case starts with `KMR` holding
`KET8_Square_8mm` after its Storage pickup. The separate recovery target is
the displaced `KET4_Square_4mm`, with `trim`, `result: square` already in
`processCompleted`. No runtime or physical observation is asserted.

- `initial_context.json`: both parts, Storage inventory, configured pose
  references, five original nominal tasks, and recovery/resumption expectations.
- `event_sequence.json`: four new ordered recovery events.
- `primitive_programs.json`: per-event programs and zero-based
  `step_expectations`, including explicitly mocked helper outputs.

| outline_id | Purpose | Primitive count |
| --- | --- | --- |
| KMR_STORAGE_INTERRUPTION_SEQ1 | Stage KET8_Square_8mm in its vacated Storage slot. | 5 |
| KMR_STORAGE_INTERRUPTION_SEQ2 | Acquire the displaced KET4_Square_4mm. | 5 |
| KMR_STORAGE_INTERRUPTION_SEQ3 | Place KET4_Square_4mm at assembly_board-v1 and retreat. | 4 |
| KMR_STORAGE_INTERRUPTION_SEQ4 | Return and reacquire KET8_Square_8mm. | 8 |

Recovery ends with `KMR` carrying `KET8_Square_8mm` at `Storage`, restoring
its initial TCP/base pose and grasp transform. The same `move_to_resource`,
`place_approach`, and `place_release` task IDs remain pending for delivery
to `M1`; the original two pickup tasks remain completed. Their later expected
completion is separate from `expected_final_conditions`.

The Storage slot and pickup dock come from the current scene. TCP poses use
`[x, y, z, qx, qy, qz, qw]`; slot source values use XYZ/RPY and base poses use
`[x, y, yaw]`. Other poses and helper outputs are synthetic. Named transport
poses move the TCP; empty base travel translates it with fixed yaw. Each grasp
transform maps TCP to part pose, and staged/released parts stay stationary.
The helper outputs in event 4 match its later primitive parameters.

`KMRResourceAgent` uses `recovery-resource-8@localhost` in the current scene
order. Primitive signatures match `KMRPrimitives`. Returning a part to Storage
and handling the displaced part are mock recovery events, not existing nominal
KMR capabilities. No `compute_place_targets` is used for Storage or the board.

Slot vacancy, support after release, parked-arm feasibility, and routes are
synthetic assumptions. `processCompleted` is preserved for both parts; no
`assembly` completion is added. The three core JSON files perform no safety/AP/DFA
evaluation and provide no runtime acknowledgement, LLM generation, dispatch, Gazebo execution, or
live resumption. It is not a `scenario_runner.py` bundle.

## Offline safety evidence

`safety_evidence.json` separately supplies the unchanged 22 steps on `[0, 22]`,
with original event/step provenance, synthetic 3D geometry and trajectories,
helper outputs, custody changes, complete process ledgers, and reviewed rules.
It declares all five robot participants and requests every mutex pair;
`ur5e-1` through `ur5e-4` remain stationary outside the area in this baseline.
The second requirement is: “KET4_Square_4mm may enter assembly_board-v1 only after
its trim with result: square is completed.” Other-resource exclusions are explicit.
These companion inputs are for the offline reviewed checker; they do not alter
the original fixture JSONs or authorize local admission, dispatch, or execution.

## Automatic grounding evidence

`automatic_grounding_evidence.json` preserves the same 22 steps, source records,
nominal tasks, and two reviewed requirements, with the complete frozen scene and
its source hash. `validate_grounded_primitive_program_safety(**inputs)` derives
all 12 configured resources and their 66 mutex pairs from
`build_environment_models(scene)`; no caller participant, pair, or exclusion list
is supplied. `requirement_scopes` retains the exact reviewed region and part bindings.

Every configured resource has explicit geometry and motion or stationary evidence.
The seven fixed resources use `stationary_only: true` and synthetic envelopes
at their configured XYZ positions; their pose stays fixed while Storage inventory
can change through KMR custody effects. `assembly_board-v1` remains a region and
inventory-only containment owner. Acceptance is conditional on these authored
envelopes and trajectories, with no runtime, physical-feasibility, or execution claim.
The earlier JSON files remain unchanged.

## Offline composition evidence

`composition_evidence.json` references the frozen automatic-grounding companion
by path and hash. Apply its `grounding_input_overrides` to that companion's
`inputs`, then pass the resolved `grounding_inputs` with its composition `inputs`.
Both alternatives preserve the four event IDs, all 22 KMR primitive parameters,
source records, and pending nominal delivery tasks.

An explicitly synthetic, already-running `ur5e-3` withdrawal occupies the board
until it retreats during `[11.5, 12]`. The immediate KMR event starts are
`[0, 5, 10, 14]` and conflict during placement. The delayed starts are
`[0, 5, 12, 16]`, with a supplied KMR wait on `[10, 12]`, and satisfy the reviewed
rules. Every trajectory, custody interval, and stationary interval is authored
for the complete `[0, 26]` horizon; the loader does not generate timing evidence.
Completion restores KMR carrying `KET8_Square_8mm` at Storage and leaves
`KET4_Square_4mm` deposited. The fixture is offline evidence only, with no runtime
admission, dispatch, or execution claim; earlier JSON files remain unchanged.
