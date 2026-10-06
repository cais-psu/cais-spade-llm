# Mock part_slippage recovery: predefined safety results

Mock plans and synthetic physical evidence; no LLM calls or robot execution.

The existing offline checker derives AP values and evaluates the unchanged predefined LTLf/DFA definitions.

| Candidate | Mutex | Precedence | Combined |
|---|---|---|---|
| [PART_SLIPPAGE_MUTEX](mutex.json) | violated | satisfied | violated |
| [PART_SLIPPAGE_PRECEDENCE](precedence.json) | satisfied | violated | violated |
| [PART_SLIPPAGE_SAFE](safe.json) | satisfied | satisfied | satisfied |

The frozen scene contains 12 configured resources. Each complete trace requires 66 mutex instances and one precedence instance.
The initial checkpoint has `gear_small` displaced near `ur5e-3`, and `KET4_Square_4mm` in `Buffer For Machined parts`.

## PART_SLIPPAGE_MUTEX

- `ur5e-4` 0–4 s: `retrieve_gear_small_after_part_slippage`.
- `ur5e-4` 4–6 s: `return_gear_small_to_assembly_board-v1`.
- `ur5e-4` 6–9 s: `place_gear_small_and_retreat_after_part_slippage`.
- `ur5e-3` 10–11 s: `clear_assembly_board-v1_after_part_slippage`.
- `ur5e-3` 12–16 s: `retrieve_KET4_Square_4mm_after_part_slippage`.
- `ur5e-3` 16–18 s: `return_KET4_Square_4mm_to_assembly_board-v1`.
- `ur5e-3` 18–21 s: `place_KET4_Square_4mm_and_retreat_after_part_slippage`.

**Rejected by `SAFE_shared_area_mutex` at t=5.287879 s** (exact time `349/66`).
AP values: `{"ap001": true, "ap002": true}`.

- Event `PART_SLIPPAGE_MUTEX_return_gear_small_to_assembly_board-v1`, primitive index `1` (zero-based): `move_cartesian` on `ur5e-4`.
- `ap001`: `ap_state/physical_observation/shared_area_first_resource`, binding `{"region": "assembly_board-v1", "resources": ["ur5e-3", "ur5e-4"], "specification": "SAFE_shared_area_mutex"}`, value `True`.
- `ap002`: `ap_state/physical_observation/shared_area_second_resource`, binding `{"region": "assembly_board-v1", "resources": ["ur5e-3", "ur5e-4"], "specification": "SAFE_shared_area_mutex"}`, value `True`.

## PART_SLIPPAGE_PRECEDENCE

- `ur5e-3` 0–1 s: `clear_assembly_board-v1_after_part_slippage`.
- `ur5e-3` 1–5 s: `retrieve_KET4_Square_4mm_after_part_slippage`.
- `ur5e-3` 5–7 s: `return_KET4_Square_4mm_to_assembly_board-v1`.
- `ur5e-3` 7–10 s: `place_KET4_Square_4mm_and_retreat_after_part_slippage`.
- `ur5e-4` 12–16 s: `retrieve_gear_small_after_part_slippage`.
- `ur5e-4` 16–18 s: `return_gear_small_to_assembly_board-v1`.
- `ur5e-4` 18–21 s: `place_gear_small_and_retreat_after_part_slippage`.

**Rejected by `SAFE_gear_small_before_KET4_Square_4mm` at t=6.287879 s** (exact time `415/66`).
AP values: `{"ap001": true, "ap002": false}`.

- Event `PART_SLIPPAGE_PRECEDENCE_return_KET4_Square_4mm_to_assembly_board-v1`, primitive index `1` (zero-based): `move_cartesian` on `ur5e-3`.
- `ap001`: `ap_event/physical_observation/part_region_entry`, binding `{"part": "KET4_Square_4mm", "region": "assembly_board-v1"}`, value `True`.
- `ap002`: `ap_state/processCompleted/process_target_completed`, binding `{"part": "gear_small", "process": "assembly", "target": "Gear_Plate/Gear_Shaft_1"}`, value `False`.

## PART_SLIPPAGE_SAFE

- `ur5e-3` 0–1 s: `clear_assembly_board-v1_after_part_slippage`.
- `ur5e-4` 0–4 s: `retrieve_gear_small_after_part_slippage`.
- `ur5e-4` 4–6 s: `return_gear_small_to_assembly_board-v1`.
- `ur5e-4` 6–9 s: `place_gear_small_and_retreat_after_part_slippage`.
- `ur5e-3` 10–14 s: `retrieve_KET4_Square_4mm_after_part_slippage`.
- `ur5e-3` 14–16 s: `return_KET4_Square_4mm_to_assembly_board-v1`.
- `ur5e-3` 16–19 s: `place_KET4_Square_4mm_and_retreat_after_part_slippage`.

**Combined result: `satisfied`.**

## What this establishes

The mutex-only candidate fails mutex and passes precedence. The precedence-only candidate passes mutex and fails precedence. The safe candidate satisfies both over its complete supplied trace.

`ur5e-4` recovers `gear_small`; `ur5e-3` clears the board and retrieves `KET4_Square_4mm`. Each part has retrieve, return, and place/retreat events, with multiple ordered primitives. Only timing/order differ between candidates.

The safe candidate has a separately supplied synthetic `gear_small` assembly acknowledgement at t=9 s. `KET4_Square_4mm` subsequently enters the board: precedence is checked non-vacuously. Its placement preserves its trim record and does not complete its pending assembly task.

## Model boundary

- All poses, trajectories, durations, envelopes, checkpoints and acknowledgements are synthetic; no Gazebo capture or dispatch occurs.
- Programs use the existing offline 3D primitive-observation contract, including move_cartesian.target and release_part.transform. These are model inputs, not native UR5e controller dispatch programs.
- Fixed-orientation, piecewise-linear paths and axis-aligned envelopes describe the complete modeled motion. Reachability, collisions outside the reviewed region, and physical feasibility are not established.
- Every configured resource has physical evidence; assembly_board-v1 is an inventory-only containment owner and the reviewed region, not an additional configured resource.
- The initial ur5e-3 occupancy is observed prospectively. Earlier execution and the physical slippage are not certified by this trace.
- The assembly completion record is a separately supplied synthetic acknowledgement associated with the configured ur5e-4 place_insert declaration. Motion, release and event names alone never establish completion.
- The mock candidates stand in for LLM outputs. No LLM was called, and executable LLM program quality is outside this demonstration.

Re-run from the project root with `poetry run python scripts/check_part_slippage_safety.py`. The detailed [report.json](report.json) includes exact formulas, AP descriptors, DFA transitions, bindings, fingerprints, and modeled final states.
