# Mock part_slippage recovery with predefined safety

Start with the [computed results](REPORT.md). They compare three supplied mock
recovery plans against the unchanged
[predefined specifications](../../../../cais_spade_llm/specification/safety/safety_assembly_board-v1_predefined.txt).

`ur5e-4` has dropped `gear_small` near `ur5e-3`. `KET4_Square_4mm` is ready in
`Buffer For Machined parts`. `ur5e-3` initially occupies `assembly_board-v1`
with an empty gripper. All three cases begin with this same synthetic physical
checkpoint; their schedules determine the outcomes.

- [mutex.json](mutex.json): `ur5e-4` returns with `gear_small` before `ur5e-3`
  clears the board. `KET4_Square_4mm` enters after gear assembly, isolating mutex.
- [precedence.json](precedence.json): `ur5e-3` brings `KET4_Square_4mm` in first
  and leaves before `ur5e-4` returns with `gear_small`, isolating precedence.
- [safe.json](safe.json): `ur5e-3` clears the board, `ur5e-4` retrieves and
  places `gear_small`, its separate assembly acknowledgement occurs, and then
  `ur5e-3` retrieves and places `KET4_Square_4mm`.

Each case contains seven recovery events and 19 ordered primitives. Each part
has retrieve, return, and place/retreat events. `outline_id`, `des_event_id`,
event names, predecessors, parameter bindings, event-local `primitive_index`,
resource-local `step_index`, expected boundary states, and synthetic evidence
are saved together. Event names are new input identifiers, not AP definitions.

The [initial context](initial_context.json) freezes the configured scene and
explicit synthetic 3D envelopes for all 12 resources. The existing automatic
grounding code constructs 66 mutex pairs. The additional `assembly_board-v1`
inventory record is a containment owner, not a thirteenth spatial resource.

The programs use the **existing offline 3D primitive-observation contract**:
`move_cartesian.target`, `grasp_part.part_name` / `initial`, and
`release_part.transform`. These are model inputs, **not native UR5e dispatch
programs**. The trajectories are fixed-orientation, piecewise linear synthetic
paths. They do not claim robot reachability, continuous joint-space enclosure,
or Gazebo execution. Existing historical part_slippage records are unchanged.

The mock `gear_small` completion is explicitly linked to the configured
`ur5e-4` `place_insert` declaration and the exact
`{"process": "assembly", "target": "Gear_Plate/Gear_Shaft_1"}` record.
`kind: acknowledged` is a **synthetic trace assumption**, not an acknowledgement
from a running resource. Neither release nor the recovery event name establishes
assembly completion. `KET4_Square_4mm` keeps its trim record; its assembly task
remains pending even after placement.

Run from the project root:

```bash
poetry run python scripts/check_part_slippage_safety.py
```

This compiles the supplied definitions through the existing strict compiler,
checks each complete trace jointly and each specification separately, and saves
[REPORT.md](REPORT.md) and [report.json](report.json). The JSON includes exact
AP descriptors/bindings, DFA transitions, counterexample primitive references,
input hashes, and modeled endpoints. `dispatch_authorized` is always false.

These mock plans stand in for LLM outputs. This demonstrates checking supported
modeled behavior with previously unseen recovery identifiers, rather than LLM
generation quality or executable recovery. Renaming events, incomplete evidence,
and forged completion records are covered in `test/test_primitive_program_safety.py`.
No safety definitions, approvals, live monitor state, or runtime code are changed.
