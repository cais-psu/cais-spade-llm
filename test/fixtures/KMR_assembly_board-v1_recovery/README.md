# KMR assembly_board-v1 recovery fixture

Manually authored, explicitly synthetic checkpoint and complete four-event
recovery sequence for a future offline study of the
[assembly_board-v1 mutex requirement](../../../cais_spade_llm/specification/safety/safety_assembly_board-v1_mutex.txt).

For KMR already carrying a different part from Storage, see the
[Storage interruption case](storage_interruption/README.md). It uses multiple
primitives per recovery event and preserves the pending M1 delivery.

- `initial_context.json`: initial custody, poses, expected part poses after each
  event, and final conditions.
- `event_sequence.json`: four ordered outline events with expected states.
- `primitive_programs.json`: one program per `outline_id`, preserving the existing
  recovery program record fields. All projected values are authored expectations.

Initially `KMR` is `carrying` `KET4_Square_4mm` with its gripper `closed` and an
identity `grasp_transform`; the part pose equals the TCP pose. This custody is
assumed, with no pickup or preceding failure execution.

| outline_id | event_name | Primitive | Expected result |
| --- | --- | --- | --- |
| RECOVERY_SEQ1 | KMR_recovery_approach_assembly_board-v1 | move_cartesian | TCP and held part reach x = 0.5. |
| RECOVERY_SEQ2 | KMR_recovery_placement_assembly_board-v1 | move_cartesian | TCP and held part reach x = 1.0. |
| RECOVERY_SEQ3 | KMR_recovery_release_KET4_Square_4mm | release_part | TCP stays at x = 1.0; held_part becomes null and gripper opens. |
| RECOVERY_SEQ4 | KMR_recovery_retreat_assembly_board-v1 | move_cartesian | Empty TCP returns to x = 0.0; released part stays at x = 1.0. |

Finally `KMR` is `idle`, `held_part` is null, and its gripper is `open`.
`KET4_Square_4mm` remains at the placement pose with expected `part_location`
`assembly_board-v1`; stationary support is assumed, and no `assembled` state is asserted.

Poses use `[x, y, z, qx, qy, qz, qw]` in `world`; `base_pose` uses `[x, y, yaw]`.
Coordinates are arbitrary fixture values, not executable configuration. The
orientation and base stay fixed; `resource_location` stays null.

The scene order and `create_environment_resource_agents` bind the existing
`KMRResourceAgent` to `recovery-resource-8@localhost`; recheck if that order changes.
Programs use `KMRPrimitives.move_cartesian(target, waypoints=None, seed=None)`
and `release_part(transform=None)`, both selectable in the current scene. Release
receives the initial `grasp_transform`; it opens and detaches without proving assembly.

No safety validator is attached. States and poses are synthetic expectations,
not accepted transitions or measured evidence. This is neither an LLM response
nor the existing `part_slippage` scenario. Physical feasibility, safety coverage,
and Gazebo execution remain unvalidated; no dispatch or model integration occurs.
