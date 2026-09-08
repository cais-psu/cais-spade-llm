# Primitive catalog references

Primitive catalogs are owned by the exact selected RobotAgent and captured as paired Phase 5.1 state/catalog snapshots after current completion passes review/reachability recovery checks. Preserve exact symbols, order, typed parameters, results, limits and provenance; no fixed catalog cardinality is assumed.

Composition uses the current catalog to author one `primitive_steps` program directly from grounded context. RA chooses sequence and available parameters together; omitted required parameters display as `<unbound>`. Supplied names, types and references are checked without filling missing values. Phase 4 arm assignment validates robot position plans, while Phase 5 separately validates supported primitive bindings, nominal rigid grasp/custody, carried meshes and ordered Cartesian segments. Execution-binding validation, contact physics and execution remain future work.

Do not supply completed task recipes or expected sequences through catalog metadata. See [RA behavior](../../../agents/ra/README.md) and [bias validation](../../../BIAS_VALIDATION.md).

The authoritative capture retains full runtime conditions/effects and nested parameter/result schemas. The composition projection limits grasp/release formal contracts to `held_part`, removes `model_name` from inputs, outputs and custody effects, and excludes recovery metadata. It describes composition rather than directly callable Python signatures. Remaining requirements stay separate from `x-grounding-required` and `x-grounding-fields`; missing geometry is a reported gap, not a stronger runtime signature. The future execution adapter owns the simulator binding.

Historical candidates are inspected using the catalog in their own saved request; a subsequent attempt uses the current projection. No catalog snapshot or saved program is rewritten, and only a preceding candidate from the same bounded run may become refinement feedback.

Pick/place helper fields describe object geometry, tool/grasp offsets and named outputs. `approach_pose` remains available, and placement may return distinct conditional `pre_insert_pose`/`insert_pose`. These outputs do not prescribe their execution order. The strict calculation adapter evaluates only selected, completely bound helper calls and records their actual outputs. `move_cartesian` consumes x/y/z in the configured planning frame, with optional orientation/speed. See [evaluation of these information sources](../../../COMPOSITION_EVALUATION.md).
