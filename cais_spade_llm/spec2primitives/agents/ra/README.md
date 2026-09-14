# RA: one primitive program

RA means RobotAgent. This directory owns selected-RA context handoff, direct program composition, derived input diagnostics, and the separate `PrimitiveExecutionRuntime` coordinator for Run in Gazebo. Phase 4 supplies observed product information and desired assembly relationships. Phase 5 uses that evidence to select operations and, within bounded refinement, calculate robot targets. Composition, strict numerical helper evaluation and isolated motion validation execute no robot actions. Only the separate operator execution action can dispatch the saved validated program.

## Context and model input

`activate_selected_ra_context` validates completion, evidence, capability and assignment lineage before addressing the exact selected RA. The owned adapter reuses that JID/mode or starts only that context-only RA under existing simulation readiness gates. Paired state/catalog snapshots are append-only. A new capture gets current metadata; old records are not upgraded or rewritten.

Fresh `robot_state.motion_context` records the exact configured `frame_id`, `ee_link`, and `tcp_link` from `controller_config.move_group`. Unavailable values remain null. Configuration names are not measured feedback or a TCP transform. Allocation base-distance frames and the controller planning frame have different purposes.

The captured catalog retains all runtime conditions/effects, parameter declarations, and output declarations. The model-facing projection limits `grasp_part` and `release_part` formal conditions/effects to `held_part` and removes `model_name` from parameters, nested helper inputs, outputs, required lists and custody effects. Primitive names, `part_name`, geometry requirements and movement parameters remain intact. This is a composition interface, not a directly executable Python signature. Recovery examples, decompositions and unrelated recovery metadata are excluded from initial state and historical evidence displays. New authoring has no model-driven evidence reader. The shared RA call has no execution tools or recovery instructions.

## Composition and binding checks

`author_primitive_program_candidate` reconstructs the grounded target, current/desired relationships, exact resource assignment, and accepted ontology projection from current completion and paired context. There is no preliminary sequence call or active draft stage.

RA chooses every primitive, order, intermediate movement and control parameter in one authoring call per semantic revision. It returns `propose` with `primitive_steps` and optional `context_requests`, or `unsupported`. New prompts contain the complete catalog once, compact accepted task/robot context and current findings. The 64,000-character limit leaves room for the complete bound program and revision feedback, and rejects oversized essential input before calling RA; no contracts are truncated. Raw correspondence records, observation inventories and previous evidence exchanges remain in disk audits.

RA returns `primitive_steps`; each step's `params` is a JSON-encoded object in the structured response. The host preserves this proposal exactly. Deterministic binding produces a separate checked program without adding or reordering primitives. Values can be literals, `value_ref: {record_ref, field_path}`, or `result_ref: {step_index, field_path}` selecting an earlier one-based step's declared output. Unknown primitives/parameters, invalid supplied types/bounds/array lengths, duplicate keys, nonfinite numbers and malformed or unauthorized references remain errors.

Omitted arguments and partial geometry objects remain visible proposals. Remaining `typed_parameters.required` declarations preserve their requirements after execution-only fields are excluded. `x-grounding-required` and `x-grounding-fields` describe extra information needed to ground a calculation; they do not add mandatory arguments to existing recovery calls. The diagnostic checker leaves the proposal unchanged; deterministic PA resolution and binding fill only its requested missing or incompatible measurement paths. New submissions containing `model_name` arguments, including nested or referenced objects, are rejected and preserved in the trace. References to removed helper outputs remain undeclared-field errors.

`read_primitive_composition_diagnostic` derives `binding_issues` with `step_index`, `parameter_path`, `status`, and `message`:

| Status | Meaning |
| --- | --- |
| `missing` | An argument, geometry field or configured frame/link is unavailable. |
| `incompatible` | Selected information conflicts with a declared need, such as a frame mismatch or raw CAD supplied as placement geometry. |
| `unverified` | A value's meaning/provenance is insufficient, such as a descriptive destination. Historical attempts may also report their original identifier gaps. |
| `deferred` | A primitive output is declared but has not been computed; this is different from missing source evidence. |

The report never rewrites a candidate and is not an execution-ready verdict. Type-valid references may still have incorrect physical meaning. This initial binding report is followed by dependency analysis, supplemental evidence, selected calculation and finite modeled validation in `refinement.py`. A modeled pass remains distinct from physical success.

## Geometry and execution identifiers

| Interface | Declared information and limits |
| --- | --- |
| `compute_pick_targets` | `target_pose` is an observed object location in world coordinates, or `detected_parts` supplies observations. The supported vertical calculation consumes `product_geometry.board_center.z`, `part_height_m`, configured EE/TCP feedback and controller policies. A raw CAD record is not that geometry. |
| Pick outputs | `approach_pose`, computed EE `target_pose`, `tx`, `ty`, `tz`, `pick_z`, `travel_z`, `part_height`, `tcp_offset_z`, `pick_tcp_z`, `start_x`, `start_y`, `start_z`, and `part_name`. Runtime `model_name` is excluded from composition. |
| `compute_place_targets` | Grounded geometry uses `board_center.x/y`, relative `slot_xy` and `slot_floor_z_m`. `pick_ctx` supplies `tz`, `pick_tcp_z`, `tcp_offset_z` and `part_height`; geometry may override height. Supported `target_reference`/`target_origin_pose` fields distinguish final part-origin semantics from a tool pose. |
| Place outputs | Approach/target poses, scalar place context and, when the selected branch returns them, `pre_insert_pose`, `insert_pose`, insertion direction and target-reference fields. `target_pose` may equal pre-insertion. |
| `move_cartesian` | Required `x`, `y`, `z` address the controlled EE in the configured planning frame; orientation and speed remain optional. An observed object point does not automatically include a grasp/tool offset. |
| Runtime `model_name` | Absent from composition inputs/reports. Run in Gazebo resolves the accepted CAD bytes through the configured Gazebo model definition and verifies the live instance pose, including the SDF visual transform. Exactly one instance must match. The binding remains in execution-only records. |

The vertical helpers assume world product geometry agrees with configured EE feedback. A different or unavailable planning frame is reported; no world-to-base conversion is applied. CAD-local dimensions need orientation evidence before representing vertical height. Controller fallbacks and specialized physical branches still exist; the new declarations do not make the helpers general assembly solvers.

`approach_pose` remains useful available geometry. RA decides whether and when to use it. Output extraction preserves returned pose orientation and conditional insertion fields without manufacturing absent insertion poses. Reaching a pre-insertion pose or releasing a part does not prove completed insertion.

## Persistence and UI

Each independent attempt writes `request.json`, `exchange_*.json`, and `candidate.json` under `composition/primitive_program_candidates/attempt_XXXX/`. `context_refs` pins completion, assignment, robot-state and catalog snapshots. Rejected/unsupported results and interrupted attempts remain auditable. A fresh attempt starts without previous candidates, traces or historical drafts as model input.

The history reader uses the catalog embedded in that attempt's saved `request.prompt` for validation, binding diagnostics and UI formatting. It checks the catalog against the pinned context after projection, without replacing the original declarations. Thus older programs may still contain or report `model_name`; the next attempt uses the current simplified interface. No historical bytes, hashes or parameter values change.

The UI shows one numbered program, display-only `<unbound>` markers for missing required/calculation inputs, and a short input report. Full records, report and trace expand on demand. The displayed binding resolves measured numbers and completed calculations; uncomputed result dependencies show `<pending: step …>`. Original values/references remain in the expandable records. Blocking validation and diagnostic refresh run outside the event loop; duplicate clicks cannot create another in-flight attempt. `proposed` means an unvalidated program proposal.

Restart an already running application to load the current composition code. Complete existing snapshots can supply the simplified projection to a new attempt; recapture only when metadata or robot context needs refreshing. New snapshots append; saved programs are not automatically regenerated. Incompatible authority records still cannot authorize new RA work.

## Subsequent work

[VALIDATION_AND_REVISION.md](../../VALIDATION_AND_REVISION.md) specifies PA evidence requests, RA-owned geometry/identifier adapters, complete binding validation, modeled/physical/outcome checks, RA revisions, freshness and stopping rules. [COMPOSITION_EVALUATION.md](../../COMPOSITION_EVALUATION.md) defines fair experiments on intermediate dependencies absent from supplied formal contracts. Neither document describes an implemented physical execution loop.

## Deterministic refinement

Read `refinement.py` (`compose`, `_run`) → `primitive_context.py` (`request_primitive_context`) → `primitive_context_messages.py` → `primitive_input_resolution.py` → `program_binding.py` → `program_validation.py`. Input checks precede robot capture and motion validation. PA uses real correlated SPADE messages with deterministic measurements and no Phase 5 LLM fallback. `PrimitiveProgramBinding` pins the original proposal and checked answers. Readers replay the binding; validation, the UI and `program_execution.py` select that same pin. Only semantic or motion findings return to the RA model.

The existing deterministic-input and SPADE binding regressions are in [test_primitive_refinement.py](../../tests/test_primitive_refinement.py); [handoff tests](../../tests/test_ra_context_handoff.py) cover bounded single-call prompts and immutable history. [UI](../../tests/test_pa_ui_connection.py) and [execution tests](../../tests/test_primitive_execution.py) cover numerical display, pending calculations, reconnect and rejected binding/report substitutions. These are controlled fixtures, not live LLM or physical-success evidence. See [validation and revision](../../VALIDATION_AND_REVISION.md) for freshness, budgets and scope limits.
