# RA: one primitive program

RA means RobotAgent. This directory owns selected-RA context handoff, direct program composition, and derived input diagnostics. Phase 4 supplies observed product information and desired assembly relationships. Phase 5 uses that evidence to select operations and, within bounded refinement, calculate robot targets. Strict numerical helper evaluation and isolated motion validation execute no robot actions.

## Context and model input

`activate_selected_ra_context` validates completion, evidence, capability and assignment lineage before addressing the exact selected RA. The owned adapter reuses that JID/mode or starts only that context-only RA under existing simulation readiness gates. Paired state/catalog snapshots are append-only. A new capture gets current metadata; old records are not upgraded or rewritten.

Fresh `robot_state.motion_context` records the exact configured `frame_id`, `ee_link`, and `tcp_link` from `controller_config.move_group`. Unavailable values remain null. Configuration names are not measured feedback or a TCP transform. Allocation base-distance frames and the controller planning frame have different purposes.

The captured catalog retains all runtime conditions/effects, parameter declarations, and output declarations. The model-facing projection limits `grasp_part` and `release_part` formal conditions/effects to `held_part` and removes `model_name` from parameters, nested helper inputs, outputs, required lists and custody effects. Primitive names, `part_name`, geometry requirements and movement parameters remain intact. This is a composition interface, not a directly executable Python signature. Recovery examples, decompositions and unrelated recovery metadata are excluded from initial state and subsequent `read_record` results. Raw catalog snapshots cannot be requested through that reader. The shared RA call has no execution tools or recovery instructions.

## Composition and binding checks

`author_primitive_program_candidate` reconstructs the grounded target, current/desired relationships, exact resource assignment, and accepted ontology projection from current completion and paired context. There is no preliminary sequence call or active draft stage.

RA chooses every primitive, its order, intermediate movements and supplied parameters. It returns `read_record`, `query_ontology`, `propose`, or `unsupported`; within refinement it may also return a bounded `request_context` action. The initial ontology index lists subjects/predicates; exact filtered queries return up to 32 accepted assertions. Record reads use pinned refs and RFC 6901 field paths and do not follow embedded artifact paths. The default limit is 12 evidence requests plus one final model call.

RA returns `primitive_steps`; each step's `params` is a JSON-encoded object in the structured response. The host parses it without adding, reordering or repairing steps/values. Values can be literals, `value_ref: {record_ref, field_path}`, or `result_ref: {step_index, field_path}` selecting an earlier one-based step's declared output. Unknown primitives/parameters, invalid supplied types/bounds/array lengths, duplicate keys, nonfinite numbers and malformed or unauthorized references remain errors.

Omitted arguments and partial geometry objects remain visible proposals. Remaining `typed_parameters.required` declarations preserve their requirements after execution-only fields are excluded. `x-grounding-required` and `x-grounding-fields` describe extra information needed to ground a calculation; they do not add mandatory arguments to existing recovery calls. The checker resolves only RA-selected references, never alternative sources or helper values. New submissions containing `model_name` arguments, including nested or referenced objects, are rejected and preserved in the trace. References to removed helper outputs remain undeclared-field errors.

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
| Runtime `model_name` | Deferred to the future execution adapter and absent from new composition inputs/reports. The backend attachment call still requires it. `part_name`, an ontology label, and a CAD filename do not identify the particular simulator instance. No identifier resolver is introduced. |

The vertical helpers assume world product geometry agrees with configured EE feedback. A different or unavailable planning frame is reported; no world-to-base conversion is applied. CAD-local dimensions need orientation evidence before representing vertical height. Controller fallbacks and specialized physical branches still exist; the new declarations do not make the helpers general assembly solvers.

`approach_pose` remains useful available geometry. RA decides whether and when to use it. Output extraction preserves returned pose orientation and conditional insertion fields without manufacturing absent insertion poses. Reaching a pre-insertion pose or releasing a part does not prove completed insertion.

## Persistence and UI

Each independent attempt writes `request.json`, `exchange_*.json`, and `candidate.json` under `composition/primitive_program_candidates/attempt_XXXX/`. `context_refs` pins completion, assignment, robot-state and catalog snapshots. Rejected/unsupported results and interrupted attempts remain auditable. A fresh attempt starts without previous candidates, traces or historical drafts as model input.

The history reader uses the catalog embedded in that attempt's saved `request.prompt` for validation, binding diagnostics and UI formatting. It checks the catalog against the pinned context after projection, without replacing the original declarations. Thus older programs may still contain or report `model_name`; the next attempt uses the current simplified interface. No historical bytes, hashes or parameter values change.

The UI shows one numbered program, display-only `<unbound>` markers for missing required/calculation inputs, and a short input report. Full records, report and trace expand on demand. Supplied values/references remain as RA submitted them. Blocking validation and diagnostic refresh run outside the event loop; duplicate clicks cannot create another in-flight attempt. `proposed` means an unvalidated program proposal.

Restart an already running application to load the current composition code. Complete existing snapshots can supply the simplified projection to a new attempt; recapture only when metadata or robot context needs refreshing. New snapshots append; saved programs are not automatically regenerated. Incompatible authority records still cannot authorize new RA work.

## Subsequent work

[VALIDATION_AND_REVISION.md](../../VALIDATION_AND_REVISION.md) specifies PA evidence requests, RA-owned geometry/identifier adapters, complete binding validation, modeled/physical/outcome checks, RA revisions, freshness and stopping rules. [COMPOSITION_EVALUATION.md](../../COMPOSITION_EVALUATION.md) defines fair experiments on intermediate dependencies absent from supplied formal contracts. Neither document describes an implemented physical execution loop.

## Code-reading handoff

### Outcome

New RA programs use a composition interface without `model_name`. Missing geometry remains visible. Saved attempts retain the catalog they were authored against; no target calculation, identifier lookup or robot execution is added.

### Process flow

UI → validated pinned context → composition catalog → selected RA → supplied-value checks → saved unchanged program. Reload → request's recorded catalog → validation and binding report → compact display.

### Read these locations in order

1. [Owned adapter](../../adapters/in_process_robot_agent.py): `request_assigned_context` captures configuration and full runtime data; `_phase_5_1_primitive_catalog` retains declared metadata.
2. [Composition context](composition_context.py): `_composition_input`, `_composition_state_view`, and `_composition_catalog_view` define exactly what the model sees.
3. [Composer](primitive_composition.py): `author_primitive_program_candidate` preserves RA decisions; `_recorded_request_inputs` restores an attempt's original catalog for inspection; `_validate_parameter` rejects execution-only arguments in new proposals and checks references/types without requiring complete geometry.
4. [Binding assessment](parameter_binding.py): `assess_parameter_bindings` and `_BindingReport.visit` inspect only selected inputs and distinguish gaps from deferred outputs.
5. [UI](../../spec2primitives_ui.py): `_format_primitive_program` and `_format_binding_issues` produce the concise display; `_start_primitive_composition` retains the in-flight guard and worker dispatch.
6. Shared metadata: [FunctionAnalyzer](../../../function_analyzer.py) `analyze_function`; [controller declarations](../../../resources/robot/gazebo_pick_place_controller.py) for target helpers and moves; [output schemas/extractors](../../../resources/robot/robot_primitives.py) preserve actual pick context and conditional placement fields.

### Read this test

In [RA tests](../../tests/test_ra_context_handoff.py), `test_geometry_gaps_and_conditional_results_preserve_the_exact_program` proves missing geometry remains visible while RA's steps/parameters stay exact. `test_composition_projects_minimal_contracts_and_filters_every_state_read` checks both model-input paths and immutable snapshots. [UI tests](../../tests/test_pa_ui_connection.py) cover compact gaps, full expandable reports, real worker responsiveness, duplicate clicks and detached pages. Shared catalog/extractor tests use no helper/controller execution.

`test_saved_attempt_uses_its_recorded_catalog_and_next_attempt_uses_new_interface` checks an older identifier-bearing program, unchanged saved files, and a subsequent mocked proposal without `model_name`. Nested-argument and removed-output tests prove identifiers cannot be supplied through helper inputs. `test_primitive_program_uses_attempt_catalog_for_execution_bindings` checks the corresponding old/new UI displays.

### Runtime evidence

Read-only reload of `interaction_116f33c8cce64d33b0066891bf228572` and `interaction_4178296ec9e04fb19e6e780371fd062b` returns the same 10-step `proposed` programs under their recorded catalogs. Their 141 and 179 files respectively match the before-change hashes, with no additions or removals. The history display retains their original identifier arguments/findings, while the inputs prepared for a new attempt exclude `model_name`. No live model request, context capture or program regeneration was performed.

### You can ignore

Shared ProductAgent/RobotAgent internals, `SystemBridge`, PA allocation/proximity/reassignment implementation, collision-scene synchronization and ground-truth answers are outside this increment's changed runtime path. Earlier local changes to those owned paths remain present.

### Refactoring performed

No unrelated refactoring. A recursive projection removes the same execution field from nested declarations and custody effects. The saved-request reader reuses the recorded catalog instead of introducing another record format. Shared source files, controller actions, output extractors and recovery sequences are unchanged from this increment's baseline.

### Verification and intentionally unchanged behavior

For identifier separation, 30 focused offline tests passed in two disjoint batches of 15. They cover new/old contracts, rejected inputs, exact history, recovery metadata requirements, UI rendering, and worker responsiveness. The initial sandbox batch timed out without results; passing batches ran with normal worker-thread access and mocked RA responses. Scoped Ruff (`F401,F821,E9`), compilation of the four touched Python files, and `git diff --check -- .` within Spec2Primitives passed. Local links in the 17 changed maintained Markdown files resolved. The two saved interactions and three shared source files retained their baseline hashes. No live model, target-helper, ROS/MoveIt or robot execution test ran. See [implementation status](../../IMPLEMENTATION_PLAN.md) for the earlier primitive-input increment's separate verification.

### Changed-file index for this increment

Runtime and tests are linked above. Affected maintained documents are:

- [Local instructions](../../AGENTS.md), [overview](../../README.md), [implementation status](../../IMPLEMENTATION_PLAN.md), [research scope](../../ICRA_SCOPE.md), [research positioning](../../RESEARCH_POSITIONING.md), [ontology](../../ASSEMBLY_ONTOLOGY.md), [bias audit](../../BIAS_VALIDATION.md), [validation/revision](../../VALIDATION_AND_REVISION.md), and [composition evaluation](../../COMPOSITION_EVALUATION.md).
- [RA](README.md), [adapters](../../adapters/README.md), [contexts](../../contexts/README.md), [schemas](../../schemas/README.md), [tests](../../tests/README.md), and [fixtures](../../tests/fixtures/README.md).
- [References](../../references/README.md) and [primitive catalogs](../../references/resources/primitive_catalogs/README.md).

## Bounded refinement implementation

Read `refinement.py` (`compose`, `_run`) → `program_dependencies.py` → PA `primitive_context.py` and the measured RA adapter → `program_validation.py` → the next exact RA candidate. `refinement_records.py` supplies immutable records and source-chain checks. Only the current run's previous candidate is refinement input. Default limits are three candidate versions, two PA batches, twelve PA operations and five minutes. A finite supported validation model yields `validated_for_declared_scope`; unknown coverage stays explicit. See [the full scope](../../VALIDATION_AND_REVISION.md).
