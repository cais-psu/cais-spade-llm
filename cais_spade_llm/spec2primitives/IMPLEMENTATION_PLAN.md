# Spec2Primitives implementation status

The broader composition/execution architecture remains proposed in [ICRA_SCOPE.md](ICRA_SCOPE.md). This page describes current runtime behavior.

## Implemented phases

| Phase | Boundary |
| --- | --- |
| 1 | Approved source serving, observation bundles, demand-driven capture |
| 3 | Non-executing PA interaction, clarification history, persistence/recovery |
| 4.0 | Supplied PPR TBox, configured resource/workcell authorities, interaction ABox |
| 4.1 | Document source index and PA-authored questions |
| 4.2 | CAD/RGB-D preprocessing, segmentation/crops, morphological review, all-candidate CAD measurement, same-view layout, calibrated location conversion |
| 4.3 | Typed evidence, hashes, uncertainty and provider contracts |
| 4.4 | PA-authored product goal with deterministic validation, pairwise relationships, required PA arm assignment backed by capability/reachability |
| 5.1 | Exact selected-RA assignment, in-process activation/reuse, paired state/catalog snapshots |
| 5.2 | One direct RA-authored program, minimal contracts, filtered reads and explicit selected dependencies |
| 5.3–5.5 | Supplemental PA geometry, measured RA context, audited selected calculations, private Cartesian validation and bounded RA revision; no robot motion |
| Run in Gazebo | Separate simulation execution of a saved validated program, automatic CAD/instance binding, command feedback, stopping and execution records |
| Later | Hardware execution, force/contact validation, independently observed assembly success and broader assembly families |

## Robot-context parameter-service collision (2026-09-08)

The shared [dual_moveit_gazebo.launch.py](../../ros2/cais_lab_robotics/launch/dual_moveit_gazebo.launch.py) no longer supplies `name='move_group'` to the MoveIt `Node`. The executable keeps `/move_group`; its internal helper keeps `/moveit_simple_controller_manager`. The removed global remap made both nodes advertise the same parameter services. In [run_0006](contexts/interaction_a2a6cd2e885943ec9dfca72ad3fc28ed/composition/refinement_runs/run_0006/result.json), consecutive captures returned different parameter sets and stopped as `stale` with `model_parameters_sha256` differing. That diagnostic did not establish physical robot motion.

The repair adds the explanatory comment and `annotations` future import. Parameter contents, comparison fields and thresholds, PA/RA behavior, public interfaces and saved interactions remain unchanged. No refactoring was performed.

Verification for this repair:

- The isolated parameter-service fixture kept the main and helper service names distinct. All 24 repeated reads from the unchanged main node agreed; an explicit parameter change remained detectable.
- `poetry run pytest -q cais_spade_llm/spec2primitives/tests/test_primitive_refinement.py -k robot_comparison`: **3 passed, 43 deselected**. Touched-file syntax checks and `git diff --check` passed.
- `make bootstrap-gazebo`: **16 packages finished**. The installed launch file matched the corrected source byte-for-byte.
- Dual Gazebo was already stopped and was started through its existing UI control. **Compose Primitive Program** then created [run_0007](contexts/interaction_a2a6cd2e885943ec9dfca72ad3fc28ed/composition/refinement_runs/run_0007/result.json). Its [first robot-context capture](contexts/interaction_a2a6cd2e885943ec9dfca72ad3fc28ed/composition/refinement_runs/run_0007/robot_context_0001.json) succeeded, validation ran, and PA completed one batch using five operations. The run stopped as `budget_exhausted` at the unchanged 300-second deadline during RA revision, before a second capture. Live comparison across decisions therefore remains unconfirmed. No robot motion was executed, and assembly validation remains incomplete.

See [the comparison policy and deployment steps](VALIDATION_AND_REVISION.md#finite-modeled-validation). The fixture establishes the service contract; it does not prove complete live refinement, geometry availability or assembly success.

## Grounding changes

`ground_product_context` asks PA for one complete `target_feature`. Current states describe the product now; desired states describe its required assembled condition. Association `state_names` identifies relationship membership independently of endpoint observation bindings. Thus two currently observed objects may anchor a desired relationship.

The proposal permits zero or more pairwise `assembly_feature_association` entries. Each endpoint's binding fields are both valid or both null. Unbound endpoints retain document-only knowledge; PA remains responsible for whole-goal completeness and task-role interpretation. Owners are reused only by exact identity. Each association becomes its own individual under its `Assembly`; the TBox cardinality and symbols stay unchanged. Assertion count is variable.

## PA investigation and deterministic correction

PA controls retrieval, interpretation, current/desired state membership and complete goal coverage. The host validates structure, exact references, evidence hashes and ontology constraints. It returns generic validation feedback rather than a task answer. A moving part's current location is not automatically a destination; PA must justify the observed reference's role in the desired outcome.

The mandatory second semantic reviewer and its runtime contracts are removed. Observation morphology review and document VLM tools remain evidence producers. Deterministic validation cannot establish semantic correctness or absence of bias.

One configured investigation budget defaults to 24 evidence operations and 6 proposals. Corrections reuse the pinned observation and accumulate operation usage. Budget exhaustion and repeated invalid proposals without progress stop with explicit incomplete diagnostics; only genuine requirement ambiguity asks the user. Rejected assertions never merge. Source uncertainty is derived from typed evidence and preserved beside affected claims.

## Required resource assignment

`_complete_resource_assignment` derives every grounded coordinate-bearing current and desired state reference. It does not select the first relationship or require a unique Cartesian pair. Other state values remain semantic evidence.

PA calls `check_reachability` for every configured capable arm using the same grounded current and desired locations, then chooses an arm with an accepted check. An omitted check permits one correction that lists unchecked resources and completed results. Checks are reused within that allocation attempt; continued omission returns `invalid_resource_selection`. Unavailable planning remains distinct from rejected planning. Checks use calibrated locations and the selected simulated robot’s live MoveIt state, joint limits and collision scene; all grounded locations must be covered. PA prefers the accepted arm with the smaller mean distance from its live TF base to the current locations; choosing a farther arm requires supporting task evidence. Equal or unavailable measurements retain neutral selection. The host measures and validates evidence without substituting its own choice.

`config/resource_base_frames.json` maps `xarm6` to `xarm6_link_base` and `ur5e` to `ur5e_base_link`. The owned read-only TF adapter supplies each base in the grounded target frame. Allocation tool results add proximity, exact location handles and base read time; completion validates finite coordinates, frame agreement and recomputed distances. The reachability record contract stays unchanged, and missing TF never becomes a rejection or zero distance.

Each reachability request carries the configured robot planning profile. MoveIt position planning does not start, reuse, or contact a RobotAgent. Selected-RA activation remains an explicit context-capture action after grounding completion.

Commit revalidates evidence and adds the four existing `processExecution`/resource assertions. Missing locations or failed assignment leave Phase 4 incomplete with the grounded result visible. Current Phase 4 does not invoke Cartesian planning or manufacture an allocation completion.

## Current format and recovery boundary

Each record type has one current format without a version marker: proposal with `grounding_evidence` → reachability → selection → completion → RA envelope. Accepted proposals pin the precommit ProductContextView, source and typed-artifact hashes. There are no semantic-review links or compatibility readers.

Completion pins proposal, selection, capability snapshots, evidence, assignment delta and final context. Recovery validates deterministic proposal/evidence lineage, persisted MoveIt results and all-arm coverage through pinned tool-call references; it does not request new planning. Reviewer-bearing and other incompatible records require “Start a fresh interaction”. Saved interactions remain untouched. Initial and clarification-resumed interactions call the same completion writer.

Explicitly authorized reassignment of a valid current interaction is separate from ordinary recovery. `reassign_completed_interaction` archives the full original beside the active interaction with a `_resource_reassignment_` name excluded from UI discovery; attempt metadata lives under `contexts/_resource_reassignments/`. The sibling location preserves the existing source-cache authority. It removes only the terminal assignment from a staged ontology and reuses the accepted proposal, original presentation and calibrated locations. Fresh PA allocation records the optional user constraint, checks every arm, and writes selection, assignment, ProductContextView and completion through existing validators. Activation requires a matching accepted selection and unchanged original files. Failures preserve the active original. Fresh selected-RA state/catalog capture follows activation; composition and robot motion are not invoked. An unconstrained `activate=False` trial remains separate from the user-requested replacement.

## UI, Phase 5, and validation limits

Live activity shows shared evidence-operation usage, proposal attempts, elapsed time, validation feedback and stop reasons. The current timeline ends with Arm assigned → Product grounding and arm assignment complete. Operator labels describe RobotAgent context capture and primitive composition; implementation phase references remain in documents and internal identifiers. Each exact grounded state statement appears once, with a four-line preview and “Show more”. Caveat counts remain visible; full caveats, exact references, endpoints and raw records expand inside responsive cards. The Desired State gallery contains only directly bound images. Cross-state relationship images remain under evidence details, labeled with their source state. Missing desired images are stated explicitly. A resource comparison distinguishes reachable, planning rejected, unavailable and unchecked arms.

Phase 5.1 consumes current completion and writes the assignment envelope before contacting only the assigned RA. Existing readiness/context-only startup behavior remains. Restart appends paired snapshots under the same assignment. Composition reconstructs target/typed values directly from Phase 4 evidence and captured RA context. Historical `PrimitiveProgramDraft` records remain unchanged and are excluded from new composition inputs.

**Compose Primitive Program** permits at most 12 read-only evidence requests followed by a submission or stop. RA can request exact fields of completion-pinned records and filtered, paginated accepted ontology assertions. It chooses primitive order, intermediate movements and available parameters together, authoring `primitive_steps` with literals, `value_ref` bindings and `result_ref` dependencies on earlier declared outputs. Required parameters may be omitted; missing geometry must not prevent a sequence proposal. The host checks supplied structure and references without generating or repairing steps. A submitted candidate ends one authoring decision. Revisions inside the same bounded run receive that run's preceding candidate, supplemental evidence and findings; independent new runs exclude historical programs and traces.

Requests, exchanges, rejected candidates, `unsupported` results and budget/service failures are recorded under `composition/primitive_program_candidates/`. Each request pins completion, assignment, state and catalog records directly. Snapshots with complete `parameter_schemas` and `result_schemas` are required. The UI renders one compact numbered program with omitted required parameters as `<unbound>`; context, raw records and traces expand on demand. Blocking evidence checks run outside the UI event loop, and duplicate Compose clicks cannot create another in-flight attempt. The first `proposed` candidate remains unvalidated. The refinement flow subsequently checks complete selected bindings, helper results, modeled custody, ordered Cartesian paths and final seating; only a complete recorded pass yields `validated_for_declared_scope`. Focused mocked tests establish these software boundaries, not live model accuracy.

Phase 4 validates robot position plans. It does not certify grasp, orientation, attached-part motion, insertion, force, tolerance, primitive composition correctness, execution, or outcome.

## Bias audit

Real sensor metadata and canonical ordering remain internal; randomized observation handles/order are pinned across retries. PA requests, projected tool exchanges, validation feedback and operation/proposal counters are preserved for inspection. No second model certifies the PA interpretation.

Use [BIAS_VALIDATION.md](BIAS_VALIDATION.md) for controlled input audits, permutation/counterfactual/ambiguity trials and reporting. Offline fixtures validate gates and persistence only. Live scene/model experiments remain separate.

The primitive-input increment also includes the explicitly authorized shared metadata corrections in `function_analyzer.py`, `gazebo_pick_place_controller.py`, and `robot_primitives.py`. Shared agent interfaces, `SystemBridge`, robot actions, recovery sequences and pairwise ontology cardinality remain unchanged. Recognition cannot read simulator identities, world contents, configured object poses, detector responses or evaluator answers.

## Historical grounding handoff

The following handoff records earlier grounding work; its file list and verification counts are historical. The current primitive-input handoff is in [the RA guide](agents/ra/README.md).

### Outcome

Desired relationship membership is separate from observed endpoint state. PA owns complete goals and task roles; host checks deterministic validity. Required arm assignment checks every grounded location. Source caveats and ontology/reference images survive reload; historical results cannot start new RA work.

### Process flow

UI → PA investigation → deterministic validation and bounded correction → ontology commit → PA arm check/selection → assignment/completion → selected-RA context → primitive program proposal.

### Read these locations in order

1. [PA orchestration](agents/pa/production_grounding.py): `ground_product_context`, `_complete_resource_assignment`, `_proposal_state_location_handles`.
2. [Ontology proposal](agents/pa/ontology_grounding.py): proposal validation and association compilation; [evidence validation](agents/pa/grounding_contracts.py): `validate_grounding_evidence`.
3. [Resource commit](agents/pa/resource_grounding.py): `commit_resource_assignment`; [completion recovery](agents/pa/grounding_contracts.py): `_validated_two_decision_completion`.
4. [UI](spec2primitives_ui.py): `_validated_grounding_result`, `_final_grounding_result`; [RA gate](agents/ra/context_handoff.py): `activate_selected_ra_context`.
5. [Bias audit and experiments](BIAS_VALIDATION.md): actual input records, counterfactual matrix, scoring and claim limits.

### Read this test

[test_current_completion_uses_reviewed_destinations_and_reachability](tests/test_pa_completion.py) demonstrates both current endpoint observations, a desired relationship, all-location reachability, either configured arm in a reachable case, and no motion-validation record. Production tests cover bounded correction, no progress, evidence provenance and budget exhaustion.

### Runtime evidence

New runs write proposal/evidence, selection and completion under the paths in [contexts](contexts/README.md). Existing saved runs were not rewritten. Offline fixtures exercise the new chain; live VLM, counterfactual scene and hardware outcome validation have not been performed as part of this change.

### You can ignore

Shared `SystemBridge`, shared ProductAgent/RobotAgent internals, robot execution, evaluator answers are outside the changed active path.

### Refactoring performed

Removed retired format classes, writers, dispatch, compatibility readers and their exclusive helpers. Initial and clarification-resumed interactions use one completion writer. Completion reuses one grounded-location validator for the selected and unselected arm records. The owned MoveIt adapter retains only the current position-planning route.

### Verification and intentionally unchanged behavior

Historical verification before removal of the semantic reviewer: focused offline batches passed: completion/interaction/clarification (64), contract integration (75), review/segmentation (57), resource/configuration (41), RA (56), UI/document evidence (37), and geometry (39). These batches overlap. All 75 production tests passed using a temporary inline-worker pytest harness because even a minimal `asyncio.to_thread` example hangs in this environment; production asynchronous dispatch remains unchanged. The allocation matrix covers nine result/correction cases in both resource orders.

Scoped compilation, Ruff undefined-name/import checks, `poetry check`, and `git diff --check` passed. Poetry reported existing metadata deprecation warnings. Calibration values, IDs, frames and validity intervals are unchanged; all four affected provenance hashes match. Offline NiceGUI rendering covers 320- and 1280-pixel card containers; no browser engine was available to verify actual pixel overflow. Live VLM, MoveIt/ROS and hardware validation were not performed. Offline contracts do not establish model accuracy or absence of bias. Shared runtime interfaces, formal symbols, evidence integrity, non-executing RA boundaries and saved interaction bytes remain unchanged.

### Changed-file index

The following links cover the implementation, regression coverage and framework documentation for this change, including related local work already present when implementation resumed.

- [AGENTS.md](AGENTS.md)
- [ASSEMBLY_ONTOLOGY.md](ASSEMBLY_ONTOLOGY.md)
- [BACKGROUND_LITERATURE_REVIEW_BRIEF.md](BACKGROUND_LITERATURE_REVIEW_BRIEF.md)
- [BIAS_VALIDATION.md](BIAS_VALIDATION.md)
- [ICRA_SCOPE.md](ICRA_SCOPE.md)
- [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md)
- [README.md](README.md)
- [RESEARCH_POSITIONING.md](RESEARCH_POSITIONING.md)
- [adapters/README.md](adapters/README.md)
- [adapters/ui_runtime.py](adapters/ui_runtime.py)
- [agents/README.md](agents/README.md)
- [agents/pa/README.md](agents/pa/README.md)
- [agents/pa/__init__.py](agents/pa/__init__.py)
- [agents/pa/context_assessment.py](agents/pa/context_assessment.py)
- [agents/pa/context_interaction.py](agents/pa/context_interaction.py)
- [agents/pa/grounding_contracts.py](agents/pa/grounding_contracts.py)
- [agents/pa/ontology_grounding.py](agents/pa/ontology_grounding.py)
- [agents/pa/production_grounding.py](agents/pa/production_grounding.py)
- [agents/pa/resource_grounding.py](agents/pa/resource_grounding.py)
- [agents/ra/README.md](agents/ra/README.md)
- [agents/ra/context_handoff.py](agents/ra/context_handoff.py)
- [agents/ra/composition_context.py](agents/ra/composition_context.py)
- [agents/ra/primitive_composition.py](agents/ra/primitive_composition.py)
- [cases/README.md](cases/README.md)
- [contexts/README.md](contexts/README.md)
- [evaluations/README.md](evaluations/README.md)
- [evaluations/ground_truth/README.md](evaluations/ground_truth/README.md)
- [references/README.md](references/README.md)
- [references/products/README.md](references/products/README.md)
- [references/resources/README.md](references/resources/README.md)
- [references/resources/primitive_catalogs/README.md](references/resources/primitive_catalogs/README.md)
- [schemas/README.md](schemas/README.md)
- [spec2primitives_ui.py](spec2primitives_ui.py)
- [tests/README.md](tests/README.md)
- [tests/fixtures/README.md](tests/fixtures/README.md)
- [tests/pa_grounding_test_support.py](tests/pa_grounding_test_support.py)
- [tests/test_pa_completion.py](tests/test_pa_completion.py)
- [tests/test_pa_production_grounding.py](tests/test_pa_production_grounding.py)
- [tests/test_pa_resource_grounding.py](tests/test_pa_resource_grounding.py)
- [tests/test_pa_ui_connection.py](tests/test_pa_ui_connection.py)
- [tests/test_ra_context_handoff.py](tests/test_ra_context_handoff.py)
- [tests/test_rgbd_segmentation.py](tests/test_rgbd_segmentation.py)
- [tools/README.md](tools/README.md)
- [tools/document_evidence/README.md](tools/document_evidence/README.md)
- [tools/observation_presentation.py](tools/observation_presentation.py)
- [tools/rgb_d_cad_grounding/README.md](tools/rgb_d_cad_grounding/README.md)
- [tools/rgb_d_cad_grounding/observation_review.py](tools/rgb_d_cad_grounding/observation_review.py)
- [adapters/in_process_robot_agent.py](adapters/in_process_robot_agent.py)
- [adapters/moveit_plan_only.py](adapters/moveit_plan_only.py)
- [agents/pa/presentation_records.py](agents/pa/presentation_records.py)
- [agents/pa/product_context.py](agents/pa/product_context.py)
- [agents/ra/__init__.py](agents/ra/__init__.py)
- [agents/ra/feasibility_validation.py](agents/ra/feasibility_validation.py)
- [config/camera_to_world_calibration.py](config/camera_to_world_calibration.py)
- [config/gazebo_camera_to_world_calibration.json](config/gazebo_camera_to_world_calibration.json)
- [config/gazebo_camera_to_world_calibration_provenance.json](config/gazebo_camera_to_world_calibration_provenance.json)
- [config/model_runtime.json](config/model_runtime.json)
- [config/model_runtime.py](config/model_runtime.py)
- [config/workcell_profile.json](config/workcell_profile.json)
- [config/workcell_profile.py](config/workcell_profile.py)
- [ontology/resource_registry.py](ontology/resource_registry.py)
- [ontology/workcell.py](ontology/workcell.py)
- [tests/test_cad_pose_estimation.py](tests/test_cad_pose_estimation.py)
- [tests/test_cad_size_correspondence.py](tests/test_cad_size_correspondence.py)
- [tests/test_camera_to_world_calibration_config.py](tests/test_camera_to_world_calibration_config.py)
- [tests/test_document_interpretation.py](tests/test_document_interpretation.py)
- [tests/test_model_runtime_config.py](tests/test_model_runtime_config.py)
- [tests/test_moveit_plan_only.py](tests/test_moveit_plan_only.py)
- [tests/test_pa_context_interaction.py](tests/test_pa_context_interaction.py)
- [tests/test_pa_grounding_contracts.py](tests/test_pa_grounding_contracts.py)
- [tests/test_predefined_workcell.py](tests/test_predefined_workcell.py)
- [tests/test_resource_registry.py](tests/test_resource_registry.py)
- [tests/test_robot_frame_conversion.py](tests/test_robot_frame_conversion.py)
- [tools/document_evidence/__init__.py](tools/document_evidence/__init__.py)
- [tools/document_evidence/interpreter.py](tools/document_evidence/interpreter.py)
- [tools/rgb_d_cad_grounding/__init__.py](tools/rgb_d_cad_grounding/__init__.py)
- [tools/rgb_d_cad_grounding/candidate_layout.py](tools/rgb_d_cad_grounding/candidate_layout.py)
- [tools/rgb_d_cad_grounding/diagnostic.py](tools/rgb_d_cad_grounding/diagnostic.py)
- [tools/rgb_d_cad_grounding/frame_conversion.py](tools/rgb_d_cad_grounding/frame_conversion.py)
- [tools/rgb_d_cad_grounding/pose_estimation.py](tools/rgb_d_cad_grounding/pose_estimation.py)
- [tools/rgb_d_cad_grounding/preprocessor.py](tools/rgb_d_cad_grounding/preprocessor.py)
- [tools/rgb_d_cad_grounding/segmenter.py](tools/rgb_d_cad_grounding/segmenter.py)
- [tools/rgb_d_cad_grounding/size_correspondence.py](tools/rgb_d_cad_grounding/size_correspondence.py)

## Correct primitive inputs: historical increment

Phase 4 remains the source of observed product facts and desired assembly relationships. Phase 5 composes using those records. At that increment, target calculation was deferred. The current bounded validation flow is described below and in [VALIDATION_AND_REVISION.md](VALIDATION_AND_REVISION.md).

- Fresh captures retain authoritative contracts and add configuration-sourced planning frame, controlled-link and TCP names. These names do not establish measured transforms.
- Composition projects grasp/release conditions/effects to `held_part`; authoritative callable names/parameters and full captured runtime contracts remain intact. The later simulator-identifier increment below excludes `model_name` from new composition interfaces. Recovery examples and decomposition metadata are excluded from initial input and record reads.
- Catalog generation preserves nested schemas. `x-grounding-required` and `x-grounding-fields` distinguish calculation inputs from signature-required arguments. Existing recovery calls do not acquire new required geometry arguments.
- Supplied symbols, types, bounds, array lengths and reference paths are checked. Partial objects remain proposals. A derived `binding_issues` view identifies missing, incompatible, unverified and deferred data without changing the program or saving a replacement record.
- Frame agreement, object-versus-EE semantics, raw CAD inputs and unresolved destinations are reported. Historical interfaces also retain controller-identifier findings. No transform, source substitution or target helper is invoked.
- Output extraction retains returned orientation and conditional insertion fields. Placement `target_pose` may be pre-insertion; no missing insertion output is fabricated.
- The UI retains one numbered program, display-only `<unbound>` markers, a short binding summary and expandable records. Blocking checks and refresh run outside the event loop; in-flight Compose clicks remain excluded.

This earlier increment preceded the now-implemented [validation/revision loop](VALIDATION_AND_REVISION.md). The [evaluation protocol](COMPOSITION_EVALUATION.md) remains experimental work. Offline tests do not demonstrate live RA accuracy or successful assembly. Existing snapshots and saved programs keep their original bytes; corrected declarations require a fresh context capture.

### Verification of the primitive-input increment

Focused offline batches passed: 50 composition/catalog/UI cases; 23 adapter and Compose callback cases; 8 geometry/frame/output cases; and 1 native recovery catalog/reference-card case. Batches overlap. Native catalog tests keep existing recovery required arguments and full runtime conditions/effects, while composition tests inspect the restricted projection. Real worker threads verify responsive validation/refresh and duplicate-click handling; these tests ran outside the sandbox after the sandbox run stalled.

Scoped Ruff import/undefined-name/syntax checks, `poetry check`, and `poetry run python -m compileall -q cais_spade_llm ros2` passed. Poetry reported existing metadata deprecation warnings. The default `git diff --check` flags CRLF on newly added lines of the already-CRLF `function_analyzer.py`. Preserving its line endings, `git -c core.whitespace=trailing-space,space-before-tab,cr-at-eol diff --check` passed. Local links in all 28 affected maintained Markdown files resolved.

The controller's executable AST is unchanged after removing docstrings. The shared robot primitive functions changed only in the two output extractors, with declaration/output-preservation helpers added; recovery sequences are unchanged. All 179 files of the saved medium-gear interaction retained their before-change hashes, with no additions or removals. No live model request, new context capture, geometry calculation, ROS/MoveIt validation or robot motion was performed. Physical composition quality and publication claims require the documented experiments.

A final read-only reload of the actual saved program exposed an empty JSON Pointer case in the new report. The checker now accepts whole-record references just like the composer; two targeted regressions passed. Reload returns the same 10-step `proposed` program for `xarm6@localhost`, with step 4 reporting the missing controller identifier. All 179 saved files were checked again and remain unchanged.

## Simulator identifier separation: implemented increment

New RA composition interfaces omit `model_name` from parameters, nested helper schemas, outputs, required lists and custody effects. `part_name`, primitive names, geometry requirements and movement parameters remain intact. The prompt distinguishes the composition interface from directly callable runtime signatures. The separate [Run in Gazebo adapter](VALIDATION_AND_REVISION.md#run-in-gazebo) binds the recognized instance after composition; composition performs no simulator lookup or execution.

New proposals reject supplied `model_name`, including nested and referenced objects, and references to removed helper outputs. Rejected submissions remain exact in the trace. Missing geometry and deferred numerical outputs remain composition findings; a simulator identifier is no longer a new composition input requirement.

Saved attempts use the catalog embedded in their original hash-checked `request.prompt` for validation, diagnostics and UI rendering. That catalog must agree with the pinned context after projection. Old programs may still display identifier arguments or gaps. New attempts use the current projection without receiving prior programs as input. The existing record formats, saved bytes, full runtime snapshots, shared controller and recovery consumers remain unchanged.

Verification: 30 focused offline tests passed in two disjoint 15-case batches covering projection, nested/indirect identifier rejection, removed outputs, unchanged geometry requirements, exact old/new programs, context/trace guards, recovery metadata, UI formatting and worker responsiveness. The sandbox batch timed out; passing batches used normal worker-thread access with mocked RA responses. Scoped Ruff (`F401,F821,E9`), compilation of the four touched Python files and `git diff --check -- .` within Spec2Primitives passed; local links in all 17 affected maintained Markdown files resolved.

Read-only checks reopened both medium-gear runs as unchanged 10-step proposals. All 141 files in `interaction_116f33c8cce64d33b0066891bf228572` and 179 files in `interaction_4178296ec9e04fb19e6e780371fd062b` retained their baseline hashes, as did all three shared source files. New inputs prepared from those contexts exclude `model_name`; neither program was regenerated. No live model request, simulator lookup, helper calculation or robot motion occurred. Broader repository and live physical checks were not run for this composition-interface change.


## Phase 5 bounded validation milestone

### Outcome

The UI now runs RA proposal → selected-dependency gaps → supplemental PA/RA evidence → RA binding/revision → strict target calculation → private motion/model validation → bounded revision or stop. The only accepting run status is `validated_for_declared_scope`; it is a recorded rigid vertical gear geometry/motion verdict, not execution permission or physical success. The shared API and runtime robot actions are unchanged.

### Process flow

`_start_primitive_composition` → `PrimitiveRefinementRuntime.compose` → `author_primitive_program_candidate` → `assess_program_dependencies` → `ProductPrimitiveContextRuntime.investigate` / `capture_validation_context` → `validate_program` → another RA decision or immutable result.

Defaults: three candidate versions, two PA batches, twelve PA evidence operations, five minutes, 32 steps and twelve bounded reads per authoring decision. Read-only worker startup has a separate 30-second allowance; ordinary service/planning limits remain 10/20 seconds. Cancellation tears down the private worker. Unknown coverage, unavailable measurements and missing tolerances cannot produce a pass.

### Read these locations in order

1. [Refinement](agents/ra/refinement.py): `compose` joins duplicate calls; `_run` preserves candidates and coordinates bounded evidence/validation. `_robot_changed` and source checks invalidate stale context.
2. [Dependencies](agents/ra/program_dependencies.py): `assess_program_dependencies` propagates only selected output dependencies and routes requests.
3. [Supplemental PA](agents/pa/primitive_context.py): `investigate` returns explicitly selected approved facts; [geometry](tools/assembly_geometry.py) measures features and derives PA-selected seating relationships. Supplemental [native retrieval](agents/pa/production_grounding.py) validates records without committing Phase 4.
4. [Measured context](adapters/robot_validation_context.py): `capture` reads robot feedback/configuration without action clients. [Strict calculations](adapters/target_calculation.py) evaluate exact helper inputs against predicted preceding state.
5. [Validation](agents/ra/program_validation.py): `validate_program` checks the immutable candidate and returns findings. [Private MoveIt](adapters/isolated_moveit.py) accepts only complete collision-checked Cartesian segments, with execution disabled.
6. [UI](spec2primitives_ui.py): `_start_primitive_composition` displays the first proposal and progress; `_apply_primitive_composition_diagnostic` reopens records. The [full behavior/scope](VALIDATION_AND_REVISION.md) and [experiment protocol](COMPOSITION_EVALUATION.md) separate modeled checks from physical work.

Runtime/config/test files changed by this milestone:

- [../resources/robot/gazebo_pick_place_controller.py](../resources/robot/gazebo_pick_place_controller.py)
- [../resources/robot/target_calculations.py](../resources/robot/target_calculations.py)
- [adapters/in_process_robot_agent.py](adapters/in_process_robot_agent.py)
- [adapters/isolated_moveit.py](adapters/isolated_moveit.py)
- [adapters/robot_validation_context.py](adapters/robot_validation_context.py)
- [adapters/target_calculation.py](adapters/target_calculation.py)
- [adapters/ui_runtime.py](adapters/ui_runtime.py)
- [agents/pa/context_serving.py](agents/pa/context_serving.py)
- [agents/pa/primitive_context.py](agents/pa/primitive_context.py)
- [agents/pa/production_grounding.py](agents/pa/production_grounding.py)
- [agents/ra/parameter_binding.py](agents/ra/parameter_binding.py)
- [agents/ra/primitive_composition.py](agents/ra/primitive_composition.py)
- [agents/ra/program_dependencies.py](agents/ra/program_dependencies.py)
- [agents/ra/program_validation.py](agents/ra/program_validation.py)
- [agents/ra/refinement.py](agents/ra/refinement.py)
- [agents/ra/refinement_records.py](agents/ra/refinement_records.py)
- [config/phase5_validation.json](config/phase5_validation.json)
- [spec2primitives_ui.py](spec2primitives_ui.py)
- [tests/test_pa_production_grounding.py](tests/test_pa_production_grounding.py)
- [tests/test_pa_ui_connection.py](tests/test_pa_ui_connection.py)
- [tests/test_primitive_refinement.py](tests/test_primitive_refinement.py)
- [tools/assembly_geometry.py](tools/assembly_geometry.py)

Affected maintained Markdown guides: [AGENTS.md](AGENTS.md), [ASSEMBLY_ONTOLOGY.md](ASSEMBLY_ONTOLOGY.md), [BACKGROUND_LITERATURE_REVIEW_BRIEF.md](BACKGROUND_LITERATURE_REVIEW_BRIEF.md), [BIAS_VALIDATION.md](BIAS_VALIDATION.md), [COMPOSITION_EVALUATION.md](COMPOSITION_EVALUATION.md), [ICRA_SCOPE.md](ICRA_SCOPE.md), [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md), [README.md](README.md), [RESEARCH_POSITIONING.md](RESEARCH_POSITIONING.md), [VALIDATION_AND_REVISION.md](VALIDATION_AND_REVISION.md), [adapters/README.md](adapters/README.md), [agents/README.md](agents/README.md), [agents/pa/README.md](agents/pa/README.md), [agents/ra/README.md](agents/ra/README.md), [cases/README.md](cases/README.md), [contexts/README.md](contexts/README.md), [evaluations/README.md](evaluations/README.md), [evaluations/ground_truth/README.md](evaluations/ground_truth/README.md), [references/README.md](references/README.md), [references/products/README.md](references/products/README.md), [references/resources/README.md](references/resources/README.md), [references/resources/primitive_catalogs/README.md](references/resources/primitive_catalogs/README.md), [schemas/README.md](schemas/README.md), [tests/README.md](tests/README.md), [tests/fixtures/README.md](tests/fixtures/README.md), [tools/README.md](tools/README.md), [tools/document_evidence/README.md](tools/document_evidence/README.md), [tools/rgb_d_cad_grounding/README.md](tools/rgb_d_cad_grounding/README.md).

### Read this test

[`test_refinement_preserves_first_pass_and_only_ra_supplies_revision`](tests/test_primitive_refinement.py) starts from unbound geometry, records the first proposal, supplies controlled PA evidence, and accepts an RA-authored second version using real pure calculations and mocked Cartesian validation. It checks unchanged Phase 4 bytes and saved reload. Other tests reject carried-part collisions, premature release/pre-insertion, changed meshes, stale context, bad frames/identities, missing tolerances and incomplete coverage.

### Runtime evidence

Two separate live no-motion checks revalidated the existing saved medium-gear proposal without regenerating it:

| Pilot | Measurements | Private worker | Program verdict | Elapsed |
| --- | --- | --- | --- | --- |
| [pilot_0001](contexts/interaction_116f33c8cce64d33b0066891bf228572/composition/validation_pilots/pilot_0001/result.json) | xarm6 joints and world EE/TCP captured | Initial service discovery timed out | `unknown` | 13.59 s |
| [pilot_0002](contexts/interaction_116f33c8cce64d33b0066891bf228572/composition/validation_pilots/pilot_0002/result.json) | xarm6 joints and world EE/TCP captured | Private empty smoke-test scene accepted | `unknown` | 10.71 s |

Both report missing supplemental part/goal/scene/specification roles and robot feedback that became stale during the separate worker startup probe. Empty-scene service availability is not collision coverage for a program. No live PA investigation or RA model revision was performed: Gazebo/MoveIt and XMPP were running, but no active application/selected-RA integration was available to drive that flow. Both used zero model API calls and executed no robot motion. The first failure remains recorded; it was not replaced by the retry.

The independent controlled fixture passes its declared numerical/model chain after an RA-authored revision; this is not a live LLM or assembly-success result. First-pass and revised candidate/report references remain distinct in actual refinement runs. [COMPOSITION_EVALUATION.md](COMPOSITION_EVALUATION.md) specifies the required research measurements.

### You can ignore

- Shared ProductAgent/RobotAgent lifecycle implementations and `SystemBridge`: no interface changes.
- Shared recovery examples and old draft histories: excluded from new initial composition input.
- Evaluator answers and simulator instance identities: unavailable to recognition/refinement.
- Historical handoff test counts above: they describe earlier increments.

### Refactoring performed

The user-approved shared extraction moved numerical pick/place arithmetic into [target_calculations.py](../resources/robot/target_calculations.py); existing controller wrappers call those functions. Runtime signatures, actions and recovery branches remain in place. Focused parity tests compare actual helper outputs with the strict calculation adapter. No shared ProductAgent/RobotAgent refactor was performed. Earlier dirty changes in `function_analyzer.py` and `robot_primitives.py` were preserved byte-for-byte during this milestone.

### Verification and intentionally unchanged behavior

Final focused offline results: 22 refinement tests (131.01 s), 14 UI tests (71.13 s), two PA retrieval/document tests, and four existing context/catalog/recovery tests (29.35 s): **42 distinct tests passed**. The PA cases passed alongside eight additional refinement cases in a 10-case batch (44.17 s); those eight are already included in the 22 total. Tests use mocked models and motion results; parity tests initialize no controller.

Scoped Ruff import/undefined-name/syntax checks passed. All changed Python files passed syntax compilation. `poetry check` passed with existing metadata deprecation warnings. The default `git diff --check` still reports the pre-existing CRLF lines in untouched-this-increment `function_analyzer.py`; `git -c core.whitespace=trailing-space,space-before-tab,cr-at-eol diff --check` passed. Maintained Markdown links resolve. Broader repository tests, full workspace compile/CLI/launch checks and hardware experiments were not run; no corresponding entrypoint or launch behavior changed.

All 1,114 baseline context JSON files retained their original hashes; no baseline file was removed. The two pilots append separate records. Existing programs, original Phase 4 evidence/identifiers, authoritative snapshots and prior local changes remain preserved. No simulator lookup, force/contact validation, robot execution or observed-success claim is added. The application must be restarted to load the new code before a live composition/refinement test.
