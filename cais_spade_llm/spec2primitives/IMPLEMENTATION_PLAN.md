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
| 5.2A | RA-authored unbound structural draft per context pair |
| Later | SPADE delivery, binding/context exchange, executable validation, execution and observed outcomes are not implemented |

## Grounding changes

`ground_product_context` asks PA for one complete `target_feature`. Current states describe the product now; desired states describe its required assembled condition. Association `state_names` identifies relationship membership independently of endpoint observation bindings. Thus two currently observed objects may anchor a desired relationship.

The proposal permits zero or more pairwise `assembly_feature_association` entries. Each endpoint's binding fields are both valid or both null. Unbound endpoints retain document-only knowledge; PA remains responsible for whole-goal completeness and task-role interpretation. Owners are reused only by exact identity. Each association becomes its own individual under its `Assembly`; the TBox cardinality and symbols stay unchanged. Assertion count is variable.

## PA investigation and deterministic correction

PA controls retrieval, interpretation, current/desired state membership and complete goal coverage. The host validates structure, exact references, evidence hashes and ontology constraints. It returns generic validation feedback rather than a task answer. A moving part's current location is not automatically a destination; PA must justify the observed reference's role in the desired outcome.

The mandatory second semantic reviewer and its runtime contracts are removed. Observation morphology review and document VLM tools remain evidence producers. Deterministic validation cannot establish semantic correctness or absence of bias.

One configured investigation budget defaults to 24 evidence operations and 6 proposals. Corrections reuse the pinned observation and accumulate operation usage. Budget exhaustion and repeated invalid proposals without progress stop with explicit incomplete diagnostics; only genuine requirement ambiguity asks the user. Rejected assertions never merge. Source uncertainty is derived from typed evidence and preserved beside affected claims.

## Required resource assignment

`_complete_resource_assignment` derives every grounded coordinate-bearing current and desired state reference. It does not select the first relationship or require a unique Cartesian pair. Other state values remain semantic evidence.

PA calls `check_reachability` for every configured capable arm using the same grounded current and desired locations, then chooses an arm with an accepted check. An omitted check permits one correction that lists unchecked resources and completed results. Checks are reused within that allocation attempt; continued omission returns `invalid_resource_selection`. Unavailable planning remains distinct from rejected planning. Checks use calibrated locations and the selected simulated robot’s live MoveIt state, joint limits and collision scene; all grounded locations must be covered. Either eligible arm is acceptable without an invented advantage. The host cannot substitute or prefer an arm.

Each reachability request carries the configured robot planning profile. MoveIt position planning does not start, reuse, or contact a RobotAgent. Selected-RA activation remains an explicit context-capture action after grounding completion.

Commit revalidates evidence and adds the four existing `processExecution`/resource assertions. Missing locations or failed assignment leave Phase 4 incomplete with the grounded result visible. Current Phase 4 does not invoke Cartesian planning or manufacture an allocation completion.

## Current format and recovery boundary

Each record type has one current format without a version marker: proposal with `grounding_evidence` → reachability → selection → completion → RA envelope. Accepted proposals pin the precommit ProductContextView, source and typed-artifact hashes. There are no semantic-review links or compatibility readers.

Completion pins proposal, selection, capability snapshots, evidence, assignment delta and final context. Recovery validates deterministic proposal/evidence lineage, persisted MoveIt results and all-arm coverage through pinned tool-call references; it does not request new planning. Reviewer-bearing and other incompatible records require “Start a fresh interaction”. Saved interactions remain untouched. Initial and clarification-resumed interactions call the same completion writer.

## UI, Phase 5, and validation limits

Live activity shows shared evidence-operation usage, proposal attempts, elapsed time, validation feedback and stop reasons. The current timeline ends with Arm assigned → Product grounding and arm assignment complete. Operator labels describe RobotAgent context capture and structural primitive drafting; implementation phase references remain in documents and internal identifiers. Each exact grounded state statement appears once, with a four-line preview and “Show more”. Caveat counts remain visible; full caveats, exact references, endpoints and raw records expand inside responsive cards. The Desired State gallery contains only directly bound images. Cross-state relationship images remain under evidence details, labeled with their source state. Missing desired images are stated explicitly. A resource comparison distinguishes reachable, planning rejected, unavailable and unchecked arms.

Phase 5.1 consumes current completion and writes the assignment envelope before contacting only the assigned RA. Existing readiness/context-only startup behavior remains. Restart appends paired snapshots under the same assignment. Phase 5.2A reconstructs the target/typed values and asks that RA for exact catalog symbols with no tools or bindings. `PrimitiveProgramDraft` remains unchanged.

Phase 4 validates robot position plans. It does not certify grasp, orientation, attached-part motion, insertion, force, tolerance, primitive composition correctness, execution, or outcome.

## Bias audit

Real sensor metadata and canonical ordering remain internal; randomized observation handles/order are pinned across retries. PA requests, projected tool exchanges, validation feedback and operation/proposal counters are preserved for inspection. No second model certifies the PA interpretation.

Use [BIAS_VALIDATION.md](BIAS_VALIDATION.md) for controlled input audits, permutation/counterfactual/ambiguity trials and reporting. Offline fixtures validate gates and persistence only. Live scene/model experiments remain separate.

All changes remain inside Spec2Primitives. Shared agents, `SystemBridge`, execution and pairwise ontology cardinality remain unchanged. Recognition cannot read simulator identities, world contents, configured object poses, detector responses or evaluator answers.

## Code-reading handoff

### Outcome

Desired relationship membership is separate from observed endpoint state. PA owns complete goals and task roles; host checks deterministic validity. Required arm assignment checks every grounded location. Source caveats and ontology/reference images survive reload; historical results cannot start new RA work.

### Process flow

UI → PA investigation → deterministic validation and bounded correction → ontology commit → PA arm check/selection → assignment/completion → selected-RA context → unbound draft.

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
- [agents/ra/primitive_draft.py](agents/ra/primitive_draft.py)
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
