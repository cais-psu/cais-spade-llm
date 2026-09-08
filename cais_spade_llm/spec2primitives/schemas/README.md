# Schemas

Contracts live in their owning package modules. This directory indexes contracts, not experiment results. Fixed ontology identifiers and pairwise cardinality remain unchanged.

## Current records

| Record | Owner |
| --- | --- |
| `OntologyGroundingProposal` | `agents/pa/ontology_grounding.py` |
| `ObservationPresentationRecord` | `tools/observation_presentation.py` |
| `EvidencePresentationRecord`, `AllocationPresentationRecord` | `agents/pa/presentation_records.py` |
| `ReachabilityCheckRecord` | `agents/pa/resource_grounding.py` |
| `ResourceSelectionRecord` | same module |
| `PAContextGroundingCompletion` | `agents/pa/grounding_contracts.py` |
| `SelectedRAAssignmentEnvelope` | `agents/ra/context_handoff.py` |
| `PrimitiveCompositionRequest`, `PrimitiveCompositionExchange`, `PrimitiveProgramCandidate` | `agents/ra/primitive_composition.py` |

## Proposal and evidence

Proposal preserves one `target_feature` with evidence-cited required process, current/desired statements and `state_values`. Each value has an exact PA-authored name, one typed record/JSON-pointer reference and citations. Multiple values may reference one record. Names are not prescribed by a host enum.

For assembly, `assembly_feature_association` is a collection. Each element has `assembly`, `state_names`, exactly two `assembly_features`, and citations. Endpoint binding fields are both valid or both null. Relationship membership is independent of observed endpoint bindings. Owners are reused only by exact identity; each association compiles as a separate individual. See [the exact shape](../ASSEMBLY_ONTOLOGY.md).

`grounding_evidence` pins the precommit `ProductContextView`, typed artifacts and exact source hashes. Deterministic validation checks artifact coverage, references and committed delta linkage. Source uncertainty is derived from typed evidence and retained with exact references. No semantic-review record or review linkage fields remain.

Configured `grounding_limits` default to 24 evidence operations and 6 proposals across one investigation. Existing PA requests, tool records and terminal output retain host-owned `grounding_progress` counters, validation feedback and stop reasons. They do not copy evidence into a model-authored contract. Budget exhaustion and repeated invalid proposals without progress leave grounding incomplete.

## Assignment and completion

Selection uses current/desired `state_locations` lists covering every grounded coordinate-bearing reference, deduplicated exactly per state. Other typed values remain semantic evidence. There is no forced first association or unique Cartesian pair. PA checks every capable arm against the same grounded locations and cites an accepted check for its choice. One correction lists unchecked resources and existing results; continued omission returns `invalid_resource_selection`. Completed checks are reused within the attempt. Either passing arm remains eligible, independent of presentation order. Selection pins the proposal without manufacturing `robot_agent_validation_ref` fields.

Commit adds the four existing process-execution/resource assertions. Completion pins proposal/evidence, selection/reachability, source/tool evidence, presentations, capability snapshots, assignment delta and final context. It reports `validation_scope: moveit_state_location_reachability`, `motion_validation_performed: true`, and `motion_executed: false`.

Checked constraints are process capability, live robot state, joint limits, collision-aware position plans and all grounded state locations. Grasp, tool orientation, attached-part collision geometry, insertion, force, tolerance and primitive composition remain unvalidated. The reachability record retains the exact planning request, result, joint paths, start state and timestamp. Recovery rechecks all lineage, controller settings, complete capable-arm coverage through persisted tool calls, and request/evidence bindings. A grounded but unassigned result produces no current completion or RA authorization.

## Source and downstream contracts

- Documents: deterministic `DocumentSourceIndexRecord`, optional PA-authored `DocumentQueryRecord`, retained overview caches/readers.
- Geometry: `CADMeshRecord`, point-cloud records, `RGBDSegmentationRecord`, morphological `ObservationCandidateReview` with source/crop hashes and uncertainty.
- `CADSizeCorrespondenceRecord` contains all measured candidates without a winner; model order is randomized independently of canonical storage order.
- `CandidateSpatialRelationRecord` contains same-view geometry without semantic roles.
- Calibration and `RobotFrameLocationRecord` pin approved transforms and translated references without assigning state meaning.
- Pose diagnostics consume current candidate measurements; automatic size-based correspondence and the superseded Cartesian allocation path are removed.

`ProductContextView` links ABox meaning to validated typed records and uncertainty. Numeric payloads remain outside RDF where no TBox property represents them. No `TargetFeatureGeometryRecord`, new duplicate `TypedGroundingContract`, or `TaskTransitionContract` is introduced.

There are no format-version fields, compatibility aliases, migrations, or historical readers. Saved interactions remain untouched. Incompatible shapes fail with “Start a fresh interaction”; their fields or validation rules are never inferred or relaxed.

Phase 5.1 paired context and direct RA-authored `primitive_steps` proposals exist. Each composition request pins completion, assignment, robot-state and primitive-catalog records through `context_refs`; supplied parameters are validated and ungrounded parameters may remain omitted. Historical draft-dependent attempts remain untouched and are excluded from new inputs. The bounded loop adds `PrimitiveContextRequest`, `PrimitiveRefinementContext`, `RobotValidationContext`, `PrimitiveCalculationRecord`, `PrimitiveValidationReport` and immutable run events/results. Complete execution-identifier binding, physical validation, execution and observed updates remain future work. Offline contract tests and [live bias experiments](../BIAS_VALIDATION.md) establish different evidence.

## Composition declarations and derived findings

`parameter_schemas` retains nested `properties`, `items`, bounds and declaration metadata. `typed_parameters.required` retains callable/catalog requirements. `x-grounding-required` marks an optional argument needed to ground the calculation, and `x-grounding-fields` identifies needed nested fields without adding required recovery arguments. `x-frame-source` and `x-binding-role` state coordinate and identifier meanings. The model-facing grasp/release conditions/effects contain only `held_part`; authoritative snapshots retain the complete runtime definitions.

The composition projection removes `model_name` from typed parameters/results, nested schemas, required/grounding lists and custody effects. Remaining declarations keep their requirements. This interface describes program composition; the captured runtime signature still requires adapter-supplied execution bindings. New proposals reject `model_name` even inside partial objects, arrays or resolved evidence objects, and cannot reference the removed helper outputs.

No new record format or version field is added. The history reader extracts each attempt's catalog from the `COMPOSITION_INPUT` JSON already saved in `request.prompt`, validates its structure and agreement with the pinned context after projection, and uses it for candidate validation, diagnostics and display. It retains original context/trace hashes and exact RA-submission checks. New attempts always use the current projection.

Fresh `RobotStateSnapshot.robot_state.motion_context` contains configuration `source`, exact `frame_id`, `ee_link` and `tcp_link`, each unavailable name null. This is configuration evidence, not a measured tool transform. `move_cartesian` consumes the configured planning frame. World geometry does not receive an automatic base-frame conversion.

`read_primitive_composition_diagnostic` derives `binding_issues`: `{step_index, parameter_path, status, message}`. Status is `missing`, `incompatible`, `unverified`, or `deferred`. Missing geometry does not invalidate a sequence proposal. Supplied invalid types, bounds, malformed/unauthorized refs and unknown primitives still fail. The report does not mutate `primitive_steps`, add sources or call helpers.

Pick results declare scalar pick context as well as `approach_pose`/`target_pose`. Place results also declare conditional `pre_insert_pose`, `insert_pose`, insertion direction and target references; orientation exists only when returned. A declared conditional result is not a value or a successful insertion. Runtime `model_name` requires an execution-adapter source, separate from recognized `part_name`; it is absent from new composition reports. Saved attempts retain their original interface and findings. See [modeled checks and future execution work](../VALIDATION_AND_REVISION.md).

## Refinement records

`PrimitiveContextRequest` pins affected steps, quantities/schemas, routing and issued evidence. `PrimitiveRefinementContext` pins the current run request, preceding candidate, supplemental records, measured context and findings. `RobotValidationContext` retains timestamps, joints, EE/TCP transforms and configuration/model hashes; its large model parameters are excluded from RA reads. `PrimitiveCalculationRecord` stores selected resolved arguments and actual outputs separately from the program. `PrimitiveValidationReport` records `passed`/`failed`/`unknown`, coverage, checked prefix steps, calculations and predicted final custody/pose. `PrimitiveRefinementResult` pins all decisions, candidates, reports, progress events, counters and stopping reason. Every run-owned record has a host-derived fingerprint. Program `proposed` and run `validated_for_declared_scope` remain different claims.
