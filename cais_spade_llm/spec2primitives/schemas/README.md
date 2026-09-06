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
| `PrimitiveProgramDraft` | `agents/ra/primitive_draft.py` |

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

Phase 5.1 paired context and Phase 5.2A structural drafts exist. `MissingContextBatch`, `CompositionContextBundle`, bound `primitive_steps`, executable validation, execution and observed updates remain future contracts. Offline contract tests and [live bias experiments](../BIAS_VALIDATION.md) establish different evidence.
