# ProductAgent grounding

This package owns the narrow, non-executing Spec2Primitives ProductAgent
connection. It does not start the shared ProductAgent lifecycle or modify the
shared agent.

## Native investigation

`start_pa_context_interaction` records the exact requirement, initializes the
interaction ABox, and asks the production grounding runtime to conduct one PA
investigation. The PA call receives:

- the exact requirement;
- an allowed PPR schema projection;
- the current ABox and required-output projection;
- approved discovery metadata; and
- one controlled native `retrieve(evidence_id)` tool.

PA can retrieve zero or more approved documents, CAD files, or fresh live
observations in any order. Tool results return to the same conversation as
compact typed evidence. PA then directly returns a proposal candidate, a
clarification question, or an insufficient-evidence explanation.

The model never supplies paths, provider IDs, hashes, frames, record types, or
arbitrary source names. Tool calls are resolved and audited by the system. A
malformed, unauthorized, stale, altered, or unavailable handle fails closed.

Clarification is reserved for requirement meaning that approved evidence cannot
resolve after relevant approved evidence has been retrieved and considered. The
runtime rejects premature clarification and questions that delegate a supplied
required output or approved evidence-category choice to the user, returns
prompt-only feedback, and keeps the retry inside the bounded investigation. It
does not supply a requirement interpretation in that feedback.

An answered clarification is evidence from its persisted append-only
`interaction_record/clarification_<question_turn>.json` record. The native
completion contract pins that exact record through `source_refs`; it does not
create or accept a separate clarification alias.

## Evidence processing

`production_grounding.py` exposes only source-level retrieval to PA. The system
performs document extraction, STL measurement, live capture, segmentation,
correspondence, calibration, frame conversion, and coarse resource checks.
Those operations are descriptor-driven services, not additional PA choices.

Document retrieval yields the complete ordered `DocumentOverviewRecord` v3.
CAD retrieval yields `CADMeshRecord`. Observation retrieval yields typed RGB-D,
point-cloud, segmentation, stable crop, and `ObservationCandidateReview`
records. The review describes visible candidates without assigning state, CAD,
process, or resource meaning. The PA binds exactly one supplied candidate to
each state in `state_values`; deterministic CAD-size validation checks the
unchanged current candidate. During allocation, `check_reachability` accepts
only the PA-selected `resource_symbol` and derives both locations from those
already accepted state bindings.

Before retrieval, `EvidencePresentationRecord` maps canonical sources to
randomized opaque handles. Approved CAD names and exact `context_ref` values are
available to PA, while repository paths, Gazebo identities, predetermined
matches, URLs, and semantic camera roles are absent. After semantic grounding,
`AllocationPresentationRecord` pins one randomized capable-resource order and
one shared neutral candidate pool. The same stored order drives the prompt,
tool enums, and response schema; it is audit evidence, not selection priority.

After a proposal, the runtime builds a provisional graph without merging it and
returns deterministic validation feedback without repairing the PA result. The
active production path supplies a fixed `RGBDSegmentationRecord`
required-output projection. PA may retrieve another approved source in the same
logical investigation, but provider-descriptor expansion through
`_producer_descriptors`, `_required_record_plan`, and `_grounding_gap` is not
connected to this path and remains deferred.

## Ontology grounding

`ontology_grounding.py` validates a transient `OntologyGroundingProposal` v8
candidate containing exactly one model-authored `target_feature`. This
increment supports one requirement and one feature. The target contains:

- one evidence-cited `required_process.process_iri` selected from the configured
  process authority;
- one complete evidence-cited `current_state.statement`;
- one complete evidence-cited `desired_state.statement`; and
- zero, one, or multiple `state_values` for each state, each with a unique PA-authored semantic
  name, one accepted typed-record ref, one JSON Pointer, and direct evidence.

PA chooses the process, statement text, state-value count and names, record
refs, field paths, and citations from retrieved evidence. Deterministic code
does not fill those semantic values. It validates process authority, citations,
accepted bindings, exact record hashes, JSON Pointer resolution, nonempty
resolved values, and unique names.

The host generates `feature_0001`, `currentstate_0001`, and
`desiredstate_0001` and compiles exactly seven assertions: all three types,
`specification ppr:defines feature_0001`, the selected process
`ppr:realizes feature_0001`, and the feature's `ppr:hascurrentstate` and
`ppr:hasdesiredstate` links. Rich PA-authored state meaning remains in the
accepted proposal while RDF records the explicit feature-state structure.

If one PA proposal violates these invariants, the runtime preserves the
rejected audit record and returns the exact deterministic validation failure as
prompt-only feedback for a bounded correction round. It does not synthesize a
missing `defines`, `realizes`, or other assertion on PA's behalf.

Structural validity alone does not commit the candidate. A separate model call
produces `TargetFeatureSemanticReview` v2, containing only `verdict` and `gap`.
It checks whether the statement and included values adequately represent the
requirement under the evidence retrieved so far. An incomplete review returns
the concise gap to the bounded PA retrieval/revision loop; it is not persisted
as accepted missing information and no private reasoning is stored.

The accepted v8 proposal creates explicit `currentstate_0001` and
`desiredstate_0001` individuals attached to `feature_0001`. The unresolved
`processExecution` then activates a separate PA allocation call. The call reuses
the two accepted state bindings; PA chooses one capable resource, invokes
`check_reachability(resource_symbol)`, and cites the resulting two-state evidence. The exact
provisional RobotAgent performs plan-only endpoint IK/collision/path validation.
Only its `accepted` verdict allows the runtime to commit the assignment;
rejection returns evidence for another PA choice without host substitution.
The PA may revise the resource, but allocation cannot revise either state image.
Semantic feature/state assertions stay fixed during physical-evidence revision.

`RobotFrameLocationRecord` version 2 is semantically neutral. Its current or
desired meaning comes from the PA state assignment. The architecture does not
use `TargetFeatureGeometryRecord`; verifier-required numeric geometry is
derived on demand from the selected evidence and approved calibration.

## Completion and compatibility

New interactions write append-only native tool audits, direct PA turns,
`EvidencePresentationRecord` v1, `AllocationPresentationRecord` v1,
`ReachabilityCheckRecord` v2, `ResourceSelectionRecord` v4,
`PlanOnlyFeasibilityValidationRecord` v2, `TargetFeatureSemanticReview` v2,
`TypedGroundingContract` v6, and `PAContextGroundingCompletion` v6.
Clarification resumes the same conversation context using the exact persisted
question/reply history. Cancellation invokes no PA call.

The v6 contract and completion pin the selected process, both state IRIs, the
accepted proposal, semantic review, nested evidence, both presentation records,
registry/workcell snapshots, referenced typed records, reachability, exact
RobotAgent endpoint-motion validation, resource selection, assignment delta,
and final ABox.
They do not copy `target_feature`; downstream consumers reconstruct it from the
hash-verified accepted proposal.

Read-only validation remains for ontology proposal v3-v7, document overview
v1, resource selection v1-v3, feasibility validation v1, and completion/session
v2-v5 records. New Phase 5 handoff requires completion v6. New runs never
produce or migrate the old records.

This boundary performs schema-constrained, evidence-backed instance grounding.
Multi-feature requirements, primitive parameter binding, motion execution,
post-process observation, and feature-state update are not implemented here.
