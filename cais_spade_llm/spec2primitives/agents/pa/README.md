# ProductAgent grounding

This package owns the narrow, non-executing Spec2Primitives ProductAgent
connection. It does not start the shared ProductAgent lifecycle or modify the
shared agent.

## Native investigation

`start_pa_context_interaction` records the exact requirement, initializes the
interaction ABox, and asks the production grounding runtime to conduct the two
Phase 4 PA decisions. The first call receives:

- the exact requirement;
- an allowed PPR schema projection;
- the configured process authority;
- every approved source as a neutral prompt-local handle; and
- controlled `retrieve`, `query_document`, `compare_cad_size`, and
  `analyze_candidate_layout` tools.

PA retrieves approved documents, CAD files, and one fresh live observation as
needed. After retrieving both inputs, PA may request neutral CAD-to-candidate
measurements. Tool results return to the same conversation as compact typed
evidence. PA then directly returns a proposal candidate, a clarification
question, or `unsupported_process`.

The model never supplies paths, provider IDs, hashes, frames, record types, or
arbitrary source names. Tool calls are resolved and audited by the system. A
malformed, unauthorized, stale, altered, or unavailable handle fails closed.

Clarification is reserved for genuine ambiguity in the requirement meaning.
There is no semantic reviewer, readiness contract, expected-revision feedback,
or outer correction loop. A malformed final response fails once rather than
being re-prompted toward an expected answer.

An answered clarification is evidence from its persisted append-only
`interaction_record/clarification_<question_turn>.json` record. The native
completion contract pins that exact record through `source_refs`; it does not
create or accept a separate clarification alias.

## Evidence processing

`production_grounding.py` exposes source-level retrieval and neutral evidence
tools to PA. The system performs document extraction, STL measurement, live
capture, segmentation, calibration, frame conversion, and location
reachability. These operations validate PA tool calls; they do not decide which
source, candidate, state meaning, location, or resource PA should choose.

Document retrieval yields the complete ordered `DocumentOverviewRecord` v3.
CAD retrieval yields `CADMeshRecord`. Observation retrieval yields typed RGB-D,
point-cloud, segmentation, stable crop, and `ObservationCandidateReview`
records. The review describes morphology, dimensions, surfaces, and support
contacts without assigning a part identity, state, CAD, process, or resource
meaning. The active `compare_cad_size` implementation persists
`CADSizeCorrespondenceRecord` v3 and returns every candidate measurement in
observation order without ranking or selecting a winner.
`analyze_candidate_layout` accepts any two or more PA-selected same-frame
candidates and reports raw positions, pairwise displacement vectors, distances,
and collinearity measurements without a built-in relation verdict. PA may
include zero, one, or multiple evidence-backed `state_values` from any accepted
typed record. Those values are semantic evidence links, not mandatory allocation
geometry.

The second PA call receives all capable resources and all neutral location
handles available from the retrieved evidence. PA submits one or more handles
for each state and one capable resource to `check_reachability`. If a selected
handle is a segmentation candidate, the tool materializes its approved
robot-frame location on demand. It reports every submitted location
independently and executes no motion. It does not certify grasping, insertion,
or manufacturing execution.

Before retrieval, `EvidencePresentationRecord` maps canonical sources to
randomized opaque handles. Approved CAD names and exact `context_ref` values are
available to PA, while repository paths, Gazebo identities, predetermined
matches, URLs, and semantic camera roles are absent. After semantic grounding,
`AllocationPresentationRecord` pins one randomized capable-resource order and
one shared neutral candidate pool. The same stored order drives the prompt,
tool enums, and response schema; it is audit evidence, not selection priority.

Presentation order is pinned for audit and has no priority. The controller
checks handle authority, hashes, ontology structure, configured capability, and
reachability only. It never ranks semantic candidates or resources.

## Ontology grounding

`ontology_grounding.py` validates a transient `OntologyGroundingProposal` v9
candidate containing exactly one model-authored `target_feature`. This
increment supports one requirement and one feature. The target contains:

- one evidence-cited `required_process.process_iri` selected from the configured
  process authority;
- one complete evidence-cited `current_state.statement`;
- one complete evidence-cited `desired_state.statement`; and
- zero, one, or multiple `state_values` for each state, each with a unique PA-authored semantic
  name, one accepted typed-record ref, one JSON Pointer, and direct evidence.

State values may reference any accepted typed record and are independent of the
location-handle lists used by resource allocation.

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

If PA's final proposal violates these invariants, the runtime preserves a
rejected audit record and emits a deterministic stage code. It neither repairs
the response nor returns the failure to PA. The host does not synthesize a
missing `defines`, `realizes`, state statement, or state value.

The accepted v9 proposal creates explicit `currentstate_0001` and
`desiredstate_0001` individuals attached to `feature_0001`. The unresolved
`processExecution` then activates the separate PA allocation call. Only the
unchanged PA resource and location lists cited from an accepted v4 reachability
record can add four assignment assertions. A rejection ends the stage without
host substitution or another PA correction round.

`RobotFrameLocationRecord` version 2 is semantically neutral. Its current or
desired role comes only from PA's allocation request. The architecture does not
use `TargetFeatureGeometryRecord`; verifier-required numeric geometry is
derived on demand from selected evidence and approved calibration.

## Completion and compatibility

New interactions write append-only native tool audits, direct PA turns,
`EvidencePresentationRecord` v1, `AllocationPresentationRecord` v1,
`OntologyGroundingProposal` v9, `ReachabilityCheckRecord` v4,
`ResourceSelectionRecord` v5, `PlanOnlyFeasibilityValidationRecord` v4, and
`PAContextGroundingCompletion` v7.
Clarification resumes the same conversation context using the exact persisted
question/reply history. Cancellation invokes no PA call.

The v7 completion directly pins the accepted proposal, nested evidence, both
presentation records, registry/workcell snapshots, referenced typed records,
reachability, exact RobotAgent validation, resource selection, assignment
delta, and final ABox. It does not create a duplicate new-run
`TypedGroundingContract` or copy `target_feature`; downstream consumers
reconstruct it from the hash-verified accepted proposal.

Historical completion v4-v6 records remain readable and are not migrated. New
Phase 5 handoff accepts completion v7 while retaining historical readers.

This boundary performs schema-constrained, evidence-backed instance grounding.
Multi-feature requirements, primitive parameter binding, motion execution,
post-process observation, and feature-state update are not implemented here.
