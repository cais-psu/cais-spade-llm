# Spec2Primitives implementation status

This file describes verified repository behavior. It is not an aspirational
phase schedule.

## Implemented native PA grounding

```text
blank UI requirement input
→ start one PA investigation
→ expose the PPR projection and approved evidence handles
→ PA optionally calls retrieve(evidence_id) multiple times
→ return typed evidence into the same PA conversation
→ PA directly returns proposal, clarification, or insufficient evidence
→ validate the evidence-cited proposal against a transient ABox
→ derive the active consumer's typed prerequisite closure
→ return any source-evidence gap to PA and allow more retrieval
→ accept the proposal only after its required evidence chain is valid
→ evaluate configured resource candidates in profile order
→ commit the semantic ABox and selected assignment
→ persist hash-pinned completion against the final ABox
```

New model responses contain no `next_action`, focused inspection request,
separate grounding-proposal action, provider ID, source path, frame, or record
type. The one PA tool is:

```json
{
  "name": "retrieve",
  "arguments": {"evidence_id": "evidence_0001"}
}
```

PA may call it zero, one, or multiple times and may choose document, CAD, or
live observation evidence in any order. Prompt-local IDs resolve only to the
currently approved and eligible catalog. Unknown, stale, unavailable, altered,
or unauthorized IDs fail closed before source access. Static typed evidence is
reused only while its source hash remains valid; live observation always makes
a fresh revision.

The shared ProductAgent is unchanged. The Spec2Primitives adapter supplies the
controlled tools and bridges the shared synchronous callback to asynchronous
evidence services with a bounded timeout.

The loop is bounded by the existing emergency PA-turn ceiling. A candidate
proposal and its provisional graph remain transient between rounds; retrieved
evidence, citations, tool failures, and audit records remain available. The gap
shown to PA is computed from consumer and provider descriptors, not from a
fixed CAD/observation condition.

## Evidence records

### Documents

One document retrieval creates `DocumentOverviewRecord` version 2. It processes
every page in order and records extracted text, rendered-page hashes, neutral
visual observations, uncertainty, and page citations. Processing is independent
of the product requirement and ontology. There is no new
`DocumentEvidenceRecord` production. Version-1 overview records remain readable
for recovered interactions; a new retrieval builds version 2.

### CAD

One approved STL retrieval creates `CADMeshRecord` with the authority-pinned
source hash, units, CAD-local frame, counts, dimensions, centroid, and mesh
artifact references. A filename is evidence metadata, not an ontology answer.

### Observation and location

A live RGB-D retrieval creates hash-pinned observation, point-cloud, and
segmentation records. Compatible CAD and observation records automatically
activate descriptor-driven correspondence. Low-level derived services are not
PA tools.

Accepted correspondence plus approved calibration creates
`RobotFrameLocationRecord` version 1 containing the observed candidate center
translated from the camera frame to the required resource frame. Full
`CADPoseEstimationRecord` remains available only for a future orientation-
sensitive consumer.

## Ontology proposal

New runs write `OntologyGroundingProposal` version 5 using the multi-feature v4
payload:

- each individual has an index, PPR class, grounded meaning, and citations;
- each relation has assertion-specific citations;
- every feature is defined by the existing specification;
- exactly one feature is also realized by the configured process; and
- proposal context, uncertainty, and missing information remain cited.

Validation rejects unsupported vocabulary, literal facts, recipes, primitives,
resources, process creation, orphan features, duplicate indices, empty
meanings, unauthorized citations, and zero or multiple primary joins. The
system compiles each RDF assertion with only its own citations.

This is schema-constrained, evidence-backed instance grounding under the input
PPR TBox. It is not independent ontology-schema discovery.

Proposal acceptance is evidence-gated. The unique `defines`/`realizes` join in
the provisional graph identifies the primary feature and activates the current
resource consumer. Only CAD cited by that feature can participate in its
physical correspondence. Typed geometry remains outside RDF, but a required
typed chain must be current, hash-valid, accepted, and unambiguous before the
proposal is written as accepted or its assertions are merged.

## Resource grounding

`config/workcell_profile.json` defines the process identity, ordered resource
identities, and authoritative manifest references. Resource candidates come
from this profile and PPR graph relations. Product names, CAD filenames,
modality branches, and task-support flags do not route selection.

`ResourceAssignmentNeed` requires `RobotFrameLocationRecord`. Its target frame
is derived from all candidates' `gripper_reach.frame`; inconsistent frames fail
closed. Coarse reach reads validated manifest geometry and selects the first
reachable resource in profile order. `ResourceSelectionRecord` version 2 pins a
generic grounding-record reference and hash. Version-1 pose-linked records are
read-only recovery inputs.

For `assemble medium gear`, a first proposal based on the manual and
`Gear_Medium.STL` activates the consumer requirement for
`RobotFrameLocationRecord`. If observation-produced segmentation is absent,
the descriptor closure exposes that gap to PA without committing the feature
assertions. PA may retrieve the live observation; the system derives
correspondence and calibrated location, accepts the proposal, then evaluates
the PPR `capableOf` candidates. In the current scene, `xarm6` is selected only
after its configured reach contains that accepted target-specific location.

## Completion and UI

New runs write session-free `PAContextGroundingCompletion` version 3 and
`TypedGroundingContract` version 3. They pin the direct PA decision, proposal,
typed evidence, source authority, tool audits, resource selection, assignment
delta, and final ABox. Existing version-2 completion/session records remain
readable but are never produced or migrated.

A completed UI timeline contains exactly:

1. `Requirement received`
2. `Evidence investigated`
3. `Context grounded`
4. `Resource selected`
5. `Grounding complete`

The visible target state is location-based. Tool protocol details remain in
diagnostics. Genuine proposal `missing_information` and document uncertainty
appear under **Known non-blocking context limits**; historical unresolved
markers do not. Completion does not claim execution readiness.

## Deliberately deferred

- future RA `MissingContextBatch → PA producers/clarification →
  CompositionContextBundle`;
- primitive catalogs and composition;
- orientation-sensitive manipulation requirements;
- IK, collision checking, execution, and outcome validation;
- public API, `SystemBridge`, PPR TBox, or persisted-record migration.

Destination-shaft, attachment, and placement-order facts remain non-blocking
when the current location-based resource consumer does not require them. A
future authorized consumer may request them dynamically.
