# Spec2Primitives implementation status

This file describes verified repository behavior. It is not an aspirational
phase schedule.

## Framework phase status

**Phase 4 is implemented through Phase 4.4 under the current framework.** The
implemented boundary includes Phase 4.0 ontology context, Phase 4.1 document
evidence, Phase 4.2 CAD/RGB-D preprocessing, segmentation, correspondence,
camera-frame pose estimation, and frame conversion, Phase 4.3 typed grounding
contracts, and Phase 4.4 evidence-gated semantic acceptance and resource
assignment.

**Phase 5.1 is implemented as a contract-first assigned-RA activation and
context-snapshot boundary.** It produces the minimum verified assignment
envelope, can start or reuse only the exact Phase 4-selected in-process
RobotAgent from the UI, and persists fresh state and the complete returned
recovery synthesis catalog. It does not start the full Agent System. Live SPADE
message delivery remains Phase 5.1b.

**Phase 5.2A is implemented as RA-authored structural composition.** From the
latest validated Phase 5.1 state/catalog pair, the exact selected RobotAgent
returns only an ordered sequence of existing catalog symbols or an explicit
unsupported result. The host pins that response to its Phase 4 completion,
assignment, state, and catalog and appends one immutable
`PrimitiveProgramDraft` per context pair. Parameter binding, missing-context
requests, feasibility validation, and execution remain later steps.

Phase 4 completion means the current location-based pre-RA consumer can reach a
hash-pinned post-assignment completion. It does not claim cross-camera fusion,
activation of an orientation-sensitive consumer, parameter binding, IK or
collision validation, execution, or an observed assembly outcome.

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

For fresh version-3 completions, `context_summary` is explicitly marked as the
ProductAgent proposal narrative captured before deterministic typed grounding
and resource assignment. The original narrative remains unchanged in the
accepted `OntologyGroundingProposal`. Final hash-pinned typed records and
`ResourceSelectionRecord` are authoritative for completion state. Existing
unmarked version-3 records remain readable and are not migrated.

A completed UI timeline contains exactly:

1. `Requirement received`
2. `Evidence investigated`
3. `Context grounded`
4. `Resource selected`
5. `Grounding complete`

The visible target state is location-based. Tool protocol details remain in
diagnostics. Genuine proposal `missing_information` appears under **Known
non-blocking context limits** unless an accepted final typed record supersedes
it. Generic document uncertainty remains in its evidence record for audit;
historical unresolved markers also do not appear. Completion does not claim
execution readiness.

The same page contains an always-visible temporary **Phase 5 · RobotAgent
Diagnostics** card. Its current **5.1 · Assigned RA activation and context
snapshot** section reads the active interaction's validated Phase 4 completion,
immutable assignment audit, and paired RA state/catalog revisions. It shows the
exact selected JID and execution mode, snapshot refs and counts, catalog
fingerprint, exact primitive symbols, and expandable raw `robot_state` and
`primitive_catalog`. **Start Phase 5** invokes
`activate_selected_ra_context(...)` for the unchanged active Phase 4 result and
reuses its exact live in-process RobotAgent. If the shared Agent System is
stopped, the same action requires the running Spec2Primitives Dual Gazebo
environment, waits for simulation readiness, and starts only the exact selected
RobotAgent in a context-only Phase 4-selected simulation profile before
capture. The standalone profile exposes no task tools or failure scenarios and
constructs no controller, so it cannot wait for perception or motion services.
An unavailable or mismatched agent or startup failure fails closed and leaves
the same pinned assignment retryable. It never launches Gazebo, starts CCA or a
ProductAgent, creates product-order tasks, or starts the second RobotAgent.
Refresh remains read-only. Later Phase 5 implementation steps may add their own
diagnostics to this temporary card. After Phase 5 is implemented, the diagnostic
surface is intended to be replaced by the operator-facing primitive composition
card.

## Implemented Phase 5.1 assigned-RA context handoff

`activate_selected_ra_context(...)` loads and verifies one version-3
`PAContextGroundingCompletion`, its version-2 `ResourceSelectionRecord`, and the
exact `resource_grounding_host` assignment delta. It persists one immutable
`SelectedRAAssignmentEnvelope` containing the requirement, semantic task IRIs,
exact selected resource identity and execution mode, and pinned Phase 4 record
refs, hashes, and fingerprints.

The injected `RobotAgentCompositionRuntime` receives that envelope before it
can return context. The receiver must confirm that the envelope targets its
exact JID. The response must echo the same JID and assignment fingerprint and
provide a non-empty JSON robot state plus the complete composition catalog.
The host preserves catalog order and exact symbols, validates the typed
primitive interface, and writes matching append-only `RobotStateSnapshot` and
`PrimitiveCatalogSnapshot` revisions. Existing paired revisions are reloaded
and revalidated before another RA request; malformed, changed, unpaired, or
gapped histories fail closed.

`read_phase_5_1_diagnostic(...)` exposes those validations as the UI statuses
`waiting_for_phase_4`, `ready_for_assignment`, `waiting_for_ra`,
`context_captured`, and `blocked`. It never repairs or extends the immutable
history.

The UI composition root now supplies a Spec2Primitives-owned in-process adapter.
It finds exactly the JID selected by Phase 4, requires that RobotAgent to be
alive with the selected execution mode, and reads its authoritative recovery
snapshot and composer-visible recovery synthesis catalog on the agent loop. If
the selected RobotAgent is unavailable, the adapter can ask the shared runtime
to start that one exact RobotAgent without CCA, ProductAgents, UserAgent, or the
other RobotAgent. This is not a SPADE message connection. The Phase 5.1 read
sends no full ABox, typed evidence payload, raw RDF, or composition prompt and
creates no primitive draft, missing-context batch, candidate, validation
result, or execution command. The separately invoked Phase 5.2A operation
reuses the same exact-agent adapter for structural authoring.

## Implemented Phase 5.2A structural primitive draft

`author_primitive_program_draft(...)` reloads the validated Phase 4 assignment
and latest Phase 5.1 pair, then builds a bounded composition input containing
the task IRIs and requirement, selected resource, completion-consistent
post-assignment ontology assertions and TBox/ABox fingerprints, current robot
state, complete primitive catalog, grounded context summary, known limits, and
typed-record identities. Before contacting the RA, the host validates the exact
`defines`, `realizes`, `hasProcessExecution`, `runsProcess`, and `runsOnResource`
chain. It does not send raw RDF, unrelated `ProductContextView` fields, typed
record payloads, or bound primitive parameters.

The exact selected RobotAgent receives a strict response contract. It may return
`proposed` with a non-empty ordered sequence of exact catalog symbols, including
intentional repetition, or `unsupported` with an explanation. Unknown symbols,
extra fields, empty proposals, malformed responses, stale or altered evidence,
and a second draft for the same context pair fail closed.

The host derives `step_index`, provenance, hashes, fingerprints, and the
append-only filename. `read_phase_5_2_diagnostic(...)` exposes
`waiting_for_context`, `ready_for_draft`, `draft_authored`, `unsupported`, or
`blocked` without contacting the RobotAgent. Restarting Phase 5.1 appends a new
state/catalog pair and makes that new pair eligible for one new structural
draft; earlier records remain unchanged.

For the current authored or unsupported draft, the Phase 5.2 UI reconstructs
the exact transient `COMPOSITION_INPUT` from the draft's validated hash-pinned
inputs. It shows the task, selected resource, `ontology_projection`,
`robot_state`, `primitive_catalog`, and `grounded_context`, with counts and full
TBox/ABox fingerprints plus an expandable exact JSON view. The panel explicitly
distinguishes this input provenance from private model reasoning, feasibility
validation, and execution evidence. No prompt or additional evidence record is
persisted.

## Deliberately deferred

- Phase 5.1b live exact-JID SPADE message delivery to the shared RobotAgent;
- future RA `MissingContextBatch → PA producers/clarification →
  CompositionContextBundle`;
- deterministic parameter-binding preflight and fully bound `primitive_steps`;
- orientation-sensitive manipulation requirements;
- IK, collision checking, execution, and outcome validation;
- public API, `SystemBridge`, PPR TBox, or persisted-record migration.

Destination-shaft, attachment, and placement-order facts remain non-blocking
when the current location-based resource consumer does not require them. A
future authorized consumer may request them dynamically.

The future RA binding preflight will examine only the required inputs declared
by exact primitives selected in the RA-authored `PrimitiveProgramDraft`. An
unbound product or scene input such as `target_feature` becomes a
`MissingContextBatch` need only when that draft and primitive interface activate
it; it is not a fixed PA slot or task workflow. Inputs absent from every
selected primitive interface, grounded outcome requirement, and validator are
`unmodeled` and must fail closed rather than being silently inferred.
