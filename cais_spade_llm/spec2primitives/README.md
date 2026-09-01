# Spec2Primitives

Spec2Primitives is the isolated ICRA 2027 case study for dynamic primitive
composition in industrial robotic assembly. The implemented ProductAgent (PA)
boundary performs schema-constrained, evidence-backed instance grounding. It
does not discover or change the PPR TBox.

## Implementation status

**Phase 4 is implemented through Phase 4.4 under the current framework.** This
includes ontology context, document evidence, CAD/RGB-D grounding, typed
grounding contracts, evidence-gated semantic acceptance, location-based coarse
reach, and the selected `processExecution` assignment. Phase 5 selected-RA
handoff was previously unimplemented. Phase 5.1 now provides the contract-first
assignment envelope and injected-runtime state/catalog snapshots. Phase 5.2A
now lets that exact selected RobotAgent author one immutable, unbound
`PrimitiveProgramDraft` from the latest captured pair. Live SPADE delivery,
parameter binding, robot-local validation, execution, and observed outcomes
remain unimplemented.

Phase 4 completion does not add cross-camera fusion or activate the available
orientation-sensitive pose path for the current location-based consumer.

## Current PA workflow

```text
requirement
→ PA sees the PPR projection and approved evidence catalog
→ PA optionally calls retrieve(evidence_id) zero or more times
→ PA returns a grounding proposal, clarification, or insufficient evidence
→ deterministic validation creates a transient provisional ABox
→ the provisional graph activates its typed downstream requirements
→ missing prerequisites are reported to PA, which may retrieve again
→ accepted typed evidence gates the semantic ABox commit
→ configured resources are checked in profile order
→ the semantic ABox and selected assignment are committed
→ a hash-pinned completion record is written
```

PA has one controlled native tool:

```json
{
  "name": "retrieve",
  "arguments": {"evidence_id": "evidence_0001"}
}
```

The `evidence_id` is prompt-local. PA cannot provide a filesystem path, provider
ID, frame, record type, or arbitrary source name. It may retrieve documents,
approved CAD, and live observations in any order. The system validates the
handle, processes the source, and returns compact typed evidence in the same PA
conversation. Derived segmentation, correspondence, calibration, frame
conversion, and resource checks are system operations, not PA tools.

The proposal is not accepted merely because its RDF vocabulary is valid. The
system first evaluates it against a provisional graph. If that graph activates
a consumer whose declared typed input is unavailable, PA receives the derived
record gap and the still-eligible evidence handles. It may call `retrieve`
again in the same logical investigation. There is no fixed modality order or
product-specific missing-evidence rule.

The required-output projection is supplied by the system, not chosen by the
user. PA may ask the user about genuinely ambiguous requirement meaning, but it
may not ask whether a required record should be satisfied or whether an
approved evidence category should be retrieved. Such a response receives
internal correction feedback instead of becoming a UI clarification.

After tool use, PA directly returns exactly one of:

- `OntologyGroundingProposal` version 5;
- `clarification_question`; or
- `insufficient_evidence`.

New runs have no `next_action`, `inspect`, `propose_grounding`, or active
`GroundingSession` action loop. Read-only loaders for old version-2 completions
and version-3/4 ontology proposals remain so an existing interaction can still
be displayed without migration.

## Evidence and ontology boundary

Every approved PDF and STL is registered with an expected SHA-256 in
`references/products/approved_sources.json`. Static evidence is reused only
when its source revision still matches. A live-observation handle always
creates a fresh capture revision.

A document retrieval creates `DocumentOverviewRecord` version 2. It contains
every page in order, extracted text, rendered-page hashes, neutral visual
observations, uncertainty, and exact page citations. There is no targeted
inspection question, preferred page, product-specific answer, RAG ranking, or
`DocumentEvidenceRecord` in the new path.

An STL retrieval creates `CADMeshRecord` with hash-pinned source provenance,
units, CAD-local geometry, triangle and vertex counts, dimensions, and
centroid. `Gear_Medium.STL` is therefore typed evidence; it is not a TBox class
and does not add a product-specific RDF predicate. PA may cite it when grounding
an ABox feature.

A live RGB-D retrieval creates hash-pinned observation, point-cloud, and
segmentation records. When compatible CAD and observation evidence exist,
descriptor-driven services calculate correspondence. Accepted correspondence
plus approved calibration produces `RobotFrameLocationRecord` version 1. Coarse
resource reach uses this translated 3D location. Full pose estimation remains
available only for a future consumer that explicitly requires orientation.

These physical records remain outside RDF because the PPR TBox does not model
their numeric payloads. They nevertheless gate acceptance when the provisional
graph activates a consumer that requires them. The CAD used in correspondence
must be cited by the proposal's unique primary feature; unrelated retrieved CAD
is never used as a fallback.

`missing_information` means a relevant evidence-backed fact remains unknown.
It must not repeat a fact already present in an accepted typed record merely
because the current RDF projection has no property for it. Destination-shaft
or placement-order uncertainty is non-blocking for current coarse resource
selection. A future RA `MissingContextBatch` may reactivate that need; this path
is not implemented yet.

## Ontology grounding

PA may ground one or more evidence-supported `feature` individuals. Every
feature must be defined by the existing specification. Exactly one feature must
also be realized by the configured process; this unique graph join identifies
the primary resource-assignment target. Supporting features remain context.

Each individual and relation carries its own authorized evidence citations.
Deterministic validation rejects unsupported classes or properties, orphan
features, duplicate indices, empty meanings, unauthorized citations, zero or
multiple primary joins, recipes, primitives, resources, process creation,
literal facts, and unsupported relations. Each accepted RDF assertion retains
only its assertion-specific citations.

A structurally invalid PA proposal is never repaired by inserting an assertion.
The rejected output is audited and its deterministic validation message is
returned to PA for a bounded correction round. Only a corrected proposal can
enter the typed-evidence gate or be committed.

## Resource grounding

`config/workcell_profile.json` is the authority for the process identity,
ordered resource identities, and manifest references. The registry validates
those manifests and builds PPR `resource` and `capableOf` graph relations from
configuration. Python routing contains no product filename rule and no
`supports_manipulator_pick_place` task flag.

`ResourceAssignmentNeed` requires `RobotFrameLocationRecord`. Its target frame
comes from the candidate manifests' `gripper_reach.frame`; inconsistent frames
fail closed. Candidates are checked deterministically in validated profile
order and the first coarsely reachable resource is recorded in
`ResourceSelectionRecord` version 2.

For example, for `assemble medium gear`, PA may first retrieve the assembly
manual and `Gear_Medium.STL`, then propose the specification-defined feature
realized by the assembly process. The proposal remains transient because coarse
resource selection requires a `RobotFrameLocationRecord`. The descriptor
closure identifies the missing observation-produced prerequisite, so PA can
retrieve the approved live observation. The system then derives segmentation,
target-specific correspondence, calibration, and location. Only after that
chain is accepted does it commit the feature assertions, test the configured
resources against the observed location, select (for the current scene)
`xarm6`, and commit the assignment. A future direct-location provider could
satisfy the same typed requirement without CAD or RGB-D-specific routing.

## UI

The `/spec2primitives` page starts with a blank requirement. A completed run
shows only:

1. `Requirement received`
2. `Evidence investigated`
3. `Context grounded`
4. `Resource selected`
5. `Grounding complete`

Tool IDs, hashes, provider mechanics, and failures remain in diagnostics. The
final view shows the accepted location and selected resource. Genuine proposal
`missing_information` appears under **Known non-blocking context limits** unless
an accepted final typed record supersedes it. Generic document uncertainty
remains in its evidence record for audit and is not copied into that summary.
Historical `unresolved_evidence_needs` are also not copied. “Grounding complete”
does not mean the assembly is ready to execute.

Below the final grounding result, the temporary **Phase 5 · RobotAgent
Diagnostics** card provides operator activation and persisted inspection. Its current
**5.1 · Assigned RA activation and context snapshot** section shows
`xarm6@localhost` and its execution mode when selected, then paired state and
catalog revision counts, exact primitive symbols, the catalog fingerprint, and
expandable `robot_state` and full `primitive_catalog` after capture. A red
fail-closed diagnostic identifies an unavailable exact RobotAgent or invalid or
unpaired evidence. **Start Phase 5** consumes the unchanged active Phase 4
completion and reads the exact selected live in-process RobotAgent. If that
agent is not live and the full shared Agent System is stopped, the same action
requires the Spec2Primitives Dual Gazebo environment, waits for simulation
readiness, and starts only the exact Phase 4-selected RobotAgent in the selected
context-only simulation profile. The context-only agent has no task tools,
failure scenarios, or ROS controller and therefore does not wait for
`/detect_all`, TF, or motion services. It never starts CCA, ProductAgents,
UserAgent, a second RobotAgent, product orders, or `REQ_*` tasks; launches
Gazebo; selects a fallback; or executes a primitive. **Retry Phase 5** reuses
the same pinned assignment after a failed startup or contact. **Restart Phase
5** preserves the same Phase 4 assignment and existing immutable evidence while
appending a fresh paired state and synthesis-catalog snapshot. Refresh only
rereads the active interaction. Later Phase 5 diagnostics can share this
temporary card. After Phase 5 is complete, it is intended to be replaced by the
primitive composition card.

The existing application entry point remains:

```bash
poetry run python -m cais_spade_llm.ui_main
```

## MUST: Do not leak the answer

Allowed recognition inputs are only the user requirement, approved NIST
documents, approved candidate CAD files, RGB, depth, and approved camera
calibration. Gazebo model or entity names, world/SDF contents, spawn manifests,
configured spawn poses, `/gazebo/model_states`, `/get_entity_state`, detector
responses, and evaluator labels are forbidden recognition inputs.

Candidate CAD filenames and manual part names are allowed evidence, but
recognition must still determine which observed object corresponds to which
candidate and where it is. Ground truth is available only to a separate
post-prediction evaluator.

## Runtime boundary

Shared ProductAgent and RobotAgent implementations remain read-only authorities
reached through package-owned adapters. The public `SystemBridge` API is
unchanged; its shared Gazebo process classification recognizes
`gazebo_dual_spec2primitives` for core-service simulation readiness and hardware
interlocks. Because this world intentionally launches with
`run_perception:=false`, it does not queue the shared perception-dependent
controller prewarm. The current RA handoff reuses an exact compatible live agent
or starts only the exact Phase 4-selected context-only RobotAgent, then performs
an in-process state/catalog read. The same exact-agent adapter can then request
an ordered structural sequence of catalog symbols with no tools or execution.
SPADE handoff, parameter binding, IK, collision checking, robot execution, and
outcome validation remain unavailable.

Important locations:

- `agents/pa/`: native PA adapter, grounding, validation, and completion.
- `config/`: camera calibration and workcell profile authorities.
- `ontology/`: immutable PPR TBox, configured registry, and Workcell ABox.
- `tools/document_evidence/`: one-step full-document evidence.
- `tools/rgb_d_cad_grounding/`: CAD, RGB-D, correspondence, and location records.
- `contexts/`: ignored per-interaction runtime records.
- `evaluations/`: isolated post-prediction ground truth and evaluation.
- `references/products/`: approved documents and CAD source authority.
