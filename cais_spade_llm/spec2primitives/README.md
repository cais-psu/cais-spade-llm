# Spec2Primitives

Spec2Primitives is the isolated ICRA 2027 case study for dynamic primitive
composition in industrial robotic assembly. The implemented ProductAgent (PA)
boundary performs schema-constrained, evidence-backed instance grounding. It
does not discover or change the PPR TBox.

## Current PA workflow

```text
requirement
→ PA sees the PPR projection and approved evidence catalog
→ PA optionally calls retrieve(evidence_id) zero or more times
→ PA returns a grounding proposal, clarification, or insufficient evidence
→ deterministic validation commits accepted ABox assertions
→ geometry services derive observed location when required
→ configured resources are checked in profile order
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

## UI

The `/spec2primitives` page starts with a blank requirement. A completed run
shows only:

1. `Requirement received`
2. `Evidence investigated`
3. `Context grounded`
4. `Resource selected`
5. `Grounding complete`

Tool IDs, hashes, provider mechanics, and failures remain in diagnostics. The
final view shows the accepted location and selected resource. Proposal
`missing_information` and document uncertainty appear under **Known
non-blocking context limits**. Historical `unresolved_evidence_needs` are not
copied into that summary. “Grounding complete” does not mean the assembly is
ready to execute.

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

All Spec2Primitives implementation remains in this directory. Shared
ProductAgent and RobotAgent implementations are read-only authorities reached
through package-owned adapters. `SystemBridge` and the public UI/runtime API are
unchanged. There is no current RA handoff, primitive composition, IK, collision
checking, robot execution, or outcome validation.

Important locations:

- `agents/pa/`: native PA adapter, grounding, validation, and completion.
- `config/`: camera calibration and workcell profile authorities.
- `ontology/`: immutable PPR TBox, configured registry, and Workcell ABox.
- `tools/document_evidence/`: one-step full-document evidence.
- `tools/rgb_d_cad_grounding/`: CAD, RGB-D, correspondence, and location records.
- `contexts/`: ignored per-interaction runtime records.
- `evaluations/`: isolated post-prediction ground truth and evaluation.
- `references/products/`: approved documents and CAD source authority.
