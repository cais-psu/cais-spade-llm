# Spec2Primitives

Spec2Primitives is the isolated ICRA 2027 case study for product-specification-
driven dynamic primitive composition. Assembly is the current example; the
target-feature contract is process-independent so later cases may express
welding, painting, milling, or other desired product states. The implemented
ProductAgent (PA) boundary performs schema-constrained, evidence-backed instance
grounding. It does not discover or change the PPR TBox.

## Implementation status

**Phase 4 is implemented through Phase 4.4 under the current framework.** This
includes ontology context, document evidence, CAD/RGB-D grounding, typed
grounding contracts, evidence-gated semantic acceptance, location-based coarse
reach, and the selected `processExecution` assignment. Phase 5 selected-RA
handoff is partially implemented. Phase 5.1 provides the contract-first
assignment envelope and injected-runtime state/catalog snapshots. Phase 5.2A
now lets that exact selected RobotAgent author one immutable, unbound
`PrimitiveProgramDraft` from the latest captured pair. Live SPADE delivery,
parameter binding, executable primitive-program validation, execution, and
observed outcomes remain unimplemented.

Phase 4 completion does not add cross-camera fusion, execute motion, or claim
an observed manufacturing outcome. It derives numeric state locations only
when PA invokes a verifier; the orientation-sensitive pose path remains
available only to consumers that explicitly require it.

## Current PA workflow

```text
requirement
→ PA sees the PPR projection and approved evidence catalog
→ PA optionally calls retrieve(evidence_id) zero or more times
→ PA authors one evidence-grounded target_feature, clarification, or insufficiency
→ deterministic validation creates a transient provisional ABox
→ the provisional graph activates its typed downstream requirements
→ missing prerequisites are reported to PA, which may retrieve again
→ a separate PA semantic review accepts the target meaning or requests revision
→ exact candidate refs, hashes, CAD size, and state-value consistency are checked
→ accepted typed evidence gates the semantic ABox commit
→ PA chooses a provisional resource without reselecting either state
→ PA calls check_reachability(resource_symbol) for that resource
→ the exact selected RobotAgent returns a plan-only endpoint verdict
→ rejection evidence returns to PA without automatic substitution
→ only an accepted PA choice is committed as processExecution
→ a hash-pinned completion record is written
```

PA has one controlled native grounding tool:

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
conversion, and prerequisite checks are controlled system operations. During
allocation, PA receives a separate controlled `check_reachability` tool. It
derives and checks the two PA-selected locations and returns evidence without
choosing a resource.

The proposal is not accepted merely because its RDF vocabulary is valid. The
system first evaluates it against a provisional graph and returns deterministic
validation feedback without repairing the proposal. The current production
path supplies a fixed `RGBDSegmentationRecord` required-output projection, while
PA may call `retrieve` again in the same logical investigation and may choose
approved sources in any order. Descriptor-derived record planning through
`_producer_descriptors`, `_required_record_plan`, and `_grounding_gap` is not
active in this path and remains deferred.

The required-output projection is supplied by the system, not chosen by the
user. PA may ask the user about genuinely ambiguous requirement meaning, but it
may not ask whether a required record should be satisfied or whether an
approved evidence category should be retrieved. Such a response receives
internal correction feedback instead of becoming a UI clarification.

After tool use, PA directly returns exactly one of:

- `OntologyGroundingProposal` version 8 containing exactly one `target_feature`
  with explicit current and desired states;
- `clarification_question`; or
- `insufficient_evidence`.

New runs have no `next_action`, `inspect`, `propose_grounding`, or active
`GroundingSession` action loop. They persist host-only
`EvidencePresentationRecord` v1 and `AllocationPresentationRecord` v1,
`TargetFeatureSemanticReview` v2, `ReachabilityCheckRecord` v2,
`PlanOnlyFeasibilityValidationRecord` v2, `ResourceSelectionRecord` v4,
`TypedGroundingContract` v6, and `PAContextGroundingCompletion` v6. Older
proposal, selection, validation, and completion versions remain read-only.

## Evidence and ontology boundary

Every approved PDF and STL is registered with an expected SHA-256 in
`references/products/approved_sources.json`. Static evidence is reused only
when its source revision still matches. A live-observation handle always
creates a fresh capture revision.

A document retrieval creates `DocumentOverviewRecord` version 3. It contains
every page in order, extracted text, rendered-page hashes, neutral visual
observations that retain visible callouts, labels, and spatial relationships,
uncertainty, and exact page citations. There is no targeted
inspection question, preferred page, product-specific answer, RAG ranking, or
`DocumentEvidenceRecord` in the new path.

An STL retrieval creates `CADMeshRecord` with hash-pinned source provenance,
units, CAD-local geometry, triangle and vertex counts, dimensions, and
centroid. `Gear_Medium.STL` is therefore typed evidence; it is not a TBox class
and does not add a product-specific RDF predicate. PA may cite it when grounding
an ABox feature.

A live RGB-D retrieval creates hash-pinned observation, point-cloud, uniform
neutral segmentation, stable crop, and `ObservationCandidateReview` records.
The observation VLM describes every opaque candidate and its uncertainty but
cannot assign a state, CAD identity, process, or resource. PA-facing projections
omit semantic camera names and source/target shortcuts. Each candidate that PA
binds through `state_values` plus approved calibration produces a neutral
`RobotFrameLocationRecord` version 2. Coarse reach evaluates both translated 3D
locations. Full pose estimation remains available only for a consumer that
explicitly requires orientation.

These physical records remain outside RDF because the PPR TBox does not model
their numeric payloads. PA's explicit state assignment gives a selected neutral
record its current or desired meaning. Workspace and distance checks return
evidence only; the exact PA-selected RobotAgent then returns a closed plan-only
endpoint verdict. Neither verifier chooses or substitutes a resource.

## Ontology grounding

PA authors exactly one `target_feature` in this increment. Its
`required_process` selects one configured process and cites direct evidence. Its
`current_state.statement` and `desired_state.statement` state the observed and
requested feature conditions and cite direct evidence. `state_values`
carry PA-authored semantic names,
accepted typed-record refs, JSON Pointer field paths, and direct citations.
There is no host enum of value names. The general proposal schema permits zero,
one, or multiple values, while fresh production completion requires exactly one
`RGBDSegmentationRecord` candidate value for each state. Reusing one candidate
is rejected only when the two PA-authored value names are incompatible.

`config/model_runtime.json` schema version 3 configures `product_agent_llm`,
`document_vlm`, and `observation_vlm` as `gpt-5.6` with `medium` reasoning. Both
vision calls use high image detail, a 4096 output-token limit, a 90-second
timeout, and `store: false`. Document cache keys include the complete model
configuration and overview schema, so this change creates a new cache entry
without rewriting an older one.

The host generates `feature_0001`, `currentstate_0001`, and
`desiredstate_0001`. It compiles the feature and state types,
`specification ppr:defines feature_0001`, the selected process
`ppr:realizes feature_0001`, and exact `ppr:hascurrentstate` and
`ppr:hasdesiredstate` links. State citations back their feature-state
relationships and process citations back the process relation. Deterministic
validation checks process authority, citations, accepted record bindings,
record hashes, JSON Pointers, nonempty resolved values, unique names, and this
exact RDF projection. It does not author the target meaning or select
primitives.

A separate structured `TargetFeatureSemanticReview` asks PA whether the
statement and included values adequately represent the requirement under the
currently retrieved evidence. An incomplete review returns a concise gap to the
bounded retrieval/revision loop and is never accepted as missing information.

A structurally invalid PA proposal is never repaired by inserting an assertion.
The rejected output is audited and its deterministic validation message is
returned to PA for a bounded correction round. Only a corrected proposal can
enter the typed-evidence gate or be committed.

### Geometry is verifier-derived

The architecture does not contain `TargetFeatureGeometryRecord`. PA assigns
neutral approved evidence to `current_state` and `desired_state`; numeric
location is derived from the selected candidate and approved calibration only
when reachability or another verifier requests it. The derived value is
evidence for checking the PA-authored state assignment, not a predetermined
target answer. Document, CAD, and RGB-D retrieval order remains PA-controlled.

## Resource grounding

`config/workcell_profile.json` schema v2 is the authority for the complete
configured process catalog, resource identities, capable-process IRIs, and
manifest references. The registry validates
those manifests and builds PPR `resource` and `capableOf` graph relations from
configuration. Python routing contains no product filename rule and no
`supports_manipulator_pick_place` task flag.

The grounded feature, its two states, the required process, and unresolved
`processExecution` activate allocation directly. PA chooses a resource and one
approved value for each state, then calls controlled `check_reachability`.
Workspace and distance checks evaluate both locations and return evidence; they
never choose a robot and registry order carries no preference. The provisional
choice is sent only to that resource's RobotAgent for plan-only endpoint IK,
collision, and path validation. Only an `accepted` verdict permits
`processExecution` and `runsOnResource` to be committed.

For example, for `assemble medium gear`, PA may retrieve approved requirement,
CAD, and RGB-D evidence, author both feature states, and assign neutral regions
to each state. PA freely chooses a capable resource and invokes reachability on
those two regions. Calibration and robot-frame locations are derived at that
point. If the exact selected RobotAgent rejects plan-only feasibility, PA sees
the evidence and makes another free choice; the host does not substitute a
resource. An accepted choice is recorded as a validated endpoint-motion
allocation, not completed manufacturing, and no motion is executed. Grasping,
end-effector orientation, attached-object geometry, process tolerance,
force/contact, and insertion constraints remain explicitly unvalidated.

## PA-to-RA target-feature handoff

New Phase 5 handoff requires `PAContextGroundingCompletion` v6 and writes
`SelectedRAAssignmentEnvelope` v3. Phase 5.2A hash-verifies that completion and
its accepted v8 proposal, then reconstructs a
transient `target_feature` containing `product_requirement`,
`specification_iri`, host-generated `feature_iri`, the PA-authored required
process and both current and desired states, and `resolved_state_values`. Every
included state value is reloaded from its completion-pinned typed record,
resolved through its JSON Pointer, and projected to a bounded JSON value before
the RobotAgent call.

The RobotAgent LLM treats this target feature as the authoritative semantic
product outcome and selects and orders only exact symbols from its current
primitive catalog. Deterministic code validates completion authority, lineage,
the exact feature/process/resource chain, record hashes, resolved values, and
symbol membership; it does not solve the composition. The transient input has
no top-level `task` section.

`PrimitiveProgramDraft` remains unchanged. It pins the completion, assignment,
robot-state, and catalog records but does not copy the target feature. The Phase
5.2 evidence panel reconstructs the same input from those pinned authorities and
shows the statement, state-value record/path/evidence refs, and bounded resolved
values.

## UI

The `/spec2primitives` page starts with a blank requirement. A completed run
shows only:

1. `Requirement received`
2. `Evidence investigated`
3. `Target feature grounded`
4. `Endpoint-motion allocation validated`
5. `Grounding complete`

Tool IDs, hashes, provider mechanics, and failures remain in diagnostics. The
final view shows both feature-state statements, optional semantic values with
their record/path refs, annotated PA-selected RGB regions, post-acceptance CAD
identity (or all cited candidates as ambiguous), both locations and reach
distances, the PA-selected process/resource, and the exact RobotAgent
endpoint-motion verdict with checked and unvalidated constraints. Generic
document uncertainty remains in its evidence record for audit. “Validated
endpoint-motion allocation” does not mean manufacturing was completed or
motion was executed.

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

- Allowed recognition inputs are only the user requirement, approved NIST
  documents, approved candidate CAD files, RGB, depth, and camera calibration.
- Forbidden recognition inputs are Gazebo model names, Gazebo entity names,
  world or SDF contents, spawn manifests, configured spawn poses,
  `/gazebo/model_states`, `/get_entity_state`, current detector responses, and
  evaluator labels.
- Candidate CAD filenames and document part names are allowed because they are
  part of the supplied runtime corpus. Recognition must still determine which
  observed object matches which candidate and where it belongs.
- Ground truth may be read only by a separate evaluator after the prediction is
  finalized.
- Recognition code must not import, invoke, or share runtime objects with the
  ground-truth evaluator.
- Any experiment that violates this boundary is invalid and must not be
  reported.

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
