# Spec2Primitives

Spec2Primitives is the isolated ICRA 2027 case study for product-specification-
driven dynamic primitive composition. Assembly is the current example; the
target-feature contract is process-independent so later cases may express
welding, painting, milling, or other desired product states. The implemented
ProductAgent (PA) boundary performs schema-constrained, evidence-backed instance
grounding. It does not discover or change the PPR TBox.

## Implementation status

**Phase 4 is implemented as two autonomous ProductAgent decisions.** PA first
investigates approved evidence and authors one complete `target_feature`. After
that seven-assertion projection is validated and committed, PA independently
selects state-location evidence and a capable resource, invokes reachability,
and returns its cited selection. An accepted selection adds four
`processExecution` assertions, for exactly eleven Phase 4 assertions.

New runs persist `OntologyGroundingProposal` v9, `ReachabilityCheckRecord` v4,
`PlanOnlyFeasibilityValidationRecord` v4, `ResourceSelectionRecord` v5, and
`PAContextGroundingCompletion` v7. Completion v7 directly pins the proposal,
selection, reachability and RobotAgent validation records, final ABox, source
and tool evidence, presentation records, registry, and workcell. It does not
create a new `TypedGroundingContract`. Historical v4-v6 completions remain
readable.

Phase 5 selected-RA handoff remains partially implemented. Phase 5.1 provides
the assignment envelope and injected-runtime state/catalog snapshots. Phase
5.2A lets that exact selected RobotAgent author one immutable, unbound
`PrimitiveProgramDraft`. Primitive binding, executable-program validation,
execution, and observed outcomes remain unimplemented.

## Current PA workflow

```text
arbitrary requirement
→ initialize the TBox and interaction ABox
→ present every approved source through neutral prompt-local handles
→ PA autonomously retrieves and analyzes the evidence it chooses
→ PA returns one complete target_feature or a genuine clarification
→ validate its structure, provenance, ontology consistency, and seven assertions once
→ commit the unchanged target_feature projection
→ present all capable resources and every neutral location handle
→ PA chooses location lists and resources and invokes reachability as it chooses
→ validate the unchanged cited reachable selection once
→ commit four processExecution/resource assertions
→ persist PAContextGroundingCompletion v7
```

PA has four controlled native grounding tools: `retrieve`, `query_document`,
`compare_cad_size`, and `analyze_candidate_layout`. Example calls include:

```json
{
  "name": "retrieve",
  "arguments": {"evidence_id": "evidence_0001"}
}
```

```json
{
  "name": "compare_cad_size",
  "arguments": {
    "cad_evidence_id": "evidence_0002",
    "observation_evidence_id": "evidence_0003"
  }
}
```

The `evidence_id` is prompt-local. PA cannot supply a filesystem path, hidden
provider identity, ground-truth label, or arbitrary source. It may retrieve
documents, approved CAD, and observations in any order. `query_document`
accepts PA's exact question. `compare_cad_size` persists and returns every
evaluated candidate's measurements in observation order; it does not rank,
select, or label a winner. `analyze_candidate_layout` accepts any two or more
PA-selected candidates from one frame and returns raw positions, pairwise
displacements, distances, and collinearity measurements. It does not produce a
built-in spatial-relation verdict.

PA's target-feature response is final for that conversation. There is no
controller evidence plan, readiness contract, semantic-review call, expected
revision, or outer correction loop. A malformed or unsupported response fails
without answer-shaping feedback. The only normal alternatives are one complete
`target_feature`, a genuine `clarification_question`, or
`unsupported_process`. The latter is accepted only when no configured process
represents the requirement; the controller never rewrites a requirement as
`assembly`.

New incomplete results use only the deterministic stage codes
`unsupported_process`, `invalid_target_feature`,
`evidence_reference_invalid`, `location_evidence_unavailable`,
`no_reachable_resource`, and `invalid_resource_selection`. Their display text
is controller-authored and is not sent back to PA.

## Evidence and ontology boundary

Every approved PDF and STL is registered with an expected SHA-256 in
`references/products/approved_sources.json`. Static evidence is reused only
when its source revision still matches. A live-observation handle creates one
fresh capture per grounding run, reuses it for repeated requests in that run,
and creates a new capture after clarification resume.

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
The observation VLM describes every opaque candidate morphologically and its
uncertainty but cannot assign a part identity, state, CAD identity, process, or
resource. PA-facing projections omit semantic camera names and source/target
shortcuts. Target-feature `state_values` and allocation locations are separate:
the former may cite any accepted typed record, while the latter are chosen from
the neutral location catalog. If PA selects a segmentation candidate for
reachability, approved calibration converts only that selected candidate to a
`RobotFrameLocationRecord` v2.

These physical records remain outside RDF because the PPR TBox does not model
their numeric payloads. PA's explicit state assignment gives a selected neutral
record its current or desired meaning. The exact PA-selected RobotAgent returns
the per-location reachability result. Neither the evidence host nor the
RobotAgent chooses or substitutes a resource.

## Ontology grounding

PA authors exactly one `target_feature` in this increment. Its
`required_process` selects one configured process and cites direct evidence. Its
`current_state.statement` and `desired_state.statement` state the observed and
requested feature conditions and cite direct evidence. `state_values`
carry PA-authored semantic names,
accepted typed-record refs, JSON Pointer field paths, and direct citations.
There is no host enum of value names. Grounding permits zero, one, or multiple
values from any accepted typed record. They need not be RGB-D candidates or
locations. Location evidence is selected independently during the second PA
decision.

`config/model_runtime.json` schema version 3 configures `product_agent_llm` as
`gpt-5.6` with `none` reasoning for function tools through Chat Completions.
`document_vlm` and `observation_vlm` remain `gpt-5.6` with `medium` reasoning
through the Responses API. Both vision calls use high image detail, a 4096
output-token limit, a 90-second timeout, and `store: false`. Document cache keys
include the complete model configuration and overview schema, so this change
creates a new cache entry without rewriting an older one.

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

A structurally invalid PA proposal is never repaired, ranked, or sent back for
a convergence retry. Validation either accepts the unchanged response or emits
a stage-level code before ABox mutation.

### Geometry is verifier-derived

The architecture does not contain `TargetFeatureGeometryRecord`. Numeric
location is derived from a PA-selected location handle and approved calibration
only when PA invokes reachability. That derived value checks the submitted
resource/location choice; it does not retroactively constrain `state_values` or
provide a predetermined target answer. Document, CAD, and RGB-D retrieval order
remains PA-controlled.

## Resource grounding

`config/workcell_profile.json` schema v2 is the authority for the complete
configured process catalog, resource identities, capable-process IRIs, and
manifest references. The registry validates
those manifests and builds PPR `resource` and `capableOf` graph relations from
configuration. Python routing contains no product filename rule and no
`supports_manipulator_pick_place` task flag.

The grounded feature, its two states, the required process, and unresolved
`processExecution` activate allocation directly. PA receives every configured
resource capable of the selected process and every approved neutral location
handle. It submits one or more handles for each state and any capable resource
to `check_reachability`. The result reports each submitted location
independently. The controller accepts any PA-selected resource when its cited
record is current, capability-valid, and fully reachable; registry order and
distance do not rank choices.

Reachability does not certify grasping, insertion, force, tolerance, or process
execution, and it executes no motion. A failure does not trigger automatic
resource substitution or a controller-authored retry. Only an unchanged PA
selection backed by an accepted reachability record commits
`processExecution`, its process/resource links, and the state transition.

## PA-to-RA target-feature handoff

New Phase 5 handoff accepts `PAContextGroundingCompletion` v7 and writes
`SelectedRAAssignmentEnvelope` v3. Phase 5.2A hash-verifies that completion and
its accepted v9 proposal, then reconstructs a
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
4. `State-location allocation validated`
5. `Grounding complete`

Tool IDs, hashes, provider mechanics, and failures remain in diagnostics. The
final view shows both feature-state statements, optional semantic values with
their record/path refs, the PA-selected state-location lists and resource, and
the per-location reachability verdicts. Generic document uncertainty remains in
its evidence record for audit. A validated reachability allocation does not
mean manufacturing was completed or motion was executed.

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
