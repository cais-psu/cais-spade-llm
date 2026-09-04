# Spec2Primitives implementation status

This file describes verified repository behavior. It is not an aspirational
phase schedule.

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

## Framework phase status

**Phase 4 is implemented through Phase 4.4 under the current framework.** The
implemented boundary includes Phase 4.0 ontology context, Phase 4.1 document
evidence, Phase 4.2 CAD/RGB-D preprocessing, segmentation, neutral measurement,
camera-frame pose estimation, and frame conversion, Phase 4.3 typed records, and
Phase 4.4 PA-authored single-target-feature grounding plus independent
state-location/resource assignment.

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

Phase 4 completion means PA has grounded both feature states, selected one or
more location handles for each state and a capable resource, and cited accepted
hash-pinned reachability for every submitted location. It does not claim
grasping, insertion, cross-camera fusion, activation of an orientation-sensitive
consumer, primitive parameter binding, motion execution, or an observed
manufacturing outcome.

## Implemented native PA grounding

```text
blank UI requirement input
→ start one PA investigation
→ expose the PPR projection and approved evidence handles
→ PA optionally calls retrieve(evidence_id) multiple times
→ return typed evidence into the same PA conversation
→ PA directly returns one complete target_feature proposal or clarification
→ validate structure, provenance, hashes, and ontology consistency once
→ commit exactly seven target-feature assertions
→ expose all capable resources and neutral location handles in a second PA call
→ let PA call check_reachability for its selected resource and location lists
→ PA returns one resource backed by its cited accepted reachability result
→ validate the unchanged selection once, without substitution or feedback
→ commit exactly four processExecution/resource assertions
→ persist hash-pinned completion against the final ABox
```

New model responses contain no `next_action`, focused inspection request,
separate grounding-proposal action, provider ID, source path, frame, or record
type. One of the neutral PA tools is:

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

Each PA stage has one native tool-using conversation bounded by its existing
tool-call ceiling. Retrieved evidence, citations, tool failures, and audit
records remain available within that conversation. The controller supplies no
required-record plan, modality route, semantic gap, expected revision, or
candidate/resource preference.

## Evidence records

### Documents

One document retrieval creates `DocumentOverviewRecord` version 3. It processes
every page in order and records extracted text, rendered-page hashes, neutral
visual observations, uncertainty, and page citations. Processing is independent
of the product requirement and ontology. There is no new
`DocumentEvidenceRecord` production. Historical overview records remain readable
for recovered interactions; a new retrieval builds version 3.

### CAD

One approved STL retrieval creates `CADMeshRecord` with the authority-pinned
source hash, units, CAD-local frame, counts, dimensions, centroid, and mesh
artifact references. A filename is evidence metadata, not an ontology answer.

### Observation and location

A live RGB-D retrieval creates hash-pinned observation, point-cloud, and
uniform neutral segmentation records. PA sees neutral observation and candidate
handles without camera-role, source/target, evaluator, or simulator shortcuts.
PA may request all-candidate CAD measurements and raw candidate-layout geometry;
neither tool ranks or labels candidates.

When PA invokes `check_reachability`, any submitted segmentation candidate plus
approved calibration creates a neutral `RobotFrameLocationRecord` version 2
translated into the common resource frame. The allocation request supplies its
current or desired meaning. `CADPoseEstimationRecord` remains available only for
an orientation-sensitive consumer.

## Ontology proposal

New runs write `OntologyGroundingProposal` version 9 containing exactly one
model-authored `target_feature`. This increment supports one requirement and one
feature. The exact active payload is:

```json
{
  "target_feature": {
    "required_process": {
      "process_iri": "<authorized process IRI>",
      "evidence_refs": ["<direct evidence>"]
    },
    "current_state": {
      "statement": {
        "text": "<complete current product state>",
        "evidence_refs": ["<direct evidence>"]
      },
      "state_values": []
    },
    "desired_state": {
      "statement": {
        "text": "<complete desired product state>",
        "evidence_refs": ["<direct evidence>"]
      },
      "state_values": [
        {
          "name": "<unique PA-authored semantic name>",
          "value_ref": {
            "record_ref": "<accepted typed-record path>",
            "field_path": "<JSON Pointer>"
          },
          "evidence_refs": ["<direct evidence>"]
        }
      ]
    }
  }
}
```

PA chooses the process, both state statements, zero/one/many state values,
semantic names, record refs, field paths, and citations from retrieved evidence. The
host does not deterministically fill those values. It validates the authorized
process, every citation, accepted typed bindings, exact hashes, JSON Pointer
resolution, nonempty values, and unique names.

The host generates `feature_0001`, `currentstate_0001`, and
`desiredstate_0001`. It compiles their exact types, `ppr:hascurrentstate` and
`ppr:hasdesiredstate` links, `specification ppr:defines feature_0001`, and the
selected process `ppr:realizes feature_0001`. The authored state semantics
remain in the accepted proposal.

This is schema-constrained, evidence-backed instance grounding under the input
PPR TBox. It is not independent ontology-schema discovery.

Proposal acceptance is a single fail-closed validation step after PA's native
tool-using conversation ends. The host checks structure, process authority,
provenance, record hashes, JSON Pointers, and ontology consistency. There is no
semantic-review call, readiness contract, or controller-authored correction
loop. A malformed final response fails without answer-shaping feedback.

After the seven target-feature assertions are committed, the grounded feature,
both states, required process, and unresolved `processExecution` activate the
independent allocation decision. Typed numeric geometry remains outside RDF and
is derived only for location handles PA submits to reachability.

## Resource grounding

`config/workcell_profile.json` defines the process identity, ordered resource
identities, and authoritative manifest references. Resource candidates come
from this profile and PPR graph relations. Product names, CAD filenames,
modality branches, and task-support flags do not route selection.

The grounded feature, `currentstate`, `desiredstate`, required process, and
unresolved `processExecution` activate allocation directly. PA receives every
capable resource and every neutral location handle, then chooses a resource and
one or more handles for each state. Controlled `check_reachability` reports each
submitted location independently without selecting a robot. A
`ReachabilityCheckRecord` v4 and `ResourceSelectionRecord` v5 pin the unchanged
PA request and result. Any capable resource that reaches all cited locations is
valid; rejection ends the stage without substitution or corrective re-prompting.

## Completion and UI

New runs write session-free `PAContextGroundingCompletion` version 7. It pins
the direct PA decisions, accepted v9 proposal, selected process, both state
IRIs, nested evidence, both presentation records, registry/workcell snapshots,
referenced typed records, source authority, tool audits, v4 reachability, exact
RobotAgent validation, v5 resource selection, assignment delta, and final ABox.
It neither creates a new `TypedGroundingContract` nor copies `target_feature`;
consumers reconstruct the target from the pinned proposal. Historical v4-v6
completion records remain readable and are not migrated. New Phase 5 handoff
accepts completion v7 while preserving its historical readers.

A completed UI timeline contains exactly:

1. `Requirement received`
2. `Evidence investigated`
3. `Target feature grounded`
4. `State-location allocation validated`
5. `Grounding complete`

The final view shows both PA-authored state statements, optional state values
and their record/path/evidence refs, the exact submitted location handles and
per-location reachability, and the PA-selected resource. Tool protocol details
remain in diagnostics. Generic document uncertainty remains in its evidence
record for audit. Completion is labeled validated state-location allocation and
does not claim grasp feasibility, end-effector orientation, attached-object
geometry, process tolerance, force/contact, insertion feasibility, completed
manufacturing, parameter bindings, or motion execution.

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

`activate_selected_ra_context(...)` loads and verifies one version-4
`PAContextGroundingCompletion`, its version-2 `ResourceSelectionRecord`, and the
exact `resource_grounding_host` assignment delta. It persists one immutable
`SelectedRAAssignmentEnvelope` containing the requirement, semantic feature IRIs,
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
a reconstructed `target_feature`, selected resource, completion-consistent
post-assignment ontology assertions and TBox/ABox fingerprints, current robot
state, complete primitive catalog, and typed-record identities. The target
feature contains the requirement, specification and host-generated feature
IRIs, PA-authored required process and desired state, and bounded resolved-value
projections for referenced state values. There is no top-level `task` section.

Before contacting the RA, the host reloads the hash-pinned proposal and typed
records; validates the exact `defines`, `realizes`, `hasProcessExecution`,
`runsProcess`, and `runsOnResource` chain; resolves each JSON Pointer; and
rejects changed hashes, missing records, invalid paths, and empty values. It
does not send raw RDF, unrelated `ProductContextView` fields, unreferenced typed
record payloads, or bound primitive parameters.

The exact selected RobotAgent LLM treats `target_feature` as the authoritative
semantic product outcome and receives a strict response contract. It may return
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
inputs. It shows `target_feature`, selected resource, `ontology_projection`,
`robot_state`, `primitive_catalog`, and `grounded_context`, with counts and full
TBox/ABox fingerprints plus statement, state-value record/path/evidence refs,
bounded resolved values, and an expandable exact JSON view. The panel explicitly
distinguishes this input provenance from private model reasoning, feasibility
validation, and execution evidence. No prompt or additional evidence record is
persisted. `PrimitiveProgramDraft` remains unchanged and does not copy the
target feature; reconstruction follows its pinned completion.

## Deliberately deferred

- Phase 5.1b live exact-JID SPADE message delivery to the shared RobotAgent;
- future RA `MissingContextBatch → PA producers/clarification →
  CompositionContextBundle`;
- deterministic parameter-binding preflight and fully bound `primitive_steps`;
- multi-feature requirements;
- orientation-sensitive manipulation requirements beyond the two selected
  state locations;
- primitive-level validation, execution, and outcome validation;
- public API, `SystemBridge`, PPR TBox, or persisted-record migration.

The architecture intentionally has no `TargetFeatureGeometryRecord`. PA
dynamically retrieves approved evidence and assigns neutral observations to
both feature states. Calibration and numeric locations are derived on demand by
the verifier, never supplied as the predetermined target answer. No retrieval
or resource order is prescribed, and absent orientation/tolerance evidence is
not silently inferred.

The future RA binding preflight will examine only the required inputs declared
by exact primitives selected in the RA-authored `PrimitiveProgramDraft`. An
unbound product or scene value becomes a `MissingContextBatch` need only when
that draft and primitive interface activate it; it is not a fixed PA slot or
process-specific workflow. Inputs absent from every selected primitive
interface, grounded outcome requirement, and validator are `unmodeled` and must
fail closed rather than being silently inferred.
