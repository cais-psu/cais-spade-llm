# RA

RA means RobotAgent. This directory owns the implemented Phase 5.1
contract-first assignment/context-snapshot boundary and Phase 5.2A structural
primitive draft, plus the Phase 4 plan-only feasibility record boundary.

## Implemented Phase 4 allocation validation

For simulation allocation, the PA-requested `check_reachability` call contacts
only the exact provisional RobotAgent JID and validates its execution mode. The
Spec2Primitives adapter bypasses the shared recovery workspace precheck for this
one `motion_mode: cartesian_pick_place` path and asks the configured live MoveIt
`GetCartesianPath` service for chained pick and place phases. It uses live TF,
the configured group and EE/TCP links, collision checking, `max_step = 0.01`,
`jump_threshold = 0`, and a minimum fraction of `0.999`. It never contacts an
execution action. Missing TF, service, CAD, or support evidence returns
`needs_context`; physical mode retains its existing validation behavior.

## Implemented Phase 5.1

`activate_selected_ra_context(...)` requires and verifies the Phase 4 v6
completion, PA-authored two-state allocation, reachability, exact RobotAgent
plan-only validation, resource selection, and assignment delta before
persisting one `SelectedRAAssignmentEnvelope` v3. The envelope pins the exact
process/resource, both state-evidence assignments, evidence and allocation
presentations, registry/workcell snapshots, reachability, endpoint-motion
or Cartesian validation, and no-motion claim. It tells the exact selected RA which
requirement, specification, host-generated feature and state individuals, and
process it owns. An injected
`RobotAgentCompositionRuntime` must confirm that assignment before returning
fresh JSON state and its complete primitive-only composition catalog.

The UI supplies an in-process Spec2Primitives adapter for this contract. The
adapter requires exactly the JID and execution mode selected by Phase 4. It
reuses that live agent when available. Otherwise, only when the full shared
Agent System is stopped and the Spec2Primitives Dual Gazebo environment is
running, it waits for simulation readiness and starts only the selected
RobotAgent in a context-only simulation profile. That profile exposes no task
tools or failure scenarios and constructs no ROS controller, so state capture
cannot wait for perception or motion services. The adapter then reads the
RobotAgent-owned logical state and composer-visible recovery synthesis catalog
on the agent loop and converts that catalog without changing symbol order or
names. It does not start CCA, ProductAgents, UserAgent, product orders, safety
generation, tools generation, or another RobotAgent. If startup or contact
fails, the assignment audit remains unchanged and the UI offers a retry of the
same handoff.

After a successful capture, **Restart Phase 5** dispatches the same immutable
assignment again and appends the next paired state and synthesis-catalog
snapshot revision. It does not overwrite the earlier capture or rerun Phase 4.

The host validates exact JID and assignment correlation, preserves catalog
order and symbols, and writes matching append-only `RobotStateSnapshot` and
`PrimitiveCatalogSnapshot` revisions. A runtime failure leaves the immutable
assignment audit without claiming that RA state was retrieved. Existing
snapshot histories must remain complete, paired, hash-valid, and gap-free.

`read_phase_5_1_diagnostic(...)` validates the same persisted authority without
contacting or activating an RA. The temporary **Phase 5 · RobotAgent
Diagnostics** UI card uses it for its 5.1 section to show whether Phase 4 is
incomplete, the selected assignment is ready, the assignment is waiting for an
RA response, paired context was captured, or persisted evidence is blocked.
When context exists, the section exposes the exact selected JID, snapshot refs,
current `robot_state`, ordered primitive symbols, full catalog, and catalog
fingerprint. Its refresh control only rereads persisted records. Later Phase 5
diagnostics can be added to that temporary card without treating it as the
final primitive composition UI.

The Start control starts or reuses only the exact selected RobotAgent but does
not launch Gazebo, start the full Agent System, send a SPADE message, author
primitives, load raw RDF, send the full ABox or typed context, plan motion,
validate a candidate, or execute anything.

## Implemented Phase 5.2A

`author_primitive_program_draft(...)` requires the latest validated Phase 5.1
pair and asks the same exact selected RobotAgent for only a structural sequence.
The bounded input begins with a reconstructed `target_feature` containing the
exact product requirement, specification IRI, host-generated feature IRI,
PA-authored required process, current state, desired state, and a bounded
projection of every referenced state value. It also includes the selected resource,
completion-consistent post-assignment ontology assertions and TBox/ABox
fingerprints, current state, complete catalog, and typed-record identities.

Before the RA call, the host reloads the hash-pinned v8 proposal, v6 completion,
both state IRIs, accepted typed bindings, reachability, plan-only validation,
and referenced JSON records. It resolves each JSON
Pointer and rejects a changed hash, missing record, invalid path, empty value,
or mismatched feature/process/resource chain. These checks establish authority
and lineage; they do not select or order primitives. The input has no top-level
`task` section and excludes raw RDF, unrelated `ProductContextView` fields,
unreferenced typed-record payloads, and parameter bindings.

The RobotAgent LLM treats `target_feature` as the authoritative semantic product
outcome. It may select and order only exact catalog symbols, may repeat a
symbol, and may instead return an explicit unsupported result. The host derives
step indexes and all record identity fields, then writes one append-only
`composition/primitive_program_drafts/draft_<number>.json` per state/catalog
pair. A second draft for the same pair, an invented symbol, malformed output,
or altered pinned evidence fails closed. The structured call exposes no task
tools and cannot execute a primitive.

The 5.2 section in the temporary Phase 5 card enables **Create Primitive
Draft** only when the latest captured pair has no draft. It displays the ordered
symbols or unsupported reason and the full persisted `PrimitiveProgramDraft`.
Refresh remains read-only.

For the current authored or unsupported draft, the same card also reconstructs
the exact transient `COMPOSITION_INPUT` from the draft's hash-pinned completion,
assignment, robot-state, and primitive-catalog records. The read-only evidence
panel labels and displays `target_feature`, including the statement,
record/path/evidence references, and bounded resolved-value projections, and
exposes the complete JSON payload.
It is input provenance, not private model reasoning, feasibility validation, or
execution evidence, and the reconstructed payload is not persisted.

`PrimitiveProgramDraft` itself is unchanged. It pins the v6 completion and
other inputs but does not copy `target_feature`, its statement, or its resolved
values; diagnostics reconstruct them from the same persisted authority.

## Planned continuation

Phase 5.1b will map the same narrow runtime contract to live exact-JID SPADE
message delivery. A deterministic binding preflight may identify missing inputs
but cannot create or repair the structural draft. RA-owned inputs remain local;
product or scene inputs are deduplicated into a `MissingContextBatch` for PA.
After receiving a versioned `CompositionContextBundle`, RA alone authors the
fully bound `primitive_steps` candidate.

Multi-feature input, orientation/tolerance grounding, parameter binding,
`MissingContextBatch`, primitive-level execution validation, execution, and
post-process state update remain future increments. Assembly is the current example; the target-feature handoff
does not hard-code an assembly-only target shape and may later carry welding,
painting, milling, or other product outcomes when their evidence and primitive
contracts exist.
