# RA

RA means RobotAgent. This directory owns the implemented Phase 5.1
contract-first assignment and context-snapshot boundary plus future
Spec2Primitives RobotAgent connection, primitive composition, and robot-local
validation work.

## Implemented Phase 5.1

`activate_selected_ra_context(...)` verifies the Phase 4 completion, resource
selection, and assignment delta before persisting one
`SelectedRAAssignmentEnvelope`. The envelope tells the exact selected RA which
requirement and semantic task it owns. An injected
`RobotAgentCompositionRuntime` must confirm that assignment before returning
fresh JSON state and its complete primitive-only composition catalog.

The host validates exact JID and assignment correlation, preserves catalog
order and symbols, and writes matching append-only `RobotStateSnapshot` and
`PrimitiveCatalogSnapshot` revisions. A runtime failure leaves the immutable
assignment audit without claiming that RA state was retrieved. Existing
snapshot histories must remain complete, paired, hash-valid, and gap-free.

Phase 5.1 does not connect to a running SPADE RobotAgent, author primitives,
load raw RDF, send the full ABox or typed context, plan motion, validate a
candidate, or execute anything.

## Planned continuation

Phase 5.1b will map the same narrow runtime contract to live exact-JID SPADE
delivery. Phase 5.2 will build the bounded composition input and allow the
selected RA to author a structural `PrimitiveProgramDraft`. A deterministic
binding preflight may identify missing inputs but cannot create or repair
primitive steps. RA-owned inputs remain local; product or scene inputs are
deduplicated into a `MissingContextBatch` for PA. After receiving a versioned
`CompositionContextBundle`, RA alone authors the fully bound `primitive_steps`
candidate.
