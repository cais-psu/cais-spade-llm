# PA

`context_interaction.py` implements Phase 3.1 through composition with the shared
ProductAgent. It accepts the exact `product_requirement`, exposes only
`ask_llm_structured(...)` through `ProductAgentContextRuntime`, and records one
validated `needed_context` decision under the caller-owned interaction root.

`context_serving.py` implements Phase 3.2. It reads the completed Phase 3.1
records, resolves exactly one approved document or CAD `context_ref`, or captures
one explicitly requested numbered fresh live RGB-D observation. It preserves
static evidence, live artifact references, internal `provenance`, and
`retrieval_error` under the same caller-owned interaction root without
substituting fallback evidence. The public Phase 3.2 entrypoint retains its exact
one-request behavior.

`context_grounding.py` defines the injected schema-only TBox configuration and
the narrow controlled interpretation/assessment protocol. `context_assessment.py`
implements the Phase 3.3 outer loop: serve one selected source, dispatch exactly
one evidence-type-routed producer, validate and merge its generic triple delta,
then persist one Phase 4.3-style assessment over the updated ABox. Raw served
summaries cannot directly authorize clarification or completion.

## Planned production behavior: how PA determines needed information

PA does not infer a universal input checklist from the TBox. The TBox defines
legal meaning, not which document, sensor, CAD, or calibration operation is
needed. The next consumer declares semantic or typed inputs, and the runtime
compares them with the validated product context:

```text
required consumer inputs
    - valid current ProductContextView
    = ContextNeeds
```

Before allocation, the consumer is an evolving `TaskTransitionDraft`. It
identifies the currently blocking product, process, `target_feature`, outcome,
constraint, or task binding needed for a robot-independent
`TaskTransitionContract`. After allocation, the consumer is the selected RA's
`PrimitiveProgramDraft` and primitive-interface contracts. RA deduplicates the
product or scene inputs they expose into a `MissingContextBatch`; robot-owned
state, limits, IK, collision, and trajectory inputs remain local to RA.

`ProductContextView` will combine the compact ABox with validated
`TypedContextBinding` entries. Each binding records its subject or task role,
record type, ref, hash, status, frame, time or validity, producer, and
provenance. An ambiguous, rejected, stale, wrong-frame, missing, or tampered
record does not satisfy a `ContextNeed`. Numeric poses, transforms,
observations, and tolerances remain in typed records rather than being forced
into the ABox.

The application-owned `GroundingProducerDescriptor` registry maps a missing
semantic or typed output to an authorized controlled producer and its evidence
dependencies. PDF, CAD, RGB-D, and calibration are producer inputs, not the
needs exchanged between PA and RA. The exact CAD remains caller- or
corpus-authorized, and calibration remains injected by an approved
environmental authority; PA never searches an unrestricted inventory or
guesses either value.

| Stable mechanism | Dynamic interaction values |
|---|---|
| Official TBox and exact symbols | Requirement and runtime ABox individuals |
| `ContextNeed` and `TypedContextBinding` schemas | Currently blocking semantic or typed inputs |
| Producer capability descriptors and validators | Selected approved evidence and produced refs |
| Progress and authority rules | Selected RA, frame, catalog, and `primitive_steps` |

For `assemble Medium Gear`, PA may first ground the requested process, product,
and `target_feature` when the evolving task draft identifies those gaps. If the
task draft requires a current loose-part pose, a typed pose need routes to the
approved CAD/RGB-D producer using the exact authorized CAD. If no geometry is
needed for the task-level handoff, PA invokes no geometry producer. After a
robot frame is selected, RA may expose source pose, target pose, insertion axis,
or tolerance inputs in one batch. A robot-frame-pose need can then use an
accepted camera-frame pose plus a valid frame- and time-matched calibration.
Another task can expose a different set and order without changing orchestration
code.

One source per existing Phase 3.2 operation remains an internal audit boundary.
A single `MissingContextBatch` may therefore cause PA to run several numbered,
single-source operations before it returns one new versioned
`CompositionContextBundle`. There is no fixed semantic batch-round count.
Another round is permitted only after new accepted context, a reclassified
need, or a structurally different RA draft demonstrates progress. An identical
request against unchanged context, ambiguity, unsupported output, unavailable
producer, or no new accepted binding stops fail-closed.

`context understanding complete` means only that every currently blocking
PA-owned need referenced by the `TaskTransitionDraft` is satisfied for Phase 5.
It does not mean that every later primitive input is already available.

This production mechanism is not implemented. The current Phase 4.3 assessor,
typed-context index, producer-descriptor registry, complete document/scene/pose/
frame producer chain, Phase 5 handoff, and downstream batch-resumption path are
absent. Controlled tests inject assessment and grounding doubles, so the live UI
remains fail-closed rather than making these decisions from raw summaries.

`product_agent_runtime.py` composes one shared ProductAgent behind the narrow
`ProductAgentContextRuntime` interface used by the Phase 2.1 UI connection. It
delegates only `ask_llm_structured(...)` and never calls ProductAgent `setup()`.

`product_context.py` implements the PA-owned part of the standalone Phase 4.0
foundation. Shared immutable TBox loading, profile validation, fingerprinting,
and class-hierarchy queries live in `ontology/ppr_tbox.py`. Product context
creates an independent writable Turtle ABox from the exact unresolved
requirement, validates evidence-backed triple deltas, and persists accepted
assertions with provenance. Its minimal semantic
bridge is `specification defines required feature` and `requested process
realizes the same feature`.

PA owns and mutates only its interaction ABox. The PA ABox does not contain RA,
primitive-offering, or `capableOf` assertions. A future RA resource ABox will
have separate ownership; PA and RA share the immutable TBox rather than one
writable graph. No PA-to-RA ontology projection is implemented in this phase.

The merge caller supplies the trusted `authorized_evidence_refs` allowlist; a
tool or its proposed delta cannot authorize its own evidence. Typed-context refs
remain opaque JSON records under `products/grounding/` until their separately
authorized producer contracts are implemented.

Phase 4.0 is called by `context_interaction.py` before the first PA request and
reloaded by `context_assessment.py` before interpretation. The production UI
injects no complete grounding runtime, so it stops with
`grounding_unavailable`. No connected PA runtime step starts ProductAgent SPADE
behaviours, calls a VLM, retrieves the primitive catalog, builds an assembly
plan, connects RA, performs real grounding, or executes robot behavior. The
separate Phase 4.1 diagnostic does not call ProductAgent or assess completion.
