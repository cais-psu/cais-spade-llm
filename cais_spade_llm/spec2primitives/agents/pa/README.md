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

## How PA determines needed information

PA determines a context need by comparing the information required by the next
consumer with the validated runtime context already accumulated for the exact
requirement:

```text
required information for the current handoff
    - validated accumulated runtime context
    = one unresolved information need
```

PA cannot determine sufficiency from the requirement alone. Its decision uses:

- the exact `product_requirement`;
- the current validated product context and typed context-record refs;
- the information required by the current handoff, initially the future
  robot-independent assembly-planning input;
- attempted evidence, unresolved results, contradictions, and freshness;
- the currently available approved evidence and controlled tool descriptions;
- a later downstream `missing_context` response when a consumer exposes a
  product or scene input that was not previously needed.

PA identifies the missing information before it selects an evidence source. A
missing meaning, relationship, typed context record, or user intention is
persisted as one `unresolved_semantic_need` with `kind`, `symbol`, and
`description`. The source is then selected because it can address that need,
not because the workflow follows a fixed modality order.

| Unresolved information | Possible evidence or authority |
|---|---|
| Meaning of the requested assembly | Approved document interpretation |
| Intended receiving feature | Approved document interpretation |
| Expected component dimensions | Exact approved CAD selected for that component |
| Current loose-component location | Fresh RGB-D observation |
| Which observed candidate matches the requested component | Segmented RGB-D candidates plus the exact selected CAD |
| Robot-frame pick coordinate | Future frame transformation and fresh resource context |
| Ambiguous user intention | Focused user clarification after permitted evidence is exhausted |

This table describes possible resolution paths, not mandatory fields or a
required retrieval sequence. An interaction may use none, one, or several of
these paths. PA requests only one source, validates and persists its interpreted
result, and reassesses before requesting another.

For `assemble Medium Gear`, a valid dynamic sequence could be:

1. Record that the receiving feature or requested assembly meaning is
   unresolved and request the approved document.
2. Reassess the updated product context. If the current component location is
   required for the current handoff, record `current_part_location` as the next
   unresolved information need and request fresh RGB-D.
3. If the observation contains several unresolved candidates, request
   `Gear_Medium.STL` because its geometry can support candidate association.
4. Persist the accepted, ambiguous, or rejected size-association result and
   reassess again.
5. Keep robot-frame location or complete pose unresolved when only a
   camera-frame center exists. Do not claim completion from that partial result.

Another requirement may produce a different order or require neither CAD nor
RGB-D. PA does not retrieve a source merely because it is available.

PA proposes the unresolved information need and evidence request; the
controlled Phase 4.3 boundary validates the request, rejects unsupported or
unjustified duplicate retrieval, and permits clarification or completion only
from persisted assessment over the updated context. Future consumers may return
a new `missing_context` need, so a previously sufficient planning handoff does
not imply that every later primitive input is already available.

The production Phase 4.3 assessor and the Phase 4.2B2A association connection
are not configured yet. Current controlled tests inject assessment and
grounding doubles; the live UI therefore remains fail-closed instead of making
these decisions from raw retrieved summaries.

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
