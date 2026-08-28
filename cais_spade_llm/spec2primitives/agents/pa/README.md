# ProductAgent grounding

This package owns the narrow ProductAgent connection and the complete pre-RA
grounding boundary. It never starts ProductAgent lifecycle behavior; the only
model operation exposed by `ProductAgentContextRuntime` is
`ask_llm_structured(...)`.

## Runtime flow

```text
exact user requirement
→ initialize GroundingSession
→ collect authorized, inference-free provider previews
→ PA writes cited statements and InformationNeeds
→ PA selects one exact eligible provider action, waits for user intent,
  becomes incomplete, or becomes ready for ontology
→ accepted new evidence updates the session
→ directly supported statements enter one late ontology-mapping call
→ deterministic TBox/evidence/ABox validation and atomic commit
→ PAContextGroundingCompletion v2
```

The first PA grounding prompt contains the exact requirement, provider-owned
previews, source identifiers, and eligible action identifiers. It contains no
TBox, ABox, ontology vocabulary, expected destination, evaluator label,
simulator truth, or process-specific controller hint. The same contracts apply
to assembly, drilling, welding, inspection, and other requirements.

`GroundingSession` is PA's understanding state. It stores:

- `GroundingStatement` values with plain text, `directly_stated` or `inferred`
  status, source refs, and a reason;
- `InformationNeed` questions, accepted record types, supporting source refs,
  and `open`, `resolved`, or `exhausted` state;
- `GroundingActionAttempt` values keyed by
  `need_id + provider_id + source_revision`;
- one `GroundingDecision`; and
- the current `grounding`, `waiting_for_evidence`, `waiting_for_user`,
  `ready_for_ontology`, `complete`, `incomplete`, or `ontology_gap` state.

An inferred statement may guide the next evidence request, but it cannot
resolve a required information need or reach the ABox. A new required need
introduced after the first PA call must cite newly accepted evidence. An
unchanged provider/source revision cannot count as progress.

## Provider capabilities

`GroundingProducerDescriptor` is a provider-owned capability description, not
a centrally assigned priority. It declares the provider's exact ID,
description, accepted evidence types, produced record types, per-output
prerequisites, availability, and estimated cost.

For each open information need, the controller discovers eligible actions from:

```text
required record type
+ satisfied prerequisites
+ authorized source revision
- already attempted provider/source revisions
```

PA can select only an exact action in that discovered set. The controller has
no fixed document → CAD → RGB-D sequence, and registration order does not
control selection. A new provider that supplies the same descriptor contract
can participate without editing orchestration code.

Provider previews are deliberately limited:

- document previews expose validated cached observations;
- CAD previews expose approved corpus metadata and content revisions;
- live-observation previews expose availability, not detections or simulator
  truth;
- accepted typed records expose their validated, hash-pinned contents; and
- clarification records expose only the user's persisted reply.

The existing public `needed_context` request shape remains unchanged for
document, CAD, observation, and clarification serving. If no eligible untried
action remains, or the emergency operational ceiling is reached, the
controller persists `incomplete`. It does not throw a grounding-loop error,
retry unchanged evidence, or manufacture an assertion.

## Documents and visual evidence

Every PDF registered in `references/products/approved_sources.json` can have a
content-addressed `DocumentOverviewRecord`. Startup only validates sources and
loads cache indexes. It performs no LLM or VLM request.

A prepared overview is an inference-free runtime preview. If PA needs visual
detail that it does not contain, PA may select the exact document provider
action. The controller deterministically selects relevant pages and makes at
most one targeted VLM request for that need, document hash, and source
revision. The result is a neutral `DocumentEvidenceRecord`, not an ontology
delta.

Document-free CAD/RGB-D paths never invoke the document VLM.

## Late ontology mapping

`ontology_grounding.py` receives only directly supported session statement IDs
plus the fixed vocabulary, property signatures, current individuals, and the
initialized specification IRI. This is the only PA call that sees ontology
details.

The mapper may type controller-owned new individuals, refer to validated
existing individuals, and propose relations that satisfy the TBox property
signature. It may not create or rename the specification individual, map
inferred or missing statements, add primitive/resource/recipe facts, or force
`realizes` without a supported process and feature.

The mapper reports directly supported statements the current TBox cannot
represent. Those statements remain in the typed grounding contract and do not
invalidate otherwise valid evidence. The existing deterministic triple-delta
validator is the only authority that can commit an ABox change.

The ontology is supporting infrastructure for shared PPR meaning. It is not the
evidence router, task-understanding engine, primitive composer, or document-
understanding contribution.

## Completion boundary

New production interactions write `PAContextGroundingCompletion` version 2.
The completion pins the session, ontology proposal, ontology projection, typed
grounding contract, typed evidence records, approved sources, and clarification
records by hash. Earlier completion formats are not loaded or migrated.

The official future PA→RA input is:

```text
validated ontology projection
+ typed grounding contract
+ hash-pinned typed evidence records
```

The ontology carries shared PPR meaning. Goals, constraints, missing
information, geometry, poses, and statements outside the current TBox remain
in the typed contract and typed records.

`ProductContextView` remains the final validated ontology/typed-record view; it
is not PA's reasoning state. PA owns only its interaction ABox. A future RA
resource ABox remains separately owned, and ProductAgent, ResourceAgent,
RobotAgent, `SystemBridge`, primitive composition, and execution are unchanged.
