# ProductAgent grounding

This package owns the narrow ProductAgent connection and the complete pre-RA
grounding boundary. It never starts ProductAgent lifecycle behavior; the only
model operation exposed by `ProductAgentContextRuntime` is
`ask_llm_structured(...)`.

## Runtime flow

```text
requirement + authoritative ontology
→ discover approved documents, files, observations, and retrieved records
→ PA chooses one semantic action
→ host retrieves and journals evidence
→ PA fills the ontology and writes one cited context summary
→ deterministic TBox/evidence/ABox validation
→ PAContextGroundingCompletion v2
```

The PA action response is exactly one of:

```json
{"next_action": {"action": "retrieve", "source_ref": "..."}}
{"next_action": {"action": "inspect", "source_ref": "...", "question": "..."}}
{"next_action": {"action": "propose_grounding"}}
{"next_action": {"action": "ask_user", "question": "..."}}
{"next_action": {"action": "incomplete", "reason": "..."}}
```

`next_action` is only the object-rooted structured-output envelope required by
the model API. The nested value remains the one semantic choice made by PA.

The model does not author IDs, provider IDs, source revisions, hashes, attempts,
intermediate understanding, statement classifications, or state transitions.
The host resolves the selected source operation and persists those mechanical
fields in `GroundingSession` schema version 2. Exact action replays and invalid
actions receive one repair opportunity; a second invalid response becomes a
persisted `incomplete` state without executing unsafe output.

## Provider capabilities

`GroundingProducerDescriptor` describes each provider's accepted evidence,
produced records, prerequisites, availability, and cost. Registration order
does not control selection. PA sees available sources and retrieved typed
records together with the exact requirement and TBox.

The public `needed_context` shape remains unchanged for document, CAD,
observation, and clarification serving. Provider output stays in hash-pinned
typed records rather than being copied into a model-authored reasoning state.

## CAD hypothesis convergence

PA selects one approved CAD source at a time. That selection remains the active
hypothesis until an RGB-D segmentation is available and the host evaluates the
exact CAD/segmentation record pair. Other CAD choices are withheld while the
hypothesis is unresolved.

An accepted correspondence advances automatically through its hash-pinned pose
lineage. A rejected or ambiguous pair runs once, leaves the current segmentation
available, and reopens the remaining CAD sources for one new PA choice. Cached
CAD records may be selected without preprocessing the STL again. Retrieval of
another record never replaces an accepted lineage by recency.

Live capture remains unavailable while the selected CAD is unresolved or while
another untried CAD can use the current scene. After useful hypotheses are
exhausted, one controlled recapture may reopen them for the new scene. A pose
failure after accepted correspondence uses that recapture for the same CAD
source rather than silently choosing a different CAD.

Live observation revisions describe the decision frontier rather than capture
IDs or timestamps. A repeated rejected or ambiguous frontier therefore cannot
reopen unchanged capture and ends in controlled `incomplete`; changed CAD
evidence or a materially different candidate decision can reopen it.

## Documents

The first selected PDF operation creates or loads a generic content-addressed
`DocumentOverviewRecord` without a question. If PA later chooses `inspect`,
the document provider sends every cached page exactly once and in order with
PA's focused question. For the approved NIST PDF this is pages 1 through 6.
The result is a neutral `DocumentEvidenceRecord`, not an ontology delta.

No RAG, embeddings, vector database, or independent-page ranking is used.

## Final ontology and context

When PA chooses `propose_grounding`, one grounding call receives the requirement,
TBox, current ABox, retrieved typed records, and authorized evidence refs. It
returns:

- TBox-level individuals, relations, and literal facts;
- one concise `context_summary`;
- proposal-level `evidence_refs`; and
- `missing_information` that is useful but not represented by the ontology.

The host validates citations, class/property authority, domain/range signatures,
the controller-owned specification individual, and the atomic ABox delta. One
invalid proposal is repaired once; a second invalid proposal is persisted as
`ontology_gap` without changing the ABox.

`TypedGroundingContract` schema version 2 preserves the final context summary,
missing information, ontology projection, and hash-pinned typed/source/
clarification records. The ontology carries shared PPR meaning; exact part
numbers, geometry, poses, page observations, and uncertainty remain in typed
records and the final context.

ProductAgent, ResourceAgent, RobotAgent, `SystemBridge`, primitive composition,
and execution remain unchanged.
