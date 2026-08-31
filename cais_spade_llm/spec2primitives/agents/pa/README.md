# ProductAgent grounding

This package owns the narrow, non-executing Spec2Primitives ProductAgent
connection. It does not start the shared ProductAgent lifecycle or modify the
shared agent.

## Native investigation

`start_pa_context_interaction` records the exact requirement, initializes the
interaction ABox, and asks the production grounding runtime to conduct one PA
investigation. The PA call receives:

- the exact requirement;
- an allowed PPR schema projection;
- the current ABox and required-output projection;
- approved discovery metadata; and
- one controlled native `retrieve(evidence_id)` tool.

PA can retrieve zero or more approved documents, CAD files, or fresh live
observations in any order. Tool results return to the same conversation as
compact typed evidence. PA then directly returns a proposal candidate, a
clarification question, or an insufficient-evidence explanation.

The model never supplies paths, provider IDs, hashes, frames, record types, or
arbitrary source names. Tool calls are resolved and audited by the system. A
malformed, unauthorized, stale, altered, or unavailable handle fails closed.

Clarification is reserved for requirement meaning that approved evidence cannot
resolve after relevant approved evidence has been retrieved and considered. The
runtime rejects premature clarification and questions that delegate a supplied
required output or approved evidence-category choice to the user, returns
prompt-only feedback, and keeps the retry inside the bounded investigation. It
does not supply a requirement interpretation in that feedback.

An answered clarification is evidence from its persisted append-only
`interaction_record/clarification_<question_turn>.json` record. The native
completion contract pins that exact record through `source_refs`; it does not
create or accept a separate clarification alias.

## Evidence processing

`production_grounding.py` exposes only source-level retrieval to PA. The system
performs document extraction, STL measurement, live capture, segmentation,
correspondence, calibration, frame conversion, and coarse resource checks.
Those operations are descriptor-driven services, not additional PA choices.

Document retrieval yields the complete ordered `DocumentOverviewRecord` v2.
CAD retrieval yields `CADMeshRecord`. Observation retrieval yields typed RGB-D,
point-cloud, and segmentation records. Compatible CAD and observation evidence
automatically activates correspondence. Accepted correspondence and calibration
yield `RobotFrameLocationRecord` for current coarse reachability.

After a proposal, the runtime builds a provisional graph without merging it.
That graph identifies the active consumer and its declared record requirement.
Provider descriptors expand the requirement into a prerequisite closure, run
any derivable providers, and map missing source-produced records back to the
eligible evidence catalog. PA receives this prompt-only gap and may retrieve
again. The same mechanism can use a future compatible provider without adding
a product, filename, or modality branch.

## Ontology grounding

`ontology_grounding.py` validates a transient `OntologyGroundingProposal` v5
candidate. PA may ground
multiple evidence-supported feature individuals under the supplied PPR TBox.
All features must be specification-defined and exactly one must participate in
the configured process `realizes` join. That graph topology identifies the
single resource-assignment target.

Individuals and relations carry their own evidence citations. Deterministic
validation remains the only ABox commit authority and rejects unsupported
vocabulary, unsafe assertions, recipes, primitives, resources, process
creation, literal facts, orphan features, duplicate indices, and invalid
primary joins.

If one PA proposal violates these invariants, the runtime preserves the
rejected audit record and returns the exact deterministic validation failure as
prompt-only feedback for a bounded correction round. It does not synthesize a
missing `defines`, `realizes`, or other assertion on PA's behalf.

Syntactic and semantic validity alone do not commit the candidate. When the
provisional graph activates coarse resource selection, its
`RobotFrameLocationRecord` requirement must be satisfied by an accepted,
unambiguous chain for the primary feature. Correspondence uses only CAD cited
by that feature. After the gate passes, the runtime persists the accepted v5
proposal, merges its assertion-specific RDF delta, selects a reachable
resource, and commits the assignment.

`missing_information` contains relevant evidence-backed facts that remain
unknown. It does not describe a typed fact merely omitted from RDF. Unknown
placement details remain non-blocking unless an authorized downstream consumer
requires them.

## Completion and compatibility

New interactions write append-only native tool audits, direct PA turns,
`ResourceSelectionRecord` v2, `TypedGroundingContract` v3, and
`PAContextGroundingCompletion` v3. Clarification resumes the same conversation
context using the exact persisted question/reply history. Cancellation invokes
no PA call.

For fresh version-3 completions, `context_summary` is explicitly marked as the
ProductAgent proposal narrative captured before deterministic typed grounding
and resource assignment. Consumers, including the future RA adapter, must use
the final hash-pinned typed records and `ResourceSelectionRecord` as the
authoritative completion state. Existing unmarked version-3 records remain
readable and are not migrated.

Read-only validation remains for old ontology proposal v3/v4, document overview
v1, pose-linked resource selection v1, and completion/session v2 records. New
runs never produce the old action protocol or migrate old interactions.

This boundary performs schema-constrained, evidence-backed instance grounding.
RA context requests, primitive composition, robot planning, and execution are
not implemented here.
