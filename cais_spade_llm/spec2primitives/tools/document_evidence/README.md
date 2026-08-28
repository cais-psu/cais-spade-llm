# Document evidence tool

Phase 4.1 implements this boundary for every PDF explicitly registered in
`../../references/products/approved_sources.json`:

```text
generic DocumentOverviewRecord cache
→ GroundingSession preview or selected targeted evidence action
→ PA OntologyGroundingProposal from directly supported statement IDs
→ deterministic TBox/evidence/ABox validation
```

Run `poetry run python -m
cais_spade_llm.spec2primitives.tools.document_evidence.prepare --all` or use
`--context-ref <exact-ref>` before starting the system. Preparation validates
the PDF, hashes it, extracts ordered page text and metadata, renders bounded
page images, invokes the VLM once, and atomically caches a generic overview by
source hash, full model configuration, and overview-schema version. The cache
lives under `contexts/source_cache/` and is not committed.

The overview contains only a summary, surface-form observations, uncertainty,
and exact page refs. Its preparation request contains no user requirement,
TBox, ABox, ontology vocabulary, entity keys, relations, or triple delta.
Runtime interpretation returns an assertion-free delta referencing the
overview. If a `GroundingSession` cannot resolve an `InformationNeed` from that
preview, PA may select one eligible document action. The controller ranks
relevant pages deterministically and permits one targeted neutral
`DocumentEvidenceRecord` request for the exact need and source revision.

The document tool never authors ontology facts. Late ontology mapping occurs
in `agents/pa/ontology_grounding.py` after PA understanding is sufficient and
uses directly supported session statement IDs. The diagnostic displays the
overview, targeted evidence, PA statements, provider action, ontology proposal,
and accepted assertions as separate stages.

`OpenAIDocumentVisionRuntime` uses `../../config/model_runtime.json`, sends
`store: false`, and exposes no tools. F5 startup only validates registered
sources and reports `prepared`, `missing`, `stale`, or invalid cache state; it
makes no ProductAgent, LLM, or VLM request. Warm overview-supported needs make
zero document-VLM calls. CAD, RGB-D, and document-free paths never call this
VLM.
