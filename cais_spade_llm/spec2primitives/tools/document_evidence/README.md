# Document evidence tool

Phase 4.1 implements this boundary for every PDF explicitly registered in
`../../references/products/approved_sources.json`:

```text
PA calls retrieve for an approved document
→ one complete ordered-page typed evidence record
→ PA interprets it in the current requirement and ontology context
→ transient ontology candidate enters the evidence gate
```

Run `poetry run python -m
cais_spade_llm.spec2primitives.tools.document_evidence.prepare --all` or use
`--context-ref <exact-ref>` before starting the system. Preparation validates
the PDF, hashes it, extracts ordered page text and metadata, renders page
images, invokes the VLM once, and atomically caches a generic overview by
source hash, full model configuration, and overview-schema version. The cache
lives under `contexts/source_cache/` and is not committed.

The version-2 overview contains every ordered page's extracted text,
rendered-page hash, neutral visual observations, uncertainty, and exact page
refs. Its preparation request contains no question, user requirement, TBox,
ABox, ontology vocabulary, entity keys, relations, or triple delta.

When PA retrieves the current approved NIST PDF, the system supplies pages 1
through 6, including page 4, exactly once and in document order in one
`DocumentOverviewRecord` version 2. New production retrieval does not ask a
focused question or produce `DocumentEvidenceRecord`. Older records remain
readable only where recovery validation explicitly supports them.

The document tool never authors ontology facts. PA uses the requirement,
authoritative ontology, and retrieved typed records together, then returns a
cited v8 `target_feature` with a required process, explicit current and desired
states, optional typed state-value refs, and direct evidence. A separate PA
semantic-review call checks adequacy before commit. That proposal remains transient until every
typed prerequisite activated by its provisional graph is accepted.

The VLM overview may provide semantic evidence that a manual describes a
destination or final condition, but it neither labels RGB-D regions nor emits
robot-frame geometry. `TargetFeatureGeometryRecord` is not part of the
architecture. PA assigns approved typed evidence to feature states, while
verifiers derive only the numeric values they require.

No RAG, embeddings, vector database, independent-page ranking, or section
retrieval is part of this boundary. Large-document retrieval remains future
work.

`OpenAIDocumentVisionRuntime` uses `../../config/model_runtime.json`, sends
`store: false`, and exposes no tools. Startup only validates registered
sources and reports `prepared`, `missing`, `stale`, or invalid cache state;
it makes no ProductAgent, LLM, or VLM request. CAD, RGB-D, and document-free
paths never call the document VLM.
