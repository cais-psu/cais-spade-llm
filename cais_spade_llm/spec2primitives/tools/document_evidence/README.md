# Document evidence tool

Phase 4.1 implements this boundary for every PDF explicitly registered in
`../../references/products/approved_sources.json`:

```text
generic whole-document overview
→ optional question-targeted whole-document inspection
→ typed evidence record
→ final ontology/context proposal
```

Run `poetry run python -m
cais_spade_llm.spec2primitives.tools.document_evidence.prepare --all` or use
`--context-ref <exact-ref>` before starting the system. Preparation validates
the PDF, hashes it, extracts ordered page text and metadata, renders page
images, invokes the VLM once, and atomically caches a generic overview by
source hash, full model configuration, and overview-schema version. The cache
lives under `contexts/source_cache/` and is not committed.

The overview contains only a summary, surface-form observations, uncertainty,
and exact page refs. Its preparation request contains no question, user
requirement, TBox, ABox, ontology vocabulary, entity keys, relations, or triple
delta.

PA may later choose `inspect` with a focused question. The controller supplies
all cached pages exactly once and in document order. The current approved NIST
PDF therefore supplies pages 1 through 6, including page 4, in one
`DocumentEvidenceRecord`. `selected_pages` records the complete ordered
range and citations outside that range are rejected.

The document tool never authors ontology facts. The final ontology call uses
the requirement, authoritative ontology, and retrieved typed records together,
then returns a TBox-valid proposal plus one cited context summary.

No RAG, embeddings, vector database, independent-page ranking, or section
retrieval is part of this boundary. Large-document retrieval remains future
work.

`OpenAIDocumentVisionRuntime` uses `../../config/model_runtime.json`, sends
`store: false`, and exposes no tools. Startup only validates registered
sources and reports `prepared`, `missing`, `stale`, or invalid cache state;
it makes no ProductAgent, LLM, or VLM request. CAD, RGB-D, and document-free
paths never call the document VLM.
